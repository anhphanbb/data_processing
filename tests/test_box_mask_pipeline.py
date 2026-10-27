"""Synthetic checks; no trained model or AWE files required. Run beside revised scripts."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

ff = load('box_ff', 'filter_flatfield_to_1lf.py')
rg = load('box_regrid', 'regrid_l1f_to_l1r.py')
cl = load('box_cloud', '4_and_5_mlcloud.py')
co = load('box_coadd', 'coadd_to_l1a.py')
gen = load('box_products', 'generate_l2a_l3a_l3c.py')

def values(ds, name):
    return np.asarray(np.ma.filled(np.ma.asarray(ds[name][:], dtype=float), np.nan))


def numerical_checks():
    # Exact tiling including rows/columns at historical split boundaries.
    total = np.zeros((256, 256), int)
    for y in range(6):
        for x in range(6):
            boxes = np.zeros((6,6), bool); boxes[y,x] = True
            total += ff.box_mask(boxes)
    assert (total == 1).all()
    boxes = np.zeros((10,6,6), bool); boxes[0,0,1] = True
    epoch = np.arange(10)*ff.EPOCH_CADENCE_SECONDS; epoch[5:] += 10
    assert ff.longest_clean_interval(boxes,epoch,2) == (5,10)
    try: ff.longest_clean_interval(np.ones_like(boxes),epoch,1)
    except ValueError: pass
    else: raise AssertionError('Must reject an orbit without a clean reference')
    # Contaminated values cannot affect histogram, neighboring filters, or normalization.
    image = np.full((256,256),10000,np.uint16); valid = np.ones_like(image,bool)
    valid[85:171,85:171] = False
    a = image.copy(); b = image.copy(); a[~valid] = 0; b[~valid] = 65535
    fa, fb = ff.filter_image_masked(a,valid), ff.filter_image_masked(b,valid)
    np.testing.assert_array_equal(fa,fb)
    ca, method = ff.flatfield_image_masked(fa,np.full(image.shape,10000.),valid)
    cb, method2 = ff.flatfield_image_masked(fb,np.full(image.shape,10000.),valid)
    assert method == method2 == 2
    np.testing.assert_array_equal(ca,cb)
    assert np.isnan(ca[~valid]).all() and (ca[valid] == 10000).all()
    allmissing, method = ff.flatfield_image_masked(image,np.ones(image.shape),np.zeros_like(valid))
    assert method == 0 and np.isnan(allmissing).all()
    # A masked interpolation corner must never be converted into dark radiance.
    counts = np.full((4,4),5000.); counts[1,1] = np.nan
    indices = np.array([[[0,2]],[[0,2]]],np.int16)
    weights = np.full((2,1,2),.5)
    out, valid_out = rg.apply_map(counts,indices,weights)
    assert not valid_out[0,0] and np.isnan(out[0,0])
    assert valid_out[0,1] and out[0,1] == 5
    # Missing observations cannot lower an average; clean overlap can restore coverage.
    counts = np.array([[[5000,0]],[[0,7000]]],np.uint16)
    np.testing.assert_array_equal(co.average_counts(counts,0,np.zeros((2,2),int),1000),[[5000,7000]])
    cl.RUNNING_AVERAGE_WINDOW = 5
    p, supported = cl.process_mlcloud_masked(np.full((25,3,15),np.nan,np.float32))
    assert (p == 1).all() and not supported.any()
    print('PASS numerical mask, clean-reference, normalization, interpolation and missing-cloud checks')


def pipeline_checks(root):
    # Keep intentionally bright/full-panel frames to exercise mask exclusions.
    ff.BAD_FRAME_SCREENING = False
    inp = root/'l0c'; ml = root/'l0d'; out = root/'l1f'
    inp.mkdir(); ml.mkdir()
    n = 36
    e = 1356566418.0 + np.arange(n)*ff.EPOCH_CADENCE_SECONDS
    e[3] = e[2]; e[22:] += 5*ff.EPOCH_CADENCE_SECONDS
    boxes = np.zeros((n,6,6),bool)
    boxes[:5,1,1] = True
    boxes[-6:,2:4,2:4] = True
    boxes[-1] = True  # retain timestamp even if a whole frame has no support
    x,y = np.meshgrid(np.arange(256),np.arange(256),indexing='ij')
    lat, lon = (y-127)*.02, (x-127)*.02
    for filt, brightness in [('q20',5.),('p12',10.),('p14',6.),('bkg',2.)]:
        name = f'awe_l0c_{filt}_2023001T0000_00001_v02.nc'
        with Dataset(inp/name,'w') as ds:
            for dim, size in [('time',n),('y_cross_track',256),('x_along_track',256)]: ds.createDimension(dim,size)
            for key in ('Epoch','ISS_Latitude','ISS_Longitude','ISS_Altitude','ISS_Pitch','ISS_Roll','ISS_Yaw','Center_Latitude','Center_Longitude'):
                v = ds.createVariable(key,'f8' if key == 'Epoch' else 'f4',('time',))
                v[:] = e if key == 'Epoch' else np.arange(n)*.06 if key in ('Center_Longitude','ISS_Longitude') else 400 if key == 'ISS_Altitude' else 0
            for key in ('Radiance','Latitude','Longitude'):
                ds.createVariable(key,'f4',('time','y_cross_track','x_along_track'),zlib=True)
            for i in range(n):
                a = np.full((256,256),brightness,np.float32); a[ff.box_mask(boxes[i])] = 60
                ds['Radiance'][i] = a
                ds['Latitude'][i] = rg.disk_plane(lat)
                ds['Longitude'][i] = rg.disk_plane(lon+i*.06)
        if filt == 'q20':
            import shutil
            qpath = ml/name; shutil.copyfile(inp/name,qpath)
            with Dataset(qpath,'a') as ds:
                ds.createDimension('y_box_across_track',6); ds.createDimension('x_box_along_track',6)
                ds.createVariable('MLSPB','u1',('time','y_box_across_track','x_box_along_track'))[:] = boxes
    outputs, failures = ff.process_orbit_batch([qpath],inp,out,ml,'cxx',False,2,1,
                                               mlspb_mode='boxes',min_clean_frames=5)
    assert not failures and len(outputs) == 4
    with Dataset(outputs[0]) as ds:
        selected = np.array([np.flatnonzero(e == epoch)[0] for epoch in values(ds, 'Epoch')])
    lo,last=ff.select_frame_range(boxes[selected].any(axis=(1,2)), 'cxx')
    hi=last+1
    for path in outputs:
        with Dataset(path) as ds:
            np.testing.assert_array_equal(values(ds,'Epoch'),e[selected])
            rad = values(ds,'Radiance')
            for j,i in enumerate(selected):
                assert np.isfinite(rad[j][ff.box_mask(boxes[i])]).all()
                np.testing.assert_array_equal(values(ds,'MLSPB_Pixel_Mask')[j], ff.box_mask(boxes[i]))
            assert np.isfinite(rad[10]).all()
            assert ds.Flat_Field_Reference_Start == lo and ds.Flat_Field_Reference_Stop_Exclusive == hi
            assert ds.Flat_Field_Empty_Frames == 1
            assert (values(ds,'Flat_Field_Normalization')[lo:hi] == 3).all()
            assert np.isfinite(rad[-1]).all()
    print('PASS four-filter L1F: duplicate removal, gap retention, full timeline, exact box masks and fixed reference')
    # Legacy trim still takes the original numerical implementation (serial branch).
    legacy, failed = ff.process_orbit_batch([qpath],inp,root/'legacy_l1f',ml,'cxx',False,1,1,
                                           mlspb_mode='trim',min_clean_frames=100)
    assert not failed and len(legacy) == 4
    with Dataset(legacy[0]) as ds:
        assert len(ds.dimensions['time']) < len(selected)
        assert 'MLSPB_Mask_Mode' not in ds.ncattrs()
    for baseline_path in legacy:
        filt=baseline_path.name.split('_')[2]
        box_path=next(p for p in outputs if p.name.split('_')[2]==filt)
        with Dataset(baseline_path) as a, Dataset(box_path) as b:
            original=values(a,'Radiance')
            expected=original.copy()
            actual=values(b,'Radiance')[lo:hi]
            np.testing.assert_array_equal(actual,expected)
            # NaNs aside, comparison is exact, with no numeric tolerance.
            np.testing.assert_array_equal(values(a,'Epoch'),values(b,'Epoch')[lo:hi])
    print('PASS EXACT legacy trim overlap for all four filters, including flagged pixels')
    rout = root/'l1r'
    rg.process_orbit(rg.discover(out,rout,False)[0],dict(overwrite=False,threads=1,block=4,center_search='cxx'))
    # Regrid the legacy NaN representation independently and compare every pixel.
    import shutil
    old_f = root/'nan_l1f'; old_f.mkdir()
    for path in outputs:
        target_f = old_f/path.name
        shutil.copyfile(path, target_f)
        with Dataset(target_f, 'r+') as ds:
            radiance = values(ds, 'Radiance')
            radiance[values(ds, 'MLSPB_Pixel_Mask') != 0] = np.nan
            ds['Radiance'][:] = radiance
            ds['MLSPB_Pixel_Mask'][:] = 0
    old_r = root/'nan_l1r'
    rg.process_orbit(rg.discover(old_f,old_r,False)[0],dict(overwrite=False,threads=1,block=4,center_search='cxx'))
    for path in rout.glob('*.nc'):
        with Dataset(path) as new, Dataset(old_r/path.name) as old:
            effective = values(new, 'Radiance')
            effective[values(new, 'MLSPB_Pixel_Mask') != 0] = np.nan
            np.testing.assert_array_equal(effective, values(old, 'Radiance'))
    print('PASS exact regridding equivalence after applying the propagated mask')
    q = next(rout.glob('*q20*.nc'))
    with Dataset(q) as ds:
        r = values(ds,'Radiance'); la = values(ds,'Latitude')
        assert np.isfinite(r[-1]).any() and np.isfinite(la[-1]).any()
        assert (values(ds,'MLSPB_Pixel_Mask')[-1][np.isfinite(la[-1])] == 1).all()
        assert (np.isfinite(r[-2]) & (values(ds,'MLSPB_Pixel_Mask')[-2] == 1)).any()
    cl.SPACE = 1; cl.BATCH_SIZE = 64; cl.PREFETCH_BATCHES = False
    normal, support = cl.read_normalized(q,return_support=True)
    class Model:
        def __init__(self): self.count = 0
        def __call__(self, images, training=False):
            self.count += len(images)
            return np.full((len(images),1),.1,np.float32)
    import shutil
    old_q = root/'old_q.nc'
    shutil.copyfile(q, old_q)
    with Dataset(old_q, 'r+') as ds:
        radiance = values(ds, 'Radiance')
        radiance[values(ds, 'MLSPB_Pixel_Mask') != 0] = np.nan
        ds['Radiance'][:] = radiance
        ds.renameVariable('MLSPB_Pixel_Mask', 'Legacy_Mask_For_Test')
    old_normal, old_support = cl.read_normalized(old_q, return_support=True)
    np.testing.assert_array_equal(normal, old_normal)
    np.testing.assert_array_equal(support, old_support)
    model = Model()
    raw = cl.predict_mlcloud(normal,model,lambda a:a,support)
    assert 0 < model.count < (len(normal)-2)*15
    processed, supported = cl.process_mlcloud_masked(raw)
    target = root/'l1c'/q.name.replace('_l1r_','_l1c_')
    cl.write_output(q,target,raw,processed,supported)
    for f in ('q20','p12','p14','bkg'):
        source = target if f == 'q20' else next(rout.glob(f'*_{f}_*.nc'))
        result = co.process_file((str(source),str(root/'l1a'),f),dict(window=19,cpp_compat=False,epoch_mode='awe-gps',block_size=3,overwrite=False))
        assert result['status'] == 'done', result
        old_source = root/('old_' + source.name)
        shutil.copyfile(source, old_source)
        with Dataset(old_source, 'r+') as ds:
            radiance = values(ds, 'Radiance')
            radiance[values(ds, 'MLSPB_Pixel_Mask') != 0] = np.nan
            ds['Radiance'][:] = radiance
            ds.renameVariable('MLSPB_Pixel_Mask', 'Legacy_Mask_For_Test')
        old_result = co.process_file((str(old_source),str(root/'old_l1a'),f),dict(window=19,cpp_compat=False,epoch_mode='awe-gps',block_size=3,overwrite=False))
        assert old_result['status'] == 'done', old_result
        with Dataset(old_result['output']) as old, Dataset(result['output']) as new:
            np.testing.assert_array_equal(values(old,'Radiance'), values(new,'Radiance'))
            assert 'MLSPB_Pixel_Mask' not in new.variables
        with Dataset(result['output']) as ds:
            assert ds.MLSPB_Mask_Mode == 'boxes_trim_reference'
            if f == 'q20': assert 'MLCloud_Supported' not in ds.variables
    print('PASS regridding, cloud crop exclusion with mock model, coadding and geometry preservation')
    gen.main(['--l1a-input',str(root/'l1a'),'--l2a-output',str(root/'l2a'),
              '--l3a-output',str(root/'l3a'),'--l3c-output',str(root/'l3c'),
              '--master-input',str(rout),'--workers','1','--resolutions','2','10'])
    for path in (root/'l3c').rglob('*.nc'):
        with Dataset(path) as ds:
            rad = values(ds,'Radiance'); pe = values(ds,'Pixel_Epoch')
            assert np.isfinite(rad).any()
            assert np.isfinite(pe[np.isfinite(rad) & (rad > 0)]).all()
            assert np.isnan(pe[~np.isfinite(rad) | (rad == 0)]).all()
    assert len(list((root/'l3a').rglob('*.nc'))) == 2
    print('PASS L2A/L3A 2 & 10 km and L3C with Pixel_Epoch; synthetic full processing path complete')


if __name__ == '__main__':
    numerical_checks()
    with tempfile.TemporaryDirectory(prefix='awe_box_test_', dir=ROOT) as tmp:
        pipeline_checks(Path(tmp))
    print('ALL BOX-MASK TESTS PASSED (cloud inference mocked; no real model validation)')
