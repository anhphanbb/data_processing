#!/usr/bin/env python3
"""Standalone AWE L1F -> L1R regridding (four-filter C++ workflow).

v3 compatibility fix: create x, y, time in that order (IDs 0,1,2), with
unlimited time, to support the C++ writer that reuses input dimension IDs.
Image variables still have axes (time,y_cross_track,x_along_track).

Install: pip install numpy netCDF4 numba scipy
Edit DEFAULT_INPUT / DEFAULT_OUTPUT, then run in Spyder or a terminal.
    python l1f_to_l1r_optimized.py
    python l1f_to_l1r_optimized.py INPUT OUTPUT --workers 4 --numba-threads 1
    python l1f_to_l1r_optimized.py --self-test

Uses p12 to construct one orbit master grid, reused for p12/p14/bkg/q20.
No external script, calibration file, or pre-existing master grid is needed.
Writes MasterNNNNN.mgf (C++ little-endian int32 sizes, then float64 lat/lon)
and L1R files, preserving input filename timestamps/version/suffix and subfolders.
Requires a matching p12 for every orbit. Filters may have different timestamps
and frame counts: each uses its own Center_Latitude/Center_Longitude vectors.

Scientific compatibility:
* 6371+87 km sphere, 2 km grid, 300x300 output; floating midpoint lat/lon
  as in OnDataprocessingRegridding4files (the single-file handler differs).
* C++ rotations, grid padding, XEq override, longitude wrapping, 5x7 center
  tracking, ordered neighbor walk, sticky boundary mask and interpolation.
* Read float32 radiance, multiply by float32 1000, truncate to uint16;
  interpolate integer counts, truncate again, write float32 kR.
* NetCDF image plane -> C++ matrix: plane[::-1, :].T; reverse on output.
  Output arrays use (time, y_cross_track, x_along_track).
* Undefined C++ cases (overflow, degenerate cells, invalid centers, grid bounds)
  are handled explicitly: input counts clipped; invalid cells masked; invalid
  centers/out-of-grid windows raise errors instead of accessing bad memory.
* Default center search reproduces C++'s restricted forward window; --center-search
  global is available for irregular cadence/gaps and intentionally changes it.

Speed changes (v2):
* Processes schedule individual filter files, sharing read-only master pages.
* SciPy spherical cKDTree finds global centers; the C++ 5x7 subsequent search
  remains the default. Exact center vectors reuse their position sequence.
* Read blocks default to 32 frames; float32 reads avoid the former intermediate
  float64 cube. Set --block-size 0 for automatic temporal chunk alignment.
* Geometry and intensity application are separate Numba kernels. --reuse-maps
  optionally shares exact-geometry maps via a temporary disk cache (about 1.8 MB
  per unique frame; deleted once all files for that orbit finish). Default OFF: hashing/cache I/O may cost more than it saves
  when filter geolocation differs. Cache hits and geometry time are reported.
* CSV timing reports separate orbit setup/index/center search and per-file read,
  geometry/cache, application, conversion, output, and close times. JIT startup
  is excluded from per-file timings but included in run wall time.
* File worker and Numba thread settings remain independent. No fastmath or
  lower-precision interpolation weights are used. Existing outputs are skipped unless --overwrite.
Each output is written to a .partial file and renamed only after closing.
Ancillary non-image variables are copied; unknown spatial variables are rejected
rather than silently resized. Metadata is inherited with level/grid updates;
exact metadata equality and real-data numerical equality still need comparison
against an actual C++ L1R product. An extracted C++ kernel comparison on a
synthetic circular FOV matched every jointly valid radiance pixel exactly; 83
invalid boundary cells were masked by Python where C++ wrote zero.
--self-test exercises the numerical kernel,
orientation, dateline handling and synthetic NetCDF batch processing.
"""
from __future__ import annotations
import argparse
import os
import re
import struct
import hashlib
import tempfile
import csv
import shutil
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import numpy as np
try:
    from numba import njit, prange, set_num_threads
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f
    prange = range
    set_num_threads = None

DEFAULT_INPUT = r'E:\soc\l1f\2023\12'
DEFAULT_OUTPUT = r'E:\soc\l1r\2023\12'
DEFAULT_WORKERS = 20
DEFAULT_NUMBA_THREADS = 1
DEFAULT_BLOCK_SIZE = 32  # Frames per block; 0: automatic temporal chunk alignment
DEFAULT_REUSE_MAPS = False  # Optional disk cache; useful only for matching geometry
DEFAULT_RECURSIVE = False
DEFAULT_OVERWRITE = True
FILTERS = ('p12', 'p14', 'bkg', 'q20')
RADIUS = 6458.0
RESOLUTION = 2.0
SIZE = 300
PATTERN = re.compile(r'^(?P<prefix>.*?)l1f_(?P<filter>p12|p14|bkg|q20)_(?P<stamp>\d{7}T\d{4})_(?P<orbit>\d{5})_(?P<version>v\d+)(?P<suffix>.*?)\.nc$', re.I)


def plain(value):
    return np.asarray(np.ma.filled(np.ma.asarray(value, dtype=np.float64), np.nan))


def rotate(lat, lon, angle, axis):
    a, b, h = np.deg2rad(lat), np.deg2rad(lon), np.deg2rad(angle)
    if axis == 'x':
        top = np.sin(b)*np.cos(a)*np.cos(h)-np.sin(a)*np.sin(h)
        bot = np.cos(a)*np.cos(b)
        z = np.cos(a)*np.sin(b)*np.sin(h)+np.sin(a)*np.cos(h)
    else:
        top = np.cos(a)*np.sin(b)
        bot = np.cos(a)*np.cos(b)*np.cos(h)-np.sin(a)*np.sin(h)
        z = np.cos(a)*np.cos(b)*np.sin(h)+np.sin(a)*np.cos(h)
    return np.rad2deg(np.arcsin(np.clip(z, -1, 1))), np.rad2deg(np.arctan2(top, bot))


def make_master(lat, lon):
    if len(lat) < 3 or not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError('Master-grid p12 needs at least three finite center positions')
    mid = (len(lat)-1)//2
    latmid, lonmid = lat[mid], lon[mid]
    dl = lon[mid+1]-lon[mid-1]  # deliberately raw C++ longitude difference
    da = lat[mid+1]-lat[mid-1]
    heading = (90.0 if da > 0 else -90.0) if dl == 0 else np.rad2deg(np.arctan(da/(np.cos(np.deg2rad(latmid))*dl)))
    a, b = rotate(lat, lon-lonmid, -latmid, 'y')
    a, b = rotate(a, b, -heading, 'x')
    def rounded(distance):
        return (int(distance/100)+1)*100
    xmin = rounded(abs(np.deg2rad(b.min())*RADIUS)+300)
    xmax = rounded(np.deg2rad(b.max())*RADIUS+400)
    ymin = rounded(abs(np.deg2rad(a.min())*RADIUS)+300)
    ymax = rounded(np.deg2rad(a.max())*RADIUS+300)
    nx, ny = int((xmin+xmax)/2)+1, int((ymin+ymax)/2)+1
    if nx < SIZE or ny < SIZE or nx*ny > 50_000_000:
        raise ValueError(f'Unexpected master grid dimensions {nx}x{ny}')
    xeq, yeq = nx//2, int(ymax/2)
    a = np.broadcast_to(np.rad2deg(2*(yeq-np.arange(ny))/RADIUS), (nx, ny)).copy()
    b = np.rad2deg(2*(np.arange(nx)[:, None]-xeq)/(RADIUS*np.cos(np.deg2rad(a))))
    a, b = rotate(a, b, heading, 'x')
    a, b = rotate(a, b, latmid, 'y')
    b += lonmid
    invert = bool(np.any((b < -180) | (b > 180)))
    b[b < -180] += 360
    b[b > 180] -= 360
    return np.ascontiguousarray(a), np.ascontiguousarray(b), invert


@njit(cache=True)
def hav(lat, lon, a, b):
    p = np.pi/180.0
    v = np.sin((a-lat)/2*p)**2+np.cos(a*p)*np.cos(lat*p)*np.sin((b-lon)/2*p)**2
    return 2*RADIUS*np.arcsin(np.sqrt(min(1.0, max(0.0, v))))


@njit(cache=True)
def nearest_center(lat, lon, a, b, previous_x, previous_y, global_search):
    nx, ny = a.shape
    if global_search:
        x0, x1, y0, y1 = 0, nx, 0, ny
    else:
        x0, x1 = previous_x+1, min(nx, previous_x+6)
        y0, y1 = max(0, previous_y-3), min(ny, previous_y+4)
    best, bx, by = np.inf, -1, -1
    for x in range(x0, x1):
        for y in range(y0, y1):
            d = hav(lat, lon, a[x, y], b[x, y])
            if d < best:
                best, bx, by = d, x, y
    return bx, by


@njit(cache=True, parallel=True)
def build_map(lat, lon, target_lat, target_lon):
    nx, ny = lat.shape
    sx, sy = target_lat.shape
    indices = np.full((2, sx, sy), -1, dtype=np.int16)
    weights = np.zeros((2, sx, sy), dtype=np.float64)
    for j in prange(sx):
        startx, starty = nx//2-1, ny//2-1
        for k in range(sy):
            a, b = target_lat[j, k], target_lon[j, k]
            cx, cy = startx, starty
            distmin = (lat[cx, cy]-a)**2+(lon[cx, cy]-b)**2
            bad = not np.isfinite(distmin)
            # C++ order +X, -X, +Y, -Y, including its sticky PxNaN flag
            while True:
                ux, uy = cx, cy
                for direction in range(4):
                    vx, vy = ux, uy
                    if direction == 0: vx += 1
                    elif direction == 1: vx -= 1
                    elif direction == 2: vy += 1
                    else: vy -= 1
                    if 0 <= vx < nx and 0 <= vy < ny and np.isfinite(lat[vx, vy]) and np.isfinite(lon[vx, vy]):
                        d = (lat[vx, vy]-a)**2+(lon[vx, vy]-b)**2
                        if d < distmin:
                            distmin, cx, cy = d, vx, vy
                    else:
                        bad, cx, cy = True, ux, uy
                if cx == ux and cy == uy:
                    break
            if bad:
                continue
            x, y = ux, uy
            if hav(a,b,lat[x,y-1],lon[x,y-1]) < hav(a,b,lat[x,y+1],lon[x,y+1]): y -= 1
            if hav(a,b,lat[ux-1,uy],lon[ux-1,uy]) < hav(a,b,lat[ux+1,uy],lon[ux+1,uy]): x -= 1
            # Fractions are C++ point-to-line distances, not a generic resampler.
            dlx, dax = lon[x+1,y]-lon[x,y], lat[x+1,y]-lat[x,y]
            dly, day = lon[x,y+1]-lon[x,y], lat[x,y+1]-lat[x,y]
            denom = np.sqrt(dlx*dlx+dax*dax)*np.sqrt(dly*dly+day*day)
            if not np.isfinite(denom) or denom == 0:
                continue
            fx = abs(dly*(lat[x,y]-a)-(lon[x,y]-b)*day)/denom
            fy = abs(dlx*(lat[x,y]-a)-(lon[x,y]-b)*dax)/denom
            if not np.isfinite(fx) or not np.isfinite(fy):
                continue
            indices[0,j,k], indices[1,j,k] = x, y
            weights[0,j,k], weights[1,j,k] = fx, fy
            startx, starty = ux, uy
    return indices, weights


@njit(cache=True, parallel=True)
def apply_map(counts, indices, weights):
    sx, sy = indices.shape[1:]
    result = np.full((sx, sy), np.nan, dtype=np.float32)
    valid = np.zeros((sx, sy), dtype=np.bool_)
    for j in prange(sx):
        for k in range(sy):
            x,y = int(indices[0,j,k]), int(indices[1,j,k])
            if x < 0:
                continue
            fx,fy = weights[0,j,k],weights[1,j,k]
            v0 = (float(counts[x+1,y])-float(counts[x,y]))*fx+float(counts[x,y])
            v1 = (float(counts[x+1,y+1])-float(counts[x,y+1]))*fx+float(counts[x,y+1])
            value = (v1-v0)*fy+v0
            if not np.isfinite(value):
                continue
            result[j,k] = np.float32(int(min(65535.0,max(0.0,value))))/np.float32(1000.0)
            valid[j,k] = True
    return result, valid


def interpolate(counts, lat, lon, target_lat, target_lon):
    return apply_map(counts, *build_map(lat, lon, target_lat, target_lon))


def geometry_map(lat, lon, ta, tb, directory):
    """Optional cross-process cache keyed by exact float64 geometry bytes.

    Atomic replacement permits concurrent builders without lock waits. A cache
    miss may be computed twice; partially written maps are never consumed.
    Cache lifetime is one run, and no map is keyed by timestamp/filter alone.
    """
    if directory is None:
        return (*build_map(lat,lon,ta,tb), False)
    digest=hashlib.sha256(b'AWE-map-v2')
    for arr in (lat,lon,ta,tb):
        digest.update(np.asarray(arr.shape,dtype='<i8').tobytes())
        digest.update(memoryview(arr).cast('B'))
    path=Path(directory)/(digest.hexdigest()+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            return data['indices'],data['weights'],True
    indices,weights=build_map(lat,lon,ta,tb)
    temp=path.with_name(path.name+f'.{os.getpid()}.partial')
    try:
        with temp.open('wb') as stream:
            np.savez(stream,indices=indices,weights=weights)
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)
    return indices,weights,False


def xyz(lat,lon):
    a,b=np.deg2rad(lat),np.deg2rad(lon)
    return np.stack((np.cos(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a)),axis=-1)


def indexed_center(lat,lon,master_lat,master_lon,tree):
    if tree is None:
        return nearest_center(lat,lon,master_lat,master_lon,0,0,True)
    point=xyz(lat,lon)
    distance,nearest=tree.query(point)
    # Re-evaluate close ties using the original haversine and scan ordering.
    candidates=sorted(tree.query_ball_point(point,float(distance)+1e-12))
    if not candidates: candidates=[int(nearest)]
    best,bx,by=np.inf,-1,-1
    for flat in candidates:
        x,y=divmod(flat,master_lat.shape[1])
        d=hav(lat,lon,master_lat[x,y],master_lon[x,y])
        if d<best: best,bx,by=d,x,y
    return bx,by


def center_positions(a,b,lat,lon,tree,mode):
    positions=np.empty((len(a),2),dtype=np.int32)
    px,py=0,0
    for i in range(len(a)):
        if i==0 or mode=='global':
            px,py=indexed_center(a[i],b[i],lat,lon,tree)
        else:
            px,py=nearest_center(a[i],b[i],lat,lon,px,py,False)
        if px<150 or py<150 or px+150>lat.shape[0] or py+150>lat.shape[1]:
            raise ValueError(f'Frame {i}: target window outside master grid at ({px},{py})')
        positions[i]=px,py
    return positions


def choose_block(src, requested):
    if requested: return requested
    chunks=[]
    for name in ('Radiance','Latitude','Longitude'):
        value=src[name].chunking()
        if isinstance(value,(list,tuple)): chunks.append(int(value[0]))
    # Align starts to a shared chunk period when practical. Otherwise use the
    # largest time chunk and bounded HDF5 caching to reduce decompression.
    period=int(np.lcm.reduce(chunks)) if chunks else 16
    return max(1,min(128,period if period<=128 else max(chunks)))


def warm_worker(threads):
    if set_num_threads is not None: set_num_threads(threads)
    # Compile/load outside per-file stage timings (included in run wall time).
    lat=np.zeros((4,4)); lon=np.zeros((4,4))
    interpolate(np.zeros((4,4),dtype=np.uint16),lat,lon,np.zeros((1,1)),np.zeros((1,1)))


def read_plane(var, i):
    # Match nc_get_vara_float and C++ Matrix[x][height-1-y].
    p = plain(var[i]).astype(np.float32)
    return np.ascontiguousarray(p[::-1, :].T, dtype=np.float64)


def to_counts(radiance):
    f = np.asarray(radiance, dtype=np.float32)
    f = np.where(np.isfinite(f) & (f > 0), f, np.float32(0))
    return np.ascontiguousarray(np.clip(np.trunc(f*np.float32(1000)), 0, 65535).astype(np.uint16))


def disk_plane(matrix):
    return matrix[:, ::-1].T


def centers(ds):
    a = plain(ds['Center_Latitude'][:]).astype(np.float32).astype(np.float64)
    b = plain(ds['Center_Longitude'][:]).astype(np.float32).astype(np.float64)
    if a.ndim != 1 or b.shape != a.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Invalid center latitude/longitude vectors')
    return a, b


def validate(ds):
    dims = ds['Radiance'].dimensions
    if len(dims) != 3 or dims[0] != 'time' or ds['Radiance'].shape[1:] != (256,256):
        raise ValueError(f'Expected Radiance(time, spatial, spatial) with 256x256 planes; got {dims}, {ds["Radiance"].shape}')
    for name in ('Latitude', 'Longitude'):
        if ds[name].dimensions != dims or ds[name].shape != ds['Radiance'].shape:
            raise ValueError(f'{name} must match Radiance dimensions and shape')
    if 'MLSPB_Pixel_Mask' in ds.variables:
        mask = ds['MLSPB_Pixel_Mask']
        if mask.dimensions != dims or mask.shape != ds['Radiance'].shape:
            raise ValueError('MLSPB_Pixel_Mask must match Radiance dimensions and shape')
    for name, var in ds.variables.items():
        if name not in ('Radiance','Latitude','Longitude','MLSPB_Pixel_Mask') and any(d in dims[1:] for d in var.dimensions):
            raise ValueError(f'Unsupported spatial variable {name}: {var.dimensions}')
    a,b = centers(ds)
    if len(a) != ds['Radiance'].shape[0] or len(a) == 0:
        raise ValueError('Center vector length does not match Radiance frames')
    return dims, a, b


def prepare_output(src, dst, output_name, block):
    spatial = src['Radiance'].dimensions[1:]
    # Files::CreateNetCDF creates x=0, y=1, time=2 but reuses the input
    # Header.dimids array for its output variables. Match those IDs explicitly.
    # Variable axis order remains (time, y_cross_track, x_along_track).
    dst.createDimension('x_along_track', SIZE)
    dst.createDimension('y_cross_track', SIZE)
    dst.createDimension('time', None)
    for name, dim in src.dimensions.items():
        if name not in spatial and name != 'time':
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
    # Match the C++ global attributes reported for L1R products.
    # Attribute names are case-sensitive: Data_level is the canonical name.
    excluded_attrs = {'_NCProperties', 'Data_Level', 'Python_regridding_version', 'history'}
    attrs = {n: src.getncattr(n) for n in src.ncattrs() if n not in excluded_attrs}
    for n,v in list(attrs.items()):
        if isinstance(v,str) and n in ('Logical_source_description','TEXT'):
            attrs[n] = v.replace('L1F','L1R').replace('l1f','l1r')
    attrs.update(Data_level='L1R', Data_type='L1R > Level 1R',
                 Horizontal_Resolution=np.float32(2), Logical_file_id=output_name.split('.nc')[0].split('_v')[0]+'_v'+output_name.split('_v')[1][:2],
                 Logical_source=output_name[:11])
    dst.setncatts(attrs)
    for name,var in src.variables.items():
        image = name in ('Radiance','Latitude','Longitude','MLSPB_Pixel_Mask')
        dims = ('time','y_cross_track','x_along_track') if image else var.dimensions
        kwargs = {'zlib':True, 'complevel':1, 'shuffle':True} if dims else {}
        if image: kwargs['chunksizes'] = (1, SIZE, SIZE)
        if '_FillValue' in var.ncattrs(): kwargs['fill_value'] = var.getncattr('_FillValue')
        out = dst.createVariable(name, 'u1' if name == 'MLSPB_Pixel_Mask' else 'f4' if image else var.datatype, dims, **kwargs)
        att = {n:var.getncattr(n) for n in var.ncattrs() if n != '_FillValue'}
        if image:
            # Arrays are written as unpacked float32 physical values.
            for n in ('scale_factor','add_offset','_Unsigned','actual_range','valid_range'):
                att.pop(n,None)
            for n in ('NC_DIMENSION_V','NC_DIMENSIONS_V','DEPEND_1','DEPEND_2'):
                if n in att: att[n] = {'NC_DIMENSION_V':'time,y_cross_track,x_along_track', 'NC_DIMENSIONS_V':'time,y_cross_track,x_along_track', 'DEPEND_1':'y_cross_track','DEPEND_2':'x_along_track'}[n]
        out.setncatts(att)
        if not image:
            var.set_auto_maskandscale(False)
            out.set_auto_maskandscale(False)
            if 'time' in dims:
                axis = dims.index('time')
                for start in range(0,var.shape[axis],block):
                    sl = [slice(None)]*len(dims)
                    sl[axis] = slice(start,min(start+block,var.shape[axis]))
                    out[tuple(sl)] = var[tuple(sl)]
            else:
                out[...] = var[...]


def process_file(source, target, master_lat, master_lon, positions, invert, options):
    from netCDF4 import Dataset
    timing={n:0.0 for n in ('setup_s','read_s','convert_s','map_s','apply_s','write_s','close_s','total_s')}
    timing.update(file=str(source),status='ok',frames=0,map_hits=0,block=0)
    if target.exists() and not options['overwrite']:
        print(f'SKIP {target.name}',flush=True)
        timing['status']='skipped'
        return timing
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+f'.{os.getpid()}.partial')
    began=time.perf_counter()
    try:
        with Dataset(source) as src:
            _,a,b=validate(src)
            if len(a)!=len(positions): raise ValueError('Center/frame count changed after preparation')
            block=choose_block(src,options['block'])
            timing['block']=block; timing['frames']=len(a)
            for name in ('Radiance','Latitude','Longitude'):
                src[name].set_var_chunk_cache(size=16*1024**2,nelems=1009,preemption=0.75)
            with Dataset(temporary,'w',format='NETCDF4') as dst:
                prepare_output(src,dst,target.name,block)
                timing['setup_s']=time.perf_counter()-began
                print(f'  {source.name}: read block={block}',flush=True)
                for start in range(0,len(a),block):
                    end=min(start+block,len(a))
                    output=np.empty((3,end-start,SIZE,SIZE),dtype=np.float32)
                    tick=time.perf_counter()
                    planes={n:np.asarray(np.ma.filled(np.ma.asarray(src[n][start:end],dtype=np.float32),np.nan)) for n in ('Radiance','Latitude','Longitude')}
                    timing['read_s']+=time.perf_counter()-tick
                    for i in range(start,end):
                        tick=time.perf_counter(); q=i-start
                        lat=np.ascontiguousarray(planes['Latitude'][q,::-1,:].T,dtype=np.float64)
                        lon=np.ascontiguousarray(planes['Longitude'][q,::-1,:].T,dtype=np.float64)
                        disk_radiance = planes['Radiance'][q,::-1,:].T
                        counts=to_counts(disk_radiance)
                        box_mode = getattr(src, 'MLSPB_Mask_Mode', '') in ('boxes_clean_reference', 'boxes_trim_reference')
                        if box_mode:
                            counts = counts.astype(np.float64)
                            counts[~np.isfinite(disk_radiance)] = np.nan
                        if invert: lon[lon<0]+=360
                        px,py=positions[i]
                        ta=np.ascontiguousarray(master_lat[px-150:px+150,py-150:py+150])
                        tb=np.ascontiguousarray(master_lon[px-150:px+150,py-150:py+150])
                        timing['convert_s']+=time.perf_counter()-tick
                        tick=time.perf_counter()
                        indices,weights,hit=geometry_map(lat,lon,ta,tb,options.get('map_directory'))
                        timing['map_s']+=time.perf_counter()-tick
                        timing['map_hits']+=int(hit)
                        tick=time.perf_counter()
                        result,valid=apply_map(counts,indices,weights)
                        if 'MLSPB_Pixel_Mask' in src.variables:
                            # Match the radiance stencil, including zero-weight corners:
                            # legacy NaNs invalidated any stencil touching a flagged pixel.
                            excluded = np.asarray(src['MLSPB_Pixel_Mask'][i], dtype=bool)[::-1,:].T
                            x, y = indices
                            mapped = x >= 0
                            target_mask = np.zeros(x.shape, dtype=np.uint8)
                            mx, my = x[mapped], y[mapped]
                            target_mask[mapped] = (excluded[mx,my] | excluded[mx+1,my]
                                                   | excluded[mx,my+1] | excluded[mx+1,my+1])
                            dst['MLSPB_Pixel_Mask'][i] = disk_plane(target_mask)
                        timing['apply_s']+=time.perf_counter()-tick
                        tick=time.perf_counter()
                        # Missing brightness must not remove grid anchors used downstream.
                        geometry_valid = indices[0] >= 0 if box_mode else valid
                        ta=np.where(geometry_valid,ta,np.nan); tb=np.where(geometry_valid,tb,np.nan)
                        if invert: tb[tb>180]-=360
                        output[0,q],output[1,q],output[2,q]=disk_plane(result),disk_plane(ta),disk_plane(tb)
                        timing['convert_s']+=time.perf_counter()-tick
                    tick=time.perf_counter()
                    for q,n in enumerate(('Radiance','Latitude','Longitude')): dst[n][start:end]=output[q]
                    timing['write_s']+=time.perf_counter()-tick
                    if end==len(a) or start//block%10==0: print(f'  {source.name}: {end}/{len(a)}',flush=True)
                tick=time.perf_counter()
            timing['close_s']+=time.perf_counter()-tick
        os.replace(temporary,target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    timing['total_s']=time.perf_counter()-began
    print(f"DONE {target.name}: read {timing['read_s']:.1f}s, geometry {timing['map_s']:.1f}s, "
          f"apply {timing['apply_s']:.1f}s, write/close {timing['write_s']+timing['close_s']:.1f}s, "
          f"total {timing['total_s']:.1f}s; reused {timing['map_hits']}/{len(a)} maps",flush=True)
    return timing


def prepare_orbit(job,options):
    from netCDF4 import Dataset
    tick=time.perf_counter()
    entries={f:(Path(s),Path(t)) for f,s,t in job}
    reference=entries['p12'][0]
    if not options['overwrite'] and all(t.exists() for _,t in entries.values()):
        print(f'SKIP orbit {PATTERN.match(reference.name)["orbit"]}: all outputs exist',flush=True)
        return [],dict(orbit=PATTERN.match(reference.name)['orbit'],master_s=0.0,index_s=0.0,centers_s=0.0)
    with Dataset(reference) as ds: _,a,b=validate(ds)
    lat,lon,invert=make_master(a,b)
    orbit=PATTERN.match(reference.name)['orbit']
    master_path=entries['p12'][1].parent/f'Master{orbit}.mgf'
    master_path.parent.mkdir(parents=True,exist_ok=True)
    header=struct.pack('<ii',*lat.shape)
    if master_path.exists() and not options['overwrite']:
        with master_path.open('rb') as f:
            same=f.read(8)==header
            if same:
                for array in (lat,lon):
                    old=np.fromfile(f,dtype='<f8',count=array.size)
                    same=same and old.size==array.size and np.array_equal(old,array.ravel())
                same=same and f.read(1)==b''
        if not same: raise ValueError(f'Existing master differs: {master_path}; use separate output folder or --overwrite')
    else:
        tmp=master_path.with_name(master_path.name+f'.{os.getpid()}.partial')
        try:
            with tmp.open('wb') as f:
                f.write(header);lat.astype('<f8').tofile(f);lon.astype('<f8').tofile(f)
            os.replace(tmp,master_path)
        finally: tmp.unlink(missing_ok=True)
    master_seconds=time.perf_counter()-tick
    tick=time.perf_counter();tree=None
    if options.get('spatial_index',True):
        try:
            from scipy.spatial import cKDTree
            tree=cKDTree(xyz(lat.ravel(),lon.ravel()))
        except ImportError:
            print('SciPy unavailable: using original full center scan',flush=True)
    index_seconds=time.perf_counter()-tick
    tick=time.perf_counter();tasks=[];center_cache={}
    for filt in FILTERS:
        if filt not in entries: continue
        source,target=entries[filt]
        if target.exists() and not options['overwrite']:
            print(f'SKIP {target.name}',flush=True);continue
        with Dataset(source) as ds: _,a,b=validate(ds)
        # Exact bytes, no time/geometry approximation between filters.
        key=(a.tobytes(),b.tobytes())
        if key not in center_cache:
            center_cache[key]=center_positions(a,b,lat,lon,tree,options['center_search'])
        tasks.append((str(source),str(target),str(master_path),lat.shape,invert,center_cache[key]))
    center_seconds=time.perf_counter()-tick
    print(f'Orbit {orbit}: master {master_seconds:.2f}s, index {index_seconds:.2f}s, centers {center_seconds:.2f}s; {len(tasks)} file(s)',flush=True)
    return tasks,dict(orbit=orbit,master_s=master_seconds,index_s=index_seconds,centers_s=center_seconds)


def map_folder(root,master_path):
    return Path(root)/hashlib.sha256(str(master_path).encode('utf-8')).hexdigest()


def file_worker(task,options):
    source,target,master_path,shape,invert,positions=task
    options=options.copy()
    if options.get('map_directory'):
        folder=map_folder(options['map_directory'],master_path)
        folder.mkdir(parents=True,exist_ok=True)
        options['map_directory']=str(folder)
    # Read-only mappings share master-grid pages across file processes.
    lat=np.memmap(master_path,mode='r',dtype='<f8',offset=8,shape=shape)
    lon=np.memmap(master_path,mode='r',dtype='<f8',offset=8+int(np.prod(shape))*8,shape=shape)
    try:
        working_lon=np.where(lon<0,lon+360,lon) if invert else lon
        return process_file(Path(source),Path(target),lat,working_lon,positions,invert,options)
    finally:
        # Release Windows handles before overwriting or removing master files.
        del working_lon
        lat._mmap.close();lon._mmap.close()


def process_orbit(job,options):
    """Sequential convenience wrapper also used by --self-test."""
    warm_worker(options['threads'])
    tasks,_=prepare_orbit(job,options)
    return [file_worker(task,options) for task in tasks]


def discover(input_path, output_path, recursive):
    if not input_path.is_dir(): raise ValueError(f'Input folder does not exist: {input_path}')
    groups = {}
    for path in sorted(input_path.rglob('*.nc') if recursive else input_path.glob('*.nc')):
        m = PATTERN.match(path.name)
        if not m: continue
        relative = path.relative_to(input_path)
        key = (str(relative.parent),m['stamp'][:4],m['orbit'],m['version'],m['suffix'])
        group = groups.setdefault(key,{})
        filt = m['filter'].lower()
        if filt in group: raise ValueError(f'Ambiguous {filt} inputs: {group[filt][1]} and {path}')
        target = output_path/relative.parent/re.sub('l1f_', 'l1r_',path.name,count=1,flags=re.I)
        group[filt] = (filt,str(path),str(target))
    jobs = []
    master_targets = set()
    for key,group in groups.items():
        if 'p12' not in group: raise ValueError(f'Missing p12 reference for orbit {key[2]}')
        master_target = str(Path(group['p12'][2]).parent/f'Master{key[2]}.mgf')
        if master_target in master_targets:
            raise ValueError(f'Multiple versions would share {master_target}; separate input/output folders by version')
        master_targets.add(master_target)
        jobs.append(list(group.values()))
    if not jobs: raise ValueError('No matching awe_l1f_{filter}_YYYYDDDT... NetCDF files found')
    return jobs


def self_test():
    # Independent affine-field expectation checks interpolation and orientation.
    x,y = np.meshgrid(np.arange(256),np.arange(256),indexing='ij')
    lat,lon = (y-127)*0.02,(x-127)*0.02
    counts = (2000+10*x+20*y).astype(np.uint16)
    ta = np.array([[0.006,0.026],[0.006,0.026]])
    tb = np.array([[0.008,0.008],[0.028,0.028]])
    result,valid = interpolate(counts,lat,lon,ta,tb)
    expected = (2000+10*(tb/0.02+127)+20*(ta/0.02+127))/1000
    assert valid.all() and np.max(abs(result-expected)) <= 0.00101
    assert np.array_equal(disk_plane(counts)[::-1,:].T,counts)
    ta2,tb2 = np.array([[20.0]]),np.array([[20.0]])
    r,v = interpolate(counts,lat,lon,ta2,tb2)
    assert not v.any() and np.isnan(r).all()
    a=np.linspace(10,10.01,5); b=np.linspace(179.96,180.03,5); b[b>180]-=360
    ml,mo,invert = make_master(a,b)
    assert invert and np.isfinite(ml).all() and np.max(abs(mo))<=180
    # End-to-end tiny orbit, including all four filters and metadata copying.
    from tempfile import TemporaryDirectory
    from netCDF4 import Dataset
    with TemporaryDirectory() as folder:
        root=Path(folder); inp=root/'in'; out=root/'out'; inp.mkdir()
        for filt in FILTERS:
            with Dataset(inp/f'awe_l1f_{filt}_2024001T0000_00621_v23.nc','w') as ds:
                ds.createDimension('time',None); ds.createDimension('y',256); ds.createDimension('x',256)
                for n,values in [('Center_Latitude',np.zeros(3)),('Center_Longitude',np.arange(3)*0.06),('Epoch',np.arange(3))]:
                    ds.createVariable(n,'f4',('time',))[:] = values
                for n in ('Radiance','Latitude','Longitude'):
                    ds.createVariable(n,'f4',('time','y','x'),zlib=True)
                for i in range(3):
                    ds['Radiance'][i]=disk_plane(counts.astype('f4')/1000)
                    ds['Latitude'][i]=disk_plane(lat)
                    ds['Longitude'][i]=disk_plane(lon+i*0.06)
        opts=dict(overwrite=False,threads=1,block=2,center_search='cxx')
        process_orbit(discover(inp,out,False)[0],opts)
        with Dataset(out/'awe_l1r_p12_2024001T0000_00621_v23.nc') as ds:
            assert ds['Radiance'].shape==(3,300,300)
            assert ds['Radiance'].dimensions==('time','y_cross_track','x_along_track')
            assert np.isfinite(plain(ds['Radiance'][:])).sum()>10000
            assert np.array_equal(ds['Epoch'][:],np.arange(3))
            assert ds.Data_level=='L1R'
            assert list(ds.dimensions)[:3]==['x_along_track','y_cross_track','time']
            assert ds.dimensions['time'].isunlimited()
        assert (out/'Master00621.mgf').exists()
    print('Self-test passed.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input',nargs='?',default=DEFAULT_INPUT)
    p.add_argument('output',nargs='?',default=DEFAULT_OUTPUT)
    p.add_argument('--workers',type=int,default=DEFAULT_WORKERS,help='Concurrent FILE processes (all filters eligible)')
    p.add_argument('--numba-threads',type=int,default=DEFAULT_NUMBA_THREADS)
    p.add_argument('--block-size',type=int,default=DEFAULT_BLOCK_SIZE,help='frames per block (default: %(default)s); 0: automatic temporal chunk alignment')
    p.add_argument('--recursive',action='store_true',default=DEFAULT_RECURSIVE)
    p.add_argument('--overwrite',action='store_true',default=DEFAULT_OVERWRITE)
    p.add_argument('--center-search',choices=('cxx','global'),default='cxx')
    p.add_argument('--no-spatial-index',action='store_true',help='Use original full scan instead of SciPy tree')
    p.add_argument('--reuse-maps',action='store_true',default=DEFAULT_REUSE_MAPS,help='Enable temporary disk cache of identical geometry maps')
    p.add_argument('--cache-parent',default=None,help='Local SSD folder for temporary map cache; removed after run')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args(argv)
    if min(args.workers,args.numba_threads)<1 or args.block_size<0: p.error('workers/threads must be positive; block size must be >=0')
    if set_num_threads is None: print('Numba unavailable: slow Python fallback. Install numba for production.')
    if args.self_test:
        warm_worker(args.numba_threads); self_test(); return
    options=dict(overwrite=args.overwrite,threads=args.numba_threads,block=args.block_size,
                 center_search=args.center_search,spatial_index=not args.no_spatial_index)
    jobs=discover(Path(args.input),Path(args.output),args.recursive)
    print(f'{len(jobs)} orbit(s); file workers={args.workers}, threads/worker={args.numba_threads}',flush=True)
    run_start=time.perf_counter();failures=[];results=[];orbit_timings=[]
    # Build one master per orbit in the parent; release each tree immediately.
    tasks=[]
    for job in jobs:
        try:
            new,record=prepare_orbit(job,options);tasks.extend(new);orbit_timings.append(record)
        except Exception as exc:
            failures.append((job[0][1],str(exc)));print(f'ERROR preparing {job[0][1]}: {exc}',flush=True)
    with tempfile.TemporaryDirectory(prefix='awe_regrid_',dir=args.cache_parent) as cache:
        options['map_directory']=cache if args.reuse_maps else None
        remaining={}
        for task in tasks: remaining[task[2]]=remaining.get(task[2],0)+1
        def finished(task):
            remaining[task[2]]-=1
            if args.reuse_maps and remaining[task[2]]==0:
                shutil.rmtree(map_folder(cache,task[2]),ignore_errors=True)
        if args.reuse_maps: print('Map reuse enabled: exact geometry hashes; temporary cache can use ~1.8 MB per unique frame.',flush=True)
        if args.workers==1:
            warm_worker(args.numba_threads)
            for task in tasks:
                try: results.append(file_worker(task,options))
                except Exception as exc:
                    failures.append((task[0],str(exc)));print(f'ERROR {task[0]}: {exc}',flush=True)
                finally: finished(task)
        elif tasks:
            import multiprocessing as mp
            with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),mp_context=mp.get_context('spawn'),
                                     initializer=warm_worker,initargs=(args.numba_threads,)) as pool:
                pending={pool.submit(file_worker,task,options):task for task in tasks}
                for future in as_completed(pending):
                    try: results.append(future.result())
                    except Exception as exc:
                        name=pending[future][0];failures.append((name,str(exc)));print(f'ERROR {name}: {exc}',flush=True)
                    finally: finished(pending[future])
    elapsed=time.perf_counter()-run_start
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    for label,rows in [('file_timings',results),('orbit_timings',orbit_timings),('errors',[dict(file=f,error=e) for f,e in failures])]:
        if rows:
            report=output/f'regridding_{label}_{stamp}.csv'
            with report.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            print(f'Saved {report}',flush=True)
    print(f'Total wall time: {elapsed/60:.2f} min; successful files: {len(results)}; failures: {len(failures)}')
    if failures: raise SystemExit(1)


if __name__=='__main__':
    main()
