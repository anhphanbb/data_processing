#!/usr/bin/env python3
"""Standalone AWE L1A -> L2A -> L3A and L1A q20 -> L3C.

Requirements: numpy, scipy, netCDF4. No other processing script is imported.
Edit DEFAULT_* and run in Spyder, or run from Anaconda Prompt:
  python awe_l1a_to_l2a_l3a_l3c.py
  python awe_l1a_to_l2a_l3a_l3c.py --mode l2a --workers 4
  python awe_l1a_to_l2a_l3a_l3c.py --mode l3a --master-input E:\\soc\\l1r\\2023\\12
  python awe_l1a_to_l2a_l3a_l3c.py --mode l3c

2x2 and 10x10 mean km, following the supplied C++ SOC routines. They are
native 300x300 images and 5x5-pixel means (60x60), respectively. L3A uses
that L2A resolution; it does not bin a second time. L3C remains at 2 km.
Outputs: L2A_OUTPUT/{2x2,10x10}, L3A_OUTPUT/{2x2,10x10}, L3C_OUTPUT.
MasterNNNNN.mgf files from the earlier regridding step are required for L3.
Search DEFAULT_MASTER_INPUT recursively; duplicate orbit masters are rejected.

Scientific reference: AWE SOC SoftwareDlg(7).cpp:
  OnDataprocessingCreatel2filescorrection (native 2 km)
  OnDataprocessingCreatel2filesMLCorBin (10 km)
  OnDataprocessingCreatel3anobinning / Createl3aswathswithmlcloud
  OnDataprocessingCreateqlineswathsfroml1c (the MLCloud-carrying Q swath)
The last routine is applied to coadded L1A q20, as requested.

Compatibility notes (read before comparing):
* Default cpp_geometry=True preserves the source's coordinate choices:
  native L2A uses Q20 coordinates, although temperatures are on the P12 grid;
  10 km L2A uses P12 bin-centre coordinates. L3A later frames locate centre-1
  but paste from centre-2; L3C locates centre-2 but pastes from centre-3.
  --no-cpp-geometry fixes these offsets: both L2A outputs use P12 coordinates,
  and every swath strip is located by its actual first column. Do not mix
  geometry settings between stages.
* C++ local forward placement search and overwrite-by-later-frame are retained
  in compatibility mode. Safe nearest-grid fallback is used for missing edge
  coordinates, including an entirely masked anchor column, with both pixel
  offsets accounted for. Frames with no finite coordinate pairs are skipped
  and logged; placement after a gap uses a full-grid search. An entirely empty
  orbit fails. Non-overlap / out-of-bounds placement still raises an error.
* Native temperature is not clipped to 130..270 K. The 10 km average includes
  only quantized pixels strictly between 130 and 270 K, as in C++.
* Q-based iterative correction uses 2.644, 259.58 K, tolerance 0.01 K, at most
  11 iterations. BKG affects the common spatial bounds but is NOT subtracted
  (background subtraction is commented out in the supplied C++ routine).
* Invalid/negative/overflow conversion is defined as zero, avoiding undefined
  C++ floating-point-to-uint16 casts. Empty bins become NaN. Ordinary valid
  values retain the C++ truncation to 0.01 K / 0.001 kR.
* Pair filters by year/orbit/version/suffix, not HHMM; frame indices correspond
  as in C++. Require equal frame counts by default. --frame-policy common
  explicitly processes the common prefix if upstream files differ in length;
  this is NOT a time-matching operation and reports the trim.
* Global attributes are inherited from Q20 for L2A and from the input for L3,
  updating Data_level, Data_type, Logical_file_id/source, Filter, resolution.
  Telescope and Generation_Date are omitted, matching Files::CreateNetCDF.
  Logical identifiers use the real output filename (C++ had stale identifiers
  in some swath routines). No processing-history attributes are added.
* C++ temperature VALIDMIN/MAX inconsistently use 130/270 (native L2A) versus
  13000/27000 (binned L2A and L3A). These are retained for comparison; actual
  stored temperatures are float32 K. Readers here ignore those metadata bounds.
* Dimensions are CREATED in C++ order x_along_track, y_cross_track, time,
  x_box_along_track, y_box_cross_track. time is unlimited, spatial dims fixed.
  Image variables are (time,y_cross_track,x_along_track); MLCloud uses its
  separate box dimensions. Swaths have time length 1 and pixel-sized MLCloud.
* L3A/L3C Pixel_Epoch stores the source frame Epoch at every valid swath
  pixel (float64, same units/reference as Epoch). Overlapping strips replace
  data and time together. Epoch(time) remains unchanged for compatibility.
  Times represent coadding-window center frames, not individual exposures.
  Epoch and ISS vectors retain their first sample for swaths, like C++.
* Unknown scalar/time-only ancillary variables are retained. Unexpected spatial
  variables raise an error rather than silently dropping or mis-binning them.
* Writes use a temporary file then atomic rename. Existing outputs are skipped
  unless --overwrite is set. Do not reuse old outputs after changing settings.

Validation: iterative temperatures matched an independently compiled C++
loop for 12 randomized images. Synthetic NetCDF integration checks covered all
five products, dimension IDs, metadata, strip placement, MLCloud orientation,
both geometry settings, parallel execution, dateline handling, empty bins,
frame-count rejection, and a missing binned anchor. No real orbit NetCDFs or compiled full SOC executable were supplied;
real-data numerical/attribute equivalence still needs a comparison run.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import time
import traceback
import numpy as np
from netCDF4 import Dataset, num2date
from scipy.spatial import cKDTree

DEFAULT_L1A_INPUT = r'E:\soc\l1a\2023\12'
DEFAULT_L2A_OUTPUT = r'E:\soc\l2a\2023\12'
DEFAULT_L3A_OUTPUT = r'E:\soc\l3a\2023\12'
DEFAULT_L3C_OUTPUT = r'E:\soc\l3c\2023\12'
DEFAULT_MASTER_INPUT = r'E:\soc\l1r\2023\12'  # also accepts l1f folder with .mgf
DEFAULT_MODE = 'all'  # all, l2a, l3a, l3c
DEFAULT_RESOLUTIONS = (2, 10)  # km, each written in a separate folder
DEFAULT_WORKERS = 20
DEFAULT_BLOCK_SIZE = 32
DEFAULT_RECURSIVE = False
DEFAULT_OVERWRITE = True
DEFAULT_CPP_GEOMETRY = True
DEFAULT_FRAME_POLICY = 'strict'  # strict or common (explicit prefix trimming)
DEFAULT_EPOCH_MODE = 'awe-gps'

DIMS = ('time', 'y_cross_track', 'x_along_track')
BOX_DIMS = ('time', 'y_box_cross_track', 'x_box_along_track')
VECTORS = ('Epoch', 'ISS_Latitude', 'ISS_Longitude', 'ISS_Altitude',
           'ISS_Pitch', 'ISS_Roll', 'ISS_Yaw', 'Center_Latitude', 'Center_Longitude')
PATTERN = re.compile(r'^(?P<prefix>awe_)(?P<level>l[123][ac])_(?P<filter>p12|p14|bkg|q20|tmp)_'
                     r'(?P<date>\d{7})T\d{4}_(?P<orbit>\d{5})_(?P<version>v\d+)(?P<suffix>.*?)\.nc$', re.I)


def read_values(v, key, dtype=np.float32):
    v.set_auto_maskandscale(False)
    a = np.asarray(v[key])
    bad = np.zeros(a.shape, bool)
    for name in ('_FillValue', 'missing_value'):
        if name in v.ncattrs():
            for value in np.asarray(v.getncattr(name)).reshape(-1):
                bad |= a == value
    a = a.astype(dtype)
    if 'scale_factor' in v.ncattrs(): a *= v.scale_factor
    if 'add_offset' in v.ncattrs(): a += v.add_offset
    return np.where(bad, np.nan, a)


def matrix(a):
    """Disk (...,y,x) -> C++ (...,x,y), reversing cross-track direction."""
    return np.ascontiguousarray(np.swapaxes(a[..., ::-1, :], -1, -2))


def disk(a):
    return np.ascontiguousarray(np.swapaxes(a[..., :, ::-1], -1, -2))


def counts(a, scale):
    a = np.asarray(a, np.float32) * np.float32(scale)
    good = np.isfinite(a) & (a >= 0) & (a < 65536)
    return np.where(good, a, 0).astype(np.uint16)


def epoch_date(src, mode):
    value = float(read_values(src['Epoch'], 0, np.float64))
    if not np.isfinite(value): raise ValueError('First Epoch is not finite')
    if mode == 'awe-gps':
        return datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=value-1356566418)
    v = src['Epoch']
    units = getattr(v, 'units', getattr(v, 'UNITS', None))
    if not units: raise ValueError('CF Epoch requires units or UNITS')
    return num2date(value, units, calendar=getattr(v, 'calendar', 'standard'))


def output_name(src, source, level, mode):
    m = PATTERN.fullmatch(Path(source).name)
    emission = 'q20' if level == 'l3c' else 'tmp'
    date = epoch_date(src, mode)
    return f"awe_{level}_{emission}_{date.strftime('%Y%jT%H%M')}_{m['orbit']}_{m['version']}{m['suffix']}.nc"


def validate(src, name):
    if src.groups: raise ValueError('Grouped NetCDF is unsupported')
    v = src[name]
    if v.dimensions != DIMS or v.shape[0] < 1:
        raise ValueError(f'{name} expected {DIMS}, nonempty; got {v.dimensions} {v.shape}')
    for n in ('Latitude', 'Longitude'):
        if src[n].dimensions != DIMS or src[n].shape != v.shape:
            raise ValueError(f'{n} dimensions/shape do not match {name}')
    if src['Epoch'].dimensions != ('time',): raise ValueError('Epoch must be (time,)')
    if 'MLCloud' in src.variables:
        c = src['MLCloud']
        if c.ndim != 3 or c.dimensions[0] != 'time' or c.shape[1] != 3 or c.shape[2] < 8:
            raise ValueError(f'Expected MLCloud (time,3,>=8), got {c.shape}')
    return v.shape


def cloud_attributes():
    return dict(CATDESC='MLCloud array', DEPEND_0='Time', DISPLAY_TYPE='time_series',
                FIELDNAM='MLCloud', FILLVAL=np.int32(0), FORMAT='I', LABLAXIS='MLCloud',
                NC_DEFLATE='1', NC_SHUFFLE='true', NC_STORAGE='chunked', UNITS='Percentage',
                VALIDMIN=np.int32(0), VALIDMAX=np.int32(100), VAR_TYPE='data')


def writer(src, dst, name, level, nx, ny, nt, resolution, cloud_shape, block):
    """Create the canonical C++ dimension IDs and preserve source metadata types."""
    for d, size in [('x_along_track', nx), ('y_cross_track', ny), ('time', None)]:
        dst.createDimension(d, size)
    if cloud_shape:
        dst.createDimension('x_box_along_track', cloud_shape[0])
        dst.createDimension('y_box_cross_track', cloud_shape[1])
    attrs = {k: src.getncattr(k) for k in src.ncattrs()
             if k not in ('_NCProperties', 'Telescope', 'Generation_Date')}
    attrs.update(Data_level=level.upper(), Data_type=f'{level.upper()} > Level {level[1:].upper()}',
                 Logical_file_id=Path(name).stem, Logical_source=name[:11],
                 Filter='Q OH emission (1434.5 nm)' if level == 'l3c' else 'N/A',
                 Horizontal_Resolution=np.float32(resolution))
    if 'Data_Level' in attrs: attrs['Data_Level'] = level.upper()
    dst.setncatts(attrs)
    source_image = 'Temperature' if 'Temperature' in src.variables else 'Radiance'
    output_image = 'Radiance' if level == 'l3c' else 'Temperature'
    special = {'Temperature', 'Radiance', 'Latitude', 'Longitude', 'MLCloud'}
    extra = [n for n in src.variables if n not in special and n not in VECTORS]
    # Unexpected spatial fields must not be copied to a grid of a different size.
    for n in extra:
        if src[n].dimensions not in ((), ('time',)):
            raise ValueError(f'Unsupported ancillary variable {n}: {src[n].dimensions}')
    for n in [x for x in VECTORS if x in src.variables] + extra:
        v = src[n]
        if v.dimensions not in ((), ('time',)):
            raise ValueError(f'{n} must be scalar or (time,)')
        kw = dict(zlib=True, complevel=1, shuffle=True, chunksizes=(512 if n == 'Epoch' else 1024,)) if v.dimensions else {}
        if '_FillValue' in v.ncattrs(): kw['fill_value'] = v.getncattr('_FillValue')
        out = dst.createVariable(n, v.datatype, v.dimensions, **kw)
        out.setncatts({a: v.getncattr(a) for a in v.ncattrs() if a != '_FillValue'})
        v.set_auto_maskandscale(False)
        out.set_auto_maskandscale(False)
        if not v.dimensions: out[...] = v[...]
        else:
            for start in range(0, nt, block): out[start:min(start+block, nt)] = v[start:min(start+block, nt)]
    for n in (output_image, 'Latitude', 'Longitude'):
        v = src[source_image if n == output_image else n]
        out = dst.createVariable(n, 'f4', DIMS, zlib=True, complevel=1, shuffle=True,
                                 chunksizes=(1, ny, nx))
        att = {a: v.getncattr(a) for a in v.ncattrs() if a not in
               ('_FillValue', 'scale_factor', 'add_offset', '_Unsigned', 'actual_range',
                'valid_range', 'valid_min', 'valid_max', 'missing_value', 'NC_DIMENSIONS_V')}
        if n == output_image:
            lo, hi = ((0, 65535) if level == 'l3c' else
                      (130, 270) if level == 'l2a' and resolution == 2 else (13000, 27000))
            att.update(CATDESC='Q line swath image' if level == 'l3c' else 'Temperature images',
                       LABLAXIS='Q line radiance' if level == 'l3c' else 'OH (3,1) Rotational Temperature',
                       FIELDNAM=n, UNITS='kRayleigh' if level == 'l3c' else 'K',
                       VALIDMIN=np.float32(lo), VALIDMAX=np.float32(hi))
            if 'units' in att: att['units'] = att['UNITS']
        out.setncatts(att)
    if cloud_shape:
        out = dst.createVariable('MLCloud', 'i4', BOX_DIMS, zlib=True, complevel=1, shuffle=True,
                                 chunksizes=(1, cloud_shape[1], cloud_shape[0]))
        out.setncatts(cloud_attributes())


def shift_to_reference(a, shift):
    dx, dy = shift
    nx, ny = a.shape
    out = np.zeros_like(a)
    valid = np.zeros(a.shape, bool)
    x0, x1 = max(0, dx), min(nx, nx+dx)
    y0, y1 = max(0, dy), min(ny, ny+dy)
    out[x0:x1, y0:y1] = a[x0-dx:x1-dx, y0-dy:y1-dy]
    valid[x0:x1, y0:y1] = True
    return out, valid


def find_shift(ref_lat, ref_lon, lat, lon):
    cx, cy = np.array(ref_lat.shape)//2
    region = (slice(cx-20, cx+20), slice(cy-20, cy+20))
    hit = np.argwhere((lat[region] == ref_lat[cx, cy]) & (lon[region] == ref_lon[cx, cy]))
    if not len(hit):
        raise ValueError('No exact common centre-grid point in 40x40 search; check filter pairing / master grids')
    # Last match wins in the C++ j,k loops.
    x, y = hit[-1] + (cx-20, cy-20)
    return int(cx-x), int(cy-y)


def temperature(p12, p14, q20, valid_initial, valid_all):
    """P12-grid temperature counts; already aligned P14/Q20, double arithmetic."""
    a, b, q = (v.astype(np.float64) for v in (p12, p14, q20))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ratio = 2.644*a/b
        good = valid_initial & (ratio > 0)
        avg = np.mean(259.58/np.log(ratio[good])) if good.any() else np.nan
        result = np.zeros(a.shape, np.uint16)
        for _ in range(11):
            correction = np.where(valid_all & (a > q), 2.644*(a-q)/np.exp(259.58/avg), 0)
            ratio = np.where(correction > 0, 2.644*q/(b-correction), 2.644*a/b)
            good = valid_all & (ratio > 0)
            t = np.where(good, 259.58/np.log(ratio), 0)
            encoded = t*100
            safe = np.isfinite(encoded) & (encoded >= 0) & (encoded < 65536)
            result = np.where(safe, encoded, 0).astype(np.uint16)
            new_avg = np.mean(t[good]) if good.any() else np.nan
            # C++ stops on NaN too: fabs(NaN)>0.01 is false.
            if not abs(avg-new_avg) > 0.01: break
            avg = new_avg
    return result


def bin_temperature(t, lat, lon, factor):
    if factor == 1: return t, lat.copy(), lon.copy()
    nx, ny = t.shape
    if nx % factor or ny % factor: raise ValueError('Image size not divisible by bin factor')
    shaped = t.reshape(nx//factor, factor, ny//factor, factor)
    good = (shaped > 13000) & (shaped < 27000)
    number = good.sum(axis=(1, 3))
    sums = np.where(good, shaped, 0).sum(axis=(1, 3), dtype=np.float64)
    out = np.divide(sums, number, out=np.zeros_like(sums), where=number > 0).astype(np.uint16)
    centre = factor//2
    la, lo = lat[centre::factor, centre::factor].copy(), lon[centre::factor, centre::factor].copy()
    la[number == 0] = np.nan
    lo[number == 0] = np.nan
    return out, la, lo


def write_plane(dst, i, data, lat, lon, scale, image):
    values = data.astype(np.float32)/np.float32(scale)
    values[~np.isfinite(lat)] = np.nan
    dst[image][i] = disk(values)
    dst['Latitude'][i] = disk(lat)
    dst['Longitude'][i] = disk(lon)


def process_l2a(entries, options):
    tick = time.perf_counter()
    partials, finished = [], []
    try:
        with ExitStack() as stack:
            sources = {f: stack.enter_context(Dataset(p)) for f, p in entries.items()}
            shapes = {f: validate(s, 'Radiance') for f, s in sources.items()}
            if any(s[1:] != (300, 300) for s in shapes.values()):
                raise ValueError(f'L1A must be 300x300 native 2 km images: {shapes}')
            lengths = {f: s[0] for f, s in shapes.items()}
            if len(set(lengths.values())) != 1:
                if options['frame_policy'] == 'strict':
                    raise ValueError(f'Filter frame counts differ: {lengths}; inspect upstream selection or explicitly use --frame-policy common')
                print(f'WARNING common-prefix frame trim: {lengths}', flush=True)
            nt = min(lengths.values())
            ref = sources['q20']  # C++ final OpenNetCDF provides Q header / vectors / MLCloud
            name = output_name(ref, entries['q20'], 'l2a', options['epoch_mode'])
            outputs = {}
            for resolution in options['resolutions']:
                target = Path(options['l2a_output'])/f'{resolution}x{resolution}'/name
                if target.exists() and not options['overwrite']:
                    print(f'SKIP {target}', flush=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name+f'.{os.getpid()}.partial')
                partials.append((tmp, target))
                dst = stack.enter_context(Dataset(tmp, 'w', format='NETCDF4'))
                factor = resolution//2
                cloud_shape = (ref['MLCloud'].shape[2], ref['MLCloud'].shape[1]) if 'MLCloud' in ref.variables else None
                writer(ref, dst, name, 'l2a', 300//factor, 300//factor, nt, resolution,
                       cloud_shape, options['block_size'])
                outputs[resolution] = dst
            if not outputs: return f'L2A skipped {name}'
            for start in range(0, nt, options['block_size']):
                end = min(nt, start+options['block_size'])
                block = {}
                for f, s in sources.items():
                    block[f] = {n: matrix(read_values(s[n], slice(start, end))) for n in ('Radiance', 'Latitude', 'Longitude')}
                    block[f]['Radiance'] = counts(block[f]['Radiance'], 1000)
                for j, i in enumerate(range(start, end)):
                    p = {f: {n: a[j] for n, a in b.items()} for f, b in block.items()}
                    la, lo = p['p12']['Latitude'], p['p12']['Longitude']
                    aligned, bounds = {}, {}
                    for f in ('p14', 'bkg', 'q20'):
                        shift = find_shift(la, lo, p[f]['Latitude'], p[f]['Longitude'])
                        aligned[f], bounds[f] = shift_to_reference(p[f]['Radiance'], shift)
                    initial_valid = bounds['p14'] & bounds['bkg']
                    all_valid = initial_valid & bounds['q20']
                    if getattr(ref, 'MLSPB_Mask_Mode', '') in ('boxes_clean_reference', 'boxes_trim_reference'):
                        # Zero counts encode absent radiance. Do not solve temperature
                        # from a subset of the required aligned spectral measurements.
                        measured = (p['p12']['Radiance'] > 0)
                        for f in ('p14', 'q20', 'bkg'): measured &= aligned[f] > 0
                        initial_valid &= measured
                        all_valid &= measured
                    t = temperature(p['p12']['Radiance'], aligned['p14'], aligned['q20'],
                                    initial_valid, all_valid)
                    for resolution, dst in outputs.items():
                        # Preserve the specific native-coordinate choice in source C++.
                        base_la, base_lo = ((p['q20']['Latitude'], p['q20']['Longitude'])
                                           if resolution == 2 and options['cpp_geometry'] else (la, lo))
                        data, out_la, out_lo = bin_temperature(t, base_la, base_lo, resolution//2)
                        write_plane(dst, i, data, out_la, out_lo, 100, 'Temperature')
                if 'MLCloud' in ref.variables:
                    cloud = read_values(ref['MLCloud'], slice(start, end))
                    if not np.all(np.isfinite(cloud)): raise ValueError('Nonfinite MLCloud values')
                    for dst in outputs.values(): dst['MLCloud'][start:end] = cloud.astype(np.int32)
        for tmp, target in partials:
            os.replace(tmp, target)
            finished.append(str(target))
        return f'L2A {name}: {nt} frames, {len(finished)} resolution(s), {time.perf_counter()-tick:.1f}s'
    finally:
        for tmp, _ in partials:
            if tmp.exists(): tmp.unlink()


def read_master(path):
    with open(path, 'rb') as f:
        sizes = np.fromfile(f, dtype='<i4', count=2)
    if len(sizes) != 2 or min(sizes) < 1: raise ValueError(f'Invalid master header: {path}')
    nx, ny = map(int, sizes)
    if Path(path).stat().st_size != 8+16*nx*ny: raise ValueError(f'Master file size mismatch: {path}')
    lat = np.memmap(path, dtype='<f8', mode='r', offset=8, shape=(nx, ny))
    lon = np.memmap(path, dtype='<f8', mode='r', offset=8+8*nx*ny, shape=(nx, ny))
    return lat, lon


class GridIndex:
    def __init__(self, lat, lon, invert):
        self.lat = np.asarray(lat)
        self.lon = np.where(np.asarray(lon) < 0, np.asarray(lon)+360, lon) if invert else np.asarray(lon)
        self.invert = invert
        self.ids = np.flatnonzero(np.isfinite(self.lat) & np.isfinite(self.lon))
        if not len(self.ids): raise ValueError('Master grid has no finite coordinates')
        self.tree = cKDTree(np.column_stack((self.lat.ravel()[self.ids], self.lon.ravel()[self.ids])))

    def query(self, la, lo):
        if not np.isfinite(la) or not np.isfinite(lo): raise ValueError('Nonfinite swath anchor')
        if self.invert and lo < 0: lo += 360
        distance, index = self.tree.query([la, lo])
        pos = np.unravel_index(self.ids[index], self.lat.shape)
        return tuple(map(int, pos)), float(distance)


def invert_longitudes(first, last):
    a, b = first[np.isfinite(first)], last[np.isfinite(last)]
    if not len(a) or not len(b): raise ValueError('First/last frame has no finite longitudes')
    return bool(np.min(b) < np.min(a) or (np.min(a) < -90 and np.max(a) > 0))


def anchor_position(grid, lat, lon, column, pixel_step=1):
    # Recover the requested column's row-zero position, even when that
    # entire column is masked. Arrays here use (along-track, cross-track).
    if np.isfinite(lat[column, 0]) and np.isfinite(lon[column, 0]):
        return grid.query(lat[column, 0], lon[column, 0])[0]
    valid = np.isfinite(lat) & np.isfinite(lon)
    ys = np.flatnonzero(valid[column])
    if len(ys):
        x, y = column, int(ys[0])
    else:
        candidates = np.argwhere(valid)
        if not len(candidates):
            raise ValueError('Frame has no finite coordinate pairs')
        # Prefer the nearest column, then its first valid row.
        order = np.lexsort((candidates[:, 1], np.abs(candidates[:, 0]-column)))
        x, y = map(int, candidates[order[0]])
    (xg, yg), _ = grid.query(lat[x, y], lon[x, y])
    return xg-(x-column)*pixel_step, yg-y*pixel_step


def coordinate_frame(src, indices):
    """Find an endpoint with coordinates; empty endpoints cannot set the phase."""
    for frame in indices:
        lat, lon = [matrix(read_values(src[n], frame))
                    for n in ('Latitude', 'Longitude')]
        valid = np.isfinite(lat) & np.isfinite(lon)
        if valid.any():
            return frame, lat, np.where(valid, lon, np.nan)
    raise ValueError('Orbit has no finite coordinate pairs; no swath written')


def local_position(grid, lat, lon, column, previous, q_swath):
    if not np.isfinite(lat[column, 0]) or not np.isfinite(lon[column, 0]):
        return anchor_position(grid, lat, lon, column)
    px, py = previous
    best, distance = None, np.inf
    target_lon = float(lon[column, 0])
    if grid.invert and target_lon < 0: target_lon += 360
    for dx in (range(1, 6) if q_swath else range(0, 5)):
        for dy in range(-2, 3):
            x, y = px+dx, py+dy
            if 0 <= x < grid.lat.shape[0] and 0 <= y < grid.lat.shape[1]:
                d = (grid.lat[x, y]-lat[column, 0])**2 + (grid.lon[x, y]-target_lon)**2
                if d < distance: best, distance = (x, y), d
    if best is None: raise ValueError('C++ local swath search found no in-bounds pixel')
    return best


def process_swath(source, master, level, resolution, options):
    tick = time.perf_counter()
    temp_path = None
    try:
        with Dataset(source) as src:
            image = 'Radiance' if level == 'l3c' else 'Temperature'
            nt, ny, nx = validate(src, image)
            factor = resolution//2
            if (nx, ny) != (300//factor, 300//factor):
                raise ValueError(f'{source}: dimensions disagree with {resolution} km resolution')
            actual = float(getattr(src, 'Horizontal_Resolution', resolution))
            if not np.isclose(actual, resolution): raise ValueError(f'Input Horizontal_Resolution={actual}, expected {resolution}')
            name = output_name(src, source, level, options['epoch_mode'])
            target = Path(options[f'{level}_output'])
            if level == 'l3a': target /= f'{resolution}x{resolution}'
            target /= name
            if target.exists() and not options['overwrite']: return f'SKIP {target}'
            first_valid, la0, lo0 = coordinate_frame(src, range(nt))
            _, _, last_lon = coordinate_frame(src, range(nt-1, first_valid-1, -1))
            invert = invert_longitudes(lo0, last_lon)
            ml, mo = read_master(master)
            grid = GridIndex(ml, mo, invert)
            width = 6 if level == 'l3c' else 4
            first_column = nx//2-width//2
            first_anchor = nx//2-2 if options['cpp_geometry'] else first_column
            position = anchor_position(grid, la0, lo0, first_anchor, factor)
            if factor > 1:
                # Exact C++ 10 km sampling phase: remainder of (position-2)/5,
                # dimension = int(full_size/5 - 1). C++ truncates toward zero.
                phase = tuple((p-2)-int((p-2)/factor)*factor for p in position)
                size = tuple(int(s/factor-1) for s in ml.shape)
                ix = np.arange(size[0])*factor+2+phase[0]
                iy = np.arange(size[1])*factor+2+phase[1]
                if min(ix) < 0 or min(iy) < 0 or max(ix) >= ml.shape[0] or max(iy) >= ml.shape[1]:
                    raise ValueError('Binned master indices out of bounds')
                sampled_lat, sampled_lon = ml[np.ix_(ix, iy)], mo[np.ix_(ix, iy)]
                del grid
                grid = GridIndex(sampled_lat, sampled_lon, invert)
                position = anchor_position(grid, la0, lo0, first_anchor)
            epochs = read_values(src['Epoch'], slice(0, nt), np.float64)
            pixel_epoch = np.full(grid.lat.shape, np.nan, dtype=np.float64)
            swath = np.zeros(grid.lat.shape, np.uint16)
            cloud_swath = np.zeros(grid.lat.shape, np.int32) if 'MLCloud' in src.variables else None
            skipped_frames, fallback_frames = [], []
            previous_frame = None
            for start in range(0, nt, options['block_size']):
                end = min(nt, start+options['block_size'])
                data = counts(matrix(read_values(src[image], slice(start, end))), 1000 if level == 'l3c' else 100)
                lat = matrix(read_values(src['Latitude'], slice(start, end)))
                lon = matrix(read_values(src['Longitude'], slice(start, end)))
                cloud = matrix(read_values(src['MLCloud'], slice(start, end))) if cloud_swath is not None else None
                if cloud is not None and not np.all(np.isfinite(cloud)): raise ValueError('Nonfinite MLCloud values')
                for j, frame in enumerate(range(start, end)):
                    valid = np.isfinite(lat[j]) & np.isfinite(lon[j])
                    if not valid.any():
                        skipped_frames.append(frame)
                        continue
                    anchor = (first_anchor if frame == first_valid else
                              (nx//2-2 if level == 'l3c' else nx//2-1)
                              if options['cpp_geometry'] else first_column)
                    if not valid[anchor].any():
                        fallback_frames.append(frame)
                    if frame != first_valid:
                        # A gap may exceed the small C++ forward search window.
                        if options['cpp_geometry'] and previous_frame == frame-1:
                            position = local_position(grid, lat[j], lon[j], anchor, position, level == 'l3c')
                        else:
                            position = anchor_position(grid, lat[j], lon[j], anchor)
                    previous_frame = frame
                    x, y = position
                    if x < 0 or y < 0 or x+width > swath.shape[0] or y+ny > swath.shape[1]:
                        raise ValueError(f'Frame {frame}: strip at {position} outside master {swath.shape}')
                    strip = data[j, first_column:first_column+width, :]
                    keep = np.ones(strip.shape, bool)
                    if getattr(src, 'MLSPB_Mask_Mode', '') in ('boxes_clean_reference', 'boxes_trim_reference'):
                        keep = (strip > 0) & valid[first_column:first_column+width, :]
                    # A missing sample must not erase an earlier valid overlap.
                    view = swath[x:x+width, y:y+ny]
                    view[keep] = strip[keep]
                    epoch_view = pixel_epoch[x:x+width, y:y+ny]
                    epoch_view[keep] = epochs[frame]
                    if cloud is not None:
                        values = np.broadcast_to(cloud[j, 7, np.arange(ny)*3//ny].astype(np.int32)[None, :], strip.shape)
                        cloud_view = cloud_swath[x:x+width, y:y+ny]
                        cloud_view[keep] = values[keep]
            for label, frames in [('skipped frames with no coordinate pairs', skipped_frames),
                                  ('used another anchor column', fallback_frames)]:
                if frames:
                    print(f'WARNING {name}: {label}: {len(frames)}; '
                          f'zero-based frame indices={frames}', flush=True)
            out_lat, out_lon = grid.lat.copy(), grid.lon.copy()
            if invert: out_lon[out_lon > 180] -= 360
            pixel_epoch[(swath == 0) | ~np.isfinite(grid.lat) |
                        ~np.isfinite(grid.lon) | ~np.isfinite(pixel_epoch)] = np.nan
            out_lat[swath == 0] = np.nan
            out_lon[swath == 0] = np.nan
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_path = target.with_name(target.name+f'.{os.getpid()}.partial')
            with Dataset(temp_path, 'w', format='NETCDF4') as dst:
                writer(src, dst, name, level, swath.shape[0], swath.shape[1], 1, resolution,
                       swath.shape if cloud_swath is not None else None, options['block_size'])
                write_plane(dst, 0, swath, out_lat, out_lon, 1000 if level == 'l3c' else 100, image)
                epoch_out = dst.createVariable('Pixel_Epoch', 'f8', DIMS,
                                               fill_value=np.nan, zlib=True, complevel=1,
                                               shuffle=True,
                                               chunksizes=(1, swath.shape[1], swath.shape[0]))
                # Values are decoded by read_values; do not copy packing or
                # frame-coordinate metadata from the one-dimensional Epoch.
                epoch_out.setncatts({a: src['Epoch'].getncattr(a)
                                    for a in ('units', 'UNITS', 'calendar',
                                              'TIME_BASE', 'TIME_SCALE', 'REFERENCE_POSITION')
                                    if a in src['Epoch'].ncattrs()})
                epoch_out.setncatts(dict(
                    long_name='Representative observation time for each swath pixel',
                    FIELDNAM='Pixel_Epoch', VAR_TYPE='support_data',
                    CATDESC='Source frame Epoch for each valid swath pixel',
                    comment='Same time units and reference as Epoch. Source frame time '
                            'represents the center frame of the upstream coadding window. '
                            'Later strips overwrite both pixel values and times; '
                            'pixels without valid data or time are missing.'))
                epoch_out[0] = disk(pixel_epoch)
                if cloud_swath is not None: dst['MLCloud'][0] = disk(cloud_swath)
        os.replace(temp_path, target)
        return f'{level.upper()} {name}: {nt} -> 1 frame, {swath.shape[0]}x{swath.shape[1]}, {time.perf_counter()-tick:.1f}s'
    finally:
        if temp_path is not None and temp_path.exists(): temp_path.unlink()


def discover(root, level, recursive):
    root = Path(root)
    if not root.is_dir(): raise ValueError(f'Input folder does not exist: {root}')
    found = {}
    for p in sorted(root.rglob('*.nc') if recursive else root.glob('*.nc')):
        m = PATTERN.fullmatch(p.name)
        if not m or m['level'].lower() != level: continue
        key = (m['date'][:4], m['orbit'], m['version'].lower(), m['suffix'])
        filt = m['filter'].lower()
        group = found.setdefault(key, {})
        if filt in group: raise ValueError(f'Ambiguous duplicate: {group[filt]} and {p}')
        group[filt] = str(p)
    return found


def masters(root):
    root = Path(root)
    if not root.is_dir(): raise ValueError(f'Master folder does not exist: {root}')
    result = {}
    for p in sorted(root.rglob('*.mgf')):
        m = re.fullmatch(r'Master(\d{5})\.mgf', p.name, re.I)
        if not m: continue
        orbit = m[1]
        if orbit in result: raise ValueError(f'Multiple master grids for orbit {orbit}; narrow --master-input')
        result[orbit] = str(p)
    return result


def run_task(task, options):
    if task[0] == 'l2a': return process_l2a(task[1], options)
    return process_swath(*task[1:], options)


def run_batch(tasks, options):
    if not tasks: raise ValueError('No matching input files found')
    failures = []
    if options['workers'] == 1:
        for task in tasks:
            try: print(run_task(task, options), flush=True)
            except Exception as e:
                failures.append(str(e)); traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=options['workers']) as pool:
            jobs = {pool.submit(run_task, task, options): task for task in tasks}
            for future in as_completed(jobs):
                try: print(future.result(), flush=True)
                except Exception as e:
                    failures.append(str(e)); print(f'ERROR {jobs[future]}: {e}', flush=True)
    if failures: raise RuntimeError(f'{len(failures)} task(s) failed; downstream processing stopped. First: {failures[0]}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag, value in [('l1a-input', DEFAULT_L1A_INPUT), ('l2a-output', DEFAULT_L2A_OUTPUT),
                        ('l3a-output', DEFAULT_L3A_OUTPUT), ('l3c-output', DEFAULT_L3C_OUTPUT),
                        ('master-input', DEFAULT_MASTER_INPUT)]: parser.add_argument('--'+flag, default=value)
    parser.add_argument('--mode', choices=['all', 'l2a', 'l3a', 'l3c'], default=DEFAULT_MODE)
    parser.add_argument('--resolutions', type=int, nargs='+', choices=[2, 10], default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    parser.add_argument('--block-size', type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument('--recursive', action=argparse.BooleanOptionalAction, default=DEFAULT_RECURSIVE)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=DEFAULT_OVERWRITE)
    parser.add_argument('--cpp-geometry', action=argparse.BooleanOptionalAction, default=DEFAULT_CPP_GEOMETRY)
    parser.add_argument('--frame-policy', choices=['strict', 'common'], default=DEFAULT_FRAME_POLICY)
    parser.add_argument('--epoch-mode', choices=['awe-gps', 'cf'], default=DEFAULT_EPOCH_MODE)
    options = vars(parser.parse_args(argv))
    if options['workers'] < 1 or options['block_size'] < 1: parser.error('workers and block-size must be >=1')
    options['resolutions'] = list(dict.fromkeys(options['resolutions']))
    tick = time.perf_counter()
    print(f"Mode={options['mode']}; resolution(s)={options['resolutions']} km; workers={options['workers']}; cpp_geometry={options['cpp_geometry']}", flush=True)
    need_l1 = options['mode'] in ('all', 'l2a', 'l3c')
    groups = discover(options['l1a_input'], 'l1a', options['recursive']) if need_l1 else {}
    master_map = masters(options['master_input']) if options['mode'] != 'l2a' else {}
    # Preflight before expensive processing; no silent missing-filter/orbit skips.
    if options['mode'] in ('all', 'l2a'):
        tasks = []
        for key, entries in groups.items():
            missing = set(('p12', 'p14', 'bkg', 'q20'))-entries.keys()
            if missing: raise ValueError(f'Orbit {key}: missing filters {sorted(missing)}')
            if options['mode'] == 'all' and key[1] not in master_map:
                raise ValueError(f'Missing Master{key[1]}.mgf under {options["master_input"]}')
            tasks.append(('l2a', {f: entries[f] for f in ('p12', 'p14', 'bkg', 'q20')}))
        run_batch(tasks, options)
    if options['mode'] in ('all', 'l3a'):
        tasks = []
        for resolution in options['resolutions']:
            entries = discover(Path(options['l2a_output'])/f'{resolution}x{resolution}', 'l2a', options['recursive'])
            for key, files in entries.items():
                if options['mode'] == 'all' and key not in groups: continue
                if 'tmp' not in files: continue
                if key[1] not in master_map: raise ValueError(f'Missing Master{key[1]}.mgf')
                tasks.append(('swath', files['tmp'], master_map[key[1]], 'l3a', resolution))
        run_batch(tasks, options)
    if options['mode'] in ('all', 'l3c'):
        tasks = []
        for key, entries in groups.items():
            if 'q20' not in entries:
                raise ValueError(f'Orbit {key}: missing q20')
            if key[1] not in master_map: raise ValueError(f'Missing Master{key[1]}.mgf')
            tasks.append(('swath', entries['q20'], master_map[key[1]], 'l3c', 2))
        run_batch(tasks, options)
    print(f'DONE total wall time: {time.perf_counter()-tick:.1f}s', flush=True)


if __name__ == '__main__':
    main()
