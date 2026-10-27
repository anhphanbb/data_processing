#!/usr/bin/env python3
"""Standalone AWE co-adding: L1R p12/p14/bkg and post-ML L1C q20 -> L1A.

Install: pip install numpy netCDF4 numba
Edit DEFAULT_* below and run this file in Spyder, or:
  python coadd_to_l1a.py --mode both
  python coadd_to_l1a.py --mode p --l1r-input E:\\soc\\l1r\\2023\\12
  python coadd_to_l1a.py --mode q --l1c-input E:\\soc\\l1c\\2023\\12
  python coadd_to_l1a.py --workers 4 --block-size 32 --cpp-compat
  python coadd_to_l1a.py --self-test

Source: AWE SOC SoftwareDlg(6).cpp, OnDataprocessingCo32828 and
OnDataprocessingCoML; Files(6).cpp, OpenNextNetCDF/CreateNetCDF/AddNetCDF.
No other Python script or master-grid file is required. Each filter is
co-added independently; p12, p14 and bkg are NOT averaged together.

Algorithm:
* 19 consecutive frames, one output per window, middle-frame geolocation,
  Epoch and ancillary data. No additional ML radiance masking is performed.
* C++ orientation: disk plane[::-1, :].T; float32 Radiance * 1000 -> uint16.
* C++ five-pixel right-edge trimming, including its pre-decrement and j=10
  stopping rule. Latitude/Longitude at those pixels become NaN.
* Ordered four-neighbor walk finds the middle image's central coordinate in
  each source image. If the walk stalls, scan the whole source image and accept
  only a unique exact match. Missing/ambiguous matches fail with diagnostics;
  no approximate alignment is substituted. This also applies in --cpp-compat.
* Integer translations; accept counts >1000 for p12/q20, >100 for p14/bkg.
  Arithmetic mean of accepted counts, truncate to integer, /1000 -> float32.
  No contributors -> zero; middle-frame NaN latitude -> NaN radiance.
* Q cloud values: MLCloud integer percentages, or Processed_MLCloud *100 if
  MLCloud is absent. The actual C++ code computes a shifted arithmetic mean,
  NOT the maximum described by its comments. C++ truncation toward zero and
  duplicated contributions from the two box offsets are retained.

Intentional corrections, enabled by default:
* Use all N-window+1 complete windows (19 -> lose 9 frames at each end).
  C++ uses N-window, dropping one extra final window.
* Aggregate every cloud column (usually 15). C++ only updates columns 0..13;
  its final column is left from the newest input frame. --cpp-compat restores
  BOTH of those behaviors for comparisons (N must then exceed window).
* Output filename YYYYDDDTxxxx is computed directly from Epoch[window//2],
  including date/year rollover. Default Epoch convention matches the supplied
  AWE C++ anchor: 1356566418 = 2023-01-01T00:00:00 UTC (GPS-UTC=18 seconds).
  Use --epoch-mode cf for genuinely CF UTC time values instead; the source
  files' 'seconds since 1980...' label alone does not distinguish GPS/UTC.

Output: NETCDF4; dimension IDs x=0, y=1, time=2, matching the C++ writer;
unlimited time (as in CreateNetCDF); image dimensions
(time,y_cross_track,x_along_track); float32 images, integer MLCloud.
Ancillary variables are copied at middle-frame indices; other input metadata
is retained, with Data_level/Data_type, identifiers, Number_coadded_images
updated. Existing Data_Level alias is also updated. This preserves extra
Python/ML metadata rather than promising byte-identical C++ headers.
Date_Start/Date_End and other inherited acquisition metadata stay unchanged,
matching the C++ writer. Source paths, modes, and timings go to the run CSV.

One input handle and output handle per worker; bounded blocks with a window
halo; image chunks (1,ny,nx), deflate 1; optional Numba; no fastmath.
Start with 4 workers and increase after checking memory/disk throughput.
Outputs are written to unique temporary files, renamed after successful close.
Existing files are skipped unless --overwrite. Real C++ product comparison
is still needed: the supplied attachments contain source code, not NetCDF data.

Validation during conversion: extracted C++ alignment/averaging code matched
every shift and all 4096 integer pixels on a synthetic moving grid. Synthetic
NetCDF tests passed for all filters, both compatibility modes, block-size
invariance, midnight/year rollover, cloud conversion, dateline crossing,
parallel workers, failure cleanup, and the fallback without Numba.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import re
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from netCDF4 import Dataset, num2date
try:
    from numba import njit, set_num_threads
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        return lambda f: f

DEFAULT_L1R_INPUT = r'E:\soc\l1r\2023\11'
DEFAULT_L1C_INPUT = r'E:\soc\l1c\2023\11'
DEFAULT_OUTPUT = r'E:\soc\l1a\2023\11'
DEFAULT_MODE = 'q'  # 'p': p12/p14/bkg; 'q': q20; 'both': both inputs
DEFAULT_WORKERS = 20
DEFAULT_NUMBA_THREADS = 1
DEFAULT_BLOCK_SIZE = 32  # Output frames per block; reads block+window-1 inputs
DEFAULT_NB_COADD = 19
DEFAULT_RECURSIVE = False
DEFAULT_OVERWRITE = True
DEFAULT_CPP_COMPAT = False  # True: extra end trim + original 14-column ML rule
DEFAULT_EPOCH_MODE = 'awe-gps'
PATTERN = re.compile(r'^(?P<prefix>.*?)l1(?P<level>[rc])_(?P<filter>p12|p14|bkg|q20)_(?P<stamp>\d{7}T\d{4})_(?P<orbit>\d{5})_(?P<version>v\d+)(?P<suffix>.*?)\.nc$', re.I)
IMAGES = ('Radiance', 'Latitude', 'Longitude')
ORDER = ('Epoch', 'ISS_Latitude', 'ISS_Longitude', 'ISS_Altitude', 'ISS_Pitch',
         'ISS_Roll', 'ISS_Yaw', 'Center_Latitude', 'Center_Longitude', *IMAGES, 'MLCloud')


def read_values(var, key, dtype=np.float32):
    # Honor fill values but do not mask valid counts using stale valid_range.
    var.set_auto_maskandscale(False)
    raw = np.asarray(var[key])
    bad = np.zeros(raw.shape, dtype=bool)
    for name in ('_FillValue', 'missing_value'):
        if name in var.ncattrs():
            for value in np.asarray(var.getncattr(name)).reshape(-1):
                bad |= raw == value
    result = raw.astype(dtype)
    if 'scale_factor' in var.ncattrs(): result *= var.scale_factor
    if 'add_offset' in var.ncattrs(): result += var.add_offset
    result[bad] = np.nan
    return result


def cloud_percentages(values, name, first_frame=0):
    """Allow float32 endpoint roundoff; reject missing or genuinely invalid scores."""
    upper = 1.0 if name == 'Processed_MLCloud' else 100.0
    # 1e-5 in probability units, or 0.001 percentage points.
    tolerance = upper * 1e-5
    bad = ~np.isfinite(values) | (values < -tolerance) | (values > upper+tolerance)
    if np.any(bad):
        index = tuple(np.argwhere(bad)[0])
        frame = first_frame + int(index[0])
        raise ValueError(
            f'{name}: {np.count_nonzero(bad)} invalid cloud scores in block; '
            f'first at source frame {frame}, box indices {index[1:]}: '
            f'{values[index]!r}; expected finite 0..{upper:g} '
            f'(endpoint tolerance {tolerance:g})')
    values = np.clip(values, 0.0, upper)
    if name == 'Processed_MLCloud':
        values = values * np.float32(100)
    return values.astype(np.int32)


def matrix_cube(a):
    return np.ascontiguousarray(a[:, ::-1, :].transpose(0, 2, 1))


def disk_plane(a):
    return a[:, ::-1].T


def to_counts(a):
    a = np.where(np.isfinite(a) & (a > 0), a, np.float32(0))
    return np.clip(a * np.float32(1000), 0, 65535).astype(np.uint16)


@njit(cache=True)
def trim_edges(counts, lat, lon):
    nt, nx, ny = counts.shape
    for t in range(nt):
        for y in range(ny):
            x = nx - 2  # C++ starts at nx-1 and decrements before checking
            while counts[t, x, y] == 0 and x != 10:
                x -= 1
            for c in range(5):
                counts[t, x-c+1, y] = 0
                lat[t, x-c+1, y] = np.nan
                lon[t, x-c+1, y] = np.nan


@njit(cache=True)
def invert_longitudes(first, last):
    # C++ extrema are integers, so truncate on EACH assignment, in matrix order.
    low, high, lastlow = 180, -180, 180
    for x in range(first.shape[0]):
        for y in range(first.shape[1]):
            a, b = first[x, y], last[x, y]
            if np.isfinite(a):
                if a < low: low = int(a)
                if a > high: high = int(a)
            if np.isfinite(b) and b > -180 and b < lastlow: lastlow = int(b)
    return lastlow < low or (low < 0 and high > 0)


@njit(cache=True)
def exact_grid_match(lat, lon, target_lat, target_lon):
    """Full-image fallback: never replace a missing match with a nearest pixel."""
    bx, by, matches = -1, -1, 0
    for x in range(lat.shape[0]):
        for y in range(lat.shape[1]):
            if lat[x,y] == target_lat and lon[x,y] == target_lon:
                bx, by = x, y
                matches += 1
    if matches == 0:
        raise ValueError('No exact shared-grid match after full-image search')
    if matches > 1:
        raise ValueError('Ambiguous exact shared-grid matches after full-image search')
    return bx, by


def alignment_diagnostics(lat, lon, start, window, block_start):
    """Failure-only report; indices are zero-based, in internal (x,y) order."""
    cx, cy = lat.shape[1]//2, lat.shape[2]//2
    ta, tb = lat[start+window//2,cx,cy], lon[start+window//2,cx,cy]
    details = [f'target lat/lon=({ta:.12g}, {tb:.12g}); indices zero-based; '
               'coordinates after longitude normalization and edge trimming']
    if not np.isfinite(ta) or not np.isfinite(tb):
        return '; '.join(details)
    for offset in range(window):
        a, b = lat[start+offset], lon[start+offset]
        finite = np.isfinite(a) & np.isfinite(b)
        matches = np.count_nonzero(finite & (a == ta) & (b == tb))
        if matches == 1:
            continue
        desc = f'source frame {block_start+start+offset}: exact_matches={matches}'
        if finite.any():
            distance2 = np.where(finite, (a-ta)**2+(b-tb)**2, np.inf)
            x, y = np.unravel_index(np.argmin(distance2), a.shape)
            desc += (f', nearest internal (x,y)=({x},{y}), '
                     f'lat/lon=({a[x,y]:.12g}, {b[x,y]:.12g}), '
                     f'delta lat/lon=({a[x,y]-ta:.12g}, {b[x,y]-tb:.12g}) deg')
        else:
            desc += ', no finite coordinate pairs'
        details.append(desc)
    return '; '.join(details)


@njit(cache=True)
def find_shifts(lat, lon, start, window):
    nx, ny = lat.shape[1:]
    cx, cy = nx//2, ny//2
    target_lat, target_lon = lat[start+window//2,cx,cy], lon[start+window//2,cx,cy]
    if not np.isfinite(target_lat) or not np.isfinite(target_lon):
        raise ValueError('Non-finite middle-frame central coordinate')
    shifts = np.empty((window, 2), dtype=np.int64)
    for i in range(window):
        bx, by, best = cx, cy, 10000.0
        # The source checks neighbors before the current center; retain its order.
        for step in range(nx*ny+1):
            x, y = bx, by
            for direction in range(4):
                u = x + (1 if direction == 0 else -1 if direction == 1 else 0)
                v = y + (1 if direction == 2 else -1 if direction == 3 else 0)
                if 0 <= u < nx and 0 <= v < ny:
                    da = lat[start+i,u,v] - target_lat
                    db = lon[start+i,u,v] - target_lon
                    d = math.sqrt(da*da + db*db)
                    if d < best: best, bx, by = d, u, v
            if best == 0: break
            if bx == x and by == y:
                bx, by = exact_grid_match(lat[start+i], lon[start+i], target_lat, target_lon)
                break
        else:
            bx, by = exact_grid_match(lat[start+i], lon[start+i], target_lat, target_lon)
        shifts[i,0], shifts[i,1] = cx-bx, cy-by
    return shifts


def average_counts(counts, start, shifts, threshold):
    # Slice-based accumulation avoids a Python loop over individual pixels.
    nx, ny = counts.shape[1:]
    total = np.zeros((nx,ny), dtype=np.uint32)
    number = np.zeros((nx,ny), dtype=np.uint16)
    for i, (dx,dy) in enumerate(shifts):
        x0, x1 = max(0,dx), min(nx,nx+dx)
        y0, y1 = max(0,dy), min(ny,ny+dy)
        a = counts[start+i,x0-dx:x1-dx,y0-dy:y1-dy]
        valid = a > threshold
        total[x0:x1,y0:y1] += np.where(valid,a,0).astype(np.uint32)
        number[x0:x1,y0:y1] += valid
    np.floor_divide(total, number, out=total, where=number != 0)
    return total.astype(np.uint16)


def average_cloud(cloud, start, window, cpp_compat):
    # Disk order (time,box_y,box_x); y reversal cancels in this row-wise rule.
    cols = 14 if cpp_compat else cloud.shape[2]
    out = cloud[start+window-1].copy()  # C++ untouched column comes from newest
    for j in range(cols):
        total = np.zeros(cloud.shape[1], dtype=np.int64)
        number = 0
        for m in range(window):
            for n in (-1,0):
                pos = j + int((m-window//2)/4.0+n)  # truncation, NOT floor
                if 0 <= pos < cols:
                    total += cloud[start+m,:,pos]
                    number += 1
        out[:,j] = (total.astype(np.float32)/np.float32(number)).astype(np.int32)
    return out


def epoch_datetime(src, index, mode):
    value = float(read_values(src['Epoch'], slice(index,index+1), np.float64)[0])
    if not np.isfinite(value): raise ValueError('Epoch is non-finite')
    if mode == 'awe-gps':
        return datetime(2023,1,1,tzinfo=timezone.utc) + timedelta(seconds=value-1356566418)
    var = src['Epoch']
    units = getattr(var,'units',getattr(var,'UNITS',None))
    if not units: raise ValueError('CF Epoch mode requires units/UNITS')
    return num2date(value, units, calendar=getattr(var,'calendar','standard'))


def output_name(src, source, options):
    m = PATTERN.match(Path(source).name)
    date = epoch_datetime(src, options['window']//2, options['epoch_mode'])
    return (f"{m['prefix']}l1a_{m['filter'].lower()}_{date.strftime('%Y%jT%H%M')}_"
            f"{m['orbit']}_{m['version']}{m['suffix']}.nc")


def validate(src, filt, options):
    if src.groups: raise ValueError('Grouped NetCDF input is not supported')
    r = src['Radiance']
    if r.dimensions != ('time','y_cross_track','x_along_track'):
        raise ValueError(f'Expected regridded (time,y_cross_track,x_along_track), got {r.dimensions}')
    if r.shape[1] != r.shape[2] or r.shape[2] < 12:
        raise ValueError('Expected square regridded images, width >=12 (normally 300)')
    for name in IMAGES:
        if src[name].dimensions != r.dimensions or src[name].shape != r.shape:
            raise ValueError(f'{name} does not match Radiance')
    if src['Epoch'].dimensions != ('time',): raise ValueError('Epoch must be (time,)')
    count = r.shape[0]-options['window']+1-int(options['cpp_compat'])
    if count < 1: raise ValueError('Not enough frames for requested co-adding mode')
    cloud_name = None
    if filt == 'q20':
        cloud_name = next((n for n in ('MLCloud','Processed_MLCloud') if n in src.variables),None)
        if cloud_name is None: raise ValueError('Q co-adding requires MLCloud or Processed_MLCloud')
        v = src[cloud_name]
        if v.ndim != 3 or v.dimensions[0] != 'time' or v.shape[1] != 3 or v.shape[2] not in (14,15):
            raise ValueError(f'{cloud_name} must be (time,3,14 or 15); got {v.shape}')
    return count, cloud_name


def prepare_output(src, dst, name, count, cloud_name, options):
    w, half, block = options['window'], options['window']//2, options['block_size']
    # Match Files::CreateNetCDF IDs: x=0, y=1, time=2. The C++ writer
    # reuses input Header.dimids when defining output variables, so creation
    # order is required for compatibility even though NetCDF itself allows any.
    dimension_names = {'y_box_across_track': 'y_box_cross_track'}
    if cloud_name:
        cloud_dims = src[cloud_name].dimensions
        dimension_names[cloud_dims[1]] = 'y_box_cross_track'
        dimension_names[cloud_dims[2]] = 'x_box_along_track'
    primary = ('x_along_track', 'y_cross_track', 'time')
    for n in primary:
        dst.createDimension(n, None if n == 'time' else len(src.dimensions[n]))
    for n, d in src.dimensions.items():
        if n not in primary:
            target_dim = dimension_names.get(n, n)
            if target_dim in dst.dimensions:
                existing = dst.dimensions[target_dim]
                if len(existing) != len(d) or existing.isunlimited() != d.isunlimited():
                    raise ValueError(f'Conflicting dimensions mapped to {target_dim}')
            else:
                dst.createDimension(target_dim, None if d.isunlimited() else len(d))
    attrs = {n:src.getncattr(n) for n in src.ncattrs() if n != '_NCProperties'}
    attrs.update(Data_level='L1A',Data_type='L1A > Level 1A',
                 Logical_file_id=Path(name).stem,Logical_source=name[:11],
                 Number_coadded_images=np.int32(w))
    if 'Data_Level' in attrs: attrs['Data_Level'] = 'L1A'
    dst.setncatts(attrs)
    names = [n for n in ORDER if n in src.variables or (n == 'MLCloud' and cloud_name)]
    names += [n for n in src.variables if n not in names]
    for n in names:
        if cloud_name and n == 'Processed_MLCloud': continue  # converted into MLCloud
        v = src[cloud_name] if n == 'MLCloud' and cloud_name else src[n]
        image, cloud = n in IMAGES, n == 'MLCloud' and cloud_name is not None
        dtype = 'f4' if image else 'i4' if cloud else v.datatype
        kwargs = dict(zlib=True,complevel=1,shuffle=True) if v.dimensions else {}
        if image: kwargs['chunksizes'] = (1,*v.shape[1:])
        if '_FillValue' in v.ncattrs() and not cloud: kwargs['fill_value'] = v.getncattr('_FillValue')
        output_dims = tuple(dimension_names.get(d, d) for d in v.dimensions)
        out = dst.createVariable(n,dtype,output_dims,**kwargs)
        att = {a:v.getncattr(a) for a in v.ncattrs() if a != '_FillValue'}
        if image or cloud:
            for a in ('scale_factor','add_offset','_Unsigned','actual_range','valid_range'):
                att.pop(a,None)
        if cloud:
            att.update(UNITS='Percentage',VALIDMIN=np.int32(0),VALIDMAX=np.int32(100),FIELDNAM='MLCloud')
            if 'units' in att: att['units'] = 'Percentage'
            for a in ('valid_min','valid_max','missing_value'):
                att.pop(a,None)
            # C++ MLCloud variable attributes from the comparison report.
            att.update(CATDESC='MLCloud array', DEPEND_0='Time',
                       DISPLAY_TYPE='time_series', FILLVAL=np.int32(0),
                       FORMAT='I', LABLAXIS='MLCloud', NC_DEFLATE='1',
                       NC_SHUFFLE='true', NC_STORAGE='chunked', VAR_TYPE='data')
        out.setncatts(att)
        out.set_auto_maskandscale(False)
        if image or cloud: continue
        v.set_auto_maskandscale(False)
        if 'time' not in v.dimensions:
            out[...] = v[...]
        else:
            axis = v.dimensions.index('time')
            for start in range(0,count,block):
                end = min(start+block,count)
                a = [slice(None)]*v.ndim
                b = a.copy()
                a[axis], b[axis] = slice(start+half,end+half), slice(start,end)
                out[tuple(b)] = v[tuple(a)]


def process_file(task, options):
    source, folder, filt = task
    started = time.perf_counter()
    result = dict(source=source,filter=filt,status='failed',output='',frames=0,
                  read_s=0.,coadd_s=0.,write_s=0.,total_s=0.,error='',
                  cpp_compat=options['cpp_compat'],window=options['window'],
                  epoch_mode=options['epoch_mode'],numba=HAVE_NUMBA)
    temporary = None
    try:
        with Dataset(source) as src:
            count, cloud_name = validate(src,filt,options)
            target = Path(folder)/output_name(src,source,options)
            result['output'] = str(target)
            if target.exists() and not options['overwrite']:
                result['status'] = 'skipped'
                return result
            target.parent.mkdir(parents=True,exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=target.name+'.',suffix='.partial',dir=target.parent)
            os.close(fd)
            lon0 = matrix_cube(read_values(src['Longitude'],slice(0,1)).astype(np.float64))[0]
            lon1 = matrix_cube(read_values(src['Longitude'],slice(-1,None)).astype(np.float64))[0]
            invert = invert_longitudes(lon0,lon1)
            w, half, block = options['window'],options['window']//2,options['block_size']
            with Dataset(temporary,'w',format='NETCDF4') as dst:
                t = time.perf_counter()
                prepare_output(src,dst,target.name,count,cloud_name,options)
                result['write_s'] += time.perf_counter()-t
                for start in range(0,count,block):
                    end = min(start+block,count)
                    t = time.perf_counter()
                    sl = slice(start,end+w-1)
                    counts = to_counts(matrix_cube(read_values(src['Radiance'],sl)))
                    lat = matrix_cube(read_values(src['Latitude'],sl)).astype(np.float64)
                    lon = matrix_cube(read_values(src['Longitude'],sl)).astype(np.float64)
                    if invert: lon[lon < 0] += 360
                    trim_edges(counts,lat,lon)
                    cloud = None
                    if cloud_name:
                        cloud = read_values(src[cloud_name],sl)
                        cloud = cloud_percentages(cloud, cloud_name, start)
                    result['read_s'] += time.perf_counter()-t
                    t = time.perf_counter()
                    shape = (end-start,*src['Radiance'].shape[1:])
                    rad_out,lat_out,lon_out = (np.empty(shape,np.float32) for _ in range(3))
                    ml_out = np.empty((end-start,*cloud.shape[1:]),np.int32) if cloud_name else None
                    for i in range(end-start):
                        try:
                            shifts = find_shifts(lat,lon,i,w)
                        except ValueError as exc:
                            diagnostic = alignment_diagnostics(lat,lon,i,w,start)
                            raise ValueError(f'Output {start+i}, source center {start+i+half}: {exc}; {diagnostic}') from exc
                        out = average_counts(counts,i,shifts,1000 if filt in ('p12','q20') else 100)
                        a, b = lat[i+half], lon[i+half].copy()
                        if invert: b[b > 180] -= 360
                        rad = out.astype(np.float32)/np.float32(1000)
                        rad[np.isnan(a)] = np.nan
                        rad_out[i],lat_out[i],lon_out[i] = disk_plane(rad),disk_plane(a),disk_plane(b)
                        if cloud_name: ml_out[i] = average_cloud(cloud,i,w,options['cpp_compat'])
                    result['coadd_s'] += time.perf_counter()-t
                    t = time.perf_counter()
                    for n,a in zip(IMAGES,(rad_out,lat_out,lon_out)): dst[n][start:end] = a
                    if cloud_name: dst['MLCloud'][start:end] = ml_out
                    result['write_s'] += time.perf_counter()-t
                t = time.perf_counter()
            result['write_s'] += time.perf_counter()-t
        os.replace(temporary,target)
        temporary = None
        result.update(status='done',frames=count)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if temporary and Path(temporary).exists(): Path(temporary).unlink()
        result['total_s'] = time.perf_counter()-started
    return result


def discover(options):
    tasks, identities = [], set()
    for mode, root_key, level, filters in (
        ('p','l1r_input','r',('p12','p14','bkg')),
        ('q','l1c_input','c',('q20',))):
        if options['mode'] not in (mode,'both'): continue
        root = Path(options[root_key])
        if not root.is_dir(): raise ValueError(f'Input folder does not exist: {root}')
        found = 0
        for path in sorted(root.rglob('*.nc') if options['recursive'] else root.glob('*.nc')):
            m = PATTERN.match(path.name)
            if not m or m['level'].lower() != level or m['filter'].lower() not in filters: continue
            filt = m['filter'].lower()
            relative = path.relative_to(root).parent
            key = (str(relative),m['stamp'][:4],m['orbit'],filt,m['version'],m['suffix'])
            if key in identities: raise ValueError(f'Ambiguous duplicate orbit/filter/version: {path}')
            identities.add(key)
            tasks.append((str(path),str(Path(options['output'])/relative),filt))
            found += 1
        if not found: raise ValueError(f'No matching L1{level.upper()} files in {root}')
    return tasks


def warm_worker(threads):
    if HAVE_NUMBA and threads: set_num_threads(threads)
    a = np.indices((16,16)).astype(np.float64)
    lat,lon = a[0][None].copy(),a[1][None].copy()
    trim_edges(np.full((1,16,16),2000,np.uint16),lat,lon)
    find_shifts(lat,lon,0,1)
    invert_longitudes(lon[0],lon[0])


def self_test():
    import tempfile
    # Moving shared grid: each frame shifts 1 along x. Known exact translations.
    n, size, w = 23, 48, 19
    x,y = np.indices((size,size))
    lat = np.stack([x+t for t in range(n)]).astype(np.float64)
    lon = np.broadcast_to(y,(n,size,size)).astype(np.float64).copy()
    shifts = find_shifts(lat,lon,0,w)
    np.testing.assert_array_equal(shifts[:,0],np.arange(w)-w//2)
    np.testing.assert_array_equal(shifts[:,1],0)
    counts = (2000+3*lat+lon).astype(np.uint16)
    out = average_counts(counts,0,shifts,1000)
    np.testing.assert_array_equal(out[12:35,5:35],counts[w//2,12:35,5:35])
    a = np.array([1000,1001,2000],np.uint16)[:,None,None]
    assert average_counts(a,0,np.zeros((3,2),int),1000)[0,0] == 1500
    assert average_counts(a,0,np.zeros((3,2),int),2000)[0,0] == 0
    broken = lat[:w].copy(); broken[0] += .125
    try: find_shifts(broken,lon,0,w)
    except ValueError: pass
    else: raise AssertionError('Nonmatching grid must fail instead of hanging')
    obstructed_lat, obstructed_lon = lat[:w].copy(), lon[:w].copy()
    tx, ty = size//2+w//2, size//2
    for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
        obstructed_lat[0,tx+dx,ty+dy] = np.nan
        obstructed_lon[0,tx+dx,ty+dy] = np.nan
    np.testing.assert_array_equal(find_shifts(obstructed_lat,obstructed_lon,0,w),shifts)
    diagnostic = alignment_diagnostics(broken,lon,0,w,100)
    assert 'source frame 100: exact_matches=0' in diagnostic
    assert 'delta lat/lon=' in diagnostic
    duplicate_lat, duplicate_lon = obstructed_lat.copy(), obstructed_lon.copy()
    duplicate_lat[0,0,0], duplicate_lon[0,0,0] = lat[w//2,size//2,size//2], lon[w//2,size//2,size//2]
    try: find_shifts(duplicate_lat,duplicate_lon,0,w)
    except ValueError as exc: assert 'Ambiguous' in str(exc)
    else: raise AssertionError('Ambiguous fallback must fail')
    # Check all-15 and literal-14 cloud paths against direct independent loops.
    cloud = (np.arange(n*3*15).reshape(n,3,15)%101).astype(np.int32)
    for compat in (False,True):
        actual = average_cloud(cloud,0,w,compat)
        for j in range(14 if compat else 15):
            for k in range(3):
                vals = [cloud[m,k,j+math.trunc((m-9)/4+offset)]
                        for m in range(w) for offset in (-1,0)
                        if 0 <= j+math.trunc((m-9)/4+offset) < (14 if compat else 15)]
                assert actual[k,j] == int(np.float32(sum(vals))/np.float32(len(vals)))
        if compat: np.testing.assert_array_equal(actual[:,14],cloud[w-1,:,14])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        options = dict(window=w,cpp_compat=False,epoch_mode='awe-gps',block_size=2,overwrite=False)
        for filt in ('p12','p14','bkg','q20'):
            level = 'c' if filt == 'q20' else 'r'
            source = root/f'awe_l1{level}_{filt}_2023365T2359_00001_v02.nc'
            with Dataset(source,'w') as ds:
                for d,s in [('time',n),('y_cross_track',size),('x_along_track',size),('box_y',3),('box_x',15)]: ds.createDimension(d,s)
                ds.createVariable('Epoch','f8',('time',))[:] = 1356566418+365*86400-5+np.arange(n)
                ds.createVariable('ISS_Latitude','f4',('time',))[:] = np.arange(n)
                ds.setncatts(dict(Data_level='L1'+level.upper(),Data_Level='L1'+level.upper(),test_attribute='keep'))
                for name,data in [('Radiance',counts.astype(np.float32)/1000),('Latitude',lat),('Longitude',lon)]:
                    v = ds.createVariable(name,'f4',('time','y_cross_track','x_along_track'))
                    v[:] = np.stack([disk_plane(a) for a in data])
                if filt == 'q20': ds.createVariable('MLCloud','i4',('time','box_y','box_x'))[:] = cloud
            for compat in (False,True):
                opts = dict(options,cpp_compat=compat)
                r = process_file((str(source),str(root/str(compat)),filt),opts)
                assert r['status'] == 'done',r
                with Dataset(r['output']) as ds:
                    expected_n = n-w+1-int(compat)
                    assert len(ds.dimensions['time']) == expected_n and ds.dimensions['time'].isunlimited()
                    assert '2024001T0000' in Path(r['output']).name
                    np.testing.assert_array_equal(ds['ISS_Latitude'][:],np.arange(9,9+expected_n))
                    assert ds.Data_level == ds.Data_Level == 'L1A' and ds.test_attribute == 'keep'
                    assert ds.Number_coadded_images == 19
                    assert list(ds.dimensions)[:3] == ['x_along_track','y_cross_track','time']
                    assert [ds.dimensions[d]._dimid for d in ds['Radiance'].dimensions] == [2,1,0]
                    if filt == 'q20':
                        assert ds['MLCloud'].dimensions == ('time','y_box_cross_track','x_box_along_track')
                        assert ds['MLCloud'].CATDESC == 'MLCloud array'
                        assert ds['MLCloud'].DEPEND_0 == 'Time'
                        assert ds['MLCloud'].FILLVAL == 0
                        np.testing.assert_array_equal(ds['MLCloud'][0],average_cloud(cloud,0,w,compat))
                    baseline = np.asarray(ds['Radiance'][:])
                # A different block size must not change any radiance pixel.
                r2 = process_file((str(source),str(root/(str(compat)+'_block')),filt),dict(opts,block_size=7))
                assert r2['status'] == 'done',r2
                with Dataset(r2['output']) as ds: np.testing.assert_array_equal(baseline,np.asarray(ds['Radiance'][:]))
                assert process_file((str(source),str(root/str(compat)),filt),opts)['status'] == 'skipped'
    print('Self-test passed: shifts, exact-search fallback, ambiguous/missing matches, diagnostics, thresholds, empty pixels, search failure, cloud rules, all four filters, frame counts, midnight/year rollover, metadata, block invariance, skip behavior.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag,default in [('l1r-input',DEFAULT_L1R_INPUT),('l1c-input',DEFAULT_L1C_INPUT),('output',DEFAULT_OUTPUT)]: p.add_argument('--'+flag,default=default)
    p.add_argument('--mode',choices=['p','q','both'],default=DEFAULT_MODE)
    p.add_argument('--workers',type=int,default=DEFAULT_WORKERS)
    p.add_argument('--numba-threads',type=int,default=DEFAULT_NUMBA_THREADS)
    p.add_argument('--block-size',type=int,default=DEFAULT_BLOCK_SIZE)
    p.add_argument('--window',type=int,default=DEFAULT_NB_COADD)
    p.add_argument('--epoch-mode',choices=['awe-gps','cf'],default=DEFAULT_EPOCH_MODE)
    p.add_argument('--cpp-compat',action=argparse.BooleanOptionalAction,default=DEFAULT_CPP_COMPAT)
    p.add_argument('--recursive',action=argparse.BooleanOptionalAction,default=DEFAULT_RECURSIVE)
    p.add_argument('--overwrite',action=argparse.BooleanOptionalAction,default=DEFAULT_OVERWRITE)
    p.add_argument('--self-test',action='store_true')
    args = p.parse_args(argv)
    if args.self_test: self_test(); return
    if args.window < 3 or args.window > 29 or args.window%2 != 1: p.error('window must be odd, between 3 and 29')
    if args.workers < 1 or args.block_size < 1 or args.numba_threads < 0: p.error('workers/block-size must be positive; numba-threads nonnegative')
    options = vars(args)
    tasks = discover(options)
    print(f'{len(tasks)} files; {args.workers} workers; Numba={HAVE_NUMBA}; window={args.window}; block={args.block_size}; cpp_compat={args.cpp_compat}',flush=True)
    started = time.perf_counter()
    results = []
    def report(r):
        results.append(r)
        print(f"{r['status'].upper()} {Path(r['output'] or r['source']).name}: {r['frames']} frames; read {r['read_s']:.1f}s, coadd {r['coadd_s']:.1f}s, write/close {r['write_s']:.1f}s, total {r['total_s']:.1f}s {r['error']}",flush=True)
    if args.workers == 1:
        warm_worker(args.numba_threads)
        for task in tasks: report(process_file(task,options))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),initializer=warm_worker,initargs=(args.numba_threads,)) as pool:
            futures = {pool.submit(process_file,t,options):t for t in tasks}
            for future in as_completed(futures): report(future.result())
    directory = Path(args.output); directory.mkdir(parents=True,exist_ok=True)
    log = directory/f"coadd_timings_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%f}.csv"
    with log.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print(f'Wall time: {time.perf_counter()-started:.1f}s; timing report: {log}')
    if any(r['status'] == 'failed' for r in results): raise SystemExit(1)


if __name__ == '__main__':
    main()