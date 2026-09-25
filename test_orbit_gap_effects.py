#!/usr/bin/env python3
r"""Controlled gap experiment: November 2023, orbit 00001, pipeline steps 1--6.

Put beside run_awe_pipeline_orbit.py and its processing scripts. Edit SETTINGS
and run in Spyder. Dependencies: numpy, scipy, netCDF4, matplotlib, and the existing pipeline/ML
models. The runner's SCRIPTS, Python interpreters, RESOLUTIONS and EXTRA_ARGS
are reused, so all experiments have the same processing configuration.

    python test_orbit_gap_effects.py --report-only E:\soc\gap_experiments\orbit00001_RUN
    python test_orbit_gap_effects.py --plan-only
    python test_orbit_gap_effects.py --cases baseline single_20s_middle
    python test_orbit_gap_effects.py

The default suite has 38 cases: baseline; one missing frame at 20/50/80 percent
of the orbit; 1 gap at 5/10/20/50 seconds at those same positions;
2/5/10/20 evenly spaced one-missing-frame gaps;
2/5/10/20 evenly spaced gaps at 5/20/50 seconds;
and 10 gaps of 20 seconds clustered early/middle/late or randomly placed with
three fixed seeds. All positions refer to the ORIGINAL observation span.
Targets are quantized DOWN to available frame times. dt means the difference
between retained bounding Epoch values, NOT the duration of removed samples.
At most 20 artificial gaps per case; each realized dt <= requested dt <= 50 s
in ALL four filters. The same frame indices are removed from all four files.
Existing gaps remain part of the baseline and are reported separately.
At the observed ~1.1-second cadence, one missing frame adds ~1.1 seconds
to the normal interval: the measured bounding-frame dt is ~2.2 seconds.
One-frame cases remove exactly one frame per gap (rather than targeting a
1.1-second bounding dt). Their dt field is the 50-second safety cap; see the
actual per-filter dt_s in plan.json and max_realized_dt_s in case_plan.csv.

Every invocation gets a new experiment directory: source files and existing
production outputs are never modified. Cases run sequentially, each from its
own L0C copy; all intermediate products and logs are retained for inspection.
Failures stop that case, not the remaining cases. Baseline failure stops the
experiment. Infeasible cases are reported, never silently shortened.

Outputs:
  case_reports.pdf           one page per case, L3C radiance fixed at 0-20 kR
  # Lower panel: radiance minus baseline, latitude/longitude differences at the same stored indices, fixed +/-0.1 deg.
  report_issues.json          missing/unreadable plots, including failed cases
  plan.json                  exact source paths, configuration, every gap
  case_plan.csv              case design, realized dt, frame loss
  baseline/, single_.../      L0C through L3C plus logs/status.json
  experiment_status.json     completion/failure status for each case
  product_metrics.csv        coverage, Epoch and Pixel_Epoch statistics
  metrics_vs_baseline.csv    aggregate differences from unchanged-input run

Summary comparisons are NOT pixelwise errors: footprints and frame selections
can change. L3 Epoch has one element; use Pixel_Epoch's distinct source times
and coverage instead. Zero-valued pixels are reported separately from nonfinite
pixels because several stages use zero as a sentinel. Metadata VALIDMIN/MAX
are NOT used to mask values (AWE temperature metadata can have scaled bounds).
No interpolation or gap-aware changes to the science pipeline are introduced.
Use matched coordinates/times for a later quantitative pixelwise comparison.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from netCDF4 import Dataset

# ------------------------- SETTINGS -------------------------
SOURCE_FOLDER = Path(r'E:\soc\l0c\2023\11')  # Use the same prepared L0C as normal step 1.
OUTPUT_ROOT = Path(r'E:\soc\gap_experiments')
RUNNER = Path(__file__).resolve().parent / 'run_awe_pipeline_orbit.py'
ORBIT = '00001'
WORKERS = 4
PLAN_ONLY = False
SELECT_CASES = None  # E.g. ['baseline', 'single_1frame_middle']; None = all 38.
BLOCK_FRAMES = 32
COORDINATE_LIMIT_DEG = 0.1  # Fixed color bars for both coordinate differences: +/-0.1 degree.
# ------------------------------------------------------------
FILTERS = ('q20', 'p12', 'p14', 'bkg')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_runner(path):
    spec = importlib.util.spec_from_file_location('_gap_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unit_scale(var):
    units = getattr(var, 'units', getattr(var, 'UNITS', 'seconds'))
    if isinstance(units, bytes):
        units = units.decode()
    token = str(units).lower().split(' since ')[0].strip()
    scales = {'s': 1., 'sec': 1., 'second': 1., 'seconds': 1.,
              'ms': .001, 'millisecond': .001, 'milliseconds': .001,
              'us': 1e-6, 'microsecond': 1e-6, 'microseconds': 1e-6,
              'min': 60., 'minute': 60., 'minutes': 60.,
              'h': 3600., 'hour': 3600., 'hours': 3600.,
              'd': 86400., 'day': 86400., 'days': 86400.}
    if token not in scales:
        raise ValueError(f'Unsupported Epoch units: {units!r}')
    return scales[token]


def values(var, key=Ellipsis):
    # Honor packing, but ignore incorrect VALIDMIN/MAX/valid_range attributes.
    var.set_auto_mask(False)
    var.set_auto_scale(True)
    raw = np.asarray(var[key], dtype=np.float64)
    valid = np.isfinite(raw)
    for name in ('_FillValue', 'missing_value'):
        if name in var.ncattrs():
            fill = np.asarray(var.getncattr(name), dtype=float)
            fill = fill * float(getattr(var, 'scale_factor', 1)) + float(getattr(var, 'add_offset', 0))
            for sentinel in fill.ravel():
                valid &= raw != sentinel
    return np.where(valid, raw, np.nan)


def read_epochs(group):
    epochs, dimensions, metadata = {}, {}, {}
    for filt, path in group.items():
        with Dataset(path) as ds:
            if ds.groups:
                raise ValueError(f'Grouped NetCDF not supported: {path}')
            v = ds['Epoch']
            if v.ndim != 1:
                raise ValueError(f'Epoch must have exactly one frame dimension: {path}')
            e = values(v) * unit_scale(v)
            if len(e) < 10 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0):
                raise ValueError(f'Epoch must be finite, strictly increasing, and have >=10 frames: {path}')
            epochs[filt] = e
            dimensions[filt] = v.dimensions[0]
            cadence = float(np.median(np.diff(e)))
            metadata[filt] = {'path': str(path), 'frames': len(e), 'cadence_s': cadence,
                              'existing_gaps': int(np.sum(np.diff(e) > max(1.5*cadence, cadence+.05))),
                              'max_existing_dt_s': float(np.max(np.diff(e))),
                              'epoch_units': str(getattr(v, 'units', getattr(v, 'UNITS', 'seconds (assumed)')))}
    ref = epochs['q20']
    for filt, e in epochs.items():
        # Constant exposure offsets are allowed; drifting/missing frame alignment is not.
        if len(e) != len(ref) or not np.allclose(e-e[0], ref-ref[0], rtol=0, atol=.05):
            raise ValueError(f'{filt}: frames do not align with q20 within 0.05 s after a constant offset; '
                             'cannot safely apply the same frame-index gaps.')
    return epochs, dimensions, metadata


def designs():
    result = [dict(name='baseline', count=0, dt=0., pattern='none', seed=0)]
    for label, position in (('early', .2), ('middle', .5), ('late', .8)):
        result.append(dict(name=f'single_1frame_{label}', count=1, dt=50.,
                           remove_per_gap=1, pattern='single', position=position, seed=0))
    for dt in (5, 10, 20, 50):
        for label, position in (('early', .2), ('middle', .5), ('late', .8)):
            result.append(dict(name=f'single_{dt}s_{label}', count=1, dt=float(dt),
                               pattern='single', position=position, seed=0))
    for count in (2, 5, 10, 20):
        result.append(dict(name=f'even_{count}x1frame', count=count, dt=50.,
                           remove_per_gap=1, pattern='even', seed=0))
        for dt in (5, 20, 50):
            result.append(dict(name=f'even_{count}x{dt}s', count=count, dt=float(dt), pattern='even', seed=0))
    for label, position in (('early', .2), ('middle', .5), ('late', .8)):
        result.append(dict(name=f'cluster_10x20s_{label}', count=10, dt=20.,
                           pattern='cluster', position=position, seed=0))
    for seed in (101, 202, 303):
        result.append(dict(name=f'random_10x20s_seed{seed}', count=10, dt=20., pattern='random', seed=seed))
    return result


def make_gaps(case, epochs):
    """Choose retained bounding indices (a,b); remove only a+1 through b-1."""
    if not case['count']:
        return []
    if not 1 <= case['count'] <= 20 or not 0 < case['dt'] <= 50:
        raise ValueError('Gap limits exceeded')
    t = epochs['q20'] - epochs['q20'][0]
    cadence = float(np.median(np.diff(t)))
    # Do not absorb a pre-existing gap into an injected gap.
    bad = np.zeros(len(t)-1, dtype=bool)
    for e in epochs.values():
        c = float(np.median(np.diff(e)))
        bad |= np.diff(e) > max(1.5*c, c+.05)
    prefix = np.r_[0, np.cumsum(bad)]
    candidates = []
    for a in range(1, len(t)-2):
        if 'remove_per_gap' in case:
            b = a + case['remove_per_gap'] + 1
            if b > len(t)-2 or any(e[b]-e[a] > case['dt']+1e-7 for e in epochs.values()):
                continue
        else:
            b = min(int(np.searchsorted(e, e[a]+case['dt']+1e-7, side='right')-1)
                    for e in epochs.values())
            b = min(b, len(t)-2)  # Keep source endpoints and a margin unchanged.
        if b >= a+2 and prefix[b] == prefix[a]:
            candidates.append((a, b, (t[a]+t[b])/2))
    if case['pattern'] == 'single':
        centers = [case['position']*t[-1]]
    elif case['pattern'] == 'cluster':
        spacing = case['dt'] + 5*cadence
        centers = case['position']*t[-1] + (np.arange(case['count'])-(case['count']-1)/2)*spacing
    elif case['pattern'] == 'random':
        centers = None
    else:
        centers = np.linspace(0, t[-1], case['count']+2)[1:-1]
    chosen = []
    rng = np.random.default_rng(case['seed'])
    for i in range(case['count']):
        available = [p for p in candidates if all(p[1]+3 < a or b+3 < p[0] for a,b,_ in chosen)]
        if not available:
            raise ValueError('Cannot fit the requested number/duration of separated gaps in this orbit')
        if centers is None:
            pick = available[int(rng.integers(len(available)))]
        else:
            pick = min(available, key=lambda p: abs(p[2]-centers[i]))
            if abs(pick[2]-centers[i]) > max(2*cadence, .02*t[-1]):
                raise ValueError('Requested gap position is unavailable; refusing to relocate the pattern')
        chosen.append(pick)
    result = []
    for a,b,center in sorted(chosen):
        dt = {f: float(e[b]-e[a]) for f,e in epochs.items()}
        if max(dt.values()) > case['dt']+1e-6:
            raise ValueError('Realized gap exceeds requested dt')
        result.append(dict(index_before=int(a), index_after=int(b), removed_frames=int(b-a-1),
                           center_fraction=float(center/t[-1]), dt_s=dt,
                           before_epoch_seconds={f: float(e[a]) for f,e in epochs.items()},
                           after_epoch_seconds={f: float(e[b]) for f,e in epochs.items()}))
    return result


def keep_indices(n, gaps):
    keep = np.ones(n, dtype=bool)
    for gap in gaps:
        keep[gap['index_before']+1:gap['index_after']] = False
    return np.flatnonzero(keep)


def clone_subset(source, target, frame_dim, keep, case_name):
    """Raw-storage copy; subset EVERY variable using the Epoch dimension."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Dataset(source) as src, Dataset(target, 'w', format=src.data_model) as dst:
            dst.setncatts({k:src.getncattr(k) for k in src.ncattrs() if k != '_NCProperties'})
            dst.setncattr('Gap_experiment_case', case_name)
            for name, dim in src.dimensions.items():
                dst.createDimension(name, None if dim.isunlimited() else (len(keep) if name == frame_dim else len(dim)))
            for name, v in src.variables.items():
                options = {}
                if '_FillValue' in v.ncattrs():
                    options['fill_value'] = v.getncattr('_FillValue')
                if src.data_model.startswith('NETCDF4') and v.ndim and v.dtype != str:
                    options.update(zlib=True, complevel=1)
                out = dst.createVariable(name, v.datatype, v.dimensions, **options)
                out.setncatts({k:v.getncattr(k) for k in v.ncattrs() if k != '_FillValue'})
                v.set_auto_maskandscale(False)
                out.set_auto_maskandscale(False)
                if hasattr(v, 'set_auto_chartostring'):
                    v.set_auto_chartostring(False)
                    out.set_auto_chartostring(False)
                if frame_dim not in v.dimensions:
                    out[...] = v[...]
                    continue
                if v.dimensions.count(frame_dim) != 1:
                    raise ValueError(f'Repeated frame dimension in {name}')
                axis = v.dimensions.index(frame_dim)
                for start in range(0, len(keep), BLOCK_FRAMES):
                    indices = keep[start:start+BLOCK_FRAMES]
                    source_key = [slice(None)]*v.ndim
                    target_key = source_key.copy()
                    source_key[axis] = indices
                    target_key[axis] = slice(start, start+len(indices))
                    out[tuple(target_key)] = v[tuple(source_key)]
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def sequence_stats(e, reference_cadence):
    e = np.asarray(e).ravel()
    e = e[np.isfinite(e)]
    dt = np.diff(e)
    return {'samples': int(len(e)), 'span_s': float(e[-1]-e[0]) if len(e) else 0.,
            'gaps': int(np.sum(dt > max(1.5*reference_cadence, reference_cadence+.05))),
            'max_dt_s': float(np.max(dt)) if len(dt) else 0.,
            'nonincreasing': int(np.sum(dt <= 0))}


def product_metrics(case, paths, cadence):
    rows = []
    for level, folder in paths.items():
        for path in sorted(folder.rglob('*.nc')):
            row = {'case': case, 'level': level, 'relative_folder': str(path.parent.relative_to(folder)),
                   'filter': path.name.split('_')[2], 'file': str(path)}
            try:
                with Dataset(path) as ds:
                    if 'Epoch' in ds.variables:
                        e = values(ds['Epoch'])*unit_scale(ds['Epoch'])
                        row.update({'epoch_'+k:v for k,v in sequence_stats(e, cadence).items()})
                    for name in ('Radiance', 'Temperature'):
                        if name not in ds.variables:
                            continue
                        v = ds[name]
                        row.update(variable=name, shape=str(v.shape), units=str(getattr(v,'units','')))
                        total=valid=nonzero=0
                        summ=squares=0.
                        # Streaming statistics avoid loading an orbit's full image cube.
                        for start in range(0, v.shape[0] if v.ndim else 1, BLOCK_FRAMES):
                            a = values(v, slice(start,start+BLOCK_FRAMES) if v.ndim else Ellipsis)
                            finite = a[np.isfinite(a)]
                            good = finite[finite != 0]
                            total += a.size; valid += finite.size; nonzero += good.size
                            summ += float(np.sum(good)); squares += float(np.sum(good*good))
                        mean = summ/nonzero if nonzero else None
                        row.update(total_values=total, finite_values=valid, nonzero_finite_values=nonzero,
                                   finite_fraction=valid/total if total else 0.,
                                   nonzero_finite_fraction=nonzero/total if total else 0.,
                                   mean_nonzero=mean,
                                   std_nonzero=float(np.sqrt(max(0.,squares/nonzero-mean*mean))) if nonzero else None)
                        break
                    if 'Pixel_Epoch' in ds.variables:
                        v = ds['Pixel_Epoch']; a = values(v)*unit_scale(v)
                        good = a[np.isfinite(a)]
                        row['pixel_epoch_valid_fraction'] = good.size/a.size if a.size else 0.
                        row.update({'pixel_source_'+k:v for k,v in sequence_stats(np.unique(good),cadence).items()})
                        row['pixel_time_note'] = 'Distinct source times; counts are not frame counts or spatial hole counts'
            except Exception as exc:
                row['metrics_error'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
    return rows


def comparison_rows(rows):
    def key(r):
        return r['level'], r['relative_folder'], r['filter']
    base = {key(r):r for r in rows if r['case']=='baseline'}
    result = []
    for row in rows:
        if row['case']=='baseline' or key(row) not in base:
            continue
        out = {k:row[k] for k in ('case','level','relative_folder','filter')}
        for name in ('epoch_samples','epoch_gaps','epoch_max_dt_s','finite_fraction',
                     'nonzero_finite_values','nonzero_finite_fraction','mean_nonzero','std_nonzero',
                     'pixel_epoch_valid_fraction','pixel_source_samples','pixel_source_max_dt_s'):
            a,b = row.get(name),base[key(row)].get(name)
            if isinstance(a,(int,float)) and isinstance(b,(int,float)):
                out['delta_'+name] = a-b
        result.append(out)
    return result


def swath_plane(var):
    """Decode one stored swath as (cross-track, along-track), without squeezing spatial axes."""
    key = tuple(0 if d == 'time' else slice(None) for d in var.dimensions)
    plane = values(var, key)
    dims = [d for d in var.dimensions if d != 'time']
    if plane.ndim != 2:
        raise ValueError(f'{var.name}: expected one 2-D swath, got {plane.shape}')
    if len(dims) == 2 and 'along' in dims[0] and 'cross' in dims[1]:
        plane = plane.T
    elif not (len(dims) == 2 and 'cross' in dims[0] and 'along' in dims[1]):
        raise ValueError(f'{var.name}: unsupported spatial dimensions {dims}')
    return plane



def coordinate_differences(lat, lon, base_lat, base_lon):
    """Case minus baseline at identical stored (row, column) indices.

    This diagnoses grid geometry, NOT geolocation error at matched observations.
    For unequal shapes, compare only their top-left index overlap. Coordinates
    outside that overlap, or invalid in either file, remain missing. No nearest
    neighbor matching or interpolation is used, because it would hide shifts.
    """
    if lat.ndim != 2 or lat.shape != lon.shape or base_lat.ndim != 2 or base_lat.shape != base_lon.shape:
        raise ValueError('Latitude/longitude must be matching 2-D arrays within each file')
    ny,nx = min(lat.shape[0],base_lat.shape[0]),min(lat.shape[1],base_lat.shape[1])
    a,b,c,d = lat[:ny,:nx],lon[:ny,:nx],base_lat[:ny,:nx],base_lon[:ny,:nx]
    valid = np.isfinite(a)&np.isfinite(b)&np.isfinite(c)&np.isfinite(d)&(np.abs(a)<=90)&(np.abs(c)<=90)
    dlat,dlon = np.full(lat.shape,np.nan),np.full(lat.shape,np.nan)
    dlat[:ny,:nx] = np.where(valid,a-c,np.nan)
    dlon[:ny,:nx] = np.where(valid,(b-d+180)%360-180,np.nan)
    return dlat,dlon,(ny,nx)


def create_case_report(experiment):
    """Regenerate one PDF from saved inputs/outputs; never run science stages."""
    import textwrap
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import colormaps
    from matplotlib.patches import Rectangle

    experiment = Path(experiment).resolve()
    plan = json.loads((experiment/'plan.json').read_text(encoding='utf-8'))
    status_file = experiment/'experiment_status.json'
    states = {s['case']:s for s in json.loads(status_file.read_text(encoding='utf-8'))} if status_file.exists() else {}
    summary = {}
    metrics_file = experiment/'product_metrics.csv'
    if metrics_file.exists():
        with metrics_file.open(newline='',encoding='utf-8') as stream:
            summary = {r['case']:r for r in csv.DictReader(stream) if r['level']=='l3c'}
    target = experiment/'case_reports.pdf'
    temporary = experiment/'case_reports.partial.pdf'
    issues = []
    def fmt(value):
        try: return f'{float(value):.4g}'
        except (ValueError,TypeError): return 'n/a'
    try:
        with PdfPages(temporary) as pdf:
            pdf.infodict()['Title'] = f"AWE orbit {plan['orbit']} - controlled gap experiments"
            for page,case in enumerate(plan['cases'],1):
                name = case['name']; state=states.get(name,{})
                status=state.get('status',case.get('status','not run'))
                home=experiment/name
                fig=Figure(figsize=(12,16),facecolor='white')
                fig.add_artist(Rectangle((0,.934),1,.066,transform=fig.transFigure,color='#19394b'))
                fig.text(.035,.966,f"Orbit {plan['orbit']} | November 2023 | {name}",
                         color='white',fontsize=16,va='center')
                gaps=case.get('gaps',[])
                largest=max((max(g['dt_s'].values()) for g in gaps),default=0.)
                nremoved=sum(g['removed_frames'] for g in gaps)
                source=plan['source']['q20']; nsource=source['frames']
                fig.text(.035,.911,f"Status: {status.upper()}   |   Injected gaps: {len(gaps)}   |   "
                         f"Largest dt: {largest:.6f} s   |   Removed: {nremoved}/{nsource} frames",fontsize=10)
                fig.text(.035,.894,'Same removed frame indices in q20, p12, p14 and bkg. '
                         f"Source cadence: {source['cadence_s']:.6f} s; source gaps: {source['existing_gaps']}.",fontsize=9)
                step_levels=[('L0D',1),('L1F',2),('L1R',3),('L1C',4),('L1A',5),('L2A',6),('L3A',6),('L3C',6)]
                passed={s['step'] for s in state.get('steps',[])}
                for i,(level,step) in enumerate(step_levels):
                    folder=home/level.lower()/'2023'/'11'
                    files=list(folder.rglob('*.nc')) if folder.exists() else []
                    label='VALIDATED' if step in passed else ('UNVALIDATED' if files else 'NO FILES')
                    color='#168053' if step in passed else '#ad4d30'
                    fig.text(.035+(i%4)*.24,.868-(i//4)*.019,
                             f'{level}: {label} ({len(files)} files)',fontsize=9,color=color)
                note=state.get('error',case.get('error',''))
                if note:
                    wrapped=textwrap.wrap(note,145)
                    fig.text(.035,.820,'\n'.join(wrapped[:2])+(' ... (see logs)' if len(wrapped)>2 else ''),
                             fontsize=8,color='#a53b2a',va='top')
                else:
                    fig.text(.035,.820,'VALIDATED = the pipeline output checks passed. File presence alone is not validation.',fontsize=8,color='#526575')
                fig.text(.035,.787,'INJECTED GAP DETAILS - indices refer to the original L0C frames (zero-based)',fontsize=10)
                if gaps:
                    rows=[]
                    for i,g in enumerate(gaps,1):
                        a,b=g['index_before'],g['index_after']
                        ds=list(g['dt_s'].values()); qdt=g['dt_s']['q20']
                        rows.append([i,f'{a} -> {b}',f'{a+1}..{b-1}',g['removed_frames'],
                                     f'{qdt:.6f}',f'{qdt-source["cadence_s"]:.6f}',
                                     f'{min(ds):.6f}..{max(ds):.6f}'])
                    table_height=min(.305,.026*(len(rows)+1))
                    ax=fig.add_axes([.035,.767-table_height,.93,table_height]); ax.axis('off')
                    table=ax.table(cellText=rows,colLabels=['#','Retained bounds','Removed indices','Frames',
                                 'q20 dt (s)','Extra dt (s)','All-filter dt range (s)'],
                                 colWidths=[.035,.15,.15,.07,.16,.16,.275],cellLoc='center',bbox=[0,0,1,1])
                    table.auto_set_font_size(False); table.set_fontsize(8)
                    for (r,c),cell in table.get_celld().items():
                        cell.set_edgecolor('#dde4e8'); cell.set_linewidth(.4)
                        cell.set_facecolor('#e5edf2' if r==0 else ('#f5f8fa' if r%2==0 else 'white'))
                else:
                    fig.text(.035,.758,'No artificial gaps.' if status!='infeasible' else 'Case could not be constructed; see error above.',fontsize=10)
                metric=summary.get(name,{})
                base=summary.get('baseline',{})
                delta='n/a'
                try: delta=f"{float(metric['mean_nonzero'])-float(base['mean_nonzero']):+.4g}"
                except (KeyError,ValueError,TypeError): pass
                fig.text(.035,.441,'L3C: nonzero finite pixels = '+metric.get('nonzero_finite_values','n/a')+
                         '; mean radiance = '+fmt(metric.get('mean_nonzero'))+' kR; mean difference from baseline = '+delta+' kR.',fontsize=8)
                fig.text(.035,.425,'Means above use each case\'s coverage. Lower plots: case minus baseline at identical stored indices; geometry diagnostic, not geolocation error.',fontsize=8,color='#526575')
                files=sorted((home/'l3c'/'2023'/'11').glob('*_l3c_q20_*.nc'))
                axes=[fig.add_axes([.075,.255,.81,.137]),
                      fig.add_axes([.075,.085,.345,.115]),fig.add_axes([.565,.085,.32,.115])]
                axes[0].set_title('L3C Q-line radiance | fixed 0-20 kR',loc='left',fontsize=11)
                axes[1].set_title('Latitude: case - baseline',loc='left',fontsize=10)
                axes[2].set_title('Longitude: case - baseline',loc='left',fontsize=10)
                try:
                    if len(files)!=1:
                        raise ValueError(f'Expected one L3C q20 file; found {len(files)}. See stage logs.')
                    with Dataset(files[0]) as ds:
                        rad=swath_plane(ds['Radiance'])
                        invalid=~np.isfinite(rad)|(rad==0)
                        for coord in ('Latitude','Longitude'):
                            if coord in ds.variables:
                                invalid |= ~np.isfinite(swath_plane(ds[coord]))
                        cmap=colormaps['gray'].copy(); cmap.set_bad('#fff8ea')
                        # L3C is a 2 km grid in the supplied generator. Distances
                        # below are offsets from the stored master-grid origin.
                        ny,nx=rad.shape; extent=(0,nx*2,0,ny*2)
                        im=axes[0].imshow(np.ma.array(rad,mask=invalid),origin='lower',aspect='auto',
                                          extent=extent,cmap=cmap,vmin=0,vmax=20,interpolation='nearest')
                        cax=fig.add_axes([.905,.255,.015,.137])
                        fig.colorbar(im,cax=cax,ticks=np.arange(0,21,2),label='kR',extend='max')
                        baseline_files = sorted((experiment/'baseline'/'l3c'/'2023'/'11').glob('*_l3c_q20_*.nc'))
                        try:
                            if len(baseline_files) != 1:
                                raise ValueError(f'Expected one baseline L3C file; found {len(baseline_files)}')
                            with Dataset(baseline_files[0]) as baseline:
                                base_lat,base_lon = swath_plane(baseline['Latitude']),swath_plane(baseline['Longitude'])
                            lat,lon = swath_plane(ds['Latitude']),swath_plane(ds['Longitude'])
                            dlat,dlon,overlap = coordinate_differences(lat,lon,base_lat,base_lon)
                            diff_cmap = colormaps['RdBu_r'].copy()
                            diff_cmap.set_bad('#fff8ea')
                            for ax,difference,label,bar_x in zip(axes[1:],(dlat,dlon),('Latitude','Longitude'),(.435,.905)):
                                common = difference[np.isfinite(difference)]
                                if not common.size:
                                    ax.text(.5,.5,'No valid overlapping coordinate indices',ha='center',va='center',transform=ax.transAxes,fontsize=8)
                                    continue
                                im = ax.imshow(np.ma.masked_invalid(difference),origin='lower',aspect='auto',
                                               extent=extent,cmap=diff_cmap,vmin=-COORDINATE_LIMIT_DEG,
                                               vmax=COORDINATE_LIMIT_DEG,interpolation='nearest')
                                cax = fig.add_axes([bar_x,.085,.012,.115])
                                fig.colorbar(im,cax=cax,label='degrees',extend='both')
                                ax.set_title(f'{label}: case - baseline\n'
                                             f'mean={np.mean(common):+.4g}; max |diff|={np.max(np.abs(common)):.4g} deg',
                                             loc='left',fontsize=8)
                            shape_note = (f'Case shape {lat.shape}; baseline {base_lat.shape}; '
                                          f'index overlap {overlap}. Longitude differences wrapped to [-180, 180).')
                            fig.text(.035,.049,shape_note,fontsize=8,color='#526575')
                        except Exception as exc:
                            message = f'Coordinate difference unavailable: {exc}'
                            issues.append({'case':name,'plot_issue':message})
                            for ax in axes[1:]:
                                ax.text(.5,.5,'\n'.join(textwrap.wrap(message,45)),ha='center',
                                        va='center',transform=ax.transAxes,fontsize=8)
                        for ax in axes:
                            ax.set_xlabel('Along-track grid offset (km)',fontsize=9)
                            ax.set_ylabel('Cross-track offset (km)',fontsize=9)
                            ax.tick_params(labelsize=8)
                    fig.text(.035,.224,files[0].name,fontsize=7,color='#526575')
                except Exception as exc:
                    message=f'{type(exc).__name__}: {exc}'
                    issues.append({'case':name,'plot_issue':message})
                    for ax in axes:
                        if not ax.images:
                            ax.set_axis_off()
                            ax.text(.5,.5,'\n'.join(textwrap.wrap(message,95)),ha='center',va='center',transform=ax.transAxes,fontsize=10)
                fig.text(.035,.032,'dt = full bounding interval; extra dt = dt minus cadence. Cream = missing radiance or invalid/non-overlapping coordinates. Both coordinate color bars: +/-0.1 degree.',fontsize=8,color='#526575')
                fig.text(.035,.017,f"Case {page}/{len(plan['cases'])} | Controlled gap experiment | No UTC conversion of source Epoch",fontsize=8,color='#526575')
                pdf.savefig(fig,dpi=160)
                fig.clear()
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(experiment/'report_issues.json',issues)
    print(f'Case report: {target}',flush=True)
    return target


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-folder', type=Path, default=SOURCE_FOLDER)
    p.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    p.add_argument('--runner', type=Path, default=RUNNER)
    p.add_argument('--workers', type=int, default=WORKERS)
    p.add_argument('--plan-only', action=argparse.BooleanOptionalAction, default=PLAN_ONLY)
    p.add_argument('--cases', nargs='+', default=SELECT_CASES)
    p.add_argument('--report-only', type=Path, help='Existing experiment directory; rebuild PDF without processing')
    args=p.parse_args(argv)
    if args.report_only:
        create_case_report(args.report_only)
        return 0
    import matplotlib  # Check report dependency before starting long pipeline runs.
    if args.workers < 1: p.error('workers must be positive')
    core=load_runner(args.runner.resolve())
    group=core.discover_orbits(args.source_folder.resolve(), {ORBIT})[ORBIT]
    # Reject a wrong month even if the folder happens to contain orbit 00001.
    for path in group.values():
        stamp=core.INPUT_RE.fullmatch(path.name).group(2)
        date=datetime.strptime(stamp,'%Y%jT%H%M')
        if (date.year,date.month)!=(2023,11):
            raise ValueError(f'Expected November 2023 source: {path}')
    epochs,dims,source_info=read_epochs(group)
    selected=designs()
    if args.cases:
        unknown=set(args.cases)-{c['name'] for c in selected}
        if unknown: p.error(f'Unknown case names: {sorted(unknown)}')
        selected=[c for c in selected if c['name'] in set(args.cases)|{'baseline'}]
    plans=[]
    for case in selected:
        try:
            case['gaps']=make_gaps(case,epochs)
            case['status']='planned'
            case['removed_frames']=sum(g['removed_frames'] for g in case['gaps'])
        except ValueError as exc:
            case.update(status='infeasible', error=str(exc))
        plans.append(case)
    if not args.plan_only:
        for step in range(1,7):
            if not Path(core.SCRIPTS[step]).is_file():
                raise FileNotFoundError(f'Step {step}: {core.SCRIPTS[step]}')
    out=args.output_root.resolve()/('orbit00001_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    config={'runner':str(args.runner.resolve()), 'scripts':{str(k):str(v) for k,v in core.SCRIPTS.items()},
            'python':str(core.PYTHON_EXE),'ml_python':str(core.ML_PYTHON_EXE),
            'resolutions':list(core.RESOLUTIONS),'extra_args':core.EXTRA_ARGS,'workers':args.workers}
    write_json(out/'plan.json',dict(orbit=ORBIT,source=source_info,configuration=config,cases=plans))
    write_csv(out/'case_plan.csv',[{**{k:v for k,v in c.items() if k!='gaps'},
              'max_realized_dt_s':max((max(g['dt_s'].values()) for g in c.get('gaps',[])),default=0.)} for c in plans])
    print(f'Experiment: {out}\nCases: {len(plans)}; all products retained.',flush=True)
    for c in plans:
        print(f"  {c['name']}: {c['status']}; removed={c.get('removed_frames','-')} frames",flush=True)
    if args.plan_only:
        return 0
    results=[]; metrics=[]
    cadence=source_info['q20']['cadence_s']
    for case in plans:
        state={'case':case['name'],'status':case['status'],'steps':[]}
        results.append(state)
        if case['status']=='infeasible':
            state['error']=case['error']
            write_json(out/'experiment_status.json',results)
            continue
        home=out/case['name']; home.mkdir()
        paths=core.month_paths(home,2023,11)
        for folder in paths.values(): folder.mkdir(parents=True)
        logs=home/'logs'; logs.mkdir()
        step=0
        state['status']='running'
        try:
            keep=keep_indices(len(epochs['q20']),case['gaps'])
            for filt,source in group.items():
                target=paths['l0c']/source.name
                clone_subset(source,target,dims[filt],keep,case['name'])
                with Dataset(target) as ds:
                    actual=values(ds['Epoch'])*unit_scale(ds['Epoch'])
                    if not np.array_equal(actual,epochs[filt][keep]):
                        raise ValueError(f'Unexpected Epoch changes while subsetting {filt}')
            for step in range(1,7):
                print(f"\n{case['name']} | step {step}: {core.LABELS[step]}",flush=True)
                tick=time.monotonic()
                core.check_inputs(step,paths)
                command=core.make_command(step,paths,args.workers)
                core.run_logged(command,Path(core.SCRIPTS[step]).resolve().parent,logs/f'step_{step}.log')
                if step==4: core.normalize_mlcloud_output(paths,ORBIT)
                core.validate_outputs(step,paths,ORBIT,logs)
                state['steps'].append({'step':step,'seconds':round(time.monotonic()-tick,2),'command':command})
                write_json(logs/'status.json',state)
            state['status']='completed'
        except Exception as exc:
            state.update(status='failed',failed_step=step,error=f'{type(exc).__name__}: {exc}')
            print(f"FAILED {case['name']}: {state['error']}",flush=True)
        except BaseException:
            state.update(status='interrupted',failed_step=step)
            raise
        finally:
            write_json(logs/'status.json',state)
            write_json(out/'experiment_status.json',results)
        metrics.extend(product_metrics(case['name'],paths,cadence))
        write_csv(out/'product_metrics.csv',metrics)
        write_csv(out/'metrics_vs_baseline.csv',comparison_rows(metrics))
        if case['name']=='baseline' and state['status']!='completed':
            print('Baseline failed; stopping before gap cases. Check its logs.',flush=True)
            break
    create_case_report(out)
    print(f'\nResults: {out}',flush=True)
    return 1 if any(r['status']!='completed' for r in results) else 0


if __name__=='__main__':
    raise SystemExit(main())
