#!/usr/bin/env python3
"""Process complete AWE orbits sequentially, retaining processed L0C/L0D/L1F/L1R only on request.
Python 3.10+. See README.md for usage, storage behavior, and failure recovery.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import queue
import signal
import threading
import shutil
import struct
import subprocess
import sys
import time
import uuid

# ----------------------- EDIT SETTINGS HERE -----------------------
YEAR = 2023
MONTH = 11
SOC_ROOT = Path(r'E:\soc')
SCRIPT_DIR = Path(__file__).resolve().parent
GITHUB_ROOT = Path(r'D:\Github')
SCRIPTS = {
    0: SCRIPT_DIR / '0_copy_and_remove_glare.py',
    1: GITHUB_ROOT / 'solar_panels' / '4_and_5_mlspb.py',
    2: SCRIPT_DIR / 'filter_flatfield_to_1lf.py',
    3: SCRIPT_DIR / 'regrid_l1f_to_l1r.py',
    4: GITHUB_ROOT / 'cloud_restructured' / '4_and_5_mlcloud.py',
    5: SCRIPT_DIR / 'coadd_to_l1a.py',
    6: SCRIPT_DIR / 'generate_l2a_l3a_l3c.py',
}
PYTHON_EXE = sys.executable
ML_PYTHON_EXE = sys.executable
L0C_SOURCE_ROOT = Path(r'Y:\soc\l0c\version02')
GLARE_CSV = None  # None: SCRIPT_DIR/csv/IndividualMonths/glare_intervals_MonYYYY.csv
SAVE_L0C = False  # Save processed L0C to SOC_ROOT/l0c/YYYY/MM
SAVE_L0D = False
SAVE_L1F = False
SAVE_L1R = False
CONTINUE_ON_ERROR = True  # Log a failed orbit and try the next one
START_STEP = 0  # 0: copy/remove glare; 1: existing L0C; 2: existing L0C + L0D
WORKERS = 4     # Parallel filters WITHIN one orbit
RESOLUTIONS = (2, 10)
# Change these only for scientific options, not input/output paths or mode.
MLSPB_MODE = 'boxes'  # 'trim' restores legacy frame trimming
MIN_CLEAN_FRAMES = 100
EXTRA_ARGS = {2: ['--mlspb-mode', MLSPB_MODE, '--min-clean-frames', str(MIN_CLEAN_FRAMES)], 3: [], 5: [], 6: []}
# -----------------------------------------------------------------
FILTERS = ('q20', 'p12', 'p14', 'bkg')
LEVELS = ('l0c','l0d','l1f','l1r','l1c','l1a','l2a','l3a','l3c')
LABELS = {0:'Copy / remove glare',1:'MLSPB',2:'Filter / flat-field',3:'Regrid',4:'MLCloud',5:'Co-add',6:'Temperature / swaths'}
INPUT_RE = re.compile(r'^awe_l0c_(q20|p12|p14|bkg)_(\d{7}T\d{4})_(\d{5})_(v\d+)(.*?)\.nc$', re.I)


def month_paths(root, year, month):
    return {level: Path(root)/level/f'{year:04d}'/f'{month:02d}' for level in LEVELS}


def discover_orbits(folder, requested=None):
    """Select v02 inputs only; reject duplicate filters or mismatched timestamps."""
    groups = {}
    for path in sorted(folder.glob('*.nc')):
        m = INPUT_RE.fullmatch(path.name)
        if not m: continue
        filt, stamp, orbit, version, suffix = m.groups()
        if version.lower() != 'v02': continue
        filt = filt.lower()
        if requested and orbit not in requested: continue
        group = groups.setdefault(orbit, {})
        if filt in group:
            raise ValueError(f'Multiple L0C {filt} files for orbit {orbit}; isolate one v02 file: {folder}')
        group[filt] = path
    if requested and requested - groups.keys():
        raise ValueError(f'Orbits not found: {sorted(requested - groups.keys())}')
    for orbit, group in groups.items():
        if set(group) != set(FILTERS):
            raise ValueError(f'Orbit {orbit}: missing L0C {sorted(set(FILTERS)-group.keys())}')
        identities = {INPUT_RE.fullmatch(p.name).groups()[1:4] for p in group.values()}
        if len(identities) != 1:
            raise ValueError(f'Orbit {orbit}: L0C timestamps/versions do not match')
    if not groups: raise FileNotFoundError(f'No complete v02 L0C orbits in {folder}')
    return dict(sorted(groups.items()))


def input_folder(args, permanent):
    if args.start_step != 0:
        return permanent['l0c']
    source = (args.l0c_source_root / f'{args.year:04d}' / f'{args.month:02d}').expanduser().resolve()
    if source == permanent['l0c'].resolve():
        raise ValueError('Raw L0C source and processed L0C destination must differ.')
    months = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
    args.glare_csv = (args.glare_csv or SCRIPT_DIR/'csv'/'IndividualMonths'/
                      f'glare_intervals_{months[args.month-1]}{args.year:04d}.csv').expanduser().resolve()
    if not args.glare_csv.is_file(): raise FileNotFoundError(f'Glare CSV: {args.glare_csv}')
    return source


def make_glare_command(group, paths, args):
    return [str(PYTHON_EXE),'-u',str(Path(SCRIPTS[0]).resolve()),
            '--source',str(group['q20'].parent),'--output',str(paths['l0c']),
            '--glare-csv',str(args.glare_csv),'--workers',str(min(args.workers,4)),
            '--overwrite','--files',*[str(group[f]) for f in FILTERS]]


def prepare_l0c(orbit, group, paths, args, logdir, records):
    if args.start_step != 0:
        for filt in FILTERS: shutil.copy2(group[filt],paths['l0c']/group[filt].name)
        return
    print(f'Orbit {orbit}, step 0: {LABELS[0]}',flush=True)
    tick = time.perf_counter()
    run_logged(make_glare_command(group,paths,args),Path(SCRIPTS[0]).resolve().parent,logdir/'step_0.log')
    validate_outputs(0,paths,orbit,logdir)
    records.append({'step':0,'seconds':round(time.perf_counter()-tick,2),'validated':True})


def make_command(step, paths, workers):
    script = Path(SCRIPTS[step]).resolve()
    if step in (1,4):
        source, target = ('l0c','l0d') if step == 1 else ('l1r','l1c')
        return [str(ML_PYTHON_EXE),'-u',str(Path(__file__).resolve()),'--ml-child',
                str(script),str(paths[source]),str(paths[target]),'1']
    options = {
        2:[str(paths['l0d']),str(paths['l0c']),str(paths['l1f']),'--orbits-at-a-time','1'],
        3:[str(paths['l1f']),str(paths['l1r'])],
        5:['--l1r-input',str(paths['l1r']),'--l1c-input',str(paths['l1c']),
           '--output',str(paths['l1a']),'--mode','both'],
        6:['--l1a-input',str(paths['l1a']),'--l2a-output',str(paths['l2a']),
           '--l3a-output',str(paths['l3a']),'--l3c-output',str(paths['l3c']),
           '--master-input',str(paths['l1r']),'--mode','all','--resolutions',*map(str,RESOLUTIONS)],
    }[step]
    return [str(PYTHON_EXE),'-u',str(script),*options,'--workers',str(workers),
            '--overwrite',*EXTRA_ARGS.get(step,[])]
def run_ml_child(script, source, target, overwrite):
    """Load existing ML script, override month paths, then call its main()."""
    script = Path(script).resolve()
    os.chdir(script.parent)  # Relative models/ paths stay in their own repository.
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location('_awe_ml_step', script)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {script}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ('INPUT_FOLDER', 'OUTPUT_FOLDER', 'main'):
        if not hasattr(module, name):
            raise AttributeError(f'{script.name} is missing {name}')
    module.INPUT_FOLDER = Path(source)
    module.OUTPUT_FOLDER = Path(target)
    # Avoid a hard-coded CSV month if optional CSV output is later enabled.
    module.CSV_OUTPUT_FOLDER = Path(target) / 'prediction_csv'
    if overwrite == '1':
        module.OVERWRITE = True
    print(f'Input: {module.INPUT_FOLDER}\nOutput: {module.OUTPUT_FOLDER}', flush=True)
    print(f'Model: {module.MODEL_PATH.resolve()}; overwrite={module.OVERWRITE}', flush=True)
    result = module.main()
    if isinstance(result, int) and result != 0:
        raise SystemExit(result)


def require_files(folder, pattern):
    if not folder.is_dir() or not any(folder.glob(pattern)):
        raise FileNotFoundError(f'No {pattern} files in {folder}')


def check_inputs(step, paths):
    requirements = {
        1: [('l0c', '*_q20_*.nc')],
        2: [('l0d', '*_q20_*.nc')] + [('l0c', f'*_{f}_*.nc') for f in ('q20','p12','p14','bkg')],
        3: [('l1f', f'*_{f}_*.nc') for f in ('q20','p12','p14','bkg')],
        4: [('l1r', '*_q20_*.nc')],
        5: [('l1r', f'*_{f}_*.nc') for f in ('p12','p14','bkg')] + [('l1c', '*_q20_*.nc')],
        6: [('l1a', f'*_{f}_*.nc') for f in ('q20','p12','p14','bkg')] + [('l1r', 'Master*.mgf')],
    }
    for level, pattern in requirements[step]:
        require_files(paths[level], pattern)


# Each preparation thread may supply its own cancellation event. Main-thread
# scientific stages have no event and retain their normal interrupt behavior.
LOG_CONTEXT = threading.local()


class PreparationCancelled(Exception):
    pass


def run_logged(command, cwd, logfile):
    stop = getattr(LOG_CONTEXT, 'stop_event', None)
    if stop is not None and stop.is_set():
        raise PreparationCancelled('Background preparation cancelled')
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    with logfile.open('w', encoding='utf-8') as log:
        log.write(subprocess.list2cmdline(command) + '\n\n')
        log.flush()
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', errors='replace', bufsize=1,
                              start_new_session=(os.name != 'nt')) as child:
            lines = queue.Queue()
            def read_output():
                try:
                    for line in child.stdout: lines.put(line)
                finally:
                    lines.put(None)
            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            try:
                output_closed = False
                while not output_closed or child.poll() is None:
                    if stop is not None and stop.is_set():
                        raise PreparationCancelled('Background preparation cancelled')
                    try: line = lines.get(timeout=0.1)
                    except queue.Empty: continue
                    if line is None:
                        output_closed = True
                    else:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                code = child.wait()
            except BaseException:
                # Kill the entire process tree, including the glare filter pool.
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    try: os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                if child.poll() is None: child.kill()
                child.wait()
                raise
            finally:
                reader.join(timeout=2)
    if code:
        raise RuntimeError(f'Step exited with code {code}. Log: {logfile}')


def exactly_one(folder, pattern):
    found = sorted(folder.glob(pattern))
    if len(found) != 1:
        raise ValueError(f'Expected one {pattern} in {folder}; found {len(found)}')
    return found[0]


def exactly_one_of(folder, patterns):
    """Accept either ML output naming convention, but never select a duplicate."""
    found = sorted({path for pattern in patterns for path in folder.glob(pattern)})
    if len(found) != 1:
        raise ValueError(f'Expected one of {patterns} in {folder}; found {len(found)}')
    return found[0]


def ml_output(folder, level, orbit, stamp='*', version='*'):
    source_level = {'l0d':'l0c', 'l1c':'l1r'}[level]
    return exactly_one_of(folder, [f'awe_{name}_q20_{stamp}_{orbit}_{version}.nc'
                                   for name in (level,source_level)])


def normalize_mlcloud_output(paths, orbit):
    # Coadding explicitly discovers L1C filenames. Only rename this generated
    # workspace output; leave the source L1R and all NetCDF content untouched.
    source = ml_output(paths['l1c'],'l1c',orbit)
    if source.name.startswith('awe_l1r_'):
        target = source.with_name(source.name.replace('awe_l1r_','awe_l1c_',1))
        if target.exists(): raise FileExistsError(f'MLCloud output collision: {target}')
        source.rename(target)
        print(f'MLCloud output renamed for coadding: {target.name}',flush=True)


def expected_outputs(step, paths, orbit):
    checks = []
    specifications = {
        0:[('l0c',FILTERS,'Radiance')],
        1:[('l0d',('q20',),'Radiance')],
        2:[('l1f',FILTERS,'Radiance')],
        3:[('l1r',FILTERS,'Radiance')],
        4:[('l1c',('q20',),'Radiance')],
        5:[('l1a',FILTERS,'Radiance')],
    }
    for level, filters, variable in specifications.get(step,[]):
        for filt in filters:
            p = (ml_output(paths[level],level,orbit) if step in (1,4) else
                 exactly_one(paths[level],f'*_{level}_{filt}_*_{orbit}_*.nc'))
            required = [variable,'Latitude','Longitude','Epoch']
            if step == 1: required.append('MLSPB')
            check = {'path':str(p),'variables':required}
            if step == 4:
                # Same preference/order as coadd_to_l1a.py; processed-only L1C
                # is the ML producer's default and is already supported there.
                check['variable_alternatives'] = [['MLCloud','Processed_MLCloud']]
            checks.append(check)
    if step == 6:
        for level in ('l2a','l3a'):
            for resolution in RESOLUTIONS:
                p = exactly_one(paths[level]/f'{resolution}x{resolution}',f'*_{level}_tmp_*_{orbit}_*.nc')
                checks.append({'path':str(p),'variables':['Temperature','Latitude','Longitude','Epoch']})
        p = exactly_one(paths['l3c'],f'*_l3c_q20_*_{orbit}_*.nc')
        checks.append({'path':str(p),'variables':['Radiance','Latitude','Longitude','Epoch']})
    if step in (3,6):
        master = paths['l1r']/f'Master{orbit}.mgf'
        with master.open('rb') as stream:
            header = stream.read(8)
        if len(header) != 8: raise ValueError(f'Incomplete master: {master}')
        nx, ny = struct.unpack('<ii',header)
        if min(nx,ny) <= 0 or master.stat().st_size != 8+16*nx*ny:
            raise ValueError(f'Invalid master size: {master}')
    return checks


def validate_child(specification):
    """Open every product and sample endpoints; no full radiance reread."""
    from netCDF4 import Dataset
    for item in json.loads(Path(specification).read_text(encoding='utf-8')):
        with Dataset(item['path'],'r') as ds:
            required = list(item['variables'])
            for alternatives in item.get('variable_alternatives',[]):
                name = next((n for n in alternatives if n in ds.variables),None)
                if name is None:
                    raise ValueError(f"{item['path']}: missing required cloud variable; expected one of {alternatives}")
                required.append(name)
            for name in required:
                if name not in ds.variables: raise ValueError(f"{item['path']}: missing {name}")
                v = ds[name]
                if not v.size: raise ValueError(f"{item['path']}: empty {name}")
                if v.ndim:
                    v[tuple(0 for _ in v.shape)]
                    v[tuple(n-1 for n in v.shape)]
                else: v[...]
    print(f'Validated {len(json.loads(Path(specification).read_text()))} NetCDF product(s).')


def validate_outputs(step, paths, orbit, logdir):
    checks = expected_outputs(step,paths,orbit)
    spec = logdir/f'validate_{step}.json'
    spec.write_text(json.dumps(checks,indent=2),encoding='utf-8')
    run_logged([str(PYTHON_EXE),'-u',str(Path(__file__).resolve()),'--validate-child',str(spec)],
               SCRIPT_DIR,logdir/f'validate_{step}.log')


def fingerprint(group, args):
    """Completion receipts depend on inputs, script bytes, and run settings."""
    payload = {'inputs':[(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns)
                          for p in sorted(group.values())],
               'settings':[args.start_step,args.save_l0c,args.save_l0d,args.save_l1f,args.save_l1r,args.workers,
                           RESOLUTIONS,EXTRA_ARGS,str(PYTHON_EXE),str(ML_PYTHON_EXE)],
               'scripts':[(str(Path(p).resolve()),hashlib.sha256(Path(p).read_bytes()).hexdigest())
                          for p in [Path(__file__),*[SCRIPTS[s] for s in range(args.start_step,7)]]]}
    if args.start_step == 0:
        payload['glare_csv'] = [str(args.glare_csv),hashlib.sha256(args.glare_csv.read_bytes()).hexdigest()]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def receipt_matches(path, signature):
    if not path.is_file(): return False
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return (data['signature'] == signature and bool(data['products']) and
                all(Path(p['path']).is_file() and Path(p['path']).stat().st_size == p['size']
                    and Path(p['path']).stat().st_mtime_ns == p['mtime_ns'] for p in data['products']))
    except (OSError,ValueError,KeyError): return False


def publish_file(source, target):
    """Copy to a sibling partial and atomically replace, including across drives."""
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        shutil.copy2(source,temporary)
        os.replace(temporary,target)
    finally:
        if temporary.exists(): temporary.unlink()


def process_one(orbit, group, args, permanent, work, logdir, receipt, signature):
    paths = {level: work/level for level in LEVELS}
    for p in paths.values(): p.mkdir(parents=True,exist_ok=True)
    state = {'orbit':orbit,'work':str(work),'status':'running','steps':[]}
    status_path = logdir/'status.json'
    def save_status(): status_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save_status()
    try:
        # Real copies prevent an ML script that edits its input from altering L0C.
        prepare_l0c(orbit,group,paths,args,logdir,state['steps'])
        save_status()
        if args.start_step == 2:
            source = group['l0d']
            shutil.copy2(source,paths['l0d']/source.name)
            validate_outputs(1,paths,orbit,logdir)
        for step in range(max(1,args.start_step),7):
            print(f'\nOrbit {orbit}, step {step}: {LABELS[step]}',flush=True)
            tick = time.perf_counter()
            check_inputs(step,paths)
            command = make_command(step,paths,args.workers)
            run_logged(command,Path(SCRIPTS[step]).resolve().parent,logdir/f'step_{step}.log')
            if step == 4: normalize_mlcloud_output(paths,orbit)
            validate_outputs(step,paths,orbit,logdir)
            for report in paths[{1:'l0d',2:'l1f',3:'l1r',4:'l1c',5:'l1a',6:'l3a'}[step]].glob('*.csv'):
                shutil.copy2(report,logdir/report.name)
            state['steps'].append({'step':step,'seconds':round(time.perf_counter()-tick,2)})
            save_status()
            # Delete only generated files in this uniquely owned orbit workspace,
            # after the next stage has successfully written and validated its outputs.
            if step == 2:
                if not (args.start_step == 0 and args.save_l0c): shutil.rmtree(paths['l0c'])
                if not args.save_l0d: shutil.rmtree(paths['l0d'])
            if step == 3 and not args.save_l1f: shutil.rmtree(paths['l1f'])
            if step == 5 and not args.save_l1r:
                for p in paths['l1r'].glob('*.nc'): p.unlink()
        products = []
        for level in LEVELS:
            if level == 'l0c' and (args.start_step != 0 or not args.save_l0c): continue
            if level == 'l0d' and (args.start_step == 2 or not args.save_l0d): continue
            if level == 'l1f' and not args.save_l1f: continue
            for source in sorted(paths[level].rglob('*')):
                if not source.is_file() or source.suffix not in ('.nc','.mgf'): continue
                target = permanent[level]/source.relative_to(paths[level])
                products.append((source,target))
        if not products: raise ValueError('No products to publish')
        conflicts = [str(dst) for _,dst in products if dst.exists()]
        if conflicts and not args.overwrite:
            raise FileExistsError('Existing outputs; use --overwrite to replace: '+', '.join(conflicts))
        # Publish only after all selected stages pass; originals remain in work if publication fails.
        for src,dst in products: publish_file(src,dst)
        saved = [{'path':str(dst),'size':dst.stat().st_size,'mtime_ns':dst.stat().st_mtime_ns}
                 for _,dst in products]
        receipt.parent.mkdir(parents=True,exist_ok=True)
        temp_receipt = logdir/'completion.json'
        temp_receipt.write_text(json.dumps({'signature':signature,'products':saved},indent=2),encoding='utf-8')
        publish_file(temp_receipt,receipt)
        state['status'] = 'completed'
        save_status()
        shutil.rmtree(work)
        print(f'Orbit {orbit} complete; temporary files removed.',flush=True)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        save_status()
        print(f'Orbit {orbit} stopped. Remaining temporary files: {work}\nLogs: {logdir}',flush=True)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year',type=int,default=YEAR)
    p.add_argument('--month',type=int,default=MONTH)
    p.add_argument('--soc-root',type=Path,default=SOC_ROOT)
    p.add_argument('--l0c-source-root',type=Path,default=L0C_SOURCE_ROOT)
    p.add_argument('--glare-csv',type=Path,default=GLARE_CSV)
    p.add_argument('--save-l0c',action=argparse.BooleanOptionalAction,default=SAVE_L0C)
    p.add_argument('--work-root',type=Path,help='Temporary parent; default SOC_ROOT/_orbit_work')
    p.add_argument('--orbits',nargs='+',type=int,help='Only these orbit numbers; default all in month')
    p.add_argument('--limit',type=int,help='Maximum number of pending orbits to process')
    p.add_argument('--start-step',type=int,choices=(0,1,2),default=START_STEP)
    p.add_argument('--save-l0d',action=argparse.BooleanOptionalAction,default=SAVE_L0D)
    p.add_argument('--save-l1f',action=argparse.BooleanOptionalAction,default=SAVE_L1F)
    p.add_argument('--save-l1r',action=argparse.BooleanOptionalAction,default=SAVE_L1R)
    p.add_argument('--workers',type=int,default=WORKERS)
    p.add_argument('--continue-on-error',action=argparse.BooleanOptionalAction,
                   default=CONTINUE_ON_ERROR,help='Continue with other orbits after a failure (default: %(default)s)')
    p.add_argument('--overwrite',action='store_true',help='Reprocess and replace existing outputs')
    p.add_argument('--dry-run',action='store_true')
    args = p.parse_args(argv)
    if not 1 <= args.month <= 12 or not 1 <= args.year <= 9999: p.error('Invalid year/month')
    if args.workers < 1 or (args.limit is not None and args.limit < 1): p.error('workers/limit must be positive')
    if args.orbits and any(o < 0 or o > 99999 for o in args.orbits): p.error('Orbit must be 0..99999')
    args.soc_root = args.soc_root.expanduser().resolve()
    permanent = month_paths(args.soc_root,args.year,args.month)
    requested = {f'{o:05d}' for o in args.orbits} if args.orbits else None
    groups = discover_orbits(input_folder(args,permanent),requested)
    if args.start_step == 2:
        for orbit,group in groups.items():
            q = INPUT_RE.fullmatch(group['q20'].name)
            group['l0d'] = ml_output(permanent['l0d'],'l0d',orbit,q[2],q[4]+'*')
    work_root = (args.work_root or args.soc_root/'_orbit_work').expanduser().resolve()
    receipts = args.soc_root/'pipeline_orbit_state'/f'{args.year:04d}'/f'{args.month:02d}'
    print(f'Orbit pipeline: {args.year:04d}/{args.month:02d}; save L0C={args.save_l0c}; save L0D={args.save_l0d}; save L1F={args.save_l1f}; save L1R={args.save_l1r}')
    print(f'Temporary parent: {work_root}; master grids always retained in {permanent["l1r"]}')
    if args.dry_run:
        for orbit in list(groups)[:args.limit]:
            print(f'Orbit {orbit}: steps {args.start_step}..6')
            paths = {level:work_root/f'preview_{orbit}'/level for level in LEVELS}
            if args.start_step == 0: print(subprocess.list2cmdline(make_glare_command(groups[orbit],paths,args)))
            for step in range(max(1,args.start_step),7): print(subprocess.list2cmdline(make_command(step,paths,args.workers)))
        print('Dry run only; no files changed. Completion receipts are checked during a real run.')
        return 0
    for step in range(args.start_step,7):
        if not Path(SCRIPTS[step]).is_file(): raise FileNotFoundError(f'Step {step}: {SCRIPTS[step]}')
    for exe in (PYTHON_EXE,ML_PYTHON_EXE):
        if not Path(exe).is_file(): raise FileNotFoundError(f'Python interpreter: {exe}')
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_logs = args.soc_root/'pipeline_logs'/f'{args.year:04d}'/f'{args.month:02d}'/run_id
    records = []
    attempted = 0
    run_logs.mkdir(parents=True,exist_ok=True)
    summary_path = run_logs/'summary.json'
    print(f'Continue on error: {args.continue_on_error}',flush=True)
    try:
        for orbit,group in groups.items():
            if args.limit is not None and attempted >= args.limit: break
            record = {'orbit':orbit,'status':'running'}
            attempted += 1
            try:
                signature = fingerprint(group,args)
                receipt = receipts/f'{orbit}.json'
                if not args.overwrite and receipt_matches(receipt,signature):
                    print(f'SKIP completed orbit {orbit}',flush=True)
                    record['status'] = 'skipped'
                    attempted -= 1
                    continue
                # Do not replace untracked outputs without explicit --overwrite.
                if not args.overwrite:
                    conflicts = [p for level in (('l0c',) if args.start_step == 0 and args.save_l0c else ()) + (('l0d',) if args.start_step <= 1 and args.save_l0d else ()) + ('l1f','l1r','l1c','l1a','l2a','l3a','l3c')
                                 for p in permanent[level].rglob(f'*_{orbit}_*.nc')]
                    master = permanent['l1r']/f'Master{orbit}.mgf'
                    if conflicts or master.exists():
                        raise FileExistsError(f'Orbit {orbit} already has outputs but no matching completion receipt. '
                                              'Choose a fresh SOC root or use --overwrite.')
                logdir = run_logs/orbit
                logdir.mkdir(parents=True,exist_ok=True)
                work = work_root/f'{args.year:04d}{args.month:02d}_{orbit}_{uuid.uuid4().hex}'
                work.mkdir(parents=True,exist_ok=False)
                record.update(work=str(work),logs=str(logdir))
                process_one(orbit,group,args,permanent,work,logdir,receipt,signature)
                record['status'] = 'completed'
            except Exception as exc:
                record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                print(f'FAILED orbit {orbit}: {record["error"]}',flush=True)
                if not args.continue_on_error: raise
                print('Continuing to the next orbit.',flush=True)
            except BaseException as exc:
                # User interrupts/SystemExit must never silently start another orbit.
                record.update(status='interrupted',error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                records.append(record)
                summary_path.write_text(json.dumps(records,indent=2),encoding='utf-8')
    finally:
        completed = sum(r['status']=='completed' for r in records)
        skipped = sum(r['status']=='skipped' for r in records)
        failures = [r for r in records if r['status']=='failed']
        print(f'Finished: {completed} completed; {len(failures)} failed; {skipped} skipped.',flush=True)
        if failures:
            print('Failed orbits: '+', '.join(r['orbit'] for r in failures),flush=True)
        print(f'Run summary: {summary_path}',flush=True)
    return 1 if failures else 0



if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--ml-child':
        if len(sys.argv)!=6: raise SystemExit('Invalid ML child arguments')
        run_ml_child(*sys.argv[2:])
    elif len(sys.argv)>1 and sys.argv[1]=='--validate-child':
        if len(sys.argv)!=3: raise SystemExit('Invalid validation arguments')
        validate_child(sys.argv[2])
    else:
        raise SystemExit(main())
