#!/usr/bin/env python3
"""Run bounded batches of AWE orbits through L0C -> L3A/L3C (Python 3.10+).
Keep run_awe_pipeline_orbit.py and the four processing scripts beside this file.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
# Spyder may execute this file without adding its folder to sys.path.
# Load the exact sibling helper, also avoiding a stale cached copy in Spyder.
_HELPER_PATH = Path(__file__).resolve().with_name('run_awe_pipeline_orbit.py')
if not _HELPER_PATH.is_file():
    raise FileNotFoundError(
        f'Required helper is missing: {_HELPER_PATH}. '
        'Extract the whole awe_storage_saving ZIP and keep both runners together.')
_HELPER_SPEC = importlib.util.spec_from_file_location('_awe_batch_core', _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError(f'Cannot load helper: {_HELPER_PATH}')
core = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = core
_HELPER_SPEC.loader.exec_module(core)

# ----------------------- EDIT SETTINGS HERE -----------------------
YEAR = 2023
MONTH = 11
SOC_ROOT = Path(r'E:\soc')
GITHUB_ROOT = Path(r'D:\Github')
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS = {
    1: GITHUB_ROOT/'solar_panels'/'4_and_5_mlspb.py',
    2: SCRIPT_DIR/'filter_flatfield_to_1lf.py',
    3: SCRIPT_DIR/'regrid_l1f_to_l1r.py',
    4: GITHUB_ROOT/'cloud_restructured'/'4_and_5_mlcloud.py',
    5: SCRIPT_DIR/'coadd_to_l1a.py',
    6: SCRIPT_DIR/'generate_l2a_l3a_l3c.py',
}
PYTHON_EXE = sys.executable
ML_PYTHON_EXE = sys.executable
ORBITS_AT_A_TIME = 5
WORKERS = 20  # Total file workers for the batch, not 20 per orbit
SAVE_L0D = False
SAVE_L1F = False
SAVE_L1R = False
CONTINUE_ON_ERROR = True
OVERWRITE = True  # Set True to reprocess and replace matching outputs
START_STEP = 1  # 2 reuses existing L0D; both choices still require L0C
RESOLUTIONS = (2,10)
EXTRA_ARGS = {2:[],3:[],5:[],6:[]}
# -----------------------------------------------------------------


def configure_core():
    for name in ('SCRIPTS','PYTHON_EXE','ML_PYTHON_EXE','RESOLUTIONS','EXTRA_ARGS'):
        setattr(core,name,globals()[name])


def command_for(step, paths, args, orbit_count):
    command = core.make_command(step,paths,args.workers)
    if step == 2:
        # The flat-field script has its own batch-size flag. It must match the
        # outer batch or only four tasks at a time would be submitted.
        i = command.index('--orbits-at-a-time')
        command[i+1] = str(orbit_count)
    return command


def paths_in(root):
    return {level:root/level for level in core.LEVELS}


def orbit_files(folder, orbit):
    """Only this orbit's AWE products, including partials; never substring IDs."""
    pattern = re.compile(r'^awe_l\d[a-z]_[a-z0-9]+_\d{7}T\d{4}_'+re.escape(orbit)+r'_v\d+',re.I)
    master = re.compile(r'^Master'+re.escape(orbit)+r'\.mgf(?:\.|$)',re.I)
    return sorted(p for p in folder.rglob('*') if p.is_file() and
                  (pattern.match(p.name) or master.match(p.name)))


def relocate_orbit(source_paths, target_paths, orbit):
    # Both roots live on the same volume. Moves avoid another orbit-sized copy.
    pairs = [(p,target_paths[level]/p.relative_to(source_paths[level]))
             for level in core.LEVELS for p in orbit_files(source_paths[level],orbit)]
    for _,dst in pairs:
        if dst.exists(): raise FileExistsError(f'Conflicting staged file: {dst}')
    for src,dst in pairs:
        dst.parent.mkdir(parents=True,exist_ok=True)
        src.rename(dst)


def cleanup_after(step, paths, orbit, args):
    levels = []
    if step == 2:
        levels.append('l0c')
        if not args.save_l0d: levels.append('l0d')
    if step == 3 and not args.save_l1f: levels.append('l1f')
    for level in levels:
        for p in orbit_files(paths[level],orbit): p.unlink()
    if step == 5 and not args.save_l1r:
        for p in orbit_files(paths['l1r'],orbit):
            if p.suffix == '.nc': p.unlink()  # Keep MasterNNNNN.mgf


def save_products(paths, permanent, job, args, logdir):
    orbit = job['orbit']
    products = []
    for level in core.LEVELS[1:]:
        if level == 'l0d' and (args.start_step == 2 or not args.save_l0d): continue
        if level == 'l1f' and not args.save_l1f: continue
        for source in orbit_files(paths[level],orbit):
            if source.suffix not in ('.nc','.mgf'): continue
            if level == 'l1r' and source.suffix == '.nc' and not args.save_l1r: continue
            products.append((source,permanent[level]/source.relative_to(paths[level])))
    if not products: raise ValueError(f'Orbit {orbit}: no products to publish')
    if not args.overwrite:
        for _,dst in products:
            if dst.exists(): raise FileExistsError(f'Existing output: {dst}; use --overwrite')
    for src,dst in products: core.publish_file(src,dst)
    saved = [{'path':str(dst),'size':dst.stat().st_size,'mtime_ns':dst.stat().st_mtime_ns} for _,dst in products]
    completion = logdir/'completion.json'
    completion.write_text(json.dumps({'signature':job['signature'],'products':saved},indent=2),encoding='utf-8')
    core.publish_file(completion,job['receipt'])
    # Receipt is committed. Remove only this orbit's remaining staged data.
    for folder in paths.values():
        for p in orbit_files(folder,orbit): p.unlink()


def process_batch(jobs, args, permanent, work, batch_logs, checkpoint):
    """One child per stage; on failure retry that stage per orbit in isolation."""
    paths = paths_in(work/'active')
    for p in paths.values(): p.mkdir(parents=True,exist_ok=True)
    batch_logs.mkdir(parents=True,exist_ok=True)
    stage_records = []
    active = []
    started = time.perf_counter()

    def log_for(job):
        p = batch_logs/'orbits'/job['orbit']
        p.mkdir(parents=True,exist_ok=True)
        return p

    def update():
        for job in jobs:
            (log_for(job)/'status.json').write_text(json.dumps(job['record'],indent=2),encoding='utf-8')
        (batch_logs/'batch_status.json').write_text(json.dumps({
            'orbits':[j['orbit'] for j in jobs], 'stage_timings':stage_records,
            'elapsed_seconds':round(time.perf_counter()-started,2)},indent=2),encoding='utf-8')
        checkpoint()

    def fail(job, exc, source_paths=paths):
        target = paths_in(work/'failed'/job['orbit'])
        relocate_orbit(source_paths,target,job['orbit'])
        job['record'].update(status='failed',error=f'{type(exc).__name__}: {exc}',
                             work=str(work/'failed'/job['orbit']))
        print(f'FAILED orbit {job["orbit"]}: {exc}',flush=True)
        update()
        if not args.continue_on_error: raise exc

    try:
        for job in jobs:
            job['record'].update(status='running',work=str(work/'active'),logs=str(log_for(job)),steps=[])
            try:
                for filt in core.FILTERS:
                    src = job['group'][filt]
                    shutil.copy2(src,paths['l0c']/src.name)
                if args.start_step == 2:
                    src = job['group']['l0d']
                    shutil.copy2(src,paths['l0d']/src.name)
                    core.validate_outputs(1,paths,job['orbit'],log_for(job))
                active.append(job)
            except Exception as exc: fail(job,exc)
        update()
        for step in range(args.start_step,7):
            if not active: break
            print(f'\nBatch {batch_logs.name}, step {step}: {core.LABELS[step]}; orbits '+
                  ', '.join(j['orbit'] for j in active),flush=True)
            tick = time.perf_counter()
            batch_error = None
            try:
                core.check_inputs(step,paths)
                core.run_logged(command_for(step,paths,args,len(active)),Path(SCRIPTS[step]).resolve().parent,
                                batch_logs/f'step_{step}.log')
            except Exception as exc:
                batch_error = exc
                print(f'Batch step reported: {exc}',flush=True)
                if not args.continue_on_error: raise
            stage = {'step':step,'batch_process_seconds':round(time.perf_counter()-tick,2),
                     'orbits':[j['orbit'] for j in active]}
            if batch_error: stage['error'] = f'{type(batch_error).__name__}: {batch_error}'
            stage_records.append(stage)
            survivors = []
            for job in active:
                orbit = job['orbit']
                error = batch_error
                retried = False
                if error is None:
                    try:
                        if step == 4: core.normalize_mlcloud_output(paths,orbit)
                        core.validate_outputs(step,paths,orbit,log_for(job))
                    except Exception as exc: error = exc
                if error is not None:
                    if not args.continue_on_error: raise error
                    # A batch error can prevent otherwise good orbits reaching
                    # later substages (e.g. L2A failure before L3A). Retry all
                    # affected orbits once with only their own files visible.
                    retry_root = work/'retry'/orbit
                    retry_paths = paths_in(retry_root)
                    for p in retry_paths.values(): p.mkdir(parents=True,exist_ok=True)
                    relocate_orbit(paths,retry_paths,orbit)
                    retry_logs = log_for(job)/f'retry_step_{step}'
                    retry_logs.mkdir(parents=True,exist_ok=True)
                    print(f'Orbit {orbit}: retrying step {step} alone.',flush=True)
                    retried = True
                    job['record'].setdefault('retries',[]).append({'step':step,'reason':str(error),'logs':str(retry_logs)})
                    try:
                        core.check_inputs(step,retry_paths)
                        core.run_logged(command_for(step,retry_paths,args,1),Path(SCRIPTS[step]).resolve().parent,
                                        retry_logs/f'step_{step}.log')
                        if step == 4: core.normalize_mlcloud_output(retry_paths,orbit)
                        core.validate_outputs(step,retry_paths,orbit,retry_logs)
                    except Exception as exc:
                        fail(job,exc,retry_paths)
                        continue
                    relocate_orbit(retry_paths,paths,orbit)
                    shutil.rmtree(retry_root)
                job['record']['steps'].append({'step':step,'validated':True,'retried':retried,
                                                'batch_log':str(batch_logs/f'step_{step}.log')})
                cleanup_after(step,paths,orbit,args)
                survivors.append(job)
            active = survivors
            stage['stage_total_seconds'] = round(time.perf_counter()-tick,2)
            # Preserve timing/error CSVs before workspace cleanup, including L2A subfolders.
            for level,folder in paths.items():
                for report in folder.rglob('*.csv'):
                    dst = batch_logs/'reports'/level/report.relative_to(folder)
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(report,dst)
            update()
        for job in active:
            try:
                save_products(paths,permanent,job,args,log_for(job))
                job['record']['status'] = 'completed'
                print(f'Orbit {job["orbit"]} complete; products saved.',flush=True)
                update()
            except Exception as exc: fail(job,exc)
    except BaseException as exc:
        for job in jobs:
            if job['record']['status'] in ('pending','running'):
                job['record'].update(status='interrupted' if not isinstance(exc,Exception) else 'aborted',
                                     error=f'{type(exc).__name__}: {exc}',work=str(work))
        raise
    finally:
        update()
    if all(j['record']['status']=='completed' for j in jobs):
        shutil.rmtree(work)
    else:
        print(f'Failed-orbit temporary data retained under {work}',flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year',type=int,default=YEAR)
    p.add_argument('--month',type=int,default=MONTH)
    p.add_argument('--soc-root',type=Path,default=SOC_ROOT)
    p.add_argument('--work-root',type=Path)
    p.add_argument('--orbits',nargs='+',type=int)
    p.add_argument('--limit',type=int,help='Maximum pending orbit attempts, including failures')
    p.add_argument('--orbits-at-a-time','--batch-size',dest='orbits_at_a_time',type=int,default=ORBITS_AT_A_TIME)
    p.add_argument('--workers',type=int,default=WORKERS)
    p.add_argument('--start-step',type=int,choices=(1,2),default=START_STEP)
    for name,value in [('save-l0d',SAVE_L0D),('save-l1f',SAVE_L1F),('save-l1r',SAVE_L1R),
                       ('continue-on-error',CONTINUE_ON_ERROR)]:
        p.add_argument('--'+name,action=argparse.BooleanOptionalAction,default=value)
    p.add_argument('--overwrite',action=argparse.BooleanOptionalAction,default=OVERWRITE)
    p.add_argument('--dry-run',action='store_true')
    args = p.parse_args(argv)
    if not 1 <= args.month <= 12 or not 1 <= args.year <= 9999: p.error('Invalid year/month')
    if min(args.workers,args.orbits_at_a_time) < 1 or (args.limit is not None and args.limit < 1):
        p.error('workers, orbits-at-a-time and limit must be positive')
    if args.orbits and any(o < 0 or o > 99999 for o in args.orbits): p.error('Orbit must be 0..99999')
    configure_core()
    args.soc_root = args.soc_root.expanduser().resolve()
    permanent = core.month_paths(args.soc_root,args.year,args.month)
    requested = {f'{o:05d}' for o in args.orbits} if args.orbits else None
    groups = core.discover_orbits(permanent['l0c'],requested)
    work_root = (args.work_root or args.soc_root/'_orbit_work').expanduser().resolve()
    print(f'Batch size={args.orbits_at_a_time}; total file workers={args.workers}; '
          f'save L0D/L1F/L1R={args.save_l0d}/{args.save_l1f}/{args.save_l1r}',flush=True)
    if args.dry_run:
        selected = list(groups)[:args.limit]
        for start in range(0,len(selected),args.orbits_at_a_time):
            batch = selected[start:start+args.orbits_at_a_time]
            print('Batch: '+', '.join(batch))
            paths = paths_in(work_root/f'preview_batch_{start//args.orbits_at_a_time+1}')
            for step in range(args.start_step,7):
                print(subprocess.list2cmdline(command_for(step,paths,args,len(batch))))
        print('Dry run: no files changed. Completion receipts are checked during a real run.')
        return 0
    for step in range(args.start_step,7):
        if not Path(SCRIPTS[step]).is_file(): raise FileNotFoundError(f'Step {step}: {SCRIPTS[step]}')
    for exe in (PYTHON_EXE,ML_PYTHON_EXE):
        if not Path(exe).is_file(): raise FileNotFoundError(f'Python interpreter: {exe}')
    receipts = args.soc_root/'pipeline_orbit_state'/f'{args.year:04d}'/f'{args.month:02d}'
    run_logs = args.soc_root/'pipeline_logs'/f'{args.year:04d}'/f'{args.month:02d}'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_logs.mkdir(parents=True,exist_ok=True)
    records = []
    def checkpoint(): (run_logs/'summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    pending = []
    attempted = 0
    batch_number = 0
    def flush_batch():
        nonlocal batch_number
        batch_number += 1
        work = work_root/f'{args.year:04d}{args.month:02d}_batch_{batch_number:04d}_{uuid.uuid4().hex}'
        process_batch(list(pending),args,permanent,work,run_logs/f'batch_{batch_number:04d}',checkpoint)
        pending.clear()
    try:
        for orbit,group in groups.items():
            if args.limit is not None and attempted >= args.limit: break
            record = {'orbit':orbit,'status':'pending'}
            records.append(record)
            attempted += 1
            try:
                if args.start_step == 2:
                    m = core.INPUT_RE.fullmatch(group['q20'].name)
                    group['l0d'] = core.ml_output(permanent['l0d'],'l0d',orbit,m[2],m[4]+'*')
                signature = hashlib.sha256((core.fingerprint(group,args)+
                    hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()
                receipt = receipts/f'{orbit}.json'
                if not args.overwrite and core.receipt_matches(receipt,signature):
                    record['status'] = 'skipped'
                    attempted -= 1
                    print(f'SKIP completed orbit {orbit}',flush=True)
                    continue
                if not args.overwrite:
                    levels = (('l0d',) if args.start_step==1 and args.save_l0d else ()) + ('l1f','l1r','l1c','l1a','l2a','l3a','l3c')
                    existing = [f for level in levels for f in orbit_files(permanent[level],orbit)
                                if f.suffix in ('.nc','.mgf')]
                    if existing:
                        raise FileExistsError(f'Orbit {orbit}: existing outputs without a matching batch receipt; '
                                              'select new orbits or use --overwrite to reprocess')
                pending.append({'orbit':orbit,'group':group,'record':record,'receipt':receipt,'signature':signature})
            except Exception as exc:
                record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                print(f'FAILED orbit {orbit}: {exc}',flush=True)
                if not args.continue_on_error: raise
            finally: checkpoint()
            if len(pending) == args.orbits_at_a_time: flush_batch()
        if pending: flush_batch()
    finally:
        for record in records:
            if record['status'] == 'pending': record['status'] = 'not_started'
        checkpoint()
        completed = sum(x['status']=='completed' for x in records)
        failed = [x['orbit'] for x in records if x['status'] in ('failed','aborted')]
        skipped = sum(x['status']=='skipped' for x in records)
        print(f'Finished: {completed} completed; {len(failed)} failed/aborted; {skipped} skipped.',flush=True)
        if failed: print('Failed/aborted orbits: '+', '.join(failed),flush=True)
        print(f'Run summary: {run_logs/"summary.json"}',flush=True)
    return 1 if failed else 0


if __name__ == '__main__': raise SystemExit(main())
