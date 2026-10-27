#!/usr/bin/env python3
"""Run bounded batches of AWE orbits through L0C -> L3A/L3C (Python 3.10+).
Keep run_awe_pipeline_orbit.py, 0_copy_and_remove_glare.py and the four processing scripts beside this file.
"""
from __future__ import annotations
import argparse
import copy
import csv
import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import queue
import threading
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
SAVE_TO_SERVER = True  # True: publish master grids, L1R, L1A, L3A and L3C on server
SERVER_ROOT = Path(r'Y:\soc')
VERSION_FOLDER = 'version02p1'
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS = {
    0: SCRIPT_DIR/'0_copy_and_remove_glare.py',
    1: SCRIPT_DIR/'4_and_5_mlspb.py',
    2: SCRIPT_DIR/'filter_flatfield_to_1lf.py',
    3: SCRIPT_DIR/'regrid_l1f_to_l1r.py',
    4: SCRIPT_DIR/'4_and_5_mlcloud.py',
    5: SCRIPT_DIR/'coadd_to_l1a.py',
    6: SCRIPT_DIR/'generate_l2a_l3a_l3c.py',
}
PYTHON_EXE = sys.executable
ML_PYTHON_EXE = sys.executable
ORBITS_AT_A_TIME = 5
PREFETCH_STEP0 = True  # Prepare one next batch during current steps 1-6
OVERLAP_BATCHES = True
MAX_PROCESSING_BATCHES = 3  # Plus at most one preparing/prepared batch
WORKERS = 20  # Fallback file workers per CPU stage, not per orbit
# None inherits WORKERS / --workers. Tune one stage at a time.
STEP_WORKERS = {2: None, 3: None, 5: 10, 6: None}
L0C_SOURCE_ROOT = Path(r'Y:\soc\l0c\version02')
GLARE_CSV = None  # None: SCRIPT_DIR/csv/all_months.csv
SAVE_L0C = False  # Save processed L0C to SOC_ROOT/l0c/YYYY/MM
SAVE_L0D = False
SAVE_L1F = False
SAVE_L1R = True
CONTINUE_ON_ERROR = True
OVERWRITE = True  # Set True to reprocess and replace matching outputs
START_STEP = 0  # 0: copy/remove glare; 1: existing L0C; 2: existing L0C + L0D
RESOLUTIONS = (2,10)
MLSPB_MODE = 'boxes'  # 'trim' restores legacy frame trimming
MIN_CLEAN_FRAMES = 100  # Minimum reference interval; tune after validation
EXTRA_ARGS = {2:['--mlspb-mode', MLSPB_MODE, '--min-clean-frames', str(MIN_CLEAN_FRAMES)],3:[],5:[],6:[]}
# -----------------------------------------------------------------


def configure_core():
    for name in ('SCRIPTS','PYTHON_EXE','ML_PYTHON_EXE','RESOLUTIONS','EXTRA_ARGS'):
        setattr(core,name,globals()[name])


def command_for(step, paths, args, orbit_count):
    workers = getattr(args, "step_workers", {}).get(step, args.workers)
    command = core.make_command(step,paths,workers)
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


# In-memory all-filter findings survive intermediate-file cleanup.
QUALITY_RESULTS = {}


def collect_quality(paths, jobs):
    for job in jobs:
        orbit = job['orbit']
        reports = [p for p in orbit_files(paths['l1f'], orbit)
                   if p.name.endswith('.bad_frames.csv')]
        if reports:
            rows = []
            for report in reports:
                with report.open(newline='', encoding='utf-8-sig') as stream:
                    rows.extend(csv.DictReader(stream))
            # Archive previews before retries, failures or workspace cleanup.
            run_root = Path(job['record']['logs']).parents[2]
            for row in rows:
                source = Path(row['image_path']) if row.get('image_path') else None
                if source is not None and source.is_file():
                    target = run_root/'bad_frames'/source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    row['image_path'] = str(target)
            QUALITY_RESULTS[orbit] = rows


def write_run_quality(run_logs, records):
    fields = ['orbit','image_path','pipeline_status','filter','screening_status',
              'frames_screened_per_filter','removed_frames_all_filters',
              'original_frame_index','epoch','reasons','review_flag','action',
              'adjacent_to_gap','dt_previous_s','dt_next_s','footprint_pixels',
              'valid_fraction','zero_fraction','saturation_fraction','median_radiance','p99_radiance']
    target = run_logs/'frame_quality_all_orbits.csv'
    temporary = target.with_suffix('.csv.partial')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for record in records:
            rows = QUALITY_RESULTS.get(record['orbit']) or [dict(filter='all', screening_status='not_confirmed')]
            for row in rows:
                writer.writerow(dict(row, orbit=record['orbit'], pipeline_status=record['status']))
    os.replace(temporary, target)


def cleanup_after(step, paths, orbit, args):
    levels = []
    if step == 2:
        if not (args.start_step == 0 and args.save_l0c): levels.append('l0c')
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
    for level in core.LEVELS:
        if level == 'l0c' and (args.start_step != 0 or not args.save_l0c): continue
        if level == 'l0d' and (args.start_step == 2 or not args.save_l0d): continue
        if level == 'l1f' and not args.save_l1f: continue
        for source in orbit_files(paths[level],orbit):
            if source.suffix not in ('.nc','.mgf'): continue
            if level == 'l1r' and source.suffix == '.nc' and not args.save_l1r: continue
            products.append((source,core.product_target(source,level,paths,permanent)))
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


def isolate_previous_outputs(step, paths, orbit, archive):
    """Start with empty per-orbit stage outputs; preserve old attempts for diagnosis.

    Empty destinations prove that validated products came from this invocation.
    Step 6's master grid is an INPUT and must remain in l1r.
    """
    levels = {1: ('l0d',), 2: ('l1f',), 3: ('l1r',), 4: ('l1c',),
              5: ('l1a',), 6: ('l2a', 'l3a', 'l3c')}[step]
    moved = 0
    for level in levels:
        for source in orbit_files(paths[level], orbit):
            target = archive / orbit / level / source.relative_to(paths[level])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f'Previous-attempt archive already exists: {target}')
            source.rename(target)
            moved += 1
    return moved


def process_batch_steps(jobs, args, permanent, work, batch_logs, checkpoint):
    """Validate fresh outputs per orbit; retry only missing/invalid products."""
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
        collect_quality(source_paths, [job])
        target = paths_in(work/'failed'/job['orbit'])
        relocate_orbit(source_paths,target,job['orbit'])
        job['record'].update(status='failed',error=f'{type(exc).__name__}: {exc}',
                             work=str(work/'failed'/job['orbit']))
        print(f'FAILED orbit {job["orbit"]}: {exc}',flush=True)
        update()
        if not args.continue_on_error: raise exc

    try:
        # Setup must not queue behind heavy CPU stages before GPU work can start.
        yield ('setup', 'setup')
        for job in jobs:
            if job['record']['status'] == 'failed': continue
            job['record'].update(status='running',work=str(work/'active'),logs=str(log_for(job)))
            job['record'].setdefault('steps',[])
            try:
                if not job.get('l0c_prepared'):
                    core.prepare_l0c(job['orbit'],job['group'],paths,args,log_for(job),job['record']['steps'])
                if args.start_step == 2:
                    src = job['group']['l0d']
                    shutil.copy2(src,paths['l0d']/src.name)
                    core.validate_outputs(1,paths,job['orbit'],log_for(job))
                active.append(job)
            except Exception as exc: fail(job,exc)
        update()
        for step in range(max(1,args.start_step),7):
            if not active: break
            waiting_since = time.perf_counter()
            resource = 'gpu' if step in (1,4) else 'cpu'
            yield (resource, step)
            wait_seconds = time.perf_counter()-waiting_since
            print(f'\nBatch {batch_logs.name}, step {step}: {core.LABELS[step]}; orbits '+
                  ', '.join(j['orbit'] for j in active),flush=True)
            tick = time.perf_counter()
            batch_error = None
            archived = sum(isolate_previous_outputs(
                step, paths, job['orbit'], work/'previous_attempts'/f'step_{step}'/'batch')
                for job in active)
            validation_seconds = 0.0
            retry_seconds = 0.0
            retry_count = 0
            reused_after_batch_error = []
            validation_failures = []
            process_tick = time.perf_counter()
            try:
                core.check_inputs(step,paths)
                core.run_logged(command_for(step,paths,args,len(active)),Path(SCRIPTS[step]).resolve().parent,
                                batch_logs/f'step_{step}.log')
            except core.PreparationCancelled:
                raise
            except Exception as exc:
                batch_error = exc
                print(f'Batch step reported: {exc}',flush=True)
                if not args.continue_on_error: raise
            process_seconds = time.perf_counter()-process_tick
            if step == 2: collect_quality(paths, active)
            stage = {'step':step,'resource':resource,'resource_wait_seconds':round(wait_seconds,2),'batch_process_seconds':round(process_seconds,2),
                     'workers':getattr(args, 'step_workers', {}).get(step,args.workers) if step not in (1,4) else None,
                     'archived_previous_files':archived,
                     'orbits':[j['orbit'] for j in active]}
            if batch_error: stage['error'] = f'{type(batch_error).__name__}: {batch_error}'
            stage_records.append(stage)
            survivors = []
            for job in active:
                orbit = job['orbit']
                error = None
                retried = False
                validation_tick = time.perf_counter()
                try:
                    if step == 4: core.normalize_mlcloud_output(paths,orbit)
                    core.validate_outputs(step,paths,orbit,log_for(job))
                except core.PreparationCancelled:
                    raise
                except Exception as exc:
                    error = exc
                    validation_failures.append({'orbit':orbit, 'error':f'{type(exc).__name__}: {exc}'})
                finally:
                    validation_seconds += time.perf_counter()-validation_tick
                if error is None and batch_error is not None:
                    reused_after_batch_error.append(orbit)
                    print(f'Orbit {orbit}: fresh outputs validated; no retry needed.',flush=True)
                if error is not None:
                    if not args.continue_on_error: raise error
                    # expected_outputs checks ALL required products, including every
                    # Step 6 resolution. Retry only this incomplete/invalid orbit.
                    retry_tick = time.perf_counter()
                    retry_count += 1
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
                        isolate_previous_outputs(step,retry_paths,orbit,
                            work/'previous_attempts'/f'step_{step}'/'retry')
                        core.check_inputs(step,retry_paths)
                        core.run_logged(command_for(step,retry_paths,args,1),Path(SCRIPTS[step]).resolve().parent,
                                        retry_logs/f'step_{step}.log')
                        if step == 4: core.normalize_mlcloud_output(retry_paths,orbit)
                        core.validate_outputs(step,retry_paths,orbit,retry_logs)
                    except core.PreparationCancelled:
                        raise
                    except Exception as exc:
                        fail(job,exc,retry_paths)
                        continue
                    finally:
                        retry_seconds += time.perf_counter()-retry_tick
                    if step == 2: collect_quality(retry_paths, [job])
                    relocate_orbit(retry_paths,paths,orbit)
                    shutil.rmtree(retry_root)
                job['record']['steps'].append({'step':step,'validated':True,'retried':retried,
                                                'batch_log':str(batch_logs/f'step_{step}.log')})
                cleanup_after(step,paths,orbit,args)
                survivors.append(job)
            active = survivors
            stage.update(validation_seconds=round(validation_seconds,2),
                         retry_seconds=round(retry_seconds,2), retry_count=retry_count,
                         retained_after_batch_error=reused_after_batch_error,
                         validation_failures=validation_failures)
            stage['stage_total_seconds'] = round(time.perf_counter()-tick,2)
            # Preserve timing/error CSVs before workspace cleanup, including L2A subfolders.
            for level,folder in paths.items():
                for report in folder.rglob('*.csv'):
                    dst = batch_logs/'reports'/level/report.relative_to(folder)
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(report,dst)
            update()
        yield ('cpu', 'save')
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
                job['record'].update(status='interrupted' if isinstance(exc,(core.PreparationCancelled,KeyboardInterrupt,GeneratorExit)) else 'aborted',
                                     error=f'{type(exc).__name__}: {exc}',work=str(work))
        raise
    finally:
        update()
    if all(j['record']['status']=='completed' for j in jobs):
        shutil.rmtree(work)
    else:
        print(f'Failed-orbit temporary data retained under {work}',flush=True)


def process_batch(jobs, args, permanent, work, batch_logs, checkpoint):
    # Compatibility path: execute the same stages serially.
    for _ in process_batch_steps(jobs,args,permanent,work,batch_logs,checkpoint):
        pass


def prepare_batch(batch, args, stop):
    """Worker owns private job records and files; never writes shared summary."""
    jobs, work, logs = batch
    jobs = copy.deepcopy(jobs)
    paths = paths_in(work/'active')
    error = None
    core.LOG_CONTEXT.stop_event = stop
    try:
        for folder in paths.values(): folder.mkdir(parents=True,exist_ok=True)
        for job in jobs:
            if stop.is_set(): raise core.PreparationCancelled('Background preparation cancelled')
            logdir = logs/'orbits'/job['orbit']
            logdir.mkdir(parents=True,exist_ok=True)
            job['record'].update(status='preparing',work=str(work/'active'),logs=str(logdir),steps=[])
            (logdir/'status.json').write_text(json.dumps(job['record'],indent=2),encoding='utf-8')
            try:
                core.prepare_l0c(job['orbit'],job['group'],paths,args,logdir,job['record']['steps'])
                job['l0c_prepared'] = True
                job['record']['status'] = 'prepared'
            except core.PreparationCancelled:
                raise
            except Exception as exc:
                target = paths_in(work/'failed'/job['orbit'])
                relocate_orbit(paths,target,job['orbit'])
                job['record'].update(status='failed',error=f'{type(exc).__name__}: {exc}',
                                     work=str(work/'failed'/job['orbit']))
                print(f'FAILED preparation orbit {job["orbit"]}: {exc}',flush=True)
                if not args.continue_on_error: raise
    except BaseException as exc:
        error = exc
        for job in jobs:
            if job['record']['status'] in ('pending','preparing','prefetching'):
                job['record'].update(status='interrupted' if isinstance(exc,(core.PreparationCancelled,KeyboardInterrupt)) else 'aborted',
                                     error=f'{type(exc).__name__}: {exc}',work=str(work))
    finally:
        del core.LOG_CONTEXT.stop_event
        for job in jobs:
            logdir = logs/'orbits'/job['orbit']
            logdir.mkdir(parents=True,exist_ok=True)
            (logdir/'status.json').write_text(json.dumps(job['record'],indent=2),encoding='utf-8')
    return jobs, error


def merge_preparation(batch, result):
    prepared, error = result
    for original, ready in zip(batch[0],prepared):
        original['record'].clear()
        original['record'].update(ready['record'])
        original['l0c_prepared'] = ready.get('l0c_prepared',False)
    return error


def run_batches_serial(batches, args, permanent, checkpoint):
    """One consumer plus at most one ahead-of-consumer preparation batch."""
    batches = iter(batches)
    if args.start_step != 0 or not args.prefetch_step0:
        for jobs,work,logs in batches:
            process_batch(jobs,args,permanent,work,logs,checkpoint)
        return
    stop = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='awe-step0')
    future = None
    preparing = None
    def submit(batch):
        for job in batch[0]:
            job['record'].update(status='prefetching',work=str(batch[1]/'active'),logs=str(batch[2]/'orbits'/job['orbit']))
        checkpoint()
        print(f'Preparing step 0 for {batch[2].name}',flush=True)
        return executor.submit(prepare_batch,batch,args,stop)
    try:
        current = next(batches,None)
        if current is None: return
        preparing = current
        future = submit(current)
        while current is not None:
            error = merge_preparation(current,future.result())
            future = None
            checkpoint()
            if error is not None: raise error
            following = next(batches,None)
            if following is not None:
                preparing = following
                future = submit(following)
            jobs,work,logs = current
            process_batch(jobs,args,permanent,work,logs,checkpoint)
            current = following
    finally:
        stop.set()
        # Cancellation is polled even while a child is silent. Wait until no
        # background child can write into the retained workspace before return.
        executor.shutdown(wait=True,cancel_futures=True)
        if future is not None and not future.cancelled():
            merge_preparation(preparing,future.result())
        checkpoint()


def run_batches(batches, args, permanent, checkpoint):
    """Central dispatcher: one preparation, setup, GPU and CPU task at a time.

    Worker-owned records are copied into a queue. Only the dispatcher updates
    the global summary records, avoiding concurrent JSON serialization/mutation.
    A CPU/GPU reservation covers the stage's retries, validation and cleanup.
    """
    if not args.overlap_batches:
        return run_batches_serial(batches,args,permanent,checkpoint)
    batches = iter(batches)
    stop = threading.Event()
    snapshots = queue.Queue()
    originals = {}
    states = []
    running = {}
    pending = None
    exhausted = False
    timeline = []
    run_root = None
    origin = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=4,thread_name_prefix='awe-scheduler')

    def merge_snapshots():
        changed = False
        while True:
            try: key, records = snapshots.get_nowait()
            except queue.Empty: break
            for job, record in zip(originals[key][0],records):
                job['record'].clear(); job['record'].update(record)
            changed = True
        if changed: checkpoint()

    def save_timeline():
        if run_root is None: return
        target = run_root/'scheduler_timing.json'
        temporary = target.with_suffix('.json.partial')
        temporary.write_text(json.dumps({'elapsed_seconds':round(time.perf_counter()-origin,3),
            'max_processing_batches':args.max_processing_batches,
            'resource_limits':{'prepare':1,'setup':1,'gpu':1,'cpu':1},'tasks':timeline},indent=2),encoding='utf-8')
        os.replace(temporary,target)

    def advance(state):
        core.LOG_CONTEXT.stop_event = stop
        try:
            if stop.is_set(): raise core.PreparationCancelled('Scheduler stopped')
            try: return next(state['generator'])
            except StopIteration: return None
        finally:
            del core.LOG_CONTEXT.stop_event

    def record_task(resource, batch, step, waiting=0):
        item = dict(batch=batch[2].name,resource=resource,step=step,
                    wait_seconds=round(waiting,3),started_seconds=round(time.perf_counter()-origin,3),status='running')
        timeline.append(item)
        print(f'[scheduler] {batch[2].name}: {resource} {step}',flush=True)
        return item

    try:
        while True:
            merge_snapshots()
            # Finish resource tasks before dispatching any replacements.
            for future in list(running):
                if not future.done(): continue
                info = running.pop(future)
                item = info['timing']
                item['finished_seconds'] = round(time.perf_counter()-origin,3)
                item['elapsed_seconds'] = round(item['finished_seconds']-item['started_seconds'],3)
                try:
                    result = future.result()
                    if info['resource'] == 'prepare':
                        error = merge_preparation(info['batch'],result)
                        checkpoint()
                        if error is not None: raise error
                        pending['ready'] = True
                    else:
                        state = info['state']; state['request'] = result
                        state['ready_since'] = time.perf_counter()
                        if result is None: states.remove(state)
                    item['status'] = 'completed'
                except BaseException as exc:
                    item.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                    raise
            merge_snapshots()
            # Admit oldest prepared batch. At most N processing plus one prepared.
            if pending is not None and pending['ready'] and len(states)<args.max_processing_batches:
                batch = pending['batch']; pending = None
                local_jobs = copy.deepcopy(batch[0]); key = str(batch[1])
                def publish(jobs=local_jobs, k=key):
                    snapshots.put((k,copy.deepcopy([j['record'] for j in jobs])))
                state = dict(batch=batch,ready_since=time.perf_counter(),request=None)
                state['generator'] = process_batch_steps(local_jobs,args,permanent,batch[1],batch[2],publish)
                # Prime only: the first yield precedes all stage work.
                state['request'] = next(state['generator'])
                states.append(state)
            # Never prefetch more than one batch; generator selection stays on this thread.
            prefetch_allowed = args.prefetch_step0 or len(states)<args.max_processing_batches
            if pending is None and not exhausted and prefetch_allowed:
                batch = next(batches,None)
                if batch is None:
                    exhausted = True
                else:
                    originals[str(batch[1])] = batch
                    run_root = batch[2].parent; run_root.mkdir(parents=True,exist_ok=True)
                    for job in batch[0]:
                        job['record'].update(status='prefetching',work=str(batch[1]/'active'),
                                             logs=str(batch[2]/'orbits'/job['orbit']))
                    checkpoint()
                    pending = dict(batch=batch,ready=False)
                    item = record_task('prepare',batch,0 if args.start_step==0 else 'input_copy')
                    future = executor.submit(prepare_batch,batch,args,stop)
                    running[future] = dict(resource='prepare',batch=batch,timing=item)
            busy = {info['resource'] for info in running.values()}
            active_ids = {id(info['state']) for info in running.values() if 'state' in info}
            for resource in ('setup','gpu','cpu'):
                if resource in busy: continue
                # States remain in admission order: oldest ready batch wins.
                state = next((s for s in states if id(s) not in active_ids and
                              s['request'] is not None and s['request'][0]==resource),None)
                if state is None: continue
                item = record_task(resource,state['batch'],state['request'][1],
                                   time.perf_counter()-state['ready_since'])
                future = executor.submit(advance,state)
                running[future] = dict(resource=resource,state=state,timing=item)
                busy.add(resource)
            save_timeline()
            if exhausted and pending is None and not states and not running: break
            if running:
                wait(tuple(running),timeout=0.25,return_when=FIRST_COMPLETED)
    finally:
        stop.set()
        # run_logged sees this event and terminates its full child process tree.
        executor.shutdown(wait=True,cancel_futures=True)
        for future, info in running.items():
            if future.cancelled(): continue
            try:
                result = future.result()
                if info['resource']=='prepare': merge_preparation(info['batch'],result)
                info['timing']['status']='completed'
            except BaseException as exc:
                info['timing'].update(status='interrupted',error=f'{type(exc).__name__}: {exc}')
            info['timing']['finished_seconds']=round(time.perf_counter()-origin,3)
            info['timing']['elapsed_seconds']=round(info['timing']['finished_seconds']-info['timing']['started_seconds'],3)
        for state in states:
            state['generator'].close()
        merge_snapshots()
        checkpoint()
        save_timeline()
        if run_root is not None:
            print(f'Scheduler timing: {run_root / "scheduler_timing.json"}',flush=True)
            print(f'Scheduler elapsed: {time.perf_counter()-origin:.1f}s (includes preparation and saving)',flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year',type=int,default=YEAR)
    p.add_argument('--month',type=int,default=MONTH)
    p.add_argument('--soc-root',type=Path,default=SOC_ROOT)
    p.add_argument('--save-to-server',action=argparse.BooleanOptionalAction,default=SAVE_TO_SERVER)
    p.add_argument('--server-root',type=Path,default=SERVER_ROOT)
    p.add_argument('--version-folder',default=VERSION_FOLDER)
    p.add_argument('--logs-root',type=Path,default=SCRIPT_DIR/'pipeline_logs')
    p.add_argument('--l0c-source-root',type=Path,default=L0C_SOURCE_ROOT)
    p.add_argument('--glare-csv',type=Path,default=GLARE_CSV)
    p.add_argument('--save-l0c',action=argparse.BooleanOptionalAction,default=SAVE_L0C)
    #p.add_argument('--work-root',type=Path)
    p.add_argument(
        '--work-root',
        type=Path,
        default=Path(r'E:\_orbit_work') if Path('E:/').is_dir()
        else SCRIPT_DIR / '_orbit_work',
    )
    p.add_argument('--orbits',nargs='+',type=int)
    p.add_argument('--limit',type=int,help='Maximum pending orbit attempts, including failures')
    p.add_argument('--orbits-at-a-time','--batch-size',dest='orbits_at_a_time',type=int,default=ORBITS_AT_A_TIME)
    p.add_argument('--workers',type=int,default=WORKERS)
    for step in (2,3,5,6):
        p.add_argument(f'--step-{step}-workers',type=int,default=STEP_WORKERS[step],
                       help=f'File workers for step {step}; default inherits --workers')
    p.add_argument('--overlap-batches',action=argparse.BooleanOptionalAction,default=OVERLAP_BATCHES)
    p.add_argument('--max-processing-batches',type=int,default=MAX_PROCESSING_BATCHES)
    p.add_argument('--prefetch-step0',action=argparse.BooleanOptionalAction,default=PREFETCH_STEP0)
    p.add_argument('--start-step',type=int,choices=(0,1,2),default=START_STEP)
    for name,value in [('save-l0d',SAVE_L0D),('save-l1f',SAVE_L1F),('save-l1r',SAVE_L1R),
                       ('continue-on-error',CONTINUE_ON_ERROR)]:
        p.add_argument('--'+name,action=argparse.BooleanOptionalAction,default=value)
    p.add_argument('--overwrite',action=argparse.BooleanOptionalAction,default=OVERWRITE)
    p.add_argument('--dry-run',action='store_true')
    args = p.parse_args(argv)
    if not 1 <= args.month <= 12 or not 1 <= args.year <= 9999: p.error('Invalid year/month')
    if min(args.workers,args.orbits_at_a_time) < 1 or (args.limit is not None and args.limit < 1):
        p.error('workers, orbits-at-a-time and limit must be positive')
    args.step_workers = {step: (getattr(args,f'step_{step}_workers')
                               if getattr(args,f'step_{step}_workers') is not None else args.workers)
                         for step in (2,3,5,6)}
    if any(value < 1 for value in args.step_workers.values()):
        p.error('Per-stage workers must be positive')
    if args.max_processing_batches < 1: p.error('max-processing-batches must be positive')
    if args.orbits and any(o < 0 or o > 99999 for o in args.orbits): p.error('Orbit must be 0..99999')
    configure_core()
    args.soc_root = args.soc_root.expanduser().resolve()
    permanent = core.permanent_paths(args)
    core.print_storage(permanent, args)
    requested = {f'{o:05d}' for o in args.orbits} if args.orbits else None
    groups = core.discover_orbits(core.input_folder(args,permanent),requested)
    work_root = (args.work_root or SCRIPT_DIR/'_orbit_work').expanduser().resolve()
    print(f'Batch size={args.orbits_at_a_time}; total file workers={args.workers}; '
          f'save L0C/L0D/L1F/L1R={args.save_l0c}/{args.save_l0d}/{args.save_l1f}/{args.save_l1r}',flush=True)
    print(f'CPU file workers by step: {args.step_workers}',flush=True)
    print(f'Step 0 prefetch: {args.prefetch_step0 and args.start_step == 0} (at most one next batch)',flush=True)
    print(f'Stage overlap={args.overlap_batches}; processing batches={args.max_processing_batches}; '
          'GPU slots=1; CPU slots=1; preparation slots=1; setup slots=1',flush=True)
    if args.dry_run:
        selected = list(groups)[:args.limit]
        for start in range(0,len(selected),args.orbits_at_a_time):
            batch = selected[start:start+args.orbits_at_a_time]
            print('Batch: '+', '.join(batch))
            paths = paths_in(work_root/f'preview_batch_{start//args.orbits_at_a_time+1}')
            if args.start_step == 0:
                for orbit in batch: print(subprocess.list2cmdline(core.make_glare_command(groups[orbit],paths,args)))
            for step in range(max(1,args.start_step),7):
                print(subprocess.list2cmdline(command_for(step,paths,args,len(batch))))
        print('Dry run: no files changed. Completion receipts are checked during a real run.')
        return 0
    for step in range(args.start_step,7):
        if not Path(SCRIPTS[step]).is_file(): raise FileNotFoundError(f'Step {step}: {SCRIPTS[step]}')
    for exe in (PYTHON_EXE,ML_PYTHON_EXE):
        if not Path(exe).is_file(): raise FileNotFoundError(f'Python interpreter: {exe}')
    receipts = SCRIPT_DIR/'pipeline_orbit_state'/f'{args.year:04d}'/f'{args.month:02d}'
    run_logs = args.logs_root.expanduser().resolve()/f'{args.year:04d}'/f'{args.month:02d}'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_logs.mkdir(parents=True,exist_ok=True)
    records = []
    QUALITY_RESULTS.clear()
    checkpoint_lock = threading.Lock()
    def checkpoint():
        with checkpoint_lock:
            (run_logs/'summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            write_run_quality(run_logs, records)
    def selected_batches():
        pending = []
        attempted = 0
        batch_number = 0
        def make_batch(jobs):
            nonlocal batch_number
            batch_number += 1
            work = work_root/f'{args.year:04d}{args.month:02d}_batch_{batch_number:04d}_{uuid.uuid4().hex}'
            return list(jobs),work,run_logs/f'batch_{batch_number:04d}'
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
                    levels = (('l0c',) if args.start_step == 0 and args.save_l0c else ()) + (('l0d',) if args.start_step<=1 and args.save_l0d else ()) + ('l1f','l1r','l1c','l1a','l2a','l3a','l3c')
                    existing = [f for level in levels for f in orbit_files(permanent[level],orbit)
                                if f.suffix in ('.nc','.mgf')]
                    master = permanent['master']/f'Master{orbit}.mgf'
                    if master.exists() and master not in existing: existing.append(master)
                    if existing:
                        raise FileExistsError(f'Orbit {orbit}: existing outputs without a matching batch receipt; '
                                              'select new orbits or use --overwrite to reprocess')
                pending.append({'orbit':orbit,'group':group,'record':record,'receipt':receipt,'signature':signature})
            except Exception as exc:
                record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                print(f'FAILED orbit {orbit}: {exc}',flush=True)
                if not args.continue_on_error: raise
            finally: checkpoint()
            if len(pending) == args.orbits_at_a_time:
                yield make_batch(pending)
                pending = []
        if pending: yield make_batch(pending)
    try:
        run_batches(selected_batches(),args,permanent,checkpoint)
    finally:
        for record in records:
            if record['status'] in ('pending','prefetching','prepared'):
                record['status'] = 'not_started'
                if record.get('logs'):
                    folder = Path(record['logs'])
                    folder.mkdir(parents=True,exist_ok=True)
                    (folder/'status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        checkpoint()
        completed = sum(x['status']=='completed' for x in records)
        failed = [x['orbit'] for x in records if x['status'] in ('failed','aborted')]
        skipped = sum(x['status']=='skipped' for x in records)
        print(f'Finished: {completed} completed; {len(failed)} failed/aborted; {skipped} skipped.',flush=True)
        if failed: print('Failed/aborted orbits: '+', '.join(failed),flush=True)
        print(f'Run summary: {run_logs/"summary.json"}',flush=True)
        print(f'Frame quality report: {run_logs/"frame_quality_all_orbits.csv"}',flush=True)
    return 1 if failed else 0


if __name__ == '__main__': raise SystemExit(main())
