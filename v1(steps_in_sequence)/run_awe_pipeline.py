#!/usr/bin/env python3
"""Run the six AWE processing scripts in sequence (Python 3.10+).

Save in D:\\Github\\data_processing and run in Spyder or an Anaconda prompt.
Only YEAR and MONTH need changing for another month. Original scripts are
not edited. Their numerical, worker, model, and overwrite defaults are retained.

Examples:
    python run_awe_pipeline.py
    python run_awe_pipeline.py --year 2024 --month 3
    python run_awe_pipeline.py --dry-run
    python run_awe_pipeline.py --start-step 4
    python run_awe_pipeline.py --start-step 6 --overwrite

--overwrite forces replacement in all selected steps. Without it, each script's
own overwrite setting applies (these settings differ in the supplied scripts).
When changing an upstream algorithm/model, rerun downstream steps with
--overwrite so existing downstream products are not silently reused.

Logs: SOC_ROOT/pipeline_logs/YYYY/MM/<run timestamp>/
The master requires only the standard library. Child interpreters need the
same dependencies as when you run each processing script independently.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

# ----------------------- EDIT SETTINGS HERE -----------------------
YEAR = 2023
MONTH = 11
SOC_ROOT = Path(r'E:\soc')
GITHUB_ROOT = Path(r'D:\Github')

SCRIPTS = {
    1: GITHUB_ROOT / 'solar_panels' / '4_and_5_mlspb.py',
    2: GITHUB_ROOT / 'data_processing' / 'filter_flatfield_to_1lf.py',
    3: GITHUB_ROOT / 'data_processing' / 'regrid_l1f_to_l1r.py',
    4: GITHUB_ROOT / 'cloud_restructured' / '4_and_5_mlcloud.py',
    5: GITHUB_ROOT / 'data_processing' / 'coadd_to_l1a.py',
    6: GITHUB_ROOT / 'data_processing' / 'generate_l2a_l3a_l3c.py',
}
# Usually use the active Anaconda/Spyder environment for every step.
# If ML uses a separate environment, enter its full python.exe path here.
PYTHON_EXE = sys.executable
ML_PYTHON_EXE = sys.executable

START_STEP = 6
END_STEP = 6
DRY_RUN = False
FORCE_OVERWRITE = False  # False retains each child script's own setting.
RESOLUTIONS = (2, 10)  # Existing final script: 2 km and 10 km outputs.
# Folders are named 2x2 and 10x10; these are km resolutions, not bin factors.
# Optional extra CLI options; retain the six scripts' defaults when empty.
EXTRA_ARGS = {2: [], 3: [], 5: [], 6: []}
# -----------------------------------------------------------------

LABELS = {
    1: 'MLSPB: L0C -> L0D',
    2: 'Filter / flat-field: L0D + L0C -> L1F',
    3: 'Regrid: L1F -> L1R + Master grids',
    4: 'MLCloud: L1R q20 -> L1C q20',
    5: 'Co-add: L1R p12/p14/bkg + L1C q20 -> L1A',
    6: 'Temperature / swaths: L1A -> L2A / L3A / L3C',
}


def month_paths(root, year, month):
    return {level: Path(root) / level / f'{year:04d}' / f'{month:02d}'
            for level in ('l0c', 'l0d', 'l1f', 'l1r', 'l1c', 'l1a', 'l2a', 'l3a', 'l3c')}


def make_command(step, paths, overwrite=False):
    script = Path(SCRIPTS[step]).resolve()
    if step in (1, 4):
        source, target = ('l0c', 'l0d') if step == 1 else ('l1r', 'l1c')
        # Only ML scripts are imported by the child bridge. Multiprocessing
        # processing scripts run directly, preserving Windows spawn/Numba behavior.
        return [str(ML_PYTHON_EXE), '-u', str(Path(__file__).resolve()),
                '--ml-child', str(script), str(paths[source]), str(paths[target]),
                '1' if overwrite else '0']
    args = {
        2: [str(paths['l0d']), str(paths['l0c']), str(paths['l1f'])],
        3: [str(paths['l1f']), str(paths['l1r'])],
        5: ['--l1r-input', str(paths['l1r']), '--l1c-input', str(paths['l1c']),
            '--output', str(paths['l1a']), '--mode', 'both'],
        6: ['--l1a-input', str(paths['l1a']), '--l2a-output', str(paths['l2a']),
            '--l3a-output', str(paths['l3a']), '--l3c-output', str(paths['l3c']),
            '--master-input', str(paths['l1r']), '--mode', 'all',
            '--resolutions', *map(str, RESOLUTIONS)],
    }[step]
    return [str(PYTHON_EXE), '-u', str(script), *args,
            *(['--overwrite'] if overwrite else []), *EXTRA_ARGS.get(step, [])]


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


def run_logged(command, cwd, logfile):
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    with logfile.open('w', encoding='utf-8') as log:
        log.write(subprocess.list2cmdline(command) + '\n\n')
        log.flush()
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', errors='replace', bufsize=1) as child:
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
            except BaseException:
                # Stop descendants too when Ctrl+C is used on Windows.
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    child.terminate()
                raise
    if code:
        raise RuntimeError(f'Step exited with code {code}. Log: {logfile}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--year', type=int, default=YEAR)
    parser.add_argument('--month', type=int, default=MONTH)
    parser.add_argument('--soc-root', type=Path, default=SOC_ROOT)
    parser.add_argument('--start-step', type=int, choices=range(1, 7), default=START_STEP)
    parser.add_argument('--end-step', type=int, choices=range(1, 7), default=END_STEP)
    parser.add_argument('--dry-run', action='store_true', default=DRY_RUN)
    parser.add_argument('--overwrite', action='store_true', default=FORCE_OVERWRITE)
    args = parser.parse_args(argv)
    if not 1 <= args.month <= 12 or not 1 <= args.year <= 9999:
        parser.error('Use a valid year and month (month 1..12).')
    if args.start_step > args.end_step:
        parser.error('start-step must not exceed end-step')
    paths = month_paths(args.soc_root.expanduser().resolve(), args.year, args.month)
    steps = list(range(args.start_step, args.end_step + 1))
    commands = {step: make_command(step, paths, args.overwrite) for step in steps}
    print(f'AWE pipeline: {args.year:04d}/{args.month:02d}', flush=True)
    print('Overwrite: ' + ('FORCED for all selected steps' if args.overwrite else 'each script keeps its own default'))
    for step in steps:
        print(f'\n{step}. {LABELS[step]}\n   {subprocess.list2cmdline(commands[step])}')
    if args.dry_run:
        print('\nDry run only; no files changed and no processing started.')
        return 0
    # Catch missing scripts/interpreters before starting expensive processing.
    for step in steps:
        if not Path(SCRIPTS[step]).is_file():
            raise FileNotFoundError(f'Step {step} script not found: {SCRIPTS[step]}')
        if not Path(commands[step][0]).is_file():
            raise FileNotFoundError(f'Python interpreter not found: {commands[step][0]}')
    logdir = (args.soc_root / 'pipeline_logs' / f'{args.year:04d}' / f'{args.month:02d}' /
              datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    try:
        for step in steps:
            tick = time.perf_counter()
            record = {'step': step, 'label': LABELS[step], 'status': 'running',
                      'command': commands[step], 'log': str(logdir / f'step_{step}.log')}
            records.append(record)
            print(f'\nStarting step {step}: {LABELS[step]}', flush=True)
            try:
                check_inputs(step, paths)
                run_logged(commands[step], Path(SCRIPTS[step]).resolve().parent, Path(record['log']))
                record['status'] = 'completed'
            except BaseException as exc:
                record['status'] = 'failed'
                record['error'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                record['seconds'] = round(time.perf_counter() - tick, 2)
                (logdir / 'summary.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
            print(f'Step {step} finished in {record["seconds"]/60:.1f} min.', flush=True)
    finally:
        print(f'\nElapsed: {(time.perf_counter()-started)/60:.1f} min. Logs: {logdir}', flush=True)
    print('All selected steps completed.', flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--ml-child':
        if len(sys.argv) != 6:
            raise SystemExit('Internal ML bridge requires script, source, target, overwrite.')
        run_ml_child(*sys.argv[2:])
    else:
        raise SystemExit(main())
