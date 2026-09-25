#!/usr/bin/env python3
"""Audit Epoch continuity within each NetCDF file; never read image variables.

Run in Spyder (F5) after editing DEFAULT_* below, or:
  python check_epoch_continuity.py
  python check_epoch_continuity.py --root Y:\\soc\\l0c\\2023\\11
  python check_epoch_continuity.py --pattern '*_q20_*.nc' --cadence 1.1

Requires numpy and netCDF4. Inputs are opened read-only. Files are processed
sequentially, keeping only one Epoch vector in memory. Each run produces a
new report directory. CSVs are UTF-8 with BOM for Windows Excel.

Checks are WITHIN files, not across orbits/filters/versions. All .nc files
are included by default. Normal orbital observation gaps are not assessed.
Epoch values remain in the source time scale: no GPS-to-UTC claim/conversion.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

DEFAULT_ROOT = r"Y:\soc\l0c"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "epoch_continuity_reports"
# Save reports beside this script, independently of the input drive.
DEFAULT_PATTERN = "*.nc"  # Use '*_q20_*.nc' for q20 only
DEFAULT_CADENCE_SECONDS = None  # None: use Cadence attribute, then median dt
DEFAULT_TOLERANCE_SECONDS = 0.05
DEFAULT_GAP_FACTOR = 1.5

FILE_FIELDS = [
    'path', 'orbit', 'filter', 'version', 'status', 'frames', 'valid_frames',
    'epoch_dtype', 'epoch_units', 'seconds_per_epoch_unit', 'cadence_s',
    'cadence_source', 'tolerance_s', 'first_epoch', 'middle_index',
    'middle_epoch', 'last_epoch', 'duration_s', 'first_to_middle_s',
    'middle_to_last_s', 'half_difference_s', 'min_dt_s', 'median_dt_s',
    'max_dt_s', 'gaps', 'estimated_missing_frames', 'duplicates',
    'backward_intervals', 'irregular_intervals', 'invalid_timestamps',
    'warnings', 'error',
]
EVENT_FIELDS = [
    'path', 'orbit', 'filter', 'version', 'event', 'index_before',
    'index_after', 'epoch_before', 'epoch_after', 'dt_s', 'expected_dt_s',
    'deviation_s', 'estimated_missing_frames',
]
NAME_RE = re.compile(
    r'^awe_l\d[a-z]_(?P<filter>[^_]+)_\d{7}T\d{4}_'
    r'(?P<orbit>\d+)_(?P<version>v\d+)', re.I
)


def seconds_per_unit(units):
    """No date conversion: differences need only the unit multiplier."""
    token = str(units).strip().lower().split(' since ')[0].strip()
    scales = {
        's': 1., 'sec': 1., 'second': 1., 'seconds': 1.,
        'ms': .001, 'millisecond': .001, 'milliseconds': .001,
        'us': .000001, 'microsecond': .000001, 'microseconds': .000001,
        'min': 60., 'minute': 60., 'minutes': 60.,
        'h': 3600., 'hour': 3600., 'hours': 3600.,
        'd': 86400., 'day': 86400., 'days': 86400.,
    }
    if token not in scales:
        raise ValueError(f'Unsupported Epoch units: {units!r}')
    return scales[token]


def analyze_file(path, cadence_override=None, tolerance=0.05, gap_factor=1.5):
    row = dict.fromkeys(FILE_FIELDS, '')
    row['path'] = str(path)
    match = NAME_RE.match(Path(path).name)
    if match:
        row.update(match.groupdict())
    events, warnings = [], []
    for name in ('gaps', 'estimated_missing_frames', 'duplicates',
                 'backward_intervals', 'irregular_intervals', 'invalid_timestamps'):
        row[name] = 0
    row['tolerance_s'] = tolerance
    try:
        with Dataset(path, 'r') as ds:
            if 'Epoch' not in ds.variables:
                raise ValueError('Epoch variable not found in root group')
            var = ds.variables['Epoch']
            if var.ndim != 1 or np.dtype(var.dtype).kind not in 'iuf':
                raise ValueError(f'Epoch must be a numeric 1-D vector: {var.shape}, {var.dtype}')
            row['epoch_dtype'] = str(var.dtype)
            # This is the only variable read; all other access is metadata.
            values = np.ma.asarray(var[:], dtype=np.float64)
            times = np.asarray(values.filled(np.nan), dtype=np.float64)
            for name in ('FILLVAL', '_FillValue', 'missing_value'):
                if name in var.ncattrs():
                    for fill in np.atleast_1d(var.getncattr(name)):
                        try:
                            times[times == float(fill)] = np.nan
                        except (TypeError, ValueError):
                            pass
            units = getattr(var, 'units', getattr(var, 'UNITS', ''))
            if isinstance(units, bytes):
                units = units.decode('utf-8')
            if not str(units).strip():
                units = 'seconds (assumed; missing units)'
                scale = 1.
                warnings.append('Epoch units absent; assuming seconds for AWE')
            else:
                scale = seconds_per_unit(units)
            row['epoch_units'] = str(units)
            row['seconds_per_epoch_unit'] = scale
            cadence_attr = getattr(ds, 'Cadence', None)
        n = len(times)
        valid = np.isfinite(times)
        row.update(frames=n, valid_frames=int(valid.sum()),
                   invalid_timestamps=int((~valid).sum()))
        if n:
            mid = (n - 1) // 2
            row['middle_index'] = mid
            for name, index in [('first_epoch', 0), ('middle_epoch', mid), ('last_epoch', n-1)]:
                if valid[index]:
                    row[name] = float(times[index])
            for name, i, j in [('duration_s', 0, n-1), ('first_to_middle_s', 0, mid),
                               ('middle_to_last_s', mid, n-1)]:
                if valid[i] and valid[j]:
                    row[name] = float((times[j] - times[i]) * scale)
            if all(valid[[0, mid, n-1]]):
                row['half_difference_s'] = row['first_to_middle_s'] - row['middle_to_last_s']
        dt = np.diff(times) * scale
        usable = np.isfinite(dt)
        positive = dt[usable & (dt > 0)]
        cadence = cadence_override
        source = 'override' if cadence is not None else ''
        if cadence is None and cadence_attr is not None:
            try:
                cadence = float(cadence_attr)  # AWE Cadence is in seconds.
                if not np.isfinite(cadence) or cadence <= 0:
                    raise ValueError('nonpositive/nonfinite cadence')
                source = 'Cadence attribute (seconds)'
            except (TypeError, ValueError):
                cadence = None
                warnings.append('Invalid Cadence attribute; using median positive dt')
        if cadence is None and positive.size:
            cadence = float(np.median(positive))
            source = 'median positive dt (inferred)'
            warnings.append('Cadence inferred; systematic missing frames may be undetectable')
        if cadence is not None:
            row.update(cadence_s=cadence, cadence_source=source)
        if usable.any():
            row.update(min_dt_s=float(dt[usable].min()),
                       median_dt_s=float(np.median(dt[usable])), max_dt_s=float(dt[usable].max()))
        identity = {k: row[k] for k in ('path', 'orbit', 'filter', 'version')}
        for i in np.flatnonzero(~valid):
            events.append(dict(identity, event='invalid_timestamp', index_after=int(i)))
        for i in np.flatnonzero(usable):
            delta = float(dt[i])
            missing = ''
            if delta < 0:
                kind, count = 'backward', 'backward_intervals'
            elif delta == 0:
                kind, count = 'duplicate', 'duplicates'
            elif cadence is not None and delta > max(gap_factor*cadence, cadence+tolerance):
                kind, count = 'gap', 'gaps'
                missing = max(1, int(np.floor(delta/cadence + .5))-1)
                row['estimated_missing_frames'] += missing
            elif cadence is not None and abs(delta-cadence) > tolerance:
                kind, count = 'irregular_interval', 'irregular_intervals'
            else:
                continue
            row[count] += 1
            events.append(dict(identity, event=kind, index_before=int(i), index_after=int(i+1),
                               epoch_before=float(times[i]), epoch_after=float(times[i+1]),
                               dt_s=delta, expected_dt_s=cadence if cadence is not None else '',
                               deviation_s=delta-cadence if cadence is not None else '',
                               estimated_missing_frames=missing))
        if events:
            row['status'] = 'ISSUES'
        elif n < 2 or cadence is None:
            row['status'] = 'INSUFFICIENT_DATA'
        else:
            row['status'] = 'CONTINUOUS'
    except Exception as exc:
        row.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
    row['warnings'] = '; '.join(warnings)
    return row, events


def discover(root, output_root, pattern, onerror):
    """Stream directory traversal; don't keep an entire drive listing in memory."""
    excluded = output_root.resolve()
    for directory, dirs, files in os.walk(root, onerror=onerror, followlinks=False):
        dirs[:] = sorted(d for d in dirs if (Path(directory)/d).resolve() != excluded)
        for name in sorted(files):
            if fnmatch.fnmatchcase(name.lower(), pattern.lower()):
                yield Path(directory)/name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--pattern', default=DEFAULT_PATTERN)
    parser.add_argument('--cadence', type=float, default=DEFAULT_CADENCE_SECONDS, help='Expected interval in seconds')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE_SECONDS)
    parser.add_argument('--gap-factor', type=float, default=DEFAULT_GAP_FACTOR)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f'Input directory is unavailable: {args.root}')
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error('--tolerance must be finite and >= 0')
    if not np.isfinite(args.gap_factor) or args.gap_factor <= 1:
        parser.error('--gap-factor must be finite and > 1')
    if args.cadence is not None and (not np.isfinite(args.cadence) or args.cadence <= 0):
        parser.error('--cadence must be finite and > 0')
    output_root = Path(args.output) if args.output is not None else Path(__file__).resolve().parent/'epoch_continuity_reports'
    out = output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    statuses, event_counts = Counter(), Counter()
    scanned = warned = missing_total = 0
    interrupted = False
    traversal_errors = []
    print(f'Scanning {args.root} recursively; pattern={args.pattern}', flush=True)
    print(f'Reports: {out}', flush=True)
    with (out/'files.csv').open('w', newline='', encoding='utf-8-sig') as fs, \
         (out/'issues.csv').open('w', newline='', encoding='utf-8-sig') as ins, \
         (out/'events.csv').open('w', newline='', encoding='utf-8-sig') as es:
        all_writer = csv.DictWriter(fs, fieldnames=FILE_FIELDS)
        issue_writer = csv.DictWriter(ins, fieldnames=FILE_FIELDS)
        event_writer = csv.DictWriter(es, fieldnames=EVENT_FIELDS)
        for writer in (all_writer, issue_writer, event_writer):
            writer.writeheader()
        try:
            for path in discover(args.root, output_root, args.pattern, lambda e: traversal_errors.append(str(e))):
                row, events = analyze_file(path, args.cadence, args.tolerance, args.gap_factor)
                all_writer.writerow(row)
                if row['status'] != 'CONTINUOUS' or row['warnings']:
                    issue_writer.writerow(row)
                event_writer.writerows(events)
                scanned += 1
                statuses[row['status']] += 1
                event_counts.update(e['event'] for e in events)
                warned += bool(row['warnings'])
                missing_total += row['estimated_missing_frames']
                if scanned % 100 == 0 or row['status'] != 'CONTINUOUS':
                    print(f'[{scanned}] {row["status"]}: {path.name} | gaps={row["gaps"]} '
                          f'duplicates={row["duplicates"]} backward={row["backward_intervals"]}'
                          f'{" | " + row["error"] if row["error"] else ""}', flush=True)
                for handle in (fs, ins, es):
                    handle.flush()
        except KeyboardInterrupt:
            interrupted = True
    partial = interrupted or bool(traversal_errors)
    lines = [
        'AWE Epoch continuity audit', f'Root: {args.root}', f'Pattern: {args.pattern}',
        f'Scan status: {"PARTIAL" if partial else "FINISHED"}', f'Files scanned: {scanned}',
        *[f'{key}: {statuses[key]}' for key in ('CONTINUOUS', 'ISSUES', 'INSUFFICIENT_DATA', 'ERROR')],
        f'Files with warnings: {warned}', f'Events: {dict(event_counts)}',
        f'Estimated missing frames summed over files: {missing_total}',
        f'Cadence: {args.cadence if args.cadence is not None else "Cadence attribute, otherwise inferred median"}',
        f'Interval tolerance: {args.tolerance} seconds',
        f'Gap: dt > max({args.gap_factor} * cadence, cadence + tolerance)',
        'All frame indices are zero-based. Epoch values use source units/time scale.',
        'Curve/travel distances and image arrays are not read or calculated.',
        'Continuity is tested within each file; inter-orbit gaps are not assessed.',
        'Missing frames are estimates, not proof of loss/removal or its cause.',
        'Counts include every matching filter/version; do not interpret as unique missing observations.',
        'files.csv: every file; issues.csv: problems/warnings; events.csv: individual events.',
        'Reports contain completed-file results even if interrupted.',
    ]
    if not scanned:
        lines.append('WARNING: No matching files were scanned; nothing has been verified.')
    if traversal_errors:
        lines.extend(['Directory traversal errors:', *traversal_errors])
    summary = '\n'.join(lines) + '\n'
    (out/'summary.txt').write_text(summary, encoding='utf-8')
    print('\n'+summary, flush=True)
    print(f'Reports saved to {out}', flush=True)
    return 2 if partial or not scanned or statuses['ERROR'] else (1 if statuses['ISSUES'] or statuses['INSUFFICIENT_DATA'] else 0)


if __name__ == '__main__':
    main()  # Spyder-friendly: no SystemExit; use the printed report for status.
