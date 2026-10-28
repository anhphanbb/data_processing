"""Find and convert all CSV-labeled orbits under an L1R directory recursively."""
import argparse
import csv
import os
from pathlib import Path
import re

from convert_cloud_labels_to_epoch import convert, integer


def batch_convert(root, labels, output_dir, filter_name='q20', format_name='nc'):
    root, labels, output_dir = map(Path, (root, labels, output_dir))
    if filter_name not in ('q20', 'bkg', 'p12', 'p14') or format_name not in ('nc', 'csv'):
        raise ValueError('Invalid filter or output format')
    if not root.is_dir():
        raise ValueError(f'L1R directory is unavailable: {root}')
    with labels.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if 'Orbit #' not in (reader.fieldnames or []):
            raise ValueError('Labels require an Orbit # column')
        orbits = {integer(row['Orbit #'].strip()) for row in reader if (row.get('Orbit #') or '').strip()}
    if not orbits:
        raise ValueError('No labeled orbits found')
    candidates = {orbit: [] for orbit in orbits}
    pattern = re.compile(rf'awe_l1r_{filter_name}_\d{{7}}T\d{{4}}_(\d{{5}})_v01\.nc')
    print(f'Scanning {root} for {len(orbits)} labeled orbits ({filter_name}, v01)...', flush=True)
    def scan_error(error):
        raise error
    for folder, _, files in os.walk(root, onerror=scan_error):
        for name in files:
            match = pattern.fullmatch(name)
            if match and int(match[1]) in candidates:
                candidates[int(match[1])].append(Path(folder) / name)
    output_dir.mkdir(parents=True, exist_ok=True)
    # A separate report per invocation preserves previous conversion records.
    from datetime import datetime
    report = output_dir / f'batch_report_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
    counts = dict(converted=0, missing=0, ambiguous=0, error=0)
    with report.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['orbit', 'status', 'source', 'output', 'epochs', 'boxes', 'detail'])
        writer.writeheader()
        for orbit, paths in sorted(candidates.items()):
            row = dict(orbit=orbit, source='; '.join(str(p) for p in sorted(paths)))
            if not paths:
                row.update(status='missing', detail=f'No {filter_name} v01 file found')
            elif len(paths) > 1:
                row.update(status='ambiguous', detail='Multiple files match; narrow --l1r-root to the intended dataset')
            else:
                output = output_dir / f'{paths[0].stem}_cloud_labels.{format_name}'
                row['output'] = str(output)
                try:
                    frames, boxes = convert(paths[0], labels, output)
                    row.update(status='converted', epochs=frames, boxes=boxes)
                except (ValueError, OSError, KeyError, RuntimeError) as exc:
                    row.update(status='error', detail=str(exc))
            counts[row['status']] += 1
            writer.writerow(row)
            stream.flush()
            print(f'Orbit {orbit:05d}: {row["status"]} {row.get("detail", "")}', flush=True)
    print(f'{counts}; report: {report}', flush=True)
    return counts, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--l1r-root', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--filter', choices=['q20', 'bkg', 'p12', 'p14'], default='q20',
                        help='Filter used for original labeling; no automatic fallback (default: q20)')
    parser.add_argument('--format', choices=['nc', 'csv'], default='nc')
    args = parser.parse_args()
    try:
        counts, _ = batch_convert(args.l1r_root, args.labels, args.output_dir, args.filter, args.format)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    if any(counts[key] for key in ('missing', 'ambiguous', 'error')):
        parser.exit(1, 'Some labeled orbits were not converted; see the batch report.\n')


if __name__ == '__main__':
    main()
