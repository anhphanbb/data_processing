#!/usr/bin/env python3
r"""Check v01 products for orbits with events in v01 L0C files.

Run beside check_epoch_continuity.py (F5 in Spyder), or:
  python trace_epoch_event_products.py --events "D:\reports\events.csv"
Writes two CSVs plus orbit_photos/<year>_<orbit>_report_001.png, etc.
Matching original L3C PNGs are also copied into the same orbit_photos folder.
Requires Pillow: python -m pip install pillow
No NetCDF arrays are read. A nonempty matching filename means FOUND, not
verified processing success or survival of the affected frames.

Matching: filename year + orbit + filter + v01, allowing changed start time.
L1F/L1R/L1A require all four filters; L3C requires q20. Any resolution counts.
Events with identical type, zero-based frame indices and timing values
(rounded to 6 decimals) are combined across filters. This compares frame
patterns, not absolute event times: filter exposures can be staggered.
Event counts are unique reported patterns, not summed across filters.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVENTS_CSV = None  # Or r'D:\reports\events.csv'; use the original detailed report.
SOC_ROOT = Path(r'Y:\soc')
OUTPUT_ROOT = HERE / 'epoch_product_reports'
L3C_PNG_ROOT = None  # Default: SOC_ROOT / 'l3c'; searches all subfolders.
FILTERS = ('bkg', 'p12', 'p14', 'q20')
LEVELS = ('l1f', 'l1r', 'l1a', 'l3c')
EVENT_TYPES = {'gap', 'duplicate', 'backward', 'irregular_interval', 'invalid_timestamp'}
EVENT_FIELDS = ('event', 'index_before', 'index_after', 'dt_s', 'expected_dt_s',
                'deviation_s', 'estimated_missing_frames')
NAME = re.compile(r'^awe_(l\d[a-z])_([^_]+)_(\d{4})\d{3}T\d{4}_(\d+)_v(\d+).*\.nc$', re.I)


def parse_name(path):
    match = NAME.fullmatch(re.split(r'[\\/]', str(path))[-1])
    if not match:
        return None
    level, filt, year, orbit, version = match.groups()
    return level.lower(), filt.lower(), year, orbit.zfill(5), int(version)


def normalized(value):
    if value is None or not str(value).strip():
        return ''
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f'Invalid numeric event value: {value!r}') from None
    return f'{number:.6f}'.rstrip('0').rstrip('.') if math.isfinite(number) else ''


def read_events(path):
    orbits = defaultdict(lambda: defaultdict(set))
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'path', *EVENT_FIELDS}.issubset(reader.fieldnames or []):
            raise ValueError('Use the original events.csv, which contains frame indices and timing details.')
        for line, row in enumerate(reader, 2):
            meta = parse_name(row['path'])
            if meta is None:
                raise ValueError(f'Line {line}: unrecognized source filename: {row["path"]}')
            level, filt, year, orbit, version = meta
            if level != 'l0c' or version != 1:
                continue
            kind = row['event'].strip()
            if filt not in FILTERS or kind not in EVENT_TYPES:
                raise ValueError(f'Line {line}: unexpected filter/event: {filt}/{kind}')
            # Catch a mislabeled gap such as "duplicate" with dt=3.3 seconds.
            if kind != 'invalid_timestamp':
                dt = float(row['dt_s'])
                if (not math.isfinite(dt) or (kind == 'duplicate' and dt != 0)
                        or (kind == 'backward' and dt >= 0)
                        or (kind in ('gap', 'irregular_interval') and dt <= 0)):
                    raise ValueError(f'Line {line}: {kind} conflicts with dt_s={dt}. Check the input report.')
            key = (kind, *(normalized(row[f]) for f in EVENT_FIELDS[1:]))
            orbits[(year, orbit)][key].add(filt)
    return orbits


def scan_products(root, level, wanted):
    found = defaultdict(set)
    errors = []
    if not root.is_dir():
        return found, f'Unavailable folder: {root}'
    for folder, _, names in os.walk(root, onerror=lambda e: errors.append(str(e))):
        for name in names:
            meta = parse_name(name)
            if not meta:
                continue
            product_level, filt, year, orbit, version = meta
            if product_level != level or version != 1 or (year, orbit) not in wanted:
                continue
            if filt not in (('q20',) if level == 'l3c' else FILTERS):
                continue
            try:
                if (Path(folder) / name).stat().st_size > 0:
                    found[(year, orbit)].add(filt)
                else:
                    errors.append(f'Zero-byte product: {Path(folder) / name}')
            except OSError as exc:
                errors.append(str(exc))
    return found, ' | '.join(errors)


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def event_sort(item):
    e = item[0]
    return (float(e[2]) if e[2] else -1, e)


def scan_pngs(root, wanted):
    matches = defaultdict(list)
    errors = []
    if not root.is_dir():
        return matches, f'PNG folder unavailable: {root}'
    for folder, _, names in os.walk(root, onerror=lambda e: errors.append(str(e))):
        for name in sorted(names):
            if not name.lower().endswith('.png'):
                continue
            meta = parse_name(name[:-4] + '.nc')
            if meta:
                level, filt, year, orbit, version = meta
                if level == 'l3c' and filt == 'q20' and version == 1 and (year, orbit) in wanted:
                    matches[(year, orbit)].append(Path(folder) / name)
    return matches, ' | '.join(errors)


def save_orbit_photos(row, png_paths, folder, png_note=''):
    """Paginate all details and images; never truncate events or crop images."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    folder.mkdir(parents=True, exist_ok=True)
    def font(size):
        for name in ('DejaVuSans.ttf', 'arial.ttf', 'C:/Windows/Fonts/arial.ttf'):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        return ImageFont.load_default()
    body, heading = font(23), font(34)
    width, height, margin = 1600, 2400, 55
    pages, errors = [], []
    canvas = draw = None
    y = 0
    def new_page():
        nonlocal canvas, draw, y
        canvas = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, width, 125), fill='#19364b')
        draw.text((margin, 35), f"Orbit {row['orbit']} | {row['year']} | v01", font=heading, fill='white')
        y = 155
    def finish():
        used_height = min(height, y + 90)
        draw.text((margin, used_height-45), f'Page {len(pages)+1} | L0C Epoch event / product availability report', font=body, fill='#607080')
        path = folder / f"{row['year']}_{row['orbit']}_report_{len(pages)+1:03d}.png"
        canvas.crop((0, 0, width, used_height)).save(path)
        pages.append(str(path))
    def ensure(space):
        if y + space > height-80:
            finish()
            new_page()
    def line(text, color='#243746'):
        nonlocal y
        # Measure actual font width, including long file paths.
        words = str(text)
        while words:
            end = len(words)
            while draw.textlength(words[:end], font=body) > width-2*margin:
                end -= 1
            if end < len(words) and ' ' in words[:end]:
                end = words.rfind(' ', 0, end) or end
            ensure(36)
            draw.text((margin, y), words[:end], font=body, fill=color)
            y += 34
            words = words[end:].lstrip()
    new_page()
    line(f"Events: {row['event_count']} | Gaps: {row['gap_count']} | Largest gap interval: {row['max_gap_dt_s'] or '-'} s")
    line(f"Affected filters: {row['affected_filters']}")
    line(f"Same reported patterns across all four filters: {row['same_events_all_four_filters']}")
    y += 15
    for level in LEVELS:
        missing = row[level+'_missing_filters']
        state = row[level+'_status']
        line(f"{level.upper()}: {state}" + (f" | Not located: {missing}" if missing else ''),
             '#18734b' if state == 'FOUND' else '#a14a21')
    y += 15
    line('EVENT DETAILS (zero-based L0C frame indices)')
    for number, detail in enumerate(row['event_details'].split(' | '), 1):
        line(f'{number}. {detail}')
        y += 8
    line('dt = full interval; deviation = dt minus expected interval.')
    line('FOUND means matching files exist; missing intermediates may have been deleted.')
    if row['scan_notes']:
        line('Product scan notes: ' + row['scan_notes'], '#a14a21')
    if png_note:
        line('PNG scan notes: ' + png_note, '#a14a21')
    y += 15
    line(f'L3C q20 v01 images: {len(png_paths)} matching files')
    if not png_paths:
        line('No matching PNG located.' if not png_note else 'PNG availability unknown: scan incomplete.')
    for number, source in enumerate(sorted(png_paths), 1):
        target = folder / f"{row['year']}_{row['orbit']}_l3c_{number:03d}_{source.name}"
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            errors.append(f'Could not copy {source}: {exc}')
        try:
            with Image.open(source) as im:
                im = ImageOps.exif_transpose(im).convert('RGBA')
                background = Image.new('RGBA', im.size, 'white')
                background.alpha_composite(im)
                preview = background.convert('RGB')
                preview.thumbnail((width-2*margin, 1500), Image.Resampling.LANCZOS)
            ensure(preview.height + 150)
            line(f'Image {number}: {source.name}')
            line(f'Folder: {source.parent}')
            ensure(preview.height + 20)
            canvas.paste(preview, ((width-preview.width)//2, y))
            y += preview.height + 25
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            message = f'Cannot display {source}: {exc}'
            errors.append(message)
            line(message, '#a14a21')
    for error in errors:
        line(error, '#a14a21')
    finish()
    return ';'.join(pages), ' | '.join(errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--events', type=Path, default=EVENTS_CSV)
    parser.add_argument('--soc-root', type=Path, default=SOC_ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--l3c-png-root', type=Path, default=L3C_PNG_ROOT)
    args = parser.parse_args(argv)
    try:
        import PIL
    except ImportError:
        parser.error('PNG reports require Pillow. Install with: python -m pip install pillow')
    events = Path(args.events) if args.events else None
    if events is None:
        choices = sorted((HERE / 'epoch_continuity_reports').glob('*/events.csv'))
        if not choices:
            parser.error('Set EVENTS_CSV or pass --events with the original detailed events.csv.')
        events = choices[-1]
    print(f'Event report: {events}', flush=True)
    orbits = read_events(events)
    products, notes = {}, {}
    for level in LEVELS:
        root = args.soc_root / level
        print(f'Scanning v01 {root}', flush=True)
        products[level], notes[level] = scan_products(root, level, orbits) if orbits else ({}, 'No affected orbits; scan not run')
        if notes[level]:
            print(f'WARNING {level}: {notes[level]}', flush=True)
    png_root = Path(args.l3c_png_root) if args.l3c_png_root else args.soc_root / 'l3c'
    print(f'Scanning L3C PNGs: {png_root}', flush=True)
    pngs, png_note = scan_pngs(png_root, orbits) if orbits else ({}, '')
    rows = []
    for (year, orbit), patterns in sorted(orbits.items()):
        counts = Counter(e[0] for e in patterns)
        details = []
        for e, filters in sorted(patterns.items(), key=event_sort):
            kind, before, after, dt, expected, deviation, missing = e
            frames = f'{before}->{after}' if before else after
            detail = f'{kind}: indices {frames}; filters={"/".join(sorted(filters))}'
            if dt:
                detail += f'; dt={dt} s; expected={expected} s; deviation={deviation} s'
            if kind == 'gap':
                detail += f'; estimated_missing_frames={missing}'
            details.append(detail)
        gap_values = [float(e[3]) for e in patterns if e[0] == 'gap' and e[3]]
        row = dict(year=year, orbit=orbit,
                   affected_filters=';'.join(sorted(set().union(*patterns.values()))),
                   same_events_all_four_filters='YES' if all(f == set(FILTERS) for f in patterns.values()) else 'NO',
                   event_types=';'.join(sorted(counts)), event_count=len(patterns),
                   gap_count=counts['gap'], max_gap_dt_s=max(gap_values) if gap_values else '',
                   event_details=' | '.join(details))
        for level in LEVELS:
            required = {'q20'} if level == 'l3c' else set(FILTERS)
            missing = required - products[level].get((year, orbit), set())
            row[level+'_status'] = 'FOUND' if not missing else ('UNKNOWN' if notes[level] else 'NOT_FOUND')
            row[level+'_missing_filters'] = ';'.join(sorted(missing))
        row['scan_notes'] = ' | '.join(f'{l}: {n}' for l, n in notes.items() if n)
        rows.append(row)
    stats = []
    groups = [('all', 'all', rows)]
    groups += [('event', kind, [r for r in rows if kind in r['event_types'].split(';')])
               for kind in sorted(EVENT_TYPES)]
    groups += [('year', year, [r for r in rows if r['year'] == year])
               for year in sorted({r['year'] for r in rows})]
    for group, label, subset in groups:
        for level in LEVELS:
            counts = Counter(r[level+'_status'] for r in subset)
            known = counts['FOUND'] + counts['NOT_FOUND']
            stats.append(dict(group=group, label=label, level=level, total_orbits=len(subset),
                              v01_found=counts['FOUND'], v01_not_found=counts['NOT_FOUND'], unknown=counts['UNKNOWN'],
                              found_pct_known=round(100*counts['FOUND']/known, 2) if known else '',
                              scan_notes=notes[level]))
    out = args.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    for number, row in enumerate(rows, 1):
        key = (row['year'], row['orbit'])
        matches = pngs.get(key, [])
        row['l3c_png_count'] = len(matches)
        row['report_pngs'], image_errors = save_orbit_photos(
            row, matches, out / 'orbit_photos', png_note)
        row['image_notes'] = ' | '.join(n for n in (png_note, image_errors) if n)
        print(f"PNG report {number}/{len(rows)}: orbit {row['orbit']}", flush=True)
    fields = ['year', 'orbit', 'affected_filters', 'same_events_all_four_filters', 'event_types',
              'event_count', 'gap_count', 'max_gap_dt_s', 'event_details']
    fields += [l+s for l in LEVELS for s in ('_status', '_missing_filters')] + ['scan_notes', 'l3c_png_count', 'report_pngs', 'image_notes']
    write_csv(out/'orbit_products.csv', rows, fields)
    write_csv(out/'statistics.csv', stats, list(stats[0]))
    print(f'\nAffected v01 orbits: {len(rows)}')
    for s in stats[:4]:
        print(f"{s['level']}: found={s['v01_found']}, not found={s['v01_not_found']}, unknown={s['unknown']}")
    print('Frame indices are zero-based L0C indices, not downstream frame numbers.')
    print('Gap dt is the full interval; deviation is dt minus the expected interval.')
    print('Same-events flag compares only reported frame/timing patterns, not absolute times.')
    print('NOT_FOUND means at least one required v01 filter is absent; UNKNOWN means scan incomplete.')
    print('FOUND means nonempty matching files exist; intermediates may have been deleted.')
    print('Statistics count orbits once per group. Event groups overlap; percentages exclude UNKNOWN.')
    print(f'Reports saved: {out}')
    return out


if __name__ == '__main__':
    main()
