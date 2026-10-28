"""Convert inclusive frame intervals to explicit v01 epoch labels.

Requires numpy and netCDF4. See epoch_labels.md for usage and semantics.
"""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
import re
import tempfile

import numpy as np
from netCDF4 import Dataset


def integer(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError(f'Expected an integer: {value!r}')
        return int(number)
    except InvalidOperation as exc:
        raise ValueError(f'Expected an integer: {value!r}') from exc


def read_intervals(path, orbit, count, absent_is_clear=False):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if 'Orbit #' not in fields:
            raise ValueError('Labels require an Orbit # column')
        pairs = []
        for field in fields:
            match = re.fullmatch(r'\((\d+),(\d+)\)start', field)
            if match:
                end = field[:-5] + 'end'
                if end not in fields:
                    raise ValueError(f'Missing column {end}')
                pairs.append((f'({match[1]},{match[2]})', field, end))
        if not pairs:
            for y, region in ((2, 'Top'), (1, 'Middle'), (0, 'Bottom')):
                start, end = f'Start ({region})', f'End ({region})'
                if start not in fields or end not in fields:
                    raise ValueError('Unrecognized interval columns')
                pairs.append((f'(7,{y})', start, end))
        cloud = np.zeros((count, len(pairs)), dtype=np.uint8)
        matched = False
        for line, row in enumerate(reader, 2):
            raw_orbit = (row.get('Orbit #') or '').strip()
            if not raw_orbit:
                if any((row.get(k) or '').strip() for _, a, b in pairs for k in (a, b)):
                    raise ValueError(f'Row {line}: intervals without an orbit')
                continue
            if integer(raw_orbit) != orbit:
                continue
            matched = True
            if (row.get('Ground Lights') or '').strip():
                raise ValueError(f'Row {line}: ground-light annotation is not a binary cloud label; use the filtered cloud CSV')
            for box, (_, start, end) in enumerate(pairs):
                a, b = (row.get(start) or '').strip(), (row.get(end) or '').strip()
                if not a and not b:
                    continue
                try:
                    a, b = integer(a), integer(b)
                    if not 0 <= a <= b < count:
                        raise ValueError(f'Interval {a}:{b} outside 0:{count - 1}')
                except ValueError as exc:
                    raise ValueError(f'Row {line}, {start}: {exc}') from exc
                cloud[a:b + 1, box] = 1
        if not matched and not absent_is_clear:
            raise ValueError(f'Orbit {orbit} absent from labels; use --absent-orbit-is-clear only if it was reviewed and is cloud-free')
    return [p[0] for p in pairs], cloud


def convert(l1r, labels, output, absent_is_clear=False):
    l1r, labels, output = map(Path, (l1r, labels, output))
    match = re.fullmatch(r'awe_l1r_(?:q20|bkg|p12|p14)_\d{7}T\d{4}_(\d{5})_v01\.nc', l1r.name)
    if not match:
        raise ValueError('Input must be an original awe_l1r_<filter>_<YYYYDDDThhmm>_<orbit>_v01.nc file')
    orbit = int(match[1])
    if output.suffix.lower() not in ('.csv', '.nc'):
        raise ValueError('Output extension must be .nc or .csv')
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    with Dataset(l1r) as source:
        var = source['Epoch']
        epoch = var[:]
        if epoch.ndim != 1 or not len(epoch) or np.ma.getmaskarray(epoch).any():
            raise ValueError('Epoch must be a nonempty, unmasked one-dimensional array')
        epoch = np.asarray(epoch)
        if epoch.dtype.kind not in 'ifu' or not np.isfinite(epoch).all():
            raise ValueError('Epoch must contain finite numeric values')
        if len(np.unique(epoch)) != len(epoch):
            raise ValueError('Duplicate epochs cannot identify frames uniquely')
        if source['Radiance'].shape[0] != len(epoch):
            raise ValueError('Epoch and Radiance frame counts differ')
        # Preserve decoded values: do not reapply source packing attributes.
        attrs = {k: var.getncattr(k) for k in var.ncattrs()
                 if k not in ('_FillValue', 'scale_factor', 'add_offset', '_Unsigned', 'valid_range', 'valid_min', 'valid_max')}
    boxes, cloud = read_intervals(labels, orbit, len(epoch), absent_is_clear)
    metadata = dict(source_l1r=str(l1r.resolve()), source_labels=str(labels.resolve()),
                    labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
                    source_version='v01', interval_convention='zero-based inclusive',
                    epoch_policy='Original numeric Epoch values; no UTC conversion or interpolation',
                    missing_epoch_policy='Epochs absent from this output are unknown, not clear',
                    absent_orbit_is_clear=int(absent_is_clear))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=output.suffix, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        if output.suffix.lower() == '.csv':
            with temporary.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['orbit', 'epoch', 'box', 'cloud', 'epoch_units', 'epoch_calendar', 'source_l1r'])
                for frame, value in enumerate(epoch):
                    for column, box in enumerate(boxes):
                        writer.writerow([orbit, repr(value.item()), box, int(cloud[frame, column]),
                                         attrs.get('units', attrs.get('UNITS', '')), attrs.get('calendar', ''), l1r.name])
        else:
            with Dataset(temporary, 'w', format='NETCDF4') as target:
                target.setncatts(metadata)
                target.createDimension('epoch', len(epoch))
                target.createDimension('box', len(boxes))
                times = target.createVariable('epoch', epoch.dtype, ('epoch',))
                times.setncatts(attrs)
                times[:] = epoch
                target.createVariable('orbit', 'i4', ())[:] = orbit
                target.createVariable('box', str, ('box',))[:] = np.array(boxes, dtype=object)
                flags = target.createVariable('cloud', 'u1', ('epoch', 'box'), zlib=True)
                flags.setncatts(dict(flag_values=np.array([0, 1], dtype='u1'), flag_meanings='no_cloud cloud'))
                flags[:] = cloud
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(epoch), len(boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--l1r', required=True, type=Path)
    parser.add_argument('--labels', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--absent-orbit-is-clear', action='store_true',
                        help='Treat an orbit entirely absent from the CSV as reviewed and cloud-free')
    args = parser.parse_args()
    try:
        frames, boxes = convert(args.l1r, args.labels, args.output, args.absent_orbit_is_clear)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(f'Wrote {frames} epochs x {boxes} boxes to {args.output}')


if __name__ == '__main__':
    main()
