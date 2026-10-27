"""Copy selected AWE L0C files to E: while trimming and removing glare.

Uses csv/all_months.csv beside this script by default.
Requires numpy, pandas and netCDF4. Source files are never modified.
Glare indices are zero-based, inclusive, and refer to frames AFTER the
first/last 10 frames have been removed (same as the original script).
"""
import argparse
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from multiprocessing import freeze_support
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

# Glare indices refer to the files AFTER trimming.
# Both start and end indices are inclusive.
csv_file_path = str(Path(__file__).resolve().parent / 'csv' / 'all_months.csv')

source_directory = r'Y:\soc\l0c\version02\2023\12'
output_directory = r'E:\soc\l0c\2023\12'

MIN_ORBIT = 0
MAX_ORBIT = 99999
MAX_WORKERS = 4  # Increase if your network/disk and RAM allow it.
OVERWRITE = True  # Replace existing destination files only after success.
FILE_PATTERN = re.compile(r'_(\d{5})_v02\.nc$')

TRIM_START = 10
TRIM_END = 10


def extract_glare_intervals(data, year=None, month=None, orbit=None):
    if {'Year', 'Month'} <= set(data.columns):
        if year is None or month is None:
            raise ValueError('Combined CSV requires a year and month')
        data = data[(pd.to_numeric(data['Year'], errors='coerce') == year)
                    & (pd.to_numeric(data['Month'], errors='coerce') == month)]
        if data.empty:
            raise ValueError(f'Combined CSV has no coverage for {year:04d}-{month:02d}')
    if orbit is not None:
        orbit_column = 'Orbit #' if 'Orbit #' in data.columns else 'Orbit'
        data = data[pd.to_numeric(data[orbit_column], errors='coerce') == orbit]
    if 'Glare' in data.columns:
        intervals = []
        for _, row in data.iterrows():
            if pd.to_numeric(row['Glare'], errors='coerce') != 1:
                continue
            for index in (1, 2):
                start, end = row.get(f'Glare Start {index}'), row.get(f'Glare End {index}')
                if pd.notna(start) and pd.notna(end):
                    intervals.append({'Orbit #': row.get('Orbit #', row.get('Orbit')),
                                      'glare_initial': start, 'glare_final': end})
        data = pd.DataFrame(intervals, columns=['Orbit #', 'glare_initial', 'glare_final'])
    glare_intervals = {}

    for _, row in data.iterrows():
        if (
            pd.notna(row['Orbit #'])
            and pd.notna(row['glare_initial'])
            and pd.notna(row['glare_final'])
        ):
            orbit = int(row['Orbit #'])
            start = int(row['glare_initial'])
            end = int(row['glare_final'])

            if start < 0 or end < start:
                raise ValueError(
                    f"Invalid glare interval for orbit {orbit}: {start}..{end}"
                )

            glare_intervals.setdefault(orbit, []).append((start, end))

    return glare_intervals


def remove_glare_and_save(nc_file_path, glare_intervals, output_file_path):
    match = re.search(r'_(\d{5})_', os.path.basename(nc_file_path))
    if not match:
        print(f"Skipping file (orbit not found in name): {nc_file_path}")
        return

    orbit_number = int(match.group(1))

    with Dataset(nc_file_path, 'r') as nc:
        total_frames = len(nc.dimensions['time'])

        if total_frames <= TRIM_START + TRIM_END:
            print(
                f"Skipping {os.path.basename(nc_file_path)}: "
                f"only {total_frames} frames; none would remain after trimming."
            )
            return

        # Original-file indices remaining after trimming.
        trimmed_indices = np.arange(
            TRIM_START, total_frames - TRIM_END
        )

        # Glare intervals are indexed relative to this shortened sequence.
        keep_mask = np.ones(len(trimmed_indices), dtype=bool)

        for start, end in glare_intervals.get(orbit_number, []):
            # Clip intervals to the shortened file's bounds.
            start = max(0, start)
            stop = min(len(trimmed_indices), end + 1)

            if start < stop:
                keep_mask[start:stop] = False

        # Map surviving shortened-file indices back to original-file indices.
        keep_indices = trimmed_indices[keep_mask]
        glare_removed = int(np.count_nonzero(~keep_mask))

        if len(keep_indices) == 0:
            print(
                f"Skipping {os.path.basename(nc_file_path)}: "
                "no frames remain after glare removal."
            )
            return

        with Dataset(output_file_path, 'w', format='NETCDF4') as new_nc:
            new_nc.setncatts({
                attr: nc.getncattr(attr)
                for attr in nc.ncattrs()
            })

            output_sizes = {}
            for dim_name, dim in nc.dimensions.items():
                size = (
                    len(keep_indices)
                    if dim_name == 'time'
                    else len(dim)
                )
                output_sizes[dim_name] = size
                new_nc.createDimension(
                    dim_name, None if dim.isunlimited() else size
                )

            for var_name, var in nc.variables.items():
                original_chunks = var.chunking()
                safe_chunks = None

                if isinstance(original_chunks, (list, tuple)):
                    safe_chunks = tuple(
                        max(1, min(output_sizes[dim], chunk))
                        for dim, chunk in zip(
                            var.dimensions, original_chunks
                        )
                    )

                filters = var.filters() or {}
                create_options = {
                    'zlib': filters.get('zlib', False),
                    'chunksizes': safe_chunks,
                }

                if filters.get('zlib', False):
                    create_options.update(
                        complevel=filters.get('complevel', 4),
                        shuffle=filters.get('shuffle', True),
                    )

                # _FillValue must be supplied when creating the variable.
                if '_FillValue' in var.ncattrs():
                    create_options['fill_value'] = var.getncattr(
                        '_FillValue'
                    )

                new_var = new_nc.createVariable(
                    var_name,
                    var.datatype,
                    var.dimensions,
                    **create_options
                )

                new_var.setncatts({
                    attr: var.getncattr(attr)
                    for attr in var.ncattrs()
                    if attr != '_FillValue'
                })

                # Copy stored values without unpacking/repacking.
                var.set_auto_maskandscale(False)
                new_var.set_auto_maskandscale(False)
                var.set_auto_chartostring(False)
                new_var.set_auto_chartostring(False)

                if 'time' in var.dimensions:
                    # Select along the time axis, wherever it occurs.
                    selection = [slice(None)] * var.ndim
                    selection[var.dimensions.index('time')] = keep_indices
                    new_var[...] = var[tuple(selection)]
                else:
                    new_var[...] = var[...]

    print(
        f"Processed: {os.path.basename(nc_file_path)}\n"
        f"  Original: {total_frames}; "
        f"trimmed: {TRIM_START + TRIM_END}; "
        f"glare removed: {glare_removed}; "
        f"remaining: {len(keep_indices)}"
    )


    return True


def process_file(nc_file_path, glare_intervals, destination_dir=None, overwrite=None):
    """Write a processed copy, then atomically publish the completed file."""
    destination_dir = output_directory if destination_dir is None else destination_dir
    overwrite = OVERWRITE if overwrite is None else overwrite
    destination = os.path.join(destination_dir, os.path.basename(nc_file_path))
    if os.path.exists(destination) and not overwrite:
        print(f"Skipping existing: {destination}", flush=True)
        return "skipped"
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(nc_file_path) + ".", suffix=".partial",
        dir=destination_dir,
    )
    os.close(fd)
    try:
        if not remove_glare_and_save(nc_file_path, glare_intervals, temporary):
            return "skipped"
        os.replace(temporary, destination)
        return "processed"
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=source_directory)
    parser.add_argument('--output', default=output_directory)
    parser.add_argument('--glare-csv', default=csv_file_path)
    parser.add_argument('--files', nargs='+', help='Exact source files for a pipeline orbit')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=OVERWRITE)
    args = parser.parse_args(argv)
    if args.workers < 1: parser.error('workers must be positive')
    source_directory_local = args.source
    output_directory_local = args.output
    if os.path.normcase(os.path.realpath(source_directory_local)) == os.path.normcase(
        os.path.realpath(output_directory_local)
    ):
        raise ValueError("Source and output directories must be different.")
    if not os.path.isdir(source_directory_local):
        raise FileNotFoundError(f"Source directory not found: {source_directory_local}")
    if MIN_ORBIT > MAX_ORBIT or min(TRIM_START, TRIM_END) < 0:
        raise ValueError("Invalid orbit range or trim settings.")
    data = pd.read_csv(args.glare_csv, low_memory=False)
    nc_files = list(args.files or [])
    if args.files:
        for path in nc_files:
            if not os.path.isfile(path): raise FileNotFoundError(path)
            if os.path.normcase(os.path.realpath(path)) == os.path.normcase(
                os.path.realpath(os.path.join(output_directory_local, os.path.basename(path)))
            ): raise ValueError('Source files must not be overwritten.')
    else:
        for filename in sorted(os.listdir(source_directory_local)):
            match = FILE_PATTERN.search(filename)
            path = os.path.join(source_directory_local, filename)
            if match and MIN_ORBIT <= int(match.group(1)) <= MAX_ORBIT and os.path.isfile(path):
                nc_files.append(path)

    # Select by the observation date, not just orbit number, which may repeat.
    interval_cache = {}
    file_intervals = {}
    for path in nc_files:
        if {'Year', 'Month'} <= set(data.columns):
            match = re.search(r'_(\d{7})T\d{4}_\d{5}_', os.path.basename(path))
            if not match:
                raise ValueError(f'Cannot determine observation month: {path}')
            date = datetime.strptime(match[1], '%Y%j')
            orbit = int(re.search(r'_(\d{5})_v02', os.path.basename(path))[1])
            key = (date.year, date.month, orbit)
        else:
            key = (None, None, None)
        if key not in interval_cache:
            interval_cache[key] = extract_glare_intervals(data, *key)
        file_intervals[path] = interval_cache[key]

    os.makedirs(output_directory_local, exist_ok=True)
    print(f"Selected {len(nc_files)} files from {source_directory_local}")
    print(f"Final output: {output_directory_local}")
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_file, path, file_intervals[path], output_directory_local, args.overwrite): path
            for path in nc_files
        }
        for future in as_completed(futures):
            try:
                counts[future.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"Error processing {futures[future]}: {exc}", flush=True)
    print(f"Done: {counts['processed']} processed, {counts['skipped']} skipped, "
          f"{counts['failed']} failed.")
    if counts["failed"]:
        raise RuntimeError("Some files failed; see the errors above.")


if __name__ == "__main__":
    freeze_support()
    main()
