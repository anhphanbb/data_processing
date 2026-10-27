#!/usr/bin/env python3
"""Plot L3C radiance, file Epoch, and per-pixel Epoch on the same swath grid.

Dependencies: numpy, matplotlib, netCDF4
Run directly in Spyder after changing DEFAULT_INPUT, or from a terminal:
  python plot_l3c.py --input "E:\\soc\\l3c\\2023\\11"
  python plot_l3c.py --input "path/to/awe_l3c_q20_file.nc" --show
  python plot_l3c.py --input "path/to/folder" --vmin 0 --vmax 2

AWE-GPS conversion matches generate_l2a_l3a_l3c.py: Epoch 1356566418
corresponds to 2023-01-01 00:00:00 UTC. This fixed offset applies to the
current pipeline era; it is not a general historical leap-second converter.
For CF-encoded times, use --epoch-mode cf (requires units containing 'since').
Times represent source coadding-window center frames.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

DEFAULT_INPUT = r'E:\soc\l3c\2023\12'  # File or folder
DEFAULT_OUTPUT = None  # None -> a 'plots' folder beside the input files
DEFAULT_EPOCH_MODE = 'awe-gps'
DEFAULT_SHOW = False
DEFAULT_RECURSIVE = False
DEFAULT_DPI = 180
DEFAULT_VMIN = None  # None -> 2nd percentile
DEFAULT_VMAX = None  # None -> 98th percentile


def read_numeric(variable, key):
    """Decode NetCDF packing and missing values without losing time precision."""
    variable.set_auto_maskandscale(True)
    return np.ma.asarray(variable[key], dtype=np.float64).filled(np.nan)


def read_plane(variable):
    expected = ('time', 'y_cross_track', 'x_along_track')
    if variable.dimensions != expected or variable.shape[0] != 1:
        raise ValueError(f'{variable.name}: expected dimensions {expected} with time=1; '
                         f'got {variable.dimensions}, {variable.shape}')
    return read_numeric(variable, 0)


def to_plot_dates(values, variable, epoch_variable, mode):
    """Return Matplotlib UTC date numbers, preserving missing values."""
    import matplotlib.dates as mdates
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    if not valid.any():
        return result
    if mode == 'awe-gps':
        base = mdates.date2num(datetime(2023, 1, 1, tzinfo=timezone.utc))
        result[valid] = base + (values[valid] - 1356566418.0) / 86400.0
    else:
        units = getattr(variable, 'units', getattr(variable, 'UNITS', None))
        if not units:
            units = getattr(epoch_variable, 'units', getattr(epoch_variable, 'UNITS', None))
        if not units or 'since' not in units.lower():
            raise ValueError('CF mode requires time units such as seconds since 1970-01-01')
        calendar = getattr(variable, 'calendar', getattr(epoch_variable, 'calendar', 'standard'))
        dates = num2date(values[valid], units, calendar=calendar,
                        only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        result[valid] = mdates.date2num(dates)
    return result


def plot_file(path, output_dir, options):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import Normalize

    with Dataset(path) as ds:
        radiance = read_plane(ds['Radiance'])
        valid = np.isfinite(radiance) & (radiance > 0)
        for name in ('Latitude', 'Longitude'):
            if name in ds.variables:
                valid &= np.isfinite(read_plane(ds[name]))
        if not valid.any():
            raise ValueError('No valid radiance pixels')
        radiance = np.where(valid, radiance, np.nan)
        rv = ds['Radiance']
        units = getattr(rv, 'units', getattr(rv, 'UNITS', 'kRayleigh'))
        ev = ds['Epoch']
        epoch = to_plot_dates(read_numeric(ev, slice(0, 1)), ev, ev, options.epoch_mode)[0]
        if not np.isfinite(epoch):
            raise ValueError('Epoch[0] is missing or nonfinite')
        single_time = np.where(valid, epoch, np.nan)
        has_pixel_time = 'Pixel_Epoch' in ds.variables
        pixel_time = np.full(radiance.shape, np.nan)
        if has_pixel_time:
            pv = ds['Pixel_Epoch']
            pixel_values = read_plane(pv)
            if pixel_values.shape != radiance.shape:
                raise ValueError('Pixel_Epoch and Radiance shapes differ')
            pixel_time = np.where(valid, to_plot_dates(pixel_values, pv, ev, options.epoch_mode), np.nan)

    times = pixel_time[np.isfinite(pixel_time)]
    tmin = min(epoch, float(times.min())) if times.size else epoch
    tmax = max(epoch, float(times.max())) if times.size else epoch
    if tmin == tmax:
        tmin -= 0.5 / 86400
        tmax += 0.5 / 86400
    time_norm = Normalize(tmin, tmax)
    low, high = np.nanpercentile(radiance, [2, 98])
    low = low if options.vmin is None else options.vmin
    high = high if options.vmax is None else options.vmax
    if high < low:
        raise ValueError('Radiance vmax must be at least vmin')
    if high == low:
        high = low + max(abs(low)*0.01, 0.001)

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    try:
        fig.suptitle(path.name, fontsize=12)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#eeeeee')
        im = axes[0].imshow(np.ma.masked_invalid(radiance), origin='lower', aspect='auto',
                            interpolation='nearest', cmap=cmap, vmin=low, vmax=high)
        axes[0].set_title('Radiance', loc='left')
        fig.colorbar(im, ax=axes[0], pad=0.015, fraction=0.045, label=f'Radiance ({units})', extend='both')
        timestamp = mdates.num2date(epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        for ax, data, title in zip(axes[1:], (single_time, pixel_time),
                                  (f'Single Epoch: {timestamp} UTC', 'Pixel_Epoch: source-frame time')):
            im = ax.imshow(np.ma.masked_invalid(data), origin='lower', aspect='auto',
                           interpolation='nearest', cmap=cmap, norm=time_norm)
            ax.set_title(title, loc='left')
            cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045, label='UTC')
            cb.locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            cb.formatter = mdates.DateFormatter('%m-%d\n%H:%M:%S', tz=timezone.utc)
            cb.update_ticks()
        if not times.size:
            note = ('Pixel_Epoch is absent. Regenerate this L3C file with the updated script.'
                    if not has_pixel_time else 'Pixel_Epoch contains no valid times over valid radiance.')
            axes[2].text(0.5, 0.5, note, ha='center', va='center', transform=axes[2].transAxes,
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
            print(f'WARNING {path.name}: {note}')
        for ax in axes:
            ax.set_ylabel('Cross-track pixel')
        axes[-1].set_xlabel('Along-track pixel (stored swath grid; aspect stretched for readability)')
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f'{path.stem}_radiance_time.png'
        fig.savefig(target, dpi=options.dpi)
        print(f'SAVED {target}')
        if options.show:
            plt.show()
        return target
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--epoch-mode', choices=('awe-gps', 'cf'), default=DEFAULT_EPOCH_MODE)
    parser.add_argument('--show', action=argparse.BooleanOptionalAction, default=DEFAULT_SHOW)
    parser.add_argument('--recursive', action=argparse.BooleanOptionalAction, default=DEFAULT_RECURSIVE)
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI)
    parser.add_argument('--vmin', type=float, default=DEFAULT_VMIN)
    parser.add_argument('--vmax', type=float, default=DEFAULT_VMAX)
    options = parser.parse_args()
    import matplotlib
    if not options.show:
        matplotlib.use('Agg')
    source = options.input
    if source.is_file():
        files, base = [source], source.parent
    elif source.is_dir():
        files = sorted(source.rglob('awe_l3c_*.nc') if options.recursive else source.glob('awe_l3c_*.nc'))
        base = source
    else:
        parser.error(f'Input does not exist: {source}')
    if not files:
        parser.error(f'No awe_l3c_*.nc files found in {source}')
    failed = []
    for path in files:
        target_dir = (options.output or base / 'plots') / path.parent.relative_to(base)
        try:
            plot_file(path, target_dir, options)
        except Exception as exc:
            failed.append(path)
            print(f'ERROR {path.name}: {exc}')
    print(f'Done: {len(files)-len(failed)} plotted, {len(failed)} failed.')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
