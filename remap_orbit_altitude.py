"""Project orbit geolocation from 85 km to 15 km using local awepy.

Radiance is preserved; this changes geolocation, not the radiance sampling grid.
Requires numpy, netCDF4 and pyproj (available in the local awepy environment).
"""
from pathlib import Path
import argparse
import importlib.util
import hashlib
import json
import shutil
from datetime import datetime, timezone

import numpy as np
from netCDF4 import Dataset


def array(value):
    return np.asarray(np.ma.filled(np.ma.asarray(value, dtype=float), np.nan))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('E:/soc/l1r/2023/11'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--orbit', type=int, default=4)
    parser.add_argument('--source-km', type=float, default=85.)
    parser.add_argument('--target-km', type=float, default=15.)
    parser.add_argument('--awepy', type=Path, default=Path('C:/Users/domin/Downloads/awepy-main/awepy-main/awepy/remap_altitude.py'))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('awepy_remap', args.awepy)
    remap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remap)
    files = sorted(args.input.glob(f'awe_l1r_*_{args.orbit:05d}_v*.nc'))
    if not files:
        raise FileNotFoundError('No matching orbit files')
    output = args.output or args.input / f'remap_{args.source_km:g}km_to_{args.target_km:g}km'
    # Validate every input before creating outputs.
    for path in files:
        with Dataset(path) as ds:
            base = remap.parse_oh_layer_altitude_km(ds)
            if not np.isclose(base, args.source_km):
                raise ValueError(f'{path}: altitude {base} differs from requested {args.source_km}')
            for name in ('Latitude', 'Longitude', 'Center_Latitude', 'Center_Longitude',
                         'ISS_Latitude', 'ISS_Longitude', 'ISS_Altitude'):
                if name not in ds.variables:
                    raise ValueError(f'{path}: missing {name}')
        dest = output / f'{path.stem}_remap{args.target_km:g}km.nc'
        if dest.exists() or dest.with_suffix('.nc.partial').exists():
            raise FileExistsError(dest)
    output.mkdir(parents=True, exist_ok=True)
    report = []
    source_hash = hashlib.sha256(args.awepy.read_bytes()).hexdigest()
    for path in files:
        dest = output / f'{path.stem}_remap{args.target_km:g}km.nc'
        partial = dest.with_suffix('.nc.partial')
        print(f'Copying {path.name}', flush=True)
        shutil.copy2(path, partial)
        valid_count = 0
        with Dataset(path) as src, Dataset(partial, 'r+') as dst:
            n = len(src.variables['Epoch'])
            sat = [array(src.variables[k][:]) for k in ('ISS_Latitude', 'ISS_Longitude', 'ISS_Altitude')]
            for i in range(n):
                lat = array(src.variables['Latitude'][i])
                lon = array(src.variables['Longitude'][i])
                clat = array(src.variables['Center_Latitude'][i])
                clon = array(src.variables['Center_Longitude'][i])
                flat_lat = np.append(lat.ravel(), clat)
                flat_lon = np.append(lon.ravel(), clon)
                la, lo = remap.remap_latlon_to_altitudes(
                    *(s[i] for s in sat), flat_lat, flat_lon,
                    [args.target_km], base_alt_km=args.source_km)
                la, lo = la[args.target_km], lo[args.target_km]
                valid = np.isfinite(flat_lat) & np.isfinite(flat_lon)
                if not np.all(np.isfinite(la[valid]) & np.isfinite(lo[valid])):
                    raise ValueError(f'Invalid remap at frame {i}')
                if np.any(np.abs(la[valid]) > 90) or np.any(np.abs(lo[valid]) > 180):
                    raise ValueError(f'Out-of-range coordinates at frame {i}')
                valid_count += int(valid[:-1].sum())
                dst.variables['Latitude'][i] = np.ma.masked_invalid(la[:-1].reshape(lat.shape))
                dst.variables['Longitude'][i] = np.ma.masked_invalid(lo[:-1].reshape(lon.shape))
                dst.variables['Center_Latitude'][i] = la[-1]
                dst.variables['Center_Longitude'][i] = lo[-1]
                if (i + 1) % 200 == 0 or i + 1 == n:
                    print(f'{path.name}: {i+1}/{n} frames', flush=True)
            dst.OH_layer_altitude = f'{args.target_km:.2f} km'
            dst.remap_source_altitude_km = args.source_km
            dst.remap_target_altitude_km = args.target_km
            dst.remap_source_file = str(path)
            dst.remap_code_sha256 = source_hash
            dst.remap_method = 'awepy LOS intersection with WGS84 axes expanded by target altitude; radiance unchanged; no spatial resampling'
            dst.history = getattr(src, 'history', '') + '\n' + datetime.now(timezone.utc).isoformat() + f' Remapped geolocation {args.source_km:g} km to {args.target_km:g} km.'
        # Reopen and check all untouched variable values in bounded chunks.
        print(f'Verifying {path.name}', flush=True)
        with Dataset(path) as src, Dataset(partial) as dst:
            changed = {'Latitude', 'Longitude', 'Center_Latitude', 'Center_Longitude'}
            for name in src.variables:
                if name in changed:
                    continue
                a, b = src.variables[name], dst.variables[name]
                a.set_auto_maskandscale(False)
                b.set_auto_maskandscale(False)
                for start in range(0, a.shape[0], 32):
                    if not np.array_equal(a[start:start+32], b[start:start+32], equal_nan=True):
                        raise ValueError(f'Unexpected modification: {name}')
        partial.rename(dest)
        report.append(dict(input=str(path), output=str(dest), frames=n, valid_pixels=valid_count,
                           source_km=args.source_km, target_km=args.target_km,
                           unchanged_variables_verified=True, awepy_sha256=source_hash))
        (output / 'remap_report.json').write_text(json.dumps(report, indent=2))
        print(f'Completed {dest}', flush=True)


if __name__ == '__main__':
    main()
