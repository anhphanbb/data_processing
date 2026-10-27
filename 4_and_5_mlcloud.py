# -*- coding: utf-8 -*-
"""Combined L1R q20 -> MLCloud prediction -> processed L1C (Python 3.10+).

Requirements: numpy netCDF4 tensorflow (same environment as your cloud model).
Edit SETTINGS and run in Spyder or: python 4_and_5_mlcloud.py
Optional config.txt beside this script overrides SPACE, MIN_RADIANCE and
MAX_RADIANCE using keys space, min_radiance and max_radiance.

Compatibility with the supplied four MLCloud scripts:
* Radiance must have shape (time, 300, 300). Predict x boxes 5..9 in all
  three y rows, using 100-high x 20-wide crops; other raw boxes remain zero.
* uint8 channels are [frame-space, frame, frame+space]. No resize and no
  extra RGB/BGR conversion: this matches cv2.imwrite -> cv2.imread exactly.
* Prediction averaging window is 1, as in the original active call.
* Repeat the first/last valid prediction over SPACE endpoint frames.
* Float32 raw probabilities, then the original five-frame cumulative-sum
  average, including its unsmoothed endpoint copies (no zero padding).
* Average nonzero values from self and along-track neighbors at time/box
  offsets (+6,-1), (+13,-2), (+19,-3), (-6,+1), (-13,+2), (-19,+3).
  This may populate originally ignored boxes. No binary threshold is applied.
* Output variable is Processed_MLCloud, float32, (time,3,15). By default
  raw MLCloud is NOT saved, matching the final original script.

The model stays loaded. One background thread prepares the next batch;
NetCDF access and inference stay on the main thread. No PNGs or intermediate
NetCDF/CSV files are required. Optional CSV is diagnostic only.

Source files are byte-copied before appending predictions: existing values,
variable order, global/variable attributes, storage layout, file format and
fixed/unlimited dimensions are preserved. Unlike the original final script,
source arrays are not recompressed. Inputs containing MLCloud or
Processed_MLCloud are rejected to avoid retaining stale results. Output uses
l1c in place of the l1r filename token. Existing outputs are skipped unless
OVERWRITE=True. A temporary file is renamed only after a successful write.
Real model inference equivalence and speed must be checked with your data.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import csv
import os
import re
import shutil
import tempfile
import warnings

import numpy as np

# ----------------------------- SETTINGS ----------------------------------
INPUT_FOLDER = Path(r'E:\soc\l1r\2023\12')
OUTPUT_FOLDER = Path(r'E:\soc\l1c\2023\12')
MODEL_PATH = Path(__file__).resolve().parent / Path('models/tf_model_cloud_py310_space_8_max_rad_20_oct_27_2025.h5')
CONFIG_PATH = Path(__file__).resolve().with_name('config.txt')
SPACE = 5                       # Original fallback; config.txt may override.
MIN_RADIANCE = 0
MAX_RADIANCE = 24                # Original fallback; config.txt may override.
BATCH_SIZE = 2048                # Original active prediction batch size.
READ_BLOCK_FRAMES = 128
PREFETCH_BATCHES = True
RUNNING_AVERAGE_WINDOW = 5
MIN_ORBIT_NUMBER = 1
RECURSIVE = False                # True preserves relative subfolders.
OVERWRITE = True
SAVE_MLCLOUD = False             # Optional raw, endpoint-filled probabilities.
SAVE_CSV = False
CSV_OUTPUT_FOLDER = Path(r'E:\soc\l1r\2026\02\orbit_predictions_combined')
# ------------------------------------------------------------------------
BOX_DIMS = ('time', 'y_box_across_track', 'x_box_along_track')
BOXES = [(x, y, 20*x, 20*x+19, 100*y, 100*y+99)
         for y in range(3) for x in range(5, 10)]


def load_config(path):
    cfg = {}
    if not path.is_file():
        print(f'Config not found: {path}; using SETTINGS defaults.')
        return cfg
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.split('#', 1)[0].strip()
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        value = value.strip().rstrip(',').strip()
        try:
            number = float(value)
            cfg[key.strip()] = int(number) if number.is_integer() else number
        except ValueError:
            cfg[key.strip()] = value
    return cfg


def normalize_radiance(frame):
    # Preserve the original arithmetic and masked-array assignment behavior.
    frame = np.asarray(np.ma.filled(np.ma.asarray(frame, dtype=np.float32), np.nan))
    frame = np.where(np.isfinite(frame), frame, MIN_RADIANCE)
    return np.clip((frame - MIN_RADIANCE) / (MAX_RADIANCE - MIN_RADIANCE)
                   * 255, 0, 255).astype(np.uint8)



def read_normalized(path, return_support=False):
    from netCDF4 import Dataset
    with Dataset(path, 'r') as ds:
        var = ds.variables['Radiance']
        if (var.ndim != 3 or var.dimensions[0] != 'time'
                or var.shape[1:] != (300, 300)):
            raise ValueError(f'Expected Radiance(time, 300, 300), got '
                             f'{var.dimensions} {var.shape}')
        n = var.shape[0]
        if n <= 2 * SPACE:
            raise ValueError(f'{n} frames: need at least {2 * SPACE + 1}')
        normalized = np.empty((n, 300, 300), dtype=np.uint8)
        box_mode = getattr(ds, 'MLSPB_Mask_Mode', '') in ('boxes_clean_reference', 'boxes_trim_reference')
        support = np.ones((n, 3, 15), bool) if box_mode else None
        for start in range(0, n, READ_BLOCK_FRAMES):
            stop = min(start + READ_BLOCK_FRAMES, n)
            values = np.asarray(np.ma.filled(np.ma.asarray(var[start:stop], dtype=np.float32), np.nan))
            if 'MLSPB_Pixel_Mask' in ds.variables:
                # Mask only the inference copy; stored Radiance remains intact.
                values[np.asarray(ds['MLSPB_Pixel_Mask'][start:stop]) != 0] = np.nan
            normalized[start:stop] = normalize_radiance(values)
            if support is not None:
                geometry = np.isfinite(np.asarray(np.ma.filled(ds['Latitude'][start:stop], np.nan)))
                geometry &= np.isfinite(np.asarray(np.ma.filled(ds['Longitude'][start:stop], np.nan)))
                missing = geometry & ~np.isfinite(values)
                for x, y, xs, xe, ys, ye in BOXES:
                    patch = (slice(None), slice(ys, ye+1), slice(xs, xe+1))
                    support[start:stop, y, x] = ~missing[patch].any(axis=(1, 2)) & geometry[patch].any(axis=(1, 2))
    return (normalized, support) if return_support else normalized


def prepare_batch(normalized, start, stop):
    images = np.empty((stop-start, 100, 20, 3), dtype=np.uint8)
    for k, index in enumerate(range(start, stop)):
        frame = SPACE + index // len(BOXES)
        _, _, xs, xe, ys, ye = BOXES[index % len(BOXES)]
        images[k, ..., 0] = normalized[frame-SPACE, ys:ye+1, xs:xe+1]
        images[k, ..., 1] = normalized[frame, ys:ye+1, xs:xe+1]
        images[k, ..., 2] = normalized[frame+SPACE, ys:ye+1, xs:xe+1]
    return images


def image_batches(normalized, batch_size, prefetch):
    total = (len(normalized) - 2 * SPACE) * len(BOXES)
    spans = iter((start, min(start + batch_size, total))
                 for start in range(0, total, batch_size))
    if not prefetch:
        for start, stop in spans:
            yield start, stop, prepare_batch(normalized, start, stop)
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        start, stop = next(spans)
        pending = pool.submit(prepare_batch, normalized, start, stop)
        while True:
            images = pending.result()
            following = next(spans, None)
            if following is not None:
                pending = pool.submit(prepare_batch, normalized, *following)
            yield start, stop, images
            if following is None:
                break
            start, stop = following


def predict_mlcloud(normalized, model, preprocess_input, support=None):
    n = len(normalized)
    total = (n - 2 * SPACE) * len(BOXES)
    mlcloud = np.full((n, 3, 15), np.nan if support is not None else 0, dtype=np.float32)
    box_x = np.array([b[0] for b in BOXES])
    box_y = np.array([b[1] for b in BOXES])
    last_log = perf_counter()
    batches = image_batches(normalized, BATCH_SIZE, PREFETCH_BATCHES)
    try:
        for start, stop, images in batches:
            # Same uint8 -> preprocess_input path as the original script.
            indices = np.arange(start, stop)
            boxes = indices % len(BOXES)
            frames = SPACE + indices // len(BOXES)
            usable = np.ones(stop-start, bool)
            if support is not None:
                for dt in (-SPACE, 0, SPACE):
                    usable &= support[frames+dt, box_y[boxes], box_x[boxes]]
            probabilities = np.full(stop-start, np.nan, np.float32)
            if usable.any():
                prediction = np.asarray(model(preprocess_input(images[usable]), training=False))
                if prediction.shape not in {(int(usable.sum()),), (int(usable.sum()), 1)}:
                    raise ValueError(f'Expected one probability per image; got {prediction.shape}')
                prediction = prediction.reshape(-1)
                if not np.all(np.isfinite(prediction)) or np.any((prediction < 0) | (prediction > 1)):
                    raise ValueError('Model returned invalid probabilities')
                probabilities[usable] = prediction
            mlcloud[SPACE + indices // len(BOXES), box_y[boxes], box_x[boxes]] = probabilities
            if perf_counter() - last_log >= 5 or stop == total:
                print(f'  Prediction: {stop:,}/{total:,} images', flush=True)
                last_log = perf_counter()
    finally:
        batches.close()
    mlcloud[:SPACE] = mlcloud[SPACE]
    mlcloud[-SPACE:] = mlcloud[-SPACE-1]
    if support is not None:
        # Endpoint repeats are valid only where that endpoint's crop is supported.
        for x, y, *_ in BOXES:
            mlcloud[~support[:, y, x], y, x] = np.nan
    return mlcloud


def process_mlcloud(raw):
    """Vectorized original 5_2 processing, retaining float32 arithmetic."""
    data = np.asarray(raw, dtype=np.float32)
    n, ny, nx = data.shape
    w = RUNNING_AVERAGE_WINDOW
    if w < 1 or w % 2 != 1 or n < w:
        raise ValueError('Need an odd positive smoothing window <= frame count')
    if w == 1:
        smoothed = data.copy()
    else:
        half = w // 2
        # cumsum remains float32, just as the intermediate MLCloud NetCDF.
        cumsum = np.cumsum(np.concatenate((np.zeros((1, ny, nx), dtype=data.dtype),
                                          data), axis=0), axis=0)
        smoothed = np.empty_like(data)
        smoothed[half:-half] = (cumsum[w:] - cumsum[:-w]) / float(w)
        smoothed[:half] = data[half]
        smoothed[-half:] = data[-half-1]

    totals = np.zeros_like(data)
    counts = np.zeros(data.shape, dtype=np.uint8)
    # Same neighbor order as the source loop; ignore zero probabilities.
    for dt, dx in [(0, 0), (6, -1), (13, -2), (19, -3),
                   (-6, 1), (-13, 2), (-19, 3)]:
        t0, t1 = max(0, -dt), min(n, n-dt)
        x0, x1 = max(0, -dx), min(nx, nx-dx)
        if t0 >= t1 or x0 >= x1:
            continue
        neighbor = smoothed[t0+dt:t1+dt, :, x0+dx:x1+dx]
        valid = neighbor != 0
        totals[t0:t1, :, x0:x1] += np.where(valid, neighbor, 0)
        counts[t0:t1, :, x0:x1] += valid
    result = np.zeros_like(data)
    np.divide(totals, counts, out=result, where=counts != 0)
    return result


def process_mlcloud_masked(raw):
    """Ignore unknown predictions; never interpret an unavailable crop as clear."""
    data = np.asarray(raw, np.float32)
    n, ny, nx = data.shape
    w = RUNNING_AVERAGE_WINDOW
    if w < 1 or w % 2 != 1 or n < w: raise ValueError('Invalid smoothing window')
    if w == 1:
        smooth = data.copy()
    else:
        good = np.isfinite(data)
        sums = np.cumsum(np.concatenate((np.zeros((1, ny, nx)), np.where(good, data, 0))), axis=0)
        nums = np.cumsum(np.concatenate((np.zeros((1, ny, nx)), good)), axis=0)
        num = nums[w:]-nums[:-w]
        middle = np.divide(sums[w:]-sums[:-w], num, out=np.full(num.shape, np.nan), where=num > 0)
        h = w//2
        smooth = np.empty_like(data)
        smooth[h:-h] = middle
        smooth[:h] = data[h]; smooth[-h:] = data[-h-1]
    total = np.zeros_like(data); number = np.zeros(data.shape, np.uint8)
    for dt, dx in [(0, 0), (6, -1), (13, -2), (19, -3), (-6, 1), (-13, 2), (-19, 3)]:
        t0, t1 = max(0, -dt), min(n, n-dt)
        x0, x1 = max(0, -dx), min(nx, nx-dx)
        if t0 >= t1 or x0 >= x1: continue
        a = smooth[t0+dt:t1+dt, :, x0+dx:x1+dx]
        good = np.isfinite(a)
        total[t0:t1, :, x0:x1] += np.where(good, a, 0)
        number[t0:t1, :, x0:x1] += good
    supported = number > 0
    # Existing consumers require finite percentages. 1 means conservatively
    # excluded, not a measured 100% cloud probability; companion flag records this.
    result = np.divide(total, number, out=np.ones_like(total), where=supported)
    return result, supported


def validate_source(source):
    from netCDF4 import Dataset
    with Dataset(source) as ds:
        if ds.groups:
            raise ValueError('Expected a flat AWE NetCDF file without groups')
        existing = {'MLCloud', 'Processed_MLCloud'} & set(ds.variables)
        if existing:
            raise ValueError(f'Use original L1R input; already contains {sorted(existing)}')
        for name, size in zip(BOX_DIMS[1:], (3, 15)):
            if name in ds.dimensions and len(ds.dimensions[name]) != size:
                raise ValueError(f'Incompatible existing dimension {name}')


def write_output(source, destination, raw, processed, supported=None):
    from netCDF4 import Dataset
    if source.resolve() == destination.resolve():
        raise ValueError('Input and output file must differ')
    validate_source(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=destination.stem + '.', suffix='.partial',
                                   dir=destination.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        t0 = perf_counter()
        shutil.copyfile(source, temp)
        t1 = perf_counter()
        with Dataset(temp, 'a') as ds:
            for name, length in zip(BOX_DIMS, processed.shape):
                if name not in ds.dimensions:
                    ds.createDimension(name, length)
                elif len(ds.dimensions[name]) != length:
                    raise ValueError(f'Dimension {name} has incompatible size')
            options = dict(zlib=True, complevel=4) if ds.data_model.startswith('NETCDF4') else {}
            var = ds.createVariable('Processed_MLCloud', 'f4', BOX_DIMS, **options)
            var[:] = processed
            if supported is not None:
                var.comment = 'Unknown cloud support encoded as 1 for conservative exclusion; see MLCloud_Supported'
                status = ds.createVariable('MLCloud_Supported', 'u1', BOX_DIMS, **options)
                status[:] = supported.astype('u1')
                status.flag_values = np.array([0, 1], dtype='u1')
                status.flag_meanings = 'unknown_excluded supported_prediction'
                ds.MLCloud_Missing_Policy = 'Skip crops with missing radiance inside geometry in any input channel; use supported neighbors; otherwise exclude'
            if SAVE_MLCLOUD:
                var = ds.createVariable('MLCloud', 'f4', BOX_DIMS, **options)
                var[:] = raw
        if destination.exists() and not OVERWRITE:
            raise FileExistsError(f'Output appeared during processing: {destination}')
        os.replace(temp, destination)
        print(f'  Output: copy {t1-t0:.1f}s, append {perf_counter()-t1:.1f}s')
    finally:
        if temp.exists():
            temp.unlink()



def write_csv(path, mlcloud):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Frame', 'Box', 'Probability', 'RunningAverageProbability'])
        for x, y, *_ in sorted(BOXES, key=lambda b: (b[0], b[1])):
            for frame in range(SPACE, len(mlcloud)-SPACE):
                p = mlcloud[frame, y, x]
                writer.writerow([frame, f'({x},{y})', p, p])


def main():
    global SPACE, MIN_RADIANCE, MAX_RADIANCE
    cfg = load_config(CONFIG_PATH)
    SPACE = cfg.get('space', SPACE)
    MIN_RADIANCE = cfg.get('min_radiance', MIN_RADIANCE)
    MAX_RADIANCE = cfg.get('max_radiance', MAX_RADIANCE)
    for name, value in [('SPACE', SPACE), ('BATCH_SIZE', BATCH_SIZE),
                        ('READ_BLOCK_FRAMES', READ_BLOCK_FRAMES),
                        ('RUNNING_AVERAGE_WINDOW', RUNNING_AVERAGE_WINDOW)]:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if RUNNING_AVERAGE_WINDOW % 2 != 1:
        raise ValueError('RUNNING_AVERAGE_WINDOW must be odd')
    if not (np.isfinite(MIN_RADIANCE) and np.isfinite(MAX_RADIANCE)
            and MAX_RADIANCE > MIN_RADIANCE):
        raise ValueError('Radiance limits must be finite with max > min')
    print(f'Using space={SPACE}, radiance={MIN_RADIANCE}..{MAX_RADIANCE}, '
          f'batch={BATCH_SIZE}, smoothing={RUNNING_AVERAGE_WINDOW}')
    hint = re.search(r'space_(\d+)_max_rad_(\d+)', MODEL_PATH.name)
    if hint and (int(hint[1]) != SPACE or int(hint[2]) != MAX_RADIANCE):
        warnings.warn(f'Model filename suggests space={hint[1]}, max_radiance={hint[2]}; '
                      'check config.txt against the training settings. Using values printed above.')
    if not INPUT_FOLDER.is_dir():
        raise FileNotFoundError(f'Input folder not found: {INPUT_FOLDER}')
    if INPUT_FOLDER.resolve() == OUTPUT_FOLDER.resolve():
        raise ValueError('Input and output folders must differ')
    paths = INPUT_FOLDER.rglob('*.nc') if RECURSIVE else INPUT_FOLDER.glob('*.nc')
    jobs = []
    for path in sorted(paths):
        if '_q20_' not in path.name or OUTPUT_FOLDER.resolve() in path.resolve().parents:
            continue
        relative = path.relative_to(INPUT_FOLDER)
        if any(part in {'nc_files_with_mlcloud', 'images_to_predict',
                        'orbit_predictions', 'orbit_predictions_combined'}
               for part in relative.parts[:-1]):
            continue
        match = re.search(r'_(\d{5})_', path.name)
        if not match:
            print(f'Skip unrecognized orbit filename: {path}')
            continue
        if int(match[1]) < MIN_ORBIT_NUMBER:
            continue
        if '_l1r_' not in path.name:
            print(f'Skip non-L1R file: {path}')
            continue
        destination = OUTPUT_FOLDER / relative.with_name(path.name.replace('_l1r_', '_l1c_', 1))
        if destination.exists() and not OVERWRITE:
            print(f'Skip existing: {destination}')
            continue
        jobs.append((path, destination))
    if not jobs:
        print('No new L1R q20 files to process.')
        return
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Model not found: {MODEL_PATH.resolve()}')
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.models import load_model
    print('CUDA version:', tf.sysconfig.get_build_info().get('cuda_version', 'Not Found'))
    print('cuDNN version:', tf.sysconfig.get_build_info().get('cudnn_version', 'Not Found'))
    gpus = tf.config.list_physical_devices('GPU')
    print('GPU detected:', gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            warnings.warn(f'GPU already initialized; memory growth unchanged: {exc}')
    wall_start = perf_counter()
    model = load_model(str(MODEL_PATH), compile=False)
    shape = model.input_shape
    if isinstance(shape, list) or len(shape) != 4 or any(
            actual is not None and actual != expected
            for actual, expected in zip(shape[1:], (100, 20, 3))):
        raise ValueError(f'Expected model input (batch,100,20,3), got {shape}; no resizing is applied')
    print(f'Model loaded: {perf_counter()-wall_start:.1f}s; files: {len(jobs)}')
    successes = 0
    failures = []
    for index, (source, destination) in enumerate(jobs, 1):
        print(f'\n[{index}/{len(jobs)}] {source.name}', flush=True)
        try:
            t0 = perf_counter()
            validate_source(source)
            normalized, support = read_normalized(source, return_support=True)
            if len(normalized) < RUNNING_AVERAGE_WINDOW:
                raise ValueError('Orbit shorter than running-average window')
            t1 = perf_counter()
            raw = predict_mlcloud(normalized, model, preprocess_input, support)
            del normalized
            t2 = perf_counter()
            supported = None
            if support is None:
                processed = process_mlcloud(raw)
            else:
                processed, supported = process_mlcloud_masked(raw)
                print(f'  Cloud boxes without prediction support: {(~supported).sum()}; conservatively excluded')
            t3 = perf_counter()
            write_output(source, destination, raw, processed, supported)
            if SAVE_CSV:
                # Full stem and relative directories avoid orbit/version collisions.
                csv_path = CSV_OUTPUT_FOLDER / source.relative_to(INPUT_FOLDER).with_suffix('.csv')
                write_csv(csv_path, raw)
            print(f'  Read/normalize {t1-t0:.1f}s | batches/inference {t2-t1:.1f}s | '
                  f'postprocess {t3-t2:.1f}s | save {perf_counter()-t3:.1f}s | '
                  f'total {perf_counter()-t0:.1f}s\n  Saved: {destination}')
            successes += 1
        except Exception as exc:
            failures.append((source, str(exc)))
            print(f'  ERROR: {source.name}: {exc}', flush=True)
    print(f'\nFinished: {successes} succeeded, {len(failures)} failed; '
          f'wall time {perf_counter()-wall_start:.1f}s (including model load).')
    if failures:
        for path, error in failures:
            print(f'  FAILED {path}: {error}')
        raise RuntimeError(f'{len(failures)} file(s) failed; see messages above')


if __name__ == '__main__':
    main()
