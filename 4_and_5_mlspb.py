# -*- coding: utf-8 -*-
"""Adaptive q20 -> MLSP predictions -> MLSPB pipeline (Python 3.10+).

Run in the same environment as the original prediction script. Requirements:
    numpy scipy opencv-python netCDF4 tensorflow
Edit SETTINGS below, then run this file in Spyder or with Python.

No prediction PNGs, CSV round trip, or intermediate NetCDF are needed.
The model stays loaded. A single background thread prepares the next batch;
all NetCDF access stays on the main thread. Memory holds one orbit of uint8
frames plus at most a few batches, not all cropped images for the orbit.

Compatibility with the four supplied scripts:
* Normalize 0..24 to uint8, use channels [i-5, i, i+5], preserve cv2 resizing
  and ResNet preprocess_input exactly (do NOT add a BGR/RGB conversion).
* Input images are exactly 256x256. Use 32 boxes: six by six grid excluding
  the four corners. Edge rows/columns have 42 pixels; resize crops to 43x43.
* Raw probabilities (prediction running-average window was 1); copy first/last
  available predictions over five endpoint frames; float32 MLSP storage precision.
* Zero x columns 0..2, five-frame zero-padded temporal average, strict >0.6,
  six-connected clusters touching frame 0 or the last frame, expansion 1.5
  using candidates strictly >0.4. Preserve the original hull fallback.
* Zero padding is intentionally retained; it suppresses endpoint probabilities.

Output keeps the input filename, including l0c, in OUTPUT_FOLDER (as before).
All output dimensions are fixed-size, including time. NETCDF4 sources with
only fixed dimensions are byte-copied; sources with unlimited dimensions are
rewritten in blocks to make those dimensions fixed-size. Standard zlib settings
and valid chunk sizes are preserved during rewriting. Input files are never modified. Existing outputs are
skipped unless OVERWRITE=True. Writes use a temporary file and atomic rename.

ADAPTIVE_PREDICTION first tests alternate valid frame centers, including both
valid endpoints. Missing probabilities are linearly interpolated on the original
frame axis for provisional five-frame smoothing and endpoint-cluster selection.
Test every remaining center in the selected expanded hull time ranges plus
REFINE_PADDING, then rebuild the provisional map. Repeat if the selected ranges
grow or change. Cached model outputs are never recomputed. All 32 boxes are
inferred at each refinement center. Final MLSPB uses the original processing.
Untested centers outside the refinement ranges retain interpolated estimates,
not assumed-clear zeros. Brief features can still be missed; this is approximate.
Set ADAPTIVE_PREDICTION=False for full inference and comparison.
The first/last SPACE frames copy the nearest valid prediction as before.

SAVE_MLSP stores probabilities including interpolated estimates and endpoint copies.
SAVE_CSV optionally exports the original four-column prediction CSV for checks.
Actual inference equivalence and speed must be checked with your model/data.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import csv
import os
import shutil
import tempfile
import warnings

import numpy as np
from scipy.ndimage import label
from scipy.spatial import ConvexHull, Delaunay

# ----------------------------- SETTINGS ----------------------------------
INPUT_FOLDER = Path(r'E:\soc\l0c\2025\08')
OUTPUT_FOLDER = Path(r'E:\soc\l0d\2025\08')
MODEL_PATH = Path(__file__).resolve().parent / Path('models/tf_model_py310_sp_acc_and_recall_july_27_soc_2_2025.h5')
BATCH_SIZE = 1024              # Same as the active call in the original script.
READ_BLOCK_FRAMES = 128        # Bounds temporary floating-point Radiance memory.
PREFETCH_BATCHES = True        # Prepare next image batch during model inference.
RECURSIVE = False              # True preserves relative subfolders in output.
OVERWRITE = False
SAVE_MLSP = False
SAVE_CSV = False
CSV_OUTPUT_FOLDER = Path('sp_orbit_predictions/csv_combined')
ADAPTIVE_PREDICTION = True     # False tests every valid frame center.
COARSE_STRIDE = 2             # First pass: every other valid frame center.
REFINE_PADDING = 4            # Original frames beyond each expanded hull time range.
SPACE = 5
RUNNING_AVERAGE_WINDOW = 5
THRESHOLD = 0.6
EXPANSION_CANDIDATE_THRESHOLD = 0.4
EXPANSION_FACTOR = 1.5
# ------------------------------------------------------------------------

BOX_DIMS = ('time', 'y_box_across_track', 'x_box_along_track')
# Inclusive endpoints. Index 255 is the last pixel of each 256-pixel axis.
RANGES = [
    (0, 41),     # 42 pixels
    (42, 84),    # 43 pixels
    (85, 127),   # 43 pixels
    (128, 170),  # 43 pixels
    (171, 213),  # 43 pixels
    (214, 255),  # 42 pixels
]

BOXES = [(x, y, xs, xe, ys, ye)
         for y, (ys, ye) in enumerate(RANGES)
         for x, (xs, xe) in enumerate(RANGES)
         if (x, y) not in {(0, 0), (0, 5), (5, 0), (5, 5)}]


def normalize_radiance(frame, min_radiance=0, max_radiance=24):
    # Preserve original arithmetic, dtype conversion, and masked-array behavior.
    return np.clip((frame - min_radiance) / (max_radiance - min_radiance)
                   * 255, 0, 255).astype(np.uint8)


def read_normalized(path):
    from netCDF4 import Dataset
    with Dataset(path, 'r') as ds:
        var = ds.variables['Radiance']
        if (var.ndim != 3 or var.dimensions[0] != 'time'
                or var.shape[1:] != (256, 256)):
            raise ValueError(f'Expected Radiance(time, 256, 256), got '
                             f'{var.dimensions} {var.shape}')
        n = var.shape[0]
        if n <= 2 * SPACE:
            raise ValueError(f'{n} frames: need at least {2 * SPACE + 1}')
        normalized = np.empty((n, 256, 256), dtype=np.uint8)
        for start in range(0, n, READ_BLOCK_FRAMES):
            stop = min(start + READ_BLOCK_FRAMES, n)
            normalized[start:stop] = normalize_radiance(var[start:stop, :, :])
    return normalized


def prepare_batch(normalized, start, stop, frame_ids=None):
    import cv2
    images = np.empty((stop - start, 43, 43, 3), dtype=np.uint8)
    for k, index in enumerate(range(start, stop)):
        frame = (SPACE + index // len(BOXES) if frame_ids is None
                 else int(frame_ids[index // len(BOXES)]))
        _, _, xs, xe, ys, ye = BOXES[index % len(BOXES)]
        crop = np.stack((normalized[frame - SPACE, ys:ye+1, xs:xe+1],
                         normalized[frame, ys:ye+1, xs:xe+1],
                         normalized[frame + SPACE, ys:ye+1, xs:xe+1]), axis=-1)
        images[k] = cv2.resize(crop, (43, 43))
    return images


def image_batches(normalized, batch_size, prefetch, frame_ids=None):
    total = ((len(normalized) - 2 * SPACE) if frame_ids is None
             else len(frame_ids)) * len(BOXES)
    if not total:
        return
    spans = iter((start, min(start + batch_size, total))
                 for start in range(0, total, batch_size))
    if not prefetch:
        for start, stop in spans:
            yield start, stop, prepare_batch(normalized, start, stop, frame_ids)
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        start, stop = next(spans)
        pending = pool.submit(prepare_batch, normalized, start, stop, frame_ids)
        while True:
            images = pending.result()
            following = next(spans, None)
            if following is not None:
                pending = pool.submit(prepare_batch, normalized, *following, frame_ids)
            yield start, stop, images
            if following is None:
                break
            start, stop = following


def interpolate_untested(mlsp, tested):
    """Fill estimates only; preserve every actual model prediction exactly."""
    n = len(mlsp)
    known = np.flatnonzero(tested)
    if not len(known):
        raise ValueError('Interpolation requires model-tested centers')
    missing = np.flatnonzero(~tested)
    missing = missing[(missing >= SPACE) & (missing < n - SPACE)]
    for y in range(6):
        for x in range(6):
            mlsp[missing, y, x] = np.interp(missing, known, mlsp[known, y, x])
    mlsp[:SPACE] = mlsp[SPACE]
    mlsp[-SPACE:] = mlsp[-SPACE-1]


def refinement_centers(mlsp):
    """Conservative temporal envelope of each selected expanded 3-D hull."""
    working = mlsp.copy()
    working[:, :, :3] = 0
    averaged = calculate_running_average(working, RUNNING_AVERAGE_WINDOW)
    selected = boundary_cluster_points(averaged > THRESHOLD)
    wanted = np.zeros(len(mlsp), dtype=bool)
    for _, points in selected:
        temporal = points[:, 0]
        if len(points) >= 4 and np.all(np.ptp(points, axis=0) > 0):
            try:
                vertices = points[ConvexHull(points).vertices]
                center = vertices.mean(axis=0)
                expanded = center + EXPANSION_FACTOR * (vertices - center)
                temporal = np.concatenate((temporal, expanded[:, 0]))
            except Exception as exc:
                warnings.warn(f'Refinement hull failed; using padded cluster time range: {exc}')
        lo = max(SPACE, int(np.floor(temporal.min())) - REFINE_PADDING)
        hi = min(len(mlsp) - SPACE - 1,
                 int(np.ceil(temporal.max())) + REFINE_PADDING)
        wanted[lo:hi+1] = True
    return np.flatnonzero(wanted)


def cluster_guided_search(mlsp, tested, infer):
    n = len(mlsp)
    lo, hi = SPACE, n - SPACE - 1
    if lo > hi:
        raise ValueError('Orbit has no valid three-channel frame centers')
    coarse = sorted(set(range(lo, hi + 1, COARSE_STRIDE)) | {hi})
    infer(coarse)
    print(f'  Coarse pass: {len(coarse):,} model-tested centers', flush=True)
    iteration = 0
    while True:
        interpolate_untested(mlsp, tested)
        wanted = refinement_centers(mlsp)
        pending = wanted[~tested[wanted]]
        if not len(pending):
            break
        iteration += 1
        before = int(tested.sum())
        print(f'  Cluster refinement {iteration}: {len(pending):,} additional '
              'centers; all 32 boxes per center', flush=True)
        infer(pending)
        if int(tested.sum()) <= before:
            raise RuntimeError('Refinement inference made no progress')
    print(f'  Refinement complete after {iteration} pass(es)', flush=True)


def predict_mlsp(normalized, model, preprocess_input):
    n = len(normalized)
    mlsp = np.zeros((n, 6, 6), dtype=np.float32)
    tested = np.zeros(n, dtype=bool)
    box_x = np.array([b[0] for b in BOXES])
    box_y = np.array([b[1] for b in BOXES])
    last_log = perf_counter()

    def infer(frame_ids):
        nonlocal last_log
        ids = np.array(sorted({int(i) for i in frame_ids if not tested[int(i)]}),
                       dtype=np.int64)
        if not len(ids):
            return
        batches = image_batches(normalized, BATCH_SIZE, PREFETCH_BATCHES, ids)
        try:
            for start, stop, images in batches:
                # Preserve the original uint8 -> preprocess_input path.
                probabilities = np.asarray(model(preprocess_input(images), training=False))
                if probabilities.shape not in {(stop-start,), (stop-start, 1)}:
                    raise ValueError(f'Expected one probability per image; got {probabilities.shape}')
                probabilities = probabilities.reshape(-1)
                if not np.all(np.isfinite(probabilities)):
                    raise ValueError('Model returned non-finite probabilities')
                if np.any((probabilities < 0) | (probabilities > 1)):
                    raise ValueError('Model output is outside probability range 0..1')
                indices = np.arange(start, stop)
                boxes = indices % len(BOXES)
                mlsp[ids[indices // len(BOXES)], box_y[boxes], box_x[boxes]] = probabilities
                if perf_counter() - last_log >= 5:
                    print(f'  Prediction batch: {stop:,}/{len(ids)*len(BOXES):,} images', flush=True)
                    last_log = perf_counter()
        finally:
            batches.close()
        tested[ids] = True

    if ADAPTIVE_PREDICTION:
        cluster_guided_search(mlsp, tested, infer)
    else:
        infer(range(SPACE, n - SPACE))
    mlsp[:SPACE] = mlsp[SPACE]
    mlsp[-SPACE:] = mlsp[-SPACE-1]
    count = int(tested.sum())
    valid = n - 2 * SPACE
    print(f'  Model-tested frame centers: {count:,}/{valid:,}; '
          f'interpolated centers: {valid-count:,}; '
          f'crop predictions: {count*len(BOXES):,} '
          f'({100*(1-count/valid):.1f}% fewer than full inference)', flush=True)
    return mlsp


def calculate_running_average(data, window_size):
    # Retain np.convolve arithmetic and zero padding for threshold compatibility.
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'),
                               axis=0, arr=data)


def expand_cluster_points(cluster_points, mlsp, expansion_factor):
    if len(cluster_points) < 4:
        return cluster_points
    try:
        dimensions = (cluster_points.max(axis=0) - cluster_points.min(axis=0)) > 0
        if dimensions.sum() < 3:
            return cluster_points
        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
        centroid = np.mean(hull_points, axis=0)
        expanded = centroid + expansion_factor * (hull_points - centroid)
        candidates = np.column_stack(np.where(mlsp > EXPANSION_CANDIDATE_THRESHOLD))
        inside = Delaunay(expanded).find_simplex(candidates) >= 0
        return np.unique(np.vstack((cluster_points, candidates[inside])), axis=0)
    except Exception as exc:
        warnings.warn(f'Cluster expansion failed; retaining original cluster: {exc}')
        return cluster_points


def select_and_expand_clusters(thresholded, mlsp, expansion_factor):
    selected_points = boundary_cluster_points(thresholded)
    result = np.zeros_like(thresholded, dtype=np.uint8)
    for cluster_id, points in selected_points:
        expanded = expand_cluster_points(points, mlsp, expansion_factor)
        result[tuple(expanded.T)] = 1
        print(f'  Cluster {cluster_id}: {len(points):,} -> {len(expanded):,} points')
    return result


def boundary_cluster_points(thresholded):
    """Largest six-connected component at each endpoint, with original ties."""
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = structure[1, :, 1] = structure[:, 1, 1] = 1
    labeled, _ = label(thresholded, structure=structure)
    sizes = np.bincount(labeled.ravel())
    selected = []
    for boundary in (labeled[0], labeled[-1]):
        ids = np.unique(boundary)
        ids = ids[ids != 0]
        if ids.size:
            # Sorted ids preserve the original smallest-label tie break.
            best = int(ids[np.argmax(sizes[ids])])
            if best not in selected:
                selected.append(best)
    return [(i, np.column_stack(np.where(labeled == i))) for i in selected]


def make_mlspb(mlsp):
    working = mlsp.copy()
    working[:, :, :3] = 0
    averaged = calculate_running_average(working, RUNNING_AVERAGE_WINDOW)
    return select_and_expand_clusters(averaged > THRESHOLD, averaged, EXPANSION_FACTOR)


def clone_source(source, target):
    from netCDF4 import Dataset
    with Dataset(source) as src:
        source_format = src.data_model
        has_unlimited = any(dim.isunlimited() for dim in src.dimensions.values())
        empty_dims = [name for name, dim in src.dimensions.items() if len(dim) == 0]
    if empty_dims:
        raise ValueError(f'Cannot create fixed-size zero-length dimensions: {empty_dims}')
    if source_format == 'NETCDF4' and not has_unlimited:
        shutil.copyfile(source, target)
        return
    # Unlimited dimensions cannot be changed in place: rebuild the file.
    # Classic formats also require conversion to store unsigned MLSPB.
    with Dataset(source) as src, Dataset(target, 'w', format='NETCDF4') as dst:
        dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(dim))
        for name, var in src.variables.items():
            if name in {'MLSP', 'MLSPB'}:
                continue
            filters = var.filters() or {}
            kwargs = dict(zlib=filters.get('zlib', False))
            if kwargs['zlib']:
                kwargs.update(complevel=filters.get('complevel', 4),
                              shuffle=filters.get('shuffle', True))
            if filters.get('fletcher32', False):
                kwargs['fletcher32'] = True
            if '_FillValue' in var.ncattrs():
                kwargs['fill_value'] = var.getncattr('_FillValue')
            chunking = var.chunking()
            if isinstance(chunking, (list, tuple)):
                kwargs['chunksizes'] = tuple(
                    min(int(chunk), len(src.dimensions[dim]))
                    for dim, chunk in zip(var.dimensions, chunking))
            out = dst.createVariable(name, var.datatype, var.dimensions, **kwargs)
            out.setncatts({a: var.getncattr(a) for a in var.ncattrs() if a != '_FillValue'})
            var.set_auto_maskandscale(False)
            out.set_auto_maskandscale(False)
            var.set_auto_chartostring(False)
            out.set_auto_chartostring(False)
            if var.ndim == 0:
                out[...] = var[...]
            else:
                for start in range(0, var.shape[0], READ_BLOCK_FRAMES):
                    stop = min(start + READ_BLOCK_FRAMES, var.shape[0])
                    out[start:stop] = var[start:stop]


def write_output(source, destination, mlsp, mlspb):
    from netCDF4 import Dataset
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=destination.stem + '.', suffix='.partial',
                                   dir=destination.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        t0 = perf_counter()
        clone_source(source, temp)
        t1 = perf_counter()
        with Dataset(temp, 'a') as ds:
            if any(dim.isunlimited() for dim in ds.dimensions.values()):
                raise RuntimeError('Output dimensions must all be fixed-size')
            for name, length in zip(BOX_DIMS, mlspb.shape):
                if name not in ds.dimensions:
                    ds.createDimension(name, length)
                elif len(ds.dimensions[name]) != length:
                    raise ValueError(f'Dimension {name} has incompatible size')
            if 'MLSPB' in ds.variables:
                raise ValueError('Input already contains MLSPB; use original q20 input')
            var = ds.createVariable('MLSPB', 'u1', BOX_DIMS, zlib=True, complevel=4)
            var.description = f'Binary MLSP > {THRESHOLD}, expanded clusters touching frame 0 or last'
            var.prediction_method = ('alternate_frames_cluster_refinement' if ADAPTIVE_PREDICTION else 'all_frames')
            if ADAPTIVE_PREDICTION:
                var.coarse_stride_frames = COARSE_STRIDE
                var.refine_padding_frames = REFINE_PADDING
                var.skipped_frame_policy = 'Linear interpolation between model-tested centers; not measured probabilities'
                var.refinement_policy = 'All boxes in expanded hull time ranges plus padding; repeat until no untested centers remain in selected ranges'
            var[:] = mlspb
            if SAVE_MLSP:
                if 'MLSP' in ds.variables:
                    raise ValueError('Input already contains MLSP; use original q20 input')
                raw = ds.createVariable('MLSP', 'f4', BOX_DIMS, zlib=True, complevel=4)
                raw.description = ('Model probabilities before masking/smoothing; endpoint filled; '
                                   'adaptive mode linearly interpolates untested centers (estimates, not model outputs)')
                raw[:] = mlsp
        if destination.exists() and not OVERWRITE:
            raise FileExistsError(f'Output appeared during processing: {destination}')
        os.replace(temp, destination)
        print(f'  Output: copy/convert {t1-t0:.1f}s, append {perf_counter()-t1:.1f}s')
    finally:
        if temp.exists():
            temp.unlink()


def write_csv(path, mlsp):
    # In adaptive mode, probabilities include interpolated estimates.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Frame', 'Box', 'Probability', 'RunningAverageProbability'])
        for x, y, *_ in sorted(BOXES, key=lambda b: (b[0], b[1])):
            for frame in range(SPACE, len(mlsp)-SPACE):
                p = mlsp[frame, y, x]
                writer.writerow([frame, f'({x},{y})', p, p])


def main():
    import cv2
    if not isinstance(COARSE_STRIDE, int) or COARSE_STRIDE < 1:
        raise ValueError('COARSE_STRIDE must be a positive integer')
    if not isinstance(REFINE_PADDING, int) or REFINE_PADDING < RUNNING_AVERAGE_WINDOW // 2:
        raise ValueError('REFINE_PADDING must cover at least half the running-average window')
    if BATCH_SIZE < 1 or READ_BLOCK_FRAMES < 1 or SPACE < 1:
        raise ValueError('Batch size, read block, and space must be positive')
    if RUNNING_AVERAGE_WINDOW < 1 or RUNNING_AVERAGE_WINDOW % 2 != 1:
        raise ValueError('Running-average window must be positive and odd')
    if not INPUT_FOLDER.is_dir():
        raise FileNotFoundError(f'Input folder not found: {INPUT_FOLDER}')
    if INPUT_FOLDER.resolve() == OUTPUT_FOLDER.resolve():
        raise ValueError('Input and output folders must differ')
    paths = INPUT_FOLDER.rglob('*.nc') if RECURSIVE else INPUT_FOLDER.glob('*.nc')
    jobs = []
    for path in sorted(paths):
        if '_q20_' not in path.name or OUTPUT_FOLDER.resolve() in path.resolve().parents:
            continue
        # Exclude old intermediate folders if recursive discovery is enabled.
        if any(part in {'nc_files_with_mlsp', 'sp_images_to_predict'}
               for part in path.relative_to(INPUT_FOLDER).parts[:-1]):
            continue
        destination = OUTPUT_FOLDER / path.relative_to(INPUT_FOLDER)
        if destination.exists() and not OVERWRITE:
            print(f'Skip existing: {destination}')
            continue
        jobs.append((path, destination))
    if not jobs:
        print('No new q20 files to process.')
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
    cv2.setNumThreads(1)
    wall_start = perf_counter()
    model = load_model(str(MODEL_PATH), compile=False)
    print(f'Model loaded: {perf_counter()-wall_start:.1f}s; files: {len(jobs)}')
    successes = 0
    failures = []
    for index, (source, destination) in enumerate(jobs, 1):
        print(f'\n[{index}/{len(jobs)}] {source.name}', flush=True)
        try:
            t0 = perf_counter()
            normalized = read_normalized(source)
            if len(normalized) < RUNNING_AVERAGE_WINDOW:
                raise ValueError('Orbit shorter than running-average window')
            t1 = perf_counter()
            mlsp = predict_mlsp(normalized, model, preprocess_input)
            del normalized
            t2 = perf_counter()
            mlspb = make_mlspb(mlsp)
            t3 = perf_counter()
            write_output(source, destination, mlsp, mlspb)
            if SAVE_CSV:
                # Full stem avoids collisions between versions or repeated orbit IDs.
                csv_path = CSV_OUTPUT_FOLDER / source.relative_to(INPUT_FOLDER).with_suffix('.csv')
                write_csv(csv_path, mlsp)
            positives = int(np.count_nonzero(np.any(mlspb, axis=(1, 2))))
            print(f'  Read/normalize {t1-t0:.1f}s | batches/inference {t2-t1:.1f}s | '
                  f'postprocess {t3-t2:.1f}s | save {perf_counter()-t3:.1f}s | '
                  f'total {perf_counter()-t0:.1f}s')
            print(f'  MLSPB-positive frames: {positives}/{len(mlspb)}\n  Saved: {destination}')
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
