#!/usr/bin/env python3
r"""Filter/flat-field AWE files after removing later duplicate Epoch frames.

Gaps of any length and positive irregular intervals are preserved and reported.
No gap interpolation, no filled/unfilled branches, no long-gap skipping.
Backward or invalid Epoch still fail; all filters must share the removal map.
Default: mask MLSPB boxes, retain all observed frames, and use the exact legacy trim flat field before extending.
Use --mlspb-mode trim for the legacy frame-selection algorithm.
Retained samples are read directly from original files through a shared index map.
No full cleaned-input copy is made; filtering/flat-fielding still modifies Radiance.

Outputs: L1F for all filters. Screen raw deduplicated frames in all four filters.
The union of flagged original indices is removed from all filters BEFORE
MLSPB selection, image filtering and flat-field calculation. No interpolation.
Thresholds are heuristic; inspect the combined report for false positives.
One per-orbit .bad_frames.csv records triggering filters and removed indices;
the batch runner consolidates these into frame_quality_all_orbits.csv.

Defaults read August 2025 inputs and write a separate l1f_dedup_test folder.
Run from Spyder/F5 with MLSPB inputs, or --skip-mlspb for an isolated L0C test.
Normal pipeline positional paths and worker options remain supported.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from numba import get_num_threads, njit, prange, set_num_threads
except ImportError:
    njit = None
    get_num_threads = None
    prange = range
    set_num_threads = None


FILTERS = ("q20", "p12", "p14", "bkg")
TIME_DIM = "time"
MLSPB_NAME = "MLSPB"
RADIANCE_NAME = "Radiance"
RADIANCE_SCALE = 1000.0
RADIANCE_READ_BLOCK_SIZE = 32
RADIANCE_WRITE_BLOCK_SIZE = 100

# Settings for running directly from Visual Studio, Spyder, or an IDE.
DEFAULT_MLSPB_INPUT = r"E:\soc\l0d\2025\08"
DEFAULT_L0C_INPUT = r"E:\soc\l0c\2025\08"
DEFAULT_OUTPUT = r"E:\soc\l1f_dedup_test\2025\08"
DEFAULT_SELECTION_MODE = "cxx"  # Legacy trim mode: "cxx" or "clean"
DEFAULT_MLSPB_MODE = "boxes"  # "boxes" or "trim"
DEFAULT_MIN_CLEAN_FRAMES = 100  # Operational guard; validate with real orbits
BAD_FRAME_SCREENING = True  # False disables screening, removal and previews.
SAVE_BAD_FRAME_IMAGES = True
TEMP_ROOT = None  # None: beside L1F outputs; or set a fast local SSD folder.
DEFAULT_RECURSIVE = False
DEFAULT_OVERWRITE = True
DEFAULT_ORBITS_AT_A_TIME = 6
DEFAULT_WORKERS = 24
DEFAULT_NUMBA_THREADS = 1

# L0D MLSPB files can retain an L0C filename, optionally with a suffix.
NAME_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<level>l0[cd])_(?P<filter>q20|p12|p14|bkg)_"
    r"(?P<stamp>\d{7}T\d{4})_(?P<orbit>\d{5})_(?P<version>v\d{2})"
    r"(?P<suffix>(?:_[A-Za-z0-9-]+)*)\.nc$",
    re.IGNORECASE,
)


def _import_netcdf4():
    try:
        import netCDF4
    except ImportError as exc:
        raise SystemExit(
            "netCDF4 is required. Install it with: pip install netCDF4"
        ) from exc
    return netCDF4


def _as_plain_array(value, *, fill=np.nan) -> np.ndarray:
    """Convert a NetCDF result (possibly masked) to a normal ndarray."""
    if np.ma.isMaskedArray(value):
        return np.asarray(np.ma.filled(value, fill))
    return np.asarray(value)


def radiance_to_cpp_uint16(radiance: np.ndarray) -> np.ndarray:
    """Scale positive finite radiance by 1000 and truncate into uint16."""
    values = np.asarray(radiance, dtype=np.float64)
    output = np.zeros(values.shape, dtype=np.uint16)
    valid = np.isfinite(values) & (values > 0)
    scaled = np.trunc(values[valid] * RADIANCE_SCALE)
    output[valid] = np.clip(scaled, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return output


def _gaussian_3x3_cpp() -> np.ndarray:
    """Normalized 3x3 kernel with the C++ integer-arithmetic denominator."""
    kernel = np.empty((3, 3), dtype=np.float32)
    denominator = 1  # C++ float(2 * 3 * 3 / 16) evaluates to 1.
    total = np.float32(0.0)
    for row in range(3):
        for col in range(3):
            distance2 = (row - 1) ** 2 + (col - 1) ** 2
            kernel[row, col] = np.float32(np.exp(-np.float32(distance2) / denominator))
            total = np.float32(total + kernel[row, col])
    return (kernel / total).astype(np.float32)


GAUSSIAN_3X3_CPP = _gaussian_3x3_cpp()


def _cpp_float32_roundtrip_lut() -> np.ndarray:
    """Map uint16 counts through the former float32 NetCDF round trip."""
    counts = np.arange(65536, dtype=np.uint16)
    stored = counts.astype(np.float32) / np.float32(RADIANCE_SCALE)
    return np.trunc(stored.astype(np.float64) * RADIANCE_SCALE).astype(np.uint16)


CPP_FLOAT32_ROUNDTRIP_LUT = _cpp_float32_roundtrip_lut()


def _filter_image_numpy(image: np.ndarray) -> np.ndarray:
    """Existing NumPy fallback for the C++ bright- and dark-pixel passes."""
    source = np.asarray(image, dtype=np.uint16)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {source.shape}")
    histogram = np.bincount(source.ravel(), minlength=65536)
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    threshold = int(np.searchsorted(cumulative, 0.95 * source.size, side="left"))

    padded1 = np.pad(source.astype(np.float32), 1, mode="edge")
    local_mean = np.zeros(source.shape, dtype=np.float32)
    for row in range(3):
        for col in range(3):
            local_mean += GAUSSIAN_3X3_CPP[row, col] * padded1[
                row : row + source.shape[0], col : col + source.shape[1]
            ]
    bright = local_mean < (
        source.astype(np.float32) - np.float32(threshold / 2.0)
    )

    # Original C++ sorts only ring entries 0..14 and selects entry 8.
    ny, nx = source.shape
    stage1 = source.copy()
    bright_y, bright_x = np.nonzero(bright)
    if bright_y.size:
        padded3 = np.pad(source, 3, mode="edge")
        samples = []
        for dx in range(-3, 4):
            samples.append(padded3[bright_y + 5, bright_x + 3 + dx])
            samples.append(padded3[bright_y + 1, bright_x + 3 + dx])
        samples.append(padded3[bright_y + 2, bright_x + 5])
        replacement = np.partition(np.stack(samples, axis=0), 8, axis=0)[8]
        stage1[bright_y, bright_x] = replacement

    # Retain C++ bounds 1 <= index < size-2, including asymmetric edge handling.
    result = stage1.copy()
    if ny >= 4 and nx >= 4:
        work = stage1.astype(np.float32)
        neighbor_sum = np.zeros((ny - 3, nx - 3), dtype=np.float32)
        target_y = slice(1, ny - 2)
        target_x = slice(1, nx - 2)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_sum += work[
                    1 + dy : ny - 2 + dy,
                    1 + dx : nx - 2 + dx,
                ]
        neighbor_mean = neighbor_sum / np.float32(8.0)
        center = work[target_y, target_x]
        dark = (neighbor_mean - center) > np.float32(1000.0)
        interior = result[target_y, target_x]
        interior[dark] = np.trunc(neighbor_mean[dark]).astype(np.uint16)
        result[target_y, target_x] = interior
    return result


if njit is not None:
    @njit(cache=True)
    def _filter_image_numba_impl(source: np.ndarray) -> np.ndarray:
        """Compiled pixel-loop version preserving the existing operation order."""
        ny, nx = source.shape
        histogram = np.zeros(65536, dtype=np.int64)
        for y in range(ny):
            for x in range(nx):
                histogram[source[y, x]] += 1
        target = 0.95 * source.size
        cumulative = 0
        threshold = 65535
        for value in range(65536):
            cumulative += histogram[value]
            if cumulative >= target:
                threshold = value
                break

        stage1 = source.copy()
        half_threshold = np.float32(threshold / 2.0)
        samples = np.empty(15, dtype=np.uint16)
        for y in range(ny):
            for x in range(nx):
                local_mean = np.float32(0.0)
                for row in range(3):
                    yy = y + row - 1
                    if yy < 0:
                        yy = 0
                    elif yy >= ny:
                        yy = ny - 1
                    for col in range(3):
                        xx = x + col - 1
                        if xx < 0:
                            xx = 0
                        elif xx >= nx:
                            xx = nx - 1
                        product = np.float32(
                            GAUSSIAN_3X3_CPP[row, col] * np.float32(source[yy, xx])
                        )
                        local_mean = np.float32(local_mean + product)

                if local_mean < np.float32(np.float32(source[y, x]) - half_threshold):
                    sample_index = 0
                    for dx in range(-3, 4):
                        xx = x + dx
                        if xx < 0:
                            xx = 0
                        elif xx >= nx:
                            xx = nx - 1
                        yy = y + 2
                        if yy >= ny:
                            yy = ny - 1
                        samples[sample_index] = source[yy, xx]
                        sample_index += 1
                        yy = y - 2
                        if yy < 0:
                            yy = 0
                        samples[sample_index] = source[yy, xx]
                        sample_index += 1
                    yy = y - 1
                    if yy < 0:
                        yy = 0
                    xx = x + 2
                    if xx >= nx:
                        xx = nx - 1
                    samples[14] = source[yy, xx]
                    samples.sort()
                    stage1[y, x] = samples[8]

        result = stage1.copy()
        if ny >= 4 and nx >= 4:
            for y in range(1, ny - 2):
                for x in range(1, nx - 2):
                    neighbor_sum = np.float32(0.0)
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy != 0 or dx != 0:
                                neighbor_sum = np.float32(
                                    neighbor_sum + np.float32(stage1[y + dy, x + dx])
                                )
                    neighbor_mean = np.float32(neighbor_sum / np.float32(8.0))
                    if np.float32(neighbor_mean - np.float32(stage1[y, x])) > np.float32(1000.0):
                        result[y, x] = np.uint16(np.trunc(neighbor_mean))
        return result


    @njit(cache=True, parallel=True)
    def _filter_cube_numba_impl(images: np.ndarray) -> np.ndarray:
        """Filter independent frames concurrently using Numba threads."""
        output = np.empty_like(images)
        for index in prange(images.shape[0]):
            output[index] = _filter_image_numba_impl(images[index])
        return output
else:
    _filter_image_numba_impl = None
    _filter_cube_numba_impl = None


def filter_image_cpp(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image, dtype=np.uint16)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {source.shape}")
    if _filter_image_numba_impl is not None:
        return _filter_image_numba_impl(source)
    return _filter_image_numpy(source)


def filter_cube_cpp(images: np.ndarray) -> np.ndarray:
    source = np.asarray(images, dtype=np.uint16)
    if source.ndim != 3:
        raise ValueError(f"Expected a 3-D image cube, got shape {source.shape}")
    if _filter_cube_numba_impl is not None:
        return _filter_cube_numba_impl(source)
    return np.stack([_filter_image_numpy(image) for image in source], axis=0)


def flatfield_image_cpp(image: np.ndarray, flat_field: np.ndarray) -> np.ndarray:
    """Existing flat-field calculation, including the central normalization."""
    source = np.asarray(image, dtype=np.uint16)
    ff = np.asarray(flat_field, dtype=np.float64)
    if source.ndim != 2 or ff.shape != source.shape:
        raise ValueError(
            f"Image and flat field must be matching 2-D arrays; got "
            f"{source.shape} and {ff.shape}"
        )
    ny, nx = source.shape
    if ny < 20 or nx < 20:
        raise ValueError("Images must be at least 20x20 for central normalization")

    divided = np.zeros(source.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        divided[1:-1, 1:-1] = source[1:-1, 1:-1] / ff[1:-1, 1:-1]
    y0, x0 = ny // 2 - 10, nx // 2 - 10
    center_image = source[y0 : y0 + 20, x0 : x0 + 20]
    center_divided = divided[y0 : y0 + 20, x0 : x0 + 20]
    image_mean = float(center_image.sum(dtype=np.float64) / 400.0)
    valid_center = center_divided > 0
    if not np.any(valid_center):
        raise ValueError("Flat-field normalization has no positive central pixels")
    flat_mean = float(center_divided[valid_center].mean(dtype=np.float64))
    if not np.isfinite(flat_mean) or flat_mean == 0:
        raise ValueError(f"Invalid flat-field central mean: {flat_mean}")

    scale = image_mean / flat_mean
    corrected = divided[1:-1, 1:-1] * scale
    surrounding = (
        divided[:-2, :-2]
        + divided[:-2, 1:-1]
        + divided[:-2, 2:]
        + divided[1:-1, :-2]
        + divided[1:-1, 2:]
        + divided[2:, :-2]
        + divided[2:, 1:-1]
        + divided[2:, 2:]
    ) / 8.0 * scale
    use_surrounding = (corrected > 50000.0) | (corrected == 0.0)
    selected = np.where(use_surrounding, surrounding, corrected)
    # Permanently invalid pixels may have 0/0; output masking later uses Latitude.
    selected = np.where(np.isfinite(selected), selected, 0.0)
    result = source.copy()
    result[1:-1, 1:-1] = np.clip(
        np.trunc(selected), 0, np.iinfo(np.uint16).max
    ).astype(np.uint16)
    return result


def select_frame_range(flagged: np.ndarray, mode: str = "cxx") -> tuple[int, int]:
    """Return inclusive first and last retained frame indices."""
    flags = np.asarray(flagged, dtype=bool).ravel()
    if flags.size == 0:
        raise ValueError("MLSPB contains no time samples")
    last_index = flags.size - 1
    if mode == "clean":
        first = 0
        while first <= last_index and flags[first]:
            first += 1
        last = last_index
        while last >= first and flags[last]:
            last -= 1
        if first > last:
            raise ValueError("Every frame is MLSPB-positive; no clean range remains")
        return first, last
    if mode != "cxx":
        raise ValueError(f"Unknown selection mode: {mode}")

    # Bounded reproduction of the existing C++ do/while behavior.
    new_first = 0
    new_last = last_index
    i = 1
    while i <= last_index:
        current_flag = bool(flags[i])
        if current_flag:
            new_first = i
        i += 1
        if not (current_flag or i == last_index):
            break
    if new_first == last_index:
        return 0, last_index
    i = last_index - 1
    while i >= 0:
        current_flag = bool(flags[i])
        if current_flag:
            new_last = i
        i -= 1
        if not current_flag:
            break
    return new_first, new_last


# Exact inclusive detector ranges from 4_and_5_mlspb.py (disk y, x order).
MLSPB_RANGES = ((0, 41), (42, 84), (85, 127), (128, 170), (171, 213), (214, 255))


def read_mlspb_boxes(path):
    with _import_netcdf4().Dataset(path) as ds:
        var = ds[MLSPB_NAME]
        dims = ('time', 'y_box_across_track', 'x_box_along_track')
        if set(var.dimensions) != set(dims) or var.ndim != 3:
            raise ValueError(f'Expected MLSPB dimensions {dims}; got {var.dimensions}')
        a = np.asarray(np.ma.filled(np.ma.asarray(var[:], dtype=float), np.nan))
        a = np.transpose(a, [var.dimensions.index(d) for d in dims])
        if a.shape[1:] != (6, 6) or not np.all(np.isfinite(a)) or not np.all((a == 0) | (a == 1)):
            raise ValueError('MLSPB must contain finite binary flags on a 6x6 grid')
        return a.astype(bool)


def box_mask(boxes, shape=(256, 256)):
    if shape != (256, 256):
        raise ValueError(f'MLSPB detector mapping requires 256x256, got {shape}')
    mask = np.zeros(shape, bool)
    for y, x in np.argwhere(boxes):
        y0, y1 = MLSPB_RANGES[y]; x0, x1 = MLSPB_RANGES[x]
        mask[y0:y1+1, x0:x1+1] = True
    return mask


def longest_clean_interval(boxes, epoch, min_frames):
    clean = ~np.any(boxes, axis=(1, 2))
    # Do not call a span across a time gap one continuous reference interval.
    delta = np.diff(epoch)
    positive = delta[delta > 0]
    cadence = EPOCH_CADENCE_SECONDS or (float(np.median(positive)) if positive.size else 0)
    best = (0, 0); start = 0
    for i in range(len(clean)):
        if i and delta[i-1] > 1.5*cadence:
            if i-start > best[1]-best[0]: best = (start, i)
            start = i
        if not clean[i]:
            if i-start > best[1]-best[0]: best = (start, i)
            start = i+1
    if len(clean)-start > best[1]-best[0]: best = (start, len(clean))
    if best[1]-best[0] < min_frames:
        raise ValueError(f'Longest fully clean interval has {best[1]-best[0]} frames; '
                         f'need {min_frames}. No contaminated-reference fallback is used.')
    return best  # half-open, deduplicated frame indices


def filter_image_masked(image, valid):
    """Same bright/dark scheme, excluding invalid samples from every statistic."""
    a = np.asarray(image, dtype=np.uint16)
    valid = np.asarray(valid, bool)
    if valid.all(): return filter_image_cpp(a)
    if not valid.any(): return np.zeros_like(a)
    values = a[valid]
    threshold = np.searchsorted(np.cumsum(np.bincount(values, minlength=65536)), .95*values.size)
    ny, nx = a.shape
    p = np.pad(a.astype(np.float64), 3, mode='edge')
    v = np.pad(valid, 3, mode='edge')
    def shifted(dy, dx):
        return p[3+dy:3+dy+ny, 3+dx:3+dx+nx], v[3+dy:3+dy+ny, 3+dx:3+dx+nx]
    total = np.zeros(a.shape); weight = np.zeros(a.shape)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            b, good = shifted(dy, dx); w = float(GAUSSIAN_3X3_CPP[dy+1, dx+1])
            total += np.where(good, b*w, 0); weight += good*w
    mean = np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)
    bright = valid & (mean < a.astype(float)-threshold/2)
    stage = a.copy()
    if bright.any():
        offsets = [(dy, dx) for dx in range(-3, 4) for dy in (2, -2)] + [(-1, 2)]
        samples = []
        for dy, dx in offsets:
            b, good = shifted(dy, dx)
            samples.append(np.where(good[bright], b[bright], np.inf))
        samples = np.sort(np.array(samples), axis=0)
        n = np.isfinite(samples).sum(axis=0)
        # Full support reproduces C++ order statistic #8; partial support uses its median.
        rank = np.where(n == 15, 8, np.maximum(0, n//2))
        replacement = samples[rank, np.arange(len(n))]
        old = stage[bright].copy(); use = n > 0
        old[use] = replacement[use].astype(np.uint16); stage[bright] = old
    p = np.pad(stage.astype(float), 3, mode='edge')
    total.fill(0); weight.fill(0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0: continue
            b, good = shifted(dy, dx)
            total += np.where(good, b, 0); weight += good
    mean = np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)
    dark = valid & (weight > 0) & (mean-stage.astype(float) > 1000)
    bounds = np.zeros_like(valid); bounds[1:-2, 1:-2] = True
    use = dark & bounds
    stage[use] = np.trunc(mean[use]).astype(np.uint16)
    stage[~valid] = 0
    return stage


def flatfield_image_masked(image, flat_field, valid):
    """Return float counts and normalization code: 0 none, 1 center, 2 footprint."""
    a = np.asarray(image, dtype=float); ff = np.asarray(flat_field, dtype=float)
    good = valid & np.isfinite(ff) & (ff > 0)
    divided = np.divide(a, ff, out=np.full(a.shape, np.nan), where=good)
    ny, nx = a.shape
    support = good & (a > 0)
    center = np.zeros_like(good)
    center[ny//2-10:ny//2+10, nx//2-10:nx//2+10] = True
    norm = support & center
    code = 1
    if not norm.any(): norm = support; code = 2
    if not norm.any(): return np.full(a.shape, np.nan), 0
    scale = a[norm].mean()/divided[norm].mean()
    corrected = divided*scale
    # Repair only valid pixels, and only using valid neighboring ratios.
    p = np.pad(divided, 1, constant_values=np.nan)
    total = np.zeros_like(a); count = np.zeros_like(a)
    for dy in range(3):
        for dx in range(3):
            if dy == dx == 1: continue
            b = p[dy:dy+ny, dx:dx+nx]; ok = np.isfinite(b)
            total += np.where(ok, b, 0); count += ok
    repair = good & ((corrected > 50000) | (corrected == 0)) & (count > 0)
    corrected[repair] = total[repair]/count[repair]*scale
    corrected[~good] = np.nan
    return np.clip(np.trunc(corrected), 0, 65535), code


def process_box_masked_file(input_path, output_path, boxes, reference, overwrite,
                           label, numba_threads, quality_metrics, frame_indices=None):
    """Run the unchanged trim algorithm, then extend with its exact flat field."""
    if output_path.exists() and not overwrite: raise FileExistsError(output_path)
    if set_num_threads is not None: set_num_threads(numba_threads)
    netCDF4 = _import_netcdf4()
    lo, hi = reference
    temp_parent = Path(TEMP_ROOT) if TEMP_ROOT else output_path.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='awe_trim_reference_', dir=temp_parent) as temp:
        baseline_path = Path(temp)/output_path.name
        calibration = {}
        print(f'{label}: run original trim processing first, frames {lo}..{hi-1}', flush=True)
        _process_filter_file_continuous(input_path, baseline_path, lo, hi-1, False,
                                       label, True, numba_threads, calibration=calibration, frame_indices=frame_indices)
        _extend_trim_result(input_path, output_path, baseline_path, boxes, reference,
                            calibration['flat_field'], label, quality_metrics, frame_indices)


def _extend_trim_result(input_path, output_path, baseline_path, boxes, reference,
                        ff, label, quality_metrics, frame_indices=None):
    """Copy unmasked baseline values exactly; calculate only the added frames."""
    netCDF4 = _import_netcdf4()
    lo, hi = reference
    with netCDF4.Dataset(input_path) as src, netCDF4.Dataset(baseline_path) as baseline:
        rad = src[RADIANCE_NAME]
        indices = np.arange(rad.shape[0]) if frame_indices is None else np.asarray(frame_indices)
        if len(indices) != len(boxes): raise ValueError('MLSPB/retained-input length mismatch')
        def read_frame(i):
            raw = raw_block[i-start]
            geo = None if geometry_block is None else geometry_block[i-start]
            valid = np.isfinite(raw) & (raw >= 0)
            if geo is not None: valid &= geo
            valid &= ~box_mask(boxes[i], raw.shape)
            return raw, valid, geo
        clone_trimmed_netcdf(input_path, output_path, 0, len(boxes)-1,
                            skip_data_variables=frozenset((RADIANCE_NAME,)), frame_indices=indices)
        try:
            with netCDF4.Dataset(output_path, 'r+') as dst:
                dst.MLSPB_Mask_Mode = 'boxes_trim_reference'
                dst.Flat_Field_Reference_Start = int(lo)
                dst.Flat_Field_Reference_Stop_Exclusive = int(hi)
                dst.Flat_Field_Reference_Frames = int(hi-lo)
                dst.Flat_Field_Reference_Epoch = np.array([read_epoch(input_path)[indices[lo]], read_epoch(input_path)[indices[hi-1]]])
                dst.Flat_Field_Policy = 'Exact legacy trim interval, filtering, denominator and normalization; copy baseline then extend using its frozen flat field'
                dst.Trim_Overlap_Policy = 'Baseline Radiance copied exactly except explicitly MLSPB-masked boxes become NaN'
                dst.MLSPB_Box_Ranges = json.dumps(MLSPB_RANGES)
                dst.Missing_Radiance_Policy = 'NaN missing; coordinates retained for geometry'
                code = dst.createVariable('Flat_Field_Normalization', 'u1', ('time',), fill_value=False)
                code.flag_values = np.array([0, 1, 2, 3], dtype='u1')
                code.flag_meanings = 'no_valid_support central_20x20 valid_footprint_fallback copied_legacy_trim'
                countvar = dst.createVariable('MLSPB_Masked_Pixels', 'i4', ('time',), fill_value=False)
                dst[RADIANCE_NAME].set_auto_maskandscale(False)
                fallbacks = empty = 0
                for start in range(0, len(boxes), RADIANCE_READ_BLOCK_SIZE):
                    end = min(start+RADIANCE_READ_BLOCK_SIZE, len(boxes))
                    raw_block = read_selected_images(rad, indices[start:end])
                    geometry_block = selected_latitude_mask(src, indices[start:end], raw_block.shape[1:])
                    saved = np.empty((end-start, 256, 256), np.float32)
                    methods = []; masked = []
                    for q, i in enumerate(range(start, end)):
                        raw, valid, geo = read_frame(i)
                        if quality_metrics is not None: quality_metrics.append(raw_frame_metrics(raw, geo))
                        if lo <= i < hi:
                            # Do not re-filter, re-normalize, or quantize baseline values.
                            saved[q] = _read_image_block(baseline[RADIANCE_NAME], i-lo, i-lo+1)[0]
                            saved[q][box_mask(boxes[i])] = np.nan
                            method = 3
                        else:
                            filtered = CPP_FLOAT32_ROUNDTRIP_LUT[filter_image_masked(radiance_to_cpp_uint16(raw), valid)]
                            corrected, method = flatfield_image_masked(filtered, ff, valid)
                            saved[q] = corrected.astype(np.float32)/np.float32(RADIANCE_SCALE)
                        methods.append(method); masked.append(int(box_mask(boxes[i]).sum()))
                        fallbacks += method == 2; empty += method == 0
                    _write_image_block(dst[RADIANCE_NAME], start, saved)
                    code[start:end] = methods; countvar[start:end] = masked
                    print(f'  {label}: box-masked flat field {end}/{len(boxes)}', flush=True)
                dst.Flat_Field_Fallback_Frames = int(fallbacks)
                dst.Flat_Field_Empty_Frames = int(empty)
            update_l1f_metadata(output_path)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise



def read_mlspb_flags(q20_path: Path) -> np.ndarray:
    netCDF4 = _import_netcdf4()
    with netCDF4.Dataset(q20_path, "r") as dataset:
        if MLSPB_NAME not in dataset.variables:
            raise KeyError(f"{q20_path.name} does not contain {MLSPB_NAME}")
        variable = dataset.variables[MLSPB_NAME]
        if TIME_DIM not in variable.dimensions:
            raise ValueError(
                f"{MLSPB_NAME} dimensions do not contain '{TIME_DIM}': "
                f"{variable.dimensions}"
            )
        values = _as_plain_array(variable[:], fill=0)
        time_axis = variable.dimensions.index(TIME_DIM)
        values = np.moveaxis(values, time_axis, 0)
        return np.any(values != 0, axis=tuple(range(1, values.ndim)))


def parse_name(path: Path) -> re.Match[str]:
    match = NAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"Unexpected filename '{path.name}'. Expected something like "
            "awe_l0c_q20_2024001T0123_00621_v01.nc"
        )
    return match


def matching_l0c_filter_paths(
    mlspb_q20_path: Path, l0c_root: Path, mlspb_root: Path,
) -> dict[str, Path]:
    """Locate the four L0C image files matching one MLSPB q20 file."""
    match = parse_name(mlspb_q20_path)
    if match.group("filter").lower() != "q20":
        raise ValueError(f"The selected MLSPB file must be q20: {mlspb_q20_path}")
    if not l0c_root.is_dir():
        raise NotADirectoryError(f"L0C input folder does not exist: {l0c_root}")
    relative_parent = Path()
    if mlspb_root.is_dir():
        relative_parent = mlspb_q20_path.parent.relative_to(mlspb_root)
    l0c_dir = l0c_root / relative_parent
    paths: dict[str, Path] = {}
    for filter_name in FILTERS:
        base = (
            f"{match.group('prefix')}l0c_{filter_name}_{match.group('stamp')}_"
            f"{match.group('orbit')}_{match.group('version')}"
        )
        suffix = match.group("suffix") or ""
        preferred = l0c_dir / f"{base}{suffix}.nc"
        standard = l0c_dir / f"{base}.nc"
        if preferred.is_file():
            paths[filter_name] = preferred
        elif standard.is_file():
            paths[filter_name] = standard
        else:
            candidates = sorted(l0c_dir.glob(f"{base}*.nc"))
            if len(candidates) == 1:
                paths[filter_name] = candidates[0]
            elif len(candidates) > 1:
                raise FileNotFoundError(
                    f"Multiple possible L0C matches for {filter_name}:\n  "
                    + "\n  ".join(str(path) for path in candidates)
                )
            else:
                paths[filter_name] = preferred
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing matching input file(s):\n  " + "\n  ".join(missing))
    return paths


def discover_mlspb_q20_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        match = parse_name(input_path)
        if match.group("filter").lower() != "q20":
            raise ValueError(f"Expected a q20 file containing MLSPB: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    iterator: Iterable[Path]
    iterator = input_path.rglob("*.nc") if recursive else input_path.glob("*.nc")
    q20_files = []
    for path in iterator:
        match = NAME_RE.match(path.name)
        if match and match.group("filter").lower() == "q20":
            q20_files.append(path)
    if not q20_files:
        raise FileNotFoundError(
            f"No q20 files matching the AWE naming convention were found in "
            f"the MLSPB folder {input_path}"
        )
    return sorted(q20_files)


def adjusted_stamp(q20_path: Path, first_index: int) -> str:
    """Build YYYYDDDTHHMM from the first retained frame's absolute Epoch.

    Use the AWE/C++ reference: Epoch 1356566418 is 2023-01-01
    00:00:00 UTC. This preserves the C++ time convention while updating
    the full date across midnight, leap days, and year boundaries.
    The reference assumes the same fixed GPS/UTC offset as the C++ code.
    """
    netCDF4 = _import_netcdf4()
    with netCDF4.Dataset(q20_path, "r") as dataset:
        if "Epoch" not in dataset.variables:
            raise ValueError(f"{q20_path}: Epoch is required for output naming")
        variable = dataset.variables["Epoch"]
        if variable.dimensions != (TIME_DIM,):
            raise ValueError(f"{q20_path}: Epoch must have dimensions ('time',)")
        if first_index < 0 or first_index >= variable.shape[0]:
            raise ValueError(f"{q20_path}: invalid first frame index {first_index}")
        epoch_first = float(_as_plain_array(variable[first_index], fill=np.nan))
    if not np.isfinite(epoch_first):
        raise ValueError(f"{q20_path}: Epoch[{first_index}] is missing or non-finite")
    reference_utc = datetime(2023, 1, 1, tzinfo=timezone.utc)
    # Drop fractional seconds before datetime conversion so microsecond
    # rounding cannot promote a value just below a minute boundary.
    elapsed_seconds = int(np.floor(epoch_first - 1356566418.0))
    first_utc = reference_utc + timedelta(seconds=elapsed_seconds)
    return first_utc.strftime("%Y%jT%H%M")


def output_path_for(input_path: Path, output_dir: Path, stamp: str) -> Path:
    match = parse_name(input_path)
    filename = (
        f"{match.group('prefix')}l1f_{match.group('filter').lower()}_{stamp}_"
        f"{match.group('orbit')}_{match.group('version')}"
        f"{match.group('suffix') or ''}.nc"
    )
    return output_dir / filename


def selected_runs(indices):
    """Yield destination slices and contiguous source slices, in retained order."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError('Retained indices must be nonempty, nonnegative and strictly increasing')
    boundaries = np.r_[0, np.flatnonzero(np.diff(indices) != 1)+1, len(indices)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        yield slice(int(a), int(b)), slice(int(indices[a]), int(indices[b-1])+1)


def read_selected_images(variable, indices):
    blocks = [_read_image_block(variable, src.start, src.stop)
              for _, src in selected_runs(indices)]
    return blocks[0] if len(blocks)==1 else np.concatenate(blocks, axis=0)


def selected_latitude_mask(dataset, indices, shape):
    if 'Latitude' not in dataset.variables or TIME_DIM not in dataset['Latitude'].dimensions:
        return None
    latitude = read_selected_images(dataset['Latitude'], indices)
    return np.isfinite(latitude) if latitude.shape[1:] == shape else None


def _copy_variable_data(source_var, output_var, first: int, last: int, frame_indices=None) -> None:
    output_var.set_auto_maskandscale(False)
    source_var.set_auto_maskandscale(False)
    if TIME_DIM not in source_var.dimensions:
        output_var[...] = source_var[...]
        return
    axis = source_var.dimensions.index(TIME_DIM)
    output_count = last - first + 1
    for out_start in range(0, output_count, 32):
        count = min(32, output_count - out_start)
        src_key = [slice(None)] * source_var.ndim
        dst_key = [slice(None)] * output_var.ndim
        src_key[axis] = slice(first + out_start, first + out_start + count)
        dst_key[axis] = slice(out_start, out_start + count)
        if frame_indices is None:
            output_var[tuple(dst_key)] = source_var[tuple(src_key)]
        else:
            indices = frame_indices[first+out_start:first+out_start+count]
            for dest, src in selected_runs(indices):
                src_key[axis] = src
                dst_key[axis] = slice(out_start+dest.start, out_start+dest.stop)
                output_var[tuple(dst_key)] = source_var[tuple(src_key)]


# C++ L1F writer functions are embedded below (no companion module required).

def clone_trimmed_netcdf(
    input_path: Path,
    output_path: Path,
    first: int,
    last: int,
    skip_data_variables: frozenset[str] = frozenset(),
    frame_indices=None,
) -> None:
    """Write the L1F schema from the supplied C++ comparison report.

    Preserve common source attributes, replace reported differences, and copy
    only C++ L1F variables. Array axes are renamed, never transposed. Radiance
    is populated later by the existing filter/flat-field processing routine.
    """
    netCDF4 = _import_netcdf4()
    output_count = last - first + 1
    image_names = ("Radiance", "Latitude", "Longitude")
    scalar_names = (
        "Center_Latitude", "Center_Longitude", "Epoch", "ISS_Altitude",
        "ISS_Latitude", "ISS_Longitude", "ISS_Pitch", "ISS_Roll", "ISS_Yaw",
    )
    # Match Files::CreateNetCDF variable-definition order as well.
    required_names = (
        "Epoch", "ISS_Latitude", "ISS_Longitude", "ISS_Altitude",
        "ISS_Pitch", "ISS_Roll", "ISS_Yaw", "Center_Latitude",
        "Center_Longitude", "Radiance", "Latitude", "Longitude",
    )
    target_image_dims = ("time", "y_cross_track", "x_along_track")
    # The previous writer attached reversed spatial labels to the same stored
    # image rows/columns. Match the C++ writer by correcting labels only;
    # transposing would change the pixel order used by the processing pipeline.
    supported_image_dims = {
        target_image_dims,
        ("time", "x_along_track", "y_cross_track"),
        ("time", "y_along_track", "x_cross_track"),
    }
    units = {
        "Radiance": "kRayleigh", "Latitude": "Degrees", "Longitude": "Degrees",
        "Center_Latitude": "Degrees", "Center_Longitude": "Degrees",
        "ISS_Altitude": "km", "ISS_Latitude": "Degrees",
        "ISS_Longitude": "Degrees", "ISS_Pitch": "Degrees",
        "ISS_Roll": "Degrees", "ISS_Yaw": "Degrees",
    }
    output_path = Path(output_path)
    input_path = Path(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output NetCDF paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(input_path, "r") as source:
        missing = sorted(set(required_names) - set(source.variables))
        if missing:
            raise ValueError(f"Missing C++ L1F variables in {input_path}: {missing}")
        if "time" not in source.dimensions:
            raise ValueError("Input has no time dimension")
        source_count = len(source.dimensions["time"])
        if frame_indices is not None:
            frame_indices = np.asarray(frame_indices, dtype=np.int64)
            list(selected_runs(frame_indices))
            if frame_indices[-1] >= source_count: raise ValueError('Retained index outside source')
        logical_count = source_count if frame_indices is None else len(frame_indices)
        if first < 0 or last < first or last >= logical_count:
            raise ValueError(f"Invalid inclusive retained range {first}..{last}")
        source_radiance = source.variables["Radiance"]
        spatial_shape = source_radiance.shape[1:]
        # Validate before opening/truncating an output. All image variables
        # must use the same source layout, even when spatial sizes are equal.
        for name in required_names:
            variable = source.variables[name]
            if name in image_names:
                if (variable.dimensions not in supported_image_dims
                        or variable.dimensions != source_radiance.dimensions):
                    raise ValueError(
                        f"{name}: unsupported or inconsistent dimensions "
                        f"{variable.dimensions}; all image variables must share "
                        f"one supported layout (Radiance: {source_radiance.dimensions})"
                    )
            elif variable.dimensions != ("time",):
                raise ValueError(
                    f"{name}: expected ('time',), got {variable.dimensions}"
                )
            if name in image_names and variable.shape[1:] != spatial_shape:
                raise ValueError(f"{name}: spatial shape differs from Radiance")
            if np.dtype(variable.dtype) != np.dtype("f8" if name == "Epoch" else "f4"):
                raise ValueError(f"{name}: expected the floating dtype in the C++ report")
            if any(a in variable.ncattrs() for a in ("scale_factor", "add_offset", "_Unsigned")):
                raise ValueError(f"{name}: packed data require explicit conversion")
            # Report inputs use NaN fill. Do not silently lose a finite fill
            # sentinel when removing the Python-only _FillValue attribute.
            if "_FillValue" in variable.ncattrs():
                if not np.isnan(variable.getncattr("_FillValue")):
                    raise ValueError(f"{name}: finite _FillValue needs explicit handling")

        globals_out = {
            name: source.getncattr(name) for name in source.ncattrs()
            if name not in {"Generation_Date", "history", "Bad_pixel_mask", "Dark_correction"}
        }
        # Defaults below are exactly the values printed in the supplied
        # reference report. Prefer existing acquisition metadata where present.
        defaults = {
            "Cadence": np.float32(1.1),
            "Exposure_Time": np.float32(1.0),
            "Horizontal_Resolution": np.float32(0.0),
            "Number_coadded_images": np.int32(1),
        }
        for name, default in defaults.items():
            globals_out.setdefault(name, default)
        globals_out["NC_KIND"] = "netCDF-4"
        globals_out["TEXT"] = (
            "AWE investigates the connection between tropospheric weather and space weather "
            "- Visit 'https://www.awemission.org' for more details"
        )
        try:
            with netCDF4.Dataset(output_path, "w", format="NETCDF4") as output:
                output.setncatts(globals_out)
                # Compatibility with the legacy C++ writer: it reuses input
                # Header.dimids in the new file instead of rebuilding them.
                # Thus IDs must be x=0, y=1, time=2, as in CreateNetCDF.
                # This is definition order, NOT the image array axis order.
                output.createDimension("x_along_track", spatial_shape[1])
                output.createDimension("y_cross_track", spatial_shape[0])
                output.createDimension("time", None)
                for name in required_names:
                    source_var = source.variables[name]
                    dimensions = target_image_dims if name in image_names else ("time",)
                    # One full detector frame per compressed chunk, independent of input layout.
                    kwargs = dict(zlib=True, complevel=1, shuffle=True)
                    kwargs['chunksizes'] = ((1,) + tuple(spatial_shape) if name in image_names
                                            else (min(1024, output_count),))
                    # Omitting fill_value avoids an explicit _FillValue attr;
                    # standard prefill stays enabled until every value is written.
                    output_var = output.createVariable(
                        name, "f8" if name == "Epoch" else "f4", dimensions, **kwargs
                    )
                    attrs = {
                        a: source_var.getncattr(a) for a in source_var.ncattrs()
                        if a not in {
                            "_FillValue", "LONG_NAME", "units",
                            "remap_altitude_km", "remap_altitude_note",
                        }
                    }
                    if name == "Epoch":
                        attrs.update(
                            VAR_TYPE="support_data", LABLAXIS="GPS seconds",
                            UNITS="seconds since 1980-01-06 00:00:00 UTC",
                        )
                    else:
                        attrs.update(
                            FORMAT="F", NC_DEFLATE="1", NC_SHUFFLE="true",
                            NC_STORAGE="chunked", UNITS=units[name],
                        )
                    output_var.setncatts(attrs)
                    if name not in skip_data_variables:
                        copy_start = time.perf_counter()
                        _copy_variable_data(source_var, output_var, first, last, frame_indices)
                        if name in image_names:
                            print(f'  {output_path.name}: copy {name} {time.perf_counter()-copy_start:.1f}s', flush=True)
            update_l1f_metadata(output_path)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise


def update_l1f_metadata(output_path: Path) -> None:
    """Set C++-style product identifiers without adding Python-only history.

    Software_version is the vNN product compatibility version, as in the C++
    reference. It is not a claim that C++ generated this Python product.
    """
    output_path = Path(output_path)
    match = re.fullmatch(
        r"(?P<source>.*?l1f_(?:q20|p12|p14|bkg))_"
        r"(?P<stamp>\d{7}T\d{4})_(?P<orbit>\d{5})_v(?P<version>\d{2})"
        r"(?:_[A-Za-z0-9-]+)*", output_path.stem, re.I,
    )
    if match is None:
        raise ValueError(f"Unexpected L1F filename: {output_path.name}")
    logical_id = (
        f"{match['source']}_{match['stamp']}_{match['orbit']}_v{match['version']}"
    )
    netCDF4 = _import_netcdf4()
    with netCDF4.Dataset(output_path, "r+") as dataset:
        for name in ("Data_level", "Data_Level"):
            if name in dataset.ncattrs():
                dataset.setncattr(name, "L1F")
        if "Data_type" in dataset.ncattrs():
            dataset.setncattr("Data_type", "L1F > Level 1F")
        dataset.setncattr("Logical_source", match["source"])
        dataset.setncattr("Logical_file_id", logical_id)
        dataset.setncattr("Software_version", match["version"])
        for name in ("Generation_Date", "history", "Bad_pixel_mask", "Dark_correction"):
            if name in dataset.ncattrs():
                dataset.delncattr(name)


def _frame_key(variable, index: int) -> tuple:
    if TIME_DIM not in variable.dimensions:
        raise ValueError(f"{variable.name} has no '{TIME_DIM}' dimension")
    key = [slice(None)] * variable.ndim
    key[variable.dimensions.index(TIME_DIM)] = index
    return tuple(key)


def _radiance_write_block_size(frame_count: int) -> int:
    return max(1, min(RADIANCE_WRITE_BLOCK_SIZE, frame_count))


def _write_image_block(variable, start: int, images: np.ndarray) -> None:
    """Write a (time, y, x) image block for any position of the time axis."""
    if TIME_DIM not in variable.dimensions:
        raise ValueError(f"{variable.name} has no '{TIME_DIM}' dimension")
    block = np.asarray(images)
    if block.ndim != 3:
        raise ValueError(f"Expected a 3-D image block, got shape {block.shape}")
    time_axis = variable.dimensions.index(TIME_DIM)
    non_time_shape = tuple(
        int(variable.shape[axis])
        for axis in range(variable.ndim)
        if axis != time_axis
    )
    if int(np.prod(non_time_shape, dtype=np.int64)) != int(np.prod(block.shape[1:])):
        raise ValueError(
            f"Cannot write image block {block.shape} to {variable.name} "
            f"with dimensions {variable.dimensions} and shape {variable.shape}"
        )
    reshaped = block.reshape((block.shape[0],) + non_time_shape)
    positioned = np.moveaxis(reshaped, 0, time_axis)
    key = [slice(None)] * variable.ndim
    key[time_axis] = slice(start, start + block.shape[0])
    variable[tuple(key)] = positioned


def _read_image(variable, index: int) -> np.ndarray:
    raw = _as_plain_array(variable[_frame_key(variable, index)], fill=np.nan)
    squeezed = np.squeeze(raw)
    if squeezed.ndim != 2:
        raise ValueError(
            f"{variable.name} frame is not 2-D after selecting time: {raw.shape}"
        )
    return squeezed


def _read_image_block(variable, start: int, stop: int) -> np.ndarray:
    """Read several images with one NetCDF call and return (time, y, x)."""
    if TIME_DIM not in variable.dimensions:
        raise ValueError(f"{variable.name} has no '{TIME_DIM}' dimension")
    if stop <= start:
        raise ValueError(f"Invalid image block [{start}, {stop})")
    time_axis = variable.dimensions.index(TIME_DIM)
    key = [slice(None)] * variable.ndim
    key[time_axis] = slice(start, stop)
    raw = _as_plain_array(variable[tuple(key)], fill=np.nan)
    moved = np.moveaxis(raw, time_axis, 0)
    non_singleton_shape = tuple(size for size in moved.shape[1:] if size != 1)
    if len(non_singleton_shape) != 2:
        raise ValueError(
            f"{variable.name} block is not a 2-D image sequence after selecting "
            f"time: {raw.shape}"
        )
    return moved.reshape((moved.shape[0],) + non_singleton_shape)


def _latitude_valid_mask(dataset, index: int, shape: tuple[int, int]) -> np.ndarray | None:
    if "Latitude" not in dataset.variables:
        return None
    variable = dataset.variables["Latitude"]
    if TIME_DIM not in variable.dimensions:
        return None
    latitude = _read_image(variable, index)
    if latitude.shape != shape:
        return None
    return np.isfinite(latitude)


def _latitude_valid_mask_block(
    dataset, start: int, stop: int, shape: tuple[int, int],
) -> np.ndarray | None:
    """Read and evaluate several Latitude validity masks in one call."""
    if "Latitude" not in dataset.variables:
        return None
    variable = dataset.variables["Latitude"]
    if TIME_DIM not in variable.dimensions:
        return None
    latitude = _read_image_block(variable, start, stop)
    if latitude.shape[1:] != shape:
        return None
    return np.isfinite(latitude)


def _print_progress(message: str, separate_lines: bool) -> None:
    print(message, end="\n" if separate_lines else "\r", flush=True)


# Heuristic rejection thresholds, applied to all four filters before flat-fielding.
QUALITY_NEIGHBORS = 5
QUALITY_MIN_VALID_FRACTION = 0.50
QUALITY_EXTREME_ZERO_FRACTION = 0.95
QUALITY_SATURATION_FRACTION = 0.05
QUALITY_ROBUST_Z = 8.0
QUALITY_RELATIVE_BRIGHTNESS_CHANGE = 0.50


def raw_frame_metrics(image,geometry):
    values=np.asarray(image,dtype=np.float64)
    support=np.ones(values.shape,dtype=bool) if geometry is None else np.asarray(geometry,dtype=bool)
    n=int(support.sum());finite=support&np.isfinite(values)
    sample=values[finite]
    return dict(footprint_pixels=n,valid_fraction=float(finite.sum()/n) if n else 0.0,
                zero_fraction=float(np.mean(sample<=0)) if sample.size else 0.0,
                saturation_fraction=float(np.mean(sample>=65535/RADIANCE_SCALE)) if sample.size else 0.0,
                median_radiance=float(np.median(sample)) if sample.size else np.nan,
                p99_radiance=float(np.percentile(sample,99)) if sample.size else np.nan)


def report_path_for(output_path, suffix):
    return (output_path.parent / 'reports' / output_path.name).with_suffix(suffix)


def quality_rows(metrics,plan,first=0):
    rows=[];t=plan['epoch'][first:first+len(metrics)]
    originals=plan['left'][first:first+len(metrics)]
    for i,item in enumerate(metrics):
        reasons=[]
        if item['footprint_pixels']==0:reasons.append('no_valid_geometry')
        elif item['valid_fraction']<QUALITY_MIN_VALID_FRACTION:reasons.append('low_valid_fraction')
        if item['zero_fraction']>=QUALITY_EXTREME_ZERO_FRACTION:reasons.append('mostly_zero_or_negative')
        if item['saturation_fraction']>=QUALITY_SATURATION_FRACTION:reasons.append('many_pixels_at_count_limit')
        # Compare nearby exposures only; do not use frames across a long gap.
        neighbors=[j for j in range(max(0,i-QUALITY_NEIGHBORS),min(len(metrics),i+QUALITY_NEIGHBORS+1))
                   if j!=i and abs(t[j]-t[i]) <= (QUALITY_NEIGHBORS+1)*plan['cadence']]
        for field in ('median_radiance','p99_radiance'):
            reference=np.array([metrics[j][field] for j in neighbors
                                if metrics[j]['valid_fraction']>=QUALITY_MIN_VALID_FRACTION],dtype=float)
            reference=reference[np.isfinite(reference)]
            value=item[field]
            if len(reference)>=4 and np.isfinite(value):
                baseline=float(np.median(reference));mad=float(np.median(np.abs(reference-baseline)))
                threshold=max(QUALITY_ROBUST_Z*1.4826*mad,
                              QUALITY_RELATIVE_BRIGHTNESS_CHANGE*abs(baseline),0.001)
                if abs(value-baseline)>threshold:reasons.append(field+'_temporal_outlier')
        dt=float(t[i]-t[i-1]) if i else ''
        dt_next=float(t[i+1]-t[i]) if i+1<len(t) else ''
        near_gap=(i>0 and dt>1.5*plan['cadence']) or (i+1<len(t) and dt_next>1.5*plan['cadence'])
        rows.append(dict(l1f_frame_index=i,original_frame_index=int(originals[i]),epoch=float(t[i]),
                         dt_previous_s=dt,dt_next_s=dt_next,adjacent_to_gap=int(near_gap),
                         **item,review_flag=int(bool(reasons)),reasons=';'.join(reasons)))
    return rows


def write_quality_report(output_path,metrics,plan,first):
    rows=quality_rows(metrics,plan,first)
    target=report_path_for(output_path,'.frame_quality.csv')
    flagged=sum(r['review_flag'] for r in rows)
    if flagged:
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return dict(report=str(target) if flagged else None,frames_review_flagged=flagged,
                scope='raw retained input before filtering; within finite Latitude footprint',
                action='review only; no exclusion from filtering or flat-field average',
                thresholds=dict(min_valid_fraction=QUALITY_MIN_VALID_FRACTION,
                                zero_fraction=QUALITY_EXTREME_ZERO_FRACTION,
                                saturation_fraction=QUALITY_SATURATION_FRACTION,
                                robust_z=QUALITY_ROBUST_Z,relative_change=QUALITY_RELATIVE_BRIGHTNESS_CHANGE))


def _process_filter_file_continuous(
    input_path: Path,
    output_path: Path,
    first: int,
    last: int,
    overwrite: bool,
    progress_label: str | None = None,
    separate_progress_lines: bool = False,
    numba_threads: int = DEFAULT_NUMBA_THREADS,
    inserted_frames: np.ndarray | None = None,
    quality_metrics: list | None = None,
    calibration: dict | None = None,
    frame_indices=None,
) -> None:
    netCDF4 = _import_netcdf4()
    if set_num_threads is not None:
        set_num_threads(numba_threads)
    total_start = time.perf_counter()
    progress_prefix = f"{progress_label} " if progress_label else ""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    clone_start = time.perf_counter()
    clone_trimmed_netcdf(
        input_path, output_path, first, last,
        skip_data_variables=frozenset((RADIANCE_NAME,)), frame_indices=frame_indices,
    )
    clone_seconds = time.perf_counter() - clone_start
    frame_count = last - first + 1
    selected_indices = (np.arange(first,last+1) if frame_indices is None
                        else np.asarray(frame_indices[first:last+1], dtype=np.int64))
    inserted_frames = (np.zeros(frame_count,dtype=bool) if inserted_frames is None
                       else np.asarray(inserted_frames,dtype=bool))
    real_frame_count = int((~inserted_frames).sum())
    if inserted_frames.shape != (frame_count,) or real_frame_count == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError('No observed frames remain for flat-field construction')

    try:
        # Compact uint16 cube avoids an intermediate Radiance write and rereads.
        filter_start = time.perf_counter()
        filtered_cube = None
        flat_field = None
        packed_valid_masks = None
        valid_mask_shape = None
        validity_cache_decided = False
        radiance_read_seconds = 0.0
        input_conversion_seconds = 0.0
        filter_compute_seconds = 0.0
        roundtrip_seconds = 0.0
        filter_latitude_seconds = 0.0
        with netCDF4.Dataset(input_path, "r") as source:
            if RADIANCE_NAME not in source.variables:
                raise KeyError(f"{input_path.name} has no {RADIANCE_NAME} variable")
            source_var = source.variables[RADIANCE_NAME]
            source_var.set_auto_maskandscale(False)
            read_block_size = min(RADIANCE_READ_BLOCK_SIZE, frame_count)
            for block_out_start in range(0, frame_count, read_block_size):
                block_count = min(read_block_size, frame_count - block_out_start)
                block_input_start = first + block_out_start
                block_input_stop = block_input_start + block_count

                operation_start = time.perf_counter()
                block_indices = selected_indices[block_out_start:block_out_start+block_count]
                radiance_block = read_selected_images(source_var, block_indices)
                radiance_read_seconds += time.perf_counter() - operation_start
                operation_start = time.perf_counter()
                integer_block = radiance_to_cpp_uint16(radiance_block)
                input_conversion_seconds += time.perf_counter() - operation_start
                operation_start = time.perf_counter()
                real_block = ~inserted_frames[block_out_start:block_out_start+block_count]
                filtered_block = np.zeros_like(integer_block)
                if real_block.any():
                    filtered_block[real_block] = filter_cube_cpp(integer_block[real_block])
                filter_compute_seconds += time.perf_counter() - operation_start

                # Preserve the former intermediate float32 storage round trip.
                operation_start = time.perf_counter()
                filtered_block = CPP_FLOAT32_ROUNDTRIP_LUT[filtered_block]
                roundtrip_seconds += time.perf_counter() - operation_start
                operation_start = time.perf_counter()
                valid_block = selected_latitude_mask(source, block_indices, filtered_block.shape[1:])
                filter_latitude_seconds += time.perf_counter() - operation_start
                if quality_metrics is not None:
                    for q in range(block_count):
                        quality_metrics.append(raw_frame_metrics(radiance_block[q],
                            valid_block[q] if valid_block is not None else None))

                if not validity_cache_decided:
                    validity_cache_decided = True
                    if valid_block is not None:
                        valid_mask_shape = filtered_block.shape[1:]
                        packed_size = (filtered_block[0].size + 7) // 8
                        packed_valid_masks = np.empty(
                            (frame_count, packed_size), dtype=np.uint8
                        )
                if valid_block is not None:
                    if valid_block.shape[1:] != valid_mask_shape:
                        raise ValueError(
                            f"Latitude validity-mask shape changed from "
                            f"{valid_mask_shape} to {valid_block.shape[1:]}"
                        )
                    filtered_block[~valid_block] = 0
                    if packed_valid_masks is not None:
                        packed_valid_masks[
                            block_out_start : block_out_start + block_count
                        ] = np.packbits(
                            valid_block.reshape(block_count, -1),
                            axis=1, bitorder="little",
                        )
                elif packed_valid_masks is not None:
                    raise ValueError(
                        "Latitude validity mask became unavailable after caching started"
                    )

                if filtered_cube is None:
                    try:
                        filtered_cube = np.empty(
                            (frame_count,) + filtered_block.shape[1:], dtype=np.uint16
                        )
                    except MemoryError as exc:
                        needed_mib = frame_count * filtered_block[0].size * 2 / 1024**2
                        raise MemoryError(
                            f"Filtered cube needs about {needed_mib:.0f} MiB of RAM"
                        ) from exc
                    flat_field = np.zeros(filtered_block.shape[1:], dtype=np.float64)
                filtered_cube[
                    block_out_start : block_out_start + block_count
                ] = filtered_block
                # Preserve the frame-by-frame floating-point accumulation order.
                for filtered in filtered_block[real_block]:
                    flat_field += filtered.astype(np.float64) / real_frame_count
                processed = block_out_start + block_count
                if block_out_start == 0 or processed % 100 == 0 or processed == frame_count:
                    _print_progress(
                        f"    {progress_prefix}filtering {processed}/{frame_count}",
                        separate_progress_lines,
                    )
        if not separate_progress_lines:
            print()
        if filtered_cube is None or flat_field is None:
            raise ValueError("No frames were available for flat-field construction")
        if packed_valid_masks is not None:
            cache_mib = packed_valid_masks.nbytes / 1024**2
            print(
                f"    {progress_prefix}cached Latitude validity masks: "
                f"{cache_mib:.1f} MiB"
            )
        filter_seconds = time.perf_counter() - filter_start
        categorized_filter_seconds = (
            radiance_read_seconds + input_conversion_seconds + filter_compute_seconds
            + roundtrip_seconds + filter_latitude_seconds
        )
        filter_other_seconds = max(0.0, filter_seconds - categorized_filter_seconds)
        print(f"      {progress_prefix}Radiance block reads:    {radiance_read_seconds:.1f}s")
        print(f"      {progress_prefix}input uint16 conversion: {input_conversion_seconds:.1f}s")
        print(f"      {progress_prefix}Numba image filter:     {filter_compute_seconds:.1f}s")
        print(f"      {progress_prefix}uint16 round-trip LUT:   {roundtrip_seconds:.1f}s")
        print(f"      {progress_prefix}Latitude block reads:    {filter_latitude_seconds:.1f}s")
        print(f"      {progress_prefix}mask/cache/flat mean:    {filter_other_seconds:.1f}s")

        final_stage_start = time.perf_counter()
        flatfield_compute_seconds = 0.0
        conversion_seconds = 0.0
        latitude_seconds = 0.0
        radiance_write_seconds = 0.0
        with netCDF4.Dataset(output_path, "r+") as output:
            output_var = output.variables[RADIANCE_NAME]
            output_var.set_auto_maskandscale(False)
            write_block_size = _radiance_write_block_size(frame_count)
            print(
                f"    {progress_prefix}Radiance storage chunks: "
                f"{output_var.chunking()}; write block: {write_block_size} frame(s)"
            )
            for block_start in range(0, frame_count, write_block_size):
                block_stop = min(block_start + write_block_size, frame_count)
                block_count = block_stop - block_start
                saved_block = np.empty(
                    (block_count,) + filtered_cube.shape[1:], dtype=np.float32
                )
                valid_block = None
                if packed_valid_masks is not None:
                    operation_start = time.perf_counter()
                    pixel_count = int(np.prod(filtered_cube.shape[1:]))
                    valid_block = np.unpackbits(
                        packed_valid_masks[block_start:block_stop],
                        axis=1, count=pixel_count, bitorder="little",
                    ).reshape((block_count,) + filtered_cube.shape[1:]).astype(
                        bool, copy=False
                    )
                    latitude_seconds += time.perf_counter() - operation_start
                for local_index, index in enumerate(range(block_start, block_stop)):
                    if inserted_frames[index]:
                        saved_block[local_index] = np.nan
                        continue
                    filtered = filtered_cube[index]
                    operation_start = time.perf_counter()
                    corrected = flatfield_image_cpp(filtered, flat_field)
                    flatfield_compute_seconds += time.perf_counter() - operation_start
                    operation_start = time.perf_counter()
                    saved = corrected.astype(np.float32) / np.float32(RADIANCE_SCALE)
                    conversion_seconds += time.perf_counter() - operation_start
                    operation_start = time.perf_counter()
                    if valid_block is not None:
                        valid = valid_block[local_index]
                    else:
                        valid = _latitude_valid_mask(output, index, corrected.shape)
                    if valid is not None:
                        saved = saved.copy()
                        saved[~valid] = np.nan
                    latitude_seconds += time.perf_counter() - operation_start
                    saved_block[local_index] = saved

                operation_start = time.perf_counter()
                _write_image_block(output_var, block_start, saved_block)
                radiance_write_seconds += time.perf_counter() - operation_start
                processed = block_stop
                if block_start == 0 or processed % 100 == 0 or processed == frame_count:
                    _print_progress(
                        f"    {progress_prefix}flat-fielding {processed}/{frame_count}",
                        separate_progress_lines,
                    )
        if not separate_progress_lines:
            print()
        final_stage_seconds = time.perf_counter() - final_stage_start
        update_l1f_metadata(output_path)
        total_seconds = time.perf_counter() - total_start
        engine = "Numba" if _filter_image_numba_impl is not None else "NumPy fallback"
        print(
            f"    {progress_prefix}timing: clone {clone_seconds:.1f}s, "
            f"filter {filter_seconds:.1f}s, "
            f"final stage {final_stage_seconds:.1f}s, total {total_seconds:.1f}s "
            f"({engine})"
        )
        print(f"      {progress_prefix}flat-field calculation: {flatfield_compute_seconds:.1f}s")
        print(f"      {progress_prefix}float32 conversion:     {conversion_seconds:.1f}s")
        latitude_label = (
            "Latitude cached mask:   " if packed_valid_masks is not None
            else "Latitude read/mask:     "
        )
        print(f"      {progress_prefix}{latitude_label}{latitude_seconds:.1f}s")
        print(f"      {progress_prefix}NetCDF Radiance write:  {radiance_write_seconds:.1f}s")
        if calibration is not None:
            calibration['flat_field'] = flat_field.copy()
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


# Epoch repair runs before MLSPB boundary selection and image filtering.
# Values are in decoded Epoch seconds, as used by AWE L0C.
EPOCH_CADENCE_SECONDS = 1.1015625  # Set None to infer median positive spacing.
EPOCH_INTERVAL_TOLERANCE = 0.01  # seconds; allows timestamp quantization/jitter
EPOCH_DUPLICATE_TOLERANCE = 1e-6
def read_epoch(path):
    with _import_netcdf4().Dataset(path) as ds:
        var = ds.variables['Epoch']
        if var.dimensions != ('time',):
            raise ValueError(f'{path}: Epoch must have dimensions (time,)')
        return _as_plain_array(var[:], fill=np.nan).astype(np.float64)


def verify_continuity(epoch, cadence):
    values = np.asarray(epoch,dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Epoch empty or non-finite')
    dt = np.diff(values)
    bad = np.flatnonzero(dt <= EPOCH_DUPLICATE_TOLERANCE)
    if bad.size:
        i=int(bad[0]);raise ValueError(f'Epoch not strictly increasing: {i}->{i+1}, dt={dt[i]}')
    gaps = dt > 1.5*cadence
    irregular = (np.abs(dt-cadence)>EPOCH_INTERVAL_TOLERANCE)&~gaps
    return dict(frames=len(values),min_dt_s=float(dt.min()) if len(dt) else None,
                max_dt_s=float(dt.max()) if len(dt) else None,
                continuity='GAPS_OR_IRREGULAR_INTERVALS_RETAINED' if (gaps|irregular).any() else 'PASS',
                gap_intervals=int(gaps.sum()),irregular_intervals=int(irregular.sum()),
                monotonicity='PASS')


def epoch_repair_plan(epoch):
    epoch=np.asarray(epoch,dtype=np.float64)
    if epoch.ndim!=1 or not epoch.size or not np.isfinite(epoch).all():
        raise ValueError('Cannot process empty or invalid Epoch')
    positive=np.diff(epoch);positive=positive[positive>EPOCH_DUPLICATE_TOLERANCE]
    cadence=EPOCH_CADENCE_SECONDS
    if cadence is None:
        if not positive.size:raise ValueError('Cannot infer cadence')
        cadence=float(np.median(positive))
    if not np.isfinite(cadence) or cadence<=0:raise ValueError('Invalid cadence')
    kept=[0];events=[]
    for i in range(1,len(epoch)):
        previous=kept[-1];dt=float(epoch[i]-epoch[previous])
        if abs(dt)<=EPOCH_DUPLICATE_TOLERANCE:
            events.append(dict(event='duplicate',kept_original_index=previous,
                               removed_original_index=i,dt_s=dt));continue
        if dt<0:raise ValueError(f'Backward Epoch {previous}->{i}: {dt} seconds')
        if abs(dt-cadence)>EPOCH_INTERVAL_TOLERANCE:
            events.append(dict(event='gap' if dt>1.5*cadence else 'irregular_interval',
                               index_before=previous,index_after=i,dt_s=dt,action='unchanged'))
        kept.append(i)
    kept=np.asarray(kept,dtype=int);values=epoch[kept]
    return dict(left=kept,right=kept,weight=np.zeros(len(kept)),epoch=values,
                inserted=np.zeros(len(kept),dtype=bool),cadence=cadence,
                original_frames=len(epoch),events=events,check=verify_continuity(values,cadence))


def repair_report(plan):
    return dict(gap_treatment='unchanged',original_frames=plan['original_frames'],
                repaired_frames=len(plan['epoch']),inserted_frames=0,
                removed_duplicates=sum(e['event']=='duplicate' for e in plan['events']),
                removed_bad_frames=plan.get('removed_bad_frames',0),
                expected_dt_s=plan['cadence'],tolerance_s=EPOCH_INTERVAL_TOLERANCE,
                events=plan['events'],after_repair=plan['check'],
                indices='zero-based original L0C indices; retained range uses indices after duplicate and bad-frame removal',
                interpolation='none',duplicate_policy='keep first timestamp occurrence; remove later repeats')


def write_repaired_input(source_path,target_path,plan):
    """Raw storage copy of retained samples; no decoding/repacking/interpolation."""
    nc=_import_netcdf4()
    with nc.Dataset(source_path) as src,nc.Dataset(target_path,'w') as dst:
        if src.groups:raise ValueError('Nested NetCDF groups unsupported')
        dst.setncatts({k:src.getncattr(k) for k in src.ncattrs()})
        for name,dim in src.dimensions.items():
            dst.createDimension(name,len(plan['epoch']) if name=='time' else len(dim))
        for name,var in src.variables.items():
            kwargs={}
            if '_FillValue' in var.ncattrs():kwargs['fill_value']=var.getncattr('_FillValue')
            if var.ndim and np.dtype(var.dtype).kind in 'fiu':kwargs.update(zlib=True,complevel=1)
            out=dst.createVariable(name,var.datatype,var.dimensions,**kwargs)
            out.setncatts({k:var.getncattr(k) for k in var.ncattrs() if k!='_FillValue'})
            var.set_auto_maskandscale(False);out.set_auto_maskandscale(False)
            if 'time' not in var.dimensions:out[...]=var[...];continue
            axis=var.dimensions.index('time')
            for start in range(0,len(plan['epoch']),32):
                stop=min(start+32,len(plan['epoch']))
                a=[slice(None)]*var.ndim;b=a.copy()
                a[axis]=plan['left'][start:stop];b[axis]=slice(start,stop)
                out[tuple(b)]=var[tuple(a)]
    verify_continuity(read_epoch(target_path),plan['cadence'])


def process_filter_file(input_path, output_path, first, last, overwrite,
                        progress_label=None, separate_progress_lines=False,
                        numba_threads=DEFAULT_NUMBA_THREADS, box_config=None, shared_plan=None):
    """first/last index the repaired timeline selected during orbit preparation."""
    plan = shared_plan if shared_plan is not None else epoch_repair_plan(read_epoch(input_path))
    report = repair_report(plan)
    report.update(source=str(input_path), output=str(output_path), retained_first=first, retained_last=last)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    report_path = report_path_for(output_path,'.epoch_repair.json')
    report_enabled=parse_name(input_path).group('filter').lower() == 'q20'
    quality_metrics=None  # Screening already completed jointly before task submission.
    try:
        if box_config is not None:
            boxes, reference = box_config
            process_box_masked_file(input_path, output_path, boxes, reference, overwrite,
                                   progress_label, numba_threads, quality_metrics, plan['left'])
        else:
            _process_filter_file_continuous(input_path, output_path, first, last, overwrite,
                    progress_label, separate_progress_lines, numba_threads,
                    plan['inserted'][first:last+1], quality_metrics, frame_indices=plan['left'])
        with _import_netcdf4().Dataset(output_path,'r+') as ds:
            selected = slice(first,last+1)
            flag = ds.createVariable('Frame_Interpolated','u1',('time',),fill_value=False)
            flag.setncatts(dict(long_name='Frame inserted during Epoch gap repair',
                flag_values=np.array([0,1],dtype='u1'),
                flag_meanings='observed interpolated_ancillary_missing_radiance',
                comment='1: ancillary variables interpolated; Radiance entirely NaN. Excluded from flat-field averaging.',
                units='1'))
            flag[:] = plan['inserted'][selected].astype('u1')
            index = ds.createVariable('Original_Frame_Index','i4',('time',),fill_value=False)
            index.setncatts(dict(long_name='Zero-based frame index in original L0C input',
                comment='-1 denotes an inserted gap frame. Indices refer to the source file before duplicate removal or trimming.',
                units='1'))
            index[:] = np.where(plan['inserted'][selected],-1,plan['left'][selected]).astype('i4')
            ds.Epoch_Repair_Source = str(input_path)
            ds.Epoch_Gap_Policy = 'preserve all gaps; no interpolation'
            ds.Bad_Frame_Policy = 'Screen all four filters; remove union of rejected original indices before filtering and flat-fielding'
            ds.Bad_Frames_Removed = int(plan.get('removed_bad_frames', 0))
            ds.Epoch_Repair = json.dumps(report)
        report['output_continuity'] = verify_continuity(read_epoch(output_path),plan['cadence'])
        report['removed_bad_frames'] = plan.get('removed_bad_frames', 0)
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAILED',error=str(exc))
        if quality_metrics:
            try:
                report['quality_screen'] = write_quality_report(output_path,quality_metrics,plan,first)
            except Exception as quality_error:
                report['quality_report_error'] = str(quality_error)
        raise
    finally:
        has_abnormal_frames=report.get('quality_screen',{}).get('frames_review_flagged',0)>0
        if report_enabled and (report['events'] or has_abnormal_frames):
            report_path.parent.mkdir(parents=True,exist_ok=True)
            report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"{progress_label or input_path.name}: inserted={report['inserted_frames']}, removed duplicates={report['removed_duplicates']}; continuity {report['output_continuity']['continuity']}",flush=True)


def save_bad_frame_previews(inputs, rejected, findings, output_dir):
    """Save original, unfiltered images before any common removal is applied."""
    if not rejected:
        return {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from contextlib import ExitStack
    import textwrap
    folder = output_dir/'reports'/'bad_frames'
    folder.mkdir(parents=True, exist_ok=True)
    images = {}
    with ExitStack() as stack:
        datasets = {f:stack.enter_context(_import_netcdf4().Dataset(inputs[f])) for f in FILTERS}
        for index in sorted(rejected):
            fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
            try:
                for filt, ax in zip(FILTERS, axes.flat):
                    raw = _read_image(datasets[filt][RADIANCE_NAME], index)
                    finite = raw[np.isfinite(raw)]
                    lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
                    if hi <= lo: hi = lo + max(abs(lo)*0.01, 0.001)
                    im = ax.imshow(np.ma.masked_invalid(raw), origin='lower', cmap='viridis', vmin=lo, vmax=hi)
                    reasons = '; '.join(r['reasons'] for r in findings
                                       if r['filter']==filt and r['original_frame_index']==index)
                    ax.set_title(filt + '\n' + textwrap.fill(reasons or 'No screening trigger', 48), fontsize=10)
                    fig.colorbar(im, ax=ax, label='Raw radiance (input units)')
                    ax.set_xlabel('Detector column'); ax.set_ylabel('Detector row')
                orbit = parse_name(inputs['q20']).group('orbit')
                fig.suptitle(f'Orbit {orbit} | original frame {index} (zero-based)\n'
                             'Removed from all four filters; independent 1-99% display scales', fontsize=13)
                target = folder/(inputs['q20'].stem + f'_bad_frame_{index:06d}.png')
                fig.savefig(target, dpi=120)
                images[index] = str(target)
            finally:
                plt.close(fig)
    return images


def screen_all_filters(inputs, plan, output_dir):
    """One common removal map; comparisons use each filter's own brightness."""
    screening_start = time.perf_counter()
    rejected = set()
    findings = []
    for filt in FILTERS:
        metrics = []
        with _import_netcdf4().Dataset(inputs[filt]) as ds:
            rad = ds[RADIANCE_NAME]
            # Read sequential blocks, retaining only the first duplicate occurrence.
            kept = set(int(i) for i in plan['left'])
            for start in range(0, plan['original_frames'], RADIANCE_READ_BLOCK_SIZE):
                stop = min(start + RADIANCE_READ_BLOCK_SIZE, plan['original_frames'])
                block = _read_image_block(rad, start, stop)
                geometry = _latitude_valid_mask_block(ds, start, stop, block.shape[1:])
                for offset, raw in enumerate(block):
                    if start + offset in kept:
                        metrics.append(raw_frame_metrics(raw, None if geometry is None else geometry[offset]))
        for row in quality_rows(metrics, plan):
            if row['review_flag']:
                rejected.add(row['original_frame_index'])
                findings.append(dict(row, filter=filt))
        print(f"Quality screen {filt}: {sum(r['filter']==filt for r in findings)} flagged frames", flush=True)
    print(f'All-filter screening: {time.perf_counter()-screening_start:.1f}s; '
          f'{len(rejected)} unique frames rejected', flush=True)
    preview_start = time.perf_counter()
    previews = (save_bad_frame_previews(inputs, rejected, findings, output_dir)
                if SAVE_BAD_FRAME_IMAGES else {})
    print(f'Bad-frame previews: {time.perf_counter()-preview_start:.1f}s', flush=True)
    for row in findings:
        row['image_path'] = previews.get(row['original_frame_index'], '')
    keep = np.array([int(i) not in rejected for i in plan['left']], dtype=bool)
    clean = dict(plan)
    for key in ('left','right','weight','epoch','inserted'):
        clean[key] = plan[key][keep]
    clean['events'] = list(plan['events']) + [dict(event='bad_frame', removed_original_index=i,
        action='removed_from_all_four_filters') for i in sorted(rejected)]
    clean['removed_bad_frames'] = len(rejected)
    # Always record screening completion, even when nothing was rejected.
    target = report_path_for(output_path_for(inputs['q20'], output_dir,
                          parse_name(inputs['q20']).group('stamp')), '.bad_frames.csv')
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ['image_path','filter','screening_status','frames_screened_per_filter','removed_frames_all_filters',
              'original_frame_index','epoch','reasons','review_flag','action',
              'adjacent_to_gap','dt_previous_s','dt_next_s','footprint_pixels',
              'valid_fraction','zero_fraction','saturation_fraction','median_radiance','p99_radiance']
    with target.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for row in findings or [dict(filter='all', review_flag=0)]:
            writer.writerow(dict(row, screening_status='completed',
                frames_screened_per_filter=len(plan['epoch']), removed_frames_all_filters=len(rejected),
                action='remove_from_all_four_filters' if findings else 'none'))
    if not keep.any():
        raise ValueError('Bad-frame screening rejected every frame; see bad_frames.csv')
    clean['check'] = verify_continuity(clean['epoch'], clean['cadence'])
    return clean


def _prepare_orbit_tasks(
    mlspb_q20_path: Path,
    l0c_root: Path,
    output_root: Path,
    mlspb_root: Path,
    selection_mode: str,
    skip_mlspb: bool = False,
    mlspb_mode: str = "boxes",
    min_clean_frames: int = DEFAULT_MIN_CLEAN_FRAMES,
) -> list[tuple]:
    """Resolve one orbit into four independent file-processing tasks."""
    inputs = matching_l0c_filter_paths(mlspb_q20_path, l0c_root, mlspb_root)
    plans = {f:epoch_repair_plan(read_epoch(p)) for f,p in inputs.items()}
    reference = plans['q20']
    for filt, plan in plans.items():
        if (plan['original_frames'] != reference['original_frames'] or
                not np.array_equal(plan['left'],reference['left']) or
                not np.array_equal(plan['right'],reference['right']) or
                not np.allclose(plan['epoch'], reference['epoch'], rtol=0, atol=EPOCH_DUPLICATE_TOLERANCE)):
            raise ValueError(f'{filt}: Epoch repair pattern differs from q20; cannot align filters safely')
    if skip_mlspb:
        flags = np.zeros(reference['original_frames'],dtype=bool)
    else:
        flags = read_mlspb_flags(mlspb_q20_path)
        ml_epoch, source_epoch = read_epoch(mlspb_q20_path), read_epoch(inputs['q20'])
        if (flags.size != len(source_epoch) or ml_epoch.shape != source_epoch.shape or
                not np.allclose(ml_epoch,source_epoch,rtol=0,atol=EPOCH_DUPLICATE_TOLERANCE)):
            raise ValueError('MLSPB timestamps/length do not match the original q20 input; rerun MLSPB on these L0C files')
    output_dir = (output_root / mlspb_q20_path.parent.relative_to(mlspb_root)
                  if mlspb_root.is_dir() else output_root)
    if BAD_FRAME_SCREENING:
        reference = screen_all_filters(inputs, reference, output_dir)
    else:
        print('Bad-frame screening disabled; duplicate removal remains enabled.', flush=True)
    # Remap MLSPB flags using the shared duplicate + bad-frame removal map.
    flags = flags[reference['left']]
    print(f"Epoch repair: {reference['original_frames']} -> {len(flags)} frames; "
          f"inserted={int(reference['inserted'].sum())}; "
          f"duplicates removed={sum(e['event']=='duplicate' for e in reference['events'])}; "
          f"cadence={reference['cadence']:.9f} s; all four filters PASS")
    box_config = None
    if not skip_mlspb and mlspb_mode == 'boxes':
        boxes = read_mlspb_boxes(mlspb_q20_path)[reference['left']]
        lo, last_trim = select_frame_range(flags, selection_mode)
        hi = last_trim + 1
        clean_count = int((~flags[lo:hi]).sum())
        if clean_count < min_clean_frames:
            raise ValueError(f'Trim reference contains {clean_count} fully unflagged frames; need {min_clean_frames}')
        # Preserve even the historical boundary behavior when selection_mode='cxx'.
        # Excluding flagged reference frames here would change the old flat field.
        box_config = (boxes, (lo, hi))
    first, last = ((0,len(flags)-1) if skip_mlspb or box_config is not None
                   else select_frame_range(flags, selection_mode))
    retained = last - first + 1
    epoch_first = float(reference['epoch'][first])
    stamp = (datetime(2023,1,1,tzinfo=timezone.utc) +
             timedelta(seconds=int(np.floor(epoch_first-1356566418.0)))).strftime('%Y%jT%H%M')
    match = parse_name(mlspb_q20_path)
    print(
        f"Orbit {match.group('orbit')}: retaining repaired frames {first}..{last} "
        f"({retained}/{flags.size}); MLSPB-positive frames={int(flags.sum())}"
    )
    if mlspb_root.is_dir():
        relative_parent = mlspb_q20_path.parent.relative_to(mlspb_root)
        output_dir = output_root / relative_parent
    else:
        output_dir = output_root
    tasks = []
    for filter_name in FILTERS:
        input_path = inputs[filter_name]
        output_path = output_path_for(input_path, output_dir, stamp)
        print(f"  {filter_name}: {input_path.name} -> {output_path.name}")
        tasks.append((
            mlspb_q20_path, match.group("orbit"), filter_name,
            input_path, output_path, first, last, box_config, reference,
        ))
    return tasks


def process_orbit_batch(
    mlspb_q20_paths: Sequence[Path],
    l0c_root: Path,
    output_root: Path,
    mlspb_root: Path,
    selection_mode: str,
    overwrite: bool,
    workers: int,
    numba_threads: int = DEFAULT_NUMBA_THREADS,
    skip_mlspb: bool = False,
    mlspb_mode: str = "boxes",
    min_clean_frames: int = DEFAULT_MIN_CLEAN_FRAMES,
) -> tuple[list[Path], set[Path]]:
    """Process all filter files from several orbits in one shared pool."""
    tasks = []
    failed_orbits: set[Path] = set()
    for mlspb_q20_path in mlspb_q20_paths:
        try:
            tasks.extend(_prepare_orbit_tasks(
                mlspb_q20_path, l0c_root, output_root, mlspb_root, selection_mode, skip_mlspb, mlspb_mode, min_clean_frames,
            ))
        except Exception as exc:
            failed_orbits.add(mlspb_q20_path)
            print(f"ERROR preparing {mlspb_q20_path}: {exc}", file=sys.stderr)
    if not tasks:
        return [], failed_orbits
    completed: dict[tuple[Path, str], Path] = {}
    if workers == 1:
        for orbit_path, orbit, filter_name, input_path, output_path, first, last, box_config, shared_plan in tasks:
            label = f"{orbit}/{filter_name}"
            try:
                process_filter_file(
                    input_path, output_path, first, last, overwrite, label,
                    False, numba_threads, box_config, shared_plan,
                )
                completed[(orbit_path, filter_name)] = output_path
            except Exception as exc:
                failed_orbits.add(orbit_path)
                print(f"ERROR processing {label}: {exc}", file=sys.stderr)
    else:
        worker_count = min(workers, len(tasks))
        print(
            f"Processing {len(mlspb_q20_paths)} orbit(s), {len(tasks)} files "
            f"with {worker_count} workers"
        )
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=get_context("spawn")) as executor:
            future_to_task = {}
            for task in tasks:
                orbit_path, orbit, filter_name, input_path, output_path, first, last, box_config, shared_plan = task
                label = f"{orbit}/{filter_name}"
                future = executor.submit(
                    process_filter_file, input_path, output_path, first, last,
                    overwrite, label, True, numba_threads, box_config, shared_plan,
                )
                future_to_task[future] = task
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                orbit_path, orbit, filter_name, _, output_path, _, _, _, _ = task
                label = f"{orbit}/{filter_name}"
                try:
                    future.result()
                    completed[(orbit_path, filter_name)] = output_path
                    print(f"  {label}: completed")
                except Exception as exc:
                    failed_orbits.add(orbit_path)
                    print(f"ERROR processing {label}: {exc}", file=sys.stderr)
    outputs = []
    for orbit_path, _, filter_name, _, _, _, _, _, _ in tasks:
        output_path = completed.get((orbit_path, filter_name))
        if output_path is not None:
            outputs.append(output_path)
    return outputs, failed_orbits


def process_orbit(
    mlspb_q20_path: Path,
    l0c_root: Path,
    output_root: Path,
    mlspb_root: Path,
    selection_mode: str,
    overwrite: bool,
    workers: int,
    numba_threads: int = DEFAULT_NUMBA_THREADS,
) -> list[Path]:
    """Backward-compatible wrapper for processing a single orbit."""
    outputs, failed_orbits = process_orbit_batch(
        [mlspb_q20_path], l0c_root, output_root, mlspb_root, selection_mode,
        overwrite, workers, numba_threads,
    )
    if failed_orbits:
        raise RuntimeError(f"Orbit processing failed: {mlspb_q20_path}")
    return outputs


def run_self_test() -> None:
    """Run the original numerical processing checks without opening input files."""
    base = np.full((30, 30), 10000, dtype=np.uint16)
    test = base.copy()
    test[10, 10] = 30000
    test[20, 20] = 100
    filtered = filter_image_cpp(test)
    filtered_numpy = _filter_image_numpy(test)
    assert filtered.shape == test.shape
    assert filtered.dtype == np.uint16
    assert np.array_equal(filtered, filtered_numpy)
    assert filtered[10, 10] == 10000
    assert filtered[20, 20] == 10000
    cube = np.stack((test, np.rot90(test).copy()))
    assert np.array_equal(
        filter_cube_cpp(cube),
        np.stack([_filter_image_numpy(image) for image in cube]),
    )
    all_counts = np.arange(65536, dtype=np.uint16)
    old_roundtrip = radiance_to_cpp_uint16(
        all_counts.astype(np.float32) / np.float32(RADIANCE_SCALE)
    )
    assert np.array_equal(CPP_FLOAT32_ROUNDTRIP_LUT, old_roundtrip)
    stack = np.stack([base, base + 100, base + 200]).astype(np.float64)
    ff = stack.mean(axis=0)
    corrected = flatfield_image_cpp(stack[1].astype(np.uint16), ff)
    assert corrected.shape == base.shape
    assert corrected.dtype == np.uint16
    image_with_invalid_region = stack[1].astype(np.uint16)
    image_with_invalid_region[:5, :5] = 0
    ff_with_invalid_region = ff.copy()
    ff_with_invalid_region[:5, :5] = 0
    corrected_invalid = flatfield_image_cpp(
        image_with_invalid_region, ff_with_invalid_region
    )
    assert np.all(corrected_invalid[:4, :4] == 0)
    flags = np.array([True, True, False, False, True, True])
    assert select_frame_range(flags, "clean") == (2, 3)
    assert select_frame_range(np.zeros(5, dtype=bool), "cxx") == (0, 4)
    print("Self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AWE MLSPB filtering and flat fielding with C++-style L1F output."
    )
    parser.add_argument(
        "mlspb_input", type=Path, nargs="?",
        help="A q20 NetCDF file with MLSPB, or the L0D directory containing it.",
    )
    parser.add_argument(
        "l0c_input", type=Path, nargs="?",
        help="Directory containing matching L0C q20, p12, p14, and bkg files.",
    )
    parser.add_argument("output", type=Path, nargs="?", help="Output directory for L1F files.")
    parser.add_argument(
        "--selection-mode", choices=("cxx", "clean"), default=DEFAULT_SELECTION_MODE,
        help="cxx preserves the C++ boundary behavior; clean removes all flagged edge frames.",
    )
    parser.add_argument(
        "--recursive", action="store_true", default=DEFAULT_RECURSIVE,
        help="Recursively search the input directory and preserve subdirectories.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=DEFAULT_OVERWRITE,
        help="Replace output files if they already exist.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Simultaneous filter-file processes (default: {DEFAULT_WORKERS}); 1 disables multiprocessing.",
    )
    parser.add_argument(
        "--orbits-at-a-time", type=int, default=DEFAULT_ORBITS_AT_A_TIME,
        help=f"Orbits in each batch (default: {DEFAULT_ORBITS_AT_A_TIME}); each orbit has four filter files.",
    )
    parser.add_argument(
        "--numba-threads", type=int, default=DEFAULT_NUMBA_THREADS,
        help=f"Numba threads per worker (default: {DEFAULT_NUMBA_THREADS}).",
    )
    parser.add_argument('--gap-treatment',choices=('leave',),default='leave',
                        help='Compatibility option: gaps are always left unchanged.')
    parser.add_argument("--skip-mlspb", action="store_true",
                        help="Isolated L0C test: keep the full repaired orbit, without MLSPB trimming.")
    parser.add_argument(
        "--self-test", action="store_true",
        help="Test numerical processing functions without opening NetCDF files.",
    )
    parser.add_argument('--mlspb-mode', choices=('boxes', 'trim'), default=DEFAULT_MLSPB_MODE,
                        help='boxes: preserve legacy trim result and extend with its flat field; trim: legacy behavior')
    parser.add_argument('--min-clean-frames', type=int, default=DEFAULT_MIN_CLEAN_FRAMES,
                        help='Minimum fully unflagged frames within the legacy trim reference interval')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run_start = time.perf_counter()
    args = build_parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.min_clean_frames < 1: raise SystemExit('--min-clean-frames must be positive')
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.orbits_at_a_time < 1:
        raise SystemExit("--orbits-at-a-time must be at least 1")
    if args.numba_threads < 1:
        raise SystemExit("--numba-threads must be at least 1")
    selected_mlspb_input = args.mlspb_input or (
        Path(DEFAULT_MLSPB_INPUT) if DEFAULT_MLSPB_INPUT else None
    )
    selected_l0c_input = args.l0c_input or (
        Path(DEFAULT_L0C_INPUT) if DEFAULT_L0C_INPUT else None
    )
    if args.skip_mlspb:
        selected_mlspb_input = args.mlspb_input or selected_l0c_input
        print('TEST MODE: MLSPB trimming disabled; retaining the full repaired orbit.',flush=True)
    selected_output = args.output or (Path(DEFAULT_OUTPUT) if DEFAULT_OUTPUT else None)
    if selected_mlspb_input is None or selected_l0c_input is None or selected_output is None:
        raise SystemExit(
            "MLSPB_INPUT, L0C_INPUT, and OUTPUT are required unless --self-test "
            "is used. Supply them on the command line or set the default paths "
            "near the top of the script."
        )
    mlspb_input = selected_mlspb_input.expanduser().resolve()
    l0c_input = selected_l0c_input.expanduser().resolve()
    output_root = selected_output.expanduser().resolve()
    q20_files = discover_mlspb_q20_files(mlspb_input, args.recursive)
    output_root.mkdir(parents=True, exist_ok=True)

    # Compile once before Windows workers import and load the cached functions.
    if _filter_cube_numba_impl is not None:
        print("Preparing the cached Numba filter...")
        if set_num_threads is not None:
            set_num_threads(args.numba_threads)
        filter_cube_cpp(np.zeros((1, 4, 4), dtype=np.uint16))
        print(f"Numba threads per worker: {get_num_threads()}; workers: {args.workers}")
    all_outputs: list[Path] = []
    failed_orbits: set[Path] = set()
    for batch_start in range(0, len(q20_files), args.orbits_at_a_time):
        orbit_batch = q20_files[batch_start : batch_start + args.orbits_at_a_time]
        batch_number = batch_start // args.orbits_at_a_time + 1
        batch_count = (len(q20_files) + args.orbits_at_a_time - 1) // args.orbits_at_a_time
        print(
            f"Starting orbit batch {batch_number}/{batch_count}: "
            f"{len(orbit_batch)} orbit(s), up to {len(orbit_batch) * len(FILTERS)} files"
        )
        batch_outputs, batch_failures = process_orbit_batch(
            mlspb_q20_paths=orbit_batch,
            l0c_root=l0c_input,
            output_root=output_root,
            mlspb_root=mlspb_input,
            selection_mode=args.selection_mode,
            overwrite=args.overwrite,
            workers=args.workers,
            numba_threads=args.numba_threads,
            skip_mlspb=args.skip_mlspb,
            mlspb_mode=args.mlspb_mode, min_clean_frames=args.min_clean_frames,
        )
        all_outputs.extend(batch_outputs)
        failed_orbits.update(batch_failures)
    failures = len(failed_orbits)
    print(f"Finished: {len(all_outputs)} output file(s), {failures} failed orbit(s)")
    elapsed = time.perf_counter() - run_start
    print(f"Total wall time: {elapsed:.1f}s ({elapsed / 60:.2f} min)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
