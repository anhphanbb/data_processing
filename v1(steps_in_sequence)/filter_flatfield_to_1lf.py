#!/usr/bin/env python3
"""Filter and flat-field AWE L0C images after MLSPB detection.

Standalone version with the C++ L1F NetCDF schema from the comparison report.

Workflow
--------
1. Read MLSPB from q20 in L0D (the filename may still contain l0c).
2. Select one common retained frame range.
3. Process matching L0C q20, p12, p14, and bkg files.
4. Filter retained images using the existing NumPy/Numba implementation.
5. Calculate a temporal-mean flat field separately for each filter.
6. Write flat-fielded Radiance and trimmed ancillary data in C++-style L1F files.

Output changes
--------------
* Only the 12 variables present in the supplied C++ report are written.
* Image dimensions are (time, y_cross_track, x_along_track).
* time is unlimited; both spatial dimensions are fixed.
* Dimension IDs and variable definition order match the legacy C++ writer.
  Dimensions are created x, y, time; image axes remain time, y, x.
* Spatial axes are renamed, not transposed.
* Reported global and variable metadata differences are corrected.
* Extra MLSPB/noise/dark-response variables and box dimensions are omitted.
* Compression is zlib level 1 with shuffle; source chunk sizes are retained.

Common attributes not printed in the differences-only report are preserved
from the input. Exact undocumented C++ attribute rules and numeric attribute
storage types cannot be established from that report alone. Acquisition
metadata use source values when present, otherwise the report's defaults.
The full filename date/time comes directly from the first retained Epoch,
including midnight/year rollover. Bad_pixel_mask and Dark_correction are
omitted to match the supplied C++ report. The processing suffix is retained;
Logical_file_id uses the corrected timestamp and omits the suffix. Software_version is the
vNN product compatibility version, not a claim of C++ execution.

Requirements
------------
    pip install numpy netCDF4 numba

Numba is optional but strongly recommended. No updater or other local Python
module is needed. Edit DEFAULT_MLSPB_INPUT, DEFAULT_L0C_INPUT, and DEFAULT_OUTPUT
below to run directly in Spyder or another IDE.

Terminal examples
-----------------
    python mlspb_filter_flatfield_cpp_metadata.py
    python mlspb_filter_flatfield_cpp_metadata.py --self-test
    python mlspb_filter_flatfield_cpp_metadata.py MLSPB_INPUT L0C_INPUT OUTPUT --workers 4 --numba-threads 4

Existing outputs are protected by default. Use --overwrite to replace them.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
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
DEFAULT_MLSPB_INPUT = r"E:\soc\l0d\2023\12"
DEFAULT_L0C_INPUT = r"E:\soc\l0c\2023\12"
DEFAULT_OUTPUT = r"E:\soc\l1f\2023\12"
DEFAULT_SELECTION_MODE = "cxx"  # "cxx" or "clean"
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


def _copy_variable_data(source_var, output_var, first: int, last: int) -> None:
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
        output_var[tuple(dst_key)] = source_var[tuple(src_key)]


# C++ L1F writer functions are embedded below (no companion module required).

def clone_trimmed_netcdf(
    input_path: Path,
    output_path: Path,
    first: int,
    last: int,
    skip_data_variables: frozenset[str] = frozenset(),
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
        if first < 0 or last < first or last >= len(source.dimensions["time"]):
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
                    # Keep source chunk sizes for throughput, but set actual
                    # zlib level 1 and shuffle to agree with C++ storage labels.
                    kwargs = dict(zlib=True, complevel=1, shuffle=True)
                    chunks = source_var.chunking()
                    if isinstance(chunks, (list, tuple)):
                        kwargs["chunksizes"] = tuple(
                            max(1, min(int(c), output_count if i == 0 else source_var.shape[i]))
                            for i, c in enumerate(chunks)
                        )
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
                        _copy_variable_data(source_var, output_var, first, last)
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


def process_filter_file(
    input_path: Path,
    output_path: Path,
    first: int,
    last: int,
    overwrite: bool,
    progress_label: str | None = None,
    separate_progress_lines: bool = False,
    numba_threads: int = DEFAULT_NUMBA_THREADS,
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
        skip_data_variables=frozenset((RADIANCE_NAME,)),
    )
    clone_seconds = time.perf_counter() - clone_start
    frame_count = last - first + 1

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
                radiance_block = _read_image_block(
                    source_var, block_input_start, block_input_stop
                )
                radiance_read_seconds += time.perf_counter() - operation_start
                operation_start = time.perf_counter()
                integer_block = radiance_to_cpp_uint16(radiance_block)
                input_conversion_seconds += time.perf_counter() - operation_start
                operation_start = time.perf_counter()
                filtered_block = filter_cube_cpp(integer_block)
                filter_compute_seconds += time.perf_counter() - operation_start

                # Preserve the former intermediate float32 storage round trip.
                operation_start = time.perf_counter()
                filtered_block = CPP_FLOAT32_ROUNDTRIP_LUT[filtered_block]
                roundtrip_seconds += time.perf_counter() - operation_start
                operation_start = time.perf_counter()
                valid_block = _latitude_valid_mask_block(
                    source, block_input_start, block_input_stop,
                    filtered_block.shape[1:],
                )
                filter_latitude_seconds += time.perf_counter() - operation_start

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
                for filtered in filtered_block:
                    flat_field += filtered.astype(np.float64) / frame_count
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
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


def _prepare_orbit_tasks(
    mlspb_q20_path: Path,
    l0c_root: Path,
    output_root: Path,
    mlspb_root: Path,
    selection_mode: str,
) -> list[tuple[Path, str, str, Path, Path, int, int]]:
    """Resolve one orbit into four independent file-processing tasks."""
    inputs = matching_l0c_filter_paths(mlspb_q20_path, l0c_root, mlspb_root)
    flags = read_mlspb_flags(mlspb_q20_path)
    first, last = select_frame_range(flags, selection_mode)
    retained = last - first + 1
    stamp = adjusted_stamp(inputs["q20"], first)
    match = parse_name(mlspb_q20_path)
    print(
        f"Orbit {match.group('orbit')}: retaining frames {first}..{last} "
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
            input_path, output_path, first, last,
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
) -> tuple[list[Path], set[Path]]:
    """Process all filter files from several orbits in one shared pool."""
    tasks = []
    failed_orbits: set[Path] = set()
    for mlspb_q20_path in mlspb_q20_paths:
        try:
            tasks.extend(_prepare_orbit_tasks(
                mlspb_q20_path, l0c_root, output_root, mlspb_root, selection_mode,
            ))
        except Exception as exc:
            failed_orbits.add(mlspb_q20_path)
            print(f"ERROR preparing {mlspb_q20_path}: {exc}", file=sys.stderr)
    if not tasks:
        return [], failed_orbits
    completed: dict[tuple[Path, str], Path] = {}
    if workers == 1:
        for orbit_path, orbit, filter_name, input_path, output_path, first, last in tasks:
            label = f"{orbit}/{filter_name}"
            try:
                process_filter_file(
                    input_path, output_path, first, last, overwrite, label,
                    False, numba_threads,
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
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {}
            for task in tasks:
                orbit_path, orbit, filter_name, input_path, output_path, first, last = task
                label = f"{orbit}/{filter_name}"
                future = executor.submit(
                    process_filter_file, input_path, output_path, first, last,
                    overwrite, label, True, numba_threads,
                )
                future_to_task[future] = task
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                orbit_path, orbit, filter_name, _, output_path, _, _ = task
                label = f"{orbit}/{filter_name}"
                try:
                    future.result()
                    completed[(orbit_path, filter_name)] = output_path
                    print(f"  {label}: completed")
                except Exception as exc:
                    failed_orbits.add(orbit_path)
                    print(f"ERROR processing {label}: {exc}", file=sys.stderr)
    outputs = []
    for orbit_path, _, filter_name, _, _, _, _ in tasks:
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
    parser.add_argument(
        "--self-test", action="store_true",
        help="Test numerical processing functions without opening NetCDF files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
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
        )
        all_outputs.extend(batch_outputs)
        failed_orbits.update(batch_failures)
    failures = len(failed_orbits)
    print(f"Finished: {len(all_outputs)} output file(s), {failures} failed orbit(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
