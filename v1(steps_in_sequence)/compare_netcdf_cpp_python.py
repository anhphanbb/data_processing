#!/usr/bin/env python3
"""Compare one NetCDF file produced by C++ with one produced by Python.

The script compares:

* file format and dimensions;
* global attributes;
* variable names, dimensions, dtypes, and attributes;
* Radiance first, with global bias/error statistics;
* per-frame Radiance statistics and diagnostic plots, matched by Epoch even
  when the files contain different numbers of frames;
* absolute-error percentiles and fractions outside the requested tolerance;
* separate error measurements for the circular valid-data boundary and interior;
* fractions exceeding practical absolute-error thresholds;
* spatial Radiance bias, MAE, RMSE, and tolerance-failure maps;
* the worst-RMSE Radiance frame, including its difference and failure mask;
* a configurable montage of several diverse high-RMSE frame pairs;
* all other variables, attributes, dimensions, and NaN/fill-value locations.

Large variables are read in chunks, so a full AWE image cube does not need to
fit in memory. By default, full pixel-by-pixel data comparison is limited to
Radiance to keep the run focused and faster. Other variables still have their
structure and attributes checked; use --all-variable-data to compare their
values too.

Requirements
------------
    pip install numpy netCDF4 matplotlib

Usage from a terminal
---------------------
    python compare_netcdf_cpp_python.py CPP_FILE PYTHON_FILE OUTPUT_FOLDER

You can also fill in the settings below and run the script directly in Spyder.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Settings for running directly in Spyder
# ---------------------------------------------------------------------------

CPP_FILE = r"E:\soc\l3a\2023\12\cpp\awe_l3a_tmp_2023335T0154_00141_v02.nc"
PYTHON_FILE = r"E:\soc\l3a\2023\12\awe_l3a_tmp_2023335T0154_00141_v02.nc"
OUTPUT_FOLDER = r"E:\soc\l3a\2023\12\comparison_results"

DEFAULT_ATOL = 1.0e-6
DEFAULT_RTOL = 1.0e-6
DEFAULT_CHUNK_SIZE = 16
DEFAULT_BORDER_WIDTH = 3
DEFAULT_EPOCH_TOLERANCE = None  # None = automatically use 25% of the cadence
DEFAULT_ABSOLUTE_THRESHOLDS = (0.002, 0.01, 0.1, 1.0)
SPATIAL_DIAGNOSTIC_THRESHOLD = 0.01
LOG_ABS_DIFFERENCE_FLOOR = 1.0e-6
DEFAULT_WORST_FRAME_COUNT = 6
DEFAULT_WORST_FRAME_SEPARATION = 20
RADIANCE_VARIABLE = "Temperature"
TIME_VARIABLE = "Epoch"
MAKE_RADIANCE_PLOTS = True
COMPARE_OTHER_VARIABLE_DATA = True


def _import_netcdf4():
    try:
        import netCDF4  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "netCDF4 is required. Install it with: pip install netCDF4"
        ) from exc
    return netCDF4


def _plain(value: Any) -> Any:
    """Convert NetCDF/NumPy values into stable Python values for reporting."""
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _values_equal(left: Any, right: Any) -> bool:
    """Compare attributes, treating NaN as equal to NaN."""
    left = _plain(left)
    right = _plain(right)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        if len(left) != len(right):
            return False
        return all(_values_equal(a, b) for a, b in zip(left, right))
    try:
        if math.isnan(left) and math.isnan(right):
            return True
    except (TypeError, ValueError):
        pass
    return left == right


def _format_value(value: Any, maximum: int = 180) -> str:
    text = repr(_plain(value))
    if len(text) > maximum:
        return text[: maximum - 3] + "..."
    return text


def _attribute_differences(left, right) -> list[str]:
    left_names = set(left.ncattrs())
    right_names = set(right.ncattrs())
    lines: list[str] = []
    for name in sorted(left_names - right_names):
        lines.append(f"  only C++: {name} = {_format_value(left.getncattr(name))}")
    for name in sorted(right_names - left_names):
        lines.append(f"  only Python: {name} = {_format_value(right.getncattr(name))}")
    for name in sorted(left_names & right_names):
        left_value = left.getncattr(name)
        right_value = right.getncattr(name)
        if not _values_equal(left_value, right_value):
            lines.append(
                f"  different {name}: C++={_format_value(left_value)}; "
                f"Python={_format_value(right_value)}"
            )
    return lines


def _invalid_values(variable) -> list[Any]:
    values: list[Any] = []
    for name in ("_FillValue", "missing_value"):
        if name in variable.ncattrs():
            raw = np.asarray(variable.getncattr(name)).ravel()
            values.extend(raw.tolist())
    return values


def _numeric_data_and_valid(variable, key) -> tuple[np.ndarray, np.ndarray]:
    variable.set_auto_maskandscale(False)
    raw = variable[key]
    masked = np.ma.isMaskedArray(raw)
    if masked:
        mask = np.ma.getmaskarray(raw).copy()
        data = np.asarray(np.ma.getdata(raw), dtype=np.float64)
    else:
        data = np.asarray(raw, dtype=np.float64)
        mask = np.zeros(data.shape, dtype=bool)

    mask |= ~np.isfinite(data)
    for invalid in _invalid_values(variable):
        try:
            if np.isnan(invalid):
                mask |= np.isnan(data)
            else:
                mask |= data == invalid
        except (TypeError, ValueError):
            continue
    return data, ~mask


def _chunk_keys(shape: tuple[int, ...], chunk_size: int) -> Iterable[tuple[tuple, int]]:
    """Yield (slice key, first-axis offset) without loading a whole cube."""
    if not shape:
        yield (), 0
        return
    for start in range(0, shape[0], chunk_size):
        key = (slice(start, min(start + chunk_size, shape[0])),) + (
            slice(None),
        ) * (len(shape) - 1)
        yield key, start


def _global_index(local_flat_index: int, local_shape: tuple[int, ...], offset: int):
    if not local_shape:
        return ()
    index = list(np.unravel_index(local_flat_index, local_shape))
    index[0] += offset
    return tuple(int(item) for item in index)


@dataclass
class NumericStats:
    total: int = 0
    valid_both: int = 0
    invalid_both: int = 0
    cpp_only_valid: int = 0
    python_only_valid: int = 0
    exact: int = 0
    close: int = 0
    sum_abs: float = 0.0
    sum_diff: float = 0.0
    sum_abs_cpp: float = 0.0
    sum_sq: float = 0.0
    max_abs: float = -1.0
    max_rel: float = -1.0
    worst_index: tuple[int, ...] | None = None
    worst_cpp: float = math.nan
    worst_python: float = math.nan
    sum_cpp: float = 0.0
    sum_python: float = 0.0
    sum_cpp_sq: float = 0.0
    sum_python_sq: float = 0.0
    sum_cross: float = 0.0

    def update(
        self,
        cpp: np.ndarray,
        cpp_valid: np.ndarray,
        python: np.ndarray,
        python_valid: np.ndarray,
        atol: float,
        rtol: float,
        offset: int,
    ) -> None:
        self.total += cpp.size
        both = cpp_valid & python_valid
        neither = ~cpp_valid & ~python_valid
        self.valid_both += int(np.count_nonzero(both))
        self.invalid_both += int(np.count_nonzero(neither))
        self.cpp_only_valid += int(np.count_nonzero(cpp_valid & ~python_valid))
        self.python_only_valid += int(np.count_nonzero(~cpp_valid & python_valid))
        if not np.any(both):
            return

        cpp_values = cpp[both]
        python_values = python[both]
        difference = python_values - cpp_values
        absolute = np.abs(difference)
        denominator = np.maximum(np.abs(cpp_values), np.finfo(np.float64).tiny)
        relative = absolute / denominator

        self.exact += int(np.count_nonzero(cpp_values == python_values))
        self.close += int(
            np.count_nonzero(np.isclose(cpp_values, python_values, atol=atol, rtol=rtol))
        )
        self.sum_abs += float(absolute.sum(dtype=np.float64))
        self.sum_diff += float(difference.sum(dtype=np.float64))
        self.sum_abs_cpp += float(np.abs(cpp_values).sum(dtype=np.float64))
        self.sum_sq += float(np.square(difference).sum(dtype=np.float64))
        self.sum_cpp += float(cpp_values.sum(dtype=np.float64))
        self.sum_python += float(python_values.sum(dtype=np.float64))
        self.sum_cpp_sq += float(np.square(cpp_values).sum(dtype=np.float64))
        self.sum_python_sq += float(np.square(python_values).sum(dtype=np.float64))
        self.sum_cross += float((cpp_values * python_values).sum(dtype=np.float64))

        local_valid_flat = np.flatnonzero(both)
        local_worst_in_valid = int(np.argmax(absolute))
        local_max = float(absolute[local_worst_in_valid])
        if local_max > self.max_abs:
            local_flat = int(local_valid_flat[local_worst_in_valid])
            self.max_abs = local_max
            self.worst_index = _global_index(local_flat, cpp.shape, offset)
            self.worst_cpp = float(cpp_values[local_worst_in_valid])
            self.worst_python = float(python_values[local_worst_in_valid])
        self.max_rel = max(self.max_rel, float(relative.max()))

    @property
    def mean_abs(self) -> float:
        return self.sum_abs / self.valid_both if self.valid_both else math.nan

    @property
    def mean_bias(self) -> float:
        """Mean Python minus C++ Radiance."""
        return self.sum_diff / self.valid_both if self.valid_both else math.nan

    @property
    def mean_cpp(self) -> float:
        return self.sum_cpp / self.valid_both if self.valid_both else math.nan

    @property
    def mean_python(self) -> float:
        return self.sum_python / self.valid_both if self.valid_both else math.nan

    @property
    def relative_bias_percent(self) -> float:
        denominator = self.sum_abs_cpp / self.valid_both if self.valid_both else 0.0
        return 100.0 * self.mean_bias / denominator if denominator > 0 else math.nan

    @property
    def relative_mae_percent(self) -> float:
        denominator = self.sum_abs_cpp / self.valid_both if self.valid_both else 0.0
        return 100.0 * self.mean_abs / denominator if denominator > 0 else math.nan

    @property
    def rmse(self) -> float:
        return math.sqrt(self.sum_sq / self.valid_both) if self.valid_both else math.nan

    @property
    def exact_fraction(self) -> float:
        return self.exact / self.valid_both if self.valid_both else math.nan

    @property
    def close_fraction(self) -> float:
        return self.close / self.valid_both if self.valid_both else math.nan

    @property
    def correlation(self) -> float:
        n = self.valid_both
        if n < 2:
            return math.nan
        covariance = self.sum_cross - self.sum_cpp * self.sum_python / n
        variance_cpp = self.sum_cpp_sq - self.sum_cpp**2 / n
        variance_python = self.sum_python_sq - self.sum_python**2 / n
        denominator = math.sqrt(max(variance_cpp, 0.0) * max(variance_python, 0.0))
        return covariance / denominator if denominator > 0 else math.nan


def compare_numeric_variable(
    cpp_variable,
    python_variable,
    atol: float,
    rtol: float,
    chunk_size: int,
) -> NumericStats:
    stats = NumericStats()
    for key, offset in _chunk_keys(tuple(cpp_variable.shape), chunk_size):
        cpp, cpp_valid = _numeric_data_and_valid(cpp_variable, key)
        python, python_valid = _numeric_data_and_valid(python_variable, key)
        stats.update(cpp, cpp_valid, python, python_valid, atol, rtol, offset)
    return stats


def compare_non_numeric_variable(cpp_variable, python_variable, chunk_size: int) -> tuple[int, int]:
    total = 0
    equal = 0
    cpp_variable.set_auto_maskandscale(False)
    python_variable.set_auto_maskandscale(False)
    for key, _ in _chunk_keys(tuple(cpp_variable.shape), chunk_size):
        cpp = np.asarray(cpp_variable[key])
        python = np.asarray(python_variable[key])
        matches = cpp == python
        total += matches.size
        equal += int(np.count_nonzero(matches))
    return total, equal


def _fmt_number(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.8g}"


def _stats_line(name: str, stats: NumericStats) -> str:
    return (
        f"{name}: valid={stats.valid_both:,}; "
        f"exact={stats.exact_fraction:.6%}; close={stats.close_fraction:.6%}; "
        f"bias(Python-C++)={_fmt_number(stats.mean_bias)}; "
        f"MAE={_fmt_number(stats.mean_abs)}; RMSE={_fmt_number(stats.rmse)}; "
        f"max_abs={_fmt_number(stats.max_abs)}; r={_fmt_number(stats.correlation)}"
    )


@dataclass
class RadianceOutputs:
    csv_path: Path | None = None
    timeseries_plot: Path | None = None
    spatial_plot: Path | None = None
    worst_frame_plot: Path | None = None
    worst_frames_plot: Path | None = None
    worst_frames_csv: Path | None = None
    border_plot: Path | None = None
    global_stats: NumericStats | None = None
    frame_pairing: str | None = None
    compared_frame_pairs: int = 0
    worst_cpp_frame: int | None = None
    worst_python_frame: int | None = None
    worst_frame_rmse: float = math.nan
    median_border_rmse: float = math.nan
    median_interior_rmse: float = math.nan
    median_outside_fraction: float = math.nan
    median_abs_epoch_difference: float = math.nan
    max_abs_epoch_difference: float = math.nan
    absolute_threshold_fractions: dict[float, float] = field(default_factory=dict)
    selected_worst_frames: list[WorstFramePair] = field(default_factory=list)
    warning: str | None = None


@dataclass
class FramePairs:
    cpp_indices: np.ndarray
    python_indices: np.ndarray
    method: str
    warning: str | None = None
    epoch_differences: np.ndarray | None = None


@dataclass(frozen=True)
class WorstFramePair:
    rank: int
    comparison_pair: int
    cpp_frame: int
    python_frame: int
    rmse: float
    mae: float
    max_abs: float
    boundary_rmse: float
    interior_rmse: float


def _epoch_values(dataset, expected_length: int) -> tuple[np.ndarray, np.ndarray] | None:
    if TIME_VARIABLE not in dataset.variables:
        return None
    variable = dataset.variables[TIME_VARIABLE]
    if variable.ndim != 1 or variable.shape[0] < expected_length:
        return None
    values, valid = _numeric_data_and_valid(variable, slice(0, expected_length))
    return values.ravel(), valid.ravel()


def _match_radiance_frames(
    cpp_variable,
    python_variable,
    cpp_dataset,
    python_dataset,
    epoch_tolerance: float | None,
) -> FramePairs:
    """Pair Radiance frames by Epoch, with index overlap as a safe fallback."""
    n_cpp = cpp_variable.shape[0]
    n_python = python_variable.shape[0]
    cpp_epoch_result = _epoch_values(cpp_dataset, n_cpp)
    python_epoch_result = _epoch_values(python_dataset, n_python)

    if cpp_epoch_result is not None and python_epoch_result is not None:
        cpp_epoch, cpp_valid = cpp_epoch_result
        python_epoch, python_valid = python_epoch_result
        cpp_valid_indices = np.flatnonzero(cpp_valid)
        python_valid_indices = np.flatnonzero(python_valid)
        cpp_values = cpp_epoch[cpp_valid_indices]
        python_values = python_epoch[python_valid_indices]

        if cpp_values.size and python_values.size:
            cpp_steps = np.abs(np.diff(np.sort(cpp_values)))
            python_steps = np.abs(np.diff(np.sort(python_values)))
            positive_steps = np.concatenate(
                [cpp_steps[cpp_steps > 0], python_steps[python_steps > 0]]
            )
            cadence = float(np.median(positive_steps)) if positive_steps.size else 0.0
            scale = max(
                float(np.max(np.abs(cpp_values))),
                float(np.max(np.abs(python_values))),
                1.0,
            )
            # Output files may store the same observation with slightly different
            # floating-point Epoch values.  A quarter cadence is permissive enough
            # for rounding differences but still prevents adjacent frames from
            # being paired.  The command line can override this when needed.
            inferred_tolerance = cadence * 0.25 if cadence > 0 else 0.0
            tolerance = (
                float(epoch_tolerance)
                if epoch_tolerance is not None
                else max(inferred_tolerance, np.spacing(scale) * 16.0)
            )

            cpp_order = np.argsort(cpp_values)
            python_order = np.argsort(python_values)
            sorted_cpp = cpp_values[cpp_order]
            sorted_python = python_values[python_order]
            cpp_matches: list[int] = []
            python_matches: list[int] = []
            i = 0
            j = 0
            while i < sorted_cpp.size and j < sorted_python.size:
                difference = sorted_python[j] - sorted_cpp[i]
                if abs(difference) <= tolerance:
                    cpp_matches.append(int(cpp_valid_indices[cpp_order[i]]))
                    python_matches.append(int(python_valid_indices[python_order[j]]))
                    i += 1
                    j += 1
                elif difference < 0:
                    j += 1
                else:
                    i += 1

            if cpp_matches:
                cpp_indices = np.asarray(cpp_matches, dtype=int)
                python_indices = np.asarray(python_matches, dtype=int)
                order = np.argsort(cpp_indices)
                cpp_indices = cpp_indices[order]
                python_indices = python_indices[order]
                unmatched_cpp = n_cpp - cpp_indices.size
                unmatched_python = n_python - python_indices.size
                warning = None
                if unmatched_cpp or unmatched_python:
                    warning = (
                        f"Epoch matching excluded {unmatched_cpp} C++ frame(s) and "
                        f"{unmatched_python} Python frame(s) without counterparts"
                    )
                return FramePairs(
                    cpp_indices,
                    python_indices,
                    f"Epoch nearest match ({cpp_indices.size} pairs; "
                    f"tolerance={tolerance:.8g} Epoch units)",
                    warning,
                    python_epoch[python_indices] - cpp_epoch[cpp_indices],
                )

    overlap = min(n_cpp, n_python)
    warning = (
        "No matching Epoch values were found; compared the overlapping frames by "
        "index. Confirm that frame 0 represents the same observation in both files"
    )
    return FramePairs(
        np.arange(overlap, dtype=int),
        np.arange(overlap, dtype=int),
        f"frame index fallback ({overlap} overlapping pairs)",
        warning,
    )


def _erode_valid_mask(valid: np.ndarray, iterations: int) -> np.ndarray:
    """Erode a 2-D validity mask without requiring SciPy."""
    interior = np.asarray(valid, dtype=bool).copy()
    for _ in range(max(iterations, 0)):
        padded = np.pad(interior, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(interior)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy : dy + interior.shape[0], dx : dx + interior.shape[1]]
        interior = eroded
    return interior


def _region_masks(valid: np.ndarray, border_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return valid-data boundary and interior masks for one frame.

    Unlike an array-edge mask, this follows the circular AWE FOV and any other
    invalid-data boundary.  A width of three selects the first three valid pixel
    layers adjacent to invalid data.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or border_width <= 0:
        return np.zeros_like(valid), valid.copy()
    interior = _erode_valid_mask(valid, border_width)
    boundary = valid & ~interior
    return boundary, interior


def _threshold_field(threshold: float) -> str:
    text = f"{threshold:g}".replace(".", "p").replace("-", "m")
    return f"fraction_abs_difference_gt_{text}"


def _positive_vmax(array: np.ndarray, percentile: float = 99.0) -> float:
    values = np.asarray(array)[np.isfinite(array) & (np.asarray(array) >= 0)]
    if values.size == 0:
        return 1.0
    high = float(np.percentile(values, percentile))
    if not np.isfinite(high) or high <= 0:
        high = float(np.max(values)) if values.size else 1.0
    return high if high > 0 else 1.0


def _masked_metrics(
    cpp: np.ndarray,
    python: np.ndarray,
    valid: np.ndarray,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    """Return RMSE and fraction outside tolerance for selected valid pixels."""
    if not np.any(valid):
        return math.nan, math.nan
    cpp_values = cpp[valid]
    difference = python[valid] - cpp_values
    rmse = float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
    limit = atol + rtol * np.abs(cpp_values)
    outside = float(np.mean(np.abs(difference) > limit))
    return rmse, outside


def _finite_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [1.0, 99.0])
    if low == high:
        padding = abs(low) * 0.01 or 1.0
        low -= padding
        high += padding
    return float(low), float(high)


def _save_figure(fig, requested_path: Path, dpi: int = 180) -> Path:
    """Save a plot, using a new filename if Windows cannot replace the target."""
    requested_path.parent.mkdir(parents=True, exist_ok=True)
    image_format = requested_path.suffix.lstrip(".") or "png"
    try:
        fig.savefig(
            str(requested_path),
            dpi=dpi,
            format=image_format,
        )
        return requested_path
    except OSError as first_error:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fallback_path = requested_path.with_name(
            f"{requested_path.stem}_{timestamp}{requested_path.suffix}"
        )
        try:
            fig.savefig(
                str(fallback_path),
                dpi=dpi,
                format=image_format,
            )
        except OSError as fallback_error:
            raise RuntimeError(
                f"Could not save plot to {requested_path} or {fallback_path}. "
                f"Original error: {first_error}; fallback error: {fallback_error}"
            ) from fallback_error
        print(
            f"WARNING: Could not overwrite {requested_path} ({first_error}). "
            f"Saved the plot as {fallback_path} instead."
        )
        return fallback_path


def _select_worst_frame_pairs(
    pairs: FramePairs,
    rmses: np.ndarray,
    maes: np.ndarray,
    max_abses: np.ndarray,
    boundary_rmses: np.ndarray,
    interior_rmses: np.ndarray,
    count: int,
    minimum_separation: int,
) -> list[WorstFramePair]:
    """Select high-RMSE pairs while avoiding many adjacent frames."""
    if count <= 0:
        return []
    candidates = np.flatnonzero(np.isfinite(rmses))
    candidates = candidates[np.argsort(rmses[candidates])[::-1]]
    selected_indices: list[int] = []
    selected_cpp_frames: list[int] = []
    for pair_index in candidates:
        cpp_frame = int(pairs.cpp_indices[pair_index])
        if all(
            abs(cpp_frame - previous) >= minimum_separation
            for previous in selected_cpp_frames
        ):
            selected_indices.append(int(pair_index))
            selected_cpp_frames.append(cpp_frame)
            if len(selected_indices) >= count:
                break

    return [
        WorstFramePair(
            rank=rank,
            comparison_pair=pair_index,
            cpp_frame=int(pairs.cpp_indices[pair_index]),
            python_frame=int(pairs.python_indices[pair_index]),
            rmse=float(rmses[pair_index]),
            mae=float(maes[pair_index]),
            max_abs=float(max_abses[pair_index]),
            boundary_rmse=float(boundary_rmses[pair_index]),
            interior_rmse=float(interior_rmses[pair_index]),
        )
        for rank, pair_index in enumerate(selected_indices, start=1)
    ]


def _write_worst_frames_csv(
    output_path: Path, selected: Sequence[WorstFramePair]
) -> Path:
    fields = [item.name for item in dataclass_fields(WorstFramePair)]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in selected:
            writer.writerow({name: getattr(item, name) for name in fields})
    return output_path


def _make_worst_frames_montage(
    output_folder: Path,
    selected: Sequence[WorstFramePair],
    cpp_variable,
    python_variable,
) -> Path | None:
    """Plot C++, Python, signed difference, and log absolute difference."""
    if not selected:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for the worst-frame montage") from exc

    prepared: list[tuple[WorstFramePair, np.ndarray, np.ndarray, np.ndarray]] = []
    global_max_abs = LOG_ABS_DIFFERENCE_FLOOR
    for item in selected:
        cpp_data, cpp_valid = _numeric_data_and_valid(
            cpp_variable, item.cpp_frame
        )
        python_data, python_valid = _numeric_data_and_valid(
            python_variable, item.python_frame
        )
        both = cpp_valid & python_valid
        cpp_image = np.where(both, cpp_data, np.nan)
        python_image = np.where(both, python_data, np.nan)
        difference = python_image - cpp_image
        prepared.append((item, cpp_image, python_image, difference))
        finite_abs = np.abs(difference[np.isfinite(difference)])
        if finite_abs.size:
            global_max_abs = max(global_max_abs, float(np.max(finite_abs)))

    log_vmin = math.log10(LOG_ABS_DIFFERENCE_FLOOR)
    log_vmax = math.ceil(math.log10(global_max_abs))
    if log_vmax <= log_vmin:
        log_vmax = log_vmin + 1.0

    fig, axes = plt.subplots(
        len(selected),
        4,
        figsize=(18, max(3.3 * len(selected), 4.5)),
        squeeze=False,
        constrained_layout=True,
    )
    log_im = None
    for row, (item, cpp_image, python_image, difference) in enumerate(prepared):
        image_low, image_high = _finite_limits(cpp_image, python_image)
        diff_limit = float(np.nanpercentile(np.abs(difference), 99.9))
        if not np.isfinite(diff_limit) or diff_limit <= 0:
            diff_limit = 1.0

        cpp_im = axes[row, 0].imshow(
            cpp_image,
            origin="lower",
            cmap="viridis",
            vmin=image_low,
            vmax=image_high,
        )
        axes[row, 1].imshow(
            python_image,
            origin="lower",
            cmap="viridis",
            vmin=image_low,
            vmax=image_high,
        )
        diff_im = axes[row, 2].imshow(
            difference,
            origin="lower",
            cmap="RdBu_r",
            vmin=-diff_limit,
            vmax=diff_limit,
        )
        log_abs_difference = np.where(
            np.isfinite(difference),
            np.log10(
                np.maximum(np.abs(difference), LOG_ABS_DIFFERENCE_FLOOR)
            ),
            np.nan,
        )
        log_im = axes[row, 3].imshow(
            log_abs_difference,
            origin="lower",
            cmap="magma",
            vmin=log_vmin,
            vmax=log_vmax,
        )
        if np.any(np.isfinite(difference)):
            max_flat = int(np.nanargmax(np.abs(difference)))
            max_y, max_x = np.unravel_index(max_flat, difference.shape)
            axes[row, 3].plot(
                max_x,
                max_y,
                marker="x",
                color="cyan",
                markersize=7,
                markeredgewidth=1.2,
            )
            coordinate_text = f"max at (x={max_x}, y={max_y})"
        else:
            coordinate_text = "no valid difference"
        axes[row, 0].set_title(
            f"#{item.rank} C++ frame {item.cpp_frame}", fontsize=10
        )
        axes[row, 1].set_title(f"Python frame {item.python_frame}", fontsize=10)
        axes[row, 2].set_title(
            f"Python - C++ | RMSE={item.rmse:.4g}, max={item.max_abs:.4g}",
            fontsize=10,
        )
        axes[row, 3].set_title(
            f"log10(|difference|) | {coordinate_text}", fontsize=10
        )
        for axis in axes[row]:
            axis.set_xlabel("x pixel")
            axis.set_ylabel("y pixel")
        fig.colorbar(
            cpp_im,
            ax=axes[row, :2],
            shrink=0.72,
            pad=0.01,
            label="Radiance (kR)",
        )
        fig.colorbar(
            diff_im,
            ax=axes[row, 2],
            shrink=0.72,
            pad=0.01,
            label="Difference (kR)",
        )

    if log_im is not None:
        fig.colorbar(
            log_im,
            ax=axes[:, 3],
            shrink=0.65,
            pad=0.01,
            label="log10(|difference| [kR])",
        )

    fig.suptitle(
        "Diverse high-RMSE Radiance frame pairs\n"
        f"minimum C++ frame separation = "
        f"{min((abs(a.cpp_frame - b.cpp_frame) for i, a in enumerate(selected) for b in selected[i + 1:]), default=0)}",
        fontsize=15,
    )
    output_path = _save_figure(
        fig, output_folder / "radiance_worst_frames_montage.png"
    )
    plt.close(fig)
    return output_path


def _make_radiance_plots(
    output_folder: Path,
    pair_numbers: np.ndarray,
    cpp_means: np.ndarray,
    python_means: np.ndarray,
    biases: np.ndarray,
    maes: np.ndarray,
    rmses: np.ndarray,
    correlations: np.ndarray,
    p95_abs: np.ndarray,
    p99_abs: np.ndarray,
    outside_fractions: np.ndarray,
    absolute_threshold_fractions_by_frame: dict[float, np.ndarray],
    border_rmses: np.ndarray,
    interior_rmses: np.ndarray,
    spatial_bias: np.ndarray | None,
    spatial_mae: np.ndarray | None,
    spatial_rmse: np.ndarray | None,
    spatial_diagnostic_fraction: np.ndarray | None,
    cpp_variable,
    python_variable,
    worst_cpp_frame: int | None,
    worst_python_frame: int | None,
    atol: float,
    rtol: float,
) -> tuple[Path, Path | None, Path | None, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed; CSV and text reports were still created"
        ) from exc

    timeseries_path = output_folder / "radiance_frame_diagnostics.png"
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    axes[0, 0].plot(pair_numbers, cpp_means, label="C++", lw=1.2)
    axes[0, 0].plot(pair_numbers, python_means, label="Python", lw=1.0, alpha=0.85)
    axes[0, 0].set(title="Mean Radiance", xlabel="Matched frame pair", ylabel="Radiance")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(pair_numbers, biases, color="tab:purple", lw=1.0)
    axes[0, 1].axhline(0.0, color="black", lw=0.8)
    axes[0, 1].set(
        title="Mean Radiance bias", xlabel="Matched frame pair", ylabel="Python - C++"
    )
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(pair_numbers, maes, label="MAE", lw=1.0)
    axes[1, 0].plot(pair_numbers, rmses, label="RMSE", lw=1.0)
    axes[1, 0].set(title="Radiance error", xlabel="Matched frame pair", ylabel="Radiance")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(pair_numbers, correlations, color="tab:green", lw=1.0)
    axes[1, 1].set(title="Pixel correlation", xlabel="Matched frame pair", ylabel="r")
    axes[1, 1].grid(alpha=0.25)

    axes[2, 0].plot(pair_numbers, p95_abs, label="95th percentile", lw=1.0)
    axes[2, 0].plot(pair_numbers, p99_abs, label="99th percentile", lw=1.0)
    axes[2, 0].set(
        title="Large absolute differences",
        xlabel="Matched frame pair",
        ylabel="|Python - C++|",
    )
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.25)

    axes[2, 1].plot(
        pair_numbers,
        100.0 * outside_fractions,
        color="0.55",
        lw=0.8,
        label="Strict atol/rtol",
    )
    for threshold, fractions in absolute_threshold_fractions_by_frame.items():
        axes[2, 1].plot(
            pair_numbers,
            100.0 * fractions,
            lw=1.0,
            label=f"|difference| > {threshold:g} kR",
        )
    axes[2, 1].set(
        title="Pixels exceeding error thresholds",
        xlabel="Matched frame pair",
        ylabel="Percent",
    )
    axes[2, 1].set_yscale("log")
    axes[2, 1].legend(fontsize=8, ncol=2)
    axes[2, 1].grid(alpha=0.25)
    fig.suptitle("C++ versus Python Radiance by frame", fontsize=15)
    timeseries_path = _save_figure(fig, timeseries_path)
    plt.close(fig)

    spatial_path: Path | None = None
    if (
        spatial_bias is not None
        and spatial_mae is not None
        and spatial_rmse is not None
        and spatial_diagnostic_fraction is not None
    ):
        spatial_path = output_folder / "radiance_spatial_diagnostics.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        axes = axes.ravel()
        bias_limit = float(np.nanpercentile(np.abs(spatial_bias), 99.0))
        if not np.isfinite(bias_limit) or bias_limit == 0:
            bias_limit = 1.0
        images = [
            axes[0].imshow(spatial_bias, origin="lower", cmap="RdBu_r", vmin=-bias_limit, vmax=bias_limit),
            axes[1].imshow(
                spatial_mae,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=_positive_vmax(spatial_mae),
            ),
            axes[2].imshow(
                spatial_rmse,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=_positive_vmax(spatial_rmse),
            ),
            axes[3].imshow(
                100.0 * spatial_diagnostic_fraction,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=_positive_vmax(100.0 * spatial_diagnostic_fraction),
            ),
        ]
        titles = [
            "Mean bias (Python - C++)",
            "Mean absolute error",
            "RMSE",
            f"|Difference| > {SPATIAL_DIAGNOSTIC_THRESHOLD:g} kR (%)",
        ]
        for axis, image, title in zip(axes, images, titles):
            axis.set_title(title)
            axis.set_xlabel("x pixel")
            axis.set_ylabel("y pixel")
            fig.colorbar(image, ax=axis, shrink=0.82)
        fig.suptitle("Spatial Radiance differences", fontsize=15)
        spatial_path = _save_figure(fig, spatial_path)
        plt.close(fig)

    worst_path: Path | None = None
    if (
        worst_cpp_frame is not None
        and worst_python_frame is not None
        and cpp_variable.ndim == 3
    ):
        cpp_data, cpp_valid = _numeric_data_and_valid(cpp_variable, worst_cpp_frame)
        python_data, python_valid = _numeric_data_and_valid(
            python_variable, worst_python_frame
        )
        both = cpp_valid & python_valid
        cpp_image = np.where(both, cpp_data, np.nan)
        python_image = np.where(both, python_data, np.nan)
        difference = python_image - cpp_image
        image_low, image_high = _finite_limits(cpp_image, python_image)
        diff_limit = float(np.nanpercentile(np.abs(difference), 99.0))
        if not np.isfinite(diff_limit) or diff_limit == 0:
            diff_limit = 1.0

        practical_failure = both & (
            np.abs(difference) > SPATIAL_DIAGNOSTIC_THRESHOLD
        )

        worst_path = output_folder / "radiance_worst_frame.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        axes = axes.ravel()
        cpp_im = axes[0].imshow(cpp_image, origin="lower", cmap="viridis", vmin=image_low, vmax=image_high)
        py_im = axes[1].imshow(python_image, origin="lower", cmap="viridis", vmin=image_low, vmax=image_high)
        diff_im = axes[2].imshow(difference, origin="lower", cmap="RdBu_r", vmin=-diff_limit, vmax=diff_limit)
        mask_im = axes[3].imshow(
            practical_failure, origin="lower", cmap="gray_r", vmin=0, vmax=1
        )
        for axis, title in zip(
            axes,
            [
                "C++ Radiance",
                "Python Radiance",
                "Python - C++",
                f"|Difference| > {SPATIAL_DIAGNOSTIC_THRESHOLD:g} kR",
            ],
        ):
            axis.set_title(title)
            axis.set_xlabel("x pixel")
            axis.set_ylabel("y pixel")
        fig.colorbar(cpp_im, ax=axes[:2], shrink=0.82, label="Radiance")
        fig.colorbar(diff_im, ax=axes[2], shrink=0.82, label="Radiance difference")
        fig.colorbar(mask_im, ax=axes[3], shrink=0.82, ticks=[0, 1])
        fig.suptitle(
            "Radiance pair with largest RMSE: "
            f"C++ frame {worst_cpp_frame}, Python frame {worst_python_frame}",
            fontsize=15,
        )
        worst_path = _save_figure(fig, worst_path)
        plt.close(fig)

    region_path = output_folder / "radiance_border_vs_interior.png"
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(pair_numbers, border_rmses, label="Valid-boundary RMSE", lw=1.0)
    ax.plot(pair_numbers, interior_rmses, label="Interior RMSE", lw=1.0)
    ax.set(
        title="Radiance error at valid-data boundary versus interior",
        xlabel="Matched frame pair",
        ylabel="RMSE",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    region_path = _save_figure(fig, region_path)
    plt.close(fig)

    return timeseries_path, spatial_path, worst_path, region_path


def compare_radiance_by_frame(
    cpp_variable,
    python_variable,
    cpp_dataset,
    python_dataset,
    output_csv: Path,
    atol: float,
    rtol: float,
    make_plots: bool,
    border_width: int,
    epoch_tolerance: float | None,
    worst_frame_count: int,
    worst_frame_separation: int,
) -> RadianceOutputs:
    outputs = RadianceOutputs()
    if cpp_variable.ndim != python_variable.ndim or cpp_variable.ndim < 3:
        outputs.warning = (
            "Radiance comparison skipped: the variables have different numbers of "
            "dimensions or fewer than 3 dimensions"
        )
        return outputs
    if cpp_variable.shape[1:] != python_variable.shape[1:]:
        outputs.warning = (
            "Radiance comparison skipped: spatial/channel dimensions differ "
            f"(C++ {cpp_variable.shape[1:]} versus Python {python_variable.shape[1:]})"
        )
        return outputs
    time_axis = 0
    if "time" in cpp_variable.dimensions:
        time_axis = cpp_variable.dimensions.index("time")
    python_time_axis = 0
    if "time" in python_variable.dimensions:
        python_time_axis = python_variable.dimensions.index("time")
    if time_axis != 0 or python_time_axis != 0:
        outputs.warning = "Radiance frame diagnostics skipped because time is not the first axis"
        return outputs

    pairs = _match_radiance_frames(
        cpp_variable,
        python_variable,
        cpp_dataset,
        python_dataset,
        epoch_tolerance,
    )
    outputs.frame_pairing = pairs.method
    outputs.compared_frame_pairs = int(pairs.cpp_indices.size)
    outputs.warning = pairs.warning
    if pairs.epoch_differences is not None and pairs.epoch_differences.size:
        absolute_epoch_difference = np.abs(pairs.epoch_differences)
        outputs.median_abs_epoch_difference = float(
            np.median(absolute_epoch_difference)
        )
        outputs.max_abs_epoch_difference = float(np.max(absolute_epoch_difference))
    if pairs.cpp_indices.size == 0:
        outputs.warning = "No overlapping Radiance frames are available"
        return outputs

    cpp_epoch_result = _epoch_values(cpp_dataset, cpp_variable.shape[0])
    python_epoch_result = _epoch_values(python_dataset, python_variable.shape[0])
    cpp_epoch = cpp_epoch_result[0] if cpp_epoch_result is not None else None
    python_epoch = python_epoch_result[0] if python_epoch_result is not None else None

    fields = [
        "comparison_pair",
        "cpp_frame",
        "python_frame",
        "cpp_epoch",
        "python_epoch",
        "epoch_difference",
        "valid_both",
        "invalid_both",
        "cpp_only_valid",
        "python_only_valid",
        "cpp_mean",
        "python_mean",
        "python_minus_cpp_mean",
        "mae",
        "rmse",
        "max_abs",
        "exact_fraction",
        "close_fraction",
        "outside_tolerance_fraction",
        "abs_difference_p50",
        "abs_difference_p95",
        "abs_difference_p99",
        "correlation",
        "border_rmse",
        "interior_rmse",
        "border_outside_tolerance_fraction",
        "interior_outside_tolerance_fraction",
    ]
    threshold_fields = {
        threshold: _threshold_field(threshold)
        for threshold in DEFAULT_ABSOLUTE_THRESHOLDS
    }
    fields.extend(threshold_fields.values())

    n_pairs = pairs.cpp_indices.size
    pair_numbers = np.arange(n_pairs, dtype=int)
    cpp_means = np.full(n_pairs, np.nan)
    python_means = np.full(n_pairs, np.nan)
    biases = np.full(n_pairs, np.nan)
    maes = np.full(n_pairs, np.nan)
    rmses = np.full(n_pairs, np.nan)
    max_abses = np.full(n_pairs, np.nan)
    correlations = np.full(n_pairs, np.nan)
    p95_abs = np.full(n_pairs, np.nan)
    p99_abs = np.full(n_pairs, np.nan)
    outside_fractions = np.full(n_pairs, np.nan)
    border_rmses = np.full(n_pairs, np.nan)
    interior_rmses = np.full(n_pairs, np.nan)
    border_outside_fractions = np.full(n_pairs, np.nan)
    interior_outside_fractions = np.full(n_pairs, np.nan)
    absolute_threshold_fractions_by_frame = {
        threshold: np.full(n_pairs, np.nan)
        for threshold in DEFAULT_ABSOLUTE_THRESHOLDS
    }
    absolute_threshold_counts = {
        threshold: 0 for threshold in DEFAULT_ABSOLUTE_THRESHOLDS
    }
    global_stats = NumericStats()
    cached_valid_mask: np.ndarray | None = None
    cached_border_mask: np.ndarray | None = None
    cached_interior_mask: np.ndarray | None = None

    spatial_sum: np.ndarray | None = None
    spatial_sum_abs: np.ndarray | None = None
    spatial_sum_sq: np.ndarray | None = None
    spatial_count: np.ndarray | None = None
    spatial_diagnostic_count: np.ndarray | None = None
    if cpp_variable.ndim == 3:
        spatial_shape = tuple(cpp_variable.shape[1:])
        spatial_sum = np.zeros(spatial_shape, dtype=np.float64)
        spatial_sum_abs = np.zeros(spatial_shape, dtype=np.float64)
        spatial_sum_sq = np.zeros(spatial_shape, dtype=np.float64)
        spatial_count = np.zeros(spatial_shape, dtype=np.int64)
        spatial_diagnostic_count = np.zeros(spatial_shape, dtype=np.int64)

    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair_number, (cpp_frame, python_frame) in enumerate(
            zip(pairs.cpp_indices, pairs.python_indices)
        ):
            cpp_frame = int(cpp_frame)
            python_frame = int(python_frame)
            cpp, cpp_valid = _numeric_data_and_valid(cpp_variable, cpp_frame)
            python, python_valid = _numeric_data_and_valid(
                python_variable, python_frame
            )
            stats = NumericStats()
            stats.update(cpp, cpp_valid, python, python_valid, atol, rtol, 0)
            global_stats.update(
                cpp[np.newaxis, ...],
                cpp_valid[np.newaxis, ...],
                python[np.newaxis, ...],
                python_valid[np.newaxis, ...],
                atol,
                rtol,
                cpp_frame,
            )
            both = cpp_valid & python_valid
            cpp_mean = float(cpp[both].mean()) if np.any(both) else math.nan
            python_mean = float(python[both].mean()) if np.any(both) else math.nan
            cpp_means[pair_number] = cpp_mean
            python_means[pair_number] = python_mean
            biases[pair_number] = stats.mean_bias
            maes[pair_number] = stats.mean_abs
            rmses[pair_number] = stats.rmse
            max_abses[pair_number] = stats.max_abs
            correlations[pair_number] = stats.correlation

            if np.any(both):
                abs_difference = np.abs(python[both] - cpp[both])
                p50, p95, p99 = np.percentile(abs_difference, [50.0, 95.0, 99.0])
                p95_abs[pair_number] = float(p95)
                p99_abs[pair_number] = float(p99)
                outside_fractions[pair_number] = 1.0 - stats.close_fraction
                for threshold in DEFAULT_ABSOLUTE_THRESHOLDS:
                    count = int(np.count_nonzero(abs_difference > threshold))
                    absolute_threshold_counts[threshold] += count
                    absolute_threshold_fractions_by_frame[threshold][pair_number] = (
                        count / abs_difference.size
                    )
            else:
                p50 = math.nan

            if cached_valid_mask is None or not np.array_equal(
                both, cached_valid_mask
            ):
                border_mask, interior_mask = _region_masks(both, border_width)
                cached_valid_mask = both.copy()
                cached_border_mask = border_mask
                cached_interior_mask = interior_mask
            else:
                assert cached_border_mask is not None
                assert cached_interior_mask is not None
                border_mask = cached_border_mask
                interior_mask = cached_interior_mask
            border_rmses[pair_number], border_outside_fractions[pair_number] = (
                _masked_metrics(cpp, python, border_mask, atol, rtol)
            )
            interior_rmses[pair_number], interior_outside_fractions[pair_number] = (
                _masked_metrics(cpp, python, interior_mask, atol, rtol)
            )

            if spatial_sum is not None and np.any(both):
                difference = np.where(both, python - cpp, 0.0)
                spatial_sum += difference
                spatial_sum_abs += np.abs(difference)
                spatial_sum_sq += np.square(difference)
                spatial_count += both
                spatial_diagnostic_count += both & (
                    np.abs(difference) > SPATIAL_DIAGNOSTIC_THRESHOLD
                )
            cpp_time = (
                float(cpp_epoch[cpp_frame]) if cpp_epoch is not None else math.nan
            )
            python_time = (
                float(python_epoch[python_frame])
                if python_epoch is not None
                else math.nan
            )
            row = {
                    "comparison_pair": pair_number,
                    "cpp_frame": cpp_frame,
                    "python_frame": python_frame,
                    "cpp_epoch": cpp_time,
                    "python_epoch": python_time,
                    "epoch_difference": python_time - cpp_time,
                    "valid_both": stats.valid_both,
                    "invalid_both": stats.invalid_both,
                    "cpp_only_valid": stats.cpp_only_valid,
                    "python_only_valid": stats.python_only_valid,
                    "cpp_mean": cpp_mean,
                    "python_mean": python_mean,
                    "python_minus_cpp_mean": python_mean - cpp_mean,
                    "mae": stats.mean_abs,
                    "rmse": stats.rmse,
                    "max_abs": stats.max_abs,
                    "exact_fraction": stats.exact_fraction,
                    "close_fraction": stats.close_fraction,
                    "outside_tolerance_fraction": 1.0 - stats.close_fraction,
                    "abs_difference_p50": p50,
                    "abs_difference_p95": p95_abs[pair_number],
                    "abs_difference_p99": p99_abs[pair_number],
                    "correlation": stats.correlation,
                    "border_rmse": border_rmses[pair_number],
                    "interior_rmse": interior_rmses[pair_number],
                    "border_outside_tolerance_fraction": border_outside_fractions[pair_number],
                    "interior_outside_tolerance_fraction": interior_outside_fractions[pair_number],
                }
            for threshold, field_name in threshold_fields.items():
                row[field_name] = absolute_threshold_fractions_by_frame[threshold][
                    pair_number
                ]
            writer.writerow(row)

    outputs.csv_path = output_csv
    outputs.global_stats = global_stats
    outputs.median_border_rmse = float(np.nanmedian(border_rmses))
    outputs.median_interior_rmse = float(np.nanmedian(interior_rmses))
    outputs.median_outside_fraction = float(np.nanmedian(outside_fractions))
    if global_stats.valid_both:
        outputs.absolute_threshold_fractions = {
            threshold: count / global_stats.valid_both
            for threshold, count in absolute_threshold_counts.items()
        }
    finite_rmse = np.isfinite(rmses)
    if np.any(finite_rmse):
        worst_pair = int(np.nanargmax(rmses))
        outputs.worst_cpp_frame = int(pairs.cpp_indices[worst_pair])
        outputs.worst_python_frame = int(pairs.python_indices[worst_pair])
        outputs.worst_frame_rmse = float(rmses[worst_pair])

    outputs.selected_worst_frames = _select_worst_frame_pairs(
        pairs,
        rmses,
        maes,
        max_abses,
        border_rmses,
        interior_rmses,
        worst_frame_count,
        worst_frame_separation,
    )
    if outputs.selected_worst_frames:
        outputs.worst_frames_csv = _write_worst_frames_csv(
            output_csv.parent / "radiance_worst_frames.csv",
            outputs.selected_worst_frames,
        )

    spatial_bias: np.ndarray | None = None
    spatial_mae: np.ndarray | None = None
    spatial_rmse: np.ndarray | None = None
    spatial_diagnostic_fraction: np.ndarray | None = None
    if spatial_count is not None:
        assert spatial_diagnostic_count is not None
        with np.errstate(invalid="ignore", divide="ignore"):
            spatial_bias = np.divide(
                spatial_sum,
                spatial_count,
                out=np.full_like(spatial_sum, np.nan),
                where=spatial_count > 0,
            )
            spatial_mae = np.divide(
                spatial_sum_abs,
                spatial_count,
                out=np.full_like(spatial_sum_abs, np.nan),
                where=spatial_count > 0,
            )
            spatial_rmse = np.sqrt(
                np.divide(
                    spatial_sum_sq,
                    spatial_count,
                    out=np.full_like(spatial_sum_sq, np.nan),
                    where=spatial_count > 0,
                )
            )
            spatial_diagnostic_fraction = np.divide(
                spatial_diagnostic_count,
                spatial_count,
                out=np.full(spatial_count.shape, np.nan, dtype=np.float64),
                where=spatial_count > 0,
            )

    if make_plots:
        try:
            (
                outputs.timeseries_plot,
                outputs.spatial_plot,
                outputs.worst_frame_plot,
                outputs.border_plot,
            ) = _make_radiance_plots(
                output_csv.parent,
                pair_numbers,
                cpp_means,
                python_means,
                biases,
                maes,
                rmses,
                correlations,
                p95_abs,
                p99_abs,
                outside_fractions,
                absolute_threshold_fractions_by_frame,
                border_rmses,
                interior_rmses,
                spatial_bias,
                spatial_mae,
                spatial_rmse,
                spatial_diagnostic_fraction,
                cpp_variable,
                python_variable,
                outputs.worst_cpp_frame,
                outputs.worst_python_frame,
                atol,
                rtol,
            )
            outputs.worst_frames_plot = _make_worst_frames_montage(
                output_csv.parent,
                outputs.selected_worst_frames,
                cpp_variable,
                python_variable,
            )
        except RuntimeError as exc:
            outputs.warning = (
                f"{outputs.warning}; {exc}" if outputs.warning else str(exc)
            )
    return outputs


def compare_files(
    cpp_path: Path,
    python_path: Path,
    output_folder: Path,
    atol: float,
    rtol: float,
    chunk_size: int,
    make_radiance_plots: bool,
    compare_other_variable_data: bool,
    border_width: int,
    epoch_tolerance: float | None,
    worst_frame_count: int,
    worst_frame_separation: int,
) -> tuple[Path, RadianceOutputs]:
    netCDF4 = _import_netcdf4()
    output_folder.mkdir(parents=True, exist_ok=True)
    report_path = output_folder / "netcdf_comparison_report.txt"
    frame_csv_path = output_folder / "radiance_frame_comparison.csv"
    lines: list[str] = []
    radiance_outputs = RadianceOutputs()
    radiance_stats: NumericStats | None = None

    with netCDF4.Dataset(cpp_path, "r") as cpp, netCDF4.Dataset(
        python_path, "r"
    ) as python:
        lines.extend(
            [
                "NetCDF C++ versus Python comparison",
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
                f"C++ file: {cpp_path}",
                f"Python file: {python_path}",
                f"C++ size: {cpp_path.stat().st_size:,} bytes",
                f"Python size: {python_path.stat().st_size:,} bytes",
                f"C++ data model: {cpp.data_model}",
                f"Python data model: {python.data_model}",
                f"Tolerance: atol={atol:g}, rtol={rtol:g}",
                f"Radiance valid-boundary width: {border_width} pixels",
                f"Worst-frame montage count: {worst_frame_count}",
                f"Worst-frame minimum separation: {worst_frame_separation} frames",
                "Epoch matching tolerance: "
                + (
                    f"{epoch_tolerance:g} Epoch units"
                    if epoch_tolerance is not None
                    else "automatic (25% of inferred cadence)"
                ),
                "Other variable values: "
                + ("compared" if compare_other_variable_data else "metadata only"),
                "",
                "RADIANCE SUMMARY (PRIMARY COMPARISON)",
            ]
        )

        if RADIANCE_VARIABLE not in cpp.variables:
            lines.append(f"  {RADIANCE_VARIABLE!r} is missing from the C++ file.")
        elif RADIANCE_VARIABLE not in python.variables:
            lines.append(f"  {RADIANCE_VARIABLE!r} is missing from the Python file.")
        else:
            cpp_radiance = cpp.variables[RADIANCE_VARIABLE]
            python_radiance = python.variables[RADIANCE_VARIABLE]
            lines.append(
                f"  C++: dtype={cpp_radiance.dtype}, dimensions={cpp_radiance.dimensions}, "
                f"shape={cpp_radiance.shape}"
            )
            lines.append(
                f"  Python: dtype={python_radiance.dtype}, dimensions={python_radiance.dimensions}, "
                f"shape={python_radiance.shape}"
            )
            if not (
                np.issubdtype(cpp_radiance.dtype, np.number)
                and np.issubdtype(python_radiance.dtype, np.number)
            ):
                lines.append("  DATA COMPARISON SKIPPED: Radiance is not numeric in both files")
                radiance_outputs.warning = "Radiance is not numeric in both files"
            else:
                radiance_outputs = compare_radiance_by_frame(
                    cpp_radiance,
                    python_radiance,
                    cpp,
                    python,
                    frame_csv_path,
                    atol,
                    rtol,
                    make_radiance_plots,
                    border_width,
                    epoch_tolerance,
                    worst_frame_count,
                    worst_frame_separation,
                )
                radiance_stats = radiance_outputs.global_stats
                if radiance_stats is not None:
                    lines.extend(
                        [
                            f"  Frame pairing: {radiance_outputs.frame_pairing}",
                            f"  Compared frame pairs: {radiance_outputs.compared_frame_pairs:,}",
                            "  " + _stats_line(RADIANCE_VARIABLE, radiance_stats),
                            f"  C++ paired-valid mean={_fmt_number(radiance_stats.mean_cpp)}",
                            f"  Python paired-valid mean={_fmt_number(radiance_stats.mean_python)}",
                            f"  Mean bias (Python - C++)={_fmt_number(radiance_stats.mean_bias)} "
                            f"({_fmt_number(radiance_stats.relative_bias_percent)}% of mean |C++|)",
                            f"  Relative MAE={_fmt_number(radiance_stats.relative_mae_percent)}% "
                            "of mean |C++|",
                            f"  Validity mismatch: C++ only={radiance_stats.cpp_only_valid:,}, "
                            f"Python only={radiance_stats.python_only_valid:,}; "
                            f"both invalid={radiance_stats.invalid_both:,}",
                            f"  Median frame fraction outside tolerance="
                            f"{radiance_outputs.median_outside_fraction:.6%}",
                            f"  Median valid-boundary RMSE ({border_width} pixel layers)="
                            f"{_fmt_number(radiance_outputs.median_border_rmse)}",
                            f"  Median interior RMSE="
                            f"{_fmt_number(radiance_outputs.median_interior_rmse)}",
                        ]
                    )
                    if np.isfinite(radiance_outputs.max_abs_epoch_difference):
                        lines.append(
                            "  Epoch match |difference|: "
                            f"median={_fmt_number(radiance_outputs.median_abs_epoch_difference)}, "
                            f"maximum={_fmt_number(radiance_outputs.max_abs_epoch_difference)} "
                            "Epoch units"
                        )
                    if radiance_outputs.absolute_threshold_fractions:
                        lines.append("  Absolute-difference exceedance fractions:")
                        for threshold, fraction in (
                            radiance_outputs.absolute_threshold_fractions.items()
                        ):
                            lines.append(
                                f"    > {threshold:g} kR: {fraction:.6%}"
                            )
                    if radiance_stats.worst_index is not None:
                        lines.append(
                            f"  Largest absolute difference at C++ index "
                            f"{radiance_stats.worst_index}: "
                            f"C++={radiance_stats.worst_cpp:.12g}, "
                            f"Python={radiance_stats.worst_python:.12g}"
                        )
                    print("\nRADIANCE (primary comparison)")
                    print(f"Frame pairing: {radiance_outputs.frame_pairing}")
                    print(_stats_line(RADIANCE_VARIABLE, radiance_stats))
                else:
                    lines.append("  DATA COMPARISON SKIPPED")
                if radiance_outputs.worst_cpp_frame is not None:
                    lines.append(
                        "  Frame pair with largest RMSE: "
                        f"C++ frame={radiance_outputs.worst_cpp_frame}, "
                        f"Python frame={radiance_outputs.worst_python_frame} "
                        f"(RMSE={_fmt_number(radiance_outputs.worst_frame_rmse)})"
                    )
                if radiance_outputs.selected_worst_frames:
                    lines.append("  Diverse high-RMSE frame pairs selected for montage:")
                    for item in radiance_outputs.selected_worst_frames:
                        lines.append(
                            f"    #{item.rank}: C++={item.cpp_frame}, "
                            f"Python={item.python_frame}, RMSE={item.rmse:.8g}, "
                            f"boundary RMSE={item.boundary_rmse:.8g}, "
                            f"interior RMSE={item.interior_rmse:.8g}"
                        )
                if radiance_outputs.warning:
                    lines.append(f"  WARNING: {radiance_outputs.warning}")

        lines.extend(
            [
                "",
                "DIMENSIONS",
            ]
        )

        dimension_names = sorted(set(cpp.dimensions) | set(python.dimensions))
        for name in dimension_names:
            if name not in cpp.dimensions:
                lines.append(f"  {name}: only Python, length={len(python.dimensions[name])}")
            elif name not in python.dimensions:
                lines.append(f"  {name}: only C++, length={len(cpp.dimensions[name])}")
            else:
                cpp_dim = cpp.dimensions[name]
                python_dim = python.dimensions[name]
                status = "MATCH" if len(cpp_dim) == len(python_dim) else "DIFFERENT"
                lines.append(
                    f"  {name}: C++={len(cpp_dim)}, Python={len(python_dim)}, "
                    f"unlimited=({cpp_dim.isunlimited()}, {python_dim.isunlimited()}) [{status}]"
                )

        lines.extend(["", "GLOBAL ATTRIBUTES"])
        global_differences = _attribute_differences(cpp, python)
        lines.extend(global_differences or ["  All global attributes match."])

        cpp_names = set(cpp.variables)
        python_names = set(python.variables)
        lines.extend(["", "VARIABLE INVENTORY"])
        lines.append(f"  Only C++: {sorted(cpp_names - python_names) or 'none'}")
        lines.append(f"  Only Python: {sorted(python_names - cpp_names) or 'none'}")

        lines.extend(["", "COMMON VARIABLE COMPARISONS"])
        common_names = sorted(
            cpp_names & python_names,
            key=lambda name: (name != RADIANCE_VARIABLE, name.lower()),
        )
        for name in common_names:
            cpp_var = cpp.variables[name]
            python_var = python.variables[name]
            lines.append("")
            lines.append(f"[{name}]")
            lines.append(
                f"  C++: dtype={cpp_var.dtype}, dimensions={cpp_var.dimensions}, "
                f"shape={cpp_var.shape}"
            )
            lines.append(
                f"  Python: dtype={python_var.dtype}, dimensions={python_var.dimensions}, "
                f"shape={python_var.shape}"
            )
            variable_attr_differences = _attribute_differences(cpp_var, python_var)
            if variable_attr_differences:
                lines.append("  Attribute differences:")
                lines.extend("  " + item for item in variable_attr_differences)

            if name == RADIANCE_VARIABLE and radiance_stats is not None:
                lines.append(
                    "  See RADIANCE SUMMARY above; values were compared using matched frame pairs."
                )
                continue

            if cpp_var.dimensions != python_var.dimensions or cpp_var.shape != python_var.shape:
                lines.append("  DATA COMPARISON SKIPPED: dimensions or shapes differ")
                continue

            if name != RADIANCE_VARIABLE and not compare_other_variable_data:
                lines.append(
                    "  DATA VALUES NOT COMPARED (use --all-variable-data to enable)"
                )
                continue

            cpp_numeric = np.issubdtype(cpp_var.dtype, np.number)
            python_numeric = np.issubdtype(python_var.dtype, np.number)
            if cpp_numeric and python_numeric:
                stats = (
                    radiance_stats
                    if name == RADIANCE_VARIABLE and radiance_stats is not None
                    else compare_numeric_variable(
                        cpp_var, python_var, atol, rtol, chunk_size
                    )
                )
                lines.append("  " + _stats_line(name, stats))
                lines.append(
                    f"  validity mismatch: C++ only={stats.cpp_only_valid:,}, "
                    f"Python only={stats.python_only_valid:,}; "
                    f"both invalid={stats.invalid_both:,}"
                )
                if stats.worst_index is not None:
                    lines.append(
                        f"  worst index={stats.worst_index}: C++={stats.worst_cpp:.12g}, "
                        f"Python={stats.worst_python:.12g}"
                    )
                if name != RADIANCE_VARIABLE:
                    print(_stats_line(name, stats))
            else:
                total, equal = compare_non_numeric_variable(
                    cpp_var, python_var, chunk_size
                )
                fraction = equal / total if total else math.nan
                lines.append(
                    f"  exact matches={equal:,}/{total:,} ({fraction:.6%})"
                )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, radiance_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare NetCDF files generated by the AWE C++ and Python processing."
    )
    parser.add_argument("cpp_file", type=Path, nargs="?", help="C++ NetCDF file")
    parser.add_argument("python_file", type=Path, nargs="?", help="Python NetCDF file")
    parser.add_argument(
        "output_folder", type=Path, nargs="?", help="Folder for comparison reports"
    )
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of first-dimension records read at once (default: 16)",
    )
    parser.add_argument(
        "--no-radiance-plots",
        action="store_true",
        help="Create text and CSV outputs but skip Radiance PNG plots",
    )
    parser.add_argument(
        "--all-variable-data",
        action="store_true",
        help="Also compare the values of every common variable (slower)",
    )
    parser.add_argument(
        "--border-width",
        type=int,
        default=DEFAULT_BORDER_WIDTH,
        help=(
            "Width of the image border analyzed separately from the interior "
            "(default: 3 pixels)"
        ),
    )
    parser.add_argument(
        "--epoch-tolerance",
        type=float,
        default=DEFAULT_EPOCH_TOLERANCE,
        help=(
            "Maximum Epoch difference for frame matching, in Epoch variable units; "
            "default is 25%% of the inferred cadence"
        ),
    )
    parser.add_argument(
        "--worst-frame-count",
        type=int,
        default=DEFAULT_WORST_FRAME_COUNT,
        help="Number of diverse high-RMSE frame pairs in the montage (default: 6)",
    )
    parser.add_argument(
        "--worst-frame-separation",
        type=int,
        default=DEFAULT_WORST_FRAME_SEPARATION,
        help=(
            "Minimum separation between selected C++ frames, preventing adjacent "
            "frames from dominating the montage (default: 20)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cpp_path = args.cpp_file or (Path(CPP_FILE) if CPP_FILE else None)
    python_path = args.python_file or (Path(PYTHON_FILE) if PYTHON_FILE else None)
    output_folder = args.output_folder or (
        Path(OUTPUT_FOLDER) if OUTPUT_FOLDER else None
    )
    if cpp_path is None or python_path is None:
        raise SystemExit(
            "Specify CPP_FILE and PYTHON_FILE near the top of the script, or "
            "provide both paths on the command line."
        )

    cpp_path = cpp_path.expanduser().resolve()
    python_path = python_path.expanduser().resolve()
    if not cpp_path.is_file():
        raise FileNotFoundError(f"C++ file not found: {cpp_path}")
    if not python_path.is_file():
        raise FileNotFoundError(f"Python file not found: {python_path}")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.border_width < 0:
        raise ValueError("--border-width cannot be negative")
    if args.epoch_tolerance is not None and args.epoch_tolerance < 0:
        raise ValueError("--epoch-tolerance cannot be negative")
    if args.worst_frame_count < 0:
        raise ValueError("--worst-frame-count cannot be negative")
    if args.worst_frame_separation < 0:
        raise ValueError("--worst-frame-separation cannot be negative")
    if output_folder is None:
        output_folder = python_path.parent / "comparison_cpp_vs_python"
    else:
        output_folder = output_folder.expanduser().resolve()

    report, radiance_outputs = compare_files(
        cpp_path,
        python_path,
        output_folder,
        args.atol,
        args.rtol,
        args.chunk_size,
        MAKE_RADIANCE_PLOTS and not args.no_radiance_plots,
        COMPARE_OTHER_VARIABLE_DATA or args.all_variable_data,
        args.border_width,
        args.epoch_tolerance,
        args.worst_frame_count,
        args.worst_frame_separation,
    )
    print(f"\nComparison report: {report}")
    if radiance_outputs.csv_path is not None:
        print(f"Per-frame Radiance report: {radiance_outputs.csv_path}")
    if radiance_outputs.timeseries_plot is not None:
        print(f"Radiance frame plot: {radiance_outputs.timeseries_plot}")
    if radiance_outputs.spatial_plot is not None:
        print(f"Radiance spatial plot: {radiance_outputs.spatial_plot}")
    if radiance_outputs.worst_frame_plot is not None:
        print(f"Radiance worst-frame plot: {radiance_outputs.worst_frame_plot}")
    if radiance_outputs.worst_frames_plot is not None:
        print(f"Radiance worst-frames montage: {radiance_outputs.worst_frames_plot}")
    if radiance_outputs.worst_frames_csv is not None:
        print(f"Radiance worst-frames table: {radiance_outputs.worst_frames_csv}")
    if radiance_outputs.border_plot is not None:
        print(f"Radiance border/interior plot: {radiance_outputs.border_plot}")
    if radiance_outputs.warning:
        print(f"Radiance warning: {radiance_outputs.warning}")
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
