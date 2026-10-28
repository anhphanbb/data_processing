from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from pyproj import Transformer

# WGS84 base ellipsoid parameters (meters)
A_WGS84 = 6378137.0                  # semi-major axis
F_WGS84 = 1.0 / 298.257223563        # flattening
B_WGS84 = A_WGS84 * (1.0 - F_WGS84)  # semi-minor axis

# Global transformers for geodetic <-> ECEF (WGS84)
# EPSG:4979 = WGS84 (lon, lat, ellipsoidal height)
# EPSG:4978 = WGS84 ECEF
LLA_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
ECEF_TO_LLA = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic (lat, lon, alt) to ECEF (x, y, z) in meters."""
    x, y, z = LLA_TO_ECEF.transform(lon_deg, lat_deg, alt_m)
    return np.asarray(x), np.asarray(y), np.asarray(z)


def ecef_to_geodetic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF (x, y, z) to geodetic (lat, lon, alt)."""
    lon_deg, lat_deg, alt_m = ECEF_TO_LLA.transform(x, y, z)
    return np.asarray(lat_deg), np.asarray(lon_deg), np.asarray(alt_m)


def intersect_los_with_alt_shell(sat_ecef: np.ndarray, fov_ecef: np.ndarray, alt_shell_m: float, a0: float = A_WGS84, b0: float = B_WGS84) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute intersection between satellite LOS and an oblate spheroid shell at a given altitude.

    Parameters
    ----------
    sat_ecef : array-like, shape (3,)
        Satellite position in ECEF (meters).
    fov_ecef : array-like, shape (..., 3)
        Field-of-view points in ECEF (meters).
    alt_shell_m : float
        Target altitude in meters.
    a0, b0 : float
        Base ellipsoid axes (meters).
    """
    sat_ecef = np.asarray(sat_ecef, dtype=float)
    fov_ecef = np.asarray(fov_ecef, dtype=float)

    x0, y0, z0 = sat_ecef
    x1 = fov_ecef[..., 0]
    y1 = fov_ecef[..., 1]
    z1 = fov_ecef[..., 2]

    # LOS direction vector (not normalized)
    ux = x1 - x0
    uy = y1 - y0
    uz = z1 - z0

    # Shell ellipsoid axes at given altitude
    a = a0 + alt_shell_m
    b = b0 + alt_shell_m
    a2 = a * a
    b2 = b * b

    # Quadratic coefficients A, B, C for At^2 + Bt + C = 0
    A = (ux**2 + uy**2) / a2 + (uz**2) / b2
    B = 2.0 * ((x0 * ux + y0 * uy) / a2 + (z0 * uz) / b2)
    C = (x0**2 + y0**2) / a2 + (z0**2) / b2 - 1.0

    # Discriminant
    disc = B**2 - 4.0 * A * C

    # Initialize t with NaN
    t = np.full_like(disc, np.nan, dtype=float)

    # Valid where discriminant is non-negative and A != 0
    mask = (disc >= 0.0) & (A != 0.0)

    sqrt_disc = np.zeros_like(disc, dtype=float)
    sqrt_disc[mask] = np.sqrt(disc[mask])

    t1 = np.full_like(disc, np.nan, dtype=float)
    t2 = np.full_like(disc, np.nan, dtype=float)

    t1[mask] = (-B[mask] - sqrt_disc[mask]) / (2.0 * A[mask])
    t2[mask] = (-B[mask] + sqrt_disc[mask]) / (2.0 * A[mask])

    # We want the nearest intersection in the forward LOS direction (t > 0)
    t1_pos = np.where(t1 > 0.0, t1, np.nan)
    t2_pos = np.where(t2 > 0.0, t2, np.nan)

    # Take the smaller positive t; nanmin ignores NaNs
    t = np.nanmin(np.stack([t1_pos, t2_pos], axis=0), axis=0)

    # Compute intersection point: r = r0 + t * u
    x_int = x0 + t * ux
    y_int = y0 + t * uy
    z_int = z0 + t * uz

    return x_int, y_int, z_int


def parse_oh_layer_altitude_km(meta: Any) -> float:
    """
    Parse OH_layer_altitude from a mapping/xarray/nc4-like object and return altitude in km.
    Accepts strings such as '87.00 km' or numeric values.
    """
    value = None
    if isinstance(meta, Mapping) and "OH_layer_altitude" in meta:
        value = meta["OH_layer_altitude"]
    elif hasattr(meta, "attrs") and isinstance(getattr(meta, "attrs"), Mapping):
        attrs = meta.attrs
        if "OH_layer_altitude" in attrs:
            value = attrs["OH_layer_altitude"]
    elif hasattr(meta, "getncattr") and hasattr(meta, "ncattrs"):
        if "OH_layer_altitude" in meta.ncattrs():  # type: ignore[attr-defined]
            value = meta.getncattr("OH_layer_altitude")  # type: ignore[call-arg]
    elif hasattr(meta, "OH_layer_altitude"):
        value = getattr(meta, "OH_layer_altitude")

    if value is None:
        raise ValueError("Global attribute 'OH_layer_altitude' not found.")

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value)
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        raise ValueError(f"Could not parse OH_layer_altitude: {s!r}")
    return float(m.group(0))


def remap_latlon_to_altitudes(
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_km: float,
    fov_lat_deg: np.ndarray,
    fov_lon_deg: np.ndarray,
    target_alts_km: Sequence[float],
    *,
    base_alt_km: float,
    output_dtype: np.dtype = np.float32,
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """
    Remap FOV geolocations from the nominal OH layer altitude to one or more target altitudes.

    Parameters
    ----------
    sat_lat_deg, sat_lon_deg : float
        Satellite geodetic latitude/longitude in degrees.
    sat_alt_km : float
        Satellite altitude in kilometers.
    fov_lat_deg, fov_lon_deg : np.ndarray
        Field-of-view geolocations at the nominal OH altitude (degrees).
    target_alts_km : sequence of float
        Target altitudes (km) to map to along the line-of-sight.
    base_alt_km : float
        Nominal OH layer altitude (km) of the provided lat/lon grid.
    output_dtype : numpy dtype
        Dtype for the returned arrays (default: float32).
    """
    alts = [float(a) for a in target_alts_km]
    lat_out: Dict[float, np.ndarray] = {}
    lon_out: Dict[float, np.ndarray] = {}
    fov_lat = np.asarray(fov_lat_deg, dtype=float)
    fov_lon = np.asarray(fov_lon_deg, dtype=float)

    # Handle invalid satellite position gracefully.
    if not (np.isfinite(sat_lat_deg) and np.isfinite(sat_lon_deg) and np.isfinite(sat_alt_km)):
        for alt in alts:
            lat_out[alt] = np.full_like(fov_lat, np.nan, dtype=output_dtype)
            lon_out[alt] = np.full_like(fov_lon, np.nan, dtype=output_dtype)
        return lat_out, lon_out

    base_alt_m = float(base_alt_km) * 1000.0
    sat_alt_m = float(sat_alt_km) * 1000.0

    # Convert satellite position to ECEF
    sat_x, sat_y, sat_z = geodetic_to_ecef(sat_lat_deg, sat_lon_deg, sat_alt_m)
    sat_ecef = np.array([sat_x, sat_y, sat_z], dtype=float)

    # Convert FOV points at base altitude to ECEF
    base_alt_arr = np.full_like(fov_lat, base_alt_m, dtype=float)
    fov_x, fov_y, fov_z = geodetic_to_ecef(fov_lat, fov_lon, base_alt_arr)
    fov_ecef = np.stack([fov_x, fov_y, fov_z], axis=-1)

    for alt_km in alts:
        alt_m = alt_km * 1000.0
        x_int, y_int, z_int = intersect_los_with_alt_shell(
            sat_ecef,
            fov_ecef,
            alt_shell_m=alt_m,
        )
        lat_t, lon_t, _ = ecef_to_geodetic(x_int, y_int, z_int)
        lat_out[alt_km] = lat_t.astype(output_dtype, copy=False)
        lon_out[alt_km] = lon_t.astype(output_dtype, copy=False)

    return lat_out, lon_out


__all__ = [
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "intersect_los_with_alt_shell",
    "parse_oh_layer_altitude_km",
    "remap_latlon_to_altitudes",
]
