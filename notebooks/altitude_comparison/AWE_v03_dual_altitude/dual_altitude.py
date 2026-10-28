"""v03: fixed-grid altitude comparison and experimental joint static-layer fit."""
from pathlib import Path
import json
import warnings
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
from scipy.signal import fftconvolve
from scipy.sparse.linalg import LinearOperator, lsmr
from netCDF4 import Dataset
from remap_altitude import remap_latlon_to_altitudes, parse_oh_layer_altitude_km


def asfloat(a):
    return np.asarray(np.ma.filled(np.ma.asarray(a, dtype=float), np.nan))


def prepare(folder):
    """Use identical Epochs, grid centers, interpolation and mask rules at both heights."""
    folder = Path(folder)
    grid = np.load(folder / 'projected_frames.npz')
    lat, lon = grid['latitude'], grid['longitude']
    xx, yy = np.meshgrid(lon, lat)
    with Dataset(folder / 'original_v02_matched_frames.nc') as ds:
        base = parse_oh_layer_altitude_km(ds)
        if not np.isclose(base, 85):
            raise ValueError(f'Expected 85 km source, got {base}')
        epoch = asfloat(ds['Epoch'][:])
        if not np.array_equal(epoch, grid['epoch']):
            raise ValueError('Epochs do not match bundled reference')
        raw = asfloat(ds['Radiance'][:])
        la85, lo85 = asfloat(ds['Latitude'][:]), asfloat(ds['Longitude'][:])
        la15, lo15 = np.empty_like(la85), np.empty_like(lo85)
        for i in range(len(epoch)):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='All-NaN slice encountered')
                a, b = remap_latlon_to_altitudes(
                    float(ds['ISS_Latitude'][i]), float(ds['ISS_Longitude'][i]),
                    float(ds['ISS_Altitude'][i]), la85[i], lo85[i], [15], base_alt_km=base)
            la15[i], lo15[i] = a[15.], b[15.]
        frames = np.array(ds['source_v02_frame_index'][:])
    cubes = {}
    for altitude, (la, lo) in {85: (la85, lo85), 15: (la15, lo15)}.items():
        projected = []
        for i, radiance in enumerate(raw):
            valid_geo = np.isfinite(la[i]) & np.isfinite(lo[i])
            triangulation = Delaunay(np.column_stack((lo[i][valid_geo], la[i][valid_geo])))
            # Keep NaN radiance vertices: dropping them would bridge masked holes.
            interpolator = LinearNDInterpolator(triangulation, radiance[valid_geo], fill_value=np.nan)
            projected.append(interpolator(xx, yy).astype(np.float32))
        cubes[altitude] = np.stack(projected)
        print(f'Projected {len(epoch)} frames at {altitude} km', flush=True)
    ref = int(np.flatnonzero(grid['matching_v01_frame_index'] == 955)[0])
    common = np.all(np.isfinite(cubes[85]) & np.isfinite(cubes[15]), axis=0)
    np.savez_compressed(folder / 'dual_projected_frames.npz',
                        radiance85=cubes[85], radiance15=cubes[15], latitude=lat,
                        longitude=lon, epoch=epoch, frame_index=frames,
                        reference_index=ref, common_coverage=common)
    # Geometry for the joint model uses the original observations, once each.
    np.savez_compressed(folder / 'joint_geometry.npz', radiance=raw,
                        latitude85=la85, longitude85=lo85, latitude15=la15, longitude15=lo15)
    return cubes


def highpass(a, sigma=5):
    ok = np.isfinite(a)
    weight = gaussian_filter(ok.astype(float), sigma)
    bg = gaussian_filter(np.nan_to_num(a), sigma) / np.maximum(weight, 1e-9)
    return np.where(ok & (weight > .98), a-bg, np.nan)


def ncc(template, search):
    ok = np.isfinite(search)
    b = np.nan_to_num(search)
    kernel = np.ones(template.shape)
    n = template.size
    count = fftconvolve(ok.astype(float), kernel, mode='valid')
    sm = fftconvolve(b, kernel, mode='valid')
    ss = fftconvolve(b*b, kernel, mode='valid')
    t = template-template.mean()
    denom = np.sqrt(np.maximum(ss-sm*sm/n, 0)*np.sum(t*t))
    c = fftconvolve(b, t[::-1, ::-1], mode='valid')/np.maximum(denom, 1e-12)
    c[count < n-.1] = np.nan
    return c


def motions(cube, times, max_speed=1.1):
    """Patch NCC with a search radius based on actual elapsed seconds, including gaps."""
    H = np.array([highpass(a) for a in cube])
    ny, nx = cube.shape[1:]
    rows = []
    half = 15
    for lag in (8, 16, 24):
        for start in (0, 12, 25):
            end = start + lag
            if end >= len(cube):
                continue
            dt = times[end]-times[start]
            radius = int(np.ceil(dt*max_speed))
            for cy in range(half+radius, ny-half-radius, 18):
                for cx in range(half+radius, nx-half-radius, 18):
                    t = H[start, cy-half:cy+half+1, cx-half:cx+half+1]
                    if not np.isfinite(t).all() or t.std() < .08:
                        continue
                    s = H[end, cy-half-radius:cy+half+radius+1, cx-half-radius:cx+half+radius+1]
                    c = ncc(t, s)
                    filled = np.nan_to_num(c, nan=-2)
                    py, px = np.where((filled == maximum_filter(filled, size=5)) & (c > .35))
                    for rank, k in enumerate(np.argsort(c[py, px])[::-1][:4]):
                        dy, dx = float(py[k]-radius), float(px[k]-radius)
                        if abs(dy) >= radius or abs(dx) >= radius:
                            continue
                        for axis in (0, 1):
                            p, q = int(py[k]), int(px[k])
                            z = c[p-1:p+2, q] if axis == 0 else c[p, q-1:q+2]
                            if len(z) == 3 and np.isfinite(z).all():
                                den = z[0]-2*z[1]+z[2]
                                delta = np.clip(.5*(z[0]-z[2])/den, -.5, .5) if den < 0 else 0
                                if axis == 0:
                                    dy += delta
                                else:
                                    dx += delta
                        rows.append([cx, cy, start, end, dx/dt, dy/dt, float(c[py[k], px[k]]), rank])
    result = np.asarray(rows).reshape(-1, 8)
    if len(result) < 10:
        raise ValueError('Too few motion candidates; change the ROI/search settings')
    return result


def two_modes(rows, separation=.15):
    """Select two supported velocity modes without slow/fast physical class thresholds."""
    velocity = rows[:, 4:6]
    weights = np.maximum(rows[:, 6]-.35, 0)**2/(1+rows[:, 7])
    bins = np.linspace(-1.1, 1.1, 111)
    hist, xe, ye = np.histogram2d(*velocity.T, bins=(bins, bins), weights=weights)
    hist = gaussian_filter(hist, 1)
    cx, cy = np.meshgrid((xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2, indexing='ij')
    modes, supports = [], []
    for _ in range(2):
        if hist.max() <= 0:
            raise ValueError('A second supported motion mode was not found')
        ix, iy = np.unravel_index(hist.argmax(), hist.shape)
        center = np.array([cx[ix, iy], cy[ix, iy]])
        close = np.linalg.norm(velocity-center, axis=1) < .09
        if close.sum() < 5:
            raise ValueError('Velocity mode has fewer than five candidate peaks')
        v = np.average(velocity[close], axis=0, weights=weights[close])
        modes.append(v)
        supports.append(int(close.sum()))
        hist[np.hypot(cx-v[0], cy-v[1]) < separation] = 0
    order = np.argsort(np.linalg.norm(modes, axis=1))
    return np.asarray(modes)[order], np.asarray(supports)[order]


def warp(a, velocity, dt):
    yy, xx = np.indices(a.shape, dtype=float)
    coords = [yy-velocity[1]*dt, xx-velocity[0]*dt]
    weight = map_coordinates(np.isfinite(a).astype(float), coords, order=1, mode='constant', cval=0)
    b = map_coordinates(np.nan_to_num(a), coords, order=1, mode='constant', cval=0)
    return np.where(weight > .999, b/np.maximum(weight, 1e-12), np.nan)


def predict(a, v, times):
    return np.array([warp(a, v, t) for t in times])


def stack_layer(cube, v, times):
    aligned = np.array([warp(a, v, -t) for a, t in zip(cube, times)])
    count = np.isfinite(aligned).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        template = np.nanmedian(aligned, axis=0)
    template[count < 10] = np.nan
    return template, count


def separate(cube, times, velocities, iterations, folder):
    """Original alternating-median model; labels mean relative speed in this mapping only."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    vslow, vfast = velocities
    slow, ns = stack_layer(cube, vslow, times)
    fast = np.zeros_like(slow)
    slowseq = predict(slow, vslow, times)
    history = []
    fixed = None
    save_at = {iterations} | {2**i for i in range(iterations.bit_length()) if 2**i <= iterations}
    for iteration in range(1, iterations+1):
        candidate, nf = stack_layer(cube-slowseq, vfast, times)
        fast = np.where(np.isfinite(candidate), candidate, fast)
        offset = np.nanmedian(fast)
        fast -= offset
        slow += offset
        fastseq = predict(fast, vfast, times)
        candidate, ns = stack_layer(cube-fastseq, vslow, times)
        slow = np.where(np.isfinite(candidate), candidate, slow)
        slowseq = predict(slow, vslow, times)
        residual = cube-slowseq-fastseq
        if fixed is None:
            fixed = np.all(np.isfinite(residual), axis=0)
            if not fixed.any():
                raise ValueError('No supported evaluation region')
        if not np.isfinite(residual[:, fixed]).all():
            raise ValueError('Evaluation support changed')
        rmse = float(np.sqrt(np.mean(residual[:, fixed]**2)))
        history.append(rmse)
        if iteration in save_at:
            np.savez_compressed(folder / f'checkpoint_{iteration:04d}.npz', slow=slow,
                                fast=fast, slow_support=ns, fast_support=nf,
                                velocities=velocities, evaluation_mask=fixed)
            print(f'{folder.name}, iteration {iteration}: RMSE {rmse:.5f}', flush=True)
    return dict(slow=slow, fast=fast, velocities=velocities, history=np.array(history),
                mask=fixed, residual=residual, slow_support=ns, fast_support=nf)


def bilinear_indices(lat, lon, grid_lat, grid_lon):
    y = (lat-grid_lat[0])/(grid_lat[1]-grid_lat[0])
    x = (lon-grid_lon[0])/(grid_lon[1]-grid_lon[0])
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x < len(grid_lon)-1) & (y < len(grid_lat)-1)
    x, y = x[valid], y[valid]
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x-ix, y-iy
    idx = np.stack([iy*len(grid_lon)+ix, iy*len(grid_lon)+ix+1,
                    (iy+1)*len(grid_lon)+ix, (iy+1)*len(grid_lon)+ix+1], axis=1)
    w = np.stack([(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy], axis=1)
    return valid, idx, w


def joint_static(folder, stride=2, iterations=64, smoothness=.3):
    """Experimental two-height model in original observation space, no double-counted data.

    Fits I[t,p] = A85(geo85[t,p]) + C15(geo15[t,p]) using bilinear sampling.
    Static maps, first-difference regularization; gauge sets mean(C15)=0 afterward.
    It does not model winds or intrinsic wave evolution. Outputs are signed components.
    """
    folder = Path(folder)
    d = np.load(folder / 'dual_projected_frames.npz')
    g = np.load(folder / 'joint_geometry.npz')
    lat, lon = d['latitude'], d['longitude']
    shape = (len(lat), len(lon))
    size = int(np.prod(shape))
    selection = (slice(None), slice(None, None, stride), slice(None, None, stride))
    values = g['radiance'][selection].ravel()
    geos = [(g[f'latitude{h}'][selection].ravel(), g[f'longitude{h}'][selection].ravel()) for h in (85, 15)]
    masks = [bilinear_indices(a, b, lat, lon)[0] for a, b in geos]
    valid = masks[0] & masks[1] & np.isfinite(values)
    operators = [bilinear_indices(a[valid], b[valid], lat, lon)[1:] for a, b in geos]
    b = values[valid]
    if not len(b):
        raise ValueError('No valid joint observations')
    nxedges, nyedges = shape[0]*(shape[1]-1), (shape[0]-1)*shape[1]
    nedges = nxedges+nyedges
    lam = np.sqrt(smoothness)

    def matvec(x):
        out = sum(np.sum(x[k*size:(k+1)*size][idx]*w, axis=1) for k, (idx, w) in enumerate(operators))
        grads = []
        for a in x.reshape(2, *shape):
            grads.extend([lam*np.diff(a, axis=1).ravel(), lam*np.diff(a, axis=0).ravel()])
        return np.concatenate([out, *grads])

    def rmatvec(y):
        result = np.zeros((2, *shape))
        for k, (idx, w) in enumerate(operators):
            result[k] = np.bincount(idx.ravel(), (w*y[:len(b), None]).ravel(), minlength=size).reshape(shape)
            offset = len(b)+k*nedges
            dx = lam*y[offset:offset+nxedges].reshape(shape[0], shape[1]-1)
            dy = lam*y[offset+nxedges:offset+nedges].reshape(shape[0]-1, shape[1])
            result[k, :, :-1] -= dx
            result[k, :, 1:] += dx
            result[k, :-1, :] -= dy
            result[k, 1:, :] += dy
        return result.ravel()

    op = LinearOperator((len(b)+2*nedges, 2*size), matvec=matvec, rmatvec=rmatvec, dtype=float)
    # An adjoint mismatch invalidates this iterative least-squares solver.
    rng = np.random.default_rng(42)
    xcheck, ycheck = rng.normal(size=op.shape[1]), rng.normal(size=op.shape[0])
    if not np.isclose(np.dot(op@xcheck, ycheck), np.dot(xcheck, op.rmatvec(ycheck)), rtol=1e-9, atol=1e-7):
        raise AssertionError('Joint operator failed adjoint test')
    solution = lsmr(op, np.r_[b, np.zeros(2*nedges)], maxiter=iterations, atol=1e-6, btol=1e-6)
    maps = solution[0].reshape(2, *shape)
    support = np.array([np.bincount(idx.ravel(), w.ravel(), minlength=size).reshape(shape) for idx, w in operators])
    offset = np.mean(maps[1][support[1] > 0])
    maps[1] -= offset
    maps[0] += offset
    predicted = matvec(maps.ravel())[:len(b)]
    rmse = float(np.sqrt(np.mean((b-predicted)**2)))
    maps[support == 0] = np.nan
    np.savez_compressed(folder / 'joint_static_result.npz', airglow85_candidate=maps[0],
                        cloud15_signed_candidate=maps[1], support=support,
                        latitude=lat, longitude=lon, reconstruction_rmse=rmse,
                        solver_stop_code=solution[1], iterations=solution[2], observations=len(b))
    return maps, dict(rmse=rmse, stop_code=int(solution[1]), iterations=int(solution[2]),
                      observations=len(b), stride=stride, smoothness=smoothness)
