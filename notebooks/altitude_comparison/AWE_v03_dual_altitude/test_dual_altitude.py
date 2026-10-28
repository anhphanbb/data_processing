"""Numerical checks of shifts, mask behavior and two-height forward/adjoint fit."""
from pathlib import Path
import sys
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import dual_altitude as da

a = np.full((12,12), np.nan)
a[3:9,3:9] = np.arange(36).reshape(6,6)
b = da.warp(a, np.array([1.,0.]), 1.)
assert np.array_equal(b[3:9,4:10], a[3:9,3:9])
assert np.isnan(b[:,:4]).all()
grid = np.arange(28., dtype=float)
y,x = np.meshgrid(grid,grid,indexing='ij')
A = 4 + np.sin(x*.4) + .5*np.cos(y*.6)
C = np.sin(y*.35+x*.2)
la85=[]; lo85=[]; la15=[]; lo15=[]; values=[]
for t in np.linspace(-3,3,13):
    yy,xx=np.meshgrid(np.arange(4.,20.),np.arange(4.,20.),indexing='ij')
    a85,b85=yy,xx+t
    a15,b15=yy+1.4*np.sin(t),xx-t+.15*t*t
    _,idx,w=da.bilinear_indices(a85.ravel(),b85.ravel(),grid,grid)
    first=(A.ravel()[idx]*w).sum(axis=1)
    _,idx,w=da.bilinear_indices(a15.ravel(),b15.ravel(),grid,grid)
    second=(C.ravel()[idx]*w).sum(axis=1)
    values.append((first+second).reshape(xx.shape))
    la85.append(a85); lo85.append(b85); la15.append(a15); lo15.append(b15)
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
    p=Path(tmp)
    np.savez(p/'dual_projected_frames.npz',latitude=grid,longitude=grid)
    np.savez(p/'joint_geometry.npz',radiance=values,latitude85=la85,longitude85=lo85,latitude15=la15,longitude15=lo15)
    maps,metrics=da.joint_static(p,stride=1,iterations=160,smoothness=.001)
    assert metrics['rmse'] < .01, metrics
    # Gauge-invariant contrast recovery on the well-observed interior.
    interior=np.s_[5:19,5:19]
    for recovered,truth in zip(maps,(A,C)):
        err=recovered[interior]-truth[interior]
        assert np.std(err)<.05, np.std(err)
    print('Shift/mask, joint adjoint, synthetic recovery passed:',metrics)
