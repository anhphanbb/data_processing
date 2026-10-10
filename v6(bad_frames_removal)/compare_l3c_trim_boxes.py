# -*- coding: utf-8 -*-
r"""Compare AWE L3C trim and boxes products by geography, not array indices.

Edit SETTINGS and run with Spyder/F5, or:
  python compare_l3c_trim_boxes.py
  python compare_l3c_trim_boxes_parallel.py --workers 4
  python compare_l3c_trim_boxes.py --orbits 9704 9943 --match-km 0.5
  python compare_l3c_trim_boxes.py --self-test
Requirements: numpy scipy matplotlib netCDF4
Install in the environment used to run this script:
  python -m pip install numpy scipy matplotlib netCDF4

Outputs (monthly reports plus individual orbit plots):
  orbit_geographic_bins/*.png: original-pixel trim/boxes panels and 10 km bin differences.
  orbit_geographic_bins/*.csv: shared 10 km bin means, differences,
      counts and mean Pixel_Epoch differences. No epoch cutoff; no interpolation.
      Geographic bin statistics weight occupied common bins equally.
  orbit_swaths/*.png: trim/boxes/difference, aligned horizontally by Pixel_Epoch
  per_orbit.csv: counts, grid shapes, latitude ranges, common-pixel differences
  global_1deg.csv: all 64800 bins, observation counts and distinct orbit counts
  summary.json / summary.txt: pooled statistics, pairing/matching definitions
  inventory.csv: missing counterparts, ambiguous filenames, processing errors
  coverage_1deg.png: valid observation counts, net change, orbit coverage change
  common_radiance_1deg.png: bias and RMSE (boxes minus trim), common pairs only
  latitude_coverage.png: valid observations and orbit coverage by latitude
  orbit_comparison.png: valid counts and latitude envelopes for each orbit

Files are paired by orbit/filter/version/suffix, ignoring filename start time.
Default scans are NOT recursive. Nested counterpart folders are excluded from scans.
Only successfully read pairs enter overall comparisons, to avoid unequal samples.
Valid = finite Radiance > 0 with finite coordinates, latitude -90..90 and
longitude -180..360. NetCDF fill/missing values are excluded; scale/offset decoded.
Pixel counts are observations, NOT unique ground locations or area. Occupied
1-degree bins indicate coarse coverage; degree bins are not equal-area cells.
No cloud threshold is applied.

Common pixels = mutual nearest neighbors on a spherical Earth, within MATCH_KM.
This is a one-to-one near-coincident comparison, not resampling. Exact same-grid
matches have near-zero distance. Unmatched does not necessarily mean new/lost
coverage: different grid placement can also prevent matching. Match distance
quantiles and nearest-neighbor distances are recorded for diagnosis. Tighten or
loosen --match-km with care relative to the nominal 2 km pixel spacing.
For monthly statistics, Pixel_Epoch diagnoses source times but does not restrict
geographic matching. Per-orbit DIFFERENCE panels additionally require |delta time|
<= EPOCH_TOLERANCE_SECONDS. Top panels use each pixel's Pixel_Epoch for horizontal
placement. Vertical coordinates are native centered cross-track grid offsets,
not a common geographic projection. Missing/incompatible epochs skip the orbit
plot and are reported; no time is inferred from array indices. No interpolation is performed.
Global common radiance statistics are pixel-weighted, not averages of orbit means.
Maps use longitude/latitude graticules; no coastline download or internet needed.
Input NetCDF files are opened read-only. Existing report files are overwritten.
"""
from pathlib import Path
import argparse
import csv
import json
import re
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from netCDF4 import Dataset

# -------------------------- SETTINGS --------------------------
TRIM_FOLDER = Path(r'E:\soc\l3c\2023\11')
BOXES_FOLDER = Path(r'E:\soc\l3c\2023\11\boxes')
OUTPUT_FOLDER = TRIM_FOLDER / 'comparison_trim_boxes'
ORBITS = None                 # e.g. [9704, 9943]; None = all matching orbits
MATCH_KM = 0.9                # Geographic tolerance, not pixel size
RECURSIVE = False             # Nested counterpart and output folders are excluded
WORKERS = 4                  # Orbit pairs processed simultaneously; 1 = serial
MAKE_PLOTS = True
GEOGRAPHIC_BIN_KM = 10.0      # Shared orbit-aligned geographic bins
MAKE_ORBIT_PLOTS = True       # One three-panel image per successfully paired file
EPOCH_TOLERANCE_SECONDS = 0.05 # For per-orbit difference panel only
TIME_BIN_SECONDS = 1.1015625   # Display bin width; widened for very long swaths
MAX_TIME_PLOT_BINS = 1800
MAX_CROSS_PLOT_BINS = 600
MISSING_EPOCH_UNIT_SECONDS = 1.0  # AWE Epoch convention when units are absent
SWATH_RADIANCE_LIMITS = (0.0, 20.0) # Same grayscale in trim/boxes; None = shared 1..99 percentiles
SWATH_DIFFERENCE_LIMIT = None  # Symmetric color limit; None = 99th percentile |difference|
# --------------------------------------------------------------
EARTH_KM = 6371.0
NAME = re.compile(r'^awe_l3c_(?P<filter>q20|p12|p14|bkg)_(?P<stamp>\d{7}T\d{4})_'
                  r'(?P<orbit>\d{5})_(?P<version>v\d+)(?P<suffix>.*?)\.nc$', re.I)


def numeric(var):
    a = np.ma.asarray(var[:], dtype=np.float64)
    return np.asarray(a.filled(np.nan))


def read_swath(path):
    with Dataset(path, 'r') as ds:
        arrays = {}
        for name in ('Radiance', 'Latitude', 'Longitude'):
            if name not in ds.variables: raise ValueError(f'Missing variable {name}')
            v = ds[name]
            a = numeric(v)
            dims = list(v.dimensions)
            if 'time' in dims:
                axis = dims.index('time')
                if a.shape[axis] != 1: raise ValueError('Expected a single-time L3C swath')
                a = np.take(a, 0, axis=axis); dims.pop(axis)
            if a.ndim != 2: raise ValueError(f'{name}: expected 2D swath; got {a.shape}')
            arrays[name] = (a, tuple(dims))
        r, dims = arrays['Radiance']
        la, ld = arrays['Latitude']; lo, od = arrays['Longitude']
        if la.shape != r.shape or lo.shape != r.shape or dims != ld or dims != od:
            raise ValueError('Radiance and coordinates have different shapes/dimensions')
        valid = np.isfinite(r) & (r > 0) & np.isfinite(la) & np.isfinite(lo)
        valid &= (abs(la) <= 90) & (lo >= -180) & (lo <= 360)
        epoch = None
        if 'Pixel_Epoch' in ds.variables:
            v = ds['Pixel_Epoch']; e = numeric(v); ed = list(v.dimensions)
            if 'time' in ed:
                axis = ed.index('time')
                if e.shape[axis] != 1: raise ValueError('Pixel_Epoch must have one time plane')
                e = np.take(e, 0, axis=axis); ed.pop(axis)
            if e.shape != r.shape or tuple(ed) != dims: raise ValueError('Pixel_Epoch grid mismatch')
            epoch = e[valid]
        units = getattr(ds['Radiance'], 'units', getattr(ds['Radiance'], 'UNITS', 'unspecified'))
        ep_signature = None
        if epoch is not None:
            v = ds['Pixel_Epoch']
            ep_signature = (str(getattr(v, 'units', getattr(v, 'UNITS', ''))),
                            str(getattr(v, 'calendar', '')),
                            str(getattr(v, 'TIME_BASE', '')), str(getattr(v, 'TIME_SCALE', '')))
        resolution=float(getattr(ds,'Horizontal_Resolution',2.0))
        if not np.isfinite(resolution) or resolution<=0: raise ValueError('Invalid Horizontal_Resolution')
        cross_axis=next((k for k,d in enumerate(dims) if 'cross' in d.lower()),0)
        cross_km=(np.nonzero(valid)[cross_axis]-(r.shape[cross_axis]-1)/2)*resolution
        return dict(cross_km=cross_km,resolution_km=resolution,r=r[valid],lat=la[valid],lon=(lo[valid]+180)%360-180,
                    epoch=epoch,epoch_signature=ep_signature,units=str(units),
                    shape=r.shape,dims=dims,total=r.size,n=int(valid.sum()))


def xyz(a):
    lat, lon = np.deg2rad(a['lat']), np.deg2rad(a['lon'])
    return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def distance_km(chord):
    return 2*EARTH_KM*np.arcsin(np.clip(chord/2, 0, 1))


def match_pixels(a,b,tolerance):
    if not a['n'] or not b['n']:
        return np.array([],int),np.array([],int),np.array([]),np.array([])
    xa, xb = xyz(a), xyz(b)
    db, ib = cKDTree(xb).query(xa, k=1)
    _, ia = cKDTree(xa).query(xb, k=1)
    nearest = distance_km(db)
    keep = (ia[ib] == np.arange(a['n'])) & (nearest <= tolerance)
    i = np.flatnonzero(keep)
    return i, ib[i], nearest[keep], nearest


def bins(lat,lon):
    y = np.clip(np.floor(lat+90).astype(int),0,179)
    x = np.clip(np.floor(((lon+180)%360)).astype(int),0,359)
    return y*360+x


def hist(a,selected=None):
    idx = bins(a['lat'],a['lon'])
    if selected is not None: idx = idx[selected]
    return np.bincount(idx,minlength=64800).reshape(180,360)


def csv_write(path,rows,fields=None):
    if fields is None: fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def stats(v,prefix):
    v = np.asarray(v); v = v[np.isfinite(v)]
    return {prefix+'_'+k:float(fn(v)) if v.size else None for k,fn in
            [('min',np.min),('median',np.median),('p95',lambda x:np.percentile(x,95)),('max',np.max)]}


def scan(folder,recursive,exclude,orbits):
    groups={}
    files=folder.rglob('*.nc') if recursive else folder.glob('*.nc')
    for path in sorted(files):
        resolved=path.resolve()
        if any(resolved == p or p in resolved.parents for p in exclude): continue
        m=NAME.match(path.name)
        if not m: continue
        if orbits is not None and int(m['orbit']) not in orbits: continue
        key=(m['orbit'],m['filter'].lower(),m['version'].lower(),m['suffix'].lower())
        groups.setdefault(key,[]).append(path)
    return groups


def make_plots(out,grids,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    def draw(ax, data, title, label, signed=False, count=False, vmax=None):
        data=np.asarray(data,float)
        if count:
            show=np.ma.masked_where(data <= 0,data)
            norm=LogNorm(vmin=1,vmax=max(2,vmax or float(np.nanmax(data))))
            image=ax.pcolormesh(np.arange(-180,181),np.arange(-90,91),show,
                                cmap='viridis',norm=norm,rasterized=True)
        else:
            limit=max(float(np.nanmax(abs(data))) if np.isfinite(data).any() else 0,1e-12)
            kw=dict(cmap='RdBu_r',vmin=-limit,vmax=limit) if signed else dict(cmap='magma',vmin=0)
            image=ax.pcolormesh(np.arange(-180,181),np.arange(-90,91),data,**kw,rasterized=True)
        ax.set(xlim=(-180,180),ylim=(-90,90),title=title,xlabel='Longitude (°)',ylabel='Latitude (°)')
        ax.set_xticks(np.arange(-180,181,60));ax.set_yticks(np.arange(-90,91,30));ax.grid(alpha=.18)
        fig.colorbar(image,ax=ax,label=label,shrink=.85)
    fig,axes=plt.subplots(2,2,figsize=(15,8),constrained_layout=True)
    vmax=max(grids['trim'].max(),grids['boxes'].max(),2)
    draw(axes[0,0],grids['trim'],'Trim: valid observations','Pixel observations',count=True,vmax=vmax)
    draw(axes[0,1],grids['boxes'],'Boxes: valid observations','Pixel observations',count=True,vmax=vmax)
    draw(axes[1,0],grids['boxes']-grids['trim'],'Boxes − trim: observation count','Δ pixel observations',signed=True)
    draw(axes[1,1],grids['boxes_orbits']-grids['trim_orbits'],'Boxes − trim: distinct orbit count','Δ orbits',signed=True)
    fig.suptitle('1° × 1° geographic bins • successfully paired files only • bins are not equal area')
    fig.savefig(out/'coverage_1deg.png',dpi=170);plt.close(fig)
    n=grids['common_trim']
    mean=np.divide(grids['delta_sum'],n,out=np.full(n.shape,np.nan),where=n>0)
    rmse=np.sqrt(np.divide(grids['delta_sq'],n,out=np.full(n.shape,np.nan),where=n>0))
    fig,axes=plt.subplots(1,2,figsize=(15,4.5),constrained_layout=True)
    draw(axes[0],mean,'Common pixels: mean boxes − trim','Radiance difference (file units)',signed=True)
    draw(axes[1],rmse,'Common pixels: RMSE','Radiance RMSE (file units)')
    fig.suptitle('One-to-one geographic matches • assigned to trim pixel bins')
    fig.savefig(out/'common_radiance_1deg.png',dpi=170);plt.close(fig)
    lat=np.arange(-89.5,90,1)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for mode in ('trim','boxes'):
        axes[0].plot(grids[mode].sum(axis=1),lat,label=mode)
        axes[1].plot(grids[mode+'_latitude_orbits'],lat,label=mode)
    for ax in axes: ax.set(ylabel='Latitude (°)',ylim=(-90,90));ax.grid(alpha=.25);ax.legend()
    axes[0].set(xlabel='Valid pixel observations per 1° latitude band',title='Observation coverage')
    axes[1].set(xlabel='Distinct orbits per 1° latitude band',title='Orbit coverage')
    fig.savefig(out/'latitude_coverage.png',dpi=170);plt.close(fig)
    x=np.arange(len(rows));fig,axes=plt.subplots(2,1,figsize=(13,7),constrained_layout=True)
    for mode,color in [('trim','tab:blue'),('boxes','tab:orange')]:
        axes[0].plot(x,[r[mode+'_valid'] for r in rows],label=mode,color=color)
        low=np.array([r[mode+'_lat_min'] if r[mode+'_lat_min'] is not None else np.nan for r in rows])
        high=np.array([r[mode+'_lat_max'] if r[mode+'_lat_max'] is not None else np.nan for r in rows])
        axes[1].plot(x,low,color=color,label=mode+' min');axes[1].plot(x,high,color=color,ls='--',label=mode+' max')
    ticks=np.unique(np.linspace(0,len(rows)-1,min(len(rows),15)).astype(int))
    for ax in axes:
        ax.set_xticks(ticks,[rows[i]['orbit'] for i in ticks],rotation=45);ax.grid(alpha=.2);ax.legend()
    axes[0].set(ylabel='Valid pixels',title='Successfully paired files (multiple versions/filters may share an orbit label)')
    axes[1].set(xlabel='Orbit',ylabel='Latitude envelope (°)')
    fig.savefig(out/'orbit_comparison.png',dpi=170);plt.close(fig)


def epoch_unit_seconds(a):
    """Convert time differences; absolute origins must agree between the files."""
    unit=a['epoch_signature'][0].strip().lower() if a['epoch_signature'] else ''
    first=unit.split()[0] if unit else ''
    factors={'s':1.,'sec':1.,'second':1.,'seconds':1.,'ms':.001,
             'millisecond':.001,'milliseconds':.001,'us':1e-6,'microseconds':1e-6,
             'ns':1e-9,'nanoseconds':1e-9,'minutes':60.,'minute':60.,
             'hours':3600.,'hour':3600.,'days':86400.,'day':86400.}
    if first in factors: return factors[first],False
    if not unit: return MISSING_EPOCH_UNIT_SECONDS,True
    raise ValueError(f'Unrecognized Pixel_Epoch units: {unit!r}; cannot label a seconds axis safely')


def raster_mean(t,y,z,time_edges,y_edges):
    """Bin existing pixels for display only; empty bins stay NaN, no interpolation."""
    good=np.isfinite(t)&np.isfinite(y)&np.isfinite(z)
    count=np.histogram2d(y[good],t[good],bins=(y_edges,time_edges))[0]
    total=np.histogram2d(y[good],t[good],bins=(y_edges,time_edges),weights=z[good])[0]
    return np.divide(total,count,out=np.full(total.shape,np.nan),where=count>0)


def plot_orbit_swaths(out,key,a,b,i,j,args):
    """Shared Pixel_Epoch x axis; difference only at spatial AND temporal matches."""
    if a['epoch'] is None or b['epoch'] is None:
        return {'swath_plot_status':'skipped: Pixel_Epoch missing; no index-based time substituted'}
    if a['epoch_signature']!=b['epoch_signature']:
        return {'swath_plot_status':'skipped: Pixel_Epoch time metadata differ'}
    factor,assumed=epoch_unit_seconds(a)
    fa=np.isfinite(a['epoch']);fb=np.isfinite(b['epoch'])
    if not fa.any() or not fb.any():
        return {'swath_plot_status':'skipped: no finite Pixel_Epoch in one file'}
    origin=min(a['epoch'][fa].min(),b['epoch'][fb].min())
    ta=(a['epoch']-origin)*factor;tb=(b['epoch']-origin)*factor
    end=max(ta[fa].max(),tb[fb].max())
    width=max(args.time_bin_seconds,end/MAX_TIME_PLOT_BINS)
    time_edges=np.arange(max(1,int(np.floor(end/width))+1)+1)*width
    dy=min(a['resolution_km'],b['resolution_km'])
    ymin=min(a['cross_km'][fa].min(),b['cross_km'][fb].min())-dy/2
    ymax=max(a['cross_km'][fa].max(),b['cross_km'][fb].max())+dy/2
    dy=max(dy,(ymax-ymin)/MAX_CROSS_PLOT_BINS)
    y_edges=ymin+np.arange(max(1,int(np.ceil((ymax-ymin)/dy)))+1)*dy
    same_time=np.isfinite(ta[i])&np.isfinite(tb[j])&(abs(tb[j]-ta[i])<=args.epoch_tolerance_seconds)
    ii,jj=i[same_time],j[same_time]
    delta=b['r'][jj]-a['r'][ii]
    images=[raster_mean(ta,a['cross_km'],a['r'],time_edges,y_edges),
            raster_mean(tb,b['cross_km'],b['r'],time_edges,y_edges),
            raster_mean(ta[ii],a['cross_km'][ii],delta,time_edges,y_edges)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(17,8),sharex=True,sharey=True,constrained_layout=True)
    cmap=plt.get_cmap('gray').copy();cmap.set_bad('#faf6ed')
    diffmap=plt.get_cmap('RdBu_r').copy();diffmap.set_bad('#faf6ed')
    pool=np.concatenate((a['r'][fa],b['r'][fb]))
    vmin,vmax=SWATH_RADIANCE_LIMITS if SWATH_RADIANCE_LIMITS is not None else np.percentile(pool,[1,99])
    if vmax<=vmin:vmax=vmin+1
    lim=SWATH_DIFFERENCE_LIMIT
    if lim is None:lim=float(np.percentile(abs(delta),99)) if delta.size else 1.
    lim=max(lim,1e-6)
    panels=[('Trim',cmap,vmin,vmax),('Boxes',cmap,vmin,vmax),
            ('Boxes − trim: geographic + Pixel_Epoch matches',diffmap,-lim,lim)]
    tx=(time_edges[:-1]+time_edges[1:])/120
    yy=(y_edges[:-1]+y_edges[1:])/2
    for k,(ax,img,(label,color,lo,hi)) in enumerate(zip(axes,images,panels)):
        mesh=ax.pcolormesh(time_edges/60,y_edges,img,cmap=color,vmin=lo,vmax=hi,rasterized=True)
        ax.set_title(label,loc='left',fontsize=11)
        ax.set_ylabel('Native cross-track offset (km)')
        ax.set_facecolor('#faf6ed')
        fig.colorbar(mesh,ax=ax,pad=.01,label=a['units'] if k<2 else 'Δ '+a['units'],extend='both')
        for bound in (ta[fa].min(),ta[fa].max()):ax.axvline(bound/60,color='tab:blue',lw=.8,ls='--',alpha=.7)
        if k<2 and len(tx)>1 and len(yy)>1:
            src=a if k==0 else b;t=ta if k==0 else tb
            lat=raster_mean(t,src['cross_km'],src['lat'],time_edges,y_edges)
            finite=lat[np.isfinite(lat)]
            levels=[x for x in range(-80,81,10) if finite.size and finite.min()<x<finite.max()]
            if levels:
                c=ax.contour(tx,yy,np.ma.masked_invalid(lat),levels=levels,colors='darkorange',linewidths=.7)
                ax.clabel(c,fmt=lambda x: 'EQ' if x==0 else f'{abs(x):g}'+('N' if x>0 else 'S'),fontsize=8)
    if not len(delta):axes[2].text(.5,.5,'No pairs satisfy both location and Pixel_Epoch tolerances',
                                   transform=axes[2].transAxes,ha='center',va='center')
    axes[-1].set_xlabel('Minutes since earliest valid Pixel_Epoch across both files')
    axes[-1].set_xlim(time_edges[0]/60,time_edges[-1]/60)
    fig.suptitle(f'Orbit {key[0]} | {key[1]} {key[2]}{key[3]} | trim {a["n"]:,}, boxes {b["n"]:,} valid pixels\n'
                 f'Common time/location pairs: {len(delta):,} | Δt ≤ {args.epoch_tolerance_seconds:g} s, distance ≤ {args.match_km:g} km | '
                 f'display bins {width:.3g} s × {dy:.3g} km',fontsize=12)
    note='Dashed lines: trim extent. Blank: no data. Display bins show averages; no interpolation.'
    note+='\nCross-track axes centered separately; differences use geographic matches.'
    if assumed:note+='\nMissing time units: using configured conversion.'
    fig.supxlabel(note,fontsize=8)
    folder=out/'orbit_swaths';folder.mkdir(exist_ok=True)
    filename=f'orbit_{key[0]}_{key[1]}_{key[2]}{key[3]}_pixel_epoch.png'
    path=folder/filename
    fig.savefig(path,dpi=160);plt.close(fig)
    return dict(swath_plot_status='saved',swath_plot=str(path),
                time_location_common_pairs=len(delta),
                time_location_bias=float(delta.mean()) if len(delta) else None,
                time_location_rmse=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
                swath_time_bin_seconds=width,swath_epoch_origin_stored_units=float(origin),
                swath_epoch_units_assumed=assumed)


def common_geographic_bins(a,b,size,metadata=None):
    """Shared spherical oblique grid aligned with trim track; no interpolation.

    x = R * oblique longitude, y = R * oblique latitude. Cells are nominal
    size x size km: along-track physical width scales by cos(y/R).
    """
    if not a['n'] or not b['n']: raise ValueError('No valid pixels in one product')
    av,bv=xyz(a),xyz(b)
    center=av.mean(axis=0);center/=np.linalg.norm(center)
    sample=av[::max(1,len(av)//20000)]
    tangent=sample-np.outer(sample@center,center)
    _,_,vt=np.linalg.svd(tangent,full_matrices=False)
    along=vt[0];along-=np.dot(along,center)*center;along/=np.linalg.norm(along)
    if a['epoch'] is not None:
        good=np.isfinite(a['epoch'])
        if good.sum()>1 and np.dot((av[good]@along),a['epoch'][good]-a['epoch'][good].mean())<0:along=-along
    normal=np.cross(center,along)
    def project(v):
        return EARTH_KM*np.arctan2(v@along,v@center),EARTH_KM*np.arcsin(np.clip(v@normal,-1,1))
    ax,ay=project(av);bx,by=project(bv)
    xe=np.arange(np.floor(min(ax.min(),bx.min())/size),np.floor(max(ax.max(),bx.max())/size)+2)*size
    ye=np.arange(np.floor(min(ay.min(),by.min())/size),np.floor(max(ay.max(),by.max())/size)+2)*size
    if (len(xe)-1)*(len(ye)-1)>10000000:raise ValueError('Projected grid too large; check orbit geometry')
    def aggregate(s,x,y):
        count=np.histogram2d(y,x,bins=(ye,xe))[0].astype(int)
        mean=raster_mean(x,y,s['r'],xe,ye)
        return count,mean
    na,ra=aggregate(a,ax,ay);nb,rb=aggregate(b,bx,by)
    common=(na>0)&(nb>0)
    delta=np.where(common,rb-ra,np.nan)
    dt=np.full(delta.shape,np.nan)
    if a['epoch'] is not None and b['epoch'] is not None and a['epoch_signature']==b['epoch_signature']:
        factor,_=epoch_unit_seconds(a)
        finite=a['epoch'][np.isfinite(a['epoch'])]
        if finite.size:
            origin=finite.min()
            ea=raster_mean(ax,ay,(a['epoch']-origin)*factor,xe,ye)
            eb=raster_mean(bx,by,(b['epoch']-origin)*factor,xe,ye)
            dt=np.where(common,eb-ea,np.nan)
    if metadata is not None:
        xx,yy=np.meshgrid((xe[:-1]+xe[1:])/2/EARTH_KM,(ye[:-1]+ye[1:])/2/EARTH_KM)
        z=np.cos(yy)*(np.cos(xx)*center[2]+np.sin(xx)*along[2])+np.sin(yy)*normal[2]
        metadata.update(trim_x=ax,trim_y=ay,boxes_x=bx,boxes_y=by,
                        latitude=np.rad2deg(np.arcsin(np.clip(z,-1,1))),
                        trim_bounds=(float(ax.min()),float(ax.max())))
    return xe,ye,na,nb,ra,rb,delta,dt


def plot_geographic_bins(out,key,a,b,args):
    size=args.geographic_bin_km
    metadata={}
    xe,ye,na,nb,ra,rb,delta,dt=common_geographic_bins(a,b,size,metadata)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'orbit_geographic_bins';folder.mkdir(exist_ok=True)
    stem=f'orbit_{key[0]}_{key[1]}_{key[2]}{key[3]}_{size:g}km'
    occupied=(na+nb)>0;iy,ix=np.nonzero(occupied)
    csv_write(folder/(stem+'.csv'),[dict(x_km=(xe[x]+xe[x+1])/2,y_km=(ye[y]+ye[y+1])/2,
        trim_count=int(na[y,x]),boxes_count=int(nb[y,x]),trim_mean=ra[y,x],boxes_mean=rb[y,x],
        boxes_minus_trim=delta[y,x],mean_epoch_difference_seconds=dt[y,x]) for y,x in zip(iy,ix)])
    finite=delta[np.isfinite(delta)]
    lim=max(float(np.percentile(abs(finite),99)),1e-6) if finite.size else 1.
    pool=np.r_[a['r'],b['r']]
    lo,hi=SWATH_RADIANCE_LIMITS if SWATH_RADIANCE_LIMITS is not None else np.percentile(pool,[1,99])
    fig,axes=plt.subplots(3,1,figsize=(17,8),sharex=True,sharey=True,constrained_layout=True)
    for panel,(ax,data,title,cmap,vmin,vmax,unit) in enumerate(zip(axes,[ra,rb,delta],
        ['Trim: original pixels','Boxes: original pixels','Boxes − trim: shared geographic bin means'],
        ['gray','gray','RdBu_r'],[lo,lo,-lim],[hi,hi,lim],
        [a['units'],a['units'],'Δ '+a['units']])):
        color=plt.get_cmap(cmap).copy();color.set_bad('#faf6ed')
        ax.set_facecolor('#faf6ed')
        if panel<2:
            src=a if panel==0 else b
            prefix='trim' if panel==0 else 'boxes'
            # Draw every valid native pixel center; no averaging or subsampling.
            mesh=ax.scatter(metadata[prefix+'_x'],metadata[prefix+'_y'],c=src['r'],
                            s=.4,marker='s',linewidths=0,edgecolors='none',
                            cmap=color,vmin=vmin,vmax=vmax,rasterized=True)
        else:
            mesh=ax.pcolormesh(xe,ye,data,cmap=color,vmin=vmin,vmax=vmax,rasterized=True)
        ax.set_xlim(xe[0],xe[-1]);ax.set_ylim(ye[0],ye[-1])
        fig.colorbar(mesh,ax=ax,label=unit,pad=.01,extend='both')
        ax.set_title(title,loc='left');ax.set_ylabel('Cross-track (km)')
    for ax in axes:
        for bound in metadata['trim_bounds']:
            ax.axvline(bound,color='tab:blue',lw=.8,ls='--',alpha=.7)
    xc=(xe[:-1]+xe[1:])/2;yc=(ye[:-1]+ye[1:])/2
    for ax,count in zip(axes[:2],[na,nb]):
        lat=np.where(count>0,metadata['latitude'],np.nan)
        finite_lat=lat[np.isfinite(lat)]
        levels=[v for v in range(-80,81,10) if finite_lat.size and finite_lat.min()<v<finite_lat.max()]
        if levels and len(xc)>1 and len(yc)>1:
            contours=ax.contour(xc,yc,np.ma.masked_invalid(lat),levels=levels,colors='darkorange',linewidths=.7)
            ax.clabel(contours,fmt=lambda v:'EQ' if v==0 else f'{abs(v):g}'+('N' if v>0 else 'S'),fontsize=8)
    axes[-1].set_xlabel('Along-track coordinate on shared geographic grid (km)')
    common=int(np.isfinite(delta).sum())
    bias=float(finite.mean()) if finite.size else None
    rmse=float(np.sqrt(np.mean(finite**2))) if finite.size else None
    stats=f'common bins {common:,}'
    if common:stats+=f' | mean difference {bias:.4g}, RMSE {rmse:.4g} {a["units"]}'
    fig.suptitle(f'Orbit {key[0]} | {key[1]} {key[2]}{key[3]} | trim {a["n"]:,}, boxes {b["n"]:,} valid pixels\nShared {size:g} km bins | '+stats)
    fig.supxlabel('Top: original pixels. Bottom: binned differences. Dashed lines: trim extent.\n'
                 'No interpolation or time filter.',fontsize=9)
    path=folder/(stem+'.png');fig.savefig(path,dpi=150);plt.close(fig)
    return dict(geographic_bin_km=size,geographic_common_bins=common,geographic_bin_bias=bias,
                geographic_bin_rmse=rmse,geographic_trim_only_bins=int(((na>0)&(nb==0)).sum()),
                geographic_boxes_only_bins=int(((nb>0)&(na==0)).sum()),geographic_bin_plot=str(path))


def process_pair(pair, args):
    """Read, compare and plot one pair within an isolated process."""
    key,ap,bp,record = pair
    out=args.output.resolve()
    fields=['orbit','filter','version','suffix']
    grids={k:np.zeros((180,360),dtype=np.float64 if k.startswith('delta_') else np.int64)
           for k in ['trim','boxes','common_trim','common_boxes','delta_sum','delta_sq']}
    dsums=np.zeros(7)
    messages=[]
    def log(message): messages.append(message)
    started=perf_counter()
    try:
        a,b=read_swath(ap),read_swath(bp)
        if a['units'].lower()!=b['units'].lower(): raise ValueError(f'Radiance units differ: {a["units"]} vs {b["units"]}')
        i,j,dist,nearest=match_pixels(a,b,args.match_km)
        d=b['r'][j]-a['r'][i]; n=len(i)
        row=dict(zip(fields[:4],key));row.update(trim_file=str(ap),boxes_file=str(bp))
        for mode,v in [('trim',a),('boxes',b)]:
            row.update({mode+'_shape':'x'.join(map(str,v['shape'])),mode+'_dimensions':','.join(v['dims']),
                        mode+'_array_pixels':v['total'],mode+'_valid':v['n'],
                        mode+'_valid_fraction':v['n']/v['total'],
                        mode+'_lat_min':float(v['lat'].min()) if v['n'] else None,
                        mode+'_lat_max':float(v['lat'].max()) if v['n'] else None})
        row.update(valid_gain=b['n']-a['n'],valid_gain_percent=100*(b['n']-a['n'])/a['n'] if a['n'] else None,
                   common_pairs=n,trim_unmatched=a['n']-n,boxes_unmatched=b['n']-n,
                   trim_matched_percent=100*n/a['n'] if a['n'] else None,
                   boxes_matched_percent=100*n/b['n'] if b['n'] else None,
                   bias=float(d.mean()) if n else None,mae=float(abs(d).mean()) if n else None,
                   rmse=float(np.sqrt(np.mean(d*d))) if n else None,
                   delta_median=float(np.median(d)) if n else None,
                   delta_p05=float(np.percentile(d,5)) if n else None,
                   delta_p95=float(np.percentile(d,95)) if n else None,
                   common_trim_mean=float(a['r'][i].mean()) if n else None,
                   common_boxes_mean=float(b['r'][j].mean()) if n else None)
        row.update(stats(dist,'match_km'));row.update(stats(nearest,'trim_nearest_boxes_km'))
        row['pixel_epoch_status']='unavailable'
        dt=np.array([])
        if a['epoch'] is not None and b['epoch'] is not None:
            if a['epoch_signature'] != b['epoch_signature']:
                row['pixel_epoch_status']='units_or_calendar_mismatch_not_compared'
            else:
                dt=b['epoch'][j]-a['epoch'][i];dt=dt[np.isfinite(dt)]
                row['pixel_epoch_status']='compared_in_stored_units'
        row['pixel_epoch_units']=a['epoch_signature'][0] if a['epoch_signature'] else ''
        row['common_finite_epoch_pairs']=len(dt)
        row.update(stats(abs(dt),'abs_pixel_epoch_difference'))
        # Commit to totals only after the pair has been fully calculated.
        ha,hb=hist(a),hist(b);hca,hcb=hist(a,i),hist(b,j)
        row['trim_occupied_1deg_bins']=int((ha>0).sum())
        row['boxes_occupied_1deg_bins']=int((hb>0).sum())
        for mode in ('trim','boxes'):
            low,high=row[mode+'_lat_min'],row[mode+'_lat_max']
            row[mode+'_latitude_span']=high-low if low is not None else None
        grids['trim']+=ha;grids['boxes']+=hb
        grids['common_trim']+=hca;grids['common_boxes']+=hcb
        idx=bins(a['lat'][i],a['lon'][i])
        grids['delta_sum']+=np.bincount(idx,weights=d,minlength=64800).reshape(180,360)
        grids['delta_sq']+=np.bincount(idx,weights=d*d,minlength=64800).reshape(180,360)
        dsums+=np.array([n,d.sum(),abs(d).sum(),np.sum(d*d),np.sum(a['r'][i]),np.sum(b['r'][j]),np.sum(dist)])
        if not args.no_plots and getattr(args,'orbit_plots',MAKE_ORBIT_PLOTS):
            try:
                row.update(plot_orbit_swaths(out,key,a,b,i,j,args))
                log('  Per-orbit plot: '+row['swath_plot_status'])
            except Exception as exc:
                row['swath_plot_status']='plot error: '+str(exc)
                log('  PLOT ERROR (comparison retained): '+str(exc))
        if not args.no_plots and getattr(args,'orbit_plots',MAKE_ORBIT_PLOTS):
            try:
                row.update(plot_geographic_bins(out,key,a,b,args))
                log('  Shared geographic bins saved')
            except Exception as exc:
                row['geographic_bin_error']=str(exc)
                log('  Geographic bin plot error: '+str(exc))
        if a['n'] and n/a['n'] < .5:
            log(f'  NOTE: only {100*n/a["n"]:.1f}% of trim pixels matched; inspect grid offsets/tolerance.')
        log(f'  valid {a["n"]:,} -> {b["n"]:,}; common={n:,}; net={b["n"]-a["n"]:+,}')
        return dict(row=row, grids=grids, dsums=dsums, units=a['units'],
                    messages=messages, elapsed=perf_counter()-started)
    except Exception as exc:
        return dict(error=str(exc), messages=messages, elapsed=perf_counter()-started)
    finally:
        # Also close partially created figures after plotting errors.
        import sys
        if 'matplotlib.pyplot' in sys.modules:
            sys.modules['matplotlib.pyplot'].close('all')


def pair_results(pairs,args):
    """Bound queued results to limit RAM; reduce in original order for reproducibility."""
    workers=min(getattr(args,'workers',WORKERS),len(pairs))
    if workers<=1:
        for pair in pairs:
            yield pair,process_pair(pair,args)
        return
    # Spawn avoids inheriting open NetCDF/HDF5 state and matches Windows behavior.
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        remaining=iter(pairs)
        pending=deque()
        for _ in range(min(2*workers,len(pairs))):
            pair=next(remaining)
            pending.append((pair,pool.submit(process_pair,pair,args)))
        while pending:
            pair,future=pending.popleft()
            result=future.result()
            yield pair,result
            following=next(remaining,None)
            if following is not None:
                pending.append((following,pool.submit(process_pair,following,args)))


def run(args):
    trim=args.trim.resolve();boxes=args.boxes.resolve();out=args.output.resolve()
    if not trim.is_dir() or not boxes.is_dir(): raise FileNotFoundError('Both input folders must exist')
    if trim == boxes: raise ValueError('Trim and boxes folders must differ')
    if out in (trim,boxes): raise ValueError('Use a dedicated output subfolder')
    out.mkdir(parents=True,exist_ok=True)
    wanted=set(args.orbits) if args.orbits else None
    trim_exclude = [out]
    boxes_exclude = [out]
    if trim in boxes.parents:
        trim_exclude.append(boxes)
    elif boxes in trim.parents:
        boxes_exclude.append(trim)
    ga = scan(trim, args.recursive, trim_exclude, wanted)
    gb = scan(boxes, args.recursive, boxes_exclude, wanted)
    fields=['orbit','filter','version','suffix','status','trim_file','boxes_file','error']
    inventory=[]; rows=[]; pairs=[]
    for key in sorted(ga.keys()|gb.keys()):
        a,b=ga.get(key,[]),gb.get(key,[])
        record=dict(zip(fields[:4],key));record.update(trim_file='; '.join(map(str,a)),boxes_file='; '.join(map(str,b)),error='')
        if len(a)>1 or len(b)>1: record['status']='ambiguous_multiple_files'
        elif not a: record['status']='missing_trim'
        elif not b: record['status']='missing_boxes'
        else: pairs.append((key,a[0],b[0],record));continue
        inventory.append(record)
    grids={k:np.zeros((180,360),dtype=np.float64 if k.startswith('delta_') else np.int64) for k in
           ['trim','boxes','common_trim','common_boxes','trim_orbits','boxes_orbits','delta_sum','delta_sq']}
    for mode in ('trim','boxes'): grids[mode+'_latitude_orbits']=np.zeros(180,np.int64)
    # Union per orbit across filters/versions; avoids inflated distinct-orbit counts.
    orbit_bins={}; dsums=np.zeros(7)
    units=None
    started=perf_counter()
    workers=getattr(args,'workers',WORKERS)
    if workers < 1: raise ValueError('workers must be >= 1')
    print(f'Comparing {len(pairs)} pairs with {min(workers,len(pairs))} worker(s).',flush=True)
    for number,(pair,result) in enumerate(pair_results(pairs,args),1):
        key,ap,bp,record=pair
        print(f'[{number}/{len(pairs)}] Orbit {key[0]} {key[1]} {key[2]} ({result["elapsed"]:.1f} s)',flush=True)
        error=result.get('error')
        if not error and units is not None and result['units'].lower()!=units.lower():
            error='Radiance units differ from previous pairs'
        if error:
            record.update(status='error',error=error)
            inventory.append(record)
            print('  ERROR: '+error,flush=True)
            continue
        for name,values in result['grids'].items(): grids[name]+=values
        for mode in ('trim','boxes'):
            orbit_bins.setdefault((key[0],mode),np.zeros((180,360),bool))[:] |= result['grids'][mode]>0
        dsums+=result['dsums']
        units=result['units'];rows.append(result['row'])
        record['status']='compared';inventory.append(record)
        for message in result['messages']: print(message,flush=True)
    print(f'Pair processing wall time: {perf_counter()-started:.1f} s',flush=True)
    csv_write(out/'inventory.csv',inventory,fields)
    if not rows: raise RuntimeError(f'No pairs successfully compared. See {out/"inventory.csv"}')
    csv_write(out/'per_orbit.csv',rows)
    for (orbit,mode),mask in orbit_bins.items():
        grids[mode+'_orbits']+=mask
        grids[mode+'_latitude_orbits']+=mask.any(axis=1)
    n=dsums[0];ta=int(grids['trim'].sum());tb=int(grids['boxes'].sum())
    summary=dict(successful_file_pairs=len(rows),distinct_orbits=len({r['orbit'] for r in rows}),
                 excluded_inventory_entries=sum(r['status']!='compared' for r in inventory),
                 trim_valid=ta,boxes_valid=tb,valid_gain=tb-ta,
                 valid_gain_percent=100*(tb-ta)/ta if ta else None,
                 common_pairs=int(n),trim_unmatched=ta-int(n),boxes_unmatched=tb-int(n),
                 common_bias=float(dsums[1]/n) if n else None,common_mae=float(dsums[2]/n) if n else None,
                 common_rmse=float(np.sqrt(dsums[3]/n)) if n else None,
                 common_trim_mean=float(dsums[4]/n) if n else None,
                 common_boxes_mean=float(dsums[5]/n) if n else None,
                 mean_match_km=float(dsums[6]/n) if n else None,radiance_units=units,
                 match_tolerance_km=args.match_km,trim_folder=str(trim),boxes_folder=str(boxes),
                 valid_definition='finite Radiance > 0 and valid geographic coordinates; no cloud threshold',
                 matching='mutual nearest spherical neighbors; one-to-one; no time restriction or interpolation',
                 population='successfully paired files only; pixel observations may repeat geographic locations',
                 unmatched_caveat='unmatched pixels are not necessarily added/lost ground coverage',
                 bin_definition='1 degree, left-closed; +90 latitude included in final bin; +180 longitude wraps to -180')
    for mode in ('trim','boxes'):
        for bound in ('min','max'):
            v=[r[f'{mode}_lat_{bound}'] for r in rows if r[f'{mode}_lat_{bound}'] is not None]
            summary[f'{mode}_latitude_{bound}']=(min(v) if bound=='min' else max(v)) if v else None
        summary[mode+'_occupied_1deg_bins']=int((grids[mode]>0).sum())
    summary['boxes_only_occupied_1deg_bins']=int(((grids['boxes']>0)&(grids['trim']==0)).sum())
    summary['trim_only_occupied_1deg_bins']=int(((grids['trim']>0)&(grids['boxes']==0)).sum())
    with (out/'global_1deg.csv').open('w',newline='',encoding='utf-8-sig') as f:
        names=['latitude_center','longitude_center','trim_valid','boxes_valid','valid_gain','common_at_trim','common_at_boxes',
               'trim_unmatched','boxes_unmatched','trim_orbits','boxes_orbits','common_bias','common_rmse']
        w=csv.writer(f);w.writerow(names)
        for y in range(180):
            for x in range(360):
                v={k:float(a[y,x]) for k,a in grids.items() if a.ndim==2};c=v['common_trim']
                w.writerow([y-89.5,x-179.5,int(v['trim']),int(v['boxes']),int(v['boxes']-v['trim']),
                            int(c),int(v['common_boxes']),int(v['trim']-c),int(v['boxes']-v['common_boxes']),
                            int(v['trim_orbits']),int(v['boxes_orbits']),v['delta_sum']/c if c else '',
                            np.sqrt(v['delta_sq']/c) if c else ''])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['AWE L3C: boxes versus trim','',*[f'{k}: {v}' for k,v in summary.items()],
           '', 'Latitude min/max are envelopes; gaps inside the interval may remain.',
           'Common radiance differences may include different source times; see per_orbit Pixel_Epoch diagnostics.',
           'No matching pixels: coverage statistics remain valid; radiance difference statistics are unavailable.']
    (out/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not args.no_plots: make_plots(out,grids,rows)
    print(f'\nPaired files: {len(rows)}; valid pixels: {ta:,} -> {tb:,}; common: {int(n):,}')
    print(f'Reports: {out}')
    return summary


def self_test():
    def sw(lat,lon):
        lat=np.array(lat,float);return dict(lat=lat,lon=np.array(lon,float),n=len(lat))
    a=sw([0,0],[179.999,-179.99]);b=sw([0,0],[-179.999,-179.99])
    i,j,d,_=match_pixels(a,b,.5);assert len(i)==2 and d.max()<.5
    a=sw([0,0],[0,0]);b=sw([0],[0]);assert len(match_pixels(a,b,.5)[0])==1
    assert bins(np.array([-90,90]),np.array([-180,180])).tolist()==[0,179*360]
    # Display bins average actual samples and preserve empty cells.
    raster=raster_mean(np.array([.1,.2]),np.array([0.,0.]),np.array([2.,4.]),
                       np.array([0.,1.,2.]),np.array([-.5,.5]))
    assert raster[0,0]==3 and np.isnan(raster[0,1])
    assert epoch_unit_seconds({'epoch_signature':('milliseconds since 2000-01-01',)})==(.001,False)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);old=root/'old';old.mkdir()
        for folder,nx,offset in [(old,3,0),(root,5,2)]:
            path=folder/'awe_l3c_q20_2023326T0000_00004_v02.nc'
            with Dataset(path,'w') as ds:
                ds.createDimension('time',1);ds.createDimension('y',2);ds.createDimension('x',nx)
                dims=('time','y','x')
                r=ds.createVariable('Radiance','f4',dims,fill_value=np.nan);r.units='kRayleigh';r[:]=10+offset
                ds.createVariable('Latitude','f4',dims)[:]=np.array([0,1])[None,:,None]
                ds.createVariable('Longitude','f4',dims)[:]=np.arange(nx)[None,None,:]
                e=ds.createVariable('Pixel_Epoch','f8',dims);e.units='seconds';e[:]=100+offset
                r[0,0,0]=0  # invalid zero in both modes
        args=argparse.Namespace(trim=old,boxes=root,output=root/'reports',recursive=True,orbits=None,match_km=.5,no_plots=True)
        s=run(args)
        assert s['successful_file_pairs']==1 and s['trim_valid']==5 and s['boxes_valid']==9
        assert s['common_pairs']==5 and s['common_bias']==2 and s['common_rmse']==2
        assert s['trim_unmatched']==0 and s['boxes_unmatched']==4
        # Confirm no false comparisons when grids are separated beyond tolerance.
        path=next(root.glob('*.nc'))
        with Dataset(path,'a') as ds: ds['Longitude'][:]+=30
        s=run(args);assert s['common_pairs']==0 and s['common_bias'] is None
        # Reversed layout: trim in parent, boxes in child; both scan modes.
        with Dataset(path,'a') as ds: ds['Longitude'][:]-=30
        child=root/'boxes';child.mkdir()
        path.rename(child/path.name)
        old_file=next(old.glob('*.nc'))
        old_file.rename(root/old_file.name)
        args.trim=root;args.boxes=child
        for recursive in (False,True):
            args.recursive=recursive
            s=run(args)
            assert s['successful_file_pairs']==1
            assert s['trim_valid']==5 and s['boxes_valid']==9 and s['common_pairs']==5
    print('SELF-TEST PASSED: larger swath, invalid zeros, geographic match, dateline, one-to-one, bins, no-match case.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trim',type=Path,default=TRIM_FOLDER);p.add_argument('--boxes',type=Path,default=BOXES_FOLDER)
    p.add_argument('--output',type=Path,default=OUTPUT_FOLDER);p.add_argument('--orbits',type=int,nargs='+',default=ORBITS)
    p.add_argument('--workers',type=int,default=WORKERS,help='Parallel orbit pairs (1 = serial)')
    p.add_argument('--match-km',type=float,default=MATCH_KM)
    p.add_argument('--recursive',action=argparse.BooleanOptionalAction,default=RECURSIVE)
    p.add_argument('--no-plots',action='store_true',default=not MAKE_PLOTS)
    p.add_argument('--orbit-plots',action=argparse.BooleanOptionalAction,default=MAKE_ORBIT_PLOTS)
    p.add_argument('--epoch-tolerance-seconds',type=float,default=EPOCH_TOLERANCE_SECONDS)
    p.add_argument('--time-bin-seconds',type=float,default=TIME_BIN_SECONDS)
    p.add_argument('--geographic-bin-km',type=float,default=GEOGRAPHIC_BIN_KM)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args(argv)
    if args.workers<1: p.error('--workers must be >= 1')
    if args.self_test: self_test();return
    if not np.isfinite(args.match_km) or args.match_km<=0: p.error('--match-km must be positive and finite')
    if not np.isfinite(args.epoch_tolerance_seconds) or args.epoch_tolerance_seconds<0: p.error('Epoch tolerance must be finite and >=0')
    if not np.isfinite(args.time_bin_seconds) or args.time_bin_seconds<=0: p.error('Time bin must be finite and >0')
    if not np.isfinite(args.geographic_bin_km) or args.geographic_bin_km<=0: p.error('Geographic bin size must be positive and finite')
    run(args)


if __name__=='__main__':
    mp.freeze_support()
    # Spyder/IPython runs scripts in an interactive __main__, which spawn cannot
    # reliably import. Launch this saved file in a regular Python interpreter.
    import sys
    if 'IPython' in sys.modules:
        import subprocess
        completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
        if completed.returncode:
            raise RuntimeError(f'Comparison subprocess exited with code {completed.returncode}')
    else:
        main()
