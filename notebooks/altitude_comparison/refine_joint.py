from pathlib import Path
import os,sys,json
folder=Path(__file__).parent/'AWE_v03_dual_altitude'
sys.path.insert(0,str(folder.parent/'.runtime'))
sys.path.insert(0,str(folder))
os.environ['MPLBACKEND']='Agg'
os.chdir(folder)
import matplotlib.pyplot as plt
import numpy as np
import dual_altitude as da
path=folder/'AWE_v03_dual_altitude.ipynb'
nb=json.loads(path.read_text(encoding='utf-8'))
nb['cells'][1]['source']=nb['cells'][1]['source'].replace('JOINT_PIXEL_STRIDE = 4    # Every fourth','JOINT_PIXEL_STRIDE = 2    # Every second')
maps,metrics=da.joint_static(folder,stride=2,iterations=64,smoothness=.3)
d=np.load('dual_projected_frames.npz')
lat,lon=d['latitude'],d['longitude']
dx,dy=np.median(np.diff(lon)),np.median(np.diff(lat))
extent=[lon[0]-dx/2,lon[-1]+dx/2,lat[0]-dy/2,lat[-1]+dy/2]
lo,hi=np.nanpercentile(np.concatenate([d['radiance85'].ravel(),d['radiance15'].ravel()]),[2,98])
fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
for ax,a,title,cmap,limits in zip(axes,maps,
        ['85 km layer candidate (brightness baseline)','15 km layer candidate (signed contrast)'],
        ['gray','RdBu_r'],[(lo,hi),(-2,2)]):
    im=ax.imshow(a,origin='lower',extent=extent,aspect=1/np.cos(np.deg2rad(lat.mean())),cmap=cmap,vmin=limits[0],vmax=limits[1])
    ax.set(title=title,xlabel='Longitude',ylabel='Latitude')
    fig.colorbar(im,ax=ax,shrink=.7,label='kR')
fig.savefig('dual_results_64/joint_layers.png',dpi=150)
Path('dual_results_64/joint_metrics.json').write_text(json.dumps(metrics,indent=2))
nb['cells'][11]['outputs']=[dict(output_type='stream',name='stdout',text=json.dumps(metrics,indent=2))]
path.write_text(json.dumps(nb,indent=1),encoding='utf-8')
print(metrics)
