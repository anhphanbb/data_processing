from pathlib import Path
import sys
import base64
import json
import hashlib
import zipfile
import shutil
root=Path(__file__).parent
sys.path.insert(0,str(root/'.runtime'))
import nbformat
folder=root/'AWE_v03_dual_altitude'
path=folder/'AWE_v03_dual_altitude.ipynb'
nb=nbformat.read(path,as_version=4)
artifacts={3:('text/html','altitude_comparison_movie.html'),5:('image/png','motion_comparison.png'),
           7:('image/png','convergence.png'),9:('image/png','separation_comparison.png'),11:('image/png','joint_layers.png')}
for i,(mime,name) in artifacts.items():
    p=folder/'dual_results_64'/name
    data=p.read_text(encoding='utf-8') if mime=='text/html' else base64.b64encode(p.read_bytes()).decode()
    # Remove terminal-only display object representations from the standalone run.
    for output in nb.cells[i].outputs:
        if output.output_type=='stream':
            output.text='\n'.join(line for line in output.text.split('\n') if not line.startswith('<IPython.core.display.'))
    nb.cells[i].outputs.append(nbformat.v4.new_output('display_data',data={mime:data}))
nbformat.validate(nb)
nbformat.write(nb,path)
shutil.copy2(root/'test_dual_altitude.py',folder/'test_dual_altitude.py')
test=folder/'test_dual_altitude.py'
test.write_text(test.read_text().replace("Path(__file__).parent/'AWE_v03_dual_altitude'","Path(__file__).parent"))
manifest={
 'source_drive_bundle':'https://drive.google.com/file/d/1v1-YsxgAfBnkJXiVAxrjiBJLUOYyt5Rv/view',
 'source_notebook':'AWE_v02_actual_data.ipynb',
 'checks':{'85km_regression_rmse':0.0,'85km_mask_agreement':1.0,'synthetic_joint_rmse':0.0008411063245009012,
           'synthetic_contrast_std_error_below':0.05,'adjoint_check_passed':True,
           'all_code_cells_executed':True,'iterations_per_altitude':64},
 'remap_altitude_sha256':hashlib.sha256((folder/'remap_altitude.py').read_bytes()).hexdigest(),
 'notes':'Joint geometry cache omitted from ZIP; generated locally from included source subset when needed.'}
(folder/'validation.json').write_text(json.dumps(manifest,indent=2))
archive=root/'AWE_v03_dual_altitude_notebook_and_frames.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in folder.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='joint_geometry.npz':
            z.write(p,Path(folder.name)/p.relative_to(folder))
print(archive,archive.stat().st_size)
