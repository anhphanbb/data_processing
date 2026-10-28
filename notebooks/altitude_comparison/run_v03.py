"""Execute the notebook code locally and keep a concise log; no package installation."""
import os
from pathlib import Path
import json
os.environ['MPLBACKEND'] = 'Agg'
folder = Path(__file__).parent/'AWE_v03_dual_altitude'
os.chdir(folder)
import sys
sys.path.insert(0, str(folder.parent/'.runtime'))
sys.path.insert(0, str(folder.resolve()))
import nbformat
from IPython.utils.capture import capture_output
from IPython.display import display, Image
import matplotlib.pyplot as plt
from io import BytesIO

def show(*args, **kwargs):
    for number in plt.get_fignums():
        fig = plt.figure(number)
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100)
        display(Image(data=buffer.getvalue()))
    plt.close('all')

plt.show = show
notebook = json.loads(Path('AWE_v03_dual_altitude.ipynb').read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = cell['source']
    compile(source, f'cell_{i}', 'exec')
    print(f'Executing cell {i}', flush=True)
    with capture_output() as captured:
        exec(compile(source, f'cell_{i}', 'exec'), namespace)
    cell['execution_count'] = i+1
    cell['outputs'] = []
    if captured.stdout:
        print(captured.stdout, flush=True)
        cell['outputs'].append(dict(output_type='stream', name='stdout', text=captured.stdout))
    if captured.stderr:
        print(captured.stderr, flush=True)
        cell['outputs'].append(dict(output_type='stream', name='stderr', text=captured.stderr))
    for output in captured.outputs:
        cell['outputs'].append(dict(output_type='display_data', data=output.data, metadata=output.metadata))
    Path('AWE_v03_dual_altitude.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    nbformat.validate(nbformat.from_dict(notebook))
print('All notebook code cells completed.', flush=True)
