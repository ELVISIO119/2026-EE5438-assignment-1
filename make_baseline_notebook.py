from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

root=Path(__file__).resolve().parent
nb=nbf.v4.new_notebook(cells=[
    nbf.v4.new_markdown_cell('# Sigmoid/SGD baseline\nCai Haochen | 58561440\n\nFresh 54,000/6,000 stratified split; training-only normalization and validation checkpoint selection.'),
    nbf.v4.new_code_cell((root/'baseline.py').read_text()),
    nbf.v4.new_code_cell("import matplotlib.pyplot as plt\nplt.plot([r['epoch'] for r in history],[100*r['val_accuracy'] for r in history])\nplt.xlabel('Epoch'); plt.ylabel('Validation accuracy (%)'); plt.grid(alpha=.2); plt.show()")],
    metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
NotebookClient(nb,timeout=600,resources={'metadata':{'path':str(root)}}).execute()
nbf.write(nb,root/'notebooks/01_baseline.ipynb')
