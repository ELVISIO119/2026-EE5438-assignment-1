"""Runnable evidence checks; no fitting and no test-driven model selection."""
import argparse
import ast
import json
import re
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZipFile

import nbformat
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from train import OUT,load_data,model_cost
from evaluation import load_model,probabilities,probability_metrics,digest


def checkpoints():
    x,y,vx,vy,_,_=load_data()
    del x,y
    checks=[]
    for path in sorted(OUT.glob('*.json')):
        record=json.loads(path.read_text())
        if 'val_accuracy' not in record:
            continue
        assert record.get('complete',True), path.name
        model=load_model(path.stem)
        measured=probability_metrics(probabilities(model,vx),vy)
        assert abs(measured['accuracy']-record['val_accuracy'])<1e-6,(path.name,measured,record['val_accuracy'])
        params,macs=model_cost(model)
        assert params==record.get('parameters',params) and macs==record.get('macs',macs),path.name
        checks.append(dict(name=path.stem,accuracy=measured['accuracy'],parameters=params,macs=macs,sha256=digest(path.stem)))
        print('Verified checkpoint:',path.stem,measured['accuracy'],flush=True)
        del model
    result=dict(verified_at=datetime.now(timezone.utc).isoformat(),checks=checks,check_count=len(checks))
    recipe_path=OUT/'final_recipe.json'
    if recipe_path.exists():
        recipe=json.loads(recipe_path.read_text())
        components=recipe['components']
        weights=np.asarray([c.get('weight',1.) for c in components],dtype=float)
        assert np.isfinite(weights).all() and (weights>0).all()
        weights/=weights.sum()
        total=None; params=macs=0
        for component,weight in zip(components,weights):
            assert digest(component['name'])==component['sha256']
            model=load_model(component['name'])
            probs=probabilities(model,vx,component['views'])*weight
            total=probs if total is None else total+probs
            count,cost=model_cost(model)
            params+=count; macs+=cost*component['views']
            del model
        measured=probability_metrics(total,vy)
        assert measured['correct']==recipe['correct'],(measured,recipe['correct'])
        assert (params,macs)==(recipe['parameters'],recipe['macs'])
        result['frozen_recipe']=dict(**measured,parameters=params,macs=macs)
        print('Verified frozen ensemble:',measured,flush=True)
    (OUT/'checkpoint_verification.json').write_text(json.dumps(result,indent=2))


def submission():
    directory=Path('submission')
    notebook_path=directory/'Assign01_Cai_Haochen_58561440.ipynb'
    notebook=nbformat.read(notebook_path,as_version=4)
    nbformat.validate(notebook)
    figures=0
    for cell in notebook.cells:
        if cell.cell_type!='code':
            continue
        ast.parse(cell.source)
        assert cell.execution_count is not None
        assert not any(out.output_type=='error' for out in cell.outputs)
        figures+=sum('image/png' in out.get('data',{}) for out in cell.outputs)
    assert figures>=4,figures
    data=np.load(OUT/'final_test_predictions.npz')
    metrics=json.loads((OUT/'final_test_metrics.json').read_text())
    probs=data['probabilities']; pred=probs.argmax(1); labels=data['labels']
    assert probs.shape==(10000,10) and np.isfinite(probs).all()
    assert np.allclose(probs.sum(1),1,atol=1e-5)
    assert np.isclose(accuracy_score(labels,pred),metrics['accuracy'])
    p,r,f,_=precision_recall_fscore_support(labels,pred,average='macro',zero_division=0)
    assert np.allclose([p,r,f],[metrics['macro_precision'],metrics['macro_recall'],metrics['macro_f1']])
    recipe=json.loads((OUT/'final_recipe.json').read_text())
    for component in recipe['components']:
        assert digest(component['name'])==component['sha256']
    archive_path=directory/'Assign01_Cai_Haochen_58561440.zip'
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==2
        assert sum(name.endswith('.pdf') for name in archive.namelist())==1
        assert archive.read(notebook_path.name)==notebook_path.read_bytes()
    secret=re.compile(rb'(?:github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,})')
    tracked=subprocess.check_output(['git','ls-files','-z']).split(b'\0')
    assert not [name.decode() for name in tracked if name and secret.search(Path(name.decode()).read_bytes())]
    history=subprocess.check_output(['git','log','--all','-p','--format=%H'])
    assert not secret.search(history), 'Credential pattern found in Git history.'
    print(f'Notebook cells executed; {figures} figures; ZIP valid; accuracy {metrics["accuracy"]:.2%}; macro F1 {f:.6f}; no GitHub credential pattern in tracked files/history.')
    print('Manual requirement: replace the typed Section A PDF with the student\'s genuine handwritten scan.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--submission',action='store_true')
    args=parser.parse_args()
    submission() if args.submission else checkpoints()
