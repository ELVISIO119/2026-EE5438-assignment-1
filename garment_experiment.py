"""Validation-only garment refinement and accuracy/compute selection; never loads test data."""
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from train import OUT, load_data
from evaluation import calibrate, digest, load_model, probabilities, probability_metrics


def details(probs, labels):
    labels=labels.cpu()
    pred=probs.argmax(1)
    report=classification_report(labels,pred,output_dict=True,zero_division=0)
    return dict(**probability_metrics(probs,labels),shirt_precision=report['6']['precision'],
                shirt_recall=report['6']['recall'],shirt_f1=report['6']['f1-score'],
                macro_f1=report['macro avg']['f1-score'],
                validation_half_correct=[int((pred[s]==labels[s]).sum()) for s in (slice(0,3000),slice(3000,None))])


def run(spatial=False):
    prefix='spatial' if spatial else 'garment'
    reference=json.loads((OUT/'final_recipe.json').read_text())
    x,y,vx,vy,_,_=load_data()
    del x,y
    vy=vy.cpu()
    cache={}; records={}; rows=[]
    def predict(name,views):
        key=(name,views)
        if key not in cache:
            model=load_model(name)
            cache[key]=probabilities(model,vx,views)
            del model
        return cache[key]
    def score(components):
        total=sum(c.get('weight',1.) for c in components)
        components=[dict(c,weight=c.get('weight',1.)/total) for c in components]
        probs=sum(c['weight']*calibrate(predict(c['name'],c['views']),c.get('temperature',1.)) for c in components)
        for c in components:
            if c['name'] not in records:
                records[c['name']]=json.loads((OUT/f"{c['name']}.json").read_text())
        row=dict(components=components,parameters=sum(records[n]['parameters'] for n in {c['name'] for c in components}),
                 macs=sum(records[c['name']]['macs']*c['views'] for c in components),**details(probs,vy))
        rows.append(row)
        return row,probs
    baseline,base_probs=score(reference['components'])
    assert baseline['correct']==reference['correct'], 'Frozen baseline validation changed.'
    print('Reference validation:',{k:v for k,v in baseline.items() if k!='components'},flush=True)
    (OUT/f'{prefix}_reference.json').write_text(json.dumps(dict(recipe=reference,validation=baseline,
        confusion_matrix=confusion_matrix(vy,base_probs.argmax(1)).tolist(),
        report=classification_report(vy,base_probs.argmax(1),output_dict=True,zero_division=0)),indent=2))
    # Exact class counts before any new training objective is allowed to affect selection.
    trials=[]
    stems=('spatial_wide','spatial_p2','spatial_pruned','spatial_pruned_garment') if spatial else ('garment_wide','garment_p2','focal_wide','focal_p2')
    for stem in stems:
        for suffix in ('','_ema','_swa'):
            name=stem+suffix
            assert json.loads((OUT/f'{name}.json').read_text())['complete']
            trials.append(name)
    for name in trials:
        for views in (1,10,18,30):
            probs=predict(name,views)
            temperature=min((.75,1.,1.25,1.5,2.),key=lambda t:probability_metrics(calibrate(probs,t),vy)['loss'])
            for alpha in (.1,.2,.35,.5,1.):
                components=[dict(c,weight=c.get('weight',1.)*(1-alpha)) for c in baseline['components']] if alpha<1 else []
                components.append(dict(name=name,views=views,temperature=temperature,weight=alpha))
                score(components)
        print('Garment validation views completed:',name,flush=True)
    # Finite deployment search: smaller view counts or drop members, fixed baseline weights/temperatures.
    for counts in itertools.product((0,1,2,4,10,18,30),repeat=len(baseline['components'])):
        components=[dict(c,views=v) for c,v in zip(baseline['components'],counts) if v]
        if components:
            score(components)
    valid=[row for row in rows if row['shirt_f1']>=baseline['shirt_f1'] and
           all(a>=b for a,b in zip(row['validation_half_correct'],baseline['validation_half_correct']))]
    chosen=min(valid,key=lambda r:(-r['correct'],r['macs'],r['parameters'],r['loss']))
    chosen=dict(chosen)
    for c in chosen['components']:
        c['sha256']=digest(c['name'])
    chosen.update(name=' + '.join(f"{c['weight']:.4f}*{c['name']}:{c['views']}:T{c.get('temperature',1.)}" for c in chosen['components']),
                  frozen_at=datetime.now(timezone.utc).isoformat(),seed=58561440,candidates=len(rows),
                  selection_split='validation_only',objective='accuracy_then_cost',
                  rule='Maximize validation correct count, then minimize MACs, parameters, NLL; neither validation half correct count nor Shirt F1 may fall below the frozen reference.',
                  benchmark_status='Exploratory continuation: garment hypothesis informed by previous public test errors; fitting/selection uses training/validation only.')
    (OUT/f'{prefix}_recipe.json').write_text(json.dumps(chosen,indent=2))
    pd.DataFrame([{**{k:v for k,v in r.items() if k!='components'},'components':json.dumps(r['components'])} for r in rows]).to_csv(OUT/f'{prefix}_candidates.csv',index=False)
    torch.save(cache,OUT/f'{prefix}_validation_cache.pt')
    print('Selected validation recipe:',json.dumps(chosen,indent=2),flush=True)
    with Path('JOURNAL.md').open('a') as journal:
        journal.write(f"\n## {chosen['frozen_at']} - {prefix.capitalize()} refinement and scoring-rule selection\n\n"
                      f"Validation: {baseline['correct']} -> {chosen['correct']}/6000 correct; Shirt F1 {baseline['shirt_f1']:.6f} -> {chosen['shirt_f1']:.6f}. "
                      f"Dense MACs/image {baseline['macs']:,} -> {chosen['macs']:,}. Searched {len(rows)} recorded recipes; "
                      "accuracy ranks first, with lower MACs/parameters breaking ties. Both validation halves and Shirt F1 were guarded. "
                      f"No new test evaluation was used for this selection. Evidence: `results/{prefix}_recipe.json`, `results/{prefix}_candidates.csv`.\n")


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--spatial',action='store_true')
    run(parser.parse_args().spatial)
