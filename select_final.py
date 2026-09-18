"""Select only on validation data, then write a hash-bound frozen test recipe."""
import itertools
import json
from datetime import datetime,timezone
import pandas as pd
import torch
from train import OUT,SEED,load_data
from evaluation import load_model,probabilities,probability_metrics,digest


def select():
    rows=[]
    for path in sorted(OUT.glob('*.json')):
        result=json.loads(path.read_text())
        if 'parameters' in result and 'val_accuracy' in result and result.get('complete'):
            rows.append(dict(name=path.stem,parameters=result['parameters'],macs=result['macs'],
                             val_accuracy=result['val_accuracy'],best_epoch=result.get('best_epoch'),
                             seconds=result.get('seconds'),shared_run=result.get('shared_training_run','')))
    for row in rows:
        row['pareto']=not any(other['parameters']<=row['parameters'] and other['macs']<=row['macs']
                              and other['val_accuracy']>=row['val_accuracy']
                              and (other['parameters']<row['parameters'] or other['macs']<row['macs']
                                   or other['val_accuracy']>row['val_accuracy']) for other in rows)
    pd.DataFrame(rows).to_csv(OUT/'pareto.csv',index=False)
    x,y,vx,vy,_,_=load_data()
    del x,y
    best_accuracy=max(row['val_accuracy'] for row in rows)
    eligible=[row for row in rows if row['val_accuracy']>=best_accuracy-.015 and not row['name'].endswith('_initial')]
    candidates=[]; cached={}
    for row in eligible:
        model=load_model(row['name'])
        for views in (1,2,4):
            probs=probabilities(model,vx,views)
            key=f"{row['name']}:{views}"
            cached[key]=probs
            candidates.append(dict(name=key,components=[dict(name=row['name'],views=views)],
                                   parameters=row['parameters'],macs=row['macs']*views,
                                   **probability_metrics(probs,vy)))
        del model
    rank=lambda row:(-row['correct'],row['macs'],row['parameters'],row['loss'])
    leaders=[]
    for candidate in sorted(candidates,key=rank):
        if candidate['components'][0]['name'] not in {c['components'][0]['name'] for c in leaders}:
            leaders.append(candidate)
        if len(leaders)==4:
            break
    for size in (2,3):
        for group in itertools.combinations(leaders,size):
            probs=torch.stack([cached[c['name']] for c in group]).mean(0)
            candidates.append(dict(name=' + '.join(c['name'] for c in group),
                                   components=[item for c in group for item in c['components']],
                                   parameters=sum(c['parameters'] for c in group),macs=sum(c['macs'] for c in group),
                                   **probability_metrics(probs,vy)))
    selected=min(candidates,key=rank)
    for component in selected['components']:
        component['sha256']=digest(component['name'])
    recipe=dict(selected,frozen_at=datetime.now(timezone.utc).isoformat(),seed=SEED,
                candidates=len(candidates),selection_split='validation_only',
                rule='Maximize validation correct count; tie-break by fewer total dense MACs, then parameters, then NLL.',
                search='Models within 1.5 percentage points of best raw validation; 1/2/4 views; equally weighted pairs/triples from top four distinct models.',
                view_definition='1: identity; 2: identity + horizontal flip; 4: those plus left/right one-pixel zero-padded shifts.',
                cost_note='Sum stored parameters across members; MACs include every model/view. Dense FLOPs approximately twice MACs.')
    (OUT/'final_recipe.json').write_text(json.dumps(recipe,indent=2))
    pd.DataFrame([{k:v for k,v in c.items() if k!='components'} for c in sorted(candidates,key=rank)]).to_csv(OUT/'validation_candidates.csv',index=False)
    print(json.dumps(recipe,indent=2))
    with open('JOURNAL.md','a') as journal:
        journal.write(f"\n## {recipe['frozen_at']} - Freeze final recipe\n\n"
                      f"Selected `{recipe['name']}` from {len(candidates)} validation-only candidates: "
                      f"{recipe['correct']}/{recipe['examples']} correct ({recipe['accuracy']:.2%}). "
                      f"Total stored parameters {recipe['parameters']:,}; dense MACs/image {recipe['macs']:,}, "
                      f"including all members and views. Checkpoint hashes and the selection rule are in `results/final_recipe.json`. "
                      f"No test labels were loaded by selection.\n")
    return recipe


if __name__=='__main__':
    select()
