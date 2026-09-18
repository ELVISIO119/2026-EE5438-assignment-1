"""Select only on validation data, then write a hash-bound frozen test recipe."""
import itertools
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import torch
from train import OUT,SEED,load_data
from evaluation import load_model,probabilities,probability_metrics,digest,calibrate


def write_pareto(exclude_prefixes=()):
    rows=[]
    for path in sorted(OUT.glob('*.json')):
        if path.stem.startswith(exclude_prefixes):
            continue
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
    return rows


def select(accuracy_first=False,exclude_prefixes=()):
    if accuracy_first:
        for path in Path('configs').glob('accuracy_*.json'):
            cfg=json.loads(path.read_text())
            result=json.loads((OUT/f"{cfg['name']}.json").read_text())
            assert result.get('complete'), f"Finish {cfg['name']} before accuracy-first selection."
    rows=write_pareto(exclude_prefixes)
    x,y,vx,vy,_,_=load_data()
    del x,y
    best_accuracy=max(row['val_accuracy'] for row in rows)
    margin=.035 if accuracy_first else .015
    eligible=[row for row in rows if row['val_accuracy']>=best_accuracy-margin and not row['name'].endswith('_initial')]
    candidates=[]; cached={}
    cache_dir=OUT/'validation_cache'
    cache_dir.mkdir(exist_ok=True)
    code_hash=hashlib.sha256(b''.join(Path(name).read_bytes() for name in ('models.py','train.py','evaluation.py'))).hexdigest()
    for row in eligible:
        model=load_model(row['name'])
        checkpoint_hash=digest(row['name'])
        for views in ((1,2,4,10,18,30,50) if accuracy_first else (1,2,4)):
            cache_path=cache_dir/f"{row['name']}_{views}_{checkpoint_hash[:12]}_{code_hash[:12]}.pt"
            if cache_path.exists():
                probs=torch.load(cache_path,map_location='cpu',weights_only=True)
                assert probs.shape==(len(vy),10) and torch.isfinite(probs).all()
            else:
                probs=probabilities(model,vx,views)
                torch.save(probs,cache_path)
            key=f"{row['name']}:{views}"
            cached[key]=probs
            candidates.append(dict(name=key,components=[dict(name=row['name'],views=views)],
                                   parameters=row['parameters'],macs=row['macs']*views,
                                   **probability_metrics(probs,vy)))
        del model
        print('Validation views completed:',row['name'],flush=True)
    rank=(lambda row:(-row['correct'],row['loss'])) if accuracy_first else (lambda row:(-row['correct'],row['macs'],row['parameters'],row['loss']))
    leaders=[]
    for candidate in sorted(candidates,key=rank):
        if candidate['components'][0]['name'] not in {c['components'][0]['name'] for c in leaders}:
            leaders.append(candidate)
        if not accuracy_first and len(leaders)==4:
            break
    for size in (2,3):
        for group in itertools.combinations(leaders[:10],size):
            probs=torch.stack([cached[c['name']] for c in group]).mean(0)
            candidates.append(dict(name=' + '.join(c['name'] for c in group),
                                   components=[item for c in group for item in c['components']],
                                   parameters=sum(c['parameters'] for c in group),macs=sum(c['macs'] for c in group),
                                   **probability_metrics(probs,vy)))
    if accuracy_first:
        # Bounded validation search: equal-weight prefixes plus at most eight greedy additions.
        # Both fixed validation halves must avoid losing correct predictions on an accepted step.
        calibrated=[]
        for candidate in leaders:
            temperature=min((.75,1.,1.25,1.5,2.),key=lambda t:probability_metrics(calibrate(cached[candidate['name']],t),vy)['loss'])
            key=candidate['name']+f':T{temperature}'
            cached[key]=calibrate(cached[candidate['name']],temperature)
            calibrated.append(dict(candidate,name=key,
                                   components=[dict(candidate['components'][0],temperature=temperature)],
                                   **probability_metrics(cached[key],vy)))
        candidates.extend(calibrated)
        candidates.extend(accuracy_ensembles(calibrated,cached,vy,rank))
    selected=min(candidates,key=rank)
    for component in selected['components']:
        component['sha256']=digest(component['name'])
    recipe=dict(selected,frozen_at=datetime.now(timezone.utc).isoformat(),seed=SEED,
                candidates=len(candidates),selection_split='validation_only',
                objective='accuracy_first' if accuracy_first else 'accuracy_with_cost_ties',
                rule=('Maximize validation correct count; tie-break only by validation NLL; no compute penalty.' if accuracy_first else 'Maximize validation correct count; tie-break by fewer total dense MACs, then parameters, then NLL.'),
                search=('Models within 3.5 percentage points of best raw validation; 1/2/4/10/18/30/50 views; raw equal pairs/triples from top ten distinct checkpoints; calibrated prefixes and bounded greedy combinations can use every eligible checkpoint.' if accuracy_first else 'Models within 1.5 percentage points of best raw validation; 1/2/4 views; equally weighted pairs/triples from top four distinct models.'),
                view_definition='1: identity; 2: identity + horizontal flip; 4: those plus left/right one-pixel shifts; 10: center/cardinal shifts with flips; 18: all 3x3 offsets with flips; 30: ten views at affine-grid scales 1/0.96/1.04; 50: all 5x5 offsets with flips. Shifts are zero-padded.',
                benchmark_status='Exploratory continuation on a previously evaluated public benchmark; selection code uses validation only, not a new blinded test.',
                calibration='Greedy/prefix pool selects per-model post-view probability temperature from [0.75,1,1.25,1.5,2] by validation NLL. Raw individual/pair/triple candidates remain eligible.' if accuracy_first else 'None',
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


def accuracy_ensembles(leaders,cached,labels,rank):
    pool={candidate['name']:candidate for candidate in leaders}
    def combine(weights):
        weights={key:value/sum(weights.values()) for key,value in weights.items() if value>1e-8}
        probs=sum(cached[key]*weight for key,weight in weights.items())
        components=[dict(pool[key]['components'][0],weight=weight) for key,weight in weights.items()]
        row=dict(name=' + '.join(f'{weight:.4f}*{key}' for key,weight in weights.items()),
                 components=components,parameters=sum(pool[key]['parameters'] for key in weights),
                 macs=sum(pool[key]['macs'] for key in weights),**probability_metrics(probs,labels))
        row['validation_half_correct']=[probability_metrics(probs[part],labels[part])['correct'] for part in (slice(0,3000),slice(3000,None))]
        return row
    outcomes=[]
    for size in range(2,len(leaders)+1):
        outcomes.append(combine({candidate['name']:1/size for candidate in leaders[:size]}))
    trials=0
    # Three starts reduce dependence on the standalone winner; constants are fixed before test access.
    for start in leaders[:3]:
        weights={start['name']:1.}
        current=combine(weights)
        for _ in range(8):
            proposals=[]
            for candidate in leaders:
                for alpha in (.1,.2,.35,.5):
                    proposed={key:value*(1-alpha) for key,value in weights.items()}
                    proposed[candidate['name']]=proposed.get(candidate['name'],0)+alpha
                    row=combine(proposed)
                    trials+=1
                    if all(a>=b for a,b in zip(row['validation_half_correct'],current['validation_half_correct'])):
                        proposals.append((row,proposed))
            if not proposals:
                break
            better,proposed=min(proposals,key=lambda pair:rank(pair[0]))
            if rank(better)>=rank(current):
                break
            current,weights=better,proposed
            outcomes.append(current)
    (OUT/'weighted_search.json').write_text(json.dumps(dict(
        attempted_weight_settings=trials,retained_candidates=len(outcomes),
        temperature_grid=[.75,1.,1.25,1.5,2.],temperature_selection='Minimum validation NLL per model after averaging views',
        constraint='Accepted greedy steps cannot reduce correct count on either fixed 3000-example validation half; halves are development data, not independent test sets.',
        alpha_grid=[.1,.2,.35,.5],max_steps_per_start=8,starts=min(3,len(leaders))),indent=2))
    return outcomes


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--accuracy-first',action='store_true')
    args=parser.parse_args()
    select(accuracy_first=args.accuracy_first)
