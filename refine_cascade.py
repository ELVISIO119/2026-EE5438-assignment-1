"""Validation-only view-budget refinement of the two frozen hybrid cascades."""
import argparse
import json
from datetime import datetime, timezone
from itertools import product

import pandas as pd
import torch

from evaluation import calibrate, digest, early_mask, load_model, probabilities
from garment_experiment import details
from hybrid_experiment import benchmark, exit_mask, front_probabilities, model_names, predict
from train import DEVICE, OUT, SEED, load_data


def guarded(row, base, mac_ratio=.95):
    return (row['correct'] >= base['correct'] and row['shirt_f1'] >= base['shirt_f1']
            and all(a >= b for a, b in zip(row['validation_half_correct'], base['validation_half_correct']))
            and row['macs'] <= mac_ratio * base['macs'] and row['parameters'] <= base['parameters'])


@torch.inference_mode()
def run():
    references={g:json.loads((OUT/f'hybrid_{g}_recipe.json').read_text()) for g in ('balanced','accuracy')}
    x,y,vx,vy,_,_=load_data(); del x,y
    models={n:load_model(n) for r in references.values() for n in model_names(r)}
    records={n:json.loads((OUT/f'{n}.json').read_text()) for n in models}
    components=references['balanced']['back']['components']
    gate=references['balanced']['back']['cascade']
    cheap=probabilities(models[gate['name']],vx,1)
    gate_mask=early_mask(cheap,gate)
    cache={}
    for component,view_counts in zip(components,((2,4,10),(4,10,18,30),(2,4,10))):
        for views in view_counts:
            name=component['name']
            cache[name,views]=calibrate(probabilities(models[name],vx,views),component['temperature'])
            print('Cached validation probabilities:',name,views,flush=True)
    rows=[]
    for goal,base in references.items():
        p,agreement=front_probabilities(base['front'],vx,models)
        for views in product((2,4,10),(4,10,18,30),(2,4,10)):
            members=[dict(c,views=v) for c,v in zip(components,views)]
            cost=sum(records[c['name']]['macs']*c['views'] for c in members)
            back=dict(components=members,cascade=gate,full_macs=cost,macs=cost,
                      worst_case_macs=cost+gate['macs'])
            weights=torch.tensor([c['weight'] for c in members]); weights/=weights.sum()
            bp=sum(w*cache[c['name'],c['views']] for w,c in zip(weights,members))
            bp[gate_mask]=cheap[gate_mask]
            per_image_cost=gate['macs']+(~gate_mask).double()*cost
            for threshold in (.875,.9,.925,.95,.975):
                recipe=dict(front=base['front'],back=back,threshold=threshold,classes=base['classes'],
                            parameters=base['parameters'],goal=goal,label=f'{goal}:{views}:{threshold}')
                mask=exit_mask(p,agreement,recipe)
                output=bp.clone(); output[mask]=p[mask]
                row=dict(recipe,**details(output,vy),
                         macs=float((base['front']['macs']+(~mask)*per_image_cost).mean()),candidate_id=len(rows))
                rows.append(row)
    select_candidates(rows,references,models,vx,vy,'refine',.95,
        'No fitting or test access. Preserve the two legacy front architectures and fallback gate/weights/temperatures. Wide and pruned views 2/4/10; fine-patch views 4/10/18/30; front thresholds .875/.9/.925/.95/.975. All actual work is charged; no cache reuse discount.')


@torch.inference_mode()
def select_candidates(rows,references,models,vx,vy,prefix,mac_ratio,protocol):
    pd.DataFrame([{k:r[k] for k in ('candidate_id','label','goal','correct','accuracy','shirt_f1',
                                   'validation_half_correct','macs','parameters')} for r in rows]).to_csv(OUT/f'{prefix}_candidates.csv',index=False)
    actual=[]; selected={}; reference_timings={}
    for goal,base in references.items():
        reference_timings[goal]=[benchmark(base,part,models) for part in (vx[:1024],vx[-1024:])]
        short=sorted([r for r in rows if r['goal']==goal and guarded(r,base,mac_ratio)],
                     key=lambda r:(-r['correct'],r['macs'],r['loss']))[:5]
        qualified=[]
        for row in short:
            p,execution=predict(row,vx,models)
            checked=dict(row,**details(p,vy),macs=execution['macs_per_image'],
                         worst_case_macs=execution['worst_case_macs'],validation_execution=execution)
            checked['timings']=[benchmark(checked,part,models) for part in (vx[:1024],vx[-1024:])]
            checked['passes']=guarded(checked,base,mac_ratio) and all(
                t['median_ms']<=.95*b['median_ms'] for t,b in zip(checked['timings'],reference_timings[goal]))
            actual.append(checked)
            print('Actual finalist:',checked['label'],checked['correct'],round(checked['macs']),
                  [round(t['median_ms'],2) for t in checked['timings']],checked['passes'],flush=True)
            if checked['passes']: qualified.append(checked)
        if qualified:
            chosen=min(qualified,key=lambda r:(-r['correct'],r['macs'],r['timings'][0]['median_ms'],r['loss']))
            selected[goal]=dict(chosen,checkpoint_hashes={n:digest(n) for n in sorted(model_names(chosen))},
                                selection_split='validation_only',seed=SEED,frozen_at=datetime.now(timezone.utc).isoformat(),
                                reference_latency=reference_timings[goal],rule=f'Preserve reference correct count, Shirt F1 and both development halves; average MAC ratio <= {mac_ratio}, no parameter increase, >=5% lower median latency on BOTH first/last 1024 validation images. Recompute cached top five per goal with actual routing. Rank correct, MACs, first-slice latency, NLL.')
            (OUT/f'{prefix}_{goal}_recipe.json').write_text(json.dumps(selected[goal],indent=2))
    result=dict(complete=True,candidate_count=len(rows),references=references,reference_timings=reference_timings,
                selected=selected,actual_finalists=actual,completed_at=datetime.now(timezone.utc).isoformat(),
                environment=dict(torch=torch.__version__,device=torch.cuda.get_device_name() if DEVICE.type=='cuda' else 'CPU'),
                protocol=protocol+' Finite top-five prefilter, not exhaustive real-routing optimization. Validation and public test have been repeatedly reused throughout this project.')
    (OUT/f'{prefix}_experiment.json').write_text(json.dumps(result,indent=2))
    print('Frozen selections:',{g:(r['correct'],r['macs']) for g,r in selected.items()},flush=True)


@torch.inference_mode()
def progressive():
    references={g:json.loads((OUT/f'hybrid_{g}_recipe.json').read_text()) for g in ('balanced','accuracy')}
    x,y,vx,vy,_,_=load_data(); del x,y
    models={n:load_model(n) for r in references.values() for n in model_names(r)}
    cache={n:probabilities(models[n],vx,1) for n in ('legacy_clean','legacy_muon')}
    rows=[]
    for goal,base in references.items():
        old,_=predict(base,vx,models)
        p,agree=front_probabilities(base['front'],vx,models)
        front_mask=exit_mask(p,agree,base)
        back=base['back']; gate=back['cascade']
        gate_p=probabilities(models[gate['name']],vx,1)
        cost=base['front']['macs']+(~front_mask).double()*(gate['macs']+(~early_mask(gate_p,gate)).double()*back['full_macs'])
        names=('legacy_clean','legacy_muon') if goal=='balanced' else ('legacy_muon',)
        for name,threshold,classes in product(names,(.975,.99,.995,.999),(list(range(10)),[1,5,7,8,9])):
            gate=dict(name=name,views=1,macs=29003008,threshold=threshold,classes=classes)
            mask=early_mask(cache[name],gate)
            probs=old.clone(); probs[mask]=cache[name][mask]
            charged=cost.clone(); charged[mask]=gate['macs']
            recipe={k:base[k] for k in ('front','back','threshold','classes','parameters')}
            rows.append(dict(recipe,pre_exit=gate,goal=goal,label=f'{goal}:{name}:{threshold}:{len(classes)}',
                             **details(probs,vy),macs=float(charged.mean()),candidate_id=len(rows)))
    select_candidates(rows,references,models,vx,vy,'progressive',.99,
        'No fitting or test access. First-view confidence .975/.99/.995/.999, allowed predicted classes all or 1/5/7/8/9. Balanced first model clean or Muon; accuracy first model Muon, then flip only if needed. Reuse first probabilities in unresolved two-pass mean; do not recompute them. Preserve original outer thresholds, weights and fallback.')


@torch.inference_mode()
def batched():
    evidence=json.loads((OUT/'progressive_experiment.json').read_text())
    references=evidence['references']
    x,y,vx,vy,_,_=load_data(); del x,y
    models={n:load_model(n) for r in references.values() for n in model_names(r)}
    rows=[]
    # Only already evaluated progressive finalists; no new confidence/architecture search.
    for base in evidence['actual_finalists']:
        if not guarded(base,references[base['goal']],.99): continue
        for view_batch in (2,4):
            recipe={k:base[k] for k in ('front','back','threshold','classes','parameters','goal','pre_exit')}
            recipe['back']=dict(recipe['back'],view_batch=view_batch)
            recipe['label']=base['label']+f':view_batch={view_batch}'
            p,execution=predict(recipe,vx,models)
            row=dict(recipe,**details(p,vy),macs=execution['macs_per_image'],candidate_id=len(rows))
            rows.append(row)
            print('Grouped-view validation:',row['label'],row['correct'],flush=True)
    select_candidates(rows,references,models,vx,vy,'batched',.99,
        'Only accuracy/compute-qualified actual progressive finalists, fallback views grouped 2 or 4 in each forward. No new weights, thresholds or view counts. All candidates evaluated on actual routed validation. Grouping preserves mathematical MACs and can change BF16 rounding, which is checked before selection. Require the progressive accuracy/compute/latency guards. View grouping increases activation memory, not stored parameters.')


def evaluate(prefix='refine'):
    """Only the validation-qualified, already frozen endpoints receive test evaluation."""
    from hybrid_experiment import evaluate_frozen
    evidence=json.loads((OUT/f'{prefix}_experiment.json').read_text())
    if not evidence['selected']:
        print('No candidate qualified; preserve existing hybrids without new test evaluation.')
        return
    return evaluate_frozen(prefix=prefix,goals=tuple(evidence['selected']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--test',action='store_true'); parser.add_argument('--progressive',action='store_true'); parser.add_argument('--batched',action='store_true')
    args=parser.parse_args()
    evaluate('batched' if args.batched else 'progressive' if args.progressive else 'refine') if args.test else batched() if args.batched else progressive() if args.progressive else run()
