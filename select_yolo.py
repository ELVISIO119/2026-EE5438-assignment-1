"""Bounded validation-only comparison of detector-inspired pure MLP experiments."""
import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import torch

from evaluation import calibrate, digest, early_mask, load_model, probabilities, recipe_probabilities
from garment_experiment import details
from train import OUT, SEED, load_data


@torch.inference_mode()
def run(stems,prefix='yolo'):
    assert prefix in ('yolo','features')
    reference_path=OUT/f'{prefix}_reference.json'
    if not reference_path.exists():
        reference_path.write_text((OUT/'final_recipe.json').read_text())
    reference=json.loads(reference_path.read_text())
    x,y,vx,vy,_,_=load_data()
    del x,y
    gate=reference['cascade']
    cheap=probabilities(load_model(gate['name']),vx,gate['views'])
    mask=early_mask(cheap,gate)
    hard=vx[(~mask).to(vx.device)]
    records={}; cache={}; rows=[]; summary=[]
    def record(name):
        if name not in records:
            records[name]=json.loads((OUT/f'{name}.json').read_text())
            assert records[name]['complete']
        return records[name]
    def predict(name,views,routed):
        key=(name,views,routed)
        if key not in cache:
            cache[key]=probabilities(load_model(name),hard if routed else vx,views)
        return cache[key]
    def score(components,routed=True):
        total=sum(c.get('weight',1.) for c in components)
        components=[dict(c,weight=c.get('weight',1.)/total) for c in components]
        p=sum(c['weight']*calibrate(predict(c['name'],c['views'],routed),c.get('temperature',1.)) for c in components)
        full_macs=sum(record(c['name'])['macs']*c['views'] for c in components)
        names={c['name'] for c in components}
        if routed:
            assert gate['name'] in names, 'Gate weights must be shared with an ensemble member.'
            mixed=cheap.clone(); mixed[~mask]=p; p=mixed
        macs=gate['macs']+len(hard)/len(vx)*full_macs if routed else full_macs
        row=dict(components=components,parameters=sum(record(n)['parameters'] for n in names),
                 macs=macs,full_macs=full_macs,worst_case_macs=full_macs+(gate['macs'] if routed else 0),
                 **details(p,vy))
        if routed:
            row['cascade']=gate
        rows.append(row)
        return row
    base=score(reference['components'])
    assert base['correct']==reference['correct'] and base['shirt_f1']==reference['shirt_f1']
    # Choose one raw-validation checkpoint per architecture family before the
    # fixed view/combination grid. Ordinary/EMA/SWA and clean refinement qualify.
    for stem in stems:
        names=[stem+stage+average for stage in ('','_clean') for average in ('','_ema','_swa')
               if (OUT/f'{stem+stage+average}.json').exists()]
        assert names
        best=min(names,key=lambda n:(-record(n)['val_accuracy'],record(n)['val_loss']))
        single=score([dict(name=best,views=1,temperature=1.,weight=1.)],routed=False)
        assert abs(single['accuracy']-record(best)['val_accuracy'])<1e-6
        summary.append(dict(family=stem,selected_checkpoint=best,available_checkpoints=names,**{k:single[k] for k in ('accuracy','correct','shirt_recall','shirt_f1','parameters','macs')}))
        for views in (1,10,30):
            p=predict(best,views,False)
            temperature=min((.75,1.,1.25,1.5,2.),key=lambda t:details(calibrate(p,t),vy)['loss'])
            candidate=dict(name=best,views=views,temperature=temperature,weight=1.)
            score([candidate],routed=False)
            for alpha in (.1,.2,.35):
                components=[dict(c,weight=c['weight']*(1-alpha)) for c in reference['components']]
                score(components+[dict(candidate,weight=alpha)])
            for i,c in enumerate(reference['components']):
                if c['name']!=gate['name']:
                    components=[dict(member) for member in reference['components']]
                    components[i]=dict(candidate,weight=c['weight'])
                    score(components)
        print('Completed validation grid:',best,single['accuracy'],flush=True)
    eligible=[r for r in rows if r['shirt_f1']>=base['shirt_f1'] and
              all(a>=b for a,b in zip(r['validation_half_correct'],base['validation_half_correct']))]
    chosen=dict(min(eligible,key=lambda r:(-r['correct'],r['macs'],r['parameters'],r['loss'])))
    for c in chosen['components']:
        c['sha256']=digest(c['name'])
    actual,execution=recipe_probabilities(chosen,vx)
    measured=details(actual,vy)
    assert measured['correct']==chosen['correct'] and measured['shirt_f1']==chosen['shirt_f1']
    assert abs(execution['macs_per_image']-chosen['macs'])<1e-5
    chosen.update(**measured,name=' + '.join(f"{c['weight']:.4f}*{c['name']}:{c['views']}:T{c.get('temperature',1.)}" for c in chosen['components'])+('; frozen confidence gate' if chosen.get('cascade') else ''),
                  frozen_at=datetime.now(timezone.utc).isoformat(),seed=SEED,candidates=len(rows),
                  selection_split='validation_only',objective='accuracy_then_expected_cost',
                  rule='Guard both validation half counts and Shirt F1; rank correct count, average MACs, parameters, NLL. Fixed gate; one best raw checkpoint per family, views 1/10/30, additions 0.1/0.2/0.35 or replacement of either non-gate member.',
                  benchmark_status='Exploratory continuation on a previously observed public test benchmark.',
                  validation_execution=execution)
    (OUT/f'{prefix}_recipe.json').write_text(json.dumps(chosen,indent=2))
    (OUT/f'{prefix}_summary.json').write_text(json.dumps(dict(complete=True,families=summary,reference_correct=base['correct'],selected_correct=chosen['correct'],candidate_count=len(rows),selected_components=chosen['components'],frozen_at=chosen['frozen_at']),indent=2))
    pd.DataFrame([{**{k:v for k,v in r.items() if k not in ('components','cascade')},'components':json.dumps(r['components'])} for r in rows]).to_csv(OUT/f'{prefix}_candidates.csv',index=False)
    torch.save(actual,OUT/f'{prefix}_validation.pt')
    print(json.dumps(chosen,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stems',nargs='+',default=['yolo_pyramid_control','yolo_pyramid_fusion','yolo_pyramid_aux','yolo_pyramid_csp'])
    run(parser.parse_args().stems)
