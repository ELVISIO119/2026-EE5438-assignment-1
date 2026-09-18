"""Validation-only accuracy/compute frontier using earlier and current frozen MLPs."""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from evaluation import digest, early_mask, load_model, probabilities, recipe_probabilities
from garment_experiment import details
from train import DEVICE, OUT, SEED, load_data


def model_names(recipe):
    if 'front' in recipe:
        return model_names(recipe['front']) | model_names(recipe['back'])
    return {c['name'] for c in recipe['components']} | ({recipe['cascade']['name']} if recipe.get('cascade') else set())


def exit_mask(p,agreement,recipe):
    confidence,predicted=p.max(1)
    return (confidence>=recipe['threshold']) & agreement & torch.isin(predicted,torch.tensor(recipe['classes']))


@torch.inference_mode()
def front_probabilities(front,x,models):
    individual=[probabilities(models[c['name']],x,c['views']) for c in front['components']]
    p=sum(individual)/len(individual)
    predictions=torch.stack([v.argmax(1) for v in individual])
    return p,(predictions==predictions[0]).all(0)


@torch.inference_mode()
def predict(recipe,x,models=None):
    models={name:load_model(name) for name in model_names(recipe)} if models is None else models
    if 'front' not in recipe:
        return recipe_probabilities(recipe,x,models)
    p,agree=front_probabilities(recipe['front'],x,models)
    mask=exit_mask(p,agree,recipe)
    count=int((~mask).sum())
    back_cost=0.; nested=None
    if count:
        difficult, nested=predict(recipe['back'],x[(~mask).to(x.device)],models)
        p[~mask]=difficult
        back_cost=nested['macs_per_image']
    return p,dict(examples=len(x),early_exit=int(mask.sum()),full_ensemble=count,
                  macs_per_image=recipe['front']['macs']+count/len(x)*back_cost,
                  worst_case_macs=recipe['front']['macs']+recipe['back'].get('worst_case_macs',recipe['back']['macs']),
                  fallback_execution=nested)


@torch.inference_mode()
def self_check():
    class Fake(torch.nn.Module):
        def __init__(self,gate):
            super().__init__(); self.gate=gate; self.rows=0
        def forward(self,x):
            self.rows+=len(x)
            out=torch.zeros(len(x),10)
            out[:,0 if self.gate else 1]=20*x[:,0,0,0] if self.gate else 20
            return out
    x=torch.zeros(3,1,28,28); x[0]=1
    front=dict(components=[dict(name='gate',views=1)],macs=10)
    back=dict(components=[dict(name='back',views=1)],macs=20)
    for threshold,expected in ((0,0),(1,2),(.9,2)):
        models=dict(gate=Fake(True),back=Fake(False))
        recipe=dict(front=front,back=back,threshold=threshold,classes=list(range(10)))
        p,execution=predict(recipe,x,models)
        assert models['gate'].rows==3 and models['back'].rows==expected
        assert abs(execution['macs_per_image']-(10+expected/3*20))<1e-7
        assert torch.allclose(p.sum(1),torch.ones(3))
        assert p[0].argmax()==0 and (not expected or (p[1:].argmax(1)==1).all())
    p=torch.tensor([[.95,.05],[.05,.95]])
    assert exit_mask(p,torch.tensor([True,False]),dict(threshold=.9,classes=[0,1])).tolist()==[True,False]
    print('Actual skipped rows, all-exit/partial routing, disagreement rejection and MAC accounting: PASS',flush=True)


@torch.inference_mode()
def benchmark(recipe,x,models):
    def infer():
        output=predict(recipe,x,models)
        if x.is_cuda: torch.cuda.synchronize()
        return output
    for _ in range(3): infer()
    times=[]
    for _ in range(7):
        start=time.perf_counter(); infer(); times.append((time.perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(times),samples_ms=times,examples=len(x),batch_size=128)


@torch.inference_mode()
def run():
    self_check()
    source=OUT/'hybrid_reference.json'
    if not source.exists(): source.write_text((OUT/'final_recipe.json').read_text())
    reference=json.loads(source.read_text())
    x,y,vx,vy,_,_=load_data(); del x,y
    records={name:json.loads((OUT/f'{name}.json').read_text()) for name in
             ['legacy_clean','legacy_muon']+[c['name'] for c in reference['components']]}
    models={name:load_model(name) for name in records}
    def plain(components,label):
        return dict(components=components,label=label,
                    parameters=sum(records[n]['parameters'] for n in {c['name'] for c in components}),
                    macs=sum(records[c['name']]['macs']*c['views'] for c in components))
    legacy=plain([dict(name=n,views=10,weight=.5,temperature=1.) for n in ('legacy_clean','legacy_muon')],'legacy_dual_10')
    reduced=plain([dict(c,views=v) for c,v in zip(reference['components'],(2,1,4))],'current_reduced_2_1_4')
    reference=dict(reference,label='current_cascade')
    rows=[]; cache={}; back_cache={}
    def record(recipe,p,cost):
        row=dict(recipe,**details(p,vy),macs=float(cost.mean()),
                 parameters=sum(records[n]['parameters'] for n in model_names(recipe)),
                 candidate_id=len(rows))
        rows.append(row); return row
    for base in (legacy,reference,reduced):
        p,execution=predict(base,vx,models)
        if base.get('cascade'):
            gate=base['cascade']; cheap=probabilities(models[gate['name']],vx,gate['views'])
            cost=gate['macs']+(~early_mask(cheap,gate)).double()*base['full_macs']
        else: cost=torch.full((len(vx),),base['macs'],dtype=torch.float64)
        assert abs(float(cost.mean())-execution['macs_per_image'])<1e-5
        back_cache[base['label']]=(p,cost)
        record(base,p,cost)
    old_base,new_base=rows[:2]
    assert old_base['correct']==5695 and new_base['correct']==reference['correct']
    for group in (['legacy_clean'],['legacy_muon'],['legacy_clean','legacy_muon']):
        for views in (1,2,10):
            front=plain([dict(name=n,views=views,weight=1/len(group),temperature=1.) for n in group],'+'.join(group)+f':{views}')
            individual=[]
            for name in group:
                key=(name,views)
                if key not in cache: cache[key]=probabilities(models[name],vx,views)
                individual.append(cache[key])
            p=sum(individual)/len(individual)
            pred=torch.stack([v.argmax(1) for v in individual]); agree=(pred==pred[0]).all(0)
            record(front,p,torch.full((len(vx),),front['macs'],dtype=torch.float64))
            for back in (legacy,reference,reduced):
                bp,bcost=back_cache[back['label']]
                for threshold in (.8,.9,.95,.975,.99,.995):
                    for classes in (list(range(10)),[1,5,7,8,9]):
                        recipe=dict(front=front,back=back,threshold=threshold,classes=classes,
                                    label=f"{front['label']}->{back['label']}:{threshold}:{len(classes)}")
                        mask=exit_mask(p,agree,recipe)
                        mixed=bp.clone(); mixed[mask]=p[mask]
                        record(recipe,mixed,front['macs']+(~mask)*bcost)
            if views in (1,2):
                for alpha in (.25,.5,.75):
                    blend=plain([dict(c,weight=c['weight']*alpha) for c in front['components']]+
                                [dict(c,weight=c['weight']*(1-alpha)) for c in reduced['components']],
                                f"blend:{front['label']}:{alpha}")
                    # Reuse current member temperatures exactly as recorded in the reduced recipe.
                    bp,_=back_cache[reduced['label']]
                    record(blend,alpha*p+(1-alpha)*bp,torch.full((len(vx),),blend['macs'],dtype=torch.float64))
    pd.DataFrame([dict(candidate_id=r['candidate_id'],label=r['label'],**{k:r[k] for k in
        ('accuracy','correct','shirt_recall','shirt_f1','macs','parameters','validation_half_correct')}) for r in rows]).to_csv(OUT/'hybrid_candidates.csv',index=False)
    def eligible(row,base,budget):
        return row['correct']>=base['correct'] and row['macs']<=budget and row['shirt_f1']>=base['shirt_f1'] and all(a>=b for a,b in zip(row['validation_half_correct'],base['validation_half_correct']))
    def rank(row): return (-row['correct'],row['macs'],row['parameters'],row['loss'])
    finalists={}; selected={}
    for goal,base,budget in (('balanced',old_base,legacy['macs']),('accuracy',new_base,new_base['macs'])):
        shortlist=sorted([r for r in rows if eligible(r,base,budget)],key=rank)[:5]
        shortlist.append(base)
        actual=[]
        for row in shortlist:
            key=row['candidate_id']
            if key not in finalists:
                p,execution=predict(row,vx,models)
                finalists[key]=dict(row,**details(p,vy),macs=execution['macs_per_image'],
                                    worst_case_macs=execution['worst_case_macs'],validation_execution=execution)
            checked=finalists[key]
            if eligible(checked,base,budget): actual.append(checked)
        chosen=dict(min(actual,key=rank),frozen_at=datetime.now(timezone.utc).isoformat(),seed=SEED,
                    checkpoint_hashes={n:digest(n) for n in sorted(model_names(min(actual,key=rank)))},
                    selection_split='validation_only',goal=goal,
                    rule='Bounded cached screen, recompute top five eligible candidates plus reference with actual subset routing. Guard accuracy, Shirt F1, both reused validation halves and average MAC budget; maximize correct, then minimize average MACs, parameters, NLL.')
        selected[goal]=chosen
        (OUT/f'hybrid_{goal}_recipe.json').write_text(json.dumps(chosen,indent=2))
    timings={label:benchmark(r,vx[:1024],models) for label,r in
             dict(legacy=legacy,current=reference,reduced=reduced,**selected).items()}
    result=dict(complete=True,candidate_count=len(rows),references=dict(legacy=old_base,current=new_base,reduced=rows[2]),
                selected=selected,actual_finalists=list(finalists.values()),timings=timings,
                environment=dict(torch=torch.__version__,device=torch.cuda.get_device_name() if DEVICE.type=='cuda' else 'CPU'),
                protocol='Inference only, no new training. Equal-weight legacy front; multiple front models must agree before exit. Thresholds .8/.9/.95/.975/.99/.995; allowed predicted classes all or 1/5/7/8/9. Legacy single/dual fronts at 1/2/10 views. Backends: legacy 10 views, original cascade, reduced current 2/1/4 views. Also fixed .25/.5/.75 blends of 1/2-view fronts with reduced current. All stored models count once; reused images/views are recomputed and charged. No test labels loaded.',
                limitations='Previously observed public benchmark and repeatedly reused validation. Top-five cached prefilter is bounded, not a proof of global optimality. Sequential shared-GPU latency includes routing, views, softmax, aggregation and CPU output; excludes loading and input transfer; three warmups/seven repeats, 1024 validation inputs. Worst-case compute can exceed either baseline.',
                completed_at=datetime.now(timezone.utc).isoformat())
    (OUT/'hybrid_experiment.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(candidate_count=len(rows),selected={k:{f:r[f] for f in ('label','correct','macs','parameters','shirt_f1')} for k,r in selected.items()},timings=timings),indent=2),flush=True)


@torch.inference_mode()
def choose_fast():
    """Amended before test access: require >=5% measured latency reduction too."""
    evidence=json.loads((OUT/'hybrid_compute_screen.json').read_text())
    x,y,vx,vy,_,_=load_data(); del x,y,vy
    candidates=evidence['actual_finalists']
    models={n:load_model(n) for r in candidates for n in model_names(r)}
    timings={r['label']:benchmark(r,vx[:1024],models) for r in candidates}
    chosen={}
    for goal,base in (('balanced',evidence['references']['legacy']),('accuracy',evidence['references']['current'])):
        eligible=[r for r in candidates if r['correct']>=base['correct'] and r['shirt_f1']>=base['shirt_f1'] and
                  all(a>=b for a,b in zip(r['validation_half_correct'],base['validation_half_correct'])) and
                  r['macs']<base['macs'] and timings[r['label']]['median_ms']<=.95*timings[base['label']]['median_ms']]
        assert eligible, f'No simultaneous improvement met {goal} limits; keep existing references.'
        selected=min(eligible,key=lambda r:(-r['correct'],timings[r['label']]['median_ms'],r['macs'],r['parameters'],r['loss']))
        chosen[goal]=dict(selected,goal=goal,selection_split='validation_only',seed=SEED,
                          frozen_at=datetime.now(timezone.utc).isoformat(),
                          checkpoint_hashes={n:digest(n) for n in sorted(model_names(selected))},
                          rule='Latency amendment before any new test access. Among the previously recomputed finite shortlist, guard correct count, Shirt F1 and both development halves; require lower average MACs and >=5% faster observed median latency than corresponding reference. Rank correct count, latency, MACs, total parameters, NLL.',
                          measured_latency=timings[selected['label']],reference_latency=timings[base['label']])
        (OUT/f'hybrid_{goal}_recipe.json').write_text(json.dumps(chosen[goal],indent=2))
    evidence.update(selected=chosen,latency_finalists=timings,
                    latency_amendment='Initial MAC-only selections were 1-2% slower than their references. Before any new test access, require at least 5% lower observed median latency as well as lower average MACs on the same validation sample. Archived compute-screen choices remain available. The amendment reuses validation and is not independent confirmation.',
                    frozen_at=datetime.now(timezone.utc).isoformat())
    (OUT/'hybrid_experiment.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({k:{f:r[f] for f in ('label','correct','macs','parameters','measured_latency','reference_latency')} for k,r in chosen.items()},indent=2))


@torch.inference_mode()
def evaluate_frozen():
    from torchvision.datasets import FashionMNIST
    from sklearn.metrics import classification_report, confusion_matrix
    recipes={goal:json.loads((OUT/f'hybrid_{goal}_recipe.json').read_text()) for goal in ('balanced','accuracy')}
    for recipe in recipes.values():
        assert set(recipe['checkpoint_hashes'])==model_names(recipe)
        assert recipe['selection_split']=='validation_only' and recipe['frozen_at']
        assert all(digest(name)==value for name,value in recipe['checkpoint_hashes'].items())
    data=FashionMNIST('data',train=False,download=True)
    x=data.data.unsqueeze(1).float().div(255).to(DEVICE); labels=data.targets
    outputs={}; saved=dict(labels=labels.numpy())
    models={name:load_model(name) for recipe in recipes.values() for name in model_names(recipe)}
    for goal,recipe in recipes.items():
        p,execution=predict(recipe,x,models)
        metrics=details(p,labels); metrics.pop('validation_half_correct')
        outputs[goal]=dict(**metrics,parameters=recipe['parameters'],execution=execution,
                           classification_report=classification_report(labels,p.argmax(1),target_names=data.classes,output_dict=True,zero_division=0),
                           confusion_matrix=confusion_matrix(labels,p.argmax(1)).tolist(),
                           recipe_label=recipe['label'],recipe_frozen_at=recipe['frozen_at'],
                           checkpoint_hashes=recipe['checkpoint_hashes'])
        saved[goal]=p.numpy()
    result=dict(complete=True,selected=outputs,evaluated_at=datetime.now(timezone.utc).isoformat(),
                protocol='Exactly two endpoints frozen by validation before test access. No recipe, threshold or checkpoint is changed from these outcomes. Previously observed public benchmark, not an independent blinded test.')
    (OUT/'hybrid_test_metrics.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/'hybrid_test_predictions.npz',**saved)
    print(json.dumps({k:{f:v[f] for f in ('accuracy','correct','shirt_f1','parameters','execution')} for k,v in outputs.items()},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); parser.add_argument('--test',action='store_true'); parser.add_argument('--latency',action='store_true')
    args=parser.parse_args()
    self_check() if args.check else evaluate_frozen() if args.test else choose_fast() if args.latency else run()
