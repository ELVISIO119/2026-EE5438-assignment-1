"""Training-only ridge readouts on frozen MLP features, inspired by PIL.

min mean(||H W + b - one_hot(y)||^2) + lambda ||W||^2.
CPU float64 thin SVD, unpenalized intercept, no explicit inverse/normal equations.
"""
import itertools
import json
import time
from datetime import datetime, timezone

import pandas as pd
import torch
from torch.nn import functional as F

from evaluation import load_model, digest, probabilities, calibrate, recipe_probabilities, early_mask
from garment_experiment import details
from train import OUT, DEVICE, SEED, load_data, model_cost

LAMBDAS=(0.,1e-6,1e-5,1e-4,1e-3,.01,.1,1.)
TEMPERATURES=(.05,.1,.2,.5,1.,2.)


def ridge_solutions(features, targets, penalties=LAMBDAS):
    h=features.double(); t=targets.double()
    hm=h.mean(0); tm=t.mean(0)
    u,s,vh=torch.linalg.svd(h-hm,full_matrices=False)
    projection=u.T@(t-tm)
    tolerance=max(h.shape)*torch.finfo(h.dtype).eps*s.max()
    for penalty in penalties:
        assert penalty>=0
        factor=torch.where(s>tolerance,s.reciprocal(),0.) if penalty==0 else s/(s.square()+len(h)*penalty)
        w=vh.T@(factor[:,None]*projection)
        yield penalty,w.T.contiguous(),tm-hm@w


@torch.inference_mode()
def features(model,images):
    chunks=[]
    def capture(module,args):
        # Match the dtype reaching the BF16 classifier GEMM after autocast.
        h=args[0].to(torch.bfloat16) if model.bf16 and images.is_cuda else args[0]
        chunks.append(h.float().cpu())
    hook=model.net.head.register_forward_pre_hook(capture)
    try:
        for batch in images.split(128):
            model(batch)
    finally:
        hook.remove()
    return torch.cat(chunks)


@torch.inference_mode()
def head_probabilities(model,h):
    output=[]
    for batch in h.split(128):
        with torch.autocast(device_type=DEVICE.type,dtype=torch.bfloat16,enabled=model.bf16 and DEVICE.type=='cuda'):
            logits=model.net.head(batch.to(DEVICE))
        output.append(logits.float().softmax(1).cpu())
    return torch.cat(output)


def self_check():
    generator=torch.Generator().manual_seed(SEED)
    h=torch.randn(40,5,generator=generator,dtype=torch.float64)
    h=torch.cat((h,h[:,:1]),1)  # Deliberately rank deficient.
    t=torch.randn(40,3,generator=generator,dtype=torch.float64)+2
    for lam,w,b in ridge_solutions(h,t,(0.,.01)):
        a=torch.cat((h,torch.ones(len(h),1,dtype=h.dtype)),1)
        if lam:
            regularizer=torch.diag(torch.tensor([len(h)*lam]*h.shape[1]+[0.],dtype=h.dtype).sqrt())
            a=torch.cat((a,regularizer)); target=torch.cat((t,torch.zeros(h.shape[1]+1,3,dtype=h.dtype)))
        else:
            target=t
        oracle=torch.linalg.lstsq(a,target,driver='gelsd').solution
        assert torch.allclose(h@w.T+b,torch.cat((h,torch.ones(len(h),1,dtype=h.dtype)),1)@oracle,atol=1e-10)
    print('Ridge SVD agrees with augmented least squares, including rank deficiency and unpenalized bias.',flush=True)


@torch.inference_mode()
def run():
    self_check()
    reference=json.loads((OUT/'final_recipe.json').read_text())
    (OUT/'pil_reference.json').write_text(json.dumps(reference,indent=2))
    result=dict(started_at=datetime.now(timezone.utc).isoformat(),complete=False,
                method='Frozen trained MLP features; CPU float64 centered thin SVD; ridge penalty on weights only; one-hot squared-error targets. Lambda is scaled for mean loss. Lambda zero is the thresholded pseudoinverse solution. No arctanh of one-hot targets.',
                lambdas=LAMBDAS,temperatures=TEMPERATURES,
                data='54000 training images fit the readouts; 6000 validation images select lambda by actual BF16 correct count then NLL, and post-view temperature by NLL. No test split loaded.',
                deployment_search='Keep the original gate and its shared wide model. Compare standalone refits, nine additions (three heads x weights .1/.2/.35) and three combinations replacing the two non-gate members. Original views retained. Rank correct count, average MACs, parameters, NLL; guard both validation halves and Shirt F1.',
                limitations='PIL-inspired ridge readout, not reproduction of original full-network PIL. Backbone fitting remains gradient-based; repeated validation selection is exploratory. Saved heads retain identical dimensions, but additional ensemble members incur full extra backbone cost.',
                heads=[])
    (OUT/'pil_experiment.json').write_text(json.dumps(result,indent=2))
    x,y,vx,vy,_,_=load_data()
    targets=F.one_hot(y.cpu(),10).double()
    fitted=[]
    for component in reference['components']:
        parent=component['name']; name='pil_'+parent
        assert digest(parent)==component['sha256']
        model=load_model(parent).requires_grad_(False)
        checkpoint=torch.load(OUT/f'{parent}.pt',map_location='cpu',weights_only=True)
        original=probabilities(model,vx)
        started=time.perf_counter()
        h=features(model,x); vh=features(model,vx)
        assert torch.equal(head_probabilities(model,vh),original), 'Cached head path differs from full original model.'
        rows=[]; best=None; best_state=None
        for penalty,w,b in ridge_solutions(h,targets):
            model.net.head.weight.copy_(w.to(DEVICE)); model.net.head.bias.copy_(b.to(DEVICE))
            p=head_probabilities(model,vh)
            row=dict(penalty=penalty,**details(p,vy)); rows.append(row)
            rank=(-row['correct'],row['loss'])
            if best is None or rank<best:
                best=rank; selected=row; best_state=(w.clone(),b.clone())
        model.net.head.weight.copy_(best_state[0].to(DEVICE)); model.net.head.bias.copy_(best_state[1].to(DEVICE))
        actual=probabilities(model,vx)
        assert torch.equal(actual,head_probabilities(model,vh))
        assert details(actual,vy)['correct']==selected['correct']
        params,macs=model_cost(model)
        cfg=dict(checkpoint['config'],name=name,pil_parent=parent,pil_lambda=selected['penalty'])
        state={k:v.cpu() for k,v in model.state_dict().items()}
        assert all(torch.equal(v,checkpoint['state_dict'][k]) for k,v in state.items() if not k.startswith('net.head.'))
        torch.save(dict(state_dict=state,config=cfg,mean=model.mean,std=model.std),OUT/f'{name}.pt')
        record=dict(config=cfg,complete=True,seed=SEED,parameters=params,macs=macs,val_accuracy=selected['accuracy'],val_loss=selected['loss'],
                    seconds=time.perf_counter()-started,best_epoch=None,history=[],parent_sha256=digest(parent),
                    fitting='Closed-form ridge head; no gradient epochs. Timing includes feature extraction, SVD and validation selection; parent training excluded.',
                    completed_at=datetime.now(timezone.utc).isoformat())
        (OUT/f'{name}.json').write_text(json.dumps(record,indent=2))
        row=dict(name=name,parent=parent,parent_sha256=digest(parent),original=details(original,vy),selected=selected,candidates=rows,
                 parameters=params,macs=macs,feature_dimension=h.shape[1],seconds=record['seconds'])
        result['heads'].append(row); fitted.append(dict(component,name=name,sha256=digest(name)))
        print(parent,'->',selected['correct'],'original',row['original']['correct'],'lambda',selected['penalty'],flush=True)
        (OUT/'pil_experiment.json').write_text(json.dumps(result,indent=2))
        del model,h,vh
    del x,y,targets
    gate=reference['cascade']; cheap=probabilities(load_model(gate['name']),vx,gate['views'])
    mask=early_mask(cheap,gate); hard=vx[(~mask).to(DEVICE)]
    cache={}; costs={}; models={}
    def get(c,images,key):
        name=c['name']
        if name not in models:
            models[name]=load_model(name); costs[name]=model_cost(models[name])
        if (name,key) not in cache:
            cache[name,key]=probabilities(models[name],images,c['views'])
        return cache[name,key]
    for c in fitted:
        p=get(c,vx,'full')
        c['temperature']=min(TEMPERATURES,key=lambda t:details(calibrate(p,t),vy)['loss'])
    rows=[]
    def score(components,routed=True):
        p=sum(c['weight']*calibrate(get(c,hard if routed else vx,'hard' if routed else 'full'),c['temperature']) for c in components)
        full_macs=sum(costs[c['name']][1]*c['views'] for c in components)
        if routed:
            assert any(c['name']==gate['name'] for c in components)
            q=cheap.clone(); q[~mask]=p; p=q
        row=dict(components=components,parameters=sum(costs[n][0] for n in {c['name'] for c in components}),
                 full_macs=full_macs,macs=gate['macs']+len(hard)/len(vx)*full_macs if routed else full_macs,
                 worst_case_macs=full_macs+(gate['macs'] if routed else 0),**details(p,vy))
        if routed: row['cascade']=gate
        rows.append(row)
        return row
    base=score(reference['components']); assert base['correct']==reference['correct']
    for c in fitted:
        score([dict(c,weight=1.)],False)
        for alpha in (.1,.2,.35):
            score([dict(old,weight=old['weight']*(1-alpha)) for old in reference['components']]+[dict(c,weight=alpha)])
    for switches in itertools.product((False,True),repeat=2):
        if any(switches):
            score([reference['components'][0]]+[fitted[i] if switch else reference['components'][i] for i,switch in enumerate(switches,1)])
    valid=[r for r in rows if r['shirt_f1']>=base['shirt_f1'] and all(a>=b for a,b in zip(r['validation_half_correct'],base['validation_half_correct']))]
    chosen=dict(min(valid,key=lambda r:(-r['correct'],r['macs'],r['parameters'],r['loss'])))
    actual,execution=recipe_probabilities(chosen,vx)
    assert details(actual,vy)['correct']==chosen['correct']
    assert details(actual,vy)['shirt_f1']==chosen['shirt_f1']
    chosen.update(validation_execution=execution,seed=SEED,frozen_at=datetime.now(timezone.utc).isoformat(),
                  name=' + '.join(f"{c['weight']:.4f}*{c['name']}:{c['views']}:T{c['temperature']}" for c in chosen['components'])+('; frozen confidence gate' if chosen.get('cascade') else ''),
                  candidates=len(rows),selection_split='validation_only',objective='accuracy_then_expected_cost',rule=result['deployment_search'])
    (OUT/'pil_recipe.json').write_text(json.dumps(chosen,indent=2))
    pd.DataFrame([{**{k:v for k,v in r.items() if k not in ('components','cascade')},'components':json.dumps(r['components'])} for r in rows]).to_csv(OUT/'pil_candidates.csv',index=False)
    result.update(complete=True,completed_at=chosen['frozen_at'],deployment_candidates=len(rows),reference_correct=base['correct'],selected_correct=chosen['correct'],selected_components=chosen['components'],calibrated_heads=fitted)
    (OUT/'pil_experiment.json').write_text(json.dumps(result,indent=2))
    print('Selected:',chosen['correct'],'reference:',base['correct'],'candidates:',len(rows),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true')
    if parser.parse_args().check: self_check()
    else: run()
