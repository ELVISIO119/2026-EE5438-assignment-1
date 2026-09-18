"""Validation-only sparse MLP-expert experiment."""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from evaluation import load_model, probabilities
from garment_experiment import details
from run_yolo import train
from select_yolo import run as select
from train import DEVICE, OUT, load_data, model_cost


def self_check():
    from train import Model, load_initial_state, seed_all
    seed_all()
    cfg=dict(model='mixer',patch=4,width=16,depth=1,dropout=.1)
    parent=Model(cfg,.28,.35).eval()
    moe=Model(dict(model='moe',patch=4,width=16,depth=1,dropout=.1,num_experts=3,expert_hidden=32),.28,.35).eval()
    load_initial_state(moe,dict(state_dict=parent.state_dict(),mean=.28,std=.35))
    x=torch.rand(9,1,28,28)
    torch.testing.assert_close(parent(x),moe(x),rtol=1e-5,atol=1e-6)
    assert moe.net.last_router_probs.shape==(9,3)
    assert torch.isfinite(moe(x)).all()
    assert not any(isinstance(m,torch.nn.Conv2d) for m in moe.modules())
    # Check real sparse dispatch with nonzero expert outputs and all three routes.
    with torch.no_grad():
        for i,e in enumerate(moe.net.experts):
            e[-1].bias.fill_(i+.5)
        moe.net.router.weight.zero_()
        counts=[0,0,0]
        hooks=[e.register_forward_hook(lambda m,a,o,i=i:counts.__setitem__(i,counts[i]+len(a[0]))) for i,e in enumerate(moe.net.experts)]
        for i in range(3):
            moe.net.router.bias.fill_(-10); moe.net.router.bias[i]=10
            actual=moe(x)
            f=moe.net.features((x-.28)/.35)
            expected=moe.net.head(f+moe.net.experts[i](f))
            torch.testing.assert_close(actual,expected)
        for hook in hooks: hook.remove()
        assert counts==[18,18,18], counts  # Nine routed + nine manual oracle rows.
    moe.train(); moe(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in moe.parameters())
    print('MoE warm-start, top-1 routing, residual experts, finite gradients and no convolution: PASS',flush=True)


@torch.no_grad()
def route_counts(model,x):
    counts=torch.zeros(model.net.num_experts,dtype=torch.long)
    for batch in x.split(128):
        model(batch)
        counts += model.net.last_router_probs.argmax(1).cpu().bincount(minlength=model.net.num_experts)
    return counts.tolist()


def sparse_cost(model):
    params,sparse=model_cost(model)
    expert=sum(m.in_features*m.out_features for m in model.net.experts[0] if isinstance(m,torch.nn.Linear))
    return dict(parameters=params,sparse_macs=sparse,soft_training_macs=sparse+expert*(model.net.num_experts-1),full_expert_macs=sparse+expert*(model.net.num_experts-1),expert_macs=expert)


@torch.inference_mode()
def latency(model,x):
    def infer():
        for batch in x.split(128): model(batch)
        if x.is_cuda: torch.cuda.synchronize()
    for _ in range(3): infer()
    times=[]
    for _ in range(7):
        started=time.perf_counter(); infer(); times.append(1000*(time.perf_counter()-started))
    return dict(milliseconds=times,median_ms=statistics.median(times),images=len(x),batch_size=128)


def run(force=False):
    self_check()
    for stem in ('moe_control','moe_mixer'):
        train(Path('configs')/f'{stem}.json',force)
    x,y,vx,vy,_,_=load_data(); del x,y
    records=[]
    for name in [stem+suffix for stem in ('moe_control','moe_mixer') for suffix in ('','_ema','_swa')]:
        model=load_model(name)
        p=probabilities(model,vx,1)
        row=dict(name=name,**details(p,vy),routes=route_counts(model,vx),**sparse_cost(model))
        if model.net.num_experts>1:
            model.net.dense_inference=True
            row['dense_metrics']=details(probabilities(model,vx,1),vy)
        records.append(row)
        del model
    selected=min((r for r in records if r['name'].startswith('moe_mixer')),key=lambda r:(-r['accuracy'],r['loss']))
    control=min((r for r in records if r['name'].startswith('moe_control')),key=lambda r:(-r['accuracy'],r['loss']))
    timings={}
    for name in ('accuracy_wide_clean_swa',control['name'],selected['name']):
        model=load_model(name)
        timings[name]=latency(model,vx[:1024])
        if name==selected['name']:
            model.net.dense_inference=True
            timings[name+'_dense']=latency(model,vx[:1024])
        del model
    result=dict(complete=True,selected=selected,control=control,checkpoints=records,timings=timings,parent='accuracy_wide_clean_swa',
                protocol='Three experts share a Mixer trunk. Training uses differentiable soft routing; evaluation uses top-1 routing and executes only the selected residual MLP expert per image. The parent trunk/head is warm-started and expert residual output layers start at zero. Validation only; test labels are not evaluated.',
                routing_note='Shared-trunk soft expert training is not independently bootstrap-trained bagging. No explicit balancing loss is used; route counts expose collapse. A route is not a correctness guarantee. Dense metrics diagnose train/eval mismatch; dense checkpoints are not deployment candidates. MACs exclude normalization/activation/routing/memory overhead. Timings are sequential eager measurements on a shared device, not dedicated-device speed claims.',
                frozen_at=datetime.now(timezone.utc).isoformat())
    (OUT/'moe_experiment.json').write_text(json.dumps(result,indent=2))
    select(['moe_control','moe_mixer'],prefix='moe')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--retrain',action='store_true')
    args=parser.parse_args()
    self_check() if args.check else run(args.retrain)
