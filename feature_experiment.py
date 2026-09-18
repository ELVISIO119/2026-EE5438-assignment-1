"""Matched training-only fixed-image-feature trials and validation selection."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix

from evaluation import load_model, probabilities
from garment_experiment import details
from models import image_features
from run_yolo import train
from select_yolo import run as select
from train import Model, OUT, load_data, load_initial_state, seed_all

STEMS=['features_gray','features_edges','features_contrast']


def self_check():
    x=torch.arange(28.).reshape(1,1,1,28).expand(2,1,28,28)
    f=image_features(x,'contrast')
    assert torch.equal(f[:,:1],x)
    assert torch.equal(f[:,1,:,1:-1],torch.ones_like(f[:,1,:,1:-1]))
    assert not f[:,2].any()
    assert image_features(torch.full_like(x,.75),'contrast')[:,1:].abs().max()<1e-6
    assert torch.allclose(image_features(x.transpose(2,3),'edges')[:,2],f[:,1].transpose(1,2))
    cfg=dict(model='mixer',width=16,depth=1,dropout=.1,patch=4)
    seed_all(); original=Model(cfg,.28,.35).eval()
    rng=torch.random.get_rng_state()
    batch=torch.rand(8,1,28,28)
    for mode in ('edges','contrast'):
        seed_all(); model=Model(dict(cfg,image_features=mode),.28,.35).eval()
        assert torch.equal(torch.random.get_rng_state(),rng)
        load_initial_state(model,dict(mean=.28,std=.35,state_dict=original.state_dict()))
        torch.testing.assert_close(model(batch),original(batch),rtol=1e-5,atol=1e-6)
        model(batch).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.net.embed.weight.grad[:,16:].abs().sum()>0
        assert not any(isinstance(m,torch.nn.Conv2d) for m in model.modules())
    print('Feature directions, constant boundaries, grayscale preservation, paired RNG, warm start and gradients: PASS',flush=True)


def run(force=False):
    self_check()
    for name,mode in zip(STEMS,('gray','edges','contrast')):
        path=Path('configs')/f'{name}.json'
        assert json.loads(path.read_text())['image_features']==mode
        train(path,force)
    select(STEMS,prefix='features')
    summary=json.loads((OUT/'features_summary.json').read_text())
    x,y,vx,vy,_,_=load_data(); del x,y
    rows=[]
    for family in summary['families']:
        name=family['selected_checkpoint']
        p=probabilities(load_model(name),vx,1)
        cm=confusion_matrix(vy.cpu(),p.argmax(1),labels=list(range(10)))
        rows.append(dict(name=name,**details(p,vy),confusion_matrix=cm.tolist(),
                         class_recall=(cm.diagonal()/cm.sum(1)).tolist(),
                         fixed_feature_scalar_operations_per_view={'features_gray':0,'features_edges':3136,'features_contrast':10976}[family['family']]))
    (OUT/'features_details.json').write_text(json.dumps(dict(complete=True,rows=rows,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        cost_note='Dense MACs exclude normalization, activation and fixed feature arithmetic. Central differences: 4*784 scalar operations; extra 3x3 mean subtraction: 10*784. Padding, copies and memory traffic are excluded; no measured latency claim.',
        protocol='Same parent, 30 clean epochs, optimizer, LR, seed, shuffle/dropout RNG and ordinary/EMA/SWA selection. New input weights start at zero. Features are computed after geometry on every training/inference view. No test evaluation or feature-statistic fitting.'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--retrain',action='store_true')
    args=parser.parse_args()
    self_check() if args.check else run(args.retrain)
