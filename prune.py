"""Physically remove low-magnitude channel neurons, then measure and fine-tune."""
import argparse
import copy
import json
from datetime import datetime,timezone
from pathlib import Path

import torch
from train import Model,DEVICE,OUT,load_data,model_cost,evaluate,run


def prune(checkpoint,ratio):
    assert 0<ratio<=1 and checkpoint['config']['model']=='mixer'
    cfg=copy.deepcopy(checkpoint['config'])
    old_hidden=cfg.get('channel_hidden',2*cfg['width'])
    keep=max(1,round(old_hidden*ratio))
    cfg['channel_hidden']=keep
    state=copy.deepcopy(checkpoint['state_dict'])
    for block in range(cfg['depth']):
        prefix=f'net.blocks.{block}.channel.'
        first,last=state[prefix+'0.weight'],state[prefix+'3.weight']
        score=first.norm(dim=1)*last.norm(dim=0)
        ids=score.topk(keep).indices.sort().values
        state[prefix+'0.weight']=first[ids].clone()
        state[prefix+'0.bias']=state[prefix+'0.bias'][ids].clone()
        state[prefix+'3.weight']=last[:,ids].clone()
    model=Model(cfg,checkpoint['mean'],checkpoint['std']).to(DEVICE)
    model.load_state_dict(state)
    return model,cfg


def check():
    cfg=dict(model='mixer',width=16,depth=2,dropout=0.,bf16=False)
    model=Model(cfg,.28,.35).to(DEVICE).eval()
    ckpt=dict(state_dict=model.state_dict(),config=cfg,mean=.28,std=.35)
    x=torch.rand(4,1,28,28,device=DEVICE)
    unchanged,_=prune(ckpt,1.)
    assert torch.equal(model(x),unchanged.eval()(x))
    smaller,_=prune(ckpt,.5)
    assert all(a<b for a,b in zip(model_cost(smaller),model_cost(model)))
    assert smaller(x).shape==(4,10)
    print('Pruning identity, checkpoint shape and physical cost checks passed.')


def trials():
    checkpoint=torch.load(OUT/'mixer_adamw.pt',map_location=DEVICE,weights_only=True)
    x,y,vx,vy,mean,std=load_data()
    del x,y
    for ratio in (.75,.5):
        model,cfg=prune(checkpoint,ratio)
        name=f'mixer_pruned_{round(100*ratio)}'
        loss,accuracy=evaluate(model,vx,vy)
        params,macs=model_cost(model)
        cfg.update(name=name,epochs=25,lr=.0001,decay=.01,augmentation=False,
                   label_smoothing=0.,init=name+'_initial',optimizer='adamw',
                   hypothesis=f'Physically retain {ratio:.0%} of channel-MLP neurons ranked by incoming/outgoing weight norms, then clean fine-tune. Compare pre- and post-fine-tuning validation accuracy and real matrix dimensions. Source: mixer_adamw.')
        torch.save(dict(state_dict=model.state_dict(),config=cfg,mean=mean,std=std),OUT/f'{name}_initial.pt')
        initial=dict(config=cfg,val_accuracy=accuracy,val_loss=loss,parameters=params,macs=macs,
                     complete=True,kind='pruned_before_finetuning',source='mixer_adamw',
                     completed_at=datetime.now(timezone.utc).isoformat())
        (OUT/f'{name}_initial.json').write_text(json.dumps(initial,indent=2))
        Path(f'configs/{name}.json').write_text(json.dumps(cfg,indent=2)+'\n')
        with Path('JOURNAL.md').open('a') as journal:
            journal.write(f"\n## {initial['completed_at']} - {name} before fine-tuning\n\n"
                          f"Keep {ratio:.0%} of channel neurons: validation {accuracy:.2%}; "
                          f"{params:,} parameters; {macs:,} MACs/image. "
                          f"Weights, not validation labels, determine neuron ranking. Evidence: `results/{name}_initial.json`.\n")
        del model
        run(cfg)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    check()
    if not args.check:
        trials()
