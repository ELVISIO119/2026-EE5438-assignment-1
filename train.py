"""Training-only fitting and validation selection for the assignment experiments."""
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split

SEED = 58561440
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = Path('results')
torch.set_num_threads(4)

def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_data():
    data = FashionMNIST('data', train=True, download=True)
    train_ids, val_ids = train_test_split(np.arange(len(data)), test_size=6000,
                                        stratify=data.targets.numpy(), random_state=SEED)
    assert not set(train_ids) & set(val_ids)
    assert np.array_equal(np.bincount(data.targets[val_ids]), np.full(10, 600))
    x = data.data.unsqueeze(1).float() / 255
    mean, std = x[train_ids].mean().item(), x[train_ids].std().item()
    return (x[train_ids].to(DEVICE), data.targets[train_ids].to(DEVICE),
            x[val_ids].to(DEVICE), data.targets[val_ids].to(DEVICE), mean, std)

class Model(nn.Module):
    def __init__(self, cfg, mean, std):
        super().__init__()
        self.mean, self.std = mean, std
        self.bf16=cfg.get('bf16',False)
        if cfg['model']=='baseline':
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(784,128), nn.Sigmoid(),
                                     nn.Linear(128,64), nn.Sigmoid(), nn.Linear(64,10))
        else:
            from models import ResidualMLP,MLPMixer
            self.net = MLPMixer(cfg) if cfg['model']=='mixer' else ResidualMLP(cfg)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=self.bf16 and x.is_cuda):
            return self.net((x-self.mean)/self.std).float()

def model_cost(model):
    cost = [0]
    def count(layer, inputs, output):
        cost[0] += output.numel() * layer.in_features
    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, nn.Linear)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1,1,28,28,device=DEVICE))
    for hook in hooks:
        hook.remove()
    return sum(p.numel() for p in model.parameters()), cost[0]

def augment(x):
    n=len(x)
    angle=(torch.rand(n,device=x.device)*2-1)*math.pi*8/180
    scale=1+(torch.rand(n,device=x.device)*2-1)*.08
    flip=torch.where(torch.rand(n,device=x.device)<.5,-1.,1.)
    theta=torch.zeros(n,2,3,device=x.device)
    theta[:,0,0]=angle.cos()*scale*flip; theta[:,0,1]=-angle.sin()*scale
    theta[:,1,0]=angle.sin()*scale*flip; theta[:,1,1]=angle.cos()*scale
    theta[:,:,2]=(torch.rand(n,2,device=x.device)*2-1)*(4/28)
    return F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),align_corners=False)

@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = torch.cat([model(b).float() for b in x.split(128)])
    return F.cross_entropy(logits, y).item(), (logits.argmax(1)==y).float().mean().item()

def run(cfg):
    OUT.mkdir(exist_ok=True)
    seed_all()
    x,y,vx,vy,mean,std = load_data()
    model = Model(cfg,mean,std).to(DEVICE)
    if cfg.get('init'):
        initial=torch.load(OUT/f"{cfg['init']}.pt",map_location=DEVICE,weights_only=True)
        assert abs(initial['mean']-mean)<1e-7 and abs(initial['std']-std)<1e-7
        model.load_state_dict(initial['state_dict'])
        del initial
    params,macs = model_cost(model)
    teacher=None
    teacher_params=teacher_macs=0
    if cfg.get('teacher'):
        checkpoint=torch.load(OUT/f"{cfg['teacher']}.pt",map_location=DEVICE,weights_only=True)
        # Keep teacher construction from changing the student's random-number stream.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEVICE.type=='cuda' else []):
            teacher=Model(checkpoint['config'],checkpoint['mean'],checkpoint['std']).to(DEVICE)
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.requires_grad_(False).eval()
        teacher_params,teacher_macs=model_cost(teacher)
        del checkpoint
        assert not cfg.get('sam'), 'Distillation and SAM are separate controlled trials.'
    hidden=[p for name,p in model.named_parameters() if '.blocks.' in name and p.ndim==2] if cfg.get('optimizer')=='muon' else []
    hidden_ids={id(p) for p in hidden}
    other=[p for p in model.parameters() if id(p) not in hidden_ids]
    optimizers=[torch.optim.AdamW([
        {'params':[p for p in other if p.ndim>1],'weight_decay':cfg['decay']},
        {'params':[p for p in other if p.ndim<=1],'weight_decay':0.0}],lr=cfg['lr'])]
    if hidden:
        optimizers.append(torch.optim.Muon(hidden,lr=cfg['lr'],weight_decay=cfg['decay'],
                                          adjust_lr_fn='match_rms_adamw',momentum=.95,ns_steps=5))
    grouped=[p for opt in optimizers for g in opt.param_groups for p in g['params']]
    assert len(grouped)==len({id(p) for p in grouped})==len(list(model.parameters()))
    best = (-1.,-float('inf'))
    history = []
    averages={}
    if cfg.get('averages'):
        from torch.optim.swa_utils import AveragedModel,get_ema_multi_avg_fn
        assert not any(isinstance(m,nn.BatchNorm1d) for m in model.modules())
        averages={'ema':AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(.995)),
                  'swa':AveragedModel(model)}
    average_best={key:(-1.,-float('inf')) for key in averages}
    average_history={key:[] for key in averages}
    average_epochs={}
    started = time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        lr = cfg['lr']
        if cfg.get('schedule'):
            lr *= epoch/5 if epoch<=5 else .01+.99*(1+math.cos(math.pi*(epoch-5)/(cfg['epochs']-5)))/2
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['lr']=lr
        total_loss = torch.zeros((),device=DEVICE)
        correct = torch.zeros((),device=DEVICE)
        for ids in torch.randperm(len(x),device=DEVICE).split(128):
            images,labels=x[ids],y[ids]
            if cfg.get('augmentation'):
                images=augment(images)
            targets=F.one_hot(labels,10).float()
            smoothing=cfg.get('label_smoothing',0.)
            targets=targets*(1-smoothing)+smoothing/10
            if cfg.get('mixup',0)>0:
                lam=float(np.random.beta(cfg['mixup'],cfg['mixup']))
                perm=torch.randperm(len(ids),device=DEVICE)
                images=lam*images+(1-lam)*images[perm]
                targets=lam*targets+(1-lam)*targets[perm]
            model.zero_grad(set_to_none=True)
            logits=model(images); loss=F.cross_entropy(logits,targets)
            if teacher is not None:
                temperature=cfg['temperature']
                with torch.no_grad():
                    teacher_prob=F.softmax(teacher(images)/temperature,dim=1)
                kl=F.kl_div(F.log_softmax(logits/temperature,dim=1),teacher_prob,reduction='batchmean')*temperature**2
                loss=(1-cfg['distill_alpha'])*loss+cfg['distill_alpha']*kl
            loss.backward()
            if cfg.get('sam'):
                from sharpness import perturb,restore
                assert cfg['norm']=='layer', 'SAM trials require LayerNorm to avoid updating BatchNorm statistics twice.'
                saved=perturb(model,cfg['rho'],cfg['sam']=='asam')
                model.zero_grad(set_to_none=True)
                try:
                    F.cross_entropy(model(images),targets).backward()
                finally:
                    restore(saved)
            for optimizer in optimizers:
                optimizer.step()
            if averages:
                averages['ema'].update_parameters(model)
            total_loss += loss.detach()*len(ids)
            correct += (logits.argmax(1)==y[ids]).sum()
        val_loss,val_acc = evaluate(model,vx,vy)
        if averages and epoch>math.floor(.8*cfg['epochs']):
            averages['swa'].update_parameters(model)
        for key,average in averages.items():
            if int(average.n_averaged)==0:
                continue
            av_loss,av_acc=evaluate(average,vx,vy)
            average_history[key].append(dict(epoch=epoch,val_loss=av_loss,val_accuracy=av_acc))
            if (av_acc,-av_loss)>average_best[key]:
                average_best[key]=(av_acc,-av_loss)
                average_epochs[key]=epoch
                torch.save(dict(state_dict=average.module.state_dict(),config=cfg,mean=mean,std=std),OUT/f"{cfg['name']}_{key}.pt")
        row=dict(epoch=epoch,train_loss=total_loss.item()/len(x),train_accuracy=correct.item()/len(x),
                 val_loss=val_loss,val_accuracy=val_acc,lr=lr,seconds=time.perf_counter()-started)
        history.append(row)
        if (val_acc,-val_loss)>best:
            best=(val_acc,-val_loss)
            best_epoch=epoch
            torch.save(dict(state_dict=model.state_dict(),config=cfg,mean=mean,std=std),OUT/f"{cfg['name']}.pt")
        result=dict(config=cfg,seed=SEED,parameters=params,macs=macs,best_epoch=best_epoch,
                    val_accuracy=best[0],val_loss=-best[1],history=history,
                    seconds=time.perf_counter()-started,complete=epoch==cfg['epochs'])
        result['gradient_evaluations']=epoch*math.ceil(len(x)/128)*(2 if cfg.get('sam') else 1)
        if teacher is not None:
            result.update(teacher_parameters=teacher_params,teacher_macs=teacher_macs,
                          teacher_training_cost_included=False,teacher_forward_passes=epoch*len(x))
        (OUT/f"{cfg['name']}.json").write_text(json.dumps(result,indent=2))
        if epoch==1 or epoch%20==0 or epoch==cfg['epochs']:
            print(cfg['name'],row,flush=True)
    result['completed_at']=datetime.now(timezone.utc).isoformat()
    (OUT/f"{cfg['name']}.json").write_text(json.dumps(result,indent=2))
    with Path('JOURNAL.md').open('a') as journal:
        journal.write(f"\n## {result['completed_at']} — {cfg['name']}\n\n"
                      f"Hypothesis: {cfg['hypothesis']}\n\n"
                      f"Measured validation accuracy: {best[0]:.2%}; checkpoint epoch {best_epoch}; "
                      f"{params:,} parameters; {macs:,} dense MACs/image; {result['seconds']:.1f}s training/validation wall time. "
                      f"Configuration and every epoch: `results/{cfg['name']}.json`. Test set not evaluated.\n")
    for key in averages:
        averaged=dict(result,history=average_history[key],val_accuracy=average_best[key][0],
                      val_loss=-average_best[key][1],best_epoch=average_epochs[key],
                      averaging=key,shared_training_run=cfg['name'])
        (OUT/f"{cfg['name']}_{key}.json").write_text(json.dumps(averaged,indent=2))
        with Path('JOURNAL.md').open('a') as journal:
            journal.write(f"\n### {result['completed_at']} - {cfg['name']} {key.upper()}\n\n"
                          f"Validation accuracy {averaged['val_accuracy']:.2%} at epoch {averaged['best_epoch']}. "
                          f"Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. "
                          f"EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. "
                          f"LayerNorm needs no BatchNorm recalibration. Evidence: `results/{cfg['name']}_{key}.json`.\n")
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('config',type=Path)
    args=parser.parse_args()
    run(json.loads(args.config.read_text()))
