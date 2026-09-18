"""Shared checkpoint loading and explicit-cost probability averaging."""
import hashlib
import torch
from torch.nn import functional as F
from train import Model,DEVICE,OUT


def digest(name):
    return hashlib.sha256((OUT/f'{name}.pt').read_bytes()).hexdigest()


def load_model(name):
    checkpoint=torch.load(OUT/f'{name}.pt',map_location=DEVICE,weights_only=True)
    state=checkpoint['state_dict']
    if name=='baseline_sgd_sigmoid':
        cfg={'model':'baseline'}
        state={f'net.{int(key.split(".")[1])+1}.{key.split(".")[2]}':value for key,value in state.items()}
    else:
        cfg=checkpoint['config']
    model=Model(cfg,checkpoint['mean'],checkpoint['std']).to(DEVICE)
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def probabilities(model,x,views=1):
    assert views in (1,2,4)
    outputs=[]
    for batch in x.split(128):
        images=[batch]
        if views>=2:
            images.append(batch.flip(-1))
        if views==4:
            images.extend([F.pad(batch,(1,0,0,0))[...,:28],F.pad(batch,(0,1,0,0))[...,1:]])
        outputs.append(torch.stack([model(image).softmax(1) for image in images]).mean(0).cpu())
    return torch.cat(outputs)


def probability_metrics(probs,labels):
    labels=labels.cpu()
    correct=int((probs.argmax(1)==labels).sum())
    loss=float(-probs[torch.arange(len(labels)),labels].clamp_min(1e-12).log().mean())
    return dict(correct=correct,examples=len(labels),accuracy=correct/len(labels),loss=loss)
