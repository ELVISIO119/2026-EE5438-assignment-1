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
def probabilities(model,x,views=1,view_batch=1):
    assert views in (1,2,4,10,18,30,50)
    assert view_batch in (1,2,4)
    outputs=[]
    for batch in x.split(128):
        images=[batch]
        if views>=2:
            images.append(batch.flip(-1))
        if views==4:
            images.extend([F.pad(batch,(1,0,0,0))[...,:28],F.pad(batch,(0,1,0,0))[...,1:]])
        if views in (10,18,30,50):
            radius=2 if views==50 else 1
            shifts=[(0,0),(-1,0),(1,0),(0,-1),(0,1)] if views in (10,30) else [(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)]
            images=[]
            for scale in ((1.,.96,1.04) if views==30 else (1.,)):
                scaled=batch
                if scale!=1.:
                    theta=torch.zeros(len(batch),2,3,device=batch.device)
                    theta[:,0,0]=theta[:,1,1]=scale
                    scaled=F.grid_sample(batch,F.affine_grid(theta,batch.shape,align_corners=False),align_corners=False)
                padded=F.pad(scaled,(radius,)*4)
                shifted=[padded[:,:,radius+dy:radius+dy+28,radius+dx:radius+dx+28] for dy,dx in shifts]
                images.extend(shifted+[image.flip(-1) for image in shifted])
        assert len(images)==views
        per_view=[]
        for start in range(0,len(images),view_batch):
            group=images[start:start+view_batch]
            joined=group[0] if len(group)==1 else torch.cat(group)
            per_view.extend(model(joined).softmax(1).reshape(len(group),len(batch),10).unbind(0))
        outputs.append(torch.stack(per_view).mean(0).cpu())
    return torch.cat(outputs)


def probability_metrics(probs,labels):
    labels=labels.cpu()
    correct=int((probs.argmax(1)==labels).sum())
    loss=float(-probs[torch.arange(len(labels)),labels].clamp_min(1e-12).log().mean())
    return dict(correct=correct,examples=len(labels),accuracy=correct/len(labels),loss=loss)


def calibrate(probs,temperature=1.):
    """Temperature scaling of log probabilities AFTER averaging a model's views."""
    assert temperature>0
    return probs if temperature==1. else (probs.clamp_min(1e-12).log()/temperature).softmax(1)


def early_mask(probs,cascade):
    confidence,predicted=probs.max(1)
    threshold=cascade['threshold']
    if 'class_thresholds' in cascade:
        thresholds=torch.as_tensor(cascade['class_thresholds'],device=probs.device,dtype=probs.dtype)
        assert thresholds.shape==(probs.shape[1],) and ((thresholds>=0)&(thresholds<=1)).all()
        threshold=thresholds[predicted]
    return (confidence>=threshold) & torch.isin(predicted,torch.tensor(cascade['classes'],device=predicted.device))


@torch.no_grad()
def recipe_probabilities(recipe,x,models=None):
    """Evaluate a frozen ensemble, optionally with a confidence-gated cheap first stage."""
    models={} if models is None else models
    def model(name):
        if name not in models:
            models[name]=load_model(name)
        return models[name]
    cascade=recipe.get('cascade')
    view_batch=recipe.get('view_batch',1)
    routed=x
    accepted=0
    if cascade:
        assert 0<cascade['threshold']<=1 and all(0<=c<10 for c in cascade['classes'])
        cheap=probabilities(model(cascade['name']),x,cascade['views'],view_batch)
        mask=early_mask(cheap,cascade)
        accepted=int(mask.sum())
        routed=x[(~mask).to(x.device)]
    components=recipe['components']
    weights=torch.tensor([c.get('weight',1.) for c in components])
    assert torch.isfinite(weights).all() and (weights>0).all()
    weights=weights/weights.sum()
    probs=torch.empty(0,10) if len(routed)==0 else sum(w*calibrate(probabilities(model(c['name']),routed,c['views'],view_batch),c.get('temperature',1.)) for w,c in zip(weights,components))
    full_macs=recipe.get('full_macs',recipe['macs'])
    macs=full_macs
    if cascade:
        cheap[~mask]=probs
        probs=cheap
        macs=cascade['macs']+(len(x)-accepted)/len(x)*full_macs
    return probs,dict(examples=len(x),early_exit=accepted,full_ensemble=len(x)-accepted,
                      macs_per_image=macs,worst_case_macs=full_macs+(cascade['macs'] if cascade else 0))


if __name__=='__main__':
    p=torch.tensor([[.8,.2],[.2,.8]])
    assert early_mask(p,dict(threshold=.9,classes=[0,1],class_thresholds=[.75,.85])).tolist()==[True,False]
    assert early_mask(p,dict(threshold=.7,classes=[0])).tolist()==[True,False]
    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls=0
        def forward(self,x):
            self.calls+=1
            assert x.shape[1:]==(1,28,28)
            return torch.zeros(len(x),10,device=x.device)
    for views in (1,2,4,10,18,30,50):
        model=Constant()
        probs=probabilities(model,torch.rand(3,1,28,28),views)
        assert model.calls==views and probs.shape==(3,10)
        assert torch.allclose(probs,torch.full_like(probs,.1),atol=1e-7)
    print('All seven inference view counts preserve shape and normalized probabilities.')
    class Pixels(torch.nn.Module):
        def forward(self,x):
            return x.flatten(1)[:,:10]*4
    torch.manual_seed(58561440)
    inputs=torch.rand(131,1,28,28)
    for views in (1,2,4,10,18,30,50):
        expected=probabilities(Pixels(),inputs,views)
        for grouped in (2,4):
            assert torch.allclose(probabilities(Pixels(),inputs,views,grouped),expected,atol=1e-7)
    print('Grouped views match sequential probabilities across partial image/view batches.')
