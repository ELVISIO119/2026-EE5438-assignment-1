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
def probabilities(model,x,views=1,view_batch=1,cached_first=None,cached_count=1):
    assert views in (1,2,4,10,18,30,50)
    assert view_batch in (1,2,4,8)
    if cached_first is not None:
        assert views in (1,2,4,10,30) and cached_first.shape==(len(x),10)
        assert cached_count==1 or (cached_count==10 and views==30)
    outputs=[]
    for index,batch in enumerate(x.split(128)):
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
        if cached_first is not None:
            # A repeated cached mean has the same total weight as its original view sum.
            per_view=[cached_first[index*128:index*128+len(batch)].to(batch.device)]*cached_count
            images=images[cached_count:]
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
def adaptive_views(recipe,x,models):
    """Use ten fine-patch views first; append the remaining twenty only if needed."""
    gate=recipe['view_exit']; members=recipe['components']
    index=next(i for i,c in enumerate(members) if c['name']==gate['name'])
    assert members[index]['views']==30 and gate['initial_views']==10
    weights=torch.tensor([c.get('weight',1.) for c in members]); weights/=weights.sum()
    raw=[probabilities(models[c['name']],x,10 if i==index else c['views'],recipe.get('view_batch',1))
         for i,c in enumerate(members)]
    p=sum(w*calibrate(v,c.get('temperature',1.)) for w,v,c in zip(weights,raw,members))
    accepted=early_mask(p,gate)
    remaining=~accepted
    if remaining.any():
        extra=probabilities(models[gate['name']],x[remaining.to(x.device)],30,recipe.get('view_batch',1),
                            raw[index][remaining],cached_count=10)
        completed=[extra if i==index else v[remaining] for i,v in enumerate(raw)]
        p[remaining]=sum(w*calibrate(v,c.get('temperature',1.)) for w,v,c in zip(weights,completed,members))
    cost=recipe['full_macs']-int(accepted.sum())/len(x)*gate['remaining_macs']
    return p,cost,dict(examples=len(x),view_exit=int(accepted.sum()),extra_views=int(remaining.sum()))


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
    reuse=recipe.get('reuse_gate',False)
    if reuse:
        assert cascade and cascade['views']==1
        assert sum(c['name']==cascade['name'] for c in components)==1
    weights=torch.tensor([c.get('weight',1.) for c in components])
    assert torch.isfinite(weights).all() and (weights>0).all()
    weights=weights/weights.sum()
    full_macs=recipe.get('full_macs',recipe['macs'])-(cascade['macs'] if reuse else 0)
    worst_macs=full_macs+(cascade['macs'] if cascade else 0)
    extra={}
    if recipe.get('view_exit') and len(routed):
        assert not reuse, 'Gate reuse and adaptive views are separate experiments.'
        probs,full_macs,counts=adaptive_views(recipe,routed,{c['name']:model(c['name']) for c in components})
        extra=dict(adaptive_views=counts)
    else:
        probs=torch.empty(0,10) if len(routed)==0 else sum(w*calibrate(
            probabilities(model(c['name']),routed,c['views'],view_batch,
                          cheap[~mask] if reuse and c['name']==cascade['name'] else None),
            c.get('temperature',1.)) for w,c in zip(weights,components))
    macs=full_macs
    if cascade:
        cheap[~mask]=probs
        probs=cheap
        macs=cascade['macs']+(len(x)-accepted)/len(x)*full_macs
    return probs,dict(examples=len(x),early_exit=accepted,full_ensemble=len(x)-accepted,
                      macs_per_image=macs,worst_case_macs=worst_macs,**extra)


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
        for grouped in (2,4,8):
            assert torch.allclose(probabilities(Pixels(),inputs,views,grouped),expected,atol=1e-7)
    print('Grouped views match sequential probabilities across partial image/view batches.')
    class CountPixels(Pixels):
        def __init__(self):
            super().__init__(); self.rows=0
        def forward(self,x):
            self.rows+=len(x)
            return super().forward(x)
    for views in (1,2,4,10,30):
        for threshold in (1.,1e-6):
            recipe=dict(components=[dict(name='shared',views=views)],macs=10*views,full_macs=10*views,
                        view_batch=8,cascade=dict(name='shared',views=1,threshold=threshold,classes=list(range(10)),macs=10))
            original=CountPixels(); reused=CountPixels()
            expected,old_cost=recipe_probabilities(recipe,inputs,dict(shared=original))
            actual,new_cost=recipe_probabilities(dict(recipe,reuse_gate=True),inputs,dict(shared=reused))
            assert torch.allclose(actual,expected,atol=1e-7)
            routed=new_cost['full_ensemble']
            assert original.rows-reused.rows==routed
            assert abs(old_cost['macs_per_image']-new_cost['macs_per_image']-routed/len(inputs)*10)<1e-7
            assert old_cost['worst_case_macs']-new_cost['worst_case_macs']==10
    print('Reused identity view matches output and skips/charges exactly one pass per fallback image.')
    model=CountPixels()
    prefix=probabilities(model,inputs,10,4)
    completed=probabilities(model,inputs,30,4,prefix,cached_count=10)
    expected=probabilities(Pixels(),inputs,30,4)
    assert model.rows==len(inputs)*30
    assert torch.allclose(completed,expected,atol=1e-7)
    for threshold,extra_rows in ((0.,0),(1.,len(inputs)*20)):
        recipe=dict(components=[dict(name='fine',views=30)],macs=300,full_macs=300,view_batch=4,
                    view_exit=dict(name='fine',initial_views=10,remaining_macs=200,threshold=threshold,classes=list(range(10))))
        model=CountPixels()
        p,cost=recipe_probabilities(recipe,inputs,dict(fine=model))
        assert model.rows==len(inputs)*10+extra_rows
        assert cost['macs_per_image']==(100 if extra_rows==0 else 300)
        assert cost['worst_case_macs']==300 and torch.allclose(p.sum(1),torch.ones(len(inputs)))
    print('Adaptive views: ten-view reuse, appended-only work, early/full exits and MACs: PASS')
