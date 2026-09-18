"""Test confident/non-confident routing, including no hard examples and no early exits."""
import torch
from evaluation import early_mask,recipe_probabilities

class Cheap(torch.nn.Module):
    def forward(self,x):
        logits=torch.zeros(len(x),10)
        logits[:,1]=x[:,0,0,0]*20
        return logits

class Full(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.examples=0
    def forward(self,x):
        self.examples+=len(x)
        logits=torch.zeros(len(x),10); logits[:,6]=10
        return logits

gate=dict(name='cheap',views=1,threshold=.99,classes=[1,5,7,8,9],macs=10)
recipe=dict(components=[dict(name='full',views=1,weight=1.)],macs=100,full_macs=100,cascade=gate)
for values,expected in (([1,0,1,0],2),([1,1],2),([0,0],0)):
    x=torch.zeros(len(values),1,28,28); x[:,0,0,0]=torch.tensor(values)
    full=Full()
    p,stats=recipe_probabilities(recipe,x,{'cheap':Cheap(),'full':full})
    assert stats['early_exit']==expected and full.examples==len(x)-expected
    assert stats['macs_per_image']==10+100*(len(x)-expected)/len(x)
    assert torch.allclose(p.sum(1),torch.ones(len(x)))
    assert p.argmax(1).tolist()==[1 if v else 6 for v in values]
shirt=torch.zeros(1,10); shirt[:,6]=1
assert not early_mask(shirt,gate).any(), 'A confident Shirt must still use the full ensemble.'
print('Cascade confidence/class guards, hard-example routing and exact conditional MACs verified.')
