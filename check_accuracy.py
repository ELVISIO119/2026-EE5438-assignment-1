"""Small deterministic check of expanded views and weighted ensemble selection."""
import tempfile
from pathlib import Path

import torch
import select_final
from evaluation import probabilities,probability_metrics,calibrate


class InspectViews(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen=[]
    def forward(self,x):
        self.seen.append(x.clone())
        return torch.zeros(len(x),10)


x=torch.zeros(1,1,28,28)
x[0,0,14,10]=1.
for views,unique in ((10,10),(18,18),(30,30),(50,50)):
    model=InspectViews()
    p=probabilities(model,x,views)
    transforms={image.numpy().tobytes() for image in model.seen}
    assert len(transforms)==unique
    if views!=30:
        assert all(image.sum()==1 for image in model.seen)
    assert torch.allclose(p.sum(1),torch.ones(1))

labels=torch.zeros(6000,dtype=torch.long)
example=torch.softmax(torch.randn(8,10),dim=1)
for temperature in (.75,1.,1.25,1.5,2.):
    calibrated=calibrate(example,temperature)
    assert torch.equal(example.argmax(1),calibrated.argmax(1))
    assert torch.allclose(calibrated.sum(1),torch.ones(8),atol=1e-6)
assert torch.equal(example,calibrate(example))
leaders=[]; cached={}
for index in range(3):
    key=f'model{index}:10'
    p=torch.full((6000,10),.1/9); p[:,0]=.9
    p[index*100:(index+1)*100,0]=.3
    p[index*100:(index+1)*100,1]=.611111111
    assert torch.allclose(p.sum(1),torch.ones(6000),atol=1e-6)
    cached[key]=p
    leaders.append(dict(name=key,components=[dict(name=f'model{index}',views=10)],parameters=10,macs=100,
                        **probability_metrics(p,labels)))
with tempfile.TemporaryDirectory(prefix='ee5438_ensemble_check_') as directory:
    select_final.OUT=Path(directory)
    outcomes=select_final.accuracy_ensembles(leaders,cached,labels,lambda row:(-row['correct'],row['loss']))
assert max(row['correct'] for row in outcomes)==6000
for row in outcomes:
    assert abs(sum(c['weight'] for c in row['components'])-1)<1e-6
    assert row['parameters']==10*len(row['components'])
    assert row['macs']==100*len(row['components'])
    assert sum(row['validation_half_correct'])==row['correct']
print('Distinct shifted/scaled views, convex ensemble weights, complementary errors and total costs verified.')
