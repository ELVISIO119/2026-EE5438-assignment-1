"""Check objective algebra and gradients, including batches with no garment examples."""
import torch
from torch.nn import functional as F
from train import Model,load_initial_state,training_loss

torch.manual_seed(58561440)
logits=torch.randn(7,10,requires_grad=True)
labels=torch.tensor([0,2,3,4,6,1,9])
targets=F.one_hot(labels,10).float()
ce=F.cross_entropy(logits,labels)
assert torch.allclose(training_loss(logits,targets,{}),ce)
group=[0,2,3,4,6]
expected=ce+.5*F.cross_entropy(logits[:5,group],torch.arange(5))*5/7
actual=training_loss(logits,targets,{'garment_loss_weight':.5})
assert torch.allclose(actual,expected)
assert torch.allclose(training_loss(logits[5:],targets[5:],{'garment_loss_weight':.5}),F.cross_entropy(logits[5:],labels[5:]))
focal=training_loss(logits,targets,{'focal_gamma':1.})
assert 0<focal<ce
focal2=training_loss(logits,targets,{'focal_gamma':2.})
expected_focal2=((1-logits.softmax(1)[torch.arange(len(labels)),labels]).square()*F.cross_entropy(logits,labels,reduction='none')).mean()
torch.testing.assert_close(focal2,expected_focal2)
weights=torch.ones(10); weights[[0,2,4,6]]=1.5
weighted=training_loss(logits,targets,{'class_weights':weights.tolist()})
torch.testing.assert_close(weighted,F.cross_entropy(logits,labels,weight=weights))
torch.testing.assert_close(training_loss(logits,targets,{'class_weights':[1.]*10}),ce)
(weighted+focal2).backward(retain_graph=True)
(actual+focal).backward()
assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
print('Garment conditional CE, ordinary CE equivalence, focal loss and gradients verified.')

cfg=dict(model='mixer',patch=4,width=32,depth=2,dropout=0.)
old=Model(cfg,.28,.35).eval()
new=Model(dict(cfg,spatial_head=True),.28,.35).eval()
load_initial_state(new,dict(state_dict=old.state_dict(),mean=.28,std=.35))
x=torch.rand(3,1,28,28)
assert torch.allclose(old(x),new(x),atol=1e-6,rtol=1e-5)
assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())==(49-1)*32*10
new(x).square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
assert not any(isinstance(m,(torch.nn.Conv2d,torch.nn.MultiheadAttention)) for m in new.modules())
print('Spatial head preserves initial logits, adds the expected parameters and has finite gradients.')
