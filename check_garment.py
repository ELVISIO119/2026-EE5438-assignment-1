"""Check objective algebra and gradients, including batches with no garment examples."""
import torch
from torch.nn import functional as F
from train import training_loss

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
(actual+focal).backward()
assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
print('Garment conditional CE, ordinary CE equivalence, focal loss and gradients verified.')
