"""Check spatial grouping, both feature paths, and pure-MLP inference."""
import torch
from models import PyramidMLP
from train import Model, model_cost

torch.manual_seed(58561440)
grid=torch.arange(196.).reshape(1,196,1)
merged=PyramidMLP.merge_tokens(grid)
assert merged[0,0].tolist()==[0,1,14,15]
assert merged[0,-1].tolist()==[180,181,194,195]
assert torch.equal(merged.flatten().sort().values,grid.flatten())
for fusion in (False,True):
    cfg=dict(model='pyramid',width=16,depth=1,dropout=0.,multiscale_fusion=fusion)
    model=Model(cfg,.28,.35)
    x=torch.rand(4,1,28,28)
    logits=model(x)
    assert logits.shape==(4,10)
    logits.square().mean().backward()
    # In coarse-only control, the fine pooled normalization is deliberately unused.
    for name,p in model.named_parameters():
        if not fusion and 'fine_norm' in name:
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert not any(isinstance(m,(torch.nn.Conv2d,torch.nn.MultiheadAttention)) for m in model.modules())
    print('Pyramid fusion:',fusion,'deployment parameters/MACs:',model_cost(model))
print('Adjacent token grouping, finite gradients and output shapes verified.')

# Feature fusion changes the head, but not backbone initialization or training RNG.
torch.manual_seed(58561440)
control=PyramidMLP(dict(width=16,depth=1,dropout=.1,multiscale_fusion=False))
control_rng=torch.get_rng_state()
torch.manual_seed(58561440)
fusion=PyramidMLP(dict(width=16,depth=1,dropout=.1,multiscale_fusion=True))
assert torch.equal(control_rng,torch.get_rng_state())
for name,weight in control.state_dict().items():
    if not name.startswith('head.'):
        assert torch.equal(weight,fusion.state_dict()[name]),name
print('Paired backbones and post-initialization RNG are exactly matched.')

model=Model(dict(model='pyramid',width=16,depth=1,dropout=0.),.28,.35)
auxiliary=torch.nn.Linear(16,10)
model.eval()
logits,features=model(x,return_features=True)
assert torch.equal(logits,model(x))
auxiliary(features).square().mean().backward()
assert model.net.embed.weight.grad.abs().sum()>0
assert model.net.merge[1].weight.grad is None, 'Fine auxiliary loss must not traverse the coarse stage.'
assert not any('auxiliary' in n for n in model.state_dict())
print('Auxiliary gradient reaches fine features; training head is absent from deployment state.')
