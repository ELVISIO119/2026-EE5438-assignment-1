"""SAM/ASAM perturbation with exact weight restoration before the base update."""
import torch

@torch.no_grad()
def perturb(model,rho,adaptive=False):
    params=[p for p in model.parameters() if p.grad is not None]
    scales=[p.abs()+.01 if adaptive and p.ndim>1 else torch.ones_like(p) for p in params]
    norm=torch.stack([(s*p.grad).norm() for p,s in zip(params,scales)]).norm().clamp_min(1e-12)
    original=[p.detach().clone() for p in params]
    for p,s in zip(params,scales):
        p.add_(s.square()*p.grad*(rho/norm))
    return list(zip(params,original))

@torch.no_grad()
def restore(saved):
    for parameter,value in saved:
        parameter.copy_(value)

if __name__=='__main__':
    model=torch.nn.Linear(3,2)
    model(torch.ones(4,3)).square().mean().backward()
    for adaptive in [False,True]:
        original=[p.detach().clone() for p in model.parameters()]
        saved=perturb(model,.05,adaptive)
        assert any(not torch.equal(a,b) for a,b in zip(original,model.parameters()))
        restore(saved)
        assert all(torch.equal(a,b) for a,b in zip(original,model.parameters()))
    print('SAM/ASAM perturbation and exact restoration checks passed.')
