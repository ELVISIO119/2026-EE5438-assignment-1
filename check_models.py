import json
import torch
from pathlib import Path
from train import Model, model_cost, DEVICE

for path in Path('configs').glob('*.json'):
    cfg=json.loads(path.read_text())
    model=Model(cfg,.28,.35).to(DEVICE)
    x=torch.rand(8,1,28,28,device=DEVICE)
    out=model(x)
    assert out.shape==(8,10)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any(isinstance(m,(torch.nn.Conv2d,torch.nn.MultiheadAttention)) for m in model.modules())
    print(path.name,model_cost(model))
