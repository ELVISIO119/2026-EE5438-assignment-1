import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,width,activation,norm,dropout):
        super().__init__()
        normalizer=nn.LayerNorm if norm=='layer' else nn.BatchNorm1d
        act={'gelu':nn.GELU,'silu':nn.SiLU,'relu':nn.ReLU,'swiglu':nn.SiLU}[activation]
        self.norm=normalizer(width)
        hidden=round(4*width/3) if activation=='swiglu' else 2*width
        self.fc1=nn.Linear(width,hidden)
        self.gate=nn.Linear(width,hidden) if activation=='swiglu' else None
        self.act=act()
        self.drop=nn.Dropout(dropout)
        self.fc2=nn.Linear(hidden,width)

    def forward(self,x):
        normalized=self.norm(x)
        hidden=self.act(self.fc1(normalized))
        if self.gate is not None:
            hidden=hidden*self.gate(normalized)
        return x+self.fc2(self.drop(hidden))

class ResidualMLP(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        width=cfg['width']
        self.embed=nn.Linear(784,width)
        self.blocks=nn.Sequential(*[ResidualBlock(width,cfg['activation'],cfg['norm'],cfg['dropout']) for _ in range(cfg['depth'])])
        self.norm=nn.LayerNorm(width)
        self.head=nn.Linear(width,10)

    def forward(self,x):
        return self.head(self.norm(self.blocks(self.embed(x.flatten(1)))))
