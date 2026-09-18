import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,width,activation,norm,dropout):
        super().__init__()
        normalizer=nn.LayerNorm if norm=='layer' else nn.BatchNorm1d
        act={'gelu':nn.GELU,'silu':nn.SiLU,'relu':nn.ReLU}[activation]
        self.norm=normalizer(width)
        self.fc1=nn.Linear(width,2*width)
        self.act=act()
        self.drop=nn.Dropout(dropout)
        self.fc2=nn.Linear(2*width,width)

    def forward(self,x):
        return x+self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))

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
