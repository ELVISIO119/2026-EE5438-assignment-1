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

class MixerBlock(nn.Module):
    def __init__(self,tokens,width,hidden,dropout):
        super().__init__()
        self.token_norm=nn.LayerNorm(width)
        self.token=nn.Sequential(nn.Linear(tokens,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,tokens))
        self.channel_norm=nn.LayerNorm(width)
        self.channel=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,width))

    def forward(self,x):
        x=x+self.token(self.token_norm(x).transpose(1,2)).transpose(1,2)
        return x+self.channel(self.channel_norm(x))

class MLPMixer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.patch=cfg.get('patch',4)
        assert 28%self.patch==0
        width=cfg['width']; tokens=(28//self.patch)**2
        self.embed=nn.Linear(self.patch**2,width)
        hidden=cfg.get('channel_hidden',2*width)
        self.blocks=nn.Sequential(*[MixerBlock(tokens,width,hidden,cfg['dropout']) for _ in range(cfg['depth'])])
        self.norm=nn.LayerNorm(width)
        self.spatial_head=cfg.get('spatial_head',False)
        self.head=nn.Linear(tokens*width if self.spatial_head else width,10)

    def forward(self,x):
        n=len(x); p=self.patch; side=28//p
        x=x.reshape(n,1,side,p,side,p).permute(0,2,4,1,3,5).reshape(n,side*side,p*p)
        x=self.norm(self.blocks(self.embed(x)))
        return self.head(x.flatten(1) if self.spatial_head else x.mean(1))


class PyramidMLP(nn.Module):
    """Dense 14x14 -> 7x7 token hierarchy with optional feature-level fusion."""
    def __init__(self,cfg):
        super().__init__()
        width=cfg['width']
        self.fuse=cfg.get('multiscale_fusion',True)
        self.embed=nn.Linear(4,width)
        self.fine=nn.Sequential(*[MixerBlock(196,width,2*width,cfg['dropout']) for _ in range(cfg['depth'])])
        self.fine_norm=nn.LayerNorm(width) if self.fuse else nn.Identity()
        self.merge=nn.Sequential(nn.LayerNorm(4*width),nn.Linear(4*width,2*width))
        self.coarse=nn.Sequential(*[MixerBlock(49,2*width,4*width,cfg['dropout']) for _ in range(cfg['depth'])])
        self.coarse_norm=nn.LayerNorm(2*width)
        # Changing head width must not change the subsequent training RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.head=nn.Linear(3*width if self.fuse else 2*width,10)

    @staticmethod
    def merge_tokens(x):
        # Each output concatenates one adjacent 2x2 group in raster order.
        n,_,width=x.shape
        return x.reshape(n,7,2,7,2,width).permute(0,1,3,2,4,5).reshape(n,49,4*width)

    def forward(self,x,return_features=False):
        n=len(x)
        tokens=x.reshape(n,1,14,2,14,2).permute(0,2,4,1,3,5).reshape(n,196,4)
        fine=self.fine(self.embed(tokens))
        fine_features=self.fine_norm(fine).mean(1)
        coarse=self.coarse_norm(self.coarse(self.merge(self.merge_tokens(fine)))).mean(1)
        features=torch.cat((fine_features,coarse),1) if self.fuse else coarse
        logits=self.head(features)
        return (logits,fine_features) if return_features else logits
