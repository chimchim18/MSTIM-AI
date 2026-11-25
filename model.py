import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class modelspec:
    vocab_size:int
    block_size:int=128
    n_layer:int=4
    n_head:int=4
    n_embd:int=128
    dropout:float=0.1
    tie_w:bool=True

    def __post_init__(self):
        if self.n_embd%self.n_head!=0:
            raise ValueError("n_embd must divide by n_head")

class mhca(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.num_heads=cfg.n_head
        self.emb_dim=cfg.n_embd
        self.head_dim=cfg.n_embd//cfg.n_head

        self.qkv=nn.Linear(cfg.n_embd,3*cfg.n_embd)
        self.proj=nn.Linear(cfg.n_embd,cfg.n_embd)
        self.proj._res_scale=True

        self.attn_drop=nn.Dropout(cfg.dropout)
        self.resid_drop=nn.Dropout(cfg.dropout)

        tri=torch.tril(torch.ones(cfg.block_size,cfg.block_size))
        self.register_buffer("causal_mask",tri.view(1,1,cfg.block_size,cfg.block_size))

    def split(self,t):
        b,seq,_=t.size()
        return t.view(b,seq,self.num_heads,self.head_dim).transpose(1,2)

    def fwd(self,x):
        b,seq,c=x.size()
        q0,k0,v0=self.qkv(x).split(self.emb_dim,dim=-1)
        q,k,v=self.split(q0),self.split(k0),self.split(v0)

        sc=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        sc=sc.masked_fill(self.causal_mask[:,:,:seq,:seq]==0,float("-inf"))
        w=self.attn_drop(F.softmax(sc,dim=-1))

        out=(w@v).transpose(1,2).contiguous().view(b,seq,c)
        return self.resid_drop(self.proj(out))

    def forward(self,x):
        return self.fwd(x)

class ffn(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.hidden_dim=4*cfg.n_embd
        self.up=nn.Linear(cfg.n_embd,self.hidden_dim)
        self.act=nn.GELU()
        self.down=nn.Linear(self.hidden_dim,cfg.n_embd)
        self.down._res_scale=True
        self.drop=nn.Dropout(cfg.dropout)

    def fwd(self,x):
        h=self.act(self.up(x))
        return self.drop(self.down(h))

    def forward(self,x):
        return self.fwd(x)

class tblock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg.n_embd)
        self.attn=mhca(cfg)
        self.ln2=nn.LayerNorm(cfg.n_embd)
        self.mlp=ffn(cfg)

    def fwd(self,x):
        x=x+self.attn(self.ln1(x))
        x=x+self.mlp(self.ln2(x))
        return x

    def forward(self,x):
        return self.fwd(x)

class minigpt(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.tok_emb=nn.Embedding(cfg.vocab_size,cfg.n_embd)
        self.pos_emb=nn.Embedding(cfg.block_size,cfg.n_embd)
        self.drop=nn.Dropout(cfg.dropout)

        self.blocks=nn.ModuleList([tblock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f=nn.LayerNorm(cfg.n_embd)
        self.head=nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False)

        if cfg.tie_w:
            self.head.weight=self.tok_emb.weight

        self.apply(self.initw)

    def initw(self,mod):
        rscale=1.0/math.sqrt(2.0*max(1,self.cfg.n_layer))
        if isinstance(mod,nn.Linear):
            sd=0.02*rscale if getattr(mod,"_res_scale",False) else 0.02
            nn.init.normal_(mod.weight,mean=0.0,std=sd)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod,nn.Embedding):
            nn.init.normal_(mod.weight,mean=0.0,std=0.02)
        elif isinstance(mod,nn.LayerNorm):
            nn.init.ones_(mod.weight)
            nn.init.zeros_(mod.bias)

    def pcount(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fwd(self,idx,targets=None):
        b,seq=idx.size()
        if seq>self.cfg.block_size:
            raise ValueError("seq longer than block_size")

        pos=torch.arange(0,seq,device=idx.device,dtype=torch.long)
        x=self.drop(self.tok_emb(idx)+self.pos_emb(pos))

        for blk in self.blocks:
            x=blk(x)

        logits=self.head(self.ln_f(x))
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss

    def forward(self,idx,targets=None):
        return self.fwd(idx,targets)

def chimnaz(vocab_size,block_size=128,n_layer=4,n_head=4,n_embd=128,dropout=0.1,tie_w=True):
    cfg=modelspec(vocab_size=vocab_size,block_size=block_size,n_layer=n_layer,n_head=n_head,n_embd=n_embd,dropout=dropout,tie_w=tie_w)
    return minigpt(cfg)
