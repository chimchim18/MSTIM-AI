import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size:int=1024
    block_size:int=128
    n_layer:int=4
    n_head:int=4
    n_embd:int=128
    dropout:float=0.1

    def __post_init__(self):
        if self.n_embd%self.n_head!=0:
            raise ValueError("n_embd must divide by n_head")

class mhca(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.num_heads=cfg.n_head
        self.emb_dim=cfg.n_embd
        self.head_dim=cfg.n_embd//cfg.n_head
        self.qkv=nn.Linear(cfg.n_embd,3*cfg.n_embd,bias=False)
        self.proj=nn.Linear(cfg.n_embd,cfg.n_embd,bias=False)
        self.attn_drop=nn.Dropout(cfg.dropout)
        tri=torch.tril(torch.ones(cfg.block_size,cfg.block_size))
        self.register_buffer("causal_mask",tri.view(1,1,cfg.block_size,cfg.block_size),persistent=False)

    def split(self,t):
        b,seq,_=t.size()
        return t.view(b,seq,self.num_heads,self.head_dim).transpose(1,2)

    def forward(self,x,past_kv=None,use_cache=False):
        b,seq,c=x.size()
        q0,k0,v0=self.qkv(x).split(self.emb_dim,dim=-1)
        q,k,v=self.split(q0),self.split(k0),self.split(v0)
        past_len=0
        if past_kv is not None:
            pk,pv=past_kv
            past_len=pk.size(2)
            k=torch.cat([pk,k],dim=2)
            v=torch.cat([pv,v],dim=2)
        tot_len=k.size(2)
        sc=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        sc=sc.masked_fill(self.causal_mask[:,:,past_len:past_len+seq,:tot_len]==0,float("-inf"))
        w=self.attn_drop(F.softmax(sc,dim=-1))
        out=(w@v).transpose(1,2).contiguous().view(b,seq,c)
        out=self.proj(out)
        if use_cache:
            return out,(k,v)
        return out

class ffn(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.hidden_dim=4*cfg.n_embd
        self.up=nn.Linear(cfg.n_embd,self.hidden_dim,bias=False)
        self.down=nn.Linear(self.hidden_dim,cfg.n_embd,bias=False)
        self.drop=nn.Dropout(cfg.dropout)

    def forward(self,x):
        h=F.gelu(self.up(x),approximate="tanh")
        return self.drop(self.down(h))

class tblock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg.n_embd)
        self.attn=mhca(cfg)
        self.ln2=nn.LayerNorm(cfg.n_embd)
        self.mlp=ffn(cfg)

    def forward(self,x,past_kv=None,use_cache=False):
        if use_cache:
            h,next_kv=self.attn(self.ln1(x),past_kv=past_kv,use_cache=True)
            x=x+h
            x=x+self.mlp(self.ln2(x))
            return x,next_kv
        x=x+self.attn(self.ln1(x))
        x=x+self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.cfg=config
        cfg=config
        self.tok_emb=nn.Embedding(cfg.vocab_size,cfg.n_embd)
        self.pos_emb=nn.Embedding(cfg.block_size,cfg.n_embd)
        self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([tblock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f=nn.LayerNorm(cfg.n_embd)
        self.head=nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False)
        self.head.weight=self.tok_emb.weight
        self.apply(self.initw)

    def initw(self,mod):
        if isinstance(mod,nn.Linear):
            nn.init.normal_(mod.weight,mean=0.0,std=0.02)
        elif isinstance(mod,nn.Embedding):
            nn.init.normal_(mod.weight,mean=0.0,std=0.02)
        elif isinstance(mod,nn.LayerNorm):
            nn.init.ones_(mod.weight)
            nn.init.zeros_(mod.bias)

    def num_params(self)->int:
        return sum(p.numel() for p in self.parameters())

    def forward(self,idx,targets=None,past_kv=None,use_cache=False):
        b,seq=idx.size()
        past_len=past_kv[0][0].size(2) if past_kv is not None else 0
        if past_len+seq>self.cfg.block_size:
            raise ValueError("seq longer than block_size")
        pos=torch.arange(past_len,past_len+seq,device=idx.device,dtype=torch.long)
        x=self.drop(self.tok_emb(idx)+self.pos_emb(pos))
        next_kvs=[] if use_cache else None
        for i,blk in enumerate(self.blocks):
            layer_past=past_kv[i] if past_kv is not None else None
            if use_cache:
                x,kv=blk(x,past_kv=layer_past,use_cache=True)
                next_kvs.append(kv)
            else:
                x=blk(x)
        logits=self.head(self.ln_f(x))
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        if use_cache:
            return logits,loss,next_kvs
        return logits,loss
