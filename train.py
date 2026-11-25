import argparse
import csv
import json
import math
import os
import time
import urllib.request
import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

DATA_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def load_text(path="data/input.txt"):
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL,path)
    with open(path,"r",encoding="utf-8") as f:
        return f.read()

def get_splits(text):
    cut=int(0.9*len(text))
    return text[:cut],text[cut:]

def get_tok(train_txt,tok_path="tokenizer.json",vocab_size=1024):
    if os.path.exists(tok_path):
        tok=BPETokenizer.load(tok_path)
        if tok.vocab_size==vocab_size:
            return tok
    tok=BPETokenizer()
    tok.train(train_txt,vocab_size)
    tok.save(tok_path)
    return tok

def load_cached_ids(tok,train_txt,val_txt,cache_dir="data/cache"):
    os.makedirs(cache_dir,exist_ok=True)
    tr_p=os.path.join(cache_dir,f"train_v{tok.vocab_size}.pt")
    va_p=os.path.join(cache_dir,f"val_v{tok.vocab_size}.pt")
    if os.path.exists(tr_p) and os.path.exists(va_p):
        return torch.load(tr_p),torch.load(va_p)
    tr_ids=torch.tensor(tok.encode(train_txt),dtype=torch.long)
    va_ids=torch.tensor(tok.encode(val_txt),dtype=torch.long)
    torch.save(tr_ids,tr_p)
    torch.save(va_ids,va_p)
    return tr_ids,va_ids

def sample_batch(data,block_size,batch_size,gen,device):
    max_start=len(data)-block_size-1
    ix=torch.randint(0,max_start,(batch_size,),generator=gen)
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device),y.to(device)

def get_lr(step,max_steps,warmup_steps,max_lr):
    min_lr=0.1*max_lr
    if warmup_steps>0 and step<warmup_steps:
        return max_lr*(step+1)/warmup_steps
    if step>=max_steps:
        return min_lr
    denom=max(1,max_steps-warmup_steps)
    ratio=(step-warmup_steps)/denom
    coeff=0.5*(1.0+math.cos(math.pi*ratio))
    return min_lr+coeff*(max_lr-min_lr)

def make_opt(model,lr):
    decay=[p for p in model.parameters() if p.requires_grad and p.ndim>=2]
    nodecay=[p for p in model.parameters() if p.requires_grad and p.ndim<2]
    groups=[
        {"params":decay,"weight_decay":0.1},
        {"params":nodecay,"weight_decay":0.0},
    ]
    return torch.optim.AdamW(groups,lr=lr,betas=(0.9,0.95))

@torch.no_grad()
def eval_val(model,val_ids,block_size,batch_size,device,eval_batches=50):
    model.eval()
    g=torch.Generator()
    g.manual_seed(12345)
    total=0.0
    for _ in range(eval_batches):
        xb,yb=sample_batch(val_ids,block_size,batch_size,g,device)
        _,loss=model(xb,yb)
        total+=loss.item()
    model.train()
    return total/eval_batches

def append_run_csv(row,path="runs.csv"):
    cols=[
        "run_id","n_layer","n_head","n_embd","block_size","max_lr","max_steps",
        "ablation","params","best_val_loss","final_train_loss","tokens_per_sec","wall_time_min"
    ]
    exists=os.path.exists(path)
    with open(path,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow(row)

def train_from_cfg(cfg):
    seed=int(cfg.get("seed",42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen=torch.Generator()
    gen.manual_seed(seed)

    device="cuda" if torch.cuda.is_available() else "cpu"
    txt=load_text()
    train_txt,val_txt=get_splits(txt)
    vocab_size=int(cfg.get("vocab_size",1024))
    tok=get_tok(train_txt,vocab_size=vocab_size)
    train_ids,val_ids=load_cached_ids(tok,train_txt,val_txt)

    block_size=int(cfg.get("block_size",128))
    batch_size=int(cfg.get("batch_size",32))
    max_steps=int(cfg.get("max_steps",3000))
    max_lr=float(cfg.get("max_lr",1e-3))
    warmup_steps=int(cfg.get("warmup_steps",200))
    eval_interval=int(cfg.get("eval_interval",100))
    run_id=str(cfg.get("run_id","run"))
    ablation=str(cfg.get("ablation","none"))

    mcfg=GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=block_size,
        n_layer=int(cfg.get("n_layer",4)),
        n_head=int(cfg.get("n_head",4)),
        n_embd=int(cfg.get("n_embd",128)),
        dropout=float(cfg.get("dropout",0.1)),
    )
    model=GPT(mcfg).to(device)
    opt=make_opt(model,max_lr)

    os.makedirs("logs",exist_ok=True)
    os.makedirs("checkpoints",exist_ok=True)
    log_path=os.path.join("logs",f"{run_id}.csv")
    ckpt_path=os.path.join("checkpoints",f"{run_id}_best.pt")

    best_val=float("inf")
    final_train=0.0
    tokens_seen=0
    t0=time.time()

    with open(log_path,"w",newline="",encoding="utf-8") as lf:
        lw=csv.writer(lf)
        lw.writerow(["step","train_loss","val_loss","lr","tokens_seen"])

        for step in range(1,max_steps+1):
            lr=get_lr(step-1,max_steps,warmup_steps,max_lr)
            for pg in opt.param_groups:
                pg["lr"]=lr

            xb,yb=sample_batch(train_ids,block_size,batch_size,gen,device)
            _,loss=model(xb,yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()

            tokens_seen+=xb.numel()
            final_train=loss.item()

            if step%eval_interval==0 or step==max_steps:
                vloss=eval_val(model,val_ids,block_size,batch_size,device,50)
                lw.writerow([step,f"{final_train:.6f}",f"{vloss:.6f}",f"{lr:.8f}",tokens_seen])
                lf.flush()
                if vloss<best_val:
                    best_val=vloss
                    torch.save(model.state_dict(),ckpt_path)
                print(f"[{run_id}] step={step}/{max_steps} train={final_train:.4f} val={vloss:.4f} lr={lr:.2e}")

    elapsed=max(1e-6,time.time()-t0)
    row={
        "run_id":run_id,
        "n_layer":mcfg.n_layer,
        "n_head":mcfg.n_head,
        "n_embd":mcfg.n_embd,
        "block_size":mcfg.block_size,
        "max_lr":max_lr,
        "max_steps":max_steps,
        "ablation":ablation,
        "params":model.num_params(),
        "best_val_loss":f"{best_val:.6f}",
        "final_train_loss":f"{final_train:.6f}",
        "tokens_per_sec":f"{tokens_seen/elapsed:.2f}",
        "wall_time_min":f"{elapsed/60.0:.4f}",
    }
    append_run_csv(row)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    args=p.parse_args()
    with open(args.config,"r",encoding="utf-8") as f:
        cfg=json.load(f)
    train_from_cfg(cfg)

if __name__=="__main__":
    main()
