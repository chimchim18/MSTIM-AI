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

CORPUS_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
RUNS_HEADER=[
    "run_id","n_layer","n_head","n_embd","block_size","max_lr","max_steps",
    "ablation","params","best_val_loss","final_train_loss","tokens_per_sec","wall_time_min"
]

class corpus_store:
    def __init__(self,raw_path="data/input.txt",tok_path="tokenizer.json",cache_dir="data/cache"):
        self.raw_path=raw_path
        self.tok_path=tok_path
        self.cache_dir=cache_dir

    def read_corpus(self):
        os.makedirs(os.path.dirname(self.raw_path) or ".",exist_ok=True)
        if not os.path.exists(self.raw_path):
            urllib.request.urlretrieve(CORPUS_URL,self.raw_path)
        with open(self.raw_path,"r",encoding="utf-8") as f:
            raw=f.read()
        pivot=int(0.9*len(raw))
        return raw[:pivot],raw[pivot:]

    def prepare(self,vocab_size=1024):
        tr_txt,va_txt=self.read_corpus()
        if os.path.exists(self.tok_path):
            tok=BPETokenizer.load(self.tok_path)
            if tok.vocab_size!=vocab_size:
                tok=BPETokenizer()
                tok.train(tr_txt,vocab_size)
                tok.save(self.tok_path)
        else:
            tok=BPETokenizer()
            tok.train(tr_txt,vocab_size)
            tok.save(self.tok_path)

        os.makedirs(self.cache_dir,exist_ok=True)
        tr_file=os.path.join(self.cache_dir,f"tr_{tok.vocab_size}.pt")
        va_file=os.path.join(self.cache_dir,f"va_{tok.vocab_size}.pt")
        if os.path.exists(tr_file) and os.path.exists(va_file):
            return tok,torch.load(tr_file),torch.load(va_file)

        tr_ids=torch.tensor(tok.encode(tr_txt),dtype=torch.long)
        va_ids=torch.tensor(tok.encode(va_txt),dtype=torch.long)
        torch.save(tr_ids,tr_file)
        torch.save(va_ids,va_file)
        return tok,tr_ids,va_ids

class model_trainer:
    def __init__(self,cfg):
        self.cfg=cfg
        self.run_id=str(cfg.get("run_id","run"))
        self.ablation=str(cfg.get("ablation","none"))
        self.seed=int(cfg.get("seed",42))
        self.block_size=int(cfg.get("block_size",128))
        self.batch_size=int(cfg.get("batch_size",32))
        self.max_steps=int(cfg.get("max_steps",3000))
        self.max_lr=float(cfg.get("max_lr",1e-3))
        self.warmup_steps=int(cfg.get("warmup_steps",200))
        self.eval_interval=int(cfg.get("eval_interval",100))
        self.device="cuda" if torch.cuda.is_available() else "cpu"

    def _lr_at(self,idx):
        floor_lr=0.1*self.max_lr
        if self.warmup_steps>0 and idx<self.warmup_steps:
            return self.max_lr*(idx+1)/self.warmup_steps
        if idx>=self.max_steps:
            return floor_lr
        span=max(1,self.max_steps-self.warmup_steps)
        prog=(idx-self.warmup_steps)/span
        return floor_lr+0.5*(1.0+math.cos(math.pi*prog))*(self.max_lr-floor_lr)

    def _draw_batch(self,seq,rng):
        limit=len(seq)-self.block_size-1
        starts=torch.randint(0,limit,(self.batch_size,),generator=rng)
        offsets=torch.arange(self.block_size)
        grid=starts.unsqueeze(1)+offsets.unsqueeze(0)
        return seq[grid].to(self.device),seq[grid+1].to(self.device)

    def _build_optimizer(self,model):
        mat_params=[]
        vec_params=[]
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim>=2:
                mat_params.append(p)
            else:
                vec_params.append(p)
        return torch.optim.AdamW(
            [{"params":mat_params,"weight_decay":0.1},{"params":vec_params,"weight_decay":0.0}],
            lr=self.max_lr,
            betas=(0.9,0.95),
        )

    @torch.no_grad()
    def _estimate_val(self,model,val_ids,batches=50):
        model.eval()
        val_rng=torch.Generator()
        val_rng.manual_seed(98765)
        acc=0.0
        for _ in range(batches):
            x,y=self._draw_batch(val_ids,val_rng)
            _,loss=model(x,y)
            acc+=loss.item()
        model.train()
        return acc/batches

    def execute(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        rng=torch.Generator()
        rng.manual_seed(self.seed)

        tok,train_ids,val_ids=corpus_store().prepare(int(self.cfg.get("vocab_size",1024)))
        mcfg=GPTConfig(
            vocab_size=tok.vocab_size,
            block_size=self.block_size,
            n_layer=int(self.cfg.get("n_layer",4)),
            n_head=int(self.cfg.get("n_head",4)),
            n_embd=int(self.cfg.get("n_embd",128)),
            dropout=float(self.cfg.get("dropout",0.1)),
        )
        model=GPT(mcfg).to(self.device)
        opt=self._build_optimizer(model)

        os.makedirs("logs",exist_ok=True)
        os.makedirs("checkpoints",exist_ok=True)
        log_file=os.path.join("logs",f"{self.run_id}.csv")
        ckpt_file=os.path.join("checkpoints",f"{self.run_id}_best.pt")

        best_val=float("inf")
        last_train=0.0
        seen=0
        start_t=time.time()

        with open(log_file,"w",newline="",encoding="utf-8") as fh:
            writer=csv.writer(fh)
            writer.writerow(["step","train_loss","val_loss","lr","tokens_seen"])

            for step in range(1,self.max_steps+1):
                cur_lr=self._lr_at(step-1)
                for group in opt.param_groups:
                    group["lr"]=cur_lr

                xb,yb=self._draw_batch(train_ids,rng)
                _,loss=model(xb,yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()

                seen+=xb.numel()
                last_train=loss.item()

                if step%self.eval_interval==0 or step==self.max_steps:
                    val_loss=self._estimate_val(model,val_ids,50)
                    writer.writerow([step,f"{last_train:.6f}",f"{val_loss:.6f}",f"{cur_lr:.8f}",seen])
                    fh.flush()
                    if val_loss<best_val:
                        best_val=val_loss
                        torch.save(model.state_dict(),ckpt_file)
                    print(f"[{self.run_id}] step={step}/{self.max_steps} train={last_train:.4f} val={val_loss:.4f} lr={cur_lr:.2e}")

        dt=max(1e-6,time.time()-start_t)
        summary={
            "run_id":self.run_id,
            "n_layer":mcfg.n_layer,
            "n_head":mcfg.n_head,
            "n_embd":mcfg.n_embd,
            "block_size":mcfg.block_size,
            "max_lr":self.max_lr,
            "max_steps":self.max_steps,
            "ablation":self.ablation,
            "params":model.num_params(),
            "best_val_loss":f"{best_val:.6f}",
            "final_train_loss":f"{last_train:.6f}",
            "tokens_per_sec":f"{seen/dt:.2f}",
            "wall_time_min":f"{dt/60.0:.4f}",
        }
        has_runs=os.path.exists("runs.csv")
        with open("runs.csv","a",newline="",encoding="utf-8") as rf:
            rw=csv.DictWriter(rf,fieldnames=RUNS_HEADER)
            if not has_runs:
                rw.writeheader()
            rw.writerow(summary)

def parse_args_and_run():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    with open(args.config,"r",encoding="utf-8") as f:
        cfg=json.load(f)
    model_trainer(cfg).execute()

if __name__=="__main__":
    parse_args_and_run()
