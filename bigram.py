from collections import Counter
import csv
import math
import os
import time
from train import CorpusStore, RUNS_HEADER

class BigramBaseline:
    def __init__(self,vocab_size):
        self.v=vocab_size
        self.trans=[0]*(vocab_size*vocab_size)
        self.totals=[0]*vocab_size
        self.ready=False

    def fit(self,seq):
        v=self.v
        self.trans=[0]*(v*v)
        self.totals=[0]*v
        pairs=Counter(zip(seq,seq[1:]))
        for (a,b),cnt in pairs.items():
            if 0<=a<v and 0<=b<v:
                self.trans[a*v+b]=cnt
                self.totals[a]+=cnt
        self.ready=True

    def loss_nats(self,seq):
        if not self.ready:
            raise RuntimeError("model not fitted")
        if len(seq)<2:
            return 0.0
        v=self.v
        nll=0.0
        steps=0
        for a,b in zip(seq,seq[1:]):
            if 0<=a<v and 0<=b<v:
                p=(self.trans[a*v+b]+1)/(self.totals[a]+v)
                nll-=math.log(p)
                steps+=1
        return nll/max(1,steps)

    def num_params(self):
        return self.v*self.v

def evaluate_baseline(vocab_size=1024):
    t0=time.time()
    tok,tr_tensor,va_tensor=CorpusStore().prepare(vocab_size)
    tr_ids=tr_tensor.tolist()
    va_ids=va_tensor.tolist()

    bg=BigramBaseline(tok.vocab_size)
    bg.fit(tr_ids)
    tr_nll=bg.loss_nats(tr_ids)
    va_nll=bg.loss_nats(va_ids)
    dt=max(1e-6,time.time()-t0)

    print(f"bigram validation loss: {va_nll:.6f} nats/token")

    row={
        "run_id":"bigram",
        "n_layer":0,
        "n_head":0,
        "n_embd":0,
        "block_size":1,
        "max_lr":0,
        "max_steps":0,
        "ablation":"add_one_bigram",
        "params":bg.num_params(),
        "best_val_loss":f"{va_nll:.6f}",
        "final_train_loss":f"{tr_nll:.6f}",
        "tokens_per_sec":f"{len(tr_ids)/dt:.2f}",
        "wall_time_min":f"{dt/60.0:.4f}",
    }
    exists=os.path.exists("runs.csv")
    with open("runs.csv","a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=RUNS_HEADER)
        if not exists:
            w.writeheader()
        w.writerow(row)
    return va_nll

if __name__=="__main__":
    evaluate_baseline()
