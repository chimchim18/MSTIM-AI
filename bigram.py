import csv
import math
import os
import time
import urllib.request
import torch
from tokenizer import BPETokenizer

DATA_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
RUNS_COLS=[
    "run_id","n_layer","n_head","n_embd","block_size","max_lr","max_steps",
    "ablation","params","best_val_loss","final_train_loss","tokens_per_sec","wall_time_min"
]

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

def load_ids(tok,train_txt,val_txt,cache_dir="data/cache"):
    os.makedirs(cache_dir,exist_ok=True)
    tr_p=os.path.join(cache_dir,f"train_v{tok.vocab_size}.pt")
    va_p=os.path.join(cache_dir,f"val_v{tok.vocab_size}.pt")
    if os.path.exists(tr_p) and os.path.exists(va_p):
        return torch.load(tr_p).tolist(),torch.load(va_p).tolist()
    tr_ids=tok.encode(train_txt)
    va_ids=tok.encode(val_txt)
    torch.save(torch.tensor(tr_ids,dtype=torch.long),tr_p)
    torch.save(torch.tensor(va_ids,dtype=torch.long),va_p)
    return tr_ids,va_ids

class BigramModel:
    def __init__(self,vocab_size):
        self.vocab_size=vocab_size
        self.pair_counts=[[0]*vocab_size for _ in range(vocab_size)]
        self.prev_counts=[0]*vocab_size
        self.fitted=False

    def fit(self,ids):
        v=self.vocab_size
        self.pair_counts=[[0]*v for _ in range(v)]
        self.prev_counts=[0]*v
        for i in range(len(ids)-1):
            a=ids[i]
            b=ids[i+1]
            if 0<=a<v and 0<=b<v:
                self.pair_counts[a][b]+=1
                self.prev_counts[a]+=1
        self.fitted=True

    def nll_per_token(self,ids):
        if not self.fitted:
            raise RuntimeError("bigram model not fitted")
        if len(ids)<2:
            return 0.0
        v=self.vocab_size
        total_nll=0.0
        count=0
        for i in range(len(ids)-1):
            a=ids[i]
            b=ids[i+1]
            if 0<=a<v and 0<=b<v:
                prob=(self.pair_counts[a][b]+1)/(self.prev_counts[a]+v)
                total_nll+=-math.log(prob)
                count+=1
        return total_nll/max(1,count)

    def num_params(self):
        return self.vocab_size*self.vocab_size

def write_runs_row(row,path="runs.csv"):
    exists=os.path.exists(path)
    with open(path,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=RUNS_COLS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def run_bigram(vocab_size=1024):
    t0=time.time()
    txt=load_text()
    train_txt,val_txt=get_splits(txt)
    tok=get_tok(train_txt,vocab_size=vocab_size)
    train_ids,val_ids=load_ids(tok,train_txt,val_txt)

    model=BigramModel(tok.vocab_size)
    model.fit(train_ids)
    train_loss=model.nll_per_token(train_ids)
    val_loss=model.nll_per_token(val_ids)
    elapsed=max(1e-6,time.time()-t0)

    print(f"bigram val_loss={val_loss:.6f} nats/token")

    row={
        "run_id":"bigram",
        "n_layer":0,
        "n_head":0,
        "n_embd":0,
        "block_size":1,
        "max_lr":0,
        "max_steps":0,
        "ablation":"add_one_bigram",
        "params":model.num_params(),
        "best_val_loss":f"{val_loss:.6f}",
        "final_train_loss":f"{train_loss:.6f}",
        "tokens_per_sec":f"{len(train_ids)/elapsed:.2f}",
        "wall_time_min":f"{elapsed/60.0:.4f}",
    }
    write_runs_row(row)
    return val_loss

if __name__=="__main__":
    run_bigram()
