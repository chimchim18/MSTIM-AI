import torch
import torch.nn.functional as F

class token_sampler:
    def __init__(self,temperature,top_k,top_p,rng):
        self.temp=float(temperature)
        self.top_k=top_k
        self.top_p=top_p
        self.rng=rng

    def apply_truncation(self,row):
        work=row.clone()
        vsize=work.size(-1)
        if self.top_k is not None:
            k_cap=max(1,min(int(self.top_k),vsize))
            k_min=torch.topk(work,k_cap).values[-1]
            work=torch.where(work<k_min,torch.full_like(work,float("-inf")),work)
        if self.top_p is not None and self.top_p<1.0:
            desc_vals,desc_idx=torch.sort(work,descending=True)
            desc_probs=F.softmax(desc_vals,dim=-1)
            running=torch.cumsum(desc_probs,dim=-1)
            over=torch.nonzero(running>=self.top_p,as_tuple=False)
            keep_n=int(over[0].item())+1 if over.numel()>0 else 1
            p_min=desc_vals[keep_n-1]
            work=torch.where(work<p_min,torch.full_like(work,float("-inf")),work)
        return work

    def draw_token(self,last_logits):
        if self.temp<=0.0:
            return int(torch.argmax(last_logits).item())
        scaled=last_logits/self.temp
        pruned=self.apply_truncation(scaled)
        dist=F.softmax(pruned,dim=-1)
        picked=torch.multinomial(dist,num_samples=1,generator=self.rng)
        return int(picked.item())

    def decode_cached(self,model,seed_ids,steps,device):
        out=list(seed_ids)
        ctx=torch.tensor([out],dtype=torch.long,device=device)
        logits,_,cache=model(ctx,use_cache=True)
        tok=self.draw_token(logits[0,-1])
        out.append(tok)
        for _ in range(steps-1):
            cur=torch.tensor([[tok]],dtype=torch.long,device=device)
            logits,_,cache=model(cur,past_kv=cache,use_cache=True)
            tok=self.draw_token(logits[0,-1])
            out.append(tok)
        return out

    def decode_plain(self,model,seed_ids,steps,device):
        out=list(seed_ids)
        for _ in range(steps):
            ctx=torch.tensor([out],dtype=torch.long,device=device)
            logits,_=model(ctx)
            tok=self.draw_token(logits[0,-1])
            out.append(tok)
        return out

@torch.no_grad()
def generate(
    model,
    prompt_ids:list[int],
    max_new_tokens:int,
    temperature:float=1.0,
    top_k:int|None=None,
    top_p:float|None=None,
    use_cache:bool=True,
    seed:int|None=None,
)->list[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids cannot be empty")
    if max_new_tokens<0:
        raise ValueError("max_new_tokens must be non-negative")
    if len(prompt_ids)+max_new_tokens>model.cfg.block_size:
        raise ValueError("total tokens exceed block_size")
    if max_new_tokens==0:
        return list(prompt_ids)

    device=next(model.parameters()).device
    prev_mode=model.training
    model.eval()

    rng=None
    if seed is not None:
        rng=torch.Generator(device=device)
        rng.manual_seed(int(seed))

    sampler=token_sampler(temperature,top_k,top_p,rng)
    try:
        if use_cache:
            return sampler.decode_cached(model,prompt_ids,max_new_tokens,device)
        return sampler.decode_plain(model,prompt_ids,max_new_tokens,device)
    finally:
        if prev_mode:
            model.train()
