import torch
import torch.nn.functional as F

def _filter_logits(logits,top_k,top_p):
    out=logits.clone()
    vocab=out.size(-1)
    if top_k is not None:
        k=max(1,min(int(top_k),vocab))
        thresh=torch.topk(out,k).values[-1]
        out=out.masked_fill(out<thresh,float("-inf"))
    if top_p is not None and top_p<1.0:
        sorted_vals,sorted_idx=torch.sort(out,descending=True)
        probs=F.softmax(sorted_vals,dim=-1)
        cum_probs=torch.cumsum(probs,dim=-1)
        drop_mask=cum_probs>top_p
        drop_mask[1:]=drop_mask[:-1].clone()
        drop_mask[0]=False
        sorted_vals=sorted_vals.masked_fill(drop_mask,float("-inf"))
        out=torch.full_like(out,float("-inf"))
        out.scatter_(0,sorted_idx,sorted_vals)
    return out

def _pick_next(logits,temperature,top_k,top_p,rng):
    if temperature<=0.0:
        return int(torch.argmax(logits).item())
    scaled=logits/float(temperature)
    filtered=_filter_logits(scaled,top_k,top_p)
    probs=F.softmax(filtered,dim=-1)
    idx=torch.multinomial(probs,num_samples=1,generator=rng)
    return int(idx.item())

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
    total_target=len(prompt_ids)+max_new_tokens
    if total_target>model.cfg.block_size:
        raise ValueError("total tokens exceed block_size")
    if max_new_tokens==0:
        return list(prompt_ids)

    device=next(model.parameters()).device
    was_training=model.training
    model.eval()

    rng=None
    if seed is not None:
        rng=torch.Generator(device=device)
        rng.manual_seed(int(seed))

    seq=list(prompt_ids)
    try:
        if use_cache:
            inp=torch.tensor([seq],dtype=torch.long,device=device)
            logits,_,past_kv=model(inp,use_cache=True)
            nxt=_pick_next(logits[0,-1],temperature,top_k,top_p,rng)
            seq.append(nxt)
            for _ in range(max_new_tokens-1):
                step_inp=torch.tensor([[nxt]],dtype=torch.long,device=device)
                logits,_,past_kv=model(step_inp,past_kv=past_kv,use_cache=True)
                nxt=_pick_next(logits[0,-1],temperature,top_k,top_p,rng)
                seq.append(nxt)
        else:
            for _ in range(max_new_tokens):
                inp=torch.tensor([seq],dtype=torch.long,device=device)
                logits,_=model(inp)
                nxt=_pick_next(logits[0,-1],temperature,top_k,top_p,rng)
                seq.append(nxt)
    finally:
        if was_training:
            model.train()

    return seq
