import json

class bpetok:
    def __init__(self):
        self.merges=[]

    @property
    def vocab_size(self):
        return 256+len(self.merges)

    @staticmethod
    def _counts(seq):
        # count adjacent pairs in seq
        cnt={}
        for i in range(len(seq)-1):
            p=(seq[i],seq[i+1])
            cnt[p]=cnt.get(p,0)+1
        return cnt

    @staticmethod
    def _merge(seq,pair,tid):
        # replace non-overlapping pair left-to-right with tid
        a,b=pair
        out=[]
        i=0
        n=len(seq)
        while i<n:
            if i+1<n and seq[i]==a and seq[i+1]==b:
                out.append(tid)
                i+=2
            else:
                out.append(seq[i])
                i+=1
        return out

    def train(self,text,vocab_size):
        # learn bpe merges on utf-8 bytes
        self.merges=[]
        ids=list(text.encode("utf-8"))
        steps=max(0,vocab_size-256)
        for k in range(steps):
            if len(ids)<2:
                break
            cnt=self._counts(ids)
            if not cnt:
                break
            # highest count first, tie-break by smaller ids
            best=min(cnt,key=lambda p:(-cnt[p],p[0],p[1]))
            ids=self._merge(ids,best,256+k)
            self.merges.append(best)

    def encode(self,text):
        ids=list(text.encode("utf-8"))
        for k,pair in enumerate(self.merges):
            if len(ids)<2:
                break
            ids=self._merge(ids,pair,256+k)
        return ids

    def decode(self,ids):
        vocab={i:bytes([i]) for i in range(256)}
        for k,(a,b) in enumerate(self.merges):
            vocab[256+k]=vocab[a]+vocab[b]
        buf=bytearray()
        for tid in ids:
            piece=vocab.get(tid)
            if piece is not None:
                buf.extend(piece)
        return buf.decode("utf-8",errors="replace")

    def save(self,path):
        data={"v":1,"merges":[[a,b] for a,b in self.merges]}
        with open(path,"w",encoding="utf-8") as f:
            json.dump(data,f,indent=2)

    @classmethod
    def load(cls,path):
        with open(path,"r",encoding="utf-8") as f:
            data=json.load(f)
        tok=cls()
        tok.merges=[(int(a),int(b)) for a,b in data["merges"]]
        return tok


def chimnaz(text,vocab_size=300):
    t=bpetok()
    t.train(text,vocab_size)
    return t
