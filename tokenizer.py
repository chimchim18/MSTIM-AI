# Author: Chimnaz
import json
 
class BPETokenizer:
    def __init__(self)->None:
        self.merges:list[tuple[int,int]]=[]
        self._vocab:dict[int,bytes]={}
        self._build_vocab()
 
    @property
    def vocab_size(self)->int:
        return 256+len(self.merges)
 
    def _build_vocab(self)->None:
        vocab={i:bytes([i]) for i in range(256)}
        for k,(a,b) in enumerate(self.merges):
            vocab[256+k]=vocab[a]+vocab[b]
        self._vocab=vocab
 
    @staticmethod
    def _counts(seq:list[int])->dict[tuple[int,int],int]:
        cnt={}
        for i in range(len(seq)-1):
            p=(seq[i],seq[i+1])
            cnt[p]=cnt.get(p,0)+1
        return cnt
 
    @staticmethod
    def _merge(seq:list[int],pair:tuple[int,int],tid:int)->list[int]:
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
 
    def train(self,text:str,vocab_size:int)->None:
        self.merges=[]
        ids=list(text.encode("utf-8"))
        steps=max(0,vocab_size-256)
        for k in range(steps):
            if len(ids)<2:
                break
            cnt=self._counts(ids)
            if not cnt:
                break
            best=min(cnt,key=lambda p:(-cnt[p],p[0],p[1]))
            ids=self._merge(ids,best,256+k)
            self.merges.append(best)
        self._build_vocab()
 
    def encode(self,text:str)->list[int]:
        ids=list(text.encode("utf-8"))
        for k,pair in enumerate(self.merges):
            if len(ids)<2:
                break
            ids=self._merge(ids,pair,256+k)
        return ids
 
    def decode(self,ids:list[int])->str:
        buf=bytearray()
        for tid in ids:
            piece=self._vocab.get(tid)
            if piece is None:
                raise ValueError(f"unknown token id: {tid}")
            buf.extend(piece)
        return buf.decode("utf-8",errors="replace")
 
    def save(self,path:str)->None:
        data={"v":1,"author":"Chimnaz","merges":[[a,b] for a,b in self.merges]}
        with open(path,"w",encoding="utf-8") as f:
            json.dump(data,f,indent=2)
 
    @classmethod
    def load(cls,path:str)->"BPETokenizer":
        with open(path,"r",encoding="utf-8") as f:
            data=json.load(f)
        tok=cls()
        tok.merges=[(int(a),int(b)) for a,b in data["merges"]]
        tok._build_vocab()
        return tok
