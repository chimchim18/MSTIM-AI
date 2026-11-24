import json
from typing import Dict, List, Tuple

class BPETokenizer:
    def __init__(self)->None:
        self.merges:List[Tuple[int, int]]=[]
    @staticmethod
    def _tally_pairs(sequence:List[int])->Dict[Tuple[int, int],int]:
        """here we count occurrences of every adjacent pair in sequence"""
        tally:Dict[Tuple[int, int], int]= {}
        limit =len(sequence) - 1
        idx =0
        while idx<limit:
            pair=(sequence[idx], sequence[idx + 1])
            tally[pair] = tally.get(pair, 0) + 1
            idx+=1
        return tally

    @staticmethod
    def _apply_merge(
        sequence: List[int],target_pair: Tuple[int, int], new_id: int
    )->List[int]:
        """and here we replace non-overlapping occurrences of target_pair left to right."""
        first, second =target_pair
        updated: List[int]= []
        idx =0
        n =len(sequence)
        while idx<n:
            if idx+1<n and sequence[idx]==first and sequence[idx+1] == second:
                updated.append(new_id)
                idx +=2
            else:
                updated.append(sequence[idx])
                idx +=1
        return updated
    def train(self, text: str, vocab_size: int)->None:
        """now learning bpe merges from utf-8 bytes up to vocab_size"""
        self.merges =[]
        stream=list(text.encode("utf-8"))
        total_merges=max(0, vocab_size - 256)
        for step in range(total_merges):
            if len(stream)<2:
                break
            tally = self._tally_pairs(stream)
            if not tally:
                break
          """now here we pick highest frequesncy and break ties by smallest first id first and tgen smallest second id """
            chosen_pair= min(tally.keys(), key=lambda p: (-tally[p], p[0], p[1]))
            merged_id= 256 + step
            stream= self._apply_merge(stream, chosen_pair, merged_id)
            self.merges.append(chosen_pair)

    def encode(self, text: str)->List[int]:
        """Now here we enncode text by applying learned merges in training order"""
        stream= list(text.encode("utf-8"))
        for step, pair in enumerate(self.merges):
            if len(stream)< 2:
                break
            stream=self._apply_merge(stream, pair, 256 + step)
        return stream
    def decode(self, ids: List[int])->str:
        """And here decode token ids back into a utf-8 string"""
        byte_table: Dict[int, bytes] ={b: bytes([b]) for b in range(256)}
        for step, (left, right) in enumerate(self.merges):
            byte_table[256 + step]= byte_table[left] + byte_table[right]

        raw= bytearray()
        for token_id in ids:
            chunk =byte_table.get(token_id)
            if chunk is not None:
                raw.extend(chunk)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str)->None:
        """Then we persist learned merges to a json file"""
        serialized = {"version": 1, "merges": [[a, b] for a, b in self.merges]}
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(serialized, fp, indent=2)

    @classmethod
    def load(cls, path: str)->"BPETokenizer":
        """And finally we restore a trained tokenizer from a json file."""
        with open(path,"r",encoding="utf-8") as fp:
            serialized=json.load(fp)
        tok=cls()
        tok.merges=[(int(a), int(b)) for a, b in serialized["merges"]]
        return tok
