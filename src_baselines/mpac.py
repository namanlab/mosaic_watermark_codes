"""MPAC-style multi-bit watermark (Yoo, Ahn, Kwak, NAACL 2024), radix 2.

Position allocation: a context-keyed PRF assigns every position to one payload
bit. A second PRF colors the vocabulary into two balanced halves (gamma = 1/2)
per context; the half matching the assigned bit's value receives the +delta
bias. Extraction is a per-bit majority vote over the positions allocated to
that bit, so its cost is O(T + k).
"""
import numpy as np

from .common import PRF


class MPACWatermark:
    name = "mpac"
    feasible_ks = [16, 20, 24, 32, 48]

    def __init__(self, secret: str, k_bits: int, vocab_size: int):
        self.k = k_bits
        self.vocab_size = vocab_size
        self.prf_pos = PRF(secret, "mpac.position", vocab_size)
        self.prf_col = PRF(secret, "mpac.color", vocab_size)

    def reconfigure(self, k_bits: int):
        self.k = k_bits

    def _bit_index(self, prev_id: int) -> int:
        return self.prf_pos.scalar(prev_id) % self.k

    def _colors(self, prev_id: int) -> np.ndarray:
        return (self.prf_col.over_vocab(prev_id) & np.uint64(1)).astype(np.int8)

    # ---------------------------------------------------------- embedding ---
    def favored_set(self, prev_id: int, pos: int, message) -> np.ndarray:
        i = self._bit_index(prev_id)
        return self._colors(prev_id) == message[i]

    # ---------------------------------------------------------- detection ---
    def detect(self, gen_ids, prompt_ids):
        """Majority vote per bit. Model-free: tokenizer ids and the key only."""
        votes = np.zeros((self.k, 2), dtype=np.int64)
        prev = prompt_ids[-1]
        for tok in gen_ids:
            i = self._bit_index(prev)
            if tok < self.vocab_size:
                col = int(self._colors(prev)[tok])
                votes[i, col] += 1
            prev = tok
        decoded = (votes[:, 1] > votes[:, 0]).astype(int).tolist()
        agree = int(votes[np.arange(self.k), decoded].sum())
        total = int(votes.sum())
        return {"message": decoded,
                "agreement": agree / max(total, 1),
                "num_tokens": len(gen_ids)}
