"""Fernandez et al. style multi-bit watermark (WIFS 2023): key shift.

The message selects one of 2^k keyed green-list families: the green list at a
position is a fresh gamma-fraction pseudorandom subset determined jointly by
the context hash and the message value, so the lists for different messages
are independent (this decorrelation is what makes the argmax identifiable).
Embedding biases the selected family's green tokens by +delta. Extraction
scores every candidate message s by counting tokens that are green under s and
returns the argmax.

The defining property of this construction is preserved: extraction enumerates
all 2^k candidates, so its cost is O(T * 2^k). We run it where that scan is
practical for a full-dataset evaluation (k <= 20 here; about 0.1 s per text at
k = 16 and a few seconds at k = 20) and mark larger payloads infeasible, which
is how this family of schemes is reported in the literature.
"""
import numpy as np

from .common import PRF, mix64

_U64 = np.uint64
_SALT = _U64(0xA5A5A5A55A5A5A5A)


class CyclicShiftWatermark:
    name = "cyclic"
    feasible_ks = [16, 20, 24]

    def __init__(self, secret: str, k_bits: int, vocab_size: int, gamma: float = 0.5):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.thr = _U64(int(gamma * 2**64))
        self.prf = PRF(secret, "cyclic.hash", vocab_size)
        self.reconfigure(k_bits)

    def reconfigure(self, k_bits: int):
        self.k = k_bits
        self.M = 1 << k_bits
        with np.errstate(over="ignore"):
            # one mixed key per candidate message, precomputed once per k
            self._shift_keys = mix64(np.arange(self.M, dtype=_U64) ^ _SALT)

    @staticmethod
    def _msg_int(message):
        return int("".join(str(b) for b in message), 2)

    # ---------------------------------------------------------- embedding ---
    def favored_set(self, prev_id: int, pos: int, message) -> np.ndarray:
        m = self._msg_int(message)
        h = self.prf.over_vocab(prev_id)             # 64-bit hash per candidate token
        with np.errstate(over="ignore"):
            return mix64(h ^ self._shift_keys[m]) < self.thr

    # ---------------------------------------------------------- detection ---
    def detect(self, gen_ids, prompt_ids):
        """Exhaustive scan over all 2^k candidate messages. Model-free,
        O(T * 2^k) time, O(2^k) memory."""
        scores = np.zeros(self.M, dtype=np.int32)
        prev, n = prompt_ids[-1], 0
        for tok in gen_ids:
            if tok < self.vocab_size:
                h = self.prf.over_vocab(prev)[tok]
                with np.errstate(over="ignore"):
                    scores += (mix64(h ^ self._shift_keys) < self.thr)
                n += 1
            prev = tok
        s = int(scores.argmax())
        decoded = [int(b) for b in format(s, f"0{self.k}b")]
        return {"message": decoded,
                "agreement": float(scores[s]) / max(n, 1),
                "num_tokens": len(gen_ids)}
