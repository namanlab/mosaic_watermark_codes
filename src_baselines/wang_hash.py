"""Wang et al. style multi-bit watermark (CTWL / Balance-Marking, ICLR 2024).

The whole k-bit message seeds the vocabulary partition: the green list at a
position is a fresh gamma-fraction pseudorandom subset determined jointly by
the context and the entire message m, so the partitions for different messages
are independent. Embedding biases the green tokens of the true message by
+delta. Extraction scores every candidate message m' by counting green tokens
under m' and returns the argmax (Eq. 12 of Wang et al., argmax_{m'} P_w(m'|t)).

Extraction is therefore exponential in the payload, O(T * 2^k); we run it where
that scan is practical for a full-dataset evaluation (k <= 20) and mark larger
payloads infeasible, matching how this scheme is reported in the literature.
Balance-Marking additionally uses a proxy language model to make the two parts
probability-balanced; that step improves text quality rather than match rate,
and is omitted here so the physical layer is identical to the other baselines.
"""
import numpy as np

from .common import PRF, mix64

_U64 = np.uint64
_SALT = _U64(0x1234567890ABCDEF)


class WangHashWatermark:
    name = "wang"
    feasible_ks = [16, 20, 24]

    def __init__(self, secret: str, k_bits: int, vocab_size: int, gamma: float = 0.5):
        self.vocab_size = vocab_size
        self.thr = _U64(int(gamma * 2**64))
        self.prf = PRF(secret, "wang.hash", vocab_size)
        self.reconfigure(k_bits)

    def reconfigure(self, k_bits: int):
        self.k = k_bits
        self.M = 1 << k_bits
        with np.errstate(over="ignore"):
            # one independent mixing key per candidate message
            self._msg_keys = mix64(np.arange(self.M, dtype=_U64) ^ _SALT)

    @staticmethod
    def _msg_int(message):
        return int("".join(str(b) for b in message), 2)

    # ---------------------------------------------------------- embedding ---
    def favored_set(self, prev_id: int, pos: int, message) -> np.ndarray:
        m = self._msg_int(message)
        h = self.prf.over_vocab(prev_id)
        with np.errstate(over="ignore"):
            return mix64(h ^ self._msg_keys[m]) < self.thr

    # ---------------------------------------------------------- detection ---
    def detect(self, gen_ids, prompt_ids):
        """Argmax over all 2^k candidate messages of the green-token count."""
        scores = np.zeros(self.M, dtype=np.int32)
        prev, n = prompt_ids[-1], 0
        for tok in gen_ids:
            if tok < self.vocab_size:
                h = self.prf.over_vocab(prev)[tok]
                with np.errstate(over="ignore"):
                    scores += (mix64(h ^ self._msg_keys) < self.thr)
                n += 1
            prev = tok
        s = int(scores.argmax())
        decoded = [int(b) for b in format(s, f"0{self.k}b")]
        return {"message": decoded,
                "agreement": float(scores[s]) / max(n, 1),
                "num_tokens": len(gen_ids)}
