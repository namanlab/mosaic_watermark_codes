"""Qu et al. multi-bit watermark (USENIX Security 2025): pseudo-random segment
assignment with Reed-Solomon error correction.

Faithful to github.com/randomizedtree/segment-watermark: the payload is split
into 4-bit symbols over GF(2^4) and Reed-Solomon encoded (their default is a
[14, 8] code, correcting up to 3 symbol errors). Each RS codeword symbol is
carried by one segment of tokens; a token's segment is fixed by a context-keyed
hash. Within a segment, the symbol value v selects a *balanced* green list
(gamma = 1/2 of the vocabulary), and the +delta bias is written onto it, so the
per-token signal is as reliable as a standard green-list watermark. Detection
scores, for every segment and every candidate symbol value, how many tokens fall
in that value's green list, takes the argmax per segment to recover the codeword,
and Reed-Solomon decodes it to correct the remaining symbol errors and return the
payload. This error-correction stage is what sustains a high match rate at large
payloads.

The earlier version of this file used 1/16-of-vocab count cells and no error
correction; on the green-list layer that partition is too small to bias
reliably at delta = 6, which collapsed the match rate at large payloads. The
balanced half-vocab green list and the RS code restore the reported behavior.
"""
import numpy as np

from .common import PRF
from . import reed_solomon as rs

SEG_BITS = 4          # symbol size in bits; GF(2^4), 16 values per segment
N_VALUES = 1 << SEG_BITS
MAX_N = rs.FIELD_CHARAC          # RS codeword length capped at 2^m - 1 = 15


class QuSegmentWatermark:
    name = "qu"
    feasible_ks = [16, 20, 24, 32, 48]

    def __init__(self, secret: str, k_bits: int, vocab_size: int, gamma: float = 0.5):
        self.vocab_size = vocab_size
        self.thr = np.uint64(int(gamma * 2**64))
        self.prf_seg = PRF(secret, "qu.segment", vocab_size)
        self.prf_green = PRF(secret, "qu.green", vocab_size)
        self._cw_key = None
        self._cw = None
        self.reconfigure(k_bits)

    def reconfigure(self, k_bits: int):
        assert k_bits % SEG_BITS == 0, "payload must be a multiple of 4 bits"
        self.k = k_bits
        self.k_data = k_bits // SEG_BITS               # RS data symbols
        self.n_code = min(MAX_N, self.k_data + 6)      # [k_data+6, k_data] (=> [14,8])
        self.nsym = self.n_code - self.k_data          # parity symbols
        self._cw_key = None

    # ---------------------------------------------------------- helpers -----
    def _symbols(self, message):
        return [int("".join(str(b) for b in message[s * SEG_BITS:(s + 1) * SEG_BITS]), 2)
                for s in range(self.k_data)]

    def _codeword(self, message):
        key = tuple(message)
        if key != self._cw_key:
            self._cw = rs.rs_encode(self._symbols(message), self.nsym)
            self._cw_key = key
        return self._cw

    def _segment(self, prev_id: int) -> int:
        return self.prf_seg.scalar(prev_id) % self.n_code

    # ---------------------------------------------------------- embedding ---
    def favored_set(self, prev_id: int, pos: int, message) -> np.ndarray:
        j = self._segment(prev_id)
        v = self._codeword(message)[j]
        return self.prf_green.salted_over_vocab(prev_id, j * N_VALUES + v) < self.thr

    # ---------------------------------------------------------- detection ---
    def detect(self, gen_ids, prompt_ids):
        scores = np.zeros((self.n_code, N_VALUES), dtype=np.int64)
        prev = prompt_ids[-1]
        for tok in gen_ids:
            if tok < self.vocab_size:
                j = self._segment(prev)
                salts = np.arange(j * N_VALUES, j * N_VALUES + N_VALUES)
                green = self.prf_green.token_over_salts(prev, tok, salts) < self.thr
                scores[j] += green
            prev = tok
        codeword = scores.argmax(axis=1).tolist()
        data = rs.rs_decode(codeword, self.nsym, self.k_data)
        decoded = []
        for v in data:
            decoded.extend(int(b) for b in format(int(v) & (N_VALUES - 1), f"0{SEG_BITS}b"))
        agree = int(scores.max(axis=1).sum())     # winning votes, one per segment
        return {"message": decoded,
                "agreement": agree / max(len(gen_ids), 1),
                "num_tokens": len(gen_ids)}
