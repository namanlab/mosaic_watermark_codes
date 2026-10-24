"""Qu et al. multi-bit watermark (USENIX Security 2025), as released.

Re-implementation of the method the paper evaluates, "rsbh" in the authors'
release github.com/randomizedtree/segment-watermark (wm/generator.py
RSBHGenerator, wm/detector.py RSBHDecoder, balance_hash/, rs_search.py). Each
step follows that code.

Payload and code. A b-bit payload, read as an integer, is cut into k segments of
m bits, segment i holding bits [i m, (i + 1) m) counted from the least
significant bit. The k segments are encoded with a systematic Reed-Solomon
[n, k] code over GF(2^m) into n codeword symbols c_0 .. c_{n-1}. (n, k, m) is
the code rs_search.py selects: the smallest n with k m = b, rate k / n >= 0.6 and
t / n >= 0.15, giving (6, 4, 4), (6, 4, 5), (5, 3, 8), (6, 4, 8), and
(10, 6, 8) for b = 16, 20, 24, 32, and 48.

Seeding. 'hash' seeding with a one-token context (ngram = 1): the seed of a
position is (s * 35317 + previous token) mod (2^64 - 1), where s is the integer
watermark key (their --seed; derived here from the shared secret string) and
35317 their default --hash_key.

Green list. A torch.Generator seeded with that seed draws randperm(V); its first
gamma V entries form the green list.

Segment assignment. The position belongs to segment j = mapping[previous token],
the balanced token-to-segment map of qu_balance.py (their balance_hash/).

Embedding. The bias vector (delta on the green list) is rolled by -c_j, so token
t receives +delta exactly when (t + c_j) mod V is in the green list. After a
generated newline, the newline logit is lowered by 100, as in their generator.

Detection. From the third token of the continuation onward (their
start_pos = ngram + 1), with the previous continuation token as context, every
token adds [(token + v) mod V in green list] to COUNT[j][v] for each value
v < 2^m (their default scoring_method = none scores every token). Each segment
takes the argmax (first maximum), the n symbols are Reed-Solomon decoded, and
when decoding fails the first k symbols are used.

Generation uses the shared loop of common.py (same model, prompts, temperature,
and nucleus threshold for every method), and detection reads the generated
token ids.
"""
import hashlib

import numpy as np
import torch

from .qu_balance import cached_mapping
from .reed_solomon import ReedSolomon

SALT_KEY = 35317          # their --hash_key default
MOD = 2 ** 64 - 1
NEWLINE_ID = 13           # "\n" in the Llama tokenizer (their newline_idx for Llama)
NEWLINE_PENALTY = 100.0
RATE_MIN, T_MIN = 0.6, 0.15   # thresholds of rs_search.py


def rs_params(b: int):
    """(n, k, m) chosen by rs_search.py for a b-bit payload: the first code with
    the smallest n, scanning m = 2..8, k, then n."""
    best = None
    for m in range(2, 9):
        for k in range(1, 2 ** m):
            if m * k != b:
                continue
            for n in range(k + 2, 2 ** m + 1):
                if (n - k) % 2:
                    continue
                t = (n - k) / 2
                if k / n < RATE_MIN or t / n < T_MIN:
                    continue
                if best is None or n < best[0]:
                    best = (n, k, m)
    assert best is not None, f"no RS code satisfies rs_search.py's rule for {b} bits"
    return best


def _key_int(secret: str, domain: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{secret}|{domain}".encode()).digest()[:8], "little")


class QuSegmentWatermark:
    name = "qu"
    feasible_ks = [16, 20, 24, 32, 48]
    implementation = "released RSBH (github.com/randomizedtree/segment-watermark)"
    faithful_under_edits = True      # segment and green list keyed on the previous token

    def __init__(self, secret: str, k_bits: int, vocab_size: int, gamma: float = 0.5,
                 freq=None):
        self.V = vocab_size
        self.n_green = int(gamma * vocab_size)
        self.seed0 = _key_int(secret, "qu.seed") % MOD
        self.map_key = _key_int(secret, "qu.map")
        self._freq = freq                    # token frequencies; None -> data/qu file
        self._gen = torch.Generator()
        self.reconfigure(k_bits)

    def reconfigure(self, k_bits: int):
        self.k = k_bits
        self.n, self.kseg, self.m = rs_params(k_bits)
        self.nsym = self.n - self.kseg
        self.rs = ReedSolomon(self.m)
        self.mapping = cached_mapping(self.n, self.map_key, self.V, freq=self._freq)
        self._cw_key = self._cw = None

    # ------------------------------------------------------------ helpers ---
    def _green(self, prev_id: int) -> np.ndarray:
        self._gen.manual_seed((self.seed0 * SALT_KEY + int(prev_id)) % MOD)
        perm = torch.randperm(self.V, generator=self._gen).numpy()
        green = np.zeros(self.V, dtype=bool)
        green[perm[:self.n_green]] = True
        return green

    def _codeword(self, message):
        key = tuple(message)
        if key != self._cw_key:
            payload = int("".join(str(int(b)) for b in message), 2)
            mask = (1 << self.m) - 1
            segments = [(payload >> (self.m * i)) & mask for i in range(self.kseg)]
            self._cw = self.rs.encode(segments, self.nsym)
            self._cw_key = key
        return self._cw

    # ---------------------------------------------------------- embedding ---
    def favored_set(self, prev_id: int, pos: int, message) -> np.ndarray:
        c = self._codeword(message)[int(self.mapping[prev_id])]
        return np.roll(self._green(prev_id), -c)     # bias.roll(-c): t favoured iff (t + c) mod V green

    def logit_adjustment(self, generated_ids):
        """Their generator lowers the newline logit by 100 right after a
        generated newline."""
        if generated_ids and generated_ids[-1] == NEWLINE_ID:
            return [(NEWLINE_ID, -NEWLINE_PENALTY)]
        return []

    # ---------------------------------------------------------- detection ---
    def detect(self, gen_ids, prompt_ids):
        """RSBHDecoder: score continuation tokens from position 2 with the
        previous continuation token as context; the prompt is not used."""
        values = np.arange(1 << self.m)
        counts = np.zeros((self.n, 1 << self.m), dtype=np.int64)
        greens = {}
        for pos in range(2, len(gen_ids)):
            prev, tok = int(gen_ids[pos - 1]), int(gen_ids[pos])
            if prev not in greens:
                greens[prev] = self._green(prev)
            counts[int(self.mapping[prev])] += greens[prev][(tok + values) % self.V]
        symbols = counts.argmax(axis=1).tolist()
        data = self.rs.decode(symbols, self.nsym, self.kseg)
        payload = sum(int(s) << (i * self.m) for i, s in enumerate(data))
        decoded = [int(b) for b in format(payload, f"0{self.k}b")]
        scored = max(len(gen_ids) - 2, 1)
        return {"message": decoded,
                "agreement": float(counts.max(axis=1).sum()) / scored,
                "num_tokens": len(gen_ids)}
