"""Cohen, Hoover, Schoenbach L-bit watermark (IEEE S&P 2025), Figure 2.

The paper gives a black-box construction of an L-bit watermark from any
zero-bit watermark W' = (KeyGen', Wat', Detect') that is block-by-block and
R1-robustly detectable (detection succeeds whenever a single block of watermarked
text survives). It has no implementation or experiments. This file implements
Figure 2 exactly:

  KeyGen   2L independent zero-bit keys k[i, 0], k[i, 1] for i in [L].
  Wat      block by block: draw a fresh uniform index i from [L] (randomness
           independent of the keys, unknown to the detector) and generate the
           next block with W' under key k[i, m[i]].
  Extract  for every i and b in {0, 1}: z[i, b] = Detect'_{k[i, b]}(text).
           m_hat[i] = bottom if z[i, 0] = z[i, 1] = false; 0 if z[i, 0] = true;
           1 otherwise. The payload is recovered only when no bit is bottom.

Zero-bit primitive W'. The paper proves the construction for the undetectable
watermark of Christ, Gunn, and Zamir; the construction itself is generic. Here W'
is the green-list watermark shared by every method in the comparison (gamma =
1/2, bias delta, green list keyed on the previous token), which leaves two
choices to the primitive, fixed as follows:
  - block length B = 21 tokens: the smallest block whose expected z-score alone
    reaches the detection threshold at the measured per-token flip rate 0.063
    ((1 - 0.063 - 1/2) sqrt(B) / (1/2) >= 4), which is what R1-robust detection
    asks of a block;
  - Detect'_k fires when some window of B consecutive tokens has a green-count
    z-score of at least 4 under k, the standard green-list threshold, so a single
    surviving block anywhere in the text suffices.
The 200-token budget holds 9 full blocks and a final partial block, which is
watermarked in the same way.

Decoded bits that are bottom are reported as -1, so they never match.
"""
import hashlib
import math

import numpy as np

from .common import PRF

_U64 = np.uint64
BLOCK = 21            # tokens per block (see module docstring)
Z_THRESHOLD = 4.0     # zero-bit detection threshold on the z-score
BOTTOM = -1


class CohenWatermark:
    name = "cohen"
    feasible_ks = [16, 20, 24, 32, 48]
    implementation = "Cohen et al. Figure 2 (L-bit from zero-bit) on the green-list layer"
    faithful_under_edits = True      # fresh random allocation, oblivious whole-text detection

    def __init__(self, secret: str, k_bits: int, vocab_size: int, gamma: float = 0.5):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.thr = _U64(int(gamma * 2**64))
        self.prf = PRF(secret, "cohen.zerobit", vocab_size)     # key k[i, b] = salt 2i + b
        # fresh per-block randomness for the embedder; the detector never sees it
        seed = int.from_bytes(hashlib.sha256(f"{secret}|cohen.embed".encode()).digest()[:8], "little")
        self._embed_rng = np.random.default_rng(seed)
        self._block_bit = 0
        # smallest green count in a window of BLOCK tokens with z >= Z_THRESHOLD
        mu, sd = gamma * BLOCK, math.sqrt(BLOCK * gamma * (1 - gamma))
        self.window_count = math.ceil(mu + Z_THRESHOLD * sd)
        self.reconfigure(k_bits)

    def reconfigure(self, k_bits: int):
        self.k = k_bits

    # ---------------------------------------------------------- embedding ---
    def favored_set(self, prev_id: int, pos: int, message) -> np.ndarray:
        if pos % BLOCK == 0:                       # new block: fresh uniform i in [L]
            self._block_bit = int(self._embed_rng.integers(0, self.k))
        i = self._block_bit
        salt = 2 * i + int(message[i])
        return self.prf.salted_over_vocab(prev_id, salt) < self.thr

    # ---------------------------------------------------------- detection ---
    def detect(self, gen_ids, prompt_ids):
        """Run Detect' for all 2L keys over the whole text and decode each bit
        with the rule of Figure 2."""
        n = len(gen_ids)
        salts = np.arange(2 * self.k)
        green = np.zeros((2 * self.k, n), dtype=np.int64)      # green[key, position]
        prev = prompt_ids[-1]
        for pos, tok in enumerate(gen_ids):
            if tok < self.vocab_size:
                green[:, pos] = self.prf.token_over_salts(prev, tok, salts) < self.thr
            prev = tok
        if n >= BLOCK:                                          # best window per key
            csum = np.concatenate([np.zeros((2 * self.k, 1), dtype=np.int64),
                                   np.cumsum(green, axis=1)], axis=1)
            best = (csum[:, BLOCK:] - csum[:, :-BLOCK]).max(axis=1)
        else:
            best = green.sum(axis=1)
        fired = best >= self.window_count                        # z[key]
        decoded = []
        for i in range(self.k):
            z0, z1 = bool(fired[2 * i]), bool(fired[2 * i + 1])
            decoded.append(BOTTOM if not (z0 or z1) else (0 if z0 else 1))
        return {"message": decoded,
                "agreement": float(best.max()) / BLOCK if n else 0.0,
                "num_tokens": n}
