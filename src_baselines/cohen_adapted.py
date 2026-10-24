"""Cohen, Hoover, Schoenbach L-bit watermark (IEEE S&P 2025), adapted variant.

This is the variant used in the earlier runs, kept alongside the faithful Figure 2
implementation in cohen.py. Both share the core of Figure 2: the key holds 2L
independent zero-bit keys k[i,0], k[i,1], one pair per message bit, a block that
carries bit i is watermarked with key k[i, m[i]], and bit i is read off by which
of k[i,0], k[i,1] carries the mark. They differ in how blocks are allocated to
bits and how the detector finds them.

Here a block is one token and the zero-bit watermark is the green-list scheme:
key k[i,b] induces a gamma-fraction green list keyed on the previous token, and
the +delta bias is applied to that list.

Adaptation to the green-list layer: the paper's construction assumes an
undetectable (distortion-free) zero-bit primitive whose per-token signal is
strong, so it can allocate blocks to bits by fresh randomness and detect
obliviously over the whole text. On the green-list layer at delta = 6 the
per-token signal is weaker, so an oblivious whole-text detector would drown
each bit's 1/L share of tokens in baseline noise (cohen.py shows this: it
recovers no payload at 200 tokens). This variant therefore uses a keyed block
schedule (the paper leaves the block structure to W'): a key-derived map sends
each position to a bit, the detector knows the map and isolates that bit's
tokens, and bit i is decoded as m_hat[i] = argmax_b count[i,b] (the
maximum-likelihood version of the paper's per-key detection test). Extraction
is O(T), polynomial in the payload, so all payload sizes are feasible.

Because the schedule is indexed by position, insertions and deletions shift
every later token onto the wrong bit; this is a property of the adaptation,
which the robustness comparison measures as is.
"""
import numpy as np

from .common import PRF

_U64 = np.uint64


class CohenAdaptedWatermark:
    name = "cohen_adapted"
    feasible_ks = [16, 20, 24, 32, 48]
    implementation = ("Cohen et al. adapted: keyed position-to-bit schedule known "
                      "to the detector, per-bit argmax between the two keys")
    faithful_under_edits = True      # the stored detector is exactly this one

    def __init__(self, secret: str, k_bits: int, vocab_size: int, gamma: float = 0.5):
        self.vocab_size = vocab_size
        self.thr = _U64(int(gamma * 2**64))
        # domain strings as in the earlier runs, so stored generations re-detect exactly
        self.prf = PRF(secret, "cohen.greenlist", vocab_size)   # keys via salt = 2i+b
        self.prf_alloc = PRF(secret, "cohen.alloc", vocab_size)
        self.reconfigure(k_bits)

    def reconfigure(self, k_bits: int):
        self.k = k_bits

    def _bit_index(self, pos: int) -> int:
        # uniform random allocation of position -> bit, fixed by the key
        return self.prf_alloc.scalar(pos) % self.k

    # ---------------------------------------------------------- embedding ---
    def favored_set(self, prev_id: int, pos: int, message) -> np.ndarray:
        i = self._bit_index(pos)
        salt = 2 * i + int(message[i])
        return self.prf.salted_over_vocab(prev_id, salt) < self.thr

    # ---------------------------------------------------------- detection ---
    def detect(self, gen_ids, prompt_ids):
        """For each position, look up its bit i and add a green vote under both
        of bit i's keys; decode bit i by which key has more green tokens."""
        counts = np.zeros((self.k, 2), dtype=np.int64)
        seen = np.zeros(self.k, dtype=np.int64)
        prev = prompt_ids[-1]
        for pos, tok in enumerate(gen_ids):
            i = self._bit_index(pos)
            if tok < self.vocab_size:
                g = self.prf.token_over_salts(
                    prev, tok, np.array([2 * i, 2 * i + 1])) < self.thr
                counts[i, 0] += int(g[0])
                counts[i, 1] += int(g[1])
                seen[i] += 1
            prev = tok
        decoded = (counts[:, 1] > counts[:, 0]).astype(int).tolist()
        agree = int(counts[np.arange(self.k), decoded].sum())
        return {"message": decoded,
                "agreement": agree / max(int(seen.sum()), 1),
                "num_tokens": len(gen_ids)}
