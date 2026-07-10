"""Cohen, Hoover, Schoenbach L-bit watermark (IEEE S&P 2025), Figure 2.

Reduction from a zero-bit watermark to an L-bit scheme. The key holds 2L
independent zero-bit keys k[i,0], k[i,1], one pair per message bit. To embed
message m, each generated block is watermarked with key k[i, m[i]] for a
uniformly random bit index i (so all blocks share the same text, and detection
does not need to know the allocation). Extraction runs the zero-bit detector
for every one of the 2L keys over the whole text; bit i is decoded by which of
k[i,0], k[i,1] carries the mark.

Here a block is one token and the zero-bit watermark is the green-list scheme:
key k[i,b] induces a gamma-fraction green list keyed on the previous token, and
the +delta bias is applied to that list. Bit i is decoded by which of k[i,0],
k[i,1] carries the mark, i.e. m_hat[i] = argmax_b count[i,b] (the
maximum-likelihood version of the paper's per-key detection test).

Adaptation to the green-list layer: the paper's construction assumes an
undetectable (distortion-free) zero-bit primitive whose per-token signal is
strong, so it can allocate blocks to bits by fresh randomness and detect
obliviously over the whole text. On the green-list layer at delta = 6 the
per-token signal is weaker, so an oblivious whole-text detector would drown
each bit's 1/L share of tokens in baseline noise. We therefore use a keyed
block schedule (the paper leaves the block structure to W'): a key-derived
map sends each position to a bit, and detection isolates that bit's tokens.
The essential Cohen mechanism, two independent keys per bit with the bit read
off by which key carries the mark, is preserved. Extraction is O(T),
polynomial in the payload, so all payload sizes are feasible.
"""
import numpy as np

from .common import PRF

_U64 = np.uint64


class CohenWatermark:
    name = "cohen"
    feasible_ks = [16, 20, 24, 32, 48]

    def __init__(self, secret: str, k_bits: int, vocab_size: int, gamma: float = 0.5):
        self.vocab_size = vocab_size
        self.thr = _U64(int(gamma * 2**64))
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
