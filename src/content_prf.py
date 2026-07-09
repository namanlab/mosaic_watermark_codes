"""
Content-addressed keyed PRF for MOSAIC.

The defining choice: every per-position quantity (green mask, chip, and the
fountain check it emits) is keyed on an ORDER-CANONICAL feature of the local
context --- the *sorted multiset* of the preceding `window` tokens --- rather
than on the absolute position t or the ordered context. Two consequences:

  * Rearranging tokens that preserves a position's predecessor multiset leaves
    that position's vote completely unchanged. Block/sentence reordering keeps
    every interior position intact; only the few tokens straddling a new
    boundary are disturbed.
  * Detection re-derives every key from the (possibly reordered) text with no
    notion of position, so accumulation is commutative and order-free.

Keying on a *sorted* window (vs. the ordered window a sequential scheme would
use) additionally makes the scheme invariant to local word swaps inside the
window, and turns "robust to rearrangement" from an incidental property into a
designed one.
"""

import hashlib
from typing import List, Tuple

import numpy as np


class ContentPRF:
    def __init__(self, secret_key: str, vocab_size: int, k_bits: int,
                 gamma: float = 0.5, window: int = 3,
                 degree_dist: np.ndarray = None, max_degree: int = 8):
        self.key = secret_key.encode()
        self.vocab_size = vocab_size
        self.k_bits = k_bits
        self.gamma = gamma
        self.window = window
        self.degree_dist = degree_dist
        self.max_degree = max_degree
        self._cache = {}

    def _seed(self, context_tokens, tag: bytes) -> int:
        # order-canonical: sort the predecessor window (multiset, no order)
        win = sorted(int(t) for t in context_tokens[-self.window:])
        h = hashlib.blake2b(self.key + tag, digest_size=8)
        for t in win:
            h.update(t.to_bytes(4, "big", signed=False))
        return int.from_bytes(h.digest(), "big")

    def context_id(self, context_tokens) -> Tuple[int, ...]:
        """The canonical key (used to deduplicate repeated contexts)."""
        return tuple(sorted(int(t) for t in context_tokens[-self.window:]))

    def green_mask(self, context_tokens) -> np.ndarray:
        cid = self.context_id(context_tokens)
        cached = self._cache.get(cid)
        if cached is not None:
            return cached
        rng = np.random.Generator(np.random.PCG64(self._seed(context_tokens, b"|mask")))
        mask = rng.random(self.vocab_size) < self.gamma
        self._cache[cid] = mask
        return mask

    def chip(self, context_tokens) -> int:
        rng = np.random.Generator(np.random.PCG64(self._seed(context_tokens, b"|chip")))
        return int(rng.integers(0, 2))

    def check_subset(self, context_tokens) -> np.ndarray:
        """
        The fountain check this position emits: a content-chosen subset of
        message-bit indices whose XOR-parity the position votes on.
        """
        rng = np.random.Generator(np.random.PCG64(self._seed(context_tokens, b"|check")))
        if self.degree_dist is not None:
            d = int(rng.choice(len(self.degree_dist), p=self.degree_dist)) + 1
        else:
            d = 3
        d = min(d, self.max_degree, self.k_bits)
        return rng.choice(self.k_bits, size=d, replace=False)

    def clear_cache(self):
        self._cache.clear()
