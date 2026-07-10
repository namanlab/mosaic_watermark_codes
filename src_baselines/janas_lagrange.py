"""Janas, Morawiecki, Pieprzyk watermark (arXiv:2505.05712): Lagrange
interpolation over a Galois field, recovered by Maximum Collinear Points.

The k-bit payload is two coefficients of a line y = a1 * x + a0 over GF(2^t)
with t = k/2. Public x-coordinates x_j are derived from the key; the encoder
computes y_j = a1 * x_j + a0 and embeds each t-bit y_j across t tokens, one bit
per token via the green-list physical layer (green if the bit is 1). Extraction
reads back noisy points (x_j, y_hat_j) and solves the Maximum Collinear Points
problem: the line through the largest number of recovered points gives (a1, a0).
Bit errors knock individual points off the line, and the scheme succeeds as
long as enough correct points remain collinear, which is why its accuracy
degrades with the payload as fewer, longer points are packed into 200 tokens.

Extraction is O(N^2) in the number of points N = floor(T / t), polynomial in
the payload, so all payload sizes are feasible.
"""
import numpy as np

from .common import PRF

_U64 = np.uint64

# primitive polynomials (low t bits, x^t term implicit) for the fields we need
_POLY = {8: 0x1B, 10: 0x09, 12: 0x53, 16: 0x100B, 24: 0x1B}


class GF:
    def __init__(self, t):
        assert t in _POLY, f"no polynomial for GF(2^{t})"
        self.t = t
        self.red = _POLY[t]
        self.mask = (1 << t) - 1
        self.hibit = 1 << (t - 1)

    def mul(self, a, b):
        p = 0
        for _ in range(self.t):
            if b & 1:
                p ^= a
            b >>= 1
            carry = a & self.hibit
            a = (a << 1) & self.mask
            if carry:
                a ^= self.red
        return p

    def inv(self, a):
        result, base, e = 1, a, (1 << self.t) - 2   # a^(2^t - 2) = a^{-1}
        while e:
            if e & 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            e >>= 1
        return result


class JanasWatermark:
    name = "janas"
    feasible_ks = [16, 20, 24, 32, 48]

    def __init__(self, secret: str, k_bits: int, vocab_size: int, gamma: float = 0.5):
        self.vocab_size = vocab_size
        self.thr = _U64(int(gamma * 2**64))
        self.prf_green = PRF(secret, "janas.green", vocab_size)
        self.prf_x = PRF(secret, "janas.x", vocab_size)
        self.reconfigure(k_bits)

    def reconfigure(self, k_bits: int):
        assert k_bits % 2 == 0, "Janas payload must be even (two GF(2^t) coeffs)"
        self.k = k_bits
        self.t = k_bits // 2
        self.gf = GF(self.t)
        self.x = [1 + (self.prf_x.scalar(j) % ((1 << self.t) - 1))
                  for j in range(256)]   # public x-coordinates, per point index

    def _coeffs(self, message):
        a1 = int("".join(str(b) for b in message[: self.t]), 2)
        a0 = int("".join(str(b) for b in message[self.t:]), 2)
        return a1, a0

    def _green(self, prev_id: int) -> np.ndarray:
        return self.prf_green.over_vocab(prev_id) < self.thr

    # ---------------------------------------------------------- embedding ---
    def favored_set(self, prev_id: int, pos: int, message) -> np.ndarray:
        a1, a0 = self._coeffs(message)
        j, b = pos // self.t, pos % self.t
        y = self.gf.mul(a1, self.x[j]) ^ a0
        bit = (y >> b) & 1
        mask = self._green(prev_id)
        return mask if bit else ~mask

    # ---------------------------------------------------------- detection ---
    def detect(self, gen_ids, prompt_ids):
        n_pts = len(gen_ids) // self.t
        pts = []
        for j in range(n_pts):
            y = 0
            for b in range(self.t):
                p = j * self.t + b
                prev = gen_ids[p - 1] if p > 0 else prompt_ids[-1]
                tok = gen_ids[p]
                if tok < self.vocab_size and bool(self._green(prev)[tok]):
                    y |= (1 << b)
            pts.append((self.x[j], y))

        a1, a0, best = 0, 0, -1
        for i in range(len(pts)):
            x1, y1 = pts[i]
            for j in range(i + 1, len(pts)):
                x2, y2 = pts[j]
                if x1 == x2:
                    continue
                dx, dy = x1 ^ x2, y1 ^ y2
                cnt = 0
                for x3, y3 in pts:                    # collinearity, no division
                    if self.gf.mul(dy, x3 ^ x1) == self.gf.mul(y3 ^ y1, dx):
                        cnt += 1
                if cnt > best:
                    slope = self.gf.mul(dy, self.gf.inv(dx))
                    best = cnt
                    a1, a0 = slope, y1 ^ self.gf.mul(slope, x1)

        decoded = [int(b) for b in format(a1, f"0{self.t}b")] + \
                  [int(b) for b in format(a0, f"0{self.t}b")]
        return {"message": decoded,
                "agreement": best / max(len(pts), 1),
                "num_tokens": len(gen_ids)}
