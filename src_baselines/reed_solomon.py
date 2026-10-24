"""Systematic Reed-Solomon codes over GF(2^m) for the Qu et al. baseline.

Qu et al. (github.com/randomizedtree/segment-watermark) protect the payload with
an RS [n, k] code over GF(2^m) whose (n, k, m) comes from their rs_search.py
rule, and decode with a bounded-distance decoder that corrects up to
t = (n - k) / 2 symbol errors, falling back to the first k received symbols when
decoding fails. This module provides such codes for any m in 2..8 with the
standard primitive polynomials and generator alpha = 2. The encode / syndrome /
Berlekamp-Massey / Chien / Forney routines follow the standard "Reed-Solomon for
coders" reference (first consecutive root alpha^0). Any [n, k] RS code over the
same field corrects the same number of symbol errors, so the watermark behaves as
with the implementation Qu et al. use.

    rs = ReedSolomon(m=8)
    cw = rs.encode(msg, nsym)           # msg: k symbols in [0, 2^m)
    msg_hat = rs.decode(cw, nsym, k)    # k symbols; falls back to cw[:k]

The module-level functions (rs_encode, rs_decode, ...) are GF(2^4) instances,
kept for callers written against the earlier GF(2^4)-only version.
"""

# primitive polynomials (with the x^m term) for GF(2^m)
PRIMITIVE = {2: 0x7, 3: 0xB, 4: 0x13, 5: 0x25, 6: 0x43, 7: 0x89, 8: 0x11D}
GENERATOR = 2


class ReedSolomon:
    def __init__(self, m):
        assert m in PRIMITIVE, f"no primitive polynomial for GF(2^{m})"
        self.m = m
        self.prim = PRIMITIVE[m]
        self.order = (1 << m) - 1                 # multiplicative group order
        self._exp = [0] * (2 * self.order)
        self._log = [0] * (self.order + 1)
        x = 1
        for i in range(self.order):
            self._exp[i] = x
            self._log[x] = i
            x = self._mul_nolut(x, GENERATOR)
        for i in range(self.order, 2 * self.order):
            self._exp[i] = self._exp[i - self.order]

    # ---------------------------------------------------------- field ---
    def _mul_nolut(self, x, y):
        r = 0
        while y:
            if y & 1:
                r ^= x
            y >>= 1
            x <<= 1
            if x & (1 << self.m):
                x ^= self.prim
        return r

    def mul(self, x, y):
        if x == 0 or y == 0:
            return 0
        return self._exp[self._log[x] + self._log[y]]

    def div(self, x, y):
        if x == 0:
            return 0
        return self._exp[(self._log[x] + self.order - self._log[y]) % self.order]

    def pow(self, x, power):
        return self._exp[(self._log[x] * power) % self.order]

    def inverse(self, x):
        return self._exp[self.order - self._log[x]]

    # ----------------------------------------------------- polynomials ---
    def poly_scale(self, p, x):
        return [self.mul(c, x) for c in p]

    @staticmethod
    def poly_add(p, q):
        r = [0] * max(len(p), len(q))
        for i in range(len(p)):
            r[i + len(r) - len(p)] = p[i]
        for i in range(len(q)):
            r[i + len(r) - len(q)] ^= q[i]
        return r

    def poly_mul(self, p, q):
        r = [0] * (len(p) + len(q) - 1)
        for j in range(len(q)):
            for i in range(len(p)):
                r[i + j] ^= self.mul(p[i], q[j])
        return r

    def poly_eval(self, p, x):
        y = p[0]
        for i in range(1, len(p)):
            y = self.mul(y, x) ^ p[i]
        return y

    # ------------------------------------------------------------ code ---
    def generator_poly(self, nsym):
        g = [1]
        for i in range(nsym):
            g = self.poly_mul(g, [1, self.pow(GENERATOR, i)])
        return g

    def encode(self, msg, nsym):
        """Systematic encode: k symbols -> k + nsym symbols, message first."""
        assert len(msg) + nsym <= self.order, "codeword longer than 2^m - 1"
        assert all(0 <= s <= self.order for s in msg), "symbol out of range"
        gen = self.generator_poly(nsym)
        out = [0] * (len(msg) + nsym)
        out[:len(msg)] = msg
        for i in range(len(msg)):
            coef = out[i]
            if coef != 0:
                for j in range(1, len(gen)):
                    out[i + j] ^= self.mul(gen[j], coef)
        out[:len(msg)] = msg
        return out

    def _syndromes(self, msg, nsym):
        return [0] + [self.poly_eval(msg, self.pow(GENERATOR, i)) for i in range(nsym)]

    def _error_locator(self, synd, nsym):
        err_loc, old_loc = [1], [1]
        for i in range(nsym):
            old_loc = old_loc + [0]
            delta = synd[i + 1]
            for j in range(1, len(err_loc)):
                delta ^= self.mul(err_loc[len(err_loc) - 1 - j], synd[i + 1 - j])
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = self.poly_scale(old_loc, delta)
                    old_loc = self.poly_scale(err_loc, self.inverse(delta))
                    err_loc = new_loc
                err_loc = self.poly_add(err_loc, self.poly_scale(old_loc, delta))
        while len(err_loc) and err_loc[0] == 0:
            del err_loc[0]
        return err_loc

    def _find_errors(self, err_loc, nmess):
        positions = [nmess - 1 - i for i in range(nmess)
                     if self.poly_eval(err_loc, self.pow(GENERATOR, i)) == 0]
        return positions if len(positions) == len(err_loc) - 1 else None

    def _correct(self, msg, synd, err_pos):
        coef_pos = [len(msg) - 1 - p for p in err_pos]
        err_loc = [1]
        for i in coef_pos:
            err_loc = self.poly_mul(err_loc, self.poly_add([1], [self.pow(GENERATOR, i), 0]))
        rem = self.poly_mul(synd[1:][::-1], err_loc)
        nsym = len(err_loc) - 1
        err_eval = rem[len(rem) - (nsym + 1):][::-1]
        X = [self.pow(GENERATOR, p) for p in coef_pos]
        E = [0] * len(msg)
        for i, Xi in enumerate(X):
            Xi_inv = self.inverse(Xi)
            prime = 1
            for j in range(len(X)):
                if j != i:
                    prime = self.mul(prime, 1 ^ self.mul(Xi_inv, X[j]))
            if prime == 0:
                return None
            E[err_pos[i]] = self.div(self.poly_eval(err_eval[::-1], Xi_inv), prime)
        return self.poly_add(msg, E)

    def decode(self, codeword, nsym, k):
        """Correct up to nsym // 2 symbol errors and return the k message
        symbols; when decoding fails, return the first k received symbols, as
        Qu et al.'s detector does."""
        msg = list(codeword)
        synd = self._syndromes(msg, nsym)
        if max(synd) == 0:
            return msg[:k]
        err_loc = self._error_locator(synd, nsym)
        if len(err_loc) - 1 > nsym // 2:
            return msg[:k]
        err_pos = self._find_errors(err_loc[::-1], len(msg))
        if err_pos is None:
            return msg[:k]
        corrected = self._correct(msg, synd, err_pos)
        if corrected is None or max(self._syndromes(corrected, nsym)) != 0:
            return msg[:k]
        return corrected[:k]


# GF(2^4) module-level API (compatibility with the earlier version)
_GF16 = ReedSolomon(4)
PRIM = _GF16.prim
FIELD_CHARAC = _GF16.order


def rs_encode(msg, nsym):
    return _GF16.encode(msg, nsym)


def rs_decode(codeword, nsym, k):
    return _GF16.decode(codeword, nsym, k)


if __name__ == "__main__":
    import random
    random.seed(0)
    # the (n, k, m) codes Qu et al.'s rs_search rule selects for 16..48 bits
    for n, k, m in [(6, 4, 4), (6, 4, 5), (5, 3, 8), (6, 4, 8), (10, 6, 8), (14, 8, 4)]:
        rs, nsym, t = ReedSolomon(m), n - k, (n - k) // 2
        ok = 0
        for _ in range(2000):
            msg = [random.randrange(1 << m) for _ in range(k)]
            cw = rs.encode(msg, nsym)
            assert cw[:k] == msg, "encoding is not systematic"
            recv = list(cw)
            for p in random.sample(range(n), random.randint(0, t)):
                recv[p] ^= random.randint(1, (1 << m) - 1)
            ok += rs.decode(recv, nsym, k) == msg
        print(f"GF(2^{m}) [{n},{k}] t={t}: corrected {ok}/2000 trials with <= t errors")
