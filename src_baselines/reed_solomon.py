"""Reed-Solomon over GF(2^4) with 4-bit symbols, matching the segment_bit=4
setting of Qu et al. (github.com/randomizedtree/segment-watermark). The field
uses the primitive polynomial x^4 + x + 1 and generator alpha = 2. The encode /
syndrome / Berlekamp-Massey / Chien / Forney routines follow the standard
"Reed-Solomon for coders" reference (fcr = 0, generator = 2), specialized to
this small field. A [14, 8] code over GF(2^4) is Qu et al.'s default and
corrects up to 3 symbol errors.
"""

PRIM = 0x13         # x^4 + x + 1
GENERATOR = 2       # alpha
FIELD_CHARAC = 15   # 2^4 - 1

_exp = [0] * (FIELD_CHARAC * 2)
_log = [0] * (FIELD_CHARAC + 1)


def _mul_noLUT(x, y):
    """Carry-less multiply of x, y in GF(2^4) with reduction, no tables."""
    r = 0
    while y:
        if y & 1:
            r ^= x
        y >>= 1
        x <<= 1
        if x & 0x10:
            x ^= PRIM
    return r


def _init_tables():
    x = 1
    for i in range(FIELD_CHARAC):
        _exp[i] = x
        _log[x] = i
        x = _mul_noLUT(x, GENERATOR)
    for i in range(FIELD_CHARAC, FIELD_CHARAC * 2):
        _exp[i] = _exp[i - FIELD_CHARAC]


_init_tables()


def gf_mul(x, y):
    if x == 0 or y == 0:
        return 0
    return _exp[_log[x] + _log[y]]


def gf_div(x, y):
    if x == 0:
        return 0
    return _exp[(_log[x] + FIELD_CHARAC - _log[y]) % FIELD_CHARAC]


def gf_pow(x, power):
    return _exp[(_log[x] * power) % FIELD_CHARAC]


def gf_inverse(x):
    return _exp[FIELD_CHARAC - _log[x]]


def gf_poly_scale(p, x):
    return [gf_mul(c, x) for c in p]


def gf_poly_add(p, q):
    r = [0] * max(len(p), len(q))
    for i in range(len(p)):
        r[i + len(r) - len(p)] = p[i]
    for i in range(len(q)):
        r[i + len(r) - len(q)] ^= q[i]
    return r


def gf_poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for j in range(len(q)):
        for i in range(len(p)):
            r[i + j] ^= gf_mul(p[i], q[j])
    return r


def gf_poly_eval(p, x):
    y = p[0]
    for i in range(1, len(p)):
        y = gf_mul(y, x) ^ p[i]
    return y


def rs_generator_poly(nsym):
    g = [1]
    for i in range(nsym):
        g = gf_poly_mul(g, [1, gf_pow(GENERATOR, i)])
    return g


def rs_encode(msg, nsym):
    """Systematic encode: msg (k symbols) -> codeword (k + nsym symbols)."""
    gen = rs_generator_poly(nsym)
    out = [0] * (len(msg) + nsym)
    out[:len(msg)] = msg
    for i in range(len(msg)):
        coef = out[i]
        if coef != 0:
            for j in range(1, len(gen)):
                out[i + j] ^= gf_mul(gen[j], coef)
    out[:len(msg)] = msg
    return out


def _calc_syndromes(msg, nsym):
    return [0] + [gf_poly_eval(msg, gf_pow(GENERATOR, i)) for i in range(nsym)]


def _find_error_locator(synd, nsym):
    err_loc = [1]
    old_loc = [1]
    for i in range(nsym):
        old_loc = old_loc + [0]
        delta = synd[i + 1]
        for j in range(1, len(err_loc)):
            delta ^= gf_mul(err_loc[len(err_loc) - 1 - j], synd[i + 1 - j])
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = gf_poly_scale(old_loc, delta)
                old_loc = gf_poly_scale(err_loc, gf_inverse(delta))
                err_loc = new_loc
            err_loc = gf_poly_add(err_loc, gf_poly_scale(old_loc, delta))
    while len(err_loc) and err_loc[0] == 0:
        del err_loc[0]
    return err_loc


def _find_errors(err_loc, nmess):
    errs = len(err_loc) - 1
    positions = []
    for i in range(nmess):
        if gf_poly_eval(err_loc, gf_pow(GENERATOR, i)) == 0:
            positions.append(nmess - 1 - i)
    if len(positions) != errs:
        return None
    return positions


def _find_error_evaluator(synd, err_loc, nsym):
    remainder = gf_poly_mul(synd, err_loc)
    remainder = remainder[len(remainder) - (nsym + 1):]
    return remainder


def _correct_errata(msg, synd, err_pos):
    coef_pos = [len(msg) - 1 - p for p in err_pos]
    err_loc = [1]
    for i in coef_pos:
        err_loc = gf_poly_mul(err_loc, gf_poly_add([1], [gf_pow(GENERATOR, i), 0]))
    err_eval = _find_error_evaluator(synd[1:][::-1], err_loc, len(err_loc) - 1)[::-1]
    X = [gf_pow(GENERATOR, p) for p in coef_pos]
    E = [0] * len(msg)
    for i, Xi in enumerate(X):
        Xi_inv = gf_inverse(Xi)
        err_loc_prime = 1
        for j in range(len(X)):
            if j != i:
                err_loc_prime = gf_mul(err_loc_prime, 1 ^ gf_mul(Xi_inv, X[j]))
        if err_loc_prime == 0:
            return None
        y = gf_poly_eval(err_eval[::-1], Xi_inv)
        magnitude = gf_div(y, err_loc_prime)
        E[err_pos[i]] = magnitude
    return gf_poly_add(msg, E)


def rs_decode(codeword, nsym, k):
    """Correct up to (nsym // 2) symbol errors; return the k data symbols.

    Falls back to the raw hard-decision data symbols if decoding fails, so the
    caller always receives a payload of the right length.
    """
    msg = list(codeword)
    synd = _calc_syndromes(msg, nsym)
    if max(synd) == 0:
        return msg[:k]
    err_loc = _find_error_locator(synd, nsym)
    if len(err_loc) - 1 > nsym // 2:
        return msg[:k]
    err_pos = _find_errors(err_loc[::-1], len(msg))
    if err_pos is None:
        return msg[:k]
    corrected = _correct_errata(msg, synd, err_pos)
    if corrected is None:
        return msg[:k]
    if max(_calc_syndromes(corrected, nsym)) != 0:
        return msg[:k]
    return corrected[:k]


if __name__ == "__main__":
    import random
    random.seed(0)
    for k, nsym in [(8, 6), (4, 6), (12, 3), (6, 6), (5, 4)]:
        n = k + nsym
        t = nsym // 2
        ok = 0
        for _ in range(2000):
            msg = [random.randrange(16) for _ in range(k)]
            cw = rs_encode(msg, nsym)
            assert cw[:k] == msg, "encode not systematic"
            # inject up to t errors
            recv = list(cw)
            npos = random.randint(0, t)
            for p in random.sample(range(n), npos):
                recv[p] ^= random.randint(1, 15)
            dec = rs_decode(recv, nsym, k)
            ok += (dec == msg)
        print(f"[{n},{k}] t={t}: corrected {ok}/2000 within-t-error trials")
