"""Balanced token-to-segment map of Qu et al. (balance_hash/ in their release).

Qu et al. assign each generated token to a payload segment by its previous
token, through a map from token id to segment built offline so that every
segment receives about the same expected number of tokens. Their procedure
(balance_hash/helper.py and dp_balance.cpp):

  1. token frequencies f (token_freq_<model>.pkl in their data release), with
     zero counts raised to 1;
  2. a keyed shuffle of the token ids: numpy's legacy global RNG seeded with the
     first 4 bytes of sha256(KEY as 16 little-endian bytes), then shuffle;
  3. a dynamic program splitting the shuffled sequence into p contiguous groups
     that minimises the sum of squared group frequencies, ties broken towards
     the first candidate;
  4. group j of the shuffled order becomes segment j.

This module re-implements that procedure exactly (same seed derivation, same
shuffle, same recurrence, same tie-breaking and back-tracking), with exact
integer arithmetic.

    mapping = build_mapping(freq, p, key)     # int array, token id -> segment
"""
import hashlib
import os
import pickle
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "qu")
DEFAULT_MODEL = "meta-llama/Llama-2-7b-hf"
# frequency files of Qu et al.'s data release; any other model uses
# token_freq_<model tag>.pkl built by qu_token_freq.py
RELEASED_FREQ = {DEFAULT_MODEL: "token_freq_llama.pkl"}
FREQ_FILE = os.path.join(DATA_DIR, RELEASED_FREQ[DEFAULT_MODEL])


def model_tag(model_name: str) -> str:
    """Short lower-case tag of a Hugging Face model id, e.g. mistral-7b-v0.1."""
    return model_name.rstrip("/").split("/")[-1].lower()


def freq_file_for(model_name: str = DEFAULT_MODEL) -> str:
    name = RELEASED_FREQ.get(model_name, f"token_freq_{model_tag(model_name)}.pkl")
    return os.path.join(DATA_DIR, name)


def seed_from_key(key: int) -> int:
    """get_rand of their helper.py: sha256 of the key's 16 little-endian bytes,
    first 4 bytes read as an unsigned int."""
    digest = hashlib.sha256(key.to_bytes(16, "little", signed=False)).digest()
    return struct.unpack("<I", digest[0:4])[0]


def keyed_shuffle(n: int, key: int) -> list:
    indices = list(range(n))
    rng = np.random.RandomState(seed_from_key(key))   # legacy global-RNG algorithm
    rng.shuffle(indices)
    return indices


def dp_balance(freq, p: int) -> list:
    """Group lengths of the optimal contiguous p-partition, as dp_balance.cpp.

    Q[i][j] is the least sum of squared group sums over the LAST i + 1 elements
    split into j + 1 groups. dp_balance.cpp works in 128-bit integers; here each
    minimisation is searched in float64 and every candidate within a safe
    tolerance of the float minimum is re-evaluated in exact Python integers, so
    the chosen value and the first-minimum tie-break equal the exact ones.
    """
    f = [int(x) for x in freq]
    n = len(f)
    S = [0] * (n + 1)
    for t in range(n):
        S[t + 1] = S[t] + f[t]
    assert S[-1] < 2 ** 52, "prefix sums must be exact in float64"
    Sf = np.array(S, dtype=np.float64)
    Srev = [S[n - k - 1] for k in range(n)]          # Srev[k] = S[n - k - 1]
    Srevf = np.array(Srev, dtype=np.float64)
    Q = [[0] * p for _ in range(n)]                   # exact values
    Qf = np.zeros((n, p), dtype=np.float64)           # float copies for the search
    best = [[0] * p for _ in range(n)]
    for i in range(n - p + 1):
        Q[i][0] = (S[n] - S[n - i - 1]) ** 2          # D(n - i - 1, n - 1)
        Qf[i, 0] = float(Q[i][0])
    for j in range(1, p):
        for i in range(j, n - p + j + 1):
            # candidates over k = j-1 .. i-1:  Q[k][j-1] + D(n-i-1, n-k-2)
            d = Srevf[j - 1:i] - Sf[n - i - 1]
            cand = Qf[j - 1:i, j - 1] + d * d
            lo = float(cand.min())
            # each float candidate is within a few units in the last place (ulp) of
            # its exact value, so the exact minimum lies within 16 ulp of lo
            near = np.nonzero(cand <= lo + 16.0 * np.spacing(lo))[0]
            bv, bk = None, None
            for idx in near:                          # exact, first minimum wins
                k = int(idx) + j - 1
                dd = Srev[k] - S[n - i - 1]
                v = Q[k][j - 1] + dd * dd
                if bv is None or v < bv:
                    bv, bk = v, k
            Q[i][j] = bv
            Qf[i, j] = float(bv)
            best[i][j] = bk
    i, j = n - 1, p - 1
    seg = [best[i][j]]
    while j > 1:
        i = best[i][j]
        j -= 1
        seg.append(best[i][j])
    res_len = [n - seg[0] - 1]
    for t in range(1, p - 1):
        res_len.append(seg[t - 1] - seg[t])
    res_len.append(seg[-1] + 1)
    assert sum(res_len) == n and all(r > 0 for r in res_len)
    return res_len


def build_mapping(freq, p: int, key: int) -> np.ndarray:
    """Token id -> segment in [0, p), following gen_mapping / get_mapping."""
    freq = np.asarray(freq, dtype=np.int64)
    freq = freq + (freq == 0)                        # patch zero counts to 1
    indices = keyed_shuffle(len(freq), key)
    res_len = dp_balance(freq[indices], p)
    mapping = np.empty(len(freq), dtype=np.int64)
    base = 0
    for seg_idx, length in enumerate(res_len):
        mapping[indices[base:base + length]] = seg_idx
        base += length
    return mapping


def load_freq(path: str = FREQ_FILE) -> np.ndarray:
    if not os.path.exists(path):
        if os.path.basename(path) in RELEASED_FREQ.values():
            hint = ("download data_share.zip from the link in the README of "
                    "github.com/randomizedtree/segment-watermark and copy "
                    f"{os.path.basename(path)} to data/qu/.")
        else:
            hint = ("build it with: python -m src_baselines.qu_token_freq "
                    "--model <model id>")
        raise FileNotFoundError(
            f"{path} not found. Qu et al.'s balanced map needs a token frequency "
            f"file: {hint}")
    with open(path, "rb") as fh:
        return np.asarray(pickle.load(fh), dtype=np.int64)


def cached_mapping(p: int, key: int, vocab_size: int, freq=None,
                   freq_file: str = FREQ_FILE) -> np.ndarray:
    """Build once per (frequency file, p, key) and cache under data/qu/. Maps
    built from the released Llama file keep their original cache name."""
    if freq is not None:                             # explicit frequencies (tests)
        return build_mapping(freq, p, key)
    os.makedirs(DATA_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(freq_file))[0]
    tag = "" if freq_file == FREQ_FILE else stem.replace("token_freq_", "") + "_"
    path = os.path.join(DATA_DIR, f"map_{tag}p{p}_key{key}.npy")
    if os.path.exists(path):
        mapping = np.load(path)
    else:
        mapping = build_mapping(load_freq(freq_file), p, key)
        np.save(path, mapping)
    assert len(mapping) >= vocab_size, "frequency file shorter than the vocabulary"
    return mapping
