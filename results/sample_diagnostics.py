#!/usr/bin/env python3
"""Per-sample diagnostics and the common-sample comparison.

Reads the stored result files in this folder (read only) and writes derived
outputs to results/derived/. Each output is skipped when it already exists, so
make_paper_assets.R can call this script cheaply on every run.

  derived/mosaic_sample_diagnostics.csv   one row per MOSAIC (prompt, payload):
      exact match, bit errors, true-channel agreement (share of tokens whose
      colour matches the colour the true payload requests, i.e. 1 - empirical
      p_e), distinct sorted windows, largest window multiplicity, distinct
      (window, token) pairs, and the fewest distinct checks covering any bit.
  derived/baseline_sample_repetition.csv  one row per baseline (method, prompt,
      payload): exact match and the number of distinct sorted w = 2 windows, so
      repetition is measured identically for every method.
  derived/common_sample.json               match rates of MOSAIC and the
      baselines on the shared prompts 0-999 with 95% bootstrap intervals, and
      exact McNemar tests of MOSAIC against each baseline paired by (prompt,
      payload).

No language model is loaded: everything runs on the stored token ids with the
tokenizer and the watermark keys. Resource use (this runs on a 16 GB laptop):
one process, result files streamed one sample at a time (stream_samples.py), and
the detector's mask cache cleared for every text, so memory stays flat.

Usage:  python results/sample_diagnostics.py
"""
import csv
import json
import math
import os
import sys

import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RES)
sys.path.insert(0, ROOT)
sys.path.insert(0, RES)
from stream_samples import iter_samples      # noqa: E402
OUT = os.path.join(RES, "derived")
MODEL = "meta-llama/Llama-2-7b-hf"
TAG = MODEL.replace("/", "_")
MAIN = os.path.join(RES, f"opengen_main_{TAG}.json")
BASE = os.path.join(RES, "other_models", "opengen_baseline_{}_" + TAG + ".json")
# cohen_adapted last: the bootstrap draws in this order, so adding it leaves the
# other methods' intervals unchanged
METHODS = ["mpac", "qu", "cohen", "janas", "cyclic", "wang", "cohen_adapted"]
KS = [16, 20, 24, 32, 48]
N_COMMON = 1000          # the baselines were run on prompts 0..999
B, SEED = 10000, 20240611
SOFT_MAG, WINDOW = 4.0, 2   # detection settings of run_experiments_full.py

F_DIAG = os.path.join(OUT, "mosaic_sample_diagnostics.csv")
F_REP = os.path.join(OUT, "baseline_sample_repetition.csv")
F_COMMON = os.path.join(OUT, "common_sample.json")


# ------------------------------------------------------------------ workers --
_wm = None


def _init():
    global _wm
    from src.mosaic import MosaicWatermark
    _wm = MosaicWatermark(MODEL, secret_key="mosaic_opengen", k_bits=16,
                          soft_mag=SOFT_MAG, window=WINDOW, load_model=False)


def _features(s):
    wm = _wm
    k = s["k"]
    wm.reconfigure(k_bits=k)
    wm.prf.clear_cache()
    msg = np.array(s["message"], dtype=np.int8)
    ids = [int(t) for t in s["generated_ids"]]
    T = len(ids)
    ctx = list(wm.tokenizer(s["prompt"])["input_ids"])
    mult, pairs, seen = {}, set(), set()
    cover = np.zeros(k, dtype=np.int64)
    agree_true = 0
    for t in ids:
        w = tuple(sorted(ctx[-WINDOW:]))
        mult[w] = mult.get(w, 0) + 1
        pairs.add((w, t))
        sub = wm.prf.check_subset(ctx)
        if w not in seen:
            seen.add(w)
            cover[sub] += 1
        if not s["match"]:
            # the stored agreement is measured against the decoded payload; for
            # a failure recompute it against the true payload
            r = wm.prf.chip(ctx)
            g = int(np.bitwise_xor.reduce(msg[sub])) ^ r
            col = bool(wm.prf.green_mask(ctx)[t]) if t < wm.vocab_size else False
            agree_true += int(col == bool(g))
            wm.prf.clear_cache()
        ctx.append(t)
    a_true = agree_true / T if not s["match"] else s["agreement"]
    return {
        "prompt_index": s["prompt_index"], "k": k, "match": int(s["match"]),
        "bit_errors": int(np.sum(np.array(s["decoded"]) != msg)),
        "agreement_decoded": s["agreement"], "agreement_true": a_true,
        "T": T, "distinct_windows": len(mult), "max_window_repeat": max(mult.values()),
        "distinct_pairs": len(pairs), "min_bit_cover": int(cover.min()),
        "p_value": s["p_value"],
    }


# ------------------------------------------------------------------ helpers --
def prompt_ids_common():
    """Prompt ids exactly as the baseline runner built them (pool order)."""
    import run_baseline_experiments as rb
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    pool = rb.load_pool()
    out = {}
    for i in range(N_COMMON):
        ids = tok(pool[i][0], truncation=True, max_length=rb.PROMPT_MAX_TOKENS)["input_ids"]
        out[i] = tok(tok.decode(ids, skip_special_tokens=True))["input_ids"]
    return out


def distinct_windows(prompt_ids, gen_ids, w=WINDOW):
    ctx, seen = list(prompt_ids), set()
    for t in gen_ids:
        seen.add(tuple(sorted(ctx[-w:])))
        ctx.append(t)
    return len(seen)


def boot_ci(v, rng):
    v = np.asarray(v, dtype=float)
    m = np.empty(B)                      # blocks of 500 rows; same random stream
    for a in range(0, B, 500):
        e = min(B, a + 500)
        m[a:e] = v[rng.integers(0, len(v), size=(e - a, len(v)))].mean(axis=1) * 100
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows), "rows", flush=True)


# --------------------------------------------------------------------- main --
def main():
    os.makedirs(OUT, exist_ok=True)
    main_fields = ("prompt_index", "prompt", "k", "message", "decoded", "match",
                   "agreement", "p_value", "generated_ids")

    if not os.path.exists(F_DIAG):
        _init()
        rows = [_features(s) for s in iter_samples(MAIN, fields=main_fields)]
        write_csv(F_DIAG, rows)
        del rows

    if not os.path.exists(F_REP):
        pids = prompt_ids_common()
        rows = [{"method": m, "prompt_index": s["prompt_index"], "k": s["k"],
                 "match": int(s["match"]), "T": len(s["generated_ids"]),
                 "distinct_windows": distinct_windows(pids[s["prompt_index"]],
                                                      s["generated_ids"])}
                for m in METHODS
                for s in iter_samples(BASE.format(m),
                                      fields=("prompt_index", "k", "match", "generated_ids"))]
        write_csv(F_REP, rows)

    if not os.path.exists(F_COMMON):
        rng = np.random.default_rng(SEED)
        match = {("mosaic", s["prompt_index"], s["k"]): bool(s["match"])
                 for s in iter_samples(MAIN, fields=("prompt_index", "k", "match"))
                 if s["prompt_index"] < N_COMMON}
        for m in METHODS:
            for s in iter_samples(BASE.format(m), fields=("prompt_index", "k", "match")):
                match[(m, s["prompt_index"], s["k"])] = bool(s["match"])
        res = {"prompts": [0, N_COMMON - 1], "B": B, "seed": SEED, "rates": [], "mcnemar": []}
        for m in ["mosaic"] + METHODS:
            for k in KS:
                v = [match[(m, i, k)] for i in range(N_COMMON) if (m, i, k) in match]
                if v:
                    lo, hi = boot_ci(v, rng)
                    res["rates"].append({"method": m, "k": k, "n": len(v),
                                         "match_rate": 100 * float(np.mean(v)),
                                         "ci_lo": lo, "ci_hi": hi})
        for m in METHODS:
            for k in KS:
                if (m, 0, k) in match:
                    b = sum(match[("mosaic", i, k)] and not match[(m, i, k)]
                            for i in range(N_COMMON))
                    c = sum(match[(m, i, k)] and not match[("mosaic", i, k)]
                            for i in range(N_COMMON))
                    res["mcnemar"].append({"baseline": m, "k": k, "mosaic_only": b,
                                           "baseline_only": c, "p_value": mcnemar_exact(b, c)})
        json.dump(res, open(F_COMMON, "w"), indent=2)
        print("wrote", F_COMMON, flush=True)


if __name__ == "__main__":
    main()
