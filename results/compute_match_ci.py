#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals for the exact-match rate.

Reads the per-prompt "match" flags from a main-results JSON and resamples over
prompts (nonparametric bootstrap) to get a 95% CI for the match rate at each
payload size. Writes a small JSON that make_paper_assets.R reads for the "+/-95%"
column of the main results table, so the CI is produced by the same reproducible
pipeline as every other number in the paper.

Usage:
    python compute_match_ci.py [main_json] [out_json]

Defaults:
    main_json = opengen_main_meta-llama_Llama-2-7b-hf.json  (this folder)
    out_json  = main_bootstrap_ci.json                      (this folder)

The samples are streamed one at a time (stream_samples.py), keeping only the
"k" and "match" fields, so memory stays small on a 16 GB laptop.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stream_samples import iter_samples  # noqa: E402
DEFAULT_IN = os.path.join(HERE, "opengen_main_meta-llama_Llama-2-7b-hf.json")
DEFAULT_OUT = os.path.join(HERE, "main_bootstrap_ci.json")

KS = (16, 20, 24, 32, 48)
B = 10000            # bootstrap resamples
SEED = 20240611      # fixed for reproducibility


def main(in_path=DEFAULT_IN, out_path=DEFAULT_OUT):
    # stream one sample at a time (the file is 147 MB); keep only k and match
    pairs = [(s["k"], float(s["match"]))
             for s in iter_samples(in_path, fields=("k", "match"))]
    ks = np.array([k for k, _ in pairs])
    matches = np.array([m for _, m in pairs])
    assert len(ks) == len(matches) and len(ks) > 0, "sample extraction failed"

    rng = np.random.default_rng(SEED)
    out = {}
    for k in KS:
        v = matches[ks == k]
        n = len(v)
        if n == 0:
            continue
        # draw the (B, n) resampling indices in blocks of 500 rows (about 31 MB)
        # instead of one 0.6 GB array; the random stream is identical
        boot = np.empty(B)
        for a in range(0, B, 500):
            e = min(B, a + 500)
            boot[a:e] = v[rng.integers(0, n, size=(e - a, n))].mean(axis=1) * 100
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[str(k)] = {
            "n": int(n),
            "match": round(float(v.mean() * 100), 4),
            "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4),
            "half": round(float((hi - lo) / 2), 4),
            "B": B,
            "seed": SEED,
        }
        print(f"k={k:2d} n={n} match={v.mean()*100:5.2f} "
              f"95%CI=[{lo:5.2f},{hi:5.2f}] half={ (hi-lo)/2:.2f}")

    json.dump(out, open(out_path, "w"), indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args) if args else main()
