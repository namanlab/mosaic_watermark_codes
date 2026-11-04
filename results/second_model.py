#!/usr/bin/env python3
"""MOSAIC and the baselines on a second generation model.

For every model in SECOND_MODELS whose MOSAIC result file exists, this reads the
MOSAIC file and the baseline files in other_models/ for that model (read only)
and writes derived/second_model_<tag>.json with

  rates       match rate of each method on the shared prompts with a 95%
              bootstrap interval (same resampling as sample_diagnostics.py);
  mcnemar     exact McNemar test of MOSAIC against each baseline, paired by
              (prompt, payload);
  pe          MOSAIC's per-token flip rate against the true payload, pooled and
              per payload size (the measured channel of the analysis);
  repetition  share of 16-bit texts with more than half of their sorted w = 2
              windows repeated, per method.

No language model is loaded: only that model's tokenizer and the keys. Resource
use: one process, result files streamed one sample at a time (stream_samples.py),
peak memory well under 0.5 GB. An existing output is kept; make_paper_assets.R
removes it when an input is newer.

Usage:  python results/second_model.py
"""
import json
import os
import sys

import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RES)
sys.path.insert(0, ROOT)
sys.path.insert(0, RES)
from stream_samples import iter_samples                                        # noqa: E402
from sample_diagnostics import (B, KS, SEED, SOFT_MAG, WINDOW, boot_ci,        # noqa: E402
                                distinct_windows, mcnemar_exact)

OUT = os.path.join(RES, "derived")
# model id -> short tag used in file names and paper macros
SECOND_MODELS = {"mistralai/Mistral-7B-v0.1": "mistral", "Qwen/Qwen2.5-7B": "qwen"}
BASELINES = ["qu", "janas", "mpac", "cyclic", "wang", "cohen_adapted", "cohen"]
N_COMMON = 1000


def files_for(model):
    tag = model.replace("/", "_")
    main = os.path.join(RES, f"opengen_main_{tag}.json")
    base = {m: os.path.join(RES, "other_models", f"opengen_baseline_{m}_{tag}.json")
            for m in BASELINES}
    return main, {m: p for m, p in base.items() if os.path.exists(p)}


def mosaic_true_agreement(wm, s):
    """Share of tokens whose colour matches the colour the TRUE payload requests.
    For a decoded text the stored agreement is that share already."""
    if s["match"]:
        return s["agreement"]
    wm.reconfigure(k_bits=s["k"])
    msg = np.array(s["message"], dtype=np.int8)
    ctx = list(wm.tokenizer(s["prompt"])["input_ids"])
    agree = 0
    for t in s["generated_ids"]:
        sub = wm.prf.check_subset(ctx)
        g = int(np.bitwise_xor.reduce(msg[sub])) ^ wm.prf.chip(ctx)
        col = bool(wm.prf.green_mask(ctx)[t]) if t < wm.vocab_size else False
        agree += int(col == bool(g))
        wm.prf.clear_cache()
        ctx.append(int(t))
    return agree / len(s["generated_ids"])


def prompt_ids(model):
    """Prompt ids exactly as the baseline runner built them for this model."""
    import run_baseline_experiments as rb
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    pool = rb.load_pool()
    out = {}
    for i in range(N_COMMON):
        ids = tok(pool[i][0], truncation=True, max_length=rb.PROMPT_MAX_TOKENS)["input_ids"]
        out[i] = tok(tok.decode(ids, skip_special_tokens=True))["input_ids"]
    return out


def analyse(model, tag):
    from src.mosaic import MosaicWatermark
    main, base = files_for(model)
    wm = MosaicWatermark(model, secret_key="mosaic_opengen", k_bits=16,
                         soft_mag=SOFT_MAG, window=WINDOW, load_model=False)
    pids = prompt_ids(model)
    match, pe, dup16 = {}, {k: [] for k in KS}, {}
    fields = ("prompt_index", "prompt", "k", "message", "match", "agreement", "generated_ids")
    for s in iter_samples(main, fields=fields):
        if s["prompt_index"] >= N_COMMON:
            continue
        match[("mosaic", s["prompt_index"], s["k"])] = bool(s["match"])
        pe[s["k"]].append(1 - mosaic_true_agreement(wm, s))
        wm.prf.clear_cache()
        if s["k"] == 16:
            T = len(s["generated_ids"])
            dup16.setdefault("mosaic", []).append(
                1 - distinct_windows(pids[s["prompt_index"]], s["generated_ids"]) / T)
    for m, path in base.items():
        for s in iter_samples(path, fields=("prompt_index", "k", "match", "generated_ids")):
            if s["prompt_index"] >= N_COMMON:
                continue
            match[(m, s["prompt_index"], s["k"])] = bool(s["match"])
            if s["k"] == 16:
                T = len(s["generated_ids"])
                dup16.setdefault(m, []).append(
                    1 - distinct_windows(pids[s["prompt_index"]], s["generated_ids"]) / T)
    rng = np.random.default_rng(SEED)
    res = {"model": model, "tag": tag, "prompts": [0, N_COMMON - 1], "B": B, "seed": SEED,
           "rates": [], "mcnemar": [], "repetition": [],
           "pe": {"pooled": float(np.mean([v for k in KS for v in pe[k]])),
                  "by_k": {str(k): float(np.mean(pe[k])) for k in KS if pe[k]}}}
    for m in ["mosaic"] + list(base):
        for k in KS:
            v = [match[(m, i, k)] for i in range(N_COMMON) if (m, i, k) in match]
            if v:
                lo, hi = boot_ci(v, rng)
                res["rates"].append({"method": m, "k": k, "n": len(v),
                                     "match_rate": 100 * float(np.mean(v)),
                                     "ci_lo": lo, "ci_hi": hi})
    for m in base:
        for k in KS:
            if (m, 0, k) in match:
                b = sum(match[("mosaic", i, k)] and not match[(m, i, k)] for i in range(N_COMMON))
                c = sum(match[(m, i, k)] and not match[("mosaic", i, k)] for i in range(N_COMMON))
                res["mcnemar"].append({"baseline": m, "k": k, "mosaic_only": b,
                                       "baseline_only": c, "p_value": mcnemar_exact(b, c)})
    for m, d in dup16.items():
        res["repetition"].append({"method": m, "n": len(d),
                                  "share50": 100 * float(np.mean(np.array(d) > 0.5))})
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    for model, tag in SECOND_MODELS.items():
        main_file, _ = files_for(model)
        path = os.path.join(OUT, f"second_model_{tag}.json")
        if not os.path.exists(main_file) or os.path.exists(path):
            continue
        print(f"analysing {model} ...", flush=True)
        json.dump(analyse(model, tag), open(path, "w"), indent=2)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
