#!/usr/bin/env python3
"""Robustness of MOSAIC and the six baselines under one attack protocol.

Every method's stored generations on the shared prompts 0-999 (the prompts the
baselines were run on) are attacked with the token-level attacks of
run_experiments_full.py (the same functions and strengths) and re-detected with
the method's own model-free detector. The attack randomness is seeded per
(prompt, payload, attack, strength), so all methods face the same edit pattern
on a given prompt (common random numbers). Stored result files are only read.

Output: results/derived/robustness_comparison.csv, one row per
(method, payload, prompt, attack, strength) with the exact-match outcome and the
bit accuracy. No language model is loaded: only the tokenizer, the keys, and
each detector.

Resource use (this runs on a 16 GB laptop): one process, no multiprocessing;
result files are streamed one sample at a time (stream_samples.py), so peak
memory stays in the low hundreds of MB. Rows are appended to a .tmp file after
every text, and a rerun resumes from it, keeping only (method, payload, prompt)
groups with all attack settings present. The finished file is renamed from .tmp
at the end, and the script exits at once when the finished file exists.

Usage:  python results/robustness_comparison.py
"""
import csv
import json
import os
import sys
import time
from collections import Counter

import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RES)
sys.path.insert(0, ROOT)
sys.path.insert(0, RES)
import run_experiments_full as rx            # noqa: E402  attack functions
import run_baseline_experiments as rb        # noqa: E402  baseline registry
from stream_samples import iter_samples      # noqa: E402

OUT = os.path.join(RES, "derived")
MODEL = "meta-llama/Llama-2-7b-hf"
TAG = MODEL.replace("/", "_")
MAIN = os.path.join(RES, f"opengen_main_{TAG}.json")
BASE = os.path.join(RES, "other_models", "opengen_baseline_{}_" + TAG + ".json")
N_COMMON = 1000
SEED = 42
# Low, middle, and high payload. The 2^k extraction of Fernandez et al. and Wang
# et al. makes 16 attack settings x 1,000 texts impractical at k = 20, so they
# enter at k = 16 only (their feasible payloads are 16 and 20).
PAYLOADS = (16, 32, 48)
ATTACKS = ([("clean", 0.0)]
           + [(a, e) for a in ("substitution", "insertion", "deletion")
              for e in rx.ATTACK_FRACTIONS]
           + [("reordering", float(b)) for b in rx.REORDER_BLOCKS])
ACODE = {"clean": 0, "substitution": 1, "insertion": 2, "deletion": 3, "reordering": 4}
HEADER = ["method", "k", "prompt_index", "attack", "strength", "match", "bit_accuracy"]
FIELDS = ("prompt_index", "k", "message", "generated_ids")


def attack(ids, name, strength, rng, vocab):
    if name == "clean":
        return list(ids)
    if name == "substitution":
        return rx.attack_substitute(ids, strength, vocab, rng)
    if name == "insertion":
        return rx.attack_insert(ids, strength, vocab, rng)
    if name == "deletion":
        return rx.attack_delete(ids, strength, rng)
    return rx.attack_reorder(ids, int(strength), rng)


def make_detector(method, k, vocab):
    if method == "mosaic":
        from src.mosaic import MosaicWatermark
        return MosaicWatermark(MODEL, secret_key="mosaic_opengen", k_bits=k,
                               soft_mag=rx.SOFT_MAG, window=rx.WINDOW, load_model=False)
    cls = rb.METHODS[method]
    d = cls(rb.SECRET_KEY, k, vocab) if method != "cyclic" else \
        cls(rb.SECRET_KEY, k, vocab, rb.GAMMA)
    d.reconfigure(k)
    return d


def load_done(tmp):
    """Complete (method, k, prompt) groups already in the .tmp file; rewrite the
    file to hold exactly those rows so appends never duplicate."""
    if not os.path.exists(tmp):
        with open(tmp, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
        return set()
    rows = []
    with open(tmp, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) == len(HEADER):
                rows.append(row)
    cnt = Counter((row[0], int(row[1]), int(row[2])) for row in rows)
    done = {g for g, c in cnt.items() if c == len(ATTACKS)}
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(row for row in rows if (row[0], int(row[1]), int(row[2])) in done)
    return done


def source_signatures():
    """Size and modification time of every result file the comparison reads."""
    src = {"mosaic": MAIN, **{m: BASE.format(m) for m in rb.METHODS}}
    return {m: [os.path.getsize(p), int(os.path.getmtime(p))] for m, p in src.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "robustness_comparison.csv")
    manifest = path + ".sources.json"
    tmp = path + ".tmp"
    sig = source_signatures()
    if os.path.exists(path):
        old = json.load(open(manifest)) if os.path.exists(manifest) else {}
        stale = sorted(m for m in sig if old.get(m) != sig[m])
        if not stale:
            print("up to date:", path)
            return
        # a method's result file changed (e.g. regenerated): keep the other
        # methods' rows and recompute only the changed ones
        print("recomputing changed methods:", stale, flush=True)
        with open(path, newline="") as f:
            rows = [r for r in csv.reader(f)][1:]
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(r for r in rows if r[0] not in stale)
        os.remove(path)
    done = load_done(tmp)
    print(f"resuming with {len(done)} complete (method, payload, prompt) groups", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    vocab = len(tok)
    pool_rows = rb.load_pool()
    pids = {}
    for i in range(N_COMMON):
        ids = tok(pool_rows[i][0], truncation=True, max_length=rb.PROMPT_MAX_TOKENS)["input_ids"]
        pids[i] = tok(tok.decode(ids, skip_special_tokens=True))["input_ids"]
    del pool_rows

    sources = [("mosaic", MAIN)] + [(m, BASE.format(m)) for m in rb.METHODS]
    t0, n_new = time.time(), 0
    with open(tmp, "a", newline="") as f:
        w = csv.writer(f)
        for method, src in sources:
            dets = {}
            for s in iter_samples(src, fields=FIELDS):
                k, pi = s["k"], s["prompt_index"]
                if pi >= N_COMMON or k not in PAYLOADS or (method, k, pi) in done:
                    continue
                if k not in dets:
                    dets[k] = make_detector(method, k, vocab)
                det = dets[k]
                rows = []
                for name, strength in ATTACKS:
                    rng = np.random.default_rng(
                        [SEED, pi, k, ACODE[name], int(round(strength * 100))])
                    ids = attack(s["generated_ids"], name, strength, rng, vocab)
                    if method == "mosaic":
                        det.prf.clear_cache()
                        dec = det.detect_ids(ids, prompt_ids=pids[pi])["message"]
                    else:
                        dec = det.detect(ids, pids[pi])["message"]
                    rows.append((method, k, pi, name, strength, int(dec == s["message"]),
                                 float(np.mean(np.array(dec) == np.array(s["message"])))))
                w.writerows(rows)
                f.flush()
                n_new += 1
                if n_new % 25 == 0:
                    print(f"  {method} k={k} prompt={pi}: {n_new} texts this run, "
                          f"{time.time() - t0:.0f}s", flush=True)
            if method == "mosaic":
                dets.clear()
            print(f"finished {method} ({time.time() - t0:.0f}s)", flush=True)
    os.replace(tmp, path)                      # only a complete run leaves the file
    json.dump(sig, open(manifest, "w"), indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
