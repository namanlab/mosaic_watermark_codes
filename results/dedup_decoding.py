#!/usr/bin/env python3
"""MOSAIC decoding with repeated observations counted once.

A token whose sorted predecessor window and identity both repeat an earlier
position emits exactly the same observation of exactly the same check, so it
adds no new information about the payload. The deployed decoder nevertheless
passes every repeat to belief propagation as an independent check. This script
re-decodes the stored MOSAIC samples at the small payloads with each distinct
(window, token) observation kept once (tokens that differ under the same window
remain separate observations) and records the outcome, so the effect of
repetition on the decoder is measured on all prompts. Stored result files are
only read.

Output: results/derived/mosaic_dedup_decoding.csv, one row per (prompt,
payload). No language model is loaded.

Resource use (this runs on a 16 GB laptop): one process, no multiprocessing;
the main result file is streamed one sample at a time and the detector's mask
cache is cleared for every text, so memory stays flat. Rows are appended to a
.tmp file after every text and a rerun resumes from it; the finished file is
renamed from .tmp at the end, and the script exits at once when it exists.

Usage:  python results/dedup_decoding.py
"""
import csv
import os
import sys
import time

import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RES)
sys.path.insert(0, ROOT)
sys.path.insert(0, RES)
from stream_samples import iter_samples      # noqa: E402

OUT = os.path.join(RES, "derived")
MODEL = "meta-llama/Llama-2-7b-hf"
MAIN = os.path.join(RES, f"opengen_main_{MODEL.replace('/', '_')}.json")
SOFT_MAG, WINDOW, MAX_ITERS = 4.0, 2, 80     # detection settings of the main run
PAYLOADS = (16, 20)                          # the small-payload regime under study
HEADER = ["prompt_index", "k", "match", "match_dedup", "bit_errors_dedup", "checks_kept", "T"]
FIELDS = ("prompt_index", "prompt", "k", "message", "match", "generated_ids")


def main():
    from src.mosaic import MosaicWatermark
    from src.fountain_bp import bp_decode
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "mosaic_dedup_decoding.csv")
    if os.path.exists(path):
        print("exists:", path)
        return
    tmp = path + ".tmp"
    done = set()
    if os.path.exists(tmp):
        with open(tmp, newline="") as f:
            rows = [r for r in csv.reader(f)][1:]
        rows = [r for r in rows if len(r) == len(HEADER)]
        done = {(int(r[0]), int(r[1])) for r in rows}
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
    else:
        with open(tmp, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
    print(f"resuming with {len(done)} decoded samples", flush=True)

    wm = MosaicWatermark(MODEL, secret_key="mosaic_opengen", k_bits=PAYLOADS[0],
                         soft_mag=SOFT_MAG, window=WINDOW, load_model=False)
    t0, n = time.time(), 0
    with open(tmp, "a", newline="") as f:
        w = csv.writer(f)
        for s in iter_samples(MAIN, fields=FIELDS):
            key = (s["prompt_index"], s["k"])
            if s["k"] not in PAYLOADS or key in done:
                continue
            wm.reconfigure(k_bits=s["k"])
            wm.prf.clear_cache()
            prompt_ids = wm.tokenizer(s["prompt"])["input_ids"]
            checks, _ = wm._collect_checks(s["generated_ids"], prompt_ids)
            wm.prf.clear_cache()
            ctx, seen, kept = list(prompt_ids), set(), []
            for c, t in zip(checks, s["generated_ids"]):
                obs = (tuple(sorted(ctx[-WINDOW:])), int(t))
                ctx.append(t)
                if obs not in seen:
                    seen.add(obs)
                    kept.append(c)
            est = [int(b) for b in bp_decode(s["k"], kept, max_iters=MAX_ITERS)]
            w.writerow([s["prompt_index"], s["k"], int(s["match"]), int(est == s["message"]),
                        int(np.sum(np.array(est) != np.array(s["message"]))),
                        len(kept), len(checks)])
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {n} samples this run, {time.time() - t0:.0f}s", flush=True)
    os.replace(tmp, path)
    print(f"wrote {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
