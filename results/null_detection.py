#!/usr/bin/env python3
"""False positives: MOSAIC's detector on text that carries no watermark.

For the shared prompts 0-999 of a model, the unwatermarked generation from the
same prompt (plain_text) and the human continuation (reference) are scored at
every payload size, with the deployed detector (one position per distinct
(window, token) pair) and with the detector that counts every position. Under
the null the deployed p-value is valid (Appendix F); the second shows what
repeated observations do to it.

Output: derived/null_detection_<model>.csv, one row per (source, prompt, payload)
with both p-values and both counts. No language model is loaded. Resource use:
one process, the main file streamed, the mask cache cleared per text, rows
appended to a .tmp file so a rerun resumes; peak memory well under 0.5 GB.

Usage:  python results/null_detection.py [model id]   (default Llama-2-7B)
"""
import csv
import os
import sys
import time

RES = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RES)
sys.path.insert(0, ROOT)
sys.path.insert(0, RES)
from stream_samples import iter_samples      # noqa: E402

OUT = os.path.join(RES, "derived")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
TAG = MODEL.replace("/", "_")
MAIN = os.path.join(RES, f"opengen_main_{TAG}.json")
PATH = os.path.join(OUT, f"null_detection_{TAG}.csv")
N_COMMON, KS = 1000, (16, 20, 24, 32, 48)
SOFT_MAG, WINDOW = 4.0, 2
HEADER = ["source", "prompt_index", "k", "p_value", "num_distinct", "p_value_all", "num_tokens"]


def main():
    from src.mosaic import MosaicWatermark
    if os.path.exists(PATH):
        print("exists:", PATH)
        return
    os.makedirs(OUT, exist_ok=True)
    tmp = PATH + ".tmp"
    done = set()
    if os.path.exists(tmp):
        with open(tmp, newline="") as f:
            rows = [r for r in csv.reader(f)][1:]
        rows = [r for r in rows if len(r) == len(HEADER)]
        done = {(r[0], int(r[1]), int(r[2])) for r in rows}
        with open(tmp, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
            csv.writer(f).writerows(rows)
    else:
        with open(tmp, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
    wm = MosaicWatermark(MODEL, secret_key="mosaic_opengen", k_bits=16,
                         soft_mag=SOFT_MAG, window=WINDOW, load_model=False)
    t0, n = time.time(), 0
    with open(tmp, "a", newline="") as f:
        w = csv.writer(f)
        for s in iter_samples(MAIN, fields=("prompt_index", "prompt", "k", "plain_text", "reference")):
            pi = s["prompt_index"]
            if pi >= N_COMMON or s["k"] != KS[0]:      # the texts are the same for every payload
                continue
            pids = wm.tokenizer(s["prompt"])["input_ids"]
            for src in ("plain_text", "reference"):
                ids = wm.tokenizer(s[src] or "", add_special_tokens=False)["input_ids"]
                if not ids:
                    continue
                for k in KS:
                    if (src, pi, k) in done:
                        continue
                    wm.reconfigure(k_bits=k)
                    d = wm.detect_ids(ids, prompt_ids=pids)
                    a = wm.detect_ids(ids, prompt_ids=pids, distinct=False)
                    wm.prf.clear_cache()
                    w.writerow([src, pi, k, d["p_value"], d["num_tokens"], a["p_value"], len(ids)])
                    n += 1
            f.flush()
            if pi % 50 == 0:
                print(f"  prompt {pi}: {n} detections this run, {time.time() - t0:.0f}s", flush=True)
    os.replace(tmp, PATH)
    print(f"wrote {PATH} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
