#!/usr/bin/env python3
"""MOSAIC's main-run texts decoded with the deployed detector.

The generation runs decoded every position. The deployed detector
(MosaicWatermark.detect_ids, distinct=True) keeps one position per distinct
(sorted window, token) pair, which makes the false-positive guarantee exact
(Appendix F) and removes the double counting of repeated observations. This
script re-decodes every stored text of a model's main result file with it and
writes

  derived/opengen_main_<model>_distinct.json  the main file with match, decoded,
      bit_accuracy, p_value, score, and agreement replaced by the deployed
      detector's (agreement: share of all positions agreeing with the decoded
      payload), the stored decisions kept as match_all and p_value_all, and the
      aggregate recomputed; every analysis reads this file;
  derived/mosaic_distinct_<model>.csv  one compact row per text, both decoders.

No language model is loaded: only the tokenizer and the key. Resource use: one
process, the main file streamed one sample at a time, the detector's mask cache
cleared per text, rows appended to a .tmp file so a rerun resumes; peak memory
well under 0.5 GB. The script exits at once when its output exists.

Usage:  python results/redecode_main.py [model id]   (default Llama-2-7B)
"""
import csv
import json
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
MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
TAG = MODEL.replace("/", "_")
MAIN = os.path.join(RES, f"opengen_main_{TAG}.json")
DEST = os.path.join(OUT, f"opengen_main_{TAG}_distinct.json")
CSV = os.path.join(OUT, f"mosaic_distinct_{TAG}.csv")
SOFT_MAG, WINDOW = 4.0, 2                    # detection settings of the main run
HEADER = ["prompt_index", "k", "match", "match_all", "p_value", "p_value_all",
          "num_distinct", "num_tokens"]


def config_and_aggregate(path):
    with open(path) as fh:
        head = fh.read(1 << 16)
    dec = json.JSONDecoder()
    cfg = dec.raw_decode(head, head.index("{", head.index('"config"')))[0]
    agg = dec.raw_decode(head, head.index("[", head.index('"aggregate"')))[0]
    return cfg, agg


def main():
    from src.mosaic import MosaicWatermark
    if os.path.exists(DEST) and os.path.exists(CSV):
        print("exists:", DEST)
        return
    os.makedirs(OUT, exist_ok=True)
    tmp = DEST + ".jsonl.tmp"
    done = set()
    if os.path.exists(tmp):                  # keep only complete lines
        with open(tmp) as f:
            lines = [ln for ln in f if ln.endswith("\n")]
        good = []
        for ln in lines:
            try:
                s = json.loads(ln)
            except ValueError:
                continue
            good.append(ln)
            done.add((s["prompt_index"], s["k"]))
        with open(tmp, "w") as f:
            f.writelines(good)
    print(f"resuming with {len(done)} decoded texts", flush=True)

    wm = MosaicWatermark(MODEL, secret_key="mosaic_opengen", k_bits=16,
                         soft_mag=SOFT_MAG, window=WINDOW, load_model=False)
    t0, n = time.time(), 0
    with open(tmp, "a") as f:
        for s in iter_samples(MAIN):
            if (s["prompt_index"], s["k"]) in done:
                continue
            wm.reconfigure(k_bits=s["k"])
            wm.prf.clear_cache()
            d = wm.detect_ids(s["generated_ids"], prompt_ids=wm.tokenizer(s["prompt"])["input_ids"])
            wm.prf.clear_cache()
            msg = np.array(s["message"])
            s.update({"match_all": s["match"], "p_value_all": s["p_value"],
                      "match": d["message"] == s["message"], "decoded": d["message"],
                      "bit_accuracy": float(np.mean(np.array(d["message"]) == msg)),
                      "p_value": d["p_value"], "score": d["score"],
                      "agreement": d["agreement_all"], "num_distinct": d["num_tokens"]})
            f.write(json.dumps(s) + "\n")
            n += 1
            if n % 500 == 0:
                f.flush()
                print(f"  {n} texts this run, {time.time() - t0:.0f}s", flush=True)

    cfg, agg = config_and_aggregate(MAIN)
    cfg["detector"] = "distinct (sorted window, token) pairs"
    rows = {}
    with open(tmp) as f:
        for ln in f:
            s = json.loads(ln)
            rows.setdefault(s["k"], []).append((s["match"], s["bit_accuracy"], s["p_value"]))
    for a in agg:
        r = rows[a["k"]]
        a.update({"match_rate": 100 * float(np.mean([x[0] for x in r])),
                  "bit_accuracy": 100 * float(np.mean([x[1] for x in r])),
                  "median_p_value": float(np.median([x[2] for x in r])), "n": len(r)})
    with open(DEST + ".part", "w") as out, open(CSV + ".part", "w", newline="") as cf, open(tmp) as f:
        w = csv.writer(cf)
        w.writerow(HEADER)
        out.write('{"config": ' + json.dumps(cfg) + ', "aggregate": ' + json.dumps(agg)
                  + ', "samples": [\n')
        for i, ln in enumerate(f):
            s = json.loads(ln)
            out.write((",\n" if i else "") + ln.rstrip("\n"))
            w.writerow([s["prompt_index"], s["k"], int(s["match"]), int(s["match_all"]),
                        s["p_value"], s["p_value_all"], s["num_distinct"],
                        len(s["generated_ids"])])
        out.write("\n]}\n")
    os.replace(CSV + ".part", CSV)
    os.replace(DEST + ".part", DEST)
    os.remove(tmp)
    print(f"wrote {DEST} and {CSV} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
