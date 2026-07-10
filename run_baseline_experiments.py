"""Main match-rate experiment for three popular multi-bit watermark baselines,
run under the EXACT protocol of run_experiments_full.py stage 1:

  data      OpenGen parquet, prompts truncated to 128 tokens (same tokenizer)
  model     meta-llama/Llama-2-7b-hf, fp16
  decoding  nucleus sampling, temperature 0.8, top-p 0.95, eos masked
  bias      constant logit bias delta = 6, gamma = 0.5 where applicable
  payloads  16, 20, 24, 32, 48 bits, message_for(i, k) with the same seeding
  tokens    200 new tokens per generation
  detection model-free, from the generated token ids

Methods (see src_baselines/), all six from Janas et al. Table 1:
  mpac    Yoo et al., NAACL 2024      position allocation, majority vote
  qu      Qu et al., USENIX Sec 2025  segment assignment, COUNT argmax
  cyclic  Fernandez et al., WIFS 2023 key shift, 2^k scan (k <= 20 here)
  wang    Wang et al., ICLR 2024      message hash, 2^k scan (k <= 20 here)
  cohen   Cohen et al., S&P 2025      2L keyed green-lists, per-bit argmax
  janas   Janas et al., 2025          Lagrange line over GF(2^t) + MCP decode

Usage:
  python run_baseline_experiments.py --method all   --model meta-llama/Llama-2-7b-hf
  python run_baseline_experiments.py --method mpac  --model meta-llama/Llama-2-7b-hf

Env knobs (same semantics as run_experiments_full.py):
  MAIN_MAX_PROMPTS (0 = all, else first N in order)   CHECKPOINT_EVERY (10)
  LOG_EVERY (10)   MOSAIC_QUICK=1 (3 prompts, gpt2-friendly, _quick suffix)

Output: results/opengen_baseline_<method>_<model>.json
  {config, aggregate: [{k, match_rate, bit_accuracy, n}], samples: [...]}
Existing MOSAIC files and code are untouched; this script only reads the
shared dataset and writes its own result files. Checkpoints every
CHECKPOINT_EVERY prompts and resumes automatically from its own output.
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src_baselines.common import BaselineLM                     # noqa: E402
from src_baselines.mpac import MPACWatermark                    # noqa: E402
from src_baselines.qu_segment import QuSegmentWatermark         # noqa: E402
from src_baselines.cyclic_shift import CyclicShiftWatermark     # noqa: E402
from src_baselines.wang_hash import WangHashWatermark           # noqa: E402
from src_baselines.cohen import CohenWatermark                  # noqa: E402
from src_baselines.janas_lagrange import JanasWatermark         # noqa: E402

# ---------------------------------------------------------------- protocol --
# These constants intentionally mirror run_experiments_full.py one for one.
DATA_PATH        = os.path.join(os.path.dirname(__file__), "data", "opengen_data.parquet")
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "results")
SECRET_KEY       = "mosaic_opengen"
WATERMARK_SIZES  = [16, 20, 24, 32, 48]
DELTA            = 6.0
GAMMA            = 0.5
TEMPERATURE      = 0.8
TOP_P            = 0.95
MAX_NEW_TOKENS   = 200
PROMPT_MAX_TOKENS = 128
SEED             = 42
QUICK            = os.environ.get("MOSAIC_QUICK") == "1"
MAIN_MAX_PROMPTS = int(os.environ.get("MAIN_MAX_PROMPTS", "0"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "10"))
LOG_EVERY        = int(os.environ.get("LOG_EVERY", "10"))

METHODS = {"mpac": MPACWatermark, "qu": QuSegmentWatermark,
           "cyclic": CyclicShiftWatermark, "wang": WangHashWatermark,
           "cohen": CohenWatermark, "janas": JanasWatermark}

log = logging.getLogger("baselines")


def setup_logging(tag, model):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logfile = os.path.join(RESULTS_DIR, "run_baseline_%s_%s_%s.log" % (
        tag, model.replace("/", "_"), datetime.now().strftime("%Y%m%d_%H%M%S")))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(logfile)])
    log.info("logging to %s", logfile)


def message_for(i, k):                       # identical to run_experiments_full
    return np.random.default_rng([SEED, k, i]).integers(0, 2, k).tolist()


def out_path(model, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "_quick" if QUICK else ""
    return os.path.join(RESULTS_DIR, f"{name}_{model.replace('/', '_')}{suffix}.json")


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _fmt_dt(sec):
    return "%dm%02ds" % (sec // 60, sec % 60)


def load_pool():                             # identical to run_experiments_full
    t = pq.read_table(DATA_PATH, columns=["prefix", "targets"])
    pre = t.column("prefix").to_pylist()
    tgt = t.column("targets").to_pylist()
    pool = []
    for p, g in zip(pre, tgt):
        if isinstance(p, str) and p.strip():
            ref = ""
            try:
                arr = json.loads(g) if isinstance(g, str) else g
                if arr:
                    ref = str(arr[0])
            except Exception:
                ref = ""
            pool.append((p, ref))
    return pool


def main_indices(total):
    if QUICK:
        return list(range(min(3, total)))
    if 0 < MAIN_MAX_PROMPTS < total:
        return list(range(MAIN_MAX_PROMPTS))
    return list(range(total))


def build_prompts(lm, indices):
    """Truncated prompts, identical to prompts_refs in the MOSAIC runner."""
    pool = load_pool()
    prompts = []
    for i in indices:
        ids = lm.tokenizer(pool[i][0], truncation=True,
                           max_length=PROMPT_MAX_TOKENS)["input_ids"]
        prompts.append(lm.tokenizer.decode(ids, skip_special_tokens=True))
    return prompts


def aggregate(samples, ks):
    agg = []
    for k in ks:
        rows = [s for s in samples if s["k"] == k]
        if rows:
            agg.append({"k": k,
                        "match_rate": float(np.mean([r["match"] for r in rows]) * 100),
                        "bit_accuracy": float(np.mean([r["bit_accuracy"] for r in rows]) * 100),
                        "n": len(rows)})
    return agg


def save(model, method, ks, samples, t0):
    payload = {"config": {"model": model, "method": method, "delta": DELTA,
                          "gamma": GAMMA, "temperature": TEMPERATURE,
                          "top_p": TOP_P, "tokens": MAX_NEW_TOKENS,
                          "watermark_sizes": ks, "dataset": "OpenGen",
                          "context": "previous token (h=1)",
                          "n_prompts": len(set(s["prompt_index"] for s in samples))},
               "aggregate": aggregate(samples, ks), "samples": samples,
               "elapsed_sec": time.time() - t0}
    _atomic_write(out_path(model, f"opengen_baseline_{method}"), payload)


def run_method(lm, model, method):
    cls = METHODS[method]
    ks = [k for k in WATERMARK_SIZES if k in cls.feasible_ks]
    skipped = [k for k in WATERMARK_SIZES if k not in cls.feasible_ks]
    if skipped:
        log.info("[%s] payloads %s skipped: extraction infeasible (2^k scan)",
                 method, skipped)
    wm = cls(SECRET_KEY, ks[0], lm.vocab_size) if method != "cyclic" else \
        cls(SECRET_KEY, ks[0], lm.vocab_size, GAMMA)

    pool_n = len(load_pool())
    idx = main_indices(pool_n)
    path = out_path(model, f"opengen_baseline_{method}")

    # resume: keep prompts that already have every payload size
    samples, done = [], set()
    if os.path.exists(path):
        try:
            prev = json.load(open(path))
            cnt = {}
            for s in prev.get("samples", []):
                cnt[s["prompt_index"]] = cnt.get(s["prompt_index"], 0) + 1
            done = {pi for pi, c in cnt.items() if c >= len(ks)}
            samples = [s for s in prev["samples"] if s["prompt_index"] in done]
            log.info("[%s] RESUME from %s: %d prompts already complete",
                     method, path, len(done))
        except Exception as e:
            log.warning("[%s] could not resume from %s (%s); starting fresh",
                        method, path, e)

    prompts = build_prompts(lm, idx)
    todo = [j for j, i in enumerate(idx) if i not in done]
    log.info("[%s] %d prompts total, %d done, %d to do; ks=%s; checkpoint every %d",
             method, len(idx), len(done), len(todo), ks, CHECKPOINT_EVERY)
    run_match = {k: [s["match"] for s in samples if s["k"] == k] for k in ks}
    t0 = time.time()
    for n, j in enumerate(todo, 1):
        i, prompt = idx[j], prompts[j]
        for k in ks:
            wm.reconfigure(k)
            msg = message_for(i, k)
            gen = lm.generate(prompt, wm, msg, MAX_NEW_TOKENS, DELTA)
            det = wm.detect(gen["generated_ids"], gen["prompt_ids"])
            match = bool(det["message"] == msg)
            run_match[k].append(match)
            samples.append({
                "prompt_index": i, "k": k, "message": msg,
                "decoded": det["message"], "match": match,
                "bit_accuracy": float(np.mean(np.array(det["message"]) == np.array(msg))),
                "agreement": det["agreement"], "num_tokens": det["num_tokens"],
                "generated_ids": gen["generated_ids"],
                "watermarked_text": gen["generated_text"],
            })
        if n % CHECKPOINT_EVERY == 0:
            save(model, method, ks, samples, t0)
            log.info("[%s] CHECKPOINT saved (%d/%d prompts) -> %s",
                     method, len(done) + n, len(idx), path)
        if n % LOG_EVERY == 0 or n == len(todo):
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            mr = ", ".join(f"k{k}={np.mean(run_match[k]) * 100:.0f}%" for k in ks)
            log.info("[%s] %d/%d prompts  elapsed %s  eta %s  match [%s]",
                     method, len(done) + n, len(idx), _fmt_dt(el), _fmt_dt(eta), mr)
    save(model, method, ks, samples, t0)
    for a in aggregate(samples, ks):
        log.info("[%s] k=%-2d  match=%6.2f%%  bitacc=%6.2f%%  n=%d",
                 method, a["k"], a["match_rate"], a["bit_accuracy"], a["n"])
    log.info("[%s] DONE -> %s", method, path)


def main():
    ap = argparse.ArgumentParser(description="Baseline watermark match-rate experiments")
    ap.add_argument("--method", required=True, choices=list(METHODS) + ["all"])
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    args = ap.parse_args()

    setup_logging(args.method, args.model)
    if not os.path.exists(DATA_PATH):
        log.error("OpenGen parquet not found at %s", DATA_PATH)
        sys.exit(1)
    methods = list(METHODS) if args.method == "all" else [args.method]
    log.info("methods=%s model=%s MAIN_MAX_PROMPTS=%s QUICK=%s delta=%s "
             "temp=%s top_p=%s tokens=%d", methods, args.model,
             MAIN_MAX_PROMPTS or "ALL", QUICK, DELTA, TEMPERATURE, TOP_P,
             MAX_NEW_TOKENS)
    lm = BaselineLM(args.model, TEMPERATURE, TOP_P)
    t0 = time.time()
    for m in methods:
        try:
            run_method(lm, args.model, m)
        except Exception:
            log.exception("method '%s' failed", m)
            raise
    log.info("all done in %s", _fmt_dt(time.time() - t0))


if __name__ == "__main__":
    main()
