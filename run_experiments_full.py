"""
MOSAIC experiments on the FULL OpenGen dataset, split into four stages.
Each stage (and sub-stage) is one --experiment value, so they run independently
and only one heavy model is in memory at a time.

  STAGE 1  generate            generation model    prompts + watermark, save outputs
  STAGE 2  quality / quality-perplexity   oracle + MAUVE   perplexity (Llama-3.1) + MAUVE
  STAGE 3  robust-substitution / robust-insertion / robust-deletion /
           robust-reordering / robust-paraphrase   one attack each
           robust              runs all attacks; results merge into one file
           (robust-paraphrase generates DIPPER paraphrases on demand, then detects;
            all other attacks are model-free)
  STAGE 4  sweep-delta         gen model + oracle  match rate + perplexity vs delta
           sweep-tokens        gen model           bits x tokens heatmap
           sweep               runs both

Dependencies: stage 1 first; stages 2, 3, 4 all need stage 1. robust-paraphrase
self-generates its paraphrases (DIPPER), so it has no extra prerequisite.

Env knobs: MAIN_MAX_PROMPTS, SUBSET_FRAC, ORACLE_MODEL, PARAPHRASE_MODEL,
PARAPHRASE_TOK, QUANTIZE_8BIT. Smoke test: prepend MOSAIC_QUICK=1.
"""

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time

import numpy as np
import torch
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from src.mosaic import MosaicWatermark

# ----------------------------------------------------------------------
# logging + fixed settings (self-contained; the base script lives in old/)
# ----------------------------------------------------------------------
log = logging.getLogger("mosaic")


def setup_logging(experiment, model):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(
        RESULTS_DIR, f"run_{experiment}_{model.replace('/', '_')}_{stamp}.log")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(logfile)):
        h.setFormatter(fmt); log.addHandler(h)
    log.propagate = False
    log.info("logging to %s", logfile)
    return logfile


DATA_PATH        = os.path.join(os.path.dirname(__file__), "data", "opengen_data.parquet")
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "results")
SECRET_KEY       = "mosaic_opengen"
WATERMARK_SIZES  = [16, 20, 24, 32, 48]
DELTA            = 6.0
GAMMA            = 0.5
# Decoding matches the segment-watermark / Qu et al (arXiv:2401.16820) protocol:
# temperature 0.8 with nucleus (top-p) 0.95. The top-p cutoff truncates the long
# green-list tail before sampling, which keeps perplexity low at delta=6 (pure
# temperature-1.0 multinomial sampling instead samples deep into that tail and
# degrades quality badly). Set TEMPERATURE=0 for greedy decoding.
TEMPERATURE      = 0.8
TOP_P            = 0.95
WINDOW           = 2
SOFT_MAG         = 4.0
MAX_NEW_TOKENS   = 200
PROMPT_MAX_TOKENS = 128
SEED             = 42
PLOT_K           = 32
ATTACK_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
REORDER_BLOCKS   = [2, 5, 10]
QUICK            = os.environ.get("MOSAIC_QUICK") == "1"


def make_wm(model, k, delta=DELTA, window=WINDOW, soft_mag=SOFT_MAG, load_model=True):
    return MosaicWatermark(model_name=model, secret_key=SECRET_KEY, k_bits=k,
                           delta=delta, gamma=GAMMA, temperature=TEMPERATURE,
                           top_p=TOP_P, window=window, soft_mag=soft_mag,
                           load_model=load_model)


def message_for(i, k):
    return np.random.default_rng([SEED, k, i]).integers(0, 2, k).tolist()


def out_path(model, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "_quick" if QUICK else ""
    return os.path.join(RESULTS_DIR, f"{name}_{model.replace('/', '_')}{suffix}")


def attack_substitute(ids, eps, vocab, rng):
    ids = list(ids); n = int(round(eps * len(ids)))
    for p in rng.choice(len(ids), size=n, replace=False):
        ids[p] = int(rng.integers(0, vocab))
    return ids


def attack_insert(ids, eps, vocab, rng):
    ids = list(ids); n = int(round(eps * len(ids)))
    for _ in range(n):
        ids.insert(int(rng.integers(0, len(ids) + 1)), int(rng.integers(0, vocab)))
    return ids


def attack_delete(ids, eps, rng):
    ids = list(ids); n = int(round(eps * len(ids)))
    drop = set(rng.choice(len(ids), size=n, replace=False).tolist())
    return [t for j, t in enumerate(ids) if j not in drop]


def attack_reorder(ids, n_blocks, rng):
    ids = list(ids); L = len(ids)
    if L < n_blocks:
        return ids
    cuts = [0] + sorted(rng.choice(range(1, L), size=n_blocks - 1, replace=False).tolist()) + [L]
    blocks = [ids[cuts[j]:cuts[j + 1]] for j in range(n_blocks)]
    return [t for o in rng.permutation(n_blocks) for t in blocks[o]]

# ----------------------------------------------------------------------
# extended fixed settings
# ----------------------------------------------------------------------
SUBSET_FRAC       = float(os.environ.get("SUBSET_FRAC", "0.15"))   # robustness/plots subset
MAIN_MAX_PROMPTS  = int(os.environ.get("MAIN_MAX_PROMPTS", "0"))   # 0 = all prompts
DELTA_SWEEP_FULL  = list(range(2, 11))         # delta = 1,2,...,10 (interval 1)
TOKEN_SWEEP       = [100, 125, 150, 175, 200, 225]
HEATMAP_TOKENS    = TOKEN_SWEEP
HEATMAP_KS        = WATERMARK_SIZES
ROB_LOG_EVERY     = 5                          # robustness progress every N prompts
LOG_EVERY         = int(os.environ.get("LOG_EVERY", "10"))   # progress cadence
CHECKPOINT_EVERY  = int(os.environ.get("CHECKPOINT_EVERY", "10"))  # save + resume cadence

# quality models (override via env). QUANTIZE_8BIT=1 loads the big models in
# 8-bit (needs bitsandbytes) so DIPPER-XXL fits on a 24 GB GPU.
ORACLE_MODEL      = os.environ.get("ORACLE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
MAUVE_FEATURIZE   = os.environ.get("MAUVE_FEATURIZE", "gpt2-large")
MAUVE_MIN_SAMPLES = 20
QUANTIZE_8BIT     = os.environ.get("QUANTIZE_8BIT", "0") == "1"

# DIPPER paraphraser. On a 24 GB GPU (e.g. A10G) use the XL variant
# (kalpeshk2011/dipper-paraphraser-xl, tokenizer google/t5-v1_1-xl) or set
# QUANTIZE_8BIT=1 for the XXL.
PARAPHRASE_MODEL  = os.environ.get("PARAPHRASE_MODEL", "kalpeshk2011/dipper-paraphraser-xxl")
PARAPHRASE_TOK    = os.environ.get("PARAPHRASE_TOK", "google/t5-v1_1-xxl")
DIPPER_LEX        = 20
DIPPER_ORDER      = 20

if QUICK:                                        # tiny stand-ins for a laptop smoke test
    SUBSET_FRAC = 1.0
    DELTA_SWEEP_FULL = [2, 6]
    TOKEN_SWEEP = [100, 200]; HEATMAP_TOKENS = TOKEN_SWEEP; HEATMAP_KS = WATERMARK_SIZES
    ROB_LOG_EVERY = 1
    ORACLE_MODEL = "gpt2"
    MAUVE_FEATURIZE = "gpt2"; MAUVE_MIN_SAMPLES = 2
    PARAPHRASE_MODEL = "t5-small"; PARAPHRASE_TOK = "t5-small"


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _atomic_write(path, obj):
    """Write JSON to a temp file then rename, so an interrupt never corrupts it."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _read_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _device_id(device):
    return 0 if device == "cuda" else -1


def _load_kwargs(device):
    """Loading kwargs: 8-bit (device_map auto) if requested, else fp16 on device."""
    if QUANTIZE_8BIT and device == "cuda":
        return {"load_in_8bit": True, "device_map": "auto"}, True
    return ({"torch_dtype": torch.float16} if device in ("cuda", "mps") else {}), False


# ----------------------------------------------------------------------
# data loading: OpenGen prefixes (prompts) + targets (human references)
# ----------------------------------------------------------------------
def load_pool():
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


def prompts_refs(wm, indices):
    """Truncated prompts and human references for the given pool indices."""
    pool = load_pool()
    prompts, refs = [], []
    for i in indices:
        prefix, ref = pool[i]
        ids = wm.tokenizer(prefix, truncation=True, max_length=PROMPT_MAX_TOKENS)["input_ids"]
        prompts.append(wm.tokenizer.decode(ids, skip_special_tokens=True))
        refs.append(ref)
    return prompts, refs


def dataset_size():
    return len(load_pool())


def subset_indices(n, frac, seed=SEED):
    m = max(1, int(math.ceil(frac * n)))
    return sorted(np.random.default_rng([seed, 7]).choice(n, size=min(m, n),
                                                          replace=False).tolist())


# ----------------------------------------------------------------------
# Oracle perplexity (independent model, default Llama-3.1-8B-Instruct)
# ----------------------------------------------------------------------
class OracleScorer:
    def __init__(self, oracle_name, device, share=None):
        if share is not None:
            self.tok, self.model, self.device = share[0], share[1], device
            log.info("[oracle] reusing provided model for perplexity")
        else:
            log.info("[oracle] loading perplexity oracle: %s%s", oracle_name,
                     " (8-bit)" if QUANTIZE_8BIT else "")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(oracle_name)
            kw, quant = _load_kwargs(device)
            self.model = AutoModelForCausalLM.from_pretrained(oracle_name, **kw)
            if not quant:
                self.model.to(device)
            self.model.eval()
            self.device = "cuda" if quant else device

    @torch.no_grad()
    def perplexity(self, prompt_text, gen_text):
        if not gen_text.strip():
            return float("nan")
        pids = self.tok(prompt_text)["input_ids"]
        gids = self.tok(gen_text, add_special_tokens=False)["input_ids"]
        if len(gids) == 0:
            return float("nan")
        cap = getattr(self.model.config, "max_position_embeddings", 2048) or 2048
        full = (pids + gids)[:cap]
        ids = torch.tensor([full], device=self.device)
        logits = self.model(ids).logits[0].float()
        logprobs = torch.log_softmax(logits[:-1], dim=-1)
        targets = torch.tensor(full[1:], device=self.device)
        tok_lp = logprobs[torch.arange(len(targets)), targets]
        start = min(len(pids), len(full) - 1)
        gen_lp = tok_lp[start - 1:]
        if gen_lp.numel() == 0:
            return float("nan")
        return float(math.exp(-gen_lp.mean().item()))

    def free(self):
        del self.model
        _cleanup()


# ----------------------------------------------------------------------
# DIPPER paraphraser (kalpeshk2011/dipper-paraphraser-xxl)
# ----------------------------------------------------------------------
def _split_sentences(text):
    text = " ".join(text.split())
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class DipperParaphraser:
    def __init__(self, model_name, tok_name, device):
        log.info("[dipper] loading paraphraser: %s%s", model_name,
                 " (8-bit)" if QUANTIZE_8BIT else "")
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        self.tok = T5Tokenizer.from_pretrained(tok_name)
        kw, quant = _load_kwargs(device)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, **kw)
        if not quant:
            self.model.to(device)
        self.model.eval()
        self.device = "cuda" if quant else device

    @torch.inference_mode()
    def paraphrase(self, text, lex=DIPPER_LEX, order=DIPPER_ORDER, sent_interval=3):
        lex_code, order_code = int(100 - lex), int(100 - order)
        sentences = _split_sentences(text)
        prefix, out = "", ""
        for i in range(0, max(1, len(sentences)), sent_interval):
            window = " ".join(sentences[i:i + sent_interval]) or text
            inp = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                inp += f" {prefix}"
            inp += f" <sent> {window} </sent>"
            enc = self.tok([inp], return_tensors="pt", truncation=True,
                           max_length=512).to(self.device)
            gen = self.model.generate(**enc, do_sample=True, top_p=0.75,
                                      max_length=512)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True)[0]
            prefix += " " + dec
            out += " " + dec
        return out.strip()

    def free(self):
        del self.model
        _cleanup()


# ----------------------------------------------------------------------
# MAUVE (distributional similarity to human text)
# ----------------------------------------------------------------------
def compute_mauve_safe(p_text, q_text, device):
    p_text = [t for t in p_text if isinstance(t, str) and t.strip()]
    q_text = [t for t in q_text if isinstance(t, str) and t.strip()]
    if len(p_text) < MAUVE_MIN_SAMPLES or len(q_text) < MAUVE_MIN_SAMPLES:
        log.info("[mauve] too few samples (%d/%d); skipping", len(p_text), len(q_text))
        return None
    try:
        import mauve
    except Exception as e:
        log.warning("[mauve] package unavailable (%s); skipping", e)
        return None
    try:
        out = mauve.compute_mauve(p_text=p_text, q_text=q_text,
                                  featurize_model_name=MAUVE_FEATURIZE,
                                  device_id=_device_id(device), max_text_length=256,
                                  verbose=False)
        _cleanup()
        return float(out.mauve)
    except Exception as e:
        log.warning("[mauve] computation failed: %s", e)
        return None



def device_of(model):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _fmt_dt(s):
    s = int(s)
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def _progress(tag, done, total, t0, extra=""):
    """Detailed progress line with elapsed time and a rough ETA."""
    el = time.time() - t0
    rate = el / max(done, 1)
    eta = rate * (total - done)
    log.info("  [%s] %d/%d (%.0f%%)  elapsed %s  eta %s%s",
             tag, done, total, 100.0 * done / max(total, 1),
             _fmt_dt(el), _fmt_dt(eta), ("  " + extra) if extra else "")


def _banner(stage, model):
    log.info("=" * 70)
    log.info("STAGE %s   model=%s", stage, model)
    log.info("  knobs: MAIN_MAX_PROMPTS=%s  SUBSET_FRAC=%.3f  LOG_EVERY=%d  QUICK=%s",
             MAIN_MAX_PROMPTS or "ALL", SUBSET_FRAC, LOG_EVERY, QUICK)
    log.info("  models: gen=%s  oracle=%s  paraphraser=%s  mauve=%s  8bit=%s",
             model, ORACLE_MODEL, PARAPHRASE_MODEL, MAUVE_FEATURIZE, QUANTIZE_8BIT)
    log.info("  payloads=%s  tokens=%d  delta=%s  device=%s",
             WATERMARK_SIZES, MAX_NEW_TOKENS, DELTA, device_of(model))
    log.info("=" * 70)


def _main_indices():
    """Prompt indices for the generation stage, always in dataset order.

    Default (MAIN_MAX_PROMPTS=0) is ALL prompts. A positive MAIN_MAX_PROMPTS
    takes the FIRST N in order (not a random subset), so raising the cap on a
    later run keeps every already-generated prompt and resume just continues.
    """
    total = dataset_size()
    if QUICK:
        return list(range(min(3, total)))
    if 0 < MAIN_MAX_PROMPTS < total:
        return list(range(MAIN_MAX_PROMPTS))
    return list(range(total))


def _path(model, name):
    return out_path(model, name) + ".json"


def _load_main(model):
    p = _path(model, "opengen_main")
    if not os.path.exists(p):
        log.error("missing %s; run --experiment generate first", p); sys.exit(1)
    return json.load(open(p))


# ======================================================================
# STAGE 1: generate  (prompts + watermark)            [generation model]
# ======================================================================
def _main_aggregate(samples):
    agg = []
    for k in WATERMARK_SIZES:
        rows = [s for s in samples if s["k"] == k]
        if not rows:
            continue
        agg.append({"k": k, "match_rate": float(np.mean([r["match"] for r in rows]) * 100),
                    "bit_accuracy": float(np.mean([r["bit_accuracy"] for r in rows]) * 100),
                    "median_p_value": float(np.median([r["p_value"] for r in rows])),
                    "n": len(rows)})
    return agg


def _save_main(model, samples, t0):
    payload = {"config": {"model": model, "delta": DELTA, "gamma": GAMMA,
                          "temperature": TEMPERATURE, "top_p": TOP_P, "window": WINDOW,
                          "tokens": MAX_NEW_TOKENS,
                          "n_prompts": len(set(s["prompt_index"] for s in samples)),
                          "watermark_sizes": WATERMARK_SIZES, "dataset": "OpenGen"},
               "aggregate": _main_aggregate(samples), "samples": samples,
               "elapsed_sec": time.time() - t0}
    _atomic_write(_path(model, "opengen_main"), payload)


def run_generate(model):
    _banner("1: generate", model)
    idx = _main_indices()
    path = _path(model, "opengen_main")

    # resume: reuse any completed prompts from a previous (interrupted) run
    samples, done = [], set()
    prev = _read_json(path) if os.path.exists(path) else None
    if prev and prev.get("samples"):
        cnt = {}
        for s in prev["samples"]:
            cnt[s["prompt_index"]] = cnt.get(s["prompt_index"], 0) + 1
        done = {pi for pi, c in cnt.items() if c >= len(WATERMARK_SIZES)}
        samples = [s for s in prev["samples"] if s["prompt_index"] in done]
        log.info("[generate] RESUME from %s: %d prompts already complete", path, len(done))

    wm = make_wm(model, WATERMARK_SIZES[0])
    log.info("[generate] loading prompts and human references...")
    prompts, refs = prompts_refs(wm, idx)
    todo = [i for i in range(len(prompts)) if i not in done]
    log.info("[generate] %d prompts total, %d done, %d to do; checkpoint every %d",
             len(prompts), len(done), len(todo), CHECKPOINT_EVERY)
    run_match = {k: [s["match"] for s in samples if s["k"] == k] for k in WATERMARK_SIZES}
    t0 = time.time()
    for n, i in enumerate(todo, 1):
        prompt = prompts[i]
        plain = wm.generate(prompt, [0] * WATERMARK_SIZES[0],
                            max_new_tokens=MAX_NEW_TOKENS, watermark=False)
        for k in WATERMARK_SIZES:
            wm.reconfigure(k_bits=k)
            msg = message_for(i, k)
            gen = wm.generate(prompt, msg, max_new_tokens=MAX_NEW_TOKENS)
            det = wm.detect_ids(gen["generated_ids"], prompt_ids=gen["prompt_ids"])
            match = bool(det["message"] == msg); run_match[k].append(match)
            samples.append({
                "prompt_index": i, "prompt": prompt, "reference": refs[i], "k": k,
                "message": msg, "decoded": det["message"], "match": match,
                "bit_accuracy": float(np.mean(np.array(det["message"]) == np.array(msg))),
                "p_value": det["p_value"], "agreement": det["match_fraction"],
                "score": det["score"], "num_tokens": det["num_tokens"],
                "generated_ids": gen["generated_ids"],
                "watermarked_text": gen["generated_text"],
                "plain_text": plain["generated_text"],
            })
        if n % CHECKPOINT_EVERY == 0:
            _save_main(model, samples, t0)
            log.info("[generate] CHECKPOINT saved (%d/%d prompts complete) -> %s",
                     len(done) + n, len(prompts), path)
        if n % LOG_EVERY == 0 or n == len(todo):
            mr = ", ".join(f"k{k}={np.mean(run_match[k])*100:.0f}%" for k in WATERMARK_SIZES)
            _progress("generate", len(done) + n, len(prompts), t0, f"running match [{mr}]")
    del wm; _cleanup()
    _save_main(model, samples, t0)
    for a in _main_aggregate(samples):
        log.info("[generate] k=%d  match=%.1f%%  bitacc=%.2f%%  n=%d",
                 a["k"], a["match_rate"], a["bit_accuracy"], a["n"])
    log.info("[generate] DONE: %d prompts saved to %s", len(set(s["prompt_index"] for s in samples)), path)


# ======================================================================
# STAGE 2: quality       (perplexity + MAUVE only)
#   perplexity: oracle model (+ MAUVE featurizer)     [Llama-3.1 + gpt2-large]
#   NOTE: paraphrasing is a robustness attack (stage 3), not a quality metric.
# ======================================================================
def run_quality(model, sub="all"):
    _quality_perplexity(model)


def _quality_perplexity(model):
    _banner("2: quality-perplexity", model)
    data = _load_main(model); samples = data["samples"]
    todo = [s for s in samples if "ppl_watermarked" not in s]   # resume
    log.info("[quality:perplexity] %d samples, %d already scored, %d to score "
             "(oracle %s; checkpoint every %d)", len(samples),
             len(samples) - len(todo), len(todo), ORACLE_MODEL, CHECKPOINT_EVERY * 10)
    if todo:
        oracle = OracleScorer(ORACLE_MODEL, device_of(model))
        t0 = time.time()
        for j, s in enumerate(todo, 1):
            s["ppl_watermarked"] = oracle.perplexity(s["prompt"], s["watermarked_text"])
            if s["k"] == WATERMARK_SIZES[0]:
                s["_ppl_plain"] = oracle.perplexity(s["prompt"], s["plain_text"])
            if j % (CHECKPOINT_EVERY * 10) == 0:
                _atomic_write(_path(model, "opengen_main"), data)
                log.info("[quality:perplexity] CHECKPOINT (%d/%d scored)", j, len(todo))
            if j % (LOG_EVERY * 10) == 0 or j == len(todo):
                _progress("perplexity", j, len(todo), t0)
        log.info("[quality:perplexity] perplexity pass done in %s; freeing oracle",
                 _fmt_dt(time.time() - t0))
        oracle.free()
    plain_ppl = {s["prompt_index"]: s["_ppl_plain"] for s in samples if "_ppl_plain" in s}
    for s in samples:
        s["ppl_plain"] = plain_ppl.get(s["prompt_index"], s.get("ppl_plain", float("nan")))
        s.pop("_ppl_plain", None)

    refs0 = [s["reference"] for s in samples if s["k"] == WATERMARK_SIZES[0]]
    plain0 = [s["plain_text"] for s in samples if s["k"] == WATERMARK_SIZES[0]]
    log.info("[quality:mauve] computing MAUVE (featurizer=%s) ...", MAUVE_FEATURIZE)
    mauve_plain = compute_mauve_safe(plain0, refs0, device_of(model))
    log.info("[quality:mauve] plain vs human references: %s", mauve_plain)
    for a in data["aggregate"]:
        rows = [s for s in samples if s["k"] == a["k"]]
        ppw = float(np.nanmean([r["ppl_watermarked"] for r in rows]))
        ppp = float(np.nanmean([r["ppl_plain"] for r in rows]))
        mauve_wm = compute_mauve_safe([r["watermarked_text"] for r in rows],
                                      [r["reference"] for r in rows], device_of(model))
        a.update({"mean_ppl_watermarked": ppw, "mean_ppl_plain": ppp,
                  "ppl_increase_pct": float((ppw / ppp - 1) * 100),
                  "mauve_watermarked": mauve_wm, "mauve_plain": mauve_plain})
        log.info("[quality] k=%d  ppl_wm=%.1f  ppl_base=%.1f  MAUVE_wm=%s  MAUVE_base=%s",
                 a["k"], ppw, ppp, mauve_wm, mauve_plain)
    data["config"]["oracle"] = ORACLE_MODEL
    data["config"]["mauve_featurize"] = MAUVE_FEATURIZE
    _atomic_write(_path(model, "opengen_main"), data)
    log.info("[quality:perplexity] updated %s", _path(model, "opengen_main"))


def _ensure_paraphrases(model):
    """Generate DIPPER paraphrases of the watermarked subset for the paraphrase
    robustness attack (loads/frees DIPPER). Resumable and checkpointed; a no-op
    if every (prompt_index, k) in the subset already has a paraphrase on disk."""
    data = _load_main(model); samples = data["samples"]
    prompt_ids_all = sorted(set(s["prompt_index"] for s in samples))
    keep = set(np.array(prompt_ids_all)[
        subset_indices(len(prompt_ids_all), SUBSET_FRAC)].tolist())
    sub = [s for s in samples if s["prompt_index"] in keep]
    ppath = _path(model, "opengen_paraphrase")

    # resume: keep paraphrases already produced
    out, done = [], set()
    prev = _read_json(ppath) if os.path.exists(ppath) else None
    if prev and prev.get("items"):
        out = prev["items"]
        done = {(it["prompt_index"], it["k"]) for it in out}
        log.info("[robust:paraphrase] RESUME from %s: %d paraphrases already done",
                 ppath, len(done))
    todo = [s for s in sub if (s["prompt_index"], s["k"]) not in done]
    log.info("[robust:paraphrase] generating paraphrases: %.0f%% subset = %d prompts, "
             "%d texts total, %d to do with %s (checkpoint every %d)", SUBSET_FRAC * 100,
             len(keep), len(sub), len(todo), PARAPHRASE_MODEL, CHECKPOINT_EVERY)

    def _save():
        _atomic_write(ppath, {"config": data["config"], "subset_frac": SUBSET_FRAC,
                              "paraphraser": PARAPHRASE_MODEL, "items": out})

    if todo:
        wm = make_wm(model, WATERMARK_SIZES[0], load_model=False)   # tokenizer only
        dipper = DipperParaphraser(PARAPHRASE_MODEL, PARAPHRASE_TOK, device_of(model))
        t0 = time.time()
        for j, s in enumerate(todo, 1):
            text = dipper.paraphrase(s["watermarked_text"])
            out.append({"prompt_index": s["prompt_index"], "k": s["k"],
                        "para_ids": wm.tokenizer(text, add_special_tokens=False)["input_ids"]})
            if j % CHECKPOINT_EVERY == 0:
                _save(); log.info("[robust:paraphrase] CHECKPOINT (%d/%d) -> %s",
                                  len(done) + j, len(sub), ppath)
            if j % ROB_LOG_EVERY == 0 or j == len(todo):
                _progress("paraphrase-gen", len(done) + j, len(sub), t0)
        log.info("[robust:paraphrase] paraphrase generation done in %s; freeing DIPPER",
                 _fmt_dt(time.time() - t0))
        dipper.free(); _cleanup()
    _save()
    log.info("[robust:paraphrase] %d paraphrases ready in %s", len(out), ppath)


# ======================================================================
# STAGE 3: robustness    attack = substitution|insertion|deletion|
#                                 reordering|paraphrase|all       [no model]
#   model-free: reuses stage-1 ids (+ stage-2 paraphrases). Results merge into
#   one robustness JSON, so sub-experiments can run in any order.
# ======================================================================
def run_robust(model, attack="all"):
    _banner("3: robust-" + attack, model)
    data = _load_main(model); samples = data["samples"]
    prompt_ids_all = sorted(set(s["prompt_index"] for s in samples))
    keep = set(np.array(prompt_ids_all)[
        subset_indices(len(prompt_ids_all), SUBSET_FRAC)].tolist())
    samples = [s for s in samples if s["prompt_index"] in keep]
    by_k = {}
    for s in samples:
        by_k.setdefault(s["k"], []).append(s)
    sizes = sorted(by_k)
    log.info("[robust:%s] model-free detection; tokenizing prompts...", attack)
    wm = make_wm(model, sizes[0], load_model=False)             # tokenizer only
    vocab = wm.vocab_size
    rng = np.random.default_rng(SEED)
    for k in sizes:
        for s in by_k[k]:
            s["_pids"] = wm.tokenizer(s["prompt"])["input_ids"]

    # which attack settings to run
    want = ["substitution", "insertion", "deletion", "reordering", "paraphrase"] \
        if attack == "all" else [attack]
    jobs = []
    if "clean" in want or attack == "all":
        jobs.append(("clean", 0.0, lambda ids, s: ids))
    for a in want:
        if a in ("substitution", "insertion", "deletion"):
            for eps in ATTACK_FRACTIONS:
                fn = {"substitution": lambda ids, s, e=eps: attack_substitute(ids, e, vocab, rng),
                      "insertion":    lambda ids, s, e=eps: attack_insert(ids, e, vocab, rng),
                      "deletion":     lambda ids, s, e=eps: attack_delete(ids, e, rng)}[a]
                jobs.append((a, eps, fn))
        elif a == "reordering":
            for b in REORDER_BLOCKS:
                jobs.append(("reordering", float(b),
                             lambda ids, s, bb=b: attack_reorder(ids, bb, rng)))
        elif a == "paraphrase":
            _ensure_paraphrases(model)          # generate with DIPPER if needed, then free it
            para = _load_paraphrases(model)
            jobs.append(("paraphrase", 1.0,
                         lambda ids, s: para.get((s["prompt_index"], s["k"]), ids)))

    # resume: load existing robustness rows, skip (attack,strength,k) already done
    rob_path = _path(model, "opengen_robustness")
    prev = _read_json(rob_path) if os.path.exists(rob_path) else None
    results = {(r["attack"], r["strength"], r["k"]): r
               for r in (prev.get("results", []) if prev else [])}

    def _save():
        _atomic_write(rob_path, {"config": data["config"], "subset_frac": SUBSET_FRAC,
                                 "results": list(results.values())})

    njobs = len(jobs) * len(sizes)
    log.info("[robust:%s] subset=%.0f%% (%d prompts) -> %d samples, sizes %s, "
             "%d jobs (%d already in %s)", attack, SUBSET_FRAC * 100, len(keep),
             len(samples), sizes, njobs,
             sum(1 for n, st, _ in jobs for k in sizes if (n, st, k) in results),
             rob_path)
    t0, job = time.time(), 0
    for name, strength, fn in jobs:
        for k in sizes:
            job += 1
            if (name, strength, k) in results:
                log.info("  [%2d/%2d] %-13s strength=%-5.2f k=%-2d  SKIP (already done)",
                         job, njobs, name, strength, k)
                continue
            wm.reconfigure(k_bits=k)
            rows = by_k[k]; matches, bitaccs, dets = [], [], []; jt0 = time.time()
            log.info("  -> [%2d/%2d] attack=%s strength=%.2f k=%d: detecting %d texts",
                     job, njobs, name, strength, k, len(rows))
            for j, s in enumerate(rows):
                det = wm.detect_ids(fn(s["generated_ids"], s), prompt_ids=s["_pids"])
                matches.append(det["message"] == s["message"])
                bitaccs.append(float(np.mean(np.array(det["message"]) == np.array(s["message"]))))
                dets.append(bool(det["detected"]))
                if (j + 1) % ROB_LOG_EVERY == 0:
                    log.info("       [%s s=%.2f k=%d] %d/%d, running match=%.1f%%",
                             name, strength, k, j + 1, len(rows), np.mean(matches) * 100)
            results[(name, strength, k)] = {
                "attack": name, "strength": strength, "k": k,
                "match_rate": float(np.mean(matches) * 100),
                "bit_accuracy": float(np.mean(bitaccs) * 100),
                "detection_rate": float(np.mean(dets) * 100), "n": len(rows)}
            _save()   # checkpoint after every (attack, strength, k)
            r = results[(name, strength, k)]
            log.info("  [%2d/%2d] %-13s strength=%-5.2f k=%-2d  match=%5.1f%%  "
                     "bitacc=%5.1f%%  detect=%5.1f%%  (n=%d, %s)  [saved]", job, njobs,
                     name, strength, k, r["match_rate"], r["bit_accuracy"],
                     r["detection_rate"], r["n"], _fmt_dt(time.time() - jt0))
    log.info("[robust:%s] done; %d total rows in %s (%s)", attack, len(results),
             rob_path, _fmt_dt(time.time() - t0))


def _load_paraphrases(model):
    p = _path(model, "opengen_paraphrase")
    if not os.path.exists(p):
        log.error("paraphrase attack needs %s (should have been generated by "
                  "_ensure_paraphrases)", p)
        sys.exit(1)
    items = json.load(open(p))["items"]
    return {(it["prompt_index"], it["k"]): it["para_ids"] for it in items}


# ======================================================================
# STAGE 4: sweeps        sub = delta | tokens | all
#   delta : match rate + perplexity vs delta (step 1)   [gen model + oracle]
#   tokens: bits x tokens match-rate heatmap            [gen model]
# ======================================================================
def _save_sweep(model, data, pdir, idx):
    _atomic_write(os.path.join(pdir, "sweep_data.json"),
                  {"config": {"model": model, "k": PLOT_K, "n_prompts": len(idx),
                              "subset_frac": SUBSET_FRAC}, "data": data})



def run_sweep(model, sub="all"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = dataset_size()
    if QUICK:
        n = min(n, 3)

    idx = subset_indices(n, SUBSET_FRAC)
    pdir = out_path(model, "plots")
    os.makedirs(pdir, exist_ok=True)
    data = _load_sweep_data(model)

    _banner("4: sweep-" + sub, model)
    log.info(
        "[sweep:%s] subset=%.0f%% of %d -> %d prompts",
        sub,
        SUBSET_FRAC * 100,
        n,
        len(idx),
    )

    ###########################################################################
    # DELTA SWEEP
    ###########################################################################

    if sub in ("delta", "all"):

        existing = {float(e["delta"]): e for e in data.get("delta", [])}

        done = {
            d for d, e in existing.items()
            if all(
                str(k) in e.get("results", {})
                and e["results"][str(k)].get("mean_ppl") is not None
                for k in HEATMAP_KS
            )
        }

        todo = [d for d in DELTA_SWEEP_FULL if float(d) not in done]

        log.info(
            "[sweep:delta] deltas=%s ks=%s; %d done, %d to do",
            DELTA_SWEEP_FULL,
            HEATMAP_KS,
            len(DELTA_SWEEP_FULL) - len(todo),
            len(todo),
        )

        if todo:

            wm = make_wm(model, max(HEATMAP_KS))
            prompts, _ = prompts_refs(wm, idx)
            t0 = time.time()

            for di, d in enumerate(todo):

                log.info(
                    "[delta=%s] (%d/%d)",
                    d,
                    di + 1,
                    len(todo),
                )

                delta_results = {}

                for ki, k in enumerate([16, 32]):

                    wm.reconfigure(
                        k_bits=k,
                        delta=float(d),
                        window=WINDOW,
                    )

                    matches, bitaccs, rows = [], [], []

                    log.info(
                        "   k=%d (%d/%d)",
                        k,
                        ki + 1,
                        2,
                    )

                    for jj, (i, p) in enumerate(zip(idx, prompts)):

                        msg = message_for(i, k)

                        gen = wm.generate(
                            p,
                            msg,
                            max_new_tokens=MAX_NEW_TOKENS,
                        )

                        det = wm.detect_ids(
                            gen["generated_ids"],
                            prompt_ids=gen["prompt_ids"],
                        )

                        match = bool(det["message"] == msg)
                        bitacc = float(
                            np.mean(
                                np.array(det["message"]) ==
                                np.array(msg)
                            )
                        )

                        matches.append(match)
                        bitaccs.append(bitacc)

                        rows.append({
                            "idx": int(i),
                            "prompt": p,
                            "true_message": msg,
                            "detected_message": det["message"],
                            "match": match,
                            "bit_accuracy": bitacc,
                            "p_value": det.get("p_value"),
                            "agreement": det.get("match_fraction"),
                            "score": det.get("score"),
                            "num_tokens": det.get("num_tokens"),
                            "generated_ids": gen["generated_ids"],
                            "watermarked_text": gen["generated_text"],
                            "detector_output": det,
                        })

                        if (jj + 1) % LOG_EVERY == 0:
                            log.info(
                                "      delta=%2d k=%2d "
                                "%4d/%4d "
                                "match=%5.1f%% "
                                "bitacc=%5.1f%%",
                                d,
                                k,
                                jj + 1,
                                len(idx),
                                np.mean(matches) * 100,
                                np.mean(bitaccs) * 100,
                            )

                    delta_results[str(k)] = {
                        "match_rate":
                            float(np.mean(matches) * 100),
                        "mean_bit_accuracy":
                            float(np.mean(bitaccs) * 100),
                        "mean_ppl":
                            None,
                        "results":
                            rows,
                    }

                    log.info(
                        "   DONE delta=%2d k=%2d "
                        "match=%5.1f%% "
                        "bitacc=%5.1f%%",
                        d,
                        k,
                        delta_results[str(k)]["match_rate"],
                        delta_results[str(k)]["mean_bit_accuracy"],
                    )

                existing[float(d)] = {
                    "delta": float(d),
                    "results": delta_results,
                }

                data["delta"] = [
                    existing[float(x)]
                    for x in DELTA_SWEEP_FULL
                    if float(x) in existing
                ]

                _save_sweep(model, data, pdir, idx)

                _progress(
                    "sweep:delta-gen",
                    di + 1,
                    len(todo),
                    t0,
                    f"delta={d} [saved]",
                )

            log.info(
                "[sweep:delta] generation done; "
                "freeing model, loading oracle"
            )

            del wm
            _cleanup()

            oracle = OracleScorer(
                ORACLE_MODEL,
                device_of(model),
            )

            for d in todo:
                for k in HEATMAP_KS:

                    entry = existing[float(d)]["results"][str(k)]
                    ppls = []

                    for r in entry["results"]:

                        ppl = oracle.perplexity(
                            r["prompt"],
                            r["watermarked_text"],
                        )

                        r["perplexity"] = float(ppl)
                        ppls.append(ppl)

                    entry["mean_ppl"] = float(
                        np.nanmean(ppls)
                    )

                    data["delta"] = [
                        existing[float(x)]
                        for x in DELTA_SWEEP_FULL
                        if float(x) in existing
                    ]

                    _save_sweep(
                        model,
                        data,
                        pdir,
                        idx,
                    )

                    log.info(
                        "   delta=%2d k=%2d "
                        "match=%5.1f%% "
                        "bitacc=%5.1f%% "
                        "ppl=%6.2f",
                        d,
                        k,
                        entry["match_rate"],
                        entry["mean_bit_accuracy"],
                        entry["mean_ppl"],
                    )

            oracle.free()

        data["delta"] = [
            existing[float(x)]
            for x in DELTA_SWEEP_FULL
            if float(x) in existing
        ]

    ###########################################################################
    # TOKEN SWEEP
    ###########################################################################

    if sub in ("tokens", "all"):

        hmap = data.get("heatmap", {})
        todo = [
            k for k in HEATMAP_KS
            if str(k) not in hmap
        ]

        log.info(
            "[sweep:tokens] heatmap ks=%s tokens=%s; "
            "%d done, %d to do",
            HEATMAP_KS,
            HEATMAP_TOKENS,
            len(HEATMAP_KS) - len(todo),
            len(todo),
        )

        if todo:

            wm = make_wm(model, max(HEATMAP_KS))
            prompts, _ = prompts_refs(wm, idx)
            Tmax = max(HEATMAP_TOKENS)
            t0 = time.time()

            for kk, k in enumerate(todo):

                wm.reconfigure(k_bits=k)
                kres = {}

                for jj, (i, p) in enumerate(zip(idx, prompts)):
                    matches, bitaccs, rows = [], [], []
                    msg = message_for(i, k)
                    gen = wm.generate(
                            p,
                            msg,
                            max_new_tokens=Tmax,
                    )
                    for T in HEATMAP_TOKENS:

                        det = wm.detect_ids(
                            gen["generated_ids"][:T],
                            prompt_ids=gen["prompt_ids"],
                        )
                        match = bool(det["message"] == msg)

                        bitacc = float(
                            np.mean(
                                np.array(det["message"]) ==
                                np.array(msg)
                            )
                        )

                        matches.append(match)
                        bitaccs.append(bitacc)

                        rows.append({
                            "T": T,
                            "prompt": p,
                            "true_message": msg,
                            "detected_message": det["message"],
                            "match": match,
                            "bit_accuracy": bitacc,
                            "p_value": det.get("p_value"),
                            "agreement": det.get("match_fraction"),
                            "score": det.get("score"),
                            "num_tokens": det.get("num_tokens"),
                            "generated_ids": gen["generated_ids"][:T],
                            "watermarked_text": gen["generated_text"],
                            "detector_output": det,
                        })


                    kres[str(i)] = {
                        "results": rows
                    }

                    if (jj + 1) % LOG_EVERY == 0:
                            log.info(jj + 1)

                hmap[str(k)] = kres
                data["heatmap"] = hmap

                _save_sweep(
                    model,
                    data,
                    pdir,
                    idx,
                )

                _progress(
                    "sweep:tokens",
                    kk + 1,
                    len(todo),
                    t0,
                    f"k={k} [saved]",
                )

            del wm
            _cleanup()

        heat = np.array([
            [
                hmap.get(str(k), {})
                    .get(str(T), {})
                    .get("match_rate", np.nan)
                for T in HEATMAP_TOKENS
            ]
            for k in HEATMAP_KS
        ])

        _plot_heatmap(
            heat,
            pdir,
            plt,
        )

    _save_sweep(
        model,
        data,
        pdir,
        idx,
    )

    log.info(
        "[sweep:%s] DONE; saved figures and %s",
        sub,
        os.path.join(
            pdir,
            "sweep_data.json",
        ),
    )




def _load_sweep_data(model):
    p = os.path.join(out_path(model, "plots"), "sweep_data.json")
    if os.path.exists(p):
        return json.load(open(p)).get("data", {"delta": [], "tokens": [], "heatmap": {}})
    return {"delta": [], "tokens": [], "heatmap": {}}


def _plot_delta(dd, pdir, plt):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot([x["delta"] for x in dd], [x["match_rate"] for x in dd], "o-", color="C0")
    ax.set_xlabel("logit bias delta"); ax.set_ylabel("match rate (%)"); ax.set_ylim(-5, 105)
    ax2 = ax.twinx()
    ax2.plot([x["delta"] for x in dd], [x.get("mean_ppl") for x in dd], "s--", color="C3", alpha=.7)
    ax2.set_ylabel("perplexity", color="C3")
    ax.set_title("Match rate and perplexity vs delta"); fig.tight_layout()
    fig.savefig(os.path.join(pdir, "delta_sweep_full.png"), dpi=130); plt.close(fig)


def _plot_heatmap(heat, pdir, plt):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=100, origin="lower")
    ax.set_xticks(range(len(HEATMAP_TOKENS))); ax.set_xticklabels(HEATMAP_TOKENS)
    ax.set_yticks(range(len(HEATMAP_KS))); ax.set_yticklabels(HEATMAP_KS)
    ax.set_xlabel("number of tokens generated"); ax.set_ylabel("payload size k (bits)")
    ax.set_title("Exact match rate (%) vs bits and tokens")
    for ki in range(len(HEATMAP_KS)):
        for ti in range(len(HEATMAP_TOKENS)):
            ax.text(ti, ki, f"{heat[ki, ti]:.0f}", ha="center", va="center",
                    color="white" if heat[ki, ti] < 60 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="match rate (%)"); fig.tight_layout()
    fig.savefig(os.path.join(pdir, "bits_tokens_heatmap.png"), dpi=130); plt.close(fig)


# ======================================================================
EXP = ["generate",
       "quality", "quality-perplexity",
       "robust", "robust-substitution", "robust-insertion", "robust-deletion",
       "robust-reordering", "robust-paraphrase",
       "sweep", "sweep-delta", "sweep-tokens",
       "all"]


def main():
    ap = argparse.ArgumentParser(description="MOSAIC staged experiments (FULL OpenGen)")
    ap.add_argument("--experiment", required=True, choices=EXP)
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    args = ap.parse_args()

    setup_logging(args.experiment, args.model)
    if not os.path.exists(DATA_PATH):
        log.error("OpenGen parquet not found at %s", DATA_PATH); sys.exit(1)
    e = args.experiment; m = args.model
    log.info("stage=%s  model=%s  MAIN_MAX_PROMPTS=%s  SUBSET_FRAC=%.2f  oracle=%s  dipper=%s",
             e, m, MAIN_MAX_PROMPTS or "ALL", SUBSET_FRAC, ORACLE_MODEL, PARAPHRASE_MODEL)

    t0 = time.time()
    try:
        if e in ("generate", "all"):
            run_generate(m); _cleanup()
        if e in ("quality", "quality-perplexity", "all"):
            run_quality(m); _cleanup()
        if e == "robust" or e == "all":
            run_robust(m, "all"); _cleanup()
        elif e.startswith("robust-"):
            run_robust(m, e.split("robust-")[1]); _cleanup()
        if e == "sweep" or e == "all":
            run_sweep(m, "all"); _cleanup()
        elif e.startswith("sweep-"):
            run_sweep(m, e.split("sweep-")[1]); _cleanup()
    except Exception:
        log.exception("experiment failed"); raise
    log.info("experiment '%s' done in %.0f s", e, time.time() - t0)


def _atomic_write(path, obj):
    """Write JSON to a temp file then rename, so an interrupt never corrupts it."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def merge_delta_sweeps(
    file1="sweep_data_delta_2_3.json",
    file2="sweep_data_delta_4_5_6_7_8_9.json",
    out_file="sweep_data_delta.json",
):
    """
    Merge two delta sweep files into one.

    Keeps the same schema as sweep_data.json.
    """

    d1 = json.load(open(file1))
    d2 = json.load(open(file2))

    merged = {}

    for x in d1["data"]["delta"]:
        merged[float(x["delta"])] = x

    for x in d2["data"]["delta"]:
        merged[float(x["delta"])] = x

    out = {
        "config": d1.get("config", d2.get("config", {})),
        "data": {
            "delta": [
                merged[d] for d in sorted(merged)
            ]
        },
    }

    _atomic_write(out_file, out)

    print(f"Saved merged sweep to {out_file}")
    
# merge_delta_sweeps()

def add_delta_perplexity(
    sweep_file="sweep_data_delta.json",
):
    """
    Adds oracle perplexity to every prompt output in the delta sweep.

    Adds:
        result["perplexity"]

    and

        delta_entry["results"][k]["mean_ppl"]

    Can safely resume if interrupted.
    """

    data = json.load(open(sweep_file))

    oracle = OracleScorer(
        ORACLE_MODEL,
        device_of(None),
    )

    try:

        for delta_entry in data["data"]["delta"]:

            delta = delta_entry["delta"]

            print(f"\nDelta = {delta}")

            for k, info in delta_entry["results"].items():

                rows = info["results"]

                # Resume support
                if (
                    info.get("mean_ppl") is not None
                    and all("perplexity" in r for r in rows)
                ):
                    print(f"  k={k}: already complete")
                    continue

                ppls = []

                for i, r in enumerate(rows):

                    if "perplexity" not in r:

                        ppl = oracle.perplexity(
                            r["prompt"],
                            r["watermarked_text"],
                        )

                        r["perplexity"] = float(ppl)

                    ppls.append(r["perplexity"])

                    if (i + 1) % 20 == 0:
                        print(
                            f"      {i+1}/{len(rows)}"
                        )

                info["mean_ppl"] = float(np.nanmean(ppls))

                print(
                    f"  k={k} mean perplexity = {info['mean_ppl']:.2f}"
                )

                # checkpoint after every k
                _atomic_write(
                    sweep_file,
                    data,
                )

    finally:
        oracle.free()
        _cleanup()

    print("\nDone.")
