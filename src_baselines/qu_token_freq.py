"""Token frequency file for Qu et al.'s balanced map on a model outside their release.

Qu et al. distribute token frequencies for Llama, Falcon, and Guanaco only, and do
not document how they computed them. For another model this script counts the
tokens of the OpenGen texts (prefix and first target) of the prompts that the
evaluation never uses, index EVAL_PROMPTS onward, with that model's tokenizer,
and writes data/qu/token_freq_<model tag>.pkl in the format of the released
files (one count per token id; qu_balance.py raises zero counts to 1).

  python -m src_baselines.qu_token_freq --model mistralai/Mistral-7B-v0.1

--check validates the procedure on Llama-2, where the released file exists: it
builds the balanced maps from the OpenGen counts and from token_freq_llama.pkl
and reports each segment's share of the previous-token load in the stored
Llama-2 Qu et al. generations (prompts 0-999, which the counts exclude). On
those texts the two maps balance equally well (shares 0.159-0.177 against
0.156-0.177 for 6 segments, 0.090-0.105 against 0.091-0.113 for 10). Only the
tokenizer is loaded (CPU, well under 1 GB), never the model.
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src_baselines.qu_balance import DATA_DIR, DEFAULT_MODEL, FREQ_FILE, freq_file_for, load_freq  # noqa: E402

EVAL_PROMPTS = 1000       # the baselines are evaluated on prompts 0..999


def count_tokens(model_name: str, start: int = EVAL_PROMPTS) -> np.ndarray:
    import run_baseline_experiments as rb           # same prompt pool as the experiments
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    counts = np.zeros(len(tok), dtype=np.int64)
    pool = rb.load_pool()
    for prefix, ref in pool[start:]:
        for text in (prefix, ref):
            if text:
                ids = tok(text, add_special_tokens=False)["input_ids"]
                counts += np.bincount(ids, minlength=len(tok))[:len(tok)]
    print(f"{model_name}: {len(pool) - start} texts, {int(counts.sum()):,} tokens, "
          f"{int((counts > 0).sum()):,} of {len(tok):,} ids seen", flush=True)
    return counts


def check_llama():
    """Segment shares of the generated previous-token load under the map from
    OpenGen counts and under the map from the released frequencies."""
    import run_baseline_experiments as rb
    from src_baselines.qu_balance import build_mapping
    from src_baselines.qu_segment import _key_int
    sys.path.insert(0, os.path.join(ROOT, "results"))
    from stream_samples import iter_samples
    released = load_freq(FREQ_FILE)
    ours = count_tokens(DEFAULT_MODEL)
    V = len(ours)
    load = np.zeros(V)
    src = os.path.join(ROOT, "results", "other_models",
                       "opengen_baseline_qu_" + DEFAULT_MODEL.replace("/", "_") + ".json")
    for s in iter_samples(src, fields=("generated_ids",)):
        prev = np.asarray(s["generated_ids"][:-1])
        load += np.bincount(prev[prev < V], minlength=V)
    key = _key_int(rb.SECRET_KEY, "qu.map")
    for p in (6, 10):
        for name, freq in (("released frequencies", released), ("OpenGen counts", ours)):
            share = np.bincount(build_mapping(freq, p, key)[:V], weights=load, minlength=p)
            share = share / load.sum()
            print(f"  {p} segments, map from {name:20s}: shares {share.min():.3f}-"
                  f"{share.max():.3f} (ideal {1 / p:.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=None, help="Hugging Face model id to count for")
    ap.add_argument("--check", action="store_true",
                    help="validate this procedure against the released Llama file")
    args = ap.parse_args()
    if args.check:
        check_llama()
    if args.model:
        path = freq_file_for(args.model)
        if os.path.basename(path) == os.path.basename(FREQ_FILE):
            sys.exit(f"{args.model} uses the released file {path}; nothing to build")
        counts = count_tokens(args.model)
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(counts.tolist(), fh)
        print("wrote", path)


if __name__ == "__main__":
    main()
