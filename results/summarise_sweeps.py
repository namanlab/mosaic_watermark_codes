#!/usr/bin/env python3
"""Compact summaries of the two sweep files, read line by line.

sweep_data_delta.json (104 MB) and sweep_data_tokens.json (171 MB) hold every
per-prompt record with its texts, while the paper needs only
  - the match rate and mean perplexity per (delta, payload), and
  - the exact-match outcome per (payload, prompt, token budget T).
Both files are pretty-printed with one key per line, so this script reads them
line by line and keeps only those values; memory stays at a few MB instead of
the gigabytes a full JSON parse would take.

Outputs (results/derived/):
  sweep_delta_summary.csv    delta, k, match_rate, mean_ppl
  sweep_tokens_records.csv   k, prompt_key, T, match
Each output is skipped when it already exists. The row counts are checked
against the configured number of prompts.

Usage:  python results/summarise_sweeps.py
"""
import csv
import os
import re

RES = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(RES, "derived")
F_DELTA = os.path.join(RES, "sweep_data_delta.json")
F_TOK = os.path.join(RES, "sweep_data_tokens.json")
O_DELTA = os.path.join(OUT, "sweep_delta_summary.csv")
O_TOK = os.path.join(OUT, "sweep_tokens_records.csv")

NUM = r"(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)"


def n_prompts(path):
    """n_prompts from the config block at the top of a sweep file."""
    with open(path) as f:
        head = f.read(4096)
    return int(re.search(r'"n_prompts":\s*(\d+)', head).group(1))


def summarise_delta():
    # layout: data.delta[i] = {"delta": d, "results": {"<k>": {"match_rate": ..,
    #   "mean_bit_accuracy": .., "mean_ppl": .., "results": [per-prompt ...]}}}
    re_delta = re.compile(r'^ {8}"delta": ' + NUM + r',?\s*$')
    re_k = re.compile(r'^ {10}"(\d+)": \{\s*$')
    re_mr = re.compile(r'^ {12}"match_rate": ' + NUM + r',?\s*$')
    re_ppl = re.compile(r'^ {12}"mean_ppl": ' + NUM + r',?\s*$')
    rows, d, k, mr = [], None, None, None
    with open(F_DELTA) as f:
        for line in f:
            if (m := re_delta.match(line)):
                d, k, mr = float(m.group(1)), None, None
            elif (m := re_k.match(line)):
                k, mr = int(m.group(1)), None
            elif (m := re_mr.match(line)) and k is not None:
                mr = float(m.group(1))
            elif (m := re_ppl.match(line)) and mr is not None:
                rows.append((d, k, mr, float(m.group(1))))
                mr = None
    assert rows, "no delta summaries found"
    with open(O_DELTA, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta", "k", "match_rate", "mean_ppl"])
        w.writerows(rows)
    print(f"wrote {O_DELTA} ({len(rows)} rows)", flush=True)


def summarise_tokens():
    # layout: data.heatmap["<k>"]["<prompt>"]["results"] = [{"T": .., ...,
    #   "match": true|false, ...}, ...]
    re_k = re.compile(r'^ {6}"(\d+)": \{\s*$')
    re_p = re.compile(r'^ {8}"(\d+)": \{\s*$')
    re_T = re.compile(r'^\s*"T": (\d+),?\s*$')
    re_match = re.compile(r'^\s*"match": (true|false),?\s*$')
    rows, k, p, T = [], None, None, None
    in_heatmap = False
    with open(F_TOK) as f:
        for line in f:
            if not in_heatmap:
                in_heatmap = line.strip().startswith('"heatmap"')
                continue
            if (m := re_k.match(line)):
                k, p, T = int(m.group(1)), None, None
            elif (m := re_p.match(line)):
                p, T = int(m.group(1)), None
            elif (m := re_T.match(line)):
                T = int(m.group(1))
            elif (m := re_match.match(line)) and T is not None:
                rows.append((k, p, T, int(m.group(1) == "true")))
                T = None
    n = n_prompts(F_TOK)
    per_k = {}
    for r in rows:
        per_k.setdefault(r[0], set()).add(r[1])
    for kk, ps in per_k.items():
        assert len(ps) == n, f"k={kk}: {len(ps)} prompts, expected {n}"
    with open(O_TOK, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "prompt_key", "T", "match"])
        w.writerows(rows)
    print(f"wrote {O_TOK} ({len(rows)} rows)", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(O_DELTA):
        summarise_delta()
    if not os.path.exists(O_TOK):
        summarise_tokens()


if __name__ == "__main__":
    main()
