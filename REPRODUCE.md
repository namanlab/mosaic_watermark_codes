# Reproducibility guide

This repository contains the complete source code, experiment scripts, the prompt
data, and the analysis scripts needed to reproduce every table, figure, and
quoted number in the MOSAIC paper. The one external input is the token-frequency
file of the Qu et al. baseline, which its authors distribute (step 2).

Everything is deterministic under a single fixed seed, and the same per-prompt
payloads are used for MOSAIC and every baseline.

## 1. Environment

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # torch, transformers, numpy, ...
```

Experiments were run on a single GPU with about 16 GB of memory (the paper's runs
use Llama-2-7B in half precision). A CUDA GPU is recommended. Analysis needs R with
`jsonlite`, `ggplot2`, `scales`, `dplyr`, and `tidyr`, plus `python3` with `numpy`
and `transformers` (the R analysis script calls the Python helpers in `results/`,
which use only the tokeniser and load no language model).

Model weights for `meta-llama/Llama-2-7b-hf` are downloaded from the Hugging Face
Hub on first use and require accepting the Llama-2 license on your HF account. The
OpenGen prompts are bundled here in `data/opengen_data.parquet`, so no dataset
download is needed.

## 2. Qu et al.'s token frequencies

The Qu et al. baseline needs the token-frequency file from its authors' data
release, which is distributed by them (CC BY-NC 4.0) and is not included here:

1. Download `data_share.zip` from the "Download data" link in the README of
   https://github.com/randomizedtree/segment-watermark.
2. Copy `token_freq_llama.pkl` from it to `data/qu/token_freq_llama.pkl`.

The balanced token-to-segment maps are built from it on first use and cached in
`data/qu/`. Every other method runs without this file.

## 3. Generating the result files (GPU)

```
# MOSAIC: main table on all 7,711 prompts
python run_experiments_full.py --experiment generate --model meta-llama/Llama-2-7b-hf
python run_experiments_full.py --experiment quality  --model meta-llama/Llama-2-7b-hf

# MOSAIC: robustness, paraphrasing, delta and token-budget sweeps on the 10% subset
SUBSET_FRAC=0.1 python run_experiments_full.py --experiment robust --model meta-llama/Llama-2-7b-hf
SUBSET_FRAC=0.1 python run_experiments_full.py --experiment sweep  --model meta-llama/Llama-2-7b-hf

# The six baselines on the first 1,000 prompts
MAIN_MAX_PROMPTS=1000 python run_baseline_experiments.py --method all --model meta-llama/Llama-2-7b-hf
```

The seed is fixed inside these scripts; rerunning reproduces the same payloads,
generations, and decoded messages. Every script checkpoints and resumes from its
own output file, so remove an older output of the same name before rerunning a
method with a different implementation. See the top of each script for its
options.

Place the outputs where the analysis expects them:

- MOSAIC files stay in `results/`: `opengen_main_<model>.json`,
  `opengen_main_<model>_Perplexity.json`, `opengen_robustness_<model>.json`,
  `opengen_paraphrase_robustness_<model>.json`, `sweep_data_delta.json`,
  `sweep_data_tokens.json`.
- Baseline files go to `results/other_models/`:
  `opengen_baseline_<method>_<model>.json`.

## 4. Regenerating tables, figures, and quoted numbers (laptop)

```
Rscript results/make_paper_assets.R
```

This single command writes `paper/tables/*.tex`, `paper/figures/*.pdf`, and
`paper/tables/numbers.tex` (every number quoted in the text, cited in the paper
as `\val{key}`). It calls the Python helpers below only when their output in
`results/derived/` is missing or older than its inputs; after a baseline is
regenerated, only the outputs that depend on it are recomputed.

| helper | output in `results/derived/` | used for |
|---|---|---|
| `compute_match_ci.py` | `../main_bootstrap_ci.json` | 95% bootstrap CI column of Table 1 |
| `summarise_sweeps.py` | `sweep_delta_summary.csv`, `sweep_tokens_records.csv` | delta trade-off and token-budget figures |
| `sample_diagnostics.py` | `mosaic_sample_diagnostics.csv`, `baseline_sample_repetition.csv`, `common_sample.json` | small-payload analysis; Table 2 and its paired tests |
| `robustness_comparison.py` | `robustness_comparison.csv` | robustness comparison figure and table |
| `dedup_decoding.py` | `mosaic_dedup_decoding.csv` | decoding with repeated observations counted once |

Resource use: every helper runs in a single process, reads the large result
files one sample at a time (`stream_samples.py`), and loads no language model;
peak memory stays below about 0.5 GB. The robustness comparison is the longest
step (a few hours on one core for all methods); the rest take minutes.

Paper artifact -> source:

| Paper item                                     | Produced by                                  |
|------------------------------------------------|----------------------------------------------|
| Table 1 (main results, 95% CI, perplexity)     | `make_paper_assets.R`, `compute_match_ci.py` |
| Table 2 (baselines on the shared prompts)      | `make_paper_assets.R`, `sample_diagnostics.py` |
| Robustness comparison (figure and table)       | `make_paper_assets.R`, `robustness_comparison.py` |
| MOSAIC robustness on the 10% subset            | `make_paper_assets.R`                        |
| Token-budget heatmap, delta trade-off          | `make_paper_assets.R`, `summarise_sweeps.py` |
| Small-payload analysis (figures and tables)    | `make_paper_assets.R`, `sample_diagnostics.py`, `dedup_decoding.py` |
| Paired McNemar tests and bootstrap intervals   | `sample_diagnostics.py`                      |
| Every number quoted in the text                | `make_paper_assets.R` -> `numbers.tex`       |

The bootstrap intervals use 10,000 resamples over prompts with seed 20240611.

## 5. Method and baseline source

- `src/mosaic.py`          MOSAIC embedding and belief-propagation decoding
- `src/content_prf.py`     order-canonical content-addressed keying
- `src/fountain_bp.py`     fountain/LT parity layer and BP
- `src/conv_code.py`       earlier convolutional code, kept for reference
- `src_baselines/`         the six baselines on the shared green-list layer (Yoo,
                           Qu, Fernandez, Wang, Cohen, Janas); `qu_balance.py` and
                           `reed_solomon.py` serve the Qu et al. method
- `results/`               the analysis pipeline listed above

See `README.md` for the API, the baselines' fidelity notes, and a description of
each file.
