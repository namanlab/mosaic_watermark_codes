# Reproducibility guide

This repository contains the complete source code, experiment scripts, the prompt
data, and the analysis scripts needed to reproduce every table and figure in the
MOSAIC paper. It is self-contained.

Everything is deterministic under a single fixed seed, and the same per-prompt
payloads are used for MOSAIC and every baseline.

## 1. Environment

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # torch, transformers, numpy, ...
```

Experiments were run on a single GPU with about 16 GB of memory (the paper's runs
use Llama-2-7B in half precision). A CUDA GPU is recommended. Analysis needs R with
`jsonlite`, `ggplot2`, `scales`, and `dplyr`, plus `python3` with `numpy` (the R
analysis script shells out to `compute_match_ci.py` for the bootstrap intervals).

Model weights for `meta-llama/Llama-2-7b-hf` are downloaded from the Hugging Face
Hub on first use and require accepting the Llama-2 license on your HF account. The
OpenGen prompts are bundled here in `data/opengen_data.parquet`, so no dataset
download is needed.

## 2. Generating the result files

```
# Main table, robustness, delta sweep, token-budget sweep (writes JSONs to results/)
python run_experiments_full.py --model meta-llama/Llama-2-7b-hf

# The six baselines under the shared protocol
python run_baseline_experiments.py --model meta-llama/Llama-2-7b-hf
```

The seed is fixed inside these scripts; rerunning reproduces the same payloads,
generations, and decoded messages. See the top of each script for the exact CLI
flags used for each part of the paper (payload sizes, prompt subsets, attack
grids, `delta` and token-budget grids).

## 3. Regenerating tables and figures

The run scripts write their JSON output into `results/`. The analysis script lives
in `results/` and reads those JSONs, writing `paper/tables/*.tex` and
`paper/figures/*.pdf`:

```
cd results
Rscript make_paper_assets.R
```

Paper artifact -> source:

| Paper item                         | Produced by                              |
|------------------------------------|------------------------------------------|
| Table 1 (main results + 95% CI)    | `results/make_paper_assets.R`            |
| Table 2 (baseline comparison)      | `results/make_paper_assets.R`            |
| Robustness table / figures         | `results/make_paper_assets.R`            |
| Token-budget heatmap               | `results/make_paper_assets.R`            |
| Delta trade-off figure             | `results/make_paper_assets.R`            |

The `+/-95%` column of Table 1 is a nonparametric bootstrap confidence interval
for the exact-match rate, resampled over the prompts (10,000 resamples, seed
20240611). `make_paper_assets.R` computes it by invoking `compute_match_ci.py`,
which streams the per-prompt match flags out of the main result JSON and writes
`results/main_bootstrap_ci.json`; that step reruns automatically whenever the
result JSON changes. To run it on its own: `python results/compute_match_ci.py`.

## 4. Method and baseline source

- `src/mosaic.py`          MOSAIC embedding and belief-propagation decoding
- `src/content_prf.py`     order-canonical content-addressed keying
- `src/fountain_bp.py`     fountain/LT parity layer and BP
- `src/conv_code.py`       coding-abstraction helpers
- `src_baselines/`         faithful reimplementations of the six baselines on the
                           shared green-list layer (Yoo, Qu, Fernandez, Wang,
                           Cohen, Janas); `reed_solomon.py` is the GF(2^4) codec
                           used by the Qu reimplementation
- `results/make_paper_assets.R`   regenerates all tables and figures
- `results/compute_match_ci.py`   bootstrap CI for the main-results table
- `results/merge_main_results.R`  merges sharded main-run JSONs

See `README.md` for the API and a full description of each file.
