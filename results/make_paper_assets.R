#!/usr/bin/env Rscript
# make_paper_assets.R
# -------------------
# Regenerates every table (paper/tables/*.tex), every data figure
# (paper/figures/*.pdf), and every number quoted in the text of the MOSAIC paper
# (paper/tables/numbers.tex) from the result files in this results/ folder. Old
# tables and figures are deleted first so the paper always reflects the current
# results. The paper cites numbers as \val{key}; a key missing from numbers.tex
# stops the LaTeX build.
#
#   Stored experiment outputs (read only):
#     opengen_main_meta-llama_Llama-2-7b-hf.json              all 7,711 prompts
#     opengen_main_meta-llama_Llama-2-7b-hf_Perplexity.json   all 7,711 prompts
#     opengen_robustness_meta-llama_Llama-2-7b-hf.json        10% subset
#     opengen_paraphrase_robustness_meta-llama_Llama-2-7b-hf.json
#     sweep_data_delta.json, sweep_data_tokens.json           10% subset
#     other_models/opengen_baseline_<method>_<model>.json     prompts 0-999
#
#   Derived files (written by the helper scripts below into derived/ when
#   missing; no language model is loaded by any of them, each runs in a single
#   process and streams the large files, so the pipeline fits a 16 GB laptop):
#     compute_match_ci.py       -> main_bootstrap_ci.json
#     summarise_sweeps.py       -> derived/sweep_delta_summary.csv,
#                                  derived/sweep_tokens_records.csv
#     sample_diagnostics.py     -> derived/mosaic_sample_diagnostics.csv,
#                                  derived/baseline_sample_repetition.csv,
#                                  derived/common_sample.json
#     robustness_comparison.py  -> derived/robustness_comparison.csv
#     redecode_main.py          -> derived/opengen_main_<model>_distinct.json,
#                                  derived/mosaic_distinct_<model>.csv (per model)
#     null_detection.py         -> derived/null_detection_<model>.csv (per model)
#
# Run:  Rscript make_paper_assets.R
suppressPackageStartupMessages({
  library(jsonlite); library(ggplot2); library(scales); library(dplyr)
})

# ---------------------------------------------------------------- paths -----
args <- commandArgs(FALSE)
fa <- grep("--file=", args, value = TRUE)
RES <- if (length(fa)) dirname(normalizePath(sub("--file=", "", fa[1]))) else getwd()
PAPER <- normalizePath(file.path(RES, "..", "paper"))
TAB <- file.path(PAPER, "tables"); FIG <- file.path(PAPER, "figures")
DER <- file.path(RES, "derived")
for (d in c(TAB, FIG, DER)) dir.create(d, showWarnings = FALSE, recursive = TRUE)

MODEL <- "meta-llama_Llama-2-7b-hf"
f_raw  <- file.path(RES, paste0("opengen_main_", MODEL, ".json"))      # as generated
# every MOSAIC analysis reads the texts decoded with the deployed detector
f_main <- file.path(RES, "derived", paste0("opengen_main_", MODEL, "_distinct.json"))
f_ppl  <- file.path(RES, paste0("opengen_main_", MODEL, "_Perplexity.json"))
f_rob  <- file.path(RES, paste0("opengen_robustness_", MODEL, ".json"))
f_par  <- file.path(RES, paste0("opengen_paraphrase_robustness_", MODEL, ".json"))
f_dlt  <- file.path(RES, "sweep_data_delta.json")
f_tok  <- file.path(RES, "sweep_data_tokens.json")
stopifnot(all(file.exists(c(f_raw, f_ppl, f_rob, f_par, f_dlt, f_tok))))

# ---- Python for the helper scripts: the project venv when present ----
PY <- local({
  # keep the venv's own symlink: resolving it (normalizePath) would run the base
  # interpreter outside the venv, without the venv's packages
  v <- file.path(normalizePath(file.path(RES, "..")), "venv", "bin", "python")
  if (file.exists(v)) v else {
    p <- Sys.which("python3"); if (p == "") p <- Sys.which("python"); p
  }
})
# Run a helper script when any of its outputs is missing or older than one of
# its inputs. deps maps each output to the result files it is computed from; a
# stale output is removed first, since each helper recomputes only missing
# outputs. keep_stale leaves stale outputs in place for helpers that update
# themselves incrementally (robustness_comparison.py recomputes only the
# methods whose result files changed).
run_helper <- function(script, deps, keep_stale = FALSE, args = character()) {
  stale <- vapply(names(deps), function(o) !file.exists(o) ||
                    any(file.info(deps[[o]])$mtime > file.info(o)$mtime), TRUE)
  if (!any(stale)) return(invisible(TRUE))
  if (!keep_stale) invisible(file.remove(names(deps)[stale & file.exists(names(deps))]))
  cat("  running", script, "for", paste(basename(names(deps)[stale]), collapse = ", "), "...\n")
  status <- system2(PY, shQuote(c(file.path(RES, script), args)))
  stopifnot(status == 0, all(file.exists(names(deps))))
}
f_diag   <- file.path(DER, "mosaic_sample_diagnostics.csv")
f_rep    <- file.path(DER, "baseline_sample_repetition.csv")
f_common <- file.path(DER, "common_sample.json")
f_comp   <- file.path(DER, "robustness_comparison.csv")
f_dist   <- file.path(DER, paste0("mosaic_distinct_", MODEL, ".csv"))
f_sdelta <- file.path(DER, "sweep_delta_summary.csv")
f_stok   <- file.path(DER, "sweep_tokens_records.csv")
BL_METHODS <- c("mpac", "qu", "cyclic", "wang", "cohen", "cohen_adapted", "janas")
f_base <- file.path(RES, "other_models",
                    sprintf("opengen_baseline_%s_%s.json", BL_METHODS, MODEL))
stopifnot(all(file.exists(f_base)))
run_helper("redecode_main.py", setNames(list(f_raw, f_raw), c(f_main, f_dist)))
run_helper("summarise_sweeps.py", setNames(list(f_dlt, f_tok), c(f_sdelta, f_stok)))
run_helper("sample_diagnostics.py",
           setNames(list(f_main, f_base, c(f_main, f_base)), c(f_diag, f_rep, f_common)))
run_helper("robustness_comparison.py", setNames(list(c(f_main, f_base)), f_comp),
           keep_stale = TRUE)
# second generation models: each enters once its MOSAIC result file exists
SECOND <- c(mistral = "mistralai/Mistral-7B-v0.1", qwen = "Qwen/Qwen2.5-7B")
SM_BASE <- c("qu", "janas", "mpac", "cyclic", "wang", "cohen_adapted", "cohen")
sm_raw <- function(model) file.path(RES, sprintf("opengen_main_%s.json", gsub("/", "_", model)))
sm_files <- function(model) {                          # decoded MOSAIC file first, then baselines
  tag <- gsub("/", "_", model)
  b <- file.path(RES, "other_models", sprintf("opengen_baseline_%s_%s.json", SM_BASE, tag))
  c(file.path(DER, sprintf("opengen_main_%s_distinct.json", tag)), b[file.exists(b)])
}
SECOND <- SECOND[file.exists(vapply(SECOND, sm_raw, ""))]
for (m in SECOND)
  run_helper("redecode_main.py", setNames(list(sm_raw(m), sm_raw(m)),
             c(sm_files(m)[1], file.path(DER, sprintf("mosaic_distinct_%s.csv", gsub("/", "_", m))))),
             args = m)
f_sm <- setNames(file.path(DER, sprintf("second_model_%s.json", names(SECOND))), names(SECOND))
if (length(SECOND)) run_helper("second_model.py", setNames(lapply(SECOND, sm_files), f_sm))
# the same token-level attacks on each further model's stored texts
f_comp_sm <- setNames(file.path(DER, sprintf("robustness_comparison_%s.csv", names(SECOND))),
                      names(SECOND))
# the detector on unwatermarked and human text, every model
ALLM <- c(llama = "meta-llama/Llama-2-7b-hf", SECOND)
f_null <- setNames(file.path(DER, sprintf("null_detection_%s.csv", gsub("/", "_", ALLM))), names(ALLM))
for (tag in names(ALLM))
  run_helper("null_detection.py", setNames(list(sm_raw(ALLM[[tag]])), f_null[[tag]]),
             args = ALLM[[tag]])
for (tag in names(SECOND))
  run_helper("robustness_comparison.py", setNames(list(sm_files(SECOND[[tag]])), f_comp_sm[[tag]]),
             keep_stale = TRUE, args = SECOND[[tag]])

# ---- clear old assets so nothing stale survives ----
old <- c(list.files(TAB, "\\.tex$", full.names = TRUE),
         list.files(FIG, "\\.pdf$", full.names = TRUE))
invisible(suppressWarnings(file.remove(old)))
cat("cleared", length(old), "old tables/figures\n")

KS <- c(16, 20, 24, 32, 48)

# ----------------------------------------------------------- helpers --------
# Extract just the "aggregate":[...] block from a large JSON without parsing the
# (very large) samples array. The aggregate entries are flat objects, so the
# first closing bracket ends the array.
read_aggregate <- function(path) {
  # the aggregate block sits in the first few KB (before "samples"), so read only
  # the head of the file instead of the whole multi-hundred-MB JSON
  txt <- readChar(path, min(65536, file.info(path)$size), useBytes = TRUE)
  m <- regmatches(txt, regexpr('(?s)"aggregate"\\s*:\\s*\\[.*?\\]', txt, perl = TRUE))
  stopifnot(length(m) == 1)
  fromJSON(paste0("{", m, "}"))$aggregate
}

# Nonparametric bootstrap 95% CI half-widths (in %) for the exact-match rate,
# resampling over prompts. The per-prompt "match" flags live in the (multi-
# hundred-MB) samples array; streaming them is delegated to compute_match_ci.py
# (R's regex engine mis-handles subject strings this large), which writes a small
# JSON that we read back. The helper is rerun whenever its output is missing or
# older than the main results file. Returns a vector keyed by payload.
bootstrap_match_ci <- function(path, ks) {
  ci_json <- file.path(RES, "main_bootstrap_ci.json")
  helper  <- file.path(RES, "compute_match_ci.py")
  stale <- !file.exists(ci_json) ||
    file.info(ci_json)$mtime < file.info(path)$mtime
  if (stale) {
    status <- system2(PY, c(shQuote(helper), shQuote(path), shQuote(ci_json)))
    stopifnot(status == 0, file.exists(ci_json))
  }
  ci <- fromJSON(ci_json)
  vapply(ks, function(k) ci[[as.character(k)]]$half, 0) |> setNames(as.character(ks))
}

sci_tex <- function(x, digits = 1) {            # 2.66e-37 -> "$2.7\times10^{-37}$"
  e <- floor(log10(x)); mant <- x / 10^e
  sprintf(paste0("$%.", digits, "f\\times10^{%d}$"), mant, e)
}
p_tex <- function(p) {                          # p-value as math content (no $...$)
  if (p >= 0.1) sprintf("%.2f", p) else if (p >= 0.001) sprintf("%.3f", p) else
    gsub("\\$", "", sci_tex(p))
}
f0 <- function(x) sprintf("%.0f", x)
f1 <- function(x) sprintf("%.1f", x)
f2 <- function(x) sprintf("%.2f", x)
f3 <- function(x) sprintf("%.3f", x)
WORD <- c("one", "two", "three", "four", "five", "six")
big <- function(n) gsub(",", "{,}", format(n, big.mark = ",", trim = TRUE))

# ---- registry of every number the text quotes (written to numbers.tex) ----
NUM <- list()
# method ids such as cohen_adapted enter keys without the underscore
setv <- function(key, value) { NUM[[gsub("_", "", key)]] <<- as.character(value); invisible(value) }

theme_paper <- theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        legend.position = "top",
        legend.title = element_text(size = 10),
        strip.background = element_rect(fill = "grey92", colour = "grey70"),
        plot.margin = margin(4, 8, 4, 4))

PAL <- c("16" = "#1b9e77", "20" = "#7570b3", "24" = "#d95f02",
         "32" = "#e7298a", "48" = "#666666")

savefig <- function(name, plot, w, h) {
  ggsave(file.path(FIG, name), plot, width = w, height = h, units = "in",
         device = "pdf", useDingbats = FALSE)
  cat("  figure:", name, "\n")
}
writetab <- function(lines, name) {
  writeLines(lines, file.path(TAB, name)); cat("  table:", name, "\n")
}

# Method labels, colours, and the keying of each scheme's payload allocation
# (how a token is assigned to a payload part in our implementation). The tables
# and figures show the adapted instantiation of Cohen et al.; the exact Figure 2
# instantiation ("cohen") is reported in its own appendix paragraph (section 3).
METH <- c("mosaic", "qu", "janas", "cyclic", "wang", "mpac", "cohen_adapted")
LAB <- c(mosaic = "MOSAIC (ours)", qu = "Qu et al.", janas = "Janas et al.",
         cyclic = "Fernandez et al.", wang = "Wang et al.", mpac = "Yoo et al.",
         cohen = "Cohen et al.", cohen_adapted = "Cohen et al. (adapted)")
MCOL <- c("MOSAIC (ours)" = "#000000", "Qu et al." = "#e41a1c", "Janas et al." = "#377eb8",
          "Fernandez et al." = "#4daf4a", "Wang et al." = "#984ea3",
          "Yoo et al." = "#ff7f00", "Cohen et al." = "#a65628",
          "Cohen et al. (adapted)" = "#f781bf")

# =====================================================================
# 1. MAIN RESULTS  (Table 1 + match-vs-bits + detection-strength figures)
# =====================================================================
ag_main <- read_aggregate(f_main)
ag_ppl  <- read_aggregate(f_ppl)
main <- data.frame(
  k = ag_main$k, match = ag_main$match_rate, bitacc = ag_main$bit_accuracy,
  medp = ag_main$median_p_value, n = ag_main$n,
  ppl_base = ag_ppl$mean_ppl_plain[match(ag_main$k, ag_ppl$k)],
  ppl_mark = ag_ppl$mean_ppl_watermarked[match(ag_main$k, ag_ppl$k)])
main <- main[order(main$k), ]
ci_half <- bootstrap_match_ci(f_main, main$k)

# Table 1: MOSAIC on every generation model, Llama-2 over all prompts and the
# further models over the shared prompts (second_model.py)
SMD <- lapply(f_sm, fromJSON)
SMAG <- lapply(SECOND, function(m) read_aggregate(sm_files(m)[1]))
MSHORT <- c(llama = "Llama-2-7B", mistral = "Mistral-7B", qwen = "Qwen2.5-7B")
mos <- c(list(llama = data.frame(k = main$k, match = main$match,
                                 half = unname(ci_half[as.character(main$k)]),
                                 bit = main$bitacc, ppl = main$ppl_mark, n = main$n)),
         lapply(setNames(names(SECOND), names(SECOND)), function(tag) {
           r <- SMD[[tag]]$rates; r <- r[r$method == "mosaic", ]; r <- r[order(r$k), ]
           a <- SMAG[[tag]]; a <- a[match(r$k, a$k), ]
           data.frame(k = r$k, match = r$match_rate, half = (r$ci_hi - r$ci_lo) / 2,
                      bit = a$bit_accuracy, ppl = a$mean_ppl_watermarked, n = r$n)
         }))
stopifnot(all(vapply(mos, function(d) isTRUE(all.equal(as.numeric(d$k), as.numeric(KS))), TRUE)))
nm <- length(mos)
t1_rows <- vapply(seq_along(KS), function(i) paste0(KS[i], " & ", paste(vapply(mos, function(d)
  sprintf("%.1f & %.2f & %.1f & %.1f", d$match[i], d$half[i], d$bit[i], d$ppl[i]), ""),
  collapse = " & "), " \\\\"), "")
writetab(c(
  sprintf("\\begin{tabular}{@{}r %s@{}}", paste(rep("rrrr", nm), collapse = " ")), "\\toprule",
  paste0(" & ", paste(sprintf("\\multicolumn{4}{c}{%s, %s prompts}", MSHORT[names(mos)],
                              vapply(mos, function(d) big(unique(d$n)), "")), collapse = " & "),
         " \\\\"),
  paste(sprintf("\\cmidrule(lr){%d-%d}", 2 + 4 * (seq_len(nm) - 1), 1 + 4 * seq_len(nm)),
        collapse = ""),
  paste0("$k$ & ", paste(rep("Match & $\\pm$ & Bit acc. & PPL", nm), collapse = " & "), " \\\\"),
  "\\midrule", t1_rows, "\\bottomrule", "\\end{tabular}"), "main_results.tex")

N_MAIN <- unique(main$n); stopifnot(length(N_MAIN) == 1)
setv("n.main", big(N_MAIN))
for (i in seq_len(nrow(main))) {
  k <- main$k[i]
  setv(sprintf("main.match.k%d", k), f1(main$match[i]))
  setv(sprintf("main.medp.exp.k%d", k), floor(log10(main$medp[i])))
}
setv("main.bitacc.min", f1(min(main$bitacc)))
setv("main.ci.max", f2(max(ci_half)))
setv("ppl.base", f1(unique(round(main$ppl_base, 1))))
setv("ppl.marked.min", f1(min(main$ppl_mark)))
setv("ppl.marked.max", f1(max(main$ppl_mark)))
setv("ppl.marked.mean", f0(mean(main$ppl_mark)))

dfm <- rbind(
  data.frame(k = main$k, value = main$match,  Metric = "Exact match rate"),
  data.frame(k = main$k, value = main$bitacc, Metric = "Bit accuracy"))
p <- ggplot(dfm, aes(k, value, colour = Metric, shape = Metric)) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.4) +
  scale_colour_manual(values = c("Exact match rate" = "#d95f02", "Bit accuracy" = "#1b9e77")) +
  scale_x_continuous(breaks = KS) + scale_y_continuous(limits = c(80, 100)) +
  labs(x = "Payload size k (bits)", y = "Percent", colour = NULL, shape = NULL) +
  theme_paper
savefig("match_vs_bits.pdf", p, 4.3, 3.2)

dfp <- data.frame(k = factor(main$k, levels = KS), strength = -log10(main$medp))
p <- ggplot(dfp, aes(k, strength)) +
  geom_col(fill = "#7570b3", width = 0.65) +
  geom_text(aes(label = sprintf("%.0f", strength)), vjust = -0.4, size = 3.1) +
  labs(x = "Payload size k (bits)", y = expression(-log[10]~"(median p-value)")) +
  expand_limits(y = max(dfp$strength) * 1.12) + theme_paper
savefig("detection_strength.pdf", p, 4.3, 3.2)

# =====================================================================
# 2. CHANNEL AND THEORY CONSTANTS  (measured p_e, token budget, thresholds)
# =====================================================================
diag <- read.csv(f_diag)
diag$pe_true <- 1 - diag$agreement_true              # per-text flip rate
PE <- round(mean(diag$pe_true), 3)                   # pooled (T = 200 for all texts)
pe_k <- tapply(diag$pe_true, diag$k, mean)
EPS <- 0.01; T0 <- 200; W <- 2; DELTA <- 6; TAU <- 0.8; A_USED <- 4
Z <- 2 * sqrt(PE * (1 - PE))
budget <- function(k, eps = EPS) 2 * (k * log(2) + log(1 / eps)) / (1 - Z)
# Robust recovery (Theorem robust): corrupting a fraction eta of the checks with
# fair coins gives a channel with flip rate p_eta = p_e + eta (1/2 - p_e) and
# Bhattacharyya parameter Z_eta; recovery at level EPS needs
# (1 - Z_eta) T0 >= (1 - Z) T_star. The substitution threshold is the largest
# eta allowed, divided by the 1 + w checks one substitution corrupts.
thresh <- function(k) {
  zmax <- 1 - (1 - Z) * budget(k) / T0               # largest admissible Z_eta
  pmax <- (1 - sqrt(1 - zmax^2)) / 2                  # its flip rate
  (pmax - PE) / (0.5 - PE) / (1 + W)
}
setv("pe", f3(PE)); setv("pe.min", f3(min(pe_k))); setv("pe.max", f3(max(pe_k)))
setv("Z", f2(Z)); setv("oneminusZ", f2(1 - Z)); setv("eps", EPS)
setv("rate", f1(2 * log(2) / (1 - Z)))
setv("A.used", A_USED); setv("A.matched", f1(log((1 - PE) / PE)))
setv("A.used.pe", f3(1 / (1 + exp(A_USED))))       # flip rate for which A = 4 is matched
setv("pi.half", sprintf("%.4f", exp(DELTA / TAU) / (exp(DELTA / TAU) + 1)))
for (k in KS) {
  setv(sprintf("budget.k%d", k), f0(ceiling(budget(k))))
  setv(sprintf("thr.k%d", k), f2(thresh(k)))
}

# =====================================================================
# 3. BASELINE COMPARISON ON THE SHARED PROMPTS  (Table 2 + appendix table)
# =====================================================================
cs <- fromJSON(f_common)
rates <- cs$rates; mcn <- cs$mcnemar
N_COMMON <- unique(rates$n); stopifnot(length(N_COMMON) == 1)
setv("n.common", big(N_COMMON))
rate_of <- function(m, k) rates$match_rate[rates$method == m & rates$k == k]
cell_rate <- function(m, k) { v <- rate_of(m, k); if (length(v) == 0) "NA" else f1(v) }
ROWS_BL <- c("cyclic", "wang", "mpac", "cohen_adapted", "qu", "janas")
# Table 2: every method on the shared prompts of every generation model; NA marks
# payloads whose extraction is exponential in k, -- a method not run on that model
T2M <- c("llama", names(SECOND))
mrates <- function(tag) if (tag == "llama") rates else SMD[[tag]]$rates
t2_cell <- function(tag, m, k) {
  r <- mrates(tag)
  if (!(m %in% r$method)) return("--")
  v <- r$match_rate[r$method == m & r$k == k]
  if (length(v) == 0) "NA" else f1(v)
}
t2_line <- function(m) paste0(LAB[[m]], " & ", paste(vapply(T2M, function(tag)
  paste(vapply(KS, function(k) t2_cell(tag, m, k), ""), collapse = " & "), ""),
  collapse = " & "), " \\\\")
nt <- length(T2M)
writetab(c(
  sprintf("\\begin{tabular}{@{}l %s@{}}", paste(rep("rrrrr", nt), collapse = " ")), "\\toprule",
  paste0(" & ", paste(sprintf("\\multicolumn{5}{c}{%s}", MSHORT[T2M]), collapse = " & "), " \\\\"),
  paste(sprintf("\\cmidrule(lr){%d-%d}", 2 + 5 * (seq_len(nt) - 1), 1 + 5 * seq_len(nt)),
        collapse = ""),
  paste0("Method & ", paste(rep(paste(KS, collapse = " & "), nt), collapse = " & "), " \\\\"),
  "\\midrule", vapply(ROWS_BL, t2_line, ""), "\\midrule", t2_line("mosaic"),
  "\\bottomrule", "\\end{tabular}"), "baseline_reproduced.tex")

# match rate against payload size, one panel per generation model
cmp_df <- do.call(rbind, lapply(T2M, function(tag) {
  r <- mrates(tag)[, c("method", "k", "match_rate")]
  r <- r[r$method %in% c("mosaic", ROWS_BL), ]
  r$model <- MSHORT[[tag]]
  r
}))
cmp_df$Method <- factor(LAB[cmp_df$method], levels = LAB[METH])
cmp_df$model <- factor(cmp_df$model, levels = MSHORT[T2M])
p <- ggplot(cmp_df, aes(k, match_rate, colour = Method, group = Method)) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.5) +
  facet_wrap(~model, nrow = 1) +
  scale_colour_manual(values = MCOL, name = NULL) +
  scale_x_continuous(breaks = KS) + scale_y_continuous(limits = c(0, 100)) +
  labs(x = "Payload size k (bits)", y = "Exact match rate (%)") +
  theme_paper + guides(colour = guide_legend(nrow = 2))
savefig("comparison_models.pdf", p, 7.0, 3.0)

# appendix: every rate with its 95% interval, and the paired McNemar test
ci_cell <- function(m, k) {
  r <- rates[rates$method == m & rates$k == k, ]
  if (nrow(r) == 0) return("NA")
  sprintf("%.1f [%.1f, %.1f]", r$match_rate, r$ci_lo, r$ci_hi)
}
mc_cell <- function(m, k) {
  r <- mcn[mcn$baseline == m & mcn$k == k, ]
  if (nrow(r) == 0) return("NA")
  sprintf("%d/%d, $%s$", r$mosaic_only, r$baseline_only, p_tex(r$p_value))
}
ci_rows <- vapply(c("mosaic", ROWS_BL), function(m) paste0(
  LAB[[m]], " & ", paste(vapply(KS, function(k) ci_cell(m, k), ""), collapse = " & "),
  " \\\\"), "")
mc_rows <- vapply(ROWS_BL, function(m) paste0(
  LAB[[m]], " & ", paste(vapply(KS, function(k) mc_cell(m, k), ""), collapse = " & "),
  " \\\\"), "")
writetab(c(
  "\\begin{tabular}{@{}l ccccc@{}}", "\\toprule",
  "& \\multicolumn{5}{c}{Payload size $k$ (bits)} \\\\", "\\cmidrule(lr){2-6}",
  paste0("Method & ", paste(KS, collapse = " & "), " \\\\"), "\\midrule",
  "\\multicolumn{6}{@{}l}{Exact match rate (\\%) with 95\\% bootstrap interval} \\\\",
  ci_rows, "\\midrule",
  "\\multicolumn{6}{@{}l}{McNemar test against MOSAIC: MOSAIC-only / baseline-only successes, exact $p$} \\\\",
  mc_rows, "\\bottomrule", "\\end{tabular}"), "common_sample.tex")

for (m in c("mosaic", ROWS_BL)) for (k in KS) {
  v <- rate_of(m, k); if (length(v)) setv(sprintf("cs.%s.k%d", m, k), f1(v))
}
for (i in seq_len(nrow(mcn))) {
  r <- mcn[i, ]
  setv(sprintf("mc.%s.k%d.p", r$baseline, r$k), p_tex(r$p_value))
  setv(sprintf("mc.%s.k%d.mo", r$baseline, r$k), r$mosaic_only)
  setv(sprintf("mc.%s.k%d.bo", r$baseline, r$k), r$baseline_only)
}
# exact Figure 2 instantiation of Cohen et al. (appendix paragraph): match rate,
# bit accuracy, and the block-coverage arithmetic behind its zero match rate
COHEN_B <- as.integer(sub(".*=\\s*(\\d+).*", "\\1",
  grep("^BLOCK\\s*=", readLines(file.path(RES, "..", "src_baselines", "cohen.py")), value = TRUE)))
ag_cohen <- read_aggregate(f_base[match("cohen", BL_METHODS)])
for (k in KS) {
  setv(sprintf("cs.cohen.k%d", k), f1(rate_of("cohen", k)))
  setv(sprintf("cohen.bitacc.k%d", k), f1(ag_cohen$bit_accuracy[ag_cohen$k == k]))
}
setv("cohen.match.max", f1(max(rates$match_rate[rates$method == "cohen"])))
setv("cohen.B", COHEN_B)
setv("cohen.blocks", ceiling(T0 / COHEN_B))                       # blocks in T0 tokens
setv("cohen.mintok.k16", 16 * COHEN_B)                            # one block per bit
setv("cohen.covtok.k16", big(round(COHEN_B * 16 * sum(1 / (1:16)))))      # expected, random allocation
setv("cs.mosaic.fail.k16", round(N_COMMON * (1 - rate_of("mosaic", 16) / 100)))
setv("cs.mosaic.drop", f1(rate_of("mosaic", 16) - rate_of("mosaic", 48)))
# gap between the most accurate baseline and MOSAIC where a baseline leads
gaps <- vapply(KS, function(k) {
  b <- rates$match_rate[rates$k == k & rates$method %in% ROWS_BL]
  max(b) - rate_of("mosaic", k)
}, 0)
setv("gap.min", f1(min(gaps[gaps > 0]))); setv("gap.max", f1(max(gaps[gaps > 0])))
setv("gap.kmax", max(KS[gaps > 0]))
# payloads at which MOSAIC exceeds every baseline, significantly so against the
# strongest one (exact McNemar p < 0.05)
leads <- KS[vapply(KS, function(k) {
  r <- rates[rates$k == k & rates$method %in% ROWS_BL, ]
  best <- r$method[which.max(r$match_rate)]
  p <- mcn$p_value[mcn$baseline == best & mcn$k == k]
  rate_of("mosaic", k) > max(r$match_rate) && length(p) == 1 && p < 0.05
}, TRUE)]
and_list <- function(x) if (length(x) == 0) "none" else if (length(x) < 2) as.character(x) else
  paste0(paste(head(x, -1), collapse = ", "), if (length(x) > 2) "," else "", " and ", tail(x, 1))
setv("lead.ks", and_list(leads))

# =====================================================================
# 3b. SECOND GENERATION MODELS  (appendix table + quoted numbers)
# =====================================================================
MLAB <- c(mistral = "Mistral-7B-v0.1", qwen = "Qwen2.5-7B")
sm_rows <- c(); XM <- list()                 # XM collects per-model values for the cross-model text
for (tag in names(SECOND)) {
  sm <- SMD[[tag]]
  sr <- sm$rates; smc <- sm$mcnemar
  srate <- function(m, k) sr$match_rate[sr$method == m & sr$k == k]
  sp <- function(m, k) smc$p_value[smc$baseline == m & smc$k == k]
  meths <- c("mosaic", intersect(ROWS_BL, unique(smc$baseline)))    # order of Table 2
  bl <- setdiff(meths, "mosaic")
  key <- function(...) paste0("sm.", tag, ".", paste0(...))
  setv(key("name"), MLAB[[tag]]); setv(key("n"), big(unique(sr$n)))
  setv(key("nbl"), WORD[length(bl)])
  for (m in meths) for (k in KS) { v <- srate(m, k); if (length(v)) setv(key(m, ".k", k), f1(v)) }
  for (i in seq_len(nrow(smc))) setv(key("mc.", smc$baseline[i], ".k", smc$k[i], ".p"),
                                     p_tex(smc$p_value[i]))
  # measured channel and the budgets it implies (Theorem recovery, same EPS)
  pe_m <- round(sm$pe$pooled, 3); Z_m <- 2 * sqrt(pe_m * (1 - pe_m))
  setv(key("pe"), f3(pe_m)); setv(key("Z"), f2(Z_m))
  for (k in KS) setv(key("budget.k", k),
                     f0(ceiling(2 * (k * log(2) + log(1 / EPS)) / (1 - Z_m))))
  # payloads where MOSAIC leads every baseline, significantly against the best,
  # and where some baseline is significantly ahead of MOSAIC
  best <- function(k) bl[which.max(vapply(bl, function(m) srate(m, k), 0))]
  lead <- KS[vapply(KS, function(k) srate("mosaic", k) > srate(best(k), k) &&
                      sp(best(k), k) < 0.05, TRUE)]
  behind <- KS[vapply(KS, function(k) any(vapply(bl, function(m)
    srate(m, k) > srate("mosaic", k) && sp(m, k) < 0.05, TRUE)), TRUE)]
  setv(key("lead.ks"), and_list(lead)); setv(key("behind.ks"), and_list(behind))
  setv(key("mosaic.drop"), f1(srate("mosaic", 16) - srate("mosaic", 48)))
  XM[[tag]] <- list(pe = sm$pe$pooled, m48 = srate("mosaic", 48), lead = lead, behind = behind,
                    gap = vapply(behind, function(k)
                      max(vapply(bl, function(m) srate(m, k), 0)) - srate("mosaic", k), 0))
  # the text names Janas et al. as the scheme ahead at small payloads
  stopifnot(all(vapply(behind, function(k) all(vapply(setdiff(bl, "janas"), function(m)
    !(srate(m, k) > srate("mosaic", k) && sp(m, k) < 0.05), TRUE)), TRUE)))
  setv(key("behind.max"), f1(max(vapply(behind, function(k)
    max(vapply(bl, function(m) srate(m, k), 0)) - srate("mosaic", k), 0))))
  # the text states that MOSAIC exceeds Qu et al. at every payload size
  if ("qu" %in% bl) {
    stopifnot(all(vapply(KS, function(k) srate("mosaic", k) > srate("qu", k) &&
                           sp("qu", k) < 0.05, TRUE)))
    setv(key("qu.pmax"), p_tex(max(smc$p_value[smc$baseline == "qu"])))
  }
  ag <- SMAG[[tag]]
  for (k in c(16, 48)) setv(key("medp.exp.k", k), floor(log10(ag$median_p_value[ag$k == k])))
  setv(key("bitacc.min"), f1(min(ag$bit_accuracy)))
  setv(key("ci.max"), f2(max((sr$ci_hi - sr$ci_lo)[sr$method == "mosaic"] / 2)))
  setv(key("ppl.base"), f1(unique(round(ag$mean_ppl_plain, 1))))
  setv(key("ppl.marked.min"), f1(min(ag$mean_ppl_watermarked)))
  setv(key("ppl.marked.max"), f1(max(ag$mean_ppl_watermarked)))
  rp <- sm$repetition
  for (i in seq_len(nrow(rp))) setv(key("rep.", rp$method[i]), f1(rp$share50[i]))
  # appendix table block: rates with intervals, then the paired tests
  cell <- function(m, k) { r <- sr[sr$method == m & sr$k == k, ]
    if (nrow(r) == 0) "NA" else sprintf("%.1f [%.1f, %.1f]", r$match_rate, r$ci_lo, r$ci_hi) }
  mcell <- function(m, k) { r <- smc[smc$baseline == m & smc$k == k, ]
    if (nrow(r) == 0) "NA" else sprintf("%d/%d, $%s$", r$mosaic_only, r$baseline_only, p_tex(r$p_value)) }
  sm_rows <- c(sm_rows, if (length(sm_rows)) "\\midrule",
    sprintf("\\multicolumn{6}{@{}l}{%s: exact match rate (\\%%) with 95\\%% bootstrap interval} \\\\",
            MLAB[[tag]]),
    vapply(meths, function(m) paste0(LAB[[m]], " & ", paste(vapply(KS, function(k) cell(m, k), ""),
                                                            collapse = " & "), " \\\\"), ""),
    "\\multicolumn{6}{@{}l}{McNemar test against MOSAIC: MOSAIC-only / baseline-only successes, exact $p$} \\\\",
    vapply(bl, function(m) paste0(LAB[[m]], " & ", paste(vapply(KS, function(k) mcell(m, k), ""),
                                                         collapse = " & "), " \\\\"), ""))
}
# across generation models: Llama-2 on the shared prompts against the others
pe_common <- mean(diag$pe_true[diag$prompt_index < N_COMMON])
setv("pe.common", f3(pe_common))
Z_common <- 2 * sqrt(round(pe_common, 3) * (1 - round(pe_common, 3)))
setv("budget.common.k48", f0(ceiling(2 * (48 * log(2) + log(1 / EPS)) / (1 - Z_common))))
llama_behind <- KS[vapply(KS, function(k) any(vapply(ROWS_BL, function(m) {
  v <- rate_of(m, k); p <- mcn$p_value[mcn$baseline == m & mcn$k == k]
  length(v) == 1 && length(p) == 1 && v > rate_of("mosaic", k) && p < 0.05
}, TRUE)), TRUE)]
XM[["llama"]] <- list(pe = pe_common, m48 = rate_of("mosaic", 48), lead = leads,
                      behind = llama_behind,
                      gap = vapply(llama_behind, function(k)
                        max(rates$match_rate[rates$k == k & rates$method %in% ROWS_BL]) -
                          rate_of("mosaic", k), 0))
lead_all <- Reduce(intersect, lapply(XM, `[[`, "lead"))
stopifnot(length(lead_all) > 0)                       # the text claims a common lead
setv("lead.all.ks", and_list(sort(lead_all)))
setv("nmodels", WORD[length(XM)])
behind_all <- unlist(lapply(XM, `[[`, "behind"))
setv("behind.all.kmax", max(behind_all))
setv("behind.all.max", f1(max(unlist(lapply(XM, `[[`, "gap")))))
# the text: on Llama-2 no baseline is significantly ahead, and Janas et al. are
# level with MOSAIC at 16 and 20 bits (the deployed detector)
stopifnot(length(llama_behind) == 0,
          all(mcn$p_value[mcn$baseline == "janas" & mcn$k %in% c(16, 20)] >= 0.05))
# spread of MOSAIC's 16-bit match rate across models, and the best 48-bit rate of
# Yoo et al. on any model (shared prompts)
m16 <- c(rate_of("mosaic", 16), vapply(SMD, function(d)
  d$rates$match_rate[d$rates$method == "mosaic" & d$rates$k == 16], 0))
setv("xm.k16.spread", f1(max(m16) - min(m16)))
setv("xm.mpac.k48.max", f1(max(c(rate_of("mpac", 48), unlist(lapply(SMD, function(d)
  d$rates$match_rate[d$rates$method == "mpac" & d$rates$k == 48]))))))
# the text states that MOSAIC exceeds Qu et al. at every payload size on Llama-2 too
stopifnot(all(vapply(KS, function(k) rate_of("mosaic", k) > rate_of("qu", k) &&
                       mcn$p_value[mcn$baseline == "qu" & mcn$k == k] < 0.05, TRUE)))
ppl_all <- c(main$ppl_mark, unlist(lapply(SMAG, `[[`, "mean_ppl_watermarked")))
setv("ppl.all.min", f1(min(ppl_all))); setv("ppl.all.max", f1(max(ppl_all)))
setv("bitacc.all.min", f1(min(c(main$bitacc, unlist(lapply(SMAG, `[[`, "bit_accuracy"))))))
setv("ci.sm.max", f2(max(unlist(lapply(SMD, function(d) {
  r <- d$rates[d$rates$method == "mosaic", ]; (r$ci_hi - r$ci_lo) / 2 })))))
setv("medp.all.max", max(c(floor(log10(main$medp)),
                           unlist(lapply(SMAG, function(a) floor(log10(a$median_p_value)))))))
if (all(c("mistral", "qwen") %in% names(XM))) {
  # the text orders the models Mistral, Llama-2, Qwen by flip rate and states
  # that the 48-bit match rate falls in that order
  pes <- vapply(XM[c("mistral", "llama", "qwen")], `[[`, 0, "pe")
  m48 <- vapply(XM[c("mistral", "llama", "qwen")], `[[`, 0, "m48")
  stopifnot(!is.unsorted(pes, strictly = TRUE), !is.unsorted(-m48, strictly = TRUE))
}

if (length(SECOND)) writetab(c(
  "\\begin{tabular}{@{}l ccccc@{}}", "\\toprule",
  "& \\multicolumn{5}{c}{Payload size $k$ (bits)} \\\\", "\\cmidrule(lr){2-6}",
  paste0("Method & ", paste(KS, collapse = " & "), " \\\\"), "\\midrule",
  sm_rows, "\\bottomrule", "\\end{tabular}"), "second_model.tex")

# =====================================================================
# 4. MOSAIC ROBUSTNESS ON THE 10% SUBSET  (figure + threshold check)
# =====================================================================
rob <- fromJSON(f_rob)$results
rob$k <- as.integer(rob$k)
N_SUB <- unique(rob$n); stopifnot(length(N_SUB) == 1)
setv("n.sub", N_SUB)
rv <- function(a, s, k) rob$match_rate[rob$attack == a & abs(rob$strength - s) < 1e-9 & rob$k == k]

# edit-attack curves, faceted by attack, coloured by payload
ed <- rob %>%
  mutate(frac = if_else(attack == "reordering", strength, 100 * strength),
         attack = factor(attack,
           levels = c("substitution", "insertion", "deletion", "reordering"),
           labels = c("Substitution\nEdited fraction of tokens (%)",
                      "Insertion\nEdited fraction of tokens (%)",
                      "Deletion\nEdited fraction of tokens (%)",
                      "Reordering\nNumber of reordered blocks"))) %>%
  filter(!is.na(attack))
p <- ggplot(ed, aes(frac, match_rate, colour = factor(k), group = k)) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.7) +
  facet_wrap(~attack, scales = "free_x") +
  scale_colour_manual(values = PAL, name = "Payload k") +
  labs(x = NULL, y = "Exact match rate (%)") + theme_paper
savefig("all_robustness_edits.pdf", p, 7.0, 7.0)

# empirical substitution rate at which the match rate crosses 50%, by linear
# interpolation between the evaluated strengths (clean run at strength 0)
cross50 <- function(k) {
  s <- c(0, 0.05, 0.10, 0.20, 0.30)
  m <- c(rv("clean", 0, k), vapply(s[-1], function(x) rv("substitution", x, k), 0))
  j <- which(m < 50)[1]; stopifnot(!is.na(j), j > 1)
  s[j - 1] + (m[j - 1] - 50) / (m[j - 1] - m[j]) * (s[j] - s[j - 1])
}
for (k in c(16, 32, 48)) setv(sprintf("emp50.k%d", k), f2(cross50(k)))
# empirical 50% crossing relative to the guaranteed substitution threshold
ratio50 <- vapply(c(16, 32, 48), function(k) cross50(k) / thresh(k), 0)
setv("emp50.ratio.min", f1(min(ratio50))); setv("emp50.ratio.max", f1(max(ratio50)))
for (k in KS) {
  setv(sprintf("rob.clean.k%d", k), f1(rv("clean", 0, k)))
  for (a in c("substitution", "insertion", "deletion"))
    for (s in c(0.05, 0.1, 0.2, 0.3))
      setv(sprintf("rob.%s.%d.k%d", substr(a, 1, 3), round(100 * s), k), f1(rv(a, s, k)))
  for (b in c(2, 5, 10)) setv(sprintf("rob.reo.%d.k%d", b, k), f1(rv("reordering", b, k)))
  setv(sprintf("rob.reo.min.k%d", k), f1(min(vapply(c(2, 5, 10), function(b) rv("reordering", b, k), 0))))
}

# ---- paraphrase attack (DIPPER), same subset ----
par <- fromJSON(f_par)$results
pv <- function(k, col) par[[col]][par$attack == "paraphrase" & par$k == k]
for (k in KS) setv(sprintf("para.match.k%d", k), f1(pv(k, "match_rate")))
setv("para.det.k16", f1(pv(16, "detection_rate")))
setv("para.bitacc.k16", f1(pv(16, "bit_accuracy")))

# =====================================================================
# 5. ROBUSTNESS COMPARISON ON THE SHARED PROMPTS  (figure + appendix table)
# =====================================================================
comp <- read.csv(f_comp)
cagg <- comp %>% group_by(method, k, attack, strength) %>%
  summarise(match = 100 * mean(match), n = n(), .groups = "drop")
stopifnot(all(cagg$n == N_COMMON))
COMP_K <- sort(unique(cagg$k))
setv("cohen.cmp.max", f1(max(cagg$match[cagg$method == "cohen"])))
cagg <- cagg[cagg$method %in% METH, ]
# Qu et al. and Cohen et al. assign payload parts by position in our
# reimplementations (Qu et al.'s release assigns segments by the previous token;
# Cohen et al.'s detector is oblivious to the allocation), so they enter the
# position-shifting attacks only through substitution, which keeps positions.
bl_config <- function(m) {                           # config block at the file head
  f <- f_base[match(m, BL_METHODS)]
  txt <- readChar(f, min(65536, file.info(f)$size), useBytes = TRUE)
  cfg <- regmatches(txt, regexpr('(?s)"config"\\s*:\\s*\\{.*?\\}', txt, perl = TRUE))
  fromJSON(paste0("{", cfg, "}"))$config
}
# a method enters the position-shifting attacks when its result file declares an
# implementation whose allocation matches the original scheme under edits; the
# earlier Qu et al. and Cohen et al. runs predate that field and are position keyed.
# The adapted Cohen et al. variant is reported as its own method with its own
# detector, so it enters every attack as it stands.
POS_ONLY_SUB <- Filter(function(m) !isTRUE(bl_config(m)$faithful_under_edits),
                       c("qu", "cohen"))
setv("cmp.posonly", paste(LAB[POS_ONLY_SUB], collapse = " and "))
cplot <- cagg %>%
  filter(attack != "clean", !(method %in% POS_ONLY_SUB & attack != "substitution")) %>%
  bind_rows(cagg %>% filter(attack == "clean") %>% select(-attack) %>%
              tidyr::crossing(attack = c("substitution", "insertion", "deletion", "reordering")) %>%
              filter(!(method %in% POS_ONLY_SUB & attack != "substitution")) %>%
              mutate(strength = if_else(attack == "reordering", 1, strength))) %>%
  mutate(x = if_else(attack == "reordering", strength, 100 * strength),
         Method = factor(LAB[method], levels = LAB[METH]),
         attack = factor(attack, levels = c("substitution", "insertion", "deletion", "reordering"),
                         labels = c("Substitution (%)", "Insertion (%)", "Deletion (%)",
                                    "Reordering (blocks)")),
         kf = factor(paste0("k = ", k), levels = paste0("k = ", COMP_K)))
p <- ggplot(cplot, aes(x, match, colour = Method, group = Method)) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.4) +
  facet_grid(kf ~ attack, scales = "free_x") +
  scale_x_continuous(breaks = function(l) if (l[2] < 12) c(1, 2, 5, 10) else c(0, 10, 20, 30)) +
  scale_colour_manual(values = MCOL, name = NULL) +
  scale_y_continuous(limits = c(0, 100), breaks = c(0, 50, 100)) +
  labs(x = NULL, y = "Exact match rate (%)") +
  theme_paper + guides(colour = guide_legend(nrow = 2))
savefig("robustness_comparison.pdf", p, 7.0, 5.2)

# appendix table: every method x attack x strength at each compared payload
cv <- function(m, k, a, s) {
  v <- cagg$match[cagg$method == m & cagg$k == k & cagg$attack == a & abs(cagg$strength - s) < 1e-9]
  if (length(v) == 0 || (m %in% POS_ONLY_SUB && !(a %in% c("clean", "substitution")))) "--" else f1(v)
}
SETTINGS <- list(c("clean", 0), c("substitution", 0.05), c("substitution", 0.1),
                 c("substitution", 0.2), c("substitution", 0.3), c("insertion", 0.05),
                 c("insertion", 0.1), c("insertion", 0.2), c("insertion", 0.3),
                 c("deletion", 0.05), c("deletion", 0.1), c("deletion", 0.2),
                 c("deletion", 0.3), c("reordering", 2), c("reordering", 5), c("reordering", 10))
SLAB <- c(clean = "Clean", substitution = "Sub.", insertion = "Ins.", deletion = "Del.",
          reordering = "Reord.")
comp_rows <- c()
for (k in COMP_K) {
  ms <- METH[METH %in% unique(cagg$method[cagg$k == k])]
  comp_rows <- c(comp_rows, sprintf("\\multicolumn{%d}{@{}l}{$k = %d$} \\\\", length(METH) + 1, k))
  for (st in SETTINGS) {
    s <- as.numeric(st[2])
    lab <- if (st[1] == "clean") "Clean" else if (st[1] == "reordering")
      sprintf("%s %d", SLAB[[st[1]]], as.integer(s)) else sprintf("%s %d\\%%", SLAB[[st[1]]], round(100 * s))
    comp_rows <- c(comp_rows, paste0(lab, " & ", paste(vapply(METH, function(m)
      if (m %in% ms) cv(m, k, st[1], s) else "NA", ""), collapse = " & "), " \\\\"))
  }
  if (k != max(COMP_K)) comp_rows <- c(comp_rows, "\\midrule")
}
writetab(c(
  sprintf("\\begin{tabular}{@{}l %s@{}}", paste(rep("r", length(METH)), collapse = "")),
  "\\toprule",
  paste0("Attack & ", paste(c("MOSAIC", "Qu", "Janas", "Fern.", "Wang", "Yoo", "Cohen ad."),
                            collapse = " & "), " \\\\"),
  "\\midrule", comp_rows, "\\bottomrule", "\\end{tabular}"), "robustness_comparison.tex")

# MOSAIC's decisions on the unedited texts, stored (produced on the generation
# server) against re-detected here: belief propagation runs in floating point,
# so a decision near the boundary can differ across processors and numerical
# libraries; the baselines count integer votes and reproduce exactly
stored  <- diag[diag$prompt_index < N_COMMON, c("prompt_index", "k", "match")]
reclean <- comp[comp$method == "mosaic" & comp$attack == "clean", c("prompt_index", "k", "match")]
mm <- merge(stored, reclean, by = c("prompt_index", "k"), suffixes = c(".stored", ".here"))
for (k in COMP_K) setv(sprintf("flip.k%d", k), sum(mm$k == k & mm$match.stored != mm$match.here))
stopifnot(all(mm$match.stored == mm$match.here))  # the text says they reproduce exactly
for (m in setdiff(METH, "mosaic")) {                   # baselines: must reproduce exactly
  st <- rates[rates$method == m, c("k", "match_rate")]
  here <- cagg[cagg$method == m & cagg$attack == "clean", c("k", "match")]
  j <- merge(st, here, by = "k"); stopifnot(all(abs(j$match_rate - j$match) < 1e-9))
}
exh <- cagg[cagg$method %in% c("cyclic", "wang") & cagg$k == 16 &
              cagg$attack %in% c("substitution", "insertion", "deletion") &
              abs(cagg$strength - 0.2) < 1e-9, ]
setv("cmp.exh.min.20.k16", f1(min(exh$match)))
s48 <- cagg[cagg$k == 48 & cagg$attack == "substitution" & abs(cagg$strength - 0.2) < 1e-9, ]
setv("cmp.max.sub.20.k48", f1(max(s48$match)))

for (m in METH) for (k in COMP_K) for (st in SETTINGS) {
  s <- as.numeric(st[2])
  v <- cagg$match[cagg$method == m & cagg$k == k & cagg$attack == st[1] & abs(cagg$strength - s) < 1e-9]
  if (length(v)) setv(sprintf("cmp.%s.%s.%s.k%d", m, substr(st[1], 1, 3),
                              if (st[1] == "reordering") as.integer(s) else round(100 * s), k), f1(v))
}

# =====================================================================
# 5b. ROBUSTNESS ON THE FURTHER MODELS  (cross-model figure + appendix tables)
# =====================================================================
RB_METH <- c("mosaic", "qu", "janas", "mpac")          # the methods run on every model
agg_rob <- function(f) read.csv(f) %>% group_by(method, k, attack, strength) %>%
  summarise(match = 100 * mean(match), n = n(), .groups = "drop")
CAGG <- c(list(llama = cagg), lapply(f_comp_sm, agg_rob))
for (tag in names(SECOND)) {
  ca <- CAGG[[tag]]; stopifnot(all(ca$n == N_COMMON))
  sr <- SMD[[tag]]$rates
  for (m in setdiff(unique(ca$method), "mosaic")) {   # baselines reproduce Table 2 exactly
    j <- merge(sr[sr$method == m, c("k", "match_rate")],
               ca[ca$method == m & ca$attack == "clean", c("k", "match")], by = "k")
    stopifnot(nrow(j) > 0, all(abs(j$match_rate - j$match) < 1e-9))
  }
  for (m in unique(ca$method)) for (k in COMP_K) for (st in SETTINGS) {
    v <- ca$match[ca$method == m & ca$k == k & ca$attack == st[1] &
                    abs(ca$strength - as.numeric(st[2])) < 1e-9]
    if (length(v)) setv(sprintf("cmp.%s.%s.%s.%s.k%d", tag, m, substr(st[1], 1, 3),
                                if (st[1] == "reordering") as.integer(as.numeric(st[2]))
                                else round(100 * as.numeric(st[2])), k), f1(v))
  }
  # appendix table: every attack setting at each payload for this model
  rows <- c()
  for (k in COMP_K) {
    rows <- c(rows, sprintf("\\multicolumn{%d}{@{}l}{$k = %d$} \\\\", length(RB_METH) + 1, k))
    for (st in SETTINGS) {
      sv <- as.numeric(st[2])
      lab <- if (st[1] == "clean") "Clean" else if (st[1] == "reordering")
        sprintf("%s %d", SLAB[[st[1]]], as.integer(sv)) else
          sprintf("%s %d\\%%", SLAB[[st[1]]], round(100 * sv))
      rows <- c(rows, paste0(lab, " & ", paste(vapply(RB_METH, function(m) {
        v <- ca$match[ca$method == m & ca$k == k & ca$attack == st[1] & abs(ca$strength - sv) < 1e-9]
        if (length(v)) f1(v) else "--" }, ""), collapse = " & "), " \\\\"))
    }
    if (k != max(COMP_K)) rows <- c(rows, "\\midrule")
  }
  writetab(c("\\begin{tabular}{@{}l rrrr@{}}", "\\toprule",
             "Attack & MOSAIC & Qu & Janas & Yoo \\\\", "\\midrule", rows,
             "\\bottomrule", "\\end{tabular}"), sprintf("robustness_%s.tex", tag))
}
# the robustness claims of the text, on every model: MOSAIC keeps more payloads
# than Qu et al. under light edits and reordering at every compared payload, Qu
# et al. keep more under 20% deletion at 32 bits, MOSAIC keeps most 32-bit
# payloads under 5% insertion and deletion and most 48-bit payloads under
# reordering into 10 blocks, and Janas et al. lose most 16-bit payloads under 5%
# insertion and deletion
LIGHT <- list(c("substitution", 0.05), c("substitution", 0.1), c("insertion", 0.05),
              c("insertion", 0.1), c("deletion", 0.05), c("deletion", 0.1),
              c("reordering", 2), c("reordering", 5), c("reordering", 10))
for (tag in names(CAGG)) {
  g <- function(m, k, a, s) { ca <- CAGG[[tag]]
    ca$match[ca$method == m & ca$k == k & ca$attack == a & abs(ca$strength - s) < 1e-9] }
  for (k in COMP_K) for (st in LIGHT)
    stopifnot(g("mosaic", k, st[1], as.numeric(st[2])) > g("qu", k, st[1], as.numeric(st[2])))
  stopifnot(g("qu", 32, "deletion", 0.2) > g("mosaic", 32, "deletion", 0.2),
            g("mosaic", 32, "insertion", 0.05) > 50, g("mosaic", 32, "deletion", 0.05) > 50,
            g("mosaic", 48, "reordering", 10) > 50,
            g("janas", 16, "insertion", 0.05) < 50, g("janas", 16, "deletion", 0.05) < 50)
}
# MOSAIC re-decoded on this machine against the stored decisions (floating-point
# belief propagation, as for Llama-2-7B above): largest gap in match rate
setv("rb.redetect.maxdiff", f1(max(vapply(names(SECOND), function(tag) {
  sr <- SMD[[tag]]$rates; sr <- sr[sr$method == "mosaic", c("k", "match_rate")]
  j <- merge(sr, CAGG[[tag]][CAGG[[tag]]$method == "mosaic" & CAGG[[tag]]$attack == "clean",
                             c("k", "match")], by = "k")
  max(abs(j$match_rate - j$match)) }, 0))))
stopifnot(NUM[["rb.redetect.maxdiff"]] == "0.0")       # the text: they reproduce exactly
# cross-model figure at 32 bits: the methods run on every model
RB_K <- 32
xr <- do.call(rbind, lapply(names(CAGG), function(tag) {
  ca <- CAGG[[tag]] %>% filter(method %in% RB_METH, k == RB_K)
  rbind(ca %>% filter(attack != "clean"),
        ca %>% filter(attack == "clean") %>% select(-attack) %>%
          tidyr::crossing(attack = c("substitution", "insertion", "deletion", "reordering")) %>%
          mutate(strength = if_else(attack == "reordering", 1, strength))) %>%
    mutate(model = MSHORT[[tag]])
}))
xr <- xr %>% mutate(x = if_else(attack == "reordering", strength, 100 * strength),
  Method = factor(LAB[method], levels = LAB[METH]),
  model = factor(model, levels = MSHORT[names(CAGG)]),
  attack = factor(attack, levels = c("substitution", "insertion", "deletion", "reordering"),
                  labels = c("Substitution (%)", "Insertion (%)", "Deletion (%)", "Reordering (blocks)")))
p <- ggplot(xr, aes(x, match, colour = Method, group = Method)) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.4) +
  facet_grid(model ~ attack, scales = "free_x") +
  scale_x_continuous(breaks = function(l) if (l[2] < 12) c(1, 2, 5, 10) else c(0, 10, 20, 30)) +
  scale_colour_manual(values = MCOL, name = NULL) +
  scale_y_continuous(limits = c(0, 100), breaks = c(0, 50, 100)) +
  labs(x = NULL, y = sprintf("Exact match rate at %d bits (%%)", RB_K)) + theme_paper
savefig("robustness_models.pdf", p, 7.0, 5.0)
setv("rb.k", RB_K)

# =====================================================================
# 6. DELTA SWEEP  (quality / robustness trade-off, k = 16 and k = 32)
# =====================================================================
sd <- read.csv(f_sdelta)                       # summarise_sweeps.py output
drows <- data.frame(delta = sd$delta, k = as.integer(sd$k),
                    match = sd$match_rate, ppl = sd$mean_ppl)
long <- rbind(
  data.frame(delta = drows$delta, k = factor(drows$k), value = drows$match,
             Metric = "Exact match rate (%)"),
  data.frame(delta = drows$delta, k = factor(drows$k), value = drows$ppl,
             Metric = "Perplexity (Llama-3.1 oracle)"))
p <- ggplot(long, aes(delta, value, colour = k, group = k)) +
  geom_line(linewidth = 0.7) + geom_point(size = 2) +
  facet_wrap(~Metric, scales = "free_y") +
  geom_vline(xintercept = DELTA, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = PAL, name = "Payload k") +
  scale_x_continuous(breaks = sort(unique(drows$delta))) +
  labs(x = expression("Logit bias "*delta), y = NULL) + theme_paper
savefig("delta_tradeoff.pdf", p, 7.0, 3.2)
dv <- function(d, k, col) drows[[col]][drows$delta == d & drows$k == k]
for (d in range(drows$delta)) for (k in unique(drows$k))
  setv(sprintf("dl.match.d%d.k%d", d, k), f0(dv(d, k, "match")))
for (k in unique(drows$k)) setv(sprintf("dl.match.d%d.k%d", DELTA, k), f0(dv(DELTA, k, "match")))
setv("dl.min", min(drows$delta)); setv("dl.max", max(drows$delta))
setv("dl.ppl.min", f1(mean(drows$ppl[drows$delta == min(drows$delta)])))
setv("dl.ppl.max", f0(mean(drows$ppl[drows$delta == max(drows$delta)])))
# a lower-distortion operating point
DLOW <- 4
for (k in unique(drows$k)) setv(sprintf("dl.match.d%d.k%d", DLOW, k), f1(dv(DLOW, k, "match")))
for (d in c(DLOW, DELTA)) setv(sprintf("dl.ppl.d%d", d), f1(mean(drows$ppl[drows$delta == d])))

# =====================================================================
# 7. TOKEN BUDGET HEATMAP  (match rate by payload size x token length)
# =====================================================================
hm <- read.csv(f_stok)                        # summarise_sweeps.py output
hm_ag <- aggregate(match ~ k + T, hm, function(z) 100 * mean(z))
tv <- function(k, t) hm_ag$match[hm_ag$k == k & hm_ag$T == t]
TS <- sort(unique(hm_ag$T))
for (k in KS) for (t in TS) setv(sprintf("tok.k%d.T%d", k, t), f0(tv(k, t)))
setv("tok.Tmin", min(TS)); setv("tok.Tmax", max(TS))
hm_ag$k <- factor(hm_ag$k, levels = KS)
hm_ag$T <- factor(hm_ag$T, levels = TS)
p <- ggplot(hm_ag, aes(T, k, fill = match)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.0f", match), colour = match > 55),
            size = 3.1, show.legend = FALSE) +
  scale_colour_manual(values = c("TRUE" = "white", "FALSE" = "black")) +
  scale_fill_viridis_c(name = "Match (%)", limits = c(40, 100), option = "D") +
  labs(x = "Generated tokens T", y = "Payload size k (bits)") +
  theme_paper + theme(panel.grid = element_blank())
savefig("tokens_heatmap.pdf", p, 5.2, 3.2)

# =====================================================================
# 8. LOW-PAYLOAD FAILURE ANALYSIS  (repetition, coverage, decoder)
# =====================================================================
diag$dup <- 1 - diag$distinct_windows / diag$T      # share of repeated windows
DUP_BINS <- c(-Inf, 0.1, 0.2, 0.3, 0.5, Inf)
DUP_LAB  <- c("\\le 10", "10\\text{--}20", "20\\text{--}30", "30\\text{--}50", "> 50")
diag$bin <- cut(diag$dup, DUP_BINS, labels = FALSE)
LOWK <- c(16, 20)
for (k in KS) {
  s <- diag[diag$k == k, ]; f <- s[s$match == 0, ]; ok <- s[s$match == 1, ]
  setv(sprintf("low.fail.k%d", k), nrow(f))
  setv(sprintf("low.dup.fail.k%d", k), f2(median(f$dup)))
  setv(sprintf("low.dup.succ.k%d", k), f2(median(ok$dup)))
  setv(sprintf("low.pe.fail.k%d", k), f3(median(f$pe_true)))
  setv(sprintf("low.pe.succ.k%d", k), f3(median(ok$pe_true)))
  setv(sprintf("low.cov.fail.k%d", k), f0(median(f$min_bit_cover)))
  setv(sprintf("low.cov.succ.k%d", k), f0(median(ok$min_bit_cover)))
  setv(sprintf("low.rep.fail.k%d", k), f0(median(f$max_window_repeat)))
  setv(sprintf("low.rep.succ.k%d", k), f0(median(ok$max_window_repeat)))
  setv(sprintf("low.faildup20.k%d", k), sum(f$dup > 0.2))
  setv(sprintf("low.share50.k%d", k), f1(100 * mean(s$dup > 0.5)))
  for (b in seq_along(DUP_LAB)) {
    sb <- s[s$bin == b, ]
    setv(sprintf("low.bin%d.rate.k%d", b, k), f1(100 * (1 - mean(sb$match))))
    setv(sprintf("low.bin%d.n.k%d", b, k), big(nrow(sb)))
  }
}

# failure rate by share of repeated windows, all payloads
fb <- diag %>% mutate(bin = factor(bin, levels = seq_along(DUP_LAB),
                                   labels = c("<=10", "10-20", "20-30", "30-50", ">50"))) %>%
  group_by(k, bin) %>% summarise(fail = 100 * (1 - mean(match)), n = n(), .groups = "drop")
p <- ggplot(fb, aes(bin, fail, colour = factor(k), group = k)) +
  geom_line(linewidth = 0.7) + geom_point(size = 2) +
  scale_colour_manual(values = PAL, name = "Payload k") +
  labs(x = "Repeated sorted windows in the text (%)", y = "Decoding failures (%)") +
  theme_paper
savefig("failure_vs_repetition.pdf", p, 4.3, 3.2)

# repetition of each method's own generations, by the keying of its bias
rep <- read.csv(f_rep)
rep$dup <- 1 - rep$distinct_windows / rep$T
mos16 <- diag[diag$k == 16 & diag$prompt_index < N_COMMON, ]
repk <- rbind(rep[rep$k == 16, c("method", "dup", "match")],
              data.frame(method = "mosaic", dup = mos16$dup, match = mos16$match))
rs <- repk %>% group_by(method) %>%
  summarise(share50 = 100 * mean(dup > 0.5), n50 = sum(dup > 0.5),
            match50 = if (sum(dup > 0.5) > 0) 100 * mean(match[dup > 0.5]) else NA_real_,
            med = median(dup), .groups = "drop")
for (i in seq_len(nrow(rs))) {
  setv(sprintf("rep.%s.share50", rs$method[i]), f1(rs$share50[i]))
  setv(sprintf("rep.%s.n50", rs$method[i]), rs$n50[i])
  if (!is.na(rs$match50[i])) setv(sprintf("rep.%s.match50", rs$method[i]), f0(rs$match50[i]))
}
# What the favoured (boosted) set of a token depends on: a favoured set fixed by
# the context is the same at every repetition of that context and keeps favouring
# a repeated continuation; one that also changes with the position or with a
# random per-block choice differs from one repetition to the next. Qu et al. is
# classified from its result file: the released method keys segment and green
# list on the previous token, and the earlier runs used a position-keyed
# schedule; the adapted Cohen et al. instantiation keeps its keyed position
# schedule.
K_WIN  <- "Sorted window (w = 2)"
K_PREV <- "Previous token"
K_POS  <- "Previous token and position"
K_BLK  <- "Previous token and a random bit per block"
KEYING <- c(mosaic = K_WIN, cyclic = K_PREV, wang = K_PREV, mpac = K_PREV, janas = K_POS,
            qu = if (isTRUE(bl_config("qu")$faithful_under_edits)) K_PREV else K_POS,
            cohen_adapted = K_POS)
KLEV <- intersect(c(K_WIN, K_PREV, K_POS, K_BLK), KEYING)
KCOL <- c("#000000", "#4daf4a", "#e41a1c", "#377eb8")[match(KLEV, c(K_WIN, K_PREV, K_POS, K_BLK))]
prev_m  <- names(KEYING)[KEYING == K_PREV]
other_m <- names(KEYING)[!(KEYING %in% c(K_WIN, K_PREV))]
share_of <- function(ms) rs$share50[rs$method %in% ms]
setv("rep.prev.min", f1(min(share_of(prev_m))))
setv("rep.prev.max", f1(max(share_of(prev_m))))
setv("rep.prev.n", WORD[length(prev_m)])
setv("rep.pos.max", f1(max(share_of(other_m))))
setv("rep.pos.n", WORD[length(other_m)])
rs <- rs[rs$method %in% names(KEYING), ]
rs$Keying <- factor(KEYING[rs$method], levels = KLEV)
ORDER <- c("mosaic", names(KEYING)[KEYING == K_PREV], names(KEYING)[!(KEYING %in% c(K_WIN, K_PREV))])
rs$Method <- factor(LAB[rs$method], levels = LAB[ORDER])
p <- ggplot(rs, aes(Method, share50, fill = Keying)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.1f", share50)), vjust = -0.4, size = 3) +
  # room on the left for the first slanted label
  scale_x_discrete(expand = expansion(add = c(1.3, 0.6))) +
  scale_fill_manual(values = setNames(KCOL, KLEV), name = "Favoured set keyed on") +
  expand_limits(y = max(rs$share50) * 1.15) +
  labs(x = NULL, y = "Texts with > 50% repeated windows (%)") +
  theme_paper + theme(axis.text.x = element_text(angle = 30, hjust = 1),
                      legend.position = "top", legend.justification = "left",
                      plot.margin = margin(4, 8, 4, 22)) +
  guides(fill = guide_legend(ncol = 1, title.position = "top"))
savefig("repetition_by_method.pdf", p, 4.3, 4.2)

# the deployed decoder (distinct observations) against counting every position
dd <- read.csv(f_dist)
for (k in KS) {
  s <- dd[dd$k == k, ]
  setv(sprintf("dd.match.k%d", k), f1(100 * mean(s$match)))
  setv(sprintf("dd.all.k%d", k), f1(100 * mean(s$match_all)))
  setv(sprintf("dd.recover.k%d", k), sum(s$match_all == 0 & s$match == 1))
  setv(sprintf("dd.break.k%d", k), sum(s$match_all == 1 & s$match == 0))
}
setv("dd.ks", and_list(sort(unique(dd$k))))
setv("dd.kept.min", f1(100 * min(tapply(dd$num_distinct / dd$num_tokens, dd$k, mean))))
dd_rows <- vapply(KS, function(k) {
  s <- dd[dd$k == k, ]
  sprintf("%d & %.1f & %.1f & %d & %d & %d \\\\", k, 100 * mean(s$match_all),
          100 * mean(s$match), sum(s$match_all == 0),
          sum(s$match_all == 0 & s$match == 1), sum(s$match_all == 1 & s$match == 0))
}, "")
writetab(c(
  "\\begin{tabular}{@{}r rr rrr@{}}", "\\toprule",
  "& \\multicolumn{2}{c}{Match (\\%)} & & \\multicolumn{2}{c}{Changed outcomes} \\\\",
  "\\cmidrule(lr){2-3}\\cmidrule(l){5-6}",
  "$k$ & Every position & Distinct & Failures & Recovered & Lost \\\\", "\\midrule",
  dd_rows, "\\bottomrule", "\\end{tabular}"), "dedup_decoding.tex")

# false positives: unwatermarked generations and human continuations
nd <- do.call(rbind, lapply(names(f_null), function(tag) cbind(model = tag, read.csv(f_null[[tag]]))))
nd_g <- nd %>% group_by(model, source) %>%
  summarise(n = n(), fp = sum(p_value < 0.01), fp3 = sum(p_value < 0.001), minp = min(p_value),
            fpa = sum(p_value_all < 0.01), minpa = min(p_value_all), .groups = "drop")
setv("null.n", format(nrow(nd), big.mark = ","))
setv("null.nper", format(min(nd_g$n), big.mark = ","))
setv("null.fp", sum(nd$p_value < 0.01)); setv("null.fp3", sum(nd$p_value < 0.001))
setv("null.minp", p_tex(min(nd$p_value)))
setv("null.all.fp", sum(nd$p_value_all < 0.01)); setv("null.all.minp", p_tex(min(nd$p_value_all)))
SRC <- c(plain_text = "Unwatermarked", reference = "Human")
writetab(c("\\begin{tabular}{@{}ll r rr rr@{}}", "\\toprule",
  "& & & \\multicolumn{2}{c}{Distinct (deployed)} & \\multicolumn{2}{c}{Every position} \\\\",
  "\\cmidrule(lr){4-5}\\cmidrule(l){6-7}",
  "Model & Text & $n$ & $p < 0.01$ & min $p$ & $p < 0.01$ & min $p$ \\\\", "\\midrule",
  vapply(seq_len(nrow(nd_g)), function(i) with(nd_g[i, ], sprintf(
    "%s & %s & %s & %d & $%s$ & %d & $%s$ \\\\", MSHORT[[model]], SRC[[source]],
    format(n, big.mark = ","), fp, p_tex(minp), fpa, p_tex(minpa))), ""),
  "\\bottomrule", "\\end{tabular}"), "null_detection.tex")

# failure diagnostics table (low payloads + the largest payload for contrast)
diag_rows <- vapply(c(16, 20, 48), function(k) {
  s <- diag[diag$k == k, ]; f <- s[s$match == 0, ]; ok <- s[s$match == 1, ]
  sprintf("%d & %d & %.2f / %.2f & %.3f / %.3f & %d / %d & %d / %d \\\\", k, nrow(f),
          median(f$dup), median(ok$dup), median(f$pe_true), median(ok$pe_true),
          as.integer(median(f$max_window_repeat)), as.integer(median(ok$max_window_repeat)),
          as.integer(median(f$min_bit_cover)), as.integer(median(ok$min_bit_cover)))
}, "")
writetab(c(
  "\\begin{tabular}{@{}r r c c c c@{}}", "\\toprule",
  "$k$ & Failures & Repeated windows & $\\hat p_e$ & Largest repeat & Fewest checks per bit \\\\",
  "\\midrule", diag_rows, "\\bottomrule", "\\end{tabular}"), "failure_diagnostics.tex")

# =====================================================================
# 9. NUMBERS QUOTED IN THE TEXT
# =====================================================================
bad <- names(NUM)[lengths(NUM) != 1]
if (length(bad)) stop("numbers without exactly one value: ", paste(bad, collapse = ", "))
writeLines(c("% Generated by results/make_paper_assets.R from the result files. Do not edit.",
             sprintf("\\expandafter\\def\\csname mw@%s\\endcsname{%s}", names(NUM), unlist(NUM))),
           file.path(TAB, "numbers.tex"))
cat(sprintf("  numbers: numbers.tex (%d values)\n", length(NUM)))
cat("done. assets written to", PAPER, "\n")
