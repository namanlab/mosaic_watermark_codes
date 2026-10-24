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
#     dedup_decoding.py         -> derived/mosaic_dedup_decoding.csv
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
f_main <- file.path(RES, paste0("opengen_main_", MODEL, ".json"))
f_ppl  <- file.path(RES, paste0("opengen_main_", MODEL, "_Perplexity.json"))
f_rob  <- file.path(RES, paste0("opengen_robustness_", MODEL, ".json"))
f_par  <- file.path(RES, paste0("opengen_paraphrase_robustness_", MODEL, ".json"))
f_dlt  <- file.path(RES, "sweep_data_delta.json")
f_tok  <- file.path(RES, "sweep_data_tokens.json")
stopifnot(all(file.exists(c(f_main, f_ppl, f_rob, f_par, f_dlt, f_tok))))

# ---- Python for the helper scripts: the project venv when present ----
PY <- local({
  v <- file.path(RES, "..", "venv", "bin", "python")
  if (file.exists(v)) normalizePath(v) else {
    p <- Sys.which("python3"); if (p == "") p <- Sys.which("python"); p
  }
})
# Run a helper script when any of its outputs is missing or older than one of
# its inputs. deps maps each output to the result files it is computed from; a
# stale output is removed first, since each helper recomputes only missing
# outputs. keep_stale leaves stale outputs in place for helpers that update
# themselves incrementally (robustness_comparison.py recomputes only the
# methods whose result files changed).
run_helper <- function(script, deps, keep_stale = FALSE) {
  stale <- vapply(names(deps), function(o) !file.exists(o) ||
                    any(file.info(deps[[o]])$mtime > file.info(o)$mtime), TRUE)
  if (!any(stale)) return(invisible(TRUE))
  if (!keep_stale) invisible(file.remove(names(deps)[stale & file.exists(names(deps))]))
  cat("  running", script, "for", paste(basename(names(deps)[stale]), collapse = ", "), "...\n")
  status <- system2(PY, shQuote(file.path(RES, script)))
  stopifnot(status == 0, all(file.exists(names(deps))))
}
f_diag   <- file.path(DER, "mosaic_sample_diagnostics.csv")
f_rep    <- file.path(DER, "baseline_sample_repetition.csv")
f_common <- file.path(DER, "common_sample.json")
f_comp   <- file.path(DER, "robustness_comparison.csv")
f_dedup  <- file.path(DER, "mosaic_dedup_decoding.csv")
f_sdelta <- file.path(DER, "sweep_delta_summary.csv")
f_stok   <- file.path(DER, "sweep_tokens_records.csv")
BL_METHODS <- c("mpac", "qu", "cyclic", "wang", "cohen", "janas")
f_base <- file.path(RES, "other_models",
                    sprintf("opengen_baseline_%s_%s.json", BL_METHODS, MODEL))
stopifnot(all(file.exists(f_base)))
run_helper("summarise_sweeps.py", setNames(list(f_dlt, f_tok), c(f_sdelta, f_stok)))
run_helper("sample_diagnostics.py",
           setNames(list(f_main, f_base, c(f_main, f_base)), c(f_diag, f_rep, f_common)))
run_helper("robustness_comparison.py", setNames(list(c(f_main, f_base)), f_comp),
           keep_stale = TRUE)
run_helper("dedup_decoding.py", setNames(list(f_main), f_dedup))

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
big <- function(n) gsub(",", "{,}", format(n, big.mark = ",", trim = TRUE))

# ---- registry of every number the text quotes (written to numbers.tex) ----
NUM <- list()
setv <- function(key, value) { NUM[[key]] <<- as.character(value); invisible(value) }

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
# (how a token is assigned to a payload part in our implementation).
METH <- c("mosaic", "qu", "janas", "cyclic", "wang", "mpac", "cohen")
LAB <- c(mosaic = "MOSAIC (ours)", qu = "Qu et al.", janas = "Janas et al.",
         cyclic = "Fernandez et al.", wang = "Wang et al.", mpac = "Yoo et al.",
         cohen = "Cohen et al.")
MCOL <- c("MOSAIC (ours)" = "#000000", "Qu et al." = "#e41a1c", "Janas et al." = "#377eb8",
          "Fernandez et al." = "#4daf4a", "Wang et al." = "#984ea3",
          "Yoo et al." = "#ff7f00", "Cohen et al." = "#a65628")

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

rows <- vapply(seq_len(nrow(main)), function(i) with(main[i, ], sprintf(
  "%d & %.1f & %.2f & %.1f & %s & %.1f \\\\", as.integer(k), match,
  ci_half[as.character(k)], bitacc, sci_tex(medp), ppl_mark)), "")
writetab(c(
  "\\begin{tabular}{@{}r r r r c r@{}}", "\\toprule",
  "Payload $k$ & Match & $\\pm$95\\% & Bit acc. & Median $p$ & PPL \\\\",
  " (bits) & (\\%) & (\\%) & (\\%) & & \\\\", "\\midrule",
  rows, "\\bottomrule", "\\end{tabular}"), "main_results.tex")

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
thresh <- function(k) (1 - budget(k) / T0) / (1 + W)
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
ROWS_BL <- c("cyclic", "wang", "mpac", "cohen", "qu", "janas")
bl_line <- function(m, lab) paste0(lab, " & ", paste(vapply(KS, function(k) cell_rate(m, k), ""),
                                                  collapse = " & "), " \\\\")
writetab(c(
  "\\begin{tabular}{@{}l rrrrr@{}}", "\\toprule",
  "Method & \\multicolumn{5}{c}{Payload size $k$ (bits)} \\\\", "\\cmidrule(lr){2-6}",
  paste0(" & ", paste(KS, collapse = " & "), " \\\\"), "\\midrule",
  vapply(ROWS_BL, function(m) bl_line(m, LAB[[m]]), ""), "\\midrule",
  bl_line("mosaic", "MOSAIC (ours)"), "\\bottomrule", "\\end{tabular}"),
  "baseline_reproduced.tex")

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
setv("cs.mosaic.fail.k16", round(N_COMMON * (1 - rate_of("mosaic", 16) / 100)))
setv("cs.mosaic.drop", f1(rate_of("mosaic", 16) - rate_of("mosaic", 48)))
# gap between the most accurate baseline and MOSAIC where a baseline leads
gaps <- vapply(KS, function(k) {
  b <- rates$match_rate[rates$k == k & rates$method != "mosaic"]
  max(b) - rate_of("mosaic", k)
}, 0)
setv("gap.min", f1(min(gaps[gaps > 0]))); setv("gap.max", f1(max(gaps[gaps > 0])))
setv("gap.kmax", max(KS[gaps > 0]))

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
# earlier Qu et al. and Cohen et al. runs predate that field and are position keyed
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
  paste0("Attack & ", paste(c("MOSAIC", "Qu", "Janas", "Fern.", "Wang", "Yoo", "Cohen"),
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
share_of <- function(ms) rs$share50[rs$method %in% ms]
setv("rep.prev.min", f1(min(share_of(c("cyclic", "wang", "mpac")))))
setv("rep.prev.max", f1(max(share_of(c("cyclic", "wang", "mpac")))))
setv("rep.pos.max", f1(max(share_of(c("qu", "janas", "cohen")))))
# what the favoured (boosted) set of a token depends on: a context whose favoured
# set is the same at every position keeps favouring a repeated continuation
KEYING <- c(mosaic = "Sorted window (w = 2)", cyclic = "Previous token", wang = "Previous token",
            mpac = "Previous token", qu = "Previous token and position",
            janas = "Previous token and position", cohen = "Previous token and position")
KLEV <- c("Sorted window (w = 2)", "Previous token", "Previous token and position")
rs$Keying <- factor(KEYING[rs$method], levels = KLEV)
ORDER <- c("mosaic", "cyclic", "wang", "mpac", "qu", "janas", "cohen")
rs$Method <- factor(LAB[rs$method], levels = LAB[ORDER])
p <- ggplot(rs, aes(Method, share50, fill = Keying)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.1f", share50)), vjust = -0.4, size = 3) +
  scale_fill_manual(values = setNames(c("#000000", "#4daf4a", "#e41a1c"), KLEV),
                    name = "Favoured set keyed on") +
  expand_limits(y = max(rs$share50) * 1.15) +
  labs(x = NULL, y = "Texts with > 50% repeated windows (%)") +
  theme_paper + theme(axis.text.x = element_text(angle = 30, hjust = 1),
                      legend.position = "top", legend.justification = "left",
                      plot.margin = margin(4, 8, 4, 22)) +
  guides(fill = guide_legend(ncol = 1, title.position = "top"))
savefig("repetition_by_method.pdf", p, 4.3, 4.2)

# decoder with repeated observations counted once
dd <- read.csv(f_dedup)
DDK <- sort(unique(dd$k))                             # payloads re-decoded (16 and 20)
setv("dd.ks", paste(DDK, collapse = " and "))
for (k in DDK) {
  s <- dd[dd$k == k, ]
  setv(sprintf("dd.match.k%d", k), f1(100 * mean(s$match_dedup)))
  setv(sprintf("dd.recover.k%d", k), sum(s$match == 0 & s$match_dedup == 1))
  setv(sprintf("dd.break.k%d", k), sum(s$match == 1 & s$match_dedup == 0))
}
dd_rows <- vapply(DDK, function(k) {
  s <- dd[dd$k == k, ]
  sprintf("%d & %.1f & %.1f & %d & %d & %d \\\\", k, 100 * mean(s$match),
          100 * mean(s$match_dedup), sum(s$match == 0),
          sum(s$match == 0 & s$match_dedup == 1), sum(s$match == 1 & s$match_dedup == 0))
}, "")
writetab(c(
  "\\begin{tabular}{@{}r rr rrr@{}}", "\\toprule",
  "& \\multicolumn{2}{c}{Match (\\%)} & & \\multicolumn{2}{c}{Changed outcomes} \\\\",
  "\\cmidrule(lr){2-3}\\cmidrule(l){5-6}",
  "$k$ & All checks & Distinct & Failures & Recovered & Lost \\\\", "\\midrule",
  dd_rows, "\\bottomrule", "\\end{tabular}"), "dedup_decoding.tex")

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
writeLines(c("% Generated by results/make_paper_assets.R from the result files. Do not edit.",
             sprintf("\\expandafter\\def\\csname mw@%s\\endcsname{%s}", names(NUM), unlist(NUM))),
           file.path(TAB, "numbers.tex"))
cat(sprintf("  numbers: numbers.tex (%d values)\n", length(NUM)))
cat("done. assets written to", PAPER, "\n")
