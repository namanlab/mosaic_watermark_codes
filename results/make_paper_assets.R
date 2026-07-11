#!/usr/bin/env Rscript
# make_paper_assets.R
# -------------------
# Regenerates every table (paper/tables/*.tex) and figure (paper/figures/*.pdf)
# in the MOSAIC paper from the Llama-2-7B + top-p sampling result JSONs that live
# in this results/ folder. Old tables and figures are deleted first so the paper
# always reflects the current results.
#
#   Inputs (this folder):
#     opengen_main_meta-llama_Llama-2-7b-hf.json             (all 7711 prompts)
#     opengen_main_meta-llama_Llama-2-7b-hf_Perplexity.json  (all 7711 prompts)
#     opengen_robustness_meta-llama_Llama-2-7b-hf.json       (10% subset)
#     sweep_data_delta.json                                  (10% subset)
#     sweep_data_tokens.json                                 (10% subset)
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
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

# The AAAI paper (single-column, no inline citations in the comparison table)
# reuses the same generated tables. We write an AAAI-formatted copy of every
# table it uses into aaai/tables/ when that folder is present, so the AAAI paper
# is never hand-edited: it is the single source of truth here. ATAB stays NULL
# when the aaai/ tree is absent (e.g. the standalone code release).
AAAI <- file.path(RES, "..", "aaai")
ATAB <- if (dir.exists(AAAI)) {
  d <- file.path(normalizePath(AAAI), "tables")
  dir.create(d, showWarnings = FALSE, recursive = TRUE); d
} else NULL

MODEL <- "meta-llama_Llama-2-7b-hf"
f_main <- file.path(RES, paste0("opengen_main_", MODEL, ".json"))
f_ppl  <- file.path(RES, paste0("opengen_main_", MODEL, "_Perplexity.json"))
f_rob  <- file.path(RES, paste0("opengen_robustness_", MODEL, ".json"))
f_dlt  <- file.path(RES, "sweep_data_delta.json")
f_tok  <- file.path(RES, "sweep_data_tokens.json")
stopifnot(all(file.exists(c(f_main, f_ppl, f_rob, f_dlt, f_tok))))

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
  txt <- readChar(path, file.info(path)$size, useBytes = TRUE)
  m <- regmatches(txt, regexpr('(?s)"aggregate"\\s*:\\s*\\[.*?\\]', txt, perl = TRUE))
  stopifnot(length(m) == 1)
  fromJSON(paste0("{", m, "}"))$aggregate
}

# Nonparametric bootstrap 95% CI half-widths (in %) for the exact-match rate,
# resampling over prompts. The per-prompt "match" flags live in the (multi-
# hundred-MB) samples array; streaming them is delegated to compute_match_ci.py
# (R's regex engine mis-handles subject strings this large), which writes a small
# JSON that we read back. The helper is rerun whenever its output is missing or
# older than the main results file, so a single `Rscript make_paper_assets.R`
# still reproduces the CI column end to end. Returns a vector keyed by payload.
bootstrap_match_ci <- function(path, ks) {
  ci_json <- file.path(RES, "main_bootstrap_ci.json")
  helper  <- file.path(RES, "compute_match_ci.py")
  stale <- !file.exists(ci_json) ||
    file.info(ci_json)$mtime < file.info(path)$mtime
  if (stale) {
    py <- Sys.which("python3"); if (py == "") py <- Sys.which("python")
    stopifnot(nzchar(py), file.exists(helper))
    status <- system2(py, c(shQuote(helper), shQuote(path), shQuote(ci_json)))
    stopifnot(status == 0, file.exists(ci_json))
  }
  ci <- fromJSON(ci_json)
  vapply(ks, function(k) ci[[as.character(k)]]$half, 0) |> setNames(as.character(ks))
}

sci_tex <- function(x) {                       # 2.66e-37 -> "$2.7\times10^{-37}$"
  e <- floor(log10(x)); mant <- x / 10^e
  sprintf("$%.1f\\times10^{%d}$", mant, e)
}

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

# =====================================================================
# 1. MAIN RESULTS  (table + match-vs-bits + detection-strength figures)
# =====================================================================
ag_main <- read_aggregate(f_main)
ag_ppl  <- read_aggregate(f_ppl)
main <- data.frame(
  k = ag_main$k, match = ag_main$match_rate, bitacc = ag_main$bit_accuracy,
  medp = ag_main$median_p_value,
  ppl_base = ag_ppl$mean_ppl_plain[match(ag_main$k, ag_ppl$k)],
  ppl_mark = ag_ppl$mean_ppl_watermarked[match(ag_main$k, ag_ppl$k)],
  dppl = ag_ppl$ppl_increase_pct[match(ag_main$k, ag_ppl$k)])
main <- main[order(main$k), ]
 
# # --- main results table ---
# rows <- apply(main, 1, function(r) sprintf(
#   "%d & %.1f & %.1f & %s & %.1f & %.1f & %.0f \\\\",
#   as.integer(r["k"]), r["match"], r["bitacc"], sci_tex(r["medp"]),
#   r["ppl_base"], r["ppl_mark"], r["dppl"]))
# tab <- c(
#   "\\begin{tabular}{r r r c r r r}", "\\toprule",
#   "Payload $k$ & Match & Bit acc. & Median $p$ & PPL base & PPL marked & $\\Delta$PPL \\\\",
#   " (bits) & (\\%) & (\\%) & & & & (\\%) \\\\", "\\midrule",
#   rows, "\\bottomrule", "\\end{tabular}")
# writeLines(tab, file.path(TAB, "main_results.tex"))
# cat("  table: main_results.tex\n")

# --- main results table (with bootstrap CI half-width over prompts) ---
ci_half <- bootstrap_match_ci(f_main, main$k)
rows <- apply(main, 1, function(r) sprintf(
  "%d & %.1f & %.1f & %.1f & %s  \\\\",
  as.integer(r["k"]), r["match"], ci_half[as.character(as.integer(r["k"]))],
  r["bitacc"], sci_tex(r["medp"])))
tab <- c(
  "\\begin{tabular}{r r r r c}", "\\toprule",
  "Payload $k$ & Match & $\\pm$95\\% & Bit acc. & Median $p$ \\\\",
  " (bits) & (\\%) & (\\%) & (\\%) & \\\\", "\\midrule",
  rows, "\\bottomrule", "\\end{tabular}")
writeLines(tab, file.path(TAB, "main_results.tex"))
if (!is.null(ATAB)) writeLines(tab, file.path(ATAB, "main_results.tex"))
cat("  table: main_results.tex\n")

# --- figure: match rate and bit accuracy vs payload size ---
dfm <- rbind(
  data.frame(k = main$k, value = main$match,  Metric = "Exact match rate"),
  data.frame(k = main$k, value = main$bitacc, Metric = "Bit accuracy"))
p <- ggplot(dfm, aes(k, value, colour = Metric, shape = Metric)) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.4) +
  scale_colour_manual(values = c("Exact match rate" = "#d95f02", "Bit accuracy" = "#1b9e77")) +
  scale_x_continuous(breaks = KS) +
  scale_y_continuous(limits = c(80, 100)) +
  labs(x = "Payload size k (bits)", y = "Percent", colour = NULL, shape = NULL) +
  theme_paper
savefig("match_vs_bits.pdf", p, 4.3, 3.2)

# --- figure: detection strength -log10(median p) vs payload size ---
dfp <- data.frame(k = factor(main$k, levels = KS), strength = -log10(main$medp))
p <- ggplot(dfp, aes(k, strength)) +
  geom_col(fill = "#7570b3", width = 0.65) +
  geom_text(aes(label = sprintf("%.0f", strength)), vjust = -0.4, size = 3.1) +
  labs(x = "Payload size k (bits)", y = expression(-log[10]~"(median p-value)")) +
  expand_limits(y = max(dfp$strength) * 1.12) +
  theme_paper
savefig("detection_strength.pdf", p, 4.3, 3.2)

# =====================================================================
# 1b. BASELINE COMPARISON UNDER A COMMON PROTOCOL (our own runs of the baselines)
#     Reads results/opengen_baseline_<method>_<model>.json if present, else
#     emits \tbd so the table self-fills once the server runs are dropped in.
#     Cells at payload sizes a method cannot reach (exponential extraction) are
#     marked NA, matching the feasibility of each scheme.
# =====================================================================
repro_ks <- c(16, 20, 24, 32, 48)
# feasibility used only for placeholder cells (when a method has not been run
# yet); once a result file exists, feasibility is read from it (the runner
# writes an aggregate row only for payload sizes the scheme can reach).
feas <- list(cyclic = c(16, 20, 24), wang = c(16, 20, 24), mpac = repro_ks,
             cohen = repro_ks, qu = repro_ks, janas = repro_ks)
labels <- c(cyclic = "Fernandez et al.\\ \\citep{fernandez2023bricks}",
            wang   = "Wang et al.\\ \\citep{wang2023codable}",
            mpac   = "Yoo et al.\\ \\citep{yoo2024multibit}",
            cohen  = "Cohen et al.\\ \\citep{cohen2025adaptive}",
            qu     = "Qu et al.\\ \\citep{qu2024provably}",
            janas  = "Janas et al.\\ \\citep{janas2025lagrange}")
# Plain labels for the AAAI single-column table (each method is cited in the
# body text there, so the table stays narrow enough for one column).
labels_plain <- c(cyclic = "Fernandez et al.", wang = "Wang et al.",
                  mpac = "Yoo et al.", cohen = "Cohen et al.",
                  qu = "Qu et al.", janas = "Janas et al.")
read_bl <- function(m) {
  f <- file.path(RES, "other_models", sprintf("opengen_baseline_%s_%s.json", m, MODEL))
  if (!file.exists(f)) return(NULL)
  ag <- read_aggregate(f)
  setNames(ag$match_rate, as.character(ag$k))
}
bl_vals <- lapply(names(feas), read_bl); names(bl_vals) <- names(feas)
n_have <- sum(vapply(bl_vals, function(v) !is.null(v), TRUE))
cell_bl <- function(m, k) {
  v <- bl_vals[[m]]
  if (is.null(v)) {                                  # not run yet: placeholder
    return(if (k %in% feas[[m]]) "\\tbd" else "NA")
  }
  # run: the aggregate rows are exactly the feasible payload sizes
  if (is.na(v[as.character(k)])) "NA" else sprintf("%.1f", v[as.character(k)])
}
make_bl_rows <- function(lab) vapply(names(feas), function(m) paste0(
  lab[[m]], " & ",
  paste(vapply(repro_ks, function(k) cell_bl(m, k), ""), collapse = " & "),
  " \\\\"), "")
ours_row <- paste0("MOSAIC (ours) & ", paste(vapply(repro_ks, function(k) {
  i <- match(k, main$k); if (is.na(i)) "\\tbd" else sprintf("%.1f", main$match[i])
}, ""), collapse = " & "), " \\\\")
bl_table <- function(colspec, lab) c(
  sprintf("\\begin{tabular}{%s}", colspec), "\\toprule",
  "Method & \\multicolumn{5}{c}{Payload size $k$ (bits)} \\\\",
  "\\cmidrule(lr){2-6}",
  paste0(" & ", paste(repro_ks, collapse = " & "), " \\\\"), "\\midrule",
  make_bl_rows(lab), "\\midrule", ours_row, "\\bottomrule", "\\end{tabular}")
writeLines(bl_table("l rrrrr", labels), file.path(TAB, "baseline_reproduced.tex"))
if (!is.null(ATAB))                       # AAAI: plain labels, tight column spec
  writeLines(bl_table("@{}l rrrrr@{}", labels_plain),
             file.path(ATAB, "baseline_reproduced.tex"))
cat(sprintf("  table: baseline_reproduced.tex  (%d/%d baseline result files found)\n",
            n_have, length(feas)))

# =====================================================================
# 2. ROBUSTNESS  (table + edit curves + reorder curves)
# =====================================================================
rob <- fromJSON(f_rob)$results
rob$k <- as.integer(rob$k)

# --- robustness table: rows = attack x strength, cols = k in {16,24,32,48} ---
cols_k <- c(16, 24, 32, 48)
cell <- function(attack, strength, k) {
  v <- rob$match_rate[rob$attack == attack & rob$strength == strength & rob$k == k]
  if (length(v) == 0) "--" else sprintf("%.1f", v)
}
emit_rows <- function(attack, label, strengths, fmt) {
  vapply(strengths, function(st) sprintf(
    "%s & %s & %s & %s & %s & %s \\\\", label, fmt(st),
    cell(attack, st, 16), cell(attack, st, 24),
    cell(attack, st, 32), cell(attack, st, 48)), "")
}
frac <- function(st) sprintf("%.0f\\%%", st * 100)
blk  <- function(st) sprintf("%d", as.integer(st))
tab <- c(
  "\\begin{tabular}{l l r r r r}", "\\toprule",
  "Attack & Strength & $k{=}16$ & $k{=}24$ & $k{=}32$ & $k{=}48$ \\\\",
  "\\midrule",
  sprintf("Clean & -- & %s & %s & %s & %s \\\\",
          cell("clean", 0, 16), cell("clean", 0, 24),
          cell("clean", 0, 32), cell("clean", 0, 48)),
  "\\addlinespace",
  emit_rows("substitution", "Substitution", c(0.05, 0.1, 0.2, 0.3), frac),
  "\\addlinespace",
  emit_rows("insertion", "Insertion", c(0.05, 0.1, 0.2, 0.3), frac),
  "\\addlinespace",
  emit_rows("deletion", "Deletion", c(0.05, 0.1, 0.2, 0.3), frac),
  "\\addlinespace",
  emit_rows("reordering", "Reordering", c(2, 5, 10),
            function(st) sprintf("%d blocks", as.integer(st))),
  "\\bottomrule", "\\end{tabular}")
writeLines(tab, file.path(TAB, "robustness.tex"))
cat("  table: robustness.tex\n")

# --- figure: edit-attack curves, faceted by attack, coloured by k ---
ed <- rob[rob$attack %in% c("substitution", "insertion", "deletion"), ]
ed$attack <- factor(ed$attack, levels = c("substitution", "insertion", "deletion"),
                    labels = c("Substitution", "Insertion", "Deletion"))
ed$frac <- ed$strength * 100
p <- ggplot(ed, aes(frac, match_rate, colour = factor(k), group = k)) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.7) +
  facet_wrap(~attack) +
  scale_colour_manual(values = PAL, name = "Payload k") +
  labs(x = "Edited fraction of tokens (%)", y = "Exact match rate (%)") +
  theme_paper
savefig("robustness_edits.pdf", p, 7.0, 3.0)

# --- figure: reordering curves ---
ro <- rob[rob$attack == "reordering", ]
p <- ggplot(ro, aes(strength, match_rate, colour = factor(k), group = k)) +
  geom_line(linewidth = 0.6) + geom_point(size = 2) +
  scale_colour_manual(values = PAL, name = "Payload k") +
  scale_x_continuous(breaks = c(2, 5, 10)) +
  scale_y_continuous(limits = c(70, 100)) +
  labs(x = "Number of reordered blocks", y = "Exact match rate (%)") +
  theme_paper
savefig("robustness_reorder.pdf", p, 4.3, 3.2)




ed <- rob %>%
  mutate(
    frac = if_else(attack == "reordering", strength, 100 * strength),
    attack = factor(
      attack,
      levels = c("substitution", "insertion", "deletion", "reordering"),
      labels = c(
        "Substitution\nEdited fraction of tokens (%)",
        "Insertion\nEdited fraction of tokens (%)",
        "Deletion\nEdited fraction of tokens (%)",
        "Reordering\nNumber of reordered blocks"
      )
    )
  ) %>%
  filter(attack != "clean")
p <- ggplot(ed, aes(frac, match_rate, colour = factor(k), group = k)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.7) +
  facet_wrap(~attack, scales = "free_x") +
  scale_colour_manual(values = PAL, name = "Payload k") +
  labs(x = NULL, y = "Exact match rate (%)") +
  theme_paper
savefig("all_robustness_edits.pdf", p, 7.0, 7.0)


# =====================================================================
# 3. DELTA SWEEP  (quality / robustness trade-off, k=16 and k=32)
# =====================================================================
dlt <- fromJSON(f_dlt, simplifyVector = FALSE)$data$delta
drows <- do.call(rbind, lapply(dlt, function(e) {
  do.call(rbind, lapply(names(e$results), function(k) data.frame(
    delta = e$delta, k = as.integer(k),
    match = e$results[[k]]$match_rate, ppl = e$results[[k]]$mean_ppl)))
}))
rm(dlt); gc()
long <- rbind(
  data.frame(delta = drows$delta, k = factor(drows$k), value = drows$match,
             Metric = "Exact match rate (%)"),
  data.frame(delta = drows$delta, k = factor(drows$k), value = drows$ppl,
             Metric = "Perplexity (Llama-3.1 oracle)"))
p <- ggplot(long, aes(delta, value, colour = k, group = k)) +
  geom_line(linewidth = 0.7) + geom_point(size = 2) +
  facet_wrap(~Metric, scales = "free_y") +
  geom_vline(xintercept = 6, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = PAL, name = "Payload k") +
  scale_x_continuous(breaks = 2:9) +
  labs(x = expression("Logit bias "*delta), y = NULL) +
  theme_paper
savefig("delta_tradeoff.pdf", p, 7.0, 3.2)

# =====================================================================
# 4. TOKEN BUDGET HEATMAP  (match rate by payload size x token length)
# =====================================================================
tok <- fromJSON(f_tok, simplifyVector = FALSE)$data$heatmap
hm <- do.call(rbind, lapply(names(tok), function(k) {
  recs <- tok[[k]]
  do.call(rbind, lapply(recs, function(obj) data.frame(
    k = as.integer(k),
    T = vapply(obj$results, function(x) as.integer(x$T), 0L),
    match = vapply(obj$results, function(x) isTRUE(x$match), TRUE))))
}))
rm(tok); gc()
hm_ag <- aggregate(match ~ k + T, hm, function(z) 100 * mean(z))
hm_ag$k <- factor(hm_ag$k, levels = KS)
hm_ag$T <- factor(hm_ag$T, levels = sort(unique(hm_ag$T)))
p <- ggplot(hm_ag, aes(T, k, fill = match)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.0f", match),
                colour = match > 55), size = 3.1, show.legend = FALSE) +
  scale_colour_manual(values = c("TRUE" = "white", "FALSE" = "black")) +
  scale_fill_viridis_c(name = "Match (%)", limits = c(40, 100), option = "D") +
  labs(x = "Generated tokens T", y = "Payload size k (bits)") +
  theme_paper + theme(panel.grid = element_blank())
savefig("tokens_heatmap.pdf", p, 5.2, 3.2)

cat("done. assets written to", PAPER, "\n")




# Parameters
pe <- 0.07
eps <- 0.01
k_vals <- c(16, 32, 48)

# Compute Z
Z <- 2 * sqrt(pe * (1 - pe))

# Compute token budget
T_required <- (2 * k_vals * log(2) + log(1 / eps)) / (1 - Z)

# Results
data.frame(
  k = k_vals,
  Z = round(Z, 6),
  T_exact = T_required,
  T_ceiling = ceiling(T_required)
)


# Parameters
pe <- 0.07
eps <- 0.01
T <- 200
w <- 2
k_vals <- c(16, 32, 48)

# Compute Z
Z <- 2 * sqrt(pe * (1 - pe))

# Theorem 1 token budget
T_star <- (2 * k_vals * log(2) + log(1 / eps)) / (1 - Z)
T_star <- ceiling(T_star)

# Robustness threshold from Theorem 2
eps_sub <- (1 - T_star / T) / (1 + w)

# Results
results <- data.frame(
  k = k_vals,
  Z = round(Z, 6),
  T_star = T_star,
  T_star_ceiling = ceiling(T_star),
  eps_sub = eps_sub
)

print(results, digits = 6)
