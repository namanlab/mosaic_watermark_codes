#!/usr/bin/env Rscript
# merge_main_results.R
# --------------------
# Stage 1 (generation) was run in two parts:
#   opengen_main_meta-llama_Llama-2-7b-hf_1.json  -> prompts 0..4469   (60%)
#   opengen_main_meta-llama_Llama-2-7b-hf_2.json  -> prompts 4470..7710 (40%)
# This stitches them into the single canonical file the later stages read:
#   opengen_main_meta-llama_Llama-2-7b-hf.json
# The `aggregate` block is recomputed over the combined samples so match-rate,
# bit-accuracy and median p-value are correct for the full dataset.
#
# Run from this results/ folder (or anywhere):  Rscript merge_main_results.R
# NOTE: the inputs total ~250 MB; parsing them needs a few GB of RAM.

suppressPackageStartupMessages(library(jsonlite))

# locate this script's directory so the paths work regardless of cwd
args_all <- commandArgs(FALSE)
fa <- grep("--file=", args_all, value = TRUE)
here <- if (length(fa)) dirname(normalizePath(sub("--file=", "", fa[1]))) else getwd()

model_tag <- "meta-llama_Llama-2-7b-hf"
parts <- file.path(here, paste0("opengen_main_", model_tag, "_", 1:2, ".json"))
outfile <- file.path(here, paste0("opengen_main_", model_tag, ".json"))
stopifnot("both part files must exist" = all(file.exists(parts)))

cat("reading", parts[1], "...\n"); d1 <- fromJSON(parts[1], simplifyVector = FALSE)
cat("reading", parts[2], "...\n"); d2 <- fromJSON(parts[2], simplifyVector = FALSE)

samples <- c(d1$samples, d2$samples)
cat(sprintf("combined %d samples (%d + %d)\n",
            length(samples), length(d1$samples), length(d2$samples)))

# safety: drop any duplicate (prompt_index, k), keeping the first occurrence
keys <- vapply(samples, function(s) paste(s$prompt_index, s$k, sep = "_"), "")
if (any(duplicated(keys))) {
  cat("warning: dropping", sum(duplicated(keys)), "duplicate (prompt_index,k) rows\n")
  samples <- samples[!duplicated(keys)]
}
prompts <- length(unique(vapply(samples, function(s) s$prompt_index, numeric(1))))
cat(sprintf("unique: %d samples, %d prompts\n", length(samples), prompts))

# recompute aggregate exactly like Python _main_aggregate()
ks <- unlist(d1$config$watermark_sizes)
agg <- lapply(ks, function(k) {
  rows  <- Filter(function(s) s$k == k, samples)
  match <- vapply(rows, function(s) isTRUE(s$match), logical(1))
  bacc  <- vapply(rows, function(s) as.numeric(s$bit_accuracy), numeric(1))
  pval  <- vapply(rows, function(s) as.numeric(s$p_value), numeric(1))
  a <- list(k = k,
            match_rate     = mean(match) * 100,
            bit_accuracy   = mean(bacc) * 100,
            median_p_value = median(pval),
            n              = length(rows))
  cat(sprintf("  k=%-2d  match=%6.2f%%  bitacc=%6.2f%%  median_p=%.3g  n=%d\n",
              a$k, a$match_rate, a$bit_accuracy, a$median_p_value, a$n))
  a
})

cfg <- d1$config
cfg$n_prompts <- prompts
elapsed <- (if (is.null(d1$elapsed_sec)) 0 else d1$elapsed_sec) +
           (if (is.null(d2$elapsed_sec)) 0 else d2$elapsed_sec)

out <- list(config = cfg, aggregate = agg, samples = samples, elapsed_sec = elapsed)

cat("writing", outfile, "...\n")
write_json(out, outfile, auto_unbox = TRUE, digits = NA, null = "null")
cat(sprintf("done: %d samples, %d prompts -> %s\n", length(samples), prompts, outfile))
