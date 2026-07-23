#!/usr/bin/env Rscript
# Export GSM-Symbolic estimates for the MultiZebra generalization comparison, using
# the SAME structure the zebra per-puzzle model uses, so fixed effects, the
# transfer variance component sd(model:language), and the variance explained are
# all directly comparable. Reads the cells saved by transfer_analysis.Rmd.
#
# Both predictors are on a log2 (per-doubling) scale, centred at ~32M Common Crawl
# pages and an 8B model (shared with the zebra fit). Only resource and size are
# used, since the Germanic-only zebra set cannot test typology/fertility.
suppressPackageStartupMessages({library(dplyr); library(lme4)})
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
out_dir <- normalizePath(file.path(here, "..", "..", "artifacts", "analysis"))

RESOURCE_CENTER <- 7.5 # log10 CC pages (~32M)
SIZE_CENTER <- 3       # log2 params (= 8B)

cells <- readRDS(file.path(out_dir, "model_cells.rds")) %>%
  mutate(
    resource_c = (log10_common_crawl_pages - RESOURCE_CENTER) * log2(10), # doublings of CC pages
    size_c     = log2_params - SIZE_CENTER,                                # doublings of params
    reasoning  = factor(reasoning, levels = c("off", "on"))
  ) %>%
  mutate(across(c(model, language, template), as.factor))

# Reasoning enters as a fixed-effect covariate so resource/size are compared to the
# MultiZebra fit on equal footing -- most MultiZebra runs are reasoning models, so both
# sides control for it rather than restricting to the non-reasoning subset. It barely
# moves GSM's resource/size (GSM's reasoning effect is mild), but keeps the spec matched.
message("Fitting GSM full and baseline models ...")
full <- glmer(
  cbind(n_correct, n_total - n_correct) ~ resource_c * size_c + reasoning +
    (1 | model) + (1 | language) + (1 | template) + (1 | model:language),
  data = cells, family = binomial())

base <- glmer( # drops the resource (language) feature relative to full
  cbind(n_correct, n_total - n_correct) ~ size_c + reasoning +
    (1 | model) + (1 | language) + (1 | template) + (1 | model:language),
  data = cells, family = binomial())

co <- summary(full)$coefficients
estimates <- data.frame(task = "GSM-Symbolic", term = rownames(co),
                        estimate = co[, "Estimate"], se = co[, "Std. Error"],
                        p_value = co[, "Pr(>|z|)"], row.names = NULL)

sd_ml <- function(m) attr(VarCorr(m)$`model:language`, "stddev")[[1]]
variance <- data.frame(task = "GSM-Symbolic",
                       sd_ml_full = sd_ml(full), sd_ml_baseline = sd_ml(base),
                       variance_explained = 1 - sd_ml(full)^2 / sd_ml(base)^2)

write.csv(estimates, file.path(out_dir, "gsm_generalization_estimates.csv"), row.names = FALSE)
write.csv(variance, file.path(out_dir, "gsm_generalization_variance.csv"), row.names = FALSE)
cat("GSM cells:", nrow(cells), "\n"); print(estimates); print(variance)
