#!/usr/bin/env Rscript
# Bayesian fit of the cross-lingual transfer GLMM.
#
# Split out from transfer_analysis.Rmd so the hours-long sampling runs as a
# background batch job while the lme4 fit and EDA stay fast to iterate on in the
# Rmd. Reads the cells that the Rmd prepares and saves; run the Rmd first.
#
#   Rscript paper/scripts/analysis/transfer_brms.R
#
# Fits are cached to paper/artifacts/analysis/ via brms' `file` argument, so a
# re-run reuses whatever completed; delete the .rds files to refit.
#
# Backend is cmdstanr with the default diagonal metric.
#
# This posterior is expensive for HMC: each cell has ~2000 draws, so the
# likelihood pins its total logit tightly and the additive terms that sum to it
# are strongly correlated, which makes NUTS saturate its treedepth (~3.5 s/iter,
# so the full run is a few hours). This is inefficiency, not invalidity -- the fit
# converges (Rhat ~ 1, no divergences, sd(model:language) matching lme4). Levers
# tried and rejected: moving template and family to fixed effects (no change), and
# a dense metric (`metric = "dense_e"`), which at ~400 parameters could not
# estimate its covariance from the warmup draws and came out worse -- slower, 4%
# divergences, still treedepth-saturated. If revisiting, a within-chain
# reduce_sum threading approach is the more promising direction.

suppressPackageStartupMessages({
  library(dplyr)
  library(brms)
  library(cmdstanr)
  library(posterior)
})

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
here <- if (!is.na(script_path)) dirname(normalizePath(script_path)) else normalizePath(".")
out_dir <- normalizePath(file.path(here, "..", "..", "artifacts", "analysis"))
source(file.path(here, "transfer_spec.R"))

cells_path <- file.path(out_dir, "model_cells.rds")
if (!file.exists(cells_path)) {
  stop("model_cells.rds not found. Knit transfer_analysis.Rmd first to prepare it.")
}
cells <- readRDS(cells_path)
message(sprintf("Loaded %d cells from %s", nrow(cells), basename(cells_path)))

fit_one <- function(rhs, name) {
  formula <- as.formula(paste("n_correct | trials(n_total) ~", rhs))
  message("Fitting ", name, " ...")
  brm(
    formula = formula,
    data = cells,
    family = binomial(),
    prior = make_priors(formula, cells),
    backend = "cmdstanr",
    chains = 4,
    iter = 2000,
    warmup = 1000,
    # Two chains at a time keeps peak memory modest so the machine stays usable.
    cores = 2,
    seed = 42,
    file = file.path(out_dir, paste0("transfer_", name)),
    file_refit = "on_change"
  )
}

bfit_baseline <- fit_one(MODEL_RHS_BASELINE, "baseline")
bfit_full <- fit_one(MODEL_RHS, "full")

# Diagnostics -- nothing downstream should be trusted until these pass: zero
# divergences, Rhat < 1.01, effective sample sizes comfortably above ~400.
diagnostics <- function(fit) {
  np <- nuts_params(fit)
  draws <- summarise_draws(fit)
  c(
    divergent = sum(subset(np, Parameter == "divergent__")$Value),
    max_treedepth_hit = sum(subset(np, Parameter == "treedepth__")$Value >= 10),
    max_rhat = max(draws$rhat, na.rm = TRUE),
    min_bulk_ess = min(draws$ess_bulk, na.rm = TRUE),
    min_tail_ess = min(draws$ess_tail, na.rm = TRUE)
  )
}

cat("\n=== diagnostics ===\n")
print(rbind(baseline = diagnostics(bfit_baseline), full = diagnostics(bfit_full)))

# The quantity of interest: how much of the model-by-language transfer variance
# the language features absorb, propagated through the full posterior rather than
# dividing two point estimates.
sd_ml <- function(fit) as_draws_df(fit)[["sd_model:language__Intercept"]]
reduction <- 1 - (sd_ml(bfit_full)^2) / (sd_ml(bfit_baseline)^2)

cat("\n=== transfer variance explained by language features ===\n")
print(tibble(
  median = median(reduction),
  q2.5 = quantile(reduction, 0.025),
  q97.5 = quantile(reduction, 0.975)
))

cat("\n=== full-model fixed effects ===\n")
print(fixef(bfit_full))
