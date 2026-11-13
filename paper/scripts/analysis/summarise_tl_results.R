## Results tables for the promoted (template x language) specification.
##   Rscript summarise_tl_results.R  -> artifacts/analysis/tl_tables/*.csv
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(lme4) })
A <- "../../artifacts/analysis"; out <- file.path(A, "tl_tables"); dir.create(out, showWarnings = FALSE)
cache <- function(pat) { f <- list.files("_cache/transfer_analysis", pat, full.names = TRUE); f <- f[grepl("[.]RData$", f)]
  e <- new.env(); load(f[order(file.mtime(f), decreasing = TRUE)][1], envir = e); e }
tl <- readRDS("fit_template_language.rds"); tl_b <- readRDS("fit_tl_baseline.rds")
old <- readRDS(file.path(A, "original_spec/fit_main.rds")); old_b <- readRDS(file.path(A, "original_spec/fit_baseline.rds"))

## ---- Table 3 replacement: fixed effects + variance components ----------------
co <- summary(tl)$coefficients
fx <- data.frame(term = rownames(co), estimate = co[, 1], se = co[, 2], z = co[, 3], p = co[, 4],
                 ci_low = co[, 1] - 1.96 * co[, 2], ci_high = co[, 1] + 1.96 * co[, 2],
                 estimate_old = summary(old)$coefficients[rownames(co), 1])
write.csv(fx, file.path(out, "table3_fixed_effects_tl.csv"), row.names = FALSE)
sdl <- function(f, g) { v <- VarCorr(f); if (is.null(v[[g]])) NA_real_ else as.numeric(attr(v[[g]], "stddev")[1]) }
gs <- unique(c(names(VarCorr(tl)), names(VarCorr(old))))
vc <- data.frame(component = gs, sd_tl = sapply(gs, sdl, f = tl), sd_old = sapply(gs, sdl, f = old))
write.csv(vc, file.path(out, "table3_variance_components_tl.csv"), row.names = FALSE)
fit_stats <- data.frame(model = c("template_x_language", "original"),
  N = c(nobs(tl), nobs(old)), AIC = c(AIC(tl), AIC(old)), BIC = c(BIC(tl), BIC(old)),
  logLik = c(as.numeric(logLik(tl)), as.numeric(logLik(old))))
write.csv(fit_stats, file.path(out, "table3_fit_stats.csv"), row.names = FALSE)

## ---- variance explained: which model underlies each headline number ----------
sp <- function(b, f, g) 100 * (1 - sdl(f, g)^2 / sdl(b, g)^2)
cmb <- function(b, f) { comp <- c("language", "model:language")
  100 * (1 - sum(sapply(comp, function(g) sdl(f, g)^2)) / sum(sapply(comp, function(g) sdl(b, g)^2))) }
ve <- data.frame(
  quantity = c("between-language (language sd)", "model-by-language", "combined"),
  original_model = c(sp(old_b, old, "language"), sp(old_b, old, "model:language"), cmb(old_b, old)),
  template_x_language = c(sp(tl_b, tl, "language"), sp(tl_b, tl, "model:language"), cmb(tl_b, tl)))
write.csv(ve, file.path(out, "variance_explained.csv"), row.names = FALSE)
cat("== variance explained (%)\n"); print(ve %>% mutate(across(where(is.numeric), ~round(., 1))), row.names = FALSE)

## ---- LOLO baselines with bootstrap CIs over languages -----------------------
p <- read.csv(file.path(A, "lolo_predictions_tl.csv"))
cc <- read.csv(file.path(A, "transfer/language_features/language_features.csv")) %>% distinct(language, lcc = log10_common_crawl_pages)
p <- p %>% left_join(cc, by = "language") %>% mutate(model = factor(model)); p$mr <- NA_real_
for (L in unique(p$language)) {
  tr <- p[p$language != L, ]; te <- p$language == L
  m <- glm(cbind(n_correct, n_total - n_correct) ~ model + lcc, binomial, tr)
  p$mr[te] <- predict(m, p[te, ], type = "response") }
M <- c(features = "features", `model + resource` = "mr", `size only` = "size_only", `model mean` = "model_mean")
agg <- function(d, by) d %>% group_by(across(all_of(by))) %>%
  summarise(across(all_of(c("observed", unname(M))), ~weighted.mean(., n_total)), .groups = "drop")
mae_by_lang <- function(d, by) { a <- agg(d, by)
  sapply(M, function(m) tapply(abs(a[[m]] - a$observed), a$language, mean) * 100) }   # languages x methods
set.seed(1)
boot <- function(mat, B = 4000) { n <- nrow(mat)
  t(replicate(B, colMeans(mat[sample(n, replace = TRUE), , drop = FALSE]))) }
rows <- list()
for (lvl in list(c("language", "mean of a novel language", "language"), c(c("language", "model"), "specific model in a novel language", "model"))) {
  by <- if (length(lvl) == 3) lvl[1] else c("language", "model")
  mat <- mae_by_lang(p, by); b <- boot(mat)
  rows[[length(rows) + 1]] <- data.frame(level = lvl[length(lvl) - 1], method = colnames(mat), MAE = colMeans(mat),
      lo = apply(b, 2, quantile, .025), hi = apply(b, 2, quantile, .975))
  d <- b[, "features"] - b[, "model + resource"]
  cat(sprintf("\n%s: features - (model + resource) MAE diff = %.2f [%.2f, %.2f]\n", lvl[length(lvl) - 1],
              mean(mat[, "features"] - mat[, "model + resource"]), quantile(d, .025), quantile(d, .975)))
}
lolo <- bind_rows(rows); write.csv(lolo, file.path(out, "lolo_baselines_bootstrap.csv"), row.names = FALSE)
cat("\n== LOLO MAE with 95% bootstrap CI over the 15 languages\n"); print(lolo %>% mutate(across(where(is.numeric), ~round(., 2))), row.names = FALSE)
cat("\n== fixed effects (new vs original)\n"); print(fx %>% transmute(term, new = round(estimate, 3), se = round(se, 3), original = round(estimate_old, 3)), row.names = FALSE)
