## Appendix F tables under the template x language primary model.
##   Rscript appendix_f_tables.R -> artifacts/analysis/tl_tables/appendix_f_*.{csv,tex}
## (1) LOLO baselines incl. the reviewer-requested "model + resource level" predictor
## (2) Table 5 replacement: few-shot template budget, language-mean AND model-in-language
## Intervals: 95% percentile bootstrap over the 15 languages.
suppressPackageStartupMessages({ library(dplyr); library(tidyr) })
A <- "../../artifacts/analysis"; out <- file.path(A, "tl_tables")
set.seed(20261112)
boot_ci <- function(x, B = 10000) { b <- replicate(B, mean(x[sample(length(x), replace = TRUE)]))
  c(mean = mean(x), lo = unname(quantile(b, .025)), hi = unname(quantile(b, .975))) }
fmt <- function(v) sprintf("%.2f [%.2f, %.2f]", v[1], v[2], v[3])

## ---------------- (1) LOLO baselines ----------------
p <- read.csv(file.path(A, "lolo_predictions_tl.csv"))
cc <- read.csv(file.path(A, "transfer/language_features/language_features.csv")) %>%
  distinct(language, lcc = log10_common_crawl_pages)
p <- p %>% left_join(cc, by = "language") %>% mutate(model = factor(model)); p$model_resource <- NA_real_
for (L in unique(p$language)) {
  tr <- p[p$language != L, ]; te <- p$language == L
  m <- glm(cbind(n_correct, n_total - n_correct) ~ model + lcc, binomial, tr)
  p$model_resource[te] <- predict(m, p[te, ], type = "response") }
M <- c(`Model mean` = "model_mean", `Model size` = "size_only", `Model + resource level` = "model_resource", `Full model` = "features")
agg <- function(by) p %>% group_by(across(all_of(by))) %>%
  summarise(across(all_of(c("observed", unname(M))), ~weighted.mean(., n_total)), .groups = "drop")
per_lang <- function(by) { a <- agg(by)
  sapply(M, function(m) tapply(abs(a[[m]] - a$observed), a$language, mean) * 100) }
lang_mat <- per_lang("language"); ml_mat <- per_lang(c("language", "model"))
cor_of <- function(by) { a <- agg(by); sapply(M, function(m) cor(a[[m]], a$observed)) }
r_lang <- cor_of("language"); r_ml <- cor_of(c("language", "model"))
tab1 <- data.frame(predictor = names(M),
  language_mean_MAE = sapply(seq_along(M), function(i) fmt(boot_ci(lang_mat[, i]))),
  model_in_language_MAE = sapply(seq_along(M), function(i) fmt(boot_ci(ml_mat[, i]))))
d <- function(mat) { x <- mat[, "Full model"] - mat[, "Model + resource level"]; fmt(boot_ci(x)) }
cat("\n== Table F1: LOLO MAE (pp), 95% CI over languages\n"); print(tab1, row.names = FALSE)
cat("\nFull - (model + resource) difference: language mean", d(lang_mat), "| model in language", d(ml_mat), "\n")
red <- function(base, x) 100 * (base - x) / base
cat(sprintf("\nReduction of full model vs model mean: %.0f%% (language) / %.0f%% (model-in-language)\n",
  red(mean(lang_mat[, "Model mean"]), mean(lang_mat[, "Full model"])), red(mean(ml_mat[, "Model mean"]), mean(ml_mat[, "Full model"]))))
cat(sprintf("Reduction of model+resource vs model mean: %.0f%% / %.0f%%\n",
  red(mean(lang_mat[, "Model mean"]), mean(lang_mat[, "Model + resource level"])), red(mean(ml_mat[, "Model mean"]), mean(ml_mat[, "Model + resource level"]))))
cat(sprintf("Reduction of full vs model+resource: %.0f%% / %.0f%%\n",
  red(mean(lang_mat[, "Model + resource level"]), mean(lang_mat[, "Full model"])), red(mean(ml_mat[, "Model + resource level"]), mean(ml_mat[, "Full model"]))))
cat(sprintf("Pearson r (full model): language %.3f, model-in-language %.3f\n", r_lang["Full model"], r_ml["Full model"]))
write.csv(tab1, file.path(out, "appendix_f_baselines.csv"), row.names = FALSE)
tex1 <- c("\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{lcc}", "\\hline",
  "Predictor & Language mean & Model in language \\\\", "\\hline",
  apply(tab1, 1, function(r) paste0(r[1], " & ", r[2], " & ", r[3], " \\\\")), "\\hline", "\\end{tabular}",
  "\\caption{Leave-one-language-out error (mean absolute error in accuracy points, with 95\\% bootstrap intervals over the 15 languages). \\emph{Model mean} is the model's mean accuracy over the training languages; \\emph{Model size} adds log parameters and reasoning; \\emph{Model + resource level} is a binomial regression on model identity and log Common Crawl pages, refitted with each language held out; \\emph{Full model} is Equation~1.}",
  "\\label{tab:lolo-baselines}", "\\end{table}")
writeLines(tex1, file.path(out, "appendix_f_baselines.tex"))

## ---------------- (2) Table 5: few-shot template budget ----------------
fw <- read.csv(file.path(A, "few_shot_refit_tl.csv"))
by_lang <- fw %>% group_by(language, k) %>%
  summarise(model_in_language = 100 * mean(mae_heldout), language_mean = 100 * mean(abs(err_heldout)), .groups = "drop")
k0 <- data.frame(k = 0, language = rownames(lang_mat), model_in_language = ml_mat_k0 <- NA, language_mean = NA)
k0$language_mean <- lang_mat[k0$language, "Full model"]
k0$model_in_language <- sapply(k0$language, function(l) {
  a <- agg(c("language", "model")); mean(abs(a$features[a$language == l] - a$observed[a$language == l])) * 100 })
tab5 <- bind_rows(k0, by_lang) %>% group_by(k) %>%
  summarise(`Model in language` = fmt(boot_ci(model_in_language)), `Language mean` = fmt(boot_ci(language_mean)), .groups = "drop")
cat("\n== Table 5 (new): few-shot template budget\n"); print(as.data.frame(tab5), row.names = FALSE)
write.csv(tab5, file.path(out, "table5_fewshot_tl.csv"), row.names = FALSE)
tex5 <- c("\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{rcc}", "\\hline",
  "$k$ & Model in language & Language mean \\\\", "\\hline",
  apply(tab5, 1, function(r) paste0(trimws(r[1]), " & ", r[2], " & ", r[3], " \\\\")), "\\hline", "\\end{tabular}",
  "\\caption{Effect of including a few templates when predicting on an otherwise held-out language. Mean absolute error in accuracy points on the templates not used in fitting, over three draws per language, with 95\\% bootstrap intervals over the 15 languages.}",
  "\\label{tab:fewshot}", "\\end{table}")
writeLines(tex5, file.path(out, "table5_fewshot_tl.tex"))
