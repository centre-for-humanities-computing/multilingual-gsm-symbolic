## One fit, one process, one result file.
##
## Called by predict_new_language.Rmd. It exists because R does not return heap
## freed by glmer to the OS: fitting 15 folds in a single session grows RSS
## monotonically until the machine runs out, even with rm() and gc() between
## folds. Running each fold as its own process is the only reliable fix, and it
## makes the whole job resumable for free -- a completed fold is a file on disk.
##
##   Rscript run_fold.R lolo    <language>
##   Rscript run_fold.R fewshot <language> <k> <rep>

suppressPackageStartupMessages({ library(dplyr); library(lme4) })
setwd("/Users/au561649/Github/multilingual-gsm-symbolic/paper/scripts/analysis")
a <- commandArgs(trailingOnly = TRUE)
mode <- a[1]; L <- a[2]
out <- "../../artifacts/analysis/folds"; dir.create(out, showWarnings = FALSE, recursive = TRUE)

cells <- readRDS("../../artifacts/analysis/model_cells.rds")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 3e5))
RE <- paste("(1 | family/base_model) + (1 | model) + (1 | template) +",
            "(1 | language) + (1 | model:language)")
FEAT <- "(relative_fertility_within_z + typological_distance_z + log_resource_z)"
RHS <- paste(FEAT, "* log_params_z +",
             "reasoning * (relative_fertility_within_z +",
             "typological_distance_z + log_resource_z + log_params_z) +", RE)
Y <- "cbind(n_correct, n_total - n_correct) ~"

## training-fold standardisation, applied to the held-out language
apply_z <- function(train, test) {
  for (p in list(c("relative_fertility_within", "relative_fertility_within_z"),
                 c("typological_distance_from_english", "typological_distance_z"),
                 c("log10_common_crawl_pages", "log_resource_z"),
                 c("log2_params", "log_params_z"))) {
    s <- c(mean(train[[p[1]]]), sd(train[[p[1]]]))
    train[[p[2]]] <- (train[[p[1]]] - s[1]) / s[2]
    test[[p[2]]]  <- (test[[p[1]]] - s[1]) / s[2]
  }
  list(train = train, test = test)
}
truth <- cells %>% filter(language == L) %>% group_by(model) %>%
  summarise(truth = sum(n_correct) / sum(n_total), .groups = "drop")

if (mode == "lolo") {
  f <- file.path(out, paste0("lolo_", L, ".rds")); if (file.exists(f)) quit(save = "no")
  z <- apply_z(cells %>% filter(language != L) %>% droplevels(),
               cells %>% filter(language == L) %>% droplevels())
  ## a language never evaluated has no language or model:language intercept
  KNOWN_RE <- ~ (1 | family/base_model) + (1 | model) + (1 | template)
  feat <- glmer(as.formula(paste(Y, RHS)), data = z$train, family = binomial(), control = ctrl)
  size <- glmer(as.formula(paste(Y, "log_params_z + reasoning +", RE)),
                data = z$train, family = binomial(), control = ctrl)
  mm <- z$train %>% group_by(model) %>%
    summarise(model_mean = sum(n_correct) / sum(n_total), .groups = "drop")
  saveRDS(z$test %>%
    transmute(language = L, model, template, n_correct, n_total,
              observed = n_correct / n_total,
              features = plogis(predict(feat, newdata = z$test, re.form = KNOWN_RE, allow.new.levels = TRUE)),
              size_only = plogis(predict(size, newdata = z$test, re.form = KNOWN_RE, allow.new.levels = TRUE)),
              grand_mean = sum(z$train$n_correct) / sum(z$train$n_total)) %>%
    left_join(mm, by = "model"), f)

} else {
  k <- as.integer(a[3]); rep <- as.integer(a[4])
  f <- file.path(out, sprintf("few_%s_k%02d_r%d.rds", L, k, rep))
  if (file.exists(f)) quit(save = "no")
  ## Nested draws: the k-sets for a given (language, rep) are prefixes of one
  ## shuffle, so movement along the curve is not confounded with the draw. The
  ## seed depends only on language and rep, so every k sees the same shuffle.
  set.seed(20260916L + as.integer(factor(L, levels = sort(unique(cells$language)))) * 100L + rep)
  keep <- head(sample(unique(cells$template)), k)
  z <- apply_z(cells %>% filter(language != L | template %in% keep) %>% droplevels(),
               cells %>% filter(language == L) %>% droplevels())
  fit <- glmer(as.formula(paste(Y, RHS)), data = z$train, family = binomial(), control = ctrl)
  ## re.form = NULL: the target's language and model:language levels are now
  ## estimated from the k templates
  p <- plogis(predict(fit, newdata = z$test, re.form = NULL, allow.new.levels = TRUE))
  sc <- function(sel) {
    d <- z$test[sel, ] %>% mutate(p = p[sel]) %>% group_by(model) %>%
      summarise(pred = mean(p), .groups = "drop") %>% inner_join(truth, by = "model")
    c(mae = mean(abs(d$pred - d$truth)), err = mean(d$pred) - mean(d$truth))
  }
  seen <- z$test$template %in% keep
  all_t <- sc(rep(TRUE, nrow(z$test))); held <- sc(!seen)
  saveRDS(data.frame(language = L, k = k, rep = rep,
                     mae_all = all_t[["mae"]], err_all = all_t[["err"]],
                     mae_heldout = held[["mae"]], err_heldout = held[["err"]],
                     sd_language = attr(VarCorr(fit)$language, "stddev")[[1]]), f)
}
