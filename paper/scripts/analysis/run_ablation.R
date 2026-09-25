## One ablation fit, one process, one file. Same rationale as run_fold.R: R does
## not return heap freed by glmer to the OS, and these two fits carry 1500 extra
## template:language levels, so running both in one session is the largest
## single memory demand in the pipeline.
##
##   Rscript run_ablation.R full      -> fit_template_language.rds
##   Rscript run_ablation.R baseline  -> fit_tl_baseline.rds

suppressPackageStartupMessages({ library(dplyr); library(lme4) })
setwd("/Users/au561649/Github/multilingual-gsm-symbolic/paper/scripts/analysis")
which <- commandArgs(trailingOnly = TRUE)[1]
out <- if (which == "full") "fit_template_language.rds" else "fit_tl_baseline.rds"
if (file.exists(out)) quit(save = "no")

cells <- readRDS("../../artifacts/analysis/model_cells.rds")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e5))
RE_NEW <- paste("(1 | family/base_model) + (1 | model) + (1 | template) +",
                "(1 | language) + (1 | model:language) + (1 | template:language)")
FEAT <- "(relative_fertility_within_z + typological_distance_z + log_resource_z)"
Y <- "cbind(n_correct, n_total - n_correct) ~"
rhs <- if (which == "full")
  paste(FEAT, "* log_params_z +",
        "reasoning * (relative_fertility_within_z +",
        "typological_distance_z + log_resource_z + log_params_z) +", RE_NEW) else
  paste("log_params_z + reasoning +", RE_NEW)

t0 <- Sys.time()
saveRDS(glmer(as.formula(paste(Y, rhs)), data = cells, family = binomial(), control = ctrl), out)
cat(which, "fitted in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
