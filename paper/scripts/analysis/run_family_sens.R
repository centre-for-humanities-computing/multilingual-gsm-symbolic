## Sensitivity of the fixed effects to how model lineage enters the random-effects
## structure (reviewer concern: only 9 families, so the family variance is poorly
## estimated). All variants keep the template x language term (primary model).
##
##   Rscript run_family_sens.R <variant>
##
##   full          reference: family/base_model + model (as in the primary model)
##   no_family     drop the family intercept, keep base_model-within-family + model
##   no_model      drop the (1 | model) term
##   family_fixed  family as a fixed effect instead of a random intercept
##   no_lineage    drop family and base_model entirely; only (1 | model)
##
## One variant per process, for the same memory reasons as run_fold.R.

suppressPackageStartupMessages({ library(dplyr); library(lme4) })
setwd(if (dir.exists("/work/githubs/mgsm-run")) "/work/githubs/mgsm-run/paper/scripts/analysis"
      else "/Users/au561649/Github/multilingual-gsm-symbolic/paper/scripts/analysis")
v <- commandArgs(trailingOnly = TRUE)[1]
out_dir <- "../../artifacts/analysis/family_sens"; dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
f_out <- file.path(out_dir, paste0(v, ".rds")); if (file.exists(f_out)) quit(save = "no")

cells <- readRDS("../../artifacts/analysis/model_cells.rds")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e5))
FEAT <- "(relative_fertility_within_z + typological_distance_z + log_resource_z)"
FX <- paste(FEAT, "* log_params_z + reasoning * (relative_fertility_within_z +",
            "typological_distance_z + log_resource_z + log_params_z)")
COMMON <- "(1 | template) + (1 | language) + (1 | model:language) + (1 | template:language)"
lineage <- switch(v,
  full         = "(1 | family/base_model) + (1 | model)",
  no_family    = "(1 | family:base_model) + (1 | model)",
  no_model     = "(1 | family/base_model)",
  family_fixed = "(1 | family:base_model) + (1 | model)",
  no_lineage   = "(1 | model)",
  stop("unknown variant"))
fx <- if (v == "family_fixed") paste("family +", FX) else FX
form <- paste("cbind(n_correct, n_total - n_correct) ~", fx, "+", lineage, "+", COMMON)

t0 <- Sys.time()
fit <- glmer(as.formula(form), data = cells, family = binomial(), control = ctrl)
est <- summary(fit)$coefficients
vc <- as.data.frame(VarCorr(fit)); vc <- vc[is.na(vc$var2), ]
saveRDS(list(variant = v, minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")),
             AIC = AIC(fit), singular = isSingular(fit),
             converged = fit@optinfo$conv$opt == 0,
             fixef = data.frame(term = rownames(est), est = est[, 1], se = est[, 2], p = est[, 4]),
             vc = data.frame(group = vc$grp, sd = vc$sdcor)), f_out)
