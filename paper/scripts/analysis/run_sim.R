## One simulated dataset, one refit, one result file.
##
## Parametric-bootstrap recovery study for the template x language GLMM (the
## primary specification). Same one-process-per-fit rationale as run_fold.R: R
## does not return glmer's heap to the OS, so a loop in one session dies.
##
##   Rscript run_sim.R <scenario> <rep>
##
## scenarios
##   asfit   generate from the fitted model on the real design (20 instances/cell)
##   n10     as asfit, but each cell keeps a random 10 of its 20 instances
##   n5      as asfit, but each cell keeps a random 5 of its 20 instances
##   fam5    as asfit, but 5 of the 9 families are drawn at random per replicate
##   mis     generate WITH template:language, refit WITHOUT it (the original spec)
##
## The truth is the fitted model's own fixed effects and variance components, so
## this measures estimator reliability on this design, not model misspecification
## in the real world (only `mis` probes one specific misspecification).

suppressPackageStartupMessages({ library(dplyr); library(lme4) })
setwd("/Users/au561649/Github/multilingual-gsm-symbolic/paper/scripts/analysis")
a <- commandArgs(trailingOnly = TRUE)
scen <- a[1]; rep <- as.integer(a[2])
out_dir <- "../../artifacts/analysis/sim"; dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
f_out <- file.path(out_dir, sprintf("sim_%s_r%03d.rds", scen, rep))
if (file.exists(f_out)) quit(save = "no")

cells <- readRDS("../../artifacts/analysis/model_cells.rds")
truth_fit <- readRDS("fit_template_language.rds")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e5))

seed <- 20261111L + 1000L * match(scen, c("asfit", "n10", "n5", "fam5", "mis")) + rep
set.seed(seed)

## simulate() draws fresh random effects from the fitted variance components, then
## binomial counts. Response comes back as a two-column matrix (success, failure).
s <- simulate(truth_fit, nsim = 1, seed = seed)[[1]]
d <- cells
d$n_correct <- s[, 1]; d$n_total <- s[, 1] + s[, 2]

if (scen %in% c("n10", "n5")) {
  m <- if (scen == "n10") 10L else 5L
  ## hypergeometric thinning == drawing m of the n instances without replacement
  d$n_correct <- rhyper(nrow(d), d$n_correct, d$n_total - d$n_correct, m)
  d$n_total <- m
}
if (scen == "fam5") {
  keep <- sample(levels(factor(d$family)), 5)
  d <- droplevels(d[d$family %in% keep, ])
}

FEAT <- "(relative_fertility_within_z + typological_distance_z + log_resource_z)"
RE <- "(1 | family/base_model) + (1 | model) + (1 | template) + (1 | language) + (1 | model:language)"
RE_TL <- paste(RE, "+ (1 | template:language)")
fx <- paste(FEAT, "* log_params_z + reasoning * (relative_fertility_within_z +",
            "typological_distance_z + log_resource_z + log_params_z)")
form <- paste("cbind(n_correct, n_total - n_correct) ~", fx, "+", if (scen == "mis") RE else RE_TL)

t0 <- Sys.time()
fit <- tryCatch(glmer(as.formula(form), data = d, family = binomial(), control = ctrl),
                error = function(e) e)
mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
if (inherits(fit, "error")) {
  saveRDS(list(scenario = scen, rep = rep, error = conditionMessage(fit), minutes = mins), f_out)
  quit(save = "no")
}

est <- summary(fit)$coefficients
vc <- as.data.frame(VarCorr(fit))
vc <- vc[is.na(vc$var2), ]
saveRDS(list(
  scenario = scen, rep = rep, minutes = mins,
  singular = isSingular(fit), converged = fit@optinfo$conv$opt == 0,
  fixef = data.frame(term = rownames(est), est = est[, 1], se = est[, 2]),
  vc = data.frame(group = vc$grp, sd = vc$sdcor),
  n_families = nlevels(factor(d$family))
), f_out)
