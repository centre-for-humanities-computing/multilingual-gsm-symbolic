## Is "size shrinks the resource gap" a ceiling artifact?
##
## The concern: a negative resource x size interaction is what you mechanically
## get if the high-resource arm saturates while the low-resource arm still has
## headroom. The logit link mitigates this but does not remove it, and a handful
## of 97-99% cells carry large leverage.
##
## Three restrictions, in decreasing order of how defensible they are:
##
##   eng90  keep only models scoring < 90% in English. A MODEL-level restriction
##          on one summary number, not a per-observation filter, and it leaves
##          zero cells above 90% while retaining the full 0.5-70B size range
##          (sd of log2 params 1.90, against 1.87 in the full data). This is the
##          test the claim should stand or fall on.
##   eng95  the same, at a laxer threshold, as a dose-response check.
##   cell90 drop individual cells above 90%, which is what a reviewer is most
##          likely to ask for -- but it conditions on the outcome being modelled,
##          removing 46% of the high-resource/large-model cells against 3% of the
##          low-resource/small-model ones, i.e. exactly the contrast the
##          interaction measures. It is also redundant: eng90 already leaves zero
##          saturated cells without conditioning on y. Kept here so the answer
##          exists if asked for, but NOT reported as a column in the paper.
##
##   Rscript run_saturation.R <eng90|eng95|cell90>

suppressPackageStartupMessages({ library(dplyr); library(lme4) })
setwd("/Users/au561649/Github/multilingual-gsm-symbolic/paper/scripts/analysis")
which <- commandArgs(trailingOnly = TRUE)[1]
out <- sprintf("fit_sat_%s.rds", which)
if (file.exists(out)) quit(save = "no")

cells <- readRDS("../../artifacts/analysis/model_cells.rds")
acc <- cells %>% group_by(model, language) %>%
  summarise(a = sum(n_correct) / sum(n_total), .groups = "drop")
d <- switch(which,
  eng90  = cells %>% filter(model %in% acc$model[acc$language == "eng" & acc$a < 0.90]),
  eng95  = cells %>% filter(model %in% acc$model[acc$language == "eng" & acc$a < 0.95]),
  cell90 = cells %>% anti_join(filter(acc, a > 0.90), by = c("model", "language"))
) %>% droplevels()

## Standardisation is recomputed on the retained data: leaving the full-sample
## z-scores in place would put the restricted subset off-centre and make the
## interaction coefficient incomparable.
z <- function(x) as.numeric(scale(x))
d <- d %>% mutate(log_params_z = z(log2_params), log_resource_z = z(log10_common_crawl_pages),
                  typological_distance_z = z(typological_distance_from_english),
                  relative_fertility_within_z = z(relative_fertility_within))

cat(which, "| models:", nlevels(d$model), "| cells:", n_distinct(paste(d$model, d$language)),
    "| rows:", nrow(d), "\n")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 3e5))
RE <- paste("(1 | family/base_model) + (1 | model) + (1 | template) +",
            "(1 | language) + (1 | model:language)")
FEAT <- "(relative_fertility_within_z + typological_distance_z + log_resource_z)"
RHS <- paste(FEAT, "* log_params_z +",
             "reasoning * (relative_fertility_within_z +",
             "typological_distance_z + log_resource_z + log_params_z) +", RE)
t0 <- Sys.time()
saveRDS(glmer(as.formula(paste("cbind(n_correct, n_total - n_correct) ~", RHS)),
              data = d, family = binomial(), control = ctrl), out)
cat("  fitted in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
