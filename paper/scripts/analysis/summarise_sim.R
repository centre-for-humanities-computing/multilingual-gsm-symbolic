## Summarise the parametric-bootstrap recovery study (run_sim.R).
## Truth = the fitted template x language model (fit_template_language.rds).
##   Rscript summarise_sim.R   -> artifacts/analysis/sim_summary/*.csv
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(lme4) })
setwd(if (dir.exists("/Users/au561649")) "/Users/au561649/Github/multilingual-gsm-symbolic/paper/scripts/analysis" else ".")
out <- "../../artifacts/analysis/sim_summary"; dir.create(out, showWarnings = FALSE, recursive = TRUE)

tf <- readRDS("fit_template_language.rds")
truth_fx <- fixef(tf)
vcs <- as.data.frame(VarCorr(tf)); truth_vc <- setNames(vcs$sdcor[is.na(vcs$var2)], vcs$grp[is.na(vcs$var2)])

R <- lapply(list.files("../../artifacts/analysis/sim", full.names = TRUE), readRDS)
R <- Filter(function(x) is.null(x$error), R)

meta <- bind_rows(lapply(R, function(x) data.frame(
  scenario = x$scenario, rep = x$rep, minutes = x$minutes,
  converged = x$converged, singular = x$singular)))
cat("replicates per scenario:\n"); print(table(meta$scenario))
write.csv(meta %>% group_by(scenario) %>% summarise(n = n(), converged = mean(converged),
          singular = mean(singular), median_min = median(minutes)),
          file.path(out, "run_overview.csv"), row.names = FALSE)

## fixed effects: bias, RMSE, coverage of the Wald 95% CI, relative SE bias
fx <- bind_rows(lapply(R, function(x) cbind(scenario = x$scenario, x$fixef))) %>%
  mutate(truth = truth_fx[term], err = est - truth,
         cover = abs(err) <= 1.96 * se)
fx_sum <- fx %>% group_by(scenario, term) %>%
  summarise(n = n(), truth = first(truth), bias = mean(err), rmse = sqrt(mean(err^2)),
            emp_sd = sd(est), mean_se = mean(se), coverage = mean(cover), .groups = "drop")
write.csv(fx_sum, file.path(out, "fixed_effects.csv"), row.names = FALSE)

## variance components: bias, spread, boundary rate (sd < 0.01)
vc <- bind_rows(lapply(R, function(x) cbind(scenario = x$scenario, rep = x$rep, x$vc))) %>%
  mutate(group = as.character(group), truth = truth_vc[group])
vc_sum <- vc %>% group_by(scenario, group) %>%
  summarise(n = n(), truth = first(truth), mean_est = mean(sd), bias = mean(sd) - first(truth),
            rel_bias_pct = 100 * (mean(sd) - first(truth)) / first(truth),
            q05 = quantile(sd, .05), q95 = quantile(sd, .95),
            boundary = mean(sd < 0.01), .groups = "drop")
write.csv(vc_sum, file.path(out, "variance_components.csv"), row.names = FALSE)

cat("\n== Fixed effects, asfit: bias / RMSE / mean SE / empirical SD / coverage\n")
print(as.data.frame(fx_sum %>% filter(scenario == "asfit") %>%
      mutate(across(where(is.numeric), ~round(., 3))) %>% select(-scenario)), row.names = FALSE)
cat("\n== Coverage by scenario (rows = terms)\n")
print(as.data.frame(fx_sum %>% select(scenario, term, coverage) %>%
      pivot_wider(names_from = scenario, values_from = coverage) %>%
      mutate(across(where(is.numeric), ~round(., 2)))), row.names = FALSE)
cat("\n== Variance components by scenario: truth, mean estimate, relative bias %, 90% range, boundary rate\n")
print(as.data.frame(vc_sum %>% mutate(across(where(is.numeric), ~round(., 2)))), row.names = FALSE)
