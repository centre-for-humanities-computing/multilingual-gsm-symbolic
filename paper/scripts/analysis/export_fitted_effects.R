## Export the primary model's fitted effects as plain CSV (the .rds fits are git-ignored).
suppressPackageStartupMessages(library(lme4))
f <- readRDS("fit_template_language.rds")
out <- "../../artifacts/analysis/fitted_effects"; dir.create(out, showWarnings = FALSE)
co <- summary(f)$coefficients
write.csv(data.frame(term = rownames(co), co, row.names = NULL, check.names = FALSE),
          file.path(out, "fixed_effects.csv"), row.names = FALSE)
vc <- as.data.frame(VarCorr(f))
write.csv(vc[, c("grp", "sdcor")], file.path(out, "random_effect_sd.csv"), row.names = FALSE)
re <- ranef(f)
for (g in names(re)) write.csv(data.frame(level = rownames(re[[g]]), effect = re[[g]][[1]]),
  file.path(out, sprintf("ranef_%s.csv", gsub("[:/]", "_", g))), row.names = FALSE)
