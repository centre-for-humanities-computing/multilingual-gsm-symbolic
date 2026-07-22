# Shared model specification for the cross-lingual transfer GLMM.
#
# Sourced by both transfer_analysis.Rmd (the lme4 fit + EDA) and transfer_brms.R
# (the Bayesian fit) so the effects of interest and the target variance component
# never drift between them.
#
# The two fits share the fixed part (language features x size) and the two random
# terms that matter -- `(1 | model)`, needed to identify the size effect, and
# `(1 | model:language)`, the transfer variance we are measuring -- but treat the
# nuisance groupings `template` and `family` differently, for estimator-specific
# reasons:
#
#   lme4:  template and family stay random. glmer estimates them fast and, with
#          ~2000 draws per template, the parameterisation does not change
#          sd(model:language). This is the model as written in analysis.md.
#   brms:  template and family become fixed effects (sum-to-zero contrasts). As
#          random effects they create a posterior funnel (template) and a flat
#          ridge (the single-model family, Apertus) that saturate the NUTS
#          treedepth; fixed, they leave sd(model:language) unchanged. glmer, by
#          contrast, is very slow with ~100 fixed template dummies, which is why
#          only the Bayesian fit uses this form.

FEATURES_FULL <- "(fertility_z + typological_distance_z + log_resource_z) * log_params_z"
FEATURES_BASELINE <- "log_params_z"
TARGET_RANDOM <- "(1 | model) + (1 | model:language)"

# lme4: nuisances random (fast, matches analysis.md).
LME4_RHS <- paste(FEATURES_FULL, "+ (1 | family / model) + (1 | template) + (1 | model:language)")
LME4_RHS_BASELINE <- paste(FEATURES_BASELINE, "+ (1 | family / model) + (1 | template) + (1 | model:language)")

# brms: nuisances fixed (NUTS-friendly geometry).
MODEL_RHS <- paste(FEATURES_FULL, "+ template + family +", TARGET_RANDOM)
MODEL_RHS_BASELINE <- paste(FEATURES_BASELINE, "+ template + family +", TARGET_RANDOM)


#' Apply the factor coding the model expects (sum-to-zero contrasts on the fixed
#' nuisance factors). Call once during prep; saved into model_cells.rds so the
#' brms job inherits it without recomputation.
code_cells_factors <- function(cells) {
  cells <- dplyr::mutate(cells, dplyr::across(c(model, family, language, template), as.factor))
  contrasts(cells$template) <- contr.sum(nlevels(cells$template))
  contrasts(cells$family) <- contr.sum(nlevels(cells$family))
  cells
}


# Standardised effects of interest get a mild normal(0, 1) regulariser -- a
# coefficient of 1 is a full SD of the feature moving the log-odds by 1, already
# large here, so the prior gently stabilises the typological-distance and resource
# terms, which rest on only 7 languages. The wide normal(0, 5) default covers the
# fixed template/family contrasts, whose difficulty spans the whole logit range; a
# tight prior there would distort the fit.
TIGHT_TERMS <- c(
  "fertility_z", "typological_distance_z", "log_resource_z", "log_params_z",
  "fertility_z:log_params_z", "typological_distance_z:log_params_z", "log_resource_z:log_params_z"
)


#' Build the prior for a given brms formula, keying the tight prior to whichever
#' TIGHT_TERMS the formula actually contains (so the baseline, with only
#' log_params_z, does not get priors for absent coefficients).
make_priors <- function(formula, data) {
  coefs <- brms::get_prior(formula, data = data, family = binomial())
  present <- intersect(coefs$coef[coefs$class == "b"], TIGHT_TERMS)
  base <- c(
    brms::prior(normal(0, 2), class = "Intercept"),
    brms::prior(normal(0, 5), class = "b"),
    brms::prior(student_t(3, 0, 2.5), class = "sd")
  )
  Reduce(`+`, lapply(present, function(cf) brms::set_prior("normal(0, 1)", class = "b", coef = cf)), base)
}
