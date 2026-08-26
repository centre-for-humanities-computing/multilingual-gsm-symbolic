## Draft results figures. Reads fitted models from the knitr cache (does not
## refit) and prepared cells from ../../artifacts/analysis/model_cells.rds.
##
## Figure order follows the "levers" spine:
##   1 fig_levers        - the two interventions, on the probability scale  [SPINE]
##   2 fig_levers_raw    - model-free companion to (1)
##   3 fig_effects       - all fitted effects, grouped by inferential stratum
##   4 fig_ladder        - model-free degradation curve
##   5 fig_ladder_fitted - fitted effects by rung
##   6 fig_design_space  - language selection / confound structure
##   7 fig_variance      - adequacy of the feature set (supporting, not headline)

suppressPackageStartupMessages({
  library(lme4); library(dplyr); library(tidyr); library(ggplot2); library(scales)
})

setwd("/Users/au561649/Github/multilingual-gsm-symbolic/paper/scripts/analysis")
dir.create("figures", showWarnings = FALSE)

cache <- function(pattern) {
  f <- list.files("_cache/transfer_analysis", pattern = pattern, full.names = TRUE)
  f <- f[grepl("\\.RData$", f)]
  stopifnot(length(f) >= 1)
  ## a re-knit leaves the previous generation behind under a different hash;
  ## always take the most recently written one.
  f <- f[order(file.mtime(f), decreasing = TRUE)][1]
  e <- new.env(); load(f, envir = e); e
}
cells    <- readRDS("../../artifacts/analysis/model_cells.rds")
## The fertility columns were renamed to relative_fertility_*; alias both ways so
## this script works against either cache generation.
if (!"relative_fertility_within_z" %in% names(cells) && "fertility_within_z" %in% names(cells))
  cells$relative_fertility_within_z <- cells$fertility_within_z
if (!"fertility_within_z" %in% names(cells) && "relative_fertility_within_z" %in% names(cells))
  cells$fertility_within_z <- cells$relative_fertility_within_z
fit_main <- cache("^fit-lme4_")$fit_main
fit_base <- cache("^fit-baseline_")$fit_baseline
E_lad    <- cache("^fit-ladder_")
fit_lhs  <- E_lad$fit_lhs; fit_rhs <- E_lad$fit_rhs

## back-transforms, so axes can be labelled in native units
mu_p  <- mean(cells$log2_params);                     sd_p  <- sd(cells$log2_params)
mu_r  <- mean(cells$log10_common_crawl_pages);        sd_r  <- sd(cells$log10_common_crawl_pages)
mu_t  <- mean(cells$typological_distance_from_english); sd_t <- sd(cells$typological_distance_from_english)
z_params <- function(b) (log2(b) - mu_p)/sd_p
z_res    <- function(x) (x - mu_r)/sd_r
z_typ    <- function(x) (x - mu_t)/sd_t
## Resource is plotted on an absolute log axis of Common Crawl pages. Log spacing
## keeps a doubling a constant distance (so the per-doubling effect size is still
## readable off the slope) while the tick labels stay concrete and need no
## reference language.
dbl_res  <- function(log10pages) log10pages          # identity; kept for call sites
pages_lab <- function(l) {
  v <- 10^l
  ifelse(v >= 1e9, paste0(round(v/1e9, 1), "B"),
  ifelse(v >= 1e6, paste0(round(v/1e6), "M"), format(v, big.mark = ",")))
}
RES_BREAKS <- c(6, 7, 8, 9)

## One hue per lever so the two rows cannot be confused; light -> dark within a
## lever encodes less -> more of it. Blue vs orange is colour-vision-safe.
PAL <- c(small = "#9EC5DE", large = "#14425C",   # model size  (blues)
         off   = "#F0B67F", on    = "#A8501E")   # reasoning   (oranges)

## Two-line axis title: bold main label, plain qualifier underneath (no
## parentheses). ggtext is unavailable, so this uses plotmath.
axlab <- function(main, sub) bquote(atop(bold(.(main)), .(sub)))

## Construct vs. measurement: the bold line names what we mean, the line under it
## names the proxy we actually have. Both features are operationalisations, and
## saying so on the axis is cheaper than a footnote.
LAB_RES <- 'atop(bold("Resource level"), "Common Crawl pages")'
LAB_TYP <- 'atop(bold("Typological distance"), "URIEL syntactic distance from English")'
LEG_RES <- axlab("Resource level", "Common Crawl pages")
## No panel grid. Reference lines that carry meaning (parity, zero) are drawn
## explicitly per figure; a background lattice competes with them and with the
## data marks, and none of these figures asks the reader to look up a precise
## value off the panel.
theme_paper <- theme_minimal(base_size = 10) +
  theme(panel.grid = element_blank(),
        axis.line = element_line(colour = "grey65", linewidth = 0.3),
        axis.ticks = element_line(colour = "grey65", linewidth = 0.3),
        axis.ticks.length = unit(2.5, "pt"),
        plot.title = element_text(face = "bold", size = 11),
        plot.subtitle = element_text(colour = "grey35", size = 9),
        strip.text = element_text(face = "bold", size = 9),
        axis.title = element_text(face = "plain"),
        legend.position = "bottom")
ok <- function(p, file, w, h) {
  ggsave(file.path("figures", paste0(file, ".pdf")), p, width = w, height = h, device = cairo_pdf)
  ggsave(file.path("figures", paste0(file, ".png")), p, width = w, height = h, dpi = 200)
  cat("wrote", file, "\n")
}


## Coefficient lookup that tolerates the fertility -> relative_fertility rename,
## so this script works against either cache generation.
bget <- function(fit, nm) {
  b <- fixef(fit)
  alt <- c(nm, sub("^relative_", "", nm), paste0("relative_", nm))
  hit <- alt[alt %in% names(b)]
  stopifnot(length(hit) >= 1)
  b[[hit[1]]]
}
cname <- function(fit, nm) {
  alt <- c(nm, sub("^relative_", "", nm), paste0("relative_", nm))
  hit <- alt[alt %in% names(fixef(fit))]; hit[1]
}

lang_meta <- cells %>%
  distinct(language, log10_common_crawl_pages, typological_distance_from_english) %>%
  mutate(script = if_else(language %in% c("ara","hin","jpn","mar","rus","ukr","zho"),
                          "non-Latin", "Latin"))

## =====================================================================
## 1. THE LEVERS  (spine figure)
##
## Two versions, because the scale matters for honesty:
##
##  (a) fig_levers -- predicted ERROR-ODDS RELATIVE TO ENGLISH, log scale.
##      This is the quantity the model actually estimates, so lines are
##      straight and the interaction IS the difference in slope. It is also
##      the same quantity as the model-free figure, so the two are directly
##      comparable. Flat line = no penalty.
##
##  (b) fig_levers_prob -- the same thing on the probability scale, which
##      readers find more concrete but which re-expresses the interaction
##      through the base rate: parallel log-odds lines look very different
##      near the floor than near the ceiling. Use with care.
## =====================================================================
eng_res <- lang_meta$log10_common_crawl_pages[lang_meta$language == "eng"]
eng_res0 <<- eng_res
SMALL <- 4; LARGE <- 32   # avoid the 1B floor, where probabilities compress

lever_grid <- function(feature, native_range) {
  x <- seq(native_range[1], native_range[2], length.out = 60)
  bind_rows(
    expand_grid(x_native = x, lever = "Model size", level = c("small","large")) %>%
      mutate(log_params_z = if_else(level == "small", z_params(SMALL), z_params(LARGE)),
             reasoning = factor("off", levels = c("off","on")),
             level_label = if_else(level == "small", paste0(SMALL, "B"), paste0(LARGE, "B"))),
    expand_grid(x_native = x, lever = "Reasoning", level = c("off","on")) %>%
      mutate(log_params_z = z_params(8),
             reasoning = factor(level, levels = c("off","on")),
             level_label = if_else(level == "off", "reasoning off", "reasoning on"))
  ) %>% mutate(feature = feature)
}

nd <- bind_rows(
  lever_grid("Common Crawl pages", range(lang_meta$log10_common_crawl_pages)),
  lever_grid("Typological distance from English", range(lang_meta$typological_distance_from_english))
) %>%
  mutate(is_res = feature == "Common Crawl pages",
         # the non-varying feature is held at ENGLISH's value, not the sample
         # mean, so that every curve passes through parity at English.
         log_resource_z         = if_else(is_res, z_res(x_native), z_res(eng_res)),
         typological_distance_z = if_else(is_res, z_typ(0), z_typ(x_native)),
         relative_fertility_within_z = 0, fertility_within_z = 0)

## reference row: same lever settings, but the language features set to English.
## English is raw typological distance 0, which standardises to z_typ(0), NOT 0
## (that would be the sample mean, ~0.2).
ref <- nd %>% mutate(log_resource_z = z_res(eng_res), typological_distance_z = z_typ(0))
nd$lp     <- predict(fit_main, newdata = nd,  re.form = NA)
nd$lp_eng <- predict(fit_main, newdata = ref, re.form = NA)
nd <- nd %>%
  mutate(pred = plogis(lp), pred_eng = plogis(lp_eng),
         # odds ratio for a wrong answer, relative to English
         err_or = exp(lp_eng - lp),
         # RISK ratio: literally "times more likely to get it wrong". This is
         # NOT the odds ratio -- conflating them overstates the gap by ~40-85%
         # at our accuracy levels -- and it is the quantity the intuitive
         # phrasing actually names.
         err_rr = (1 - pred)/(1 - pred_eng),
         x_plot = if_else(is_res, dbl_res(x_native), x_native),
         feature = factor(feature, levels = c("Common Crawl pages",
                                              "Typological distance from English")),
         feature_lab = factor(if_else(is_res, LAB_RES, LAB_TYP), levels = c(LAB_RES, LAB_TYP)),
         lever = factor(lever, levels = c("Model size","Reasoning")),
         level_label = factor(level_label,
           levels = c(paste0(SMALL,"B"), paste0(LARGE,"B"), "reasoning off", "reasoning on")))

## Language ticks are drawn only in the bottom row: both rows share the same x
## scale, so one set is enough, and without a panel grid a rug in the upper row
## has no baseline to sit on and reads as stray marks mid-figure.
anchor <- lang_meta %>%
  transmute(feature_lab = LAB_RES, x_plot = dbl_res(log10_common_crawl_pages)) %>%
  bind_rows(lang_meta %>% transmute(feature_lab = LAB_TYP,
                                    x_plot = typological_distance_from_english)) %>%
  mutate(feature_lab = factor(feature_lab, levels = c(LAB_RES, LAB_TYP)),
         lever = factor("Reasoning", levels = c("Model size", "Reasoning")))

lev_cols <- setNames(c(PAL[["small"]], PAL[["large"]], PAL[["off"]], PAL[["on"]]), levels(nd$level_label))
lev_ltys <- setNames(c("22","solid","22","solid"), levels(nd$level_label))

p_levers <- ggplot(nd, aes(x_plot, err_rr, colour = level_label, linetype = level_label)) +
  geom_hline(yintercept = 1, colour = "grey55", linewidth = 0.35) +
  geom_rug(data = anchor, aes(x = x_plot), inherit.aes = FALSE, sides = "b",
           alpha = 0.45, length = unit(0.02, "npc"), colour = "grey40") +
  geom_line(linewidth = 1) +
  facet_grid(lever ~ feature_lab, scales = "free_x", switch = "x",
             labeller = labeller(feature_lab = label_parsed)) +
  scale_x_continuous(labels = function(b) ifelse(b > 1, pages_lab(b), b)) +
  scale_colour_manual(values = lev_cols, name = NULL) +
  scale_linetype_manual(values = lev_ltys, name = NULL) +
  scale_y_log10(breaks = c(1, 1.5, 2, 3, 5), labels = function(x) paste0(x, "x")) +
  labs(x = NULL, y = axlab("Times more likely to get the answer wrong than in English", "predicted"),
       title = "Reasoning narrows both gaps; scale is a different story",
       subtitle = paste("Flatter = smaller penalty; 1x = parity with English. Ticks mark the observed languages.",
                        "\nRisk ratios depend on the base rate they are measured against, so the size rows are not",
                        "directly\ncomparable: at 32B the English error rate is ~7%, inflating the ratio. The fitted",
                        "interaction,\nestimated on the odds scale, has scale REDUCING the resource penalty",
                        "(see the odds-scale figure).")) +
  theme_paper + theme(strip.placement = "outside")
ok(p_levers, "fig_levers", 7.4, 5.4)

## Odds-ratio companion. The main figure uses the risk ratio because that is what
## "times more likely to be wrong" means, but the risk ratio depends on the base
## rate: at 32B the English error rate is ~7%, so a given absolute drop produces a
## larger ratio than the same drop at 4B. Consequently the SIZE row reverses
## between the two scales -- on the odds scale (below, and in the fitted model)
## the 32B resource slope is the flatter one, matching resource x size = -0.25.
## Publish this alongside, or cite it in the caption, so the two do not read as a
## contradiction.
p_levers_or <- p_levers %+% nd + aes(y = err_or) +
  scale_y_log10(breaks = c(1, 2, 3, 5, 10, 20), labels = function(x) paste0(x, "x")) +
  labs(y = axlab("Odds of a wrong answer, relative to English", "predicted"),
       title = "Two levers (odds-ratio scale)",
       subtitle = paste("The scale the model is linear in: lines are straight and each interaction is exactly the",
                        "\ndifference in slope. Scale flattens the resource slope but not the distance slope;",
                        "reasoning flattens both."))
ok(p_levers_or, "fig_levers_odds", 7.4, 5.4)

p_levers_prob <- ggplot(nd, aes(x_plot, pred, colour = level_label, linetype = level_label)) +
  geom_rug(data = anchor, aes(x = x_plot), inherit.aes = FALSE, sides = "b",
           alpha = 0.45, length = unit(0.02, "npc"), colour = "grey40") +
  geom_line(linewidth = 1) +
  facet_grid(lever ~ feature_lab, scales = "free_x", switch = "x",
             labeller = labeller(feature_lab = label_parsed)) +
  scale_x_continuous(labels = function(b) ifelse(b > 1, pages_lab(b), b)) +
  scale_colour_manual(values = lev_cols, name = NULL) +
  scale_linetype_manual(values = lev_ltys, name = NULL) +
  scale_y_continuous(labels = percent) +
  labs(x = NULL, y = axlab("Accuracy", "predicted"),
       title = "Two levers, on the probability scale",
       subtitle = paste("Same fit as the previous figure. Note the probability scale re-expresses each",
                        "interaction through\nthe base rate, so slopes are not directly comparable across panels.")) +
  theme_paper + theme(strip.placement = "outside")
ok(p_levers_prob, "fig_levers_prob", 7.4, 5.4)


## =====================================================================
## 1b. THE SAME PENALTY, PRICED IN MODEL SIZE
## The fitted size coefficient gives an exchange rate: ~1 log-odds per
## doubling of parameters. So an observed gap can be quoted as "this
## language costs the equivalent of N doublings", which is the unit a
## practitioner actually budgets in. The GAP is observed; only the
## exchange rate is fitted.
## =====================================================================
beta_per_doubling <- bget(fit_main, "log_params_z") / sd(cells$log2_params)

obs_gap <- cells %>%
  group_by(language) %>%
  summarise(acc = sum(n_correct)/sum(n_total), .groups = "drop") %>%
  mutate(eng = acc[language == "eng"],
         err_or = ((1-acc)/acc)/((1-eng)/eng),
         doublings = log(err_or)/beta_per_doubling,
         param_mult = 2^doublings) %>%
  filter(language != "eng") %>%
  left_join(lang_meta, by = "language") %>%
  arrange(doublings) %>%
  mutate(language = factor(language, levels = language))

## Axis and labels must be in the SAME unit or they read as contradicting each
## other. Plot the parameter multiple directly, on a log2 axis so equal distances
## are equal doublings; 1x = parity with English.
p_cost <- ggplot(obs_gap, aes(param_mult, language)) +
  geom_vline(xintercept = 1, colour = "grey55", linewidth = 0.35) +
  geom_segment(aes(x = 1, xend = param_mult, yend = language), colour = "grey78", linewidth = 0.8) +
  geom_point(aes(colour = log10_common_crawl_pages), size = 3.4) +
  geom_text(aes(label = sprintf("%.1fx", param_mult)), hjust = -0.45, size = 2.9, colour = "grey30") +
  scale_colour_viridis_c(option = "mako", begin = 0.1, end = 0.85, direction = -1,
                         name = LEG_RES, breaks = RES_BREAKS, labels = pages_lab) +
  scale_x_continuous(trans = "log2", breaks = c(1, 1.5, 2, 3, 4),
                     labels = function(x) paste0(x, "x"),
                     expand = expansion(mult = c(0.02, 0.12))) +
  labs(x = axlab("Equivalent cost in model size", "parameter multiple, log scale"), y = NULL,
       title = "What each language costs, priced in model size",
       subtitle = paste("Working in Marathi costs about as much accuracy as shrinking the model by that factor.",
                        "\nGaps observed; exchange rate from the fitted size effect.")) +
  theme_paper + theme(legend.position = "right")
ok(p_cost, "fig_scale_cost", 7.2, 4.6)

cat(sprintf("  exchange rate: %.2f log-odds per doubling of parameters\n", beta_per_doubling))

## =====================================================================
## 2. MODEL-FREE COMPANION: the same two claims, no model fitted
## =====================================================================
gap_scale <- cells %>%
  group_by(model, language, log2_params) %>%
  summarise(acc = sum(n_correct)/sum(n_total), .groups = "drop") %>%
  group_by(model, log2_params) %>% mutate(eng = acc[language == "eng"]) %>% ungroup() %>%
  filter(language != "eng") %>%
  mutate(err_ratio = (1-acc)/(1-eng)) %>%   # risk ratio, matching the fitted figure
  left_join(lang_meta, by = "language")

pa <- ggplot(gap_scale, aes(2^log2_params, err_ratio)) +
  geom_hline(yintercept = 1, colour = "grey55", linewidth = 0.35) +
  geom_point(aes(colour = dbl_res(log10_common_crawl_pages)), alpha = 0.5, size = 1.4) +
  geom_smooth(method = "loess", formula = y ~ x, se = TRUE, colour = "grey15",
              fill = "grey80", linewidth = 0.8) +
  scale_x_log10() +
  scale_y_log10(breaks = c(0.5, 1, 2, 3, 5, 10), labels = function(x) paste0(x, "x")) +
  scale_colour_viridis_c(option = "mako", begin = 0.1, end = 0.85, direction = -1,
                         name = LEG_RES) +
  labs(x = "Parameters (B, log scale)", y = axlab("Times more likely to be wrong than in English", "observed"),
       title = "Scale narrows the gap", subtitle = "Each point: one model, one language") +
  theme_paper

paired <- cells %>% distinct(base_model, reasoning) %>% count(base_model) %>%
  filter(n == 2) %>% pull(base_model)
by_lang <- cells %>% filter(base_model %in% paired) %>%
  group_by(reasoning, language) %>% summarise(acc = sum(n_correct)/sum(n_total), .groups="drop")
eng_ref <- by_lang %>% filter(language == "eng") %>% select(reasoning, eng = acc)
reason_df <- by_lang %>% filter(language != "eng") %>% left_join(eng_ref, by = "reasoning") %>%
  mutate(err_ratio = (1-acc)/(1-eng)) %>%   # risk ratio
  group_by(language) %>% mutate(ord = err_ratio[reasoning=="off"]) %>% ungroup() %>%
  mutate(language = reorder(language, ord))

pb <- ggplot(reason_df, aes(err_ratio, language)) +
  geom_vline(xintercept = 1, colour = "grey55", linewidth = 0.35) +
  geom_line(aes(group = language), colour = "grey75", linewidth = 0.6) +
  geom_point(aes(colour = reasoning), size = 2.4) +
  scale_colour_manual(values = PAL[c("off","on")], name = NULL) +
  scale_x_log10(breaks = c(1, 1.5, 2, 3, 4), labels = function(x) paste0(x, "x")) +
  labs(x = axlab("Times more likely to be wrong than in English", "observed"), y = NULL,
       title = "Reasoning narrows it too",
       subtitle = "12 base models run both ways") +
  theme_paper

## Models that score 0% or 100% in a language give infinite / zero error-odds
## and cannot be shown on a log scale. Report how many, rather than dropping
## them silently.
n_drop <- sum(!is.finite(log(gap_scale$err_ratio)))
cat("  fig_levers_raw: dropped", n_drop, "of", nrow(gap_scale),
    "model x language points with 0% or 100% accuracy\n")

ok(pa, "fig_levers_raw_scale", 6.4, 4.2)
ok(pb, "fig_levers_raw_reasoning", 5.6, 4.4)

## =====================================================================
## 3. EFFECT SIZES, grouped by the level at which each predictor varies
## =====================================================================
labels <- c(log_params_z="Model size", log_resource_z="Resource level",
  typological_distance_z="Typological distance", relative_fertility_within_z="Relative fertility (within-language)",
  reasoningon="Reasoning on", `log_resource_z:log_params_z`="Resource x size",
  `typological_distance_z:log_params_z`="Typ. distance x size",
  `relative_fertility_within_z:log_params_z`="Rel. fertility x size",
  `log_resource_z:reasoningon`="Resource x reasoning",
  `typological_distance_z:reasoningon`="Typ. distance x reasoning",
  `relative_fertility_within_z:reasoningon`="Rel. fertility x reasoning",
  `log_params_z:reasoningon`="Size x reasoning")
strata <- c(log_params_z="Model-level\n(48 models)", reasoningon="Paired manipulation\n(12 base models)",
  `log_params_z:reasoningon`="Paired manipulation\n(12 base models)",
  log_resource_z="Language-level\n(16 languages)", typological_distance_z="Language-level\n(16 languages)",
  relative_fertility_within_z="Model x language\n(768 cells)", `log_resource_z:log_params_z`="Model x language\n(768 cells)",
  `typological_distance_z:log_params_z`="Model x language\n(768 cells)",
  `relative_fertility_within_z:log_params_z`="Model x language\n(768 cells)",
  `log_resource_z:reasoningon`="Model x language\n(768 cells)",
  `typological_distance_z:reasoningon`="Model x language\n(768 cells)",
  `relative_fertility_within_z:reasoningon`="Model x language\n(768 cells)")

canon <- function(x) sub("relative_fertility", "fertility", x, fixed = TRUE)
names(labels) <- canon(names(labels)); names(strata) <- canon(names(strata))
co <- as.data.frame(summary(fit_main)$coefficients)
co$term <- canon(rownames(co))
eff <- co %>% filter(term != "(Intercept)") %>%
  mutate(label = labels[term],
         stratum = factor(strata[term], levels = c("Paired manipulation\n(12 base models)",
             "Model x language\n(768 cells)", "Model-level\n(48 models)", "Language-level\n(16 languages)")),
         lo = Estimate - 1.96*`Std. Error`, hi = Estimate + 1.96*`Std. Error`,
         is_lever = factor(if_else(grepl("reasoning|params", term),
                                   "Lever (size / reasoning)", "Language feature"),
                           levels = c("Lever (size / reasoning)", "Language feature"))) %>%
  arrange(stratum, Estimate) %>% mutate(label = factor(label, levels = label))

p_eff <- ggplot(eff, aes(Estimate, label)) +
  geom_vline(xintercept = 0, colour = "grey55", linewidth = 0.35) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, linewidth = 0.7, colour = "grey35") +
  ## Filled vs hollow does the work here; hue alone was too weak at this mark
  ## size. Levers are the paper's actionable terms, so they get the solid mark.
  geom_point(aes(colour = is_lever, fill = is_lever), shape = 21, size = 2.8, stroke = 0.9) +
  scale_colour_manual(values = c(`Lever (size / reasoning)` = "#A8501E",
                                 `Language feature` = "grey35"), name = NULL) +
  scale_fill_manual(values = c(`Lever (size / reasoning)` = "#A8501E",
                               `Language feature` = "white"), name = NULL) +
  facet_grid(stratum ~ ., scales = "free_y", space = "free_y", switch = "y") +
  labs(x = axlab("Effect size", "log-odds per SD of predictor"), y = NULL,
       title = "Fitted effects, grouped by what identifies them",
       subtitle = paste("95% Wald intervals. Reasoning is a within-model manipulation;",
                        "\nlanguage-level terms rest on 16 observational units.")) +
  theme_paper +
  theme(strip.placement = "outside", strip.text.y.left = element_text(angle = 0, hjust = 0, size = 7.5),
        panel.spacing = unit(0.5, "lines"))
ok(p_eff, "fig_effects", 7.4, 5.4)

## =====================================================================
## 4-5. THE LADDER
## =====================================================================
rungs <- cells %>% group_by(language) %>%
  summarise(`Operands\nread` = sum(n_lhs_ret)/sum(n_lhs),
            `Intermediates\ncomputed` = sum(n_rhs_ret)/sum(n_rhs),
            `Final\nanswer` = sum(n_correct)/sum(n_total), .groups = "drop") %>%
  pivot_longer(-language, names_to = "stage", values_to = "rate") %>%
  mutate(stage = factor(stage, levels = c("Operands\nread","Intermediates\ncomputed","Final\nanswer"))) %>%
  left_join(lang_meta, by = "language")

p_ladder <- ggplot(rungs, aes(stage, rate, group = language, colour = log10_common_crawl_pages)) +
  geom_line(linewidth = 0.8, alpha = 0.9) + geom_point(size = 1.8) +
  geom_text(data = filter(rungs, stage == "Final\nanswer"), aes(label = language),
            hjust = -0.3, size = 2.7, show.legend = FALSE) +
  ## mako: sequential, monotone in lightness (so the ordering survives greyscale
  ## and colour-vision deficiency) and it avoids the pale yellow end of viridis
  ## that disappears against a white panel. Darker = more resource.
  scale_colour_viridis_c(option = "mako", begin = 0.1, end = 0.85, direction = -1,
                         name = LEG_RES, breaks = RES_BREAKS, labels = pages_lab) +
  scale_y_continuous(labels = percent) + expand_limits(x = 3.7) +
  labs(x = NULL, y = axlab("Success rate", "observed"),
       title = "The gap widens with depth into the solution",
       subtitle = "Marginal rates, no model fitted") +
  theme_paper + theme(legend.position = "right")
ok(p_ladder, "fig_ladder", 7.0, 4.4)

lad_terms <- c(relative_fertility_within_z="Relative fertility (within)", typological_distance_z="Typological distance",
               log_resource_z="Resource level")
lad <- bind_rows(lapply(names(lad_terms), function(tm)
  data.frame(term = lad_terms[[tm]],
             stage = c("Operands\nread","Intermediates\ncomputed","Final\nanswer"),
             est = c(bget(fit_lhs, tm), bget(fit_rhs, tm), bget(fit_main, tm))))) %>%
  mutate(stage = factor(stage, levels = c("Operands\nread","Intermediates\ncomputed","Final\nanswer")))

p_lad_fit <- ggplot(lad, aes(stage, abs(est), group = term, colour = term)) +
  geom_line(linewidth = 0.9) + geom_point(size = 2.4) +
  scale_colour_manual(values = c("#2A6F7F","#B4656F","#7D6B9E"), name = NULL) +
  labs(x = NULL, y = "|log-odds| per SD",
       title = "Every fitted effect grows with depth",
       subtitle = "Which a difference in outcome scale would also produce - hence the adjusted analysis") +
  theme_paper
ok(p_lad_fit, "fig_ladder_fitted", 6.0, 3.8)

## =====================================================================
## 6. DESIGN SPACE - what is and is not confounded
## =====================================================================
acc_by_lang <- cells %>% group_by(language) %>%
  summarise(acc = sum(n_correct)/sum(n_total), .groups = "drop")
r_conf <- with(lang_meta, cor(log10_common_crawl_pages, typological_distance_from_english))

p_design <- lang_meta %>% left_join(acc_by_lang, by = "language") %>%
  ggplot(aes(dbl_res(log10_common_crawl_pages), typological_distance_from_english)) +
  geom_point(aes(fill = acc, shape = script), size = 5, colour = "grey25", stroke = 0.4) +
  geom_text(aes(label = language), nudge_y = 0.021, size = 3, colour = "grey20") +
  scale_shape_manual(values = c(Latin = 21, `non-Latin` = 24), name = NULL) +
  scale_x_continuous(breaks = RES_BREAKS, labels = pages_lab) +
  scale_fill_viridis_c(option = "mako", begin = 0.12, end = 0.88, direction = -1,
                       labels = percent, name = "mean accuracy") +
  labs(x = axlab("Resource level", "Common Crawl pages"),
       y = axlab("Typological distance", "URIEL syntactic distance from English"),
       title = "The language design space",
       subtitle = sprintf("Resource varies within family and script; the two axes are only weakly related (r = %.2f)", r_conf)) +
  theme_paper + theme(legend.position = "right", legend.box = "vertical")
ok(p_design, "fig_design_space", 7.2, 5.0)

## =====================================================================
## 7. ADEQUACY of the feature set (supporting, not the headline)
## =====================================================================
vc <- function(f,g) attr(VarCorr(f)[[g]], "stddev")[[1]]
vardf <- tibble(
  component = c("Between-language\n(why languages differ)","Model-by-language\n(why models differ on a language)"),
  baseline = c(vc(fit_base,"language")^2, vc(fit_base,"model:language")^2),
  full     = c(vc(fit_main,"language")^2, vc(fit_main,"model:language")^2)) %>%
  mutate(explained = 1 - full/baseline)
combined <- 1 - sum(vardf$full)/sum(vardf$baseline)

p_var <- vardf %>%
  pivot_longer(c(baseline, full), names_to = "m", values_to = "variance") %>%
  mutate(m = factor(if_else(m=="baseline","Size only","+ language features"),
                    levels = c("Size only","+ language features"))) %>%
  ggplot(aes(variance, component, fill = m)) +
  geom_col(position = position_dodge(0.65), width = 0.6) +
  geom_text(data = vardf, aes(x = baseline, y = component, label = sprintf("%.0f%%", 100*explained)),
            inherit.aes = FALSE, hjust = -0.25, size = 3.2, colour = "grey25") +
  scale_fill_manual(values = c(`Size only`="grey72", `+ language features`="#2A6F7F"), name = NULL) +
  expand_limits(x = 0.9) +
  labs(x = "Variance component (log-odds scale)", y = NULL,
       title = sprintf("How much of transfer the features account for (~%.0f%% combined)", 100*combined),
       subtitle = "An adequacy check on this 16-language sample, not a claim about what causes transfer") +
  theme_paper
ok(p_var, "fig_variance", 7.0, 3.6)

cat("\ncombined variance explained:", round(100*combined,1), "%\n")
cat("resource-typology correlation across languages:", round(r_conf,3), "\n")
