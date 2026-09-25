## Draft results figures. Reads fitted models from the knitr cache (does not
## refit) and prepared cells from ../../artifacts/analysis/model_cells.rds.
##
## The ten figures the paper uses, in the order they are built:
##   fig_levers_odds          Fig. 3   the four lever x language-feature interactions
##   fig_scale_cost           Fig. 14  each language's gap priced in model size
##   fig_effects              Fig. 2   all fitted effects, by inferential stratum
##   fig_ladder               Fig. 5   observed success rate by solution stage
##   fig_ladder_fitted        Fig. 16  fitted effects by stage
##   fig_design_space         Fig. 15  language selection / confound structure
##   fig_predict_language     Fig. 6   forecasting a held-out language
##   fig_predict_model_language Fig. 7 forecasting a model in a held-out language
##   fig_predict_by_language  Fig. 13  the same, decomposed per language
##   fig_symbolic_penalty     Fig. 4   symbolic penalty by language

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
## lever encodes less -> more of it. Navy and forest green, sampled from the
## equivalence figure so the paper's figures share a palette.
PAL <- c(small = "#92A9C2", large = "#1F3A5F",   # model size  (navy)
         off   = "#8FBBA1", on    = "#2E6B48")   # reasoning   (green)

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

## Spelled-out names for axis labels. Bokmal is shortened to "Norwegian": the
## macrolanguage distinction matters for the resource counts, not for a tick.
LANG_NAME <- c(ara="Arabic", dan="Danish", deu="German", eng="English", est="Estonian",
               fra="French", hin="Hindi", isl="Icelandic", ita="Italian", jpn="Japanese",
               mar="Marathi", nld="Dutch", nob="Norwegian", rus="Russian",
               ukr="Ukrainian", zho="Chinese")
lang_label <- function(x) sprintf("%s (%s)", LANG_NAME[as.character(x)], x)

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

## Odds-ratio companion. The main figure uses the risk ratio because that is what
## "times more likely to be wrong" means, but the risk ratio depends on the base
## rate: at 32B the English error rate is ~7%, so a given absolute drop produces a
## larger ratio than the same drop at 4B. Consequently the SIZE panels reverse
## between the two scales -- on the odds scale (below, and in the fitted model)
## the 32B resource slope is the flatter one, matching resource x size = -0.25.
##
## Laid out as a single row of four panels to save vertical space. Lines are
## labelled directly at their left end instead of via a legend, which removes a
## whole legend block and makes each panel readable on its own; the takeaway is a
## single header line rather than a band above each row.
nd_row <- nd %>%
  mutate(panel = factor(
    paste0(ifelse(lever == "Model size", "Model size", "Reasoning"), " \u00d7 ",
           ifelse(grepl("Resource", feature_lab), "resource", "distance")),
    levels = c("Model size \u00d7 resource", "Model size \u00d7 distance",
               "Reasoning \u00d7 resource", "Reasoning \u00d7 distance")))

## Label each line where its panel's curves are furthest apart: the resource
## panels diverge at the low end, the distance panels converge to 1x at zero and
## diverge at the high end, so labelling both on the same side would stack them.
lab_row <- nd_row %>%
  mutate(side = if_else(grepl("resource", panel), "left", "right")) %>%
  group_by(panel, level_label, side) %>%
  slice(which.min(if (first(side) == "left") x_plot else -x_plot)) %>%
  ungroup()

anchor_row <- bind_rows(
  anchor %>% filter(feature_lab == LAB_RES) %>%
    transmute(x_plot, panel = "Model size \u00d7 resource"),
  anchor %>% filter(feature_lab == LAB_TYP) %>%
    transmute(x_plot, panel = "Model size \u00d7 distance"),
  anchor %>% filter(feature_lab == LAB_RES) %>%
    transmute(x_plot, panel = "Reasoning \u00d7 resource"),
  anchor %>% filter(feature_lab == LAB_TYP) %>%
    transmute(x_plot, panel = "Reasoning \u00d7 distance")) %>%
  mutate(panel = factor(panel, levels = levels(nd_row$panel)))

lev_cols_row <- setNames(c(PAL[["small"]], PAL[["large"]], PAL[["off"]], PAL[["on"]]),
                         c(paste0(SMALL, "B"), paste0(LARGE, "B"),
                           "reasoning off", "reasoning on"))

## Every curve is expressed relative to English, so all four pass through exactly
## 1x at English's own feature values -- the far right of the resource panels
## (English is the highest-resource language) and the far left of the distance
## panels (distance from English is 0 by definition). Marking that point makes
## the reference visible instead of leaving it implied by the axis label.
eng_point <- tibble::tibble(
  panel = factor(levels(nd_row$panel), levels = levels(nd_row$panel)),
  x_plot = c(dbl_res(eng_res), 0, dbl_res(eng_res), 0),
  y = 1)

## The four panels do not share an x variable, so no single axis title serves
## them and facet_wrap offers no per-panel one. The row is therefore assembled
## from four plots with patchwork. y is held to a common range across panels so
## the leftmost axis labels the whole row -- which is what the faceted version
## did, and what makes the panels comparable at all.
X_TITLE <- c("Common Crawl pages", "URIEL syntactic distance",
             "Common Crawl pages", "URIEL syntactic distance")

lever_row <- function(yvar, yscale, ylab, file, h) {
  pnls <- levels(nd_row$panel)
  ps <- lapply(seq_along(pnls), function(i) {
    P <- pnls[i]
    g <- ggplot(filter(nd_row, panel == P),
                aes(x_plot, .data[[yvar]], colour = level_label, linetype = level_label)) +
      geom_rug(data = filter(anchor_row, panel == P), aes(x = x_plot), inherit.aes = FALSE,
               sides = "b", alpha = 0.4, length = unit(0.02, "npc"), colour = "grey45") +
      geom_line(linewidth = 0.9) +
      geom_point(data = filter(eng_point, panel == P), aes(x_plot, y),
                 inherit.aes = FALSE, size = 1.9, colour = "grey20") +
      ## Below the point, not beside it: every curve converges ON this point, so
      ## the space to its left and right is exactly where the lines are. The
      ## y-scale's lower expansion is opened up to make room underneath.
      geom_text(data = filter(eng_point, panel == P), aes(x_plot, y, label = "English"),
                inherit.aes = FALSE, size = 2.5, colour = "grey20",
                hjust = if (grepl("resource", P)) 0.9 else 0.1, vjust = 2.1) +
      ggrepel::geom_text_repel(data = filter(lab_row, panel == P),
                               aes(label = level_label, hjust = if_else(side == "left", 0, 1)),
                               size = 2.7, direction = "y", seed = 1, box.padding = 0.35,
                               point.padding = 0.2, segment.colour = NA, show.legend = FALSE) +
      scale_x_continuous(expand = expansion(mult = 0.12),
                         labels = function(b) ifelse(b > 1, pages_lab(b), b)) +
      scale_colour_manual(values = lev_cols_row, guide = "none") +
      scale_linetype_manual(values = setNames(c("22", "solid", "22", "solid"),
                                              names(lev_cols_row)), guide = "none") +
      yscale +
      labs(x = X_TITLE[i], y = if (i == 1) ylab else NULL, title = P) +
      theme_paper +
      theme(plot.title = element_text(face = "bold", size = 8.5, hjust = 0.5,
                                      margin = margin(b = 3)),
            axis.title.x = element_text(size = 8, colour = "grey30"),
            plot.margin = margin(2, 4, 2, 2))
    if (i > 1) g <- g + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
    g
  })
  ok(patchwork::wrap_plots(ps, nrow = 1), file, 8.6, h)
}

lever_row("err_or",
          scale_y_log10(breaks = c(1, 2, 5, 10, 20), labels = function(x) paste0(x, "x"),
                        limits = range(nd_row$err_or),
                        expand = expansion(mult = c(0.13, 0.13))),
          axlab("Odds of a wrong answer", "relative to English, predicted"),
          "fig_levers_odds", 3.2)

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
       title = NULL, subtitle = NULL) +
  theme_paper +
  theme(legend.position = "right", axis.text = element_text(size = 8),
        plot.margin = margin(2, 3, 2, 2))
ok(p_cost, "fig_scale_cost", 6.6, 3.8)

cat(sprintf("  exchange rate: %.2f log-odds per doubling of parameters\n", beta_per_doubling))

## =====================================================================
## 3. EFFECT SIZES, grouped by the level at which each predictor varies
## =====================================================================
labels <- c(log_params_z="Model size", log_resource_z="Resource level",
  typological_distance_z="Typological distance", relative_fertility_within_z="Relative fertility",
  reasoningon="Reasoning on", `log_resource_z:log_params_z`="Resource x size",
  `typological_distance_z:log_params_z`="Distance x size",
  `relative_fertility_within_z:log_params_z`="Fertility x size",
  `log_resource_z:reasoningon`="Resource x reasoning",
  `typological_distance_z:reasoningon`="Distance x reasoning",
  `relative_fertility_within_z:reasoningon`="Fertility x reasoning",
  `log_params_z:reasoningon`="Size x reasoning")
## Stratum sizes are read off the data rather than written in: they moved once
## already when nob was dropped (16 languages / 768 cells -> 15 / 720), and a
## hardcoded label that silently goes stale is worse than no label.
N_LANG <- n_distinct(cells$language)
N_MODEL <- n_distinct(cells$model)
N_CELL <- n_distinct(paste(cells$model, cells$language))
## base models run BOTH ways -- the stratum the reasoning terms are tested on
N_BASE <- cells %>% distinct(base_model, reasoning) %>% count(base_model) %>%
  filter(n > 1) %>% nrow()
S_LANG  <- sprintf("Language-level\n(%d languages)", N_LANG)
S_ML    <- sprintf("Model x language\n(%d cells)", N_CELL)
S_MODEL <- sprintf("Model-level\n(%d models)", N_MODEL)
S_PAIR  <- sprintf("Paired manipulation\n(%d base models)", N_BASE)

strata <- c(log_params_z=S_MODEL, reasoningon=S_PAIR,
  `log_params_z:reasoningon`=S_PAIR,
  log_resource_z=S_LANG, typological_distance_z=S_LANG,
  relative_fertility_within_z=S_ML, `log_resource_z:log_params_z`=S_ML,
  `typological_distance_z:log_params_z`=S_ML,
  `relative_fertility_within_z:log_params_z`=S_ML,
  `log_resource_z:reasoningon`=S_ML,
  `typological_distance_z:reasoningon`=S_ML,
  `relative_fertility_within_z:reasoningon`=S_ML)

canon <- function(x) sub("relative_fertility", "fertility", x, fixed = TRUE)
names(labels) <- canon(names(labels)); names(strata) <- canon(names(strata))
co <- as.data.frame(summary(fit_main)$coefficients)
co$term <- canon(rownames(co))
eff <- co %>% filter(term != "(Intercept)") %>%
  mutate(label = labels[term],
         stratum = factor(strata[term], levels = c(S_PAIR, S_ML, S_MODEL, S_LANG)),
         lo = Estimate - 1.96*`Std. Error`, hi = Estimate + 1.96*`Std. Error`,
         is_lever = factor(if_else(grepl("reasoning|params", term),
                                   "Lever (size / reasoning)", "Language feature"),
                           levels = c("Lever (size / reasoning)", "Language feature"))) %>%
  arrange(stratum, Estimate) %>% mutate(label = factor(label, levels = label))

p_eff <- ggplot(eff, aes(Estimate, label)) +
  geom_vline(xintercept = 0, colour = "grey55", linewidth = 0.35) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, linewidth = 0.6, colour = "grey35") +
  ## Filled vs hollow does the work here; hue alone was too weak at this mark
  ## size. Levers are the paper's actionable terms, so they get the solid mark.
  geom_point(size = 2.4, colour = "black") +
  facet_grid(stratum ~ ., scales = "free_y", space = "free_y", switch = "y") +
  labs(x = "Effect size", y = NULL, title = NULL, subtitle = NULL) +
  theme_paper +
  ## Taller than the other figures on purpose: 13 terms on a shared x axis need
  ## the vertical room, and the stratum strips are the figure's argument rather
  ## than decoration, so they are set at readable size rather than squeezed.
  theme(strip.placement = "outside",
        strip.text.y.left = element_text(angle = 0, hjust = 0, size = 8.5, lineheight = 0.95),
        axis.text.y = element_text(size = 9.5),
        axis.text.x = element_text(size = 9),
        axis.title.x = element_text(size = 10),
        panel.spacing = unit(0.5, "lines"),
        plot.margin = margin(3, 5, 3, 3))
ok(p_eff, "fig_effects", 7.0, 4.6)

## =====================================================================
## 4-5. THE LADDER
## =====================================================================
rungs <- cells %>% group_by(language) %>%
  summarise(`Operands read` = sum(n_lhs_ret)/sum(n_lhs),
            `Intermediates` = sum(n_rhs_ret)/sum(n_rhs),
            `Final answer` = sum(n_correct)/sum(n_total), .groups = "drop") %>%
  pivot_longer(-language, names_to = "stage", values_to = "rate") %>%
  mutate(stage = factor(stage, levels = c("Operands read","Intermediates","Final answer"))) %>%
  left_join(lang_meta, by = "language")

p_ladder <- ggplot(rungs, aes(stage, rate, group = language, colour = log10_common_crawl_pages)) +
  geom_line(linewidth = 0.8, alpha = 0.9) + geom_point(size = 1.8) +
  ## Repelled: rus/nob and zho/nld collide in the middle of the final-answer band.
  ggrepel::geom_text_repel(
    data = filter(rungs, stage == "Final answer"), aes(label = language),
    size = 2.6, show.legend = FALSE, direction = "y", hjust = 0,
    nudge_x = 0.08, min.segment.length = 0.25, segment.size = 0.25,
    segment.colour = "grey70", box.padding = 0.10, max.overlaps = Inf, seed = 1) +
  ## mako: sequential, monotone in lightness (so the ordering survives greyscale
  ## and colour-vision deficiency) and it avoids the pale yellow end of viridis
  ## that disappears against a white panel. Darker = more resource.
  scale_colour_viridis_c(option = "mako", begin = 0.1, end = 0.85, direction = -1,
                         name = "Resource level", breaks = RES_BREAKS, labels = pages_lab) +
  scale_y_continuous(labels = percent) + expand_limits(x = 3.7) +
  ## Takeaway on the plot rather than in a title, in the empty lower-left region
  ## the fan-out leaves behind.
  annotate("text", x = 0.6, y = 0.455, hjust = 0, vjust = 1, size = 3.5, colour = "grey10",
           lineheight = 0.95, label = "The gap widens with\ndepth of solution") +
  labs(x = NULL, y = axlab("Success rate", "observed"), title = NULL, subtitle = NULL) +
  ## Sized for half-column placement: a right-hand legend would eat ~30% of the
  ## width at 0.48\linewidth, so it goes above as a horizontal bar. Above rather
  ## than below also lines this figure's caption up with its side-by-side partner.
  theme_paper +
  theme(legend.position = "top",
        legend.key.width = unit(1.1, "cm"), legend.key.height = unit(0.22, "cm"),
        legend.title = element_text(size = 8), legend.text = element_text(size = 7.5),
        legend.margin = margin(0, 0, 0, 0), legend.box.spacing = unit(3, "pt"),
        axis.text = element_text(size = 8),
        plot.margin = margin(2, 3, 2, 2))
ok(p_ladder, "fig_ladder", 4.1, 3.5)

lad_terms <- c(relative_fertility_within_z="Relative fertility (within)", typological_distance_z="Typological distance",
               log_resource_z="Resource level")
STAGES <- c("Operands read", "Intermediates", "Final answer")
lad <- bind_rows(lapply(names(lad_terms), function(tm)
  data.frame(term = lad_terms[[tm]], stage = STAGES,
             est = c(bget(fit_lhs, tm), bget(fit_rhs, tm), bget(fit_main, tm))))) %>%
  ## Ordered by effect magnitude so the mako ramp encodes it: resource is the
  ## largest and darkest, fertility the smallest and lightest.
  mutate(stage = factor(stage, levels = STAGES),
         term = factor(term, levels = c("Resource level", "Typological distance",
                                        "Relative fertility (within)")))

## mako, the same family fig_ladder, fig_scale_cost and fig_design_space use for
## anything language-level, sampled at three points. This is the appendix
## companion to fig_ladder, so it reads as that figure's sibling.

p_lad_fit <- ggplot(lad, aes(stage, abs(est), group = term, colour = term)) +
  geom_line(linewidth = 0.9) + geom_point(size = 2.2) +
  ## Direct labels rather than a legend: three lines, and a legend row costs as
  ## much height as the panel gains from being compressed.
  ggrepel::geom_text_repel(
    data = filter(lad, stage == "Final answer"), aes(label = term),
    size = 2.8, hjust = 0, direction = "y", nudge_x = 0.06,
    min.segment.length = Inf, box.padding = 0.15, seed = 1, show.legend = FALSE) +
  scale_colour_viridis_d(option = "mako", begin = 0.18, end = 0.68,
                         direction = 1, guide = "none") +
  scale_x_discrete(expand = expansion(add = c(0.12, 1.45))) +
  labs(x = NULL, y = axlab("|log-odds| per SD", "fitted effect at each rung"),
       title = NULL, subtitle = NULL) +
  theme_paper + theme(plot.margin = margin(4, 3, 2, 2))
ok(p_lad_fit, "fig_ladder_fitted", 5.2, 2.9)

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
       ## Title and subtitle go in the LaTeX caption; the r is still printed
       ## below so it can be quoted there.
       title = NULL, subtitle = NULL) +
  theme_paper + theme(legend.position = "right", legend.box = "vertical")
ok(p_design, "fig_design_space", 7.2, 4.7)
cat(sprintf("  design space: corr(resource, typological distance) = %.2f\n", r_conf))

## =====================================================================
## 7. ADEQUACY of the feature set (numbers only; the figure was cut)
## =====================================================================
vc <- function(f,g) attr(VarCorr(f)[[g]], "stddev")[[1]]
vardf <- tibble(
  component = c("Between-language\n(why languages differ)","Model-by-language\n(why models differ on a language)"),
  baseline = c(vc(fit_base,"language")^2, vc(fit_base,"model:language")^2),
  full     = c(vc(fit_main,"language")^2, vc(fit_main,"model:language")^2)) %>%
  mutate(explained = 1 - full/baseline)
combined <- 1 - sum(vardf$full)/sum(vardf$baseline)

## fig_variance removed: the figure is cut from the paper and the two ratios it
## showed are quoted in the prose, so only the numbers are computed here.

cat("\nvariance explained: between-language", sprintf("%.0f%%", 100*vardf$explained[1]),
    "| model-by-language", sprintf("%.0f%%", 100*vardf$explained[2]),
    "| combined", sprintf("%.1f%%", 100*combined), "\n")
cat("resource-typology correlation across languages:", round(r_conf,3), "\n")

## =====================================================================
## 8. PREDICTING AN UNEVALUATED LANGUAGE (leave-one-language-out)
## Produced by predict_new_language.R. Every predictor is obtainable without
## running the model on the language, so this is a forecast, not a fit.
## =====================================================================
lolo_path <- "../../artifacts/analysis/lolo_predictions.csv"
if (file.exists(lolo_path)) {
  lolo <- read.csv(lolo_path)

  by_lang <- lolo %>%
    group_by(language) %>%
    summarise(observed = weighted.mean(observed, n_total),
              predicted = weighted.mean(features, n_total),
              baseline = weighted.mean(model_mean, n_total), .groups = "drop")

  rng <- range(c(by_lang$observed, by_lang$predicted))
  p_pred <- ggplot(by_lang, aes(observed, predicted)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey60", linewidth = 0.35) +
    geom_segment(aes(xend = observed, yend = observed), colour = "grey80", linewidth = 0.5) +
    geom_point(size = 2.8, colour = "#14425C") +
    geom_text(aes(label = language), hjust = -0.4, size = 2.7, colour = "grey30") +
    scale_x_continuous(labels = percent) + scale_y_continuous(labels = percent) +
    coord_equal(xlim = rng + c(-0.02, 0.05), ylim = rng + c(-0.02, 0.05)) +
    ## Estimates in the panel rather than a subtitle, so the figure carries them
    ## when it is read apart from the caption.
    annotate("text", x = rng[1] - 0.015, y = rng[2] + 0.045, hjust = 0, vjust = 1,
             size = 3.4, colour = "grey15", lineheight = 0.95,
             label = sprintf("MAE %.1f pts\nr = %.2f",
                             100 * mean(abs(by_lang$predicted - by_lang$observed)),
                             cor(by_lang$predicted, by_lang$observed))) +
    labs(x = axlab("Observed accuracy", "held-out language"),
         y = axlab("Predicted accuracy", "fitted without that language"),
         title = NULL, subtitle = NULL) +
    theme_paper
  ok(p_pred, "fig_predict_language", 6.2, 5.6)

  ## fig_predict_error removed: the MAE comparison it showed is now stated in
  ## the prose, and nothing referenced the figure.


  cat("\nLOLO language-level MAE:",
      round(100 * mean(abs(by_lang$predicted - by_lang$observed)), 2), "points\n")
} else cat("\n(lolo_predictions.csv not found; skipping prediction figures)\n")

## =====================================================================
## 9. (model, language) PREDICTED vs OBSERVED
## The decision-relevant target: how well will THIS model do on THIS
## unevaluated language. Faceting by language shows the two questions
## separately -- vertical spread is error in the level, within-panel
## ordering is whether we would pick the right model.
## =====================================================================
if (file.exists(lolo_path)) {
  ml <- lolo %>%
    group_by(language, model) %>%
    summarise(observed = weighted.mean(observed, n_total),
              predicted = weighted.mean(features, n_total),
              baseline = weighted.mean(model_mean, n_total),
              params = first(2^0), .groups = "drop") %>%
    left_join(cells %>% distinct(model, log2_params, language) %>%
                group_by(model) %>% slice(1) %>% ungroup() %>% select(model, log2_params),
              by = "model")

  lims <- range(c(ml$observed, ml$predicted))
  p_ml <- ggplot(ml, aes(observed, predicted)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey60", linewidth = 0.3) +
    geom_point(aes(colour = log2_params), alpha = 0.75, size = 1.5) +
    scale_colour_viridis_c(option = "mako", begin = 0.1, end = 0.85, direction = -1,
                           name = axlab("Model size", "log2 parameters (B)")) +
    scale_x_continuous(labels = percent) + scale_y_continuous(labels = percent) +
    coord_equal(xlim = lims, ylim = lims) +
    annotate("text", x = lims[1], y = lims[2], hjust = 0, vjust = 1,
             size = 3.4, colour = "grey15", lineheight = 0.95,
             label = sprintf("MAE %.1f pts\nr = %.2f",
                             100 * mean(abs(ml$predicted - ml$observed)),
                             cor(ml$predicted, ml$observed))) +
    labs(x = axlab("Observed accuracy", "held-out language"),
         y = axlab("Predicted accuracy", "fitted without that language"),
         title = NULL, subtitle = NULL) +
    theme_paper + theme(legend.position = "right")
  ok(p_ml, "fig_predict_model_language", 6.6, 5.4)

  ## Per-language panels: the level can be off while the ordering is right.
  p_ml_facet <- ggplot(ml, aes(observed, predicted)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey65", linewidth = 0.3) +
    geom_point(aes(colour = log2_params), alpha = 0.8, size = 1) +
    facet_wrap(~language, ncol = 4) +
    scale_colour_viridis_c(option = "mako", begin = 0.1, end = 0.85, direction = -1,
                           name = axlab("Model size", "log2 parameters (B)")) +
    scale_x_continuous(labels = percent, breaks = c(0.25, 0.75)) +
    scale_y_continuous(labels = percent, breaks = c(0.25, 0.75)) +
    geom_text(data = ml %>% group_by(language) %>%
                summarise(mae = 100 * mean(abs(predicted - observed)), .groups = "drop"),
              aes(x = 0.02, y = 0.98, label = sprintf("%.1f", mae)),
              inherit.aes = FALSE, hjust = 0, vjust = 1, size = 2.6, colour = "grey30") +
    labs(x = axlab("Observed accuracy", "held-out language"),
         y = axlab("Predicted accuracy", "fitted without that language"),
         title = NULL, subtitle = NULL) +
    theme_paper + theme(legend.position = "right", panel.spacing = unit(0.4, "lines"))
  ok(p_ml_facet, "fig_predict_by_language", 8.2, 7.0)
}

## =====================================================================
## 10. THE SYMBOLIC PENALTY, BY LANGUAGE
## Fig. 2 (original vs synthetic scatter) shows that the penalty exists;
## this shows how it differs across languages, which the scatter cannot,
## since language is not encoded there.
## =====================================================================
sym_path <- "../../artifacts/transfer_tables/analysis.parquet"
if (file.exists(sym_path)) {
  sym <- nanoparquet::read_parquet(sym_path) %>%
    filter(grepl("@refs/pr/16/", task, fixed = TRUE),
           ## nob is excluded here for the same reason as in transfer_analysis.Rmd:
           ## it is not one of the benchmark's 15 languages.
           !language %in% c("eng_metric", "uncorrected_isl", "nob"),
           model != "gpt-5.4-nano") %>%
    group_by(model, language, split) %>%
    summarise(acc = mean(correct), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = split, values_from = acc) %>%
    filter(!is.na(original), !is.na(synthetic)) %>%
    mutate(gap = 100 * (original - synthetic))

  by_lang <- sym %>%
    group_by(language) %>%
    summarise(mean = mean(gap), se = sd(gap) / sqrt(n()), .groups = "drop") %>%
    mutate(lo = mean - 1.96 * se, hi = mean + 1.96 * se)

  ## Overall row: the mean of the 16 language means, with a CI from the spread
  ## ACROSS languages rather than across the 768 model x language pairs. Language
  ## is the unit that varies here, so pooling the pairs would understate the
  ## interval considerably.
  n_l <- nrow(by_lang)
  se_l <- sd(by_lang$mean) / sqrt(n_l)
  tcrit <- qt(0.975, n_l - 1)
  overall_row <- tibble(language = "All languages",
                        mean = mean(by_lang$mean),
                        se = se_l,
                        lo = mean(by_lang$mean) - tcrit * se_l,
                        hi = mean(by_lang$mean) + tcrit * se_l)

  plot_df <- bind_rows(by_lang %>% mutate(language = as.character(language)), overall_row) %>%
    mutate(is_overall = language == "All languages",
           language = factor(language,
             levels = c("All languages",
                        by_lang$language[order(by_lang$mean)] %>% as.character())))

  p_sym <- ggplot(plot_df, aes(mean, language)) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.35) +
    ## Rule separating the summary row from the per-language rows.
    geom_hline(yintercept = 1.5, colour = "grey75", linewidth = 0.35) +
    ## Uniform weight: the separating rule already marks the summary row, and a
    ## heavier mark would read as a stronger estimate rather than a different one.
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, linewidth = 0.6, colour = "grey40") +
    geom_point(size = 1.9, colour = "grey15") +
    scale_x_continuous(breaks = seq(0, 6, 2)) +
    scale_y_discrete(expand = expansion(add = c(0.8, 0.6)),
                     labels = function(x) if_else(x == "All languages", x, lang_label(x))) +
    labs(x = axlab("Symbolic penalty", "accuracy points lost on symbolic variants"),
         y = NULL) +
    theme_paper +
    theme(axis.text = element_text(size = 8), plot.margin = margin(2, 3, 2, 2))
  ok(p_sym, "fig_symbolic_penalty", 4.3, 3.5)

  cat("\nsymbolic penalty:", sprintf("%.2f pp [%.2f, %.2f] overall | per-language range %.2f - %.2f\n",
      overall_row$mean, overall_row$lo, overall_row$hi,
      min(by_lang$mean), max(by_lang$mean)))
}
