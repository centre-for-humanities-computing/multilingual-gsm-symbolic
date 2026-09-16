# Transfer analysis

Statistical analysis behind Sections 4 and the appendices: the GLMM of
cross-lingual transfer, the symbolic-penalty model, the leave-one-language-out
forecast, and every figure generated in R.

## Language set

The benchmark's **15** languages: `ara dan deu eng est fra hin isl ita jpn mar
nld rus ukr zho`. Three labels in the parquet are deliberately excluded in
`transfer_analysis.Rmd` (`EXCLUDED_LANGUAGES`):

| excluded | why |
|---|---|
| `eng_metric` | English with metric units — not a language, and has no typological or resource features. Used only for the localisation check (Appendix L). |
| `uncorrected_isl` | machine-translated Icelandic before human correction. Used only for the translation-quality check (Appendix K). |
| `nob` | not one of the benchmark's 15 languages and never validated. Present in the parquet; earlier fits included it by mistake, which is why they reported 16 languages against the paper's 15. |

## Run order

Each step depends on the one above it. Everything whose output is meant to be
*read* is an `.Rmd` that knits to HTML with its fits cached; only the figure
script is a plain `.R`, since it writes files rather than answers.

```sh
# 0. upstream: builds analysis.parquet from the eval logs
python ../collect_transfer_tables.py

# 1. main model, solution ladder, resource x distance and script ablations.
#    Writes model_cells.rds, which every document below reads.
Rscript -e 'rmarkdown::render("transfer_analysis.Rmd")'

# 2. symbolic vs original penalty (Appendix C.1)
Rscript -e 'rmarkdown::render("symbolic_penalty.Rmd")'

# 3. forecasting an unseen language, with and without k evaluated templates
#    (Section 4.4, Appendix E). 15 LOLO folds then 225 refits: several hours.
Rscript -e 'rmarkdown::render("predict_new_language.Rmd")'

# 4. template-by-language ablation (Appendix A.1)
Rscript -e 'rmarkdown::render("model_ablations.Rmd")'

# 5. all 11 figures used in the paper
Rscript make_results_figures.R
```

## Where the fits live

Fitted models are **not** committed: `_cache/` and `*.rds` are gitignored, and
together they run to hundreds of megabytes. They are regenerable by rerunning the
step that produced them.

To inspect a fit without refitting, open the knitted HTML next to each `.Rmd` ---
every model is printed there in full.

Each document caches its fits (`cache=TRUE` with `cache.extra` on the formula)
while leaving every summary and print chunk uncached. Editing what a chunk
*prints* therefore re-runs only the print; a fit is recomputed only when the data,
the formula or an exclusion actually changes. This is why the expensive documents
are safe to reopen and adjust.

## Figures

`make_results_figures.R` reads the fits out of the knitr cache and never refits.
It writes the 11 figures the paper uses:

| figure | paper |
|---|---|
| `fig_effects` | Fig. 2, fixed effects |
| `fig_levers_odds` | Fig. 3, the four interactions |
| `fig_symbolic_penalty` | Fig. 4, penalty by language |
| `fig_ladder` | Fig. 5, success rate by stage |
| `fig_predict_language` | Fig. 6, forecasting a language |
| `fig_predict_model_language` | Fig. 7, forecasting a model in a language |
| `fig_levers_raw_reasoning` | Fig. 11, model-free reasoning gap |
| `fig_predict_by_language` | Fig. 13, per-language forecast |
| `fig_scale_cost` | Fig. 14, cost priced in model size |
| `fig_design_space` | Fig. 15, feature design space |
| `fig_ladder_fitted` | Fig. 16, fitted effects by stage |

Stratum sizes in `fig_effects` are derived from the data rather than written in,
so they follow the language set instead of going stale.
