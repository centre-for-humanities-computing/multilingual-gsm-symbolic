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

## Excluded models

Two runs sit at the probability floor and are excluded (`EXCLUDED_RUNS`), along
with `gpt-5.4-nano` (proprietary: no parameter count, closed tokenizer):

| excluded run | accuracy | why |
|---|---|---|
| `Qwen3.5-0.8B (reasoning on)` | 0.28% | returns an empty completion in 96.5% of cases |
| `EuroLLM-1.7B-Instruct` | 0.40% | truncated by its own 4,096-token context in 55.9% of cases |

Between them they supplied almost all the model-level variance --- `sd(model)`
falls from 0.83 to 0.13 when they are dropped --- and the reasoning main effect is
tested against that stratum. **46 models, 15 languages, 690 (model, language)
pairs, 69,000 cells.**

Note the exclusions are declared in three places, because `symbolic_penalty.Rmd`
and `sample_size.Rmd` read the parquet directly rather than `model_cells.rds`. If
the model set changes, change all three.

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

# 4. template-by-language ablation (Appendix A.1). Two fits, ~20 min.
Rscript -e 'rmarkdown::render("model_ablations.Rmd")'

# 5. supporting appendices: how many templates the benchmark needs, the
#    leaderboard, and the open-data training-distribution analysis
Rscript -e 'rmarkdown::render("sample_size.Rmd")'
Rscript -e 'rmarkdown::render("benchmark_table.Rmd")'
Rscript -e 'rmarkdown::render("training_data.Rmd")'

# 6. the 10 figures used in the paper
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
It writes the 10 figures the paper uses:

| figure | paper |
|---|---|
| `fig_effects` | Fig. 2, fixed effects |
| `fig_levers_odds` | Fig. 3, the four interactions |
| `fig_symbolic_penalty` | Fig. 4, penalty by language |
| `fig_ladder` | Fig. 5, success rate by stage |
| `fig_predict_language` | Fig. 6, forecasting a language |
| `fig_predict_model_language` | Fig. 7, forecasting a model in a language |
| `fig_predict_by_language` | Fig. 13, per-language forecast |
| `fig_scale_cost` | Fig. 14, cost priced in model size |
| `fig_design_space` | Fig. 15, feature design space |
| `fig_ladder_fitted` | Fig. 16, fitted effects by stage |

Stratum sizes in `fig_effects` are derived from the data rather than written in,
so they follow the language set instead of going stale.
