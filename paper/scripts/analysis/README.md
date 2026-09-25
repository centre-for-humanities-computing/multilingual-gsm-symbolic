# Transfer analysis

Statistical analysis behind the results and the appendices: the GLMM of
cross-lingual transfer, the symbolic-penalty model, the leave-one-language-out
forecast, and every figure generated in R.

## Language set

The benchmark's **15** languages: `ara dan deu eng est fra hin isl ita jpn mar
nld rus ukr zho`. Three labels in the parquet are deliberately excluded in
`transfer_analysis.Rmd` (`EXCLUDED_LANGUAGES`):

| excluded | why |
|---|---|
| `eng_metric` | English with metric units — not a language, and has no typological or resource features. Used only for the imperial-vs-metric localisation check. |
| `uncorrected_isl` | machine-translated Icelandic before human correction. Used only for the machine-vs-human translation check. |
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

Note the exclusions are declared in four places, because `symbolic_penalty.Rmd`,
`sample_size.Rmd` and `benchmark_table.Rmd` read the parquet directly rather than
`model_cells.rds`. If the model set changes, change all four.

The parquet's model naming is **not uniform**, which is the trap here. Qwen3 and
Qwen3.5 label only the reasoning-off run, so the bare name is the reasoning-ON
run; Granite labels both explicitly; Olmo-3-Think appears only as
`(reasoning on)`. A document reading the parquet must reproduce
`transfer_analysis.Rmd`'s canonicalisation before matching an exclusion list, or
it silently keeps a model it meant to drop. `benchmark_table.Rmd` does this and
guards the result with a `stopifnot`.

`benchmark_table.Rmd` deliberately reports **48 runs** --- the 46 that enter the
model plus the two floor runs, marked with an asterisk --- so that a reader
looking up a model finds it either way.

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

# 2. symbolic vs original penalty: is it harder, and does it vary by language?
Rscript -e 'rmarkdown::render("symbolic_penalty.Rmd")'

# 3. forecasting an unseen language, with and without k evaluated templates.
#    15 leave-one-language-out folds then 225 refits (15 languages x 5 budgets
#    x 3 replicates): several hours. Each fit is its own process via
#    run_fold.R, so the job is resumable.
Rscript -e 'rmarkdown::render("predict_new_language.Rmd")'

# 4. template-by-language random effect, an alternative specification.
#    Two fits, ~20 min.
#    Calls run_ablation.R, one fit per process.
Rscript -e 'rmarkdown::render("model_ablations.Rmd")'

# 5. is the size interaction a ceiling artifact? One refit per restriction,
#    ~10 min each; cell90 is run but not reported (see the script header).
for r in eng90 eng95 cell90; do Rscript run_saturation.R $r; done

# 6. supporting appendices: how many templates the benchmark needs, the
#    leaderboard and per-language results, and the open-data
#    training-distribution analysis
Rscript -e 'rmarkdown::render("sample_size.Rmd")'
Rscript -e 'rmarkdown::render("benchmark_table.Rmd")'
Rscript -e 'rmarkdown::render("training_data.Rmd")'

# 7. answer-format compliance by language. The collector parses all 958
#    synthetic eval logs off the Hub and writes compliance.csv; it is
#    resumable and appends per log.
python collect_compliance.py
Rscript -e 'rmarkdown::render("format_compliance.Rmd")'

# 8. the ten figures that are not written by a document
Rscript make_results_figures.R
```

`benchmark_table.Rmd` also writes `tables/*.tex`, which the paper `\input`s
directly, so the leaderboard and the two per-language matrices are never
transcribed by hand.

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
Eleven figures go into the paper; it writes ten of them, and `sample_size.Rmd`
writes its own.

| figure | what it shows | written by |
|---|---|---|
| `fig_effects` | all fitted effects, by inferential stratum | `make_results_figures.R` |
| `fig_ladder` | observed success rate by solution stage | `make_results_figures.R` |
| `fig_levers_odds` | the four lever x language-feature interactions | `make_results_figures.R` |
| `fig_predict_language` | forecasting a held-out language | `make_results_figures.R` |
| `fig_predict_model_language` | forecasting a model in a held-out language | `make_results_figures.R` |
| `fig_sample_size` | precision and ranking vs template budget | `sample_size.Rmd` |
| `fig_symbolic_penalty` | symbolic penalty by language | `make_results_figures.R` |
| `fig_predict_by_language` | the per-language forecast decomposition | `make_results_figures.R` |
| `fig_scale_cost` | each language's gap priced in model size | `make_results_figures.R` |
| `fig_design_space` | language selection / confound structure | `make_results_figures.R` |
| `fig_ladder_fitted` | fitted effects by stage | `make_results_figures.R` |

Stratum sizes in `fig_effects` are derived from the data rather than written in,
so they follow the language set instead of going stale.
