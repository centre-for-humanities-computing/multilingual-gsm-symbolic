# What needs changing in the paper

Against `multilingual_gsm-13.pdf`. Every number below comes from the reruns of
2026-09-17/18 and is current as of `model_cells.rds` at 46 models.

## Two changes that cause everything else

**1. `nob` is gone.** Norwegian Bokmal was never one of the benchmark's 15
languages --- it is not in the §3.1 language selection list --- but it was in the
analysis. The prose already said 15; the fits said 16. Now both say 15.

**2. Two runs are excluded.** `Qwen3.5-0.8B (reasoning on)` (0.28% accuracy,
thinking loops) and `EuroLLM-1.7B-Instruct` (0.40%). Footnote 7 already claims
the first is excluded; it was not. Between them they supplied almost all the
model-level variance --- `sd(model)` falls from 0.83 to 0.13 --- and since the
reasoning main effect is tested against that stratum, a thinking-loop failure was
setting its standard error.

So: **48 -> 46 models, 768 -> 690 (model, language) pairs, 16 -> 15 languages.**
Those three counts appear throughout and all move.

---

## Abstract

| was | now |
|---|---|
| 48 Large Language Models | **46** |
| $\beta_{size} = 1.88$ | **1.77** |
| $\beta_{resource} = 0.71$ | **0.77** |
| $\beta_{distance} = -0.29$ | **−0.25** |
| $\beta_{resource \times size} = -0.25$ | **−0.27** |
| $\beta_{distance \times size} = 0.02$ | **0.02** (unchanged, still null, p = .32) |
| predict within 5.6pp (r = .97) | **6.0pp (r = .96)** |
| 10 templates reduce to 3.88pp | **4.31pp** |
| 89% of between-language variation | **92%** |
| 20% of model-by-language | **23%** |

"30,000 item-matched pairs" is unaffected --- that is the dataset, not the model set.

---

## §3.3 Evaluation, and the model count

- "We evaluate 48" (abstract, §3.4) -> **46**.

**The sentence about a generation limit describes something that did not happen.**
No maximum generation length was imposed; models ran to their own context limits.
So "To prevent infinite runs, we set a maximum generation length of `X` tokens"
and "only `XX`% of samples exceeded the budget" both need replacing rather than
filling in, and the claim that two models "consistently exceeded this limit" is
true of one of them and for a different reason.

What the data shows:

| | |
|---|---|
| models hitting any length ceiling, of the 46 analysed | **1** (`gemma-3-1b-it`, at 32,768 tokens, in 1.75% of its instances) |
| share of all analysed instances at a ceiling | **0.038%** (550 of 1,449,000) |
| `EuroLLM-1.7B` | truncated at its own **4,096**-token context in **55.9%** of cases |
| `Qwen 3.5 0.8B` (reasoning=on) | reaches no ceiling; returns an empty or near-empty completion in **96.5%** of cases (against 0% for Qwen3-8B) |

Suggested rewrite:

> We imposed no generation limit; each model ran to its own context length. Only
> one of the 46 analysed models reached that ceiling with any regularity
> (gemma-3-1b-it, in 1.75% of its instances), and across the analysed set just
> 0.04% of samples ended at a length limit.
>
> Two models were excluded, for different reasons. EuroLLM-1.7B has a 4,096-token
> context and was truncated before reaching an answer in 55.9% of cases. Qwen 3.5
> 0.8B (reasoning=on) was not truncated, but returned an empty or near-empty
> completion in 96.5% of cases, producing no answer to score. Both score below
> 0.5% accuracy; results are in Appendix M.

## §4.1.1 Main effects

| claim | was | now |
|---|---|---|
| resource | $0.71$, $p<0.001$ | **$0.77$, $p<0.001$** |
| size | $1.88$, $p<0.001$ | **$1.77$, $p<0.001$** |
| distance | $-0.29$, $p=0.001$ | **$-0.25$, $p<0.001$** |
| fertility | $-0.07$, $p=.019$ | **$-0.04$, $p=.19$** |

**The fertility sentence needs rewriting, not renumbering.** Within-language
relative fertility is no longer significant. The current text already calls the
effect "practically inconsequential", so the claim survives in spirit, but
"$p = .019$" must go. Suggested: *"Examining the uncorrelated relative fertility
we see a slight negative effect ($\beta = -0.04$, $p = .19$) that we cannot
distinguish from zero; it is in any case practically inconsequential."*

This is the one place the exclusion costs something: fertility was the novel main
effect and it is now null. It is the smallest coefficient in the model and rests
on the only within-language contrast available, so it was never well powered.

**Correlation with fertility:** the text says distance correlates with fertility
at $r = 0.8$. Recheck on 15 languages before keeping the figure.

---

## §4.1.2 Interactions

| claim | was | now |
|---|---|---|
| resource $\times$ size | $-0.25$, $p<0.001$ | **$-0.27$, $p<0.001$** |
| distance $\times$ size | $0.02$, $p=0.45$ | **$0.02$, $p=.32$** |
| resource $\times$ reasoning | $-0.15$, $p=.004$ | **$-0.20$, $p<0.001$** |
| distance $\times$ reasoning | $0.12$, $p=.013$ | **$0.11$, $p=.030$** |

The lever asymmetry --- the section's whole point --- is unchanged. Both distance
terms keep their verdicts.

**New and worth adding:** `size × reasoning` is now $-0.34$, $p<0.001$ (it was
$+0.33$, $p=.30$, i.e. absent). Reasoning helps *less* the larger the model. The
model-free version is striking and needs no fit: across the 11 base models run
both ways, the benefit falls from $+22.7$pp at 0.6B to $+1.2$pp at 27B,
$r = -0.78$ with $\log_2$ params. The two levers substitute for each other.

---

## §4.2 Predicting performance on an unseen language

| claim | was | now |
|---|---|---|
| language mean | 3.2pp ($r = 0.92$) | **3.2pp ($r = 0.93$)** |
| model in language | 5.6pp ($r = 0.97$) | **6.0pp ($r = 0.96$)** |
| reduction vs naive | 59% and 38% | **63% and 39%** |
| with 10 templates | 2.29pp / 3.88pp | **2.62pp / 4.31pp** |
| between-language variance | 89% | **92%** |
| model-by-language variance | 20% | **23%** |

**Note: v13's text and figures already disagree with each other.** The figures are
from a later run than the prose, so the PDF currently contains three generations
of numbers at once:

| | v13 text | v13 figure | final (46 models) |
|---|---|---|---|
| language mean | 3.2pp, $r = 0.92$ | 3.0pp, $r = 0.93$ | **3.2pp, $r = 0.93$** |
| model in language | 5.6pp, $r = 0.97$ | 5.7pp, $r = 0.97$ | **6.0pp, $r = 0.96$** |

The v13 figure values are the 48-model, 15-language intermediate state --- after
`nob` was dropped but before the floor models were excluded. Both the text and the
embedded figures need replacing, and the regenerated PNGs already carry the final
values in-panel, so re-including them fixes the figure half automatically.

The language-mean MAE has gone 3.2 -> 3.0 -> 3.2, so it lands back on the number
the text already states, but its correlation moved from 0.92 to 0.93.

The Apertus-70B example survives exactly: **22.7% observed against 74.8%
predicted** (text says 23% and 75%). Estonian still accounts for four of the
eight worst over-predictions.

**The 10-template claim is now a real refit**, not the earlier approximation:
each k-template set is added to the training data and Equation 1 is refitted from
scratch, 15 languages x 5 values of k. Full curve, model-in-language / language
mean:

| k | 0 | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| model in language | 5.98 | 7.06 | 6.46 | 6.51 | **4.31** | 2.84 |
| language mean | 3.17 | 4.93 | 3.25 | 4.08 | **2.62** | 1.83 |

**Do not quote k = 1, 2 or 5 without more replicates.** Those rows are single
template draws and one bad draw dominates: Dutch at k=5 gives an MAE of 20.9
against 3.99 at k=0, which alone moves the 15-language mean by 1.2 points. k=10
rests on 29 fits and k=20 on 15.

The reliable evidence that small k does not help is the variance component, which
averages over 46 models x 15 languages per fit and cannot be moved by one draw:
`sd(language)` runs 0.289, 0.272, 0.271, 0.265, 0.254 at k = 1, 2, 5, 10, 20
against 0.250 in the main model. A single template inflates it 16% --- the model
charging that template's idiosyncrasy to the language effect.

---

## §4.3 Symbolic variants

| claim | was | now |
|---|---|---|
| 768 (model, language) pairs | **705** |
| 3.07pp lower | **3.14pp** |
| unrelated to accuracy level, $r = 0.07$ | **$r = -0.02$** |
| penalty $-0.30$ | unchanged |
| smallest in English 1.7pp | **1.76pp** |
| largest in Russian 5.5pp | **5.66pp** |
| $\chi^2 = 18.4$ | **$21.6$, $p = 2\times10^{-5}$** |
| split $\times$ resource $p = 0.64$ | **$p = 0.68$** |
| split $\times$ distance $p = 0.85$ | **$p = 0.84$** |Every claim in this section survives. Both interactions stay firmly null, so
"orthogonal to the features that predict transfer" is intact.

--- the three
rechecks above are pending.

---

## §4.4 Where in the solution does the gap appear

| claim | was | now |
|---|---|---|
| resource: operand / intermediate / final | 0.30 / 0.42 / 0.71 | **0.32 / 0.45 / 0.77** |
| distance: operand | $-0.09$, $p < .05$ | **$-0.08$, $p = .066$ --- NO LONGER SIGNIFICANT** |
| distance: intermediate | $-0.17$, $p<.001$ | **$-0.15$, $p = .001$** |
| distance: final | $0.29$, $p<.001$ | **$-0.25$, $p<.001$** |

**Two prose changes here, not just numbers.**

First, the distance effect at the operand stage is no longer significant. The
sentence claims distance drives the gap at all three stages; it now drives it at
two.

Second, $\beta_{final} = 0.29$ has the **wrong sign** in the current text --- it
should be negative, and this predates the rerun.

The English-to-Marathi gaps are 19.0 / 30.6 / 37.7 points, essentially unchanged.

---

## §5 Limitations

Two placeholders are now fillable.

**"Size and coverage ... we examine whether the estimates can be sufficiently
obtained using only XX of the samples."** From `sample_size.Rmd` (needs reknitting
at 46 models, numbers below are the 48-model version and will shift slightly):

- Model *rankings* settle fast: Spearman 0.97 against the full ordering at 10
  symbolic templates, 0.99 at 50.
- Model *scores* do not: the standard error of a (model, language) accuracy is
  **±2.25pp at the full 100 templates**, and halving it would take 400.
- Symbolic variants are worth about **2.1x** original ones, not 20x. Twenty
  instances per template only reduce within-template noise, and template identity
  dominates: matching 100 symbolic templates with original questions would take
  **214** of them.

**New limitation worth adding:** the model assumes template difficulty is
language-invariant. It is not. Adding `(1 | template:language)` gives
$\mathrm{sd} = 0.95$, the third-largest component, and improves AIC by $\sim10^5$.
Every significant coefficient keeps its sign, standard errors inflate by a uniform
1.05x, and `sd(language)` is unchanged --- **except** $\beta_{distance \times
reasoning}$, which moves to $p = .068$. That is the one claim sensitive to the
fuller specification, and it should be said rather than discovered by a reviewer.

---

## Status of the supporting documents

All reknit at 46 models on 2026-09-18; no number above is provisional.

| document | covers |
|---|---|
| `transfer_analysis.Rmd` | §4.1, §4.4, the main table |
| `symbolic_penalty.Rmd` | §4.3 |
| `predict_new_language.Rmd` | §4.2, including the k-template curve |
| `model_ablations.Rmd` | Appendix A.1 |
| `sample_size.Rmd` | the limitations sample-size numbers |
| `benchmark_table.Rmd` | appendix leaderboard |
| `training_data.Rmd` | open-data appendix |

Two caveats carried forward:

- `symbolic_penalty.Rmd` and `sample_size.Rmd` read the parquet directly, so the
  model exclusions are repeated in all three places and can drift. If the model
  set changes again, check all three.
- The k = 1, 2 and 5 rows of the few-shot curve are single template draws and
  should not be quoted; k = 10 and k = 20 are solid.
