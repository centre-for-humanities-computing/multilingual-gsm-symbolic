# Results and Discussion (draft)

Numbers from `paper/scripts/analysis/transfer_analysis.Rmd` (knitted 2026-08-27);
figures from `paper/scripts/analysis/figures/`.

**Two structural notes.**

1. Results and Discussion are merged; Ablations and Limitations are separate
   top-level sections.
2. The design and model specification are **not** repeated here -- §3.2 and §3.3
   already carry them. Two modelling decisions currently missing from Methods
   belong there rather than in Results; they are written up at the end of this
   file under *Methods additions*.

---

# 4 Results and Discussion

## 4.1 What determines transfer

### 4.1.1 Main effects

Three of these reproduce known results and we state them briefly, because they are
not the contribution. The fourth is new.

**Resource level dominates.** Higher-resource languages are substantially easier
(+0.71 log-odds per SD, p < 1e-16), consistent with Shi et al. (2022), Blevins and
Zettlemoyer (2022) and Barua et al. (2026).

**Scale helps.** Larger models are better overall (+1.88 per SD, p < 1e-16),
reproducing the MGSM scaling result.

**Typological similarity matters.** Structurally distant languages transfer worse
(-0.29, p = 1.5e-4), consistent with Chang et al. (2023). Read this as the
language-level axis of *structural distance from English, which travels with
tokenising the language badly*: the between-language component of fertility
correlates at r = 0.80 with it across our 16 languages and is not separately
identifiable, so this is not a distance effect net of tokenisation.

**Tokenisation, separated from the language it is confounded with.** Splitting
relative fertility into a between-language mean and a within-language deviation,
and entering only the within arm, isolates an effect orthogonal by construction to
every language-level predictor (r = 0.00 with resource, typological distance and
the between arm). Holding the language fixed, **a model that tokenises it worse
than its peers do performs worse on it** (-0.07, p = .019). Pooling the two arms
is what produced a positively-signed fertility coefficient in earlier versions of
this analysis -- that sign was an artifact of the between-language confound, not a
finding.

The effect is small, and we flag an uncomfortable pattern rather than leave it to
be found: our largest coefficients sit on our weakest identification. Resource and
typological distance rest on 16 mutually-correlated languages; within-language
fertility is the only language-adjacent feature with a within-language contrast,
and it is the smallest effect in the model.

> **Figure: `fig_effects`** -- all fitted effects, grouped by the level at which
> each predictor varies, with effective N per panel.

### 4.1.2 Interactions: which levers close which gaps

Prior work has examined these factors one at a time. Modelling them jointly lets
us ask which *actionable* choices close which gaps -- and the two levers a
developer controls behave differently:

| lever | resource gap | typological-distance gap |
|---|---|---|
| **model size** | closes it (-0.25, p < 1e-16) | **does nothing** (0.02, p = .45) |
| **reasoning** | closes it (-0.15, p = .004) | **closes it** (+0.12, p = .013) |

**Reasoning is the only lever that does anything about structural distance**,
which is the feature you cannot change about a target language. Scale helps only
against data scarcity.

The reasoning *main* effect is not significant in the model (+0.49, p = .12),
because `(1 | model)` puts it on the correct stratum: it is tested against how
much reasoning helps across the 12 base models run both ways. The average lift is
better evidenced by the paired comparison, which does not depend on the model at
all -- across those 12 bases, enabling reasoning raises mean accuracy from 0.62 to
0.69 and shrinks the English-to-target gap from 20.6 to 12.9 accuracy points. On
the relative scale a model is **2.2x more likely to answer incorrectly** in a
non-English language with reasoning off, falling to **1.7x** with it on.

> Risk ratio, not odds ratio. The corresponding odds ratios are 3.1x and 2.1x;
> quoting those with "times more likely" wording overstates the gap by ~40%.

> **Figure: `fig_levers`** -- predicted error rate relative to English, by
> resource and by typological distance, for each lever.

### 4.1.3 Effect sizes in a unit practitioners budget in

Both size and resource are log-transformed, so dividing by their standard
deviations puts them in **doublings**, and the two can be quoted against each
other:

- **0.99 log-odds per doubling of parameters**
- **0.25 log-odds per doubling of Common Crawl pages**
- therefore **one doubling of model size is worth about four doublings of
  language data** (~15x the data)

The same exchange rate prices each language's disadvantage. Working in Marathi
costs about as much accuracy as shrinking the model **3.0x**; Icelandic 2.7x,
Estonian 2.5x, Hindi 2.3x, down to Italian 1.3x.

> **Figure: `fig_scale_cost`** -- each language's gap expressed as an equivalent
> parameter multiple.

These are main effects at the sample means and so ignore the `resource x size`
interaction, which makes the true exchange rate depend on model size; and Common
Crawl pages is a proxy for a language's share of the training mixture rather than
a measurement of it.

---

## 4.2 Symbolic variants are harder, and unevenly so across languages

Moved to `draft_synt_orig.md` (results subsection + appendix, LaTeX).

---

## 4.3 Where in a solution the gap accrues

Moved to `draft_ladder.md` (results subsection + appendix, LaTeX).

---

## 4.4 Predicting performance on an unevaluated language

Moved to `draft_prediction.md` (results subsection + appendix, LaTeX).

---

## 4.5 What the features account for

Moved to `draft_variance_summary.md`. Merges the old 4.4 (variance) with the old
4.6 (summary) and closes the results; it has to follow prediction, since the
wrap-up half synthesises those numbers.

---

# 5 Ablations

## 5.1 Model ablations

**Resource x typological distance: no interaction** (0.02, p = .61). An earlier
specification found this significant, but that was a third language-level term
tested against model-by-language variance -- the same stratum error the language
intercept corrects.

**Script adds nothing** (-0.03, p = .90), and typological distance barely moves
when it is added (-0.29 to -0.28), so the distance effect is not a writing-system
effect in disguise. The URIEL `syntax_knn` vector encodes no orthographic
information.

## 5.2 Dataset ablations

> **Both of these currently say the opposite of what the artifact says.** The
> tests are in `paper/artifacts/equivalence/isl_translation_tost.txt`; both
> *fail* equivalence at alpha = 0.05. §5.1.1 and §5.1.2 of the current draft
> report them as passing.

**Machine vs human translation.** Icelandic machine-translated templates score
**2.43 points lower** than human-verified ones (0.296 vs 0.320, 24 paired models),
against an equivalence margin of +/-0.5 points. TOST p = 0.985; **not
equivalent** -- the difference is roughly five times the margin, and machine
translation is *worse*, not indistinguishable.

This matters beyond the sentence: it is the stated basis for extending the dataset
to 100 languages by machine translation. That extension is still defensible as a
*resource* -- a 2.4-point degradation is usable for many purposes -- but not on
the grounds that MT and human-verified templates perform equivalently. Suggested
reframing: report the measured penalty and label the machine-translated languages
as a distinct, lower-quality tier.

**English vs English-metric.** `eng_metric` scores 0.43 points lower than `eng`
(48 paired models) -- inside the +/-0.5 margin as a point estimate, but TOST
p = 0.162, so equivalence is **not established**. This is an underpowered null
rather than a clear difference, and the argument survives in weaker form: 0.43
points is negligible against the 20-point cross-lingual gaps, so unit localisation
is not what drives the transfer results. State it as "much smaller than the
effects of interest", not as "equivalent (p < .05)".

## 5.3 Benchmark size

Model rankings are stable well below the full template set: a random 10 templates
already correlate at **r = 0.96** with the full 100-template ranking, 25 at 0.99
and 50 at **0.996** (200 resamples per point, per language and split). The
100-template design is comfortably past the point of diminishing returns, which
also answers the criticism levelled at MGSM's 250-item subset.

> Artifact: `paper/artifacts/subset_size_correlation/`. Could go to an appendix.

---

# 6 Limitations

- **Resource proxy.** Common Crawl page count, with known bias (Chinese and
  Russian appear undercounted) and unreliable language ID for low-resource
  languages.
- **Typological distance is a composite.** It is not separable from the
  between-language tokenisation axis (r = 0.80), and its URIEL vector is 23-39%
  k-NN imputed for several of our languages, so it partly encodes family
  membership rather than measured typology.
- **Model-by-language variation is largely unexplained** (80%), which sets the
  floor on per-model forecasting and is where the largest prediction failures sit.
- **Instruction-tuned, dense models only**; the estimates characterise the
  instruction-aligned model and do not separate pretraining from alignment.
- **Machine-translated languages carry a measured 2.4-point penalty** (§5.2), so
  the 100-language extension is a lower-quality tier rather than an equivalent one.
- **Generalization.** A preliminary analysis on MultiZebraLogic reproduced the
  direction of the resource and fertility effects but did not reproduce
  `resource x size`. We do not report it as a replication: 33 of its 44 languages
  are machine-translated without review, its reasoning runs truncate at a rate
  that varies with the language features themselves, and the human-validated
  subset is 9/11 Germanic and yields singular fits. See
  `paper/scripts/analysis/transfer_generalization.Rmd`.

---

# Methods additions (belong in §3.2 / §3.3, not Results)

**The fertility split.** Relative fertility is the only feature that varies across
models within a language. We split it into a between-language mean and a
within-language deviation and enter only the within arm; the between arm
correlates at r = 0.80 with typological distance across our 16 languages and is
not separately identifiable from it. Entering both is what produced a
positively-signed fertility coefficient in earlier versions of this analysis.

**Why `(1 | language)` is load-bearing.** Resource and typological distance vary
only across the 16 languages. Without a language intercept their standard errors
are computed from the model-by-language stratum and are anticonservative:
refitting without it inflates the SEs by **2.6x on exactly those two terms** and
leaves all eleven others unchanged (0.90-1.00x), while point estimates barely
move. The consequential case is typological distance, z = -9.8 without the
intercept and -3.8 with it. We report the corrected version throughout.
