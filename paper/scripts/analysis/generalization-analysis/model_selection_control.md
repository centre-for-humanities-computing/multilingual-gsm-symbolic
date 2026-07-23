# What the MultiZebra selection can recover — a GSM design simulation

We use GSM-Symbolic (where we have the full model × language grid) as a testbed to
predict what each MultiZebra selection can and cannot estimate — *before* spending
compute on new runs. Each row refits the **same** model on a subset of the GSM cells:

```
cbind(n_correct, n_total - n_correct) ~ resource_c * size_c + reasoning +
  (1 | model) + (1 | language) + (1 | template) + (1 | model:language)
```

Predictors are per-doubling (log2), centred at ~32M CC pages / 8B params. Significance:
`*** p<0.001  ** p<0.01  * p<0.05  (·) p<0.1  ns`.

All fits include `reasoning` as a covariate, but its coefficient is **not shown** — in
this pooled `(1|model)` spec it is a between-model nuisance term (it discards the on/off
pairing), not the reasoning effect. Reasoning is estimated **within base model** in
`transfer_generalization.Rmd`; see the note below.

| Scenario | Langs | Models | Size range | Resource | Size | Resource × Size | sd(mdl:lang) | Var. expl. |
|---|---|---|---|---|---|---|---|---|
| **All models & languages** (full GSM) | 7 | 45 | 0.5–72B | 0.23 \*\*\* | 0.99 \*\*\* | **−0.039 \*\*\*** | 0.49 | **21%** |
| **Current zebra selection** (Qwen+EuroLLM, Germanic) | 4 | 9 | 1.7–27B | 0.19 \*\*\* | 1.24 \* | +0.014 (ns) | 0.77 | 0.6% |
| **+ all languages** (zebra models, all langs) | 7 | 9 | 1.7–27B | 0.15 \*\* | 1.29 \* | +0.016 (ns) | 0.62 | 1% |
| **+ all models** (all models, Germanic) | 4 | 45 | 0.5–72B | 0.28 \*\*\* | 1.03 \*\*\* | **−0.038 \*\*\*** | 0.62 | **21%** |
| **+ suggested models** (roster w/ 1B anchors, Germanic) | 4 | 19 | 0.8–32B | 0.26 \*\*\* | 1.35 \*\*\* | **−0.031 \*** | 0.77 | **10%** |
| **+ suggested models & all languages** (full recommended) | 7 | 19 | 0.8–32B | 0.21 \*\*\* | 1.32 \*\*\* | **−0.035 \*\*\*** | 0.62 | **12%** |

## Reading it

- **Resource and size main effects are robust everywhere** (resource 0.15–0.28, size
  ~1.0–1.35, all significant). They are never the identifiability problem. *On the actual
  zebra task* the resource main effect is weak (~0.05) — but the "current zebra selection"
  row keeps it at 0.19 on the same models/languages, so that weakness is a genuine **task**
  difference, not a selection artifact.
- **The interaction and variance-explained are the fragile pair.** They are dead in the
  current selection (+0.014 ns, 0.6%) and **stay dead when you add all languages**
  (+0.016 ns, 1%) — so the language expansion does *not* recover them.
- **Models recover them.** Adding all models (−0.038 \*\*\*, 21%) or the suggested reduced
  roster (−0.031 \*, 10%) both bring the interaction back to a significant, correctly-signed
  estimate and restore variance-explained — Germanic-only, no extra languages needed.
- **Languages then sharpen — but only after the models are in place.** Suggested roster +
  all languages takes the interaction from −0.031 (p=.03) to **−0.035 (p<.001)** and
  variance-explained 10%→12%, approaching full-GSM significance at just 19 models. The two
  expansions are complementary and ordered: **models make the interaction identifiable;
  languages tighten it** (and add resource range for the main effect). Languages alone,
  without the model expansion, do nothing (rows 2→3).
- **The operative change is size range, not model count or family count.** The suggested
  roster works because it spans down to **0.8B**; a version that stops at 1.7B (the roster
  without small anchors) leaves the interaction at ~0 (ns) even with the same families. The
  tiny-vs-large size contrast is what anchors "big models depend less on resource."

## Caveats

- This is a GSM simulation, not the zebra task. The suggested-roster row uses GSM proxies
  where checkpoints differ (e.g. OLMo-2 sizes stand in for OLMo-3-32B, which GSM lacks).
- **Reasoning is not a quantity this table can read.** In the pooled `(1|model)` spec the
  reasoning coefficient is a between-model contrast that discards the on/off pairing, so it
  looks non-significant in the smaller sets (e.g. suggested models: +1.06, p=0.33). Refit
  on the *same* data with the correct within-base spec (`(1|family/base_model)`) and it is
  significant (−0.51, p=0.045) — so reasoning is identifiable with the roster; the table's
  "ns" was a spec artifact. Its magnitude is also task-specific: within-base on GSM's
  easy-ish math it is small/slightly negative, whereas on MultiZebra's logic puzzles the
  same design gives a large positive effect (+2.4, p<0.001). The reasoning effect is
  therefore estimated within-base in `transfer_generalization.Rmd`, not here — and the
  roster *adds* a second reasoning family (OLMo-3), which strengthens it.
- Language expansion is still worth doing — it is what makes typology, fertility, and
  script testable, and adds resource range — it just is not the lever for these two
  estimates.

See `model_roster.md` for the resulting model list.
