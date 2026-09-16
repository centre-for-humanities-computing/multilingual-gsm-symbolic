# Symbolic vs. original: results subsection + appendix

Two pieces to paste. Numbers below come from the simplified model
(`paper/scripts/analysis/fit_symbolic.R`), which drops the English indicator and
the model-size terms: model ability is already carried by the model, base-model
and family random effects, so the fixed part only needs the split and the two
features the text cites.

**Numbers that changed** from the earlier, larger model: resource interaction
`p = 0.87 -> 0.64`, LRT `chi-square = 16.3 -> 18.4`. The penalty itself is
unchanged at `-0.30` and typological distance stays at `p = 0.85`.

**On English.** Reporting it with its interval is a reasonable call, and the text
below does that. Worth knowing what the interval means here: English's raw
penalty is the lowest of the 16 but its standard error across models is 1.18
against 0.38-0.65 for every other language, so the interval is wide because the
estimate is noisy rather than because the effect is small. In the model the
shrunken per-language penalty places English at -0.272 against an average of
-0.297 -- close to the middle of the 16, not at the bottom. The wording below says
"smallest" of the raw values, which is true, and lets the interval speak.

---

## Results subsection

```latex
\subsection{Symbolic variants are harder, and unevenly so across languages}
Similar to previous works \citet{Mirzadeh2024GSMSymbolicUT, xu2026mgsmprosimplestrategyrobust}
we show that symbolic variants are harder than the originals (see
Figure~\ref{fig:symbolic-penalty}). Across 768 (model, language) pairs, accuracy on
symbolic instances is 3.07pp lower than on the original GSM8K questions and the gap is
unrelated to the model's accuracy level ($r = 0.07$). Fitting both splits jointly
(Appendix~\ref{app:synth-vs-orig}) we observe a penalty of $-0.30$ in log-odds
($p < 0.001$).

Additionally we find that the penalty varies across languages ($\chi^2 = 18.4$,
$p < 0.001$): it is smallest in English (1.7pp, 95\% CI $[-0.6, 4.0]$) and largest in
Russian (5.5pp $[4.6, 6.5]$), though the intervals for individual languages are wide. It
is, however, orthogonal to the features that predict transfer (resource level $p = 0.64$,
typological distance $p = 0.85$, see Appendix~\ref{app:synth-vs-orig}). So the features
that explain how well a model does in a language do not seem to predict how fragile it is
there.
```

---

## Comparison with MGSM-Pro (optional paragraph)

MGSM-Pro reports that *"many low-resource languages suffer large performance drops
when tested on digit instantiations different from those in the original test
set"*, and that robustness in high-resource languages does not transfer to
low-resource ones. We find no such relationship. This is worth stating rather than
leaving for a reader to notice.

```latex
Our results are in tension with \citet{xu2026mgsmprosimplestrategyrobust}, who report that
low-resource languages suffer the largest drops under digit perturbation. We find the
direction consistent but the effect negligible: the penalty correlates $r = -0.10$ with
resource level across our 16 languages, and the interaction is null in the joint model
($p = 0.64$). The clearest counterexample is Russian, the second-highest-resource language
in our set, which shows the \emph{largest} symbolic penalty (5.5pp), while Marathi, the
lowest-resource, is second (4.4pp) --- both ends of the resource axis sit near the top. The
discrepancy is not explained by coverage: our lowest-resource language (Marathi,
$10^{5.8}$ Common Crawl pages) is close to theirs (Swahili, $10^{5.4}$). Two differences
remain plausible. MGSM-Pro perturbs digits alone, whereas GSM-Symbolic instantiations also
vary names and entities, so the two measure sensitivity to different manipulations; and
their evaluation covers frontier proprietary models where ours covers open-weight
instruction-tuned ones.
```

**Before using this**, check two things in the paper itself. First, whether their
resource claim is supported by a statistical test or is a descriptive reading of
per-language drops --- one automated extraction suggested the latter, but the same
extraction disagreed with the abstract about which models were evaluated, so it
should not be relied on. Second, exactly which languages and models they use; the
Swahili figure above assumes their low-resource set is the MGSM one.

If their claim turns out to be descriptive, the disagreement is milder than a
contradiction and the paragraph should say so: per-language drops estimated from a
handful of models per language are noisy, which our own English case illustrates
(1.7pp raw, interval spanning zero, and mid-pack once the model accounts for
clustering across models).

---

## Appendix: modelling the symbolic penalty

```latex
\subsection{Modelling the Symbolic Penalty}
\label{app:synth-vs-orig}
Whether symbolic variants are harder, and whether the penalty differs across languages, is
estimated over both splits jointly rather than from a paired test on per-cell differences.
A paired test would treat the 48 models evaluated in a language as that many independent
observations of that language's fragility, inflating the precision of any per-language
claim for the same reason the language term is needed in Equation~\ref{eq:model}.

Let $z_{s} = 1$ if an instance comes from the symbolic split and $0$ if it comes from the
original questions. Extending Equation~\ref{eq:model},

\begin{equation}
\begin{aligned}
y_{mltsi} &\sim \mathrm{Bernoulli}(\pi_{mltsi}), \\[2pt]
\mathrm{logit}(\pi_{mltsi}) &= \mathbf{x}_{l}^{\top}\boldsymbol{\beta}
  \;+\; z_{s}\!\left(\delta + \mathbf{x}_{l}^{\top}\boldsymbol{\delta}\right)
  \;+\; \sum_{g\in\mathcal{G}} u^{(g)}_{j_g(m,l,t)}
  \;+\; z_{s}\, v_{l}, \\[4pt]
\begin{pmatrix} u^{(\text{language})}_{l} \\[2pt] v_{l} \end{pmatrix}
  &\overset{\text{iid}}{\sim} \mathcal{N}\!\left(\mathbf{0}, \boldsymbol{\Sigma}\right)
\end{aligned}
\label{eq:symbolic}
\end{equation}

where $\mathbf{x}_{l}$ holds the two language features and $\mathcal{G}$ is as in
Equation~\ref{eq:model}. Model size and reasoning need no fixed terms here: the model,
base-model and family random effects already absorb capability, and the question is only
how the split effect behaves.

Each new term answers one question. The scalar $\delta$ is the average symbolic penalty.
The vector $\boldsymbol{\delta}$ asks whether that penalty is predicted by the language
features. The language-specific slope $v_{l}$ measures how much it varies across languages
beyond those features, so $\tau^2 = \mathrm{Var}(v_l)$ is the quantity of interest: a
likelihood-ratio test of $\tau = 0$ against a model with $v_{l}$ removed tests whether
fragility differs across languages at all. Correlating $v_{l}$ with the language intercept
lets harder languages be more or less fragile rather than assuming the two unrelated.

We fit over all $153{,}600$ (model $\times$ language $\times$ template $\times$ split)
cells. The splits differ in instances per cell --- one per template for the original
questions, twenty for the symbolic variants --- which the binomial response accommodates
directly, the original arm simply carrying less information per cell. We obtain
$\hat{\delta} = -0.297$ ($p < 10^{-15}$) and $\hat{\tau} = 0.059$ log-odds
($\chi^{2} = 18.4$, $\mathrm{df} = 2$, $p < 0.001$), with both elements of
$\boldsymbol{\delta}$ null (resource $p = 0.64$, typological distance $p = 0.85$). Both
fits carry a borderline convergence warning ($\max|\mathrm{grad}| \approx 0.003$--$0.004$
against a $0.002$ tolerance), which at this scale reflects \texttt{lme4}'s sensitivity
rather than a failure to converge.
```

---

## Provenance

- Model: `paper/scripts/analysis/fit_symbolic.R` -> `fit_symbolic_slope.rds`,
  `fit_symbolic_intercept.rds`
- Figure: `fig_symbolic_penalty` in `paper/scripts/analysis/make_results_figures.R`
- Both restricted to the `refs/pr/16` dataset revision, matching the main analysis.

---

## Side-by-side placement with the ladder figure

Two `\caption` calls inside one float each increment the figure counter, so these
render as two independently numbered figures that happen to sit side by side, not
a subfigure pair. Both figures are now generated at 4.1 x 4.3 inches, close to
their rendered size at `0.48\linewidth`, so type is not shrunk.

```latex
\begin{figure}[t]
\begin{minipage}[t]{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fig_ladder.pdf}
    \caption{\kenneth{placeholder still needs some work}}
    \label{fig:ladder}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fig_symbolic_penalty.pdf}
    \caption{\textbf{Symbolic penalty}: Mean penalty across languages along with a 95\% CI.}
    \label{fig:symbolic-penalty}
\end{minipage}
\end{figure}
```

The ladder's colour bar sits under its panel, so its caption starts slightly lower
than the symbolic figure's; if that reads as uneven, `[b]` alignment on both
minipages evens the captions instead of the tops.
