# Where the gap accrues: results subsection + appendix

Replaces §4.3 of `draft_results.md`. Numbers from
`paper/scripts/analysis/transfer_analysis.Rmd` (chunk `fit-ladder`).

**Withdrawn from the earlier version of this draft:** the mediation analysis and
its "76% / 56% / 50% propagated" table, and the conclusion that the gap is
"predominantly a failure to take the problem in". Operand retrieval and
intermediate retrieval are produced by the same number-extractor run over the same
response, so their measurement errors share a language-dependent component --- the
Japanese kanji failure is a documented instance --- and an adjustment on the one
is not identified with respect to the other. The marginal rates agree: about half
the final gap is present at the operand stage and about half accrues after it.

---

## Results subsection

Significance is marked with the usual stars, defined in the appendix table
caption. `eq:model` is a placeholder for whatever you label the main model
equation in Methods.

```latex
As our templates carry GSM-style \texttt{<<lhs=rhs>>} calculator annotations we can extract
operand reads (\texttt{lhs}), intermediate computations (\texttt{rhs}) along with the final
answer.

We see that multilingual performance isn't simply a question of parsing the problem
(operand reads), but is accumulated at every stage (see Figure~\ref{fig:ladder}). Fitting a
model similar to Equation~\ref{eq:model} at each stage (see Appendix~\ref{app:ladder}), we
see that resource level is the dominant feature throughout
($0.30^{***} \rightarrow 0.42^{***} \rightarrow 0.71^{***}$ log-odds per SD), typological
distance roughly a third its size
($-0.09^{*} \rightarrow -0.17^{***} \rightarrow -0.29^{***}$), and relative fertility
smaller still and indistinguishable from zero at the operand stage
($-0.03 \rightarrow -0.07^{***} \rightarrow -0.07^{*}$). The ordering within a stage is
interpretable; the growth across stages is not, as the three outcomes differ in base rate
and residual dispersion.
```

**Optional extra sentence**, if you want one observation that survives the
cross-stage objection --- a variance component reaching the boundary is not a
matter of degree:

```latex
Model family is irrelevant to reproducing the numbers in a problem and decisive for solving
it: the between-family variance is effectively zero at the operand stage
($\mathrm{sd} = 0.001$) and the largest random effect in the model at the final answer
($1.33$).
```

Typo in the current text: "accumalated" -> "accumulated".

---

## Appendix section

```latex
\section{The solution ladder}
\label{app:ladder}

\paragraph{Outcomes.} Each template ships with GSM-style \texttt{<<lhs=rhs>>} calculator
annotations taken from the gold solution, giving two intermediate outcomes alongside the
final answer: of the left-hand-side operands a correct solution uses, how many appear in
the response; and of the right-hand-side results it produces, how many appear. Both are
marginal --- the second does not condition on having passed the first --- so the three
outcomes measure accumulated degradation at increasing depth rather than per-stage
conditional success, and no post-treatment variable is conditioned on.

The denominators are taken from the gold solution and are language-invariant in practice:
mean operand count per item ranges 5.31--5.45 across the 16 languages and mean intermediate
count 3.09--3.17. Cells carrying no annotations are 2.1\% of the total (1{,}584 of
76{,}800) and evenly distributed across languages; they drop out of the two intermediate
fits, which are estimated on 75{,}216 cells.

\paragraph{Model.} Each stage is fitted with the specification of
Equation~\ref{eq:model} --- identical fixed and random parts --- changing only the binomial
response to that stage's numerator and denominator. Holding the specification fixed is what
makes the coefficients comparable at all.

\begin{table}[h]
\centering
\begin{tabular}{lrrr}
\toprule
 & Operand reads & Intermediates & Final answer \\
\midrule
Resource level              & $0.299^{***}$  & $0.420^{***}$  & $0.712^{***}$ \\
Typological distance        & $-0.091^{*}$   & $-0.172^{***}$ & $-0.288^{***}$ \\
Relative fertility (within) & $-0.035$       & $-0.067^{***}$ & $-0.071^{*}$ \\
Model size                  & $0.493^{***}$  & $0.934^{***}$  & $1.881^{***}$ \\
Resource $\times$ size      & $-0.070^{***}$ & $-0.120^{***}$ & $-0.246^{***}$ \\
\midrule
$\mathrm{sd}(\text{language})$       & 0.136 & 0.163 & 0.272 \\
$\mathrm{sd}(\text{model:language})$ & 0.390 & 0.390 & 0.580 \\
$\mathrm{sd}(\text{family})$         & 0.001 & 0.541 & 1.328 \\
\bottomrule
\end{tabular}
\caption{Fixed effects in log-odds per SD and selected variance components at each stage.
$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}
\label{tab:ladder}
\end{table}

\paragraph{Interpretation.} Every coefficient grows with depth
(Figure~\ref{fig:ladder-fitted}), but the three outcomes differ in base rate and residual
dispersion, so a difference in scale alone would inflate all of them by roughly a common
factor. We therefore compare features within a stage, where they share a scale, and do not
read the growth across stages as a measure of where the gap accrues.

Attributing the gap to a particular stage would require adjusting for operand retrieval
when modelling the later outcomes. That is not available here: operand and intermediate
retrieval are both measured by running the same number-extractor over the same response, so
their measurement errors share a component that is itself language-dependent, and an
adjustment on one is not identified with respect to the other.
```

---

## Cuts from the previous draft of this appendix

Removed as out of place, all recoverable from git if you want any of them back:

- the collider argument against conditioning on a full operand parse --- the
  measurement point above already rules the analysis out, so the second reason
  was redundant;
- the Japanese kanji retraction (an earlier draft read typological distance as
  flat across stages; that was an extraction artifact). It is a story about a
  previous version of our own analysis, not about this one. If it belongs
  anywhere it is Limitations, as evidence that the extractor's error is
  language-dependent;
- the enumerated outcome list, folded into prose;
- the growth factors (3.2x, 2.4x, 2.0x), since the paragraph now says the
  cross-stage growth should not be read quantitatively.

## Figures

| macro | file |
|---|---|
| `fig:ladder` | `figures/fig_ladder.pdf` (main text) |
| `fig:ladder-fitted` | `figures/fig_ladder_fitted.pdf` (appendix) |

```latex
\caption{Fitted effect of each language feature at each stage, in absolute log-odds per SD.
Every effect grows with depth, but because the three outcomes differ in base rate and
residual dispersion a difference in scale alone would produce the same pattern.}
```
