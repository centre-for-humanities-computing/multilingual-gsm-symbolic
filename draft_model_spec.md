# Appendix: full model specification

Everything below is read out of the fitted model in
`paper/scripts/analysis/transfer_analysis.Rmd` (chunk `fit-lme4`). Two labels are
placeholders: `eq:model` for the equation you reference from the results, and
`app:sec:model-ablations` for the ablation subsection.

Drop the `\mathrm{sd}` column of Table~\ref{tab:random-effects} if space is tight
--- the variances are redundant with it. Everything else earns its place: a
reader who wants to know why the language-level terms have the standard errors
they do needs the group counts, and a reader reproducing the fit needs the
standardisation constants.

---

```latex
\section{Full model specification}
\label{app:sec:model}

\subsection{Model}
Each observation is one (model, language, template) cell, aggregated over the 20 instances
generated from that template, giving $y_{i} \sim \mathrm{Binomial}(20, \pi_i)$ over
$N = 76{,}800$ cells. We fit a binomial GLMM with a logit link,

\begin{equation}
\label{eq:model}
\begin{split}
\mathrm{logit}(\pi_i) = {}& \beta_0
 + \big(\mathrm{fert}_i + \mathrm{dist}_i + \mathrm{res}_i\big) \times \mathrm{size}_i \\
 &+ \mathrm{reasoning}_i \times \big(\mathrm{fert}_i + \mathrm{dist}_i + \mathrm{res}_i
   + \mathrm{size}_i\big) \\
 &+ u_{\mathrm{family}} + u_{\mathrm{base\;model}} + u_{\mathrm{model}}
   + u_{\mathrm{template}} + u_{\mathrm{language}} + u_{\mathrm{model:language}},
\end{split}
\end{equation}

where $\times$ denotes a main effect and interaction, base models are nested within family,
and every random effect is $\mathcal{N}(0, \sigma^2_{\cdot})$ with an independent variance.
In \texttt{lme4} syntax:

\begin{verbatim}
cbind(n_correct, n_total - n_correct) ~
  (relative_fertility_within_z + typological_distance_z + log_resource_z) * log_params_z +
  reasoning * (relative_fertility_within_z + typological_distance_z +
               log_resource_z + log_params_z) +
  (1 | family/base_model) + (1 | model) + (1 | template) +
  (1 | language) + (1 | model:language)
\end{verbatim}

\paragraph{Predictors.} All four continuous predictors are standardised to mean zero and
unit variance, so coefficients are log-odds per standard deviation and are comparable
across predictors within the model. Resource level and model size are logged first, which
makes a standard deviation of each a fixed number of doublings and lets the two be quoted
against one another (Section~\ref{sec:effect-sizes}). \texttt{reasoning} is a two-level
factor with \texttt{off} as the reference. Table~\ref{tab:standardisation} gives the
constants needed to return any coefficient to its raw scale.

\begin{table}[h]
\centering
\begin{tabular}{llrr}
\toprule
Term & Raw quantity & Mean & SD \\
\midrule
$\mathrm{fert}$ & relative fertility, within-language deviation & $0.000$ & $0.460$ \\
$\mathrm{dist}$ & URIEL \texttt{syntax\_knn} cosine distance from English & $0.211$ & $0.137$ \\
$\mathrm{res}$  & $\log_{10}$ Common Crawl pages & $7.362$ & $0.845$ \\
$\mathrm{size}$ & $\log_{2}$ parameters (billions) & $2.684$ & $1.900$ \\
\bottomrule
\end{tabular}
\caption{Standardisation constants. A coefficient divided by the SD in this table is the
effect per raw unit; for the two logged predictors that unit is one doubling.}
\label{tab:standardisation}
\end{table}

\paragraph{Why relative fertility enters as a within-language deviation.} It is the only
feature that varies across models within a language. We split it into a between-language
mean and a within-language deviation and enter only the latter; the between-language arm
correlates at $r = 0.80$ with typological distance across our 16 languages and is not
separately identifiable from it. Entering both arms is what produced a positively signed
fertility coefficient in earlier versions of this analysis.

\paragraph{Why the language intercept is load-bearing.} Resource level and typological
distance vary only across the 16 languages. Without $u_{\mathrm{language}}$ their standard
errors are computed from the model-by-language stratum and are anticonservative: refitting
without it inflates the standard errors by $2.6\times$ on exactly those two terms and
leaves the other eleven unchanged ($0.90$--$1.00\times$), while the point estimates barely
move. The consequential case is typological distance, $z = -9.8$ without the intercept
against $-3.8$ with it. We report the corrected version throughout.

\subsection{Estimates}

\begin{table}[h]
\centering
\begin{tabular}{lrrrr}
\toprule
Term & $\hat\beta$ & SE & $z$ & $p$ \\
\midrule
Intercept                              & $-0.165$ & $0.454$ & $-0.36$ & $.716$ \\
\addlinespace
Relative fertility (within)            & $-0.071$ & $0.030$ & $-2.34$ & $.020$ \\
Typological distance                   & $-0.288$ & $0.076$ & $-3.79$ & $1.5\times10^{-4}$ \\
Resource level                         & $0.712$  & $0.076$ & $9.38$  & $6.7\times10^{-21}$ \\
Model size                             & $1.881$  & $0.213$ & $8.82$  & $1.2\times10^{-18}$ \\
Reasoning (on)                         & $0.492$  & $0.314$ & $1.57$  & $.117$ \\
\addlinespace
Fertility $\times$ size                & $-0.023$ & $0.029$ & $-0.80$ & $.423$ \\
Distance $\times$ size                 & $0.017$  & $0.023$ & $0.76$  & $.448$ \\
Resource $\times$ size                 & $-0.246$ & $0.023$ & $-10.72$& $8.2\times10^{-27}$ \\
\addlinespace
Fertility $\times$ reasoning           & $0.176$  & $0.076$ & $2.32$  & $.021$ \\
Distance $\times$ reasoning            & $0.124$  & $0.050$ & $2.49$  & $.013$ \\
Resource $\times$ reasoning            & $-0.146$ & $0.050$ & $-2.91$ & $.004$ \\
Size $\times$ reasoning                & $0.321$  & $0.309$ & $1.04$  & $.299$ \\
\bottomrule
\end{tabular}
\caption{Fixed effects, in log-odds per standard deviation of the predictor.
$p$-values are Wald.}
\label{tab:fixed-effects}
\end{table}

\begin{table}[h]
\centering
\begin{tabular}{lrrr}
\toprule
Grouping factor & Levels & $\mathrm{sd}$ & Variance \\
\midrule
Family                    & 9   & $1.328$ & $1.765$ \\
Template                  & 100 & $1.221$ & $1.490$ \\
Base model within family  & 36  & $0.913$ & $0.834$ \\
Model                     & 48  & $0.828$ & $0.685$ \\
Model $\times$ language   & 768 & $0.580$ & $0.337$ \\
Language                  & 16  & $0.272$ & $0.074$ \\
\bottomrule
\end{tabular}
\caption{Random effects, ordered by magnitude. The two components the language features are
asked to explain are the last two; Section~\ref{sec:variance} reports how much of each they
account for.}
\label{tab:random-effects}
\end{table}

\subsection{Estimation}
Fitted by maximum likelihood with the Laplace approximation in \texttt{lme4} 1.1.37
\citep{bates2015lme4} under R 4.5.1, using the \texttt{bobyqa} optimiser with a limit of
$3\times10^{5}$ function evaluations. The fit converged with no warnings
($\mathrm{AIC} = 447{,}430$, $\mathrm{BIC} = 447{,}605$, $\log L = -223{,}696$).
Specifications considered and rejected are reported in
Appendix~\ref{app:sec:model-ablations}.
\end{verbatim}
```

---

## Notes on choices in the draft

- **Equation and `lme4` syntax both included.** The equation is what a reader
  parses; the syntax is what a reader reproduces, and the nesting
  (`family/base_model`) and the `*` expansions are easier to get wrong from the
  equation than to copy from the formula.
- **Group counts in the random-effects table.** They are what makes the standard
  errors in Table~\ref{tab:fixed-effects} legible --- 16 languages is the reason
  the language-level terms have the SEs they do, and it is the fact the
  load-bearing paragraph turns on.
- **Two paragraphs carried over from `draft_results.md`** ("Methods additions"):
  the fertility split and why `(1|language)` matters. They were flagged there as
  belonging in Methods; if you have already placed them in §3.3, delete them here
  and cross-reference instead of repeating.
- `sec:effect-sizes` and `sec:variance` are referenced; rename to match your
  labels.

## Numbers to double-check against your build

`N = 76{,}800` cells, 20 instances each. Level counts 9 / 100 / 36 / 48 / 768 /
16. If the dataset revision moves, all of these move with it.
