# Predicting performance on an unevaluated language

Replaces §4.5 of `draft_results.md` (now a pointer). Results subsection is terse
and figure-led; the protocol and the error analysis move to the appendix.

---

## Results subsection

```latex
\subsection{Predicting performance on an unevaluated language}
The point of modelling transfer is to avoid evaluating every language. Every predictor in
Equation~\ref{eq:model} can be obtained without running the model on the language --- a
Common Crawl page count, a URIEL vector, and tokenising text with the model's own
tokenizer --- so the model can forecast rather than describe. We hold out each of the 16
languages in turn, refit on the remaining 15, and predict the held-out one
(Appendix~\ref{app:prediction}).

A language's average difficulty is predictable to 3.2 accuracy points
(Figure~\ref{fig:predict-language}), against 7.8 points for the only alternative available
without evaluation --- assuming a new language behaves like those already tested. For a
specific model in that language the error is 5.6 points (Figure~\ref{fig:predict-model}),
a 38\% reduction over the same baseline. The ordering of the two follows from
Section~\ref{sec:variance}: the language features account for 89\% of between-language
variance but only 20\% of model-by-language variance, so forecasting \emph{which language
is hard} works better than forecasting \emph{which model will struggle with it}.

The features do not, however, improve the model \emph{ranking} within a new language
(Spearman $\rho = 0.97$ either way): rankings are carried by how good a model is in
general, which transfers across languages almost perfectly, and the best model is
identified in 15 of 16 held-out languages at a mean cost of 0.13 accuracy points. Choosing
a model for a new language therefore needs no new evaluation and no language features;
knowing how well that choice will do is what the features supply.

\begin{figure}[t]
\begin{minipage}[t]{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fig_predict_language.pdf}
    \caption{\textbf{Forecasting a language.} Each language held out in turn; grey
    segments give the error.}
    \label{fig:predict-language}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fig_predict_model_language.pdf}
    \caption{\textbf{Forecasting a model in a language.} 768 (model, language) pairs.}
    \label{fig:predict-model}
\end{minipage}
\end{figure}
```

---

## Appendix

```latex
\subsection{Leave-one-language-out Prediction}
\label{app:prediction}
Each of the 16 languages is held out in turn and Equation~\ref{eq:model} refitted on the
remaining 15. Standardisation is recomputed on each training fold and applied to the
held-out language, so its own mean and spread never enter the fit. Prediction uses the
fixed effects together with the family, model, variant and template random effects, all of
which are estimated from the other languages; the language and model-by-language terms are
set to zero, since neither can be known for a language that has not been evaluated. Those
two components are therefore irreducible error, and their sizes are what separate the two
forecasting targets.

Predicted accuracy for a (model, language) pair is the mean predicted probability over the
100 templates. Note this is a conditional-mode rather than a marginal prediction; the
resulting bias is small at these probabilities and applies equally to every method
compared, so it does not affect the comparison.

Baselines are what is available without the language features: the training-set mean, the
model's own mean accuracy over the languages already evaluated, and a fitted model with
model size but no language features. The second is the relevant one --- it is what a
practitioner would assume in the absence of any feature-based estimate.

\paragraph{Where the forecast fails.}
Error is smallest at the extremes of the accuracy range and largest in the middle (MAE 5.1
points below 20\% accuracy, 9.7 between 20 and 40\%, 3.2 above 80\%), with low performers
over-predicted and high performers under-predicted --- the shrinkage expected when the
language-specific terms are set to zero. The largest individual failures are
model-by-language idiosyncrasies rather than failures of the language features:
Apertus-70B scores 23\% on English where its size and the language's features imply 75\%,
and several large models fall well below prediction on Estonian, which accounts for four
of the eight worst over-predictions (Figure~\ref{fig:predict-by-language}).

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{figures/fig_predict_by_language.pdf}
    \caption{\textbf{Forecasts by held-out language.} Vertical offset from the diagonal is
    error in the level; the ordering within a panel determines which model would be
    chosen. Numbers give per-language MAE in accuracy points.}
    \label{fig:predict-by-language}
\end{figure}
```

---

## Figure changes

All three now render without titles or subtitles, with the estimates inside the
panel so each figure carries its own numbers when read apart from the caption:

- `fig_predict_language` --- MAE and $r$ top-left
- `fig_predict_model_language` --- MAE and $r$ top-left
- `fig_predict_by_language` --- per-language MAE in each facet

`fig_predict_error` (the MAE-by-method bar chart) is no longer referenced: the
three numbers it carried now sit in the prose. It is still generated if you want
it.

## Provenance

- `paper/scripts/analysis/predict_new_language.R` -> `lolo_predictions.csv`
- Figures in `paper/scripts/analysis/make_results_figures.R`
