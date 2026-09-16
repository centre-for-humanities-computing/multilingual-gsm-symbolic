# What the features account for (merged §4.4 + §4.6)

Merged, as you suggested. It works, and it removes a real duplication: the old
§4.4 ended on the residual languages and the old §4.6 ended on "the residuals are
a target, not a limitation" --- the same paragraph written twice.

**It does force a reordering.** The wrap-up half synthesises the prediction
results, so the merged section has to come *after* prediction, not before it:

| | before | after |
|---|---|---|
| 4.4 | variance | **prediction** |
| 4.5 | prediction | **variance + wrap-up (closes the results)** |
| 4.6 | summary | --- |

That ordering is arguably better on its own terms. Prediction is the empirical,
practitioner-facing result; the variance split is the mechanism that explains why
it comes out the way it does. Stating the result and then explaining it reads
more naturally than the reverse, and it lets the results end on something with
weight rather than on an adequacy check.

**One edit needed in `draft_prediction.md`.** It currently forward-references the
variance split and quotes the 89/20 numbers itself, which would now duplicate the
section that follows it. Replace

> The ordering of the two follows from Section~\ref{sec:variance}: the language
> features account for 89\% of between-language variance but only 20\% of
> model-by-language variance, so forecasting \emph{which language is hard} works
> better than forecasting \emph{which model will struggle with it}.

with

> Forecasting \emph{which language is hard} therefore works considerably better
> than forecasting \emph{which model will struggle with it} ---
> Section~\ref{sec:variance} takes up why.

**Title.** `What the features account for` below. Alternatives if you want the
wrap-up signalled more strongly: `Taking stock`, or `How much of transfer we
account for`.

---

## Merged subsection

```latex
\subsection{What the features account for}
\label{sec:variance}
With a language term in the model, ``how much do these features explain'' separates into
two questions with different answers. Adding the language features to a size-only baseline
removes 89\% of the between-language variance but only 20\% of the model-by-language
variance. Three features are close to sufficient for how hard a language is, and close to
useless for which model will struggle with it. The second is a genuine negative result
rather than a limitation of the fit, and it is what sets the ceiling on the forecasts
above: model-specific transfer failure is not predictable from properties of the language,
because there is little in it to predict from them.

These are variance-component ratios on the log-odds scale rather than a model $R^2$, and
they describe this 16-language sample, which was constructed to span the resource axis and
therefore raises the share the features can account for.

The languages the features miss are informative. They over-predict Russian, German,
Icelandic and Estonian, and under-predict Norwegian, Arabic, Italian and Danish. Russian is
the clearest case: second-highest resource and low typological distance, so the model
expects far more than it observes. The languages that outperform their features share a
property those features do not encode --- Norwegian, Danish and Italian all have close,
high-resource relatives, whereas we measure distance to English alone. A resource-weighted
proximity to the whole training mixture is the obvious feature to add, and the same
languages recur out of sample, which is what makes this a target for future work rather
than a post-hoc reading of 16 residuals.
% KCE: Q: isn't this the resource x typoligical 

Two of these results are one result seen from opposite sides. Language difficulty is
largely a function of properties measurable without running anything, while which
particular model falters on which particular language is mostly idiosyncrasy. That
asymmetry determines what the model can be used for, and for a practitioner it splits
cleanly. Choosing a model for a language you have not evaluated needs no new evaluation and
no language features: the best model on the languages already tested is the best model
there too, in 15 of our 16 held-out languages. Knowing how well it will do does need the
features, and they roughly halve the error. When the answer is not good enough the
available levers are not symmetric --- scale buys back the penalty for a language being
low-resource but does nothing for one being structurally distant from English, while
reasoning buys back both. Since typological distance is the one property no amount of
engineering can change, that asymmetry is where the practical weight of these results sits.
```

---

## Notes

- `fig_variance` is cut. Removed from `make_results_figures.R` (the two ratios are
  still computed and printed, since the prose quotes them) and the PDF/PNG
  deleted. The section now carries the numbers in text, which is enough for two
  of them.
- Kept `\label{sec:variance}` so the reference from `draft_prediction.md` still
  resolves; it is now a forward reference rather than a backward one.
- `sec:predict` is no longer referenced from here --- the merged text says "the
  forecasts above" instead, since prediction is now the preceding section.

---

## current draft:


\subsection{Predicting performance on an unseen language}
Besides determining best practices for model developers, modeling how capabilities transfer also allow us estimate the performance on unseen languages, which can both save compute, but also allow plausible performance estimates of language without evaluation data.

We estimate how well we can predict performance on a language by holding out the given language and refitting eq.~\ref{eq:model} on the remaining and predict the target language.

A language's average difficulty is predictable to 3.2pp (Spearman $\rho = 0.92$)
(Figure~\ref{fig:predict-language}) and for a specific model in that language the error is 5.6pp (Spearman $\rho = 0.97$)(Figure~\ref{fig:predict-model}), a 41\% and 38\% reduction, respectively, over the naïve average baseline. If use just 10 templates we can further reduce it to 2.29pp and 3.88pp respectively. The largest failures in the held-out case are model-by-language idiosyncrasies with e.g. Apertus-70B scoring 23\% on English where its size and the language's features imply 75\%, and several large models fall well below prediction on Estonian (see Appendix~\ref{app:sec:prediction}).

\begin{figure}[t]
\begin{minipage}[t]{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fig_predict_language.pdf}
    \caption{\textbf{Forecasting language performance}: Aggregate language performance with each language held out in turn.}
    \label{fig:predict-language}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fig_predict_model_language.pdf}
    \caption{\textbf{Forecasting model performance} on a given language, showing a total of 768 (model, language) pairs. For a per language decomposition see Appendix~\ref{app:sec:prediction}.}
    \label{fig:predict-model}
\end{minipage}
\end{figure}

% some sort of bridge
Given the a language term in Eq.~\ref{eq:model} we can examine how much of our features explain. We do this by fitting a size-only baseline and see that it removes 89\% of the between-language variance but only 20\% of the model-by-language. As such we a high degree of confidence estimate language difficulty, but which specific model falters in a target language remains hard to estimate. It is likely that this estimate can be derived 
