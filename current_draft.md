
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
