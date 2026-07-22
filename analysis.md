Analysis
Statistical model

To estimate which language-level features govern cross-lingual transfer, we model performance on individual problem instances with a generalized linear mixed model (GLMM). We work at the instance level with a Bernoulli response, since each generated problem is either solved correctly or not, and the matched design — approximately identical templates across languages — let us separate language effects from problem difficulty. Let $y_{mlti} \in {0,1}$ denote whether model $m$ solves instance $i$ of template $t$ in language $l$, where $m$ indexes the evaluated models, $l$ the languages, $t$ the problem templates, and $i$ the individual instances generated from a given template. We model


$$ 
\begin{aligned} y_{mlti} &\sim \mathrm{Bernoulli}(\pi_{mlti}), \ \mathrm{logit}(\pi_{mlti}) &= \mathbf{x}{ml}^\top \boldsymbol{\beta} + u_m + v_t + w{ml}, \ u_m &\sim \mathcal{N}(0, \sigma^2_m), \ v_t &\sim \mathcal{N}(0, \sigma^2_t), \ w_{ml} &\sim \mathcal{N}(0, \sigma^2_{ml}), \end{aligned} 
$$
(KCE: since slack does not render latex, here is a screenshot render:)




where $\mathbf{x}{ml}$ is a vector of language-level features (e.g. tokenizer fertility, typological distance from the reference language, and a resource-quantity proxy), possibly interacted with model-level covariates, and $\boldsymbol{\beta}$ are the corresponding fixed effects. The random intercepts $u_m$ and $v_t$ capture overall differences in model capability and template difficulty, respectively, while the model-by-language term $w{ml}$ absorbs residual variation in how a given model transfers to a given language. This last term is our quantity of interest: $\sigma^2_{ml}$ measures the cross-lingual transfer variation that remains after accounting for model strength and problem difficulty, and the reduction in $\sigma^2_{ml}$ when the language features $\mathbf{x}_{ml}$ are added to the fixed effects quantifies how much of the transfer variation those features explain.
The random effects for model and template account for the non-independence induced by the matched design — the same templates are reused across all models and languages, so their difficulties are shared rather than private to any one model — and partial pooling stabilizes per-cell estimates for cells with few instances. 

Practicalities
This model can be fit either using a Bayesian or frequentist framework, though a bayesian allow us to obtain full posterior intervals on both the fixed-effect coefficients and the variance components, the latter being important given the limited number of languages, where a point estimate of "variance explained" would otherwise be misleadingly precise. However will probably do the first set of modelling in lme4

Features
(KCE: This is just the initial suggested features which can be expanded)

We distinguish language features (properties of a language, possibly relative to a reference) from model features (properties of the evaluated system). 
Language features enter the model as the fixed-effect vector $\mathbf{x}_{ml}$; model features are either modeled as random-effect groupings or included as covariates.

Language features
Tokenizer fertility. Tokens per character when the model's tokenizer encodes the language, normalized to the reference language (English) to make values comparable across tokenizers with different vocabulary sizes. 
Typological distance. Distance from the reference language using e.g. URIEL/lang2vec feature vectors (syntactic and inventory features), capturing how structurally far a language is from the one the model knows best.
Resource quantity (proxy). A proxy for how much of the language the model was exposed to in training, since true per-language pretraining counts are usually undisclosed. We use [Common Crawl token share / Wikipedia size / documented coverage] on a log scale.
Script (shared vs. not). Whether the language shares its script with the model's dominant training language. 
Model features
Family. Architecture, tokenizer, and training recipe, held as a grouping factor.
Size. Parameter count (log scale), to estimate scaling of the transfer gap within family.

Using standard lme4/brms formula syntax, the model is:
where fertility, typological distance and log resource is the example of x

correct ~ fertility + typological_distance + log_resource +
          (1 | model) + (1 | template) + (1 | model:language)

