# Bayesian Gaussian Mixture (sklearn.mixture.BayesianGaussianMixture)

<!--
tier: 3-component
status: draft
baseline-commit: 4ae6af03
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/mixture/_bayesian_mixture.py   # _log_dirichlet_norm (:26-41); _log_wishart_norm (:44-72); class BayesianGaussianMixture(BaseMixture) (:75); _parameter_constraints (:349-367); __init__ defaults n_components=1/covariance_type='full'/tol=1e-3/reg_covar=1e-6/max_iter=100/n_init=1/init_params='kmeans'/weight_concentration_prior_type='dirichlet_process'/weight_concentration_prior=None/mean_precision_prior=None/mean_prior=None/degrees_of_freedom_prior=None/covariance_prior=None/random_state=None/warm_start=False (:369-409); _check_parameters/priors (:411-511); _estimate_weights (Dirichlet/DP cumsum :530-549); _estimate_means (Bayesian shrinkage :551-563); _estimate_precisions/_estimate_wishart_* (inverse-Wishart covariances_ + degrees_of_freedom_ + precisions_cholesky_ :565-721); _m_step (:723-741); _estimate_log_weights (digamma :743-759); _estimate_log_prob (Wishart precision expectation :761-777); _compute_lower_bound (true ELBO with KL terms :779-836); _get/_set_parameters + weights_/precisions_ derivation (:838-889)
  - sklearn/mixture/_base.py               # BaseMixture (:43); _parameter_constraints (:50-63); fit (:154-182); fit_predict — n_init loop, _initialize_parameters, E/M loop, lower_bound convergence, converged_/n_iter_/lower_bound_, final e-step argmax labels (:185-288); _e_step (:290-307); _initialize_parameters init_params kmeans/random/random_from_data/k-means++ (:99-140); score_samples/score/predict/predict_proba (:331-404); _estimate_weighted_log_prob (:468-479); _estimate_log_prob_resp logsumexp normalization (:507-531)
ferrolearn-module: ferrolearn-cluster/src/bayesian_gmm.rs
parity-ops: BayesianGaussianMixture (.__init__, .fit, .fit_predict, .predict, .predict_proba, .score_samples, .score, .bic, .aic, .weights_, .means_, .covariances_, .converged_, .lower_bound_, .n_iter_, .weight_concentration_, .mean_precision_, .degrees_of_freedom_, .precisions_, .precisions_cholesky_)
crosslink-issue: 1056
-->

## Summary

`ferrolearn-cluster/src/bayesian_gmm.rs` is a translation TARGET for scikit-learn's
`BayesianGaussianMixture` (`sklearn/mixture/_bayesian_mixture.py`,
`class BayesianGaussianMixture(BaseMixture)` `:75`) — variational Bayesian estimation
of a Gaussian mixture with automatic component pruning via a Dirichlet/Dirichlet-Process
weight prior. It exposes the unfitted `BayesianGaussianMixture<F>`
(`n_components`, `covariance_type`, `max_iter`, `tol`, `weight_concentration_prior_type`,
`weight_concentration_prior`, `random_state`), the fitted
`FittedBayesianGaussianMixture<F>` (`weights_`, `means_`, `covariances_`, `converged_`,
`lower_bound_`, `n_features_`), a `Fit<Array2<F>, ()>` impl, a `Predict<Array2<F>>` impl
(hard argmax of responsibilities), a `fit_predict` convenience, and the
`predict_proba`/`score_samples`/`score`/`bic`/`aic` methods. It is re-exported at the
crate root (`pub use bayesian_gmm::{BayesianCovType, BayesianGaussianMixture,
FittedBayesianGaussianMixture, WeightPriorType}` in `ferrolearn-cluster/src/lib.rs`).

**Honest assessment (R-HONEST-3) — the structural carve-out.** ferrolearn's
`bayesian_gmm.rs` is a **heuristic plain-EM approximation, NOT scikit-learn's
variational Bayes**. This is the headline divergence, not a numerical tolerance gap.
What SHIPS, through the crate re-export, is narrow:

1. the **hard-label PARTITION up to a permutation on well-separated blobs** (REQ-1) —
   the in-tree `test_bayesian_gmm_two_blobs_separation` confirms ferrolearn does
   separate well-separated blobs despite the heuristic; and
2. the **API/output-shape contracts** of `predict` / `predict_proba` /
   `score_samples` / `score` and the **weights-sum-to-1** invariant (REQ-2) — shapes
   and the row-stochastic responsibility matrix are correct.

Everything that makes this estimator *Bayesian* DIVERGES. ferrolearn `fn
run_variational_em` runs a **plain maximum-likelihood EM** (ML mean update `means =
Σ resp·x / nk`, ML covariance, a simplified Dirichlet/DP `alpha` update, weights =
`alpha/alpha_sum`) with **no digamma E-step, no Wishart precision expectation, no
mean-precision/degrees-of-freedom priors, and no `precisions_cholesky_`**. The
`weights_` / `means_` / `covariances_` / `lower_bound_` VALUES and the whole Bayesian
attribute surface (`weight_concentration_`, `mean_precision_`, `degrees_of_freedom_`,
`precisions_`, `precisions_cholesky_`, `n_iter_`) diverge (REQ-3..REQ-9). And there is
**no PyO3 binding for `BayesianGaussianMixture`** (confirmed by grep — see Probe 5):
`ferrolearn-python` registers `_RsGaussianMixture` for the PLAIN GaussianMixture
(`gmm.rs`), not `_RsBayesianGaussianMixture`. The only non-test consumer is the crate
re-export (grandfathered S5/R-DEFER-1 — `BayesianGaussianMixture` IS the public API
type).

**Conclusion on a minimal fix: green-guard only — there is no single clean
minimal-fixable divergence.** The divergence IS the entire VB algorithm. The defaults
ferrolearn already exposes mostly match sklearn (`covariance_type='full'`,
`max_iter=100`, `tol=1e-3`, `weight_concentration_prior_type='dirichlet_process'` —
REQ-9 SHIPPED). There is no wrong-default to flip and no localized contract bug whose
fix would close any DIVERGING value REQ; correctness requires replacing plain EM with
the digamma/Wishart variational machinery wholesale (REQ-3 through REQ-7), which is
builder-scale work, NOT a fixer pin. The one localized DIVERGENCE that is itself a
sub-component — the **diagonal-only Mahalanobis in `Full`/`Tied` scoring** (REQ-4,
ignores off-diagonal covariance, R-DEV-1, wrong on correlated data) — is a genuine
localized bug but still requires a full-covariance Cholesky-solve rewrite of the score
path; it is pinned as its own NOT-STARTED REQ.

## Live oracle probes (sklearn 1.5.2, run from /tmp; ferrolearn via `cargo test -p ferrolearn-cluster --lib`)

All expected values come from the installed sklearn 1.5.2 oracle (run from `/tmp`,
`random_state` fixed) or a sklearn `file:line`, never literal-copied from ferrolearn
(R-CHAR-3).

### Probe 1 — well-separated 3-blob partition (the SHIPPED co-membership contract)

```
python3 -c "import numpy as np; from sklearn.mixture import BayesianGaussianMixture; \
b1=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.05,0.05]]); \
b2=np.array([[10.,10.],[10.1,10.],[10.,10.1],[10.05,10.05]]); \
b3=np.array([[0.,10.],[0.1,10.],[0.,10.1],[0.05,10.05]]); \
X=np.vstack([b1,b2,b3]); m=BayesianGaussianMixture(n_components=5,random_state=0).fit(X); \
print(m.fit_predict(X).tolist())"
# labels [1, 1, 1, 1, 4, 4, 4, 4, 3, 3, 3, 3]   (three blobs -> three distinct labels)
```
The 12 points form three well-separated 4-point blobs and sklearn assigns each blob a
single (distinct) component label. ferrolearn's `fn fit` + `Predict::predict` recovers
the SAME 3-block co-membership partition (points within a blob share a label, points
across blobs do not) — this is the kind of result the in-tree
`test_bayesian_gmm_two_blobs_separation` pins. The PARTITION up-to-permutation SHIPS;
the absolute label integers and which components survive pruning do NOT (sklearn prunes
via the variational weight; ferrolearn's heuristic alpha-update does not match).

### Probe 2 — the Bayesian attribute surface ferrolearn LACKS (REQ-3..REQ-7)

```
python3 -c "import numpy as np; from sklearn.mixture import BayesianGaussianMixture; \
X=np.array([[1.,2.],[1.,4.],[1.,0.],[4.,2.],[12.,4.],[10.,7.]]); \
m=BayesianGaussianMixture(n_components=2,random_state=42).fit(X); \
print('weights_', np.round(m.weights_,5).tolist()); \
print('means_', np.round(m.means_,4).tolist()); \
print('covariances_.shape', m.covariances_.shape); \
print('weight_concentration_(DP tuple)', [np.round(t,4).tolist() for t in m.weight_concentration_]); \
print('mean_precision_', np.round(m.mean_precision_,4).tolist()); \
print('degrees_of_freedom_', np.round(m.degrees_of_freedom_,4).tolist()); \
print('precisions_cholesky_.shape', m.precisions_cholesky_.shape); \
print('n_iter_', m.n_iter_, 'lower_bound_', round(float(m.lower_bound_),6))"
# weights_ [0.68141, 0.31859]
# means_ [[2.4976, 2.2923], [8.4553, 4.5226]]
# covariances_.shape (2, 2, 2)                       <- (n_components, n_features, n_features)
# weight_concentration_(DP tuple) [[4.8636, 3.1364], [2.6364, 0.5]]   <- Beta (a, b) pair
# mean_precision_ [4.8636, 3.1364]
# degrees_of_freedom_ [5.8636, 4.1364]
# precisions_cholesky_.shape (2, 2, 2)
# n_iter_ 5   lower_bound_ -29.559463
```
sklearn exposes `weight_concentration_` (a `(a, b)` Beta-parameter tuple for the
Dirichlet-Process; an `array` for the Dirichlet-distribution), `mean_precision_`,
`degrees_of_freedom_`, `precisions_`, `precisions_cholesky_`, `n_iter_`, and a true
`lower_bound_` (`-29.559463`). The `means_` and `covariances_` are **inverse-Wishart /
Bayesian-shrinkage** quantities (`_estimate_means` `:551-563`, `_estimate_wishart_full`
`:592-626`), NOT the ML means/covariances ferrolearn computes. ferrolearn's
`FittedBayesianGaussianMixture` exposes ONLY `weights()` / `means()` / `covariances()`
/ `converged()` / `lower_bound()` (a proxy) / `n_features()` / `weight_prior_type()` —
none of the Wishart/Dirichlet attributes, no `n_iter_`. Note `covariances_` shape also
differs: sklearn `Full` = `(k, d, d)` 3-D; ferrolearn flattens to a 2-D
`(k·d, d)` block layout (`FittedBayesianGaussianMixture.covariances_` doc).

### Probe 3 — DirichletDistribution `weight_concentration_` (REQ-3/REQ-6)

```
python3 -c "import numpy as np; from sklearn.mixture import BayesianGaussianMixture; \
X=np.array([[1.,2.],[1.,4.],[1.,0.],[4.,2.],[12.,4.],[10.,7.]]); \
m=BayesianGaussianMixture(n_components=3,weight_concentration_prior_type='dirichlet_distribution',random_state=42).fit(X); \
print('weight_concentration_', np.round(m.weight_concentration_,4).tolist()); \
print('weights_', np.round(m.weights_,5).tolist())"
# weight_concentration_ [0.3405, 2.6833, 3.9762]    <- Dirichlet params = prior + nk
# weights_ [0.04864, 0.38333, 0.56803]
```
For the Dirichlet-distribution prior sklearn `_estimate_weights` (`:547-549`) sets
`weight_concentration_ = weight_concentration_prior_ + nk` (here `nk` are
**digamma-based variational** counts, not ML counts), then derives `weights_ =
weight_concentration_ / sum` (`_set_parameters` `:870-873`). ferrolearn's `alpha`
update (`fn run_variational_em`, the `DirichletDistribution` arm) does `alpha[ki] =
weight_concentration + n_k[ki]` with `n_k` from the **heuristic responsibilities**, so
both `weight_concentration_` (not exposed at all) and the resulting `weights_` VALUES
diverge.

### Probe 4 — correlated covariance: diagonal-only vs full Mahalanobis (REQ-4, R-DEV-1)

```
python3 -c "import numpy as np; from sklearn.mixture import BayesianGaussianMixture; \
np.random.seed(1); \
A=np.random.multivariate_normal([0,0],np.array([[1.,0.9],[0.9,1.]]),40); \
B=np.random.multivariate_normal([0,0],np.array([[1.,-0.9],[-0.9,1.]]),40); \
X=np.vstack([A,B]); m=BayesianGaussianMixture(n_components=2,covariance_type='full',random_state=0,max_iter=300).fit(X); \
tp=np.array([[2.,2.],[2.,-2.]]); \
print('predict test pts', m.predict(tp).tolist()); \
print('score_samples', np.round(m.score_samples(tp),5).tolist()); \
print('covariances_', np.round(m.covariances_,4).tolist())"
# predict test pts [0, 1]            <- (2,2) -> +corr cluster 0 ; (2,-2) -> -corr cluster 1
# score_samples [-5.35346, -5.59716]
# covariances_ [[[0.6348, 0.6149], [0.6149, 0.8278]], [[0.6043, -0.4915], [-0.4915, 0.5658]]]
```
Two zero-mean clusters distinguished ONLY by the sign of their correlation
(off-diagonal covariance ≈ ±0.6). sklearn's `_estimate_log_prob` uses the full
`precisions_cholesky_` (a Cholesky solve against the FULL covariance), so a point at
`(2,2)` (along the +corr axis) scores into cluster 0 and `(2,-2)` into cluster 1.
ferrolearn `fn compute_responsibilities` / `fn unnormalized_log_prob` (the `Full |
Tied` arm) read ONLY `covariances[[offset+j, j]]` — the **diagonal** — and compute
`Σ d²/var_jj + Σ ln var_jj` (a diagonal pseudo-Mahalanobis, with the explicit comment
`// Simple squared distance using diagonal only for robustness.`). Because both
clusters have near-identical diagonal variances and zero mean, the diagonal-only score
cannot separate them — ferrolearn assigns both test points to whichever component the
heuristic favors. Even when the M-step *computes* the off-diagonal covariance
(`fn run_variational_em`, the `Full|Tied` arm fills `covariances[[offset+j1, j2]]`),
prediction/scoring throws it away. This is a real R-DEV-1 divergence: ferrolearn is
wrong on correlated data.

### Probe 5 — non-test consumer: NO PyO3 binding for BayesianGaussianMixture

```
grep -rn "Bayesian\|GaussianMixture" ferrolearn-python/src/lib.rs ferrolearn-python/src/extras.rs
# ferrolearn-python/src/lib.rs:69:    m.add_class::<extras::RsGaussianMixture>()?;
# ferrolearn-python/src/extras.rs:1038: #[pyclass(name = "_RsGaussianMixture")]  (PLAIN GaussianMixture/gmm.rs)
# ferrolearn-python/src/extras.rs:156:  RsBayesianRidge ...                       (linear, unrelated)
# (no _RsBayesianGaussianMixture, no RsBayesianGaussianMixture anywhere)
grep -n "bayesian_gmm\|BayesianGaussianMixture" ferrolearn-cluster/src/lib.rs
# ferrolearn-cluster/src/lib.rs:98-99: pub use bayesian_gmm::{BayesianCovType,
#   BayesianGaussianMixture, FittedBayesianGaussianMixture, WeightPriorType};
```
`_RsGaussianMixture` wraps `ferrolearn_cluster::FittedGaussianMixture` (the PLAIN
`gmm.rs`), NOT the Bayesian variant; `RsBayesianRidge` is an unrelated linear
estimator. There is **no `_RsBayesianGaussianMixture`**, so `import ferrolearn` cannot
reach `BayesianGaussianMixture`. The only non-test production consumer is the crate
re-export in `ferrolearn-cluster/src/lib.rs`. Per S5/R-DEFER-1 the boundary type
`BayesianGaussianMixture` IS the public API (grandfathered), so the partition + API
contract REQs SHIP on the strength of that re-export + external library callers; a
dedicated PyO3-binding REQ (REQ-10) is NOT-STARTED.

### Probe 6 — defaults (REQ-9 SHIPPED; REQ-8 missing params)

sklearn `__init__` (`_bayesian_mixture.py:369-409`) + `BaseMixture._parameter_constraints`
(`_base.py:50-63`): `covariance_type='full'`, `max_iter=100`, `tol=1e-3`,
`weight_concentration_prior_type='dirichlet_process'` — all four MATCH ferrolearn
`fn new` (`covariance_type: BayesianCovType::Full`, `max_iter: 100`, `tol: 1e-3`,
`weight_concentration_prior_type: WeightPriorType::DirichletProcess`). The divergent
defaults: sklearn `n_components=1` (ferrolearn `new(n_components)` requires it
explicitly — minor); sklearn `init_params='kmeans'` (ferrolearn samples k random rows
+ jitter, a `random_from_data`-like init — REQ-8); sklearn `random_state=None` is
nondeterministic, ferrolearn `fn fit` uses `random_state.unwrap_or(42)` (seeds to 42 —
deterministic, R-DEV-1: documented determinism). MISSING sklearn params entirely:
`reg_covar` (ferrolearn hardcodes `1e-6`), `n_init`, `init_params`,
`mean_precision_prior`, `mean_prior`, `degrees_of_freedom_prior`, `covariance_prior`,
`warm_start`, `verbose`, `verbose_interval` (REQ-8).

## Requirements

- REQ-1 (well-separated PARTITION): `fit_predict` / `fit().predict()` recover the same
  hard-label co-membership (up to a permutation) as
  `BayesianGaussianMixture.fit_predict` on well-separated blobs (Probe 1; sklearn final
  e-step `argmax` `_base.py:286-288`).
- REQ-2 (API/output-shape contracts): `predict_proba` returns an
  `(n_samples, n_components)` row-stochastic matrix (each row sums to 1, mirroring
  `BaseMixture.predict_proba` `_base.py:387-404` + `_estimate_log_prob_resp` logsumexp
  normalization `:507-531`); `predict` returns `(n_samples,)`; `score_samples` returns
  `(n_samples,)`; `score` is their mean; and `weights_` sums to 1.
- REQ-3 (variational E/M-step — the algorithm): the E-step is the digamma-based
  `_estimate_log_weights` (Dirichlet/DP expectation, `:743-759`) + `_estimate_log_prob`
  (Wishart precision expectation with `degrees_of_freedom_` and `log|Λ|` digamma sum,
  `:761-777`); the M-step is the Bayesian conjugate update `_estimate_weights` /
  `_estimate_means` / `_estimate_precisions` (`:530-721`). ferrolearn runs plain ML-EM
  with a simplified alpha update — no digamma, no Wishart.
- REQ-4 (full-covariance Mahalanobis in scoring, R-DEV-1): `Full`/`Tied` scoring uses
  the FULL covariance via `precisions_cholesky_` (true `dᵀΣ⁻¹d` + `log|Σ|`,
  `_gaussian_mixture._estimate_log_gaussian_prob` via `_estimate_log_prob` `:761-777`).
  ferrolearn `fn compute_responsibilities` / `fn unnormalized_log_prob` use the
  diagonal only (Probe 4) — wrong on correlated data.
- REQ-5 (`lower_bound_` true ELBO): `lower_bound_` is sklearn's `_compute_lower_bound`
  (`-Σ exp(log_resp)·log_resp - log_wishart - log_norm_weight - 0.5·d·Σ
  ln(mean_precision_)`, `:779-836`) — the variational ELBO with Dirichlet/Wishart KL
  terms. ferrolearn `fn run_variational_em` stores a PROXY: the per-sample mean of
  `logsumexp(log resp)` (average responsibility log-likelihood).
- REQ-6 (Bayesian fitted attributes): expose `weight_concentration_`,
  `mean_precision_`, `degrees_of_freedom_`, `precisions_`, `precisions_cholesky_`,
  and inverse-Wishart-derived `covariances_`/`means_` (`_get/_set_parameters`
  `:838-889`, Probe 2/3). ferrolearn exposes only `weights_`/`means_`(ML)/
  `covariances_`(ML)/`converged_`/`lower_bound_`(proxy)/`n_features_`.
- REQ-7 (Bayesian priors + `n_init`/`warm_start`/`reg_covar` params): support
  `mean_precision_prior`, `mean_prior`, `degrees_of_freedom_prior`, `covariance_prior`
  (`:155-178`, `_check_*` `:411-511`), `reg_covar` (`:116`, ferrolearn hardcodes 1e-6),
  `n_init` (best-of-N by `lower_bound_`, `_base.py:231-264`), `warm_start`
  (`_base.py:222-223`). The `means_`/`covariances_`/`degrees_of_freedom_` VALUES are
  derived from these priors (Probe 2).
- REQ-8 (`init_params` + remaining ctor surface): support
  `init_params ∈ {'kmeans','k-means++','random','random_from_data'}` default `'kmeans'`
  (`_base.py:99-140`); `verbose`/`verbose_interval`; and `n_components=1` default.
  ferrolearn `fn init_means` does a `random_from_data`-like sample + jitter only.
- REQ-9 (matching defaults — SHIPPED): `covariance_type='full'`, `max_iter=100`,
  `tol=1e-3`, `weight_concentration_prior_type='dirichlet_process'` match sklearn
  (Probe 6). `random_state` default is a documented R-DEV-1 deviation (ferrolearn seeds
  to 42 when `None`; sklearn `None` is nondeterministic).
- REQ-10 (PyO3 binding): bind `BayesianGaussianMixture` into `ferrolearn-python` as
  `_RsBayesianGaussianMixture` (mirroring `_RsGaussianMixture` for the plain variant),
  a non-test CPython consumer. None exists (Probe 5).
- REQ-11 (`n_iter_` attribute): expose `n_iter_` = the number of EM steps of the best
  init (`_base.py:263,280`). ferrolearn stores no iteration count.
- REQ-12 (numpy-RNG parity): exact `init_params`-driven VALUE parity requires matching
  numpy's `RandomState`/`check_random_state` stream; ferrolearn uses `StdRng` (Probe 6).
  Couples to REQ-13.
- REQ-13 (ferray substrate, R-SUBSTRATE): the array / linalg / RNG / special-function
  layer is `ferray-core` / `ferray::linalg` / `ferray::random` / `ferray::stats`
  (digamma/gammaln/betaln live in scipy.special; the ferray analog), not `ndarray` /
  `rand` / (absent) `statrs`.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`,
`random_state` fixed), never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED): on the 3-blob fixture (Probe 1) the ferrolearn co-membership
  partition equals sklearn's up to a permutation — each 4-point blob gets one label,
  no label shared across blobs (in-tree `test_bayesian_gmm_two_blobs_separation`
  exercises the 2-blob case).
- AC-2 (REQ-2, SHIPPED): `predict_proba(X)` rows each sum to 1 (in-tree
  `test_bayesian_gmm_weights_sum_to_one` pins the analogous `weights_` sum); shapes
  `predict → (n,)`, `predict_proba → (n, k)`, `score_samples → (n,)`,
  `means_ → (k, d)` (`test_bayesian_gmm_means_shape`).
- AC-3 (REQ-3/REQ-6, DIVERGES): on Probe 2's `X`, `means_` →
  `[[2.4976,2.2923],[8.4553,4.5226]]`, `weights_` → `[0.68141,0.31859]`,
  `weight_concentration_` → `([4.8636,3.1364],[2.6364,0.5])`, `degrees_of_freedom_` →
  `[5.8636,4.1364]`. ferrolearn produces ML means + has none of the Bayesian
  attributes.
- AC-4 (REQ-4, DIVERGES): on Probe 4's correlated fixture, `predict([[2,2],[2,-2]])` →
  sklearn `[0,1]` (full Mahalanobis separates by correlation sign); ferrolearn's
  diagonal-only score cannot separate the two clusters.
- AC-5 (REQ-5, DIVERGES): on Probe 2's `X`, `lower_bound_` → sklearn `-29.559463` (true
  ELBO); ferrolearn's proxy (mean logsumexp of responsibilities) is a different
  quantity with a different scale and sign behavior.
- AC-6 (REQ-9, SHIPPED): `BayesianGaussianMixture().get_params()` reports
  `covariance_type='full'`, `max_iter=100`, `tol=0.001`,
  `weight_concentration_prior_type='dirichlet_process'` — matching ferrolearn `fn new`.
- AC-7 (REQ-10/REQ-11, DIVERGES): `import ferrolearn; ferrolearn.BayesianGaussianMixture`
  does not exist (no binding); `fitted.n_iter_` has no ferrolearn analog.

## REQ status

Binary (R-DEFER-2). `BayesianGaussianMixture` / `FittedBayesianGaussianMixture` are
existing pub APIs re-exported at the crate root (the only non-test consumer;
grandfathered S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2, run from
`/tmp`. Honest assessment (R-HONEST-3): **two REQs SHIP** — the API/output-shape +
weights-sum-to-1 contracts (REQ-2) and the matching defaults (REQ-9) — through the
crate re-export. **REQ-1 (partition matches sklearn) is NOT-STARTED #1067: ferrolearn
lacks sklearn's DP component pruning, so the partition diverges when sklearn prunes
(2-blob: sklearn 1 cluster, ferrolearn 2); the entire VB algorithm, every Bayesian
fitted-attribute value, the true ELBO, the full-covariance score, and the PyO3 binding
are NOT-STARTED.** Blocker numbers below
are the real filed issues (#1057–#1066).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`labels_` PARTITION matches sklearn) | NOT-STARTED | open prereq blocker #1067. ferrolearn does NOT replicate sklearn's automatic DP component pruning, so the partition diverges whenever sklearn prunes: on two blobs at (0,0)/(20,20) with `n_components=5`, live sklearn prunes to ONE dominant component (`fit_predict -> [1;8]`, `weights_ ~ [0.12,0.86,...]`) while ferrolearn keeps two. A prior draft green-guarded a FABRICATED 2-blob expected value (R-CHAR-3 violation, removed; R-HONEST-4). The `char_bgm_three_blob_no_prune_partition_matches_sklearn` characterization (live-oracle-verified) shows they coincide ONLY on no-prune fixtures; the general partition-match is gated on the VB/DP algorithm (REQ-3 #1057). |
| REQ-2 (API/output-shape + weights-sum-to-1 contracts) | SHIPPED | impl `fn predict_proba` (`compute_responsibilities`, row logsumexp-normalized) returns an `(n,k)` row-stochastic matrix mirroring `BaseMixture.predict_proba`/`_estimate_log_prob_resp` (`_base.py:387-404`,`:507-531`); `Predict::predict` → `(n,)`; `fn score_samples` → `(n,)`; `fn score` = their mean. `weights_` sums to 1 (`fn run_variational_em` `weights[ki] = alpha[ki]/alpha_sum`). Consumer: crate re-export in `lib.rs`. Verification: `test_bayesian_gmm_weights_sum_to_one`, `test_bayesian_gmm_means_shape`, `test_bayesian_gmm_predict_shape_mismatch` pass. Underclaim: SHAPES + normalization only — the proba/score VALUES are built on the diagonal-only diverging score (REQ-4). |
| REQ-9 (matching defaults) | SHIPPED | impl `fn new` sets `covariance_type=Full`, `max_iter=100`, `tol=1e-3`, `weight_concentration_prior_type=DirichletProcess` — matching sklearn `__init__` (`_bayesian_mixture.py:373-379`) + `BaseMixture._parameter_constraints` (`_base.py:50-63`) (Probe 6/AC-6). Consumer: crate re-export in `lib.rs`. `random_state` default is a documented R-DEV-1 deviation (`fn fit` `random_state.unwrap_or(42)` — deterministic vs sklearn nondeterministic `None`). Underclaim: the `n_components=1` default + `init_params='kmeans'` default do NOT match (REQ-8). |
| REQ-3 (variational E/M-step — the algorithm) | NOT-STARTED | open prereq blocker #1057. ferrolearn `fn run_variational_em` runs plain ML-EM (`means = Σ resp·x / nk`, ML covariance, `alpha[ki]=wc+n_k[ki]` heuristic) — NO digamma E-step (`_estimate_log_weights` `:743-759`), NO Wishart precision expectation (`_estimate_log_prob` `:761-777`), NO Bayesian conjugate M-step (`_estimate_means` `:551-563`, `_estimate_precisions` `:565-721`). The whole algorithm diverges; this is builder-scale, not a fixer pin. |
| REQ-4 (full-covariance Mahalanobis in scoring, R-DEV-1) | NOT-STARTED | open prereq blocker #1058. `fn compute_responsibilities` / `fn unnormalized_log_prob` `Full \| Tied` arm reads only `covariances[[offset+j, j]]` (the diagonal; explicit comment `// Simple squared distance using diagonal only for robustness.`) → `Σ d²/var_jj + Σ ln var_jj`, ignoring off-diagonal terms and the true `dᵀΣ⁻¹d`/`log|Σ|`. sklearn scores via full `precisions_cholesky_` (`_estimate_log_prob` `:761-777`). Wrong on correlated data (Probe 4/AC-4): sklearn separates by correlation sign, ferrolearn cannot. Localized but needs a full-covariance Cholesky-solve rewrite. |
| REQ-5 (`lower_bound_` true ELBO) | NOT-STARTED | open prereq blocker #1059. `fn run_variational_em` stores `lower_bound_ = mean_i logsumexp_k(log resp[i,k])` (a proxy, the `elbo = ll / n_f` block). sklearn `_compute_lower_bound` (`:779-836`) is the variational ELBO `-Σ exp(log_resp)·log_resp - log_wishart - log_norm_weight - 0.5·d·Σ ln(mean_precision_)` with Dirichlet/Wishart KL terms (`-29.559463` on Probe 2, AC-5). Different quantity; depends on REQ-3/REQ-6. |
| REQ-6 (Bayesian fitted attributes) | NOT-STARTED | open prereq blocker #1060. `FittedBayesianGaussianMixture` exposes `weights()`/`means()`(ML)/`covariances()`(ML, 2-D block layout)/`converged()`/`lower_bound()`(proxy)/`n_features()`/`weight_prior_type()`. MISSING `weight_concentration_` (DP Beta tuple / Dirichlet array), `mean_precision_`, `degrees_of_freedom_`, `precisions_`, `precisions_cholesky_`, and inverse-Wishart `means_`/`covariances_` `(k,d,d)` shape (sklearn `_get/_set_parameters` `:838-889`, Probe 2/3). Depends on REQ-3. |
| REQ-7 (Bayesian priors + `n_init`/`warm_start`/`reg_covar`) | NOT-STARTED | open prereq blocker #1061. `BayesianGaussianMixture<F>` has `n_components`/`covariance_type`/`max_iter`/`tol`/`weight_concentration_prior_type`/`weight_concentration_prior`/`random_state` only. MISSING `mean_precision_prior`/`mean_prior`/`degrees_of_freedom_prior`/`covariance_prior` (`:155-178`), `reg_covar` (hardcoded `1e-6` in `fn run_variational_em`), `n_init` (best-of-N, `_base.py:231-264`), `warm_start` (`_base.py:222-223`). These priors drive the `means_`/`covariances_`/`degrees_of_freedom_` VALUES (Probe 2). Depends on REQ-3. |
| REQ-8 (`init_params` + remaining ctor surface) | NOT-STARTED | open prereq blocker #1062. sklearn `init_params ∈ {'kmeans','k-means++','random','random_from_data'}` default `'kmeans'` (`_base.py:99-140`,`:56-58`). ferrolearn `fn init_means` samples k random rows + jitter (a `random_from_data`-like init), no `init_params` param, no `verbose`/`verbose_interval`, and requires explicit `n_components` (sklearn default 1). |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #1063. `grep` of `ferrolearn-python/` shows `_RsGaussianMixture` (PLAIN `gmm.rs`) and `RsBayesianRidge` (linear) but NO `_RsBayesianGaussianMixture` (Probe 5). `import ferrolearn` cannot reach `BayesianGaussianMixture`. The only non-test consumer is the crate re-export (`lib.rs`). |
| REQ-11 (`n_iter_` attribute) | NOT-STARTED | open prereq blocker #1064. sklearn `n_iter_` = EM steps of the best init (`_base.py:263,280`; `5` on Probe 2). `FittedBayesianGaussianMixture` stores no iteration count (`fn run_variational_em` returns `converged`/`prev_elbo` but discards the loop counter `_iteration`). |
| REQ-12 (numpy-RNG parity) | NOT-STARTED | open prereq blocker #1065. Exact `init_params`-driven VALUE parity needs numpy's `RandomState`/`check_random_state` stream (`_base.py:228`); ferrolearn `fn fit` uses `StdRng::seed_from_u64` (Probe 6). Couples to REQ-13 (ferray::random) — exact-value pins blocked until the RNG substrate matches. |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1066. `bayesian_gmm.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` + `rand::{Rng, SeedableRng, rngs::StdRng}`, not `ferray-core` / `ferray::linalg` / `ferray::random` / `ferray::stats` (digamma/gammaln/betaln for the VB E-step + ELBO live in `ferray::stats`, the scipy.special analog). R-SUBSTRATE-1/2. |

## Architecture

`bayesian_gmm.rs` follows the crate's unfitted/fitted split (CLAUDE.md naming):
`BayesianGaussianMixture<F>` (seven fields + builder setters
`with_covariance_type`/`with_max_iter`/`with_tol`/`with_weight_prior_type`/
`with_weight_concentration_prior`/`with_random_state`) → `Fit<Array2<F>, ()>` →
`FittedBayesianGaussianMixture<F>` (private `weights_`, `means_`, `covariances_`,
`converged_`, `lower_bound_`, `n_features_`, `covariance_type_`, `weight_prior_type_`).
Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2). Two public enums `WeightPriorType`
(`DirichletProcess`/`DirichletDistribution`) and `BayesianCovType`
(`Full`/`Tied`/`Diag`/`Spherical`) mirror the sklearn string options
(`_bayesian_mixture.py:351-354`,`:102`). `FittedBayesianGaussianMixture` implements
`Predict<Array2<F>>` (argmax of `compute_responsibilities`), mirroring sklearn
`BaseMixture.predict` (`_base.py:369-385`), and a `fit_predict` convenience mirrors
`fit_predict` (`_base.py:185-288`).

**The fit path (`fn fit` → `fn run_variational_em`) is plain ML-EM, NOT variational
Bayes** — this is the structural carve-out (Summary + Probes 2-4):

- **Init.** `fn init_means` samples `k` random data rows + tiny jitter (a
  `random_from_data`-like init); covariances start at identity-diagonal; weights
  uniform; `alpha` = `weight_concentration`. sklearn `_initialize_parameters`
  (`_base.py:99-140`) defaults to `init_params='kmeans'` (KMeans responsibilities)
  then runs the Bayesian `_initialize` (`_bayesian_mixture.py:513-528`) — REQ-8.
- **E-step.** `fn compute_responsibilities` computes `log π_k - 0.5·mahal` then row
  logsumexp-normalizes. sklearn's E-step is `_estimate_log_prob_resp`
  (`_base.py:507-531`) where `_estimate_log_weights` is the **digamma** Dirichlet/DP
  expectation (`:743-759`) and `_estimate_log_prob` is the **Wishart** precision
  expectation (`:761-777`) — REQ-3.
- **M-step.** `fn run_variational_em` updates `means` by ML (`Σ resp·x / nk`),
  `covariances` by ML weighted scatter (`Full`/`Tied` build the off-diagonal but it is
  later ignored in scoring — REQ-4), and `alpha`/`weights` by a simplified
  Dirichlet/DP rule. sklearn's M-step is the conjugate Bayesian update
  `_estimate_weights`/`_estimate_means`/`_estimate_precisions` producing
  `weight_concentration_`/`mean_precision_`/`degrees_of_freedom_`/`covariances_`/
  `precisions_cholesky_` (`:530-721`) — REQ-3/REQ-6.
- **ELBO.** Convergence uses `elbo = mean_i logsumexp_k(log resp)` — a proxy, NOT
  sklearn's `_compute_lower_bound` (`:779-836`) — REQ-5.
- **Scoring (`fn compute_responsibilities` / `fn unnormalized_log_prob`).** The
  `Full | Tied` arm uses the **diagonal of the covariance only** (REQ-4, R-DEV-1) —
  the load-bearing localized bug, wrong on correlated data (Probe 4).

`fn n_free_params` (used by `bic`/`aic`) mirrors the GMM parameter count
(`k·d` means + covariance shape + `k-1` weights); `bic`/`aic` themselves are
structurally correct accounting but their VALUES inherit the diverging diagonal-only
`score_samples` (REQ-4).

**Invariants held vs sklearn:** well-separated PARTITION co-membership (Probe 1/AC-1,
REQ-1); `predict_proba` rows sum to 1 + correct shapes (REQ-2); `weights_` sums to 1;
`means_.shape == (k, d)`; predict/predict_proba/score_samples shape-mismatch errors;
deterministic fit (seeded `StdRng`); matching `covariance_type`/`max_iter`/`tol`/
`weight_concentration_prior_type` defaults (REQ-9). Edge cases: `n_components==0` →
`FerroError::InvalidParameter`; `n_samples < n_components` →
`FerroError::InsufficientSamples` (sklearn raises `ValueError` here, `_base.py:213-218`
— error-TYPE ABI differs, folded into REQ-7's parameter-validation surface).

**Invariants NOT held vs sklearn:** the entire VB algorithm (REQ-3); the
full-covariance score (REQ-4); the true ELBO `lower_bound_` (REQ-5); every Bayesian
fitted attribute `weight_concentration_`/`mean_precision_`/`degrees_of_freedom_`/
`precisions_`/`precisions_cholesky_` + inverse-Wishart `means_`/`covariances_` and the
`(k,d,d)` covariance shape (REQ-6); the Bayesian priors + `n_init`/`warm_start`/
`reg_covar` params (REQ-7); `init_params`/`verbose` + the `n_components=1` default
(REQ-8); the PyO3 binding (REQ-10); `n_iter_` (REQ-11); numpy-RNG parity (REQ-12); the
ferray substrate (REQ-13).

**Consumer wiring.** The only non-test consumer is the crate re-export
(`pub use bayesian_gmm::{BayesianCovType, BayesianGaussianMixture,
FittedBayesianGaussianMixture, WeightPriorType}`, `ferrolearn-cluster/src/lib.rs`).
There is no `ferrolearn-python` binding (Probe 5) and no other in-crate consumer.

## Verification

Library crate (green at baseline `4ae6af03` for the existing heuristic behavior):
```bash
cargo test -p ferrolearn-cluster --lib bayesian_gmm   # 16 passed; 0 failed (run this iteration)
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The 16 in-tree `#[test]`s (`test_bayesian_gmm_basic_predict`,
`test_bayesian_gmm_two_blobs_separation`, `test_bayesian_gmm_dirichlet_distribution`,
`test_bayesian_gmm_spherical_cov`, `test_bayesian_gmm_diag_cov`,
`test_bayesian_gmm_tied_cov`, `test_bayesian_gmm_weights_sum_to_one`,
`test_bayesian_gmm_zero_components_error`, `test_bayesian_gmm_insufficient_samples`,
`test_bayesian_gmm_predict_shape_mismatch`, `test_bayesian_gmm_means_shape`,
`test_bayesian_gmm_weight_concentration_prior`, `test_bayesian_gmm_n_components_getter`,
`test_bayesian_gmm_converged_field`, `test_bayesian_gmm_f32`,
`test_bayesian_gmm_lower_bound_finite`) pin ferrolearn's current **heuristic** behavior
— label co-membership on the 2-blob fixture, shapes, the weights-sum-to-1 invariant,
error edges, f32 support, and `lower_bound().is_finite()`. **None compares `weights_` /
`means_` / `covariances_` / `lower_bound_` VALUES, the Bayesian attribute surface, or
the correlated-data prediction against the live sklearn `BayesianGaussianMixture`
oracle**, so they stay green despite the divergences. In particular
`test_bayesian_gmm_lower_bound_finite` only asserts finiteness (it cannot catch the
proxy-vs-true-ELBO divergence, REQ-5), and `test_bayesian_gmm_tied_cov` /
`test_bayesian_gmm_basic_predict` never exercise correlated covariance (REQ-4).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values), each into
`ferrolearn-cluster/tests/divergence_bayesian_gmm.rs` and FAILING against current
`bayesian_gmm.rs`:
```
# REQ-4 (diagonal-only score — the load-bearing localized bug)
python3 -c "import numpy as np; from sklearn.mixture import BayesianGaussianMixture; \
np.random.seed(1); \
A=np.random.multivariate_normal([0,0],np.array([[1.,0.9],[0.9,1.]]),40); \
B=np.random.multivariate_normal([0,0],np.array([[1.,-0.9],[-0.9,1.]]),40); \
X=np.vstack([A,B]); m=BayesianGaussianMixture(n_components=2,covariance_type='full',random_state=0,max_iter=300).fit(X); \
print(m.predict(np.array([[2.,2.],[2.,-2.]])).tolist())"   # [0, 1]
# REQ-3/REQ-6 (Bayesian attributes + ML-vs-Wishart values)
python3 -c "import numpy as np; from sklearn.mixture import BayesianGaussianMixture; \
X=np.array([[1.,2.],[1.,4.],[1.,0.],[4.,2.],[12.,4.],[10.,7.]]); \
m=BayesianGaussianMixture(n_components=2,random_state=42).fit(X); \
print(np.round(m.means_,4).tolist(), np.round(m.weights_,5).tolist(), m.degrees_of_freedom_.tolist())"
# [[2.4976,2.2923],[8.4553,4.5226]] [0.68141,0.31859] [5.8636..., 4.1364...]
# REQ-5 (true ELBO)
# lower_bound_ = -29.559463  (vs ferrolearn proxy mean-logsumexp)
```

ferrolearn-python (REQ-10 binding parity, after the binding lands):
```bash
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_bayesian_gmm.py -q
```
asserting `ferrolearn.BayesianGaussianMixture` exists and exposes `weights_`/`means_`/
`covariances_`/`weight_concentration_`/`mean_precision_`/`degrees_of_freedom_`/
`precisions_cholesky_`/`n_iter_`/`lower_bound_`, matching
`sklearn.mixture.BayesianGaussianMixture` on the AC fixtures.

## Blockers (filed iter 129: #1057–#1066)

REQ-1 (partition), REQ-2 (API contracts), REQ-9 (defaults) SHIP — no blocker. The
divergence IS the whole VB algorithm — there is **no clean minimal-fixable single
divergence** (no wrong default to flip; correctness needs the digamma/Wishart machinery
wholesale). The NOT-STARTED REQs:

1. "Blocker for REQ-3 of bayesian_gmm: replace plain ML-EM (`fn run_variational_em`)
   with the variational E/M-step — digamma `_estimate_log_weights` (`_bayesian_mixture.py:743-759`)
   + Wishart `_estimate_log_prob` (`:761-777`) + conjugate `_estimate_weights`/
   `_estimate_means`/`_estimate_precisions` (`:530-721`). Builder-scale." -p high -l blocker
   **(the core: the whole algorithm)**
2. "Blocker for REQ-4 of bayesian_gmm: `Full`/`Tied` scoring must use the FULL
   covariance via a Cholesky solve (true `dᵀΣ⁻¹d` + `log|Σ|`, sklearn `_estimate_log_prob`
   `:761-777`), not the diagonal only (`fn compute_responsibilities`/`fn unnormalized_log_prob`
   `Full|Tied` arm). Wrong on correlated data (R-DEV-1)." -p high -l blocker
   **(the load-bearing localized bug)**
3. "Blocker for REQ-5 of bayesian_gmm: `lower_bound_` must be the true ELBO
   `_compute_lower_bound` with Dirichlet/Wishart KL terms (`:779-836`), not the
   mean-logsumexp proxy. Depends on #1." -p high -l blocker
4. "Blocker for REQ-6 of bayesian_gmm: expose the Bayesian fitted attributes
   `weight_concentration_`/`mean_precision_`/`degrees_of_freedom_`/`precisions_`/
   `precisions_cholesky_` + inverse-Wishart `means_`/`covariances_` `(k,d,d)` shape
   (`_get/_set_parameters` `:838-889`). Depends on #1." -p high -l blocker
5. "Blocker for REQ-7 of bayesian_gmm: add the Bayesian prior params
   `mean_precision_prior`/`mean_prior`/`degrees_of_freedom_prior`/`covariance_prior`
   (`:155-178`), `reg_covar` (currently hardcoded 1e-6), `n_init` (best-of-N by
   lower_bound, `_base.py:231-264`), `warm_start`. Depends on #1." -p high -l blocker
6. "Blocker for REQ-8 of bayesian_gmm: add `init_params ∈ {kmeans,k-means++,random,
   random_from_data}` default 'kmeans' (`_base.py:99-140`) + `verbose`/`verbose_interval`
   + `n_components=1` default; current init is random_from_data-like only." -p medium -l blocker
7. "Blocker for REQ-10 of bayesian_gmm: bind BayesianGaussianMixture into
   ferrolearn-python as `_RsBayesianGaussianMixture` (mirror `_RsGaussianMixture` for
   the plain variant)." -p medium -l blocker
8. "Blocker for REQ-11 of bayesian_gmm: expose `n_iter_` (best-init EM step count,
   `_base.py:263,280`); `fn run_variational_em` discards the loop counter." -p medium -l blocker
9. "Blocker for REQ-12 of bayesian_gmm: exact init/RNG VALUE parity needs numpy's
   RandomState stream (`_base.py:228`); ferrolearn uses StdRng. Couples to #10
   (ferray::random)." -p low -l blocker
10. "Blocker for REQ-13 of bayesian_gmm: migrate off ndarray/rand/num-traits to
    ferray-core/ferray::linalg/ferray::random/ferray::stats (digamma/gammaln/betaln for
    the VB E-step + ELBO live in ferray::stats). R-SUBSTRATE-2." -p medium -l blocker
```
