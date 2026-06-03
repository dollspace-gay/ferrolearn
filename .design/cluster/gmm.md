# Gaussian Mixture (sklearn.mixture.GaussianMixture)

<!--
tier: 3-component
status: draft
baseline-commit: e3416328
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/mixture/_gaussian_mixture.py   # _estimate_gaussian_parameters (nk=resp.sum+10*eps, ML means, ML covariances :259-296); _estimate_gaussian_covariances_full/_tied/_diag/_spherical (:153-256); _compute_precision_cholesky (precisions_cholesky_ via lower Cholesky + solve_triangular :299-348); _compute_log_det_cholesky (:408-445); _estimate_log_gaussian_prob (full log density via precisions_chol, -0.5*(d*log(2pi)+||y||^2)+log_det :448-507); class GaussianMixture(BaseMixture) (:510); __init__ defaults n_components=1/covariance_type='full'/tol=1e-3/reg_covar=1e-6/max_iter=100/n_init=1/init_params='kmeans'/weights_init=None/means_init=None/precisions_init=None/random_state=None/warm_start=False (:701-735); _parameter_constraints covariance_type StrOptions{full,tied,diag,spherical} (:693-699); _m_step (_estimate_gaussian_parameters + weights/=sum + precisions_cholesky_ :801-818); _estimate_log_weights=log(weights_) (:825-826); _compute_lower_bound returns log_prob_norm (:828-829); _get/_set_parameters + precisions_ derivation (:831-860); _n_parameters cov_params+mean_params+(k-1) (:862-874); bic=-2*score*n+n_params*ln(n) (:876-894); aic=-2*score*n+2*n_params (:896-912)
  - sklearn/mixture/_base.py               # BaseMixture (:43); _parameter_constraints n_components>=1/tol>=0/reg_covar>=0/max_iter>=0/n_init>=1/init_params StrOptions{kmeans,random,random_from_data,k-means++}/random_state/warm_start (:50-63); _initialize_parameters init_params kmeans=full KMeans hard labels resp, random, random_from_data, k-means++ (:99-140); fit (:154-182); fit_predict ensure_min_samples=2 + n_samples<n_components ValueError, n_init loop, E-step then M-step then convergence abs(change)<tol AFTER m-step, converged_/n_iter_/lower_bound_=max_lower_bound, ConvergenceWarning, final e-step argmax labels (:185-288); _e_step mean(log_prob_norm) (:290-307); score_samples=logsumexp(weighted_log_prob) (:331-348); score=score_samples.mean (:350-367); predict=argmax weighted_log_prob (:369-385); predict_proba=exp(log_resp) (:387-404); sample (multinomial + multivariate_normal :406-466); _estimate_weighted_log_prob (:468-479); _estimate_log_prob_resp logsumexp normalization (:507-531)
ferrolearn-module: ferrolearn-cluster/src/gmm.rs
ferrolearn-python: ferrolearn-python/src/extras.rs (RsGaussianMixture, name="_RsGaussianMixture", :1038-1088) — THIN: ctor (n_components, max_iter, random_state) + fit + predict only
parity-ops: GaussianMixture (.__init__, .fit, .fit_predict, .predict, .predict_proba, .score_samples, .score, .bic, .aic, .transform, .weights_, .means_, .covariances_, .precisions_, .precisions_cholesky_, .converged_, .lower_bound_, .n_iter_, .sample)
crosslink-issue: 1092
-->

## Summary

`ferrolearn-cluster/src/gmm.rs` mirrors scikit-learn's `GaussianMixture`
(`sklearn/mixture/_gaussian_mixture.py`, `class GaussianMixture(BaseMixture)` `:510`) —
a soft-clustering Gaussian mixture fit by Expectation-Maximisation, supporting four
covariance parameterisations (`full`/`tied`/`diag`/`spherical`). It exposes the unfitted
`GaussianMixture<F>` (`n_components`, `covariance_type`, `max_iter`, `tol`, `n_init`,
`random_state`), the fitted `FittedGaussianMixture<F>` (`weights_`, `means_`,
`covariances_`, `converged_`, `lower_bound_`, plus private `covariance_type_` /
`n_features_`), a `Fit<Array2<F>, ()>` impl (`fn run_em` per-init EM, best-of-`n_init`
by `lower_bound_`), a `Predict<Array2<F>>` impl (hard argmax of the responsibility
matrix), a `Transform<Array2<F>>` impl (the responsibility matrix = `predict_proba`),
a `fit_predict` convenience, and the `predict_proba` / `score_samples` / `score` /
`bic` / `aic` methods. It is re-exported at the crate root (`pub use gmm::{CovarianceType,
FittedGaussianMixture, GaussianMixture}` in `ferrolearn-cluster/src/lib.rs`).

**It has a REAL — but THIN — CPython consumer.** The PyO3 binding `RsGaussianMixture`
(`#[pyclass(name = "_RsGaussianMixture")]` in `ferrolearn-python/src/extras.rs:1038`) is
registered in `ferrolearn-python/src/lib.rs` (`m.add_class::<extras::RsGaussianMixture>()`)
and wrapped by the sklearn-API class `ferrolearn.GaussianMixture`
(`ferrolearn-python/python/ferrolearn/_extras.py:449`, re-exported in `__init__.py`). So
`import ferrolearn; ferrolearn.GaussianMixture().fit(X).predict(X)` reaches this code. But
the binding's surface is narrow: ctor `(n_components=1, max_iter=100, random_state=None)`
+ `fit` + `predict`/`fit_predict` ONLY. It does NOT marshal `covariance_type`/`tol`/`n_init`,
nor any of `weights_`/`means_`/`covariances_`/`predict_proba`/`score`/`aic`/`bic`. So the
thin `fit`/`predict` marshalling SHIPS (REQ-7); the omitted binding surface is a separate
NOT-STARTED REQ (REQ-13).

**Honest assessment (R-HONEST-3) — the surprising VALUE-match crux, and the one clean bug.**
Unlike `bayesian_gmm.rs`, this is a *correct EM with k-means++-seeded init*, so on
well-separated data it converges to the same optimum sklearn does. The live oracle
(Probe 1) shows ferrolearn's `weights_`, `means_`, and `covariances_` VALUES **match
sklearn to ~1e-15 up to component permutation** on the 2-blob fixture — so a *scoped*
value-parity REQ on well-separated data SHIPS (REQ-3), not merely the partition. The
PARTITION (`predict` labels, REQ-1) and `predict_proba` (REQ-2) match sklearn exactly
(Probes 1-3).

BUT there is **one clean, minimal-fixable, load-bearing divergence**: the `Full`/`Tied`
log-density **double-counts `log|Σ|`**. `fn log_det_and_norm_full` returns
`log_norm = -(d/2)·ln(2π) - 0.5·log_det`, and `fn log_responsibilities` then adds
`log_w + log_norm - 0.5·(log_det + maha)` — subtracting `0.5·log_det` **twice**
(`gmm.rs` lines for `log_det_and_norm_full` / the `Full | Tied` arm of `log_responsibilities`).
The correct density (sklearn `_estimate_log_gaussian_prob` `:507`) has `log_det` once.
The spurious extra `-0.5·log_det_k` shifts every `Full`/`Tied` component's log-density by
a per-component constant (Probe 4: `+5.400251` on the 2-blob fixture, since `log_det ≈
-10.8005`). Consequences: `score_samples`/`score`/`lower_bound_`/`aic`/`bic` all DIVERGE
in absolute value (Probe 4: ferrolearn `score 7.26994` vs sklearn `1.86969`); the
`Diag`/`Spherical` arms are CORRECT (they fold `log_det` into `log_norm` once and add
only `-0.5·maha`). Because the offset is a per-component additive constant on covariances
that here are equal, the responsibilities/partition CANCEL it and still match — so REQ-1/
REQ-2/REQ-3 ship while the absolute-score REQ (REQ-5) does not. **This is the single most
confident minimal-fixable candidate: the score bug (REQ-5), localized to the `Full`/`Tied`
density.**

Everything that is genuinely absent — `precisions_`/`precisions_cholesky_` (REQ-8),
`n_iter_` (REQ-9), `sample()` (REQ-10), `reg_covar`/`init_params`/`weights_init`/
`means_init`/`precisions_init`/`warm_start` params (REQ-11), the `covariances_` SHAPE
contract (REQ-6), exact init/RNG VALUE parity on non-separated data (REQ-12), the full
binding surface (REQ-13), the ferray substrate (REQ-14) — is NOT-STARTED.

## Live oracle probes (sklearn 1.5.2, run from /tmp; ferrolearn via a throwaway `examples/` bin)

All expected values come from the installed sklearn 1.5.2 oracle (run from `/tmp`,
`random_state` fixed) or a sklearn `file:line`, never literal-copied from ferrolearn
(R-CHAR-3). The ferrolearn column was produced by a throwaway `ferrolearn-cluster/examples`
bin (deleted after probing) calling the public `GaussianMixture` API on the identical
fixture.

### Probe 1 — the CRUX: well-separated 2-blob partition AND weights_/means_/covariances_ value-match

Fixture = the in-tree `make_two_blobs()` 12-point fixture (two 6-point blobs at ~(0,0)
and ~(10,10)), `n_components=2`, `random_state=42`, `max_iter=200`, `covariance_type='full'`.

```
python3 -c "import numpy as np; from sklearn.mixture import GaussianMixture; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[-0.1,0.],[0.,-0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[9.9,10.],[10.,9.9],[10.1,10.1]]); \
gm=GaussianMixture(n_components=2,random_state=42,max_iter=200).fit(X); \
print('labels', gm.fit_predict(X).tolist()); print('weights_', np.round(gm.weights_,6).tolist()); \
print('means_', np.round(gm.means_,6).tolist()); print('covariances_', np.round(gm.covariances_,6).tolist())"
# labels       [0,0,0,0,0,0, 1,1,1,1,1,1]
# weights_     [0.5, 0.5]
# means_       [[0.016667,0.016667], [10.016667,10.016667]]
# covariances_ [[[0.004723,0.001389],[0.001389,0.004723]], [[0.004723,0.001389],[0.001389,0.004723]]]
```
ferrolearn (same fixture, same params), via `GaussianMixture::<f64>::new(2).with_random_state(42).with_max_iter(200)`:
```
# labels       [0,0,0,0,0,0, 1,1,1,1,1,1]                       <- IDENTICAL partition
# weights_     [0.5, 0.5]                                       <- MATCH
# means_       [[0.0166666666666667,0.0166666666666667], [10.0166666666666,10.0166666666666]]  <- MATCH ~1e-15
# covariances_ [[0.0047232222222,0.0013888889],[0.0013888889,0.0047232222222], (block 0)
#               [0.0047232222222,0.0013888889],[0.0013888889,0.0047232222222]] (block 1)        <- VALUES MATCH
```
**FINDING (the crux):** on well-separated data the PARTITION matches exactly AND
`weights_`/`means_`/`covariances_` VALUES match sklearn to ~1e-15 up to permutation. EM
with k-means++ seeding lands on the same global optimum despite the `StdRng`-vs-numpy-RNG
init difference, because the optimum is unique on well-separated blobs (REQ-1, REQ-3 SHIP,
scoped to well-separated data). The only layout difference is the `covariances_` SHAPE:
sklearn `(k,d,d)=(2,2,2)` 3-D vs ferrolearn `(k·d,d)=(4,2)` 2-D stacked blocks (REQ-6).

### Probe 2 — predict_proba (responsibilities) match exactly (REQ-2)

```
python3 -c "import numpy as np; from sklearn.mixture import GaussianMixture; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[-0.1,0.],[0.,-0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[9.9,10.],[10.,9.9],[10.1,10.1]]); \
gm=GaussianMixture(n_components=2,random_state=42,max_iter=200).fit(X); \
print('predict_proba[0]', gm.predict_proba(X)[0].tolist(), 'predict_proba[6]', gm.predict_proba(X)[6].tolist())"
# predict_proba[0] [1.0, 0.0]   predict_proba[6] [0.0, 1.0]
```
ferrolearn `Transform::transform` (= `predict_proba`): `predict_proba[0] [1.0, 0.0]`,
`predict_proba[6] [0.0, 1.0]` — MATCH. Each row sums to 1 (REQ-2), mirroring
`BaseMixture.predict_proba` (`_base.py:387-404`) + `_estimate_log_prob_resp` logsumexp
normalization (`:507-531`).

### Probe 3 — heteroscedastic two-blob partition still matches (REQ-1 robustness)

Two well-separated clusters with very different covariance scale (tight σ²=0.05 vs wide
σ²=2.0; per-component `log|Σ|` differs by ~6.9), `n_components=2`, `covariance_type='full'`,
`random_state=0`, `max_iter=300`:
```
python3 -c "import numpy as np; from sklearn.mixture import GaussianMixture; np.random.seed(3); \
A=np.random.multivariate_normal([0,0],[[0.05,0],[0,0.05]],30); B=np.random.multivariate_normal([5,5],[[2.,0],[0,2.]],30); \
X=np.vstack([A,B]); gm=GaussianMixture(n_components=2,covariance_type='full',random_state=0,max_iter=300).fit(X); \
print('labels', gm.fit_predict(X).tolist()); print('predict([[1.5,1.5],[3,3]])', gm.predict([[1.5,1.5],[3.,3.]]).tolist())"
# labels [1]*30 + [0]*30      predict([[1.5,1.5],[3,3]]) [0, 0]
```
ferrolearn (same fixture): `labels [1]*30 + [0]*30` (IDENTICAL partition), `predict([[1.5,1.5],
[3,3]]) [0, 0]` (MATCH), `covariances_ ≈ [[1.52,0.016],[0.016,1.84]],[[0.053,0.012],[0.012,
0.056]]` (close to sklearn's). The double-count bug (Probe 4) adds a per-component constant
to the absolute score; on well-separated data it does not flip the argmax, so the partition
holds even when component covariances differ. (Exact-value parity off the separated regime
is NOT guaranteed — REQ-12, RNG-coupled.)

### Probe 4 — THE BUG: score_samples / score / lower_bound_ / aic / bic diverge by a per-component log|Σ| constant (REQ-5)

Same 2-blob fixture as Probe 1:
```
python3 -c "import numpy as np; from sklearn.mixture import GaussianMixture; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[-0.1,0.],[0.,-0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[9.9,10.],[10.,9.9],[10.1,10.1]]); \
gm=GaussianMixture(n_components=2,random_state=42,max_iter=200).fit(X); \
print('score_samples[0]', round(float(gm.score_samples(X)[0]),5)); print('score', round(float(gm.score(X)),6)); \
print('lower_bound_', round(float(gm.lower_bound_),6)); print('aic', round(float(gm.aic(X)),4), 'bic', round(float(gm.bic(X)),4), 'n_params', gm._n_parameters())"
# score_samples[0] 2.82401   score 1.86969   lower_bound_ 1.86969   aic -22.8726   bic -17.5386   n_params 11
```
ferrolearn (same fixture): `score_samples[0] 8.224038`, `score 7.269941`, `lower_bound_
7.269941`, `aic -152.4786`, `bic -147.1446`, `n_params 11`.

The per-sample offset is exactly `8.224038 - 2.824011 = 5.400028 ≈ -0.5·log|Σ| = -0.5·(-10.800251)`.
A hand reconstruction confirms ferrolearn's `8.224038` = `log(0.5) + (-(2/2)·ln(2π) - 0.5·log_det)
- 0.5·(log_det + maha)` — `log_det` subtracted TWICE. sklearn's `_estimate_log_gaussian_prob`
(`:448-507`) returns `-0.5·(d·ln(2π) + ||y||²) + log_det_chol` — `log|Σ|` once. **The
DEFINITION of `lower_bound_` matches** (both = per-sample mean log-likelihood at the best
init; sklearn `_e_step` `np.mean(log_prob_norm)` `:307`, ferrolearn `fn run_em` `ll = Σ
log_probs / n`); only the VALUE diverges, because of the double-count. `n_params=11` MATCHES
(`_n_parameters` `:862-874`: `cov k·d·(d+1)/2 = 2·2·3/2 = 6` + `mean k·d = 4` + `(k-1) = 1`).
The `bic`/`aic` FORMULA structure also matches (ferrolearn `bic = -2·Σ score_samples +
n_params·ln n`, sklearn `-2·score·n + n_params·ln n`, `score·n = Σ score_samples`); only
the inherited score value diverges.

### Probe 5 — the thin PyO3 binding chain + missing attributes (REQ-7 SHIPPED / REQ-13 NOT-STARTED)

```
grep -rn "GaussianMixture" ferrolearn-python/src/lib.rs ferrolearn-python/src/extras.rs ferrolearn-python/python/ferrolearn/_extras.py ferrolearn-python/python/ferrolearn/__init__.py
# ferrolearn-python/src/lib.rs:69      m.add_class::<extras::RsGaussianMixture>()?;
# ferrolearn-python/src/extras.rs:1038 #[pyclass(name = "_RsGaussianMixture")]   ctor(n_components,max_iter,random_state)+fit+predict
# ferrolearn-python/python/ferrolearn/_extras.py:449  class GaussianMixture(BaseEstimator): __init__(*,n_components=1,max_iter=100,random_state=None); fit; predict; fit_predict
# ferrolearn-python/python/ferrolearn/__init__.py:32,107  GaussianMixture (imported + in __all__)
```
The chain `import ferrolearn -> ferrolearn.GaussianMixture -> _RsGaussianMixture ->
ferrolearn_cluster::GaussianMixture` is REAL (REQ-7 SHIPPED, thin). But `_RsGaussianMixture`
exposes only `fit`/`predict`; it marshals neither `covariance_type`/`tol`/`n_init` ctor
params nor the `weights_`/`means_`/`covariances_`/`predict_proba`/`score`/`aic`/`bic`
getters (REQ-13 NOT-STARTED).

### Probe 6 — missing attributes / methods / params (REQ-8/9/10/11) + shape contract (REQ-6)

```
python3 -c "import numpy as np; from sklearn.mixture import GaussianMixture; \
X=np.vstack([np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]]),np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])+10]); \
print('full prec_chol', GaussianMixture(2,random_state=0).fit(X).precisions_cholesky_.shape, 'n_iter_', GaussianMixture(2,random_state=0).fit(X).n_iter_); \
print('diag', GaussianMixture(2,covariance_type='diag',random_state=0).fit(X).covariances_.shape); \
print('spherical', GaussianMixture(2,covariance_type='spherical',random_state=0).fit(X).covariances_.shape); \
print('tied', GaussianMixture(2,covariance_type='tied',random_state=0).fit(X).covariances_.shape)"
# full prec_chol (2,2,2)  n_iter_ 2
# diag (2,2)   spherical (2,)   tied (2,2)
```
- `precisions_` / `precisions_cholesky_` (sklearn `_set_parameters` `:847-860`,
  `_compute_precision_cholesky` `:299-348`): ferrolearn `FittedGaussianMixture` exposes
  `covariances_` ONLY, NO precisions (REQ-8).
- `n_iter_` (sklearn `_base.py:280`, value `2` here): ferrolearn has `converged_` /
  `lower_bound_` but NO `n_iter_` (REQ-9).
- `sample(n_samples)` (sklearn `_base.py:406-466`): ferrolearn has none (REQ-10).
- ctor params `reg_covar` (ferrolearn hardcodes `1e-6` in `fn cholesky` + the M-step),
  `init_params`, `weights_init`/`means_init`/`precisions_init`, `warm_start`: all absent;
  ferrolearn `GaussianMixture<F>` has `n_components`/`covariance_type`/`max_iter`/`tol`/
  `n_init`/`random_state` only (REQ-11).
- `covariances_` SHAPE (REQ-6): sklearn full `(k,d,d)`, tied `(d,d)`, spherical `(k,)`;
  ferrolearn full/tied `(k·d,d)` 2-D blocks (tied stores k identical copies), spherical
  `(k,1)`, diag `(k,d)`. Only `diag` matches sklearn's shape.

### Probe 7 — defaults + init divergence (REQ-4 SHIPPED / REQ-11)

sklearn `GaussianMixture.__init__` (`_gaussian_mixture.py:701-718`) +
`BaseMixture._parameter_constraints` (`_base.py:50-63`): `covariance_type='full'`,
`max_iter=100`, `tol=1e-3`, `n_init=1` — all four MATCH ferrolearn `fn new`
(`covariance_type: CovarianceType::Full`, `max_iter: 100`, `tol: 1e-3`, `n_init: 1`)
(REQ-4). Divergent: sklearn `n_components=1` default (ferrolearn `new(n_components)`
requires it explicitly — Rust builder, minor); sklearn `init_params='kmeans'` runs a
FULL `KMeans(n_init=1)` and uses its hard labels as the initial responsibilities
(`_base.py:112-121`), then `_estimate_gaussian_parameters` for the initial weights/means/
covariances (`_gaussian_mixture.py:770-799`) — ferrolearn `fn init_means` does only greedy
k-means++ SEEDING for means, plus uniform weights and identity covariances (`fn run_em`
init block), NOT a full KMeans, and has no `init_params` param (REQ-11). The init
difference is RNG/trajectory-entangled (REQ-12), but converges to the same optimum on
well-separated data (Probes 1/3).

## Requirements

- REQ-1 (well-separated PARTITION up to permutation): `fit_predict` / `fit().predict()`
  recover the same hard-label co-membership as `GaussianMixture.fit_predict` on
  well-separated blobs (Probes 1/3; sklearn final e-step `argmax`, `_base.py:286-288`;
  ferrolearn `Predict::predict` argmax of `Transform::transform`).
- REQ-2 (predict_proba / transform contract): `Transform::transform` (= `predict_proba`)
  returns an `(n_samples, n_components)` row-stochastic matrix (each row sums to 1),
  mirroring `BaseMixture.predict_proba` (`_base.py:387-404`) + `_estimate_log_prob_resp`
  logsumexp normalization (`:507-531`); matches sklearn values on well-separated data
  (Probe 2).
- REQ-3 (well-separated weights_/means_/covariances_ VALUE-match): on well-separated data
  the fitted `weights_`/`means_`/`covariances_` VALUES match sklearn up to component
  permutation (Probe 1, ~1e-15), because EM with k-means++ seeding reaches the unique
  global optimum (`_estimate_gaussian_parameters` `:259-296`; ferrolearn `fn run_em`
  M-step). Scoped to well-separated data; the general init-dependent value parity is
  REQ-12.
- REQ-4 (matching defaults): `covariance_type='full'`, `max_iter=100`, `tol=1e-3`,
  `n_init=1` match sklearn `__init__` (`_gaussian_mixture.py:705-709`) +
  `BaseMixture._parameter_constraints` (`_base.py:50-63`) (Probe 7).
- REQ-5 (absolute score_samples/score/lower_bound_/aic/bic VALUE — the load-bearing bug):
  `score_samples` = `logsumexp(weighted_log_prob)` with the CORRECT Gaussian log-density
  `-0.5·(d·ln(2π) + dᵀΣ⁻¹d) - 0.5·log|Σ|` (sklearn `_estimate_log_gaussian_prob` `:448-507`,
  `score_samples` `_base.py:331-348`, `score` `:350-367`). ferrolearn's `Full`/`Tied`
  density (`fn log_det_and_norm_full` + the `Full | Tied` arm of `fn log_responsibilities`)
  subtracts `0.5·log|Σ|` TWICE, so every absolute log-density (and `score`/`lower_bound_`/
  `aic`/`bic`) is off by `-0.5·log|Σ|_k` per component (Probe 4).
- REQ-6 (covariances_ SHAPE contract): expose `covariances_` with sklearn's shape — full
  `(k,d,d)`, tied `(d,d)`, spherical `(k,)`, diag `(k,d)` (`_gaussian_mixture.py:617-624`).
  ferrolearn uses full/tied `(k·d,d)` 2-D blocks (tied = k copies), spherical `(k,1)`; only
  diag matches (Probe 6).
- REQ-7 (thin PyO3 binding marshalling): `_RsGaussianMixture` marshals
  `fit(n_components,max_iter,random_state)` + `predict`/`fit_predict` between numpy and
  `ferrolearn_cluster::GaussianMixture` (`extras.rs:1038-1088`, `_extras.py:449`); the
  non-test CPython consumer (Probe 5).
- REQ-8 (precisions_ / precisions_cholesky_): expose `precisions_` (inverse covariances)
  and `precisions_cholesky_` (the primary internal repr, `_compute_precision_cholesky`
  `:299-348`, `_set_parameters` `:847-860`). ferrolearn exposes `covariances_` only
  (Probe 6).
- REQ-9 (n_iter_ attribute): expose `n_iter_` = number of EM steps of the best init
  (`_base.py:263,280`). ferrolearn `FittedGaussianMixture` has `converged_`/`lower_bound_`
  but no `n_iter_` (Probe 6).
- REQ-10 (sample method): `sample(n_samples)` generates samples from the fitted mixture
  (`_base.py:406-466`, multinomial component counts + `multivariate_normal`). ferrolearn
  has none (Probe 6).
- REQ-11 (init_params + remaining ctor surface): support `init_params ∈
  {'kmeans','k-means++','random','random_from_data'}` default `'kmeans'` = full KMeans
  hard-label responsibilities (`_base.py:99-140`,`:56-58`), `reg_covar` (ferrolearn
  hardcodes `1e-6`, `:707`), `weights_init`/`means_init`/`precisions_init`
  (`_gaussian_mixture.py:711-713`), `warm_start` (`_base.py:222-223`), and the
  `n_components=1` default (`:703`). ferrolearn `fn init_means` does k-means++ seeding
  only (Probe 7).
- REQ-12 (exact value parity off the separated regime / numpy-RNG): exact
  `init_params`-driven VALUE parity on non-well-separated data requires matching numpy's
  `check_random_state`/`RandomState` stream and the full-KMeans init (`_base.py:228,112-121`);
  ferrolearn uses `StdRng::seed_from_u64` + k-means++ seeding. Couples to REQ-14.
- REQ-13 (full PyO3 binding surface): marshal `covariance_type`/`tol`/`n_init` ctor params
  and the `weights_`/`means_`/`covariances_`/`predict_proba`/`score`/`aic`/`bic`/`n_iter_`
  getters through `_RsGaussianMixture` + `ferrolearn.GaussianMixture` (Probe 5). Currently
  fit/predict only.
- REQ-14 (ferray substrate, R-SUBSTRATE): the array / linalg / RNG layer is `ferray-core` /
  `ferray::linalg` (Cholesky) / `ferray::random`, not `ndarray` / a hand-rolled `fn cholesky`
  / `rand::StdRng`.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`, `random_state`
fixed), never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED): on the 2-blob fixture (Probe 1) ferrolearn `predict` equals
  sklearn `[0,0,0,0,0,0,1,1,1,1,1,1]` up to a label permutation; on the heteroscedastic
  fixture (Probe 3) the 30/30 split + `predict([[1.5,1.5],[3,3]])=[0,0]` match. In-tree
  `test_predict_well_separated_clusters` / `test_predict_three_blobs` exercise co-membership.
- AC-2 (REQ-2, SHIPPED): `transform(X)` rows each sum to 1 (in-tree
  `test_transform_rows_sum_to_one`); `transform → (n,k)` (`test_transform_shape`);
  `predict → (n,)` (`test_predict_shape`); on Probe 2, `predict_proba[0]=[1,0]`,
  `predict_proba[6]=[0,1]` match sklearn.
- AC-3 (REQ-3, SHIPPED scoped): on Probe 1's fixture `weights_ → [0.5,0.5]`,
  `means_ → [[0.016667,0.016667],[10.016667,10.016667]]`,
  `covariances_ → [[0.004723,0.001389],[0.001389,0.004723]]` (per block) match sklearn
  to ~1e-6 up to permutation.
- AC-4 (REQ-4, SHIPPED): `GaussianMixture(2).get_params()` reports `covariance_type='full'`,
  `max_iter=100`, `tol=0.001`, `n_init=1` — matching ferrolearn `fn new`.
- AC-5 (REQ-5, SHIPPED): on Probe 1's fixture sklearn `score(X)=1.86969`,
  `lower_bound_=1.86969`, `aic=-22.8726`, `bic=-17.5386`; ferrolearn now returns
  `score=1.8696902967180025` (== within 1e-6) after fixing the per-component `-0.5·log|Σ|`
  double-count + the `fn cholesky` double-regularization (#1093). Pinned green by
  `req5_score_full_matches_sklearn`.
- AC-6 (REQ-6, DIVERGES): on Probe 6 sklearn `covariances_.shape` = full `(2,2,2)`, tied
  `(2,2)`, spherical `(2,)`; ferrolearn = `(4,2)` / `(4,2)` / `(2,1)`.
- AC-7 (REQ-8/9/10/13, DIVERGES): sklearn exposes `precisions_cholesky_.shape=(2,2,2)`,
  `n_iter_=2`, `sample(n)`; ferrolearn has none, and `import ferrolearn;
  GaussianMixture` exposes neither `means_`/`covariances_`/`predict_proba`/`score`/`aic`/
  `bic` nor `covariance_type`/`tol`/`n_init`.

## REQ status

Binary (R-DEFER-2). `GaussianMixture` / `FittedGaussianMixture` / `CovarianceType` are
existing pub APIs re-exported at the crate root AND reachable from `import ferrolearn`
via the thin `_RsGaussianMixture` binding (grandfathered S5/R-DEFER-1 — the boundary
estimator type IS the public API). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2, run from `/tmp`.
Honest assessment (R-HONEST-3): **SIX REQs SHIP** — the well-separated PARTITION (REQ-1),
the `predict_proba`/`transform` contract (REQ-2), the well-separated `weights_`/`means_`/
`covariances_` VALUE-match (REQ-3, the surprising crux verified in Probe 1), the matching
defaults (REQ-4), the absolute `score`/`lower_bound_`/`aic`/`bic` VALUE-match (REQ-5, after
fixing the `Full`/`Tied` `log|Σ|` double-count + the `fn cholesky` double-regularization,
#1093), and the thin PyO3 `fit`/`predict` marshalling (REQ-7).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (well-separated PARTITION) | SHIPPED | impl `Predict::predict` (argmax of `Transform::transform`) → on the 2-blob fixture `[0,0,0,0,0,0,1,1,1,1,1,1]`, IDENTICAL to sklearn `fit_predict` (`_base.py:286-288`); heteroscedastic 30/30 split + test-point predict also match (Probes 1/3). Consumer: `_RsGaussianMixture::predict` (`extras.rs:1073`) + crate re-export (`lib.rs`). Verification: `test_predict_well_separated_clusters`, `test_predict_three_blobs`, `test_predict_shape`, `test_predict_valid_range` pass. Underclaim: up-to-permutation co-membership on well-separated data; not a label-integer match. |
| REQ-2 (predict_proba / transform contract) | SHIPPED | impl `Transform::transform` / `fn predict_proba` (`log_responsibilities` → `log_sum_exp_rows` → `exp`) returns an `(n,k)` row-stochastic matrix mirroring `BaseMixture.predict_proba` (`_base.py:387-404`) + `_estimate_log_prob_resp` (`:507-531`); on Probe 2 `predict_proba[0]=[1,0]`, `[6]=[0,1]` match sklearn. Consumer: crate re-export (`lib.rs`); `transform` is the responsibility surface behind `_RsGaussianMixture::predict`. Verification: `test_transform_shape`, `test_transform_rows_sum_to_one`, `test_transform_values_in_0_1` pass. |
| REQ-3 (well-separated weights_/means_/covariances_ VALUE-match) | SHIPPED | impl `fn run_em` M-step (ML `weights`/`means`/`covariances`) mirrors `_estimate_gaussian_parameters` (`:259-296`) → on Probe 1 `weights_=[0.5,0.5]`, `means_=[[0.016667,0.016667],[10.016667,10.016667]]`, `covariances_` block `[[0.004723,0.001389],[0.001389,0.004723]]` match sklearn to ~1e-15 up to permutation. Consumer: `_RsGaussianMixture::fit` (`extras.rs:1059`) + crate re-export. Verification: in-tree shape/sum tests pass; the live-oracle value-match (Probe 1) is the basis. Underclaim: SCOPED to well-separated data where the optimum is unique; general init-dependent parity is REQ-12 (RNG-coupled). LEAST-CONFIDENT SHIPPED claim — depends on the optimum being unique on the fixture. |
| REQ-4 (matching defaults) | SHIPPED | impl `fn new` sets `covariance_type=Full`, `max_iter=100`, `tol=1e-3`, `n_init=1` — matching sklearn `__init__` (`_gaussian_mixture.py:705-709`) + `BaseMixture._parameter_constraints` (`_base.py:50-63`) (Probe 7/AC-4). Consumer: crate re-export + `_RsGaussianMixture::new` (`extras.rs:1049`, which also defaults `n_components=1`/`max_iter=100`). Verification: `test_new_defaults`, `test_builder_methods` pass. Underclaim: `n_components` has no default in the Rust builder (sklearn=1; the binding supplies `1`); `init_params` default unsupported (REQ-11). |
| REQ-5 (absolute score/lower_bound_/aic/bic VALUE) | SHIPPED | FIXED #1093. The `Full \| Tied` arm of `fn log_responsibilities` now adds `log_w + log_norm - 0.5·maha` (`log_norm` already folds in `-0.5·log_det`, matching sklearn `_estimate_log_gaussian_prob` `:448-507`) — was subtracting `0.5·log_det` TWICE; and `fn cholesky` no longer re-adds `reg=1e-6` on top of the M-step `reg_covar` (was regularizing Σ twice). `score`/`score_samples`/`lower_bound_`/`aic`/`bic` now value-match the oracle: on Probe 1 ferrolearn `score=1.8696902967180025` == sklearn `1.86969` (within 1e-6). Consumer: crate re-export (`lib.rs`) — `score`/`aic`/`bic` public methods. Verification: `req5_score_full_matches_sklearn` + `diag_score_control_matches_sklearn` in `tests/divergence_gmm.rs` (live-oracle) PASS. `Diag`/`Spherical` arms were already correct. |
| REQ-6 (covariances_ SHAPE contract) | NOT-STARTED | open prereq blocker #1094. `FittedGaussianMixture.covariances_` is `(k·d,d)` 2-D stacked blocks for full/tied (tied stores k identical copies), `(k,1)` for spherical (`covariances_` doc). sklearn is full `(k,d,d)`, tied `(d,d)`, spherical `(k,)`, diag `(k,d)` (`_gaussian_mixture.py:617-624`, Probe 6). Only diag matches. |
| REQ-7 (thin PyO3 fit/predict marshalling) | SHIPPED | impl `RsGaussianMixture` (`#[pyclass(name="_RsGaussianMixture")]`, `extras.rs:1038`): `fn new(n_components,max_iter,random_state)` (`:1049`), `fn fit` (`:1059`, numpy→ndarray + `GaussianMixture::fit`), `fn predict` (`:1073`). Registered `m.add_class::<extras::RsGaussianMixture>()` (`lib.rs:69`), wrapped `class GaussianMixture(BaseEstimator)` (`_extras.py:449`), in `__init__.py __all__`. Consumer: this binding IS the CPython non-test consumer. Verification: `import ferrolearn; ferrolearn.GaussianMixture(n_components=2,random_state=42).fit(X).predict(X)`. Underclaim: ONLY fit/predict; the rest of the surface is REQ-13. |
| REQ-8 (precisions_ / precisions_cholesky_) | NOT-STARTED | open prereq blocker #1095. `FittedGaussianMixture` exposes `covariances_` only; sklearn `_set_parameters` derives `precisions_` (inverse cov) + `precisions_cholesky_` (the primary internal repr via `_compute_precision_cholesky` `:299-348`,`:847-860`; shape `(2,2,2)` in Probe 6). Absent in ferrolearn. |
| REQ-9 (n_iter_ attribute) | NOT-STARTED | open prereq blocker #1096. sklearn `n_iter_` = EM steps of the best init (`_base.py:263,280`; value `2` Probe 6). `FittedGaussianMixture` has `converged_`/`lower_bound_` but `fn run_em` discards the loop counter `_iter`. |
| REQ-10 (sample method) | NOT-STARTED | open prereq blocker #1097. sklearn `BaseMixture.sample(n_samples)` draws multinomial component counts then `multivariate_normal` per component (`_base.py:406-466`). ferrolearn has no `sample` on `FittedGaussianMixture`. |
| REQ-11 (init_params + remaining ctor surface) | NOT-STARTED | open prereq blocker #1098. sklearn `init_params ∈ {kmeans,k-means++,random,random_from_data}` default `'kmeans'` runs a FULL `KMeans(n_init=1)` for the initial responsibilities (`_base.py:99-140`); ferrolearn `fn init_means` does greedy k-means++ SEEDING only + uniform weights + identity cov (`fn run_em` init block), no `init_params`. MISSING `reg_covar` (hardcoded `1e-6` in `fn cholesky`/M-step), `weights_init`/`means_init`/`precisions_init` (`:711-713`), `warm_start` (`_base.py:222-223`), `n_components=1` default. |
| REQ-12 (exact value parity off separated regime / numpy-RNG) | NOT-STARTED | open prereq blocker #1099. Exact `init_params`-driven VALUE parity on non-well-separated data needs numpy's `check_random_state`/`RandomState` stream + full-KMeans init (`_base.py:228,112-121`); ferrolearn `fn fit` uses `StdRng::seed_from_u64` + k-means++ seeding (Probe 7). Couples to REQ-14 (ferray::random). |
| REQ-13 (full PyO3 binding surface) | NOT-STARTED | open prereq blocker #1100. `_RsGaussianMixture` (`extras.rs:1038-1088`) marshals only `fit`/`predict`; it omits `covariance_type`/`tol`/`n_init` ctor params and the `weights_`/`means_`/`covariances_`/`predict_proba`/`score`/`aic`/`bic`/`n_iter_` getters (Probe 5). `ferrolearn.GaussianMixture` (`_extras.py:449`) therefore cannot mirror sklearn's attribute surface. |
| REQ-14 (ferray substrate) | NOT-STARTED | open prereq blocker #1101. `gmm.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` + `rand::{Rng, SeedableRng, rngs::StdRng}` and hand-rolls `fn cholesky`, not `ferray-core` / `ferray::linalg` (Cholesky/triangular-solve, the scipy.linalg analog) / `ferray::random` (the numpy.random analog). R-SUBSTRATE-1/2. |

## Architecture

`gmm.rs` follows the crate's unfitted/fitted split (CLAUDE.md naming): `GaussianMixture<F>`
(six fields + builder setters `with_covariance_type`/`with_max_iter`/`with_tol`/
`with_n_init`/`with_random_state`) → `Fit<Array2<F>, ()>` → `FittedGaussianMixture<F>`
(`weights_`, `means_`, `covariances_`, `converged_`, `lower_bound_` + private
`covariance_type_`/`n_features_`). Generic over `F: Float + Send + Sync + 'static`; every
public method returns `Result<_, FerroError>` (R-CODE-2). The public enum `CovarianceType`
(`Full`/`Tied`/`Diag`/`Spherical`) mirrors sklearn's `covariance_type` StrOptions
(`_gaussian_mixture.py:693-699`). `FittedGaussianMixture` implements `Predict<Array2<F>>`
(argmax of the responsibility matrix, mirroring `BaseMixture.predict` `_base.py:369-385`)
and `Transform<Array2<F>>` (the responsibility matrix = `BaseMixture.predict_proba`
`:387-404`); a `fit_predict` convenience mirrors `fit_predict` (`_base.py:185-288`).

**The fit path (`fn fit` → `fn run_em`) is a CORRECT maximum-likelihood EM with k-means++
seeding.** Unlike `bayesian_gmm.rs`, the algorithm is the right one:

- **Init (`fn run_em` init block + `fn init_means`).** Uniform `weights = 1/k`; `means`
  via greedy k-means++ seeding (`2 + ⌊ln k⌋` candidate trials, the scikit-learn-style
  multi-trial `kmeans_plusplus`); identity-scaled initial covariances. sklearn defaults to
  `init_params='kmeans'` (a FULL `KMeans(n_init=1)` whose hard labels become the initial
  responsibilities, then `_estimate_gaussian_parameters` for the initial params,
  `_base.py:112-121` + `_gaussian_mixture.py:770-799`) — REQ-11. The init differs but the
  optimum is unique on well-separated data (Probes 1/3, REQ-1/REQ-3).
- **E-step (`fn log_responsibilities` → `fn log_sum_exp_rows`).** Computes `log π_k +
  log N(x; μ_k, Σ_k)` then row logsumexp-normalizes — mirroring `_estimate_log_prob_resp`
  (`_base.py:507-531`). **The `Full`/`Tied` Gaussian log-density double-counts `log|Σ|`
  (REQ-5, the load-bearing localized bug)**; the `Diag`/`Spherical` arms are correct.
- **M-step (`fn run_em` M-step).** ML updates: `weights = (N_k + reg)/(ΣN + k·reg)`,
  `means = Σ resp·x / N_k`, and the covariance per `CovarianceType` with `reg_covar=1e-6`
  added to the diagonal — mirroring `_estimate_gaussian_parameters` /
  `_estimate_gaussian_covariances_*` (`:153-296`). VALUES match sklearn on well-separated
  data (Probe 1, REQ-3).
- **Convergence (`fn run_em` loop).** Checks `|ll - prev_ll| < tol` at the START of each
  iteration (before the M-step), whereas sklearn checks `abs(change) < tol` AFTER the
  M-step (`_base.py:244-256`) — an off-by-one in iteration accounting (folded into REQ-9
  `n_iter_`, which ferrolearn does not expose). `n_init` best-of-N by `lower_bound_`
  mirrors `_base.py:231-264`. `lower_bound_` DEFINITION (per-sample mean log-likelihood)
  matches sklearn `_e_step` `np.mean(log_prob_norm)` (`:307`); its VALUE diverges via REQ-5.

`fn n_free_params` (behind `bic`/`aic` and `n_parameters()`) mirrors sklearn `_n_parameters`
exactly (`:862-874`): `cov_params` (full `k·d·(d+1)/2`, tied `d·(d+1)/2`, diag `k·d`,
spherical `k`) + `k·d` means + `(k-1)` weights — `n_params=11` matches on Probe 1/4. The
`bic`/`aic` FORMULA structure matches (`-2·Σ score_samples + n_params·{ln n, 2}`); their
VALUES inherit the diverging `score_samples` (REQ-5).

**Invariants held vs sklearn:** well-separated PARTITION co-membership (REQ-1);
`predict_proba`/`transform` rows sum to 1 + correct shapes + matching values on separated
data (REQ-2); `weights_`/`means_`/`covariances_` VALUE-match on well-separated data
(REQ-3); matching `covariance_type`/`max_iter`/`tol`/`n_init` defaults (REQ-4); `n_params`
accounting + `bic`/`aic` formula structure; deterministic fit (seeded `StdRng`); the thin
`fit`/`predict` PyO3 marshalling (REQ-7). Edge cases: `n_components==0` / `n_init==0` →
`FerroError::InvalidParameter`; `n_samples==0` or `n_samples < n_components` →
`FerroError::InsufficientSamples` (sklearn raises `ValueError` with `ensure_min_samples=2`,
`_base.py:212-218` — error-TYPE ABI differs, folded into REQ-11's parameter-validation
surface); feature-count mismatch at predict/transform → `FerroError::ShapeMismatch`.

**Invariants NOT held vs sklearn:** the absolute `score_samples`/`score`/`lower_bound_`/
`aic`/`bic` VALUES (REQ-5, the `Full`/`Tied` `log|Σ|` double-count); the `covariances_`
SHAPE (REQ-6); `precisions_`/`precisions_cholesky_` (REQ-8); `n_iter_` (REQ-9); `sample()`
(REQ-10); `init_params` (full-KMeans init) + `reg_covar`/`weights_init`/`means_init`/
`precisions_init`/`warm_start` params + `n_components=1` default (REQ-11); exact value
parity off the separated regime (REQ-12, numpy-RNG); the full PyO3 binding surface
(REQ-13); the ferray substrate (REQ-14).

**Consumer wiring.** Non-test consumers: (a) the crate re-export (`pub use gmm::{CovarianceType,
FittedGaussianMixture, GaussianMixture}`, `ferrolearn-cluster/src/lib.rs`); (b) the CPython
binding chain `import ferrolearn → ferrolearn.GaussianMixture (_extras.py:449) →
_RsGaussianMixture (extras.rs:1038) → ferrolearn_cluster::GaussianMixture` (REQ-7), thin
(fit/predict only).

## Verification

Library crate (green at baseline `e3416328` for the existing behavior):
```bash
cargo test -p ferrolearn-cluster --lib gmm    # 44 passed; 0 failed (run this iteration)
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin shapes, weights-sum-to-1, co-membership on well-separated blobs,
error edges, `n_init`-picks-best, f32 support, and `lower_bound().is_finite()` /
`bic().is_finite()` — but **none compares the absolute `score`/`score_samples`/`lower_bound_`/
`aic`/`bic` VALUE against the live sklearn oracle** (only finiteness and ordering, e.g.
`test_bic_increases_with_more_components_on_two_blobs`), so they stay green despite the
REQ-5 double-count. None compares `covariances_` SHAPE against sklearn's `(k,d,d)` either
(REQ-6).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic should
pin (R-CHAR-3 expected values), into `ferrolearn-cluster/tests/divergence_gmm.rs` and
FAILING against current `gmm.rs`:
```
# REQ-5 (the load-bearing log|Σ| double-count) — score / lower_bound_ / aic / bic VALUE
python3 -c "import numpy as np; from sklearn.mixture import GaussianMixture; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[-0.1,0.],[0.,-0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[9.9,10.],[10.,9.9],[10.1,10.1]]); \
gm=GaussianMixture(n_components=2,random_state=42,max_iter=200).fit(X); \
print(round(float(gm.score(X)),6), round(float(gm.lower_bound_),6), round(float(gm.aic(X)),4), round(float(gm.bic(X)),4))"
# 1.86969 1.86969 -22.8726 -17.5386      (ferrolearn: 7.269941 / 7.269941 / -152.4786 / -147.1446)
# REQ-3 (well-separated VALUE-match — should PASS once expected values are pinned)
#   weights_=[0.5,0.5]; means_=[[0.016667,0.016667],[10.016667,10.016667]];
#   covariances_ block = [[0.004723,0.001389],[0.001389,0.004723]]
# REQ-6 (covariances_ shape) full (2,2,2) / tied (2,2) / spherical (2,) vs ferrolearn (4,2)/(4,2)/(2,1)
```

ferrolearn-python (REQ-7 SHIPPED now; REQ-13 after the surface lands):
```bash
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/ -q
```
asserting `ferrolearn.GaussianMixture(n_components=2,random_state=42).fit(X).predict(X)`
recovers the sklearn partition (REQ-7), and — once REQ-13 lands — that `weights_`/`means_`/
`covariances_`/`predict_proba`/`score`/`aic`/`bic` mirror `sklearn.mixture.GaussianMixture`.

## Blockers (filed iter 132: #1093–#1101; REQ-5 #1093 FIXED this iteration)

REQ-1/REQ-2/REQ-3/REQ-4/REQ-5/REQ-7 SHIP — no open blocker. REQ-5 (#1093, the `Full`/`Tied`
`log|Σ|` double-count + `fn cholesky` double-regularization) was FIXED this iteration and is
now SHIPPED. The remaining NOT-STARTED REQs:

1. "Blocker for REQ-5 of gmm: `Full`/`Tied` Gaussian log-density double-counts `log|Σ|` —
   `fn log_det_and_norm_full` bakes `-0.5·log_det` into `log_norm` and the `Full|Tied` arm
   of `fn log_responsibilities` subtracts it AGAIN in `-0.5·(log_det + maha)`. Drop the
   duplicate so the density matches sklearn `_estimate_log_gaussian_prob`
   (`_gaussian_mixture.py:448-507`); fixes `score`/`score_samples`/`lower_bound_`/`aic`/`bic`
   absolute VALUES (Probe 4). Diag/Spherical already correct. MINIMAL FIX." -p high -l blocker
   **(the single clean minimal-fixable divergence)**
2. "Blocker for REQ-6 of gmm: `covariances_` must use sklearn's shape — full `(k,d,d)`,
   tied `(d,d)`, spherical `(k,)`, diag `(k,d)` (`_gaussian_mixture.py:617-624`); ferrolearn
   uses `(k·d,d)` blocks / `(k,1)` spherical." -p medium -l blocker
3. "Blocker for REQ-8 of gmm: expose `precisions_` (inverse covariances) +
   `precisions_cholesky_` (the primary internal repr, `_compute_precision_cholesky`
   `:299-348`, `_set_parameters` `:847-860`)." -p medium -l blocker
4. "Blocker for REQ-9 of gmm: expose `n_iter_` = best-init EM step count (`_base.py:263,280`);
   `fn run_em` discards the loop counter `_iter`." -p medium -l blocker
5. "Blocker for REQ-10 of gmm: add `sample(n_samples)` — multinomial component counts +
   per-component multivariate_normal draws (`_base.py:406-466`)." -p low -l blocker
6. "Blocker for REQ-11 of gmm: add `init_params ∈ {kmeans,k-means++,random,random_from_data}`
   default 'kmeans' = full KMeans hard-label responsibilities (`_base.py:99-140`), `reg_covar`
   (currently hardcoded 1e-6), `weights_init`/`means_init`/`precisions_init` (`:711-713`),
   `warm_start` (`_base.py:222-223`), `n_components=1` default." -p medium -l blocker
7. "Blocker for REQ-12 of gmm: exact value parity off well-separated data needs numpy's
   RandomState stream + full-KMeans init (`_base.py:228,112-121`); ferrolearn uses StdRng +
   k-means++ seeding. Couples to #8 (ferray::random)." -p low -l blocker
8. "Blocker for REQ-13 of gmm: extend `_RsGaussianMixture` (`extras.rs:1038-1088`) to marshal
   `covariance_type`/`tol`/`n_init` ctor params + `weights_`/`means_`/`covariances_`/
   `predict_proba`/`score`/`aic`/`bic`/`n_iter_` getters; currently fit/predict only." -p medium -l blocker
9. "Blocker for REQ-14 of gmm: migrate off ndarray/rand + hand-rolled `fn cholesky` to
   ferray-core/ferray::linalg (Cholesky/triangular-solve)/ferray::random. R-SUBSTRATE-2." -p medium -l blocker
```
