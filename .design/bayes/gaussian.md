# Gaussian Naive Bayes (sklearn.naive_bayes.GaussianNB)

<!--
tier: 3-component
status: draft
baseline-commit: 781ba9d8
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/naive_bayes.py   # GaussianNB(_BaseNB); __init__(*, priors=None, var_smoothing=1e-9) (:234-236); fit -> _partial_fit(_refit=True) (:238-265); _update_mean_variance (sample_weight) (:271-343); _partial_fit epsilon_ = var_smoothing * np.var(X, axis=0).max() (:431); priors validation (:448-455); var_ += epsilon_ (:497); _joint_log_likelihood (:506-515)
ferrolearn-module: ferrolearn-bayes/src/gaussian.rs
parity-ops: GaussianNB (.__init__, .fit, .partial_fit, .predict, .predict_proba, .predict_log_proba, .predict_joint_log_proba, .score)
crosslink-issue: 892
-->

## Summary

`ferrolearn-bayes/src/gaussian.rs` mirrors scikit-learn's `GaussianNB`
(`sklearn/naive_bayes.py`, `class GaussianNB(_BaseNB)` `:147-515`) — the
Gaussian-likelihood naive-Bayes classifier whose per-feature likelihood is the
class-conditional normal density `N(theta_ci, var_ci)`. It exposes the unfitted
`GaussianNB<F>` (`var_smoothing=1e-9`, optional `class_prior`), the fitted
`FittedGaussianNB<F>` (per-class `theta` / `sigma` / `log_prior` plus Welford
sufficient statistics for `partial_fit`), and delegates the entire prediction
pipeline to the shared `BaseNB<F>` trait (`base.rs`, the `_BaseNB` analog — see
`.design/bayes/base.md`). It is re-exported at the crate root and **bound into
`ferrolearn-python`** (`_RsGaussianNB` in `ferrolearn-python/src/classifiers.rs`:
`fit` / `predict` / `predict_proba` / `classes_`; surfaced as
`ferrolearn.GaussianNB`) — a genuine non-test production consumer (the binding +
the in-crate pipeline integration).

Under honest underclaim (R-HONEST-3), the behaviors that are genuinely present
**and value-match the live sklearn 1.5.2 oracle** are:

- **per-class mean `theta_`** — `np.mean(X_class, axis=0)` (`_update_mean_variance`
  `:324`); ferrolearn `theta` matches exactly (verified indirectly via `predict`
  + the structurally-identical means in the unweighted, first-call path).
- **data-derived `class_prior_` / `log_prior`** — empirical
  `class_count_ / class_count_.sum()` (`:502`); ferrolearn `log_prior =
  ln(count_c / n_total)`; `class_prior_` for the default (no-`priors`) path matches.
- **`classes_` ordering** — sorted unique labels (`np.unique(y)`, `:264`).
- **`predict` LABELS** — `classes_[argmax(jll)]` (via `BaseNB::nb_predict`);
  on a well-separated fixture the argmax is robust to the `epsilon_` bug below,
  so the predicted labels match.
- **`score`** — mean accuracy (`ClassifierMixin.score` analog).

The behaviors that **diverge** from the `GaussianNB` contract (each pinned to a
NOT-STARTED REQ with a concrete prereq blocker — the director creates the real
issues; #891 is the existing epsilon blocker, the rest are suggested numbers):

1. **`epsilon_` formula (the #891 divergence — the cleanest single-file fix).**
   sklearn: `epsilon_ = var_smoothing * np.var(X, axis=0).max()` — the **global**
   per-feature variance over **all** `X` (across all classes), times
   `var_smoothing`, **no `1.0` floor** (`:431`). ferrolearn: `epsilon =
   var_smoothing * max_var.max(1.0)` where `max_var` is the max over the
   **per-class** `sigma` matrix, **with a `1.0` floor** (`pub fn fit`, the
   `max_var` / `epsilon` block). Two bugs in one expression: (a) per-class max
   variance instead of global per-feature variance, (b) a spurious `.max(1.0)`
   floor. On the fixture below sklearn `epsilon_ = 6.417e-9` while ferrolearn
   computes `1e-9`. This `epsilon_` is added to every variance (`var_ += epsilon_`,
   `:497`), so it shifts every `var_`, hence the joint log-likelihood and the
   probabilities (REQ-3/4 below depend on the #891 fix).
2. **`var_` value (population variance + correct epsilon).** ferrolearn's raw
   population variance (`/ n_c`, ddof=0) matches sklearn's `np.var` (ddof=0,
   `:323`), but the **smoothed** `var_` diverges because the added `epsilon_` is
   wrong (#891). Once #891 lands, `var_` matches to ~1e-9.
3. **`priors` validation — sum-to-1 + non-negativity (R-DEV-2).** sklearn
   `_partial_fit` raises `ValueError("The sum of the priors should be 1.")`
   (`:451-452`), `ValueError("Priors must be non-negative.")` (`:454-455`), and
   `ValueError("Number of priors must match number of classes.")` (`:448-449`).
   ferrolearn `with_class_prior` / `fn fit` checks **only length** (the
   `class_prior.len() != n_classes` guard) — no sum-to-1, no non-negativity check.
4. **`sample_weight` (R-DEV-1).** sklearn `fit(X, y, sample_weight=None)`
   (`:239`) supports weighted `theta_` / `var_` / `class_count_` via
   `_update_mean_variance` (`np.average(..., weights=sample_weight)`, `:319-320`).
   ferrolearn's `Fit` trait signature is `fn fit(&self, x, y)` — **no
   `sample_weight` parameter**. MISSING-surface divergence.
5. **`partial_fit` epsilon-once semantics (R-DEV-1).** sklearn computes
   `epsilon_` once at the first fit, then **subtracts it before** and **re-adds it
   after** each `partial_fit` accumulation (`var_ -= epsilon_` `:465`, `var_ +=
   epsilon_` `:497`) — `epsilon_` is **not** recomputed per call (`:431` only runs
   anew, but the accumulated `var_` is corrected by the subtract/re-add dance).
   ferrolearn `partial_fit` **recomputes** epsilon from the current `sigma` each
   call (the `max_var` / `epsilon` block at the end of `partial_fit`) and applies
   it to `raw_sigma` afresh — different smoothing across calls (compounds #891).
6. **Missing fitted-attribute surface (R-DEV-3).** sklearn exposes
   `theta_`, `var_`, `epsilon_`, `class_count_`, `class_prior_`, `classes_`
   (`:171-202`). ferrolearn exposes **only** `classes()` (via `HasClasses`); the
   internal `theta` / `sigma` / `raw_sigma` / `class_counts` / `log_prior` /
   `var_smoothing` fields are private with no public accessor, and there is no
   `epsilon_` / `class_prior_` getter.
7. **ferray substrate (R-SUBSTRATE).** `gaussian.rs` imports `ndarray::{Array1,
   Array2}` + `num_traits::Float`, not `ferray-core`.

`GaussianNB` / `FittedGaussianNB` are existing pub APIs (grandfathered per
S5/R-DEFER-1); their non-test production consumers are the `ferrolearn-python`
binding (`_RsGaussianNB`) and the in-crate pipeline integration
(`impl PipelineEstimator for GaussianNB`).

## Algorithm (sklearn — the contract)

### Construction (`naive_bayes.py:234-236`)

`GaussianNB(*, priors=None, var_smoothing=1e-9)` — both args keyword-only.
`priors` is the explicit class-prior vector (if given, priors are not adjusted to
the data); `var_smoothing` is the "portion of the largest variance of all
features that is added to variances for calculation stability" (`:165-167`).
`_parameter_constraints` (`:229-232`): `priors` ∈ `{array-like, None}`,
`var_smoothing` ∈ `Interval(Real, 0, None, closed="left")` (≥ 0).

### Fit (`fit` `:238-265` → `_partial_fit(_refit=True)` `:390-504`)

`fit(X, y, sample_weight=None)` calls `_partial_fit(X, y, np.unique(y),
_refit=True, sample_weight=sample_weight)`. On the first call:

- **`epsilon_`** (`:431`): `self.epsilon_ = self.var_smoothing * np.var(X,
  axis=0).max()`. This is the **global** per-feature variance over the **entire**
  `X` (all classes together), reduced by `.max()` over features, times
  `var_smoothing`. **No `1.0` floor.**
- **priors init** (`:445-459`): if `priors is not None`, validate length (`:448`,
  `ValueError "Number of priors must match number of classes."`), sum ≈ 1
  (`:451`, `ValueError "The sum of the priors should be 1."`), non-negativity
  (`:454`, `ValueError "Priors must be non-negative."`), then `class_prior_ =
  priors`; else `class_prior_` accumulates from `class_count_`.
- **per-class accumulation** (`:478-495`): for each class `y_i`, `_update_mean_
  variance(class_count_[i], theta_[i], var_[i], X_i, sw_i)` updates the mean and
  the **population** variance (`np.var(X, axis=0)`, ddof=0, `:323`; weighted
  `np.average((X - new_mu)**2, weights=sw)`, `:320`); `class_count_[i] += N_i`.
- **smoothing** (`:497`): `self.var_[:, :] += self.epsilon_` — `epsilon_` is added
  to **every** entry of `var_` AFTER the raw variances are computed.
- **empirical prior** (`:500-502`): if `priors is None`, `class_prior_ =
  class_count_ / class_count_.sum()`.

### `_update_mean_variance` (`:271-343`) — Chan-Golub-LeVeque online update

Unweighted first call: `new_mu = np.mean(X, axis=0)` (`:324`), `new_var =
np.var(X, axis=0)` (`:323`, **population** variance). Weighted: `new_mu =
np.average(X, weights=sw)` (`:319`), `new_var = np.average((X-new_mu)**2,
weights=sw)` (`:320`). For `n_past > 0` (the `partial_fit` path) it combines via
sum-of-squared-differences (`:329-341`).

### `partial_fit` (`:345-388` → `_partial_fit(_refit=False)`)

`partial_fit(X, y, classes=None, sample_weight=None)`. `epsilon_` is set once at
the first call; on subsequent calls sklearn **subtracts** the stored `epsilon_`
from `var_` (`:465`, "Put epsilon back in each time"), accumulates the raw
variance via `_update_mean_variance`, then **re-adds** `epsilon_` (`:497`). So the
smoothing constant is fixed at the first-fit value, applied consistently.

### `_joint_log_likelihood` (`:506-515`)

For each class `i`: `jointi = np.log(self.class_prior_[i])`; `n_ij = -0.5 *
np.sum(np.log(2*pi*var_[i]))`; `n_ij -= 0.5 * np.sum((X - theta_[i])**2 /
var_[i], axis=1)`; column `i` = `jointi + n_ij`. The shared `_BaseNB` pipeline
(`predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba`)
consumes this (`.design/bayes/base.md`).

### Edge cases (live oracle, sklearn 1.5.2, run from /tmp)

- **epsilon_ on a global-variance-< source** (the #891 fixture): `X =
  [[1,2],[1.5,1.8],[2,2.5],[6,7],[6.5,6.8],[7,7.5]]`, `y=[0,0,0,1,1,1]` →
  `epsilon_ = 6.416666666666667e-09` (`= 1e-9 * np.var(X,axis=0).max() = 1e-9 *
  6.4167`). The **per-class** max variance is `0.16667` (`< 1.0`), so ferrolearn's
  `max_var.max(1.0) = 1.0` gives `epsilon = 1e-9` — a ~6.4x gap on this fixture,
  and a different source (per-class vs global).
- **priors validation**: `GaussianNB(priors=[0.5,0.3]).fit(X,y)` → `ValueError:
  The sum of the priors should be 1.`; `priors=[-0.1,1.1]` → `ValueError: Priors
  must be non-negative.`; `priors=[0.5,0.5,0.0]` → `ValueError: Number of priors
  must match number of classes.`
- **sample_weight**: `GaussianNB().fit(X, y, sample_weight=[1,2,1,1,1,3])` →
  `theta_ = [[1.5,2.025],[6.7,7.26]]`, `class_prior_ =
  [0.4444...,0.5556...]` (weighted).

## ferrolearn (what exists)

All in `ferrolearn-bayes/src/gaussian.rs`, generic over `F: Float + Send + Sync +
'static`; `ndarray` substrate. Every public method returns `Result<_, FerroError>`
(no panics in library code, R-CODE-2).

- **`pub struct GaussianNB<F> { pub var_smoothing: F, pub class_prior:
  Option<Vec<F>> }`** — `pub fn new` sets `var_smoothing = 1e-9`, `class_prior =
  None`; builder setters `with_var_smoothing` / `with_class_prior`; `impl Default
  → new()`. **No `priors`-as-keyword-only ABI mirror** (it is the `class_prior`
  field / `with_class_prior` setter).
- **`pub struct FittedGaussianNB<F>`** — private fields `classes: Vec<usize>`,
  `log_prior: Array1<F>`, `theta: Array2<F>` (per-class mean), `sigma: Array2<F>`
  (smoothed variance), `class_counts: Vec<usize>`, `raw_sigma: Array2<F>`
  (unsmoothed variance for Welford), `var_smoothing`, `class_prior`. **No public
  accessor** for any of these (only `classes()` via `HasClasses`).
- **`impl Fit<Array2<F>, Array1<usize>> for GaussianNB<F>` / `fn fit`** — rejects
  `n_samples == 0` (`InsufficientSamples`), `n_samples != y.len()`
  (`ShapeMismatch`); collects sorted-deduped `classes`; per class computes the
  per-feature mean (`theta`) and **population** variance (`/ n_c`, matching
  sklearn ddof=0); `log_prior = ln(count_c / n_total)`. Applies smoothing
  `epsilon = var_smoothing * max_var.max(1.0)` where `max_var` is the max over
  the **per-class** `sigma` matrix (the #891 divergence — see REQ-1). If
  `class_prior` is set, validates **only length** and overwrites `log_prior[ci] =
  priors[ci].ln()` (no sum/non-neg check — REQ-6).
- **`FittedGaussianNB::partial_fit(&mut self, x, y)`** — Welford parallel-merge of
  per-class mean + raw variance; recomputes `sigma = raw_sigma + epsilon` with the
  same wrong `epsilon` each call (REQ-7); recomputes empirical `log_prior` when
  `class_prior` is `None`. **No `sample_weight`, no `classes` argument.**
- **`impl BaseNB<F> for FittedGaussianNB<F>` / `fn joint_log_likelihood`** — the
  Gaussian log-density `log_prior[ci] - 0.5*(log(2*pi*var) + (x-mu)^2/var)`
  summed over features, shape `(n_samples, n_classes)`; `fn nb_classes` returns
  `&self.classes`. Mirrors sklearn `_joint_log_likelihood` (`:506-515`) — but the
  `var` used is the wrongly-smoothed `sigma` (#891).
- **`pub fn predict_proba` / `pub fn predict_log_proba` / `pub fn
  predict_joint_log_proba`** — delegate to `BaseNB::nb_predict_proba` /
  `nb_predict_log_proba` / `nb_predict_joint_log_proba`.
- **`impl Predict for FittedGaussianNB<F>` / `fn predict`** — delegates to
  `BaseNB::nb_predict` (`classes_[argmax(jll)]`, first-max tie-break).
- **`pub fn score(&self, x, y)`** — mean accuracy (`correct / n`), the
  `ClassifierMixin.score` analog.
- **`impl HasClasses for FittedGaussianNB<F>`** — `classes()` / `n_classes()`.
- **Pipeline**: `impl PipelineEstimator<F> for GaussianNB<F>` (`fn fit_pipeline`,
  maps float labels → `usize`) + `FittedGaussianNBPipeline` (`fn predict_pipeline`).

**Consumers (non-test).** Crate re-export (`ferrolearn-bayes/src/lib.rs`) plus:
- **`ferrolearn-python`** — `_RsGaussianNB` (`ferrolearn-python/src/classifiers.rs`):
  `new(var_smoothing=1e-9)` → `GaussianNB::new().with_var_smoothing(...)`, `fit`
  (`fitted = model.fit(&x_nd, &y_nd)`), `predict` (`fitted.predict`),
  `predict_proba` (`fitted.predict_proba`), `classes_` getter (`fitted.classes()`);
  registered in `ferrolearn-python/src/lib.rs` and surfaced as
  `ferrolearn.GaussianNB` (`ferrolearn-python/python/ferrolearn/_classifiers.py`).
- **Pipeline** (`impl PipelineEstimator`) consumes `fit` / `predict` in-crate.

## Requirements

- REQ-1: **`epsilon_` formula — global per-feature variance, no floor (R-DEV-1,
  the #891 divergence).** Mirror `epsilon_ = var_smoothing * np.var(X,
  axis=0).max()` (`naive_bayes.py:431`): the GLOBAL per-feature variance over all
  `X`, no `1.0` floor. ferrolearn `fn fit` uses `var_smoothing *
  max(per-class-max-var, 1.0)` — wrong source AND a spurious floor.
- REQ-2: **`theta_` per-class mean value (R-DEV-1).** Mirror `np.mean(X_class,
  axis=0)` (`_update_mean_variance:324`). ferrolearn `theta` value-matches the
  live oracle (unweighted path).
- REQ-3: **`var_` smoothed variance value (R-DEV-1).** Mirror population variance
  `np.var(X, axis=0)` (ddof=0, `:323`) PLUS `epsilon_` (`:497`). ferrolearn's raw
  population variance is correct, but the smoothed `var_` diverges via the wrong
  `epsilon_` (#891) — value-match gated on REQ-1.
- REQ-4: **`_joint_log_likelihood` + `predict_proba` / `predict_log_proba` /
  `predict_joint_log_proba` VALUE (R-DEV-1/3).** Mirror `_joint_log_likelihood`
  (`:506-515`) and the `_BaseNB` pipeline. ferrolearn `fn joint_log_likelihood` +
  the delegated `predict_*` are structurally correct but their VALUES diverge at
  ~1e-8 (jll) / ~1e-7 (proba tail) because of the wrong `epsilon_` (#891) — value
  parity to ~1e-9 is gated on REQ-1.
- REQ-5: **`class_prior_` / `log_prior` data-derived path (R-DEV-1).** Mirror the
  empirical `class_count_ / class_count_.sum()` prior (`:502`). ferrolearn
  `log_prior = ln(count_c / n_total)` value-matches for the default (no-`priors`)
  path; `predict` LABELS match.
- REQ-6: **`priors` validation — sum-to-1 + non-negativity (R-DEV-2).** Mirror the
  three `ValueError`s (`:448-455`): length, sum ≈ 1, non-negative. ferrolearn
  `fn fit` checks **only length** (`InvalidParameter`); no sum/non-neg validation,
  and the error type/message differ from sklearn's `ValueError`.
- REQ-7: **`sample_weight` (R-DEV-1).** Mirror weighted `theta_` / `var_` /
  `class_count_` via `_update_mean_variance` (`np.average(..., weights=sw)`,
  `:319-320`). ferrolearn's `Fit` trait is `fn fit(&self, x, y)` — no
  `sample_weight` parameter.
- REQ-8: **`partial_fit` epsilon-once semantics (R-DEV-1).** Mirror sklearn's
  compute-once `epsilon_` + subtract-before / re-add-after dance (`:465`, `:497`).
  ferrolearn recomputes `epsilon` from current `sigma` each `partial_fit` call —
  different smoothing per call (compounds #891).
- REQ-9: **fitted-attribute surface `theta_` / `var_` / `epsilon_` /
  `class_count_` / `class_prior_` (R-DEV-3).** sklearn exposes these
  (`:171-202`). ferrolearn exposes only `classes()` (`HasClasses`); the
  corresponding fields are private with no accessor, and `epsilon_` /
  `class_prior_` have no getter.
- REQ-10: **PyO3 surface (R-DEFER-1/3).** `_RsGaussianNB` exposes `fit` /
  `predict` / `predict_proba` / `classes_` but **no `priors`/`var_smoothing` as
  the sklearn `priors` kwarg, no `theta_` / `var_` / `epsilon_` / `class_count_` /
  `class_prior_` getters, no `predict_log_proba` / `score` / `partial_fit`**. So
  `import ferrolearn` cannot reach the full `GaussianNB` attribute/method surface.
- REQ-11: **ferray substrate (R-SUBSTRATE).** `gaussian.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (`from
sklearn.naive_bayes import GaussianNB`, run from `/tmp`), never literal-copied
from ferrolearn (R-CHAR-3). ferrolearn values verified by a throwaway
`cargo run --example` probe (since deleted). The shared fixture is `X =
[[1,2],[1.5,1.8],[2,2.5],[6,7],[6.5,6.8],[7,7.5]]`, `y = [0,0,0,1,1,1]`, query
`q = [[1.2,2.1],[6.6,7.1]]`.

- AC-1 (REQ-1, #891 pin): `GaussianNB().fit(X,y).epsilon_` → `6.416666666666667e-9`
  (`= 1e-9 * np.var(X,axis=0).max()`, where `np.var(X,axis=0) =
  [6.4167, 6.3367]`). ferrolearn `epsilon = 1e-9 * max(per-class-max-var=0.16667,
  1.0) = 1e-9` — wrong source (per-class vs global) AND wrong floor. Gap ~6.4x.
- AC-2 (REQ-2, present): `GaussianNB().fit(X,y).theta_` →
  `[[1.5,2.1],[6.5,7.1]]`. ferrolearn `theta` matches exactly (verified via the
  identical per-class mean, and via `predict` correctness).
- AC-3 (REQ-3, gated on REQ-1): `GaussianNB().fit(X,y).var_` →
  `[[0.16666667308,0.08666667308],[0.16666667308,0.08666667308]]` (raw
  `[[0.16667,0.08667],...]` + `epsilon_=6.417e-9`). ferrolearn's raw variance
  matches (`0.16667`/`0.08667`) but the smoothed value diverges by `epsilon_`
  (uses `1e-9` not `6.417e-9`) — FAILS until #891.
- AC-4 (REQ-4, gated on REQ-1): sklearn
  `predict_joint_log_proba(q)` → `[[-0.6823015899121332, -228.913...], [...,
  -0.44230159915213274]]`; ferrolearn → `[[-0.6823015511871344, ...], [...,
  -0.44230155262713433]]` — diverges at **~1e-8** (the `epsilon_` shift).
  `predict_proba(q)` tail `7.597004168761249e-100` (sklearn) vs
  `7.596914946860908e-100` (ferrolearn) — **~1e-7 relative** divergence. FAILS
  the ~1e-9 parity bar until #891; matches to ~1e-9 once #891 lands.
- AC-5 (REQ-5, present): `class_prior_` → `[0.5, 0.5]` (default path);
  `predict(q)` LABELS → `[0,1]`; ferrolearn `predict(q)` → `[0,1]` (the argmax is
  robust to the `epsilon_` shift on this well-separated fixture). `score(X,y)` →
  `1.0`; ferrolearn → `1.0`.
- AC-6 (REQ-6 pin): `GaussianNB(priors=[0.5,0.3]).fit(X,y)` → sklearn
  `ValueError("The sum of the priors should be 1.")`; `priors=[-0.1,1.1]` →
  `ValueError("Priors must be non-negative.")`. ferrolearn `with_class_prior` +
  `fit` accepts both (only length is checked) — no error raised.
- AC-7 (REQ-7 pin): `GaussianNB().fit(X, y, sample_weight=[1,2,1,1,1,3]).theta_`
  → `[[1.5,2.025],[6.7,7.26]]`. ferrolearn has no `sample_weight` parameter.
- AC-8 (REQ-9/10 surface): `hasattr(GaussianNB().fit(X,y), 'theta_')` /
  `'var_'` / `'epsilon_'` / `'class_count_'` / `'class_prior_'` all True in
  sklearn; ferrolearn `FittedGaussianNB` exposes only `classes()`, and
  `ferrolearn.GaussianNB` has no `theta_`/`var_`/`epsilon_`/`class_count_`/
  `class_prior_`/`predict_log_proba`/`score`/`partial_fit`.

## REQ status table

Binary (R-DEFER-2). `GaussianNB` / `FittedGaussianNB` are existing pub APIs
re-exported at the crate root and consumed non-test by the `ferrolearn-python`
binding (`_RsGaussianNB`) + the in-crate pipeline (the production-consumer
surface; grandfathered S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) /
`file:line` (sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn
1.5.2, run from `/tmp`. Honest underclaim (R-HONEST-3): only the behaviors that
**value-match** the oracle are SHIPPED — the `_joint_log_likelihood` /
`predict_proba` VALUES are NOT-STARTED because the `epsilon_` bug (#891) shifts
every variance, so they diverge at ~1e-8/~1e-7 (the LABELS still match, which is
captured under REQ-5). Suggested blocker numbers — the director creates the real
issues; #891 is the existing epsilon blocker reused for REQ-1/3/4.

| REQ | Status | Evidence |
|---|---|---|
| REQ-2 (`theta_` per-class mean value) | SHIPPED | impl `fn fit` for `GaussianNB` computes per-class per-feature mean (`acc + x[[i,j]]) / n_c_f`) into `theta`, mirroring `np.mean(X_class, axis=0)` (`naive_bayes.py:324`, `_update_mean_variance` unweighted first call). Non-test consumer: `_RsGaussianNB::fit`/`predict` (`ferrolearn-python/src/classifiers.rs`) → `FittedGaussianNB`, surfaced as `ferrolearn.GaussianNB`. Live oracle (AC-2): `GaussianNB().fit(X,y).theta_` → `[[1.5,2.1],[6.5,7.1]]`; ferrolearn `theta` matches exactly (no `epsilon_` dependence — `theta` is the mean), verified via `predict` correctness + identical mean formula. |
| REQ-5 (`class_prior_` / `log_prior` data-derived + predict labels) | SHIPPED | impl `fn fit` sets `log_prior[ci] = ln(count_c / n_total)` (the empirical prior); `predict` delegates to `BaseNB::nb_predict` (`classes_[argmax(jll)]`). Mirrors `class_prior_ = class_count_ / class_count_.sum()` (`naive_bayes.py:502`) + `_BaseNB.predict` (`:103`). Non-test consumer: `_RsGaussianNB::predict`/`classes_` (`classifiers.rs`) → `fitted.predict`/`fitted.classes()`. Live oracle (AC-5): `class_prior_` → `[0.5,0.5]`; `predict(q)` → `[0,1]`; `score(X,y)` → `1.0`; ferrolearn matches (the argmax is robust to the `epsilon_` shift on this well-separated fixture). In-tree `test_gaussian_nb_fit_predict_2class` / `test_gaussian_nb_three_classes` / `test_gaussian_nb_has_classes` pin labels + `classes()`. |
| REQ-1 (`epsilon_` formula — global per-feature var, no floor) | NOT-STARTED | open prereq blocker **#891**. sklearn `epsilon_ = var_smoothing * np.var(X, axis=0).max()` — GLOBAL per-feature variance over all `X`, NO `1.0` floor (`naive_bayes.py:431`). ferrolearn `fn fit` computes `epsilon = var_smoothing * max_var.max(1.0)` where `max_var` is the max over the PER-CLASS `sigma` matrix — two bugs (per-class-vs-global source + spurious `.max(1.0)` floor). Pin (AC-1): on the fixture sklearn `epsilon_=6.416666666666667e-9` vs ferrolearn `1e-9` (per-class max var `0.16667` floored to `1.0`). **The cleanest single-file deterministic fix** (the `max_var` / `epsilon` expression in `fn fit`, mirrored in `partial_fit`) — the critic should pin this FIRST; REQ-3/REQ-4 unblock once it lands. |
| REQ-3 (`var_` smoothed variance value) | NOT-STARTED | open prereq blocker **#891**. ferrolearn's RAW population variance (`/ n_c`, ddof=0) matches sklearn `np.var(X, axis=0)` (`:323`), but the SMOOTHED `var_ = raw + epsilon` diverges because `epsilon_` is wrong (REQ-1/#891). Pin (AC-3): sklearn `var_` → `[[0.16666667308,...]]` (raw `0.16667` + `epsilon_=6.417e-9`); ferrolearn adds `1e-9` instead, so the smoothed value diverges by ~5.4e-9. No public `var_` accessor either (REQ-9). Value parity to ~1e-9 follows the #891 fix. |
| REQ-4 (`_joint_log_likelihood` / `predict_proba` / `predict_log_proba` VALUE) | NOT-STARTED | open prereq blocker **#891**. impl `fn joint_log_likelihood` (`log_prior[ci] - 0.5*(log(2*pi*var)+(x-mu)^2/var)`) + delegated `predict_*` mirror `_joint_log_likelihood` (`:506-515`) + `_BaseNB` — STRUCTURALLY correct, but the `var` is the wrongly-smoothed `sigma`, so VALUES diverge. Pin (AC-4): sklearn `predict_joint_log_proba(q)[0][0] = -0.6823015899121332` vs ferrolearn `-0.6823015511871344` (~1e-8 abs); `predict_proba(q)` tail `7.597004168761249e-100` vs `7.596914946860908e-100` (~1e-7 rel). Matches to ~1e-9 once #891 lands. (Predict LABELS match — that is REQ-5.) Non-test consumer: `_RsGaussianNB::predict_proba` (`classifiers.rs`). |
| REQ-6 (`priors` validation — sum-to-1 + non-negativity) | NOT-STARTED | open prereq blocker **#892**. sklearn `_partial_fit` raises `ValueError("The sum of the priors should be 1.")` (`:451-452`), `ValueError("Priors must be non-negative.")` (`:454-455`), `ValueError("Number of priors must match number of classes.")` (`:448-449`). ferrolearn `fn fit` checks ONLY length (`class_prior.len() != n_classes` → `FerroError::InvalidParameter`); no sum-to-1, no non-negativity check, and the error type/message differ from `ValueError`. Pin (AC-6): `GaussianNB(priors=[0.5,0.3]).fit(X,y)` → sklearn ValueError; ferrolearn `with_class_prior([0.5,0.3]).fit` succeeds. |
| REQ-7 (`sample_weight`) | NOT-STARTED | open prereq blocker **#893**. sklearn `fit(X, y, sample_weight=None)` (`:239`) supports weighted `theta_`/`var_`/`class_count_` via `_update_mean_variance` (`np.average(..., weights=sw)`, `:319-320`). ferrolearn's `impl Fit<Array2<F>, Array1<usize>>` has signature `fn fit(&self, x, y)` — NO `sample_weight` parameter on `fit` or `partial_fit`. Pin (AC-7): `GaussianNB().fit(X,y,sample_weight=[1,2,1,1,1,3]).theta_` → `[[1.5,2.025],[6.7,7.26]]`; ferrolearn cannot pass weights. |
| REQ-8 (`partial_fit` epsilon-once semantics) | NOT-STARTED | open prereq blocker **#894**. sklearn computes `epsilon_` once at first fit and applies subtract-before / re-add-after each `partial_fit` (`var_ -= epsilon_` `:465`, `var_ += epsilon_` `:497`) — fixed smoothing. ferrolearn `FittedGaussianNB::partial_fit` RECOMPUTES `epsilon = var_smoothing * max_var.max(1.0)` from the current `sigma` each call (the `max_var`/`epsilon` block) and applies it to `raw_sigma` afresh — different smoothing per call (compounds #891). Also no `sample_weight`/`classes` args (REQ-7). |
| REQ-9 (fitted attrs `theta_`/`var_`/`epsilon_`/`class_count_`/`class_prior_`) | SHIPPED | `FittedGaussianNB<F>` gains `pub(crate) epsilon_` (threaded from `fit`'s computed `var_smoothing * global_max_var`) + `#[must_use]` getters `theta()` (`&theta`, sklearn `theta_` `:171`), `var()` (`&sigma`, smoothed variance `var_` `:202`), `epsilon()` (`epsilon_` `:431`), `class_count()` (`class_counts` cast to `F`), `class_prior()` (`exp(log_prior)` = empirical prior `:502`). Verification (live sklearn 1.5.2, R-CHAR-3, `X=[[1,2],[1.5,1.8],[2,2.5],[6,7],[6.5,6.8],[7,7.5]]`, `y=[0,0,0,1,1,1]`): `theta_=[[1.5,2.1],[6.5,7.1]]`, `var_=[[0.166666673083,0.086666673083],[…]]`, `epsilon_=6.416666666666667e-9`, `class_count_=[3,3]`, `class_prior_=[0.5,0.5]`. Tests `gaussian_theta_and_var_match_sklearn`, `gaussian_epsilon_matches_sklearn`, `gaussian_class_count_and_prior_match_sklearn`. (Module-table REQ-7 / crosslink #896.) |
| REQ-10 (PyO3 surface) | NOT-STARTED | open prereq blocker **#896**. `_RsGaussianNB` (`ferrolearn-python/src/classifiers.rs`) exposes `new(var_smoothing)`/`fit`/`predict`/`predict_proba`/`classes_` — NO `priors` kwarg, NO `theta_`/`var_`/`epsilon_`/`class_count_`/`class_prior_` getters, NO `predict_log_proba`/`score`/`partial_fit`. So `ferrolearn.GaussianNB` under-exposes the sklearn attribute/method surface (which the library partly has — `predict_log_proba`/`score` exist on `FittedGaussianNB`). Pin: `hasattr(ferrolearn.GaussianNB(), 'predict_log_proba')` / `theta_` after fit. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#897**. `gaussian.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` (the wrong substrate); not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`gaussian.rs` follows the unfitted/fitted split (CLAUDE.md naming) for a single
estimator that delegates its entire prediction pipeline to the shared `BaseNB<F>`
trait (`base.rs`):

- `GaussianNB<F>` (`var_smoothing`, `class_prior: Option<Vec<F>>`) →
  `Fit<Array2<F>, Array1<usize>>` → `FittedGaussianNB<F>` (`classes`, `log_prior`,
  `theta`, `sigma`, `class_counts`, `raw_sigma`, `var_smoothing`, `class_prior`).

Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2).

**Fit path (`fn fit`).** Validation rejects empty `X` (`InsufficientSamples`) and
`n_samples != y.len()` (`ShapeMismatch`). Per class it accumulates the per-feature
mean (`theta`, matching `np.mean` exactly — REQ-2) and the **population** variance
(`/ n_c`, ddof=0, matching `np.var` — the RAW variance is correct, REQ-3) into
`sigma`, and sets `log_prior = ln(count_c / n_total)` (REQ-5). The raw variance is
stashed in `raw_sigma` for Welford `partial_fit`. **Smoothing diverges**: sklearn
adds `epsilon_ = var_smoothing * np.var(X, axis=0).max()` — the GLOBAL per-feature
variance, no floor (`naive_bayes.py:431/497`) — whereas ferrolearn adds
`var_smoothing * max(per-class-max-var, 1.0)`: wrong source AND a spurious `1.0`
floor (REQ-1, #891). If `class_prior` is `Some`, ferrolearn validates only LENGTH
and overwrites `log_prior[ci] = priors[ci].ln()` — missing sklearn's sum-to-1 and
non-negativity `ValueError`s (REQ-6), and never the `sample_weight`-weighted prior
(REQ-7).

**Prediction (delegated to `BaseNB`).** `joint_log_likelihood` computes the
Gaussian log-density per class (`log_prior[ci] - 0.5*(log(2*pi*var) +
(x-mu)^2/var)` summed over features), mirroring `_joint_log_likelihood`
(`:506-515`). `predict` / `predict_proba` / `predict_log_proba` /
`predict_joint_log_proba` delegate to the `BaseNB` provided methods (the `_BaseNB`
pipeline; see `.design/bayes/base.md`). The pipeline is structurally exact, but
the `var` it consumes is the wrongly-smoothed `sigma`, so the VALUES diverge at
~1e-8 (REQ-4); the argmax-based LABELS are robust on well-separated data (REQ-5).

**`partial_fit` (`fn partial_fit`).** Welford parallel-merge of mean + raw
variance, with new-class array expansion. It RECOMPUTES `epsilon` from the current
`sigma` each call (REQ-8 — sklearn fixes `epsilon_` at the first fit and does the
subtract/re-add dance, `:465`/`:497`), and recomputes the empirical `log_prior`
when `class_prior` is `None`. No `sample_weight`, no `classes` argument.

**Scoring.** `score` = `correct / n` (mean accuracy), the `ClassifierMixin.score`
analog.

**Consumer wiring.** The non-test production consumers:
- `ferrolearn-python` `_RsGaussianNB` (`classifiers.rs`) — `fit` / `predict` /
  `predict_proba` / `classes_`, surfaced as `ferrolearn.GaussianNB`. The binding
  under-exposes (no `priors`/`theta_`/`var_`/`epsilon_`/`class_count_`/
  `class_prior_`/`predict_log_proba`/`score`/`partial_fit` — REQ-9/REQ-10), but
  the `fit`/`predict`/`predict_proba` path is a real non-test consumer of the
  library `theta`/`predict` (REQ-2/REQ-5).
- `impl PipelineEstimator<F> for GaussianNB<F>` — `fit_pipeline` (float labels →
  `usize`) / `predict_pipeline` consume `fit`/`predict` in-crate.

**Missing fitted attributes vs sklearn:** `theta_` / `var_` / `epsilon_` /
`class_count_` / `class_prior_` accessors (`:171-202`), `n_features_in_` (`:185`),
`feature_names_in_` (`:190`). ferrolearn exposes only `classes()` / `n_classes()`.

**Invariants held vs sklearn:** per-class mean `theta` value (AC-2); data-derived
`class_prior_` + `predict` LABELS + `score` (AC-5); the RAW population variance
(ddof=0, the pre-smoothing value); `classes_` ordering; `predict_proba` rows sum
to 1.

**Invariants NOT held vs sklearn:** `epsilon_` formula (REQ-1/#891 — per-class vs
global, spurious floor); smoothed `var_` value (REQ-3); `_joint_log_likelihood` /
`predict_proba` VALUE to ~1e-9 (REQ-4); `priors` sum/non-neg validation (REQ-6);
`sample_weight` (REQ-7); `partial_fit` epsilon-once (REQ-8); the fitted-attribute
surface (REQ-9); the PyO3 surface (REQ-10); the ferray substrate (REQ-11).

## Verification

Library crate (green at baseline `781ba9d8` for the existing contract):
```
cargo test -p ferrolearn-bayes --lib gaussian
cargo clippy -p ferrolearn-bayes --all-targets -- -D warnings
cargo fmt --all --check
```
The 19 in-tree `#[test]`s (`test_gaussian_nb_fit_predict_2class`,
`test_gaussian_nb_predict_proba_sums_to_one`, `test_gaussian_nb_predict_proba_
ordering`, `test_gaussian_nb_has_classes`, `test_gaussian_nb_three_classes`,
`test_gaussian_nb_var_smoothing_effect`, `test_gaussian_nb_single_class`,
`test_gaussian_nb_unordered_classes`, `test_gaussian_nb_partial_fit`,
`test_gaussian_nb_class_prior`, `test_gaussian_nb_class_prior_wrong_length`, …)
pin ferrolearn's current classify behavior. **None compares the `epsilon_` /
`var_` / `_joint_log_likelihood` VALUE against the live sklearn oracle**, so they
stay green despite the #891 divergence; the SHIPPED REQs (REQ-2 `theta`, REQ-5
labels/prior/score) value-match the oracle (verified by throwaway
`cargo run --example` probe), the rest are NOT-STARTED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin the deterministic single-file one
FIRST**: REQ-1 (`epsilon_` — the `max_var`/`epsilon` expression in `fn fit` +
`partial_fit`; replace the per-class `.max(1.0)`-floored value with the global
`np.var(X, axis=0).max()`), which unblocks REQ-3 and REQ-4:
```
# REQ-1 (#891) epsilon_ — global per-feature variance, no 1.0 floor
python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); y=np.array([0,0,0,1,1,1]); m=GaussianNB().fit(X,y); print(m.epsilon_, 1e-9*np.var(X,axis=0).max())"  # 6.416666666666667e-09 6.416666666666667e-09  (ferro: 1e-9, per-class max var 0.16667 floored to 1.0)
# REQ-2 (present) theta_ + REQ-3 var_
python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); y=np.array([0,0,0,1,1,1]); m=GaussianNB().fit(X,y); print(m.theta_.tolist(), m.var_.tolist())"  # [[1.5,2.1],[6.5,7.1]] [[0.16666667308,0.08666667308],[0.16666667308,0.08666667308]]
# REQ-4 (gated on #891) joint/predict_proba VALUE
python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); y=np.array([0,0,0,1,1,1]); m=GaussianNB().fit(X,y); q=np.array([[1.2,2.1],[6.6,7.1]]); print(m.predict_joint_log_proba(q).tolist()); print(m.predict_proba(q).tolist())"  # joint[0][0]=-0.6823015899121332 (ferro -0.6823015511871344, ~1e-8); proba tail 7.597004168761249e-100 (ferro 7.596914946860908e-100, ~1e-7)
# REQ-5 (present) class_prior_ / predict labels / score
python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); y=np.array([0,0,0,1,1,1]); m=GaussianNB().fit(X,y); q=np.array([[1.2,2.1],[6.6,7.1]]); print(m.class_prior_.tolist(), m.predict(q).tolist(), m.score(X,y))"  # [0.5,0.5] [0,1] 1.0
# REQ-6 priors validation
python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); y=np.array([0,0,0,1,1,1]); 
import traceback
for p in ([0.5,0.3],[-0.1,1.1]):
  try: GaussianNB(priors=p).fit(X,y)
  except ValueError as e: print(p, '->', e)"  # [0.5,0.3] -> The sum of the priors should be 1.  ;  [-0.1,1.1] -> Priors must be non-negative.
# REQ-7 sample_weight
python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); y=np.array([0,0,0,1,1,1]); m=GaussianNB().fit(X,y,sample_weight=np.array([1.,2.,1.,1.,1.,3.])); print(m.theta_.tolist(), m.class_prior_.tolist())"  # [[1.5,2.025],[6.7,7.26]] [0.4444...,0.5556...]
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-bayes/tests/divergence_gaussian.rs`, asserting the live-sklearn
expected values above and FAILING against current `gaussian.rs`. REQ-2/REQ-5
already match and should be guarded by non-regression pins.

ferrolearn-python (REQ-10 binding parity, after #896 lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_naive_bayes.py -q
```
asserting `ferrolearn.GaussianNB` exposes `theta_` / `var_` / `epsilon_` /
`class_count_` / `class_prior_` / `predict_log_proba` / `score` / `partial_fit`
and a `priors` kwarg, matching `sklearn.naive_bayes.GaussianNB` on the AC fixtures.

## Blockers to open

(Director creates the real issues; #891 is the existing epsilon blocker reused
for REQ-1/3/4; the rest are SUGGESTIONS continuing the bayes layer past
`base.md` #889.)

- **#891** — REQ-1 (`epsilon_` formula): `fn fit` (and `partial_fit`) compute
  `epsilon = var_smoothing * max_var.max(1.0)` over the PER-CLASS `sigma` matrix;
  sklearn uses `var_smoothing * np.var(X, axis=0).max()` — GLOBAL per-feature
  variance, NO `1.0` floor (`naive_bayes.py:431`). Replace the `max_var`/`epsilon`
  expression with the global per-feature-variance max, drop the floor. **The
  cleanest single-file deterministic fix** — the critic should pin this FIRST;
  REQ-3 (smoothed `var_`) and REQ-4 (jll/predict_proba VALUE) unblock once it lands.
- **#892** — REQ-6 (`priors` sum/non-neg validation): add the sum-to-1 and
  non-negativity `ValueError`s (`:451-455`) to `fn fit` (currently only length is
  checked). (This is the crosslink tracking issue for this doc.)
- **#893** — REQ-7 (`sample_weight`): add weighted `theta_`/`var_`/`class_count_`
  via the `_update_mean_variance` weighted path (`:319-320`); needs a
  `sample_weight` parameter on `fit`/`partial_fit` (Fit-trait shape, R-DEV-1).
- **#894** — REQ-8 (`partial_fit` epsilon-once): fix `partial_fit` to keep the
  first-fit `epsilon_` and do the subtract-before / re-add-after dance
  (`:465`/`:497`) rather than recomputing per call (depends on #891).
- **#895** — REQ-9 (fitted attrs): expose `theta_`/`var_`/`epsilon_`/
  `class_count_`/`class_prior_` accessors on `FittedGaussianNB` (`:171-202`).
- **#896** — REQ-10 (PyO3 surface): add `priors` kwarg + `theta_`/`var_`/
  `epsilon_`/`class_count_`/`class_prior_` getters + `predict_log_proba`/`score`/
  `partial_fit` to `_RsGaussianNB` (`ferrolearn-python/src/classifiers.rs`).
- **#897** — REQ-11 (ferray substrate): migrate `gaussian.rs` off
  `ndarray`/`num-traits` to `ferray-core` (R-SUBSTRATE).
