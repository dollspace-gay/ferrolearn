# Multinomial Naive Bayes (sklearn.naive_bayes.MultinomialNB)

<!--
tier: 3-component
status: draft
baseline-commit: 963c37f6
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/naive_bayes.py   # MultinomialNB(_BaseDiscreteNB); __init__(*, alpha=1.0, fit_prior=True, class_prior=None, force_alpha=True) (:536); _parameter_constraints alpha = Interval(Real, 0, None, closed="left") (:530); _BaseDiscreteNB._check_alpha floor 1e-10 unless force_alpha (:604-626); _count = check_non_negative + feature_count_ += Y.T@X + class_count_ += Y.sum (:879-883); _update_feature_log_prob = log(fc+alpha) - log((fc+alpha).sum(axis=1)) (:885-892); _update_class_log_prior length-only check + log(class_prior) | empirical | uniform (:580-602); _joint_log_likelihood = X @ feature_log_prob_.T + class_log_prior_ (:894-896); check_non_negative "MultinomialNB (input X)" (:881); shared partial_fit (:628-709)
ferrolearn-module: ferrolearn-bayes/src/multinomial.rs
parity-ops: MultinomialNB (.__init__, .fit, .partial_fit, .predict, .predict_proba, .predict_log_proba, .predict_joint_log_proba, .score)
crosslink-issue: 899
-->

## Summary

`ferrolearn-bayes/src/multinomial.rs` mirrors scikit-learn's `MultinomialNB`
(`sklearn/naive_bayes.py`, `class MultinomialNB(_BaseDiscreteNB)` `:773-896`) —
the discrete-count naive-Bayes classifier (word-count / tf-idf text data) whose
per-feature log-likelihood is the smoothed multinomial log-probability
`log_theta_cj = log((N_cj + alpha) / (N_c + alpha * n_features))`. It exposes the
unfitted `MultinomialNB<F>` (`alpha=1.0`, `class_prior: Option<Vec<F>>`,
`fit_prior=true`, `force_alpha=true`), the fitted `FittedMultinomialNB<F>`
(per-class `feature_counts` / `log_theta` / `log_prior` / `class_counts` plus
`alpha`/`class_prior`/`fit_prior` carried for `partial_fit`), and delegates the
entire prediction pipeline to the shared `BaseNB<F>` trait (`base.rs`, the
`_BaseNB` analog — see `.design/bayes/base.md`). It is re-exported at the crate
root (`ferrolearn-bayes/src/lib.rs`) and **bound into `ferrolearn-python`**
(`RsMultinomialNB` / `_RsMultinomialNB` in `ferrolearn-python/src/extras.rs`,
surfaced as `ferrolearn.MultinomialNB` via `_extras.py`) — a genuine non-test
production consumer (the binding `fit`/`predict` + the in-crate pipeline
integration).

Under honest underclaim (R-HONEST-3), the behaviors that are genuinely present
**and value-match the live sklearn 1.5.2 oracle** (verified on the count fixture
`X = [[3,1,0],[2,0,1],[4,2,0],[0,1,4],[1,0,3],[0,2,5]]`, `y = [0,0,0,1,1,1]`,
query `q = [[2,1,1],[0,1,3]]`, run from `/tmp`) are:

- **`feature_log_prob_` smoothing VALUE** — `log(smoothed_fc) - log(smoothed_cc)`
  (`_update_feature_log_prob` `:885-892`); ferrolearn's `log_theta[ci,j] =
  ((count + alpha) / (total + alpha*n_features)).ln()` is the algebraically
  identical closed form (`log((fc+alpha)/(fc.sum+alpha*nf))` — the oracle confirms
  `np.allclose(manual, m.feature_log_prob_)` is `True`). Verified indirectly via
  `predict_joint_log_proba` / `predict_proba` (no public accessor).
- **`_joint_log_likelihood` + `predict` / `predict_proba` / `predict_log_proba` /
  `predict_joint_log_proba` VALUE** — `X @ feature_log_prob_.T + class_log_prior_`
  (`:894-896`) feeding the `_BaseNB` pipeline; ferrolearn matches to ~1e-12.
- **`class_log_prior_` empirical / uniform / explicit** — `log(class_count_) -
  log(class_count_.sum())` (fit_prior, `:600`), `-log(n_classes)` (fit_prior=False,
  `:602`), `log(class_prior)` (explicit, `:591`); ferrolearn's `log_prior` matches
  all three paths.
- **`class_prior` LENGTH-ONLY validation** — sklearn's `_update_class_log_prior`
  checks ONLY length (`:589-590`), with NO sum-to-1 and NO non-negativity check
  for discrete NB (UNLIKE GaussianNB). ferrolearn's `fn fit` checks ONLY length —
  this is a **MATCH**, not a divergence: `MultinomialNB(class_prior=[0.5,0.3])`
  (sum 0.8) fits on both sides.
- **negative-feature guard** — both REJECT negative `X` (sklearn `ValueError`,
  ferrolearn `InvalidParameter`); the exact message/type differs.
- **`partial_fit` VALUE** — incremental count accumulation then recompute; the
  oracle confirms `partial_fit` over two chunks `== fit` on the whole (the
  `feature_log_prob_` is identical), and ferrolearn's `partial_fit` accumulates
  `feature_counts` then recomputes `log_theta`/`log_prior` the same way.
- **`fit_prior` / `force_alpha` behavior** — the empirical-vs-uniform prior toggle
  and the `_check_alpha` floor (`base::check_alpha` / `clamp_alpha`).
- **`score`** — mean accuracy (`ClassifierMixin.score` analog).

The behaviors that **diverge** from the `MultinomialNB` contract (each pinned to a
NOT-STARTED REQ with a concrete prereq blocker — the director creates the real
issues; the numbers below are SUGGESTIONS continuing the bayes layer past
gaussian #897):

1. **`alpha >= 0` validation (R-DEV-2 — THE key fixable divergence).** sklearn's
   `_parameter_constraints` declares `alpha: [Interval(Real, 0, None,
   closed="left"), "array-like"]` (`:530`) → `alpha` must be `>= 0`, enforced by
   `_validate_params` at `fit`. `MultinomialNB(alpha=-0.5).fit(X,y)` raises
   `InvalidParameterError("The 'alpha' parameter of MultinomialNB must be a float
   in the range [0.0, inf) or an array-like. Got -0.5 instead.")`. ferrolearn
   `with_alpha(-0.5)` → `clamp_alpha(-0.5, force_alpha=true)` returns `-0.5`
   unchanged (the floor only fires when `force_alpha=false`), so `fn fit`
   proceeds and computes `log((count - 0.5) / denom)` — garbage / NaN for small
   counts, no error. **The single-file-fixable divergence in `multinomial.rs`
   fit** (an `alpha < 0` guard before `clamp_alpha`). NOTE: this is distinct from
   the `force_alpha` floor (REQ-7, SHIPPED): the floor handles `alpha < 1e-10`
   when `force_alpha=false`; the `>= 0` constraint is a HARD reject regardless of
   `force_alpha`.
2. **`sample_weight` (R-DEV-1).** sklearn `fit(X, y, sample_weight=None)` (`:712`)
   weights the binarized `Y` (`Y *= sample_weight.T`, `:751`) so `feature_count_ =
   Y.T @ X` and `class_count_ = Y.sum(axis=0)` become weighted counts. ferrolearn's
   `Fit` trait is `fn fit(&self, x, y)` — NO `sample_weight` parameter on `fit` or
   `partial_fit`. MISSING-surface divergence.
3. **fitted-attribute surface (R-DEV-3).** sklearn exposes `feature_log_prob_`,
   `class_log_prior_`, `feature_count_`, `class_count_`, `classes_`,
   `n_features_in_`, and the `_BaseDiscreteNB` `coef_` (`= feature_log_prob_[1:]`)
   / `intercept_` (`= class_log_prior_[1:]`) properties. ferrolearn exposes ONLY
   `classes()` (via `HasClasses`); `log_theta` / `log_prior` / `feature_counts` /
   `class_counts` are private fields with no accessor, and there is no `coef_` /
   `intercept_`.
4. **negative-feature error type/message (R-DEV-2, minor sub-item).** sklearn
   raises `ValueError("Negative values in data passed to MultinomialNB (input
   X)")` (`check_non_negative`, `:881`); ferrolearn raises
   `FerroError::InvalidParameter { name: "X", reason: "MultinomialNB requires
   non-negative feature values" }`. Both reject (REQ-5 SHIPPED on the reject
   behavior); the type/message differ.
5. **PyO3 surface (R-DEFER-1/3).** `_RsMultinomialNB` (`extras.rs`, via the
   `py_classifier!` macro) exposes ONLY `new(alpha=1.0, fit_prior=true)` + `fit` +
   `predict` — NO `class_prior` / `force_alpha` kwargs, NO `predict_proba` /
   `predict_log_proba` / `predict_joint_log_proba` / `score` / `partial_fit`, NO
   `feature_log_prob_` / `class_log_prior_` / `feature_count_` / `class_count_` /
   `classes_` / `coef_` / `intercept_` getters. So `import ferrolearn` cannot
   reach the full `MultinomialNB` attribute/method surface (the library has
   `predict_proba` / `predict_log_proba` / `score` / `partial_fit`, but the macro
   does not bridge them).
6. **ferray substrate (R-SUBSTRATE).** `multinomial.rs` imports `ndarray::{Array1,
   Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}`, not `ferray-core`.

`MultinomialNB` / `FittedMultinomialNB` are existing pub APIs (grandfathered per
S5/R-DEFER-1); their non-test production consumers are the `ferrolearn-python`
binding (`RsMultinomialNB` `fit`/`predict`) and the in-crate pipeline integration
(`impl PipelineEstimator for MultinomialNB`).

## Algorithm (sklearn — the contract)

### Construction (`naive_bayes.py:536`)

`MultinomialNB(*, alpha=1.0, fit_prior=True, class_prior=None, force_alpha=True)`
— all keyword-only. `_parameter_constraints` (`:529-534`): `alpha:
[Interval(Real, 0, None, closed="left"), "array-like"]` (**>= 0**, enforced at
`fit` by `_validate_params`); `fit_prior: ["boolean"]`; `class_prior:
["array-like", None]`; `force_alpha: ["boolean"]`.

### Fit (`_BaseDiscreteNB.fit` `:711-763`)

`fit(X, y, sample_weight=None)`: binarize `y` → `Y` (one-hot, shape
`(n_samples, n_classes)`); if `sample_weight` given, `Y *= sample_weight.T`
(`:751`); `_init_counters` zeroes `class_count_` / `feature_count_`; `_count(X,
Y)` accumulates raw counts; `alpha = self._check_alpha()`;
`_update_feature_log_prob(alpha)`; `_update_class_log_prior(class_prior)`.

- **`_count`** (`:879-883`): `check_non_negative(X, "MultinomialNB (input X)")`;
  `self.feature_count_ += safe_sparse_dot(Y.T, X)` (per-class feature sums);
  `self.class_count_ += Y.sum(axis=0)` (per-class sample counts, weighted when
  `sample_weight` was applied to `Y`).
- **`_check_alpha`** (`:604-626`): `alpha_lower_bound = 1e-10` (`:618`); if
  `alpha_min < alpha_lower_bound and not self.force_alpha` (`:619`) warn + return
  `np.maximum(alpha, alpha_lower_bound)`; else return alpha unchanged. (The `>= 0`
  HARD constraint is enforced earlier, by `_validate_params` against
  `_parameter_constraints` `:530`, NOT inside `_check_alpha` — the array-positivity
  branch `:616-617` is scalar-irrelevant.)
- **`_update_feature_log_prob`** (`:885-892`): `smoothed_fc = feature_count_ +
  alpha`; `smoothed_cc = smoothed_fc.sum(axis=1)`; `feature_log_prob_ =
  np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))`. Algebraically
  `log((N_cj + alpha) / (N_c + alpha * n_features))`.
- **`_update_class_log_prior`** (`:580-602`): if `class_prior is not None` →
  `if len(class_prior) != n_classes: raise ValueError("Number of priors must match
  number of classes.")` (`:589-590`); `class_log_prior_ = np.log(class_prior)`
  (`:591`) — **LENGTH-ONLY check, NO sum-to-1, NO non-negativity** (unlike
  GaussianNB). elif `fit_prior` → `class_log_prior_ = log(class_count_) -
  log(class_count_.sum())` (`:597-600`). else → `class_log_prior_ = np.full(
  n_classes, -np.log(n_classes))` (uniform, `:602`).

### `_joint_log_likelihood` (`:894-896`)

`return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_` —
`X @ feature_log_prob_.T + class_log_prior_`, shape `(n_samples, n_classes)`. The
shared `_BaseNB` pipeline (`predict` / `predict_proba` / `predict_log_proba` /
`predict_joint_log_proba`) consumes this (`.design/bayes/base.md`).

### `partial_fit` (`_BaseDiscreteNB.partial_fit` `:628-709`)

`partial_fit(X, y, classes=None, sample_weight=None)`. First call initializes the
counters (`classes` required); each call binarizes `y`, optionally weights `Y`,
`_count(X, Y)` accumulates, then recomputes `alpha = _check_alpha()`,
`_update_feature_log_prob(alpha)`, `_update_class_log_prior(class_prior)`. Because
the smoothing is reapplied to the accumulated counts each call, `partial_fit` over
chunks equals `fit` on the concatenation.

### Edge cases (live oracle, sklearn 1.5.2, run from /tmp)

- **feature_log_prob_ VALUE** (count fixture above, `alpha=1`):
  `feature_log_prob_ = [[-0.47000362924573524, -1.3862943611198906,
  -2.0794415416798357], [-2.251291798606495, -1.5581446180465497,
  -0.3794896217049035]]`; `feature_count_ = [[9,3,1],[1,3,12]]`; the closed-form
  `log((fc+1)/(fc.sum(axis=1)+1*3))` reproduces it exactly
  (`np.allclose(...) == True`).
- **predict VALUE** (`q = [[2,1,1],[0,1,3]]`): `predict_joint_log_proba(q) =
  [[-5.098890341851142, -7.133365017524389], [-8.317766166719343,
  -3.389760663721206]]`; `predict_log_proba(q) = [[-0.12288037781713079,
  -2.1573550534903774], [-4.935220344228254, -0.007214841230117397]]`;
  `predict_proba(q) = [[0.8843694464372913, 0.11563055356270838],
  [0.007188876743869827, 0.9928111232561301]]`; `predict(q) = [0, 1]`.
- **alpha < 0**: `MultinomialNB(alpha=-0.5).fit(X,y)` → `InvalidParameterError:
  The 'alpha' parameter of MultinomialNB must be a float in the range [0.0, inf)
  or an array-like. Got -0.5 instead.` (raised at `fit` by `_validate_params`).
- **alpha = 0, force_alpha default (True)**: `MultinomialNB(alpha=0).fit(X,y)` is
  ACCEPTED (`0` is in `[0, inf)`); `feature_log_prob_ = [[-0.3677247801253172,
  -1.466337068793427, -2.5649493574615367], [-2.772588722239781,
  -1.6739764335716714, -0.2876820724517808]]` (no `-inf` here since every count is
  positive). ferrolearn `clamp_alpha(0, true) = 0` → MATCHES.
- **class_prior length-only**: `MultinomialNB(class_prior=[0.5,0.3]).fit(X,y)`
  (sum 0.8) is ACCEPTED — `class_log_prior_ = [-0.6931471805599453,
  -1.2039728043259361]` (= `log([0.5,0.3])`), NO sum/non-neg error (discrete NB
  has no such check). `class_prior=[0.7,0.3]` (sum 1.0) →
  `[-0.35667494393873245, -1.2039728043259361]`. Wrong length
  `class_prior=[0.5]` → `ValueError: Number of priors must match number of
  classes.` ferrolearn MATCHES all three (length-only).
- **negative features**: `MultinomialNB().fit(X_with_neg, y)` → `ValueError:
  Negative values in data passed to MultinomialNB (input X)`. ferrolearn rejects
  with `InvalidParameter { name: "X" }`.
- **sample_weight**: `MultinomialNB().fit(X, y, sample_weight=[1,2,1,1,1,3])` →
  `feature_count_ = [[11,3,2],[1,7,22]]`, `class_count_ = [4,5]` (weighted).
  ferrolearn has no `sample_weight` parameter.
- **partial_fit == fit**: `partial_fit(X[:4], y[:4], classes=[0,1])` then
  `partial_fit(X[4:], y[4:])` yields `feature_log_prob_` identical to
  `fit(X, y)` (`np.allclose == True`); `feature_count_ = [[9,3,1],[1,3,12]]`.

## ferrolearn (what exists)

All in `ferrolearn-bayes/src/multinomial.rs`, generic over `F: Float + Send +
Sync + 'static`; `ndarray` substrate. Every public method returns `Result<_,
FerroError>` (no panics in library code, R-CODE-2).

- **`pub struct MultinomialNB<F> { pub alpha: F, pub class_prior: Option<Vec<F>>,
  pub fit_prior: bool, pub force_alpha: bool }`** — `pub fn new` sets `alpha = 1.0`,
  `class_prior = None`, `fit_prior = true`, `force_alpha = true` (matching sklearn
  defaults, `:536`); builder setters `with_alpha` / `with_class_prior` /
  `with_fit_prior` / `with_force_alpha`; `impl Default → new()`.
- **`pub struct FittedMultinomialNB<F>`** — private fields `classes: Vec<usize>`,
  `log_prior: Array1<F>` (the `class_log_prior_` analog), `log_theta: Array2<F>`
  (the `feature_log_prob_` analog), `feature_counts: Array2<F>` (the
  `feature_count_` analog), `class_counts: Vec<usize>` (the `class_count_`
  analog), plus `alpha` / `class_prior` / `fit_prior` carried for `partial_fit`.
  **No public accessor** for any of these (only `classes()` via `HasClasses`).
- **`impl Fit<Array2<F>, Array1<usize>> for MultinomialNB<F>` / `fn fit`** —
  rejects `n_samples == 0` (`InsufficientSamples`), `n_samples != y.len()`
  (`ShapeMismatch`), any negative feature (`InvalidParameter { name: "X" }` —
  REQ-5). Collects sorted-deduped `classes`; `alpha = crate::clamp_alpha(
  self.alpha, self.force_alpha)` (the `_check_alpha` floor — REQ-7); per class
  accumulates `all_feature_counts[[ci,j]]` and `total_count`, then `log_theta[
  [ci,j]] = ((count + alpha) / (total_count + alpha * n_features)).ln()` (the
  smoothing closed form — REQ-1) and the empirical `log_prior[ci] = (n_c / n).ln()`
  (REQ-3). Resolves priors: explicit `class_prior` (LENGTH-only check then
  `log_prior[ci] = p.ln()` — REQ-4) wins; else `fit_prior == false` → uniform
  `(1/n_classes).ln()`; else empirical (already filled). **No `alpha >= 0`
  guard** (REQ-2), **no `sample_weight`** (REQ-6).
- **`FittedMultinomialNB::partial_fit(&mut self, x, y)`** — accumulates
  `class_counts` + `feature_counts` for each existing class, then recomputes
  `log_theta` from the accumulated `feature_counts` (same smoothing) and the
  `log_prior` (empirical when `class_prior` is `None` and `fit_prior`; uniform
  otherwise; explicit `class_prior` sticky). Rejects feature-count mismatch /
  negative features. **No `sample_weight`, no `classes` argument** (existing
  classes only — REQ-8/REQ-6).
- **`impl BaseNB<F> for FittedMultinomialNB<F>` / `fn joint_log_likelihood`** —
  `score = log_prior[ci] + sum_j x[i,j] * log_theta[ci,j]`, shape `(n_samples,
  n_classes)`, mirroring `X @ feature_log_prob_.T + class_log_prior_` (`:896` —
  REQ-1); `fn nb_classes` returns `&self.classes`.
- **`pub fn predict_proba` / `pub fn predict_log_proba` / `pub fn
  predict_joint_log_proba`** — delegate to `BaseNB::nb_predict_proba` /
  `nb_predict_log_proba` / `nb_predict_joint_log_proba` (REQ-1 pipeline).
- **`impl Predict for FittedMultinomialNB<F>` / `fn predict`** — delegates to
  `BaseNB::nb_predict` (`classes_[argmax(jll)]`, first-max tie-break).
- **`pub fn score(&self, x, y)`** — mean accuracy (`correct / n`), the
  `ClassifierMixin.score` analog.
- **`impl HasClasses for FittedMultinomialNB<F>`** — `classes()` / `n_classes()`.
- **Pipeline**: `impl PipelineEstimator<F> for MultinomialNB<F>` (`fn
  fit_pipeline`, maps float labels → `usize`) + `FittedMultinomialNBPipeline`
  (`fn predict_pipeline`).

**Consumers (non-test).** Crate re-export (`ferrolearn-bayes/src/lib.rs`,
`pub use multinomial::{FittedMultinomialNB, MultinomialNB}`) plus:
- **`ferrolearn-python`** — `RsMultinomialNB` / `_RsMultinomialNB`
  (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro):
  `new(alpha=1.0, fit_prior=true)` → `MultinomialNB::<f64>::new().with_alpha(
  alpha).with_fit_prior(fit_prior)`, `fit` (`model.fit(&x_nd, &y_nd)`), `predict`
  (`fitted.predict`); registered in `ferrolearn-python/src/lib.rs`
  (`m.add_class::<extras::RsMultinomialNB>()`) and surfaced as
  `ferrolearn.MultinomialNB` (`ferrolearn-python/python/ferrolearn/_extras.py`,
  `class MultinomialNB(_ClassifierWrapper)` → `_RsMultinomialNB(alpha,
  fit_prior)`). The binding under-exposes (no `class_prior`/`force_alpha`/
  `predict_proba`/`predict_log_proba`/`score`/`partial_fit`/fitted-attr getters —
  REQ-9), but the `fit`/`predict` path is a real non-test consumer of the library
  `feature_log_prob_`/`joint_log_likelihood`/`predict` (REQ-1).
- **Pipeline** (`impl PipelineEstimator`) consumes `fit` / `predict` in-crate.

## Requirements

- REQ-1: **`feature_log_prob_` smoothing VALUE + `_joint_log_likelihood` /
  `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba`
  VALUE (R-DEV-1/3).** Mirror `_update_feature_log_prob` (`:885-892`,
  `log(fc+alpha) - log((fc+alpha).sum(axis=1))`) and `_joint_log_likelihood`
  (`:894-896`, `X @ feature_log_prob_.T + class_log_prior_`) feeding the `_BaseNB`
  pipeline. ferrolearn `fn fit` computes `log_theta` via the algebraically
  identical `((count+alpha)/(total+alpha*nf)).ln()`, and `fn
  joint_log_likelihood` + the delegated `predict_*` value-match the oracle to
  ~1e-12.
- REQ-2: **`alpha >= 0` validation (R-DEV-2 — THE key fixable divergence).**
  Mirror `_parameter_constraints` `alpha: Interval(Real, 0, None,
  closed="left")` (`:530`): reject `alpha < 0` at `fit` with an
  `InvalidParameterError`-equivalent. ferrolearn `fn fit` has NO `alpha >= 0`
  guard — `with_alpha(-0.5)` passes through `clamp_alpha` unchanged (force_alpha
  default) and fits on negative-smoothed garbage.
- REQ-3: **`class_log_prior_` empirical / uniform paths VALUE (R-DEV-1).** Mirror
  `log(class_count_) - log(class_count_.sum())` (fit_prior, `:600`) and `-log(
  n_classes)` (fit_prior=False, `:602`). ferrolearn `fn fit` sets `log_prior[ci]
  = ln(count_c / n)` / uniform `(1/n_classes).ln()`; value-matches.
- REQ-4: **`class_prior` explicit path + LENGTH-only validation (R-DEV-1/2 —
  MATCH).** Mirror the explicit branch `if len != n_classes: ValueError;
  class_log_prior_ = log(class_prior)` (`:589-591`) — LENGTH-only, NO sum/non-neg
  check for discrete NB. ferrolearn `fn fit` checks ONLY length then `log_prior[
  ci] = p.ln()` — a deliberate MATCH (a sum-0.8 prior fits on both sides; the
  error type on wrong length differs — `InvalidParameter` vs `ValueError`).
- REQ-5: **negative-feature guard (R-DEV-2).** Mirror `check_non_negative(X,
  "MultinomialNB (input X)")` → `ValueError` (`:881`). ferrolearn `fn fit`
  rejects negative `X` with `FerroError::InvalidParameter { name: "X" }` — both
  REJECT (SHIPPED on the reject behavior); the exact message/type differs.
- REQ-6: **`sample_weight` (R-DEV-1).** Mirror weighted `feature_count_` /
  `class_count_` via `Y *= sample_weight.T` (`:751`) before `_count` (`:879-883`).
  ferrolearn's `impl Fit` is `fn fit(&self, x, y)` — NO `sample_weight`
  parameter on `fit` or `partial_fit`.
- REQ-7: **`force_alpha` / `_check_alpha` floor + `fit_prior` toggle (R-DEV-1).**
  Mirror `_check_alpha` (`:604-626`, floor `1e-10` unless `force_alpha`) and the
  `fit_prior` empirical/uniform selection. ferrolearn `fn fit` calls
  `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`) and
  honors `fit_prior`.
- REQ-8: **`partial_fit` VALUE (R-DEV-1).** Mirror the shared `partial_fit`
  (`:628-709`): accumulate counts then recompute `feature_log_prob_` /
  `class_log_prior_`, so chunked `partial_fit` == whole `fit`. ferrolearn
  `FittedMultinomialNB::partial_fit` accumulates `feature_counts` then recomputes
  `log_theta`/`log_prior`; the oracle confirms two-chunk `partial_fit` ==
  `fit(X,y)`. (No `classes`/`sample_weight` args — REQ-6.)
- REQ-9: **fitted-attribute + PyO3 surface (R-DEV-3 / R-DEFER-1/3).** sklearn
  exposes `feature_log_prob_` / `class_log_prior_` / `feature_count_` /
  `class_count_` / `classes_` / `n_features_in_` + the `_BaseDiscreteNB` `coef_` /
  `intercept_` properties; `_RsMultinomialNB` exposes `class_prior` /
  `force_alpha` kwargs + `predict_proba` / `predict_log_proba` / `score` /
  `partial_fit` + those getters to Python. ferrolearn `FittedMultinomialNB`
  exposes ONLY `classes()`; `_RsMultinomialNB` (the `py_classifier!` macro)
  exposes ONLY `new(alpha, fit_prior)` + `fit` + `predict`.
- REQ-10: **ferray substrate (R-SUBSTRATE).** `multinomial.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}`,
  not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (`from
sklearn.naive_bayes import MultinomialNB`, run from `/tmp`), never literal-copied
from ferrolearn (R-CHAR-3). The shared count fixture is `X =
[[3,1,0],[2,0,1],[4,2,0],[0,1,4],[1,0,3],[0,2,5]]`, `y = [0,0,0,1,1,1]`, query
`q = [[2,1,1],[0,1,3]]`.

- AC-1 (REQ-1, present & matching): `MultinomialNB().fit(X,y).feature_log_prob_`
  → `[[-0.47000362924573524, -1.3862943611198906, -2.0794415416798357],
  [-2.251291798606495, -1.5581446180465497, -0.3794896217049035]]`; the closed
  form `log((feature_count_+1)/(feature_count_.sum(axis=1)+3))` reproduces it
  (`np.allclose == True`). `predict_joint_log_proba(q)` →
  `[[-5.098890341851142, -7.133365017524389], [-8.317766166719343,
  -3.389760663721206]]`; `predict_log_proba(q)` → `[[-0.12288037781713079,
  -2.1573550534903774], [-4.935220344228254, -0.007214841230117397]]`;
  `predict_proba(q)` → `[[0.8843694464372913, 0.11563055356270838],
  [0.007188876743869827, 0.9928111232561301]]`; `predict(q)` → `[0,1]`.
  ferrolearn matches to ~1e-12 (verified indirectly via `predict_joint_log_proba`
  / `predict_proba` — `log_theta` has no public accessor).
- AC-2 (REQ-2 pin): `MultinomialNB(alpha=-0.5).fit(X,y)` →
  `InvalidParameterError("The 'alpha' parameter of MultinomialNB must be a float
  in the range [0.0, inf) or an array-like. Got -0.5 instead.")`. ferrolearn
  `MultinomialNB::new().with_alpha(-0.5).fit(&X,&y)` → **succeeds** (no guard;
  `clamp_alpha(-0.5, true) = -0.5`), producing `log_theta` of negative-smoothed
  counts. FAILS the reject contract until the `alpha >= 0` guard lands.
- AC-3 (REQ-7, present & matching — contrast with AC-2):
  `MultinomialNB(alpha=0).fit(X,y).feature_log_prob_` →
  `[[-0.3677247801253172, -1.466337068793427, -2.5649493574615367],
  [-2.772588722239781, -1.6739764335716714, -0.2876820724517808]]` (alpha=0 in
  `[0,inf)` is ACCEPTED with `force_alpha=True` default). ferrolearn
  `clamp_alpha(0, true) = 0` → MATCHES.
- AC-4 (REQ-4, present & matching — the MATCH, NOT a divergence):
  `MultinomialNB(class_prior=[0.5,0.3]).fit(X,y).class_log_prior_` →
  `[-0.6931471805599453, -1.2039728043259361]` (sum 0.8, NO error — discrete NB
  has no sum/non-neg check). `class_prior=[0.7,0.3]` →
  `[-0.35667494393873245, -1.2039728043259361]`. `class_prior=[0.5]` →
  `ValueError("Number of priors must match number of classes.")`. ferrolearn
  `with_class_prior([0.5,0.3]).fit` succeeds (length-only), `with_class_prior(
  [0.5]).fit` errors (`InvalidParameter`) — MATCHES the accept/reject decisions
  (the wrong-length error TYPE differs).
- AC-5 (REQ-5 pin): `MultinomialNB().fit(X_neg, y)` →
  `ValueError("Negative values in data passed to MultinomialNB (input X)")`.
  ferrolearn rejects with `InvalidParameter { name: "X" }` — both reject; the
  message/type differ.
- AC-6 (REQ-6 pin): `MultinomialNB().fit(X, y,
  sample_weight=[1,2,1,1,1,3]).feature_count_` → `[[11,3,2],[1,7,22]]`,
  `class_count_` → `[4,5]`. ferrolearn has no `sample_weight` parameter.
- AC-7 (REQ-8, present & matching): `MultinomialNB().partial_fit(X[:4], y[:4],
  classes=[0,1]); .partial_fit(X[4:], y[4:]); .feature_log_prob_` equals
  `MultinomialNB().fit(X,y).feature_log_prob_` (`np.allclose == True`);
  `feature_count_` → `[[9,3,1],[1,3,12]]`. ferrolearn's two-chunk `partial_fit`
  reproduces the whole-`fit` `log_theta` (the in-tree
  `test_multinomial_nb_partial_fit` exercises the accumulate-then-recompute path).
- AC-8 (REQ-9 surface): `hasattr(MultinomialNB().fit(X,y), 'feature_log_prob_')`
  / `'class_log_prior_'` / `'feature_count_'` / `'class_count_'` / `'coef_'` /
  `'intercept_'` all True in sklearn; ferrolearn `FittedMultinomialNB` exposes
  only `classes()`, and `ferrolearn.MultinomialNB` exposes only `fit`/`predict`
  (no `predict_proba`/`predict_log_proba`/`score`/`partial_fit`/`class_prior`/
  `force_alpha`/fitted-attr getters).

## REQ status table

Binary (R-DEFER-2). `MultinomialNB` / `FittedMultinomialNB` are existing pub APIs
re-exported at the crate root and consumed non-test by the `ferrolearn-python`
binding (`RsMultinomialNB` `fit`/`predict`) + the in-crate pipeline (the
production-consumer surface; grandfathered S5/R-DEFER-1). Cites use symbol
anchors (ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live oracle =
installed sklearn 1.5.2, run from `/tmp`. Honest underclaim (R-HONEST-3): the
core smoothing + predict VALUES match to ~1e-12 and are SHIPPED; the `alpha >= 0`
reject, `sample_weight`, the fitted-attribute/PyO3 surface, and the ferray
substrate are NOT-STARTED. Suggested blocker numbers — the director creates the
real issues (continuing the bayes layer past gaussian #897).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`feature_log_prob_` smoothing + `_joint_log_likelihood` / predict VALUE) | SHIPPED | impl `fn fit` for `MultinomialNB` sets `log_theta[[ci,j]] = ((count + alpha) / (total_count + alpha*n_features)).ln()`, the algebraic identity of `_update_feature_log_prob` (`naive_bayes.py:885-892`, `log(fc+alpha) - log((fc+alpha).sum(axis=1))`); `fn joint_log_likelihood` computes `log_prior[ci] + sum_j x*log_theta`, mirroring `X @ feature_log_prob_.T + class_log_prior_` (`:894-896`); `predict`/`predict_proba`/`predict_log_proba`/`predict_joint_log_proba` delegate to `BaseNB` (`.design/bayes/base.md`). Non-test consumer: `RsMultinomialNB::fit`/`predict` (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro) → `FittedMultinomialNB`, surfaced as `ferrolearn.MultinomialNB`; plus `impl PipelineEstimator`. Live oracle (AC-1): `feature_log_prob_` → `[[-0.470..,-1.386..,-2.079..],[-2.251..,-1.558..,-0.379..]]` (`np.allclose(closed_form) == True`); `predict_joint_log_proba(q)` → `[[-5.0989..,-7.1334..],[-8.3178..,-3.3898..]]`; `predict_proba(q)` → `[[0.8844..,0.1156..],[0.00719..,0.9928..]]`; `predict(q)` → `[0,1]`; ferrolearn matches to ~1e-12. In-tree `test_multinomial_nb_fit_predict` / `test_multinomial_nb_predict_proba_sums_to_one` pin labels + normalization. |
| REQ-3 (`class_log_prior_` empirical / uniform VALUE) | SHIPPED | impl `fn fit` sets the empirical `log_prior[ci] = (n_c / n).ln()` (default) and the uniform `(1/n_classes).ln()` (`fit_prior == false`), mirroring `log(class_count_) - log(class_count_.sum())` (`naive_bayes.py:600`) and `-log(n_classes)` (`:602`). Non-test consumer: `RsMultinomialNB::predict` → `fitted.predict` (the `class_log_prior_` term feeds the jll). Live oracle: on the fixture the default empirical prior is `[-0.6931..,-0.6931..]` (balanced classes → `log(0.5)`); ferrolearn `log_prior` matches; `predict(q)` → `[0,1]` matches. In-tree `test_multinomial_nb_fit_predict`. |
| REQ-4 (`class_prior` explicit + LENGTH-only validation — MATCH) | SHIPPED | impl `fn fit` validates ONLY `priors.len() != n_classes` then sets `log_prior[ci] = p.ln()`, mirroring `_update_class_log_prior` (`naive_bayes.py:589-591`, `if len != n_classes: ValueError; class_log_prior_ = log(class_prior)`) — discrete NB has NO sum-to-1/non-neg check (UNLIKE GaussianNB). This is a deliberate MATCH. Non-test consumer: `RsMultinomialNB` builds `MultinomialNB` (the library `with_class_prior` path is exercised in-crate + pipeline). Live oracle (AC-4): `class_prior=[0.5,0.3]` (sum 0.8) ACCEPTED → `class_log_prior_ = log([0.5,0.3])`; ferrolearn `with_class_prior([0.5,0.3]).fit` also succeeds; `class_prior=[0.5]` errors on both. In-tree `test_multinomial_nb_class_prior` / `test_multinomial_nb_class_prior_wrong_length`. (Wrong-length error TYPE differs — `InvalidParameter` vs `ValueError` — a minor sub-item folded into REQ-9's surface gap, not a behavioral divergence.) |
| REQ-5 (negative-feature guard — both reject) | SHIPPED | impl `fn fit` rejects any `x[i,j] < 0` with `FerroError::InvalidParameter { name: "X", reason: "MultinomialNB requires non-negative feature values" }`, mirroring `check_non_negative(X, "MultinomialNB (input X)")` → `ValueError` (`naive_bayes.py:881`). Both REJECT. Non-test consumer: `RsMultinomialNB::fit` (`extras.rs`) maps the `FerroError` to `PyValueError`. Live oracle (AC-5): `MultinomialNB().fit(X_neg,y)` → `ValueError("Negative values in data passed to MultinomialNB (input X)")`; ferrolearn errors with `InvalidParameter`. In-tree `test_multinomial_nb_negative_features_error`. The exact sklearn MESSAGE/TYPE (`ValueError` "Negative values…") is NOT matched — that message-parity sub-item is captured under REQ-9 (the binding maps to `PyValueError` so the Python-facing TYPE coincides, but the message text differs). |
| REQ-7 (`force_alpha` floor + `fit_prior` toggle) | SHIPPED | impl `fn fit` calls `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`, the `_check_alpha` floor `1e-10` unless `force_alpha`, `naive_bayes.py:604-626`) and selects empirical/uniform prior on `fit_prior`. Non-test consumer: `RsMultinomialNB` passes `fit_prior` through `with_fit_prior`; `alpha` through `with_alpha`. Live oracle (AC-3): `MultinomialNB(alpha=0).fit(X,y).feature_log_prob_` → `[[-0.3677..,-1.466..,-2.565..],[...]]` (alpha=0 accepted, force_alpha default); ferrolearn `clamp_alpha(0,true)=0` matches. In-tree `test_multinomial_nb_alpha_smoothing_effect` / `test_multinomial_nb_default`; `base.rs` `test_check_alpha_*`. |
| REQ-8 (`partial_fit` VALUE) | SHIPPED | `FittedMultinomialNB::partial_fit` accumulates `class_counts`/`feature_counts` then recomputes `log_theta`/`log_prior` (same smoothing), mirroring the shared `_BaseDiscreteNB.partial_fit` accumulate-then-resmooth (`naive_bayes.py:628-709`, `_count` → `_update_feature_log_prob` → `_update_class_log_prior`). Non-test consumer: in-crate (no PyO3 `partial_fit` — that gap is REQ-9). Live oracle (AC-7): two-chunk `partial_fit(X[:4],y[:4],classes=[0,1])` + `partial_fit(X[4:],y[4:])` yields `feature_log_prob_` identical to `fit(X,y)` (`np.allclose == True`), `feature_count_ = [[9,3,1],[1,3,12]]`; ferrolearn's `partial_fit` reproduces the whole-`fit` `log_theta`. In-tree `test_multinomial_nb_partial_fit` / `test_multinomial_nb_partial_fit_shape_mismatch`. |
| REQ-2 (`alpha >= 0` validation) | NOT-STARTED | open prereq blocker **#900**. sklearn `_parameter_constraints` declares `alpha: [Interval(Real, 0, None, closed="left"), "array-like"]` (`naive_bayes.py:530`) → `alpha >= 0` is a HARD reject at `fit` (`_validate_params`); `MultinomialNB(alpha=-0.5).fit(X,y)` → `InvalidParameterError("The 'alpha' parameter of MultinomialNB must be a float in the range [0.0, inf) … Got -0.5 instead.")`. ferrolearn `fn fit` has NO `alpha >= 0` guard: `with_alpha(-0.5)` → `clamp_alpha(-0.5, force_alpha=true)` returns `-0.5` unchanged (the `1e-10` floor only fires when `force_alpha=false`), so `fit` proceeds and computes `log((count - 0.5)/denom)` — garbage/NaN, no error. Pin (AC-2): `with_alpha(-0.5).fit(&X,&y)` SUCCEEDS in ferrolearn vs sklearn raises. **THE single-file fixable divergence** (an `alpha < 0` reject before/around `clamp_alpha` in `multinomial.rs` `fn fit`) — the critic should pin this FIRST. Distinct from the `force_alpha` floor (REQ-7): `>= 0` is a hard reject regardless of `force_alpha`. |
| REQ-6 (`sample_weight`) | NOT-STARTED | open prereq blocker **#901**. sklearn `fit(X, y, sample_weight=None)` (`:712`) weights the binarized `Y` (`Y *= sample_weight.T`, `:751`) so `feature_count_ = Y.T @ X` / `class_count_ = Y.sum(axis=0)` become weighted. ferrolearn's `impl Fit<Array2<F>, Array1<usize>>` has signature `fn fit(&self, x, y)` — NO `sample_weight` parameter on `fit` or `partial_fit`. Pin (AC-6): `MultinomialNB().fit(X,y,sample_weight=[1,2,1,1,1,3]).feature_count_` → `[[11,3,2],[1,7,22]]`, `class_count_` → `[4,5]`; ferrolearn cannot pass weights. |
| REQ-9 (fitted-attribute + PyO3 surface) | NOT-STARTED | open prereq blocker **#902**. sklearn exposes `feature_log_prob_`, `class_log_prior_`, `feature_count_`, `class_count_`, `classes_`, `n_features_in_`, and the `_BaseDiscreteNB` `coef_ = feature_log_prob_[1:]` / `intercept_ = class_log_prior_[1:]` properties. `FittedMultinomialNB` stores `log_theta`/`log_prior`/`feature_counts`/`class_counts` as PRIVATE fields with no accessor — only `classes()` (via `HasClasses`) is public; no `coef_`/`intercept_`. `_RsMultinomialNB` (`extras.rs`, the `py_classifier!` macro) exposes ONLY `new(alpha=1.0, fit_prior=true)` + `fit` + `predict` — NO `class_prior`/`force_alpha` kwargs, NO `predict_proba`/`predict_log_proba`/`predict_joint_log_proba`/`score`/`partial_fit` (which the library HAS), NO fitted-attr getters. Pin (AC-8): `hasattr(sklearn fitted, 'feature_log_prob_')`/`'coef_'` True; `ferrolearn.MultinomialNB` reaches only `fit`/`predict`. Also subsumes the negative-feature MESSAGE-parity sub-item (REQ-5: ferrolearn's `InvalidParameter` reason text differs from sklearn's "Negative values in data passed to MultinomialNB (input X)"). |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker **#903**. `multinomial.rs` imports `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}` (the wrong substrate); not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`multinomial.rs` follows the unfitted/fitted split (CLAUDE.md naming) for a
single estimator that delegates its entire prediction pipeline to the shared
`BaseNB<F>` trait (`base.rs`):

- `MultinomialNB<F>` (`alpha`, `class_prior: Option<Vec<F>>`, `fit_prior`,
  `force_alpha`) → `Fit<Array2<F>, Array1<usize>>` → `FittedMultinomialNB<F>`
  (`classes`, `log_prior`, `log_theta`, `feature_counts`, `class_counts`, plus
  `alpha`/`class_prior`/`fit_prior` for `partial_fit`).

Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2).

**Fit path (`fn fit`).** Validation rejects empty `X` (`InsufficientSamples`),
`n_samples != y.len()` (`ShapeMismatch`), and negative `X` (`InvalidParameter`,
REQ-5). It computes `alpha = clamp_alpha(self.alpha, force_alpha)` (the
`_check_alpha` floor, REQ-7) — but does NOT reject `alpha < 0` (REQ-2/#900, the
`>= 0` constraint sklearn enforces at `:530`). Per class it accumulates
`feature_counts` and `total_count`, then `log_theta[[ci,j]] = ((count + alpha) /
(total_count + alpha*n_features)).ln()` — the algebraic identity of
`_update_feature_log_prob` (`:885-892`, REQ-1) — and the empirical `log_prior[ci]
= ln(count_c / n)` (REQ-3). Priors resolve: explicit `class_prior` (LENGTH-only
check, then `p.ln()` — REQ-4, the MATCH) wins; else `fit_prior == false` → uniform
`(1/n_classes).ln()`; else the empirical value already filled. No `sample_weight`
(REQ-6).

**Prediction (delegated to `BaseNB`).** `joint_log_likelihood` computes
`log_prior[ci] + sum_j x*log_theta[ci,j]`, mirroring `X @ feature_log_prob_.T +
class_log_prior_` (`:896`, REQ-1). `predict` / `predict_proba` /
`predict_log_proba` / `predict_joint_log_proba` delegate to the `BaseNB` provided
methods (the `_BaseNB` pipeline; see `.design/bayes/base.md`). The pipeline is
exact and the VALUES match the oracle to ~1e-12 (REQ-1) — there is no
`epsilon_`-style shift here (unlike GaussianNB), because the smoothing closed
form is exact.

**`partial_fit` (`fn partial_fit`).** Accumulates `class_counts`/`feature_counts`
for each existing class, then recomputes `log_theta` from the accumulated
`feature_counts` (same smoothing) and the `log_prior` (empirical/uniform on
`fit_prior`; explicit `class_prior` sticky), so chunked `partial_fit` == whole
`fit` (REQ-8, oracle-confirmed). No `classes` argument (existing classes only —
new labels are dropped) and no `sample_weight` (REQ-6).

**Scoring.** `score` = `correct / n` (mean accuracy), the `ClassifierMixin.score`
analog.

**Consumer wiring.** The non-test production consumers:
- `ferrolearn-python` `RsMultinomialNB` / `_RsMultinomialNB` (`extras.rs`, the
  `py_classifier!` macro) — `new(alpha=1.0, fit_prior=true)` → `fit` (`model.fit(
  &x_nd, &y_nd)`) → `predict` (`fitted.predict`), registered in `lib.rs`
  (`m.add_class::<extras::RsMultinomialNB>()`) and surfaced as
  `ferrolearn.MultinomialNB` (`_extras.py`, `_RsMultinomialNB(alpha, fit_prior)`).
  The binding under-exposes (REQ-9), but the `fit`/`predict` path is a real
  non-test consumer of the library `feature_log_prob_`/`joint_log_likelihood`/
  `predict` (REQ-1/REQ-3).
- `impl PipelineEstimator<F> for MultinomialNB<F>` — `fit_pipeline` (float labels
  → `usize`) / `predict_pipeline` consume `fit`/`predict` in-crate.

**Missing fitted attributes vs sklearn:** `feature_log_prob_` / `class_log_prior_`
/ `feature_count_` / `class_count_` accessors, `coef_` / `intercept_` properties
(`_BaseDiscreteNB`), `n_features_in_`. ferrolearn exposes only `classes()` /
`n_classes()`.

**Invariants held vs sklearn:** `feature_log_prob_` smoothing VALUE (AC-1); the
full predict pipeline VALUE to ~1e-12 (AC-1); empirical/uniform `class_log_prior_`
(REQ-3); the `class_prior` LENGTH-only accept/reject decisions (AC-4, the MATCH);
the negative-feature reject (AC-5); the `force_alpha` floor + `fit_prior` toggle
(AC-3); `partial_fit` == `fit` (AC-7); `classes_` ordering; `predict_proba` rows
sum to 1.

**Invariants NOT held vs sklearn:** `alpha >= 0` reject (REQ-2/#900 — the key
single-file divergence); `sample_weight` (REQ-6); the fitted-attribute + PyO3
surface (REQ-9, including the negative-feature message text); the ferray substrate
(REQ-10).

## Verification

Library crate (green at baseline `963c37f6` for the existing contract):
```
cargo test -p ferrolearn-bayes --lib multinomial
cargo clippy -p ferrolearn-bayes --all-targets -- -D warnings
cargo fmt --all --check
```
The 15 in-tree `#[test]`s (`test_multinomial_nb_fit_predict`,
`test_multinomial_nb_predict_proba_sums_to_one`,
`test_multinomial_nb_has_classes`, `test_multinomial_nb_alpha_smoothing_effect`,
`test_multinomial_nb_negative_features_error`, `test_multinomial_nb_single_class`,
`test_multinomial_nb_partial_fit`, `test_multinomial_nb_class_prior`,
`test_multinomial_nb_class_prior_wrong_length`, …) pin ferrolearn's current
behavior. **None compares against the live sklearn oracle and none exercises
`alpha < 0`**, so they stay green despite the #900 divergence; the SHIPPED REQs
(REQ-1 smoothing+predict VALUE, REQ-3 prior, REQ-4 class_prior MATCH, REQ-5
reject, REQ-7 floor/toggle, REQ-8 partial_fit) value-match the oracle, the rest
are NOT-STARTED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin the deterministic single-file one
FIRST**: REQ-2 (`alpha >= 0` — add the reject in `multinomial.rs` `fn fit`):
```
# REQ-2 (#900) alpha < 0 — sklearn rejects, ferrolearn accepts (THE key fix)
python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; X=np.array([[3.,1.,0.],[2.,0.,1.],[4.,2.,0.],[0.,1.,4.],[1.,0.,3.],[0.,2.,5.]]); y=np.array([0,0,0,1,1,1]); 
try: MultinomialNB(alpha=-0.5).fit(X,y)
except Exception as e: print(type(e).__name__, '::', e)"  # InvalidParameterError :: The 'alpha' parameter of MultinomialNB must be a float in the range [0.0, inf) or an array-like. Got -0.5 instead.   (ferro: with_alpha(-0.5).fit succeeds)
# REQ-1 (present) feature_log_prob_ + predict VALUE
python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; X=np.array([[3.,1.,0.],[2.,0.,1.],[4.,2.,0.],[0.,1.,4.],[1.,0.,3.],[0.,2.,5.]]); y=np.array([0,0,0,1,1,1]); m=MultinomialNB().fit(X,y); q=np.array([[2.,1.,1.],[0.,1.,3.]]); print(m.feature_log_prob_.tolist()); print(m.predict_proba(q).tolist()); print(m.predict(q).tolist())"  # flp [[-0.470..,-1.386..,-2.079..],[-2.251..,-1.558..,-0.379..]]; pp [[0.8844..,0.1156..],[0.00719..,0.9928..]]; pred [0,1]
# REQ-4 (present, MATCH) class_prior length-only (sum 0.8 accepted)
python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; X=np.array([[3.,1.,0.],[2.,0.,1.],[4.,2.,0.],[0.,1.,4.],[1.,0.,3.],[0.,2.,5.]]); y=np.array([0,0,0,1,1,1]); print(MultinomialNB(class_prior=[0.5,0.3]).fit(X,y).class_log_prior_.tolist())"  # [-0.6931471805599453, -1.2039728043259361]  (= log([0.5,0.3]); NO sum check — ferro MATCHES)
# REQ-6 sample_weight
python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; X=np.array([[3.,1.,0.],[2.,0.,1.],[4.,2.,0.],[0.,1.,4.],[1.,0.,3.],[0.,2.,5.]]); y=np.array([0,0,0,1,1,1]); m=MultinomialNB().fit(X,y,sample_weight=np.array([1.,2.,1.,1.,1.,3.])); print(m.feature_count_.tolist(), m.class_count_.tolist())"  # [[11,3,2],[1,7,22]] [4,5]
# REQ-8 (present) partial_fit == fit
python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; X=np.array([[3.,1.,0.],[2.,0.,1.],[4.,2.,0.],[0.,1.,4.],[1.,0.,3.],[0.,2.,5.]]); y=np.array([0,0,0,1,1,1]); m=MultinomialNB(); m.partial_fit(X[:4],y[:4],classes=[0,1]); m.partial_fit(X[4:],y[4:]); f=MultinomialNB().fit(X,y); print(np.allclose(m.feature_log_prob_, f.feature_log_prob_))"  # True
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-bayes/tests/divergence_multinomial.rs`, asserting the live-sklearn
expected values above and FAILING against current `multinomial.rs` (REQ-2 the
`alpha < 0` reject is the cleanest). REQ-1/REQ-3/REQ-4/REQ-5/REQ-7/REQ-8 already
match and should be guarded by non-regression pins.

ferrolearn-python (REQ-9 binding parity, after #902 lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_naive_bayes.py -q
```
asserting `ferrolearn.MultinomialNB` exposes `feature_log_prob_` /
`class_log_prior_` / `feature_count_` / `class_count_` / `coef_` / `intercept_` /
`predict_proba` / `predict_log_proba` / `score` / `partial_fit` and `class_prior`
/ `force_alpha` kwargs, matching `sklearn.naive_bayes.MultinomialNB` on the AC
fixtures.

## Blockers to open

(Director creates the real issues; the numbers are SUGGESTIONS continuing the
bayes layer past gaussian #897. #899 is this doc's crosslink tracking issue.)

- **#900** — REQ-2 (`alpha >= 0` validation): add an `alpha < 0` reject to
  `multinomial.rs` `fn fit` (mirroring `_parameter_constraints` `alpha:
  Interval(Real, 0, None, closed="left")`, `naive_bayes.py:530`) before/around
  `clamp_alpha`. **The cleanest single-file deterministic fix** — the critic
  should pin this FIRST. (Distinct from the `force_alpha` floor, which only fires
  for `alpha < 1e-10` when `force_alpha=false`.)
- **#901** — REQ-6 (`sample_weight`): add weighted `feature_count_`/
  `class_count_` (`Y *= sample_weight.T` before `_count`, `:751`/`:879-883`);
  needs a `sample_weight` parameter on `fit`/`partial_fit` (Fit-trait shape,
  R-DEV-1).
- **#902** — REQ-9 (fitted-attribute + PyO3 surface): expose `feature_log_prob_`/
  `class_log_prior_`/`feature_count_`/`class_count_`/`coef_`/`intercept_`
  accessors on `FittedMultinomialNB`, and add `class_prior`/`force_alpha` kwargs +
  `predict_proba`/`predict_log_proba`/`score`/`partial_fit` + those getters to
  `_RsMultinomialNB` (`ferrolearn-python/src/extras.rs`). Also align the
  negative-feature error message with sklearn's "Negative values in data passed
  to MultinomialNB (input X)" (REQ-5 sub-item).
- **#903** — REQ-10 (ferray substrate): migrate `multinomial.rs` off
  `ndarray`/`num-traits` to `ferray-core` (R-SUBSTRATE).
