# Bernoulli Naive Bayes (sklearn.naive_bayes.BernoulliNB)

<!--
tier: 3-component
status: draft
baseline-commit: e0940350
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/naive_bayes.py   # BernoulliNB(_BaseDiscreteNB); __init__(*, alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True, class_prior=None) (:1159-1174); _parameter_constraints binarize=[None, Interval(Real,0,None,closed="left")] (:1156), shared alpha=Interval(Real,0,None,closed="left") (:530); _check_X_y / _check_X binarize(X, threshold=self.binarize) only if binarize is not None (:1176-1187); _count feature_count_ += Y.T@X, class_count_ += Y.sum(axis=0) (:1189-1192); _update_feature_log_prob smoothed_fc=fc+alpha, smoothed_cc=cc+alpha*2, feature_log_prob_=log(smoothed_fc)-log(smoothed_cc.reshape(-1,1)) (:1194-1201); _joint_log_likelihood neg_prob=log(1-exp(flp)); jll=X@(flp-neg_prob).T; jll+=class_log_prior_+neg_prob.sum(axis=1) (:1203-1219); _update_class_log_prior LENGTH-only (:580-602); _check_alpha floor 1e-10 unless force_alpha (:604-626); shared partial_fit (:629-708); shared fit (:712-762)
ferrolearn-module: ferrolearn-bayes/src/bernoulli.rs
parity-ops: BernoulliNB (.__init__, .fit, .partial_fit, .predict, .predict_proba, .predict_log_proba, .predict_joint_log_proba, .score)
crosslink-issue: 905
-->

## Summary

`ferrolearn-bayes/src/bernoulli.rs` mirrors scikit-learn's `BernoulliNB`
(`sklearn/naive_bayes.py`, `class BernoulliNB(_BaseDiscreteNB)` `:1052-1219`) —
the binary/boolean naive-Bayes classifier whose per-feature log-likelihood is the
Bernoulli term `x_j * log(p_cj) + (1 - x_j) * log(1 - p_cj)` with the smoothed
present-probability `p_cj = (N_cj + alpha) / (N_c + 2*alpha)`. It exposes the
unfitted `BernoulliNB<F>` (`alpha=1.0`, `binarize: Option<F> = None`,
`class_prior: Option<Vec<F>>`, `fit_prior=true`, `force_alpha=true`), the fitted
`FittedBernoulliNB<F>` (per-class `feature_counts` / `log_prob` / `log_neg_prob` /
`log_prior` / `class_counts` plus `binarize`/`alpha`/`class_prior`/`fit_prior`
carried for `partial_fit`), and delegates the entire prediction pipeline to the
shared `BaseNB<F>` trait (`base.rs`, the `_BaseNB` analog — see
`.design/bayes/base.md`). It is re-exported at the crate root
(`ferrolearn-bayes/src/lib.rs`) and **bound into `ferrolearn-python`**
(`RsBernoulliNB` / `_RsBernoulliNB` in `ferrolearn-python/src/extras.rs`, surfaced
as `ferrolearn.BernoulliNB` via `_extras.py`) — a genuine non-test production
consumer (the binding `fit`/`predict` + the in-crate pipeline integration).

Under honest underclaim (R-HONEST-3), the behaviors that are genuinely present
**and value-match the live sklearn 1.5.2 oracle** (run from `/tmp`) are:

- **`feature_log_prob_` smoothing VALUE + the Bernoulli `_joint_log_likelihood` /
  `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba`
  VALUE on ALREADY-BINARY data** — verified on the 0/1 fixture `X =
  [[1,1,0],[1,0,0],[1,1,0],[0,0,1],[0,1,1],[0,0,1]]`, `y = [0,0,0,1,1,1]`, query
  `q = [[1,0,0],[0,1,1]]`. On 0/1 data ferrolearn's `binarize=None` and sklearn's
  `binarize=0.0` **coincide** (`1 > 0 → 1`, `0 > 0 → 0` is the identity on
  `{0,1}`; the oracle confirms `BernoulliNB()` and `BernoulliNB(binarize=None)`
  give identical `feature_log_prob_`). This isolates the jll/predict VALUE
  contract from the binarize-default issue; ferrolearn matches to ~1e-12.
- **binarize threshold-application VALUE when `binarize` is SET** — `v > threshold
  → 1 else 0` (`binarize_array`); the oracle confirms sklearn's `binarize(X,
  threshold=0.5)` is strictly-greater (`0.5 → 0`, `0.6 → 1`), matching
  ferrolearn's `v > threshold` (NOT `>=`).
- **`class_log_prior_` empirical / uniform / explicit VALUE + `class_prior`
  LENGTH-only validation** — `log(class_count_) - log(class_count_.sum())`
  (fit_prior, `:600`), `-log(n_classes)` (fit_prior=False, `:602`),
  `log(class_prior)` (explicit, `:591`); the explicit path checks ONLY length
  (`:589-590`), NO sum-to-1 / non-negativity for discrete NB. ferrolearn matches
  all three paths and the length-only check — a **MATCH**, not a divergence
  (`class_prior=[0.5,0.3]`, sum 0.8, fits on both sides).
- **`force_alpha` / `_check_alpha` floor + `fit_prior` toggle** — the
  `base::check_alpha` / `clamp_alpha` floor (`1e-10` unless `force_alpha`) and the
  empirical-vs-uniform prior selection.
- **`partial_fit` VALUE on same-classes data** — incremental count accumulation
  then resmooth; chunked `partial_fit` over existing classes equals `fit` on the
  concatenation (the smoothing is reapplied to accumulated counts each call).
- **`score`** — mean accuracy (`ClassifierMixin.score` analog).

The behaviors that **diverge** from the `BernoulliNB` contract (each pinned to a
NOT-STARTED REQ with a concrete prereq blocker — the director creates the real
issues; the numbers below are SUGGESTIONS continuing the bayes layer past
multinomial #899-903):

1. **`binarize` DEFAULT `0.0` vs `None` (R-DEV-2 — THE key fixable divergence).**
   sklearn `__init__` declares `binarize=0.0` (`:1164`) — by DEFAULT every `fit`
   and `predict` binarizes `X` at threshold `0.0` (`_check_X_y` / `_check_X`,
   `:1179-1186`: `if self.binarize is not None: X = binarize(X, threshold=...)`).
   ferrolearn `BernoulliNB::new()` defaults `binarize = None` (`pub fn new`), so on
   NON-pre-binarized data it uses raw values as-is — `feature_counts` become raw
   count sums, not binary occurrence counts, and `predict` labels can differ. On
   the non-binary fixture `X = [[2,0,1],[0,3,0],[1,1,2],[0,0,4]]`, `y=[0,0,1,1]`:
   sklearn `feature_count_ = [[1,1,1],[1,1,2]]` (occurrence indicators),
   `predict(X) = [1,0,1,1]`; ferrolearn with `binarize=None` would compute raw
   feature sums `[[2,3,1],[1,1,6]]`. **The single-file-fixable divergence in
   `bernoulli.rs`** (change the `new()` default to `binarize = Some(0.0)` so it
   mirrors sklearn `:1164`). NOTE: the `ferrolearn-python` binding masks this —
   `_RsBernoulliNB` defaults `binarize=0.0` and always calls `.with_binarize(...)`
   — but the pure-Rust `BernoulliNB::new()` API still diverges.
2. **`alpha >= 0` validation (R-DEV-2 — fixable, same class as multinomial #900).**
   sklearn's shared `_parameter_constraints` declares `alpha: [Interval(Real, 0,
   None, closed="left"), "array-like"]` (`:530`) → `alpha >= 0` is a HARD reject at
   `fit` (`_validate_params`); `BernoulliNB(alpha=-0.5).fit(X,y)` raises
   `InvalidParameterError("The 'alpha' parameter of BernoulliNB must be a float in
   the range [0.0, inf) or an array-like. Got -0.5 instead.")`. ferrolearn `fn fit`
   has NO `alpha >= 0` guard: `with_alpha(-0.5)` → `clamp_alpha(-0.5,
   force_alpha=true)` returns `-0.5` unchanged (the `1e-10` floor only fires when
   `force_alpha=false`), so `fit` proceeds on negative-smoothed garbage.
3. **`sample_weight` (R-DEV-1).** sklearn `fit(X, y, sample_weight=None)` (`:712`)
   weights the binarized `Y` so `feature_count_ = Y.T @ X` / `class_count_ =
   Y.sum(axis=0)` become weighted counts. ferrolearn's `Fit` trait is `fn fit(&self,
   x, y)` — NO `sample_weight` parameter on `fit` or `partial_fit`.
4. **fitted-attribute + PyO3 surface (R-DEV-3 / R-DEFER-1/3).** sklearn exposes
   `feature_log_prob_`, `class_log_prior_`, `feature_count_`, `class_count_`,
   `classes_`, `n_features_in_` (`:1088-1117`). ferrolearn `FittedBernoulliNB`
   exposes ONLY `classes()` (via `HasClasses`); `log_prob`/`log_neg_prob`/
   `log_prior`/`feature_counts`/`class_counts` are private fields with no accessor.
   `_RsBernoulliNB` (`extras.rs`, the `py_classifier!` macro) exposes ONLY
   `new(alpha=1.0, fit_prior=true, binarize=0.0)` + `fit` + `predict` — NO
   `class_prior`/`force_alpha` kwargs, NO `predict_proba`/`predict_log_proba`/
   `predict_joint_log_proba`/`score`/`partial_fit` (which the library HAS), NO
   fitted-attr getters. (NOTE: `BernoulliNB` does NOT expose `coef_`/`intercept_`
   — the deprecated `_BaseDiscreteNB.coef_`/`intercept_` are gone by 1.5.2;
   `hasattr(m, 'coef_')` is `False`.)
5. **`partial_fit` unseen-label gap (R-DEV-1, sub-item of #908).** sklearn's shared
   `partial_fit(X, y, classes=None, ...)` (`:629-708`) takes the full `classes`
   list on the first call and binarizes against it, so a label unseen in a later
   chunk is still represented. ferrolearn `FittedBernoulliNB::partial_fit` iterates
   only the existing `self.classes` — a NEW label appearing in a later chunk is
   silently DROPPED (no `classes` argument). On same-classes data `partial_fit ==
   fit` (SHIPPED, REQ-7); the new-label path diverges.
6. **ferray substrate (R-SUBSTRATE).** `bernoulli.rs` imports `ndarray::{Array1,
   Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}`, not `ferray-core`.

`BernoulliNB` / `FittedBernoulliNB` are existing pub APIs (grandfathered per
S5/R-DEFER-1); their non-test production consumers are the `ferrolearn-python`
binding (`RsBernoulliNB` `fit`/`predict`) and the in-crate pipeline integration
(`impl PipelineEstimator for BernoulliNB`).

## Algorithm (sklearn — the contract)

### Construction (`naive_bayes.py:1159-1174`)

`BernoulliNB(*, alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True,
class_prior=None)` — all keyword-only. `_parameter_constraints` (`:1154-1157`,
merging `_BaseDiscreteNB`'s `:529-534`): `alpha: [Interval(Real, 0, None,
closed="left"), "array-like"]` (**>= 0**, `:530`); `binarize: [None,
Interval(Real, 0, None, closed="left")]` (**default `0.0`**, must be `None` or
`>= 0`, `:1156`); `fit_prior: ["boolean"]`; `class_prior: ["array-like", None]`;
`force_alpha: ["boolean"]`.

### Binarization (`_check_X_y` `:1183-1187` / `_check_X` `:1176-1181`)

`if self.binarize is not None: X = binarize(X, threshold=self.binarize)` — the
`sklearn.preprocessing.binarize` is strictly-greater (`X > threshold → 1`, else
`0`). Applied in BOTH fit (`_check_X_y`) and predict (`_check_X`). With the
default `binarize=0.0`, every value `> 0` becomes `1`, so counts/continuous inputs
collapse to occurrence indicators. With `binarize=None`, `X` is presumed already
binary and used as-is.

### Fit (`_BaseDiscreteNB.fit` `:712-762`)

`fit(X, y, sample_weight=None)`: `_check_X_y` (binarizes when `binarize is not
None`); binarize `y` → one-hot `Y`; if `sample_weight` given, `Y *=
sample_weight.T`; `_init_counters` zeroes `class_count_` / `feature_count_`;
`_count(X, Y)` accumulates; `alpha = self._check_alpha()`;
`_update_feature_log_prob(alpha)`; `_update_class_log_prior(class_prior)`.

- **`_count`** (`:1189-1192`): `feature_count_ += safe_sparse_dot(Y.T, X)` (per-
  class feature occurrence sums over the BINARIZED `X`); `class_count_ +=
  Y.sum(axis=0)` (per-class sample counts).
- **`_check_alpha`** (`:604-626`): `alpha_lower_bound = 1e-10` (`:618`); if
  `alpha_min < alpha_lower_bound and not self.force_alpha` (`:619`) warn + return
  `np.maximum(alpha, alpha_lower_bound)`; else return alpha unchanged. (The `>= 0`
  HARD constraint is enforced earlier by `_validate_params` against
  `_parameter_constraints` `:530`, NOT inside `_check_alpha`.)
- **`_update_feature_log_prob`** (`:1194-1201`): `smoothed_fc = feature_count_ +
  alpha`; `smoothed_cc = class_count_ + alpha * 2`; `feature_log_prob_ =
  np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))`. Algebraically
  `log((N_cj + alpha) / (N_c + 2*alpha))` = `log(p_cj)`.
- **`_update_class_log_prior`** (`:580-602`): if `class_prior is not None` →
  `if len(class_prior) != n_classes: raise ValueError("Number of priors must match
  number of classes.")` (`:589-590`); `class_log_prior_ = np.log(class_prior)`
  (`:591`) — **LENGTH-ONLY check, NO sum-to-1, NO non-negativity** (unlike
  GaussianNB). elif `fit_prior` → `log(class_count_) - log(class_count_.sum())`
  (`:600`). else → `np.full(n_classes, -np.log(n_classes))` (uniform, `:602`).

### `_joint_log_likelihood` (`:1203-1219`)

`neg_prob = np.log(1 - np.exp(self.feature_log_prob_))` (= `log(1 - p_cj)`); `jll =
safe_sparse_dot(X, (feature_log_prob_ - neg_prob).T)`; `jll += class_log_prior_ +
neg_prob.sum(axis=1)`. This is the vectorized form of `sum_j [x_j * log(p_cj) +
(1 - x_j) * log(1 - p_cj)] + class_log_prior_`: the `x_j*log(p) + log(1-p) -
x_j*log(1-p)` regrouping gives `X @ (log_p - log_neg).T + sum_j log_neg +
class_log_prior_`, shape `(n_samples, n_classes)`. The shared `_BaseNB` pipeline
(`predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba`)
consumes this (`.design/bayes/base.md`).

### `partial_fit` (`_BaseDiscreteNB.partial_fit` `:629-708`)

`partial_fit(X, y, classes=None, sample_weight=None)`. First call initializes the
counters (`classes` required); each call binarizes `y` (against the FULL `classes`
list), optionally weights `Y`, `_count(X, Y)` accumulates, then recomputes `alpha
= _check_alpha()`, `_update_feature_log_prob(alpha)`,
`_update_class_log_prior(class_prior)`. Because the smoothing is reapplied to the
accumulated counts each call, `partial_fit` over chunks equals `fit` on the
concatenation.

### Edge cases (live oracle, sklearn 1.5.2, run from /tmp)

- **binarize DEFAULT 0.0 on NON-binary data** (`X = [[2,0,1],[0,3,0],[1,1,2],
  [0,0,4]]`, `y=[0,0,1,1]`): `BernoulliNB().binarize == 0.0`; `feature_count_ =
  [[1,1,1],[1,1,2]]` (occurrence indicators after binarizing 2→1, 3→1, 4→1);
  `class_count_ = [2,2]`; `feature_log_prob_ = [[-0.6931..,-0.6931..,-0.6931..],
  [-0.6931..,-0.6931..,-0.2877..]]`; `predict(X) = [1,0,1,1]`. ferrolearn
  `BernoulliNB::new()` (`binarize=None`) would instead use raw sums `[[2,3,1],
  [1,1,6]]` — divergent counts and potentially divergent labels.
- **binarize=0.0 vs None on ALREADY-BINARY data**: `np.allclose(BernoulliNB().fit(
  Xbin,y).feature_log_prob_, BernoulliNB(binarize=None).fit(Xbin,y).
  feature_log_prob_) == True` — they COINCIDE (so the value contract below is
  testable independent of the binarize default).
- **feature_log_prob_ + predict VALUE on binary data** (`Xbin = [[1,1,0],[1,0,0],
  [1,1,0],[0,0,1],[0,1,1],[0,0,1]]`, `y=[0,0,0,1,1,1]`, `q=[[1,0,0],[0,1,1]]`,
  `alpha=1`): `feature_log_prob_ = [[-0.2231435513142097, -0.5108256237659905,
  -1.6094379124341003], [-1.6094379124341003, -0.916290731874155,
  -0.2231435513142097]]`; `class_log_prior_ = [-0.6931471805599452,
  -0.6931471805599452]`; `predict_joint_log_proba(q) = [[-2.05572501506252,
  -4.422848629194137], [-4.422848629194137, -2.0557250150625195]]`;
  `predict_log_proba(q) = [[-0.0896121586896872, -2.456735772821304],
  [-2.4567357728213044, -0.08961215868968697]]`; `predict_proba(q) =
  [[0.9142857142857143, 0.08571428571428572], [0.08571428571428567,
  0.9142857142857145]]`; `predict(q) = [0, 1]`. ferrolearn matches to ~1e-12.
- **binarize threshold VALUE**: `binarize([[0.5,0.6],[0.4,0.5],[0.9,0.1]],
  threshold=0.5) = [[0,1],[0,0],[1,0]]` — strictly-greater (`0.5 → 0`). ferrolearn
  `binarize_array` is `v > threshold → 1`, matching exactly (NOT `>=`).
- **alpha < 0**: `BernoulliNB(alpha=-0.5).fit(X,y)` → `InvalidParameterError: The
  'alpha' parameter of BernoulliNB must be a float in the range [0.0, inf) or an
  array-like. Got -0.5 instead.` (raised at `fit` by `_validate_params`).
- **class_prior length-only**: `BernoulliNB(class_prior=[0.5,0.3]).fit(X,y)`
  (sum 0.8) ACCEPTED → `class_log_prior_ = [-0.6931471805599453,
  -1.2039728043259361]` (= `log([0.5,0.3])`), NO sum/non-neg error. Wrong length
  `class_prior=[0.5]` → `ValueError: Number of priors must match number of
  classes.` ferrolearn MATCHES all three (length-only).
- **sample_weight**: `BernoulliNB().fit(Xbin, y, sample_weight=[1,2,1,1,1,3])` →
  `feature_count_ = [[4,2,0],[0,1,5]]`, `class_count_ = [4,5]` (weighted).
  ferrolearn has no `sample_weight` parameter.
- **negative features (binarize=None)**: `BernoulliNB(binarize=None).fit(X_neg, y)`
  → ACCEPTED (no `check_non_negative` for the Bernoulli model — UNLIKE
  MultinomialNB), but emits a `RuntimeWarning: divide by zero / invalid value
  encountered in log` and `feature_count_` keeps the negatives `[[-2,2,0],[0,-1,
  3]]`. ferrolearn `fn fit` also does NOT reject negatives — a **MATCH** on the
  accept decision (neither guards non-negativity). (With the default
  `binarize=0.0`, negatives are binarized to `0`, so this only surfaces with
  `binarize=None`.)
- **coef_/intercept_**: `hasattr(BernoulliNB().fit(Xbin,y), 'coef_') == False` —
  BernoulliNB exposes NO `coef_`/`intercept_` (the deprecated `_BaseDiscreteNB`
  properties are gone by 1.5.2). So the fitted-attr surface is `feature_log_prob_`
  / `class_log_prior_` / `feature_count_` / `class_count_` / `classes_` /
  `n_features_in_` only.

## ferrolearn (what exists)

All in `ferrolearn-bayes/src/bernoulli.rs`, generic over `F: Float + Send + Sync +
'static`; `ndarray` substrate. Every public method returns `Result<_, FerroError>`
(no panics in library code, R-CODE-2).

- **`pub struct BernoulliNB<F> { pub alpha: F, pub binarize: Option<F>, pub
  class_prior: Option<Vec<F>>, pub fit_prior: bool, pub force_alpha: bool }`** —
  `pub fn new` sets `alpha = 1.0`, **`binarize = None`** (DIVERGENT default vs
  sklearn `0.0` — REQ-3), `class_prior = None`, `fit_prior = true`, `force_alpha =
  true`; builder setters `with_alpha` / `with_binarize` / `with_class_prior` /
  `with_fit_prior` / `with_force_alpha`; `impl Default → new()`.
- **`fn binarize_array<F>(x, threshold)`** — `x.mapv(|v| if v > threshold {
  one } else { zero })`, the `sklearn.preprocessing.binarize` strictly-greater
  analog (REQ-2).
- **`pub struct FittedBernoulliNB<F>`** — private fields `classes: Vec<usize>`,
  `log_prior: Array1<F>` (the `class_log_prior_` analog), `log_prob: Array2<F>`
  (the `feature_log_prob_` = `log(p_cj)` analog), `log_neg_prob: Array2<F>` (the
  `neg_prob` = `log(1 - p_cj)` analog), `binarize: Option<F>`, `feature_counts:
  Array2<F>` (the `feature_count_` analog), `class_counts: Vec<usize>` (the
  `class_count_` analog), plus `alpha` / `class_prior` / `fit_prior` carried for
  `partial_fit`. **No public accessor** for any of these (only `classes()` via
  `HasClasses`).
- **`impl Fit<Array2<F>, Array1<usize>> for BernoulliNB<F>` / `fn fit`** — rejects
  `n_samples == 0` (`InsufficientSamples`), `n_samples != y.len()`
  (`ShapeMismatch`). Binarizes `X` ONLY when `self.binarize` is `Some` (the
  default-`None` path skips binarization — REQ-3); collects sorted-deduped
  `classes`; `alpha = crate::clamp_alpha(self.alpha, self.force_alpha)` (the
  `_check_alpha` floor — REQ-5). Per class accumulates `feature_counts[[ci,j]]` and
  the present-probability `p = (fc + alpha) / (n_c + 2*alpha)` → `log_prob =
  p.ln()`, `log_neg_prob = (1 - p).ln()` (the smoothing — REQ-1) and the empirical
  `log_prior[ci] = (n_c / n).ln()` (REQ-4). Resolves priors: explicit `class_prior`
  (LENGTH-only check then `log_prior[ci] = p.ln()` — REQ-4) wins; else `fit_prior
  == false` → uniform `(1/n_classes).ln()`; else empirical (already filled). **No
  `alpha >= 0` guard** (REQ-6), **no `sample_weight`** (REQ-8), **no negative-
  feature reject** (MATCH — sklearn also accepts).
- **`FittedBernoulliNB::partial_fit(&mut self, x, y)`** — binarizes when `binarize`
  is `Some`; accumulates `class_counts` + `feature_counts` for each EXISTING class,
  then recomputes `log_prob`/`log_neg_prob` from accumulated `feature_counts` (same
  smoothing) and the `log_prior` (empirical when `class_prior` is `None` and
  `fit_prior`; uniform otherwise; explicit `class_prior` sticky). Rejects feature-
  count mismatch. **No `sample_weight`, no `classes` argument** (existing classes
  only — a new later-chunk label is dropped, REQ-7 sub-item).
- **`impl BaseNB<F> for FittedBernoulliNB<F>` / `fn joint_log_likelihood`** —
  binarizes when `binarize` is `Some`, then `score = log_prior[ci] + sum_j
  [x_j*log_prob[ci,j] + (1-x_j)*log_neg_prob[ci,j]]`, shape `(n_samples,
  n_classes)` — the row-wise form of sklearn's vectorized `X @ (flp-neg).T +
  class_log_prior_ + neg.sum` (`:1216-1217`, REQ-1); `fn nb_classes` returns
  `&self.classes`.
- **`pub fn predict_proba` / `pub fn predict_log_proba` / `pub fn
  predict_joint_log_proba`** — delegate to `BaseNB::nb_predict_proba` /
  `nb_predict_log_proba` / `nb_predict_joint_log_proba` (REQ-1 pipeline).
- **`impl Predict for FittedBernoulliNB<F>` / `fn predict`** — delegates to
  `BaseNB::nb_predict` (`classes_[argmax(jll)]`, first-max tie-break).
- **`pub fn score(&self, x, y)`** — mean accuracy (`correct / n`), the
  `ClassifierMixin.score` analog.
- **`impl HasClasses for FittedBernoulliNB<F>`** — `classes()` / `n_classes()`.
- **Pipeline**: `impl PipelineEstimator<F> for BernoulliNB<F>` (`fn fit_pipeline`,
  maps float labels → `usize`) + `FittedBernoulliNBPipeline` (`fn
  predict_pipeline`).

**Consumers (non-test).** Crate re-export (`ferrolearn-bayes/src/lib.rs`, `pub use
bernoulli::{BernoulliNB, FittedBernoulliNB}`) plus:
- **`ferrolearn-python`** — `RsBernoulliNB` / `_RsBernoulliNB`
  (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro):
  `new(alpha=1.0, fit_prior=true, binarize=0.0)` → `BernoulliNB::<f64>::new().
  with_alpha(alpha).with_fit_prior(fit_prior).with_binarize(binarize)`, `fit`
  (`model.fit(&x_nd, &y_nd)`), `predict` (`fitted.predict`); registered in
  `ferrolearn-python/src/lib.rs` (`m.add_class::<extras::RsBernoulliNB>()`) and
  surfaced as `ferrolearn.BernoulliNB` (`ferrolearn-python/python/ferrolearn/
  _extras.py`, `class BernoulliNB(_ClassifierWrapper)` →
  `_RsBernoulliNB(alpha, fit_prior, binarize)`). The binding under-exposes (REQ-4),
  but the `fit`/`predict` path is a real non-test consumer of the library
  `feature_log_prob_`/`joint_log_likelihood`/`predict` (REQ-1). NOTE: because the
  binding ALWAYS calls `.with_binarize(binarize)` with default `0.0`, the
  Python-facing default matches sklearn — the binarize-default divergence (REQ-3)
  surfaces only on the pure-Rust `BernoulliNB::new()` API.
- **Pipeline** (`impl PipelineEstimator`) consumes `fit` / `predict` in-crate.

## Requirements

- REQ-1: **`feature_log_prob_` smoothing VALUE + Bernoulli `_joint_log_likelihood`
  / `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba`
  VALUE (R-DEV-1/3).** Mirror `_update_feature_log_prob` (`:1194-1201`,
  `smoothed_fc=fc+alpha`, `smoothed_cc=cc+alpha*2`, `feature_log_prob_=log(
  smoothed_fc)-log(smoothed_cc.reshape(-1,1))`) and `_joint_log_likelihood`
  (`:1203-1219`, `neg_prob=log(1-exp(flp))`; `jll=X@(flp-neg).T +
  class_log_prior_ + neg.sum`) feeding the `_BaseNB` pipeline. ferrolearn `fn fit`
  computes `p=(fc+alpha)/(n_c+2*alpha)`, `log_prob=ln(p)`, `log_neg_prob=ln(1-p)`,
  and `fn joint_log_likelihood` computes `log_prior + sum_j [x*log_prob +
  (1-x)*log_neg_prob]`; the delegated `predict_*` value-match the oracle to ~1e-12
  on ALREADY-BINARY data (isolating this from REQ-3).
- REQ-2: **binarize threshold-application VALUE (R-DEV-1).** Mirror `binarize(X,
  threshold)` strictly-greater (`X > threshold → 1`) applied in `_check_X_y` /
  `_check_X` when `binarize is not None` (`:1179-1186`). ferrolearn `binarize_array`
  is `v > threshold → 1 else 0`, applied in `fit`/`partial_fit`/`joint_log_
  likelihood` when `binarize` is `Some`; matches sklearn's `binarize(X, 0.5)` (NOT
  `>=`).
- REQ-3: **`binarize` DEFAULT `0.0` vs `None` (R-DEV-2 — THE key fixable
  divergence).** Mirror `__init__(..., binarize=0.0, ...)` (`:1164`): the default
  must binarize at `0.0`. ferrolearn `pub fn new` defaults `binarize = None`, so
  on non-binary `X` it skips binarization and computes raw count sums (not binary
  occurrence counts), diverging in `feature_count_` and potentially `predict`.
- REQ-4: **`class_log_prior_` empirical / uniform / explicit VALUE + `class_prior`
  LENGTH-only validation (R-DEV-1/2 — MATCH).** Mirror the three prior paths
  (`log(class_count_)-log(class_count_.sum())` `:600`, `-log(n_classes)` `:602`,
  `log(class_prior)` `:591`) and the LENGTH-only check (`:589-590`, NO sum/non-neg
  for discrete NB). ferrolearn `fn fit` sets all three and checks ONLY length —
  a deliberate MATCH (sum-0.8 prior fits on both sides; wrong-length errors on both,
  type differs — `InvalidParameter` vs `ValueError`).
- REQ-5: **`force_alpha` / `_check_alpha` floor + `fit_prior` toggle (R-DEV-1).**
  Mirror `_check_alpha` (`:604-626`, floor `1e-10` unless `force_alpha`) and the
  `fit_prior` empirical/uniform selection. ferrolearn `fn fit` calls
  `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`) and
  honors `fit_prior`.
- REQ-6: **`alpha >= 0` validation (R-DEV-2 — fixable).** Mirror `_parameter_
  constraints` `alpha: Interval(Real, 0, None, closed="left")` (`:530`): reject
  `alpha < 0` at `fit`. ferrolearn `fn fit` has NO `alpha >= 0` guard —
  `with_alpha(-0.5)` passes through `clamp_alpha` unchanged (force_alpha default)
  and fits on negative-smoothed garbage.
- REQ-7: **`partial_fit` VALUE on same-classes data (R-DEV-1; unseen-label gap).**
  Mirror the shared `partial_fit` (`:629-708`): accumulate counts then recompute
  `feature_log_prob_` / `class_log_prior_`, so chunked `partial_fit` == whole `fit`.
  ferrolearn `FittedBernoulliNB::partial_fit` accumulates `feature_counts` then
  resmooths; same-classes chunks reproduce the whole `fit`. **Sub-gap:** no
  `classes` argument — a NEW label in a later chunk is silently dropped (sklearn
  binarizes against the full `classes` list passed on the first call).
- REQ-8: **`sample_weight` (R-DEV-1).** Mirror weighted `feature_count_` /
  `class_count_` via `Y *= sample_weight.T` before `_count` (`:712`/`:1189-1192`).
  ferrolearn's `impl Fit` is `fn fit(&self, x, y)` — NO `sample_weight` parameter
  on `fit` or `partial_fit`.
- REQ-9: **fitted-attribute + PyO3 surface (R-DEV-3 / R-DEFER-1/3).** sklearn
  exposes `feature_log_prob_` / `class_log_prior_` / `feature_count_` /
  `class_count_` / `classes_` / `n_features_in_` (`:1088-1117`; NO `coef_`/
  `intercept_` — deprecated, removed by 1.5.2). ferrolearn `FittedBernoulliNB`
  exposes ONLY `classes()`; `_RsBernoulliNB` (the `py_classifier!` macro) exposes
  ONLY `new(alpha, fit_prior, binarize)` + `fit` + `predict` — NO `class_prior`/
  `force_alpha` kwargs, NO `predict_proba`/`predict_log_proba`/`predict_joint_log_
  proba`/`score`/`partial_fit` (which the library HAS), NO fitted-attr getters.
- REQ-10: **ferray substrate (R-SUBSTRATE).** `bernoulli.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}`,
  not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (`from
sklearn.naive_bayes import BernoulliNB`, run from `/tmp`), never literal-copied
from ferrolearn (R-CHAR-3). The shared binary fixture is `Xbin =
[[1,1,0],[1,0,0],[1,1,0],[0,0,1],[0,1,1],[0,0,1]]`, `y = [0,0,0,1,1,1]`, query
`q = [[1,0,0],[0,1,1]]`; the non-binary fixture is `Xc = [[2,0,1],[0,3,0],[1,1,2],
[0,0,4]]`, `yc = [0,0,1,1]`.

- AC-1 (REQ-1, present & matching, binary data): `BernoulliNB().fit(Xbin,y).
  feature_log_prob_` → `[[-0.2231435513142097, -0.5108256237659905,
  -1.6094379124341003], [-1.6094379124341003, -0.916290731874155,
  -0.2231435513142097]]`; `predict_joint_log_proba(q)` → `[[-2.05572501506252,
  -4.422848629194137], [-4.422848629194137, -2.0557250150625195]]`;
  `predict_log_proba(q)` → `[[-0.0896121586896872, -2.456735772821304],
  [-2.4567357728213044, -0.08961215868968697]]`; `predict_proba(q)` →
  `[[0.9142857142857143, 0.08571428571428572], [0.08571428571428567,
  0.9142857142857145]]`; `predict(q)` → `[0,1]`. ferrolearn matches to ~1e-12
  (verified indirectly via `predict_joint_log_proba` / `predict_proba` —
  `feature_log_prob_` has no public accessor). On binary data sklearn `binarize=0.0`
  and ferrolearn `binarize=None` COINCIDE (`np.allclose == True`).
- AC-2 (REQ-2, present & matching): `binarize([[0.5,0.6],[0.4,0.5],[0.9,0.1]],
  threshold=0.5)` → `[[0,1],[0,0],[1,0]]` (strictly-greater); ferrolearn
  `binarize_array(_, 0.5)` produces the identical 0/1 matrix. On the in-tree
  continuous fixture `BernoulliNB::new().with_binarize(0.5).fit(Xcont, y).predict`
  classifies all 6 correctly.
- AC-3 (REQ-3 pin, the KEY divergence, non-binary data): `BernoulliNB().binarize ==
  0.0`; `BernoulliNB().fit(Xc,yc).feature_count_` → `[[1,1,1],[1,1,2]]`
  (occurrence indicators after binarizing at 0), `predict(Xc)` → `[1,0,1,1]`.
  ferrolearn `BernoulliNB::new()` (`binarize=None`) computes raw feature sums
  `[[2,3,1],[1,1,6]]` — divergent `feature_count_`, potentially divergent
  `predict`. FAILS the binarize-default contract until `new()` defaults to
  `Some(0.0)`.
- AC-4 (REQ-4, present & matching — the MATCH): `BernoulliNB(class_prior=[0.5,0.3]).
  fit(Xbin,y).class_log_prior_` → `[-0.6931471805599453, -1.2039728043259361]`
  (sum 0.8, NO error). `class_prior=[0.5]` → `ValueError("Number of priors must
  match number of classes.")`. ferrolearn `with_class_prior([0.5,0.3]).fit`
  succeeds (length-only), `with_class_prior([0.5]).fit` errors — MATCHES the
  accept/reject decisions (wrong-length error TYPE differs).
- AC-5 (REQ-5, present & matching): with `force_alpha=True` default and `alpha=1`,
  the AC-1 `feature_log_prob_` is reproduced; `clamp_alpha(alpha, true) = alpha`.
- AC-6 (REQ-6 pin): `BernoulliNB(alpha=-0.5).fit(Xbin,y)` → `InvalidParameterError(
  "The 'alpha' parameter of BernoulliNB must be a float in the range [0.0, inf) or
  an array-like. Got -0.5 instead.")`. ferrolearn `with_alpha(-0.5).fit(&Xbin,&y)`
  → **succeeds** (no guard; `clamp_alpha(-0.5, true) = -0.5`). FAILS the reject
  contract until the `alpha >= 0` guard lands.
- AC-7 (REQ-7, present & matching, same classes): two-chunk `partial_fit(Xbin[:4],
  y[:4])` (after `fit`) + `partial_fit(Xbin[4:], y[4:])` reproduces the
  accumulate-then-resmooth path; the in-tree `test_bernoulli_nb_partial_fit`
  exercises it. (sklearn `partial_fit` over chunks == `fit` on the whole — the
  unseen-label path is the REQ-7 sub-gap.)
- AC-8 (REQ-8 pin): `BernoulliNB().fit(Xbin, y, sample_weight=[1,2,1,1,1,3]).
  feature_count_` → `[[4,2,0],[0,1,5]]`, `class_count_` → `[4,5]`. ferrolearn has
  no `sample_weight` parameter.
- AC-9 (REQ-9 surface): `hasattr(BernoulliNB().fit(Xbin,y), 'feature_log_prob_')` /
  `'class_log_prior_'` / `'feature_count_'` / `'class_count_'` / `'classes_'` all
  True in sklearn; `hasattr(..., 'coef_')` is **False** (no coef_/intercept_).
  ferrolearn `FittedBernoulliNB` exposes only `classes()`, and
  `ferrolearn.BernoulliNB` exposes only `fit`/`predict` (no `predict_proba`/
  `predict_log_proba`/`score`/`partial_fit`/`class_prior`/`force_alpha`/fitted-attr
  getters).

## REQ status table

Binary (R-DEFER-2). `BernoulliNB` / `FittedBernoulliNB` are existing pub APIs
re-exported at the crate root and consumed non-test by the `ferrolearn-python`
binding (`RsBernoulliNB` `fit`/`predict`) + the in-crate pipeline (the
production-consumer surface; grandfathered S5/R-DEFER-1). Cites use symbol anchors
(ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live oracle =
installed sklearn 1.5.2, run from `/tmp`. Honest underclaim (R-HONEST-3): the core
smoothing + predict VALUES (on binary data), binarize threshold-application, the
prior paths + length-only check, the floor/toggle, and same-classes `partial_fit`
match and are SHIPPED; the binarize DEFAULT, `alpha >= 0` reject, `sample_weight`,
the fitted-attribute/PyO3 surface, and the ferray substrate are NOT-STARTED.
Suggested blocker numbers — the director creates the real issues (continuing the
bayes layer past multinomial #899-903).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`feature_log_prob_` smoothing + Bernoulli `_joint_log_likelihood` / predict VALUE) | SHIPPED | impl `fn fit` for `BernoulliNB` sets `p = (fc + alpha) / (n_c + 2*alpha)`, `log_prob = p.ln()`, `log_neg_prob = (1-p).ln()`, the closed form of `_update_feature_log_prob` (`naive_bayes.py:1194-1201`, `log(fc+alpha) - log(cc+alpha*2)`); `fn joint_log_likelihood` computes `log_prior[ci] + sum_j [x*log_prob + (1-x)*log_neg_prob]`, the row-wise form of `X @ (flp-neg).T + class_log_prior_ + neg.sum` (`:1203-1219`); `predict`/`predict_proba`/`predict_log_proba`/`predict_joint_log_proba` delegate to `BaseNB` (`.design/bayes/base.md`). Non-test consumer: `RsBernoulliNB::fit`/`predict` (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro) → `FittedBernoulliNB`, surfaced as `ferrolearn.BernoulliNB`; plus `impl PipelineEstimator`. Live oracle on BINARY data (AC-1, where sklearn `binarize=0.0` ≡ ferro `binarize=None`, `np.allclose == True`): `feature_log_prob_` → `[[-0.2231..,-0.5108..,-1.6094..],[-1.6094..,-0.9163..,-0.2231..]]`; `predict_joint_log_proba(q)` → `[[-2.0557..,-4.4228..],[-4.4228..,-2.0557..]]`; `predict_proba(q)` → `[[0.9143..,0.0857..],[0.0857..,0.9143..]]`; `predict(q)` → `[0,1]`; ferrolearn matches to ~1e-12. In-tree `test_bernoulli_nb_fit_predict` / `test_bernoulli_nb_predict_proba_sums_to_one` / `test_bernoulli_nb_predict_proba_ordering` pin labels + normalization. |
| REQ-2 (binarize threshold-application VALUE) | SHIPPED | impl `fn binarize_array` is `x.mapv(\|v\| if v > threshold { 1 } else { 0 })` — strictly-greater — applied in `fit`/`partial_fit`/`joint_log_likelihood` when `binarize` is `Some`, mirroring `binarize(X, threshold=self.binarize)` (`sklearn.preprocessing.binarize`, `X > threshold`) invoked by `_check_X_y`/`_check_X` only when `binarize is not None` (`naive_bayes.py:1179-1186`). Non-test consumer: `RsBernoulliNB` builds `with_binarize(binarize)`; the threshold path feeds `fit`/`predict`. Live oracle (AC-2): `binarize([[0.5,0.6],[0.4,0.5],[0.9,0.1]], 0.5)` → `[[0,1],[0,0],[1,0]]` (strictly-greater, `0.5→0`); ferrolearn `binarize_array(_, 0.5)` matches (NOT `>=`). In-tree `test_bernoulli_nb_binarize_threshold` / `test_bernoulli_nb_binarize_zero_threshold`. |
| REQ-4 (`class_log_prior_` empirical/uniform/explicit + LENGTH-only validation — MATCH) | SHIPPED | impl `fn fit` sets the empirical `log_prior[ci] = (n_c / n).ln()`, the uniform `(1/n_classes).ln()` (`fit_prior == false`), and the explicit `log_prior[ci] = p.ln()` after validating ONLY `priors.len() != n_classes`, mirroring `_update_class_log_prior` (`naive_bayes.py:580-602`: `log(class_count_)-log(class_count_.sum())` `:600`, `-log(n_classes)` `:602`, `log(class_prior)` after length-only check `:589-591`) — discrete NB has NO sum/non-neg check. Deliberate MATCH. Non-test consumer: `RsBernoulliNB::predict` → `fitted.predict` (the `class_log_prior_` term feeds the jll); `with_fit_prior` passes `fit_prior` through. Live oracle (AC-4): `class_prior=[0.5,0.3]` (sum 0.8) ACCEPTED → `class_log_prior_ = log([0.5,0.3])`; ferrolearn `with_class_prior([0.5,0.3]).fit` also succeeds; `class_prior=[0.5]` errors on both. In-tree `test_bernoulli_nb_class_prior` / `test_bernoulli_nb_class_prior_wrong_length`. (Wrong-length error TYPE differs — `InvalidParameter` vs `ValueError` — folded into REQ-9's surface gap.) |
| REQ-5 (`force_alpha` floor + `fit_prior` toggle) | SHIPPED | impl `fn fit` calls `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`, the `_check_alpha` floor `1e-10` unless `force_alpha`, `naive_bayes.py:604-626`) and selects empirical/uniform prior on `fit_prior`. Non-test consumer: `RsBernoulliNB` passes `fit_prior` through `with_fit_prior`; `alpha` through `with_alpha`. Live oracle (AC-5): with `force_alpha=true` default and `alpha=1` the AC-1 `feature_log_prob_` is reproduced; `clamp_alpha(1, true)=1`. In-tree `test_bernoulli_nb_default`; `base.rs` `test_check_alpha_force_alpha_keeps_value` / `test_check_alpha_floors_when_not_forced`. |
| REQ-7 (`partial_fit` VALUE on same-classes data) | SHIPPED | `FittedBernoulliNB::partial_fit` binarizes when `binarize` is `Some`, accumulates `class_counts`/`feature_counts` for each EXISTING class, then recomputes `log_prob`/`log_neg_prob` (same `p=(fc+alpha)/(n_c+2*alpha)` smoothing) and `log_prior`, mirroring the shared `_BaseDiscreteNB.partial_fit` accumulate-then-resmooth (`naive_bayes.py:629-708`, `_count` → `_update_feature_log_prob` → `_update_class_log_prior`). Non-test consumer: in-crate (no PyO3 `partial_fit` — that gap is REQ-9). Live oracle: sklearn `partial_fit` over chunks == `fit` on the whole; ferrolearn's `partial_fit` reproduces the whole-`fit` `log_prob` on same-classes data. In-tree `test_bernoulli_nb_partial_fit` / `test_bernoulli_nb_partial_fit_shape_mismatch`. **Sub-gap (NOT-STARTED, folded into #908):** no `classes` argument — a NEW label in a later chunk is silently dropped (sklearn binarizes against the full `classes` list from the first call). |
| REQ-3 (`binarize` DEFAULT 0.0 vs None) | NOT-STARTED | open prereq blocker **#906**. sklearn `__init__(..., binarize=0.0, ...)` (`naive_bayes.py:1164`) → by DEFAULT every `fit`/`predict` binarizes `X` at `0.0` (`_check_X_y`/`_check_X`, `:1179-1186`). ferrolearn `pub fn new` defaults `binarize = None`, so on NON-binary `X` it skips binarization and computes raw count SUMS instead of binary occurrence counts. Pin (AC-3): `BernoulliNB().binarize == 0.0`; `BernoulliNB().fit(Xc,yc).feature_count_` → `[[1,1,1],[1,1,2]]`, `predict(Xc)` → `[1,0,1,1]`; ferrolearn `BernoulliNB::new()` would compute raw sums `[[2,3,1],[1,1,6]]` (binarize=None). **THE single-file fixable divergence in `bernoulli.rs`** (change `new()` to `binarize = Some(F::from(0.0))` mirroring sklearn `:1164`) — the critic should pin this FIRST. NOTE: the PyO3 binding masks it (`_RsBernoulliNB` defaults `binarize=0.0` and always calls `.with_binarize`), so this is a pure-Rust-API divergence; the in-tree fixtures are already 0/1 so they don't surface it. |
| REQ-6 (`alpha >= 0` validation) | NOT-STARTED | open prereq blocker **#907**. sklearn `_parameter_constraints` declares `alpha: [Interval(Real, 0, None, closed="left"), "array-like"]` (`naive_bayes.py:530`) → `alpha >= 0` is a HARD reject at `fit` (`_validate_params`); `BernoulliNB(alpha=-0.5).fit(X,y)` → `InvalidParameterError("The 'alpha' parameter of BernoulliNB must be a float in the range [0.0, inf) … Got -0.5 instead.")`. ferrolearn `fn fit` has NO `alpha >= 0` guard: `with_alpha(-0.5)` → `clamp_alpha(-0.5, force_alpha=true)` returns `-0.5` unchanged (the `1e-10` floor only fires when `force_alpha=false`), so `fit` proceeds on negative-smoothed garbage. Pin (AC-6): `with_alpha(-0.5).fit(&Xbin,&y)` SUCCEEDS in ferrolearn vs sklearn raises. **Single-file fixable** (an `alpha < 0` reject before/around `clamp_alpha` in `bernoulli.rs` `fn fit`) — same class as multinomial #900. Distinct from the `force_alpha` floor (REQ-5): `>= 0` is a hard reject regardless of `force_alpha`. |
| REQ-8 (`sample_weight`) | NOT-STARTED | open prereq blocker **#908**. sklearn `fit(X, y, sample_weight=None)` (`:712`) weights the binarized `Y` so `feature_count_ = Y.T @ X` / `class_count_ = Y.sum(axis=0)` become weighted (`:1189-1192`). ferrolearn's `impl Fit<Array2<F>, Array1<usize>>` has signature `fn fit(&self, x, y)` — NO `sample_weight` parameter on `fit` or `partial_fit`; also no `classes` argument on `partial_fit` (the unseen-label sub-gap of REQ-7). Pin (AC-8): `BernoulliNB().fit(Xbin,y,sample_weight=[1,2,1,1,1,3]).feature_count_` → `[[4,2,0],[0,1,5]]`, `class_count_` → `[4,5]`; ferrolearn cannot pass weights. |
| REQ-9 (fitted-attribute + PyO3 surface) | NOT-STARTED | open prereq blocker **#909**. sklearn exposes `feature_log_prob_`, `class_log_prior_`, `feature_count_`, `class_count_`, `classes_`, `n_features_in_` (`naive_bayes.py:1088-1117`); `hasattr(fitted, 'coef_') == False` (the deprecated `_BaseDiscreteNB` `coef_`/`intercept_` are gone by 1.5.2). `FittedBernoulliNB` stores `log_prob`/`log_neg_prob`/`log_prior`/`feature_counts`/`class_counts` as PRIVATE fields with no accessor — only `classes()` (via `HasClasses`) is public. `_RsBernoulliNB` (`extras.rs`, the `py_classifier!` macro) exposes ONLY `new(alpha=1.0, fit_prior=true, binarize=0.0)` + `fit` + `predict` — NO `class_prior`/`force_alpha` kwargs, NO `predict_proba`/`predict_log_proba`/`predict_joint_log_proba`/`score`/`partial_fit` (which the library HAS), NO fitted-attr getters. Pin (AC-9): `hasattr(sklearn fitted, 'feature_log_prob_')` True, `'coef_'` False; `ferrolearn.BernoulliNB` reaches only `fit`/`predict`. Also subsumes the `class_prior` wrong-length MESSAGE/TYPE-parity sub-item (REQ-4: `InvalidParameter` vs `ValueError`). |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker **#910**. `bernoulli.rs` imports `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}` (the wrong substrate); not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`bernoulli.rs` follows the unfitted/fitted split (CLAUDE.md naming) for a single
estimator that delegates its entire prediction pipeline to the shared `BaseNB<F>`
trait (`base.rs`):

- `BernoulliNB<F>` (`alpha`, `binarize: Option<F>`, `class_prior: Option<Vec<F>>`,
  `fit_prior`, `force_alpha`) → `Fit<Array2<F>, Array1<usize>>` →
  `FittedBernoulliNB<F>` (`classes`, `log_prior`, `log_prob`, `log_neg_prob`,
  `binarize`, `feature_counts`, `class_counts`, plus
  `alpha`/`class_prior`/`fit_prior` for `partial_fit`).

Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2).

**Construction.** `pub fn new` sets `alpha=1.0`, **`binarize=None`** (DIVERGENT —
sklearn defaults `0.0`, REQ-3/#906), `class_prior=None`, `fit_prior=true`,
`force_alpha=true`. Builder setters `with_alpha`/`with_binarize`/`with_class_prior`/
`with_fit_prior`/`with_force_alpha`.

**Fit path (`fn fit`).** Validation rejects empty `X` (`InsufficientSamples`) and
`n_samples != y.len()` (`ShapeMismatch`). Binarizes `X` via `binarize_array` (`v >
threshold → 1`, REQ-2) ONLY when `self.binarize` is `Some` — the default-`None`
path uses raw `X` (REQ-3/#906). It computes `alpha = clamp_alpha(self.alpha,
force_alpha)` (the `_check_alpha` floor, REQ-5) — but does NOT reject `alpha < 0`
(REQ-6/#907, the `>= 0` constraint sklearn enforces at `:530`). Per class it
accumulates `feature_counts` and the present-probability `p = (fc + alpha) / (n_c +
2*alpha)`, then `log_prob = p.ln()` / `log_neg_prob = (1-p).ln()` — the algebraic
identity of `_update_feature_log_prob` (`:1194-1201`, REQ-1) — and the empirical
`log_prior[ci] = ln(n_c / n)` (REQ-4). Priors resolve: explicit `class_prior`
(LENGTH-only check, then `p.ln()` — REQ-4, the MATCH) wins; else `fit_prior ==
false` → uniform `(1/n_classes).ln()`; else the empirical value already filled. No
`sample_weight` (REQ-8); no negative-feature reject (MATCH — sklearn also accepts
negatives for the Bernoulli model, only emitting a log-domain RuntimeWarning).

**Prediction (delegated to `BaseNB`).** `joint_log_likelihood` binarizes when
`binarize` is `Some`, then computes `log_prior[ci] + sum_j [x*log_prob +
(1-x)*log_neg_prob]`, the row-wise form of `X @ (flp-neg).T + class_log_prior_ +
neg.sum` (`:1216-1217`, REQ-1). `predict` / `predict_proba` / `predict_log_proba` /
`predict_joint_log_proba` delegate to the `BaseNB` provided methods (the `_BaseNB`
pipeline; see `.design/bayes/base.md`). The pipeline is exact and the VALUES match
the oracle to ~1e-12 on already-binary data (REQ-1) — there is no `epsilon_`-style
shift here (unlike GaussianNB), because the Bernoulli smoothing closed form is
exact.

**`partial_fit` (`fn partial_fit`).** Binarizes when `binarize` is `Some`;
accumulates `class_counts`/`feature_counts` for each EXISTING class, then
recomputes `log_prob`/`log_neg_prob` from the accumulated `feature_counts` (same
smoothing) and the `log_prior` (empirical/uniform on `fit_prior`; explicit
`class_prior` sticky), so chunked `partial_fit` == whole `fit` on same-classes data
(REQ-7). No `classes` argument (existing classes only — new labels dropped, REQ-7
sub-gap / #908) and no `sample_weight` (REQ-8).

**Scoring.** `score` = `correct / n` (mean accuracy), the `ClassifierMixin.score`
analog.

**Consumer wiring.** The non-test production consumers:
- `ferrolearn-python` `RsBernoulliNB` / `_RsBernoulliNB` (`extras.rs`, the
  `py_classifier!` macro) — `new(alpha=1.0, fit_prior=true, binarize=0.0)` → `fit`
  (`model.fit(&x_nd, &y_nd)`) → `predict` (`fitted.predict`), registered in
  `lib.rs` (`m.add_class::<extras::RsBernoulliNB>()`) and surfaced as
  `ferrolearn.BernoulliNB` (`_extras.py`, `_RsBernoulliNB(alpha, fit_prior,
  binarize)`). The binding under-exposes (REQ-9), but the `fit`/`predict` path is a
  real non-test consumer of the library `feature_log_prob_`/`joint_log_likelihood`/
  `predict` (REQ-1). The binding's default `binarize=0.0` matches sklearn, so the
  binarize-default divergence (REQ-3) is a pure-Rust-API issue only.
- `impl PipelineEstimator<F> for BernoulliNB<F>` — `fit_pipeline` (float labels →
  `usize`) / `predict_pipeline` consume `fit`/`predict` in-crate (cited as a
  pipeline consumer in `ferrolearn-core/src/pipeline.rs` REQ-1).

**Missing fitted attributes vs sklearn:** `feature_log_prob_` / `class_log_prior_`
/ `feature_count_` / `class_count_` accessors, `n_features_in_` (`:1108`).
ferrolearn exposes only `classes()` / `n_classes()`. (No `coef_`/`intercept_` to
mirror — deprecated/removed by 1.5.2.)

**Invariants held vs sklearn:** `feature_log_prob_` smoothing VALUE (AC-1); the
full predict pipeline VALUE to ~1e-12 on binary data (AC-1); binarize threshold-
application (AC-2, strictly-greater); empirical/uniform/explicit `class_log_prior_`
+ the `class_prior` LENGTH-only accept/reject (AC-4, the MATCH); the negative-
feature accept (MATCH); the `force_alpha` floor + `fit_prior` toggle (AC-5);
same-classes `partial_fit` == `fit` (AC-7); `classes_` ordering; `predict_proba`
rows sum to 1.

**Invariants NOT held vs sklearn:** the `binarize` DEFAULT `0.0` (REQ-3/#906 — the
key single-file divergence, pure-Rust API); the `alpha >= 0` reject (REQ-6/#907);
`sample_weight` + the `partial_fit` unseen-label gap (REQ-8/#908); the fitted-
attribute + PyO3 surface (REQ-9, including the `class_prior` wrong-length message
type); the ferray substrate (REQ-10).

## Verification

Library crate (green at baseline `e0940350` for the existing contract):
```
cargo test -p ferrolearn-bayes --lib bernoulli
cargo clippy -p ferrolearn-bayes --all-targets -- -D warnings
cargo fmt --all --check
```
The 17 in-tree `#[test]`s (`test_bernoulli_nb_fit_predict`,
`test_bernoulli_nb_predict_proba_sums_to_one`, `test_bernoulli_nb_has_classes`,
`test_bernoulli_nb_binarize_threshold`, `test_bernoulli_nb_binarize_zero_threshold`,
`test_bernoulli_nb_single_class`, `test_bernoulli_nb_partial_fit`,
`test_bernoulli_nb_class_prior`, `test_bernoulli_nb_class_prior_wrong_length`,
`test_bernoulli_nb_predict_proba_ordering`, `test_bernoulli_nb_default`, …) pin
ferrolearn's current behavior. **All in-tree fixtures are already 0/1, so none
surfaces the binarize-default divergence (#906), none compares against the live
sklearn oracle, and none exercises `alpha < 0`** — they stay green despite the
#906/#907 divergences; the SHIPPED REQs (REQ-1 smoothing+predict VALUE on binary
data, REQ-2 binarize threshold, REQ-4 prior/length-only, REQ-5 floor/toggle, REQ-7
same-classes partial_fit) value-match the oracle, the rest are NOT-STARTED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin the two deterministic single-file ones
FIRST** (both fixable in `bernoulli.rs` alone): REQ-3 (`binarize` default 0.0) and
REQ-6 (`alpha >= 0`):
```
# REQ-3 (#906) binarize DEFAULT 0.0 vs None — THE key divergence (non-binary data)
python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; X=np.array([[2.,0.,1.],[0.,3.,0.],[1.,1.,2.],[0.,0.,4.]]); y=np.array([0,0,1,1]); m=BernoulliNB().fit(X,y); print('binarize default ==', BernoulliNB().binarize); print(m.feature_count_.tolist(), m.predict(X).tolist())"  # 0.0 ; [[1,1,1],[1,1,2]] [1,0,1,1]  (ferro new() binarize=None -> raw sums [[2,3,1],[1,1,6]])
# REQ-6 (#907) alpha < 0 — sklearn rejects, ferrolearn accepts
python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; X=np.array([[1.,1.,0.],[1.,0.,0.],[1.,1.,0.],[0.,0.,1.],[0.,1.,1.],[0.,0.,1.]]); y=np.array([0,0,0,1,1,1]);
try: BernoulliNB(alpha=-0.5).fit(X,y)
except Exception as e: print(type(e).__name__,'::',e)"  # InvalidParameterError :: The 'alpha' parameter of BernoulliNB must be a float in the range [0.0, inf) or an array-like. Got -0.5 instead.   (ferro with_alpha(-0.5).fit succeeds)
# REQ-1 (present) feature_log_prob_ + predict VALUE on BINARY data (binarize 0.0 ≡ None)
python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; X=np.array([[1.,1.,0.],[1.,0.,0.],[1.,1.,0.],[0.,0.,1.],[0.,1.,1.],[0.,0.,1.]]); y=np.array([0,0,0,1,1,1]); m=BernoulliNB().fit(X,y); q=np.array([[1.,0.,0.],[0.,1.,1.]]); print(m.feature_log_prob_.tolist()); print(m.predict_proba(q).tolist()); print(m.predict(q).tolist())"  # flp [[-0.2231..,-0.5108..,-1.6094..],[-1.6094..,-0.9163..,-0.2231..]]; pp [[0.9143..,0.0857..],[0.0857..,0.9143..]]; pred [0,1]
# REQ-2 (present, MATCH) binarize strictly-greater
python3 -c "from sklearn.preprocessing import binarize; import numpy as np; print(binarize(np.array([[0.5,0.6],[0.4,0.5],[0.9,0.1]]), threshold=0.5).tolist())"  # [[0,1],[0,0],[1,0]]
# REQ-4 (present, MATCH) class_prior length-only (sum 0.8 accepted)
python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; X=np.array([[1.,1.,0.],[1.,0.,0.],[1.,1.,0.],[0.,0.,1.],[0.,1.,1.],[0.,0.,1.]]); y=np.array([0,0,0,1,1,1]); print(BernoulliNB(class_prior=[0.5,0.3]).fit(X,y).class_log_prior_.tolist())"  # [-0.6931471805599453, -1.2039728043259361]  (= log([0.5,0.3]); NO sum check — ferro MATCHES)
# REQ-8 sample_weight
python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; X=np.array([[1.,1.,0.],[1.,0.,0.],[1.,1.,0.],[0.,0.,1.],[0.,1.,1.],[0.,0.,1.]]); y=np.array([0,0,0,1,1,1]); m=BernoulliNB().fit(X,y,sample_weight=np.array([1.,2.,1.,1.,1.,3.])); print(m.feature_count_.tolist(), m.class_count_.tolist())"  # [[4,2,0],[0,1,5]] [4,5]
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-bayes/tests/divergence_bernoulli.rs`, asserting the live-sklearn
expected values above and FAILING against current `bernoulli.rs` (REQ-3 the
binarize default and REQ-6 the `alpha < 0` reject are the cleanest single-file
pins). REQ-1/REQ-2/REQ-4/REQ-5/REQ-7 already match and should be guarded by
non-regression pins.

ferrolearn-python (REQ-9 binding parity, after #909 lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_naive_bayes.py -q
```
asserting `ferrolearn.BernoulliNB` exposes `feature_log_prob_` /
`class_log_prior_` / `feature_count_` / `class_count_` / `predict_proba` /
`predict_log_proba` / `score` / `partial_fit` and `class_prior` / `force_alpha`
kwargs, matching `sklearn.naive_bayes.BernoulliNB` on the AC fixtures.

## Blockers to open

(Director creates the real issues; the numbers are SUGGESTIONS continuing the
bayes layer past multinomial #899-903. #905 is this doc's crosslink tracking
issue.)

- **#906** — REQ-3 (`binarize` DEFAULT 0.0): change `bernoulli.rs` `pub fn new` to
  default `binarize = Some(F::from(0.0))` mirroring sklearn `__init__(...,
  binarize=0.0, ...)` (`naive_bayes.py:1164`) so non-binary `X` is binarized at `0`
  by default. **The cleanest single-file deterministic fix** — the critic should
  pin this FIRST (the in-tree 0/1 fixtures hide it; add a non-binary fixture).
- **#907** — REQ-6 (`alpha >= 0` validation): add an `alpha < 0` reject to
  `bernoulli.rs` `fn fit` (mirroring `_parameter_constraints` `alpha: Interval(
  Real, 0, None, closed="left")`, `naive_bayes.py:530`) before/around `clamp_alpha`.
  Same class as multinomial #900. (Distinct from the `force_alpha` floor, which
  only fires for `alpha < 1e-10` when `force_alpha=false`.)
- **#908** — REQ-8 (`sample_weight` + `partial_fit` classes): add weighted
  `feature_count_`/`class_count_` (`Y *= sample_weight.T` before `_count`,
  `:712`/`:1189-1192`) — needs a `sample_weight` parameter on `fit`/`partial_fit`
  (Fit-trait shape, R-DEV-1); add a `classes` argument to `partial_fit` so a new
  later-chunk label is not dropped (REQ-7 sub-gap).
- **#909** — REQ-9 (fitted-attribute + PyO3 surface): expose `feature_log_prob_`/
  `class_log_prior_`/`feature_count_`/`class_count_` accessors on
  `FittedBernoulliNB`, and add `class_prior`/`force_alpha` kwargs +
  `predict_proba`/`predict_log_proba`/`predict_joint_log_proba`/`score`/
  `partial_fit` + those getters to `_RsBernoulliNB` (`ferrolearn-python/src/
  extras.rs`). Also align the `class_prior` wrong-length error with sklearn's
  `ValueError("Number of priors must match number of classes.")` (REQ-4 sub-item).
- **#910** — REQ-10 (ferray substrate): migrate `bernoulli.rs` off
  `ndarray`/`num-traits` to `ferray-core` (R-SUBSTRATE).
