# Complement Naive Bayes (sklearn.naive_bayes.ComplementNB)

<!--
tier: 3-component
status: draft
baseline-commit: 45f1c66e
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/naive_bayes.py   # ComplementNB(_BaseDiscreteNB) :899-1049; __init__(*, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None, norm=False) (:1005-1020); _parameter_constraints norm=["boolean"] (:1000-1003) + shared alpha=Interval(Real,0,None,closed="left") (:530); _count check_non_negative + feature_count_ += Y.T@X + class_count_ += Y.sum + feature_all_ = feature_count_.sum(axis=0) (:1025-1030); _update_feature_log_prob comp_count=feature_all_+alpha-feature_count_, logged=log(comp_count/comp_count.sum(axis=1,keepdims=True)), norm? logged/logged.sum : -logged (:1032-1042); _joint_log_likelihood jll=X@flp.T; if len(classes_)==1: jll+=class_log_prior_ (:1044-1049); shared _update_class_log_prior LENGTH-only (:580-602); _check_alpha floor 1e-10 unless force_alpha (:604-626); shared fit/partial_fit (:628-762)
ferrolearn-module: ferrolearn-bayes/src/complement.rs
parity-ops: ComplementNB (.__init__, .fit, .partial_fit, .predict, .predict_proba, .predict_log_proba, .predict_joint_log_proba, .score)
crosslink-issue: 913
-->

## Summary

`ferrolearn-bayes/src/complement.rs` mirrors scikit-learn's `ComplementNB`
(`sklearn/naive_bayes.py`, `class ComplementNB(_BaseDiscreteNB)` `:899-1049`) —
the Rennie et al. (2003) Multinomial-NB variant for imbalanced data whose
per-feature weight is the negated smoothed complement-class log-probability
`w_cj = -log((N_~cj + alpha) / (N_~c + alpha * n_features))`, where `N_~cj` is
the total count of feature `j` over all classes EXCEPT `c` and `N_~c` is the
total count over all features in all classes except `c`. It exposes the unfitted
`ComplementNB<F>` (`alpha=1.0`, `class_prior: Option<Vec<F>>`, `fit_prior=true`,
`force_alpha=true`, `norm=false`), the fitted `FittedComplementNB<F>` (per-class
`weights` / `feature_counts` / `class_counts` plus `alpha`/`norm` carried for
`partial_fit`), and delegates the entire prediction pipeline to the shared
`BaseNB<F>` trait (`base.rs`, the `_BaseNB` analog — see `.design/bayes/base.md`).
It is re-exported at the crate root (`ferrolearn-bayes/src/lib.rs`) and **bound
into `ferrolearn-python`** (`RsComplementNB` / `_RsComplementNB` in
`ferrolearn-python/src/extras.rs`, surfaced as `ferrolearn.ComplementNB` via
`_extras.py`) — a genuine non-test production consumer (the binding `fit`/`predict`
+ the in-crate pipeline integration).

Under honest underclaim (R-HONEST-3), the behaviors that are genuinely present
**and value-match the live sklearn 1.5.2 oracle** (verified on the count fixture
`X = [[5,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]]`, `y = [0,0,0,1,1,1]`,
query `q = [[3,1,1],[0,1,4]]`, run from `/tmp`) are:

- **`feature_log_prob_` complement-weight VALUE (norm=False)** — sklearn's
  `_update_feature_log_prob` computes `comp_count = feature_all_ + alpha -
  feature_count_`; `logged = log(comp_count / comp_count.sum(axis=1,
  keepdims=True))`; `feature_log_prob_ = -logged` (`:1032-1042`). ferrolearn `fn
  fit` computes `complement_count_j = total_feature_counts[j] -
  class_feature_counts[ci,j]` and `weights[ci,j] = -((complement_count_j + alpha)
  / (complement_total + alpha*n_features)).ln()` — the algebraic identity
  (`feature_all_[j] == total_feature_counts[j]`, so `comp_count[ci,j] ==
  complement_count_j` and `comp_count.sum(axis=1) == complement_total +
  alpha*n_features`). The oracle confirms `np.allclose(-logged,
  m.feature_log_prob_) == True`; ferrolearn has NO public `feature_log_prob_`
  accessor, so the VALUE is verified indirectly through `predict_joint_log_proba`
  / `predict_proba`.
- **`_joint_log_likelihood` + `predict` / `predict_proba` / `predict_log_proba` /
  `predict_joint_log_proba` VALUE (norm=False)** — sklearn's `jll = X @
  feature_log_prob_.T` (`:1046`) feeding the `_BaseNB` pipeline; with the
  sklearn-parity sign, higher jll wins and `argmax` predicts. ferrolearn's `fn
  joint_log_likelihood` computes `X @ weights.T` directly and the delegated
  `predict_*` match the oracle to ~1e-12 (oracle `predict_joint_log_proba(q) =
  [[9.2168..,5.0580..],[2.9785..,11.2963..]]`, `predict_proba(q) =
  [[0.9846..,0.0153..],[0.000244..,0.99975..]]`, `predict(q) = [0,1]`; ferrolearn
  identical).
- **`norm=True` VALUE** — sklearn's `feature_log_prob_ = logged / logged.sum(
  axis=1, keepdims=True)` when `norm=True` (`:1037-1039`). ferrolearn stores
  `weights = -logged` then `apply_norm_inplace` divides each row by its sum:
  `(-logged) / sum(-logged)`. These are algebraically identical
  (`(-logged)/sum(-logged) == logged/sum(logged)`, the two minus signs cancel),
  confirmed numerically: oracle `norm=True predict_proba(q) =
  [[0.7192..,0.2807..],[0.1322..,0.8677..]]`, `predict(q) = [0,1]`; ferrolearn
  `with_norm(true)` produces the IDENTICAL proba/labels.
- **`class_prior` LENGTH-ONLY validation** — sklearn's shared
  `_update_class_log_prior` checks ONLY length (`:589-590`), with NO sum-to-1 and
  NO non-negativity check for discrete NB. ferrolearn's `fn fit` checks ONLY
  length — a **MATCH** (`ComplementNB(class_prior=[0.5,0.3])`, sum 0.8, fits on
  both sides). NOTE: for ComplementNB `class_prior` / `class_log_prior_` is "Not
  used" except the single-class jll add (`:929`, `:944-945`), so this affects only
  the length-validation decision, not multi-class predictions.
- **`force_alpha` / `_check_alpha` floor + `fit_prior` carry** — the
  `base::check_alpha` / `clamp_alpha` floor (`1e-10` unless `force_alpha`); the
  `fit_prior` flag is stored but (matching sklearn) only matters in the
  single-class edge case.
- **negative-feature reject** — both REJECT negative `X` (sklearn `ValueError`,
  ferrolearn `InvalidParameter`); the exact message/type differs. (Unlike
  BernoulliNB, ComplementNB DOES call `check_non_negative`, `:1027`.)
- **`partial_fit` VALUE on same-classes data** — incremental count accumulation
  then recompute; the oracle confirms chunked `partial_fit` over existing classes
  == `fit` on the concatenation (`np.allclose == True`); ferrolearn's `partial_fit`
  accumulates `feature_counts` then recomputes `weights` the same way.
- **single-class jll `class_log_prior_` add — BENIGN (no observable divergence)**
  — sklearn's `_joint_log_likelihood` ADDS `class_log_prior_` ONLY when
  `len(classes_) == 1` (`:1047-1048`); ferrolearn's `X @ weights.T` does NOT add a
  prior. This is NOT observable: with a single class, `class_log_prior_ = [0.0]`
  (`log(class_count_ / class_count_.sum()) = log(1) = 0`; oracle confirms
  `class_log_prior_ = [0.0]`, and `[-0.0]` under `fit_prior=False`), so the add is
  literally zero; and even were it nonzero, softmax over one column is always
  `[[1.0]]`. Oracle single-class `predict_proba = [[1.0]]`, `predict = [0]`;
  ferrolearn `predict_proba = [[1.0]]`, `predict = [0]` — COINCIDE.
- **`score`** — mean accuracy (`ClassifierMixin.score` analog).

The behaviors that **diverge** from the `ComplementNB` contract (each pinned to a
NOT-STARTED REQ with a concrete prereq blocker — the director creates the real
issues; the numbers below are SUGGESTIONS continuing the bayes layer past
bernoulli #905-910):

1. **`alpha >= 0` validation (R-DEV-2 — THE key fixable divergence).** sklearn's
   shared `_parameter_constraints` declares `alpha: [Interval(Real, 0, None,
   closed="left"), "array-like"]` (`:530`, inherited by `ComplementNB` at
   `:1000-1001`) → `alpha >= 0` is a HARD reject at `fit` (`_validate_params`);
   `ComplementNB(alpha=-0.5).fit(X,y)` raises `InvalidParameterError("The 'alpha'
   parameter of ComplementNB must be a float in the range [0.0, inf) or an
   array-like. Got -0.5 instead.")`. ferrolearn `fn fit` has NO `alpha >= 0`
   guard: `with_alpha(-0.5)` → `clamp_alpha(-0.5, force_alpha=true)` returns
   `-0.5` unchanged (the `1e-10` floor only fires when `force_alpha=false`), so
   `fit` proceeds and computes `-log((complement_count - 0.5)/denom)` — garbage /
   NaN for small complement counts, no error. **The single-file-fixable divergence
   in `complement.rs` `fn fit`** (same class as multinomial #900 / bernoulli #907).
2. **`sample_weight` (R-DEV-1).** sklearn `fit(X, y, sample_weight=None)` (`:712`)
   weights the binarized `Y` so `feature_count_ = Y.T @ X` / `class_count_ =
   Y.sum(axis=0)` / `feature_all_ = feature_count_.sum(axis=0)` become weighted
   counts. ferrolearn's `Fit` trait is `fn fit(&self, x, y)` — NO `sample_weight`
   parameter on `fit` or `partial_fit`.
3. **fitted-attribute + PyO3 surface (R-DEV-3 / R-DEFER-1/3).** sklearn exposes
   `feature_log_prob_`, `feature_all_`, `feature_count_`, `class_count_`,
   `class_log_prior_`, `classes_`, `n_features_in_` (`:937-970`; NO `coef_`/
   `intercept_` — the deprecated `_BaseDiscreteNB` properties are gone by 1.5.2).
   ferrolearn `FittedComplementNB` exposes ONLY `classes()` (via `HasClasses`);
   `weights` / `feature_counts` / `class_counts` are private fields with no
   accessor (and there is no stored `feature_all_` / `class_log_prior_`).
   `_RsComplementNB` (`extras.rs`, the `py_classifier!` macro) exposes ONLY
   `new(alpha=1.0, fit_prior=true, norm=false)` + `fit` + `predict` — NO
   `class_prior`/`force_alpha` kwargs, NO `predict_proba`/`predict_log_proba`/
   `predict_joint_log_proba`/`score`/`partial_fit` (which the library HAS), NO
   fitted-attr getters.
4. **`partial_fit` unseen-label gap (R-DEV-1, sub-item of #915).** sklearn's shared
   `partial_fit(X, y, classes=None, ...)` (`:628-709`) takes the full `classes`
   list on the first call and binarizes against it, so a label unseen in a later
   chunk is still represented. ferrolearn `FittedComplementNB::partial_fit`
   iterates only the existing `self.classes` — a NEW label appearing in a later
   chunk is silently DROPPED (no `classes` argument). On same-classes data
   `partial_fit == fit` (SHIPPED, REQ-6); the new-label path diverges.
5. **ferray substrate (R-SUBSTRATE).** `complement.rs` imports `ndarray::{Array1,
   Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}`, not `ferray-core`.

`ComplementNB` / `FittedComplementNB` are existing pub APIs (grandfathered per
S5/R-DEFER-1); their non-test production consumers are the `ferrolearn-python`
binding (`RsComplementNB` `fit`/`predict`) and the in-crate pipeline integration
(`impl PipelineEstimator for ComplementNB`).

## Algorithm (sklearn — the contract)

### Construction (`naive_bayes.py:1005-1020`)

`ComplementNB(*, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None,
norm=False)` — all keyword-only. `_parameter_constraints` (`:1000-1003`, merging
`_BaseDiscreteNB`'s `:529-534`): `alpha: [Interval(Real, 0, None, closed="left"),
"array-like"]` (**>= 0**, `:530`); `norm: ["boolean"]` (default `False`, `:1002`);
`fit_prior: ["boolean"]`; `class_prior: ["array-like", None]` ("Not used", `:929`);
`force_alpha: ["boolean"]`. `_more_tags` declares `requires_positive_X: True`
(`:1022-1023`).

### Fit (`_BaseDiscreteNB.fit` `:712-762`)

`fit(X, y, sample_weight=None)`: binarize `y` → one-hot `Y`; if `sample_weight`
given, `Y *= sample_weight.T`; `_init_counters` zeroes `class_count_` /
`feature_count_`; `_count(X, Y)` accumulates; `alpha = self._check_alpha()`;
`_update_feature_log_prob(alpha)`; `_update_class_log_prior(class_prior)`.

- **`_count`** (`:1025-1030`): `check_non_negative(X, "ComplementNB (input X)")`
  (`:1027`); `feature_count_ += safe_sparse_dot(Y.T, X)` (per-class feature sums);
  `class_count_ += Y.sum(axis=0)` (per-class sample counts); `feature_all_ =
  feature_count_.sum(axis=0)` (`:1030`, the per-feature total over all classes).
- **`_check_alpha`** (`:604-626`): `alpha_lower_bound = 1e-10` (`:618`); if
  `alpha_min < alpha_lower_bound and not self.force_alpha` (`:619`) warn + return
  `np.maximum(alpha, alpha_lower_bound)`; else return alpha unchanged. (The `>= 0`
  HARD constraint is enforced earlier by `_validate_params` against
  `_parameter_constraints` `:530`, NOT inside `_check_alpha`.)
- **`_update_feature_log_prob`** (`:1032-1042`): `comp_count = self.feature_all_ +
  alpha - self.feature_count_` (the complement count `N_~cj + alpha` per
  (class, feature)); `logged = np.log(comp_count / comp_count.sum(axis=1,
  keepdims=True))` (the complement log-probability `log((N_~cj + alpha) / (N_~c +
  alpha*n_features))`); then **`if self.norm: feature_log_prob_ = logged /
  logged.sum(axis=1, keepdims=True)` else `feature_log_prob_ = -logged`**
  (`:1037-1042`). The `-logged` sign makes ComplementNB's `_BaseNB.predict`
  argmax (higher weight = LOWER complement probability = better fit), per the
  inline comment "`_BaseNB.predict uses argmax, but ComplementNB operates with
  argmin`" (`:1036`).
- **`_update_class_log_prior`** (`:580-602`): if `class_prior is not None` →
  `if len(class_prior) != n_classes: raise ValueError("Number of priors must match
  number of classes.")` (`:589-590`); `class_log_prior_ = np.log(class_prior)`
  (`:591`) — **LENGTH-ONLY check, NO sum-to-1, NO non-negativity**. elif
  `fit_prior` → `log(class_count_) - log(class_count_.sum())` (`:600`). else →
  `np.full(n_classes, -np.log(n_classes))` (uniform, `:602`). For ComplementNB
  `class_log_prior_` is used ONLY in the single-class jll branch (`:1047-1048`).

### `_joint_log_likelihood` (`:1044-1049`)

`jll = safe_sparse_dot(X, self.feature_log_prob_.T)` (`:1046`, `X @
feature_log_prob_.T`); `if len(self.classes_) == 1: jll += self.class_log_prior_`
(`:1047-1048`). **NOTE: the multi-class jll does NOT add `class_log_prior_`** —
only the single-class case does. Shape `(n_samples, n_classes)`. The shared
`_BaseNB` pipeline (`predict` / `predict_proba` / `predict_log_proba` /
`predict_joint_log_proba`) consumes this (`.design/bayes/base.md`).

### `partial_fit` (`_BaseDiscreteNB.partial_fit` `:628-709`)

`partial_fit(X, y, classes=None, sample_weight=None)`. First call initializes the
counters (`classes` required); each call binarizes `y` (against the FULL `classes`
list), optionally weights `Y`, `_count(X, Y)` accumulates (re-deriving
`feature_all_` each call), then recomputes `alpha = _check_alpha()`,
`_update_feature_log_prob(alpha)`, `_update_class_log_prior(class_prior)`. Because
the smoothing is reapplied to the accumulated counts each call, `partial_fit` over
chunks equals `fit` on the concatenation.

### Edge cases (live oracle, sklearn 1.5.2, run from /tmp)

- **feature_log_prob_ VALUE (norm=False)** (count fixture above, `alpha=1`):
  `feature_count_ = [[15,3,1],[1,3,15]]`; `feature_all_ = [16,6,16]`;
  `class_count_ = [3,3]`; `feature_log_prob_ = [[2.3978952727983707,
  1.7047480922384253, 0.3184537311185346], [0.3184537311185346,
  1.7047480922384253, 2.3978952727983707]]` (the `-logged` positive weights); the
  closed form `-log(comp_count / comp_count.sum(axis=1, keepdims=True))` with
  `comp_count = feature_all_ + 1 - feature_count_` reproduces it
  (`np.allclose == True`).
- **predict VALUE (norm=False)** (`q = [[3,1,1],[0,1,4]]`):
  `predict_joint_log_proba(q) = [[9.216887641752072, 5.058004558392399],
  [2.9785630167125636, 11.296329183431908]]`; `predict_log_proba(q) =
  [[-0.015504186535965303, -4.174387269895638], [-8.318010277546872,
  -0.00024411082752706648]]`; `predict_proba(q) = [[0.9846153846153846,
  0.015384615384615375], [0.0002440810349035878, 0.9997559189650967]]`;
  `predict(q) = [0, 1]`.
- **norm=True VALUE** (same fixture): `feature_log_prob_ =
  [[0.5423756186860554, 0.3855939046715139, 0.07203047664243067],
  [0.07203047664243067, 0.3855939046715139, 0.5423756186860554]]` (`logged /
  logged.sum`); `predict_joint_log_proba(q) = [[2.0847512373721107,
  1.1440609532848613], [0.6737158112412366, 2.5550963794157355]]`;
  `predict_proba(q) = [[0.7192390704948571, 0.2807609295051429],
  [0.13223037910101987, 0.8677696208989801]]`; `predict(q) = [0, 1]`. ferrolearn
  `with_norm(true)` (`(-logged)/sum(-logged)`) reproduces the IDENTICAL proba and
  labels (the two minus signs cancel: `(-logged)/sum(-logged) ==
  logged/sum(logged)`).
- **alpha < 0**: `ComplementNB(alpha=-0.5).fit(X,y)` → `InvalidParameterError: The
  'alpha' parameter of ComplementNB must be a float in the range [0.0, inf) or an
  array-like. Got -0.5 instead.` (raised at `fit` by `_validate_params`).
- **single-class jll**: `ComplementNB().fit([[1,2,3],[4,5,6],[7,8,9]], [0,0,0])`:
  `classes_ = [0]`; `class_log_prior_ = [0.0]` (= `log(3/3) = log(1) = 0`);
  `predict_joint_log_proba([[2,1,1]]) = [[4.394449154672439]]` (the jll INCLUDES
  the `+ class_log_prior_` add, but it is `+0.0`); `predict_proba = [[1.0]]`;
  `predict = [0]`. Under `fit_prior=False`, `class_log_prior_ = [-0.0]`,
  `predict_proba = [[1.0]]`. So the single-class `class_log_prior_` add is
  **BENIGN** — both the prior is zero AND softmax over one column is always
  `[[1.0]]`; ferrolearn's omission of the add is not observable.
- **class_prior length-only**: `ComplementNB(class_prior=[0.5,0.3]).fit(X,y)`
  (sum 0.8) ACCEPTED → `class_log_prior_ = [-0.6931471805599453,
  -1.2039728043259361]` (= `log([0.5,0.3])`), NO sum/non-neg error. Wrong length
  `class_prior=[0.5]` → `ValueError: Number of priors must match number of
  classes.` ferrolearn MATCHES all three (length-only). (`class_prior` does not
  affect multi-class predictions — "Not used" except single-class.)
- **negative features**: `ComplementNB().fit(X_with_neg, y)` → `ValueError:
  Negative values in data passed to ComplementNB (input X)`. ferrolearn rejects
  with `InvalidParameter { name: "X" }`.
- **sample_weight**: `ComplementNB().fit(X, y, sample_weight=...)` weights
  `feature_count_` / `class_count_` / `feature_all_`. ferrolearn has no
  `sample_weight` parameter.
- **partial_fit == fit**: `ComplementNB().partial_fit(X[:4], y[:4], classes=[0,1])`
  then `partial_fit(X[4:], y[4:])` yields `feature_log_prob_` identical to
  `fit(X, y)` (`np.allclose == True`).
- **coef_/intercept_**: `hasattr(ComplementNB().fit(X,y), 'coef_') == False` —
  ComplementNB exposes NO `coef_`/`intercept_`. The fitted-attr surface is
  `feature_log_prob_` / `feature_all_` / `feature_count_` / `class_count_` /
  `class_log_prior_` / `classes_` / `n_features_in_`.

## ferrolearn (what exists)

All in `ferrolearn-bayes/src/complement.rs`, generic over `F: Float + Send + Sync
+ 'static`; `ndarray` substrate. Every public method returns `Result<_,
FerroError>` (no panics in library code, R-CODE-2).

- **`pub struct ComplementNB<F> { pub alpha: F, pub class_prior: Option<Vec<F>>,
  pub fit_prior: bool, pub force_alpha: bool, pub norm: bool }`** — `pub fn new`
  sets `alpha = 1.0`, `class_prior = None`, `fit_prior = true`, `force_alpha =
  true`, `norm = false` (matching sklearn defaults, `:1005-1020`); builder setters
  `with_alpha` / `with_class_prior` / `with_fit_prior` / `with_force_alpha` /
  `with_norm`; `impl Default → new()`.
- **`pub struct FittedComplementNB<F>`** — private fields `classes: Vec<usize>`,
  `weights: Array2<F>` (the `feature_log_prob_` analog, stored as sklearn's
  positive `-log(complement_prob)` so `argmax` predicts), `feature_counts:
  Array2<F>` (the `feature_count_` analog), `class_counts: Vec<usize>` (the
  `class_count_` analog), plus `alpha` / `norm` carried for `partial_fit`. **No
  public accessor** for any of these (only `classes()` via `HasClasses`); no
  stored `feature_all_` / `class_log_prior_`.
- **`impl Fit<Array2<F>, Array1<usize>> for ComplementNB<F>` / `fn fit`** — rejects
  `n_samples == 0` (`InsufficientSamples`), `n_samples != y.len()`
  (`ShapeMismatch`), any negative feature (`InvalidParameter { name: "X" }` —
  REQ-7). Collects sorted-deduped `classes`; `alpha = crate::clamp_alpha(
  self.alpha, self.force_alpha)` (the `_check_alpha` floor — REQ-5). Accumulates
  per-class `class_feature_counts` and `class_counts`; derives
  `total_feature_counts` (the `feature_all_` analog) and `total_all`. For each
  class: `complement_total = total_all - class_feature_counts.row(ci).sum()`,
  `denom = complement_total + alpha*n_features`, then `weights[[ci,j]] =
  -((total_feature_counts[j] - class_feature_counts[ci,j] + alpha) / denom).ln()`
  — the algebraic identity of `_update_feature_log_prob`'s `-logged` (`:1032-1042`,
  REQ-1). If `self.norm`, `apply_norm_inplace` divides each row by its sum (REQ-3).
  Validates `class_prior` LENGTH-only (REQ-4). **No `alpha >= 0` guard** (REQ-2),
  **no `sample_weight`** (REQ-8). (No `class_log_prior_` computed — ComplementNB
  multi-class jll does not use it; the single-class add is benign — REQ-1.)
- **`fn apply_norm_inplace<F>(weights)`** — divides each row of `weights` by its
  row sum (skipping zero-sum rows), mirroring sklearn's `logged / logged.sum(
  axis=1)` (REQ-3; `(-logged)/sum(-logged) == logged/sum(logged)`).
- **`FittedComplementNB::partial_fit(&mut self, x, y)`** — accumulates
  `class_counts` + `feature_counts` for each EXISTING class, then re-derives
  `total_feature_counts` / `total_all` and recomputes `weights` (same `-log`
  complement smoothing), re-applying `apply_norm_inplace` when `norm`. Rejects
  feature-count mismatch (`ShapeMismatch`) / negative features (`InvalidParameter`).
  **No `sample_weight`, no `classes` argument** (existing classes only — a new
  later-chunk label is dropped, REQ-6 sub-item).
- **`impl BaseNB<F> for FittedComplementNB<F>` / `fn joint_log_likelihood`** —
  `score[i,ci] = sum_j x[i,j] * weights[ci,j]` (= `X @ weights.T`), shape
  `(n_samples, n_classes)`, mirroring sklearn's `X @ feature_log_prob_.T` (`:1046`,
  REQ-1). Does NOT add `class_log_prior_` in the single-class case (benign —
  REQ-1). `fn nb_classes` returns `&self.classes`.
- **`pub fn predict_proba` / `pub fn predict_log_proba` / `pub fn
  predict_joint_log_proba`** — delegate to `BaseNB::nb_predict_proba` /
  `nb_predict_log_proba` / `nb_predict_joint_log_proba` (REQ-1 pipeline).
- **`impl Predict for FittedComplementNB<F>` / `fn predict`** — delegates to
  `BaseNB::nb_predict` (`classes_[argmax(jll)]`, first-max tie-break). With the
  `-logged` sign, the highest jll wins (sklearn's argmax/argmin convention).
- **`pub fn score(&self, x, y)`** — mean accuracy (`correct / n`), the
  `ClassifierMixin.score` analog.
- **`impl HasClasses for FittedComplementNB<F>`** — `classes()` / `n_classes()`.
- **Pipeline**: `impl PipelineEstimator<F> for ComplementNB<F>` (`fn fit_pipeline`,
  maps float labels → `usize`) + `FittedComplementNBPipeline` (`fn
  predict_pipeline`).

**Consumers (non-test).** Crate re-export (`ferrolearn-bayes/src/lib.rs`, `pub use
complement::{ComplementNB, FittedComplementNB}`) plus:
- **`ferrolearn-python`** — `RsComplementNB` / `_RsComplementNB`
  (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro): `new(alpha=1.0,
  fit_prior=true, norm=false)` → `ComplementNB::<f64>::new().with_alpha(alpha).
  with_fit_prior(fit_prior).with_norm(norm)`, `fit` (`model.fit(&x_nd, &y_nd)`),
  `predict` (`fitted.predict`); registered in `ferrolearn-python/src/lib.rs`
  (`m.add_class::<extras::RsComplementNB>()`) and surfaced as
  `ferrolearn.ComplementNB` (`ferrolearn-python/python/ferrolearn/_extras.py`,
  `class ComplementNB(_ClassifierWrapper)` → `_RsComplementNB(alpha, fit_prior,
  norm)`). The binding under-exposes (REQ-9), but the `fit`/`predict` path is a
  real non-test consumer of the library `feature_log_prob_`/`joint_log_likelihood`/
  `predict` (REQ-1) and threads the `norm` flag through (REQ-3).
- **Pipeline** (`impl PipelineEstimator`) consumes `fit` / `predict` in-crate.

## Requirements

- REQ-1: **`feature_log_prob_` complement-weight VALUE (norm=False) +
  `_joint_log_likelihood` / `predict` / `predict_proba` / `predict_log_proba` /
  `predict_joint_log_proba` VALUE (R-DEV-1/3).** Mirror `_update_feature_log_prob`
  norm=False branch (`:1032-1041`, `comp_count = feature_all_ + alpha -
  feature_count_`; `logged = log(comp_count / comp_count.sum(axis=1))`;
  `feature_log_prob_ = -logged`) and `_joint_log_likelihood` (`:1044-1049`, `X @
  feature_log_prob_.T`, single-class adds `class_log_prior_` — benign) feeding the
  `_BaseNB` pipeline. ferrolearn `fn fit` computes `weights[[ci,j]] =
  -((total_feature_counts[j] - class_feature_counts[ci,j] + alpha) / (
  complement_total + alpha*n_features)).ln()`, and `fn joint_log_likelihood`
  computes `X @ weights.T`; the delegated `predict_*` value-match the oracle to
  ~1e-12. (No public `feature_log_prob_` accessor — verified indirectly via
  `predict_joint_log_proba` / `predict_proba`.) Single-class `class_log_prior_`
  add omitted = BENIGN (prior is `0.0` and one-column softmax is `[[1.0]]`).
- REQ-2: **`alpha >= 0` validation (R-DEV-2 — THE key fixable divergence).** Mirror
  the inherited `_parameter_constraints` `alpha: Interval(Real, 0, None,
  closed="left")` (`:530`): reject `alpha < 0` at `fit` with an
  `InvalidParameterError`-equivalent. ferrolearn `fn fit` has NO `alpha >= 0`
  guard — `with_alpha(-0.5)` passes through `clamp_alpha` unchanged (force_alpha
  default) and fits on negative-smoothed garbage / NaN.
- REQ-3: **`norm=True` VALUE (R-DEV-1).** Mirror the `norm=True` branch
  `feature_log_prob_ = logged / logged.sum(axis=1, keepdims=True)` (`:1037-1039`).
  ferrolearn `fn fit` / `partial_fit` call `apply_norm_inplace` which divides each
  `weights` row (= `-logged`) by its sum: `(-logged)/sum(-logged)`, the algebraic
  identity of `logged/sum(logged)` (the two minus signs cancel). Value-matches the
  oracle (`norm=True predict_proba(q) = [[0.7192..,0.2807..],[0.1322..,0.8677..]]`,
  `predict(q) = [0,1]`).
- REQ-4: **`class_prior` LENGTH-only validation (R-DEV-1/2 — MATCH).** Mirror the
  explicit branch `if len(class_prior) != n_classes: ValueError; class_log_prior_
  = log(class_prior)` (`:589-591`) — LENGTH-only, NO sum/non-neg check. ferrolearn
  `fn fit` checks ONLY `priors.len() != n_classes` — a deliberate MATCH (sum-0.8
  prior fits on both sides; wrong-length errors on both, type differs —
  `InvalidParameter` vs `ValueError`). For ComplementNB this affects only the
  length-validation decision (`class_prior` is "Not used" in multi-class predict).
- REQ-5: **`force_alpha` / `_check_alpha` floor + `fit_prior` carry (R-DEV-1).**
  Mirror `_check_alpha` (`:604-626`, floor `1e-10` unless `force_alpha`).
  ferrolearn `fn fit` calls `crate::clamp_alpha(self.alpha, self.force_alpha)`
  (`base::check_alpha`); `fit_prior` is stored (matching sklearn, only the
  single-class edge case consults the prior — benign here).
- REQ-6: **`partial_fit` VALUE on same-classes data (R-DEV-1; unseen-label gap).**
  Mirror the shared `partial_fit` (`:628-709`): accumulate counts (re-deriving
  `feature_all_`) then recompute `feature_log_prob_`, so chunked `partial_fit` ==
  whole `fit`. ferrolearn `FittedComplementNB::partial_fit` accumulates
  `feature_counts` then recomputes `weights`; the oracle confirms two-chunk
  `partial_fit` == `fit(X,y)` (`np.allclose == True`). **Sub-gap:** no `classes`
  argument — a NEW label in a later chunk is silently dropped (sklearn binarizes
  against the full `classes` list passed on the first call).
- REQ-7: **negative-feature guard (R-DEV-2 — both reject).** Mirror
  `check_non_negative(X, "ComplementNB (input X)")` → `ValueError` (`:1027`).
  ferrolearn `fn fit` rejects negative `X` with `FerroError::InvalidParameter {
  name: "X" }` — both REJECT (SHIPPED on the reject behavior); the exact
  message/type differs.
- REQ-8: **`sample_weight` (R-DEV-1).** Mirror weighted `feature_count_` /
  `class_count_` / `feature_all_` via `Y *= sample_weight.T` before `_count`
  (`:712`/`:1025-1030`). ferrolearn's `impl Fit` is `fn fit(&self, x, y)` — NO
  `sample_weight` parameter on `fit` or `partial_fit`.
- REQ-9: **fitted-attribute + PyO3 surface (R-DEV-3 / R-DEFER-1/3).** sklearn
  exposes `feature_log_prob_` / `feature_all_` / `feature_count_` / `class_count_`
  / `class_log_prior_` / `classes_` / `n_features_in_` (`:937-970`; NO `coef_`/
  `intercept_` — deprecated, removed by 1.5.2). ferrolearn `FittedComplementNB`
  exposes ONLY `classes()`; `weights`/`feature_counts`/`class_counts` are private
  fields with no accessor (and there is no `feature_all_` / `class_log_prior_`
  stored). `_RsComplementNB` (the `py_classifier!` macro) exposes ONLY
  `new(alpha, fit_prior, norm)` + `fit` + `predict` — NO `class_prior`/
  `force_alpha` kwargs, NO `predict_proba`/`predict_log_proba`/`predict_joint_log_
  proba`/`score`/`partial_fit` (which the library HAS), NO fitted-attr getters.
- REQ-10: **ferray substrate (R-SUBSTRATE).** `complement.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}`,
  not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (`from
sklearn.naive_bayes import ComplementNB`, run from `/tmp`), never literal-copied
from ferrolearn (R-CHAR-3). The shared count fixture is `X =
[[5,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]]`, `y = [0,0,0,1,1,1]`, query
`q = [[3,1,1],[0,1,4]]`.

- AC-1 (REQ-1, present & matching, norm=False): `ComplementNB().fit(X,y).
  feature_log_prob_` → `[[2.3978952727983707, 1.7047480922384253,
  0.3184537311185346], [0.3184537311185346, 1.7047480922384253,
  2.3978952727983707]]`; the closed form `-log(comp_count / comp_count.sum(axis=1))`
  with `comp_count = feature_all_ + 1 - feature_count_` reproduces it
  (`np.allclose == True`). `predict_joint_log_proba(q)` → `[[9.216887641752072,
  5.058004558392399], [2.9785630167125636, 11.296329183431908]]`;
  `predict_log_proba(q)` → `[[-0.015504186535965303, -4.174387269895638],
  [-8.318010277546872, -0.00024411082752706648]]`; `predict_proba(q)` →
  `[[0.9846153846153846, 0.015384615384615375], [0.0002440810349035878,
  0.9997559189650967]]`; `predict(q)` → `[0,1]`. ferrolearn matches to ~1e-12
  (verified indirectly via `predict_joint_log_proba` / `predict_proba` —
  `feature_log_prob_` has no public accessor).
- AC-2 (REQ-2 pin): `ComplementNB(alpha=-0.5).fit(X,y)` → `InvalidParameterError(
  "The 'alpha' parameter of ComplementNB must be a float in the range [0.0, inf)
  or an array-like. Got -0.5 instead.")`. ferrolearn `ComplementNB::new().
  with_alpha(-0.5).fit(&X,&y)` → **succeeds** (no guard; `clamp_alpha(-0.5, true)
  = -0.5`), producing `weights` of negative-smoothed complement counts. FAILS the
  reject contract until the `alpha >= 0` guard lands.
- AC-3 (REQ-3, present & matching, norm=True): `ComplementNB(norm=True).fit(X,y).
  feature_log_prob_` → `[[0.5423756186860554, 0.3855939046715139,
  0.07203047664243067], [0.07203047664243067, 0.3855939046715139,
  0.5423756186860554]]`; `predict_joint_log_proba(q)` → `[[2.0847512373721107,
  1.1440609532848613], [0.6737158112412366, 2.5550963794157355]]`;
  `predict_proba(q)` → `[[0.7192390704948571, 0.2807609295051429],
  [0.13223037910101987, 0.8677696208989801]]`; `predict(q)` → `[0,1]`. ferrolearn
  `with_norm(true)` produces the IDENTICAL proba/labels (`(-logged)/sum(-logged) ==
  logged/sum(logged)`).
- AC-4 (REQ-4, present & matching — the MATCH): `ComplementNB(class_prior=[0.5,0.3]).
  fit(X,y)` succeeds (sum 0.8, `class_log_prior_ = log([0.5,0.3])`, NO error).
  `class_prior=[0.5]` → `ValueError("Number of priors must match number of
  classes.")`. ferrolearn `with_class_prior([0.5,0.3]).fit` succeeds (length-only),
  `with_class_prior([0.5]).fit` errors (`InvalidParameter`) — MATCHES the
  accept/reject decisions (wrong-length error TYPE differs).
- AC-5 (REQ-5, present & matching): with `force_alpha=True` default and `alpha=1`,
  the AC-1 `feature_log_prob_` is reproduced; `clamp_alpha(1, true) = 1`.
- AC-6 (REQ-6, present & matching, same classes): `ComplementNB().partial_fit(
  X[:4], y[:4], classes=[0,1]); .partial_fit(X[4:], y[4:]); .feature_log_prob_`
  equals `ComplementNB().fit(X,y).feature_log_prob_` (`np.allclose == True`).
  ferrolearn's two-chunk `partial_fit` reproduces the whole-`fit` `weights` (the
  in-tree `test_complement_nb_partial_fit` exercises the accumulate-then-recompute
  path). (The unseen-label path is the REQ-6 sub-gap.)
- AC-7 (REQ-7 reject — both error): `ComplementNB().fit(X_neg, y)` → `ValueError(
  "Negative values in data passed to ComplementNB (input X)")`. ferrolearn rejects
  with `InvalidParameter { name: "X" }` — both reject; the message/type differ.
- AC-8 (REQ-1 single-class BENIGN): `ComplementNB().fit([[1,2,3],[4,5,6],[7,8,9]],
  [0,0,0])`: `classes_ = [0]`; `class_log_prior_ = [0.0]`; `predict_proba([[2,1,1]])
  = [[1.0]]`; `predict = [0]`. ferrolearn single-class `predict_proba = [[1.0]]`,
  `predict = [0]` — COINCIDE (the omitted `class_log_prior_` add is `+0.0` AND
  one-column softmax is always `[[1.0]]`; NOT an observable divergence).
- AC-9 (REQ-8 pin): `ComplementNB().fit(X, y, sample_weight=...)` weights
  `feature_count_` / `class_count_` / `feature_all_`. ferrolearn has no
  `sample_weight` parameter.
- AC-10 (REQ-9 surface): `hasattr(ComplementNB().fit(X,y), 'feature_log_prob_')` /
  `'feature_all_'` / `'feature_count_'` / `'class_count_'` / `'class_log_prior_'` /
  `'classes_'` all True in sklearn; `hasattr(..., 'coef_')` is **False**.
  ferrolearn `FittedComplementNB` exposes only `classes()`, and
  `ferrolearn.ComplementNB` exposes only `fit`/`predict` (no `predict_proba`/
  `predict_log_proba`/`score`/`partial_fit`/`class_prior`/`force_alpha`/fitted-attr
  getters).

## REQ status table

Binary (R-DEFER-2). `ComplementNB` / `FittedComplementNB` are existing pub APIs
re-exported at the crate root and consumed non-test by the `ferrolearn-python`
binding (`RsComplementNB` `fit`/`predict`) + the in-crate pipeline (the
production-consumer surface; grandfathered S5/R-DEFER-1). Cites use symbol anchors
(ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live oracle =
installed sklearn 1.5.2, run from `/tmp`. Honest underclaim (R-HONEST-3): the
complement-weight + predict VALUES (norm=False AND norm=True), the `class_prior`
length-only check, the floor, same-classes `partial_fit`, the negative-feature
reject, and the BENIGN single-class jll all match and are SHIPPED; the `alpha >= 0`
reject, `sample_weight`, the fitted-attribute/PyO3 surface, and the ferray
substrate are NOT-STARTED. Suggested blocker numbers — the director creates the
real issues (continuing the bayes layer past bernoulli #905-910).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`feature_log_prob_` complement-weight + `_joint_log_likelihood` / predict VALUE, norm=False) | SHIPPED | impl `fn fit` for `ComplementNB` sets `weights[[ci,j]] = -((total_feature_counts[j] - class_feature_counts[ci,j] + alpha) / (complement_total + alpha*n_features)).ln()`, the algebraic identity of `_update_feature_log_prob`'s `-logged` (`naive_bayes.py:1032-1041`, `comp_count = feature_all_ + alpha - feature_count_`; `logged = log(comp_count / comp_count.sum(axis=1))`; `feature_log_prob_ = -logged`); `fn joint_log_likelihood` computes `X @ weights.T`, mirroring `:1046` (the single-class `+ class_log_prior_` add `:1047-1048` is BENIGN — see AC-8); `predict`/`predict_proba`/`predict_log_proba`/`predict_joint_log_proba` delegate to `BaseNB` (`.design/bayes/base.md`). Non-test consumer: `RsComplementNB::fit`/`predict` (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro) → `FittedComplementNB`, surfaced as `ferrolearn.ComplementNB`; plus `impl PipelineEstimator`. Live oracle (AC-1): `feature_log_prob_` → `[[2.3979..,1.7047..,0.3185..],[0.3185..,1.7047..,2.3979..]]` (`np.allclose(-logged) == True`); `predict_joint_log_proba(q)` → `[[9.2169..,5.0580..],[2.9786..,11.2963..]]`; `predict_proba(q)` → `[[0.9846..,0.0154..],[0.000244..,0.99976..]]`; `predict(q)` → `[0,1]`; ferrolearn matches to ~1e-12 (verified by scratch integration run). Single-class (AC-8): `predict_proba = [[1.0]]`, `predict = [0]` on both. In-tree `test_complement_nb_fit_predict` / `test_complement_nb_predict_proba_sums_to_one` / `test_complement_nb_imbalanced_data` / `test_complement_nb_three_classes` pin labels + normalization. |
| REQ-3 (`norm=True` VALUE) | SHIPPED | impl `fn fit` / `partial_fit` call `fn apply_norm_inplace` which divides each `weights` row (= `-logged`) by its row sum, the algebraic identity of sklearn's `logged / logged.sum(axis=1, keepdims=True)` (`naive_bayes.py:1037-1039`) — the two minus signs in `(-logged)/sum(-logged)` cancel. Non-test consumer: `RsComplementNB` threads `norm` through `with_norm(norm)` (`extras.rs`); surfaced as `ferrolearn.ComplementNB(norm=...)` (`_extras.py`). Live oracle (AC-3): `ComplementNB(norm=True).fit(X,y)` → `feature_log_prob_` `[[0.5424..,0.3856..,0.0720..],[0.0720..,0.3856..,0.5424..]]`, `predict_proba(q)` → `[[0.7192..,0.2808..],[0.1322..,0.8678..]]`, `predict(q)` → `[0,1]`; ferrolearn `with_norm(true)` produces the IDENTICAL proba/labels (verified by scratch integration run). |
| REQ-4 (`class_prior` LENGTH-only validation — MATCH) | SHIPPED | impl `fn fit` validates ONLY `priors.len() != n_classes` (then carries the priors), mirroring `_update_class_log_prior` (`naive_bayes.py:589-591`, `if len != n_classes: ValueError; class_log_prior_ = log(class_prior)`) — discrete NB has NO sum/non-neg check. Deliberate MATCH. Non-test consumer: `RsComplementNB` builds `ComplementNB` (the `with_class_prior` path is exercised in-crate + pipeline). Live oracle (AC-4): `class_prior=[0.5,0.3]` (sum 0.8) ACCEPTED; ferrolearn `with_class_prior([0.5,0.3]).fit` also succeeds; `class_prior=[0.5]` errors on both. In-tree `test_complement_nb_class_prior` / `test_complement_nb_class_prior_wrong_length`. (Wrong-length error TYPE differs — `InvalidParameter` vs `ValueError` — folded into REQ-9's surface gap. For ComplementNB `class_prior` is "Not used" in multi-class predict, `naive_bayes.py:929` — only the length decision is observable.) |
| REQ-5 (`force_alpha` floor) | SHIPPED | impl `fn fit` / `partial_fit` call `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`, the `_check_alpha` floor `1e-10` unless `force_alpha`, `naive_bayes.py:604-626`). Non-test consumer: `RsComplementNB` passes `fit_prior` through `with_fit_prior`, `alpha` through `with_alpha`. Live oracle (AC-5): with `force_alpha=true` default and `alpha=1` the AC-1 `feature_log_prob_` is reproduced; `clamp_alpha(1, true)=1`. In-tree `test_complement_nb_default`; `base.rs` `test_check_alpha_force_alpha_keeps_value` / `test_check_alpha_floors_when_not_forced`. |
| REQ-6 (`partial_fit` VALUE on same-classes data) | SHIPPED | `FittedComplementNB::partial_fit` accumulates `class_counts`/`feature_counts` for each EXISTING class, then re-derives `total_feature_counts` (the `feature_all_` analog) / `total_all` and recomputes `weights` (same `-log` complement smoothing), re-applying `apply_norm_inplace` when `norm`, mirroring the shared `_BaseDiscreteNB.partial_fit` accumulate-then-recompute (`naive_bayes.py:628-709`, `_count` re-deriving `feature_all_` → `_update_feature_log_prob`). Non-test consumer: in-crate (no PyO3 `partial_fit` — that gap is REQ-9). Live oracle (AC-6): two-chunk `partial_fit(X[:4],y[:4],classes=[0,1])` + `partial_fit(X[4:],y[4:])` yields `feature_log_prob_` identical to `fit(X,y)` (`np.allclose == True`); ferrolearn's `partial_fit` reproduces the whole-`fit` `weights`. In-tree `test_complement_nb_partial_fit` / `test_complement_nb_partial_fit_shape_mismatch`. **Sub-gap (NOT-STARTED, folded into #915):** no `classes` argument — a NEW label in a later chunk is silently dropped (sklearn binarizes against the full `classes` list from the first call). |
| REQ-7 (negative-feature guard — both reject) | SHIPPED | impl `fn fit` (and `partial_fit`) reject any `x[i,j] < 0` with `FerroError::InvalidParameter { name: "X", reason: "ComplementNB requires non-negative feature values" }`, mirroring `check_non_negative(X, "ComplementNB (input X)")` → `ValueError` (`naive_bayes.py:1027`; ComplementNB DOES guard non-negativity, unlike BernoulliNB). Both REJECT. Non-test consumer: `RsComplementNB::fit` (`extras.rs`) maps the `FerroError` to a `PyErr`. Live oracle (AC-7): `ComplementNB().fit(X_neg,y)` → `ValueError("Negative values in data passed to ComplementNB (input X)")`; ferrolearn errors with `InvalidParameter`. In-tree `test_complement_nb_negative_features_error`. The exact sklearn MESSAGE/TYPE is NOT matched — that sub-item is captured under REQ-9. |
| REQ-2 (`alpha >= 0` validation) | NOT-STARTED | open prereq blocker **#914**. sklearn's inherited `_parameter_constraints` declares `alpha: [Interval(Real, 0, None, closed="left"), "array-like"]` (`naive_bayes.py:530`, merged at `:1000-1001`) → `alpha >= 0` is a HARD reject at `fit` (`_validate_params`); `ComplementNB(alpha=-0.5).fit(X,y)` → `InvalidParameterError("The 'alpha' parameter of ComplementNB must be a float in the range [0.0, inf) … Got -0.5 instead.")`. ferrolearn `fn fit` has NO `alpha >= 0` guard: `with_alpha(-0.5)` → `clamp_alpha(-0.5, force_alpha=true)` returns `-0.5` unchanged (the `1e-10` floor only fires when `force_alpha=false`), so `fit` proceeds and computes `-log((complement_count - 0.5)/denom)` — garbage/NaN, no error. Pin (AC-2): `with_alpha(-0.5).fit(&X,&y)` SUCCEEDS in ferrolearn (verified by scratch run, `is_ok=true`) vs sklearn raises. **THE single-file fixable divergence** (an `alpha < 0` reject before/around `clamp_alpha` in `complement.rs` `fn fit`) — same class as multinomial #900 / bernoulli #907; the critic should pin this FIRST. Distinct from the `force_alpha` floor (REQ-5): `>= 0` is a hard reject regardless of `force_alpha`. |
| REQ-8 (`sample_weight`) | NOT-STARTED | open prereq blocker **#915**. sklearn `fit(X, y, sample_weight=None)` (`:712`) weights the binarized `Y` so `feature_count_ = Y.T @ X` / `class_count_ = Y.sum(axis=0)` / `feature_all_ = feature_count_.sum(axis=0)` become weighted (`:1025-1030`). ferrolearn's `impl Fit<Array2<F>, Array1<usize>>` has signature `fn fit(&self, x, y)` — NO `sample_weight` parameter on `fit` or `partial_fit`; also no `classes` argument on `partial_fit` (the unseen-label sub-gap of REQ-6). Pin (AC-9): `ComplementNB().fit(X,y,sample_weight=...)` weights the counts; ferrolearn cannot pass weights. |
| REQ-9a (Rust fitted accessors) | SHIPPED | `FittedComplementNB<F>` gains `#[must_use]` getters `feature_log_prob()` (`&weights`, sklearn `feature_log_prob_` `:1042`), `feature_count()` (`&feature_counts`, `:961`), `class_count()` (`class_counts` cast to `F`, `:951`), `feature_all()` (DERIVED `feature_counts.sum_axis(Axis(0))`, `:1029`), `class_log_prior()` (DERIVED empirical `ln(class_count_[i]/Σ)`, `:600`). `coef_`/`intercept_` intentionally NOT exposed — gone in sklearn 1.5.2. Verification (live sklearn 1.5.2, R-CHAR-3, `X=[[5,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]]`, `y=[0,0,0,1,1,1]`): `feature_log_prob_=[[2.3978952728,1.7047480922,0.3184537311],[0.3184537311,1.7047480922,2.3978952728]]`, `feature_all_=[16,6,16]`, `feature_count_=[[15,3,1],[1,3,15]]`, `class_count_=[3,3]`, `class_log_prior_=[-0.6931471806,-0.6931471806]`. Tests `complement_feature_log_prob_and_count_match_sklearn`, `complement_feature_all_class_count_prior_match_sklearn`. |
| REQ-9b (PyO3 surface + sample_weight) | NOT-STARTED | open prereq blocker **#916** (multi-file `ferrolearn-python`). `_RsComplementNB` (`extras.rs`, the `py_classifier!` macro) exposes ONLY `new(alpha=1.0, fit_prior=true, norm=false)` + `fit` + `predict` — NO `class_prior`/`force_alpha` kwargs, NO `predict_proba`/`predict_log_proba`/`predict_joint_log_proba`/`score`/`partial_fit` (which the library HAS), NO fitted-attr getters (the Rust accessors now exist — REQ-9a — but are not surfaced to Python). Also subsumes the negative-feature MESSAGE/TYPE-parity sub-item (REQ-7) and the `class_prior` wrong-length TYPE sub-item (REQ-4). |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker **#917**. `complement.rs` imports `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}` (the wrong substrate); not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`complement.rs` follows the unfitted/fitted split (CLAUDE.md naming) for a single
estimator that delegates its entire prediction pipeline to the shared `BaseNB<F>`
trait (`base.rs`):

- `ComplementNB<F>` (`alpha`, `class_prior: Option<Vec<F>>`, `fit_prior`,
  `force_alpha`, `norm`) → `Fit<Array2<F>, Array1<usize>>` →
  `FittedComplementNB<F>` (`classes`, `weights`, `feature_counts`, `class_counts`,
  plus `alpha`/`norm` for `partial_fit`).

Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2).

**Construction.** `pub fn new` sets `alpha=1.0`, `class_prior=None`,
`fit_prior=true`, `force_alpha=true`, `norm=false` (matching sklearn defaults
`:1005-1020`). Builder setters `with_alpha`/`with_class_prior`/`with_fit_prior`/
`with_force_alpha`/`with_norm`.

**Fit path (`fn fit`).** Validation rejects empty `X` (`InsufficientSamples`),
`n_samples != y.len()` (`ShapeMismatch`), and negative `X` (`InvalidParameter`,
REQ-7 — ComplementNB guards non-negativity, `:1027`). It computes `alpha =
clamp_alpha(self.alpha, force_alpha)` (the `_check_alpha` floor, REQ-5) — but does
NOT reject `alpha < 0` (REQ-2/#914, the `>= 0` constraint sklearn enforces at
`:530`). Per class it accumulates `class_feature_counts` and `class_counts`,
derives `total_feature_counts` (the `feature_all_` analog) and `total_all`, then
`complement_total = total_all - class_feature_counts.row(ci).sum()`, `denom =
complement_total + alpha*n_features`, and `weights[[ci,j]] = -((
total_feature_counts[j] - class_feature_counts[ci,j] + alpha) / denom).ln()` —
the algebraic identity of `_update_feature_log_prob`'s `-logged` (`:1032-1041`,
REQ-1). When `self.norm`, `apply_norm_inplace` divides each row by its sum
(`(-logged)/sum(-logged) == logged/sum(logged)`, REQ-3). `class_prior` is
validated LENGTH-only (REQ-4). No `class_log_prior_` is computed (ComplementNB's
multi-class jll does not use it; the single-class add is benign — REQ-1). No
`sample_weight` (REQ-8).

**Prediction (delegated to `BaseNB`).** `joint_log_likelihood` computes `X @
weights.T` (`score[i,ci] = sum_j x*weights[ci,j]`), mirroring `X @
feature_log_prob_.T` (`:1046`, REQ-1) — with the `-logged` sign, higher jll wins
so `_BaseNB.predict`'s argmax reproduces sklearn's argmin-on-complement-probability
convention (`:1036`). It does NOT add `class_log_prior_` in the single-class case;
that omission is BENIGN (the prior is `0.0` and one-column softmax is `[[1.0]]` —
AC-8). `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba`
delegate to the `BaseNB` provided methods (the `_BaseNB` pipeline; see
`.design/bayes/base.md`). The VALUES match the oracle to ~1e-12 (REQ-1) for both
`norm=False` and `norm=True` (REQ-3).

**`partial_fit` (`fn partial_fit`).** Accumulates `class_counts`/`feature_counts`
for each EXISTING class, then re-derives `total_feature_counts`/`total_all` and
recomputes `weights` (same `-log` complement smoothing), re-applying
`apply_norm_inplace` when `norm`, so chunked `partial_fit` == whole `fit` (REQ-6,
oracle-confirmed). No `classes` argument (existing classes only — new labels are
dropped) and no `sample_weight` (REQ-8).

**Scoring.** `score` = `correct / n` (mean accuracy), the `ClassifierMixin.score`
analog.

**Consumer wiring.** The non-test production consumers:
- `ferrolearn-python` `RsComplementNB` / `_RsComplementNB` (`extras.rs`, the
  `py_classifier!` macro) — `new(alpha=1.0, fit_prior=true, norm=false)` → `fit`
  (`model.fit(&x_nd, &y_nd)`) → `predict` (`fitted.predict`), registered in
  `lib.rs` (`m.add_class::<extras::RsComplementNB>()`) and surfaced as
  `ferrolearn.ComplementNB` (`_extras.py`, `_RsComplementNB(alpha, fit_prior,
  norm)`). The binding under-exposes (REQ-9), but the `fit`/`predict` path is a
  real non-test consumer of the library `feature_log_prob_`/`joint_log_likelihood`/
  `predict` (REQ-1) and threads the `norm` flag through (REQ-3).
- `impl PipelineEstimator<F> for ComplementNB<F>` — `fit_pipeline` (float labels →
  `usize`) / `predict_pipeline` consume `fit`/`predict` in-crate.

**Missing fitted attributes vs sklearn:** `feature_log_prob_` / `feature_all_` /
`feature_count_` / `class_count_` / `class_log_prior_` accessors, `n_features_in_`.
ferrolearn exposes only `classes()` / `n_classes()`. (NO `coef_`/`intercept_` on
either side — deprecated/removed by 1.5.2.)

**Invariants held vs sklearn:** the `feature_log_prob_` complement-weight VALUE
(norm=False AND norm=True, AC-1/AC-3); the full predict pipeline VALUE to ~1e-12
(AC-1/AC-3); the `class_prior` LENGTH-only accept/reject decisions (AC-4, the
MATCH); the negative-feature reject (AC-7); the `force_alpha` floor (AC-5);
`partial_fit` == `fit` (AC-6); the BENIGN single-class jll (AC-8); `classes_`
ordering; `predict_proba` rows sum to 1.

**Invariants NOT held vs sklearn:** `alpha >= 0` reject (REQ-2/#914 — the key
single-file divergence); `sample_weight` (REQ-8); the fitted-attribute + PyO3
surface (REQ-9, including the negative-feature and wrong-length message/type); the
`partial_fit` unseen-label path (REQ-6 sub-gap); the ferray substrate (REQ-10).

## Verification

Library crate (green at baseline `45f1c66e` for the existing contract):
```
cargo test -p ferrolearn-bayes --lib complement
cargo clippy -p ferrolearn-bayes --all-targets -- -D warnings
cargo fmt --all --check
```
The 15 in-tree `#[test]`s (`test_complement_nb_fit_predict`,
`test_complement_nb_predict_proba_sums_to_one`, `test_complement_nb_has_classes`,
`test_complement_nb_negative_features_error`, `test_complement_nb_single_class`,
`test_complement_nb_partial_fit`, `test_complement_nb_class_prior`,
`test_complement_nb_class_prior_wrong_length`, `test_complement_nb_imbalanced_data`,
`test_complement_nb_three_classes`, …) pin ferrolearn's current behavior. **None
compares against the live sklearn oracle and none exercises `alpha < 0`**, so they
stay green despite the #914 divergence; the SHIPPED REQs (REQ-1 complement-weight +
predict VALUE, REQ-3 norm=True VALUE, REQ-4 class_prior MATCH, REQ-5 floor, REQ-6
partial_fit, REQ-7 reject) value-match the oracle, the rest are NOT-STARTED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin the deterministic single-file one
FIRST**: REQ-2 (`alpha >= 0` — add the reject in `complement.rs` `fn fit`):
```
# REQ-2 (#914) alpha < 0 — sklearn rejects, ferrolearn accepts (THE key fix)
python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; X=np.array([[5.,1.,0.],[4.,2.,0.],[6.,0.,1.],[0.,1.,5.],[1.,0.,4.],[0.,2.,6.]]); y=np.array([0,0,0,1,1,1]);
try: ComplementNB(alpha=-0.5).fit(X,y)
except Exception as e: print(type(e).__name__, '::', e)"  # InvalidParameterError :: The 'alpha' parameter of ComplementNB must be a float in the range [0.0, inf) or an array-like. Got -0.5 instead.   (ferro: with_alpha(-0.5).fit succeeds)
# REQ-1 (present) feature_log_prob_ + predict VALUE (norm=False)
python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; X=np.array([[5.,1.,0.],[4.,2.,0.],[6.,0.,1.],[0.,1.,5.],[1.,0.,4.],[0.,2.,6.]]); y=np.array([0,0,0,1,1,1]); m=ComplementNB().fit(X,y); q=np.array([[3.,1.,1.],[0.,1.,4.]]); print(m.feature_log_prob_.tolist()); print(m.predict_proba(q).tolist()); print(m.predict(q).tolist())"  # flp [[2.398..,1.705..,0.318..],[0.318..,1.705..,2.398..]]; pp [[0.985..,0.015..],[0.000244..,0.99976..]]; pred [0,1]
# REQ-3 (present) norm=True VALUE
python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; X=np.array([[5.,1.,0.],[4.,2.,0.],[6.,0.,1.],[0.,1.,5.],[1.,0.,4.],[0.,2.,6.]]); y=np.array([0,0,0,1,1,1]); m=ComplementNB(norm=True).fit(X,y); q=np.array([[3.,1.,1.],[0.,1.,4.]]); print(m.predict_proba(q).tolist()); print(m.predict(q).tolist())"  # pp [[0.7192..,0.2808..],[0.1322..,0.8678..]]; pred [0,1]  (ferro with_norm(true) IDENTICAL)
# REQ-4 (present, MATCH) class_prior length-only (sum 0.8 accepted)
python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; X=np.array([[5.,1.,0.],[4.,2.,0.],[6.,0.,1.],[0.,1.,5.],[1.,0.,4.],[0.,2.,6.]]); y=np.array([0,0,0,1,1,1]); print(ComplementNB(class_prior=[0.5,0.3]).fit(X,y).class_log_prior_.tolist())"  # [-0.6931471805599453, -1.2039728043259361]  (NO sum check — ferro MATCHES)
# REQ-1 single-class BENIGN: class_log_prior_=[0.0]; predict_proba=[[1.0]]
python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; X=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]); y=np.array([0,0,0]); m=ComplementNB().fit(X,y); print(m.class_log_prior_.tolist(), m.predict_proba(np.array([[2.,1.,1.]])).tolist())"  # [0.0] [[1.0]]  (ferro coincides — omitted add is benign)
# REQ-6 (present) partial_fit == fit
python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; X=np.array([[5.,1.,0.],[4.,2.,0.],[6.,0.,1.],[0.,1.,5.],[1.,0.,4.],[0.,2.,6.]]); y=np.array([0,0,0,1,1,1]); m=ComplementNB(); m.partial_fit(X[:4],y[:4],classes=[0,1]); m.partial_fit(X[4:],y[4:]); f=ComplementNB().fit(X,y); print(np.allclose(m.feature_log_prob_, f.feature_log_prob_))"  # True
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-bayes/tests/divergence_complement.rs`, asserting the live-sklearn
expected values above and FAILING against current `complement.rs` (REQ-2 the
`alpha < 0` reject is the cleanest). REQ-1/REQ-3/REQ-4/REQ-5/REQ-6/REQ-7 already
match and should be guarded by non-regression pins (the norm=True VALUE pin and
the single-class benign pin are the most informative new ones).

ferrolearn-python (REQ-9 binding parity, after #916 lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_naive_bayes.py -q
```
asserting `ferrolearn.ComplementNB` exposes `feature_log_prob_` / `feature_all_` /
`feature_count_` / `class_count_` / `class_log_prior_` / `predict_proba` /
`predict_log_proba` / `score` / `partial_fit` and `class_prior` / `force_alpha`
kwargs, matching `sklearn.naive_bayes.ComplementNB` on the AC fixtures.

## Blockers to open

(Director creates the real issues; the numbers are SUGGESTIONS continuing the
bayes layer past bernoulli #905-910. #913 is this doc's crosslink tracking issue.)

- **#914** — REQ-2 (`alpha >= 0` validation): add an `alpha < 0` reject to
  `complement.rs` `fn fit` (mirroring the inherited `_parameter_constraints`
  `alpha: Interval(Real, 0, None, closed="left")`, `naive_bayes.py:530`)
  before/around `clamp_alpha`. **The cleanest single-file deterministic fix** — the
  critic should pin this FIRST. Same class as multinomial #900 / bernoulli #907.
  (Distinct from the `force_alpha` floor, which only fires for `alpha < 1e-10` when
  `force_alpha=false`.)
- **#915** — REQ-8 (`sample_weight`): add weighted `feature_count_`/`class_count_`/
  `feature_all_` (`Y *= sample_weight.T` before `_count`, `:712`/`:1025-1030`);
  needs a `sample_weight` parameter on `fit`/`partial_fit` (Fit-trait shape,
  R-DEV-1). Also covers the `partial_fit` unseen-label sub-gap (add the `classes`
  argument so a new later-chunk label is represented, REQ-6 sub-item).
- **#916** — REQ-9 (fitted-attribute + PyO3 surface): expose `feature_log_prob_`/
  `feature_all_`/`feature_count_`/`class_count_`/`class_log_prior_` accessors on
  `FittedComplementNB`, and add `class_prior`/`force_alpha` kwargs +
  `predict_proba`/`predict_log_proba`/`score`/`partial_fit` + those getters to
  `_RsComplementNB` (`ferrolearn-python/src/extras.rs`). Also align the
  negative-feature error message with sklearn's "Negative values in data passed to
  ComplementNB (input X)" (REQ-7 sub-item) and the `class_prior` wrong-length
  TYPE (`InvalidParameter` vs `ValueError`, REQ-4 sub-item).
- **#917** — REQ-10 (ferray substrate): migrate `complement.rs` off
  `ndarray`/`num-traits` to `ferray-core` (R-SUBSTRATE).
