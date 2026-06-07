# k-Nearest Neighbors (sklearn.neighbors.KNeighborsClassifier / KNeighborsRegressor)

<!--
tier: 3-component
status: draft
baseline-commit: 992a3bb6
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/neighbors/_classification.py   # KNeighborsClassifier(KNeighborsMixin, ClassifierMixin, NeighborsBase); __init__ n_neighbors=5/weights='uniform'/algorithm='auto'/leaf_size=30/p=2/metric='minkowski'/metric_params=None/n_jobs=None (:193-214); predict (:240-305, argmax-first); predict_proba (:307-...); _get_weights distance + zero-dist (_base.py)
  - sklearn/neighbors/_regression.py       # KNeighborsRegressor(KNeighborsMixin, RegressorMixin, NeighborsBase); __init__ same surface (:178-199); predict weighted mean + multi-output 2D y (:229-270); RegressorMixin.score = r2_score(multioutput='uniform_average')
  - sklearn/neighbors/_base.py             # KNeighborsMixin.kneighbors: k>0 guard (:808), n_neighbors>n_samples_fit ValueError (:828-832); fit does NOT validate n_neighbors vs n_samples (deferred to query); _get_weights
ferrolearn-module: ferrolearn-neighbors/src/knn.rs
parity-ops: KNeighborsClassifier (.__init__, .fit, .predict, .predict_proba, .score), KNeighborsRegressor (.__init__, .fit, .predict, .score)
crosslink-issue: 873
-->

## Summary

`ferrolearn-neighbors/src/knn.rs` mirrors scikit-learn's
`KNeighborsClassifier` (`sklearn/neighbors/_classification.py`) and
`KNeighborsRegressor` (`sklearn/neighbors/_regression.py`), both of which are thin
estimators over the shared `KNeighborsMixin` / `NeighborsBase` machinery in
`sklearn/neighbors/_base.py`. It exposes the unfitted `KNeighborsClassifier<F>` /
`KNeighborsRegressor<F>` (`n_neighbors=5`, `algorithm=Auto`, `weights=Uniform`),
their fitted `FittedKNeighborsClassifier<F>` / `FittedKNeighborsRegressor<F>`, the
`Weights::{Uniform, Distance}` weighting enum (with sklearn's zero-distance
special-casing), and the shared `kneighbors_impl` k-NN search. Both estimators are
re-exported at the crate root (`ferrolearn-neighbors/src/lib.rs`: `pub use
knn::{FittedKNeighborsClassifier, FittedKNeighborsRegressor, KNeighborsClassifier,
KNeighborsRegressor, Weights}`) **and bound into `ferrolearn-python`**
(`_RsKNeighborsClassifier` in `classifiers.rs`, `_RsKNeighborsRegressor` in
`extras.rs`; surfaced as `ferrolearn.KNeighborsClassifier` /
`ferrolearn.KNeighborsRegressor`) — so they carry a genuine non-test production
consumer (the binding + the pipeline integration).

Under honest underclaim (R-HONEST-3), the **behaviors that are genuinely present
and value-match the live sklearn 1.5.2 oracle** are:

- **classifier `predict` value** (uniform + distance weights, argmax-with-smallest-
  label tie-break) — matches sklearn `KNeighborsClassifier.predict`
  (`_classification.py:240-305`) on a tie-free fixture and on the even-`k` vote
  tie.
- **classifier `predict_proba` value** — normalized weighted class-vote shares,
  matching sklearn `predict_proba` (`_classification.py:307`) on a non-trivial
  distance-weighted fixture (`[0.5714, 0.4286]`).
- **classifier `score`** — mean accuracy, the `ClassifierMixin.score` analog.
- **regressor `predict` value** (uniform + distance weights, zero-distance special
  case) — weighted mean of neighbor targets, matching
  `KNeighborsRegressor.predict` (`_regression.py:229-270`) for **1-D `y`**.
- **regressor `score`** — R² (`RegressorMixin.score` / `r2_score` with
  `multioutput='uniform_average'`).
- the **shared k-NN search value** (`kneighbors_impl`) — `(distances, indices)`
  nearest-first, the kernel both `predict` paths and the `kneighbors` method
  consume.
- the **zero-distance distance-weight special case** — when a query coincides with
  a training point, that point takes all weight (clf proba `[[0,1]]`, reg `20.0`),
  matching sklearn's `_get_weights` zero-distance branch.

Everything else diverges from the `KNeighbors*` contract:

1. **Fit-time `InsufficientSamples` guard has no sklearn analog (the #872-class
   divergence, present in BOTH `fit` methods).** ferrolearn `fit` rejects
   `n_samples < n_neighbors` (clf `fn fit`, reg `fn fit` — the
   `InsufficientSamples` guard). sklearn does NOT validate `n_neighbors` vs
   `n_samples` at `fit` (`_base.py` `_fit` has no such check); it **succeeds** and
   defers the `n_neighbors <= n_samples_fit` `ValueError` to `kneighbors` /
   `predict` query time (`_base.py:828-832`). A `KNeighborsClassifier(n_neighbors=5)
   .fit(X_3rows, y_3)` that sklearn accepts is rejected by ferrolearn.
2. **Regressor is 1-D `y` only — no multi-output 2-D `y`.** `FittedKNeighbors
   Regressor` stores `y_train: Array1<F>`; sklearn's `KNeighborsRegressor.predict`
   supports `(n_samples, n_outputs)` `y`, returning a 2-D prediction
   (`_regression.py:253-270`). MISSING-surface divergence (R-DEV-3).
3. **Missing constructor params**: `weights` is exposed (`Weights` enum) but
   `leaf_size=30` (`_classification.py:199`), `p=2` (`:200`), `metric='minkowski'`
   (`:201`), `metric_params=None` (`:202`), `n_jobs=None` (`:203`) are ABSENT, and
   `weights` cannot be a **callable** (sklearn accepts `'uniform'|'distance'|
   callable`, `:190`). ferrolearn is **Euclidean-only** with the two named
   weighting schemes.
4. **The PyO3 binding under-exposes the surface.** `_RsKNeighborsClassifier`
   exposes only `n_neighbors` (no `weights`/`algorithm` param), `fit`, `predict`,
   `classes_` — **no `predict_proba`, no `score`**; `_RsKNeighborsRegressor` (via
   `py_regressor!`) exposes `n_neighbors`, `fit`, `predict`, `score` — no `weights`
   param. So `import ferrolearn` cannot reach distance-weighting or
   `predict_proba`.
5. The crate is on the **`ndarray` substrate** (`use ndarray::{Array1, Array2}` +
   `num_traits::Float`), not ferray (R-SUBSTRATE).

`KNeighborsClassifier` / `KNeighborsRegressor` (and the fitted types) are existing
pub APIs (grandfathered per S5/R-DEFER-1); their non-test production consumers are
the `ferrolearn-python` bindings (`_RsKNeighborsClassifier.predict` →
`fitted.predict`; `_RsKNeighborsRegressor` `py_regressor!` → `predict`/`score`) and
the in-crate pipeline integration (`impl PipelineEstimator for KNeighbors*`).

## Algorithm (sklearn — the contract)

### Construction (`_classification.py:193-214`, `_regression.py:178-199`)

`KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto',
leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)` and the
identical `KNeighborsRegressor(...)` signature — all args after `n_neighbors` are
keyword-only (`*`). `weights` is `{'uniform', 'distance'}`, a callable, or `None`
(`_classification.py:190`, `_regression.py:173`). `fit(X, y)` forwards to
`self._fit(X, y)` (`NeighborsBase._fit`), which validates `X`, records
`n_samples_fit_`/`classes_` (classifier) / `_y` (regressor), and builds the chosen
spatial index — but does **NOT** check `n_neighbors` against `n_samples`.

### Classifier `predict` (`_classification.py:240-305`)

For `weights='uniform'` (the standard path), sklearn 1.5.2 computes
`probabilities = self.predict_proba(X)` (or the `ArgKminClassMode` fast path) and
returns `self.classes_[np.argmax(probabilities, axis=1)]` (`:259-268`); for
weighted/non-uniform it calls `weighted_mode` (`:297`). **`np.argmax` returns the
FIRST (smallest-index, hence smallest-label since `classes_` is sorted) maximum on
ties** — the tie-break is "smallest class label wins". `_get_weights(neigh_dist,
'distance')` weights each neighbor by `1/dist`; for a row containing a
**zero-distance** neighbor, `_get_weights` puts all weight on the coincident
point(s) (`_base.py` `_get_weights`).

### Classifier `predict_proba` (`_classification.py:307`)

Returns, per query row, the normalized weighted class-vote shares with classes in
`classes_` (lexicographic) order. Uniform → vote counts / `k`; distance → `(1/dist)`
weights normalized per row; a zero-distance neighbor takes all the weight. A row
where every neighbor weight is 0 raises a `ValueError` (`:383-387`) — the
all-zero-weight case only arises with user callables, not the built-in schemes.

### Regressor `predict` (`_regression.py:229-270`)

`_y` is reshaped to 2-D; uniform → `np.mean(_y[neigh_ind], axis=1)`; distance →
per-output `np.sum(_y[neigh_ind,j]*weights, axis=1) / np.sum(weights, axis=1)`
(`:257-265`). For a zero-distance row, `_get_weights` puts all weight on the
coincident neighbor. **If `_y.ndim == 1` the result is `ravel`ed back to 1-D
(`:267-268`); otherwise the prediction is 2-D `(n_queries, n_outputs)`** — the
multi-output contract.

### `score`

Classifier `score` is `ClassifierMixin.score` = mean accuracy. Regressor `score`
is `RegressorMixin.score` = `r2_score(y, predict(X),
multioutput='uniform_average')` — `1 - SS_res/SS_tot`; for a constant `y`
(`SS_tot == 0`) sklearn returns `1.0` if `SS_res == 0` else `-inf`.

### k-NN search / error timing (`_base.py`)

`KNeighborsMixin.kneighbors(X, n_neighbors=None, return_distance=True)`:
`n_neighbors <= 0` → `ValueError("Expected n_neighbors > 0. ...")` (`:808`);
`n_neighbors > n_samples_fit_` → `ValueError("Expected n_neighbors <=
n_samples_fit, but n_neighbors = {k}, n_samples_fit = {n}, n_samples = {nq}")`
(`:828-832`). The neighbor rows are sorted nearest-first. **The `n_neighbors` vs
`n_samples` check lives at query time** — `fit` does not perform it.

### Edge cases (live oracle, sklearn 1.5.2, run from /tmp)

- `KNeighborsClassifier(n_neighbors=5).fit([[0],[1],[2]], [0,1,0])` → **fit OK** (no
  error); `predict` then raises `ValueError("Expected n_neighbors <= n_samples_fit,
  but n_neighbors = 5, n_samples_fit = 3, n_samples = 1")`. Same for
  `KNeighborsRegressor`.
- `KNeighborsRegressor().fit(X, Y_2col).predict(Xq)` → 2-D `(n_queries, 2)`.
- `KNeighborsClassifier(n_neighbors=2)` on `X=[[0],[1]]`, `y=[0,1]`, query `0.5`
  (vote tie 1–1) → predicts `0` (smallest label).

## ferrolearn (what exists)

All in `ferrolearn-neighbors/src/knn.rs`, generic over `F: Float + Send + Sync +
'static`; `ndarray` substrate.

- **`pub struct KNeighborsClassifier<F> { pub n_neighbors, pub algorithm: Algorithm,
  pub weights: Weights, _marker }`** and the analogous **`KNeighborsRegressor<F>`**.
  `pub fn new` sets `n_neighbors=5`, `algorithm=Auto`, `weights=Uniform`; builder
  setters `with_n_neighbors`/`with_algorithm`/`with_weights`; `impl Default → new()`.
  **No `leaf_size`/`p`/`metric`/`metric_params`/`n_jobs` fields.**
- **`pub enum Weights { Uniform, Distance }`** — sklearn's `'uniform'`/`'distance'`;
  **no callable variant**.
- **`impl Fit<Array2<F>, Array1<usize>> for KNeighborsClassifier<F>` / `fn fit`** —
  rejects `n_samples != y.len()` (`ShapeMismatch`), `n_neighbors == 0`
  (`InvalidParameter`), and **`n_samples < n_neighbors` (`InsufficientSamples`,
  the fit-time guard with no sklearn analog)**; computes sorted-deduped `classes`;
  builds the spatial index. The regressor `fn fit` is identical but stores
  `y_train: Array1<F>` and has no `classes` (same `InsufficientSamples` guard).
- **`impl Predict for FittedKNeighborsClassifier<F>` / `fn predict`** — per query
  row, `find_neighbors` → `weighted_vote`. `fn class_score_vec` builds per-class
  weighted vote sums in `self.classes` order (uniform → counts; distance →
  `1/dist`, with the **zero-distance branch** giving coincident neighbors weight 1
  each via `has_zero_dist`/`eps=1e-15`); `fn weighted_vote` argmaxes, **tie-breaking
  to the smallest class label** (first index in sorted `self.classes`).
- **`pub fn predict_proba(&self, x) -> Result<Array2<F>>`** — normalizes
  `class_score_vec` per row (`scores[ci]/total`); when `total == 0` falls back to
  uniform `1/n_classes` (this fallback has no built-in-weights analog; sklearn
  *raises* on all-zero weights, but that only arises with callables — N/A here).
- **`pub fn score(&self, x, y) -> Result<F>`** (classifier) — mean accuracy
  (`correct / n`); the `ClassifierMixin.score` analog.
- **`impl HasClasses for FittedKNeighborsClassifier<F>`** — `classes()` /
  `n_classes()`, sorted unique labels.
- **`impl Predict for FittedKNeighborsRegressor<F>` / `fn predict`** — per query
  row, `find_neighbors` → `fn weighted_mean` (uniform → mean; distance →
  `Σ(y/d)/Σ(1/d)`, with the **zero-distance branch** averaging the coincident
  targets).
- **`pub fn score(&self, x, y) -> Result<F>`** (regressor) → `fn r2_score`
  (`1 - SS_res/SS_tot`; constant-`y` → `1.0` if `SS_res==0` else
  `F::neg_infinity()`), the `RegressorMixin.score` analog.
- **`pub fn kneighbors(&self, x, n_neighbors: Option<usize>)`** (both fitted types)
  → shared **`pub(crate) fn kneighbors_impl`** — validates feature count, `k == 0`
  (`InvalidParameter`), `k > n_train` (`InsufficientSamples`); per row
  `find_neighbors` fills `(n_queries, k)` distance + index arrays. **`pub fn
  n_samples_fit`** (both) is the `n_samples_fit_` analog.
- **Pipeline integration**: `impl PipelineEstimator<F> for KNeighbors{Classifier,
  Regressor}<F>` (`fn fit_pipeline`) + `FittedKNeighbors{Classifier,Regressor}
  Pipeline` wrappers (`fn predict_pipeline`).
- **Spatial index**: `fn build_spatial_index` — `Algorithm::Auto` → `KdTree` if
  `n_features <= 15` else `BallTree`; `KdTree`/`BallTree`/`BruteForce` map directly
  (the `enum Algorithm { Auto, BruteForce, KdTree, BallTree }`). Distance is
  Euclidean throughout (the kd/ball backends + `kdtree::brute_force_knn`).

**Consumers (non-test).** Crate re-export (`ferrolearn-neighbors/src/lib.rs`) plus:
- **`ferrolearn-python`** — `_RsKNeighborsClassifier` (`classifiers.rs`): `new(
  n_neighbors=5)` → `KNeighborsClassifier::new().with_n_neighbors(...)`, `fit`,
  `predict` (`fitted.predict`), `classes_` getter (no `predict_proba`/`score`/
  `weights`). `_RsKNeighborsRegressor` (`extras.rs`, `py_regressor!` macro):
  `new(n_neighbors=5)`, `fit`, `predict`, `score` (no `weights`). Both surfaced in
  `ferrolearn/__init__.py` as `KNeighborsClassifier`/`KNeighborsRegressor`.
- **Pipeline** (`impl PipelineEstimator`) consumes `fit`/`predict` in-crate.

These are existing pub APIs (grandfathered, S5/R-DEFER-1) and the non-test
production consumers that back the value REQs.

## Requirements

- REQ-1: **classifier `predict` value + tie-break (R-DEV-1/3).** Mirror
  `KNeighborsClassifier.predict` (`_classification.py:240-305`): weighted-vote
  argmax with **smallest-label tie-break** (`np.argmax` first-max, `:268`), uniform
  and distance weights. ferrolearn `fn predict`/`fn weighted_vote` value-matches the
  live oracle on a tie-free fixture and on the even-`k` vote tie.
- REQ-2: **classifier `predict_proba` value (R-DEV-3).** Mirror
  `predict_proba` (`_classification.py:307`): normalized weighted class-vote shares
  in `classes_` order, with the zero-distance branch. ferrolearn `pub fn
  predict_proba` value-matches on a non-trivial distance-weighted fixture.
- REQ-3: **classifier `score` accuracy (R-DEV-1).** Mirror `ClassifierMixin.score`
  (mean accuracy). ferrolearn `pub fn score` value-matches.
- REQ-4: **regressor `predict` value, 1-D y, uniform + distance (R-DEV-1/3).**
  Mirror `KNeighborsRegressor.predict` (`_regression.py:229-270`) for 1-D `y`:
  weighted mean of neighbor targets, zero-distance special case. ferrolearn `fn
  predict`/`fn weighted_mean` value-matches the live oracle.
- REQ-5: **regressor `score` R² (R-DEV-1).** Mirror `RegressorMixin.score` =
  `r2_score(multioutput='uniform_average')`, constant-`y` → `1.0`/`-inf`.
  ferrolearn `pub fn score`/`fn r2_score` value-matches.
- REQ-6: **shared k-NN search value + DEFAULT-backend exact-tie SET/ORDER
  (R-DEV-1/3).** Mirror the `KNeighborsMixin` k-NN kernel both predict paths
  consume: nearest-first `(distances, indices)`. The exact-tie SET/ORDER is
  **NOT algorithm-invariant** — sklearn's default `algorithm='auto'` resolves
  (`_base.py:607-640`, euclidean/p=2) to `brute` (`ArgKmin` `parallel_on_Y`)
  when `n_features > 15` or `k >= n_samples // 2`, else `kd_tree`
  (`KDTree.query`), and these two backends return genuinely different tie sets
  (#2143). ferrolearn `fn find_neighbors` routes by the same rule:
  `fn brute_parallel_on_y` (the `_argkmin.pyx.tp` double-heap) for brute,
  `sk_kdtree::SkKdTree` (bit-exact `KDTree` build + depth-first query, with
  `partition_node_indices` delegated to the libstdc++ `std::nth_element` port
  `introselect::nth_element`) for kd_tree. Both reuse `heap_push` +
  `simultaneous_sort`. ferrolearn `pub(crate) fn kneighbors_impl` / `pub fn
  kneighbors` value-matches the live oracle bit-for-bit across both regimes.
- REQ-7: **`HasClasses` / `classes_` (R-DEV-3).** Expose sorted unique class labels
  (`classes_` lexicographic order, `_classification.py:120`). ferrolearn `impl
  HasClasses` (`classes()`/`n_classes()`) provides this and the binding's
  `classes_` getter consumes it.
- REQ-8: **fit-timing parity — the #872-class divergence (R-DEV-2).** sklearn `fit`
  does NOT check `n_neighbors` vs `n_samples` (no such check in `_base.py` `_fit`);
  it succeeds and defers the `ValueError` to query time (`_base.py:828-832`).
  ferrolearn `fn fit` (BOTH clf + reg) rejects `n_samples < n_neighbors` with
  `InsufficientSamples` at fit time.
- REQ-9: **regressor multi-output 2-D y (R-DEV-3).** Mirror
  `KNeighborsRegressor.predict` for `(n_samples, n_outputs)` `y` → 2-D prediction
  (`_regression.py:253-270`). ferrolearn `y_train: Array1<F>` is **1-D only**.
- REQ-10: **constructor params + callable weights (R-DEV-2).** Match
  `leaf_size=30`, `p=2`, `metric='minkowski'`, `metric_params=None`, `n_jobs=None`
  (`_classification.py:199-203`) and `weights` as a **callable**
  (`:190`). ferrolearn has only `weights∈{Uniform,Distance}` and
  `n_neighbors`/`algorithm`; Euclidean-only, no callable weights.
- REQ-11: **PyO3 binding `predict_proba` / `score` / `weights` (R-DEFER-1/3).**
  Expose distance weighting, `predict_proba`, and classifier `score` through
  `ferrolearn.KNeighborsClassifier`, and `weights` through
  `ferrolearn.KNeighborsRegressor`. The bindings exist but under-expose the surface
  (`_RsKNeighborsClassifier`: no `predict_proba`/`score`/`weights`;
  `_RsKNeighborsRegressor`: no `weights`).
- REQ-12: **ferray substrate (R-SUBSTRATE).** `knn.rs` imports `ndarray::{Array1,
  Array2}` + `num_traits::Float`, not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (`from sklearn.neighbors
import KNeighborsClassifier, KNeighborsRegressor`, run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). ferrolearn values verified by a
throwaway `cargo run --example` probe (since deleted).

- AC-1 (REQ-1, present & matching): clf `X=[[0,0],[1,0],[0,1],[5,5],[6,5],[5,6]]`,
  `y=[0,0,0,1,1,1]`, query `[[0.2,0.1],[5.2,5.1]]`, `k=3`. uniform `predict` →
  `[0,1]`; distance `predict` → `[0,1]`. ferrolearn matches both. Tie (AC-1b):
  `X=[[0],[1]]`, `y=[0,1]`, `k=2`, query `0.5` → sklearn `[0]` (smallest label);
  ferrolearn `[0]`.
- AC-2 (REQ-2, present & matching): `X=[[0],[1],[2],[10],[11]]`, `y=[0,1,0,1,1]`,
  `k=3`, query `1.5`. `weights='distance'` `predict_proba` →
  `[[0.5714285714285715, 0.4285714285714286]]`; uniform → `[[0.6666666666666666,
  0.3333333333333333]]`. ferrolearn `predict_proba` matches both to full precision.
- AC-3 (REQ-3, present): clf `X=[[0],[1],[5],[6]]`, `y=[0,0,1,1]`, `k=1` →
  `score(X,y) == 1.0`; ferrolearn `score` → `1.0`.
- AC-4 (REQ-4, present & matching): reg `X=[[0],[1],[2],[3],[100]]`,
  `y=[0,10,20,30,1000]`, `k=3`, query `1.0`. uniform `predict` → `[10.0]`; distance
  `predict` → `[10.0]`. ferrolearn matches. Zero-dist (AC-4b): `X=[[0],[1],[2]]`,
  `y=[10,20,30]`, `k=3`, `weights='distance'`, query `1.0` → `[20.0]`; ferrolearn
  `[20.0]` (coincident point takes all weight). Clf zero-dist proba (REQ-2):
  `y=[0,1,0]`, query `1.0` → `[[0.0, 1.0]]`; ferrolearn `[[0.0, 1.0]]`.
- AC-5 (REQ-5, present): reg `X=[[0],[1],[5],[6]]`, `y=[0,10,50,60]`, `k=1` →
  `score == 1.0`; ferrolearn R² → `1.0`.
- AC-6 (REQ-6, present): the shared k-NN search backs AC-1/2/4; `kneighbors`
  returns nearest-first `(d, i)` (in-crate `kneighbors_impl` tests + the
  `nearest_neighbors.md` AC-1 oracle parity for the same kernel).
- AC-7 (REQ-8 pin): `KNeighborsClassifier(n_neighbors=5).fit([[0],[1],[2]],
  [0,1,0])` → sklearn **fit OK** (no error); `.predict([[0.5]])` → `ValueError(
  "Expected n_neighbors <= n_samples_fit, but n_neighbors = 5, n_samples_fit = 3,
  n_samples = 1")`. ferrolearn `fit` → `Err(InsufficientSamples { required: 5,
  actual: 3, ... })`. Same for `KNeighborsRegressor`.
- AC-8 (REQ-9 pin): `KNeighborsRegressor(n_neighbors=3).fit(X, Y_2col).predict(Xq)`
  → 2-D `(1, 2)` value `[[1.0, 200.0]]`. ferrolearn `Fit<Array2<F>, Array1<F>>`
  cannot accept a 2-D `y`; no 2-D `predict`.
- AC-9 (REQ-10 surface): `KNeighborsClassifier()` → `n_neighbors==5`,
  `weights=='uniform'`, `algorithm=='auto'`, `leaf_size==30`, `p==2`,
  `metric=='minkowski'`, `metric_params==None`, `n_jobs==None` (ditto
  `KNeighborsRegressor()`). ferrolearn `new()` has `n_neighbors`/`algorithm`/
  `weights` only — REQ-10 FAILS on the missing fields + callable weights.
- AC-10 (REQ-11 binding): `ferrolearn.KNeighborsClassifier` has no
  `predict_proba`/`score`; its `_RsKNeighborsClassifier` exposes only `fit`/
  `predict`/`classes_`. `ferrolearn.KNeighborsRegressor` has no `weights` knob.

## REQ status table

Binary (R-DEFER-2). `KNeighbors{Classifier,Regressor}` / their fitted types are
existing pub APIs re-exported at the crate root and consumed non-test by the
`ferrolearn-python` bindings + pipeline integration (the production-consumer
surface; grandfathered S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) /
`file:line` (sklearn 1.5.2). Live oracle = installed sklearn 1.5.2, run from
`/tmp`. Honest underclaim (R-HONEST-3): seven REQs are SHIPPED (clf predict +
tie-break, clf predict_proba, clf score, reg predict, reg score, shared k-NN
value, HasClasses — every behavior that value-matches the oracle with a non-test
consumer); the rest are NOT-STARTED with open prereq blockers (suggested numbers
— the director creates the real issues).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (clf `predict` value + tie-break) | SHIPPED | impl `fn predict` for `FittedKNeighborsClassifier` (per-row `find_neighbors` → `fn weighted_vote` over `fn class_score_vec`, argmax tie-break to smallest label since `self.classes` is sorted) mirrors `KNeighborsClassifier.predict` (`_classification.py:240-305`; `np.argmax` first-max `:268`). Non-test consumer: `_RsKNeighborsClassifier::predict` (`ferrolearn-python/src/classifiers.rs`) → `fitted.predict`, surfaced as `ferrolearn.KNeighborsClassifier`. Live oracle (AC-1): `X=[[0,0],[1,0],[0,1],[5,5],[6,5],[5,6]]`, `k=3`, query `[[0.2,0.1],[5.2,5.1]]` → uniform `[0,1]`, distance `[0,1]`; ferrolearn matches. Tie (AC-1b): `X=[[0],[1]]`,`y=[0,1]`,`k=2`,query`0.5` → sklearn `[0]`, ferrolearn `[0]` (smallest label). Verified throwaway probe. |
| REQ-2 (clf `predict_proba` value) | SHIPPED | impl `pub fn predict_proba` in `knn.rs` (`class_score_vec` weighted sums, per-row `scores[ci]/total`, zero-dist branch in `class_score_vec`) mirrors `predict_proba` (`_classification.py:307`, classes in `classes_` order). Non-test consumer: same fitted type behind `_RsKNeighborsClassifier` (predict path); `predict_proba` itself is a pub method of the bound type (binding getter not yet wired — REQ-11). Live oracle (AC-2): `X=[[0],[1],[2],[10],[11]]`,`y=[0,1,0,1,1]`,`k=3`,query`1.5`,distance → `[[0.5714285714285715,0.4285714285714286]]`, uniform → `[[0.6666666666666666,0.3333333333333333]]`; ferrolearn matches both to full precision. Zero-dist (AC-4b): query`1.0` on `y=[0,1,0]` → `[[0.0,1.0]]`; ferrolearn matches. Verified probe. |
| REQ-3 (clf `score` accuracy) | SHIPPED | impl `pub fn score` for `FittedKNeighborsClassifier` (`correct/n` over `predict`) mirrors `ClassifierMixin.score` (mean accuracy). Non-test consumer: the fitted type behind `_RsKNeighborsClassifier` (predict path; score not yet wired in the binding — REQ-11), and in-crate pipeline. Live oracle (AC-3): `X=[[0],[1],[5],[6]]`,`y=[0,0,1,1]`,`k=1` → `score==1.0`; ferrolearn `score` → `1.0`. Verified probe. |
| REQ-4 (reg `predict` value, 1-D y) | SHIPPED | impl `fn predict` for `FittedKNeighborsRegressor` (per-row `find_neighbors` → `fn weighted_mean`: uniform mean, distance `Σ(y/d)/Σ(1/d)`, zero-dist averages coincident targets) mirrors `KNeighborsRegressor.predict` (`_regression.py:229-270`) for 1-D `y`. Non-test consumer: `_RsKNeighborsRegressor::predict` (`ferrolearn-python/src/extras.rs`, `py_regressor!`), surfaced as `ferrolearn.KNeighborsRegressor`. Live oracle (AC-4): `X=[[0],[1],[2],[3],[100]]`,`y=[0,10,20,30,1000]`,`k=3`,query`1.0` → uniform `[10.0]`, distance `[10.0]`; zero-dist (AC-4b) `X=[[0],[1],[2]]`,`y=[10,20,30]`,distance,query`1.0` → `[20.0]`; ferrolearn matches all. Verified probe. 1-D only — 2-D `y` is REQ-9. |
| REQ-5 (reg `score` R²) | SHIPPED | impl `pub fn score` for `FittedKNeighborsRegressor` → `pub(crate) fn r2_score` (`1 - SS_res/SS_tot`; constant-`y` → `1.0` if `SS_res==0` else `-inf`) mirrors `RegressorMixin.score` = `r2_score(multioutput='uniform_average')`. Non-test consumer: `_RsKNeighborsRegressor::score` (`py_regressor!` macro). Live oracle (AC-5): `X=[[0],[1],[5],[6]]`,`y=[0,10,50,60]`,`k=1` → `score==1.0`; ferrolearn R² → `1.0`. Verified probe. (Multi-output `uniform_average` averaging is N/A while `y` is 1-D — REQ-9.) |
| REQ-6 (shared k-NN search value + DEFAULT-backend exact-tie SET/ORDER) | SHIPPED | impl `pub(crate) fn kneighbors_impl` + `pub fn kneighbors` (both fitted types) → per-row `fn find_neighbors` fills `(n_queries,k)` nearest-first `(distances, indices)`. The f64 path replicates the DEFAULT `auto` backend bit-for-bit (NOT algorithm-invariant): `fn brute_parallel_on_y` (ArgKmin `parallel_on_Y` double-heap) when `n_features > 15` or `k >= n_samples // 2`, else `sk_kdtree::SkKdTree::{build,query}` (bit-exact KDTree, partition via `introselect::nth_element` = libstdc++ `std::nth_element`). Non-test consumers: `fn predict` (clf + reg) call `find_neighbors`; `kneighbors` is pub on the bound types; `graph.rs` `kneighbors_graph` calls `self.kneighbors(...)`. Verified bit-exact vs live sklearn 1.5.2 over 400 fuzz fixtures / 1600 queries (brute + kd_tree regimes); pin `divergence_kneighbors_order_differs_from_sklearn_default` (#2143) green. Backs AC-1/2/4. |
| REQ-7 (`HasClasses` / `classes_`) | SHIPPED | impl `HasClasses for FittedKNeighborsClassifier` (`classes()` sorted-unique labels, `n_classes()`) mirrors `classes_` lexicographic order (`_classification.py:120`, "Classes are ordered by lexicographic order" `:321`). Non-test consumer: `_RsKNeighborsClassifier::classes_` getter (`classifiers.rs`) → `fitted.classes()`, surfaced as `ferrolearn.KNeighborsClassifier.classes_`. In-tree `test_classifier_has_classes` pins `classes()==[0,1,2]`. |
| REQ-8 (fit-timing parity — #872-class) | NOT-STARTED | open prereq blocker #874. BOTH `fn fit` methods (clf line ~292, reg line ~696) reject `n_samples < n_neighbors` with `InsufficientSamples` at fit time; sklearn `_fit` has **no** such check — it succeeds and defers `ValueError("Expected n_neighbors <= n_samples_fit, ...")` to query time (`_base.py:828-832`). Pin (AC-7): `KNeighborsClassifier(n_neighbors=5).fit([[0],[1],[2]],[0,1,0])` → sklearn fit OK then `predict` raises; ferrolearn `fit` → `Err(InsufficientSamples{required:5,actual:3,...})`. Same for the regressor. **The cleanest single-file deterministic fix** (remove/defer the two fit-time guards). Same class as `nearest_neighbors.md` REQ-4 (#865). |
| REQ-9 (reg multi-output 2-D y) | NOT-STARTED | open prereq blocker #875. `FittedKNeighborsRegressor` stores `y_train: Array1<F>` and `impl Fit<Array2<F>, Array1<F>>` — 1-D `y` only; sklearn `KNeighborsRegressor.predict` reshapes `_y` to 2-D and returns `(n_queries, n_outputs)` when `_y.ndim > 1` (`_regression.py:253-270`). Pin (AC-8): `KNeighborsRegressor(n_neighbors=3).fit(X, Y_2col).predict(Xq)` → sklearn `(1,2)` `[[1.0,200.0]]`; ferrolearn has no 2-D `y` surface. |
| REQ-10 (constructor params + callable weights) | NOT-STARTED | open prereq blocker #876. `KNeighbors{Classifier,Regressor}` have `n_neighbors`/`algorithm`/`weights∈{Uniform,Distance}` but **no `leaf_size=30` (`_classification.py:199`)**, **`p=2` (`:200`)**, **`metric='minkowski'` (`:201`)**, **`metric_params` (`:202`)**, **`n_jobs` (`:203`)**, and `Weights` has **no callable variant** (sklearn `weights` accepts a callable, `:190`). Euclidean-only. Pin (AC-9): `KNeighborsClassifier().leaf_size==30`, `.p==2`, `.metric=='minkowski'`; ferrolearn lacks all. |
| REQ-11 (PyO3 `predict_proba`/`score`/`weights`) | NOT-STARTED | open prereq blocker #877. `_RsKNeighborsClassifier` (`classifiers.rs`) exposes only `new(n_neighbors)`/`fit`/`predict`/`classes_` — **no `predict_proba`, no `score`, no `weights`/`algorithm` knob**; `_RsKNeighborsRegressor` (`extras.rs`, `py_regressor!`) exposes `n_neighbors`/`fit`/`predict`/`score` — **no `weights`**. So `import ferrolearn` cannot reach distance-weighting or `predict_proba` (which exist in the library, REQ-2). Pin (AC-10): `hasattr(ferrolearn.KNeighborsClassifier(), 'predict_proba')` False / `weights` kwarg rejected. |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #878. `knn.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` (the wrong substrate); not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`knn.rs` follows the unfitted/fitted split (CLAUDE.md naming) for two estimators
that share the `Weights` enum, the `Algorithm`/`SpatialIndex` machinery, the
`find_neighbors` k-NN kernel, and `kneighbors_impl`:

- `KNeighborsClassifier<F>` (`n_neighbors`/`algorithm`/`weights`) → `Fit<Array2<F>,
  Array1<usize>>` → `FittedKNeighborsClassifier<F>` (`x_train`, `y_train:
  Array1<usize>`, `n_neighbors`, `weights`, `spatial_index`, sorted `classes`).
- `KNeighborsRegressor<F>` (same params) → `Fit<Array2<F>, Array1<F>>` →
  `FittedKNeighborsRegressor<F>` (`x_train`, `y_train: Array1<F>`, ...). No
  `classes`; **1-D `y_train`** (REQ-9).

Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (no panics in library code, R-CODE-2 — note the
`class_idx` closure's `.expect("label not in fitted classes")` is an internal
invariant on data built from `y_train`, not user input).

**Fit path (both `fn fit`).** Validation rejects `n_samples != y.len()`
(`ShapeMismatch`), `n_neighbors == 0` (`InvalidParameter`), and **`n_samples <
n_neighbors` (`InsufficientSamples`)** — the last has **no sklearn analog**:
sklearn's `_fit` performs no `n_neighbors`/`n_samples` check and defers the
`ValueError` to `kneighbors` query time (`_base.py:828-832`), so a fit sklearn
accepts (`n_neighbors=5` on 3 rows) is rejected by ferrolearn (REQ-8, the
#872-class divergence present in BOTH fits). The classifier then computes
sorted-deduped `classes`; both build the spatial index via `build_spatial_index`
(`Auto` → KdTree ≤15 features else BallTree; otherwise direct).

**Classifier predict (`fn predict` → `fn weighted_vote` → `fn class_score_vec`).**
`class_score_vec` returns per-class weighted vote sums in sorted-`classes` order:
uniform → `+1` per neighbor; distance → `+1/d` per neighbor, **except** when any
neighbor has `d < eps=1e-15` (a zero-distance/coincident point), in which case
only the zero-distance neighbors get `+1` each — mirroring sklearn's `_get_weights`
zero-distance branch. `weighted_vote` argmaxes; because the scan starts at index 0
and only replaces on a **strict** `>`, equal scores keep the earlier (smaller)
label — the smallest-label tie-break matching `np.argmax`'s first-max
(`_classification.py:268`). `predict_proba` normalizes the same `class_score_vec`
per row; the `total == 0` uniform fallback has no built-in-weights analog (sklearn
raises only for user callables, N/A here). Threshold `PAR_THRESHOLD=256` switches
to a `rayon` parallel map — a result-invariant performance choice (R-DEV-7).

**Regressor predict (`fn predict` → `fn weighted_mean`).** uniform → arithmetic
mean of the `k` neighbor targets; distance → `Σ(w·y)/Σw` with `w=1/d`, **except**
the zero-distance branch averages only the coincident targets — mirroring
`_regression.py:257-265` + `_get_weights`. This is the **1-D** path; sklearn's
2-D `(n_queries, n_outputs)` multi-output path (`_regression.py:253-270`) is
absent (REQ-9).

**Scoring.** Classifier `score` = `correct/n` (mean accuracy). Regressor `score`
→ `pub(crate) fn r2_score`: `1 - SS_res/SS_tot`, constant-`y` (`SS_tot==0`) → `1.0`
if `SS_res==0` else `F::neg_infinity()` — the `RegressorMixin.score` convention.
(With 1-D `y` the `multioutput='uniform_average'` averaging is a no-op; it would
matter only once REQ-9 lands.)

**Shared k-NN (`fn kneighbors_impl` / `fn find_neighbors`).** `find_neighbors`
dispatches per query row to `KdTree::query` / `BallTree::query` /
`kdtree::brute_force_knn`, converting results to `(index, true-Euclidean-distance)`
nearest-first. `kneighbors_impl` validates feature count, `k == 0`
(`InvalidParameter`), `k > n_train` (`InsufficientSamples`) and fills the
`(n_queries, k)` arrays. This is the kernel both `predict` paths and the public
`kneighbors` method consume; its oracle value parity is the same Euclidean k-NN
search documented in `.design/neighbors/nearest_neighbors.md` (AC-1) and
`balltree.md` (AC-1).

**Consumer wiring.** The non-test production consumers:
- `ferrolearn-python` `_RsKNeighborsClassifier` (`classifiers.rs`) — `fit` /
  `predict` / `classes_`; `_RsKNeighborsRegressor` (`extras.rs`, `py_regressor!`)
  — `fit` / `predict` / `score`. Surfaced as `ferrolearn.KNeighborsClassifier` /
  `ferrolearn.KNeighborsRegressor`. The bindings under-expose (no
  `predict_proba`/classifier-`score`/`weights` knob — REQ-11), but the predict
  path is a real non-test consumer of the library `predict`/`r2_score`.
- `impl PipelineEstimator<F> for KNeighbors{Classifier,Regressor}<F>` —
  `fit_pipeline`/`predict_pipeline` consume `fit`/`predict` in-crate (the
  classifier wrapper maps float labels → `usize` and back).

**Missing fitted attributes vs sklearn:** `effective_metric_` /
`effective_metric_params_` (`_classification.py:123-132`), `n_features_in_`
(`:134`), `feature_names_in_` (`:139`), `outputs_2d_` (`:148`). ferrolearn exposes
`classes()`/`n_classes()` (clf), `n_samples_fit()` (both).

**Invariants held vs sklearn:** clf `predict` value + smallest-label tie-break
(AC-1); `predict_proba` value incl. zero-dist (AC-2/4b); clf `score` accuracy
(AC-3); reg `predict` value uniform/distance/zero-dist for 1-D `y` (AC-4); reg
`score` R² (AC-5); the shared nearest-first k-NN search (AC-6).

**Invariants NOT held vs sklearn:** fit-vs-query timing of the `n_neighbors >
n_samples` check (REQ-8); regressor multi-output 2-D `y` (REQ-9); constructor
`leaf_size`/`p`/`metric`/`metric_params`/`n_jobs` + callable weights (REQ-10); the
PyO3 `predict_proba`/`score`/`weights` surface (REQ-11); the ferray substrate
(REQ-12).

## Verification

Library crate (green at baseline `992a3bb6` for the existing contract):
```
cargo test -p ferrolearn-neighbors --lib knn
cargo clippy -p ferrolearn-neighbors --all-targets -- -D warnings
cargo fmt --all --check
```
The 27 in-tree `#[test]`s (`test_classifier_simple`, `test_classifier_k1_memorizes`,
`test_classifier_k_equals_n_predicts_mode`, `test_classifier_distance_weighting`,
`test_classifier_tied_votes`, `test_classifier_brute_force_algorithm`,
`test_classifier_kdtree_algorithm`, `test_classifier_has_classes`,
`test_classifier_pipeline_integration`, `test_classifier_new_data_prediction`,
`test_regressor_simple`, `test_regressor_mean_of_neighbors`,
`test_regressor_distance_weighting`, `test_regressor_exact_match_distance_weighting`,
`test_regressor_pipeline_integration`, … plus shape/k/f32/default guards) pin
ferrolearn's current k-NN classify/regress behavior. **None compares against the
live sklearn oracle**, but the seven SHIPPED REQs value-match the oracle (verified
by throwaway `cargo run --example` probe), so they are SHIPPED; the rest are
NOT-STARTED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin the deterministic ones FIRST**: REQ-8
(fit-timing — the cleanest single-file fix; remove/defer the two `InsufficientSamples`
fit guards), then REQ-9 (multi-output 2-D `y`), then REQ-10 (surface/hasattr) and
REQ-11 (binding hasattr):
```
# REQ-1 (present, must stay green): clf predict value parity (uniform + distance)
python3 -c "import numpy as np; from sklearn.neighbors import KNeighborsClassifier as C; X=np.array([[0.,0.],[1.,0.],[0.,1.],[5.,5.],[6.,5.],[5.,6.]]); y=np.array([0,0,0,1,1,1]); Xq=np.array([[0.2,0.1],[5.2,5.1]]); print(C(n_neighbors=3).fit(X,y).predict(Xq).tolist(), C(n_neighbors=3,weights='distance').fit(X,y).predict(Xq).tolist())"  # [0,1] [0,1]
# REQ-1b (tie -> smallest label):
python3 -c "import numpy as np; from sklearn.neighbors import KNeighborsClassifier as C; print(C(n_neighbors=2).fit(np.array([[0.],[1.]]),np.array([0,1])).predict(np.array([[0.5]])).tolist())"  # [0]
# REQ-2 (present): predict_proba distance + uniform
python3 -c "import numpy as np; from sklearn.neighbors import KNeighborsClassifier as C; X=np.array([[0.],[1.],[2.],[10.],[11.]]); y=np.array([0,1,0,1,1]); q=np.array([[1.5]]); print(C(n_neighbors=3,weights='distance').fit(X,y).predict_proba(q).tolist(), C(n_neighbors=3).fit(X,y).predict_proba(q).tolist())"  # [[0.5714...,0.4286...]] [[0.6667...,0.3333...]]
# REQ-4 (present): reg predict uniform + distance + zero-dist
python3 -c "import numpy as np; from sklearn.neighbors import KNeighborsRegressor as R; X=np.array([[0.],[1.],[2.],[3.],[100.]]); y=np.array([0.,10.,20.,30.,1000.]); q=np.array([[1.0]]); print(R(n_neighbors=3).fit(X,y).predict(q).tolist(), R(n_neighbors=3,weights='distance').fit(X,y).predict(q).tolist())"  # [10.0] [10.0]
python3 -c "import numpy as np; from sklearn.neighbors import KNeighborsRegressor as R; print(R(n_neighbors=3,weights='distance').fit(np.array([[0.],[1.],[2.]]),np.array([10.,20.,30.])).predict(np.array([[1.0]])).tolist())"  # [20.0]
# REQ-8 (fit-timing, the #872-class divergence): sklearn fit OK, defers to predict
python3 -c "import numpy as np; from sklearn.neighbors import KNeighborsClassifier as C; m=C(n_neighbors=5).fit(np.array([[0.],[1.],[2.]]),np.array([0,1,0])); print('clf fit OK')"  # fit OK (ferro: InsufficientSamples)
python3 -c "import numpy as np; from sklearn.neighbors import KNeighborsClassifier as C; m=C(n_neighbors=5).fit(np.array([[0.],[1.],[2.]]),np.array([0,1,0])); m.predict(np.array([[0.5]]))"  # ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 5, n_samples_fit = 3, n_samples = 1
# REQ-9 (multi-output 2-D y):
python3 -c "import numpy as np; from sklearn.neighbors import KNeighborsRegressor as R; Y=np.array([[0.,100.],[1.,200.],[2.,300.],[3.,400.],[4.,500.]]); p=R(n_neighbors=3).fit(np.array([[0.],[1.],[2.],[3.],[4.]]),Y).predict(np.array([[1.0]])); print(p.shape, p.tolist())"  # (1,2) [[1.0,200.0]]
# REQ-10 (defaults / missing params):
python3 -c "from sklearn.neighbors import KNeighborsClassifier as C; c=C(); print(c.n_neighbors,c.weights,c.algorithm,c.leaf_size,c.p,c.metric,c.metric_params,c.n_jobs)"  # 5 uniform auto 30 2 minkowski None None
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-neighbors/tests/divergence_knn.rs`, asserting the live-sklearn expected
values above and FAILING against current `knn.rs`. REQ-1..7 already match and
should be guarded by non-regression pins.

ferrolearn-python (REQ-11 binding parity, after #877 lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_neighbors.py -q
```
asserting `ferrolearn.KNeighborsClassifier` exposes `predict_proba`/`score`/
`weights` and `ferrolearn.KNeighborsRegressor` exposes `weights`, matching
`sklearn.neighbors` outputs (predict / predict_proba / score) on the AC fixtures.

## Blockers to open

(Director creates the real issues; numbers below are SUGGESTIONS continuing the
neighbors layer past `nearest_neighbors.md` #865-#871 and `balltree.md` #855-#863.
#874 is the same fit-vs-query-timing class as `nearest_neighbors.md` #865 but here
it is split across BOTH `KNeighbors*` fits.)

- #874 — REQ-8 (fit-timing parity, #872-class): BOTH `fn fit` (clf ~line 292, reg
  ~line 696) reject `n_samples < n_neighbors` with `InsufficientSamples` at fit
  time; sklearn `_fit` has no such check and defers `ValueError("Expected
  n_neighbors <= n_samples_fit, ...")` to query time (`_base.py:828-832`).
  Remove/defer the two fit-time guards (let the existing `kneighbors_impl`
  `k > n_train` guard fire at predict). **The cleanest single-file deterministic
  fix** — the critic should pin this first. Pin: `fit([[0],[1],[2]],...,k=5)` →
  sklearn OK, ferro `Err(InsufficientSamples)`.
- #875 — REQ-9 (regressor multi-output 2-D `y`): `FittedKNeighborsRegressor` is
  `y_train: Array1<F>` / `Fit<…, Array1<F>>`; add a 2-D `y` path returning
  `(n_queries, n_outputs)` (`_regression.py:253-270`). Pin: `fit(X,Y_2col).
  predict(Xq)` → `(1,2)` `[[1.0,200.0]]`.
- #876 — REQ-10 (constructor params + callable weights): no `leaf_size`/`p`/
  `metric`/`metric_params`/`n_jobs` fields and no callable `Weights` variant
  (`_classification.py:199-203`, `:190`). Add the param surface + a `metric`
  abstraction + callable weights.
- #877 — REQ-11 (PyO3 `predict_proba`/`score`/`weights`): `_RsKNeighborsClassifier`
  (`classifiers.rs`) exposes only `fit`/`predict`/`classes_`; add `predict_proba`,
  `score`, and a `weights`/`algorithm` constructor knob, and add `weights` to
  `_RsKNeighborsRegressor` (`extras.rs`). Pin: `hasattr(ferrolearn.
  KNeighborsClassifier(), 'predict_proba')` / `weights=` kwarg.
- #878 — REQ-12 (ferray substrate): migrate `knn.rs` off `ndarray`/`num-traits` to
  `ferray-core` (R-SUBSTRATE).
