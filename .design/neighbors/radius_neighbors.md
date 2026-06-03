# Radius Neighbors (sklearn.neighbors.RadiusNeighborsClassifier / RadiusNeighborsRegressor)

<!--
tier: 3-component
status: draft
baseline-commit: 5def81e5
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/neighbors/_classification.py   # RadiusNeighborsClassifier(RadiusNeighborsMixin, ClassifierMixin, NeighborsBase); __init__ radius=1.0/weights='uniform'/algorithm='auto'/leaf_size=30/p=2/metric='minkowski'/outlier_label=None/metric_params=None/n_jobs=None (:578-601); fit handles outlier_label None/'most_frequent'/manual (:607-676); predict (:678-718, argmax of predict_proba + outlier-label assign); predict_proba (:720-836, empty→ValueError if outlier_label_ is None :781-787, else one-hot/all-zero+warn :813-824)
  - sklearn/neighbors/_regression.py       # RadiusNeighborsRegressor(RadiusNeighborsMixin, RegressorMixin, NeighborsBase); __init__ radius=1.0/weights='uniform'/algorithm='auto'/leaf_size=30/p=2/metric='minkowski'/metric_params=None/n_jobs=None (:412-433); predict weighted mean + empty→np.nan+warn (:459-514, multi-output 2D y); RegressorMixin.score = r2_score(multioutput='uniform_average')
  - sklearn/neighbors/_base.py             # RadiusNeighborsMixin.radius_neighbors(X=None, radius=None, return_distance=True, sort_results=False) — default sort_results=False (native order); _get_weights distance + zero-dist; n_samples_fit_/_fit_X; fit does NOT validate radius vs n_samples
ferrolearn-module: ferrolearn-neighbors/src/radius_neighbors.rs
parity-ops: RadiusNeighborsClassifier (.__init__, .fit, .predict, .predict_proba, .score, .radius_neighbors), RadiusNeighborsRegressor (.__init__, .fit, .predict, .score, .radius_neighbors)
crosslink-issue: 880
-->

## Summary

`ferrolearn-neighbors/src/radius_neighbors.rs` mirrors scikit-learn's
`RadiusNeighborsClassifier` (`sklearn/neighbors/_classification.py`) and
`RadiusNeighborsRegressor` (`sklearn/neighbors/_regression.py`), both thin
estimators over the shared `RadiusNeighborsMixin` / `NeighborsBase` machinery in
`sklearn/neighbors/_base.py`. They classify / regress using **all** training
samples within a fixed `radius` of each query point (Euclidean distance), rather
than a fixed neighbor count. It exposes the unfitted `RadiusNeighborsClassifier<F>`
/ `RadiusNeighborsRegressor<F>` (`radius=1.0`, `weights=Uniform`,
`algorithm=Auto`; classifier adds `outlier_label: Option<usize>`), their fitted
`FittedRadiusNeighborsClassifier<F>` / `FittedRadiusNeighborsRegressor<F>`, the
`Weights::{Uniform, Distance}` enum (reused from `knn.rs`, with sklearn's
zero-distance special-casing), and the shared `radius_neighbors_impl` /
`find_radius_neighbors` radius search. Both estimators are re-exported at the crate
root (`ferrolearn-neighbors/src/lib.rs`: `pub use radius_neighbors::{Fitted
RadiusNeighborsClassifier, FittedRadiusNeighborsRegressor, RadiusNeighbors
Classifier, RadiusNeighborsRegressor}`) and consumed **non-test** by `graph.rs`
(the `radius_neighbors_graph` free function builds a transient
`RadiusNeighborsClassifier`, calls `fit` then `radius_neighbors`; the
`FittedRadiusNeighbors{Classifier,Regressor}::radius_neighbors_graph` methods call
each estimator's `radius_neighbors`) plus the in-crate pipeline integration.

Under honest underclaim (R-HONEST-3), the **behaviors that are genuinely present
and value-match the live sklearn 1.5.2 oracle** are:

- **classifier `predict` value** (uniform + distance weights, argmax-with-smallest-
  label tie-break) — matches `RadiusNeighborsClassifier.predict`
  (`_classification.py:678-718`) on a tie-free fixture: `[0, 1]`.
- **classifier `predict_proba` value (with neighbors)** — normalized weighted
  class-vote shares in `classes_` order, matching `predict_proba`
  (`_classification.py:720-836`) on a non-trivial distance-weighted fixture
  (`[0.5714…, 0.4286…]`) and uniform (`[0.6667…, 0.3333…]`).
- **classifier `score`** — mean accuracy, the `ClassifierMixin.score` analog.
- **classifier `outlier_label` value for `Some(label)` / `None`** — a query with no
  in-radius neighbor returns the outlier label when set (`predict` far → `99`),
  matching sklearn's manual-outlier-label `predict` (`_classification.py:710-713`);
  `None` + an outlier **raises** (both error, type/message differ — REQ split).
- **classifier `classes` / `n_classes`** — sorted unique class labels, the
  `classes_` lexicographic-order analog.
- **regressor `predict` value (with neighbors)** (uniform + distance, zero-distance
  special case) — weighted mean of in-radius targets, matching
  `RadiusNeighborsRegressor.predict` (`_regression.py:459-514`) for **1-D `y`**:
  uniform `[10.0]`, distance `[10.0]`.
- **regressor `score`** — R² (`RegressorMixin.score` / `r2_score` with
  `multioutput='uniform_average'`).
- the **shared radius search value** (`radius_neighbors_impl` /
  `find_radius_neighbors`) — the in-radius `(distances, indices)` SET both
  `predict` paths and the public `radius_neighbors` method consume, matching
  sklearn's `radius_neighbors(X, radius)` **as a set** (and matching the
  `sort_results=True` ascending order).

Everything else diverges from the `RadiusNeighbors*` contract:

1. **Regressor no-neighbor: ferrolearn RAISES, sklearn returns `np.nan` + warns
   (THE KEY divergence — the cleanest single-file fixable).** ferrolearn `fn
   predict` returns `Err(InvalidParameter)` for any query with no in-radius
   neighbor (`radius_neighbors.rs` regressor `fn predict`). sklearn does **NOT**
   raise — it assigns `np.nan` to the empty row (`empty_obs = np.full_like(_y[0],
   np.nan)`, `_regression.py:482,487,498`) and emits `UserWarning("One or more
   samples have no neighbors within specified radius; predicting NaN.")`
   (`_regression.py:504-509`). Live oracle: `RadiusNeighborsRegressor(radius=0.01)
   .fit([[0],[10]],[0,100]).predict([[5]])` → `[nan]` + warning; ferrolearn →
   `Err(InvalidParameter { reason: "query sample 0 has no neighbors within
   radius=0.01" })`. R-DEV-1 (NaN/empty handling).
2. **Classifier `outlier_label` modes — `'most_frequent'` and manual-label edge.**
   ferrolearn's `outlier_label: Option<usize>` covers only "a specific label" /
   "None". sklearn additionally accepts **`outlier_label='most_frequent'`**
   (`_classification.py:636-642`: outliers get the training **mode**; live oracle
   `outlier_label_ == [0]`, `predict` far → `0`) — ferrolearn has no such mode. AND
   for a manual `outlier_label` **not in `classes_`** the two diverge on
   `predict_proba` (REQ split): sklearn keeps `classes_` unchanged, assigns the
   outlier row **all-zero** then normalizes-to-zero (`_classification.py:813-824`,
   live oracle `classes_==[0,1]`, `predict_proba` far → `[[0.0, 0.0]]` + warning)
   while ferrolearn falls back to **uniform** `[[0.5, 0.5]]` (its `predict_proba`
   `binary_search` miss → uniform branch). `predict` itself matches (`99`) for the
   manual label.
3. **Regressor no multi-output 2-D `y`.** `FittedRadiusNeighborsRegressor` stores
   `y_train: Array1<F>` and `impl Fit<Array2<F>, Array1<F>>` — 1-D only; sklearn
   `RadiusNeighborsRegressor.predict` reshapes `_y` to 2-D and returns
   `(n_queries, n_outputs)` (`_regression.py:478-514`). Live oracle:
   `fit(X, Y_2col).predict(Xq)` → `(1, 2)` `[[10.0, 200.0]]`. MISSING-surface
   (R-DEV-3).
4. **`radius_neighbors` always sorts ascending**, whereas sklearn's default is
   `sort_results=False` → native (tree/brute) order; there is no `sort_results`
   toggle and no `X=None` self-exclusion. The neighbor SET matches; the default
   ORDER diverges (R-DEV-3).
5. **Missing constructor params**: `leaf_size=30`, `p=2`, `metric='minkowski'`,
   `metric_params=None`, `n_jobs=None` are ABSENT on BOTH estimators
   (`_classification.py:584-589`, `_regression.py:418-422`), and `weights` cannot be
   a **callable** (sklearn accepts `'uniform'|'distance'|callable`,
   `_classification.py:573`, `_regression.py:408`). ferrolearn is Euclidean-only
   with the two named weighting schemes.
6. **No-neighbor error type/message (both estimators).** Where sklearn raises
   `ValueError("No neighbors found for test samples %r, ...")`
   (`_classification.py:781-787`) for the classifier with `outlier_label=None`, and
   warns-not-raises for the regressor, ferrolearn raises
   `FerroError::InvalidParameter` with a different message (and the regressor raises
   where sklearn does not — REQ-7). R-DEV-2 (exception type/message).
7. **No PyO3 binding / no meta-crate re-export** — `import sklearn.neighbors` gives
   `RadiusNeighborsClassifier` / `RadiusNeighborsRegressor`; `import ferrolearn`
   gives nothing (no `RsRadiusNeighbors*` in `ferrolearn-python/src/`, verified by
   `grep -ni radiusneighbors`).
8. The crate is on the **`ndarray` substrate** (`use ndarray::{Array1, Array2}` +
   `num_traits::Float`), not ferray (R-SUBSTRATE).

`RadiusNeighbors{Classifier,Regressor}` (and the fitted types) are existing pub
APIs (grandfathered per S5/R-DEFER-1); their non-test production consumers are
`graph.rs` (the `radius_neighbors_graph` free function + the fitted
`radius_neighbors_graph` methods, which consume `fit`/`radius_neighbors`) and the
in-crate pipeline integration (`impl PipelineEstimator for RadiusNeighbors*`).

## Algorithm (sklearn — the contract)

### Construction (`_classification.py:578-601`, `_regression.py:412-433`)

`RadiusNeighborsClassifier(radius=1.0, *, weights='uniform', algorithm='auto',
leaf_size=30, p=2, metric='minkowski', outlier_label=None, metric_params=None,
n_jobs=None)` and `RadiusNeighborsRegressor(radius=1.0, *, weights='uniform',
algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None,
n_jobs=None)` — all args after `radius` are keyword-only (`*`). `weights` is
`{'uniform', 'distance'}`, a callable, or `None` (`_classification.py:573`,
`_regression.py:408`). `outlier_label` is `{Integral, str, "array-like",
'most_frequent', None}` (`_classification.py:574`, :481-491). `fit(X, y)` forwards
to `self._fit(X, y)` (`NeighborsBase._fit`), recording `n_samples_fit_` /
`classes_` (classifier) / `_y`, and building the chosen spatial index — and (clf
only) resolves `outlier_label_` (`_classification.py:625-674`). Neither `fit`
validates `radius` against `n_samples`.

### Classifier `predict` (`_classification.py:678-718`)

`predict` computes `probs = self.predict_proba(X)`, takes `classes_[argmax(probs,
axis=1)]` (`:707-708`), then **overrides** any row whose proba is all-zero
(`(prob == 0).all(axis=1)`, `:710`) with `self.outlier_label_[k]` (`:713`). Because
`np.argmax` returns the FIRST (smallest-index → smallest-label, since `classes_` is
sorted) maximum, ties break to the **smallest class label**. `_get_weights(neigh_
dist, 'distance')` weights each neighbor by `1/dist`; a zero-distance neighbor takes
all the weight.

### Classifier `predict_proba` (`_classification.py:720-836`)

Per query row: gather in-radius neighbor labels; uniform → `np.bincount`; distance
→ `np.bincount(..., weights)` (`:803-810`). **Empty (outlier) rows**: if
`self.outlier_label_ is None` and any outlier exists → **`ValueError("No neighbors
found for test samples %r, you can try using larger radius, giving a label for
outliers, or considering removing them from your dataset.")`** (`:781-787`); else,
for each outlier row, if `outlier_label_[k]` **is in** `classes_` set its one-hot
column to `1.0` (`:815-817`), else **warn** ("Outlier label {} is not in training
classes. All class probabilities of outliers will be assigned with 0.", `:819-824`)
and leave the row all-zero. Finally normalize each row by its sum (a row summing to
0 is left at 0, `:826-829`). Classes in `classes_` (lexicographic) order.

### Regressor `predict` (`_regression.py:459-514`)

`_y` reshaped to 2-D; per query row, uniform → `np.mean(_y[ind, :], axis=0)`,
distance → `np.average(_y[ind, :], weights=...)` (`:484-502`). **For an empty
row** (`len(ind) == 0`) the prediction is `empty_obs = np.full_like(_y[0], np.nan)`
(`:482,487,498`) — i.e. **`np.nan`, NOT an error** — and if any output is NaN a
`UserWarning("One or more samples have no neighbors within specified radius;
predicting NaN.")` is emitted (`:504-509`). If `_y.ndim == 1` the result is
`ravel`ed to 1-D (`:511-512`); otherwise 2-D `(n_queries, n_outputs)`.

### `score`

Classifier `score` = `ClassifierMixin.score` = mean accuracy. Regressor `score` =
`RegressorMixin.score` = `r2_score(y, predict(X), multioutput='uniform_average')` —
`1 - SS_res/SS_tot`; constant-`y` (`SS_tot == 0`) → `1.0` if `SS_res == 0` else
`-inf`.

### `radius_neighbors` (`RadiusNeighborsMixin.radius_neighbors`, `_base.py`)

`radius_neighbors(X=None, radius=None, return_distance=True, sort_results=False)`.
Returns, per query row, all training points within `radius` as arrays-of-arrays
`(distances, indices)`. **Default `sort_results=False`** → native (tree/brute)
order, NOT sorted by distance; `sort_results=True` sorts each row ascending.
`X is None` excludes each point's own self-match. `radius` defaults to
`self.radius`.

### Edge cases (live oracle, sklearn 1.5.2, run from /tmp)

- clf no-neighbor + `outlier_label=None`: `RadiusNeighborsClassifier(radius=0.01)
  .fit(X,y).predict([[100,100]])` → `ValueError("No neighbors found for test
  samples array([0]), you can try using larger radius, ...")`.
- clf `outlier_label='most_frequent'`: `RadiusNeighborsClassifier(radius=0.01,
  outlier_label='most_frequent').fit(X, y).outlier_label_` → `[0]` (training mode);
  `predict` far → `[0]`.
- clf `outlier_label=99` (manual, not in `y`): `classes_` → `[0, 1]` (unchanged),
  `outlier_label_` → `[99]`, `predict` far → `[99]`, `predict_proba` far →
  `[[0.0, 0.0]]` (+ "Outlier label 99 is not in training classes" warning).
- reg no-neighbor: `RadiusNeighborsRegressor(radius=0.01).fit([[0],[10]],[0,100])
  .predict([[5]])` → `[nan]` + `UserWarning("One or more samples have no neighbors
  within specified radius; predicting NaN.")` — **does not raise**.
- reg multi-output: `RadiusNeighborsRegressor(radius=1.5).fit([[0],[1],[2]],
  [[0,100],[10,200],[20,300]]).predict([[1.0]])` → `(1, 2)` `[[10.0, 200.0]]`.

## ferrolearn (what exists)

All in `ferrolearn-neighbors/src/radius_neighbors.rs`, generic over
`F: Float + Send + Sync + 'static`; `ndarray` substrate; `Weights` /
`Algorithm` reused from `knn.rs`.

- **`pub struct RadiusNeighborsClassifier<F> { pub radius: F, pub weights: Weights,
  pub algorithm: Algorithm, pub outlier_label: Option<usize>, _marker }`** and the
  analogous **`RadiusNeighborsRegressor<F>`** (no `outlier_label`). `pub fn new`
  sets `radius=1.0`, `weights=Uniform`, `algorithm=Auto` (clf `outlier_label=None`);
  builder setters `with_radius`/`with_weights`/`with_algorithm`
  (`with_outlier_label` on the clf); `impl Default → new()`. **No `leaf_size`/`p`/
  `metric`/`metric_params`/`n_jobs` fields.**
- **`pub enum Weights { Uniform, Distance }`** (from `knn.rs`) — sklearn's
  `'uniform'`/`'distance'`; **no callable variant**.
- **`impl Fit<Array2<F>, Array1<usize>> for RadiusNeighborsClassifier<F>` /
  `fn fit`** — rejects `n_samples != y.len()` (`ShapeMismatch`), `radius <= 0`
  (`InvalidParameter`), `n_samples == 0` (`InsufficientSamples`); computes
  sorted-deduped `classes`; builds the spatial index. The regressor `fn fit` is
  identical but stores `y_train: Array1<F>` and has no `classes`. (Neither matches
  sklearn's no-`radius`-vs-`n_samples` check class, but neither performs such a
  check — only the empty-input guard, which sklearn also enforces via `_fit`.)
- **`impl Predict for FittedRadiusNeighborsClassifier<F>` / `fn predict`** — per
  query row, `find_radius_neighbors` → `fn weighted_vote` (over `fn
  class_score_vec`); on an **empty** neighborhood, returns `outlier_label` if `Some`
  else `Err(InvalidParameter)`. `class_score_vec` builds per-class weighted vote
  sums in `self.classes` order (uniform → counts; distance → `1/dist` with the
  **zero-distance branch** giving coincident neighbors weight 1 each via
  `has_zero_dist`/`eps=1e-15`); `fn weighted_vote` argmaxes, tie-breaking to the
  **smallest class label** (first index in sorted `self.classes`).
- **`pub fn predict_proba(&self, x) -> Result<Array2<F>>`** — normalizes
  `class_score_vec` per row (`scores[ci]/total`). On an **empty** neighborhood: if
  `outlier_label` is `Some` AND that label is in `classes` (`binary_search` hit),
  one-hot that column; **else** uniform `1/n_classes` per row. (The "else uniform"
  fallback is where ferrolearn diverges from sklearn's all-zero-then-normalize-to-0
  for an outlier label not in `classes_` — REQ-2.)
- **`pub fn score(&self, x, y) -> Result<F>`** (classifier) — mean accuracy
  (`correct/n`); the `ClassifierMixin.score` analog.
- **`pub fn classes(&self) -> &[usize]` / `pub fn n_classes(&self) -> usize`** —
  sorted unique labels (the `classes_` analog) + count.
- **`impl Predict for FittedRadiusNeighborsRegressor<F>` / `fn predict`** — per
  query row, `find_radius_neighbors` → `fn weighted_mean` (uniform → mean; distance
  → `Σ(y/d)/Σ(1/d)`, with the **zero-distance branch** averaging coincident
  targets); on an **empty** neighborhood, returns `Err(InvalidParameter)` (the KEY
  divergence — sklearn returns NaN + warns, REQ-7).
- **`pub fn score(&self, x, y) -> Result<F>`** (regressor) → `crate::knn::r2_score`
  (`1 - SS_res/SS_tot`; constant-`y` → `1.0`/`-inf`), the `RegressorMixin.score`
  analog.
- **`pub fn radius_neighbors(&self, x, radius: Option<F>)`** (both fitted types) →
  shared **`pub(crate) fn radius_neighbors_impl`** → `fn find_radius_neighbors`
  (BallTree `within_radius` or `fn brute_force_radius`), which **always sorts
  ascending** by distance (`results.sort_by(...)`). Returns
  `RadiusNeighborsResult<F> = (Vec<Vec<F>>, Vec<Vec<usize>>)` (jagged per-query).
  **No `sort_results` toggle; no `X=None` self-exclusion; no `return_distance`
  toggle.** **`pub fn n_samples_fit`** (both) is the `n_samples_fit_` analog.
- **Pipeline integration**: `impl PipelineEstimator<F> for RadiusNeighbors{
  Classifier,Regressor}<F>` (`fn fit_pipeline`) + `FittedRadiusNeighbors{
  Classifier,Regressor}Pipeline` wrappers (`fn predict_pipeline`).
- **Spatial index**: `fn build_spatial_index` — `Algorithm::Auto` → `KdTree` if
  `n_features <= 15` else `BallTree`; `KdTree`/`BallTree`/`BruteForce` map directly.
  Distance is Euclidean throughout. (Note: `find_radius_neighbors` uses brute-force
  radius search for both `KdTree` and `BruteForce` modes; only `BallTree` uses the
  tree's `within_radius`.)

**Consumers (non-test).** Crate re-export (`ferrolearn-neighbors/src/lib.rs`) plus
`graph.rs`:
- `pub fn radius_neighbors_graph` (free function) constructs a transient
  `RadiusNeighborsClassifier::new().with_radius(radius)`, calls `clf.fit(x,
  &dummy_y)` then `fitted.radius_neighbors(x, Some(radius))`, then drops the
  self-edge (zero diagonal) — emulating sklearn's `include_self=False` /
  `X=None` graph semantics (`sklearn/neighbors/_graph.py:255`).
- `FittedRadiusNeighborsClassifier::radius_neighbors_graph` /
  `FittedRadiusNeighborsRegressor::radius_neighbors_graph` (`graph.rs`) call each
  estimator's `radius_neighbors(x, radius)`.

These are existing pub APIs (grandfathered, S5/R-DEFER-1) and the non-test
production consumers backing the value REQs.

## Requirements

- REQ-1: **classifier `predict` value + tie-break (R-DEV-1/3).** Mirror
  `RadiusNeighborsClassifier.predict` (`_classification.py:678-718`): argmax of the
  weighted class-vote shares with **smallest-label tie-break** (`np.argmax`
  first-max, `:707`), uniform and distance weights. ferrolearn `fn predict`/`fn
  weighted_vote` value-matches the live oracle.
- REQ-2: **classifier `predict_proba` value, with neighbors (R-DEV-3).** Mirror
  `predict_proba` (`_classification.py:720-836`): normalized weighted class-vote
  shares in `classes_` order, with the zero-distance branch. ferrolearn `pub fn
  predict_proba` value-matches on a non-trivial distance-weighted fixture (the empty
  / outlier-label-not-in-classes row handling is REQ-6).
- REQ-3: **classifier `score` accuracy (R-DEV-1).** Mirror `ClassifierMixin.score`
  (mean accuracy). ferrolearn `pub fn score` value-matches.
- REQ-4: **classifier `outlier_label` `Some(label)` / `None` value (R-DEV-1/3).**
  Mirror sklearn's manual-outlier-label `predict`: an empty neighborhood returns
  the label (`_classification.py:710-713`) when set, and raises when `None`. (The
  `'most_frequent'` mode and the proba all-zero handling for a label not in
  `classes_` are REQ-6.) ferrolearn `fn predict` empty branch value-matches for
  `Some(label)`; both raise for `None` (type/message is REQ-9).
- REQ-5: **classifier `classes` / `n_classes` (R-DEV-3).** Expose sorted unique
  class labels (`classes_` lexicographic order). ferrolearn `pub fn classes` /
  `pub fn n_classes` provide this; `graph.rs` builds a clf with `dummy_y` and
  consumes the fitted type.
- REQ-6: **classifier `outlier_label='most_frequent'` + manual-label proba
  semantics (R-DEV-2/3).** Mirror `outlier_label='most_frequent'` (training mode,
  `_classification.py:636-642`) and the manual-label-not-in-`classes_` all-zero
  `predict_proba` row (`:813-824`). ferrolearn's `Option<usize>` has no
  `'most_frequent'`, and its empty-row `predict_proba` falls back to **uniform**
  rather than all-zero when the label is not in `classes`.
- REQ-7: **regressor no-neighbor NaN + warn (the KEY divergence, R-DEV-1).** Mirror
  `RadiusNeighborsRegressor.predict` empty-row handling: assign `np.nan` (NOT an
  error) + `UserWarning` (`_regression.py:482,504-509`). ferrolearn `fn predict`
  **raises** `Err(InvalidParameter)`. The cleanest single-file deterministic fix.
- REQ-8: **regressor `predict` value (with neighbors), 1-D y, uniform + distance
  (R-DEV-1/3).** Mirror `RadiusNeighborsRegressor.predict` for 1-D `y`: (weighted)
  mean of in-radius targets, zero-distance special case. ferrolearn `fn predict`/`fn
  weighted_mean` value-matches the live oracle.
- REQ-9: **regressor `score` R² (R-DEV-1).** Mirror `RegressorMixin.score` =
  `r2_score(multioutput='uniform_average')`, constant-`y` → `1.0`/`-inf`. ferrolearn
  `pub fn score` → `crate::knn::r2_score` value-matches.
- REQ-10: **shared `radius_neighbors` set (R-DEV-1/3).** Mirror the
  `RadiusNeighborsMixin` radius kernel both predict paths consume: the in-radius
  `(distances, indices)` SET per row. ferrolearn `pub(crate) fn
  radius_neighbors_impl` / `pub fn radius_neighbors` value-matches the oracle **as a
  set** (and matches the `sort_results=True` order).
- REQ-11: **`radius_neighbors` `sort_results` default + `X=None` (R-DEV-3).** Match
  the default `sort_results=False` (native order) + the `sort_results=True` toggle,
  and the `X=None` self-exclusion. ferrolearn **always sorts ascending** (matches
  `sort_results=True`, diverges from the default order); no `X=None` path.
- REQ-12: **no-neighbor error type/message (both, R-DEV-2).** Match clf's
  `ValueError("No neighbors found for test samples %r, ...")`
  (`_classification.py:781-787`). ferrolearn raises `FerroError::InvalidParameter`
  with a different message; the regressor raises where sklearn does not (REQ-7).
- REQ-13: **regressor multi-output 2-D y (R-DEV-3).** Mirror
  `RadiusNeighborsRegressor.predict` for `(n_samples, n_outputs)` `y` → 2-D
  prediction (`_regression.py:478-514`). ferrolearn `y_train: Array1<F>` is **1-D
  only**.
- REQ-14: **constructor params + callable weights (R-DEV-2).** Match `leaf_size=30`,
  `p=2`, `metric='minkowski'`, `metric_params=None`, `n_jobs=None`
  (`_classification.py:584-589`, `_regression.py:418-422`) and `weights` as a
  **callable** (`:573`/`:408`). ferrolearn has only `radius`/`weights∈{Uniform,
  Distance}`/`algorithm` (+ clf `outlier_label`); Euclidean-only, no callable.
- REQ-15: **PyO3 binding + meta-crate re-export (R-DEFER-1).** `import ferrolearn`
  exposes `RadiusNeighborsClassifier` / `RadiusNeighborsRegressor` mirroring `import
  sklearn`. No shim and no meta-crate re-export exist.
- REQ-16: **ferray substrate (R-SUBSTRATE).** `radius_neighbors.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (`from sklearn.neighbors
import RadiusNeighborsClassifier, RadiusNeighborsRegressor`, run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). ferrolearn values verified by a throwaway
`cargo run --example` probe (since deleted).

- AC-1 (REQ-1, present & matching): clf `X=[[0,0],[0.5,0],[0,0.5],[5,5],[5.5,5],
  [5,5.5]]`, `y=[0,0,0,1,1,1]`, query `[[0.2,0.1],[5.2,5.1]]`, `radius=1.5`. uniform
  `predict` → `[0,1]`; distance → `[0,1]`. ferrolearn matches both.
- AC-2 (REQ-2, present & matching): clf `X=[[0],[1],[2],[10],[11]]`, `y=[0,1,0,1,1]`,
  `radius=5.0`, query `[[1.5]]`. distance `predict_proba` →
  `[[0.5714285714285715, 0.4285714285714286]]`; uniform →
  `[[0.6666666666666666, 0.3333333333333333]]`. ferrolearn matches both to full
  precision.
- AC-3 (REQ-3, present): clf `X=[[0,0],[0.5,0],[0,0.5],[5,5],[5.5,5],[5,5.5]]`,
  `y=[0,0,0,1,1,1]`, `radius=1.5` → `score(X,y) == 1.0`; ferrolearn `score` → `1.0`.
- AC-4 (REQ-4, present & matching for `Some`/`None`): clf `outlier_label=Some(99)`,
  `radius=0.01`, query far `[[100]]` → `predict` `[99]`; ferrolearn `[99]`. With
  `outlier_label=None`, query far → sklearn raises / ferrolearn `Err` (both raise —
  type is REQ-12).
- AC-5 (REQ-5, present): clf `y=[0,0,0,1,1,1]` → `classes() == [0,1]`,
  `n_classes() == 2`; ferrolearn matches (in-tree `test_classifier_classes`).
- AC-6 (REQ-6 pin): clf `outlier_label='most_frequent'`, `y=[0,0,0,0,1,1]`,
  `radius=0.01`, far query → sklearn `outlier_label_==[0]`, `predict` `[0]`;
  ferrolearn has no `'most_frequent'`. AND clf `outlier_label=99` (not in `y=[0,1,
  0]`), far query → sklearn `predict_proba` `[[0.0,0.0]]`; ferrolearn `[[0.5,0.5]]`
  (uniform fallback).
- AC-7 (REQ-7 pin, KEY): reg `X=[[0],[10]]`, `y=[0,100]`, `radius=0.01`, query
  `[[5]]` → sklearn `[nan]` + `UserWarning("One or more samples have no neighbors
  within specified radius; predicting NaN.")`; ferrolearn `predict` →
  `Err(InvalidParameter { reason: "query sample 0 has no neighbors within
  radius=0.01" })`.
- AC-8 (REQ-8, present & matching): reg `X=[[0],[1],[2]]`, `y=[0,10,20]`,
  `radius=1.5`, query `[[1.0]]` uniform → `[10.0]`; reg `X=[[0],[10]]`, `y=[0,100]`,
  `radius=15.0`, distance, query `[[1.0]]` → `[10.0]`. ferrolearn matches both.
- AC-9 (REQ-9, present): reg constant-/perfect-fit → `score == 1.0`; ferrolearn R²
  → `1.0` (in-tree exact-match regressor tests + `crate::knn::r2_score` parity).
- AC-10 (REQ-10, present & matching as set): reg `X=[[10,10],[1,0],[0,1],[0,0],
  [1,1]]`, query `[[0.2,0.1]]`, `radius=2.0` → sklearn set `{1,2,3,4}`, distances
  `{0.2236,0.8062,0.922,1.2042}`. ferrolearn `radius_neighbors(&xq, None)` →
  `i=[3,1,2,4]`, `d=[0.2236,0.8062,0.922,1.2042]` — matches as a set (= sklearn's
  `sort_results=True` order).
- AC-11 (REQ-11 pin): same fixture, `radius_neighbors([[0.2,0.1]])` default
  (`sort_results=False`) → `i=[[1,2,3,4]]` (native, dist `[0.8062,0.922,0.2236,
  1.2042]` NOT ascending); `sort_results=True` → `[[3,1,2,4]]`. ferrolearn always
  returns ascending `[3,1,2,4]` — matches `sort_results=True`, diverges from the
  default order.
- AC-12 (REQ-12 pin): clf no-neighbor + `outlier_label=None` → sklearn
  `ValueError("No neighbors found for test samples array([0]), you can try using
  larger radius, giving a label for outliers, or considering removing them from your
  dataset.")`; ferrolearn `FerroError::InvalidParameter { reason: "query sample 0
  has no neighbors within radius=0.01; set outlier_label to handle this case" }`.
- AC-13 (REQ-13 pin): reg `X=[[0],[1],[2]]`, `Y=[[0,100],[10,200],[20,300]]`,
  `radius=1.5`, query `[[1.0]]` → sklearn `(1,2)` `[[10.0,200.0]]`; ferrolearn
  `Fit<Array2<F>, Array1<F>>` cannot accept a 2-D `y`; no 2-D `predict`.
- AC-14 (REQ-14 surface): `RadiusNeighborsClassifier()` → `radius==1.0`,
  `weights=='uniform'`, `algorithm=='auto'`, `leaf_size==30`, `p==2`,
  `metric=='minkowski'`, `outlier_label==None`, `metric_params==None`,
  `n_jobs==None` (`RadiusNeighborsRegressor()` ditto, no `outlier_label`).
  ferrolearn `new()` has `radius`/`weights`/`algorithm` (+ clf `outlier_label`)
  only — REQ-14 FAILS on the missing fields + callable weights.
- AC-15 (REQ-15 surface): `import ferrolearn; ferrolearn.RadiusNeighborsClassifier`
  → AttributeError (no shim, `grep -ni radiusneighbors ferrolearn-python/src/`
  empty); the meta-crate has no re-export.

## REQ status table

Binary (R-DEFER-2). `RadiusNeighbors{Classifier,Regressor}` / their fitted types are
existing pub APIs re-exported at the crate root and consumed non-test by `graph.rs`
(the `radius_neighbors_graph` free function + the fitted `radius_neighbors_graph`
methods) + pipeline integration (the production-consumer surface; grandfathered
S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2).
Live oracle = installed sklearn 1.5.2, run from `/tmp`. Honest underclaim
(R-HONEST-3): eight REQs are SHIPPED (clf predict + tie-break, clf predict_proba
with-neighbors, clf score, clf outlier_label Some/None value, clf classes, reg
predict with-neighbors, reg score, shared radius set — every behavior that
value-matches the oracle with a non-test consumer); the rest are NOT-STARTED with
open prereq blockers (suggested numbers — the director creates the real issues).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (clf `predict` value + tie-break) | SHIPPED | impl `fn predict` for `FittedRadiusNeighborsClassifier` (per-row `fn find_radius_neighbors` → `fn weighted_vote` over `fn class_score_vec`, argmax tie-break to smallest label since `self.classes` is sorted) mirrors `RadiusNeighborsClassifier.predict` (`_classification.py:678-718`; `np.argmax` first-max `:707`). Non-test consumer: `graph.rs` `pub fn radius_neighbors_graph` builds a `RadiusNeighborsClassifier`, `clf.fit(x,&dummy_y)` → `fitted.radius_neighbors(...)` (the same fitted type whose predict path this is); also the `impl PipelineEstimator` wrapper. Live oracle (AC-1): `X=[[0,0],[0.5,0],[0,0.5],[5,5],[5.5,5],[5,5.5]]`, `y=[0,0,0,1,1,1]`, `radius=1.5`, query `[[0.2,0.1],[5.2,5.1]]` → uniform `[0,1]`, distance `[0,1]`; ferrolearn matches both. Verified throwaway probe. |
| REQ-2 (clf `predict_proba` value, with neighbors) | SHIPPED | impl `pub fn predict_proba` in `radius_neighbors.rs` (`fn class_score_vec` weighted sums, per-row `scores[ci]/total`, zero-dist branch) mirrors `predict_proba` (`_classification.py:720-836`, classes in `classes_` order, `np.bincount` `:803-810`). Non-test consumer: the fitted classifier type built/consumed by `graph.rs` `radius_neighbors_graph` (predict_proba itself is a pub method of that type; no separate binding getter — REQ-15). Live oracle (AC-2): `X=[[0],[1],[2],[10],[11]]`, `y=[0,1,0,1,1]`, `radius=5.0`, query `[[1.5]]`, distance → `[[0.5714285714285715,0.4285714285714286]]`, uniform → `[[0.6666666666666666,0.3333333333333333]]`; ferrolearn matches both to full precision. Verified probe. Empty/outlier-not-in-classes row handling is REQ-6. |
| REQ-3 (clf `score` accuracy) | SHIPPED | impl `pub fn score` for `FittedRadiusNeighborsClassifier` (`correct/n` over `predict`) mirrors `ClassifierMixin.score` (mean accuracy). Non-test consumer: the fitted classifier type built by `graph.rs` `radius_neighbors_graph` (predict path) + the pipeline wrapper. Live oracle (AC-3): `X=[[0,0],[0.5,0],[0,0.5],[5,5],[5.5,5],[5,5.5]]`, `y=[0,0,0,1,1,1]`, `radius=1.5` → `score==1.0`; ferrolearn `score` → `1.0`. Verified probe. |
| REQ-4 (clf `outlier_label` Some/None value) | SHIPPED | impl `fn predict` empty-neighborhood branch (`match self.outlier_label { Some(label) => push(label), None => Err(InvalidParameter) }`) mirrors sklearn's manual-outlier-label `predict` assign (`_classification.py:710-713`) and the `outlier_label_ is None` raise (`:781-787`). Non-test consumer: `graph.rs` `radius_neighbors_graph` constructs the clf (here with `dummy_y` / default `outlier_label`). Live oracle (AC-4): `outlier_label=Some(99)`, `radius=0.01`, far query `[[100]]` → `predict` `[99]`; ferrolearn `[99]`. `None` + outlier → both raise (type/message is REQ-12). `'most_frequent'` + manual-label-proba-all-zero are REQ-6. Verified probe + in-tree `test_classifier_with_outlier_label`, `test_classifier_no_neighbors_no_outlier_label_errors`. |
| REQ-5 (clf `classes` / `n_classes`) | SHIPPED | impl `pub fn classes` / `pub fn n_classes` for `FittedRadiusNeighborsClassifier` (sorted-unique labels) mirrors `classes_` lexicographic order. Non-test consumer: `graph.rs` `radius_neighbors_graph` builds + holds the fitted classifier; `class_score_vec`/`predict_proba`/`weighted_vote` all index by `self.classes`. In-tree `test_classifier_classes` pins `classes()==[0,1]`, `n_classes()==2` (AC-5). |
| REQ-6 (clf `'most_frequent'` + manual-label proba) | NOT-STARTED | open prereq blocker #881. `outlier_label: Option<usize>` has **no `'most_frequent'`** mode (sklearn assigns the training mode, `_classification.py:636-642`; live oracle `outlier_label_==[0]`), and `predict_proba`'s empty branch falls back to **uniform `1/n_classes`** when the label is not in `classes` (`binary_search` miss), whereas sklearn leaves the row **all-zero** then normalizes-to-0 (`_classification.py:813-824`). Pin (AC-6): `outlier_label='most_frequent'`,`y=[0,0,0,0,1,1]` → sklearn `[0]`; and `outlier_label=99` on `y=[0,1,0]`, far `predict_proba` → sklearn `[[0.0,0.0]]`, ferro `[[0.5,0.5]]`. |
| REQ-7 (reg no-neighbor NaN + warn — KEY) | NOT-STARTED | open prereq blocker #882. reg `fn predict` returns `Err(InvalidParameter)` on an empty neighborhood; sklearn assigns `np.nan` (`empty_obs = np.full_like(_y[0], np.nan)`, `_regression.py:482,487,498`) and emits `UserWarning("One or more samples have no neighbors within specified radius; predicting NaN.")` (`:504-509`) — **does not raise**. Pin (AC-7): `RadiusNeighborsRegressor(radius=0.01).fit([[0],[10]],[0,100]).predict([[5]])` → sklearn `[nan]`+warn; ferro `Err(InvalidParameter{reason:"query sample 0 has no neighbors within radius=0.01"})`. **The cleanest single-file deterministic fix** — the critic should pin this first (replace the regressor empty-row `Err` with `F::nan()` + an emitted warning). |
| REQ-8 (reg `predict` value, 1-D y) | SHIPPED | impl `fn predict` for `FittedRadiusNeighborsRegressor` (per-row `fn find_radius_neighbors` → `fn weighted_mean`: uniform mean, distance `Σ(y/d)/Σ(1/d)`, zero-dist averages coincident targets) mirrors `RadiusNeighborsRegressor.predict` (`_regression.py:459-514`) for 1-D `y` with neighbors present. Non-test consumer: `FittedRadiusNeighborsRegressor::radius_neighbors_graph` (`graph.rs`) consumes the same fitted type's `radius_neighbors`; the `impl PipelineEstimator` regressor wrapper consumes `predict`. Live oracle (AC-8): `X=[[0],[1],[2]]`,`y=[0,10,20]`,`radius=1.5`,query`[[1.0]]` uniform → `[10.0]`; `X=[[0],[10]]`,`y=[0,100]`,`radius=15.0`,distance,query`[[1.0]]` → `[10.0]`; ferrolearn matches both. Verified probe. Empty-row NaN is REQ-7; 2-D `y` is REQ-13. |
| REQ-9 (reg `score` R²) | SHIPPED | impl `pub fn score` for `FittedRadiusNeighborsRegressor` → `crate::knn::r2_score` (`1 - SS_res/SS_tot`; constant-`y` → `1.0` if `SS_res==0` else `-inf`) mirrors `RegressorMixin.score` = `r2_score(multioutput='uniform_average')`. Non-test consumer: the fitted regressor type behind `graph.rs` `radius_neighbors_graph` (predict path) + the pipeline wrapper; `r2_score` is the shared `knn.rs` symbolic constant whose oracle parity is established in `.design/neighbors/knn.md` (REQ-5/AC-5). With 1-D `y` the `uniform_average` averaging is a no-op (REQ-13). In-tree exact-match regressor tests pin perfect-fit behavior (AC-9). |
| REQ-10 (shared `radius_neighbors` set) | SHIPPED | impl `pub(crate) fn radius_neighbors_impl` + `pub fn radius_neighbors` (both fitted types) → per-row `fn find_radius_neighbors` (BallTree `within_radius` / `fn brute_force_radius`, then `sort_by` ascending) returns the in-radius `(distances, indices)` SET, the kernel both `predict` paths consume, mirroring `RadiusNeighborsMixin.radius_neighbors(X, radius)` **as a set**. Non-test consumers: `graph.rs` `radius_neighbors_graph` free function (`fitted.radius_neighbors(x, Some(radius))`) + `FittedRadiusNeighbors{Classifier,Regressor}::radius_neighbors_graph` (`self.radius_neighbors(x, radius)`). Live oracle (AC-10): `X=[[10,10],[1,0],[0,1],[0,0],[1,1]]`, query `[[0.2,0.1]]`, `radius=2.0` → set `{1,2,3,4}`, dist `{0.2236,0.8062,0.922,1.2042}`; ferrolearn → `i=[3,1,2,4]`, `d=[0.2236,0.8062,0.922,1.2042]` — matches as a set. Verified probe. Always-ascending sort vs `sort_results=False` default is REQ-11. |
| REQ-11 (`radius_neighbors` `sort_results` default + `X=None`) | NOT-STARTED | open prereq blocker #883. `fn find_radius_neighbors` ALWAYS sorts ascending (`results.sort_by(...)`), matching sklearn `sort_results=True`, but sklearn's DEFAULT is `sort_results=False` → native (tree/brute) order; there is no `sort_results` toggle and no `X=None` self-exclusion. Pin (AC-11): on `X=[[10,10],[1,0],[0,1],[0,0],[1,1]]`, `radius_neighbors([[0.2,0.1]])` default → `i=[[1,2,3,4]]` (native, dist `[0.8062,0.922,0.2236,1.2042]` NOT ascending); ferro `[3,1,2,4]` (ascending). SET matches, ORDER diverges. Same class as `nearest_neighbors.md` REQ-6 (#867). |
| REQ-12 (no-neighbor error type/message) | NOT-STARTED | open prereq blocker #884. clf `fn predict` / `pub fn predict_proba` empty branch raise `FerroError::InvalidParameter { reason: "query sample {i} has no neighbors within radius=...; set outlier_label to handle this case" }` where sklearn raises `ValueError("No neighbors found for test samples %r, you can try using larger radius, giving a label for outliers, or considering removing them from your dataset.")` (`_classification.py:781-787`). The regressor's raise where sklearn warns-not-raises is REQ-7. Pin (AC-12): clf far query, `outlier_label=None` → sklearn `ValueError("No neighbors found for test samples array([0]), ...")`; ferro `InvalidParameter`. |
| REQ-13 (reg multi-output 2-D y) | NOT-STARTED | open prereq blocker #885. `FittedRadiusNeighborsRegressor` stores `y_train: Array1<F>` and `impl Fit<Array2<F>, Array1<F>>` — 1-D `y` only; sklearn `RadiusNeighborsRegressor.predict` reshapes `_y` to 2-D and returns `(n_queries, n_outputs)` when `_y.ndim > 1` (`_regression.py:478-514`). Pin (AC-13): `fit(X, Y_2col).predict(Xq)` → sklearn `(1,2)` `[[10.0,200.0]]`; ferrolearn has no 2-D `y` surface. Same class as `.design/neighbors/knn.md` REQ-9 (#875). |
| REQ-14 (constructor params + callable weights) | NOT-STARTED | open prereq blocker #886. Both estimators have `radius`/`weights∈{Uniform,Distance}`/`algorithm` (+ clf `outlier_label`) but **no `leaf_size=30`** (`_classification.py:584`/`_regression.py:418`), **`p=2`** (`:585`/`:419`), **`metric='minkowski'`** (`:586`/`:421`), **`metric_params`** (`:588`/`:421`), **`n_jobs`** (`:589`/`:422`), and `Weights` has **no callable variant** (`_classification.py:573`/`_regression.py:408`). Euclidean-only. Pin (AC-14): `RadiusNeighborsClassifier().leaf_size==30`, `.p==2`, `.metric=='minkowski'`, `.metric_params==None`, `.n_jobs==None`; ferrolearn lacks all. Same class as `knn.md` REQ-10 (#876). |
| REQ-15 (PyO3 binding + meta-crate re-export) | NOT-STARTED | open prereq blocker #887. No `RsRadiusNeighborsClassifier`/`RsRadiusNeighborsRegressor` (or equivalent) in `ferrolearn-python/src/` (verified absent by `grep -ni radiusneighbors`) and no meta-crate re-export. `import ferrolearn` cannot construct either estimator. NOT-STARTED until the library REQs (esp. #882/#881) land and the shim exposes them. |
| REQ-16 (ferray substrate) | NOT-STARTED | open prereq blocker #888. `radius_neighbors.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` (the wrong substrate); not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`radius_neighbors.rs` follows the unfitted/fitted split (CLAUDE.md naming) for two
estimators that share `Weights` / `Algorithm` / `SpatialIndex` (the latter local to
this file, distinct from `nearest_neighbors.rs`'s), the `find_radius_neighbors`
radius kernel, and `radius_neighbors_impl`:

- `RadiusNeighborsClassifier<F>` (`radius`/`weights`/`algorithm`/`outlier_label`) →
  `Fit<Array2<F>, Array1<usize>>` → `FittedRadiusNeighborsClassifier<F>` (`x_train`,
  `y_train: Array1<usize>`, `radius`, `weights`, `spatial_index`, `outlier_label`,
  sorted `classes`).
- `RadiusNeighborsRegressor<F>` (`radius`/`weights`/`algorithm`) →
  `Fit<Array2<F>, Array1<F>>` → `FittedRadiusNeighborsRegressor<F>` (`x_train`,
  `y_train: Array1<F>`, `radius`, `weights`, `spatial_index`). No `classes`; **1-D
  `y_train`** (REQ-13).

Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (no panics in library code, R-CODE-2 — note the
`class_score_vec` closure's `.expect("label not in fitted classes")` is an internal
invariant on data built from `y_train`, not user input).

**Fit path (both `fn fit`).** Validation rejects `n_samples != y.len()`
(`ShapeMismatch`), `radius <= 0` (`InvalidParameter`), and `n_samples == 0`
(`InsufficientSamples`). The classifier then computes sorted-deduped `classes`;
both build the spatial index via `build_spatial_index` (`Auto` → KdTree ≤15
features else BallTree; otherwise direct). Neither performs sklearn's missing
`'most_frequent'`/manual `outlier_label_` resolution at fit (REQ-6).

**Classifier predict (`fn predict` → `fn weighted_vote` → `fn class_score_vec`).**
`class_score_vec` returns per-class weighted vote sums in sorted-`classes` order:
uniform → `+1` per neighbor; distance → `+1/d` per neighbor, **except** when any
neighbor has `d < eps=1e-15` (a coincident point), in which case only the
zero-distance neighbors get `+1` each — mirroring sklearn's `_get_weights`
zero-distance branch. `weighted_vote` argmaxes; because the scan starts at index 0
and only replaces on a **strict** `>`, equal scores keep the earlier (smaller)
label — the smallest-label tie-break matching `np.argmax`'s first-max
(`_classification.py:707`). On an **empty** neighborhood, `predict` returns
`outlier_label` if `Some` else `Err(InvalidParameter)` (REQ-4 value; REQ-12 message;
REQ-6 `'most_frequent'`).

**Classifier predict_proba (`pub fn predict_proba`).** Normalizes the same
`class_score_vec` per row (`scores[ci]/total`). On an empty row: one-hot the
`outlier_label` column if `Some` AND the label is in `classes` (`binary_search`
hit); **else uniform `1/n_classes`** — the latter diverges from sklearn's all-zero
(normalized-to-0) outlier-row for a label not in `classes_` (REQ-6).

**Regressor predict (`fn predict` → `fn weighted_mean`).** uniform → arithmetic
mean of the in-radius targets; distance → `Σ(w·y)/Σw` with `w=1/d`, **except** the
zero-distance branch averages only the coincident targets — mirroring
`_regression.py:484-502` + `_get_weights`. This is the **1-D** path with neighbors
present (REQ-8). On an **empty** neighborhood it returns `Err(InvalidParameter)` —
the KEY divergence from sklearn's `np.nan` + warn (REQ-7); sklearn's 2-D
multi-output path (`_regression.py:478-514`) is absent (REQ-13).

**Scoring.** Classifier `score` = `correct/n` (mean accuracy). Regressor `score` →
`crate::knn::r2_score`: `1 - SS_res/SS_tot`, constant-`y` (`SS_tot==0`) → `1.0` if
`SS_res==0` else `F::neg_infinity()` — the `RegressorMixin.score` convention. (With
1-D `y` the `multioutput='uniform_average'` averaging is a no-op; it would matter
only once REQ-13 lands.)

**Shared radius search (`fn radius_neighbors_impl` / `fn find_radius_neighbors`).**
`find_radius_neighbors` dispatches per query row: `BallTree::within_radius` (ball
mode) or `fn brute_force_radius` (KdTree + BruteForce modes both use brute force for
the radius query), converting to `(index, true-Euclidean-distance)` and **always
sorting ascending** by distance. `radius_neighbors_impl` validates feature count and
`radius > 0` and fills the jagged `(Vec<Vec<F>>, Vec<Vec<usize>>)` per-query result.
This always-sorted output matches sklearn's `sort_results=True` order but diverges
from the `sort_results=False` default native order (REQ-11); there is no toggle and
no `X=None` self-exclusion. Its oracle SET parity is the same Euclidean radius
search documented in `.design/neighbors/nearest_neighbors.md` (AC-2).

**Consumer wiring (`graph.rs`).** The non-test production consumers of this file:
- `pub fn radius_neighbors_graph` (free function) constructs a transient
  `RadiusNeighborsClassifier::new().with_radius(radius)`, `clf.fit(x, &dummy_y)`,
  then `fitted.radius_neighbors(x, Some(radius))`, then drops the self-edge (zero
  diagonal) — emulating sklearn's `include_self=False` / `X=None` graph semantics
  (`sklearn/neighbors/_graph.py:255`).
- `FittedRadiusNeighborsClassifier::radius_neighbors_graph` /
  `FittedRadiusNeighborsRegressor::radius_neighbors_graph` (`graph.rs`) call each
  estimator's `radius_neighbors(x, radius)`.
- `impl PipelineEstimator<F> for RadiusNeighbors{Classifier,Regressor}<F>` —
  `fit_pipeline`/`predict_pipeline` consume `fit`/`predict` in-crate (the classifier
  wrapper maps float labels → `usize` and back).

**Missing fitted attributes vs sklearn:** `effective_metric_` /
`effective_metric_params_` (`_classification.py:507-516`, `_regression.py:355-364`),
`n_features_in_` (`:518`/`:366`), `feature_names_in_` (`:523`/`:371`),
`outlier_label_` (`:532`, clf), `outputs_2d_` (`:536`). ferrolearn exposes
`classes()`/`n_classes()` (clf), `n_samples_fit()` (both).

**Invariants held vs sklearn:** clf `predict` value + smallest-label tie-break
(AC-1); `predict_proba` value with neighbors (AC-2); clf `score` accuracy (AC-3);
clf `outlier_label` Some/None predict value (AC-4); clf `classes` (AC-5); reg
`predict` value uniform/distance/zero-dist for 1-D `y` with neighbors (AC-8); reg
`score` R² (AC-9); the shared in-radius search SET (AC-10).

**Invariants NOT held vs sklearn:** `outlier_label='most_frequent'` + manual-label
all-zero proba (REQ-6); regressor no-neighbor NaN+warn (REQ-7, KEY);
`radius_neighbors` `sort_results=False` default order + `X=None` (REQ-11);
no-neighbor error type/message (REQ-12); regressor multi-output 2-D `y` (REQ-13);
constructor `leaf_size`/`p`/`metric`/`metric_params`/`n_jobs` + callable weights
(REQ-14); the PyO3 binding + meta-crate re-export (REQ-15); the ferray substrate
(REQ-16).

## Verification

Library crate (green at baseline `5def81e5` for the existing contract):
```
cargo test -p ferrolearn-neighbors --lib radius_neighbors
cargo clippy -p ferrolearn-neighbors --all-targets -- -D warnings
cargo fmt --all --check
```
The 39 in-tree `#[test]`s (`test_classifier_basic`,
`test_classifier_with_outlier_label`,
`test_classifier_no_neighbors_no_outlier_label_errors`,
`test_classifier_distance_weighting`, `test_classifier_exact_match_vote`,
`test_classifier_classes`, `test_classifier_brute_force`, `test_classifier_balltree`,
`test_classifier_pipeline`, `test_regressor_basic`, `test_regressor_mean_of_neighbors`,
`test_regressor_distance_weighting`, `test_regressor_exact_match_distance_weighting`,
`test_regressor_no_neighbors_errors`, `test_regressor_brute_force`,
`test_regressor_balltree`, `test_regressor_pipeline`, … plus shape/radius/f32/default
guards) pin ferrolearn's current radius classify/regress behavior. **None compares
against the live sklearn oracle**, but the eight SHIPPED REQs value-match the oracle
(verified by throwaway `cargo run --example` probe, since deleted), so they are
SHIPPED; the rest are NOT-STARTED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin the KEY deterministic one FIRST**:
REQ-7 (reg no-neighbor NaN+warn — the cleanest single-file fix; replace the
regressor empty-row `Err` with `F::nan()` + a warning), then REQ-11 (radius
native-vs-sorted order — a fixed permutation), REQ-6 (`'most_frequent'` + all-zero
proba), REQ-13 (multi-output 2-D `y`), then REQ-12/REQ-14/REQ-15 (message/surface):
```
# REQ-1 (present, must stay green): clf predict value (uniform + distance)
python3 -c "import numpy as np; from sklearn.neighbors import RadiusNeighborsClassifier as C; X=np.array([[0.,0.],[0.5,0.],[0.,0.5],[5.,5.],[5.5,5.],[5.,5.5]]); y=np.array([0,0,0,1,1,1]); Xq=np.array([[0.2,0.1],[5.2,5.1]]); print(C(radius=1.5).fit(X,y).predict(Xq).tolist(), C(radius=1.5,weights='distance').fit(X,y).predict(Xq).tolist())"  # [0,1] [0,1]
# REQ-2 (present): predict_proba distance + uniform
python3 -c "import numpy as np; from sklearn.neighbors import RadiusNeighborsClassifier as C; X=np.array([[0.],[1.],[2.],[10.],[11.]]); y=np.array([0,1,0,1,1]); q=np.array([[1.5]]); print(C(radius=5.0,weights='distance').fit(X,y).predict_proba(q).tolist(), C(radius=5.0).fit(X,y).predict_proba(q).tolist())"  # [[0.5714...,0.4286...]] [[0.6667...,0.3333...]]
# REQ-6 ('most_frequent' + manual-label all-zero proba):
python3 -c "import numpy as np; from sklearn.neighbors import RadiusNeighborsClassifier as C; X=np.array([[0.,0.],[0.5,0.],[0.,0.5],[5.,5.],[5.5,5.],[5.,5.5]]); y=np.array([0,0,0,0,1,1]); m=C(radius=0.01,outlier_label='most_frequent').fit(X,y); print(m.outlier_label_, m.predict(np.array([[100.,100.]])).tolist())"  # [0] [0]
python3 -c "import numpy as np; from sklearn.neighbors import RadiusNeighborsClassifier as C; m=C(radius=0.01,outlier_label=99).fit(np.array([[0.],[1.],[2.]]),np.array([0,1,0])); print(m.classes_.tolist(), m.predict_proba(np.array([[100.]])).tolist())"  # [0,1] [[0.0,0.0]]
# REQ-7 (reg no-neighbor NaN + warn — KEY):
python3 -c "import numpy as np,warnings; from sklearn.neighbors import RadiusNeighborsRegressor as R; \
\
ws=warnings.catch_warnings(record=True); ws.__enter__(); warnings.simplefilter('always'); \
p=R(radius=0.01).fit(np.array([[0.],[10.]]),np.array([0.,100.])).predict(np.array([[5.]])); print(p.tolist())"  # [nan]  (+ 'One or more samples have no neighbors within specified radius; predicting NaN.')
# REQ-8 (present): reg predict uniform + distance
python3 -c "import numpy as np; from sklearn.neighbors import RadiusNeighborsRegressor as R; print(R(radius=1.5).fit(np.array([[0.],[1.],[2.]]),np.array([0.,10.,20.])).predict(np.array([[1.0]])).tolist(), R(radius=15.0,weights='distance').fit(np.array([[0.],[10.]]),np.array([0.,100.])).predict(np.array([[1.0]])).tolist())"  # [10.0] [10.0]
# REQ-11 (radius sort_results default native order):
python3 -c "import numpy as np; from sklearn.neighbors import RadiusNeighborsRegressor as R; m=R(radius=2.0).fit(np.array([[10.,10.],[1.,0.],[0.,1.],[0.,0.],[1.,1.]]),np.zeros(5)); print([a.tolist() for a in m.radius_neighbors(np.array([[0.2,0.1]]))[1]], [a.tolist() for a in m.radius_neighbors(np.array([[0.2,0.1]]),sort_results=True)[1]])"  # [[1,2,3,4]] [[3,1,2,4]]
# REQ-12 (clf no-neighbor exact ValueError):
python3 -c "import numpy as np; from sklearn.neighbors import RadiusNeighborsClassifier as C; C(radius=0.01).fit(np.array([[0.,0.],[0.5,0.],[0.,0.5],[5.,5.],[5.5,5.],[5.,5.5]]),np.array([0,0,0,1,1,1])).predict(np.array([[100.,100.]]))"  # ValueError: No neighbors found for test samples array([0]), you can try using larger radius, ...
# REQ-13 (multi-output 2-D y):
python3 -c "import numpy as np; from sklearn.neighbors import RadiusNeighborsRegressor as R; Y=np.array([[0.,100.],[10.,200.],[20.,300.]]); p=R(radius=1.5).fit(np.array([[0.],[1.],[2.]]),Y).predict(np.array([[1.0]])); print(p.shape, p.tolist())"  # (1,2) [[10.0,200.0]]
# REQ-14 (defaults / missing params):
python3 -c "from sklearn.neighbors import RadiusNeighborsClassifier as C, RadiusNeighborsRegressor as R; c=C(); print(c.radius,c.weights,c.algorithm,c.leaf_size,c.p,c.metric,c.outlier_label,c.metric_params,c.n_jobs); r=R(); print(r.radius,r.weights,r.algorithm,r.leaf_size,r.p,r.metric,r.metric_params,r.n_jobs)"  # 1.0 uniform auto 30 2 minkowski None None None / 1.0 uniform auto 30 2 minkowski None None
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-neighbors/tests/divergence_radius_neighbors.rs`, asserting the
live-sklearn expected values above and FAILING against current
`radius_neighbors.rs`. REQ-1..5, REQ-8..10 already match and should be guarded by
non-regression pins (REQ-10 compares set + distance, not a fixed permutation, since
ferrolearn's always-ascending order differs from sklearn's default native order —
REQ-11 — and that order difference must NOT be pinned against REQ-10).

ferrolearn-python (REQ-15 binding parity, after the library REQs land):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_neighbors.py -q
```
asserting `ferrolearn.RadiusNeighborsClassifier` / `ferrolearn.Radius
NeighborsRegressor` match `sklearn.neighbors` on `radius`/`weights`/`algorithm`/
`outlier_label`, the `predict`/`predict_proba`/`score`/`radius_neighbors` outputs,
the reg no-neighbor NaN behavior, and the exact `ValueError` mapping.

## Blockers to open

(Director creates the real issues; numbers below are SUGGESTIONS continuing the
neighbors layer past `knn.md` #874-#878 and `nearest_neighbors.md` #865-#871.
#883 is the same `sort_results`-default class as `nearest_neighbors.md` #867;
#885/#886 are the same multi-output / constructor-param classes as `knn.md`
#875/#876.)

- #881 — REQ-6 (clf `'most_frequent'` + manual-label proba): add the
  `outlier_label='most_frequent'` mode (training mode, `_classification.py:636-642`)
  and make `predict_proba`'s empty row all-zero (not uniform) when the outlier label
  is not in `classes_` (`:813-824`). Pin: `outlier_label='most_frequent'` → `[0]`;
  `outlier_label=99`, far proba → `[[0.0,0.0]]`.
- #882 — REQ-7 (reg no-neighbor NaN + warn — KEY): the regressor `fn predict`
  returns `Err(InvalidParameter)` on an empty neighborhood; sklearn assigns
  `np.nan` and warns, does NOT raise (`_regression.py:482,504-509`). Replace the
  empty-row `Err` with `F::nan()` + an emitted warning. **The cleanest single-file
  deterministic fix** — the critic should pin this first. Pin:
  `RadiusNeighborsRegressor(radius=0.01).fit([[0],[10]],[0,100]).predict([[5]])` →
  sklearn `[nan]`+warn, ferro `Err`.
- #883 — REQ-11 (`radius_neighbors` `sort_results` default + `X=None`):
  `find_radius_neighbors` always sorts ascending; sklearn's default is
  `sort_results=False` (native order). Add a `sort_results` toggle defaulting to
  native order, and an `X=None` self-exclusion path. Pin: default → `[[1,2,3,4]]`,
  `sort_results=True` → `[[3,1,2,4]]`. (Same class as `nearest_neighbors.md` #867.)
- #884 — REQ-12 (no-neighbor error type/message): clf `predict`/`predict_proba`
  raise `FerroError::InvalidParameter` with a ferrolearn-specific message where
  sklearn raises `ValueError("No neighbors found for test samples %r, ...")`
  (`_classification.py:781-787`). Align the message at the binding boundary
  (`ValueError` mapping). Pin: clf far query, `outlier_label=None`.
- #885 — REQ-13 (regressor multi-output 2-D `y`): `FittedRadiusNeighborsRegressor`
  is `y_train: Array1<F>` / `Fit<…, Array1<F>>`; add a 2-D `y` path returning
  `(n_queries, n_outputs)` (`_regression.py:478-514`). Pin: `fit(X,Y_2col).
  predict(Xq)` → `(1,2)` `[[10.0,200.0]]`. (Same class as `knn.md` #875.)
- #886 — REQ-14 (constructor params + callable weights): no `leaf_size`/`p`/
  `metric`/`metric_params`/`n_jobs` fields on either estimator and no callable
  `Weights` variant (`_classification.py:584-589`/`:573`,
  `_regression.py:418-422`/`:408`). Add the param surface + a `metric` abstraction +
  callable weights. (Same class as `knn.md` #876.)
- #887 — REQ-15 (PyO3 binding + meta-crate re-export): no `RsRadiusNeighbors*` in
  `ferrolearn-python` and no meta-crate re-export; expose both estimators +
  `predict`/`predict_proba`/`score`/`radius_neighbors`/`outlier_label` at sklearn
  parity once #881/#882 land.
- #888 — REQ-16 (ferray substrate): migrate `radius_neighbors.rs` off
  `ndarray`/`num-traits` to `ferray-core` (R-SUBSTRATE).
