# MultiOutputClassifier / MultiOutputRegressor (multi-output meta-estimators)

<!--
tier: 3-component
status: draft
baseline-commit: c32b462f7bed67a4f8e2f55e361bc801e0b7b000
upstream-paths:
  - sklearn/multioutput.py            # _MultiOutputEstimator (:103), fit (:216-290), predict (:292-340), _fit_estimator (:62-68); MultiOutputRegressor (:342); MultiOutputClassifier (:445), fit (:517-557), classes_ (:555), predict_proba (:559-583), score (:585-635)
-->

## Summary

`ferrolearn-model-sel/src/multioutput.rs` mirrors scikit-learn's multi-output
meta-estimators from `sklearn/multioutput.py`: `MultiOutputRegressor` (`:342`)
and `MultiOutputClassifier` (`:445`), both built on the `_MultiOutputEstimator`
base (`:103`). The strategy is **one independent clone of the base estimator per
output column** (`fit` `:280-289`), with `predict` column-stacking the per-output
predictions into a `(n_samples, n_outputs)` matrix (`:333-339`). The base
estimator is expressed as a `PipelineFactory` CLOSURE
(`Box<dyn Fn() -> Pipeline<f64> + Send + Sync>`) rather than a wrapped, cloned
`estimator` object — the same sanctioned R-DEV-7 Rust idiom used in
`grid_search.rs`/`calibration.rs`/`multiclass.rs`, NOT a bug.

ferrolearn ships the **fit-per-column + predict-stack core end-to-end** for both
estimators, plus shape/empty-y validation. This is a **VERIFY-AND-DOCUMENT**
unit: the implemented surface MATCHES sklearn exactly (verified live below —
`MultiOutputRegressor(<deterministic base>).fit(X,Y).predict(X)` agrees
column-by-column with ferrolearn, including column independence, original output
order, and `(n_samples, n_targets)` shape). **There is NO deterministic fixable
divergence in the implemented surface for the critic to pin.** Every remaining
gap is a missing-feature blocker: `MultiOutputClassifier.predict_proba` (#1821),
per-output `classes_` (#1822), `score` (subset-accuracy / r2, #1823),
`sample_weight` threading (#1824), `partial_fit` (#1825), `n_jobs` +
`**fit_params` (#1826), and the `ferray` substrate (#1827).

## Upstream reference (scikit-learn 1.5.2, tag 1.5.2, commit 156ef14)

### `_MultiOutputEstimator` (`sklearn/multioutput.py:103`)
- `:103` — `class _MultiOutputEstimator(MetaEstimatorMixin, BaseEstimator,
  metaclass=ABCMeta)` — the shared fit/predict base.
- `:227` — `fit` requires the base supports fit:
  `if not hasattr(self.estimator, "fit"): raise ValueError(...)`.
- `:229` — `y = self._validate_data(X="no_validation", y=y, multi_output=True)`
  — validates `y` as 2D multi-output targets.
- `:234-238` — `if y.ndim == 1: raise ValueError("y must have at least two
  dimensions for multi-output regression but has only one.")`.
- `:280-289` — the core fit:
  `self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_estimator)(
  self.estimator, X, y[:, i], **routed_params.estimator.fit) for i in
  range(y.shape[1]))` — **ONE clone of the base estimator per output column
  `i`**, fit on `(X, y[:, i])`.
- `:260-279` — `sample_weight`/`**fit_params` routing: weights and fit-params
  are threaded into `routed_params.estimator.fit` and passed to every
  `_fit_estimator`; raises if a weight is given but the base lacks support.
- `:292-340` — `predict`:
  `y = Parallel(...)(delayed(e.predict)(X) for e in self.estimators_);
  return np.asarray(y).T` (`:311-314`) — per-output `predict`, then **transpose
  to `(n_samples, n_outputs)`**, columns in original output order.

### `_fit_estimator` (`sklearn/multioutput.py:62-68`)
- `:62-68` — `_fit_estimator(estimator, X, y, sample_weight=None, **fit_params)`:
  `estimator = clone(estimator)`; then `estimator.fit(X, y, ...)` — a **fresh
  clone per call** (the per-column independence guarantee).

### `MultiOutputRegressor` (`sklearn/multioutput.py:342`)
- `:342` — `class MultiOutputRegressor(RegressorMixin, _MultiOutputEstimator)`;
  `__init__(self, estimator, *, n_jobs=None)` — inherits `fit`/`predict` from the
  base unchanged.
- `:412` — `partial_fit` (incremental fit, base-supported only).
- `score` — inherited from `RegressorMixin`: R² with `uniform_average`
  multi-output averaging.

### `MultiOutputClassifier` (`sklearn/multioutput.py:445`)
- `:445` — `class MultiOutputClassifier(ClassifierMixin,
  _MultiOutputEstimator)`; `__init__(self, estimator, *, n_jobs=None)`.
- `:517-557` — `fit(X, Y, sample_weight=None, **fit_params)`: calls
  `super().fit(...)` then `:555` `self.classes_ = [estimator.classes_ for
  estimator in self.estimators_]` — a **LIST of per-output class arrays**.
- `:559-583` — `predict_proba`: `results = [estimator.predict_proba(X) for
  estimator in self.estimators_]; return results` — a **LIST of
  `(n_samples, n_classes_i)` arrays, one per output**; requires every
  sub-estimator implements `predict_proba` (`_check_predict_proba`, `:548-557`).
- `:585-635` — `score(X, y)`: **SUBSET ACCURACY** —
  `return np.mean(np.all(y == y_pred, axis=1))` (`:635`); raises if `y` is 1D
  (`:614-617`) or output count mismatches (`:618-622`).

## Requirements

R-DEV mental test applied per REQ ("numerical/API/structural contract" → MATCH;
"Cython/CPython footgun" → deviate; "missing feature" → NOT-STARTED with a
blocker). Per R-DEFER-2, classification is binary (SHIPPED / NOT-STARTED).

- REQ-FIT-PER-COLUMN (one estimator per output column): ferrolearn `fit` (both
  estimators) loops `for t in 0..n_targets`, takes `y_col = y.column(t)
  .to_owned()`, calls `(self.make_pipeline)()` for a FRESH pipeline, and fits it
  on `(x, y_col)` — pushing one `FittedPipeline` per column into `estimators`.
  Mirrors sklearn `Parallel(_fit_estimator(self.estimator, X, y[:, i]))`
  (`:280-289`) with `clone` per call (`:62-68`). `n_estimators() == n_targets()`.
  **MATCH** (R-DEV-1/R-DEV-3 — structural contract; per-column independence is
  the defining behavior). SHIPPED for BOTH `MultiOutputClassifier` and
  `MultiOutputRegressor`.
- REQ-PREDICT-STACK (predict = column-stack of per-output predictions, original
  order, `(n_samples, n_targets)`): ferrolearn `impl Predict` (both Fitted types)
  allocates `Array2::zeros((n_samples, self.n_targets))` and for each estimator
  `t` writes `result[[i, t]] = preds[i]` — column `t` is estimator `t`'s
  prediction, in fit order. Mirrors sklearn `np.asarray([e.predict(X) for e in
  estimators_]).T` (`:311-314`) — the `.T` puts output `i` in column `i`.
  **MATCH** (R-DEV-3 — output object shape/order contract). Verified
  END-TO-END below against a deterministic base (mean / linear): the full
  `(n_samples, n_targets)` matrix agrees column-by-column with sklearn, and a
  column-swap of `Y` permutes the predicted columns identically (independence).
  SHIPPED for BOTH.
- REQ-VALIDATION (shape + empty-y): ferrolearn `fit` rejects
  `y.nrows() != x.nrows()` with `FerroError::ShapeMismatch` and `n_targets == 0`
  (`y.ncols() == 0`) with `FerroError::InvalidParameter`. sklearn validates `y`
  as 2D multi-output (`:229`) and raises `ValueError` on a 1D `y` (`:234-238`).
  **MATCH (with a typed-deviation note, R-DEV-7).** The sklearn "y must have at
  least two dimensions" error is **structurally UNREACHABLE** in ferrolearn:
  the API takes `y: &Array2<f64>`, which is ALWAYS 2D — Rust's type system
  enforces at compile time what sklearn checks at runtime. The remaining runtime
  invariants (row-count agreement, non-empty target set) are both validated.
  SHIPPED.
- REQ-PREDICT-PROBA (`MultiOutputClassifier.predict_proba` — list of per-output
  prob arrays): ABSENT. `FittedMultiOutputClassifier` exposes only `n_targets()`,
  `n_estimators()`, and `Predict::predict` (label scores), no `predict_proba`.
  sklearn returns a LIST of `(n_samples, n_classes_i)` arrays (`:559-583`).
  **MATCH-intent / missing-feature.** NOT-STARTED (#1821).
- REQ-CLASSES (`MultiOutputClassifier` per-output `classes_`): ABSENT — there is
  no `classes()` accessor on `FittedMultiOutputClassifier`; the fitted pipelines
  do not surface per-output class arrays. sklearn sets `classes_ =
  [est.classes_ for est in estimators_]` (`:555`). **MATCH-intent /
  missing-feature.** NOT-STARTED (#1822).
- REQ-SCORE (`MultiOutputClassifier.score` = subset accuracy; `MultiOutputRegressor
  .score` = r2 uniform-avg): ABSENT — neither Fitted type defines `score`. sklearn
  MOC.score = `np.mean(np.all(y == y_pred, axis=1))` (`:635`); MOR.score = R²
  with `uniform_average` (RegressorMixin). **MATCH-intent / missing-feature.**
  NOT-STARTED (#1823).
- REQ-SAMPLE-WEIGHT (`sample_weight` threading): ABSENT — `fit(x, y)` takes only
  `(x, y)`, no weight channel. sklearn threads `sample_weight` into each
  `_fit_estimator` (`:280-289`, routed via `:260-279`). **MATCH-intent /
  missing-feature.** NOT-STARTED (#1824).
- REQ-PARTIAL-FIT (incremental `partial_fit`): ABSENT. sklearn exposes
  `partial_fit` on the base (`:119-214`) and `MultiOutputRegressor` (`:412`).
  **MATCH-intent / missing-feature.** NOT-STARTED (#1825).
- REQ-NJOBS-FIT-PARAMS (`n_jobs` parallelism + `**fit_params` threading): ABSENT
  — `new(make_pipeline)` takes only the factory; `fit(x, y)` only `(x, y)`. No
  `n_jobs` constructor param, no fit-param channel. sklearn parallelizes the
  per-output fits with `Parallel(n_jobs=self.n_jobs)` (`:280`) and threads
  `**fit_params` (`:260-289`). **MATCH-intent / missing-feature.** NOT-STARTED
  (#1826).
- REQ-X-1 (R-SUBSTRATE ndarray→ferray-core): production code imports
  `use ndarray::Array2` and operates on `Array2<f64>`; the destination substrate
  is `ferray-core` (R-SUBSTRATE-1). NOT-STARTED (#1827).
- REQ-X-2 (non-test production consumer): the boundary meta-estimator types
  `MultiOutputClassifier`/`MultiOutputRegressor`/`FittedMultiOutputClassifier`/
  `FittedMultiOutputRegressor` are the public API (S5 / R-DEFER-1) and are
  re-exported from `lib.rs`. SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side). The oracle is the
installed sklearn 1.5.2; run from `/tmp` (the source clone at
`/home/doll/scikit-learn` is the read-only cite tree, not built).

- AC-PREDICT-STACK (REQ-PREDICT-STACK + REQ-FIT-PER-COLUMN — SHIPPED, the core,
  verified END-TO-END with a deterministic base): use a base whose single-output
  fit is exactly recoverable — a no-noise linear function per output — so the
  predicted `(n_samples, n_outputs)` matrix can be matched element-for-element.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multioutput import MultiOutputRegressor
  from sklearn.linear_model import LinearRegression
  X = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.],[0.,2.],[2.,2.]])
  # y0 = 2*x0 + 3*x1 + 1 ;  y1 = -1*x0 + 4*x1 + 0   (exact, no noise)
  Y = np.column_stack([2*X[:,0]+3*X[:,1]+1.0, -1*X[:,0]+4*X[:,1]])
  mor = MultiOutputRegressor(LinearRegression()).fit(X, Y)
  P = mor.predict(X)
  print('shape:', P.shape)                         # -> (6, 2)
  print('matches Y exactly:', np.allclose(P, Y))   # -> True
  print('n_est:', len(mor.estimators_))            # -> 2
  # column independence: swapping output columns permutes predicted columns
  P2 = MultiOutputRegressor(LinearRegression()).fit(X, Y[:, ::-1]).predict(X)
  print('col-swap == swapped predict:', np.allclose(P2, P[:, ::-1]))  # -> True
  "
  # -> shape: (6, 2) / matches Y exactly: True / n_est: 2 / col-swap == swapped predict: True
  ```
  ferrolearn's implemented surface MATCHES: `fit` trains one pipeline per
  `y.column(t)`, `predict` writes `result[[i, t]]`, giving `(n_samples,
  n_targets)` with output `t` in column `t` and full per-column independence.
  The in-file tests are pinned against a `MeanEstimator` whose sklearn analog is
  `DummyRegressor(strategy='mean')` (next AC).
- AC-MEAN-BASE (REQ-PREDICT-STACK with the ferrolearn test fixture's base —
  SHIPPED, oracle-grounded): ferrolearn's `MeanEstimator` predicts the training
  mean of each output column; the sklearn analog is `DummyRegressor(
  strategy='mean')`. The oracle reproduces EXACTLY the constants the in-file
  tests assert.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multioutput import MultiOutputRegressor
  from sklearn.dummy import DummyRegressor
  X = np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
  Y = np.array([[2.,4.],[2.,4.],[2.,4.],[2.,4.]])
  P = MultiOutputRegressor(DummyRegressor(strategy='mean')).fit(X, Y).predict(X)
  print('two-output predict[0]:', P[0].tolist(), 'shape:', P.shape)
  X3 = np.array([[1.],[2.]]); Y3 = np.array([[1.,2.,3.],[1.,2.,3.]])
  P3 = MultiOutputRegressor(DummyRegressor(strategy='mean')).fit(X3, Y3).predict(X3)
  print('three-output predict[0]:', P3[0].tolist())
  "
  # -> two-output predict[0]: [2.0, 4.0] shape: (4, 2)
  # -> three-output predict[0]: [1.0, 2.0, 3.0]
  ```
  ferrolearn `test_mor_fit_predict_two_targets` asserts `preds[[i,0]]==2.0`,
  `preds[[i,1]]==4.0`; `test_mor_three_targets` asserts column constants
  `1.0, 2.0, 3.0`; `test_moc_fit_predict_two_targets` asserts the classifier
  column constants `0.0, 1.0` — each matching the oracle.
- AC-VALIDATION (REQ-VALIDATION — SHIPPED): ferrolearn rejects mismatched rows
  (ShapeMismatch) and empty targets (InvalidParameter). The sklearn 1D-y error
  is structurally unreachable under `Array2`.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multioutput import MultiOutputRegressor
  from sklearn.linear_model import LinearRegression
  X = np.zeros((4,2)); y1d = np.zeros(4)
  try:
      MultiOutputRegressor(LinearRegression()).fit(X, y1d)
  except ValueError as e:
      print('1D y raises ValueError:', 'two dimensions' in str(e))
  "
  # -> 1D y raises ValueError: True
  ```
  ferrolearn's `fit` signature takes `y: &Array2<f64>` (always 2D), so the
  sklearn 1D error cannot arise; `test_moc_shape_mismatch`/`test_mor_shape_mismatch`
  pin the row-count error and `test_moc_zero_targets`/`test_mor_zero_targets`
  pin the empty-target error.
- AC-PREDICT-PROBA (REQ-PREDICT-PROBA — ABSENT): sklearn MOC returns a LIST of
  per-output probability arrays.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multioutput import MultiOutputClassifier
  from sklearn.linear_model import LogisticRegression
  X = np.array([[0.],[1.],[2.],[3.],[4.],[5.]])
  Y = np.array([[0,0],[0,1],[0,1],[1,0],[1,1],[1,1]])
  moc = MultiOutputClassifier(LogisticRegression()).fit(X, Y)
  probs = moc.predict_proba(X)
  print('type:', type(probs).__name__, 'len:', len(probs), 'each shape:', [p.shape for p in probs])
  "
  # -> type: list len: 2 each shape: [(6, 2), (6, 2)]
  ```
  `grep -n "predict_proba" ferrolearn-model-sel/src/multioutput.rs` is empty.
  NOT-STARTED (#1821).
- AC-CLASSES (REQ-CLASSES — ABSENT): sklearn MOC exposes a list of per-output
  class arrays.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multioutput import MultiOutputClassifier
  from sklearn.linear_model import LogisticRegression
  X = np.array([[0.],[1.],[2.],[3.]]); Y = np.array([[0,5],[1,5],[0,7],[1,7]])
  moc = MultiOutputClassifier(LogisticRegression()).fit(X, Y)
  print('classes_:', [c.tolist() for c in moc.classes_])
  "
  # -> classes_: [[0, 1], [5, 7]]
  ```
  `FittedMultiOutputClassifier` has no `classes()` accessor. NOT-STARTED (#1822).
- AC-SCORE (REQ-SCORE — ABSENT): sklearn MOC.score is subset accuracy.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multioutput import MultiOutputClassifier
  from sklearn.dummy import DummyClassifier
  X = np.array([[0.],[1.],[2.],[3.]]); Y = np.array([[0,1],[0,1],[0,1],[0,1]])
  moc = MultiOutputClassifier(DummyClassifier(strategy='most_frequent')).fit(X, Y)
  print('subset-accuracy score:', moc.score(X, Y))
  "
  # -> subset-accuracy score: 1.0
  ```
  No `score` on either Fitted type. NOT-STARTED (#1823).
- AC-SAMPLE-WEIGHT / AC-NJOBS-FIT-PARAMS (REQ-SAMPLE-WEIGHT, REQ-NJOBS-FIT-PARAMS
  — ABSENT): sklearn accepts `n_jobs` and threads `sample_weight`/`**fit_params`.
  ```
  cd /tmp && python3 -c "
  import inspect
  from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
  print('MOR init:', list(inspect.signature(MultiOutputRegressor.__init__).parameters)[1:])
  print('MOR fit :', list(inspect.signature(MultiOutputRegressor.fit).parameters)[1:])
  "
  # -> MOR init: ['estimator', 'n_jobs']
  # -> MOR fit : ['X', 'y', 'sample_weight', 'fit_params']
  ```
  ferrolearn `new(make_pipeline)` and `fit(x, y)` expose none of these channels.
  NOT-STARTED (#1824, #1826). `partial_fit` likewise absent (#1825).
- AC-X-2 (REQ-X-2 — SHIPPED): `grep -n "pub use multioutput" ferrolearn-model-sel
  /src/lib.rs` shows the re-export of `FittedMultiOutputClassifier,
  FittedMultiOutputRegressor, MultiOutputClassifier, MultiOutputRegressor`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-FIT-PER-COLUMN (one estimator per output column) | SHIPPED | impl `pub fn fit in multioutput.rs` (BOTH `MultiOutputClassifier` and `MultiOutputRegressor`): `for t in 0..n_targets { let y_col = y.column(t).to_owned(); let pipeline = (self.make_pipeline)(); let fitted = pipeline.fit(x, &y_col)?; estimators.push(fitted); }` — one FRESH pipeline per output column, fit on that column. Mirrors sklearn `self.estimators_ = Parallel(...)(delayed(_fit_estimator)(self.estimator, X, y[:, i], ...) for i in range(y.shape[1]))` (`sklearn/multioutput.py:280-289`) with `clone(estimator)` per call (`:62-68`). `FittedMultiOutputClassifier::n_estimators()`/`n_targets()` and the MOR analog confirm `n_estimators == n_targets`. LIVE ORACLE (AC-PREDICT-STACK): `len(mor.estimators_) == 2` for a 2-output dataset; ferrolearn `test_mor_three_targets` asserts `n_targets()==3` / `n_estimators` per-column. Non-test consumer: REQ-X-2 (boundary type + `lib.rs` re-export). |
| REQ-PREDICT-STACK (predict = column-stack, original order, `(n_samples, n_targets)`) | SHIPPED | impl `impl Predict<Array2<f64>> for FittedMultiOutputClassifier`/`...Regressor` in `multioutput.rs`: `let mut result = Array2::<f64>::zeros((n_samples, self.n_targets)); for (t, est) in self.estimators.iter().enumerate() { let preds = est.predict(x)?; for i in 0..n_samples { result[[i, t]] = preds[i]; } }` — output `t` lands in column `t`, fit order preserved. Mirrors sklearn `y = Parallel(...)(delayed(e.predict)(X) for e in self.estimators_); return np.asarray(y).T` (`sklearn/multioutput.py:311-314`). LIVE ORACLE (AC-PREDICT-STACK, deterministic LinearRegression base): `MultiOutputRegressor(LinearRegression()).fit(X,Y).predict(X)` shape `(6,2)`, `np.allclose(P, Y) == True` (exact column-by-column recovery of two distinct no-noise linear outputs), and a column-swap of `Y` permutes the predicted columns identically (`col-swap == swapped predict: True`) — confirming per-column INDEPENDENCE and ORDER. AC-MEAN-BASE: with `DummyRegressor(strategy='mean')` (ferrolearn's `MeanEstimator` analog) `predict[0] == [2.0, 4.0]` and 3-output `[1.0, 2.0, 3.0]` — exactly the constants `test_mor_fit_predict_two_targets`/`test_mor_three_targets`/`test_moc_fit_predict_two_targets` assert. VERIFIED MATCH, no divergence. Non-test consumer: REQ-X-2. |
| REQ-VALIDATION (shape + empty-y; 1D-y unreachable by typing) | SHIPPED | impl `pub fn fit in multioutput.rs` (both): `if y.nrows() != n_samples { return Err(FerroError::ShapeMismatch { ... }) }` and `if n_targets == 0 { return Err(FerroError::InvalidParameter { name: "y", reason: "target matrix must have at least one column" }) }`. sklearn validates 2D multi-output `y` (`sklearn/multioutput.py:229`) and raises `ValueError` on a 1D `y` (`:234-238`). R-DEV-7 NOTE: ferrolearn's `fit` takes `y: &Array2<f64>` — ALWAYS 2D — so the sklearn "y must have at least two dimensions" runtime error is STRUCTURALLY UNREACHABLE (the type system enforces it at compile time); the residual runtime invariants (row count, non-empty targets) are both checked. LIVE ORACLE (AC-VALIDATION): sklearn 1D-y → `ValueError('...two dimensions...')`; ferrolearn forbids it at the type level. Pinned by `test_moc_shape_mismatch`/`test_mor_shape_mismatch` (row count) and `test_moc_zero_targets`/`test_mor_zero_targets` (empty). Non-test consumer: REQ-X-2. |
| REQ-PREDICT-PROBA (MOC predict_proba — list of per-output prob arrays) | NOT-STARTED | open prereq blocker #1821. `FittedMultiOutputClassifier` in `multioutput.rs` exposes only `n_targets()`, `n_estimators()`, and `impl Predict::predict` (label scores) — NO `predict_proba` (`grep -n "predict_proba" multioutput.rs` is empty). sklearn `predict_proba` (`sklearn/multioutput.py:559-583`) returns a LIST of `(n_samples, n_classes_i)` arrays, one per output, requiring each sub-estimator implement `predict_proba`. LIVE ORACLE (AC-PREDICT-PROBA): `MultiOutputClassifier(LogisticRegression()).fit(X,Y).predict_proba(X)` is a `list` of length 2 with shapes `[(6,2),(6,2)]`; ferrolearn cannot produce probabilities. Absent end-to-end. |
| REQ-CLASSES (MOC per-output `classes_`) | NOT-STARTED | open prereq blocker #1822. `FittedMultiOutputClassifier` has no `classes()` accessor and stores no per-output class arrays (`grep -n "classes" multioutput.rs` finds only the doc-comment "class labels"). sklearn sets `self.classes_ = [estimator.classes_ for estimator in self.estimators_]` (`sklearn/multioutput.py:555`). LIVE ORACLE (AC-CLASSES): `MultiOutputClassifier(LogisticRegression()).fit(X,Y).classes_ == [[0,1],[5,7]]`; ferrolearn exposes nothing analogous. Absent end-to-end. |
| REQ-SCORE (MOC subset accuracy / MOR r2 uniform-avg) | NOT-STARTED | open prereq blocker #1823. Neither `FittedMultiOutputClassifier` nor `FittedMultiOutputRegressor` defines a `score` method (`grep -n "fn score" multioutput.rs` is empty). sklearn MOC.score = `np.mean(np.all(y == y_pred, axis=1))` SUBSET ACCURACY (`sklearn/multioutput.py:635`); MOR.score = R² with `uniform_average` (RegressorMixin). LIVE ORACLE (AC-SCORE): `MultiOutputClassifier(DummyClassifier('most_frequent')).fit(X,Y).score(X,Y) == 1.0` (subset accuracy); ferrolearn has no `score`. Absent end-to-end. |
| REQ-SAMPLE-WEIGHT (sample_weight threading) | NOT-STARTED | open prereq blocker #1824. impl `pub fn fit in multioutput.rs` takes only `(x: &Array2<f64>, y: &Array2<f64>)` — no `sample_weight` channel. sklearn threads `sample_weight` into each per-column `_fit_estimator` (`sklearn/multioutput.py:280-289`, routed at `:260-279`), raising if the base lacks weight support. LIVE ORACLE (AC-SAMPLE-WEIGHT): `MultiOutputRegressor.fit` signature is `['X','y','sample_weight','fit_params']`; ferrolearn has no weight argument. Absent end-to-end. |
| REQ-PARTIAL-FIT (incremental partial_fit) | NOT-STARTED | open prereq blocker #1825. No `partial_fit` on either estimator (`grep -n "partial_fit" multioutput.rs` is empty). sklearn exposes `partial_fit` on `_MultiOutputEstimator` (`sklearn/multioutput.py:119-214`) and `MultiOutputRegressor` (`:412`), cloning on first call and incrementally updating per-output estimators. Absent end-to-end. |
| REQ-NJOBS-FIT-PARAMS (n_jobs parallelism + **fit_params threading) | NOT-STARTED | open prereq blocker #1826. impl `pub fn new in multioutput.rs` takes ONLY `make_pipeline: PipelineFactory`; `fit(x, y)` takes only `(x, y)`. No `n_jobs` constructor param, no `**fit_params` channel. sklearn `__init__(estimator, *, n_jobs=None)` (`sklearn/multioutput.py:342`/`:445`) parallelizes the per-output fits with `Parallel(n_jobs=self.n_jobs)` (`:280`) and threads `**fit_params` to each sub-fit (`:260-289`). LIVE ORACLE (AC-NJOBS-FIT-PARAMS): MOR init params `['estimator','n_jobs']`, fit params include `fit_params`; ferrolearn exposes neither. Absent end-to-end. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker #1827. Production code in `multioutput.rs` imports `use ndarray::Array2` and operates on `Array2<f64>` throughout (`fit` builds `Array2::<f64>::zeros((n_samples, self.n_targets))`, `predict` returns `Array2<f64>`). Per R-SUBSTRATE-1 the destination array type is `ferray-core`, not `ndarray`. Not migrated (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | Crate re-export: `ferrolearn-model-sel/src/lib.rs` `pub mod multioutput;` + `pub use multioutput::{FittedMultiOutputClassifier, FittedMultiOutputRegressor, MultiOutputClassifier, MultiOutputRegressor};`. Per S5 / R-DEFER-1 the boundary meta-estimator types ARE the public API and are grandfathered (existing pub surface). CAVEAT (honest underclaim): `grep -rn "MultiOutputClassifier\|MultiOutputRegressor" ferrolearn-*/src/ \| grep -v 'tests\|multioutput.rs'` finds the `lib.rs` re-export plus a `chain.rs` DOC-COMMENT cross-reference (`//! - [\`ClassifierChain\`] — like [\`MultiOutputClassifier\`]`) — no dedicated non-test internal CALLER and NO `ferrolearn-python` binding yet. SHIPPED on the boundary re-export per S5, not a dedicated production caller; the missing Python binding is noted. The base estimator is supplied as a `PipelineFactory` CLOSURE (an R-DEV-7 Rust idiom for sklearn's wrapped/`clone`d-`estimator` pattern) — noted, not pinned. |

## Architecture

ferrolearn splits each meta-estimator into an unfitted/Fitted pair (CLAUDE.md
naming): `MultiOutputClassifier { make_pipeline: PipelineFactory }` →
`FittedMultiOutputClassifier { estimators: Vec<FittedPipeline<f64>>, n_targets:
usize }`, and the structurally IDENTICAL `MultiOutputRegressor` →
`FittedMultiOutputRegressor`. sklearn instead has a shared `_MultiOutputEstimator`
base (`:103`) whose post-`fit` state is `estimators_` (+ `MultiOutputClassifier`
adds `classes_`, `:555`). The two ferrolearn estimators are byte-for-byte the
same fit/predict logic differing only in their type names — a deliberate
duplication of the sklearn base/subclass split.

**The base-estimator representation is an R-DEV-7 deviation.** sklearn wraps an
`estimator` object and `clone`s it per output column (`_fit_estimator`,
`:62-68`); ferrolearn takes a `make_pipeline: Box<dyn Fn() -> Pipeline<f64> +
Send + Sync>` closure called once per column to produce a fresh binary pipeline.
This is the sanctioned Rust analog (same idiom as
`grid_search.rs`/`calibration.rs`/`multiclass.rs`) — noted, not pinned. Both
produce the per-column INDEPENDENCE the strategy requires (fresh state per
output).

**The fit-per-column + predict-stack core (REQ-FIT-PER-COLUMN,
REQ-PREDICT-STACK — SHIPPED) is a faithful 1:1 translation.** `fit` loops over
`y.column(t)`, fits a fresh pipeline per column, and stores them in order.
`predict` allocates `(n_samples, n_targets)` and writes column `t` from
estimator `t` — the materialized analog of sklearn's `np.asarray([e.predict(X)
for e]).T` (`:311-314`). Verified END-TO-END against the live oracle with a
deterministic base (LinearRegression with two distinct no-noise linear outputs):
the full predicted matrix equals sklearn's column-by-column, including original
output order and per-column independence under a `Y` column-swap. The in-file
tests pin the same behavior with a `MeanEstimator` whose exact sklearn analog is
`DummyRegressor(strategy='mean')` — the asserted constants (`[2.0, 4.0]`,
`[1.0, 2.0, 3.0]`, classifier `[0.0, 1.0]`) match the oracle. **There is NO
hidden divergence in the implemented surface** — no stacking-order bug, no
column-aliasing, no validation gap within the typed contract — so this unit
has nothing for the critic to pin as a FAILING test.

**Validation (REQ-VALIDATION — SHIPPED).** Row-count mismatch →
`FerroError::ShapeMismatch`; empty target matrix → `FerroError::InvalidParameter`.
sklearn's 1D-y `ValueError` (`:234-238`) is structurally unreachable because the
Rust API types `y` as `Array2<f64>` (always 2D) — a typed-deviation note, not a
behavioral gap (R-DEV-7).

What is structurally ABSENT vs sklearn (all missing-feature blockers, none
fixable in the implemented surface): `MultiOutputClassifier.predict_proba`
(REQ-PREDICT-PROBA, #1821), per-output `classes_` (REQ-CLASSES, #1822), `score`
— MOC subset accuracy / MOR r2 (REQ-SCORE, #1823), `sample_weight` threading
(REQ-SAMPLE-WEIGHT, #1824), `partial_fit` (REQ-PARTIAL-FIT, #1825), `n_jobs` +
`**fit_params` (REQ-NJOBS-FIT-PARAMS, #1826), and the `ferray` substrate
(REQ-X-1, #1827). SHIPPED: the fit-per-column + predict-stack core + validation
(REQ-FIT-PER-COLUMN, REQ-PREDICT-STACK, REQ-VALIDATION) and the boundary
re-export (REQ-X-2).

Invariants: `y.nrows() == x.nrows()` (`FerroError::ShapeMismatch`);
`y.ncols() >= 1` (`FerroError::InvalidParameter`); `n_estimators == n_targets`;
`predict` returns `(n_samples, n_targets)` with output `t` in column `t`, fit
order preserved; each output's estimator is independent (fresh pipeline).

## Verification

Commands establishing the SHIPPED claims (baseline
`c32b462f7bed67a4f8e2f55e361bc801e0b7b000`). The oracle is the installed sklearn
1.5.2 (`cd /tmp`; the source clone at `/home/doll/scikit-learn` is the read-only
cite tree):

- `cargo test -p ferrolearn-model-sel --lib multioutput` → **10 passed, 0 failed**
  (`multioutput::tests::{test_moc_fit_predict_two_targets, test_moc_shape_mismatch,
  test_moc_zero_targets, test_moc_single_target, test_mor_fit_predict_two_targets,
  test_mor_with_sum_estimator, test_mor_shape_mismatch, test_mor_zero_targets,
  test_mor_single_target, test_mor_three_targets}`).
- REQ-PREDICT-STACK + REQ-FIT-PER-COLUMN SHIPPED oracle (live sklearn,
  deterministic LinearRegression base, R-CHAR-3): AC-PREDICT-STACK —
  `MultiOutputRegressor(LinearRegression()).fit(X,Y).predict(X)` shape `(6,2)`,
  `np.allclose(P, Y) == True` (exact column-by-column recovery), `len(estimators_)
  == 2`, and column-swap independence `col-swap == swapped predict: True`.
  AC-MEAN-BASE — `DummyRegressor(strategy='mean')` (ferrolearn `MeanEstimator`
  analog) gives `predict[0] == [2.0, 4.0]` and 3-output `[1.0, 2.0, 3.0]`,
  matching `test_mor_fit_predict_two_targets`/`test_mor_three_targets`/
  `test_moc_fit_predict_two_targets`.
- REQ-VALIDATION SHIPPED oracle: AC-VALIDATION — sklearn 1D `y` →
  `ValueError('...two dimensions...')`; ferrolearn's `Array2<f64>` typing makes
  this unreachable; row-count and empty-target invariants pinned by the four
  `*_shape_mismatch`/`*_zero_targets` tests.
- REQ-PREDICT-PROBA ABSENT oracle (#1821): AC-PREDICT-PROBA —
  `MultiOutputClassifier(LogisticRegression()).fit(X,Y).predict_proba(X)` is a
  `list` of length 2, shapes `[(6,2),(6,2)]`; `grep -n "predict_proba"
  multioutput.rs` empty.
- REQ-CLASSES ABSENT oracle (#1822): AC-CLASSES —
  `MultiOutputClassifier(...).fit(X,Y).classes_ == [[0,1],[5,7]]`; no `classes()`
  on `FittedMultiOutputClassifier`.
- REQ-SCORE ABSENT oracle (#1823): AC-SCORE —
  `MultiOutputClassifier(DummyClassifier('most_frequent')).fit(X,Y).score(X,Y)
  == 1.0` (subset accuracy); no `fn score` in `multioutput.rs`.
- REQ-SAMPLE-WEIGHT / REQ-NJOBS-FIT-PARAMS ABSENT oracle (#1824/#1826):
  AC-SAMPLE-WEIGHT / AC-NJOBS-FIT-PARAMS — `MultiOutputRegressor.__init__` params
  `['estimator','n_jobs']`, `fit` params include `sample_weight`/`fit_params`;
  ferrolearn `new(make_pipeline)`/`fit(x,y)` expose none. REQ-PARTIAL-FIT (#1825):
  no `partial_fit` symbol.
- REQ-X-2 consumer: `grep -n "pub use multioutput" ferrolearn-model-sel/src/lib.rs`
  shows `pub use multioutput::{FittedMultiOutputClassifier,
  FittedMultiOutputRegressor, MultiOutputClassifier, MultiOutputRegressor};`.
  `grep -rn "MultiOutputClassifier\|MultiOutputRegressor" ferrolearn-*/src/ |
  grep -v 'tests\|multioutput.rs'` shows the re-export + a `chain.rs`
  doc-comment cross-reference only (no dedicated internal caller, no Python
  binding — honest underclaim).
- REQ-X-1 substrate: `grep -n "ndarray" ferrolearn-model-sel/src/multioutput.rs`
  shows `use ndarray::Array2` — wrong substrate, migration owed (#1827).

SHIPPED: REQ-FIT-PER-COLUMN (one estimator per output column — both estimators),
REQ-PREDICT-STACK (column-stack `(n_samples, n_targets)` — VERIFIED END-TO-END
column-by-column vs sklearn LinearRegression base, with column independence),
REQ-VALIDATION (shape + empty-y; 1D-y unreachable by typing), REQ-X-2 (boundary
re-export consumer; no dedicated caller / no Python binding — honest underclaim).
NOT-STARTED: REQ-PREDICT-PROBA (#1821), REQ-CLASSES (#1822), REQ-SCORE (#1823),
REQ-SAMPLE-WEIGHT (#1824), REQ-PARTIAL-FIT (#1825), REQ-NJOBS-FIT-PARAMS (#1826),
REQ-X-1 (ferray substrate, #1827).

Per R-DEFER-2 every REQ is binary SHIPPED/NOT-STARTED. This is a
**VERIFY-AND-DOCUMENT** unit: the implemented surface (fit-per-column +
predict-stack + validation) MATCHES sklearn exactly — verified END-TO-END against
the live oracle — so **there is NO deterministic fixable divergence for the
critic to pin as a FAILING test.** The remaining NOT-STARTED REQs are
missing-feature blockers (#1821-#1826) or substrate (#1827) — blockers, not pins.
The next ACToR step is therefore acto-builder on the highest-value missing
feature (predict_proba + classes_ + score for `MultiOutputClassifier`, #1821-#1823),
NOT a critic divergence pin.

Least-confident SHIPPED claim: REQ-X-2 — SHIPPED rests on the `lib.rs` boundary
re-export (S5 grandfathering), but there is NO dedicated non-test internal caller
and NO `ferrolearn-python` binding for either estimator yet; the only other
reference is a `chain.rs` doc-comment cross-reference. The honest reading is
"public API surface present at the crate boundary," not "exercised by a
production consumer pipeline."
