# validation_curve

<!--
tier: 3-component
status: draft
baseline-commit: a09c63dd661f684379b70a0ac8866d1b62e52863
upstream-paths:
  - sklearn/model_selection/_validation.py   # validation_curve (:2149), _fit_and_score (:729), _aggregate_score_dicts (:2319)
-->

## Summary

`ferrolearn-model-sel/src/validation_curve.rs` mirrors scikit-learn's
`validation_curve` (`sklearn/model_selection/_validation.py:2149`) — the utility
that, for each value of a single hyperparameter, runs cross-validation and reports
per-fold TRAIN and TEST scores, yielding two `(n_ticks, n_cv_folds)` matrices for
diagnosing under/over-fitting.

ferrolearn exposes `pub fn validation_curve(x, y, cv: &dyn CrossValidator,
param_values: &[f64], make_pipeline: impl Fn(f64) -> Pipeline, scoring: fn(...) ->
Result<f64, FerroError>) -> Result<ValidationCurveResult, FerroError>`. The result
struct `ValidationCurveResult { param_values: Vec<f64>, train_scores:
Array2<f64> (n_params, n_folds), test_scores: Array2<f64> (n_params, n_folds) }`
matches sklearn's returned shape and orientation.

The CORE MECHANIC is faithful: ferrolearn fills the `(n_params, n_folds)` matrices
in `(param outer, fold inner)` row-major order, which is element-for-element
IDENTICAL to sklearn's `(fold outer, param inner)` flat list `.reshape(-1,
n_params).T` (verified below — this ordering-equivalence is a real REQ and it
agrees). ferrolearn always returns train scores, matching sklearn's hardcoded
`return_train_score=True`.

Several SURROUNDING behaviors diverge and are NOT-STARTED with concrete blockers:
ferrolearn requires an explicit `&dyn CrossValidator` (no default 5-fold / no
classifier-aware StratifiedKFold), an explicit `scoring` fn (no default
accuracy/r2 via `check_scoring`), propagates a fit/score error via `?` instead of
sklearn's `error_score=np.nan` continue-the-curve path, has no `groups` channel,
and restricts the varied parameter to `f64` (sklearn varies arbitrary param types
via `set_params`).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### `validation_curve` (`sklearn/model_selection/_validation.py:2149`)
- `:2149-2164` — signature `validation_curve(estimator, X, y, *, param_name,
  param_range, groups=None, cv=None, scoring=None, n_jobs=None,
  pre_dispatch="all", verbose=0, error_score=np.nan, fit_params=None)`. All params
  after `y` are keyword-only.
- `:2197-2200` — `groups`: "Only used in conjunction with a 'Group' cv instance",
  threaded to `cv.split(X, y, groups)` at `:2307`.
- `:2202-2214` — `cv` default `None` → 5-fold; "if the estimator is a classifier
  and y is either binary or multiclass, StratifiedKFold is used. In all other
  cases, KFold is used. These splitters are instantiated with shuffle=False".
- `:2222-2225` — `scoring` default `None`; resolved by `check_scoring` at `:2287`
  (regressor → r2, classifier → accuracy).
- `:2243-2246` — `error_score : 'raise' or numeric, default=np.nan`. "Value to
  assign to the score if an error occurs in estimator fitting. If a numeric value
  is given, FitFailedWarning is raised."
- `:2257-2261` — Returns `train_scores : array of shape (n_ticks, n_cv_folds)` and
  `test_scores : array of shape (n_ticks, n_cv_folds)`.
- `:2284` — `X, y, groups = indexable(X, y, groups)`.
- `:2286` — `cv = check_cv(cv, y, classifier=is_classifier(estimator))`.
- `:2287` — `scorer = check_scoring(estimator, scoring=scoring)`.
- `:2290-2309` — `results = parallel(delayed(_fit_and_score)(clone(estimator), X,
  y, scorer=scorer, train=train, test=test, parameters={param_name: v},
  return_train_score=True, error_score=error_score) ...)`. The iteration order
  (`:2306-2308`): `for train, test in cv.split(X, y, groups) for v in
  param_range` — **fold OUTER, param INNER** (the `# NOTE do not change order of
  iteration` comment at `:2306` pins this as load-bearing).
- `:2310` — `n_params = len(param_range)`.
- `:2312-2314` — `results = _aggregate_score_dicts(results); train_scores =
  results["train_scores"].reshape(-1, n_params).T; test_scores = ... .T`. The flat
  list (length `n_folds * n_params`, fold-major) is reshaped to `(n_folds,
  n_params)` then transposed to `(n_params, n_folds)`.
- `:2316` — `return train_scores, test_scores`.

### `_fit_and_score` (`sklearn/model_selection/_validation.py:729`)
The per-`(fold, param)` primitive: clones the estimator, applies `parameters`
(`set_params`), fits on `X[train]`/`y[train]`, scores on train (when
`return_train_score=True`) and test. On a fit failure with numeric `error_score`,
it sets the score to `error_score` (np.nan) and raises `FitFailedWarning` rather
than aborting (the `error_score` branch).

### `_aggregate_score_dicts` (`sklearn/model_selection/_validation.py:2319`)
Turns the list-of-dicts (one per `(fold, param)`) into a dict-of-arrays; the
`"train_scores"`/`"test_scores"` arrays are flat, fold-major.

## Requirements

- REQ-1 (core mechanic + iteration-order equivalence — the KEY REQ): for each
  `(param, fold)` pair, fit a fresh pipeline on the fold's train subset and score
  on both train and test, producing `(n_params, n_folds)` train and test matrices.
  ferrolearn fills the matrices `(param OUTER, fold INNER)` row-major; sklearn
  builds a `(fold OUTER, param INNER)` flat list then `.reshape(-1, n_params).T`.
  These yield element `[param i][fold j]` IDENTICALLY (proven below).
  **MATCH** (R-DEV-1 numerical/structural contract): the observable
  `(n_ticks, n_cv_folds)` matrices and their element ordering are the contract; the
  loop nesting is an internal re-indexing that produces the same array. SHIPPED.
- REQ-2 (return_train_score always on): ferrolearn unconditionally computes and
  returns `train_scores`, matching sklearn's hardcoded `return_train_score=True`
  (`:2303`). **MATCH** (R-DEV-3 output-object contract). SHIPPED.
- REQ-3 (result shape + orientation `(n_ticks, n_cv_folds)`): both `train_scores`
  and `test_scores` are `(n_params, n_folds)` — row=param tick, col=fold — matching
  sklearn's post-transpose `(n_ticks, n_cv_folds)` (`:2257-2261`, `:2313-2314`).
  **MATCH** (R-DEV-3). SHIPPED.
- REQ-4 (vary-one-param-over-a-range via a closure): sklearn clones the estimator
  and sets a NAMED param (`parameters={param_name: v}`, `:2299`); ferrolearn takes a
  `make_pipeline: impl Fn(f64) -> Pipeline` closure that rebuilds the pipeline per
  value. **DEVIATE** (R-DEV-7 — Rust has no reflection / `set_params`; the closure
  is the sanctioned analog preserving the observable "vary one param over a range"
  contract). The MECHANIC (one fresh estimator per value, varied over a range) is
  SHIPPED. The TYPE RESTRICTION is a separate gap — see REQ-9.
- REQ-5 (default cv: 5-fold + classifier-aware StratifiedKFold-vs-KFold,
  shuffle=False): sklearn `cv=None` → `check_cv` (`:2286`) gives 5-fold, picking
  StratifiedKFold for a classifier with binary/multiclass `y` else KFold, always
  `shuffle=False` (`:2206-2214`). ferrolearn REQUIRES an explicit `&dyn
  CrossValidator` — no default, no `is_classifier` stratification dispatch.
  **MATCH-intent / gap**: the default-cv behavior is a real semantic gap.
  NOT-STARTED.
- REQ-6 (default scoring: r2/accuracy via check_scoring): sklearn `scoring=None`
  → `check_scoring` (`:2287`) resolves to accuracy (classifier) or r2 (regressor).
  ferrolearn REQUIRES an explicit `scoring: fn(&Array1<f64>, &Array1<f64>) ->
  Result<f64, FerroError>`. **DEVIATE-partial** (R-DEV-7): the explicit-scorer
  closure is the sanctioned analog for a PROVIDED scorer, but the DEFAULT-scorer
  behavior (no `scoring` passed → resolve r2/accuracy from the estimator type) is
  unrepresentable and is a real gap. NOT-STARTED.
- REQ-7 (error_score=np.nan continue-the-curve): on a per-`(fold, param)` fit/score
  failure sklearn assigns `error_score` (default np.nan) to that cell, raises
  `FitFailedWarning`, and CONTINUES the curve (`:2304`, `_fit_and_score` `:729`).
  ferrolearn PROPAGATES the error via `?` (`pipeline.fit(...)?`,
  `scoring(...)?`) and ABORTS the whole call. **MATCH-default / gap**: behavioral
  divergence — sklearn returns a partial nan-bearing curve, ferrolearn returns
  `Err`. NOT-STARTED.
- REQ-8 (groups channel for Group cv): sklearn threads `groups` to `cv.split(X, y,
  groups)` (`:2307`). ferrolearn calls `cv.fold_indices(n_samples)` —
  `CrossValidator::fold_indices(n_samples)` has NO group/label channel, so Group cv
  is unreachable. **MATCH-intent / gap** (same channel-gap as group_splitters).
  NOT-STARTED.
- REQ-9 (arbitrary param TYPE): sklearn `param_range` varies a param of ANY type
  (int `max_depth`, str `kernel`, bool, float `C`) via `set_params`. ferrolearn's
  `param_values: &[f64]` only carries floats. **MATCH-intent / gap**. NOT-STARTED.
- REQ-X-1 (R-SUBSTRATE): production code imports `ndarray::{Array1, Array2}` (the
  array type); the destination substrate is `ferray-core` (R-SUBSTRATE-1).
  NOT-STARTED.
- REQ-X-2 (non-test production consumer): the crate re-export boundary
  (`pub use validation_curve::{ValidationCurveResult, validation_curve} in
  lib.rs`, S5/R-DEFER-1 grandfathering). SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side).

- AC-1 (REQ-1, ordering equivalence — the KEY pin): the `(param OUTER, fold INNER)`
  row-major fill equals sklearn's `(fold OUTER, param INNER)` `.reshape(-1,
  n_params).T`. Oracle (a pure-numpy reproduction of sklearn's reshape, with a
  synthetic per-`(fold, param)` value `f*10 + p` to isolate the index algebra):
  ```
  python3 -c "import numpy as np
  n_folds=3; n_params=4
  flat = np.array([f*10+p for f in range(n_folds) for p in range(n_params)])  # sklearn fold-major order
  M = flat.reshape(-1, n_params).T                                            # sklearn (n_params, n_folds)
  F = np.array([[f*10+p for f in range(n_folds)] for p in range(n_params)])   # ferrolearn param-outer fill
  print('IDENTICAL:', np.array_equal(M, F))"   # -> IDENTICAL: True
  ```
  STRUCTURAL: ferrolearn `result.train_scores[[i, j]]` (param `i`, fold `j`) equals
  sklearn `train_scores[i, j]`.
- AC-2 (REQ-2 / REQ-3, shape + orientation): live oracle
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import validation_curve, KFold
  from sklearn.tree import DecisionTreeRegressor
  X=np.arange(30).reshape(15,2).astype(float); y=np.arange(15).astype(float)
  tr,te=validation_curve(DecisionTreeRegressor(random_state=0), X, y,
      param_name='max_depth', param_range=[1,2,3], cv=KFold(3),
      scoring='neg_mean_squared_error')
  print(tr.shape, te.shape)"   # -> (3, 3) (3, 3)  == (n_ticks=3, n_folds=3)
  ```
  ferrolearn `validation_curve(..., &KFold::new(3), &[..3 vals], ..)` returns
  `train_scores.shape() == [3, 3]` and `test_scores.shape() == [3, 3]`
  (`test_validation_curve_basic`).
- AC-3 (REQ-4, vary-one-param mechanic): a closure that ignores the param produces
  identical rows; a closure whose output depends on the param produces a row whose
  score is best at the matching value. ferrolearn:
  `test_validation_curve_mean_estimator_ignores_param` (all rows equal) and
  `test_validation_curve_best_at_target` (constant estimator best at `param == y`).
- AC-5 (REQ-5, default cv DIVERGENCE): sklearn `cv=None` runs 5-fold KFold for a
  regressor (StratifiedKFold for a binary/multiclass classifier), shuffle=False:
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import check_cv
  from sklearn.tree import DecisionTreeClassifier
  y=np.array([0,1,0,1,0,1])
  print(type(check_cv(None, y, classifier=True)).__name__, 'n_splits',
        check_cv(None, y, classifier=True).get_n_splits())"  # StratifiedKFold n_splits 5
  ```
  ferrolearn has no `validation_curve(..., cv=None, ..)` form — `&dyn
  CrossValidator` is mandatory and no classifier-aware dispatch exists. DIVERGENCE.
  NOT-STARTED (blocker #1757).
- AC-6 (REQ-6, default scoring DIVERGENCE): sklearn `scoring=None` on a regressor
  resolves to r2:
  ```
  python3 -c "from sklearn.tree import DecisionTreeRegressor
  from sklearn.metrics import check_scoring
  print(check_scoring(DecisionTreeRegressor(), scoring=None)._score_func.__name__)"  # r2_score
  ```
  ferrolearn requires an explicit `scoring` fn; the default-resolution behavior is
  absent. DIVERGENCE. NOT-STARTED (blocker #1757).
- AC-7 (REQ-7, error_score=np.nan DIVERGENCE): when the estimator fails to fit for
  ONE param value, sklearn fills that cell with nan and CONTINUES:
  ```
  python3 -c "import numpy as np, warnings
  from sklearn.model_selection import validation_curve, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class F(BaseEstimator, RegressorMixin):
      def __init__(self,d=1): self.d=d
      def fit(self,X,y):
          if self.d==2: raise ValueError('boom')
          self.m_=y.mean(); return self
      def predict(self,X): return np.full(X.shape[0], self.m_)
  X=np.arange(30).reshape(15,2).astype(float); y=np.arange(15).astype(float)
  with warnings.catch_warnings(): warnings.simplefilter('ignore')
  tr,te=validation_curve(F(), X, y, param_name='d', param_range=[1,2,3],
      cv=KFold(3), scoring='neg_mean_squared_error')
  print('row d=2:', te[1], 'any nan:', np.isnan(te).any())"   # row d=2: [nan nan nan] any nan: True
  ```
  ferrolearn `pipeline.fit(...)?` / `scoring(...)?` PROPAGATES the error → the whole
  `validation_curve` returns `Err`, no partial curve. DIVERGENCE. NOT-STARTED
  (blocker #1758).
- AC-8 (REQ-8, groups DIVERGENCE): sklearn threads `groups` to a Group cv:
  `validation_curve(est, X, y, param_name=.., param_range=.., cv=GroupKFold(3),
  groups=g)` produces group-respecting folds. ferrolearn's
  `cv.fold_indices(n_samples)` has no `groups` argument, so no Group cv is
  reachable. DIVERGENCE. NOT-STARTED (blocker #1759).
- AC-9 (REQ-9, param type): sklearn `param_range=["linear","rbf"]` (a STR param)
  or `param_range=[1,2,3]` (an INT `max_depth`) varies non-float params.
  ferrolearn `param_values: &[f64]` cannot carry these. DIVERGENCE. NOT-STARTED
  (blocker #1760).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (core mechanic + iteration-order equivalence — KEY REQ) | SHIPPED | impl `pub fn validation_curve in validation_curve.rs`: outer `for &param in param_values` builds a fresh `make_pipeline(param)`, inner `for (train_idx, test_idx) in &folds` builds train/test subsets, `pipeline.fit(&x_train, &y_train)?` then `scoring(&y_train, &pred)` / `scoring(&y_test, &pred)`, and `train_scores_data.push(...)` / `test_scores_data.push(...)` in `(param OUTER, fold INNER)` order, finalized as `Array2::from_shape_vec((n_params, n_folds), ...)` (row-major). This is element-for-element IDENTICAL to sklearn's `(fold OUTER, param INNER)` flat list `.reshape(-1, n_params).T` (`sklearn/model_selection/_validation.py:2306-2314`): `M[param i][fold j] == flat[j*n_params + i]` and ferrolearn's `data[i*n_folds + j]` are the same `(i, j)` cell. Verification (AC-1): `np.array_equal(M, F)` → `True`. The per-`(fold, param)` fit+score primitive mirrors `_fit_and_score` (`:729`). Tests: `test_validation_curve_basic` (`(3,3)` shapes), `test_validation_curve_best_at_target` (row for `param==y` is best), `test_validation_curve_mean_estimator_ignores_param` (param-independent estimator → equal rows). Non-test consumer: REQ-X-2 (re-export). |
| REQ-2 (return_train_score always on) | SHIPPED | impl `pub fn validation_curve in validation_curve.rs` unconditionally computes `let y_train_pred = fitted.predict(&x_train)?; let train_score = scoring(&y_train, &y_train_pred)?;` and returns `train_scores` in every call, mirroring sklearn's hardcoded `return_train_score=True` (`sklearn/model_selection/_validation.py:2303`). `ValidationCurveResult` always carries `train_scores`. Test: `test_validation_curve_scores_finite` (asserts every train score finite). Non-test consumer: REQ-X-2. |
| REQ-3 (result shape + orientation `(n_ticks, n_cv_folds)`) | SHIPPED | impl `pub fn validation_curve in validation_curve.rs` returns both matrices as `Array2::from_shape_vec((n_params, n_folds), ...)` — `struct ValidationCurveResult` documents `train_scores`/`test_scores` as `(n_params, n_folds)`, row=param tick, col=fold. This is sklearn's post-transpose `(n_ticks, n_cv_folds)` (`sklearn/model_selection/_validation.py:2257-2261`, `:2313-2314`). Live oracle (AC-2): `validation_curve(DecisionTreeRegressor(), X, y, param_range=[1,2,3], cv=KFold(3))` → both `(3, 3)`; ferrolearn `test_validation_curve_basic` asserts `train_scores.shape() == [3, 3]`, `test_scores.shape() == [3, 3]`. Non-test consumer: REQ-X-2. |
| REQ-4 (vary-one-param mechanic via closure) | SHIPPED | impl `pub fn validation_curve in validation_curve.rs` rebuilds `let pipeline = make_pipeline(param);` once per `param` value and runs the full fold loop on it — one fresh estimator per value, varied over the `param_values` range. R-DEV-7 DEVIATE: sklearn `clone(estimator)` + `parameters={param_name: v}` (`set_params`, `sklearn/model_selection/_validation.py:2292`/`:2299`) is unrepresentable in Rust without reflection; the `make_pipeline: impl Fn(f64) -> Pipeline` closure is the sanctioned analog and preserves the observable "vary one param over a range" contract. Tests: `test_validation_curve_mean_estimator_ignores_param` (closure ignores param → identical rows), `test_validation_curve_best_at_target` (closure threads param into a `ConstantEstimator` → row best at `param==y`), `test_validation_curve_single_param`. Non-test consumer: REQ-X-2. The param-TYPE restriction is the separate REQ-9 (NOT-STARTED). |
| REQ-5 (default cv: 5-fold + classifier-aware StratifiedKFold/KFold, shuffle=False) | NOT-STARTED | open prereq blocker #1757. sklearn `cv=None` → `check_cv(cv, y, classifier=is_classifier(estimator))` (`sklearn/model_selection/_validation.py:2286`) yields 5-fold, StratifiedKFold for a classifier with binary/multiclass `y` else KFold, always `shuffle=False` (`:2206-2214`). impl `pub fn validation_curve in validation_curve.rs` takes `cv: &dyn CrossValidator` as a MANDATORY arg, calls `cv.fold_indices(n_samples)?`, and has no `is_classifier`-based StratifiedKFold-vs-KFold dispatch — the no-cv default form is unrepresentable. Oracle (AC-5): `check_cv(None, y, classifier=True)` → `StratifiedKFold` n_splits 5. |
| REQ-6 (default scoring: r2/accuracy via check_scoring) | NOT-STARTED | open prereq blocker #1757. sklearn `scoring=None` → `check_scoring(estimator, scoring=scoring)` (`sklearn/model_selection/_validation.py:2287`) resolves to accuracy (classifier) or r2 (regressor). impl `pub fn validation_curve in validation_curve.rs` requires an explicit `scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>`; the DEFAULT-scorer resolution (no scoring passed → derive r2/accuracy from the estimator type) is absent (R-DEV-7 covers a PROVIDED scorer closure, not the default-resolution behavior). Oracle (AC-6): `check_scoring(DecisionTreeRegressor(), scoring=None)` → `r2_score`. |
| REQ-7 (error_score=np.nan continue-the-curve) | NOT-STARTED | open prereq blocker #1758. sklearn assigns `error_score` (default np.nan) to a failing `(fold, param)` cell, raises `FitFailedWarning`, and CONTINUES the curve (`sklearn/model_selection/_validation.py:2304`, `_fit_and_score` `:729`). impl `pub fn validation_curve in validation_curve.rs` propagates failure via `pipeline.fit(&x_train, &y_train)?` and `scoring(...)?` — the whole call returns `Err`, no partial nan-bearing curve. Behavioral divergence. Oracle (AC-7): an estimator failing for `d==2` yields `te[1] == [nan nan nan]`, `np.isnan(te).any() == True`; ferrolearn aborts. |
| REQ-8 (groups channel for Group cv) | NOT-STARTED | open prereq blocker #1759. sklearn threads `groups` to `cv.split(X, y, groups)` (`sklearn/model_selection/_validation.py:2307`). impl `pub fn validation_curve in validation_curve.rs` calls `cv.fold_indices(n_samples)?`; `CrossValidator::fold_indices(n_samples) in cross_validation.rs` has NO group/label channel, so a Group cv (GroupKFold etc.) is unreachable from `validation_curve` (same channel-gap as `group_splitters`). |
| REQ-9 (arbitrary param TYPE) | NOT-STARTED | open prereq blocker #1760. sklearn `param_range` varies a param of ANY type — int (`max_depth=[1,2,3]`), str (`kernel=["linear","rbf"]`), bool — via `set_params` (`sklearn/model_selection/_validation.py:2299`). impl `pub fn validation_curve in validation_curve.rs` declares `param_values: &[f64]` and `make_pipeline: impl Fn(f64) -> Pipeline`, so only `f64` params can be varied; non-float param ranges are unrepresentable. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker #1761. Production code in `validation_curve.rs` imports `use ndarray::{Array1, Array2}` (array type) and builds `Array2::from_shape_vec(...)` / `Array1` subsets. Per R-SUBSTRATE-1 the destination is `ferray-core`; `ndarray` is the wrong substrate. Until migrated this unit is not on the ferray substrate (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | `pub fn validation_curve` and `ValidationCurveResult` are re-exported at `pub use validation_curve::{ValidationCurveResult, validation_curve} in lib.rs` — the boundary public API per S5/R-DEFER-1 grandfathering. Honest underclaim: the SOLE non-test production reach is this re-export; `grep -rn "validation_curve" ferrolearn-model-sel/src ferrolearn/src ferrolearn-python/src` shows no internal caller (no `learning_curve`/`grid_search`/estimator calls it) and no `ferrolearn-python` binding. The function IS the public boundary surface, consumed by external users. |

## Architecture

ferrolearn implements `validation_curve` as a single free function plus a plain
result struct (`struct ValidationCurveResult in validation_curve.rs`). It takes the
feature matrix `x: &Array2<f64>`, target `y: &Array1<f64>`, a cross-validator
`cv: &dyn CrossValidator`, the float parameter grid `param_values: &[f64]`, a
pipeline factory `make_pipeline: impl Fn(f64) -> Pipeline`, and a scoring function
`scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>`.

The core loop is the faithful translation. sklearn builds a FLAT list over
`for train, test in cv.split(X, y, groups) for v in param_range` (fold OUTER, param
INNER, `:2306-2308`), aggregates it, and reshapes `flat.reshape(-1, n_params).T`
into `(n_params, n_folds)` (`:2313-2314`). ferrolearn instead nests `for &param in
param_values` OUTER and `for (train_idx, test_idx) in &folds` INNER, pushing into a
row-major `(n_params, n_folds)` buffer. These are NOT the same loop nesting, but
they produce the SAME array: cell `(param i, fold j)` is `flat[j*n_params + i]` in
sklearn (the `.reshape(-1, n_params).T` algebra) and `data[i*n_folds + j]` in
ferrolearn — both the `(i, j)` element. AC-1's `np.array_equal(M, F)` proves the
equivalence on a synthetic `f*10 + p` value that isolates the index algebra from
estimator noise. This ordering-equivalence is the KEY REQ (REQ-1) and it agrees.

Per fold, ferrolearn gathers the train/test sample subsets by row-copying
`x.row(i)` into fresh `Array2`s and collecting `y[i]` into `Array1`s, fits the
fold-local pipeline, and scores predictions on BOTH train and test (REQ-2 —
unconditional, matching sklearn's hardcoded `return_train_score=True` at `:2303`).
A fresh `make_pipeline(param)` per param value mirrors sklearn's
`clone(estimator)` + `set_params` (REQ-4, R-DEV-7 — the closure is Rust's analog
for the missing reflection/`set_params`).

Validation: `y.len() != n_samples` → `FerroError::ShapeMismatch`; empty
`param_values` → `FerroError::InvalidParameter`. Both are eager guards before the
fold loop.

Where ferrolearn DIVERGES (all NOT-STARTED, each with a filed blocker):

- **Default cv / classifier-aware stratification** (REQ-5, #1757): `cv` is a
  mandatory `&dyn CrossValidator`; there is no `check_cv` 5-fold default and no
  `is_classifier` → StratifiedKFold-vs-KFold dispatch (`:2286`, `:2206-2214`).
- **Default scoring** (REQ-6, #1757): `scoring` is a mandatory fn; there is no
  `check_scoring` default (r2 for regressor / accuracy for classifier, `:2287`).
- **error_score=np.nan** (REQ-7, #1758): ferrolearn propagates a fit/score failure
  via `?` and aborts, where sklearn fills the cell with nan, warns
  (`FitFailedWarning`), and continues (`:2304`, `_fit_and_score` `:729`).
- **groups** (REQ-8, #1759): `cv.fold_indices(n_samples)` has no group channel, so
  Group cv (which sklearn threads via `cv.split(X, y, groups)`, `:2307`) is
  unreachable — the same `CrossValidator` channel-gap documented for
  `group_splitters`.
- **param type** (REQ-9, #1760): `param_values: &[f64]` only carries floats;
  sklearn's `param_range` varies any param type via `set_params` (`:2299`).
- **substrate** (REQ-X-1, #1761): `ndarray::{Array1, Array2}` must migrate to
  `ferray-core` (R-SUBSTRATE-1/2).

The sole non-test production consumer is the crate re-export (REQ-X-2,
`pub use validation_curve::{ValidationCurveResult, validation_curve} in lib.rs`).
No internal ferrolearn caller (`learning_curve`, `grid_search`, any estimator) and
no `ferrolearn-python` binding consume it; the function IS the public boundary
surface.

## Verification

Commands establishing the SHIPPED claims (baseline
`a09c63dd661f684379b70a0ac8866d1b62e52863`):

- `cargo test -p ferrolearn-model-sel --lib validation_curve` → 7 passed, 0 failed
  (`validation_curve::tests::{test_validation_curve_basic,
  test_validation_curve_best_at_target, test_validation_curve_empty_params_error,
  test_validation_curve_mean_estimator_ignores_param,
  test_validation_curve_scores_finite, test_validation_curve_shape_mismatch,
  test_validation_curve_single_param}`).
- REQ-1 ordering-equivalence oracle (the KEY pin — pure-numpy reproduction of
  sklearn's reshape algebra, synthetic per-`(fold, param)` value isolates the index
  math; R-CHAR-3, oracle-derived):
  ```
  python3 -c "import numpy as np
  n_folds=3; n_params=4
  flat = np.array([f*10+p for f in range(n_folds) for p in range(n_params)])
  M = flat.reshape(-1, n_params).T
  F = np.array([[f*10+p for f in range(n_folds)] for p in range(n_params)])
  print('IDENTICAL:', np.array_equal(M, F))"   # -> IDENTICAL: True
  ```
- REQ-2 / REQ-3 shape+orientation oracle (live sklearn 1.5.2):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import validation_curve, KFold
  from sklearn.tree import DecisionTreeRegressor
  X=np.arange(30).reshape(15,2).astype(float); y=np.arange(15).astype(float)
  tr,te=validation_curve(DecisionTreeRegressor(random_state=0), X, y,
      param_name='max_depth', param_range=[1,2,3], cv=KFold(3),
      scoring='neg_mean_squared_error')
  print(tr.shape, te.shape)"   # -> (3, 3) (3, 3)
  ```
  matches ferrolearn `test_validation_curve_basic` (`[3,3]`/`[3,3]`).
- REQ-5 default-cv DIVERGENCE oracle:
  `check_cv(None, y, classifier=True)` → `StratifiedKFold` n_splits 5;
  `check_cv(None, y, classifier=False)` → `KFold` n_splits 5 — ferrolearn has no
  default-cv form (#1757).
- REQ-6 default-scoring DIVERGENCE oracle:
  `check_scoring(DecisionTreeRegressor(), scoring=None)._score_func.__name__` →
  `r2_score` — ferrolearn requires an explicit `scoring` fn (#1757).
- REQ-7 error_score DIVERGENCE oracle (live sklearn — the nan-continue pin):
  ```
  python3 -c "import numpy as np, warnings
  from sklearn.model_selection import validation_curve, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class F(BaseEstimator, RegressorMixin):
      def __init__(self,d=1): self.d=d
      def fit(self,X,y):
          if self.d==2: raise ValueError('boom')
          self.m_=y.mean(); return self
      def predict(self,X): return np.full(X.shape[0], self.m_)
  X=np.arange(30).reshape(15,2).astype(float); y=np.arange(15).astype(float)
  with warnings.catch_warnings(): warnings.simplefilter('ignore')
  tr,te=validation_curve(F(), X, y, param_name='d', param_range=[1,2,3],
      cv=KFold(3), scoring='neg_mean_squared_error')
  print('row d=2:', te[1], 'any nan:', np.isnan(te).any())"
  # row d=2: [nan nan nan] any nan: True  -> ferrolearn aborts with Err (#1758)
  ```
- REQ-X-1 substrate: `grep -n "ndarray" validation_curve.rs` shows production
  `use ndarray::{Array1, Array2}` and `Array2::from_shape_vec` — wrong substrate,
  migration owed (#1761).
- REQ-X-2 consumer: `grep -rn "validation_curve" ferrolearn-model-sel/src
  ferrolearn/src ferrolearn-python/src` shows only the `lib.rs` re-export
  (`pub use validation_curve::{ValidationCurveResult, validation_curve}`) and the
  module declaration — no internal caller, no Python binding.

SHIPPED: REQ-1 (core mechanic + iteration-order equivalence — KEY REQ, agrees),
REQ-2 (return_train_score), REQ-3 (shape/orientation), REQ-4 (vary-one-param via
closure — R-DEV-7), REQ-X-2 (consumer — re-export boundary only). NOT-STARTED:
REQ-5 (default cv + classifier stratification, #1757), REQ-6 (default scoring,
#1757), REQ-7 (error_score=np.nan continue, #1758), REQ-8 (groups channel,
#1759), REQ-9 (arbitrary param type, #1760), REQ-X-1 (ferray substrate, #1761).
Per R-DEFER-2 every REQ is binary SHIPPED/NOT-STARTED.

Least-confident SHIPPED claim: REQ-4 — the vary-one-param mechanic is SHIPPED, but
only the R-DEV-7 closure analog of `set_params` for FLOAT params; the parameter
TYPE restriction (REQ-9, NOT-STARTED) means the "vary one param over a range"
contract is honored only for the `f64` subset of sklearn's param space, so the
SHIPPED surface is the float-parameter mechanic, not the full `set_params`
generality.
