# grid_search

<!--
tier: 3-component
status: draft
baseline-commit: dd9512946f731430f86fc1e2ffb81f2e110b2b2b
upstream-paths:
  - sklearn/model_selection/_search.py   # GridSearchCV (:1210), _run_search (:1571), BaseSearchCV (:436), fit (:890), _select_best_index (:829), _format_results._store (:1087)
-->

## Summary

`ferrolearn-model-sel/src/grid_search.rs` mirrors scikit-learn's
`GridSearchCV` (`sklearn/model_selection/_search.py:1210`, subclass of
`BaseSearchCV` `:436`) — exhaustive hyperparameter search that, for every point in
a parameter grid, runs cross-validation, records per-fold scores, and reports the
best parameter combination.

ferrolearn exposes `pub struct GridSearchCV<'a>` with a
`pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>`, a pre-expanded
`param_grid: Vec<ParamSet>`, a `cv: Box<dyn CrossValidator>`, and a `scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>`. `fit(&mut self, x, y)`
rejects an empty grid (`InvalidParameter`) then, for each `ParamSet`, builds a
pipeline via the factory and calls `cross_val_score`, pushing the per-fold scores
into a shared `CvResults { params, mean_scores, all_scores }`. Accessors
`cv_results()`, `best_params()`, `best_score()` read back the search. The
`CvResults` type is SHARED — it is also consumed by `RandomizedSearchCV`
(`random_search.rs`), `HalvingGridSearchCV` (`halving_grid_search.rs`), and
`HalvingRandomSearchCV` (`halving_random_search.rs`).

The CORE SEARCH MECHANIC is faithful: iterate every candidate, cross-validate each,
collect per-fold scores; and `mean_test_score` is the UNWEIGHTED fold mean —
element-for-element matching sklearn's `evaluate_candidates(ParameterGrid(...))`
loop and `np.average(array, axis=1, weights=None)`.

Several behaviors DIVERGE and are NOT-STARTED with concrete blockers. One is a
SINGLE-SPOT DETERMINISTIC FIXABLE divergence: the best-index TIE-BREAK — sklearn's
`rank_test_score.argmin()` picks the FIRST (lowest-index) tied candidate, while
ferrolearn's `max_by` returns the LAST (REQ-BESTIDX, #1776). The rest are
MISSING-FEATURE/architectural blockers: no refit / `best_estimator_` /
delegating predict/score (REQ-REFIT, #1777); the sparse `CvResults` lacks
`std_test_score`/`rank_test_score`/per-split-keyed/timing keys (REQ-CVRESULTS,
#1778); no default cv / default scoring (REQ-DEFAULT, #1779); no
`n_jobs`/`pre_dispatch`/`verbose`/`return_train_score`/multimetric (REQ-PARALLEL,
#1780); the `ndarray` substrate (REQ-X-1, #1781).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### `GridSearchCV` (`sklearn/model_selection/_search.py:1210`)
- `:1210` — `class GridSearchCV(BaseSearchCV)`. `__init__(self, estimator,
  param_grid, *, scoring=None, refit=True, cv=None, verbose=0,
  pre_dispatch="2*n_jobs", error_score=np.nan, return_train_score=False)` (plus
  `n_jobs=None`). All params after `param_grid` are keyword-only; `refit` DEFAULTS
  to `True`, `cv`/`scoring` DEFAULT to `None`.
- `:1571-1573` — `_run_search(self, evaluate_candidates)`: the whole GridSearchCV
  body is `evaluate_candidates(ParameterGrid(self.param_grid))` — the candidate set
  is the Cartesian product of `param_grid` in `ParameterGrid` order (sorted-key,
  routed/SHIPPED in `param_grid.rs`).

### `BaseSearchCV.fit` (`sklearn/model_selection/_search.py:890-1075`)
- `:921` — `scorers, refit_metric = self._get_scorers()` — `scoring=None` resolved
  by `check_scoring` (`:857`, regressor → r2, classifier → accuracy).
- `:928` — `cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))` —
  `cv=None` → 5-fold, StratifiedKFold for a classifier with binary/multiclass `y`
  else KFold, `shuffle=False`.
- `:931` — `base_estimator = clone(self.estimator)`.
- `:952-1017` — `evaluate_candidates`: for each `(candidate, split)` pair (`:977`
  `product(enumerate(candidate_params), enumerate(cv.split(...)))` — candidate
  OUTER, split INNER), call `_fit_and_score(clone(base_estimator), X, y, ...)`,
  then `results = self._format_results(all_candidate_params, n_splits, all_out, ...)`.
  `error_score` (default np.nan) flows to `_fit_and_score`; a fit failure fills the
  cell with `error_score` and warns rather than aborting (`_warn_or_raise_about_fit_failures`, `:996`).
- `:1019` — `self._run_search(evaluate_candidates)`.
- `:1034-1044` — `best_index_ = self._select_best_index(self.refit, refit_metric,
  results)`; `best_score_ = results[f"mean_test_{refit_metric}"][best_index_]`;
  `best_params_ = results["params"][best_index_]`.
- `:1046-1061` — if `self.refit` (DEFAULT True): `best_estimator_ =
  clone(base_estimator).set_params(**clone(best_params_, safe=False))`;
  `best_estimator_.fit(X, y)`; `refit_time_ = refit_end - refit_start`.
- `:1072-1073` — `self.cv_results_ = results`; `self.n_splits_ = n_splits`.

### `_select_best_index` (`sklearn/model_selection/_search.py:829-841`)
- `:839-840` — non-callable refit: `best_index = results[f"rank_test_{refit_metric}"].argmin()`.
  `np.argmin` returns the FIRST occurrence of the minimum → on tied
  `rank_test_score` the LOWEST-INDEX candidate wins.

### `_format_results._store` (`sklearn/model_selection/_search.py:1087-1132`)
- `:1091` — `array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)`.
- `:1095` — `results["split%d_%s" % (split_idx, key_name)] = array[:, split_idx]`
  (per-split keyed scores).
- `:1097-1098` — `array_means = np.average(array, axis=1, weights=None)`;
  `results["mean_%s"] = array_means` — UNWEIGHTED mean over folds.
- `:1112-1117` — `array_stds = np.sqrt(np.average((array - array_means[:, None])**2,
  axis=1, weights=None))`; `results["std_%s"]` — population std (ddof=0).
- `:1123-1132` — rank: `np.nan_to_num(array_means, nan=nanmin-1)` then
  `rankdata(-array_means, method="min")` (NaN means treated as tied-worst);
  `results["rank_%s"]`. method="min" → tied means share the SAME (lowest) rank.
- `:1134-1139` — `_store("fit_time", ...)`, `_store("score_time", ...)`,
  `param_<name>` masked arrays, `results["params"] = candidate_params`.

### Search-as-estimator (`BaseSearchCV.predict` `:577`, `score` `:546-552`)
`BaseSearchCV` exposes `predict`/`predict_proba`/`score`/`transform`/
`score_samples` via `@available_if(_estimator_has(...))`, each delegating to
`self.best_estimator_` — available ONLY after `refit` ran.

## Requirements

R-DEV mental test applied per REQ ("numerical/API contract" → MATCH; "Cython/CPython
footgun" → deviate; "missing feature" → NOT-STARTED with a blocker).

- REQ-1 (exhaustive search mechanic): iterate every `ParamSet` in the grid, build a
  fresh pipeline per candidate, run cross-validation, collect per-fold scores.
  Mirrors `evaluate_candidates(ParameterGrid(self.param_grid))` (`:1571-1573`,
  `:952-1017`). **MATCH** (R-DEV-1 structural contract): the observable "every
  candidate cross-validated, scores collected" is the contract. ferrolearn takes a
  PRE-EXPANDED `Vec<ParamSet>` (the grid is expanded by `param_grid.rs`'s
  `ParameterGrid` analog, sorted-key order — routed/SHIPPED) plus a `factory`
  closure that rebuilds the pipeline per candidate — the R-DEV-7 analog of sklearn's
  `clone(base_estimator).set_params(**parameters)` (no reflection in Rust).
  SHIPPED.
- REQ-2 (mean_test_score = unweighted fold mean): ferrolearn `CvResults::push`
  computes `let mean = scores.mean()` (the simple arithmetic mean over folds);
  sklearn `np.average(array, axis=1, weights=None)` (`:1097`). **MATCH** (R-DEV-1
  numerical contract). SHIPPED.
- REQ-BESTIDX (best_index tie-break — FIXABLE DIVERGENCE): on tied mean scores,
  sklearn `best_index_ = rank_test_score.argmin()` with `rankdata(-means,
  method="min")` (`:840`, `:1129`) selects the FIRST (lowest-index) tied candidate;
  `np.argmin` returns the first minimum. ferrolearn `CvResults::best_index` uses
  `self.mean_scores.iter().enumerate().max_by(...)` and Rust's `Iterator::max_by`
  returns the LAST element among equals. **MATCH-class fixable divergence** (R-DEV-1
  tie-breaking is an explicit MATCH item): the contract is "first on a tie"; the
  current code ships "last". A single deterministic single-spot fix (replace
  `max_by` with a first-wins reduction). NOT-STARTED — the critic pins a FAILING
  tie-break test (blocker #1776).
- REQ-REFIT (refit + best_estimator_ + delegating predict/score/predict_proba):
  sklearn `refit=True` DEFAULT (`:1210` init) refits the best params on the whole
  dataset (`:1046-1061`), exposes `best_estimator_`/`refit_time_`, and delegates
  `predict`/`predict_proba`/`score`/`transform` to it (`:577`, `:546-552`).
  ferrolearn `GridSearchCV` has NO refit, NO `best_estimator_`, NO
  `predict`/`score` — it is a search-only struct, not a fitted estimator. **MATCH-
  intent / missing-feature** (architectural). NOT-STARTED (blocker #1777).
- REQ-CVRESULTS (cv_results_ richness): sklearn `cv_results_` carries
  `std_test_score` (`:1117`), `rank_test_score` (`:1129`),
  `split{i}_test_score` per fold (`:1095`),
  `mean_fit_time`/`std_fit_time`/`mean_score_time`/`std_score_time` (`:1134-1135`),
  `param_<name>` masked arrays, and `params` (`:1137-1139`). ferrolearn `CvResults`
  has ONLY `params: Vec<ParamSet>`, `mean_scores: Vec<f64>`, `all_scores:
  Vec<Array1<f64>>` — no std, no rank, no per-split-keyed scores, no timing.
  **MATCH-intent / missing-feature**. NOT-STARTED (blocker #1778).
- REQ-DEFAULT-CV (default cv=None ⇒ 5-fold classifier-aware): sklearn `cv=None` →
  `check_cv(cv, y, classifier=is_classifier(estimator))` (`:928`) gives 5-fold,
  StratifiedKFold for a binary/multiclass-classifier `y` else KFold, shuffle=False.
  ferrolearn requires an explicit `cv: Box<dyn CrossValidator>`. **MATCH-intent /
  gap** (same gap as `validation_curve` REQ-5). NOT-STARTED (blocker #1779).
- REQ-DEFAULT-SCORING (default scoring=None ⇒ estimator default scorer): sklearn
  `scoring=None` → `check_scoring` (`:857`) resolves to r2 (regressor) / accuracy
  (classifier). ferrolearn requires an explicit `scoring: fn(...)`. **MATCH-intent /
  gap** (same gap as `validation_curve` REQ-6). NOT-STARTED (blocker #1779).
- REQ-ERROR-SCORE (error_score=np.nan): sklearn `error_score=np.nan` (`:1210` init)
  fills a failing `(candidate, split)` cell with nan and CONTINUES the search
  (`_warn_or_raise_about_fit_failures`, `:996`). ferrolearn's `GridSearchCV::fit`
  delegates per-candidate evaluation to `cross_val_score`, which propagates a
  fit/score failure via `?` (`pipeline.fit(...)?` / `scoring(...)?` in
  `cross_validation.rs`). The nan-continue divergence is REAL, but it lives in
  `cross_val_score` (a DIFFERENT routed unit — `cross_validation.rs`), not in
  `grid_search.rs`. Per S8, this is a CROSS-UNIT observation owned by
  `cross_validation.rs`, NOT a grid_search blocker. Noted here, classified to the
  owning unit.
- REQ-PARALLEL (n_jobs/pre_dispatch/verbose/return_train_score/multimetric):
  sklearn exposes `n_jobs`, `pre_dispatch="2*n_jobs"`, `verbose`,
  `return_train_score`, and multimetric scoring (`:1210` init,
  `_get_scorers` `:843`). ferrolearn has none. **MATCH-intent / missing-feature**.
  NOT-STARTED (blocker #1780).
- REQ-X-1 (R-SUBSTRATE ndarray→ferray-core): production code imports
  `use ndarray::{Array1, Array2}` (the array type); the destination substrate is
  `ferray-core` (R-SUBSTRATE-1). NOT-STARTED (blocker #1781).
- REQ-X-2 (non-test production consumer): the crate re-export boundary
  (`pub use grid_search::{CvResults, GridSearchCV} in lib.rs`; `CvResults`
  additionally consumed by `random_search.rs`/`halving_grid_search.rs`/
  `halving_random_search.rs`). SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side).

- AC-1 (REQ-1, exhaustive mechanic): a 3-value grid runs cross-validation 3 times
  and records 3 candidate rows. Oracle (live sklearn — every candidate scored):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import GridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  X=np.zeros((30,2)); y=np.full(30,1.0)
  gs=GridSearchCV(Const(), {'c':[0.0,1.0,2.0]}, scoring='neg_mean_squared_error',
      cv=KFold(3), refit=False).fit(X,y)
  print('n candidates:', len(gs.cv_results_['params']))"   # -> n candidates: 3
  ```
  ferrolearn `test_grid_search_runs_all_combinations` asserts
  `results.params.len() == 3`, `mean_scores.len() == 3`, `all_scores.len() == 3`.
- AC-2 (REQ-2, unweighted fold mean): `mean_test_score` equals the simple mean of
  the per-split scores. Oracle (live sklearn):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import GridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Mean(BaseEstimator, RegressorMixin):
      def __init__(self, off=0.0): self.off=off
      def fit(self,X,y): self.m_=y.mean()+self.off; return self
      def predict(self,X): return np.full(X.shape[0], self.m_)
  X=np.zeros((9,1)); y=np.arange(9.0)
  gs=GridSearchCV(Mean(), {'off':[0.0]}, scoring='neg_mean_squared_error',
      cv=KFold(3), refit=False).fit(X,y)
  splits=[gs.cv_results_['split%d_test_score'%i][0] for i in range(3)]
  print('mean==simple mean:', np.isclose(gs.cv_results_['mean_test_score'][0], np.mean(splits)))"
  # -> mean==simple mean: True
  ```
  ferrolearn `CvResults::push` sets `mean = scores.mean()`; `test_cv_results_structure`
  / `test_grid_search_best_score_is_zero_for_perfect` exercise the mean path.
- AC-BESTIDX (REQ-BESTIDX, tie-break DIVERGENCE — the critic's pin): a grid with two
  candidates that TIE for the best mean score. sklearn `best_index_` is the FIRST
  (index 0); ferrolearn `max_by` returns the LAST (index 1). Oracle (live sklearn,
  deterministic tie — two constants equidistant from a constant `y`, each MSE=1):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import GridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  X=np.zeros((30,2)); y=np.full(30,5.0)
  gs=GridSearchCV(Const(), {'c':[4.0,6.0]}, scoring='neg_mean_squared_error',
      cv=KFold(3), refit=True).fit(X,y)
  print('best_index_:', gs.best_index_, 'best c:', gs.best_params_['c'],
        'ranks:', gs.cv_results_['rank_test_score'].tolist())"
  # -> best_index_: 0 best c: 4.0 ranks: [1, 1]
  ```
  sklearn picks index 0 (`c=4.0`); ferrolearn `best_index()`'s `max_by(partial_cmp)`
  on the tied `mean_scores` returns index 1 (`c=6.0`). The critic pins this as a
  FAILING `#[test]` asserting `best_index() == Some(0)` on a constructed tie
  (blocker #1776).
- AC-REFIT (REQ-REFIT, refit/predict DIVERGENCE): sklearn `refit=True` (default)
  exposes `best_estimator_` and a delegating `predict`. Oracle (live sklearn):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import GridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  X=np.zeros((30,2)); y=np.full(30,5.0)
  gs=GridSearchCV(Const(), {'c':[4.0,6.0]}, scoring='neg_mean_squared_error',
      cv=KFold(3), refit=True).fit(X,y)
  print('has best_estimator_:', hasattr(gs,'best_estimator_'),
        'predict:', gs.predict(np.zeros((2,2))).tolist())"
  # -> has best_estimator_: True predict: [4.0, 4.0]
  ```
  ferrolearn `GridSearchCV` has no `best_estimator_`, no `predict`/`score`/
  `predict_proba`. DIVERGENCE. NOT-STARTED (blocker #1777).
- AC-CVRESULTS (REQ-CVRESULTS, missing keys DIVERGENCE): sklearn `cv_results_`
  exposes `std_test_score`/`rank_test_score`/`split{i}_test_score`/timing/
  `param_<name>` keys. Oracle (live sklearn — key set):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import GridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  X=np.zeros((30,2)); y=np.full(30,5.0)
  gs=GridSearchCV(Const(), {'c':[4.0,6.0]}, scoring='neg_mean_squared_error',
      cv=KFold(3), refit=False).fit(X,y)
  print(sorted(gs.cv_results_.keys()))"
  # -> ['mean_fit_time','mean_score_time','mean_test_score','param_c','params',
  #     'rank_test_score','split0_test_score','split1_test_score','split2_test_score',
  #     'std_fit_time','std_score_time','std_test_score']
  ```
  ferrolearn `CvResults` carries only `params`/`mean_scores`/`all_scores` — no
  `std_test_score`, no `rank_test_score`, no per-split-keyed access, no timing.
  DIVERGENCE. NOT-STARTED (blocker #1778).
- AC-DEFAULT (REQ-DEFAULT-CV / REQ-DEFAULT-SCORING DIVERGENCE): sklearn `cv=None`
  yields 5-fold classifier-aware folds and `scoring=None` resolves a default scorer.
  Oracle (live sklearn):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import check_cv
  from sklearn.base import is_regressor
  from sklearn.tree import DecisionTreeRegressor
  print('default cv classifier:', type(check_cv(None, np.array([0,1,0,1]),
        classifier=True)).__name__, check_cv(None, np.array([0,1]),
        classifier=True).get_n_splits())
  print('regressor default score is R^2:', is_regressor(DecisionTreeRegressor()))"
  # -> default cv classifier: StratifiedKFold 5
  # -> regressor default score is R^2: True
  ```
  ferrolearn `GridSearchCV::new` requires explicit `cv` and `scoring` args; there is
  no `cv=None`/`scoring=None` default-resolution form. DIVERGENCE. NOT-STARTED
  (blocker #1779).
- AC-PARALLEL (REQ-PARALLEL): sklearn accepts `n_jobs`, `pre_dispatch`, `verbose`,
  `return_train_score`. ferrolearn `GridSearchCV::new` has no such params (its
  signature is `(pipeline_factory, param_grid, cv, scoring)`). STRUCTURAL — the
  constructor surface lacks these channels. NOT-STARTED (blocker #1780).
- AC-X-1 (REQ-X-1, substrate): `grep -n "ndarray" grid_search.rs` shows production
  `use ndarray::{Array1, Array2}` — wrong substrate, migration to `ferray-core`
  owed (#1781).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (exhaustive search mechanic) | SHIPPED | impl `pub fn fit in grid_search.rs`: after rejecting an empty grid (`FerroError::InvalidParameter`), `for params in &self.param_grid { let pipeline = (self.pipeline_factory)(params); let scores = cross_val_score(&pipeline, x, y, self.cv.as_ref(), self.scoring)?; results.push(params.clone(), scores); }` — every candidate cross-validated, per-fold scores collected. Mirrors sklearn `_run_search` → `evaluate_candidates(ParameterGrid(self.param_grid))` (`sklearn/model_selection/_search.py:1571-1573`, `:952-1017`). The factory closure is the R-DEV-7 analog of `clone(base_estimator).set_params(**parameters)` (`:1051`/`:967`); the grid is pre-expanded by `param_grid.rs` (ParameterGrid sorted-key order). Oracle (AC-1): `len(gs.cv_results_['params']) == 3`; test `test_grid_search_runs_all_combinations` asserts `params.len()==3`/`mean_scores.len()==3`/`all_scores.len()==3`. Non-test consumer: REQ-X-2 (re-export + RandomizedSearchCV/halving share `CvResults`). |
| REQ-2 (mean_test_score = unweighted fold mean) | SHIPPED | impl `pub(crate) fn push in grid_search.rs` (`impl CvResults`): `let mean = scores.mean().unwrap_or(f64::NEG_INFINITY); ... self.mean_scores.push(mean);` — the simple arithmetic mean over folds, matching sklearn `np.average(array, axis=1, weights=None)` (`sklearn/model_selection/_search.py:1097`, UNWEIGHTED). Oracle (AC-2): `np.isclose(gs.cv_results_['mean_test_score'][0], np.mean(splits)) == True`. Tests: `test_grid_search_best_score_is_zero_for_perfect` (mean predictor on constant y → mean MSE 0), `test_cv_results_structure`. Non-test consumer: REQ-X-2. |
| REQ-BESTIDX (best_index tie-break — FIXABLE DIVERGENCE) | NOT-STARTED | open prereq blocker #1776. sklearn `best_index_ = rank_test_score.argmin()` with `rankdata(-means, method="min")` (`sklearn/model_selection/_search.py:840`, `:1129`): `np.argmin` returns the FIRST minimum → on a tie the LOWEST-INDEX candidate wins. impl `pub fn best_index in grid_search.rs` (`impl CvResults`): `self.mean_scores.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|(i, _)| i)` — Rust `Iterator::max_by` returns the LAST element among equals → on a tie the HIGHEST-INDEX candidate wins. R-DEV-1 (tie-breaking is an explicit MATCH item). Oracle (AC-BESTIDX): for grid `{'c':[4.0,6.0]}`, constant `y=5`, each MSE=1 → `best_index_ == 0` (`c=4.0`); ferrolearn returns index 1 (`c=6.0`). SINGLE-SPOT DETERMINISTIC FIX (first-wins reduction); the critic pins a FAILING `#[test]` asserting `best_index() == Some(0)` on a constructed tie. |
| REQ-REFIT (refit + best_estimator_ + delegating predict/score) | NOT-STARTED | open prereq blocker #1777. sklearn `refit=True` DEFAULT (`sklearn/model_selection/_search.py:1210`) refits the best params on the full data (`:1046-1061`), exposing `best_estimator_`/`refit_time_` and delegating `predict`/`predict_proba`/`score`/`transform` to it (`:577`, `:546-552`). impl `pub struct GridSearchCV in grid_search.rs` is search-only — fields `pipeline_factory`/`param_grid`/`cv`/`scoring`/`results`, accessors `cv_results`/`best_params`/`best_score` only; NO refit, NO `best_estimator_`, NO `predict`/`score`. Oracle (AC-REFIT): `hasattr(gs,'best_estimator_') == True`, `gs.predict(...) == [4.0,4.0]`; ferrolearn has neither. Architectural blocker (no single-fixer test). |
| REQ-CVRESULTS (cv_results_ richness) | NOT-STARTED | open prereq blocker #1778. sklearn `cv_results_` has `std_test_score` (`:1117`), `rank_test_score` (`:1129`), `split{i}_test_score` (`:1095`), `mean_fit_time`/`std_fit_time`/`mean_score_time`/`std_score_time` (`:1134-1135`), `param_<name>`/`params` (`:1137-1139`). impl `pub struct CvResults in grid_search.rs` has ONLY `params: Vec<ParamSet>`, `mean_scores: Vec<f64>`, `all_scores: Vec<Array1<f64>>` — no std/rank/per-split-keyed/timing. Oracle (AC-CVRESULTS): sklearn key set includes `std_test_score`/`rank_test_score`/`split0_test_score`/`mean_fit_time`; ferrolearn exposes none. Missing-feature blocker (no single-fixer test). |
| REQ-DEFAULT-CV (default cv=None ⇒ 5-fold classifier-aware) | NOT-STARTED | open prereq blocker #1779. sklearn `cv=None` → `check_cv(self.cv, y, classifier=is_classifier(estimator))` (`sklearn/model_selection/_search.py:928`) gives 5-fold, StratifiedKFold for a binary/multiclass-classifier `y` else KFold, shuffle=False. impl `pub fn new in grid_search.rs` takes `cv: Box<dyn CrossValidator>` as MANDATORY; no `cv=None` default form, no `is_classifier` stratification dispatch. Oracle (AC-DEFAULT): `check_cv(None, [0,1,..], classifier=True)` → `StratifiedKFold` n_splits 5. Same gap as `validation_curve` REQ-5. |
| REQ-DEFAULT-SCORING (default scoring=None ⇒ estimator default scorer) | NOT-STARTED | open prereq blocker #1779. sklearn `scoring=None` → `check_scoring` (`sklearn/model_selection/_search.py:857` via `_get_scorers`) resolves to r2 (regressor) / accuracy (classifier). impl `pub fn new in grid_search.rs` requires an explicit `scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>`; no default-resolution from the estimator type. Oracle (AC-DEFAULT): a regressor's default `.score` is R^2 (`is_regressor(DecisionTreeRegressor()) == True`). Same gap as `validation_curve` REQ-6. |
| REQ-PARALLEL (n_jobs/pre_dispatch/verbose/return_train_score/multimetric) | NOT-STARTED | open prereq blocker #1780. sklearn exposes `n_jobs`, `pre_dispatch="2*n_jobs"`, `verbose`, `return_train_score`, multimetric scoring (`sklearn/model_selection/_search.py:1210` init, `_get_scorers` `:843`). impl `pub fn new in grid_search.rs` signature is `(pipeline_factory, param_grid, cv, scoring)` — none of these channels exist. Missing-feature blocker (no single-fixer test). |
| REQ-ERROR-SCORE (error_score=np.nan continue) | NOT-STARTED | CROSS-UNIT (owned by `cross_validation.rs`, NOT a grid_search blocker — S8). sklearn `error_score=np.nan` fills a failing `(candidate, split)` cell with nan and CONTINUES (`sklearn/model_selection/_search.py:996`, `:943`). impl `pub fn fit in grid_search.rs` delegates each candidate to `cross_val_score(...)?`; `cross_val_score in cross_validation.rs` propagates a failure via `?` (`pipeline.fit(...)?` / `scoring(...)?`) and aborts. The divergence is real but lives in `cross_validation.rs`'s iteration unit — classified to that owning unit per S8/R-DEFER-5, not pinned as a grid_search blocker. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker #1781. Production code in `grid_search.rs` imports `use ndarray::{Array1, Array2}` (array type); `CvResults::all_scores` is `Vec<Array1<f64>>`, `fit` takes `x: &Array2<f64>`/`y: &Array1<f64>`. Per R-SUBSTRATE-1 the destination is `ferray-core`; `ndarray` is the wrong substrate. Not on the ferray substrate (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | `pub struct GridSearchCV` and `pub struct CvResults` are re-exported at `pub use grid_search::{CvResults, GridSearchCV} in lib.rs` — the boundary public API per S5/R-DEFER-1 grandfathering. `CvResults` is FURTHER consumed by non-test production code: `use crate::grid_search::CvResults in random_search.rs` (`RandomizedSearchCV::fit` calls `results.push(...)`/`best_index()`), `in halving_grid_search.rs` (`HalvingGridSearchCV`), and `in halving_random_search.rs` (`HalvingRandomSearchCV`). `GridSearchCV` itself reaches production only via the re-export (no internal caller, no `ferrolearn-python` binding) — it IS the public boundary surface. |

## Architecture

ferrolearn implements `GridSearchCV<'a>` as a search-only struct plus the shared
plain result type `CvResults` (`grid_search.rs`). The struct holds a
`pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>`, a pre-expanded
`param_grid: Vec<ParamSet>`, a `cv: Box<dyn CrossValidator>`, a
`scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>` (higher is
better), and `results: Option<CvResults>` populated by `fit`.

The CORE SEARCH MECHANIC (REQ-1) is the faithful translation. sklearn's GridSearchCV
body is `_run_search(evaluate_candidates)` → `evaluate_candidates(ParameterGrid(
self.param_grid))` (`:1571-1573`), which `product`s over candidates and splits
(`:977`) and calls `_fit_and_score(clone(base_estimator), ...)` per cell.
ferrolearn loops `for params in &self.param_grid`, builds a fresh
`(self.pipeline_factory)(params)`, and calls `cross_val_score(&pipeline, x, y, cv,
scoring)` to get the per-fold scores — the factory closure is the R-DEV-7 analog of
`clone(base_estimator).set_params(**parameters)` (Rust has no `set_params`
reflection), and the grid is pre-expanded by `param_grid.rs`'s `ParameterGrid`
analog (sorted-key Cartesian product, routed/SHIPPED). An empty grid is an eager
`FerroError::InvalidParameter` guard (`:187-192`).

`CvResults::push` records `(params, mean = scores.mean(), all_scores = scores)`.
The UNWEIGHTED fold mean (REQ-2) matches sklearn's `np.average(array, axis=1,
weights=None)` (`:1097`). `best_index()` returns the candidate with the highest mean
score via `max_by(partial_cmp)`.

The SINGLE FIXABLE DIVERGENCE is the TIE-BREAK (REQ-BESTIDX, #1776). sklearn's
`best_index_ = rank_test_score.argmin()` (`:840`) over `rankdata(-means,
method="min")` (`:1129`) selects the FIRST (lowest-index) candidate among ties
(`np.argmin` returns the first minimum). ferrolearn's `max_by` returns the LAST
element among equals (a documented `Iterator::max_by` property). On a constructed
tie (`{'c':[4.0,6.0]}`, constant `y=5`, each MSE=1, AC-BESTIDX), sklearn picks index
0 (`c=4.0`); ferrolearn picks index 1 (`c=6.0`). This is the ONE deterministic
single-spot fixable divergence — the critic pins a FAILING `#[test]` asserting
`best_index() == Some(0)` on the tie, and a fixer replaces `max_by` with a
first-wins reduction (e.g. fold tracking strict-greater-than).

Where ferrolearn DIVERGES as MISSING FEATURES (all NOT-STARTED, each with a filed
blocker — architectural / missing-feature, no single-fixer test):

- **refit / best_estimator_ / search-as-estimator** (REQ-REFIT, #1777): sklearn
  `refit=True` (default) refits the best params on the full data and delegates
  `predict`/`predict_proba`/`score`/`transform` to `best_estimator_` (`:1046-1061`,
  `:577`, `:546-552`). ferrolearn `GridSearchCV` is a search-only struct — no
  refit, no `best_estimator_`, no delegating estimator surface.
- **cv_results_ richness** (REQ-CVRESULTS, #1778): sklearn `cv_results_` carries
  `std_test_score`/`rank_test_score`/`split{i}_test_score`/timing/`param_<name>`
  (`:1095`, `:1117`, `:1129`, `:1134-1139`); `CvResults` has only
  `params`/`mean_scores`/`all_scores`.
- **default cv** (REQ-DEFAULT-CV, #1779): `cv` is a mandatory `Box<dyn
  CrossValidator>`; no `check_cv` 5-fold default and no `is_classifier` →
  StratifiedKFold/KFold dispatch (`:928`).
- **default scoring** (REQ-DEFAULT-SCORING, #1779): `scoring` is a mandatory fn; no
  `check_scoring` r2/accuracy default (`:857`).
- **n_jobs/pre_dispatch/verbose/return_train_score/multimetric** (REQ-PARALLEL,
  #1780): absent from the constructor.
- **substrate** (REQ-X-1, #1781): `ndarray::{Array1, Array2}` must migrate to
  `ferray-core` (R-SUBSTRATE-1/2).

A CROSS-UNIT note (REQ-ERROR-SCORE): sklearn's `error_score=np.nan` fills a failing
cell with nan and continues (`:996`); ferrolearn's `cross_val_score` propagates a
fit/score failure via `?` and aborts the whole search. The divergence is real but
OWNED by `cross_validation.rs` (the iteration unit that calls `pipeline.fit(...)?`),
not `grid_search.rs` — per S8 it is classified to that unit, not pinned here.

`CvResults` is a SHARED type: re-exported at `pub use grid_search::{CvResults,
GridSearchCV} in lib.rs`, and consumed in non-test production by
`RandomizedSearchCV` (`random_search.rs`), `HalvingGridSearchCV`
(`halving_grid_search.rs`), and `HalvingRandomSearchCV`
(`halving_random_search.rs`) — all of which call `CvResults::push`/`best_index`.
`GridSearchCV` itself reaches production only via the re-export boundary (REQ-X-2).

## Verification

Commands establishing the SHIPPED claims (baseline
`dd9512946f731430f86fc1e2ffb81f2e110b2b2b`):

- `cargo test -p ferrolearn-model-sel --lib grid_search` → 6 passed, 0 failed
  (`grid_search::tests::{test_grid_search_runs_all_combinations,
  test_grid_search_finds_best_params, test_grid_search_best_score_is_zero_for_perfect,
  test_grid_search_empty_grid_returns_error, test_grid_search_returns_none_before_fit,
  test_cv_results_structure}`).
- REQ-1 exhaustive-mechanic oracle (live sklearn 1.5.2 — every candidate scored):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import GridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  X=np.zeros((30,2)); y=np.full(30,1.0)
  gs=GridSearchCV(Const(), {'c':[0.0,1.0,2.0]}, scoring='neg_mean_squared_error',
      cv=KFold(3), refit=False).fit(X,y)
  print('n candidates:', len(gs.cv_results_['params']))"   # -> 3
  ```
- REQ-2 unweighted-mean oracle (live sklearn 1.5.2):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import GridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Mean(BaseEstimator, RegressorMixin):
      def __init__(self, off=0.0): self.off=off
      def fit(self,X,y): self.m_=y.mean()+self.off; return self
      def predict(self,X): return np.full(X.shape[0], self.m_)
  X=np.zeros((9,1)); y=np.arange(9.0)
  gs=GridSearchCV(Mean(), {'off':[0.0]}, scoring='neg_mean_squared_error',
      cv=KFold(3), refit=False).fit(X,y)
  splits=[gs.cv_results_['split%d_test_score'%i][0] for i in range(3)]
  print('mean==simple mean:', np.isclose(gs.cv_results_['mean_test_score'][0], np.mean(splits)))"
  # -> mean==simple mean: True
  ```
- REQ-BESTIDX tie-break DIVERGENCE oracle (the critic's pin — live sklearn 1.5.2):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import GridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  X=np.zeros((30,2)); y=np.full(30,5.0)
  gs=GridSearchCV(Const(), {'c':[4.0,6.0]}, scoring='neg_mean_squared_error',
      cv=KFold(3), refit=True).fit(X,y)
  print('best_index_:', gs.best_index_, 'best c:', gs.best_params_['c'])"
  # -> best_index_: 0 best c: 4.0   (ferrolearn max_by -> index 1, c=6.0)  (#1776)
  ```
- REQ-REFIT DIVERGENCE oracle: `hasattr(gs,'best_estimator_') == True`,
  `gs.predict(np.zeros((2,2))) == [4.0, 4.0]` (refit=True default) — ferrolearn has
  no `best_estimator_`/`predict` (#1777).
- REQ-CVRESULTS DIVERGENCE oracle: `sorted(gs.cv_results_.keys())` includes
  `std_test_score`/`rank_test_score`/`split0_test_score`/`mean_fit_time` —
  ferrolearn `CvResults` has only `params`/`mean_scores`/`all_scores` (#1778).
- REQ-DEFAULT DIVERGENCE oracle: `check_cv(None, [0,1,0,1], classifier=True)` →
  `StratifiedKFold` n_splits 5; `is_regressor(DecisionTreeRegressor()) == True`
  (default score R^2) — ferrolearn requires explicit `cv` + `scoring` (#1779).
- REQ-X-1 substrate: `grep -n "ndarray" grid_search.rs` shows production
  `use ndarray::{Array1, Array2}` — wrong substrate, migration owed (#1781).
- REQ-X-2 consumer: `grep -rn "CvResults\|GridSearchCV" ferrolearn-model-sel/src`
  shows the `lib.rs` re-export plus `use crate::grid_search::CvResults` in
  `random_search.rs`/`halving_grid_search.rs`/`halving_random_search.rs` (each
  calling `CvResults::push`/`best_index`).

SHIPPED: REQ-1 (exhaustive search mechanic), REQ-2 (unweighted fold mean),
REQ-X-2 (consumer — re-export + RandomizedSearchCV/halving share `CvResults`).
NOT-STARTED: REQ-BESTIDX (tie-break — the SINGLE FIXABLE DIVERGENCE, #1776),
REQ-REFIT (refit/best_estimator_/predict, #1777), REQ-CVRESULTS (#1778),
REQ-DEFAULT-CV (#1779), REQ-DEFAULT-SCORING (#1779), REQ-PARALLEL (#1780),
REQ-ERROR-SCORE (cross-unit, owned by `cross_validation.rs`, S8 — not pinned here),
REQ-X-1 (ferray substrate, #1781). Per R-DEFER-2 every REQ is binary
SHIPPED/NOT-STARTED.

Least-confident SHIPPED claim: REQ-2 — the unweighted fold mean matches sklearn
EXACTLY when every candidate is evaluated on a constant fold count, but ferrolearn's
`scores.mean().unwrap_or(f64::NEG_INFINITY)` substitutes `NEG_INFINITY` for an empty
fold set where sklearn `np.average` of an empty array raises; on the non-degenerate
path the means agree (AC-2 oracle: `np.isclose(...) == True`), so the SHIPPED surface
is the standard non-empty-fold mean, not the empty-fold edge.
