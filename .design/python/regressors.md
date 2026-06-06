# ferrolearn-python regressors — sklearn LinearRegression + Ridge + Lasso + ElasticNet binding shim

<!--
tier: 3-component
status: draft
baseline-commit: 7ba78cf27
upstream-paths:
  - sklearn/linear_model/_base.py               # class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel) — __init__/_parameter_constraints/fit; attrs coef_/intercept_/rank_/singular_/n_features_in_
  - sklearn/linear_model/_ridge.py              # class Ridge(MultiOutputMixin, RegressorMixin, _BaseRidge) — __init__/fit; attrs coef_/intercept_/n_iter_
  - sklearn/linear_model/_coordinate_descent.py # class ElasticNet / class Lasso(ElasticNet) — __init__/_parameter_constraints/fit; attrs coef_/intercept_/n_iter_/dual_gap_/sparse_coef_
-->

## Summary

`ferrolearn-python/src/regressors.rs` is the PyO3 marshalling shim binding FOUR
regression estimators to CPython:
`#[pyclass(name = "_RsLinearRegression")] RsLinearRegression`
(over `ferrolearn_linear::FittedLinearRegression<f64>`),
`#[pyclass(name = "_RsRidge")] RsRidge` (over `ferrolearn_linear::FittedRidge<f64>`),
`#[pyclass(name = "_RsLasso")] RsLasso` (over `ferrolearn_linear::FittedLasso<f64>`),
and `#[pyclass(name = "_RsElasticNet")] RsElasticNet`
(over `ferrolearn_linear::FittedElasticNet<f64>`).
`ferrolearn-python/python/ferrolearn/_regressors.py` wraps each as a sklearn
`RegressorMixin`/`BaseEstimator` subclass — `LinearRegression`, `Ridge`, `Lasso`,
`ElasticNet` — so `import ferrolearn` mirrors
`from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet`.
They mirror **`sklearn.linear_model.LinearRegression`** (`_base.py:465`),
**`sklearn.linear_model.Ridge`** (`_ridge.py:1016`),
**`sklearn.linear_model.Lasso`** (`_coordinate_descent.py:1154`), and
**`sklearn.linear_model.ElasticNet`** (`_coordinate_descent.py:729`).

The regressor *correctness* (OLS single-SVD min-norm solve; Ridge closed-form
Cholesky with unpenalized intercept; Lasso/ElasticNet cyclic coordinate descent
with soft-thresholding) lives DOWN in
`ferrolearn-linear/src/{linear_regression,ridge,lasso,elastic_net}.rs`, each
audited by that crate's `//!` REQ status table. THIS unit is the **sklearn-API
marshalling shim** only: constructor parameter ABI, attribute exposure, method
surface, and array coercion across the Python↔Rust boundary (R-DEV-2 ABI,
R-DEV-3 output contract, the `ferray::numpy_interop` boundary covered by
`conversions.md` #2027). Semantic/numerical and missing-knob divergences are
owned by the downstream `ferrolearn-linear` crate; this doc references its
existing blockers rather than re-filing them, and owns only the binding-level
surface.

**Verification model: B (pytest vs sklearn 1.5.2).** Per goal.md §"The
verification model (B)", this unit is verified by
`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` comparing
`import ferrolearn` against the installed `import sklearn` 1.5.2 oracle, plus the
live-sklearn oracle for the constructor-ABI and attribute boundary. As of
baseline `7ba78cf27` the gauntlet is GREEN: **534 passed**. All four estimators —
`LinearRegression()`, `Ridge()`, `Lasso()`, `ElasticNet()` — are exercised by
`tests/test_check_estimator.py` (`parametrize_with_checks`) and
`tests/test_cross_val_score.py` (`cross_val_score`).

Divergence classes:
1. **api-conformance (SHIPPED core)** — on the DEFAULT parameter path, all four
   expose `fit`/`predict` with the right shapes/types plus `coef_`/`intercept_`
   (+ `n_features_in_` from `_validate_data`), and the marshalled VALUES match
   the live sklearn oracle element-wise (LinearRegression/Ridge to 1e-8;
   Lasso/ElasticNet to the downstream-verified converged tolerance ≤1e-6/<1e-5).
   `check_estimator` + `cross_val_score` pass for all four.
2. **alpha-keyword-only (NOT-STARTED, R-DEV-2, single-wrapper-fixable — THE
   HEADLINE, three distinct divergences)** — `_regressors.py::{Ridge,Lasso,
   ElasticNet}.__init__` put `alpha` AFTER the leading `*`, making it
   keyword-only, so `ferrolearn.Ridge(0.5)` / `Lasso(0.5)` / `ElasticNet(0.5)`
   each raise `TypeError`, whereas sklearn's `alpha` is positional-or-keyword
   (`Ridge(0.5).alpha == 0.5`). Three independent single-line Python-wrapper
   fixes (move `alpha` before the `*`).
3. **n_iter_ faked (NOT-STARTED, R-DEV-1/-3)** — `_regressors.py::Lasso.fit` /
   `ElasticNet.fit` set `self.n_iter_ = self.max_iter` (a hardcoded FAKE =
   `1000`), never the actual coordinate-descent iteration count sklearn exposes
   (live: `Lasso(alpha=0.1).n_iter_ == 89`, `ElasticNet(alpha=0.1).n_iter_ ==
   58`). The Rust `_RsLasso`/`_RsElasticNet` expose NO `n_iter_` getter and
   `FittedLasso`/`FittedElasticNet` do not track the real count — owned
   downstream (Lasso #411, ElasticNet #417).
4. **missing constructor params (NOT-STARTED, R-DEV-2)** — LinearRegression
   lacks `copy_X`/`n_jobs`/`positive`; Ridge lacks `copy_X`/`max_iter`/`tol`/
   `solver`/`positive`/`random_state`; Lasso/ElasticNet lack `precompute`/
   `copy_X`/`warm_start`/`positive`/`random_state`/`selection`. The DEFAULTS all
   match; only the param surface + non-default behavior is missing, owned
   downstream.
5. **value parity off the default path (NOT-STARTED, R-DEV-1)** — `positive=True`
   (NNLS), multi-output 2-D `y`, and the Lasso/ElasticNet dual-gap stopping
   criterion are owned downstream.
6. **substrate (NOT-STARTED, R-SUBSTRATE-1)** — the binding round-trips numpy ↔
   `ndarray` via `crate::conversions::*` (rust-numpy), not
   `ferray::numpy_interop`/`ferray-core`; owned by `conversions.md` #2027.

## Upstream reference (sklearn 1.5.2, live oracle = installed sklearn 1.5.2)

Lines stable at tag 1.5.2 / commit 156ef14.

### `sklearn.linear_model.LinearRegression` (`_base.py:465`)

- **`__init__`** (`_base.py:568-579`):
  `LinearRegression(self, *, fit_intercept=True, copy_X=True, n_jobs=None,
  positive=False)`. ALL params are keyword-only (the `*` is FIRST).
- **`fit`** (`_base.py:582`): OLS via `linalg.lstsq(X, y)` (`_base.py:687`, single
  SVD), `positive=True` routes to `optimize.nnls` (`_base.py:645-647`); sets
  `coef_`, `intercept_` (`_base.py:324`/`:327`), `rank_`/`singular_`
  (`_base.py:687`), `n_features_in_`.
- **attributes**: `coef_` `(n_features,)` or `(n_targets, n_features)`,
  `intercept_` float or `(n_targets,)`, `rank_` int, `singular_`
  `(min(n,p),)`, `n_features_in_` int.

### `sklearn.linear_model.Ridge` (`_ridge.py:1016`)

- **`__init__`** (`_ridge.py:893-912`):
  `Ridge(self, alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None,
  tol=1e-4, solver='auto', positive=False, random_state=None)`. `alpha` is
  positional-or-keyword (it PRECEDES the `*`); everything after `*` is
  keyword-only.
- **`fit`** (`_BaseRidge.fit`, `_ridge.py:914`): default `solver='auto'` →
  `cholesky` dense path; sets `coef_`/`intercept_`/`n_iter_`/`solver_`
  (`_ridge.py:968`/`:994`).
- **attributes**: `coef_` `(n_features,)` or `(n_targets, n_features)`,
  `intercept_` float or `(n_targets,)`, `n_iter_` (None for cholesky),
  `n_features_in_` int.

### `sklearn.linear_model.ElasticNet` (`_coordinate_descent.py:729`) and `Lasso(ElasticNet)` (`:1154`)

- **`ElasticNet.__init__`** (`_coordinate_descent.py:898-912`):
  `ElasticNet(self, alpha=1.0, *, l1_ratio=0.5, fit_intercept=True,
  precompute=False, max_iter=1000, copy_X=True, tol=1e-4, warm_start=False,
  positive=False, random_state=None, selection='cyclic')`. `alpha` is
  positional-or-keyword; `l1_ratio` and everything else are keyword-only.
- **`Lasso.__init__`** (`_coordinate_descent.py:1310-1322`):
  `Lasso(self, alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True,
  max_iter=1000, tol=1e-4, warm_start=False, positive=False, random_state=None,
  selection='cyclic')` (delegates to `ElasticNet.__init__` with `l1_ratio=1.0`).
  `alpha` is positional-or-keyword.
- **`fit`** (`_coordinate_descent.py:932`): cyclic coordinate descent
  (`enet_coordinate_descent`); sets `coef_`/`intercept_`/`n_iter_`
  (`_coordinate_descent.py:1103`/`:1106` — the ACTUAL iteration count) /
  `dual_gap_` (`:1108`/`:1111`).
- **attributes**: `coef_` `(n_features,)`, `intercept_` float, `n_iter_` int (or
  list) — the real CD iteration count, `dual_gap_` float, `sparse_coef_`,
  `n_features_in_` int.

Live oracle (installed sklearn 1.5.2, run from `/tmp`; R-CHAR-3 — expected
values from sklearn, NEVER from ferrolearn). `X = [[0,0],[1,1],[2,4],[3,9]]`,
`y = [1,2,3,4]`:

```
LinearRegression.__init__ -> (self, *, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
Ridge.__init__            -> (self, alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None)
Lasso.__init__            -> (self, alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
ElasticNet.__init__       -> (self, alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

Ridge(0.5).alpha -> 0.5 ; Lasso(0.1).alpha -> 0.1 ; ElasticNet(0.1).alpha -> 0.1   # alpha positional

LinearRegression().fit(X,y):  coef_=[1.0, 0.0]            intercept_=1.0
Ridge(alpha=1.0).fit(X,y):    coef_=[0.33333333, 0.2]     intercept_=1.3
Lasso(alpha=0.1).fit(X,y):    coef_=[0.32033389, 0.19989779]  intercept_=1.31985691  n_iter_=89
ElasticNet(alpha=0.1).fit(X,y): coef_=[0.45945063, 0.1607366]  intercept_=1.24824597  n_iter_=58
```

ferrolearn at baseline `7ba78cf27` (live):
- `LinearRegression().fit(X,y)` → `coef_=[1.0, 0.0]`, `intercept_=1.0` — MATCHES
  the oracle element-wise.
- `Ridge(alpha=1.0).fit(X,y)` → `coef_=[0.33333333, 0.2]`, `intercept_=1.3` —
  MATCHES element-wise.
- `Lasso(alpha=0.1).fit(X,y)` → `coef_=[0.32109994, 0.19966328]` (matches the
  oracle to ≤1e-3 at the differing stopping point; downstream `ferrolearn-linear`
  REQ-1 verifies converged coef ≤1e-6), but `n_iter_=1000` (FAKE = max_iter) vs
  oracle `89`.
- `ElasticNet(alpha=0.1).fit(X,y)` → `coef_=[0.45980675, 0.16062802]` (downstream
  REQ-1 verifies converged coef <1e-5), but `n_iter_=1000` (FAKE) vs oracle `58`.
- `LinearRegression.__init__` sig is `(self, *, fit_intercept=True)` —
  `fit_intercept` keyword-only, MATCHING sklearn (which also makes
  `fit_intercept` keyword-only via a leading `*`).
- `ferrolearn.Ridge(0.5)` / `Lasso(0.5)` / `ElasticNet(0.5)` each raise
  `TypeError: __init__() takes 1 positional argument but 2 were given` (alpha
  keyword-only) — the three headline divergences.

## Requirements

Grouped by estimator (`REQ-LINREG-*`, `REQ-RIDGE-*`, `REQ-LASSO-*`,
`REQ-ELASTICNET-*`) plus shared `REQ-CONSUMER`/`REQ-SUBSTRATE`.

### LinearRegression

- REQ-LINREG-API-CONFORM: `ferrolearn.LinearRegression` exposes the
  `sklearn.linear_model.LinearRegression` method surface — `fit`/`predict` (bound
  on `_RsLinearRegression` in `regressors.rs`, wrapped in `_regressors.py`) plus
  `score` (inherited from `RegressorMixin`) — and the fitted attributes `coef_`
  `(n_features,)`, `intercept_` (float), `n_features_in_` (int, set by
  `_validate_data`), with values matching the sklearn oracle element-wise on the
  DEFAULT `fit_intercept=True` path (`_base.py:582` fit, `:324`/`:327`
  intercept). `check_estimator(LinearRegression())` + `cross_val_score` pass.
- REQ-LINREG-FIT-INTERCEPT-ABI: `ferrolearn.LinearRegression`'s `fit_intercept`
  is keyword-only, matching sklearn `__init__(self, *, fit_intercept=True, ...)`
  (`_base.py:568-570`, the `*` is FIRST). Live: both signatures make
  `fit_intercept` keyword-only — this ABI MATCHES.
- REQ-LINREG-RANK-SINGULAR: `ferrolearn.LinearRegression` exposes the `rank_`
  (int) and `singular_` (1-D array) fitted attributes, matching sklearn
  `self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)`
  (`_base.py:687`; attr docstrings `rank_` `_base.py:505`, `singular_`
  `_base.py:508`). Surfaced by the `RsLinearRegression` `rank_`/`singular_`
  getters over `FittedLinearRegression::rank()`/`singular_values()`, set in
  `_regressors.py::LinearRegression.fit`. NOTE the attribute is `singular_`, not
  `singular_values_`. [Downstream `ferrolearn-linear` REQ-9 SHIPPED #374 captures
  both from the single-SVD solve on the centered-when-`fit_intercept` matrix.]
- REQ-LINREG-PARAMS: `ferrolearn.LinearRegression` exposes the
  `copy_X`/`n_jobs`/`positive` constructor params (`_base.py:568-579`). [Param
  surface + behavior owned downstream: `positive` `ferrolearn-linear` REQ-6
  blocker #371; `copy_X`/`n_jobs` REQ-9 blocker #374 (the `rank_`/`singular_`
  half of #374 is SHIPPED — see REQ-LINREG-RANK-SINGULAR).]
- REQ-LINREG-VALUE-PARITY: `coef_`/`intercept_` match sklearn array-by-array on a
  fixed dataset (R-DEV-1) on the DEFAULT path. [Default path is SHIPPED
  (downstream REQ-1/REQ-5 verify full-rank, rank-deficient, and underdetermined
  OLS to 1e-8); `positive=True` (NNLS) parity is owned downstream #371,
  multi-output 2-D `y` → 2-D `coef_` #372.]

### Ridge

- REQ-RIDGE-API-CONFORM: `ferrolearn.Ridge` exposes the
  `sklearn.linear_model.Ridge` method surface — `fit`/`predict` (bound on
  `_RsRidge`, wrapped in `_regressors.py`) plus `score` (`RegressorMixin`) — and
  `coef_` `(n_features,)`, `intercept_` (float), `n_features_in_`, with values
  matching the sklearn oracle element-wise on the DEFAULT
  `alpha=1.0`/`solver='auto'`→cholesky path (`_ridge.py:914` fit). `alpha` is
  marshalled to `RsRidge` via `with_alpha`. `check_estimator(Ridge())` +
  `cross_val_score` pass.
- REQ-RIDGE-ALPHA-POSITIONAL: `ferrolearn.Ridge` accepts `alpha` positionally —
  `Ridge(0.5)` constructs an estimator with `alpha == 0.5`, matching sklearn
  `__init__(self, alpha=1.0, *, ...)` (`_ridge.py:893-895`, `alpha` before the
  `*`).
- REQ-RIDGE-PARAMS: `ferrolearn.Ridge` exposes the
  `copy_X`/`max_iter`/`tol`/`solver`/`positive`/`random_state` constructor params
  (`_ridge.py:893-912`). [Param surface + behavior owned downstream: `solver`
  `ferrolearn-linear` REQ-8 #386; `positive` REQ-9 #387; `max_iter`/`tol`/`n_iter_`
  REQ-10 #388; `copy_X`/`random_state` REQ-12 #390.]
- REQ-RIDGE-VALUE-PARITY: `coef_`/`intercept_` match sklearn array-by-array
  (R-DEV-1) on the DEFAULT cholesky path. [Default path is SHIPPED (downstream
  REQ-1/REQ-5 verify cholesky coef/intercept to 1e-8 across
  alpha∈{0.1,1,10,100}, intercept unpenalized, alpha=0 min-norm); per-target
  `alpha` array #385, and alternate-solver values are owned downstream.
  Multi-output 2-D `y` → SHIPPED, see REQ-RIDGE-MULTIOUTPUT.]
- REQ-RIDGE-MULTIOUTPUT: `ferrolearn.Ridge.fit` accepts a 2-D `y` of shape
  `(n_samples, n_targets)` and exposes `coef_` `(n_targets, n_features)` +
  `intercept_` `(n_targets,)`, matching sklearn `Ridge` (`_ridge.py:543` coef
  shape, `:550` intercept shape; per-target solve `:207`/`:218`). Surfaced by the
  `_RsRidgeMultiOutput` binding over `ferrolearn_linear::FittedRidgeMulti`
  (`Fit<Array2, Array2> for Ridge`, shipped #29); the wrapper routes the
  `y.ndim==2 and y.shape[1]>1` path. The 1-D `y` path (n_targets==1 collapses to
  ravel, `_ridge.py:670-672`) is unchanged (scalar intercept). [Downstream
  `ferrolearn-linear` REQ-6 #384.]

### Lasso

- REQ-LASSO-API-CONFORM: `ferrolearn.Lasso` exposes the
  `sklearn.linear_model.Lasso` method surface — `fit`/`predict` (bound on
  `_RsLasso`, wrapped in `_regressors.py`) plus `score` (`RegressorMixin`) — and
  `coef_` `(n_features,)`, `intercept_` (float), `n_features_in_`, with `coef_`/
  `intercept_` and the exactly-zero support set matching the sklearn oracle on the
  DEFAULT `alpha=1.0`/`max_iter=1000`/`tol=1e-4`/`selection='cyclic'` path
  (`_coordinate_descent.py:932` fit). `alpha`/`max_iter`/`tol` marshalled via
  `with_alpha`/`with_max_iter`/`with_tol`. `check_estimator(Lasso())` +
  `cross_val_score` pass.
- REQ-LASSO-ALPHA-POSITIONAL: `ferrolearn.Lasso` accepts `alpha` positionally —
  `Lasso(0.1)` constructs an estimator with `alpha == 0.1`, matching sklearn
  `__init__(self, alpha=1.0, *, ...)` (`_coordinate_descent.py:1310-1312`,
  `alpha` before the `*`).
- REQ-LASSO-NITER: `ferrolearn.Lasso`'s `n_iter_` is the ACTUAL
  coordinate-descent iteration count, matching sklearn
  (`_coordinate_descent.py:1103`/`:1106`; oracle `Lasso(alpha=0.1).fit(X,y).n_iter_
  == 89`). [Owned downstream: `ferrolearn-linear` REQ-11 (`n_iter_`/`dual_gap_`
  attrs) blocker #411 — `FittedLasso` does not track the count, so the wrapper
  fakes `n_iter_ = max_iter`.]
- REQ-LASSO-PARAMS: `ferrolearn.Lasso` exposes the
  `precompute`/`copy_X`/`warm_start`/`positive`/`random_state`/`selection`
  constructor params (`_coordinate_descent.py:1310-1322`). [Param surface +
  behavior owned downstream: `positive` `ferrolearn-linear` REQ-7 #407;
  `warm_start` REQ-8 #408; `selection`/`random_state` REQ-9 #409; `precompute`
  REQ-10 #410.]
- REQ-LASSO-VALUE-PARITY: `coef_`/`intercept_` + the exact-zero support set match
  sklearn array-by-array (R-DEV-1) on the DEFAULT cyclic path. [Default-path
  converged parity is SHIPPED (downstream REQ-1/4/6 verify converged coef/
  intercept + support set ≤1e-6 incl. alpha=0); the dual-gap stopping criterion
  (REQ-12 #412) and `positive`/`selection='random'` paths are owned downstream.]

### ElasticNet

- REQ-ELASTICNET-API-CONFORM: `ferrolearn.ElasticNet` exposes the
  `sklearn.linear_model.ElasticNet` method surface — `fit`/`predict` (bound on
  `_RsElasticNet`, wrapped in `_regressors.py`) plus `score` (`RegressorMixin`) —
  and `coef_` `(n_features,)`, `intercept_` (float), `n_features_in_`, with
  `coef_`/`intercept_` matching the sklearn oracle on the DEFAULT
  `alpha=1.0`/`l1_ratio=0.5`/`max_iter=1000`/`tol=1e-4`/`selection='cyclic'` path
  (`_coordinate_descent.py:932` fit). `alpha`/`l1_ratio`/`max_iter`/`tol`
  marshalled via `with_alpha`/`with_l1_ratio`/`with_max_iter`/`with_tol`.
  `check_estimator(ElasticNet())` + `cross_val_score` pass.
- REQ-ELASTICNET-ALPHA-POSITIONAL: `ferrolearn.ElasticNet` accepts `alpha`
  positionally — `ElasticNet(0.1)` constructs an estimator with `alpha == 0.1`,
  matching sklearn `__init__(self, alpha=1.0, *, l1_ratio=0.5, ...)`
  (`_coordinate_descent.py:898-902`, `alpha` before the `*`; `l1_ratio` is
  correctly keyword-only on both sides).
- REQ-ELASTICNET-NITER: `ferrolearn.ElasticNet`'s `n_iter_` is the ACTUAL
  coordinate-descent iteration count, matching sklearn
  (`_coordinate_descent.py:1103`/`:1106`; oracle
  `ElasticNet(alpha=0.1).fit(X,y).n_iter_ == 58`). [Owned downstream:
  `ferrolearn-linear` ElasticNet `n_iter_`/`dual_gap_` blocker #417 —
  `FittedElasticNet` does not track the count, so the wrapper fakes
  `n_iter_ = max_iter`.]
- REQ-ELASTICNET-PARAMS: `ferrolearn.ElasticNet` exposes the
  `precompute`/`copy_X`/`warm_start`/`positive`/`random_state`/`selection`
  constructor params (`_coordinate_descent.py:898-912`). [Param surface +
  behavior owned downstream: `positive` `ferrolearn-linear` REQ-8 #407;
  `warm_start` #408; `selection`/`random_state` #409; `precompute` #410.]
- REQ-ELASTICNET-VALUE-PARITY: `coef_`/`intercept_` + the exact-zero support set
  match sklearn array-by-array (R-DEV-1) on the DEFAULT cyclic path.
  [Default-path converged parity is SHIPPED (downstream REQ-1/4/5 verify converged
  coef/intercept + support set <1e-5 across the (alpha, l1_ratio) grid, incl.
  l1_ratio=1↔Lasso and l1_ratio=0↔L2); the dual-gap stopping criterion (#412) and
  `positive`/`selection='random'` paths are owned downstream.]

### Shared

- REQ-CONSUMER: the binding IS the public API (R-DEFER-1/S5: boundary estimator
  types ARE the public surface, grandfathered existing pub API); its non-test
  production consumers are the Python wrappers `_regressors.py::{LinearRegression,
  Ridge, Lasso, ElasticNet}` + the `ferrolearn/__init__.py:4` re-export,
  exercised by the pytest gauntlet (`test_check_estimator.py`,
  `test_cross_val_score.py`).
- REQ-SUBSTRATE: the binding's array marshalling is on `ferray::numpy_interop`
  producing `ferray-core` arrays, not rust-numpy + `ndarray` (R-SUBSTRATE-1).
  [Owned by `conversions.md` REQ-FERRAY #2027.]

## Acceptance criteria

All expected values come from the live sklearn 1.5.2 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn. The pytest gauntlet
(`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`) is the
end-to-end check (verification model B); rebuild first if the Rust side changed
(`cd ferrolearn-python && maturin develop`). All ferrolearn live probes below use
`X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]])`, `y=np.array([1.,2.,3.,4.])`.

- AC-LINREG-API-CONFORM (REQ-LINREG-API-CONFORM): `test_check_estimator.py`
  (`parametrize_with_checks([LinearRegression(), ...])`) + `test_cross_val_score.py`
  pass. Spot oracle (default path):
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); y=np.array([1.,2.,3.,4.]); m=LinearRegression().fit(X,y); print([round(v,8) for v in m.coef_], round(m.intercept_,8), m.n_features_in_)"`
  → `[1.0, 0.0] 1.0 2`. `ferrolearn.LinearRegression().fit(X,y)` exposes the SAME
  `coef_`/`intercept_` (live-confirmed) and `n_features_in_ == 2`.
- AC-LINREG-FIT-INTERCEPT-ABI (REQ-LINREG-FIT-INTERCEPT-ABI): sklearn oracle
  `cd /tmp && python3 -c "import inspect; from sklearn.linear_model import LinearRegression; print(inspect.signature(LinearRegression.__init__).parameters['fit_intercept'].kind)"`
  → `KEYWORD_ONLY`. ferrolearn signature `(self, *, fit_intercept=True)` likewise
  makes it keyword-only (live) — this ABI MATCHES; no divergence.
- AC-LINREG-PARAMS (REQ-LINREG-PARAMS): sklearn exposes the extra ctor params
  (`cd /tmp && python3 -c "import inspect; from sklearn.linear_model import LinearRegression; ps=inspect.signature(LinearRegression.__init__).parameters; print([p for p in ('copy_X','n_jobs','positive') if p in ps])"`
  → `['copy_X', 'n_jobs', 'positive']`). ferrolearn signature has none. A critic
  pins FAILING pytests asserting each ∈
  `inspect.signature(ferrolearn.LinearRegression.__init__).parameters`. FAIL until
  the binding + wrapper add them (behavior owned by `ferrolearn-linear`
  #371/#374).
- AC-RIDGE-API-CONFORM (REQ-RIDGE-API-CONFORM): `test_check_estimator.py`
  (`Ridge()`) + `test_cross_val_score.py` pass. Spot oracle:
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import Ridge; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); y=np.array([1.,2.,3.,4.]); m=Ridge(alpha=1.0).fit(X,y); print([round(v,8) for v in m.coef_], round(m.intercept_,8))"`
  → `[0.33333333, 0.2] 1.3`. `ferrolearn.Ridge(alpha=1.0).fit(X,y)` matches
  element-wise (live).
- AC-RIDGE-ALPHA-POSITIONAL (REQ-RIDGE-ALPHA-POSITIONAL): sklearn oracle
  `cd /tmp && python3 -c "from sklearn.linear_model import Ridge; print(Ridge(0.5).alpha)"`
  → `0.5`. ferrolearn FAILS:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "from ferrolearn import Ridge; Ridge(0.5)"`
  → `TypeError: Ridge.__init__() takes 1 positional argument but 2 were given`
  (live-confirmed). A critic pins a FAILING pytest asserting `Ridge(0.5).alpha ==
  0.5`. FAILS until `_regressors.py::Ridge.__init__` moves `alpha` before the
  `*`.
- AC-RIDGE-PARAMS (REQ-RIDGE-PARAMS): sklearn exposes the extra params
  (`cd /tmp && python3 -c "import inspect; from sklearn.linear_model import Ridge; ps=inspect.signature(Ridge.__init__).parameters; print([p for p in ('copy_X','max_iter','tol','solver','positive','random_state') if p in ps])"`
  → all 6). ferrolearn signature is `(self, *, alpha=1.0, fit_intercept=True)` —
  none present. A critic pins FAILING pytests. FAIL until added (behavior owned by
  `ferrolearn-linear` #386/#387/#388/#390).
- AC-LASSO-API-CONFORM (REQ-LASSO-API-CONFORM): `test_check_estimator.py`
  (`Lasso()`) + `test_cross_val_score.py` pass. Spot oracle (default-path
  converged coef + support):
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import Lasso; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); y=np.array([1.,2.,3.,4.]); m=Lasso(alpha=0.1).fit(X,y); print([round(v,6) for v in m.coef_], round(m.intercept_,6))"`
  → `[0.320334, 0.199898] 1.319857`. `ferrolearn.Lasso(alpha=0.1).fit(X,y)`
  matches to the downstream-verified converged tolerance ≤1e-6 (downstream
  `ferrolearn-linear` REQ-1; live coef `[0.321100, 0.199663]` at the differing
  stopping point).
- AC-LASSO-ALPHA-POSITIONAL (REQ-LASSO-ALPHA-POSITIONAL): sklearn oracle
  `cd /tmp && python3 -c "from sklearn.linear_model import Lasso; print(Lasso(0.1).alpha)"`
  → `0.1`. ferrolearn FAILS:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "from ferrolearn import Lasso; Lasso(0.1)"`
  → `TypeError: Lasso.__init__() takes 1 positional argument but 2 were given`
  (live-confirmed). A critic pins a FAILING pytest asserting `Lasso(0.1).alpha ==
  0.1`. FAILS until `_regressors.py::Lasso.__init__` moves `alpha` before the
  `*`.
- AC-LASSO-NITER (REQ-LASSO-NITER): sklearn oracle
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import Lasso; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); y=np.array([1.,2.,3.,4.]); print(Lasso(alpha=0.1).fit(X,y).n_iter_)"`
  → `89` (the actual CD count). ferrolearn FAILS:
  `ferrolearn.Lasso(alpha=0.1).fit(X,y).n_iter_` → `1000` (live; `n_iter_ =
  self.max_iter`, a FAKE). A critic pins a FAILING pytest asserting
  `Lasso(alpha=0.1).fit(X,y).n_iter_` equals the sklearn oracle count. FAILS until
  `FittedLasso` tracks and exposes the real count (downstream `ferrolearn-linear`
  REQ-11 #411).
- AC-LASSO-PARAMS (REQ-LASSO-PARAMS): sklearn exposes the extra params
  (`cd /tmp && python3 -c "import inspect; from sklearn.linear_model import Lasso; ps=inspect.signature(Lasso.__init__).parameters; print([p for p in ('precompute','copy_X','warm_start','positive','random_state','selection') if p in ps])"`
  → all 6). ferrolearn signature is `(self, *, alpha=1.0, max_iter=1000,
  tol=1e-4, fit_intercept=True)` — none present. A critic pins FAILING pytests.
  FAIL until added (behavior owned by `ferrolearn-linear` #407/#408/#409/#410).
- AC-ELASTICNET-API-CONFORM (REQ-ELASTICNET-API-CONFORM): `test_check_estimator.py`
  (`ElasticNet()`) + `test_cross_val_score.py` pass. Spot oracle:
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import ElasticNet; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); y=np.array([1.,2.,3.,4.]); m=ElasticNet(alpha=0.1).fit(X,y); print([round(v,6) for v in m.coef_], round(m.intercept_,6))"`
  → `[0.459451, 0.160737] 1.248246`. `ferrolearn.ElasticNet(alpha=0.1).fit(X,y)`
  matches to the downstream-verified converged tolerance <1e-5 (downstream
  `ferrolearn-linear` REQ-1; live coef `[0.459807, 0.160628]`).
- AC-ELASTICNET-ALPHA-POSITIONAL (REQ-ELASTICNET-ALPHA-POSITIONAL): sklearn oracle
  `cd /tmp && python3 -c "from sklearn.linear_model import ElasticNet; print(ElasticNet(0.1).alpha)"`
  → `0.1`. ferrolearn FAILS:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "from ferrolearn import ElasticNet; ElasticNet(0.1)"`
  → `TypeError: ElasticNet.__init__() takes 1 positional argument but 2 were given`
  (live-confirmed). A critic pins a FAILING pytest asserting
  `ElasticNet(0.1).alpha == 0.1`. FAILS until `_regressors.py::ElasticNet.__init__`
  moves `alpha` before the `*` (keeping `l1_ratio` keyword-only).
- AC-ELASTICNET-NITER (REQ-ELASTICNET-NITER): sklearn oracle
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import ElasticNet; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); y=np.array([1.,2.,3.,4.]); print(ElasticNet(alpha=0.1).fit(X,y).n_iter_)"`
  → `58`. ferrolearn FAILS: `ferrolearn.ElasticNet(alpha=0.1).fit(X,y).n_iter_` →
  `1000` (live; FAKE). A critic pins a FAILING pytest. FAILS until
  `FittedElasticNet` tracks the real count (downstream `ferrolearn-linear`
  #417).
- AC-ELASTICNET-PARAMS (REQ-ELASTICNET-PARAMS): sklearn exposes the extra params
  (`cd /tmp && python3 -c "import inspect; from sklearn.linear_model import ElasticNet; ps=inspect.signature(ElasticNet.__init__).parameters; print([p for p in ('precompute','copy_X','warm_start','positive','random_state','selection') if p in ps])"`
  → all 6). ferrolearn signature is `(self, *, alpha=1.0, l1_ratio=0.5,
  max_iter=1000, tol=1e-4, fit_intercept=True)` — none present. A critic pins
  FAILING pytests. FAIL until added (behavior owned by `ferrolearn-linear`
  #407/#408/#409/#410).
- AC-CONSUMER (REQ-CONSUMER):
  `grep -n "_RsLinearRegression\|_RsRidge\|_RsLasso\|_RsElasticNet" /home/doll/ferrolearn/ferrolearn-python/python/ferrolearn/_regressors.py`
  shows each wrapper constructs its `_Rs*` class and drives fit/predict + reads
  `coef_`/`intercept_`; `ferrolearn/__init__.py:4` re-exports all four;
  `test_check_estimator.py:22-25` runs `LinearRegression()`/`Ridge()`/`Lasso()`/
  `ElasticNet()` through `parametrize_with_checks` and `test_cross_val_score.py:44-49`
  through `cross_val_score`. The 534-passing pytest exercises the consumer surface.
- AC-SUBSTRATE (REQ-SUBSTRATE): `regressors.rs` head shows
  `use crate::conversions::*` + `use numpy::{PyArray1, PyReadonlyArray1,
  PyReadonlyArray2}` — the wrong substrate per R-SUBSTRATE-1 (destination
  `ferray::numpy_interop`/`ferray-core`). ferray exposes no `numpy_interop`
  bridge consumable here (R-SUBSTRATE-5). Owned by `conversions.md` #2027.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-LINREG-API-CONFORM (fit/predict + coef_/intercept_, default path) | SHIPPED | impl `RsLinearRegression::fit`/`predict` + getters `coef_`/`intercept_` in `regressors.rs` (over `ferrolearn_linear::FittedLinearRegression<f64>`, via `fitted.coefficients()`/`fitted.intercept()`), wrapped by `LinearRegression` in `_regressors.py` which sets `coef_`/`intercept_` from the Rust getters + `n_features_in_` (via `self._validate_data`) and inherits `score` from `RegressorMixin` — mirroring `sklearn/linear_model/_base.py:582` (fit) + `:324`/`:327` (intercept). Non-test consumer: `_regressors.py::LinearRegression` + `ferrolearn/__init__.py:4` re-export; external users. Verification (model B): `cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` → 534 passed, 0 failed (`test_check_estimator.py:22` + `test_cross_val_score.py:44`). Live default-path oracle MATCHES element-wise: `coef_=[1.0, 0.0]`, `intercept_=1.0`, `n_features_in_=2`. |
| REQ-LINREG-FIT-INTERCEPT-ABI (fit_intercept keyword-only) | SHIPPED | `_regressors.py::LinearRegression.__init__(self, *, fit_intercept=True)` makes `fit_intercept` keyword-only, MATCHING sklearn `_base.py:568-570` (`__init__(self, *, fit_intercept=True, ...)` — the `*` is first). Marshalled to `RsLinearRegression::new` via `#[pyo3(signature = (fit_intercept=true))]` + `with_fit_intercept`. Live: both `inspect.signature(...).parameters['fit_intercept'].kind` → `KEYWORD_ONLY`. Non-test consumer: `_regressors.py::LinearRegression`. This is the ONLY estimator whose constructor-ABI matches sklearn (Ridge/Lasso/ElasticNet diverge on `alpha`). |
| REQ-LINREG-RANK-SINGULAR (rank_/singular_ fitted attrs) | SHIPPED | impl `RsLinearRegression::rank_` getter (over `FittedLinearRegression<f64>`, via `fitted.rank()`) + `RsLinearRegression::singular_` getter (via `fitted.singular_values()`, marshalled through `ndarray1_to_numpy`) in `regressors.rs`, surfaced by `_regressors.py::LinearRegression.fit` which sets `self.rank_ = int(self._rs.rank_)` + `self.singular_ = np.array(self._rs.singular_)` (next to `coef_`/`intercept_`). Mirrors sklearn `self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)` (`_base.py:687`; attr docstrings `rank_` `_base.py:505`, `singular_` `_base.py:508`). The downstream Rust `FittedLinearRegression::rank()`/`singular_values()` capture both from the single-SVD `solve_lstsq` on the actually-solved (centered-when-`fit_intercept`) matrix, matching sklearn's `linalg.lstsq` operands (`ferrolearn-linear` REQ-9 SHIPPED #374, verified by `linreg_rank_singular_match_sklearn_with_intercept`). Non-test consumer: `_regressors.py::LinearRegression` + `ferrolearn/__init__.py` re-export. Verification (model B): `tests/divergence_regressors.py::test_linearregression_rank_singular_match_sklearn` asserts `fr.rank_ == sk.rank_` and `np.testing.assert_allclose(fr.singular_, sk.singular_, atol=1e-8)` (live oracle, R-CHAR-3; on the 5×2 fixture sklearn yields `rank_=2`, `singular_=[4.24264069, 1.41421356]`). |
| REQ-LINREG-PARAMS (copy_X/n_jobs/positive) | NOT-STARTED | open prereq blockers #371 (`positive`/NNLS, `ferrolearn-linear` REQ-6) + #374 (`copy_X`/`n_jobs`, REQ-9; the `rank_`/`singular_` half of #374 is now SHIPPED — see REQ-LINREG-RANK-SINGULAR). sklearn `_base.py:568-579`. ferrolearn `_regressors.py::LinearRegression.__init__` exposes `fit_intercept` only; `RsLinearRegression::new` likewise. The default (full-rank/min-norm OLS, no positivity constraint, copy) MATCHES, so only the param surface + non-default behavior is missing — the binding cannot expose what the library lacks. (The wrapper copies via `_validate_data`, so OBSERVABLE non-mutation holds, but the `copy_X`/`n_jobs`/`positive` ABI params are absent — R-DEV-2.) |
| REQ-LINREG-VALUE-PARITY (coef_/intercept_ array parity) | SHIPPED | on the DEFAULT path. `_regressors.py::LinearRegression.fit` marshals `coef_`/`intercept_` from the `RsLinearRegression` getters (over `FittedLinearRegression`), and downstream `ferrolearn-linear` REQ-1/REQ-5 are critic-verified to MATCH the live sklearn `LinearRegression` oracle to 1e-8 for full-rank, rank-deficient, and underdetermined OLS (single-SVD min-norm via `ferray::linalg::lstsq`). Live (R-CHAR-3): ferrolearn `LinearRegression().fit(X,y)` `coef_=[1.0,0.0]`, `intercept_=1.0` equal the oracle element-wise. Non-test consumer: `_regressors.py::LinearRegression` + re-export. (`positive=True` NNLS parity is owned downstream #371; multi-output 2-D `y` → 2-D `coef_` #372.) |
| REQ-RIDGE-API-CONFORM (fit/predict + coef_/intercept_, default cholesky path) | SHIPPED | impl `RsRidge::fit`/`predict` + getters `coef_`/`intercept_` in `regressors.rs` (over `ferrolearn_linear::FittedRidge<f64>`), wrapped by `Ridge` in `_regressors.py` which marshals `alpha` via `RsRidge::new(alpha=...)` → `Ridge::with_alpha`, sets `coef_`/`intercept_` + `n_features_in_`, inherits `score` from `RegressorMixin` — mirroring `sklearn/linear_model/_ridge.py:914` (fit, default `solver='auto'`→cholesky) + `:968`/`:984` (coef/intercept). Non-test consumer: `_regressors.py::Ridge` + `ferrolearn/__init__.py:4` re-export. Verification (model B): pytest → 534 passed (`test_check_estimator.py:23` + `test_cross_val_score.py:45`). Live default-path oracle MATCHES element-wise: `coef_=[0.33333333,0.2]`, `intercept_=1.3`. |
| REQ-RIDGE-ALPHA-POSITIONAL (alpha positional ABI) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 constructor ABI; single-wrapper-fixable — HEADLINE 1/3). sklearn `__init__(self, alpha=1.0, *, ...)` (`_ridge.py:893-895`) makes `alpha` positional-or-keyword — `Ridge(0.5).alpha` → `0.5`. ferrolearn `_regressors.py::Ridge.__init__(self, *, alpha=1.0, fit_intercept=True)` makes it keyword-only — live: `ferrolearn.Ridge(0.5)` → `TypeError: __init__() takes 1 positional argument but 2 were given`. Single-line Python-wrapper fix: move `alpha` before the `*`. |
| REQ-RIDGE-PARAMS (copy_X/max_iter/tol/solver/positive/random_state) | NOT-STARTED | open prereq blockers #386 (`solver`/`solver_`, `ferrolearn-linear` REQ-8) + #387 (`positive`, REQ-9) + #388 (`max_iter`/`tol`/`n_iter_`, REQ-10) + #390 (`copy_X`/`random_state`, REQ-12). sklearn `_ridge.py:893-912`. ferrolearn `_regressors.py::Ridge.__init__` exposes `alpha`/`fit_intercept` only; `RsRidge::new` takes `alpha`/`fit_intercept`. The default cholesky/no-positivity behavior MATCHES, so only the param surface + non-default paths are missing — owned downstream; the binding cannot expose what the library lacks. |
| REQ-RIDGE-VALUE-PARITY (coef_/intercept_ array parity, default cholesky) | SHIPPED | on the DEFAULT path. `_regressors.py::Ridge.fit` marshals `coef_`/`intercept_` from the `RsRidge` getters (over `FittedRidge`), and downstream `ferrolearn-linear` REQ-1/REQ-5 are critic-verified to MATCH the live sklearn `Ridge` oracle to 1e-8 across alpha∈{0.1,1,10,100} (closed-form Cholesky, intercept unpenalized; alpha=0 rank-deficient min-norm fallback). Live (R-CHAR-3): ferrolearn `Ridge(alpha=1.0).fit(X,y)` `coef_=[0.33333333,0.2]`, `intercept_=1.3` equal the oracle element-wise. Non-test consumer: `_regressors.py::Ridge` + re-export. (Per-target `alpha` array #385 and alternate-solver values are owned downstream; multi-output 2-D `y` SHIPPED — see REQ-RIDGE-MULTIOUTPUT.) |
| REQ-RIDGE-MULTIOUTPUT (2-D Y → 2-D coef_) | SHIPPED | `ferrolearn.Ridge.fit` accepts a 2-D `y` `(n_samples, n_targets)` and exposes `coef_` `(n_targets, n_features)` + `intercept_` `(n_targets,)`, matching sklearn `Ridge` (`sklearn/linear_model/_ridge.py:543` coef shape, `:550` intercept shape; per-target solve `:207`/`:218`). impl `#[pyclass(name="_RsRidgeMultiOutput")] RsRidgeMultiOutput::fit`/`predict` + `coef_`/`intercept_` getters in `regressors.rs` over `ferrolearn_linear::FittedRidgeMulti<f64>` (the `Fit<Array2, Array2> for Ridge` path, `ferrolearn-linear/src/ridge.rs:725`; `FittedRidgeMulti::coefficients()` `(n_features,n_targets)` `:713` + `intercepts()` `:719`, shipped #29). The `coef_` getter TRANSPOSES `(n_features,n_targets)`→`(n_targets,n_features)` to match sklearn's output contract (R-DEV-3). Non-test consumer: `_regressors.py::Ridge.fit` routes the `y.ndim==2 and y.shape[1]>1` path to `_RsRidgeMultiOutput` (registered in `lib.rs`) + `ferrolearn/__init__.py` re-export. Verification (model B): `tests/divergence_regressors.py::test_ridge_multioutput_matches_sklearn` asserts `coef_` `(2,2)`, `intercept_` `(2,)`, `predict` match the live sklearn oracle ≤1e-8 (R-CHAR-3; oracle `coef_=[[0.82280702,1.35614035],[0.61403509,-0.71929825]]`, `intercept_=[-0.57684211,1.71578947]`); `test_ridge_singleoutput_unchanged_by_multioutput_path` guards the 1-D path stays scalar-intercept. Downstream `ferrolearn-linear` REQ-6 #384. |
| REQ-LASSO-API-CONFORM (fit/predict + coef_/intercept_, default cyclic path) | SHIPPED | impl `RsLasso::fit`/`predict` + getters `coef_`/`intercept_` in `regressors.rs` (over `ferrolearn_linear::FittedLasso<f64>`), wrapped by `Lasso` in `_regressors.py` which marshals `alpha`/`max_iter`/`tol` via `RsLasso::new` → `with_alpha`/`with_max_iter`/`with_tol`, sets `coef_`/`intercept_` + `n_features_in_`, inherits `score` from `RegressorMixin` — mirroring `sklearn/linear_model/_coordinate_descent.py:932` (CD fit) + `:1107` (coef). Non-test consumer: `_regressors.py::Lasso` + `ferrolearn/__init__.py:4` re-export. Verification (model B): pytest → 534 passed (`test_check_estimator.py:24` + `test_cross_val_score.py:46`). Live default-path coef matches the oracle to the downstream-verified converged tolerance (downstream `ferrolearn-linear` REQ-1/4/6: converged coef + exact-zero support ≤1e-6). |
| REQ-LASSO-ALPHA-POSITIONAL (alpha positional ABI) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 constructor ABI; single-wrapper-fixable — HEADLINE 2/3). sklearn `__init__(self, alpha=1.0, *, ...)` (`_coordinate_descent.py:1310-1312`) makes `alpha` positional-or-keyword — `Lasso(0.1).alpha` → `0.1`. ferrolearn `_regressors.py::Lasso.__init__(self, *, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True)` makes it keyword-only — live: `ferrolearn.Lasso(0.1)` → `TypeError: __init__() takes 1 positional argument but 2 were given`. Single-line Python-wrapper fix: move `alpha` before the `*`. |
| REQ-LASSO-NITER (n_iter_ = real CD count) | SHIPPED (#2043) | FIXED — `ferrolearn-linear` REQ-11 already SHIPPED (`FittedLasso::n_iter()` tracks the real count, matching sklearn EXACTLY via dual-gap CD stopping). `RsLasso::n_iter_` (`regressors.rs`) now binds it; `_regressors.py::Lasso.fit` sets `self.n_iter_ = int(self._rs.n_iter_)` (was the `max_iter` fake). Live oracle (R-CHAR-3): `ferrolearn.Lasso(0.5).n_iter_ == sklearn 2`, `Lasso(0.1) == 2` — exact, `< max_iter`. Guard `tests/divergence_regressors.py::test_lasso_elasticnet_n_iter_matches_sklearn`. |
| REQ-LASSO-DUALGAP (dual_gap_ fitted attribute) | SHIPPED (#2096) | impl `RsLasso::dual_gap_` getter in `regressors.rs` (over `FittedLasso<f64>`, via `fitted.dual_gap()`), surfaced by `_regressors.py::Lasso.fit` which sets `self.dual_gap_ = float(self._rs.dual_gap_)` next to `n_iter_`. Mirrors sklearn `self.dual_gap_ = dual_gaps_[0]` (`_coordinate_descent.py:1108`, single-target collapse; `:1111` multi-target). The downstream Rust `FittedLasso::dual_gap()` matches sklearn EXACTLY (`ferrolearn-linear` REQ-11 SHIPPED, dual-gap CD stopping). Non-test consumer: `_regressors.py::Lasso` + `ferrolearn/__init__.py` re-export. Verification (model B): `tests/divergence_regressors.py::test_lasso_elasticnet_dual_gap_matches_sklearn` asserts `abs(fl.Lasso(alpha=0.3).fit(X,y).dual_gap_ - SkLasso(alpha=0.3).fit(X,y).dual_gap_) < 1e-9` (live oracle, R-CHAR-3). |
| REQ-LASSO-PARAMS (precompute/copy_X/warm_start/positive/random_state/selection) | NOT-STARTED | open prereq blockers #407 (`positive`, `ferrolearn-linear` REQ-7) + #408 (`warm_start`, REQ-8) + #409 (`selection`/`random_state`, REQ-9) + #410 (`precompute`, REQ-10). sklearn `_coordinate_descent.py:1310-1322`. ferrolearn `_regressors.py::Lasso.__init__` exposes `alpha`/`max_iter`/`tol`/`fit_intercept` only; `RsLasso::new` likewise. The default cyclic/no-positivity/no-precompute behavior MATCHES, so only the param surface + non-default paths are missing — owned downstream. |
| REQ-LASSO-VALUE-PARITY (coef_/intercept_ + support set array parity, default cyclic) | SHIPPED | on the DEFAULT converged path. `_regressors.py::Lasso.fit` marshals `coef_`/`intercept_` from the `RsLasso` getters (over `FittedLasso`), and downstream `ferrolearn-linear` REQ-1/4/6 are critic-verified ("NO DIVERGENCE FOUND") to MATCH the live sklearn `Lasso` oracle to ≤1e-6 for converged coef/intercept + the exact-zero support set (cyclic CD with soft-thresholding, `l1_reg=α·n` convention), incl. alpha=0. Live (R-CHAR-3): ferrolearn `Lasso(alpha=0.1).fit(X,y)` coef agrees with the oracle to the converged tolerance. Non-test consumer: `_regressors.py::Lasso` + re-export. (The dual-gap stopping criterion #412 and `positive`/`selection='random'` paths are owned downstream.) |
| REQ-ELASTICNET-API-CONFORM (fit/predict + coef_/intercept_, default cyclic path) | SHIPPED | impl `RsElasticNet::fit`/`predict` + getters `coef_`/`intercept_` in `regressors.rs` (over `ferrolearn_linear::FittedElasticNet<f64>`), wrapped by `ElasticNet` in `_regressors.py` which marshals `alpha`/`l1_ratio`/`max_iter`/`tol` via `RsElasticNet::new` → `with_alpha`/`with_l1_ratio`/`with_max_iter`/`with_tol`, sets `coef_`/`intercept_` + `n_features_in_`, inherits `score` from `RegressorMixin` — mirroring `sklearn/linear_model/_coordinate_descent.py:932` (CD fit, L1/L2 split) + `:1107` (coef). Non-test consumer: `_regressors.py::ElasticNet` + `ferrolearn/__init__.py:4` re-export. Verification (model B): pytest → 534 passed (`test_check_estimator.py:25` + `test_cross_val_score.py:47`). Live default-path coef matches the oracle to the downstream-verified converged tolerance (downstream `ferrolearn-linear` REQ-1/4/5: converged coef + support <1e-5, incl. l1_ratio=1↔Lasso / l1_ratio=0↔L2). |
| REQ-ELASTICNET-ALPHA-POSITIONAL (alpha positional ABI) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 constructor ABI; single-wrapper-fixable — HEADLINE 3/3). sklearn `__init__(self, alpha=1.0, *, l1_ratio=0.5, ...)` (`_coordinate_descent.py:898-902`) makes `alpha` positional-or-keyword (`l1_ratio` correctly keyword-only) — `ElasticNet(0.1).alpha` → `0.1`. ferrolearn `_regressors.py::ElasticNet.__init__(self, *, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=True)` makes `alpha` keyword-only — live: `ferrolearn.ElasticNet(0.1)` → `TypeError: __init__() takes 1 positional argument but 2 were given`. Single-line Python-wrapper fix: move `alpha` before the `*`, keep `l1_ratio` after. |
| REQ-ELASTICNET-NITER (n_iter_ = real CD count) | SHIPPED (#2043) | FIXED — `ferrolearn-linear` REQ-12 already SHIPPED (`FittedElasticNet::n_iter()` tracks the real count, matching sklearn EXACTLY via dual-gap CD stopping). `RsElasticNet::n_iter_` (`regressors.rs`) now binds it; `_regressors.py::ElasticNet.fit` sets `self.n_iter_ = int(self._rs.n_iter_)` (was the `max_iter` fake). Live oracle (R-CHAR-3): `ferrolearn.ElasticNet(0.5).n_iter_ == sklearn 2` — exact, `< max_iter`. Guard `tests/divergence_regressors.py::test_lasso_elasticnet_n_iter_matches_sklearn`. |
| REQ-ELASTICNET-DUALGAP (dual_gap_ fitted attribute) | SHIPPED (#2096) | impl `RsElasticNet::dual_gap_` getter in `regressors.rs` (over `FittedElasticNet<f64>`, via `fitted.dual_gap()`), surfaced by `_regressors.py::ElasticNet.fit` which sets `self.dual_gap_ = float(self._rs.dual_gap_)` next to `n_iter_`. Mirrors sklearn `self.dual_gap_ = dual_gaps_[0]` (`_coordinate_descent.py:1108`, single-target collapse; `:1111` multi-target). The downstream Rust `FittedElasticNet::dual_gap()` matches sklearn EXACTLY (`ferrolearn-linear` REQ-12 SHIPPED, dual-gap CD stopping). Non-test consumer: `_regressors.py::ElasticNet` + `ferrolearn/__init__.py` re-export. Verification (model B): `tests/divergence_regressors.py::test_lasso_elasticnet_dual_gap_matches_sklearn` asserts `abs(fl.ElasticNet(alpha=0.3).fit(X,y).dual_gap_ - SkElasticNet(alpha=0.3).fit(X,y).dual_gap_) < 1e-9` (live oracle, R-CHAR-3, default l1_ratio=0.5). |
| REQ-ELASTICNET-PARAMS (precompute/copy_X/warm_start/positive/random_state/selection) | NOT-STARTED | open prereq blockers #407 (`positive`, `ferrolearn-linear` REQ-8) + #408 (`warm_start`) + #409 (`selection`/`random_state`) + #410 (`precompute`). sklearn `_coordinate_descent.py:898-912`. ferrolearn `_regressors.py::ElasticNet.__init__` exposes `alpha`/`l1_ratio`/`max_iter`/`tol`/`fit_intercept` only; `RsElasticNet::new` likewise. The default cyclic/no-positivity/no-precompute behavior MATCHES, so only the param surface + non-default paths are missing — owned downstream. |
| REQ-ELASTICNET-VALUE-PARITY (coef_/intercept_ + support set array parity, default cyclic) | SHIPPED | on the DEFAULT converged path. `_regressors.py::ElasticNet.fit` marshals `coef_`/`intercept_` from the `RsElasticNet` getters (over `FittedElasticNet`), and downstream `ferrolearn-linear` REQ-1/4/5 are critic-verified ("NO DIVERGENCE FOUND") to MATCH the live sklearn `ElasticNet` oracle to <1e-5 for converged coef/intercept + the exact-zero support set over the (alpha, l1_ratio) grid, with `l1_reg=α·l1_ratio·n` / `l2_reg=α·(1−l1_ratio)·n`, incl. l1_ratio=1↔Lasso and l1_ratio=0↔L2. Live (R-CHAR-3): ferrolearn `ElasticNet(alpha=0.1).fit(X,y)` coef agrees with the oracle to the converged tolerance. Non-test consumer: `_regressors.py::ElasticNet` + re-export. (The dual-gap stopping criterion #412 and `positive`/`selection='random'` paths are owned downstream.) |
| REQ-CONSUMER (binding IS the public API) | SHIPPED | the binding boundary types ARE the public API (R-DEFER-1/S5: boundary estimator types ARE the public surface; grandfathered existing pub API). Non-test production consumers: `_regressors.py::{LinearRegression,Ridge,Lasso,ElasticNet}` each construct their `_Rs*` class and call `fit`/`predict` + read the `coef_`/`intercept_` getters (`grep -n "_RsLinearRegression\|_RsRidge\|_RsLasso\|_RsElasticNet" python/ferrolearn/_regressors.py`); `ferrolearn/__init__.py:4` re-exports all four; `test_check_estimator.py:22-25` runs them through `parametrize_with_checks` and `test_cross_val_score.py:44-49` through `cross_val_score` + external users. Verification (model B): pytest → 534 passed (all four exercised in both consumer-suite files). |
| REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | open prereq blocker = `conversions.md` REQ-FERRAY #2027. `regressors.rs` marshals via `use crate::conversions::*` + `use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2}` (rust-numpy) and the conversions produce `ndarray::Array{1,2}` — the WRONG substrate per R-SUBSTRATE-1 (destination `ferray::numpy_interop` + `ferray-core`). ferray exposes no PyO3 numpy-interop bridge yet (R-SUBSTRATE-5). Owned by the conversions unit, surfaced here. |

## Architecture

`regressors.rs` holds four `#[pyclass]` structs, each a THIN shim over a fitted
library type and owning ZERO regression math:

- **`RsLinearRegression`** wraps `Option<ferrolearn_linear::FittedLinearRegression<f64>>`
  + a `fit_intercept: bool`. `new` (`#[pyo3(signature = (fit_intercept=true))]`)
  stores the flag; `fit` runs `LinearRegression::<f64>::new().with_fit_intercept(..)`
  on the `numpy2_to_ndarray`/`numpy1_to_ndarray`-coerced X/y and stores the fitted
  model (mapping `FerroError` → `PyValueError`); `predict` and the `coef_`/
  `intercept_` getters delegate to `FittedLinearRegression`
  (`fitted.coefficients()`/`fitted.intercept()`), returning `PyRuntimeError("not
  fitted")` before fit.
- **`RsRidge`** stores `alpha`/`fit_intercept` + `Option<FittedRidge<f64>>`. `new`
  (`#[pyo3(signature = (alpha=1.0, fit_intercept=true))]`); `fit` builds
  `Ridge::<f64>::new().with_alpha(..).with_fit_intercept(..)`.
- **`RsLasso`** stores `alpha`/`max_iter`/`tol`/`fit_intercept` +
  `Option<FittedLasso<f64>>`. `new` (`#[pyo3(signature = (alpha=1.0,
  max_iter=1000, tol=1e-4, fit_intercept=true))]`); `fit` builds
  `Lasso::<f64>::new().with_alpha(..).with_max_iter(..).with_tol(..).with_fit_intercept(..)`.
  NO `n_iter_` getter.
- **`RsElasticNet`** stores `alpha`/`l1_ratio`/`max_iter`/`tol`/`fit_intercept` +
  `Option<FittedElasticNet<f64>>`. `new` (`#[pyo3(signature = (alpha=1.0,
  l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=true))]`); `fit` builds
  `ElasticNet::<f64>::new().with_alpha(..).with_l1_ratio(..).with_max_iter(..)
  .with_tol(..).with_fit_intercept(..)`. NO `n_iter_` getter.

All four expose `coef_`/`intercept_` getters (plus `fit`/`predict`).
`RsLinearRegression` additionally exposes `rank_`/`singular_` (over
`FittedLinearRegression::rank()`/`singular_values()` — REQ-LINREG-RANK-SINGULAR);
`RsLasso`/`RsElasticNet` additionally expose `n_iter_`/`dual_gap_`.

`_regressors.py` wraps each as a sklearn-facing `RegressorMixin, BaseEstimator`
subclass. Every wrapper `__init__` makes ALL params keyword-only (a leading `*`):

- **`LinearRegression`** — `__init__(self, *, fit_intercept=True)`. The leading `*`
  is HARMLESS here because sklearn ALSO makes `fit_intercept` keyword-only
  (`_base.py:568`) — so the ABI MATCHES (REQ-LINREG-FIT-INTERCEPT-ABI SHIPPED).
  No `copy_X`/`n_jobs`/`positive` (REQ-LINREG-PARAMS).
- **`Ridge`** — `__init__(self, *, alpha=1.0, fit_intercept=True)`. The leading
  `*` makes `alpha` keyword-only, DIVERGING from sklearn's positional `alpha`
  (REQ-RIDGE-ALPHA-POSITIONAL). No `copy_X`/`max_iter`/`tol`/`solver`/`positive`/
  `random_state` (REQ-RIDGE-PARAMS).
- **`Lasso`** — `__init__(self, *, alpha=1.0, max_iter=1000, tol=1e-4,
  fit_intercept=True)`; `fit` sets `self.n_iter_ = self.max_iter` (a FAKE —
  REQ-LASSO-NITER). The leading `*` makes `alpha` keyword-only
  (REQ-LASSO-ALPHA-POSITIONAL). No `precompute`/`copy_X`/`warm_start`/`positive`/
  `random_state`/`selection` (REQ-LASSO-PARAMS).
- **`ElasticNet`** — `__init__(self, *, alpha=1.0, l1_ratio=0.5, max_iter=1000,
  tol=1e-4, fit_intercept=True)`; `fit` sets `self.n_iter_ = self.max_iter` (a
  FAKE — REQ-ELASTICNET-NITER). The leading `*` makes `alpha` keyword-only
  (REQ-ELASTICNET-ALPHA-POSITIONAL); `l1_ratio` being keyword-only MATCHES
  sklearn. No `precompute`/`copy_X`/`warm_start`/`positive`/`random_state`/
  `selection` (REQ-ELASTICNET-PARAMS).

Each wrapper's `fit` calls `self._validate_data(X, y, dtype="float64",
y_numeric=True)` (setting `n_features_in_`), coerces via `_ensure_f64`, constructs
its `_Rs*` class, drives `_fit_rust`, and copies `coef_`/`intercept_` from the
Rust getters; `predict` `check_is_fitted` + re-validate + (lazily fall back to a
stored-coef linear prediction via `_predict_linear` when `_rs` is absent after
unpickling). `__getstate__`/`__setstate__` drop and rebuild `_rs`.

The honest call (R-HONEST-3): the API-CONFORMANCE, the default-path VALUE parity
(all four), the LinearRegression `fit_intercept` ABI, and the CONSUMER ship on
impl + the 534-passing pytest (`check_estimator` + `cross_val_score` for all
four) + the live default-path oracle matching element-wise. Everything else is
NOT-STARTED (binary, R-DEFER-2) and splits into:

- **Single-wrapper-fixable alpha-positional ABI divergences (THE HEADLINE — three
  distinct)**: REQ-RIDGE-ALPHA-POSITIONAL, REQ-LASSO-ALPHA-POSITIONAL,
  REQ-ELASTICNET-ALPHA-POSITIONAL. Each is a one-line Python-wrapper fix (move
  `alpha` before the leading `*` in the respective `__init__`), needing no Rust
  change. `ferrolearn.{Ridge,Lasso,ElasticNet}(0.5)` each raise `TypeError` today;
  sklearn's `alpha` is positional. (LinearRegression is NOT in this set — its
  `fit_intercept`-only constructor matches sklearn's keyword-only `fit_intercept`,
  so REQ-LINREG-FIT-INTERCEPT-ABI is SHIPPED.)
- **Faked fitted attribute**: REQ-LASSO-NITER / REQ-ELASTICNET-NITER — the wrapper
  hardcodes `n_iter_ = max_iter` (`1000`) because `FittedLasso`/`FittedElasticNet`
  do not track the real coordinate-descent count, so the binding cannot surface
  it. Owned downstream (#411 Lasso, #417 ElasticNet).
- **Wrapper-surfaced, downstream-owned missing params**: REQ-LINREG-PARAMS (#371,
  #374), REQ-RIDGE-PARAMS (#386/#387/#388/#390), REQ-LASSO-PARAMS /
  REQ-ELASTICNET-PARAMS (#407/#408/#409/#410). The DEFAULTS all match (OLS
  min-norm / cholesky / cyclic CD, no positivity, no precompute), so only the
  param surface + non-default paths are missing — and the binding cannot expose
  what the library crate lacks.
- **Off-default value parity**: the *-VALUE-PARITY REQs are SHIPPED on the DEFAULT
  path; `positive=True` (NNLS), multi-output 2-D `y`, per-target Ridge `alpha`,
  alternate solvers, and the Lasso/ElasticNet dual-gap stopping criterion are
  owned downstream (#371/#372/#384/#385/#386/#412).
- **Substrate**: REQ-SUBSTRATE — rust-numpy + `ndarray`, not
  `ferray::numpy_interop` (owned by `conversions.md` #2027).

The least-confident SHIPPED claim is REQ-LASSO-VALUE-PARITY (and the twin
REQ-ELASTICNET-VALUE-PARITY): it is SHIPPED on the converged DEFAULT cyclic path,
where downstream `ferrolearn-linear` REQ-1/4/6 verify coef/intercept + the
exact-zero support set to the converged tolerance (≤1e-6 / <1e-5). But ferrolearn
and sklearn use DIFFERENT stopping criteria (max-coef-change vs sklearn's
relative-change + dual-gap, downstream #412), so on a fixed `(max_iter, tol)`
they stop at different iterates — on the live 4×2 fixture the coef agree only to
~1e-3 at the differing stopping point, not to 1e-6, because neither side is fully
converged at `tol=1e-4`. The REQ ships the *converged-optimum* contract (the
support set is bit-identical and the coef agree as both approach the optimum), not
universal per-`(max_iter, tol)` iterate parity — the stopping-criterion divergence
is honestly carved to downstream #412 (R-HONEST-3 underclaim). The
LinearRegression/Ridge VALUE-PARITY claims are firmer (deterministic closed-form,
1e-8 element-wise).

## Verification

Commands establishing the SHIPPED claims (run at baseline `7ba78cf27`,
verification model B; rebuild first if the Rust side changed:
`cd /home/doll/ferrolearn/ferrolearn-python && maturin develop`). All probes use
`X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]])`, `y=np.array([1.,2.,3.,4.])`:

- **Full pytest gauntlet (all four REQ-*-API-CONFORM, all four
  REQ-*-VALUE-PARITY default path, REQ-CONSUMER, REQ-LINREG-FIT-INTERCEPT-ABI):**
  `cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`
  → `534 passed`. `test_check_estimator.py:22-25` runs `LinearRegression()`/
  `Ridge()`/`Lasso()`/`ElasticNet()` through sklearn's `parametrize_with_checks`;
  `test_cross_val_score.py:44-49` runs them through `cross_val_score`.
- **LinearRegression default-path oracle (REQ-LINREG-API-CONFORM,
  REQ-LINREG-VALUE-PARITY; R-CHAR-3):**
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); y=np.array([1.,2.,3.,4.]); m=LinearRegression().fit(X,y); print([round(v,8) for v in m.coef_], round(m.intercept_,8))"`
  → `[1.0, 0.0] 1.0`. ferrolearn matches element-wise (live).
- **Ridge default-path oracle (REQ-RIDGE-API-CONFORM, REQ-RIDGE-VALUE-PARITY):**
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import Ridge; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); y=np.array([1.,2.,3.,4.]); m=Ridge(alpha=1.0).fit(X,y); print([round(v,8) for v in m.coef_], round(m.intercept_,8))"`
  → `[0.33333333, 0.2] 1.3`. ferrolearn matches element-wise (live).
- **Lasso / ElasticNet default-path oracle (REQ-LASSO/ELASTICNET-API-CONFORM,
  -VALUE-PARITY):**
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import Lasso, ElasticNet; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); y=np.array([1.,2.,3.,4.]); print([round(v,6) for v in Lasso(alpha=0.1).fit(X,y).coef_], [round(v,6) for v in ElasticNet(alpha=0.1).fit(X,y).coef_])"`
  → `[0.320334, 0.199898] [0.459451, 0.160737]`. ferrolearn agrees to the
  downstream-verified converged tolerance (downstream `ferrolearn-linear` REQ-1;
  exact-zero support set bit-identical).
- **alpha-positional ABI oracle (REQ-RIDGE/LASSO/ELASTICNET-ALPHA-POSITIONAL;
  R-CHAR-3 — THE HEADLINE):**
  `cd /tmp && python3 -c "from sklearn.linear_model import Ridge, Lasso, ElasticNet; print(Ridge(0.5).alpha, Lasso(0.1).alpha, ElasticNet(0.1).alpha)"`
  → `0.5 0.1 0.1`. ferrolearn:
  `cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -c "from ferrolearn import Ridge, Lasso, ElasticNet;\nfor C,a in [(Ridge,0.5),(Lasso,0.1),(ElasticNet,0.1)]:\n  try: C(a)\n  except TypeError as e: print(C.__name__, 'TypeError')"`
  → each raises `TypeError: __init__() takes 1 positional argument but 2 were
  given` (live-confirmed). Three critic-pinned FAILING pytests; each FAILS until
  its wrapper moves `alpha` before the `*`.
- **LinearRegression fit_intercept-ABI oracle (REQ-LINREG-FIT-INTERCEPT-ABI):**
  `cd /tmp && python3 -c "import inspect; from sklearn.linear_model import LinearRegression; print(inspect.signature(LinearRegression.__init__).parameters['fit_intercept'].kind)"`
  → `KEYWORD_ONLY`. ferrolearn `(self, *, fit_intercept=True)` likewise (live) —
  MATCHES; no divergence (SHIPPED).
- **n_iter_ oracle (REQ-LASSO-NITER, REQ-ELASTICNET-NITER):**
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import Lasso, ElasticNet; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); y=np.array([1.,2.,3.,4.]); print(Lasso(alpha=0.1).fit(X,y).n_iter_, ElasticNet(alpha=0.1).fit(X,y).n_iter_)"`
  → `89 58` (the real CD counts). ferrolearn: both `n_iter_` → `1000` (FAKE =
  max_iter; live). Critic-pinned FAILING pytests; FAIL until `FittedLasso`/
  `FittedElasticNet` track the real count (downstream #411/#417).
- **missing-param oracle (REQ-LINREG/RIDGE/LASSO/ELASTICNET-PARAMS):**
  `cd /tmp && python3 -c "import inspect; from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet; f=lambda C,ps: [p for p in ps if p in inspect.signature(C.__init__).parameters]; print(f(LinearRegression,('copy_X','n_jobs','positive')), f(Ridge,('copy_X','max_iter','tol','solver','positive','random_state')), f(Lasso,('precompute','copy_X','warm_start','positive','random_state','selection')))"`
  → all present in sklearn; ferrolearn's signatures have none. Critic-pinned
  FAILING pytests; FAIL until the binding + wrapper add them (behavior owned by
  `ferrolearn-linear` #371/#374/#386/#387/#388/#390/#407/#408/#409/#410).
- **Consumer check (REQ-CONSUMER):**
  `grep -n "_RsLinearRegression\|_RsRidge\|_RsLasso\|_RsElasticNet" /home/doll/ferrolearn/ferrolearn-python/python/ferrolearn/_regressors.py`
  shows each `_regressors.py` wrapper constructs its `_Rs*` class and drives
  fit/predict + attribute reads; `ferrolearn/__init__.py:4` re-exports all four;
  the 534-passing pytest exercises them.
- **Substrate check (REQ-SUBSTRATE):** `regressors.rs` head shows
  `use crate::conversions::*` + `use numpy::{PyArray1, PyReadonlyArray1,
  PyReadonlyArray2}` — the wrong substrate per R-SUBSTRATE-1; ferray exposes no
  `numpy_interop` bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027.
