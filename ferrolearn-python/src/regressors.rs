//! PyO3 bindings for regression models.
//!
//! ## REQ status
//!
//! The `sklearn.linear_model` `LinearRegression`/`Ridge`/`Lasso`/`ElasticNet`
//! binding shims: `#[pyclass]` `RsLinearRegression`/`RsRidge`/`RsLasso`/`RsElasticNet`
//! over `ferrolearn_linear::{LinearRegression,Ridge,Lasso,ElasticNet}`, wrapped by
//! the matching `ferrolearn.*` classes (`RegressorMixin`/`BaseEstimator`) in
//! `python/ferrolearn/_regressors.py`. This unit owns the sklearn-API marshalling
//! surface only (constructor ABI, attribute exposure, method surface, array
//! coercion); the regressor MATH lives downstream in `ferrolearn-linear`.
//! Verification model B: pytest comparing `import ferrolearn` against
//! `import sklearn` 1.5.2 (live oracle; `sklearn/linear_model/_base.py` +
//! `_ridge.py` + `_coordinate_descent.py`). Design doc:
//! `.design/python/regressors.md` (20 REQs). Every REQ is BINARY (R-DEFER-2):
//! SHIPPED or NOT-STARTED (with a concrete blocker). Verified via
//! `tests/divergence_regressors.py` + `tests/test_check_estimator.py` +
//! `tests/test_cross_val_score.py` (616 pytest pass).
//!
//! **23 SHIPPED / 2 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-LINREG-API-CONFORM (fit/predict + coef_/intercept_) | SHIPPED | `RsLinearRegression::fit`/`predict` + getters `coef_`/`intercept_`, wrapped by `_regressors.py::LinearRegression` (+ `n_features_in_`, `score` from `RegressorMixin`) — mirroring `_base.py:582`. Live default-path oracle matches element-wise. |
//! | REQ-LINREG-FIT-INTERCEPT-ABI (fit_intercept keyword-only) | SHIPPED | `LinearRegression.__init__(self, *, fit_intercept=True)` matches sklearn `_base.py:568` (the `*` is first). The only estimator whose constructor ABI already matched sklearn (Ridge/Lasso/ElasticNet diverged on `alpha`). |
//! | REQ-LINREG-VALUE-PARITY (coef_/intercept_ array parity) | SHIPPED | default path: marshalled from the Rust getters; downstream `ferrolearn-linear` REQ-1/5 match the sklearn oracle ≤1e-8 (full-rank/rank-deficient/underdetermined min-norm OLS). (`positive=True` NNLS #371; multi-output 2-D `y` → SHIPPED, see REQ-LINREG-MULTIOUTPUT.) |
//! | REQ-LINREG-MULTIOUTPUT (2-D Y → 2-D coef_) | SHIPPED | `ferrolearn.LinearRegression.fit` accepts a 2-D `y` of shape `(n_samples, n_targets)` and produces `coef_` `(n_targets, n_features)` + `intercept_` `(n_targets,)`, matching sklearn `LinearRegression` (`sklearn/linear_model/_base.py:687` `coef_.T`, `_set_intercept` `:319-327`). impl `#[pyclass(name="_RsLinearRegressionMultiOutput")] RsLinearRegressionMultiOutput::fit`/`predict` + getters `coef_`/`intercept_` in `regressors.rs` over `ferrolearn_linear::linear_regression::FittedMultiOutputLinearRegression<f64>` (the `Fit<Array2,Array2> for LinearRegression` path, `ferrolearn-linear/src/linear_regression.rs:491`, REQ-7/#372 SHIPPED; `coefficients()` is ALREADY `(n_targets,n_features)` `:458` + `intercepts()` `(n_targets,)` `:465`). UNLIKE the Ridge multi-output shim, the `coef_` getter does NOT transpose — the downstream storage is already in sklearn's `coef_` orientation. fit_intercept=False → scalar `0.0` (set in the wrapper, `_base.py:327`). After a pickle round-trip (`__getstate__` pops `_rs`), `predict` falls back to `_predict_linear(X, coef_, intercept_)` = `X @ coef_.T + intercept_`, matching sklearn's `_decision_function` (`_base.py:290`/`:364` `X @ coef_.T`); the `.T` is a no-op for 1-D single-output `coef_` and gives the correct `(n, n_targets)` orientation for 2-D multi-output `coef_` `(n_targets, n_features)` (#2125 — also repairs the shared Ridge pickle-predict path). Non-test consumer: `_regressors.py::LinearRegression.fit` routes the `y.ndim==2` path to `_RsLinearRegressionMultiOutput` (registered in `lib.rs`); `ferrolearn/__init__.py` re-export. Verification (model B): `tests/divergence_regressors.py::test_linearregression_multioutput_matches_sklearn` asserts `coef_` `(2,2)`, `intercept_` `(2,)`, and `predict` match the live sklearn oracle ≤1e-8 (R-CHAR-3); `_column_vector_y` ((n,1) → coef_ `(1,n_features)`), `_no_intercept_scalar` (fit_intercept=False → scalar 0.0), a single-output guard, and `test_red_linearregression_multioutput_pickle_predict_shape` (#2125; pickle round-trip predict matches the live sklearn oracle ≤1e-8). Downstream `ferrolearn-linear` REQ-7 #372. |
//! | REQ-LINREG-RANK-SINGULAR (rank_/singular_ fitted attrs, single AND multi-output) | SHIPPED | impl `RsLinearRegression::rank_` getter (over `FittedLinearRegression<f64>`, via `fitted.rank()`, `linear_regression.rs:471`) + `RsLinearRegression::singular_` getter (via `fitted.singular_values()`, `linear_regression.rs:478`, marshalled through `ndarray1_to_numpy`), surfaced by `_regressors.py::LinearRegression.fit` which sets `self.rank_ = int(self._rs.rank_)` + `self.singular_ = np.array(self._rs.singular_)` (next to `coef_`/`intercept_`). The MULTI-OUTPUT path is identical (#2124): `RsLinearRegressionMultiOutput::rank_`/`singular_` getters (over `FittedMultiOutputLinearRegression<f64>`, via `fitted.rank()`/`fitted.singular_values()`), surfaced in the `y.ndim==2` branch of `_regressors.py::LinearRegression.fit` (`self.rank_ = int(self._rs.rank_)` + `self.singular_ = np.array(self._rs.singular_)`). Mirrors sklearn `self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)` set REGARDLESS of single vs multi-output (`_base.py:687`; attr docstrings `rank_` `_base.py:505`, `singular_` `_base.py:508`). The downstream Rust captures both from the single-SVD `solve_lstsq` on the actually-solved (centered-when-`fit_intercept`) matrix, matching sklearn's `linalg.lstsq` operands (`ferrolearn-linear` REQ-9 SHIPPED #374). Non-test consumer: `_regressors.py::LinearRegression` + `ferrolearn/__init__.py` re-export. Verification (model B): `tests/divergence_regressors.py::test_linearregression_rank_singular_match_sklearn` (single-output, 5×2 fixture: `rank_=2`, `singular_=[4.24264069, 1.41421356]`) + `test_red_linearregression_multioutput_rank_singular_missing` (#2124; multi-output 8×3/8×2 fixture: asserts `fr.rank_ == sk.rank_` and `assert_allclose(fr.singular_, sk.singular_, atol=1e-8)`, live oracle on the DEFAULT `fit_intercept=True` (centered design) path `rank_=3`, `singular_=[0.9470684906780553, 0.7398714017099933, 0.45797423542006155]`) (R-CHAR-3). |
//! | REQ-LINREG-PARAMS (copy_X/n_jobs/positive) | NOT-STARTED (positive SHIPPED #2129) | `positive` IS now surfaced: `RsLinearRegression::new` takes `positive=false` (LAST, `#[pyo3(signature = (fit_intercept=true, positive=false))]`) + threads `.with_positive(self.positive)` into the `fit` builder; `_regressors.py::LinearRegression.__init__(self, *, fit_intercept=True, positive=False)` (keyword-only, matching sklearn `_base.py:574`) passes it to the single-output `_RsLinearRegression` and (when `positive=True` AND `y` is 2-D) to a PER-TARGET LOOP over the single-output positive binding (sklearn's `optimize.nnls` per-target, `_base.py:649-653`). Downstream NNLS `ferrolearn-linear` REQ-6 SHIPPED #371. Verification: `tests/divergence_regressors.py::test_linearregression_positive_matches_sklearn` (+ multi-output) + `test_linearregression_positive_false_unchanged`. Still MISSING `copy_X`/`n_jobs` — downstream #374. (`rank_`/`singular_` surfaced separately — see REQ-LINREG-RANK-SINGULAR SHIPPED.) |
//! | REQ-RIDGE-API-CONFORM (fit/predict + coef_/intercept_, default cholesky) | SHIPPED | `RsRidge::fit`/`predict` + getters, wrapped by `_regressors.py::Ridge` (marshals `alpha` via `with_alpha`) — mirroring `_ridge.py:914`. Live default-path oracle matches element-wise. |
//! | REQ-RIDGE-ALPHA-POSITIONAL (alpha positional ABI) | SHIPPED | FIXED #2040: `_regressors.py::Ridge.__init__(self, alpha=1.0, *, fit_intercept=True)` moves `alpha` before the `*`, so `ferrolearn.Ridge(0.5).alpha == 0.5` matching sklearn `_ridge.py:893`. Guard `test_red_ridge_alpha_positional`. |
//! | REQ-RIDGE-VALUE-PARITY (coef_/intercept_ array parity, default cholesky) | SHIPPED | default path: downstream `ferrolearn-linear` REQ-1/5 match the sklearn `Ridge` oracle ≤1e-8 across alpha∈{0.1,1,10,100} (closed-form Cholesky, unpenalized intercept). (Per-target alpha #385, multi-output #384, alt solvers downstream.) |
//! | REQ-RIDGE-MULTIOUTPUT (2-D Y → 2-D coef_) | SHIPPED | `ferrolearn.Ridge.fit` accepts a 2-D `y` of shape `(n_samples, n_targets)` and produces `coef_` `(n_targets, n_features)` + `intercept_` `(n_targets,)`, matching sklearn `Ridge` (`sklearn/linear_model/_ridge.py:543` coef shape, `:550` intercept shape, per-target solve `:207`/`:218`). impl `#[pyclass(name="_RsRidgeMultiOutput")] RsRidgeMultiOutput::fit`/`predict` + getters `coef_`/`intercept_` in `regressors.rs` over `ferrolearn_linear::FittedRidgeMulti<f64>` (the `Fit<Array2,Array2> for Ridge` path, `ferrolearn-linear/src/ridge.rs:725`, producing `FittedRidgeMulti` with `coefficients()` `(n_features,n_targets)` `:713` + `intercepts()` `:719`, shipped #29). The `coef_` getter TRANSPOSES the `(n_features,n_targets)` storage to sklearn's `(n_targets,n_features)` (R-DEV-3). Non-test consumer: `_regressors.py::Ridge.fit` routes the `y.ndim==2 and y.shape[1]>1` path to `_RsRidgeMultiOutput` (registered in `lib.rs`); `ferrolearn/__init__.py` re-export. Verification (model B): `tests/divergence_regressors.py::test_ridge_multioutput_matches_sklearn` asserts `coef_` `(2,2)`, `intercept_` `(2,)`, and `predict` match the live sklearn oracle ≤1e-8 (R-CHAR-3); `test_ridge_singleoutput_unchanged_by_multioutput_path` guards the 1-D path stays scalar-intercept. Downstream `ferrolearn-linear` REQ-6 #384. |
//! | REQ-RIDGE-PARAMS (copy_X/max_iter/tol/solver/positive/random_state) | SHIPPED (iterative solvers NOT-STARTED #2133) | ALL of sklearn's `Ridge.__init__` params are now surfaced (`_ridge.py:893-904`): `RsRidge::new` takes `#[pyo3(signature = (alpha=1.0, fit_intercept=true, copy_x=true, max_iter=None, tol=1e-4, solver="auto".to_string(), positive=false, random_state=None))]` (sklearn order) + threads `.with_copy_x(self.copy_x).with_max_iter(self.max_iter).with_tol(self.tol).with_solver(solver_enum).with_random_state(self.random_state).with_positive(self.positive)` into the `fit` builder. The `solver` string maps to `ferrolearn_linear::RidgeSolver` (`auto`→`Auto`, `cholesky`→`Cholesky`, `svd`→`Svd`); any other value (the iterative families `lsqr`/`sag`/`saga`/`sparse_cg`/`lbfgs`, `_ridge.py:885`) raises `PyNotImplementedError` (NOT-STARTED downstream, #2133). Wrapper `_regressors.py::Ridge.__init__(self, alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=1e-4, solver="auto", positive=False, random_state=None)` (sklearn order/defaults, `_ridge.py:893-904`) stores each (`self.copy_X` capital-X) and threads them into the single-output `_RsRidge`; the multi-output direct-solver path (`_RsRidgeMultiOutput`) leaves solver/tol/max_iter/random_state as no-ops (closed-form per-target Cholesky) but `get_params()` returns all 8 for sklearn parity. Downstream `ferrolearn-linear` REQ-8a (solver auto/cholesky/svd, all dense solvers give the identical strictly-convex coef) #386, REQ-10 (max_iter/tol no-ops for the direct solver) #388, REQ-12 (copy_x ABI-only, random_state stored-but-no-op) #390 all SHIPPED. Verification: `tests/divergence_regressors.py::{test_ridge_solver_svd_cholesky_auto_match_oracle, test_ridge_copy_x_max_iter_tol_random_state_accepted, test_ridge_get_params_all_eight_and_clone, test_ridge_unsupported_solver_raises}` (live oracle, R-CHAR-3). Iterative solvers `lsqr`/`sag`/`saga`/`sparse_cg`/`lbfgs` remain NOT-STARTED — #2133. |
//! | REQ-LASSO-API-CONFORM (fit/predict + coef_/intercept_, default cyclic) | SHIPPED | `RsLasso::fit`/`predict` + getters, wrapped by `_regressors.py::Lasso` (marshals `alpha`/`max_iter`/`tol`) — mirroring `_coordinate_descent.py:932`. Live default-path coef matches the downstream-verified converged tolerance. |
//! | REQ-LASSO-ALPHA-POSITIONAL (alpha positional ABI) | SHIPPED | FIXED #2041: `_regressors.py::Lasso.__init__(self, alpha=1.0, *, ...)` moves `alpha` before the `*`, so `ferrolearn.Lasso(0.1).alpha == 0.1` matching sklearn `_coordinate_descent.py:1310`. Guard `test_red_lasso_alpha_positional`. |
//! | REQ-LASSO-VALUE-PARITY (coef_/intercept_ + support set, default cyclic) | SHIPPED | default converged path: downstream `ferrolearn-linear` REQ-1/4/6 match the sklearn `Lasso` oracle ≤1e-6 for converged coef/intercept + exact-zero support (cyclic CD, `l1_reg=α·n`). (Dual-gap stopping #412, positive/selection='random' downstream.) |
//! | REQ-LASSO-NITER (n_iter_ = real CD count) | SHIPPED | impl `RsLasso::n_iter_` getter in `regressors.rs` (over `FittedLasso<f64>`, via `fitted.n_iter()`), surfaced by `_regressors.py::Lasso.fit` which now sets `self.n_iter_ = int(self._rs.n_iter_)` (was the faked `self.max_iter`/1000). Mirrors sklearn's ACTUAL CD count `self.n_iter_.append(this_iter[0])` (`_coordinate_descent.py:1103`, single-target collapse `:1106`). The downstream Rust `FittedLasso::n_iter()` is bit-faithful to sklearn via the dual-gap CD stopping criterion (`ferrolearn-linear` REQ-11/12 SHIPPED, #411 closed). Non-test consumer: `_regressors.py::Lasso` + `ferrolearn/__init__.py` re-export. Verification (model B): `tests/divergence_regressors.py::test_lasso_elasticnet_n_iter_matches_sklearn` asserts `fl.Lasso(alpha=0.5).fit(X,y).n_iter_ == SkLasso(alpha=0.5).fit(X,y).n_iter_` (live oracle, n_iter_==2 on the fixture, < max_iter — no longer faked). |
//! | REQ-LASSO-DUALGAP (dual_gap_ fitted attribute) | SHIPPED | impl `RsLasso::dual_gap_` getter in `regressors.rs` (over `FittedLasso<f64>`, via `fitted.dual_gap()`), surfaced by `_regressors.py::Lasso.fit` which sets `self.dual_gap_ = float(self._rs.dual_gap_)` (next to `n_iter_`). Mirrors sklearn `self.dual_gap_ = dual_gaps_[0]` (`_coordinate_descent.py:1108`). Downstream `FittedLasso::dual_gap()` matches sklearn EXACTLY (`ferrolearn-linear` REQ-11). Non-test consumer: `_regressors.py::Lasso` + re-export. Verification (model B): `tests/divergence_regressors.py::test_lasso_elasticnet_dual_gap_matches_sklearn` (`abs(fl.Lasso(0.3).dual_gap_ - SkLasso(0.3).dual_gap_) < 1e-9`, live oracle). |
//! | REQ-LASSO-PARAMS (precompute/copy_X/warm_start/positive/random_state/selection) | SHIPPED (selection='random' bit-parity NOT-STARTED — numpy-MT RNG) | ALL of sklearn's `Lasso.__init__` params are now surfaced (`_coordinate_descent.py:1310-1322`): `RsLasso::new` carries `#[pyo3(signature = (alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=true, positive=false, precompute=false, copy_x=true, warm_start=false, random_state=None, selection="cyclic".to_string(), coef_init=None))]` + `fit` threads `.with_precompute(self.precompute).with_selection(coord_selection_from_str(&self.selection)?).with_warm_start(self.warm_start)`, calls `.with_random_state(seed)` only when `random_state` is `Some` (the Rust builder takes `u64`; `_coordinate_descent.py:803`), and `.with_coef_init(Array1::from(init))` when `warm_start && coef_init.is_some()` (R-DEV-4: ferrolearn estimators are immutable value types, so the prior fit's `coef_` is passed back explicitly rather than read off a mutated `self.coef_`, sklearn `_coordinate_descent.py:1062`). `selection` maps via `fn coord_selection_from_str` (`cyclic`→`Cyclic`, `random`→`Random`, any other → `ValueError`, mirroring sklearn `_parameter_constraints["selection"]=[StrOptions({"cyclic","random"})]` `_coordinate_descent.py:893`). `copy_X` is ABI-only (`#[allow(dead_code, reason=...)]` field — CD fit never mutates `X`, `_coordinate_descent.py:1314`); stored for `get_params`/`set_params` round-trip on the wrapper. Wrapper `_regressors.py::Lasso.__init__(self, alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=1e-4, warm_start=False, positive=False, random_state=None, selection='cyclic')` (sklearn order/defaults, `_coordinate_descent.py:1310-1322`) stores each and threads them into the single-output `_RsLasso` AND each per-target `_RsLasso` in the multi-output loop; warm_start is single-output-scoped (reuses the prior `self.coef_` as `coef_init` on a 1-D refit; multi-output warm_start is NOT-STARTED — stored for get_params but the per-target loop ignores it). Downstream `ferrolearn-linear` Lasso REQ-8 (warm_start) / REQ-9 (selection+random_state) / REQ-10 (precompute) all SHIPPED. `selection='random'` converges to the unique Lasso optimum but is NOT bit-identical to sklearn (Rust `StdRng` ≠ numpy MT19937 — bit-parity NOT-STARTED). Verification (live oracle, R-CHAR-3): `tests/divergence_regressors.py::{test_lasso_precompute_matches_sklearn, test_lasso_selection_cyclic_default_unchanged, test_lasso_selection_random_converges_to_optimum, test_lasso_warm_start_refit_fewer_iters, test_lasso_copy_x_accepted_default_unchanged, test_lasso_get_params_all_and_clone, test_lasso_invalid_selection_raises}`. |
//! | REQ-ELASTICNET-API-CONFORM (fit/predict + coef_/intercept_, default cyclic) | SHIPPED | `RsElasticNet::fit`/`predict` + getters, wrapped by `_regressors.py::ElasticNet` (marshals `alpha`/`l1_ratio`/`max_iter`/`tol`) — mirroring `_coordinate_descent.py:932`. Live default-path coef matches the downstream-verified converged tolerance. |
//! | REQ-ELASTICNET-ALPHA-POSITIONAL (alpha positional ABI) | SHIPPED | FIXED #2042: `_regressors.py::ElasticNet.__init__(self, alpha=1.0, *, l1_ratio=0.5, ...)` moves ONLY `alpha` before the `*` (l1_ratio stays keyword-only), so `ferrolearn.ElasticNet(0.1).alpha == 0.1` matching sklearn `_coordinate_descent.py:898`. Guard `test_red_elasticnet_alpha_positional`. |
//! | REQ-ELASTICNET-VALUE-PARITY (coef_/intercept_ + support set, default cyclic) | SHIPPED | default converged path: downstream `ferrolearn-linear` REQ-1/4/5 match the sklearn `ElasticNet` oracle <1e-5 over the (alpha,l1_ratio) grid (`l1_reg=α·l1_ratio·n`, `l2_reg=α·(1−l1_ratio)·n`), incl. l1_ratio=1↔Lasso / 0↔L2. (Dual-gap #412 downstream.) |
//! | REQ-ELASTICNET-NITER (n_iter_ = real CD count) | SHIPPED | impl `RsElasticNet::n_iter_` getter in `regressors.rs` (over `FittedElasticNet<f64>`, via `fitted.n_iter()`), surfaced by `_regressors.py::ElasticNet.fit` which now sets `self.n_iter_ = int(self._rs.n_iter_)` (was the faked `self.max_iter`/1000). Mirrors sklearn's ACTUAL CD count `self.n_iter_.append(this_iter[0])` (`_coordinate_descent.py:1103`, single-target collapse `:1106`). The downstream Rust `FittedElasticNet::n_iter()` is bit-faithful to sklearn via the dual-gap CD stopping criterion (`ferrolearn-linear` REQ-12/13 SHIPPED, #417 closed). Non-test consumer: `_regressors.py::ElasticNet` + `ferrolearn/__init__.py` re-export. Verification (model B): `tests/divergence_regressors.py::test_lasso_elasticnet_n_iter_matches_sklearn` asserts `fl.ElasticNet(alpha=0.5).fit(X,y).n_iter_ == SkElasticNet(alpha=0.5).fit(X,y).n_iter_` (live oracle, n_iter_==2 on the fixture, < max_iter — no longer faked). |
//! | REQ-ELASTICNET-DUALGAP (dual_gap_ fitted attribute) | SHIPPED | impl `RsElasticNet::dual_gap_` getter in `regressors.rs` (over `FittedElasticNet<f64>`, via `fitted.dual_gap()`), surfaced by `_regressors.py::ElasticNet.fit` which sets `self.dual_gap_ = float(self._rs.dual_gap_)` (next to `n_iter_`). Mirrors sklearn `self.dual_gap_ = dual_gaps_[0]` (`_coordinate_descent.py:1108`). Downstream `FittedElasticNet::dual_gap()` matches sklearn EXACTLY (`ferrolearn-linear` REQ-12). Non-test consumer: `_regressors.py::ElasticNet` + re-export. Verification (model B): `tests/divergence_regressors.py::test_lasso_elasticnet_dual_gap_matches_sklearn` (`abs(fl.ElasticNet(0.3).dual_gap_ - SkElasticNet(0.3).dual_gap_) < 1e-9`, live oracle, default l1_ratio=0.5). |
//! | REQ-ELASTICNET-PARAMS (precompute/copy_X/warm_start/positive/random_state/selection) | SHIPPED (selection='random' bit-parity NOT-STARTED — numpy-MT RNG) | ALL of sklearn's `ElasticNet.__init__` params are now surfaced (`_coordinate_descent.py:898-912`): `RsElasticNet::new` carries `#[pyo3(signature = (alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=true, positive=false, precompute=false, copy_x=true, warm_start=false, random_state=None, selection="cyclic".to_string(), coef_init=None))]` + `fit` threads `.with_precompute(self.precompute).with_selection(coord_selection_from_str(&self.selection)?).with_warm_start(self.warm_start)`, calls `.with_random_state(seed)` only when `random_state` is `Some` (Rust builder takes `u64`; `_coordinate_descent.py:803`), and `.with_coef_init(Array1::from(init))` when `warm_start && coef_init.is_some()` (R-DEV-4 immutable-value-type adaptation of sklearn's reused `self.coef_`, `_coordinate_descent.py:1062`). `selection` maps via `fn coord_selection_from_str` (`cyclic`→`Cyclic`, `random`→`Random`, any other → `ValueError`, mirroring `_parameter_constraints["selection"]=[StrOptions({"cyclic","random"})]` `_coordinate_descent.py:893`). `copy_X` is ABI-only (`#[allow(dead_code, reason=...)]` field — CD fit never mutates `X`, `_coordinate_descent.py:906`); stored for `get_params`/`set_params` round-trip on the wrapper. Wrapper `_regressors.py::ElasticNet.__init__(self, alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, precompute=False, max_iter=1000, copy_X=True, tol=1e-4, warm_start=False, positive=False, random_state=None, selection='cyclic')` (sklearn order/defaults, `_coordinate_descent.py:898-912`) stores each and threads them into the single-output `_RsElasticNet` AND each per-target `_RsElasticNet` in the multi-output loop; warm_start is single-output-scoped (reuses the prior 1-D `self.coef_` as `coef_init`; multi-output warm_start NOT-STARTED — stored for get_params, ignored in the per-target loop). Downstream `ferrolearn-linear` ElasticNet REQ-9 (warm_start) / REQ-10 (selection+random_state) / REQ-11 (precompute) all SHIPPED. `selection='random'` converges to the unique ElasticNet optimum but is NOT bit-identical to sklearn (Rust `StdRng` ≠ numpy MT19937 — bit-parity NOT-STARTED). Verification (live oracle, R-CHAR-3): `tests/divergence_regressors.py::{test_elasticnet_precompute_matches_sklearn, test_elasticnet_selection_cyclic_default_unchanged, test_elasticnet_selection_random_converges_to_optimum, test_elasticnet_warm_start_refit_fewer_iters, test_elasticnet_copy_x_accepted_default_unchanged, test_elasticnet_get_params_all_and_clone, test_elasticnet_invalid_selection_raises}`. |
//! | REQ-CONSUMER (binding IS the public API) | SHIPPED | non-test consumers: `_regressors.py::{LinearRegression,Ridge,Lasso,ElasticNet}` construct their `_Rs*` class and drive fit/predict + coef_/intercept_ reads; `ferrolearn/__init__.py` re-exports all four; `test_check_estimator.py` + `test_cross_val_score.py` exercise them (542 pytest pass). |
//! | REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | marshals via `crate::conversions::*` (rust-numpy + `ndarray`), not `ferray::numpy_interop`/`ferray-core` (R-SUBSTRATE-1); ferray exposes no numpy bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027. |

use crate::conversions::*;
use ferrolearn_core::{Fit, HasCoefficients, Predict};
use ferrolearn_linear::lasso::CoordSelection;
use ferrolearn_linear::ridge::RidgeSolver;
use ndarray::Array1;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Map the sklearn `selection` string to the Rust [`CoordSelection`] enum.
///
/// Mirrors sklearn's `_parameter_constraints["selection"] =
/// [StrOptions({"cyclic", "random"})]` (`_coordinate_descent.py:893` for
/// `ElasticNet`, inherited by `Lasso`): only `'cyclic'` / `'random'` are valid;
/// any other value raises `ValueError` (the binding analog of sklearn's
/// `InvalidParameterError`, a `ValueError` subclass).
fn coord_selection_from_str(selection: &str) -> PyResult<CoordSelection> {
    match selection {
        "cyclic" => Ok(CoordSelection::Cyclic),
        "random" => Ok(CoordSelection::Random),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "The 'selection' parameter must be a str among {{'cyclic', 'random'}}. \
             Got {other:?} instead."
        ))),
    }
}

// ---------------------------------------------------------------------------
// LinearRegression
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsLinearRegression")]
pub struct RsLinearRegression {
    fit_intercept: bool,
    positive: bool,
    fitted: Option<ferrolearn_linear::FittedLinearRegression<f64>>,
}

#[pymethods]
impl RsLinearRegression {
    #[new]
    #[pyo3(signature = (fit_intercept=true, positive=false))]
    fn new(fit_intercept: bool, positive: bool) -> Self {
        Self {
            fit_intercept,
            positive,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::LinearRegression::<f64>::new()
            .with_fit_intercept(self.fit_intercept)
            .with_positive(self.positive);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }

    #[getter]
    fn rank_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.rank())
    }

    #[getter]
    fn singular_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.singular_values()))
    }
}

// ---------------------------------------------------------------------------
// LinearRegression (multi-output 2-D Y)
// ---------------------------------------------------------------------------

/// Multi-output OLS binding shim over
/// [`ferrolearn_linear::FittedMultiOutputLinearRegression`] (the
/// `Fit<Array2, Array2>` path).
///
/// The Python wrapper `_regressors.py::LinearRegression.fit` routes to this
/// class whenever `y` is 2-D, mirroring sklearn's
/// `LinearRegression.coef_` shape `(n_targets, n_features)` / `intercept_`
/// `(n_targets,)` (`sklearn/linear_model/_base.py:687` `coef_.T`,
/// `_set_intercept` `:319-327`). UNLIKE the Ridge multi-output shim, the
/// downstream `FittedMultiOutputLinearRegression::coefficients()` is ALREADY in
/// sklearn's `(n_targets, n_features)` orientation
/// (`ferrolearn-linear/src/linear_regression.rs:458`), so `coef_` is marshalled
/// straight through with NO transpose.
#[pyclass(name = "_RsLinearRegressionMultiOutput")]
pub struct RsLinearRegressionMultiOutput {
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::linear_regression::FittedMultiOutputLinearRegression<f64>>,
}

#[pymethods]
impl RsLinearRegressionMultiOutput {
    #[new]
    #[pyo3(signature = (fit_intercept=true))]
    fn new(fit_intercept: bool) -> Self {
        Self {
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy2_to_ndarray(y);
        let model = ferrolearn_linear::LinearRegression::<f64>::new()
            .with_fit_intercept(self.fit_intercept);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &preds))
    }

    /// Coefficient matrix, shape `(n_targets, n_features)` — already in the
    /// sklearn `coef_` orientation downstream, so NO transpose (unlike the Ridge
    /// multi-output shim).
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.coefficients()))
    }

    #[getter]
    fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.intercepts()))
    }

    /// Effective rank of the (centered-when-`fit_intercept`) design matrix from
    /// the single-SVD lstsq solve — set by sklearn REGARDLESS of single vs
    /// multi-output (`sklearn/linear_model/_base.py:687`,
    /// `self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)`; attr
    /// docstring `_base.py:505`). Mirrors the single-output `RsLinearRegression`
    /// `rank_` getter.
    #[getter]
    fn rank_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.rank())
    }

    /// Singular values of the (centered-when-`fit_intercept`) design matrix from
    /// the single-SVD lstsq solve — set by sklearn REGARDLESS of single vs
    /// multi-output (`sklearn/linear_model/_base.py:687`; attr docstring
    /// `_base.py:508`). Mirrors the single-output `RsLinearRegression`
    /// `singular_` getter.
    #[getter]
    fn singular_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.singular_values()))
    }
}

// ---------------------------------------------------------------------------
// Ridge
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsRidge")]
pub struct RsRidge {
    alpha: f64,
    fit_intercept: bool,
    copy_x: bool,
    max_iter: Option<usize>,
    tol: f64,
    solver: String,
    positive: bool,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_linear::FittedRidge<f64>>,
}

#[pymethods]
impl RsRidge {
    #[new]
    #[pyo3(signature = (
        alpha = 1.0,
        fit_intercept = true,
        copy_x = true,
        max_iter = None,
        tol = 1e-4,
        solver = "auto".to_string(),
        positive = false,
        random_state = None
    ))]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors sklearn Ridge.__init__ constructor ABI (alpha, fit_intercept, copy_X, max_iter, tol, solver, positive, random_state); _ridge.py:893-904"
    )]
    fn new(
        alpha: f64,
        fit_intercept: bool,
        copy_x: bool,
        max_iter: Option<usize>,
        tol: f64,
        solver: String,
        positive: bool,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            alpha,
            fit_intercept,
            copy_x,
            max_iter,
            tol,
            solver,
            positive,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        // Map the sklearn `solver` string to the Rust `RidgeSolver` enum. Only
        // the dense direct solvers are implemented downstream; the iterative
        // families ('lsqr'/'sag'/'saga'/'sparse_cg'/'lbfgs', sklearn
        // `_ridge.py:885`) are NOT-STARTED (#2133) and raise NotImplementedError.
        let solver_enum = match self.solver.as_str() {
            "auto" => RidgeSolver::Auto,
            "cholesky" => RidgeSolver::Cholesky,
            "svd" => RidgeSolver::Svd,
            other => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "Ridge solver '{other}' is not supported (only 'auto'/'svd'/'cholesky'; \
                     iterative solvers lsqr/sag/saga/sparse_cg/lbfgs are NOT-STARTED, see #2133)"
                )));
            }
        };
        let model = ferrolearn_linear::Ridge::<f64>::new()
            .with_alpha(self.alpha)
            .with_fit_intercept(self.fit_intercept)
            .with_copy_x(self.copy_x)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_solver(solver_enum)
            .with_random_state(self.random_state)
            .with_positive(self.positive);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }
}

// ---------------------------------------------------------------------------
// Ridge (multi-output 2-D Y)
// ---------------------------------------------------------------------------

/// Multi-output Ridge binding shim over
/// [`ferrolearn_linear::FittedRidgeMulti`] (the `Fit<Array2, Array2>` path).
///
/// The Python wrapper `_regressors.py::Ridge.fit` routes to this class when
/// `y.ndim == 2 and y.shape[1] > 1`, mirroring sklearn's
/// `Ridge.coef_` shape `(n_targets, n_features)` / `intercept_` `(n_targets,)`
/// (`sklearn/linear_model/_ridge.py:914`). The downstream Rust stores
/// `coefficients()` as `(n_features, n_targets)`, so `coef_` is TRANSPOSED here
/// to match the sklearn output contract (R-DEV-3).
#[pyclass(name = "_RsRidgeMultiOutput")]
pub struct RsRidgeMultiOutput {
    alpha: f64,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedRidgeMulti<f64>>,
}

#[pymethods]
impl RsRidgeMultiOutput {
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_intercept=true))]
    fn new(alpha: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy2_to_ndarray(y);
        let model = ferrolearn_linear::Ridge::<f64>::new()
            .with_alpha(self.alpha)
            .with_fit_intercept(self.fit_intercept);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &preds))
    }

    /// Coefficient matrix, shape `(n_targets, n_features)` — the TRANSPOSE of the
    /// downstream `(n_features, n_targets)` storage, matching sklearn's `coef_`.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        // Rust stores (n_features, n_targets); sklearn `coef_` is
        // (n_targets, n_features). `.t()` gives a view; `.to_owned()` makes a
        // contiguous owned array for marshalling.
        let coef_t = fitted.coefficients().t().to_owned();
        Ok(ndarray2_to_numpy(py, &coef_t))
    }

    #[getter]
    fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.intercepts()))
    }
}

// ---------------------------------------------------------------------------
// Lasso
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsLasso")]
pub struct RsLasso {
    alpha: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    positive: bool,
    precompute: bool,
    // ABI-only (sklearn `Lasso(copy_X=...)`, `_coordinate_descent.py:1314`): the
    // coordinate-descent fit never mutates `X` in place, so there is no Rust
    // builder to thread it to. Stored on the wrapper for `get_params`/`set_params`
    // round-trip; held here too so the binding mirrors the full sklearn ctor ABI.
    #[allow(
        dead_code,
        reason = "copy_X is ABI-only — CD fit never mutates X, so there is no Rust builder; stored for sklearn constructor-ABI parity (_coordinate_descent.py:1314)"
    )]
    copy_x: bool,
    warm_start: bool,
    random_state: Option<u64>,
    selection: String,
    coef_init: Option<Vec<f64>>,
    fitted: Option<ferrolearn_linear::FittedLasso<f64>>,
}

#[pymethods]
impl RsLasso {
    #[new]
    #[pyo3(signature = (
        alpha = 1.0,
        max_iter = 1000,
        tol = 1e-4,
        fit_intercept = true,
        positive = false,
        precompute = false,
        copy_x = true,
        warm_start = false,
        random_state = None,
        selection = "cyclic".to_string(),
        coef_init = None
    ))]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors sklearn Lasso.__init__ constructor ABI (alpha, fit_intercept, precompute, copy_X, max_iter, tol, warm_start, positive, random_state, selection) plus the R-DEV-4 coef_init warm-start seed; _coordinate_descent.py:1310-1322"
    )]
    fn new(
        alpha: f64,
        max_iter: usize,
        tol: f64,
        fit_intercept: bool,
        positive: bool,
        precompute: bool,
        copy_x: bool,
        warm_start: bool,
        random_state: Option<u64>,
        selection: String,
        coef_init: Option<Vec<f64>>,
    ) -> Self {
        Self {
            alpha,
            max_iter,
            tol,
            fit_intercept,
            positive,
            precompute,
            copy_x,
            warm_start,
            random_state,
            selection,
            coef_init,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let selection_enum = coord_selection_from_str(&self.selection)?;
        let mut model = ferrolearn_linear::Lasso::<f64>::new()
            .with_alpha(self.alpha)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_fit_intercept(self.fit_intercept)
            .with_positive(self.positive)
            .with_precompute(self.precompute)
            .with_selection(selection_enum)
            .with_warm_start(self.warm_start);
        // `random_state` is `Option<u64>` here (sklearn default `None`); the Rust
        // builder takes a `u64`, so only call it when a seed was supplied,
        // otherwise leave the estimator's own default (`_coordinate_descent.py:803`,
        // only used when `selection == 'random'`).
        if let Some(seed) = self.random_state {
            model = model.with_random_state(seed);
        }
        // warm_start (R-DEV-4): ferrolearn estimators are immutable value types,
        // so the prior fit's coefficients are passed back in explicitly as
        // `coef_init` (sklearn reuses the mutable `self.coef_`,
        // `_coordinate_descent.py:1062`). Only seed when both `warm_start` is set
        // AND a prior coefficient vector was supplied by the wrapper.
        if self.warm_start
            && let Some(ref init) = self.coef_init
        {
            model = model.with_coef_init(Array1::from(init.clone()));
        }
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }

    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_iter())
    }

    #[getter]
    fn dual_gap_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.dual_gap())
    }
}

// ---------------------------------------------------------------------------
// ElasticNet
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsElasticNet")]
pub struct RsElasticNet {
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    positive: bool,
    precompute: bool,
    // ABI-only (sklearn `ElasticNet(copy_X=...)`, `_coordinate_descent.py:906`):
    // the coordinate-descent fit never mutates `X` in place, so there is no Rust
    // builder to thread it to. Stored on the wrapper for `get_params`/`set_params`
    // round-trip; held here too so the binding mirrors the full sklearn ctor ABI.
    #[allow(
        dead_code,
        reason = "copy_X is ABI-only — CD fit never mutates X, so there is no Rust builder; stored for sklearn constructor-ABI parity (_coordinate_descent.py:906)"
    )]
    copy_x: bool,
    warm_start: bool,
    random_state: Option<u64>,
    selection: String,
    coef_init: Option<Vec<f64>>,
    fitted: Option<ferrolearn_linear::FittedElasticNet<f64>>,
}

#[pymethods]
impl RsElasticNet {
    #[new]
    #[pyo3(signature = (
        alpha = 1.0,
        l1_ratio = 0.5,
        max_iter = 1000,
        tol = 1e-4,
        fit_intercept = true,
        positive = false,
        precompute = false,
        copy_x = true,
        warm_start = false,
        random_state = None,
        selection = "cyclic".to_string(),
        coef_init = None
    ))]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors sklearn ElasticNet.__init__ constructor ABI (alpha, l1_ratio, fit_intercept, precompute, max_iter, copy_X, tol, warm_start, positive, random_state, selection) plus the R-DEV-4 coef_init warm-start seed; _coordinate_descent.py:898-912"
    )]
    fn new(
        alpha: f64,
        l1_ratio: f64,
        max_iter: usize,
        tol: f64,
        fit_intercept: bool,
        positive: bool,
        precompute: bool,
        copy_x: bool,
        warm_start: bool,
        random_state: Option<u64>,
        selection: String,
        coef_init: Option<Vec<f64>>,
    ) -> Self {
        Self {
            alpha,
            l1_ratio,
            max_iter,
            tol,
            fit_intercept,
            positive,
            precompute,
            copy_x,
            warm_start,
            random_state,
            selection,
            coef_init,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let selection_enum = coord_selection_from_str(&self.selection)?;
        let mut model = ferrolearn_linear::ElasticNet::<f64>::new()
            .with_alpha(self.alpha)
            .with_l1_ratio(self.l1_ratio)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_fit_intercept(self.fit_intercept)
            .with_positive(self.positive)
            .with_precompute(self.precompute)
            .with_selection(selection_enum)
            .with_warm_start(self.warm_start);
        // `random_state` is `Option<u64>` (sklearn default `None`); the Rust
        // builder takes a `u64`, so only call it when a seed was supplied
        // (`_coordinate_descent.py:803`, used only when `selection == 'random'`).
        if let Some(seed) = self.random_state {
            model = model.with_random_state(seed);
        }
        // warm_start (R-DEV-4): the prior fit's coefficients are passed back in
        // explicitly as `coef_init` (sklearn reuses the mutable `self.coef_`,
        // `_coordinate_descent.py:1062`).
        if self.warm_start
            && let Some(ref init) = self.coef_init
        {
            model = model.with_coef_init(Array1::from(init.clone()));
        }
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }

    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_iter())
    }

    #[getter]
    fn dual_gap_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.dual_gap())
    }
}
