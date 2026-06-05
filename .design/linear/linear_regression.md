# LinearRegression — Ordinary Least Squares

<!--
tier: 3-component
status: draft
baseline-commit: 35992d9
upstream-paths:
  - sklearn/linear_model/_base.py
-->

## Summary

This module mirrors scikit-learn's `LinearRegression` (`sklearn/linear_model/_base.py:465`), an
ordinary-least-squares regressor. `LinearRegression<F>` holds the `fit_intercept` hyperparameter;
`Fit` produces a `FittedLinearRegression<F>` carrying `coefficients` (sklearn `coef_`) and
`intercept` (sklearn `intercept_`). It fits via the centering trick + normal equations (Cholesky)
with a QR least-squares fallback, rather than sklearn's LAPACK `gelsd` (SVD) path — see
[Architecture](#architecture) for the resulting numerical-method divergence on rank-deficient `X`.

## Requirements

- REQ-1: OLS fit produces `coef_` and `intercept_` minimizing `||X·w − y||²`, with the intercept
  recovered from centered data when `fit_intercept=true` (sklearn `LinearRegression.fit`,
  `_base.py:582`; intercept set in `_set_intercept`, `_base.py:308`).
- REQ-2: `predict(X)` returns `X·coef + intercept` (sklearn `LinearModel._decision_function`,
  `_base.py:282`; `predict`, `_base.py:292`).
- REQ-3: The `fit_intercept` parameter (default `True`) is honored, including `fit_intercept=false`,
  which forces `intercept_ = 0` (sklearn ctor `_base.py:571`; `intercept_` semantics `_base.py:511`).
- REQ-4: Fitted coefficients and intercept are introspectable as `coef_`/`intercept_` analogs
  (sklearn fitted attributes `_base.py:499`, `_base.py:511`).
- REQ-5: `positive=True` constrains coefficients to be non-negative via NNLS (sklearn ctor
  `_base.py:574`; `optimize.nnls` path `_base.py:645`).
- REQ-6: Multi-output `Y` of shape `(n_samples, n_targets)` yields a 2-D `coef_` of shape
  `(n_targets, n_features)` (sklearn `MultiOutputMixin`, `_base.py:465`; `coef_` shape `_base.py:499`;
  ravel-to-1D guard `_base.py:690`).
- REQ-7: `fit(X, y, sample_weight=None)` accepts per-sample weights, rescaling rows by `√w`
  (sklearn `fit` signature `_base.py:582`; `_rescale_data` `_base.py:641`).
- REQ-8: Fitted attributes `rank_` (rank of `X`) and `singular_` (singular values), plus the
  `copy_X` and `n_jobs` constructor parameters, match sklearn's surface (sklearn `_parameter_constraints`
  `_base.py:561`; `rank_`/`singular_` set from `linalg.lstsq` `_base.py:687`; attr docs `_base.py:505`).
- REQ-substrate: The internal solve and coefficient storage run on the ferray substrate
  (`ferray::linalg` / `ferray-core`) rather than `faer` + `ndarray::Array1` (goal.md R-SUBSTRATE-1).

## Acceptance criteria

- AC-1: On a well-conditioned full-rank `X`, `coef_`/`intercept_` match sklearn's
  `LinearRegression().fit(X, y)` to within `1e-8` (oracle `X=[[1,1],[1,2],[2,2],[2,3]]`,
  `y = X·[1,2] + 3` → `coef_=[1,2]`, `intercept_≈3`).
- AC-2: `predict(X)` reproduces `X @ coef_ + intercept_` elementwise (closed-form, no tolerance
  beyond float rounding).
- AC-3: With `fit_intercept=false`, `intercept_` is exactly `0` and `coef_` matches sklearn
  `LinearRegression(fit_intercept=False)` (oracle `y=2x` through origin → `coef_=[2]`, `intercept_=0`).
- AC-4: `coefficients()` and `intercept()` return the fitted values via `HasCoefficients`.
- AC-5 (REQ-5): `positive=True` returns all-non-negative `coef_` matching `scipy.optimize.nnls`.
- AC-6 (REQ-6): 2-D `Y` returns `coef_.shape == (n_targets, n_features)`.
- AC-7 (REQ-7): `sample_weight` reproduces sklearn weighted OLS.
- AC-8 (REQ-8): `rank_`/`singular_` match `scipy.linalg.lstsq(X, y)[2:4]`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (OLS fit) | SHIPPED | impl `Fit::fit for LinearRegression in linear_regression.rs` (centering trick: `x_centered = x - x_mean`, `solve_normal_equations` → `solve_lstsq` fallback, `intercept = y_mean - x_mean.dot(&w)`) mirrors sklearn `_base.py:629` (`_preprocess_data` centering) + `_base.py:308` `_set_intercept` (`intercept_ = y_offset - X_offset @ coef_`, `_base.py:320`). Non-test consumer: `RsLinearRegression::fit in ferrolearn-python/src/regressors.rs` (`model.fit(&x_nd, &y_nd)`). Verification: oracle `coef_=[1.0, 2.0]`, `intercept_≈3.0` vs `test_multiple_linear_regression` (`coef[0]=1`, `coef[1]=2`, `intercept=3`). |
| REQ-2 (predict) | SHIPPED | impl `Predict::predict for FittedLinearRegression in linear_regression.rs` (`x.dot(&self.coefficients) + self.intercept`) mirrors sklearn `_base.py:288` (`return X @ coef_ + self.intercept_`). Non-test consumer: `RsLinearRegression::predict in ferrolearn-python/src/regressors.rs` (`fitted.predict(&x_nd)`). Verification: `cargo test -p ferrolearn-linear` — `test_simple_linear_regression` asserts preds match `y` to `1e-10`. |
| REQ-3 (fit_intercept incl. False) | SHIPPED | impl `LinearRegression::with_fit_intercept` + the `else` branch of `Fit::fit in linear_regression.rs` (`intercept: F::zero()`) mirrors sklearn ctor default `fit_intercept=True` (`_base.py:571`) and `intercept_ = 0.0 if fit_intercept = False` (`_base.py:512`). Non-test consumer: `RsLinearRegression::new` / `fit in ferrolearn-python/src/regressors.rs` (`.with_fit_intercept(self.fit_intercept)`). Verification: `test_no_intercept` asserts `intercept = 0` for `y = 2x` (oracle `coef_=[2.0]`, `intercept_=0.0`). |
| REQ-4 (HasCoefficients introspection) | SHIPPED | impl `HasCoefficients for FittedLinearRegression in linear_regression.rs` (`coefficients()`/`intercept()`) maps to sklearn fitted attrs `coef_` (`_base.py:499`) and `intercept_` (`_base.py:511`). Non-test consumer: `RsLinearRegression::coef_`/`intercept_` getters `in ferrolearn-python/src/regressors.rs` (`fitted.coefficients()`, `fitted.intercept()`). Verification: `test_has_coefficients` (`cargo test -p ferrolearn-linear`). |
| REQ-5 (positive / NNLS) | NOT-STARTED | open prereq blocker #371. `LinearRegression<F>` has no `positive` field; the `Fit` impl never branches to a non-negative least-squares solve, so the `optimize.nnls` path (`_base.py:645`) is unimplemented. |
| REQ-6 (multi-output 2-D Y) | SHIPPED | #372: additive `impl Fit<Array2<F>, Array2<F>> for LinearRegression` (the 1-D `Fit<Array2<F>, Array1<F>>` arm is untouched) producing `FittedMultiOutputLinearRegression<F>` with `coefficients: Array2<F>` of shape `(n_targets, n_features)` (sklearn `coef_` orientation, transpose of the lstsq `(n_features, n_targets)` solution, `_base.py:688`) and `intercepts: Array1<F>` of shape `(n_targets,)`. Solves all targets in one SVD via `linalg::solve_lstsq_multi` → `ferray::linalg::lstsq` with a 2-D `b` (mirrors `linalg.lstsq(X, Y)`, `_base.py:687`); shared X-centering + per-target y-offset, `intercepts = y_off − coefficients · x_off` (`_set_intercept`, `_base.py:322`). `impl Predict<Array2<F>, Output = Array2<F>>` returns `X · coef_.T + intercepts` shape `(n_samples, n_targets)` (`_base.py:290`). Oracle-verified vs live sklearn: `coef_=[[2.06666667,-0.06666667],[0.86666667,0.23333333]]`, `intercept_=[-0.06666667,0.13333333]`, `predict(X[:2])=[[2.0,1.0],[4.0,2.1]]` (`linreg_multioutput_*` tests); 1-D path byte-identical (`linreg_single_output_unchanged`). |
| REQ-7 (sample_weight) | SHIPPED | #373: `pub fn fit_with_sample_weight(&self, x, y, sample_weight: Option<&Array1<F>>)` implements weighted OLS — weighted offsets `Σwx/Σw`/`Σwy/Σw` (`_average`, `_base.py:193`) + `√w` row-rescaling (`_rescale_data`, `_base.py:641`) + `solve_lstsq` + `intercept = y_off − x_off·coef` (`_set_intercept`, `_base.py:320`). `Fit::fit` delegates with `None` (byte-identical unweighted path). Oracle-verified: weighted coef `2.0935828877`/intercept `-0.2326203209` (intercept) and coef `2.0350877193` (no-intercept) vs live sklearn; `linreg_fit_sample_weight_*` tests. |
| REQ-8 (rank_/singular_, copy_X/n_jobs) | NOT-STARTED | open prereq blocker #374. `FittedLinearRegression<F>` stores only `coefficients`/`intercept` — no `rank_`/`singular_` (`_base.py:505`); `LinearRegression<F>` exposes only `fit_intercept`, omitting `copy_X` and `n_jobs` from `_parameter_constraints` (`_base.py:561`). |
| REQ-substrate (ferray) | NOT-STARTED | open prereq blocker #375. The solve calls `crate::linalg::solve_normal_equations`/`solve_lstsq` (faer-backed) and stores `coefficients: Array1<F>` (`ndarray`); migrating the solve to `ferray::linalg` and the coef storage/return type off `ndarray` cascades through `HasCoefficients` (#359). |

## Architecture

**Types.** `LinearRegression<F>` (`linear_regression.rs`) is the unfitted configuration struct holding
the single hyperparameter `fit_intercept: bool` (sklearn ctor `_base.py:568`; ferrolearn omits
`copy_X`, `n_jobs`, `positive` — see REQ-5/REQ-8). `new()` defaults `fit_intercept = true` matching
sklearn's `fit_intercept=True` (`_base.py:571`). `FittedLinearRegression<F>` holds
`coefficients: Array1<F>` (sklearn `coef_`, 1-D only — REQ-6) and `intercept: F` (sklearn
`intercept_`). The unfitted/fitted split is the compile-time analog of sklearn's runtime
`check_is_fitted` (`ferrolearn-core` traits doc).

**Fit (REQ-1/REQ-3).** `Fit::fit` validates shapes (`ShapeMismatch` when `y.len() != n_samples`;
`InsufficientSamples` when `n_samples == 0`). When `fit_intercept`, it applies the **centering
trick**: subtract `x_mean`/`y_mean`, solve the intercept-free system, then recover
`intercept = y_mean − x_mean·w`. This is mathematically equivalent to sklearn's `_preprocess_data`
column-centering (`_base.py:629`) followed by `_set_intercept` (`intercept_ = y_offset − X_offset @ coef_`,
`_base.py:320`). When `!fit_intercept`, it solves on raw `X` and sets `intercept = 0`
(sklearn `_base.py:512`).

**Numerical method (known divergence — do NOT claim bit-exact lstsq parity).** ferrolearn solves
via `solve_normal_equations` (Cholesky on `XᵀX`) with an `.or_else` fallback to `solve_lstsq`
(QR via faer, `linalg.rs`). sklearn's dense path uses `scipy.linalg.lstsq` → LAPACK `gelsd`
(SVD-based), which returns the **minimum-norm** solution for rank-deficient `X` and also exposes
`rank_`/`singular_` (`_base.py:687`). Consequences for the critic to pin:
- **Full-rank, well-conditioned `X`:** both methods converge to the same OLS solution; coefficients
  match to high precision (oracle full-rank case → `coef_=[1,2]`, agreeing within `1e-8`).
- **Rank-deficient `X`:** sklearn returns the unique minimum-norm solution (oracle: duplicate-column
  `X` → `coef_=[0.5, 0.5]`, the min-norm split). ferrolearn's normal equations are singular there;
  the QR fallback returns *a* least-squares solution that is **not guaranteed** to be minimum-norm —
  the two **diverge**. This is a numerical-method difference inherent to the normal-equations + QR
  approach, not a coefficient bug on full-rank inputs.

**Predict (REQ-2).** `Predict::predict` checks feature-count agreement (`ShapeMismatch`) then computes
`X·coefficients + intercept`, the 1-D arm of sklearn `_decision_function` (`_base.py:288`).

**Introspection (REQ-4).** `HasCoefficients for FittedLinearRegression` returns
`&Array1<F>`/`F` for `coef_`/`intercept_` (sklearn `_base.py:499`, `_base.py:511`).

**Pipeline / consumers.** `PipelineEstimator`/`FittedPipelineEstimator` impls box the estimator for
pipeline use. The production (non-test) consumers of this estimator are the PyO3 binding
`RsLinearRegression in ferrolearn-python/src/regressors.rs` (constructed in `LinearRegression.__init__`
in `ferrolearn-python/python/ferrolearn/_regressors.py`, registered in `ferrolearn-python/src/lib.rs`)
and `RansacRegressor in ransac.rs`, which consumes any `Fit`/`Predict` estimator generically
(documented with `LinearRegression` as the canonical base). As a boundary public estimator type,
`LinearRegression` is grandfathered under goal.md R-DEFER-1.

## Verification

Commands establishing the SHIPPED claims:

- `cargo test -p ferrolearn-linear` — 8/8 `linear_regression::tests` pass (`test_simple_linear_regression`,
  `test_multiple_linear_regression`, `test_no_intercept`, `test_has_coefficients`,
  `test_shape_mismatch_fit`, `test_shape_mismatch_predict`, `test_pipeline_integration`,
  `test_f32_support`).
- sklearn oracle (full-rank, REQ-1/REQ-3):
  `python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; X=np.array([[1.,1.],[1.,2.],[2.,2.],[2.,3.]]); y=X@np.array([1.,2.])+3; m=LinearRegression().fit(X,y); print(m.coef_, m.intercept_)"`
  → `coef_=[1. 2.]`, `intercept_≈3.0`, matching `test_multiple_linear_regression`.
- sklearn oracle (no-intercept, REQ-3):
  `python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; m=LinearRegression(fit_intercept=False).fit(np.array([[1.],[2.],[3.],[4.]]), np.array([2.,4.,6.,8.])); print(m.coef_, m.intercept_)"`
  → `coef_=[2.]`, `intercept_=0.0`, matching `test_no_intercept`.
- Rank-deficient divergence reference (REQ-1 method note, NOT a SHIPPED parity claim):
  `python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; m=LinearRegression(fit_intercept=False).fit(np.array([[1.,1.],[2.,2.],[3.,3.]]), np.array([1.,2.,3.])); print(m.coef_)"`
  → `[0.5 0.5]` (sklearn min-norm); ferrolearn's QR fallback need not reproduce this split.

REQs 5–8 and REQ-substrate are NOT-STARTED; their acceptance criteria (AC-5..AC-8) have no green
verification until blockers #371, #372, #373, #374, #375 land.
