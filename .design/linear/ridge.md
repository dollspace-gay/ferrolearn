# Ridge Regression

<!--
tier: 3-component
status: draft
baseline-commit: f012c75716c4d80c74b236b7628c43df9ef4075e
upstream-paths:
  - sklearn/linear_model/_ridge.py
  - sklearn/linear_model/_base.py
-->

## Summary
`ridge.rs` mirrors scikit-learn's `class Ridge` (`sklearn/linear_model/_ridge.py:1016`): linear
least squares with an L2 penalty, `||y - Xw||^2_2 + alpha * ||w||^2_2`, solved in closed form. The
ferrolearn implementation covers sklearn's default dense path — `solver='auto'` → `'cholesky'` with
`fit_intercept` handled by centering — and exposes `Ridge<F>` / `FittedRidge<F>` (single output) plus
`FittedRidgeMulti<F>` (2-D `Y`). It also exposes the dense single-output
`sklearn.linear_model.ridge_regression` helper through `ridge_regression`. The remaining estimator gaps
are sklearn's iterative solvers and the ferray substrate migration.

## Requirements
- REQ-1: Fit `coef_` and `intercept_` by minimizing `||y - Xw||^2_2 + alpha * ||w||^2_2` via Cholesky
  on `(X^T X + alpha * I) w = X^T y`, with the intercept handled by centering so the intercept term
  is NOT penalized (sklearn's default `solver='auto'` → `'cholesky'` dense path).
- REQ-2: `predict(X) = X @ coef_ + intercept_`.
- REQ-3: `fit_intercept` (default `true`); when `false`, `X`/`y` are used uncentered and `intercept_ = 0`.
- REQ-4: Expose `coef_`/`intercept_` for introspection via `HasCoefficients`.
- REQ-5: `alpha` must be a non-negative float in `[0, inf)`; reject negative `alpha` with a parameter
  error; `alpha = 0` reduces to ordinary least squares, including the **minimum-norm** solution on a
  rank-deficient `X` (mirrors sklearn's `'cholesky'` → SVD fallback on a singular `X^T X`,
  `sklearn/linear_model/_ridge.py:752-756`), rather than erroring.
- REQ-6: Multi-output regression — accept a 2-D target `Y` of shape `(n_samples, n_targets)` and produce
  a `(n_features, n_targets)` coefficient matrix with a per-target intercept vector.
- REQ-7: Per-target `alpha` — accept `alpha` as an array of shape `(n_targets,)` (one penalty per target).
- REQ-8a: Solver selection (dense closed-form) — support `solver` in {`auto`, `cholesky`, `svd`} on the
  single-output path and expose the resolved `solver_` fitted attribute (`auto` → `cholesky` for the
  dense path). All dense solvers yield the identical unique ridge solution.
- REQ-8b: Solver selection (iterative) — support `solver` in {`lsqr`, `sparse_cg`, `sag`, `saga`,
  `lbfgs`}. Blocked on the iterative/SGD substrate + RNG (#386).
- REQ-9: `positive=True` constrained (non-negative coefficient) fitting via the lbfgs solver.
- REQ-10: `max_iter`/`tol` controls for iterative solvers and the `n_iter_` fitted attribute.
- REQ-11: `sample_weight` support (per-sample weighting of the loss).
- REQ-12: `copy_X` and `random_state` constructor parameters.
- REQ-13 (substrate): run the owned array/linalg computation on ferray (`ferray-core` array type,
  `ferray::linalg` Cholesky) rather than `ndarray` + the in-crate `linalg` Cholesky.
- REQ-14: Public `ridge_regression` helper — solve the dense single-output helper contract from
  `sklearn.linear_model.ridge_regression`, defaulting to the uncentered normal equation while supporting
  sample weights, dense solver selection (`auto`/`cholesky`/`svd`), `positive=True`, and optional
  intercept return through a fixed Rust result struct.

## Acceptance criteria
- AC-1: On `X = [[1,0],[0,1],[1,1],[2,2]]`, `y = [1,2,3,6]`, `alpha=1.0`, `fit_intercept=true`,
  `coef_ ≈ [0.875, 1.375]` and `intercept_ ≈ 0.75` (matches `sklearn.linear_model.Ridge`).
- AC-2: `predict` on the training `X` returns a length-`n_samples` vector equal to `X @ coef_ + intercept_`.
- AC-3: With `fit_intercept=false`, `alpha=0`, `X=[[1],[2],[3],[4]]`, `y=[2,4,6,8]`: `coef_ ≈ [2.0]`,
  `intercept_ = 0.0` (matches `Ridge(alpha=0, fit_intercept=False)`).
- AC-4: `HasCoefficients::coefficients()` and `intercept()` return the fitted values.
- AC-5: `alpha = -1.0` causes `fit` to return an error; `alpha = 0` recovers the OLS slope/intercept.
- AC-6: For a 2-D `Y` synthesized from two distinct linear targets with `alpha → 0`, the recovered
  `(n_features, n_targets)` coefficients and per-target intercepts match the generating coefficients.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (L2 cholesky fit, intercept unpenalized) | SHIPPED | impl `pub fn fit (Fit<Array2<F>, Array1<F>>) in ridge.rs` centers `x`/`y` then solves via `crate::linalg::solve_ridge` (`let x_centered = x - &x_mean; ... let w = linalg::solve_ridge(&x_centered, &y_centered, self.alpha)?; let intercept = y_mean - x_mean.dot(&w);`). Centering means `alpha` never touches the intercept — mirrors sklearn `_ridge_regression` cholesky branch `sklearn/linear_model/_ridge.py:741-756` (`coef = _solve_cholesky(X, y, alpha)`) preceded by `_preprocess_data` centering `sklearn/linear_model/_base.py:116` and `_set_intercept` `sklearn/linear_model/_base.py:308`. Non-test consumer: `ferrolearn-python/src/regressors.rs` (`RsRidge::fit` → `ferrolearn_linear::Ridge::<f64>::new().with_alpha(..).with_fit_intercept(..).fit(..)`); also `ridge_cv.rs` (`FittedRidgeCV` fits inner `Ridge::<F>::new()`). Verification: live `Ridge(alpha=1.0).fit` gives `coef_=[0.875,1.375]`, `intercept_=0.75`; `cargo test -p ferrolearn-linear` `sklearn_equiv_ridge`/`test_ridge_oracle` green. |
| REQ-2 (predict = X·coef + intercept) | SHIPPED | impl `pub fn predict (Predict<Array2<F>>) for FittedRidge in ridge.rs` (`let preds = x.dot(&self.coefficients) + self.intercept;`). Mirrors sklearn `LinearModel.predict` / `_decision_function` `sklearn/linear_model/_base.py` (`X @ self.coef_.T + self.intercept_`). Non-test consumer: `ferrolearn-python/src/regressors.rs` (`RsRidge::predict` → `fitted.predict(&x_nd)`); `ferrolearn-io/src/onnx.rs` + `pmml.rs` serialize `FittedRidge` via `coefficients()`/`intercept()`. Verification: `cargo test -p ferrolearn-linear` `test_ridge_predict` green. |
| REQ-3 (fit_intercept incl. false) | SHIPPED | impl `pub fn with_fit_intercept in ridge.rs` + the `if self.fit_intercept { ... } else { let w = linalg::solve_ridge(x, y, self.alpha)?; ... intercept: F::zero() }` branch in `Fit::fit`. Mirrors sklearn ctor `fit_intercept=True` `sklearn/linear_model/_ridge.py:897` and `_preprocess_data(fit_intercept=...)` `sklearn/linear_model/_base.py:116`. Non-test consumer: `ferrolearn-python/src/regressors.rs` (`RsRidge::new(alpha, fit_intercept)` threads `fit_intercept` through `with_fit_intercept`). Verification: `cargo test -p ferrolearn-linear` `test_ridge_no_intercept` green (`coef_=2.0`, `intercept_=0.0`). |
| REQ-4 (HasCoefficients introspection) | SHIPPED | impl `impl HasCoefficients<F> for FittedRidge in ridge.rs` (`fn coefficients(&self) -> &Array1<F>`, `fn intercept(&self) -> F`). Mirrors sklearn fitted attrs `coef_`/`intercept_` `sklearn/linear_model/_ridge.py:994`/`:1011`. Non-test consumer: `ferrolearn-io/src/onnx.rs` (`linear_regressor_to_onnx(self.coefficients(), self.intercept(), "FittedRidge")`) and `ferrolearn-io/src/pmml.rs` (`linear_model_to_pmml`), plus `ferrolearn-python/src/regressors.rs` `coef_` getter. Verification: `cargo test -p ferrolearn-linear` `test_ridge_has_coefficients` green. |
| REQ-5 (alpha ≥ 0 validation; alpha=0 → OLS incl. rank-deficient min-norm) | SHIPPED | impl `Fit::fit in ridge.rs` (`if self.alpha < <F as num_traits::Zero>::zero() { return Err(FerroError::InvalidParameter { name: "alpha".into(), reason: "must be non-negative".into() }); }`). Mirrors sklearn `_ridge_regression` `check_scalar(alpha, "alpha", min_val=0.0, include_boundaries="left")` `sklearn/linear_model/_ridge.py:693-699` and the `alpha = 0 → OLS` contract documented at `sklearn/linear_model/_ridge.py:1037`. The `alpha = 0` solve on a **rank-deficient** `X` now returns the minimum-norm coefficients instead of erroring: `pub(crate) fn solve_ridge in linalg.rs` chains `cholesky_solve(..).or_else(\|_\| gaussian_solve(..)).or_else(\|_\| solve_lstsq(x, y))`, so a singular `X^T X` falls through to the gelsd min-norm lstsq — the Rust analog of sklearn's `'cholesky'` branch switching to `solver = "svd"` on `linalg.LinAlgError` (`sklearn/linear_model/_ridge.py:752-756`). For `alpha > 0` the PD Cholesky always succeeds so the lstsq branch is unreachable (no behavior change). Non-test consumer: `ferrolearn-python/src/regressors.rs` (`RsRidge::fit` → `ferrolearn_linear::Ridge::<f64>::new()...fit(..)`, `f64: ferray::linalg::LinalgFloat`) maps the negative-alpha error to `PyValueError` (R-DEV-2). Verification: `cargo test -p ferrolearn-linear` `test_ridge_negative_alpha`, `test_ridge_no_regularization`, proptest `ridge_alpha_zero_matches_ols`, and `tests/divergence_ridge_numeric.rs::divergence_ridge_alpha_zero_rank_deficient_min_norm` (un-ignored; coef `[0.2, 0.4]`, intercept ≈ 0 vs the live sklearn oracle) all green. Closed #392. |
| REQ-6 (multi-output 2-D Y) | SHIPPED | `FittedRidgeMulti<F>` + the `Fit<Array2<F>, Array2<F>> for Ridge` impl (per-target `solve_ridge` over centered design; `coefficients()` `(n_features, n_targets)`, `intercepts()` `(n_targets,)`), mirroring sklearn `Ridge` 2-D `coef_` `(n_targets, n_features)` / `intercept_` `(n_targets,)` (`sklearn/linear_model/_ridge.py:543`/`:550`). Non-test production consumer (R-HONEST-4 correction — the prior NOT-STARTED note predates the multi-output Python binding #29): `RsRidgeMultiOutput in ferrolearn-python/src/regressors.rs` holds `Option<FittedRidgeMulti<f64>>` and transposes `coefficients()` to sklearn's `(n_targets, n_features)`; `_regressors.py::Ridge.fit` routes the `y.ndim == 2 && y.shape[1] > 1` path to `_RsRidgeMultiOutput`. Verification (live sklearn 1.5.2, R-CHAR-3): `tests/divergence_regressors.py::test_ridge_multioutput_matches_sklearn` (coef `(2,2)`, intercept `(2,)`, predict ≤ 1e-8) + `test_ridge_singleoutput_unchanged_by_multioutput_path`; `ferrolearn.Ridge(alpha=a).fit(X, Y_2d)` matches sklearn coef/predict to ≤ 1e-16. |
| REQ-7 (per-target alpha array) | SHIPPED | `Ridge<F>` carries `pub alpha_per_target: Option<Array1<F>>` (default `None`) + `with_alpha_per_target` builder. The multi-output `Fit<Array2,Array2>::fit` branches: `Some(alphas)` validates `len == n_targets` (`ShapeMismatch`) + each `>= 0` (`InvalidParameter`), then solves each target column independently via `linalg::solve_ridge(&x_used, &y_col, alphas[k])` (mathematically identical to an independent scalar-alpha Ridge per target, mirroring `sklearn/linear_model/_ridge.py:701-712`). `None` path byte-identical (`solve_ridge_multi` with scalar `self.alpha`). Verification (live sklearn 1.5.2, R-CHAR-3): `Ridge(alpha=[0.5,2.0]).fit(X,Y)` → coef cols `[0.79891892,1.43891892]`/`[0.78,0.355]`, intercepts `[-0.75351351,-0.065]`. Tests `ridge_per_target_alpha_matches_sklearn`, `ridge_per_target_alpha_equals_independent_scalar_fits`, `ridge_per_target_alpha_length_mismatch_errors`, `ridge_multi_scalar_alpha_unchanged`. |
| REQ-8a (dense solver variants auto/cholesky/svd + solver_) | SHIPPED | `pub enum RidgeSolver { #[default] Auto, Cholesky, Svd } in ridge.rs` + `Ridge<F>` gains `pub solver: RidgeSolver` (default `Auto`) and `with_solver(RidgeSolver)`; `FittedRidge<F>` gains `solver_: RidgeSolver` + `pub fn solver(&self) in ridge.rs`. The single-output `fit_with_sample_weight` computes `let resolved = self.solver.resolve();` (`Auto` → `Cholesky`, sklearn `resolve_solver` `sklearn/linear_model/_ridge.py:830`) and dispatches the unconstrained dense solve through `fn solve_unconstrained in ridge.rs`: `RidgeSolver::Svd` → `crate::linalg::solve_ridge_svd` (new `pub(crate) fn solve_ridge_svd in linalg.rs`, computing `coef = V·diag(sᵢ/(sᵢ²+alpha))·Uᵀy` from the thin SVD `X=U·diag(s)·Vᵀ` via `ferray::linalg::svd(.., false)`, the Rust analog of sklearn `_solve_svd` `sklearn/linear_model/_ridge.py:200-216`); `Cholesky`/`Auto` → the unchanged `crate::linalg::solve_ridge` (byte-identical to today). All dense solvers yield the IDENTICAL unique solution (strictly convex problem). `solver_` stores the resolved value. Governs only the single-output unconstrained dense path (multi-output + `positive=true` unaffected). Non-test consumer: `Fit<Array2<F>, Array1<F>>::fit for Ridge in ridge.rs` (→ `fit_with_sample_weight(x, y, None)` → `solve_unconstrained(resolved, ..)`), reached in production by `RsRidge::fit` in `ferrolearn-python/src/regressors.rs` and by `ridge_cv.rs` (`FittedRidgeCV` refits inner `Ridge`). Verification (live sklearn 1.5.2, R-CHAR-3): `Ridge(alpha=1.0, solver='svd').fit(X,y)` on `X=[[1,2],[2,1],[3,4],[4,3],[5,5]]`, `y=[3.0,2.5,7.1,6.0,11.2]` → coef `[0.8228070175, 1.3561403509]`, intercept `-0.5768421053`. Tests `ridge_solver_svd_matches_sklearn_and_cholesky` (svd matches oracle ≤1e-7 AND Cholesky ≤1e-9), `ridge_solver_resolution` (`Auto`→`Cholesky`, explicit `Svd`), `ridge_solver_default_cholesky_unchanged` (byte-identical regression guard), `ridge_solver_svd_no_intercept` (svd == cholesky no-intercept ≤1e-9). |
| REQ-8b (iterative solver variants lsqr/sparse_cg/sag/saga/lbfgs) | NOT-STARTED | open prereq blocker #386 — needs the iterative/SGD substrate + RNG. sklearn dispatches these via `resolve_solver` (`sklearn/linear_model/_ridge.py:830`); they are intentionally NOT represented as `RidgeSolver` variants until the substrate lands. On the strictly-convex dense problem they would converge to the same unique solution the dense solvers (REQ-8a) already produce. |
| REQ-9 (positive=True) | SHIPPED | `Ridge<F>` gains `pub positive: bool` (default `false`, `_ridge.py:902`/`:911`) + `with_positive(bool)`. The fit's coefficient solve routes through `solve_nonneg_ridge` (projected coordinate descent: `new = max(0, (A[:,j]ᵀr + col_sq[j]·old)/(col_sq[j] + alpha))`, incremental residual update, `max_iter`/`tol` from REQ-10) on the centered/`√w`-rescaled design, instead of `solve_ridge`, when `positive=True` — the same unique non-negative-ridge optimum sklearn's L-BFGS-B reaches (`_solve_lbfgs`, `sklearn/linear_model/_ridge.py:300`, objective `0.5·‖Xw−y‖²+0.5·alpha·‖w‖²` with bounds `[(0,inf)]`, dispatched at `:923-928`). Intercept recovered identically; `n_iter_ = Some(iters)` on the CD path, `None` for direct Cholesky; `positive=false` byte-identical to today. Oracle test `ridge_positive_matches_sklearn` (coef `[1.19891304, 0.0]`, intercept `-6.17744565`) + `ridge_positive_false_unchanged` + `ridge_positive_all_nonneg_equals_unconstrained`. Closes #387. |
| REQ-10 (max_iter/tol + n_iter_) | SHIPPED | `Ridge<F>` adds `pub max_iter: Option<usize>` (default `None`) and `pub tol: F` (default `1e-4`) with `with_max_iter`/`with_tol` builders, mirroring sklearn ctor `max_iter=None, tol=1e-4` (`sklearn/linear_model/_ridge.py:899-900`). `FittedRidge<F>` adds `n_iter_: Option<usize>` (always `None` for the direct Cholesky solver) with `pub fn n_iter() in ridge.rs`, mirroring `n_iter_` set at `_ridge.py:994`; `max_iter`/`tol` are exposed for ABI parity but are no-ops for the direct closed-form solver (no iteration) — matching sklearn's `cholesky`/`svd` paths which also yield `n_iter_=None`. When an iterative solver is added (future REQ-8 #386), they will take effect. Test: `ridge_max_iter_tol_niter_defaults_and_builders`. Closes #388. |
| REQ-11 (sample_weight) | SHIPPED | impl `pub fn fit_with_sample_weight(x, y, sample_weight: Option<&Array1<F>>) in ridge.rs` solves WEIGHTED ridge `min Σᵢ wᵢ(yᵢ−xᵢ·coef)² + alpha·‖coef‖²`: weighted offsets `x_off[j]=Σwᵢx[i,j]/Σwᵢ`, `y_off=Σwᵢyᵢ/Σwᵢ` for `fit_intercept`, centering, then `√wᵢ` row-rescaling (sklearn `_rescale_data`, `sklearn/linear_model/_ridge.py:682-688`), `crate::linalg::solve_ridge(&x_scaled, &y_scaled, self.alpha)` with the penalty `alpha` UNSCALED (since `(√w·Xc)ᵀ(√w·Xc) == Xcᵀ·W·Xc`, the cholesky solve `(Xsᵀ·Xs + alpha·I)·coef = Xsᵀ·ys` IS the weighted ridge normal equation), `intercept = y_off − x_off·coef`; `fit_intercept=false` skips centering (raw `√w`-rescale, intercept 0). `Fit::fit` now delegates `self.fit_with_sample_weight(x, y, None)`, keeping the `None` path BYTE-IDENTICAL to the historic centering + `solve_ridge` body (and the alpha=0 OLS min-norm fallback from REQ-5). Shape validation: `sample_weight` length must equal `n_samples` else `ShapeMismatch`. Verification: live `Ridge(alpha=1.0).fit(X,y,sample_weight=w)` gives `coef_=[0.9233502538, 1.39678511]`, `intercept_=-0.8033840948`; `cargo test -p ferrolearn-linear` `ridge_fit_sample_weight_with_intercept_matches_sklearn`, `ridge_fit_sample_weight_no_intercept_matches_sklearn` (alpha=2 fit_intercept=False coef `[0.7273779983, 1.3737799835]`), `ridge_fit_none_sample_weight_equals_unweighted` all green. Closed #389. |
| REQ-12 (copy_X / random_state) | SHIPPED | `Ridge<F>` adds `pub copy_x: bool` (default `true`) and `pub random_state: Option<u64>` (default `None`) fields with `with_copy_x`/`with_random_state` builders, mirroring sklearn ctor `copy_X=True` (`_ridge.py:898`) and `random_state=None` (`_ridge.py:903`). `copy_x` is ABI-only (fit never mutates `x`); `random_state` is stored but no-op for the deterministic Cholesky solver (only `sag`/`saga` use it, `_ridge.py:898`/`:903`). Test: `ridge_copy_x_random_state_defaults_and_builders`. Closes #390. |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #391 (refs #359 coef-return shape, #375 solve_ridge → `ferray::linalg`). `ridge.rs` operates on `ndarray::{Array1, Array2}` and `crate::linalg::solve_ridge` (in-crate Cholesky), not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). |
| REQ-14 (`ridge_regression` public helper) | SHIPPED | `pub fn ridge_regression` + `RidgeRegressionOptions` / `RidgeRegressionResult` in `ridge.rs`, re-exported from `lib.rs`, provide the dense single-output Rust analogue of `sklearn.linear_model.ridge_regression` (`_ridge.py:627`). Default options set `return_intercept=false`, so the helper solves the raw normal equation `min ||y-Xw||² + alpha||w||²` without centering, matching sklearn's direct helper default. `sample_weight` delegates to the same row-rescaling path as `Ridge::fit_with_sample_weight`; `solver=Svd` delegates to `linalg::solve_ridge_svd`; `positive=true` delegates to the non-negative ridge CD path. Oracle tests: `ridge_regression_default_matches_sklearn`, `ridge_regression_sample_weight_matches_sklearn`, `ridge_regression_svd_alpha_zero_matches_sklearn`, `ridge_regression_positive_matches_sklearn`; API proof: `api_proof_ridge_family`. |

## Architecture
The module owns two estimator surfaces, both produced by the unfitted `Ridge<F> { alpha, fit_intercept }`
(`pub struct Ridge in ridge.rs`, builders `pub fn new`/`with_alpha`/`with_fit_intercept`/`Default`):

- Single-output: `impl Fit<Array2<F>, Array1<F>> for Ridge<F>` → `FittedRidge<F> { coefficients: Array1<F>, intercept: F }`.
- Multi-output: `impl Fit<Array2<F>, Array2<F>> for Ridge<F>` → `FittedRidgeMulti<F> { coefficients: Array2<F>, intercepts: Array1<F> }`.

**Intercept invariant (REQ-1/REQ-3).** sklearn never penalizes the intercept: `_preprocess_data`
(`sklearn/linear_model/_base.py:116`) subtracts the column means of `X` and the mean of `y` before the
penalized solve, then `_set_intercept` (`sklearn/linear_model/_base.py:308`) recovers
`intercept_ = y_offset - X_offset @ coef_`. `ridge.rs` reproduces this exactly: when `fit_intercept` is
true it forms `x_centered = x - x_mean`, `y_centered = y - y_mean`, calls `linalg::solve_ridge(x_centered,
y_centered, alpha)`, and sets `intercept = y_mean - x_mean.dot(w)`. Because the penalty is applied only to
the centered system, the intercept is unpenalized — matching sklearn's observable contract. When
`fit_intercept` is false, the raw `x`/`y` are solved and `intercept = 0`.

**Solver (REQ-1).** `crate::linalg::solve_ridge` (`pub(crate) fn solve_ridge in linalg.rs`) forms
`X^T X + alpha * I` and runs a Cholesky solve (falling back to a Gaussian solve on factorization
failure). This is the Rust analog of sklearn's `_solve_cholesky` reached from the `'cholesky'` branch
(`sklearn/linear_model/_ridge.py:741-756`); for dense `n_features <= n_samples` data sklearn's
`solver='auto'` resolves to `'cholesky'`, so the default contract is mirrored. The Cholesky factor of
`X^T X + alpha * I` is shared across target columns in `solve_ridge_multi` for the multi-output path.

**Predict (REQ-2).** `FittedRidge::predict` computes `x.dot(coefficients) + intercept`;
`FittedRidgeMulti::predict` computes `x.dot(coefficients)` and broadcast-adds the per-target intercepts.

**Introspection / pipeline (REQ-4).** `FittedRidge` implements `HasCoefficients<F>` and
`FittedPipelineEstimator<F>`; `Ridge<F>` implements `PipelineEstimator<F>`, so a `Ridge` is composable in
the core pipeline. `FittedRidgeMulti` exposes inherent `coefficients()`/`intercepts()` accessors but does
not implement `HasCoefficients` (whose `intercept()` returns a scalar `F`).

**Solver dispatch (REQ-8a).** The single-output path additionally honors the `solver` parameter for the
dense closed-form solvers: `RidgeSolver::{Auto, Cholesky, Svd}`. `Auto` resolves to `Cholesky` for the
dense path (sklearn `resolve_solver`, `_ridge.py:830`); `Svd` routes the unconstrained solve through
`linalg::solve_ridge_svd` (`coef = V·diag(sᵢ/(sᵢ²+alpha))·Uᵀy`, the analog of sklearn `_solve_svd`,
`_ridge.py:200`). Because the ridge problem is strictly convex (unique minimizer), every dense solver
returns the identical solution — the choice only affects the factorization. The resolved solver is stored
as `FittedRidge::solver_`. The iterative solvers (`lsqr`/`sparse_cg`/`sag`/`saga`/`lbfgs`, REQ-8b) remain
NOT-STARTED, pinned on #386 (iterative/SGD substrate + RNG).

**Gaps.** The remaining unimplemented surface is NOT-STARTED: the iterative solvers (REQ-8b) and the
ferray substrate (REQ-13). Each is pinned by an open blocker referenced above.

## Verification
Commands that establish the SHIPPED claims (REQ-1..REQ-5):

- `cargo test -p ferrolearn-linear` — the `ridge::tests` module (30 lib tests incl.
  `test_ridge_no_regularization`, `test_ridge_no_intercept`, `test_ridge_negative_alpha`,
  `test_ridge_predict`, `test_ridge_has_coefficients`), plus `tests/oracle_tests.rs::test_ridge_oracle`,
  `tests/sklearn_equivalence.rs::sklearn_equiv_ridge`, and `tests/proptest_invariants.rs`
  (`ridge_coef_len_equals_n_features`, `ridge_alpha_zero_matches_ols`) all pass.
- sklearn oracle (live):
  `python3 -c "from sklearn.linear_model import Ridge; import numpy as np;
  X=np.array([[1.,0.],[0.,1.],[1.,1.],[2.,2.]]); y=np.array([1.,2.,3.,6.]);
  m=Ridge(alpha=1.0).fit(X,y); print(m.coef_.tolist(), m.intercept_)"`
  → `[0.875..., 1.375...] 0.75` — matches ferrolearn `Ridge::<f64>::new()` on the same input (AC-1).
- Gauntlet: `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`, `cargo fmt --all --check`.

REQ-8a (dense solvers) is verified by `cargo test -p ferrolearn-linear` (`ridge::tests::ridge_solver_*`,
4 tests) against the live sklearn 1.5.2 oracle (`Ridge(alpha=1.0, solver='svd')` coef
`[0.8228070175, 1.3561403509]`, intercept `-0.5768421053`).

REQ-14 (`ridge_regression`) is verified by `cargo test -p ferrolearn-linear --test
divergence_ridge_numeric ridge_regression -- --nocapture` and `cargo test -p ferrolearn-linear --test
api_proof api_proof_ridge_family -- --exact`.

The still-NOT-STARTED REQs (REQ-8b, REQ-13) have no green verification by construction — they are
pinned by open blockers #386 and #391 and become SHIPPED only when the impl plus a non-test consumer plus
a passing sklearn-grounded test all land.
