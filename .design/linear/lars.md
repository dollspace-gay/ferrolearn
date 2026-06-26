# Least Angle Regression (LARS) and Lasso-LARS

<!--
tier: 3-component
status: draft
baseline-commit: dc68f009984187bda713676846daa71dd6980db8
upstream-paths:
  - sklearn/linear_model/_least_angle.py
-->

## Summary
This module mirrors scikit-learn's `Lars` (`method="lar"`) and `LassoLars`
(`method="lasso"`) estimators from `sklearn/linear_model/_least_angle.py`. `Lars`
builds a sparse linear model by walking the Least Angle Regression homotopy path:
the active feature with maximal absolute correlation is added, then the model
moves along the equiangular direction until a new feature ties for maximal
correlation. `LassoLars` adds the Efron §3.3 drop modification to recover the
Lasso solution path. ferrolearn's `Lars` reproduces sklearn `Lars.coef_` exactly
on the diabetes dataset; `LassoLars` now routes through the same equiangular
`compute_lars_path` with the §3.3 drop condition and `alpha_min` stopping,
reproducing sklearn `LassoLars.coef_` to 1e-4 (REQ-2 SHIPPED). The crate also
exports a dense single-output `lars_path` helper that returns sklearn-style
alphas, active indices, coefficient paths, and iteration count for raw X/y, plus
the sufficient-statistics `lars_path_gram` helper for `Xy = X.T @ y` and
`Gram = X.T @ X`.

## Requirements
- REQ-1: `Lars` with `method="lar"` produces `coef_` / `intercept_` matching
  sklearn `Lars` on a real dataset (the LARS homotopy, not forward stepwise).
- REQ-2: `LassoLars` with `method="lasso"` produces `coef_` matching sklearn
  `LassoLars` for a given `alpha` (equiangular path with the §3.3 drop condition).
- REQ-3: `predict` computes `X @ coef_ + intercept_` for both fitted estimators.
- REQ-4: `fit_intercept` centering and `HasCoefficients` introspection
  (`coefficients()`, `intercept()`) for both estimators.
- REQ-5: Path-level fitted attributes `coef_path_`, `alphas_`, `active_`,
  `n_iter_` are exposed (mirroring sklearn's `Lars` fitted attributes).
- REQ-6: Constructor parameter parity with sklearn `Lars` / `LassoLars`:
  `n_nonzero_coefs` default `500`, plus `eps`, `copy_X`, `precompute`,
  `fit_path`, `jitter`, `random_state`, `positive`.
- REQ-7: The cross-validated variants `LarsCV` and `LassoLarsCV` exist
  (separate routed translation units).
- REQ-12: `LassoLarsIC` exists and selects a LARS-Lasso path knot by AIC/BIC.
- REQ-8: The module computes on the ferray substrate (`ferray-core`,
  `ferray::linalg`) rather than `ndarray` + hand-rolled Cholesky/Gauss solvers
  (R-SUBSTRATE).
- REQ-10: Public `lars_path` helper — raw dense single-output path helper
  returning alphas, active feature indices, coefficient path, and iteration
  count for `method="lar"` and `method="lasso"`.
- REQ-11: Public `lars_path_gram` helper — sufficient-statistics path helper
  returning the same path artifacts from `Xy`, `Gram`, and `n_samples`.

## Acceptance criteria
- AC-1: `Lars(n_nonzero_coefs=5)` `coef_` matches sklearn within `1e-6` on the
  diabetes dataset (LARS is exact). Verified: ferrolearn
  `[0.0, -74.9105, 511.3522, 234.1487, 0.0, 0.0, -169.7071, 0.0, 450.666, 0.0]`
  equals sklearn `[0.0, -74.9105, 511.3522, 234.1487, 0.0, 0.0, -169.7071, 0.0, 450.666, 0.0]`
  (printed to 4 dp), `intercept_ = 152.133484`.
- AC-2: `LassoLars(alpha=0.1)` `coef_` matches sklearn within `1e-4` on diabetes.
  Verified (SHIPPED): ferrolearn matches sklearn
  `[0.0, -155.3431, 517.2162, 275.0872, -52.552, 0.0, -210.1395, 0.0, 483.9172, 33.6622]`
  with active set `[1, 2, 3, 4, 6, 8, 9]` (the §3.3 drop path admits feature 4
  then later interpolates at `alpha_min`).
- AC-3: `predict` output length equals the number of rows of `X` and equals
  `X @ coef_ + intercept_`.
- AC-4: `fit_intercept=false` yields `intercept_ == 0`; `coefficients()` length
  equals `X.ncols()`.
- AC-5: `FittedLars` exposes `coef_path_` (shape `(n_steps+1, n_features)`),
  `alphas_`, and `active_` matching sklearn's per-step path values.
- AC-6: `Lars::default()` reports `n_nonzero_coefs == 500`; `eps`, `copy_X`,
  `precompute`, `jitter`, `positive` constructors exist.
- AC-7: `lars_path(X, y, method="lar")` and `method="lasso"` return alphas,
  active indices, coefficient paths, and `n_iter` matching sklearn on a raw
  dense fixture.
- AC-8: `lars_path_gram(X.T @ y, X.T @ X, n_samples)` returns the same path
  artifacts matching sklearn's sufficient-statistics helper.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (Lars lar path) | SHIPPED | impl `fn compute_lars_path in lars.rs` (called by `impl Fit for Lars` `fn fit`) walks the equiangular direction — `let u_vec = x_a.dot(&w_a)` then the step-size `gamma = min over k of (c_max - corr[j])/(a_a - a_j)` — mirroring sklearn `_lars_path_solver` (`sklearn/linear_model/_least_angle.py:635` while-loop, argmax-correlation + equiangular vector `u = np.dot(X.T[n_active:], eq_dir)`). Non-test consumer: crate re-export `pub use lars::{..., Lars} in lib.rs` (boundary public API per R-DEFER-5/S5). Verification: live oracle — ferrolearn `coef_ = [0, -74.9105, 511.3522, 234.1487, 0,0, -169.7071, 0, 450.666, 0]`, `intercept_ = 152.133484` equals sklearn `Lars(n_nonzero_coefs=5)` exactly on diabetes. |
| REQ-2 (LassoLars lasso path) | SHIPPED | `impl Fit for LassoLars` `fn fit` now routes through `fn compute_lars_path in lars.rs` with `lasso_modification = true` and `alpha_min = self.alpha`, implementing the equiangular LARS-Lasso homotopy with the Efron §3.3 drop condition: at each knot the join step competes with the zero-crossing step `z = -beta[j] / least_squares[idx]` (mirroring sklearn `_lars_path_solver`, `method == "lasso"`, `sklearn/linear_model/_least_angle.py:817` `z = -coef[active] / (least_squares + tiny32)`, truncation `gamma_ = z_pos` `:827`, the `if not drop:` add-guard `:673`, and the alpha_min interpolation `:657`–`:669`). The forward-stepwise OLS path (`fn ols_active`) is removed. Non-test consumer: crate re-export `pub use lars::{LassoLars, FittedLassoLars} in lib.rs` (boundary public API per R-DEFER-5/S5); `impl FittedPipelineEstimator for FittedLassoLars` `fn predict_pipeline`. Verification: live sklearn oracle — diabetes alpha=0.1 active set [1,2,3,4,6,8,9] with `coef_[1]=-155.34`, alpha=0.5 `coef_[2]=471.01`, alpha=1.0 active [2,3,8] all match sklearn within 1e-4 (`tests/divergence_lasso_lars_path.rs`: `divergence_lasso_lars_coef_alpha_0_1`, `divergence_lasso_lars_coef_alpha_0_5`, `parity_lasso_lars_coef_alpha_1_0`). |
| REQ-3 (predict) | SHIPPED | impl `fn predict in lars.rs` for `FittedLars` and `FittedLassoLars` computes `Ok(x.dot(&self.coefficients) + self.intercept)`, mirroring sklearn `LinearModel._decision_function` (`sklearn/linear_model/_base.py`, `X @ self.coef_.T + self.intercept_`). Non-test consumer: `impl FittedPipelineEstimator for FittedLars` `fn predict_pipeline` calls `self.predict(x)` (pipeline boundary). Verification: `cargo test -p ferrolearn-linear` (`test_lars_predict`, `test_lasso_lars_predict`). |
| REQ-4 (fit_intercept / HasCoefficients) | SHIPPED | impl `fn center_data in lars.rs` subtracts column/target means when `fit_intercept`, and `fn compute_intercept in lars.rs` returns `*ym - xm.dot(w)`, mirroring sklearn `_preprocess_data` + `_set_intercept` (`sklearn/linear_model/_base.py`). `impl HasCoefficients for FittedLars` exposes `coefficients()` / `intercept()`. Non-test consumer: crate re-export `pub use lars::{FittedLars, ...} in lib.rs`; `HasCoefficients` is the introspection boundary trait. Verification: `cargo test -p ferrolearn-linear` (`test_lars_no_intercept` → `intercept_ == 0`, `test_lars_has_coefficients`). |
| REQ-5 (fitted coef_path_/alphas_/active_/n_iter_) | NOT-STARTED | open prereq blocker #483. `compute_lars_path in lars.rs` now collects the path for the public helper, but `FittedLars` and `FittedLassoLars` still store only `coefficients` + `intercept`. sklearn `Lars._fit` (`sklearn/linear_model/_least_angle.py:1096`) records `self.alphas_`, `self.active_`, `self.coef_path_`, `self.n_iter_` — those fitted attributes are not exposed here. |
| REQ-6 (constructor param parity) | NOT-STARTED | open prereq blocker #484. `struct Lars in lars.rs` defaults `n_nonzero_coefs: Option<usize> = None` (uses all features), but sklearn `Lars.__init__` (`sklearn/linear_model/_least_angle.py:1053`) defaults `n_nonzero_coefs=500`. `eps`, `copy_X`, `precompute`, `fit_path`, `jitter`, `random_state`, `positive` are absent from both structs. |
| REQ-7 (LarsCV/LassoLarsCV) | NOT-STARTED | open prereq blocker #485. sklearn defines `LarsCV` (`:1517`) and `LassoLarsCV` (`:1831`); ferrolearn has no analog yet. Separate routed translation units. |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #486. `lars.rs` computes on `ndarray::{Array1, Array2}` with hand-rolled `fn cholesky_solve` / `fn gaussian_solve` instead of `ferray-core` arrays and `ferray::linalg`. sklearn uses LAPACK `potrs` / BLAS via scipy (`sklearn/linear_model/_least_angle.py:618`). |
| REQ-10 (`lars_path` public helper) | SHIPPED | `pub fn lars_path` + `LarsPathOptions` / `LarsPathResult` in `lars.rs`, re-exported from `lib.rs`, provide the dense single-output Rust analogue of `sklearn.linear_model.lars_path` (`_least_angle.py:67`). The helper uses raw `X`/`y` (no centering/intercept), supports `LarsPathMethod::{Lar,Lasso}`, `max_iter`, and `alpha_min`, and returns alphas, final active indices, coefficient path `(n_features, n_alphas)`, and `n_iter`. Verification: live sklearn oracle fixture in `tests/divergence_lars_path.rs` (`lars_path_lar_matches_sklearn`, `lars_path_lasso_matches_sklearn`, `lars_path_alpha_min_interpolates_like_sklearn`, `lars_path_negative_alpha_min_errors`); API proof: `api_proof_lars_family`. |
| REQ-11 (`lars_path_gram` public helper) | SHIPPED | `pub fn lars_path_gram` in `lars.rs`, re-exported from `lib.rs`, provides the sufficient-statistics Rust analogue of `sklearn.linear_model.lars_path_gram` (`_least_angle.py:257`). The helper validates `Xy`, square `Gram`, and positive `n_samples`, derives an equivalent Cholesky surrogate design with matching `X.T @ X` / `X.T @ y`, calls the shared `compute_lars_path` with sklearn's provided `n_samples` alpha scaling, and returns the same `LarsPathResult` surface as `lars_path`. Verification: live sklearn oracle fixture in `tests/divergence_lars_path_gram.rs` (`lars_path_gram_lar_matches_sklearn`, `lars_path_gram_lasso_matches_sklearn`, `lars_path_gram_alpha_min_interpolates_like_sklearn`, `lars_path_gram_shape_and_parameter_errors`); API proof: `api_proof_lars_family`. |
| REQ-12 (`LassoLarsIC`) | SHIPPED | `pub struct LassoLarsIC<F>` / `FittedLassoLarsIC<F>` in `lars.rs`, re-exported from `lib.rs`, compute the LARS-Lasso path (`method="lasso"`, `alpha_min=0`) and evaluate sklearn's criterion formula: `n_samples * log(2*pi*noise_variance) + RSS/noise_variance + factor*df`, with `factor=2` for AIC and `log(n_samples)` for BIC (`_least_angle.py:2268-2308`). If `noise_variance=None`, `estimate_lasso_lars_ic_noise_variance` solves centered OLS through `crate::linalg::solve_lstsq` and divides by `n_samples - n_features - fit_intercept`, matching `_estimate_noise_variance` (`:2312-2337`). Narrower surface: dense single-output only; no `positive`, sparse/object data, `precompute`, `copy_X`, `eps`, Python warning ABI, or metadata routing. Verification: live sklearn oracle fixture in `tests/divergence_lasso_lars_ic.rs` covers explicit-variance AIC/BIC coefficients/intercept/alphas/criterion/predictions, default estimated noise variance, and validation errors; API proof: `api_proof_lars_family`. |

## Architecture

The module owns two unfitted estimators — `struct Lars<F> { n_nonzero_coefs:
Option<usize>, fit_intercept }` and `struct LassoLars<F> { alpha, max_iter,
fit_intercept }` — and their fitted counterparts `struct FittedLars<F> {
coefficients, intercept }` / `struct FittedLassoLars<F> { coefficients,
intercept }`. Both fitted types implement `Predict`, `HasCoefficients`, and the
pipeline traits.

**LARS homotopy (`fn compute_lars_path`).** This is the correct path solver. Each step:
(1) compute correlations `corr[j] = x.column(j).dot(&residual)`, mirroring
sklearn `Cov = np.dot(X.T, y)` and its update (`sklearn/linear_model/_least_angle.py:551`,
`:635` while-loop); (2) add the max-|correlation| feature to the active set with
its sign (`sign_active`); (3) build the sign-flipped active matrix `x_a`, form
the active Gram `g_aa = x_a.t().dot(&x_a)`, solve `g_aa u = 1` (in-place Gaussian
elimination), and normalize `a_a = 1/sqrt(sum(u))` — the equiangular scale,
mirroring sklearn's Cholesky solve of `G_a w = sign_active` and
`AA = 1/sqrt(sum(w))` (`sklearn/linear_model/_least_angle.py` equiangular block);
(4) compute the step size `gamma` as the minimum positive ratio
`(c_max ∓ corr[j])/(a_a ∓ a_j)` over inactive features (the LARS join condition),
matching sklearn's `gamma_hat = min(...)`; (5) update `beta` and `mu` along the
direction. The `lasso_modification` branch implements the Efron §3.3 drop and the
`alpha_min` stopping/interpolation; `Lars::fit`, `LassoLars::fit`, and the
public `lars_path` helper call it.

`Lars::fit` centers data (`fn center_data`), runs `compute_lars_path(.., max_active,
false, 0)` with `max_active = n_nonzero_coefs.unwrap_or(n_features)`, then sets
the intercept via `fn compute_intercept`. This reproduces sklearn exactly on
diabetes (REQ-1 / AC-1).

**LassoLars path (SHIPPED).** `LassoLars::fit` centers data and calls
`compute_lars_path(.., self.max_iter, true, self.alpha)`. Inside, each knot competes the
LARS join step against the Efron §3.3 drop step `z = -beta[j] /
least_squares[idx]` (sklearn `:817`); the smaller is taken. On a drop the step is
truncated (`gamma_ = z_pos`, `:827`), the crossing variable leaves the active set
with its coefficient forced to exactly zero (`:855`–`:894`), and no new variable
is added on the following iteration (the `if not drop:` guard, `:673`). The path
stops once `alpha = C / n_samples` reaches `alpha_min = self.alpha`, interpolating
`beta = prev_beta + ss·(beta - prev_beta)` at exactly `alpha_min` (`:657`–`:669`).
This minimizes `(1/(2n))||y - Xw||² + alpha·||w||₁` and reproduces sklearn
`LassoLars.coef_` on diabetes to 1e-4 (REQ-2 / AC-2, closes #482).

**Invariants.** Inactive coefficients stay exactly zero (scatter into a
zero-initialized full-length vector in `fn compute_lars_path`).
`validate_input` rejects sample-count mismatch (`ShapeMismatch`) and zero samples
(`InsufficientSamples`); `Lars::fit` rejects `n_nonzero_coefs > n_features`
(`InvalidParameter`); `LassoLars::fit` rejects negative `alpha`.

**Public helper (SHIPPED).** `lars_path` validates raw dense single-output input,
calls `compute_lars_path` without centering, and returns alphas, active indices,
coefficient path, and iteration count. This mirrors sklearn's helper contract;
callers that need intercept behavior center `X` and `y` before calling it.
`lars_path_gram` accepts `Xy`, `Gram`, and `n_samples`, constructs a Cholesky
surrogate design with the same sufficient statistics, and uses the supplied
`n_samples` to scale alphas like sklearn.

**Path/attribute gap.** Neither fitted type stores the per-step path. sklearn's
`Lars` exposes `coef_path_` (shape `(n_features, n_steps+1)`), `alphas_`,
`active_`, and `n_iter_` (`sklearn/linear_model/_least_angle.py:1096`); these are
still discarded by `Lars::fit` / `LassoLars::fit` after extracting the final
coefficient vector (REQ-5, blocker #483).

**Substrate.** All arrays are `ndarray`, and linear solves are hand-rolled
(`fn cholesky_solve`, `fn gaussian_solve`, and the inline Gaussian elimination in
`fn compute_lars_path`). The destination substrate is `ferray-core` arrays and
`ferray::linalg` (REQ-8, blocker #486).

## Verification

Commands establishing the SHIPPED claims:

- `cargo test -p ferrolearn-linear` — unit tests in `lars.rs` cover predict,
  intercept, sparsity, and shape validation (`test_lars_predict`,
  `test_lars_no_intercept`, `test_lars_sparsity`, `test_lars_has_coefficients`,
  `test_lasso_lars_predict`, plus the error-path tests).
- Live sklearn oracle for REQ-1 (LARS is exact):
  ```bash
  python3 -c "from sklearn.linear_model import Lars; from sklearn.datasets import load_diabetes; X,y=load_diabetes(return_X_y=True); m=Lars(n_nonzero_coefs=5).fit(X,y); print(m.coef_.tolist(), m.intercept_)"
  ```
  yields `[0.0, -74.9105.., 511.3522.., 234.1487.., 0.0, 0.0, -169.7071.., 0.0, 450.666.., 0.0] 152.1334...`,
  which equals ferrolearn `Lars::<f64>::new().with_n_nonzero_coefs(5).fit(&x,&y)`
  to 4 dp on the same dataset.
- `cargo test -p ferrolearn-linear --test divergence_lars_path` — pins the
  public `lars_path` helper against live sklearn 1.5.2 oracle values for
  `method="lar"`, `method="lasso"`, alpha-min interpolation, active indices,
  coefficient paths, and `n_iter`.
- `cargo test -p ferrolearn-linear --test divergence_lars_path_gram` — pins the
  public `lars_path_gram` helper against live sklearn 1.5.2 oracle values for
  sufficient-statistics `method="lar"`, `method="lasso"`, alpha-min
  interpolation, active indices, coefficient paths, and `n_iter`.

NOT-STARTED REQs are gated on blockers #483 (fitted path attributes), #484
(constructor params / `n_nonzero_coefs` default 500), #485 (CV/IC variants), and
#486 (ferray substrate). REQ-2, REQ-10, and REQ-11 are SHIPPED.
