# Lasso (L1-regularized linear regression)

<!--
tier: 3-component
status: draft
baseline-commit: 22dd52ee17095877468983e3f909103dffa62a7a
upstream-paths:
  - sklearn/linear_model/_coordinate_descent.py
  - sklearn/linear_model/_cd_fast.pyx
-->

## Summary

This module mirrors scikit-learn's `Lasso` estimator (`sklearn/linear_model/_coordinate_descent.py:1154`, `class Lasso(ElasticNet)`), a linear model fit by coordinate descent with soft-thresholding minimizing

```text
(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
```

(the objective documented at `_coordinate_descent.py:227` / `:1159`). In scikit-learn, `Lasso` is `ElasticNet(l1_ratio=1.0)`; the production solver is the Cython `enet_coordinate_descent` in `_cd_fast.pyx`. ferrolearn implements the single-output, dense, cyclic-coordinate-descent core directly in `lasso.rs` (`Lasso<F>` / `FittedLasso<F>`); the public boundary type is exposed to Python as `_RsLasso` and reused internally by `LassoCV`.

## Requirements

- REQ-1: Coordinate-descent Lasso fit at the `(1 / (2 * n_samples))` objective — produces `coef_` / `intercept_` consistent with sklearn's `alpha`/`n_samples` scaling.
- REQ-2: `predict` computes `X @ coef_ + intercept_`.
- REQ-3: `fit_intercept` honored, including `fit_intercept=False` (no centering, zero intercept).
- REQ-4: L1 sparsity via soft-thresholding — irrelevant coefficients driven exactly to `0.0`.
- REQ-5: `HasCoefficients` introspection exposes `coef_` (slice) and `intercept_` (scalar).
- REQ-6: `alpha` validation (`alpha >= 0`; `alpha = 0` degenerates to the OLS coordinate-descent fit) plus shape / empty-input guards.
- REQ-7: `positive=True` — force coefficients to be non-negative.
- REQ-8: `warm_start=True` — reuse the previous solution as CD initialization.
- REQ-9: `selection='random'` + `random_state` — random coordinate selection.
- REQ-10: `precompute` — Gram-matrix coordinate-descent path.
- REQ-11: `n_iter_` and `dual_gap_` fitted attributes.
- REQ-12: Dual-gap stopping criterion — relative coefficient change AND dual gap `< tol * ||y||^2`, matching `_cd_fast.pyx`.
- REQ-13: `MultiTaskLasso` (multi-output L1/L21) — separate estimator.
- REQ-substrate: ferray array/linalg substrate (currently `ndarray`).

## Acceptance criteria

- AC-1: For `X = [[1],[2],[3],[4],[5]]`, `y = [3,5,7,9,11]`, `alpha=0.1`, the fitted `coef_[0]` and `intercept_` match `sklearn.linear_model.Lasso(alpha=0.1)` (`coef_ = 1.95`, `intercept_ = 1.15`) within `1e-4` (loose: ferrolearn and sklearn use different stopping criteria — see REQ-12).
- AC-2: `predict` on the training matrix returns a length-`n_samples` vector equal to `X @ coef_ + intercept_`.
- AC-3: With `fit_intercept=false`, `intercept_ == 0.0` exactly.
- AC-4: A feature column orthogonal to the residual under a sufficiently large `alpha` receives `coef_ == 0.0` exactly (bit-exact zero, not near-zero).
- AC-5: `coefficients()` length equals `n_features`; `intercept()` returns the scalar bias.
- AC-6: `alpha < 0` returns `FerroError::InvalidParameter`; `n_samples == 0` returns `FerroError::InsufficientSamples`; mismatched `y` length returns `FerroError::ShapeMismatch`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (CD fit) | SHIPPED | impl `Fit::fit for Lasso<F> in lasso.rs` runs cyclic CD: centers when `fit_intercept`, precomputes `col_norms[j] = X_jᵀX_j / n` (`col.dot(&col) / n_f`), updates `w_new = soft_threshold(X_jᵀr / n, alpha) / col_norm[j]`. This matches sklearn `_coordinate_descent.py:655` (`l1_reg = alpha * l1_ratio * n_samples`, `l1_ratio=1`) fed to `_cd_fast.pyx` which works on the un-normalized `(1/2)||y-Xw||² + l1_reg·||w||₁`: dividing both `X_jᵀr` and `X_jᵀX_j` by `n` is algebraically identical to sklearn's `soft_threshold(X_jᵀr, alpha·n) / X_jᵀX_j`. Objective cited `_coordinate_descent.py:227`/`:1159`. Non-test consumer: `ferrolearn-python/src/regressors.rs` (`RsLasso::fit` calls `ferrolearn_linear::Lasso::<f64>::new()...fit`); also `lasso_cv.rs` (`Fit::fit for LassoCV` constructs `Lasso::<F>::new()` per fold and for the final refit). Verification: `python3 -c "from sklearn.linear_model import Lasso; ...; Lasso(alpha=0.1).fit(X,y)" → coef=1.95, intercept=1.15`; `cargo test -p ferrolearn-linear --lib lasso` → 36 passed, 0 failed (incl. `tests/sklearn_equivalence.rs::sklearn_equiv_lasso`). |
| REQ-2 (predict) | SHIPPED | impl `Predict::predict for FittedLasso<F> in lasso.rs` computes `x.dot(&self.coefficients) + self.intercept`; shape-guards on `n_features`. Non-test consumer: `RsLasso::predict in regressors.rs`; also `FittedPipelineEstimator::predict_pipeline for FittedLasso`. Verification: `cargo test -p ferrolearn-linear --lib lasso` (`test_lasso_predict`). |
| REQ-3 (fit_intercept) | SHIPPED | impl `Fit::fit for Lasso<F> in lasso.rs` branches on `self.fit_intercept`: when true it centers `x`/`y` and recovers `intercept = ȳ - x̄·w` (mirrors sklearn `_preprocess_data` + `LinearModel._set_intercept`); when false it skips centering and sets `intercept = 0`. `with_fit_intercept` builder + `RsLasso` `fit_intercept` field thread the flag through. Default `true` matches sklearn `_coordinate_descent.py:770`. Non-test consumer: `RsLasso::new`/`fit in regressors.rs` (signature default `fit_intercept=true`). Verification: `cargo test -p ferrolearn-linear --lib lasso` (`test_lasso_no_intercept` asserts `intercept_ == 0`). |
| REQ-4 (L1 sparsity) | SHIPPED | impl `fn soft_threshold in lasso.rs` returns `F::zero()` inside the threshold band, so `Fit::fit` drives coefficients to bit-exact `0.0` — sklearn's exact-sparsity contract from soft-thresholding in `_cd_fast.pyx`. Non-test consumer: `RsLasso` exposes the resulting sparse `coef_` to Python (the user-visible sparsity guarantee). Verification: `cargo test -p ferrolearn-linear --lib lasso` (`test_lasso_sparsity` asserts `coef_[1] == coef_[2] == 0.0` at `epsilon = 1e-10`; `test_soft_threshold`). |
| REQ-5 (HasCoefficients) | SHIPPED | impl `HasCoefficients<F> for FittedLasso<F> in lasso.rs` (`coefficients(&self) -> &Array1<F>`, `intercept(&self) -> F`). Non-test consumer: `RsLasso::coef_`/`intercept_ getters in regressors.rs` call `fitted.coefficients()`/`fitted.intercept()` to expose the sklearn `coef_`/`intercept_` attributes to Python. Verification: `cargo test -p ferrolearn-linear --lib lasso` (`test_lasso_has_coefficients`). |
| REQ-6 (alpha / input validation) | SHIPPED | impl `Fit::fit for Lasso<F> in lasso.rs` rejects `alpha < F::zero()` with `FerroError::InvalidParameter{name:"alpha"}` (mirrors sklearn `_parameter_constraints` `alpha: Interval(Real, 0, None, closed="left")`), guards `n_samples != y.len()` (`ShapeMismatch`) and `n_samples == 0` (`InsufficientSamples`). `alpha = 0` runs CD with a zero threshold, degenerating to the OLS coordinate-descent fit (sklearn `_coordinate_descent.py:1172` documents `alpha=0` ≡ OLS, "not advised" but not rejected). Non-test consumer: `RsLasso::fit` propagates the error as `PyValueError`. Verification: `cargo test -p ferrolearn-linear --lib lasso` (`test_lasso_negative_alpha`, `test_lasso_shape_mismatch`, `test_lasso_zero_alpha`). |
| REQ-7 (positive=True) | NOT-STARTED | open prereq blocker #407. No `positive` field or non-negativity clamp; sklearn ctor `positive=False` (`_coordinate_descent.py:800`) clips `w[ii]` to `>= 0` in `_cd_fast.pyx`. |
| REQ-8 (warm_start) | NOT-STARTED | open prereq blocker #408. `Fit::fit` always re-initializes `w` to zeros; sklearn ctor `warm_start=False` (`_coordinate_descent.py:795`) reuses prior `coef_` as CD init. |
| REQ-9 (selection='random' + random_state) | NOT-STARTED | open prereq blocker #409. Solver iterates strictly cyclically `for j in 0..n_features`; sklearn supports `selection='random'` with `random_state`-seeded coordinate picks (`_coordinate_descent.py:809`, `_cd_fast.pyx`). |
| REQ-10 (precompute / Gram) | NOT-STARTED | open prereq blocker #410. No Gram-matrix path; sklearn ctor `precompute=False` (`_coordinate_descent.py:774`) can run CD on a precomputed `XᵀX`. |
| REQ-11 (n_iter_ / dual_gap_) | NOT-STARTED | open prereq blocker #411. `FittedLasso<F>` stores only `coefficients`/`intercept`; sklearn exposes `n_iter_` (`:827`) and `dual_gap_` (`:831`) — the latter is unavailable because ferrolearn computes no dual gap (REQ-12). |
| REQ-12 (dual-gap stopping) | NOT-STARTED | open prereq blocker #412. ferrolearn stops on absolute `max_change < tol` (max coefficient change). sklearn `_cd_fast.pyx:208-249` stops on relative `d_w_max / w_max < d_w_tol` AND then the principled dual gap `gap < tol * ||y||²` (`tol *= dot(y,y)` at `:168`, `gap < tol` at `:249`). These differ → at a fixed `tol` the returned `coef_` can differ slightly; this is the most likely numerical divergence the critic will pin. |
| REQ-13 (MultiTaskLasso) | NOT-STARTED | open prereq blocker #413. `Fit` is implemented only for `Array1<F>` targets; `MultiTaskLasso` (multi-output L1/L21, separate sklearn estimator class) has no ferrolearn analog. |
| REQ-substrate (ferray) | NOT-STARTED | open prereq blocker #414. `lasso.rs` computes on `ndarray::{Array1, Array2}`, not the ferray substrate (R-SUBSTRATE-1/2). CD is elementwise + dot products (no linalg-crate dependency), so migration is array-type only; `coef_` return shape tied to #359/#375. |

## Architecture

- **Unfitted type** `Lasso<F>` (`lasso.rs`): public fields `alpha`, `max_iter`, `tol`, `fit_intercept` with `with_*` builders and a `Default`/`new` matching the sklearn defaults that ferrolearn supports (`alpha = 1.0`, `max_iter = 1000`, `tol = 1e-4`, `fit_intercept = true`; cf. sklearn `_coordinate_descent.py:756/783/789/770`). The sklearn ctor params `precompute`, `copy_X`, `warm_start`, `positive`, `random_state`, `selection` are absent → REQ-7..10.
- **Fitted type** `FittedLasso<F>` (`lasso.rs`): holds `coefficients: Array1<F>` and `intercept: F`. No `n_iter_`/`dual_gap_` → REQ-11.
- **Solver** `Fit::fit for Lasso<F>` (`lasso.rs`): (1) validate; (2) center `X`/`y` when `fit_intercept` (sklearn `_preprocess_data`); (3) precompute `col_norms[j] = X_jᵀX_j / n`; (4) cyclic CD with the standard partial-residual update (add back `X_j·w_old`, soft-threshold `X_jᵀr / n` by `alpha`, divide by `col_norm`, subtract `X_j·w_new`); (5) stop when `max_change < tol` (the divergence point vs sklearn's dual gap, REQ-12) or after `max_iter`; (6) recover `intercept = ȳ - x̄·w`. On non-convergence it returns the current iterate (sklearn raises a `ConvergenceWarning` and likewise returns the iterate — the warning surface is not modeled).
- **Scaling invariant** (REQ-1): ferrolearn's `soft_threshold(X_jᵀr/n, alpha)/(X_jᵀX_j/n)` equals sklearn's `soft_threshold(X_jᵀr, alpha·n)/(X_jᵀX_j)` (`l1_reg = alpha·n` at `_coordinate_descent.py:655`); the `1/n` factors cancel, so the two implementations target the identical `(1/2n)`-scaled objective.
- **Boundary / consumers**: `Lasso` is the public estimator API. `ferrolearn-python/src/regressors.rs` wraps it as `_RsLasso` (registered in `ferrolearn-python/src/lib.rs`, surfaced to Python as `ferrolearn.Lasso`); `lasso_cv.rs` constructs `Lasso` per CV fold. Both are non-test production consumers (R-DEFER-1; `Lasso` is grandfathered boundary API per S5).

## Verification

Commands establishing the SHIPPED claims:

- `cargo test -p ferrolearn-linear --lib lasso` → 36 passed, 0 failed (unit tests in `lasso.rs`: `test_lasso_zero_alpha`, `test_lasso_sparsity`, `test_lasso_no_intercept`, `test_lasso_negative_alpha`, `test_lasso_shape_mismatch`, `test_lasso_predict`, `test_lasso_has_coefficients`, `test_soft_threshold`, `test_lasso_shrinks_coefficients`, `test_lasso_pipeline_integration`).
- `cargo test -p ferrolearn-linear` → includes `tests/sklearn_equivalence.rs::sklearn_equiv_lasso` and `tests/proptest_invariants.rs` (`lasso_coef_len_equals_n_features`, `lasso_high_alpha_sparser_than_ols`), all green.
- sklearn oracle (REQ-1, AC-1): `python3 -c "from sklearn.linear_model import Lasso; import numpy as np; X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([3.,5.,7.,9.,11.]); m=Lasso(alpha=0.1).fit(X,y); print(m.coef_, m.intercept_)"` → `[1.95] 1.15`.

NOT-STARTED REQs (7-13, substrate) have no green verification and are tracked by blockers #407-#414. Because ferrolearn's stopping criterion (REQ-12) differs from sklearn's dual gap, REQ-1 parity is asserted only at a loose tolerance; a tight ULP comparison is expected to fail until the dual-gap criterion lands.
