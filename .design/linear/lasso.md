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

- AC-1: For `X = [[1],[2],[3],[4],[5]]`, `y = [3,5,7,9,11]`, `alpha=0.1`, the fitted `coef_[0]` and `intercept_` match `sklearn.linear_model.Lasso(alpha=0.1)` (`coef_ = 1.95`, `intercept_ = 1.15`). Now that REQ-12 lands sklearn's dual-gap stopping criterion, coef parity holds to `1e-7` (`lasso_dual_gap_stopping_matches_sklearn_coef_and_niter` asserts the multi-feature fixture at `1e-7`).
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
| REQ-7 (positive=True) | SHIPPED | `Lasso<F>` carries `pub positive: bool` (default `false`) with a `with_positive` builder, threading sklearn's `positive` ctor param (`_coordinate_descent.py:800`, `"positive": ["boolean"]` at `:891`). In the CD coordinate update `Fit::fit for Lasso<F>` branches on `self.positive`: when `true` it applies `fn soft_threshold_positive` (`max(rho − alpha, 0)/col_norm`), the non-negative soft-threshold mirroring `_cd_fast.pyx:191-194` (`if positive and tmp < 0: w[ii] = 0.0` else the signed soft-threshold). `positive=false` is byte-identical to the prior fit (`lasso_positive_false_unchanged` asserts `coef_`/`intercept_` equality). The intercept recovery is unchanged (the constraint binds coef, not the intercept — oracle intercept `-5.98636364` is unconstrained). Verification (live sklearn 1.5.2 oracle, R-CHAR-3): on `X=[[1,3],[2,1],[3,4],[4,2],[5,5],[6,1],[2,4],[5,2]]`, `y=X[:,0]-2*X[:,1]+noise`, `Lasso(alpha=0.3, positive=True)` → `coef_=[1.14431818, 0.0]`, `intercept_=-5.98636364` vs unconstrained `[0.8946582, -1.83087261]` — `cargo test -p ferrolearn-linear --lib lasso` (`lasso_positive_matches_sklearn`, `lasso_positive_false_unchanged`, `lasso_positive_all_nonneg_unconstrained_equals`, `test_soft_threshold_positive`). Non-test consumer: `Lasso` boundary API (grandfathered, S5). |
| REQ-8 (warm_start) | NOT-STARTED | open prereq blocker #408. `Fit::fit` always re-initializes `w` to zeros; sklearn ctor `warm_start=False` (`_coordinate_descent.py:795`) reuses prior `coef_` as CD init. |
| REQ-9 (selection='random' + random_state) | SHIPPED | `CoordSelection {Cyclic, Random}` enum + `selection`/`random_state` fields + `with_selection`/`with_random_state` builders; `Fit::fit` shuffles the per-sweep coordinate order via `StdRng::seed_from_u64` when Random (`_coordinate_descent.py:809`, `_cd_fast.pyx` random branch). Cyclic default byte-identical (bit-exact to sklearn). Random verified to converge to the unique optimum; exact bitwise parity to sklearn's `selection='random'` is numpy-MT19937-RNG-blocked (StdRng != MT). Tests `lasso_selection_cyclic_default_unchanged`, `lasso_selection_random_converges_to_optimum`. |
| REQ-10 (precompute / Gram) | NOT-STARTED | open prereq blocker #410. No Gram-matrix path; sklearn ctor `precompute=False` (`_coordinate_descent.py:774`) can run CD on a precomputed `XᵀX`. |
| REQ-11 (n_iter_ / dual_gap_) | SHIPPED | `FittedLasso<F> in lasso.rs` now carries `n_iter`/`dual_gap` fields with `#[must_use]` getters `n_iter()`/`dual_gap()`, mirroring sklearn `Lasso.n_iter_` (`_coordinate_descent.py:1103`/`:827`) and `dual_gap_` (`:1108`/`:831`). `fn lasso_dual_gap in lasso.rs` implements the duality gap from `_cd_fast.pyx:216-247` (`XtA = Xcᵀ·R − β·w` with `β=0`, `l1_reg = α·n` per `_coordinate_descent.py:655`, branch on `dual_norm_XtA > l1_reg`) and divides by `n` to map sklearn's un-normalized `(1/2)` objective to ferrolearn's `(1/2n)` scaling; `Fit::fit` calls it on the (centered/raw) CD design with the final coef. With REQ-12's dual-gap stopping criterion landed, `n_iter_`'s VALUE now matches sklearn exactly (`n_iter_ == 20` at alpha=0.3 and alpha=0.1 on the fixture); `dual_gap_` matches sklearn's formula and value (`lasso_dual_gap` → `0.00011701482` at the optimum). Verification (R-CHAR-3, numpy/sklearn oracle points): `cargo test -p ferrolearn-linear --lib lasso` — `lasso_dual_gap_formula_matches_numpy` (gap at `w=[0.5,1.0]`→`0.465888`, at `w=[0.66691036,1.46647171]`→`0.0001170161`), `lasso_fitted_dual_gap_and_n_iter`, `lasso_fields_dont_change_coef`, `lasso_dual_gap_stopping_matches_sklearn_coef_and_niter`. |
| REQ-12 (dual-gap stopping) | SHIPPED | `Fit::fit for Lasso<F> in lasso.rs` replaces the old absolute `max_change < tol` break with sklearn's two-level criterion (`_cd_fast.pyx:167-249`): `tol_scaled = tol·(target·target)` (`tol *= dot(y,y)`, `:167-168`); per sweep track `w_max`/`d_w_max`; gate on `w_max == 0 || d_w_max/w_max < tol || last_iter` (`:207-211`); inside the gate break only when the UN-normalized gap `lasso_dual_gap(...)·n < tol_scaled` (`gap < tol` at `:249`). `lasso_dual_gap` returns the `/n` attribute value (REQ-11); multiplying back by `n` recovers the `(1/2)||·||² + (α·n)||w||₁` objective sklearn's stop test compares. This makes `coef_` match sklearn to ≤1e-7 AND `n_iter_` match exactly (resolving the REQ-11 `n_iter_` value caveat). Verification (R-CHAR-3, live sklearn 1.5.2 oracle): `Lasso(alpha=0.3).fit(X,y)` → `coef_=[0.66691036, 1.46647171]`, `n_iter_=20`; `Lasso(alpha=0.1)` → `coef_=[0.72247514, 1.52201988]`, `n_iter_=20` — `cargo test -p ferrolearn-linear --lib lasso` (`lasso_dual_gap_stopping_matches_sklearn_coef_and_niter`, `lasso_dual_gap_stopping_second_alpha`). |
| REQ-13 (MultiTaskLasso) | NOT-STARTED | open prereq blocker #413. `Fit` is implemented only for `Array1<F>` targets; `MultiTaskLasso` (multi-output L1/L21, separate sklearn estimator class) has no ferrolearn analog. |
| REQ-substrate (ferray) | NOT-STARTED | open prereq blocker #414. `lasso.rs` computes on `ndarray::{Array1, Array2}`, not the ferray substrate (R-SUBSTRATE-1/2). CD is elementwise + dot products (no linalg-crate dependency), so migration is array-type only; `coef_` return shape tied to #359/#375. |

## Architecture

- **Unfitted type** `Lasso<F>` (`lasso.rs`): public fields `alpha`, `max_iter`, `tol`, `fit_intercept` with `with_*` builders and a `Default`/`new` matching the sklearn defaults that ferrolearn supports (`alpha = 1.0`, `max_iter = 1000`, `tol = 1e-4`, `fit_intercept = true`; cf. sklearn `_coordinate_descent.py:756/783/789/770`). The sklearn ctor params `precompute`, `copy_X`, `warm_start`, `positive`, `random_state`, `selection` are absent → REQ-7..10.
- **Fitted type** `FittedLasso<F>` (`lasso.rs`): holds `coefficients: Array1<F>`, `intercept: F`, plus `n_iter: usize` and `dual_gap: F` (REQ-11; `n_iter()`/`dual_gap()` getters mirror sklearn `n_iter_`/`dual_gap_`).
- **Solver** `Fit::fit for Lasso<F>` (`lasso.rs`): (1) validate; (2) center `X`/`y` when `fit_intercept` (sklearn `_preprocess_data`); (3) precompute `col_norms[j] = X_jᵀX_j / n`; (4) cyclic CD with the standard partial-residual update (add back `X_j·w_old`, soft-threshold `X_jᵀr / n` by `alpha`, divide by `col_norm`, subtract `X_j·w_new`); (5) stop using sklearn's two-level criterion (REQ-12, `_cd_fast.pyx:167-249`): the relative-change gate `d_w_max/w_max < tol` (or `w_max == 0`, or last iter) opens the dual-gap check, and the loop breaks when the un-normalized gap `lasso_dual_gap(...)·n < tol·(target·target)`, else after `max_iter`; (6) recover `intercept = ȳ - x̄·w`. On non-convergence it returns the current iterate (sklearn raises a `ConvergenceWarning` and likewise returns the iterate — the warning surface is not modeled).
- **Scaling invariant** (REQ-1): ferrolearn's `soft_threshold(X_jᵀr/n, alpha)/(X_jᵀX_j/n)` equals sklearn's `soft_threshold(X_jᵀr, alpha·n)/(X_jᵀX_j)` (`l1_reg = alpha·n` at `_coordinate_descent.py:655`); the `1/n` factors cancel, so the two implementations target the identical `(1/2n)`-scaled objective.
- **Boundary / consumers**: `Lasso` is the public estimator API. `ferrolearn-python/src/regressors.rs` wraps it as `_RsLasso` (registered in `ferrolearn-python/src/lib.rs`, surfaced to Python as `ferrolearn.Lasso`); `lasso_cv.rs` constructs `Lasso` per CV fold. Both are non-test production consumers (R-DEFER-1; `Lasso` is grandfathered boundary API per S5).

## Verification

Commands establishing the SHIPPED claims:

- `cargo test -p ferrolearn-linear --lib lasso` → 36 passed, 0 failed (unit tests in `lasso.rs`: `test_lasso_zero_alpha`, `test_lasso_sparsity`, `test_lasso_no_intercept`, `test_lasso_negative_alpha`, `test_lasso_shape_mismatch`, `test_lasso_predict`, `test_lasso_has_coefficients`, `test_soft_threshold`, `test_lasso_shrinks_coefficients`, `test_lasso_pipeline_integration`).
- `cargo test -p ferrolearn-linear` → includes `tests/sklearn_equivalence.rs::sklearn_equiv_lasso` and `tests/proptest_invariants.rs` (`lasso_coef_len_equals_n_features`, `lasso_high_alpha_sparser_than_ols`), all green.
- sklearn oracle (REQ-1, AC-1): `python3 -c "from sklearn.linear_model import Lasso; import numpy as np; X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([3.,5.,7.,9.,11.]); m=Lasso(alpha=0.1).fit(X,y); print(m.coef_, m.intercept_)"` → `[1.95] 1.15`.

NOT-STARTED REQs (8-10, 13, substrate) have no green verification and are tracked by blockers #408-#414. With REQ-12's dual-gap stopping criterion landed, ferrolearn's stopping point matches sklearn's, so REQ-1 coef parity holds at `1e-7` (`lasso_dual_gap_stopping_matches_sklearn_coef_and_niter`) and `n_iter_` matches sklearn exactly.
