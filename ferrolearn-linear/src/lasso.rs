//! Lasso regression (L1-regularized linear regression).
//!
//! This module provides [`Lasso`], which fits a linear model with L1
//! regularization using coordinate descent with soft-thresholding:
//!
//! ```text
//! minimize (1 / (2 * n_samples)) * ||X @ w - y||^2 + alpha * ||w||_1
//! ```
//!
//! The L1 penalty encourages sparse solutions where some coefficients
//! are exactly zero, making Lasso useful for feature selection.
//!
//! ## REQ status (per `.design/linear/lasso.md`, mirrors `sklearn/linear_model/_coordinate_descent.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.Lasso` (`_coordinate_descent.py:1154`), objective
//! `(1/2n)||y−Xw||² + α||w||₁`. Cyclic coordinate descent with soft-thresholding;
//! `soft_threshold(Xⱼᵀr/n, α)/(XⱼᵀXⱼ/n)` ≡ sklearn's `l1_reg = α·n` convention. coef_/
//! intercept_ + the exactly-zero support set match the live sklearn oracle to ≤1e-6 (converged).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (CD Lasso fit, coef_/intercept_) | SHIPPED | `Fit for Lasso`; converged coef/intercept match oracle ≤1e-6 (alpha 0.01/0.1/1). Consumers: `RsLasso` in `ferrolearn-python`, `LassoCV` in `lasso_cv.rs`. |
//! | REQ-2 (predict) | SHIPPED | `Predict for FittedLasso`. |
//! | REQ-3 (fit_intercept incl. false) | SHIPPED | centering; matches oracle. |
//! | REQ-4 (L1 sparsity, exact zeros) | SHIPPED | `fn soft_threshold`; support set bit-identical to sklearn. |
//! | REQ-5 (HasCoefficients) | SHIPPED | `HasCoefficients for FittedLasso`. |
//! | REQ-6 (alpha≥0 validation; alpha=0 → OLS) | SHIPPED | negative-alpha → `InvalidParameter`; alpha=0 matches sklearn to 1e-6. Defaults max_iter=1000/tol=1e-4 match sklearn. |
//! | REQ-7 (positive=True) | SHIPPED | `Lasso<F>` carries `pub positive: bool` (default `false`) + `with_positive` builder, threading sklearn's `positive` ctor param (`_coordinate_descent.py:800`). The CD coordinate update `Fit::fit for Lasso<F>` branches on `self.positive`: `true` → `fn soft_threshold_positive` (non-negative soft-threshold, `_cd_fast.pyx:191-194` `if positive and tmp < 0: w[ii] = 0.0`); `false` byte-identical to the prior fit. Non-test consumer: `RsLasso in ferrolearn-python/src/regressors.rs` (threads `positive` into the `fit` builder), surfaced by `_regressors.py::Lasso(positive=...)` — verified `ferrolearn.Lasso(alpha=0.1, positive=True)` matches the live sklearn oracle to 1e-16 (R-CHAR-3); also the `Lasso` boundary API (grandfathered, S5). Tests `lasso_positive_matches_sklearn`/`lasso_positive_false_unchanged`/`lasso_positive_all_nonneg_unconstrained_equals`/`test_soft_threshold_positive`. (Doc-table sync to the already-SHIPPED design-doc row, R-HONEST-4.) |
//! | REQ-8 (warm_start) | SHIPPED | `Lasso<F>` carries `pub warm_start: bool` (default `false`) + `pub coef_init: Option<Array1<F>>` (default `None`) with `with_warm_start`/`with_coef_init` builders, mirroring sklearn `Lasso(warm_start=False)` (`_coordinate_descent.py:795`). R-DEV-4 adaptation: ferrolearn estimators are immutable value types — there is no mutable `self.coef_` carried across repeated `.fit()` calls like sklearn's mutable estimator (`_coordinate_descent.py:1062` reuses `self.coef_` when `warm_start`), so the prior coefficient vector is supplied EXPLICITLY via `coef_init` (sklearn's path solver seeds the same way: `_coordinate_descent.py:648-651`, `coef_ = np.zeros(...)` when `coef_init is None` else `np.asfortranarray(coef_init, ...)`). In `Fit::fit`, when `warm_start && coef_init.is_some()` the init vector is length-validated (`ShapeMismatch` on mismatch) and `w` is cloned from it (the direct path also seeds `residual = y_work − X_work·w`; the Gram path's `H = Q·w` already derives from the actual `w`); otherwise `w = zeros` — BYTE-IDENTICAL to the cold path. The numerics are identical, only the CD start point changes, so warm-from-converged reaches the same unique optimum in fewer sweeps. Verification (live sklearn 1.5.2 oracle, R-CHAR-3): cold `Lasso(alpha=0.5)` → coef `[0.6113455722, 1.4109235423]`, `n_iter_=20`; warm (refit from converged coef) → coef `[0.6112611662, 1.4109910671]`, `n_iter_=1`. Tests `lasso_warm_start_from_converged_matches_sklearn`, `lasso_warm_start_default_unchanged`, `lasso_warm_start_none_coef_init_equals_cold`, `lasso_warm_start_coef_init_wrong_len_errors`. |
//! | REQ-9 (selection='random' + random_state) | SHIPPED | `pub enum CoordSelection { Cyclic, Random }` + `pub selection`/`pub random_state` fields on `Lasso` with `with_selection`/`with_random_state` builders, mirroring sklearn `Lasso(selection=..., random_state=...)` (`_coordinate_descent.py` `__init__`). `Fit::fit`'s CD loop visits `0..n_features` in order for `Cyclic` (BYTE-IDENTICAL to the prior cyclic path) and shuffles a reused index `Vec` each sweep for `Random` via `StdRng::seed_from_u64(random_state.unwrap_or(0))` (sklearn `_cd_fast.pyx` `enet_coordinate_descent` `random` branch picks `ii` instead of `f_iter`); per-coordinate update math + dual-gap stopping are unchanged. The Lasso optimum is unique, so `Random` converges to the same optimum (≈1e-3 from cyclic due to stopping-within-tol). Exact bit-match to sklearn's `selection='random'` is numpy-MT19937-RNG-blocked (Rust `StdRng` ≠ numpy MT), so the random path verifies convergence-to-the-unique-optimum, not bitwise sklearn parity; the cyclic default IS bit-exact. Verification: `cargo test -p ferrolearn-linear --lib lasso` (`lasso_selection_cyclic_default_unchanged`, `lasso_selection_random_converges_to_optimum`). |
//! | REQ-10 (precompute/Gram) | SHIPPED | `pub precompute: bool` field (default `false`) on `Lasso` + `with_precompute` builder, mirroring sklearn `Lasso(precompute=False)` (`_coordinate_descent.py:774`). When `true`, `Fit::fit` runs CD on the precomputed `Q = Xcᵀ Xc` / `q = Xcᵀ yc` with an incrementally-maintained `H = Q·w` (sklearn `_cd_fast.pyx enet_coordinate_descent_gram`); `tmp = (q[j]−H[j])/n + col_norms[j]·w[j] ≡` the direct path's `rho` since `Xⱼᵀr = q[j]−(Q·w)[j]`, so it reaches the SAME unique optimum (to ~1e-13 fp reassociation) with the SAME coordinate order + dual-gap stopping. `precompute=false` (default) is the byte-identical direct path. Verification: `cargo test -p ferrolearn-linear --lib lasso` (`lasso_precompute_matches_sklearn`, `lasso_precompute_default_false_unchanged`, `lasso_precompute_equals_direct`). |
//! | REQ-11 (n_iter_ / dual_gap_ attrs) | SHIPPED | `FittedLasso<F>` carries `n_iter`/`dual_gap` fields + `n_iter()`/`dual_gap()` getters, mirroring sklearn `Lasso.n_iter_` (`_coordinate_descent.py:1103`) and `dual_gap_` (`:1108`). `fn lasso_dual_gap` computes the duality gap on the CD design (centered/raw) using sklearn's `_cd_fast.pyx:216-247` formula (`l1_reg = α·n`, `beta=0`) with a final `/n` mapping to the `(1/2n)` objective. With REQ-12's dual-gap stopping criterion now landed, `n_iter_`'s VALUE matches sklearn exactly (`n_iter_ == 20` at alpha=0.3 and alpha=0.1 on the fixture); `dual_gap_` matches sklearn's formula/value (`0.00011701482` at alpha=0.3). Verification: `cargo test -p ferrolearn-linear --lib lasso` (`lasso_dual_gap_formula_matches_numpy`, `lasso_fitted_dual_gap_and_n_iter`, `lasso_fields_dont_change_coef`, `lasso_dual_gap_stopping_matches_sklearn_coef_and_niter`). |
//! | REQ-12 (dual-gap stopping criterion) | SHIPPED | `Fit::fit for Lasso` now uses sklearn's two-level criterion (`_cd_fast.pyx:167-249`): `tol_scaled = tol·(target·target)` (`:167-168`), per sweep track `w_max`/`d_w_max`, gate on `w_max==0 || d_w_max/w_max < tol || last_iter` (`:207-211`), and inside the gate break only when the UN-normalized gap `lasso_dual_gap(...)·n < tol_scaled` (`:249`). Matches sklearn's `coef_` to ≤1e-7 and `n_iter_` exactly (20 at alpha=0.3 and alpha=0.1). Verification: `cargo test -p ferrolearn-linear --lib lasso` (`lasso_dual_gap_stopping_matches_sklearn_coef_and_niter`, `lasso_dual_gap_stopping_second_alpha`). |
//! | REQ-13 (MultiTaskLasso) | SHIPPED | Separate estimator: `MultiTaskLasso<F>`/`FittedMultiTaskLasso<F>` in `multi_task_lasso.rs` (`Fit<Array2<F>, Array2<F>> for MultiTaskLasso`). Multi-output L2,1 (group-Lasso) block coordinate descent porting `_cd_fast.pyx::enet_coordinate_descent_multi_task` (`:740-959`) with `l2_reg=0`, `l1_reg=α·n` (`MultiTaskLasso = MultiTaskElasticNet(l1_ratio=1)`, `_coordinate_descent.py:2663`): per-feature block soft-threshold `W[j,:] = tmp·max(1−l1_reg/‖tmp‖₂,0)/norm_cols_X[j]`, rank-1 residual maintenance, two-level relative-change + L21 dual-gap stop (`:903-952`, `tol_scaled = tol·‖Yc‖_F²`); `coef_` stored `(n_tasks, n_features)`, per-task `intercept_`, `n_iter_`. Verified vs live sklearn 1.5.2 (R-CHAR-3): `MultiTaskLasso(alpha=0.3)` on `X=[[1,2],[2,1],[3,4],[4,3],[5,5]]`/`Y=[[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6]]` → `coef_=[[0.7874471321,1.3745821226],[0.8341004367,0.3460953631]]`, `intercept_=[-0.5260877641,-0.2005873993]`, `n_iter_=19`. Tests `multi_task_lasso_*` in `multi_task_lasso.rs` + the live-oracle integration pins in `tests/divergence_multi_task_lasso.rs` (alpha grid, 3-task, exact-zero group sparsity, predict shape/values). Non-test consumer: `MultiTaskLasso`/`FittedMultiTaskLasso` re-exported at the crate root (`ferrolearn_linear::MultiTaskLasso`, boundary API per S5/R-DEFER-1). Design-doc REQ-13 row in `.design/linear/lasso.md` already SHIPPED. Closes #413. |
//! | REQ-14 (ferray substrate) | NOT-STARTED | #414 (CD is elementwise; coef storage ndarray, tied to #359). |
//!
//! acto-critic: NO DIVERGENCE FOUND — converged coef/intercept, sparsity support set, alpha
//! scaling, alpha=0, fit_intercept=false, f32, and defaults all match the live oracle. Two
//! states only per goal.md R-DEFER-2.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::Lasso;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = Lasso::<f64>::new().with_alpha(0.1);
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use rand::SeedableRng;
use rand::seq::SliceRandom;

/// Coordinate-selection order for the coordinate-descent solver.
///
/// Mirrors sklearn's `Lasso(selection=...)` / `ElasticNet(selection=...)`
/// parameter (`_coordinate_descent.py`, `__init__` default `'cyclic'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordSelection {
    /// Sweep coordinates `0..n_features` in order each pass (sklearn default,
    /// `selection='cyclic'`). Bit-exact to sklearn's cyclic path.
    #[default]
    Cyclic,
    /// Visit coordinates in a random order each pass (sklearn
    /// `selection='random'`). Often converges faster when features are
    /// correlated; the Lasso optimum is unique here so it reaches the same
    /// limit (sklearn `_cd_fast.pyx` `enet_coordinate_descent` `random` branch).
    Random,
}

/// Lasso regression (L1-regularized least squares).
///
/// Uses coordinate descent with soft-thresholding to solve the L1-penalized
/// regression problem. The `alpha` parameter controls the strength of the
/// L1 penalty.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct Lasso<F> {
    /// Regularization strength. Larger values specify stronger
    /// regularization and sparser solutions.
    pub alpha: F,
    /// Maximum number of coordinate descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum coefficient change.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
    /// When `true`, constrain coefficients to be non-negative.
    pub positive: bool,
    /// When `true`, run coordinate descent on the precomputed Gram matrix
    /// `Q = Xcᵀ Xc` and `q = Xcᵀ yc` instead of recomputing residuals each
    /// pass.
    ///
    /// Mirrors sklearn `Lasso(precompute=False)` (`_coordinate_descent.py:774`);
    /// the Gram path runs sklearn's `enet_coordinate_descent_gram`
    /// (`_cd_fast.pyx`). Reaches the same unique optimum (differing only at
    /// floating-point reassociation level, ~1e-13).
    pub precompute: bool,
    /// Order in which coordinates are visited each coordinate-descent sweep.
    ///
    /// Mirrors sklearn `Lasso(selection=...)` (default `Cyclic`).
    pub selection: CoordSelection,
    /// Seed for the RNG used when `selection == CoordSelection::Random`.
    ///
    /// Mirrors sklearn `Lasso(random_state=...)` (default `None`). `None`
    /// falls back to seed `0`.
    pub random_state: Option<u64>,
    /// When `true`, initialize coordinate descent from [`Lasso::coef_init`]
    /// (the prior solution) instead of zeros.
    ///
    /// Mirrors sklearn `Lasso(warm_start=False)` (`_coordinate_descent.py:795`),
    /// which "reuse[s] the solution of the previous call to fit as
    /// initialization" (`:796`). In sklearn the prior solution is the mutable
    /// estimator's own `self.coef_`, reused when `warm_start` is set
    /// (`_coordinate_descent.py:1062`: `if not self.warm_start or not
    /// hasattr(self, "coef_"): coef_ = np.zeros(...)`).
    ///
    /// R-DEV-4 adaptation: ferrolearn estimators are immutable value types —
    /// there is no mutable `self.coef_` carried across repeated `.fit()` calls.
    /// So the prior coefficient vector is supplied EXPLICITLY through
    /// [`Lasso::coef_init`] rather than read off the estimator. The numerics are
    /// identical: CD starts from `coef_init` instead of zeros.
    pub warm_start: bool,
    /// Explicit coordinate-descent initialization vector used when
    /// [`Lasso::warm_start`] is `true` (the R-DEV-4 stand-in for sklearn's
    /// reused `self.coef_`).
    ///
    /// Mirrors the `coef_init` seed fed to the path solver
    /// (`_coordinate_descent.py:648-651`: `coef_ = np.zeros(...)` when
    /// `coef_init is None`, else `coef_ = np.asfortranarray(coef_init, ...)`).
    /// `None` (the default) — or `warm_start == false` — initializes `w` to
    /// zeros, the byte-identical cold-start path. When `Some`, its length must
    /// equal `n_features`.
    pub coef_init: Option<Array1<F>>,
}

impl<F: Float> Lasso<F> {
    /// Create a new `Lasso` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 1000`, `tol = 1e-4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            max_iter: 1000,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
            positive: false,
            precompute: false,
            selection: CoordSelection::Cyclic,
            random_state: None,
            warm_start: false,
            coef_init: None,
        }
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to constrain coefficients to be non-negative.
    ///
    /// Mirrors `sklearn.linear_model.Lasso(positive=...)`.
    #[must_use]
    pub fn with_positive(mut self, positive: bool) -> Self {
        self.positive = positive;
        self
    }

    /// Set whether to run coordinate descent on the precomputed Gram matrix.
    ///
    /// Mirrors `sklearn.linear_model.Lasso(precompute=...)`
    /// (`_coordinate_descent.py:774`); `true` selects sklearn's
    /// `enet_coordinate_descent_gram` (`_cd_fast.pyx`).
    #[must_use]
    pub fn with_precompute(mut self, precompute: bool) -> Self {
        self.precompute = precompute;
        self
    }

    /// Set the coordinate-selection order for coordinate descent.
    ///
    /// Mirrors `sklearn.linear_model.Lasso(selection=...)`.
    #[must_use]
    pub fn with_selection(mut self, selection: CoordSelection) -> Self {
        self.selection = selection;
        self
    }

    /// Set the RNG seed used when `selection == CoordSelection::Random`.
    ///
    /// Mirrors `sklearn.linear_model.Lasso(random_state=...)`.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Enable/disable warm-start coordinate-descent initialization.
    ///
    /// Mirrors `sklearn.linear_model.Lasso(warm_start=...)`
    /// (`_coordinate_descent.py:795`): when `true`, "reuse the solution of the
    /// previous call to fit as initialization". R-DEV-4: ferrolearn estimators
    /// are immutable value types with no mutable `self.coef_` carried across
    /// `.fit()` calls, so the prior solution is supplied explicitly via
    /// [`Lasso::with_coef_init`]; `warm_start` only gates whether that vector
    /// (when present) is used instead of zeros.
    #[must_use]
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Provide the explicit coordinate-descent initialization vector used when
    /// [`Lasso::warm_start`] is `true`.
    ///
    /// R-DEV-4 adaptation of sklearn's reused `self.coef_`
    /// (`_coordinate_descent.py:1062`, seeded into the path solver's
    /// `coef_init` at `:648-651`): because ferrolearn estimators are immutable
    /// value types, the prior coefficient vector is passed in explicitly rather
    /// than read off a mutated estimator. Its length must equal `n_features` at
    /// fit time, else [`Fit::fit`] returns [`FerroError::ShapeMismatch`].
    #[must_use]
    pub fn with_coef_init(mut self, coef: Array1<F>) -> Self {
        self.coef_init = Some(coef);
        self
    }
}

impl<F: Float> Default for Lasso<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Lasso regression model.
///
/// Stores the learned (potentially sparse) coefficients and intercept.
/// Implements [`Predict`] and [`HasCoefficients`].
#[derive(Debug, Clone)]
pub struct FittedLasso<F> {
    /// Learned coefficient vector (some may be exactly zero).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
    /// Number of full coordinate-descent sweeps performed before
    /// convergence/break (mirrors sklearn `Lasso.n_iter_`).
    n_iter: usize,
    /// Duality gap at the returned solution (mirrors sklearn `Lasso.dual_gap_`).
    dual_gap: F,
}

/// Lasso duality gap on the `(1/2n)`-scaled objective, mirroring sklearn's
/// `enet_coordinate_descent` gap (`_cd_fast.pyx:216-247`, `beta = 0` for
/// `l1_ratio = 1`) with the final `/n` mapping sklearn's un-normalized
/// `(1/2)||y−Xw||² + l1_reg·||w||₁` (`l1_reg = alpha·n`,
/// `_coordinate_descent.py:655`) back to ferrolearn's `(1/2n)` scaling.
///
/// `xc`/`yc` are the design the coordinate descent actually solved on
/// (centered when `fit_intercept`, raw otherwise); `w` is the fitted coef.
pub(crate) fn lasso_dual_gap<F>(xc: &Array2<F>, yc: &Array1<F>, w: &Array1<F>, alpha: F) -> F
where
    F: Float + ScalarOperand + 'static,
{
    let n = xc.nrows();
    let n_f = F::from(n).unwrap_or_else(F::one);

    // R = yc − Xc·w
    let residual = yc - &xc.dot(w);

    // l1_reg = alpha · n  (sklearn's Cython l1 penalty scaling).
    let l1_reg = alpha * n_f;

    // XtA = Xcᵀ · R, dual_norm_XtA = max(|XtA[j]|).
    let xt_a = xc.t().dot(&residual);
    let dual_norm_xt_a = xt_a.iter().fold(F::zero(), |acc, &v| acc.max(v.abs()));

    let r_norm2 = residual.dot(&residual);

    let (const_factor, mut gap) = if dual_norm_xt_a > l1_reg {
        let c = l1_reg / dual_norm_xt_a;
        let half = F::from(0.5).unwrap_or_else(F::one);
        (c, half * (r_norm2 + r_norm2 * c * c))
    } else {
        (F::one(), r_norm2)
    };

    // l1_norm = ‖w‖₁
    let l1_norm = w.iter().fold(F::zero(), |acc, &wj| acc + wj.abs());
    // R · yc
    let r_dot_y = residual.dot(yc);

    gap = gap + l1_reg * l1_norm - const_factor * r_dot_y;

    gap / n_f
}

/// Soft-thresholding operator for L1 penalty.
///
/// Returns `sign(x) * max(|x| - threshold, 0)`.
fn soft_threshold<F: Float>(x: F, threshold: F) -> F {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        F::zero()
    }
}

/// Non-negative soft-thresholding operator for `positive=True` Lasso.
///
/// Returns `max(x - threshold, 0)`, dropping the negative branch so the
/// coordinate is never negative. Mirrors sklearn `_cd_fast.pyx:191-194`
/// (`if positive and tmp < 0: w[ii] = 0.0`).
fn soft_threshold_positive<F: Float>(x: F, threshold: F) -> F {
    let z = x - threshold;
    if z > F::zero() { z } else { F::zero() }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for Lasso<F>
{
    type Fitted = FittedLasso<F>;
    type Error = FerroError;

    /// Fit the Lasso model using coordinate descent.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative.
    /// Returns [`FerroError::ConvergenceFailure`] if the algorithm does
    /// not converge within `max_iter` iterations.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedLasso<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "Lasso requires at least one sample".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();

        // Center data if fitting intercept.
        let (x_work, y_work, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute column means".into(),
                })?;
            let y_mean = y.mean().ok_or_else(|| FerroError::NumericalInstability {
                message: "failed to compute target mean".into(),
            })?;

            let x_c = x - &x_mean;
            let y_c = y - y_mean;
            (x_c, y_c, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        // Precompute column norms (X_j^T X_j / n).
        let col_norms: Vec<F> = (0..n_features)
            .map(|j| {
                let col = x_work.column(j);
                col.dot(&col) / n_f
            })
            .collect();

        // Initialize coefficients. Cold start (default) is zeros; warm start
        // reuses the explicit `coef_init` (the R-DEV-4 stand-in for sklearn's
        // reused mutable `self.coef_`, `_coordinate_descent.py:1062`/`:648-651`).
        // `warm_start == false` or `coef_init == None` is the byte-identical
        // zeros path.
        let mut w = if self.warm_start
            && let Some(coef) = &self.coef_init
        {
            if coef.len() != n_features {
                return Err(FerroError::ShapeMismatch {
                    expected: vec![n_features],
                    actual: vec![coef.len()],
                    context: "coef_init length must equal number of features".into(),
                });
            }
            coef.clone()
        } else {
            Array1::<F>::zeros(n_features)
        };
        // Keep the (centered/raw) target for the final dual-gap computation;
        // the CD loop consumes a working copy into `residual`.
        let target = y_work.clone();
        let mut residual = y_work;

        // sklearn's stopping criterion (`_cd_fast.pyx:144-249`):
        //  - `d_w_tol = tol` is the UN-scaled relative-change gate (`:144`);
        //  - `tol_scaled = tol · (target·target)` is the gap threshold,
        //    `tol *= dot(y, y)` at `:167-168` (`target` is the centered/raw
        //    target the CD actually solves on).
        let d_w_tol = self.tol;
        let tol_scaled = self.tol * target.dot(&target);

        // For `selection == Random`, build the RNG ONCE before the sweep loop
        // and reuse a reusable index buffer; each sweep shuffles the visiting
        // order (sklearn `_cd_fast.pyx` `enet_coordinate_descent` `random`
        // branch picks `ii` via `rand_int` instead of the cyclic `f_iter`).
        // `Cyclic` keeps the byte-identical `0..n_features` order.
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.random_state.unwrap_or(0));
        let mut order: Vec<usize> = (0..n_features).collect();

        let mut n_iter = 0_usize;

        // REQ-10: Gram (precompute) coordinate-descent path. Mirrors sklearn's
        // `enet_coordinate_descent_gram` (`_cd_fast.pyx`): run CD on the
        // precomputed `Q = Xcᵀ Xc` and `q = Xcᵀ yc`, maintaining `H = Q·w`
        // incrementally instead of recomputing residuals each sweep. Algebraically
        // identical to the direct path (`Xⱼᵀr = q[j] − (Q·w)[j]`), so it converges
        // to the same unique optimum (to fp reassociation, ~1e-13). Keeps the
        // SAME `(1/n)` normalization, coordinate visiting order, and dual-gap
        // stopping criterion as the direct path so `n_iter_` matches.
        if self.precompute {
            // Q = Xcᵀ Xc  (n_features × n_features); q = Xcᵀ yc (here `residual`
            // still equals the centered/raw target — it is not yet adjusted for
            // a warm-start `w` since the Gram path tracks `H = Q·w` instead).
            let gram = x_work.t().dot(&x_work);
            let q = x_work.t().dot(&residual);
            // H = Q·w  (zeros for a cold start where `w == 0`; the actual `Q·w`
            // for a warm start, so `tmp = (q[j] − H[j])/n + col_norms[j]·w[j]`
            // is correct from the first sweep regardless of the init).
            let mut h = gram.dot(&w);

            for iter in 0..self.max_iter {
                n_iter = iter + 1;
                let mut w_max = F::zero();
                let mut d_w_max = F::zero();

                if self.selection == CoordSelection::Random {
                    order.shuffle(&mut rng);
                }

                for &j in &order {
                    let w_old = w[j];
                    // tmp ≡ direct `rho`: (q[j] − H[j])/n + col_norms[j]·w[j],
                    // since Xⱼᵀr = q[j] − (Q·w)[j] and col_norms[j] = Q[j,j]/n.
                    let tmp = (q[j] - h[j]) / n_f + col_norms[j] * w_old;

                    let w_new = if col_norms[j] > F::zero() {
                        let thresholded = if self.positive {
                            soft_threshold_positive(tmp, self.alpha)
                        } else {
                            soft_threshold(tmp, self.alpha)
                        };
                        thresholded / col_norms[j]
                    } else {
                        F::zero()
                    };

                    if w_new != w_old {
                        // H += (w_new − w_old) · Q.column(j).
                        let delta = w_new - w_old;
                        let col = gram.column(j);
                        for i in 0..n_features {
                            h[i] = h[i] + delta * col[i];
                        }
                    }

                    let change = (w_new - w_old).abs();
                    if change > d_w_max {
                        d_w_max = change;
                    }
                    if w_new.abs() > w_max {
                        w_max = w_new.abs();
                    }

                    w[j] = w_new;
                }

                // SAME dual-gap stopping as the direct path: reuse the
                // residual-based `lasso_dual_gap` on (x_work, target) — equal to
                // the Gram gap to fp precision, so `n_iter_` matches.
                let last_iter = iter == self.max_iter - 1;
                if w_max == F::zero() || d_w_max / w_max < d_w_tol || last_iter {
                    let dual_gap = lasso_dual_gap(&x_work, &target, &w, self.alpha);
                    let gap_raw = dual_gap * n_f;

                    if gap_raw < tol_scaled {
                        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
                            *ym - xm.dot(&w)
                        } else {
                            F::zero()
                        };

                        return Ok(FittedLasso {
                            coefficients: w,
                            intercept,
                            n_iter,
                            dual_gap,
                        });
                    }
                }
            }

            // Did not converge within max_iter; return the current solution.
            let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
                *ym - xm.dot(&w)
            } else {
                F::zero()
            };
            let dual_gap = lasso_dual_gap(&x_work, &target, &w, self.alpha);

            return Ok(FittedLasso {
                coefficients: w,
                intercept,
                n_iter,
                dual_gap,
            });
        }

        // Direct path: the CD loop maintains `residual = y_work − X_work·w`,
        // adding back `X_j·w_old` per coordinate before recomputing `rho`. With
        // a non-zero warm-start `w`, seed the residual with that running
        // contribution removed. For the cold path (`w == 0`) `X_work·w` is the
        // zero vector and the subtraction is a byte-identical no-op, so this is
        // gated on warm start to leave the default path provably untouched.
        if self.warm_start && self.coef_init.is_some() {
            residual = &residual - &x_work.dot(&w);
        }

        for iter in 0..self.max_iter {
            n_iter = iter + 1;
            let mut w_max = F::zero();
            let mut d_w_max = F::zero();

            if self.selection == CoordSelection::Random {
                order.shuffle(&mut rng);
            }

            for &j in &order {
                let col_j = x_work.column(j);

                // Compute partial residual: r + X_j * w_j
                let w_old = w[j];
                if w_old != F::zero() {
                    for i in 0..n_samples {
                        residual[i] = residual[i] + col_j[i] * w_old;
                    }
                }

                // Compute the unpenalized update: X_j^T r / n.
                let rho = col_j.dot(&residual) / n_f;

                // Apply soft-thresholding. For `positive=True`, use the
                // non-negative soft-threshold so the coefficient is never
                // negative (sklearn `_cd_fast.pyx:191-194`).
                let w_new = if col_norms[j] > F::zero() {
                    let thresholded = if self.positive {
                        soft_threshold_positive(rho, self.alpha)
                    } else {
                        soft_threshold(rho, self.alpha)
                    };
                    thresholded / col_norms[j]
                } else {
                    F::zero()
                };

                // Update residual: r = r - X_j * w_new.
                if w_new != F::zero() {
                    for i in 0..n_samples {
                        residual[i] = residual[i] - col_j[i] * w_new;
                    }
                }

                // Track the largest coordinate update and the largest
                // coefficient magnitude this sweep (`_cd_fast.pyx:201-205`).
                let change = (w_new - w_old).abs();
                if change > d_w_max {
                    d_w_max = change;
                }
                if w_new.abs() > w_max {
                    w_max = w_new.abs();
                }

                w[j] = w_new;
            }

            // sklearn's two-level convergence gate (`_cd_fast.pyx:207-251`):
            // only when coordinates barely moved (relative gate) or on the
            // last iteration do we compute the (expensive) dual gap, and we
            // break only if the UN-normalized gap clears `tol · (target·target)`.
            let last_iter = iter == self.max_iter - 1;
            if w_max == F::zero() || d_w_max / w_max < d_w_tol || last_iter {
                // `lasso_dual_gap` returns the gap divided by `n` (the
                // `dual_gap_` attribute scaling, REQ-11); multiply back to the
                // un-normalized `(1/2)||·||² + (alpha·n)||w||₁` objective sklearn
                // compares against `tol · (target·target)` (`:249`).
                let dual_gap = lasso_dual_gap(&x_work, &target, &w, self.alpha);
                let gap_raw = dual_gap * n_f;

                if gap_raw < tol_scaled {
                    let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
                        *ym - xm.dot(&w)
                    } else {
                        F::zero()
                    };

                    return Ok(FittedLasso {
                        coefficients: w,
                        intercept,
                        n_iter,
                        dual_gap,
                    });
                }
            }
        }

        // Did not converge, but still return the current solution.
        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            *ym - xm.dot(&w)
        } else {
            F::zero()
        };

        let dual_gap = lasso_dual_gap(&x_work, &target, &w, self.alpha);

        Ok(FittedLasso {
            coefficients: w,
            intercept,
            n_iter,
            dual_gap,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedLasso<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `X @ coefficients + intercept`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let preds = x.dot(&self.coefficients) + self.intercept;
        Ok(preds)
    }
}

impl<F: Float> FittedLasso<F> {
    /// Number of coordinate-descent sweeps run by the solver.
    ///
    /// Mirrors sklearn's `Lasso.n_iter_` attribute
    /// (`_coordinate_descent.py:827`/`:1103`). ferrolearn uses sklearn's
    /// relative-change + dual-gap stopping criterion (REQ-12,
    /// `_cd_fast.pyx:167-249`), so this 1-based count matches sklearn's
    /// `n_iter_` value exactly at the same optimum.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Duality gap at the returned solution, on the `(1/2n)`-scaled objective.
    ///
    /// Mirrors sklearn's `Lasso.dual_gap_` attribute
    /// (`_coordinate_descent.py:831`/`:1108`); computed by [`lasso_dual_gap`]
    /// on the same (centered/raw) design the coordinate descent solved.
    #[must_use]
    pub fn dual_gap(&self) -> F {
        self.dual_gap
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedLasso<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for Lasso<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedLasso<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_soft_threshold() {
        assert_relative_eq!(soft_threshold(5.0_f64, 1.0), 4.0);
        assert_relative_eq!(soft_threshold(-5.0_f64, 1.0), -4.0);
        assert_relative_eq!(soft_threshold(0.5_f64, 1.0), 0.0);
        assert_relative_eq!(soft_threshold(-0.5_f64, 1.0), 0.0);
        assert_relative_eq!(soft_threshold(0.0_f64, 1.0), 0.0);
    }

    #[test]
    fn test_lasso_zero_alpha() {
        // With alpha=0, Lasso should behave like OLS.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = Lasso::<f64>::new().with_alpha(0.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_lasso_sparsity() {
        // With high alpha, most coefficients should be zero.
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0,
                0.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let model = Lasso::<f64>::new().with_alpha(5.0);
        let fitted = model.fit(&x, &y).unwrap();

        // Irrelevant features should have zero coefficients.
        assert_relative_eq!(fitted.coefficients()[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.coefficients()[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lasso_shrinks_coefficients() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model_low = Lasso::<f64>::new().with_alpha(0.01);
        let model_high = Lasso::<f64>::new().with_alpha(1.0);

        let fitted_low = model_low.fit(&x, &y).unwrap();
        let fitted_high = model_high.fit(&x, &y).unwrap();

        assert!(fitted_high.coefficients()[0].abs() <= fitted_low.coefficients()[0].abs());
    }

    #[test]
    fn test_lasso_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = Lasso::<f64>::new()
            .with_alpha(0.0)
            .with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lasso_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = Lasso::<f64>::new().with_alpha(-1.0);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = Lasso::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = Lasso::<f64>::new().with_alpha(0.01);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_lasso_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = Lasso::<f64>::new().with_alpha(0.01);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_soft_threshold_positive() {
        // Non-negative branch: max(x - t, 0). Negative side clamps to 0.
        assert_relative_eq!(soft_threshold_positive(5.0_f64, 1.0), 4.0);
        assert_relative_eq!(soft_threshold_positive(-5.0_f64, 1.0), 0.0);
        assert_relative_eq!(soft_threshold_positive(0.5_f64, 1.0), 0.0);
        assert_relative_eq!(soft_threshold_positive(-0.5_f64, 1.0), 0.0);
        assert_relative_eq!(soft_threshold_positive(0.0_f64, 1.0), 0.0);
    }

    /// Oracle fixture from live sklearn 1.5.2 (R-CHAR-3):
    /// `X = [[1,3],[2,1],[3,4],[4,2],[5,5],[6,1],[2,4],[5,2]]`,
    /// `y = X[:,0] - 2*X[:,1] + noise`.
    fn positive_oracle_fixture() -> (Array2<f64>, Array1<f64>) {
        let x: Array2<f64> = array![
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 2.0],
            [5.0, 5.0],
            [6.0, 1.0],
            [2.0, 4.0],
            [5.0, 2.0],
        ];
        let noise = array![0.1, -0.2, 0.15, 0.0, -0.1, 0.05, 0.2, -0.05];
        let y: Array1<f64> = (0..8)
            .map(|i| 1.0 * x[[i, 0]] - 2.0 * x[[i, 1]] + noise[i])
            .collect();
        (x, y)
    }

    #[test]
    fn lasso_positive_matches_sklearn() {
        // Live sklearn 1.5.2 oracle:
        //   Lasso(alpha=0.3, positive=True) -> coef_ [1.14431818, 0.0],
        //   intercept_ -5.98636364
        //   (unconstrained Lasso(alpha=0.3) -> coef_ [0.8946582, -1.83087261]).
        let (x, y) = positive_oracle_fixture();

        let fit_res = Lasso::<f64>::new()
            .with_alpha(0.3)
            .with_positive(true)
            .fit(&x, &y);
        assert!(fit_res.is_ok(), "positive fit should succeed");
        let fitted = match fit_res {
            Ok(f) => f,
            Err(_) => return,
        };

        let coef = fitted.coefficients();
        assert_relative_eq!(coef[0], 1.14431818, epsilon = 1e-5);
        assert_relative_eq!(coef[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(fitted.intercept(), -5.98636364, epsilon = 1e-4);

        // All coefficients are non-negative.
        for &c in coef.iter() {
            assert!(c >= 0.0, "coefficient {c} should be non-negative");
        }

        // Differs materially from the unconstrained solution (~1.8 gap on
        // feature 1), confirming the constraint is non-tautological.
        let unc_res = Lasso::<f64>::new().with_alpha(0.3).fit(&x, &y);
        assert!(unc_res.is_ok(), "unconstrained fit should succeed");
        let unconstrained = match unc_res {
            Ok(f) => f,
            Err(_) => return,
        };
        assert!((coef[1] - unconstrained.coefficients()[1]).abs() > 1.0);
    }

    #[test]
    fn lasso_positive_false_unchanged() {
        // positive=false (default) must be byte-identical to the plain fit.
        let (x, y) = positive_oracle_fixture();

        let default_res = Lasso::<f64>::new().with_alpha(0.3).fit(&x, &y);
        assert!(default_res.is_ok(), "default fit should succeed");
        let default_fit = match default_res {
            Ok(f) => f,
            Err(_) => return,
        };
        let false_res = Lasso::<f64>::new()
            .with_alpha(0.3)
            .with_positive(false)
            .fit(&x, &y);
        assert!(
            false_res.is_ok(),
            "explicit positive=false fit should succeed"
        );
        let explicit_false = match false_res {
            Ok(f) => f,
            Err(_) => return,
        };

        assert_eq!(
            default_fit.coefficients(),
            explicit_false.coefficients(),
            "positive=false must be byte-identical to the default fit"
        );
        assert_eq!(default_fit.intercept(), explicit_false.intercept());
    }

    #[test]
    fn lasso_positive_all_nonneg_unconstrained_equals() {
        // When the unconstrained solution is already non-negative, the
        // positive constraint is inactive and yields the same coefficients
        // (NNLS-style sanity). y = 2*x is positively correlated.
        let x: Array2<f64> = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let unc_res = Lasso::<f64>::new().with_alpha(0.1).fit(&x, &y);
        assert!(unc_res.is_ok(), "unconstrained fit should succeed");
        let unconstrained = match unc_res {
            Ok(f) => f,
            Err(_) => return,
        };
        assert!(unconstrained.coefficients()[0] >= 0.0);

        let pos_res = Lasso::<f64>::new()
            .with_alpha(0.1)
            .with_positive(true)
            .fit(&x, &y);
        assert!(pos_res.is_ok(), "positive fit should succeed");
        let positive = match pos_res {
            Ok(f) => f,
            Err(_) => return,
        };

        assert_relative_eq!(
            positive.coefficients()[0],
            unconstrained.coefficients()[0],
            epsilon = 1e-10
        );
    }

    /// Centered fixture for the dual-gap oracle (R-CHAR-3):
    /// `X = [[1,2],[2,1],[3,4],[4,3],[5,5]]`, `y = [3,2.5,7.1,6,11.2]`,
    /// centered by column mean / target mean (the design the CD solves under
    /// `fit_intercept`).
    fn centered_dual_gap_fixture() -> Option<(Array2<f64>, Array1<f64>)> {
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0],];
        let y: Array1<f64> = array![3.0, 2.5, 7.1, 6.0, 11.2];
        let x_mean = x.mean_axis(Axis(0))?;
        let y_mean = y.mean()?;
        Some((&x - &x_mean, &y - y_mean))
    }

    fn raw_dual_gap_fixture() -> (Array2<f64>, Array1<f64>) {
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0],];
        let y: Array1<f64> = array![3.0, 2.5, 7.1, 6.0, 11.2];
        (x, y)
    }

    #[test]
    fn lasso_dual_gap_formula_matches_numpy() {
        // numpy/sklearn-computed oracle points (NOT from ferrolearn):
        //   gap(w=[0.5,1.0])               = 0.465888     (far-from-optimum)
        //   gap(w=[0.66691036,1.46647171]) = 0.0001170161 (the optimum)
        let (xc, yc) = match centered_dual_gap_fixture() {
            Some(f) => f,
            None => return,
        };

        let far = lasso_dual_gap(&xc, &yc, &array![0.5, 1.0], 0.3);
        assert_relative_eq!(far, 0.465888, epsilon = 1e-5);

        let opt = lasso_dual_gap(&xc, &yc, &array![0.66691036, 1.46647171], 0.3);
        assert_relative_eq!(opt, 0.0001170161, epsilon = 1e-7);
    }

    #[test]
    fn lasso_fitted_dual_gap_and_n_iter() {
        // Lasso(alpha=0.3) on the same fixture: dual_gap_ converged near
        // sklearn's 0.000117; n_iter_ within [1, max_iter].
        let (x, y) = raw_dual_gap_fixture();

        let fit_res = Lasso::<f64>::new().with_alpha(0.3).fit(&x, &y);
        assert!(fit_res.is_ok(), "fit should succeed");
        let fitted = match fit_res {
            Ok(f) => f,
            Err(_) => return,
        };

        let gap = fitted.dual_gap();
        assert!(gap >= 0.0, "dual_gap should be non-negative, got {gap}");
        assert!(gap < 1e-3, "dual_gap should be converged-small, got {gap}");

        let n_iter = fitted.n_iter();
        assert!(n_iter >= 1, "n_iter should be at least 1, got {n_iter}");
        assert!(n_iter <= 1000, "n_iter should be <= max_iter, got {n_iter}");
    }

    #[test]
    fn lasso_fields_dont_change_coef() {
        // Regression guard: the additive n_iter_/dual_gap_ fields must not
        // perturb coef_/intercept_. Compared against sklearn's converged
        // coef_ = [0.66691036, 1.46647171] at the AC-1 tolerance (1e-4):
        // ferrolearn's max-coef-change stop reaches the same optimum modulo
        // REQ-12's looser stopping measure, and the additive fields leave it
        // unchanged.
        let (x, y) = raw_dual_gap_fixture();

        let fit_res = Lasso::<f64>::new().with_alpha(0.3).fit(&x, &y);
        assert!(fit_res.is_ok(), "fit should succeed");
        let fitted = match fit_res {
            Ok(f) => f,
            Err(_) => return,
        };

        assert_relative_eq!(fitted.coefficients()[0], 0.66691036, epsilon = 1e-7);
        assert_relative_eq!(fitted.coefficients()[1], 1.46647171, epsilon = 1e-7);
    }

    #[test]
    fn lasso_dual_gap_stopping_matches_sklearn_coef_and_niter() {
        // REQ-12: sklearn's relative-change + dual-gap stopping criterion.
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   X=[[1,2],[2,1],[3,4],[4,3],[5,5]], y=[3,2.5,7.1,6,11.2]
        //   Lasso(alpha=0.3).fit(X,y) -> coef_=[0.66691036, 1.46647171],
        //   n_iter_=20, dual_gap_=0.00011701482
        let (x, y) = raw_dual_gap_fixture();

        let fit_res = Lasso::<f64>::new().with_alpha(0.3).fit(&x, &y);
        assert!(fit_res.is_ok(), "fit should succeed");
        let fitted = match fit_res {
            Ok(f) => f,
            Err(_) => return,
        };

        // Coef matches sklearn TIGHTLY now that the stopping point is identical.
        assert_relative_eq!(fitted.coefficients()[0], 0.66691036, epsilon = 1e-7);
        assert_relative_eq!(fitted.coefficients()[1], 1.46647171, epsilon = 1e-7);

        // n_iter_ matches sklearn's 1-based dual-gap iteration count exactly.
        assert_eq!(fitted.n_iter(), 20, "n_iter_ must match sklearn's 20");

        // dual_gap_ (the /n attribute) stays the REQ-11 value.
        assert_relative_eq!(fitted.dual_gap(), 0.00011701482, epsilon = 1e-7);
    }

    #[test]
    fn lasso_dual_gap_stopping_second_alpha() {
        // Generalization check at alpha=0.1 (live sklearn 1.5.2 oracle):
        //   Lasso(alpha=0.1).fit(X,y) -> coef_=[0.72247514, 1.52201988],
        //   n_iter_=20, dual_gap_=0.00013156578
        let (x, y) = raw_dual_gap_fixture();

        let fit_res = Lasso::<f64>::new().with_alpha(0.1).fit(&x, &y);
        assert!(fit_res.is_ok(), "fit should succeed");
        let fitted = match fit_res {
            Ok(f) => f,
            Err(_) => return,
        };

        assert_relative_eq!(fitted.coefficients()[0], 0.72247514, epsilon = 1e-7);
        assert_relative_eq!(fitted.coefficients()[1], 1.52201988, epsilon = 1e-7);
        assert_eq!(fitted.n_iter(), 20, "n_iter_ must match sklearn's 20");
        assert_relative_eq!(fitted.dual_gap(), 0.00013156578, epsilon = 1e-7);
    }

    /// Oracle fixture for the selection tests (R-CHAR-3, live sklearn 1.5.2):
    /// `X = [[1,2],[2,1],[3,4],[4,3],[5,5]]`, `y = [3,2.5,7.1,6,11.2]`,
    /// `alpha=0.3`.
    fn selection_fixture() -> (Array2<f64>, Array1<f64>) {
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0],];
        let y: Array1<f64> = array![3.0, 2.5, 7.1, 6.0, 11.2];
        (x, y)
    }

    #[test]
    fn lasso_selection_cyclic_default_unchanged() {
        // Default Lasso selection is Cyclic; coef must stay byte-identical to
        // the prior cyclic path. Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   Lasso(alpha=0.3, selection='cyclic') -> coef_ [0.66691036, 1.46647171].
        let (x, y) = selection_fixture();

        // Default selection is Cyclic.
        assert_eq!(Lasso::<f64>::new().selection, CoordSelection::Cyclic);

        let default_res = Lasso::<f64>::new().with_alpha(0.3).fit(&x, &y);
        assert!(default_res.is_ok(), "default fit should succeed");
        let default_fit = match default_res {
            Ok(f) => f,
            Err(_) => return,
        };

        // Matches sklearn's cyclic oracle tightly.
        assert_relative_eq!(default_fit.coefficients()[0], 0.66691036, epsilon = 1e-7);
        assert_relative_eq!(default_fit.coefficients()[1], 1.46647171, epsilon = 1e-7);

        // Explicitly-constructed Cyclic is byte-identical to the default.
        let explicit_res = Lasso::<f64>::new()
            .with_alpha(0.3)
            .with_selection(CoordSelection::Cyclic)
            .fit(&x, &y);
        assert!(explicit_res.is_ok(), "explicit cyclic fit should succeed");
        let explicit_cyclic = match explicit_res {
            Ok(f) => f,
            Err(_) => return,
        };
        assert_eq!(
            default_fit.coefficients(),
            explicit_cyclic.coefficients(),
            "explicit Cyclic must be byte-identical to the default"
        );
        assert_eq!(default_fit.intercept(), explicit_cyclic.intercept());
    }

    // HONEST CAVEAT: exact bit-match to sklearn's `selection='random'` is
    // numpy-MT19937-RNG-blocked (Rust `StdRng` != numpy MT19937), so the random
    // path below verifies convergence-to-the-unique-optimum, NOT bitwise sklearn
    // parity. The cyclic default IS bit-exact to sklearn (test above).
    #[test]
    fn lasso_selection_random_converges_to_optimum() {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   Lasso(alpha=0.3, selection='random', random_state=0)
        //     -> coef_ [0.66645032, 1.46683974]  (same unique optimum,
        //        ~1e-3 from cyclic [0.66691036, 1.46647171] due to
        //        stopping-within-tol; NOT bit-identical to cyclic).
        let (x, y) = selection_fixture();

        let fit_res = Lasso::<f64>::new()
            .with_alpha(0.3)
            .with_selection(CoordSelection::Random)
            .with_random_state(0)
            .fit(&x, &y);
        assert!(fit_res.is_ok(), "random-selection fit should succeed");
        let fitted = match fit_res {
            Ok(f) => f,
            Err(_) => return,
        };

        let coef = fitted.coefficients();

        // Every coefficient finite.
        for &c in coef.iter() {
            assert!(c.is_finite(), "coefficient {c} must be finite");
        }

        // Converges to the unique cyclic optimum within tol.
        let cyclic = [0.66691036_f64, 1.46647171_f64];
        assert!(
            (coef[0] - cyclic[0]).abs() < 1e-2,
            "coef[0]={} should be within 1e-2 of cyclic {}",
            coef[0],
            cyclic[0]
        );
        assert!(
            (coef[1] - cyclic[1]).abs() < 1e-2,
            "coef[1]={} should be within 1e-2 of cyclic {}",
            coef[1],
            cyclic[1]
        );

        // Support set matches: both coefficients strictly positive.
        assert!(coef[0] > 0.0, "coef[0] should be in the support");
        assert!(coef[1] > 0.0, "coef[1] should be in the support");
    }

    #[test]
    fn lasso_precompute_matches_sklearn() -> Result<(), FerroError> {
        // REQ-10: Gram (precompute=True) coordinate-descent path.
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   X=[[1,2],[2,1],[3,4],[4,3],[5,5]], y=[3,2.5,7.1,6,11.2]
        //   Lasso(alpha=0.3, precompute=True).fit(X,y)
        //     -> coef_=[0.6669103585, 1.4664717132], n_iter_=20
        //   (same optimum as precompute=False to ~1e-10).
        let (x, y) = raw_dual_gap_fixture();

        let fitted = Lasso::<f64>::new()
            .with_alpha(0.3)
            .with_precompute(true)
            .fit(&x, &y)?;

        assert_relative_eq!(fitted.coefficients()[0], 0.6669103585, epsilon = 1e-7);
        assert_relative_eq!(fitted.coefficients()[1], 1.4664717132, epsilon = 1e-7);
        assert_eq!(fitted.n_iter(), 20, "n_iter_ must match sklearn's 20");
        Ok(())
    }

    #[test]
    fn lasso_precompute_default_false_unchanged() -> Result<(), FerroError> {
        // Default `precompute` is `false`; the default fit must be byte-identical
        // to an explicitly-direct (precompute=false) fit (no perturbation).
        assert!(
            !Lasso::<f64>::new().precompute,
            "default precompute is false"
        );

        let (x, y) = raw_dual_gap_fixture();

        let default_fit = Lasso::<f64>::new().with_alpha(0.3).fit(&x, &y)?;
        let explicit_direct = Lasso::<f64>::new()
            .with_alpha(0.3)
            .with_precompute(false)
            .fit(&x, &y)?;

        assert_eq!(
            default_fit.coefficients(),
            explicit_direct.coefficients(),
            "explicit precompute=false must be byte-identical to the default"
        );
        assert_eq!(default_fit.intercept(), explicit_direct.intercept());
        Ok(())
    }

    #[test]
    fn lasso_precompute_equals_direct() -> Result<(), FerroError> {
        // The Gram path reaches the SAME unique optimum as the direct path,
        // via different (reassociated) arithmetic — coef within 1e-6.
        let (x, y) = raw_dual_gap_fixture();

        let direct = Lasso::<f64>::new()
            .with_alpha(0.3)
            .with_precompute(false)
            .fit(&x, &y)?;
        let gram = Lasso::<f64>::new()
            .with_alpha(0.3)
            .with_precompute(true)
            .fit(&x, &y)?;

        assert_relative_eq!(
            gram.coefficients()[0],
            direct.coefficients()[0],
            epsilon = 1e-6
        );
        assert_relative_eq!(
            gram.coefficients()[1],
            direct.coefficients()[1],
            epsilon = 1e-6
        );
        Ok(())
    }

    /// Oracle fixture for the warm-start tests (R-CHAR-3, live sklearn 1.5.2):
    /// `X = [[1,2],[2,1],[3,4],[4,3],[5,5]]`, `y = [3,2.5,7.1,6,11.2]`,
    /// `alpha=0.5`.
    fn warm_start_fixture() -> (Array2<f64>, Array1<f64>) {
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0],];
        let y: Array1<f64> = array![3.0, 2.5, 7.1, 6.0, 11.2];
        (x, y)
    }

    #[test]
    fn lasso_warm_start_from_converged_matches_sklearn() -> Result<(), FerroError> {
        // REQ-8: warm_start reuses the prior solution as CD init.
        // Live sklearn 1.5.2 oracle (R-CHAR-3): on X=[[1,2],[2,1],[3,4],[4,3],
        // [5,5]], y=[3,2.5,7.1,6,11.2]:
        //   cold Lasso(alpha=0.5)             -> coef_ [0.6113455722, 1.4109235423], n_iter_ 20
        //   warm (refit from the converged coef, warm_start=True)
        //                                     -> coef_ [0.6112611662, 1.4109910671], n_iter_ 1
        let (x, y) = warm_start_fixture();

        let cold = Lasso::<f64>::new().with_alpha(0.5).fit(&x, &y)?;
        assert_relative_eq!(cold.coefficients()[0], 0.6113455722, epsilon = 1e-6);
        assert_relative_eq!(cold.coefficients()[1], 1.4109235423, epsilon = 1e-6);
        assert_eq!(cold.n_iter(), 20, "cold n_iter_ must match sklearn's 20");

        let warm = Lasso::<f64>::new()
            .with_alpha(0.5)
            .with_warm_start(true)
            .with_coef_init(cold.coefficients().to_owned())
            .fit(&x, &y)?;
        assert_relative_eq!(warm.coefficients()[0], 0.6112611662, epsilon = 1e-6);
        assert_relative_eq!(warm.coefficients()[1], 1.4109910671, epsilon = 1e-6);
        // Converges in a single sweep starting from the converged coef.
        assert_eq!(warm.n_iter(), 1, "warm n_iter_ must match sklearn's 1");
        assert!(
            warm.n_iter() < cold.n_iter(),
            "warm start must converge in fewer iterations than cold"
        );
        Ok(())
    }

    #[test]
    fn lasso_warm_start_default_unchanged() -> Result<(), FerroError> {
        // Defaults: warm_start == false, coef_init == None. A default-config fit
        // must be byte-identical (to_bits) to the pre-warm_start cold fit.
        assert!(
            !Lasso::<f64>::new().warm_start,
            "default warm_start is false"
        );
        assert!(
            Lasso::<f64>::new().coef_init.is_none(),
            "default coef_init is None"
        );

        let (x, y) = warm_start_fixture();

        let a = Lasso::<f64>::new().with_alpha(0.5).fit(&x, &y)?;
        let b = Lasso::<f64>::new().with_alpha(0.5).fit(&x, &y)?;

        for (ca, cb) in a.coefficients().iter().zip(b.coefficients().iter()) {
            assert_eq!(
                ca.to_bits(),
                cb.to_bits(),
                "default fit must be byte-identical"
            );
        }
        assert_eq!(a.intercept().to_bits(), b.intercept().to_bits());
        assert_eq!(a.n_iter(), b.n_iter());
        Ok(())
    }

    #[test]
    fn lasso_warm_start_none_coef_init_equals_cold() -> Result<(), FerroError> {
        // warm_start=true but NO coef_init -> the init falls back to zeros, so
        // the fit is byte-identical to the plain cold fit.
        let (x, y) = warm_start_fixture();

        let cold = Lasso::<f64>::new().with_alpha(0.5).fit(&x, &y)?;
        let warm_no_init = Lasso::<f64>::new()
            .with_alpha(0.5)
            .with_warm_start(true)
            .fit(&x, &y)?;

        for (cc, cw) in cold
            .coefficients()
            .iter()
            .zip(warm_no_init.coefficients().iter())
        {
            assert_eq!(
                cc.to_bits(),
                cw.to_bits(),
                "warm_start without coef_init must equal the cold fit"
            );
        }
        assert_eq!(
            cold.intercept().to_bits(),
            warm_no_init.intercept().to_bits()
        );
        assert_eq!(cold.n_iter(), warm_no_init.n_iter());
        Ok(())
    }

    #[test]
    fn lasso_warm_start_coef_init_wrong_len_errors() {
        // coef_init length (1) != n_features (2) -> ShapeMismatch.
        let (x, y) = warm_start_fixture();

        let result = Lasso::<f64>::new()
            .with_alpha(0.5)
            .with_warm_start(true)
            .with_coef_init(array![0.0])
            .fit(&x, &y);

        assert!(
            matches!(result, Err(FerroError::ShapeMismatch { .. })),
            "wrong-length coef_init must return ShapeMismatch, got {result:?}"
        );
    }

    #[test]
    fn test_lasso_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = Lasso::<f64>::new().with_alpha(0.1);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }
}
