//! ElasticNet regression (combined L1 and L2 regularization).
//!
//! This module provides [`ElasticNet`], which fits a linear model with a
//! blended L1/L2 regularization penalty using coordinate descent with
//! soft-thresholding:
//!
//! ```text
//! minimize (1/(2n)) * ||X @ w - y||^2
//!        + alpha * l1_ratio * ||w||_1
//!        + (alpha/2) * (1 - l1_ratio) * ||w||_2^2
//! ```
//!
//! When `l1_ratio = 1`, ElasticNet is equivalent to Lasso. When
//! `l1_ratio = 0`, it is equivalent to Ridge. Intermediate values produce
//! solutions that are both sparse (L1) and small in magnitude (L2).
//!
//! ## REQ status (per `.design/linear/elastic_net.md`, mirrors `sklearn/linear_model/_coordinate_descent.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.ElasticNet` (`_coordinate_descent.py:729`). CD with the L1/L2
//! split `soft_threshold(Xⱼᵀr/n, α·l1_ratio)/(XⱼᵀXⱼ/n + α·(1−l1_ratio))` ≡ sklearn's
//! `l1_reg=α·l1_ratio·n` / `l2_reg=α·(1−l1_ratio)·n`, stopped on sklearn's relative-change +
//! dual-gap criterion (REQ-13). coef_/intercept_ AND `n_iter_` match the live oracle exactly
//! (coef_ ≤1e-7); default `l1_ratio=0.5` matches sklearn.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (CD ElasticNet fit, L1/L2 split) | SHIPPED | `Fit for ElasticNet`; converged coef/intercept match oracle <1e-5 over alpha∈{0.01,0.1,1}×l1_ratio∈{0.1,0.5,0.9}. Consumers: `RsElasticNet` in `ferrolearn-python`, `ElasticNetCV`. |
//! | REQ-2 (predict) | SHIPPED | `Predict for FittedElasticNet`. |
//! | REQ-3 (fit_intercept incl. false) | SHIPPED | centering. |
//! | REQ-4 (l1_ratio mixing; =1→Lasso, =0→L2) | SHIPPED | l1_ratio=1 ≡ Lasso; l1_ratio=0 ≡ sklearn ElasticNet L2; both match oracle. |
//! | REQ-5 (L1 sparsity) | SHIPPED | exact-zero support set bit-identical to sklearn. |
//! | REQ-6 (HasCoefficients) | SHIPPED | `HasCoefficients for FittedElasticNet`. |
//! | REQ-7 (alpha/l1_ratio validation; l1_ratio∈[0,1]) | SHIPPED | matches sklearn's `_parameter_constraints` (l1_ratio=0 accepted by the class; the auto-grid error is owned by elastic_net_cv). |
//! | REQ-8 (positive=True) | SHIPPED | `positive` field + `with_positive` builder; CD loop branches on `self.positive` to `soft_threshold_positive(rho_j, alpha_l1) / denominators[j]` (non-negative soft-threshold, L2 in the denominator unchanged), mirroring sklearn's `positive` param (`_coordinate_descent.py:800`) clip `if positive and tmp < 0: w[ii] = 0.0` (`_cd_fast.pyx:191-195`). Oracle test `elasticnet_positive_matches_sklearn` → coef `[1.13685345, 0.0]`, intercept `-5.96023707` (live sklearn 1.5.2, differs from unconstrained `[0.9081389, -1.7687475]`); `elasticnet_positive_false_unchanged` regression guard. |
//! | REQ-12 (n_iter_ / dual_gap_ attrs) | SHIPPED | `FittedElasticNet<F>` carries `n_iter`/`dual_gap` fields + `n_iter()`/`dual_gap()` getters, mirroring sklearn `ElasticNet.n_iter_` (`_coordinate_descent.py:827`) and `dual_gap_` (`:831`). `fn enet_dual_gap` computes the duality gap on the CD design (centered/raw) using sklearn's `_cd_fast.pyx:216-247` formula (`l1_reg = α·l1_ratio·n`, `beta = α·(1−l1_ratio)·n`, the `XtA = XᵀR − beta·w` term + `0.5·beta·(1+const²)·‖w‖²`) with a final `/n` mapping to the `(1/2n)` objective; reduces to `lasso_dual_gap` when `l1_ratio = 1` (`beta = 0`). With REQ-13's dual-gap stopping criterion now landed, `n_iter_`'s VALUE matches sklearn exactly (`n_iter_ == 16` at alpha=0.3, `== 19` at alpha=0.1 on the fixture); `dual_gap_` matches sklearn's formula/value (`0.00010575563` at `alpha=0.3, l1_ratio=0.5`). Verification: `cargo test -p ferrolearn-linear --lib elastic_net` (`enet_dual_gap_formula_matches_numpy`, `enet_fitted_dual_gap_and_n_iter`, `enet_fields_dont_change_coef`, `enet_dual_gap_stopping_matches_sklearn_coef_and_niter`). |
//! | REQ-13 (dual-gap stopping criterion) | SHIPPED | `Fit::fit for ElasticNet` now uses sklearn's two-level criterion (`_cd_fast.pyx:167-249`): `tol_scaled = tol·(target·target)` (`:167-168`), per sweep track `w_max`/`d_w_max`, gate on `w_max==0 || d_w_max/w_max < tol || last_iter` (`:207-211`), and inside the gate break only when the UN-normalized gap `enet_dual_gap(...)·n < tol_scaled` (`:249`) — `enet_dual_gap` already carries the L2/beta term. Matches sklearn's `coef_` to ≤1e-7 and `n_iter_` exactly (16 at alpha=0.3, 19 at alpha=0.1). Verification: `cargo test -p ferrolearn-linear --lib elastic_net` (`enet_dual_gap_stopping_matches_sklearn_coef_and_niter`, `enet_dual_gap_stopping_second_alpha`). |
//! | REQ-10 (selection='random' + random_state) | SHIPPED | Reuses `pub enum CoordSelection { Cyclic, Random }` from `lasso.rs` + `pub selection`/`pub random_state` fields on `ElasticNet` with `with_selection`/`with_random_state` builders, mirroring sklearn `ElasticNet(selection=..., random_state=...)` (`_coordinate_descent.py` `__init__`). `Fit::fit`'s CD loop visits `0..n_features` in order for `Cyclic` (BYTE-IDENTICAL to the prior cyclic path, so coef_/`n_iter_`/dual-gap stay unchanged) and shuffles a reused index `Vec` each sweep for `Random` via `StdRng::seed_from_u64(random_state.unwrap_or(0))` (sklearn `_cd_fast.pyx` `enet_coordinate_descent` `random` branch picks `ii` instead of `f_iter`); per-coordinate update math + dual-gap stopping (REQ-13) are unchanged. The ElasticNet optimum is unique, so `Random` converges to the same optimum (≈3e-4 from cyclic due to stopping-within-tol). Exact bit-match to sklearn's `selection='random'` is numpy-MT19937-RNG-blocked (Rust `StdRng` ≠ numpy MT), so the random path verifies convergence-to-the-unique-optimum, not bitwise sklearn parity; the cyclic default IS bit-exact. Verification: `cargo test -p ferrolearn-linear --lib elastic_net` (`enet_selection_cyclic_default_unchanged`, `enet_selection_random_converges_to_optimum`). |
//! | REQ-9, 11, 14..15 NOT-STARTED | warm_start (#408), precompute (#410), MultiTaskElasticNet (#418), ferray substrate (#419). |
//!
//! acto-critic: NO DIVERGENCE FOUND — coef/intercept grid parity, l1_ratio=1↔Lasso, l1_ratio=0↔L2,
//! sparsity support, default l1_ratio, and a badly-scaled-feature stress all match the live oracle.
//! Two states only per goal.md R-DEFER-2.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::ElasticNet;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = ElasticNet::<f64>::new()
//!     .with_alpha(0.1)
//!     .with_l1_ratio(0.5);
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use crate::lasso::CoordSelection;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use rand::SeedableRng;
use rand::seq::SliceRandom;

/// ElasticNet regression (L1 + L2 regularized least squares).
///
/// Minimizes a combination of L1 and L2 penalties controlled by
/// `alpha` and `l1_ratio`. Uses coordinate descent with soft-thresholding
/// to handle the non-smooth L1 component.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct ElasticNet<F> {
    /// Overall regularization strength. Larger values enforce stronger
    /// regularization.
    pub alpha: F,
    /// Mix between L1 and L2 regularization.
    /// - `l1_ratio = 1.0` → pure Lasso (L1 only)
    /// - `l1_ratio = 0.0` → pure Ridge (L2 only)
    /// - `0.0 < l1_ratio < 1.0` → ElasticNet blend
    pub l1_ratio: F,
    /// Maximum number of coordinate descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum coefficient change per pass.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
    /// When `true`, constrain coefficients to be non-negative.
    pub positive: bool,
    /// Order in which coordinates are visited each coordinate-descent sweep.
    ///
    /// Mirrors sklearn `ElasticNet(selection=...)` (default `Cyclic`).
    pub selection: CoordSelection,
    /// Seed for the RNG used when `selection == CoordSelection::Random`.
    ///
    /// Mirrors sklearn `ElasticNet(random_state=...)` (default `None`). `None`
    /// falls back to seed `0`.
    pub random_state: Option<u64>,
}

impl<F: Float + FromPrimitive> ElasticNet<F> {
    /// Create a new `ElasticNet` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `l1_ratio = 0.5`, `max_iter = 1000`,
    /// `tol = 1e-4`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            l1_ratio: F::from(0.5).unwrap(),
            max_iter: 1000,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
            positive: false,
            selection: CoordSelection::Cyclic,
            random_state: None,
        }
    }

    /// Set the overall regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the L1/L2 mixing ratio.
    ///
    /// Must be in `[0.0, 1.0]`. Values outside this range will be rejected
    /// at fit time.
    #[must_use]
    pub fn with_l1_ratio(mut self, l1_ratio: F) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    /// Set the maximum number of coordinate descent iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance on maximum coefficient change.
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
    /// Mirrors `sklearn.linear_model.ElasticNet(positive=...)`.
    #[must_use]
    pub fn with_positive(mut self, positive: bool) -> Self {
        self.positive = positive;
        self
    }

    /// Set the coordinate-selection order for coordinate descent.
    ///
    /// Mirrors `sklearn.linear_model.ElasticNet(selection=...)`.
    #[must_use]
    pub fn with_selection(mut self, selection: CoordSelection) -> Self {
        self.selection = selection;
        self
    }

    /// Set the RNG seed used when `selection == CoordSelection::Random`.
    ///
    /// Mirrors `sklearn.linear_model.ElasticNet(random_state=...)`.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float + FromPrimitive> Default for ElasticNet<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted ElasticNet regression model.
///
/// Stores the learned (potentially sparse) coefficients and intercept.
/// Implements [`Predict`] and [`HasCoefficients`].
#[derive(Debug, Clone)]
pub struct FittedElasticNet<F> {
    /// Learned coefficient vector (some may be exactly zero when L1 > 0).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
    /// Number of full coordinate-descent sweeps performed before
    /// convergence/break (mirrors sklearn `ElasticNet.n_iter_`).
    n_iter: usize,
    /// Duality gap at the returned solution (mirrors sklearn `ElasticNet.dual_gap_`).
    dual_gap: F,
}

impl<F: Float> FittedElasticNet<F> {
    /// Returns the intercept (bias) term learned during fitting.
    pub fn intercept(&self) -> F {
        self.intercept
    }

    /// Number of coordinate-descent sweeps run by the solver.
    ///
    /// Mirrors sklearn's `ElasticNet.n_iter_` attribute
    /// (`_coordinate_descent.py:827`). ferrolearn uses sklearn's relative-change
    /// and dual-gap stopping criterion (REQ-13, `_cd_fast.pyx:167-249`), so this
    /// 1-based count matches sklearn's `n_iter_` value exactly at the same
    /// optimum.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Duality gap at the returned solution, on the `(1/2n)`-scaled objective.
    ///
    /// Mirrors sklearn's `ElasticNet.dual_gap_` attribute
    /// (`_coordinate_descent.py:831`); computed by [`enet_dual_gap`] on the same
    /// (centered/raw) design the coordinate descent solved.
    #[must_use]
    pub fn dual_gap(&self) -> F {
        self.dual_gap
    }
}

/// ElasticNet duality gap on the `(1/2n)`-scaled objective, mirroring sklearn's
/// `enet_coordinate_descent` gap (`_cd_fast.pyx:216-247`) with the final `/n`
/// mapping sklearn's un-normalized `(1/2)||y−Xw||² + l1_reg·||w||₁ +
/// (1/2)·l2_reg·||w||²` (`l1_reg = alpha·l1_ratio·n`, `l2_reg =
/// alpha·(1−l1_ratio)·n`, `_coordinate_descent.py:655-656`) back to ferrolearn's
/// `(1/2n)` scaling. Reduces to the Lasso gap when `l1_ratio = 1` (`beta = 0`).
///
/// `xc`/`yc` are the design the coordinate descent actually solved on
/// (centered when `fit_intercept`, raw otherwise); `w` is the fitted coef.
pub(crate) fn enet_dual_gap<F>(
    xc: &Array2<F>,
    yc: &Array1<F>,
    w: &Array1<F>,
    alpha: F,
    l1_ratio: F,
) -> F
where
    F: Float + ScalarOperand + 'static,
{
    let n = xc.nrows();
    let n_f = F::from(n).unwrap_or_else(F::one);

    // R = yc − Xc·w
    let residual = yc - &xc.dot(w);

    // l1_reg = alpha · l1_ratio · n ; beta = alpha · (1 − l1_ratio) · n.
    let l1_reg = alpha * l1_ratio * n_f;
    let beta = alpha * (F::one() - l1_ratio) * n_f;

    // XtA = Xcᵀ·R − beta·w ; dual_norm_XtA = max(|XtA[j]|).
    let xt_a = xc.t().dot(&residual) - &(w * beta);
    let dual_norm_xt_a = xt_a.iter().fold(F::zero(), |acc, &v| acc.max(v.abs()));

    let r_norm2 = residual.dot(&residual);
    let w_norm2 = w.dot(w);

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
    let half = F::from(0.5).unwrap_or_else(F::one);

    gap = gap + l1_reg * l1_norm - const_factor * r_dot_y
        + half * beta * (F::one() + const_factor * const_factor) * w_norm2;

    gap / n_f
}

/// Soft-thresholding operator used in coordinate descent for L1 penalty.
///
/// Returns `sign(x) * max(|x| - threshold, 0)`.
#[inline]
fn soft_threshold<F: Float>(x: F, threshold: F) -> F {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        F::zero()
    }
}

/// Non-negative soft-thresholding operator for `positive=True` ElasticNet.
///
/// Returns `max(x - threshold, 0)`, dropping the negative branch so the
/// coordinate is never negative. Mirrors sklearn `_cd_fast.pyx:191-195`
/// (`if positive and tmp < 0: w[ii] = 0.0`); the L2 term lives in the
/// denominator and is unaffected.
#[inline]
fn soft_threshold_positive<F: Float>(x: F, threshold: F) -> F {
    let z = x - threshold;
    if z > F::zero() { z } else { F::zero() }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for ElasticNet<F>
{
    type Fitted = FittedElasticNet<F>;
    type Error = FerroError;

    /// Fit the ElasticNet model using coordinate descent.
    ///
    /// Centers the data if `fit_intercept` is `true`, then alternates
    /// coordinate updates using the soft-threshold rule with L2 scaling.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers
    ///   of samples.
    /// - [`FerroError::InvalidParameter`] if `alpha` is negative, `l1_ratio`
    ///   is outside `[0, 1]`, or `tol` is non-positive.
    /// - [`FerroError::InsufficientSamples`] if `n_samples == 0`.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedElasticNet<F>, FerroError> {
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

        if self.l1_ratio < F::zero() || self.l1_ratio > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "l1_ratio".into(),
                reason: "must be in [0, 1]".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "ElasticNet requires at least one sample".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();

        // Center data when fitting intercept.
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

        // Precompute per-column X_j^T X_j / n (used as denominator).
        let col_norms: Vec<F> = (0..n_features)
            .map(|j| {
                let col = x_work.column(j);
                col.dot(&col) / n_f
            })
            .collect();

        // L1 and L2 penalty strengths split from alpha/l1_ratio.
        let alpha_l1 = self.alpha * self.l1_ratio;
        let alpha_l2 = self.alpha * (F::one() - self.l1_ratio);

        // Effective denominator per column: (X_j^T X_j / n) + alpha_l2.
        let denominators: Vec<F> = col_norms.iter().map(|&cn| cn + alpha_l2).collect();

        let mut w = Array1::<F>::zeros(n_features);
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
        // `Cyclic` keeps the byte-identical `0..n_features` order, so the
        // per-coordinate update math AND the dual-gap stopping criterion
        // (REQ-13) stay unchanged.
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.random_state.unwrap_or(0));
        let mut order: Vec<usize> = (0..n_features).collect();

        let mut n_iter = 0_usize;
        for iter in 0..self.max_iter {
            n_iter = iter + 1;
            let mut w_max = F::zero();
            let mut d_w_max = F::zero();

            if self.selection == CoordSelection::Random {
                order.shuffle(&mut rng);
            }

            for &j in &order {
                let col_j = x_work.column(j);
                let w_old = w[j];

                // Add back contribution of current coefficient j to residual.
                if w_old != F::zero() {
                    for i in 0..n_samples {
                        residual[i] = residual[i] + col_j[i] * w_old;
                    }
                }

                // Unpenalized correlation: X_j^T r / n.
                let rho_j = col_j.dot(&residual) / n_f;

                // Apply soft-threshold for L1, then divide by (col_norm + alpha_l2).
                // For `positive=True`, use the non-negative soft-threshold so the
                // coefficient is never negative (sklearn `_cd_fast.pyx:191-195`); the
                // L2 term in the denominator is unchanged.
                let w_new = if denominators[j] > F::zero() {
                    let thresholded = if self.positive {
                        soft_threshold_positive(rho_j, alpha_l1)
                    } else {
                        soft_threshold(rho_j, alpha_l1)
                    };
                    thresholded / denominators[j]
                } else {
                    F::zero()
                };

                // Update residual with new coefficient.
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
                // `enet_dual_gap` returns the gap divided by `n` (the
                // `dual_gap_` attribute scaling, REQ-12); multiply back to the
                // un-normalized objective sklearn compares against
                // `tol · (target·target)` (`:249`). The L2/beta term is already
                // included in `enet_dual_gap`.
                let dual_gap = enet_dual_gap(&x_work, &target, &w, self.alpha, self.l1_ratio);
                let gap_raw = dual_gap * n_f;

                if gap_raw < tol_scaled {
                    let intercept = compute_intercept(&x_mean, &y_mean, &w);
                    return Ok(FittedElasticNet {
                        coefficients: w,
                        intercept,
                        n_iter,
                        dual_gap,
                    });
                }
            }
        }

        // Return best solution found even without full convergence.
        let intercept = compute_intercept(&x_mean, &y_mean, &w);
        let dual_gap = enet_dual_gap(&x_work, &target, &w, self.alpha, self.l1_ratio);
        Ok(FittedElasticNet {
            coefficients: w,
            intercept,
            n_iter,
            dual_gap,
        })
    }
}

/// Compute intercept from the centered means and fitted coefficients.
fn compute_intercept<F: Float + 'static>(
    x_mean: &Option<Array1<F>>,
    y_mean: &Option<F>,
    w: &Array1<F>,
) -> F {
    if let (Some(xm), Some(ym)) = (x_mean, y_mean) {
        *ym - xm.dot(w)
    } else {
        F::zero()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedElasticNet<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedElasticNet<F> {
    /// Returns the learned coefficient vector.
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    /// Returns the learned intercept term.
    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for ElasticNet<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    /// Fit the model and return it as a boxed pipeline estimator.
    ///
    /// # Errors
    ///
    /// Propagates any [`FerroError`] from `fit`.
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedElasticNet<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    /// Generate predictions via the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates any [`FerroError`] from `predict`.
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // ---- soft_threshold helpers ----

    #[test]
    fn test_soft_threshold_positive() {
        assert_relative_eq!(soft_threshold(5.0_f64, 1.0), 4.0);
    }

    #[test]
    fn test_soft_threshold_negative() {
        assert_relative_eq!(soft_threshold(-5.0_f64, 1.0), -4.0);
    }

    #[test]
    fn test_soft_threshold_within_band() {
        assert_relative_eq!(soft_threshold(0.5_f64, 1.0), 0.0);
        assert_relative_eq!(soft_threshold(-0.5_f64, 1.0), 0.0);
        assert_relative_eq!(soft_threshold(0.0_f64, 1.0), 0.0);
    }

    // ---- Builder ----

    #[test]
    fn test_default_builder() {
        let m = ElasticNet::<f64>::new();
        assert_relative_eq!(m.alpha, 1.0);
        assert_relative_eq!(m.l1_ratio, 0.5);
        assert_eq!(m.max_iter, 1000);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_builder_setters() {
        let m = ElasticNet::<f64>::new()
            .with_alpha(0.5)
            .with_l1_ratio(0.2)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert_relative_eq!(m.alpha, 0.5);
        assert_relative_eq!(m.l1_ratio, 0.2);
        assert_eq!(m.max_iter, 500);
        assert!(!m.fit_intercept);
    }

    // ---- Validation errors ----

    #[test]
    fn test_negative_alpha_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = ElasticNet::<f64>::new().with_alpha(-1.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_l1_ratio_out_of_range_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = ElasticNet::<f64>::new().with_l1_ratio(1.5).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        let result = ElasticNet::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    // ---- Correctness ----

    #[test]
    fn test_lasso_limit_l1_ratio_one() {
        // With l1_ratio=1, ElasticNet should behave like Lasso.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = ElasticNet::<f64>::new().with_alpha(0.0).with_l1_ratio(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_ridge_limit_l1_ratio_zero() {
        // With l1_ratio=0 and alpha=0, should recover OLS.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = ElasticNet::<f64>::new().with_alpha(0.0).with_l1_ratio(0.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_sparsity_with_high_l1_ratio() {
        // High alpha with l1_ratio=1 should zero out irrelevant features.
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0,
                0.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let model = ElasticNet::<f64>::new().with_alpha(5.0).with_l1_ratio(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.coefficients()[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_higher_alpha_shrinks_more() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let low = ElasticNet::<f64>::new()
            .with_alpha(0.01)
            .with_l1_ratio(0.5)
            .fit(&x, &y)
            .unwrap();
        let high = ElasticNet::<f64>::new()
            .with_alpha(2.0)
            .with_l1_ratio(0.5)
            .fit(&x, &y)
            .unwrap();

        assert!(high.coefficients()[0].abs() <= low.coefficients()[0].abs());
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = ElasticNet::<f64>::new()
            .with_alpha(0.0)
            .with_l1_ratio(0.5)
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_correct_length() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = ElasticNet::<f64>::new()
            .with_alpha(0.01)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x_train = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = ElasticNet::<f64>::new()
            .with_alpha(0.01)
            .fit(&x_train, &y)
            .unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_coefficients_length() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = ElasticNet::<f64>::new()
            .with_alpha(0.1)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = ElasticNet::<f64>::new().with_alpha(0.01);
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- positive=True (REQ-8) ----

    #[test]
    fn test_soft_threshold_positive_helper() {
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
    fn elasticnet_positive_matches_sklearn() {
        // Live sklearn 1.5.2 oracle:
        //   ElasticNet(alpha=0.3, l1_ratio=0.5, positive=True)
        //     -> coef_ [1.13685345, 0.0], intercept_ -5.96023707
        //   (unconstrained ElasticNet(alpha=0.3, l1_ratio=0.5)
        //     -> coef_ [0.9081389, -1.7687475], intercept_ -0.29568051).
        let (x, y) = positive_oracle_fixture();

        let fit_res = ElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
            .with_positive(true)
            .fit(&x, &y);
        assert!(fit_res.is_ok(), "positive fit should succeed");
        let fitted = match fit_res {
            Ok(f) => f,
            Err(_) => return,
        };

        let coef = fitted.coefficients();
        assert_relative_eq!(coef[0], 1.13685345, epsilon = 1e-5);
        assert_relative_eq!(coef[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(fitted.intercept(), -5.96023707, epsilon = 1e-4);

        // All coefficients are non-negative.
        for &c in coef.iter() {
            assert!(c >= 0.0, "coefficient {c} should be non-negative");
        }

        // Differs materially from the unconstrained solution (~1.77 gap on
        // feature 1), confirming the constraint is non-tautological.
        let unc_res = ElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
            .fit(&x, &y);
        assert!(unc_res.is_ok(), "unconstrained fit should succeed");
        let unconstrained = match unc_res {
            Ok(f) => f,
            Err(_) => return,
        };
        assert!((coef[1] - unconstrained.coefficients()[1]).abs() > 1.0);
    }

    #[test]
    fn elasticnet_positive_false_unchanged() {
        // positive=false (default) must be byte-identical to the plain fit.
        let (x, y) = positive_oracle_fixture();

        let default_res = ElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
            .fit(&x, &y);
        assert!(default_res.is_ok(), "default fit should succeed");
        let default_fit = match default_res {
            Ok(f) => f,
            Err(_) => return,
        };
        let false_res = ElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
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

    // ---- n_iter_ / dual_gap_ (REQ-12) ----

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
    fn enet_dual_gap_formula_matches_numpy() {
        // numpy/sklearn-computed oracle points (NOT from ferrolearn),
        // alpha=0.3, l1_ratio=0.5:
        //   gap(w=[0.5,1.0])                     = 0.6369296296 (far-from-optimum)
        //   gap(w=[0.77323348,1.35480299])       = 0.0001057556 (the optimum)
        let (xc, yc) = match centered_dual_gap_fixture() {
            Some(f) => f,
            None => return,
        };

        let far = enet_dual_gap(&xc, &yc, &array![0.5, 1.0], 0.3, 0.5);
        assert_relative_eq!(far, 0.6369296296, epsilon = 1e-5);

        let opt = enet_dual_gap(&xc, &yc, &array![0.77323348, 1.35480299], 0.3, 0.5);
        assert_relative_eq!(opt, 0.0001057556, epsilon = 1e-7);
    }

    #[test]
    fn enet_fitted_dual_gap_and_n_iter() {
        // ElasticNet(alpha=0.3, l1_ratio=0.5) on the same fixture: dual_gap_
        // converged near sklearn's 0.000106; n_iter_ within [1, max_iter].
        let (x, y) = raw_dual_gap_fixture();

        let fit_res = ElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
            .fit(&x, &y);
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
    fn enet_fields_dont_change_coef() {
        // Regression guard: the additive n_iter_/dual_gap_ fields must not
        // perturb coef_/intercept_. Compared against sklearn's converged
        // coef_ = [0.77323348, 1.35480299]: with REQ-13's dual-gap stopping
        // criterion the stop point is identical to sklearn, so the comparison
        // is tight (1e-7) — matching sklearn BETTER, never loosened.
        let (x, y) = raw_dual_gap_fixture();

        let fit_res = ElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
            .fit(&x, &y);
        assert!(fit_res.is_ok(), "fit should succeed");
        let fitted = match fit_res {
            Ok(f) => f,
            Err(_) => return,
        };

        assert_relative_eq!(fitted.coefficients()[0], 0.77323348, epsilon = 1e-7);
        assert_relative_eq!(fitted.coefficients()[1], 1.35480299, epsilon = 1e-7);
    }

    #[test]
    fn enet_dual_gap_stopping_matches_sklearn_coef_and_niter() {
        // REQ-13: sklearn's relative-change + dual-gap stopping criterion.
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   X=[[1,2],[2,1],[3,4],[4,3],[5,5]], y=[3,2.5,7.1,6,11.2]
        //   ElasticNet(alpha=0.3, l1_ratio=0.5).fit(X,y)
        //     -> coef_=[0.77323348, 1.35480299], n_iter_=16,
        //        dual_gap_=0.00010575563
        let (x, y) = raw_dual_gap_fixture();

        let fit_res = ElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
            .fit(&x, &y);
        assert!(fit_res.is_ok(), "fit should succeed");
        let fitted = match fit_res {
            Ok(f) => f,
            Err(_) => return,
        };

        // Coef matches sklearn TIGHTLY now that the stopping point is identical.
        assert_relative_eq!(fitted.coefficients()[0], 0.77323348, epsilon = 1e-7);
        assert_relative_eq!(fitted.coefficients()[1], 1.35480299, epsilon = 1e-7);

        // n_iter_ matches sklearn's 1-based dual-gap iteration count exactly.
        assert_eq!(fitted.n_iter(), 16, "n_iter_ must match sklearn's 16");

        // dual_gap_ (the /n attribute) stays the REQ-12 value.
        assert_relative_eq!(fitted.dual_gap(), 0.00010575563, epsilon = 1e-7);
    }

    #[test]
    fn enet_dual_gap_stopping_second_alpha() {
        // Generalization check at alpha=0.1 (live sklearn 1.5.2 oracle):
        //   ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X,y)
        //     -> coef_=[0.76514609, 1.47598354], n_iter_=19,
        //        dual_gap_=9.422349e-05
        let (x, y) = raw_dual_gap_fixture();

        let fit_res = ElasticNet::<f64>::new()
            .with_alpha(0.1)
            .with_l1_ratio(0.5)
            .fit(&x, &y);
        assert!(fit_res.is_ok(), "fit should succeed");
        let fitted = match fit_res {
            Ok(f) => f,
            Err(_) => return,
        };

        assert_relative_eq!(fitted.coefficients()[0], 0.76514609, epsilon = 1e-7);
        assert_relative_eq!(fitted.coefficients()[1], 1.47598354, epsilon = 1e-7);
        assert_eq!(fitted.n_iter(), 19, "n_iter_ must match sklearn's 19");
        assert_relative_eq!(fitted.dual_gap(), 9.422349e-05, epsilon = 1e-7);
    }

    // ---- selection='random' + random_state (REQ-10) ----

    /// Oracle fixture for the selection tests (R-CHAR-3, live sklearn 1.5.2):
    /// `X = [[1,2],[2,1],[3,4],[4,3],[5,5]]`, `y = [3,2.5,7.1,6,11.2]`,
    /// `alpha=0.3`, `l1_ratio=0.5`.
    fn selection_fixture() -> (Array2<f64>, Array1<f64>) {
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0],];
        let y: Array1<f64> = array![3.0, 2.5, 7.1, 6.0, 11.2];
        (x, y)
    }

    #[test]
    fn enet_selection_cyclic_default_unchanged() {
        // Default ElasticNet selection is Cyclic; coef must stay byte-identical
        // to the prior cyclic path. Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   ElasticNet(alpha=0.3, l1_ratio=0.5, selection='cyclic')
        //     -> coef_ [0.77323348, 1.35480299].
        let (x, y) = selection_fixture();

        // Default selection is Cyclic.
        assert_eq!(ElasticNet::<f64>::new().selection, CoordSelection::Cyclic);

        let default_res = ElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
            .fit(&x, &y);
        assert!(default_res.is_ok(), "default fit should succeed");
        let default_fit = match default_res {
            Ok(f) => f,
            Err(_) => return,
        };

        // Matches sklearn's cyclic oracle tightly.
        assert_relative_eq!(default_fit.coefficients()[0], 0.77323348, epsilon = 1e-7);
        assert_relative_eq!(default_fit.coefficients()[1], 1.35480299, epsilon = 1e-7);

        // Explicitly-constructed Cyclic is byte-identical to the default.
        let explicit_res = ElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
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
    fn enet_selection_random_converges_to_optimum() {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   ElasticNet(alpha=0.3, l1_ratio=0.5, selection='random',
        //              random_state=0)
        //     -> coef_ [0.77289352, 1.35505598]  (same unique optimum,
        //        ~3e-4 from cyclic [0.77323348, 1.35480299] due to
        //        stopping-within-tol; NOT bit-identical to cyclic).
        let (x, y) = selection_fixture();

        let fit_res = ElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
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
        let cyclic = [0.77323348_f64, 1.35480299_f64];
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
}
