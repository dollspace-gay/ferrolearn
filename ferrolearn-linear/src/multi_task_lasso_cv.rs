//! Multi-task Lasso regression with built-in cross-validation for alpha
//! selection.
//!
//! This module provides [`MultiTaskLassoCV`], the `l1_ratio = 1.0`
//! specialization of [`crate::MultiTaskElasticNetCV`]: it auto-generates the
//! multi-task L21 alpha grid, runs k-fold cross-validation fitting a
//! [`crate::MultiTaskLasso`] per fold, selects the alpha minimizing mean CV MSE,
//! and refits on the full data. Unlike `MultiTaskElasticNetCV` it has NO
//! `l1_ratio` parameter (fixed at `1.0`) and exposes no `l1_ratio_` attribute.
//!
//! Mirrors `sklearn.linear_model.MultiTaskLassoCV`
//! (`sklearn/linear_model/_coordinate_descent.py:3061`,
//! `class MultiTaskLassoCV(RegressorMixin, LinearModelCV)`), whose `__init__`
//! (`:3228-3256`) drops `l1_ratio` entirely and whose `_get_estimator`
//! (`:3258`) returns a `MultiTaskLasso()`. The shared `LinearModelCV.fit`
//! machinery is the same path used by `MultiTaskElasticNetCV`; with the L1/L2
//! mixing fixed to `1.0` the inner solver reduces to the pure L21 (group-Lasso)
//! `MultiTaskLasso`. This implementation DELEGATES to
//! [`crate::MultiTaskElasticNetCV`] with `l1_ratios = [1.0]` (so the CV core is
//! written once), matching the sklearn pattern where `MultiTaskLassoCV` is the
//! `l1_ratio=1` specialization of the same `LinearModelCV` base.
//!
//! ## REQ status (per `.design/linear/lasso_cv.md`, mirrors `sklearn/linear_model/_coordinate_descent.py` @ 1.5.2)
//!
//! Scope mirrors the single-task `LassoCV` REQ-1/2 precedent and the
//! `MultiTaskElasticNetCV` core: the CORE CV path (L21 alpha grid + contiguous
//! k-fold + MSE-select + refit, fixed l1_ratio=1.0) SHIPPED; advanced attrs
//! NOT-STARTED.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (L21 alpha grid, l1_ratio=1) | SHIPPED | delegates to `MultiTaskElasticNetCV::new().with_l1_ratios(vec![1.0])`, whose `compute_alpha_max_mtenet` computes `max_j ||Xcᵀ Yc[j,:]||_2 / (n·1)` — the multi-output `_alpha_grid` (`_coordinate_descent.py:178`) with `l1_ratio=1`. Defaults `n_alphas=100`, `eps=1e-3` (`:3230`,`:3231`). Oracle (R-CHAR-3): on the 12×2 fixture `n_alphas=5,cv=3` → `alpha_=0.00525642486974421`. Non-test consumer: `pub use … MultiTaskLassoCV` in `lib.rs`. |
//! | REQ-2 (alpha CV select + refit) | SHIPPED | the delegated `Fit for MultiTaskElasticNetCV` runs contiguous k-fold CV over the single `l1_ratio=1` grid (inner `MultiTaskElasticNet(l1_ratio=1) == MultiTaskLasso`), `argmin` mean-CV-MSE select, full-data refit. `alpha_` matches the live oracle EXACTLY; `coef_`/`intercept_` within CD-stopping tol (~1e-4, shared #412). NO `l1_ratio_` exposed (sklearn `MultiTaskLassoCV` has no `l1_ratio_`, `_coordinate_descent.py:1831-1832` deletes it). Non-test consumer: `pub use … MultiTaskLassoCV`. |
//! | REQ-3 (predict / fit_intercept) | SHIPPED | `Predict for FittedMultiTaskLassoCV` = `X·coefᵀ + intercept` → `(n,t)`; `with_fit_intercept` threads into the delegate. |
//! | REQ-4 (contiguous KFold) | SHIPPED | inherited from the delegate's `kfold_indices` (sklearn non-shuffled `KFold`), so the selected `alpha_` matches sklearn. |
//! | REQ-5..N NOT-STARTED | `mse_path_`/`alphas_`/`dual_gap_`/`n_iter_` path attrs (shared single-task #433/#434), `eps` param (#435), `n_jobs`/sparse, `random_state`/`selection` (#438), ferray substrate (#439), exact `coef_` parity gated by shared CD-stopping #412 — mirroring `lasso_cv.md` advanced-attr scope. Two states only (R-DEFER-2). |
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::MultiTaskLassoCV;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let model = MultiTaskLassoCV::<f64>::new().with_n_alphas(5).with_cv(3);
//! let x: Array2<f64> = array![
//!     [1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0], [2.0, 3.0],
//!     [6.0, 1.0], [3.0, 3.0], [7.0, 2.0], [1.0, 5.0], [4.0, 6.0], [5.0, 2.0],
//! ];
//! let y: Array2<f64> = array![
//!     [3.0, 1.0], [2.5, 2.0], [7.1, 3.5], [6.0, 4.2], [11.2, 6.0], [5.0, 3.0],
//!     [9.0, 2.0], [6.5, 3.3], [12.0, 3.5], [3.0, 5.5], [8.5, 7.0], [9.5, 3.2],
//! ];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.dim(), (12, 2));
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::MultiTaskElasticNetCV;
use crate::multi_task_elastic_net_cv::FittedMultiTaskElasticNetCV;

/// Multi-task Lasso regression with built-in cross-validation for alpha
/// selection.
///
/// The `l1_ratio = 1.0` specialization of [`MultiTaskElasticNetCV`]: it
/// auto-generates the L21 alpha grid, runs k-fold CV fitting a
/// [`crate::MultiTaskLasso`] per fold, selects the alpha minimizing mean CV MSE,
/// and refits on the full data. Has NO `l1_ratio` parameter.
///
/// Mirrors `sklearn.linear_model.MultiTaskLassoCV`
/// (`_coordinate_descent.py:3061`).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct MultiTaskLassoCV<F> {
    /// Number of alphas generated when no explicit grid is supplied.
    n_alphas: usize,
    /// Number of cross-validation folds.
    cv: usize,
    /// Maximum block-coordinate-descent iterations per inner fit.
    max_iter: usize,
    /// Convergence tolerance for block coordinate descent.
    tol: F,
    /// Whether to fit per-task intercept terms.
    fit_intercept: bool,
}

impl<F: Float + FromPrimitive> MultiTaskLassoCV<F> {
    /// Create a new `MultiTaskLassoCV` with default settings.
    ///
    /// Defaults mirror `sklearn.linear_model.MultiTaskLassoCV.__init__`
    /// (`_coordinate_descent.py:3228-3256`):
    /// - `n_alphas = 100`
    /// - `cv = 5`
    /// - `max_iter = 1000`
    /// - `tol = 1e-4`
    /// - `fit_intercept = true`
    ///
    /// There is NO `l1_ratio` parameter (fixed at `1.0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_alphas: 100,
            cv: 5,
            max_iter: 1000,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            fit_intercept: true,
        }
    }

    /// Set the number of alphas generated for the path.
    #[must_use]
    pub fn with_n_alphas(mut self, n_alphas: usize) -> Self {
        self.n_alphas = n_alphas;
        self
    }

    /// Set the number of cross-validation folds (must be at least 2).
    #[must_use]
    pub fn with_cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    /// Set the maximum number of block-coordinate-descent iterations.
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

    /// Set whether to fit per-task intercept terms.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for MultiTaskLassoCV<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted multi-task Lasso model with cross-validated alpha.
///
/// Stores the selected alpha, learned coefficient matrix (shape
/// `(n_tasks, n_features)`, the sklearn `coef_` layout), and per-task intercept
/// vector. Has NO `l1_ratio_` (fixed `l1_ratio = 1.0`).
#[derive(Debug, Clone)]
pub struct FittedMultiTaskLassoCV<F> {
    /// The alpha that achieved the lowest CV error (sklearn `alpha_`).
    alpha: F,
    /// Learned coefficients, shape `(n_tasks, n_features)` (sklearn `coef_`).
    coefficients: Array2<F>,
    /// Per-task intercept vector, length `n_tasks` (sklearn `intercept_`).
    intercepts: Array1<F>,
}

impl<F: Float + Clone> FittedMultiTaskLassoCV<F> {
    /// Build from a fitted `MultiTaskElasticNetCV` (the delegate). Drops the
    /// `l1_ratio_` (sklearn `MultiTaskLassoCV` deletes it, `:1831-1832`).
    fn from_enet_cv(inner: &FittedMultiTaskElasticNetCV<F>) -> Self {
        Self {
            alpha: inner.alpha(),
            coefficients: inner.coef().clone(),
            intercepts: inner.intercept().clone(),
        }
    }

    /// Returns the alpha value selected by cross-validation (sklearn `alpha_`).
    #[must_use]
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// Borrow the learned coefficient matrix, shape `(n_tasks, n_features)`
    /// (sklearn `coef_`).
    #[must_use]
    pub fn coef(&self) -> &Array2<F> {
        &self.coefficients
    }

    /// Borrow the per-task intercept vector, length `n_tasks` (sklearn
    /// `intercept_`).
    #[must_use]
    pub fn intercept(&self) -> &Array1<F> {
        &self.intercepts
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array2<F>>
    for MultiTaskLassoCV<F>
{
    type Fitted = FittedMultiTaskLassoCV<F>;
    type Error = FerroError;

    /// Fit the `MultiTaskLassoCV` model by delegating to
    /// [`MultiTaskElasticNetCV`] with `l1_ratio` fixed to `1.0`.
    ///
    /// # Errors
    ///
    /// Forwards every error from the delegated `MultiTaskElasticNetCV::fit`
    /// (shape mismatch, invalid parameters, insufficient samples, non-finite
    /// input).
    fn fit(&self, x: &Array2<F>, y: &Array2<F>) -> Result<FittedMultiTaskLassoCV<F>, FerroError> {
        // sklearn `MultiTaskLassoCV` IS the `l1_ratio=1` specialization of the
        // same `LinearModelCV` base; `_get_estimator` returns `MultiTaskLasso()`
        // (`_coordinate_descent.py:3258`), and `MultiTaskLasso ==
        // MultiTaskElasticNet(l1_ratio=1.0)`. We delegate to the ENet-CV core so
        // the CV machinery (L21 alpha grid + contiguous folds + MSE-select +
        // refit) is written once.
        let one = F::one();
        let enet_cv = MultiTaskElasticNetCV::<F>::new()
            .with_l1_ratios(vec![one])
            .with_n_alphas(self.n_alphas)
            .with_cv(self.cv)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_fit_intercept(self.fit_intercept);

        let inner = enet_cv.fit(x, y)?;
        Ok(FittedMultiTaskLassoCV::from_enet_cv(&inner))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedMultiTaskLassoCV<F>
{
    type Output = Array2<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `X · coefᵀ + intercept` (broadcast per task column), returning
    /// an `(n_samples, n_tasks)` array.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.coefficients.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.ncols()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let mut preds = x.dot(&self.coefficients.t());
        for (k, &b) in self.intercepts.iter().enumerate() {
            let mut col = preds.column_mut(k);
            col.mapv_inplace(|v| v + b);
        }
        Ok(preds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mtlasso_cv_default_builder() {
        let m = MultiTaskLassoCV::<f64>::new();
        assert_eq!(m.n_alphas, 100);
        assert_eq!(m.cv, 5);
        assert_eq!(m.max_iter, 1000);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_mtlasso_cv_builder_setters() {
        let m = MultiTaskLassoCV::<f64>::new()
            .with_n_alphas(5)
            .with_cv(3)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert_eq!(m.n_alphas, 5);
        assert_eq!(m.cv, 3);
        assert_eq!(m.max_iter, 500);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_mtlasso_cv_shape_mismatch_error() {
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]];
        let y: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0]];
        let res = MultiTaskLassoCV::<f64>::new().fit(&x, &y);
        assert!(matches!(res, Err(FerroError::ShapeMismatch { .. })));
    }
}
