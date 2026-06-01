//! Ridge regression (L2-regularized linear regression).
//!
//! This module provides [`Ridge`], which fits a linear model with L2
//! regularization using the closed-form solution:
//!
//! ```text
//! w = (X^T X + alpha * I)^{-1} X^T y
//! ```
//!
//! The regularization parameter `alpha` controls the strength of the
//! L2 penalty, shrinking coefficients toward zero.
//!
//! ## REQ status (per `.design/linear/ridge.md`, mirrors `sklearn/linear_model/_ridge.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.Ridge` (`_ridge.py:1016`), default dense path
//! `solver='auto'`→`'cholesky'` with `fit_intercept` via centering (intercept unpenalized).
//! coef_/intercept_ match the live sklearn oracle to 1e-8 across alpha∈{0.1,1,10,100}.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (L2 cholesky fit, intercept unpenalized) | SHIPPED | `Fit for Ridge` (centering + `linalg::solve_ridge`). Consumer: `RsRidge` in `ferrolearn-python/src/regressors.rs`. |
//! | REQ-2 (predict = X·coef + intercept) | SHIPPED | `Predict for FittedRidge`. |
//! | REQ-3 (fit_intercept incl. false) | SHIPPED | `with_fit_intercept`. |
//! | REQ-4 (HasCoefficients introspection) | SHIPPED | `HasCoefficients for FittedRidge`. |
//! | REQ-5 (alpha≥0 validation; alpha=0 → OLS incl. rank-deficient min-norm) | SHIPPED | negative-alpha → `InvalidParameter`; alpha=0 singular falls back `solve_ridge` → `solve_lstsq` (ferray min-norm), mirroring sklearn cholesky→SVD (`_ridge.py:752-756`). Closed #392; test `divergence_ridge_alpha_zero_rank_deficient_min_norm`. |
//! | REQ-6 (multi-output 2-D Y → 2-D coef_) | NOT-STARTED | `FittedRidgeMulti` exists, no production consumer (#384). |
//! | REQ-7 (per-target alpha array) | NOT-STARTED | #385. |
//! | REQ-8 (solver variants + solver_) | NOT-STARTED | #386. |
//! | REQ-9 (positive=True) | NOT-STARTED | #387. |
//! | REQ-10 (max_iter/tol + n_iter_) | NOT-STARTED | #388. |
//! | REQ-11 (sample_weight) | NOT-STARTED | #389. |
//! | REQ-12 (copy_X/random_state) | NOT-STARTED | #390. |
//! | REQ-13 (ferray substrate) | NOT-STARTED | #391 (alpha=0 fallback already on ferray::linalg::lstsq; coef return tied to #359). |
//!
//! acto-critic: core L2 numerics (coef/intercept, alpha scaling, fit_intercept, f32) match the
//! live oracle; one divergence (#392, alpha=0 rank-deficient min-norm) found and fixed.
//! Two states only per goal.md R-DEFER-2.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::Ridge;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = Ridge::<f64>::new();
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferray::linalg::LinalgFloat;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::linalg;

/// Ridge regression (L2-regularized least squares).
///
/// Adds an L2 penalty to the ordinary least squares objective, which
/// shrinks coefficients toward zero and can help with multicollinearity.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct Ridge<F> {
    /// Regularization strength. Larger values specify stronger
    /// regularization.
    pub alpha: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float> Ridge<F> {
    /// Create a new `Ridge` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            fit_intercept: true,
        }
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for Ridge<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Ridge regression model.
///
/// Stores the learned coefficients and intercept. Implements [`Predict`]
/// to generate predictions and [`HasCoefficients`] for introspection.
#[derive(Debug, Clone)]
pub struct FittedRidge<F> {
    /// Learned coefficient vector (one per feature).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static>
    Fit<Array2<F>, Array1<F>> for Ridge<F>
{
    type Fitted = FittedRidge<F>;
    type Error = FerroError;

    /// Fit the Ridge regression model using Cholesky decomposition.
    ///
    /// Solves `(X^T X + alpha * I)^{-1} X^T y`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedRidge<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // `<F as num_traits::Zero>::zero()`: the `LinalgFloat` bound pulls
        // `ferray::Element` (which also defines a `zero`) into scope, so a bare
        // `F::zero()` is ambiguous between `Element` and `num_traits::Zero`.
        if self.alpha < <F as num_traits::Zero>::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "Ridge requires at least one sample".into(),
            });
        }

        if self.fit_intercept {
            // Center the data to handle the intercept.
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute column means".into(),
                })?;
            let y_mean = y.mean().ok_or_else(|| FerroError::NumericalInstability {
                message: "failed to compute target mean".into(),
            })?;

            let x_centered = x - &x_mean;
            let y_centered = y - y_mean;

            let w = linalg::solve_ridge(&x_centered, &y_centered, self.alpha)?;
            let intercept = y_mean - x_mean.dot(&w);

            Ok(FittedRidge {
                coefficients: w,
                intercept,
            })
        } else {
            let w = linalg::solve_ridge(x, y, self.alpha)?;

            Ok(FittedRidge {
                coefficients: w,
                // Disambiguate `Element::zero` vs `num_traits::Zero::zero`
                // (both in scope under the `LinalgFloat` bound).
                intercept: <F as num_traits::Zero>::zero(),
            })
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedRidge<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedRidge<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

/// Fitted multi-output Ridge regression model.
///
/// Companion to [`FittedRidge`] for the case where `Y` has multiple
/// target columns. Stores a `(n_features, n_targets)` coefficient matrix
/// and a per-target intercept vector. The Cholesky factor of
/// `X^T X + alpha * I` is computed once during [`Ridge::fit`] and shared
/// across all targets, so multi-output fitting costs the same `O(p^3)`
/// factorization as the single-output path.
#[derive(Debug, Clone)]
pub struct FittedRidgeMulti<F> {
    /// Learned coefficients, shape `(n_features, n_targets)`.
    coefficients: Array2<F>,
    /// Per-target intercept vector, length `n_targets`. Filled with
    /// zeros when `fit_intercept = false`.
    intercepts: Array1<F>,
}

impl<F: Float> FittedRidgeMulti<F> {
    /// Borrow the learned coefficient matrix `(n_features, n_targets)`.
    #[must_use]
    pub fn coefficients(&self) -> &Array2<F> {
        &self.coefficients
    }

    /// Borrow the per-target intercept vector `(n_targets,)`.
    #[must_use]
    pub fn intercepts(&self) -> &Array1<F> {
        &self.intercepts
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array2<F>>
    for Ridge<F>
{
    type Fitted = FittedRidgeMulti<F>;
    type Error = FerroError;

    /// Fit the multi-output Ridge regression model using a single
    /// shared Cholesky factorization across all `Y` columns.
    ///
    /// Solves `(X^T X + alpha * I)^{-1} X^T Y` where `Y` is
    /// `(n_samples, n_targets)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative.
    fn fit(&self, x: &Array2<F>, y: &Array2<F>) -> Result<FittedRidgeMulti<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.nrows()],
                context: "y rows must match number of samples in X".into(),
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
                context: "Ridge requires at least one sample".into(),
            });
        }

        let n_targets = y.ncols();

        if self.fit_intercept {
            // Center the data to handle the intercept (per-target).
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute column means of X".into(),
                })?;
            let y_mean = y
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute column means of Y".into(),
                })?;

            let x_centered = x - &x_mean;
            let y_centered = y - &y_mean;

            let w = linalg::solve_ridge_multi(&x_centered, &y_centered, self.alpha)?;
            // intercept[k] = y_mean[k] - x_mean · w[:, k]
            let mut intercepts = Array1::<F>::zeros(n_targets);
            for k in 0..n_targets {
                let col = w.column(k);
                let dot = x_mean.dot(&col);
                intercepts[k] = y_mean[k] - dot;
            }

            Ok(FittedRidgeMulti {
                coefficients: w,
                intercepts,
            })
        } else {
            let w = linalg::solve_ridge_multi(x, y, self.alpha)?;
            Ok(FittedRidgeMulti {
                coefficients: w,
                intercepts: Array1::<F>::zeros(n_targets),
            })
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedRidgeMulti<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `X @ coefficients + intercepts` and returns an
    /// `(n_samples, n_targets)` array.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.coefficients.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.nrows()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let mut preds = x.dot(&self.coefficients);
        // Broadcast-add per-target intercepts.
        for (k, &b) in self.intercepts.iter().enumerate() {
            let mut col = preds.column_mut(k);
            col.mapv_inplace(|v| v + b);
        }
        Ok(preds)
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for Ridge<F>
where
    F: Float + FromPrimitive + ScalarOperand + LinalgFloat + Send + Sync + 'static,
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

impl<F> FittedPipelineEstimator<F> for FittedRidge<F>
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
    fn test_ridge_no_regularization() {
        // With alpha=0, Ridge should behave like OLS.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = Ridge::<f64>::new().with_alpha(0.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_ridge_shrinks_coefficients() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model_low = Ridge::<f64>::new().with_alpha(0.01);
        let model_high = Ridge::<f64>::new().with_alpha(100.0);

        let fitted_low = model_low.fit(&x, &y).unwrap();
        let fitted_high = model_high.fit(&x, &y).unwrap();

        // Higher alpha should shrink coefficients more.
        assert!(fitted_high.coefficients()[0].abs() < fitted_low.coefficients()[0].abs());
    }

    #[test]
    fn test_ridge_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = Ridge::<f64>::new()
            .with_alpha(0.0)
            .with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ridge_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = Ridge::<f64>::new().with_alpha(-1.0);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = Ridge::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 6.0];

        let model = Ridge::<f64>::new().with_alpha(0.01);
        let fitted = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_ridge_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = Ridge::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_ridge_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = Ridge::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }

    // -- Multi-output Ridge -------------------------------------------------

    #[test]
    fn test_ridge_multi_recovers_two_targets_with_zero_alpha() {
        // Two synthetic targets sharing the same features:
        //   y1 = 2*x1 - 3*x2 + 5
        //   y2 = 3*x1 +   x2 - 1
        let n = 50;
        let x_data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let i = i as f64;
                [i / 10.0, (i / 7.0).sin()]
            })
            .collect();
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y_data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let x1 = i as f64 / 10.0;
                let x2 = (i as f64 / 7.0).sin();
                [2.0 * x1 - 3.0 * x2 + 5.0, 3.0 * x1 + x2 - 1.0]
            })
            .collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();

        let model = Ridge::<f64>::new().with_alpha(1e-8);
        let fitted: FittedRidgeMulti<f64> = model.fit(&x, &y).unwrap();

        let coef = fitted.coefficients();
        assert_eq!(coef.shape(), &[2, 2]);
        // Target 0
        assert_relative_eq!(coef[[0, 0]], 2.0, epsilon = 1e-4);
        assert_relative_eq!(coef[[1, 0]], -3.0, epsilon = 1e-4);
        // Target 1
        assert_relative_eq!(coef[[0, 1]], 3.0, epsilon = 1e-4);
        assert_relative_eq!(coef[[1, 1]], 1.0, epsilon = 1e-4);
        // Intercepts
        assert_relative_eq!(fitted.intercepts()[0], 5.0, epsilon = 1e-4);
        assert_relative_eq!(fitted.intercepts()[1], -1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_ridge_multi_no_intercept() {
        // y = X @ B with no bias; verify intercepts come out zero and
        // coefficients match the OLS solve.
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y =
            Array2::from_shape_vec((4, 2), vec![2.0, 4.0, 4.0, 8.0, 6.0, 12.0, 8.0, 16.0]).unwrap();

        let model = Ridge::<f64>::new()
            .with_alpha(0.0)
            .with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        // y[:, 0] = 2*x, y[:, 1] = 4*x
        assert_relative_eq!(fitted.coefficients()[[0, 0]], 2.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.coefficients()[[0, 1]], 4.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.intercepts()[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(fitted.intercepts()[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_ridge_multi_shrinks_with_large_alpha() {
        // Heavy regularization should pull all targets toward zero
        // coefficients (intercepts may still recover the means).
        let n = 20;
        let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64).collect()).unwrap();
        let y_data: Vec<f64> = (0..n)
            .flat_map(|i| [(i as f64) * 10.0, (i as f64) * 5.0])
            .collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();

        let model = Ridge::<f64>::new()
            .with_alpha(1e6)
            .with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        assert!(fitted.coefficients()[[0, 0]].abs() < 1.0);
        assert!(fitted.coefficients()[[0, 1]].abs() < 1.0);
    }

    #[test]
    fn test_ridge_multi_predict_round_trips_training_data() {
        // Verify Fit→Predict round-trip on the training data: the model
        // should reproduce y up to the regularization-induced bias.
        let n = 40;
        let x_data: Vec<f64> = (0..n)
            .flat_map(|i| [(i as f64) / 10.0, ((i as f64) / 3.0).cos()])
            .collect();
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y_data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let x1 = (i as f64) / 10.0;
                let x2 = ((i as f64) / 3.0).cos();
                [1.5 * x1 + 0.7 * x2 + 0.2, -0.5 * x1 + 2.0 * x2 - 1.0]
            })
            .collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();

        let model = Ridge::<f64>::new().with_alpha(1e-6);
        let fitted = model.fit(&x, &y).unwrap();
        let y_hat = fitted.predict(&x).unwrap();

        assert_eq!(y_hat.shape(), y.shape());
        // Maximum element-wise error across both targets.
        let max_err = y_hat
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-3, "max_err = {max_err}");
    }

    #[test]
    fn test_ridge_multi_shape_mismatch() {
        // 5-sample X but 3-sample Y — should error.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let model = Ridge::<f64>::new();
        let err = <Ridge<f64> as Fit<Array2<f64>, Array2<f64>>>::fit(&model, &x, &y).unwrap_err();
        matches!(err, FerroError::ShapeMismatch { .. });
    }

    #[test]
    fn test_ridge_multi_single_target_matches_single_output_path() {
        // The same data routed through the single-output and the
        // multi-output Fit impls should produce coefficients that agree
        // to within numerical noise. This pins the parallel paths from
        // drifting apart over time.
        let n = 30;
        let x = Array2::from_shape_vec(
            (n, 3),
            (0..n)
                .flat_map(|i| {
                    let i = i as f64;
                    [i / 10.0, (i / 5.0).sin(), (i / 11.0).cos()]
                })
                .collect(),
        )
        .unwrap();
        let y_1d: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|r| 2.0 * r[0] - r[1] + 0.5 * r[2] + 3.0)
            .collect();
        let y_2d = Array2::from_shape_vec((n, 1), y_1d.to_vec()).unwrap();

        let model = Ridge::<f64>::new().with_alpha(0.1);
        let single: FittedRidge<f64> = model.fit(&x, &y_1d).unwrap();
        let multi: FittedRidgeMulti<f64> = model.fit(&x, &y_2d).unwrap();

        // Coefficients agree element-wise.
        for j in 0..3 {
            assert_relative_eq!(
                single.coefficients()[j],
                multi.coefficients()[[j, 0]],
                epsilon = 1e-10
            );
        }
        // Intercepts agree.
        assert_relative_eq!(single.intercept(), multi.intercepts()[0], epsilon = 1e-10);
    }
}
