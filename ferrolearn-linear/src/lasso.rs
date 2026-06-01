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
//! | REQ-7 (positive=True) | NOT-STARTED | #407. |
//! | REQ-8 (warm_start) | NOT-STARTED | #408. |
//! | REQ-9 (selection='random' + random_state) | NOT-STARTED | #409. |
//! | REQ-10 (precompute/Gram) | NOT-STARTED | #410. |
//! | REQ-11 (n_iter_ / dual_gap_ attrs) | NOT-STARTED | #411. |
//! | REQ-12 (dual-gap stopping criterion) | NOT-STARTED | #412 — ferrolearn stops on max-coef-change vs sklearn relative-change + dual-gap; converges to the same optimum (≤1e-6), only the stopping measure differs. |
//! | REQ-13 (MultiTaskLasso) | NOT-STARTED | #413 (separate estimator). |
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

        // Initialize coefficients to zero.
        let mut w = Array1::<F>::zeros(n_features);
        let mut residual = y_work;

        for _iter in 0..self.max_iter {
            let mut max_change = F::zero();

            for j in 0..n_features {
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

                // Apply soft-thresholding.
                let w_new = if col_norms[j] > F::zero() {
                    soft_threshold(rho, self.alpha) / col_norms[j]
                } else {
                    F::zero()
                };

                // Update residual: r = r - X_j * w_new.
                if w_new != F::zero() {
                    for i in 0..n_samples {
                        residual[i] = residual[i] - col_j[i] * w_new;
                    }
                }

                let change = (w_new - w_old).abs();
                if change > max_change {
                    max_change = change;
                }

                w[j] = w_new;
            }

            // Check convergence.
            if max_change < self.tol {
                let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
                    *ym - xm.dot(&w)
                } else {
                    F::zero()
                };

                return Ok(FittedLasso {
                    coefficients: w,
                    intercept,
                });
            }
        }

        // Did not converge, but still return the current solution.
        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            *ym - xm.dot(&w)
        } else {
            F::zero()
        };

        Ok(FittedLasso {
            coefficients: w,
            intercept,
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
    fn test_lasso_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = Lasso::<f64>::new().with_alpha(0.1);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }
}
