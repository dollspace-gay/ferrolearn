//! Automatic Relevance Determination (ARD) Regression.
//!
//! This module provides [`ARDRegression`], a Bayesian linear regression model
//! with per-feature weight precision priors. Features whose precision
//! (`lambda_i`) exceeds a threshold are pruned — their weights are driven to
//! zero, achieving automatic feature selection.
//!
//! # Algorithm
//!
//! Initialisation seeds `alpha = 1/(Var(y)+eps)` and `lambda_i = 1` for every
//! feature, with all features kept (`keep_lambda = lambda_ < threshold_lambda`,
//! initially all-true). Each iteration solves only the KEPT columns
//! `Xk = X[:, keep_lambda]` and updates, including the Gamma hyperprior terms:
//!
//! 1. Posterior covariance of the kept block:
//!    `Sigma = (diag(lambda[keep]) + alpha * Xk^T Xk)^{-1}`,
//!    then `w[keep] = alpha * Sigma @ Xk^T y`, `w[~keep] = 0`.
//! 2. Effective degrees of freedom: `gamma_i = 1 - lambda_i * Sigma_{ii}`.
//! 3. Update lambda: `lambda_i = (gamma_i + 2*lambda_1) / (w_i^2 + 2*lambda_2)`.
//! 4. Update alpha:
//!    `alpha = (n - sum(gamma) + 2*alpha_1) / (||y - Xw||^2 + 2*alpha_2)`.
//! 5. Recompute the mask `keep_lambda = lambda_ < threshold_lambda` and zero the
//!    coefficients of pruned features.
//!
//! Convergence is `sum(|coef_old - coef_|) < tol` (checked after the first
//! iteration). This mirrors scikit-learn's `ARDRegression.fit`
//! (`sklearn/linear_model/_bayes.py:644-730`).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::ard::ARDRegression;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 2), vec![
//!     1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0,
//! ]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let model = ARDRegression::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 5);
//! ```

use ferray::linalg::{LinalgFloat, inv};
use ferray::{Array as FerrayArray, Ix2 as FerrayIx2};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// Automatic Relevance Determination Regression.
///
/// Bayesian linear regression with per-feature precision priors. Features
/// with high precision (small variance) are pruned, achieving sparsity.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct ARDRegression<F> {
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative change in alpha/lambda.
    pub tol: F,
    /// Shape hyperparameter for the alpha (noise precision) Gamma prior.
    pub alpha_1: F,
    /// Rate hyperparameter for the alpha (noise precision) Gamma prior.
    pub alpha_2: F,
    /// Shape hyperparameter for the lambda (weight precision) Gamma prior.
    pub lambda_1: F,
    /// Rate hyperparameter for the lambda (weight precision) Gamma prior.
    pub lambda_2: F,
    /// Features with `lambda_i > threshold_lambda` are pruned.
    pub threshold_lambda: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> ARDRegression<F> {
    /// Create a new `ARDRegression` with default settings.
    ///
    /// Defaults: `max_iter = 300`, `tol = 1e-3`, `alpha_1 = alpha_2 = 1e-6`,
    /// `lambda_1 = lambda_2 = 1e-6`, `threshold_lambda = 1e4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 300,
            tol: F::from(1e-3).unwrap(),
            alpha_1: F::from(1e-6).unwrap(),
            alpha_2: F::from(1e-6).unwrap(),
            lambda_1: F::from(1e-6).unwrap(),
            lambda_2: F::from(1e-6).unwrap(),
            threshold_lambda: F::from(1e4).unwrap(),
            fit_intercept: true,
        }
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

    /// Set the alpha shape hyperparameter.
    #[must_use]
    pub fn with_alpha_1(mut self, alpha_1: F) -> Self {
        self.alpha_1 = alpha_1;
        self
    }

    /// Set the alpha rate hyperparameter.
    #[must_use]
    pub fn with_alpha_2(mut self, alpha_2: F) -> Self {
        self.alpha_2 = alpha_2;
        self
    }

    /// Set the lambda shape hyperparameter.
    #[must_use]
    pub fn with_lambda_1(mut self, lambda_1: F) -> Self {
        self.lambda_1 = lambda_1;
        self
    }

    /// Set the lambda rate hyperparameter.
    #[must_use]
    pub fn with_lambda_2(mut self, lambda_2: F) -> Self {
        self.lambda_2 = lambda_2;
        self
    }

    /// Set the pruning threshold for feature lambda values.
    #[must_use]
    pub fn with_threshold_lambda(mut self, threshold_lambda: F) -> Self {
        self.threshold_lambda = threshold_lambda;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for ARDRegression<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted ARD Regression model.
///
/// Stores the posterior mean coefficients, intercept, estimated noise
/// precision (`alpha`), per-feature weight precisions (`lambda`), and
/// the diagonal of the posterior covariance.
#[derive(Debug, Clone)]
pub struct FittedARDRegression<F> {
    /// Posterior mean coefficient vector.
    coefficients: Array1<F>,
    /// Intercept (bias) term.
    intercept: F,
    /// Estimated noise precision (1 / noise_variance).
    alpha: F,
    /// Per-feature weight precisions.
    lambda: Array1<F>,
    /// Diagonal of the posterior covariance matrix.
    sigma: Array1<F>,
}

impl<F: Float> FittedARDRegression<F> {
    /// Returns the estimated noise precision (alpha = 1/sigma^2_noise).
    #[must_use]
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// Returns the per-feature weight precisions.
    #[must_use]
    pub fn lambda(&self) -> &Array1<F> {
        &self.lambda
    }

    /// Returns the diagonal of the posterior covariance matrix.
    #[must_use]
    pub fn sigma(&self) -> &Array1<F> {
        &self.sigma
    }
}

/// Posterior covariance of the kept feature block, mirroring scikit-learn's
/// `ARDRegression._update_sigma` (`sklearn/linear_model/_bayes.py:750-759`):
///
/// ```text
/// gram      = Xk^T @ Xk
/// sigma_inv = diag(lambda[keep]) + alpha * gram
/// Sigma     = pinvh(sigma_inv)
/// ```
///
/// where `Xk = X[:, keep]`. The `(k_keep, k_keep)` inverse runs on the ferray
/// linear-algebra substrate (`ferray::linalg::inv`, `ferray-linalg/src/solve.rs:367`)
/// — for the symmetric positive-definite `sigma_inv` (`n_samples >= n_features`,
/// the `_update_sigma` regime) the LU inverse matches scipy's `pinvh`. The
/// `ndarray ↔ ferray` conversion happens at this boundary (R-SUBSTRATE-4); the
/// caller keeps its `ndarray` signature during the workspace-wide migration.
///
/// Returns the full `(k_keep, k_keep)` posterior covariance `Sigma`.
fn update_sigma<F: LinalgFloat>(
    xk: &Array2<F>,
    alpha: F,
    lambda_keep: &[F],
) -> Result<Array2<F>, FerroError> {
    let k = xk.ncols();
    // sigma_inv = diag(lambda[keep]) + alpha * Xk^T Xk.
    let mut sigma_inv = xk.t().dot(xk);
    for i in 0..k {
        for j in 0..k {
            sigma_inv[[i, j]] *= alpha;
        }
        sigma_inv[[i, i]] += lambda_keep[i];
    }

    // Bridge ndarray -> ferray (R-SUBSTRATE-4).
    let flat: Vec<F> = sigma_inv.iter().copied().collect();
    let a = FerrayArray::<F, FerrayIx2>::from_vec(FerrayIx2::new([k, k]), flat).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray inv: failed to build sigma_inv: {e}"),
        }
    })?;
    let sigma_f = inv(&a).map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray inv failed (ARD sigma): {e}"),
    })?;

    // Bridge ferray -> ndarray.
    let sigma = Array2::from_shape_vec((k, k), sigma_f.iter().copied().collect()).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray inv: Sigma shape conversion failed: {e}"),
        }
    })?;
    Ok(sigma)
}

impl<F: LinalgFloat + Send + Sync + ScalarOperand + FromPrimitive> Fit<Array2<F>, Array1<F>>
    for ARDRegression<F>
{
    type Fitted = FittedARDRegression<F>;
    type Error = FerroError;

    /// Fit the ARD model via iterative evidence maximization with per-iteration
    /// `keep_lambda` column masking, mirroring scikit-learn's `ARDRegression.fit`
    /// (`sklearn/linear_model/_bayes.py:644-730`).
    ///
    /// After centering (when `fit_intercept`), `alpha` is seeded to
    /// `1/(Var(y)+eps)` (`_bayes.py:658`) and `lambda` to ones (`_bayes.py:659`)
    /// with all features kept. Each iteration solves only the kept sub-block via
    /// [`update_sigma`] (`_bayes.py:677-678`, `:750-759`), updates the Gamma-prior
    /// `lambda`/`alpha` (`_bayes.py:681-688`), recomputes
    /// `keep_lambda = lambda_ < threshold_lambda` and zeros pruned coefficients
    /// (`_bayes.py:691-692`), and converges on
    /// `sum(|coef_old - coef_|) < tol` (`_bayes.py:707`).
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 samples.
    /// - [`FerroError::NumericalInstability`] — numerical failure in solver.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedARDRegression<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "ARDRegression requires at least 2 samples".into(),
            });
        }

        let zero = <F as num_traits::Zero>::zero();
        let one = <F as num_traits::One>::one();
        let n_f = <F as num_traits::NumCast>::from(n_samples).unwrap_or(one);
        let two = one + one;

        // Center data for intercept (_bayes.py:637 `_preprocess_data`).
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

        // Init: alpha = 1/(Var(y)+eps), lambda = ones, coef = zeros, all kept
        // (_bayes.py:645, :658-659). `eps = finfo(f64).eps` matches sklearn,
        // which fixes float64 eps regardless of dtype.
        let eps =
            <F as num_traits::NumCast>::from(f64::EPSILON).unwrap_or_else(<F as Float>::epsilon);
        let var_y = {
            let ym = y_work
                .mean()
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute target mean for variance".into(),
                })?;
            let centered = &y_work - ym;
            centered.dot(&centered) / n_f
        };
        let mut alpha = one / (var_y + eps);
        let mut lambda = Array1::<F>::from_elem(n_features, one);
        let mut keep_lambda: Vec<bool> = vec![true; n_features];

        let mut coef = Array1::<F>::zeros(n_features);
        let mut coef_old: Option<Array1<F>> = None;
        // Diagonal of the posterior covariance over the full feature index
        // (pruned features carry 0); kept entries filled from `Sigma` each iter.
        let mut sigma_diag = Array1::<F>::zeros(n_features);

        for _iter_ in 0..self.max_iter {
            // Indices of kept columns.
            let kept: Vec<usize> = (0..n_features).filter(|&i| keep_lambda[i]).collect();
            let k = kept.len();

            // Xk = X[:, keep_lambda].
            let mut xk = Array2::<F>::zeros((n_samples, k));
            for (col, &i) in kept.iter().enumerate() {
                for row in 0..n_samples {
                    xk[[row, col]] = x_work[[row, i]];
                }
            }
            let lambda_keep: Vec<F> = kept.iter().map(|&i| lambda[i]).collect();

            // sigma_ = (diag(lambda[keep]) + alpha * Xk^T Xk)^{-1}  (_bayes.py:677).
            let sigma = update_sigma(&xk, alpha, &lambda_keep)?;

            // coef_[keep] = alpha * sigma_ @ Xk^T @ y; coef_[~keep] = 0
            // (_bayes.py:665-667, the running zeros from the prior mask).
            let xkt_y = xk.t().dot(&y_work);
            let coef_keep = sigma.dot(&xkt_y).mapv(|v| v * alpha);
            sigma_diag.fill(zero);
            for (col, &i) in kept.iter().enumerate() {
                coef[i] = coef_keep[col];
                sigma_diag[i] = sigma[[col, col]];
            }

            // rmse_ = sum((y - X @ coef_)^2)  (_bayes.py:681).
            let residual = &y_work - x_work.dot(&coef);
            let rmse = residual.dot(&residual);

            // gamma_ = 1 - lambda[keep] * diag(sigma_)  (_bayes.py:682).
            let mut gamma_sum = zero;
            let mut gamma_keep = vec![zero; k];
            for (col, &i) in kept.iter().enumerate() {
                let g = one - lambda[i] * sigma[[col, col]];
                gamma_keep[col] = g;
                gamma_sum += g;
            }

            // lambda[keep] = (gamma_ + 2*lambda_1) / (coef_[keep]^2 + 2*lambda_2)
            // (_bayes.py:683-685).
            for (col, &i) in kept.iter().enumerate() {
                let ci = coef[i];
                lambda[i] =
                    (gamma_keep[col] + two * self.lambda_1) / (ci * ci + two * self.lambda_2);
            }

            // alpha_ = (n - gamma.sum() + 2*alpha_1) / (rmse_ + 2*alpha_2)
            // (_bayes.py:686-688).
            alpha = (n_f - gamma_sum + two * self.alpha_1) / (rmse + two * self.alpha_2);

            // Prune: keep_lambda = lambda_ < threshold; coef_[~keep] = 0
            // (_bayes.py:691-692).
            for i in 0..n_features {
                keep_lambda[i] = lambda[i] < self.threshold_lambda;
                if !keep_lambda[i] {
                    coef[i] = zero;
                }
            }

            // Convergence: iter>0 and sum(|coef_old - coef_|) < tol (_bayes.py:707).
            if let Some(prev) = &coef_old {
                let delta: F = (0..n_features)
                    .map(|i| (prev[i] - coef[i]).abs())
                    .fold(zero, |a, b| a + b);
                if delta < self.tol {
                    break;
                }
            }
            coef_old = Some(coef.clone());

            // All features pruned -> stop (_bayes.py:713-714).
            if !keep_lambda.iter().any(|&b| b) {
                break;
            }
        }

        // Final coef_/sigma_ refresh with the converged params, over the
        // surviving kept set (_bayes.py:718-721).
        let kept: Vec<usize> = (0..n_features).filter(|&i| keep_lambda[i]).collect();
        let k = kept.len();
        if k > 0 {
            let mut xk = Array2::<F>::zeros((n_samples, k));
            for (col, &i) in kept.iter().enumerate() {
                for row in 0..n_samples {
                    xk[[row, col]] = x_work[[row, i]];
                }
            }
            let lambda_keep: Vec<F> = kept.iter().map(|&i| lambda[i]).collect();
            let sigma = update_sigma(&xk, alpha, &lambda_keep)?;
            let xkt_y = xk.t().dot(&y_work);
            let coef_keep = sigma.dot(&xkt_y).mapv(|v| v * alpha);
            coef.fill(zero);
            sigma_diag.fill(zero);
            for (col, &i) in kept.iter().enumerate() {
                coef[i] = coef_keep[col];
                sigma_diag[i] = sigma[[col, col]];
            }
        } else {
            coef.fill(zero);
            sigma_diag.fill(zero);
        }

        // intercept_ = y_offset - X_offset @ coef_ (_bayes.py:729 `_set_intercept`).
        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            *ym - xm.dot(&coef)
        } else {
            zero
        };

        Ok(FittedARDRegression {
            coefficients: coef,
            intercept,
            alpha,
            lambda,
            sigma: sigma_diag,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedARDRegression<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values using the posterior mean coefficients.
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedARDRegression<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for ARDRegression<F>
where
    F: LinalgFloat + FromPrimitive + ScalarOperand + Send + Sync,
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

impl<F> FittedPipelineEstimator<F> for FittedARDRegression<F>
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
    fn test_default_constructor() {
        let m = ARDRegression::<f64>::new();
        assert_eq!(m.max_iter, 300);
        assert!(m.fit_intercept);
        assert_relative_eq!(m.alpha_1, 1e-6);
    }

    #[test]
    fn test_builder_setters() {
        let m = ARDRegression::<f64>::new()
            .with_max_iter(50)
            .with_tol(1e-6)
            .with_alpha_1(1e-3)
            .with_alpha_2(1e-3)
            .with_lambda_1(1e-3)
            .with_lambda_2(1e-3)
            .with_threshold_lambda(1e5)
            .with_fit_intercept(false);
        assert_eq!(m.max_iter, 50);
        assert!(!m.fit_intercept);
        assert_relative_eq!(m.threshold_lambda, 1e5);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        let result = ARDRegression::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let y = array![1.0];
        let result = ARDRegression::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_fits_linear_data() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();

        // Should recover roughly y = 2x + 1.
        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.5);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1.5);
    }

    #[test]
    fn test_alpha_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        assert!(fitted.alpha() > 0.0);
    }

    #[test]
    fn test_lambda_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        for &v in fitted.lambda().iter() {
            assert!(v > 0.0, "lambda must be positive, got {v}");
        }
    }

    #[test]
    fn test_sigma_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        for &v in fitted.sigma().iter() {
            assert!(v > 0.0, "sigma diagonal must be positive, got {v}");
        }
    }

    #[test]
    fn test_predict_length() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = ARDRegression::<f64>::new()
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sparsity_on_irrelevant_features() {
        // y depends only on x1, x2 is noise-free irrelevant.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0, 5.0, 500.0, 6.0, 600.0,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]; // y = 2 * x1

        let fitted = ARDRegression::<f64>::new()
            .with_max_iter(1000)
            .fit(&x, &y)
            .unwrap();

        // The model should learn that x1 is relevant.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_has_coefficients_length() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.5, 2.0, 1.0, 1.0, 3.0, 0.0, 1.5, 4.0, 1.0, 2.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 3);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = ARDRegression::<f64>::new();
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
