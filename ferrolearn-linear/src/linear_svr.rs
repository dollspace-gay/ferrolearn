//! Linear Support Vector Regressor.
//!
//! This module provides [`LinearSVR`], a liblinear-faithful linear SVR that
//! operates directly in the primal space without kernel overhead. The fit
//! minimizes the L2-regularized epsilon-insensitive (or squared
//! epsilon-insensitive) objective
//!
//! ```text
//!   min_w  0.5 * ||w||^2  +  C * sum_i  L_eps(y_i - w . x_i)
//! ```
//!
//! (NO `1/n` averaging — the summed loss is scaled by `C`, matching
//! `sklearn/svm/_base.py` `_fit_liblinear`). The solver is liblinear's dual
//! coordinate descent for SVR (`solve_l2r_l1l2_svr` in
//! `sklearn/svm/src/liblinear/linear.cpp:1051`), which converges to the unique
//! minimizer of the strongly convex objective.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::linear_svr::LinearSVR;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let model = LinearSVR::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 5);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

/// Loss function for [`LinearSVR`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSVRLoss {
    /// Epsilon-insensitive loss: `max(0, |y - f(x)| - epsilon)` (L1, the
    /// standard SVR loss). liblinear solver `L2R_L1LOSS_SVR_DUAL` (type 13).
    EpsilonInsensitive,
    /// Squared epsilon-insensitive loss: `max(0, |y - f(x)| - epsilon)^2` (L2).
    /// liblinear solver `L2R_L2LOSS_SVR_DUAL` (type 12).
    SquaredEpsilonInsensitive,
}

/// Linear Support Vector Regressor (primal objective, liblinear dual CD).
///
/// Solves the L2-regularized epsilon-insensitive or squared
/// epsilon-insensitive objective `0.5*||w||^2 + C * sum_i L_eps(y_i - w.x_i)`
/// via liblinear's dual coordinate descent. Mirrors `sklearn.svm.LinearSVR`.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LinearSVR<F> {
    /// Regularization parameter. The strength of the regularization is
    /// inversely proportional to `C`. Must be strictly positive.
    pub c: F,
    /// Width of the epsilon-insensitive tube (sklearn default `0.0`).
    pub epsilon: F,
    /// Maximum number of dual coordinate descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the projected-gradient L1 norm.
    pub tol: F,
    /// Loss function to use.
    pub loss: LinearSVRLoss,
    /// Whether to fit an intercept. When `true`, the design matrix is augmented
    /// with a synthetic constant column equal to [`intercept_scaling`]; the
    /// augmented weight is penalized like any feature (liblinear convention).
    ///
    /// [`intercept_scaling`]: Self::intercept_scaling
    pub fit_intercept: bool,
    /// Value of the synthetic intercept feature when [`fit_intercept`] is
    /// `true`. Must be strictly positive. `intercept_ = intercept_scaling *
    /// w_last`.
    ///
    /// [`fit_intercept`]: Self::fit_intercept
    pub intercept_scaling: F,
}

impl<F: Float> LinearSVR<F> {
    /// Create a new `LinearSVR` with scikit-learn's default settings.
    ///
    /// Defaults (matching `sklearn.svm.LinearSVR`, `_classes.py:519-532`):
    /// `C = 1.0`, `epsilon = 0.0`, `max_iter = 1000`, `tol = 1e-4`,
    /// `loss = EpsilonInsensitive`, `fit_intercept = true`,
    /// `intercept_scaling = 1.0`.
    ///
    /// # Panics
    ///
    /// Never panics: the literal constants `0.0`, `1e-4`, `1.0` are exactly
    /// representable in every supported float type, so the `F::from` conversions
    /// cannot fail.
    #[must_use]
    pub fn new() -> Self {
        // These literals are exactly representable, so `or(F::zero())` /
        // `or(F::one())` fallbacks are never taken (no `.unwrap()` in lib code).
        let zero = F::zero();
        let one = F::one();
        Self {
            c: one,
            epsilon: zero,
            max_iter: 1000,
            tol: F::from(1e-4).unwrap_or_else(|| {
                // 1e-4 is representable in f32/f64; fall back defensively.
                let ten = F::from(10).unwrap_or(one);
                one / (ten * ten * ten * ten)
            }),
            loss: LinearSVRLoss::EpsilonInsensitive,
            fit_intercept: true,
            intercept_scaling: one,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: F) -> Self {
        self.c = c;
        self
    }

    /// Set the epsilon tube width.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: F) -> Self {
        self.epsilon = epsilon;
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

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: LinearSVRLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set whether to fit an intercept (sklearn `fit_intercept`).
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the intercept scaling (sklearn `intercept_scaling`). Must be
    /// strictly positive when `fit_intercept` is `true`.
    #[must_use]
    pub fn with_intercept_scaling(mut self, intercept_scaling: F) -> Self {
        self.intercept_scaling = intercept_scaling;
        self
    }
}

impl<F: Float> Default for LinearSVR<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Linear Support Vector Regressor.
///
/// Stores the learned coefficients and intercept. `coef_` are the weights on
/// the original features; `intercept_ = intercept_scaling * w_last` when an
/// intercept was fit, otherwise `0`.
#[derive(Debug, Clone)]
pub struct FittedLinearSVR<F> {
    /// Learned coefficient vector (one per original feature).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<F>> for LinearSVR<F> {
    type Fitted = FittedLinearSVR<F>;
    type Error = FerroError;

    /// Fit the linear SVR model using liblinear's dual coordinate descent.
    ///
    /// Minimizes `0.5*||w||^2 + C * sum_i L_eps(y_i - w.x_i)` (no `1/n`). When
    /// `fit_intercept` is set, the design matrix is augmented with a constant
    /// column equal to `intercept_scaling`, the augmented weight is penalized
    /// like any feature, and `intercept_ = intercept_scaling * w_last`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InvalidParameter`] — `C` not positive, `epsilon`
    ///   negative, or `intercept_scaling` not positive (when fitting an
    ///   intercept).
    /// - [`FerroError::InsufficientSamples`] — no samples provided.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedLinearSVR<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "LinearSVR requires at least one sample".into(),
            });
        }

        if self.c <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "C".into(),
                reason: "must be positive".into(),
            });
        }

        if self.epsilon < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "epsilon".into(),
                reason: "must be non-negative".into(),
            });
        }

        // liblinear raises when intercept_scaling <= 0 with fit_intercept
        // (`_base.py:1190-1196`).
        if self.fit_intercept && self.intercept_scaling <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "intercept_scaling".into(),
                reason: "must be greater than 0 when fit_intercept is true".into(),
            });
        }

        // Build the (possibly augmented) design matrix. The synthetic bias
        // column holds `intercept_scaling`; its weight is penalized in ||w||^2,
        // exactly as liblinear treats it (`_base.py:1189-1198`).
        let w_size = if self.fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Per-sample squared norm QD[i] = ||x_i||^2 (including the bias column).
        let mut qd = vec![F::zero(); n_samples];
        for (i, qd_i) in qd.iter_mut().enumerate() {
            let mut acc = F::zero();
            let row = x.row(i);
            for &v in row.iter() {
                acc = acc + v * v;
            }
            if self.fit_intercept {
                acc = acc + self.intercept_scaling * self.intercept_scaling;
            }
            *qd_i = acc;
        }

        // liblinear `solve_l2r_l1l2_svr` (`linear.cpp:1051`).
        //   p   = epsilon, C = C, eps = tol
        //   L1 loss: lambda[i] = 0,        upper_bound[i] = C
        //   L2 loss: lambda[i] = 0.5 / C,  upper_bound[i] = +inf
        let p = self.epsilon;
        let inf = F::infinity();
        let (lambda, upper_bound) = match self.loss {
            LinearSVRLoss::EpsilonInsensitive => (F::zero(), self.c),
            LinearSVRLoss::SquaredEpsilonInsensitive => {
                let half = F::from(0.5).unwrap_or_else(|| F::one() / (F::one() + F::one()));
                (half / self.c, inf)
            }
        };

        let mut beta = vec![F::zero(); n_samples];
        let mut w = vec![F::zero(); w_size];
        // beta starts at 0 so w starts at 0; no contribution to accumulate.

        let mut index: Vec<usize> = (0..n_samples).collect();
        let mut active_size = n_samples;
        let mut gmax_old = inf;
        let mut gnorm1_init = -F::one();

        let tiny = F::from(1.0e-12).unwrap_or_else(|| F::epsilon());

        // Closure-free helper to compute w . x_i (augmented).
        let dot_w_xi = |w: &[F], i: usize| -> F {
            let mut acc = F::zero();
            let row = x.row(i);
            for (j, &v) in row.iter().enumerate() {
                acc = acc + v * w[j];
            }
            if self.fit_intercept {
                acc = acc + self.intercept_scaling * w[n_features];
            }
            acc
        };

        for iter in 0..self.max_iter {
            let mut gmax_new = F::zero();
            let mut gnorm1_new = F::zero();

            // liblinear shuffles `index` each sweep; the minimizer is unique so
            // order only affects the path, not the limit. We sweep in natural
            // order for determinism (no RNG), matching the converged optimum.

            let mut s = 0;
            while s < active_size {
                let i = index[s];

                // G = -y_i + lambda*beta_i + w . x_i
                let g = -y[i] + lambda * beta[i] + dot_w_xi(&w, i);
                let gp = g + p;
                let gn = g - p;

                let mut violation = F::zero();
                let mut shrink = false;

                if beta[i] == F::zero() {
                    if gp < F::zero() {
                        violation = -gp;
                    } else if gn > F::zero() {
                        violation = gn;
                    } else if gp > gmax_old && gn < -gmax_old {
                        shrink = true;
                    }
                } else if beta[i] >= upper_bound {
                    if gp > F::zero() {
                        violation = gp;
                    } else if gp < -gmax_old {
                        shrink = true;
                    }
                } else if beta[i] <= -upper_bound {
                    if gn < F::zero() {
                        violation = -gn;
                    } else if gn > gmax_old {
                        shrink = true;
                    }
                } else if beta[i] > F::zero() {
                    violation = gp.abs();
                } else {
                    violation = gn.abs();
                }

                if shrink {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue; // re-process the swapped-in element at `s`
                }

                gmax_new = if violation > gmax_new {
                    violation
                } else {
                    gmax_new
                };
                gnorm1_new = gnorm1_new + violation;

                // Newton direction d (1-D minimization of the dual along beta_i).
                let h = qd[i] + lambda;
                let d = if gp < h * beta[i] {
                    -gp / h
                } else if gn > h * beta[i] {
                    -gn / h
                } else {
                    -beta[i]
                };

                if d.abs() < tiny {
                    s += 1;
                    continue;
                }

                let beta_old = beta[i];
                let mut new_beta = beta[i] + d;
                // Clamp to [-upper_bound, upper_bound].
                if new_beta > upper_bound {
                    new_beta = upper_bound;
                } else if new_beta < -upper_bound {
                    new_beta = -upper_bound;
                }
                beta[i] = new_beta;
                let d_eff = beta[i] - beta_old;

                if d_eff != F::zero() {
                    let row = x.row(i);
                    for (j, &v) in row.iter().enumerate() {
                        w[j] = w[j] + d_eff * v;
                    }
                    if self.fit_intercept {
                        w[n_features] = w[n_features] + d_eff * self.intercept_scaling;
                    }
                }

                s += 1;
            }

            if iter == 0 {
                gnorm1_init = gnorm1_new;
            }

            // Convergence: Gnorm1_new <= tol * Gnorm1_init. If converged on the
            // active set, reactivate all and re-check once (liblinear pattern).
            if gnorm1_new <= self.tol * gnorm1_init {
                if active_size == n_samples {
                    break;
                }
                active_size = n_samples;
                gmax_old = inf;
                continue;
            }

            gmax_old = gmax_new;
        }

        // Extract coef_ / intercept_ (`_base.py:598`, `_classes.py:581-598`).
        let coefficients = Array1::from_iter(w.iter().take(n_features).copied());
        let intercept = if self.fit_intercept {
            self.intercept_scaling * w[n_features]
        } else {
            F::zero()
        };

        Ok(FittedLinearSVR {
            coefficients,
            intercept,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedLinearSVR<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedLinearSVR<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for LinearSVR<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
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

impl<F> FittedPipelineEstimator<F> for FittedLinearSVR<F>
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
        let m = LinearSVR::<f64>::new();
        assert_eq!(m.max_iter, 1000);
        assert!(m.c == 1.0);
        // sklearn default epsilon=0.0 (`_classes.py:522`).
        assert_relative_eq!(m.epsilon, 0.0);
        assert_eq!(m.loss, LinearSVRLoss::EpsilonInsensitive);
        assert!(m.fit_intercept);
        assert_relative_eq!(m.intercept_scaling, 1.0);
    }

    #[test]
    fn test_builder_setters() {
        let m = LinearSVR::<f64>::new()
            .with_c(10.0)
            .with_epsilon(0.5)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_loss(LinearSVRLoss::SquaredEpsilonInsensitive)
            .with_fit_intercept(false)
            .with_intercept_scaling(2.0);
        assert!(m.c == 10.0);
        assert_relative_eq!(m.epsilon, 0.5);
        assert_eq!(m.max_iter, 500);
        assert_eq!(m.loss, LinearSVRLoss::SquaredEpsilonInsensitive);
        assert!(!m.fit_intercept);
        assert_relative_eq!(m.intercept_scaling, 2.0);
    }

    #[test]
    fn test_fits_linear_data() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let model = LinearSVR::<f64>::new()
            .with_c(10.0)
            .with_epsilon(0.0)
            .with_max_iter(10000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Should roughly recover y = 2x.
        for (p, &t) in preds.iter().zip(y.iter()) {
            assert!(
                (p - t).abs() < 3.0,
                "prediction {p} too far from target {t}"
            );
        }
    }

    #[test]
    fn test_squared_epsilon_insensitive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let model = LinearSVR::<f64>::new()
            .with_c(10.0)
            .with_epsilon(0.0)
            .with_loss(LinearSVRLoss::SquaredEpsilonInsensitive)
            .with_max_iter(10000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = LinearSVR::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_c() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = LinearSVR::<f64>::new().with_c(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_negative_epsilon() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = LinearSVR::<f64>::new().with_epsilon(-0.1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_intercept_scaling() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = LinearSVR::<f64>::new().with_intercept_scaling(0.0);
        assert!(model.fit(&x, &y).is_err());

        // But with fit_intercept=false, intercept_scaling is ignored.
        let model = LinearSVR::<f64>::new()
            .with_fit_intercept(false)
            .with_intercept_scaling(0.0);
        assert!(model.fit(&x, &y).is_ok());
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let fitted = LinearSVR::<f64>::new()
            .with_max_iter(5000)
            .fit(&x, &y)
            .unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let fitted = LinearSVR::<f64>::new()
            .with_max_iter(5000)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_fit_intercept_false_zero_intercept() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = LinearSVR::<f64>::new()
            .with_fit_intercept(false)
            .with_max_iter(5000)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.intercept(), 0.0);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = LinearSVR::<f64>::new().with_max_iter(5000);
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
