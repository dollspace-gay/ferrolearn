//! Additive chi-squared kernel feature map approximation.
//!
//! [`AdditiveChi2Sampler`] mirrors sklearn's deterministic
//! `sklearn.kernel_approximation.AdditiveChi2Sampler`. Each nonnegative input
//! feature is expanded into `2 * sample_steps - 1` explicit features using the
//! Fourier sampling formula from Vedaldi and Zisserman (2011). Unlike
//! [`crate::RBFSampler`], this transformer has no RNG-coupled state.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.kernel_approximation.AdditiveChi2Sampler`
//! (`kernel_approximation.py:585`, v1.5.2 commit 156ef14).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (public surface) | SHIPPED | `AdditiveChi2Sampler` and `FittedAdditiveChi2Sampler` are re-exported from `lib.rs` and covered by API/conformance tests. |
//! | REQ-2 (default intervals) | SHIPPED | `sample_steps` 1/2/3 use sklearn's `0.8`/`0.5`/`0.4` defaults (`:726-736`). |
//! | REQ-3 (dense transform formula) | SHIPPED | `transform` matches sklearn `_transform_dense` (`:783-806`) against live-oracle constants. |
//! | REQ-4 (parameter validation) | SHIPPED | rejects `sample_steps == 0`, nonpositive custom intervals, and missing custom interval for steps outside 1..=3. |
//! | REQ-5 (nonnegative input validation) | SHIPPED | `fit` and `transform` reject negative entries like sklearn `ensure_non_negative=True`. |
//! | REQ-6 (sparse/dataframe/feature-names surface) | NOT-STARTED | dense `ndarray` only; no sparse return type or `get_feature_names_out`. |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;
use num_traits::Float;

/// Approximate feature map for the additive chi-squared kernel.
#[derive(Debug, Clone)]
pub struct AdditiveChi2Sampler<F> {
    /// Number of complex sampling points. Default: 2.
    sample_steps: usize,
    /// Optional sampling interval. Required when `sample_steps` is not 1, 2, or 3.
    sample_interval: Option<F>,
}

impl<F: Float + Send + Sync + 'static> AdditiveChi2Sampler<F> {
    /// Create a new additive chi-squared sampler with sklearn defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sample_steps: 2,
            sample_interval: None,
        }
    }

    /// Set the number of complex sampling points.
    #[must_use]
    pub fn with_sample_steps(mut self, sample_steps: usize) -> Self {
        self.sample_steps = sample_steps;
        self
    }

    /// Set a custom sampling interval.
    #[must_use]
    pub fn with_sample_interval(mut self, sample_interval: F) -> Self {
        self.sample_interval = Some(sample_interval);
        self
    }

    /// Return the configured sample step count.
    #[must_use]
    pub fn sample_steps(&self) -> usize {
        self.sample_steps
    }

    /// Return the configured custom interval, if any.
    #[must_use]
    pub fn sample_interval(&self) -> Option<F> {
        self.sample_interval
    }

    fn effective_sample_interval(&self) -> Result<F, FerroError> {
        if self.sample_steps == 0 {
            return Err(FerroError::InvalidParameter {
                name: "sample_steps".into(),
                reason: "must be at least 1".into(),
            });
        }

        if let Some(sample_interval) = self.sample_interval {
            if sample_interval <= F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "sample_interval".into(),
                    reason: "must be positive".into(),
                });
            }
            return Ok(sample_interval);
        }

        match self.sample_steps {
            1 => Ok(F::from(0.8).unwrap()),
            2 => Ok(F::from(0.5).unwrap()),
            3 => Ok(F::from(0.4).unwrap()),
            _ => Err(FerroError::InvalidParameter {
                name: "sample_interval".into(),
                reason: "must be provided when sample_steps is not 1, 2, or 3".into(),
            }),
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for AdditiveChi2Sampler<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted additive chi-squared feature map.
#[derive(Debug, Clone)]
pub struct FittedAdditiveChi2Sampler<F> {
    sample_steps: usize,
    sample_interval: F,
    n_features_in: usize,
}

impl<F: Float + Send + Sync + 'static> FittedAdditiveChi2Sampler<F> {
    /// Number of features observed during fitting.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Number of complex sampling points.
    #[must_use]
    pub fn sample_steps(&self) -> usize {
        self.sample_steps
    }

    /// Effective sampling interval used by the transform.
    #[must_use]
    pub fn sample_interval(&self) -> F {
        self.sample_interval
    }

    /// Number of output features produced by [`Transform::transform`].
    #[must_use]
    pub fn n_features_out(&self) -> usize {
        self.n_features_in * (2 * self.sample_steps - 1)
    }
}

fn validate_nonnegative<F: Float>(x: &Array2<F>, context: &str) -> Result<(), FerroError> {
    for &value in x {
        if value < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: format!("{context} requires nonnegative input"),
            });
        }
    }
    Ok(())
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for AdditiveChi2Sampler<F> {
    type Fitted = FittedAdditiveChi2Sampler<F>;
    type Error = FerroError;

    /// Validate parameters and input, producing the fitted stateless map.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] for invalid sampling parameters
    /// or negative input entries. Returns [`FerroError::InsufficientSamples`]
    /// when no rows are provided.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedAdditiveChi2Sampler<F>, FerroError> {
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "AdditiveChi2Sampler::fit".into(),
            });
        }
        let sample_interval = self.effective_sample_interval()?;
        validate_nonnegative(x, "AdditiveChi2Sampler::fit")?;

        Ok(FittedAdditiveChi2Sampler {
            sample_steps: self.sample_steps,
            sample_interval,
            n_features_in: x.ncols(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedAdditiveChi2Sampler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Apply sklearn's dense additive chi-squared feature map.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature count differs from
    /// fit time, or [`FerroError::InvalidParameter`] if any input is negative.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedAdditiveChi2Sampler::transform feature count must match fit data"
                    .into(),
            });
        }
        validate_nonnegative(x, "FittedAdditiveChi2Sampler::transform")?;

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut out = Array2::<F>::zeros((n_samples, self.n_features_out()));
        let pi = F::from(std::f64::consts::PI).unwrap();

        for i in 0..n_samples {
            for feature_idx in 0..n_features {
                let value = x[[i, feature_idx]];
                if value == F::zero() {
                    continue;
                }

                out[[i, feature_idx]] = (value * self.sample_interval).sqrt();

                let log_step = self.sample_interval * value.ln();
                let step = F::from(2.0).unwrap() * value * self.sample_interval;

                for j in 1..self.sample_steps {
                    let j_f = F::from(j).unwrap();
                    let factor = (step / (pi * j_f * self.sample_interval).cosh()).sqrt();
                    let angle = j_f * log_step;
                    let base = n_features * (1 + 2 * (j - 1));
                    out[[i, base + feature_idx]] = factor * angle.cos();
                    out[[i, base + n_features + feature_idx]] = factor * angle.sin();
                }
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn defaults_match_sklearn() {
        let sampler = AdditiveChi2Sampler::<f64>::new();
        assert_eq!(sampler.sample_steps(), 2);
        assert_eq!(sampler.sample_interval(), None);
        let x = array![[0.0, 1.0], [2.0, 3.0]];
        let fitted = sampler.fit(&x, &()).unwrap();
        assert_eq!(fitted.sample_steps(), 2);
        assert_abs_diff_eq!(fitted.sample_interval(), 0.5, epsilon = 1e-15);
        assert_eq!(fitted.n_features_in(), 2);
        assert_eq!(fitted.n_features_out(), 6);
    }

    #[test]
    fn transform_shape_and_zero_handling() {
        let x = array![[0.0, 1.0], [2.0, 3.0]];
        let fitted = AdditiveChi2Sampler::<f64>::new()
            .with_sample_steps(3)
            .fit(&x, &())
            .unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.dim(), (2, 10));
        for col in [0, 2, 4, 6, 8] {
            assert_abs_diff_eq!(z[[0, col]], 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn sample_steps_outside_defaults_requires_interval() {
        let x = array![[1.0, 2.0]];
        let err = AdditiveChi2Sampler::<f64>::new()
            .with_sample_steps(4)
            .fit(&x, &())
            .unwrap_err();
        assert!(matches!(err, FerroError::InvalidParameter { .. }));

        let fitted = AdditiveChi2Sampler::<f64>::new()
            .with_sample_steps(4)
            .with_sample_interval(0.7)
            .fit(&x, &())
            .unwrap();
        assert_abs_diff_eq!(fitted.sample_interval(), 0.7, epsilon = 1e-15);
    }

    #[test]
    fn rejects_negative_fit_and_transform_input() {
        let x = array![[1.0, 2.0]];
        let x_negative = array![[1.0, -2.0]];
        assert!(
            AdditiveChi2Sampler::<f64>::new()
                .fit(&x_negative, &())
                .is_err()
        );
        let fitted = AdditiveChi2Sampler::<f64>::new().fit(&x, &()).unwrap();
        assert!(fitted.transform(&x_negative).is_err());
    }

    #[test]
    fn rejects_wrong_feature_count() {
        let x = array![[1.0, 2.0]];
        let fitted = AdditiveChi2Sampler::<f64>::new().fit(&x, &()).unwrap();
        let err = fitted.transform(&array![[1.0, 2.0, 3.0]]).unwrap_err();
        assert!(matches!(err, FerroError::ShapeMismatch { .. }));
    }
}
