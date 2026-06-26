//! Kernel density estimation.
//!
//! This module provides a dense, Euclidean Gaussian KDE estimator matching the
//! normalized log-likelihood convention of `sklearn.neighbors.KernelDensity`.
//! It intentionally exposes a narrower surface than sklearn: no sample weights,
//! string bandwidth rules, non-Gaussian kernels, tree tolerance knobs, sparse
//! inputs, or random sampling.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::knn::Algorithm;

/// Kernel family for [`KernelDensity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDensityKernel {
    /// Gaussian kernel.
    Gaussian,
}

/// Dense Euclidean kernel density estimator.
///
/// Defaults mirror the main sklearn defaults where this narrowed Rust surface
/// has equivalents: `bandwidth = 1.0`, `algorithm = Auto`, `kernel = Gaussian`,
/// and `leaf_size = 40`. The current scoring implementation is exact brute
/// force regardless of the stored algorithm.
#[derive(Debug, Clone)]
pub struct KernelDensity<F> {
    bandwidth: F,
    algorithm: Algorithm,
    kernel: KernelDensityKernel,
    leaf_size: usize,
}

/// Fitted dense Euclidean kernel density estimator.
#[derive(Debug, Clone)]
pub struct FittedKernelDensity<F> {
    x_train: Array2<F>,
    bandwidth: F,
    algorithm: Algorithm,
    kernel: KernelDensityKernel,
    leaf_size: usize,
}

impl<F: Float> KernelDensity<F> {
    /// Create a new estimator with sklearn-like defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bandwidth: F::one(),
            algorithm: Algorithm::Auto,
            kernel: KernelDensityKernel::Gaussian,
            leaf_size: 40,
        }
    }

    /// Set the positive kernel bandwidth.
    #[must_use]
    pub fn with_bandwidth(mut self, bandwidth: F) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Set the algorithm hint. Scoring currently remains exact brute force.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the kernel family. Currently only Gaussian is implemented.
    #[must_use]
    pub fn with_kernel(mut self, kernel: KernelDensityKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the tree leaf size parameter retained for sklearn surface symmetry.
    #[must_use]
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size;
        self
    }

    /// Return the configured bandwidth.
    #[must_use]
    pub fn bandwidth(&self) -> F {
        self.bandwidth
    }

    /// Return the configured algorithm hint.
    #[must_use]
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Return the configured kernel.
    #[must_use]
    pub fn kernel(&self) -> KernelDensityKernel {
        self.kernel
    }

    /// Return the configured leaf size.
    #[must_use]
    pub fn leaf_size(&self) -> usize {
        self.leaf_size
    }
}

impl<F: Float> Default for KernelDensity<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for KernelDensity<F> {
    type Fitted = FittedKernelDensity<F>;
    type Error = FerroError;

    /// Fit the KDE by storing the training samples.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] for non-positive/non-finite
    /// `bandwidth`, `leaf_size == 0`, or non-finite input values. Returns
    /// [`FerroError::InsufficientSamples`] for zero training rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedKernelDensity<F>, FerroError> {
        if !self.bandwidth.is_finite() || self.bandwidth <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "bandwidth".into(),
                reason: "must be finite and strictly positive".into(),
            });
        }
        if self.leaf_size == 0 {
            return Err(FerroError::InvalidParameter {
                name: "leaf_size".into(),
                reason: "must be at least 1".into(),
            });
        }
        validate_matrix("X", x, None)?;

        Ok(FittedKernelDensity {
            x_train: x.clone(),
            bandwidth: self.bandwidth,
            algorithm: self.algorithm,
            kernel: self.kernel,
            leaf_size: self.leaf_size,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedKernelDensity<F> {
    /// Return per-sample log-likelihoods under the fitted KDE.
    ///
    /// The Gaussian path matches sklearn's normalization:
    /// `log(sum_j K((x - x_j) / bandwidth)) - log(n_train)`, with the
    /// d-dimensional Gaussian and bandwidth normalization included.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] when query features differ from
    /// training features, and [`FerroError::InvalidParameter`] for non-finite
    /// query values.
    pub fn score_samples(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        validate_matrix("X", x, Some(self.n_features_in()))?;

        match self.kernel {
            KernelDensityKernel::Gaussian => self.score_samples_gaussian(x),
        }
    }

    /// Return the total log-likelihood of all query samples.
    pub fn score(&self, x: &Array2<F>) -> Result<F, FerroError> {
        Ok(self.score_samples(x)?.sum())
    }

    /// Number of features seen during fit.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.x_train.ncols()
    }

    /// Number of samples seen during fit.
    #[must_use]
    pub fn n_samples_fit(&self) -> usize {
        self.x_train.nrows()
    }

    /// Fitted numeric bandwidth.
    #[must_use]
    pub fn bandwidth(&self) -> F {
        self.bandwidth
    }

    /// Algorithm hint retained from the unfitted estimator.
    #[must_use]
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Kernel retained from the unfitted estimator.
    #[must_use]
    pub fn kernel(&self) -> KernelDensityKernel {
        self.kernel
    }

    /// Leaf size retained from the unfitted estimator.
    #[must_use]
    pub fn leaf_size(&self) -> usize {
        self.leaf_size
    }

    fn score_samples_gaussian(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_queries = x.nrows();
        let n_train = self.x_train.nrows();
        let n_features = self.x_train.ncols();
        let n_train_f = to_float::<F>(n_train as f64, "n_samples_fit")?;
        let n_features_f = to_float::<F>(n_features as f64, "n_features_in")?;
        let half = to_float::<F>(0.5, "0.5")?;
        let two_pi = to_float::<F>(2.0 * std::f64::consts::PI, "2*pi")?;
        let bandwidth_sq = self.bandwidth * self.bandwidth;
        let log_normalizer = n_features_f * self.bandwidth.ln() + half * n_features_f * two_pi.ln();
        let log_n_train = n_train_f.ln();
        let mut out = Array1::<F>::zeros(n_queries);

        for i in 0..n_queries {
            let mut log_terms = Vec::with_capacity(n_train);
            for j in 0..n_train {
                let mut dist_sq = F::zero();
                for k in 0..n_features {
                    let diff = x[[i, k]] - self.x_train[[j, k]];
                    dist_sq = dist_sq + diff * diff;
                }
                log_terms.push(-half * dist_sq / bandwidth_sq);
            }
            out[i] = logsumexp(&log_terms) - log_n_train - log_normalizer;
        }

        Ok(out)
    }
}

fn validate_matrix<F: Float>(
    name: &str,
    x: &Array2<F>,
    expected_features: Option<usize>,
) -> Result<(), FerroError> {
    if x.nrows() == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: name.into(),
        });
    }
    if x.ncols() == 0 {
        return Err(FerroError::InvalidParameter {
            name: name.into(),
            reason: "must contain at least one feature".into(),
        });
    }
    if let Some(expected) = expected_features
        && x.ncols() != expected
    {
        return Err(FerroError::ShapeMismatch {
            expected: vec![expected],
            actual: vec![x.ncols()],
            context: "number of features must match training data".into(),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: name.into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

fn logsumexp<F: Float>(values: &[F]) -> F {
    let max_value =
        values.iter().copied().fold(
            F::neg_infinity(),
            |acc, value| {
                if value > acc { value } else { acc }
            },
        );
    if max_value == F::neg_infinity() {
        return max_value;
    }

    let sum = values
        .iter()
        .copied()
        .map(|value| (value - max_value).exp())
        .fold(F::zero(), |acc, value| acc + value);
    max_value + sum.ln()
}

fn to_float<F: Float>(value: f64, context: &str) -> Result<F, FerroError> {
    F::from(value).ok_or_else(|| FerroError::NumericalInstability {
        message: format!("could not represent {context} as floating point"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn defaults_match_sklearn_surface_defaults() {
        let model = KernelDensity::<f64>::new();
        assert_relative_eq!(model.bandwidth(), 1.0, epsilon = 1e-12);
        assert_eq!(model.algorithm(), Algorithm::Auto);
        assert_eq!(model.kernel(), KernelDensityKernel::Gaussian);
        assert_eq!(model.leaf_size(), 40);
    }

    #[test]
    fn gaussian_score_samples_match_sklearn_oracle_1d() {
        let x = array![[0.0], [1.0], [2.0]];
        let q = array![[0.0], [1.5]];
        let fitted = KernelDensity::<f64>::new().fit(&x, &()).unwrap();
        let scores = fitted.score_samples(&q).unwrap();

        assert_relative_eq!(scores[0], -1.4625939022307919, epsilon = 1e-12);
        assert_relative_eq!(scores[1], -1.2805560178145314, epsilon = 1e-12);
        assert_relative_eq!(
            fitted.score(&q).unwrap(),
            -2.7431499200453233,
            epsilon = 1e-12
        );
    }

    #[test]
    fn gaussian_score_samples_match_sklearn_oracle_2d() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let q = array![[0.0, 0.0], [1.0, 1.0]];
        let fitted = KernelDensity::<f64>::new()
            .with_bandwidth(0.5)
            .with_algorithm(Algorithm::KdTree)
            .fit(&x, &())
            .unwrap();
        let scores = fitted.score_samples(&q).unwrap();

        assert_eq!(fitted.n_features_in(), 2);
        assert_eq!(fitted.n_samples_fit(), 3);
        assert_relative_eq!(fitted.bandwidth(), 0.5, epsilon = 1e-12);
        assert_eq!(fitted.algorithm(), Algorithm::KdTree);
        assert_relative_eq!(scores[0], -1.31065022773568, epsilon = 1e-12);
        assert_relative_eq!(scores[1], -2.7915713182780513, epsilon = 1e-12);
    }

    #[test]
    fn validation_errors_are_result_based() {
        let x = array![[0.0], [1.0]];
        assert!(
            KernelDensity::<f64>::new()
                .with_bandwidth(0.0)
                .fit(&x, &())
                .is_err()
        );
        assert!(
            KernelDensity::<f64>::new()
                .with_leaf_size(0)
                .fit(&x, &())
                .is_err()
        );
        assert!(
            KernelDensity::<f64>::new()
                .fit(&array![[f64::NAN]], &())
                .is_err()
        );
        let fitted = KernelDensity::<f64>::new().fit(&x, &()).unwrap();
        assert!(fitted.score_samples(&array![[0.0, 1.0]]).is_err());
        assert!(fitted.score_samples(&array![[f64::INFINITY]]).is_err());
    }
}
