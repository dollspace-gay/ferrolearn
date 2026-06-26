//! Kernel centering transformer.
//!
//! Dense `Array2` translation of scikit-learn's
//! `sklearn.preprocessing.KernelCenterer` (`_data.py:2421`). Fitting stores the
//! training kernel column means (`K_fit_rows_`) and grand mean (`K_fit_all_`);
//! transforming subtracts those training means plus each prediction-row mean.
//!
//! ## REQ status
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (fit attrs: `K_fit_rows_`, `K_fit_all_`) | SHIPPED | `Fit::fit` requires a finite square matrix and stores per-column means plus the grand mean, mirroring sklearn `_data.py:2528-2531`. Live-oracle tests in `tests/divergence_kernel_centerer.rs`. |
//! | REQ-2 (square fit transform) | SHIPPED | `FittedKernelCenterer::transform` computes `K -= K_fit_rows_; K -= row_mean(K); K += K_fit_all_`, matching sklearn `_data.py:2563-2567`; doc-example oracle test passes. |
//! | REQ-3 (rectangular prediction kernels) | SHIPPED | transform accepts `(n_pred, n_train)` kernels and centers with stored training means, matching sklearn's out-of-sample path (`_data.py:2555-2567`). |
//! | REQ-4 (copy/no in-place parameter) | SHIPPED | ferrolearn returns an owned matrix; sklearn's `copy` is an accept-and-document no-op for this Rust API. |
//! | REQ-5 (feature names / `get_feature_names_out`) | NOT-STARTED | sklearn `ClassNamePrefixFeaturesOutMixin`; ferrolearn has no string feature-name substrate here. |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Unfitted kernel centering transformer.
#[derive(Debug, Clone)]
pub struct KernelCenterer<F> {
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> KernelCenterer<F> {
    /// Create a new `KernelCenterer`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for KernelCenterer<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted kernel centering transformer.
#[derive(Debug, Clone)]
pub struct FittedKernelCenterer<F> {
    /// Average of each training-kernel column (`K_fit_rows_` in sklearn).
    pub(crate) k_fit_rows_: Array1<F>,
    /// Grand mean of the training kernel (`K_fit_all_` in sklearn).
    pub(crate) k_fit_all_: F,
    /// Number of training kernel columns seen during fit.
    pub(crate) n_features_in_: usize,
}

impl<F: Float + Send + Sync + 'static> FittedKernelCenterer<F> {
    /// Return the average of each training-kernel column.
    #[must_use]
    pub fn k_fit_rows(&self) -> &Array1<F> {
        &self.k_fit_rows_
    }

    /// Return the grand mean of the training kernel.
    #[must_use]
    pub fn k_fit_all(&self) -> F {
        self.k_fit_all_
    }

    /// Return the number of training kernel columns seen during fit.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }
}

fn validate_kernel_matrix<F: Float>(k: &Array2<F>, context: &str) -> Result<(), FerroError> {
    if k.nrows() == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: context.into(),
        });
    }
    if k.ncols() == 0 {
        return Err(FerroError::InvalidParameter {
            name: "K".into(),
            reason: "Found array with 0 feature(s); a minimum of 1 is required.".into(),
        });
    }
    if k.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "K".into(),
            reason: "Input K contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for KernelCenterer<F> {
    type Fitted = FittedKernelCenterer<F>;
    type Error = FerroError;

    /// Fit by storing training-kernel column means and the grand mean.
    fn fit(&self, k: &Array2<F>, _y: &()) -> Result<FittedKernelCenterer<F>, FerroError> {
        validate_kernel_matrix(k, "KernelCenterer::fit")?;

        if k.nrows() != k.ncols() {
            return Err(FerroError::InvalidParameter {
                name: "K".into(),
                reason: format!(
                    "Kernel matrix must be a square matrix. Input is a {}x{} matrix.",
                    k.nrows(),
                    k.ncols()
                ),
            });
        }

        let n_samples = k.nrows();
        let n_f = F::from(n_samples).unwrap_or_else(F::one);
        let mut k_fit_rows_ = Array1::zeros(n_samples);
        for j in 0..n_samples {
            let mut sum = F::zero();
            for i in 0..n_samples {
                sum = sum + k[[i, j]];
            }
            k_fit_rows_[j] = sum / n_f;
        }
        let k_fit_all_ = k_fit_rows_
            .iter()
            .copied()
            .fold(F::zero(), |acc, v| acc + v)
            / n_f;

        Ok(FittedKernelCenterer {
            k_fit_rows_,
            k_fit_all_,
            n_features_in_: n_samples,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedKernelCenterer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Center a kernel matrix with the training means learned in [`Fit::fit`].
    fn transform(&self, k: &Array2<F>) -> Result<Array2<F>, FerroError> {
        validate_kernel_matrix(k, "FittedKernelCenterer::transform")?;

        let n_train = self.k_fit_rows_.len();
        if k.ncols() != n_train {
            return Err(FerroError::ShapeMismatch {
                expected: vec![k.nrows(), n_train],
                actual: vec![k.nrows(), k.ncols()],
                context: "FittedKernelCenterer::transform".into(),
            });
        }

        let n_f = F::from(n_train).unwrap_or_else(F::one);
        let mut out = k.to_owned();
        for i in 0..out.nrows() {
            let mut row_sum = F::zero();
            for j in 0..n_train {
                row_sum = row_sum + out[[i, j]];
            }
            let row_mean = row_sum / n_f;
            for j in 0..n_train {
                out[[i, j]] = out[[i, j]] - self.k_fit_rows_[j] - row_mean + self.k_fit_all_;
            }
        }
        Ok(out)
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for KernelCenterer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    fn transform(&self, _k: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "KernelCenterer".into(),
            reason: "centerer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for KernelCenterer<F> {
    type FitError = FerroError;

    fn fit_transform(&self, k: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(k, &())?;
        fitted.transform(k)
    }
}

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for KernelCenterer<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedKernelCenterer<F> {
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}
