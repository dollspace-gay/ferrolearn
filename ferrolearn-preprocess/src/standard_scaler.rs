//! Standard scaler: zero-mean, unit-variance scaling.
//!
//! Each feature is transformed as `(x - mean) / std`. Constant
//! (zero-variance) columns use an effective scale of 1, so each entry maps to
//! `(x - mean) / 1 = 0` (matching scikit-learn `_handle_zeros_in_scale`).
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_data.py` (`class StandardScaler`
//! `:696`, `scale` `:133`). Design doc: `.design/preprocess/standard_scaler.md`. Expected values
//! from the live sklearn 1.5.2 oracle (R-CHAR-3). Consumers: PyO3 `_RsStandardScaler`
//! (`ferrolearn-python/src/transformers.rs:12`) + `PipelineTransformer` impl + crate re-export
//! (`lib.rs`, grandfathered S5). HONEST (R-HONEST-3): always centers+scales (no with_mean/with_std
//! params); the per-column standardize VALUES match sklearn on the default path.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (per-column standardize value match, non-constant) | SHIPPED | `Fit::fit` learns per-column `mean`/`std=sqrt(population var ddof=0)`; `Transform::transform` = `(x-mean)/std`, mirroring sklearn `X -= mean_; X /= scale_` (`_data.py:1064-1067`). Critic-verified bit-identical to live oracle: `green_req1_value_match_non_constant` (`[[-1.2247..],[0],[1.2247..]]`), `green_req1_mean_and_scale_attributes` (mean()/std() == sklearn mean_/scale_ on non-constant), `green_req1_negative_decimal_fixture`. Consumers: PyO3 `_RsStandardScaler` + `FittedPipelineTransformer` + re-export. |
//! | REQ-2 (constant / zero-variance column → 0) | SHIPPED | FIXED #1191. `transform` uses `s_eff = if s==0 {1} else {s}` → constant col `(x-mean)/1 = 0`, matching sklearn `_handle_zeros_in_scale` (`_data.py:88`,`:1019-1021`); `inverse_transform` aligned to `x*s_eff+m`. Critic two-round CLEAN: 9 tests incl. constant→0 (default/single-row/mixed) + inverse non-round-trip on constant col matches `x*1+mean`. In-module test corrected (R-HONEST-4). |
//! | REQ-3 (inverse_transform round-trip) | SHIPPED | `inverse_transform` = `x*s_eff + mean`, mirroring sklearn `X *= scale_; X += mean_` (`_data.py:1106-1109`); `inverse_transform(transform(X))==X` (`green_req3_inverse_roundtrip`). Consumer: PyO3 `_RsStandardScaler::inverse_transform`. |
//! | REQ-4 (PyO3 binding) | SHIPPED | `_RsStandardScaler` (`transformers.rs:12`, registered `lib.rs:22`) marshals `fit`/`transform`/`inverse_transform`/`mean_` over `FittedStandardScaler<f64>` — a real CPython consumer. |
//! | REQ-5 (var_/scale_/n_samples_seen_ attrs) | NOT-STARTED | open prereq blocker #1192. Only `mean`/`std`; no `var_`/`scale_`/`n_samples_seen_` (`_data.py:1013-1023`). |
//! | REQ-6 (with_mean/with_std/copy params) | NOT-STARTED | open prereq blocker #1193. Always centers+scales (`_data.py:829-838`,`:1064-1067`). |
//! | REQ-7 (NaN tolerance: allow-nan) | NOT-STARTED | open prereq blocker #1194. Fold propagates NaN; sklearn ignores NaN (`_data.py:918`,`:1112-1113`). Not a rejection. |
//! | REQ-8 (scale free fn + axis) | NOT-STARTED | open prereq blocker #1195. No `scale` fn / axis=1 (`_data.py:133`). |
//! | REQ-9 (partial_fit / streaming) | NOT-STARTED | open prereq blocker #1196. Single-shot (`_data.py:880-1025`). |
//! | REQ-10 (sample_weight) | NOT-STARTED | open prereq blocker #1197. Unweighted (`_data.py:923-924`). |
//! | REQ-11 (sparse CSR/CSC) | NOT-STARTED | open prereq blocker #1198. Dense-only (`_data.py:940-983`). |
//! | REQ-12 (get_feature_names_out / n_features_in_) | NOT-STARTED | open prereq blocker #1199. None (OneToOneFeatureMixin). |
//! | REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1200. `ndarray`+`num_traits`, not `ferray-core` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// StandardScaler (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted standard scaler.
///
/// Calling [`Fit::fit`] learns the per-column means and standard deviations
/// and returns a [`FittedStandardScaler`] that can transform new data.
///
/// Constant (zero-variance) columns use an effective scale of 1, so each entry
/// maps to `(x - mean) / 1 = 0` after transformation.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::StandardScaler;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let scaler = StandardScaler::<f64>::new();
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let fitted = scaler.fit(&x, &()).unwrap();
/// let scaled = fitted.transform(&x).unwrap();
/// // Mean of each column is now ~0, std ~1
/// ```
#[derive(Debug, Clone)]
pub struct StandardScaler<F> {
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> StandardScaler<F> {
    /// Create a new `StandardScaler` with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for StandardScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedStandardScaler
// ---------------------------------------------------------------------------

/// A fitted standard scaler holding per-column means and standard deviations.
///
/// Created by calling [`Fit::fit`] on a [`StandardScaler`].
#[derive(Debug, Clone)]
pub struct FittedStandardScaler<F> {
    /// Per-column means learned during fitting.
    pub(crate) mean: Array1<F>,
    /// Per-column standard deviations learned during fitting.
    pub(crate) std: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedStandardScaler<F> {
    /// Return the per-column means learned during fitting.
    #[must_use]
    pub fn mean(&self) -> &Array1<F> {
        &self.mean
    }

    /// Return the per-column standard deviations learned during fitting.
    #[must_use]
    pub fn std(&self) -> &Array1<F> {
        &self.std
    }

    /// Inverse-transform scaled data back to original space.
    ///
    /// Applies `x_orig = x_scaled * scale + mean` per column, where constant
    /// (zero-variance) columns use an effective scale of 1 (matching sklearn
    /// `_handle_zeros_in_scale`, `_data.py:88`, `:1106-1109`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    pub fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.mean.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedStandardScaler::inverse_transform".into(),
            });
        }
        let mut out = x.to_owned();
        for (mut col, (&m, &s)) in out
            .columns_mut()
            .into_iter()
            .zip(self.mean.iter().zip(self.std.iter()))
        {
            // Constant (zero-variance) columns use an effective scale of 1,
            // matching sklearn's `scale_` (`_data.py:88`, `:1106-1109`).
            let s_eff = if s == F::zero() { F::one() } else { s };
            for v in &mut col {
                *v = *v * s_eff + m;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for StandardScaler<F> {
    type Fitted = FittedStandardScaler<F>;
    type Error = FerroError;

    /// Fit the scaler by computing per-column means and standard deviations.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedStandardScaler<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "StandardScaler::fit".into(),
            });
        }

        let n = F::from(n_samples).unwrap_or_else(F::one);
        let n_features = x.ncols();
        let mut mean = Array1::zeros(n_features);
        let mut std_arr = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            let m = col.iter().copied().fold(F::zero(), |acc, v| acc + v) / n;
            let variance = col
                .iter()
                .copied()
                .map(|v| (v - m) * (v - m))
                .fold(F::zero(), |acc, v| acc + v)
                / n;
            mean[j] = m;
            std_arr[j] = variance.sqrt();
        }

        Ok(FittedStandardScaler { mean, std: std_arr })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedStandardScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by subtracting the mean and dividing by the standard deviation.
    ///
    /// Constant (zero-variance) columns use an effective scale of 1, so each
    /// entry maps to `(x - mean) / 1 = 0` (matching scikit-learn
    /// `_handle_zeros_in_scale`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.mean.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedStandardScaler::transform".into(),
            });
        }
        let mut out = x.to_owned();
        for (mut col, (&m, &s)) in out
            .columns_mut()
            .into_iter()
            .zip(self.mean.iter().zip(self.std.iter()))
        {
            // Constant (zero-variance) columns use an effective scale of 1,
            // matching sklearn `_handle_zeros_in_scale` (`_data.py:88`,
            // `:1019-1021`): `(x - mean) / 1 = 0` since `x == mean`.
            let s_eff = if s == F::zero() { F::one() } else { s };
            for v in &mut col {
                *v = (*v - m) / s_eff;
            }
        }
        Ok(out)
    }
}

/// Implement `Transform` on the unfitted scaler to satisfy the `FitTransform: Transform`
/// supertrait bound. Calling `transform` on an unfitted scaler always returns an error
/// because no statistics have been learned yet.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for StandardScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the scaler must be fitted first.
    ///
    /// Use [`Fit::fit`] to produce a [`FittedStandardScaler`], then call
    /// [`Transform::transform`] on that.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "StandardScaler".into(),
            reason: "scaler must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for StandardScaler<F> {
    type FitError = FerroError;

    /// Fit the scaler on `x` and return the scaled output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails (e.g., zero rows).
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for StandardScaler<F> {
    /// Fit the scaler using the pipeline interface.
    ///
    /// The `y` argument is ignored; it exists only for API compatibility.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedStandardScaler<F> {
    /// Transform data using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_standard_scaler_zero_mean_unit_variance() {
        let scaler = StandardScaler::<f64>::new();
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();

        // Each column should have mean ~0
        for j in 0..scaled.ncols() {
            let col_mean: f64 = scaled.column(j).iter().sum::<f64>() / scaled.nrows() as f64;
            assert_abs_diff_eq!(col_mean, 0.0, epsilon = 1e-10);
        }

        // Each column should have population std ~1
        for j in 0..scaled.ncols() {
            let col_mean: f64 = scaled.column(j).iter().sum::<f64>() / scaled.nrows() as f64;
            let variance: f64 = scaled
                .column(j)
                .iter()
                .map(|&v| (v - col_mean).powi(2))
                .sum::<f64>()
                / scaled.nrows() as f64;
            assert_abs_diff_eq!(variance, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let scaler = StandardScaler::<f64>::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&scaled).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_constant_column_maps_to_zero() {
        let scaler = StandardScaler::<f64>::new();
        // Column 1 is constant: std = 0
        let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.std()[1], 0.0, epsilon = 1e-15);
        let scaled = fitted.transform(&x).unwrap();
        // Constant column maps to (x - mean)/1 = 0 (sklearn with_mean=True).
        for i in 0..3 {
            assert_abs_diff_eq!(scaled[[i, 1]], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let scaler = StandardScaler::<f64>::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let via_fit_transform = scaler.fit_transform(&x).unwrap();
        let fitted = scaler.fit(&x, &()).unwrap();
        let via_separate = fitted.transform(&x).unwrap();
        for (a, b) in via_fit_transform.iter().zip(via_separate.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let scaler = StandardScaler::<f64>::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = scaler.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_insufficient_samples_error() {
        let scaler = StandardScaler::<f64>::new();
        let x: Array2<f64> = Array2::zeros((0, 3));
        assert!(scaler.fit(&x, &()).is_err());
    }

    #[test]
    fn test_f32_scaler() {
        let scaler = StandardScaler::<f32>::new();
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();
        let col0_mean: f32 = scaled.column(0).iter().sum::<f32>() / 3.0;
        assert!((col0_mean).abs() < 1e-6);
    }
}
