//! Min-max scaler: scales each feature to a given range.
//!
//! Each feature is transformed as:
//! `x_scaled = (x - min) / (max - min) * (range_max - range_min) + range_min`
//!
//! The default feature range is `[0, 1]`.
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_data.py` (`class MinMaxScaler`
//! `:291`, `minmax_scale` `:589`). Design doc: `.design/preprocess/min_max_scaler.md`. Expected
//! values from the live sklearn 1.5.2 oracle (R-CHAR-3). Consumers: PyO3 `_RsMinMaxScaler`
//! (`ferrolearn-python/src/extras.rs:1148`) + `PipelineTransformer` impl + crate re-export
//! (`lib.rs:118`, grandfathered S5).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (per-column min-max value match, non-constant) | SHIPPED | `Fit::fit` learns per-column `data_min`/`data_max`; `Transform::transform` = `(x-min)/(max-min)*range_width+range_min`, mirroring sklearn affine `scale_=(fr1-fr0)/data_range` (`_data.py:508`), `min_=fr0-data_min*scale_` (`:511`). Critic-verified bit-identical to live oracle: `req1_default_range_value_match`, `req1_custom_range_value_match`, `req1_data_min_max_attributes`, `req1_nontrivial_negative_decimal_value_match` in `tests/divergence_min_max_scaler.rs`. Consumers: PyO3 `_RsMinMaxScaler` + `FittedPipelineTransformer` + re-export `lib.rs:118`. |
//! | REQ-2 (constant column → `feature_range[0]`) | SHIPPED | FIXED #1170. `transform` zero-span branch now sets a constant column to `range_min`, matching sklearn `_handle_zeros_in_scale` (`_data.py:88`,`:508-511`). Critic two-round CLEAN: 11 tests incl. constant col → `fr[0]` for default/(−1,1)/(2,5), mixed fixture, negative/zero constant, single-row fit. In-module test corrected (R-HONEST-4). |
//! | REQ-3 (feature_range validation) | SHIPPED | `with_feature_range` returns `Err(InvalidParameter)` when `range_min >= range_max`, matching sklearn "Minimum of desired feature range must be smaller than maximum" (`_data.py:476-480`). Guard `req3_feature_range_validation_rejects`. |
//! | REQ-7 (PyO3 binding) | SHIPPED | `_RsMinMaxScaler` (`extras.rs:1148`, registered `lib.rs:81`) marshals `fit`/`transform` over `FittedMinMaxScaler<f64>` (default range) — a real CPython consumer of REQ-1/REQ-2. |
//! | REQ-4 (NaN tolerance: allow-nan + nanmin/nanmax) | NOT-STARTED | open prereq blocker #1171. `fit` reduce-min/max poisons on NaN; sklearn `force_all_finite='allow-nan'` + `_nanmin`/`_nanmax` (`_data.py:490-499`). |
//! | REQ-5 (scale_/min_/data_range_/n_samples_seen_) | NOT-STARTED | open prereq blocker #1172. Only `data_min`/`data_max` stored (`_data.py:508-514`). |
//! | REQ-6 (inverse_transform) | NOT-STARTED | open prereq blocker #1173. None (`_data.py:549-587`). |
//! | REQ-8 (partial_fit / streaming) | NOT-STARTED | open prereq blocker #1174. Single-shot fit (`_data.py:489-515`). |
//! | REQ-9 (minmax_scale free fn + axis) | NOT-STARTED | open prereq blocker #1175. No free fn / axis=1 (`_data.py:589`). |
//! | REQ-10 (copy / clip params) | NOT-STARTED | open prereq blocker #1176. No `copy`/`clip`/`_parameter_constraints` (`_data.py:447-459`,`:542-543`). |
//! | REQ-11 (get_feature_names_out / n_features_in_) | NOT-STARTED | open prereq blocker #1177. None (OneToOneFeatureMixin). |
//! | REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1178. `ndarray`+`num_traits`, not `ferray-core` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// MinMaxScaler (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted min-max scaler.
///
/// Calling [`Fit::fit`] learns the per-column minimum and maximum values and
/// returns a [`FittedMinMaxScaler`] that can transform new data.
///
/// Constant columns where `max == min` are mapped to `feature_range[0]`
/// after transformation (matching scikit-learn).
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::MinMaxScaler;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let scaler = MinMaxScaler::<f64>::new();
/// let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
/// let fitted = scaler.fit(&x, &()).unwrap();
/// let scaled = fitted.transform(&x).unwrap();
/// // Values are now in [0, 1]
/// ```
#[derive(Debug, Clone)]
pub struct MinMaxScaler<F> {
    /// Target feature range `(min, max)`. Defaults to `(0, 1)`.
    pub(crate) feature_range: (F, F),
}

impl<F: Float + Send + Sync + 'static> MinMaxScaler<F> {
    /// Create a new `MinMaxScaler` scaling to the default range `[0, 1]`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            feature_range: (F::zero(), F::one()),
        }
    }

    /// Create a new `MinMaxScaler` with a custom target feature range.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `range_min >= range_max`.
    pub fn with_feature_range(range_min: F, range_max: F) -> Result<Self, FerroError> {
        if range_min >= range_max {
            return Err(FerroError::InvalidParameter {
                name: "feature_range".into(),
                reason: "range_min must be strictly less than range_max".into(),
            });
        }
        Ok(Self {
            feature_range: (range_min, range_max),
        })
    }

    /// Return the configured feature range.
    #[must_use]
    pub fn feature_range(&self) -> (F, F) {
        self.feature_range
    }
}

impl<F: Float + Send + Sync + 'static> Default for MinMaxScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedMinMaxScaler
// ---------------------------------------------------------------------------

/// A fitted min-max scaler holding per-column min, max, and the target range.
///
/// Created by calling [`Fit::fit`] on a [`MinMaxScaler`].
#[derive(Debug, Clone)]
pub struct FittedMinMaxScaler<F> {
    /// Per-column minimum values learned during fitting.
    pub(crate) data_min: Array1<F>,
    /// Per-column maximum values learned during fitting.
    pub(crate) data_max: Array1<F>,
    /// The target output range `(range_min, range_max)`.
    pub(crate) feature_range: (F, F),
}

impl<F: Float + Send + Sync + 'static> FittedMinMaxScaler<F> {
    /// Return the per-column minimum values learned during fitting.
    #[must_use]
    pub fn data_min(&self) -> &Array1<F> {
        &self.data_min
    }

    /// Return the per-column maximum values learned during fitting.
    #[must_use]
    pub fn data_max(&self) -> &Array1<F> {
        &self.data_max
    }

    /// Return the configured target feature range.
    #[must_use]
    pub fn feature_range(&self) -> (F, F) {
        self.feature_range
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for MinMaxScaler<F> {
    type Fitted = FittedMinMaxScaler<F>;
    type Error = FerroError;

    /// Fit the scaler by computing per-column minimum and maximum values.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedMinMaxScaler<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MinMaxScaler::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut data_min = Array1::zeros(n_features);
        let mut data_max = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            let min = col
                .iter()
                .copied()
                .reduce(|a, b| if a < b { a } else { b })
                .unwrap_or_else(F::zero);
            let max = col
                .iter()
                .copied()
                .reduce(|a, b| if a > b { a } else { b })
                .unwrap_or_else(F::zero);
            data_min[j] = min;
            data_max[j] = max;
        }

        Ok(FittedMinMaxScaler {
            data_min,
            data_max,
            feature_range: self.feature_range,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedMinMaxScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by scaling each feature to the configured range.
    ///
    /// Constant columns where `data_max == data_min` are mapped to
    /// `feature_range[0]` (matching scikit-learn).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.data_min.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedMinMaxScaler::transform".into(),
            });
        }

        let (range_min, range_max) = self.feature_range;
        let range_width = range_max - range_min;

        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            let min = self.data_min[j];
            let max = self.data_max[j];
            let span = max - min;
            if span == F::zero() {
                // Constant column: sklearn maps it to feature_range[0]
                // (_handle_zeros_in_scale, _data.py:88,508-511).
                for v in &mut col {
                    *v = range_min;
                }
                continue;
            }
            for v in &mut col {
                *v = (*v - min) / span * range_width + range_min;
            }
        }
        Ok(out)
    }
}

/// Implement `Transform` on the unfitted scaler to satisfy the `FitTransform: Transform`
/// supertrait bound. Calling `transform` on an unfitted scaler always returns an error.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for MinMaxScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the scaler must be fitted first.
    ///
    /// Use [`Fit::fit`] to produce a [`FittedMinMaxScaler`], then call
    /// [`Transform::transform`] on that.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "MinMaxScaler".into(),
            reason: "scaler must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for MinMaxScaler<F> {
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
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for MinMaxScaler<F> {
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedMinMaxScaler<F> {
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
    fn test_min_max_scaler_default_range() {
        let scaler = MinMaxScaler::<f64>::new();
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();

        // Min should be 0, max should be 1
        for j in 0..scaled.ncols() {
            let col_min = scaled
                .column(j)
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let col_max = scaled
                .column(j)
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            assert_abs_diff_eq!(col_min, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(col_max, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_min_max_scaler_custom_range() {
        let scaler = MinMaxScaler::<f64>::with_feature_range(-1.0, 1.0).unwrap();
        let x = array![[0.0], [5.0], [10.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(scaled[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[2, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_feature_range() {
        assert!(MinMaxScaler::<f64>::with_feature_range(1.0, 0.0).is_err());
        assert!(MinMaxScaler::<f64>::with_feature_range(1.0, 1.0).is_err());
    }

    #[test]
    fn test_constant_column_maps_to_range_min() {
        let scaler = MinMaxScaler::<f64>::new();
        let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();
        // Constant column (col 0) maps to feature_range[0] = 0.0 (sklearn parity).
        // Live oracle: MinMaxScaler().fit_transform([[5,1],[5,2],[5,3]])
        //   -> [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]].
        let expected_col1 = [0.0, 0.5, 1.0];
        for i in 0..3 {
            assert_abs_diff_eq!(scaled[[i, 0]], 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(scaled[[i, 1]], expected_col1[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let scaler = MinMaxScaler::<f64>::new();
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
        let scaler = MinMaxScaler::<f64>::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = scaler.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }
}
