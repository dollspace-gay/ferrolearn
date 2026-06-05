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
//! (`lib.rs`, grandfathered S5). HONEST (R-HONEST-3): `with_mean`/`with_std` constructor params
//! gate conditional center/scale (`_data.py:1064-1067`); the per-column standardize VALUES match
//! sklearn on every flag configuration.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (per-column standardize value match, non-constant) | SHIPPED | `Fit::fit` learns per-column `mean`/`std=sqrt(population var ddof=0)`; `Transform::transform` = `(x-mean)/std`, mirroring sklearn `X -= mean_; X /= scale_` (`_data.py:1064-1067`). Critic-verified bit-identical to live oracle: `green_req1_value_match_non_constant` (`[[-1.2247..],[0],[1.2247..]]`), `green_req1_mean_and_scale_attributes` (mean()/std() == sklearn mean_/scale_ on non-constant), `green_req1_negative_decimal_fixture`. Consumers: PyO3 `_RsStandardScaler` + `FittedPipelineTransformer` + re-export. |
//! | REQ-2 (constant / zero-variance column → 0) | SHIPPED | FIXED #1191. `transform` uses `s_eff = if s==0 {1} else {s}` → constant col `(x-mean)/1 = 0`, matching sklearn `_handle_zeros_in_scale` (`_data.py:88`,`:1019-1021`); `inverse_transform` aligned to `x*s_eff+m`. Critic two-round CLEAN: 9 tests incl. constant→0 (default/single-row/mixed) + inverse non-round-trip on constant col matches `x*1+mean`. In-module test corrected (R-HONEST-4). |
//! | REQ-3 (inverse_transform round-trip) | SHIPPED | `inverse_transform` = `x*s_eff + mean`, mirroring sklearn `X *= scale_; X += mean_` (`_data.py:1106-1109`); `inverse_transform(transform(X))==X` (`green_req3_inverse_roundtrip`). Consumer: PyO3 `_RsStandardScaler::inverse_transform`. |
//! | REQ-4 (PyO3 binding) | SHIPPED | `_RsStandardScaler` (`transformers.rs:12`, registered `lib.rs:22`) marshals `fit`/`transform`/`inverse_transform`/`mean_` over `FittedStandardScaler<f64>` — a real CPython consumer. |
//! | REQ-5 (var_/scale_/n_samples_seen_ attrs) | SHIPPED | FIXED #1192. `FittedStandardScaler<F>` now stores `var_` (population variance ddof=0, `0.0` on a constant column), `scale_` (= `_handle_zeros_in_scale(sqrt(var_))`: `std` on non-constant cols, `1.0` on constant cols — same zero-handling as `transform`'s `s_eff`), and `n_samples_seen_` (= n_rows), set in `Fit::fit`; exposed by getters `var()`/`scale()`/`n_samples_seen()` mirroring sklearn `var_`/`scale_`/`n_samples_seen_` (`_data.py:1013-1023`). Critic-grounded vs live oracle `X=[[1,5],[2,5],[3,5]]` (`var_=[0.6666666666666666,0.0]`, `scale_=[0.816496580927726,1.0]`, `n_samples_seen_=3`): `standard_scaler_var_scale_nsamples_match_sklearn`, `standard_scaler_scale_differs_from_std_on_constant_col`, `standard_scaler_var_equals_std_squared_nonconstant`. Additive getters; `transform`/`inverse_transform` unchanged. |
//! | REQ-6 (with_mean/with_std/copy params) | SHIPPED | FIXED #1193. `StandardScaler<F>` now carries `with_mean`/`with_std`/`copy` (default `true`, `new()`/`Default`), mirroring sklearn `__init__(*, copy=True, with_mean=True, with_std=True)` + `_parameter_constraints` (`_data.py:829-838`), with `#[must_use]` builders `with_with_mean`/`with_with_std`/`with_copy`. `with_mean`/`with_std` thread into `FittedStandardScaler<F>` (set in `Fit::fit`); `transform` is conditional `if with_mean: X -= mean_; if with_std: X /= scale_` and `inverse_transform` mirrors `if with_std: X *= scale_; if with_mean: X += mean_` (`_data.py:1064-1067`,`:1106-1109`). `copy` is ABI-only (fit/transform operate on owned copies). Critic-grounded vs live oracle `X=[[1,10],[2,20],[3,30]]`: default (T,T) → `[[-1.2247..],[0],[1.2247..]]` (regression-guarded bit-identical to pre-existing default fit), with_std=false (T,F) → center-only `[[-1,-10],[0,0],[1,10]]`, with_mean=false (F,T) → scale-only `[[1.2247..],[2.4495..],[3.6742..]]`, both-false (F,F) → identity, and inverse round-trip for all 4 configs. R-DEV-4 DEVIATION: sklearn nulls `mean_`/`scale_`/`var_` when a flag is `False`; ferrolearn always materializes them (getters return `&Array1`, not `Option`) — the flags govern transform APPLICATION so OUTPUT matches sklearn exactly. |
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
    /// Whether to center the data (subtract the per-column mean) on transform.
    ///
    /// Mirrors sklearn `StandardScaler(with_mean=True)` (`_data.py:835-838`):
    /// when `false`, [`Transform::transform`] does NOT subtract the mean
    /// (`if with_mean: X -= mean_`, `_data.py:1064-1065`). Default `true`.
    pub with_mean: bool,
    /// Whether to scale the data (divide by the per-column scale) on transform.
    ///
    /// Mirrors sklearn `StandardScaler(with_std=True)` (`_data.py:835-838`):
    /// when `false`, [`Transform::transform`] does NOT divide by the scale
    /// (`if with_std: X /= scale_`, `_data.py:1066-1067`). Default `true`.
    pub with_std: bool,
    /// Whether `fit`/`transform` should copy the input rather than mutate it.
    ///
    /// Mirrors sklearn `StandardScaler(copy=True)` (`_data.py:835-838`). In
    /// ferrolearn this is ABI-only: `fit`/`transform` never mutate the caller's
    /// array (they operate on owned copies), so this flag is stored for API
    /// parity but does not change observable behavior. Default `true`.
    pub copy: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> StandardScaler<F> {
    /// Create a new `StandardScaler` with default configuration.
    ///
    /// Defaults mirror sklearn `StandardScaler(*, copy=True, with_mean=True,
    /// with_std=True)` (`_data.py:835-838`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            with_mean: true,
            with_std: true,
            copy: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set whether to center (subtract the mean) on transform.
    ///
    /// Mirrors sklearn's `with_mean` constructor parameter (`_data.py:835-838`):
    /// `false` disables centering (`if with_mean: X -= mean_`,
    /// `_data.py:1064-1065`).
    #[must_use]
    pub fn with_with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    /// Set whether to scale (divide by the scale) on transform.
    ///
    /// Mirrors sklearn's `with_std` constructor parameter (`_data.py:835-838`):
    /// `false` disables scaling (`if with_std: X /= scale_`,
    /// `_data.py:1066-1067`).
    #[must_use]
    pub fn with_with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }

    /// Set the `copy` flag (ABI-only in ferrolearn).
    ///
    /// Mirrors sklearn's `copy` constructor parameter (`_data.py:835-838`).
    /// ferrolearn always operates on owned copies, so this does not change
    /// observable behavior.
    #[must_use]
    pub fn with_copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
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
///
/// # Deviation (R-DEV-4): fitted attributes are always materialized
///
/// sklearn sets `mean_` / `scale_` / `var_` to `None` when the corresponding
/// flag (`with_mean` / `with_std`) is `False` (`_data.py:986-991`,`:1019-1021`).
/// ferrolearn always materializes `mean`/`var_`/`scale_`/`n_samples_seen_`
/// regardless of the flags, because the `&Array1<F>` getters cannot return
/// `None` without changing their return type. The `with_mean`/`with_std` flags
/// instead govern *application* in [`Transform::transform`] /
/// [`FittedStandardScaler::inverse_transform`] (`if with_mean: X -= mean_`,
/// `if with_std: X /= scale_`, `_data.py:1064-1067`), so the transform OUTPUT
/// matches sklearn exactly. The getters are intentionally NOT `Option`.
#[derive(Debug, Clone)]
pub struct FittedStandardScaler<F> {
    /// Per-column means learned during fitting.
    pub(crate) mean: Array1<F>,
    /// Per-column standard deviations learned during fitting.
    pub(crate) std: Array1<F>,
    /// Per-column population variance (ddof=0) learned during fitting.
    ///
    /// Mirrors sklearn `StandardScaler.var_` (`_data.py:1013-1023`).
    pub(crate) var_: Array1<F>,
    /// Per-column scale with zeros replaced by 1 (`_handle_zeros_in_scale`).
    ///
    /// Mirrors sklearn `StandardScaler.scale_` (`_data.py:1013-1023`).
    pub(crate) scale_: Array1<F>,
    /// Number of samples (rows) seen during fitting.
    ///
    /// Mirrors sklearn `StandardScaler.n_samples_seen_` (`_data.py:1013-1023`).
    pub(crate) n_samples_seen_: usize,
    /// Whether `transform`/`inverse_transform` center (subtract/add the mean).
    ///
    /// Copied from the unfitted [`StandardScaler::with_mean`] in [`Fit::fit`];
    /// governs `if with_mean: X -= mean_` (`_data.py:1064-1065`).
    pub(crate) with_mean: bool,
    /// Whether `transform`/`inverse_transform` scale (divide/multiply by scale).
    ///
    /// Copied from the unfitted [`StandardScaler::with_std`] in [`Fit::fit`];
    /// governs `if with_std: X /= scale_` (`_data.py:1066-1067`).
    pub(crate) with_std: bool,
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

    /// Return the per-column population variance (ddof=0) learned during fitting.
    ///
    /// Mirrors sklearn `StandardScaler.var_` (`_data.py:1013-1023`): the raw
    /// variance, which is `0.0` on a constant column.
    #[must_use]
    pub fn var(&self) -> &Array1<F> {
        &self.var_
    }

    /// Return the per-column scale learned during fitting.
    ///
    /// Mirrors sklearn `StandardScaler.scale_` (`_data.py:1013-1023`,
    /// `_handle_zeros_in_scale`, `:88`): equals `std()` on non-constant columns
    /// and `1.0` on constant (zero-variance) columns.
    #[must_use]
    pub fn scale(&self) -> &Array1<F> {
        &self.scale_
    }

    /// Return the number of samples (rows) seen during fitting.
    ///
    /// Mirrors sklearn `StandardScaler.n_samples_seen_` (`_data.py:1013-1023`).
    #[must_use]
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen_
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
            .zip(self.mean.iter().zip(self.scale_.iter()))
        {
            // Conditional un-scale/un-center mirroring sklearn `inverse_transform`
            // (`_data.py:1106-1109`): `if with_std: X *= scale_` then
            // `if with_mean: X += mean_`. `scale_` has zero-variance columns
            // replaced by 1 (`_handle_zeros_in_scale`, `_data.py:88`).
            for v in &mut col {
                if self.with_std {
                    *v = *v * s;
                }
                if self.with_mean {
                    *v = *v + m;
                }
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
        let mut var_arr = Array1::zeros(n_features);

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
            var_arr[j] = variance;
            std_arr[j] = variance.sqrt();
        }

        // `scale_` mirrors sklearn `_handle_zeros_in_scale(sqrt(var_))`
        // (`_data.py:88`, `:1019-1021`): the std with zero-variance columns
        // replaced by 1.0 — exactly the `s_eff` used in `transform`.
        let scale_ = std_arr.mapv(|s| if s == F::zero() { F::one() } else { s });

        Ok(FittedStandardScaler {
            mean,
            std: std_arr,
            var_: var_arr,
            scale_,
            n_samples_seen_: n_samples,
            with_mean: self.with_mean,
            with_std: self.with_std,
        })
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
            .zip(self.mean.iter().zip(self.scale_.iter()))
        {
            // Conditional center/scale mirroring sklearn `transform`
            // (`_data.py:1064-1067`): `if with_mean: X -= mean_` then
            // `if with_std: X /= scale_`. `scale_` already has zero-variance
            // columns replaced by 1 (`_handle_zeros_in_scale`, `_data.py:88`),
            // so a constant column maps to `(x - mean) / 1 = 0`. When
            // `with_mean && with_std` this is byte-identical to the prior path.
            for v in &mut col {
                if self.with_mean {
                    *v = *v - m;
                }
                if self.with_std {
                    *v = *v / s;
                }
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

    // sklearn 1.5.2 oracle (R-CHAR-3), X = [[1,5],[2,5],[3,5]] (col 1 constant):
    //   StandardScaler().fit(X) -> mean_ = [2.0, 5.0],
    //     var_   = [0.6666666666666666, 0.0],
    //     scale_ = [0.816496580927726, 1.0],
    //     n_samples_seen_ = 3.
    #[test]
    fn standard_scaler_var_scale_nsamples_match_sklearn() -> Result<(), FerroError> {
        let scaler = StandardScaler::<f64>::new();
        let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let fitted = scaler.fit(&x, &())?;

        assert_abs_diff_eq!(fitted.var()[0], 0.6666666666666666, epsilon = 1e-9);
        assert_abs_diff_eq!(fitted.var()[1], 0.0, epsilon = 1e-9);

        assert_abs_diff_eq!(fitted.scale()[0], 0.816496580927726, epsilon = 1e-12);
        // scale_[1] is EXACTLY 1.0 on the constant column (_handle_zeros), not 0.0.
        assert_eq!(fitted.scale()[1], 1.0);

        assert_eq!(fitted.n_samples_seen(), 3);
        Ok(())
    }

    #[test]
    fn standard_scaler_scale_differs_from_std_on_constant_col() -> Result<(), FerroError> {
        let scaler = StandardScaler::<f64>::new();
        let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let fitted = scaler.fit(&x, &())?;

        // Constant column: raw std is 0.0 but scale_ is 1.0 (_handle_zeros).
        assert_eq!(fitted.std()[1], 0.0);
        assert_eq!(fitted.scale()[1], 1.0);
        // Non-constant column: scale_ equals raw std unchanged.
        assert_eq!(fitted.scale()[0], fitted.std()[0]);
        Ok(())
    }

    #[test]
    fn standard_scaler_var_equals_std_squared_nonconstant() -> Result<(), FerroError> {
        let scaler = StandardScaler::<f64>::new();
        let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let fitted = scaler.fit(&x, &())?;

        assert_abs_diff_eq!(
            fitted.var()[0],
            fitted.std()[0] * fitted.std()[0],
            epsilon = 1e-12
        );
        Ok(())
    }

    // sklearn 1.5.2 oracle (R-CHAR-3), X = [[1,10],[2,20],[3,30]]:
    //   StandardScaler(with_mean=True,  with_std=True ).fit_transform(X)
    //     -> [[-1.224744871391589,-1.224744871391589],[0,0],[1.224744871391589,1.224744871391589]]
    //   StandardScaler(with_mean=True,  with_std=False).fit_transform(X)
    //     -> [[-1,-10],[0,0],[1,10]]                       (center only)
    //   StandardScaler(with_mean=False, with_std=True ).fit_transform(X)
    //     -> [[1.224744871391589,1.224744871391589],
    //         [2.449489742783178,2.449489742783178],
    //         [3.6742346141747673,3.6742346141747673]]     (scale only)
    //   StandardScaler(with_mean=False, with_std=False).fit_transform(X)
    //     -> X                                              (identity)
    #[test]
    fn standard_scaler_with_mean_std_default_matches_sklearn() -> Result<(), FerroError> {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let scaler = StandardScaler::<f64>::new();
        let fitted = scaler.fit(&x, &())?;
        let scaled = fitted.transform(&x)?;

        let expected = array![
            [-1.224744871391589, -1.224744871391589],
            [0.0, 0.0],
            [1.224744871391589, 1.224744871391589]
        ];
        for (a, b) in scaled.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-7);
        }

        // Regression guard: default (with_mean=true, with_std=true) must be
        // BYTE-identical to a plain default fit produced the same way.
        let default_fitted = StandardScaler::<f64>::new().fit(&x, &())?;
        let default_scaled = default_fitted.transform(&x)?;
        for (a, b) in scaled.iter().zip(default_scaled.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        Ok(())
    }

    #[test]
    fn standard_scaler_with_std_false() -> Result<(), FerroError> {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let scaler = StandardScaler::<f64>::new().with_with_std(false);
        let fitted = scaler.fit(&x, &())?;
        let scaled = fitted.transform(&x)?;

        let expected = array![[-1.0, -10.0], [0.0, 0.0], [1.0, 10.0]];
        for (a, b) in scaled.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-7);
        }
        Ok(())
    }

    #[test]
    fn standard_scaler_with_mean_false() -> Result<(), FerroError> {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let scaler = StandardScaler::<f64>::new().with_with_mean(false);
        let fitted = scaler.fit(&x, &())?;
        let scaled = fitted.transform(&x)?;

        let expected = array![
            [1.224744871391589, 1.224744871391589],
            [2.449489742783178, 2.449489742783178],
            [3.6742346141747673, 3.6742346141747673]
        ];
        for (a, b) in scaled.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-7);
        }
        Ok(())
    }

    #[test]
    fn standard_scaler_both_false_identity() -> Result<(), FerroError> {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let scaler = StandardScaler::<f64>::new()
            .with_with_mean(false)
            .with_with_std(false);
        let fitted = scaler.fit(&x, &())?;
        let scaled = fitted.transform(&x)?;

        for (a, b) in scaled.iter().zip(x.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
        Ok(())
    }

    #[test]
    fn standard_scaler_inverse_roundtrip_each_config() -> Result<(), FerroError> {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        for &(with_mean, with_std) in &[(true, true), (true, false), (false, true), (false, false)]
        {
            let scaler = StandardScaler::<f64>::new()
                .with_with_mean(with_mean)
                .with_with_std(with_std);
            let fitted = scaler.fit(&x, &())?;
            let scaled = fitted.transform(&x)?;
            let recovered = fitted.inverse_transform(&scaled)?;
            for (a, b) in x.iter().zip(recovered.iter()) {
                assert_abs_diff_eq!(a, b, epsilon = 1e-9);
            }
        }
        Ok(())
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
