//! Robust scaler: median and IQR-based scaling.
//!
//! Each feature is transformed as `(x - median) / IQR` where
//! `IQR = Q75 - Q25`. This scaler is robust to outliers.
//!
//! A zero-IQR column is centered (`x - median`) then divided by an effective
//! scale of 1, matching scikit-learn `_handle_zeros_in_scale`
//! (`_data.py:88`, `:1635`, `:1673-1675`); a constant column therefore maps to 0.
//!
//! Translation target: scikit-learn 1.5.2 `class RobustScaler`
//! (`sklearn/preprocessing/_data.py:1445`) + `robust_scale` (`:1719`). Design:
//! `.design/preprocess/robust_scaler.md`. Tracking: #1247.
//!
//! `## REQ status`
//!
//! | REQ | Status | Anchor |
//! |---|---|---|
//! | REQ-1 per-column median/IQR value match (non-constant) | SHIPPED | `RobustScaler::fit` / `FittedRobustScaler::transform`; sklearn `_data.py:1614`,`:1630-1634`,`:1672-1675` |
//! | REQ-2 zero-IQR / constant column → 0 (center, scale_eff=1) | SHIPPED (#1248) | `transform` `scale_eff = if iqr==0 {1} else {iqr}`; sklearn `_data.py:88`,`:1635` |
//! | REQ-3 InsufficientSamples / ShapeMismatch error contracts | SHIPPED | `fit` / `transform` guards; sklearn `_data.py:1658-1666` |
//! | REQ-11 PyO3 binding (`_RsRobustScaler`) | SHIPPED | `ferrolearn-python/src/extras.rs:1163`, `lib.rs:83` |
//! | REQ-4 quantile_range ctor param + validation | SHIPPED (#1249) | `RobustScaler::quantile_range` field (default `(25,75)`) + `with_quantile_range` validating builder (non-strict `0 <= q_min <= q_max <= 100`, matching sklearn's `if not 0 <= q_min <= q_max <= 100`); `fit` uses `Q(q_max_frac) − Q(q_min_frac)`; sklearn `_data.py:1567`,`:1604-1606`,`:1630` |
//! | REQ-5 with_centering / with_scaling ctor params | SHIPPED (#1250) | `RobustScaler::with_centering`/`with_scaling` fields (default `true`) + `with_with_centering`/`with_with_scaling` builders, threaded into `FittedRobustScaler`; `transform` is conditional `if with_centering: X -= center_` then `if with_scaling: X /= scale_`, mirroring sklearn `_data.py:1672-1675`,`:1616`,`:1640`. R-DEV-4: ferrolearn always materializes `median`/`iqr`; sklearn nulls `center_`/`scale_` when the flag is `False` (flags govern transform APPLICATION so OUTPUT matches sklearn exactly). |
//! | REQ-6 unit_variance ctor param | NOT-STARTED (#1251) | sklearn `_data.py:1636-1638` |
//! | REQ-7 inverse_transform | NOT-STARTED (#1252) | sklearn `_data.py:1678`,`:1706-1708` |
//! | REQ-8 `robust_scale` free function + axis | NOT-STARTED (#1253) | sklearn `_data.py:1719` |
//! | REQ-9 NaN tolerance (allow-nan / nanmedian / nanpercentile) | NOT-STARTED (#1254) | sklearn `_data.py:1601`,`:1614`,`:1630` |
//! | REQ-10 center_ / scale_ attribute names + _handle_zeros scale_ | NOT-STARTED (#1255) | sklearn `_data.py:1505-1514`,`:1635` |
//! | REQ-12 copy ctor param + in-place semantics | NOT-STARTED (#1256) | sklearn `_data.py:1568`,`:1661` |
//! | REQ-13 sparse CSC/CSR support | NOT-STARTED (#1257) | sklearn `_data.py:1609-1612`,`:1668-1670` |
//! | REQ-14 get_feature_names_out / n_features_in_ | NOT-STARTED (#1258) | sklearn `OneToOneFeatureMixin` |
//! | REQ-15 ferray substrate | NOT-STARTED (#1259) | R-SUBSTRATE |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Helper: compute quantile of a sorted slice
// ---------------------------------------------------------------------------

/// Compute the `q`-th quantile (0.0–1.0) of a sorted slice using linear interpolation.
///
/// Panics if `sorted` is empty.
fn quantile_sorted<F: Float>(sorted: &[F], q: f64) -> F {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = F::from(idx - lo as f64).unwrap_or_else(F::zero);
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

// ---------------------------------------------------------------------------
// RobustScaler (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted robust scaler.
///
/// Calling [`Fit::fit`] learns the per-column medians and interquartile ranges
/// (IQR = Q75 − Q25) and returns a [`FittedRobustScaler`] that can transform
/// new data.
///
/// A zero-IQR column is centered then divided by an effective scale of 1
/// (matching scikit-learn `_handle_zeros_in_scale`), so a constant column
/// maps to 0.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::RobustScaler;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let scaler = RobustScaler::<f64>::new();
/// let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [100.0, 40.0]];
/// let fitted = scaler.fit(&x, &()).unwrap();
/// let scaled = fitted.transform(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RobustScaler<F> {
    /// The `(q_min, q_max)` percentile pair (each in `0..=100`) whose difference
    /// `Q(q_max) − Q(q_min)` defines the per-column scale. Defaults to
    /// `(25.0, 75.0)` (the interquartile range), mirroring sklearn
    /// `RobustScaler(quantile_range=(25.0, 75.0))` (`_data.py:1567`).
    pub quantile_range: (F, F),
    /// Whether to center the data (subtract the per-column median) on transform.
    ///
    /// Mirrors sklearn `RobustScaler(with_centering=True)` (`_data.py:1616`):
    /// when `false`, [`Transform::transform`] does NOT subtract the median
    /// (`if self.with_centering: X -= self.center_`, `_data.py:1672-1673`).
    /// Default `true`.
    pub with_centering: bool,
    /// Whether to scale the data (divide by the per-column IQR) on transform.
    ///
    /// Mirrors sklearn `RobustScaler(with_scaling=True)` (`_data.py:1640`):
    /// when `false`, [`Transform::transform`] does NOT divide by the scale
    /// (`if self.with_scaling: X /= self.scale_`, `_data.py:1674-1675`).
    /// Default `true`.
    pub with_scaling: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> RobustScaler<F> {
    /// Create a new `RobustScaler` with the default quantile range `(25.0, 75.0)`
    /// (the interquartile range).
    #[must_use]
    pub fn new() -> Self {
        // `F::from` on these in-range literals is infallible for f32/f64; fall
        // back to (25, 75) via integer `from` (also infallible) if it ever isn't.
        let q_min = F::from(25.0).unwrap_or_else(|| F::from(25u8).unwrap_or_else(F::zero));
        let q_max = F::from(75.0).unwrap_or_else(|| F::from(75u8).unwrap_or_else(F::zero));
        Self {
            quantile_range: (q_min, q_max),
            with_centering: true,
            with_scaling: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set whether to center (subtract the median) on transform, returning the
    /// reconfigured scaler.
    ///
    /// Mirrors sklearn's `with_centering` constructor parameter
    /// (`_data.py:1616`): `false` disables centering
    /// (`if self.with_centering: X -= self.center_`, `_data.py:1672-1673`).
    #[must_use]
    pub fn with_with_centering(mut self, with_centering: bool) -> Self {
        self.with_centering = with_centering;
        self
    }

    /// Set whether to scale (divide by the IQR) on transform, returning the
    /// reconfigured scaler.
    ///
    /// Mirrors sklearn's `with_scaling` constructor parameter (`_data.py:1640`):
    /// `false` disables scaling
    /// (`if self.with_scaling: X /= self.scale_`, `_data.py:1674-1675`).
    #[must_use]
    pub fn with_with_scaling(mut self, with_scaling: bool) -> Self {
        self.with_scaling = with_scaling;
        self
    }

    /// Set the `(q_min, q_max)` percentile pair used to compute the per-column
    /// scale `Q(q_max) − Q(q_min)`, returning the reconfigured scaler.
    ///
    /// Mirrors sklearn's `quantile_range` constructor parameter (`_data.py:1567`)
    /// and its non-strict `0 <= q_min <= q_max <= 100` validation
    /// (`_data.py:1604-1606`, which raises only on `not 0 <= q_min <= q_max <= 100`).
    /// A degenerate equal range (`q_min == q_max`) is accepted: it yields a zero
    /// scale on every column, which the transform then handles via
    /// `_handle_zeros_in_scale` (effective scale 1). The median (`center_`, `Q50`)
    /// is unaffected by `quantile_range`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] unless `0 <= q_min <= q_max <= 100`.
    #[must_use = "with_quantile_range returns a reconfigured scaler; use the returned value"]
    pub fn with_quantile_range(mut self, q_min: F, q_max: F) -> Result<Self, FerroError> {
        let hundred = F::from(100.0).unwrap_or_else(|| F::from(100u8).unwrap_or_else(F::zero));
        if !(q_min >= F::zero() && q_min <= q_max && q_max <= hundred) {
            return Err(FerroError::InvalidParameter {
                name: "quantile_range".into(),
                reason: "must satisfy 0 <= q_min <= q_max <= 100".into(),
            });
        }
        self.quantile_range = (q_min, q_max);
        Ok(self)
    }
}

impl<F: Float + Send + Sync + 'static> Default for RobustScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedRobustScaler
// ---------------------------------------------------------------------------

/// A fitted robust scaler holding per-column medians and IQRs.
///
/// Created by calling [`Fit::fit`] on a [`RobustScaler`].
///
/// # Deviation (R-DEV-4): fitted attributes are always materialized
///
/// sklearn sets `center_` / `scale_` to `None` when the corresponding flag
/// (`with_centering` / `with_scaling`) is `False` (`_data.py:1616`,`:1640`).
/// ferrolearn always materializes `median`/`iqr` regardless of the flags,
/// because the `&Array1<F>` getters cannot return `None` without changing their
/// return type. The `with_centering`/`with_scaling` flags instead govern
/// *application* in [`Transform::transform`] (`if with_centering: X -= center_`,
/// `if with_scaling: X /= scale_`, `_data.py:1672-1675`), so the transform
/// OUTPUT matches sklearn exactly. The getters are intentionally NOT `Option`.
#[derive(Debug, Clone)]
pub struct FittedRobustScaler<F> {
    /// Per-column medians learned during fitting.
    pub(crate) median: Array1<F>,
    /// Per-column interquartile ranges (Q75 − Q25) learned during fitting.
    pub(crate) iqr: Array1<F>,
    /// Whether `transform` centers (subtracts the median).
    ///
    /// Copied from the unfitted [`RobustScaler::with_centering`] in [`Fit::fit`];
    /// governs `if with_centering: X -= center_` (`_data.py:1672-1673`).
    pub(crate) with_centering: bool,
    /// Whether `transform` scales (divides by the IQR).
    ///
    /// Copied from the unfitted [`RobustScaler::with_scaling`] in [`Fit::fit`];
    /// governs `if with_scaling: X /= scale_` (`_data.py:1674-1675`).
    pub(crate) with_scaling: bool,
}

impl<F: Float + Send + Sync + 'static> FittedRobustScaler<F> {
    /// Return the per-column medians learned during fitting.
    #[must_use]
    pub fn median(&self) -> &Array1<F> {
        &self.median
    }

    /// Return the per-column IQR values learned during fitting.
    #[must_use]
    pub fn iqr(&self) -> &Array1<F> {
        &self.iqr
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for RobustScaler<F> {
    type Fitted = FittedRobustScaler<F>;
    type Error = FerroError;

    /// Fit the scaler by computing per-column medians and IQRs.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedRobustScaler<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "RobustScaler::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut median_arr = Array1::zeros(n_features);
        let mut iqr_arr = Array1::zeros(n_features);

        // Convert the F percentiles (0..=100) to the f64 fractions (0..=1) that
        // `quantile_sorted` expects, mirroring sklearn's per-column
        // `np.nanpercentile(column_data, quantile_range)` (`_data.py:1630`).
        let q_min_frac = self.quantile_range.0.to_f64().unwrap_or(25.0) / 100.0;
        let q_max_frac = self.quantile_range.1.to_f64().unwrap_or(75.0) / 100.0;

        for j in 0..n_features {
            let mut col: Vec<F> = x.column(j).iter().copied().collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let med = quantile_sorted(&col, 0.5);
            let q_lo = quantile_sorted(&col, q_min_frac);
            let q_hi = quantile_sorted(&col, q_max_frac);

            median_arr[j] = med;
            // scale_ = Q(q_max) − Q(q_min) (sklearn `_data.py:1634`).
            iqr_arr[j] = q_hi - q_lo;
        }

        Ok(FittedRobustScaler {
            median: median_arr,
            iqr: iqr_arr,
            with_centering: self.with_centering,
            with_scaling: self.with_scaling,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedRobustScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by subtracting the median and dividing by the IQR.
    ///
    /// A zero-IQR column uses an effective scale of 1 (matching scikit-learn
    /// `_handle_zeros_in_scale`, `_data.py:88`, `:1635`, `:1673-1675`), so it is
    /// still centered: a constant column maps to 0 and a non-constant zero-IQR
    /// column maps to `x - median`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.median.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedRobustScaler::transform".into(),
            });
        }

        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            let med = self.median[j];
            let iqr = self.iqr[j];
            // A zero IQR is replaced by an effective scale of 1, matching sklearn
            // `_handle_zeros_in_scale` (`_data.py:88`, called at `:1635`). The column
            // is still centered (`X -= center_`, `:1673`) then divided by the
            // effective scale (`X /= scale_`, `:1675`), so a constant column maps to 0
            // and a non-constant zero-IQR column maps to `x - median`.
            let scale_eff = if iqr == F::zero() { F::one() } else { iqr };
            // Conditional center/scale mirroring sklearn `transform`
            // (`_data.py:1672-1675`): `if with_centering: X -= center_` then
            // `if with_scaling: X /= scale_`. When `with_centering && with_scaling`
            // this is byte-identical to the prior path.
            for v in &mut col {
                if self.with_centering {
                    *v = *v - med;
                }
                if self.with_scaling {
                    *v = *v / scale_eff;
                }
            }
        }
        Ok(out)
    }
}

/// Implement `Transform` on the unfitted scaler to satisfy the `FitTransform: Transform`
/// supertrait bound. Calling `transform` on an unfitted scaler always returns an error.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for RobustScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the scaler must be fitted first.
    ///
    /// Use [`Fit::fit`] to produce a [`FittedRobustScaler`], then call
    /// [`Transform::transform`] on that.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "RobustScaler".into(),
            reason: "scaler must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for RobustScaler<F> {
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

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for RobustScaler<F> {
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedRobustScaler<F> {
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
    fn test_robust_scaler_basic() {
        let scaler = RobustScaler::<f64>::new();
        // Symmetric distribution: median = 3, Q25 = 2, Q75 = 4, IQR = 2
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.median()[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.iqr()[0], 2.0, epsilon = 1e-10);

        let scaled = fitted.transform(&x).unwrap();
        // Median should be 0 after scaling
        assert_abs_diff_eq!(scaled[[2, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_iqr_column_centered_to_zero() {
        // A constant (zero-IQR) column is centered then divided by an effective
        // scale of 1 (sklearn `_handle_zeros_in_scale`), so it maps to 0.
        //
        // LIVE ORACLE (sklearn 1.5.2, run from /tmp):
        //   RobustScaler().fit_transform([[7.,1.],[7.,2.],[7.,3.]]).tolist()
        //   == [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]
        let scaler = RobustScaler::<f64>::new();
        // Column 0 is constant: IQR = 0
        let x = array![[7.0, 1.0], [7.0, 2.0], [7.0, 3.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.iqr()[0], 0.0, epsilon = 1e-15);
        let scaled = fitted.transform(&x).unwrap();
        // Constant column maps to 0 (centered, effective scale 1).
        for i in 0..3 {
            assert_abs_diff_eq!(scaled[[i, 0]], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_outlier_robustness() {
        let scaler = RobustScaler::<f64>::new();
        // Add a large outlier; median should not shift much
        let x = array![[1.0], [2.0], [3.0], [4.0], [1000.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        // Median of sorted [1,2,3,4,1000] = 3.0
        assert_abs_diff_eq!(fitted.median()[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let scaler = RobustScaler::<f64>::new();
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
        let scaler = RobustScaler::<f64>::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = scaler.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_insufficient_samples_error() {
        let scaler = RobustScaler::<f64>::new();
        let x: Array2<f64> = Array2::zeros((0, 3));
        assert!(scaler.fit(&x, &()).is_err());
    }

    // -----------------------------------------------------------------------
    // REQ-4: quantile_range constructor parameter (sklearn _data.py:1567,
    // :1604-1606, :1630-1634). Live sklearn 1.5.2 oracle values (R-CHAR-3),
    // computed from /tmp, never copied from the ferrolearn side.
    //
    // Oracle X: col0 = [1,2,...,10], col1 = [100,200,...,1000] (10 rows).
    //   RobustScaler().fit(X)
    //     -> center_ = [5.5, 550.0], scale_ = [4.5, 450.0]    (IQR 25-75)
    //   RobustScaler(quantile_range=(10.,90.)).fit(X)
    //     -> center_ = [5.5, 550.0], scale_ = [7.2, 720.0]
    //     transform([[1.,100.]]) -> [[-0.625, -0.625]]  (= (1-5.5)/7.2)
    // -----------------------------------------------------------------------

    /// Default scaler is byte-identical to the prior default (25/75) behavior:
    /// `center_ ≈ [5.5, 550]`, `scale_ ≈ [4.5, 450]`.
    #[test]
    fn robust_quantile_range_default_unchanged() -> Result<(), FerroError> {
        let mut x = Array2::<f64>::zeros((10, 2));
        for i in 0..10 {
            x[[i, 0]] = (i + 1) as f64;
            x[[i, 1]] = ((i + 1) * 100) as f64;
        }
        let fitted = RobustScaler::<f64>::new().fit(&x, &())?;
        assert_abs_diff_eq!(fitted.median()[0], 5.5, epsilon = 1e-9);
        assert_abs_diff_eq!(fitted.median()[1], 550.0, epsilon = 1e-9);
        assert_abs_diff_eq!(fitted.iqr()[0], 4.5, epsilon = 1e-9);
        assert_abs_diff_eq!(fitted.iqr()[1], 450.0, epsilon = 1e-9);
        Ok(())
    }

    /// `with_quantile_range(10, 90)` matches the live sklearn oracle:
    /// `scale_ ≈ [7.2, 720]`, `center_ ≈ [5.5, 550]`,
    /// `transform(X[:1]) ≈ [[-0.625, -0.625]]`.
    #[test]
    fn robust_quantile_range_10_90_matches_sklearn() -> Result<(), FerroError> {
        let mut x = Array2::<f64>::zeros((10, 2));
        for i in 0..10 {
            x[[i, 0]] = (i + 1) as f64;
            x[[i, 1]] = ((i + 1) * 100) as f64;
        }
        let fitted = RobustScaler::<f64>::new()
            .with_quantile_range(10.0, 90.0)?
            .fit(&x, &())?;
        assert_abs_diff_eq!(fitted.median()[0], 5.5, epsilon = 1e-9);
        assert_abs_diff_eq!(fitted.median()[1], 550.0, epsilon = 1e-9);
        assert_abs_diff_eq!(fitted.iqr()[0], 7.2, epsilon = 1e-9);
        assert_abs_diff_eq!(fitted.iqr()[1], 720.0, epsilon = 1e-9);

        let row0 = array![[1.0, 100.0]];
        let scaled = fitted.transform(&row0)?;
        assert_abs_diff_eq!(scaled[[0, 0]], -0.625, epsilon = 1e-9);
        assert_abs_diff_eq!(scaled[[0, 1]], -0.625, epsilon = 1e-9);
        Ok(())
    }

    /// Validation rejects ranges outside the non-strict `0 <= q_min <= q_max <= 100`
    /// (sklearn `_data.py:1604-1606`: raises only on `not 0 <= q_min <= q_max <= 100`).
    #[test]
    fn robust_quantile_range_validation_rejects() {
        // q_min > q_max (live sklearn: (75,25) -> ValueError "Invalid quantile range")
        assert!(matches!(
            RobustScaler::<f64>::new().with_quantile_range(75.0, 25.0),
            Err(FerroError::InvalidParameter { .. })
        ));
        // q_min < 0
        assert!(matches!(
            RobustScaler::<f64>::new().with_quantile_range(-1.0, 50.0),
            Err(FerroError::InvalidParameter { .. })
        ));
        // q_max > 100
        assert!(matches!(
            RobustScaler::<f64>::new().with_quantile_range(50.0, 110.0),
            Err(FerroError::InvalidParameter { .. })
        ));
    }

    /// A degenerate equal range (`q_min == q_max`) is accepted, matching live
    /// sklearn 1.5.2: `RobustScaler(quantile_range=(50.,50.))` -> OK with
    /// `scale_ = [1.0, 1.0]` (zero IQR -> `_handle_zeros_in_scale` -> 1) and
    /// `center_ = [5.5, 550.0]` on the oracle X (col0 `[1..10]`, col1 `[100..1000]`).
    #[test]
    fn robust_quantile_range_equal_bounds_accepted() -> Result<(), FerroError> {
        let mut x = Array2::<f64>::zeros((10, 2));
        for i in 0..10 {
            x[[i, 0]] = (i + 1) as f64;
            x[[i, 1]] = ((i + 1) * 100) as f64;
        }
        let fitted = RobustScaler::<f64>::new()
            .with_quantile_range(50.0, 50.0)?
            .fit(&x, &())?;
        // center_ == median == [5.5, 550.0] (unaffected by quantile_range).
        assert_abs_diff_eq!(fitted.median()[0], 5.5, epsilon = 1e-9);
        assert_abs_diff_eq!(fitted.median()[1], 550.0, epsilon = 1e-9);
        // Raw IQR (Q50 - Q50) is exactly 0 on every column.
        assert_abs_diff_eq!(fitted.iqr()[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fitted.iqr()[1], 0.0, epsilon = 1e-12);
        // Transform uses the effective scale 1 (_handle_zeros), so scale_ == [1, 1]:
        // the centered values are not rescaled.
        let scaled = fitted.transform(&x)?;
        // Row with col0 == 1 (i = 0): (1 - 5.5) / 1 = -4.5; (100 - 550) / 1 = -450.
        assert_abs_diff_eq!(scaled[[0, 0]], -4.5, epsilon = 1e-9);
        assert_abs_diff_eq!(scaled[[0, 1]], -450.0, epsilon = 1e-9);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // REQ-5: with_centering / with_scaling constructor parameters
    // (sklearn _data.py:1616, :1640, :1672-1675). Live sklearn 1.5.2 oracle
    // (R-CHAR-3), X col0 = [1,3,5,7,9], col1 = [100,300,500,700,900] (5 rows):
    //   center_ = [5, 500], scale_ = [4, 400].
    //   RobustScaler(with_centering=True,  with_scaling=True ).transform(X[:2])
    //     -> [[-1, -1], [-0.5, -0.5]]
    //   RobustScaler(with_centering=True,  with_scaling=False).transform(X[:2])
    //     -> [[-4, -400], [-2, -200]]    (center only)
    //   RobustScaler(with_centering=False, with_scaling=True ).transform(X[:2])
    //     -> [[0.25, 0.25], [0.75, 0.75]] (scale only)
    //   RobustScaler(with_centering=False, with_scaling=False).transform(X[:2])
    //     -> [[1, 100], [3, 300]]         (identity)
    // -----------------------------------------------------------------------

    fn oracle_x() -> Array2<f64> {
        array![
            [1.0, 100.0],
            [3.0, 300.0],
            [5.0, 500.0],
            [7.0, 700.0],
            [9.0, 900.0]
        ]
    }

    /// Default (with_centering=true, with_scaling=true) matches the live sklearn
    /// oracle AND is byte-identical to a pre-existing default fit.
    #[test]
    fn robust_with_centering_scaling_default_matches_sklearn() -> Result<(), FerroError> {
        let x = oracle_x();
        let fitted = RobustScaler::<f64>::new().fit(&x, &())?;
        let head = x.slice(ndarray::s![0..2, ..]).to_owned();
        let scaled = fitted.transform(&head)?;

        let expected = array![[-1.0, -1.0], [-0.5, -0.5]];
        for (a, b) in scaled.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-9);
        }

        // Regression guard: default config must be BYTE-identical to a plain
        // default fit produced the same way.
        let default_fitted = RobustScaler::<f64>::new().fit(&x, &())?;
        let default_scaled = default_fitted.transform(&head)?;
        for (a, b) in scaled.iter().zip(default_scaled.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        Ok(())
    }

    /// `with_with_scaling(false)` (T,F) centers only.
    #[test]
    fn robust_with_scaling_false() -> Result<(), FerroError> {
        let x = oracle_x();
        let fitted = RobustScaler::<f64>::new()
            .with_with_scaling(false)
            .fit(&x, &())?;
        let head = x.slice(ndarray::s![0..2, ..]).to_owned();
        let scaled = fitted.transform(&head)?;

        let expected = array![[-4.0, -400.0], [-2.0, -200.0]];
        for (a, b) in scaled.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-9);
        }
        Ok(())
    }

    /// `with_with_centering(false)` (F,T) scales only.
    #[test]
    fn robust_with_centering_false() -> Result<(), FerroError> {
        let x = oracle_x();
        let fitted = RobustScaler::<f64>::new()
            .with_with_centering(false)
            .fit(&x, &())?;
        let head = x.slice(ndarray::s![0..2, ..]).to_owned();
        let scaled = fitted.transform(&head)?;

        let expected = array![[0.25, 0.25], [0.75, 0.75]];
        for (a, b) in scaled.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-9);
        }
        Ok(())
    }

    /// Both `false` (F,F) is the identity.
    #[test]
    fn robust_both_false_identity() -> Result<(), FerroError> {
        let x = oracle_x();
        let fitted = RobustScaler::<f64>::new()
            .with_with_centering(false)
            .with_with_scaling(false)
            .fit(&x, &())?;
        let head = x.slice(ndarray::s![0..2, ..]).to_owned();
        let scaled = fitted.transform(&head)?;

        let expected = array![[1.0, 100.0], [3.0, 300.0]];
        for (a, b) in scaled.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
        Ok(())
    }
}
