//! Max-absolute scaler: scale each feature by its maximum absolute value.
//!
//! Each feature is transformed as `x_scaled = x / max(|x|)` so that values
//! fall within `[-1, 1]`. This scaler does not shift the data (no centering),
//! making it suitable for sparse data.
//!
//! Columns where `max_abs = 0` (all-zero features) are left unchanged.
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_data.py` (`class MaxAbsScaler`
//! `:1116`, `maxabs_scale` `:1351`). Design doc: `.design/preprocess/max_abs_scaler.md`. Expected
//! values from the live sklearn 1.5.2 oracle (R-CHAR-3). Consumers: PyO3 `_RsMaxAbsScaler`
//! (`ferrolearn-python/src/extras.rs:1156`) + `PipelineTransformer` impl + crate re-export (S5).
//! HONEST (R-HONEST-3): verify-and-document — the dense max-abs path matches sklearn including
//! the zero-max_abs edge (which, unlike Min/StandardScaler, does NOT diverge: a zero-max_abs
//! column is all-zero, and `x/scale_(1) = x` equals ferrolearn's leave-unchanged).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (per-column max-abs value match) | SHIPPED | `Fit::fit` learns per-column `max_abs=max(|x|)`; `Transform::transform` = `x/max_abs`, mirroring sklearn `max_abs_=_nanmax(abs(X))` (`_data.py:1263`) / `scale_=_handle_zeros_in_scale(max_abs_)` (`:1272`) / `X/=scale_` (`:1305`). Critic-verified bit-identical to live oracle: `divergence_max_abs_scaler.rs` green guards (`[[-3,1],[0,-2],[2,4]]` → `[[-1,0.25],[0,-0.5],[0.6667,1]]`, all-negative, mixed, f32). Consumers: PyO3 `_RsMaxAbsScaler` + `FittedPipelineTransformer` + re-export. |
//! | REQ-2 (zero-max_abs column → identity, MATCHES sklearn) | SHIPPED | A zero-max_abs column is all-zero; sklearn `scale_=_handle_zeros_in_scale(0)=1` → `x/1=x` = ferrolearn's leave-unchanged for ANY input. Critic-verified MATCH (discriminating: `fit([[0],[0]]).transform([[5]])==[[5.0]]` in both). NOT a divergence (contrast Min/StandardScaler). |
//! | REQ-3 (inverse_transform round-trip) | SHIPPED | `inverse_transform` = `x*max_abs` (zero-max_abs left unchanged), mirroring sklearn `X *= scale_` (`_data.py:1337`); `inverse_transform(transform(X))==X` (green guard). Consumer: re-export boundary (S5). |
//! | REQ-4 (PyO3 binding) | SHIPPED | `_RsMaxAbsScaler` (`extras.rs:1156`, registered `lib.rs:82`) marshals `fit`/`transform` over `FittedMaxAbsScaler<f64>` — a real CPython consumer; maturin smoke. |
//! | REQ-5 (NaN tolerance: allow-nan) | SHIPPED | FIXED #1202. `Fit::fit`'s per-column `max(\|x\|)` reduction now SKIPS NaN via an `Option<F>` accumulator (`continue` on `is_nan()`), mirroring sklearn `max_abs = _nanmax(abs(X), axis=0)` under `force_all_finite='allow-nan'` (`_data.py:1263`,`:1256`): a column with at least one finite value gets the finite max-abs; an ALL-NaN column gets `max_abs = F::nan()` (NaN passes `_handle_zeros_in_scale` since NaN != 0 → `scale_` NaN → transform/inverse NaN; no panic, no zero-substitution). `transform`/`inverse_transform` now divide/multiply by `scale_` (so a NaN-column maps to NaN, a zero-column to identity); NaN inputs pass through (`nan/scale = nan`, `nan*scale = nan`). inf-rejection (allow-nan REJECTS ±inf, MinMaxScaler #2200 precedent): `fit`/`transform`/`inverse_transform` return `InvalidParameter` ("Input X contains infinity...") on any `is_infinite()` element. Live-oracle tests `req5_nan_fit_single_column_ignored`, `req5_nan_fit_multi_column_scattered`, `req5_all_nan_column_yields_nan_no_panic`, `req5_nan_passthrough_inverse_transform`, `inf_rejected_fit`/`inf_rejected_transform`/`inf_rejected_inverse_transform`, `nan_only_still_fits` in `tests/divergence_max_abs_scaler.rs`. Consumers: PyO3 `_RsMaxAbsScaler` + `FittedPipelineTransformer` + re-export. |
//! | REQ-6 (scale_/n_samples_seen_ attrs) | SHIPPED | `FittedMaxAbsScaler<F>` stores `scale_ = max_abs.mapv(\|m\| if m==0 {1} else {m})` (mirroring sklearn `scale_ = _handle_zeros_in_scale(max_abs_)` `_data.py:1272`,`:88` — `1.0` on all-zero columns) and `n_samples_seen_ = n_samples` (`:1266`), set in `Fit::fit`. Getters `scale()`/`n_samples_seen()` (`#[must_use]`). Oracle (`MaxAbsScaler().fit([[1,0],[-3,0],[2,0]])` → `max_abs_=[3,0]`, `scale_=[3,1]`, `n_samples_seen_=3`): tests `max_abs_scale_nsamples_match_sklearn`, `max_abs_scale_differs_from_max_abs_on_zero_col`. `transform`/`inverse_transform` unchanged (still divide/multiply by `max_abs`; identical to dividing by `scale_` since they coincide off the all-zero columns). |
//! | REQ-7 (partial_fit / streaming) | NOT-STARTED | open prereq blocker #1204. Single-shot (`_data.py:1232-1273`). |
//! | REQ-8 (maxabs_scale free fn + axis) | NOT-STARTED | open prereq blocker #1205. No `maxabs_scale` / axis=1 (`_data.py:1351`). |
//! | REQ-9 (copy param + _parameter_constraints) | SHIPPED | FIXED #1206. `MaxAbsScaler<F>` gains a `copy: bool` field (default `true`) + `#[must_use] with_copy(self, bool) -> Self` builder + `copy()` getter, mirroring sklearn `__init__(*, copy=True)` (`_data.py:1190`) under `_parameter_constraints {copy:["boolean"]}` (`:1188`). ACCEPT-AND-DOCUMENT no-op: ferrolearn's [`Transform`] always returns a freshly allocated array (`to_owned()`), so `copy` has no observable effect (documented; behavior unchanged) — the direct analog of MinMaxScaler REQ-10's `copy` no-op. Live-oracle test `req9_copy_is_no_op_on_values`. Consumers: PyO3 `_RsMaxAbsScaler` + re-export. |
//! | REQ-10 (sparse CSR/CSC) | NOT-STARTED | open prereq blocker #1207. Dense-only; MaxAbsScaler is sklearn's flagship sparse-safe scaler (`_data.py:1260-1261`,`:1303`). |
//! | REQ-11 (get_feature_names_out / n_features_in_) | NOT-STARTED | open prereq blocker #1208. None (OneToOneFeatureMixin). |
//! | REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1209. `ndarray`+`num_traits`, not `ferray-core` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// MaxAbsScaler (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted max-absolute scaler.
///
/// Calling [`Fit::fit`] learns the per-column maximum absolute values and
/// returns a [`FittedMaxAbsScaler`] that can transform new data.
///
/// Columns where the maximum absolute value is zero are left unchanged after
/// transformation.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::MaxAbsScaler;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let scaler = MaxAbsScaler::<f64>::new();
/// let x = array![[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]];
/// let fitted = scaler.fit(&x, &()).unwrap();
/// let scaled = fitted.transform(&x).unwrap();
/// // All values now in [-1, 1]
/// ```
#[derive(Debug, Clone)]
pub struct MaxAbsScaler<F> {
    /// sklearn's `copy` constructor parameter (`_data.py:1190`,
    /// `_parameter_constraints {copy:["boolean"]}` `:1188`). ACCEPT-AND-DOCUMENT
    /// no-op: ferrolearn's [`Transform`] always returns a freshly allocated
    /// array, so `copy` has no observable effect here. Retained for API parity.
    /// Defaults to `true`.
    pub(crate) copy: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> MaxAbsScaler<F> {
    /// Create a new `MaxAbsScaler`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            copy: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set sklearn's `copy` constructor parameter (`_data.py:1190`).
    ///
    /// ACCEPT-AND-DOCUMENT no-op: ferrolearn's [`Transform`] contract always
    /// returns a freshly allocated array, so `copy` has no observable effect.
    /// The flag is retained only for API parity with scikit-learn
    /// (`_parameter_constraints {copy:["boolean"]}`, `_data.py:1188`); toggling
    /// it does not change behavior.
    #[must_use]
    pub fn with_copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }

    /// Return the `copy` flag (accept-and-document no-op; see [`Self::with_copy`]).
    #[must_use]
    pub fn copy(&self) -> bool {
        self.copy
    }
}

impl<F: Float + Send + Sync + 'static> Default for MaxAbsScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedMaxAbsScaler
// ---------------------------------------------------------------------------

/// A fitted max-absolute scaler holding per-column maximum absolute values.
///
/// Created by calling [`Fit::fit`] on a [`MaxAbsScaler`].
#[derive(Debug, Clone)]
pub struct FittedMaxAbsScaler<F> {
    /// Per-column maximum absolute values learned during fitting.
    pub(crate) max_abs: Array1<F>,
    /// Per-column scaling factors, `max_abs` with all-zero columns replaced by `1.0`.
    pub(crate) scale_: Array1<F>,
    /// Number of samples (rows) seen during fitting.
    pub(crate) n_samples_seen_: usize,
}

impl<F: Float + Send + Sync + 'static> FittedMaxAbsScaler<F> {
    /// Return the per-column maximum absolute values learned during fitting.
    #[must_use]
    pub fn max_abs(&self) -> &Array1<F> {
        &self.max_abs
    }

    /// Return the per-column scaling factors used to divide each feature.
    ///
    /// Mirrors sklearn `MaxAbsScaler.scale_ = _handle_zeros_in_scale(max_abs_)`
    /// (`sklearn/preprocessing/_data.py:1272`): equal to `max_abs` on nonzero
    /// columns and exactly `1.0` on all-zero columns, so dividing by `scale_`
    /// leaves an all-zero column unchanged.
    #[must_use]
    pub fn scale(&self) -> &Array1<F> {
        &self.scale_
    }

    /// Return the number of samples (rows) seen during fitting.
    ///
    /// Mirrors sklearn `MaxAbsScaler.n_samples_seen_`
    /// (`sklearn/preprocessing/_data.py:1266`).
    #[must_use]
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen_
    }

    /// Inverse-transform scaled data back to the original space.
    ///
    /// Applies `x_orig = x_scaled * scale_` per column, mirroring sklearn
    /// `X *= self.scale_` (`_data.py:1337`) with
    /// `scale_ = _handle_zeros_in_scale(max_abs_)` (`:1272`,`:88`): a
    /// zero-`max_abs` (all-zero) column has `scale_ = 1` (`x * 1 = x`), an
    /// all-NaN fitted column has `scale_ = NaN` (`x * NaN = NaN`). NaN inputs
    /// pass through (`nan * scale = nan`), matching sklearn's `allow-nan`
    /// contract (`_data.py:1331`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting, or
    /// [`FerroError::InvalidParameter`] if any input element is +/-inf
    /// (sklearn `force_all_finite="allow-nan"` rejects infinity).
    pub fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.max_abs.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedMaxAbsScaler::inverse_transform".into(),
            });
        }
        // sklearn `inverse_transform` validates with
        // `force_all_finite="allow-nan"` (`_data.py:1331`): NaN passes through,
        // +/-inf raises ValueError.
        if x.iter().any(|v| v.is_infinite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains infinity or a value too large for dtype.".into(),
            });
        }
        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            // Multiply by `scale_` (= `_handle_zeros_in_scale`): zero-`max_abs`
            // column has `scale_ = 1` (`x * 1 = x`), all-NaN column has
            // `scale_ = NaN` (`x * NaN = NaN`), otherwise `scale_ = max_abs`.
            let scale = self.scale_[j];
            for v in &mut col {
                *v = *v * scale;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for MaxAbsScaler<F> {
    type Fitted = FittedMaxAbsScaler<F>;
    type Error = FerroError;

    /// Fit the scaler by computing per-column maximum absolute values.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedMaxAbsScaler<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MaxAbsScaler::fit".into(),
            });
        }
        // sklearn validates X with `force_all_finite="allow-nan"`
        // (`_data.py:1256`): NaN is permitted, but +/-inf raises ValueError
        // ("Input X contains infinity or a value too large for dtype('...')").
        // Mirrors the MinMaxScaler #2200 precedent (allow-nan rejects inf).
        if x.iter().any(|v| v.is_infinite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains infinity or a value too large for dtype.".into(),
            });
        }

        let n_features = x.ncols();
        let mut max_abs = Array1::zeros(n_features);

        for j in 0..n_features {
            // NaN-ignoring per-column max(|x|), mirroring sklearn's
            // `max_abs = _nanmax(abs(X), axis=0)` under
            // `force_all_finite="allow-nan"` (`_data.py:1263`,`:1256`). NaN values
            // are skipped (Option accumulator). If a column is ALL NaN (every
            // entry skipped) the accumulator stays `None` and we emit NaN —
            // matching `_nanmax` returning nan on an all-NaN slice (that column's
            // scale_/transform become NaN via `_handle_zeros_in_scale`, which
            // leaves NaN unchanged since NaN != 0; no panic, no zero substitution).
            let mut acc: Option<F> = None;
            for v in x.column(j).iter().copied() {
                if v.is_nan() {
                    continue;
                }
                let a = v.abs();
                acc = Some(match acc {
                    Some(m) if m >= a => m,
                    _ => a,
                });
            }
            max_abs[j] = acc.unwrap_or_else(F::nan);
        }

        // sklearn: scale_ = _handle_zeros_in_scale(max_abs_) (`_data.py:1272`,
        // `:114-119`): `constant_mask = scale < 10 * finfo(dtype).eps;
        // scale[constant_mask] = 1.0`. A max_abs BELOW the near-constant
        // threshold (NOT just exactly 0) becomes 1.0 so dividing leaves the
        // column unchanged (#2203). A NaN max_abs (all-NaN column) is NOT
        // `< threshold` (NaN compares false), so it passes through -> scale_ =
        // NaN -> transform/inverse yield NaN (matching sklearn).
        let ten_eps = F::epsilon() * F::from(10.0).unwrap_or_else(F::one);
        let scale_ = max_abs.mapv(|m| if m < ten_eps { F::one() } else { m });
        // sklearn: n_samples_seen_ = X.shape[0] (`_data.py:1266`).
        let n_samples_seen_ = n_samples;

        Ok(FittedMaxAbsScaler {
            max_abs,
            scale_,
            n_samples_seen_,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedMaxAbsScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by dividing each feature by its maximum absolute value.
    ///
    /// Mirrors sklearn `X /= self.scale_` (`_data.py:1305`) with
    /// `scale_ = _handle_zeros_in_scale(max_abs_)` (`:1272`,`:88`): a
    /// zero-`max_abs` (all-zero) column has `scale_ = 1`, so `x / 1 = x` leaves
    /// it unchanged; an all-NaN fitted column has `scale_ = NaN`, so `x / NaN =
    /// NaN`. NaN inputs pass through (`nan / scale = nan`), matching sklearn's
    /// `allow-nan` contract (`_data.py:1256`,`:1299`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting, or
    /// [`FerroError::InvalidParameter`] if any input element is +/-inf
    /// (sklearn `force_all_finite="allow-nan"` rejects infinity).
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.max_abs.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedMaxAbsScaler::transform".into(),
            });
        }
        // sklearn `transform` validates with `force_all_finite="allow-nan"`
        // (`_data.py:1299`): NaN passes through, +/-inf raises ValueError.
        if x.iter().any(|v| v.is_infinite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains infinity or a value too large for dtype.".into(),
            });
        }

        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            // Divide by the precomputed `scale_` (= `_handle_zeros_in_scale`):
            // a zero-`max_abs` column has `scale_ = 1` (`x / 1 = x`), an all-NaN
            // column has `scale_ = NaN` (`x / NaN = NaN`), otherwise `scale_ =
            // max_abs`. NaN inputs pass through (`nan / scale = nan`).
            let scale = self.scale_[j];
            for v in &mut col {
                *v = *v / scale;
            }
        }
        Ok(out)
    }
}

/// Implement `Transform` on the unfitted scaler to satisfy the `FitTransform: Transform`
/// supertrait bound. Calling `transform` on an unfitted scaler always returns an error.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for MaxAbsScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the scaler must be fitted first.
    ///
    /// Use [`Fit::fit`] to produce a [`FittedMaxAbsScaler`], then call
    /// [`Transform::transform`] on that.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "MaxAbsScaler".into(),
            reason: "scaler must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for MaxAbsScaler<F> {
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

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for MaxAbsScaler<F> {
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedMaxAbsScaler<F> {
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
    fn test_max_abs_scaler_basic() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        // col0: max_abs = 3.0, col1: max_abs = 4.0
        assert_abs_diff_eq!(fitted.max_abs()[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.max_abs()[1], 4.0, epsilon = 1e-10);

        let scaled = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(scaled[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[2, 0]], 2.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[2, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_values_in_range() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[-10.0, 5.0], [3.0, -8.0], [7.0, 2.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();
        for v in &scaled {
            assert!(
                *v >= -1.0 - 1e-10 && *v <= 1.0 + 1e-10,
                "value {v} out of [-1, 1]"
            );
        }
    }

    #[test]
    fn test_zero_column_unchanged() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.max_abs()[0], 0.0, epsilon = 1e-15);
        let scaled = fitted.transform(&x).unwrap();
        // All-zero column stays 0.0
        for i in 0..3 {
            assert_abs_diff_eq!(scaled[[i, 0]], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&scaled).unwrap();
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let scaler = MaxAbsScaler::<f64>::new();
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
        let scaler = MaxAbsScaler::<f64>::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = scaler.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_insufficient_samples_error() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x: Array2<f64> = Array2::zeros((0, 3));
        assert!(scaler.fit(&x, &()).is_err());
    }

    #[test]
    fn test_unfitted_transform_error() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[1.0, 2.0]];
        assert!(scaler.transform(&x).is_err());
    }

    #[test]
    fn test_negative_values() {
        let scaler = MaxAbsScaler::<f64>::new();
        // All negative values
        let x = array![[-5.0], [-3.0], [-1.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.max_abs()[0], 5.0, epsilon = 1e-10);
        let scaled = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(scaled[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[1, 0]], -0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[2, 0]], -0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[2.0, 4.0], [1.0, -2.0]];
        let y = Array1::zeros(2);
        let fitted = scaler.fit_pipeline(&x, &y).unwrap();
        let result = fitted.transform_pipeline(&x).unwrap();
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], -0.5, epsilon = 1e-10);
    }

    #[test]
    fn max_abs_scale_nsamples_match_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   MaxAbsScaler().fit([[1,0],[-3,0],[2,0]])
        //   -> max_abs_ = [3.0, 0.0], scale_ = [3.0, 1.0], n_samples_seen_ = 3
        // column 1 is all-zero: scale_ = _handle_zeros_in_scale(0) = 1 (_data.py:1272,:88).
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[1.0, 0.0], [-3.0, 0.0], [2.0, 0.0]];
        let fitted = scaler.fit(&x, &())?;

        assert_abs_diff_eq!(fitted.scale()[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fitted.scale()[1], 1.0, epsilon = 1e-12);
        // Exactly 1.0 on the all-zero column (not merely close).
        assert!(fitted.scale()[1] == 1.0);
        assert_eq!(fitted.n_samples_seen(), 3);
        Ok(())
    }

    #[test]
    fn max_abs_scale_differs_from_max_abs_on_zero_col() -> Result<(), FerroError> {
        // scale_ differs from max_abs_ exactly on all-zero columns: sklearn
        // max_abs_ = [3.0, 0.0] but scale_ = [3.0, 1.0] (_data.py:1272,:88).
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[1.0, 0.0], [-3.0, 0.0], [2.0, 0.0]];
        let fitted = scaler.fit(&x, &())?;

        // All-zero column: max_abs_ == 0.0 but scale_ == 1.0.
        assert!(fitted.max_abs()[1] == 0.0);
        assert!(fitted.scale()[1] == 1.0);
        // Nonzero column: scale_ unchanged from max_abs_.
        assert!(fitted.scale()[0] == fitted.max_abs()[0]);
        Ok(())
    }

    #[test]
    fn test_f32_scaler() {
        let scaler = MaxAbsScaler::<f32>::new();
        let x: Array2<f32> = array![[2.0f32, -4.0], [1.0, 3.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();
        assert!((scaled[[0, 0]] - 1.0f32).abs() < 1e-6);
        assert!((scaled[[0, 1]] - (-1.0f32)).abs() < 1e-6);
    }
}
