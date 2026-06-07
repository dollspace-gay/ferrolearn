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
//! | REQ-4 (NaN tolerance: allow-nan + nanmin/nanmax) | SHIPPED | FIXED #1171. `Fit::fit`'s per-column min/max reduction now SKIPS NaN values (option-accumulator), mirroring sklearn `force_all_finite='allow-nan'` + `_nanmin`/`_nanmax` (`_data.py:494`,`:497-498`): a column with at least one finite value gets the finite min/max; an ALL-NaN column gets `data_min`/`data_max` = `F::nan()` (→ `scale_`/`min_`/transform NaN, matching `_nanmin`/`_nanmax` returning nan, no panic, no zero substitution). `Transform::transform`'s affine map passes NaN inputs through unchanged (`nan*scale+min = nan`). Live-oracle tests `req4_nan_fit_single_column_ignored_for_min_max`, `req4_nan_fit_multi_column_scattered`, `req4_all_nan_column_yields_nan_no_panic` in `tests/divergence_min_max_scaler.rs`. Consumers: PyO3 `_RsMinMaxScaler` + `FittedPipelineTransformer` + re-export `lib.rs:118`. |
//! | REQ-5 (scale_/min_/data_range_/n_samples_seen_) | SHIPPED | FIXED #1172. `Fit::fit` computes `data_range_[j]=data_max[j]-data_min[j]`, `scale_[j]=(fr1-fr0)/handle_zeros(data_range_[j])`, `min_[j]=fr0-data_min[j]*scale_[j]`, `n_samples_seen_=n_rows`, mirroring sklearn (`_data.py:507-514`, `_handle_zeros_in_scale` `:88`). Additive: getters `scale()`/`min()`/`data_range()`/`n_samples_seen()`; `transform`/`inverse_transform` unchanged. Live-oracle tests `min_max_attrs_match_sklearn`, `min_max_scale_handles_zero_range_constant_col`. |
//! | REQ-6 (inverse_transform) | SHIPPED | FIXED #1173. `FittedMinMaxScaler::inverse_transform` reverses the affine map: per column `j`, `x_orig = (x_scaled - fr0) * span / range_width + data_min[j]` (`span = data_max[j]-data_min[j]`, `range_width = fr1-fr0`), matching sklearn `X -= self.min_; X /= self.scale_` (`_data.py:549-587`,`:508-511`,`:88`). The single formula round-trips both regular and constant columns (`span==0 -> data_min[j]`). `ShapeMismatch` on column-count mismatch (mirrors `transform`). Oracle-grounded in-module tests: `min_max_inverse_roundtrip_matches_sklearn`, `min_max_inverse_custom_range_matches_sklearn`, `min_max_inverse_constant_col`, `min_max_inverse_shape_mismatch`. Consumer: crate re-export (`lib.rs:118`, grandfathered S5). |
//! | REQ-8 (partial_fit / streaming) | NOT-STARTED | open prereq blocker #1174. Single-shot fit (`_data.py:489-515`). |
//! | REQ-9 (minmax_scale free fn + axis) | NOT-STARTED | open prereq blocker #1175. No free fn / axis=1 (`_data.py:589`). |
//! | REQ-10 (copy / clip params) | SHIPPED | FIXED #1176. `MinMaxScaler<F>` gains `clip: bool` (default `false`) + `#[must_use] with_clip` builder + `clip()` getter, threaded onto `FittedMinMaxScaler`. `Transform::transform` applies a NaN-safe element-wise clamp to `[feature_range.0, feature_range.1]` AFTER the affine map when `clip` is set (`if x < lo {lo} else if x > hi {hi} else {x}` leaves NaN unchanged), mirroring sklearn `if self.clip: np.clip(X, fr[0], fr[1])` (`_data.py:411`,`:545-546`). `copy: bool` (default `true`) + `with_copy`/`copy()` is an ACCEPT-AND-DOCUMENT no-op: ferrolearn's `Transform` always returns a fresh array, so `copy` has no observable effect (documented; behavior unchanged). Live-oracle tests `req10_clip_default_range_out_of_range_holdout`, `req10_clip_custom_range_out_of_range_holdout`, `req10_clip_with_nan_passthrough`, `req10_copy_is_no_op_on_values`. Consumers: PyO3 `_RsMinMaxScaler` + `FittedPipelineTransformer` + re-export `lib.rs:118`. (`_parameter_constraints` validation surface stays as the existing `with_feature_range` guard, REQ-3.) |
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
    /// If `true`, clip transformed values to `feature_range` element-wise
    /// (sklearn `MinMaxScaler(clip=...)`, `_data.py:411`,`:545-546`). Defaults
    /// to `false`.
    pub(crate) clip: bool,
    /// sklearn's `copy` constructor parameter (`_data.py:411`). ACCEPT-AND-
    /// DOCUMENT no-op: ferrolearn's [`Transform`] always returns a freshly
    /// allocated array, so `copy` has no observable effect here. Retained for
    /// API parity. Defaults to `true`.
    pub(crate) copy: bool,
}

impl<F: Float + Send + Sync + 'static> MinMaxScaler<F> {
    /// Create a new `MinMaxScaler` scaling to the default range `[0, 1]`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            feature_range: (F::zero(), F::one()),
            clip: false,
            copy: true,
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
            clip: false,
            copy: true,
        })
    }

    /// Return the configured feature range.
    #[must_use]
    pub fn feature_range(&self) -> (F, F) {
        self.feature_range
    }

    /// Set whether to clip transformed values to `feature_range` element-wise.
    ///
    /// Mirrors sklearn's `MinMaxScaler(clip=...)` constructor parameter
    /// (`_data.py:411`, `_parameter_constraints` `:408`). When `true`,
    /// [`Transform::transform`] clamps each output element to
    /// `[feature_range.0, feature_range.1]` after the affine map. This matters
    /// for held-out data that falls outside the fitted min/max range, which
    /// would otherwise map outside `feature_range`. NaN inputs are left as NaN
    /// (matching `np.clip(nan, lo, hi) == nan`, `_data.py:545-546`).
    #[must_use]
    pub fn with_clip(mut self, clip: bool) -> Self {
        self.clip = clip;
        self
    }

    /// Set sklearn's `copy` constructor parameter (`_data.py:411`).
    ///
    /// ACCEPT-AND-DOCUMENT no-op: ferrolearn's [`Transform`] contract always
    /// returns a freshly allocated array, so `copy` has no observable effect.
    /// The flag is retained only for API parity with scikit-learn; toggling it
    /// does not change behavior.
    #[must_use]
    pub fn with_copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }

    /// Return whether transformed values are clipped to `feature_range`.
    #[must_use]
    pub fn clip(&self) -> bool {
        self.clip
    }

    /// Return the `copy` flag (accept-and-document no-op; see [`Self::with_copy`]).
    #[must_use]
    pub fn copy(&self) -> bool {
        self.copy
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
    /// Per-column affine scale `scale_[j] = (fr1 - fr0) / handle_zeros(data_range_[j])`
    /// (`sklearn/preprocessing/_data.py:508-510`, `:88`).
    pub(crate) scale_: Array1<F>,
    /// Per-column affine offset `min_[j] = fr0 - data_min[j] * scale_[j]`
    /// (`sklearn/preprocessing/_data.py:511`).
    pub(crate) min_: Array1<F>,
    /// Per-column raw data range `data_range_[j] = data_max[j] - data_min[j]`
    /// (`sklearn/preprocessing/_data.py:507`,`:514`).
    pub(crate) data_range_: Array1<F>,
    /// Number of samples seen during fitting (`sklearn/preprocessing/_data.py:501`).
    pub(crate) n_samples_seen_: usize,
    /// Whether [`Transform::transform`] clips outputs to `feature_range`
    /// element-wise (sklearn `clip`, `_data.py:411`,`:545-546`). Threaded from
    /// the unfitted [`MinMaxScaler`].
    pub(crate) clip: bool,
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

    /// Return the per-column affine scale learned during fitting.
    ///
    /// `scale_[j] = (feature_range.1 - feature_range.0) / handle_zeros(data_range_[j])`,
    /// where `handle_zeros(d) = if d == 0 { 1 } else { d }`
    /// (`sklearn/preprocessing/_data.py:508-510`, `_handle_zeros_in_scale` `:88`).
    #[must_use]
    pub fn scale(&self) -> &Array1<F> {
        &self.scale_
    }

    /// Return the per-column affine offset learned during fitting.
    ///
    /// `min_[j] = feature_range.0 - data_min[j] * scale_[j]`
    /// (`sklearn/preprocessing/_data.py:511`).
    #[must_use]
    pub fn min(&self) -> &Array1<F> {
        &self.min_
    }

    /// Return the per-column raw data range learned during fitting.
    ///
    /// `data_range_[j] = data_max[j] - data_min[j]` (`0` on a constant column;
    /// `sklearn/preprocessing/_data.py:507`,`:514`).
    #[must_use]
    pub fn data_range(&self) -> &Array1<F> {
        &self.data_range_
    }

    /// Return the number of samples seen during fitting
    /// (`sklearn/preprocessing/_data.py:501`).
    #[must_use]
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen_
    }

    /// Undo the scaling of `x` according to `feature_range`, mapping scaled
    /// data back to the original feature space.
    ///
    /// This reverses the affine map applied by [`Transform::transform`]. For
    /// each column `j` with `span = data_max[j] - data_min[j]` and
    /// `range_width = feature_range.1 - feature_range.0`:
    /// `x_orig = (x_scaled - feature_range.0) * span / range_width + data_min[j]`.
    ///
    /// This single formula is correct for both regular and constant columns:
    /// for a constant column (`span == 0`, whose forward image is
    /// `feature_range.0`) it yields `data_min[j]`. It mirrors sklearn's
    /// `X -= self.min_; X /= self.scale_` where `scale_` divides by
    /// `_handle_zeros_in_scale(data_range)`
    /// (`sklearn/preprocessing/_data.py:549-587`,`:508-511`,`:88`).
    ///
    /// `range_width` is guaranteed nonzero because
    /// [`MinMaxScaler::with_feature_range`] rejects `range_min >= range_max`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    pub fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.data_min.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedMinMaxScaler::inverse_transform".into(),
            });
        }

        // sklearn `inverse_transform` validates with `force_all_finite="allow-nan"`
        // (`_data.py:571`): NaN passes through, +/-inf raises ValueError (#2202).
        if x.iter().any(|v| v.is_infinite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains infinity or a value too large for dtype.".into(),
            });
        }

        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            // sklearn inverse is `X -= self.min_; X /= self.scale_`
            // (`_data.py:586-587`). Using the precomputed `scale_`/`min_` (via
            // `_handle_zeros_in_scale`) inverts the constant-column case
            // correctly: a fit-constant column has `scale_ = range_width`, so a
            // HELD-OUT scaled value gets the affine inverse (it is NOT forced to
            // `data_min`, #2202). NaN passes through (`(nan - min)/scale = nan`).
            let scale = self.scale_[j];
            let offset = self.min_[j];
            for v in &mut col {
                *v = (*v - offset) / scale;
            }
        }
        Ok(out)
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
        // sklearn validates X with `force_all_finite="allow-nan"`
        // (`_data.py:494`): NaN is permitted, but +/-inf raises ValueError
        // ("Input X contains infinity or a value too large for dtype('...')"),
        // #2200.
        if x.iter().any(|v| v.is_infinite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains infinity or a value too large for dtype.".into(),
            });
        }

        let n_features = x.ncols();
        let mut data_min = Array1::zeros(n_features);
        let mut data_max = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            // NaN-ignoring per-column min/max, mirroring sklearn's `_nanmin` /
            // `_nanmax` under `force_all_finite="allow-nan"`
            // (`sklearn/preprocessing/_data.py:494`,`:497-498`). NaN values are
            // skipped: data_min/data_max are the min/max of the finite values.
            // If a column is ALL NaN (every entry skipped) the accumulator stays
            // `None` and we emit NaN — matching `_nanmin`/`_nanmax` returning nan
            // on an all-NaN slice (that column's scale_/min_/transform become
            // NaN; no panic, no zero substitution).
            let mut min: Option<F> = None;
            let mut max: Option<F> = None;
            for v in col.iter().copied() {
                if v.is_nan() {
                    continue;
                }
                min = Some(match min {
                    Some(m) if m < v => m,
                    _ => v,
                });
                max = Some(match max {
                    Some(m) if m > v => m,
                    _ => v,
                });
            }
            data_min[j] = min.unwrap_or_else(F::nan);
            data_max[j] = max.unwrap_or_else(F::nan);
        }

        // Derived affine attributes mirroring sklearn (_data.py:507-514, :88).
        let (fr0, fr1) = self.feature_range;
        let range_width = fr1 - fr0;
        let data_range_ = &data_max - &data_min;
        let mut scale_ = Array1::zeros(n_features);
        let mut min_ = Array1::zeros(n_features);
        for j in 0..n_features {
            // _handle_zeros_in_scale (_data.py:88): a zero (constant-column)
            // range is replaced by 1 to avoid division by zero.
            let denom = if data_range_[j] == F::zero() {
                F::one()
            } else {
                data_range_[j]
            };
            scale_[j] = range_width / denom;
            min_[j] = fr0 - data_min[j] * scale_[j];
        }

        Ok(FittedMinMaxScaler {
            data_min,
            data_max,
            feature_range: self.feature_range,
            scale_,
            min_,
            data_range_,
            n_samples_seen_: n_samples,
            clip: self.clip,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedMinMaxScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by scaling each feature to the configured range.
    ///
    /// Constant columns where `data_max == data_min` are mapped to
    /// `feature_range[0]` (matching scikit-learn). NaN inputs pass through as
    /// NaN (`nan * scale + min = nan`), and an all-NaN fitted column (whose
    /// `data_min`/`data_max` are NaN) transforms to NaN, matching sklearn's
    /// `allow-nan` contract (`_data.py:494`,`:497-498`).
    ///
    /// When `clip` is set, each output element is clamped to
    /// `[feature_range.0, feature_range.1]` after the affine map (sklearn
    /// `if self.clip: np.clip(X, fr[0], fr[1])`, `_data.py:545-546`). The clamp
    /// leaves NaN unchanged (both `< lo` and `> hi` comparisons are `false` for
    /// NaN, mirroring `np.clip(nan, lo, hi) == nan`).
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
        // sklearn `transform` validates with `force_all_finite="allow-nan"`
        // (`_data.py:539`): NaN passes through, +/-inf raises ValueError (#2200).
        if x.iter().any(|v| v.is_infinite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains infinity or a value too large for dtype.".into(),
            });
        }

        let (range_min, range_max) = self.feature_range;

        // NaN-safe element-wise clamp to [range_min, range_max]. For a NaN `v`
        // both comparisons are `false`, so NaN is returned unchanged — matching
        // `np.clip(nan, lo, hi) == nan` (`_data.py:545-546`).
        let clamp = |v: F| -> F {
            if v < range_min {
                range_min
            } else if v > range_max {
                range_max
            } else {
                v
            }
        };

        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            // sklearn transform is `X *= self.scale_; X += self.min_`
            // (`_data.py:543-544`). The precomputed `scale_`/`min_` (via
            // `_handle_zeros_in_scale`) already encode the constant-column case:
            // a fit-constant column has `scale_ = range_width`,
            // `min_ = fr0 - data_min*scale_`, so the FITTED value maps to
            // `feature_range[0]` while HELD-OUT data gets the affine map (it is
            // NOT forced to `feature_range[0]`, #2201). NaN inputs pass through
            // (`nan*scale + min = nan`).
            let scale = self.scale_[j];
            let offset = self.min_[j];
            for v in &mut col {
                let scaled = *v * scale + offset;
                *v = if self.clip { clamp(scaled) } else { scaled };
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

    // -----------------------------------------------------------------------
    // REQ-6: inverse_transform (sklearn _data.py:549-587). Live sklearn 1.5.2
    // oracle values (run from /tmp) — never copied from the ferrolearn side
    // (R-CHAR-3).
    // -----------------------------------------------------------------------

    /// inverse_transform(transform(X)) round-trips to X (default range).
    /// Live oracle: `MinMaxScaler().fit_transform([[1.,10.],[2.,20.],[3.,30.],[5.,50.]])`
    ///   -> `[[0.,0.],[0.25,0.25],[0.5,0.5],[1.,1.]]`; inverse_transform of that
    ///   returns the original X.
    #[test]
    fn min_max_inverse_roundtrip_matches_sklearn() -> Result<(), FerroError> {
        let scaler = MinMaxScaler::<f64>::new();
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [5.0, 50.0]];
        let fitted = scaler.fit(&x, &())?;
        let scaled = fitted.transform(&x)?;

        // Live sklearn 1.5.2 oracle for the forward transform.
        let sk_scaled = [[0.0_f64, 0.0], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0]];
        for i in 0..4 {
            for j in 0..2 {
                assert_abs_diff_eq!(scaled[[i, j]], sk_scaled[i][j], epsilon = 1e-9);
            }
        }

        let recovered = fitted.inverse_transform(&scaled)?;
        for i in 0..4 {
            for j in 0..2 {
                assert_abs_diff_eq!(recovered[[i, j]], x[[i, j]], epsilon = 1e-9);
            }
        }
        Ok(())
    }

    /// inverse_transform with custom range `(-1, 1)`.
    /// Live oracle: fit `MinMaxScaler(feature_range=(-1,1))` on
    ///   `[[1.,10.],[2.,20.],[3.,30.],[5.,50.]]`, then
    ///   `inverse_transform([[0.,0.],[1.,1.]])` -> `[[3.,30.],[5.,50.]]`.
    #[test]
    fn min_max_inverse_custom_range_matches_sklearn() -> Result<(), FerroError> {
        let scaler = MinMaxScaler::<f64>::with_feature_range(-1.0, 1.0)?;
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [5.0, 50.0]];
        let fitted = scaler.fit(&x, &())?;

        let scaled = array![[0.0, 0.0], [1.0, 1.0]];
        // Live sklearn 1.5.2 oracle.
        let sk = [[3.0_f64, 30.0], [5.0, 50.0]];

        let recovered = fitted.inverse_transform(&scaled)?;
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(recovered[[i, j]], sk[i][j], epsilon = 1e-9);
            }
        }
        Ok(())
    }

    /// inverse_transform of a constant column recovers `data_min` for every row.
    /// Fit on `[[1.,5.],[2.,5.],[3.,5.]]`: col 1 is constant -> forward maps to
    /// fr[0] = 0.0; inverse of that recovers 5.0 (= data_min). Col 0 round-trips
    /// to `[1,2,3]`. Live oracle:
    ///   `m=MinMaxScaler().fit([[1.,5.],[2.,5.],[3.,5.]]);
    ///    m.inverse_transform(m.transform([[1.,5.],[2.,5.],[3.,5.]]))`
    ///   -> `[[1.,5.],[2.,5.],[3.,5.]]`.
    #[test]
    fn min_max_inverse_constant_col() -> Result<(), FerroError> {
        let scaler = MinMaxScaler::<f64>::new();
        let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let fitted = scaler.fit(&x, &())?;
        let scaled = fitted.transform(&x)?;

        // Constant col 1 maps to fr[0] = 0.0 forward (REQ-2).
        for i in 0..3 {
            assert_abs_diff_eq!(scaled[[i, 1]], 0.0, epsilon = 1e-9);
        }

        let recovered = fitted.inverse_transform(&scaled)?;
        let sk_col0 = [1.0_f64, 2.0, 3.0];
        for i in 0..3 {
            assert_abs_diff_eq!(recovered[[i, 0]], sk_col0[i], epsilon = 1e-9);
            assert_abs_diff_eq!(recovered[[i, 1]], 5.0, epsilon = 1e-9);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // REQ-5: scale_/min_/data_range_/n_samples_seen_ fitted attributes
    // (sklearn _data.py:507-514, _handle_zeros_in_scale :88). Live sklearn
    // 1.5.2 oracle values (R-CHAR-3).
    // -----------------------------------------------------------------------

    /// Fitted affine attributes match sklearn (default range, non-constant cols).
    /// Live oracle: `m=MinMaxScaler().fit([[1.,10.],[2.,20.],[3.,30.],[5.,50.]])`
    ///   -> `m.scale_=[0.25,0.025]`, `m.min_=[-0.25,-0.25]`,
    ///      `m.data_range_=[4.,40.]`, `m.n_samples_seen_=4`.
    #[test]
    fn min_max_attrs_match_sklearn() -> Result<(), FerroError> {
        let scaler = MinMaxScaler::<f64>::new();
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [5.0, 50.0]];
        let fitted = scaler.fit(&x, &())?;

        let scale = fitted.scale();
        let min = fitted.min();
        let data_range = fitted.data_range();

        assert_abs_diff_eq!(scale[0], 0.25, epsilon = 1e-9);
        assert_abs_diff_eq!(scale[1], 0.025, epsilon = 1e-9);
        assert_abs_diff_eq!(min[0], -0.25, epsilon = 1e-9);
        assert_abs_diff_eq!(min[1], -0.25, epsilon = 1e-9);
        assert_abs_diff_eq!(data_range[0], 4.0, epsilon = 1e-9);
        assert_abs_diff_eq!(data_range[1], 40.0, epsilon = 1e-9);
        assert_eq!(fitted.n_samples_seen(), 4);
        Ok(())
    }

    /// Constant column: `data_range_=0` is replaced by 1 in `scale_`
    /// (_handle_zeros_in_scale, _data.py:88). Live oracle:
    ///   `m=MinMaxScaler(feature_range=(-1,1)).fit([[1.,5.],[2.,5.],[3.,5.]])`
    ///   -> `m.data_range_=[2.,0.]`, `m.scale_=[1.,2.]`, `m.min_=[-2.,-11.]`.
    #[test]
    fn min_max_scale_handles_zero_range_constant_col() -> Result<(), FerroError> {
        let scaler = MinMaxScaler::<f64>::with_feature_range(-1.0, 1.0)?;
        let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let fitted = scaler.fit(&x, &())?;

        let scale = fitted.scale();
        let min = fitted.min();
        let data_range = fitted.data_range();

        assert_abs_diff_eq!(data_range[0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(data_range[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(scale[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(scale[1], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(min[0], -2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(min[1], -11.0, epsilon = 1e-9);
        Ok(())
    }

    /// inverse_transform with the wrong number of columns -> ShapeMismatch.
    #[test]
    fn min_max_inverse_shape_mismatch() -> Result<(), FerroError> {
        let scaler = MinMaxScaler::<f64>::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = scaler.fit(&x_train, &())?;
        let x_bad = array![[0.0], [1.0]];
        assert!(matches!(
            fitted.inverse_transform(&x_bad),
            Err(FerroError::ShapeMismatch { .. })
        ));
        Ok(())
    }
}
