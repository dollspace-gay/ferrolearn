//! Standard scaler: zero-mean, unit-variance scaling.
//!
//! Each feature is transformed as `(x - mean) / std`. (Near-)constant columns —
//! detected by sklearn's variance-relative `_is_constant_feature`
//! (`_data.py:72-85`), NOT the `< 10*eps` default mask used by
//! Min/MaxAbs/Robust — use an effective scale of 1, so each fitted-constant
//! entry maps to `(x - mean) / 1 = 0`. NaN is tolerated (ignored in `fit`,
//! passed through `transform`/`inverse_transform`); +/-inf is rejected.
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
//! | REQ-2 (constant / (near-)zero-variance column → 0) | SHIPPED | FIXED #1191; constant detection UPGRADED to `_is_constant_feature` (#1194 build). The exact-zero `s == 0` mask is REPLACED by sklearn's variance-relative `_is_constant_feature(var, mean, n)` — a column is constant iff `var <= n*eps*var + (n*mean*eps)^2` (`_data.py:72-85`,`:1016-1018`), so a constant col gets stored `scale_ = 1` → `(x-mean)/1 = 0`, and now a NEAR-constant col (tiny var below the bound, e.g. `[1e8,1e8+1e-8]`) ALSO → `scale_=1` (was previously divided by a tiny std). `transform`/`inverse_transform` use the STORED `scale_` (no `s_eff` recompute), matching sklearn `_handle_zeros_in_scale(sqrt(var_), constant_mask=...)` (`:88`,`:1019-1021`). Live-oracle tests: `req7_near_constant_column_scale_one`, `req7_genuinely_constant_column_scale_one` (`tests/divergence_standard_scaler.rs`) + existing `divergence_constant_column_maps_to_zero`/`green_req2_*`. NON-constant normal-magnitude columns are UNCHANGED (regression-guarded by `green_req1_*`). |
//! | REQ-3 (inverse_transform round-trip) | SHIPPED | `inverse_transform` = `x*s_eff + mean`, mirroring sklearn `X *= scale_; X += mean_` (`_data.py:1106-1109`); `inverse_transform(transform(X))==X` (`green_req3_inverse_roundtrip`). Consumer: PyO3 `_RsStandardScaler::inverse_transform`. |
//! | REQ-4 (PyO3 binding) | SHIPPED | `_RsStandardScaler` (`transformers.rs:12`, registered `lib.rs:22`) marshals `fit`/`transform`/`inverse_transform`/`mean_` over `FittedStandardScaler<f64>` — a real CPython consumer. |
//! | REQ-5 (var_/scale_/n_samples_seen_ attrs) | SHIPPED | FIXED #1192; constant mask UPGRADED to `_is_constant_feature` (#1194 build). `FittedStandardScaler<F>` stores `var_` (population variance ddof=0 via the corrected Chan-Golub-LeVeque 2-pass reduction over FINITE values — `nuv = sum((x-mean)^2) - (sum(x-mean))^2/count`, `var = nuv/count`, matching sklearn `_incremental_mean_and_var` `extmath.py:1142-1178`), `scale_` (= `_handle_zeros_in_scale(sqrt(var_), constant_mask)`: `std` on non-constant cols, `1.0` where `_is_constant_feature` flags the col — VARIANCE-RELATIVE, not exact-zero), and `n_samples_seen_` (= n_rows), set in `Fit::fit`; getters `var()`/`scale()`/`n_samples_seen()` mirror sklearn `var_`/`scale_`/`n_samples_seen_` (`_data.py:1013-1023`). Live oracle `X=[[1,5],[2,5],[3,5]]` (`var_=[0.6666666666666666,0.0]`, `scale_=[0.816496580927726,1.0]`, `n_samples_seen_=3`): `standard_scaler_var_scale_nsamples_match_sklearn`, `standard_scaler_scale_differs_from_std_on_constant_col`, `standard_scaler_var_equals_std_squared_nonconstant`. Additive getters; `transform`/`inverse_transform` use stored `scale_`. |
//! | REQ-6 (with_mean/with_std/copy params) | SHIPPED | FIXED #1193. `StandardScaler<F>` now carries `with_mean`/`with_std`/`copy` (default `true`, `new()`/`Default`), mirroring sklearn `__init__(*, copy=True, with_mean=True, with_std=True)` + `_parameter_constraints` (`_data.py:829-838`), with `#[must_use]` builders `with_with_mean`/`with_with_std`/`with_copy`. `with_mean`/`with_std` thread into `FittedStandardScaler<F>` (set in `Fit::fit`); `transform` is conditional `if with_mean: X -= mean_; if with_std: X /= scale_` and `inverse_transform` mirrors `if with_std: X *= scale_; if with_mean: X += mean_` (`_data.py:1064-1067`,`:1106-1109`). `copy` is ABI-only (fit/transform operate on owned copies). Critic-grounded vs live oracle `X=[[1,10],[2,20],[3,30]]`: default (T,T) → `[[-1.2247..],[0],[1.2247..]]` (regression-guarded bit-identical to pre-existing default fit), with_std=false (T,F) → center-only `[[-1,-10],[0,0],[1,10]]`, with_mean=false (F,T) → scale-only `[[1.2247..],[2.4495..],[3.6742..]]`, both-false (F,F) → identity, and inverse round-trip for all 4 configs. R-DEV-4 DEVIATION: sklearn nulls `mean_`/`scale_`/`var_` when a flag is `False`; ferrolearn always materializes them (getters return `&Array1`, not `Option`) — the flags govern transform APPLICATION so OUTPUT matches sklearn exactly. |
//! | REQ-7 (NaN tolerance: allow-nan) | SHIPPED | FIXED #1194. `Fit::fit`'s per-column mean + population variance now compute over the FINITE values only (skip `is_nan()`), with the divisor = COUNT OF FINITE values (`new_sample_count = n_samples - nan_count`, `extmath.py:1132-1133`), mirroring sklearn `force_all_finite="allow-nan"` (`_data.py:918`) + `_incremental_mean_and_var` ("NaNs are ignored", `extmath.py:1100`). An ALL-NaN column → `mean`/`var_`/`scale_` = NaN (no panic, no division by zero). NaN inputs pass through `transform`/`inverse_transform` (`(nan-mean)/scale = nan`). inf-rejection (allow-nan REJECTS +/-inf, MinMaxScaler #2200 precedent): `fit`/`transform`/`inverse_transform` return `InvalidParameter` ("Input X contains infinity...") on any `is_infinite()` element. Live-oracle tests `req7_nan_fit_single_column_ignored` (mean_=2,var_=1,scale_=1,transform `[[-1],[nan],[1]]`), `req7_nan_fit_multi_column_scattered`, `req7_all_nan_column_yields_nan_no_panic`, `req7_nan_passthrough_inverse_transform`, `inf_rejected_fit`/`inf_rejected_transform`/`inf_rejected_inverse_transform`, `nan_only_still_fits` (`tests/divergence_standard_scaler.rs`). Consumers: PyO3 `_RsStandardScaler` + `FittedPipelineTransformer` + re-export. **f32 caveat (#2205):** sklearn ALWAYS computes mean/variance in float64 accumulators even for float32 input (`_data.py:81-82`, `_incremental_mean_and_var`), returning float64 `mean_`/`var_`/`scale_`; ferrolearn computes the reduction generically in `F`, so on f32 a large-magnitude column (near 2^24) diverges (e.g. `mean` f32-rounds where sklearn's float64 mean does not). Same class as OPTICS #2195 (generic-F vs sklearn-float64-upcast); the f64 path (default + the f64 Python binding) is bit-exact. Tracked #2205, pinned `#[ignore]` in `tests/divergence_standard_scaler.rs::divergence_f32_uses_float64_accumulators`. Future fix: compute the mean/var reduction in f64 regardless of `F`. |
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
    /// match the number of features seen during fitting, or
    /// [`FerroError::InvalidParameter`] if any input element is +/-inf
    /// (sklearn `force_all_finite="allow-nan"` rejects infinity, `_data.py:1094`).
    pub fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.mean.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedStandardScaler::inverse_transform".into(),
            });
        }
        // sklearn `inverse_transform` validates with `force_all_finite="allow-nan"`
        // (`_data.py:1094`): NaN passes through, +/-inf raises ValueError
        // (MinMaxScaler #2200/#2202 precedent).
        if x.iter().any(|v| v.is_infinite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains infinity or a value too large for dtype.".into(),
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
        // sklearn validates X with `force_all_finite="allow-nan"`
        // (`_data.py:918`): NaN is permitted, but +/-inf raises ValueError
        // ("Input X contains infinity or a value too large for dtype('...')").
        // Mirrors the MinMaxScaler #2200 precedent (allow-nan rejects inf).
        if x.iter().any(|v| v.is_infinite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains infinity or a value too large for dtype.".into(),
            });
        }

        let n_features = x.ncols();
        let mut mean = Array1::zeros(n_features);
        let mut std_arr = Array1::zeros(n_features);
        let mut var_arr = Array1::zeros(n_features);
        // Per-feature finite sample count (= n_samples - NaN count), reused for
        // the `_is_constant_feature` bound. sklearn's `n_samples_seen_` is the
        // finite count under `allow-nan` (`extmath.py:1132-1133`, `_data.py:996`).
        let mut finite_count = Array1::<usize>::zeros(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            // NaN-ignoring mean + population variance (ddof=0) over the FINITE
            // values only, mirroring sklearn's `_incremental_mean_and_var` under
            // `force_all_finite="allow-nan"` (`_data.py:918`,`:999`;
            // `extmath.py:1100` "NaNs are ignored"). The divisor is the COUNT OF
            // FINITE values (`new_sample_count = n_samples - nan_count`,
            // `extmath.py:1132-1133`), NOT `n_samples`. An all-NaN column has
            // count 0 → mean/var = NaN (matching sklearn's `nan` there; no panic,
            // no division materialized).
            let mut count: usize = 0;
            let mut sum = F::zero();
            for v in col.iter().copied() {
                if v.is_nan() {
                    continue;
                }
                count += 1;
                sum = sum + v;
            }
            finite_count[j] = count;
            if count == 0 {
                // All-NaN column: mean/var/std = NaN (no division by zero).
                mean[j] = F::nan();
                var_arr[j] = F::nan();
                std_arr[j] = F::nan();
                continue;
            }
            let cnt = F::from(count).unwrap_or_else(F::one);
            let m = sum / cnt;
            // Corrected 2-pass variance (Chan/Golub/LeVeque), matching sklearn's
            // `_incremental_mean_and_var` first-batch reduction (`extmath.py:1142`
            // `T = new_sum/new_sample_count`, `:1155-1162`
            // `nuv = sum((x-T)^2) - (sum(x-T))^2/count`, `:1178`
            // `var = nuv/count`). The correction term is load-bearing for
            // near-constant columns (e.g. `[1e8, 1e8+1e-8]` → `5.55e-17`, where
            // the naive `sum((x-mean)^2)/count` would give `1.11e-16`); on
            // ordinary columns it is numerically negligible (byte-identical to
            // the prior `sum((x-mean)^2)/count` path on finite, normal-magnitude
            // data).
            let mut sum_dev = F::zero();
            let mut sum_dev_sq = F::zero();
            for v in col.iter().copied() {
                if v.is_nan() {
                    continue;
                }
                let d = v - m;
                sum_dev = sum_dev + d;
                sum_dev_sq = sum_dev_sq + d * d;
            }
            let variance = (sum_dev_sq - (sum_dev * sum_dev) / cnt) / cnt;
            mean[j] = m;
            var_arr[j] = variance;
            std_arr[j] = variance.sqrt();
        }

        // `scale_` mirrors sklearn `_handle_zeros_in_scale(sqrt(var_),
        // constant_mask=_is_constant_feature(var_, mean_, n_samples_seen_))`
        // (`_data.py:1016-1021`,`:88-120`). UNLIKE Min/MaxAbs/Robust (which use
        // the default `scale < 10*eps` mask, `:114`), StandardScaler passes an
        // explicit variance-relative constant mask computed on the RAW variances:
        // a column is constant iff `var <= n*eps*var + (n*mean*eps)^2`
        // (`_is_constant_feature`, `:72-85`). `eps` is ALWAYS `f64::EPSILON`
        // (sklearn: "variance is always computed using float64 accumulators",
        // `:81-82`) — even on the f32 path the bound is computed by upcasting to
        // f64, matching sklearn. A constant column gets `scale_ = 1.0` (NOT
        // sqrt(var)); a near-constant column (tiny var below the bound) is also
        // flagged and gets `scale_ = 1.0`. A NaN var (all-NaN column) is not
        // `<= bound` (NaN compares false) → constant_mask false → scale_ = NaN.
        let f64_eps = f64::EPSILON;
        let mut scale_ = std_arr.clone();
        for j in 0..n_features {
            let var_f64 = var_arr[j].to_f64().unwrap_or(f64::NAN);
            let mean_f64 = mean[j].to_f64().unwrap_or(f64::NAN);
            let n_f64 = finite_count[j] as f64;
            let upper_bound = n_f64 * f64_eps * var_f64 + (n_f64 * mean_f64 * f64_eps).powi(2);
            // `_is_constant_feature`: var <= upper_bound (`_data.py:85`).
            if var_f64 <= upper_bound {
                scale_[j] = F::one();
            }
        }

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
    /// (Near-)constant columns use the precomputed `scale_ = 1` (sklearn
    /// `_is_constant_feature` + `_handle_zeros_in_scale`, `_data.py:72-85`,
    /// `:88-120`,`:1016-1021`), so a fitted-constant entry maps to
    /// `(x - mean) / 1 = 0`. NaN inputs pass through (`(nan - mean)/scale = nan`)
    /// and an all-NaN fitted column (whose `mean`/`scale_` are NaN) transforms to
    /// NaN, matching sklearn's `allow-nan` contract (`_data.py:1052`,`:1064-1067`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting, or
    /// [`FerroError::InvalidParameter`] if any input element is +/-inf
    /// (sklearn `force_all_finite="allow-nan"` rejects infinity, `_data.py:1052`).
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.mean.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedStandardScaler::transform".into(),
            });
        }
        // sklearn `transform` validates with `force_all_finite="allow-nan"`
        // (`_data.py:1052`): NaN passes through, +/-inf raises ValueError
        // (MinMaxScaler #2200 precedent).
        if x.iter().any(|v| v.is_infinite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains infinity or a value too large for dtype.".into(),
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
