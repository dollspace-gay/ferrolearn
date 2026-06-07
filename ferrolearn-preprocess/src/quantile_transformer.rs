//! Quantile transformer: map features to a uniform or normal distribution.
//!
//! [`QuantileTransformer`] transforms features by mapping each value through
//! its empirical cumulative distribution function (CDF), producing values
//! uniformly distributed in `[0, 1]`. Optionally, the result can be mapped
//! to a standard normal distribution using the inverse normal CDF (probit).
//!
//! This is useful for making features more Gaussian-like, which can improve
//! the performance of many machine learning algorithms.
//!
//! Translation target: scikit-learn 1.5.2 `class QuantileTransformer`
//! (`sklearn/preprocessing/_data.py:2540`) + `quantile_transform` (`:2978`).
//! Design: `.design/preprocess/quantile_transformer.md`. Tracking: #1319.
//!
//! `## REQ status`
//!
//! | REQ | Status | Anchor |
//! |---|---|---|
//! | REQ-1 forward-CDF value surface (uniform + normal, distinct + tied) | SHIPPED | `fit` references/landmarks (#1322) + `transform`; sklearn `_data.py:2694`,`:2702`,`:2795` |
//! | REQ-2 forward+reversed AVERAGED interpolation | SHIPPED (#1321) | `interpolate_cdf`/`np_interp`; sklearn `_data.py:2843-2846` |
//! | REQ-3 Normal output accuracy (Acklam ppf + clip) | SHIPPED (#1320) | `probit`; sklearn `_data.py:2855-2862` |
//! | REQ-5 error/parameter contracts (scoped) | SHIPPED | `fit`/`transform` guards; sklearn `_data.py:2654` |
//! | REQ-4 `np.maximum.accumulate` monotonic repair (unobservable) | NOT-STARTED (#1323) | sklearn `_data.py:2707` |
//! | REQ-6 random subsample + random_state + n_quantiles>subsample error | NOT-STARTED (#1324) | sklearn `_data.py:2696-2700`,`:2774-2779` |
//! | REQ-7 `inverse_transform` (reverse interp + `norm.cdf` + bounds + NaN) | SHIPPED (#1325) | `FittedQuantileTransformer::inverse_transform` + `norm_cdf` (ndtr via `erf`/`erfc`); sklearn `_data.py:2947`,`:2813-2851`,`:2821`. Uniform ~exact, Normal ~1e-7 (follows the forward REQ-3 ~1e-9 ndtr/probit contract). Consumer: crate re-export `pub use quantile_transformer::FittedQuantileTransformer` (`lib.rs`, boundary public API). Live-oracle: `tests/divergence_quantile_transformer.rs` (uniform/normal round-trip + held-out + bounds + NaN + `norm_cdf` sanity) |
//! | REQ-8 `ignore_implicit_zeros` + sparse CSC | NOT-STARTED (#1326) | sklearn `_data.py:2709-2752` |
//! | REQ-9 `quantile_transform` free function | NOT-STARTED (#1327) | sklearn `_data.py:2978` |
//! | REQ-10 `copy` + OneToOneFeatureMixin fitted-attr surface | NOT-STARTED (#1328) | sklearn `_data.py:2540`,`:2790-2795` |
//! | REQ-11 PyO3 binding | NOT-STARTED (#1329) | `ferrolearn-python/src/` (absent) |
//! | REQ-12 ferray substrate | NOT-STARTED (#1330) | R-SUBSTRATE |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_numerical::special::{erf, erfc};
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// OutputDistribution
// ---------------------------------------------------------------------------

/// Target output distribution for the quantile transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDistribution {
    /// Map to the uniform distribution on `[0, 1]`.
    Uniform,
    /// Map to the standard normal distribution via the probit function.
    Normal,
}

// ---------------------------------------------------------------------------
// QuantileTransformer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted quantile transformer.
///
/// Calling [`Fit::fit`] computes the quantiles for each feature and returns a
/// [`FittedQuantileTransformer`].
///
/// # Parameters
///
/// - `n_quantiles` — number of quantile reference points (default 1000).
/// - `output_distribution` — target distribution (default `Uniform`).
/// - `subsample` — maximum number of samples used to compute quantiles
///   (default 100_000; set to 0 to use all samples).
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::quantile_transformer::{QuantileTransformer, OutputDistribution};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
/// let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
/// let fitted = qt.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// // Values should be in [0, 1]
/// for v in out.iter() {
///     assert!(*v >= 0.0 && *v <= 1.0);
/// }
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct QuantileTransformer<F> {
    /// Number of quantile reference points.
    n_quantiles: usize,
    /// Target output distribution.
    output_distribution: OutputDistribution,
    /// Maximum number of samples for quantile computation (0 = all).
    subsample: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> QuantileTransformer<F> {
    /// Create a new `QuantileTransformer`.
    pub fn new(
        n_quantiles: usize,
        output_distribution: OutputDistribution,
        subsample: usize,
    ) -> Self {
        Self {
            n_quantiles,
            output_distribution,
            subsample,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the number of quantiles.
    #[must_use]
    pub fn n_quantiles(&self) -> usize {
        self.n_quantiles
    }

    /// Return the target output distribution.
    #[must_use]
    pub fn output_distribution(&self) -> OutputDistribution {
        self.output_distribution
    }

    /// Return the subsample size.
    #[must_use]
    pub fn subsample(&self) -> usize {
        self.subsample
    }
}

impl<F: Float + Send + Sync + 'static> Default for QuantileTransformer<F> {
    fn default() -> Self {
        Self::new(1000, OutputDistribution::Uniform, 100_000)
    }
}

// ---------------------------------------------------------------------------
// FittedQuantileTransformer
// ---------------------------------------------------------------------------

/// A fitted quantile transformer holding per-feature quantile references.
///
/// Created by calling [`Fit::fit`] on a [`QuantileTransformer`].
#[derive(Debug, Clone)]
pub struct FittedQuantileTransformer<F> {
    /// Quantile reference values per feature: `quantiles[j]` is a sorted
    /// vector of reference values for feature `j`.
    quantiles: Vec<Vec<F>>,
    /// The reference quantile levels (evenly spaced in [0, 1]).
    references: Vec<F>,
    /// Target output distribution.
    output_distribution: OutputDistribution,
}

impl<F: Float + Send + Sync + 'static> FittedQuantileTransformer<F> {
    /// Return the computed quantile reference values per feature.
    #[must_use]
    pub fn quantiles(&self) -> &[Vec<F>] {
        &self.quantiles
    }

    /// Return the number of features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.quantiles.len()
    }

    /// Back-project transformed data to the original feature space.
    ///
    /// The inverse of [`Transform::transform`]: each transformed column value is
    /// mapped back to an original feature value. Mirrors scikit-learn
    /// `QuantileTransformer.inverse_transform` (`_data.py:2947`) which runs
    /// `_transform_col(..., inverse=True)` (`_data.py:2813-2866`):
    ///
    /// - **Normal** output: the value is first mapped to its uniform rank in
    ///   `[0, 1]` via the standard normal CDF `stats.norm.cdf` (`:2821`,
    ///   [`norm_cdf`]); the bound masks use `BOUNDS_THRESHOLD = 1e-7` (`:47`)
    ///   against `0` / `1` (`:2827-2828`).
    /// - **Uniform** output: the value IS the rank; the bound masks use exact
    ///   equality `== 0` / `== 1` (`:2830-2831`).
    ///
    /// The finite ranks are interpolated with a **plain** `np.interp(rank,
    /// references_, quantiles)` — the REVERSE of the forward interp, and NOT the
    /// forward/reversed-averaged variant (the averaging is forward-only,
    /// `:2848`). Ranks at/below the lower bound map to the column minimum
    /// `quantiles[0]` and at/above the upper bound to the column maximum
    /// `quantiles[-1]` (`:2850-2851`). `NaN` passes through unchanged
    /// (`isfinite_mask`, `:2833`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    pub fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.quantiles.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedQuantileTransformer::inverse_transform".into(),
            });
        }
        // sklearn `inverse_transform` -> `_check_inputs(force_all_finite=
        // "allow-nan")` (`_data.py:2876`, called `:2965`): NaN passes through,
        // but +/-inf raises ValueError (#2212, MinMaxScaler #2200 precedent).
        if x.iter().any(|v| v.is_infinite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains infinity or a value too large for dtype.".into(),
            });
        }

        // sklearn _transform_col inverse: lower_bound_x = 0, upper_bound_x = 1,
        // lower_bound_y = quantiles[0], upper_bound_y = quantiles[-1]
        // (`_data.py:2814-2817`). BOUNDS_THRESHOLD = 1e-7 (`:47`).
        let bounds_threshold = F::from(1e-7).unwrap_or_else(F::zero);
        let zero = F::zero();
        let one = F::one();

        let mut out = x.to_owned();

        for j in 0..n_features {
            let quantiles_col = &self.quantiles[j];
            let (lower_y, upper_y) = match (quantiles_col.first(), quantiles_col.last()) {
                (Some(&first), Some(&last)) => (first, last),
                // Empty landmark column: nothing to map to; leave values as-is.
                _ => continue,
            };

            for i in 0..out.nrows() {
                let val = out[[i, j]];
                if val.is_nan() {
                    // isfinite_mask excludes NaN; it stays NaN (`_data.py:2833`).
                    continue;
                }

                // For Normal output, map the normal-space value back to the
                // uniform rank in [0, 1] (`_data.py:2821`). For Uniform the
                // value already IS the rank.
                let rank = match self.output_distribution {
                    OutputDistribution::Uniform => val,
                    OutputDistribution::Normal => norm_cdf(val),
                };

                // Bound masks (`_data.py:2826-2831`). Under errstate(invalid=
                // "ignore") NaN comparisons are false — but `rank` is finite
                // here (norm_cdf of a finite value is finite, val is finite).
                let (lower_idx, upper_idx) = match self.output_distribution {
                    OutputDistribution::Normal => (
                        rank - bounds_threshold < zero,
                        rank + bounds_threshold > one,
                    ),
                    OutputDistribution::Uniform => (rank == zero, rank == one),
                };

                // Plain reverse interp references_ -> quantiles (`_data.py:2848`).
                let mut mapped = np_interp(rank, &self.references, quantiles_col);

                // Bound overrides (`_data.py:2850-2851`): upper applied first,
                // then lower (matching sklearn's order).
                if upper_idx {
                    mapped = upper_y;
                }
                if lower_idx {
                    mapped = lower_y;
                }

                out[[i, j]] = mapped;
            }
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Inverse of the standard normal CDF (probit / norm.ppf), accurate to ~1e-9
/// via Acklam's rational approximation. Mirrors scipy `stats.norm.ppf` as used
/// by sklearn QuantileTransformer (`_data.py:2856`), with the output clipped to
/// +/- norm.ppf(1e-7 - spacing(1)) = +/- 5.199337582605575 (`_data.py:2860-2862`).
fn probit<F: Float>(p: F) -> F {
    let clip = F::from(5.199337582605575).unwrap_or_else(F::max_value);
    let cf = |k: f64| F::from(k).unwrap_or_else(F::zero);
    if p <= F::zero() {
        return -clip;
    }
    if p >= F::one() {
        return clip;
    }
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = F::from(0.02425).unwrap_or_else(F::zero);
    let p_high = F::one() - p_low;
    let two = F::from(2.0).unwrap_or_else(F::one);
    let half = F::from(0.5).unwrap_or_else(F::zero);
    let x = if p < p_low {
        let qv = (-two * p.ln()).sqrt();
        (((((cf(c[0]) * qv + cf(c[1])) * qv + cf(c[2])) * qv + cf(c[3])) * qv + cf(c[4])) * qv
            + cf(c[5]))
            / ((((cf(d[0]) * qv + cf(d[1])) * qv + cf(d[2])) * qv + cf(d[3])) * qv + F::one())
    } else if p <= p_high {
        let qv = p - half;
        let r = qv * qv;
        (((((cf(a[0]) * r + cf(a[1])) * r + cf(a[2])) * r + cf(a[3])) * r + cf(a[4])) * r
            + cf(a[5]))
            * qv
            / (((((cf(b[0]) * r + cf(b[1])) * r + cf(b[2])) * r + cf(b[3])) * r + cf(b[4])) * r
                + F::one())
    } else {
        let qv = (-two * (F::one() - p).ln()).sqrt();
        -(((((cf(c[0]) * qv + cf(c[1])) * qv + cf(c[2])) * qv + cf(c[3])) * qv + cf(c[4])) * qv
            + cf(c[5]))
            / ((((cf(d[0]) * qv + cf(d[1])) * qv + cf(d[2])) * qv + cf(d[3])) * qv + F::one())
    };
    if x < -clip {
        -clip
    } else if x > clip {
        clip
    } else {
        x
    }
}

/// numpy-`np.interp`-faithful 1-D linear interpolation (`xp` ascending, may have
/// duplicate landmarks). Used twice (ascending + reversed) and averaged in the
/// CDF lookup to mirror sklearn `_transform_col` (`_data.py:2843-2846`).
fn np_interp<F: Float>(x: F, xp: &[F], fp: &[F]) -> F {
    let n = xp.len();
    if n == 0 {
        return F::zero();
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    // first index with xp[idx] > x (numpy searchsorted side='right'); interval (idx-1, idx)
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if xp[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let i = lo - 1;
    let denom = xp[i + 1] - xp[i];
    if denom == F::zero() {
        fp[i]
    } else {
        fp[i] + (x - xp[i]) / denom * (fp[i + 1] - fp[i])
    }
}

/// Standard normal CDF Φ(x) (scipy `ndtr` / `stats.norm.cdf`), computed in f64
/// for accuracy then cast back to `F`. Used by [`FittedQuantileTransformer::inverse_transform`]
/// to map a Normal-space value back to its uniform rank in `[0, 1]`, mirroring
/// sklearn `_transform_col` inverse branch (`_data.py:2821`: `stats.norm.cdf(X_col)`).
///
/// Built from [`ferrolearn_numerical::special::erf`]/[`erfc`] (libm Cephes,
/// machine precision) via the standard three-branch `ndtr` split on
/// `t = x / sqrt(2)`, which preserves tail accuracy by using `erfc` (not
/// `1 - erf`) in the tails:
/// - `t < -1`  → `0.5 * erfc(-t)`   (left tail; no catastrophic cancellation)
/// - `-1 ≤ t ≤ 1` → `0.5 * (1 + erf(t))` (central region)
/// - `t > 1`   → `1 - 0.5 * erfc(t)` (right tail)
///
/// This is the analytic inverse of the forward [`probit`] (Acklam ppf, ~1e-9),
/// so a Normal round-trip agrees with `scipy.stats.norm.cdf` to ~1e-9. A `NaN`
/// input is passed through (returns `NaN`); ±∞ map to 1/0 since `erf(±∞)=±1`.
fn norm_cdf<F: Float>(x: F) -> F {
    let xf = x.to_f64().unwrap_or(f64::NAN);
    if xf.is_nan() {
        return F::nan();
    }
    let t = xf / std::f64::consts::SQRT_2;
    let cdf = if t < -1.0 {
        0.5 * erfc(-t)
    } else if t <= 1.0 {
        0.5 * (1.0 + erf(t))
    } else {
        1.0 - 0.5 * erfc(t)
    };
    F::from(cdf).unwrap_or_else(F::zero)
}

/// Map `value` to its quantile level by averaging the ascending and reversed
/// linear interpolations, mirroring sklearn `_transform_col`
/// (`_data.py:2843-2846`) so plateaus map to the midpoint of their span.
fn interpolate_cdf<F: Float>(value: F, quantiles: &[F], references: &[F]) -> F {
    if quantiles.is_empty() {
        return F::from(0.5).unwrap_or_else(F::zero);
    }
    let forward = np_interp(value, quantiles, references);
    let neg_q: Vec<F> = quantiles.iter().rev().map(|&qv| -qv).collect();
    let neg_r: Vec<F> = references.iter().rev().map(|&rv| -rv).collect();
    let backward = np_interp(-value, &neg_q, &neg_r);
    F::from(0.5).unwrap_or_else(F::zero) * (forward - backward)
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for QuantileTransformer<F> {
    type Fitted = FittedQuantileTransformer<F>;
    type Error = FerroError;

    /// Fit by computing per-feature quantile reference values.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has fewer than 2 rows.
    /// - [`FerroError::InvalidParameter`] if `n_quantiles` is less than 2.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedQuantileTransformer<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "QuantileTransformer::fit".into(),
            });
        }
        if self.n_quantiles < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_quantiles".into(),
                reason: "n_quantiles must be at least 2".into(),
            });
        }

        let n_features = x.ncols();
        let effective_quantiles = self.n_quantiles.min(n_samples);

        // numpy np.linspace(0,1,K): step = 1/(K-1) computed once, y[i] = i*step,
        // endpoint y[K-1] pinned to exactly 1.0 (_data.py:2795).
        let k = effective_quantiles;
        let denom = F::from(k.saturating_sub(1)).unwrap_or_else(F::one);
        let step = F::one() / denom;
        let mut references: Vec<F> = (0..k)
            .map(|i| F::from(i).unwrap_or_else(F::zero) * step)
            .collect();
        if k >= 2 {
            let last = references.len() - 1;
            references[last] = F::one();
        }

        let mut quantiles = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let mut col_vals: Vec<F> = x.column(j).iter().copied().collect();
            // Remove NaN values
            col_vals.retain(|v| !v.is_nan());
            col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Subsample if needed
            if self.subsample > 0 && col_vals.len() > self.subsample {
                let step = col_vals.len() as f64 / self.subsample as f64;
                let mut sampled = Vec::with_capacity(self.subsample);
                for i in 0..self.subsample {
                    let idx = (i as f64 * step) as usize;
                    sampled.push(col_vals[idx.min(col_vals.len() - 1)]);
                }
                col_vals = sampled;
            }

            // Compute quantile reference values
            let n = col_vals.len();
            let mut feature_quantiles = Vec::with_capacity(effective_quantiles);
            for &ref_level in &references {
                // sklearn: np.nanpercentile(X, references_*100); numpy 'linear' virtual
                // index is q/100*(n-1). Replicate the *100 then /100 round-trip
                // (_data.py:2694,:2702).
                let hundred = F::from(100.0).unwrap_or_else(F::one);
                let q = ref_level * hundred;
                let pos = (q / hundred) * F::from(n.saturating_sub(1)).unwrap_or_else(F::one);
                let lo = pos.floor().to_usize().unwrap_or(0).min(n.saturating_sub(1));
                let hi = pos.ceil().to_usize().unwrap_or(0).min(n.saturating_sub(1));
                let frac = pos - F::from(lo).unwrap_or_else(F::zero);
                let val = if lo == hi {
                    col_vals[lo]
                } else {
                    col_vals[lo] * (F::one() - frac) + col_vals[hi] * frac
                };
                feature_quantiles.push(val);
            }

            quantiles.push(feature_quantiles);
        }

        Ok(FittedQuantileTransformer {
            quantiles,
            references,
            output_distribution: self.output_distribution,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedQuantileTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by mapping each value through the empirical CDF.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.quantiles.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedQuantileTransformer::transform".into(),
            });
        }

        let mut out = x.to_owned();

        for j in 0..n_features {
            let feature_quantiles = &self.quantiles[j];
            for i in 0..out.nrows() {
                let val = out[[i, j]];
                if val.is_nan() {
                    continue;
                }
                let cdf_val = interpolate_cdf(val, feature_quantiles, &self.references);

                out[[i, j]] = match self.output_distribution {
                    OutputDistribution::Uniform => cdf_val,
                    OutputDistribution::Normal => probit(cdf_val),
                };
            }
        }

        Ok(out)
    }
}

/// Implement `Transform` on the unfitted transformer to satisfy the
/// `FitTransform: Transform` supertrait bound.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for QuantileTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the transformer must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "QuantileTransformer".into(),
            reason: "transformer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for QuantileTransformer<F> {
    type FitError = FerroError;

    /// Fit and transform in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
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
    fn test_quantile_transformer_uniform() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let fitted = qt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // All values should be in [0, 1]
        for v in &out {
            assert!(*v >= 0.0 && *v <= 1.0, "Value {v} not in [0,1]");
        }
        // First should be 0, last should be 1
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[4, 0]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_quantile_transformer_normal() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Normal, 0);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let fitted = qt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Middle value should be close to 0 (median → 0 in normal)
        assert!(out[[2, 0]].abs() < 0.5, "Median should map near 0");
        // First should be negative, last positive
        assert!(out[[0, 0]] < out[[4, 0]]);
    }

    #[test]
    fn test_quantile_transformer_monotonic() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x = array![[5.0], [3.0], [1.0], [4.0], [2.0]];
        let fitted = qt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Transform should preserve ordering: rank(5) > rank(3) > rank(1)
        assert!(out[[0, 0]] > out[[1, 0]]); // 5 > 3
        assert!(out[[1, 0]] > out[[2, 0]]); // 3 > 1
    }

    #[test]
    fn test_quantile_transformer_multiple_features() {
        let qt = QuantileTransformer::<f64>::new(50, OutputDistribution::Uniform, 0);
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let fitted = qt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
        // Each feature independently transformed
        for j in 0..2 {
            assert!(out[[0, j]] <= out[[2, j]]);
        }
    }

    #[test]
    fn test_quantile_transformer_fit_transform() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let out = qt.fit_transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[4, 0]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_quantile_transformer_insufficient_samples_error() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x = array![[1.0]];
        assert!(qt.fit(&x, &()).is_err());
    }

    #[test]
    fn test_quantile_transformer_too_few_quantiles_error() {
        let qt = QuantileTransformer::<f64>::new(1, OutputDistribution::Uniform, 0);
        let x = array![[1.0], [2.0], [3.0]];
        assert!(qt.fit(&x, &()).is_err());
    }

    #[test]
    fn test_quantile_transformer_shape_mismatch() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = qt.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_quantile_transformer_unfitted_error() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x = array![[1.0]];
        assert!(qt.transform(&x).is_err());
    }

    #[test]
    fn test_quantile_transformer_default() {
        let qt = QuantileTransformer::<f64>::default();
        assert_eq!(qt.n_quantiles(), 1000);
        assert_eq!(qt.output_distribution(), OutputDistribution::Uniform);
        assert_eq!(qt.subsample(), 100_000);
    }

    #[test]
    fn test_quantile_transformer_f32() {
        let qt = QuantileTransformer::<f32>::new(50, OutputDistribution::Uniform, 0);
        let x: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0], [5.0]];
        let fitted = qt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert!(out[[0, 0]] >= 0.0f32);
        assert!(out[[4, 0]] <= 1.0f32);
    }
}
