//! Divergence pins for `QuantileTransformer` on a CONSTANT feature (all values
//! equal). scikit-learn's forward `_transform_col` applies output-distribution
//! dependent BOUND MASKS that OVERRIDE the interpolated value
//! (`sklearn/preprocessing/_data.py:2826-2851`):
//!
//! ```text
//! if output_distribution == "uniform":
//!     lower_bounds_idx = X_col == lower_bound_x   # == quantiles[0]
//!     upper_bounds_idx = X_col == upper_bound_x   # == quantiles[-1]
//! ...
//! X_col[upper_bounds_idx] = upper_bound_y   # = 1
//! X_col[lower_bounds_idx] = lower_bound_y   # = 0
//! ```
//!
//! For a constant column every quantile landmark equals the constant `c`
//! (`np.maximum.accumulate` of equal nanpercentiles). Every input value `== c`,
//! so it matches `lower_bound_x == quantiles[0]` AND `upper_bound_x ==
//! quantiles[-1]`. sklearn applies upper THEN lower (`:2850-2851`), so the LOWER
//! override wins and the uniform output is `0.0` for every row (and for normal,
//! `lower_bound_y = 0` → `stats.norm.ppf(0)` clipped to `clip_min`).
//!
//! ferrolearn (`quantile_transformer.rs` `interpolate_cdf` / `transform`) omits
//! these forward bound masks entirely and instead returns the AVERAGED
//! forward/reversed interpolation, which for a degenerate single-plateau column
//! collapses to the midpoint `0.5` (then `probit(0.5) == 0.0` for normal).
//!
//! Expected values are computed by the LIVE sklearn 1.5.2 oracle (R-CHAR-3),
//! NOT copied from ferrolearn.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::quantile_transformer::{OutputDistribution, QuantileTransformer};
use ndarray::array;

/// Divergence: ferrolearn `QuantileTransformer::transform` (uniform) diverges
/// from `sklearn/preprocessing/_data.py:2829-2851` for a constant feature.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// X = [[5.],[5.],[5.],[5.]]
/// QuantileTransformer(n_quantiles=3, subsample=None, output_distribution='uniform')
///     .fit(X).transform(X).ravel() == [0.0, 0.0, 0.0, 0.0]
/// ```
/// ferrolearn returns `[0.5, 0.5, 0.5, 0.5]` (averaged-interp midpoint; the
/// `X_col == lower_bound_x` → `lower_bound_y = 0` override is missing).
///
/// Tracking: #2318
#[test]
fn divergence_constant_feature_uniform() {
    // sklearn oracle value (np.nanpercentile of a constant column, masked to 0).
    const SK_UNIFORM: f64 = 0.0;

    let x = array![[5.0_f64], [5.0], [5.0], [5.0]];
    let fitted = QuantileTransformer::<f64>::new(3, OutputDistribution::Uniform, 0)
        .fit(&x, &())
        .unwrap();
    let out = fitted.transform(&x).unwrap();

    for &v in out.iter() {
        assert!(
            (v - SK_UNIFORM).abs() < 1e-7,
            "constant-feature uniform: sklearn={SK_UNIFORM}, ferrolearn={v}"
        );
    }
}

/// Divergence: ferrolearn `QuantileTransformer::transform` (normal) diverges
/// from `sklearn/preprocessing/_data.py:2826-2862` for a constant feature.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// X = [[5.],[5.],[5.],[5.]]
/// QuantileTransformer(n_quantiles=3, subsample=None, output_distribution='normal')
///     .fit(X).transform(X).ravel()
///     == [-5.199337582605575, -5.199337582605575, -5.199337582605575, -5.199337582605575]
/// ```
/// (the `X_col - BOUNDS_THRESHOLD < lower_bound_x` normal mask → `lower_bound_y
/// = 0` → `stats.norm.ppf(0)` clipped to `clip_min = -5.199337582605575`.)
/// ferrolearn returns `[0.0, 0.0, 0.0, 0.0]` (`probit(0.5)`).
///
/// Tracking: #2319
#[test]
fn divergence_constant_feature_normal() {
    // sklearn oracle: stats.norm.ppf(0) clipped to clip_min.
    const SK_NORMAL_CLIP_MIN: f64 = -5.199337582605575;

    let x = array![[5.0_f64], [5.0], [5.0], [5.0]];
    let fitted = QuantileTransformer::<f64>::new(3, OutputDistribution::Normal, 0)
        .fit(&x, &())
        .unwrap();
    let out = fitted.transform(&x).unwrap();

    for &v in out.iter() {
        assert!(
            (v - SK_NORMAL_CLIP_MIN).abs() < 1e-7,
            "constant-feature normal: sklearn={SK_NORMAL_CLIP_MIN}, ferrolearn={v}"
        );
    }
}
