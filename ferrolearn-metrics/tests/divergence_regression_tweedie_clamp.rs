//! Divergence pins for the Tweedie `power < 0` negative-`y_true` clamp.
//!
//! sklearn's `_mean_tweedie_deviance` (`sklearn/metrics/_regression.py:1280-1286`)
//! computes the `p < 0` ("Extreme stable") unit deviance with the first term
//! clamped to non-negative `y_true`:
//!
//! ```python
//! if p < 0:
//!     dev = 2 * (
//!         np.power(np.maximum(y_true, 0), 2 - p) / ((1 - p) * (2 - p))
//!         - y_true * np.power(y_pred, 1 - p) / (1 - p)
//!         + np.power(y_pred, 2 - p) / (2 - p)
//!     )
//! ```
//!
//! For `power < 0`, sklearn allows `y_true` to be any real number (only
//! `y_pred > 0` is required, `_regression.py:1530`). ferrolearn's general-case
//! Tweedie branch (`regression.rs:1019-1035` / `:1226-1234`) computes
//! `t.powf(2 - p)` directly on a negative `t` with a non-integer exponent,
//! which yields `NaN` in IEEE-754, instead of clamping `t` to `0`.
//!
//! All expected values come from the live sklearn 1.5.2 oracle:
//!   python3 -c "from sklearn.metrics import mean_tweedie_deviance as f; \
//!     import numpy as np; \
//!     print(repr(float(f(np.array([-2.,1.,3.]), np.array([1.5,2.5,2.]), power=-1.5))))"
//! Tracking: #2294

use ferrolearn_metrics::regression::{d2_tweedie_score, mean_tweedie_deviance};
use ndarray::array;

/// sklearn oracle:
///   mean_tweedie_deviance([-2,1,3], [1.5,2.5,2.0], power=-1.5) == 5.59634389721963
/// ferrolearn computes `(-2)^3.5 = NaN`, so the whole result is `NaN`.
/// Tracking: #2294
const SK_TWEEDIE_NEG_YTRUE: f64 = 5.59634389721963;

#[test]
fn divergence_mean_tweedie_power_neg_negative_ytrue_clamp() {
    let y_true = array![-2.0_f64, 1.0, 3.0];
    let y_pred = array![1.5_f64, 2.5, 2.0];
    let got = mean_tweedie_deviance(&y_true, &y_pred, -1.5).unwrap();
    assert!(
        (got - SK_TWEEDIE_NEG_YTRUE).abs() < 1e-10,
        "mean_tweedie_deviance(power=-1.5) negative y_true: sklearn={SK_TWEEDIE_NEG_YTRUE}, ferrolearn={got}"
    );
}

/// sklearn oracle:
///   d2_tweedie_score([-2,1,3,5], [1.5,2.5,2.0,4.0], power=-1.5) == 0.6220978677967901
/// ferrolearn's general Tweedie branch in d2_tweedie_score (`regression.rs:1226-1234`)
/// computes `(-2)^3.5 = NaN` in the numerator and `mean(y_true)`-based denominator,
/// yielding `NaN`.
/// Tracking: #2294
const SK_D2_TWEEDIE_NEG_YTRUE: f64 = 0.6220978677967901;

#[test]
fn divergence_d2_tweedie_power_neg_negative_ytrue_clamp() {
    let y_true = array![-2.0_f64, 1.0, 3.0, 5.0];
    let y_pred = array![1.5_f64, 2.5, 2.0, 4.0];
    let got = d2_tweedie_score(&y_true, &y_pred, -1.5).unwrap();
    assert!(
        (got - SK_D2_TWEEDIE_NEG_YTRUE).abs() < 1e-10,
        "d2_tweedie_score(power=-1.5) negative y_true: sklearn={SK_D2_TWEEDIE_NEG_YTRUE}, ferrolearn={got}"
    );
}
