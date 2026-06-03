//! Divergence pins for the d2 `n>=2` zero-denominator (constant `y_true`) path
//! in `ferrolearn-metrics/src/regression.rs` vs scikit-learn 1.5.2.
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (R-CHAR-3) —
//! NEVER copied from the ferrolearn side.
//!
//! Re-audit of fixer #770 (which added the `< 2` samples NaN guard). The
//! `n >= 2` constant-`y_true` (zero-denominator) branch was NOT addressed and
//! still diverges:
//!
//!   - `d2_score_with` (d2_absolute_error / d2_pinball) returns `Ok(0.0)`
//!     whenever `den == 0`, IGNORING the numerator. sklearn's force_finite
//!     assembly leaves a ZERO-numerator score at the default `1.0`.
//!   - `d2_tweedie_score` has NO force_finite at all in sklearn; it returns
//!     `1 - num/den` literally, so zero-den yields `nan` (zero num) or
//!     `+/-inf` (nonzero num). ferrolearn returns `Ok(0.0)`.

use ferrolearn_metrics::regression::{d2_absolute_error_score, d2_pinball_score, d2_tweedie_score};
use ndarray::array;

/// Divergence: `d2_score_with` (via `d2_absolute_error_score`) returns
/// `Ok(0.0)` for a >=2-sample constant `y_true` with a PERFECT prediction
/// (zero numerator, zero denominator), because of the unconditional
/// `if den == F::zero() { return Ok(zero) }` branch
/// (`ferrolearn-metrics/src/regression.rs:1054-1056`).
///
/// sklearn's force_finite assembly initialises `output_scores = np.ones(...)`
/// and only overwrites `nonzero_numerator & ~nonzero_denominator` with 0.0
/// (`sklearn/metrics/_regression.py:1736-1739`). A ZERO numerator therefore
/// keeps the default `1.0`.
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import warnings; warnings.filterwarnings('ignore'); \
///     from sklearn.metrics import d2_absolute_error_score as d; \
///     print(float(d([5,5,5],[5,5,5])))"
///   # 1.0
/// ferrolearn returns Ok(0.0). Tracking: #771
#[test]
fn divergence_d2_absolute_error_constant_perfect_is_one() {
    let y_true = array![5.0_f64, 5.0, 5.0];
    let y_pred = array![5.0_f64, 5.0, 5.0];
    let got = d2_absolute_error_score(&y_true, &y_pred);
    // sklearn 1.5.2 live oracle: zero num & zero den -> default 1.0.
    const SK_D2AE_CONST_PERFECT: f64 = 1.0;
    assert!(
        matches!(&got, Ok(v) if (*v - SK_D2AE_CONST_PERFECT).abs() < 1e-12),
        "d2_absolute_error const-perfect: sklearn={SK_D2AE_CONST_PERFECT}, ferrolearn={got:?}"
    );
}

/// Divergence: same `d2_score_with` zero-denominator branch via
/// `d2_pinball_score(alpha=0.5)`: constant `y_true`, perfect prediction
/// (zero num, zero den) -> ferrolearn `Ok(0.0)`, sklearn `1.0`
/// (`sklearn/metrics/_regression.py:1736-1739`).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import warnings; warnings.filterwarnings('ignore'); \
///     from sklearn.metrics import d2_pinball_score as d; \
///     print(float(d([5,5,5],[5,5,5],alpha=0.5)))"
///   # 1.0
/// ferrolearn returns Ok(0.0). Tracking: #771
#[test]
fn divergence_d2_pinball_constant_perfect_is_one() {
    let y_true = array![5.0_f64, 5.0, 5.0];
    let y_pred = array![5.0_f64, 5.0, 5.0];
    let got = d2_pinball_score(&y_true, &y_pred, 0.5);
    const SK_D2P_CONST_PERFECT: f64 = 1.0;
    assert!(
        matches!(&got, Ok(v) if (*v - SK_D2P_CONST_PERFECT).abs() < 1e-12),
        "d2_pinball const-perfect: sklearn={SK_D2P_CONST_PERFECT}, ferrolearn={got:?}"
    );
}

/// Divergence: `d2_tweedie_score` returns `Ok(0.0)` for a >=2-sample constant
/// `y_true` with a PERFECT prediction (zero den)
/// (`ferrolearn-metrics/src/regression.rs:1212-1214`). sklearn applies NO
/// force_finite and returns `1 - num/den` literally
/// (`sklearn/metrics/_regression.py:1599`); zero/zero yields `nan`.
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import warnings; warnings.filterwarnings('ignore'); \
///     from sklearn.metrics import d2_tweedie_score as d; \
///     print(float(d([5,5,5],[5,5,5],power=0)))"
///   # nan
/// ferrolearn returns Ok(0.0). Tracking: #771
#[test]
fn divergence_d2_tweedie_constant_perfect_is_nan() {
    let y_true = array![5.0_f64, 5.0, 5.0];
    let y_pred = array![5.0_f64, 5.0, 5.0];
    let got = d2_tweedie_score(&y_true, &y_pred, 0.0);
    // sklearn 1.5.2 live oracle: 1 - 0/0 -> nan (no force_finite).
    assert!(
        matches!(&got, Ok(v) if v.is_nan()),
        "d2_tweedie const-perfect: sklearn=nan, ferrolearn={got:?}"
    );
}

/// Divergence: `d2_tweedie_score` constant `y_true`, IMPERFECT prediction
/// (nonzero num, zero den) -> ferrolearn `Ok(0.0)`; sklearn `1 - x/0` = `-inf`
/// (`sklearn/metrics/_regression.py:1599`).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import warnings; warnings.filterwarnings('ignore'); \
///     from sklearn.metrics import d2_tweedie_score as d; \
///     print(float(d([5,5,5],[5,5,6],power=0)))"
///   # -inf
/// ferrolearn returns Ok(0.0). Tracking: #771
#[test]
fn divergence_d2_tweedie_constant_imperfect_is_neg_inf() {
    let y_true = array![5.0_f64, 5.0, 5.0];
    let y_pred = array![5.0_f64, 5.0, 6.0];
    let got = d2_tweedie_score(&y_true, &y_pred, 0.0);
    // sklearn 1.5.2 live oracle: 1 - (positive)/0 -> -inf (no force_finite).
    assert!(
        matches!(&got, Ok(v) if v.is_infinite() && *v < 0.0),
        "d2_tweedie const-imperfect: sklearn=-inf, ferrolearn={got:?}"
    );
}
