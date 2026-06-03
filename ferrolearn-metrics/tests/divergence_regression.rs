//! Divergence pins for `ferrolearn-metrics/src/regression.rs` vs scikit-learn 1.5.2.
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (the `python3 -c`
//! call is quoted in each test, R-CHAR-3) — NEVER copied from the ferrolearn side.
//!
//! RED pins (deterministic value/edge divergences — fixers must fix this iter):
//!   - divergence_r2_force_finite_constant_imperfect   (#767)
//!   - divergence_r2_force_finite_constant_perfect      (#767)
//!   - divergence_explained_variance_force_finite       (#768)
//!   - divergence_mape_eps_clamp_zero_true              (#769)
//!   - divergence_mape_eps_clamp_all_zero_true          (#769)
//!   - divergence_d2_absolute_error_single_sample_nan   (#770)
//!   - divergence_d2_pinball_single_sample_nan          (#770)
//!   - divergence_d2_tweedie_single_sample_nan          (#770)
//!
//! GREEN guards (oracle-grounded, must pass now — guard the correct 1D formulas):
//!   - green_mse_basic
//!   - green_mae_basic
//!   - green_r2_nondegenerate
//!   - green_median_absolute_error_basic
//!   - green_max_error_basic
//!   - green_mean_poisson_deviance_basic
//!   - green_mean_gamma_deviance_basic
//!   - green_d2_tweedie_power0_basic
//!   - green_mape_basic_nonzero

use ferrolearn_metrics::regression::{
    d2_absolute_error_score, d2_pinball_score, d2_tweedie_score, explained_variance_score,
    max_error, mean_absolute_error, mean_absolute_percentage_error, mean_gamma_deviance,
    mean_poisson_deviance, mean_squared_error, median_absolute_error, r2_score,
};
use ndarray::array;

// ===========================================================================
// RED pins — deterministic value/edge divergences
// ===========================================================================

/// Divergence: `r2_score` returns `Err(NumericalInstability)` on constant
/// `y_true` (`regression.rs` r2_score `if ss_tot == F::zero()` block) instead
/// of honoring sklearn's default `force_finite=True`, which sets a nonzero-
/// numerator / zero-denominator output to `0.0`
/// (`sklearn/metrics/_regression.py:889-891`
/// "Non-zero Numerator and Zero Denominator: arbitrary set to 0.0").
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "from sklearn.metrics import r2_score; print(r2_score([3,3,3],[3,3,2]))"
///   # 0.0
/// ferrolearn returns Err. Tracking: #767
#[test]
fn divergence_r2_force_finite_constant_imperfect() {
    let y_true = array![3.0_f64, 3.0, 3.0];
    let y_pred = array![3.0_f64, 3.0, 2.0];
    let got = r2_score(&y_true, &y_pred);
    // sklearn 1.5.2 live oracle:
    const SK_R2_CONST_IMPERFECT: f64 = 0.0;
    assert!(
        matches!(&got, Ok(v) if (*v - SK_R2_CONST_IMPERFECT).abs() < 1e-12),
        "r2_score constant-imperfect: sklearn={SK_R2_CONST_IMPERFECT}, ferrolearn={got:?}"
    );
}

/// Divergence: `r2_score` errors on constant `y_true` even when predictions are
/// perfect; sklearn's `force_finite=True` returns `1.0` for a zero numerator
/// (`sklearn/metrics/_regression.py:879-881`
/// "Default = Zero Numerator = perfect predictions. Set to 1.0").
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "from sklearn.metrics import r2_score; print(r2_score([3,3,3],[3,3,3]))"
///   # 1.0
/// ferrolearn returns Err. Tracking: #767
#[test]
fn divergence_r2_force_finite_constant_perfect() {
    let y_true = array![3.0_f64, 3.0, 3.0];
    let y_pred = array![3.0_f64, 3.0, 3.0];
    let got = r2_score(&y_true, &y_pred);
    // sklearn 1.5.2 live oracle:
    const SK_R2_CONST_PERFECT: f64 = 1.0;
    assert!(
        matches!(&got, Ok(v) if (*v - SK_R2_CONST_PERFECT).abs() < 1e-12),
        "r2_score constant-perfect: sklearn={SK_R2_CONST_PERFECT}, ferrolearn={got:?}"
    );
}

/// Divergence: `explained_variance_score` returns `Err(NumericalInstability)`
/// on constant `y_true` (`regression.rs` explained_variance_score
/// `if var_true == F::zero()` block) instead of sklearn's default
/// `force_finite=True` → `0.0` for nonzero-numerator / zero-denominator
/// (`sklearn/metrics/_regression.py:889-891`).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "from sklearn.metrics import explained_variance_score as e; print(e([5,5,5],[1,2,3]))"
///   # 0.0
/// ferrolearn returns Err. Tracking: #768
#[test]
fn divergence_explained_variance_force_finite() {
    let y_true = array![5.0_f64, 5.0, 5.0];
    let y_pred = array![1.0_f64, 2.0, 3.0];
    let got = explained_variance_score(&y_true, &y_pred);
    // sklearn 1.5.2 live oracle:
    const SK_EVS_CONST: f64 = 0.0;
    assert!(
        matches!(&got, Ok(v) if (*v - SK_EVS_CONST).abs() < 1e-12),
        "explained_variance_score constant: sklearn={SK_EVS_CONST}, ferrolearn={got:?}"
    );
}

/// Divergence: `mean_absolute_percentage_error` SKIPS samples where
/// `y_true == 0` (`regression.rs` mape `if t != F::zero()` branch). sklearn
/// instead divides by `np.maximum(np.abs(y_true), eps)` with
/// `eps = np.finfo(np.float64).eps` (`sklearn/metrics/_regression.py:403-404`),
/// so a zero-`y_true` sample contributes a huge finite term — never skipped.
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "from sklearn.metrics import mean_absolute_percentage_error as m; print(repr(float(m([100,0,200],[110,999,200]))))"
///   # 1.4996986759143752e+18
/// ferrolearn returns 0.05 (skips index 1). Tracking: #769
#[test]
fn divergence_mape_eps_clamp_zero_true() {
    let y_true = array![100.0_f64, 0.0, 200.0];
    let y_pred = array![110.0_f64, 999.0, 200.0];
    let got = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
    // sklearn 1.5.2 live oracle:
    const SK_MAPE_ZERO_TRUE: f64 = 1.499_698_675_914_375_2e18;
    let rel = (got - SK_MAPE_ZERO_TRUE).abs() / SK_MAPE_ZERO_TRUE.abs();
    assert!(
        rel < 1e-9,
        "mape eps-clamp zero-true: sklearn={SK_MAPE_ZERO_TRUE:e}, ferrolearn={got:e}"
    );
}

/// Divergence: `mean_absolute_percentage_error` returns `+inf` when ALL
/// `y_true == 0` (`regression.rs` mape `if count == 0 { return Ok(infinity) }`).
/// sklearn divides each by `eps`, yielding a huge FINITE value
/// (`sklearn/metrics/_regression.py:403-404`).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "from sklearn.metrics import mean_absolute_percentage_error as m; print(repr(float(m([0,0],[1,2]))))"
///   # 6755399441055744.0
/// ferrolearn returns +inf. Tracking: #769
#[test]
fn divergence_mape_eps_clamp_all_zero_true() {
    let y_true = array![0.0_f64, 0.0];
    let y_pred = array![1.0_f64, 2.0];
    let got = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
    // sklearn 1.5.2 live oracle:
    const SK_MAPE_ALL_ZERO: f64 = 6_755_399_441_055_744.0;
    assert!(
        got.is_finite() && (got - SK_MAPE_ALL_ZERO).abs() / SK_MAPE_ALL_ZERO.abs() < 1e-9,
        "mape eps-clamp all-zero-true: sklearn={SK_MAPE_ALL_ZERO:e}, ferrolearn={got:e}"
    );
}

/// Divergence: `d2_absolute_error_score` returns `Ok(0.0)` for a single sample
/// (`regression.rs` `d2_score_with` `if den == F::zero() { return Ok(zero) }`).
/// sklearn warns `UndefinedMetricWarning` and returns `nan` for `< 2` samples
/// (`sklearn/metrics/_regression.py:1699-1702`, reached via
/// `d2_absolute_error_score` → `d2_pinball_score`).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "from sklearn.metrics import d2_absolute_error_score as d; print(d([1.0],[1.0]))"
///   # nan
/// ferrolearn returns Ok(0.0). Tracking: #770
#[test]
fn divergence_d2_absolute_error_single_sample_nan() {
    let y_true = array![1.0_f64];
    let y_pred = array![1.0_f64];
    let got = d2_absolute_error_score(&y_true, &y_pred);
    // sklearn 1.5.2 live oracle: nan for n_samples < 2.
    assert!(
        matches!(&got, Ok(v) if v.is_nan()),
        "d2_absolute_error_score single-sample: sklearn=nan, ferrolearn={got:?}"
    );
}

/// Divergence: `d2_pinball_score` returns `Ok(0.0)` for a single sample
/// (`regression.rs` `d2_score_with` zero-denominator branch). sklearn returns
/// `nan` for `< 2` samples (`sklearn/metrics/_regression.py:1699-1702`).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "from sklearn.metrics import d2_pinball_score as d; print(d([1.0],[1.0]))"
///   # nan
/// ferrolearn returns Ok(0.0). Tracking: #770
#[test]
fn divergence_d2_pinball_single_sample_nan() {
    let y_true = array![1.0_f64];
    let y_pred = array![1.0_f64];
    let got = d2_pinball_score(&y_true, &y_pred, 0.5);
    // sklearn 1.5.2 live oracle: nan for n_samples < 2.
    assert!(
        matches!(&got, Ok(v) if v.is_nan()),
        "d2_pinball_score single-sample: sklearn=nan, ferrolearn={got:?}"
    );
}

/// Divergence: `d2_tweedie_score` returns `Ok(0.0)` for a single sample
/// (`regression.rs` d2_tweedie_score `if den == F::zero() { return Ok(zero) }`).
/// sklearn returns `nan` for `< 2` samples
/// (`sklearn/metrics/_regression.py:1584-1587`).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "from sklearn.metrics import d2_tweedie_score as d; print(d([1.0],[1.0]))"
///   # nan
/// ferrolearn returns Ok(0.0). Tracking: #770
#[test]
fn divergence_d2_tweedie_single_sample_nan() {
    let y_true = array![1.0_f64];
    let y_pred = array![1.0_f64];
    let got = d2_tweedie_score(&y_true, &y_pred, 0.0);
    // sklearn 1.5.2 live oracle: nan for n_samples < 2.
    assert!(
        matches!(&got, Ok(v) if v.is_nan()),
        "d2_tweedie_score single-sample: sklearn=nan, ferrolearn={got:?}"
    );
}

// ===========================================================================
// GREEN guards — oracle-grounded; must pass now (protect the correct 1D paths)
// ===========================================================================

/// Guard: `mean_squared_error` basic 1D value matches sklearn.
/// Oracle:
///   python3 -c "from sklearn.metrics import mean_squared_error as m; print(repr(float(m([1,2,3],[1,2,4]))))"
///   # 0.3333333333333333
#[test]
fn green_mse_basic() {
    let y_true = array![1.0_f64, 2.0, 3.0];
    let y_pred = array![1.0_f64, 2.0, 4.0];
    let got = mean_squared_error(&y_true, &y_pred).unwrap();
    const SK_MSE: f64 = 0.333_333_333_333_333_3;
    assert!(
        (got - SK_MSE).abs() < 1e-12,
        "mse basic: sklearn={SK_MSE}, ferrolearn={got}"
    );
}

/// Guard: `mean_absolute_error` basic 1D value matches sklearn.
/// Oracle:
///   python3 -c "from sklearn.metrics import mean_absolute_error as m; print(repr(float(m([1,2,3],[1,2,4]))))"
///   # 0.3333333333333333
#[test]
fn green_mae_basic() {
    let y_true = array![1.0_f64, 2.0, 3.0];
    let y_pred = array![1.0_f64, 2.0, 4.0];
    let got = mean_absolute_error(&y_true, &y_pred).unwrap();
    const SK_MAE: f64 = 0.333_333_333_333_333_3;
    assert!(
        (got - SK_MAE).abs() < 1e-12,
        "mae basic: sklearn={SK_MAE}, ferrolearn={got}"
    );
}

/// Guard: `r2_score` non-degenerate value matches sklearn.
/// Oracle:
///   python3 -c "from sklearn.metrics import r2_score; print(repr(float(r2_score([1,2,3],[1.1,1.9,3.2]))))"
///   # 0.97
#[test]
fn green_r2_nondegenerate() {
    let y_true = array![1.0_f64, 2.0, 3.0];
    let y_pred = array![1.1_f64, 1.9, 3.2];
    let got = r2_score(&y_true, &y_pred).unwrap();
    const SK_R2: f64 = 0.97;
    assert!(
        (got - SK_R2).abs() < 1e-12,
        "r2 non-degenerate: sklearn={SK_R2}, ferrolearn={got}"
    );
}

/// Guard: `median_absolute_error` basic 1D value matches sklearn.
/// Oracle:
///   python3 -c "from sklearn.metrics import median_absolute_error as m; print(repr(float(m([1,2,3,100],[1,2,3,3]))))"
///   # 0.0
#[test]
fn green_median_absolute_error_basic() {
    let y_true = array![1.0_f64, 2.0, 3.0, 100.0];
    let y_pred = array![1.0_f64, 2.0, 3.0, 3.0];
    let got = median_absolute_error(&y_true, &y_pred).unwrap();
    const SK_MED: f64 = 0.0;
    assert!(
        (got - SK_MED).abs() < 1e-12,
        "median_absolute_error basic: sklearn={SK_MED}, ferrolearn={got}"
    );
}

/// Guard: `max_error` basic 1D value matches sklearn.
/// Oracle:
///   python3 -c "from sklearn.metrics import max_error; print(repr(float(max_error([1,2,3],[1.5,2,5]))))"
///   # 2.0
#[test]
fn green_max_error_basic() {
    let y_true = array![1.0_f64, 2.0, 3.0];
    let y_pred = array![1.5_f64, 2.0, 5.0];
    let got = max_error(&y_true, &y_pred).unwrap();
    const SK_MAX: f64 = 2.0;
    assert!(
        (got - SK_MAX).abs() < 1e-12,
        "max_error basic: sklearn={SK_MAX}, ferrolearn={got}"
    );
}

/// Guard: `mean_poisson_deviance` basic 1D value matches sklearn.
/// Oracle:
///   python3 -c "from sklearn.metrics import mean_poisson_deviance as p; print(repr(float(p([1,2],[2,1]))))"
///   # 0.6931471805599452  (== 2*ln2)
#[test]
fn green_mean_poisson_deviance_basic() {
    let y_true = array![1.0_f64, 2.0];
    let y_pred = array![2.0_f64, 1.0];
    let got = mean_poisson_deviance(&y_true, &y_pred).unwrap();
    const SK_MPD: f64 = 0.693_147_180_559_945_2;
    assert!(
        (got - SK_MPD).abs() < 1e-12,
        "mean_poisson_deviance basic: sklearn={SK_MPD}, ferrolearn={got}"
    );
}

/// Guard: `mean_gamma_deviance` basic 1D value matches sklearn.
/// Oracle:
///   python3 -c "from sklearn.metrics import mean_gamma_deviance as g; print(repr(float(g([1,2],[2,1]))))"
///   # 0.5
#[test]
fn green_mean_gamma_deviance_basic() {
    let y_true = array![1.0_f64, 2.0];
    let y_pred = array![2.0_f64, 1.0];
    let got = mean_gamma_deviance(&y_true, &y_pred).unwrap();
    const SK_MGD: f64 = 0.5;
    assert!(
        (got - SK_MGD).abs() < 1e-12,
        "mean_gamma_deviance basic: sklearn={SK_MGD}, ferrolearn={got}"
    );
}

/// Guard: `d2_tweedie_score(power=0)` basic value matches sklearn (>= 2 valid
/// samples — the correct ratio path the fixer must not regress).
/// Oracle:
///   python3 -c "from sklearn.metrics import d2_tweedie_score as d; print(repr(float(d([1,2,3],[1.5,2.5,2.5],power=0))))"
///   # 0.625
#[test]
fn green_d2_tweedie_power0_basic() {
    let y_true = array![1.0_f64, 2.0, 3.0];
    let y_pred = array![1.5_f64, 2.5, 2.5];
    let got = d2_tweedie_score(&y_true, &y_pred, 0.0).unwrap();
    const SK_D2T: f64 = 0.625;
    assert!(
        (got - SK_D2T).abs() < 1e-12,
        "d2_tweedie_score power=0: sklearn={SK_D2T}, ferrolearn={got}"
    );
}

/// Guard: `mean_absolute_percentage_error` with all-nonzero `y_true` matches
/// sklearn (the eps-clamp fix must not regress the no-zero case).
/// Oracle:
///   python3 -c "from sklearn.metrics import mean_absolute_percentage_error as m; print(repr(float(m([100,200,300],[110,190,300]))))"
///   # 0.05000000000000001
#[test]
fn green_mape_basic_nonzero() {
    let y_true = array![100.0_f64, 200.0, 300.0];
    let y_pred = array![110.0_f64, 190.0, 300.0];
    let got = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
    const SK_MAPE: f64 = 0.050_000_000_000_000_01;
    assert!(
        (got - SK_MAPE).abs() < 1e-12,
        "mape basic nonzero: sklearn={SK_MAPE}, ferrolearn={got}"
    );
}
