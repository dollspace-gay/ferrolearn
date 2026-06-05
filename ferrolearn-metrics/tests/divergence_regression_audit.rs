//! ACToR critic divergence audit for `ferrolearn-metrics::regression`.
//!
//! Upstream: `sklearn/metrics/_regression.py` (scikit-learn 1.5.2).
//!
//! Every expected value below was computed FRESH from the live sklearn 1.5.2
//! oracle (R-CHAR-3) via:
//!
//! ```text
//! python3 -c "from sklearn.metrics import *; print(r2_score([3,3,3],[3,3,2]))"
//! ```
//!
//! Oracle results (sklearn 1.5.2, run 2026-06):
//!   r2_score([3,3,3],[3,3,2])                                  -> 0.0
//!   r2_score([3,3,3],[3,3,3])                                  -> 1.0
//!   r2_score([1,2,3],[1.1,2.1,2.9])                            -> 0.985
//!   explained_variance_score([5,5,5],[1,2,3])                  -> 0.0
//!   explained_variance_score([5,5,5],[5,5,5])                  -> 1.0
//!   mean_absolute_percentage_error([100,0,200],[110,999,200])  -> 1.4996986759143752e+18
//!   mean_absolute_error([1,2,3],[1.5,2.0,2.5])                 -> 0.3333333333333333
//!   mean_squared_error([1,2,3],[1,2,4])                        -> 0.3333333333333333
//!   d2_absolute_error_score([1.0],[1.0])                       -> nan
//!
//! These tests call the actual COMPILED ferrolearn public API and assert the
//! live-oracle value. A test that PASSES proves the module-doc SHIPPED claim is
//! true and the older `.design/metrics/regression.md` pin is STALE. A test that
//! FAILS pins a real divergence (named `divergence_*`).

use ferrolearn_metrics::regression::{
    d2_absolute_error_score, explained_variance_score, mean_absolute_error,
    mean_absolute_percentage_error, mean_squared_error, r2_score,
};
use ndarray::array;

// --------------------------------------------------------------------------
// Claim 1: r2_score on constant y_true (force_finite) — _regression.py:866-891
// --------------------------------------------------------------------------

/// sklearn 1.5.2: `r2_score([3,3,3],[3,3,2])` -> 0.0 (force_finite, nonzero num).
#[test]
fn audit_r2_constant_y_true_imperfect_is_zero() {
    let y_true = array![3.0_f64, 3.0, 3.0];
    let y_pred = array![3.0_f64, 3.0, 2.0];
    let got = r2_score(&y_true, &y_pred).expect("r2_score must not Err on constant y_true");
    // live oracle: 0.0
    assert!(got.abs() < 1e-12, "expected sklearn 0.0, got {got}");
}

/// sklearn 1.5.2: `r2_score([3,3,3],[3,3,3])` -> 1.0 (force_finite, zero num).
#[test]
fn audit_r2_constant_y_true_perfect_is_one() {
    let y_true = array![3.0_f64, 3.0, 3.0];
    let y_pred = array![3.0_f64, 3.0, 3.0];
    let got = r2_score(&y_true, &y_pred).expect("r2_score must not Err on constant y_true");
    // live oracle: 1.0
    assert!((got - 1.0).abs() < 1e-12, "expected sklearn 1.0, got {got}");
}

// --------------------------------------------------------------------------
// Claim 2: explained_variance_score on constant y_true — _regression.py:866-891
// --------------------------------------------------------------------------

/// sklearn 1.5.2: `explained_variance_score([5,5,5],[1,2,3])` -> 0.0.
#[test]
fn audit_evs_constant_y_true_imperfect_is_zero() {
    let y_true = array![5.0_f64, 5.0, 5.0];
    let y_pred = array![1.0_f64, 2.0, 3.0];
    let got =
        explained_variance_score(&y_true, &y_pred).expect("evs must not Err on constant y_true");
    // live oracle: 0.0
    assert!(got.abs() < 1e-12, "expected sklearn 0.0, got {got}");
}

/// sklearn 1.5.2: `explained_variance_score([5,5,5],[5,5,5])` -> 1.0.
#[test]
fn audit_evs_constant_y_true_perfect_is_one() {
    let y_true = array![5.0_f64, 5.0, 5.0];
    let y_pred = array![5.0_f64, 5.0, 5.0];
    let got =
        explained_variance_score(&y_true, &y_pred).expect("evs must not Err on constant y_true");
    // live oracle: 1.0
    assert!((got - 1.0).abs() < 1e-12, "expected sklearn 1.0, got {got}");
}

// --------------------------------------------------------------------------
// Claim 3: MAPE eps-clamp (do not skip zero y_true) — _regression.py:403-404
// --------------------------------------------------------------------------

/// sklearn 1.5.2: `mean_absolute_percentage_error([100,0,200],[110,999,200])`
/// -> 1.4996986759143752e+18 (zero y_true divides by eps, NOT skipped).
#[test]
fn audit_mape_zero_y_true_divides_by_eps() {
    let y_true = array![100.0_f64, 0.0, 200.0];
    let y_pred = array![110.0_f64, 999.0, 200.0];
    let got = mean_absolute_percentage_error(&y_true, &y_pred).expect("mape must not Err");
    // live oracle: 1.4996986759143752e+18 (NOT ~0.05, which is the skip-zero bug)
    const SK: f64 = 1.499_698_675_914_375_2e18;
    let rel = (got - SK).abs() / SK.abs();
    assert!(
        got.is_finite() && rel < 1e-9,
        "expected sklearn {SK:e}, got {got:e} (rel {rel:e}); ~0.05 would mean the zero sample was skipped"
    );
}

// --------------------------------------------------------------------------
// Claim 4: d2_absolute_error_score single sample -> NaN — _regression.py:1699-1702
// --------------------------------------------------------------------------

/// sklearn 1.5.2: `d2_absolute_error_score([1.0],[1.0])` -> nan (n<2).
#[test]
fn audit_d2_absolute_error_single_sample_is_nan() {
    let y_true = array![1.0_f64];
    let y_pred = array![1.0_f64];
    let got = d2_absolute_error_score(&y_true, &y_pred)
        .expect("d2_absolute_error_score must not Err on single sample");
    // live oracle: nan
    assert!(got.is_nan(), "expected sklearn NaN, got {got}");
}

// --------------------------------------------------------------------------
// Claim 5: happy-path spot checks
// --------------------------------------------------------------------------

/// sklearn 1.5.2: `mean_absolute_error([1,2,3],[1.5,2.0,2.5])` -> 0.3333333333333333.
#[test]
fn audit_mae_happy_path() {
    let y_true = array![1.0_f64, 2.0, 3.0];
    let y_pred = array![1.5_f64, 2.0, 2.5];
    let got = mean_absolute_error(&y_true, &y_pred).unwrap();
    // live oracle: 0.3333333333333333
    assert!((got - 0.333_333_333_333_333_3).abs() < 1e-12, "got {got}");
}

/// sklearn 1.5.2: `mean_squared_error([1,2,3],[1,2,4])` -> 0.3333333333333333.
#[test]
fn audit_mse_happy_path() {
    let y_true = array![1.0_f64, 2.0, 3.0];
    let y_pred = array![1.0_f64, 2.0, 4.0];
    let got = mean_squared_error(&y_true, &y_pred).unwrap();
    // live oracle: 0.3333333333333333
    assert!((got - 0.333_333_333_333_333_3).abs() < 1e-12, "got {got}");
}

/// sklearn 1.5.2: `r2_score([1,2,3],[1.1,2.1,2.9])` -> 0.985.
#[test]
fn audit_r2_happy_path() {
    let y_true = array![1.0_f64, 2.0, 3.0];
    let y_pred = array![1.1_f64, 2.1, 2.9];
    let got = r2_score(&y_true, &y_pred).unwrap();
    // live oracle: 0.985
    assert!(
        (got - 0.985).abs() < 1e-12,
        "expected sklearn 0.985, got {got}"
    );
}
