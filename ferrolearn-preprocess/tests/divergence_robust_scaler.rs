//! Divergence tests: ferrolearn `RobustScaler` vs scikit-learn 1.5.2
//! `sklearn/preprocessing/_data.py` `class RobustScaler` (`:1445`).
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (run from /tmp)
//! or a sklearn `file:line` symbolic constant — never copied from the
//! ferrolearn side (R-CHAR-3).
//!
//! Two flavors here:
//!   * GREEN guards — pin the SHIPPED, in-regime (non-degenerate IQR) REQ-1/REQ-3
//!     behavior; these PASS today and lock parity.
//!   * RED pins — the zero-IQR / constant column divergence (DIV-1); these FAIL
//!     today and must be flipped by the generator (tracking blocker #<N>).

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::RobustScaler;
use ndarray::array;

// ===========================================================================
// RED PINS — zero-IQR / constant column divergence (the fixable one, DIV-1)
// ===========================================================================

/// Divergence: ferrolearn's `FittedRobustScaler::transform` diverges from
/// `sklearn/preprocessing/_data.py:1635,1673,1676` for a constant (zero-IQR)
/// column.
///
/// sklearn: `scale_ = q75 - q25 = 0`, then
/// `_handle_zeros_in_scale(scale_) -> 1` (`_data.py:88`, called at `:1635`), and
/// on transform `X -= center_` (`:1673`) then `X /= scale_` (`:1676`), so a
/// constant column with median 7 yields `(7 - 7) / 1 = 0` for every row.
///
/// ferrolearn (`robust_scaler.rs:182-185`): `if iqr == F::zero() { continue; }`
/// leaves the column COMPLETELY UNCHANGED (still 7.0), never centered.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   RobustScaler().fit_transform([[7.,1.],[7.,2.],[7.,3.]]).tolist()
///   == [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]
/// ferrolearn returns col 0 == [7.0, 7.0, 7.0].
///
/// Tracking: #<N>
#[test]
fn divergence_constant_column_centered_to_zero() {
    let scaler = RobustScaler::<f64>::new();
    let x = array![[7.0, 1.0], [7.0, 2.0], [7.0, 3.0]];
    let out = scaler.fit_transform(&x).unwrap();

    // LIVE sklearn 1.5.2 oracle (run from /tmp):
    //   RobustScaler().fit_transform([[7.,1.],[7.,2.],[7.,3.]]).tolist()
    let expected = array![[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]];
    for ((i, j), &e) in expected.indexed_iter() {
        assert!(
            (out[[i, j]] - e).abs() <= 1e-12,
            "constant zero-IQR column: at [{i},{j}] sklearn={e}, ferrolearn={}",
            out[[i, j]]
        );
    }
}

/// Divergence: ferrolearn's `FittedRobustScaler::transform` diverges from
/// `sklearn/preprocessing/_data.py:1635,1673,1676` for a NON-constant column
/// whose IQR happens to be zero (q25 == q75 but values differ).
///
/// For `[1,1,1,1,9]`: np.nanpercentile gives q25 = q50 = q75 = 1, so
/// `scale_ = 0 -> _handle_zeros_in_scale -> 1` (`:1635`), `center_ = 1`, and
/// transform yields `(x - 1) / 1 = x - 1`.
///
/// ferrolearn leaves the whole column unchanged because `iqr == 0` triggers the
/// `continue` at `robust_scaler.rs:182-185`.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   RobustScaler().fit_transform([[1.],[1.],[1.],[1.],[9.]]).tolist()
///   == [[0.0], [0.0], [0.0], [0.0], [8.0]]
/// ferrolearn returns [[1.0], [1.0], [1.0], [1.0], [9.0]] (unchanged).
///
/// Tracking: #<N>
#[test]
fn divergence_nonconstant_zero_iqr_centered() {
    let scaler = RobustScaler::<f64>::new();
    let x = array![[1.0], [1.0], [1.0], [1.0], [9.0]];
    let out = scaler.fit_transform(&x).unwrap();

    // LIVE sklearn 1.5.2 oracle (run from /tmp):
    //   RobustScaler().fit_transform([[1.],[1.],[1.],[1.],[9.]]).tolist()
    let expected = array![[0.0], [0.0], [0.0], [0.0], [8.0]];
    for ((i, j), &e) in expected.indexed_iter() {
        assert!(
            (out[[i, j]] - e).abs() <= 1e-12,
            "non-constant zero-IQR column: at [{i},{j}] sklearn={e}, ferrolearn={}",
            out[[i, j]]
        );
    }
}

// ===========================================================================
// GREEN GUARDS — SHIPPED REQ-1 value match + quantile interp (PASS today)
// ===========================================================================

/// REQ-1 value match (GREEN). Pins the in-regime (non-degenerate IQR) path:
/// `center_` (median) and `scale_` (IQR) and the full transform matrix.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   r = RobustScaler().fit([[1,10],[2,20],[3,30],[100,40]])
///   r.center_.tolist() == [2.5, 25.0]
///   r.scale_.tolist()  == [25.5, 15.0]
///   RobustScaler().fit_transform([[1,10],[2,20],[3,30],[100,40]]).tolist()
///   == [[-0.058823529411764705, -1.0],
///       [-0.0196078431372549,  -0.3333333333333333],
///       [ 0.0196078431372549,   0.3333333333333333],
///       [ 3.823529411764706,    1.0]]
#[test]
fn green_req1_value_match() {
    let scaler = RobustScaler::<f64>::new();
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [100.0, 40.0]];
    let fitted = scaler.fit(&x, &()).unwrap();

    // center_ / scale_ from live oracle.
    assert!((fitted.median()[0] - 2.5).abs() <= 1e-12);
    assert!((fitted.median()[1] - 25.0).abs() <= 1e-12);
    assert!((fitted.iqr()[0] - 25.5).abs() <= 1e-12);
    assert!((fitted.iqr()[1] - 15.0).abs() <= 1e-12);

    let out = scaler.fit_transform(&x).unwrap();
    let expected = array![
        [-0.058_823_529_411_764_705, -1.0],
        [-0.019_607_843_137_254_9, -0.333_333_333_333_333_3],
        [0.019_607_843_137_254_9, 0.333_333_333_333_333_3],
        [3.823_529_411_764_706, 1.0],
    ];
    for ((i, j), &e) in expected.indexed_iter() {
        assert!(
            (out[[i, j]] - e).abs() <= 1e-12,
            "REQ-1 transform: at [{i},{j}] sklearn={e}, ferrolearn={}",
            out[[i, j]]
        );
    }
}

/// REQ-1 quantile interpolation (GREEN). For `[1,2,3,4]`,
/// np.nanpercentile([1,2,3,4], [25,50,75]) == [1.75, 2.5, 3.25], so
/// median == 2.5 and IQR == 3.25 - 1.75 == 1.5 (live oracle confirmed:
/// r.center_ == [2.5], r.scale_ == [1.5]).
#[test]
fn green_req1_quantile_interp() {
    let scaler = RobustScaler::<f64>::new();
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let fitted = scaler.fit(&x, &()).unwrap();
    assert!((fitted.median()[0] - 2.5).abs() <= 1e-12);
    assert!((fitted.iqr()[0] - 1.5).abs() <= 1e-12);
}

/// REQ-3 errors (GREEN). 0-row fit -> Err; column-count mismatch transform -> Err.
#[test]
fn green_req3_errors() {
    use ndarray::Array2;
    let scaler = RobustScaler::<f64>::new();

    // 0-row fit errors.
    let empty: Array2<f64> = Array2::zeros((0, 3));
    assert!(scaler.fit(&empty, &()).is_err());

    // Column-count mismatch on transform errors.
    let x_train = array![[1.0, 2.0], [3.0, 4.0]];
    let fitted = scaler.fit(&x_train, &()).unwrap();
    let x_bad = array![[1.0, 2.0, 3.0]];
    assert!(fitted.transform(&x_bad).is_err());
}
