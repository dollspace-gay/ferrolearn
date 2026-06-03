//! Divergence audit: ferrolearn `MaxAbsScaler` vs scikit-learn 1.5.2
//! `sklearn/preprocessing/_data.py` `class MaxAbsScaler` (`:1116`).
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (run from /tmp)
//! or a sklearn `file:line` symbolic constant — NEVER copied from the
//! ferrolearn side (R-CHAR-3).
//!
//! VERDICT (verify-and-document): the implemented dense path MATCHES sklearn,
//! INCLUDING the zero-`max_abs` edge case. Unlike MinMaxScaler/StandardScaler,
//! a `max_abs == 0` column is necessarily all-zero, so sklearn's
//! `scale_ = _handle_zeros_in_scale(0) = 1` (`:1272`,`:88`) → `X / 1 = X`
//! coincides with ferrolearn's "leave unchanged" branch for ANY input.
//! These are GREEN guards (PASS today, lock parity), not RED divergence pins.

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::MaxAbsScaler;
use ndarray::array;

// ===========================================================================
// REQ-1 — per-column max-abs value match (GREEN guards)
// ===========================================================================

/// GREEN: `MaxAbsScaler::fit_transform` mirrors sklearn `X /= scale_`
/// (`sklearn/preprocessing/_data.py:1305`) with
/// `max_abs_ = _nanmax(abs(X), axis=0)` (`:1263`) for a mixed-sign fixture.
///
/// Live oracle (sklearn 1.5.2, from /tmp):
/// `MaxAbsScaler().fit_transform([[-3.,1.],[0.,-2.],[2.,4.]])`
/// -> `[[-1.0, 0.25], [0.0, -0.5], [0.6666666666666666, 1.0]]`
/// and `m.max_abs_ == [3.0, 4.0]`.
#[test]
fn green_req1_mixed_sign_value_match() {
    // sklearn oracle constants (file:line _data.py:1263,1305):
    let sk = array![[-1.0, 0.25], [0.0, -0.5], [0.666_666_666_666_666_6, 1.0]];
    let sk_max_abs = [3.0_f64, 4.0];

    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]];

    let fitted = scaler.fit(&x, &()).unwrap();
    assert!((fitted.max_abs()[0] - sk_max_abs[0]).abs() < 1e-12);
    assert!((fitted.max_abs()[1] - sk_max_abs[1]).abs() < 1e-12);

    let scaled = scaler.fit_transform(&x).unwrap();
    for (a, b) in scaled.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-12, "got {a}, sklearn {b}");
    }
}

/// GREEN: all-negative column. Live oracle:
/// `MaxAbsScaler().fit_transform([[-5.],[-3.],[-1.]])`
/// -> `[[-1.0], [-0.6], [-0.2]]` (max_abs=5, `_data.py:1263`).
#[test]
fn green_req1b_all_negative_value_match() {
    let sk = array![[-1.0], [-0.6], [-0.2]];
    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[-5.0], [-3.0], [-1.0]];
    let scaled = scaler.fit_transform(&x).unwrap();
    for (a, b) in scaled.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-12, "got {a}, sklearn {b}");
    }
}

/// GREEN: non-trivial mixed fixture (3 features, varying magnitudes).
/// Live oracle:
/// `MaxAbsScaler().fit_transform([[10.,-2.,0.5],[-4.,8.,-1.0],[2.,-16.,0.25]])`
/// -> `[[1.0,-0.125,0.5],[-0.4,0.5,-1.0],[0.2,-1.0,0.25]]`;
/// `max_abs_ == [10.0, 16.0, 1.0]`.
#[test]
fn green_req1_nontrivial_mixed_fixture() {
    let sk = array![[1.0, -0.125, 0.5], [-0.4, 0.5, -1.0], [0.2, -1.0, 0.25],];
    let sk_max_abs = [10.0_f64, 16.0, 1.0];
    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[10.0, -2.0, 0.5], [-4.0, 8.0, -1.0], [2.0, -16.0, 0.25]];
    let fitted = scaler.fit(&x, &()).unwrap();
    for (m, sk) in fitted.max_abs().iter().zip(sk_max_abs.iter()) {
        assert!((m - sk).abs() < 1e-12);
    }
    let scaled = scaler.fit_transform(&x).unwrap();
    for (a, b) in scaled.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-12, "got {a}, sklearn {b}");
    }
}

/// GREEN: f32 path mirrors sklearn float32 dtype.
/// Live oracle:
/// `MaxAbsScaler().fit_transform(float32([[2.,-4.],[1.,3.]]))`
/// -> `[[1.0, -1.0], [0.5, 0.75]]`.
#[test]
fn green_req1_f32_value_match() {
    let sk = array![[1.0_f32, -1.0], [0.5, 0.75]];
    let scaler = MaxAbsScaler::<f32>::new();
    let x = array![[2.0_f32, -4.0], [1.0, 3.0]];
    let scaled = scaler.fit_transform(&x).unwrap();
    for (a, b) in scaled.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-6, "got {a}, sklearn {b}");
    }
}

// ===========================================================================
// REQ-1 — error contracts
// ===========================================================================

/// GREEN: `fit` on zero rows errors (sklearn would also reject empty fit;
/// ferrolearn `InsufficientSamples`).
#[test]
fn green_req1_zero_rows_errors() {
    use ndarray::Array2;
    let scaler = MaxAbsScaler::<f64>::new();
    let x: Array2<f64> = Array2::zeros((0, 3));
    assert!(scaler.fit(&x, &()).is_err());
}

/// GREEN: column-count mismatch on `transform` errors (ShapeMismatch).
#[test]
fn green_req1_shape_mismatch_errors() {
    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let fitted = scaler.fit(&x, &()).unwrap();
    let bad = array![[1.0, 2.0, 3.0]];
    assert!(fitted.transform(&bad).is_err());
}

// ===========================================================================
// REQ-2 — zero-max_abs column MATCHES sklearn (GREEN guards, NOT pins)
// ===========================================================================

/// GREEN (MATCH, not a divergence): all-zero column.
/// Live oracle:
/// `MaxAbsScaler().fit_transform([[0.,1.],[0.,2.],[0.,3.]])`
/// col 0 -> `[0.0, 0.0, 0.0]` (sklearn `scale_ = _handle_zeros_in_scale(0) = 1`,
/// `_data.py:1272`,`:88`; `0 / 1 = 0`). ferrolearn `continue`s on the column
/// -> also `[0,0,0]`. MATCH.
#[test]
fn green_req2_all_zero_column_matches() {
    // Live oracle col 1 too: 1/3, 2/3, 1.
    let sk = array![
        [0.0, 0.333_333_333_333_333_3],
        [0.0, 0.666_666_666_666_666_6],
        [0.0, 1.0],
    ];
    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]];
    let scaled = scaler.fit_transform(&x).unwrap();
    for (a, b) in scaled.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-12, "got {a}, sklearn {b}");
    }
}

/// GREEN (the DISCRIMINATING case proving no divergence):
/// fit on an all-zero column, then transform a NON-zero input.
/// Live oracle: `m = MaxAbsScaler().fit([[0.],[0.]]); m.scale_ == [1.0]`;
/// `m.transform([[5.],[-2.]])` -> `[[5.0], [-2.0]]` (`x / scale_(1) = x`,
/// `_data.py:1305`). ferrolearn leaves the `max_abs==0` column unchanged ->
/// also `[[5.0], [-2.0]]`. MATCH — this is where MinMaxScaler/StandardScaler
/// would DIVERGE, and MaxAbsScaler does NOT.
#[test]
fn green_req2_fit_zero_then_transform_nonzero_matches() {
    let sk = array![[5.0], [-2.0]];
    let scaler = MaxAbsScaler::<f64>::new();
    let x_fit = array![[0.0], [0.0]];
    let fitted = scaler.fit(&x_fit, &()).unwrap();
    assert!((fitted.max_abs()[0] - 0.0).abs() < 1e-15); // -> sklearn scale_ = 1
    let x_new = array![[5.0], [-2.0]];
    let scaled = fitted.transform(&x_new).unwrap();
    for (a, b) in scaled.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-15, "got {a}, sklearn {b}");
    }
}

// ===========================================================================
// REQ-3 — inverse_transform round-trip + zero-max_abs inverse (GREEN)
// ===========================================================================

/// GREEN: `inverse_transform(transform(X)) == X` for the REQ-1 fixture.
/// Live oracle:
/// `m=MaxAbsScaler().fit([[-3.,1.],[0.,-2.],[2.,4.]]);`
/// `m.inverse_transform(m.transform(X))` -> `[[-3.,1.],[0.,-2.],[2.,4.]]`
/// (`X *= scale_`, `_data.py:1337`).
#[test]
fn green_req3_inverse_roundtrip() {
    let sk = array![[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]];
    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]];
    let fitted = scaler.fit(&x, &()).unwrap();
    let scaled = fitted.transform(&x).unwrap();
    let recovered = fitted.inverse_transform(&scaled).unwrap();
    for (a, b) in recovered.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-12, "got {a}, sklearn {b}");
    }
}

/// GREEN (MATCH): inverse of a zero-max_abs fitted column.
/// Live oracle: `m=MaxAbsScaler().fit([[0.],[0.]]);`
/// `m.inverse_transform([[5.],[-2.]])` -> `[[5.0], [-2.0]]`
/// (`x * scale_(1) = x`, `_data.py:1337`). ferrolearn leaves it unchanged.
/// MATCH.
#[test]
fn green_req3_inverse_zero_maxabs_matches() {
    let sk = array![[5.0], [-2.0]];
    let scaler = MaxAbsScaler::<f64>::new();
    let fitted = scaler.fit(&array![[0.0], [0.0]], &()).unwrap();
    let out = fitted.inverse_transform(&array![[5.0], [-2.0]]).unwrap();
    for (a, b) in out.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-15, "got {a}, sklearn {b}");
    }
}
