//! Divergence tests: ferrolearn `StandardScaler` vs scikit-learn 1.5.2
//! `sklearn/preprocessing/_data.py` `class StandardScaler` (`:696`).
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (run from /tmp)
//! or a sklearn `file:line` symbolic constant — never copied from the
//! ferrolearn side (R-CHAR-3).
//!
//! Two flavors here:
//!   * GREEN guards — pin the SHIPPED, in-regime (non-constant, finite,
//!     default-param) REQ-1/REQ-3 behavior; these PASS today and lock parity.
//!   * RED pins — the constant / zero-variance column divergence (REQ-2); these
//!     FAIL today and must be flipped by the generator (tracking blocker
//!     #<filed below>).

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::StandardScaler;
use ndarray::array;

// ===========================================================================
// RED PINS — constant / zero-variance column divergence (the fixable one)
// ===========================================================================

/// Divergence: ferrolearn's `FittedStandardScaler::transform` diverges from
/// `sklearn/preprocessing/_data.py:1066-1067` for a constant (zero-variance)
/// column.
///
/// sklearn: for a column with `var_ == 0`, `_is_constant_feature` is true
/// (`_data.py:72-85`) so `scale_ = _handle_zeros_in_scale(sqrt(var_)) = 1`
/// (`_data.py:88-120`, applied at `:1019-1021`). With default `with_mean=True`
/// `transform` does `X -= mean_` (`:1064-1065`) then `X /= scale_=1`
/// (`:1066-1067`), so each entry `(x - mean)/1 = 0` (since `x == mean`).
/// Live oracle (run from /tmp, sklearn 1.5.2):
///   `StandardScaler().fit_transform([[1.,5.],[2.,5.],[3.,5.]])`
///   -> `[[-1.224744871391589, 0.0], [0.0, 0.0], [1.224744871391589, 0.0]]`
///   (constant col 1 -> 0.0, NOT 5.0).
/// ferrolearn: `if s == F::zero() { continue }` leaves the column UNCHANGED,
/// returning the original 5.0 (`standard_scaler.rs:188-191`).
#[test]
fn divergence_constant_column_maps_to_zero() {
    // Live sklearn 1.5.2 oracle: constant col 1 standardizes to 0.0.
    const SK_CONST_COL: f64 = 0.0;

    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
    let scaled = scaler.fit_transform(&x).unwrap();

    // Constant col 1 must map to 0.0 (sklearn), not the original 5.0.
    for i in 0..3 {
        assert!(
            (scaled[[i, 1]] - SK_CONST_COL).abs() < 1e-12,
            "constant col 1 row {i}: ferrolearn={}, sklearn oracle={SK_CONST_COL}",
            scaled[[i, 1]]
        );
    }
}

/// Divergence: ferrolearn's `FittedStandardScaler::transform` diverges from
/// `sklearn/preprocessing/_data.py:1066-1067` for a single-row fit, where every
/// column is constant (n=1 => var_=0 for all columns).
///
/// Live oracle (run from /tmp, sklearn 1.5.2):
///   `StandardScaler().fit_transform([[5.,7.]])` -> `[[0.0, 0.0]]`
///   (every column constant -> scale_=1, x-mean=0 -> 0.0).
/// ferrolearn: every column has std=0 so each is left UNCHANGED, returning the
/// original `[[5.0, 7.0]]`.
#[test]
fn divergence_single_row_fit_all_columns_zero() {
    // Live sklearn 1.5.2 oracle: single-row fit_transform -> all zeros.
    const SK_VAL: f64 = 0.0;

    let scaler = StandardScaler::<f64>::new();
    let x = array![[5.0, 7.0]];
    let scaled = scaler.fit_transform(&x).unwrap();

    assert!(
        (scaled[[0, 0]] - SK_VAL).abs() < 1e-12,
        "single-row col 0: ferrolearn={}, sklearn oracle={SK_VAL}",
        scaled[[0, 0]]
    );
    assert!(
        (scaled[[0, 1]] - SK_VAL).abs() < 1e-12,
        "single-row col 1: ferrolearn={}, sklearn oracle={SK_VAL}",
        scaled[[0, 1]]
    );
}

// ===========================================================================
// GREEN GUARDS — SHIPPED REQ-1 / REQ-3 behavior (pass today, lock parity)
// ===========================================================================

/// REQ-1 value match (non-constant columns), oracle-grounded.
/// `sklearn/preprocessing/_data.py:1064-1067` (`X -= mean_; X /= scale_`),
/// population variance (ddof=0) via `_incremental_mean_and_var`.
/// Live oracle (run from /tmp, sklearn 1.5.2):
///   `StandardScaler().fit_transform([[1.,10.],[2.,20.],[3.,30.]])`
///   -> `[[-1.224744871391589, -1.224744871391589], [0.0, 0.0],
///        [1.224744871391589, 1.224744871391589]]`.
#[test]
fn green_req1_value_match_non_constant() {
    // Live sklearn 1.5.2 oracle.
    let sk = [
        [-1.224744871391589, -1.224744871391589],
        [0.0, 0.0],
        [1.224744871391589, 1.224744871391589],
    ];

    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
    let scaled = scaler.fit_transform(&x).unwrap();

    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (scaled[[i, j]] - sk[i][j]).abs() < 1e-12,
                "[{i},{j}]: ferrolearn={}, sklearn={}",
                scaled[[i, j]],
                sk[i][j]
            );
        }
    }
}

/// REQ-1 fitted-attribute match: `mean()` == sklearn `mean_`, `std()` ==
/// sklearn `scale_` on NON-constant columns (`_data.py:1013-1021`).
/// Live oracle (run from /tmp, sklearn 1.5.2) for
/// `[[1.,10.],[2.,20.],[3.,30.]]`:
///   `mean_ = [2.0, 20.0]`, `scale_ = [0.816496580927726, 8.16496580927726]`.
#[test]
fn green_req1_mean_and_scale_attributes() {
    // Live sklearn 1.5.2 oracle.
    let sk_mean = [2.0, 20.0];
    let sk_scale = [0.816496580927726, 8.16496580927726];

    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
    let fitted = scaler.fit(&x, &()).unwrap();

    for j in 0..2 {
        assert!(
            (fitted.mean()[j] - sk_mean[j]).abs() < 1e-12,
            "mean[{j}]: ferrolearn={}, sklearn mean_={}",
            fitted.mean()[j],
            sk_mean[j]
        );
        // ferrolearn std() == sklearn scale_ ONLY on non-constant columns.
        assert!(
            (fitted.std()[j] - sk_scale[j]).abs() < 1e-12,
            "std[{j}]: ferrolearn={}, sklearn scale_={}",
            fitted.std()[j],
            sk_scale[j]
        );
    }
}

/// REQ-1 value match on a non-trivial negative/decimal fixture, oracle-grounded.
/// Live oracle (run from /tmp, sklearn 1.5.2):
///   `StandardScaler().fit_transform([[-2.5,0.1],[4.0,-3.3],[1.5,2.2]])`
///   -> `[[-1.3074028933658497, 0.1912178283973915],
///        [1.1206310514564426, -1.3091066713359882],
///        [0.1867718419094071, 1.1178888429385967]]`.
#[test]
fn green_req1_negative_decimal_fixture() {
    // Live sklearn 1.5.2 oracle.
    let sk = [
        [-1.3074028933658497, 0.1912178283973915],
        [1.1206310514564426, -1.3091066713359882],
        [0.1867718419094071, 1.1178888429385967],
    ];

    let scaler = StandardScaler::<f64>::new();
    let x = array![[-2.5, 0.1], [4.0, -3.3], [1.5, 2.2]];
    let scaled = scaler.fit_transform(&x).unwrap();

    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (scaled[[i, j]] - sk[i][j]).abs() < 1e-12,
                "[{i},{j}]: ferrolearn={}, sklearn={}",
                scaled[[i, j]],
                sk[i][j]
            );
        }
    }
}

/// REQ-3 inverse_transform round-trip, oracle-grounded.
/// `sklearn/preprocessing/_data.py:1106-1109` (`X *= scale_; X += mean_`).
/// Live oracle (run from /tmp, sklearn 1.5.2) for `[[1.,2.],[3.,4.],[5.,6.]]`:
///   `inverse_transform(transform(X))`
///   -> `[[1.0000000000000002, 2.0], [3.0, 4.0], [5.0, 6.0]]` (recovers X).
#[test]
fn green_req3_inverse_roundtrip() {
    // Live sklearn 1.5.2 oracle (recovers X within ULPs).
    let sk = [[1.0000000000000002, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let fitted = scaler.fit(&x, &()).unwrap();
    let scaled = fitted.transform(&x).unwrap();
    let recovered = fitted.inverse_transform(&scaled).unwrap();

    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (recovered[[i, j]] - sk[i][j]).abs() < 1e-12,
                "[{i},{j}]: ferrolearn={}, sklearn={}",
                recovered[[i, j]],
                sk[i][j]
            );
        }
    }
}

/// REQ-3 inverse_transform of a constant column recovers the constant (mean).
/// Consistency guard for the REQ-2 fix: after the forward fix maps a constant
/// column to 0, inverse must recover the constant 5.0.
/// sklearn: `X *= scale_=1; X += mean_=5` -> 5 (`_data.py:1106-1109`).
/// ferrolearn (raw std=0): `0*0 + 5 = 5` -> ALSO recovers 5.
/// Live oracle (run from /tmp, sklearn 1.5.2):
///   `mc.inverse_transform(mc.transform([[1.,5.],[2.,5.],[3.,5.]]))`
///   -> `[[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]` (col 1 -> 5.0).
#[test]
fn green_req3_inverse_constant_column_recovers_mean() {
    // Live sklearn 1.5.2 oracle: constant col 1 recovers 5.0.
    const SK_CONST: f64 = 5.0;

    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
    let fitted = scaler.fit(&x, &()).unwrap();
    let scaled = fitted.transform(&x).unwrap();
    let recovered = fitted.inverse_transform(&scaled).unwrap();

    for i in 0..3 {
        assert!(
            (recovered[[i, 1]] - SK_CONST).abs() < 1e-12,
            "constant col 1 row {i}: ferrolearn={}, sklearn={SK_CONST}",
            recovered[[i, 1]]
        );
    }
}

/// REQ-2 mixed fixture: one constant + one non-constant column verified
/// simultaneously in a single fit_transform (re-audit of blocker #1191 fix).
/// Live oracle (run from /tmp, sklearn 1.5.2):
///   `StandardScaler().fit_transform([[1.,5.],[2.,5.],[3.,5.]])`
///   -> `[[-1.224744871391589, 0.0], [0.0, 0.0], [1.224744871391589, 0.0]]`
///   (non-constant col 0 standardized, constant col 1 -> 0.0, BOTH at once).
#[test]
fn green_req2_mixed_constant_and_nonconstant_columns() {
    // Live sklearn 1.5.2 oracle.
    let sk = [
        [-1.224744871391589, 0.0],
        [0.0, 0.0],
        [1.224744871391589, 0.0],
    ];

    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
    let scaled = scaler.fit_transform(&x).unwrap();

    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (scaled[[i, j]] - sk[i][j]).abs() < 1e-12,
                "[{i},{j}]: ferrolearn={}, sklearn={}",
                scaled[[i, j]],
                sk[i][j]
            );
        }
    }
}

/// REQ-3 inverse alignment on a NON-round-trip input over a constant column —
/// the discriminating guard for the blocker #1191 fix. Unlike the round-trip
/// guard (`green_req3_inverse_constant_column_recovers_mean`), this feeds NON-zero
/// scaled values into a constant column, which the raw-std=0 pre-fix path and the
/// aligned scale_eff=1 post-fix path map DIFFERENTLY: pre-fix raw std=0 gives
/// `x*0 + 5 == 5` for all rows (collapses to mean); post-fix scale_eff=1 (sklearn)
/// gives `x*1 + 5 == x + 5`. sklearn aligns inverse to `scale_=1` on constant cols
/// (`_data.py:1106-1109`, `_handle_zeros_in_scale` `:88-120`, `:1019-1021`).
/// Live oracle (run from /tmp, sklearn 1.5.2):
/// `m = StandardScaler().fit([[1.,5.],[2.,5.],[3.,5.]])` (col 1 constant, scale_=1,
/// mean_=5); `m.inverse_transform([[0.,2.],[0.,-1.],[0.,0.5]])` col 1 -> `[7.0, 4.0, 5.5]`
/// (= x*1 + 5), NOT `[5.0, 5.0, 5.0]`.
#[test]
fn green_req3_inverse_constant_column_non_roundtrip_alignment() {
    // Live sklearn 1.5.2 oracle: col 1 = x*scale_(1) + mean_(5) = x + 5.
    let sk_col1 = [7.0, 4.0, 5.5];

    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
    let fitted = scaler.fit(&x, &()).unwrap();

    // Non-round-trip scaled input: col 1 carries non-zero values.
    let scaled_in = array![[0.0, 2.0], [0.0, -1.0], [0.0, 0.5]];
    let recovered = fitted.inverse_transform(&scaled_in).unwrap();

    for i in 0..3 {
        assert!(
            (recovered[[i, 1]] - sk_col1[i]).abs() < 1e-12,
            "constant col 1 row {i}: ferrolearn={}, sklearn={}",
            recovered[[i, 1]],
            sk_col1[i]
        );
    }
}
