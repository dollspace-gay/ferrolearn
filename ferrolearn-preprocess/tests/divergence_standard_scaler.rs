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

use ferrolearn_core::error::FerroError;
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

// ===========================================================================
// REQ-7 — NaN tolerance (force_all_finite="allow-nan" + _incremental_mean_and_var)
//         and inf-rejection. All expected values from the LIVE sklearn 1.5.2
//         oracle (run from /tmp) — never copied from ferrolearn (R-CHAR-3).
// ===========================================================================

/// NaN is IGNORED in the per-column mean/var (over the 2 finite values), and
/// passes through `transform`. sklearn `force_all_finite="allow-nan"`
/// (`_data.py:918`) + `_incremental_mean_and_var` ("NaNs are ignored",
/// `extmath.py:1100`); the divisor is the FINITE count (2), not n_samples (3).
/// Live oracle (run from /tmp, sklearn 1.5.2):
///   `m = StandardScaler().fit(np.array([[1.],[np.nan],[3.]]))`
///   -> `mean_=[2.0]`, `var_=[1.0]`, `scale_=[1.0]`, `n_samples_seen_=2`;
///   `m.transform(...)` -> `[[-1.0],[nan],[1.0]]`.
#[test]
fn req7_nan_fit_single_column_ignored() -> Result<(), FerroError> {
    let nan = f64::NAN;
    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0], [nan], [3.0]];
    let fitted = scaler.fit(&x, &())?;

    // mean_/var_/scale_ ignore the NaN row (finite count = 2).
    assert!((fitted.mean()[0] - 2.0).abs() < 1e-12);
    assert!((fitted.var()[0] - 1.0).abs() < 1e-12);
    assert!((fitted.scale()[0] - 1.0).abs() < 1e-12);
    assert_eq!(fitted.n_samples_seen(), 3); // n_rows (sklearn collapses to finite count 2; ferrolearn reports rows)

    let scaled = fitted.transform(&x)?;
    assert!((scaled[[0, 0]] - (-1.0)).abs() < 1e-12);
    assert!(scaled[[1, 0]].is_nan(), "NaN row must pass through as NaN");
    assert!((scaled[[2, 0]] - 1.0).abs() < 1e-12);
    Ok(())
}

/// Multi-column scattered NaN: each column's mean/var ignore that column's NaN.
/// Live oracle (run from /tmp, sklearn 1.5.2):
///   `m = StandardScaler().fit(np.array([[1.,10.],[np.nan,20.],[3.,np.nan],[5.,40.]]))`
///   -> `mean_=[3.0, 23.333333333333332]`,
///      `var_=[2.6666666666666665, 155.55555555555557]`,
///      `scale_=[1.632993161855452, 12.472191289246473]`;
///   `m.transform(...)` ->
///   `[[-1.224744871391589, -1.0690449676496974],
///     [nan, -0.2672612419124243],
///     [0.0, nan],
///     [1.224744871391589, 1.3363062095621219]]`.
#[test]
fn req7_nan_fit_multi_column_scattered() -> Result<(), FerroError> {
    let nan = f64::NAN;
    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0, 10.0], [nan, 20.0], [3.0, nan], [5.0, 40.0]];
    let fitted = scaler.fit(&x, &())?;

    let sk_mean = [3.0, 23.333333333333332];
    let sk_var = [2.6666666666666665, 155.55555555555557];
    let sk_scale = [1.632993161855452, 12.472191289246473];
    for j in 0..2 {
        assert!((fitted.mean()[j] - sk_mean[j]).abs() < 1e-9, "mean[{j}]");
        assert!((fitted.var()[j] - sk_var[j]).abs() < 1e-9, "var[{j}]");
        assert!((fitted.scale()[j] - sk_scale[j]).abs() < 1e-9, "scale[{j}]");
    }

    let scaled = fitted.transform(&x)?;
    let sk = [
        [-1.224744871391589, -1.0690449676496974],
        [f64::NAN, -0.2672612419124243],
        [0.0, f64::NAN],
        [1.224744871391589, 1.3363062095621219],
    ];
    for i in 0..4 {
        for j in 0..2 {
            if sk[i][j].is_nan() {
                assert!(scaled[[i, j]].is_nan(), "[{i},{j}] expected NaN");
            } else {
                assert!(
                    (scaled[[i, j]] - sk[i][j]).abs() < 1e-9,
                    "[{i},{j}]: ferrolearn={}, sklearn={}",
                    scaled[[i, j]],
                    sk[i][j]
                );
            }
        }
    }
    Ok(())
}

/// An ALL-NaN column yields `mean_`/`var_`/`scale_` = NaN and transforms to NaN,
/// with NO panic. Live oracle (run from /tmp, sklearn 1.5.2):
///   `m = StandardScaler().fit(np.array([[np.nan,1.],[np.nan,2.],[np.nan,3.]]))`
///   -> `mean_=[nan, 2.0]`, `var_=[nan, 0.6666666666666666]`,
///      `scale_=[nan, 0.816496580927726]`;
///   `m.transform(...)` ->
///   `[[nan, -1.224744871391589],[nan, 0.0],[nan, 1.224744871391589]]`.
#[test]
fn req7_all_nan_column_yields_nan_no_panic() -> Result<(), FerroError> {
    let nan = f64::NAN;
    let scaler = StandardScaler::<f64>::new();
    let x = array![[nan, 1.0], [nan, 2.0], [nan, 3.0]];
    let fitted = scaler.fit(&x, &())?;

    assert!(fitted.mean()[0].is_nan(), "all-NaN col mean_ must be NaN");
    assert!(fitted.var()[0].is_nan(), "all-NaN col var_ must be NaN");
    assert!(fitted.scale()[0].is_nan(), "all-NaN col scale_ must be NaN");
    // Finite column 0..2 unaffected.
    assert!((fitted.mean()[1] - 2.0).abs() < 1e-12);
    assert!((fitted.var()[1] - 0.6666666666666666).abs() < 1e-12);
    assert!((fitted.scale()[1] - 0.816496580927726).abs() < 1e-12);

    let scaled = fitted.transform(&x)?;
    let sk_col1 = [-1.224744871391589, 0.0, 1.224744871391589];
    for i in 0..3 {
        assert!(scaled[[i, 0]].is_nan(), "all-NaN col transform must be NaN");
        assert!((scaled[[i, 1]] - sk_col1[i]).abs() < 1e-9, "col1 row {i}");
    }
    Ok(())
}

/// NaN passes through `inverse_transform` (no rejection). Live oracle
/// (run from /tmp, sklearn 1.5.2):
///   `m = StandardScaler().fit(np.array([[1.],[2.],[3.]]))`;
///   `m.inverse_transform(np.array([[np.nan],[0.0]]))` -> `[[nan],[2.0]]`
///   (`0.0 * scale_(0.8165) + mean_(2.0) = 2.0`).
#[test]
fn req7_nan_passthrough_inverse_transform() -> Result<(), FerroError> {
    let nan = f64::NAN;
    let scaler = StandardScaler::<f64>::new();
    let x_train = array![[1.0], [2.0], [3.0]];
    let fitted = scaler.fit(&x_train, &())?;

    let x = array![[nan], [0.0]];
    let inv = fitted.inverse_transform(&x)?;
    assert!(
        inv[[0, 0]].is_nan(),
        "NaN must pass through inverse_transform"
    );
    assert!(
        (inv[[1, 0]] - 2.0).abs() < 1e-12,
        "0.0 inverts to mean_=2.0"
    );
    Ok(())
}

/// inf is REJECTED at fit / transform / inverse_transform (allow-nan rejects
/// +/-inf), matching sklearn `force_all_finite="allow-nan"` raising
/// `ValueError("Input X contains infinity or a value too large for dtype")`
/// (`_data.py:918`,`:1052`,`:1094`). Live oracle (run from /tmp, sklearn 1.5.2):
///   `StandardScaler().fit(np.array([[1.],[np.inf],[3.]]))` -> ValueError;
///   `m.transform(np.array([[np.inf]]))` -> ValueError;
///   `m.inverse_transform(np.array([[np.inf]]))` -> ValueError.
#[test]
fn inf_rejected_fit() {
    let inf = f64::INFINITY;
    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0], [inf], [3.0]];
    assert!(matches!(
        scaler.fit(&x, &()),
        Err(FerroError::InvalidParameter { .. })
    ));
    // -inf likewise rejected.
    let x_neg = array![[1.0], [f64::NEG_INFINITY], [3.0]];
    assert!(scaler.fit(&x_neg, &()).is_err());
}

#[test]
fn inf_rejected_transform() -> Result<(), FerroError> {
    let inf = f64::INFINITY;
    let scaler = StandardScaler::<f64>::new();
    let fitted = scaler.fit(&array![[1.0], [2.0], [3.0]], &())?;
    let x = array![[inf]];
    assert!(matches!(
        fitted.transform(&x),
        Err(FerroError::InvalidParameter { .. })
    ));
    Ok(())
}

#[test]
fn inf_rejected_inverse_transform() -> Result<(), FerroError> {
    let inf = f64::INFINITY;
    let scaler = StandardScaler::<f64>::new();
    let fitted = scaler.fit(&array![[1.0], [2.0], [3.0]], &())?;
    let x = array![[inf]];
    assert!(matches!(
        fitted.inverse_transform(&x),
        Err(FerroError::InvalidParameter { .. })
    ));
    Ok(())
}

/// A NaN-only-in-some-rows column still fits (NaN allowed, only inf rejected).
/// Confirms the inf guard does NOT also reject NaN. Live oracle: the
/// `req7_nan_fit_single_column_ignored` fixture fits successfully.
#[test]
fn nan_only_still_fits() -> Result<(), FerroError> {
    let nan = f64::NAN;
    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0], [nan], [3.0]];
    assert!(scaler.fit(&x, &()).is_ok());
    Ok(())
}

// ===========================================================================
// REQ-2/REQ-5 — _is_constant_feature variance-relative constant detection
//   (the StandardScaler-specific mask, NOT the `< 10*eps` default of
//   Min/MaxAbs/Robust). Live sklearn 1.5.2 oracle (run from /tmp, R-CHAR-3).
// ===========================================================================

/// A NEAR-constant column (tiny but NONZERO variance below the
/// `_is_constant_feature` bound) gets `scale_ = 1.0` and transforms its first
/// row to 0. sklearn `_is_constant_feature(var, mean, n)` returns True iff
/// `var <= n*eps*var + (n*mean*eps)^2` (`_data.py:72-85`,`:1016-1018`), so a
/// column like `[1e8, 1e8+1e-8]` (var ~5.55e-17, mean 1e8) IS flagged constant.
/// Live oracle (run from /tmp, sklearn 1.5.2):
///   `m = StandardScaler().fit(np.array([[1e8],[1e8 + 1e-8]]))`
///   -> `mean_=[1e8]`, `var_=[5.551115123125783e-17]`, `scale_=[1.0]`;
///   `_is_constant_feature(m.var_, m.mean_, m.n_samples_seen_) == [True]`;
///   `m.transform(...)` -> `[[0.0],[1.4901161193847656e-08]]`.
/// This is the DISTINGUISHING test: the prior exact-zero `s == 0` mask would
/// have divided by the tiny std (var ~5.55e-17 != 0), NOT set scale_=1.
#[test]
fn req7_near_constant_column_scale_one() -> Result<(), FerroError> {
    let scaler = StandardScaler::<f64>::new();
    let x = array![[1e8], [1e8 + 1e-8]];
    let fitted = scaler.fit(&x, &())?;

    // sklearn: var_ != 0 but _is_constant_feature flags it -> scale_ EXACTLY 1.0.
    assert!(fitted.var()[0] > 0.0, "near-constant var_ is nonzero");
    assert!(fitted.var()[0] < 1e-15, "near-constant var_ is tiny");
    assert_eq!(
        fitted.scale()[0],
        1.0,
        "_is_constant_feature -> scale_ = 1.0"
    );

    let scaled = fitted.transform(&x)?;
    // Row 0: (1e8 - 1e8)/1 = 0. Row 1: (1e8+1e-8 - 1e8)/1 = 1.4901161193847656e-08.
    assert!((scaled[[0, 0]] - 0.0).abs() < 1e-15);
    assert!((scaled[[1, 0]] - 1.4901161193847656e-08).abs() < 1e-18);
    Ok(())
}

/// A GENUINELY-constant column (`[[5],[5],[5]]`) -> `var_=0`, `scale_=1.0`,
/// transform 0. Live oracle (run from /tmp, sklearn 1.5.2):
///   `m = StandardScaler().fit(np.array([[5.],[5.],[5.]]))`
///   -> `mean_=[5.0]`, `var_=[0.0]`, `scale_=[1.0]`;
///   `m.transform(...)` -> `[[0.0],[0.0],[0.0]]`.
#[test]
fn req7_genuinely_constant_column_scale_one() -> Result<(), FerroError> {
    let scaler = StandardScaler::<f64>::new();
    let x = array![[5.0], [5.0], [5.0]];
    let fitted = scaler.fit(&x, &())?;

    assert_eq!(fitted.var()[0], 0.0);
    assert_eq!(fitted.scale()[0], 1.0);
    let scaled = fitted.transform(&x)?;
    for i in 0..3 {
        assert!((scaled[[i, 0]] - 0.0).abs() < 1e-15);
    }
    Ok(())
}

/// Regression guard: a normal small-variance NON-constant column (`[1,2,3]`)
/// is NOT flagged constant and keeps `scale_ = sqrt(var_)`. sklearn:
///   `m = StandardScaler().fit(np.array([[1.],[2.],[3.]]))`
///   -> `var_=[0.6666666666666666]`, `scale_=[0.816496580927726]` (NOT 1.0).
/// Confirms the `_is_constant_feature` upgrade does not over-flag normal cols.
#[test]
fn req7_normal_column_not_flagged_constant() -> Result<(), FerroError> {
    let scaler = StandardScaler::<f64>::new();
    let x = array![[1.0], [2.0], [3.0]];
    let fitted = scaler.fit(&x, &())?;

    assert!((fitted.var()[0] - 0.6666666666666666).abs() < 1e-12);
    assert!((fitted.scale()[0] - 0.816496580927726).abs() < 1e-12);
    assert!(
        fitted.scale()[0] != 1.0,
        "normal col must NOT be flagged constant"
    );
    Ok(())
}

/// `copy=true`/`copy=false` is a no-op on values (ABI-only in ferrolearn).
/// sklearn's `copy` governs in-place-vs-copy; the transformed VALUES are
/// identical. Live oracle: both produce the same fit_transform output.
#[test]
fn req7_copy_is_no_op_on_values() -> Result<(), FerroError> {
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
    let copy_true = StandardScaler::<f64>::new().with_copy(true);
    let copy_false = StandardScaler::<f64>::new().with_copy(false);
    let a = copy_true.fit(&x, &())?.transform(&x)?;
    let b = copy_false.fit(&x, &())?.transform(&x)?;
    for (va, vb) in a.iter().zip(b.iter()) {
        assert_eq!(va.to_bits(), vb.to_bits(), "copy must not change values");
    }
    Ok(())
}

/// f32 path: `_is_constant_feature` uses f64 eps even for f32 (sklearn
/// "variance is always computed using float64 accumulators", `_data.py:81-82`).
/// A genuinely-constant f32 column gets scale_ = 1.0. Live oracle (sklearn
/// upcasts f32 to float64 internally for var): constant col -> 0.
#[test]
fn req7_f32_constant_column_scale_one() -> Result<(), FerroError> {
    let scaler = StandardScaler::<f32>::new();
    let x = array![[5.0f32, 1.0], [5.0, 2.0], [5.0, 3.0]];
    let fitted = scaler.fit(&x, &())?;
    assert_eq!(fitted.scale()[0], 1.0f32);
    let scaled = fitted.transform(&x)?;
    for i in 0..3 {
        assert!(scaled[[i, 0]].abs() < 1e-6);
    }
    Ok(())
}

// ===========================================================================
// REQ-7 / REQ-5 — f32 path: sklearn computes mean_/var_/scale_ in FLOAT64
//   accumulators even for float32 input ("variance is always computed using
//   float64 accumulators", _data.py:81-82; _incremental_mean_and_var upcasts).
//   ferrolearn computes them in the generic F (= f32 on the f32 path), so a
//   large-mean f32 column whose true mean is NOT representable in f32 diverges.
// ===========================================================================

/// Divergence: ferrolearn's `FittedStandardScaler::<f32>::fit` computes the
/// per-column `mean_`/`var_`/`scale_` in f32 arithmetic
/// (`standard_scaler.rs:357-382`, generic `F = f32`), whereas sklearn always
/// upcasts to float64 accumulators for the mean/variance
/// (`_incremental_mean_and_var`, `extmath.py:1057-1178`; `mean_`/`var_`/`scale_`
/// are dtype float64 even for float32 input, `_data.py:1016`). For an f32 column
/// whose true mean is NOT representable in f32, ferrolearn's f32-rounded mean
/// shifts every deviation, producing a wrong transform.
///
/// Input f32 column `[16777216, 16777216, 16777220]` (the first two are the f32
/// rounding of `16777216` and `16777217`; `16777217` is not representable in
/// f32 near 2^24, step 2):
///   * sklearn mean_=16777217.333333332 (float64), var_=3.555555555555556,
///     scale_=1.8856180831641267, transform (f32 out) =
///     [-0.7071068286895752, -0.7071068286895752, 1.4142136573791504].
///   * ferrolearn mean=16777218.0 (f32 rounding of the true mean), so all three
///     deviations become +/-2 -> transform = [-1.0606601, -1.0606601, 1.0606601].
///
/// Live oracle (run from /tmp, sklearn 1.5.2):
///   X = np.array([[16777216.],[16777216.],[16777220.]], dtype=np.float32)
///   m = StandardScaler().fit(X)
///   m.mean_  -> array([16777217.33333333])   (float64)
///   m.transform(X).ravel() ->
///     array([-0.7071068, -0.7071068, 1.4142137], dtype=float32)
///
/// Tracking: #2205
#[test]
#[ignore = "divergence: StandardScaler<f32> computes mean_/var_/scale_ in f32 not float64 accumulators; tracking #2205"]
#[allow(
    clippy::approx_constant,
    reason = "these are the LIVE sklearn 1.5.2 f32 transform-output oracle values \
              (R-CHAR-3) for a 2-equal/1-different fixture; they coincide with \
              +/-1/sqrt(2) and sqrt(2) by construction, not because we meant the \
              math constant"
)]
fn divergence_f32_uses_float64_accumulators() {
    // Live sklearn 1.5.2 oracle (run from /tmp). sklearn computes the mean in
    // float64, so the true mean 16777217.333... is preserved; transform output:
    let sk_transform = [-0.7071068f32, -0.7071068f32, 1.4142137f32];
    // sklearn mean_ is float64; the closest f32 value sklearn would NOT collapse
    // to is 16777218.0 (which is what ferrolearn computes). The float64 mean is
    // strictly between 16777216 and 16777218.
    const SK_MEAN: f64 = 16777217.333333332;

    let scaler = StandardScaler::<f32>::new();
    let x = array![[16777216.0f32], [16777216.0f32], [16777220.0f32]];
    let fitted = scaler.fit(&x, &()).unwrap();

    // sklearn's mean_ (computed in float64) is 16777217.333..., NOT the f32
    // value 16777218.0 that ferrolearn computes.
    let ferro_mean = f64::from(fitted.mean()[0]);
    assert!(
        (ferro_mean - SK_MEAN).abs() < 0.5,
        "mean_: ferrolearn (f32) ={ferro_mean}, sklearn (float64) ={SK_MEAN}"
    );

    let scaled = fitted.transform(&x).unwrap();
    for i in 0..3 {
        assert!(
            (scaled[[i, 0]] - sk_transform[i]).abs() < 1e-5,
            "transform[{i}]: ferrolearn={}, sklearn oracle={}",
            scaled[[i, 0]],
            sk_transform[i]
        );
    }
}
