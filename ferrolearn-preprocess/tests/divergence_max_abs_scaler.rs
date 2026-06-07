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

use ferrolearn_core::error::FerroError;
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

// ===========================================================================
// REQ-5 — NaN tolerance (allow-nan: ignore NaN in max(|x|), pass NaN through)
// Direct analog of MinMaxScaler REQ-4. All expected values from the LIVE
// sklearn 1.5.2 oracle (R-CHAR-3); NaN is IGNORED for max_abs (_nanmax,
// `_data.py:1263`) under force_all_finite="allow-nan" (`:1256`).
// ===========================================================================

/// NaN in a single column is ignored for `max_abs` and passes through transform.
/// Live oracle (sklearn 1.5.2, from /tmp):
/// `python3 -c "import numpy as np; from sklearn.preprocessing import MaxAbsScaler; \
///   m=MaxAbsScaler().fit(np.array([[1.],[np.nan],[-4.]])); \
///   print(m.max_abs_.tolist(), m.scale_.tolist(), \
///   m.transform(np.array([[1.],[np.nan],[-4.]])).tolist())"`
/// -> `max_abs_=[4.0]`, `scale_=[4.0]`, transform `[[0.25],[nan],[-1.0]]`.
#[test]
fn req5_nan_fit_single_column_ignored() {
    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[1.0], [f64::NAN], [-4.0]];
    let fitted = scaler.fit(&x, &()).unwrap();

    assert!((fitted.max_abs()[0] - 4.0).abs() < 1e-12);
    assert!((fitted.scale()[0] - 4.0).abs() < 1e-12);

    let scaled = fitted.transform(&x).unwrap();
    assert!((scaled[[0, 0]] - 0.25).abs() < 1e-12);
    assert!(scaled[[1, 0]].is_nan(), "NaN input must pass through");
    assert!((scaled[[2, 0]] - (-1.0)).abs() < 1e-12);
}

/// Scattered NaN across multiple columns. Each column's `max_abs` ignores its
/// own NaN entries. Live oracle:
/// `m=MaxAbsScaler().fit(np.array([[1.,10.],[np.nan,-20.],[-4.,np.nan],[2.,5.]]))`
/// -> `max_abs_=[4.0, 20.0]`, `scale_=[4.0, 20.0]`;
/// transform -> `[[0.25,0.5],[nan,-1.0],[-1.0,nan],[0.5,0.25]]`.
#[test]
fn req5_nan_fit_multi_column_scattered() {
    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[1.0, 10.0], [f64::NAN, -20.0], [-4.0, f64::NAN], [2.0, 5.0]];
    let fitted = scaler.fit(&x, &()).unwrap();

    assert!((fitted.max_abs()[0] - 4.0).abs() < 1e-12);
    assert!((fitted.max_abs()[1] - 20.0).abs() < 1e-12);
    assert!((fitted.scale()[0] - 4.0).abs() < 1e-12);
    assert!((fitted.scale()[1] - 20.0).abs() < 1e-12);

    let scaled = fitted.transform(&x).unwrap();
    // sklearn oracle, row-major.
    let sk = [[0.25, 0.5], [f64::NAN, -1.0], [-1.0, f64::NAN], [0.5, 0.25]];
    for i in 0..4 {
        for j in 0..2 {
            if sk[i][j].is_nan() {
                assert!(scaled[[i, j]].is_nan(), "expected NaN at ({i},{j})");
            } else {
                assert!(
                    (scaled[[i, j]] - sk[i][j]).abs() < 1e-12,
                    "got {} sklearn {} at ({i},{j})",
                    scaled[[i, j]],
                    sk[i][j]
                );
            }
        }
    }
}

/// An ALL-NaN column yields `max_abs_ = NaN` / `scale_ = NaN` (matching
/// `_nanmax` on an all-NaN slice → nan) and transforms to NaN — no panic, no
/// zero-substitution. Live oracle:
/// `m=MaxAbsScaler().fit(np.array([[np.nan,1.],[np.nan,-2.]]))`
/// -> `max_abs_=[nan, 2.0]`, `scale_=[nan, 2.0]`;
/// transform -> `[[nan,0.5],[nan,-1.0]]`.
#[test]
fn req5_all_nan_column_yields_nan_no_panic() {
    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[f64::NAN, 1.0], [f64::NAN, -2.0]];
    let fitted = scaler.fit(&x, &()).unwrap();

    assert!(fitted.max_abs()[0].is_nan(), "all-NaN col -> max_abs NaN");
    assert!(fitted.scale()[0].is_nan(), "all-NaN col -> scale_ NaN");
    assert!((fitted.max_abs()[1] - 2.0).abs() < 1e-12);

    let scaled = fitted.transform(&x).unwrap();
    assert!(scaled[[0, 0]].is_nan());
    assert!(scaled[[1, 0]].is_nan());
    assert!((scaled[[0, 1]] - 0.5).abs() < 1e-12);
    assert!((scaled[[1, 1]] - (-1.0)).abs() < 1e-12);
}

/// NaN passes through `inverse_transform` (`nan * scale = nan`). Live oracle:
/// `m=MaxAbsScaler().fit(np.array([[1.],[-4.]]))` -> `scale_=[4.0]`;
/// `m.inverse_transform(np.array([[0.25],[np.nan]]))` -> `[[1.0],[nan]]`.
#[test]
fn req5_nan_passthrough_inverse_transform() {
    let scaler = MaxAbsScaler::<f64>::new();
    let fitted = scaler.fit(&array![[1.0], [-4.0]], &()).unwrap();
    let out = fitted
        .inverse_transform(&array![[0.25], [f64::NAN]])
        .unwrap();
    assert!((out[[0, 0]] - 1.0).abs() < 1e-12);
    assert!(
        out[[1, 0]].is_nan(),
        "NaN must pass through inverse_transform"
    );
}

// ===========================================================================
// inf-rejection — allow-nan ALLOWS NaN but REJECTS +/-inf at fit/transform/
// inverse_transform (sklearn raises ValueError "Input contains infinity").
// MinMaxScaler #2200 precedent. Oracle:
// `MaxAbsScaler().fit(np.array([[np.inf]]))` -> ValueError; same for transform
// and inverse_transform; NaN-only still fits.
// ===========================================================================

/// `fit` rejects +inf (sklearn `force_all_finite="allow-nan"`, `_data.py:1256`).
#[test]
fn inf_rejected_fit() {
    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[f64::INFINITY]];
    assert!(matches!(
        scaler.fit(&x, &()),
        Err(FerroError::InvalidParameter { .. })
    ));
    // -inf too.
    let x = array![[1.0], [f64::NEG_INFINITY]];
    assert!(scaler.fit(&x, &()).is_err());
}

/// `transform` rejects +/-inf (sklearn `:1299`).
#[test]
fn inf_rejected_transform() {
    let scaler = MaxAbsScaler::<f64>::new();
    let fitted = scaler.fit(&array![[1.0], [2.0]], &()).unwrap();
    assert!(matches!(
        fitted.transform(&array![[f64::INFINITY]]),
        Err(FerroError::InvalidParameter { .. })
    ));
}

/// `inverse_transform` rejects +/-inf (sklearn `:1331`).
#[test]
fn inf_rejected_inverse_transform() {
    let scaler = MaxAbsScaler::<f64>::new();
    let fitted = scaler.fit(&array![[1.0], [2.0]], &()).unwrap();
    assert!(matches!(
        fitted.inverse_transform(&array![[f64::INFINITY]]),
        Err(FerroError::InvalidParameter { .. })
    ));
}

/// A NaN-only fit still succeeds (allow-nan permits NaN, rejects only inf).
/// Live oracle: `MaxAbsScaler().fit(np.array([[np.nan],[np.nan]])).max_abs_`
/// -> `[nan]` (no error).
#[test]
fn nan_only_still_fits() {
    let scaler = MaxAbsScaler::<f64>::new();
    let fitted = scaler.fit(&array![[f64::NAN], [f64::NAN]], &()).unwrap();
    assert!(fitted.max_abs()[0].is_nan());
}

// ===========================================================================
// REQ-2 preserved — zero-max_abs held-out (the NaN fix must not break it).
// ===========================================================================

/// REQ-2 preserved after the NaN/scale_ rework: fit `[[0],[0]]`, transform a
/// held-out non-zero `[[5]]` -> `[[5.0]]` (scale_=1, x/1=x). Live oracle:
/// `m=MaxAbsScaler().fit(np.array([[0.],[0.]])); m.transform(np.array([[5.]]))`
/// -> `[[5.0]]`.
#[test]
fn req2_zero_maxabs_holdout_preserved() {
    let scaler = MaxAbsScaler::<f64>::new();
    let fitted = scaler.fit(&array![[0.0], [0.0]], &()).unwrap();
    assert!(fitted.scale()[0] == 1.0, "zero-max_abs col -> scale_ = 1");
    let out = fitted.transform(&array![[5.0]]).unwrap();
    assert!((out[[0, 0]] - 5.0).abs() < 1e-15, "x/1 must equal x");
}

// ===========================================================================
// REQ-9 — copy param (accept-and-document no-op): copy=True/False identical.
// ===========================================================================

/// `copy=true` and `copy=false` produce identical output (ferrolearn's
/// `Transform` always returns a fresh array, so `copy` is a no-op). Live oracle
/// confirms sklearn's two paths are value-identical:
/// `MaxAbsScaler(copy=True).fit_transform([[2.,-4.],[1.,3.]])` ==
/// `MaxAbsScaler(copy=False).fit_transform(...)` -> `[[1.0,-1.0],[0.5,0.75]]`.
#[test]
fn req9_copy_is_no_op_on_values() {
    let sk = array![[1.0, -1.0], [0.5, 0.75]];
    let x = array![[2.0, -4.0], [1.0, 3.0]];

    let with_copy = MaxAbsScaler::<f64>::new().with_copy(true);
    let no_copy = MaxAbsScaler::<f64>::new().with_copy(false);
    assert!(with_copy.copy());
    assert!(!no_copy.copy());

    let a = with_copy.fit_transform(&x).unwrap();
    let b = no_copy.fit_transform(&x).unwrap();
    for ((va, vb), s) in a.iter().zip(b.iter()).zip(sk.iter()) {
        assert!((va - vb).abs() < 1e-15, "copy changed values: {va} vs {vb}");
        assert!((va - s).abs() < 1e-12, "got {va}, sklearn {s}");
    }
}

// ===========================================================================
// DIVERGENCE — `_handle_zeros_in_scale` near-constant threshold (not just ==0)
// ===========================================================================

/// Divergence: ferrolearn's `Fit::fit` computes
/// `scale_ = max_abs.mapv(|m| if m == 0 { 1 } else { m })`
/// (`ferrolearn-preprocess/src/max_abs_scaler.rs:267`), replacing ONLY an
/// EXACTLY-zero `max_abs` with `1.0`. scikit-learn's
/// `_handle_zeros_in_scale` (`sklearn/preprocessing/_data.py:114`) instead uses
/// `constant_mask = scale < 10 * eps; scale[constant_mask] = 1.0`, treating ANY
/// `max_abs` below `10 * np.finfo(dtype).eps` (≈ 2.22e-15 for f64) as a constant
/// column and forcing `scale_ = 1.0`.
///
/// For a tiny-but-nonzero column `[[1e-16],[-1e-16]]` (`1e-16 < 10*eps`):
///   sklearn `_data.py:114,119`: `scale_ = [1.0]`,
///     `transform([[1e-16]]) = 1e-16 / 1.0 = 1e-16`.
///   ferrolearn `:267`: `scale_ = [1e-16]` (not exactly 0),
///     `transform([[1e-16]]) = 1e-16 / 1e-16 = 1.0`.
///
/// Live sklearn 1.5.2 oracle (from /tmp):
/// `python3 -c "import numpy as np; from sklearn.preprocessing import MaxAbsScaler; \
///   m=MaxAbsScaler().fit(np.array([[1e-16],[-1e-16]])); \
///   print(m.scale_.tolist(), m.transform(np.array([[1e-16]])).tolist())"`
/// -> `scale_=[1.0]`, `transform=[[1e-16]]`.
///
/// Tracking: #2203
#[test]
#[ignore = "divergence: MaxAbsScaler scale_ only replaces exact-zero, not <10*eps near-constant per sklearn _handle_zeros_in_scale; tracking #2203"]
fn divergence_handle_zeros_near_constant_threshold() {
    // sklearn oracle constants (_data.py:114,119 — constant_mask = scale < 10*eps).
    let sk_scale0 = 1.0_f64;
    let sk_transform = 1e-16_f64; // 1e-16 / 1.0

    let scaler = MaxAbsScaler::<f64>::new();
    let x = array![[1e-16], [-1e-16]];
    let fitted = scaler.fit(&x, &()).unwrap();

    // sklearn forces scale_ to 1.0 for a near-constant column (max_abs < 10*eps).
    assert!(
        fitted.scale()[0] == sk_scale0,
        "scale_ = {} but sklearn _handle_zeros_in_scale forces 1.0 for max_abs < 10*eps",
        fitted.scale()[0]
    );

    // Consequently transform of a tiny value stays tiny (x / 1.0), not 1.0.
    let scaled = fitted.transform(&array![[1e-16]]).unwrap();
    assert!(
        (scaled[[0, 0]] - sk_transform).abs() < 1e-30,
        "transform = {} but sklearn yields {} (x / scale_(1.0))",
        scaled[[0, 0]],
        sk_transform
    );
}
