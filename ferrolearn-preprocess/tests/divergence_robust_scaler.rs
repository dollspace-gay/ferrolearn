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

use ferrolearn_core::error::FerroError;
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

/// #2204: sklearn `_handle_zeros_in_scale` (`_data.py:114-119`) replaces an
/// `IQR < 10 * eps` (near-constant) column's scale with 1.0 — NOT just an
/// exactly-zero IQR. A tiny-but-nonzero IQR column must divide by 1.0 (the
/// column is still centered), not by the tiny IQR.
///
/// Live sklearn 1.5.2 oracle (run from /tmp):
///   r = RobustScaler().fit([[1e-16],[2e-16],[1.5e-16],[1e-16]])
///   r.center_ -> [1.25e-16]; r.scale_ -> [1.0]
///   r.transform([[3e-16]]) -> [[1.7499999999999998e-16]]  (= 3e-16 - 1.25e-16)
#[test]
fn req2204_near_constant_iqr_handled_like_sklearn() {
    let x = array![[1e-16_f64], [2e-16], [1.5e-16], [1e-16]];
    let fitted = RobustScaler::<f64>::new().fit(&x, &()).unwrap();
    // IQR ~1e-16 < 10*eps -> effective scale 1.0; centered by median 1.25e-16.
    let out = fitted.transform(&array![[3e-16_f64]]).unwrap();
    let expected = 3e-16 - 1.25e-16; // = 1.75e-16
    assert!(
        (out[[0, 0]] - expected).abs() < 1e-30,
        "near-constant IQR: transform {} != sklearn {expected}",
        out[[0, 0]]
    );
}

// ===========================================================================
// REQ-9: NaN tolerance (allow-nan / nanmedian / nanpercentile) + inf rejection
// (sklearn _data.py:1601, :1614, :1630, :1665). Live sklearn 1.5.2 oracle
// (R-CHAR-3), run from /tmp, derived with np.nanmedian/np.nanpercentile and
// ASSERTED against the live RobustScaler fitted attrs — never copied from the
// ferrolearn side.
//
// NOTE: RobustScaler has no `inverse_transform` (REQ-7 NOT-STARTED, #1252), so
// the inverse_transform NaN-passthrough / inf-rejection sites the StandardScaler
// / MaxAbsScaler analogs cover do not yet exist here; they belong to the REQ-7
// build, not this REQ-9 dispatch.
// ===========================================================================

/// REQ-9 (GREEN once shipped). NaN in `fit` is IGNORED: the median/IQR are
/// computed over the FINITE values only.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.preprocessing import RobustScaler; \
///     X=np.array([[1.],[np.nan],[3.],[5.],[7.]]); r=RobustScaler().fit(X); \
///     print(r.center_.tolist(), r.scale_.tolist(), r.transform(X).tolist())"
///   center_ == [4.0]   (== np.nanmedian([1,3,5,7]) == 4.0)
///   scale_  == [3.0]   (== np.nanpercentile([1,3,5,7],[25,75]) IQR == 5.5-2.5)
///   transform == [[-1.0],[nan],[-0.3333333333333333],[0.3333333333333333],[1.0]]
#[test]
fn req9_nan_fit_single_column_ignored() {
    let nan = f64::NAN;
    let x = array![[1.0], [nan], [3.0], [5.0], [7.0]];
    let fitted = RobustScaler::<f64>::new().fit(&x, &()).unwrap();

    // center_ / scale_ over the 4 finite values (nan-ignoring).
    assert!((fitted.center()[0] - 4.0).abs() <= 1e-12);
    assert!((fitted.scale()[0] - 3.0).abs() <= 1e-12);

    let out = fitted.transform(&x).unwrap();
    let expected = [-1.0, f64::NAN, -1.0 / 3.0, 1.0 / 3.0, 1.0];
    for (i, &e) in expected.iter().enumerate() {
        if e.is_nan() {
            assert!(out[[i, 0]].is_nan(), "row {i}: expected nan passthrough");
        } else {
            assert!(
                (out[[i, 0]] - e).abs() <= 1e-12,
                "row {i}: sklearn={e}, ferrolearn={}",
                out[[i, 0]]
            );
        }
    }
}

/// REQ-9 multi-column scattered NaN: each column's median/IQR ignore its own
/// NaNs independently.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   X = [[1,10],[nan,20],[3,nan],[5,40],[7,50]]
///   r = RobustScaler().fit(X)
///   r.center_ == [4.0, 30.0]   (col0 nanmedian[1,3,5,7]=4; col1 nanmedian[10,20,40,50]=30)
///   r.scale_  == [3.0, 25.0]   (col1 nanpercentile[10,20,40,50],[25,75]=[17.5,42.5] -> 25)
///   r.transform(X) == [[-1.0,-0.8],[nan,-0.4],[-0.3333..,nan],[0.3333..,0.4],[1.0,0.8]]
#[test]
fn req9_nan_fit_multi_column_scattered() {
    let nan = f64::NAN;
    let x = array![
        [1.0, 10.0],
        [nan, 20.0],
        [3.0, nan],
        [5.0, 40.0],
        [7.0, 50.0]
    ];
    let fitted = RobustScaler::<f64>::new().fit(&x, &()).unwrap();

    assert!((fitted.center()[0] - 4.0).abs() <= 1e-12);
    assert!((fitted.center()[1] - 30.0).abs() <= 1e-12);
    assert!((fitted.scale()[0] - 3.0).abs() <= 1e-12);
    assert!((fitted.scale()[1] - 25.0).abs() <= 1e-12);

    let out = fitted.transform(&x).unwrap();
    // NaN input cells pass through.
    assert!(out[[1, 0]].is_nan());
    assert!(out[[2, 1]].is_nan());
    // A few finite cells.
    assert!((out[[0, 0]] - (-1.0)).abs() <= 1e-12);
    assert!((out[[0, 1]] - (-0.8)).abs() <= 1e-12);
    assert!((out[[4, 1]] - 0.8).abs() <= 1e-12);
}

/// REQ-9 KEY R-CODE-2 case: an ALL-NaN column yields center_/scale_ = NaN and
/// transform = NaN, with NO PANIC (the empty finite slice must not hit
/// `quantile_sorted`'s old "Panics if empty" path).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   X = [[1,nan],[3,nan],[5,nan]]
///   r = RobustScaler().fit(X)
///   r.center_ == [3.0, nan]   r.scale_ == [2.0, nan]
///   r.transform(X) == [[-1.0,nan],[0.0,nan],[1.0,nan]]
#[test]
fn req9_all_nan_column_yields_nan_no_panic() {
    let nan = f64::NAN;
    let x = array![[1.0, nan], [3.0, nan], [5.0, nan]];
    let fitted = RobustScaler::<f64>::new().fit(&x, &()).unwrap();

    // Finite column 0 is normal.
    assert!((fitted.center()[0] - 3.0).abs() <= 1e-12);
    assert!((fitted.scale()[0] - 2.0).abs() <= 1e-12);
    // All-NaN column 1: center_/scale_ = NaN (no panic).
    assert!(fitted.center()[1].is_nan());
    assert!(fitted.scale()[1].is_nan());

    let out = fitted.transform(&x).unwrap();
    for i in 0..3 {
        assert!(out[[i, 1]].is_nan(), "all-NaN col row {i} should be nan");
    }
    // Finite column still transforms correctly.
    assert!((out[[0, 0]] - (-1.0)).abs() <= 1e-12);
    assert!((out[[2, 0]] - 1.0).abs() <= 1e-12);
}

/// REQ-9 conditional `with_centering=false` path still passes NaN through.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   X = [[1.],[nan],[3.],[5.],[7.]]
///   RobustScaler(with_centering=False).fit(X).transform(X)
///   == [[0.3333..],[nan],[1.0],[1.6666..],[2.3333..]]   (scale only, scale_=3)
#[test]
fn req9_with_centering_false_nan() {
    let nan = f64::NAN;
    let x = array![[1.0], [nan], [3.0], [5.0], [7.0]];
    let fitted = RobustScaler::<f64>::new()
        .with_with_centering(false)
        .fit(&x, &())
        .unwrap();
    let out = fitted.transform(&x).unwrap();
    assert!(out[[1, 0]].is_nan());
    assert!((out[[0, 0]] - (1.0 / 3.0)).abs() <= 1e-12);
    assert!((out[[2, 0]] - 1.0).abs() <= 1e-12);
    assert!((out[[4, 0]] - (7.0 / 3.0)).abs() <= 1e-12);
}

/// REQ-9 conditional `with_scaling=false` path still passes NaN through.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   X = [[1.],[nan],[3.],[5.],[7.]]
///   RobustScaler(with_scaling=False).fit(X).transform(X)
///   == [[-3.0],[nan],[-1.0],[1.0],[3.0]]   (center only, center_=4)
#[test]
fn req9_with_scaling_false_nan() {
    let nan = f64::NAN;
    let x = array![[1.0], [nan], [3.0], [5.0], [7.0]];
    let fitted = RobustScaler::<f64>::new()
        .with_with_scaling(false)
        .fit(&x, &())
        .unwrap();
    let out = fitted.transform(&x).unwrap();
    assert!(out[[1, 0]].is_nan());
    assert!((out[[0, 0]] - (-3.0)).abs() <= 1e-12);
    assert!((out[[2, 0]] - (-1.0)).abs() <= 1e-12);
    assert!((out[[4, 0]] - 3.0).abs() <= 1e-12);
}

/// REQ-9 inf-rejection: +/-inf is REJECTED at fit and at transform (allow-nan
/// allows NaN but rejects inf), mirroring the MinMaxScaler #2200 precedent.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   RobustScaler().fit([[1.],[inf],[3.]]) -> ValueError "Input X contains infinity..."
///   RobustScaler().fit([[1.],[2.],[3.]]).transform([[inf]]) -> ValueError
#[test]
fn req9_inf_rejected_fit() {
    let x = array![[1.0], [f64::INFINITY], [3.0]];
    assert!(matches!(
        RobustScaler::<f64>::new().fit(&x, &()),
        Err(FerroError::InvalidParameter { .. })
    ));
    // -inf too.
    let x2 = array![[1.0], [f64::NEG_INFINITY], [3.0]];
    assert!(matches!(
        RobustScaler::<f64>::new().fit(&x2, &()),
        Err(FerroError::InvalidParameter { .. })
    ));
}

/// REQ-9 inf-rejection at transform.
#[test]
fn req9_inf_rejected_transform() {
    let x = array![[1.0], [2.0], [3.0]];
    let fitted = RobustScaler::<f64>::new().fit(&x, &()).unwrap();
    let bad = array![[f64::INFINITY]];
    assert!(matches!(
        fitted.transform(&bad),
        Err(FerroError::InvalidParameter { .. })
    ));
}

/// REQ-9 a NaN-only input still FITS (no inf), matching sklearn: an all-NaN
/// column has center_/scale_ = NaN but raises no error.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   r = RobustScaler().fit([[nan],[nan]])
///   r.center_ == [nan]   r.scale_ == [nan]   (no exception)
#[test]
fn req9_nan_only_still_fits() {
    let nan = f64::NAN;
    let x = array![[nan], [nan]];
    let fitted = RobustScaler::<f64>::new().fit(&x, &()).unwrap();
    assert!(fitted.center()[0].is_nan());
    assert!(fitted.scale()[0].is_nan());
}

/// REQ-9 f32 NaN path: the NaN-skip + empty-slice safety is generic in `F`.
/// (Mirrors the single-column f64 oracle; f32 IQR of [1,3,5,7] is exact.)
#[test]
fn req9_f32_nan_fit_ignored() {
    let nan = f32::NAN;
    let x = array![[1.0f32], [nan], [3.0], [5.0], [7.0]];
    let fitted = RobustScaler::<f32>::new().fit(&x, &()).unwrap();
    assert!((fitted.center()[0] - 4.0).abs() <= 1e-6);
    assert!((fitted.scale()[0] - 3.0).abs() <= 1e-6);
    let out = fitted.transform(&x).unwrap();
    assert!(out[[1, 0]].is_nan());
    assert!((out[[0, 0]] - (-1.0)).abs() <= 1e-6);
}

/// Divergence: ferrolearn's `RobustScaler::<f32>::fit` (via `quantile_sorted`,
/// `robust_scaler.rs:55-73`) performs the percentile linear interpolation
/// `sorted[lo] + (sorted[hi]-sorted[lo])*frac` in **f32** arithmetic and stores
/// `scale_` as f32. scikit-learn's `np.nanpercentile` (called at
/// `sklearn/preprocessing/_data.py:1630`) UPCASTS float32 input to float64 for
/// the interpolation and returns a float64 `scale_` (`_data.py:1634-1635`); on
/// f32 input sklearn keeps `center_` as f32 (`np.nanmedian`, `:1614`) but
/// `scale_` becomes float64. The f32 round-off therefore makes ferrolearn's f32
/// `scale_` diverge from sklearn well beyond f32 ulp. Same root-cause family as
/// StandardScaler #2205 (f32 reduction in f32 vs sklearn float64 accumulators).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.preprocessing import RobustScaler; \
///     col=np.array([9783.1357421875,9641.921875,9640.8896484375,8744.34375,np.nan],dtype=np.float32); \
///     r=RobustScaler(quantile_range=(33.,67.)).fit(col.reshape(-1,1)); \
///     print(repr(float(r.scale_[0])), r.scale_.dtype)"
///   -> scale_ == 11.409824218750146  dtype float64  (np.nanpercentile upcasts f32 -> f64)
/// ferrolearn `RobustScaler::<f32>` returns scale_ == 11.410156 (f32 interp);
/// upcast to f64 that is 11.41015625. The interpolation is computed in f32, so
/// |11.41015625 - 11.409824218750146| ~= 3.3e-4 (rel ~2.9e-5) — far beyond f32
/// ulp at this magnitude (f32 ulp(11.41) ~= 1e-6). The two finite values whose
/// f32 difference loses precision are the upper-bound interpolands of Q67.
///
/// Tracking: #2206
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "the fixture values are written at full precision precisely BECAUSE \
              they are exact f32 magnitudes chosen so the percentile interpolation \
              fraction f32-rounds differently from numpy's float64 nanpercentile \
              (the #2206 divergence); truncating them would change the f32 value"
)]
fn divergence_f32_nanpercentile_upcasts_to_float64() {
    let nan = f32::NAN;
    let x = array![
        [9783.1357421875_f32],
        [9641.921875],
        [9640.8896484375],
        [8744.34375],
        [nan]
    ];
    let fitted = RobustScaler::<f32>::new()
        .with_quantile_range(33.0, 67.0)
        .unwrap()
        .fit(&x, &())
        .unwrap();

    // LIVE sklearn 1.5.2 oracle (np.nanpercentile upcasts f32 -> f64, returns
    // a float64 scale_ == 11.409824218750146). Post-#2206, ferrolearn computes the
    // percentile interpolation in f64 and stores `scale_` as f64 too (sklearn's
    // float64 attribute dtype), so `scale()[0]` is already f64 — directly
    // comparable to sklearn's float64 result within the generous f32-ulp 1e-5.
    let sk_scale_f64: f64 = 11.409_824_218_750_146;
    let f_scale_f64 = fitted.scale()[0];
    assert!(
        (f_scale_f64 - sk_scale_f64).abs() <= 1e-5,
        "f32 scale_: sklearn(float64 nanpercentile)={sk_scale_f64}, \
         ferrolearn(f32 interp, upcast)={f_scale_f64}, \
         diff={}",
        (f_scale_f64 - sk_scale_f64).abs()
    );
}

// ===========================================================================
// REQ-7 (#1252): inverse_transform = x*scale_ + center_ (scaling first, then
// centering), force_all_finite="allow-nan" (NaN passes, +/-inf rejected).
//
// Live sklearn 1.5.2 oracle (run from /tmp):
//   X=[[1,10],[2,20],[3,30],[100,40]]; r=RobustScaler().fit(X)
//   center_=[2.5,25.0]; scale_=[25.5,15.0]
//   r.inverse_transform(r.transform(X)) == X
//   r.inverse_transform([[0,0],[1,1]]) -> [[2.5,25.0],[28.0,40.0]]
//   RobustScaler().fit([[1],[2],[3],[4]]).inverse_transform([[nan],[0]]) -> [[nan],[2.5]]
// ===========================================================================

fn fixture_4x2() -> ndarray::Array2<f64> {
    array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [100.0, 40.0]]
}

#[test]
fn req7_inverse_roundtrip() {
    let x = fixture_4x2();
    let fitted = RobustScaler::<f64>::new().fit(&x, &()).unwrap();
    let xt = fitted.transform(&x).unwrap();
    let xinv = fitted.inverse_transform(&xt).unwrap();
    for ((i, j), &orig) in x.indexed_iter() {
        assert!(
            (xinv[[i, j]] - orig).abs() < 1e-9,
            "roundtrip[{i},{j}] = {} != {orig}",
            xinv[[i, j]]
        );
    }
}

#[test]
fn req7_inverse_holdout() {
    // Live oracle: inverse([[0,0],[1,1]]) = [[2.5,25.0],[28.0,40.0]]
    // (0*scale+center=center; 1*scale+center).
    let fitted = RobustScaler::<f64>::new().fit(&fixture_4x2(), &()).unwrap();
    let out = fitted
        .inverse_transform(&array![[0.0, 0.0], [1.0, 1.0]])
        .unwrap();
    let exp = [[2.5, 25.0], [28.0, 40.0]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (out[[i, j]] - exp[i][j]).abs() < 1e-9,
                "holdout[{i},{j}] = {} != sklearn {}",
                out[[i, j]],
                exp[i][j]
            );
        }
    }
}

#[test]
fn req7_inverse_nan_passthrough() {
    // Live oracle: fit([[1],[2],[3],[4]]) center_=2.5; inverse([[nan],[0]])=[[nan],[2.5]].
    let fitted = RobustScaler::<f64>::new()
        .fit(&array![[1.0], [2.0], [3.0], [4.0]], &())
        .unwrap();
    let out = fitted
        .inverse_transform(&array![[f64::NAN], [0.0]])
        .unwrap();
    assert!(out[[0, 0]].is_nan(), "nan must pass through inverse");
    assert!(
        (out[[1, 0]] - 2.5).abs() < 1e-9,
        "0 -> center 2.5, got {}",
        out[[1, 0]]
    );
}

#[test]
fn req7_inverse_with_flags_false() {
    let x = fixture_4x2();
    // with_centering=false: inverse = x*scale_ (no +center).
    let f_nocenter = RobustScaler::<f64>::new()
        .with_with_centering(false)
        .fit(&x, &())
        .unwrap();
    let xt = f_nocenter.transform(&x).unwrap();
    let xinv = f_nocenter.inverse_transform(&xt).unwrap();
    for ((i, j), &orig) in x.indexed_iter() {
        assert!(
            (xinv[[i, j]] - orig).abs() < 1e-9,
            "nocenter roundtrip[{i},{j}]"
        );
    }
    // with_scaling=false: inverse = x + center (no *scale).
    let f_noscale = RobustScaler::<f64>::new()
        .with_with_scaling(false)
        .fit(&x, &())
        .unwrap();
    let xt2 = f_noscale.transform(&x).unwrap();
    let xinv2 = f_noscale.inverse_transform(&xt2).unwrap();
    for ((i, j), &orig) in x.indexed_iter() {
        assert!(
            (xinv2[[i, j]] - orig).abs() < 1e-9,
            "noscale roundtrip[{i},{j}]"
        );
    }
}

#[test]
fn req7_inverse_inf_rejected() {
    let fitted = RobustScaler::<f64>::new().fit(&fixture_4x2(), &()).unwrap();
    assert!(
        fitted
            .inverse_transform(&array![[f64::INFINITY, 0.0]])
            .is_err(),
        "inverse_transform must reject +inf (allow-nan)"
    );
}

// ===========================================================================
// #2305-ANALOG residual: f32 transform per-op rounding uses the f64 median
// instead of sklearn's float32 `center_`.
//
// sklearn's `np.nanmedian` of a float32 column returns a *float32* `center_`
// (`_data.py:1614`); only `np.nanpercentile` upcasts `scale_` to float64
// (`:1630`,`:1634`). So sklearn's transform computes `X(f32) -= center_(f32)`
// (`:1672-1673`) — the subtraction operand is the f32-ROUNDED median. ferrolearn
// stores `median`/`center_` as f64 (the #2206 fix upcast EVERYTHING) and
// subtracts the UNROUNDED f64 median in `transform` (robust_scaler.rs:522),
// rounding to f32 only afterward. When the f64 median is not exactly
// f32-representable, the f32-rounded subtraction result differs by up to a few
// ULP — exactly the StandardScaler #2305 residual, here on `center_`'s dtype.
// ===========================================================================

/// Divergence: ferrolearn's `FittedRobustScaler::<f32>::transform`
/// (`robust_scaler.rs:519-528`) subtracts the **f64** stored median, whereas
/// scikit-learn subtracts the **float32** `center_` returned by `np.nanmedian`
/// (`sklearn/preprocessing/_data.py:1614`,`:1672-1673`). For a column whose
/// f64 median is not exactly f32-representable, sklearn's
/// `X(f32) -= center_(f32)` rounds the operand FIRST, giving a different f32
/// result than ferrolearn's `f32(x_f64 - median_f64)`.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   X = np.array([[0.1],[0.2],[0.3],[0.4]], dtype=np.float32)
///   r = RobustScaler().fit(X)              # center_ dtype == float32, == 0.25
///                                          # scale_  dtype == float64
///   r.transform(X.copy())  -> f32 bits (view uint32):
///     row0 = 3212836864 (-1.0)
///     row1 = 3198855849 (-0.333333283662796)
///     row2 = 1051372205 ( 0.33333340287208557)
///     row3 = 1065353216 ( 1.0)
/// ferrolearn (f64 median 0.2500000074505806 subtracted, then f32-rounded)
/// returns row1 == 3198855851, row2 == 1051372203 — 2 ULP off on rows 1 and 2.
/// (Masked by the 1e-5 tolerance the other f32 tests use; visible only at the
/// bit level, like the StandardScaler #2305 residual.)
///
/// Tracking: #2306
#[test]
#[ignore = "divergence: f32 transform subtracts f64 median, not sklearn's f32 center_ (#2305-analog); tracking #2306"]
fn divergence_f32_transform_uses_f64_median_not_f32_center() {
    let x = array![[0.1f32], [0.2], [0.3], [0.4]];
    let fitted = RobustScaler::<f32>::new().fit(&x, &()).unwrap();
    let out = fitted.transform(&x).unwrap();

    // LIVE sklearn 1.5.2 oracle f32 transform bits (np.nanmedian center_ is
    // float32 == 0.25; X(f32) -= center_(f32) rounds the operand first).
    let sk_bits: [u32; 4] = [3_212_836_864, 3_198_855_849, 1_051_372_205, 1_065_353_216];
    for i in 0..4 {
        let got = out[[i, 0]].to_bits();
        assert_eq!(
            got, sk_bits[i],
            "f32 transform row {i}: sklearn bits {} ({}), ferrolearn bits {} ({})",
            sk_bits[i],
            f32::from_bits(sk_bits[i]),
            got,
            out[[i, 0]]
        );
    }
}
