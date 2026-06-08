//! Divergence tests (EDGES): ferrolearn `PowerTransformer` (Yeo-Johnson) vs
//! scikit-learn 1.5.2 `sklearn/preprocessing/_data.py` `class PowerTransformer`.
//!
//! The headline DIV-1 (missing `np.sign(x)` Jacobian factor) is already pinned
//! and FIXED in `divergence_power_transformer.rs`. This file pins the
//! UN-AUDITED edge-case VALUE divergences that remain (REQ-6 constant feature /
//! StandardScaler zero-scale, REQ-7 NaN-drop). All expected values come from the
//! LIVE sklearn 1.5.2 oracle (run from /tmp) — never copied from ferrolearn
//! (R-CHAR-3).
//!
//! These edges are now FIXED (REQ-6 constant-feature/zero-scale + REQ-7
//! NaN-drop): all 6 pins are active `#[test]`s and pass against the oracle.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::PowerTransformer;
use ndarray::array;

// ===========================================================================
// Live sklearn 1.5.2 oracle constants (run from /tmp; hard-coded — R-CHAR-3)
// ===========================================================================
//
// Constant feature X = [[3],[3],[3],[3]], method='yeo-johnson':
//   python3 -c "import numpy as np; from sklearn.preprocessing import PowerTransformer; \
//   X=np.array([[3.],[3.],[3.],[3.]]); \
//   pt=PowerTransformer(method='yeo-johnson', standardize=False).fit(X); \
//   print(pt.lambdas_[0], pt.transform(X).ravel().tolist())"
//     std=False -> lambda = 1.0 ; xform = [3.0, 3.0, 3.0, 3.0]
//     std=True  -> lambda = 1.0 ; xform = [0.0, 0.0, 0.0, 0.0]
// sklearn `_data.py:3298-3302`: constant features get lambda=1.0 (identity) and
// are skipped; `_data.py:3308` StandardScaler `_handle_zeros_in_scale` makes the
// zero std become 1 so the centered constant column is exactly 0.
const SK_CONST_LAMBDA: f64 = 1.0;

// NaN fixture X = [[1],[2],[nan],[4],[5]], method='yeo-johnson', std=False:
//   -> lambda = 0.502119082186996   (computed on the 4 FINITE values; NaN dropped)
//   -> xform  = [0.829070934, 1.46596262, NaN, 2.476916773, 2.905302558]
// sklearn `_data.py:3491` drops NaN (`x = x[~np.isnan(x)]`) before the brent MLE;
// the NaN passes THROUGH the transform (force_all_finite='allow-nan').
const SK_NAN_LAMBDA: f64 = 0.502_119_082_186_996;
const SK_NAN_XFORM_FINITE: [(usize, f64); 4] = [
    (0, 0.829_070_934),
    (1, 1.465_962_620),
    (3, 2.476_916_773),
    (4, 2.905_302_558),
];

// ===========================================================================
// REQ-6 — constant feature: sklearn lambda=1.0 (identity skip); ferrolearn
// runs the Brent optimizer on a zero-variance column (log-likelihood is -inf
// for every lambda) and wanders to the interval endpoint ~3.0.
// ===========================================================================

/// Divergence (REQ-6): a CONSTANT feature. sklearn `_data.py:3298-3302` sets
/// `lambdas_[i] = 1.0` (identity) and skips optimization. ferrolearn
/// (`power_transformer.rs:260-289`) has no constant check, so Brent on the
/// degenerate (variance=0 ⇒ -inf) objective lands on ~2.99999996.
/// sklearn λ=1.0; ferrolearn λ≈3.0. Tracking blocker #1347.
#[test]
fn divergence_constant_feature_lambda() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[3.0], [3.0], [3.0], [3.0]];
    let fitted = pt.fit(&x, &()).expect("fit must succeed");
    let got = fitted.lambdas()[0];
    assert!(
        (got - SK_CONST_LAMBDA).abs() < 1e-6,
        "constant-feature lambda diverges: ferrolearn={got}, sklearn={SK_CONST_LAMBDA} (_data.py:3298-3302)"
    );
}

/// Divergence (REQ-6): a CONSTANT feature, standardize=False. With the identity
/// lambda=1.0 sklearn leaves the column unchanged → [3,3,3,3]. ferrolearn's
/// wrong λ≈3.0 maps each 3.0 to ((3+1)^3 - 1)/3 ≈ 21.0.
/// sklearn xform=[3,3,3,3]; ferrolearn≈[21,21,21,21]. Tracking blocker #1347.
#[test]
fn divergence_constant_feature_transform_no_std() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[3.0], [3.0], [3.0], [3.0]];
    let fitted = pt.fit(&x, &()).expect("fit must succeed");
    let out = fitted.transform(&x).expect("transform must succeed");
    for i in 0..4 {
        let got = out[[i, 0]];
        assert!(
            (got - 3.0).abs() < 1e-6,
            "constant-feature transform row {i} diverges: ferrolearn={got}, sklearn=3.0 (_data.py:3298-3302)"
        );
    }
}

/// Divergence (REQ-6): a CONSTANT feature, standardize=True. sklearn applies
/// `StandardScaler` whose `_handle_zeros_in_scale` turns the zero std into 1, so
/// the centered constant column is exactly 0 → [0,0,0,0]. ferrolearn
/// (`power_transformer.rs:362` `if s > 0`) NEVER subtracts the mean when std==0,
/// so it returns the raw (wrong-λ) transform ≈21.0.
/// sklearn xform=[0,0,0,0]; ferrolearn≈[21,21,21,21]. Tracking blocker #1347.
#[test]
fn divergence_constant_feature_transform_standardize() {
    let pt = PowerTransformer::<f64>::new(); // standardize=true (default)
    let x = array![[3.0], [3.0], [3.0], [3.0]];
    let fitted = pt.fit(&x, &()).expect("fit must succeed");
    let out = fitted.transform(&x).expect("transform must succeed");
    for i in 0..4 {
        let got = out[[i, 0]];
        assert!(
            got.abs() < 1e-6,
            "constant-feature std=True row {i} diverges: ferrolearn={got}, sklearn=0.0 (_data.py:3308 _handle_zeros_in_scale)"
        );
    }
}

/// Divergence (REQ-6): a SINGLE sample, standardize=True. sklearn treats it as a
/// constant feature → lambda=1.0, and StandardScaler centers it to 0.0.
/// ferrolearn → λ≈3.0, transform ≈71.67.
/// sklearn xform=[0.0]; ferrolearn≈[71.67]. Tracking blocker #1347.
#[test]
fn divergence_single_sample_standardize() {
    let pt = PowerTransformer::<f64>::new();
    let x = array![[5.0]];
    let fitted = pt.fit(&x, &()).expect("fit must succeed");
    let out = fitted.transform(&x).expect("transform must succeed");
    let got = out[[0, 0]];
    assert!(
        got.abs() < 1e-6,
        "single-sample std=True diverges: ferrolearn={got}, sklearn=0.0 (_data.py:3298-3302)"
    );
}

// ===========================================================================
// REQ-7 — NaN handling: sklearn drops NaN before the MLE and passes NaN through
// the transform; ferrolearn maps NaN→0.0 (`power_transformer.rs:266`
// `.to_f64().unwrap_or(0.0)`), poisoning the MLE with a spurious 0.0 sample.
// ===========================================================================

/// Divergence (REQ-7): a column containing NaN. sklearn `_data.py:3491` drops
/// NaN (`x = x[~np.isnan(x)]`) so the brent MLE runs on the 4 finite values →
/// λ=0.502119. ferrolearn (`power_transformer.rs:266`) maps NaN→0.0, injecting a
/// spurious sample into the objective → a different λ (≈-0.708).
/// Tracking blocker #1348.
#[test]
fn divergence_nan_dropped_in_mle_lambda() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[1.0], [2.0], [f64::NAN], [4.0], [5.0]];
    let fitted = pt.fit(&x, &()).expect("fit must succeed");
    let got = fitted.lambdas()[0];
    assert!(
        (got - SK_NAN_LAMBDA).abs() < 1e-5,
        "NaN-fixture lambda diverges: ferrolearn={got}, sklearn={SK_NAN_LAMBDA} (_data.py:3491 NaN drop)"
    );
}

/// Divergence (REQ-7): the NaN-column transform. sklearn keeps the NaN row as
/// NaN and transforms the finite rows with the correct (NaN-dropped) λ →
/// [0.829..., 1.466..., NaN, 2.477..., 2.905...]. ferrolearn's poisoned λ makes
/// every finite row wrong (and only by side-effect leaves NaN as NaN).
/// Tracking blocker #1348.
#[test]
fn divergence_nan_transform_finite_rows() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[1.0], [2.0], [f64::NAN], [4.0], [5.0]];
    let fitted = pt.fit(&x, &()).expect("fit must succeed");
    let out = fitted.transform(&x).expect("transform must succeed");
    // NaN row must stay NaN (this part ferrolearn happens to satisfy).
    assert!(out[[2, 0]].is_nan(), "NaN row must remain NaN");
    for (i, sk) in SK_NAN_XFORM_FINITE {
        let got = out[[i, 0]];
        assert!(
            (got - sk).abs() < 1e-5,
            "NaN-fixture transform row {i} diverges: ferrolearn={got}, sklearn={sk} (_data.py:3491)"
        );
    }
}
