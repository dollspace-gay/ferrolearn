//! Divergence tests: ferrolearn `PowerTransformer` (Yeo-Johnson) vs
//! scikit-learn 1.5.2 `sklearn/preprocessing/_data.py` `class PowerTransformer`
//! (`:3122`).
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (run from /tmp)
//! or a sklearn `file:line` symbolic constant — never copied from the
//! ferrolearn side (R-CHAR-3).
//!
//! HEADLINE DIV-1 (REQ-1): `fn log_likelihood_yj` (`power_transformer.rs`
//! ~line 104-108) computes the Yeo-Johnson Jacobian term as
//! `(lambda - 1) * sum((|y| + 1).ln())` — it OMITS the `np.sign(x)` factor that
//! sklearn `_neg_log_likelihood` carries at
//! `_data.py:3485`: `loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()`.
//! On data containing NEGATIVE values the estimated lambda diverges grossly
//! (ferrolearn ~2.042 vs sklearn ~0.950). The two `divergence_div1_*` tests
//! below PIN this; they FAIL today and are `#[ignore]`d under tracking #1343.
//!
//! GREEN guards pin the all-positive path (where `sign(x) == 1` so DIV-1 drops
//! out) and the scoped error contracts (REQ-2); these PASS today.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::{PowerTransformer, power_transform};
use ndarray::{Array2, array};

// ===========================================================================
// Live sklearn 1.5.2 oracle constants (run from /tmp; hard-coded — R-CHAR-3)
// ===========================================================================
//
// Signed fixture X = [[-2],[-1],[0],[1],[2],[3]], method='yeo-johnson',
// standardize=False:
//   python3 -c "import numpy as np; from sklearn.preprocessing import PowerTransformer; \
//   X=np.array([[-2.],[-1.],[0.],[1.],[2.],[3.]]); \
//   pt=PowerTransformer(method='yeo-johnson', standardize=False).fit(X); \
//   print(pt.lambdas_[0]); print(np.round(pt.transform(X).ravel(),9).tolist())"
//     -> lambda = 0.9504965354909566
//     -> xform  = [-2.065427663, -1.019355712, 0.0, 0.981105806, 1.937094826, 2.877129732]
const SK_NEG_LAMBDA: f64 = 0.950_496_535_490_956_6; // _data.py:3464 _yeo_johnson_optimize
const SK_NEG_XFORM: [f64; 6] = [
    -2.065_427_663,
    -1.019_355_712,
    0.0,
    0.981_105_806,
    1.937_094_826,
    2.877_129_732,
];

// All-positive fixture X = [[1],[2],[3],[4],[5]], standardize=False:
//   -> lambda = 0.699807422455043
//   -> xform  = [0.892085366, 1.65361612, 2.341088887, 2.978266838, 3.578034121]
const SK_POS_LAMBDA: f64 = 0.699_807_422_455_043;
const SK_POS_XFORM: [f64; 5] = [
    0.892_085_366,
    1.653_616_12,
    2.341_088_887,
    2.978_266_838,
    3.578_034_121,
];

// All-positive fixture, standardize=True (StandardScaler ddof=0):
//   -> xform = [-1.472976434, -0.669760947, 0.055342762, 0.727398612, 1.359996006]
//   -> mean  = -1.33e-16 (~0)
const SK_POS_STD_XFORM: [f64; 5] = [
    -1.472_976_434,
    -0.669_760_947,
    0.055_342_762,
    0.727_398_612,
    1.359_996_006,
];

// ===========================================================================
// RED PINS — DIV-1: missing np.sign(x) factor in the YJ Jacobian
// ===========================================================================

/// Divergence DIV-1 (REQ-1): ferrolearn's `fn log_likelihood_yj`
/// (`power_transformer.rs` ~line 104-108) computes the Jacobian as
/// `(lambda - 1) * sum((|y| + 1).ln())`, OMITTING the `np.sign(x)` factor that
/// sklearn `_neg_log_likelihood` carries at `_data.py:3485`:
/// `loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()`.
/// On signed data the MLE lambda diverges: sklearn returns ~0.950497,
/// ferrolearn returns ~2.042041.
/// Tracking: #1342, blocker #1343.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn divergence_div1_signed_lambda() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]];
    let fitted = match pt.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let got = fitted.lambdas()[0];
    assert!(
        (got - SK_NEG_LAMBDA).abs() < 1e-4,
        "signed-fixture lambda diverges: ferrolearn={got}, sklearn={SK_NEG_LAMBDA} (_data.py:3485, missing sign(x))"
    );
}

/// Divergence DIV-1 (REQ-1): the signed-fixture transform column diverges from
/// the sklearn oracle because the wrong MLE lambda is fed into the (otherwise
/// correct) `fn yeo_johnson` transform. sklearn row 0 is `-2.065428`,
/// ferrolearn ~`-1.073628`.
/// sklearn cite: `_data.py:3485` (jacobian) -> wrong lambda -> `:3426-3446`
/// (`_yeo_johnson_transform`).
/// Tracking: #1342, blocker #1343.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn divergence_div1_signed_transform() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]];
    let fitted = match pt.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform unexpectedly failed: {e:?}");
            return;
        }
    };
    for (i, &sk) in SK_NEG_XFORM.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - sk).abs() < 1e-5,
            "signed-fixture transform row {i} diverges: ferrolearn={got}, sklearn={sk} (_data.py:3485)"
        );
    }
}

// ===========================================================================
// GREEN GUARDS — REQ-1 all-positive path (sign(x)==1 ⇒ DIV-1 vanishes) + REQ-2
// ===========================================================================

/// Green guard (REQ-1): on the all-positive fixture `sign(x) == 1` so the
/// missing sign factor drops out; ferrolearn's MLE lambda converges to the
/// sklearn oracle (`_data.py:3464` `_yeo_johnson_optimize`). PASSES today.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_positive_lambda_matches_oracle() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let fitted = match pt.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let got = fitted.lambdas()[0];
    assert!(
        (got - SK_POS_LAMBDA).abs() < 1e-4,
        "positive lambda mismatch: ferrolearn={got}, sklearn={SK_POS_LAMBDA}"
    );
}

/// Green guard (REQ-1): the all-positive transform column matches the sklearn
/// oracle (`_yeo_johnson_transform`, `_data.py:3426-3446`). PASSES today.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_positive_transform_matches_oracle() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let fitted = match pt.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform unexpectedly failed: {e:?}");
            return;
        }
    };
    for (i, &sk) in SK_POS_XFORM.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - sk).abs() < 1e-5,
            "positive transform row {i}: ferrolearn={got}, sklearn={sk}"
        );
    }
}

/// Green guard (REQ-1): with `standardize=True` on the positive fixture the
/// transformed column matches the sklearn `StandardScaler(ddof=0)` oracle and
/// has ~zero mean (`_data.py:3308-3314`). PASSES today.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_positive_standardize_zero_mean() {
    let pt = PowerTransformer::<f64>::new(); // standardize=true (default)
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let fitted = match pt.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform unexpectedly failed: {e:?}");
            return;
        }
    };
    for (i, &sk) in SK_POS_STD_XFORM.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - sk).abs() < 1e-5,
            "positive std transform row {i}: ferrolearn={got}, sklearn={sk}"
        );
    }
    let mean: f64 = out.column(0).iter().sum::<f64>() / out.nrows() as f64;
    assert!(
        mean.abs() < 1e-9,
        "standardized column mean should be ~0, got {mean}"
    );
}

/// Green guard (REQ-2): `fit` on zero-sample input returns an `Err`
/// (mirrors sklearn `_check_input` shape validation, `_data.py:3531-3537`).
#[test]
fn green_zero_samples_errors() {
    let pt = PowerTransformer::<f64>::new();
    let x: Array2<f64> = Array2::zeros((0, 2));
    assert!(
        pt.fit(&x, &()).is_err(),
        "fit on zero samples must return Err"
    );
}

/// Green guard (REQ-2): `transform` with a mismatched column count returns an
/// `Err` (mirrors `_check_input` check_shape, `_data.py:3531-3537`).
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_transform_ncols_mismatch_errors() {
    let pt = PowerTransformer::<f64>::new();
    let x_train = array![[1.0, 2.0], [3.0, 4.0]];
    let fitted = match pt.fit(&x_train, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let x_bad = array![[1.0, 2.0, 3.0]];
    assert!(
        fitted.transform(&x_bad).is_err(),
        "transform with wrong ncols must return Err"
    );
}

/// Green guard (REQ-2): the UNFITTED transformer's `transform` returns an `Err`
/// (mirrors sklearn `check_is_fitted`, `_data.py:3331`).
#[test]
fn green_unfitted_transform_errors() {
    let pt = PowerTransformer::<f64>::new();
    let x = array![[1.0, 2.0]];
    assert!(
        pt.transform(&x).is_err(),
        "unfitted transform must return Err"
    );
}

// ===========================================================================
// REQ-5 — power_transform free function (Yeo-Johnson surface)
// ===========================================================================

/// Green guard (REQ-5): `power_transform(X, standardize=true)` delegates to the
/// default Yeo-Johnson `PowerTransformer().fit_transform(X)` wrapper shape
/// (`_data.py:3741-3742`). Live sklearn oracle is the all-positive standardized
/// fixture above.
#[test]
fn green_req5_power_transform_standardized_matches_sklearn() {
    let x = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
    let out = power_transform(&x, true).unwrap();
    for (i, &sk) in SK_POS_STD_XFORM.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - sk).abs() < 1e-5,
            "power_transform standardized row {i}: ferrolearn={got}, sklearn={sk}"
        );
    }
}

/// Green guard (REQ-5): `standardize=false` is threaded to the transformer,
/// matching sklearn `PowerTransformer(standardize=False).fit_transform(X)`.
#[test]
fn green_req5_power_transform_without_standardize_matches_sklearn() {
    let x = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
    let out = power_transform(&x, false).unwrap();
    for (i, &sk) in SK_POS_XFORM.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - sk).abs() < 1e-5,
            "power_transform unstandardized row {i}: ferrolearn={got}, sklearn={sk}"
        );
    }
}

/// Green guard (REQ-5): functional wrapper propagates fit validation errors.
#[test]
fn green_req5_power_transform_zero_samples_errors() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    assert!(
        power_transform(&x, true).is_err(),
        "power_transform on zero samples must return Err"
    );
}

// ===========================================================================
// RE-AUDIT GREEN GUARDS — post-fix faithfulness of the YJ Jacobian sign factor
// across MORE signed-data configurations (all values from live sklearn 1.5.2
// oracle, method='yeo-johnson', run from /tmp, hard-coded — R-CHAR-3).
// These PASS today, confirming DIV-1 (#1342/#1343) is fixed beyond the single
// pinned fixture.
// ===========================================================================

// (a) all-NEGATIVE data X = [[-5],[-4],[-3],[-2],[-1]], standardize=False:
//   python3 -c "import numpy as np; from sklearn.preprocessing import PowerTransformer; \
//   X=np.array([[-5.],[-4.],[-3.],[-2.],[-1.]]); \
//   pt=PowerTransformer(method='yeo-johnson',standardize=False).fit(X); \
//   print(pt.lambdas_.tolist()); print(pt.transform(X).ravel().tolist())"
//   -> lambdas_ = [1.3001925976525683]
//   -> xform    = [-3.5780340436, -2.9782667812, -2.3410888492, -1.653616099, -0.8920853596]
const SK_A_NEG_LAMBDA: f64 = 1.300_192_597_652_568_3; // _data.py:3464 _yeo_johnson_optimize
const SK_A_NEG_XFORM: [f64; 5] = [
    -3.578_034_043_6,
    -2.978_266_781_2,
    -2.341_088_849_2,
    -1.653_616_099_0,
    -0.892_085_359_6,
];

/// Re-audit (a): all-NEGATIVE data exercises the `y < 0` branch of the YJ
/// transform AND the `np.sign(x) == -1` branch of the restored Jacobian
/// (`_data.py:3485`). sklearn lambda ~1.300193, transform col as above.
/// Tracking: #1342, blocker #1343 (verifying fix faithfulness).
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_all_negative_lambda_and_transform() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[-5.0], [-4.0], [-3.0], [-2.0], [-1.0]];
    let fitted = match pt.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let got_lambda = fitted.lambdas()[0];
    assert!(
        (got_lambda - SK_A_NEG_LAMBDA).abs() < 1e-4,
        "all-negative lambda diverges: ferrolearn={got_lambda}, sklearn={SK_A_NEG_LAMBDA} (_data.py:3485)"
    );
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform unexpectedly failed: {e:?}");
            return;
        }
    };
    for (i, &sk) in SK_A_NEG_XFORM.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - sk).abs() < 1e-5,
            "all-negative transform row {i} diverges: ferrolearn={got}, sklearn={sk}"
        );
    }
}

// (b) mixed-sign larger spread X = [[-10],[-3],[-0.5],[0],[0.5],[3],[10],[25]]:
//   -> lambdas_ = [0.7938572581077992]
//   -> xform    = [-14.1218859277, -3.58429336, -0.5229599056, 0.0,
//                   0.4783242247, 2.5265601548, 7.192624168, 15.4721961901]
const SK_B_LAMBDA: f64 = 0.793_857_258_107_799_2;
const SK_B_XFORM: [f64; 8] = [
    -14.121_885_927_7,
    -3.584_293_360_0,
    -0.522_959_905_6,
    0.0,
    0.478_324_224_7,
    2.526_560_154_8,
    7.192_624_168_0,
    15.472_196_190_1,
];

/// Re-audit (b): mixed-sign data with larger spread (asymmetric, includes 0).
/// Exercises both Jacobian sign branches plus a zero (sign(0)==0).
/// sklearn lambda ~0.793857. Tracking: #1342, blocker #1343.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_mixed_sign_spread_lambda_and_transform() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[-10.0], [-3.0], [-0.5], [0.0], [0.5], [3.0], [10.0], [25.0]];
    let fitted = match pt.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let got_lambda = fitted.lambdas()[0];
    assert!(
        (got_lambda - SK_B_LAMBDA).abs() < 1e-4,
        "mixed-spread lambda diverges: ferrolearn={got_lambda}, sklearn={SK_B_LAMBDA} (_data.py:3485)"
    );
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform unexpectedly failed: {e:?}");
            return;
        }
    };
    for (i, &sk) in SK_B_XFORM.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - sk).abs() < 1e-5,
            "mixed-spread transform row {i} diverges: ferrolearn={got}, sklearn={sk}"
        );
    }
}

// (c) MULTI-FEATURE, distinct sign pattern per column:
//   col0 = [1,2,3,4,5]      (all positive)
//   col1 = [-5,-1,0,2,8]    (mixed sign + zero)
//   col2 = [-1,-2,-3,-4,-5] (all negative)
//   -> lambdas_ = [0.699807422455043, 0.8452305490573978, 1.3001926287656835]
//   -> xform (row-major, 5x3) below
const SK_C_LAMBDAS: [f64; 3] = [
    0.699_807_422_455_043,
    0.845_230_549_057_397_8,
    1.300_192_628_765_683_5,
];
const SK_C_XFORM: [[f64; 3]; 5] = [
    [0.892_085_366_4, -5.990_329_338_2, -0.892_085_349_2],
    [1.653_616_119_6, -1.062_105_864_1, -1.653_616_067_1],
    [2.341_088_887_0, 0.0, -2.341_088_790_6],
    [2.978_266_838_2, 1.811_238_863_7, -2.978_266_692_9],
    [3.578_034_121_2, 6.395_329_420_3, -3.578_033_923_6],
];

/// Re-audit (c): multi-feature fit must estimate an INDEPENDENT lambda per
/// column (positive / mixed / negative sign patterns) and assemble the
/// per-column transform block (`_data.py:3464` per-column loop + `:3426`
/// transform). Catches any cross-column leakage of the Jacobian sign factor.
/// Tracking: #1342, blocker #1343.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_multifeature_per_column_lambda_and_block() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![
        [1.0, -5.0, -1.0],
        [2.0, -1.0, -2.0],
        [3.0, 0.0, -3.0],
        [4.0, 2.0, -4.0],
        [5.0, 8.0, -5.0],
    ];
    let fitted = match pt.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    for (j, &sk) in SK_C_LAMBDAS.iter().enumerate() {
        let got = fitted.lambdas()[j];
        assert!(
            (got - sk).abs() < 1e-4,
            "multifeature col {j} lambda diverges: ferrolearn={got}, sklearn={sk} (_data.py:3485)"
        );
    }
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform unexpectedly failed: {e:?}");
            return;
        }
    };
    for (i, row) in SK_C_XFORM.iter().enumerate() {
        for (j, &sk) in row.iter().enumerate() {
            let got = out[[i, j]];
            assert!(
                (got - sk).abs() < 1e-5,
                "multifeature transform [{i},{j}] diverges: ferrolearn={got}, sklearn={sk}"
            );
        }
    }
}

// (d) standardize=True on signed fixture X = [[-2],[-1],[0],[1],[2],[3]]:
//   StandardScaler(ddof=0) applied AFTER the YJ transform.
//   -> lambdas_ = [0.9504965354909566]
//   -> xform    = [-1.492215375, -0.8720923638, -0.2678070341,
//                   0.3138033282, 0.8805241827, 1.4377872621]
const SK_D_STD_LAMBDA: f64 = 0.950_496_535_490_956_6;
const SK_D_STD_XFORM: [f64; 6] = [
    -1.492_215_375_0,
    -0.872_092_363_8,
    -0.267_807_034_1,
    0.313_803_328_2,
    0.880_524_182_7,
    1.437_787_262_1,
];

/// Re-audit (d): standardize=True on SIGNED data — the YJ transform (with the
/// restored sign-factor lambda) followed by `StandardScaler(ddof=0)`
/// (`_data.py:3308-3314`). Verifies both the signed lambda AND the
/// post-transform standardization column. Tracking: #1342, blocker #1343.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_signed_standardize_transform() {
    let pt = PowerTransformer::<f64>::new(); // standardize=true
    let x = array![[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]];
    let fitted = match pt.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let got_lambda = fitted.lambdas()[0];
    assert!(
        (got_lambda - SK_D_STD_LAMBDA).abs() < 1e-4,
        "signed-standardize lambda diverges: ferrolearn={got_lambda}, sklearn={SK_D_STD_LAMBDA}"
    );
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform unexpectedly failed: {e:?}");
            return;
        }
    };
    for (i, &sk) in SK_D_STD_XFORM.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - sk).abs() < 1e-5,
            "signed-standardize transform row {i} diverges: ferrolearn={got}, sklearn={sk} (_data.py:3308)"
        );
    }
}

// (e) NEGATIVE optimal lambda: heavily right-skewed positive data
//   X = [[1],[1],[1],[2],[10],[100]] -> sklearn brent(brack=(-2,2)) lands < 0.
//   -> lambdas_ = [-0.7252485461394542]
//   -> xform    = [0.5447886477, 0.5447886477, 0.5447886477,
//                   0.7572796826, 1.1365983489, 1.3303219856]
const SK_E_NEG_LAMBDA: f64 = -0.725_248_546_139_454_2;
const SK_E_XFORM: [f64; 6] = [
    0.544_788_647_7,
    0.544_788_647_7,
    0.544_788_647_7,
    0.757_279_682_6,
    1.136_598_348_9,
    1.330_321_985_6,
];

/// Re-audit (e): the MLE optimum lambda is NEGATIVE (~-0.7252). Confirms
/// ferrolearn's bounded-Brent on `[-3,3]` still lands on sklearn's
/// `optimize.brent(brack=(-2,2))` minimum (`_data.py:3493`) when it is < 0,
/// rather than clamping at an interval endpoint. Tracking: #1342, blocker #1343.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_negative_optimal_lambda() {
    let pt = PowerTransformer::<f64>::without_standardize();
    let x = array![[1.0], [1.0], [1.0], [2.0], [10.0], [100.0]];
    let fitted = match pt.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let got_lambda = fitted.lambdas()[0];
    assert!(
        (got_lambda - SK_E_NEG_LAMBDA).abs() < 1e-4,
        "negative-lambda fixture diverges: ferrolearn={got_lambda}, sklearn={SK_E_NEG_LAMBDA} (_data.py:3493)"
    );
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform unexpectedly failed: {e:?}");
            return;
        }
    };
    for (i, &sk) in SK_E_XFORM.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - sk).abs() < 1e-5,
            "negative-lambda transform row {i} diverges: ferrolearn={got}, sklearn={sk}"
        );
    }
}
