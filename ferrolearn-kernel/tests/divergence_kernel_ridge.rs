//! Divergence + value-parity tests for `KernelRidge` against scikit-learn 1.5.2.
//!
//! Expected values are taken from a LIVE `sklearn.kernel_ridge.KernelRidge`
//! 1.5.2 oracle run from `/tmp` (never copied from ferrolearn), per goal.md
//! R-CHAR-3. Each test maps to a REQ in `.design/kernel/kernel_ridge.md`.
//!
//! Fixed fixtures shared by all tests below:
//!   X (8x4), y (8,), Xtest (3x4) — identical to the Python oracle invocation.
//!
//! Tracking: #1661.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_kernel::{KernelRidge, KernelType};
use ndarray::{Array1, Array2, array};

/// Shared 8x4 training matrix (matches the Python oracle exactly).
fn x_train() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 4),
        vec![
            0.1, -0.5, 1.2, 0.3, //
            1.0, 0.2, -0.7, 0.8, //
            -0.3, 1.1, 0.4, -0.2, //
            0.6, -1.0, 0.9, 1.5, //
            -1.2, 0.5, -0.1, 0.7, //
            0.4, 0.9, -1.3, -0.6, //
            1.1, -0.4, 0.2, 0.0, //
            -0.8, 0.3, 1.0, -1.1, //
        ],
    )
    .unwrap()
}

fn y_train() -> Array1<f64> {
    array![1.0, -2.0, 0.5, 3.0, -1.5, 0.2, 2.2, -0.7]
}

fn x_test() -> Array2<f64> {
    Array2::from_shape_vec(
        (3, 4),
        vec![
            0.0, 0.0, 0.0, 0.0, //
            0.5, -0.5, 0.5, -0.5, //
            1.0, 1.0, -1.0, -1.0, //
        ],
    )
    .unwrap()
}

// ===========================================================================
// REQ-5 — coef0 default divergence (FAILING test, release blocker)
// ===========================================================================

/// Divergence: `KernelRidge::new()` sets `coef0 = 0.0`
/// (`ferrolearn-kernel/src/kernel_ridge.rs:86` `coef0: F::zero()`), but
/// scikit-learn defaults `coef0 = 1` (`sklearn/kernel_ridge.py:153`
/// `coef0=1,`). For a polynomial kernel at the DEFAULT coef0, ferrolearn
/// computes `(gamma*<x,y> + 0)^degree` while sklearn uses
/// `(gamma*<x,y> + 1)^degree`, yielding different `dual_coef_` and predictions.
///
/// Oracle: `KernelRidge(alpha=1.0, kernel='polynomial', degree=3, gamma=0.5)`
/// (default coef0=1) on (x_train, y_train), predict(x_test):
///   predict = [0.1379504368134285, 1.2336324424391387, 0.8752611140977034]
/// ferrolearn `new().with_kernel(Polynomial)` uses coef0=0 and diverges.
/// Tracking: #1661.
#[test]
fn divergence_poly_default_coef0() {
    let fitted = KernelRidge::<f64>::new()
        .with_alpha(1.0)
        .with_kernel(KernelType::Polynomial)
        .with_degree(3)
        .with_gamma(0.5)
        // NOTE: deliberately NOT calling .with_coef0 — exercise the default.
        .fit(&x_train(), &y_train())
        .unwrap();
    let preds = fitted.predict(&x_test()).unwrap();

    // Live sklearn 1.5.2 oracle (default coef0 == 1):
    let sk_pred = [
        0.137_950_436_813_428_5,
        1.233_632_442_439_138_7,
        0.875_261_114_097_703_4,
    ];
    for i in 0..3 {
        assert!(
            (preds[i] - sk_pred[i]).abs() < 1e-9,
            "poly default-coef0 prediction[{i}] = {} but sklearn (coef0=1) = {}",
            preds[i],
            sk_pred[i]
        );
    }
}

// ===========================================================================
// REQ-10 — negative gamma not rejected (FAILING test, release blocker)
// ===========================================================================

/// Divergence: scikit-learn's `_parameter_constraints`
/// (`sklearn/kernel_ridge.py:140` `"gamma": [Interval(Real, 0, None,
/// closed="left"), None]`) rejects a negative `gamma` with an
/// `InvalidParameterError` at fit time. ferrolearn's `fit`
/// (`ferrolearn-kernel/src/kernel_ridge.rs:333`) only validates `alpha < 0`
/// and accepts a negative `gamma` silently.
/// Tracking: #1661.
#[test]
fn divergence_negative_gamma_not_rejected() {
    let res = KernelRidge::<f64>::new()
        .with_kernel(KernelType::Rbf)
        .with_gamma(-0.5)
        .fit(&x_train(), &y_train());
    assert!(
        res.is_err(),
        "sklearn rejects gamma=-0.5 (InvalidParameterError) but ferrolearn accepted it"
    );
}

// ===========================================================================
// REQ-2 — linear kernel value parity (GREEN guard, should PASS)
// ===========================================================================

/// Confirms REQ-2 SHIPPED: `kernel='linear'`, scalar alpha, single-output y.
/// Oracle: `KernelRidge(alpha=1.0, kernel='linear')`.
/// Tracking: #1661.
#[test]
fn parity_linear_dual_coef_and_predict() {
    let fitted = KernelRidge::<f64>::new()
        .with_alpha(1.0)
        .with_kernel(KernelType::Linear)
        .fit(&x_train(), &y_train())
        .unwrap();

    let sk_dual = [
        -0.217_440_215_194_337_04,
        -2.189_017_759_048_638,
        0.867_706_273_890_853_4,
        1.413_269_282_719_558,
        -0.240_901_864_517_707_9,
        1.244_536_969_605_333_2,
        0.985_805_031_804_373_1,
        -0.711_786_026_569_093_4,
    ];
    let dc = fitted.dual_coef();
    for i in 0..8 {
        assert!(
            (dc[i] - sk_dual[i]).abs() < 1e-9,
            "linear dual_coef[{i}] = {} vs sklearn {}",
            dc[i],
            sk_dual[i]
        );
    }

    let sk_pred = [0.0, 0.999_074_598_707_813_3, -0.358_005_700_224_600_9];
    let preds = fitted.predict(&x_test()).unwrap();
    for i in 0..3 {
        assert!(
            (preds[i] - sk_pred[i]).abs() < 1e-9,
            "linear predict[{i}] = {} vs sklearn {}",
            preds[i],
            sk_pred[i]
        );
    }
}

// ===========================================================================
// REQ-3 — rbf kernel value parity (GREEN guard, should PASS)
// ===========================================================================

/// Confirms REQ-3 SHIPPED: `kernel='rbf'` with the `gamma=None` default
/// (-> 1/n_features) and with an explicit gamma. Oracle:
/// `KernelRidge(alpha=1.0, kernel='rbf')` and `(..., gamma=0.3)`.
/// Tracking: #1661.
#[test]
fn parity_rbf_default_and_explicit_gamma() {
    // default gamma (1/4 = 0.25)
    let fitted_def = KernelRidge::<f64>::new()
        .with_alpha(1.0)
        .with_kernel(KernelType::Rbf)
        .fit(&x_train(), &y_train())
        .unwrap();
    let sk_dual_def = [
        0.013_948_794_081_190_938,
        -1.659_461_773_173_067_6,
        0.540_168_171_920_228_6,
        1.529_198_429_334_185,
        -0.887_361_674_104_070_4,
        0.282_959_273_963_33,
        1.299_532_977_860_967_2,
        -0.528_972_227_568_957_6,
    ];
    let dc = fitted_def.dual_coef();
    for i in 0..8 {
        assert!(
            (dc[i] - sk_dual_def[i]).abs() < 1e-9,
            "rbf default-gamma dual_coef[{i}] = {} vs sklearn {}",
            dc[i],
            sk_dual_def[i]
        );
    }
    let sk_pred_def = [
        0.201_604_504_961_992_9,
        0.819_974_646_664_030_7,
        0.115_358_998_872_679_75,
    ];
    let preds_def = fitted_def.predict(&x_test()).unwrap();
    for i in 0..3 {
        assert!(
            (preds_def[i] - sk_pred_def[i]).abs() < 1e-9,
            "rbf default-gamma predict[{i}] = {} vs sklearn {}",
            preds_def[i],
            sk_pred_def[i]
        );
    }

    // explicit gamma = 0.3
    let fitted_g = KernelRidge::<f64>::new()
        .with_alpha(1.0)
        .with_kernel(KernelType::Rbf)
        .with_gamma(0.3)
        .fit(&x_train(), &y_train())
        .unwrap();
    let sk_dual_g = [
        0.025_294_215_208_685_484,
        -1.565_816_336_965_744_7,
        0.510_972_919_413_176_5,
        1.492_188_648_692_380_8,
        -0.858_055_275_315_233_5,
        0.249_218_659_273_178_68,
        1.273_063_289_328_999_5,
        -0.497_199_500_426_364_85,
    ];
    let dcg = fitted_g.dual_coef();
    for i in 0..8 {
        assert!(
            (dcg[i] - sk_dual_g[i]).abs() < 1e-9,
            "rbf gamma=0.3 dual_coef[{i}] = {} vs sklearn {}",
            dcg[i],
            sk_dual_g[i]
        );
    }
    let sk_pred_g = [
        0.204_254_001_360_805_18,
        0.832_316_792_352_104_5,
        0.124_245_201_042_288_87,
    ];
    let preds_g = fitted_g.predict(&x_test()).unwrap();
    for i in 0..3 {
        assert!(
            (preds_g[i] - sk_pred_g[i]).abs() < 1e-9,
            "rbf gamma=0.3 predict[{i}] = {} vs sklearn {}",
            preds_g[i],
            sk_pred_g[i]
        );
    }
}

// ===========================================================================
// REQ-4 — poly with EXPLICIT coef0=1 (GREEN guard, should PASS)
// ===========================================================================

/// Confirms REQ-4 SHIPPED: with `coef0` supplied explicitly the polynomial
/// formula `(gamma*<x,y> + coef0)^degree` matches sklearn. This isolates the
/// formula from the coef0-default bug (REQ-5). Oracle:
/// `KernelRidge(alpha=1.0, kernel='polynomial', degree=3, gamma=0.5, coef0=1)`.
/// Tracking: #1661.
#[test]
fn parity_poly_explicit_coef0() {
    let fitted = KernelRidge::<f64>::new()
        .with_alpha(1.0)
        .with_kernel(KernelType::Polynomial)
        .with_degree(3)
        .with_gamma(0.5)
        .with_coef0(1.0)
        .fit(&x_train(), &y_train())
        .unwrap();

    let sk_dual = [
        -0.062_856_701_861_316_45,
        -0.395_485_744_489_745_6,
        0.187_939_369_900_790_8,
        0.080_093_563_656_862_58,
        -0.171_376_179_133_626_52,
        0.055_266_440_953_213_45,
        0.519_524_085_480_989_7,
        -0.075_154_397_693_739_48,
    ];
    let dc = fitted.dual_coef();
    for i in 0..8 {
        assert!(
            (dc[i] - sk_dual[i]).abs() < 1e-9,
            "poly coef0=1 dual_coef[{i}] = {} vs sklearn {}",
            dc[i],
            sk_dual[i]
        );
    }
    let sk_pred = [
        0.137_950_436_813_428_5,
        1.233_632_442_439_138_7,
        0.875_261_114_097_703_4,
    ];
    let preds = fitted.predict(&x_test()).unwrap();
    for i in 0..3 {
        assert!(
            (preds[i] - sk_pred[i]).abs() < 1e-9,
            "poly coef0=1 predict[{i}] = {} vs sklearn {}",
            preds[i],
            sk_pred[i]
        );
    }
}
