//! Adversarial value-parity audit of `KernelRidge` vs scikit-learn 1.5.2 (#2363).
//!
//! Targets the un-audited surfaces flagged in #2363: the sigmoid kernel dual
//! solve, the polynomial kernel under the `gamma=None` default, and the
//! single-sample edge. Expected values come from a LIVE
//! `sklearn.kernel_ridge.KernelRidge` 1.5.2 oracle (R-CHAR-3) on the same fixed
//! fixtures used by `divergence_kernel_ridge.rs`; never copied from ferrolearn.
//!
//! Tracking: #2363.

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
// Sigmoid kernel dual_coef_ + predict (gamma=0.3, coef0=1)
// sklearn `sigmoid_kernel`: tanh(gamma*<x,y> + coef0)
// (`sklearn/metrics/pairwise.py` sigmoid_kernel).
// ===========================================================================

/// Audit: `kernel='sigmoid'`, explicit gamma=0.3, coef0=1. Oracle:
/// `KernelRidge(alpha=1.0, kernel='sigmoid', gamma=0.3, coef0=1)` on
/// (x_train, y_train).
/// Tracking: #2363.
#[test]
fn audit_sigmoid_dual_coef_and_predict() {
    let fitted = KernelRidge::<f64>::new()
        .with_alpha(1.0)
        .with_kernel(KernelType::Sigmoid)
        .with_gamma(0.3)
        .with_coef0(1.0)
        .fit(&x_train(), &y_train())
        .unwrap();

    // Live sklearn 1.5.2 oracle.
    let sk_dual = [
        0.409_956_064_355_640_3,
        -3.008_618_299_207_233_4,
        0.645_461_074_774_349_6,
        2.642_699_768_439_386_7,
        -1.386_286_625_198_949_6,
        1.360_482_517_234_840_3,
        1.162_745_775_247_467_3,
        -1.315_496_191_499_959_4,
    ];
    let dc = fitted.dual_coef();
    for i in 0..8 {
        assert!(
            (dc[i] - sk_dual[i]).abs() < 1e-9,
            "sigmoid dual_coef[{i}] = {} vs sklearn {}",
            dc[i],
            sk_dual[i]
        );
    }

    let sk_pred = [
        0.389_132_028_505_414_95,
        0.875_948_035_178_690_1,
        -0.920_949_383_440_249_3,
    ];
    let preds = fitted.predict(&x_test()).unwrap();
    for i in 0..3 {
        assert!(
            (preds[i] - sk_pred[i]).abs() < 1e-9,
            "sigmoid predict[{i}] = {} vs sklearn {}",
            preds[i],
            sk_pred[i]
        );
    }
}

// ===========================================================================
// Polynomial kernel under gamma=None default.
// sklearn polynomial_kernel default gamma -> 1/n_features.
// ===========================================================================

/// Audit: `kernel='poly'`, degree=3, coef0=1, gamma=None (-> 1/n_features).
/// Oracle: `KernelRidge(alpha=1.0, kernel='poly', degree=3, gamma=None,
/// coef0=1)`.
/// Tracking: #2363.
#[test]
fn audit_poly_default_gamma_dual_coef_and_predict() {
    let fitted = KernelRidge::<f64>::new()
        .with_alpha(1.0)
        .with_kernel(KernelType::Polynomial)
        .with_degree(3)
        // gamma deliberately left as default (None -> 1/n_features)
        .fit(&x_train(), &y_train())
        .unwrap();

    let sk_dual = [
        -0.187_522_179_869_650_9,
        -0.996_875_846_119_145_1,
        0.431_438_763_241_382_3,
        0.353_162_743_670_544_57,
        -0.369_676_571_110_973_85,
        0.181_368_365_055_399_7,
        0.949_850_099_105_166_8,
        -0.242_306_538_123_184_95,
    ];
    let dc = fitted.dual_coef();
    for i in 0..8 {
        assert!(
            (dc[i] - sk_dual[i]).abs() < 1e-9,
            "poly default-gamma dual_coef[{i}] = {} vs sklearn {}",
            dc[i],
            sk_dual[i]
        );
    }

    let sk_pred = [
        0.119_438_835_849_538_52,
        1.085_341_732_605_453,
        0.692_197_498_717_279_2,
    ];
    let preds = fitted.predict(&x_test()).unwrap();
    for i in 0..3 {
        assert!(
            (preds[i] - sk_pred[i]).abs() < 1e-9,
            "poly default-gamma predict[{i}] = {} vs sklearn {}",
            preds[i],
            sk_pred[i]
        );
    }
}

// ===========================================================================
// Single-sample dual_coef_ (rbf).
// ===========================================================================

/// Audit: n_samples=1, rbf gamma=1, alpha=1.
/// Oracle: `KernelRidge(alpha=1.0, kernel='rbf', gamma=1.0).fit([[1,2]],[5])`.
/// dual_coef_ = y / (K[0,0] + alpha) = 5 / (1 + 1) = 2.5.
/// Tracking: #2363.
#[test]
fn audit_single_sample_dual_coef() {
    let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    let y = array![5.0f64];
    let fitted = KernelRidge::<f64>::new()
        .with_alpha(1.0)
        .with_kernel(KernelType::Rbf)
        .with_gamma(1.0)
        .fit(&x, &y)
        .unwrap();
    // Live sklearn 1.5.2: dual_coef_ = [2.5]
    let sk_dual = 2.5f64;
    assert!(
        (fitted.dual_coef()[0] - sk_dual).abs() < 1e-12,
        "single-sample dual_coef[0] = {} vs sklearn {}",
        fitted.dual_coef()[0],
        sk_dual
    );
}

// ===========================================================================
// DIVERGENCE — singular kernel solve fallback (FAILING).
// ===========================================================================

/// Divergence: when the regularized kernel `(K + alpha*I)` is singular,
/// scikit-learn's `_solve_cholesky_kernel` catches the `LinAlgError`, emits a
/// "Singular matrix in solving dual problem. Using least-squares solution
/// instead." warning, and falls back to `linalg.lstsq` — returning the
/// minimum-norm least-squares `dual_coef_`
/// (`sklearn/linear_model/_ridge.py:254-259`
/// `except np.linalg.LinAlgError: ... dual_coef = linalg.lstsq(K, y)[0]`).
///
/// ferrolearn's `fit` chains `cholesky_solve(...).or_else(gaussian_solve)`
/// (`ferrolearn-kernel/src/kernel_ridge.rs:400`); `gaussian_solve` returns
/// `Err(NumericalInstability)` on a near-zero pivot
/// (`kernel_ridge.rs:290-294`), so `fit` returns `Err` instead of an
/// lstsq solution.
///
/// Input: X with duplicate rows -> singular linear kernel; alpha=0.
/// Oracle: `KernelRidge(alpha=0.0, kernel='linear').fit(X, y)` succeeds with
///   dual_coef_ ≈ [0.0, 0.0, 0.2]  and  predict(X) ≈ [1.0, 1.0, 2.0].
/// ferrolearn: `fit` returns `Err(NumericalInstability)`.
/// Tracking: #2364.
#[test]
fn divergence_singular_kernel_lstsq_fallback() {
    let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 1.0, 2.0, 3.0, 1.0]).unwrap();
    let y = array![1.0, 1.0, 2.0f64];
    let fitted = KernelRidge::<f64>::new()
        .with_alpha(0.0)
        .with_kernel(KernelType::Linear)
        .fit(&x, &y)
        .expect("sklearn falls back to lstsq and succeeds on a singular kernel");

    // Live sklearn 1.5.2 oracle (minimum-norm lstsq solution).
    let sk_pred = [1.0f64, 1.0, 2.0];
    let preds = fitted.predict(&x).unwrap();
    for i in 0..3 {
        assert!(
            (preds[i] - sk_pred[i]).abs() < 1e-9,
            "singular-kernel predict[{i}] = {} vs sklearn {}",
            preds[i],
            sk_pred[i]
        );
    }
}

/// Second singular-kernel fixture: a different duplicate pattern (rows 0 and 2
/// identical, 4 samples) whose minimum-norm lstsq prediction differs from `y`
/// (no exact interpolant exists). Confirms the lstsq fallback reproduces
/// sklearn's min-norm solution element-wise, not merely "fit succeeds".
///
/// Oracle: `KernelRidge(alpha=0.0, kernel='linear').fit(X, y)`,
/// `X = [[2,0],[0,1],[2,0],[1,1]]`, `y = [3,1,3,4]`.
/// Tracking: #2364.
#[test]
fn divergence_singular_kernel_lstsq_fallback_min_norm() {
    let x = Array2::from_shape_vec((4, 2), vec![2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0]).unwrap();
    let y = array![3.0, 1.0, 3.0, 4.0f64];
    let fitted = KernelRidge::<f64>::new()
        .with_alpha(0.0)
        .with_kernel(KernelType::Linear)
        .fit(&x, &y)
        .expect("sklearn falls back to lstsq and succeeds on a singular kernel");

    // Live sklearn 1.5.2 oracle (minimum-norm lstsq predictions).
    let sk_pred = [
        3.176_470_588_235_300_6,
        1.705_882_352_941_177,
        3.176_470_588_235_300_6,
        3.294_117_647_058_828,
    ];
    let preds = fitted.predict(&x).unwrap();
    for i in 0..4 {
        assert!(
            (preds[i] - sk_pred[i]).abs() < 1e-9,
            "singular-kernel(min-norm) predict[{i}] = {} vs sklearn {}",
            preds[i],
            sk_pred[i]
        );
    }
}
