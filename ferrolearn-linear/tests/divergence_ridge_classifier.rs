//! Adversarial divergence audit of `ferrolearn_linear::RidgeClassifier`
//! against the live scikit-learn 1.5.2 oracle.
//!
//! Mirrors `sklearn.linear_model.RidgeClassifier`
//! (`sklearn/linear_model/_ridge.py:1344`) and the shared binary-predict path
//! `LinearClassifierMixin.predict` (`sklearn/linear_model/_base.py:367-388`).
//!
//! All expected values are computed by live-calling sklearn 1.5.2 (R-CHAR-3).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::RidgeClassifier;
use ndarray::{Array1, Array2};

/// Numerical-parity GUARD (not a divergence). Confirms ferrolearn's binary
/// `decision_function`/`coef_` match the live sklearn oracle so the SIGN test
/// below isolates the boundary-comparison divergence, not a numeric one.
///
/// Oracle:
///   RidgeClassifier(alpha=1.0, fit_intercept=False).fit(X, y)
///   X=[[0,0],[2,0],[0,2],[2,2],[1,1]] y=[0,0,1,1,0]
///   coef_ = [[-0.33333333333333337, 0.4666666666666666]]
///   intercept_ = 0.0
///   decision_function = [0.0, -0.6666666666666667, 0.9333333333333332,
///                        0.2666666666666665, 0.13333333333333325]
#[test]
fn guard_binary_decision_matches_sklearn() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0usize, 0, 1, 1, 0]);
    let fitted = RidgeClassifier::<f64>::new()
        .with_alpha(1.0)
        .with_fit_intercept(false)
        .fit(&x, &y)
        .unwrap();
    let dec = fitted.decision_function(&x).unwrap();
    // n_targets == 1 for binary.
    let sk = [
        0.0,
        -0.6666666666666667,
        0.9333333333333332,
        0.2666666666666665,
        0.13333333333333325,
    ];
    for i in 0..5 {
        assert!(
            (dec[[i, 0]] - sk[i]).abs() < 1e-9,
            "decision[{i}] = {} vs sklearn {}",
            dec[[i, 0]],
            sk[i]
        );
    }
}

/// Divergence: `FittedRidgeClassifier::predict` (binary branch) uses
/// `scores >= 0 -> classes[1]` in
/// `ferrolearn-linear/src/ridge_classifier.rs` (`if scores[[i, 0]] >= F::zero()`),
/// but sklearn's binary predict uses STRICT `scores > 0`:
/// `indices = xp.astype(scores > 0, ...)` (`sklearn/linear_model/_base.py:384`),
/// so a sample whose decision_function is EXACTLY 0 maps to index 0 ->
/// `classes_[0]`.
///
/// Dataset (alpha=1.0, fit_intercept=False):
///   X=[[0,0],[2,0],[0,2],[2,2],[1,1]] y=[0,0,1,1,0]
///   sample 0 has decision_function == 0.0 exactly.
///   sklearn predict = [0, 0, 1, 1, 1]   (sample 0 -> class 0)
///   ferrolearn predict = [1, 0, 1, 1, 1] (sample 0 -> class 1, the bug)
///
/// Tracking: #405
#[test]
fn divergence_binary_decision_boundary_strict_gt() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0usize, 0, 1, 1, 0]);
    let fitted = RidgeClassifier::<f64>::new()
        .with_alpha(1.0)
        .with_fit_intercept(false)
        .fit(&x, &y)
        .unwrap();
    let preds = fitted.predict(&x).unwrap();
    // sklearn RidgeClassifier(alpha=1.0, fit_intercept=False).predict(X)
    let sk_pred = [0usize, 0, 1, 1, 1];
    assert_eq!(
        preds.to_vec(),
        sk_pred.to_vec(),
        "binary boundary (decision==0) must map to classes_[0] per _base.py:384 `scores > 0`"
    );
}
