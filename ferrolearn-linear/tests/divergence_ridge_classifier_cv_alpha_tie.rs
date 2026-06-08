//! Divergence pin for `RidgeClassifierCV` alpha selection vs scikit-learn 1.5.2
//! on a degenerate n=2 fixture, surfaced while auditing `store_cv_results`/
//! `cv_results_` (#2248).
//!
//! R-CHAR-3: the expected value is the live sklearn 1.5.2 oracle, never copied
//! from the ferrolearn side.

use ferrolearn_core::traits::Fit;
use ferrolearn_linear::RidgeClassifierCV;
use ndarray::{Array2, array};

/// Divergence: `RidgeClassifierCV::fit` selects a different shared `alpha_`
/// (and hence `coef_`/`intercept_`) than sklearn on a 2-sample / 2-feature
/// binary fixture, because ferrolearn's eigen-path GCV computes every per-alpha
/// squared LOO error as EXACTLY 4.0 (a true tie across alphas) and breaks the
/// tie toward the first/smallest-index alpha (0.1), whereas sklearn's
/// decomposition produces a measurably-better (less negative) score for the
/// largest alpha and selects it.
///
/// sklearn cite: `_RidgeGCV.fit` maximises `alpha_score = -squared_errors.mean()`
/// with a STRICT update `alpha_score > best_score`
/// (`sklearn/linear_model/_ridge.py:2185`, score at `:2148-2150`/`:2216`). On
/// this fixture sklearn's per-alpha mean squared errors are
/// `4.000000000000002, 4.000000000000002, 4.0` (the alpha=10 term is exactly 4,
/// the smaller-alpha terms carry a ~2e-15 excess), so alpha=10 wins.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import RidgeClassifierCV; \
///   Xs=np.array([[1.,2.],[6.,5.]]); ys=np.array([0,1]); \
///   m=RidgeClassifierCV(alphas=[0.1,1.0,10.0]).fit(Xs,ys); \
///   print(m.alpha_, m.coef_.tolist(), m.intercept_.tolist())"
///   -> 10.0 [[0.18518518518518515, 0.11111111111111108]] [-1.0370370370370368]
/// ```
///
/// ferrolearn actual (`ridge_classifier_cv.rs:631` `gcv_scores_eigen`, the
/// `n_samples (2) <= n_features (2)` path): every `(c / G_inverse_diag)²` comes
/// out as bit-exact 4.0, so all three per-alpha totals tie at 8.0 and the
/// `select_alpha_gcv` first-wins tie-break (`:472-479`) keeps `alpha_ = 0.1`,
/// giving `coef_ = [[0.2924, 0.1754]]`, `intercept_ = [-1.6374]`. The fitted
/// `alpha_`/`coef_`/`intercept_` therefore diverge from sklearn (R-DEV-1).
///
/// Tracking: #2253
#[test]
#[ignore = "divergence: RidgeClassifierCV alpha_ tie-break (eigen path all-4.0 vs sklearn 10.0) on n=2; tracking #2253"]
fn divergence_ridge_classifier_cv_alpha_select_n2_eigen() {
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 6.0, 5.0]).unwrap();
    let y = array![0usize, 1];

    let fitted = RidgeClassifierCV::<f64>::new()
        .with_alphas(vec![0.1, 1.0, 10.0])
        .fit(&x, &y)
        .expect("fit must succeed");

    // sklearn selects alpha_ = 10.0 (oracle above); ferrolearn selects 0.1.
    assert!(
        (fitted.alpha_() - 10.0).abs() < 1e-9,
        "sklearn alpha_=10.0 (_ridge.py:2185 strict-max over per-alpha scores), \
         ferrolearn alpha_={}",
        fitted.alpha_()
    );

    // And the consequent fitted attributes (oracle above). `.coefficients()` is
    // the inherent (1, n_features) accessor.
    let coef = fitted.coefficients();
    assert!(
        (coef[[0, 0]] - 0.185_185_185_185_185_15).abs() < 1e-9
            && (coef[[0, 1]] - 0.111_111_111_111_111_08).abs() < 1e-9,
        "sklearn coef_=[[0.18518518518518515, 0.11111111111111108]], ferrolearn={coef:?}",
    );
    assert!(
        (fitted.intercepts()[0] - (-1.037_037_037_037_036_8)).abs() < 1e-9,
        "sklearn intercept_=-1.0370370370370368, ferrolearn={}",
        fitted.intercepts()[0]
    );
}
