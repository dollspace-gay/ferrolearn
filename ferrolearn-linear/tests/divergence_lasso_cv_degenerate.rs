//! Regression guard for `LassoCV`'s degenerate `alpha_max == 0` branch
//! (constant target `y`) — same root cause as the ElasticNetCV #2243 fix.
//!
//! When `y` is constant the centered cross-product `Xᵀy` is all-zero, so
//! `alpha_max = max|Xᵀy|/n = 0`. scikit-learn's `_alpha_grid`
//! (`sklearn/linear_model/_coordinate_descent.py:180-183`) handles this by
//! testing `alpha_max <= np.finfo(float).resolution` and filling the WHOLE
//! grid with `np.finfo(float).resolution`. `np.finfo(float)` is ALWAYS
//! `np.float64` regardless of input dtype, so the fill is the constant
//! `1e-15`, and `LassoCV(...).alpha_ == 1e-15`.
//!
//! ferrolearn formerly used the wrong threshold (`alpha_max <= 0`) and the
//! wrong fill (`1e-6`), so `alpha_` came out as `1e-6` — a divergence.
//!
//! Oracle: scikit-learn 1.5.2 (commit 156ef14), computed live:
//! ```text
//! import numpy as np
//! from sklearn.linear_model import LassoCV
//! X = np.array([[1.],[2.],[3.],[4.],[5.],[6.]])
//! y = np.array([5.,5.,5.,5.,5.,5.])
//! LassoCV(n_alphas=5, cv=3).fit(X, y).alpha_   # -> 1e-15  (NOT 1e-6)
//! np.finfo(float).resolution                   # -> 1e-15
//! ```
//! Expected value is the sklearn symbolic constant `np.finfo(float).resolution`
//! (= 1e-15), NEVER copied from the ferrolearn side (R-CHAR-3).

use ferrolearn_core::traits::Fit;
use ferrolearn_linear::LassoCV;
use ndarray::{Array1, Array2};

/// Constant-`y` input drives `alpha_max == 0`; sklearn fills the grid with
/// `np.finfo(float).resolution == 1e-15`, so `alpha_ == 1e-15`.
/// Before the fix ferrolearn produced `1e-6` (wrong threshold + wrong fill).
#[test]
fn lasso_cv_constant_y_alpha_is_resolution() {
    let x: Array2<f64> =
        Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y: Array1<f64> = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);

    let fitted = LassoCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .fit(&x, &y)
        .expect("LassoCV fit on constant y should succeed");

    // sklearn: np.finfo(float).resolution == 1e-15 (always f64).
    let resolution = 1e-15_f64;
    let alpha = fitted.best_alpha();

    assert!(
        (alpha - resolution).abs() <= resolution,
        "LassoCV(constant y).alpha_ should equal np.finfo(float).resolution \
         (1e-15), got {alpha:e} (pre-fix divergence value was 1e-6)"
    );
}

/// Guard against the pre-fix `1e-6` value re-appearing: 1e-6 is nine orders
/// of magnitude above the sklearn resolution, so this fails loudly on regress.
#[test]
fn lasso_cv_constant_y_not_old_fill() {
    let x: Array2<f64> =
        Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y: Array1<f64> = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);

    let fitted = LassoCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .fit(&x, &y)
        .expect("LassoCV fit on constant y should succeed");

    assert!(
        fitted.best_alpha() < 1e-9,
        "LassoCV(constant y).alpha_ must NOT be the pre-fix 1e-6 fill, got {:e}",
        fitted.best_alpha()
    );
}
