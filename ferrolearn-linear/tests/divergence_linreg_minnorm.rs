//! Divergence tests: ferrolearn `LinearRegression` vs scikit-learn 1.5.2
//! `sklearn/linear_model/_base.py` `LinearRegression`.
//!
//! sklearn's dense OLS path is `self.coef_, _, self.rank_, self.singular_ =
//! linalg.lstsq(X, y)` (`sklearn/linear_model/_base.py:687`), i.e.
//! `scipy.linalg.lstsq` → LAPACK `gelsd` (SVD-based), which returns the
//! unique **minimum-norm** least-squares solution for rank-deficient `X`
//! and accepts underdetermined systems (`n_samples < n_features`).
//!
//! ferrolearn solves via `crate::linalg::solve_normal_equations` (Cholesky on
//! `XᵀX`) with an `.or_else` fallback to `crate::linalg::solve_lstsq` (faer QR).
//! faer's QR `solve_lstsq` on a rank-deficient matrix does NOT return the
//! minimum-norm solution, and `solve_lstsq` outright rejects
//! `n_samples < n_features`. Expected values below come from the live
//! sklearn 1.5.2 oracle (see each test's `python3 -c` reproduction).
//!
//! Fix site: `ferrolearn-linear/src/linalg.rs` (the `solve_lstsq` /
//! `solve_normal_equations` solver path), NOT `linear_regression.rs`.

use ferrolearn_core::Fit;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_linear::LinearRegression;
use ndarray::{Array2, array};

/// Divergence: ferrolearn `LinearRegression::fit` (via `linalg::solve_lstsq`,
/// `ferrolearn-linear/src/linalg.rs:35`) diverges from sklearn
/// `sklearn/linear_model/_base.py:687` (`linalg.lstsq(X, y)`) on
/// rank-deficient `X` with `fit_intercept=false`.
///
/// Input: X = [[1,1],[2,2],[3,3]] (duplicate columns, rank 1), y = [1,2,3].
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.linear_model import \
///     LinearRegression; m=LinearRegression(fit_intercept=False).fit(\
///     np.array([[1.,1.],[2.,2.],[3.,3.]]), np.array([1.,2.,3.])); \
///     print(m.coef_.tolist())"
///   -> [0.4999999999999999, 0.5]   (minimum-norm split)
/// ferrolearn returns approx [1.3275, -0.3275] (a non-min-norm LS solution).
///
/// Tracking: #376  (fix in linalg.rs)
#[test]
fn divergence_rank_deficient_no_intercept_min_norm() {
    // Oracle: minimum-norm solution from scipy gelsd.
    const SK_COEF0: f64 = 0.4999999999999999;
    const SK_COEF1: f64 = 0.5;

    let x = Array2::from_shape_vec((3, 2), vec![1., 1., 2., 2., 3., 3.]).unwrap();
    let y = array![1., 2., 3.];

    let model = LinearRegression::<f64>::new().with_fit_intercept(false);
    let fitted = model
        .fit(&x, &y)
        .expect("fit should succeed on rank-deficient X");
    let coef = fitted.coefficients();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-8 && (coef[1] - SK_COEF1).abs() < 1e-8,
        "min-norm divergence: sklearn coef_=[{SK_COEF0}, {SK_COEF1}], ferrolearn={coef:?}",
    );
}

/// Divergence: same rank-deficient `X` with `fit_intercept=true`. After
/// centering, the (centered) design is still rank-1, so the intercept-free
/// solve is rank-deficient; sklearn's gelsd yields the min-norm split.
///
/// Input: X = [[1,1],[2,2],[3,3]], y = [1,3,5], fit_intercept=True.
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.linear_model import \
///     LinearRegression; m=LinearRegression().fit(\
///     np.array([[1.,1.],[2.,2.],[3.,3.]]), np.array([1.,3.,5.])); \
///     print(m.coef_.tolist(), m.intercept_)"
///   -> [1.0, 0.9999999999999998]  intercept_ -0.9999999999999996
/// ferrolearn returns coef approx [~0, 2.0], intercept -1.0
/// (the centered QR fallback collapses onto one column instead of the
/// symmetric min-norm split).
///
/// Tracking: #376  (fix in linalg.rs)
#[test]
fn divergence_rank_deficient_with_intercept_min_norm() {
    // Oracle: gelsd min-norm split of the centered system.
    const SK_COEF0: f64 = 1.0;
    const SK_COEF1: f64 = 0.9999999999999998;
    const SK_INTERCEPT: f64 = -0.9999999999999996;

    let x = Array2::from_shape_vec((3, 2), vec![1., 1., 2., 2., 3., 3.]).unwrap();
    let y = array![1., 3., 5.];

    let model = LinearRegression::<f64>::new();
    let fitted = model
        .fit(&x, &y)
        .expect("fit should succeed on rank-deficient X");
    let coef = fitted.coefficients();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-8
            && (coef[1] - SK_COEF1).abs() < 1e-8
            && (fitted.intercept() - SK_INTERCEPT).abs() < 1e-8,
        "min-norm divergence: sklearn coef_=[{SK_COEF0}, {SK_COEF1}] intercept_={SK_INTERCEPT}, \
         ferrolearn coef={coef:?} intercept={}",
        fitted.intercept(),
    );
}

/// Divergence: ferrolearn `LinearRegression::fit` REJECTS underdetermined
/// input (`n_samples < n_features`) with `FerroError::InsufficientSamples`
/// (raised in `linalg::solve_lstsq`, `ferrolearn-linear/src/linalg.rs:41`),
/// whereas sklearn `sklearn/linear_model/_base.py:687` (`linalg.lstsq`)
/// SUCCEEDS and returns the minimum-norm coefficients. ferrolearn rejects
/// valid sklearn input.
///
/// Input: X = [[1,2,3],[4,5,6]] (2 samples, 3 features), y = [1,2],
/// fit_intercept=False.
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.linear_model import \
///     LinearRegression; m=LinearRegression(fit_intercept=False).fit(\
///     np.array([[1.,2.,3.],[4.,5.,6.]]), np.array([1.,2.])); \
///     print(m.coef_.tolist())"
///   -> [-0.05555555555555583, 0.11111111111111112, 0.277777777777778]
/// ferrolearn returns Err(InsufficientSamples { required: 3, actual: 2, .. }).
///
/// Tracking: #377  (fix in linalg.rs)
#[test]
fn divergence_underdetermined_accepted_min_norm() {
    // Oracle: gelsd minimum-norm solution for the underdetermined system.
    const SK_COEF: [f64; 3] = [
        -0.055_555_555_555_555_83,
        0.111_111_111_111_111_12,
        0.277_777_777_777_778,
    ];

    let x = Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let y = array![1., 2.];

    let model = LinearRegression::<f64>::new().with_fit_intercept(false);
    let fitted = model
        .fit(&x, &y)
        .expect("sklearn accepts n_samples < n_features; ferrolearn must too");
    let coef = fitted.coefficients();

    assert!(
        (coef[0] - SK_COEF[0]).abs() < 1e-8
            && (coef[1] - SK_COEF[1]).abs() < 1e-8
            && (coef[2] - SK_COEF[2]).abs() < 1e-8,
        "underdetermined min-norm divergence: sklearn coef_={SK_COEF:?}, ferrolearn={coef:?}",
    );
}
