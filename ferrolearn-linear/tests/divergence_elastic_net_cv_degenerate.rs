//! Divergence guard for [`ElasticNetCV`]'s degenerate `alpha_max` branch against
//! scikit-learn 1.5.2 (`sklearn.linear_model.ElasticNetCV`,
//! `sklearn/linear_model/_coordinate_descent.py:2131`).
//!
//! This is the single-task spillover of the multi-task fix #2242. When `y` is
//! constant, the centered cross-product `Xᵀ yc` is all-zero, so
//! `alpha_max = max|Xᵀ yc| / (n·l1_ratio) = 0`. sklearn's `_alpha_grid`
//! (`_coordinate_descent.py:180-183`) then takes the degenerate branch:
//!
//! ```text
//! if alpha_max <= np.finfo(float).resolution:   # 1e-15
//!     alphas = np.empty(n_alphas)
//!     alphas.fill(np.finfo(float).resolution)   # 1e-15
//!     return alphas
//! ```
//!
//! so every grid alpha — and therefore `alpha_` — is `np.finfo(float).resolution
//! = 1e-15`. ferrolearn (`elastic_net_cv.rs:390-391`) instead tests
//! `alpha_max <= F::zero()` and fills with `F::from(1e-6).unwrap()`, so `alpha_`
//! is `1e-6` — wrong by ~9 orders of magnitude (and the threshold differs:
//! `<= 0` vs `<= 1e-15`, and the fill uses a bare `.unwrap()`, R-CODE-2 risk).
//!
//! Oracle: scikit-learn 1.5.2, computed live (the `python3 -c` command is quoted
//! in the test). Expected values are NEVER copied from the ferrolearn side
//! (R-CHAR-3): `1e-15` is `np.finfo(float).resolution`, a named symbolic constant
//! traceable to `_coordinate_descent.py:182`.

use ferrolearn_core::traits::Fit;
use ferrolearn_linear::ElasticNetCV;
use ndarray::{Array1, Array2};

/// The seed-1 `RandomState(1)` X fixture (n=12, p=3), identical to the matrix
/// used by `divergence_elastic_net_cv.rs::seed1_data` and the live oracle below.
#[rustfmt::skip]
fn seed1_x() -> Array2<f64> {
    Array2::from_shape_vec((12, 3), vec![
         1.6243453636632417, -0.6117564136500754, -0.5281717522634557,
        -1.0729686221561705,  0.8654076293246785, -2.3015386968802827,
         1.74481176421648,   -0.7612069008951028,  0.31903909605709857,
        -0.2493703754774101,  1.462107937044974,  -2.060140709497654,
        -0.3224172040135075, -0.38405435466841564, 1.1337694423354374,
        -1.0998912673140309, -0.17242820755043575,-0.8778584179213718,
         0.04221374671559283, 0.5828152137158222, -1.1006191772129212,
         1.1447237098396141,  0.9015907205927955,  0.5024943389018682,
         0.9008559492644118, -0.6837278591743331, -0.12289022551864817,
        -0.9357694342590688, -0.2678880796260159,  0.530355466738186,
        -0.691660751725309,  -0.39675352685597737,-0.6871727001195994,
        -0.8452056414987196, -0.671246130836819,  -0.01266459891890136,
    ]).unwrap()
}

/// Divergence: `ElasticNetCV`'s degenerate `alpha_max` branch diverges from
/// `sklearn/linear_model/_coordinate_descent.py:180-183` for a constant `y`.
///
/// With `y` constant, `alpha_max = 0`, so sklearn fills the whole grid with
/// `np.finfo(float).resolution = 1e-15` (`:182`) and selects `alpha_ = 1e-15`.
/// ferrolearn fills with `1e-6` (`elastic_net_cv.rs:391`) and selects
/// `alpha_ = 1e-6`.
///
/// Live sklearn 1.5.2 oracle (R-CHAR-3):
/// ```text
/// import numpy as np; from sklearn.linear_model import ElasticNetCV
/// X = <seed1 12x3 fixture above>
/// y = np.full(12, 7.0)
/// ElasticNetCV(n_alphas=5, cv=3, l1_ratio=0.5).fit(X, y).alpha_  -> 1e-15
/// ```
/// ferrolearn returns `alpha_ = 1e-6`.
/// Tracking: #2243
#[test]
#[ignore = "divergence: ElasticNetCV degenerate alpha_max fills 1e-6 not np.finfo.resolution 1e-15; tracking #2243"]
fn divergence_elastic_net_cv_degenerate_alpha_max_constant_y() {
    let x = seed1_x();
    let y = Array1::from_elem(12, 7.0);

    let fitted = ElasticNetCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .with_l1_ratios(vec![0.5])
        .fit(&x, &y)
        .expect("fit should succeed");

    // sklearn fills the degenerate grid with np.finfo(float).resolution = 1e-15
    // (_coordinate_descent.py:182), so alpha_ == 1e-15. ferrolearn fills 1e-6.
    const SK_RESOLUTION: f64 = 1e-15;
    assert!(
        (fitted.best_alpha() - SK_RESOLUTION).abs() <= SK_RESOLUTION * 1e-3,
        "ElasticNetCV constant-y alpha_: ferrolearn={} sklearn={} \
         (np.finfo(float).resolution, _coordinate_descent.py:182)",
        fitted.best_alpha(),
        SK_RESOLUTION,
    );
}
