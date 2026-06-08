//! Live-oracle parity tests for [`MultiTaskLassoCV`] against scikit-learn 1.5.2
//! (`sklearn.linear_model.MultiTaskLassoCV`,
//! `sklearn/linear_model/_coordinate_descent.py:3061`).
//!
//! `MultiTaskLassoCV` is the `l1_ratio = 1.0` specialization of the multi-task
//! ElasticNet-CV core (no `l1_ratio` parameter, no `l1_ratio_` attribute). The
//! CORE CV path (L21 alpha grid + contiguous k-fold + MSE-select + refit) must
//! reproduce the live oracle's `alpha_` EXACTLY, and `coef_`/`intercept_`
//! within the coordinate-descent stopping tolerance (~1e-4, shared #412).
//!
//! Oracle: scikit-learn 1.5.2 (commit 156ef14), computed live (the `python3 -c`
//! command is quoted in each test). Expected values are NEVER copied from the
//! ferrolearn side (R-CHAR-3).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::MultiTaskLassoCV;
use ndarray::{Array2, array};

/// Shared 12×2 fixture (n=12 samples, 2 features, 2 tasks). Identical to the
/// fixture used to compute the oracle.
fn fixture() -> (Array2<f64>, Array2<f64>) {
    let x: Array2<f64> = array![
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 5.0],
        [2.0, 3.0],
        [6.0, 1.0],
        [3.0, 3.0],
        [7.0, 2.0],
        [1.0, 5.0],
        [4.0, 6.0],
        [5.0, 2.0],
    ];
    let y: Array2<f64> = array![
        [3.0, 1.0],
        [2.5, 2.0],
        [7.1, 3.5],
        [6.0, 4.2],
        [11.2, 6.0],
        [5.0, 3.0],
        [9.0, 2.0],
        [6.5, 3.3],
        [12.0, 3.5],
        [3.0, 5.5],
        [8.5, 7.0],
        [9.5, 3.2],
    ];
    (x, y)
}

/// `MultiTaskLassoCV(n_alphas=5, cv=3)` → `alpha_` matches the live oracle
/// EXACTLY; `coef_`/`intercept_` within the CD tol.
///
/// Live sklearn 1.5.2 oracle (R-CHAR-3):
/// ```text
/// from sklearn.linear_model import MultiTaskLassoCV; import numpy as np
/// X=np.array([[1,2],[2,1],[3,4],[4,3],[5,5],[2,3],[6,1],[3,3],[7,2],[1,5],[4,6],[5,2]],float)
/// Y=np.array([[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6],[5,3],[9,2],[6.5,3.3],[12,3.5],[3,5.5],[8.5,7],[9.5,3.2]],float)
/// m=MultiTaskLassoCV(n_alphas=5, cv=3).fit(X,Y)
/// m.alpha_      -> 0.00525642486974421
/// m.coef_       -> [[1.6066524868454277,0.536008369924593],[0.28173028682943685,1.0247988713048561]]
/// m.intercept_  -> [-0.4681972184636116,-0.4859967143287882]
/// hasattr(m, 'l1_ratio_') -> False
/// ```
#[test]
fn mtlasso_cv_alpha_matches_sklearn() {
    let (x, y) = fixture();
    let fitted = MultiTaskLassoCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .fit(&x, &y)
        .expect("fit should succeed");

    // alpha_ is deterministic — must match EXACTLY.
    assert!(
        (fitted.alpha() - 0.005_256_424_869_744_21).abs() < 1e-12,
        "alpha_ = {} (expected 0.00525642486974421)",
        fitted.alpha()
    );

    let coef = fitted.coef();
    assert_eq!(coef.dim(), (2, 2));
    assert!((coef[[0, 0]] - 1.606_652_486_845_427_7).abs() < 1e-4);
    assert!((coef[[0, 1]] - 0.536_008_369_924_593).abs() < 1e-4);
    assert!((coef[[1, 0]] - 0.281_730_286_829_436_85).abs() < 1e-4);
    assert!((coef[[1, 1]] - 1.024_798_871_304_856_1).abs() < 1e-4);

    let intercept = fitted.intercept();
    assert!((intercept[0] - (-0.468_197_218_463_611_6)).abs() < 1e-4);
    assert!((intercept[1] - (-0.485_996_714_328_788_2)).abs() < 1e-4);
}

/// `predict` returns an `(n_samples, n_tasks)` array matching the live oracle.
///
/// Live sklearn 1.5.2 oracle (R-CHAR-3):
/// ```text
/// m=MultiTaskLassoCV(n_alphas=5, cv=3).fit(X,Y); m.predict(X[:2])
/// -> [[2.210472008231002,1.8453313151103607],[3.281116125151837,1.1022627306349415]]
/// ```
#[test]
fn mtlasso_cv_predict_shape_and_values() {
    let (x, y) = fixture();
    let fitted = MultiTaskLassoCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .fit(&x, &y)
        .expect("fit should succeed");

    let preds = fitted.predict(&x).expect("predict should succeed");
    assert_eq!(preds.dim(), (12, 2));

    let head: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0]];
    let p = fitted.predict(&head).expect("predict head");
    assert!((p[[0, 0]] - 2.210_472_008_231_002).abs() < 1e-4);
    assert!((p[[0, 1]] - 1.845_331_315_110_360_7).abs() < 1e-4);
    assert!((p[[1, 0]] - 3.281_116_125_151_837).abs() < 1e-4);
    assert!((p[[1, 1]] - 1.102_262_730_634_941_5).abs() < 1e-4);
}

/// Non-finite input is rejected (delegated to the ENet-CV core's
/// `force_all_finite` guard).
#[test]
fn mtlasso_cv_rejects_non_finite_input() {
    let (mut x, y) = fixture();
    x[[0, 0]] = f64::NAN;
    let res = MultiTaskLassoCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .fit(&x, &y);
    assert!(res.is_err(), "NaN in X must be rejected");
}

/// Divergence: `MultiTaskLassoCV`'s degenerate `alpha_max` branch diverges from
/// `sklearn/linear_model/_coordinate_descent.py:180-183` for a constant `Y`.
///
/// `MultiTaskLassoCV` delegates to the `MultiTaskElasticNetCV` core with
/// `l1_ratio=1.0`, so it inherits the same degenerate-grid bug. When `Y` is
/// constant, `Xy = Xcᵀ Yc` is all-zero and `alpha_max = 0`. sklearn's
/// `_alpha_grid` fills the grid with `np.finfo(float).resolution = 1e-15`
/// (`:180-183`), so `alpha_ = 1e-15`. ferrolearn
/// (`multi_task_elastic_net_cv.rs:457-458`) fills with `1e-6`, so `alpha_ = 1e-6`.
///
/// Live sklearn 1.5.2 oracle (R-CHAR-3):
/// ```text
/// import numpy as np; from sklearn.linear_model import MultiTaskLassoCV
/// X=np.array([[1,2],[2,1],[3,4],[4,3],[5,5],[2,3],[6,1],[3,3],[7,2],[1,5],[4,6],[5,2]],float)
/// Y=np.full((12,2),5.0)
/// MultiTaskLassoCV(n_alphas=5,cv=3).fit(X,Y).alpha_  -> 1e-15
/// ```
/// ferrolearn returns `alpha_ = 1e-6`.
/// Tracking: #2242
#[test]
#[ignore = "divergence: degenerate alpha_max fills 1e-6 not np.finfo.resolution(1e-15); tracking #2242"]
fn mtlasso_cv_degenerate_alpha_max_constant_y() {
    let (x, _y) = fixture();
    let y: Array2<f64> = Array2::from_elem((12, 2), 5.0);
    let fitted = MultiTaskLassoCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .fit(&x, &y)
        .expect("fit should succeed");

    // sklearn fills the degenerate grid with np.finfo(float).resolution = 1e-15
    // (_coordinate_descent.py:182), so alpha_ == 1e-15.
    const SK_RESOLUTION: f64 = 1e-15;
    assert!(
        (fitted.alpha() - SK_RESOLUTION).abs() <= SK_RESOLUTION * 1e-3,
        "alpha_ = {} (expected sklearn np.finfo(float).resolution = 1e-15)",
        fitted.alpha()
    );
}
