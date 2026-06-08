//! Live-oracle parity tests for [`MultiTaskElasticNetCV`] against scikit-learn
//! 1.5.2 (`sklearn.linear_model.MultiTaskElasticNetCV`,
//! `sklearn/linear_model/_coordinate_descent.py:2806`).
//!
//! The CORE CV path (per-l1_ratio L21 alpha grid + contiguous k-fold +
//! MSE-select + refit) must reproduce the live oracle's `alpha_` / `l1_ratio_`
//! EXACTLY (the alpha grid + CV-MSE ranking are deterministic), and `coef_` /
//! `intercept_` within the coordinate-descent stopping tolerance (~1e-4, the
//! shared #412 bound, mirroring the single-task `divergence_elastic_net_cv`).
//!
//! Oracle: scikit-learn 1.5.2 (commit 156ef14), computed live (the `python3 -c`
//! command is quoted in each test). Expected values are NEVER copied from the
//! ferrolearn side (R-CHAR-3).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::MultiTaskElasticNetCV;
use ndarray::{Array2, array};

/// Shared 12×2 fixture (n=12 samples, 2 features, 2 tasks) — enough samples for
/// cv=3 contiguous folds. Identical to the fixture used to compute the oracle.
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

/// `MultiTaskElasticNetCV(n_alphas=5, cv=3, l1_ratio=0.5)` → `alpha_` matches
/// the live oracle EXACTLY; `coef_`/`intercept_` within the CD tol.
///
/// Live sklearn 1.5.2 oracle (R-CHAR-3):
/// ```text
/// from sklearn.linear_model import MultiTaskElasticNetCV; import numpy as np
/// X=np.array([[1,2],[2,1],[3,4],[4,3],[5,5],[2,3],[6,1],[3,3],[7,2],[1,5],[4,6],[5,2]],float)
/// Y=np.array([[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6],[5,3],[9,2],[6.5,3.3],[12,3.5],[3,5.5],[8.5,7],[9.5,3.2]],float)
/// m=MultiTaskElasticNetCV(n_alphas=5, cv=3, l1_ratio=0.5).fit(X,Y)
/// m.alpha_      -> 0.01051284973948842
/// m.l1_ratio_   -> 0.5
/// m.coef_       -> [[1.6039496442790724,0.5343216166677465],[0.28097310368473755,1.022422076545992]]
/// m.intercept_  -> [-0.4533112100588941,-0.47595502422045133]
/// ```
#[test]
fn mtenet_cv_alpha_l1ratio_matches_sklearn() {
    let (x, y) = fixture();
    let fitted = MultiTaskElasticNetCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .with_l1_ratios(vec![0.5])
        .fit(&x, &y)
        .expect("fit should succeed");

    // alpha_ + l1_ratio_ are deterministic — must match EXACTLY.
    assert!(
        (fitted.alpha() - 0.010_512_849_739_488_42).abs() < 1e-12,
        "alpha_ = {} (expected 0.01051284973948842)",
        fitted.alpha()
    );
    assert_eq!(fitted.l1_ratio(), 0.5);

    // coef_ / intercept_ within the CD-stopping tol (#412).
    let coef = fitted.coef();
    assert_eq!(coef.dim(), (2, 2));
    assert!((coef[[0, 0]] - 1.603_949_644_279_072_4).abs() < 1e-4);
    assert!((coef[[0, 1]] - 0.534_321_616_667_746_5).abs() < 1e-4);
    assert!((coef[[1, 0]] - 0.280_973_103_684_737_55).abs() < 1e-4);
    assert!((coef[[1, 1]] - 1.022_422_076_545_992).abs() < 1e-4);

    let intercept = fitted.intercept();
    assert!((intercept[0] - (-0.453_311_210_058_894_1)).abs() < 1e-4);
    assert!((intercept[1] - (-0.475_955_024_220_451_33)).abs() < 1e-4);
}

/// A small `l1_ratios` grid `[0.3, 0.7]` → `l1_ratio_` + `alpha_` selection
/// matches the live oracle.
///
/// Live sklearn 1.5.2 oracle (R-CHAR-3):
/// ```text
/// m=MultiTaskElasticNetCV(l1_ratio=[0.3,0.7], n_alphas=5, cv=3).fit(X,Y)
/// m.alpha_     -> 0.007509178385348873
/// m.l1_ratio_  -> 0.7
/// m.coef_      -> [[1.6054929524113217,0.5352843492982494],[0.2814053089012693,1.0237788854975105]]
/// m.intercept_ -> [-0.46180982314350594,-0.4816872538468724]
/// ```
#[test]
fn mtenet_cv_l1ratio_grid_matches_sklearn() {
    let (x, y) = fixture();
    let fitted = MultiTaskElasticNetCV::<f64>::new()
        .with_l1_ratios(vec![0.3, 0.7])
        .with_n_alphas(5)
        .with_cv(3)
        .fit(&x, &y)
        .expect("fit should succeed");

    assert_eq!(fitted.l1_ratio(), 0.7);
    assert!(
        (fitted.alpha() - 0.007_509_178_385_348_873).abs() < 1e-12,
        "alpha_ = {} (expected 0.007509178385348873)",
        fitted.alpha()
    );

    let coef = fitted.coef();
    assert!((coef[[0, 0]] - 1.605_492_952_411_321_7).abs() < 1e-4);
    assert!((coef[[1, 1]] - 1.023_778_885_497_510_5).abs() < 1e-4);
}

/// `predict` returns an `(n_samples, n_tasks)` array matching the live oracle.
///
/// Live sklearn 1.5.2 oracle (R-CHAR-3):
/// ```text
/// m=MultiTaskElasticNetCV(n_alphas=5, cv=3, l1_ratio=0.5).fit(X,Y); m.predict(X[:2])
/// -> [[2.2192816675556712,1.8498622325562701],[3.288909695166997,1.1084132596950158]]
/// ```
#[test]
fn mtenet_cv_predict_shape_and_values() {
    let (x, y) = fixture();
    let fitted = MultiTaskElasticNetCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .with_l1_ratios(vec![0.5])
        .fit(&x, &y)
        .expect("fit should succeed");

    let preds = fitted.predict(&x).expect("predict should succeed");
    assert_eq!(preds.dim(), (12, 2));

    let head: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0]];
    let p = fitted.predict(&head).expect("predict head");
    assert!((p[[0, 0]] - 2.219_281_667_555_671_2).abs() < 1e-4);
    assert!((p[[0, 1]] - 1.849_862_232_556_270_1).abs() < 1e-4);
    assert!((p[[1, 0]] - 3.288_909_695_166_997).abs() < 1e-4);
    assert!((p[[1, 1]] - 1.108_413_259_695_015_8).abs() < 1e-4);
}

/// Non-finite input is rejected with an error (mirrors sklearn
/// `_validate_data(force_all_finite=True)`, `_coordinate_descent.py:1619`).
#[test]
fn mtenet_cv_rejects_non_finite_input() {
    let (mut x, y) = fixture();
    x[[0, 0]] = f64::NAN;
    let res = MultiTaskElasticNetCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .fit(&x, &y);
    assert!(res.is_err(), "NaN in X must be rejected");

    let (x2, mut y2) = fixture();
    y2[[1, 1]] = f64::INFINITY;
    let res2 = MultiTaskElasticNetCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .fit(&x2, &y2);
    assert!(res2.is_err(), "inf in Y must be rejected");
}

/// Divergence: `MultiTaskElasticNetCV`'s degenerate `alpha_max` branch diverges
/// from `sklearn/linear_model/_coordinate_descent.py:180-183` for a constant `Y`.
///
/// When `Y` is constant, the centered cross-product `Xy = Xcᵀ Yc` is all-zero, so
/// `alpha_max = sqrt(sum(Xy**2, axis=1)).max() / (n*l1_ratio) = 0`. sklearn's
/// `_alpha_grid` then takes the degenerate branch
/// (`_coordinate_descent.py:180`): `if alpha_max <= np.finfo(float).resolution:`
/// (resolution = 1e-15) and fills the WHOLE grid with `np.finfo(float).resolution`
/// (`:181-183`), so every grid alpha — and therefore `alpha_` — is `1e-15`.
///
/// ferrolearn (`multi_task_elastic_net_cv.rs:457-458`) instead tests
/// `alpha_max <= 0` and fills with `1e-6`, so `alpha_` is `1e-6` — wrong by ~9
/// orders of magnitude (and the threshold itself differs: `<= 0` vs
/// `<= 1e-15`).
///
/// Live sklearn 1.5.2 oracle (R-CHAR-3):
/// ```text
/// import numpy as np; from sklearn.linear_model import MultiTaskElasticNetCV
/// X=np.array([[1,2],[2,1],[3,4],[4,3],[5,5],[2,3],[6,1],[3,3],[7,2],[1,5],[4,6],[5,2]],float)
/// Y=np.full((12,2),5.0)
/// MultiTaskElasticNetCV(n_alphas=5,cv=3,l1_ratio=0.5).fit(X,Y).alpha_  -> 1e-15
/// ```
/// ferrolearn returns `alpha_ = 1e-6`.
/// Tracking: #2242
#[test]
fn mtenet_cv_degenerate_alpha_max_constant_y() {
    let (x, _y) = fixture();
    let y: Array2<f64> = Array2::from_elem((12, 2), 5.0);
    let fitted = MultiTaskElasticNetCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .with_l1_ratios(vec![0.5])
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
