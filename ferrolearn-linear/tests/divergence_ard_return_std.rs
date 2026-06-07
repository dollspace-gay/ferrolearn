//! Divergence pins for `ferrolearn_linear::ard::ARDRegression` REQ-8:
//! `predict(return_std=True)` and the full kept-feature `sigma_` posterior
//! covariance.
//!
//! sklearn's `ARDRegression.predict(X, return_std=True)`
//! (`sklearn/linear_model/_bayes.py:761-791`) computes
//!
//! ```text
//! y_mean    = X @ coef_ + intercept_
//! col_index = lambda_ < threshold_lambda                 # the kept-feature mask
//! Xk        = X[:, col_index]                            # kept columns only
//! y_std     = sqrt((Xk @ sigma_ * Xk).sum(axis=1) + 1/alpha_)
//! ```
//!
//! where `sigma_` is the kept-feature `(n_kept, n_kept)` covariance (sklearn
//! `self.sigma_`, `_bayes.py:727`). Pruned features (whose `coef_` is `0`)
//! contribute nothing to the predictive variance.
//!
//! Expected values are from the LIVE sklearn 1.5.2 oracle (NOT copied from
//! ferrolearn — R-CHAR-3):
//!
//! ```text
//! python3 -c "import numpy as np; from sklearn.linear_model import ARDRegression; \
//!   X=np.array([[1.,100.],[2.,200.],[3.,300.],[4.,400.],[5.,500.],[6.,600.]]); \
//!   y=np.array([2.,4.,6.,8.,10.,12.]); m=ARDRegression(max_iter=1000).fit(X,y); \
//!   Xs=np.vstack([X[:2], X[-1]*3.0]); mean,std=m.predict(Xs,return_std=True); \
//!   print(mean.tolist(), std.tolist(), m.sigma_.tolist())"
//! ```
//!
//! Tracking: #479 (REQ-8 return_std / full sigma_).

use ferrolearn_core::traits::Fit;
use ferrolearn_linear::ard::ARDRegression;
use ndarray::{Array2, array};

/// REQ-8 (2-feature design, both kept): `predict_with_std` mean+std (incl. an
/// out-of-range query point `X[-1]*3`) and the `(2,2)` `sigma_full` match the
/// live sklearn 1.5.2 oracle.
#[test]
fn divergence_ard_return_std_2feature() {
    // sklearn 1.5.2 live oracle (see module doc).
    const SK_MEAN: [f64; 3] = [
        2.000_000_027_343_324,
        4.000_000_016_405_995,
        35.999_999_841_408_716,
    ];
    const SK_STD: [f64; 3] = [
        0.000_650_274_537_103_761_9,
        0.000_701_019_524_959_021_8,
        0.002_793_870_276_725_064_3,
    ];
    const SK_SIGMA: [[f64; 2]; 2] = [
        [0.020_386_515_283_721_447, -0.000_203_865_151_672_268_78],
        [-0.000_203_865_151_672_268_78, 2.038_653_790_786_59e-6],
    ];

    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0, 5.0, 500.0, 6.0, 600.0,
        ],
    )
    .unwrap();
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

    let fitted = ARDRegression::<f64>::new()
        .with_max_iter(1000)
        .fit(&x, &y)
        .unwrap();

    // sigma_full is the kept-feature (2,2) covariance — both features kept here.
    let sigma = fitted.sigma_full();
    assert_eq!(
        sigma.dim(),
        (2, 2),
        "sigma_ must be the kept (2,2) matrix (#479)"
    );
    for i in 0..2 {
        for j in 0..2 {
            // The kept-block covariance matches sklearn's `pinvh`-based `sigma_`
            // to ~1e-6 relative; the residual is the ferray LU-inverse vs
            // scipy `pinvh` backend difference (R-DEV-7), not an ARD-math
            // divergence — the mean/std (the user-observable contract) match to
            // ~1e-7 below.
            assert!(
                (sigma[[i, j]] - SK_SIGMA[i][j]).abs() <= 1e-6 * SK_SIGMA[i][j].abs().max(1e-9),
                "sigma_[{i},{j}] diverges: ferrolearn={}, sklearn={} (#479)",
                sigma[[i, j]],
                SK_SIGMA[i][j],
            );
        }
    }

    // Query: first two training rows + an OUT-OF-RANGE row (X[-1]*3).
    let xs = Array2::from_shape_vec((3, 2), vec![1.0, 100.0, 2.0, 200.0, 18.0, 1800.0]).unwrap();
    let (mean, std) = fitted.predict_with_std(&xs).unwrap();

    for i in 0..3 {
        assert!(
            (mean[i] - SK_MEAN[i]).abs() <= 1e-7 * SK_MEAN[i].abs().max(1.0),
            "pred mean[{i}] diverges: ferrolearn={}, sklearn={} (#479)",
            mean[i],
            SK_MEAN[i],
        );
        assert!(
            (std[i] - SK_STD[i]).abs() <= 1e-7 * SK_STD[i].abs().max(1e-9),
            "pred std[{i}] diverges: ferrolearn={}, sklearn={} (#479)",
            std[i],
            SK_STD[i],
        );
    }
}

/// REQ-8 (4-feature design with PRUNED features 1, 3): `sigma_` is the kept
/// `(2,2)` block (features 0 and 2), and `predict_with_std` uses ONLY the kept
/// columns for the variance — matching the live oracle.
#[test]
fn divergence_ard_return_std_4feature_pruned() {
    // sklearn 1.5.2 live oracle.
    const SK_SIGMA: [[f64; 2]; 2] = [
        [1.439_986_606_517_746, -0.719_993_302_578_607_6],
        [-0.719_993_302_578_607_6, 0.359_996_652_649_850_9],
    ];
    const SK_MEAN: [f64; 3] = [
        2.999_999_875_812_711_5,
        5.999_999_911_294_794,
        72.000_000_691_900_61,
    ];
    const SK_STD: [f64; 3] = [
        0.000_540_848_338_470_329_6,
        0.000_559_397_071_011_681_2,
        0.002_050_385_474_645_718_6,
    ];

    let x = Array2::from_shape_vec(
        (8, 4),
        vec![
            1.0, 50.0, 2.0, -3.0, 2.0, 10.0, 4.0, 1.0, 3.0, 90.0, 6.0, -7.0, 4.0, 20.0, 8.0, 5.0,
            5.0, 70.0, 10.0, -2.0, 6.0, 40.0, 12.0, 9.0, 7.0, 60.0, 14.0, -1.0, 8.0, 30.0, 16.0,
            4.0,
        ],
    )
    .unwrap();
    let y = array![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0];

    let fitted = ARDRegression::<f64>::new()
        .with_max_iter(1000)
        .fit(&x, &y)
        .unwrap();

    // sigma_ is the kept-block (2,2) (features 0 and 2), not (4,4).
    let sigma = fitted.sigma_full();
    assert_eq!(
        sigma.dim(),
        (2, 2),
        "sigma_ must be the KEPT (2,2) block, not (4,4) (#479)",
    );
    for i in 0..2 {
        for j in 0..2 {
            // ~1e-6 relative: ferray LU-inverse vs scipy `pinvh` backend
            // (R-DEV-7), not an ARD-math divergence.
            assert!(
                (sigma[[i, j]] - SK_SIGMA[i][j]).abs() <= 1e-6 * SK_SIGMA[i][j].abs().max(1e-9),
                "sigma_[{i},{j}] diverges: ferrolearn={}, sklearn={} (#479)",
                sigma[[i, j]],
                SK_SIGMA[i][j],
            );
        }
    }
    // keep_lambda reflects the pruned set {1,3}.
    assert_eq!(
        fitted.keep_lambda(),
        &[true, false, true, false],
        "keep_lambda must be the kept-feature mask (#479)",
    );

    // Query: rows 0, 1, and an OUT-OF-RANGE row (X[-1]*3).
    let xs = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0, 50.0, 2.0, -3.0, 2.0, 10.0, 4.0, 1.0, 24.0, 90.0, 48.0, 12.0,
        ],
    )
    .unwrap();
    let (mean, std) = fitted.predict_with_std(&xs).unwrap();
    for i in 0..3 {
        assert!(
            (mean[i] - SK_MEAN[i]).abs() <= 1e-6 * SK_MEAN[i].abs().max(1.0),
            "pred mean[{i}] diverges: ferrolearn={}, sklearn={} (#479)",
            mean[i],
            SK_MEAN[i],
        );
        assert!(
            (std[i] - SK_STD[i]).abs() <= 1e-6 * SK_STD[i].abs().max(1e-9),
            "pred std[{i}] diverges: ferrolearn={}, sklearn={} (#479)",
            std[i],
            SK_STD[i],
        );
    }
}
