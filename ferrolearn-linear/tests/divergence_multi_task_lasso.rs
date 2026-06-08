//! Live-oracle divergence pins for `MultiTaskLasso` against scikit-learn 1.5.2
//! (`sklearn/linear_model/_coordinate_descent.py:2663` `class
//! MultiTaskLasso(MultiTaskElasticNet)`; solver `_cd_fast.pyx:740`
//! `enet_coordinate_descent_multi_task`, commit 156ef14).
//!
//! `MultiTaskLasso` is the multi-output linear model fit jointly under an L2,1
//! (group-Lasso) mixed-norm penalty via block coordinate descent. It is
//! `MultiTaskElasticNet(l1_ratio=1.0)` → `l2_reg = 0`, `l1_reg = alpha * n`.
//! The block CD is DETERMINISTIC under the default cyclic selection, so exact
//! value parity (coef_/intercept_/n_iter_/predict) is testable.
//!
//! Every expected value below is produced by RUNNING scikit-learn 1.5.2 (the
//! live oracle), never copied from ferrolearn (goal.md R-CHAR-3). The exact
//! `python3 -c` invocation that produced each constant is recorded in a comment.
//!
//! Tracking: #413 (MultiTaskLasso estimator).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::MultiTaskLasso;
use ndarray::{Array2, array};

/// Shared 5×2 / 2-task fixture (R-CHAR-3).
fn fixture() -> (Array2<f64>, Array2<f64>) {
    let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]];
    let y: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0], [7.1, 3.5], [6.0, 4.2], [11.2, 6.0]];
    (x, y)
}

#[test]
fn mtl_fit_matches_sklearn() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3):
    //   python3 -c "from sklearn.linear_model import MultiTaskLasso; import numpy as np;
    //     X=np.array([[1,2],[2,1],[3,4],[4,3],[5,5]],float);
    //     Y=np.array([[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6]]);
    //     m=MultiTaskLasso(alpha=0.3).fit(X,Y);
    //     print(m.coef_.tolist(), m.intercept_.tolist(), m.n_iter_)"
    //   coef_       -> [[0.7874471321, 1.3745821226], [0.8341004367, 0.3460953631]]
    //   intercept_  -> [-0.5260877641, -0.2005873993]
    //   n_iter_     -> 19
    let (x, y) = fixture();
    let fitted = match MultiTaskLasso::<f64>::new().with_alpha(0.3).fit(&x, &y) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };

    let coef = fitted.coefficients();
    assert_eq!(
        coef.dim(),
        (2, 2),
        "coef_ shape must be (n_tasks, n_features)"
    );
    assert!((coef[[0, 0]] - 0.787_447_132_1).abs() < 1e-6);
    assert!((coef[[0, 1]] - 1.374_582_122_6).abs() < 1e-6);
    assert!((coef[[1, 0]] - 0.834_100_436_7).abs() < 1e-6);
    assert!((coef[[1, 1]] - 0.346_095_363_1).abs() < 1e-6);

    let intercept = fitted.intercepts();
    assert_eq!(intercept.len(), 2, "intercept_ length must be n_tasks");
    assert!((intercept[0] - (-0.526_087_764_1)).abs() < 1e-6);
    assert!((intercept[1] - (-0.200_587_399_3)).abs() < 1e-6);

    assert_eq!(fitted.n_iter(), 19, "n_iter_ must match sklearn's 19");
}

#[test]
fn mtl_alpha_grid_matches_sklearn() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3) — coef_ across an alpha grid:
    //   for a in [0.01,0.1,0.5,1.0]:
    //     MultiTaskLasso(alpha=a).fit(X,Y).coef_.ravel()
    //   0.01 -> [0.7531383132, 1.5425361909, 1.0458441461, 0.2126418797]
    //   0.1  -> [0.7728169485, 1.4825416518, 0.9668815619, 0.2676128196]
    //   0.5  -> [0.7782781099, 1.2879823940, 0.7379552611, 0.3875942928]
    //   1.0  -> [0.7099551726, 1.1137244678, 0.5728083775, 0.4165835571]
    let (x, y) = fixture();
    let cases: [(f64, [f64; 4]); 4] = [
        (
            0.01,
            [
                0.753_138_313_2,
                1.542_536_190_9,
                1.045_844_146_1,
                0.212_641_879_7,
            ],
        ),
        (
            0.1,
            [
                0.772_816_948_5,
                1.482_541_651_8,
                0.966_881_561_9,
                0.267_612_819_6,
            ],
        ),
        (
            0.5,
            [
                0.778_278_109_9,
                1.287_982_394_0,
                0.737_955_261_1,
                0.387_594_292_8,
            ],
        ),
        (
            1.0,
            [
                0.709_955_172_6,
                1.113_724_467_8,
                0.572_808_377_5,
                0.416_583_557_1,
            ],
        ),
    ];
    for (alpha, expected) in cases {
        let fitted = match MultiTaskLasso::<f64>::new().with_alpha(alpha).fit(&x, &y) {
            Ok(f) => f,
            Err(e) => panic!("fit failed at alpha={alpha}: {e:?}"),
        };
        let coef = fitted.coefficients();
        // coef_ is (n_tasks, n_features); ravel row-major == sklearn's coef_.ravel().
        let got = [coef[[0, 0]], coef[[0, 1]], coef[[1, 0]], coef[[1, 1]]];
        for k in 0..4 {
            assert!(
                (got[k] - expected[k]).abs() < 1e-6,
                "alpha={alpha} coef[{k}]={} != sklearn {}",
                got[k],
                expected[k]
            );
        }
    }
}

#[test]
fn mtl_no_intercept_matches_sklearn() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3):
    //   m=MultiTaskLasso(alpha=0.3, fit_intercept=False).fit(X,Y)
    //   coef_      -> [[0.7223086317, 1.2938631723], [0.8006773177, 0.3236384717]]
    //   intercept_ -> [0., 0.]
    //   n_iter_    -> 85
    let (x, y) = fixture();
    let fitted = match MultiTaskLasso::<f64>::new()
        .with_alpha(0.3)
        .with_fit_intercept(false)
        .fit(&x, &y)
    {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };

    let coef = fitted.coefficients();
    assert!((coef[[0, 0]] - 0.722_308_631_7).abs() < 1e-6);
    assert!((coef[[0, 1]] - 1.293_863_172_3).abs() < 1e-6);
    assert!((coef[[1, 0]] - 0.800_677_317_7).abs() < 1e-6);
    assert!((coef[[1, 1]] - 0.323_638_471_7).abs() < 1e-6);

    let intercept = fitted.intercepts();
    assert_eq!(
        intercept[0], 0.0,
        "fit_intercept=False must zero intercept_"
    );
    assert_eq!(
        intercept[1], 0.0,
        "fit_intercept=False must zero intercept_"
    );

    assert_eq!(fitted.n_iter(), 85, "n_iter_ must match sklearn's 85");
}

#[test]
fn mtl_three_task_matches_sklearn() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3), 3-task Y -> coef_ shape (3, 2):
    //   Y3=np.array([[3,1,0.5],[2.5,2,1.0],[7.1,3.5,2.0],[6,4.2,2.5],[11.2,6,3.0]])
    //   m=MultiTaskLasso(alpha=0.3).fit(X,Y3)
    //   coef_      -> [[0.806048067,1.360381127],[0.8529029065,0.3323110206],
    //                  [0.4670291057,0.1594111645]]
    //   intercept_ -> [-0.5392875819, -0.2156417812, -0.0793208107]
    //   n_iter_    -> 19
    let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]];
    let y3: Array2<f64> = array![
        [3.0, 1.0, 0.5],
        [2.5, 2.0, 1.0],
        [7.1, 3.5, 2.0],
        [6.0, 4.2, 2.5],
        [11.2, 6.0, 3.0]
    ];
    let fitted = match MultiTaskLasso::<f64>::new().with_alpha(0.3).fit(&x, &y3) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };

    let coef = fitted.coefficients();
    assert_eq!(coef.dim(), (3, 2), "3-task coef_ must be (3, n_features)");
    let expected = [
        [0.806_048_067_0, 1.360_381_127_0],
        [0.852_902_906_5, 0.332_311_020_6],
        [0.467_029_105_7, 0.159_411_164_5],
    ];
    for t in 0..3 {
        for j in 0..2 {
            assert!(
                (coef[[t, j]] - expected[t][j]).abs() < 1e-6,
                "coef[{t},{j}]={} != sklearn {}",
                coef[[t, j]],
                expected[t][j]
            );
        }
    }

    let intercept = fitted.intercepts();
    assert_eq!(intercept.len(), 3);
    assert!((intercept[0] - (-0.539_287_581_9)).abs() < 1e-6);
    assert!((intercept[1] - (-0.215_641_781_2)).abs() < 1e-6);
    assert!((intercept[2] - (-0.079_320_810_7)).abs() < 1e-6);

    assert_eq!(fitted.n_iter(), 19);
}

#[test]
fn mtl_group_sparsity_exact_zero_pattern_matches_sklearn() {
    // The L2,1 penalty zeros WHOLE feature columns jointly across all tasks. On a
    // design where feature 1 is irrelevant noise and Y = 2*x0 (task 0) / 4*x0
    // (task 1), a high alpha drives feature 1's whole task-row to EXACTLY zero
    // while feature 0 stays active — the group-sparsity contract.
    //
    // Live sklearn 1.5.2 oracle (R-CHAR-3):
    //   Xg=np.array([[1,0.3],[2,-0.1],[3,0.2],[4,0.05],[5,-0.2],[6,0.1]])
    //   Yg=np.array([[2,4],[4,8],[6,12],[8,16],[10,20],[12,24]],float)
    //   m=MultiTaskLasso(alpha=2.0).fit(Xg,Yg)
    //   coef_      -> [[1.6933392488, 0.0], [3.3866784976, 0.0]]
    //   intercept_ -> [1.0733126292, 2.1466252584]
    //   n_iter_    -> 2
    let xg: Array2<f64> = array![
        [1.0, 0.3],
        [2.0, -0.1],
        [3.0, 0.2],
        [4.0, 0.05],
        [5.0, -0.2],
        [6.0, 0.1]
    ];
    let yg: Array2<f64> = array![
        [2.0, 4.0],
        [4.0, 8.0],
        [6.0, 12.0],
        [8.0, 16.0],
        [10.0, 20.0],
        [12.0, 24.0]
    ];
    let fitted = match MultiTaskLasso::<f64>::new().with_alpha(2.0).fit(&xg, &yg) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };

    let coef = fitted.coefficients();
    // Feature 1's task-row is EXACTLY zero for both tasks (bit-exact group zero).
    assert_eq!(
        coef[[0, 1]],
        0.0,
        "feature-1 / task-0 coef must be exactly 0"
    );
    assert_eq!(
        coef[[1, 1]],
        0.0,
        "feature-1 / task-1 coef must be exactly 0"
    );
    // Feature 0 stays active for both tasks, matching sklearn's values.
    assert!((coef[[0, 0]] - 1.693_339_248_8).abs() < 1e-6);
    assert!((coef[[1, 0]] - 3.386_678_497_6).abs() < 1e-6);

    let intercept = fitted.intercepts();
    assert!((intercept[0] - 1.073_312_629_2).abs() < 1e-6);
    assert!((intercept[1] - 2.146_625_258_4).abs() < 1e-6);

    assert_eq!(fitted.n_iter(), 2, "n_iter_ must match sklearn's 2");
}

#[test]
fn mtl_dual_gap_matches_sklearn() {
    // Pins #2239: `FittedMultiTaskLasso` must expose the `dual_gap_` fitted
    // attribute (like `n_iter_`/`coef_`). sklearn sets it from
    // `enet_coordinate_descent_multi_task` (`_coordinate_descent.py:2636`) then
    // scales `self.dual_gap_ /= n_samples` (`:2652`).
    //
    // Live sklearn 1.5.2 oracle (R-CHAR-3):
    //   python3 -c "from sklearn.linear_model import MultiTaskLasso; import numpy as np;
    //     X=np.array([[1,2],[2,1],[3,4],[4,3],[5,5]],float);
    //     Y=np.array([[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6]]);
    //     [print(a, repr(MultiTaskLasso(alpha=a).fit(X,Y).dual_gap_),
    //            MultiTaskLasso(alpha=a).fit(X,Y).n_iter_) for a in (0.3,0.1,1.0)]"
    //   0.3 -> dual_gap_=0.00021539018133829302, n_iter_=19
    //   0.1 -> dual_gap_=0.00016093048471601534, n_iter_=20
    //   1.0 -> dual_gap_=0.0001449879028545098,  n_iter_=19
    let (x, y) = fixture();
    let cases: [(f64, f64, usize); 3] = [
        (0.3, 0.000_215_390_181_338_293_02, 19),
        (0.1, 0.000_160_930_484_716_015_34, 20),
        (1.0, 0.000_144_987_902_854_509_8, 19),
    ];
    for (alpha, expected_gap, expected_n_iter) in cases {
        let fitted = match MultiTaskLasso::<f64>::new().with_alpha(alpha).fit(&x, &y) {
            Ok(f) => f,
            Err(e) => panic!("fit failed at alpha={alpha}: {e:?}"),
        };
        assert!(
            (fitted.dual_gap() - expected_gap).abs() < 1e-9,
            "alpha={alpha} dual_gap_={} != sklearn {expected_gap}",
            fitted.dual_gap()
        );
        assert_eq!(
            fitted.n_iter(),
            expected_n_iter,
            "alpha={alpha} n_iter_ must still match sklearn"
        );
    }
}

#[test]
fn mtl_predict_shape_and_values_match_sklearn() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3):
    //   m=MultiTaskLasso(alpha=0.3).fit(X,Y); m.predict(X)
    //   shape -> (5, 2); first two rows ->
    //     [[3.0105236132, 1.3257037636], [2.4233886227, 1.8137088371]]
    let (x, y) = fixture();
    let fitted = match MultiTaskLasso::<f64>::new().with_alpha(0.3).fit(&x, &y) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };

    let preds = match fitted.predict(&x) {
        Ok(p) => p,
        Err(e) => panic!("predict failed: {e:?}"),
    };
    // predict is (n_samples, n_tasks).
    assert_eq!(preds.dim(), (5, 2), "predict must be (n_samples, n_tasks)");
    assert!((preds[[0, 0]] - 3.010_523_613_2).abs() < 1e-6);
    assert!((preds[[0, 1]] - 1.325_703_763_6).abs() < 1e-6);
    assert!((preds[[1, 0]] - 2.423_388_622_7).abs() < 1e-6);
    assert!((preds[[1, 1]] - 1.813_708_837_1).abs() < 1e-6);
}
