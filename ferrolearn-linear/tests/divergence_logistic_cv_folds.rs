//! Divergence: `LogisticRegressionCV` C-selection diverges from sklearn's
//! default `StratifiedKFold` fold partition.
//!
//! ferrolearn's `stratified_kfold_split` (`logistic_regression_cv.rs`) assigns
//! sample `i` of each class to test fold `i % k` (round-robin within class).
//! scikit-learn's `StratifiedKFold._make_test_folds`
//! (`sklearn/model_selection/_split.py:786-805`) assigns each class's samples to
//! folds in CONTIGUOUS blocks:
//!   `folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])`
//! where `allocation` is the round-robin distribution over the *sorted* labels.
//! These are different partitions, so the per-C CV accuracy differs and a
//! different C is selected.
//!
//! On the dataset below (`cv=3`, 18 samples, 2 classes) the live sklearn 1.5.2
//! oracle picks:
//!   `LogisticRegressionCV(Cs=10, cv=3, max_iter=2000).fit(X, y).C_ == [0.0001]`
//! ferrolearn's `i % k` partition selects `best_c() == 2.7826...` — a different
//! grid point ~4 orders of magnitude away. The C-grid (`np.logspace(-4,4,10)`,
//! matched to 3e-13, design-doc AC-1) and the accuracy scoring are correct, so
//! the fold partition is the sole cause of this selection divergence.
//!
//! Tracking: #456

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::LogisticRegressionCV;
use ndarray::{Array1, Array2};

/// sklearn `np.logspace(-4, 4, 10)` grid, reconstructed from the same formula
/// the production code uses (design doc AC-1: max abs diff to numpy is 3.4e-13).
fn logspace_grid() -> Vec<f64> {
    (0..10)
        .map(|i| 10f64.powf(-4.0 + i as f64 * 8.0 / 9.0))
        .collect()
}

#[test]
#[ignore = "divergence: i%k folds vs sklearn StratifiedKFold contiguous-block folds select a different C; tracking #456"]
fn divergence_logistic_cv_stratified_fold_selects_different_c() {
    // RandomState(0).randn(18,2)*2 with shuffled balanced labels;
    // ferrolearn confirmed to select C=2.78 vs sklearn C_=1e-4.
    let x = Array2::from_shape_vec(
        (18, 2),
        vec![
            3.528104691935328,
            0.8003144167344466,
            1.9574759682114784,
            4.481786398402916,
            3.735115980299935,
            -1.954555759752822,
            1.9001768350511787,
            -0.3027144165953958,
            -0.2064377035871157,
            0.8211970038767447,
            0.288087142321756,
            2.90854701392595,
            1.5220754502939868,
            0.24335003298565683,
            0.8877264654908513,
            0.6673486547485337,
            2.9881581463152123,
            -0.41031652753160175,
            0.6261354033018027,
            -1.7081914786034496,
            -5.105979631668157,
            1.3072371908807212,
            1.7288723977190115,
            -1.4843300408128839,
            4.539509247975215,
            -2.9087313491975295,
            0.09151703460289214,
            -0.3743677000516672,
            3.065558428716915,
            2.93871753980057,
            0.3098948513938326,
            0.7563250392043471,
            -1.7755714952602255,
            -3.961592936447854,
            -0.6958242986523052,
            0.3126979382079601,
        ],
    )
    .unwrap();
    let y: Array1<usize> =
        Array1::from_vec(vec![1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]);

    let model = LogisticRegressionCV::<f64>::new()
        .with_cs(logspace_grid())
        .with_cv(3)
        .with_max_iter(2000);
    let fitted = model.fit(&x, &y).unwrap();

    // Live sklearn 1.5.2 oracle:
    //   LogisticRegressionCV(Cs=10, cv=3, max_iter=2000).fit(X, y).C_ == [0.0001]
    // The fold partition is deterministic, so the selected grid point must match
    // exactly.
    const SKLEARN_C: f64 = 1e-4;
    let got = fitted.best_c();
    assert!(
        (got - SKLEARN_C).abs() < SKLEARN_C * 1e-6,
        "best_c divergence: sklearn StratifiedKFold C_={SKLEARN_C}, ferrolearn i%k C_={got}"
    );

    // With the correct C selected, the refit coef_ should also match sklearn's
    // LogisticRegressionCV refit at C_=1e-4 (tolerance bounded by the inner LBFGS
    // stopping gap, #412): coef ≈ [-4.3025e-4, 8.3242e-4], intercept ≈ 1.6838e-4.
    let coef = fitted.coefficients();
    let sklearn_coef = [-0.00043025418648744704_f64, 0.0008324161004309025];
    for (k, (c, s)) in coef.iter().zip(sklearn_coef.iter()).enumerate() {
        assert!(
            (c - s).abs() < 1e-2,
            "coef[{k}] divergence: sklearn={s}, ferrolearn={c}"
        );
    }
    let sklearn_intercept = 0.0001683781116914546_f64;
    assert!(
        (fitted.intercept() - sklearn_intercept).abs() < 1e-2,
        "intercept divergence: sklearn={sklearn_intercept}, ferrolearn={}",
        fitted.intercept()
    );
}

/// Isolation: forcing the SAME single C grid point on both sides removes the
/// fold-selection variable. ferrolearn's refit at C=1e-4 should then match
/// sklearn's `LogisticRegression(C=1e-4, max_iter=2000)` refit on the full data
/// (coef/intercept/classes/predict_proba), proving the only remaining
/// divergence is the fold partition (#456) and NOT the refit/scoring/grid.
///
/// Oracle: `LogisticRegression(C=1e-4, max_iter=2000).fit(X, y)` on the same
/// seed=0 dataset.
#[test]
fn isolation_logistic_cv_refit_at_forced_c_matches_sklearn() {
    let x = Array2::from_shape_vec(
        (18, 2),
        vec![
            3.528104691935328,
            0.8003144167344466,
            1.9574759682114784,
            4.481786398402916,
            3.735115980299935,
            -1.954555759752822,
            1.9001768350511787,
            -0.3027144165953958,
            -0.2064377035871157,
            0.8211970038767447,
            0.288087142321756,
            2.90854701392595,
            1.5220754502939868,
            0.24335003298565683,
            0.8877264654908513,
            0.6673486547485337,
            2.9881581463152123,
            -0.41031652753160175,
            0.6261354033018027,
            -1.7081914786034496,
            -5.105979631668157,
            1.3072371908807212,
            1.7288723977190115,
            -1.4843300408128839,
            4.539509247975215,
            -2.9087313491975295,
            0.09151703460289214,
            -0.3743677000516672,
            3.065558428716915,
            2.93871753980057,
            0.3098948513938326,
            0.7563250392043471,
            -1.7755714952602255,
            -3.961592936447854,
            -0.6958242986523052,
            0.3126979382079601,
        ],
    )
    .unwrap();
    let y: Array1<usize> =
        Array1::from_vec(vec![1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]);

    // Single-element grid: selection is trivial, refit happens at exactly 1e-4 on
    // both sides — the fold partition cannot affect the result.
    let fitted = LogisticRegressionCV::<f64>::new()
        .with_cs(vec![1e-4])
        .with_cv(3)
        .with_max_iter(2000)
        .fit(&x, &y)
        .unwrap();

    assert!((fitted.best_c() - 1e-4).abs() < 1e-12);

    // sklearn LogisticRegression(C=1e-4, max_iter=2000) on the full data:
    let sklearn_coef = [-0.00043007509864141506_f64, 0.0008325064541056766];
    let sklearn_int = 1.6385493982169197e-7_f64;
    let coef = fitted.coefficients();
    for (k, (c, s)) in coef.iter().zip(sklearn_coef.iter()).enumerate() {
        assert!(
            (c - s).abs() < 5e-4,
            "refit coef[{k}] mismatch: sklearn={s}, ferrolearn={c}"
        );
    }
    assert!(
        (fitted.intercept() - sklearn_int).abs() < 5e-4,
        "refit intercept mismatch: sklearn={sklearn_int}, ferrolearn={}",
        fitted.intercept()
    );

    // classes_ are the original sorted labels [0, 1].
    use ferrolearn_core::introspection::HasClasses;
    assert_eq!(fitted.classes(), &[0usize, 1]);

    // predict_proba row 0 ≈ [0.50021, 0.49979] (oracle), and predict matches.
    let pp = fitted.predict_proba(&x).unwrap();
    assert!((pp[[0, 0]] - 0.5002127297874683).abs() < 1e-3);
    assert!((pp[[0, 1]] - 0.49978727021253166).abs() < 1e-3);
}
