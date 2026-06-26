//! Divergence guard for the dense `SelectorMixin` surface against scikit-learn
//! 1.5.2 `sklearn.feature_selection._base.SelectorMixin`.
//!
//! Oracle reproduction:
//! ```text
//! import numpy as np
//! from sklearn.feature_selection import VarianceThreshold
//! X = np.array([[1., 10., 100.], [2., 10., 200.], [3., 10., 300.]])
//! sel = VarianceThreshold().fit(X)
//! Xt = sel.transform(X)
//! print(sel.get_support().tolist())
//! print(sel.get_support(indices=True).tolist())
//! print(Xt.tolist())
//! print(sel.inverse_transform(Xt).tolist())
//! print(sel.get_feature_names_out().tolist())
//! print(sel.get_feature_names_out(["a", "b", "c"]).tolist())
//! ```

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::{
    Direction, RFE, RFECV, ScoreFunc, SelectFdr, SelectFpr, SelectFromModelExt, SelectFwe,
    SelectKBest, SelectPercentile, SelectorMixin, SequentialFeatureSelector, ThresholdStrategy,
    VarianceThreshold,
};
use ndarray::{Array1, Array2, array};

fn assert_matrix_eq(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            assert_eq!(actual[[i, j]], expected[[i, j]], "entry ({i},{j}) mismatch");
        }
    }
}

fn assert_selector<T: SelectorMixin<f64>>(selector: &T, n_features: usize) {
    assert_eq!(selector.get_support().len(), n_features);
    assert!(
        selector
            .get_support_indices()
            .iter()
            .all(|&idx| idx < n_features)
    );
}

#[test]
fn selector_mixin_variance_threshold_matches_sklearn_dense_oracle() {
    let x = array![
        [1.0_f64, 10.0, 100.0],
        [2.0, 10.0, 200.0],
        [3.0, 10.0, 300.0],
    ];
    let fitted = VarianceThreshold::<f64>::default().fit(&x, &()).unwrap();

    assert_eq!(fitted.get_support(), vec![true, false, true]);
    assert_eq!(fitted.get_support_indices(), vec![0, 2]);

    let reduced = fitted.transform(&x).unwrap();
    let expected_reduced = array![[1.0_f64, 100.0], [2.0, 200.0], [3.0, 300.0]];
    assert_matrix_eq(&reduced, &expected_reduced);

    let restored = fitted.inverse_transform(&reduced).unwrap();
    let expected_restored = array![[1.0_f64, 0.0, 100.0], [2.0, 0.0, 200.0], [3.0, 0.0, 300.0],];
    assert_matrix_eq(&restored, &expected_restored);

    assert_eq!(
        fitted.get_feature_names_out(None).unwrap(),
        vec!["x0".to_string(), "x2".to_string()]
    );
    let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    assert_eq!(
        fitted.get_feature_names_out(Some(&names)).unwrap(),
        vec!["a".to_string(), "c".to_string()]
    );

    let wrong_width = array![[1.0_f64], [2.0], [3.0]];
    assert!(fitted.inverse_transform(&wrong_width).is_err());
    assert!(
        fitted
            .get_feature_names_out(Some(&["a".to_string(), "b".to_string()]))
            .is_err()
    );
}

#[test]
fn selector_mixin_is_shared_across_fitted_selectors() {
    let x = array![
        [1.0_f64, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
        [4.0, 40.0, 400.0],
    ];
    let y_usize: Array1<usize> = array![0, 0, 1, 1];
    let y_f64: Array1<f64> = array![0.0, 0.0, 1.0, 1.0];

    let k_best = SelectKBest::<f64>::new(2, ScoreFunc::FClassif)
        .fit(&x, &y_usize)
        .unwrap();
    assert_selector(&k_best, 3);

    let percentile = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif)
        .fit(&x, &y_usize)
        .unwrap();
    assert_selector(&percentile, 3);

    let p_values = array![0.001_f64, 0.2, 0.9];
    let fpr = SelectFpr::<f64>::new(0.05).fit(&p_values, &()).unwrap();
    let fdr = SelectFdr::<f64>::new(0.05).fit(&p_values, &()).unwrap();
    let fwe = SelectFwe::<f64>::new(0.05).fit(&p_values, &()).unwrap();
    assert_selector(&fpr, 3);
    assert_selector(&fdr, 3);
    assert_selector(&fwe, 3);

    let importances = array![0.6_f64, 0.3, 0.1];
    let sfm = ferrolearn_preprocess::SelectFromModel::<f64>::new_from_importances(
        &importances,
        Some(0.2),
    )
    .unwrap();
    assert_selector(&sfm, 3);

    let sfm_ext = SelectFromModelExt::<f64>::new(ThresholdStrategy::Mean, None)
        .fit(&importances, &())
        .unwrap();
    assert_selector(&sfm_ext, 3);

    let rfe = RFE::<f64>::new(&importances, 2, 1).unwrap();
    let rfecv = RFECV::<f64>::new(&importances, &[0.8, 0.9, 0.7], 1).unwrap();
    assert_selector(&rfe, 3);
    assert_selector(&rfecv, 3);

    let score_fn = |x_sub: &Array2<f64>,
                    _y: &Array1<f64>|
     -> Result<f64, ferrolearn_core::FerroError> { Ok(x_sub.sum()) };
    let sfs = SequentialFeatureSelector::new(2, Direction::Forward)
        .fit(&x, &y_f64, score_fn)
        .unwrap();
    assert_selector(&sfs, 3);
}
