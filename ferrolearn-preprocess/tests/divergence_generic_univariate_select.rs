//! Divergence guard for `GenericUnivariateSelect` against scikit-learn 1.5.2
//! `sklearn.feature_selection.GenericUnivariateSelect`.
//!
//! Oracle reproduction:
//! ```text
//! import numpy as np, warnings
//! from sklearn.feature_selection import GenericUnivariateSelect, f_classif
//! X=np.array([[1.,10.,100.,0.], [2.,20.,100.,1.], [8.,10.,100.,0.],
//!             [9.,20.,100.,1.], [10.,30.,100.,0.], [11.,30.,100.,1.]])
//! y=np.array([0,0,1,1,1,1])
//! with warnings.catch_warnings():
//!     warnings.simplefilter('ignore')
//!     for mode,param in [('percentile',50),('percentile',0),('percentile',100),
//!                        ('k_best',2),('k_best','all'),
//!                        ('fpr',0.05),('fdr',0.05),('fwe',0.05)]:
//!         sel=GenericUnivariateSelect(f_classif, mode=mode, param=param).fit(X,y)
//!         print(mode, param, sel.get_support(indices=True).tolist())
//! ```

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::{
    GenericUnivariateMode, GenericUnivariateParam, GenericUnivariateSelect, ScoreFunc,
    SelectorMixin,
};
use ndarray::{Array1, Array2, array};

fn assert_matrix_eq(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            assert!(
                (actual[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "entry ({i},{j}) got {} expected {}",
                actual[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

fn fixture() -> (Array2<f64>, Array1<usize>) {
    let x = array![
        [1.0_f64, 10.0, 100.0, 0.0],
        [2.0, 20.0, 100.0, 1.0],
        [8.0, 10.0, 100.0, 0.0],
        [9.0, 20.0, 100.0, 1.0],
        [10.0, 30.0, 100.0, 0.0],
        [11.0, 30.0, 100.0, 1.0],
    ];
    let y: Array1<usize> = array![0, 0, 1, 1, 1, 1];
    (x, y)
}

fn fit_mode(
    mode: GenericUnivariateMode,
    param: GenericUnivariateParam,
) -> ferrolearn_preprocess::FittedGenericUnivariateSelect<f64> {
    let (x, y) = fixture();
    GenericUnivariateSelect::<f64>::new(ScoreFunc::FClassif, mode, param)
        .fit(&x, &y)
        .unwrap()
}

#[test]
fn generic_univariate_scores_and_percentile_match_sklearn_oracle() {
    let (x, _) = fixture();
    let fitted = fit_mode(
        GenericUnivariateMode::Percentile,
        GenericUnivariateParam::Value(50.0),
    );

    let expected_scores = [62.06060606060605, 0.9230769230769231, f64::NAN, 0.0];
    let expected_p = [0.0014036193002939595, 0.3910758879640666, f64::NAN, 1.0];
    for (idx, (&got, &expected)) in fitted
        .scores()
        .iter()
        .zip(expected_scores.iter())
        .enumerate()
    {
        if expected.is_nan() {
            assert!(got.is_nan(), "score[{idx}] should be NaN");
        } else {
            assert!(
                (got - expected).abs() < 1e-12,
                "score[{idx}] got {got} expected {expected}"
            );
        }
    }
    for (idx, (&got, &expected)) in fitted.p_values().iter().zip(expected_p.iter()).enumerate() {
        if expected.is_nan() {
            assert!(got.is_nan(), "p[{idx}] should be NaN");
        } else {
            assert!(
                (got - expected).abs() < 1e-12,
                "p[{idx}] got {got} expected {expected}"
            );
        }
    }

    assert_eq!(fitted.selected_indices(), &[0, 1]);
    assert_eq!(fitted.get_support(), vec![true, true, false, false]);
    assert_eq!(fitted.get_support_indices(), vec![0, 1]);

    let reduced = fitted.transform(&x).unwrap();
    let expected_reduced = array![
        [1.0_f64, 10.0],
        [2.0, 20.0],
        [8.0, 10.0],
        [9.0, 20.0],
        [10.0, 30.0],
        [11.0, 30.0],
    ];
    assert_matrix_eq(&reduced, &expected_reduced);

    let restored = fitted.inverse_transform(&reduced).unwrap();
    let expected_restored = array![
        [1.0_f64, 10.0, 0.0, 0.0],
        [2.0, 20.0, 0.0, 0.0],
        [8.0, 10.0, 0.0, 0.0],
        [9.0, 20.0, 0.0, 0.0],
        [10.0, 30.0, 0.0, 0.0],
        [11.0, 30.0, 0.0, 0.0],
    ];
    assert_matrix_eq(&restored, &expected_restored);

    let names = vec!["a".into(), "b".into(), "c".into(), "d".into()];
    assert_eq!(
        fitted.get_feature_names_out(Some(&names)).unwrap(),
        vec!["a".to_string(), "b".to_string()]
    );
}

#[test]
fn generic_univariate_all_modes_match_sklearn_support_oracles() {
    let cases = [
        (
            GenericUnivariateMode::Percentile,
            GenericUnivariateParam::Value(0.0),
            Vec::<usize>::new(),
        ),
        (
            GenericUnivariateMode::Percentile,
            GenericUnivariateParam::Value(100.0),
            vec![0, 1, 2, 3],
        ),
        (
            GenericUnivariateMode::KBest,
            GenericUnivariateParam::Value(2.0),
            vec![0, 1],
        ),
        (
            GenericUnivariateMode::KBest,
            GenericUnivariateParam::All,
            vec![0, 1, 2, 3],
        ),
        (
            GenericUnivariateMode::Fpr,
            GenericUnivariateParam::Value(0.05),
            vec![0],
        ),
        (
            GenericUnivariateMode::Fdr,
            GenericUnivariateParam::Value(0.05),
            vec![0],
        ),
        (
            GenericUnivariateMode::Fwe,
            GenericUnivariateParam::Value(0.05),
            vec![0],
        ),
    ];

    for (mode, param, expected) in cases {
        let fitted = fit_mode(mode, param);
        assert_eq!(
            fitted.selected_indices(),
            expected.as_slice(),
            "mode {mode:?} param {param:?}"
        );
        assert_eq!(
            fitted.get_support_indices(),
            expected,
            "support mode {mode:?} param {param:?}"
        );
    }
}

#[test]
fn generic_univariate_validates_params_and_transform_shape() {
    let (x, y) = fixture();
    let invalid_percentile = GenericUnivariateSelect::<f64>::new(
        ScoreFunc::FClassif,
        GenericUnivariateMode::Percentile,
        GenericUnivariateParam::Value(101.0),
    );
    assert!(invalid_percentile.fit(&x, &y).is_err());

    let invalid_k = GenericUnivariateSelect::<f64>::new(
        ScoreFunc::FClassif,
        GenericUnivariateMode::KBest,
        GenericUnivariateParam::Value(1.5),
    );
    assert!(invalid_k.fit(&x, &y).is_err());

    let invalid_all = GenericUnivariateSelect::<f64>::new(
        ScoreFunc::FClassif,
        GenericUnivariateMode::Fpr,
        GenericUnivariateParam::All,
    );
    assert!(invalid_all.fit(&x, &y).is_err());

    let fitted = fit_mode(
        GenericUnivariateMode::KBest,
        GenericUnivariateParam::Value(2.0),
    );
    let wrong_width = array![[1.0_f64, 2.0, 3.0]];
    assert!(fitted.transform(&wrong_width).is_err());
}
