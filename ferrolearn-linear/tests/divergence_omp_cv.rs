//! Public estimator parity for `ferrolearn_linear::OrthogonalMatchingPursuitCV`
//! against the live scikit-learn 1.5.2
//! `sklearn.linear_model.OrthogonalMatchingPursuitCV` oracle.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::OrthogonalMatchingPursuitCV;
use ndarray::{Array1, Array2, array};

fn fixture() -> (Array2<f64>, Array1<f64>) {
    (
        array![
            [1.0, 0.0, 0.0],
            [2.0, 0.1, 0.0],
            [3.0, 0.0, 0.1],
            [4.0, 0.1, 0.0],
            [5.0, 0.0, 0.1],
            [6.0, 0.2, 0.0],
        ],
        array![2.1, 4.0, 6.2, 8.1, 10.1, 12.2],
    )
}

/// Live sklearn 1.5.2 oracle:
/// `OrthogonalMatchingPursuitCV(cv=3, max_iter=2).fit(X, y)` returns
/// `coef_=[2.02, 0.0, 0.0]`, `intercept_=0.04666666666666863`,
/// `n_nonzero_coefs_=1`, `n_iter_=1`, and predictions
/// `[2.06666667, 4.08666667]` for `X[:2]`.
#[test]
fn omp_cv_matches_sklearn_cv3() {
    let (x, y) = fixture();
    let fitted = OrthogonalMatchingPursuitCV::<f64>::new()
        .with_cv(3)
        .with_max_iter(2)
        .fit(&x, &y)
        .unwrap();

    assert!((fitted.coefficients()[0] - 2.02).abs() < 1e-10);
    assert!(fitted.coefficients()[1].abs() < 1e-12);
    assert!(fitted.coefficients()[2].abs() < 1e-12);
    assert!((fitted.intercept() - 0.04666666666666863).abs() < 1e-10);
    assert_eq!(fitted.n_nonzero_coefs(), 1);
    assert_eq!(fitted.n_iter(), 1);

    let preds = fitted
        .predict(&x.slice(ndarray::s![0..2, ..]).to_owned())
        .unwrap();
    assert!((preds[0] - 2.06666667).abs() < 1e-8);
    assert!((preds[1] - 4.08666667).abs() < 1e-8);
}

/// Live sklearn 1.5.2 oracle:
/// `OrthogonalMatchingPursuitCV(cv=3, fit_intercept=False, max_iter=2)` returns
/// `coef_=[2.03076923, 0.0, 0.0]`, `intercept_=0.0`,
/// `n_nonzero_coefs_=1`, and predictions `[2.03076923, 4.06153846]`.
#[test]
fn omp_cv_no_intercept_matches_sklearn() {
    let (x, y) = fixture();
    let fitted = OrthogonalMatchingPursuitCV::<f64>::new()
        .with_cv(3)
        .with_max_iter(2)
        .with_fit_intercept(false)
        .fit(&x, &y)
        .unwrap();

    assert!((fitted.coefficients()[0] - 2.03076923).abs() < 1e-8);
    assert!(fitted.coefficients()[1].abs() < 1e-12);
    assert!(fitted.coefficients()[2].abs() < 1e-12);
    assert_eq!(fitted.intercept(), 0.0);
    assert_eq!(fitted.n_nonzero_coefs(), 1);
    assert_eq!(fitted.n_iter(), 1);

    let preds = fitted
        .predict(&x.slice(ndarray::s![0..2, ..]).to_owned())
        .unwrap();
    assert!((preds[0] - 2.03076923).abs() < 1e-8);
    assert!((preds[1] - 4.06153846).abs() < 1e-8);
}

#[test]
fn omp_cv_validation_errors() {
    let (x, y) = fixture();
    let err = OrthogonalMatchingPursuitCV::<f64>::new()
        .with_cv(1)
        .fit(&x, &y)
        .unwrap_err();
    assert!(matches!(
        err,
        FerroError::InvalidParameter {
            name,
            ..
        } if name == "cv"
    ));

    let short_y = array![1.0, 2.0];
    let err = OrthogonalMatchingPursuitCV::<f64>::new()
        .with_cv(2)
        .fit(&x, &short_y)
        .unwrap_err();
    assert!(matches!(err, FerroError::ShapeMismatch { .. }));
}
