//! Public estimator parity for `ferrolearn_linear::LassoLarsIC`.
//!
//! Oracle runtime: sklearn 1.5.2 in this workspace.
//!
//! Python oracle:
//! ```python
//! import numpy as np
//! from sklearn.linear_model import LassoLarsIC
//! np.set_printoptions(precision=17)
//! X = np.array([[0.,0.],[1.,.2],[2.,.1],[3.,.4],[4.,.2],[5.,.6]])
//! y = np.array([.1,1.2,2.1,3.3,4.,5.2])
//! for criterion in ["aic", "bic"]:
//!     m = LassoLarsIC(criterion=criterion, noise_variance=1.0, max_iter=4).fit(X, y)
//!     print(criterion, m.alpha_, m.coef_, m.intercept_, m.alphas_, m.criterion_)
//! X2 = np.array([[0.,0.],[1.,.2],[2.,.1],[3.,.4],[4.,.2],[5.,.6],[6.,.4],[7.,.7]])
//! y2 = np.array([.2,1.,2.2,2.9,4.1,5.1,6.,7.2])
//! m = LassoLarsIC(criterion="aic", max_iter=4).fit(X2, y2)
//! print(m.alpha_, m.coef_, m.intercept_, m.noise_variance_, m.criterion_)
//! ```

use approx::assert_relative_eq;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::{LassoLarsIC, LassoLarsICCriterion};
use ndarray::{Array1, Array2, array};

fn explicit_variance_data() -> (Array2<f64>, Array1<f64>) {
    let x = array![
        [0.0, 0.0],
        [1.0, 0.2],
        [2.0, 0.1],
        [3.0, 0.4],
        [4.0, 0.2],
        [5.0, 0.6],
    ];
    let y = array![0.1, 1.2, 2.1, 3.3, 4.0, 5.2];
    (x, y)
}

fn estimated_variance_data() -> (Array2<f64>, Array1<f64>) {
    let x = array![
        [0.0, 0.0],
        [1.0, 0.2],
        [2.0, 0.1],
        [3.0, 0.4],
        [4.0, 0.2],
        [5.0, 0.6],
        [6.0, 0.4],
        [7.0, 0.7],
    ];
    let y = array![0.2, 1.0, 2.2, 2.9, 4.1, 5.1, 6.0, 7.2];
    (x, y)
}

#[test]
fn lasso_lars_ic_aic_explicit_noise_matches_sklearn() {
    let (x, y) = explicit_variance_data();
    let fitted = LassoLarsIC::<f64>::new()
        .with_noise_variance(1.0)
        .with_max_iter(4)
        .fit(&x, &y)
        .expect("fit");

    assert_relative_eq!(fitted.alpha(), 0.011093585699263811, epsilon = 1e-12);
    assert_relative_eq!(
        fitted.coefficients()[0],
        0.9990536277602525,
        epsilon = 1e-12
    );
    assert_relative_eq!(fitted.coefficients()[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(fitted.intercept(), 0.15236593059936876, epsilon = 1e-12);
    assert_relative_eq!(fitted.noise_variance(), 1.0, epsilon = 1e-15);
    assert_eq!(fitted.n_iter(), 2);

    let expected_alphas = [2.9250000000000003, 0.011093585699263811, 0.0];
    for (actual, expected) in fitted.alphas().iter().zip(expected_alphas) {
        assert_relative_eq!(*actual, expected, epsilon = 1e-12);
    }
    let expected_criterion = [28.682262398456075, 13.08237270903733, 15.036363117880532];
    for (actual, expected) in fitted.criterion().iter().zip(expected_criterion) {
        assert_relative_eq!(*actual, expected, epsilon = 1e-10);
    }

    let preds = fitted.predict(&x).unwrap();
    let expected_preds = [
        0.15236593059936876,
        1.1514195583596214,
        2.150473186119874,
        3.1495268138801262,
        4.148580441640378,
        5.147634069400631,
    ];
    for (actual, expected) in preds.iter().zip(expected_preds) {
        assert_relative_eq!(*actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn lasso_lars_ic_bic_explicit_noise_matches_sklearn() {
    let (x, y) = explicit_variance_data();
    let fitted = LassoLarsIC::<f64>::new()
        .with_criterion(LassoLarsICCriterion::Bic)
        .with_noise_variance(1.0)
        .with_max_iter(4)
        .fit(&x, &y)
        .expect("fit");

    assert_relative_eq!(fitted.alpha(), 0.011093585699263811, epsilon = 1e-12);
    let expected_criterion = [28.682262398456075, 12.874132178265384, 14.619882056336643];
    for (actual, expected) in fitted.criterion().iter().zip(expected_criterion) {
        assert_relative_eq!(*actual, expected, epsilon = 1e-10);
    }
}

#[test]
fn lasso_lars_ic_estimated_noise_matches_sklearn() {
    let (x, y) = estimated_variance_data();
    let fitted = LassoLarsIC::<f64>::new()
        .with_max_iter(4)
        .fit(&x, &y)
        .expect("fit");

    assert_relative_eq!(fitted.alpha(), 0.0026726973684211286, epsilon = 1e-12);
    assert_relative_eq!(
        fitted.coefficients()[0],
        0.9983004385964913,
        epsilon = 1e-12
    );
    assert_relative_eq!(fitted.coefficients()[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(fitted.intercept(), 0.0934484649122802, epsilon = 1e-12);
    assert_relative_eq!(
        fitted.noise_variance(),
        0.016725391498881467,
        epsilon = 1e-12
    );
    let expected_criterion = [2492.455749688424, -10.720206376100238, -9.023601590620741];
    for (actual, expected) in fitted.criterion().iter().zip(expected_criterion) {
        assert_relative_eq!(*actual, expected, epsilon = 1e-9);
    }
}

#[test]
fn lasso_lars_ic_validation_errors() {
    let (x, y) = explicit_variance_data();
    let err = LassoLarsIC::<f64>::new()
        .with_noise_variance(-1.0)
        .fit(&x, &y)
        .expect_err("negative noise variance must be rejected");
    assert!(format!("{err}").contains("noise_variance"));

    let wide_x = array![[0.0, 0.0, 1.0], [1.0, 0.2, 1.0], [2.0, 0.1, 1.0]];
    let wide_y = array![0.1, 1.2, 2.1];
    let err = LassoLarsIC::<f64>::new()
        .fit(&wide_x, &wide_y)
        .expect_err("default noise variance estimate must reject n <= p + intercept");
    assert!(format!("{err}").contains("noise_variance"));
}
