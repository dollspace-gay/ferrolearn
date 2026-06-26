//! Public estimator parity for `PassiveAggressiveClassifier` and
//! `PassiveAggressiveRegressor`.
//!
//! Oracle runtime: sklearn 1.5.2 in this workspace. The tests use
//! `shuffle=False` and `tol=None` to avoid the known cross-PRNG shuffle
//! boundary and force the exact epoch count.
//!
//! Python oracle:
//! ```python
//! import numpy as np
//! from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
//! np.set_printoptions(precision=17)
//! Xc = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.]])
//! yc = np.array([0,1,1,0])
//! for loss in ["hinge", "squared_hinge"]:
//!     m = PassiveAggressiveClassifier(C=0.7, loss=loss, max_iter=4,
//!                                     tol=None, shuffle=False, fit_intercept=True)
//!     m.fit(Xc, yc)
//!     print(loss, m.coef_, m.intercept_, m.predict(Xc))
//! Xr = np.array([[0.,1.],[1.,0.],[2.,1.],[3.,2.]])
//! yr = np.array([0.5,1.0,2.5,3.5])
//! for loss in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
//!     m = PassiveAggressiveRegressor(C=0.8, loss=loss, epsilon=0.1,
//!                                    max_iter=4, tol=None, shuffle=False,
//!                                    fit_intercept=True)
//!     m.fit(Xr, yr)
//!     print(loss, m.coef_, m.intercept_, m.predict(Xr))
//! ```

use approx::assert_relative_eq;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::sgd::{PassiveAggressiveClassifierLoss, PassiveAggressiveRegressorLoss};
use ferrolearn_linear::{PassiveAggressiveClassifier, PassiveAggressiveRegressor};
use ndarray::{Array1, Array2, array};

fn classifier_data() -> (Array2<f64>, Array1<usize>) {
    let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
    let y = array![0usize, 1, 1, 0];
    (x, y)
}

fn regressor_data() -> (Array2<f64>, Array1<f64>) {
    let x = array![[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]];
    let y = array![0.5, 1.0, 2.5, 3.5];
    (x, y)
}

#[test]
fn passive_aggressive_classifier_pa1_matches_sklearn() {
    let (x, y) = classifier_data();
    let fitted = PassiveAggressiveClassifier::<f64>::new()
        .with_c(0.7)
        .with_max_iter(4)
        .with_tol(f64::NEG_INFINITY)
        .with_shuffle(false)
        .fit(&x, &y)
        .expect("fit");

    assert_relative_eq!(fitted.coefficients()[0], -1.64042, epsilon = 1e-12);
    assert_relative_eq!(
        fitted.coefficients()[1],
        1.3649399999999998,
        epsilon = 1e-12
    );
    assert_relative_eq!(fitted.intercept(), 0.6649400000000001, epsilon = 1e-12);
    assert_eq!(fitted.predict(&x).unwrap().to_vec(), vec![0, 1, 1, 0]);
}

#[test]
fn passive_aggressive_classifier_pa2_matches_sklearn() {
    let (x, y) = classifier_data();
    let fitted = PassiveAggressiveClassifier::<f64>::new()
        .with_loss(PassiveAggressiveClassifierLoss::SquaredHinge)
        .with_c(0.7)
        .with_max_iter(4)
        .with_tol(f64::NEG_INFINITY)
        .with_shuffle(false)
        .fit(&x, &y)
        .expect("fit");

    assert_relative_eq!(
        fitted.coefficients()[0],
        -1.424589456397446,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        fitted.coefficients()[1],
        1.3032392404488227,
        epsilon = 1e-12
    );
    assert_relative_eq!(fitted.intercept(), 0.49483926800314537, epsilon = 1e-12);
    assert_eq!(fitted.predict(&x).unwrap().to_vec(), vec![0, 1, 1, 0]);
}

#[test]
fn passive_aggressive_regressor_pa1_matches_sklearn() {
    let (x, y) = regressor_data();
    let fitted = PassiveAggressiveRegressor::<f64>::new()
        .with_c(0.8)
        .with_epsilon(0.1)
        .with_max_iter(4)
        .with_tol(f64::NEG_INFINITY)
        .with_shuffle(false)
        .fit(&x, &y)
        .expect("fit");

    assert_relative_eq!(
        fitted.coefficients()[0],
        0.9881699044151117,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        fitted.coefficients()[1],
        0.2266530723714156,
        epsilon = 1e-12
    );
    assert_relative_eq!(fitted.intercept(), 0.16467896222121087, epsilon = 1e-12);
    let preds = fitted.predict(&x).unwrap();
    let expected = [
        0.39133203459262644,
        1.1528488666363226,
        2.36767184342285,
        3.5824948202093774,
    ];
    for (actual, expected) in preds.iter().zip(expected) {
        assert_relative_eq!(*actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn passive_aggressive_regressor_pa2_matches_sklearn() {
    let (x, y) = regressor_data();
    let fitted = PassiveAggressiveRegressor::<f64>::new()
        .with_loss(PassiveAggressiveRegressorLoss::SquaredEpsilonInsensitive)
        .with_c(0.8)
        .with_epsilon(0.1)
        .with_max_iter(4)
        .with_tol(f64::NEG_INFINITY)
        .with_shuffle(false)
        .fit(&x, &y)
        .expect("fit");

    assert_relative_eq!(
        fitted.coefficients()[0],
        0.9138266327266349,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        fitted.coefficients()[1],
        0.24994267717387555,
        epsilon = 1e-12
    );
    assert_relative_eq!(fitted.intercept(), 0.3494232633600606, epsilon = 1e-12);
    let preds = fitted.predict(&x).unwrap();
    let expected = [
        0.5993659405339361,
        1.2632498960866956,
        2.427019205987206,
        3.5907885158877164,
    ];
    for (actual, expected) in preds.iter().zip(expected) {
        assert_relative_eq!(*actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn passive_aggressive_validation_errors() {
    let (x, yc) = classifier_data();
    let (_, yr) = regressor_data();

    let err = PassiveAggressiveClassifier::<f64>::new()
        .with_c(0.0)
        .fit(&x, &yc)
        .expect_err("C <= 0 must be rejected");
    assert!(format!("{err}").contains("eta0"));

    let err = PassiveAggressiveRegressor::<f64>::new()
        .with_c(-1.0)
        .fit(&x, &yr)
        .expect_err("C <= 0 must be rejected");
    assert!(format!("{err}").contains("eta0"));

    let err = PassiveAggressiveRegressor::<f64>::new()
        .with_epsilon(-0.1)
        .fit(&x, &yr)
        .expect_err("epsilon < 0 must be rejected");
    assert!(format!("{err}").contains("epsilon"));
}
