//! Conformance tests for ferrolearn-linear vs scikit-learn.
//!
//! Each test loads a fixture from `fixtures/<name>.json`, runs the matching
//! ferrolearn estimator on the same input with the same hyperparameters, and
//! compares the output to sklearn's via `ferrolearn-test-oracle` helpers.
//!
//! Tolerances default to algorithm-class constants
//! (`TOL_LINEAR_FIT_*`, `TOL_ITERATIVE_LINEAR_*`, `TOL_LOGISTIC_*`) and can be
//! tightened/loosened per-fixture via the JSON `tolerance` field.
//!
//! When a fixture carries a `divergence_id`, the test logs the divergence and
//! still asserts conformance — the registry entry in
//! `tests/conformance/_divergences.toml` documents why a wider tolerance is
//! justified.

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::{Fit, Predict};
use ferrolearn_test_oracle::{
    MIN_LOGISTIC_ACCURACY, TOL_ITERATIVE_LINEAR_ABS, TOL_ITERATIVE_LINEAR_REL, TOL_LINEAR_FIT_ABS,
    TOL_LINEAR_FIT_REL, TOL_LOGISTIC_ABS, TOL_LOGISTIC_REL, assert_close, assert_close_slice,
    json_to_array1, json_to_array2, json_to_labels, load_fixture,
};

// ---------------------------------------------------------------------------
// LinearRegression
// ---------------------------------------------------------------------------

#[test]
fn conformance_linear_regression() {
    let fx = load_fixture("linear_regression");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let (rel, abs) = fx.tolerance(TOL_LINEAR_FIT_REL, TOL_LINEAR_FIT_ABS);

    let fit_intercept = fx.params["fit_intercept"].as_bool().unwrap_or(true);
    let model = ferrolearn_linear::LinearRegression::<f64>::new().with_fit_intercept(fit_intercept);
    let fitted = model.fit(&x, &y).expect("LinearRegression fit");

    let expected_coefs = json_to_array1(&fx.expected["coefficients"]);
    assert_close_slice(
        fitted.coefficients().as_slice().unwrap(),
        expected_coefs.as_slice().unwrap(),
        rel,
        abs,
        "LinearRegression.coefficients",
    );
    let expected_intercept = fx.expected["intercept"].as_f64().unwrap();
    assert_close(
        fitted.intercept(),
        expected_intercept,
        rel,
        abs,
        "LinearRegression.intercept",
    );

    let preds = fitted.predict(&x).expect("LinearRegression predict");
    let expected_preds = json_to_array1(&fx.expected["predictions"]);
    assert_close_slice(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        rel,
        abs,
        "LinearRegression.predict",
    );
}

// ---------------------------------------------------------------------------
// Ridge
// ---------------------------------------------------------------------------

#[test]
fn conformance_ridge() {
    let fx = load_fixture("ridge");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let (rel, abs) = fx.tolerance(TOL_LINEAR_FIT_REL, TOL_LINEAR_FIT_ABS);

    let alpha = fx.params["alpha"].as_f64().unwrap_or(1.0);
    let fit_intercept = fx.params["fit_intercept"].as_bool().unwrap_or(true);
    let model = ferrolearn_linear::Ridge::<f64>::new()
        .with_alpha(alpha)
        .with_fit_intercept(fit_intercept);
    let fitted = model.fit(&x, &y).expect("Ridge fit");

    let expected_coefs = json_to_array1(&fx.expected["coefficients"]);
    assert_close_slice(
        fitted.coefficients().as_slice().unwrap(),
        expected_coefs.as_slice().unwrap(),
        rel,
        abs,
        "Ridge.coefficients",
    );
    let expected_intercept = fx.expected["intercept"].as_f64().unwrap();
    assert_close(
        fitted.intercept(),
        expected_intercept,
        rel,
        abs,
        "Ridge.intercept",
    );

    let preds = fitted.predict(&x).expect("Ridge predict");
    let expected_preds = json_to_array1(&fx.expected["predictions"]);
    assert_close_slice(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        rel,
        abs,
        "Ridge.predict",
    );
}

// ---------------------------------------------------------------------------
// Lasso — iterative coordinate descent, looser tolerance.
// ---------------------------------------------------------------------------

#[test]
fn conformance_lasso() {
    let fx = load_fixture("lasso");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let (rel, abs) = fx.tolerance(TOL_ITERATIVE_LINEAR_REL, TOL_ITERATIVE_LINEAR_ABS);

    let alpha = fx.params["alpha"].as_f64().unwrap_or(0.1);
    let fit_intercept = fx.params["fit_intercept"].as_bool().unwrap_or(true);
    let model = ferrolearn_linear::Lasso::<f64>::new()
        .with_alpha(alpha)
        .with_fit_intercept(fit_intercept);
    let fitted = model.fit(&x, &y).expect("Lasso fit");

    let expected_coefs = json_to_array1(&fx.expected["coefficients"]);
    assert_close_slice(
        fitted.coefficients().as_slice().unwrap(),
        expected_coefs.as_slice().unwrap(),
        rel,
        abs,
        "Lasso.coefficients",
    );
    let expected_intercept = fx.expected["intercept"].as_f64().unwrap();
    assert_close(
        fitted.intercept(),
        expected_intercept,
        rel,
        abs,
        "Lasso.intercept",
    );

    let preds = fitted.predict(&x).expect("Lasso predict");
    let expected_preds = json_to_array1(&fx.expected["predictions"]);
    assert_close_slice(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        rel,
        abs,
        "Lasso.predict",
    );
}

// ---------------------------------------------------------------------------
// ElasticNet — also iterative.
// ---------------------------------------------------------------------------

#[test]
fn conformance_elastic_net() {
    let fx = load_fixture("elastic_net");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let (rel, abs) = fx.tolerance(TOL_ITERATIVE_LINEAR_REL, TOL_ITERATIVE_LINEAR_ABS);

    let alpha = fx.params["alpha"].as_f64().unwrap_or(0.1);
    let l1_ratio = fx.params["l1_ratio"].as_f64().unwrap_or(0.5);
    let fit_intercept = fx.params["fit_intercept"].as_bool().unwrap_or(true);
    let model = ferrolearn_linear::ElasticNet::<f64>::new()
        .with_alpha(alpha)
        .with_l1_ratio(l1_ratio)
        .with_fit_intercept(fit_intercept);
    let fitted = model.fit(&x, &y).expect("ElasticNet fit");

    let expected_coefs = json_to_array1(&fx.expected["coefficients"]);
    assert_close_slice(
        fitted.coefficients().as_slice().unwrap(),
        expected_coefs.as_slice().unwrap(),
        rel,
        abs,
        "ElasticNet.coefficients",
    );
    let expected_intercept = fx.expected["intercept"].as_f64().unwrap();
    assert_close(
        fitted.intercept(),
        expected_intercept,
        rel,
        abs,
        "ElasticNet.intercept",
    );

    let preds = fitted.predict(&x).expect("ElasticNet predict");
    let expected_preds = json_to_array1(&fx.expected["predictions"]);
    assert_close_slice(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        rel,
        abs,
        "ElasticNet.predict",
    );
}

// ---------------------------------------------------------------------------
// LogisticRegression — binary classification, LBFGS.
//
// Currently fails at the default `C=1.0` because ferrolearn's loss
// normalizes the data term by `1/n` (see `src/logistic_regression.rs:278`)
// while sklearn's keeps it as a sum. This means ferrolearn's effective
// regularization at the same `C` is `n×` stronger than sklearn's.
//
// Tracked at crosslink issue #334. The test is left in place but ignored
// so the bug surface stays visible — un-ignore once #334 is fixed.
// ---------------------------------------------------------------------------

#[test]
fn conformance_logistic_regression() {
    let fx = load_fixture("logistic_regression");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let (rel, abs) = fx.tolerance(TOL_LOGISTIC_REL, TOL_LOGISTIC_ABS);

    let c = fx.params["C"].as_f64().unwrap_or(1.0);
    let fit_intercept = fx.params["fit_intercept"].as_bool().unwrap_or(true);
    let model = ferrolearn_linear::LogisticRegression::<f64>::new()
        .with_c(c)
        .with_fit_intercept(fit_intercept);
    // LogisticRegression takes labels as Array1<usize>.
    let y_usize: ndarray::Array1<usize> = y.iter().map(|&v| v as usize).collect();
    let fitted = model.fit(&x, &y_usize).expect("LogisticRegression fit");

    // Predicted classes — both libraries converge to a slightly different
    // optimum under L-BFGS so we require a high label-agreement floor rather
    // than strict equality. Documented as `logistic-lbfgs-path` divergence.
    let preds = fitted.predict(&x).expect("LogisticRegression predict");
    let preds_i64: Vec<i64> = preds.iter().map(|&v| v as i64).collect();
    let expected_classes = json_to_labels(&fx.expected["predicted_classes"]);
    assert_eq!(
        preds_i64.len(),
        expected_classes.len(),
        "LogisticRegression.predict: length mismatch"
    );
    let matches = preds_i64
        .iter()
        .zip(expected_classes.iter())
        .filter(|(a, e)| a == e)
        .count();
    let accuracy = matches as f64 / preds_i64.len() as f64;
    assert!(
        accuracy >= MIN_LOGISTIC_ACCURACY,
        "LogisticRegression.predict label-agreement {accuracy:.4} below floor \
         {MIN_LOGISTIC_ACCURACY:.4} ({matches}/{} match)",
        preds_i64.len()
    );

    // Probabilities should agree within logistic-class tolerance. Fixture
    // stores them row-major as [[p0, p1], ...].
    let proba = fitted
        .predict_proba(&x)
        .expect("LogisticRegression predict_proba");
    let expected_proba = json_to_array2(&fx.expected["predicted_proba"]);
    assert_close_slice(
        proba.as_slice().unwrap(),
        expected_proba.as_slice().unwrap(),
        rel,
        abs,
        "LogisticRegression.predict_proba",
    );
}
