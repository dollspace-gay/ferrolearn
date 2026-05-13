//! Conformance tests for ferrolearn-neighbors vs scikit-learn.
//!
//! Each test loads a fixture from `fixtures/<name>.json`, runs the matching
//! ferrolearn estimator on the input with the same hyperparameters, and
//! compares the output against sklearn's via `ferrolearn-test-oracle` helpers.
//!
//! KNN classification predictions should match sklearn's exactly when the
//! same metric and tie-breaking are used; if they don't, distance ties in
//! the training data are the most likely explanation. Use
//! `assert_labels_equal` for classification predictions and
//! `assert_close_slice` for regression predictions.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_neighbors::{KNeighborsClassifier, KNeighborsRegressor};
use ferrolearn_test_oracle::{
    TOL_NEIGHBORS_ABS, TOL_NEIGHBORS_REL, assert_close, assert_close_slice, assert_labels_equal,
    json_to_array1, json_to_array2, json_to_labels, load_fixture,
};

// ---------------------------------------------------------------------------
// 1. KNeighborsClassifier — iris, k=5.
//
// Predictions are required to match sklearn exactly. With identical Euclidean
// metric and uniform weights, the only source of disagreement is tie-breaking
// on equal distances; the iris fixture has no exact ties in feature space, so
// `assert_labels_equal` is the right gate.
// ---------------------------------------------------------------------------

#[test]
fn conformance_kneighbors_classifier() {
    let fx = load_fixture("kneighbors_classifier");
    let (rel, abs) = fx.tolerance(TOL_NEIGHBORS_REL, TOL_NEIGHBORS_ABS);

    let x = json_to_array2(&fx.input["X"]);
    let y_f64 = json_to_array1(&fx.input["y"]);
    let y: ndarray::Array1<usize> = y_f64.iter().map(|&v| v as usize).collect();

    let n_neighbors = fx.params["n_neighbors"].as_u64().unwrap_or(5) as usize;

    let model = KNeighborsClassifier::<f64>::new().with_n_neighbors(n_neighbors);
    let fitted = model.fit(&x, &y).expect("KNeighborsClassifier fit");

    let preds = fitted.predict(&x).expect("KNeighborsClassifier predict");
    let preds_i64: Vec<i64> = preds.iter().map(|&v| v as i64).collect();
    let expected_preds = json_to_labels(&fx.expected["predictions"]);
    assert_labels_equal(&preds_i64, &expected_preds, "KNeighborsClassifier.predict");

    // Accuracy is a single scalar — match within metric tolerance.
    let n_correct = preds
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| a == b)
        .count();
    let accuracy = n_correct as f64 / y.len() as f64;
    assert_close(
        accuracy,
        fx.expected["accuracy"].as_f64().unwrap(),
        rel,
        abs,
        "KNeighborsClassifier.accuracy",
    );
}

// ---------------------------------------------------------------------------
// 2. KNeighborsRegressor — diabetes first 100, k=5.
//
// Regression predictions are simple means over the k nearest neighbours' y;
// agreement should be at numerical-noise level (TOL_NEIGHBORS_*).
// ---------------------------------------------------------------------------

#[test]
fn conformance_kneighbors_regressor() {
    let fx = load_fixture("kneighbors_regressor");
    let (rel, abs) = fx.tolerance(TOL_NEIGHBORS_REL, TOL_NEIGHBORS_ABS);

    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);

    let n_neighbors = fx.params["n_neighbors"].as_u64().unwrap_or(5) as usize;

    let model = KNeighborsRegressor::<f64>::new().with_n_neighbors(n_neighbors);
    let fitted = model.fit(&x, &y).expect("KNeighborsRegressor fit");

    let preds = fitted.predict(&x).expect("KNeighborsRegressor predict");
    let expected_preds = json_to_array1(&fx.expected["predictions"]);
    assert_close_slice(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        rel,
        abs,
        "KNeighborsRegressor.predict",
    );

    // Compare R² (same closed-form definition in both libraries).
    let r2 = fitted
        .score(&x, &y)
        .expect("KNeighborsRegressor score (R²)");
    assert_close(
        r2,
        fx.expected["r2"].as_f64().unwrap(),
        rel,
        abs,
        "KNeighborsRegressor.r2",
    );
}
