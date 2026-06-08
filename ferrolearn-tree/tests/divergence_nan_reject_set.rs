//! Divergence pins: ferrolearn-tree's NaN/non-finite REJECT-set estimators do
//! not reject NaN input, whereas scikit-learn 1.5.2 raises
//! `ValueError("Input X contains NaN.")`.
//!
//! sklearn finite-check site: `sklearn/utils/validation.py:147-154`
//!   `if has_inf or has_nan_error: ... msg_err = f"Input {padded_input_name}contains {type_err}."`
//! These estimators DO NOT pass `force_all_finite=False` (unlike the
//! DecisionTree base, `sklearn/tree/_classes.py:248-250`
//! `dict(dtype=DTYPE, accept_sparse="csc", force_all_finite=False)`), so the
//! default `force_all_finite=True` of `_validate_data` rejects NaN.
//!
//! Live oracle (sklearn 1.5.2), X = [[1,2],[2,3],[3,3],[5,6],[6,7],[7,8]] with
//! X[0,0]=NaN, on EACH estimator below:
//!   `ExtraTreeClassifier(random_state=0).fit(X,y)` -> ValueError: Input X contains NaN.
//!   (identical for ExtraTreeRegressor, ExtraTreesClassifier/Regressor,
//!    AdaBoostClassifier/Regressor, GradientBoostingClassifier/Regressor,
//!    IsolationForest — all 9 raise `ValueError: Input X contains NaN.`).
//!
//! ferrolearn has NO finiteness check on any path (confirmed: 0 finite-checks),
//! so each `fit` below returns `Ok(_)` (a silently divergent tree) instead of
//! the sklearn `ValueError`. Each test asserts `fit(...).is_err()` — the sklearn
//! contract — and therefore FAILS against current ferrolearn (`Ok`).
//!
//! Tracking: #2278.

use ferrolearn_core::Fit;
use ferrolearn_tree::{
    AdaBoostClassifier, AdaBoostRegressor, ExtraTreeClassifier, ExtraTreeRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, IsolationForest,
};
use ndarray::{Array1, Array2, array};

fn x_nan() -> Array2<f64> {
    let mut x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
    )
    .unwrap();
    x[[0, 0]] = f64::NAN;
    x
}
fn yc() -> Array1<usize> {
    array![0, 0, 0, 1, 1, 1]
}
fn yr() -> Array1<f64> {
    array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
}

/// sklearn `ExtraTreeClassifier(random_state=0).fit(X_nan, y)` raises
/// `ValueError: Input X contains NaN.` (`validation.py:147-154`); ferrolearn
/// returns `Ok`. Tracking: #2278.
#[test]
#[ignore = "divergence: reject-set ExtraTreeClassifier accepts NaN; sklearn raises ValueError; tracking #2278"]
fn divergence_extra_tree_classifier_nan_not_rejected() {
    let r = ExtraTreeClassifier::<f64>::new()
        .with_random_state(0)
        .fit(&x_nan(), &yc());
    assert!(
        r.is_err(),
        "sklearn raises ValueError(Input X contains NaN.); ferrolearn returned Ok"
    );
}

/// sklearn `ExtraTreeRegressor(random_state=0).fit(X_nan, y)` -> ValueError.
/// Tracking: #2278.
#[test]
#[ignore = "divergence: reject-set ExtraTreeRegressor accepts NaN; sklearn raises ValueError; tracking #2278"]
fn divergence_extra_tree_regressor_nan_not_rejected() {
    let r = ExtraTreeRegressor::<f64>::new()
        .with_random_state(0)
        .fit(&x_nan(), &yr());
    assert!(
        r.is_err(),
        "sklearn raises ValueError; ferrolearn returned Ok"
    );
}

/// sklearn `ExtraTreesClassifier(...).fit(X_nan, y)` -> ValueError. Tracking: #2278.
#[test]
#[ignore = "divergence: reject-set ExtraTreesClassifier accepts NaN; sklearn raises ValueError; tracking #2278"]
fn divergence_extra_trees_classifier_nan_not_rejected() {
    let r = ExtraTreesClassifier::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &yc());
    assert!(
        r.is_err(),
        "sklearn raises ValueError; ferrolearn returned Ok"
    );
}

/// sklearn `ExtraTreesRegressor(...).fit(X_nan, y)` -> ValueError. Tracking: #2278.
#[test]
#[ignore = "divergence: reject-set ExtraTreesRegressor accepts NaN; sklearn raises ValueError; tracking #2278"]
fn divergence_extra_trees_regressor_nan_not_rejected() {
    let r = ExtraTreesRegressor::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &yr());
    assert!(
        r.is_err(),
        "sklearn raises ValueError; ferrolearn returned Ok"
    );
}

/// sklearn `AdaBoostClassifier(...).fit(X_nan, y)` -> ValueError. Tracking: #2278.
#[test]
#[ignore = "divergence: reject-set AdaBoostClassifier accepts NaN; sklearn raises ValueError; tracking #2278"]
fn divergence_adaboost_classifier_nan_not_rejected() {
    let r = AdaBoostClassifier::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &yc());
    assert!(
        r.is_err(),
        "sklearn raises ValueError; ferrolearn returned Ok"
    );
}

/// sklearn `AdaBoostRegressor(...).fit(X_nan, y)` -> ValueError. Tracking: #2278.
#[test]
#[ignore = "divergence: reject-set AdaBoostRegressor accepts NaN; sklearn raises ValueError; tracking #2278"]
fn divergence_adaboost_regressor_nan_not_rejected() {
    let r = AdaBoostRegressor::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &yr());
    assert!(
        r.is_err(),
        "sklearn raises ValueError; ferrolearn returned Ok"
    );
}

/// sklearn `GradientBoostingClassifier(...).fit(X_nan, y)` -> ValueError.
/// Tracking: #2278.
#[test]
#[ignore = "divergence: reject-set GradientBoostingClassifier accepts NaN; sklearn raises ValueError; tracking #2278"]
fn divergence_gradient_boosting_classifier_nan_not_rejected() {
    let r = GradientBoostingClassifier::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &yc());
    assert!(
        r.is_err(),
        "sklearn raises ValueError; ferrolearn returned Ok"
    );
}

/// sklearn `GradientBoostingRegressor(...).fit(X_nan, y)` -> ValueError.
/// Tracking: #2278.
#[test]
#[ignore = "divergence: reject-set GradientBoostingRegressor accepts NaN; sklearn raises ValueError; tracking #2278"]
fn divergence_gradient_boosting_regressor_nan_not_rejected() {
    let r = GradientBoostingRegressor::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &yr());
    assert!(
        r.is_err(),
        "sklearn raises ValueError; ferrolearn returned Ok"
    );
}

/// sklearn `IsolationForest(...).fit(X_nan)` -> ValueError. Tracking: #2278.
#[test]
#[ignore = "divergence: reject-set IsolationForest accepts NaN; sklearn raises ValueError; tracking #2278"]
fn divergence_isolation_forest_nan_not_rejected() {
    let r = IsolationForest::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &());
    assert!(
        r.is_err(),
        "sklearn raises ValueError; ferrolearn returned Ok"
    );
}
