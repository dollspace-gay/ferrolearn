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

// ---------------------------------------------------------------------------
// Infinity rejection: the reject-set rejects Inf too (the full finite-check is
// `!v.is_finite()`, covering both NaN and ±Inf). sklearn's default
// `force_all_finite=True` likewise raises `ValueError("Input X contains
// infinity ...")` (`sklearn/utils/validation.py:147-154`).
// ---------------------------------------------------------------------------

fn x_inf() -> Array2<f64> {
    let mut x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
    )
    .unwrap();
    x[[0, 0]] = f64::INFINITY;
    x
}

fn x_finite() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
    )
    .unwrap()
}

/// sklearn raises `ValueError: Input X contains infinity ...` for every
/// reject-set estimator (default `force_all_finite=True`); ferrolearn's
/// full finite-check rejects ±Inf too. Tracking: #2278.
#[test]
fn divergence_reject_set_inf_rejected() {
    let x = x_inf();
    assert!(
        ExtraTreeClassifier::<f64>::new()
            .with_random_state(0)
            .fit(&x, &yc())
            .is_err(),
        "ExtraTreeClassifier must reject Inf"
    );
    assert!(
        ExtraTreeRegressor::<f64>::new()
            .with_random_state(0)
            .fit(&x, &yr())
            .is_err(),
        "ExtraTreeRegressor must reject Inf"
    );
    assert!(
        ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yc())
            .is_err(),
        "ExtraTreesClassifier must reject Inf"
    );
    assert!(
        ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yr())
            .is_err(),
        "ExtraTreesRegressor must reject Inf"
    );
    assert!(
        AdaBoostClassifier::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yc())
            .is_err(),
        "AdaBoostClassifier must reject Inf"
    );
    assert!(
        AdaBoostRegressor::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yr())
            .is_err(),
        "AdaBoostRegressor must reject Inf"
    );
    assert!(
        GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yc())
            .is_err(),
        "GradientBoostingClassifier must reject Inf"
    );
    assert!(
        GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yr())
            .is_err(),
        "GradientBoostingRegressor must reject Inf"
    );
    assert!(
        IsolationForest::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &())
            .is_err(),
        "IsolationForest must reject Inf"
    );
}

/// The regressors reject a non-finite FLOAT target y too: sklearn validates y
/// via `_validate_data(..., y_numeric=True)` (`_weight_boosting.py:140`,
/// `_gb.py:660`, the ExtraTree(s) `_classes.py`/`_forest.py` y-path). A NaN in
/// y must raise. Tracking: #2278.
#[test]
fn divergence_regressors_reject_nan_in_y() {
    let x = x_finite();
    let mut y = yr();
    y[0] = f64::NAN;
    assert!(
        ExtraTreeRegressor::<f64>::new()
            .with_random_state(0)
            .fit(&x, &y)
            .is_err(),
        "ExtraTreeRegressor must reject NaN in y"
    );
    assert!(
        ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &y)
            .is_err(),
        "ExtraTreesRegressor must reject NaN in y"
    );
    assert!(
        AdaBoostRegressor::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &y)
            .is_err(),
        "AdaBoostRegressor must reject NaN in y"
    );
    assert!(
        GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &y)
            .is_err(),
        "GradientBoostingRegressor must reject NaN in y"
    );
}

/// No false positive: finite input still fits `Ok` for every reject-set
/// estimator (the check only fires on non-finite values). Tracking: #2278.
#[test]
fn reject_set_finite_input_fits_ok() {
    let x = x_finite();
    assert!(
        ExtraTreeClassifier::<f64>::new()
            .with_random_state(0)
            .fit(&x, &yc())
            .is_ok()
    );
    assert!(
        ExtraTreeRegressor::<f64>::new()
            .with_random_state(0)
            .fit(&x, &yr())
            .is_ok()
    );
    assert!(
        ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yc())
            .is_ok()
    );
    assert!(
        ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yr())
            .is_ok()
    );
    assert!(
        AdaBoostClassifier::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yc())
            .is_ok()
    );
    assert!(
        AdaBoostRegressor::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yr())
            .is_ok()
    );
    assert!(
        GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yc())
            .is_ok()
    );
    assert!(
        GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &yr())
            .is_ok()
    );
    assert!(
        IsolationForest::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(0)
            .fit(&x, &())
            .is_ok()
    );
}

/// `predict(X_nan)` must NOT panic (R-CODE-2 / the neighbors lesson). sklearn
/// rejects NaN at predict for the reject-set, but the minimum ferrolearn
/// contract is no stack-overflow / OOB panic. The models are fit on finite
/// data, then queried with a NaN row. Tracking: #2278.
#[test]
fn reject_set_predict_nan_does_not_panic() {
    use ferrolearn_core::Predict;
    let x = x_finite();
    let q = x_nan();

    let etc = ExtraTreeClassifier::<f64>::new()
        .with_random_state(0)
        .fit(&x, &yc())
        .unwrap();
    let _ = etc.predict(&q);

    let etr = ExtraTreeRegressor::<f64>::new()
        .with_random_state(0)
        .fit(&x, &yr())
        .unwrap();
    let _ = etr.predict(&q);

    let etsc = ExtraTreesClassifier::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x, &yc())
        .unwrap();
    let _ = etsc.predict(&q);

    let etsr = ExtraTreesRegressor::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x, &yr())
        .unwrap();
    let _ = etsr.predict(&q);

    let iso = IsolationForest::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();
    let _ = iso.predict(&q);
}
