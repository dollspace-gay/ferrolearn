//! Divergence pins (R-CODE-2 release-blocker): ferrolearn-tree's ACCEPT-set
//! DecisionTree* and Bagging* builders STACK-OVERFLOW (process abort, SIGABRT)
//! when `fit` is called with NaN in `X`, whereas scikit-learn 1.5.2 ACCEPTS NaN
//! and returns a fitted model.
//!
//! Root cause: the recursive CART builder (`build_classification_tree` /
//! `build_regression_tree` in `decision_tree.rs`) chooses a split threshold and
//! partitions with `x[[i, feature]] <= threshold`. A NaN feature value (or a
//! NaN-derived threshold) makes every `<=` comparison `false`, so a node of `n`
//! samples partitions into `(n, 0)` — the recursion descends on the SAME `n`
//! samples without shrinking, unbounded, until the stack overflows and the
//! process aborts. Bagging* fits DecisionTree base learners, so it inherits the
//! abort.
//!
//! sklearn contract: the tree base passes `force_all_finite=False`
//! (`sklearn/tree/_classes.py:248-250`) and supports missing values natively
//! (`_compute_missing_values_in_feature_mask`, `_classes.py:256-258`), so
//! `DecisionTreeClassifier(random_state=0).fit(X_nan, y)` returns a fitted model
//! (live oracle: FIT-OK). RandomForest*/HistGB* also accept NaN and do NOT abort
//! in ferrolearn (separate code paths) — those are documented NOT-STARTED
//! missing-value-support value-mismatches, not abort blockers, and are NOT
//! pinned here.
//!
//! NOTE ON THE ARTIFACT: a Rust stack overflow is uncatchable (it raises SIGABRT
//! and aborts the whole test process — `catch_unwind` and a bounded-stack thread
//! both still abort). These tests are therefore `#[ignore]`d so the default
//! suite stays green; running them with `--ignored` ABORTS the test binary
//! (signal 6), which is the failing artifact that pins the blocker. They go
//! green only when `fit(X_nan, ...)` returns (Ok, per sklearn accept-set
//! semantics, or a clean `Err` — either way: no abort).
//!
//! Tracking: #2277.

use ferrolearn_core::Fit;
use ferrolearn_tree::{
    BaggingClassifier, BaggingRegressor, DecisionTreeClassifier, DecisionTreeRegressor,
};
use ndarray::{array, Array1, Array2};

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

/// sklearn `DecisionTreeClassifier(random_state=0).fit(X_nan, y)` returns a
/// fitted model (accepts NaN, `_classes.py:248-250` `force_all_finite=False`);
/// ferrolearn STACK-OVERFLOWS (process abort). Tracking: #2277.
#[test]
#[ignore = "divergence(R-CODE-2): DecisionTreeClassifier.fit(NaN) stack-overflow abort; sklearn accepts NaN; tracking #2277"]
fn divergence_decision_tree_classifier_nan_stack_overflow() {
    // Reaching this call aborts the process (SIGABRT) on current ferrolearn.
    let r = DecisionTreeClassifier::<f64>::new().fit(&x_nan(), &yc());
    assert!(
        r.is_ok() || r.is_err(),
        "fit must return (sklearn accepts NaN), not abort via stack overflow"
    );
}

/// sklearn `DecisionTreeRegressor(random_state=0).fit(X_nan, y)` returns a
/// fitted model; ferrolearn STACK-OVERFLOWS. Tracking: #2277.
#[test]
#[ignore = "divergence(R-CODE-2): DecisionTreeRegressor.fit(NaN) stack-overflow abort; sklearn accepts NaN; tracking #2277"]
fn divergence_decision_tree_regressor_nan_stack_overflow() {
    let r = DecisionTreeRegressor::<f64>::new().fit(&x_nan(), &yr());
    assert!(
        r.is_ok() || r.is_err(),
        "fit must return (sklearn accepts NaN), not abort via stack overflow"
    );
}

/// sklearn `BaggingClassifier(...).fit(X_nan, y)` returns a fitted model
/// (DecisionTree base accepts NaN); ferrolearn STACK-OVERFLOWS. Tracking: #2277.
#[test]
#[ignore = "divergence(R-CODE-2): BaggingClassifier.fit(NaN) stack-overflow abort; sklearn accepts NaN; tracking #2277"]
fn divergence_bagging_classifier_nan_stack_overflow() {
    let r = BaggingClassifier::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &yc());
    assert!(
        r.is_ok() || r.is_err(),
        "fit must return (sklearn accepts NaN), not abort via stack overflow"
    );
}

/// sklearn `BaggingRegressor(...).fit(X_nan, y)` returns a fitted model;
/// ferrolearn STACK-OVERFLOWS. Tracking: #2277.
#[test]
#[ignore = "divergence(R-CODE-2): BaggingRegressor.fit(NaN) stack-overflow abort; sklearn accepts NaN; tracking #2277"]
fn divergence_bagging_regressor_nan_stack_overflow() {
    let r = BaggingRegressor::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &yr());
    assert!(
        r.is_ok() || r.is_err(),
        "fit must return (sklearn accepts NaN), not abort via stack overflow"
    );
}
