//! Native NaN support: the ACCEPT-set DecisionTree* (and the Bagging* learners
//! that build them) no longer STACK-OVERFLOW on `fit(X_nan, ...)`; they accept
//! NaN and produce a fitted model, matching scikit-learn 1.5.2.
//!
//! Root cause (now FIXED): the recursive CART builder partitioned with
//! `x[[i, feature]] <= threshold`, which sends every NaN right (`NaN <= t` is
//! `false`); a NaN-derived split could yield `(n, 0)` and the recursion
//! descended on the same `n` samples unbounded until the stack overflowed.
//! `decision_tree.rs` now implements sklearn's native missing-value splitter
//! (`node_split_best`, `_splitter.pyx:293`): NaN sorts last, the best split
//! records a per-node `missing_go_to_left` direction (`tree_.missing_go_to_left`,
//! `_tree.pyx:746`), the partition + predict route NaN to that direction
//! (`_apply_dense`, `_tree.pyx:1015-1025`), so a node's samples always shrink
//! and the recursion terminates.
//!
//! sklearn contract: `force_all_finite=False` (`sklearn/tree/_classes.py:248-250`)
//! ⇒ `DecisionTree*.fit(X_nan, y)` returns a fitted model. The DecisionTree pins
//! below now assert sklearn PARITY (fit succeeds AND predict matches sklearn);
//! the Bagging pins (bootstrap + RNG, so exact predictions are RNG-dependent and
//! not array-compared) assert fit SUCCEEDS — they inherit the DecisionTree fix.
//!
//! Tracking: #2277.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_tree::{
    BaggingClassifier, BaggingRegressor, DecisionTreeClassifier, DecisionTreeRegressor,
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

/// sklearn `DecisionTreeClassifier().fit(X_nan, y)` succeeds; splits on the clean
/// feature 1 (`tree_.feature[0]==1`, `threshold==4.5`); `predict == y`.
/// (oracle: `DecisionTreeClassifier().fit(X,y); c.tree_.feature[0]==1,
/// c.tree_.threshold[0]==4.5, c.predict(X)==[0,0,0,1,1,1]`.) Was a stack-overflow
/// abort. Tracking: #2277.
#[test]
fn divergence_decision_tree_classifier_nan_stack_overflow() {
    let fitted = DecisionTreeClassifier::<f64>::new()
        .fit(&x_nan(), &yc())
        .expect("fit must accept NaN (sklearn force_all_finite=False), not abort");
    // Parity with sklearn (split on the clean feature, exact predictions).
    assert_eq!(fitted.predict(&x_nan()).unwrap(), array![0, 0, 0, 1, 1, 1]);
}

/// sklearn `DecisionTreeRegressor().fit(X_nan, y)` succeeds; splits on the clean
/// feature 1 (`threshold==4.5`); `predict == y`. Was a stack-overflow abort.
/// Tracking: #2277.
#[test]
fn divergence_decision_tree_regressor_nan_stack_overflow() {
    let fitted = DecisionTreeRegressor::<f64>::new()
        .fit(&x_nan(), &yr())
        .expect("fit must accept NaN, not abort");
    let preds = fitted.predict(&x_nan()).unwrap();
    for (p, e) in preds.iter().zip([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
        assert!((p - e).abs() < 1e-12, "{p} != {e}");
    }
}

/// `BaggingClassifier` builds DecisionTree base learners; it inherits the
/// missing-value fix and no longer overflows. sklearn
/// `BaggingClassifier(...).fit(X_nan, y)` returns a fitted model (bootstrap + RNG
/// ⇒ exact predictions are RNG-dependent, so only fit-success is asserted).
/// Tracking: #2277.
#[test]
fn divergence_bagging_classifier_nan_stack_overflow() {
    let fitted = BaggingClassifier::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &yc())
        .expect("fit must accept NaN (DecisionTree base inherits the fix), not abort");
    // Smoke: predict runs without panic / abort on the NaN data.
    let preds = fitted.predict(&x_nan()).unwrap();
    assert_eq!(preds.len(), 6);
}

/// `BaggingRegressor` builds DecisionTree base learners; inherits the fix. Was a
/// stack-overflow abort. Tracking: #2277.
#[test]
fn divergence_bagging_regressor_nan_stack_overflow() {
    let fitted = BaggingRegressor::<f64>::new()
        .with_n_estimators(3)
        .with_random_state(0)
        .fit(&x_nan(), &yr())
        .expect("fit must accept NaN, not abort");
    let preds = fitted.predict(&x_nan()).unwrap();
    assert_eq!(preds.len(), 6);
}
