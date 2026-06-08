//! Divergence pins: ferrolearn-tree's `DecisionTreeClassifier` /
//! `DecisionTreeRegressor` ACCEPT `+Inf` / `-Inf` in `X` (treating them as
//! ordinary large finite values), whereas scikit-learn 1.5.2 raises
//! `ValueError("Input X contains infinity or a value too large for
//! dtype('float32').")` at BOTH `fit` and `predict`.
//!
//! The native missing-value feature (#2277) makes the DecisionTree base pass
//! `force_all_finite=False` to `_validate_data`
//! (`sklearn/tree/_classes.py:248-250`). Crucially this lowers to
//! `_assert_all_finite(..., allow_nan=True)` — which permits NaN (missing) but
//! STILL rejects infinities:
//!   `sklearn/utils/validation.py:172`
//!     `raise ValueError(msg_err)` where `msg_err` reports
//!     `"Input X contains infinity or a value too large for dtype('float32')."`
//! The `allow_nan=True` branch only suppresses the NaN error; `has_inf` is
//! always fatal (`sklearn/utils/validation.py:147-172`,
//! `_assert_all_finite_element_wise`). sklearn calls the same finite-check at
//! predict (`_classes.py` `predict` -> `_validate_X_predict`), so a `+Inf`
//! query row also raises.
//!
//! Live oracle (sklearn 1.5.2):
//!   clf fit  [[1],[2],[8],[9],[+inf]] y=[0,0,1,1,1] (max_depth=1)
//!            -> ValueError: Input X contains infinity ...
//!   clf fit  same with -inf            -> ValueError: Input X contains infinity ...
//!   clf pred fit finite, predict([[+inf]]) -> ValueError: Input X contains infinity ...
//!   reg fit  [[1],[2],[8],[9],[+inf]] y=[1,1,5,5,5] -> ValueError: Input X contains infinity ...
//!
//! ferrolearn's `sort_indices_by_feature` only special-cases `is_nan()`
//! (`decision_tree.rs:1340-1344`); `+Inf`/`-Inf` sort as ordinary extreme values
//! and `fit`/`predict` return `Ok(_)` (a silently divergent tree / prediction)
//! instead of the sklearn `ValueError`. Each test asserts `is_err()` — the
//! sklearn contract — and therefore FAILS against current ferrolearn (`Ok`).
//!
//! Every expected value is the live sklearn 1.5.2 oracle (R-CHAR-3), never
//! copied from ferrolearn. Tracking: #2279.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_tree::{DecisionTreeClassifier, DecisionTreeRegressor};
use ndarray::{Array1, Array2, array};

/// clf fit with +Inf in X: sklearn raises ValueError; ferrolearn returns Ok.
#[test]
fn clf_fit_pos_inf_rejected_like_sklearn() {
    let inf = f64::INFINITY;
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 8.0, 9.0, inf]).unwrap();
    let y = array![0, 0, 1, 1, 1];
    let res = DecisionTreeClassifier::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError('Input X contains infinity ...') for +Inf in X; \
         ferrolearn returned Ok"
    );
}

/// clf fit with -Inf in X: sklearn raises ValueError; ferrolearn returns Ok.
#[test]
fn clf_fit_neg_inf_rejected_like_sklearn() {
    let ninf = f64::NEG_INFINITY;
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 8.0, 9.0, ninf]).unwrap();
    let y = array![0, 0, 1, 1, 1];
    let res = DecisionTreeClassifier::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for -Inf in X; ferrolearn returned Ok"
    );
}

/// clf predict with +Inf in the query (trained on finite X): sklearn validates
/// predict input and raises ValueError; ferrolearn returns Ok.
#[test]
fn clf_predict_pos_inf_rejected_like_sklearn() {
    let inf = f64::INFINITY;
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 8.0, 9.0]).unwrap();
    let y = array![0, 0, 1, 1];
    let fitted = DecisionTreeClassifier::<f64>::new()
        .fit(&x, &y)
        .expect("finite fit succeeds");
    let q = Array2::from_shape_vec((1, 1), vec![inf]).unwrap();
    let res = fitted.predict(&q);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for +Inf in a predict query; ferrolearn returned Ok"
    );
}

/// reg fit with +Inf in X: sklearn raises ValueError; ferrolearn returns Ok.
#[test]
fn reg_fit_pos_inf_rejected_like_sklearn() {
    let inf = f64::INFINITY;
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 8.0, 9.0, inf]).unwrap();
    let y: Array1<f64> = array![1.0, 1.0, 5.0, 5.0, 5.0];
    let res = DecisionTreeRegressor::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for +Inf in X (regressor); ferrolearn returned Ok"
    );
}
