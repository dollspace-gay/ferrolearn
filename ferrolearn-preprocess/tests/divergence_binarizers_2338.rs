//! Divergence pins for `LabelBinarizer` / `MultiLabelBinarizer` vs scikit-learn
//! 1.5.2 (#2338).
//!
//! Each test encodes a LIVE sklearn 1.5.2 oracle value (computed via
//! `python3 -c "import sklearn; ..."`, NEVER copied from ferrolearn) and asserts
//! it against the current ferrolearn implementation. The `#[ignore]`'d tests in
//! this file are expected to FAIL against ferrolearn HEAD: they are the audit
//! artifact pinning the divergence, not a passing regression suite.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::label_binarizer::LabelBinarizer;
use ferrolearn_preprocess::multi_label_binarizer::MultiLabelBinarizer;
use ndarray::{array, Array2};

/// Divergence: `MultiLabelBinarizer::fit` on an EMPTY label-set list diverges
/// from `sklearn/preprocessing/_label.py:779` (`MultiLabelBinarizer.fit`):
/// sklearn `fit([])` SUCCEEDS with `classes_ == []` (no error), and
/// `transform([[]])` returns a `(1, 0)` array. ferrolearn returns
/// `Err(FerroError::InsufficientSamples)` from `fit` (it rejects empty `y`).
///
/// Live oracle (sklearn 1.5.2, from /tmp):
///   ```text
///   mlb = MultiLabelBinarizer().fit([])
///   mlb.classes_.tolist()                  -> []
///   mlb.transform([[]]).shape              -> (1, 0)
///   ```
/// sklearn returns: fit OK, classes_ = [].
/// ferrolearn returns: Err(InsufficientSamples).
///
/// Tracking: #2339
#[test]
#[ignore = "divergence: MultiLabelBinarizer.fit([]) errors instead of empty classes_; tracking #2339"]
fn divergence_mlb_empty_fit_succeeds() {
    let mlb = MultiLabelBinarizer::new();
    let y: Vec<Vec<usize>> = vec![];
    let fitted = mlb.fit(&y, &());
    // sklearn: fit succeeds with classes_ == [].
    assert!(
        fitted.is_ok(),
        "sklearn MultiLabelBinarizer().fit([]) succeeds with classes_=[]; ferrolearn errored"
    );
    let fitted = fitted.unwrap();
    assert_eq!(fitted.classes(), &[] as &[usize]);
    // transform([[]]) -> (1, 0) array per sklearn.
    let out = fitted.transform(&vec![vec![]]).unwrap();
    assert_eq!(out.shape(), &[1, 0]);
}

/// Divergence: `FittedLabelBinarizer::inverse_transform` on a binary-fitted
/// (k == 2) binarizer REJECTS a 2-column indicator, but sklearn
/// `_inverse_binarize_thresholding` (`sklearn/preprocessing/_label.py:651-656`)
/// ACCEPTS it: `if y.ndim == 2 and y.shape[1] == 2: return classes[y[:, 1]]`.
/// sklearn's `LabelBinarizer.inverse_transform` dispatches on the fitted
/// `y_type_` ("binary"), NOT on the column count, so a (n, 2) matrix is a valid
/// binary inverse input and is decoded via its second column. ferrolearn
/// hard-requires `expected_cols == 1` for k == 2 and returns
/// `Err(FerroError::ShapeMismatch)`.
///
/// Live oracle (sklearn 1.5.2, from /tmp):
///   ```text
///   lb = LabelBinarizer().fit([10, 20])
///   lb.inverse_transform(np.array([[1.,0.],[0.,1.]])).tolist()  -> [10, 20]
///   ```
/// sklearn returns: [10, 20].
/// ferrolearn returns: Err(ShapeMismatch).
///
/// Tracking: #2340
#[test]
#[ignore = "divergence: LabelBinarizer.inverse_transform rejects 2-col binary input; tracking #2340"]
fn divergence_lb_binary_inverse_two_column() {
    let lb = LabelBinarizer::new();
    let fitted = lb.fit(&array![10_usize, 20], &()).unwrap();
    // (n, 2) indicator: row0 -> col0 active -> class 10; row1 -> col1 active -> class 20.
    let y: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]];
    let recovered = fitted.inverse_transform(&y);
    // sklearn-oracle: [10, 20].
    assert_eq!(
        recovered.ok(),
        Some(array![10_usize, 20]),
        "sklearn decodes a 2-col binary indicator as classes[y[:,1]]; ferrolearn errored"
    );
}

/// Guard (CLEAN MATCH, expected to PASS): single-class (k == 1)
/// `inverse_transform` value parity. ferrolearn computes
/// `expected_cols = if k == 2 { 1 } else { k }` -> for k == 1 it requires 1
/// column and runs the MULTICLASS argmax branch, returning `classes[argmax]`.
/// sklearn dispatches on `y_type_` ("binary") to `_inverse_binarize_thresholding`
/// (`sklearn/preprocessing/_label.py:657-661`): `if len(classes) == 1:
/// return np.repeat(classes[0], len(y))`. The VALUES coincide (both yield
/// `classes[0]` for every row), so this guard pins the matching sklearn value;
/// if it ever fails, the k == 1 dispatch has regressed.
///
/// Live oracle (sklearn 1.5.2, from /tmp):
///   ```text
///   lb = LabelBinarizer().fit([5, 5, 5])         # classes_ = [5]
///   lb.inverse_transform(np.array([[1],[0]])).tolist()  -> [5, 5]
///   ```
/// sklearn returns: [5, 5].
#[test]
fn lb_single_class_inverse_matches() {
    let lb = LabelBinarizer::new();
    let fitted = lb.fit(&array![5_usize, 5, 5], &()).unwrap();
    let y: Array2<f64> = array![[1.0], [0.0]];
    let recovered = fitted.inverse_transform(&y).unwrap();
    // sklearn-oracle: [5, 5].
    assert_eq!(recovered, array![5_usize, 5]);
}
