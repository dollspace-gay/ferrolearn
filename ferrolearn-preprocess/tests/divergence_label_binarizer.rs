//! Divergence pins for `LabelBinarizer` vs scikit-learn 1.5.2.
//!
//! Discriminator artifact (ACToR critic). Every expected value below comes
//! from a LIVE sklearn 1.5.2 oracle call (run from /tmp) or a sklearn
//! `file:line` symbolic constant — never copied from the ferrolearn side
//! (R-CHAR-3).
//!
//! Three pins (un-ignored, release-blockers) assert sklearn behavior and
//! currently FAIL against ferrolearn. Four green-guards lock SHIPPED behavior.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::label_binarizer::LabelBinarizer;
use ndarray::{Array2, array};

// ===========================================================================
// DIV-1 (REQ-4, HEADLINE): transform of an unseen label.
//
// sklearn `label_binarize` SILENTLY IGNORES labels not in `classes`, leaving
// that row all-`neg_label` (=0); NO error, NO warning:
//   `sklearn/preprocessing/_label.py:556-558`
//     y_in_classes = np.isin(y, classes)
//     y_seen = y[y_in_classes]
//     indices = np.searchsorted(sorted_class, y_seen)
//
// Live oracle (sklearn 1.5.2, from /tmp):
//   LabelBinarizer().fit([0,1,2]).transform([0,3]).tolist()
//     -> [[1, 0, 0], [0, 0, 0]]   (label 3 ignored, row 1 all-zero)
//
// ferrolearn `transform` returns Err(InvalidParameter) for the unseen label
// (`ferrolearn-preprocess/src/label_binarizer.rs:200-203`).
// ===========================================================================
#[test]
fn divergence_transform_unknown_label_silently_ignored() {
    let lb = LabelBinarizer::new();
    let fitted = lb.fit(&array![0_usize, 1, 2], &()).unwrap();

    // sklearn-oracle value: [[1,0,0],[0,0,0]]
    let expected: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

    let got = fitted
        .transform(&array![0_usize, 3])
        .expect("sklearn ignores unseen labels (all-zero row), does not error");
    assert_eq!(got, expected);
}

// ===========================================================================
// DIV-2 (REQ-5): single-class (n_classes == 1) transform.
//
// sklearn treats n_classes==1 as the binary degenerate case and returns an
// all-`neg_label` (=0) single column:
//   `sklearn/preprocessing/_label.py:532-538`
//     if n_classes == 1:
//         Y = np.zeros((len(y), 1), dtype=int)
//         Y += neg_label
//         return Y
//
// Live oracle (sklearn 1.5.2, from /tmp):
//   LabelBinarizer().fit_transform([5,5,5]).tolist() -> [[0], [0], [0]]
//
// ferrolearn routes k==1 into the one-hot else-branch -> all-1.0 column
// (`ferrolearn-preprocess/src/label_binarizer.rs:193-206`).
// ===========================================================================
#[test]
fn divergence_single_class_all_zero_column() {
    let lb = LabelBinarizer::new();
    let y = array![5_usize, 5, 5];
    let fitted = lb.fit(&y, &()).unwrap();

    // sklearn-oracle value: [[0],[0],[0]] shape (3,1)
    let expected: Array2<f64> = array![[0.0], [0.0], [0.0]];

    let got = fitted.transform(&y).unwrap();
    assert_eq!(got.shape(), &[3, 1]);
    assert_eq!(got, expected);
}

// ===========================================================================
// DIV-3 (REQ-6): binary inverse_transform uses a STRICT threshold.
//
// sklearn `_inverse_binarize_thresholding` binarizes with a STRICT comparison
// `Y > threshold` (default threshold = 0.5), so exactly-0.5 -> class[0]:
//   `sklearn/preprocessing/_label.py:667`  y = np.array(y > threshold, dtype=int)
//   `sklearn/preprocessing/_label.py:679`  return classes[y.ravel()]
//
// Live oracle (sklearn 1.5.2, from /tmp), lb = fit([0,1]):
//   inverse_transform([[0.5]]) -> [0]
//   inverse_transform([[0.6]]) -> [1]
//   inverse_transform([[0.4]]) -> [0]
//
// ferrolearn uses non-strict `y[[i,0]] >= 0.5`
// (`ferrolearn-preprocess/src/label_binarizer.rs:99`), so 0.5 -> class[1] (=1).
// ===========================================================================
#[test]
fn divergence_binary_inverse_strict_threshold_at_half() {
    let lb = LabelBinarizer::new();
    let fitted = lb.fit(&array![0_usize, 1], &()).unwrap();

    // sklearn-oracle: exactly 0.5 -> class 0 (strict `> 0.5`)
    let at_half = fitted.inverse_transform(&array![[0.5_f64]]).unwrap();
    assert_eq!(at_half, array![0_usize]);

    // green-guards on the same fixture (sklearn-oracle values)
    let above = fitted.inverse_transform(&array![[0.6_f64]]).unwrap();
    assert_eq!(above, array![1_usize]);

    let below = fitted.inverse_transform(&array![[0.4_f64]]).unwrap();
    assert_eq!(below, array![0_usize]);
}

// ===========================================================================
// GREEN GUARDS — SHIPPED behavior, locked against sklearn-oracle values.
// ===========================================================================

/// REQ-1: fit discovers sorted unique classes.
/// Live oracle: LabelBinarizer().fit([0,1,2,1]).classes_.tolist() -> [0,1,2]
#[test]
fn green_req1_fit_sorted_unique_classes() {
    let lb = LabelBinarizer::new();
    let fitted = lb.fit(&array![0_usize, 1, 2, 1], &()).unwrap();
    assert_eq!(fitted.classes(), &[0, 1, 2]);
    assert_eq!(fitted.n_classes(), 3);
}

/// REQ-2 (PROBE A): multiclass transform one-hot.
/// Live oracle: fit([0,1,2,1]).transform([0,1,2,1]).tolist()
///   -> [[1,0,0],[0,1,0],[0,0,1],[0,1,0]]
#[test]
fn green_req2_multiclass_transform_onehot() {
    let lb = LabelBinarizer::new();
    let y = array![0_usize, 1, 2, 1];
    let fitted = lb.fit(&y, &()).unwrap();
    let expected: Array2<f64> = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ];
    assert_eq!(fitted.transform(&y).unwrap(), expected);
}

/// REQ-3 (PROBE B): binary transform single column.
/// Live oracle: fit([0,1,0,1]).transform([0,1,0,1]).tolist()
///   -> [[0],[1],[0],[1]]
#[test]
fn green_req3_binary_transform_single_column() {
    let lb = LabelBinarizer::new();
    let y = array![0_usize, 1, 0, 1];
    let fitted = lb.fit(&y, &()).unwrap();
    let expected: Array2<f64> = array![[0.0], [1.0], [0.0], [1.0]];
    assert_eq!(fitted.transform(&y).unwrap(), expected);
}

/// REQ-7: multiclass inverse round-trip.
/// Live oracle: lb.inverse_transform(lb.transform([0,1,2,1])).tolist()
///   -> [0,1,2,1]
#[test]
fn green_req7_multiclass_inverse_roundtrip() {
    let lb = LabelBinarizer::new();
    let y = array![0_usize, 1, 2, 1];
    let fitted = lb.fit(&y, &()).unwrap();
    let mat = fitted.transform(&y).unwrap();
    let recovered = fitted.inverse_transform(&mat).unwrap();
    assert_eq!(recovered, y);
}
