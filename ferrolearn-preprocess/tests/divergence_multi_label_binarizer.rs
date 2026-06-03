//! Divergence tests for `MultiLabelBinarizer` vs scikit-learn 1.5.2
//! `sklearn/preprocessing/_label.py` (`class MultiLabelBinarizer`, :688).
//!
//! All expected values come from a LIVE sklearn 1.5.2 oracle call (run from
//! /tmp) or a sklearn `file:line` citation — never copied from the ferrolearn
//! side (R-CHAR-3).

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::multi_label_binarizer::MultiLabelBinarizer;
use ndarray::array;

// ===========================================================================
// GREEN GUARDS — SHIPPED behavior (must PASS)
// ===========================================================================

/// REQ-1: fit discovers sorted unique classes.
///
/// Live oracle (PROBE A):
/// `MultiLabelBinarizer().fit([[0,2],[1],[0,1,2]]).classes_.tolist()`
/// -> `[0, 1, 2]`, `len == 3`.
#[test]
fn green_req1_fit_classes() {
    let mlb = MultiLabelBinarizer::new();
    let y = vec![vec![0, 2], vec![1], vec![0, 1, 2]];
    let fitted = mlb.fit(&y, &()).unwrap();
    assert_eq!(fitted.classes(), &[0, 1, 2]);
    assert_eq!(fitted.n_classes(), 3);
}

/// REQ-2: transform produces the dense multi-hot indicator.
///
/// Live oracle (PROBE A):
/// `MultiLabelBinarizer().fit_transform([[0,2],[1],[0,1,2]]).tolist()`
/// -> `[[1, 0, 1], [0, 1, 0], [1, 1, 1]]`.
#[test]
fn green_req2_transform_multi_hot() {
    let mlb = MultiLabelBinarizer::new();
    let y = vec![vec![0, 2], vec![1], vec![0, 1, 2]];
    let fitted = mlb.fit(&y, &()).unwrap();
    let mat = fitted.transform(&y).unwrap();
    let expected = array![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
    assert_eq!(mat, expected);
}

/// REQ-4 (valid path): inverse_transform of a 0/1 indicator.
///
/// Live oracle (PROBE D):
/// `MultiLabelBinarizer().fit([[0,1,2]]).inverse_transform([[1.,0.,1.]])`
/// -> `[(0, 2)]`.
#[test]
fn green_req4_inverse_valid() {
    let mlb = MultiLabelBinarizer::new();
    let y = vec![vec![0, 1, 2]];
    let fitted = mlb.fit(&y, &()).unwrap();
    let recovered = fitted.inverse_transform(&array![[1.0, 0.0, 1.0]]).unwrap();
    assert_eq!(recovered, vec![vec![0, 2]]);
}

// ===========================================================================
// DIVERGENCE PINS — release-blockers (must FAIL against current ferrolearn)
// ===========================================================================

/// Divergence (REQ-3, HEADLINE): `FittedMultiLabelBinarizer::transform`
/// ERRORS on an unknown label
/// (`ferrolearn-preprocess/src/multi_label_binarizer.rs:170-176`,
/// returns `FerroError::InvalidParameter`), whereas sklearn COLLECTS and
/// IGNORES unknown labels, building the indicator from only the known labels
/// and never raising
/// (`sklearn/preprocessing/_label.py:889-902`:
/// `unknown.add(label)` / `warnings.warn("unknown class(es) ... ignored")`).
///
/// The warning has no Rust analog (no log/tracing dep). The BEHAVIORAL
/// contract pinned here is: unknown labels are skipped, indicator built from
/// the known labels, no error.
///
/// Input: `fit([[0,1]]).transform([[0,5]])` (label 5 unknown).
/// Live oracle: `mlb.transform([[0,5]]).tolist()` -> `[[1, 0]]`.
/// ferrolearn: returns `Err(InvalidParameter)`.
/// Tracking: blocker filed.
#[test]
fn divergence_req3_transform_unknown_label_ignored() {
    let mlb = MultiLabelBinarizer::new();
    let fitted = mlb.fit(&vec![vec![0, 1]], &()).unwrap();
    let out = fitted
        .transform(&vec![vec![0, 5]])
        .expect("sklearn ignores unknown labels rather than erroring");
    // sklearn -> [[1, 0]] : label 0 kept, label 5 ignored.
    assert_eq!(out, array![[1.0, 0.0]]);
}

/// Divergence (REQ-3, second sample): unknown labels are dropped per-sample
/// while known labels in other samples are preserved.
///
/// Input: `fit([[0,1,2]]).transform([[2,9],[1]])` (label 9 unknown).
/// Live oracle: `mlb.transform([[2,9],[1]]).tolist()` -> `[[0, 0, 1], [0, 1, 0]]`.
/// ferrolearn: returns `Err(InvalidParameter)`.
/// Tracking: blocker filed.
#[test]
fn divergence_req3_transform_unknown_label_multi_sample() {
    let mlb = MultiLabelBinarizer::new();
    let fitted = mlb.fit(&vec![vec![0, 1, 2]], &()).unwrap();
    let out = fitted
        .transform(&vec![vec![2, 9], vec![1]])
        .expect("sklearn ignores unknown labels rather than erroring");
    assert_eq!(out, array![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]);
}

/// Divergence (REQ-4): `FittedMultiLabelBinarizer::inverse_transform`
/// thresholds arbitrary floats at 0.5 and SUCCEEDS
/// (`ferrolearn-preprocess/src/multi_label_binarizer.rs:96`: `y[[i, j]] >= 0.5`),
/// whereas sklearn RAISES `ValueError("Expected only 0s and 1s in label
/// indicator...")` on any value not in {0, 1}
/// (`sklearn/preprocessing/_label.py:941-947`:
/// `unexpected = np.setdiff1d(yt, [0, 1])` -> raise if non-empty).
///
/// Only the 0/1 strict-validation is pinned (not the tuple-vs-Vec return type).
///
/// Input: `fit([[0,1,2]]).inverse_transform([[0.4, 0.6, 0.5]])`.
/// Live oracle: raises `ValueError("Expected only 0s and 1s in label
/// indicator. Also got [0.4 0.5 0.6]")`.
/// ferrolearn: returns `Ok(vec![vec![1, 2]])`.
/// Tracking: blocker filed.
#[test]
fn divergence_req4_inverse_rejects_non_01() {
    let mlb = MultiLabelBinarizer::new();
    let fitted = mlb.fit(&vec![vec![0, 1, 2]], &()).unwrap();
    let result = fitted.inverse_transform(&array![[0.4, 0.6, 0.5]]);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on non-0/1 indicator values; \
         ferrolearn 0.5-thresholds and succeeds with {result:?}"
    );
}
