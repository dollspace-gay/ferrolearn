//! Divergence pins for `LabelBinarizer` vs scikit-learn 1.5.2.
//!
//! Discriminator artifact (ACToR critic). Every expected value below comes
//! from a LIVE sklearn 1.5.2 oracle call (run from /tmp) or a sklearn
//! `file:line` symbolic constant — never copied from the ferrolearn side
//! (R-CHAR-3).
//!
//! Three pins (un-ignored, release-blockers) assert sklearn behavior and
//! currently FAIL against ferrolearn. Four green-guards lock SHIPPED behavior.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::label_binarizer::{LabelBinarizer, label_binarize};
use ndarray::{Array1, Array2, array};

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

// ===========================================================================
// REQ-8: neg_label / pos_label ctor params + validation.
//
// sklearn `LabelBinarizer.__init__(*, neg_label=0, pos_label=1, ...)`
// (`_label.py:263`); the dense fill writes `pos_label` at active positions and
// `neg_label` everywhere else (`:579-583`); `fit` rejects `neg_label >=
// pos_label` verbatim (`:283-287`); `_inverse_binarize_thresholding` uses a
// STRICT `y > threshold` with `threshold = (pos_label + neg_label) / 2`
// (`:399-400`, `:667`).
//
// Every expected value below is a LIVE sklearn 1.5.2 oracle (run from /tmp).
// ===========================================================================

/// REQ-8: multiclass transform with neg_label=-1, pos_label=2.
/// Live oracle:
///   `LabelBinarizer(neg_label=-1,pos_label=2).fit([0,1,2]).transform([0,2]).tolist()`
///     -> `[[2, -1, -1], [-1, -1, 2]]`
#[test]
fn req8_neg_pos_multiclass_transform() {
    let lb = LabelBinarizer::new().with_neg_label(-1).with_pos_label(2);
    let fitted = lb.fit(&array![0_usize, 1, 2], &()).unwrap();
    let got = fitted.transform(&array![0_usize, 2]).unwrap();
    let expected: Array2<f64> = array![[2.0, -1.0, -1.0], [-1.0, -1.0, 2.0]];
    assert_eq!(got, expected);
}

/// REQ-8: binary (k==2) single-column transform with neg_label=-1, pos_label=1.
/// Live oracle:
///   `LabelBinarizer(neg_label=-1,pos_label=1).fit([0,1]).transform([0,1,0]).tolist()`
///     -> `[[-1], [1], [-1]]`
#[test]
fn req8_neg_pos_binary_single_column() {
    let lb = LabelBinarizer::new().with_neg_label(-1).with_pos_label(1);
    let fitted = lb.fit(&array![0_usize, 1], &()).unwrap();
    let got = fitted.transform(&array![0_usize, 1, 0]).unwrap();
    let expected: Array2<f64> = array![[-1.0], [1.0], [-1.0]];
    assert_eq!(got, expected);
}

/// REQ-8: single-class (k==1) transform -> all neg_label.
/// Live oracle:
///   `LabelBinarizer(neg_label=-1,pos_label=2).fit_transform([5,5,5]).tolist()`
///     -> `[[-1], [-1], [-1]]`
#[test]
fn req8_neg_pos_single_class_all_neg() {
    let lb = LabelBinarizer::new().with_neg_label(-1).with_pos_label(2);
    let y = array![5_usize, 5, 5];
    let fitted = lb.fit(&y, &()).unwrap();
    let got = fitted.transform(&y).unwrap();
    let expected: Array2<f64> = array![[-1.0], [-1.0], [-1.0]];
    assert_eq!(got, expected);
}

/// REQ-8: unseen label stays at neg_label (silent-ignore now -1).
/// Live oracle:
///   `LabelBinarizer(neg_label=-1,pos_label=2).fit([0,1,2]).transform([0,3]).tolist()`
///     -> `[[2, -1, -1], [-1, -1, -1]]`
#[test]
fn req8_neg_pos_unseen_label_stays_neg() {
    let lb = LabelBinarizer::new().with_neg_label(-1).with_pos_label(2);
    let fitted = lb.fit(&array![0_usize, 1, 2], &()).unwrap();
    let got = fitted.transform(&array![0_usize, 3]).unwrap();
    let expected: Array2<f64> = array![[2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]];
    assert_eq!(got, expected);
}

/// REQ-8: `neg_label >= pos_label` rejected at fit, verbatim message.
/// Live oracle:
///   `LabelBinarizer(neg_label=2,pos_label=1).fit([0,1])`
///     -> ValueError: "neg_label=2 must be strictly less than pos_label=1."
///   `LabelBinarizer(neg_label=1,pos_label=1).fit([0,1])`
///     -> ValueError: "neg_label=1 must be strictly less than pos_label=1."
#[test]
fn req8_neg_ge_pos_rejected_at_fit() {
    let err = LabelBinarizer::new()
        .with_neg_label(2)
        .with_pos_label(1)
        .fit(&array![0_usize, 1], &())
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Invalid parameter `neg_label`: neg_label=2 must be strictly less than pos_label=1."
    );

    let err = LabelBinarizer::new()
        .with_neg_label(1)
        .with_pos_label(1)
        .fit(&array![0_usize, 1], &())
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Invalid parameter `neg_label`: neg_label=1 must be strictly less than pos_label=1."
    );
}

/// REQ-8: inverse_transform binary STRICT threshold = (pos+neg)/2.
/// Live oracle (neg=-1,pos=1 -> threshold 0.0):
///   `inverse_transform([[0.0]])` -> [0]; `[[0.1]]` -> [1]; `[[-0.1]]` -> [0]
/// Live oracle (neg=2,pos=4 -> threshold 3.0):
///   `inverse_transform([[3.0]])` -> [0]; `[[3.1]]` -> [1]
#[test]
fn req8_neg_pos_inverse_threshold() {
    let fitted = LabelBinarizer::new()
        .with_neg_label(-1)
        .with_pos_label(1)
        .fit(&array![0_usize, 1], &())
        .unwrap();
    assert_eq!(
        fitted.inverse_transform(&array![[0.0_f64]]).unwrap(),
        array![0_usize]
    );
    assert_eq!(
        fitted.inverse_transform(&array![[0.1_f64]]).unwrap(),
        array![1_usize]
    );
    assert_eq!(
        fitted.inverse_transform(&array![[-0.1_f64]]).unwrap(),
        array![0_usize]
    );

    let fitted = LabelBinarizer::new()
        .with_neg_label(2)
        .with_pos_label(4)
        .fit(&array![0_usize, 1], &())
        .unwrap();
    assert_eq!(
        fitted.inverse_transform(&array![[3.0_f64]]).unwrap(),
        array![0_usize]
    );
    assert_eq!(
        fitted.inverse_transform(&array![[3.1_f64]]).unwrap(),
        array![1_usize]
    );
}

/// REQ-8: inverse_transform multiclass round-trip with neg/pos.
/// Live oracle:
///   `LabelBinarizer(neg_label=-1,pos_label=2).fit([0,1,2]).inverse_transform(
///       [[2,-1,-1],[-1,-1,2]])` -> [0, 2]
#[test]
fn req8_neg_pos_inverse_multiclass() {
    let fitted = LabelBinarizer::new()
        .with_neg_label(-1)
        .with_pos_label(2)
        .fit(&array![0_usize, 1, 2], &())
        .unwrap();
    let mat: Array2<f64> = array![[2.0, -1.0, -1.0], [-1.0, -1.0, 2.0]];
    let recovered = fitted.inverse_transform(&mat).unwrap();
    assert_eq!(recovered, array![0_usize, 2]);
}

/// REQ-1/2/3 preserved: defaults (neg=0,pos=1) reproduce the canonical 0/1.
/// Live oracle:
///   `LabelBinarizer().fit_transform([0,1,2,1]).tolist()`
///     -> `[[1,0,0],[0,1,0],[0,0,1],[0,1,0]]`
#[test]
fn req8_defaults_preserve_zero_one() {
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

// ===========================================================================
// REQ-10: `label_binarize` free function (sklearn `_label.py:430`).
//
// Encodes `y` against an EXPLICIT `classes` list (not fit-discovered). Output
// columns follow the GIVEN `classes` order via sklearn's "preserve label
// ordering" reorder (`_label.py:587-590`); collapse by `k = classes.len()`
// (k==1 all-neg col `:532-538`; k==2 single col = `Y[:,-1]` after reorder
// `:596`; k>2 one-hot in given order); unseen labels → all-neg row
// (`:556-559`); `neg>=pos` → ValueError verbatim (`:499-504`).
//
// Every expected value below is a LIVE sklearn 1.5.2 oracle (run from /tmp):
//   from sklearn.preprocessing import label_binarize
// ===========================================================================

/// REQ-10: basic multiclass.
/// Live oracle:
///   `label_binarize([0,2,1], classes=[0,1,2]).tolist()`
///     -> `[[1,0,0],[0,0,1],[0,1,0]]`
#[test]
fn req10_label_binarize_basic_multiclass() {
    let y: Array1<usize> = array![0, 2, 1];
    let got = label_binarize(&y, &[0, 1, 2], 0, 1).unwrap();
    let expected: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    assert_eq!(got, expected);
}

/// REQ-10: neg_label/pos_label.
/// Live oracle:
///   `label_binarize([0,2], classes=[0,1,2], neg_label=-1, pos_label=2).tolist()`
///     -> `[[2,-1,-1],[-1,-1,2]]`
#[test]
fn req10_label_binarize_neg_pos() {
    let y: Array1<usize> = array![0, 2];
    let got = label_binarize(&y, &[0, 1, 2], -1, 2).unwrap();
    let expected: Array2<f64> = array![[2.0, -1.0, -1.0], [-1.0, -1.0, 2.0]];
    assert_eq!(got, expected);
}

/// REQ-10: binary (k==2) single column.
/// Live oracle:
///   `label_binarize([0,1,0], classes=[0,1]).tolist()` -> `[[0],[1],[0]]`
#[test]
fn req10_label_binarize_binary_single_column() {
    let y: Array1<usize> = array![0, 1, 0];
    let got = label_binarize(&y, &[0, 1], 0, 1).unwrap();
    let expected: Array2<f64> = array![[0.0], [1.0], [0.0]];
    assert_eq!(got.shape(), &[3, 1]);
    assert_eq!(got, expected);
}

/// REQ-10 (HEADLINE): the class-ordering case. With `classes=[2,0,1]` the
/// output COLUMNS follow the GIVEN order, not a sorted order — sklearn reorders
/// columns back to the given `classes` order (`_label.py:587-590`).
/// Live oracle:
///   `label_binarize([0,2,1], classes=[2,0,1]).tolist()`
///     -> `[[0,1,0],[1,0,0],[0,0,1]]`
///
/// (Contrast: with sorted `classes=[0,1,2]` the same `y` gives
/// `[[1,0,0],[0,0,1],[0,1,0]]` — DIFFERENT, confirming columns track the GIVEN
/// order, where column `j` corresponds to `classes[j]`.)
#[test]
fn req10_label_binarize_given_class_ordering() {
    let y: Array1<usize> = array![0, 2, 1];
    let got = label_binarize(&y, &[2, 0, 1], 0, 1).unwrap();
    let expected: Array2<f64> = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    assert_eq!(got, expected);
}

/// REQ-10: a value not in `classes` leaves its row all-`neg_label`
/// (sklearn `np.isin` silent ignore, `_label.py:556-559`).
/// Live oracle:
///   `label_binarize([0,3,1], classes=[0,1,2]).tolist()`
///     -> `[[1,0,0],[0,0,0],[0,1,0]]`  (label 3 ignored, row 1 all-zero)
#[test]
fn req10_label_binarize_unseen_label_all_neg_row() {
    let y: Array1<usize> = array![0, 3, 1];
    let got = label_binarize(&y, &[0, 1, 2], 0, 1).unwrap();
    let expected: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    assert_eq!(got, expected);
}

/// REQ-10: single-class (k==1) → all-`neg_label` single column.
/// Live oracle:
///   `label_binarize([5,5,5], classes=[5]).tolist()` -> `[[0],[0],[0]]`
///   `label_binarize([5,5,5], classes=[5], neg_label=-1, pos_label=2).tolist()`
///     -> `[[-1],[-1],[-1]]`
#[test]
fn req10_label_binarize_single_class() {
    let y: Array1<usize> = array![5, 5, 5];
    let got = label_binarize(&y, &[5], 0, 1).unwrap();
    assert_eq!(got, array![[0.0], [0.0], [0.0]]);

    let got_np = label_binarize(&y, &[5], -1, 2).unwrap();
    assert_eq!(got_np, array![[-1.0], [-1.0], [-1.0]]);
}

/// REQ-10: `neg_label >= pos_label` rejected, verbatim message
/// (sklearn `_label.py:499-504`).
/// Live oracle:
///   `label_binarize([0,1], classes=[0,1], neg_label=2, pos_label=1)`
///     -> ValueError: "neg_label=2 must be strictly less than pos_label=1."
#[test]
fn req10_label_binarize_neg_ge_pos_rejected() {
    let y: Array1<usize> = array![0, 1];
    let err = label_binarize(&y, &[0, 1], 2, 1).unwrap_err();
    assert!(matches!(
        &err,
        FerroError::InvalidParameter { name, reason }
            if name == "neg_label"
                && reason == "neg_label=2 must be strictly less than pos_label=1."
    ));
}

/// REQ-10: the free fn equals the SHIPPED estimator's fit+transform when
/// `classes` is the fit-discovered (sorted-unique) set. Both sides are
/// independently checked against the same live sklearn oracle.
/// Live oracle:
///   `label_binarize([0,1,2,1], classes=[0,1,2]).tolist()`
///     == `LabelBinarizer().fit([0,1,2]).transform([0,1,2,1]).tolist()`
///     -> `[[1,0,0],[0,1,0],[0,0,1],[0,1,0]]`
///   binary: `label_binarize([0,1,0,1], classes=[0,1])`
///     == `LabelBinarizer().fit([0,1]).transform([0,1,0,1])` -> `[[0],[1],[0],[1]]`
#[test]
fn req10_label_binarize_equals_estimator() {
    // multiclass
    let y: Array1<usize> = array![0, 1, 2, 1];
    let classes = [0_usize, 1, 2];
    let free = label_binarize(&y, &classes, 0, 1).unwrap();
    let fitted = LabelBinarizer::new()
        .fit(&Array1::from(classes.to_vec()), &())
        .unwrap();
    let est = fitted.transform(&y).unwrap();
    let oracle: Array2<f64> = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ];
    assert_eq!(free, oracle);
    assert_eq!(est, oracle);
    assert_eq!(free, est);

    // binary
    let yb: Array1<usize> = array![0, 1, 0, 1];
    let cb = [0_usize, 1];
    let free_b = label_binarize(&yb, &cb, 0, 1).unwrap();
    let fitted_b = LabelBinarizer::new()
        .fit(&Array1::from(cb.to_vec()), &())
        .unwrap();
    let est_b = fitted_b.transform(&yb).unwrap();
    let oracle_b: Array2<f64> = array![[0.0], [1.0], [0.0], [1.0]];
    assert_eq!(free_b, oracle_b);
    assert_eq!(est_b, oracle_b);
    assert_eq!(free_b, est_b);

    // neg/pos estimator equivalence
    let yn: Array1<usize> = array![0, 2];
    let cn = [0_usize, 1, 2];
    let free_n = label_binarize(&yn, &cn, -1, 2).unwrap();
    let fitted_n = LabelBinarizer::new()
        .with_neg_label(-1)
        .with_pos_label(2)
        .fit(&Array1::from(cn.to_vec()), &())
        .unwrap();
    let est_n = fitted_n.transform(&yn).unwrap();
    let oracle_n: Array2<f64> = array![[2.0, -1.0, -1.0], [-1.0, -1.0, 2.0]];
    assert_eq!(free_n, oracle_n);
    assert_eq!(est_n, oracle_n);
    assert_eq!(free_n, est_n);
}

// ===========================================================================
// DIV-REQ10 (the type_of_target collapse): k==2-sized `classes` but a
// MULTICLASS `y` (plain integer labels).
//
// sklearn decides the binary single-column collapse from `type_of_target(y)`,
// NOT from `len(classes)`:
//   `sklearn/preprocessing/_label.py:519`   y_type = type_of_target(y)
//   `sklearn/preprocessing/_label.py:531`   if y_type == "binary":
//   `sklearn/preprocessing/_label.py:592-596`  if y_type == "binary": Y = Y[:,-1]...
// For `y=[0,1,2]` we have `type_of_target([0,1,2]) == "multiclass"` (3 distinct
// values), so the binary branch is NOT entered. sklearn builds a
// `(n_samples, n_classes) = (3, 2)` indicator via searchsorted
// (`:552-563`); the label `2` is not in `classes=[0,1]` so `np.isin`
// (`:556`) drops it, leaving row 2 all-`neg_label`.
//
// Live oracle (sklearn 1.5.2, from /tmp):
//   from sklearn.preprocessing import label_binarize
//   label_binarize([0,1,2], classes=[0,1]).tolist()  -> [[1, 0], [0, 1], [0, 0]]
//   (and type_of_target([0,1,2]) == "multiclass")
//
// ferrolearn collapses on `k = classes.len() == 2`
// (`ferrolearn-preprocess/src/label_binarizer.rs:445-457`) and returns a
// (3, 1) single column [[0],[1],[0]] — WRONG shape AND wrong values, silently.
// This is plain integer labels (the matched scope), distinct from the
// documented REQ-11 arbitrary-type/multilabel deferral (#1245).
//
// Tracking: #2233
// ===========================================================================
#[test]
#[ignore = "divergence: label_binarize collapses by len(classes) not type_of_target(y); tracking #2233"]
fn divergence_label_binarize_binary_classes_multiclass_y() {
    let y: Array1<usize> = array![0, 1, 2];
    // sklearn-oracle: 2 columns, row for unseen label 2 all-zero.
    let expected: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
    let got = label_binarize(&y, &[0, 1], 0, 1).unwrap();
    assert_eq!(got.shape(), &[3, 2]);
    assert_eq!(got, expected);
}
