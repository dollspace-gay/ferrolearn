//! Divergence tests: `ferrolearn-preprocess` `LabelEncoder` vs scikit-learn 1.5.2
//! `sklearn/preprocessing/_label.py::class LabelEncoder`.
//!
//! EVERY expected value below is grounded in a LIVE sklearn 1.5.2 oracle call
//! (run from /tmp) or a sklearn `file:line`. None are copied from the ferrolearn
//! side (R-CHAR-3).
//!
//! Oracle session (sklearn 1.5.2):
//! ```text
//! >>> LabelEncoder().fit([]).classes_.tolist(), .shape  -> [], (0,)   # SUCCEEDS
//! >>> LabelEncoder().fit(['cat','dog','cat','bird']).classes_         -> ['bird','cat','dog']
//! >>> .transform(['cat','dog','cat','bird'])                          -> [1,2,1,0]
//! >>> .inverse_transform([1,2,1,0])                                   -> ['cat','dog','cat','bird']
//! >>> LabelEncoder().fit_transform(['cat','dog','cat','bird'])        -> [1,2,1,0]
//! >>> le.transform([])                                                -> [] shape (0,)
//! >>> le.inverse_transform([])                                        -> [] shape (0,)
//! >>> LabelEncoder().fit(['B','a','A','b','10','2']).classes_         -> ['10','2','A','B','a','b']
//! >>> le.transform(['c'])  -> ValueError: y contains previously unseen labels
//! ```

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::LabelEncoder;
use ndarray::{Array1, array};

fn str_arr(v: &[&str]) -> Array1<String> {
    Array1::from_vec(v.iter().map(std::string::ToString::to_string).collect())
}

// ===========================================================================
// TASK 1 — DIVERGENCE PIN (must FAIL against current ferrolearn).
//
// sklearn `_label.py:84-99` `fit`: `self.classes_ = _unique(y)`. There is NO
// emptiness guard; `_unique([])` returns an empty array, so `fit([])` SUCCEEDS
// with `classes_ == []` (live oracle: `[], shape (0,)`).
//
// ferrolearn `label_encoder.rs:113-120` returns `Err(InsufficientSamples)` on
// empty input — a behavioral divergence: sklearn returns a fitted estimator
// with `n_classes == 0`, ferrolearn raises.
//
// (The in-module `test_empty_input_error` at `label_encoder.rs:272` pins the
// WRONG behavior; flagged for R-HONEST-4 fixer correction — not edited here.)
//
// Tracking: see filed -l blocker issue.
// ===========================================================================
#[test]
fn divergence_empty_fit_succeeds() {
    // sklearn: LabelEncoder().fit([]) SUCCEEDS, classes_ == [], shape (0,).
    let enc = LabelEncoder::new();
    let empty: Array1<String> = Array1::from_vec(vec![]);
    let fitted = enc
        .fit(&empty, &())
        .expect("sklearn _label.py:84-99: fit([]) succeeds with empty classes_");
    // sklearn classes_.shape == (0,)  => n_classes == 0
    assert_eq!(
        fitted.n_classes(),
        0,
        "sklearn empty-fit yields classes_ of length 0"
    );
    assert!(
        fitted.classes().is_empty(),
        "sklearn empty-fit yields empty classes_"
    );
}

// ===========================================================================
// TASK 2 — GREEN GUARDS for the shipped string path (oracle-grounded).
// These should PASS, replacing the hand-written in-module asserts (R-CHAR-3).
// ===========================================================================

/// Oracle: `LabelEncoder().fit(['cat','dog','cat','bird']).classes_.tolist()`
/// -> `['bird','cat','dog']` (sklearn `_label.py:98`, `_encode._unique`).
#[test]
fn green_fit_classes_sorted() {
    let enc = LabelEncoder::new();
    let labels = str_arr(&["cat", "dog", "cat", "bird"]);
    let fitted = enc.fit(&labels, &()).unwrap();
    assert_eq!(fitted.classes(), &["bird", "cat", "dog"]);
    assert_eq!(fitted.n_classes(), 3);
}

/// Oracle: `.transform(['cat','dog','cat','bird'])` -> `[1,2,1,0]`
/// (sklearn `_label.py:137`, `_encode`).
#[test]
fn green_transform() {
    let enc = LabelEncoder::new();
    let labels = str_arr(&["cat", "dog", "cat", "bird"]);
    let fitted = enc.fit(&labels, &()).unwrap();
    let encoded = fitted.transform(&labels).unwrap();
    assert_eq!(encoded.to_vec(), vec![1usize, 2, 1, 0]);
}

/// Oracle: `.inverse_transform([1,2,1,0])` -> `['cat','dog','cat','bird']`
/// (sklearn `_label.py:162`).
#[test]
fn green_inverse_transform_roundtrip() {
    let enc = LabelEncoder::new();
    let labels = str_arr(&["cat", "dog", "cat", "bird"]);
    let fitted = enc.fit(&labels, &()).unwrap();
    let codes = array![1usize, 2, 1, 0];
    let recovered = fitted.inverse_transform(&codes).unwrap();
    assert_eq!(
        recovered.to_vec(),
        vec![
            "cat".to_string(),
            "dog".to_string(),
            "cat".to_string(),
            "bird".to_string()
        ]
    );
}

/// Oracle: `LabelEncoder().fit_transform(['cat','dog','cat','bird'])`
/// -> `[1,2,1,0]` (sklearn `_label.py:101-116`), equal to fit-then-transform.
#[test]
fn green_fit_transform_equals_fit_then_transform() {
    let enc = LabelEncoder::new();
    let labels = str_arr(&["cat", "dog", "cat", "bird"]);
    let via_ft = enc.fit_transform(&labels).unwrap();
    assert_eq!(via_ft.to_vec(), vec![1usize, 2, 1, 0]); // oracle
    let fitted = enc.fit(&labels, &()).unwrap();
    let via_sep = fitted.transform(&labels).unwrap();
    assert_eq!(via_ft, via_sep);
}

/// Oracle: `.transform([])` -> `[]` shape `(0,)`
/// (sklearn `_label.py:134-135`: empty input returns `np.array([])`).
#[test]
fn green_empty_transform_returns_empty() {
    let enc = LabelEncoder::new();
    let fitted = enc.fit(&str_arr(&["a", "b"]), &()).unwrap();
    let empty: Array1<String> = Array1::from_vec(vec![]);
    let out = fitted
        .transform(&empty)
        .expect("sklearn _label.py:134-135: empty transform is empty array");
    assert!(out.is_empty());
}

/// Oracle: `.inverse_transform([])` -> `[]` shape `(0,)`
/// (sklearn `_label.py:155-156`: empty input returns `np.array([])`).
#[test]
fn green_empty_inverse_transform_returns_empty() {
    let enc = LabelEncoder::new();
    let fitted = enc.fit(&str_arr(&["a", "b"]), &()).unwrap();
    let empty: Array1<usize> = Array1::from_vec(vec![]);
    let out = fitted
        .inverse_transform(&empty)
        .expect("sklearn _label.py:155-156: empty inverse_transform is empty array");
    assert!(out.is_empty());
}

// ===========================================================================
// TASK 3 — classes_ sort order on mixed-ASCII strings.
// Oracle: `LabelEncoder().fit(['B','a','A','b','10','2']).classes_.tolist()`
// -> `['10','2','A','B','a','b']` (codepoint order; matches Rust String sort).
// ===========================================================================
#[test]
fn green_sort_order_mixed_ascii_matches_numpy() {
    let enc = LabelEncoder::new();
    let labels = str_arr(&["B", "a", "A", "b", "10", "2"]);
    let fitted = enc.fit(&labels, &()).unwrap();
    assert_eq!(fitted.classes(), &["10", "2", "A", "B", "a", "b"]);
}

// ===========================================================================
// TASK 4 — Unseen-label rejection (behavioral parity, NOT a divergence).
// sklearn raises ValueError "y contains previously unseen labels"
// (`_encode._encode` -> `_label.py:137`). ferrolearn rejects with
// FerroError::InvalidParameter — the correct Rust analog. Only assert BOTH
// reject; the message-string difference is cosmetic and not pinned.
// ===========================================================================
#[test]
fn green_unseen_label_rejected() {
    let enc = LabelEncoder::new();
    let fitted = enc.fit(&str_arr(&["a", "b"]), &()).unwrap();
    let unseen = str_arr(&["c"]);
    assert!(
        fitted.transform(&unseen).is_err(),
        "sklearn raises ValueError on previously unseen labels; ferrolearn must also reject"
    );
}

// ===========================================================================
// TASK 5 — POST-EMPTY-FIT behaviors (re-audit of blocker #1134 fix).
//
// After `fit([])` succeeds with empty classes_, sklearn's transform/inverse
// behave as follows (live sklearn 1.5.2 oracle, object-dtype empty fit so the
// path is the canonical _encode one, not the float-coercion edge):
//
//   le = LabelEncoder().fit(np.array([], dtype=object))   # classes_ == [] (obj)
//   le.transform([])         -> array([])         # empty -> empty, NO error
//   le.transform(['a'])      -> ValueError: y contains previously unseen labels
//   le.inverse_transform([]) -> array([])         # empty -> empty, NO error
//   le.inverse_transform([0])-> ValueError: y contains previously unseen labels
//
// These are GREEN GUARDS: ferrolearn must match (empty -> Ok(empty); unseen /
// out-of-range -> Err). The message-string difference is NOT pinned.
// ===========================================================================

/// Item 1. Oracle: empty-fit then `transform([])` -> `[]` (no error).
/// (sklearn `_label.py:134-135` empty-input short-circuit, classes_ empty.)
#[test]
fn green_empty_fit_then_empty_transform_ok() {
    let enc = LabelEncoder::new();
    let empty: Array1<String> = Array1::from_vec(vec![]);
    let fitted = enc.fit(&empty, &()).expect("fit([]) succeeds");
    let out = fitted
        .transform(&Array1::from_vec(vec![]))
        .expect("sklearn: empty transform after empty fit is Ok(empty)");
    assert!(out.is_empty());
}

/// Item 2. Oracle: empty-fit then `transform(['a'])` -> ValueError (unseen
/// label, classes_ empty). ferrolearn: empty map -> get('a') None -> Err.
/// Both REJECT; message not pinned. (sklearn `_label.py:137`, `_encode`.)
#[test]
fn green_empty_fit_then_transform_unseen_rejected() {
    let enc = LabelEncoder::new();
    let empty: Array1<String> = Array1::from_vec(vec![]);
    let fitted = enc.fit(&empty, &()).expect("fit([]) succeeds");
    assert!(
        fitted.transform(&str_arr(&["a"])).is_err(),
        "sklearn raises on unseen label after empty fit; ferrolearn must also reject"
    );
}

/// Item 3a. Oracle: empty-fit then `inverse_transform([])` -> `[]` (no error).
/// (sklearn `_label.py:155-156` empty-input short-circuit.)
#[test]
fn green_empty_fit_then_empty_inverse_ok() {
    let enc = LabelEncoder::new();
    let empty: Array1<String> = Array1::from_vec(vec![]);
    let fitted = enc.fit(&empty, &()).expect("fit([]) succeeds");
    let out = fitted
        .inverse_transform(&Array1::from_vec(vec![]))
        .expect("sklearn: empty inverse_transform after empty fit is Ok(empty)");
    assert!(out.is_empty());
}

/// Item 3b. Oracle: empty-fit then `inverse_transform([0])` -> ValueError
/// (out-of-range, len(classes_)==0). ferrolearn: idx 0 >= n_classes 0 -> Err.
/// Both REJECT; message not pinned. (sklearn `_label.py:158-160`.)
#[test]
fn green_empty_fit_then_inverse_oob_rejected() {
    let enc = LabelEncoder::new();
    let empty: Array1<String> = Array1::from_vec(vec![]);
    let fitted = enc.fit(&empty, &()).expect("fit([]) succeeds");
    assert!(
        fitted.inverse_transform(&array![0usize]).is_err(),
        "sklearn raises out-of-range on inverse_transform([0]) after empty fit; ferrolearn must reject"
    );
}
