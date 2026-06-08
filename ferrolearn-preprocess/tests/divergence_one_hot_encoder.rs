//! ACToR oracle-grounded tests for `ferrolearn-preprocess`'s
//! `OneHotEncoder` / `FittedOneHotEncoder`
//! (`ferrolearn-preprocess/src/one_hot_encoder.rs`) against scikit-learn 1.5.2
//! `sklearn/preprocessing/_encoders.py` `class OneHotEncoder(_BaseEncoder)`
//! (`:458`).
//!
//! ferrolearn now ships a numeric (`Array2<F>`-input) DENSE encoder whose
//! `fit` learns `categories_[j] = _unique(Xi)` (the per-column **sorted-unique
//! set**, `_BaseEncoder._fit:99`) and whose `transform` is one-hot by **category
//! MEMBERSHIP** (the value's index within `categories_[j]`), with the per-feature
//! one-hot blocks concatenated left-to-right. This matches sklearn's
//! `OneHotEncoder(sparse_output=False)` dense output for ANY finite numeric
//! columns — contiguous OR non-contiguous (the `[2,5,9]` headline) — not just the
//! old `max+1` contiguous regime.
//!
//! R-CHAR-3: EVERY expected value below is the output of a LIVE sklearn 1.5.2
//! call, run from `/tmp`, quoted inline — NEVER literal-copied from the
//! ferrolearn side.
//!
//! Live oracle (sklearn 1.5.2 == `python3 -c "import sklearn; sklearn.__version__"`):
//! ```text
//! # categories_ for the multi-feature non-contiguous fixture:
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   e=OneHotEncoder(sparse_output=False).fit([[2.,0.],[5.,1.],[9.,0.],[5.,1.]]); \
//!   print([c.tolist() for c in e.categories_])"
//!   -> [[2.0, 5.0, 9.0], [0.0, 1.0]]
//!
//! # transform of the same fixture (5 cols = 3 + 2):
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   print(OneHotEncoder(sparse_output=False).fit_transform([[2.,0.],[5.,1.],[9.,0.],[5.,1.]]).tolist())"
//!   -> [[1.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0,1.0],[0.0,0.0,1.0,1.0,0.0],[0.0,1.0,0.0,0.0,1.0]]
//!
//! # non-contiguous single column {2,5,9} (THE HEADLINE: 3 cols, NOT max+1==10):
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   e=OneHotEncoder(sparse_output=False).fit([[2.],[5.],[9.]]); \
//!   print([c.tolist() for c in e.categories_], e.fit_transform([[2.],[5.],[9.]]).tolist())"
//!   -> [[2.0, 5.0, 9.0]] [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
//!
//! # unordered + duplicated input -> sorted-unique categories_:
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   e=OneHotEncoder(sparse_output=False).fit([[9.],[2.],[5.],[2.],[9.]]); \
//!   print([c.tolist() for c in e.categories_])"
//!   -> [[2.0, 5.0, 9.0]]
//!
//! # unknown value at transform (default handle_unknown='error') -> ValueError:
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   e=OneHotEncoder(sparse_output=False).fit([[2.],[5.]]); e.transform([[7.]])"
//!   -> ValueError: Found unknown categories [np.float64(7.0)] in column 0 during transform
//!
//! # 3-column fixture with category counts 2,3,2 -> 7 output columns (offsets):
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   X=[[0.,1.,8.],[1.,3.,9.],[0.,5.,8.],[1.,1.,9.]]; \
//!   e=OneHotEncoder(sparse_output=False).fit(X); \
//!   print([c.tolist() for c in e.categories_]); print(e.transform(X).tolist())"
//!   -> [[0.0, 1.0], [1.0, 3.0, 5.0], [8.0, 9.0]]
//!      [[1.0,0.0,1.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,1.0,0.0,0.0,1.0],
//!       [1.0,0.0,0.0,0.0,1.0,1.0,0.0],[0.0,1.0,1.0,0.0,0.0,0.0,1.0]]
//!
//! # 0-row fit (minimum 1 sample required) raises:
//! python3 -c "import numpy as np; from sklearn.preprocessing import OneHotEncoder; \
//!   OneHotEncoder().fit(np.empty((0,1)))"
//!   -> ValueError: Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.
//! ```

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::OneHotEncoder;
use ndarray::{Array2, array};

// ---------------------------------------------------------------------------
// REQ-3 — categories_ = sorted-unique set (per column, any finite numeric).
// ---------------------------------------------------------------------------

/// `categories_` of the multi-feature fixture `[[2,0],[5,1],[9,0],[5,1]]`
/// == live sklearn `.categories_` == `[[2,5,9],[0,1]]` (sorted-unique per
/// column, including the NON-contiguous `{2,5,9}` and the duplicate-`5`/`0`/`1`
/// columns reduced to the unique set).
#[test]
fn categories_sorted_unique_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp):
    //   [[2.0, 5.0, 9.0], [0.0, 1.0]]
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    assert_eq!(
        fitted.categories(),
        &[vec![2.0, 5.0, 9.0], vec![0.0, 1.0]],
        "categories_ vs sklearn oracle"
    );
}

/// Unordered + duplicated input column `[9,2,5,2,9]` learns the SORTED-unique
/// set `[2,5,9]`, matching live sklearn `np.unique` semantics (`_fit:99`).
#[test]
fn categories_unordered_duplicates_sorted_unique_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp): [[2.0, 5.0, 9.0]]
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[9.0_f64], [2.0], [5.0], [2.0], [9.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    assert_eq!(
        fitted.categories(),
        &[vec![2.0, 5.0, 9.0]],
        "sorted-unique categories_ vs sklearn oracle"
    );
}

// ---------------------------------------------------------------------------
// REQ-1/REQ-3 — transform matrix == sklearn EXACTLY (column layout + 1/0).
// ---------------------------------------------------------------------------

/// transform of `[[2,0],[5,1],[9,0],[5,1]]` == live sklearn (5 cols = 3 + 2),
/// the per-feature blocks concatenated left-to-right by membership.
#[test]
fn transform_multifeature_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp):
    let sklearn_expected: Array2<f64> = array![
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
    ];
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let out = enc.fit(&x, &()).unwrap().transform(&x).unwrap();
    assert_eq!(out.shape(), &[4, 5], "shape vs sklearn oracle");
    assert_eq!(out, sklearn_expected, "transform vs sklearn oracle");
}

/// THE NON-CONTIGUOUS HEADLINE: single column `{2,5,9}` -> 3 columns (NOT the
/// old `max+1 == 10`). transform == live sklearn `[[1,0,0],[0,1,0],[0,0,1]]`.
#[test]
fn transform_non_contiguous_single_column_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp):
    //   categories_ == [[2.0, 5.0, 9.0]]; transform == 3x3 identity.
    let sklearn_expected: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64], [5.0], [9.0]];
    let fitted = enc.fit(&x, &()).unwrap();

    // The headline: 3 output columns, NOT 10.
    assert_eq!(
        fitted.n_output_features(),
        3,
        "non-contiguous {{2,5,9}} -> 3 output cols (sklearn), not max+1==10"
    );
    let out = fitted.transform(&x).unwrap();
    assert_eq!(out.shape(), &[3, 3], "shape vs sklearn oracle");
    assert_eq!(out, sklearn_expected, "transform vs sklearn oracle");
}

/// `fit_transform` equals `fit` then `transform`, both equal the live sklearn
/// oracle for the multi-feature fixture (the combined path must not diverge).
#[test]
fn fit_transform_equals_split_path_vs_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp), same matrix as the multifeature test:
    let sklearn_expected: Array2<f64> = array![
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
    ];
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];

    let via_fit_transform: Array2<f64> = enc.fit_transform(&x).unwrap();
    let via_split = enc.fit(&x, &()).unwrap().transform(&x).unwrap();

    assert_eq!(via_fit_transform, via_split, "fit_transform vs split path");
    assert_eq!(
        via_fit_transform, sklearn_expected,
        "fit_transform vs sklearn oracle"
    );
}

// ---------------------------------------------------------------------------
// REQ-3 — output column count == Σ len(categories_[j]); offset layout.
// ---------------------------------------------------------------------------

/// A 3-column fixture with per-feature category counts 2, 3, 2 -> 7 output
/// columns, the blocks correctly offset. categories_ AND transform == live
/// sklearn oracle.
#[test]
fn multifeature_offsets_2_3_2_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp):
    //   categories_ == [[0,1],[1,3,5],[8,9]] ; n_output == 7
    let sklearn_expected: Array2<f64> = array![
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let enc = OneHotEncoder::<f64>::new();
    let x = array![
        [0.0_f64, 1.0, 8.0],
        [1.0, 3.0, 9.0],
        [0.0, 5.0, 8.0],
        [1.0, 1.0, 9.0],
    ];
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(
        fitted.categories(),
        &[vec![0.0, 1.0], vec![1.0, 3.0, 5.0], vec![8.0, 9.0]],
        "per-feature sorted-unique categories_ vs sklearn oracle"
    );
    // Output column count == Σ len(categories_[j]) == 2 + 3 + 2 == 7.
    assert_eq!(
        fitted.n_output_features(),
        7,
        "n_output == sum(len(categories_[j])) vs sklearn"
    );
    let out = fitted.transform(&x).unwrap();
    assert_eq!(out.shape(), &[4, 7], "shape vs sklearn oracle");
    assert_eq!(
        out, sklearn_expected,
        "offset block layout vs sklearn oracle"
    );
}

// ---------------------------------------------------------------------------
// REQ-4 (membership error) / shape / 0-row error contracts.
// ---------------------------------------------------------------------------

/// An UNKNOWN value at transform (a value NOT in `categories_[j]`) errors under
/// the default `handle_unknown='error'`, matching live sklearn's `ValueError`.
/// Live sklearn: fit `[[2],[5]]` then transform `[[7]]` -> "Found unknown
/// categories [7.0] in column 0 during transform".
#[test]
fn unknown_value_at_transform_errors_vs_sklearn_oracle() {
    let enc = OneHotEncoder::<f64>::new();
    let x_train = array![[2.0_f64], [5.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    let x_bad = array![[7.0_f64]];
    let err = fitted.transform(&x_bad);
    assert!(
        err.is_err(),
        "unknown category 7.0 must error, matching sklearn handle_unknown='error'"
    );
    // The message mirrors sklearn's "Found unknown categories ... during transform".
    let msg = format!("{:?}", err.unwrap_err());
    assert!(
        msg.contains("Found unknown categories") && msg.contains("during transform"),
        "error message should mirror sklearn's 'Found unknown categories ... during transform', got: {msg}"
    );
}

/// An in-range-but-UNSEEN non-contiguous value errors — the membership contract
/// (the old `max+1` path would have silently accepted `4.0 < 10`). After fitting
/// `{2,5,9}`, transforming `4.0` is unknown. Live sklearn: fit `[[2],[5],[9]]`
/// then transform `[[4]]` -> ValueError "Found unknown categories [4.0] ...".
#[test]
fn in_range_unseen_value_errors_via_membership_vs_sklearn_oracle() {
    let enc = OneHotEncoder::<f64>::new();
    let x_train = array![[2.0_f64], [5.0], [9.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    let x_bad = array![[4.0_f64]];
    assert!(
        fitted.transform(&x_bad).is_err(),
        "4.0 is unseen (not in categories_={{2,5,9}}); membership must reject it, matching sklearn"
    );
}

/// ncols mismatch at transform -> Err (shape guard).
#[test]
fn ncols_mismatch_errors() {
    let enc = OneHotEncoder::<f64>::new();
    let x_train = array![[0.0_f64, 1.0], [1.0, 0.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    let x_bad = array![[0.0_f64]]; // 1 col, expected 2
    assert!(
        fitted.transform(&x_bad).is_err(),
        "ncols mismatch must error (ShapeMismatch)"
    );
}

/// 0-row fit errors, matching live sklearn's minimum-1-sample requirement.
/// Live sklearn: `OneHotEncoder().fit(np.empty((0,1)))` -> ValueError.
#[test]
fn zero_row_fit_errors_vs_sklearn_oracle() {
    let enc = OneHotEncoder::<f64>::new();
    let x_empty: Array2<f64> = Array2::zeros((0, 1));
    assert!(
        enc.fit(&x_empty, &()).is_err(),
        "0-row fit must error, matching sklearn's minimum-1-sample requirement"
    );
}
