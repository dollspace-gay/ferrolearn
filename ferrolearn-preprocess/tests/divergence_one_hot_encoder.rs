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
use ferrolearn_preprocess::{OneHotDrop, OneHotEncoder, OneHotHandleUnknown};
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

// ---------------------------------------------------------------------------
// REQ-4 — handle_unknown {'error','ignore'} == sklearn EXACTLY.
//   (`sklearn/preprocessing/_encoders.py` __init__ handle_unknown='error'
//    default `:750`; _transform ignore -> all-zero block `:206-240`.)
//
// All expected values below are LIVE sklearn 1.5.2 calls (sparse_output=False),
// quoted inline (R-CHAR-3) — NEVER copied from ferrolearn:
//
//   # ignore, multi-feature all-zero block:
//   python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//     e=OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit([[2.,0.],[5.,1.],[9.,0.]]); \
//     print(e.transform([[100.,0.],[5.,99.]]).tolist())"
//     -> [[0.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0,0.0]]
//
//   # ignore, fully-unknown row -> all zeros:
//   python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//     e=OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit([[2.,0.],[5.,1.],[9.,0.]]); \
//     print(e.transform([[100.,99.]]).tolist())"
//     -> [[0.0,0.0,0.0,0.0,0.0]]
//
//   # ignore, all-known row -> normal one-hot:
//   python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//     e=OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit([[2.,0.],[5.,1.],[9.,0.]]); \
//     print(e.transform([[9.,1.]]).tolist())"
//     -> [[0.0,0.0,1.0,0.0,1.0]]
//
//   # ignore, +inf still rejected (invalid input, not 'unknown category'):
//   python3 -c "import numpy as np; from sklearn.preprocessing import OneHotEncoder; \
//     OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit([[2.,0.],[5.,1.],[9.,0.]]).transform([[np.inf,0.]])"
//     -> ValueError: Input contains infinity or a value too large for dtype('float64').
//
//   # ignore, NaN with NO nan category -> all-zero block (NaN is 'unknown'):
//   python3 -c "import numpy as np; from sklearn.preprocessing import OneHotEncoder; \
//     e=OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit([[2.,0.],[5.,1.],[9.,0.]]); \
//     print(e.transform([[np.nan,0.]]).tolist())"
//     -> [[0.0,0.0,0.0,1.0,0.0]]
//
//   # ignore, NaN WITH a learned nan category -> one-hots it:
//   python3 -c "import numpy as np; from sklearn.preprocessing import OneHotEncoder; \
//     e=OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit([[2.,0.],[np.nan,1.],[9.,0.]]); \
//     print([c.tolist() for c in e.categories_]); print(e.transform([[np.nan,0.]]).tolist())"
//     -> [[2.0, 9.0, nan], [0.0, 1.0]]
//        [[0.0,0.0,1.0,1.0,0.0]]
// ---------------------------------------------------------------------------

/// `handle_unknown='ignore'`, multi-feature: an unknown value in one feature
/// yields an ALL-ZERO block for THAT feature, while a known value in the other
/// feature still one-hots. Live sklearn:
///   transform([[100,0],[5,99]]) -> [[0,0,0,1,0],[0,1,0,0,0]].
/// Row 0: col0 unknown 100 -> [0,0,0]; col1 known 0 -> [1,0].
/// Row 1: col0 known 5 -> [0,1,0]; col1 unknown 99 -> [0,0].
#[test]
fn ignore_multifeature_all_zero_block_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, sparse_output=False, handle_unknown='ignore'):
    let sklearn_expected: Array2<f64> = array![
        [0.0, 0.0, 0.0, 1.0, 0.0], // 100 unknown -> [0,0,0]; 0 known -> [1,0]
        [0.0, 1.0, 0.0, 0.0, 0.0], // 5 known -> [0,1,0]; 99 unknown -> [0,0]
    ];
    let enc = OneHotEncoder::<f64>::new().with_handle_unknown(OneHotHandleUnknown::Ignore);
    let x_train = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    let out = fitted
        .transform(&array![[100.0_f64, 0.0], [5.0, 99.0]])
        .unwrap();
    assert_eq!(out.shape(), &[2, 5], "shape vs sklearn oracle");
    assert_eq!(
        out, sklearn_expected,
        "handle_unknown='ignore' all-zero-block multi-feature vs sklearn oracle"
    );
}

/// `handle_unknown='ignore'`, a row where EVERY feature is unknown -> all zeros.
/// Live sklearn: transform([[100,99]]) -> [[0,0,0,0,0]].
#[test]
fn ignore_fully_unknown_row_all_zero_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2): [[0.0,0.0,0.0,0.0,0.0]]
    let sklearn_expected: Array2<f64> = array![[0.0, 0.0, 0.0, 0.0, 0.0]];
    let enc = OneHotEncoder::<f64>::new().with_handle_unknown(OneHotHandleUnknown::Ignore);
    let x_train = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    let out = fitted.transform(&array![[100.0_f64, 99.0]]).unwrap();
    assert_eq!(
        out, sklearn_expected,
        "handle_unknown='ignore' fully-unknown row -> all zeros vs sklearn oracle"
    );
}

/// `handle_unknown='ignore'` does NOT change a fully-KNOWN row: normal one-hot.
/// Live sklearn: transform([[9,1]]) -> [[0,0,1,0,1]].
#[test]
fn ignore_known_row_normal_one_hot_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2): [[0.0,0.0,1.0,0.0,1.0]]
    let sklearn_expected: Array2<f64> = array![[0.0, 0.0, 1.0, 0.0, 1.0]];
    let enc = OneHotEncoder::<f64>::new().with_handle_unknown(OneHotHandleUnknown::Ignore);
    let x_train = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    let out = fitted.transform(&array![[9.0_f64, 1.0]]).unwrap();
    assert_eq!(
        out, sklearn_expected,
        "handle_unknown='ignore' known row -> normal one-hot (Ignore doesn't change known values)"
    );
}

/// The DEFAULT (`handle_unknown='error'`, REQ-2 preserved) still rejects an
/// unknown category with a `ValueError`-mirroring error. Live sklearn:
/// fit([[2,0],[5,1],[9,0]]) then transform([[100,0]]) -> ValueError
/// "Found unknown categories ...".
#[test]
fn error_default_unknown_rejected_vs_sklearn_oracle() {
    // Default ctor == handle_unknown='error'; unknown -> Err (REQ-2 preserved).
    let enc = OneHotEncoder::<f64>::new();
    assert_eq!(
        enc.handle_unknown(),
        OneHotHandleUnknown::Error,
        "new() defaults to Error (sklearn handle_unknown='error')"
    );
    let x_train = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    let err = fitted.transform(&array![[100.0_f64, 0.0]]);
    assert!(
        err.is_err(),
        "default Error mode must reject the unknown 100.0 (sklearn ValueError)"
    );
    let msg = format!("{:?}", err.unwrap_err());
    assert!(
        msg.contains("Found unknown categories") && msg.contains("during transform"),
        "error message should mirror sklearn's 'Found unknown categories ... during transform', got: {msg}"
    );
    // Error mode + a fully-known transform -> normal one-hot (unchanged path).
    let ok = fitted.transform(&array![[9.0_f64, 1.0]]).unwrap();
    assert_eq!(
        ok,
        array![[0.0_f64, 0.0, 1.0, 0.0, 1.0]],
        "error-mode known transform == sklearn normal one-hot"
    );
}

/// `with_handle_unknown(Ignore)` + a value that IS known -> normal one-hot
/// (Ignore does not change KNOWN values). Live sklearn (ignore mode, known row):
/// transform([[9,1]]) -> [[0,0,1,0,1]].
#[test]
fn ignore_known_value_unchanged_vs_sklearn_oracle() {
    let enc = OneHotEncoder::<f64>::new().with_handle_unknown(OneHotHandleUnknown::Ignore);
    assert_eq!(
        enc.handle_unknown(),
        OneHotHandleUnknown::Ignore,
        "with_handle_unknown(Ignore) sets the mode"
    );
    let x_train = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    // The fitted encoder also reports the threaded mode.
    assert_eq!(fitted.handle_unknown(), OneHotHandleUnknown::Ignore);
    let out = fitted.transform(&array![[9.0_f64, 1.0]]).unwrap();
    assert_eq!(
        out,
        array![[0.0_f64, 0.0, 1.0, 0.0, 1.0]],
        "Ignore mode on a KNOWN value == sklearn normal one-hot"
    );
}

/// `handle_unknown='ignore'` does NOT relax the +/-inf rejection (#2225): an
/// infinite value is INVALID INPUT (not an "unknown category"), so it still
/// errors even in Ignore mode. Live sklearn (ignore mode):
/// transform([[inf,0]]) -> ValueError "Input contains infinity ...".
#[test]
fn ignore_inf_still_rejected_vs_sklearn_oracle() {
    let enc = OneHotEncoder::<f64>::new().with_handle_unknown(OneHotHandleUnknown::Ignore);
    let x_train = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    let err = fitted.transform(&array![[f64::INFINITY, 0.0]]);
    assert!(
        err.is_err(),
        "+inf is invalid input (not an unknown category) and must error even in Ignore mode, \
         matching sklearn's check_array 'Input contains infinity ...'"
    );
}

/// `handle_unknown='ignore'`, a `NaN` value when there is NO learned NaN
/// category -> the NaN is "unknown" -> all-zero block (NOT an error). Live
/// sklearn: fit([[2,0],[5,1],[9,0]]) then transform([[nan,0]]) ->
/// [[0,0,0,1,0]] (col0 nan-unknown -> [0,0,0]; col1 known 0 -> [1,0]).
#[test]
fn ignore_nan_no_category_all_zero_vs_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2): [[0.0,0.0,0.0,1.0,0.0]]
    let sklearn_expected: Array2<f64> = array![[0.0, 0.0, 0.0, 1.0, 0.0]];
    let enc = OneHotEncoder::<f64>::new().with_handle_unknown(OneHotHandleUnknown::Ignore);
    let x_train = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    let out = fitted.transform(&array![[f64::NAN, 0.0]]).unwrap();
    assert_eq!(
        out, sklearn_expected,
        "Ignore + NaN with no nan-category -> all-zero block vs sklearn oracle"
    );
}

/// `handle_unknown='ignore'`, a `NaN` value WHEN a NaN category WAS learned ->
/// the NaN one-hots its (sorted-last) category, exactly as a known value would.
/// Live sklearn: fit([[2,0],[nan,1],[9,0]]) -> categories_ [[2,9,nan],[0,1]];
/// transform([[nan,0]]) -> [[0,0,1,1,0]] (col0 nan at block idx 2; col1 known 0).
#[test]
fn ignore_nan_with_category_one_hots_vs_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2):
    //   categories_ == [[2.0, 9.0, nan], [0.0, 1.0]]
    //   transform([[nan,0]]) == [[0.0,0.0,1.0,1.0,0.0]]
    let sklearn_expected: Array2<f64> = array![[0.0, 0.0, 1.0, 1.0, 0.0]];
    let enc = OneHotEncoder::<f64>::new().with_handle_unknown(OneHotHandleUnknown::Ignore);
    let x_train = array![[2.0_f64, 0.0], [f64::NAN, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    // The NaN sorts LAST in col0's categories_ (#2223): {2, 9, NaN}.
    let cats0 = &fitted.categories()[0];
    assert_eq!(cats0.len(), 3, "col0 has 3 categories incl. NaN");
    assert_eq!(cats0[0], 2.0);
    assert_eq!(cats0[1], 9.0);
    assert!(cats0[2].is_nan(), "NaN category sorts last");
    let out = fitted.transform(&array![[f64::NAN, 0.0]]).unwrap();
    assert_eq!(
        out, sklearn_expected,
        "Ignore + NaN WITH a learned nan-category -> one-hots it vs sklearn oracle"
    );
}

/// `handle_unknown` ABI: default is `Error`, `with_handle_unknown` sets the mode,
/// `handle_unknown()` reads it back on both the unfitted and fitted encoder; the
/// enum default matches sklearn's `handle_unknown='error'` default (`:750`).
#[test]
fn handle_unknown_default_and_builder_abi() {
    // sklearn default handle_unknown='error' (_encoders.py:750) -> Error default.
    assert_eq!(OneHotHandleUnknown::default(), OneHotHandleUnknown::Error);
    let default_enc = OneHotEncoder::<f64>::new();
    assert_eq!(default_enc.handle_unknown(), OneHotHandleUnknown::Error);
    // Default::default() agrees with new().
    let derived: OneHotEncoder<f64> = OneHotEncoder::default();
    assert_eq!(derived.handle_unknown(), OneHotHandleUnknown::Error);

    let ignore_enc = OneHotEncoder::<f64>::new().with_handle_unknown(OneHotHandleUnknown::Ignore);
    assert_eq!(ignore_enc.handle_unknown(), OneHotHandleUnknown::Ignore);

    // The mode threads through fit to the FittedOneHotEncoder.
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    assert_eq!(
        ignore_enc.fit(&x, &()).unwrap().handle_unknown(),
        OneHotHandleUnknown::Ignore
    );
    assert_eq!(
        default_enc.fit(&x, &()).unwrap().handle_unknown(),
        OneHotHandleUnknown::Error
    );
}

// ---------------------------------------------------------------------------
// REQ-6 — inverse_transform (argmax + all-zero error) == sklearn EXACTLY.
// ---------------------------------------------------------------------------

/// `inverse_transform(transform(X)) == X` for the multi-feature fixture with the
/// NON-contiguous column `{2,5,9}`. Live sklearn:
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   e=OneHotEncoder(sparse_output=False).fit([[2.,0.],[5.,1.],[9.,0.],[5.,1.]]); \
///   print(e.inverse_transform(e.transform([[2.,0.],[5.,1.],[9.,0.],[5.,1.]])).tolist())"
///   -> [[2.0, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]]
/// ```
#[test]
fn inverse_transform_roundtrip_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp):
    //   [[2.0, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]]
    let sklearn_expected: Array2<f64> = array![[2.0, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    let encoded = fitted.transform(&x).unwrap();
    let recovered = fitted.inverse_transform(&encoded).unwrap();
    assert_eq!(
        recovered, sklearn_expected,
        "inverse_transform(transform(X)) vs sklearn oracle"
    );
    // And it recovers the original input exactly.
    assert_eq!(recovered, x, "inverse roundtrip recovers original X");
}

/// `inverse_transform` of a HELD-OUT clean one-hot matrix (not produced by this
/// fitted encoder's own transform): `[[0,1,0,1,0]]` -> `[[5,0]]` — col0 argmax
/// at block index 1 (category `5`), col1 argmax at block index 0 (category `0`).
/// Live sklearn:
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   e=OneHotEncoder(sparse_output=False).fit([[2.,0.],[5.,1.],[9.,0.],[5.,1.]]); \
///   print(e.inverse_transform([[0,1,0,1,0]]).tolist())"
///   -> [[5.0, 0.0]]
/// ```
#[test]
fn inverse_transform_held_out_one_hot_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp): [[5.0, 0.0]]
    let sklearn_expected: Array2<f64> = array![[5.0, 0.0]];
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    let held_out = array![[0.0_f64, 1.0, 0.0, 1.0, 0.0]];
    let recovered = fitted.inverse_transform(&held_out).unwrap();
    assert_eq!(
        recovered, sklearn_expected,
        "inverse_transform([[0,1,0,1,0]]) vs sklearn oracle"
    );
}

/// `inverse_transform` over MULTIPLE feature blocks (2,3,2) with a clean one-hot
/// per block recovers the original rows. Live sklearn:
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   X=[[0.,1.,8.],[1.,3.,9.],[0.,5.,8.],[1.,1.,9.]]; \
///   e=OneHotEncoder(sparse_output=False).fit(X); \
///   print(e.inverse_transform(e.transform(X)).tolist())"
///   -> [[0.0, 1.0, 8.0], [1.0, 3.0, 9.0], [0.0, 5.0, 8.0], [1.0, 1.0, 9.0]]
/// ```
#[test]
fn inverse_transform_multiblock_argmax_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp):
    let sklearn_expected: Array2<f64> = array![
        [0.0, 1.0, 8.0],
        [1.0, 3.0, 9.0],
        [0.0, 5.0, 8.0],
        [1.0, 1.0, 9.0],
    ];
    let enc = OneHotEncoder::<f64>::new();
    let x = array![
        [0.0_f64, 1.0, 8.0],
        [1.0, 3.0, 9.0],
        [0.0, 5.0, 8.0],
        [1.0, 1.0, 9.0],
    ];
    let fitted = enc.fit(&x, &()).unwrap();
    let encoded = fitted.transform(&x).unwrap();
    let recovered = fitted.inverse_transform(&encoded).unwrap();
    assert_eq!(
        recovered, sklearn_expected,
        "multiblock inverse_transform vs sklearn oracle"
    );
}

/// An ALL-ZERO per-feature block cannot be inverted (drop=None,
/// handle_unknown='error') -> Err, matching live sklearn's ValueError.
/// Live sklearn:
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   e=OneHotEncoder(sparse_output=False).fit([[2.,0.],[5.,1.],[9.,0.],[5.,1.]]); \
///   e.inverse_transform([[0,0,0,0,0]])"
///   -> ValueError: Samples [0] can not be inverted when drop=None and
///      handle_unknown='error' because they contain all zeros
/// ```
#[test]
fn inverse_transform_all_zero_block_errors_vs_sklearn_oracle() {
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    let all_zero = array![[0.0_f64, 0.0, 0.0, 0.0, 0.0]];
    let err = fitted.inverse_transform(&all_zero);
    assert!(
        err.is_err(),
        "all-zero block must error (sklearn ValueError 'can not be inverted ... all zeros')"
    );
    let msg = format!("{:?}", err.unwrap_err());
    assert!(
        msg.contains("can not be inverted") && msg.contains("all zeros"),
        "error message should mirror sklearn's all-zeros ValueError, got: {msg}"
    );
}

/// `inverse_transform` with the wrong number of columns -> Err (sklearn's
/// "Shape of the passed X data is not correct" ValueError).
/// Live sklearn: fit a 5-output encoder, `inverse_transform([[0,1,0]])`
/// -> ValueError "Expected 5 columns, got 3".
#[test]
fn inverse_transform_ncols_mismatch_errors_vs_sklearn_oracle() {
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    let bad = array![[0.0_f64, 1.0, 0.0]]; // 3 cols, expected n_output == 5
    assert!(
        fitted.inverse_transform(&bad).is_err(),
        "ncols != n_output must error (ShapeMismatch), matching sklearn"
    );
}

/// `inverse_transform` of a 0-row matrix -> Err (sklearn check_array min-1-sample).
#[test]
fn inverse_transform_zero_row_errors() {
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    let empty: Array2<f64> = Array2::zeros((0, 5));
    assert!(
        fitted.inverse_transform(&empty).is_err(),
        "0-row inverse_transform must error (InsufficientSamples)"
    );
}

// ---------------------------------------------------------------------------
// REQ-6 — get_feature_names_out == sklearn EXACTLY (float label formatting).
// ---------------------------------------------------------------------------

/// `get_feature_names_out` for WHOLE-NUMBER categories == live sklearn EXACTLY:
/// `['x0_2.0','x0_5.0','x0_9.0','x1_0.0','x1_1.0']` (default input_features
/// `x0`,`x1`; the `.0` is Python `str(np.float64)` rendering of whole floats).
/// Live sklearn:
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   e=OneHotEncoder(sparse_output=False).fit([[2.,0.],[5.,1.],[9.,0.],[5.,1.]]); \
///   print(list(e.get_feature_names_out()))"
///   -> ['x0_2.0', 'x0_5.0', 'x0_9.0', 'x1_0.0', 'x1_1.0']
/// ```
#[test]
fn get_feature_names_out_whole_numbers_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp):
    let sklearn_expected = ["x0_2.0", "x0_5.0", "x0_9.0", "x1_0.0", "x1_1.0"];
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    assert_eq!(
        fitted.get_feature_names_out(),
        sklearn_expected,
        "get_feature_names_out (whole numbers) vs sklearn oracle"
    );
}

/// `get_feature_names_out` with a FRACTIONAL category renders `2.5` as `x0_2.5`.
/// Live sklearn:
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   e=OneHotEncoder(sparse_output=False).fit([[2.5,0.],[5.,1.]]); \
///   print(list(e.get_feature_names_out()))"
///   -> ['x0_2.5', 'x0_5.0', 'x1_0.0', 'x1_1.0']
/// ```
#[test]
fn get_feature_names_out_fractional_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp):
    let sklearn_expected = ["x0_2.5", "x0_5.0", "x1_0.0", "x1_1.0"];
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.5_f64, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    assert_eq!(
        fitted.get_feature_names_out(),
        sklearn_expected,
        "get_feature_names_out (fractional 2.5) vs sklearn oracle"
    );
}

/// `get_feature_names_out` with a NEGATIVE whole category renders `-3.0` as
/// `x0_-3.0`. Live sklearn:
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   e=OneHotEncoder(sparse_output=False).fit([[-3.,0.],[5.,1.]]); \
///   print(list(e.get_feature_names_out()))"
///   -> ['x0_-3.0', 'x0_5.0', 'x1_0.0', 'x1_1.0']
/// ```
#[test]
fn get_feature_names_out_negative_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp):
    let sklearn_expected = ["x0_-3.0", "x0_5.0", "x1_0.0", "x1_1.0"];
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[-3.0_f64, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    assert_eq!(
        fitted.get_feature_names_out(),
        sklearn_expected,
        "get_feature_names_out (negative -3.0) vs sklearn oracle"
    );
}

// ---------------------------------------------------------------------------
// REQ-6 DIVERGENCE — inverse_transform does NOT validate NaN/inf input.
// tracking #2224
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `FittedOneHotEncoder::inverse_transform`
/// (`ferrolearn-preprocess/src/one_hot_encoder.rs:182`) diverges from
/// `sklearn/preprocessing/_encoders.py:1092` for an encoded matrix that
/// contains a `NaN` (or `inf`).
///
/// sklearn's `inverse_transform` runs
/// `X = check_array(X, accept_sparse="csr")` (`_encoders.py:1092`) with the
/// default `force_all_finite=True`, so ANY non-finite cell in the input matrix
/// raises before the argmax ever runs:
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import OneHotEncoder; \
///   e=OneHotEncoder(sparse_output=False).fit([[2.],[5.],[9.]]); \
///   e.inverse_transform(np.array([[0.0, np.nan, 0.0]]))"
///   -> ValueError: Input contains NaN.
/// python3 -c "import numpy as np; from sklearn.preprocessing import OneHotEncoder; \
///   e=OneHotEncoder(sparse_output=False).fit([[2.],[5.],[9.]]); \
///   e.inverse_transform(np.array([[np.inf, 0.0, 0.0]]))"
///   -> ValueError: Input contains infinity or a value too large for dtype('float64').
/// ```
///
/// ferrolearn performs NO finiteness check: for `[[0, NaN, 0]]` the block sum is
/// `NaN` (so `block_sum == 0` is false → no all-zero error) and the argmax loop
/// `v > max_val` skips the `NaN`, leaving `argmax = 0`, so it returns
/// `Ok([[2.0]])` instead of erroring. Likewise `[[inf, 0, 0]]` returns
/// `Ok([[inf... → 2.0]])`. sklearn ERRORS; ferrolearn returns `Ok`.
///
/// This is a release-blocker observable-behavior divergence (sklearn raises a
/// `ValueError`, ferrolearn silently produces output) and is NOT covered by the
/// 8 REQ-6 tests nor by the documented extreme-magnitude float-label divergence.
#[test]
fn inverse_transform_nan_input_errors_vs_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp):
    //   inverse_transform([[0.0, nan, 0.0]]) -> ValueError: Input contains NaN.
    //   inverse_transform([[inf, 0.0, 0.0]]) -> ValueError: Input contains infinity ...
    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2.0_f64], [5.0], [9.0]];
    let fitted = enc.fit(&x, &()).unwrap();

    // A block with a NaN: sklearn errors at check_array; ferrolearn must too.
    let nan_block = array![[0.0_f64, f64::NAN, 0.0]];
    assert!(
        fitted.inverse_transform(&nan_block).is_err(),
        "inverse_transform of a NaN-containing matrix must error, matching sklearn's \
         check_array `ValueError: Input contains NaN.` (_encoders.py:1092); \
         ferrolearn silently returns Ok"
    );

    // A block with an inf: sklearn errors at check_array; ferrolearn must too.
    let inf_block = array![[f64::INFINITY, 0.0, 0.0]];
    assert!(
        fitted.inverse_transform(&inf_block).is_err(),
        "inverse_transform of an inf-containing matrix must error, matching sklearn's \
         check_array `ValueError: Input contains infinity ...` (_encoders.py:1092); \
         ferrolearn silently returns Ok"
    );
}

// ---------------------------------------------------------------------------
// REQ-6 DIVERGENCE — inverse_transform of an IGNORE-mode all-zero block must
// NOT error; sklearn returns None for that feature (the OTHER features still
// invert). ferrolearn's inverse_transform ignores `handle_unknown` and ALWAYS
// errors on an all-zero block. tracking #2227
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `FittedOneHotEncoder::inverse_transform`
/// (`ferrolearn-preprocess/src/one_hot_encoder.rs:316-323`) diverges from
/// `sklearn/preprocessing/_encoders.py:1141-1158,1178-1183` for an encoder
/// fitted with `handle_unknown='ignore'` when a per-feature block is all-zero.
///
/// sklearn's `inverse_transform` consults `self.handle_unknown`: when it is
/// `'ignore'` (`_encoders.py:1141`), an all-zero block (`sub.sum(axis=1) == 0`,
/// `:1145`) is collected into `found_unknown[i]` (`:1154`) and, at `:1178-1183`,
/// the result is upcast to object and the cell set to `None` — it is NOT an
/// error. Only when `handle_unknown != 'ignore'` (the `else` at `:1159`) does an
/// all-zero block raise the "can not be inverted ... all zeros" `ValueError`
/// (`:1164-1168`). The error path is GATED on the mode; the all-zero block in
/// the OTHER (known) features still inverts normally.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import OneHotEncoder; \
///   e=OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit([[2.,0.],[5.,1.],[9.,0.]]); \
///   print(e.inverse_transform([[0.0,0.0,0.0,1.0,0.0]]).tolist())"
///   -> [[None, 0.0]]
/// ```
/// col0's block `[0,0,0]` is all-zero → sklearn returns `None` (not an error);
/// col1's block `[1,0]` argmaxes to category `0.0`. The recovered row is
/// `[None, 0.0]`.
///
/// ferrolearn's `inverse_transform` performs the all-zero check
/// (`block_sum == F::zero()` at `:316`) UNCONDITIONALLY — it never consults
/// `self.handle_unknown` — so for THIS ignore-mode encoder it returns
/// `Err(InvalidParameter "... can not be inverted ... all zeros")` instead of
/// recovering col1 and denoting col0 as unknown. sklearn returns `Ok`;
/// ferrolearn returns `Err`.
///
/// This is a release-blocker observable divergence: sklearn's documented
/// `handle_unknown='ignore'` contract is "In the inverse transform, an unknown
/// category will be denoted as None" (`_encoders.py:546-549`). ferrolearn errors
/// where sklearn succeeds. Even granting that `Array2<F>` cannot hold a Python
/// `None`, the KNOWN feature (col1 → `0.0`) is independently invertible and the
/// call must not error in `ignore` mode. This test asserts (a) the call does NOT
/// error and (b) col1 recovers `0.0` — both fail today (the call returns `Err`).
#[test]
fn inverse_transform_ignore_all_zero_block_returns_none_not_err_vs_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, handle_unknown='ignore', run from /tmp):
    //   inverse_transform([[0.0,0.0,0.0,1.0,0.0]]) -> [[None, 0.0]]
    // (col0 all-zero block -> None/unknown; col1 known one-hot -> 0.0)
    let enc = OneHotEncoder::<f64>::new().with_handle_unknown(OneHotHandleUnknown::Ignore);
    let x_train = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x_train, &()).unwrap();

    // Row: col0 block [0,0,0] all-zero (unknown), col1 block [1,0] -> category 0.0.
    let encoded = array![[0.0_f64, 0.0, 0.0, 1.0, 0.0]];
    let recovered = fitted.inverse_transform(&encoded);

    // sklearn does NOT error in ignore mode: the all-zero block becomes None and
    // every OTHER feature still inverts. ferrolearn unconditionally errors.
    assert!(
        recovered.is_ok(),
        "ignore-mode inverse_transform of an all-zero block must NOT error \
         (sklearn `_encoders.py:1141` denotes it None, not a ValueError); \
         ferrolearn errors unconditionally regardless of handle_unknown"
    );

    // The KNOWN feature (col1) must still recover its category 0.0 (sklearn's
    // [[None, 0.0]] — col1 is independently invertible).
    let out = recovered.unwrap();
    assert_eq!(
        out[[0, 1]],
        0.0,
        "ignore-mode inverse_transform must still recover the KNOWN col1 \
         category 0.0 (sklearn returns [[None, 0.0]])"
    );
}

// ---------------------------------------------------------------------------
// REQ-5a — drop {None,'first','if_binary'} == sklearn EXACTLY.
//   (`sklearn/preprocessing/_encoders.py` `drop` `_parameter_constraints` `:730`;
//    `_compute_drop_idx` `:812-831`; transform drop-shift `:1033-1046`;
//    inverse all-zero->dropped `:1124-1172`; feature-name omit `:909`,`:1209`.)
//
// Every expected value below is a LIVE sklearn 1.5.2 call (sparse_output=False,
// drop=...), quoted inline (R-CHAR-3) — NEVER copied from ferrolearn:
//
//   # drop='first' on [[2,0],[5,1],[9,0]]:
//   python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//     e=OneHotEncoder(sparse_output=False, drop='first').fit([[2.,0.],[5.,1.],[9.,0.]]); \
//     print(list(e.drop_idx_)); print(e.transform([[2.,0.],[5.,1.],[9.,0.]]).tolist()); \
//     print(list(e.get_feature_names_out())); \
//     print(e.inverse_transform(e.transform([[2.,0.],[5.,1.],[9.,0.]])).tolist())"
//     -> drop_idx_ [0, 0]
//        transform [[0,0,0],[1,0,1],[0,1,0]]
//        names ['x0_5.0','x0_9.0','x1_1.0']
//        inverse [[2,0],[5,1],[9,0]]
//
//   # drop='if_binary' on [[2,0,8],[5,1,9],[9,0,8]] (3/2/2 cats):
//   python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//     X=[[2.,0.,8.],[5.,1.,9.],[9.,0.,8.]]; \
//     e=OneHotEncoder(sparse_output=False, drop='if_binary').fit(X); \
//     print(list(e.drop_idx_)); print(e.transform(X).tolist()); \
//     print(list(e.get_feature_names_out())); print(e.inverse_transform(e.transform(X)).tolist())"
//     -> drop_idx_ [None, 0, 0]
//        transform [[1,0,0,0,0],[0,1,0,1,1],[0,0,1,0,0]]
//        names ['x0_2.0','x0_5.0','x0_9.0','x1_1.0','x2_9.0']
//        inverse [[2,0,8],[5,1,9],[9,0,8]]
//
//   # drop-shift, single col {2,5,9} drop first (cat 9 at block idx 2 -> col 1):
//   python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//     e=OneHotEncoder(sparse_output=False, drop='first').fit([[2.],[5.],[9.]]); \
//     print(e.transform([[2.]]).tolist(), e.transform([[5.]]).tolist(), e.transform([[9.]]).tolist())"
//     -> [[0,0]] [[1,0]] [[0,1]]
//
//   # single-category feature fully dropped by drop='first' (block width 0):
//   python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//     X=[[2.,7.],[5.,7.],[9.,7.]]; \
//     e=OneHotEncoder(sparse_output=False, drop='first').fit(X); \
//     print(list(e.drop_idx_), e.transform(X).shape[1]); \
//     print(e.transform(X).tolist()); print(list(e.get_feature_names_out())); \
//     print(e.inverse_transform(e.transform(X)).tolist())"
//     -> drop_idx_ [0, 0], n_out 2
//        transform [[0,0],[1,0],[0,1]]
//        names ['x0_5.0','x0_9.0']
//        inverse [[2,7],[5,7],[9,7]]
//
//   # drop + handle_unknown='ignore' is ALLOWED (does NOT raise at fit):
//   python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//     e=OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore').fit([[2.,0.],[5.,1.],[9.,0.]]); \
//     print(list(e.drop_idx_)); \
//     print(e.transform([[100.,0.]]).tolist())  # unknown -> all-zero (like dropped)"
//     -> drop_idx_ [0, 0]; transform [[0,0,0]]  (col0 100 unknown -> [0,0]; col1 0 -> [0])  (warns)
// ---------------------------------------------------------------------------

/// `drop='first'`: `drop_idx_ == [Some(0),Some(0)]`, the transform matrix, the
/// n_output (3 = 2+1), and `get_feature_names_out` all match the live sklearn
/// oracle on `[[2,0],[5,1],[9,0]]`.
#[test]
fn drop_first_transform_idx_names_match_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, drop='first'):
    //   drop_idx_ [0,0]; transform [[0,0,0],[1,0,1],[0,1,0]];
    //   names ['x0_5.0','x0_9.0','x1_1.0']
    let sklearn_transform: Array2<f64> = array![[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    let enc = OneHotEncoder::<f64>::new().with_drop(OneHotDrop::First);
    assert_eq!(enc.drop(), OneHotDrop::First, "with_drop sets the mode");
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(
        fitted.drop_idx_(),
        &[Some(0), Some(0)],
        "drop='first' drop_idx_ == [Some(0),Some(0)] vs sklearn [0,0]"
    );
    assert_eq!(
        fitted.n_output_features(),
        3,
        "drop='first' n_output == 2+1 == 3 (sklearn)"
    );
    let out = fitted.transform(&x).unwrap();
    assert_eq!(out, sklearn_transform, "drop='first' transform vs sklearn");
    assert_eq!(
        fitted.get_feature_names_out(),
        ["x0_5.0", "x0_9.0", "x1_1.0"],
        "drop='first' get_feature_names_out (dropped cat omitted) vs sklearn"
    );
}

/// `drop='if_binary'` on a 3/2/2-cat fixture: `drop_idx_ == [None,Some(0),Some(0)]`
/// (only the binary features lose a category), transform + n_output + names ==
/// the live sklearn oracle on `[[2,0,8],[5,1,9],[9,0,8]]`.
#[test]
fn drop_if_binary_idx_transform_names_match_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, drop='if_binary'):
    //   drop_idx_ [None,0,0]; n_out 5; transform [[1,0,0,0,0],[0,1,0,1,1],[0,0,1,0,0]];
    //   names ['x0_2.0','x0_5.0','x0_9.0','x1_1.0','x2_9.0']
    let sklearn_transform: Array2<f64> = array![
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ];
    let enc = OneHotEncoder::<f64>::new().with_drop(OneHotDrop::IfBinary);
    let x = array![[2.0_f64, 0.0, 8.0], [5.0, 1.0, 9.0], [9.0, 0.0, 8.0]];
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(
        fitted.drop_idx_(),
        &[None, Some(0), Some(0)],
        "drop='if_binary' drop_idx_ == [None,Some(0),Some(0)] vs sklearn [None,0,0]"
    );
    assert_eq!(
        fitted.n_output_features(),
        5,
        "drop='if_binary' n_output == 3+1+1 == 5 (sklearn)"
    );
    let out = fitted.transform(&x).unwrap();
    assert_eq!(
        out, sklearn_transform,
        "drop='if_binary' transform vs sklearn"
    );
    assert_eq!(
        fitted.get_feature_names_out(),
        ["x0_2.0", "x0_5.0", "x0_9.0", "x1_1.0", "x2_9.0"],
        "drop='if_binary' names (only binary features drop) vs sklearn"
    );
}

/// `inverse_transform(transform(X)) == X` WITH drop='first': an all-zero block
/// decodes to the DROPPED category (not an error, not None). Live sklearn:
/// inverse of the dropped-category row recovers the original.
#[test]
fn drop_first_inverse_roundtrip_dropped_category_match_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, drop='first'):
    //   inverse(transform(X)) == [[2,0],[5,1],[9,0]]  (the all-zero col0 block of
    //   row 0 decodes to the DROPPED category 2.0)
    let sklearn_expected: Array2<f64> = array![[2.0, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let enc = OneHotEncoder::<f64>::new().with_drop(OneHotDrop::First);
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    let encoded = fitted.transform(&x).unwrap();
    let recovered = fitted.inverse_transform(&encoded).unwrap();
    assert_eq!(
        recovered, sklearn_expected,
        "drop='first' inverse_transform(transform(X)) (all-zero -> dropped cat) vs sklearn"
    );
    assert_eq!(recovered, x, "drop roundtrip recovers original X");
}

/// A single-category feature with `drop='first'` is dropped ENTIRELY (block width
/// 0): the inverse still recovers that (dropped, only) category. Live sklearn on
/// `[[2,7],[5,7],[9,7]]`: col1 has the single category 7.0 -> dropped -> 0 cols;
/// inverse fills 7.0.
#[test]
fn drop_first_single_category_fully_dropped_match_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, drop='first'):
    //   drop_idx_ [0,0]; n_out 2; transform [[0,0],[1,0],[0,1]];
    //   names ['x0_5.0','x0_9.0']; inverse [[2,7],[5,7],[9,7]]
    let sklearn_transform: Array2<f64> = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let sklearn_inverse: Array2<f64> = array![[2.0, 7.0], [5.0, 7.0], [9.0, 7.0]];
    let enc = OneHotEncoder::<f64>::new().with_drop(OneHotDrop::First);
    let x = array![[2.0_f64, 7.0], [5.0, 7.0], [9.0, 7.0]];
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(
        fitted.drop_idx_(),
        &[Some(0), Some(0)],
        "single-cat col1 still gets drop_idx_ Some(0)"
    );
    assert_eq!(
        fitted.n_output_features(),
        2,
        "single-cat col1 fully dropped -> only col0's 2 kept cols (sklearn n_out 2)"
    );
    let out = fitted.transform(&x).unwrap();
    assert_eq!(
        out, sklearn_transform,
        "single-cat-dropped transform vs sklearn"
    );
    assert_eq!(
        fitted.get_feature_names_out(),
        ["x0_5.0", "x0_9.0"],
        "single-cat-dropped feature contributes no name vs sklearn"
    );
    let recovered = fitted.inverse_transform(&out).unwrap();
    assert_eq!(
        recovered, sklearn_inverse,
        "fully-dropped feature inverts to its dropped category 7.0 vs sklearn"
    );
}

/// The drop-shift: a feature with 3+ categories where the dropped idx is 0, a
/// NON-dropped category maps to the correctly shifted column. Live sklearn on
/// `{2,5,9}` drop='first': cat 9 (membership idx 2) -> block col 1 (2-1), cat 5
/// (idx 1) -> col 0, cat 2 (idx 0, dropped) -> all-zero.
#[test]
fn drop_first_shift_3cat_columns_match_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, drop='first', single col {2,5,9}):
    //   transform([[2.]]) [[0,0]]; transform([[5.]]) [[1,0]]; transform([[9.]]) [[0,1]]
    let enc = OneHotEncoder::<f64>::new().with_drop(OneHotDrop::First);
    let x = array![[2.0_f64], [5.0], [9.0]];
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(
        fitted.transform(&array![[2.0_f64]]).unwrap(),
        array![[0.0_f64, 0.0]],
        "dropped cat 2 (idx 0) -> all-zero block vs sklearn"
    );
    assert_eq!(
        fitted.transform(&array![[5.0_f64]]).unwrap(),
        array![[1.0_f64, 0.0]],
        "kept cat 5 (idx 1) -> shifted col 0 vs sklearn"
    );
    assert_eq!(
        fitted.transform(&array![[9.0_f64]]).unwrap(),
        array![[0.0_f64, 1.0]],
        "kept cat 9 (idx 2, > dropped 0) -> shifted col 1 (idx-1) vs sklearn"
    );
    // And the shifted block inverts back through cat_idx = pos+1 for pos>=d.
    assert_eq!(
        fitted.inverse_transform(&array![[0.0_f64, 1.0]]).unwrap(),
        array![[9.0_f64]],
        "block pos 1 (>= dropped 0) -> category idx 2 == 9.0 vs sklearn"
    );
}

/// `drop` + `handle_unknown='ignore'` is ALLOWED in sklearn 1.5.2 (does NOT raise
/// at fit). Mirror it: fit succeeds, and an unknown value encodes to an all-zero
/// block (identical to the dropped category). Live sklearn:
/// `OneHotEncoder(drop='first', handle_unknown='ignore').fit([[2,0],[5,1],[9,0]])`
/// does not raise; `transform([[100,0]])` -> `[[0,0,0]]` (warns).
#[test]
fn drop_plus_ignore_allowed_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2): fit does NOT raise; drop_idx_ [0,0];
    //   transform([[100,0]]) -> [[0,0,0]] (col0 100 unknown -> all-zero like dropped).
    let enc = OneHotEncoder::<f64>::new()
        .with_drop(OneHotDrop::First)
        .with_handle_unknown(OneHotHandleUnknown::Ignore);
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    // sklearn does NOT raise at fit for drop + handle_unknown='ignore'.
    let fitted = enc
        .fit(&x, &())
        .expect("drop + handle_unknown='ignore' must NOT error at fit (sklearn 1.5.2 allows it)");
    assert_eq!(fitted.drop_idx_(), &[Some(0), Some(0)]);
    // An unknown value -> all-zero block (same as the dropped category).
    let out = fitted.transform(&array![[100.0_f64, 0.0]]).unwrap();
    assert_eq!(
        out,
        array![[0.0_f64, 0.0, 0.0]],
        "drop+ignore: unknown col0 -> all-zero; col1 0 is the dropped cat -> all-zero vs sklearn"
    );
}

/// `drop` ABI: default is `None_`, `with_drop` sets the mode, `drop()` reads it
/// back; `drop_idx_()` is all-`None` for the default and exposed on the fitted
/// encoder. The enum default matches sklearn's `drop=None` default (`:747`).
#[test]
fn drop_default_and_builder_abi() {
    // sklearn default drop=None (_encoders.py:747) -> None_ default.
    assert_eq!(OneHotDrop::default(), OneHotDrop::None_);
    let default_enc = OneHotEncoder::<f64>::new();
    assert_eq!(default_enc.drop(), OneHotDrop::None_);
    let derived: OneHotEncoder<f64> = OneHotEncoder::default();
    assert_eq!(derived.drop(), OneHotDrop::None_);

    let first_enc = OneHotEncoder::<f64>::new().with_drop(OneHotDrop::First);
    assert_eq!(first_enc.drop(), OneHotDrop::First);

    // The default-mode fitted encoder reports all-None drop_idx_ (no drop).
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0]];
    let fitted = default_enc.fit(&x, &()).unwrap();
    assert_eq!(
        fitted.drop_idx_(),
        &[None, None],
        "drop=None -> drop_idx_ all None (sklearn drop_idx_ is None)"
    );
}

/// `drop=None` (the default) leaves REQ-3/4/6 unchanged: the multifeature fixture
/// transforms to the FULL one-hot (no column dropped), matching the existing
/// REQ-3 oracle. Regression guard that adding `drop` did not perturb the default.
#[test]
fn drop_none_default_transform_unchanged_vs_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, drop=None == default): the same matrix as the
    // REQ-1/REQ-3 `transform_multifeature_matches_sklearn_oracle` test.
    let sklearn_expected: Array2<f64> = array![
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
    ];
    let enc = OneHotEncoder::<f64>::new().with_drop(OneHotDrop::None_);
    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    assert_eq!(
        fitted.n_output_features(),
        5,
        "drop=None keeps all 5 columns"
    );
    assert_eq!(
        fitted.transform(&x).unwrap(),
        sklearn_expected,
        "drop=None default transform == full one-hot (REQ-3 unchanged)"
    );
    // Roundtrip still works (REQ-6 unchanged under drop=None).
    let recovered = fitted
        .inverse_transform(&fitted.transform(&x).unwrap())
        .unwrap();
    assert_eq!(recovered, x, "drop=None inverse roundtrip unchanged");
}

// ===========================================================================
// REQ-5b: infrequent grouping (min_frequency / max_categories)
//
// R-CHAR-3: every expected value below is the output of a LIVE sklearn 1.5.2
// `OneHotEncoder(sparse_output=False, min_frequency=/max_categories=)` call,
// quoted inline as the `python3 -c` command that produced it. Sparse default is
// not used (we ship dense).
// ===========================================================================

/// Build a single-column `Array2<f64>` from `(value, count)` pairs (the rows are
/// `value` repeated `count` times, concatenated in order).
fn col_from_counts(pairs: &[(f64, usize)]) -> Array2<f64> {
    let mut rows: Vec<f64> = Vec::new();
    for &(v, c) in pairs {
        for _ in 0..c {
            rows.push(v);
        }
    }
    let n = rows.len();
    Array2::from_shape_vec((n, 1), rows).unwrap()
}

/// `min_frequency=2` headline (the dispatch fixture).
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   X=[[1.]]*5+[[2.]]*5+[[3.]]*1+[[4.]]*1; \
///   e=OneHotEncoder(sparse_output=False, min_frequency=2).fit(X); \
///   print([c.tolist() for c in e.categories_]); \
///   print([c.tolist() for c in e.infrequent_categories_]); \
///   print(e.transform([[1.],[2.],[3.],[4.]]).tolist()); \
///   print(list(e.get_feature_names_out()))"
///   -> [[1.0, 2.0, 3.0, 4.0]]
///      [[3.0, 4.0]]
///      [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,1.0]]
///      ['x0_1.0', 'x0_2.0', 'x0_infrequent_sklearn']
/// ```
#[test]
fn infrequent_min_frequency_basic_vs_sklearn_oracle() {
    let x = col_from_counts(&[(1.0, 5), (2.0, 5), (3.0, 1), (4.0, 1)]);
    let enc = OneHotEncoder::<f64>::new().with_min_frequency(2);
    let fitted = enc.fit(&x, &()).unwrap();

    // categories_ keeps ALL categories (the infrequent ones are still present).
    assert_eq!(fitted.categories(), &[vec![1.0, 2.0, 3.0, 4.0]]);
    // infrequent_categories_ == [[3.0, 4.0]].
    assert_eq!(fitted.infrequent_categories(), vec![vec![3.0, 4.0]]);
    // n_output == 3 (2 frequent + 1 trailing infrequent column).
    assert_eq!(fitted.n_output_features(), 3);

    let probe = array![[1.0_f64], [2.0], [3.0], [4.0]];
    let expected: Array2<f64> = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], // 3.0 -> trailing infrequent col
        [0.0, 0.0, 1.0], // 4.0 -> trailing infrequent col
    ];
    assert_eq!(fitted.transform(&probe).unwrap(), expected);

    assert_eq!(
        fitted.get_feature_names_out(),
        vec!["x0_1.0", "x0_2.0", "x0_infrequent_sklearn"]
    );
}

/// `max_categories=3` with distinct counts `{1:5,2:4,3:3,4:1}`.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   X=[[1.]]*5+[[2.]]*4+[[3.]]*3+[[4.]]*1; \
///   e=OneHotEncoder(sparse_output=False, max_categories=3).fit(X); \
///   print([c.tolist() for c in e.infrequent_categories_]); \
///   print(e.transform([[1.],[2.],[3.],[4.]]).tolist())"
///   -> [[3.0, 4.0]]
///      [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,1.0]]
/// ```
#[test]
fn infrequent_max_categories_topk_vs_sklearn_oracle() {
    let x = col_from_counts(&[(1.0, 5), (2.0, 4), (3.0, 3), (4.0, 1)]);
    let enc = OneHotEncoder::<f64>::new().with_max_categories(3);
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(fitted.infrequent_categories(), vec![vec![3.0, 4.0]]);
    assert_eq!(fitted.n_output_features(), 3);

    let probe = array![[1.0_f64], [2.0], [3.0], [4.0]];
    let expected: Array2<f64> = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    assert_eq!(fitted.transform(&probe).unwrap(), expected);
}

/// `max_categories=3` TIE-BREAK: all four categories share count 3.
///
/// sklearn's stable argsort keeps the LARGER category indices frequent, so the
/// SMALLER ones (1.0, 2.0) become infrequent.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   X=[[1.]]*3+[[2.]]*3+[[3.]]*3+[[4.]]*3; \
///   e=OneHotEncoder(sparse_output=False, max_categories=3).fit(X); \
///   print([c.tolist() for c in e.infrequent_categories_]); \
///   print(e.transform([[1.],[2.],[3.],[4.]]).tolist())"
///   -> [[1.0, 2.0]]
///      [[0.0,0.0,1.0],[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0]]
/// ```
#[test]
fn infrequent_max_categories_tiebreak_vs_sklearn_oracle() {
    let x = col_from_counts(&[(1.0, 3), (2.0, 3), (3.0, 3), (4.0, 3)]);
    let enc = OneHotEncoder::<f64>::new().with_max_categories(3);
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(fitted.infrequent_categories(), vec![vec![1.0, 2.0]]);

    let probe = array![[1.0_f64], [2.0], [3.0], [4.0]];
    let expected: Array2<f64> = array![
        [0.0, 0.0, 1.0], // 1.0 infrequent -> trailing col
        [0.0, 0.0, 1.0], // 2.0 infrequent -> trailing col
        [1.0, 0.0, 0.0], // 3.0 frequent -> slot 0
        [0.0, 1.0, 0.0], // 4.0 frequent -> slot 1
    ];
    assert_eq!(fitted.transform(&probe).unwrap(), expected);
}

/// BOTH `min_frequency=2` AND `max_categories=3` set, counts
/// `{1:5,2:4,3:3,4:2,5:1}`. min_frequency removes only `5` (count 1); then
/// max_categories trims the survivors to the top 2 frequent `{1,2}`, grouping
/// `{3,4,5}` infrequent.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   X=[[1.]]*5+[[2.]]*4+[[3.]]*3+[[4.]]*2+[[5.]]*1; \
///   e=OneHotEncoder(sparse_output=False, min_frequency=2, max_categories=3).fit(X); \
///   print([c.tolist() for c in e.infrequent_categories_]); \
///   print(e.transform([[1.],[2.],[3.],[4.],[5.]]).tolist()); \
///   print(list(e.get_feature_names_out()))"
///   -> [[3.0, 4.0, 5.0]]
///      [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]]
///      ['x0_1.0', 'x0_2.0', 'x0_infrequent_sklearn']
/// ```
#[test]
fn infrequent_both_thresholds_vs_sklearn_oracle() {
    let x = col_from_counts(&[(1.0, 5), (2.0, 4), (3.0, 3), (4.0, 2), (5.0, 1)]);
    let enc = OneHotEncoder::<f64>::new()
        .with_min_frequency(2)
        .with_max_categories(3);
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(fitted.infrequent_categories(), vec![vec![3.0, 4.0, 5.0]]);

    let probe = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
    let expected: Array2<f64> = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    assert_eq!(fitted.transform(&probe).unwrap(), expected);
    assert_eq!(
        fitted.get_feature_names_out(),
        vec!["x0_1.0", "x0_2.0", "x0_infrequent_sklearn"]
    );
}

/// Multi-feature: only col0 has infrequent categories; col1 (counts `{10:6,20:5}`)
/// has none. The offsets must place col1's block right after col0's 3-wide block.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   X=[[1.,10.]]*5+[[2.,20.]]*5+[[3.,10.]]*1; \
///   e=OneHotEncoder(sparse_output=False, min_frequency=2).fit(X); \
///   print([None if c is None else c.tolist() for c in e.infrequent_categories_]); \
///   print(e.transform([[1.,10.],[2.,20.],[3.,10.]]).tolist()); \
///   print(list(e.get_feature_names_out()))"
///   -> [[3.0], None]
///      [[1.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0,1.0],[0.0,0.0,1.0,1.0,0.0]]
///      ['x0_1.0','x0_2.0','x0_infrequent_sklearn','x1_10.0','x1_20.0']
/// ```
#[test]
fn infrequent_multifeature_offsets_vs_sklearn_oracle() {
    // col0: 1.0 x5, 2.0 x5, 3.0 x1 (3.0 infrequent). col1: 10.0 x6, 20.0 x5.
    let mut rows: Vec<[f64; 2]> = Vec::new();
    for _ in 0..5 {
        rows.push([1.0, 10.0]);
    }
    for _ in 0..5 {
        rows.push([2.0, 20.0]);
    }
    rows.push([3.0, 10.0]);
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let x = Array2::from_shape_vec((rows.len(), 2), flat).unwrap();

    let enc = OneHotEncoder::<f64>::new().with_min_frequency(2);
    let fitted = enc.fit(&x, &()).unwrap();

    // col0 has [3.0] infrequent; col1 has none (empty).
    assert_eq!(fitted.infrequent_categories(), vec![vec![3.0], vec![]]);
    assert_eq!(fitted.n_output_features(), 5); // (2 freq + 1 infreq) + 2

    let probe = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 10.0]];
    let expected: Array2<f64> = array![
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0], // 3.0 -> col0 infrequent, 10.0 -> col1 slot 0
    ];
    assert_eq!(fitted.transform(&probe).unwrap(), expected);
    assert_eq!(
        fitted.get_feature_names_out(),
        vec![
            "x0_1.0",
            "x0_2.0",
            "x0_infrequent_sklearn",
            "x1_10.0",
            "x1_20.0"
        ]
    );
}

/// A feature whose categories are all above threshold has NO infrequent column;
/// the block is the plain full one-hot (REQ-3 layout unchanged).
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   X=[[1.]]*5+[[2.]]*5+[[3.]]*5; \
///   e=OneHotEncoder(sparse_output=False, min_frequency=2).fit(X); \
///   print([None if c is None else c.tolist() for c in e.infrequent_categories_]); \
///   print(e.transform([[1.],[2.],[3.]]).tolist()); \
///   print(list(e.get_feature_names_out()))"
///   -> [None]
///      [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
///      ['x0_1.0', 'x0_2.0', 'x0_3.0']
/// ```
#[test]
fn infrequent_no_infrequent_full_block_vs_sklearn_oracle() {
    let x = col_from_counts(&[(1.0, 5), (2.0, 5), (3.0, 5)]);
    let enc = OneHotEncoder::<f64>::new().with_min_frequency(2);
    let fitted = enc.fit(&x, &()).unwrap();

    // No infrequent categories: empty list (sklearn's None).
    assert_eq!(fitted.infrequent_categories(), vec![Vec::<f64>::new()]);
    assert_eq!(fitted.n_output_features(), 3); // no extra infrequent column

    let probe = array![[1.0_f64], [2.0], [3.0]];
    let expected: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert_eq!(fitted.transform(&probe).unwrap(), expected);
    assert_eq!(
        fitted.get_feature_names_out(),
        vec!["x0_1.0", "x0_2.0", "x0_3.0"]
    );
}

/// inverse_transform: a frequent column inverts to its category; the trailing
/// infrequent column inverts to `NaN` (DOCUMENTED divergence — sklearn returns
/// the string `'infrequent_sklearn'` which `Array2<f64>` cannot hold).
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   X=[[1.]]*5+[[2.]]*5+[[3.]]*1+[[4.]]*1; \
///   e=OneHotEncoder(sparse_output=False, min_frequency=2).fit(X); \
///   print(e.inverse_transform([[0.,0.,1.]]).tolist()); \
///   print(e.inverse_transform([[1.,0.,0.],[0.,1.,0.]]).tolist())"
///   -> [['infrequent_sklearn']]   (we map -> NaN)
///      [[1.0], [2.0]]
/// ```
#[test]
fn infrequent_inverse_trailing_col_is_nan_vs_sklearn_oracle() {
    let x = col_from_counts(&[(1.0, 5), (2.0, 5), (3.0, 1), (4.0, 1)]);
    let enc = OneHotEncoder::<f64>::new().with_min_frequency(2);
    let fitted = enc.fit(&x, &()).unwrap();

    // The infrequent column -> NaN (sklearn 'infrequent_sklearn'; not
    // representable in Array2<f64>).
    let inv_infreq = fitted
        .inverse_transform(&array![[0.0_f64, 0.0, 1.0]])
        .unwrap();
    assert!(
        inv_infreq[[0, 0]].is_nan(),
        "infrequent column inverts to NaN (sklearn 'infrequent_sklearn')"
    );

    // Frequent columns invert to their category values.
    let inv_freq = fitted
        .inverse_transform(&array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]])
        .unwrap();
    assert_eq!(inv_freq, array![[1.0_f64], [2.0]]);

    // Roundtrip: transform [1,3,4,2] then invert -> [1, NaN, NaN, 2]
    // (3.0 and 4.0 are infrequent -> NaN).
    // sklearn: inverse_transform(transform([[1.],[3.],[4.],[2.]]))
    //   -> [1.0, 'infrequent_sklearn', 'infrequent_sklearn', 2.0]
    let probe = array![[1.0_f64], [3.0], [4.0], [2.0]];
    let recovered = fitted
        .inverse_transform(&fitted.transform(&probe).unwrap())
        .unwrap();
    assert_eq!(recovered[[0, 0]], 1.0);
    assert!(recovered[[1, 0]].is_nan());
    assert!(recovered[[2, 0]].is_nan());
    assert_eq!(recovered[[3, 0]], 2.0);
}

/// `min_frequency` (or `max_categories`) combined with `drop` is a DEFERRED
/// interaction (REQ-5a × REQ-5b): `fit` returns an error (sklearn allows it).
#[test]
fn infrequent_with_drop_is_rejected() {
    let x = col_from_counts(&[(1.0, 5), (2.0, 5), (3.0, 1)]);
    let enc = OneHotEncoder::<f64>::new()
        .with_min_frequency(2)
        .with_drop(OneHotDrop::First);
    let err = enc.fit(&x, &());
    assert!(
        err.is_err(),
        "infrequent grouping + drop must be rejected (deferred interaction)"
    );

    let enc2 = OneHotEncoder::<f64>::new()
        .with_max_categories(2)
        .with_drop(OneHotDrop::IfBinary);
    assert!(
        enc2.fit(&x, &()).is_err(),
        "max_categories + drop=if_binary also rejected"
    );
}

/// Defaults (no min_frequency, no max_categories) leave REQ-3/4/5a unchanged.
///
/// Live oracle (sklearn 1.5.2, default — no infrequent params):
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   X=[[2.,0.],[5.,1.],[9.,0.],[5.,1.]]; \
///   e=OneHotEncoder(sparse_output=False).fit(X); print(e.transform(X).tolist())"
///   -> [[1.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0,1.0],
///       [0.0,0.0,1.0,1.0,0.0],[0.0,1.0,0.0,0.0,1.0]]
/// ```
#[test]
fn infrequent_disabled_default_unchanged_vs_sklearn_oracle() {
    let enc = OneHotEncoder::<f64>::new();
    assert_eq!(enc.min_frequency(), None);
    assert_eq!(enc.max_categories(), None);

    let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
    let fitted = enc.fit(&x, &()).unwrap();
    // Every feature has an EMPTY infrequent list (grouping disabled).
    assert_eq!(
        fitted.infrequent_categories(),
        vec![Vec::<f64>::new(), Vec::<f64>::new()]
    );
    let expected: Array2<f64> = array![
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
    ];
    assert_eq!(fitted.transform(&x).unwrap(), expected);
    // Inverse roundtrip unchanged.
    assert_eq!(
        fitted
            .inverse_transform(&fitted.transform(&x).unwrap())
            .unwrap(),
        x
    );
}

/// `max_categories=1` groups EVERY category into the single infrequent column.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "from sklearn.preprocessing import OneHotEncoder; \
///   X=[[1.]]*5+[[2.]]*4+[[3.]]*3; \
///   e=OneHotEncoder(sparse_output=False, max_categories=1).fit(X); \
///   print([c.tolist() for c in e.infrequent_categories_]); \
///   print(e.transform([[1.],[2.],[3.]]).tolist()); \
///   print(list(e.get_feature_names_out()))"
///   -> [[1.0, 2.0, 3.0]]
///      [[1.0],[1.0],[1.0]]
///      ['x0_infrequent_sklearn']
/// ```
#[test]
fn infrequent_max_categories_one_all_infrequent_vs_sklearn_oracle() {
    let x = col_from_counts(&[(1.0, 5), (2.0, 4), (3.0, 3)]);
    let enc = OneHotEncoder::<f64>::new().with_max_categories(1);
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(fitted.infrequent_categories(), vec![vec![1.0, 2.0, 3.0]]);
    assert_eq!(fitted.n_output_features(), 1);

    let probe = array![[1.0_f64], [2.0], [3.0]];
    let expected: Array2<f64> = array![[1.0], [1.0], [1.0]];
    assert_eq!(fitted.transform(&probe).unwrap(), expected);
    assert_eq!(
        fitted.get_feature_names_out(),
        vec!["x0_infrequent_sklearn"]
    );
}
