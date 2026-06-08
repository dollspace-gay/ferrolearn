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
use ferrolearn_preprocess::{OneHotEncoder, OneHotHandleUnknown};
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
