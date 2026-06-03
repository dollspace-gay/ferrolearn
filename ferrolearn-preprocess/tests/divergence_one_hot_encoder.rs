//! ACToR critic: oracle-grounded audit of `ferrolearn-preprocess`'s
//! `OneHotEncoder` / `FittedOneHotEncoder`
//! (`ferrolearn-preprocess/src/one_hot_encoder.rs`) against scikit-learn 1.5.2
//! `sklearn/preprocessing/_encoders.py` `class OneHotEncoder(_BaseEncoder)`
//! (`:458`).
//!
//! ferrolearn ships a SIMPLIFIED, integer-only, DENSE encoder: input
//! `Array2<usize>`; `fit` learns `n_categories[j] = max(col_j) + 1` (assumes the
//! categories are the contiguous integers `0..max`); `transform` emits a dense
//! `Array2<F>` one-hot with a per-column offset layout. It matches sklearn's
//! `sparse_output=False` dense output ONLY in the CONTIGUOUS-`0..max` integer
//! regime, where `max(col)+1 == len(_unique(col))`.
//!
//! This file pins the SHIPPED, in-regime REQ-1 path with GREEN guards. The
//! structural NOT-STARTED gaps (`categories_=_unique` vs `max+1`, REQ-3;
//! sparse-by-default, REQ-2; strings/drop/handle_unknown/infrequent/inverse/
//! feature-names, REQ-4..REQ-8) are NOT pinned here (R-DEFER-3: a committed
//! failing test must be minimally fixable this iteration; these are structural
//! blockers filed by the orchestrator without committed failing tests).
//!
//! R-CHAR-3: EVERY expected value below is the output of a LIVE sklearn 1.5.2
//! call, run from `/tmp`, quoted inline — NEVER literal-copied from the
//! ferrolearn side.
//!
//! Live oracle (sklearn 1.5.2 == `python3 -c "import sklearn; sklearn.__version__"`):
//! ```text
//! # A — single contiguous 0..max column, dense:
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   print(OneHotEncoder(sparse_output=False).fit_transform([[0],[1],[2],[1]]).tolist())"
//!   -> [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0]]
//!
//! # B — multi-column contiguous (col0: cats {0,1,2}, col1: cats {0,1}), dense:
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   print(OneHotEncoder(sparse_output=False).fit_transform([[0,0],[1,1],[2,0]]).tolist())"
//!   -> [[1.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0,1.0],[0.0,0.0,1.0,1.0,0.0]]
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   e=OneHotEncoder(sparse_output=False).fit([[0,0],[1,1],[2,0]]); \
//!   print([c.tolist() for c in e.categories_])"
//!   -> [[0, 1, 2], [0, 1]]      (left-to-right concatenated, sorted per feature)
//!
//! # C — out-of-range at transform under handle_unknown='error' (DEFAULT) raises:
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   e=OneHotEncoder(sparse_output=False).fit([[0],[1]]); e.transform([[2]])"
//!   -> ValueError: Found unknown categories [np.int64(2)] in column 0 during transform
//!
//! # D — 0-row fit raises (minimum 1 sample required):
//! python3 -c "import numpy as np; from sklearn.preprocessing import OneHotEncoder; \
//!   OneHotEncoder().fit(np.empty((0,1)))"
//!   -> ValueError: Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.
//! ```

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::OneHotEncoder;
use ndarray::{Array2, array};

// ---------------------------------------------------------------------------
// Task 1 — REQ-1 (SHIPPED, scoped) GREEN guards: dense one-hot of contiguous
// `0..max` integer columns matches the LIVE sklearn `sparse_output=False`
// oracle bit-for-bit (R-CHAR-3: expected values are the oracle outputs above).
// ---------------------------------------------------------------------------

/// Guard (Oracle A): single contiguous column `[[0],[1],[2],[1]]`.
/// Live sklearn `OneHotEncoder(sparse_output=False).fit_transform([[0],[1],[2],[1]])`
/// == `[[1,0,0],[0,1,0],[0,0,1],[0,1,0]]`. ferrolearn
/// `OneHotEncoder::<f64>::new().fit(&x).transform(&x)` must match.
#[test]
fn guard_single_contiguous_column_matches_sklearn_oracle() {
    // Live oracle A (sklearn 1.5.2, run from /tmp).
    let sklearn_expected: Array2<f64> = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ];

    let enc = OneHotEncoder::<f64>::new();
    let x = array![[0usize], [1], [2], [1]];
    let fitted = enc.fit(&x, &()).unwrap();
    let out = fitted.transform(&x).unwrap();

    assert_eq!(
        out.shape(),
        sklearn_expected.shape(),
        "shape vs sklearn oracle"
    );
    assert_eq!(out, sklearn_expected, "dense one-hot vs sklearn oracle A");
}

/// Guard (Oracle B): multi-column contiguous `[[0,0],[1,1],[2,0]]`.
/// col0 cats {0,1,2} → 3 columns, col1 cats {0,1} → 2 columns, concatenated
/// left-to-right (column-offset layout `[3,2]` → 5 cols). Live sklearn
/// `OneHotEncoder(sparse_output=False).fit_transform([[0,0],[1,1],[2,0]])`
/// == `[[1,0,0,1,0],[0,1,0,0,1],[0,0,1,1,0]]`.
#[test]
fn guard_multi_column_contiguous_matches_sklearn_oracle() {
    // Live oracle B (sklearn 1.5.2, run from /tmp).
    let sklearn_expected: Array2<f64> = array![
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
    ];

    let enc = OneHotEncoder::<f64>::new();
    let x = array![[0usize, 0], [1, 1], [2, 0]];
    let fitted = enc.fit(&x, &()).unwrap();

    // sklearn categories_ == [[0,1,2],[0,1]] → block widths 3 then 2 (oracle B-cats).
    assert_eq!(
        fitted.n_categories(),
        &[3, 2],
        "per-feature block widths vs sklearn categories_ lengths"
    );
    assert_eq!(
        fitted.n_output_features(),
        5,
        "total output cols vs sklearn"
    );

    let out = fitted.transform(&x).unwrap();
    assert_eq!(
        out.shape(),
        sklearn_expected.shape(),
        "shape vs sklearn oracle"
    );
    assert_eq!(out, sklearn_expected, "dense one-hot vs sklearn oracle B");
}

/// Guard: `fit_transform` equals `fit` then `transform` on the contiguous
/// multi-column fixture, and both equal the live sklearn oracle B (the
/// `fit_transform` path must not diverge from the split path).
#[test]
fn guard_fit_transform_equals_fit_then_transform_vs_oracle() {
    // Live oracle B (sklearn 1.5.2, run from /tmp).
    let sklearn_expected: Array2<f64> = array![
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
    ];

    let enc = OneHotEncoder::<f64>::new();
    let x = array![[0usize, 0], [1, 1], [2, 0]];

    let via_fit_transform: Array2<f64> = enc.fit_transform(&x).unwrap();
    let via_split = enc.fit(&x, &()).unwrap().transform(&x).unwrap();

    assert_eq!(via_fit_transform, via_split, "fit_transform vs split path");
    assert_eq!(
        via_fit_transform, sklearn_expected,
        "fit_transform vs sklearn oracle B"
    );
}

// ---------------------------------------------------------------------------
// Task 2 — column-offset / category-ordering matches sklearn in-regime.
// sklearn orders output columns by `categories_` (sorted) per feature,
// concatenated left-to-right. For contiguous `0..max` integers, category k →
// column k within the feature block. The Oracle-B fixture above already pins
// the multi-column column order; here we pin the WITHIN-block ordering with an
// asymmetric per-row category so a transposed/misordered block layout would
// fail. Expected value is the live sklearn oracle below.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//     print(OneHotEncoder(sparse_output=False).fit_transform([[2,1],[0,0],[1,1]]).tolist())"
//   -> [[0.0,0.0,1.0,0.0,1.0],[1.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0,1.0]]
//   python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//     e=OneHotEncoder(sparse_output=False).fit([[2,1],[0,0],[1,1]]); \
//     print([c.tolist() for c in e.categories_])"
//   -> [[0, 1, 2], [0, 1]]
// ---------------------------------------------------------------------------

/// Guard: within-block category ordering and left-to-right block concatenation
/// match sklearn's `categories_`-sorted column order. Live sklearn
/// `OneHotEncoder(sparse_output=False).fit_transform([[2,1],[0,0],[1,1]])`
/// == `[[0,0,1,0,1],[1,0,0,1,0],[0,1,0,0,1]]` (col0 block = cols 0..3 for
/// cats {0,1,2}; col1 block = cols 3..5 for cats {0,1}).
#[test]
fn guard_within_block_column_order_matches_sklearn_oracle() {
    // Live oracle (sklearn 1.5.2, run from /tmp).
    let sklearn_expected: Array2<f64> = array![
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
    ];

    let enc = OneHotEncoder::<f64>::new();
    let x = array![[2usize, 1], [0, 0], [1, 1]];
    let out = enc.fit_transform(&x).unwrap();

    assert_eq!(
        out, sklearn_expected,
        "column-offset/ordering vs sklearn oracle"
    );
}

// ---------------------------------------------------------------------------
// Task 3 — in-regime error contracts. These probe for a minimal-fixable bug in
// the EXISTING contiguous-integer path. The live oracle shows sklearn ALSO
// raises in both cases, so these are GREEN guards (both error → NO divergence),
// NOT pinned failing tests. (The NON-contiguous `[[0],[2]]` case is structural,
// REQ-3, and is deliberately NOT probed here.)
// ---------------------------------------------------------------------------

/// Guard (Oracle C): out-of-range category at transform. After a CONTIGUOUS fit
/// `[[0],[1]]` (cats {0,1}), transforming `[[2]]` is unseen. Live sklearn under
/// `handle_unknown='error'` (DEFAULT) raises `ValueError`; ferrolearn also
/// errors (`2 >= n_cats 2`). Both error → GREEN guard, NOT a divergence.
#[test]
fn guard_out_of_range_both_error_vs_sklearn_oracle() {
    let enc = OneHotEncoder::<f64>::new();
    let x_train = array![[0usize], [1]];
    let fitted = enc.fit(&x_train, &()).unwrap();
    let x_bad = array![[2usize]];
    // sklearn (oracle C): ValueError "Found unknown categories [2] ... during transform".
    assert!(
        fitted.transform(&x_bad).is_err(),
        "out-of-range category must error, matching sklearn handle_unknown='error'"
    );
}

/// Guard (Oracle D): 0-row fit. Live sklearn `OneHotEncoder().fit(np.empty((0,1)))`
/// raises `ValueError` ("Found array with 0 sample(s) ... a minimum of 1 is
/// required"); ferrolearn `fit` rejects 0 rows (`InsufficientSamples`). Both
/// reject → GREEN guard, NOT a divergence.
#[test]
fn guard_zero_row_fit_both_error_vs_sklearn_oracle() {
    let enc = OneHotEncoder::<f64>::new();
    let x_empty: Array2<usize> = Array2::zeros((0, 1));
    // sklearn (oracle D): ValueError on 0 samples; ferrolearn: InsufficientSamples.
    assert!(
        enc.fit(&x_empty, &()).is_err(),
        "0-row fit must error, matching sklearn's minimum-1-sample requirement"
    );
}
