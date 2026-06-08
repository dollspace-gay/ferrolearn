//! ACToR divergence pin for `ferrolearn-preprocess`'s `OneHotEncoder` NaN-column
//! handling vs scikit-learn 1.5.2 `OneHotEncoder(sparse_output=False)`.
//!
//! sklearn's `_unique` (`sklearn/utils/_encode.py:50-78`, `_unique_np`) treats
//! `np.nan` as a single category sorted LAST: after `np.unique`, lines 70-74
//! collapse any run of duplicate trailing NaNs to ONE (`nan_idx =
//! np.searchsorted(uniques, np.nan); uniques = uniques[: nan_idx + 1]`). So a
//! column with NaN appears in `categories_` exactly once, in last position, and
//! `transform` one-hots the NaN rows into that last block column.
//!
//! ferrolearn's `Fit::fit` (`ferrolearn-preprocess/src/one_hot_encoder.rs:201-207`)
//! sorts with `partial_cmp(...).unwrap_or(Ordering::Equal)` (NaN compares Equal,
//! so it is NOT moved to the end) and deduplicates with `Vec::dedup` (exact `==`,
//! and `NaN != NaN`, so NaN NEVER deduplicates). The result is a SILENTLY CORRUPT
//! `categories_` that violates the sorted-unique-set contract (R-DEV-3): NaN
//! appears MULTIPLE times and is NOT last. This is not "well-defined NaN scope" —
//! it is observable corruption of a fitted attribute that sklearn fills cleanly.
//!
//! R-CHAR-3: the expected values below come from the LIVE sklearn 1.5.2 oracle,
//! run from /tmp, quoted inline — NEVER copied from the ferrolearn side.
//!
//! Live oracle (sklearn 1.5.2):
//! ```text
//! python3 -c "from sklearn.preprocessing import OneHotEncoder; \
//!   X=[[1.0],[float('nan')],[2.0],[float('nan')]]; \
//!   e=OneHotEncoder(sparse_output=False).fit(X); \
//!   print([c.tolist() for c in e.categories_]); print(e.transform(X).tolist())"
//!   -> [[1.0, 2.0, nan]]
//!      [[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
//! ```
//!
//! sklearn: `categories_ == [[1.0, 2.0, nan]]` (NaN ONCE, sorted last; 3 cols).
//! ferrolearn (observed): `categories_ == [[1.0, NaN, 2.0, NaN]]` (NaN TWICE,
//! mid-list, 4 cols) and `transform` then ERRORS on the NaN rows it just fitted.
//!
//! Tracking: #2223

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::OneHotEncoder;
use ndarray::array;

/// Divergence: `FittedOneHotEncoder::categories_` for a NaN-containing column
/// diverges from sklearn's `_unique` (`sklearn/utils/_encode.py:50-78`).
///
/// sklearn collapses duplicate NaNs to a single trailing NaN, yielding the
/// sorted-unique set `[1.0, 2.0, nan]` (3 categories). ferrolearn's
/// `partial_cmp.unwrap_or(Equal)` + exact-`==` `dedup` instead yields
/// `[1.0, NaN, 2.0, NaN]` (4 entries, NaN duplicated and not last) — a corrupt
/// `categories_`, not the sorted-unique set.
///
/// Tracking: #2223
#[test]
#[ignore = "divergence: NaN-column categories_ corrupt (NaN duplicated, not sorted-last) vs sklearn _unique single-trailing-nan; tracking #2223"]
fn categories_nan_column_single_trailing_vs_sklearn_oracle() {
    // Live sklearn 1.5.2 oracle (run from /tmp):
    //   categories_ == [[1.0, 2.0, nan]]  -> exactly 3 categories for the column.
    const SKLEARN_N_CATEGORIES: usize = 3;

    let enc = OneHotEncoder::<f64>::new();
    let nan = f64::NAN;
    let x = array![[1.0_f64], [nan], [2.0], [nan]];
    let fitted = enc.fit(&x, &()).unwrap();
    let cats = &fitted.categories()[0];

    // sklearn: the NaN-containing column has exactly 3 sorted-unique categories
    // (`[1.0, 2.0, nan]`). ferrolearn produces 4 (`[1.0, NaN, 2.0, NaN]`), because
    // exact-eq dedup leaves both NaNs.
    assert_eq!(
        cats.len(),
        SKLEARN_N_CATEGORIES,
        "categories_[0] must have sklearn's 3 sorted-unique categories \
         [1.0, 2.0, nan] (one trailing NaN), got {cats:?}"
    );

    // sklearn: NaN, if present, is the LAST category (and appears exactly once).
    // Verify the two finite, non-NaN categories are 1.0 and 2.0 in sorted order,
    // and the single NaN is in last position.
    assert_eq!(cats[0], 1.0, "first sorted-unique category must be 1.0");
    assert_eq!(cats[1], 2.0, "second sorted-unique category must be 2.0");
    assert!(
        cats[2].is_nan(),
        "the single NaN must be the LAST category (sklearn _unique trailing-nan), got {cats:?}"
    );

    // And exactly one NaN overall (sklearn collapses duplicates).
    let nan_count = cats.iter().filter(|v| v.is_nan()).count();
    assert_eq!(
        nan_count, 1,
        "NaN must appear exactly once (sklearn _unique collapses duplicate NaNs), got {cats:?}"
    );
}

/// Divergence: `transform` on the SAME NaN-containing data it was fitted on
/// cannot round-trip. sklearn one-hots the NaN rows into the trailing NaN block
/// column; ferrolearn raises `InvalidParameter` ("Found unknown categories
/// [NaN] …") on a value that is in its OWN learned `categories_`.
///
/// Live oracle: `transform(X) == [[1,0,0],[0,0,1],[0,1,0],[0,0,1]]` — succeeds.
///
/// Tracking: #2223
#[test]
#[ignore = "divergence: transform errors on NaN row present in fitted categories_; sklearn one-hots it; tracking #2223"]
fn transform_nan_row_one_hots_vs_sklearn_oracle() {
    // Live sklearn 1.5.2 oracle (run from /tmp): transform succeeds and the NaN
    // rows (rows 1 and 3) are one-hot in the LAST (NaN) column.
    let enc = OneHotEncoder::<f64>::new();
    let nan = f64::NAN;
    let x = array![[1.0_f64], [nan], [2.0], [nan]];
    let fitted = enc.fit(&x, &()).unwrap();

    let out = fitted
        .transform(&x)
        .expect("sklearn transforms NaN rows (NaN is a fitted category); must not error");

    // sklearn: 3 output columns, NaN rows one-hot in the last column.
    assert_eq!(out.shape(), &[4, 3], "shape vs sklearn oracle [4,3]");
    // Row 1 (NaN) and row 3 (NaN) -> [0,0,1] (the trailing NaN block column).
    assert_eq!(out.row(1).to_vec(), vec![0.0, 0.0, 1.0], "NaN row 1 vs sklearn");
    assert_eq!(out.row(3).to_vec(), vec![0.0, 0.0, 1.0], "NaN row 3 vs sklearn");
}
