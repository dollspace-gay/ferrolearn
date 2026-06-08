//! Divergence audit (REQ-7 #1162): `ferrolearn_preprocess::OrdinalEncoder`
//! explicit `categories` param with an EMPTY per-feature list (`categories=[[]]`)
//! vs scikit-learn 1.5.2 `sklearn/preprocessing/_encoders.py::OrdinalEncoder`.
//!
//! EVERY expected value below is grounded in a LIVE sklearn 1.5.2 oracle call
//! (run from /tmp) — NEVER copied from the ferrolearn side (R-CHAR-3).
//!
//! LIVE oracle session (sklearn 1.5.2, run from /tmp):
//! ```text
//! >>> from sklearn.preprocessing import OrdinalEncoder
//! >>> OrdinalEncoder(categories=[[]], handle_unknown='use_encoded_value',
//! ...                unknown_value=-1).fit([['a'],['b']])
//!     -> IndexError: index 0 is out of bounds for axis 0 with size 0
//! >>> OrdinalEncoder(categories=[[]]).fit([['a'],['b']])
//!     -> IndexError: index 0 is out of bounds for axis 0 with size 0
//! ```
//!
//! sklearn raises on an empty explicit category list in BOTH `handle_unknown`
//! modes: `_BaseEncoder._fit` does `cats = np.array(self.categories[i], ...)`
//! (`_encoders.py:114`) then `isinstance(cats[0], bytes)` (`:117`) — indexing
//! `cats[0]` on a 0-length array raises `IndexError` BEFORE the duplicate /
//! subset checks ever run. A feature with zero predefined categories is therefore
//! never a valid fit.
//!
//! ferrolearn DIVERGES: its explicit branch builds the per-column index map
//! (`ordinal_encoder.rs:616`), an empty list trivially passes the duplicate
//! detection (no elements), and under `HandleUnknown::UseEncodedValue` the
//! fit-time subset check is SKIPPED (`ordinal_encoder.rs:634`), so `fit` returns
//! `Ok` with `categories_ == [[]]` where sklearn errors. The default
//! `Error`-mode case is masked (the subset check happens to reject the data), but
//! the `use_encoded_value` case is a clean false-accept.

use ferrolearn_core::traits::Fit;
use ferrolearn_preprocess::{HandleUnknown, OrdinalEncoder};
use ndarray::Array2;

fn make_1col(vals: &[&str]) -> Array2<String> {
    Array2::from_shape_vec(
        (vals.len(), 1),
        vals.iter().map(std::string::ToString::to_string).collect(),
    )
    .unwrap()
}

/// Divergence: `OrdinalEncoder::with_categories(vec![vec![]])` (an empty explicit
/// list for the single feature) under `HandleUnknown::UseEncodedValue` diverges
/// from `sklearn/preprocessing/_encoders.py:114-117`
/// (`isinstance(cats[0], bytes)` on a 0-length array).
///
/// Input: `categories=[[]]`, `handle_unknown='use_encoded_value'`,
/// `unknown_value=-1`, data `[['a'],['b']]`.
/// sklearn (live oracle): raises `IndexError: index 0 is out of bounds for axis 0
/// with size 0` — fit FAILS.
/// ferrolearn: `fit` returns `Ok` (empty list passes the duplicate check; the
/// subset check is skipped under use_encoded_value) — fit SUCCEEDS.
///
/// Tracking: #2229
#[test]
fn divergence_empty_explicit_list_use_encoded_value_fit_must_err() {
    let enc = OrdinalEncoder::new()
        .with_categories(vec![vec![]]) // empty list for the single feature
        .with_handle_unknown(HandleUnknown::UseEncodedValue)
        .with_unknown_value(-1.0);
    let x = make_1col(&["a", "b"]);

    // sklearn raises IndexError here (see live-oracle header). ferrolearn must
    // therefore Err to match; it currently returns Ok, so this assertion FAILS.
    assert!(
        enc.fit(&x, &()).is_err(),
        "sklearn raises IndexError on categories=[[]] (an empty predefined \
         category list); ferrolearn must Err but currently fits Ok"
    );
}
