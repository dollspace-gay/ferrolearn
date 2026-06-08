//! Divergence audit: `ferrolearn-preprocess` `OrdinalEncoder::transform` on a
//! ZERO-ROW input vs scikit-learn 1.5.2
//! `sklearn/preprocessing/_encoders.py::class OrdinalEncoder` (`:1235`).
//!
//! Expected value from a LIVE sklearn 1.5.2 oracle (R-CHAR-3), run from /tmp:
//! ```text
//! >>> from sklearn.preprocessing import OrdinalEncoder
//! >>> import numpy as np
//! >>> e = OrdinalEncoder().fit([['a','x'],['b','y']])
//! >>> e.transform(np.empty((0,2), dtype=object))
//! ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum
//!             of 1 is required.
//! ```
//!
//! sklearn's `OrdinalEncoder.transform` (`_encoders.py:1563`) calls
//! `self._transform(...)` (`:1578`), which calls `self._check_X(...)`
//! (`_encoders.py:194`), which calls `check_array(X, ...)`
//! (`_encoders.py:45`). `check_array` enforces a minimum of 1 sample
//! (`sklearn/utils/validation.py:1087`), so a 0-row TRANSFORM raises
//! `ValueError` — exactly as a 0-row FIT does.
//!
//! ferrolearn's `FittedOrdinalEncoder::transform`
//! (`ferrolearn-preprocess/src/ordinal_encoder.rs:217`) only checks the column
//! count (`x.ncols() != n_features`, `:219`); it has NO row-count guard. On a
//! 0-row input the `for i in 0..n_samples` loop body never runs, so it returns
//! `Ok(Array2::zeros((0, n_features)))` instead of an error.
//!
//! Divergence: sklearn raises (Err); ferrolearn returns Ok(empty matrix).
//! Tracking: #2220.

use ferrolearn_core::traits::Fit;
use ferrolearn_core::traits::Transform;
use ferrolearn_preprocess::OrdinalEncoder;
use ndarray::Array2;

/// Build an `n x 2` string array from `(col0, col1)` row tuples.
fn make_2col(rows: &[(&str, &str)]) -> Array2<String> {
    let flat: Vec<String> = rows
        .iter()
        .flat_map(|(a, b)| [a.to_string(), b.to_string()])
        .collect();
    Array2::from_shape_vec((rows.len(), 2), flat).unwrap()
}

/// Divergence: `FittedOrdinalEncoder::transform` accepts a 0-row input and
/// returns `Ok(empty)`, whereas sklearn 1.5.2 `OrdinalEncoder.transform`
/// (`sklearn/preprocessing/_encoders.py:1563` -> `_transform` `:194` ->
/// `_check_X` `:45` -> `check_array` `sklearn/utils/validation.py:1087`)
/// raises `ValueError: Found array with 0 sample(s) ... a minimum of 1 is
/// required.`
///
/// LIVE oracle (sklearn 1.5.2, run from /tmp):
///   `OrdinalEncoder().fit([['a','x'],['b','y']]).transform(`
///   `np.empty((0,2),dtype=object))` -> ValueError (0 sample(s), min 1).
///
/// sklearn returns: Err (ValueError). ferrolearn returns: Ok(shape (0,2)).
/// Tracking: #2220.
#[test]
fn divergence_zero_row_transform_should_error() {
    let enc = OrdinalEncoder::new();
    let x_train = make_2col(&[("a", "x"), ("b", "y")]);
    let fitted = enc.fit(&x_train, &()).unwrap();

    // 0-row, correct column count input.
    let x_empty: Array2<String> = Array2::from_shape_vec((0, 2), vec![]).unwrap();
    let result = fitted.transform(&x_empty);

    // sklearn's check_array rejects 0-sample input at transform time, mirroring
    // its 0-sample FIT rejection. ferrolearn currently returns Ok(empty matrix).
    assert!(
        result.is_err(),
        "sklearn 1.5.2 raises ValueError (min 1 sample) on a 0-row transform; \
         ferrolearn returned Ok = {:?}",
        result.map(|a| a.dim())
    );
}
