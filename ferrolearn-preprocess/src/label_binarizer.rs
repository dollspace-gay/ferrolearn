//! One-vs-rest label binarizer.
//!
//! Transforms a vector of integer class labels into a binary indicator matrix.
//! For *K* classes the output has *K* columns (one-hot rows), except in the
//! binary case (*K* = 2) and single-class case (*K* = 1) where a single column
//! is produced.
//!
//! Translation target: scikit-learn 1.5.2 `class LabelBinarizer`
//! (`sklearn/preprocessing/_label.py:180`) + `label_binarize` (`:430`). Design:
//! `.design/preprocess/label_binarizer.md`. Tracking: #1238.
//!
//! `## REQ status`
//!
//! | REQ | Status | Anchor |
//! |---|---|---|
//! | REQ-1 fit → sorted-unique classes_ (usize) | SHIPPED | `LabelBinarizer::fit`; sklearn `_label.py:306` |
//! | REQ-2 transform multiclass (k≥3) one-hot values | SHIPPED | `FittedLabelBinarizer::transform` else-branch; sklearn `_label.py:552-577` |
//! | REQ-3 transform binary (k=2) single col, pos_label on 2nd class | SHIPPED | `transform` k==2 branch; sklearn `_label.py:531`,`:592-596` |
//! | REQ-4 transform unknown-label: ignore (all-neg_label row) | SHIPPED (#1239) | `transform` `if let Some(&idx) = class_to_idx.get`; sklearn `_label.py:556-559` |
//! | REQ-5 transform single-class (k=1) → all-neg_label column | SHIPPED (#1240) | `transform` k==1 arm `Array2::from_elem`; sklearn `_label.py:532-538` |
//! | REQ-6 inverse_transform binary STRICT threshold (`> (pos+neg)/2`) | SHIPPED (#1241) | `inverse_transform` k==2 branch; sklearn `_label.py:667` |
//! | REQ-7 inverse_transform multiclass argmax | SHIPPED | `inverse_transform` else-branch; sklearn `_label.py:641` |
//! | REQ-8 neg_label/pos_label ctor params + validation | SHIPPED (#1242) | `LabelBinarizer::with_neg_label`/`with_pos_label` + `Fit::fit` validation; `transform` neg/pos base+active; `inverse_transform` `(pos+neg)/2` threshold; consumer crate re-export `lib.rs`; sklearn `_label.py:263`,`:283-287`,`:579-583`,`:667` |
//! | REQ-9 sparse_output CSR + constraint | NOT-STARTED (#1243) | sklearn `_label.py:563`,`:584-585`,`:289-294` |
//! | REQ-10 `label_binarize` free function | SHIPPED (#1244) | `pub fn label_binarize` (this file): `neg<pos` validation (verbatim msg, sklearn `_label.py:499-504`); GIVEN-order columns (sklearn's "preserve label ordering" reorder `:587-590`, so `label_binarize([0,2,1],classes=[2,0,1])` → `[[0,1,0],[1,0,0],[0,0,1]]`); k==1 all-neg col (`:532-538`); single-col collapse gated on `type_of_target(y)=="binary"` AND `len(classes)==2` (NOT `len(classes)` alone — `:519`,`:531`,`:592-596`; "binary" = ≤2 distinct values for 1D int y, verified live, #2233), giving pos where `y==classes[1]` (the kept `Y[:,-1]` after reorder, `:596`); k==2 with multiclass y (3+ distinct) emits 2 cols, no collapse; k>2 one-hot in given order (`:552-577`); unseen label → all-neg row (`:556-559`). Consumer: crate re-export `lib.rs` (`pub use label_binarizer::label_binarize`). Live-oracle parity: `tests/divergence_label_binarizer.rs` (basic/neg-pos/binary/`[2,0,1]`-ordering/unseen/neg≥pos-err/==estimator). |
//! | REQ-11 arbitrary label types + type_of_target/multilabel input | NOT-STARTED (#1245) | sklearn `_label.py:296`,`:543-550` (usize-only, R-DEV-3) |
//! | REQ-12 PyO3 binding | NOT-STARTED (#1246) | `ferrolearn-python/src/` (absent) |
//!
//! # Examples
//!
//! ```
//! use ferrolearn_preprocess::label_binarizer::LabelBinarizer;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let lb = LabelBinarizer::new();
//! let y = array![0_usize, 1, 2, 1];
//! let fitted = lb.fit(&y, &()).unwrap();
//! let mat = fitted.transform(&y).unwrap();
//! // 3 classes → (4, 3) indicator matrix
//! assert_eq!(mat.shape(), &[4, 3]);
//! assert_eq!(mat[[0, 0]], 1.0);
//! assert_eq!(mat[[0, 1]], 0.0);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// LabelBinarizer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted one-vs-rest label binarizer.
///
/// Calling [`Fit::fit`] on an `Array1<usize>` discovers the sorted set of
/// unique class labels and returns a [`FittedLabelBinarizer`].
///
/// `neg_label` / `pos_label` are the integer values written into the output
/// indicator matrix for absent / present classes, mirroring sklearn's
/// `LabelBinarizer(neg_label=0, pos_label=1)` (`sklearn/preprocessing/_label.py:263`).
/// The defaults `0` / `1` reproduce the canonical 0/1 indicator behavior.
#[derive(Debug, Clone)]
pub struct LabelBinarizer {
    /// Value written for absent classes (sklearn `neg_label`, default `0`).
    neg_label: i64,
    /// Value written for the present class (sklearn `pos_label`, default `1`).
    pos_label: i64,
}

impl Default for LabelBinarizer {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelBinarizer {
    /// Create a new `LabelBinarizer` with the default `neg_label=0`,
    /// `pos_label=1` (the canonical 0/1 indicator encoding).
    #[must_use]
    pub fn new() -> Self {
        Self {
            neg_label: 0,
            pos_label: 1,
        }
    }

    /// Set the `neg_label` (value used for absent classes).
    ///
    /// Mirrors sklearn's `LabelBinarizer(neg_label=...)`
    /// (`sklearn/preprocessing/_label.py:263`). Must be strictly less than
    /// `pos_label`; validated at [`Fit::fit`] time (`_label.py:283-287`).
    #[must_use]
    pub fn with_neg_label(mut self, neg_label: i64) -> Self {
        self.neg_label = neg_label;
        self
    }

    /// Set the `pos_label` (value used for the present class).
    ///
    /// Mirrors sklearn's `LabelBinarizer(pos_label=...)`
    /// (`sklearn/preprocessing/_label.py:263`). Must be strictly greater than
    /// `neg_label`; validated at [`Fit::fit`] time (`_label.py:283-287`).
    #[must_use]
    pub fn with_pos_label(mut self, pos_label: i64) -> Self {
        self.pos_label = pos_label;
        self
    }

    /// Return the configured `neg_label`.
    #[must_use]
    pub fn neg_label(&self) -> i64 {
        self.neg_label
    }

    /// Return the configured `pos_label`.
    #[must_use]
    pub fn pos_label(&self) -> i64 {
        self.pos_label
    }
}

// ---------------------------------------------------------------------------
// FittedLabelBinarizer
// ---------------------------------------------------------------------------

/// A fitted label binarizer holding the discovered class set.
///
/// Created by calling [`Fit::fit`] on a [`LabelBinarizer`].
#[derive(Debug, Clone)]
pub struct FittedLabelBinarizer {
    /// Sorted unique class labels observed during fitting.
    classes: Vec<usize>,
    /// Value written for absent classes (sklearn `neg_label`, default `0`).
    neg_label: i64,
    /// Value written for the present class (sklearn `pos_label`, default `1`).
    pos_label: i64,
}

impl FittedLabelBinarizer {
    /// Return the sorted class labels discovered during fitting.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Return the number of unique classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Return the configured `neg_label` (value used for absent classes).
    #[must_use]
    pub fn neg_label(&self) -> i64 {
        self.neg_label
    }

    /// Return the configured `pos_label` (value used for the present class).
    #[must_use]
    pub fn pos_label(&self) -> i64 {
        self.pos_label
    }

    /// Map a binary indicator matrix back to integer class labels.
    ///
    /// For each row the class with the largest value (argmax) is chosen.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does
    /// not match the expected output width (1 for binary, *K* for multiclass).
    pub fn inverse_transform(&self, y: &Array2<f64>) -> Result<Array1<usize>, FerroError> {
        let k = self.classes.len();
        let expected_cols = if k == 2 { 1 } else { k };

        if y.ncols() != expected_cols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![y.nrows(), expected_cols],
                actual: vec![y.nrows(), y.ncols()],
                context: "FittedLabelBinarizer::inverse_transform".into(),
            });
        }

        let n = y.nrows();
        let mut result = Array1::zeros(n);

        if k == 2 {
            // Single column: strict threshold at `(pos_label + neg_label) / 2`,
            // matching sklearn `_inverse_binarize_thresholding` (`_label.py:667`):
            // `y = np.array(y > threshold)` with default `threshold =
            // (pos_label + neg_label) / 2` (`:399-400`). The comparison is STRICT,
            // so an exact-threshold value maps to `classes[0]`. With the default
            // `neg_label=0, pos_label=1` this reduces to `> 0.5`.
            // Cast EACH to f64 BEFORE the add: `i64 + i64` would overflow (and
            // panic in debug, R-CODE-2) for large-but-valid neg/pos like 2^62
            // (#2232). sklearn computes this in arbitrary-precision then /2.0.
            let threshold = (self.pos_label as f64 + self.neg_label as f64) / 2.0;
            for i in 0..n {
                result[i] = if y[[i, 0]] > threshold {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
            }
        } else {
            // Multiclass: argmax per row
            for i in 0..n {
                let row = y.row(i);
                let mut best_j = 0;
                let mut best_v = f64::NEG_INFINITY;
                for (j, &v) in row.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best_j = j;
                    }
                }
                result[i] = self.classes[best_j];
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array1<usize>, ()> for LabelBinarizer {
    type Fitted = FittedLabelBinarizer;
    type Error = FerroError;

    /// Fit the binarizer by discovering unique class labels.
    ///
    /// # Errors
    ///
    /// - Returns [`FerroError::InvalidParameter`] if `neg_label >= pos_label`,
    ///   mirroring sklearn's `neg_label={0} must be strictly less than
    ///   pos_label={1}.` raise (`sklearn/preprocessing/_label.py:283-287`).
    /// - Returns [`FerroError::InsufficientSamples`] if the input is empty.
    fn fit(&self, y: &Array1<usize>, _target: &()) -> Result<FittedLabelBinarizer, FerroError> {
        // Validate neg_label < pos_label BEFORE class discovery, mirroring
        // sklearn `fit` (`_label.py:283-287`): the message is verbatim
        // `neg_label={neg} must be strictly less than pos_label={pos}.`.
        if self.neg_label >= self.pos_label {
            return Err(FerroError::InvalidParameter {
                name: "neg_label".into(),
                reason: format!(
                    "neg_label={} must be strictly less than pos_label={}.",
                    self.neg_label, self.pos_label
                ),
            });
        }

        if y.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "LabelBinarizer::fit".into(),
            });
        }

        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();

        Ok(FittedLabelBinarizer {
            classes,
            neg_label: self.neg_label,
            pos_label: self.pos_label,
        })
    }
}

impl Transform<Array1<usize>> for FittedLabelBinarizer {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Transform labels into a binary indicator matrix.
    ///
    /// - For *K* = 2 classes the output shape is `(n, 1)`.
    /// - For *K* > 2 classes the output shape is `(n, K)`.
    ///
    /// Absent classes are written as `neg_label` and the present class as
    /// `pos_label` (defaults `0` / `1`). Labels not seen during fitting are
    /// silently ignored: their row is left at the `neg_label` base value, with
    /// no error and no warning. This mirrors scikit-learn's `label_binarize`
    /// (`sklearn/preprocessing/_label.py:556-559`), which selects only the
    /// known labels (`y_in_classes = np.isin(y, classes)`) and leaves unseen
    /// labels contributing nothing, then fills the dense base with `neg_label`
    /// (`:579-583`).
    fn transform(&self, y: &Array1<usize>) -> Result<Array2<f64>, FerroError> {
        let k = self.classes.len();
        let n = y.len();

        // sklearn `LabelBinarizer.transform` delegates to `label_binarize`, which
        // gates the binary single-column collapse on `type_of_target(y)=="binary"`
        // (`_label.py:519`,`:531`) — computed on the TRANSFORM input `y`, NOT the
        // fitted class count (#2234). For 1D integer `y`, "binary" means at most 2
        // distinct values; a MULTICLASS transform input (3+ distinct) with 2
        // fitted classes therefore emits the (n, 2) multi-column form, not the
        // single column (e.g. fit([0,1]).transform([0,1,2]) -> [[1,0],[0,1],[0,0]]).
        let y_is_binary = {
            let mut distinct: Vec<usize> = y.iter().copied().collect();
            distinct.sort_unstable();
            distinct.dedup();
            distinct.len() <= 2
        };

        // The base ("absent") value is `neg_label`; the active ("present")
        // value is `pos_label`, mirroring sklearn `label_binarize`'s dense fill
        // `Y[Y == 0] = neg_label` (`_label.py:579-583`) and the `pos_label`
        // active positions (`:562`, `:599`).
        let neg = self.neg_label as f64;
        let pos = self.pos_label as f64;

        // Build a lookup: class_value → column index
        let class_to_idx: std::collections::HashMap<usize, usize> = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        if k == 1 {
            // Single class (n_classes == 1): sklearn treats this as the binary
            // degenerate case and returns an all-`neg_label` single column,
            // never `pos_label` (`sklearn/preprocessing/_label.py:532-538`:
            // `Y = np.zeros((len(y), 1)); Y += neg_label`).
            Ok(Array2::from_elem((n, 1), neg))
        } else if k == 2 && y_is_binary {
            // Binary: single column, `pos_label` for the second class else
            // `neg_label`. The base is filled with `neg_label` (NOT zeros).
            // Only when the transform input is itself binary (#2234).
            let mut out = Array2::from_elem((n, 1), neg);
            for (i, &label) in y.iter().enumerate() {
                // Unseen labels are silently ignored (row left at `neg_label`),
                // mirroring sklearn `_label.py:556-559`.
                if let Some(&idx) = class_to_idx.get(&label) {
                    out[[i, 0]] = if idx == 1 { pos } else { neg };
                }
            }
            Ok(out)
        } else {
            // Multiclass: one-hot rows — `pos_label` at the class column,
            // `neg_label` everywhere else. The base is filled with `neg_label`.
            let mut out = Array2::from_elem((n, k), neg);
            for (i, &label) in y.iter().enumerate() {
                // Unseen labels are silently ignored (row left all-`neg_label`),
                // mirroring sklearn `_label.py:556-559`.
                if let Some(&idx) = class_to_idx.get(&label) {
                    out[[i, idx]] = pos;
                }
            }
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// `label_binarize` free function (sklearn `label_binarize`, `_label.py:430`)
// ---------------------------------------------------------------------------

/// Binarize integer labels one-vs-all against an EXPLICIT class list — the
/// standalone, estimator-less API mirroring scikit-learn's `label_binarize`
/// free function (`sklearn/preprocessing/_label.py:430`).
///
/// Unlike [`LabelBinarizer`], which discovers its classes by fitting, this
/// function takes the class set as an explicit `classes` argument and encodes
/// `y` against it. The output is a binary indicator matrix written with
/// `pos_label` at active positions and `neg_label` everywhere else (defaults
/// `0` / `1`).
///
/// # Column ordering (the headline)
///
/// The output **columns follow the GIVEN `classes` order**, NOT a sorted order.
/// sklearn builds the indicator in sorted-class order internally
/// (`sorted_class = np.sort(classes)`, `_label.py:542`; columns via
/// `np.searchsorted`, `:558`) but then **reorders the columns back to the given
/// `classes` order** in the "preserve label ordering" step (`:587-590`:
/// `indices = np.searchsorted(sorted_class, classes); Y = Y[:, indices]`).
/// So `label_binarize([0,2,1], classes=[2,0,1])` yields
/// `[[0,1,0],[1,0,0],[0,0,1]]` — column `j` corresponds to `classes[j]`, the
/// *given* class, with `pos_label` where `y[i] == classes[j]`. (Verified live
/// vs sklearn 1.5.2; see `tests/divergence_label_binarizer.rs`.)
///
/// # Shape / collapse rules
///
/// The single-column collapse is gated on `type_of_target(y) == "binary"`, NOT
/// on `len(classes)` (`_label.py:519` `y_type = type_of_target(y)`; `:531`
/// `if y_type == "binary":`; the collapse at `:592-596`). For 1D integer `y`,
/// `type_of_target` is "binary" iff `y` has at most two distinct values, else
/// "multiclass" (verified live vs sklearn 1.5.2). Writing
/// `y_is_binary = (distinct count of y) <= 2`:
/// - `k == 1`: a single all-`neg_label` column (`_label.py:532-538`).
/// - `k == 2` AND `y_is_binary`: a single column — `pos_label` where
///   `y == classes[last]` (the LAST given class), else `neg_label`. sklearn
///   builds both columns then takes `Y[:, -1]` after the reorder
///   (`_label.py:596`), so the kept column is the one for the last *given*
///   class. (When `classes` is sorted — as the fitted estimator always is —
///   `classes[last]` is the second-sorted class, so this coincides with
///   [`FittedLabelBinarizer::transform`]'s `idx == 1` rule.)
/// - `k == 2` but `y` is multiclass (3+ distinct values): NO collapse —
///   `k == 2` columns in given order (`y_type` is not "binary", so the `:592`
///   single-column step is skipped). E.g. `label_binarize([0,1,2], classes=[0,1])`
///   → `(3, 2)` `[[1,0],[0,1],[0,0]]`, with the unseen `2` leaving an all-`neg`
///   row.
/// - `k > 2`: `k` columns in given order, `pos_label` at the value's column.
///
/// A value in `y` not present in `classes` leaves its row all-`neg_label`,
/// mirroring sklearn's `y_in_classes = np.isin(y, classes)` silent ignore
/// (`_label.py:556-559`).
///
/// `classes` is expected to be unique (sklearn: "Uniquely holds the label for
/// each class", `_label.py:447`); duplicate entries are not part of the matched
/// contract.
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] if `neg_label >= pos_label`, with
/// the same verbatim message as [`LabelBinarizer`]'s `fit`
/// (`_label.py:499-504`: `neg_label={neg} must be strictly less than
/// pos_label={pos}.`). Returns [`FerroError::InsufficientSamples`] if `classes`
/// is empty (sklearn cannot binarize against zero classes).
#[must_use = "label_binarize returns a new indicator matrix"]
pub fn label_binarize(
    y: &Array1<usize>,
    classes: &[usize],
    neg_label: i64,
    pos_label: i64,
) -> Result<Array2<f64>, FerroError> {
    // Validate neg_label < pos_label, mirroring sklearn `label_binarize`
    // (`_label.py:499-504`) — the SAME verbatim message as the estimator's
    // `fit` (`LabelBinarizer::fit`).
    if neg_label >= pos_label {
        return Err(FerroError::InvalidParameter {
            name: "neg_label".into(),
            reason: format!(
                "neg_label={neg_label} must be strictly less than pos_label={pos_label}."
            ),
        });
    }

    let k = classes.len();
    if k == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "label_binarize: classes".into(),
        });
    }

    let n = y.len();
    let neg = neg_label as f64;
    let pos = pos_label as f64;

    // Map each given class value to its GIVEN-order column index. The output
    // columns follow the given `classes` order (sklearn's "preserve label
    // ordering" reorder, `_label.py:587-590`), so column `j` belongs to
    // `classes[j]`. For unique `classes` the last write wins identically; the
    // contract assumes uniqueness (`_label.py:447`).
    let class_to_col: std::collections::HashMap<usize, usize> =
        classes.iter().enumerate().map(|(j, &c)| (c, j)).collect();

    // The single-column collapse is gated on `type_of_target(y) == "binary"`,
    // NOT on `len(classes)` (`_label.py:519` `y_type = type_of_target(y)`;
    // `:531` `if y_type == "binary":`; the collapse itself at `:592-596`). For
    // 1D integer `y`, `type_of_target` returns "binary" iff `y` has at most two
    // distinct values, else "multiclass" (verified live vs sklearn 1.5.2:
    // 1-distinct → "binary", 2-distinct → "binary", 3+ distinct → "multiclass").
    // So `[5,5]` (1 distinct) and `[0,1,0]` (2 distinct) are binary, but
    // `[0,1,2]` (3 distinct) is multiclass. When `k == 2` but `y` is multiclass,
    // sklearn promotes to the `n_classes`-column form (`:539-540` only fires for
    // `len(classes) >= 3`; here the non-binary `y_type` simply means the `:592`
    // collapse is skipped), giving a `(n, 2)` indicator.
    let mut distinct: Vec<usize> = y.iter().copied().collect();
    distinct.sort_unstable();
    distinct.dedup();
    let y_is_binary = distinct.len() <= 2;

    if k == 1 {
        // n_classes == 1: all-`neg_label` single column (`_label.py:532-538`:
        // `Y = np.zeros((len(y), 1)); Y += neg_label`). sklearn reaches this only
        // when `y_type == "binary"` too; for plain integer `y` a single class
        // implies `y` has ≤1 distinct value, which is always binary, so this
        // single-column form is unconditional here.
        Ok(Array2::from_elem((n, 1), neg))
    } else if k == 2 && y_is_binary {
        // Binary `y` with exactly two classes: the single column kept after the
        // given-order reorder is `Y[:, -1]` (`_label.py:596`) — the column for
        // the LAST given class. So `pos_label` where `y == classes[1]`, else
        // `neg_label`. Unseen labels (not in `classes`) stay at `neg_label`
        // (`:556-559`).
        let last_class = classes[1];
        let mut out = Array2::from_elem((n, 1), neg);
        for (i, &label) in y.iter().enumerate() {
            if label == last_class {
                out[[i, 0]] = pos;
            }
        }
        Ok(out)
    } else {
        // `k` columns in GIVEN order, `pos_label` at the value's column
        // (`_label.py:552-577` + the `:587-590` reorder to the given order).
        // Reached for genuine multiclass (`k > 2`) AND for `k == 2` with a
        // multiclass `y` (3+ distinct values), where sklearn skips the `:592`
        // single-column collapse and emits the full `(n, k)` indicator. Unseen
        // labels leave the row all-`neg_label` (`:556-559`).
        let mut out = Array2::from_elem((n, k), neg);
        for (i, &label) in y.iter().enumerate() {
            if let Some(&col) = class_to_col.get(&label) {
                out[[i, col]] = pos;
            }
        }
        Ok(out)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fit_discovers_sorted_classes() {
        let lb = LabelBinarizer::new();
        let y = array![2_usize, 0, 1, 2, 0];
        let fitted = lb.fit(&y, &()).unwrap();
        assert_eq!(fitted.classes(), &[0, 1, 2]);
    }

    #[test]
    fn test_fit_empty_input_error() {
        let lb = LabelBinarizer::new();
        let y: Array1<usize> = Array1::zeros(0);
        assert!(lb.fit(&y, &()).is_err());
    }

    #[test]
    fn test_binary_transform_single_column() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 0, 1];
        let fitted = lb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[4, 1]);
        assert_eq!(mat[[0, 0]], 0.0); // class 0 → 0
        assert_eq!(mat[[1, 0]], 1.0); // class 1 → 1
        assert_eq!(mat[[2, 0]], 0.0);
        assert_eq!(mat[[3, 0]], 1.0);
    }

    #[test]
    fn test_multiclass_transform_indicator_matrix() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 2, 1];
        let fitted = lb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[4, 3]);
        // Row 0: class 0 → [1, 0, 0]
        assert_eq!(mat[[0, 0]], 1.0);
        assert_eq!(mat[[0, 1]], 0.0);
        assert_eq!(mat[[0, 2]], 0.0);
        // Row 2: class 2 → [0, 0, 1]
        assert_eq!(mat[[2, 0]], 0.0);
        assert_eq!(mat[[2, 1]], 0.0);
        assert_eq!(mat[[2, 2]], 1.0);
    }

    #[test]
    fn test_inverse_transform_multiclass() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 2, 1];
        let fitted = lb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    #[test]
    fn test_inverse_transform_binary() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 0, 1];
        let fitted = lb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    /// Unseen labels are silently ignored (row left all-neg_label), mirroring
    /// sklearn `label_binarize` (`_label.py:556-559`).
    ///
    /// Live oracle (sklearn 1.5.2, from /tmp):
    ///   `LabelBinarizer().fit([0,1,2]).transform([0,3]).tolist()`
    ///     -> `[[1, 0, 0], [0, 0, 0]]`
    #[test]
    fn test_transform_unknown_label_ignored() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 2];
        let y2 = array![0_usize, 3]; // 3 not in {0,1,2}
        // Fit then transform, propagating any error into the Result we compare.
        let got = lb.fit(&y, &()).and_then(|fitted| fitted.transform(&y2));
        // sklearn-oracle value: [[1,0,0],[0,0,0]] (label 3 ignored, all-zero row).
        // Compare via Option (FerroError is not PartialEq); Ok(_) is required.
        let expected: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        assert_eq!(got.ok(), Some(expected));
    }

    #[test]
    fn test_inverse_transform_shape_mismatch() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 2];
        let fitted = lb.fit(&y, &()).unwrap();
        // 3 classes expects 3 columns, but we give 2
        let bad = Array2::<f64>::zeros((2, 2));
        assert!(fitted.inverse_transform(&bad).is_err());
    }

    /// Single class (n_classes == 1) → an all-zero single column, mirroring
    /// sklearn's binary-degenerate case (`_label.py:532-538`).
    ///
    /// Live oracle (sklearn 1.5.2):
    ///   `LabelBinarizer().fit_transform([5,5,5]).tolist()` -> `[[0],[0],[0]]`
    #[test]
    fn test_single_class() {
        let lb = LabelBinarizer::new();
        let y = array![5_usize, 5, 5];
        // Confirm exactly one class is discovered (degenerate single-class case).
        let n_classes = lb.fit(&y, &()).map(|fitted| fitted.n_classes());
        assert_eq!(n_classes.ok(), Some(1));
        // Fit then transform, propagating any error into the Result we compare.
        let got = lb.fit(&y, &()).and_then(|fitted| fitted.transform(&y));
        // 1 class → 1 column, all zeros (never 1.0); sklearn-oracle [[0],[0],[0]].
        let expected: Array2<f64> = array![[0.0], [0.0], [0.0]];
        assert_eq!(got.ok(), Some(expected));
    }

    #[test]
    fn test_non_contiguous_classes() {
        let lb = LabelBinarizer::new();
        let y = array![10_usize, 20, 30, 10];
        let fitted = lb.fit(&y, &()).unwrap();
        assert_eq!(fitted.classes(), &[10, 20, 30]);
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[4, 3]);
        assert_eq!(mat[[0, 0]], 1.0); // 10 → col 0
        assert_eq!(mat[[1, 1]], 1.0); // 20 → col 1
        assert_eq!(mat[[2, 2]], 1.0); // 30 → col 2
    }

    #[test]
    fn test_roundtrip_multiclass_non_contiguous() {
        let lb = LabelBinarizer::new();
        let y = array![10_usize, 20, 30, 20];
        let fitted = lb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    // -- REQ-8: neg_label / pos_label ctor params + validation ----------------

    /// REQ-8: builders + getters carry the configured neg/pos through fit.
    /// Defaults preserve the canonical 0/1 encoding.
    #[test]
    fn test_neg_pos_label_builders_and_getters() {
        let lb = LabelBinarizer::new();
        assert_eq!(lb.neg_label(), 0);
        assert_eq!(lb.pos_label(), 1);

        let lb = LabelBinarizer::new().with_neg_label(-1).with_pos_label(2);
        assert_eq!(lb.neg_label(), -1);
        assert_eq!(lb.pos_label(), 2);

        let fitted = lb.fit(&array![0_usize, 1, 2], &()).unwrap();
        assert_eq!(fitted.neg_label(), -1);
        assert_eq!(fitted.pos_label(), 2);
    }

    /// REQ-8: multiclass transform with neg_label=-1, pos_label=2.
    /// Live oracle (sklearn 1.5.2, from /tmp):
    ///   `LabelBinarizer(neg_label=-1,pos_label=2).fit([0,1,2]).transform([0,2]).tolist()`
    ///     -> `[[2,-1,-1],[-1,-1,2]]` (present->2, absent->-1; base is -1, not 0)
    #[test]
    fn test_neg_pos_multiclass_transform() {
        let lb = LabelBinarizer::new().with_neg_label(-1).with_pos_label(2);
        let fitted = lb.fit(&array![0_usize, 1, 2], &()).unwrap();
        let got = fitted.transform(&array![0_usize, 2]).unwrap();
        let expected: Array2<f64> = array![[2.0, -1.0, -1.0], [-1.0, -1.0, 2.0]];
        assert_eq!(got, expected);
    }

    /// REQ-8: binary (k==2) transform with neg_label=-1, pos_label=1.
    /// Live oracle (sklearn 1.5.2, from /tmp):
    ///   `LabelBinarizer(neg_label=-1,pos_label=1).fit([0,1]).transform([0,1,0]).tolist()`
    ///     -> `[[-1],[1],[-1]]` (2nd class -> pos_label, else neg_label)
    #[test]
    fn test_neg_pos_binary_transform() {
        let lb = LabelBinarizer::new().with_neg_label(-1).with_pos_label(1);
        let fitted = lb.fit(&array![0_usize, 1], &()).unwrap();
        let got = fitted.transform(&array![0_usize, 1, 0]).unwrap();
        let expected: Array2<f64> = array![[-1.0], [1.0], [-1.0]];
        assert_eq!(got, expected);
    }

    /// REQ-8: single-class (k==1) transform -> all neg_label.
    /// Live oracle (sklearn 1.5.2, from /tmp):
    ///   `LabelBinarizer(neg_label=-1,pos_label=2).fit_transform([5,5,5]).tolist()`
    ///     -> `[[-1],[-1],[-1]]`
    #[test]
    fn test_neg_pos_single_class_all_neg() {
        let lb = LabelBinarizer::new().with_neg_label(-1).with_pos_label(2);
        let y = array![5_usize, 5, 5];
        let fitted = lb.fit(&y, &()).unwrap();
        let got = fitted.transform(&y).unwrap();
        let expected: Array2<f64> = array![[-1.0], [-1.0], [-1.0]];
        assert_eq!(got, expected);
    }

    /// REQ-8: unseen labels stay at neg_label (silent-ignore, now -1).
    /// Live oracle (sklearn 1.5.2, from /tmp):
    ///   `LabelBinarizer(neg_label=-1,pos_label=2).fit([0,1,2]).transform([0,3]).tolist()`
    ///     -> `[[2,-1,-1],[-1,-1,-1]]` (label 3 ignored, row all neg_label)
    #[test]
    fn test_neg_pos_unseen_label_stays_neg() {
        let lb = LabelBinarizer::new().with_neg_label(-1).with_pos_label(2);
        let fitted = lb.fit(&array![0_usize, 1, 2], &()).unwrap();
        let got = fitted.transform(&array![0_usize, 3]).unwrap();
        let expected: Array2<f64> = array![[2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]];
        assert_eq!(got, expected);
    }

    /// REQ-8: neg_label >= pos_label is rejected at fit time, verbatim message.
    /// Live oracle (sklearn 1.5.2, from /tmp):
    ///   `LabelBinarizer(neg_label=2,pos_label=1).fit([0,1])`
    ///     -> ValueError: "neg_label=2 must be strictly less than pos_label=1."
    ///   `LabelBinarizer(neg_label=1,pos_label=1).fit([0,1])`
    ///     -> ValueError: "neg_label=1 must be strictly less than pos_label=1."
    #[test]
    fn test_neg_ge_pos_rejected() {
        // neg > pos
        let err = LabelBinarizer::new()
            .with_neg_label(2)
            .with_pos_label(1)
            .fit(&array![0_usize, 1], &())
            .unwrap_err();
        assert!(matches!(
            &err,
            FerroError::InvalidParameter { name, reason }
                if name == "neg_label"
                    && reason == "neg_label=2 must be strictly less than pos_label=1."
        ));

        // neg == pos
        let err = LabelBinarizer::new()
            .with_neg_label(1)
            .with_pos_label(1)
            .fit(&array![0_usize, 1], &())
            .unwrap_err();
        assert!(matches!(
            &err,
            FerroError::InvalidParameter { reason, .. }
                if reason == "neg_label=1 must be strictly less than pos_label=1."
        ));
    }

    /// REQ-8: inverse_transform binary threshold = (pos+neg)/2 (STRICT).
    /// Live oracle (sklearn 1.5.2, from /tmp):
    ///   neg=-1,pos=1 -> threshold 0.0:
    ///     `inverse_transform([[0.0]])` -> [0]; `[[0.1]]` -> [1]; `[[-0.1]]` -> [0]
    ///   neg=2,pos=4 -> threshold 3.0:
    ///     `inverse_transform([[3.0]])` -> [0]; `[[3.1]]` -> [1]
    #[test]
    fn test_neg_pos_inverse_threshold() {
        let fitted = LabelBinarizer::new()
            .with_neg_label(-1)
            .with_pos_label(1)
            .fit(&array![0_usize, 1], &())
            .unwrap();
        // threshold = (1 + -1)/2 = 0.0; strict `> 0.0`
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
        // threshold = (4 + 2)/2 = 3.0; strict `> 3.0`
        assert_eq!(
            fitted.inverse_transform(&array![[3.0_f64]]).unwrap(),
            array![0_usize]
        );
        assert_eq!(
            fitted.inverse_transform(&array![[3.1_f64]]).unwrap(),
            array![1_usize]
        );
    }

    /// REQ-8: inverse_transform multiclass round-trip with neg/pos (argmax
    /// unchanged — pos_label is the largest so argmax still selects it).
    /// Live oracle (sklearn 1.5.2, from /tmp):
    ///   `LabelBinarizer(neg_label=-1,pos_label=2).fit([0,1,2]).inverse_transform(
    ///       [[2,-1,-1],[-1,-1,2]])` -> [0, 2]
    #[test]
    fn test_neg_pos_inverse_multiclass_roundtrip() {
        let fitted = LabelBinarizer::new()
            .with_neg_label(-1)
            .with_pos_label(2)
            .fit(&array![0_usize, 1, 2], &())
            .unwrap();
        let mat: Array2<f64> = array![[2.0, -1.0, -1.0], [-1.0, -1.0, 2.0]];
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, array![0_usize, 2]);
    }

    /// REQ-1/2/3 preserved: defaults (0/1) reproduce the canonical encoding.
    #[test]
    fn test_defaults_preserve_zero_one() {
        let lb = LabelBinarizer::new();
        // multiclass
        let fitted = lb.fit(&array![0_usize, 1, 2, 1], &()).unwrap();
        let expected: Array2<f64> = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ];
        assert_eq!(
            fitted.transform(&array![0_usize, 1, 2, 1]).unwrap(),
            expected
        );
        // Default also via Default::default()
        let lb2 = LabelBinarizer::default();
        assert_eq!((lb2.neg_label(), lb2.pos_label()), (0, 1));
    }
}
