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
//! | REQ-4 transform unknown-label: ignore (all-zero row) | SHIPPED (#1239) | `transform` `if let Some(&idx) = class_to_idx.get`; sklearn `_label.py:556-559` |
//! | REQ-5 transform single-class (k=1) → all-zero column | SHIPPED (#1240) | `transform` k==1 arm `Array2::zeros`; sklearn `_label.py:532-538` |
//! | REQ-6 inverse_transform binary STRICT threshold (`> 0.5`) | SHIPPED (#1241) | `inverse_transform` k==2 branch; sklearn `_label.py:667` |
//! | REQ-7 inverse_transform multiclass argmax | SHIPPED | `inverse_transform` else-branch; sklearn `_label.py:641` |
//! | REQ-8 neg_label/pos_label ctor params + validation | NOT-STARTED (#1242) | sklearn `_label.py:263`,`:283-287`,`:579-583` |
//! | REQ-9 sparse_output CSR + constraint | NOT-STARTED (#1243) | sklearn `_label.py:563`,`:584-585`,`:289-294` |
//! | REQ-10 `label_binarize` free function | NOT-STARTED (#1244) | sklearn `_label.py:430` |
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
#[derive(Debug, Clone, Default)]
pub struct LabelBinarizer;

impl LabelBinarizer {
    /// Create a new `LabelBinarizer`.
    #[must_use]
    pub fn new() -> Self {
        Self
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
            // Single column: strict threshold at 0.5, matching sklearn
            // `_inverse_binarize_thresholding` (`_label.py:667`): `y > threshold`
            // with default `threshold = (pos_label + neg_label) / 2 = 0.5`, so an
            // exact 0.5 maps to `classes[0]` (since `0.5 > 0.5` is False).
            for i in 0..n {
                result[i] = if y[[i, 0]] > 0.5 {
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
    /// Returns [`FerroError::InsufficientSamples`] if the input is empty.
    fn fit(&self, y: &Array1<usize>, _target: &()) -> Result<FittedLabelBinarizer, FerroError> {
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

        Ok(FittedLabelBinarizer { classes })
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
    /// Labels not seen during fitting are silently ignored: their row is left
    /// at the all-zero (negative-label) value, with no error and no warning.
    /// This mirrors scikit-learn's `label_binarize`
    /// (`sklearn/preprocessing/_label.py:556-559`), which selects only the
    /// known labels (`y_in_classes = np.isin(y, classes)`) and leaves unseen
    /// labels contributing nothing.
    fn transform(&self, y: &Array1<usize>) -> Result<Array2<f64>, FerroError> {
        let k = self.classes.len();
        let n = y.len();

        // Build a lookup: class_value → column index
        let class_to_idx: std::collections::HashMap<usize, usize> = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        if k == 1 {
            // Single class (n_classes == 1): sklearn treats this as the binary
            // degenerate case and returns an all-`neg_label` (=0) single column,
            // never a 1.0 (`sklearn/preprocessing/_label.py:532-538`:
            // `Y = np.zeros((len(y), 1)); Y += neg_label`).
            Ok(Array2::zeros((n, 1)))
        } else if k == 2 {
            // Binary: single column, 1.0 for the second class
            let mut out = Array2::zeros((n, 1));
            for (i, &label) in y.iter().enumerate() {
                // Unseen labels are silently ignored (row left at neg_label=0.0),
                // mirroring sklearn `_label.py:556-559`.
                if let Some(&idx) = class_to_idx.get(&label) {
                    out[[i, 0]] = if idx == 1 { 1.0 } else { 0.0 };
                }
            }
            Ok(out)
        } else {
            // Multiclass: one-hot rows
            let mut out = Array2::zeros((n, k));
            for (i, &label) in y.iter().enumerate() {
                // Unseen labels are silently ignored (row left all-zero),
                // mirroring sklearn `_label.py:556-559`.
                if let Some(&idx) = class_to_idx.get(&label) {
                    out[[i, idx]] = 1.0;
                }
            }
            Ok(out)
        }
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

    /// Unseen labels are silently ignored (row left all-zero), mirroring
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
}
