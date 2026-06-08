//! Multi-label binarizer.
//!
//! Transforms a list of label sets into a multi-hot binary indicator matrix.
//! Each sample can belong to zero or more classes simultaneously.
//!
//! Translation target: scikit-learn 1.5.2 `class MultiLabelBinarizer`
//! (`sklearn/preprocessing/_label.py:688`). Design:
//! `.design/preprocess/multi_label_binarizer.md`. Tracking: #1229.
//!
//! `## REQ status`
//!
//! | REQ | Status | Anchor |
//! |---|---|---|
//! | REQ-1 fit → sorted-unique classes_ (usize) | SHIPPED | `MultiLabelBinarizer::fit`; sklearn `_label.py:779` |
//! | REQ-2 transform → dense multi-hot (known labels) | SHIPPED | `FittedMultiLabelBinarizer::transform`; sklearn `_label.py:869-907` |
//! | REQ-3 transform unknown-label: ignore, no error | SHIPPED (#1230) | `transform` skips unknown via `class_to_idx.get`; sklearn `_label.py:889-902` |
//! | REQ-4 inverse_transform 0/1 validation | SHIPPED (#1231) | `inverse_transform` rejects non-0/1, selects `== 1.0`; sklearn `_label.py:941-947` |
//! | REQ-5 `classes` ctor param | NOT-STARTED (#1232) | sklearn `_label.py:756`,`:780-785` |
//! | REQ-6 sparse_output CSR | NOT-STARTED (#1233) | sklearn `_label.py:858-859`,`:905-907` |
//! | REQ-7 arbitrary orderable+hashable labels + object dtype | NOT-STARTED (#1234) | sklearn `_label.py:788` (usize-only, R-DEV-3) |
//! | REQ-8 optimized single-pass fit_transform | NOT-STARTED (#1235) | sklearn `_label.py:814-835` |
//! | REQ-9 PyO3 binding | NOT-STARTED (#1236) | `ferrolearn-python/src/` (absent) |
//! | REQ-1 edge: empty-`y` fit yields empty classes_ (no error) | SHIPPED (#2339) | `MultiLabelBinarizer::fit` (no empty-`y` rejection): empty `y` → `classes = []`; `transform([[]])` → `(1, 0)` `Array2`; sklearn `_label.py:779` |
//!
//! # Examples
//!
//! ```
//! use ferrolearn_preprocess::multi_label_binarizer::MultiLabelBinarizer;
//! use ferrolearn_core::traits::{Fit, Transform};
//!
//! let mlb = MultiLabelBinarizer::new();
//! let y = vec![vec![0, 1], vec![1, 2], vec![0]];
//! let fitted = mlb.fit(&y, &()).unwrap();
//! let mat = fitted.transform(&y).unwrap();
//! // 3 classes → (3, 3) multi-hot matrix
//! assert_eq!(mat.shape(), &[3, 3]);
//! assert_eq!(mat[[0, 0]], 1.0); // sample 0 has label 0
//! assert_eq!(mat[[0, 1]], 1.0); // sample 0 has label 1
//! assert_eq!(mat[[0, 2]], 0.0); // sample 0 does NOT have label 2
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;

// ---------------------------------------------------------------------------
// MultiLabelBinarizer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted multi-label binarizer.
///
/// Calling [`Fit::fit`] on a `&[Vec<usize>]` discovers the sorted set of all
/// unique labels across all samples and returns a [`FittedMultiLabelBinarizer`].
#[derive(Debug, Clone, Default)]
pub struct MultiLabelBinarizer;

impl MultiLabelBinarizer {
    /// Create a new `MultiLabelBinarizer`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// FittedMultiLabelBinarizer
// ---------------------------------------------------------------------------

/// A fitted multi-label binarizer holding the discovered class set.
///
/// Created by calling [`Fit::fit`] on a [`MultiLabelBinarizer`].
#[derive(Debug, Clone)]
pub struct FittedMultiLabelBinarizer {
    /// Sorted unique class labels observed during fitting.
    classes: Vec<usize>,
}

impl FittedMultiLabelBinarizer {
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

    /// Map a multi-hot indicator matrix back to label sets.
    ///
    /// The indicator matrix must contain only exact `0.0` and `1.0` values; a
    /// class is included for a sample iff its cell is exactly `1.0`. This
    /// mirrors scikit-learn 1.5.2 `MultiLabelBinarizer.inverse_transform`
    /// (`sklearn/preprocessing/_label.py:941-947`), which validates the matrix
    /// with `np.setdiff1d(yt, [0, 1])` and raises `ValueError` on any value
    /// outside `{0, 1}` before selecting classes where the cell `== 1`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does
    /// not match the number of classes. Returns [`FerroError::InvalidParameter`]
    /// if any cell value is not exactly `0.0` or `1.0`.
    #[allow(
        clippy::float_cmp,
        reason = "indicator matrix must be exactly 0/1 per sklearn _label.py:941-947"
    )]
    pub fn inverse_transform(&self, y: &Array2<f64>) -> Result<Vec<Vec<usize>>, FerroError> {
        let k = self.classes.len();
        if y.ncols() != k {
            return Err(FerroError::ShapeMismatch {
                expected: vec![y.nrows(), k],
                actual: vec![y.nrows(), y.ncols()],
                context: "FittedMultiLabelBinarizer::inverse_transform".into(),
            });
        }

        // Validate the indicator contains only 0s and 1s, matching sklearn's
        // `np.setdiff1d(yt, [0, 1])` check (_label.py:941-947).
        if let Some(&v) = y.iter().find(|&&v| v != 0.0 && v != 1.0) {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: format!("Expected only 0s and 1s in label indicator, got {v}"),
            });
        }

        let n = y.nrows();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut labels = Vec::new();
            for (j, &cls) in self.classes.iter().enumerate() {
                if y[[i, j]] == 1.0 {
                    labels.push(cls);
                }
            }
            result.push(labels);
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Vec<Vec<usize>>, ()> for MultiLabelBinarizer {
    type Fitted = FittedMultiLabelBinarizer;
    type Error = FerroError;

    /// Fit the binarizer by discovering all unique labels.
    ///
    /// An empty input (no samples) is accepted and yields an empty `classes_`,
    /// mirroring sklearn `MultiLabelBinarizer.fit` where
    /// `classes_ = sorted(set(itertools.chain.from_iterable(y)))` is the empty
    /// set for `y == []` (`sklearn/preprocessing/_label.py:779`); sklearn raises
    /// no error and a subsequent `transform([[]])` yields a `(1, 0)` matrix.
    ///
    /// # Errors
    ///
    /// This method does not return an error in the `usize` domain (kept as
    /// `Result` for `Fit`-trait conformance and forward compatibility).
    fn fit(
        &self,
        y: &Vec<Vec<usize>>,
        _target: &(),
    ) -> Result<FittedMultiLabelBinarizer, FerroError> {
        let mut classes: Vec<usize> = y.iter().flatten().copied().collect();
        classes.sort_unstable();
        classes.dedup();

        Ok(FittedMultiLabelBinarizer { classes })
    }
}

impl Transform<Vec<Vec<usize>>> for FittedMultiLabelBinarizer {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Transform label sets into a multi-hot indicator matrix.
    ///
    /// Each row has a `1.0` in every column corresponding to one of its labels
    /// and `0.0` elsewhere.
    ///
    /// Labels not seen during fitting are silently ignored: the indicator is
    /// built only from known labels (mirroring scikit-learn 1.5.2
    /// `MultiLabelBinarizer._transform`, `sklearn/preprocessing/_label.py:889-902`).
    /// scikit-learn additionally emits a `warnings.warn("unknown class(es) ...
    /// will be ignored")`; that warning is intentionally not emitted here because
    /// the crate has no logging facade and adding one would be out of scope.
    ///
    /// The [`Result`] return type is retained because the [`Transform`] trait
    /// requires it; `transform` always returns [`Ok`].
    fn transform(&self, y: &Vec<Vec<usize>>) -> Result<Array2<f64>, FerroError> {
        let k = self.classes.len();
        let n = y.len();

        // Build lookup: class_value → column index
        let class_to_idx: std::collections::HashMap<usize, usize> = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        let mut out = Array2::zeros((n, k));

        for (i, labels) in y.iter().enumerate() {
            for &label in labels {
                // Unknown labels (not seen during fit) are silently ignored,
                // matching scikit-learn's `_transform` (_label.py:889-902).
                if let Some(&idx) = class_to_idx.get(&label) {
                    out[[i, idx]] = 1.0;
                }
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
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![2, 0], vec![1]];
        let fitted = mlb.fit(&y, &()).unwrap();
        assert_eq!(fitted.classes(), &[0, 1, 2]);
    }

    #[test]
    fn test_fit_empty_input_yields_empty_classes() {
        // sklearn 1.5.2: `MultiLabelBinarizer().fit([])` SUCCEEDS with
        // `classes_ == []` (`sorted(set(chain.from_iterable([])))` is empty,
        // `_label.py:779`), and `transform([[]])` is a `(1, 0)` matrix.
        // Live oracle (from /tmp):
        //   mlb = MultiLabelBinarizer().fit([]); mlb.classes_.tolist() -> []
        //   mlb.transform([[]]).shape -> (1, 0)
        let mlb = MultiLabelBinarizer::new();
        let empty: Vec<Vec<usize>> = vec![];
        let one_empty_sample = vec![vec![]];
        let got = mlb.fit(&empty, &()).and_then(|fitted| {
            fitted
                .transform(&one_empty_sample)
                .map(|m| m.shape().to_vec())
        });
        assert_eq!(got.ok(), Some(vec![1, 0]));
    }

    #[test]
    fn test_transform_multi_hot() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 2], vec![1], vec![0, 1, 2]];
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[3, 3]);
        // Row 0: labels {0, 2} → [1, 0, 1]
        assert_eq!(mat[[0, 0]], 1.0);
        assert_eq!(mat[[0, 1]], 0.0);
        assert_eq!(mat[[0, 2]], 1.0);
        // Row 1: labels {1} → [0, 1, 0]
        assert_eq!(mat[[1, 0]], 0.0);
        assert_eq!(mat[[1, 1]], 1.0);
        assert_eq!(mat[[1, 2]], 0.0);
        // Row 2: labels {0, 1, 2} → [1, 1, 1]
        assert_eq!(mat[[2, 0]], 1.0);
        assert_eq!(mat[[2, 1]], 1.0);
        assert_eq!(mat[[2, 2]], 1.0);
    }

    #[test]
    fn test_transform_unknown_label_ignored() {
        // Live oracle (sklearn 1.5.2):
        //   python3 -c "from sklearn.preprocessing import MultiLabelBinarizer; \
        //     import warnings; warnings.simplefilter('ignore'); \
        //     mlb=MultiLabelBinarizer().fit([[0,1]]); \
        //     print(mlb.transform([[0,5]]).tolist())"
        //   => [[1, 0]]
        // Unknown labels are skipped, not errored (_label.py:889-902).
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 1]];
        let fitted = mlb.fit(&y, &()).map_err(|e| format!("{e:?}"));
        let y2 = vec![vec![0, 5]]; // 5 not in {0, 1} → ignored
        // Transform must NOT error on the unknown label 5; it is skipped.
        let got = fitted.and_then(|f| f.transform(&y2).map_err(|e| format!("{e:?}")));
        assert_eq!(got, Ok(array![[1.0, 0.0]]));
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 2], vec![1], vec![0, 1, 2]];
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    #[test]
    fn test_inverse_transform_shape_mismatch() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 1, 2]];
        let fitted = mlb.fit(&y, &()).unwrap();
        // 3 classes expects 3 columns
        let bad = Array2::<f64>::zeros((2, 2));
        assert!(fitted.inverse_transform(&bad).is_err());
    }

    #[test]
    fn test_empty_label_set() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 1], vec![]]; // second sample has no labels
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[2, 2]);
        // Row 1 should be all zeros
        assert_eq!(mat[[1, 0]], 0.0);
        assert_eq!(mat[[1, 1]], 0.0);
    }

    #[test]
    fn test_inverse_transform_empty_row() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 1], vec![]];
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    #[test]
    fn test_non_contiguous_classes() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![10, 30], vec![20]];
        let fitted = mlb.fit(&y, &()).unwrap();
        assert_eq!(fitted.classes(), &[10, 20, 30]);
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[2, 3]);
        assert_eq!(mat[[0, 0]], 1.0); // 10
        assert_eq!(mat[[0, 1]], 0.0); // 20
        assert_eq!(mat[[0, 2]], 1.0); // 30
    }

    #[test]
    fn test_inverse_transform_non_contiguous_roundtrip() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![10, 30], vec![20]];
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    #[test]
    fn test_duplicate_labels_in_input() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 0, 1]]; // duplicate 0
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        // Still produces [1, 1] — duplicates don't cause double-counting
        assert_eq!(mat.shape(), &[1, 2]);
        assert_eq!(mat[[0, 0]], 1.0);
        assert_eq!(mat[[0, 1]], 1.0);
    }

    #[test]
    fn test_inverse_rejects_non_01() {
        // sklearn 1.5.2 validates the indicator with `np.setdiff1d(yt, [0, 1])`
        // and raises `ValueError` on any value outside {0, 1}
        // (sklearn/preprocessing/_label.py:941-947). It does NOT threshold.
        //
        // Live oracle (sklearn 1.5.2), valid 0/1 round-trip (R-CHAR-3):
        //   python3 -c "import numpy as np; \
        //     from sklearn.preprocessing import MultiLabelBinarizer; \
        //     mlb=MultiLabelBinarizer().fit([[0,1,2]]); \
        //     print(mlb.inverse_transform(np.array([[1,0,1]])))"
        //   => [(0, 2)]  ==  vec![vec![0, 2]]
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 1, 2]];
        let fitted = mlb.fit(&y, &()).map_err(|e| format!("{e:?}"));

        // Non-0/1 values are rejected (sklearn raises ValueError).
        let bad = array![[0.4, 0.6, 0.5]];
        let rejected = fitted
            .as_ref()
            .map_err(|e| e.clone())
            .map(|f| f.inverse_transform(&bad).is_err());
        assert_eq!(rejected, Ok(true));

        // A valid 0/1 indicator round-trips to the live-oracle result.
        let good = array![[1.0, 0.0, 1.0]];
        let recovered =
            fitted.and_then(|f| f.inverse_transform(&good).map_err(|e| format!("{e:?}")));
        assert_eq!(recovered, Ok(vec![vec![0, 2]]));
    }
}
