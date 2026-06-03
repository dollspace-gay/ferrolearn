//! Label encoder: maps string labels to integer indices.
//!
//! Learns an ordered mapping from unique string labels to consecutive integers
//! `0, 1, ..., n_classes - 1`. Supports forward (`label → int`) and reverse
//! (`int → label`) transformation.
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_label.py` (`class LabelEncoder`
//! `:34`). Design doc: `.design/preprocess/label_encoder.md`. Expected values from the live
//! sklearn 1.5.2 oracle (R-CHAR-3). Consumer: crate re-export (`lib.rs:116`, grandfathered S5).
//! HONEST (R-HONEST-3): ferrolearn is `Array1<String>`-only; sklearn `LabelEncoder` accepts any
//! hashable+comparable dtype. The non-empty string path value-matches the oracle exactly.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (string fit → sorted-unique classes_) | SHIPPED | `Fit::fit` collects unique labels, `Vec<String>::sort` (lexicographic), builds `label_to_index`; mirrors sklearn `classes_ = _unique(y)` (`_label.py:98`). Critic-verified vs live oracle: `green_fit_classes_sorted` (`["bird","cat","dog"]`), `green_sort_order_mixed_ascii_matches_numpy` (`["10","2","A","B","a","b"]` == np.unique). Consumer: crate re-export `lib.rs:116`. |
//! | REQ-2 (inverse_transform) | SHIPPED | `FittedLabelEncoder::inverse_transform` = `classes[idx]` with out-of-range → `InvalidParameter`; mirrors sklearn `classes_[y]` + `setdiff1d` guard (`:158-162`). Critic-verified: `green_inverse_transform_roundtrip`, out-of-range rejected. |
//! | REQ-3 (transform + fit_transform) | SHIPPED | `transform` = `label_to_index.get` (unknown → `InvalidParameter`), mirrors `_encode` (`:137`); `fit_transform` mirrors `_unique(return_inverse=True)` (`:115`). Critic-verified: `green_transform` (`[1,2,1,0]`), `green_fit_transform_equals_fit_then_transform`, empty transform/inverse → empty (`:134-135`,`:155-156`). |
//! | REQ-5 (empty-fit parity) | SHIPPED | FIXED #1134. Removed the `if x.is_empty()` → `InsufficientSamples` guard; `fit([])` now yields an empty `FittedLabelEncoder` matching sklearn `_unique([])` (`:98`). Critic-verified: `divergence_empty_fit_succeeds` + 4 post-empty-fit guards; in-module `test_empty_fit_yields_empty_classes` (R-HONEST-4). |
//! | REQ-4 (numeric/generic dtype) | NOT-STARTED | open prereq blocker #1135. `Array1<String>`-only; sklearn accepts any dtype, numeric sort `[10,2,1]→[1,2,10]` (`np.unique`) unrepresentable (R-DEV-3). |
//! | REQ-6 (error-contract parity, R-DEV-2) | NOT-STARTED | open prereq blocker #1136. Unseen-label message ("unknown label" vs "y contains previously unseen labels", `:137,160`) + unfitted-transform `InvalidParameter` vs `NotFittedError` (`:131`). Both REJECT (type maps to FerroError); message/NotFitted-analog gap. |
//! | REQ-7 (PyO3 binding) | NOT-STARTED | open prereq blocker #1137. No `ferrolearn-python` registration (R-DEFER-1). LabelEncoder has no `n_features_in_`/`get_feature_names_out` (target encoder). |
//! | REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #1138. `ndarray::Array1<String>` + `std::HashMap`, not `ferray-core` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array1;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// LabelEncoder (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted label encoder.
///
/// Calling [`Fit::fit`] on an `Array1<String>` learns an alphabetically
/// ordered mapping from unique string labels to integer indices
/// `0, 1, ..., n_classes - 1` and returns a [`FittedLabelEncoder`].
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::LabelEncoder;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let enc = LabelEncoder::new();
/// let labels = array!["cat".to_string(), "dog".to_string(), "cat".to_string()];
/// let fitted = enc.fit(&labels, &()).unwrap();
/// let encoded = fitted.transform(&labels).unwrap();
/// assert_eq!(encoded[0], 0); // "cat" → 0
/// assert_eq!(encoded[1], 1); // "dog" → 1
/// ```
#[derive(Debug, Clone, Default)]
pub struct LabelEncoder;

impl LabelEncoder {
    /// Create a new `LabelEncoder`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// FittedLabelEncoder
// ---------------------------------------------------------------------------

/// A fitted label encoder holding the bidirectional label-to-index mapping.
///
/// Created by calling [`Fit::fit`] on a [`LabelEncoder`].
#[derive(Debug, Clone)]
pub struct FittedLabelEncoder {
    /// Ordered list of unique class labels (index = class integer).
    pub(crate) classes: Vec<String>,
    /// Map from label string to integer index.
    pub(crate) label_to_index: HashMap<String, usize>,
}

impl FittedLabelEncoder {
    /// Return the ordered list of class labels.
    ///
    /// `classes[i]` is the label corresponding to integer `i`.
    #[must_use]
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Return the number of unique classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Map integer indices back to the original string labels.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if any index is out of range.
    pub fn inverse_transform(&self, y: &Array1<usize>) -> Result<Array1<String>, FerroError> {
        let n_classes = self.classes.len();
        let mut out = Vec::with_capacity(y.len());
        for (i, &idx) in y.iter().enumerate() {
            if idx >= n_classes {
                return Err(FerroError::InvalidParameter {
                    name: format!("y[{i}]"),
                    reason: format!("index {idx} is out of range (n_classes = {n_classes})"),
                });
            }
            out.push(self.classes[idx].clone());
        }
        Ok(Array1::from_vec(out))
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array1<String>, ()> for LabelEncoder {
    type Fitted = FittedLabelEncoder;
    type Error = FerroError;

    /// Fit the encoder by learning the sorted set of unique labels.
    ///
    /// Labels are sorted alphabetically; the first label maps to `0`.
    ///
    /// Empty input is accepted (matching scikit-learn): it yields a fitted
    /// encoder with an empty `classes_` (`n_classes == 0`).
    ///
    /// # Errors
    ///
    /// This method does not currently return an error.
    fn fit(&self, x: &Array1<String>, _y: &()) -> Result<FittedLabelEncoder, FerroError> {
        let mut unique: Vec<String> = x
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique.sort();

        let label_to_index: HashMap<String, usize> = unique
            .iter()
            .enumerate()
            .map(|(i, label)| (label.clone(), i))
            .collect();

        Ok(FittedLabelEncoder {
            classes: unique,
            label_to_index,
        })
    }
}

impl Transform<Array1<String>> for FittedLabelEncoder {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Transform string labels to integer indices.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if any label was not seen during fitting.
    fn transform(&self, x: &Array1<String>) -> Result<Array1<usize>, FerroError> {
        let mut out = Vec::with_capacity(x.len());
        for (i, label) in x.iter().enumerate() {
            match self.label_to_index.get(label) {
                Some(&idx) => out.push(idx),
                None => {
                    return Err(FerroError::InvalidParameter {
                        name: format!("x[{i}]"),
                        reason: format!("unknown label \"{label}\""),
                    });
                }
            }
        }
        Ok(Array1::from_vec(out))
    }
}

/// Implement `Transform` on the unfitted encoder to satisfy the `FitTransform: Transform`
/// supertrait bound. Calling `transform` on an unfitted encoder always returns an error.
impl Transform<Array1<String>> for LabelEncoder {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Always returns an error — the encoder must be fitted first.
    ///
    /// Use [`Fit::fit`] to produce a [`FittedLabelEncoder`], then call
    /// [`Transform::transform`] on that.
    fn transform(&self, _x: &Array1<String>) -> Result<Array1<usize>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "LabelEncoder".into(),
            reason: "encoder must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl FitTransform<Array1<String>> for LabelEncoder {
    type FitError = FerroError;

    /// Fit the encoder on `x` and return the encoded output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting or transformation fails.
    fn fit_transform(&self, x: &Array1<String>) -> Result<Array1<usize>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn str_arr(v: &[&str]) -> Array1<String> {
        Array1::from_vec(v.iter().map(std::string::ToString::to_string).collect())
    }

    #[test]
    fn test_label_encoder_basic() {
        let enc = LabelEncoder::new();
        let labels = str_arr(&["cat", "dog", "cat", "bird"]);
        let fitted = enc.fit(&labels, &()).unwrap();

        // Classes should be sorted alphabetically
        assert_eq!(fitted.classes(), &["bird", "cat", "dog"]);
        assert_eq!(fitted.n_classes(), 3);

        let encoded = fitted.transform(&labels).unwrap();
        assert_eq!(encoded[0], 1); // "cat" → 1
        assert_eq!(encoded[1], 2); // "dog" → 2
        assert_eq!(encoded[2], 1); // "cat" → 1
        assert_eq!(encoded[3], 0); // "bird" → 0
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let enc = LabelEncoder::new();
        let labels = str_arr(&["a", "b", "c", "a", "b"]);
        let fitted = enc.fit(&labels, &()).unwrap();
        let encoded = fitted.transform(&labels).unwrap();
        let recovered = fitted.inverse_transform(&encoded).unwrap();
        for (orig, rec) in labels.iter().zip(recovered.iter()) {
            assert_eq!(orig, rec);
        }
    }

    #[test]
    fn test_unknown_label_error() {
        let enc = LabelEncoder::new();
        let labels = str_arr(&["a", "b"]);
        let fitted = enc.fit(&labels, &()).unwrap();
        let unknown = str_arr(&["c"]);
        assert!(fitted.transform(&unknown).is_err());
    }

    #[test]
    fn test_inverse_transform_out_of_range() {
        let enc = LabelEncoder::new();
        let labels = str_arr(&["x", "y"]);
        let fitted = enc.fit(&labels, &()).unwrap();
        let bad_indices = array![5usize];
        assert!(fitted.inverse_transform(&bad_indices).is_err());
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let enc = LabelEncoder::new();
        let labels = str_arr(&["foo", "bar", "foo", "baz"]);
        let via_fit_transform = enc.fit_transform(&labels).unwrap();
        let fitted = enc.fit(&labels, &()).unwrap();
        let via_separate = fitted.transform(&labels).unwrap();
        assert_eq!(via_fit_transform, via_separate);
    }

    #[test]
    fn test_empty_fit_yields_empty_classes() {
        // sklearn LabelEncoder().fit([]) succeeds with classes_ == [] (shape (0,)).
        let enc = LabelEncoder::new();
        let empty: Array1<String> = Array1::from_vec(vec![]);
        let fitted = enc
            .fit(&empty, &())
            .ok()
            .filter(|f| f.n_classes() == 0 && f.classes().is_empty());
        assert!(fitted.is_some());
    }

    #[test]
    fn test_single_class() {
        let enc = LabelEncoder::new();
        let labels = str_arr(&["only", "only", "only"]);
        let fitted = enc.fit(&labels, &()).unwrap();
        assert_eq!(fitted.n_classes(), 1);
        let encoded = fitted.transform(&labels).unwrap();
        assert!(encoded.iter().all(|&v| v == 0));
    }
}
