//! Ordinal encoder: map string categories to integer indices.
//!
//! Each column's categories are mapped to integers `0, 1, 2, ...` in
//! **lexicographic order** (matching scikit-learn's `OrdinalEncoder`).
//! Unknown categories seen during
//! `transform` produce an error.
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_encoders.py` (`class OrdinalEncoder`
//! `:1235`). Design doc: `.design/preprocess/ordinal_encoder.md`. Expected values from the live
//! sklearn 1.5.2 oracle (R-CHAR-3). Consumer: crate re-export (`lib.rs:121`, grandfathered S5).
//! HONEST (R-HONEST-3): a FAITHFUL String-only ordinal encoder — `categories_`=sorted-unique and
//! the ordinal VALUES match sklearn bit-for-bit on the string path; divergences are the output
//! container dtype (`usize` vs float64), String-only input, and the absent param/feature surface.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (string fit → sorted-unique categories_) | SHIPPED | `Fit::fit` per column → `categories_`=sorted-unique (`Vec<String>::sort`, lexicographic) + index map; rejects 0 rows (`InsufficientSamples`, matches sklearn `check_array`). Mirrors `_BaseEncoder._fit` `categories_=_unique(Xi)` (`_encoders.py:99`). Critic-verified vs live oracle: `green_value_match_and_categories` (`[['bird','cat','dog'],['large','medium','small']]`), `green_lexicographic_sort_matches_np_unique` + `green_non_ascii_codepoint_order` (== `np.unique`), `green_empty_fit_rejected_matches_sklearn`. Consumer: re-export `lib.rs:121`. |
//! | REQ-2 (transform + fit_transform, ordinal values + unknown rejection) | SHIPPED | `Transform::transform` maps category→`usize` ordinal index, unknown → `InvalidParameter` (matches `handle_unknown='error'` default `ValueError`), ncols-mismatch → `ShapeMismatch`. Critic-verified: ordinal VALUES `[[1,2],[2,0],[1,1],[0,2]]` == live oracle (integer-equal to sklearn float), `green_unknown_category_rejected`, `green_fit_transform_equals_oracle`. Consumer: re-export `lib.rs:121`. |
//! | REQ-3 (output dtype float64 + dtype param) | NOT-STARTED | open prereq blocker #1158. `Array2<usize>` output; sklearn defaults float64 (`:1262`) + `dtype` param. Values equal, container dtype diverges (R-DEV-3); coupled to NaN-sentinel features (REQ-5/6). |
//! | REQ-4 (numeric/mixed-dtype input) | NOT-STARTED | open prereq blocker #1159. `Array2<String>`-only; sklearn accepts int/str/object (`np.unique` numeric sort). |
//! | REQ-5 (handle_unknown='use_encoded_value' + unknown_value) | NOT-STARTED | open prereq blocker #1160. Unknowns always error (`:1265`,`:1274`). |
//! | REQ-6 (encoded_missing_value / NaN) | NOT-STARTED | open prereq blocker #1161. No missing-value concept (`:1283`). |
//! | REQ-7 (explicit categories param) | NOT-STARTED | open prereq blocker #1162. Always `'auto'` (`:1252`). |
//! | REQ-8 (min_frequency/max_categories infrequent) | NOT-STARTED | open prereq blocker #1163. No infrequent folding (`:1289-1315`). |
//! | REQ-9 (inverse_transform) | NOT-STARTED | open prereq blocker #1164. None. |
//! | REQ-10 (get_feature_names_out + n_features_in_) | NOT-STARTED | open prereq blocker #1165. Only `n_features()`. |
//! | REQ-11 (full ctor + _parameter_constraints) | NOT-STARTED | open prereq blocker #1166. `new()` takes no params (`:1320-1386`). |
//! | REQ-12 (PyO3 binding) | NOT-STARTED | open prereq blocker #1167. No `ferrolearn-python` registration (R-DEFER-1). |
//! | REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1168. `ndarray`+`HashMap`, not `ferray-core` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array2;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// OrdinalEncoder (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted ordinal encoder.
///
/// Calling [`Fit::fit`] on an `Array2<String>` learns, for each column, a
/// mapping from the unique string categories (sorted lexicographically)
/// to consecutive integers `0, 1, 2, ...`, and returns a
/// [`FittedOrdinalEncoder`].
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::ordinal_encoder::OrdinalEncoder;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::Array2;
///
/// let enc = OrdinalEncoder::new();
/// let data = Array2::from_shape_vec(
///     (3, 2),
///     vec![
///         "cat".to_string(), "small".to_string(),
///         "dog".to_string(), "large".to_string(),
///         "cat".to_string(), "small".to_string(),
///     ],
/// ).unwrap();
/// let fitted = enc.fit(&data, &()).unwrap();
/// let encoded = fitted.transform(&data).unwrap();
/// assert_eq!(encoded[[0, 0]], 0); // "cat" is index 0 in col 0
/// assert_eq!(encoded[[1, 0]], 1); // "dog" is index 1 in col 0
/// ```
#[derive(Debug, Clone, Default)]
pub struct OrdinalEncoder;

impl OrdinalEncoder {
    /// Create a new `OrdinalEncoder`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// FittedOrdinalEncoder
// ---------------------------------------------------------------------------

/// A fitted ordinal encoder holding per-column category-to-index mappings.
///
/// Created by calling [`Fit::fit`] on an [`OrdinalEncoder`].
#[derive(Debug, Clone)]
pub struct FittedOrdinalEncoder {
    /// Per-column ordered category lists (index = integer value).
    pub(crate) categories: Vec<Vec<String>>,
    /// Per-column category-to-index maps.
    pub(crate) category_to_index: Vec<HashMap<String, usize>>,
}

impl FittedOrdinalEncoder {
    /// Return the ordered category list for each column.
    ///
    /// `categories()[j][i]` is the category that maps to integer `i` in column `j`.
    #[must_use]
    pub fn categories(&self) -> &[Vec<String>] {
        &self.categories
    }

    /// Return the number of input columns (features).
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.categories.len()
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<String>, ()> for OrdinalEncoder {
    type Fitted = FittedOrdinalEncoder;
    type Error = FerroError;

    /// Fit the encoder by building per-column category-to-index mappings.
    ///
    /// Categories are recorded in **lexicographic order** in each column,
    /// matching scikit-learn's `OrdinalEncoder.categories_`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<String>, _y: &()) -> Result<FittedOrdinalEncoder, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "OrdinalEncoder::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut categories = Vec::with_capacity(n_features);
        let mut category_to_index = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect unique categories then sort lexicographically so the
            // assigned indices match sklearn's `OrdinalEncoder`, which
            // documents `categories_ = sorted(unique(X[:, j]))`. (Older
            // ferrolearn versions used first-seen order — #344.)
            let mut unique: Vec<String> = Vec::new();
            let mut seen_set: std::collections::HashSet<String> = std::collections::HashSet::new();
            for i in 0..n_samples {
                let cat = &x[[i, j]];
                if seen_set.insert(cat.clone()) {
                    unique.push(cat.clone());
                }
            }
            unique.sort();

            let map: HashMap<String, usize> = unique
                .iter()
                .enumerate()
                .map(|(idx, s)| (s.clone(), idx))
                .collect();

            categories.push(unique);
            category_to_index.push(map);
        }

        Ok(FittedOrdinalEncoder {
            categories,
            category_to_index,
        })
    }
}

impl Transform<Array2<String>> for FittedOrdinalEncoder {
    type Output = Array2<usize>;
    type Error = FerroError;

    /// Transform string categories to integer indices.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    ///
    /// Returns [`FerroError::InvalidParameter`] if any category was not seen
    /// during fitting.
    fn transform(&self, x: &Array2<String>) -> Result<Array2<usize>, FerroError> {
        let n_features = self.categories.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedOrdinalEncoder::transform".into(),
            });
        }

        let n_samples = x.nrows();
        let mut out = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let map = &self.category_to_index[j];
            for i in 0..n_samples {
                let cat = &x[[i, j]];
                match map.get(cat) {
                    Some(&idx) => out[[i, j]] = idx,
                    None => {
                        return Err(FerroError::InvalidParameter {
                            name: format!("x[{i},{j}]"),
                            reason: format!("unknown category \"{cat}\" in column {j}"),
                        });
                    }
                }
            }
        }

        Ok(out)
    }
}

/// Implement `Transform` on the unfitted encoder to satisfy the
/// `FitTransform: Transform` supertrait bound.
impl Transform<Array2<String>> for OrdinalEncoder {
    type Output = Array2<usize>;
    type Error = FerroError;

    /// Always returns an error — the encoder must be fitted first.
    fn transform(&self, _x: &Array2<String>) -> Result<Array2<usize>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "OrdinalEncoder".into(),
            reason: "encoder must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl FitTransform<Array2<String>> for OrdinalEncoder {
    type FitError = FerroError;

    /// Fit the encoder on `x` and return the encoded output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting or transformation fails.
    fn fit_transform(&self, x: &Array2<String>) -> Result<Array2<usize>, FerroError> {
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
    use ndarray::Array2;

    fn make_2col(rows: &[(&str, &str)]) -> Array2<String> {
        let flat: Vec<String> = rows
            .iter()
            .flat_map(|(a, b)| [a.to_string(), b.to_string()])
            .collect();
        Array2::from_shape_vec((rows.len(), 2), flat).unwrap()
    }

    #[test]
    fn test_ordinal_encoder_basic() {
        let enc = OrdinalEncoder::new();
        let x = make_2col(&[
            ("cat", "small"),
            ("dog", "large"),
            ("cat", "medium"),
            ("bird", "small"),
        ]);
        let fitted = enc.fit(&x, &()).unwrap();

        // Categories are sorted lexicographically (sklearn convention).
        assert_eq!(fitted.categories()[0], vec!["bird", "cat", "dog"]);
        assert_eq!(fitted.categories()[1], vec!["large", "medium", "small"]);

        let encoded = fitted.transform(&x).unwrap();
        assert_eq!(encoded[[0, 0]], 1); // "cat"  -> 1 (lex pos)
        assert_eq!(encoded[[1, 0]], 2); // "dog"  -> 2
        assert_eq!(encoded[[2, 0]], 1); // "cat"  -> 1
        assert_eq!(encoded[[3, 0]], 0); // "bird" -> 0
        assert_eq!(encoded[[0, 1]], 2); // "small"  -> 2
        assert_eq!(encoded[[1, 1]], 0); // "large"  -> 0
        assert_eq!(encoded[[2, 1]], 1); // "medium" -> 1
        assert_eq!(encoded[[3, 1]], 2); // "small"  -> 2
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let enc = OrdinalEncoder::new();
        let x = make_2col(&[("a", "x"), ("b", "y"), ("a", "z")]);
        let via_ft = enc.fit_transform(&x).unwrap();
        let fitted = enc.fit(&x, &()).unwrap();
        let via_sep = fitted.transform(&x).unwrap();
        assert_eq!(via_ft, via_sep);
    }

    #[test]
    fn test_unknown_category_error() {
        let enc = OrdinalEncoder::new();
        let x_train = make_2col(&[("cat", "small"), ("dog", "large")]);
        let fitted = enc.fit(&x_train, &()).unwrap();
        let x_test = make_2col(&[("fish", "small")]);
        assert!(fitted.transform(&x_test).is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let enc = OrdinalEncoder::new();
        let x_train = make_2col(&[("a", "x")]);
        let fitted = enc.fit(&x_train, &()).unwrap();
        // Single-column input when 2 cols expected
        let x_bad = Array2::from_shape_vec((1, 1), vec!["a".to_string()]).unwrap();
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_insufficient_samples_error() {
        let enc = OrdinalEncoder::new();
        let x: Array2<String> = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        assert!(enc.fit(&x, &()).is_err());
    }

    #[test]
    fn test_unfitted_transform_error() {
        let enc = OrdinalEncoder::new();
        let x = make_2col(&[("a", "x")]);
        assert!(enc.transform(&x).is_err());
    }

    #[test]
    fn test_single_column() {
        let enc = OrdinalEncoder::new();
        let flat = vec![
            "red".to_string(),
            "green".to_string(),
            "blue".to_string(),
            "red".to_string(),
        ];
        let x = Array2::from_shape_vec((4, 1), flat).unwrap();
        let fitted = enc.fit(&x, &()).unwrap();
        // Lex order: blue (0), green (1), red (2)
        assert_eq!(fitted.categories()[0], vec!["blue", "green", "red"]);
        let encoded = fitted.transform(&x).unwrap();
        assert_eq!(encoded[[0, 0]], 2); // red
        assert_eq!(encoded[[1, 0]], 1); // green
        assert_eq!(encoded[[2, 0]], 0); // blue
        assert_eq!(encoded[[3, 0]], 2); // red
    }

    #[test]
    fn test_n_features() {
        let enc = OrdinalEncoder::new();
        let x = make_2col(&[("a", "x")]);
        let fitted = enc.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_features(), 2);
    }

    #[test]
    fn test_lexicographic_order() {
        // Categories are sorted lexicographically to match sklearn (#344).
        let enc = OrdinalEncoder::new();
        let flat = vec!["zebra".to_string(), "ant".to_string(), "moose".to_string()];
        let x = Array2::from_shape_vec((3, 1), flat).unwrap();
        let fitted = enc.fit(&x, &()).unwrap();
        // ant < moose < zebra
        assert_eq!(fitted.categories()[0][0], "ant");
        assert_eq!(fitted.categories()[0][1], "moose");
        assert_eq!(fitted.categories()[0][2], "zebra");
    }
}
