//! Ordinal encoder: map string categories to integer indices.
//!
//! Each column's categories are mapped to integers `0, 1, 2, ...` in **order
//! of first appearance** in the training data. Unknown categories seen during
//! `transform` produce an error.

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
/// mapping from the unique string categories (in order of first appearance)
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
    /// Categories are recorded in **order of first appearance** in each column.
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
            let mut seen: Vec<String> = Vec::new();
            let mut map: HashMap<String, usize> = HashMap::new();

            for i in 0..n_samples {
                let cat = x[[i, j]].clone();
                if !map.contains_key(&cat) {
                    let idx = seen.len();
                    map.insert(cat.clone(), idx);
                    seen.push(cat);
                }
            }

            categories.push(seen);
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

        // Categories should be in order of first appearance
        assert_eq!(fitted.categories()[0], vec!["cat", "dog", "bird"]);
        assert_eq!(fitted.categories()[1], vec!["small", "large", "medium"]);

        let encoded = fitted.transform(&x).unwrap();
        assert_eq!(encoded[[0, 0]], 0); // "cat" -> 0
        assert_eq!(encoded[[1, 0]], 1); // "dog" -> 1
        assert_eq!(encoded[[2, 0]], 0); // "cat" -> 0
        assert_eq!(encoded[[3, 0]], 2); // "bird" -> 2
        assert_eq!(encoded[[0, 1]], 0); // "small" -> 0
        assert_eq!(encoded[[1, 1]], 1); // "large" -> 1
        assert_eq!(encoded[[2, 1]], 2); // "medium" -> 2
        assert_eq!(encoded[[3, 1]], 0); // "small" -> 0
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
        assert_eq!(fitted.categories()[0], vec!["red", "green", "blue"]);
        let encoded = fitted.transform(&x).unwrap();
        assert_eq!(encoded[[0, 0]], 0);
        assert_eq!(encoded[[1, 0]], 1);
        assert_eq!(encoded[[2, 0]], 2);
        assert_eq!(encoded[[3, 0]], 0);
    }

    #[test]
    fn test_n_features() {
        let enc = OrdinalEncoder::new();
        let x = make_2col(&[("a", "x")]);
        let fitted = enc.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_features(), 2);
    }

    #[test]
    fn test_first_appearance_order() {
        // Categories must be in first-appearance order, NOT alphabetical
        let enc = OrdinalEncoder::new();
        let flat = vec!["zebra".to_string(), "ant".to_string(), "moose".to_string()];
        let x = Array2::from_shape_vec((3, 1), flat).unwrap();
        let fitted = enc.fit(&x, &()).unwrap();
        // zebra first, then ant, then moose
        assert_eq!(fitted.categories()[0][0], "zebra");
        assert_eq!(fitted.categories()[0][1], "ant");
        assert_eq!(fitted.categories()[0][2], "moose");
    }
}
