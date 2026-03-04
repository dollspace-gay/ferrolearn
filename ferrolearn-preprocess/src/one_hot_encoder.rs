//! One-hot encoder for categorical integer features.
//!
//! Transforms a matrix of categorical integer indices into a dense binary
//! array where each category is represented by a separate column.
//!
//! # Example
//!
//! ```text
//! Input column with categories {0, 1, 2}:
//!   [0, 1, 2, 1]  →  [[1,0,0],[0,1,0],[0,0,1],[0,1,0]]
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// OneHotEncoder (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted one-hot encoder for multi-column categorical data.
///
/// Input: `Array2<usize>` where each column contains non-negative integer
/// category indices. Calling [`Fit::fit`] learns the set of categories per
/// column and returns a [`FittedOneHotEncoder`].
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::OneHotEncoder;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let enc = OneHotEncoder::<f64>::new();
/// let x = array![[0usize, 1], [1, 0], [2, 1]];
/// let fitted = enc.fit(&x, &()).unwrap();
/// let encoded = fitted.transform(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OneHotEncoder<F> {
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> OneHotEncoder<F> {
    /// Create a new `OneHotEncoder`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for OneHotEncoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedOneHotEncoder
// ---------------------------------------------------------------------------

/// A fitted one-hot encoder holding the number of categories per input column.
///
/// Created by calling [`Fit::fit`] on a [`OneHotEncoder`].
#[derive(Debug, Clone)]
pub struct FittedOneHotEncoder<F> {
    /// Number of unique categories for each input column, in order.
    pub(crate) n_categories: Vec<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FittedOneHotEncoder<F> {
    /// Return the number of categories for each input feature column.
    #[must_use]
    pub fn n_categories(&self) -> &[usize] {
        &self.n_categories
    }

    /// Return the total number of output columns.
    #[must_use]
    pub fn n_output_features(&self) -> usize {
        self.n_categories.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<usize>, ()> for OneHotEncoder<F> {
    type Fitted = FittedOneHotEncoder<F>;
    type Error = FerroError;

    /// Fit the encoder by determining the number of unique categories per column.
    ///
    /// The number of categories for column `j` is `max(x[:, j]) + 1`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<usize>, _y: &()) -> Result<FittedOneHotEncoder<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "OneHotEncoder::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut n_categories = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            let max_cat = col.iter().copied().max().unwrap_or(0);
            n_categories.push(max_cat + 1);
        }

        Ok(FittedOneHotEncoder {
            n_categories,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<usize>> for FittedOneHotEncoder<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform categorical data into a dense one-hot encoded matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    ///
    /// Returns [`FerroError::InvalidParameter`] if any category value exceeds
    /// the maximum seen during fitting.
    fn transform(&self, x: &Array2<usize>) -> Result<Array2<F>, FerroError> {
        let n_features = self.n_categories.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedOneHotEncoder::transform".into(),
            });
        }

        let n_out_cols = self.n_output_features();
        let n_samples = x.nrows();
        let mut out = Array2::zeros((n_samples, n_out_cols));

        let mut col_offset = 0;
        for j in 0..n_features {
            let n_cats = self.n_categories[j];
            for i in 0..n_samples {
                let cat = x[[i, j]];
                if cat >= n_cats {
                    return Err(FerroError::InvalidParameter {
                        name: format!("x[{i},{j}]"),
                        reason: format!(
                            "category {cat} exceeds max seen during fitting ({})",
                            n_cats - 1
                        ),
                    });
                }
                out[[i, col_offset + cat]] = F::one();
            }
            col_offset += n_cats;
        }

        Ok(out)
    }
}

/// Implement `Transform` on the unfitted encoder to satisfy the `FitTransform: Transform`
/// supertrait bound. Calling `transform` on an unfitted encoder always returns an error.
impl<F: Float + Send + Sync + 'static> Transform<Array2<usize>> for OneHotEncoder<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the encoder must be fitted first.
    ///
    /// Use [`Fit::fit`] to produce a [`FittedOneHotEncoder`], then call
    /// [`Transform::transform`] on that.
    fn transform(&self, _x: &Array2<usize>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "OneHotEncoder".into(),
            reason: "encoder must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<usize>> for OneHotEncoder<F> {
    type FitError = FerroError;

    /// Fit the encoder on `x` and return the one-hot encoded output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting or transformation fails.
    fn fit_transform(&self, x: &Array2<usize>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

/// Convenience: encode a 1-D array of categorical integers.
///
/// This wraps the input in a single-column `Array2<usize>` and returns the
/// encoded result with one-hot columns for that single feature.
impl<F: Float + Send + Sync + 'static> FittedOneHotEncoder<F> {
    /// Transform a 1-D slice of category indices.
    ///
    /// # Errors
    ///
    /// Returns an error if any category value is out-of-range.
    pub fn transform_1d(&self, x: &[usize]) -> Result<Array2<F>, FerroError> {
        if self.n_categories.len() != 1 {
            return Err(FerroError::InvalidParameter {
                name: "transform_1d".into(),
                reason: "encoder was fitted on more than one column; use transform instead".into(),
            });
        }
        let col = Array2::from_shape_vec((x.len(), 1), x.to_vec()).map_err(|e| {
            FerroError::InvalidParameter {
                name: "x".into(),
                reason: e.to_string(),
            }
        })?;
        self.transform(&col)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_one_hot_single_column() {
        let enc = OneHotEncoder::<f64>::new();
        let x = array![[0usize], [1], [2]];
        let fitted = enc.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_categories(), &[3]);
        assert_eq!(fitted.n_output_features(), 3);

        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 3]);
        // Row 0: category 0 → [1, 0, 0]
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 0.0);
        assert_eq!(out[[0, 2]], 0.0);
        // Row 1: category 1 → [0, 1, 0]
        assert_eq!(out[[1, 0]], 0.0);
        assert_eq!(out[[1, 1]], 1.0);
        assert_eq!(out[[1, 2]], 0.0);
        // Row 2: category 2 → [0, 0, 1]
        assert_eq!(out[[2, 0]], 0.0);
        assert_eq!(out[[2, 1]], 0.0);
        assert_eq!(out[[2, 2]], 1.0);
    }

    #[test]
    fn test_one_hot_multi_column() {
        let enc = OneHotEncoder::<f64>::new();
        // Two columns: col0 has 3 categories, col1 has 2 categories
        let x = array![[0usize, 0], [1, 1], [2, 0]];
        let fitted = enc.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_categories(), &[3, 2]);
        assert_eq!(fitted.n_output_features(), 5);

        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 5]);
        // Row 0: (0, 0) → [1,0,0, 1,0]
        assert_eq!(out.row(0).to_vec(), vec![1.0, 0.0, 0.0, 1.0, 0.0]);
        // Row 1: (1, 1) → [0,1,0, 0,1]
        assert_eq!(out.row(1).to_vec(), vec![0.0, 1.0, 0.0, 0.0, 1.0]);
        // Row 2: (2, 0) → [0,0,1, 1,0]
        assert_eq!(out.row(2).to_vec(), vec![0.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_out_of_range_category_error() {
        let enc = OneHotEncoder::<f64>::new();
        let x_train = array![[0usize], [1]];
        let fitted = enc.fit(&x_train, &()).unwrap();
        // Category 2 was not seen during fitting
        let x_bad = array![[2usize]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let enc = OneHotEncoder::<f64>::new();
        let x = array![[0usize, 1], [1, 0], [2, 1]];
        let via_fit_transform: Array2<f64> = enc.fit_transform(&x).unwrap();
        let fitted = enc.fit(&x, &()).unwrap();
        let via_separate = fitted.transform(&x).unwrap();
        for (a, b) in via_fit_transform.iter().zip(via_separate.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let enc = OneHotEncoder::<f64>::new();
        let x_train = array![[0usize, 1], [1, 0]];
        let fitted = enc.fit(&x_train, &()).unwrap();
        let x_bad = array![[0usize]];
        assert!(fitted.transform(&x_bad).is_err());
    }
}
