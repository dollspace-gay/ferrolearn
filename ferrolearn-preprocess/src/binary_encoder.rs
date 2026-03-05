//! Binary encoder: encode categorical integers as binary digits.
//!
//! [`BinaryEncoder`] encodes each categorical integer feature into `ceil(log2(k))`
//! binary columns, where `k` is the number of distinct categories. This is more
//! compact than one-hot encoding for high-cardinality features.
//!
//! # Example
//!
//! ```text
//! Input column with categories {0, 1, 2, 3}:
//!   0 → [0, 0]
//!   1 → [0, 1]
//!   2 → [1, 0]
//!   3 → [1, 1]
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// BinaryEncoder (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted binary encoder.
///
/// Takes a matrix of categorical integer features and encodes each category
/// as a sequence of binary digits. For `k` categories, each feature produces
/// `ceil(log2(k))` output columns.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::binary_encoder::BinaryEncoder;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let enc = BinaryEncoder::<f64>::new();
/// let x = array![[0usize], [1], [2], [3]];
/// let fitted = enc.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// // 4 categories → ceil(log2(4)) = 2 binary columns
/// assert_eq!(out.ncols(), 2);
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct BinaryEncoder<F> {
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> BinaryEncoder<F> {
    /// Create a new `BinaryEncoder`.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for BinaryEncoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedBinaryEncoder
// ---------------------------------------------------------------------------

/// A fitted binary encoder holding the number of categories and binary digits
/// per input feature.
///
/// Created by calling [`Fit::fit`] on a [`BinaryEncoder`].
#[derive(Debug, Clone)]
pub struct FittedBinaryEncoder<F> {
    /// Number of categories for each input column.
    n_categories: Vec<usize>,
    /// Number of binary digits for each input column.
    n_digits: Vec<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FittedBinaryEncoder<F> {
    /// Return the number of categories per feature.
    #[must_use]
    pub fn n_categories(&self) -> &[usize] {
        &self.n_categories
    }

    /// Return the number of binary digits per feature.
    #[must_use]
    pub fn n_digits(&self) -> &[usize] {
        &self.n_digits
    }

    /// Return the total number of output columns.
    #[must_use]
    pub fn n_output_features(&self) -> usize {
        self.n_digits.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute ceil(log2(k)), with a minimum of 1.
fn n_binary_digits(k: usize) -> usize {
    if k <= 1 {
        return 1;
    }
    // ceil(log2(k)) = number of bits needed to represent 0..k-1
    let mut bits = 0usize;
    let mut val = k - 1; // maximum value to represent
    while val > 0 {
        bits += 1;
        val >>= 1;
    }
    bits
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<usize>, ()> for BinaryEncoder<F> {
    type Fitted = FittedBinaryEncoder<F>;
    type Error = FerroError;

    /// Fit by determining the number of categories per column.
    ///
    /// The number of categories for column `j` is `max(x[:, j]) + 1`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<usize>, _y: &()) -> Result<FittedBinaryEncoder<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "BinaryEncoder::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut n_categories = Vec::with_capacity(n_features);
        let mut n_digits_vec = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            let max_cat = col.iter().copied().max().unwrap_or(0);
            let k = max_cat + 1;
            n_categories.push(k);
            n_digits_vec.push(n_binary_digits(k));
        }

        Ok(FittedBinaryEncoder {
            n_categories,
            n_digits: n_digits_vec,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<usize>> for FittedBinaryEncoder<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform categorical data into binary encoded columns.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    ///
    /// Returns [`FerroError::InvalidParameter`] if any category value exceeds
    /// the maximum seen during fitting.
    fn transform(&self, x: &Array2<usize>) -> Result<Array2<F>, FerroError> {
        let n_features = self.n_categories.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedBinaryEncoder::transform".into(),
            });
        }

        let n_samples = x.nrows();
        let n_out = self.n_output_features();
        let mut out = Array2::zeros((n_samples, n_out));

        let mut col_offset = 0;
        for j in 0..n_features {
            let n_cats = self.n_categories[j];
            let digits = self.n_digits[j];

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

                // Encode category as binary digits (MSB first)
                for bit in 0..digits {
                    let bit_pos = digits - 1 - bit;
                    if (cat >> bit_pos) & 1 == 1 {
                        out[[i, col_offset + bit]] = F::one();
                    }
                }
            }

            col_offset += digits;
        }

        Ok(out)
    }
}

/// Implement `Transform` on the unfitted encoder.
impl<F: Float + Send + Sync + 'static> Transform<Array2<usize>> for BinaryEncoder<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the encoder must be fitted first.
    fn transform(&self, _x: &Array2<usize>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "BinaryEncoder".into(),
            reason: "encoder must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<usize>> for BinaryEncoder<F> {
    type FitError = FerroError;

    /// Fit and transform in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting or transformation fails.
    fn fit_transform(&self, x: &Array2<usize>) -> Result<Array2<F>, FerroError> {
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

    #[test]
    fn test_binary_encoder_basic() {
        let enc = BinaryEncoder::<f64>::new();
        let x = array![[0usize], [1], [2], [3]];
        let fitted = enc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // 4 categories → ceil(log2(4)) = 2 columns
        assert_eq!(out.ncols(), 2);
        // 0 → [0, 0]
        assert_eq!(out.row(0).to_vec(), vec![0.0, 0.0]);
        // 1 → [0, 1]
        assert_eq!(out.row(1).to_vec(), vec![0.0, 1.0]);
        // 2 → [1, 0]
        assert_eq!(out.row(2).to_vec(), vec![1.0, 0.0]);
        // 3 → [1, 1]
        assert_eq!(out.row(3).to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_binary_encoder_five_categories() {
        let enc = BinaryEncoder::<f64>::new();
        let x = array![[0usize], [1], [2], [3], [4]];
        let fitted = enc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // 5 categories → ceil(log2(5)) = 3 columns
        assert_eq!(out.ncols(), 3);
        // 0 → [0, 0, 0]
        assert_eq!(out.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
        // 4 → [1, 0, 0]
        assert_eq!(out.row(4).to_vec(), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_binary_encoder_single_category() {
        let enc = BinaryEncoder::<f64>::new();
        let x = array![[0usize], [0], [0]];
        let fitted = enc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // 1 category → 1 binary column (always 0)
        assert_eq!(out.ncols(), 1);
        for i in 0..3 {
            assert_eq!(out[[i, 0]], 0.0);
        }
    }

    #[test]
    fn test_binary_encoder_two_categories() {
        let enc = BinaryEncoder::<f64>::new();
        let x = array![[0usize], [1]];
        let fitted = enc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // 2 categories → ceil(log2(2)) = 1 column
        assert_eq!(out.ncols(), 1);
        assert_eq!(out[[0, 0]], 0.0);
        assert_eq!(out[[1, 0]], 1.0);
    }

    #[test]
    fn test_binary_encoder_multi_feature() {
        let enc = BinaryEncoder::<f64>::new();
        // Feature 0: 3 categories → 2 digits
        // Feature 1: 2 categories → 1 digit
        let x = array![[0usize, 0], [1, 1], [2, 0]];
        let fitted = enc.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_output_features(), 3); // 2 + 1
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 3);
    }

    #[test]
    fn test_binary_encoder_n_binary_digits() {
        assert_eq!(n_binary_digits(1), 1);
        assert_eq!(n_binary_digits(2), 1);
        assert_eq!(n_binary_digits(3), 2);
        assert_eq!(n_binary_digits(4), 2);
        assert_eq!(n_binary_digits(5), 3);
        assert_eq!(n_binary_digits(8), 3);
        assert_eq!(n_binary_digits(9), 4);
    }

    #[test]
    fn test_binary_encoder_fit_transform() {
        let enc = BinaryEncoder::<f64>::new();
        let x = array![[0usize], [1], [2], [3]];
        let out: Array2<f64> = enc.fit_transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
    }

    #[test]
    fn test_binary_encoder_zero_rows_error() {
        let enc = BinaryEncoder::<f64>::new();
        let x: Array2<usize> = Array2::zeros((0, 2));
        assert!(enc.fit(&x, &()).is_err());
    }

    #[test]
    fn test_binary_encoder_out_of_range_error() {
        let enc = BinaryEncoder::<f64>::new();
        let x_train = array![[0usize], [1]]; // max category = 1
        let fitted = enc.fit(&x_train, &()).unwrap();
        let x_bad = array![[2usize]]; // category 2 not seen
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_binary_encoder_shape_mismatch_error() {
        let enc = BinaryEncoder::<f64>::new();
        let x_train = array![[0usize, 1], [1, 0]];
        let fitted = enc.fit(&x_train, &()).unwrap();
        let x_bad = array![[0usize]]; // wrong number of columns
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_binary_encoder_unfitted_error() {
        let enc = BinaryEncoder::<f64>::new();
        let x = array![[0usize]];
        assert!(enc.transform(&x).is_err());
    }

    #[test]
    fn test_binary_encoder_accessors() {
        let enc = BinaryEncoder::<f64>::new();
        let x = array![[0usize], [1], [2], [3]];
        let fitted = enc.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_categories(), &[4]);
        assert_eq!(fitted.n_digits(), &[2]);
        assert_eq!(fitted.n_output_features(), 2);
    }

    #[test]
    fn test_binary_encoder_eight_categories() {
        let enc = BinaryEncoder::<f64>::new();
        let x = array![[0usize], [1], [2], [3], [4], [5], [6], [7]];
        let fitted = enc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // 8 categories → ceil(log2(8)) = 3 columns
        assert_eq!(out.ncols(), 3);
        // 7 → [1, 1, 1]
        assert_eq!(out.row(7).to_vec(), vec![1.0, 1.0, 1.0]);
        // 5 → [1, 0, 1]
        assert_eq!(out.row(5).to_vec(), vec![1.0, 0.0, 1.0]);
    }
}
