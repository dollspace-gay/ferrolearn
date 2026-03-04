//! Dataset trait and implementations for common array types.
//!
//! The [`Dataset`] trait provides a uniform interface for querying the
//! shape of tabular data, regardless of the underlying storage format
//! (dense or sparse). Implementations are provided for
//! [`ndarray::Array2<f32>`] and [`ndarray::Array2<f64>`].

use ndarray::Array2;
use num_traits::Float;

/// A trait for types that represent tabular datasets.
///
/// Provides basic shape information that algorithms need to validate
/// inputs and allocate output buffers.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use ferrolearn_core::Dataset;
///
/// let data = Array2::<f64>::zeros((100, 10));
/// assert_eq!(data.n_samples(), 100);
/// assert_eq!(data.n_features(), 10);
/// assert!(!data.is_sparse());
/// ```
pub trait Dataset {
    /// Returns the number of samples (rows) in the dataset.
    fn n_samples(&self) -> usize;

    /// Returns the number of features (columns) in the dataset.
    fn n_features(&self) -> usize;

    /// Returns `true` if the dataset uses a sparse representation.
    fn is_sparse(&self) -> bool;
}

/// Blanket implementation of [`Dataset`] for `ndarray::Array2<F>` where
/// `F` is any floating-point type satisfying the ferrolearn float bound.
impl<F> Dataset for Array2<F>
where
    F: Float + Send + Sync + 'static,
{
    fn n_samples(&self) -> usize {
        self.nrows()
    }

    fn n_features(&self) -> usize {
        self.ncols()
    }

    fn is_sparse(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array2_f64_dataset() {
        let data = Array2::<f64>::zeros((50, 12));
        assert_eq!(data.n_samples(), 50);
        assert_eq!(data.n_features(), 12);
        assert!(!data.is_sparse());
    }

    #[test]
    fn test_array2_f32_dataset() {
        let data = Array2::<f32>::zeros((200, 5));
        assert_eq!(data.n_samples(), 200);
        assert_eq!(data.n_features(), 5);
        assert!(!data.is_sparse());
    }

    #[test]
    fn test_empty_array_dataset() {
        let data = Array2::<f64>::zeros((0, 0));
        assert_eq!(data.n_samples(), 0);
        assert_eq!(data.n_features(), 0);
    }

    #[test]
    fn test_single_sample_dataset() {
        let data = Array2::<f64>::zeros((1, 100));
        assert_eq!(data.n_samples(), 1);
        assert_eq!(data.n_features(), 100);
    }

    #[test]
    fn test_dataset_trait_is_object_safe() {
        // Verify Dataset can be used as a trait object.
        let data = Array2::<f64>::zeros((10, 3));
        let _: &dyn Dataset = &data;
    }
}
