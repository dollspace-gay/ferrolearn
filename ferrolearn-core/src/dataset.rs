//! Dataset trait and implementations for common array types.
//!
//! The [`Dataset`] trait provides a uniform interface for querying the
//! shape of tabular data, regardless of the underlying storage format
//! (dense or sparse). Implementations are provided for the transitional
//! `ndarray::Array2<F>` substrate and the destination ferray substrate
//! (`ferray::aliases::Array2<F>`, per R-SUBSTRATE-1).
//!
//! ## REQ status
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (shape introspection) | SHIPPED | `impl Dataset for ndarray::Array2 in dataset.rs` (`n_samples` → `self.nrows()`, `n_features` → `self.ncols()`), mirroring `_num_samples` (`validation.py:364`, dense path `return x.shape[0]` at `:388`) and `_num_features` (`validation.py:311`, dense path `return X.shape[1]` at `:346`). Cross-crate production consumer: `impl Dataset for CsrMatrix in ferrolearn-sparse/src/csr.rs`. |
//! | REQ-2 (is_sparse) | SHIPPED | `is_sparse` returns `false` in `impl Dataset for ndarray::Array2 in dataset.rs` and `true` in `impl Dataset for CsrMatrix in ferrolearn-sparse/src/csr.rs`; the dense/sparse split across the two impls is the production discrimination. |
//! | REQ-3 (ferray substrate) | SHIPPED | `impl Dataset for ferray::aliases::Array2 in dataset.rs` (`n_samples`/`n_features` via `self.shape()[0]`/`self.shape()[1]`, `is_sparse` → `false`), on ferray's array type (`.shape() -> &[usize]`), per R-SUBSTRATE-1. `ferrolearn-core` declares `ferray = { workspace = true }`. |
//! | REQ-4 (validation surface) | NOT-STARTED | open prereq blocker #356. `check_array` (`validation.py:718`), `check_X_y` (`validation.py:1154`), `check_consistent_length` (`validation.py:436`), and `assert_all_finite` (`validation.py:175`) have no counterpart in `ferrolearn-core`. |

use ferray::aliases::Array2 as FerrayArray2;
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

/// Implementation of [`Dataset`] for the ferray dense array type
/// (`ferray::aliases::Array2<F>`), the destination numpy substrate
/// (R-SUBSTRATE-1).
///
/// ferray's array exposes shape via `.shape() -> &[usize]`; the dense
/// 2-D path mirrors scikit-learn's `_num_samples`
/// (`sklearn/utils/validation.py:364`, returning `x.shape[0]` at `:388`)
/// and `_num_features` (`validation.py:311`, returning `X.shape[1]` at
/// `:346`). The element bound is `F: ferray::Element` — ferray's
/// `Array<T: Element, D>` is only defined for `Element` types, a sealed
/// trait that `num_traits::Float` does not imply, so the existing
/// `Float + Send + Sync + 'static` bound cannot name the ferray type;
/// `Element` itself already requires `Send + Sync + 'static`.
impl<F> Dataset for FerrayArray2<F>
where
    F: ferray::Element,
{
    fn n_samples(&self) -> usize {
        self.shape()[0]
    }

    fn n_features(&self) -> usize {
        self.shape()[1]
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

    // ----- ferray substrate (REQ-3) -----
    //
    // These exercise the real ferray array type via the umbrella crate.
    // Expected shapes are derived from sklearn's dense shape contract:
    // `_num_samples(X) == X.shape[0]` (`validation.py:388`) and
    // `_num_features(X) == X.shape[1]` (`validation.py:346`). The test
    // fns return `FerrayResult` so the `?` operator propagates the
    // fallible ferray constructor without an `.unwrap()` escape hatch.

    use ferray::aliases::Array2 as FArray2;
    use ferray::{FerrayResult, Ix2, zeros as ferray_zeros};

    #[test]
    fn test_ferray_array2_f64_dataset() -> FerrayResult<()> {
        let data: FArray2<f64> = ferray_zeros(Ix2::new([50, 12]))?;
        assert_eq!(data.n_samples(), 50);
        assert_eq!(data.n_features(), 12);
        assert!(!data.is_sparse());
        Ok(())
    }

    #[test]
    fn test_ferray_array2_f32_dataset() -> FerrayResult<()> {
        let data: FArray2<f32> = ferray_zeros(Ix2::new([200, 5]))?;
        assert_eq!(data.n_samples(), 200);
        assert_eq!(data.n_features(), 5);
        assert!(!data.is_sparse());
        Ok(())
    }

    #[test]
    fn test_ferray_empty_array_dataset() -> FerrayResult<()> {
        let data: FArray2<f64> = ferray_zeros(Ix2::new([0, 0]))?;
        assert_eq!(data.n_samples(), 0);
        assert_eq!(data.n_features(), 0);
        Ok(())
    }

    #[test]
    fn test_ferray_single_sample_dataset() -> FerrayResult<()> {
        let data: FArray2<f64> = ferray_zeros(Ix2::new([1, 100]))?;
        assert_eq!(data.n_samples(), 1);
        assert_eq!(data.n_features(), 100);
        assert!(!data.is_sparse());
        Ok(())
    }

    #[test]
    fn test_ferray_dataset_trait_is_object_safe() -> FerrayResult<()> {
        let data: FArray2<f64> = ferray_zeros(Ix2::new([10, 3]))?;
        let _: &dyn Dataset = &data;
        Ok(())
    }
}
