//! Coordinate (COO / triplet) sparse matrix format.
//!
//! [`CooMatrix<T>`] is a newtype wrapper around [`sprs::TriMat<T>`]. It is
//! primarily useful for incrementally building a sparse matrix before
//! converting it to [`CsrMatrix`](crate::CsrMatrix) or
//! [`CscMatrix`](crate::CscMatrix) for computation.
//!
//! ## REQ status
//!
//! Mirrors `scipy.sparse.coo_matrix` (`scipy/sparse/_coo.py`; live oracle scipy
//! 1.17, deterministic construction/conversion). Design doc: `.design/sparse/coo.md`
//! (8 REQs). Every REQ is BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a
//! concrete blocker). Behavior is oracle-verified vs the live scipy (R-CHAR-3) —
//! see `tests/divergence_coo.rs`.
//!
//! **5 SHIPPED / 3 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-CONSTRUCT (from_triplets/push/shape) | SHIPPED | `from_triplets`/`new`/`with_capacity`/`push` mirror `coo_matrix((data,(row,col)),shape)`; `to_dense()` matches scipy `.toarray()`; `n_rows`/`n_cols` match `.shape`. Guards `coo_construct_to_dense_matches_scipy_toarray`/`coo_shape_matches_scipy`. |
//! | REQ-TOARRAY-DUP (duplicate summing) | SHIPPED | `to_dense()` sums duplicate entries at the same position, matching scipy `.toarray()` (e.g. `(0,0)` twice → 3). Guards `coo_to_dense_duplicate_summed_matches_scipy`/`coo_push_duplicate_nnz_counts_both`. |
//! | REQ-NNZ (stored-count semantics) | SHIPPED | `nnz()` counts STORED entries incl. duplicates (matching scipy coo `.nnz = len(data)`); CSR conversion coalesces (nnz 3→2). Guard `coo_nnz_counts_duplicates_csr_coalesces`. |
//! | REQ-ERR (construction validation) | SHIPPED | `from_triplets`/`push` return `Err(FerroError)` on out-of-bounds index / mismatched lengths, at the same point scipy raises `ValueError`. Guards `coo_from_triplets_*_is_err`/`coo_push_out_of_bounds_is_err`. |
//! | REQ-CONSUMER (production consumer) | SHIPPED | consumed in-crate by `csr.rs`/`csc.rs` (COO→CSR/CSC conversion) and `helpers.rs` (`eye`/`diags`/`hstack`/`vstack` build via COO) — real non-test callers (S5 crate boundary). |
//! | REQ-API-ACCESSORS (shape/data/row/col) | NOT-STARTED | no `.shape` tuple, no public `.data`/`.row`/`.col` (behind `inner()`). Blocker #1996. |
//! | REQ-MISSING-METHODS (COO methods) | NOT-STARTED | no `.tocsr`/`.tocsc`/`.transpose`/`.sum`/`.diagonal`/`.sum_duplicates`/`.eliminate_zeros`/`.multiply`/`.dot`/arithmetic/`.astype`/`.copy` as `CooMatrix` methods. Blocker #1997. |
//! | REQ-FERRAY (ferray sparse substrate) | NOT-STARTED | `sprs::TriMat` + `ndarray` vs ferray's sparse COO analog (R-SUBSTRATE-1; ferray has no sparse layer yet). Blocker #1998. |

use ferrolearn_core::FerroError;
use ndarray::Array2;
use num_traits::Zero;
use sprs::{SpIndex, TriMat};

/// Coordinate-format (COO / triplet) sparse matrix.
///
/// Stores non-zero entries as `(row, col, value)` triplets. Duplicate entries
/// at the same position are **summed** during conversion to CSR/CSC. This
/// format is most convenient for construction; prefer [`CsrMatrix`](crate::CsrMatrix)
/// or [`CscMatrix`](crate::CscMatrix) for arithmetic.
///
/// # Type Parameter
///
/// `T` — the scalar type stored in the matrix. No additional bounds are
/// required for construction; conversion methods impose their own bounds.
#[derive(Debug)]
pub struct CooMatrix<T> {
    inner: TriMat<T>,
}

impl<T: Clone> Clone for CooMatrix<T> {
    /// Clone by rebuilding the inner [`sprs::TriMat`] from raw components.
    ///
    /// [`sprs::TriMat`] does not implement `Clone` generically, so we
    /// reconstruct it from the stored row indices, column indices, and data.
    fn clone(&self) -> Self {
        Self {
            inner: TriMat::from_triplets(
                (self.n_rows(), self.n_cols()),
                self.inner.row_inds().to_vec(),
                self.inner.col_inds().to_vec(),
                self.inner.data().to_vec(),
            ),
        }
    }
}

impl<T> CooMatrix<T> {
    /// Create an empty COO matrix with the given shape.
    ///
    /// # Arguments
    ///
    /// * `n_rows` — number of rows.
    /// * `n_cols` — number of columns.
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            inner: TriMat::new((n_rows, n_cols)),
        }
    }

    /// Create a COO matrix with the given shape and pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `n_rows` — number of rows.
    /// * `n_cols` — number of columns.
    /// * `capacity` — expected number of non-zero entries.
    pub fn with_capacity(n_rows: usize, n_cols: usize, capacity: usize) -> Self {
        Self {
            inner: TriMat::with_capacity((n_rows, n_cols), capacity),
        }
    }

    /// Build a [`CooMatrix`] from raw triplet components.
    ///
    /// All three slices must have the same length. Row indices must be less
    /// than `n_rows`; column indices must be less than `n_cols`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if the slice lengths differ or
    /// if any index is out of bounds.
    pub fn from_triplets(
        n_rows: usize,
        n_cols: usize,
        row_inds: Vec<usize>,
        col_inds: Vec<usize>,
        data: Vec<T>,
    ) -> Result<Self, FerroError> {
        if row_inds.len() != col_inds.len() || row_inds.len() != data.len() {
            return Err(FerroError::InvalidParameter {
                name: "triplet arrays".into(),
                reason: format!(
                    "row_inds ({}), col_inds ({}), and data ({}) must all have the same length",
                    row_inds.len(),
                    col_inds.len(),
                    data.len()
                ),
            });
        }
        if let Some(&r) = row_inds.iter().find(|&&r| r >= n_rows) {
            return Err(FerroError::InvalidParameter {
                name: "row_inds".into(),
                reason: format!("index {r} is out of bounds for n_rows={n_rows}"),
            });
        }
        if let Some(&c) = col_inds.iter().find(|&&c| c >= n_cols) {
            return Err(FerroError::InvalidParameter {
                name: "col_inds".into(),
                reason: format!("index {c} is out of bounds for n_cols={n_cols}"),
            });
        }
        Ok(Self {
            inner: TriMat::from_triplets((n_rows, n_cols), row_inds, col_inds, data),
        })
    }

    /// Append a single non-zero entry `(row, col, value)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `row >= n_rows()` or
    /// `col >= n_cols()`.
    pub fn push(&mut self, row: usize, col: usize, value: T) -> Result<(), FerroError> {
        if row >= self.n_rows() {
            return Err(FerroError::InvalidParameter {
                name: "row".into(),
                reason: format!("index {row} is out of bounds for n_rows={}", self.n_rows()),
            });
        }
        if col >= self.n_cols() {
            return Err(FerroError::InvalidParameter {
                name: "col".into(),
                reason: format!("index {col} is out of bounds for n_cols={}", self.n_cols()),
            });
        }
        self.inner.add_triplet(row, col, value);
        Ok(())
    }

    /// Returns the number of rows.
    pub fn n_rows(&self) -> usize {
        self.inner.rows()
    }

    /// Returns the number of columns.
    pub fn n_cols(&self) -> usize {
        self.inner.cols()
    }

    /// Returns the number of stored non-zero entries (counting duplicates).
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns a reference to the underlying [`sprs::TriMat<T>`].
    pub fn inner(&self) -> &TriMat<T> {
        &self.inner
    }

    /// Consume this matrix and return the underlying [`sprs::TriMat<T>`].
    pub fn into_inner(self) -> TriMat<T> {
        self.inner
    }
}

impl<T> CooMatrix<T>
where
    T: Clone + Zero + num_traits::NumAssign + 'static,
{
    /// Convert this COO matrix to a dense [`Array2<T>`].
    ///
    /// Duplicate entries at the same position are summed.
    pub fn to_dense(&self) -> Array2<T> {
        let mut out = Array2::<T>::zeros((self.n_rows(), self.n_cols()));
        for (val, (r, c)) in self.inner.triplet_iter() {
            out[[r.index(), c.index()]] += val.clone();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_new() {
        let m: CooMatrix<f64> = CooMatrix::new(4, 5);
        assert_eq!(m.n_rows(), 4);
        assert_eq!(m.n_cols(), 5);
        assert_eq!(m.nnz(), 0);
    }

    #[test]
    fn test_coo_push() {
        let mut m: CooMatrix<f64> = CooMatrix::new(3, 3);
        m.push(0, 0, 1.0).unwrap();
        m.push(1, 2, 5.0).unwrap();
        assert_eq!(m.nnz(), 2);
    }

    #[test]
    fn test_coo_push_out_of_bounds() {
        let mut m: CooMatrix<f64> = CooMatrix::new(2, 2);
        assert!(m.push(2, 0, 1.0).is_err());
        assert!(m.push(0, 2, 1.0).is_err());
    }

    #[test]
    fn test_coo_from_triplets_mismatch() {
        let result = CooMatrix::<f64>::from_triplets(3, 3, vec![0, 1], vec![0], vec![1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_from_triplets_out_of_bounds() {
        let result = CooMatrix::<f64>::from_triplets(2, 2, vec![3], vec![0], vec![1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_to_dense() {
        let mut m: CooMatrix<f64> = CooMatrix::new(2, 3);
        m.push(0, 1, 3.0).unwrap();
        m.push(1, 0, 7.0).unwrap();
        let d = m.to_dense();
        assert_eq!(d[[0, 1]], 3.0);
        assert_eq!(d[[1, 0]], 7.0);
        assert_eq!(d[[0, 0]], 0.0);
    }

    #[test]
    fn test_coo_to_dense_duplicate_summed() {
        let mut m: CooMatrix<f64> = CooMatrix::new(2, 2);
        m.push(0, 0, 1.0).unwrap();
        m.push(0, 0, 2.0).unwrap(); // duplicate — should sum to 3.0
        let d = m.to_dense();
        assert_eq!(d[[0, 0]], 3.0);
    }

    #[test]
    fn test_coo_clone() {
        let mut m: CooMatrix<f64> = CooMatrix::new(2, 2);
        m.push(0, 0, 5.0).unwrap();
        let m2 = m.clone();
        assert_eq!(m2.nnz(), 1);
        assert_eq!(m2.n_rows(), 2);
        assert_eq!(m2.n_cols(), 2);
    }
}
