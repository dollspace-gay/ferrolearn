//! Compressed Sparse Column (CSC) matrix format.
//!
//! [`CscMatrix<T>`] is a newtype wrapper around [`sprs::CsMat<T>`] in CSC
//! storage. CSC matrices are efficient for column-wise operations and are the
//! natural choice when algorithms need to iterate over columns.
//!
//! ## REQ status
//!
//! Mirrors `scipy.sparse.csc_matrix` (`scipy/sparse/_csc.py`; live oracle scipy
//! 1.17, deterministic) — the column-symmetric analog of [`CsrMatrix`].
//! Design doc: `.design/sparse/csc.md` (14 REQs). Every REQ is BINARY (R-DEFER-2):
//! SHIPPED or NOT-STARTED (with a concrete blocker). Behavior is oracle-verified
//! vs the live scipy (R-CHAR-3) — see `tests/divergence_csc.rs`.
//!
//! **14 SHIPPED / 1 NOT-STARTED** (REQ-MISSING-INDEX is a SPLIT: element access
//! `get(i,j)`, rows/cols `getrow`/`getcol`, and the maintenance surface
//! `max`/`min`/`astype`/`copy`/`eliminate_zeros`/`power` all SHIPPED;
//! `sort_indices`/`sum_duplicates`/`argmax` remain NOT-STARTED).
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-CONSTRUCT-CONVERT | SHIPPED | `from_coo`/`from_dense`/`from_csr` + `to_dense`/`to_coo`/`to_csr` + `nnz` mirror `csc_matrix(...)`/`.toarray()`/`.tocsr()`/`.tocoo()`/`.nnz` (oracle nnz=5; round-trips). Guards `csc_from_*`/`csc_to_csr_roundtrip_matches_scipy`. |
//! | REQ-MATVEC | SHIPPED | `mul_vec(v)` == scipy `A@v` (`[7,6,19]`). Guard `csc_mul_vec_matches_scipy`. |
//! | REQ-ADD | SHIPPED | `add(&B)` == scipy `A+B` (elementwise). Guards `csc_add_self_matches_scipy`/`csc_add_other_matches_scipy`. |
//! | REQ-SCALAR-MUL | SHIPPED | `mul_scalar(s)`/`scale(s)` == scipy `A*s`. Guard `csc_scalar_mul_matches_scipy`. |
//! | REQ-COL-SLICE | SHIPPED | `col_slice(a,b)` == scipy `A[:,a:b]` (`[[1,0],[0,3],[4,0]]`; method vs Python-slice API, column analog of CSR `row_slice`). Guard `csc_col_slice_matches_scipy`. |
//! | REQ-ERR | SHIPPED | `add`/`mul_vec` return `Err(FerroError::ShapeMismatch)` where scipy raises `ValueError`. Guards `csc_*_shape_mismatch_is_err`. |
//! | REQ-CONSUMER | SHIPPED | in-crate CSR↔CSC conversion consumer (`csr.rs` `from_csc`/`to_csc`) + `lib.rs` re-export (grandfathered boundary type, R-DEFER-1/S5). NOTE: no cross-crate ESTIMATOR consumer (weaker than CSR's neighbors/graph.rs). |
//! | REQ-MISSING-MATMUL | SHIPPED (#2008) | `matmul(&B)` returns the `(self.n_rows, rhs.n_cols)` sparse-sparse product `A@B` via the sprs product operator `(&self.inner * &rhs.inner).to_csc()` (the column-symmetric analog of the CSR `matmul`, #2000), mirroring scipy `_matmul_sparse` (`_compressed.py:415`, SMMP). Shape-checks `self.n_cols() == rhs.n_rows()` first (mismatch → `Err(FerroError::ShapeMismatch)`, scipy `ValueError: dimension mismatch`). Live oracle: `A.matmul(B).toarray()`=`[[1,1,2],[0,3,3],[4,4,5]]`; non-square `A.matmul(C)` for `C=[[1,2],[3,4],[5,6]]`=`[[11,14],[9,12],[29,38]]` shape `(3,2)`. Guards `csc_matmul_matches_scipy`/`csc_matmul_non_square`/`csc_matmul_shape_mismatch_is_err`. |
//! | REQ-MISSING-TRANSPOSE | SHIPPED (#2009) | `transpose()` returns a `(n_cols, n_rows)` CSC of `Aᵀ` via sprs `transpose_view().to_csc()`, mirroring scipy `A.T` (`_csc.py:20` — CSR view of the same buffers, here materialized as CSC). Live oracle: `A.T.toarray()`=`[[1,0,4],[0,3,0],[2,0,5]]`; non-square `B=[[1,2,3],[4,5,6]]` -> `B.T.toarray()`=`[[1,4],[2,5],[3,6]]` shape `(3,2)`; double-transpose round-trips. Guards `csc_transpose_matches_scipy`/`csc_transpose_non_square`/`csc_transpose_twice_roundtrip`. |
//! | REQ-MISSING-REDUCE | SHIPPED (#2010) | `sum`/`sum_axis0`/`sum_axis1`/`diagonal` mirror scipy `.sum(axis=)` (`_compressed.py:492`) + `.diagonal()` (`_compressed.py:476`). Live oracle: `A.sum()`=15, `A.sum(axis=0)`=[5,3,7], `A.sum(axis=1)`=[3,3,9], `A.diagonal()`=[1,3,5], non-square `B.diagonal()`=[1,5]. Guards `csc_sum_matches_scipy`/`csc_sum_axis0_matches_scipy`/`csc_sum_axis1_matches_scipy`/`csc_diagonal_matches_scipy`/`csc_diagonal_non_square`. |
//! | REQ-MISSING-ELEMENTWISE | SHIPPED (#2011) | `multiply(&B)` (element-wise Hadamard, INTERSECTION sparsity via sprs `binop::mul_mat_same_storage`) mirrors scipy `multiply` (`_base.py:490`, `_elmul_`); `sub(&B)` (`A-B`, UNION sparsity via sprs `&CsMat - &CsMat`) mirrors scipy `_sub_sparse` (`_compressed.py:260`). Both shape-check first (`Err(FerroError::ShapeMismatch)`) like `add`. Live oracle: `A.multiply(B).toarray()`=`[[1,0,0],[0,3,0],[0,0,5]]`, `(A-B).toarray()`=`[[0,-1,2],[0,2,-1],[4,0,4]]`. Guards `csc_multiply_matches_scipy`/`csc_sub_matches_scipy`/`csc_multiply_shape_mismatch_is_err`/`csc_sub_shape_mismatch_is_err`. Sub-note: `.power` (`_data.py:99`) still NOT-STARTED. |
//! | REQ-MISSING-INDEX (element access `A[i,j]`) | SHIPPED (#2012) | `get(i,j) -> Result<T, FerroError>` returns the stored value at `(i,j)` or `T::zero()` if structurally absent (sprs `CsMat::get(i,j).copied().unwrap_or_else(T::zero)`), with an out-of-bounds `i`/`j` → `Err(FerroError::InvalidParameter)` — the column-symmetric port of the CSR `get` (sprs `get(i,j)` indexes by `(row,col)` regardless of storage), mirroring scipy `A[i,j]` (`IndexMixin.__getitem__` -> `_get_intXint`, `_index.py:29`). Live oracle (R-CHAR-3): `A=[[1,0,2],[0,3,0],[4,0,5]]` → `A[1,1]=3`, `A[0,1]=0` (absent), `A[0,0]=1`, `A[0,2]=2`, `A[2,0]=4`; out-of-bounds → `Err`. Guards `csc_get_element_matches_scipy`/`csc_get_absent_is_zero`/`csc_get_out_of_bounds_is_err`. |
//! | REQ-MISSING-INDEX (rows/cols) | SHIPPED (#2012) | `getcol(j)` returns column `j` as a `(n_rows, 1)` CSC (single-column case of `col_slice`, delegates to `self.col_slice(j, j + 1)`), mirroring scipy `getcol(j)` (`_matrix.py:104` → `_getcol`, "(m x 1) column vector"); `getrow(i)` returns row `i` as a `(1, n_cols)` CSC via `self.transpose().getcol(i)?.transpose()`, mirroring scipy `getrow(i)` (`_matrix.py:110` → `_getrow`, "(1 x n) row vector"). Both bounds-check (`j >= n_cols()`/`i >= n_rows()` → `Err(FerroError::InvalidParameter)`, scipy `IndexError`, R-DEV-2). CSC is column-natural, so `getcol` delegates to `col_slice` (the mirror of CSR's `getrow`→`row_slice`). Live oracle (R-CHAR-3): `A.getrow(0).toarray()`=`[[1,0,2]]`, `A.getrow(1).toarray()`=`[[0,3,0]]`, `A.getcol(0).toarray()`=`[[1],[0],[4]]`, `A.getcol(2).toarray()`=`[[2],[0],[5]]`. Guards `csc_getrow_matches_scipy`/`csc_getcol_matches_scipy`/`csc_getrow_getcol_out_of_bounds_is_err`. |
//! | REQ-MISSING-INDEX (maintenance) | SHIPPED (#2012) | `max()`/`min()` (`T: Copy + Zero + PartialOrd`) fold over [`data()`](CscMatrix::data) and, when not fully dense, fold an implicit zero, mirroring scipy `_minmax_mixin._min_or_max` `axis=None` (`_data.py:208`-`:224`); `astype(cast)` (closure-based dtype cast, `_data.py:69`, R-DEV-4) and `copy()` (= `clone`, `_data.py:94`) preserve `(indptr, indices, shape)`; `eliminate_zeros()` walks COLUMNS via the column-pointer `indptr`, dropping `val == 0` and rebuilding the triple (`_compressed.py:1025`, functional/new-matrix R-DEV-4); `power(n)` (`T: Float`) maps `data()` through `powf(n)` (`_data.py:99`). Storage-agnostic ports of the CSR maintenance surface; for CSC `indices` are ROW indices and `indptr` the COLUMN pointer. Live oracle (R-CHAR-3): `csc_matrix(diag(-3,-1,-5)).max()==0`, `.min()==-5`; `csc_matrix(diag(3,1,5)).max()==5`, `.min()==0`; `astype(int64)` of `[3.7,-2.9,5.0]` -> `[3,-2,5]` (truncation); `eliminate_zeros` of CSC `data=[3,0,5]`/`indices=[0,1,2]`/`indptr=[0,1,2,3]` -> `nnz=2`, `data=[3,5]`, `indices=[0,2]`, `indptr=[0,1,1,2]`; `power(2)` of `[2,-3]` -> `[4,9]`, `power(3)` -> `[8,-27]`. Guards `csc_max_min_folds_implicit_zero`/`csc_astype_truncates`/`csc_copy_preserves_structure`/`csc_eliminate_zeros_matches_scipy`/`csc_power_matches_scipy`. Still NOT-STARTED: `sort_indices`/`sum_duplicates`/`argmax`. |
//! | REQ-API-ACCESSORS | SHIPPED (#2013) | first-class `shape()`/`data()`/`indices()`/`indptr()` accessors mirror scipy `.shape` (`_compressed.py:38`) and `.data`/`.indices`/`.indptr` (`_compressed.py:76-78`), the same CSC `(data, indices, indptr)` triple. `shape()` → `(n_rows, n_cols)`; `data()` → `&[T]` (`inner.data()`); `indices()` → `&[usize]` (CSC ROW indices, `inner.indices()`); `indptr()` → owned `Vec<usize>` (COLUMN pointers, `inner.indptr().raw_storage().to_vec()` — owned because the sprs `IndPtrView` accessor borrows a temporary). Column-symmetric port of the CSR accessors (#2005). Live oracle (R-CHAR-3): `A=[[1,0,2],[0,3,0],[4,0,5]]` → `shape=(3,3)`, `data=[1,4,3,2,5]` (column-major), `indices=[0,2,1,0,2]` (row indices), `indptr=[0,2,3,5]` (column pointer). Guard `csc_shape_data_indices_indptr_match_scipy`. |
//! | REQ-FERRAY | NOT-STARTED | `sprs::CsMat` + `ndarray` vs ferray's sparse CSC analog (R-SUBSTRATE-1). Blocker #2014. |

use std::ops::{Add, AddAssign, Mul, MulAssign};

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Zero;
use sprs::CsMat;

use crate::coo::CooMatrix;
use crate::csr::CsrMatrix;

/// Compressed Sparse Column (CSC) sparse matrix.
///
/// Stores non-zero entries in column-major order using three arrays: `indptr`
/// (column pointer array of length `n_cols + 1`), `indices` (row indices of
/// each non-zero), and `data` (values of each non-zero).
///
/// # Type Parameter
///
/// `T` — the scalar element type.
#[derive(Debug, Clone)]
pub struct CscMatrix<T> {
    inner: CsMat<T>,
}

impl<T> CscMatrix<T>
where
    T: Clone,
{
    /// Construct a CSC matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `n_rows` — number of rows.
    /// * `n_cols` — number of columns.
    /// * `indptr` — column pointer array of length `n_cols + 1`.
    /// * `indices` — row index of each non-zero entry.
    /// * `data` — value of each non-zero entry.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if the data is structurally
    /// invalid (wrong lengths, out-of-bound indices, unsorted inner indices).
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<T>,
    ) -> Result<Self, FerroError> {
        CsMat::try_new_csc((n_rows, n_cols), indptr, indices, data)
            .map(|inner| Self { inner })
            .map_err(|(_, _, _, err)| FerroError::InvalidParameter {
                name: "CscMatrix raw components".into(),
                reason: err.to_string(),
            })
    }

    /// Build a [`CscMatrix`] from a pre-validated [`sprs::CsMat<T>`] in CSC storage.
    ///
    /// This is used internally for format conversions.
    pub(crate) fn from_inner(inner: CsMat<T>) -> Self {
        debug_assert!(inner.is_csc(), "inner matrix must be in CSC storage");
        Self { inner }
    }

    /// Returns the number of rows.
    pub fn n_rows(&self) -> usize {
        self.inner.rows()
    }

    /// Returns the number of columns.
    pub fn n_cols(&self) -> usize {
        self.inner.cols()
    }

    /// Returns the number of stored non-zero entries.
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns a reference to the underlying [`sprs::CsMat<T>`].
    pub fn inner(&self) -> &CsMat<T> {
        &self.inner
    }

    /// Consume this matrix and return the underlying [`sprs::CsMat<T>`].
    pub fn into_inner(self) -> CsMat<T> {
        self.inner
    }

    /// Returns the matrix shape as a `(n_rows, n_cols)` tuple.
    ///
    /// Mirrors scipy `csc_matrix.shape` (the `self._shape` tuple,
    /// `scipy/sparse/_compressed.py:38`), which is the `(M, N)` dimension pair.
    /// Equivalent to `(self.n_rows(), self.n_cols())`. Live oracle:
    /// `sp.csc_matrix([[1,0,2],[0,3,0],[4,0,5]]).shape == (3, 3)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows(), self.n_cols())
    }

    /// Returns the stored non-zero values, one per stored entry, in column-major
    /// CSC order.
    ///
    /// Mirrors scipy `csc_matrix.data` (`scipy/sparse/_compressed.py:76-78`),
    /// the `data` array of the CSC `(data, indices, indptr)` triple ferrolearn
    /// stores identically. Length equals [`nnz`](Self::nnz). Live oracle:
    /// `sp.csc_matrix([[1,0,2],[0,3,0],[4,0,5]]).data == [1,4,3,2,5]` (column-major).
    #[must_use]
    pub fn data(&self) -> &[T] {
        self.inner.data()
    }

    /// Returns the row index of each stored non-zero entry, aligned with
    /// [`data`](Self::data), in column-major CSC order.
    ///
    /// Mirrors scipy `csc_matrix.indices` (`scipy/sparse/_compressed.py:76-78`),
    /// the `indices` array of the CSC `(data, indices, indptr)` triple — for CSC
    /// these are the ROW indices (the row/column roles are swapped relative to
    /// CSR). Length equals [`nnz`](Self::nnz). Live oracle:
    /// `sp.csc_matrix([[1,0,2],[0,3,0],[4,0,5]]).indices == [0,2,1,0,2]`.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        self.inner.indices()
    }

    /// Returns the column pointer array, of length `n_cols + 1`.
    ///
    /// Mirrors scipy `csc_matrix.indptr` (`scipy/sparse/_compressed.py:76-78`),
    /// the `indptr` array of the CSC `(data, indices, indptr)` triple ferrolearn
    /// stores identically: for CSC `indptr` is the COLUMN pointer, so
    /// `indptr[j]..indptr[j+1]` is the slice of [`data`](Self::data) /
    /// [`indices`](Self::indices) belonging to column `j`.
    ///
    /// Returns an owned `Vec<usize>` rather than a borrowed slice (unlike
    /// [`data`](Self::data) / [`indices`](Self::indices)) because the sprs
    /// `CsMat::indptr` accessor yields an owned `IndPtrView` value whose
    /// `raw_storage()` slice borrows that temporary, so no `&[usize]` tied to
    /// `&self` can be returned through the public sprs API; the column-pointer
    /// storage is materialized via `IndPtrView::raw_storage().to_vec()`. Live
    /// oracle: `sp.csc_matrix([[1,0,2],[0,3,0],[4,0,5]]).indptr == [0,2,3,5]`.
    #[must_use]
    pub fn indptr(&self) -> Vec<usize> {
        self.inner.indptr().raw_storage().to_vec()
    }

    /// Construct a [`CscMatrix`] from a [`CooMatrix`] by converting to CSC.
    ///
    /// Duplicate entries at the same position are summed.
    ///
    /// # Errors
    ///
    /// This conversion is always successful for structurally valid inputs.
    pub fn from_coo(coo: &CooMatrix<T>) -> Result<Self, FerroError>
    where
        T: Clone + Add<Output = T> + 'static,
    {
        let inner: CsMat<T> = coo.inner().to_csc();
        Ok(Self { inner })
    }

    /// Construct a [`CscMatrix`] from a [`CsrMatrix`].
    ///
    /// # Errors
    ///
    /// This conversion is always successful.
    pub fn from_csr(csr: &CsrMatrix<T>) -> Result<Self, FerroError>
    where
        T: Clone + Default + 'static,
    {
        Ok(csr.to_csc())
    }

    /// Convert to [`CsrMatrix`].
    pub fn to_csr(&self) -> CsrMatrix<T>
    where
        T: Clone + Default + 'static,
    {
        // from_csc is infallible for a valid CscMatrix
        CsrMatrix::from_csc(self).unwrap()
    }

    /// Convert to [`CooMatrix`].
    pub fn to_coo(&self) -> CooMatrix<T> {
        let mut coo = CooMatrix::with_capacity(self.n_rows(), self.n_cols(), self.nnz());
        for (val, (r, c)) in &self.inner {
            // indices come from a valid matrix, so push is infallible here
            let _ = coo.push(r, c, val.clone());
        }
        coo
    }

    /// Convert this sparse matrix to a dense [`Array2<T>`].
    pub fn to_dense(&self) -> Array2<T>
    where
        T: Clone + Zero + 'static,
    {
        self.inner.to_dense()
    }

    /// Construct a [`CscMatrix`] from a dense [`Array2<T>`], dropping entries
    /// whose absolute value is less than or equal to `epsilon`.
    pub fn from_dense(dense: &ArrayView2<'_, T>, epsilon: T) -> Self
    where
        T: Copy + Zero + PartialOrd + num_traits::Signed + 'static,
    {
        let inner = CsMat::csc_from_dense(dense.view(), epsilon);
        Self { inner }
    }

    /// Return a new CSC matrix containing only the columns in `start..end`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `start > end` or
    /// `end > n_cols()`.
    pub fn col_slice(&self, start: usize, end: usize) -> Result<CscMatrix<T>, FerroError>
    where
        T: Clone + Default + 'static,
    {
        if start > end {
            return Err(FerroError::InvalidParameter {
                name: "col_slice range".into(),
                reason: format!("start ({start}) must be <= end ({end})"),
            });
        }
        if end > self.n_cols() {
            return Err(FerroError::InvalidParameter {
                name: "col_slice range".into(),
                reason: format!("end ({end}) exceeds n_cols ({})", self.n_cols()),
            });
        }
        let view = self.inner.slice_outer(start..end);
        Ok(Self {
            inner: view.to_owned(),
        })
    }

    /// Scalar multiplication in-place: multiplies every non-zero by `scalar`.
    ///
    /// Requires `T: for<'r> MulAssign<&'r T>`, which is satisfied by all
    /// primitive numeric types.
    pub fn scale(&mut self, scalar: T)
    where
        for<'r> T: MulAssign<&'r T>,
    {
        self.inner.scale(scalar);
    }

    /// Scalar multiplication returning a new matrix.
    pub fn mul_scalar(&self, scalar: T) -> CscMatrix<T>
    where
        T: Copy + Mul<Output = T> + Zero + 'static,
    {
        let new_inner = self.inner.map(|&v| v * scalar);
        Self { inner: new_inner }
    }

    /// Element-wise addition of two CSC matrices with the same shape.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the matrices have different shapes.
    pub fn add(&self, rhs: &CscMatrix<T>) -> Result<CscMatrix<T>, FerroError>
    where
        T: Zero + Default + Clone + 'static,
        for<'r> &'r T: Add<&'r T, Output = T>,
    {
        if self.n_rows() != rhs.n_rows() || self.n_cols() != rhs.n_cols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_rows(), self.n_cols()],
                actual: vec![rhs.n_rows(), rhs.n_cols()],
                context: "CscMatrix::add".into(),
            });
        }
        let result = &self.inner + &rhs.inner;
        Ok(Self { inner: result })
    }

    /// Element-wise subtraction of two CSC matrices with the same shape: `A - B`.
    ///
    /// Mirrors scipy `csc_matrix` subtraction `A - B` (`_sub_sparse`,
    /// `scipy/sparse/_compressed.py:260`, which dispatches to the `_minus_`
    /// binary op). The result has the UNION sparsity of `A` and `B`: a position
    /// stored in either operand is stored in the output, so a stored−stored
    /// difference that cancels to `0` may remain an explicit zero — scipy keeps
    /// it too, and it materializes to `0` under [`to_dense`](Self::to_dense), so
    /// `to_dense` parity holds.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the matrices have different shapes.
    pub fn sub(&self, rhs: &CscMatrix<T>) -> Result<CscMatrix<T>, FerroError>
    where
        T: Zero + Default + Clone + 'static,
        for<'r> &'r T: std::ops::Sub<&'r T, Output = T>,
    {
        if self.n_rows() != rhs.n_rows() || self.n_cols() != rhs.n_cols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_rows(), self.n_cols()],
                actual: vec![rhs.n_rows(), rhs.n_cols()],
                context: "CscMatrix::sub".into(),
            });
        }
        let result = &self.inner - &rhs.inner;
        Ok(Self { inner: result })
    }

    /// Element-wise (Hadamard) product of two CSC matrices with the same shape.
    ///
    /// Mirrors scipy `csc_matrix.multiply(other)` (`scipy/sparse/_base.py:490`),
    /// the element-wise product, which for two same-shape sparse operands runs
    /// the `_elmul_` binary op. The result keeps only positions that are stored
    /// (non-zero) in BOTH operands — the INTERSECTION sparsity — since the
    /// product is zero wherever either factor is a structural zero. Oracle:
    /// `A.multiply(B).toarray()` for `A=[[1,0,2],[0,3,0],[4,0,5]]`,
    /// `B=[[1,1,0],[0,1,1],[0,0,1]]` is `[[1,0,0],[0,3,0],[0,0,5]]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the matrices have different shapes.
    pub fn multiply(&self, rhs: &CscMatrix<T>) -> Result<CscMatrix<T>, FerroError>
    where
        T: Zero + Clone + 'static,
        for<'r> &'r T: Mul<&'r T, Output = T>,
    {
        if self.n_rows() != rhs.n_rows() || self.n_cols() != rhs.n_cols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_rows(), self.n_cols()],
                actual: vec![rhs.n_rows(), rhs.n_cols()],
                context: "CscMatrix::multiply".into(),
            });
        }
        // sprs `mul_mat_same_storage` runs `csmat_binop(|x, y| x * y)` over the
        // two same-storage (both CSC) operands, emitting a non-zero only where
        // both inputs are stored — the element-wise (Hadamard) intersection.
        let result = sprs::binop::mul_mat_same_storage(&self.inner, &rhs.inner);
        Ok(Self { inner: result })
    }

    /// Sparse-sparse matrix product: `A @ B` (scipy `A.dot(B)`).
    ///
    /// Mirrors scipy `csc_matrix @ csc_matrix` (`_matmul_sparse`,
    /// `scipy/sparse/_compressed.py:415`), the SMMP algorithm: the result is the
    /// `(self.n_rows(), rhs.n_cols())` matrix product, so `self.n_cols()` must
    /// equal `rhs.n_rows()`. Computed via the sprs product operator
    /// `&self.inner * &rhs.inner` and materialized into CSC storage with
    /// `.to_csc()`, the column-symmetric analog of the CSR `matmul` (#2000).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `self.n_cols() != rhs.n_rows()`
    /// (scipy raises `ValueError: dimension mismatch`).
    pub fn matmul(&self, rhs: &CscMatrix<T>) -> Result<CscMatrix<T>, FerroError>
    where
        T: Clone + sprs::MulAcc + Zero + Default + Send + Sync + 'static,
    {
        if self.n_cols() != rhs.n_rows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_cols()],
                actual: vec![rhs.n_rows()],
                context: "CscMatrix::matmul: A.n_cols must equal B.n_rows".into(),
            });
        }
        // The sprs product `&A * &B` computes the matrix product; `to_csc()`
        // materializes the result into CSC storage so the wrapped `inner` is CSC.
        let inner = (&self.inner * &rhs.inner).to_csc();
        Ok(Self { inner })
    }

    /// Sparse matrix-dense vector product: computes `self * rhs`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `rhs.len() != n_cols()`.
    pub fn mul_vec(&self, rhs: &Array1<T>) -> Result<Array1<T>, FerroError>
    where
        T: Clone + Zero + 'static,
        for<'r> &'r T: Mul<Output = T>,
        T: AddAssign,
    {
        if rhs.len() != self.n_cols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_cols()],
                actual: vec![rhs.len()],
                context: "CscMatrix::mul_vec".into(),
            });
        }
        let result = &self.inner * rhs;
        Ok(result)
    }

    /// Sum of all stored values.
    ///
    /// Mirrors scipy `csc_matrix.sum()` with `axis=None`
    /// (`scipy/sparse/_compressed.py:492`), which reduces over both rows and
    /// columns to a scalar. Structural zeros contribute nothing, so this is the
    /// running total of the stored non-zero entries accumulated from
    /// [`T::zero()`].
    #[must_use]
    pub fn sum(&self) -> T
    where
        T: Copy + Zero + Add<Output = T>,
    {
        let mut acc = T::zero();
        for (&val, _) in &self.inner {
            acc = acc + val;
        }
        acc
    }

    /// Column sums, a length-`n_cols` vector.
    ///
    /// Mirrors scipy `csc_matrix.sum(axis=0)`
    /// (`scipy/sparse/_compressed.py:492`), which returns a `(1, n_cols)`
    /// row vector of per-column sums; here it is returned as a length-`n_cols`
    /// [`Array1`]. For each stored entry `(row, col, val)`, `val` is added to
    /// `out[col]`.
    #[must_use]
    pub fn sum_axis0(&self) -> Array1<T>
    where
        T: Copy + Zero + Add<Output = T>,
    {
        let mut out = Array1::<T>::zeros(self.n_cols());
        for (&val, (_, c)) in &self.inner {
            out[c] = out[c] + val;
        }
        out
    }

    /// Row sums, a length-`n_rows` vector.
    ///
    /// Mirrors scipy `csc_matrix.sum(axis=1)`
    /// (`scipy/sparse/_compressed.py:492`), which returns an `(n_rows, 1)`
    /// column vector of per-row sums; here it is returned as a length-`n_rows`
    /// [`Array1`]. For each stored entry `(row, col, val)`, `val` is added to
    /// `out[row]`.
    #[must_use]
    pub fn sum_axis1(&self) -> Array1<T>
    where
        T: Copy + Zero + Add<Output = T>,
    {
        let mut out = Array1::<T>::zeros(self.n_rows());
        for (&val, (r, _)) in &self.inner {
            out[r] = out[r] + val;
        }
        out
    }

    /// Main diagonal, a length-`min(n_rows, n_cols)` vector.
    ///
    /// Mirrors scipy `csc_matrix.diagonal()` with `k=0`
    /// (`scipy/sparse/_compressed.py:476`): `out[i] == A[i, i]` for
    /// `i in 0..min(n_rows, n_cols)`. Positions absent from the CSC structure
    /// are structural zeros, so `out[i]` defaults to [`T::zero()`]. For CSC,
    /// `outer_iterator()` yields columns, so the `i`-th column's row-`i` entry
    /// is `A[i, i]`.
    #[must_use]
    pub fn diagonal(&self) -> Array1<T>
    where
        T: Copy + Zero,
    {
        let len = self.n_rows().min(self.n_cols());
        let mut out = Array1::<T>::zeros(len);
        for (i, col) in self.inner.outer_iterator().enumerate().take(len) {
            if let Some(&val) = col.get(i) {
                out[i] = val;
            }
        }
        out
    }

    /// Transpose: returns a new `(n_cols, n_rows)` CSC matrix whose dense form
    /// is `Aᵀ`.
    ///
    /// Mirrors scipy `csc_matrix.transpose()` / `.T` (`scipy/sparse/_csc.py:20`),
    /// where `A.T` reinterprets the same `(data, indices, indptr)` buffers as a
    /// CSR container of shape `(N, M)` (a no-allocation storage-order swap). Here
    /// that CSR-storage view of `Aᵀ` is materialized back into owned CSC storage
    /// via sprs `transpose_view().to_csc()`, so the result is a `CscMatrix` of
    /// `Aᵀ`.
    #[must_use]
    pub fn transpose(&self) -> CscMatrix<T>
    where
        T: Clone + Default + 'static,
    {
        Self {
            inner: self.inner.transpose_view().to_csc(),
        }
    }

    /// Scalar element access: returns `A[i, j]`.
    ///
    /// Mirrors scipy `csc_matrix.__getitem__` for a scalar `(int, int)` key
    /// (`IndexMixin.__getitem__` -> `_get_intXint`, `scipy/sparse/_index.py:29`):
    /// `A[i, j]` returns the stored value, or `0` if the position is
    /// structurally absent (no stored entry). The sprs `CsMat::get(i, j)`
    /// returns `Some(&value)` when stored and `None` when absent (it works
    /// identically on CSC storage, indexing by `(row, col)`), so an absent
    /// position yields [`T::zero()`]. Live oracle (R-CHAR-3): for
    /// `A = [[1,0,2],[0,3,0],[4,0,5]]`, `A[1, 1] == 3`, `A[0, 1] == 0`
    /// (structurally absent), `A[0, 0] == 1`, `A[0, 2] == 2`, `A[2, 0] == 4`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `i >= n_rows()` or
    /// `j >= n_cols()`. scipy raises `IndexError(f'index ({idx}) out of range')`
    /// (`scipy/sparse/_index.py`); ferrolearn maps an out-of-bounds index to
    /// `InvalidParameter` per the crate error contract (R-DEV-2).
    pub fn get(&self, i: usize, j: usize) -> Result<T, FerroError>
    where
        T: Copy + Zero,
    {
        if i >= self.n_rows() || j >= self.n_cols() {
            return Err(FerroError::InvalidParameter {
                name: "index".into(),
                reason: format!(
                    "index ({i}, {j}) out of bounds for shape ({}, {})",
                    self.n_rows(),
                    self.n_cols()
                ),
            });
        }
        Ok(self.inner.get(i, j).copied().unwrap_or_else(T::zero))
    }

    /// Single-column extraction: returns column `j` as a `(n_rows, 1)` CSC matrix.
    ///
    /// Mirrors scipy `csc_matrix.getcol(j)` (`scipy/sparse/_matrix.py:104` →
    /// `_getcol`), which returns "a copy of column j of the matrix, as an (m x 1)
    /// sparse matrix (column vector)". For CSC the column is the outer (natural)
    /// dimension, so column `j` is the half-open range `j..j+1`: this delegates to
    /// [`col_slice`](Self::col_slice)`(j, j + 1)`, which already produces the
    /// `(n_rows, 1)` CSC — the column-symmetric analog of CSR's `getrow`
    /// delegating to `row_slice`. Live oracle (R-CHAR-3): for
    /// `A = [[1,0,2],[0,3,0],[4,0,5]]`, `A.getcol(0).toarray() == [[1],[0],[4]]`
    /// (shape `(3,1)`), `A.getcol(2).toarray() == [[2],[0],[5]]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `j >= n_cols()`. scipy raises
    /// `IndexError("index out of bounds")`; ferrolearn maps the out-of-range index
    /// to `InvalidParameter` per the crate error contract (R-DEV-2).
    pub fn getcol(&self, j: usize) -> Result<CscMatrix<T>, FerroError>
    where
        T: Clone + Default + 'static,
    {
        if j >= self.n_cols() {
            return Err(FerroError::InvalidParameter {
                name: "index".into(),
                reason: format!("col index {j} out of bounds for {} cols", self.n_cols()),
            });
        }
        self.col_slice(j, j + 1)
    }

    /// Single-row extraction: returns row `i` as a `(1, n_cols)` CSC matrix.
    ///
    /// Mirrors scipy `csc_matrix.getrow(i)` (`scipy/sparse/_matrix.py:110` →
    /// `_getrow`), which returns "a copy of row i of the matrix, as a (1 x n)
    /// sparse matrix (row vector)". For CSC the row is the inner dimension, so the
    /// simplest correct path is via transpose: `Aᵀ` has shape `(n_cols, n_rows)`,
    /// its column `i` is the `(n_cols, 1)` [`getcol`](Self::getcol)`(i)`, and
    /// transposing that back is the `(1, n_cols)` row `i` of `A` — the
    /// column-symmetric analog of CSR's `getcol` going via transpose. Live oracle
    /// (R-CHAR-3): for `A = [[1,0,2],[0,3,0],[4,0,5]]`,
    /// `A.getrow(0).toarray() == [[1,0,2]]` (shape `(1,3)`),
    /// `A.getrow(1).toarray() == [[0,3,0]]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `i >= n_rows()`. scipy raises
    /// `IndexError("index out of bounds")`; ferrolearn maps the out-of-range index
    /// to `InvalidParameter` per the crate error contract (R-DEV-2).
    pub fn getrow(&self, i: usize) -> Result<CscMatrix<T>, FerroError>
    where
        T: Clone + Default + 'static,
    {
        if i >= self.n_rows() {
            return Err(FerroError::InvalidParameter {
                name: "index".into(),
                reason: format!("row index {i} out of bounds for {} rows", self.n_rows()),
            });
        }
        Ok(self.transpose().getcol(i)?.transpose())
    }

    /// Return a copy of this matrix, preserving **all** stored structure.
    ///
    /// Mirrors scipy `csc_matrix.copy()` (`scipy/sparse/_data.py:94` —
    /// `return self._with_data(self.data.copy(), copy=True)`), which returns an
    /// identical matrix with the same sparsity pattern (`indptr`/`indices`) and
    /// a copy of the `data` array. Every stored entry is preserved verbatim —
    /// including **explicit stored zeros** — without coalescing or reordering.
    /// Equivalent to [`Clone::clone`] (`CscMatrix` derives `Clone`); provided as
    /// a named method for scipy parity. Storage-agnostic port of the CSR
    /// [`copy`](crate::CsrMatrix::copy); for CSC the preserved `indptr` is the
    /// COLUMN pointer and `indices` are the ROW indices.
    ///
    /// Live oracle (R-CHAR-3): for
    /// `sp.csc_matrix(np.diag([3.7,-2.9,5.0]))`, `A.copy().nnz == 3`,
    /// `A.copy().data == [3.7, -2.9, 5.0]`, and `A.copy().toarray()` round-trips.
    #[must_use]
    pub fn copy(&self) -> CscMatrix<T> {
        self.clone()
    }

    /// Return a new CSC matrix with all explicitly-stored zero entries removed.
    ///
    /// Mirrors scipy `csc_matrix.eliminate_zeros()`
    /// (`scipy/sparse/_compressed.py:1025` — `csc_eliminate_zeros(...)` then
    /// `self.prune()`), which drops every stored entry whose value equals `0`
    /// and rebuilds the CSC `(data, indices, indptr)` triple. Walks each column
    /// `c` over its stored slice `indptr[c]..indptr[c+1]` and keeps the
    /// `(row, val)` pair only when `val != 0`, accumulating the kept count into a
    /// fresh column-pointer `indptr2`. Within-column row order is preserved
    /// (scipy keeps the existing sorted order), so the result stays canonical.
    /// Storage-agnostic port of the CSR [`eliminate_zeros`](crate::CsrMatrix::eliminate_zeros):
    /// for CSC the outer walk is over COLUMNS via the column pointer and the kept
    /// `indices` are ROW indices.
    ///
    /// **Deviation (R-DEV-4):** scipy mutates in place; ferrolearn returns a NEW
    /// matrix (functional style, consistent with the other `CscMatrix` methods).
    ///
    /// Live oracle (R-CHAR-3): for CSC `data=[3,0,5]`, `indices=[0,1,2]`,
    /// `indptr=[0,1,2,3]`, shape `(3,3)`, `eliminate_zeros()` yields `nnz == 2`,
    /// `data == [3, 5]`, `indices == [0, 2]`, `indptr == [0, 1, 1, 2]`, dense
    /// `[[3,0,0],[0,0,0],[0,0,5]]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] propagated from [`new`](Self::new); infallible for
    /// any structurally valid `CscMatrix` (filtering keeps every kept index in
    /// bounds of the unchanged shape and preserves sorted within-column order).
    pub fn eliminate_zeros(&self) -> Result<CscMatrix<T>, FerroError>
    where
        T: Clone + Zero + PartialEq,
    {
        let zero = T::zero();
        let indptr = self.indptr();
        let indices = self.indices();
        let data = self.data();
        let n_cols = self.n_cols();

        let mut indptr2 = Vec::with_capacity(n_cols + 1);
        let mut indices2 = Vec::new();
        let mut data2 = Vec::new();
        indptr2.push(0usize);
        for c in 0..n_cols {
            for k in indptr[c]..indptr[c + 1] {
                if data[k] != zero {
                    indices2.push(indices[k]);
                    data2.push(data[k].clone());
                }
            }
            indptr2.push(data2.len());
        }
        Self::new(self.n_rows(), n_cols, indptr2, indices2, data2)
    }

    /// Cast every stored value to a new scalar type `U` via a caller-supplied
    /// closure, preserving the CSC sparsity structure (`indptr`/`indices`,
    /// shape, nnz).
    ///
    /// Mirrors scipy `csc_matrix.astype(dtype)` (`scipy/sparse/_data.py:69` —
    /// `self._with_data(self.data.astype(dtype, ...))`), which C-casts every
    /// stored value in `self.data` to the requested numpy dtype while keeping the
    /// `indptr`/`indices` structure and `shape` unchanged. scipy selects the cast
    /// from a runtime numpy dtype object; Rust has no runtime dtype, so this is a
    /// **deviation** (R-DEV-4): the caller supplies the element cast as a closure.
    /// A plain `as`-cast closure (e.g. `|&v| v as i64`) reproduces numpy's C-cast
    /// semantics, including float→int **truncation toward zero**.
    ///
    /// The `indptr` (column pointers) and `indices` (row indices) arrays, the
    /// `(n_rows, n_cols)` shape, and the stored count are copied verbatim — only
    /// the data array changes type — so explicit stored zeros are preserved (no
    /// coalescing). Rebuilds a `CscMatrix<U>` via [`new`](Self::new) from the
    /// original `indptr`/`indices` and the cast data. Storage-agnostic port of
    /// the CSR [`astype`](crate::CsrMatrix::astype).
    ///
    /// Live oracle (R-CHAR-3): for `sp.csc_matrix(np.diag([3.7,-2.9,5.0]))`,
    /// `astype(np.int64)` yields `data == [3, -2, 5]` (truncated toward zero)
    /// with the `indptr`/`indices`/shape structure unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] propagated from [`new`](Self::new); infallible for
    /// any structurally valid `CscMatrix` (the unchanged `indptr`/`indices` stay
    /// valid for the unchanged shape).
    pub fn astype<U, Fc>(&self, cast: Fc) -> Result<CscMatrix<U>, FerroError>
    where
        U: Clone,
        Fc: Fn(&T) -> U,
    {
        let data: Vec<U> = self.data().iter().map(&cast).collect();
        CscMatrix::<U>::new(
            self.n_rows(),
            self.n_cols(),
            self.indptr(),
            self.indices().to_vec(),
            data,
        )
    }
}

impl<T> CscMatrix<T>
where
    T: Copy + Zero + PartialOrd,
{
    /// Maximum over all elements (scipy `axis=None`), folding in implicit zeros.
    ///
    /// Mirrors scipy `csc_matrix.max()` via the `_minmax_mixin._min_or_max`
    /// machinery with `axis=None` (`scipy/sparse/_data.py:208`-`:224`). scipy
    /// reduces over the stored data (`m = min_or_max.reduce(self._deduped_data())`,
    /// `:221`) and then, when the matrix is **not fully dense**, folds an implicit
    /// zero into the result (`if self.nnz != math.prod(self.shape): m =
    /// min_or_max(zero, m)`, `:222`-`:223`). An empty matrix (`nnz == 0`) returns
    /// `zero` (`:219`-`:220`). The per-stored-entry value logic is identical to
    /// the CSR [`max`](crate::CsrMatrix::max) — storage-agnostic (the fold is over
    /// [`data()`](Self::data) regardless of CSR/CSC layout). The matrix is fully
    /// dense iff `data().len() == n_rows * n_cols`, in which case no implicit zero
    /// exists to fold.
    ///
    /// Live oracle (R-CHAR-3): `csc_matrix(diag(-3,-1,-5))` 3x3 `.max() == 0.0`
    /// (implicit zero wins over the all-negative stored data);
    /// `csc_matrix(diag(3,1,5))` `.max() == 5.0`.
    #[must_use]
    pub fn max(&self) -> T {
        let zero = T::zero();
        let data = self.data();
        if data.is_empty() {
            return zero;
        }
        let mut running = data[0];
        for &val in &data[1..] {
            if val > running {
                running = val;
            }
        }
        if data.len() < self.n_rows() * self.n_cols() && zero > running {
            running = zero;
        }
        running
    }

    /// Minimum over all elements (scipy `axis=None`), folding in implicit zeros.
    ///
    /// Mirrors scipy `csc_matrix.min()` via the `_minmax_mixin._min_or_max`
    /// machinery with `axis=None` (`scipy/sparse/_data.py:208`-`:224`). scipy
    /// reduces over the stored data (`m = min_or_max.reduce(self._deduped_data())`,
    /// `:221`) and then, when the matrix is **not fully dense**, folds an implicit
    /// zero into the result (`if self.nnz != math.prod(self.shape): m =
    /// min_or_max(zero, m)`, `:222`-`:223`). An empty matrix (`nnz == 0`) returns
    /// `zero` (`:219`-`:220`). The per-stored-entry value logic is identical to
    /// the CSR [`min`](crate::CsrMatrix::min) — storage-agnostic (the fold is over
    /// [`data()`](Self::data) regardless of CSR/CSC layout). The matrix is fully
    /// dense iff `data().len() == n_rows * n_cols`, in which case no implicit zero
    /// exists to fold.
    ///
    /// Live oracle (R-CHAR-3): `csc_matrix(diag(3,1,5))` 3x3 `.min() == 0.0`
    /// (implicit zero wins over the all-positive stored data);
    /// `csc_matrix(diag(-3,-1,-5))` `.min() == -5.0`.
    #[must_use]
    pub fn min(&self) -> T {
        let zero = T::zero();
        let data = self.data();
        if data.is_empty() {
            return zero;
        }
        let mut running = data[0];
        for &val in &data[1..] {
            if val < running {
                running = val;
            }
        }
        if data.len() < self.n_rows() * self.n_cols() && zero < running {
            running = zero;
        }
        running
    }
}

impl<T> CscMatrix<T>
where
    T: num_traits::Float,
{
    /// Raise every stored value to the power `n`, returning a **new** matrix with
    /// the sparsity structure preserved.
    ///
    /// Mirrors scipy `csc_matrix.power(n)` (`scipy/sparse/_data.py:99` —
    /// `return self._with_data(data ** n)`), which performs an **element-wise
    /// power over `self.data` only**, leaving the `indptr` (column pointers),
    /// `indices` (row indices), the `(n_rows, n_cols)` shape, and the stored
    /// count (`nnz`) unchanged. The implicit zeros stay zero because
    /// `0 ** n == 0` for `n > 0` (scipy refuses `n == 0`, which would densify the
    /// matrix; this method does not special-case it — `0.powf(0) == 1` per IEEE,
    /// matching `T::powf`). Each stored value is mapped through
    /// [`num_traits::Float::powf`]. Storage-agnostic port of the CSR
    /// [`power`](crate::CsrMatrix::power).
    ///
    /// Explicit stored zeros are preserved (no coalescing) — only the data array
    /// is transformed; the original `indptr`/`indices` are reused verbatim.
    ///
    /// Live oracle (R-CHAR-3): for `sp.csc_matrix(np.diag([2.,-3.]))` (CSC
    /// `data=[2,-3]`), `power(2)` yields `data == [4, 9]`; `power(3)` yields
    /// `data == [8, -27]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] propagated from [`new`](Self::new); infallible for
    /// any structurally valid `CscMatrix` (the unchanged `indptr`/`indices` stay
    /// in bounds of the unchanged shape and keep their sorted within-column order).
    pub fn power(&self, n: T) -> Result<CscMatrix<T>, FerroError> {
        let data: Vec<T> = self.data().iter().map(|v| v.powf(n)).collect();
        CscMatrix::new(
            self.n_rows(),
            self.n_cols(),
            self.indptr(),
            self.indices().to_vec(),
            data,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn sample_csc() -> CscMatrix<f64> {
        // 3x3 sparse matrix (same logical matrix as CsrMatrix tests):
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        //
        // CSC indptr is per-column:
        //   col 0: rows 0, 2  → values 1, 4
        //   col 1: row 1      → value 3
        //   col 2: rows 0, 2  → values 2, 5
        CscMatrix::new(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 4.0, 3.0, 2.0, 5.0],
        )
        .unwrap()
    }

    #[test]
    fn test_new_valid() {
        let m = sample_csc();
        assert_eq!(m.n_rows(), 3);
        assert_eq!(m.n_cols(), 3);
        assert_eq!(m.nnz(), 5);
    }

    #[test]
    fn test_to_dense() {
        let m = sample_csc();
        let d = m.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 1.0);
        assert_abs_diff_eq!(d[[0, 2]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 3.0);
        assert_abs_diff_eq!(d[[2, 0]], 4.0);
        assert_abs_diff_eq!(d[[2, 2]], 5.0);
    }

    #[test]
    fn test_from_dense() {
        let dense = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let m = CscMatrix::from_dense(&dense.view(), 0.0);
        assert_eq!(m.nnz(), 2);
        let back = m.to_dense();
        assert_abs_diff_eq!(back[[0, 0]], 1.0);
        assert_abs_diff_eq!(back[[1, 1]], 2.0);
    }

    #[test]
    fn test_from_coo_roundtrip() {
        let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
        coo.push(0, 0, 1.0).unwrap();
        coo.push(1, 2, 4.0).unwrap();
        coo.push(2, 1, 7.0).unwrap();
        let csc = CscMatrix::from_coo(&coo).unwrap();
        let dense = csc.to_dense();
        assert_abs_diff_eq!(dense[[0, 0]], 1.0);
        assert_abs_diff_eq!(dense[[1, 2]], 4.0);
        assert_abs_diff_eq!(dense[[2, 1]], 7.0);
    }

    #[test]
    fn test_csc_csr_roundtrip() {
        let csc = sample_csc();
        let csr = csc.to_csr();
        let back = CscMatrix::from_csr(&csr).unwrap();
        assert_eq!(back.to_dense(), csc.to_dense());
    }

    #[test]
    fn test_col_slice() {
        let m = sample_csc();
        let sliced = m.col_slice(0, 2).unwrap();
        assert_eq!(sliced.n_rows(), 3);
        assert_eq!(sliced.n_cols(), 2);
        let d = sliced.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 1.0);
        assert_abs_diff_eq!(d[[1, 1]], 3.0);
    }

    #[test]
    fn test_col_slice_empty() {
        let m = sample_csc();
        let sliced = m.col_slice(1, 1).unwrap();
        assert_eq!(sliced.n_cols(), 0);
    }

    #[test]
    fn test_col_slice_invalid() {
        let m = sample_csc();
        assert!(m.col_slice(2, 1).is_err());
        assert!(m.col_slice(0, 4).is_err());
    }

    #[test]
    fn test_mul_scalar() {
        let m = sample_csc();
        let m2 = m.mul_scalar(2.0);
        let d = m2.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 6.0);
    }

    #[test]
    fn test_scale_in_place() {
        let mut m = sample_csc();
        m.scale(3.0);
        let d = m.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 3.0);
        assert_abs_diff_eq!(d[[2, 2]], 15.0);
    }

    #[test]
    fn test_add() {
        let m = sample_csc();
        let sum = m.add(&m).unwrap();
        let d = sum.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 6.0);
    }

    #[test]
    fn test_add_shape_mismatch() {
        let m1 = sample_csc();
        let m2 = CscMatrix::new(2, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        assert!(m1.add(&m2).is_err());
    }

    #[test]
    fn test_mul_vec() {
        let m = sample_csc();
        let v = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let result = m.mul_vec(&v).unwrap();
        assert_abs_diff_eq!(result[0], 7.0);
        assert_abs_diff_eq!(result[1], 6.0);
        assert_abs_diff_eq!(result[2], 19.0);
    }

    #[test]
    fn test_mul_vec_shape_mismatch() {
        let m = sample_csc();
        let v = Array1::from(vec![1.0_f64, 2.0]);
        assert!(m.mul_vec(&v).is_err());
    }

    // REQ-MISSING-REDUCE — live scipy 1.x oracle (R-CHAR-3). Expected values
    // from `python3 -c "import numpy as np, scipy.sparse as sp;
    //   A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]));
    //   print(A.sum(), A.sum(axis=0).tolist(), A.sum(axis=1).tolist(),
    //         A.diagonal().tolist())"`
    //   -> 15.0 [[5.0,3.0,7.0]] [[3.0],[3.0],[9.0]] [1.0,3.0,5.0]
    // and B=[[1,2,3],[4,5,6]] -> B.diagonal().tolist() == [1.0,5.0].

    fn sample_csc_b() -> CscMatrix<f64> {
        // 2x3 dense matrix B = [[1,2,3],[4,5,6]] (no zeros). Built via
        // from_dense (infallible, no unwrap) to match the test idiom.
        let dense = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        CscMatrix::from_dense(&dense.view(), 0.0)
    }

    #[test]
    fn csc_sum_matches_scipy() {
        let m = sample_csc();
        assert_abs_diff_eq!(m.sum(), 15.0);
    }

    #[test]
    fn csc_sum_axis0_matches_scipy() {
        let m = sample_csc();
        assert_eq!(m.sum_axis0(), array![5.0, 3.0, 7.0]);
    }

    #[test]
    fn csc_sum_axis1_matches_scipy() {
        let m = sample_csc();
        assert_eq!(m.sum_axis1(), array![3.0, 3.0, 9.0]);
    }

    #[test]
    fn csc_diagonal_matches_scipy() {
        let m = sample_csc();
        assert_eq!(m.diagonal(), array![1.0, 3.0, 5.0]);
    }

    #[test]
    fn csc_diagonal_non_square() {
        let m = sample_csc_b();
        assert_eq!(m.diagonal(), array![1.0, 5.0]);
    }

    // REQ-MISSING-TRANSPOSE — live scipy oracle (R-CHAR-3). Expected values from
    // `python3 -c "import numpy as np, scipy.sparse as sp;
    //   A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]));
    //   B=sp.csc_matrix(np.array([[1.,2,3],[4,5,6]]));
    //   print(A.T.toarray().tolist(), B.T.toarray().tolist())"`
    //   -> [[1,0,4],[0,3,0],[2,0,5]] [[1,4],[2,5],[3,6]].

    #[test]
    fn csc_transpose_matches_scipy() {
        let a = sample_csc();
        let at = a.transpose();
        assert_eq!(at.n_rows(), 3);
        assert_eq!(at.n_cols(), 3);
        assert_eq!(
            at.to_dense(),
            array![[1.0, 0.0, 4.0], [0.0, 3.0, 0.0], [2.0, 0.0, 5.0]]
        );
    }

    #[test]
    fn csc_transpose_non_square() {
        let b = sample_csc_b();
        let bt = b.transpose();
        assert_eq!(bt.n_rows(), 3);
        assert_eq!(bt.n_cols(), 2);
        assert_eq!(bt.to_dense(), array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    }

    #[test]
    fn csc_transpose_twice_roundtrip() {
        let b = sample_csc_b();
        let btt = b.transpose().transpose();
        assert_eq!(btt.n_rows(), 2);
        assert_eq!(btt.n_cols(), 3);
        assert_eq!(btt.to_dense(), b.to_dense());
    }
}

/// Kani proof harnesses for CscMatrix structural invariants.
///
/// These proofs verify that after construction via `new()`, `from_coo()`, and
/// `add()`, the underlying CSC representation satisfies all structural
/// invariants:
///
/// - `indptr.len() == n_cols + 1`
/// - `indptr` is monotonically non-decreasing
/// - All row indices are less than `n_rows`
/// - `indices.len() == data.len()`
///
/// All proofs use small symbolic bounds (at most 3 rows/cols) because sparse
/// matrix verification is computationally expensive for Kani.
#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::coo::CooMatrix;

    /// Maximum dimension for symbolic exploration.
    const MAX_DIM: usize = 3;

    /// Helper: assert all CSC structural invariants on the inner `CsMat`.
    fn assert_csc_invariants<T>(m: &CscMatrix<T>) {
        let inner = m.inner();

        // Invariant 1: indptr length == n_cols + 1
        let indptr = inner.indptr();
        let indptr_raw = indptr.raw_storage();
        assert!(indptr_raw.len() == m.n_cols() + 1);

        // Invariant 2: indptr is monotonically non-decreasing
        for i in 0..m.n_cols() {
            assert!(indptr_raw[i] <= indptr_raw[i + 1]);
        }

        // Invariant 3: all row indices < n_rows
        let indices = inner.indices();
        for &row_idx in indices {
            assert!(row_idx < m.n_rows());
        }

        // Invariant 4: indices.len() == data.len()
        assert!(inner.indices().len() == inner.data().len());
    }

    /// Verify `indptr.len() == n_cols + 1` after `new()` with a symbolic
    /// empty matrix of arbitrary dimensions.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_new_indptr_length() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Build a valid empty CSC matrix
        let indptr = vec![0usize; n_cols + 1];
        let indices: Vec<usize> = vec![];
        let data: Vec<i32> = vec![];

        if let Ok(m) = CscMatrix::new(n_rows, n_cols, indptr, indices, data) {
            let inner_indptr = m.inner().indptr();
            assert!(inner_indptr.raw_storage().len() == n_cols + 1);
        }
    }

    /// Verify indptr monotonicity after `new()` with a symbolic single-entry
    /// matrix.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_new_indptr_monotonic() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Place a single non-zero at a symbolic valid position
        let row: usize = kani::any();
        let col: usize = kani::any();
        kani::assume(row < n_rows);
        kani::assume(col < n_cols);

        // Build indptr for a single entry in column `col`
        let mut indptr = vec![0usize; n_cols + 1];
        for i in (col + 1)..=n_cols {
            indptr[i] = 1;
        }
        let indices = vec![row];
        let data = vec![42i32];

        if let Ok(m) = CscMatrix::new(n_rows, n_cols, indptr, indices, data) {
            let inner_indptr = m.inner().indptr().raw_storage().to_vec();
            for i in 0..m.n_cols() {
                assert!(inner_indptr[i] <= inner_indptr[i + 1]);
            }
        }
    }

    /// Verify all row indices < n_rows after `new()`.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_new_row_indices_in_bounds() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let row: usize = kani::any();
        let col: usize = kani::any();
        kani::assume(row < n_rows);
        kani::assume(col < n_cols);

        let mut indptr = vec![0usize; n_cols + 1];
        for i in (col + 1)..=n_cols {
            indptr[i] = 1;
        }
        let indices = vec![row];
        let data = vec![1i32];

        if let Ok(m) = CscMatrix::new(n_rows, n_cols, indptr, indices, data) {
            for &r in m.inner().indices() {
                assert!(r < m.n_rows());
            }
        }
    }

    /// Verify `indices.len() == data.len()` after `new()`.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_new_indices_data_same_length() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let indptr = vec![0usize; n_cols + 1];
        let indices: Vec<usize> = vec![];
        let data: Vec<i32> = vec![];

        if let Ok(m) = CscMatrix::new(n_rows, n_cols, indptr, indices, data) {
            assert!(m.inner().indices().len() == m.inner().data().len());
        }
    }

    /// Verify that `new()` rejects inputs where indices and data have
    /// mismatched lengths.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_new_rejects_mismatched_lengths() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // indices has 1 element, data has 0 — must fail
        let indptr = vec![0usize; n_cols + 1];
        let indices = vec![0usize];
        let data: Vec<i32> = vec![];

        let result = CscMatrix::new(n_rows, n_cols, indptr, indices, data);
        assert!(result.is_err());
    }

    /// Verify all structural invariants after `from_coo()` with symbolic
    /// entries.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_from_coo_invariants() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let mut coo = CooMatrix::<i32>::new(n_rows, n_cols);

        // Insert a symbolic number of entries (0 or 1)
        let do_insert: bool = kani::any();
        if do_insert {
            let row: usize = kani::any();
            let col: usize = kani::any();
            kani::assume(row < n_rows);
            kani::assume(col < n_cols);
            let _ = coo.push(row, col, 1i32);
        }

        if let Ok(csc) = CscMatrix::from_coo(&coo) {
            assert_csc_invariants(&csc);
            assert!(csc.n_rows() == n_rows);
            assert!(csc.n_cols() == n_cols);
        }
    }

    /// Verify that `add()` preserves shape and structural invariants.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_add_preserves_invariants() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Build two valid empty CSC matrices of the same shape
        let indptr = vec![0usize; n_cols + 1];
        let a = CscMatrix::<i32>::new(n_rows, n_cols, indptr.clone(), vec![], vec![]);
        let b = CscMatrix::<i32>::new(n_rows, n_cols, indptr, vec![], vec![]);

        if let (Ok(a), Ok(b)) = (a, b) {
            if let Ok(sum) = a.add(&b) {
                // Shape is preserved
                assert!(sum.n_rows() == n_rows);
                assert!(sum.n_cols() == n_cols);
                // Structural invariants hold
                assert_csc_invariants(&sum);
            }
        }
    }

    /// Verify that `add()` with non-empty matrices preserves invariants.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_add_nonempty_preserves_invariants() {
        // Fixed 2x2 matrices with one entry each in different columns
        let a = CscMatrix::<i32>::new(2, 2, vec![0, 1, 1], vec![0], vec![1]);
        let b = CscMatrix::<i32>::new(2, 2, vec![0, 0, 1], vec![1], vec![2]);

        if let (Ok(a), Ok(b)) = (a, b) {
            if let Ok(sum) = a.add(&b) {
                assert!(sum.n_rows() == 2);
                assert!(sum.n_cols() == 2);
                assert_csc_invariants(&sum);
            }
        }
    }

    /// Verify `mul_vec()` output has correct dimension and does not panic.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_mul_vec_output_dimension() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Empty matrix for tractable verification
        let indptr = vec![0usize; n_cols + 1];
        let m = CscMatrix::<f64>::new(n_rows, n_cols, indptr, vec![], vec![]);

        if let Ok(m) = m {
            let v = Array1::<f64>::zeros(n_cols);
            if let Ok(result) = m.mul_vec(&v) {
                assert!(result.len() == n_rows);
            }
        }
    }

    /// Verify `mul_vec()` rejects vectors of wrong dimension.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_mul_vec_rejects_wrong_dimension() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let indptr = vec![0usize; n_cols + 1];
        let m = CscMatrix::<f64>::new(n_rows, n_cols, indptr, vec![], vec![]);

        if let Ok(m) = m {
            let wrong_len: usize = kani::any();
            kani::assume(wrong_len <= MAX_DIM);
            kani::assume(wrong_len != n_cols);
            let v = Array1::<f64>::zeros(wrong_len);
            let result = m.mul_vec(&v);
            assert!(result.is_err());
        }
    }

    /// Verify `mul_vec()` with a non-empty matrix produces the correct
    /// output dimension and does not trigger any out-of-bounds access.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_mul_vec_nonempty_no_oob() {
        // 2x3 CSC matrix with entries at (0,1) and (1,2)
        // Column 0: empty, Column 1: row 0, Column 2: row 1
        let m = CscMatrix::<f64>::new(2, 3, vec![0, 0, 1, 2], vec![0, 1], vec![3.0, 4.0]);
        if let Ok(m) = m {
            let v = Array1::from(vec![1.0, 2.0, 3.0]);
            if let Ok(result) = m.mul_vec(&v) {
                assert!(result.len() == 2);
            }
        }
    }
}
