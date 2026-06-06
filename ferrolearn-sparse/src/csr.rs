//! Compressed Sparse Row (CSR) matrix format.
//!
//! [`CsrMatrix<T>`] is a newtype wrapper around [`sprs::CsMat<T>`] in CSR
//! storage. CSR matrices are efficient for row-wise operations, matrix-vector
//! products, and row slicing.
//!
//! ## REQ status
//!
//! Mirrors `scipy.sparse.csr_matrix` (`scipy/sparse/_csr.py`; live oracle scipy
//! 1.17, deterministic). Design doc: `.design/sparse/csr.md` (14 REQs). Every REQ
//! is BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker).
//! Behavior is oracle-verified vs the live scipy (R-CHAR-3) — see
//! `tests/divergence_csr.rs`.
//!
//! **12 SHIPPED / 2 NOT-STARTED** (REQ-MISSING-INDEX is a SPLIT: element access
//! `get(i,j)`, rows/cols `getrow`/`getcol`, max/min, and maintenance
//! `astype`/`copy`/`eliminate_zeros`/`sum_duplicates` SHIPPED; `sort_indices`
//! NOT-STARTED).
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-CONSTRUCT-CONVERT | SHIPPED | `from_coo`/`from_dense`/`from_csc` + `to_dense`/`to_coo`/`to_csc` + `nnz` mirror `csr_matrix(...)`/`.toarray()`/`.tocsc()`/`.tocoo()`/`.nnz` (oracle nnz=5; round-trips). Guards `csr_from_*`/`csr_to_csc_roundtrip_matches_scipy`. |
//! | REQ-MATVEC | SHIPPED | `mul_vec(v)` == scipy `A@v` (`[7,6,19]`). Guard `csr_mul_vec_matches_scipy`. |
//! | REQ-ADD | SHIPPED | `add(&B)` == scipy `A+B` (elementwise). Guards `csr_add_self_matches_scipy`/`csr_add_other_matches_scipy`. |
//! | REQ-SCALAR-MUL | SHIPPED | `mul_scalar(s)`/`scale(s)` == scipy `A*s`. Guard `csr_scalar_mul_matches_scipy`. |
//! | REQ-ROW-SLICE | SHIPPED | `row_slice(a,b)` == scipy `A[a:b]` (method vs Python-slice API). Guard `csr_row_slice_matches_scipy`. |
//! | REQ-ERR | SHIPPED | `add`/`mul_vec` return `Err(FerroError::ShapeMismatch)` where scipy raises `ValueError`. Guards `csr_*_shape_mismatch_is_err`. |
//! | REQ-CONSUMER | SHIPPED | real estimator consumers: `ferrolearn-neighbors/src/graph.rs` (`kneighbors_graph`/`radius_neighbors_graph` return CSR) + `ferrolearn-core/src/dataset.rs` (sparse `Dataset` arm); also `helpers.rs`/`csc.rs`. |
//! | REQ-MISSING-MATMUL | SHIPPED (#2000) | `matmul(&B)` returns the `(self.n_rows, rhs.n_cols)` sparse-sparse product `A@B` via the sprs product operator `(&self.inner * &rhs.inner).to_csr()` (`smmp::mul_csr_csr` for two CSR operands), mirroring scipy `_matmul_sparse` (`_compressed.py:415`, SMMP). Shape-checks `self.n_cols() == rhs.n_rows()` first (mismatch → `Err(FerroError::ShapeMismatch)`, scipy `ValueError: dimension mismatch`). Live oracle: `A.matmul(B).toarray()`=`[[1,1,2],[0,3,3],[4,4,5]]`; non-square `A.matmul(C)` for `C=[[1,2],[3,4],[5,6]]`=`[[11,14],[9,12],[29,38]]` shape `(3,2)`. Guards `csr_matmul_matches_scipy`/`csr_matmul_non_square`/`csr_matmul_shape_mismatch_is_err`. |
//! | REQ-MISSING-TRANSPOSE | SHIPPED (#2001) | `transpose()` returns a `(n_cols, n_rows)` CSR of `Aᵀ` via sprs `transpose_view().to_csr()`, mirroring scipy `A.T` (`_csr.py:22` — CSC view of the same buffers, here materialized as CSR). Live oracle: `A.T.toarray()`=`[[1,0,4],[0,3,0],[2,0,5]]`; non-square `B=[[1,2,3],[4,5,6]]` -> `B.T.toarray()`=`[[1,4],[2,5],[3,6]]` shape `(3,2)`; double-transpose round-trips. Guards `csr_transpose_matches_scipy`/`csr_transpose_non_square`/`csr_transpose_twice_roundtrip`. |
//! | REQ-MISSING-REDUCE | SHIPPED | `sum`/`sum_axis0`/`sum_axis1`/`diagonal` mirror scipy `.sum(axis=)` (`_compressed.py:492`) + `.diagonal()` (`_compressed.py:476`). Live oracle: `A.sum()`=15, `A.sum(axis=0)`=[5,3,7], `A.sum(axis=1)`=[3,3,9], `A.diagonal()`=[1,3,5], non-square `B.diagonal()`=[1,5]. Guards `csr_sum_matches_scipy`/`csr_sum_axis0_matches_scipy`/`csr_sum_axis1_matches_scipy`/`csr_diagonal_matches_scipy`/`csr_diagonal_non_square`. |
//! | REQ-MISSING-ELEMENTWISE | SHIPPED (#2003) | `multiply(&B)` (element-wise Hadamard, INTERSECTION sparsity via sprs `binop::mul_mat_same_storage`) mirrors scipy `multiply` (`_base.py:490`, `_elmul_`); `sub(&B)` (`A-B`, UNION sparsity via sprs `&CsMat - &CsMat`) mirrors scipy `_sub_sparse` (`_compressed.py:260`). Both shape-check first (`Err(FerroError::ShapeMismatch)`) like `add`. Live oracle: `A.multiply(B).toarray()`=`[[1,0,0],[0,3,0],[0,0,5]]`, `(A-B).toarray()`=`[[0,-1,2],[0,2,-1],[4,0,4]]`. Guards `csr_multiply_matches_scipy`/`csr_sub_matches_scipy`/`csr_multiply_shape_mismatch_is_err`/`csr_sub_shape_mismatch_is_err`. Sub-note: `.power` (`_data.py:99`) still NOT-STARTED. |
//! | REQ-MISSING-INDEX (element access) | SHIPPED (#2004) | `get(i, j)` returns the scalar `A[i,j]` — the stored value via sprs `inner.get(i,j) -> Option<&T>` (`.copied().unwrap_or_else(T::zero)`, so a structurally absent position yields `0`), bounds-checked (`i >= n_rows() \|\| j >= n_cols()` → `Err(FerroError::InvalidParameter)`), mirroring scipy `A[i,j]` (`IndexMixin.__getitem__` → `_get_intXint`, `_index.py:29`; out-of-range → `IndexError`, `_index.py:388`, mapped to `InvalidParameter` per R-DEV-2). Live oracle (R-CHAR-3): `A=[[1,0,2],[0,3,0],[4,0,5]]` → `A[1,1]=3`, `A[0,1]=0`, `A[0,0]=1`, `A[0,2]=2`, `A[2,0]=4`. Guards `csr_get_element_matches_scipy`/`csr_get_absent_is_zero`/`csr_get_out_of_bounds_is_err`. |
//! | REQ-MISSING-INDEX (rows/cols) | SHIPPED (#2004) | `getrow(i)` returns row `i` as a `(1, n_cols)` CSR (single-row case of `row_slice`, delegates to `self.row_slice(i, i + 1)`), mirroring scipy `getrow(i)` (`_matrix.py:110` → `_getrow`, `_base.py:1116`, "(1 x n) row vector"); `getcol(j)` returns column `j` as a `(n_rows, 1)` CSR via `self.transpose().getrow(j)?.transpose()`, mirroring scipy `getcol(j)` (`_matrix.py:104` → `_getcol`, `_base.py:1097`, "(m x 1) column vector"). Both bounds-check (`i >= n_rows()`/`j >= n_cols()` → `Err(FerroError::InvalidParameter)`, scipy `IndexError`, R-DEV-2). Live oracle (R-CHAR-3): `A.getrow(0).toarray()`=`[[1,0,2]]`, `A.getrow(1).toarray()`=`[[0,3,0]]`, `A.getcol(0).toarray()`=`[[1],[0],[4]]`, `A.getcol(2).toarray()`=`[[2],[0],[5]]`. Guards `csr_getrow_matches_scipy`/`csr_getcol_matches_scipy`/`csr_getrow_getcol_out_of_bounds_is_err`. |
//! | REQ-MISSING-INDEX (max/min) | SHIPPED (#2004) | `max()`/`min()` (`T: Copy + Zero + PartialOrd`) reduce over the stored `data()` and fold an implicit `T::zero()` when not fully dense (`nnz < n_rows*n_cols`); empty → `zero`. Mirrors scipy `_minmax_mixin._min_or_max(axis=None)` (`scipy/sparse/_data.py:208`-`:224`: stored-data reduce then `min_or_max(zero, m)` for the non-dense case). Live oracle (R-CHAR-3): `csr_matrix(diag(-3,-1,-5)).max()==0`, `.min()==-5`; `csr_matrix(diag(3,1,5)).max()==5`, `.min()==0`; dense `csr_matrix([[2,7]]).max()==7`, `.min()==2`. Guards `csr_max_folds_implicit_zero`/`csr_min_folds_implicit_zero`/`csr_max_min_dense_no_implicit_zero`. Direct analog of `CooMatrix::max()/min()`. |
//! | REQ-MISSING-INDEX (maintenance: astype/copy) | SHIPPED (#2004) | `astype<U,Fc>(cast)` (`scipy/sparse/_data.py:69`) casts every stored value to a new type `U` via a caller-supplied closure (Rust has no runtime dtype object — R-DEV-4 deviation; a plain `as`-cast closure reproduces numpy's C-cast float→int truncation toward zero), preserving the `indptr`/`indices` structure, `(n_rows,n_cols)` shape, and `nnz`; rebuilds via [`CsrMatrix::new`] from the original `indptr()`/`indices()` and the cast data (bound `T: Clone`, plus `U: Clone` as `new` requires). `copy()` (`scipy/sparse/_data.py:94`) clones preserving all structure including explicit stored zeros (delegates to `self.clone()`, `CsrMatrix` derives `Clone`; `#[must_use]`). Live oracle (R-CHAR-3): `csr_matrix([[3.7,0,0],[0,-2.9,0],[0,0,5.0]])` (`data=[3.7,-2.9,5.0]`, `indptr=[0,1,2,3]`, `indices=[0,1,2]`) → `astype(np.int64)` `data=[3,-2,5]` (truncated), dense `[[3,0,0],[0,-2,0],[0,0,5]]`; `astype(np.float32)` `data=[3.7f32,-2.9f32,5.0f32]`; `copy()` `nnz=3`, `data=[3.7,-2.9,5.0]`, dense unchanged. Guards `csr_astype_float_to_int_truncates`/`csr_astype_to_f32_preserves_structure`/`csr_copy_preserves_structure`. |
//! | REQ-MISSING-INDEX (maintenance: eliminate_zeros/sum_duplicates) | SHIPPED (#2004) | `eliminate_zeros()` (`T: Clone + Zero + PartialEq`) drops every explicitly-stored `0` entry and rebuilds the CSR triple per row (mirrors scipy `csr_eliminate_zeros`, `scipy/sparse/_compressed.py:1025`); `sum_duplicates()` (`T: Copy + Zero + Add`) canonicalizes by per-row `BTreeMap<col, T>` accumulation (sorted columns, sum on collision, zero sums PRESERVED — dropping zeros is `eliminate_zeros`' job), mirroring scipy `csr_sum_duplicates` (`scipy/sparse/_compressed.py:1063`). Both return a NEW matrix via [`CsrMatrix::new`] (R-DEV-4: scipy mutates in place). Live oracle (R-CHAR-3): eliminate `data=[3,0,5],indices=[0,1,2],indptr=[0,1,2,3]` → nnz 2, `data=[3,5]`, `indices=[0,2]`, `indptr=[0,1,1,2]`; sum_duplicates B `data=[3,5,2,1],indices=[0,0,2,2],indptr=[0,2,2,4]` → nnz 2, `data=[8,3]`, `indices=[0,2]`, `indptr=[0,1,1,2]`; C zero-sum `data=[4,-4,7],indices=[0,0,1],indptr=[0,2,3]` → `data=[0,7]`, `indices=[0,1]`. Guards `csr_eliminate_zeros_matches_scipy`/`csr_sum_duplicates_matches_scipy`/`csr_sum_duplicates_preserves_zero_sum`. NOTE: sprs `CsMat::try_new` rejects duplicate column indices within a row (canonical-by-construction), so a duplicate-bearing `CsrMatrix` is unconstructible via `new`/`from_coo`; `sum_duplicates`' coalescing is a scipy-parity API exercised here against canonical inputs (idempotent — already-canonical pass-through, zero sums preserved). |
//! | REQ-MISSING-INDEX (maintenance: sort_indices) | NOT-STARTED | no `sort_indices` (sprs CSR is sorted-by-construction). Blocker #2004. |
//! | REQ-API-ACCESSORS | SHIPPED (#2005) | first-class `shape()`/`data()`/`indices()`/`indptr()` accessors mirror scipy `.shape` (`_compressed.py:38`) and `.data`/`.indices`/`.indptr` (`_compressed.py:76-78`), the same CSR `(data, indices, indptr)` triple. `shape()` → `(n_rows, n_cols)`; `data()` → `&[T]` (`inner.data()`); `indices()` → `&[usize]` (CSR column indices, `inner.indices()`); `indptr()` → owned `Vec<usize>` (row pointers, `inner.indptr().raw_storage().to_vec()` — owned because the sprs `IndPtrView` accessor borrows a temporary). Live oracle (R-CHAR-3): `A=[[1,0,2],[0,3,0],[4,0,5]]` → `shape=(3,3)`, `data=[1,2,3,4,5]`, `indices=[0,2,1,0,2]`, `indptr=[0,2,3,5]`. Guard `csr_shape_data_indices_indptr_match_scipy`. |
//! | REQ-FERRAY | NOT-STARTED | `sprs::CsMat` + `ndarray` vs ferray's sparse CSR analog (R-SUBSTRATE-1). Blocker #2006. |

use std::collections::BTreeMap;
use std::ops::{Add, AddAssign, Mul, MulAssign};

use ferrolearn_core::{Dataset, FerroError};
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, Zero};
use sprs::CsMat;

use crate::coo::CooMatrix;
use crate::csc::CscMatrix;

/// Compressed Sparse Row (CSR) sparse matrix.
///
/// Stores non-zero entries in row-major order using three arrays: `indptr`
/// (row pointer array of length `n_rows + 1`), `indices` (column indices of
/// each non-zero), and `data` (values of each non-zero).
///
/// # Type Parameter
///
/// `T` — the scalar element type. No bounds are required for basic structural
/// methods; arithmetic methods impose their own bounds.
///
/// # Dataset Trait
///
/// Implements [`ferrolearn_core::Dataset`] when `T: Float + Send + Sync + 'static`,
/// reporting `n_samples() == n_rows()`, `n_features() == n_cols()`, and
/// `is_sparse() == true`.
#[derive(Debug, Clone)]
pub struct CsrMatrix<T> {
    inner: CsMat<T>,
}

impl<T> CsrMatrix<T>
where
    T: Clone,
{
    /// Construct a CSR matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `n_rows` — number of rows.
    /// * `n_cols` — number of columns.
    /// * `indptr` — row pointer array of length `n_rows + 1`.
    /// * `indices` — column index of each non-zero entry.
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
        CsMat::try_new((n_rows, n_cols), indptr, indices, data)
            .map(|inner| Self { inner })
            .map_err(|(_, _, _, err)| FerroError::InvalidParameter {
                name: "CsrMatrix raw components".into(),
                reason: err.to_string(),
            })
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
    /// Mirrors scipy `csr_matrix.shape` (the `self._shape` tuple,
    /// `scipy/sparse/_compressed.py:38`), which is the `(M, N)` dimension pair.
    /// Equivalent to `(self.n_rows(), self.n_cols())`. Live oracle:
    /// `sp.csr_matrix([[1,0,2],[0,3,0],[4,0,5]]).shape == (3, 3)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows(), self.n_cols())
    }

    /// Returns the stored non-zero values, one per stored entry, in row-major
    /// CSR order.
    ///
    /// Mirrors scipy `csr_matrix.data` (`scipy/sparse/_compressed.py:76-78`),
    /// the `data` array of the CSR `(data, indices, indptr)` triple ferrolearn
    /// stores identically. Length equals [`nnz`](Self::nnz). Live oracle:
    /// `sp.csr_matrix([[1,0,2],[0,3,0],[4,0,5]]).data == [1,2,3,4,5]`.
    #[must_use]
    pub fn data(&self) -> &[T] {
        self.inner.data()
    }

    /// Returns the column index of each stored non-zero entry, aligned with
    /// [`data`](Self::data), in row-major CSR order.
    ///
    /// Mirrors scipy `csr_matrix.indices` (`scipy/sparse/_compressed.py:76-78`),
    /// the `indices` array of the CSR `(data, indices, indptr)` triple — for CSR
    /// these are the column indices. Length equals [`nnz`](Self::nnz). Live
    /// oracle: `sp.csr_matrix([[1,0,2],[0,3,0],[4,0,5]]).indices == [0,2,1,0,2]`.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        self.inner.indices()
    }

    /// Returns the row pointer array, of length `n_rows + 1`.
    ///
    /// Mirrors scipy `csr_matrix.indptr` (`scipy/sparse/_compressed.py:76-78`),
    /// the `indptr` array of the CSR `(data, indices, indptr)` triple ferrolearn
    /// stores identically: `indptr[i]..indptr[i+1]` is the slice of
    /// [`data`](Self::data) / [`indices`](Self::indices) belonging to row `i`.
    ///
    /// Returns an owned `Vec<usize>` rather than a borrowed slice (unlike
    /// [`data`](Self::data) / [`indices`](Self::indices)) because the sprs
    /// `CsMat::indptr` accessor yields an owned `IndPtrView` value whose
    /// `raw_storage()` slice borrows that temporary, so no `&[usize]` tied to
    /// `&self` can be returned through the public sprs API; the row-pointer
    /// storage is materialized via `IndPtrView::raw_storage().to_vec()`. Live
    /// oracle: `sp.csr_matrix([[1,0,2],[0,3,0],[4,0,5]]).indptr == [0,2,3,5]`.
    #[must_use]
    pub fn indptr(&self) -> Vec<usize> {
        self.inner.indptr().raw_storage().to_vec()
    }

    /// Construct a [`CsrMatrix`] from a [`CooMatrix`] by converting to CSR.
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
        let inner: CsMat<T> = coo.inner().to_csr();
        Ok(Self { inner })
    }

    /// Construct a [`CsrMatrix`] from a [`CscMatrix`].
    ///
    /// # Errors
    ///
    /// This conversion is always successful.
    pub fn from_csc(csc: &CscMatrix<T>) -> Result<Self, FerroError>
    where
        T: Clone + Default + 'static,
    {
        let inner = csc.inner().to_csr();
        Ok(Self { inner })
    }

    /// Convert to [`CscMatrix`].
    pub fn to_csc(&self) -> CscMatrix<T>
    where
        T: Clone + Default + 'static,
    {
        CscMatrix::from_inner(self.inner.to_csc())
    }

    /// Transpose: returns a new `(n_cols, n_rows)` CSR matrix whose dense form
    /// is `Aᵀ`.
    ///
    /// Mirrors scipy `csr_matrix.transpose()` / `.T` (`scipy/sparse/_csr.py:22`),
    /// where `A.T` reinterprets the same `(data, indices, indptr)` buffers as a
    /// CSC container of shape `(N, M)` (a no-allocation storage-order swap).
    /// Here that CSC-storage view of `Aᵀ` is materialized back into owned CSR
    /// storage via sprs `transpose_view().to_csr()`, so the result is a
    /// `CsrMatrix` of `Aᵀ`.
    #[must_use]
    pub fn transpose(&self) -> CsrMatrix<T>
    where
        T: Clone + Default + 'static,
    {
        Self {
            inner: self.inner.transpose_view().to_csr(),
        }
    }

    /// Convert to [`CooMatrix`].
    ///
    /// Each non-zero becomes one triplet entry.
    pub fn to_coo(&self) -> CooMatrix<T> {
        let mut coo = CooMatrix::with_capacity(self.n_rows(), self.n_cols(), self.nnz());
        for (val, (r, c)) in &self.inner {
            // indices come from a valid matrix so push is infallible here
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

    /// Construct a [`CsrMatrix`] from a dense [`Array2<T>`], dropping entries
    /// whose absolute value is less than or equal to `epsilon`.
    ///
    /// Entries `v` where `|v| <= epsilon` are treated as structural zeros.
    /// For integer types, pass `epsilon = 0`.
    pub fn from_dense(dense: &ArrayView2<'_, T>, epsilon: T) -> Self
    where
        T: Copy + Zero + PartialOrd + num_traits::Signed + 'static,
    {
        let inner = CsMat::csr_from_dense(dense.view(), epsilon);
        Self { inner }
    }

    /// Return a new CSR matrix containing only the rows in `start..end`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `start > end` or
    /// `end > n_rows()`.
    pub fn row_slice(&self, start: usize, end: usize) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Clone + Default + 'static,
    {
        if start > end {
            return Err(FerroError::InvalidParameter {
                name: "row_slice range".into(),
                reason: format!("start ({start}) must be <= end ({end})"),
            });
        }
        if end > self.n_rows() {
            return Err(FerroError::InvalidParameter {
                name: "row_slice range".into(),
                reason: format!("end ({end}) exceeds n_rows ({})", self.n_rows()),
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
    pub fn mul_scalar(&self, scalar: T) -> CsrMatrix<T>
    where
        T: Copy + Mul<Output = T> + Zero + 'static,
    {
        let new_inner = self.inner.map(|&v| v * scalar);
        Self { inner: new_inner }
    }

    /// Element-wise addition of two CSR matrices with the same shape.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the matrices have different shapes.
    pub fn add(&self, rhs: &CsrMatrix<T>) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Zero + Default + Clone + 'static,
        for<'r> &'r T: Add<&'r T, Output = T>,
    {
        if self.n_rows() != rhs.n_rows() || self.n_cols() != rhs.n_cols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_rows(), self.n_cols()],
                actual: vec![rhs.n_rows(), rhs.n_cols()],
                context: "CsrMatrix::add".into(),
            });
        }
        let result = &self.inner + &rhs.inner;
        Ok(Self { inner: result })
    }

    /// Element-wise subtraction of two CSR matrices with the same shape: `A - B`.
    ///
    /// Mirrors scipy `csr_matrix` subtraction `A - B` (`_sub_sparse`,
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
    pub fn sub(&self, rhs: &CsrMatrix<T>) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Zero + Default + Clone + 'static,
        for<'r> &'r T: std::ops::Sub<&'r T, Output = T>,
    {
        if self.n_rows() != rhs.n_rows() || self.n_cols() != rhs.n_cols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_rows(), self.n_cols()],
                actual: vec![rhs.n_rows(), rhs.n_cols()],
                context: "CsrMatrix::sub".into(),
            });
        }
        let result = &self.inner - &rhs.inner;
        Ok(Self { inner: result })
    }

    /// Element-wise (Hadamard) product of two CSR matrices with the same shape.
    ///
    /// Mirrors scipy `csr_matrix.multiply(other)` (`scipy/sparse/_base.py:490`),
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
    pub fn multiply(&self, rhs: &CsrMatrix<T>) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Zero + Clone + 'static,
        for<'r> &'r T: Mul<&'r T, Output = T>,
    {
        if self.n_rows() != rhs.n_rows() || self.n_cols() != rhs.n_cols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_rows(), self.n_cols()],
                actual: vec![rhs.n_rows(), rhs.n_cols()],
                context: "CsrMatrix::multiply".into(),
            });
        }
        // sprs `mul_mat_same_storage` runs `csmat_binop(|x, y| x * y)` over the
        // two same-storage (both CSR) operands, emitting a non-zero only where
        // both inputs are stored — the element-wise (Hadamard) intersection.
        let result = sprs::binop::mul_mat_same_storage(&self.inner, &rhs.inner);
        Ok(Self { inner: result })
    }

    /// Sparse-sparse matrix product: `A @ B` (scipy `A.dot(B)`).
    ///
    /// Mirrors scipy `csr_matrix @ csr_matrix` (`_matmul_sparse`,
    /// `scipy/sparse/_compressed.py:415`), the SMMP algorithm: the result is the
    /// `(self.n_rows(), rhs.n_cols())` matrix product, so `self.n_cols()` must
    /// equal `rhs.n_rows()`. Computed via the sprs product operator
    /// `&self.inner * &rhs.inner`, which dispatches to `smmp::mul_csr_csr` for
    /// two CSR operands and returns a CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `self.n_cols() != rhs.n_rows()`
    /// (scipy raises `ValueError: dimension mismatch`).
    pub fn matmul(&self, rhs: &CsrMatrix<T>) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Clone + sprs::MulAcc + Zero + Default + Send + Sync + 'static,
    {
        if self.n_cols() != rhs.n_rows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_cols()],
                actual: vec![rhs.n_rows()],
                context: "CsrMatrix::matmul: A.n_cols must equal B.n_rows".into(),
            });
        }
        // Both operands are CSR storage, so the sprs product `&A * &B` dispatches
        // to `smmp::mul_csr_csr` and yields CSR; `to_csr()` is a no-op materialize
        // that guarantees the wrapped `inner` is CSR regardless.
        let inner = (&self.inner * &rhs.inner).to_csr();
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
                context: "CsrMatrix::mul_vec".into(),
            });
        }
        let result = &self.inner * rhs;
        Ok(result)
    }

    /// Sum of all stored values.
    ///
    /// Mirrors scipy `csr_matrix.sum()` with `axis=None`
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
    /// Mirrors scipy `csr_matrix.sum(axis=0)`
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
    /// Mirrors scipy `csr_matrix.sum(axis=1)`
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
    /// Mirrors scipy `csr_matrix.diagonal()` with `k=0`
    /// (`scipy/sparse/_compressed.py:476`): `out[i] == A[i, i]` for
    /// `i in 0..min(n_rows, n_cols)`. Positions absent from the CSR structure
    /// are structural zeros, so `out[i]` defaults to [`T::zero()`].
    #[must_use]
    pub fn diagonal(&self) -> Array1<T>
    where
        T: Copy + Zero,
    {
        let len = self.n_rows().min(self.n_cols());
        let mut out = Array1::<T>::zeros(len);
        for (i, row) in self.inner.outer_iterator().enumerate().take(len) {
            if let Some(&val) = row.get(i) {
                out[i] = val;
            }
        }
        out
    }

    /// Scalar element access: returns `A[i, j]`.
    ///
    /// Mirrors scipy `csr_matrix.__getitem__` for a scalar `(int, int)` key
    /// (`IndexMixin.__getitem__` -> `_get_intXint`, `scipy/sparse/_index.py:29`):
    /// `A[i, j]` returns the stored value, or `0` if the position is
    /// structurally absent (no stored entry). The sprs `CsMat::get(i, j)`
    /// returns `Some(&value)` when stored and `None` when absent, so an absent
    /// position yields [`T::zero()`]. Live oracle (R-CHAR-3): for
    /// `A = [[1,0,2],[0,3,0],[4,0,5]]`, `A[1, 1] == 3`, `A[0, 1] == 0`
    /// (structurally absent), `A[0, 0] == 1`, `A[0, 2] == 2`, `A[2, 0] == 4`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `i >= n_rows()` or
    /// `j >= n_cols()`. scipy raises `IndexError(f'index ({idx}) out of range')`
    /// (`scipy/sparse/_index.py:388`); ferrolearn maps an out-of-bounds index to
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

    /// Single-row extraction: returns row `i` as a `(1, n_cols)` CSR matrix.
    ///
    /// Mirrors scipy `csr_matrix.getrow(i)` (`scipy/sparse/_matrix.py:110` →
    /// `_getrow`, `scipy/sparse/_base.py:1116`), which returns "a copy of row i
    /// of the matrix, as a (1 x n) sparse matrix (row vector)". This is the
    /// single-row special case of [`row_slice`](Self::row_slice): row `i` is the
    /// half-open range `i..i+1`, so it delegates to `self.row_slice(i, i + 1)`,
    /// which already produces the `(1, n_cols)` CSR. Live oracle (R-CHAR-3): for
    /// `A = [[1,0,2],[0,3,0],[4,0,5]]`, `A.getrow(0).toarray() == [[1,0,2]]`
    /// (shape `(1,3)`), `A.getrow(1).toarray() == [[0,3,0]]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `i >= n_rows()`. scipy raises
    /// `IndexError("index out of bounds")` (`scipy/sparse/_base.py:1129`);
    /// ferrolearn maps the out-of-range index to `InvalidParameter` per the crate
    /// error contract (R-DEV-2).
    pub fn getrow(&self, i: usize) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Clone + Default + 'static,
    {
        if i >= self.n_rows() {
            return Err(FerroError::InvalidParameter {
                name: "index".into(),
                reason: format!("row index {i} out of bounds for {} rows", self.n_rows()),
            });
        }
        self.row_slice(i, i + 1)
    }

    /// Single-column extraction: returns column `j` as a `(n_rows, 1)` CSR matrix.
    ///
    /// Mirrors scipy `csr_matrix.getcol(j)` (`scipy/sparse/_matrix.py:104` →
    /// `_getcol`, `scipy/sparse/_base.py:1097`), which returns "a copy of column j
    /// of the matrix, as an (m x 1) sparse matrix (column vector)". The simplest
    /// correct CSR path is via transpose: `Aᵀ` has shape `(n_cols, n_rows)`, its
    /// row `j` is the `(1, n_rows)` CSR `getrow(j)`, and transposing that back is
    /// the `(n_rows, 1)` column `j` of `A`. Live oracle (R-CHAR-3): for
    /// `A = [[1,0,2],[0,3,0],[4,0,5]]`, `A.getcol(0).toarray() == [[1],[0],[4]]`
    /// (shape `(3,1)`), `A.getcol(2).toarray() == [[2],[0],[5]]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `j >= n_cols()`. scipy raises
    /// `IndexError("index out of bounds")` (`scipy/sparse/_base.py:1110`);
    /// ferrolearn maps the out-of-range index to `InvalidParameter` per the crate
    /// error contract (R-DEV-2).
    pub fn getcol(&self, j: usize) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Clone + Default + 'static,
    {
        if j >= self.n_cols() {
            return Err(FerroError::InvalidParameter {
                name: "index".into(),
                reason: format!("col index {j} out of bounds for {} cols", self.n_cols()),
            });
        }
        Ok(self.transpose().getrow(j)?.transpose())
    }

    /// Return a copy of this matrix, preserving **all** stored structure.
    ///
    /// Mirrors scipy `csr_matrix.copy()` (`scipy/sparse/_data.py:94` —
    /// `return self._with_data(self.data.copy(), copy=True)`), which returns an
    /// identical matrix with the same sparsity pattern (`indptr`/`indices`) and
    /// a copy of the `data` array. Every stored entry is preserved verbatim —
    /// including **explicit stored zeros** — without coalescing or reordering.
    /// Equivalent to [`Clone::clone`] (`CsrMatrix` derives `Clone`); provided as
    /// a named method for scipy parity.
    ///
    /// Live oracle (R-CHAR-3): for
    /// `sp.csr_matrix(np.array([[3.7,0,0],[0,-2.9,0],[0,0,5.0]]))`,
    /// `A.copy().nnz == 3`, `A.copy().data == [3.7, -2.9, 5.0]`, and
    /// `A.copy().toarray() == [[3.7,0,0],[0,-2.9,0],[0,0,5.0]]`.
    #[must_use]
    pub fn copy(&self) -> CsrMatrix<T> {
        self.clone()
    }

    /// Return a new CSR matrix with all explicitly-stored zero entries removed.
    ///
    /// Mirrors scipy `csr_matrix.eliminate_zeros()`
    /// (`scipy/sparse/_compressed.py:1025` — `csr_eliminate_zeros(...)` then
    /// `self.prune()`), which drops every stored entry whose value equals `0`
    /// and rebuilds the CSR `(data, indices, indptr)` triple. Walks each row
    /// `r` over its stored slice `indptr[r]..indptr[r+1]` and keeps the
    /// `(col, val)` pair only when `val != 0`, accumulating the kept count into a
    /// fresh `indptr2`. Within-row column order is preserved (scipy keeps the
    /// existing sorted order), so the result stays canonical.
    ///
    /// **Deviation (R-DEV-4):** scipy mutates in place; ferrolearn returns a NEW
    /// matrix (functional style, consistent with the other `CsrMatrix` methods).
    ///
    /// Live oracle (R-CHAR-3): for CSR `data=[3,0,5]`, `indices=[0,1,2]`,
    /// `indptr=[0,1,2,3]`, shape `(3,3)`, `eliminate_zeros()` yields `nnz == 2`,
    /// `data == [3, 5]`, `indices == [0, 2]`, `indptr == [0, 1, 1, 2]`, dense
    /// `[[3,0,0],[0,0,0],[0,0,5]]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] propagated from [`new`](Self::new); infallible for
    /// any structurally valid `CsrMatrix` (filtering keeps every kept index in
    /// bounds of the unchanged shape and preserves sorted within-row order).
    pub fn eliminate_zeros(&self) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Clone + Zero + PartialEq,
    {
        let zero = T::zero();
        let indptr = self.indptr();
        let indices = self.indices();
        let data = self.data();
        let n_rows = self.n_rows();

        let mut indptr2 = Vec::with_capacity(n_rows + 1);
        let mut indices2 = Vec::new();
        let mut data2 = Vec::new();
        indptr2.push(0usize);
        for r in 0..n_rows {
            for k in indptr[r]..indptr[r + 1] {
                if data[k] != zero {
                    indices2.push(indices[k]);
                    data2.push(data[k].clone());
                }
            }
            indptr2.push(data2.len());
        }
        Self::new(n_rows, self.n_cols(), indptr2, indices2, data2)
    }

    /// Return a new CSR matrix in canonical form, summing duplicate
    /// `(row, col)` entries.
    ///
    /// Mirrors scipy `csr_matrix.sum_duplicates()`
    /// (`scipy/sparse/_compressed.py:1063` — `sort_indices()` then
    /// `csr_sum_duplicates(...)` then `self.prune()`), which sorts each row's
    /// column indices and adds together any entries that share the same
    /// `(row, col)` position. For each row `r`, the stored slice
    /// `indptr[r]..indptr[r+1]` is accumulated into a
    /// [`BTreeMap<usize, T>`](std::collections::BTreeMap) keyed by column, summing
    /// on collision; the `BTreeMap` yields columns in sorted order, giving scipy's
    /// canonical `(sorted indices, no duplicates)` form. **Every** accumulated
    /// entry is kept — including positions whose duplicates sum to `0` — because
    /// canonicalization is the only job here; dropping zeros is
    /// [`eliminate_zeros`](Self::eliminate_zeros)' responsibility (scipy's
    /// `sum_duplicates` likewise keeps zero sums and only `prune`s the
    /// allocation).
    ///
    /// **Deviation (R-DEV-4):** scipy mutates in place; ferrolearn returns a NEW
    /// matrix (functional style, consistent with the other `CsrMatrix` methods).
    ///
    /// Live oracle (R-CHAR-3): for CSR `data=[3,5,2,1]`, `indices=[0,0,2,2]`,
    /// `indptr=[0,2,2,4]`, shape `(3,3)` (row 0 has col 0 twice `{3,5}`, row 2
    /// has col 2 twice `{2,1}`), `sum_duplicates()` yields `nnz == 2`,
    /// `indptr == [0,1,1,2]`, `indices == [0,2]`, `data == [8,3]`; for
    /// `data=[4,-4,7]`, `indices=[0,0,1]`, `indptr=[0,2,3]`, shape `(2,2)`
    /// (row 0 has col 0 twice `{4,-4}`), it yields `nnz == 2`, `indices == [0,1]`,
    /// `data == [0,7]` — the zero-sum `(0,0)` entry is **preserved**.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] propagated from [`new`](Self::new); infallible for
    /// any structurally valid `CsrMatrix` (the accumulated columns stay in bounds
    /// of the unchanged shape and `BTreeMap` ordering keeps within-row indices
    /// sorted).
    pub fn sum_duplicates(&self) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Copy + Zero + Add<Output = T>,
    {
        let indptr = self.indptr();
        let indices = self.indices();
        let data = self.data();
        let n_rows = self.n_rows();

        let mut indptr2 = Vec::with_capacity(n_rows + 1);
        let mut indices2 = Vec::new();
        let mut data2 = Vec::new();
        indptr2.push(0usize);
        for r in 0..n_rows {
            let mut acc: BTreeMap<usize, T> = BTreeMap::new();
            for k in indptr[r]..indptr[r + 1] {
                let entry = acc.entry(indices[k]).or_insert_with(T::zero);
                *entry = *entry + data[k];
            }
            for (col, val) in acc {
                indices2.push(col);
                data2.push(val);
            }
            indptr2.push(data2.len());
        }
        Self::new(n_rows, self.n_cols(), indptr2, indices2, data2)
    }
}

impl<T> CsrMatrix<T>
where
    T: Clone,
{
    /// Cast every stored value to a new scalar type `U` via a caller-supplied
    /// closure, preserving the CSR sparsity structure (`indptr`/`indices`,
    /// shape, nnz).
    ///
    /// Mirrors scipy `csr_matrix.astype(dtype)` (`scipy/sparse/_data.py:69` —
    /// `self._with_data(self.data.astype(dtype, ...))`), which C-casts every
    /// stored value in `self.data` to the requested numpy dtype while keeping the
    /// `indptr`/`indices` structure and `shape` unchanged. scipy selects the cast
    /// from a runtime numpy dtype object; Rust has no runtime dtype, so this is a
    /// **deviation** (R-DEV-4): the caller supplies the element cast as a closure.
    /// A plain `as`-cast closure (e.g. `|&v| v as i64`) reproduces numpy's C-cast
    /// semantics, including float→int **truncation toward zero**.
    ///
    /// The `indptr` (row pointers) and `indices` (column indices) arrays, the
    /// `(n_rows, n_cols)` shape, and the stored count are copied verbatim — only
    /// the data array changes type — so explicit stored zeros are preserved (no
    /// coalescing). Rebuilds a `CsrMatrix<U>` via [`new`](Self::new) from the
    /// original `indptr`/`indices` and the cast data.
    ///
    /// Live oracle (R-CHAR-3): for
    /// `sp.csr_matrix(np.array([[3.7,0,0],[0,-2.9,0],[0,0,5.0]]))` (CSR
    /// `data=[3.7,-2.9,5.0]`, `indptr=[0,1,2,3]`, `indices=[0,1,2]`),
    /// `astype(np.int64)` yields `data == [3, -2, 5]` (truncated toward zero) with
    /// `indptr == [0,1,2,3]`, `indices == [0,1,2]`, dense
    /// `[[3,0,0],[0,-2,0],[0,0,5]]`; `astype(np.float32)` yields
    /// `data == [3.7f32, -2.9f32, 5.0f32]` with the same structure.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] propagated from [`new`](Self::new); infallible for
    /// any structurally valid `CsrMatrix` (the unchanged `indptr`/`indices` stay
    /// valid for the unchanged shape).
    pub fn astype<U, Fc>(&self, cast: Fc) -> Result<CsrMatrix<U>, FerroError>
    where
        U: Clone,
        Fc: Fn(&T) -> U,
    {
        let data: Vec<U> = self.data().iter().map(&cast).collect();
        CsrMatrix::<U>::new(
            self.n_rows(),
            self.n_cols(),
            self.indptr(),
            self.indices().to_vec(),
            data,
        )
    }
}

impl<T> CsrMatrix<T>
where
    T: Copy + Zero + PartialOrd,
{
    /// Maximum over all elements (scipy `axis=None`), folding in implicit zeros.
    ///
    /// Mirrors scipy `csr_matrix.max()` via the `_minmax_mixin._min_or_max`
    /// machinery with `axis=None` (`scipy/sparse/_data.py:208`-`:224`). scipy
    /// reduces over the stored data (`m = min_or_max.reduce(self._deduped_data())`,
    /// `:221`) and then, when the matrix is **not fully dense**, folds an implicit
    /// zero into the result (`if self.nnz != math.prod(self.shape): m =
    /// min_or_max(zero, m)`, `:222`-`:223`). An empty matrix (`nnz == 0`) returns
    /// `zero` (`:219`-`:220`). Here `nnz` is the number of stored entries
    /// ([`data().len()`](Self::data)); the matrix is fully dense iff
    /// `nnz == n_rows * n_cols`, in which case no implicit zero exists to fold.
    ///
    /// Live oracle (R-CHAR-3): `csr_matrix(diag(-3,-1,-5))` 3x3 `.max() == 0.0`
    /// (implicit zero wins over the all-negative stored data); `csr_matrix([[2,7]])`
    /// `.max() == 7.0` (fully dense, no implicit zero).
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
    /// Mirrors scipy `csr_matrix.min()` via the `_minmax_mixin._min_or_max`
    /// machinery with `axis=None` (`scipy/sparse/_data.py:208`-`:224`). scipy
    /// reduces over the stored data (`m = min_or_max.reduce(self._deduped_data())`,
    /// `:221`) and then, when the matrix is **not fully dense**, folds an implicit
    /// zero into the result (`if self.nnz != math.prod(self.shape): m =
    /// min_or_max(zero, m)`, `:222`-`:223`). An empty matrix (`nnz == 0`) returns
    /// `zero` (`:219`-`:220`). Here `nnz` is the number of stored entries
    /// ([`data().len()`](Self::data)); the matrix is fully dense iff
    /// `nnz == n_rows * n_cols`, in which case no implicit zero exists to fold.
    ///
    /// Live oracle (R-CHAR-3): `csr_matrix(diag(3,1,5))` 3x3 `.min() == 0.0`
    /// (implicit zero wins over the all-positive stored data); `csr_matrix([[2,7]])`
    /// `.min() == 2.0` (fully dense, no implicit zero).
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

impl<T> CsrMatrix<T>
where
    T: Float + Send + Sync + num_traits::Signed + 'static,
{
    /// Construct a [`CsrMatrix`] from a dense [`Array2<T>`], treating entries
    /// with absolute value at or below `T::epsilon()` as structural zeros.
    pub fn from_dense_float(dense: &ArrayView2<'_, T>) -> Self {
        CsrMatrix::from_dense(dense, T::epsilon())
    }
}

/// Implements [`Dataset`] so that `CsrMatrix<F>` can be passed to any
/// ferrolearn algorithm that accepts a dataset.
///
/// - `n_samples()` — number of rows (one sample per row).
/// - `n_features()` — number of columns (one feature per column).
/// - `is_sparse()` — always `true`.
impl<F> Dataset for CsrMatrix<F>
where
    F: Float + Send + Sync + 'static,
{
    fn n_samples(&self) -> usize {
        self.n_rows()
    }

    fn n_features(&self) -> usize {
        self.n_cols()
    }

    fn is_sparse(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn sample_csr() -> CsrMatrix<f64> {
        // 3x3 sparse matrix:
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        CsrMatrix::new(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap()
    }

    #[test]
    fn test_new_valid() {
        let m = sample_csr();
        assert_eq!(m.n_rows(), 3);
        assert_eq!(m.n_cols(), 3);
        assert_eq!(m.nnz(), 5);
    }

    #[test]
    fn test_new_invalid() {
        // Wrong indptr length (needs n_rows+1 = 3, not 2)
        let res = CsrMatrix::<f64>::new(2, 2, vec![0, 1], vec![0], vec![1.0]);
        assert!(res.is_err());
    }

    #[test]
    fn test_to_dense() {
        let m = sample_csr();
        let d = m.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 1.0);
        assert_abs_diff_eq!(d[[0, 1]], 0.0);
        assert_abs_diff_eq!(d[[0, 2]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 3.0);
        assert_abs_diff_eq!(d[[2, 0]], 4.0);
        assert_abs_diff_eq!(d[[2, 2]], 5.0);
    }

    #[test]
    fn test_from_dense() {
        let dense = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let m = CsrMatrix::from_dense(&dense.view(), 0.0);
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
        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let dense = csr.to_dense();
        assert_abs_diff_eq!(dense[[0, 0]], 1.0);
        assert_abs_diff_eq!(dense[[1, 2]], 4.0);
        assert_abs_diff_eq!(dense[[2, 1]], 7.0);
        assert_abs_diff_eq!(dense[[0, 1]], 0.0);
    }

    #[test]
    fn test_to_coo_roundtrip() {
        let csr = sample_csr();
        let coo = csr.to_coo();
        let back = CsrMatrix::from_coo(&coo).unwrap();
        let d = back.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 1.0);
        assert_abs_diff_eq!(d[[2, 2]], 5.0);
    }

    #[test]
    fn test_csr_csc_roundtrip() {
        let csr = sample_csr();
        let csc = csr.to_csc();
        let back = CsrMatrix::from_csc(&csc).unwrap();
        assert_eq!(back.to_dense(), csr.to_dense());
    }

    #[test]
    fn test_row_slice() {
        let m = sample_csr();
        let sliced = m.row_slice(0, 2).unwrap();
        assert_eq!(sliced.n_rows(), 2);
        assert_eq!(sliced.n_cols(), 3);
        let d = sliced.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 1.0);
        assert_abs_diff_eq!(d[[1, 1]], 3.0);
    }

    #[test]
    fn test_row_slice_empty() {
        let m = sample_csr();
        let sliced = m.row_slice(1, 1).unwrap();
        assert_eq!(sliced.n_rows(), 0);
    }

    #[test]
    fn test_row_slice_invalid() {
        let m = sample_csr();
        assert!(m.row_slice(2, 1).is_err());
        assert!(m.row_slice(0, 4).is_err());
    }

    #[test]
    fn test_mul_scalar() {
        let m = sample_csr();
        let m2 = m.mul_scalar(2.0);
        let d = m2.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 6.0);
    }

    #[test]
    fn test_scale_in_place() {
        let mut m = sample_csr();
        m.scale(3.0);
        let d = m.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 3.0);
        assert_abs_diff_eq!(d[[2, 2]], 15.0);
    }

    #[test]
    fn test_add() {
        let m = sample_csr();
        let sum = m.add(&m).unwrap();
        let d = sum.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 6.0);
    }

    #[test]
    fn test_add_shape_mismatch() {
        let m1 = sample_csr();
        let m2 = CsrMatrix::new(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
        assert!(m1.add(&m2).is_err());
    }

    #[test]
    fn test_mul_vec() {
        let m = sample_csr();
        // [1 0 2]   [1]   [7]
        // [0 3 0] * [2] = [6]
        // [4 0 5]   [3]   [19]
        let v = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let result = m.mul_vec(&v).unwrap();
        assert_abs_diff_eq!(result[0], 7.0);
        assert_abs_diff_eq!(result[1], 6.0);
        assert_abs_diff_eq!(result[2], 19.0);
    }

    #[test]
    fn test_mul_vec_shape_mismatch() {
        let m = sample_csr();
        let v = Array1::from(vec![1.0_f64, 2.0]);
        assert!(m.mul_vec(&v).is_err());
    }

    #[test]
    fn test_dataset_trait() {
        let m = sample_csr();
        assert_eq!(m.n_samples(), 3);
        assert_eq!(m.n_features(), 3);
        assert!(m.is_sparse());
    }

    #[test]
    fn test_dataset_trait_object() {
        use ferrolearn_core::Dataset;
        let m: CsrMatrix<f64> = sample_csr();
        let d: &dyn Dataset = &m;
        assert_eq!(d.n_samples(), 3);
        assert!(d.is_sparse());
    }

    // REQ-MISSING-REDUCE — live scipy 1.x oracle (R-CHAR-3). Expected values
    // from `python3 -c "import numpy as np, scipy.sparse as sp;
    //   A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]));
    //   print(A.sum(), A.sum(axis=0).tolist(), A.sum(axis=1).tolist(),
    //         A.diagonal().tolist())"`
    //   -> 15.0 [[5.0,3.0,7.0]] [[3.0],[3.0],[9.0]] [1.0,3.0,5.0]
    // and B=[[1,2,3],[4,5,6]] -> B.diagonal().tolist() == [1.0,5.0].

    fn sample_csr_b() -> CsrMatrix<f64> {
        // 2x3 dense matrix B = [[1,2,3],[4,5,6]] (no zeros). Built via
        // from_dense (infallible, no unwrap) to match the test idiom.
        let dense = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        CsrMatrix::from_dense(&dense.view(), 0.0)
    }

    #[test]
    fn csr_sum_matches_scipy() {
        let m = sample_csr();
        assert_abs_diff_eq!(m.sum(), 15.0);
    }

    #[test]
    fn csr_sum_axis0_matches_scipy() {
        let m = sample_csr();
        assert_eq!(m.sum_axis0(), array![5.0, 3.0, 7.0]);
    }

    #[test]
    fn csr_sum_axis1_matches_scipy() {
        let m = sample_csr();
        assert_eq!(m.sum_axis1(), array![3.0, 3.0, 9.0]);
    }

    #[test]
    fn csr_diagonal_matches_scipy() {
        let m = sample_csr();
        assert_eq!(m.diagonal(), array![1.0, 3.0, 5.0]);
    }

    #[test]
    fn csr_diagonal_non_square() {
        let m = sample_csr_b();
        assert_eq!(m.diagonal(), array![1.0, 5.0]);
    }

    // REQ-MISSING-TRANSPOSE — live scipy oracle (R-CHAR-3). Expected values from
    // `python3 -c "import numpy as np, scipy.sparse as sp;
    //   A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]));
    //   B=sp.csr_matrix(np.array([[1.,2,3],[4,5,6]]));
    //   print(A.T.toarray().tolist(), B.T.toarray().tolist())"`
    //   -> [[1,0,4],[0,3,0],[2,0,5]] [[1,4],[2,5],[3,6]].

    #[test]
    fn csr_transpose_matches_scipy() {
        let a = sample_csr();
        let at = a.transpose();
        assert_eq!(at.n_rows(), 3);
        assert_eq!(at.n_cols(), 3);
        assert_eq!(
            at.to_dense(),
            array![[1.0, 0.0, 4.0], [0.0, 3.0, 0.0], [2.0, 0.0, 5.0]]
        );
    }

    #[test]
    fn csr_transpose_non_square() {
        let b = sample_csr_b();
        let bt = b.transpose();
        assert_eq!(bt.n_rows(), 3);
        assert_eq!(bt.n_cols(), 2);
        assert_eq!(bt.to_dense(), array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    }

    #[test]
    fn csr_transpose_twice_roundtrip() {
        let b = sample_csr_b();
        let btt = b.transpose().transpose();
        assert_eq!(btt.n_rows(), 2);
        assert_eq!(btt.n_cols(), 3);
        assert_eq!(btt.to_dense(), b.to_dense());
    }

    #[test]
    fn test_from_dense_float() {
        let dense = array![[1.0_f64, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let csr = CsrMatrix::from_dense_float(&dense.view());
        assert_eq!(csr.nnz(), 2);
        let back = csr.to_dense();
        assert_abs_diff_eq!(back[[0, 0]], 1.0);
        assert_abs_diff_eq!(back[[1, 2]], 2.0);
    }
}

/// Kani proof harnesses for CsrMatrix structural invariants.
///
/// These proofs verify that after construction via `new()`, `from_coo()`, and
/// `add()`, the underlying CSR representation satisfies all structural
/// invariants:
///
/// - `indptr.len() == n_rows + 1`
/// - `indptr` is monotonically non-decreasing
/// - All column indices are less than `n_cols`
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
    /// Maximum number of non-zero entries for symbolic exploration.
    const MAX_NNZ: usize = 4;

    /// Helper: assert all CSR structural invariants on the inner `CsMat`.
    fn assert_csr_invariants<T>(m: &CsrMatrix<T>) {
        let inner = m.inner();

        // Invariant 1: indptr length == n_rows + 1
        let indptr = inner.indptr();
        let indptr_raw = indptr.raw_storage();
        assert!(indptr_raw.len() == m.n_rows() + 1);

        // Invariant 2: indptr is monotonically non-decreasing
        for i in 0..m.n_rows() {
            assert!(indptr_raw[i] <= indptr_raw[i + 1]);
        }

        // Invariant 3: all column indices < n_cols
        let indices = inner.indices();
        for &col_idx in indices {
            assert!(col_idx < m.n_cols());
        }

        // Invariant 4: indices.len() == data.len()
        assert!(inner.indices().len() == inner.data().len());
    }

    /// Verify `indptr.len() == n_rows + 1` after `new()` with a symbolic
    /// empty matrix of arbitrary dimensions.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_new_indptr_length() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Build a valid empty CSR matrix
        let indptr = vec![0usize; n_rows + 1];
        let indices: Vec<usize> = vec![];
        let data: Vec<i32> = vec![];

        if let Ok(m) = CsrMatrix::new(n_rows, n_cols, indptr, indices, data) {
            let inner_indptr = m.inner().indptr();
            assert!(inner_indptr.raw_storage().len() == n_rows + 1);
        }
    }

    /// Verify indptr monotonicity after `new()` with a symbolic single-entry
    /// matrix.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_new_indptr_monotonic() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Place a single non-zero at a symbolic valid position
        let row: usize = kani::any();
        let col: usize = kani::any();
        kani::assume(row < n_rows);
        kani::assume(col < n_cols);

        // Build indptr for a single entry at (row, col)
        let mut indptr = vec![0usize; n_rows + 1];
        for i in (row + 1)..=n_rows {
            indptr[i] = 1;
        }
        let indices = vec![col];
        let data = vec![42i32];

        if let Ok(m) = CsrMatrix::new(n_rows, n_cols, indptr, indices, data) {
            let inner_indptr = m.inner().indptr().raw_storage().to_vec();
            for i in 0..m.n_rows() {
                assert!(inner_indptr[i] <= inner_indptr[i + 1]);
            }
        }
    }

    /// Verify all column indices < n_cols after `new()`.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_new_column_indices_in_bounds() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let col: usize = kani::any();
        let row: usize = kani::any();
        kani::assume(row < n_rows);
        kani::assume(col < n_cols);

        let mut indptr = vec![0usize; n_rows + 1];
        for i in (row + 1)..=n_rows {
            indptr[i] = 1;
        }
        let indices = vec![col];
        let data = vec![1i32];

        if let Ok(m) = CsrMatrix::new(n_rows, n_cols, indptr, indices, data) {
            for &c in m.inner().indices() {
                assert!(c < m.n_cols());
            }
        }
    }

    /// Verify `indices.len() == data.len()` after `new()`.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_new_indices_data_same_length() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Try empty matrix
        let indptr = vec![0usize; n_rows + 1];
        let indices: Vec<usize> = vec![];
        let data: Vec<i32> = vec![];

        if let Ok(m) = CsrMatrix::new(n_rows, n_cols, indptr, indices, data) {
            assert!(m.inner().indices().len() == m.inner().data().len());
        }
    }

    /// Verify that `new()` rejects inputs where indices and data have
    /// mismatched lengths.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_new_rejects_mismatched_lengths() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // indices has 1 element, data has 0 — must fail
        let indptr = vec![0usize; n_rows + 1];
        let indices = vec![0usize];
        let data: Vec<i32> = vec![];

        let result = CsrMatrix::new(n_rows, n_cols, indptr, indices, data);
        assert!(result.is_err());
    }

    /// Verify all structural invariants after `from_coo()` with symbolic
    /// entries.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_from_coo_invariants() {
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

        if let Ok(csr) = CsrMatrix::from_coo(&coo) {
            assert_csr_invariants(&csr);
            assert!(csr.n_rows() == n_rows);
            assert!(csr.n_cols() == n_cols);
        }
    }

    /// Verify that `add()` preserves shape and structural invariants.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_add_preserves_invariants() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Build two valid empty CSR matrices of the same shape
        let indptr = vec![0usize; n_rows + 1];
        let a = CsrMatrix::<i32>::new(n_rows, n_cols, indptr.clone(), vec![], vec![]);
        let b = CsrMatrix::<i32>::new(n_rows, n_cols, indptr, vec![], vec![]);

        if let (Ok(a), Ok(b)) = (a, b) {
            if let Ok(sum) = a.add(&b) {
                // Shape is preserved
                assert!(sum.n_rows() == n_rows);
                assert!(sum.n_cols() == n_cols);
                // Structural invariants hold
                assert_csr_invariants(&sum);
            }
        }
    }

    /// Verify that `add()` with non-empty matrices preserves invariants.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_add_nonempty_preserves_invariants() {
        // Fixed 2x2 matrices with one entry each in different positions
        let a = CsrMatrix::<i32>::new(2, 2, vec![0, 1, 1], vec![0], vec![1]);
        let b = CsrMatrix::<i32>::new(2, 2, vec![0, 0, 1], vec![1], vec![2]);

        if let (Ok(a), Ok(b)) = (a, b) {
            if let Ok(sum) = a.add(&b) {
                assert!(sum.n_rows() == 2);
                assert!(sum.n_cols() == 2);
                assert_csr_invariants(&sum);
            }
        }
    }

    /// Verify `mul_vec()` output has correct dimension and does not panic.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_mul_vec_output_dimension() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Empty matrix for tractable verification
        let indptr = vec![0usize; n_rows + 1];
        let m = CsrMatrix::<f64>::new(n_rows, n_cols, indptr, vec![], vec![]);

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
    fn csr_mul_vec_rejects_wrong_dimension() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let indptr = vec![0usize; n_rows + 1];
        let m = CsrMatrix::<f64>::new(n_rows, n_cols, indptr, vec![], vec![]);

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
    fn csr_mul_vec_nonempty_no_oob() {
        // 2x3 matrix with entries at (0,1) and (1,2)
        let m = CsrMatrix::<f64>::new(2, 3, vec![0, 1, 2], vec![1, 2], vec![3.0, 4.0]);
        if let Ok(m) = m {
            let v = Array1::from(vec![1.0, 2.0, 3.0]);
            if let Ok(result) = m.mul_vec(&v) {
                assert!(result.len() == 2);
            }
        }
    }
}
