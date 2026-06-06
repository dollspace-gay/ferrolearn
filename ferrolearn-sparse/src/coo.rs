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
//! **6 SHIPPED / 2 NOT-STARTED** (REQ-MISSING-METHODS is a SPLIT: conversion +
//! transpose `to_csr`/`to_csc`/`transpose` SHIPPED; the rest NOT-STARTED).
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-CONSTRUCT (from_triplets/push/shape) | SHIPPED | `from_triplets`/`new`/`with_capacity`/`push` mirror `coo_matrix((data,(row,col)),shape)`; `to_dense()` matches scipy `.toarray()`; `n_rows`/`n_cols` match `.shape`. Guards `coo_construct_to_dense_matches_scipy_toarray`/`coo_shape_matches_scipy`. |
//! | REQ-TOARRAY-DUP (duplicate summing) | SHIPPED | `to_dense()` sums duplicate entries at the same position, matching scipy `.toarray()` (e.g. `(0,0)` twice → 3). Guards `coo_to_dense_duplicate_summed_matches_scipy`/`coo_push_duplicate_nnz_counts_both`. |
//! | REQ-NNZ (stored-count semantics) | SHIPPED | `nnz()` counts STORED entries incl. duplicates (matching scipy coo `.nnz = len(data)`); CSR conversion coalesces (nnz 3→2). Guard `coo_nnz_counts_duplicates_csr_coalesces`. |
//! | REQ-ERR (construction validation) | SHIPPED | `from_triplets`/`push` return `Err(FerroError)` on out-of-bounds index / mismatched lengths, at the same point scipy raises `ValueError`. Guards `coo_from_triplets_*_is_err`/`coo_push_out_of_bounds_is_err`. |
//! | REQ-CONSUMER (production consumer) | SHIPPED | consumed in-crate by `csr.rs`/`csc.rs` (COO→CSR/CSC conversion) and `helpers.rs` (`eye`/`diags`/`hstack`/`vstack` build via COO) — real non-test callers (S5 crate boundary). |
//! | REQ-API-ACCESSORS (shape/data/row/col) | SHIPPED (#1996) | first-class `shape()`/`data()`/`row()`/`col()` accessors mirror scipy `.shape` (`_coo.py:32`/`:39` ctor tuple) and `.data`/`.row`/`.col` (`_coo.py:64`/`:106`/`:122`), the COO `(data, (row, col))` triple. `shape()` → `(n_rows, n_cols)`; `data()` → `&[T]` (`inner.data()`); `row()` → `&[usize]` (`inner.row_inds()`); `col()` → `&[usize]` (`inner.col_inds()`) — all borrow `&self` (sprs `TriMat` accessors return slices). Live oracle (R-CHAR-3): `coo_matrix(([3,5,2],([0,2,1],[0,1,2])),shape=(3,3))` → `shape=(3,3)`, `data=[3,5,2]`, `row=[0,2,1]`, `col=[0,1,2]` (insertion order preserved). Guard `coo_shape_data_row_col_match_scipy`. |
//! | REQ-MISSING-METHODS (COO methods) | SHIPPED (#1997) | conversion + transpose: `to_csr()`/`to_csc()` delegate to `CsrMatrix::from_coo`/`CscMatrix::from_coo` (both summing duplicate `(row,col)` entries), mirroring scipy `coo_matrix.tocsr` (`_coo.py:349`)/`tocsc` (`_coo.py:316`); `transpose()` swaps the row/col index arrays and the `(M,N)` shape to `(n_cols,n_rows)`, mirroring scipy `coo_matrix.transpose`/`.T` with `axes=(1,0)` (`_coo.py:229` — `permuted_coords`/`permuted_shape`). Live oracle (R-CHAR-3): `from_triplets(3,3,[0,2,1],[0,1,2],[3,5,2])` → `tocsr/tocsc.toarray()`=`[[3,0,0],[0,0,2],[0,5,0]]`, `transpose.toarray()`=`[[3,0,0],[0,0,5],[0,2,0]]` (shape `(3,3)`); non-square `from_triplets(2,3,[0,1],[2,0],[7,9])` → `transpose.toarray()`=`[[0,9],[0,0],[7,0]]` (shape `(3,2)`). Guards `coo_to_csr_matches_scipy`/`coo_to_csc_matches_scipy`/`coo_transpose_matches_scipy`/`coo_transpose_non_square`. Reductions (#1997, continuing): `sum()` (Σ all `data()` from `T::zero()`, scipy `_coo.py:1429` axis=None), `sum_axis0()` (column sums, length `n_cols`), `sum_axis1()` (row sums, length `n_rows`) — each computed from the triplets (`row()`/`col()`/`data()`), so DUPLICATE `(row,col)` entries are SUMMED (matching scipy); `diagonal()` (length `min(n_rows,n_cols)`, `out[r] += v` when `r==c`, diagonal duplicates summed, scipy `_coo.py:458`). Bounds `T: Copy + Zero + Add<Output=T>`. Live oracle (R-CHAR-3): `from_triplets(3,3,[0,2,1],[0,1,2],[3,5,2])` = `[[3,0,0],[0,0,2],[0,5,0]]` → `sum()`=10.0, `sum_axis0()`=`[3,5,2]`, `sum_axis1()`=`[3,2,5]`, `diagonal()`=`[3,0,0]`; duplicate `coo_matrix(([3,5],([0,0],[0,0])),shape=(2,2))` → `sum()`=8.0, `diagonal()`=`[8,0]`. Guards `coo_sum_matches_scipy`/`coo_sum_axis0_matches_scipy`/`coo_sum_axis1_matches_scipy`/`coo_diagonal_matches_scipy`/`coo_sum_diagonal_sum_duplicates`. `copy()` (`scipy/sparse/_data.py:94`) returns an identical matrix preserving all stored entries incl. explicit zeros/duplicates (bound `T: Clone`); `eliminate_zeros()` (`scipy/sparse/_coo.py:798`) returns a new matrix keeping only triplets with `data[k] != T::zero()` (bound `T: Clone + Zero + PartialEq`). Guards `coo_copy_preserves_stored_zero`/`coo_eliminate_zeros_matches_scipy`. Scalar reductions (#1997): `max()`/`min()` (`scipy/sparse/_data.py:208`-`:224`, `_minmax_mixin._min_or_max` axis=None) reduce over `data()` and fold an implicit `T::zero()` when the matrix is not fully dense (`data().len() < n_rows*n_cols`, scipy `:222`), returning `zero` for an empty matrix (scipy `:219`); bound `T: Copy + Zero + PartialOrd`. Live oracle (R-CHAR-3): all-negative diag `([-3,-1,-5],([0,1,2],[0,1,2]),(3,3))` → `max()`=0.0 (implicit zero wins), `min()`=-5.0; all-positive diag `([3,1,5],…)` → `max()`=5.0, `min()`=0.0; fully-dense `([2,7],([0,0],[0,1]),(1,2))` → `max()`=7.0, `min()`=2.0 (no implicit zero). Guards `coo_max_folds_implicit_zero`/`coo_min_folds_implicit_zero`/`coo_max_min_dense_no_implicit_zero`. `sum_duplicates()` (`scipy/sparse/_coo.py:768`, via `_sum_duplicates` `:779`) returns a new matrix collapsing duplicate `(row,col)` entries by SUMMING their values, in canonical `(row,col)`-sorted order (BTreeMap accumulation), and PRESERVES zero-sum entries (canonicalization only — removing zeros is `eliminate_zeros`'s separate job); bound `T: Copy + Zero + Add<Output=T>`. Live oracle (R-CHAR-3): `coo_matrix(([3,5,2,1],([0,0,2,2],[0,0,2,2])),(3,3))` → `nnz=2`, `row=[0,2]`, `col=[0,2]`, `data=[8,3]`; `coo_matrix(([4,-4,7],([0,0,1],[0,0,1])),(2,2))` → `nnz=2`, `data=[0,7]` (the zero-sum `(0,0)` preserved). Guards `coo_sum_duplicates_matches_scipy`/`coo_sum_duplicates_preserves_zero_sum`. `astype<U,Fc>(cast)` (`scipy/sparse/_data.py:69`) casts every stored value to a new type `U` via a caller-supplied closure (Rust has no runtime dtype object — R-DEV-4 deviation), preserving the row/col index structure, `(n_rows,n_cols)` shape, and `nnz` (no bounds on `U`, since `from_triplets` is unbounded). A plain `as`-cast closure reproduces numpy's C-cast float→int truncation toward zero. Live oracle (R-CHAR-3): `coo_matrix(([3.7,-2.9,5.0],([0,1,2],[0,1,2])),shape=(3,3))` → `astype(np.int64)` `data=[3,-2,5]` (truncated), `astype(np.float32)` `data=[3.7f32,-2.9f32,5.0f32]`, both with `row=[0,1,2]`/`col=[0,1,2]`/`nnz=3`. Guards `coo_astype_float_to_int_truncates`/`coo_astype_to_f32_preserves_structure`. Sub-note: `.multiply`/`.dot`/arithmetic/`.power` remain NOT-STARTED. |
//! | REQ-FERRAY (ferray sparse substrate) | NOT-STARTED | `sprs::TriMat` + `ndarray` vs ferray's sparse COO analog (R-SUBSTRATE-1; ferray has no sparse layer yet). Blocker #1998. |

use std::collections::BTreeMap;
use std::ops::Add;

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Zero;
use sprs::{SpIndex, TriMat};

use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;

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

    /// Returns the matrix shape as a `(n_rows, n_cols)` tuple.
    ///
    /// Mirrors scipy `coo_matrix.shape` (the `self._shape` tuple set in the
    /// `_coo_base` constructor, `scipy/sparse/_coo.py:32`/`:39`/`:58`), the
    /// `(M, N)` dimension pair. Equivalent to `(self.n_rows(), self.n_cols())`.
    /// Live oracle: `sp.coo_matrix((data,(row,col)),shape=(3,3)).shape == (3, 3)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows(), self.n_cols())
    }

    /// Returns the stored values, one per stored triplet, in insertion order.
    ///
    /// Mirrors scipy `coo_matrix.data` (`scipy/sparse/_coo.py:64` —
    /// `self.data = getdata(obj, ...)`), the `data` array of the COO
    /// `(data, (row, col))` triple ferrolearn stores identically. Borrows the
    /// underlying `sprs::TriMat::data()` slice (`&[T]`). Length equals
    /// [`nnz`](Self::nnz) (duplicates counted, no coalescing). Live oracle:
    /// `sp.coo_matrix(([3.,5.,2.],([0,2,1],[0,1,2])),shape=(3,3)).data == [3,5,2]`.
    #[must_use]
    pub fn data(&self) -> &[T] {
        self.inner.data()
    }

    /// Returns the row index of each stored triplet, aligned with
    /// [`data`](Self::data) and [`col`](Self::col), in insertion order.
    ///
    /// Mirrors scipy `coo_matrix.row` (`scipy/sparse/_coo.py:106` —
    /// `return self.coords[-2]`), the row coordinate of each stored entry.
    /// Borrows the underlying `sprs::TriMat::row_inds()` slice (`&[usize]`).
    /// Length equals [`nnz`](Self::nnz). Live oracle:
    /// `sp.coo_matrix(([3.,5.,2.],([0,2,1],[0,1,2])),shape=(3,3)).row == [0,2,1]`.
    #[must_use]
    pub fn row(&self) -> &[usize] {
        self.inner.row_inds()
    }

    /// Returns the column index of each stored triplet, aligned with
    /// [`data`](Self::data) and [`row`](Self::row), in insertion order.
    ///
    /// Mirrors scipy `coo_matrix.col` (`scipy/sparse/_coo.py:122` —
    /// `return self.coords[-1]`), the column coordinate of each stored entry.
    /// Borrows the underlying `sprs::TriMat::col_inds()` slice (`&[usize]`).
    /// Length equals [`nnz`](Self::nnz). Live oracle:
    /// `sp.coo_matrix(([3.,5.,2.],([0,2,1],[0,1,2])),shape=(3,3)).col == [0,1,2]`.
    #[must_use]
    pub fn col(&self) -> &[usize] {
        self.inner.col_inds()
    }
}

impl<T> CooMatrix<T> {
    /// Cast every stored value to a new scalar type `U` via a caller-supplied
    /// closure, preserving the sparsity structure (row/col indices, shape, nnz).
    ///
    /// Mirrors scipy `coo_matrix.astype(dtype)` (`scipy/sparse/_data.py:69` —
    /// `self._with_data(self.data.astype(dtype, ...))`), which casts every stored
    /// value in `self.data` to the requested numpy dtype while keeping the index
    /// structure (`coords`) and `shape` unchanged. scipy selects the cast from a
    /// runtime numpy dtype object; Rust has no runtime dtype, so this is a
    /// **deviation** (R-DEV-4): the caller supplies the element cast as a closure.
    /// A plain `as`-cast closure (e.g. `|&v| v as i64`) reproduces numpy's C-cast
    /// semantics, including float→int **truncation toward zero**.
    ///
    /// The row/column index arrays, the `(n_rows, n_cols)` shape, and the stored
    /// count are copied verbatim — only the data array changes type — so duplicate
    /// `(row, col)` entries and explicit stored zeros are preserved (no coalescing).
    ///
    /// Live oracle (R-CHAR-3): for
    /// `sp.coo_matrix(([3.7,-2.9,5.0],([0,1,2],[0,1,2])),shape=(3,3))`,
    /// `astype(np.int64)` yields `data == [3, -2, 5]` (truncated toward zero) with
    /// `row == [0,1,2]`, `col == [0,1,2]`, `nnz == 3`; `astype(np.float32)` yields
    /// `data == [3.7f32, -2.9f32, 5.0f32]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] propagated from [`from_triplets`](Self::from_triplets);
    /// infallible for any structurally valid `CooMatrix` (the unchanged indices stay
    /// in bounds of the unchanged shape).
    pub fn astype<U, Fc>(&self, cast: Fc) -> Result<CooMatrix<U>, FerroError>
    where
        Fc: Fn(&T) -> U,
    {
        let data: Vec<U> = self.data().iter().map(&cast).collect();
        CooMatrix::<U>::from_triplets(
            self.n_rows(),
            self.n_cols(),
            self.row().to_vec(),
            self.col().to_vec(),
            data,
        )
    }
}

impl<T: Clone> CooMatrix<T> {
    /// Return a copy of this matrix, preserving **all** stored entries.
    ///
    /// Mirrors scipy `coo_matrix.copy()` (`scipy/sparse/_data.py:94` —
    /// `return self._with_data(self.data.copy(), copy=True)`), which returns an
    /// identical matrix with the same sparsity pattern and data array. Every
    /// stored triplet is preserved verbatim — including **explicit stored zeros**
    /// and **duplicate `(row, col)` entries** — without coalescing or reordering.
    /// Equivalent to [`Clone::clone`]; provided as a named method for scipy parity.
    ///
    /// Live oracle (R-CHAR-3): for
    /// `sp.coo_matrix(([3.,0.,5.],([0,1,2],[0,1,2])),shape=(3,3))` (an explicit
    /// stored `0` at `(1,1)`), `m.copy().nnz == 3` and `m.copy().data == [3,0,5]`.
    #[must_use]
    pub fn copy(&self) -> CooMatrix<T> {
        self.clone()
    }
}

impl<T> CooMatrix<T>
where
    T: Clone + Zero + PartialEq,
{
    /// Remove stored entries equal to zero, returning a **new** matrix.
    ///
    /// Mirrors scipy `coo_matrix.eliminate_zeros()`
    /// (`scipy/sparse/_coo.py:798` — `mask = self.data != 0; self.data =
    /// self.data[mask]; self.coords = tuple(idx[mask] for idx in self.coords)`),
    /// which drops every explicitly stored zero. scipy mutates in place; here we
    /// return a new [`CooMatrix`] (functional style, consistent with the other
    /// COO methods that return new matrices), keeping only the triplets whose
    /// `data[k] != T::zero()`, in their original order.
    ///
    /// Live oracle (R-CHAR-3): for
    /// `sp.coo_matrix(([3.,0.,5.],([0,1,2],[0,1,2])),shape=(3,3))`,
    /// after `eliminate_zeros()` the matrix has `nnz == 2`, `row == [0, 2]`,
    /// `col == [0, 2]`, `data == [3, 5]`, and an unchanged dense form
    /// `[[3,0,0],[0,0,0],[0,0,5]]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] propagated from [`from_triplets`](Self::from_triplets);
    /// infallible for any structurally valid `CooMatrix` (filtering keeps every
    /// index in bounds of the unchanged shape).
    pub fn eliminate_zeros(&self) -> Result<CooMatrix<T>, FerroError> {
        let zero = T::zero();
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut data = Vec::new();
        for ((&r, &c), val) in self.row().iter().zip(self.col()).zip(self.data()) {
            if *val != zero {
                row.push(r);
                col.push(c);
                data.push(val.clone());
            }
        }
        Self::from_triplets(self.n_rows(), self.n_cols(), row, col, data)
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

impl<T> CooMatrix<T>
where
    T: Clone + Add<Output = T> + 'static,
{
    /// Convert this COO matrix to [`CsrMatrix`] (Compressed Sparse Row) format.
    ///
    /// Mirrors scipy `coo_matrix.tocsr()` (`scipy/sparse/_coo.py:349`), which
    /// builds the CSR `(data, indices, indptr)` triple and **sums duplicate**
    /// `(row, col)` entries (`tocsr` calls `sum_duplicates` for a non-canonical
    /// matrix). Delegates to [`CsrMatrix::from_coo`], which performs the same
    /// duplicate-summing conversion via the sprs `TriMat::to_csr`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] only if the conversion fails; it is infallible for
    /// any structurally valid `CooMatrix`.
    pub fn to_csr(&self) -> Result<CsrMatrix<T>, FerroError> {
        CsrMatrix::from_coo(self)
    }

    /// Convert this COO matrix to [`CscMatrix`] (Compressed Sparse Column) format.
    ///
    /// Mirrors scipy `coo_matrix.tocsc()` (`scipy/sparse/_coo.py:316`), which
    /// builds the CSC `(data, indices, indptr)` triple and **sums duplicate**
    /// `(row, col)` entries (`tocsc` calls `sum_duplicates` for a non-canonical
    /// matrix). Delegates to [`CscMatrix::from_coo`], which performs the same
    /// duplicate-summing conversion via the sprs `TriMat::to_csc`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] only if the conversion fails; it is infallible for
    /// any structurally valid `CooMatrix`.
    pub fn to_csc(&self) -> Result<CscMatrix<T>, FerroError> {
        CscMatrix::from_coo(self)
    }

    /// Transpose: returns a new `(n_cols, n_rows)` COO matrix whose dense form is
    /// `Aᵀ`.
    ///
    /// Mirrors scipy `coo_matrix.transpose()` / `.T` (`scipy/sparse/_coo.py:229`),
    /// which for a 2-D matrix permutes the coordinate arrays and shape with
    /// `axes=(1, 0)` (`permuted_coords = tuple(self.coords[i] for i in axes)`,
    /// `permuted_shape = tuple(self._shape[i] for i in axes)`) — i.e. it swaps the
    /// row and column index arrays and the `(M, N)` shape, keeping `data`
    /// unchanged. Here the new row indices are the old column indices, the new
    /// column indices are the old row indices, and the shape becomes
    /// `(n_cols, n_rows)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] propagated from [`from_triplets`](Self::from_triplets);
    /// infallible for any structurally valid `CooMatrix` (the swapped indices stay
    /// in bounds of the swapped shape).
    pub fn transpose(&self) -> Result<CooMatrix<T>, FerroError> {
        Self::from_triplets(
            self.n_cols(),
            self.n_rows(),
            self.col().to_vec(),
            self.row().to_vec(),
            self.data().to_vec(),
        )
    }
}

impl<T> CooMatrix<T>
where
    T: Copy + Zero + Add<Output = T>,
{
    /// Sum of all stored values.
    ///
    /// Mirrors scipy `coo_matrix.sum()` with `axis=None`
    /// (`scipy/sparse/_coo.py:1429`, `_sum_nd`), which reduces over both rows and
    /// columns to a scalar. Computed directly from the stored triplets, so
    /// **duplicate `(row, col)` entries are summed** (matching scipy, which sums
    /// duplicates in `.sum()`). The running total is accumulated from
    /// [`T::zero()`].
    #[must_use]
    pub fn sum(&self) -> T {
        let mut acc = T::zero();
        for &val in self.data() {
            acc = acc + val;
        }
        acc
    }

    /// Column sums, a length-`n_cols` vector.
    ///
    /// Mirrors scipy `coo_matrix.sum(axis=0)` (`scipy/sparse/_coo.py:1429`,
    /// `_sum_nd`), which returns a `(1, n_cols)` row vector of per-column sums;
    /// here it is returned as a length-`n_cols` [`Array1`]. For each stored
    /// triplet `(row, col, val)`, `val` is added to `out[col]`, so **duplicate
    /// entries are summed**.
    #[must_use]
    pub fn sum_axis0(&self) -> Array1<T> {
        let mut out = Array1::<T>::zeros(self.n_cols());
        for (&c, &val) in self.col().iter().zip(self.data()) {
            out[c] = out[c] + val;
        }
        out
    }

    /// Row sums, a length-`n_rows` vector.
    ///
    /// Mirrors scipy `coo_matrix.sum(axis=1)` (`scipy/sparse/_coo.py:1429`,
    /// `_sum_nd`), which returns an `(n_rows, 1)` column vector of per-row sums;
    /// here it is returned as a length-`n_rows` [`Array1`]. For each stored
    /// triplet `(row, col, val)`, `val` is added to `out[row]`, so **duplicate
    /// entries are summed**.
    #[must_use]
    pub fn sum_axis1(&self) -> Array1<T> {
        let mut out = Array1::<T>::zeros(self.n_rows());
        for (&r, &val) in self.row().iter().zip(self.data()) {
            out[r] = out[r] + val;
        }
        out
    }

    /// Collapse duplicate `(row, col)` entries by **summing** their values,
    /// returning a **new** matrix in canonical `(row, col)`-sorted order.
    ///
    /// Mirrors scipy `coo_matrix.sum_duplicates()` (`scipy/sparse/_coo.py:768`,
    /// delegating to `_sum_duplicates`, `:779`), which canonicalizes the matrix:
    /// it lexsorts the coordinates (`order = np.lexsort(coords[::-1])`, `:786`) so
    /// the entries end up in `(row, col)` order, then adds together the values of
    /// every group of identical coordinates (`np.add.reduceat`, `:795`). scipy
    /// mutates in place; here we return a new [`CooMatrix`] (functional style,
    /// consistent with the other COO methods that return new matrices).
    ///
    /// Crucially, this **only canonicalizes** — it does NOT drop entries whose
    /// summed value is zero. Removing zeros is the separate job of
    /// [`eliminate_zeros`](Self::eliminate_zeros) (scipy's `sum_duplicates` and
    /// `eliminate_zeros` are distinct methods). A group of duplicates that cancel
    /// to zero is preserved as a single zero-valued stored entry.
    ///
    /// A [`BTreeMap`] keyed on `(row, col)` accumulates the values (adding into
    /// the entry on collision) and yields keys in sorted `(row, col)` order,
    /// matching scipy's canonical form.
    ///
    /// Live oracle (R-CHAR-3): for
    /// `sp.coo_matrix(([3.,5.,2.,1.],([0,0,2,2],[0,0,2,2])),shape=(3,3))`,
    /// `sum_duplicates()` yields `nnz == 2`, `row == [0, 2]`, `col == [0, 2]`,
    /// `data == [8, 3]` (`(0,0)=3+5`, `(2,2)=2+1`); for
    /// `sp.coo_matrix(([4.,-4.,7.],([0,0,1],[0,0,1])),shape=(2,2))`, it yields
    /// `nnz == 2`, `row == [0, 1]`, `col == [0, 1]`, `data == [0, 7]` — the
    /// zero-sum `(0,0)` entry is **preserved**.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] propagated from [`from_triplets`](Self::from_triplets);
    /// infallible for any structurally valid `CooMatrix` (the accumulated indices
    /// stay in bounds of the unchanged shape).
    pub fn sum_duplicates(&self) -> Result<CooMatrix<T>, FerroError> {
        let mut acc: BTreeMap<(usize, usize), T> = BTreeMap::new();
        for ((&r, &c), &val) in self.row().iter().zip(self.col()).zip(self.data()) {
            let entry = acc.entry((r, c)).or_insert_with(T::zero);
            *entry = *entry + val;
        }
        let mut row = Vec::with_capacity(acc.len());
        let mut col = Vec::with_capacity(acc.len());
        let mut data = Vec::with_capacity(acc.len());
        for ((r, c), val) in acc {
            row.push(r);
            col.push(c);
            data.push(val);
        }
        Self::from_triplets(self.n_rows(), self.n_cols(), row, col, data)
    }

    /// Main diagonal, a length-`min(n_rows, n_cols)` vector.
    ///
    /// Mirrors scipy `coo_matrix.diagonal()` with `k=0`
    /// (`scipy/sparse/_coo.py:458`): `out[i] == A[i, i]` for
    /// `i in 0..min(n_rows, n_cols)`. Computed directly from the stored triplets:
    /// for each `(row, col, val)` with `row == col`, `val` is accumulated into
    /// `out[row]`, so **duplicate diagonal entries are summed** (scipy's
    /// `diagonal` calls `_sum_duplicates` on the masked entries). Positions with
    /// no stored entry default to [`T::zero()`].
    #[must_use]
    pub fn diagonal(&self) -> Array1<T> {
        let len = self.n_rows().min(self.n_cols());
        let mut out = Array1::<T>::zeros(len);
        for ((&r, &c), &val) in self.row().iter().zip(self.col()).zip(self.data()) {
            if r == c && r < len {
                out[r] = out[r] + val;
            }
        }
        out
    }
}

impl<T> CooMatrix<T>
where
    T: Copy + Zero + PartialOrd,
{
    /// Maximum over all elements (scipy `axis=None`), folding in implicit zeros.
    ///
    /// Mirrors scipy `coo_matrix.max()` via the `_minmax_mixin._min_or_max`
    /// machinery with `axis=None` (`scipy/sparse/_data.py:208`-`:224`). scipy
    /// reduces over the stored data (`m = min_or_max.reduce(self._deduped_data())`,
    /// `:221`) and then, when the matrix is **not fully dense**, folds an implicit
    /// zero into the result (`if self.nnz != math.prod(self.shape): m =
    /// min_or_max(zero, m)`, `:222`-`:223`). An empty matrix (`nnz == 0`) returns
    /// `zero` (`:219`-`:220`). Here `nnz` is the number of stored triplets
    /// ([`data().len()`](Self::data)); the matrix is fully dense iff
    /// `nnz == n_rows * n_cols`, in which case no implicit zero exists to fold.
    ///
    /// Live oracle (R-CHAR-3): `coo_matrix(([-3.,-1.,-5.],([0,1,2],[0,1,2])),
    /// shape=(3,3)).max() == 0.0` (implicit zero wins over the all-negative
    /// stored data); `coo_matrix(([2.,7.],([0,0],[0,1])),shape=(1,2)).max() ==
    /// 7.0` (fully dense, no implicit zero).
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
    /// Mirrors scipy `coo_matrix.min()` via the `_minmax_mixin._min_or_max`
    /// machinery with `axis=None` (`scipy/sparse/_data.py:208`-`:224`). scipy
    /// reduces over the stored data (`m = min_or_max.reduce(self._deduped_data())`,
    /// `:221`) and then, when the matrix is **not fully dense**, folds an implicit
    /// zero into the result (`if self.nnz != math.prod(self.shape): m =
    /// min_or_max(zero, m)`, `:222`-`:223`). An empty matrix (`nnz == 0`) returns
    /// `zero` (`:219`-`:220`). Here `nnz` is the number of stored triplets
    /// ([`data().len()`](Self::data)); the matrix is fully dense iff
    /// `nnz == n_rows * n_cols`, in which case no implicit zero exists to fold.
    ///
    /// Live oracle (R-CHAR-3): `coo_matrix(([3.,1.,5.],([0,1,2],[0,1,2])),
    /// shape=(3,3)).min() == 0.0` (implicit zero wins over the all-positive
    /// stored data); `coo_matrix(([2.,7.],([0,0],[0,1])),shape=(1,2)).min() ==
    /// 2.0` (fully dense, no implicit zero).
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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

    // REQ-MISSING-METHODS (conversion + transpose) — live scipy oracle (R-CHAR-3).
    // Expected values from `cd /tmp && python3 -c "import numpy as np,
    //   scipy.sparse as sp;
    //   m=sp.coo_matrix((np.array([3.,5.,2.]),(np.array([0,2,1]),
    //     np.array([0,1,2]))),shape=(3,3));
    //   print(m.tocsr().toarray().tolist(), m.tocsc().toarray().tolist(),
    //         m.transpose().toarray().tolist())"`
    //   -> [[3,0,0],[0,0,2],[0,5,0]] (csr) [[3,0,0],[0,0,2],[0,5,0]] (csc)
    //      [[3,0,0],[0,0,5],[0,2,0]] (transpose)
    // and non-square `sp.coo_matrix((np.array([7.,9.]),(np.array([0,1]),
    //   np.array([2,0]))),shape=(2,3)).transpose().toarray().tolist()`
    //   -> [[0,9],[0,0],[7,0]] (shape (3,2)).

    fn sample_coo() -> Result<CooMatrix<f64>, FerroError> {
        // [[3,0,0],[0,0,2],[0,5,0]]
        CooMatrix::from_triplets(3, 3, vec![0, 2, 1], vec![0, 1, 2], vec![3.0, 5.0, 2.0])
    }

    #[test]
    fn coo_to_csr_matches_scipy() -> Result<(), FerroError> {
        let m = sample_coo()?;
        let csr = m.to_csr()?;
        assert_eq!(
            csr.to_dense(),
            array![[3.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 5.0, 0.0]]
        );
        Ok(())
    }

    #[test]
    fn coo_to_csc_matches_scipy() -> Result<(), FerroError> {
        let m = sample_coo()?;
        let csc = m.to_csc()?;
        assert_eq!(
            csc.to_dense(),
            array![[3.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 5.0, 0.0]]
        );
        Ok(())
    }

    #[test]
    fn coo_transpose_matches_scipy() -> Result<(), FerroError> {
        let m = sample_coo()?;
        let t = m.transpose()?;
        assert_eq!(t.n_rows(), 3);
        assert_eq!(t.n_cols(), 3);
        assert_eq!(
            t.to_dense(),
            array![[3.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 2.0, 0.0]]
        );
        Ok(())
    }

    #[test]
    fn coo_transpose_non_square() -> Result<(), FerroError> {
        // [[0,0,7],[9,0,0]] (shape (2,3)); transpose -> [[0,9],[0,0],[7,0]] (3,2)
        let m = CooMatrix::from_triplets(2, 3, vec![0, 1], vec![2, 0], vec![7.0, 9.0])?;
        let t = m.transpose()?;
        assert_eq!(t.n_rows(), 3);
        assert_eq!(t.n_cols(), 2);
        assert_eq!(t.to_dense(), array![[0.0, 9.0], [0.0, 0.0], [7.0, 0.0]]);
        Ok(())
    }

    // REQ-MISSING-METHODS (reductions) — live scipy oracle (R-CHAR-3).
    // Expected values from `cd /tmp && python3 -c "import numpy as np,
    //   scipy.sparse as sp;
    //   m=sp.coo_matrix((np.array([3.,5.,2.]),(np.array([0,2,1]),
    //     np.array([0,1,2]))),shape=(3,3));
    //   print(m.sum(), m.sum(axis=0).tolist(), m.sum(axis=1).tolist(),
    //         m.diagonal().tolist())"`
    //   -> 10.0  [[3,5,2]] (axis=0 col sums)  [[3],[2],[5]] (axis=1 row sums)
    //      [3.0, 0.0, 0.0] (diagonal).

    #[test]
    fn coo_sum_matches_scipy() -> Result<(), FerroError> {
        let m = sample_coo()?;
        assert_eq!(m.sum(), 10.0);
        Ok(())
    }

    #[test]
    fn coo_sum_axis0_matches_scipy() -> Result<(), FerroError> {
        let m = sample_coo()?;
        assert_eq!(m.sum_axis0(), array![3.0, 5.0, 2.0]);
        Ok(())
    }

    #[test]
    fn coo_sum_axis1_matches_scipy() -> Result<(), FerroError> {
        let m = sample_coo()?;
        assert_eq!(m.sum_axis1(), array![3.0, 2.0, 5.0]);
        Ok(())
    }

    #[test]
    fn coo_diagonal_matches_scipy() -> Result<(), FerroError> {
        let m = sample_coo()?;
        assert_eq!(m.diagonal(), array![3.0, 0.0, 0.0]);
        Ok(())
    }

    // Duplicate-summing reductions — two entries at (0,0). Oracle:
    // `sp.coo_matrix((np.array([3.,5.]),(np.array([0,0]),np.array([0,0]))),
    //   shape=(2,2))` -> sum()==8.0, diagonal()==[8.0, 0.0].
    #[test]
    fn coo_sum_diagonal_sum_duplicates() -> Result<(), FerroError> {
        let m = CooMatrix::from_triplets(2, 2, vec![0, 0], vec![0, 0], vec![3.0, 5.0])?;
        assert_eq!(m.sum(), 8.0);
        assert_eq!(m.diagonal(), array![8.0, 0.0]);
        Ok(())
    }
}
