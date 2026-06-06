//! Divergence / conformance tests for `ferrolearn-sparse/src/coo.rs`
//! (`CooMatrix`, the `scipy.sparse.coo_matrix` analog) vs the LIVE scipy 1.17.1
//! oracle. Crosslink translation unit #1995.
//!
//! All expected values are computed by a live scipy call run from `/tmp`
//! (R-CHAR-3 — NEVER copied from the ferrolearn side). The oracle command and
//! its output are quoted in each test's doc-comment so the target is unambiguous.
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np, scipy.sparse as sp
//! m=sp.coo_matrix((np.array([1.,2.,5.]),(np.array([0,0,1]),np.array([0,0,2]))),shape=(2,3))
//! print('nnz', m.nnz, 'shape', m.shape, 'toarray', m.toarray().tolist())
//! m2=sp.coo_matrix((np.array([3.,4.,1.,2.]),(np.array([0,1,2,1]),np.array([1,0,2,0]))),shape=(3,3))
//! print('m2 nnz', m2.nnz, 'toarray', m2.toarray().tolist())
//! print('empty nnz', sp.coo_matrix((2,2)).nnz, sp.coo_matrix((2,2)).toarray().tolist())"
//! ```
//! prints (scipy 1.17.1):
//! ```text
//! nnz 3 shape (2, 3) toarray [[3.0, 0.0, 0.0], [0.0, 0.0, 5.0]]
//! m2 nnz 4 toarray [[0.0, 3.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
//! empty nnz 0 [[0.0, 0.0], [0.0, 0.0]]
//! ```
//!
//! These are GREEN guards: the SHIPPED construction / to_dense (duplicate-summing)
//! / nnz / empty / shape / error core matches the oracle and these PASS now,
//! guarding against regression. No RED pin is present: the duplicate-summing and
//! nnz semantics match scipy exactly, so there is no genuine single-file-fixable
//! divergence in the SHIPPED behavior. The NOT-STARTED REQs (API accessors,
//! missing methods, ferray substrate) are structural blockers filed as `-l blocker`
//! issues, not pinned here as doomed tests (R-DEFER-3).

use ferrolearn_core::FerroError;
use ferrolearn_sparse::{CooMatrix, CsrMatrix};

/// REQ-CONSTRUCT + REQ-API geometry. The m2-style matrix (no duplicate) round-trips
/// triplet -> dense matching scipy `.toarray()`.
///
/// Oracle: `coo_matrix((data,(row,col)),shape=(3,3))` with a duplicate at (1,0)
/// `.toarray() == [[0.0, 3.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 1.0]]`; `.shape == (3, 3)`.
/// data=[3,4,1,2] row=[0,1,2,1] col=[1,0,2,0]; duplicate (1,0)=4+2=6.
#[test]
fn coo_construct_to_dense_matches_scipy_toarray() {
    let m = CooMatrix::<f64>::from_triplets(
        3,
        3,
        vec![0, 1, 2, 1],
        vec![1, 0, 2, 0],
        vec![3.0, 4.0, 1.0, 2.0],
    )
    .unwrap();

    // scipy .shape == (3, 3)
    assert_eq!(m.n_rows(), 3);
    assert_eq!(m.n_cols(), 3);

    let d = m.to_dense();
    // scipy .toarray() == [[0,3,0],[6,0,0],[0,0,1]]
    let expected = [[0.0, 3.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    for r in 0..3 {
        for c in 0..3 {
            assert_eq!(d[[r, c]], expected[r][c], "mismatch at ({r},{c})");
        }
    }
}

/// REQ-TOARRAY-DUP. The canonical duplicate matrix: a duplicate at (0,0) sums in
/// `to_dense`, matching scipy `.toarray()`.
///
/// Oracle: `coo_matrix(([1.,2.,5.],([0,0,1],[0,0,2])),shape=(2,3)).toarray()`
/// `== [[3.0, 0.0, 0.0], [0.0, 0.0, 5.0]]`; duplicate (0,0)=1+2=3.
#[test]
fn coo_to_dense_duplicate_summed_matches_scipy() {
    let m =
        CooMatrix::<f64>::from_triplets(2, 3, vec![0, 0, 1], vec![0, 0, 2], vec![1.0, 2.0, 5.0])
            .unwrap();

    let d = m.to_dense();
    // scipy .toarray() == [[3,0,0],[0,0,5]]
    let expected = [[3.0, 0.0, 0.0], [0.0, 0.0, 5.0]];
    for r in 0..2 {
        for c in 0..3 {
            assert_eq!(d[[r, c]], expected[r][c], "mismatch at ({r},{c})");
        }
    }
    // The summed cell specifically.
    assert_eq!(d[[0, 0]], 3.0);
}

/// REQ-NNZ. `nnz()` counts STORED entries including duplicates, matching scipy
/// `coo_matrix(...).nnz` (= `len(self.data)`, `_getnnz(axis=None)`), and
/// contrasts with CSR coalescing.
///
/// Oracle: for the duplicate matrix, `m.nnz == 3` (stored, duplicate counted),
/// while `m.tocsr().nnz == 2` (CSR coalesces duplicate (0,0)).
#[test]
fn coo_nnz_counts_duplicates_csr_coalesces() {
    let m =
        CooMatrix::<f64>::from_triplets(2, 3, vec![0, 0, 1], vec![0, 0, 2], vec![1.0, 2.0, 5.0])
            .unwrap();

    // scipy m.nnz == 3
    assert_eq!(m.nnz(), 3);

    // scipy m.tocsr().nnz == 2
    let csr = CsrMatrix::from_coo(&m).unwrap();
    assert_eq!(csr.nnz(), 2);
}

/// REQ-NNZ via incremental push: two pushes at the same (0,0) leave nnz==2
/// (stored, duplicate counted), mirroring scipy storing both before coalescing.
///
/// Oracle: `coo_matrix(([1.,2.],([0,0],[0,0])),shape=(2,2)).nnz == 2`.
#[test]
fn coo_push_duplicate_nnz_counts_both() {
    let mut m: CooMatrix<f64> = CooMatrix::new(2, 2);
    m.push(0, 0, 1.0).unwrap();
    m.push(0, 0, 2.0).unwrap();
    // scipy stores both: nnz == 2
    assert_eq!(m.nnz(), 2);
    // toarray sums them -> 3.0
    assert_eq!(m.to_dense()[[0, 0]], 3.0);
}

/// REQ-CONSTRUCT empty. `CooMatrix::new(2,2)` mirrors `scipy.sparse.coo_matrix((2,2))`.
///
/// Oracle: `sp.coo_matrix((2,2)).nnz == 0`; `.toarray() == [[0,0],[0,0]]`;
/// `.shape == (2, 2)`.
#[test]
fn coo_empty_matches_scipy() {
    let m: CooMatrix<f64> = CooMatrix::new(2, 2);
    // scipy .nnz == 0
    assert_eq!(m.nnz(), 0);
    // scipy .shape == (2, 2)
    assert_eq!(m.n_rows(), 2);
    assert_eq!(m.n_cols(), 2);
    // scipy .toarray() == [[0,0],[0,0]]
    let d = m.to_dense();
    for r in 0..2 {
        for c in 0..2 {
            assert_eq!(d[[r, c]], 0.0, "expected zero at ({r},{c})");
        }
    }
}

/// REQ-API geometry / shape. `n_rows()`/`n_cols()` mirror scipy `.shape` tuple
/// elements for the canonical matrix.
///
/// Oracle: `coo_matrix(([1.,2.,5.],([0,0,1],[0,0,2])),shape=(2,3)).shape == (2, 3)`.
#[test]
fn coo_shape_matches_scipy() {
    let m =
        CooMatrix::<f64>::from_triplets(2, 3, vec![0, 0, 1], vec![0, 0, 2], vec![1.0, 2.0, 5.0])
            .unwrap();
    // scipy .shape == (2, 3)
    assert_eq!((m.n_rows(), m.n_cols()), (2, 3));
}

/// REQ-ERR. Out-of-bounds row index is rejected at construction, where scipy
/// raises `ValueError: axis 0 index 5 exceeds matrix dimension 2`.
///
/// Oracle: `sp.coo_matrix(([1.],([5],[0])),shape=(2,3))` ->
/// `ValueError: axis 0 index 5 exceeds matrix dimension 2`.
/// ferrolearn returns `Err(FerroError::InvalidParameter)` (crate contract;
/// `ValueError` marshalling is ferrolearn-python's job).
#[test]
fn coo_from_triplets_out_of_bounds_row_is_err() {
    let r = CooMatrix::<f64>::from_triplets(2, 3, vec![5], vec![0], vec![1.0]);
    assert!(
        r.is_err(),
        "out-of-bounds row must be rejected at construction"
    );
}

/// REQ-ERR. Out-of-bounds column index is rejected at construction, where scipy
/// raises `ValueError: axis 1 index ... exceeds matrix dimension ...`.
///
/// Oracle: `sp.coo_matrix(([1.],([0],[9])),shape=(2,3))` ->
/// `ValueError: axis 1 index 9 exceeds matrix dimension 3`.
#[test]
fn coo_from_triplets_out_of_bounds_col_is_err() {
    let r = CooMatrix::<f64>::from_triplets(2, 3, vec![0], vec![9], vec![1.0]);
    assert!(
        r.is_err(),
        "out-of-bounds col must be rejected at construction"
    );
}

/// REQ-ERR. Mismatched array lengths rejected at construction, where scipy raises
/// `ValueError: all index and data arrays must have the same length`.
///
/// Oracle: `sp.coo_matrix(([1.,2.],([0],[0])),shape=(2,3))` ->
/// `ValueError: all index and data arrays must have the same length`.
#[test]
fn coo_from_triplets_mismatched_lengths_is_err() {
    let r = CooMatrix::<f64>::from_triplets(2, 3, vec![0], vec![0], vec![1.0, 2.0]);
    assert!(r.is_err(), "mismatched triplet lengths must be rejected");
}

/// REQ-ERR. `push` validates bounds, where scipy validates at construction.
#[test]
fn coo_push_out_of_bounds_is_err() {
    let mut m: CooMatrix<f64> = CooMatrix::new(2, 2);
    assert!(m.push(2, 0, 1.0).is_err(), "row out of bounds must err");
    assert!(m.push(0, 2, 1.0).is_err(), "col out of bounds must err");
}

/// REQ-API-ACCESSORS. The first-class `shape()`/`data()`/`row()`/`col()`
/// accessors expose the same geometry + COO `(data, (row, col))` triple scipy
/// exposes as `.shape`/`.data`/`.row`/`.col`, in insertion order (COO does not
/// coalesce or reorder on construction).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   m=sp.coo_matrix((np.array([3.,5.,2.]),(np.array([0,2,1]),np.array([0,1,2]))),
///     shape=(3,3));
///   print(m.shape, m.data.tolist(), m.row.tolist(), m.col.tolist())"`):
/// `(3, 3) [3.0, 5.0, 2.0] [0, 2, 1] [0, 1, 2]`.
#[test]
fn coo_shape_data_row_col_match_scipy() {
    let m =
        CooMatrix::<f64>::from_triplets(3, 3, vec![0, 2, 1], vec![0, 1, 2], vec![3.0, 5.0, 2.0])
            .unwrap();
    // scipy m.shape == (3, 3)
    assert_eq!(m.shape(), (3, 3));
    // scipy m.data == [3,5,2] (insertion order, no coalescing)
    assert_eq!(m.data(), &[3.0, 5.0, 2.0]);
    // scipy m.row == [0,2,1] (row coordinate of each stored triplet)
    assert_eq!(m.row(), &[0, 2, 1]);
    // scipy m.col == [0,1,2] (column coordinate of each stored triplet)
    assert_eq!(m.col(), &[0, 1, 2]);
}

/// REQ-MISSING-METHODS (`copy`). `copy()` returns an identical matrix preserving
/// ALL stored entries, including an **explicit stored zero**, mirroring scipy
/// `coo_matrix.copy()` (`scipy/sparse/_data.py:94`). The original is unchanged.
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   m=sp.coo_matrix((np.array([3.,0.,5.]),(np.array([0,1,2]),np.array([0,1,2]))),
///     shape=(3,3));
///   c=m.copy();
///   print(c.nnz, c.data.tolist(), c.toarray().tolist(), m.toarray().tolist())"`):
/// `3 [3.0, 0.0, 5.0] [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]
///   [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]`.
#[test]
fn coo_copy_preserves_stored_zero() -> Result<(), FerroError> {
    let m = CooMatrix::<f64>::from_triplets(
        3,
        3,
        vec![0, 1, 2],
        vec![0, 1, 2],
        vec![3.0, 0.0, 5.0], // explicit stored 0 at (1,1)
    )?;

    let c = m.copy();
    // scipy c.nnz == 3 (explicit stored zero counted)
    assert_eq!(c.nnz(), 3);
    // scipy c.data == [3, 0, 5]
    assert_eq!(c.data(), &[3.0, 0.0, 5.0]);

    // scipy c.toarray() == [[3,0,0],[0,0,0],[0,0,5]]
    let expected = [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]];
    let d = c.to_dense();
    for r in 0..3 {
        for col in 0..3 {
            assert_eq!(
                d[[r, col]],
                expected[r][col],
                "copy mismatch at ({r},{col})"
            );
        }
    }

    // The original is unchanged.
    assert_eq!(m.nnz(), 3);
    assert_eq!(m.data(), &[3.0, 0.0, 5.0]);
    let dm = m.to_dense();
    for r in 0..3 {
        for col in 0..3 {
            assert_eq!(
                dm[[r, col]],
                expected[r][col],
                "original mutated at ({r},{col})"
            );
        }
    }
    Ok(())
}

/// REQ-MISSING-METHODS (`eliminate_zeros`). `eliminate_zeros()` drops the
/// explicitly stored zero, mirroring scipy `coo_matrix.eliminate_zeros()`
/// (`scipy/sparse/_coo.py:798`). The dense form is unchanged (a stored zero and
/// an implicit zero materialize identically).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   m=sp.coo_matrix((np.array([3.,0.,5.]),(np.array([0,1,2]),np.array([0,1,2]))),
///     shape=(3,3));
///   m.eliminate_zeros();
///   print(m.nnz, m.row.tolist(), m.col.tolist(), m.data.tolist(),
///         m.toarray().tolist())"`):
/// `2 [0, 2] [0, 2] [3.0, 5.0] [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]`.
#[test]
fn coo_eliminate_zeros_matches_scipy() -> Result<(), FerroError> {
    let m = CooMatrix::<f64>::from_triplets(
        3,
        3,
        vec![0, 1, 2],
        vec![0, 1, 2],
        vec![3.0, 0.0, 5.0], // explicit stored 0 at (1,1)
    )?;

    let e = m.eliminate_zeros()?;
    // scipy nnz == 2 (the stored zero removed)
    assert_eq!(e.nnz(), 2);
    // scipy row == [0, 2], col == [0, 2], data == [3, 5]
    assert_eq!(e.row(), &[0, 2]);
    assert_eq!(e.col(), &[0, 2]);
    assert_eq!(e.data(), &[3.0, 5.0]);

    // scipy toarray() unchanged == [[3,0,0],[0,0,0],[0,0,5]]
    let expected = [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]];
    let d = e.to_dense();
    for r in 0..3 {
        for col in 0..3 {
            assert_eq!(
                d[[r, col]],
                expected[r][col],
                "eliminate_zeros mismatch at ({r},{col})"
            );
        }
    }
    Ok(())
}

/// REQ-MISSING-METHODS (`sum_duplicates`). `sum_duplicates()` collapses duplicate
/// `(row, col)` entries by SUMMING their values, in canonical `(row, col)`-sorted
/// order, mirroring scipy `coo_matrix.sum_duplicates()` (`scipy/sparse/_coo.py:768`).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   A=sp.coo_matrix((np.array([3.,5.,2.,1.]),(np.array([0,0,2,2]),np.array([0,0,2,2]))),
///     shape=(3,3));
///   A.sum_duplicates();
///   print(A.nnz, A.row.tolist(), A.col.tolist(), A.data.tolist(),
///         A.toarray().tolist())"`):
/// `2 [0, 2] [0, 2] [8.0, 3.0] [[8.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]`.
#[test]
fn coo_sum_duplicates_matches_scipy() -> Result<(), FerroError> {
    let a = CooMatrix::<f64>::from_triplets(
        3,
        3,
        vec![0, 0, 2, 2],
        vec![0, 0, 2, 2],
        vec![3.0, 5.0, 2.0, 1.0],
    )?;

    let s = a.sum_duplicates()?;
    // scipy nnz == 2 (the two duplicate groups collapsed)
    assert_eq!(s.nnz(), 2);
    // scipy row == [0, 2], col == [0, 2], data == [8, 3]
    assert_eq!(s.row(), &[0, 2]);
    assert_eq!(s.col(), &[0, 2]);
    assert_eq!(s.data(), &[8.0, 3.0]);

    // scipy toarray() == [[8,0,0],[0,0,0],[0,0,3]]
    let expected = [[8.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]];
    let d = s.to_dense();
    for r in 0..3 {
        for col in 0..3 {
            assert_eq!(
                d[[r, col]],
                expected[r][col],
                "sum_duplicates mismatch at ({r},{col})"
            );
        }
    }
    Ok(())
}

/// REQ-MISSING-METHODS (`sum_duplicates`). A group of duplicates that cancel to
/// zero is PRESERVED as a single zero-valued stored entry — `sum_duplicates()`
/// only canonicalizes; removing zeros is `eliminate_zeros`'s separate job
/// (scipy `coo_matrix.sum_duplicates`, `scipy/sparse/_coo.py:768`, does not
/// eliminate zeros).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   B=sp.coo_matrix((np.array([4.,-4.,7.]),(np.array([0,0,1]),np.array([0,0,1]))),
///     shape=(2,2));
///   B.sum_duplicates();
///   print(B.nnz, B.row.tolist(), B.col.tolist(), B.data.tolist(),
///         B.toarray().tolist())"`):
/// `2 [0, 1] [0, 1] [0.0, 7.0] [[0.0, 0.0], [0.0, 7.0]]`.
#[test]
fn coo_sum_duplicates_preserves_zero_sum() -> Result<(), FerroError> {
    let b = CooMatrix::<f64>::from_triplets(
        2,
        2,
        vec![0, 0, 1],
        vec![0, 0, 1],
        vec![4.0, -4.0, 7.0], // (0,0)=4+(-4)=0 — zero-sum, must be preserved
    )?;

    let s = b.sum_duplicates()?;
    // scipy nnz == 2 (the zero-sum (0,0) entry is preserved, NOT dropped)
    assert_eq!(s.nnz(), 2);
    // scipy row == [0, 1], col == [0, 1], data == [0, 7]
    assert_eq!(s.row(), &[0, 1]);
    assert_eq!(s.col(), &[0, 1]);
    assert_eq!(s.data(), &[0.0, 7.0]);

    // scipy toarray() == [[0,0],[0,7]]
    let expected = [[0.0, 0.0], [0.0, 7.0]];
    let d = s.to_dense();
    for r in 0..2 {
        for col in 0..2 {
            assert_eq!(
                d[[r, col]],
                expected[r][col],
                "sum_duplicates zero-sum mismatch at ({r},{col})"
            );
        }
    }
    Ok(())
}

/// REQ-MISSING-METHODS (`max`/`min`). For an all-NEGATIVE sparse diagonal that is
/// not fully dense, the implicit zero wins the `max()` (folded in), while `min()`
/// is the most-negative stored value, mirroring scipy `coo_matrix.max()`/`.min()`
/// (`_minmax_mixin._min_or_max`, `scipy/sparse/_data.py:208`-`:224` — implicit
/// zero folded when `nnz != prod(shape)`).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   A=sp.coo_matrix((np.array([-3.,-1.,-5.]),(np.array([0,1,2]),np.array([0,1,2]))),
///     shape=(3,3));
///   print(A.max(), A.min())"`): `0.0 -5.0`.
#[test]
fn coo_max_folds_implicit_zero() -> Result<(), FerroError> {
    let a = CooMatrix::<f64>::from_triplets(
        3,
        3,
        vec![0, 1, 2],
        vec![0, 1, 2],
        vec![-3.0, -1.0, -5.0],
    )?;
    // scipy A.max() == 0.0 (implicit zero folded in: nnz 3 < 9)
    assert_eq!(a.max(), 0.0);
    // scipy A.min() == -5.0 (most-negative stored value)
    assert_eq!(a.min(), -5.0);
    Ok(())
}

/// REQ-MISSING-METHODS (`max`/`min`). Symmetric: an all-POSITIVE sparse diagonal
/// that is not fully dense folds the implicit zero into `min()`, while `max()` is
/// the largest stored value, mirroring scipy `_minmax_mixin._min_or_max`
/// (`scipy/sparse/_data.py:222`).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   B=sp.coo_matrix((np.array([3.,1.,5.]),(np.array([0,1,2]),np.array([0,1,2]))),
///     shape=(3,3));
///   print(B.max(), B.min())"`): `5.0 0.0`.
#[test]
fn coo_min_folds_implicit_zero() -> Result<(), FerroError> {
    let b =
        CooMatrix::<f64>::from_triplets(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![3.0, 1.0, 5.0])?;
    // scipy B.max() == 5.0 (largest stored value)
    assert_eq!(b.max(), 5.0);
    // scipy B.min() == 0.0 (implicit zero folded in: nnz 3 < 9)
    assert_eq!(b.min(), 0.0);
    Ok(())
}

/// REQ-MISSING-METHODS (`max`/`min`). A FULLY DENSE matrix (`nnz == n_rows*n_cols`)
/// has no implicit zero, so `max()`/`min()` reduce over the stored data ONLY,
/// mirroring scipy `_minmax_mixin._min_or_max` skipping the fold when
/// `nnz == prod(shape)` (`scipy/sparse/_data.py:222`).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   C=sp.coo_matrix((np.array([2.,7.]),(np.array([0,0]),np.array([0,1]))),
///     shape=(1,2));
///   print(C.max(), C.min())"`): `7.0 2.0`.
#[test]
fn coo_max_min_dense_no_implicit_zero() -> Result<(), FerroError> {
    let c = CooMatrix::<f64>::from_triplets(1, 2, vec![0, 0], vec![0, 1], vec![2.0, 7.0])?;
    // scipy C.max() == 7.0 (fully dense, no implicit zero)
    assert_eq!(c.max(), 7.0);
    // scipy C.min() == 2.0 (fully dense, no implicit zero)
    assert_eq!(c.min(), 2.0);
    Ok(())
}

/// REQ-MISSING-METHODS (`astype`). Casting a float COO matrix to integers
/// truncates each stored value toward zero (numpy C-cast = Rust `as`) while
/// preserving the row/col index structure and nnz, mirroring scipy
/// `coo_matrix.astype(np.int64)` (`scipy/sparse/_data.py:69` — casts `self.data`,
/// keeps the index structure unchanged).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   m=sp.coo_matrix((np.array([3.7,-2.9,5.0]),(np.array([0,1,2]),np.array([0,1,2]))),
///     shape=(3,3));
///   c=m.astype(np.int64);
///   print(c.data.tolist(), c.row.tolist(), c.col.tolist(), c.nnz,
///         c.toarray().tolist())"`):
/// `[3, -2, 5] [0, 1, 2] [0, 1, 2] 3 [[3, 0, 0], [0, -2, 0], [0, 0, 5]]`.
#[test]
fn coo_astype_float_to_int_truncates() -> Result<(), FerroError> {
    let m =
        CooMatrix::<f64>::from_triplets(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![3.7, -2.9, 5.0])?;

    let c: CooMatrix<i64> = m.astype(|&v| v as i64)?;
    // scipy c.data == [3, -2, 5] (truncated toward zero)
    assert_eq!(c.data(), &[3, -2, 5]);
    // scipy c.row == [0, 1, 2], c.col == [0, 1, 2] (structure preserved)
    assert_eq!(c.row(), &[0, 1, 2]);
    assert_eq!(c.col(), &[0, 1, 2]);
    // scipy c.nnz == 3
    assert_eq!(c.nnz(), 3);

    // scipy c.toarray() == [[3,0,0],[0,-2,0],[0,0,5]]
    let expected = [[3, 0, 0], [0, -2, 0], [0, 0, 5]];
    let d = c.to_dense();
    for r in 0..3 {
        for col in 0..3 {
            assert_eq!(
                d[[r, col]],
                expected[r][col],
                "astype mismatch at ({r},{col})"
            );
        }
    }
    Ok(())
}

/// REQ-MISSING-METHODS (`astype`). Casting an f64 COO matrix to f32 narrows each
/// stored value while preserving the row/col index structure and nnz, mirroring
/// scipy `coo_matrix.astype(np.float32)` (`scipy/sparse/_data.py:69`). The cast
/// is compared against the actual f32 literals so it round-trips exactly.
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   m=sp.coo_matrix((np.array([3.7,-2.9,5.0]),(np.array([0,1,2]),np.array([0,1,2]))),
///     shape=(3,3));
///   c=m.astype(np.float32);
///   print(c.data.tolist(), c.row.tolist(), c.col.tolist(), c.nnz)"`):
/// `[3.700000047683716, -2.9000000953674316, 5.0] [0, 1, 2] [0, 1, 2] 3`.
#[test]
fn coo_astype_to_f32_preserves_structure() -> Result<(), FerroError> {
    let m =
        CooMatrix::<f64>::from_triplets(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![3.7, -2.9, 5.0])?;

    let c: CooMatrix<f32> = m.astype(|&v| v as f32)?;
    // scipy c.data == [3.7f32, -2.9f32, 5.0f32] (the exact narrowed values)
    assert_eq!(c.data(), &[3.7f32, -2.9f32, 5.0f32]);
    // scipy c.row == [0, 1, 2], c.col == [0, 1, 2], c.nnz == 3 (structure preserved)
    assert_eq!(c.row(), &[0, 1, 2]);
    assert_eq!(c.col(), &[0, 1, 2]);
    assert_eq!(c.nnz(), 3);
    Ok(())
}

/// REQ-MISSING-METHODS (`power`). `power(n)` raises every STORED value to the
/// power `n`, preserving the sparsity structure (row/col indices, shape, nnz),
/// mirroring scipy `coo_matrix.power(n)` (`scipy/sparse/_data.py:99` — operates
/// on `self.data` only; implicit zeros stay zero since `0^n=0` for `n>0`).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   m=sp.coo_matrix((np.array([2.,-3.,4.]),(np.array([0,1,2]),np.array([0,1,2]))),
///     shape=(3,3));
///   p2=m.power(2); p3=m.power(3);
///   print(p2.data.tolist(), p2.row.tolist(), p2.col.tolist(), p2.nnz,
///         p2.toarray().tolist(), p3.data.tolist())"`):
/// `[4.0, 9.0, 16.0] [0, 1, 2] [0, 1, 2] 3
///   [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 16.0]] [8.0, -27.0, 64.0]`.
#[test]
fn coo_power_squares_matches_scipy() -> Result<(), FerroError> {
    let m =
        CooMatrix::<f64>::from_triplets(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![2.0, -3.0, 4.0])?;

    let p2 = m.power(2.0)?;
    // scipy p2.data == [4, 9, 16] (each stored value squared)
    assert_eq!(p2.data(), &[4.0, 9.0, 16.0]);
    // scipy p2.row == [0,1,2], p2.col == [0,1,2], p2.nnz == 3 (structure preserved)
    assert_eq!(p2.row(), &[0, 1, 2]);
    assert_eq!(p2.col(), &[0, 1, 2]);
    assert_eq!(p2.nnz(), 3);

    // scipy p2.toarray() == [[4,0,0],[0,9,0],[0,0,16]]
    let expected = [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 16.0]];
    let d = p2.to_dense();
    for r in 0..3 {
        for col in 0..3 {
            assert_eq!(
                d[[r, col]],
                expected[r][col],
                "power(2) mismatch at ({r},{col})"
            );
        }
    }

    let p3 = m.power(3.0)?;
    // scipy p3.data == [8, -27, 64] (sign-preserving odd power)
    assert_eq!(p3.data(), &[8.0, -27.0, 64.0]);
    Ok(())
}

/// REQ-MISSING-METHODS (`power`). Fractional power: `power(0.5)` takes the
/// element-wise square root of each stored value, preserving structure, mirroring
/// scipy `coo_matrix.power(0.5)` (`scipy/sparse/_data.py:99`). Square roots of
/// perfect squares are bit-exact in f64.
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   m=sp.coo_matrix((np.array([4.,9.,16.]),(np.array([0,1,2]),np.array([0,1,2]))),
///     shape=(3,3));
///   p=m.power(0.5);
///   print(p.data.tolist(), p.row.tolist(), p.col.tolist(), p.nnz)"`):
/// `[2.0, 3.0, 4.0] [0, 1, 2] [0, 1, 2] 3`.
#[test]
fn coo_power_sqrt_matches_scipy() -> Result<(), FerroError> {
    let m =
        CooMatrix::<f64>::from_triplets(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![4.0, 9.0, 16.0])?;

    let p = m.power(0.5)?;
    // scipy p.data == [2, 3, 4] (exact sqrt of perfect squares)
    assert_eq!(p.data(), &[2.0, 3.0, 4.0]);
    // structure preserved
    assert_eq!(p.row(), &[0, 1, 2]);
    assert_eq!(p.col(), &[0, 1, 2]);
    assert_eq!(p.nnz(), 3);
    Ok(())
}

/// REQ-MISSING-METHODS (`dot`). The COO `@` COO matrix product mirrors scipy
/// `coo_matrix.dot(other)` (`scipy/sparse/_base.py:587` → `_matmul_sparse`
/// `:898`): COO has no native matmul, so scipy routes through `tocsr()` and
/// returns a **CSR** result (arithmetic defaults to CSR, `_base.py:484`).
/// `dot()` delegates `self.to_csr()?.matmul(&rhs.to_csr()?)`, so its result is a
/// `CsrMatrix`; the dense form is compared to avoid coupling to CSR's internal
/// ordering.
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   A=sp.coo_matrix((np.array([1.,2.,3.]),(np.array([0,0,1]),np.array([0,1,1]))),
///     shape=(2,2));
///   B=sp.coo_matrix((np.array([4.,5.,6.]),(np.array([0,1,1]),np.array([0,0,1]))),
///     shape=(2,2));
///   print(A.dot(B).toarray().tolist())"`): `[[14.0, 12.0], [15.0, 18.0]]`.
/// A=[[1,2],[0,3]], B=[[4,0],[5,6]].
#[test]
fn coo_dot_matches_scipy() -> Result<(), FerroError> {
    // A = [[1,2],[0,3]]
    let a =
        CooMatrix::<f64>::from_triplets(2, 2, vec![0, 0, 1], vec![0, 1, 1], vec![1.0, 2.0, 3.0])?;
    // B = [[4,0],[5,6]]
    let b =
        CooMatrix::<f64>::from_triplets(2, 2, vec![0, 1, 1], vec![0, 0, 1], vec![4.0, 5.0, 6.0])?;

    let c = a.dot(&b)?;
    // scipy A.dot(B).toarray() == [[14,12],[15,18]]
    let expected = [[14.0, 12.0], [15.0, 18.0]];
    let d = c.to_dense();
    assert_eq!(c.n_rows(), 2);
    assert_eq!(c.n_cols(), 2);
    for r in 0..2 {
        for col in 0..2 {
            assert_eq!(d[[r, col]], expected[r][col], "dot mismatch at ({r},{col})");
        }
    }
    Ok(())
}

/// REQ-MISSING-METHODS (`dot`). Non-square operands: `(2×3) @ (3×2) -> (2×2)`,
/// mirroring scipy `coo_matrix.dot` (`scipy/sparse/_base.py:587` → CSR
/// `_matmul_sparse`). Result CSR's dense form is compared to the oracle.
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   D=sp.coo_matrix((np.array([1.,2.,3.,4.]),(np.array([0,0,1,1]),np.array([0,2,0,1]))),
///     shape=(2,3));
///   E=sp.coo_matrix((np.array([5.,6.,7.]),(np.array([0,1,2]),np.array([1,0,1]))),
///     shape=(3,2));
///   print(D.dot(E).toarray().tolist())"`): `[[0.0, 19.0], [24.0, 15.0]]`.
/// D=[[1,0,2],[3,4,0]], E=[[0,5],[6,0],[0,7]].
#[test]
fn coo_dot_nonsquare_matches_scipy() -> Result<(), FerroError> {
    // D = [[1,0,2],[3,4,0]] (2×3)
    let d_coo = CooMatrix::<f64>::from_triplets(
        2,
        3,
        vec![0, 0, 1, 1],
        vec![0, 2, 0, 1],
        vec![1.0, 2.0, 3.0, 4.0],
    )?;
    // E = [[0,5],[6,0],[0,7]] (3×2)
    let e_coo =
        CooMatrix::<f64>::from_triplets(3, 2, vec![0, 1, 2], vec![1, 0, 1], vec![5.0, 6.0, 7.0])?;

    let c = d_coo.dot(&e_coo)?;
    // scipy D.dot(E).toarray() == [[0,19],[24,15]] (shape (2,2))
    assert_eq!(c.n_rows(), 2);
    assert_eq!(c.n_cols(), 2);
    let expected = [[0.0, 19.0], [24.0, 15.0]];
    let dense = c.to_dense();
    for r in 0..2 {
        for col in 0..2 {
            assert_eq!(
                dense[[r, col]],
                expected[r][col],
                "dot non-square mismatch at ({r},{col})"
            );
        }
    }
    Ok(())
}
