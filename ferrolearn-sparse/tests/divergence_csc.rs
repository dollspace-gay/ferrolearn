//! Divergence / conformance tests for `ferrolearn-sparse/src/csc.rs`
//! (`CscMatrix`, the `scipy.sparse.csc_matrix` analog — the column-symmetric
//! twin of `CsrMatrix`) vs the LIVE scipy 1.17.1 oracle. Crosslink translation
//! unit #2007.
//!
//! All expected values are computed by a live scipy call run from `/tmp`
//! (R-CHAR-3 — NEVER copied from the ferrolearn side). The oracle command and
//! its output are quoted below so the target is unambiguous.
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np, scipy.sparse as sp
//! A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]))
//! print('nnz',A.nnz,'toarray',A.toarray().tolist())
//! print('matvec',(A@np.array([1.,2,3])).tolist())
//! print('A+A',(A+A).toarray().tolist())
//! print('2A',(A*2).toarray().tolist())
//! print('colslice',A[:,0:2].toarray().tolist())
//! print('tocsr',A.tocsr().toarray().tolist())
//! B=sp.csc_matrix(np.array([[1.,2,0],[0,1,3],[1,0,1]]))
//! print('A+B',(A+B).toarray().tolist())
//! print('fromdense',sp.csc_matrix(np.array([[0.,1],[2,0]])).toarray().tolist())"
//! ```
//! prints (scipy 1.17.1):
//! ```text
//! nnz 5 toarray [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
//! matvec [7.0, 6.0, 19.0]
//! A+A [[2.0, 0.0, 4.0], [0.0, 6.0, 0.0], [8.0, 0.0, 10.0]]
//! 2A [[2.0, 0.0, 4.0], [0.0, 6.0, 0.0], [8.0, 0.0, 10.0]]
//! colslice [[1.0, 0.0], [0.0, 3.0], [4.0, 0.0]]
//! tocsr [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
//! A+B [[2.0, 2.0, 2.0], [0.0, 4.0, 3.0], [5.0, 0.0, 6.0]]
//! fromdense [[0.0, 1.0], [2.0, 0.0]]
//! ```
//!
//! These are GREEN guards: the SHIPPED construction / conversion / matvec / add /
//! scalar-mul / col-slice / error core matches the oracle and PASSES now, guarding
//! against regression. No RED pin is present: construction/conversion/arithmetic
//! all match scipy exactly, so there is no genuine single-file-fixable divergence
//! in the SHIPPED behavior. The NOT-STARTED REQs (sparse-sparse matmul, transpose,
//! reduce, elementwise, indexing, API accessors, ferray substrate) are structural
//! blockers filed as `-l blocker` issues, not pinned here as doomed tests
//! (R-DEFER-3).

use ferrolearn_core::FerroError;
use ferrolearn_sparse::{CooMatrix, CscMatrix};
use ndarray::{Array1, array};

/// Canonical matrix `A = [[1,0,2],[0,3,0],[4,0,5]]` in CSC form.
///
/// CSC value order is column-major `[1,4,3,2,5]` (col 0: rows 0,2 -> 1,4;
/// col 1: row 1 -> 3; col 2: rows 0,2 -> 2,5).
fn sample_a() -> CscMatrix<f64> {
    CscMatrix::new(
        3,
        3,
        vec![0, 2, 3, 5],
        vec![0, 2, 1, 0, 2],
        vec![1.0, 4.0, 3.0, 2.0, 5.0],
    )
    .unwrap()
}

/// Helper matrix `B = [[1,2,0],[0,1,3],[1,0,1]]` in CSC form.
///
/// Column-major: col 0: rows 0,2 -> 1,1; col 1: rows 0,1 -> 2,1;
/// col 2: rows 1,2 -> 3,1.
fn sample_b() -> CscMatrix<f64> {
    CscMatrix::new(
        3,
        3,
        vec![0, 2, 4, 6],
        vec![0, 2, 0, 1, 1, 2],
        vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0],
    )
    .unwrap()
}

fn assert_dense_eq(d: &ndarray::Array2<f64>, expected: &[[f64; 3]; 3]) {
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "mismatch at ({r},{c})");
        }
    }
}

/// REQ-CONSTRUCT-CONVERT. `from_dense` -> `to_dense` round-trips and `nnz()`
/// counts distinct stored entries, matching scipy `.toarray()` / `.nnz`.
///
/// Oracle: `sp.csc_matrix([[1,0,2],[0,3,0],[4,0,5]])` -> `.nnz == 5`,
/// `.toarray() == [[1,0,2],[0,3,0],[4,0,5]]`.
#[test]
fn csc_from_dense_to_dense_and_nnz_match_scipy() {
    let dense = array![[1.0_f64, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
    let m = CscMatrix::from_dense(&dense.view(), 0.0);
    // scipy .nnz == 5
    assert_eq!(m.nnz(), 5);
    // scipy .toarray() == [[1,0,2],[0,3,0],[4,0,5]]
    let expected = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
    assert_dense_eq(&m.to_dense(), &expected);
}

/// REQ-CONSTRUCT-CONVERT (small from_dense). Oracle:
/// `sp.csc_matrix([[0,1],[2,0]]).toarray() == [[0,1],[2,0]]`.
#[test]
fn csc_from_dense_small_matches_scipy() {
    let dense = array![[0.0_f64, 1.0], [2.0, 0.0]];
    let m = CscMatrix::from_dense(&dense.view(), 0.0);
    assert_eq!(m.nnz(), 2);
    let d = m.to_dense();
    assert_eq!(d[[0, 0]], 0.0);
    assert_eq!(d[[0, 1]], 1.0);
    assert_eq!(d[[1, 0]], 2.0);
    assert_eq!(d[[1, 1]], 0.0);
}

/// REQ-CONSTRUCT-CONVERT. `from_coo` builds the canonical matrix; `to_dense`
/// matches scipy `.toarray()`.
///
/// Oracle: same matrix as `sample_a` -> `.toarray() == [[1,0,2],[0,3,0],[4,0,5]]`,
/// `.nnz == 5`.
#[test]
fn csc_from_coo_to_dense_matches_scipy() {
    let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
    coo.push(0, 0, 1.0).unwrap();
    coo.push(0, 2, 2.0).unwrap();
    coo.push(1, 1, 3.0).unwrap();
    coo.push(2, 0, 4.0).unwrap();
    coo.push(2, 2, 5.0).unwrap();
    let csc = CscMatrix::from_coo(&coo).unwrap();
    assert_eq!(csc.nnz(), 5);
    let expected = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
    assert_dense_eq(&csc.to_dense(), &expected);
}

/// REQ-CONSTRUCT-CONVERT. `to_csr()` materialized to dense round-trips the
/// canonical matrix, matching scipy `.tocsr().toarray()`.
///
/// Oracle: `A.tocsr().toarray() == [[1,0,2],[0,3,0],[4,0,5]]`.
#[test]
fn csc_to_csr_roundtrip_matches_scipy() {
    let a = sample_a();
    let csr = a.to_csr();
    let d = csr.to_dense();
    let expected = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "tocsr mismatch at ({r},{c})");
        }
    }
}

/// REQ-MATVEC. `mul_vec` computes `A @ v`, matching scipy `_matmul_vector`.
///
/// Oracle: `A @ [1,2,3] == [7.0, 6.0, 19.0]`.
#[test]
fn csc_mul_vec_matches_scipy() {
    let a = sample_a();
    let v = Array1::from(vec![1.0_f64, 2.0, 3.0]);
    let r = a.mul_vec(&v).unwrap();
    // scipy A @ [1,2,3] == [7,6,19]
    assert_eq!(r[0], 7.0);
    assert_eq!(r[1], 6.0);
    assert_eq!(r[2], 19.0);
}

/// REQ-ADD. `add(&A)` matches scipy `A + A` (elementwise).
///
/// Oracle: `(A+A).toarray() == [[2,0,4],[0,6,0],[8,0,10]]`.
#[test]
fn csc_add_self_matches_scipy() {
    let a = sample_a();
    let sum = a.add(&a).unwrap();
    let expected = [[2.0, 0.0, 4.0], [0.0, 6.0, 0.0], [8.0, 0.0, 10.0]];
    assert_dense_eq(&sum.to_dense(), &expected);
}

/// REQ-ADD. `add(&B)` matches scipy `A + B` (elementwise) for the helper B.
///
/// Oracle: B = [[1,2,0],[0,1,3],[1,0,1]];
/// `(A+B).toarray() == [[2,2,2],[0,4,3],[5,0,6]]`.
#[test]
fn csc_add_other_matches_scipy() {
    let a = sample_a();
    let b = sample_b();
    let sum = a.add(&b).unwrap();
    let expected = [[2.0, 2.0, 2.0], [0.0, 4.0, 3.0], [5.0, 0.0, 6.0]];
    assert_dense_eq(&sum.to_dense(), &expected);
}

/// REQ-SCALAR-MUL. `mul_scalar(2.0)` (new) and `scale(2.0)` (in place) match
/// scipy `A * 2`.
///
/// Oracle: `(A*2).toarray() == [[2,0,4],[0,6,0],[8,0,10]]`.
#[test]
fn csc_scalar_mul_matches_scipy() {
    let expected = [[2.0, 0.0, 4.0], [0.0, 6.0, 0.0], [8.0, 0.0, 10.0]];

    // new-matrix path
    let a = sample_a();
    let m2 = a.mul_scalar(2.0);
    assert_dense_eq(&m2.to_dense(), &expected);

    // in-place path on a clone
    let mut a2 = sample_a();
    a2.scale(2.0);
    assert_dense_eq(&a2.to_dense(), &expected);
}

/// REQ-COL-SLICE. `col_slice(0,2)` matches scipy `A[:,0:2]` (COLUMN slice — the
/// CSC analog of CSR's `row_slice`).
///
/// Oracle: `A[:,0:2].toarray() == [[1,0],[0,3],[4,0]]`.
#[test]
fn csc_col_slice_matches_scipy() {
    let a = sample_a();
    let sliced = a.col_slice(0, 2).unwrap();
    assert_eq!(sliced.n_rows(), 3);
    assert_eq!(sliced.n_cols(), 2);
    let d = sliced.to_dense();
    let expected = [[1.0, 0.0], [0.0, 3.0], [4.0, 0.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "col_slice mismatch at ({r},{c})");
        }
    }
}

/// REQ-ERR. `add` with an incompatible shape returns `Err`, where scipy raises
/// `ValueError: inconsistent shapes`.
///
/// Oracle: `A + sp.csc_matrix((2,3))` -> `ValueError: inconsistent shapes`.
#[test]
fn csc_add_shape_mismatch_is_err() {
    let a = sample_a();
    // empty 2x3 matrix (shape (2,3) vs A's (3,3))
    let c = CscMatrix::<f64>::new(2, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    assert!(
        a.add(&c).is_err(),
        "shape-mismatched add must return Err (scipy raises ValueError)"
    );
}

/// REQ-ERR. `mul_vec` with a wrong-length vector returns `Err`, where scipy
/// raises `ValueError: dimension mismatch`.
///
/// Oracle: `A @ np.array([1.,2.])` (len 2, A has 3 cols) ->
/// `ValueError: dimension mismatch`.
#[test]
fn csc_mul_vec_shape_mismatch_is_err() {
    let a = sample_a();
    let v = Array1::from(vec![1.0_f64, 2.0]);
    assert!(
        a.mul_vec(&v).is_err(),
        "wrong-length matvec must return Err (scipy raises ValueError)"
    );
}

/// REQ-CONSTRUCT-CONVERT geometry. `n_rows()`/`n_cols()` mirror scipy `.shape`
/// tuple elements; `nnz()` mirrors `.nnz`.
///
/// Oracle: `A.shape == (3, 3)`, `A.nnz == 5`.
#[test]
fn csc_geometry_matches_scipy() {
    let a = sample_a();
    assert_eq!((a.n_rows(), a.n_cols()), (3, 3));
    assert_eq!(a.nnz(), 5);
}

/// Elementwise helper matrix `B = [[1,1,0],[0,1,1],[0,0,1]]` in CSC form.
///
/// Column-major: col 0: row 0 -> 1; col 1: rows 0,1 -> 1,1; col 2: rows 1,2 ->
/// 1,1. So `indptr = [0,1,3,5]`, `indices = [0,0,1,1,2]`, `data = [1,1,1,1,1]`.
/// This is the `B` used in the REQ-MISSING-ELEMENTWISE oracle (the CSR-twin B),
/// distinct from the `sample_b` used by the add oracle above.
fn sample_b_elmul() -> CscMatrix<f64> {
    CscMatrix::new(
        3,
        3,
        vec![0, 1, 3, 5],
        vec![0, 0, 1, 1, 2],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap()
}

/// REQ-MISSING-ELEMENTWISE. `multiply(&B)` matches scipy `A.multiply(B)`
/// (element-wise Hadamard product, INTERSECTION sparsity).
///
/// Oracle: B = [[1,1,0],[0,1,1],[0,0,1]];
/// `A.multiply(B).toarray() == [[1,0,0],[0,3,0],[0,0,5]]`.
#[test]
fn csc_multiply_matches_scipy() {
    let a = sample_a();
    let b = sample_b_elmul();
    let prod = a.multiply(&b).unwrap();
    let expected = [[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]];
    assert_dense_eq(&prod.to_dense(), &expected);
}

/// REQ-MISSING-ELEMENTWISE. `sub(&B)` matches scipy `A - B` (element-wise,
/// UNION sparsity).
///
/// Oracle: B = [[1,1,0],[0,1,1],[0,0,1]];
/// `(A-B).toarray() == [[0,-1,2],[0,2,-1],[4,0,4]]`.
#[test]
fn csc_sub_matches_scipy() {
    let a = sample_a();
    let b = sample_b_elmul();
    let diff = a.sub(&b).unwrap();
    let expected = [[0.0, -1.0, 2.0], [0.0, 2.0, -1.0], [4.0, 0.0, 4.0]];
    assert_dense_eq(&diff.to_dense(), &expected);
}

/// REQ-MISSING-ELEMENTWISE / REQ-ERR. `multiply` with an incompatible shape
/// returns `Err`, where scipy raises `ValueError: inconsistent shapes`.
#[test]
fn csc_multiply_shape_mismatch_is_err() {
    let a = sample_a();
    // empty 2x3 matrix (shape (2,3) vs A's (3,3))
    let c = CscMatrix::<f64>::new(2, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    assert!(
        a.multiply(&c).is_err(),
        "shape-mismatched multiply must return Err (scipy raises ValueError)"
    );
}

/// REQ-MISSING-ELEMENTWISE / REQ-ERR. `sub` with an incompatible shape returns
/// `Err`, where scipy raises `ValueError: inconsistent shapes`.
#[test]
fn csc_sub_shape_mismatch_is_err() {
    let a = sample_a();
    // empty 2x3 matrix (shape (2,3) vs A's (3,3))
    let c = CscMatrix::<f64>::new(2, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    assert!(
        a.sub(&c).is_err(),
        "shape-mismatched sub must return Err (scipy raises ValueError)"
    );
}

// REQ-MISSING-MATMUL — live scipy oracle (R-CHAR-3). Expected values from
// `cd /tmp && python3 -c "
//   import numpy as np, scipy.sparse as sp
//   A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]))
//   B=sp.csc_matrix(np.array([[1.,1,0],[0,1,1],[0,0,1]]))
//   C=sp.csc_matrix(np.array([[1.,2],[3,4],[5,6]]))
//   print((A@B).toarray().tolist(), (A@C).toarray().tolist())"`
//   -> [[1,1,2],[0,3,3],[4,4,5]] [[11,14],[9,12],[29,38]].

/// Non-square helper `C = [[1,2],[3,4],[5,6]]` (3×2) in CSC form, built via
/// `from_dense` (infallible, no unwrap) to match the test idiom.
fn sample_c() -> CscMatrix<f64> {
    let dense = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    CscMatrix::from_dense(&dense.view(), 0.0)
}

/// REQ-MISSING-MATMUL. `matmul(&B)` matches scipy `A @ B` (sparse-sparse
/// product, SMMP).
///
/// Oracle: B = [[1,1,0],[0,1,1],[0,0,1]];
/// `(A@B).toarray() == [[1,1,2],[0,3,3],[4,4,5]]`.
#[test]
fn csc_matmul_matches_scipy() {
    let a = sample_a();
    let b = sample_b_elmul();
    let prod = a.matmul(&b).unwrap();
    let expected = [[1.0, 1.0, 2.0], [0.0, 3.0, 3.0], [4.0, 4.0, 5.0]];
    assert_dense_eq(&prod.to_dense(), &expected);
}

/// REQ-MISSING-MATMUL. `matmul(&C)` for a non-square right operand matches scipy
/// `A @ C`, including the `(3,2)` output shape.
///
/// Oracle: C = [[1,2],[3,4],[5,6]];
/// `(A@C).toarray() == [[11,14],[9,12],[29,38]]`, shape `(3,2)`.
#[test]
fn csc_matmul_non_square() {
    let a = sample_a();
    let c = sample_c();
    let prod = a.matmul(&c).unwrap();
    assert_eq!(prod.n_rows(), 3);
    assert_eq!(prod.n_cols(), 2);
    let d = prod.to_dense();
    let expected = [[11.0, 14.0], [9.0, 12.0], [29.0, 38.0]];
    for (r, row) in expected.iter().enumerate() {
        for (col, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, col]], v, "matmul mismatch at ({r},{col})");
        }
    }
}

/// REQ-MISSING-MATMUL / REQ-ERR. `matmul` with an incompatible inner dimension
/// returns `Err`, where scipy raises `ValueError: dimension mismatch`.
///
/// Oracle: `A` has 3 cols; `D` is 2×2 (2 rows != 3) -> dimension mismatch.
#[test]
fn csc_matmul_shape_mismatch_is_err() {
    let a = sample_a();
    // 2x2 matrix: D.n_rows() (2) != A.n_cols() (3)
    let d = CscMatrix::<f64>::new(2, 2, vec![0, 0, 0], vec![], vec![]).unwrap();
    assert!(
        a.matmul(&d).is_err(),
        "inner-dimension-mismatched matmul must return Err (scipy raises ValueError)"
    );
}

// REQ-API-ACCESSORS — live scipy oracle (R-CHAR-3). Expected values from
// `cd /tmp && python3 -c "
//   import numpy as np, scipy.sparse as sp
//   A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]))
//   print(A.shape, A.data.tolist(), A.indices.tolist(), A.indptr.tolist())"`
//   -> (3, 3) [1.0,4.0,3.0,2.0,5.0] [0,2,1,0,2] [0,2,3,5].
// NOTE the CSC `data` order is COLUMN-major `[1,4,3,2,5]` (vs CSR's
// `[1,2,3,4,5]`); `indices` are ROW indices and `indptr` is the COLUMN pointer.

// REQ-MISSING-INDEX (element access) — live scipy oracle (R-CHAR-3). Expected
// values from `cd /tmp && python3 -c "
//   import numpy as np, scipy.sparse as sp
//   A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]))
//   print(A[1,1], A[0,1], A[0,0], A[0,2], A[2,0])"`
//   -> 3.0 0.0 1.0 2.0 4.0  (A[0,1] is structurally absent -> 0).

/// REQ-MISSING-INDEX (element access). `get(i,j)` returns the stored value at
/// `(i,j)`, matching scipy `A[i,j]` (`IndexMixin.__getitem__` -> `_get_intXint`).
///
/// Oracle: `A[1,1]==3`, `A[0,0]==1`, `A[0,2]==2`, `A[2,0]==4`.
#[test]
fn csc_get_element_matches_scipy() {
    let a = sample_a();
    assert_eq!(a.get(1, 1).unwrap(), 3.0);
    assert_eq!(a.get(0, 0).unwrap(), 1.0);
    assert_eq!(a.get(0, 2).unwrap(), 2.0);
    assert_eq!(a.get(2, 0).unwrap(), 4.0);
}

/// REQ-MISSING-INDEX (element access). A structurally absent position returns
/// `0`, matching scipy `A[i,j]` for an unstored entry.
///
/// Oracle: `A[0,1] == 0.0` (no stored entry at row 0, col 1).
#[test]
fn csc_get_absent_is_zero() {
    let a = sample_a();
    assert_eq!(a.get(0, 1).unwrap(), 0.0);
}

/// REQ-MISSING-INDEX (element access) / REQ-ERR. An out-of-bounds index returns
/// `Err(InvalidParameter)`, where scipy raises `IndexError`.
///
/// Oracle: `A[3,0]` and `A[0,3]` (A is 3×3) -> `IndexError: ... out of range`.
#[test]
fn csc_get_out_of_bounds_is_err() {
    let a = sample_a();
    assert!(
        matches!(
            a.get(3, 0),
            Err(ferrolearn_core::FerroError::InvalidParameter { .. })
        ),
        "row index out of bounds must return Err(InvalidParameter)"
    );
    assert!(
        matches!(
            a.get(0, 3),
            Err(ferrolearn_core::FerroError::InvalidParameter { .. })
        ),
        "col index out of bounds must return Err(InvalidParameter)"
    );
}

// REQ-MISSING-INDEX (rows/cols) — live scipy oracle (R-CHAR-3). Expected values
// from `cd /tmp && python3 -c "
//   import numpy as np, scipy.sparse as sp
//   A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]))
//   print(A.getrow(0).toarray().tolist(), A.getrow(1).toarray().tolist())
//   print(A.getcol(0).toarray().tolist(), A.getcol(2).toarray().tolist())"`
//   -> [[1,0,2]] [[0,3,0]] / [[1],[0],[4]] [[2],[0],[5]].

/// REQ-MISSING-INDEX (rows/cols). `getrow(i)` returns row `i` as a `(1, n_cols)`
/// CSC, matching scipy `A.getrow(i)` (`_matrix.py:110` -> `_getrow`).
///
/// Oracle: `A.getrow(0).toarray() == [[1,0,2]]` (shape `(1,3)`),
/// `A.getrow(1).toarray() == [[0,3,0]]`.
#[test]
fn csc_getrow_matches_scipy() {
    let a = sample_a();
    let r0 = a.getrow(0).unwrap();
    assert_eq!(r0.n_rows(), 1);
    assert_eq!(r0.n_cols(), 3);
    let d0 = r0.to_dense();
    assert_eq!(d0[[0, 0]], 1.0);
    assert_eq!(d0[[0, 1]], 0.0);
    assert_eq!(d0[[0, 2]], 2.0);

    let r1 = a.getrow(1).unwrap();
    let d1 = r1.to_dense();
    assert_eq!(d1[[0, 0]], 0.0);
    assert_eq!(d1[[0, 1]], 3.0);
    assert_eq!(d1[[0, 2]], 0.0);
}

/// REQ-MISSING-INDEX (rows/cols). `getcol(j)` returns column `j` as a
/// `(n_rows, 1)` CSC, matching scipy `A.getcol(j)` (`_matrix.py:104` ->
/// `_getcol`). CSC is column-natural, so `getcol` delegates to `col_slice`.
///
/// Oracle: `A.getcol(0).toarray() == [[1],[0],[4]]` (shape `(3,1)`),
/// `A.getcol(2).toarray() == [[2],[0],[5]]`.
#[test]
fn csc_getcol_matches_scipy() {
    let a = sample_a();
    let c0 = a.getcol(0).unwrap();
    assert_eq!(c0.n_rows(), 3);
    assert_eq!(c0.n_cols(), 1);
    let d0 = c0.to_dense();
    assert_eq!(d0[[0, 0]], 1.0);
    assert_eq!(d0[[1, 0]], 0.0);
    assert_eq!(d0[[2, 0]], 4.0);

    let c2 = a.getcol(2).unwrap();
    let d2 = c2.to_dense();
    assert_eq!(d2[[0, 0]], 2.0);
    assert_eq!(d2[[1, 0]], 0.0);
    assert_eq!(d2[[2, 0]], 5.0);
}

/// REQ-MISSING-INDEX (rows/cols) / REQ-ERR. An out-of-bounds index returns
/// `Err(InvalidParameter)`, where scipy raises `IndexError`.
///
/// Oracle: `A.getrow(3)` and `A.getcol(3)` (A is 3×3) -> `IndexError`.
#[test]
fn csc_getrow_getcol_out_of_bounds_is_err() {
    let a = sample_a();
    assert!(
        matches!(
            a.getrow(3),
            Err(ferrolearn_core::FerroError::InvalidParameter { .. })
        ),
        "row index out of bounds must return Err(InvalidParameter)"
    );
    assert!(
        matches!(
            a.getcol(3),
            Err(ferrolearn_core::FerroError::InvalidParameter { .. })
        ),
        "col index out of bounds must return Err(InvalidParameter)"
    );
}

/// REQ-API-ACCESSORS. `shape()`/`data()`/`indices()`/`indptr()` match scipy's
/// `.shape`/`.data`/`.indices`/`.indptr` (`_compressed.py:38`, `:76-78`).
///
/// `A` is built via `from_dense` so the stored arrays land in scipy's canonical
/// CSC order; the assertion targets the live scipy CSC oracle exactly.
#[test]
fn csc_shape_data_indices_indptr_match_scipy() {
    let dense = array![[1.0_f64, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
    let a = CscMatrix::from_dense(&dense.view(), 0.0);
    // scipy A.shape == (3, 3)
    assert_eq!(a.shape(), (3, 3));
    // scipy A.data == [1,4,3,2,5] (column-major CSC order)
    assert_eq!(a.data(), &[1.0, 4.0, 3.0, 2.0, 5.0]);
    // scipy A.indices == [0,2,1,0,2] (row indices)
    assert_eq!(a.indices(), &[0, 2, 1, 0, 2]);
    // scipy A.indptr == [0,2,3,5] (column pointer)
    assert_eq!(a.indptr(), vec![0, 2, 3, 5]);
}

// REQ-MISSING-INDEX (maintenance: max/min/astype/copy/eliminate_zeros/power) —
// live scipy oracle (R-CHAR-3). Expected values from
// `cd /tmp && python3 -c "
//   import numpy as np, scipy.sparse as sp
//   print(sp.csc_matrix(np.diag([-3.,-1.,-5.])).max(),
//         sp.csc_matrix(np.diag([-3.,-1.,-5.])).min())
//   print(sp.csc_matrix(np.diag([3.,1.,5.])).max(),
//         sp.csc_matrix(np.diag([3.,1.,5.])).min())
//   print(sp.csc_matrix(np.diag([3.7,-2.9,5.0])).astype(np.int64).data.tolist())
//   D=sp.csc_matrix((np.array([3.,0.,5.]),np.array([0,1,2]),np.array([0,1,2,3])),shape=(3,3))
//   D.eliminate_zeros()
//   print(D.nnz, D.data.tolist(), D.indices.tolist(), D.indptr.tolist())
//   print(sp.csc_matrix(np.diag([2.,-3.])).power(2).data.tolist(),
//         sp.csc_matrix(np.diag([2.,-3.])).power(3).data.tolist())"`
//   -> 0.0 -5.0 / 5.0 0.0 / [3,-2,5] / 2 [3.0,5.0] [0,2] [0,1,1,2] /
//      [4.0,9.0] [8.0,-27.0].

/// REQ-MISSING-INDEX (maintenance). `max()`/`min()` fold an implicit zero when
/// the matrix is not fully dense, matching scipy `_minmax_mixin` `axis=None`.
///
/// Oracle: `diag(-3,-1,-5)` -> `max==0.0` (implicit zero), `min==-5.0`;
/// `diag(3,1,5)` -> `max==5.0`, `min==0.0` (implicit zero).
#[test]
fn csc_max_min_folds_implicit_zero() -> Result<(), FerroError> {
    let neg = array![[-3.0_f64, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -5.0]];
    let m_neg = CscMatrix::from_dense(&neg.view(), 0.0);
    assert_eq!(m_neg.max(), 0.0);
    assert_eq!(m_neg.min(), -5.0);

    let pos = array![[3.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]];
    let m_pos = CscMatrix::from_dense(&pos.view(), 0.0);
    assert_eq!(m_pos.max(), 5.0);
    assert_eq!(m_pos.min(), 0.0);
    Ok(())
}

/// REQ-MISSING-INDEX (maintenance). `astype` casts each stored value through the
/// supplied closure (truncation toward zero for `as i64`), preserving structure,
/// matching scipy `astype(np.int64)`.
///
/// Oracle: `diag(3.7,-2.9,5.0)` -> `astype(int64).data == [3,-2,5]`.
#[test]
fn csc_astype_truncates() -> Result<(), FerroError> {
    let dense = array![[3.7_f64, 0.0, 0.0], [0.0, -2.9, 0.0], [0.0, 0.0, 5.0]];
    let m = CscMatrix::from_dense(&dense.view(), 0.0);
    let cast: CscMatrix<i64> = m.astype(|&v| v as i64)?;
    // scipy astype(int64).data == [3,-2,5] (truncation toward zero)
    assert_eq!(cast.data(), &[3_i64, -2, 5]);
    // structure preserved: same column pointers / row indices / shape / nnz
    assert_eq!(cast.indptr(), m.indptr());
    assert_eq!(cast.indices(), m.indices());
    assert_eq!(cast.shape(), m.shape());
    assert_eq!(cast.nnz(), 3);
    Ok(())
}

/// REQ-MISSING-INDEX (maintenance). `copy()` clones every stored entry; nnz,
/// data, and dense match, and the original is unchanged, matching scipy `copy()`.
#[test]
fn csc_copy_preserves_structure() -> Result<(), FerroError> {
    let a = sample_a();
    let c = a.copy();
    assert_eq!(c.nnz(), a.nnz());
    assert_eq!(c.data(), a.data());
    assert_eq!(c.to_dense(), a.to_dense());
    // original unchanged
    assert_eq!(a.nnz(), 5);
    let expected = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
    assert_dense_eq(&a.to_dense(), &expected);
    Ok(())
}

/// REQ-MISSING-INDEX (maintenance). `eliminate_zeros()` drops an explicitly
/// stored zero (walking COLUMNS via the column pointer), matching scipy
/// `eliminate_zeros()`.
///
/// Oracle: CSC `data=[3,0,5]`, `indices=[0,1,2]`, `indptr=[0,1,2,3]`, shape
/// `(3,3)` -> `nnz==2`, `data==[3,5]`, `indices==[0,2]`, `indptr==[0,1,1,2]`,
/// dense `[[3,0,0],[0,0,0],[0,0,5]]`.
#[test]
fn csc_eliminate_zeros_matches_scipy() -> Result<(), FerroError> {
    // A distinct row per column (single entry/column, sorted) — `new` accepts a
    // stored zero at row 1, col 1.
    let m = CscMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![3.0_f64, 0.0, 5.0],
    )?;
    assert_eq!(m.nnz(), 3);
    let pruned = m.eliminate_zeros()?;
    assert_eq!(pruned.nnz(), 2);
    assert_eq!(pruned.data(), &[3.0, 5.0]);
    assert_eq!(pruned.indices(), &[0, 2]);
    assert_eq!(pruned.indptr(), vec![0, 1, 1, 2]);
    let expected = [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]];
    assert_dense_eq(&pruned.to_dense(), &expected);
    Ok(())
}

/// REQ-MISSING-INDEX (maintenance). `power(n)` raises each stored value to `n`,
/// preserving structure, matching scipy `power(n)`.
///
/// Oracle: `diag(2,-3)` -> `power(2).data == [4,9]`, `power(3).data == [8,-27]`.
#[test]
fn csc_power_matches_scipy() -> Result<(), FerroError> {
    let dense = array![[2.0_f64, 0.0], [0.0, -3.0]];
    let m = CscMatrix::from_dense(&dense.view(), 0.0);
    let p2 = m.power(2.0)?;
    assert_eq!(p2.data(), &[4.0, 9.0]);
    let p3 = m.power(3.0)?;
    assert_eq!(p3.data(), &[8.0, -27.0]);
    Ok(())
}

// REQ-MISSING-INDEX (maintenance: argmax/argmin) — live scipy oracle (R-CHAR-3).
// argmax/argmin (axis=None) return the flattened C-order (row-major) index
// `r*n_cols+c` of the extreme element over ALL positions (stored values AND
// implicit zeros), with ties broken to the smallest flat index. Expected values
// from `cd /tmp && python3 -c "
//   import numpy as np, scipy.sparse as sp
//   A=sp.csc_matrix(np.array([[1.,0,2],[0,5,0],[4,0,3]]))
//   print(A.argmax(), A.argmin())                          # 4 1
//   B=sp.csc_matrix(np.array([[-1.,-2],[-3,-4]]))
//   print(B.argmax(), B.argmin())                          # 0 3
//   C=sp.csc_matrix(np.array([[-1.,0],[-3,-4]]))
//   print(C.argmax(), C.argmin())                          # 1 3
//   Z=sp.csc_matrix(np.zeros((2,3)))
//   print(Z.argmax(), Z.argmin())                          # 0 0
//   T=sp.csc_matrix(np.array([[5.,5],[1,5]]))
//   print(T.argmax())                                      # 0
//   P=sp.csc_matrix(np.array([[2.,0,2]]))
//   print(P.argmax(), P.argmin())"`                        # 0 1

/// REQ-MISSING-INDEX (maintenance). `argmax`/`argmin` (axis=None) return the
/// C-order flat index of the extreme element, implicit zeros participating,
/// matching scipy `csc_matrix.argmax()`/`.argmin()`.
///
/// Oracle: A = [[1,0,2],[0,5,0],[4,0,3]];
/// `A.argmax() == 4` (the 5 at (1,1)), `A.argmin() == 1` (implicit zero at (0,1)).
#[test]
fn csc_argmax_matches_scipy() -> Result<(), FerroError> {
    let dense = array![[1.0_f64, 0.0, 2.0], [0.0, 5.0, 0.0], [4.0, 0.0, 3.0]];
    let a = CscMatrix::from_dense(&dense.view(), 0.0);
    assert_eq!(a.argmax()?, 4);
    assert_eq!(a.argmin()?, 1);
    Ok(())
}

/// REQ-MISSING-INDEX (maintenance). For a fully dense all-negative matrix, no
/// implicit zero exists, so the extremes are stored values.
///
/// Oracle: B = [[-1,-2],[-3,-4]] (fully dense);
/// `B.argmax() == 0` (the -1 at (0,0)), `B.argmin() == 3` (the -4 at (1,1)).
#[test]
fn csc_argmin_dense_all_negative() -> Result<(), FerroError> {
    let dense = array![[-1.0_f64, -2.0], [-3.0, -4.0]];
    let b = CscMatrix::from_dense(&dense.view(), 0.0);
    assert_eq!(b.argmax()?, 0);
    assert_eq!(b.argmin()?, 3);
    Ok(())
}

/// REQ-MISSING-INDEX (maintenance). An implicit zero participates and can be the
/// maximum when every stored value is negative.
///
/// Oracle: C = [[-1,0],[-3,-4]];
/// `C.argmax() == 1` (the zero at (0,1) beats all negatives), `C.argmin() == 3`.
#[test]
fn csc_argmax_implicit_zero() -> Result<(), FerroError> {
    let dense = array![[-1.0_f64, 0.0], [-3.0, -4.0]];
    let c = CscMatrix::from_dense(&dense.view(), 0.0);
    assert_eq!(c.argmax()?, 1);
    assert_eq!(c.argmin()?, 3);
    Ok(())
}

/// REQ-MISSING-INDEX (maintenance). An all-implicit-zeros matrix returns flat
/// index 0 for both argmax and argmin (the first position, tie to smallest).
///
/// Oracle: Z = zeros((2,3)); `Z.argmax() == 0`, `Z.argmin() == 0`.
#[test]
fn csc_argmax_all_zero() -> Result<(), FerroError> {
    let dense = array![[0.0_f64, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let z = CscMatrix::from_dense(&dense.view(), 0.0);
    assert_eq!(z.nnz(), 0);
    assert_eq!(z.argmax()?, 0);
    assert_eq!(z.argmin()?, 0);
    Ok(())
}

/// REQ-MISSING-INDEX (maintenance). Ties resolve to the EARLIEST (smallest
/// C-order flat) index.
///
/// Oracle: T = [[5,5],[1,5]]; `T.argmax() == 0` (first 5 in C-order). And
/// P = [[2,0,2]]; `P.argmax() == 0` (first 2), `P.argmin() == 1` (implicit zero).
#[test]
fn csc_argmax_ties_earliest() -> Result<(), FerroError> {
    let t_dense = array![[5.0_f64, 5.0], [1.0, 5.0]];
    let t = CscMatrix::from_dense(&t_dense.view(), 0.0);
    assert_eq!(t.argmax()?, 0);

    let p_dense = array![[2.0_f64, 0.0, 2.0]];
    let p = CscMatrix::from_dense(&p_dense.view(), 0.0);
    assert_eq!(p.argmax()?, 0);
    assert_eq!(p.argmin()?, 1);
    Ok(())
}
