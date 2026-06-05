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
