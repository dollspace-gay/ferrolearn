//! Divergence / conformance tests for `ferrolearn-sparse/src/csr.rs`
//! (`CsrMatrix`, the `scipy.sparse.csr_matrix` analog) vs the LIVE scipy 1.17.1
//! oracle. Crosslink translation unit #1999.
//!
//! All expected values are computed by a live scipy call run from `/tmp`
//! (R-CHAR-3 — NEVER copied from the ferrolearn side). The oracle command and
//! its output are quoted below so the target is unambiguous.
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np, scipy.sparse as sp
//! A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]))
//! print('nnz',A.nnz,'toarray',A.toarray().tolist())
//! print('matvec',(A@np.array([1.,2,3])).tolist())
//! print('A+A',(A+A).toarray().tolist())
//! print('2A',(A*2).toarray().tolist())
//! print('rowslice',A[0:2].toarray().tolist())
//! print('tocsc',A.tocsc().toarray().tolist())
//! B=sp.csr_matrix(np.array([[1.,1,0],[0,1,1],[0,0,1]]))
//! print('A+B',(A+B).toarray().tolist())
//! print('fromdense', sp.csr_matrix(np.array([[0.,1],[2,0]])).toarray().tolist())"
//! ```
//! prints (scipy 1.17.1):
//! ```text
//! nnz 5 toarray [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
//! matvec [7.0, 6.0, 19.0]
//! A+A [[2.0, 0.0, 4.0], [0.0, 6.0, 0.0], [8.0, 0.0, 10.0]]
//! 2A [[2.0, 0.0, 4.0], [0.0, 6.0, 0.0], [8.0, 0.0, 10.0]]
//! rowslice [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]
//! tocsc [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
//! A+B [[2.0, 1.0, 2.0], [0.0, 4.0, 1.0], [4.0, 0.0, 6.0]]
//! fromdense [[0.0, 1.0], [2.0, 0.0]]
//! ```
//!
//! These are GREEN guards: the SHIPPED construction / conversion / matvec / add /
//! scalar-mul / row-slice / error core matches the oracle and PASSES now, guarding
//! against regression. No RED pin is present: construction/conversion/arithmetic
//! all match scipy exactly, so there is no genuine single-file-fixable divergence
//! in the SHIPPED behavior. The NOT-STARTED REQs (sparse-sparse matmul, transpose,
//! reduce, elementwise, indexing, API accessors, ferray substrate) are structural
//! blockers filed as `-l blocker` issues, not pinned here as doomed tests
//! (R-DEFER-3).

use ferrolearn_core::Dataset;
use ferrolearn_sparse::{CooMatrix, CsrMatrix};
use ndarray::{Array1, array};

/// Canonical matrix `A = [[1,0,2],[0,3,0],[4,0,5]]` in CSR form.
fn sample_a() -> CsrMatrix<f64> {
    CsrMatrix::new(
        3,
        3,
        vec![0, 2, 3, 5],
        vec![0, 2, 1, 0, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap()
}

/// Helper matrix `B = [[1,1,0],[0,1,1],[0,0,1]]` in CSR form.
fn sample_b() -> CsrMatrix<f64> {
    CsrMatrix::new(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 1, 1, 2, 2],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
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
/// Oracle: `sp.csr_matrix([[1,0,2],[0,3,0],[4,0,5]])` -> `.nnz == 5`,
/// `.toarray() == [[1,0,2],[0,3,0],[4,0,5]]`.
#[test]
fn csr_from_dense_to_dense_and_nnz_match_scipy() {
    let dense = array![[1.0_f64, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
    let m = CsrMatrix::from_dense(&dense.view(), 0.0);
    // scipy .nnz == 5
    assert_eq!(m.nnz(), 5);
    // scipy .toarray() == [[1,0,2],[0,3,0],[4,0,5]]
    let expected = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
    assert_dense_eq(&m.to_dense(), &expected);
}

/// REQ-CONSTRUCT-CONVERT (small from_dense). Oracle:
/// `sp.csr_matrix([[0,1],[2,0]]).toarray() == [[0,1],[2,0]]`.
#[test]
fn csr_from_dense_small_matches_scipy() {
    let dense = array![[0.0_f64, 1.0], [2.0, 0.0]];
    let m = CsrMatrix::from_dense(&dense.view(), 0.0);
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
/// Oracle: same matrix as `sample_a` -> `.toarray() == [[1,0,2],[0,3,0],[4,0,5]]`.
#[test]
fn csr_from_coo_to_dense_matches_scipy() {
    let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
    coo.push(0, 0, 1.0).unwrap();
    coo.push(0, 2, 2.0).unwrap();
    coo.push(1, 1, 3.0).unwrap();
    coo.push(2, 0, 4.0).unwrap();
    coo.push(2, 2, 5.0).unwrap();
    let csr = CsrMatrix::from_coo(&coo).unwrap();
    assert_eq!(csr.nnz(), 5);
    let expected = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
    assert_dense_eq(&csr.to_dense(), &expected);
}

/// REQ-CONSTRUCT-CONVERT. `to_csc()` materialized to dense round-trips the
/// canonical matrix, matching scipy `.tocsc().toarray()`.
///
/// Oracle: `A.tocsc().toarray() == [[1,0,2],[0,3,0],[4,0,5]]`.
#[test]
fn csr_to_csc_roundtrip_matches_scipy() {
    let a = sample_a();
    let csc = a.to_csc();
    let d = csc.to_dense();
    let expected = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "tocsc mismatch at ({r},{c})");
        }
    }
}

/// REQ-MATVEC. `mul_vec` computes `A @ v`, matching scipy `_matmul_vector`.
///
/// Oracle: `A @ [1,2,3] == [7.0, 6.0, 19.0]`.
#[test]
fn csr_mul_vec_matches_scipy() {
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
fn csr_add_self_matches_scipy() {
    let a = sample_a();
    let sum = a.add(&a).unwrap();
    let expected = [[2.0, 0.0, 4.0], [0.0, 6.0, 0.0], [8.0, 0.0, 10.0]];
    assert_dense_eq(&sum.to_dense(), &expected);
}

/// REQ-ADD. `add(&B)` matches scipy `A + B` (elementwise) for the helper B.
///
/// Oracle: B = [[1,1,0],[0,1,1],[0,0,1]];
/// `(A+B).toarray() == [[2,1,2],[0,4,1],[4,0,6]]`.
#[test]
fn csr_add_other_matches_scipy() {
    let a = sample_a();
    let b = sample_b();
    let sum = a.add(&b).unwrap();
    let expected = [[2.0, 1.0, 2.0], [0.0, 4.0, 1.0], [4.0, 0.0, 6.0]];
    assert_dense_eq(&sum.to_dense(), &expected);
}

/// REQ-MISSING-ELEMENTWISE. `multiply(&B)` matches scipy `A.multiply(B)`
/// (element-wise Hadamard product, INTERSECTION sparsity).
///
/// Oracle: B = [[1,1,0],[0,1,1],[0,0,1]];
/// `A.multiply(B).toarray() == [[1,0,0],[0,3,0],[0,0,5]]`.
#[test]
fn csr_multiply_matches_scipy() {
    let a = sample_a();
    let b = sample_b();
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
fn csr_sub_matches_scipy() {
    let a = sample_a();
    let b = sample_b();
    let diff = a.sub(&b).unwrap();
    let expected = [[0.0, -1.0, 2.0], [0.0, 2.0, -1.0], [4.0, 0.0, 4.0]];
    assert_dense_eq(&diff.to_dense(), &expected);
}

/// REQ-MISSING-ELEMENTWISE / REQ-ERR. `multiply` with an incompatible shape
/// returns `Err`, where scipy raises `ValueError: inconsistent shapes`.
#[test]
fn csr_multiply_shape_mismatch_is_err() {
    let a = sample_a();
    // empty 2x3 matrix (shape (2,3) vs A's (3,3))
    let c = CsrMatrix::<f64>::new(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
    assert!(
        a.multiply(&c).is_err(),
        "shape-mismatched multiply must return Err (scipy raises ValueError)"
    );
}

/// REQ-MISSING-ELEMENTWISE / REQ-ERR. `sub` with an incompatible shape returns
/// `Err`, where scipy raises `ValueError: inconsistent shapes`.
#[test]
fn csr_sub_shape_mismatch_is_err() {
    let a = sample_a();
    // empty 2x3 matrix (shape (2,3) vs A's (3,3))
    let c = CsrMatrix::<f64>::new(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
    assert!(
        a.sub(&c).is_err(),
        "shape-mismatched sub must return Err (scipy raises ValueError)"
    );
}

/// REQ-MISSING-MATMUL. `matmul(&B)` matches scipy `A @ B` (sparse-sparse matrix
/// product, `_matmul_sparse`, SMMP).
///
/// Oracle: B = [[1,1,0],[0,1,1],[0,0,1]];
/// `(A@B).toarray() == [[1,1,2],[0,3,3],[4,4,5]]`.
#[test]
fn csr_matmul_matches_scipy() {
    let a = sample_a();
    let b = sample_b();
    let prod = a.matmul(&b).unwrap();
    let expected = [[1.0, 1.0, 2.0], [0.0, 3.0, 3.0], [4.0, 4.0, 5.0]];
    assert_dense_eq(&prod.to_dense(), &expected);
}

/// REQ-MISSING-MATMUL. `matmul(&C)` with a non-square right operand
/// `C = [[1,2],[3,4],[5,6]]` (3x2) yields the `(3,2)` product, matching scipy
/// `A @ C`.
///
/// Oracle: `(A@C).toarray() == [[11,14],[9,12],[29,38]]`, shape `(3,2)`.
#[test]
fn csr_matmul_non_square() {
    let a = sample_a();
    let c = CsrMatrix::new(
        3,
        2,
        vec![0, 2, 4, 6],
        vec![0, 1, 0, 1, 0, 1],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
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
/// (`A.n_cols != D.n_rows`) returns `Err`, where scipy raises
/// `ValueError: dimension mismatch`.
#[test]
fn csr_matmul_shape_mismatch_is_err() {
    let a = sample_a();
    // D is 2x2: A has 3 cols, D has 2 rows -> inner dims disagree.
    let d = CsrMatrix::<f64>::new(2, 2, vec![0, 0, 0], vec![], vec![]).unwrap();
    assert!(
        a.matmul(&d).is_err(),
        "inner-dimension-mismatched matmul must return Err (scipy raises ValueError)"
    );
}

/// REQ-SCALAR-MUL. `mul_scalar(2.0)` (new) and `scale(2.0)` (in place) match
/// scipy `A * 2`.
///
/// Oracle: `(A*2).toarray() == [[2,0,4],[0,6,0],[8,0,10]]`.
#[test]
fn csr_scalar_mul_matches_scipy() {
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

/// REQ-ROW-SLICE. `row_slice(0,2)` matches scipy `A[0:2]`.
///
/// Oracle: `A[0:2].toarray() == [[1,0,2],[0,3,0]]`.
#[test]
fn csr_row_slice_matches_scipy() {
    let a = sample_a();
    let sliced = a.row_slice(0, 2).unwrap();
    assert_eq!(sliced.n_rows(), 2);
    assert_eq!(sliced.n_cols(), 3);
    let d = sliced.to_dense();
    let expected = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "row_slice mismatch at ({r},{c})");
        }
    }
}

/// REQ-ERR. `add` with an incompatible shape returns `Err`, where scipy raises
/// `ValueError: inconsistent shapes`.
///
/// Oracle: `A + sp.csr_matrix((2,3))` -> `ValueError: inconsistent shapes`.
#[test]
fn csr_add_shape_mismatch_is_err() {
    let a = sample_a();
    // empty 2x3 matrix (shape (2,3) vs A's (3,3))
    let c = CsrMatrix::<f64>::new(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
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
fn csr_mul_vec_shape_mismatch_is_err() {
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
fn csr_geometry_matches_scipy() {
    let a = sample_a();
    assert_eq!((a.n_rows(), a.n_cols()), (3, 3));
    assert_eq!(a.nnz(), 5);
    // Dataset arm: sparse split reports is_sparse()==true.
    assert_eq!(a.n_samples(), 3);
    assert_eq!(a.n_features(), 3);
    assert!(a.is_sparse());
}

/// REQ-API-ACCESSORS. The first-class `shape()`/`data()`/`indices()`/`indptr()`
/// accessors expose the same geometry + CSR `(data, indices, indptr)` triple
/// scipy exposes as `.shape`/`.data`/`.indices`/`.indptr`.
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]));
///   print(A.shape, A.data.tolist(), A.indices.tolist(), A.indptr.tolist())"`):
/// `(3, 3) [1.0, 2.0, 3.0, 4.0, 5.0] [0, 2, 1, 0, 2] [0, 2, 3, 5]`.
#[test]
fn csr_shape_data_indices_indptr_match_scipy() {
    let a = sample_a();
    // scipy A.shape == (3, 3)
    assert_eq!(a.shape(), (3, 3));
    // scipy A.data == [1,2,3,4,5]
    assert_eq!(a.data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    // scipy A.indices == [0,2,1,0,2] (CSR column indices)
    assert_eq!(a.indices(), &[0, 2, 1, 0, 2]);
    // scipy A.indptr == [0,2,3,5] (row pointers, length n_rows+1)
    assert_eq!(a.indptr(), vec![0, 2, 3, 5]);
}

/// REQ-MISSING-INDEX (element access). `get(i, j)` returns the scalar `A[i, j]`
/// stored value, matching scipy `A[i, j]` (`IndexMixin.__getitem__` ->
/// `_get_intXint`, `_index.py:29`).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]));
///   print(A[1,1], A[0,0], A[0,2], A[2,0])"`): `3.0 1.0 2.0 4.0`.
#[test]
fn csr_get_element_matches_scipy() {
    let a = sample_a();
    assert_eq!(a.get(1, 1).unwrap(), 3.0);
    assert_eq!(a.get(0, 0).unwrap(), 1.0);
    assert_eq!(a.get(0, 2).unwrap(), 2.0);
    assert_eq!(a.get(2, 0).unwrap(), 4.0);
}

/// REQ-MISSING-INDEX (element access). `get(i, j)` at a structurally absent
/// position returns `0`, matching scipy `A[0, 1]`.
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A[0,1])"`):
/// `0.0`.
#[test]
fn csr_get_absent_is_zero() {
    let a = sample_a();
    assert_eq!(a.get(0, 1).unwrap(), 0.0);
}

/// REQ-MISSING-INDEX (rows). `getrow(i)` returns row `i` as a `(1, n_cols)` CSR,
/// matching scipy `A.getrow(i)` (`_matrix.py:110` -> `_getrow`, `_base.py:1116`,
/// "(1 x n) row vector").
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]));
///   print(A.getrow(0).shape, A.getrow(0).toarray().tolist(),
///         A.getrow(1).toarray().tolist())"`):
/// `(1, 3) [[1.0, 0.0, 2.0]] [[0.0, 3.0, 0.0]]`.
#[test]
fn csr_getrow_matches_scipy() {
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

/// REQ-MISSING-INDEX (cols). `getcol(j)` returns column `j` as a `(n_rows, 1)`
/// CSR, matching scipy `A.getcol(j)` (`_matrix.py:104` -> `_getcol`,
/// `_base.py:1097`, "(m x 1) column vector").
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]]));
///   print(A.getcol(0).shape, A.getcol(0).toarray().tolist(),
///         A.getcol(2).toarray().tolist())"`):
/// `(3, 1) [[1.0], [0.0], [4.0]] [[2.0], [0.0], [5.0]]`.
#[test]
fn csr_getcol_matches_scipy() {
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

/// REQ-MISSING-INDEX (rows/cols) / REQ-ERR. An out-of-bounds row/column index to
/// `getrow`/`getcol` returns `Err(InvalidParameter)`, where scipy raises
/// `IndexError("index out of bounds")` (`_base.py:1110`/`:1129`).
#[test]
fn csr_getrow_getcol_out_of_bounds_is_err() {
    let a = sample_a();
    assert!(matches!(
        a.getrow(3),
        Err(ferrolearn_core::FerroError::InvalidParameter { .. })
    ));
    assert!(matches!(
        a.getcol(3),
        Err(ferrolearn_core::FerroError::InvalidParameter { .. })
    ));
}

/// REQ-MISSING-INDEX (element access) / REQ-ERR. An out-of-bounds index returns
/// `Err(InvalidParameter)`, where scipy raises `IndexError: index (...) out of
/// range` (`_index.py:388`).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); A[3,0]"`):
/// `IndexError: index (3) out of range`.
#[test]
fn csr_get_out_of_bounds_is_err() {
    let a = sample_a();
    assert!(matches!(
        a.get(3, 0),
        Err(ferrolearn_core::FerroError::InvalidParameter { .. })
    ));
    assert!(matches!(
        a.get(0, 3),
        Err(ferrolearn_core::FerroError::InvalidParameter { .. })
    ));
}

use ferrolearn_core::FerroError;

/// REQ-MISSING-INDEX (max/min). `max()`/`min()` fold an implicit zero when the
/// matrix is not fully dense (`nnz < n_rows*n_cols`), mirroring scipy
/// `_minmax_mixin._min_or_max(axis=None)` (`scipy/sparse/_data.py:208`-`:224`).
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   A=sp.csr_matrix(np.array([[-3.,0,0],[0,-1,0],[0,0,-5]]));
///   print(A.max(), A.min())"`): `0.0 -5.0` — the all-negative diagonal makes the
/// implicit zero the maximum.
#[test]
fn csr_max_folds_implicit_zero() -> Result<(), FerroError> {
    // A = diag(-3, -1, -5), 3x3, nnz 3 < 9.
    let a = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![-3.0_f64, -1.0, -5.0],
    )?;
    assert_eq!(a.max(), 0.0);
    assert_eq!(a.min(), -5.0);
    Ok(())
}

/// REQ-MISSING-INDEX (max/min). Symmetric to the negative-diagonal case: an
/// all-positive diagonal makes the implicit zero the minimum.
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   B=sp.csr_matrix(np.array([[3.,0,0],[0,1,0],[0,0,5]]));
///   print(B.max(), B.min())"`): `5.0 0.0`.
#[test]
fn csr_min_folds_implicit_zero() -> Result<(), FerroError> {
    // B = diag(3, 1, 5), 3x3, nnz 3 < 9.
    let b = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![3.0_f64, 1.0, 5.0],
    )?;
    assert_eq!(b.max(), 5.0);
    assert_eq!(b.min(), 0.0);
    Ok(())
}

/// REQ-MISSING-INDEX (max/min). A fully dense matrix (`nnz == n_rows*n_cols`)
/// has no implicit zero to fold, so `max()`/`min()` are pure stored-value
/// reductions.
///
/// Oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
///   C=sp.csr_matrix(np.array([[2.,7]])); print(C.max(), C.min())"`): `7.0 2.0`.
#[test]
fn csr_max_min_dense_no_implicit_zero() -> Result<(), FerroError> {
    // C = [[2, 7]], 1x2, fully dense (nnz 2 == 2).
    let c = CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![2.0_f64, 7.0])?;
    assert_eq!(c.max(), 7.0);
    assert_eq!(c.min(), 2.0);
    Ok(())
}
