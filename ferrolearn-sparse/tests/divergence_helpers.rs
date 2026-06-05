//! Divergence / conformance tests for `ferrolearn-sparse/src/helpers.rs`
//! (the `scipy.sparse` construction helpers `eye`/`diags`/`hstack`/`vstack`)
//! vs the LIVE scipy 1.17.1 oracle. Crosslink translation unit #2015.
//!
//! All expected values are computed by a live scipy call run from `/tmp`
//! (R-CHAR-3 — NEVER copied from the ferrolearn side). The oracle command and
//! its output are quoted below so the target is unambiguous.
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np, scipy.sparse as sp
//! print('eye3', sp.eye(3).toarray().tolist())
//! print('diags_main', sp.diags([1.,2.,3.],0,shape=(3,3)).toarray().tolist())
//! print('diags_super1', sp.diags([1.,2.],1,shape=(3,3)).toarray().tolist())
//! print('diags_sub-1', sp.diags([1.,2.],-1,shape=(3,3)).toarray().tolist())
//! a=sp.eye(2,format='csr'); b=sp.diags([5.,5.],0,shape=(2,2),format='csr')
//! print('hstack', sp.hstack([a,b]).toarray().tolist())
//! print('vstack', sp.vstack([a,b]).toarray().tolist())
//! print('toolong', sp.diags([1.,2.,3.],1,shape=(3,3)).toarray().tolist())
//! try: sp.diags([1.,2.],0,shape=(3,3))
//! except Exception as e: print('tooshort ERR', type(e).__name__)"
//! ```
//! prints (scipy 1.17.1):
//! ```text
//! eye3 [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
//! diags_main [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
//! diags_super1 [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
//! diags_sub-1 [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
//! hstack [[1.0, 0.0, 5.0, 0.0], [0.0, 1.0, 0.0, 5.0]]
//! vstack [[1.0, 0.0], [0.0, 1.0], [5.0, 0.0], [0.0, 5.0]]
//! toolong [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
//! tooshort ERR ValueError
//! ```
//!
//! The eye / diags-single / hstack / vstack SHIPPED core matches the oracle and
//! these are GREEN guards locking that agreement against regression. One RED pin
//! is present (`diags_too_short_must_error_like_scipy`): scipy raises
//! `ValueError` on a too-SHORT diagonal where ferrolearn silently returns `Ok`
//! — a single-file-fixable length-validation divergence
//! (`scipy/sparse/_construct.py:435-439`). The structural NOT-STARTED REQs
//! (multi-diagonal, missing helpers, rectangular/offset eye, format/mixed
//! hstack/vstack, consumer, ferray substrate) are filed as `-l blocker` issues,
//! not pinned here as doomed tests (R-DEFER-3).

use ferrolearn_sparse::{CsrMatrix, diags, eye, hstack, vstack};

// ---------------------------------------------------------------------------
// GREEN guards — the SHIPPED helpers match the live scipy 1.17.1 oracle.
// ---------------------------------------------------------------------------

/// GREEN. REQ-EYE. `eye::<f64>(3)` builds the 3x3 identity, matching scipy
/// `eye(3)` (`scipy/sparse/_construct.py:678`).
///
/// Oracle: `sp.eye(3).toarray()` == `[[1,0,0],[0,1,0],[0,0,1]]`.
#[test]
fn eye_3_matches_scipy_identity() {
    let m: CsrMatrix<f64> = eye(3).unwrap();
    let d = m.to_dense();
    let expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "eye(3) mismatch at ({r},{c})");
        }
    }
}

/// GREEN. REQ-DIAGS-SINGLE (main diagonal). `diags(&[1,2,3], 0, 3)` matches
/// scipy `diags([1,2,3], 0, shape=(3,3))` (`scipy/sparse/_construct.py:445`).
///
/// Oracle: `[[1,0,0],[0,2,0],[0,0,3]]`.
#[test]
fn diags_main_matches_scipy() {
    let m: CsrMatrix<f64> = diags(&[1.0, 2.0, 3.0], 0, 3).unwrap();
    let d = m.to_dense();
    let expected = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "diags main mismatch at ({r},{c})");
        }
    }
}

/// GREEN. REQ-DIAGS-SINGLE (super-diagonal, offset +1). `diags(&[1,2], 1, 3)`
/// matches scipy `diags([1,2], 1, shape=(3,3))`.
///
/// Oracle: `[[0,1,0],[0,0,2],[0,0,0]]` — entries at (0,1)=1, (1,2)=2.
#[test]
fn diags_super_offset1_matches_scipy() {
    let m: CsrMatrix<f64> = diags(&[1.0, 2.0], 1, 3).unwrap();
    let d = m.to_dense();
    let expected = [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "diags super mismatch at ({r},{c})");
        }
    }
}

/// GREEN. REQ-DIAGS-SINGLE (sub-diagonal, offset -1). `diags(&[1,2], -1, 3)`
/// matches scipy `diags([1,2], -1, shape=(3,3))`. The design doc flagged the
/// sub-diagonal alignment lacked a dedicated test — this adds it.
///
/// Oracle: `[[0,0,0],[1,0,0],[0,2,0]]` — entries at (1,0)=1, (2,1)=2.
#[test]
fn diags_sub_offset_neg1_matches_scipy() {
    let m: CsrMatrix<f64> = diags(&[1.0, 2.0], -1, 3).unwrap();
    let d = m.to_dense();
    let expected = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "diags sub mismatch at ({r},{c})");
        }
    }
}

/// GREEN. REQ-DIAGS-LENGTH-VALIDATION (too-LONG agreement). scipy SILENTLY
/// TRUNCATES a too-long diagonal (`scipy/sparse/_construct.py:433`
/// `data_arr[j, k:k+length] = diagonal[...,:length]`); ferrolearn silently skips
/// the out-of-bounds entry (the `if i < n && j < n` guard). Both produce the
/// 2-entry super-diagonal — this locks the agreement on the over-long case.
///
/// Oracle: `sp.diags([1.,2.,3.],1,shape=(3,3)).toarray()` ==
/// `[[0,1,0],[0,0,2],[0,0,0]]` (third value dropped, no error).
#[test]
fn diags_too_long_truncates_like_scipy() {
    // 3 values for a len-2 super-diagonal; scipy truncates to 2, no error.
    let m: CsrMatrix<f64> = diags(&[1.0, 2.0, 3.0], 1, 3).unwrap();
    let d = m.to_dense();
    let expected = [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "diags too-long mismatch at ({r},{c})");
        }
    }
}

/// GREEN. REQ-HSTACK. `hstack(&[&a, &b])` matches scipy `hstack([a, b])`
/// (`scipy/sparse/_construct.py:1012`).
///
/// Oracle: a=eye(2), b=diags([5,5],0,2);
/// `hstack([a,b]).toarray()` == `[[1,0,5,0],[0,1,0,5]]`.
#[test]
fn hstack_matches_scipy() {
    let a: CsrMatrix<f64> = eye(2).unwrap();
    let b: CsrMatrix<f64> = diags(&[5.0, 5.0], 0, 2).unwrap();
    let h = hstack(&[&a, &b]).unwrap();
    assert_eq!(h.n_rows(), 2);
    assert_eq!(h.n_cols(), 4);
    let d = h.to_dense();
    let expected = [[1.0, 0.0, 5.0, 0.0], [0.0, 1.0, 0.0, 5.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "hstack mismatch at ({r},{c})");
        }
    }
}

/// GREEN. REQ-VSTACK. `vstack(&[&a, &b])` matches scipy `vstack([a, b])`
/// (`scipy/sparse/_construct.py:1059`).
///
/// Oracle: a=eye(2), b=diags([5,5],0,2);
/// `vstack([a,b]).toarray()` == `[[1,0],[0,1],[5,0],[0,5]]`.
#[test]
fn vstack_matches_scipy() {
    let a: CsrMatrix<f64> = eye(2).unwrap();
    let b: CsrMatrix<f64> = diags(&[5.0, 5.0], 0, 2).unwrap();
    let v = vstack(&[&a, &b]).unwrap();
    assert_eq!(v.n_rows(), 4);
    assert_eq!(v.n_cols(), 2);
    let d = v.to_dense();
    let expected = [[1.0, 0.0], [0.0, 1.0], [5.0, 0.0], [0.0, 5.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(d[[r, c]], v, "vstack mismatch at ({r},{c})");
        }
    }
}

/// GREEN. REQ-HSTACK error path. Mismatched-row blocks → `Err`, mirroring scipy
/// `hstack` raising `ValueError` on inconsistent row counts.
///
/// Oracle: `sp.hstack([sp.eye(2), sp.eye(3)])` -> ValueError.
#[test]
fn hstack_row_mismatch_is_err() {
    let a: CsrMatrix<f64> = eye(2).unwrap();
    let b: CsrMatrix<f64> = eye(3).unwrap();
    assert!(
        hstack(&[&a, &b]).is_err(),
        "hstack of mismatched rows must return Err (scipy raises ValueError)"
    );
}

/// GREEN. REQ-VSTACK error path. Mismatched-column blocks → `Err`, mirroring
/// scipy `vstack` raising `ValueError` on inconsistent column counts.
///
/// Oracle: `sp.vstack([sp.eye(2), sp.eye(3)])` -> ValueError.
#[test]
fn vstack_col_mismatch_is_err() {
    let a: CsrMatrix<f64> = eye(2).unwrap();
    let b: CsrMatrix<f64> = eye(3).unwrap();
    assert!(
        vstack(&[&a, &b]).is_err(),
        "vstack of mismatched cols must return Err (scipy raises ValueError)"
    );
}

// ---------------------------------------------------------------------------
// RED pin — diags length validation (too-SHORT diagonal).
// ---------------------------------------------------------------------------

/// RED PIN. REQ-DIAGS-LENGTH-VALIDATION (the headline divergence).
///
/// scipy raises `ValueError` when a supplied diagonal is too SHORT for its slot:
/// `scipy/sparse/_construct.py:435-439`
/// ```text
///   if len(diagonal) != length and len(diagonal) != 1:
///       raise ValueError(
///           f"Diagonal length (index {j}: {len(diagonal)} at"
///           f" offset {offset}) does not agree with array size ({m}, {n}).")
/// ```
/// where the required `length = min(m + offset, n - offset, min(m, n))`
/// (`_construct.py:429`); for an `n x n` grid at offset 0 that is `n`.
///
/// Live oracle (R-CHAR-3, run from /tmp, value NEVER copied from ferrolearn):
/// `python3 -c "import scipy.sparse as sp; sp.diags([1.,2.],0,shape=(3,3))"`
/// -> `ValueError: Diagonal length (index 0: 2 at offset 0) does not agree with
/// array size (3, 3).`
///
/// ferrolearn `diags(&[1.,2.], 0, 3)` instead silently lays a 2-entry main
/// diagonal and returns `Ok` (the `if i < n && j < n` guard accepts a short
/// diagonal — `helpers.rs` `pub fn diags`). This assertion requires it to return
/// `Err` (matching scipy's `ValueError`); it FAILS today.
///
/// Single-file fix (helpers.rs only): before the push loop, compute the required
/// length `expected = n - offset.unsigned_abs()` and, when
/// `values.len() != expected && values.len() != 1`, return
/// `Err(FerroError::InvalidParameter { .. })`. This must error ONLY on the
/// too-SHORT case — the too-LONG case must still truncate (locked green by
/// `diags_too_long_truncates_like_scipy`).
///
/// Tracking: see crosslink blocker filed for unit #2015.
#[test]
fn diags_too_short_must_error_like_scipy() {
    // 2 values for a len-3 main diagonal on a 3x3 grid: scipy -> ValueError.
    let result: Result<CsrMatrix<f64>, _> = diags(&[1.0, 2.0], 0, 3);
    assert!(
        result.is_err(),
        "diags(&[1,2], 0, 3): scipy raises ValueError (Diagonal length \
         (index 0: 2 at offset 0) does not agree with array size (3, 3)), \
         but ferrolearn returned Ok — silent short diagonal"
    );
}
