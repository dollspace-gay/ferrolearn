# scipy.sparse.csc_matrix — Compressed Sparse Column format

<!--
tier: 3-component
status: draft
baseline-commit: b4b6d5bfa
upstream-paths:
  - scipy/sparse/_csc.py          # _csc_base / csc_matrix / csc_array
  - scipy/sparse/_compressed.py   # _cs_matrix base (CSR/CSC shared impl)
-->

## Summary

`ferrolearn-sparse/src/csc.rs` provides `CscMatrix<T>`, a newtype over
`sprs::CsMat<T>` (CSC storage), mirroring a slice of **scipy.sparse.csc_matrix**
(Compressed Sparse Column). CSC is the column-symmetric analog of `CsrMatrix`
(just audited): the same `_cs_matrix` (`_compressed.py:25`) base, but with the
roles of rows and columns swapped — `indptr` is the column pointer, `indices`
holds row indices, and slicing/efficient access is column-oriented. The live
oracle is the installed **scipy 1.17.1**; CSC construction / conversion /
arithmetic / matvec is deterministic, so the scipy-1.17.1 / sklearn-1.5.2
version split is irrelevant (`A@v` is `A@v` in every scipy release).

What `CscMatrix` ships is the construction-conversion-and-arithmetic core, 1:1
with `CsrMatrix` but column-symmetric: `new`/`from_coo`/`from_csr`/`from_dense`
build it; `to_dense`/`to_coo`/`to_csr` materialize/convert it; `mul_vec`/`add`/
`mul_scalar`/`scale` do matvec, elementwise-add, and scalar multiply;
`col_slice` extracts a contiguous **column** range (the CSC analog of CSR's
`row_slice`); `n_rows`/`n_cols`/`nnz` report geometry; `inner`/`into_inner` are
the `sprs::CsMat` escape hatch. What diverges is the *column-slice API shape*,
the *accessor surface*, the large set of `csc_matrix` *methods* (sparse-sparse
matmul, transpose, sum, diagonal, elementwise multiply/subtract, indexing) that
have no ferrolearn analog, and — unlike `CsrMatrix` — the *consumer strength*:
`CscMatrix` has NO cross-crate estimator consumer; its only non-test production
consumer is the in-crate `csr.rs` CSR↔CSC round-trip.

Divergence classes:
1. **construction + toarray + nnz + format conversion (the SHIPPED core)** —
   `from_coo`/`from_dense`/`from_csr` + `to_dense`/`to_coo`/`to_csr` mirror
   `csc_matrix(...)` + `.toarray()`/`.tocsr()`/`.tocoo()`; `nnz()` mirrors scipy
   `.nnz` (CSC coalesces duplicates on construction, so it counts distinct
   stored entries).
2. **arithmetic / matvec parity (SHIPPED)** — `mul_vec(v)` ≡ scipy `A@v`,
   `add(B)` ≡ scipy `A+B` (elementwise), `mul_scalar(s)`/`scale(s)` ≡ scipy
   `A*s`.
3. **column-slice API divergence (R-DEV-3)** — ferrolearn `col_slice(start,end)`
   (a fallible method returning a new `CscMatrix`) vs scipy Python slicing
   `A[:,start:end]`. Same values, different API shape. CSC slices COLUMNS — the
   natural analog of CSR's row-slicing, because the outer dimension of CSC is
   the column.
4. **missing csc_matrix methods (NOT-STARTED)** — sparse-sparse matmul
   `A@B`/`.dot(B)`, `.transpose()`/`.T`, `.multiply(B)` (elementwise sparse),
   elementwise subtract `A-B`, `.sum(axis=)`, `.diagonal()`, element/row/column
   indexing `A[i,j]`/`A[i,:]`/`A[:,j]`/`.getrow()`/`.getcol()`,
   `.eliminate_zeros()`, `.sort_indices()`, `.sum_duplicates()`, `.power()`,
   `.max()`/`.min()`/`.argmax()`, `.astype()`, `.copy()` have no `CscMatrix`
   method.
5. **API accessors (R-DEV-2/3)** — scipy exposes `.shape` (tuple) and
   `.data`/`.indices`/`.indptr` (the CSC arrays) directly; ferrolearn exposes
   `n_rows()`/`n_cols()`/`nnz()` and gates the raw arrays behind
   `inner()`/`into_inner()` (no public `.shape`/`.data`/`.indices`/`.indptr`).
6. **error handling** — `add`/`mul_vec` return `Result<_, FerroError>` on shape
   mismatch; scipy raises `ValueError`. Validation timing matches.
7. **consumer (SHIPPED but weak)** — unlike `CsrMatrix` (which has the
   `ferrolearn-neighbors` k-NN graph estimator consumer + `impl Dataset`),
   `CscMatrix` has NO cross-crate estimator consumer. Its only non-test
   production consumer is the in-crate CSR↔CSC conversion in `csr.rs`
   (`CsrMatrix::from_csc`/`to_csc`) plus the `lib.rs` re-export. This is a
   genuine non-test consumer (the round-trip is reachable from any CSR user),
   but materially weaker than CSR's estimator-adjacent consumer — stated
   honestly per R-HONEST-3.
8. **sprs substrate (R-SUBSTRATE-1)** — `CscMatrix` wraps `sprs::CsMat` and
   materializes to `ndarray::Array2`; the destination is ferray's
   `scipy.sparse` CSC analog + `ferray-core`, not `sprs`/`ndarray`.

## Upstream reference (scipy.sparse.csc_matrix, live oracle scipy 1.17.1)

`csc_matrix(_csc_base)` (`_csc.py:17`, `class _csc_base(_cs_matrix)`) inherits
its storage and most methods from `_cs_matrix(_data_matrix, _minmax_mixin,
IndexMixin)` (`_compressed.py:25`), the shared CSR/CSC base — the SAME base
`CsrMatrix` mirrors, with the row/column roles swapped (`self._swap`). Cite the
**method/attribute names** and the **live-oracle values**, not internal `.pyx`
helper lines. Relevant surface (line numbers stable at scipy 1.17.1):

- attributes: `.shape` (tuple, `self._shape`, `_compressed.py:38`),
  `.data` / `.indices` / `.indptr` (the three CSC arrays — for CSC, `indptr` is
  the column pointer of length `n_cols+1` and `indices` holds row indices, set
  at `_compressed.py:76-78`), `.nnz`.
- `_getnnz(axis=None)` (`_compressed.py:118`) returns stored entries; CSC
  construction coalesces duplicates, so this counts DISTINCT stored positions.
- `toarray` (`_compressed.py:1002`), `tocsr` (`_csc.py:44`, `def tocsr`),
  `tocoo`, `transpose`/`T` (`_csc.py:20`, `def transpose` — CSC.T is a CSR view
  of the same buffers), `diagonal` (`_compressed.py:476`),
  `sum`/`sum(axis=)` (`_compressed.py:492`),
  `_matmul_vector` (`_compressed.py:387`, the `A@v` path),
  `_matmul_sparse` (`_compressed.py:415`, the `A@B` path),
  `_add_sparse` (`_compressed.py:257`), `_sub_sparse` (`_compressed.py:260`),
  `multiply` (`_base.py:490`, elementwise),
  `eliminate_zeros` (`_compressed.py:1025`),
  `sum_duplicates` (`_compressed.py:1063`),
  `sort_indices` (`_compressed.py:1110`), `__getitem__` (`IndexMixin`,
  `_index.py:29`), `power`/`astype`/`copy` (`_data.py:99`/`:69`/`:94`),
  `_mul_scalar` (`_data.py:134`), `max`/`min`/`argmax` (`_minmax_mixin`).

Live oracle (`cd /tmp && python3 -c "..."`, scipy 1.17.1). Canonical matrix
`A = [[1,0,2],[0,3,0],[4,0,5]]`, helper `B = [[1,1,0],[0,1,1],[0,0,1]]`, both
built with `sp.csc_matrix(...)`:

```
A.nnz            -> 5
A.shape          -> (3, 3)
A.format         -> 'csc'
A.data/.indices/.indptr -> [1,4,3,2,5] / [0,2,1,0,2] / [0,2,3,5]
A.toarray()      -> [[1,0,2],[0,3,0],[4,0,5]]
A @ [1,2,3]      -> [7.0, 6.0, 19.0]                 # matvec
(A + A).toarray()-> [[2,0,4],[0,6,0],[8,0,10]]       # elementwise add
(A * 2).toarray()-> [[2,0,4],[0,6,0],[8,0,10]]       # scalar mul
A[:,0:2].toarray()  -> [[1,0],[0,3],[4,0]]           # COLUMN slice
A.tocsr().toarray() -> [[1,0,2],[0,3,0],[4,0,5]]     # round-trips
(A + B).toarray()-> [[2,1,2],[0,4,1],[4,0,6]]        # elementwise add
(A @ B).toarray()-> [[1,1,2],[0,3,3],[4,4,5]]        # sparse-sparse matmul
A.T.toarray()    -> [[1,0,4],[0,3,0],[2,0,5]]        # transpose
A.sum(axis=0)    -> [[5,3,7]]   A.diagonal() -> [1,3,5]
A.multiply(B).toarray() -> [[1,0,0],[0,3,0],[0,0,5]] # elementwise mul
```

Note the CSC storage arrays differ from CSR: CSC stores column-by-column, so
`A.data = [1,4,3,2,5]` (col 0: rows 0,2 → 1,4; col 1: row 1 → 3; col 2: rows
0,2 → 2,5), exactly the `sample_csc()` fixture in `csc.rs`.

scipy validation (oracle):

```
A @ np.array([1.,2.]) (len 2, A has 3 cols) -> ValueError: dimension mismatch
A + sp.csc_matrix((2,3))   (shape (2,3) vs (3,3)) -> ValueError: inconsistent shapes
```

## Requirements

- REQ-CONSTRUCT-CONVERT: `CscMatrix::from_coo`/`from_csr`/`from_dense`/`new`
  construct a CSC matrix, and `to_dense`/`to_coo`/`to_csr` materialize/convert
  it — mirroring scipy `csc_matrix(...)` + `.toarray()`/`.tocoo()`/`.tocsr()`.
  `to_dense()` equals scipy `.toarray()`; `to_csr()` round-trips; `nnz()` equals
  scipy `.nnz` (CSC coalesces duplicates, so it counts distinct stored entries).
  Oracle: `A.toarray()` = `[[1,0,2],[0,3,0],[4,0,5]]`, `A.nnz` = 5,
  `A.tocsr().toarray()` round-trips.
- REQ-MATVEC: `mul_vec(v)` computes the sparse-matrix × dense-vector product
  `A @ v`, matching scipy's `_matmul_vector` (`_compressed.py:387`). Oracle:
  `A @ [1,2,3]` = `[7,6,19]`.
- REQ-ADD: `add(B)` computes the elementwise sum of two same-shape CSC matrices,
  matching scipy `A + B` (`_add_sparse`, `_compressed.py:257`). Oracle:
  `(A+A).toarray()` = `[[2,0,4],[0,6,0],[8,0,10]]`, `(A+B).toarray()` =
  `[[2,1,2],[0,4,1],[4,0,6]]`.
- REQ-SCALAR-MUL: `mul_scalar(s)` (new matrix) and `scale(s)` (in place) scale
  every stored entry by `s`, matching scipy `A * s` (`_mul_scalar`,
  `_data.py:134`). Oracle: `(A*2).toarray()` = `[[2,0,4],[0,6,0],[8,0,10]]`.
- REQ-COL-SLICE: `col_slice(start,end)` returns a new `CscMatrix` of columns
  `start..end`, matching scipy Python slicing `A[:,start:end]`, with the API
  divergence that ferrolearn uses a fallible method (start/end validated) where
  scipy uses `__getitem__` slice syntax (`_index.py:29`). CSC slices COLUMNS
  (the outer dimension of CSC), the natural analog of CSR's `row_slice`. Oracle:
  `A[:,0:2].toarray()` = `[[1,0],[0,3],[4,0]]`.
- REQ-MISSING-MATMUL: sparse-sparse matmul `A@B`/`.dot(B)` (`_matmul_sparse`,
  `_compressed.py:415`) exists on `CscMatrix`. Oracle: `(A@B).toarray()` =
  `[[1,1,2],[0,3,3],[4,4,5]]`.
- REQ-MISSING-TRANSPOSE: `.transpose()`/`.T` (`_csc.py:20`) exists on
  `CscMatrix`. Oracle: `A.T.toarray()` = `[[1,0,4],[0,3,0],[2,0,5]]`.
- REQ-MISSING-REDUCE: `.sum(axis=)` (`_compressed.py:492`) and `.diagonal()`
  (`_compressed.py:476`) exist on `CscMatrix`. Oracle: `A.sum(axis=0)` =
  `[[5,3,7]]`, `A.diagonal()` = `[1,3,5]`.
- REQ-MISSING-ELEMENTWISE: elementwise sparse multiply `.multiply(B)`
  (`_base.py:490`) and subtract `A-B` (`_sub_sparse`, `_compressed.py:260`)
  exist on `CscMatrix`. Oracle: `A.multiply(B).toarray()` =
  `[[1,0,0],[0,3,0],[0,0,5]]`.
- REQ-MISSING-INDEX: element/row/column indexing `A[i,j]`/`A[i,:]`/`A[:,j]`,
  `.getrow()`/`.getcol()`, and the housekeeping `.eliminate_zeros()`/
  `.sort_indices()`/`.sum_duplicates()`/`.power()`/`.max()`/`.min()`/
  `.argmax()`/`.astype()`/`.copy()` methods exist on `CscMatrix`. Oracle:
  `A[1,1]` = 3.0, `A[:,0]` selects the first column.
- REQ-API-ACCESSORS: ferrolearn exposes the geometry/array surface scipy
  exposes — scipy has `.shape` (tuple) + `.data`/`.indices`/`.indptr`;
  ferrolearn has `n_rows()`/`n_cols()`/`nnz()` (no `.shape()` tuple) and gates
  the three CSC arrays behind `inner()`/`into_inner()` (no public
  `.data`/`.indices`/`.indptr`).
- REQ-ERR: `add`/`mul_vec` return `Result<_, FerroError>` with scipy-matching
  validation timing — shape mismatch rejected at the operation (scipy raises
  `ValueError`; ferrolearn returns `FerroError::ShapeMismatch`).
- REQ-CONSUMER: a non-test production caller consumes `CscMatrix` so it is part
  of the live translation surface. Unlike `CsrMatrix`, there is NO cross-crate
  estimator consumer; the consumer is the in-crate CSR↔CSC conversion in
  `csr.rs` (`CsrMatrix::from_csc`/`to_csc`) plus the `lib.rs` re-export.
- REQ-FERRAY: `CscMatrix` is backed by ferray's `scipy.sparse` CSC analog and
  `ferray-core` rather than `sprs::CsMat`/`ndarray::Array2` (R-SUBSTRATE-1).

## Acceptance criteria

All expected values come from the live scipy 1.17.1 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn. Canonical matrices
`A = [[1,0,2],[0,3,0],[4,0,5]]`, `B = [[1,1,0],[0,1,1],[0,0,1]]`. In ferrolearn,
`A` is `CscMatrix::new(3,3,vec![0,2,3,5],vec![0,2,1,0,2],vec![1.,4.,3.,2.,5.])`
(note the CSC value order `[1,4,3,2,5]` — column-major — vs CSR's `[1,2,3,4,5]`).

- AC-CONSTRUCT-CONVERT (REQ-CONSTRUCT-CONVERT):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.nnz, A.shape, A.toarray().tolist(), A.tocsr().toarray().tolist())"`
  → `5 (3, 3) [[1.0,0.0,2.0],[0.0,3.0,0.0],[4.0,0.0,5.0]] [[1.0,0.0,2.0],[0.0,3.0,0.0],[4.0,0.0,5.0]]`.
  `CscMatrix::to_dense()` equals `A.toarray()` element-wise (`test_to_dense`),
  `nnz()` = 5 (`test_new_valid`), `to_csr()` then `from_csr` round-trips
  (`test_csc_csr_roundtrip`), `from_coo`/`from_dense` round-trip
  (`test_from_coo_roundtrip`, `test_from_dense`).
- AC-MATVEC (REQ-MATVEC):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print((A@np.array([1.,2,3])).tolist())"`
  → `[7.0, 6.0, 19.0]`. `CscMatrix::mul_vec(&array![1.,2.,3.])` = `[7,6,19]`
  (`test_mul_vec`, `result[0]==7.0`, `result[1]==6.0`, `result[2]==19.0`).
- AC-ADD (REQ-ADD):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print((A+A).toarray().tolist())"`
  → `[[2.0,0.0,4.0],[0.0,6.0,0.0],[8.0,0.0,10.0]]`. `A.add(&A)?.to_dense()`
  matches (`test_add`, `d[[0,0]]==2.0`, `d[[1,1]]==6.0`).
- AC-SCALAR-MUL (REQ-SCALAR-MUL):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print((A*2).toarray().tolist())"`
  → `[[2.0,0.0,4.0],[0.0,6.0,0.0],[8.0,0.0,10.0]]`. `A.mul_scalar(2.0)`
  (`test_mul_scalar` `d[[0,0]]==2.0`, `d[[1,1]]==6.0`) and `A.scale(3.0)`
  (`test_scale_in_place` `d[[0,0]]==3.0`, `d[[2,2]]==15.0`) match.
- AC-COL-SLICE (REQ-COL-SLICE):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A[:,0:2].toarray().tolist())"`
  → `[[1.0,0.0],[0.0,3.0],[4.0,0.0]]`. `A.col_slice(0,2)?.to_dense()` matches
  (`test_col_slice`, `n_rows()==3`, `n_cols()==2`, `d[[0,0]]==1.0`,
  `d[[1,1]]==3.0`); out-of-range rejected (`test_col_slice_invalid`); empty
  slice `col_slice(1,1)` → `n_cols()==0` (`test_col_slice_empty`). API
  divergence (R-DEV-3): scipy uses `A[:,0:2]` slice syntax; ferrolearn uses the
  fallible `col_slice(0,2)` method. CSC slices COLUMNS (the outer dimension),
  the natural analog of CSR's `row_slice`.
- AC-MISSING-MATMUL (REQ-MISSING-MATMUL):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); B=sp.csc_matrix(np.array([[1.,1,0],[0,1,1],[0,0,1]])); print((A@B).toarray().tolist())"`
  → `[[1.0,1.0,2.0],[0.0,3.0,3.0],[4.0,4.0,5.0]]`. `grep -n "pub fn" csc.rs`
  shows `mul_vec` (matrix×VECTOR only) — no matrix×matrix. A critic pins a
  FAILING `A.matmul(&B)` test. FAILS until implemented.
- AC-MISSING-TRANSPOSE (REQ-MISSING-TRANSPOSE):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.T.toarray().tolist())"`
  → `[[1.0,0.0,4.0],[0.0,3.0,0.0],[2.0,0.0,5.0]]`. No `transpose`/`t` method on
  `CscMatrix` (`grep -n "pub fn" csc.rs`). A critic pins a FAILING
  `A.transpose()` test. FAILS until implemented.
- AC-MISSING-REDUCE (REQ-MISSING-REDUCE):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.sum(axis=0).tolist(), A.diagonal().tolist())"`
  → `[[5.0,3.0,7.0]] [1.0,3.0,5.0]`. No `sum`/`diagonal` method on `CscMatrix`.
  A critic pins FAILING `A.sum(0)`/`A.diagonal()` tests. FAILS until implemented.
- AC-MISSING-ELEMENTWISE (REQ-MISSING-ELEMENTWISE):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); B=sp.csc_matrix(np.array([[1.,1,0],[0,1,1],[0,0,1]])); print(A.multiply(B).toarray().tolist(), (A-B).toarray().tolist())"`
  → `[[1.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,5.0]] [[0.0,-1.0,2.0],[0.0,2.0,-1.0],[4.0,0.0,4.0]]`.
  `CscMatrix` has `add` but no `multiply` (elementwise) and no `sub`. A critic
  pins FAILING `A.multiply(&B)`/`A.sub(&B)` tests. FAILS until implemented.
- AC-MISSING-INDEX (REQ-MISSING-INDEX):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A[1,1], A[:,0].toarray().tolist())"`
  → `3.0 [[1.0],[0.0],[4.0]]`. `CscMatrix` has `col_slice` (contiguous range)
  but no `A[i,j]` scalar access, no single-column `getcol`, no row
  `getrow`/`A[i,:]`, and no `eliminate_zeros`/`sort_indices`/`sum_duplicates`/
  `power`/`max`/`min`/`argmax`/`astype`/`copy`. A critic pins a FAILING
  `A.get(1,1)`/`A.getcol(0)` test. FAILS until implemented.
- AC-API-ACCESSORS (REQ-API-ACCESSORS): scipy exposes `.shape` (tuple) and
  `.data`/`.indices`/`.indptr`:
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.shape, A.data.tolist(), A.indices.tolist(), A.indptr.tolist())"`
  → `(3, 3) [1.0,4.0,3.0,2.0,5.0] [0,2,1,0,2] [0,2,3,5]`. ferrolearn exposes
  `n_rows()`/`n_cols()`/`nnz()` and only the `inner()`/`into_inner()`
  `sprs::CsMat` handle — `grep -n "pub fn" csc.rs` shows no `shape`/`data`/
  `indices`/`indptr` accessor. A critic pins a FAILING test requiring a
  `.shape() -> (usize,usize)` tuple and public `.data()`/`.indices()`/
  `.indptr()` slices. FAILS until added.
- AC-ERR (REQ-ERR):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); A@np.array([1.,2.])"`
  → `ValueError: dimension mismatch`; and shape-mismatched add → `ValueError:
  inconsistent shapes`. `CscMatrix::mul_vec(&array![1.,2.])` returns
  `Err(FerroError::ShapeMismatch{context:"CscMatrix::mul_vec",...})`
  (`test_mul_vec_shape_mismatch`); `A.add(&m_2x3)` returns
  `Err(FerroError::ShapeMismatch{context:"CscMatrix::add",...})`
  (`test_add_shape_mismatch`). Error TYPE diverges (`FerroError` vs
  `ValueError`) but is the sanctioned crate contract (CLAUDE.md / R-CODE-2);
  `ValueError` marshalling is `ferrolearn-python`'s job.
- AC-CONSUMER (REQ-CONSUMER):
  `grep -rn "CscMatrix\|csc::" --include=*.rs /home/doll/ferrolearn/ferrolearn-*/src | grep -v ferrolearn-sparse/src | grep -v '#[cfg(test'`
  returns NOTHING — there is NO cross-crate estimator consumer of `CscMatrix`.
  `grep -rn "from_csc\|to_csc\|CscMatrix" --include=*.rs /home/doll/ferrolearn/ferrolearn-sparse/src/csr.rs | grep -v '#[cfg(test'`
  shows `CsrMatrix::from_csc` (`csc.inner().to_csr()`) and `CsrMatrix::to_csc`
  (`CscMatrix::from_inner(self.inner.to_csc())`) — the in-crate CSR↔CSC
  conversion that is the sole non-test production consumer; `lib.rs` re-exports
  `pub use csc::CscMatrix`. Stated honestly (R-HONEST-3): this is a genuine
  non-test consumer but weaker than CSR's estimator-adjacent consumer.
- AC-FERRAY (REQ-FERRAY): `csc.rs` imports `sprs::CsMat` and
  `ndarray::{Array1,Array2,ArrayView2}`; the destination is ferray's sparse CSC
  analog + `ferray-core` (R-SUBSTRATE-1). ferray does not yet expose a
  `scipy.sparse` layer (R-SUBSTRATE-5).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-CONSTRUCT-CONVERT (construct + toarray + nnz + format conversion) | SHIPPED | impl `pub fn from_coo`/`from_csr`/`from_dense`/`new` + `pub fn to_dense`/`to_coo`/`to_csr`/`nnz in csc.rs` (each wraps `CsMat::try_new_csc`/`to_csc`/`csc_from_dense`/`to_dense` + `CsrMatrix::from_csc`) mirror scipy `csc_matrix(...)` + `.toarray()` (`_compressed.py:1002`)/`.tocsr()` (`_csc.py:44`)/`.tocoo()` and `.nnz` (`_getnnz`, `_compressed.py:118`). Live oracle (R-CHAR-3): `A.nnz`=5, `A.shape`=`(3,3)`, `A.toarray()`=`[[1,0,2],[0,3,0],[4,0,5]]`, `A.tocsr().toarray()` round-trips; CSC coalesces duplicates so `nnz` counts distinct stored entries. Non-test consumer: `csr.rs`'s `CsrMatrix::from_csc`/`to_csc` round-trip builds via `CscMatrix::from_inner`/`inner()`. Verification: `cargo test -p ferrolearn-sparse --lib csc` (`test_new_valid`, `test_to_dense`, `test_csc_csr_roundtrip`, `test_from_coo_roundtrip`, `test_from_dense`) → green. |
| REQ-MATVEC (sparse×dense-vector `A@v`) | SHIPPED | impl `pub fn mul_vec in csc.rs` (`let result = &self.inner * rhs;` after a `rhs.len() != n_cols()` check) mirrors scipy's `_matmul_vector` (`_compressed.py:387`, the `A@v` path). Live oracle (R-CHAR-3): `A @ [1,2,3]` = `[7.0, 6.0, 19.0]`. Non-test consumer: `mul_vec` is a public CSC primitive over the crate's `CsMat` newtype, exercised on any `CscMatrix` reached via the `csr.rs` CSR↔CSC round-trip. Verification: `test_mul_vec` (`result[0]==7.0`, `result[1]==6.0`, `result[2]==19.0`), `test_mul_vec_shape_mismatch` → green. |
| REQ-ADD (elementwise `A+B`) | SHIPPED | impl `pub fn add in csc.rs` (`let result = &self.inner + &rhs.inner;` after a shape check) mirrors scipy `A+B` (`_add_sparse`, `_compressed.py:257`). Live oracle (R-CHAR-3): `(A+A).toarray()` = `[[2,0,4],[0,6,0],[8,0,10]]`, `(A+B).toarray()` = `[[2,1,2],[0,4,1],[4,0,6]]`. Non-test consumer: `add` is a public crate primitive over `CsMat`, available on any `CscMatrix`. Verification: `test_add` (`d[[0,0]]==2.0`, `d[[1,1]]==6.0`), `test_add_shape_mismatch` → green. |
| REQ-SCALAR-MUL (`A*s`, in place + new) | SHIPPED | impl `pub fn mul_scalar in csc.rs` (`self.inner.map(\|&v\| v * scalar)`) and `pub fn scale in csc.rs` (`self.inner.scale(scalar)`) mirror scipy `A*s` (`_mul_scalar`, `_data.py:134`, `self.data * other`). Live oracle (R-CHAR-3): `(A*2).toarray()` = `[[2,0,4],[0,6,0],[8,0,10]]`. Non-test consumer: scalar scaling is a public CSC primitive over the crate's `CsMat` newtype. Verification: `test_mul_scalar` (`d[[0,0]]==2.0`, `d[[1,1]]==6.0`), `test_scale_in_place` (`d[[0,0]]==3.0`, `d[[2,2]]==15.0`) → green. |
| REQ-COL-SLICE (`A[:,start:end]` column range) | SHIPPED | impl `pub fn col_slice in csc.rs` (`self.inner.slice_outer(start..end).to_owned()` after validating `start<=end<=n_cols`) mirrors scipy column slicing `A[:,start:end]` (`IndexMixin.__getitem__`, `_index.py:29`). Live oracle (R-CHAR-3): `A[:,0:2].toarray()` = `[[1,0],[0,3],[4,0]]`. API divergence (R-DEV-3): ferrolearn uses a fallible `col_slice(start,end)` method; scipy uses Python slice syntax `A[:,a:b]` — values match but the surface differs. CSC slices COLUMNS (the outer/`slice_outer` dimension of CSC), the natural analog of CSR's `row_slice`. Non-test consumer: `col_slice` is the public column-batch-extraction primitive over CSC. Verification: `test_col_slice` (`n_rows()==3`, `n_cols()==2`, `d[[0,0]]==1.0`, `d[[1,1]]==3.0`), `test_col_slice_empty` (`n_cols()==0`), `test_col_slice_invalid` → green. |
| REQ-MISSING-MATMUL (sparse-sparse `A@B`) | NOT-STARTED | blocker issue to be filed by critic. `csc.rs` has `mul_vec` (matrix×VECTOR) but NO matrix×matrix product. scipy `A@B`/`.dot(B)` is `_matmul_sparse` (`_compressed.py:415`). Live oracle: `(A@B).toarray()` = `[[1,1,2],[0,3,3],[4,4,5]]` — no `CscMatrix` method computes it. |
| REQ-MISSING-TRANSPOSE (`.transpose()`/`.T`) | SHIPPED (#2009) | `CscMatrix<T>::transpose(&self) -> CscMatrix<T>` returns `Aᵀ` (shape `(n_cols, n_rows)`) via `self.inner.transpose_view().to_csc()` — sprs swaps CSC↔CSR storage (no-alloc view of `Aᵀ`) then materializes owned CSC, mirroring scipy `A.T` being a CSR container over the same buffers (`_csc.py:20`). `#[must_use]`. Verification (live scipy, R-CHAR-3): `A=[[1,0,2],[0,3,0],[4,0,5]]` → `A.T=[[1,0,4],[0,3,0],[2,0,5]]` (3×3); `B=[[1,2,3],[4,5,6]]` (2×3) → `B.T=[[1,4],[2,5],[3,6]]` (3×2); `B.T.T==B`. Tests `csc_transpose_matches_scipy`, `csc_transpose_non_square`, `csc_transpose_twice_roundtrip`. |
| REQ-MISSING-REDUCE (`.sum(axis=)`, `.diagonal()`) | SHIPPED (#2010) | `CscMatrix<T>` gains `#[must_use]` `sum()` (total of all stored values, scipy `_compressed.py:492`), `sum_axis0()` (column sums, len `n_cols`), `sum_axis1()` (row sums, len `n_rows`), `diagonal()` (`A[i,i]` for `i in 0..min(n_rows,n_cols)`, absent→zero, `_compressed.py:476`) — storage-agnostic port of the CSR reductions (#2002), iterating the sprs `CsMat` triplets. Verification (live scipy, R-CHAR-3, `A=[[1,0,2],[0,3,0],[4,0,5]]`): `sum()=15`, `sum_axis0=[5,3,7]`, `sum_axis1=[3,3,9]`, `diagonal=[1,3,5]`; non-square `B=[[1,2,3],[4,5,6]]` `diagonal=[1,5]`. Tests `csc_sum_matches_scipy`, `csc_sum_axis0_matches_scipy`, `csc_sum_axis1_matches_scipy`, `csc_diagonal_matches_scipy`, `csc_diagonal_non_square`. |
| REQ-MISSING-ELEMENTWISE (`.multiply(B)`, `A-B`) | NOT-STARTED | blocker issue to be filed by critic. `csc.rs` has `add` but NO elementwise sparse `multiply` and NO `sub`. scipy `multiply` (`_base.py:490`) and `_sub_sparse` (`_compressed.py:260`). Live oracle: `A.multiply(B).toarray()` = `[[1,0,0],[0,3,0],[0,0,5]]`, `(A-B).toarray()` = `[[0,-1,2],[0,2,-1],[4,0,4]]`. |
| REQ-MISSING-INDEX (`A[i,j]`/`A[i,:]`/`A[:,j]` + housekeeping) | NOT-STARTED | blocker issue to be filed by critic. `csc.rs` has `col_slice` (contiguous column range) but NO scalar `A[i,j]` access, NO single-column `getcol`/`A[:,j]`, NO row `getrow`/`A[i,:]`, and NO `eliminate_zeros` (`_compressed.py:1025`) / `sort_indices` (`_compressed.py:1110`) / `sum_duplicates` (`_compressed.py:1063`) / `power` (`_data.py:99`) / `max`/`min`/`argmax` (`_minmax_mixin`) / `astype` (`_data.py:69`) / `copy` (`_data.py:94`). Live oracle: `A[1,1]` = 3.0, `A[:,0]` selects the first column. |
| REQ-API-ACCESSORS (`.shape`/`.data`/`.indices`/`.indptr`) | NOT-STARTED | blocker issue to be filed by critic. scipy exposes `.shape` (tuple, `_compressed.py:38`) and `.data`/`.indices`/`.indptr` (`_compressed.py:76-78`) as first-class accessors; ferrolearn exposes `n_rows()`/`n_cols()`/`nnz()` (NO `.shape()` tuple) and gates the three CSC arrays behind `inner()`/`into_inner()` (the only path to the raw arrays is `csc.inner().indptr()/.indices()/.data()` on `sprs::CsMat`). No public `shape`/`data`/`indices`/`indptr` accessor on `CscMatrix` (`grep -n "pub fn" csc.rs`). R-DEV-2/3 attribute-contract divergence. |
| REQ-ERR (Result + scipy validation timing) | SHIPPED | impl `pub fn add`/`pub fn mul_vec in csc.rs` validate shape at the operation: `add` with mismatched shape → `Err(FerroError::ShapeMismatch{context:"CscMatrix::add"})`, `mul_vec` with `rhs.len() != n_cols()` → `Err(FerroError::ShapeMismatch{context:"CscMatrix::mul_vec"})`. scipy raises `ValueError` at the SAME point (`A@v` len-2 vs 3 cols → `ValueError: dimension mismatch`; shape-mismatched add → `ValueError: inconsistent shapes`). Live oracle (R-CHAR-3) confirms scipy rejects both at the op; ferrolearn rejects both at `add`/`mul_vec`. `new` also returns `FerroError::InvalidParameter` on structurally invalid raw components (`CsMat::try_new_csc` error). Verification: `test_add_shape_mismatch`, `test_mul_vec_shape_mismatch` → green. (Error TYPE is `FerroError` per CLAUDE.md/R-CODE-2; `ValueError` marshalling is `ferrolearn-python`'s.) |
| REQ-CONSUMER (non-test caller) | SHIPPED | Honest (R-HONEST-3): unlike `CsrMatrix` (k-NN graph estimator consumer + `impl Dataset`), `CscMatrix` has NO cross-crate estimator consumer — `grep -rn "CscMatrix\|csc::" --include=*.rs ferrolearn-*/src \| grep -v ferrolearn-sparse/src \| grep -v '#[cfg(test'` returns NOTHING. The sole non-test production consumer is the in-crate CSR↔CSC conversion in `csr.rs`: `pub fn from_csc in csr.rs` (`csc.inner().to_csr()`) and `pub fn to_csc in csr.rs` (`CscMatrix::from_inner(self.inner.to_csc())`), reachable from any CSR user; `lib.rs` re-exports `pub use csc::CscMatrix`. `CscMatrix::from_csr`/`to_csr` close the round-trip. This is a genuine non-test consumer (the conversion is live translation surface), so the type is grandfathered existing pub API (R-DEFER-1/S5) — SHIPPED, but materially weaker than CSR's estimator-adjacent consumer. Verification: the grep above + `cargo test -p ferrolearn-sparse --lib` (`test_csc_csr_roundtrip`, csr's `test_csr_csc_roundtrip`) exercise the path → green. |
| REQ-FERRAY (ferray sparse substrate) | NOT-STARTED | blocker issue to be filed by critic. `csc.rs` is backed by `sprs::CsMat` (`use sprs::CsMat`) and materializes to `ndarray::Array2` (`use ndarray::{Array1, Array2, ArrayView2}`) — the WRONG substrate per R-SUBSTRATE-1 (sparse → ferray's `scipy.sparse` analog, not `sprs`; dense → `ferray-core`, not `ndarray`). ferray does not yet expose a sparse CSC surface (R-SUBSTRATE-5: a ferray gap is real work, filed upstream to ferray; this REQ is NOT-STARTED until ferray ships the sparse layer). |

## Architecture

`csc.rs` is a single-type module: `CscMatrix<T> { inner: CsMat<T> }`, a newtype
over `sprs::CsMat<T>` in CSC storage. There is no unfitted/Fitted split — CSC is
a data container, not an estimator — which mirrors scipy's `csc_matrix`
(likewise a matrix class). It is the column-symmetric twin of `CsrMatrix`,
implementing the construction/conversion/arithmetic half of `_cs_matrix`
(`_compressed.py:25`, the shared CSR/CSC base) plus the CSC-specific facade
`_csc_base` (`_csc.py:17`). The row/column roles are swapped throughout: `indptr`
is the column pointer (length `n_cols+1`), `indices` holds row indices, and the
"outer" dimension `slice_outer` walks is the column.

- **Construction & conversion.** `new` validates via `CsMat::try_new_csc`
  (returns `FerroError::InvalidParameter` on bad lengths / out-of-bound indices /
  unsorted inner indices). `from_coo` calls `coo.inner().to_csc()` (which
  coalesces duplicates — the same implicit `sum_duplicates` scipy does on
  `coo.tocsc()`); `from_csr`/`to_csr` cross-convert via `CsrMatrix::from_csc` /
  `csr.to_csc()`; `from_dense` threshold-prunes via `CsMat::csc_from_dense`;
  `to_coo` re-emits one triplet per stored entry; `to_dense` materializes a zero
  `Array2` filled from the stored entries (REQ-CONSTRUCT-CONVERT SHIPPED, oracle
  `A.toarray()`/`A.nnz`/round-trips). The `from_inner` constructor (pub(crate))
  is the path `csr.rs` uses to build a `CscMatrix` from `self.inner.to_csc()`.

- **Arithmetic & matvec.** `mul_vec` is `&self.inner * rhs` — the sparse-matrix ×
  dense-vector path, matching scipy `_matmul_vector` (`_compressed.py:387`)
  (REQ-MATVEC SHIPPED, oracle `[7,6,19]`). `add` is `&self.inner + &rhs.inner`
  after a shape guard, matching `_add_sparse` (REQ-ADD SHIPPED). `mul_scalar`/
  `scale` map `v*s` over the stored data, matching `_mul_scalar`
  (REQ-SCALAR-MUL SHIPPED). `col_slice` is `slice_outer(start..end)` — for CSC
  the outer dimension is the column — matching scipy `A[:,start:end]` column
  slicing with a method-vs-slice API divergence (REQ-COL-SLICE SHIPPED). Errors
  are `FerroError::ShapeMismatch` at the op, matching scipy's `ValueError`
  timing (REQ-ERR SHIPPED).

- **Escape hatch & the missing surface.** `inner()`/`into_inner()` expose
  `&CsMat`/`CsMat` — the only path to the `.indptr()`/`.indices()`/`.data()`
  slices, so there is NO scipy-shaped public accessor (`.shape`, `.data`,
  `.indices`, `.indptr`): REQ-API-ACCESSORS NOT-STARTED. And the bulk of the
  `csc_matrix` method surface is absent on `CscMatrix`: sparse-sparse matmul
  `A@B` (REQ-MISSING-MATMUL), `transpose`/`.T` (REQ-MISSING-TRANSPOSE),
  `sum(axis=)`/`diagonal` (REQ-MISSING-REDUCE), elementwise `multiply`/`sub`
  (REQ-MISSING-ELEMENTWISE), and element/row/column indexing plus the
  housekeeping methods `eliminate_zeros`/`sort_indices`/`sum_duplicates`/`power`/
  `max`/`min`/`argmax`/`astype`/`copy` (REQ-MISSING-INDEX) — all honest
  NOT-STARTED, not out-of-scope, because scipy exposes them and downstream
  translation may need them.

The two cross-cutting structural facts are REQ-CONSUMER and REQ-FERRAY. On
REQ-CONSUMER, the honest call (R-HONEST-3) is that `CscMatrix` is WEAKER than
`CsrMatrix`: a workspace grep finds NO cross-crate estimator consumer; the only
non-test production consumer is the in-crate CSR↔CSC conversion in `csr.rs`
(`CsrMatrix::from_csc`/`to_csc`) plus the `lib.rs` re-export. That conversion is
genuine live translation surface (any CSR user can reach it), so the existing
pub API is grandfathered (R-DEFER-1/S5) and the REQ is SHIPPED — but the doc
states plainly that there is no estimator that traffics in CSC directly, where
CSR has `ferrolearn-neighbors`'s k-NN graph constructors and `impl Dataset`.
REQ-FERRAY is NOT-STARTED — `sprs::CsMat` + `ndarray::Array2` is the wrong
substrate per R-SUBSTRATE-1; ferray has no sparse layer yet (R-SUBSTRATE-5). The
overall honest position is that the construction / conversion / matvec / add /
scalar-mul / col-slice / error / (weak) consumer core ships on impl +
live-oracle + non-test consumer, while the sparse-sparse-matmul, transpose,
reduce, elementwise, indexing, accessor, and ferray-substrate surfaces do not.

## Verification

Commands establishing the SHIPPED claims (run at baseline `b4b6d5bfa`):

- `cargo test -p ferrolearn-sparse --lib csc` → 0 failed
  (`test_new_valid`, `test_to_dense`, `test_from_dense`, `test_from_coo_roundtrip`,
  `test_csc_csr_roundtrip`, `test_col_slice`, `test_col_slice_empty`,
  `test_col_slice_invalid`, `test_mul_scalar`, `test_scale_in_place`,
  `test_add`, `test_add_shape_mismatch`, `test_mul_vec`,
  `test_mul_vec_shape_mismatch`).
- construct + toarray + nnz + conversion oracle (REQ-CONSTRUCT-CONVERT;
  R-CHAR-3 — expected from scipy, never from ferrolearn):
  `python3 -c "import numpy as np, scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.nnz, A.shape, A.toarray().tolist(), A.tocsr().toarray().tolist())"`
  → `5 (3, 3) [[1.0,0.0,2.0],[0.0,3.0,0.0],[4.0,0.0,5.0]] [[1.0,0.0,2.0],[0.0,3.0,0.0],[4.0,0.0,5.0]]`.
- matvec / add / scalar-mul / col-slice oracle
  (REQ-MATVEC, REQ-ADD, REQ-SCALAR-MUL, REQ-COL-SLICE):
  `python3 -c "import numpy as np, scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print((A@np.array([1.,2,3])).tolist(), (A+A).toarray().tolist(), (A*2).toarray().tolist(), A[:,0:2].toarray().tolist())"`
  → `[7.0,6.0,19.0] [[2.0,0.0,4.0],[0.0,6.0,0.0],[8.0,0.0,10.0]] [[2.0,0.0,4.0],[0.0,6.0,0.0],[8.0,0.0,10.0]] [[1.0,0.0],[0.0,3.0],[4.0,0.0]]`.
  ferrolearn `mul_vec`/`add`/`mul_scalar`/`col_slice` match (the four `test_*`
  tests above).
- missing-method oracle (REQ-MISSING-MATMUL / -TRANSPOSE / -REDUCE /
  -ELEMENTWISE / -INDEX):
  `python3 -c "import numpy as np, scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); B=sp.csc_matrix(np.array([[1.,1,0],[0,1,1],[0,0,1]])); print((A@B).toarray().tolist(), A.T.toarray().tolist(), A.sum(axis=0).tolist(), A.diagonal().tolist(), A.multiply(B).toarray().tolist(), (A-B).toarray().tolist(), A[1,1], A[:,0].toarray().tolist())"`
  → `[[1.0,1.0,2.0],[0.0,3.0,3.0],[4.0,4.0,5.0]] [[1.0,0.0,4.0],[0.0,3.0,0.0],[2.0,0.0,5.0]] [[5.0,3.0,7.0]] [1.0,3.0,5.0] [[1.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,5.0]] [[0.0,-1.0,2.0],[0.0,2.0,-1.0],[4.0,0.0,4.0]] 3.0 [[1.0],[0.0],[4.0]]`.
  No `CscMatrix` method computes any of these — a critic pins FAILING
  `matmul`/`transpose`/`sum`/`diagonal`/`multiply`/`sub`/`get`/`getcol` tests.
- accessor oracle (REQ-API-ACCESSORS):
  `python3 -c "import numpy as np, scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.shape, A.data.tolist(), A.indices.tolist(), A.indptr.tolist())"`
  → `(3, 3) [1.0,4.0,3.0,2.0,5.0] [0,2,1,0,2] [0,2,3,5]`. `grep -n "pub fn" csc.rs`
  shows no `shape`/`data`/`indices`/`indptr` accessor.
- validation oracle (REQ-ERR):
  `python3 -c "import numpy as np, scipy.sparse as sp; A=sp.csc_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); A@np.array([1.,2.])"`
  → `ValueError: dimension mismatch`. ferrolearn
  `CscMatrix::mul_vec(&array![1.,2.]).is_err()` holds
  (`test_mul_vec_shape_mismatch`); shape-mismatched `add` likewise
  (`test_add_shape_mismatch`).
- consumer check (REQ-CONSUMER):
  `grep -rn "CscMatrix\|csc::" --include=*.rs /home/doll/ferrolearn/ferrolearn-*/src | grep -v ferrolearn-sparse/src | grep -v '#[cfg(test'`
  returns NOTHING (no cross-crate estimator consumer);
  `grep -rn "from_csc\|to_csc\|CscMatrix" --include=*.rs /home/doll/ferrolearn/ferrolearn-sparse/src/csr.rs | grep -v '#[cfg(test'`
  shows `CsrMatrix::from_csc`/`to_csc` (the in-crate CSR↔CSC conversion — the
  sole non-test production consumer) and `lib.rs` re-exports
  `pub use csc::CscMatrix`.
