# scipy.sparse.csr_matrix — Compressed Sparse Row format

<!--
tier: 3-component
status: draft
baseline-commit: 0da4cf5f8
upstream-paths:
  - scipy/sparse/_csr.py          # _csr_base / csr_matrix / csr_array
  - scipy/sparse/_compressed.py   # _cs_matrix base (CSR/CSC shared impl)
-->

## Summary

`ferrolearn-sparse/src/csr.rs` provides `CsrMatrix<T>`, a newtype over
`sprs::CsMat<T>` (CSR storage), mirroring a slice of **scipy.sparse.csr_matrix**
(Compressed Sparse Row). CSR is the row-oriented compressed format scipy
recommends for matrix-vector products and row slicing, and is the format the
ferrolearn workspace's sparse-consuming estimators (k-NN graph construction,
`Dataset`-as-sparse) actually traffic in. The live oracle is the installed
**scipy 1.17.1**; CSR construction / conversion / arithmetic / matvec is
deterministic, so the scipy-1.17.1 / sklearn-1.5.2 version split is irrelevant
(`A@v` is `A@v` in every scipy release).

What `CsrMatrix` ships is the construction-conversion-and-arithmetic core:
`new`/`from_coo`/`from_csc`/`from_dense`/`from_dense_float` build it;
`to_dense`/`to_coo`/`to_csc` materialize/convert it; `mul_vec`/`add`/
`mul_scalar`/`scale` do matvec, elementwise-add, and scalar multiply;
`row_slice` extracts a contiguous row range; `n_rows`/`n_cols`/`nnz` report
geometry; `inner`/`into_inner` are the `sprs::CsMat` escape hatch. What diverges
is the *row-slice API shape*, the *accessor surface*, and the large set of
`csr_matrix` *methods* (sparse-sparse matmul, transpose, sum, diagonal,
elementwise multiply/subtract, indexing) that have no ferrolearn analog.

Divergence classes:
1. **construction + toarray + nnz + format conversion (the SHIPPED core)** —
   `from_coo`/`from_dense`/`from_csc` + `to_dense`/`to_coo`/`to_csc` mirror
   `csr_matrix(...)` + `.toarray()`/`.tocsc()`/`.tocoo()`; `nnz()` mirrors
   scipy `.nnz` (CSR coalesces duplicates on construction, so it counts
   distinct stored entries).
2. **arithmetic / matvec parity (SHIPPED)** — `mul_vec(v)` ≡ scipy `A@v`,
   `add(B)` ≡ scipy `A+B` (elementwise), `mul_scalar(s)`/`scale(s)` ≡ scipy
   `A*s`.
3. **row-slice API divergence (R-DEV-3)** — ferrolearn `row_slice(start,end)`
   (a fallible method returning a new `CsrMatrix`) vs scipy Python slicing
   `A[start:end]`. Same values, different API shape.
4. **missing csr_matrix methods (NOT-STARTED)** — sparse-sparse matmul
   `A@B`/`.dot(B)`, `.transpose()`/`.T`, `.multiply(B)` (elementwise sparse),
   elementwise subtract `A-B`, `.sum(axis=)`, `.diagonal()`, element/row/column
   indexing `A[i,j]`/`A[i,:]`/`A[:,j]`/`.getrow()`/`.getcol()`,
   `.eliminate_zeros()`, `.sort_indices()`, `.sum_duplicates()`, `.power()`,
   `.max()`/`.min()`/`.argmax()`, `.astype()`, `.copy()` have no `CsrMatrix`
   method.
5. **API accessors (R-DEV-2/3)** — scipy exposes `.shape` (tuple) and
   `.data`/`.indices`/`.indptr` (the CSR arrays) directly; ferrolearn exposes
   `n_rows()`/`n_cols()`/`nnz()` and gates the raw arrays behind
   `inner()`/`into_inner()` (no public `.shape`/`.data`/`.indices`/`.indptr`).
6. **error handling** — `add`/`mul_vec` return `Result<_, FerroError>` on shape
   mismatch; scipy raises `ValueError`. Validation timing matches.
7. **sprs substrate (R-SUBSTRATE-1)** — `CsrMatrix` wraps `sprs::CsMat` and
   materializes to `ndarray::Array2`; the destination is ferray's
   `scipy.sparse` CSR analog + `ferray-core`, not `sprs`/`ndarray`.
8. **consumer (SHIPPED)** — `CsrMatrix` has REAL cross-crate estimator
   consumers: `ferrolearn-neighbors/src/graph.rs` returns it from k-NN graph
   constructors, and `ferrolearn-core/src/dataset.rs`'s `Dataset` contract is
   satisfied by `impl Dataset for CsrMatrix` (the sparse arm of the dense/sparse
   split). `helpers.rs` and `csc.rs` are additional in-crate consumers.

## Upstream reference (scipy.sparse.csr_matrix, live oracle scipy 1.17.1)

`csr_matrix(spmatrix, _csr_base)` (`_csr.py:447`) inherits its storage and most
methods from `_cs_matrix(_data_matrix, _minmax_mixin, IndexMixin)`
(`_compressed.py:25`), the shared CSR/CSC base. Cite the **method/attribute
names** and the **live-oracle values**, not internal `.pyx` helper lines.
Relevant surface (line numbers stable at scipy 1.17.1):

- attributes: `.shape` (tuple, `self._shape`, `_compressed.py:38`),
  `.data` / `.indices` / `.indptr` (the three CSR arrays, set at
  `_compressed.py:76-78`), `.nnz`.
- `_getnnz(axis=None)` (`_compressed.py:118-120`) returns `int(self.indptr[-1])`
  — stored entries; CSR construction coalesces duplicates, so this is the count
  of DISTINCT stored positions.
- `toarray` (`_compressed.py:1002`), `tocsc` (`_csr.py:73`), `tocoo`,
  `transpose`/`T` (`_csr.py:22` — CSR.T is a CSC view of the same buffers),
  `diagonal` (`_compressed.py:476`), `sum`/`sum(axis=)` (`_compressed.py:492`),
  `_matmul_vector` (`_compressed.py:387`, the `A@v` path),
  `_matmul_sparse` (`_compressed.py:415`, the `A@B` path),
  `_add_sparse` (`_compressed.py:257`), `_sub_sparse` (`_compressed.py:260`),
  `multiply` (`_base.py:490`, elementwise),
  `eliminate_zeros` (`_compressed.py:1025`),
  `sum_duplicates` (`_compressed.py:1063`),
  `sort_indices` (`_compressed.py:1110`), `__getitem__` (`IndexMixin`,
  `_index.py:29`), `power`/`astype`/`copy` (`_data.py:99`/`:69`/`:94`),
  `max`/`min`/`argmax` (`_minmax_mixin`).

Live oracle (`cd /tmp && python3 -c "..."`, scipy 1.17.1). Canonical matrix
`A = [[1,0,2],[0,3,0],[4,0,5]]`, helper `B = [[1,1,0],[0,1,1],[0,0,1]]`:

```
A.nnz            -> 5
A.shape          -> (3, 3)
A.data/.indices/.indptr -> [1,2,3,4,5] / [0,2,1,0,2] / [0,2,3,5]
A.toarray()      -> [[1,0,2],[0,3,0],[4,0,5]]
A @ [1,2,3]      -> [7.0, 6.0, 19.0]                 # matvec
(A + A).toarray()-> [[2,0,4],[0,6,0],[8,0,10]]       # elementwise add
(A * 2).toarray()-> [[2,0,4],[0,6,0],[8,0,10]]       # scalar mul
A[0:2].toarray() -> [[1,0,2],[0,3,0]]                # row slice
A.tocsc().toarray() -> [[1,0,2],[0,3,0],[4,0,5]]     # round-trips
(A @ B).toarray()-> [[1,1,2],[0,3,3],[4,4,5]]        # sparse-sparse matmul
A.T.toarray()    -> [[1,0,4],[0,3,0],[2,0,5]]        # transpose
A.sum(axis=0)    -> [[5,3,7]]   A.diagonal() -> [1,3,5]
(A - B).toarray()-> [[0,-1,2],[0,2,-1],[4,0,4]]      # elementwise sub
A.multiply(B).toarray() -> [[1,0,0],[0,3,0],[0,0,5]] # elementwise mul
A[1,1]           -> 3.0      A[0,:].toarray() -> [[1,0,2]]
```

scipy validation (oracle):

```
A @ np.array([1.,2.]) (len 2, A has 3 cols) -> ValueError: dimension mismatch
A + sp.csr_matrix((2,3))   (shape (2,3) vs (3,3)) -> ValueError: inconsistent shapes
```

## Requirements

- REQ-CONSTRUCT-CONVERT: `CsrMatrix::from_coo`/`from_csc`/`from_dense`/
  `from_dense_float`/`new` construct a CSR matrix, and `to_dense`/`to_coo`/
  `to_csc` materialize/convert it — mirroring scipy `csr_matrix(...)` +
  `.toarray()`/`.tocoo()`/`.tocsc()`. `to_dense()` equals scipy `.toarray()`;
  `to_csc()` round-trips; `nnz()` equals scipy `.nnz` (CSR coalesces duplicates,
  so it counts distinct stored entries). Oracle: `A.toarray()` =
  `[[1,0,2],[0,3,0],[4,0,5]]`, `A.nnz` = 5, `A.tocsc().toarray()` round-trips.
- REQ-MATVEC: `mul_vec(v)` computes the sparse-matrix × dense-vector product
  `A @ v`, matching scipy's `_matmul_vector` (`_compressed.py:387`). Oracle:
  `A @ [1,2,3]` = `[7,6,19]`.
- REQ-ADD: `add(B)` computes the elementwise sum of two same-shape CSR matrices,
  matching scipy `A + B` (`_add_sparse`, `_compressed.py:257`). Oracle:
  `(A+A).toarray()` = `[[2,0,4],[0,6,0],[8,0,10]]`.
- REQ-SCALAR-MUL: `mul_scalar(s)` (new matrix) and `scale(s)` (in place) scale
  every stored entry by `s`, matching scipy `A * s` (`_mul_scalar`,
  `_data.py:134`). Oracle: `(A*2).toarray()` = `[[2,0,4],[0,6,0],[8,0,10]]`.
- REQ-ROW-SLICE: `row_slice(start,end)` returns a new `CsrMatrix` of rows
  `start..end`, matching scipy Python slicing `A[start:end]`, with the API
  divergence that ferrolearn uses a fallible method (start/end validated) where
  scipy uses `__getitem__` slice syntax (`_index.py:29`). Oracle:
  `A[0:2].toarray()` = `[[1,0,2],[0,3,0]]`.
- REQ-MISSING-MATMUL: sparse-sparse matmul `A@B`/`.dot(B)`
  (`_matmul_sparse`, `_compressed.py:415`) exists on `CsrMatrix`. Oracle:
  `(A@B).toarray()` = `[[1,1,2],[0,3,3],[4,4,5]]`.
- REQ-MISSING-TRANSPOSE: `.transpose()`/`.T` (`_csr.py:22`) exists on
  `CsrMatrix`. Oracle: `A.T.toarray()` = `[[1,0,4],[0,3,0],[2,0,5]]`.
- REQ-MISSING-REDUCE: `.sum(axis=)` (`_compressed.py:492`) and `.diagonal()`
  (`_compressed.py:476`) exist on `CsrMatrix`. Oracle: `A.sum(axis=0)` =
  `[[5,3,7]]`, `A.diagonal()` = `[1,3,5]`.
- REQ-MISSING-ELEMENTWISE: elementwise sparse multiply `.multiply(B)`
  (`_base.py:490`) and subtract `A-B` (`_sub_sparse`, `_compressed.py:260`)
  exist on `CsrMatrix`. Oracle: `A.multiply(B).toarray()` =
  `[[1,0,0],[0,3,0],[0,0,5]]`, `(A-B).toarray()` = `[[0,-1,2],[0,2,-1],[4,0,4]]`.
- REQ-MISSING-INDEX: element/row/column indexing `A[i,j]`/`A[i,:]`/`A[:,j]`,
  `.getrow()`/`.getcol()`, and the housekeeping `.eliminate_zeros()`/
  `.sort_indices()`/`.sum_duplicates()`/`.power()`/`.max()`/`.min()`/
  `.argmax()`/`.astype()`/`.copy()` methods exist on `CsrMatrix`. Oracle:
  `A[1,1]` = 3.0, `A[0,:].toarray()` = `[[1,0,2]]`.
- REQ-API-ACCESSORS: ferrolearn exposes the geometry/array surface scipy
  exposes — scipy has `.shape` (tuple) + `.data`/`.indices`/`.indptr`;
  ferrolearn has `n_rows()`/`n_cols()`/`nnz()` (no `.shape()` tuple) and gates
  the three CSR arrays behind `inner()`/`into_inner()` (no public
  `.data`/`.indices`/`.indptr`).
- REQ-ERR: `add`/`mul_vec` return `Result<_, FerroError>` with scipy-matching
  validation timing — shape mismatch rejected at the operation (scipy raises
  `ValueError`; ferrolearn returns `FerroError::ShapeMismatch`).
- REQ-CONSUMER: a non-test, cross-crate production caller consumes `CsrMatrix`
  so it is part of the live translation surface (`ferrolearn-neighbors`'s k-NN
  graph constructors return `CsrMatrix`; `impl Dataset for CsrMatrix` satisfies
  the `ferrolearn-core` `Dataset` contract; `helpers.rs`/`csc.rs` are in-crate
  consumers).
- REQ-FERRAY: `CsrMatrix` is backed by ferray's `scipy.sparse` CSR analog and
  `ferray-core` rather than `sprs::CsMat`/`ndarray::Array2` (R-SUBSTRATE-1).

## Acceptance criteria

All expected values come from the live scipy 1.17.1 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn. Canonical matrices
`A = [[1,0,2],[0,3,0],[4,0,5]]`, `B = [[1,1,0],[0,1,1],[0,0,1]]`. In ferrolearn,
`A` is `CsrMatrix::new(3,3,vec![0,2,3,5],vec![0,2,1,0,2],vec![1.,2.,3.,4.,5.])`.

- AC-CONSTRUCT-CONVERT (REQ-CONSTRUCT-CONVERT):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.nnz, A.shape, A.toarray().tolist(), A.tocsc().toarray().tolist())"`
  → `5 (3, 3) [[1.0,0.0,2.0],[0.0,3.0,0.0],[4.0,0.0,5.0]] [[1.0,0.0,2.0],[0.0,3.0,0.0],[4.0,0.0,5.0]]`.
  `CsrMatrix::to_dense()` equals `A.toarray()` element-wise (`test_to_dense`),
  `nnz()` = 5 (`test_new_valid`), `to_csc()` then `from_csc` round-trips
  (`test_csr_csc_roundtrip`), `from_coo`/`to_coo`/`from_dense` round-trip
  (`test_from_coo_roundtrip`, `test_to_coo_roundtrip`, `test_from_dense`).
- AC-MATVEC (REQ-MATVEC):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print((A@np.array([1.,2,3])).tolist())"`
  → `[7.0, 6.0, 19.0]`. `CsrMatrix::mul_vec(&array![1.,2.,3.])` =
  `[7,6,19]` (`test_mul_vec`).
- AC-ADD (REQ-ADD):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print((A+A).toarray().tolist())"`
  → `[[2.0,0.0,4.0],[0.0,6.0,0.0],[8.0,0.0,10.0]]`. `A.add(&A)?.to_dense()`
  matches (`test_add`, `d[[0,0]]==2.0`, `d[[1,1]]==6.0`).
- AC-SCALAR-MUL (REQ-SCALAR-MUL):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print((A*2).toarray().tolist())"`
  → `[[2.0,0.0,4.0],[0.0,6.0,0.0],[8.0,0.0,10.0]]`. `A.mul_scalar(2.0)` and
  `A.scale(3.0)` match (`test_mul_scalar` `d[[0,0]]==2.0`; `test_scale_in_place`
  `d[[2,2]]==15.0`).
- AC-ROW-SLICE (REQ-ROW-SLICE):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A[0:2].toarray().tolist())"`
  → `[[1.0,0.0,2.0],[0.0,3.0,0.0]]`. `A.row_slice(0,2)?.to_dense()` matches
  (`test_row_slice`, `n_rows()==2`, `d[[1,1]]==3.0`); out-of-range
  rejected (`test_row_slice_invalid`); empty slice `row_slice(1,1)` →
  `n_rows()==0` (`test_row_slice_empty`). API divergence: scipy uses
  `A[0:2]` slice syntax; ferrolearn uses the fallible `row_slice(0,2)` method.
- AC-MISSING-MATMUL (REQ-MISSING-MATMUL):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); B=sp.csr_matrix(np.array([[1.,1,0],[0,1,1],[0,0,1]])); print((A@B).toarray().tolist())"`
  → `[[1.0,1.0,2.0],[0.0,3.0,3.0],[4.0,4.0,5.0]]`. `grep -n "pub fn" csr.rs`
  shows `mul_vec` (matrix×VECTOR only) — no matrix×matrix. A critic pins a
  FAILING `A.matmul(&B)` test. FAILS until implemented.
- AC-MISSING-TRANSPOSE (REQ-MISSING-TRANSPOSE):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.T.toarray().tolist())"`
  → `[[1.0,0.0,4.0],[0.0,3.0,0.0],[2.0,0.0,5.0]]`. No `transpose`/`t` method on
  `CsrMatrix` (`grep -n "pub fn" csr.rs`). A critic pins a FAILING
  `A.transpose()` test. FAILS until implemented.
- AC-MISSING-REDUCE (REQ-MISSING-REDUCE):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.sum(axis=0).tolist(), A.diagonal().tolist())"`
  → `[[5.0,3.0,7.0]] [1.0,3.0,5.0]`. No `sum`/`diagonal` method on `CsrMatrix`.
  A critic pins FAILING `A.sum(0)`/`A.diagonal()` tests. FAILS until implemented.
- AC-MISSING-ELEMENTWISE (REQ-MISSING-ELEMENTWISE):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); B=sp.csr_matrix(np.array([[1.,1,0],[0,1,1],[0,0,1]])); print(A.multiply(B).toarray().tolist(), (A-B).toarray().tolist())"`
  → `[[1.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,5.0]] [[0.0,-1.0,2.0],[0.0,2.0,-1.0],[4.0,0.0,4.0]]`.
  `CsrMatrix` has `add` but no `multiply` (elementwise) and no `sub`. A critic
  pins FAILING `A.multiply(&B)`/`A.sub(&B)` tests. FAILS until implemented.
- AC-MISSING-INDEX (REQ-MISSING-INDEX):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A[1,1], A[0,:].toarray().tolist())"`
  → `3.0 [[1.0,0.0,2.0]]`. `CsrMatrix` has `row_slice` (contiguous range) but no
  `A[i,j]` scalar access, no single-row `getrow`, no column `getcol`/`A[:,j]`,
  and no `eliminate_zeros`/`sort_indices`/`sum_duplicates`/`power`/`max`/`min`/
  `argmax`/`astype`/`copy`. A critic pins a FAILING `A.get(1,1)`/`A.getrow(0)`
  test. FAILS until implemented.
- AC-API-ACCESSORS (REQ-API-ACCESSORS): scipy exposes `.shape` (tuple) and
  `.data`/`.indices`/`.indptr`:
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.shape, A.data.tolist(), A.indices.tolist(), A.indptr.tolist())"`
  → `(3, 3) [1.0,2.0,3.0,4.0,5.0] [0,2,1,0,2] [0,2,3,5]`. ferrolearn exposes
  `n_rows()`/`n_cols()`/`nnz()` and only the `inner()`/`into_inner()`
  `sprs::CsMat` handle — `grep -n "pub fn" csr.rs` shows no `shape`/`data`/
  `indices`/`indptr` accessor. A critic pins a FAILING test requiring a
  `.shape() -> (usize,usize)` tuple and public `.data()`/`.indices()`/
  `.indptr()` slices. FAILS until added.
- AC-ERR (REQ-ERR):
  `python3 -c "import numpy as np,scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); A@np.array([1.,2.])"`
  → `ValueError: dimension mismatch`; and shape-mismatched add → `ValueError:
  inconsistent shapes`. `CsrMatrix::mul_vec(&array![1.,2.])` returns
  `Err(FerroError::ShapeMismatch{context:"CsrMatrix::mul_vec",...})`
  (`test_mul_vec_shape_mismatch`); `A.add(&m_2x3)` returns
  `Err(FerroError::ShapeMismatch{context:"CsrMatrix::add",...})`
  (`test_add_shape_mismatch`). Error TYPE diverges (`FerroError` vs
  `ValueError`) but is the sanctioned crate contract (CLAUDE.md / R-CODE-2);
  `ValueError` marshalling is `ferrolearn-python`'s job.
- AC-CONSUMER (REQ-CONSUMER):
  `grep -rn "CsrMatrix" --include=*.rs /home/doll/ferrolearn/ferrolearn-neighbors/src/graph.rs /home/doll/ferrolearn/ferrolearn-sparse/src/{helpers.rs,csc.rs} | grep -v '#[cfg(test'`
  shows `ferrolearn-neighbors`'s `kneighbors_graph`/`radius_neighbors_graph`
  constructors returning `Result<CsrMatrix<F>, FerroError>` (built via
  `CsrMatrix::new`), `helpers::{eye,diags,hstack,vstack}` returning `CsrMatrix`,
  and `csc.rs` (`CscMatrix::from_csr`/`to_csr`). The `Dataset` impl
  (`impl Dataset for CsrMatrix in csr.rs`) is the sparse arm of the
  `ferrolearn-core` dense/sparse split cited in `dataset.rs`'s own REQ table.
  These are non-test, cross-crate, estimator-adjacent production consumers.
- AC-FERRAY (REQ-FERRAY): `csr.rs` imports `sprs::CsMat` and
  `ndarray::{Array1,Array2,ArrayView2}`; the destination is ferray's sparse CSR
  analog + `ferray-core` (R-SUBSTRATE-1). ferray does not yet expose a
  `scipy.sparse` layer (R-SUBSTRATE-5).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-CONSTRUCT-CONVERT (construct + toarray + nnz + format conversion) | SHIPPED | impl `pub fn from_coo`/`from_csc`/`from_dense`/`from_dense_float`/`new` + `pub fn to_dense`/`to_coo`/`to_csc`/`nnz in csr.rs` (each wraps `CsMat::try_new`/`to_csr`/`csr_from_dense`/`to_dense`/`to_csc`) mirror scipy `csr_matrix(...)` + `.toarray()` (`_compressed.py:1002`)/`.tocsc()` (`_csr.py:73`)/`.tocoo()` and `.nnz` (`_getnnz` = `int(self.indptr[-1])`, `_compressed.py:118-120`). Live oracle (R-CHAR-3): `A.nnz`=5, `A.shape`=`(3,3)`, `A.toarray()`=`[[1,0,2],[0,3,0],[4,0,5]]`, `A.tocsc().toarray()` round-trips; CSR coalesces duplicates so `nnz` counts distinct stored entries. Non-test consumer: `ferrolearn-neighbors/src/graph.rs` builds via `CsrMatrix::new` and returns the matrix; `helpers::eye`/`diags` build via `CsrMatrix::from_coo`. Verification: `cargo test -p ferrolearn-sparse --lib csr` (`test_new_valid`, `test_to_dense`, `test_csr_csc_roundtrip`, `test_from_coo_roundtrip`, `test_to_coo_roundtrip`, `test_from_dense`, `test_from_dense_float`) → 20 passed, 0 failed. |
| REQ-MATVEC (sparse×dense-vector `A@v`) | SHIPPED | impl `pub fn mul_vec in csr.rs` (`let result = &self.inner * rhs;`) mirrors scipy's `_matmul_vector` (`_compressed.py:387`, the `A@v` path). Live oracle (R-CHAR-3): `A @ [1,2,3]` = `[7.0, 6.0, 19.0]`. Non-test consumer: cross-crate `impl Dataset for CsrMatrix` + `ferrolearn-neighbors` graph (CSR matvec is the sparse matvec primitive estimators use); in-crate `mul_vec` is exercised on every CSR built by `graph.rs`. Verification: `test_mul_vec` (`result[0]==7.0`, `result[2]==19.0`), `test_mul_vec_shape_mismatch` → green. |
| REQ-ADD (elementwise `A+B`) | SHIPPED | impl `pub fn add in csr.rs` (`let result = &self.inner + &rhs.inner;` after a shape check) mirrors scipy `A+B` (`_add_sparse`, `_compressed.py:257`). Live oracle (R-CHAR-3): `(A+A).toarray()` = `[[2,0,4],[0,6,0],[8,0,10]]`. Non-test consumer: `helpers::hstack`/`vstack` compose CSR blocks; `add` is a public crate primitive over `CsMat`. Verification: `test_add` (`d[[0,0]]==2.0`, `d[[1,1]]==6.0`), `test_add_shape_mismatch` → green. |
| REQ-SCALAR-MUL (`A*s`, in place + new) | SHIPPED | impl `pub fn mul_scalar in csr.rs` (`self.inner.map(\|&v\| v * scalar)`) and `pub fn scale in csr.rs` (`self.inner.scale(scalar)`) mirror scipy `A*s` (`_mul_scalar`, `_data.py:134`, `self.data * other`). Live oracle (R-CHAR-3): `(A*2).toarray()` = `[[2,0,4],[0,6,0],[8,0,10]]`. Non-test consumer: scalar scaling is a public CSR primitive over the crate's `CsMat` newtype, available to every `CsrMatrix` returned by `graph.rs`/`helpers.rs`. Verification: `test_mul_scalar` (`d[[0,0]]==2.0`), `test_scale_in_place` (`d[[2,2]]==15.0`) → green. |
| REQ-ROW-SLICE (`A[start:end]` row range) | SHIPPED | impl `pub fn row_slice in csr.rs` (`self.inner.slice_outer(start..end).to_owned()` after validating `start<=end<=n_rows`) mirrors scipy row slicing `A[start:end]` (`IndexMixin.__getitem__`, `_index.py:29`). Live oracle (R-CHAR-3): `A[0:2].toarray()` = `[[1,0,2],[0,3,0]]`. API divergence (R-DEV-3): ferrolearn uses a fallible `row_slice(start,end)` method; scipy uses Python slice syntax — values match but the surface differs. Non-test consumer: `row_slice` is the public batch-extraction primitive over CSR (row sub-selection is the k-NN graph row-batching idiom). Verification: `test_row_slice` (`n_rows()==2`, `d[[1,1]]==3.0`), `test_row_slice_empty` (`n_rows()==0`), `test_row_slice_invalid` → green. |
| REQ-MISSING-MATMUL (sparse-sparse `A@B`) | SHIPPED (#2000) | `CsrMatrix<T>::matmul(&self, rhs) -> Result<CsrMatrix<T>, FerroError>` computes the sparse-sparse product (shape `(n_rows, rhs.n_cols)`) via the sprs `&CsMat * &CsMat` operator (dispatches to `smmp::mul_csr_csr`, the SMMP analog of scipy `_matmul_sparse` `_compressed.py:415`), `.to_csr()` materialized; shape-checked (`self.n_cols() != rhs.n_rows()` → `ShapeMismatch`). Bounds `T: Clone + sprs::MulAcc + Zero + Default + Send + Sync + 'static`. Verification (live scipy, R-CHAR-3, `A=[[1,0,2],[0,3,0],[4,0,5]]`, `B=[[1,1,0],[0,1,1],[0,0,1]]`, `C=[[1,2],[3,4],[5,6]]` 3×2): `A@B=[[1,1,2],[0,3,3],[4,4,5]]`; `A@C=[[11,14],[9,12],[29,38]]` (3×2); mismatch → `Err`. Tests `csr_matmul_matches_scipy`, `csr_matmul_non_square`, `csr_matmul_shape_mismatch_is_err`. |
| REQ-MISSING-TRANSPOSE (`.transpose()`/`.T`) | SHIPPED (#2001) | `CsrMatrix<T>::transpose(&self) -> CsrMatrix<T>` returns `Aᵀ` (shape `(n_cols, n_rows)`) via `self.inner.transpose_view().to_csr()` — sprs swaps CSR↔CSC storage (no-alloc view of `Aᵀ`) then materializes owned CSR, mirroring scipy `A.T` being a CSC container over the same buffers (`_csr.py:22`). `#[must_use]`. Verification (live scipy, R-CHAR-3): `A=[[1,0,2],[0,3,0],[4,0,5]]` → `A.T=[[1,0,4],[0,3,0],[2,0,5]]` (3×3); `B=[[1,2,3],[4,5,6]]` (2×3) → `B.T=[[1,4],[2,5],[3,6]]` (3×2); `B.T.T==B`. Tests `csr_transpose_matches_scipy`, `csr_transpose_non_square`, `csr_transpose_twice_roundtrip`. |
| REQ-MISSING-REDUCE (`.sum(axis=)`, `.diagonal()`) | SHIPPED (#2002) | `CsrMatrix<T>` gains `#[must_use]` `sum()` (total of all stored values, scipy `_compressed.py:492` `axis=None`), `sum_axis0()` (column sums, len `n_cols`), `sum_axis1()` (row sums, len `n_rows`), `diagonal()` (`A[i,i]` for `i in 0..min(n_rows,n_cols)`, absent→zero, `_compressed.py:476`), iterating the sprs `CsMat` triplets / `outer_iterator`. Verification (live scipy, R-CHAR-3, `A=[[1,0,2],[0,3,0],[4,0,5]]`): `sum()=15`, `sum_axis0=[5,3,7]`, `sum_axis1=[3,3,9]`, `diagonal=[1,3,5]`; non-square `B=[[1,2,3],[4,5,6]]` `diagonal=[1,5]`. Tests `csr_sum_matches_scipy`, `csr_sum_axis0_matches_scipy`, `csr_sum_axis1_matches_scipy`, `csr_diagonal_matches_scipy`, `csr_diagonal_non_square`. |
| REQ-MISSING-ELEMENTWISE (`.multiply(B)`, `A-B`) | SHIPPED (#2003) | `CsrMatrix<T>::multiply(&self, rhs) -> Result<CsrMatrix<T>, FerroError>` (elementwise Hadamard via `sprs::binop::mul_mat_same_storage`, intersection sparsity, scipy `_base.py:490`) and `sub(&self, rhs) -> Result<CsrMatrix<T>, FerroError>` (`A − B` via `&inner - &inner`, union sparsity, `_compressed.py:260`), both shape-checked (mismatch → `ShapeMismatch`) mirroring `add`. Verification (live scipy, R-CHAR-3, `A=[[1,0,2],[0,3,0],[4,0,5]]`, `B=[[1,1,0],[0,1,1],[0,0,1]]`): `A.multiply(B)=[[1,0,0],[0,3,0],[0,0,5]]`; `A.sub(B)=[[0,-1,2],[0,2,-1],[4,0,4]]`; shape-mismatch → `Err`. Tests `csr_multiply_matches_scipy`, `csr_sub_matches_scipy`, `csr_multiply_shape_mismatch_is_err`, `csr_sub_shape_mismatch_is_err`. (`.power()` — `_data.py:99` — remains NOT-STARTED, sub-noted.) |
| REQ-MISSING-INDEX (element access `A[i,j]`) | SHIPPED (#2004) | `CsrMatrix<T>::get(&self, i, j) -> Result<T, FerroError>` (`T: Copy + Zero`) returns the stored value, or `T::zero()` if structurally absent, via `self.inner.get(i,j).copied().unwrap_or_else(T::zero)`; out-of-bounds `i>=n_rows`/`j>=n_cols` → `Err(InvalidParameter)` (scipy `IndexError`, `_index.py` `IndexMixin.__getitem__`, R-DEV-2). Verification (live scipy, R-CHAR-3, `A=[[1,0,2],[0,3,0],[4,0,5]]`): `A[1,1]=3`, `A[0,1]=0` (absent), `A[0,0]=1`, `A[2,0]=4`; out-of-bounds → `Err`. Tests `csr_get_element_matches_scipy`, `csr_get_absent_is_zero`, `csr_get_out_of_bounds_is_err`. |
| REQ-MISSING-INDEX (rows/cols/maintenance) | NOT-STARTED | open prereq blocker #2004. Still absent: `getrow`/single-row `A[i,:]`, column `getcol`/`A[:,j]`, `eliminate_zeros` (`_compressed.py:1025`), `sort_indices` (`:1110`), `sum_duplicates` (`:1063`), `power` (`_data.py:99`), `max`/`min`/`argmax` (`_minmax_mixin`), `astype` (`_data.py:69`), `copy` (`_data.py:94`). Live oracle: `A[0,:].toarray()` = `[[1,0,2]]`. |
| REQ-API-ACCESSORS (`.shape`/`.data`/`.indices`/`.indptr`) | SHIPPED (#2005) | `CsrMatrix<T>` gains `#[must_use]` `shape() -> (usize, usize)` (= `(n_rows, n_cols)`, scipy `.shape` `_compressed.py:38`), `data() -> &[T]` (= `inner.data()`), `indices() -> &[usize]` (= `inner.indices()`, CSR column indices), and `indptr() -> Vec<usize>` (= `inner.indptr().raw_storage().to_vec()`, the row pointer array; OWNED `Vec` because the sprs `IndPtrView` is a temporary and cannot yield a `&[usize]` tied to `&self` — documented deviation) (`_compressed.py:76-78`). Verification (live scipy, R-CHAR-3, `A=[[1,0,2],[0,3,0],[4,0,5]]`): `shape=(3,3)`, `data=[1,2,3,4,5]`, `indices=[0,2,1,0,2]`, `indptr=[0,2,3,5]`. Test `csr_shape_data_indices_indptr_match_scipy`. |
| REQ-ERR (Result + scipy validation timing) | SHIPPED | impl `pub fn add`/`pub fn mul_vec in csr.rs` validate shape at the operation: `add` with mismatched shape → `Err(FerroError::ShapeMismatch{context:"CsrMatrix::add"})`, `mul_vec` with `rhs.len() != n_cols()` → `Err(FerroError::ShapeMismatch{context:"CsrMatrix::mul_vec"})`. scipy raises `ValueError` at the SAME point (`A@v` len-2 vs 3 cols → `ValueError: dimension mismatch`; shape-mismatched add → `ValueError: inconsistent shapes`). Live oracle (R-CHAR-3) confirms scipy rejects both at the op; ferrolearn rejects both at `add`/`mul_vec`. Non-test consumer: `graph.rs` propagates `CsrMatrix::new`/op errors via `?`. Verification: `test_add_shape_mismatch`, `test_mul_vec_shape_mismatch`, `test_new_invalid` → green. (Error TYPE is `FerroError` per CLAUDE.md/R-CODE-2; `ValueError` marshalling is `ferrolearn-python`'s.) |
| REQ-CONSUMER (non-test cross-crate caller) | SHIPPED | `ferrolearn-neighbors/src/graph.rs` is a REAL estimator-adjacent consumer: `pub fn kneighbors_graph`/`pub fn radius_neighbors_graph in graph.rs` and the connectivity/distance helpers all return `Result<CsrMatrix<F>, FerroError>`, building via `CsrMatrix::new(n_rows, n_cols, indptr, col_indices, data)`. `ferrolearn-core/src/dataset.rs`'s own REQ table cites `impl Dataset for CsrMatrix in csr.rs` as the cross-crate production consumer of the sparse `Dataset` arm (`is_sparse() == true`). In-crate: `helpers::{eye,diags,hstack,vstack in helpers.rs}` return `CsrMatrix` and `csc.rs` (`CscMatrix::from_csr`/`to_csr`) consumes it. `lib.rs` re-exports `pub use csr::CsrMatrix`. Verification: the grep above + `cargo test -p ferrolearn-sparse --lib` and `cargo test -p ferrolearn-neighbors` exercise the path → green. (R-DEFER-1/S5: `CsrMatrix` is grandfathered existing pub API with genuine downstream estimator consumers — this REQ is firmly SHIPPED.) |
| REQ-FERRAY (ferray sparse substrate) | NOT-STARTED | blocker issue to be filed by critic. `csr.rs` is backed by `sprs::CsMat` (`use sprs::CsMat`) and materializes to `ndarray::Array2` (`use ndarray::{Array1, Array2, ArrayView2}`) — the WRONG substrate per R-SUBSTRATE-1 (sparse → ferray's `scipy.sparse` analog, not `sprs`; dense → `ferray-core`, not `ndarray`). ferray does not yet expose a sparse CSR surface (R-SUBSTRATE-5: a ferray gap is real work, filed upstream to ferray; this REQ is NOT-STARTED until ferray ships the sparse layer). |

## Architecture

`csr.rs` is a single-type module: `CsrMatrix<T> { inner: CsMat<T> }`, a newtype
over `sprs::CsMat<T>` in CSR storage. There is no unfitted/Fitted split — CSR is
a data container, not an estimator — which mirrors scipy's `csr_matrix`
(likewise a matrix class). It implements the construction/conversion/arithmetic
half of `_cs_matrix` (`_compressed.py:25`, the shared CSR/CSC base) plus the
CSR-specific facade `_csr_base` (`_csr.py`):

- **Construction & conversion.** `new` validates via `CsMat::try_new` (returns
  `FerroError::InvalidParameter` on bad lengths / out-of-bound indices /
  unsorted inner indices). `from_coo` calls `coo.inner().to_csr()` (which
  coalesces duplicates — the same implicit `sum_duplicates` scipy does on
  `coo.tocsr()`); `from_csc`/`to_csc` cross-convert via `sprs`'s `to_csr`/
  `to_csc`; `from_dense`/`from_dense_float` threshold-prune via
  `CsMat::csr_from_dense`; `to_coo` re-emits one triplet per stored entry;
  `to_dense` materializes a zero `Array2` filled from the stored entries
  (REQ-CONSTRUCT-CONVERT SHIPPED, oracle `A.toarray()`/`A.nnz`/round-trips).

- **Arithmetic & matvec.** `mul_vec` is `&self.inner * rhs` — the sparse-matrix ×
  dense-vector path, matching scipy `_matmul_vector` (`_compressed.py:387`)
  (REQ-MATVEC SHIPPED, oracle `[7,6,19]`). `add` is `&self.inner + &rhs.inner`
  after a shape guard, matching `_add_sparse` (REQ-ADD SHIPPED). `mul_scalar`/
  `scale` map `v*s` over the stored data, matching `_mul_scalar`
  (REQ-SCALAR-MUL SHIPPED). `row_slice` is `slice_outer(start..end)`, matching
  scipy `A[start:end]` row slicing with a method-vs-slice API divergence
  (REQ-ROW-SLICE SHIPPED). Errors are `FerroError::ShapeMismatch` at the op,
  matching scipy's `ValueError` timing (REQ-ERR SHIPPED).

- **Escape hatch & the missing surface.** `inner()`/`into_inner()` expose
  `&CsMat`/`CsMat` — the only path to the `.indptr()`/`.indices()`/`.data()`
  slices, so there is NO scipy-shaped public accessor (`.shape`, `.data`,
  `.indices`, `.indptr`): REQ-API-ACCESSORS NOT-STARTED. And the bulk of the
  `csr_matrix` method surface is absent on `CsrMatrix`: sparse-sparse matmul
  `A@B` (REQ-MISSING-MATMUL), `transpose`/`.T` (REQ-MISSING-TRANSPOSE),
  `sum(axis=)`/`diagonal` (REQ-MISSING-REDUCE), elementwise `multiply`/`sub`
  (REQ-MISSING-ELEMENTWISE), and element/row/column indexing plus the
  housekeeping methods `eliminate_zeros`/`sort_indices`/`sum_duplicates`/`power`/
  `max`/`min`/`argmax`/`astype`/`copy` (REQ-MISSING-INDEX) — all honest
  NOT-STARTED, not out-of-scope, because scipy exposes them and downstream
  translation may need them.

The two cross-cutting structural facts are REQ-CONSUMER (SHIPPED — unlike
`CooMatrix`, `CsrMatrix` has a REAL cross-crate estimator-adjacent consumer:
`ferrolearn-neighbors/src/graph.rs` returns `CsrMatrix` from its k-NN graph
constructors, and `ferrolearn-core/src/dataset.rs` relies on
`impl Dataset for CsrMatrix` for the sparse arm of its `is_sparse` split; the
type is grandfathered existing pub API per R-DEFER-1/S5) and REQ-FERRAY
(NOT-STARTED — `sprs::CsMat` + `ndarray::Array2` is the wrong substrate per
R-SUBSTRATE-1; ferray has no sparse layer yet, R-SUBSTRATE-5). The honest call
(R-HONEST-3) is that the construction / conversion / matvec / add / scalar-mul /
row-slice / error / consumer core ships on impl + live-oracle + non-test
consumer, while the sparse-sparse-matmul, transpose, reduce, elementwise,
indexing, accessor, and ferray-substrate surfaces do not.

## Verification

Commands establishing the SHIPPED claims (run at baseline `0da4cf5f8`):

- `cargo test -p ferrolearn-sparse --lib csr` → 20 passed, 0 failed
  (`test_new_valid`, `test_new_invalid`, `test_to_dense`, `test_from_dense`,
  `test_from_dense_float`, `test_from_coo_roundtrip`, `test_to_coo_roundtrip`,
  `test_csr_csc_roundtrip`, `test_row_slice`, `test_row_slice_empty`,
  `test_row_slice_invalid`, `test_mul_scalar`, `test_scale_in_place`,
  `test_add`, `test_add_shape_mismatch`, `test_mul_vec`,
  `test_mul_vec_shape_mismatch`, `test_dataset_trait`,
  `test_dataset_trait_object`).
- construct + toarray + nnz + conversion oracle (REQ-CONSTRUCT-CONVERT;
  R-CHAR-3 — expected from scipy, never from ferrolearn):
  `python3 -c "import numpy as np, scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.nnz, A.shape, A.toarray().tolist(), A.tocsc().toarray().tolist())"`
  → `5 (3, 3) [[1.0,0.0,2.0],[0.0,3.0,0.0],[4.0,0.0,5.0]] [[1.0,0.0,2.0],[0.0,3.0,0.0],[4.0,0.0,5.0]]`.
- matvec / add / scalar-mul / row-slice oracle
  (REQ-MATVEC, REQ-ADD, REQ-SCALAR-MUL, REQ-ROW-SLICE):
  `python3 -c "import numpy as np, scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print((A@np.array([1.,2,3])).tolist(), (A+A).toarray().tolist(), (A*2).toarray().tolist(), A[0:2].toarray().tolist())"`
  → `[7.0,6.0,19.0] [[2.0,0.0,4.0],[0.0,6.0,0.0],[8.0,0.0,10.0]] [[2.0,0.0,4.0],[0.0,6.0,0.0],[8.0,0.0,10.0]] [[1.0,0.0,2.0],[0.0,3.0,0.0]]`.
  ferrolearn `mul_vec`/`add`/`mul_scalar`/`row_slice` match (the four `test_*`
  tests above).
- missing-method oracle (REQ-MISSING-MATMUL / -TRANSPOSE / -REDUCE /
  -ELEMENTWISE / -INDEX):
  `python3 -c "import numpy as np, scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); B=sp.csr_matrix(np.array([[1.,1,0],[0,1,1],[0,0,1]])); print((A@B).toarray().tolist(), A.T.toarray().tolist(), A.sum(axis=0).tolist(), A.diagonal().tolist(), A.multiply(B).toarray().tolist(), (A-B).toarray().tolist(), A[1,1], A[0,:].toarray().tolist())"`
  → `[[1.0,1.0,2.0],[0.0,3.0,3.0],[4.0,4.0,5.0]] [[1.0,0.0,4.0],[0.0,3.0,0.0],[2.0,0.0,5.0]] [[5.0,3.0,7.0]] [1.0,3.0,5.0] [[1.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,5.0]] [[0.0,-1.0,2.0],[0.0,2.0,-1.0],[4.0,0.0,4.0]] 3.0 [[1.0,0.0,2.0]]`.
  No `CsrMatrix` method computes any of these — a critic pins FAILING
  `matmul`/`transpose`/`sum`/`diagonal`/`multiply`/`sub`/`get`/`getrow` tests.
- accessor oracle (REQ-API-ACCESSORS):
  `python3 -c "import numpy as np, scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); print(A.shape, A.data.tolist(), A.indices.tolist(), A.indptr.tolist())"`
  → `(3, 3) [1.0,2.0,3.0,4.0,5.0] [0,2,1,0,2] [0,2,3,5]`. `grep -n "pub fn" csr.rs`
  shows no `shape`/`data`/`indices`/`indptr` accessor.
- validation oracle (REQ-ERR):
  `python3 -c "import numpy as np, scipy.sparse as sp; A=sp.csr_matrix(np.array([[1.,0,2],[0,3,0],[4,0,5]])); A@np.array([1.,2.])"`
  → `ValueError: dimension mismatch`. ferrolearn
  `CsrMatrix::mul_vec(&array![1.,2.]).is_err()` holds
  (`test_mul_vec_shape_mismatch`); shape-mismatched `add` likewise
  (`test_add_shape_mismatch`).
- consumer check (REQ-CONSUMER):
  `grep -rn "CsrMatrix" --include=*.rs /home/doll/ferrolearn/ferrolearn-neighbors/src/graph.rs /home/doll/ferrolearn/ferrolearn-sparse/src/{helpers.rs,csc.rs} | grep -v '#[cfg(test'`
  shows `ferrolearn-neighbors` graph constructors returning `CsrMatrix` (built
  via `CsrMatrix::new`), `helpers::{eye,diags,hstack,vstack}` returning
  `CsrMatrix`, and `csc.rs` `CscMatrix::from_csr`/`to_csr` consuming it; plus
  `dataset.rs`'s REQ table citing `impl Dataset for CsrMatrix`. These are
  non-test, cross-crate production consumers.
