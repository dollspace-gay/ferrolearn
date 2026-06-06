# scipy.sparse.coo_matrix — COOrdinate (triplet) format

<!--
tier: 3-component
status: draft
baseline-commit: 2cef01a8e
upstream-paths:
  - scipy/sparse/_coo.py   # the _coo_base / coo_matrix / coo_array class
-->

## Summary

`ferrolearn-sparse/src/coo.rs` provides `CooMatrix<T>`, a newtype over
`sprs::TriMat<T>`, mirroring a slice of **scipy.sparse.coo_matrix** (COOrdinate /
triplet format). It stores non-zero entries as `(row, col, value)` triplets and
is the construction-time substrate the crate converts FROM (into CSR/CSC) — the
same role scipy's `coo_matrix` plays. The live oracle is the installed **scipy
1.17.1**; sparse construction / conversion / dense-materialization is
deterministic, so the scipy-1.17.1 / sklearn-1.5.2 version split is irrelevant (a
duplicate-summed COO is duplicate-summed in every scipy release).

What `CooMatrix` ships is the construction-and-materialization core:
`new`/`with_capacity`/`from_triplets`/`push` build the triplet list; `to_dense`
materializes it with duplicate-summing; `n_rows`/`n_cols`/`nnz` report geometry;
`inner`/`into_inner` are the `sprs::TriMat` escape hatch. What diverges is the
*accessor API surface* and the large set of `coo_matrix` *methods* that have no
ferrolearn analog.

Divergence classes:
1. **construction + toarray + nnz parity (the SHIPPED core)** —
   `from_triplets`/`push` + `to_dense` mirror
   `coo_matrix((data,(row,col)),shape)` + `.toarray()`, including
   duplicate-summing; `nnz()` mirrors scipy `.nnz` (stored entries, duplicates
   counted).
2. **duplicate-summing** — `to_dense` sums duplicate `(row,col)` entries, matching
   scipy `.toarray()` / `.tocsr()` (which call `sum_duplicates` implicitly).
3. **API accessors (R-DEV-2/3)** — ferrolearn exposes `n_rows()`/`n_cols()`
   (no `.shape` tuple) and hides the triplet arrays behind `inner()` (no public
   `.data`/`.row`/`.col`).
4. **missing COO methods** — `.tocsr()`/`.tocsc()`/`.todense()`,
   `.transpose()`/`.T`, `.sum(axis=)`, `.sum_duplicates()`, `.eliminate_zeros()`,
   `.diagonal()`, `.multiply()`, `.dot()`/`@`, arithmetic `+`/`-`/`*`,
   `.power()`, `.max()`/`.min()`, `.astype()`, `.copy()`, `.todia()`/`.todok()`
   are all absent as COO methods (conversion to CSR/CSC exists, but as
   `CsrMatrix::from_coo`/`CscMatrix::from_coo` in sibling modules, not as a
   `.tocsr()` method on `CooMatrix`).
5. **error handling** — `from_triplets`/`push` return `Result<_, FerroError>`;
   scipy raises `ValueError` on bad shape / out-of-bounds indices / mismatched
   array lengths. The validation *timing* matches (both at construction).
6. **sprs substrate (R-SUBSTRATE-1)** — `CooMatrix` wraps `sprs::TriMat`; the
   destination is ferray's `scipy.sparse` COO analog, not `sprs`.
7. **consumer** — `csr.rs` (`CsrMatrix::from_coo`), `csc.rs`
   (`CscMatrix::from_coo`), and `helpers.rs` (`eye`/`diags`/`hstack`/`vstack`)
   are real in-crate non-test production consumers.

## Upstream reference (scipy.sparse.coo_matrix, live oracle scipy 1.17.1)

The class is `_coo_base` (`scipy/sparse/_coo.py:28`), with `coo_matrix`/`coo_array`
as its 2-D/N-D faces. Cite the **method/attribute names** and the **live-oracle
values**, not internal `.pyx`/helper line numbers. Relevant surface:

- attributes: `.shape` (tuple), `.data` / `.row` / `.col` (the triplet arrays;
  `row` is `coords[-2]`, `col` is `coords[-1]`, `_coo.py:106`/`:122`), `.nnz`.
- `_getnnz(axis=None)` (`_coo.py:167-177`) returns `len(self.data)` — STORED
  entries, **duplicates counted** (it does NOT call `sum_duplicates`).
- `toarray` (`_coo.py:289`), `tocsr` (`_coo.py:349`), `tocsc` (`_coo.py:316`),
  `todense`, `transpose`/`T` (`_coo.py:229`), `_getnnz`,
  `sum_duplicates` (`_coo.py:768`), `eliminate_zeros` (`_coo.py:798`),
  `diagonal` (`_coo.py:458`), `sum`/`_sum_nd` (`_coo.py:1429`),
  `multiply`, `dot`/`@` (`_coo.py:1045`), arithmetic via `_add_dense`/
  `_add_sparse`/`_sub_sparse` (`_coo.py:811`/`:835`/`:847`), `power`,
  `max`/`min` (`_minmax_mixin`), `astype`, `copy`, `todia`/`todok`.

Live oracle (`cd /tmp && python3 -c "..."`, scipy 1.17.1):

```
m = sp.coo_matrix((np.array([1.,2.,5.]),(np.array([0,0,1]),np.array([0,0,2]))),shape=(2,3))
m.nnz        -> 3                              # stored entries, duplicate (0,0) counted
m.shape      -> (2, 3)
m.toarray()  -> [[3.0, 0.0, 0.0], [0.0, 0.0, 5.0]]   # duplicate (0,0)=1+2 summed -> 3
m.tocsr().nnz -> 2                             # tocsr sums duplicates -> 2 stored
m.data/m.row/m.col -> [1,2,5] / [0,0,1] / [0,0,2]
m.T.toarray() -> [[3,0],[0,0],[0,5]]
m.sum()      -> 8.0                            # 3 + 5
m.sum(axis=0)-> [[3,0,5]]   m.sum(axis=1) -> [3,5]
m.diagonal() -> [3.0, 0.0]
m.sum_duplicates(); m.nnz -> 2 ; m.data -> [3.0, 5.0]
```

scipy validation (oracle, at construction):

```
coo_matrix((data,(row,col)),shape=(2,3)) with row index 5   -> ValueError: axis 0 index 5 exceeds matrix dimension 2
coo_matrix with len(data)=2, len(row)=1                      -> ValueError: all index and data arrays must have the same length
```

## Requirements

- REQ-CONSTRUCT: `CooMatrix::from_triplets(n_rows,n_cols,row,col,data)` and the
  incremental `new`/`with_capacity` + `push(row,col,value)` builders mirror scipy
  `coo_matrix((data,(row,col)),shape)` — same `(row,col,value)` triplet
  representation, same shape, stored without coalescing.
- REQ-TOARRAY-DUP: `to_dense()` materializes the triplet list to a dense
  `Array2<T>`, **summing duplicate `(row,col)` entries**, matching scipy
  `.toarray()`. Oracle: `[[3,0,0],[0,0,5]]` (duplicate `(0,0)=1+2`).
- REQ-NNZ: `nnz()` returns the number of STORED entries counting duplicates
  (wraps `TriMat::nnz`), matching scipy `.nnz` / `_getnnz(axis=None)` =
  `len(self.data)`. Oracle: `nnz == 3` for the duplicate-bearing matrix.
- REQ-API-ACCESSORS: ferrolearn exposes the geometry/triplet surface scipy
  exposes — scipy has `.shape` (tuple) + `.data`/`.row`/`.col`; ferrolearn has
  `n_rows()`/`n_cols()` (no `.shape` tuple) and gates the triplet arrays behind
  `inner()`/`into_inner()` (no public `.data`/`.row`/`.col` accessors).
- REQ-MISSING-METHODS: the `coo_matrix` method surface beyond construction +
  materialization exists on `CooMatrix` — `.tocsr()`/`.tocsc()`/`.todense()`,
  `.transpose()`/`.T`, `.sum(axis=)`, `.sum_duplicates()`,
  `.eliminate_zeros()`, `.diagonal()`, `.multiply()`, `.dot()`/`@`, arithmetic
  `+`/`-`/`*`, `.power()`, `.max()`/`.min()`, `.astype()`, `.copy()`,
  `.todia()`/`.todok()`.
- REQ-ERR: `from_triplets`/`push` return `Result<_, FerroError>` with
  scipy-matching validation semantics — out-of-bounds row/col and
  mismatched-length triplet arrays rejected at construction (scipy raises
  `ValueError`; ferrolearn returns `FerroError::InvalidParameter`).
- REQ-CONSUMER: a non-test in-crate production caller consumes `CooMatrix` so it
  is part of the live translation surface (`CsrMatrix::from_coo` /
  `CscMatrix::from_coo` / `helpers::{eye,diags,hstack,vstack}`).
- REQ-FERRAY: `CooMatrix` is backed by ferray's `scipy.sparse` COO analog rather
  than `sprs::TriMat` (R-SUBSTRATE-1).

## Acceptance criteria

All expected values come from the live scipy 1.17.1 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn. Canonical matrix:
`data=[1,2,5]`, `row=[0,0,1]`, `col=[0,0,2]`, `shape=(2,3)` (duplicate at `(0,0)`).

- AC-CONSTRUCT (REQ-CONSTRUCT):
  `python3 -c "import numpy as np,scipy.sparse as sp; m=sp.coo_matrix((np.array([1.,2.,5.]),(np.array([0,0,1]),np.array([0,0,2]))),shape=(2,3)); print(m.shape, m.data.tolist(), m.row.tolist(), m.col.tolist())"`
  → `(2, 3) [1.0, 2.0, 5.0] [0, 0, 1] [0, 0, 2]`. `CooMatrix::from_triplets(2,3,vec![0,0,1],vec![0,0,2],vec![1.,2.,5.])`
  yields `n_rows()==2`, `n_cols()==3`, three stored triplets.
- AC-TOARRAY-DUP (REQ-TOARRAY-DUP):
  `python3 -c "import numpy as np,scipy.sparse as sp; print(sp.coo_matrix((np.array([1.,2.,5.]),(np.array([0,0,1]),np.array([0,0,2]))),shape=(2,3)).toarray().tolist())"`
  → `[[3.0, 0.0, 0.0], [0.0, 0.0, 5.0]]`. `CooMatrix::to_dense()` on the same
  triplets equals it element-wise (pinned by `test_coo_to_dense_duplicate_summed`,
  `d[[0,0]] == 3.0`).
- AC-NNZ (REQ-NNZ):
  `python3 -c "import numpy as np,scipy.sparse as sp; print(sp.coo_matrix((np.array([1.,2.,5.]),(np.array([0,0,1]),np.array([0,0,2]))),shape=(2,3)).nnz)"`
  → `3` (duplicate counted). `CooMatrix::nnz()` returns `3`. Contrast
  `m.tocsr().nnz` → `2` (CSR coalesces); `CsrMatrix::from_coo(&coo).nnz()` likewise
  drops to 2.
- AC-API-ACCESSORS (REQ-API-ACCESSORS): scipy exposes `.shape` (tuple) and
  `.data`/`.row`/`.col` arrays directly; ferrolearn exposes only `n_rows()`,
  `n_cols()`, `nnz()`, and the `inner()`/`into_inner()` `sprs::TriMat` handle —
  `grep -n "pub fn" coo.rs` shows no `shape`, `data`, `row`, or `col` accessor.
  A critic pins a FAILING test requiring a `.shape() -> (usize,usize)` tuple and
  public `.row()`/`.col()`/`.data()` slices. FAILS until added.
- AC-MISSING-METHODS (REQ-MISSING-METHODS):
  `python3 -c "import numpy as np,scipy.sparse as sp; m=sp.coo_matrix((np.array([1.,2.,5.]),(np.array([0,0,1]),np.array([0,0,2]))),shape=(2,3)); print(m.T.toarray().tolist(), m.sum(), m.sum(axis=0).tolist(), m.diagonal().tolist())"`
  → `[[3.0,0.0],[0.0,0.0],[0.0,5.0]] 8.0 [[3.0,0.0,5.0]] [3.0,0.0]`. `grep -n "pub fn" coo.rs`
  shows only `new`,`with_capacity`,`from_triplets`,`push`,`n_rows`,`n_cols`,
  `nnz`,`inner`,`into_inner`,`to_dense` — no `transpose`/`sum`/`diagonal`/
  `multiply`/`dot`/`sum_duplicates`/`eliminate_zeros`/`tocsr`/`tocsc`/`power`/
  `max`/`min`/`astype`/`copy`. A critic pins a FAILING `.transpose()` /
  `.sum(axis)` / `.diagonal()` test. FAILS until implemented.
- AC-ERR (REQ-ERR):
  `python3 -c "import numpy as np,scipy.sparse as sp; sp.coo_matrix((np.array([1.]),(np.array([5]),np.array([0]))),shape=(2,3))"`
  → `ValueError: axis 0 index 5 exceeds matrix dimension 2`; and mismatched
  lengths → `ValueError: all index and data arrays must have the same length`.
  `CooMatrix::from_triplets(2,3,vec![5],vec![0],vec![1.0])` returns
  `Err(FerroError::InvalidParameter{name:"row_inds", ...})` and the
  length-mismatch case returns `Err(FerroError::InvalidParameter{name:"triplet arrays", ...})`
  (pinned by `test_coo_from_triplets_out_of_bounds`, `test_coo_from_triplets_mismatch`,
  `test_coo_push_out_of_bounds`). Error TYPE diverges (`FerroError` vs
  `ValueError`) but is the sanctioned crate contract (CLAUDE.md / R-CODE-2);
  exception-type marshalling to `ValueError` is `ferrolearn-python`'s job.
- AC-CONSUMER (REQ-CONSUMER):
  `grep -rn "from_coo\|CooMatrix" --include=*.rs /home/doll/ferrolearn/ferrolearn-sparse/src/{csr.rs,csc.rs,helpers.rs} | grep -v '#\[cfg(test'`
  shows `CsrMatrix::from_coo`/`CscMatrix::from_coo` (`let inner = coo.inner().to_csr()`)
  and `helpers::eye`/`diags`/`hstack`/`vstack` building via `CooMatrix::new`/
  `with_capacity` + `push` then `CsrMatrix::from_coo`. These are non-test
  production consumers within the crate boundary (S5: COO→CSR conversion is a
  real consumer).
- AC-FERRAY (REQ-FERRAY): `coo.rs` imports `sprs::{SpIndex, TriMat}` and
  `ndarray::Array2`; the destination is ferray's sparse COO analog
  (R-SUBSTRATE-1). ferray does not yet expose a `scipy.sparse` layer
  (R-SUBSTRATE-5).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-CONSTRUCT (triplet construction) | SHIPPED | impl `pub fn from_triplets` + `pub fn new`/`with_capacity`/`pub fn push in coo.rs` (each wraps `TriMat::from_triplets`/`TriMat::new`/`add_triplet`) mirror scipy `coo_matrix((data,(row,col)),shape)` (`_coo.py:62-64` `self.coords = …; self.data = …`). Live oracle (R-CHAR-3): `shape=(2,3)`, `data=[1,2,5]`, `row=[0,0,1]`, `col=[0,0,2]` stored without coalescing. Non-test consumer: `helpers.rs` `eye` (`CooMatrix::<T>::with_capacity(n,n,n)` then `coo.push(i,i,T::one())`) and `diags`/`hstack`/`vstack`. Verification: `cargo test -p ferrolearn-sparse --lib coo` (`test_coo_new`, `test_coo_push`) → green. |
| REQ-TOARRAY-DUP (toarray + duplicate-summing) | SHIPPED | impl `pub fn to_dense in coo.rs` (`for (val,(r,c)) in self.inner.triplet_iter() { out[[r.index(),c.index()]] += val.clone(); }`) mirrors scipy `toarray` (`_coo.py:289`), which sums duplicates. Live oracle (R-CHAR-3): `m.toarray()` = `[[3,0,0],[0,0,5]]` (duplicate `(0,0)=1+2`); ferrolearn `to_dense()` accumulates with `+=`, giving `d[[0,0]]==3.0`. Non-test consumer: `CsrMatrix::from_coo` (`coo.inner().to_csr()` performs the same duplicate-summing into CSR). Verification: `test_coo_to_dense`, `test_coo_to_dense_duplicate_summed` (`d[[0,0]] == 3.0`) → green. |
| REQ-NNZ (stored-nnz incl. duplicates) | SHIPPED | impl `pub fn nnz in coo.rs` (`self.inner.nnz()`) mirrors scipy `_getnnz(axis=None)` = `len(self.data)` (`_coo.py:169-177`) — stored entries, duplicates counted, no implicit `sum_duplicates`. Live oracle (R-CHAR-3): `m.nnz == 3` for the duplicate-bearing matrix (vs `m.tocsr().nnz == 2` after coalescing). Non-test consumer: `csr.rs`/`csc.rs` `to_coo` use `CooMatrix::with_capacity(self.n_rows(), self.n_cols(), self.nnz())`. Verification: `test_coo_new` (`nnz()==0`), `test_coo_push` (`nnz()==2`), `test_coo_clone` (`nnz()==1`) → green. |
| REQ-API-ACCESSORS (`.shape`/`.data`/`.row`/`.col`) | SHIPPED (#1996) | `CooMatrix<T>` gains `#[must_use]` `shape() -> (usize, usize)` (= `(n_rows, n_cols)`, scipy `.shape`), `data() -> &[T]` (= `inner.data()`, `_coo.py:64`), `row() -> &[usize]` (= `inner.row_inds()`, `_coo.py:106`), `col() -> &[usize]` (= `inner.col_inds()`, `_coo.py:122`) — all borrow `&self` directly (sprs `TriMat` slices, no owned-`Vec` materialization needed, unlike CSR's `indptr`). Verification (live scipy `coo_matrix`, R-CHAR-3, `from_triplets(3,3,row=[0,2,1],col=[0,1,2],data=[3,5,2])`): `shape=(3,3)`, `data=[3,5,2]`, `row=[0,2,1]`, `col=[0,1,2]` (insertion order preserved, sprs `TriMat` does not coalesce/reorder). Test `coo_shape_data_row_col_match_scipy`. |
| REQ-MISSING-METHODS (tocsr/tocsc/transpose) | SHIPPED (#1997) | `CooMatrix<T>::to_csr() -> Result<CsrMatrix<T>, FerroError>` (delegates `CsrMatrix::from_coo(self)`, scipy `_coo.py:349`), `to_csc() -> Result<CscMatrix<T>, FerroError>` (`CscMatrix::from_coo`, `:316`), `transpose() -> Result<CooMatrix<T>, FerroError>` (swaps row↔col index arrays + `(M,N)→(N,M)` shape via `from_triplets(n_cols, n_rows, col, row, data)`, `:229`). Bounds `T: Clone + Add<Output=T> + 'static`. Verification (live scipy, R-CHAR-3, `from_triplets(3,3,row=[0,2,1],col=[0,1,2],data=[3,5,2])` = `[[3,0,0],[0,0,2],[0,5,0]]`): `tocsr`/`tocsc` `[[3,0,0],[0,0,2],[0,5,0]]`; `transpose` `[[3,0,0],[0,0,5],[0,2,0]]`; non-square `[[0,0,7],[9,0,0]]` (2×3) → transpose `[[0,9],[0,0],[7,0]]` (3×2). Tests `coo_to_csr_matches_scipy`, `coo_to_csc_matches_scipy`, `coo_transpose_matches_scipy`, `coo_transpose_non_square`. |
| REQ-MISSING-METHODS (sum/diagonal) | SHIPPED (#1997) | `CooMatrix<T>` gains `#[must_use]` `sum() -> T` (total of all stored values, duplicates summed, `_coo.py:1429`), `sum_axis0() -> Array1<T>` (column sums, len `n_cols`), `sum_axis1() -> Array1<T>` (row sums, len `n_rows`), `diagonal() -> Array1<T>` (`A[i,i]` for `i<min(n_rows,n_cols)`, diagonal duplicates summed, `:458`) — computed directly from the COO triplets (`row()`/`col()`/`data()`) so duplicates are summed (matching scipy). Bounds `T: Copy + Zero + Add<Output=T>`. Verification (live scipy, R-CHAR-3, `from_triplets(3,3,row=[0,2,1],col=[0,1,2],data=[3,5,2])` = `[[3,0,0],[0,0,2],[0,5,0]]`): `sum()=10`, `sum_axis0=[3,5,2]`, `sum_axis1=[3,2,5]`, `diagonal=[3,0,0]`; duplicate `(0,0)×2` data `[3,5]` → `sum()=8`, `diagonal=[8,0]`. Tests `coo_sum_matches_scipy`, `coo_sum_axis0_matches_scipy`, `coo_sum_axis1_matches_scipy`, `coo_diagonal_matches_scipy`, `coo_sum_diagonal_sum_duplicates`. |
| REQ-MISSING-METHODS (rest of coo_matrix surface) | NOT-STARTED | open prereq blocker #1997. Still absent: `sum_duplicates` (`_coo.py:768`), `eliminate_zeros` (`:798`), `multiply`, `dot`/`@` (`:1045`), arithmetic `+`/`-`/`*` (`:811`/`:835`/`:847`), `power`, `max`/`min`, `astype`, `copy`, `todia`/`todok`. |
| REQ-ERR (Result + scipy validation timing) | SHIPPED | impl `pub fn from_triplets`/`pub fn push in coo.rs` validate at construction: mismatched lengths → `Err(FerroError::InvalidParameter{name:"triplet arrays"})`, out-of-bounds row/col → `Err(FerroError::InvalidParameter{name:"row_inds"/"col_inds"/"row"/"col"})`. scipy raises `ValueError` at the SAME point (`coo_matrix((data,(row,col)),shape)` with row 5 → `ValueError: axis 0 index 5 exceeds matrix dimension 2`; mismatched lengths → `ValueError: all index and data arrays must have the same length`, `_coo.py:171`). Live oracle (R-CHAR-3) confirms scipy rejects both at construction; ferrolearn rejects both at `from_triplets`/`push`. Non-test consumer: `helpers::eye`/`diags` propagate `coo.push(...)` errors via `map_err`. Verification: `test_coo_from_triplets_out_of_bounds`, `test_coo_from_triplets_mismatch`, `test_coo_push_out_of_bounds` → green. (Error TYPE is `FerroError` per CLAUDE.md/R-CODE-2; `ValueError` marshalling is `ferrolearn-python`'s, not this unit's.) |
| REQ-CONSUMER (non-test in-crate caller) | SHIPPED | `CsrMatrix::from_coo in csr.rs` (`let inner: CsMat<T> = coo.inner().to_csr(); Ok(Self { inner })`) and `CscMatrix::from_coo in csc.rs` are non-test production consumers, as are `helpers::eye`/`diags`/`hstack`/`vstack in helpers.rs` (build via `CooMatrix::new`/`with_capacity` + `push`, then `CsrMatrix::from_coo`). `lib.rs` re-exports `pub use coo::CooMatrix` and the `helpers` constructors. Verification: `grep -rn "from_coo\|CooMatrix" csr.rs csc.rs helpers.rs \| grep -v '#[cfg(test'` shows the calls; `cargo test -p ferrolearn-sparse --lib` (`test_from_coo_roundtrip` in csr/csc) exercises the path → green. NOTE: an ESTIMATOR does not yet consume `CooMatrix` directly — `ferrolearn-neighbors/src/graph.rs` consumes `CsrMatrix`, not `CooMatrix`. The in-crate conversion + builder consumers (S5, within the crate boundary) make this SHIPPED; the wider estimator-consumption story is carried by the CSR/CSC docs. |
| REQ-FERRAY (ferray sparse substrate) | NOT-STARTED | blocker issue to be filed by critic. `coo.rs` is backed by `sprs::TriMat` (`use sprs::{SpIndex, TriMat}`) and materializes to `ndarray::Array2` — the WRONG substrate per R-SUBSTRATE-1 (sparse → ferray's `scipy.sparse` analog, not `sprs`; dense → `ferray-core`, not `ndarray`). ferray does not yet expose a sparse COO surface (R-SUBSTRATE-5: a ferray gap is real work, filed upstream to ferray; this REQ is NOT-STARTED until ferray ships the sparse layer). |

## Architecture

`coo.rs` is a single-type module: `CooMatrix<T> { inner: TriMat<T> }`, a newtype
over `sprs::TriMat<T>`. There is no unfitted/Fitted split — COO is a data
container, not an estimator — which is appropriate, since scipy's `coo_matrix` is
likewise a matrix class, not a fitted model. The type mirrors the construction
half of scipy's `_coo_base` (`_coo.py:28`):

- **Construction.** `new`/`with_capacity` wrap `TriMat::new`/`with_capacity` (an
  empty triplet list with a shape). `from_triplets` validates
  (lengths-equal, row/col in-bounds) then wraps `TriMat::from_triplets`. `push`
  validates row/col in-bounds then `TriMat::add_triplet`. This is exactly scipy's
  triplet ingestion (`self.coords = (row, col); self.data = data`,
  `_coo.py:62-64`), with the difference that scipy raises `ValueError` and
  ferrolearn returns `FerroError` (REQ-ERR — same timing, different error type by
  crate contract).

- **Materialization.** `to_dense` allocates a zero `Array2` and accumulates each
  triplet with `+=` over `triplet_iter()`, so duplicate `(row,col)` entries SUM —
  matching scipy `.toarray()` (`_coo.py:289`), which coalesces duplicates
  (REQ-TOARRAY-DUP SHIPPED, oracle `[[3,0,0],[0,0,5]]`). `nnz()` wraps
  `TriMat::nnz` = stored entries WITHOUT coalescing, matching scipy `_getnnz`
  (`len(self.data)`, `_coo.py:169`) — so `CooMatrix::nnz()` of the
  duplicate-bearing matrix is 3, dropping to 2 only after CSR conversion
  (`CsrMatrix::from_coo`), exactly as scipy `m.nnz == 3` vs `m.tocsr().nnz == 2`
  (REQ-NNZ SHIPPED).

- **Escape hatch.** `inner()`/`into_inner()` expose `&TriMat`/`TriMat`. This is
  how the crate reaches the row/col/data slices (`coo.inner().row_inds()` etc.)
  and how CSR/CSC conversion is performed — but it also means there is NO
  scipy-shaped public accessor surface (`.shape`, `.data`, `.row`, `.col`):
  REQ-API-ACCESSORS NOT-STARTED. And the bulk of `coo_matrix` methods
  (`transpose`, `sum`, `diagonal`, `multiply`, `dot`, arithmetic, `sum_duplicates`,
  `eliminate_zeros`, `power`, `max`/`min`, `astype`, `copy`, `todia`/`todok`)
  have no `CooMatrix` method — REQ-MISSING-METHODS NOT-STARTED. The pragmatic
  reason these are tolerable at baseline: COO's job in this crate is to be BUILT
  and CONVERTED, with arithmetic living on CSR/CSC — which mirrors scipy's own
  advice that COO is a construction format. But scipy still exposes the methods,
  so they are honest NOT-STARTED surface, not out-of-scope.

The two cross-cutting structural facts are REQ-CONSUMER (SHIPPED — `csr.rs`,
`csc.rs`, `helpers.rs` are genuine non-test in-crate callers; the COO→CSR
conversion is a real production consumer within the crate boundary, S5) and
REQ-FERRAY (NOT-STARTED — `sprs::TriMat` + `ndarray::Array2` is the wrong
substrate per R-SUBSTRATE-1; ferray has no sparse layer yet, R-SUBSTRATE-5). The
honest call (R-HONEST-3) is that the construction / toarray / nnz / error /
consumer core ships on impl + live-oracle + non-test consumer, while the accessor
API, the method surface, and the ferray substrate do not.

## Verification

Commands establishing the SHIPPED claims (run at baseline `2cef01a8e`):

- `cargo test -p ferrolearn-sparse --lib coo` → 11 passed, 0 failed
  (`test_coo_new`, `test_coo_push`, `test_coo_push_out_of_bounds`,
  `test_coo_from_triplets_mismatch`, `test_coo_from_triplets_out_of_bounds`,
  `test_coo_to_dense`, `test_coo_to_dense_duplicate_summed`, `test_coo_clone`,
  plus the `from_coo`/`to_coo` roundtrips in csr/csc).
- construction + toarray + nnz oracle (REQ-CONSTRUCT, REQ-TOARRAY-DUP, REQ-NNZ;
  R-CHAR-3 — expected from scipy, never from ferrolearn):
  `python3 -c "import numpy as np, scipy.sparse as sp; m=sp.coo_matrix((np.array([1.,2.,5.]),(np.array([0,0,1]),np.array([0,0,2]))),shape=(2,3)); print(m.nnz, m.shape, m.toarray().tolist(), m.tocsr().nnz)"`
  → `3 (2, 3) [[3.0, 0.0, 0.0], [0.0, 0.0, 5.0]] 2`. ferrolearn `to_dense()`
  matches the array; `nnz()` = 3; `CsrMatrix::from_coo(&coo).nnz()` = 2.
- transpose / sum / diagonal oracle (REQ-MISSING-METHODS):
  `python3 -c "import numpy as np, scipy.sparse as sp; m=sp.coo_matrix((np.array([1.,2.,5.]),(np.array([0,0,1]),np.array([0,0,2]))),shape=(2,3)); print(m.T.toarray().tolist(), m.sum(), m.sum(axis=0).tolist(), m.diagonal().tolist())"`
  → `[[3.0,0.0],[0.0,0.0],[0.0,5.0]] 8.0 [[3.0,0.0,5.0]] [3.0,0.0]`. No `CooMatrix`
  method computes any of these — a critic pins a FAILING `.transpose()`/`.sum()`/
  `.diagonal()` test.
- validation oracle (REQ-ERR):
  `python3 -c "import numpy as np, scipy.sparse as sp; sp.coo_matrix((np.array([1.]),(np.array([5]),np.array([0]))),shape=(2,3))"`
  → `ValueError: axis 0 index 5 exceeds matrix dimension 2`. ferrolearn
  `CooMatrix::<f64>::from_triplets(2,3,vec![5],vec![0],vec![1.0]).is_err()` holds
  (`test_coo_from_triplets_out_of_bounds`).
- consumer check (REQ-CONSUMER):
  `grep -rn "from_coo\|CooMatrix" --include=*.rs /home/doll/ferrolearn/ferrolearn-sparse/src/{csr.rs,csc.rs,helpers.rs} | grep -v '#[cfg(test'`
  shows `CsrMatrix::from_coo`/`CscMatrix::from_coo` and `helpers::{eye,diags,hstack,vstack}`
  consuming `CooMatrix` in production code.
