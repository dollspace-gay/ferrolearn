# scipy.sparse construction helpers — eye / diags / hstack / vstack

<!--
tier: 3-component
status: draft
baseline-commit: 29ea38b42
upstream-paths:
  - scipy/sparse/_construct.py    # eye / identity / diags / spdiags / hstack / vstack / bmat / block_diag / kron / random
  - scipy/sparse/_extract.py      # tril / triu
-->

## Summary

`ferrolearn-sparse/src/helpers.rs` provides four free functions —
[`eye`](eye), [`diags`](diags), [`hstack`](hstack), [`vstack`](vstack) — that
mirror a slice of **scipy.sparse**'s construction helpers (`scipy.sparse.eye`,
`scipy.sparse.diags`, `scipy.sparse.hstack`, `scipy.sparse.vstack` in
`scipy/sparse/_construct.py`). Each returns a `CsrMatrix<T>` built through the
crate's own `CooMatrix` → `CsrMatrix::from_coo` pipeline. The live oracle is the
installed **scipy 1.17.1** (`_construct.py`); construction is deterministic, so
the scipy-1.17.1 / sklearn-1.5.2 version split is irrelevant — `eye(3).toarray()`
is the 3×3 identity in every scipy release.

What ships is a narrow, square/single-diagonal core: `eye(n)` builds the `n×n`
identity (scipy `eye(n)`); `diags(values, offset, n)` lays a SINGLE diagonal at a
SINGLE signed offset on an `n×n` grid (scipy `diags([values], [offset],
shape=(n,n))`); `hstack`/`vstack` concatenate a slice of equal-row / equal-column
`CsrMatrix` blocks (scipy `hstack`/`vstack`). What diverges is the *parameter
shape* (ferrolearn `diags` takes one diagonal where scipy takes a LIST of
diagonals + LIST of offsets), the *length-validation contract* (ferrolearn
SILENTLY SKIPS out-of-bounds entries where scipy raises `ValueError` on a
too-short diagonal), the *rectangular/offset gap* in `eye`, the *format/mixed-type
gap* in `hstack`/`vstack`, and the large set of scipy.sparse construction helpers
(`identity`, `spdiags`, `bmat`, `block_diag`, `block_array`, `kron`/`kronsum`,
`random`/`rand`, `tril`/`triu`) with no ferrolearn analog.

Divergence classes:
1. **eye square-identity parity (the SHIPPED core)** — `eye(n)` == scipy
   `eye(n).toarray()` (n×n identity); the rectangular `eye(m, n)` + offset `k`
   generality is the gap.
2. **diags single-diagonal + alignment (SHIPPED)** — `diags(values, offset, n)`
   for offset 0 / >0 / <0 aligns exactly as scipy `diags([values], [offset],
   shape=(n,n))`; the multi-diagonal API and the length-validation contract are
   separate divergences.
3. **diags multi-diagonal (NOT-STARTED)** — scipy `diags(diagonals, offsets)`
   takes a LIST of diagonals + LIST of offsets; ferrolearn takes a SINGLE
   diagonal + single offset.
4. **diags length-validation divergence (NOT-STARTED)** — ferrolearn silently
   skips out-of-bounds positions (`if i < n && j < n`); scipy raises `ValueError`
   when a supplied diagonal is too SHORT for its offset. The headline divergence.
5. **hstack / vstack CSR-concat parity (SHIPPED)** — block concatenation matches
   scipy `hstack`/`vstack`; the `format=`/`dtype=` params and mixed-format /
   mixed-type input are the gap.
6. **missing construction helpers (NOT-STARTED)** — `identity`, `spdiags`,
   `bmat`, `block_diag`, `block_array`, `kron`/`kronsum`, `random`/`rand`,
   `tril`/`triu` have no ferrolearn analog.
7. **consumer (NOT-STARTED)** — NO ferrolearn estimator consumes `eye`/`diags`/
   `hstack`/`vstack`; the only non-test reference is the `lib.rs` re-export.
8. **ferray sparse substrate (NOT-STARTED)** — the helpers build on the crate's
   `CooMatrix`/`CsrMatrix` (which wrap `sprs`) and materialize through `ndarray`;
   the destination is ferray's `scipy.sparse` analog (R-SUBSTRATE-1).

## Upstream reference (scipy.sparse construction helpers, live oracle scipy 1.17.1)

Cite the scipy.sparse **function names + signatures** and the **live-oracle
values**, not internal helper lines. Signatures stable at scipy 1.17.1
(`scipy/sparse/_construct.py`, `_extract.py`):

- `eye(m, n=None, k=0, dtype=float, format=None)` (`_construct.py:678`) — `m×n`
  matrix with ones on the `k`-th diagonal; `n` defaults to `m` (square) and `k`
  defaults to `0` (main diagonal). `identity(n, dtype='d', format=None)`
  (`_construct.py:547`) is the square-only `eye(n)` shorthand.
- `diags(diagonals, offsets=0, shape=None, format=None, dtype=...)`
  (`_construct.py:445`) — `diagonals` is a **sequence of array_like** (one array
  per offset), `offsets` is a **sequence of int or an int**: `k=0` main, `k>0`
  upper, `k<0` lower. `spdiags(data, diags, m=None, n=None, format=None)`
  (`_construct.py:207`) is the MATLAB-style variant.
- `hstack(blocks, format=None, dtype=None)` (`_construct.py:1012`) /
  `vstack(blocks, format=None, dtype=None)` (`_construct.py:1059`) — stack sparse
  matrices horizontally / vertically; accept a `format=` result-format selector
  and a `dtype=` coercion, and mixed input formats.
- absent in ferrolearn: `bmat` (`_construct.py:1107`), `block_array`
  (`_construct.py:1171`), `block_diag` (`_construct.py:1313`), `kron`
  (`_construct.py:728`), `kronsum` (`_construct.py:843`), `random_array`
  (`_construct.py:1398`) / `random` (`_construct.py:1550`) / `rand`
  (`_construct.py:1656`), `tril` (`_extract.py:46`) / `triu` (`_extract.py:113`).

Live oracle (`cd /tmp && python3 -c "..."`, scipy 1.17.1):

```
sp.eye(3).toarray()                          -> [[1,0,0],[0,1,0],[0,0,1]]
sp.eye(2,3,k=1).toarray()                    -> [[0,1,0],[0,0,1]]          # rectangular + offset
sp.diags([1.,2.,3.],0,shape=(3,3)).toarray() -> [[1,0,0],[0,2,0],[0,0,3]]  # main
sp.diags([1.,2.],1,shape=(3,3)).toarray()    -> [[0,1,0],[0,0,2],[0,0,0]]  # super, at (0,1),(1,2)
sp.diags([1.,2.],-1,shape=(3,3)).toarray()   -> [[0,0,0],[1,0,0],[0,2,0]]  # sub,   at (1,0),(2,1)
sp.diags([[1.,2.,3.],[4.,5.]],[0,1],shape=(3,3)).toarray()
                                             -> [[1,4,0],[0,2,5],[0,0,3]]   # multi-diagonal
a=sp.eye(2,format='csr'); b=sp.diags([5.,5.],0,shape=(2,2),format='csr')
sp.hstack([a,b]).toarray()                   -> [[1,0,5,0],[0,1,0,5]]
sp.vstack([a,b]).toarray()                   -> [[1,0],[0,1],[5,0],[0,5]]
```

scipy `diags` length-validation oracle (the headline divergence). scipy's rule is
asymmetric: a diagonal that is TOO LONG for its offset is silently truncated, a
broadcastable length-1 list is broadcast, but a diagonal that is TOO SHORT raises
`ValueError`:

```
sp.diags([1.,2.,3.],1,shape=(3,3))    # 3 values for a len-2 super-diagonal
    -> OK, truncates -> [[0,1,0],[0,0,2],[0,0,0]]
sp.diags([1.,2.],0,shape=(3,3))       # 2 values for a len-3 main diagonal (too short)
    -> ValueError: Diagonal length (index 0: 2 at offset 0) does not agree with array size (3, 3).
```

## Requirements

- REQ-EYE: `eye(n)` builds the `n×n` sparse identity, mirroring scipy `eye(n)`
  (`_construct.py:678`, square `m=n`, `k=0`). Square/`k=0` only — the rectangular
  `eye(m, n)` and offset-diagonal `k` generality scipy supports is NOT in
  ferrolearn's `eye(n)` (single `usize` parameter). Oracle: `eye(3).toarray()` =
  `[[1,0,0],[0,1,0],[0,0,1]]`.
- REQ-DIAGS-SINGLE: `diags(values, offset, n)` lays `values` on the single
  diagonal at signed `offset` of an `n×n` grid, with alignment matching scipy
  `diags([values], [offset], shape=(n,n))` (`_construct.py:445`): offset 0 →
  `(k,k)`, offset > 0 (super) → `(k, k+offset)`, offset < 0 (sub) →
  `(k+|offset|, k)`. Oracle: super-1 entries at `(0,1),(1,2)`; sub-(-1) entries
  at `(1,0),(2,1)`.
- REQ-DIAGS-MULTI: `diags` accepts a LIST of diagonals + LIST of offsets in one
  call (scipy `diags(diagonals, offsets)`, `_construct.py:445`, `diagonals` =
  *sequence of array_like*, `offsets` = *sequence of int*). Oracle:
  `diags([[1,2,3],[4,5]],[0,1],shape=(3,3)).toarray()` = `[[1,4,0],[0,2,5],[0,0,3]]`.
- REQ-DIAGS-LENGTH-VALIDATION: when a supplied diagonal does not fit its offset,
  ferrolearn matches scipy's validation contract — scipy raises `ValueError` when
  a diagonal is too SHORT for its slot (`_construct.py` `diags`, oracle
  `diags([1,2],0,(3,3))` → `ValueError: Diagonal length ... does not agree with
  array size`). ferrolearn's `diags` instead SILENTLY SKIPS any `(i,j)` with
  `i >= n || j >= n` (the `if i < n && j < n` guard), producing a short diagonal
  with no error. Divergence per R-DEV-1/2 (silent truncation vs `ValueError`).
- REQ-HSTACK: `hstack(matrices)` horizontally concatenates equal-row `CsrMatrix`
  blocks (summing column widths), mirroring scipy `hstack(blocks)`
  (`_construct.py:1012`). `&[&CsrMatrix<T>]` only — scipy's `format=`/`dtype=`
  params and mixed-format inputs are not supported. Oracle: `hstack([eye(2),
  diags([5,5],0,2)]).toarray()` = `[[1,0,5,0],[0,1,0,5]]`.
- REQ-VSTACK: `vstack(matrices)` vertically concatenates equal-column `CsrMatrix`
  blocks (summing row counts), mirroring scipy `vstack(blocks)`
  (`_construct.py:1059`). `&[&CsrMatrix<T>]` only; same `format=`/`dtype=`/mixed
  gap. Oracle: `vstack([eye(2), diags([5,5],0,2)]).toarray()` =
  `[[1,0],[0,1],[5,0],[0,5]]`.
- REQ-MISSING-HELPERS: the scipy.sparse construction helpers `identity`
  (`_construct.py:547`), `spdiags` (`:207`), `bmat` (`:1107`), `block_array`
  (`:1171`), `block_diag` (`:1313`), `kron` (`:728`), `kronsum` (`:843`),
  `random_array`/`random`/`rand` (`:1398`/`:1550`/`:1656`), and `tril`/`triu`
  (`_extract.py:46`/`:113`) have a ferrolearn analog. (`block_diag`/`kron` appear
  in some sklearn paths — kernel construction, graph Laplacians.)
- REQ-CONSUMER: a non-test, cross-crate production caller consumes
  `eye`/`diags`/`hstack`/`vstack` so they are part of the live translation
  surface.
- REQ-FERRAY: the helpers build on ferray's `scipy.sparse` analog rather than the
  crate's `sprs`-backed `CooMatrix`/`CsrMatrix` materializing through `ndarray`
  (R-SUBSTRATE-1).

## Acceptance criteria

All expected values come from the live scipy 1.17.1 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn.

- AC-EYE (REQ-EYE):
  `python3 -c "import scipy.sparse as sp; print(sp.eye(3).toarray().tolist())"`
  → `[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]`. ferrolearn
  `eye::<f64>(3)?.to_dense()` equals it (`test_eye_basic`). Rectangular/offset
  gap: `sp.eye(2,3,k=1).toarray()` = `[[0,1,0],[0,0,1]]` has no ferrolearn
  expression (`eye(n)` takes a single `usize`).
- AC-DIAGS-SINGLE (REQ-DIAGS-SINGLE):
  `python3 -c "import scipy.sparse as sp; print(sp.diags([1.,2.,3.],0,shape=(3,3)).toarray().tolist(), sp.diags([1.,2.],1,shape=(3,3)).toarray().tolist(), sp.diags([1.,2.],-1,shape=(3,3)).toarray().tolist())"`
  → `[[1,0,0],[0,2,0],[0,0,3]] [[0,1,0],[0,0,2],[0,0,0]] [[0,0,0],[1,0,0],[0,2,0]]`.
  ferrolearn `diags(&[1.,2.,3.],0,3)`, `diags(&[1.,2.],1,3)`, `diags(&[1.,2.],-1,3)`
  match (`test_diags_main_diagonal`, `test_diags_super_diagonal`; sub-diagonal
  alignment `(1,0),(2,1)` confirmed by the offset `< 0` branch
  `(k + (-offset) as usize, k)`).
- AC-DIAGS-MULTI (REQ-DIAGS-MULTI):
  `python3 -c "import scipy.sparse as sp; print(sp.diags([[1.,2.,3.],[4.,5.]],[0,1],shape=(3,3)).toarray().tolist())"`
  → `[[1.0,4.0,0.0],[0.0,2.0,5.0],[0.0,0.0,3.0]]`. ferrolearn `diags(values,
  offset, n)` takes a SINGLE `&[T]` + single `isize` — there is no expression for
  a multi-diagonal call. A critic pins a FAILING multi-diagonal test. FAILS until
  a multi-diagonal API exists.
- AC-DIAGS-LENGTH-VALIDATION (REQ-DIAGS-LENGTH-VALIDATION):
  `python3 -c "import scipy.sparse as sp; sp.diags([1.,2.],0,shape=(3,3))"`
  → `ValueError: Diagonal length (index 0: 2 at offset 0) does not agree with
  array size (3, 3).` ferrolearn `diags(&[1.,2.],0,3)` does NOT error — it lays
  two entries and returns `Ok` (the `if i < n && j < n` guard silently accepts a
  short diagonal). Likewise `diags(&[1.,2.,3.],1,3)` in ferrolearn silently skips
  the third value (super-diagonal position `(2,3)` has `j=3 >= n=3`) and returns a
  2-entry super-diagonal with no error. A critic pins a FAILING test requiring
  `diags(&[1.,2.],0,3)` to return `Err(FerroError::...)`. FAILS until a length
  check is added.
- AC-HSTACK (REQ-HSTACK):
  `python3 -c "import scipy.sparse as sp; a=sp.eye(2,format='csr'); b=sp.diags([5.,5.],0,shape=(2,2),format='csr'); print(sp.hstack([a,b]).toarray().tolist())"`
  → `[[1.0,0.0,5.0,0.0],[0.0,1.0,0.0,5.0]]`. ferrolearn `hstack(&[&eye(2)?,
  &diags(&[5.,5.],0,2)?])?` has `n_rows()==2`, `n_cols()==4`, `d[[0,2]]==5.0`
  (`test_hstack_basic`); mismatched-row input → `Err(FerroError::ShapeMismatch)`;
  empty slice → `Err(FerroError::InvalidParameter)`.
- AC-VSTACK (REQ-VSTACK):
  `python3 -c "import scipy.sparse as sp; a=sp.eye(2,format='csr'); b=sp.diags([5.,5.],0,shape=(2,2),format='csr'); print(sp.vstack([a,b]).toarray().tolist())"`
  → `[[1.0,0.0],[0.0,1.0],[5.0,0.0],[0.0,5.0]]`. ferrolearn `vstack(&[&eye(2)?,
  &diags(&[5.,5.],0,2)?])?` has `n_rows()==4`, `n_cols()==2`, `d[[2,0]]==5.0`
  (`test_vstack_basic`); mismatched-column input → `Err(FerroError::ShapeMismatch)`;
  empty slice → `Err(FerroError::InvalidParameter)`.
- AC-MISSING-HELPERS (REQ-MISSING-HELPERS):
  `python3 -c "import scipy.sparse as sp; print(sp.block_diag([sp.eye(2), sp.eye(1)]).toarray().tolist(), sp.kron(sp.eye(2), sp.eye(2)).toarray().shape)"`
  runs in scipy. `grep -n "pub fn" helpers.rs` lists only `eye`/`diags`/`hstack`/
  `vstack` — no `identity`/`spdiags`/`bmat`/`block_diag`/`block_array`/`kron`/
  `kronsum`/`random`/`rand`/`tril`/`triu`. A critic pins a FAILING test invoking a
  missing helper. FAILS until implemented.
- AC-CONSUMER (REQ-CONSUMER):
  `grep -rn "\beye\|\bdiags\|\bhstack\|\bvstack\b" --include=*.rs /home/doll/ferrolearn/ferrolearn-*/src | grep -v 'helpers.rs' | grep -v '#\[cfg(test'`
  finds NO estimator/transformer call to `ferrolearn_sparse::{eye,diags,hstack,
  vstack}` — the only non-test reference is `lib.rs` `pub use helpers::{diags,
  eye, hstack, vstack}` (the re-export itself). `column_transformer.rs::hstack`
  is a DIFFERENT, in-module dense `Array2<f64>` helper; the `Array2::eye`/
  `vstack` hits elsewhere are `ndarray`'s own methods / doc comments. FAILS the
  non-test-consumer requirement until an estimator consumes these helpers.
- AC-FERRAY (REQ-FERRAY): `helpers.rs` imports `crate::coo::CooMatrix` and
  `crate::csr::CsrMatrix` (both `sprs`-backed, materializing through
  `ndarray::Array2`); the destination is ferray's `scipy.sparse` construction
  analog (R-SUBSTRATE-1). ferray does not yet expose a sparse layer
  (R-SUBSTRATE-5).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-EYE (n×n identity) | SHIPPED | impl `pub fn eye in helpers.rs` (loops `0..n`, `coo.push(i, i, T::one())`, then `CsrMatrix::from_coo`) mirrors scipy `eye(m, n=None, k=0)` (`_construct.py:678`) for the square `m=n`, `k=0` case. Live oracle (R-CHAR-3): `sp.eye(3).toarray()` = `[[1,0,0],[0,1,0],[0,0,1]]`. Gap (documented, NOT a separate REQ failure): scipy's rectangular `eye(m, n)` + offset `k` (`sp.eye(2,3,k=1)` = `[[0,1,0],[0,0,1]]`) has no ferrolearn expression — `eye(n)` takes a single `usize`. Verification: `cargo test -p ferrolearn-sparse --lib helpers::tests::test_eye_basic` → green. (Consumer caveat: see REQ-CONSUMER — no estimator consumes `eye`; this REQ is SHIPPED on impl+oracle, but `eye` is a NEWLY-translated standalone helper, not a grandfathered boundary estimator type.) |
| REQ-DIAGS-SINGLE (single diagonal + alignment) | SHIPPED | impl `pub fn diags in helpers.rs` computes `(i,j) = (k, k+offset)` for `offset >= 0` and `(k+|offset|, k)` for `offset < 0`, matching scipy `diags([values], [offset], shape=(n,n))` (`_construct.py:445`) alignment. Live oracle (R-CHAR-3): main `diags([1,2,3],0,(3,3))` = `[[1,0,0],[0,2,0],[0,0,3]]`; super `diags([1,2],1,(3,3))` puts entries at `(0,1),(1,2)`; sub `diags([1,2],-1,(3,3))` at `(1,0),(2,1)`. Verification: `test_diags_main_diagonal` (`d[[0,0]]==1.0`, `d[[1,1]]==2.0`, `d[[2,2]]==3.0`), `test_diags_super_diagonal` (`d[[0,1]]==1.0`, `d[[1,2]]==2.0`) → green; the sub-diagonal branch is exercised by `test_hstack_basic`/`test_vstack_basic` building `diags([5,5],0,2)`. (Same standalone-helper consumer caveat as REQ-EYE.) |
| REQ-DIAGS-MULTI (list of diagonals + list of offsets) | NOT-STARTED | blocker issue to be filed by critic. ferrolearn `diags(values: &[T], offset: isize, n: usize)` takes a SINGLE diagonal + single offset; scipy `diags(diagonals, offsets)` (`_construct.py:445`) takes a *sequence of array_like* + *sequence of int*. Live oracle: `diags([[1,2,3],[4,5]],[0,1],(3,3)).toarray()` = `[[1,4,0],[0,2,5],[0,0,3]]` — no ferrolearn call expresses a multi-diagonal construction. API-shape divergence (R-DEV-2). |
| REQ-DIAGS-LENGTH-VALIDATION (ValueError on bad length vs silent skip) | NOT-STARTED | blocker issue to be filed by critic. **Headline divergence.** ferrolearn `diags in helpers.rs` SILENTLY SKIPS any out-of-bounds `(i,j)` via `if i < n && j < n` — `diags(&[1.,2.,3.],1,3)` quietly drops the third value and returns a 2-entry super-diagonal `Ok`, and `diags(&[1.,2.],0,3)` returns a 2-entry main diagonal `Ok`. scipy `diags` instead raises `ValueError` when a diagonal is too SHORT for its slot (oracle: `sp.diags([1.,2.],0,shape=(3,3))` → `ValueError: Diagonal length (index 0: 2 at offset 0) does not agree with array size (3, 3)`). Note scipy is asymmetric — it truncates a too-LONG diagonal silently (`sp.diags([1.,2.,3.],1,shape=(3,3))` → OK, 2 entries), so the silent-skip matches scipy ONLY for the over-long case, not the under-length case. R-DEV-1/2 (silent truncation vs `ValueError`). Likely single-file-fixable: add a length check in `diags` returning `FerroError::InvalidParameter` when `values.len() < min(n, n) - |offset|`, mirroring scipy's "does not agree with array size" path (critic to confirm scope). |
| REQ-HSTACK (horizontal CSR concat) | SHIPPED | impl `pub fn hstack in helpers.rs` validates equal rows, sums `n_cols()`, and re-bases each block's columns by a running `col_offset` into a `CooMatrix` → `CsrMatrix::from_coo`, mirroring scipy `hstack(blocks)` (`_construct.py:1012`). Live oracle (R-CHAR-3): `hstack([eye(2), diags([5,5],0,2)]).toarray()` = `[[1,0,5,0],[0,1,0,5]]`. Gap (documented): scipy's `format=`/`dtype=` params + mixed-format inputs are unsupported (ferrolearn takes `&[&CsrMatrix<T>]`). Verification: `test_hstack_basic` (`n_rows()==2`, `n_cols()==4`, `d[[0,2]]==5.0`); shape/empty guards return `FerroError`. (Standalone-helper consumer caveat as REQ-EYE.) |
| REQ-VSTACK (vertical CSR concat) | SHIPPED | impl `pub fn vstack in helpers.rs` validates equal columns, sums `n_rows()`, and re-bases each block's rows by a running `row_offset`, mirroring scipy `vstack(blocks)` (`_construct.py:1059`). Live oracle (R-CHAR-3): `vstack([eye(2), diags([5,5],0,2)]).toarray()` = `[[1,0],[0,1],[5,0],[0,5]]`. Same `format=`/`dtype=`/mixed gap as REQ-HSTACK. Verification: `test_vstack_basic` (`n_rows()==4`, `n_cols()==2`, `d[[2,0]]==5.0`); shape/empty guards return `FerroError`. (Standalone-helper consumer caveat as REQ-EYE.) |
| REQ-MISSING-HELPERS (identity/spdiags/bmat/block_diag/block_array/kron/kronsum/random/rand/tril/triu) | NOT-STARTED | blocker issue to be filed by critic. `helpers.rs` exposes only `eye`/`diags`/`hstack`/`vstack` (`grep -n "pub fn" helpers.rs`). scipy.sparse offers `identity` (`_construct.py:547`), `spdiags` (`:207`), `bmat` (`:1107`), `block_array` (`:1171`), `block_diag` (`:1313`), `kron` (`:728`), `kronsum` (`:843`), `random_array`/`random`/`rand` (`:1398`/`:1550`/`:1656`), `tril`/`triu` (`_extract.py:46`/`:113`) — none have a ferrolearn analog. `block_diag`/`kron` are used in some sklearn paths (block-kernel / graph-Laplacian construction), so this is honest NOT-STARTED, not out-of-scope. |
| REQ-CONSUMER (non-test cross-crate caller) | NOT-STARTED | blocker issue to be filed by critic. **Honest finding: NO estimator consumes these helpers.** `grep -rn "eye\|diags\|hstack\|vstack"` over `ferrolearn-*/src` (excluding `helpers.rs` and `#[cfg(test)]`) finds only the `lib.rs` re-export `pub use helpers::{diags, eye, hstack, vstack}` — no estimator/transformer calls them. (`ferrolearn-preprocess/src/column_transformer.rs::hstack` is a DIFFERENT, private, dense `Array2<f64>` helper; `Array2::eye`/`vstack` references in `backend_*.rs`/`sparse_eig.rs`/`incremental_pca.rs` are `ndarray`'s own methods or doc comments, not these functions.) Per R-DEFER-1, a newly-translated pub helper needs a non-test production consumer; these four are standalone with none. Unlike `CsrMatrix` (a grandfathered boundary type with real `ferrolearn-neighbors` consumers), `eye`/`diags`/`hstack`/`vstack` are not consumed by any estimator. NOT-STARTED until an estimator (or `ferrolearn-python` exposure of a scipy.sparse-construction surface) consumes them. |
| REQ-FERRAY (ferray sparse substrate) | NOT-STARTED | blocker issue to be filed by critic. `helpers.rs` builds on `crate::coo::CooMatrix` (`sprs::TriMat`-backed) and `crate::csr::CsrMatrix` (`sprs::CsMat`-backed), materializing through `ndarray::Array2` — the WRONG substrate per R-SUBSTRATE-1 (sparse → ferray's `scipy.sparse` analog, not `sprs`; dense → `ferray-core`, not `ndarray`). ferray does not yet expose a sparse construction surface (R-SUBSTRATE-5: a ferray gap is real work filed upstream to ferray; this REQ is NOT-STARTED until ferray ships the sparse layer). |

## Architecture

`helpers.rs` is four free functions over the crate's own sparse types — there is
no struct, no unfitted/Fitted split (these are construction utilities, mirroring
scipy.sparse's module-level functions, not classes). Each builds a `CooMatrix<T>`
incrementally and finishes through `CsrMatrix::from_coo`, so the helpers are
in-crate consumers of `coo.rs`/`csr.rs` (the construction pipeline the `csr.md`
REQ table cites).

- **`eye`** loops `0..n` pushing `(i, i, T::one())` (bound `T: Clone + One +
  Add<Output = T>`), mirroring scipy `eye(n)` (`_construct.py:678`) restricted to
  the square, main-diagonal case. The rectangular `m×n` + offset-`k` generality
  scipy supports is the gap (REQ-EYE SHIPPED for the square core, oracle `eye(3)`
  identity).

- **`diags`** enumerates `values` and places each at `(k, k+offset)` (super /
  main) or `(k+|offset|, k)` (sub), matching scipy `diags([values], [offset],
  shape=(n,n))` alignment exactly (REQ-DIAGS-SINGLE SHIPPED, oracle super at
  `(0,1),(1,2)` / sub at `(1,0),(2,1)`). Two divergences live here: (1) scipy's
  `diags` takes a LIST of diagonals + LIST of offsets, ferrolearn one of each
  (REQ-DIAGS-MULTI NOT-STARTED, oracle `diags([[1,2,3],[4,5]],[0,1])` =
  `[[1,4,0],[0,2,5],[0,0,3]]`); and (2) the `if i < n && j < n` guard SILENTLY
  SKIPS out-of-bounds positions where scipy raises `ValueError` on a too-short
  diagonal (REQ-DIAGS-LENGTH-VALIDATION NOT-STARTED — the headline divergence;
  scipy truncates a too-LONG diagonal silently but rejects a too-SHORT one, so
  the silent-skip is contract-matching only for over-long input). The
  length-validation gap is the single most likely fixer target: a length check in
  `diags` returning `FerroError::InvalidParameter` against scipy's "does not agree
  with array size" message is plausibly single-file.

- **`hstack`/`vstack`** validate equal rows (resp. columns), sum column widths
  (resp. row counts), and re-base each block's coordinates by a running
  `col_offset` (resp. `row_offset`) into one `CooMatrix` before `from_coo`,
  mirroring scipy `hstack`/`vstack` (`_construct.py:1012`/`:1059`)
  (REQ-HSTACK/REQ-VSTACK SHIPPED, oracle `[[1,0,5,0],[0,1,0,5]]` /
  `[[1,0],[0,1],[5,0],[0,5]]`). Empty input → `FerroError::InvalidParameter`;
  shape mismatch → `FerroError::ShapeMismatch`; scipy's `format=`/`dtype=` params
  and mixed-format/mixed-type inputs are the gap (ferrolearn takes
  `&[&CsrMatrix<T>]`).

The two cross-cutting structural facts are REQ-CONSUMER (NOT-STARTED — the
honest finding is that NO ferrolearn estimator consumes `eye`/`diags`/`hstack`/
`vstack`; the only non-test reference is the `lib.rs` re-export, and unlike
`CsrMatrix` these helpers are not grandfathered boundary types with real
downstream consumers, so R-DEFER-1's non-test-consumer requirement is unmet) and
REQ-FERRAY (NOT-STARTED — the helpers sit on `sprs`-backed `CooMatrix`/`CsrMatrix`
materializing through `ndarray`, the wrong substrate per R-SUBSTRATE-1; ferray has
no sparse layer yet, R-SUBSTRATE-5). The honest call (R-HONEST-3) is that the
square-identity / single-diagonal-with-alignment / hstack / vstack construction
core ships on impl + live-oracle, while the multi-diagonal API, the
length-validation contract, the broad missing-helper surface, the consumer
linkage, and the ferray substrate do not.

## Verification

Commands establishing the SHIPPED claims (run at baseline `29ea38b42`):

- `cargo test -p ferrolearn-sparse --lib helpers` → `test_eye_basic`,
  `test_diags_main_diagonal`, `test_diags_super_diagonal`, `test_hstack_basic`,
  `test_vstack_basic` pass, 0 failed.
- eye / diags-single oracle (REQ-EYE, REQ-DIAGS-SINGLE; R-CHAR-3 — expected from
  scipy, never from ferrolearn):
  `python3 -c "import scipy.sparse as sp; print(sp.eye(3).toarray().tolist(), sp.diags([1.,2.,3.],0,shape=(3,3)).toarray().tolist(), sp.diags([1.,2.],1,shape=(3,3)).toarray().tolist(), sp.diags([1.,2.],-1,shape=(3,3)).toarray().tolist())"`
  → `[[1,0,0],[0,1,0],[0,0,1]] [[1,0,0],[0,2,0],[0,0,3]] [[0,1,0],[0,0,2],[0,0,0]] [[0,0,0],[1,0,0],[0,2,0]]`.
  ferrolearn `eye(3)`/`diags(&[1.,2.,3.],0,3)`/`diags(&[1.,2.],1,3)`/
  `diags(&[1.,2.],-1,3)` match.
- hstack / vstack oracle (REQ-HSTACK, REQ-VSTACK):
  `python3 -c "import scipy.sparse as sp; a=sp.eye(2,format='csr'); b=sp.diags([5.,5.],0,shape=(2,2),format='csr'); print(sp.hstack([a,b]).toarray().tolist(), sp.vstack([a,b]).toarray().tolist())"`
  → `[[1,0,5,0],[0,1,0,5]] [[1,0],[0,1],[5,0],[0,5]]`. ferrolearn
  `hstack`/`vstack` match (`test_hstack_basic`, `test_vstack_basic`).
- multi-diagonal oracle (REQ-DIAGS-MULTI):
  `python3 -c "import scipy.sparse as sp; print(sp.diags([[1.,2.,3.],[4.,5.]],[0,1],shape=(3,3)).toarray().tolist())"`
  → `[[1.0,4.0,0.0],[0.0,2.0,5.0],[0.0,0.0,3.0]]`. ferrolearn `diags` has a
  single-diagonal signature — no expression for this. A critic pins a FAILING
  multi-diagonal test.
- length-validation oracle (REQ-DIAGS-LENGTH-VALIDATION — headline):
  `python3 -c "import scipy.sparse as sp; sp.diags([1.,2.],0,shape=(3,3))"`
  → `ValueError: Diagonal length (index 0: 2 at offset 0) does not agree with
  array size (3, 3).`; whereas `python3 -c "import scipy.sparse as sp;
  print(sp.diags([1.,2.,3.],1,shape=(3,3)).toarray().tolist())"` → OK
  `[[0,1,0],[0,0,2],[0,0,0]]` (too-long truncates). ferrolearn `diags(&[1.,2.],0,3)`
  returns `Ok` (silent short diagonal) and `diags(&[1.,2.,3.],1,3)` returns `Ok`
  (silent skip of the out-of-bounds third value). A critic pins a FAILING test
  requiring `diags(&[1.,2.],0,3)` to return `Err`.
- missing-helper oracle (REQ-MISSING-HELPERS):
  `python3 -c "import scipy.sparse as sp; print(sp.block_diag([sp.eye(2), sp.eye(1)]).toarray().tolist())"`
  → `[[1,0,0],[0,1,0],[0,0,1]]` runs in scipy; `grep -n "pub fn" helpers.rs`
  shows only `eye`/`diags`/`hstack`/`vstack`. A critic pins a FAILING test
  calling a missing helper.
- consumer check (REQ-CONSUMER):
  `grep -rn "eye\|diags\|hstack\|vstack" --include=*.rs /home/doll/ferrolearn/ferrolearn-*/src | grep -v 'helpers.rs' | grep -v '#\[cfg(test'`
  shows only `lib.rs` `pub use helpers::{diags, eye, hstack, vstack}` (the
  re-export) and the unrelated `column_transformer.rs::hstack` (a private dense
  `Array2<f64>` helper) / `ndarray`'s own `Array2::eye`/doc-comment `vstack` —
  NO estimator calls `ferrolearn_sparse::{eye,diags,hstack,vstack}`. The
  non-test-consumer requirement is unmet.
- ferray-substrate check (REQ-FERRAY): `helpers.rs` imports
  `crate::coo::CooMatrix` + `crate::csr::CsrMatrix` (both `sprs`-backed,
  materializing through `ndarray::Array2`); the destination is ferray's
  `scipy.sparse` construction analog (R-SUBSTRATE-1). ferray has no sparse layer
  yet (R-SUBSTRATE-5).
