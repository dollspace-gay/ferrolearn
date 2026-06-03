# Neighbor Graph Constructors (sklearn.neighbors graph functions)

<!--
tier: 3-component
status: draft
baseline-commit: 2013942c
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/neighbors/_graph.py     # _query_include_self (:34); kneighbors_graph (:59); radius_neighbors_graph (:164); KNeighborsTransformer (:259); RadiusNeighborsTransformer (:492)
  - sklearn/neighbors/_base.py      # sort_graph_by_row_values (:201); _is_sorted_by_data; _kneighbors_from_graph (:292)
ferrolearn-module: ferrolearn-neighbors/src/graph.rs
parity-ops: kneighbors_graph, radius_neighbors_graph, sort_graph_by_row_values
crosslink-issue: 822
-->

## Summary

`ferrolearn-neighbors/src/graph.rs` mirrors the free graph constructors of
scikit-learn's `sklearn/neighbors/_graph.py` — `kneighbors_graph` (`:59`) and
`radius_neighbors_graph` (`:164`) — plus the `sort_graph_by_row_values` utility
that lives in `sklearn/neighbors/_base.py` (`:201`). All three return CSR sparse
matrices via `ferrolearn_sparse::CsrMatrix`. The module also hangs `kneighbors_graph`
/ `radius_neighbors_graph` method-form inherent impls off the fitted neighbor
estimators (mirroring `KNeighborsMixin.kneighbors_graph` /
`RadiusNeighborsMixin.radius_neighbors_graph`).

Under honest underclaim (R-HONEST-3), **the two headline free functions diverge
from the live sklearn 1.5.2 oracle on deterministic, value-level contracts** —
not edge cases. The two that a critic must pin first:

1. **`include_self` is absent, and ferrolearn's de-facto default is the OPPOSITE
   of sklearn's.** sklearn `kneighbors_graph(X, n_neighbors)` defaults
   `include_self=False` and, via `_query_include_self` (`:34`), passes `X=None`
   to the query so **each sample's own row is excluded** — the graph has a zero
   diagonal. ferrolearn's free `kneighbors_graph` builds a `NearestNeighbors`
   index over `x` and queries `x` against itself with no self-exclusion, so the
   nearest neighbor of every point is **itself at distance 0** — a `1` (or a
   `0.0` distance edge) on the diagonal. The results differ on the very first
   example in sklearn's own docstring.
2. **`sort_graph_by_row_values` sorts by the WRONG key.** sklearn sorts each row
   by ascending **stored value (distance)** (`_base.py:271-289`, `np.argsort` over
   `graph.data`). ferrolearn's `sort_graph_by_row_values` rebuilds the matrix in
   ascending **column-index** order (it densifies then re-emits left-to-right by
   column). For a distance graph whose columns are not already in distance order,
   the two produce different `data`/`indices` layouts.

On top of those, the module is **euclidean-only** (no `metric`/`p`/`metric_params`),
exposes **no `n_jobs`**, and `ferrolearn_sparse::CsrMatrix` is built on **`sprs`**,
not a ferray sparse analog (R-SUBSTRATE). The `mode` contract and the per-row
CSR-sorted-by-column structural invariant ARE matched (for the rows ferrolearn
actually emits).

## Algorithm (sklearn — the contract)

### `kneighbors_graph(X, n_neighbors, *, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False, n_jobs=None)` (`_graph.py:59`)

1. If `X` is not already a fitted `KNeighborsMixin`, fit a
   `NearestNeighbors(n_neighbors, metric, p, metric_params, n_jobs)` on it
   (`:136-143`).
2. `query = _query_include_self(X._fit_X, include_self, mode)` (`:147`). Because
   `_query_include_self` (`:34`) returns `None` when `include_self` is falsy (the
   default), **the kneighbors query runs with `X=None`**, which sklearn's
   `KNeighborsMixin.kneighbors` treats as "query the training data but exclude
   each sample's own row" — the diagonal is suppressed.
3. Return `X.kneighbors_graph(X=query, n_neighbors=n_neighbors, mode=mode)`
   (`:148`), a `scipy.sparse.csr_matrix` of shape `(n_samples, n_samples)`,
   indices sorted ascending by column within each row.

`mode` (`:82-85`): `'connectivity'` (default) stores `1.0` at every edge;
`'distance'` stores the metric distance. `include_self` (`:105-108`): `bool` or
`'auto'`; `'auto'` → `True` for `mode='connectivity'`, `False` for
`mode='distance'`.

**Oracle (live, run from `/tmp`), `X=[[0,0],[1,1],[2,2]]`:**
```
kneighbors_graph(X, 1, mode='connectivity').toarray()       # default include_self=False
[[0. 1. 0.]      <- point 0's nearest OTHER point is 1 (no diagonal)
 [1. 0. 0.]
 [0. 1. 0.]]     indptr=[0,1,2,3] indices=[1,0,1] data=[1,1,1]
kneighbors_graph(X, 2, mode='connectivity').toarray()       # zero diagonal
[[0. 1. 1.] [1. 0. 1.] [1. 1. 0.]]
kneighbors_graph(X, 2, mode='connectivity', include_self=True).toarray()
[[1. 1. 0.] [1. 1. 0.] [0. 1. 1.]]    <- diagonal present
kneighbors_graph(X, 1, mode='distance').toarray()           # sqrt(2) edges, zero diagonal
[[0. 1.41421356 0.] [1.41421356 0. 0.] [0. 1.41421356 0.]]
```

### `radius_neighbors_graph(X, radius, *, mode='connectivity', ..., include_self=False, ...)` (`_graph.py:164`)

Same structure: fit `NearestNeighbors(radius=…)`, then
`query = _query_include_self(X._fit_X, include_self, mode)` (`:255`) and
`X.radius_neighbors_graph(query, radius, mode)` (`:256`). Default
`include_self=False` ⇒ query `X=None` ⇒ **the self-edge (distance 0, always
within any radius) is excluded**. Output `(n_samples, n_samples)` CSR.

**Oracle, `X=[[0,0],[1,1],[2,2]]`, `radius=1.5`:**
```
radius_neighbors_graph(X, 1.5, mode='connectivity').toarray()    # default exclude-self
[[0. 1. 0.] [1. 0. 1.] [0. 1. 0.]]
radius_neighbors_graph(X, 1.5, mode='connectivity', include_self=True).toarray()
[[1. 1. 0.] [1. 1. 1.] [0. 1. 1.]]
radius_neighbors_graph(X, 1.5, mode='distance').toarray()        # zero diagonal
[[0. 1.41421356 0.] [1.41421356 0. 1.41421356] [0. 1.41421356 0.]]
```

### `sort_graph_by_row_values(graph, copy=False, warn_when_not_sorted=True)` (`_base.py:201`)

Sorts each CSR row so its **stored values (distances) are ascending**, permuting
`data` AND `indices` together by `np.argsort(graph.data[start:stop])`
(`:271-289`). When already sorted by data it returns the input unchanged
(`:242-243`); otherwise, with `warn_when_not_sorted=True`, raises an
`EfficiencyWarning` (`:245-254`). Non-CSR input requires `copy=True`
(`:256-267`). Used for precomputed sparse distance graphs.

**Oracle:** sklearn's own docstring (`:236-240`): a row stored `data=[3.,1.]` →
sorted `data=[1.,3.]` with `indices` permuted to match. Live (a 2-row CSR with
row 0 `data=[0.5,0.1]` at `indices=[2,1]`):
```
sort_graph_by_row_values(m, copy=True, warn_when_not_sorted=False)
input  data=[0.5,0.1,0.3] indices=[2,1,0]
sorted data=[0.1,0.5,0.3] indices=[1,2,0]    <- by VALUE, not by column
```

## ferrolearn (what exists)

All public items live in `ferrolearn-neighbors/src/graph.rs`, generic over
`F: Float + Send + Sync + 'static`, returning `Result<CsrMatrix<F>, FerroError>`:

- **`pub enum GraphMode { Connectivity, Distance }`** — mirrors sklearn's `mode`
  string. `Connectivity` stores `F::one()`, `Distance` stores the actual
  distance (`knn_to_csr` / `radius_to_csr` match arms). There is **no `'auto'`
  variant** (relevant only to `include_self='auto'`, which is also absent), and
  **no default** — `mode` is a required positional argument.
- **`pub fn kneighbors_graph(x, n_neighbors, mode)`** — fits a transient
  `NearestNeighbors::new().with_n_neighbors(n_neighbors)` over `x`, then
  `nn.kneighbors(x, Some(n_neighbors))` **against `x` itself with no
  self-exclusion**, then `knn_to_csr`. No `include_self`, no `metric`/`p`/
  `metric_params`, no `n_jobs`.
- **`pub fn radius_neighbors_graph(x, radius, mode)`** — fits a
  `RadiusNeighborsClassifier::new().with_radius(radius)` over `x` (with a dummy
  `y` of zeros), then `fitted.radius_neighbors(x, Some(radius))` against `x`,
  then `radius_to_csr`. The `radius_to_csr` helper **deduplicates repeated
  columns** but does **not** suppress the self-edge. No `include_self`/`metric`/
  `p`/`n_jobs`.
- **`pub fn sort_graph_by_row_values(graph)`** — densifies `graph.to_dense()`
  and re-emits non-zeros **in ascending column order** per row. Sorts by
  **column index**, not by stored value; takes **no `copy`/`warn_when_not_sorted`
  arguments** and emits no `EfficiencyWarning`.

**Private CSR builders:**
- **`pub(crate) fn knn_to_csr(distances, indices, n_rows, n_cols, mode)`** —
  builds a `(col, value)` buffer per row, `sort_by_key` on the column, pushes
  into `indptr`/`col_indices`/`data`. **This correctly produces per-row
  ascending-column CSR indices** (the structural invariant scipy/sklearn require).
- **`pub(crate) fn radius_to_csr(distances, indices, n_rows, n_cols, mode)`** —
  same, over jagged `Vec<Vec<…>>`, additionally **deduplicating** adjacent equal
  columns after the column sort.

**Method-form inherent impls** (peers of the free functions, mirroring the
mixin methods — these are themselves pub API, not consumers of the free
functions): `FittedNearestNeighbors::{kneighbors_graph, radius_neighbors_graph}`,
`FittedKNeighborsClassifier::kneighbors_graph`,
`FittedKNeighborsRegressor::kneighbors_graph`,
`FittedRadiusNeighborsClassifier::radius_neighbors_graph`,
`FittedRadiusNeighborsRegressor::radius_neighbors_graph`. Each delegates to a
`kneighbors`/`radius_neighbors` query against `x` then `knn_to_csr`/`radius_to_csr`
with `n_cols = self.n_samples_fit()` — i.e. they query the **fitted training
data** and likewise **do not exclude the query's own row** when `x` is the
training data (no `include_self`).

**The underlying query has no self-exclusion.** `NearestNeighbors::kneighbors`
(`nearest_neighbors.rs`) calls `find_knn(&query, k)` and returns the `k` closest
training rows including any row identical to the query — there is no
`_query_include_self`-equivalent path anywhere in the crate.

**Metric.** The crate is **euclidean-only**: `find_knn` →
`kdtree::euclidean_distance` (`nearest_neighbors.rs`). There is no
`metric`/`p`/`metric_params` parameter; sklearn's `'minkowski'`, `p=2` default
coincides numerically with euclidean, but `p=1`/other metrics are inexpressible.

**Consumers (non-test).** The only non-test production consumer is the crate-root
re-export in `ferrolearn-neighbors/src/lib.rs`:
`pub use graph::{GraphMode, kneighbors_graph, radius_neighbors_graph,
sort_graph_by_row_values};`. These are existing pub APIs (grandfathered,
S5/R-DEFER-1); the re-export is the production-consumer surface. **No
`ferrolearn-python` binding** exposes the graph functions (a binding gap), and
**there is no `KNeighborsTransformer`/`RadiusNeighborsTransformer`** estimator
(the routed `parity-ops` list these; the module ships only the free-function form).

**Sparse substrate.** `CsrMatrix` comes from `ferrolearn-sparse`, whose
`Cargo.toml` depends on **`sprs`** + `ndarray` — it is NOT a ferray sparse
analog. R-SUBSTRATE-1 names "sparse → ferray's sparse analog (not `sprs`)" as the
destination; `graph.rs` also imports `ndarray::{Array1, Array2}` and
`num_traits::Float` directly rather than `ferray-core`.

## Requirements

- REQ-1: **`kneighbors_graph` value parity (R-DEV-1/3).** Match
  `kneighbors_graph(X, n_neighbors, *, mode='connectivity', include_self=False)`
  (`_graph.py:59,147-148`): default **excludes each sample's own row** (zero
  diagonal), `mode` ∈ {connectivity→1.0, distance→metric distance}, output CSR
  `(n_samples, n_samples)` with indices sorted ascending by column.
- REQ-2: **`radius_neighbors_graph` value parity (R-DEV-1/3).** Match `:164,255-256`:
  default **excludes the self-edge** (distance-0-to-self within any radius),
  `mode` semantics, CSR `(n_samples, n_samples)`.
- REQ-3: **`sort_graph_by_row_values` parity (R-DEV-1).** Match `_base.py:201`:
  sort each row by **ascending stored value (distance)**, permuting `data` and
  `indices` together — NOT by column index; `copy`/`warn_when_not_sorted`
  parameters + `EfficiencyWarning` on unsorted input.
- REQ-4: **`GraphMode` / `mode` contract (R-DEV-2).** `Connectivity`→`1.0`,
  `Distance`→distance matches sklearn `mode` (`:82-85`). (The connectivity/distance
  *value mapping itself* matches; the missing default-string and `'auto'`
  interaction are folded into REQ-5.)
- REQ-5: **`include_self` parameter (R-DEV-1/2) — HEADLINE.** sklearn exposes
  `include_self: bool | 'auto'`, default `False`, routed through
  `_query_include_self` (`:34`) so the default **excludes** the self row.
  ferrolearn exposes **no `include_self` parameter** and its de-facto behavior is
  the **opposite** of sklearn's default (it includes self). This is the primary
  value divergence on both `kneighbors_graph` and `radius_neighbors_graph`.
- REQ-6: **`metric`/`p`/`metric_params` parameters (R-DEV-2).** sklearn defaults
  `metric='minkowski'`, `p=2` and accepts other metrics (`:64-66`). ferrolearn is
  euclidean-only with no metric parameter.
- REQ-7: **CSR sorted-indices structural contract (R-DEV-3).** Output rows must
  have indices ascending by column (scipy/sklearn CSR invariant). `knn_to_csr`/
  `radius_to_csr` already do this via per-row `sort_by_key(col)`.
- REQ-8: **`n_jobs` parameter (R-DEV-2).** sklearn exposes `n_jobs` (`:68`);
  ferrolearn does not.
- REQ-9: **`KNeighborsTransformer` / `RadiusNeighborsTransformer` (R-DEV-2/3).**
  The routed `parity-ops` include these estimator classes (`_graph.py:259,492`);
  ferrolearn ships only the free-function form, with no transformer estimator
  (`mode` default `'distance'`, the `n_neighbors+1` "extra neighbor" rule
  `:455-457`, `fit_transform`).
- REQ-10: **PyO3 binding (R-DEFER-1).** `import sklearn.neighbors` exposes
  `kneighbors_graph`/`radius_neighbors_graph`/`sort_graph_by_row_values`;
  `ferrolearn-python` exposes no shim.
- REQ-11: **ferray substrate (R-SUBSTRATE).** `graph.rs` uses `sprs`-backed
  `ferrolearn_sparse::CsrMatrix`, `ndarray::{Array1,Array2}`, and `num_traits::Float`,
  not ferray's sparse analog / `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (run from `/tmp`),
never literal-copied from ferrolearn (R-CHAR-3). `X=[[0,0],[1,1],[2,2]]` unless
noted.

- AC-1 (REQ-1/5 headline pin): `kneighbors_graph(X, 1, mode='connectivity').toarray()`
  must equal sklearn `[[0,1,0],[1,0,0],[0,1,0]]` (zero diagonal — self excluded).
  ferrolearn (self included) yields a diagonal `1` at the nearest-neighbor slot
  and FAILS.
- AC-2 (REQ-1 distance): `kneighbors_graph(X, 1, mode='distance').toarray()` must
  equal sklearn `[[0, √2, 0],[√2, 0, 0],[0, √2, 0]]` (`√2 = 1.4142135623730951`).
  ferrolearn's nearest is self at distance 0.0 → wrong row content.
- AC-3 (REQ-5 include_self toggle): `kneighbors_graph(X, 2, mode='connectivity',
  include_self=True).toarray()` must equal sklearn `[[1,1,0],[1,1,0],[0,1,1]]`.
  ferrolearn cannot express `include_self`.
- AC-4 (REQ-2 radius pin): `radius_neighbors_graph(X, 1.5, mode='connectivity').toarray()`
  must equal sklearn `[[0,1,0],[1,0,1],[0,1,0]]` (self-edge excluded). ferrolearn
  (after self-dedup) still includes the self-edge → diagonal divergence.
- AC-5 (REQ-3 sort pin): a CSR row stored `data=[0.5,0.1]` at `indices=[2,1]`,
  after `sort_graph_by_row_values`, must become `data=[0.1,0.5]` at
  `indices=[1,2]` (ascending by VALUE, sklearn `_base.py:271-289`). ferrolearn
  sorts by column → `data=[0.1,0.5]` at `indices=[1,2]` only coincidentally when
  value-order equals column-order; for `data=[3.,1.]` at `indices=[1,2]` sklearn
  gives `data=[1.,3.] indices=[2,1]` while ferrolearn returns `data=[3.,1.]
  indices=[1,2]` unchanged → FAILS.
- AC-6 (REQ-7 structural, must stay green): for any ferrolearn-emitted graph,
  each row's `indices` slice (per `indptr`) is strictly ascending — this invariant
  IS held by `knn_to_csr`/`radius_to_csr`.
- AC-7 (REQ-4 mode value, must stay green): `GraphMode::Distance` edges carry the
  euclidean distance and `GraphMode::Connectivity` edges carry `1.0` — the
  value-mapping matches sklearn `mode` for the edges ferrolearn does emit.

## REQ status table

Binary (R-DEFER-2). The free functions and `GraphMode` are existing pub APIs
re-exported at the crate root (`ferrolearn-neighbors/src/lib.rs` — the non-test
production-consumer surface; grandfathered per S5/R-DEFER-1). Cites use symbol
anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed
sklearn 1.5.2, run from `/tmp`. Honest underclaim (R-HONEST-3).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`kneighbors_graph` value) | NOT-STARTED | open prereq blocker #823. `pub fn kneighbors_graph in graph.rs` queries `nn.kneighbors(x, …)` against `x` itself with **no self-exclusion**; sklearn default `include_self=False` excludes the own row (`_graph.py:59,147` via `_query_include_self:34`). Pin: `kneighbors_graph([[0,0],[1,1],[2,2]],1,'connectivity').toarray()` → sklearn `[[0,1,0],[1,0,0],[0,1,0]]`, ferro emits a diagonal edge. |
| REQ-2 (`radius_neighbors_graph` value) | NOT-STARTED | open prereq blocker #824. `pub fn radius_neighbors_graph in graph.rs` queries `x` against itself; `radius_to_csr` dedups columns but does **not** drop the self-edge; sklearn default excludes it (`_graph.py:164,255`). Pin: `radius_neighbors_graph(X,1.5,'connectivity').toarray()` → sklearn `[[0,1,0],[1,0,1],[0,1,0]]` (zero diagonal), ferro includes self. |
| REQ-3 (`sort_graph_by_row_values`) | NOT-STARTED | open prereq blocker #825. `pub fn sort_graph_by_row_values in graph.rs` re-emits non-zeros in ascending **column** order; sklearn sorts by ascending **stored value** permuting `data`+`indices` (`_base.py:201,271-289`). Pin: row `data=[3.,1.]`@`indices=[1,2]` → sklearn `data=[1.,3.]`@`indices=[2,1]`, ferro returns it unchanged. No `copy`/`warn_when_not_sorted`/`EfficiencyWarning`. |
| REQ-4 (`GraphMode`/`mode` value mapping) | SHIPPED | impl `pub enum GraphMode in graph.rs` (`Connectivity`→`F::one()`, `Distance`→actual distance) mirrors sklearn `mode` value semantics (`_graph.py:82-85`: connectivity = ones, distance = metric distances). The connectivity-1.0 / distance-edge mapping matches the oracle (`kneighbors_graph(X,2,'distance')` edges = √2; connectivity edges = 1.0). Non-test consumer: re-exported `pub use graph::GraphMode` in `lib.rs`. NOTE: this REQ covers ONLY the edge value-mapping; the missing default string and `include_self='auto'` interaction are REQ-5. Least-confident SHIPPED in this doc. |
| REQ-5 (`include_self` — HEADLINE) | NOT-STARTED | open prereq blocker #826. **No `include_self` parameter exists** on `kneighbors_graph`/`radius_neighbors_graph`, and ferrolearn's de-facto default (self INCLUDED) is the **opposite** of sklearn's default `include_self=False` (self EXCLUDED, `_graph.py:67,147` / `:172,255` via `_query_include_self:34-43`). Also no `'auto'` (connectivity→True, distance→False). This is the primary divergence underlying REQ-1/REQ-2. |
| REQ-6 (`metric`/`p`/`metric_params`) | NOT-STARTED | open prereq blocker #827. Crate is euclidean-only (`find_knn` → `kdtree::euclidean_distance` in `nearest_neighbors.rs`); sklearn defaults `metric='minkowski'`, `p=2` and accepts other metrics (`_graph.py:64-66`). `p=2` coincides with euclidean, but `p=1`/other metrics are inexpressible. |
| REQ-7 (CSR sorted-by-column structure) | SHIPPED | impl `pub(crate) fn knn_to_csr in graph.rs` builds per-row `(col,val)` pairs then `row_pairs.sort_by_key(\|(c,_)\| *c)` before pushing — producing CSR `indices` ascending by column per row (the scipy/sklearn CSR invariant; sklearn returns sorted CSR `_graph.py:148`). `radius_to_csr` does the same + dedups. Non-test consumer: `pub fn kneighbors_graph` / `pub fn radius_neighbors_graph` (re-exported in `lib.rs`) emit through these helpers. This structural invariant holds for every row ferrolearn emits (independent of the self-row divergence). |
| REQ-8 (`n_jobs`) | NOT-STARTED | open prereq blocker #828. No `n_jobs` parameter; sklearn exposes it (`_graph.py:68,173`). |
| REQ-9 (`KNeighborsTransformer`/`RadiusNeighborsTransformer`) | NOT-STARTED | open prereq blocker #829. Routed `parity-ops` list both transformer estimators (`_graph.py:259,492`); ferrolearn ships only the free-function form — no transformer struct, no `mode='distance'` default, no `n_neighbors+1` extra-neighbor rule (`:455-457`), no `fit_transform`. |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #830. `ferrolearn-python` exposes no graph-function shim; `import ferrolearn.neighbors` cannot call what `import sklearn.neighbors` provides. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #831. `graph.rs` uses `sprs`-backed `ferrolearn_sparse::CsrMatrix` + `ndarray::{Array1,Array2}` + `num_traits::Float`, not ferray's sparse analog / `ferray-core` (R-SUBSTRATE-1). |

## Architecture

`graph.rs` is a flat module of free functions plus method-form inherent impls,
generic over `F: Float + Send + Sync + 'static`, returning
`Result<CsrMatrix<F>, FerroError>`. There are no fitted/unfitted graph types — the
free functions are stateless constructors that fit a transient neighbor index
internally.

**Two free constructors, one utility, two CSR builders:**

1. **`kneighbors_graph`** fits `NearestNeighbors::new().with_n_neighbors(n).fit(x)`
   then `nn.kneighbors(x, Some(n))` and hands the `(distances, indices)` to
   `knn_to_csr`. The query target is `x` itself; with no `_query_include_self`
   analog, the nearest neighbor of each row is the row itself at distance 0 — the
   **self-row divergence** (REQ-1/5). sklearn's `kneighbors_graph` passes `X=None`
   to the query by default to suppress exactly this.
2. **`radius_neighbors_graph`** fits a `RadiusNeighborsClassifier` (with a dummy
   zero `y`) and queries `fitted.radius_neighbors(x, Some(radius))`. `radius_to_csr`
   sorts by column and **dedups repeated columns** (a guard for near-duplicate
   returns), but never removes the self-edge — so the diagonal (distance 0,
   always within radius) survives (REQ-2/5).
3. **`sort_graph_by_row_values`** densifies the input (`graph.to_dense()`) and
   re-emits non-zeros left-to-right by column. This sorts by **column index**, a
   structurally valid CSR but the wrong sort key — sklearn sorts by **stored
   value** (REQ-3). It also drops the `copy`/`warn_when_not_sorted` parameters and
   the `EfficiencyWarning`.
4. **`knn_to_csr` / `radius_to_csr`** are the shared CSR assemblers. Both build a
   per-row `(col, value)` buffer, `sort_by_key` on the column, and append to
   `indptr`/`col_indices`/`data`, so the emitted CSR **always has per-row
   ascending-column indices** (REQ-7, the one structural invariant that fully
   matches). `value` is `F::one()` for `Connectivity` and the distance for
   `Distance` (REQ-4 value mapping — matches).

**Method-form impls** (`Fitted*::{kneighbors_graph, radius_neighbors_graph}`)
mirror `KNeighborsMixin`/`RadiusNeighborsMixin`. They query the **fitted training
data** with `n_cols = self.n_samples_fit()` and route through the same CSR
builders. They are pub-API peers (mirroring the mixin methods), not consumers of
the free functions, and inherit the same no-`include_self` self-row behavior when
`x` is the training data.

**Invariants held vs sklearn:** `mode` value mapping (connectivity→1.0,
distance→distance, REQ-4/AC-7); per-row CSR `indices` ascending by column
(REQ-7/AC-6); output shape `(n_rows, n_cols)`; euclidean distances numerically
equal sklearn's `minkowski` `p=2`. **Invariants NOT held vs sklearn:**
self-row exclusion (the default `include_self=False`, REQ-1/2/5); `sort_graph_by_row_values`
sort key (value vs column, REQ-3); `metric`/`p`/`metric_params` (REQ-6); `n_jobs`
(REQ-8); the transformer estimators (REQ-9); the PyO3 binding (REQ-10); the
ferray/non-`sprs` substrate (REQ-11).

## Verification

Library crate (green at baseline `2013942c` for the existing — narrower —
contract):
```
cargo test -p ferrolearn-neighbors
cargo clippy -p ferrolearn-neighbors --all-targets -- -D warnings
cargo fmt --all --check
```
The existing `tests/api_proof.rs` exercises `kneighbors_graph`/
`radius_neighbors_graph`/`sort_graph_by_row_values` only for type/shape surface
(it does not compare row contents against the sklearn oracle), so it does NOT
establish parity and makes no value REQ SHIPPED. REQ-4 and REQ-7 are SHIPPED on
the strength of the edge-value mapping and the CSR sorted-by-column structural
invariant respectively (both independent of the self-row divergence).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the deterministic
divergences a critic should pin first (R-CHAR-3 expected values), all with
`X=[[0,0],[1,1],[2,2]]`:
```
# REQ-1/5 HEADLINE (kneighbors self-exclusion): sklearn zero diagonal vs ferro diagonal edge
python3 -c "from sklearn.neighbors import kneighbors_graph; print(kneighbors_graph([[0,0],[1,1],[2,2]],1,mode='connectivity').toarray())"   # [[0,1,0],[1,0,0],[0,1,0]]
python3 -c "from sklearn.neighbors import kneighbors_graph; print(kneighbors_graph([[0,0],[1,1],[2,2]],1,mode='distance').toarray())"       # [[0,√2,0],[√2,0,0],[0,√2,0]]
python3 -c "from sklearn.neighbors import kneighbors_graph; print(kneighbors_graph([[0,0],[1,1],[2,2]],2,mode='connectivity',include_self=True).toarray())"  # [[1,1,0],[1,1,0],[0,1,1]]
# REQ-2 (radius self-exclusion): sklearn zero diagonal vs ferro self-edge
python3 -c "from sklearn.neighbors import radius_neighbors_graph; print(radius_neighbors_graph([[0,0],[1,1],[2,2]],1.5,mode='connectivity').toarray())"  # [[0,1,0],[1,0,1],[0,1,0]]
# REQ-3 (sort by VALUE not column): sklearn permutes data+indices by distance
python3 -c "
import numpy as np, scipy.sparse as sp
from sklearn.neighbors import sort_graph_by_row_values
m=sp.csr_matrix((np.array([3.,1.]),np.array([1,2]),np.array([0,2])),shape=(1,3))
out=sort_graph_by_row_values(m,copy=True,warn_when_not_sorted=False)
print(out.data.tolist(), out.indices.tolist())"   # [1.0, 3.0] [2, 1]  (ferro: [3.0,1.0] [1,2])
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-neighbors/tests/divergence_graph.rs`, asserting the live-sklearn
expected values above and FAILING against current `graph.rs`. **Cleanest
deterministic divergences to pin first:** (1) `kneighbors_graph` self-exclusion
(AC-1, the headline — affects every default call), (2) `radius_neighbors_graph`
self-edge (AC-4), (3) `sort_graph_by_row_values` value-vs-column sort key (AC-5).
All three are fully deterministic, single-call, and need no `sample_weight`/
`random_state`.

## Blockers to open

- #823 — REQ-1 (`kneighbors_graph` value): no self-exclusion; sklearn default
  `include_self=False` excludes own row (`_graph.py:59,147`). Pin:
  `kneighbors_graph([[0,0],[1,1],[2,2]],1,'connectivity')` → sklearn zero diagonal.
- #824 — REQ-2 (`radius_neighbors_graph` value): self-edge not dropped; sklearn
  default excludes it (`_graph.py:164,255`). Pin: `radius_neighbors_graph(X,1.5,
  'connectivity')` → sklearn `[[0,1,0],[1,0,1],[0,1,0]]`.
- #825 — REQ-3 (`sort_graph_by_row_values`): sorts by column not by stored value
  (`_base.py:201,271-289`); no `copy`/`warn_when_not_sorted`/`EfficiencyWarning`.
  Pin: row `data=[3.,1.]`@`[1,2]` → sklearn `data=[1.,3.]`@`[2,1]`.
- #826 — REQ-5 (`include_self` HEADLINE): absent parameter; de-facto default is
  the OPPOSITE of sklearn's `include_self=False` (`_graph.py:34-43,67,172`); no
  `'auto'`. Underlies #823/#824.
- #827 — REQ-6 (`metric`/`p`/`metric_params`): euclidean-only; sklearn
  `minkowski`/`p=2` default + other metrics (`_graph.py:64-66`).
- #828 — REQ-8 (`n_jobs`): no `n_jobs` parameter (`_graph.py:68,173`).
- #829 — REQ-9: no `KNeighborsTransformer`/`RadiusNeighborsTransformer` estimator
  (`_graph.py:259,492`) — routed in `parity-ops`, unshipped.
- #830 — REQ-10: no `ferrolearn-python` graph-function binding.
- #831 — REQ-11: migrate `graph.rs` off `sprs`-backed `CsrMatrix` + `ndarray` +
  `num-traits` to the ferray sparse/array substrate (R-SUBSTRATE-1).
