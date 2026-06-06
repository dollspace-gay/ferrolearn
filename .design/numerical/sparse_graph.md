# scipy.sparse.csgraph graph algorithms

<!--
tier: 3-component
status: draft
baseline-commit: 1aa756a40
upstream-paths:
  - scipy/sparse/csgraph/__init__.py   # documented csgraph function surface + signatures
-->

## Summary

`ferrolearn-numerical/src/sparse_graph.rs` is the scipy-analog substrate that
mirrors a slice of **scipy.sparse.csgraph**, operating on sparse CSR adjacency /
weight matrices (`sprs::CsMat<f64>`). It implements four routines:
`dijkstra` (single-source shortest paths via a `BinaryHeap` min-heap),
`dijkstra_all_pairs` (repeated single-source Dijkstra mirroring
`scipy.sparse.csgraph.shortest_path`/`dijkstra(indices=None)`),
`connected_components` (undirected/weak BFS labelling), and
`minimum_spanning_tree` (Kruskal with union-find). The numbers it computes —
shortest-path *distances* and component *labels* — match the live oracle
(**scipy 1.17.1**); graph algorithms are deterministic, so the scipy-1.17.1 /
sklearn-1.5.2 version split is irrelevant (a shortest path is a shortest path in
every scipy release). What diverges is the *output representation* and the *API
surface*.

Divergence classes:
1. **MST-representation (the headline, deterministic, FIXABLE)** —
   `minimum_spanning_tree` stores BOTH directions per edge, returning a symmetric
   CSR with `nnz = 2·n_edges` and entry-sum `= 2·(MST weight)`;
   `scipy.sparse.csgraph.minimum_spanning_tree` returns ONE entry per edge,
   upper-triangular (`i < j`), `nnz = n_edges`, sum `= MST weight`.
2. **signature/flags** — `dijkstra(graph, source)` is single-source returning a
   struct; scipy's `dijkstra` / `connected_components` carry flags
   (`directed`, `unweighted`, `limit`, `min_only`, `indices=None`,
   `connection='strong'`) that ferrolearn hardcodes or omits.
3. **predecessor-sentinel** — `usize::MAX` vs scipy's `-9999` (`int32`).
4. **missing-functions** — `shortest_path`, `floyd_warshall`, `bellman_ford`,
   `johnson`, `yen`, `breadth_first_order`/`depth_first_order`, `…_tree`,
   `laplacian`, `reconstruct_path`, `reverse_cuthill_mckee`, `maximum_flow`,
   `maximum_bipartite_matching`, `structural_rank` — all absent.
5. **error-type** — `Result<_, String>` rather than `Result<_, FerroError>`.
6. **no-consumer** — `pub mod sparse_graph` with no non-test caller; `Isomap`
   reimplements its own private `dijkstra`.
7. **ferray-substrate** — `sprs::CsMat` + `ndarray` rather than ferray's sparse /
   csgraph analog (R-SUBSTRATE-1).

## Upstream reference (scipy.sparse.csgraph, live oracle scipy 1.17.1)

The documented function surface is in `scipy/sparse/csgraph/__init__.py:15-44`
(the `autosummary` list: `connected_components`, `laplacian`, `shortest_path`,
`dijkstra`, `floyd_warshall`, `bellman_ford`, `johnson`, `yen`,
`breadth_first_order`, `depth_first_order`, `breadth_first_tree`,
`depth_first_tree`, `minimum_spanning_tree`, `reverse_cuthill_mckee`,
`maximum_flow`, `maximum_bipartite_matching`, `structural_rank`,
`reconstruct_path`, `NegativeCycleError`). The kernels are Cython, so cite the
csgraph **function names** and the **live-oracle values**, never `.pyx` line
numbers. An undirected graph is "represented by a symmetric matrix"
(`__init__.py:79`); non-edges are non-entries in the sparse representation
(`__init__.py:61-63`).

Live oracle (`cd /tmp && python3 -c "..."`, scipy 1.17.1), on the 4-node graph
`G = [[0,1,0,4],[1,0,2,0],[0,2,0,3],[4,0,3,0]]`:

- `minimum_spanning_tree(G).toarray()` →
  `[[0,1,0,0],[0,0,2,0],[0,0,0,3],[0,0,0,0]]`; `nnz = 3`, `sum = 6.0`
  (upper-triangular, one entry per edge).
- `dijkstra(G)` (all-pairs) →
  `[[0,1,3,4],[1,0,2,5],[3,2,0,3],[4,5,3,0]]`.
- `dijkstra(G, indices=0, return_predecessors=True)` →
  `(array([0,1,3,4]), array([-9999, 0, 1, 0], dtype=int32))`.
- `connected_components(G)` → `(1, array([0,0,0,0], dtype=int32))`
  (default `directed=True, connection='weak'`; `directed=False` gives the same).

MST orientation, established by running the oracle on multiple graphs:
- Symmetric input ⇒ scipy emits each MST edge ONCE, **upper-triangular**
  (`i < j`). Verified on the 4-node graph (edges `(0,1),(1,2),(2,3)`) and a
  7-node symmetric graph (edges all `i < j`:
  `(0,1),(0,3),(1,4),(2,4),(3,5),(4,6)`).
- Asymmetric/lower-triangular-only input ⇒ scipy preserves the *given*
  orientation (a lower-triangular input yields lower-triangular output). But the
  csgraph contract is a symmetric input for an undirected graph
  (`__init__.py:79`), so the **target orientation is upper-triangular `i < j`,
  one entry per edge**. That is the concrete fix target for the headline REQ.

2-component labelling oracle (ascending node-discovery order): for
`{0,1,2}` + `{3,4}`, `connected_components(..., directed=False)` →
`(2, array([0,0,0,1,1]))` — the low-index component gets label 0; for
`{0,1}` + `{2,3}`, `(2, array([0,0,1,1]))`.

## Requirements

- REQ-DIJ-DIST: `dijkstra(graph, source).distances` matches
  `scipy.sparse.csgraph.dijkstra(graph, indices=source)` element-wise, with
  `f64::INFINITY` for unreachable vertices (matching scipy's `inf`).
- REQ-DIJ-SIG: `dijkstra`'s signature/flags mirror scipy's
  `dijkstra(csgraph, directed=True, indices=None, return_predecessors=False,
  unweighted=False, limit=inf, min_only=False)` — multi-source (`indices=None` =
  all sources), `directed`, `unweighted`, `limit`, `min_only`.
- REQ-PRED-SENTINEL: `DijkstraResult.predecessors` mirror scipy's
  `return_predecessors=True` output — `-9999` sentinel for the source / no
  predecessor, `int32` dtype (R-DEV-3 output contract).
- REQ-ALLPAIRS: `dijkstra_all_pairs(graph)` matches
  `scipy.sparse.csgraph.dijkstra(graph)` / `shortest_path(graph, method='D')` —
  the full `n×n` distance matrix.
- REQ-CC-LABELS: `connected_components(graph)` matches
  `scipy.sparse.csgraph.connected_components(graph, directed=False)` on both
  `n_components` AND the per-vertex label array (ascending node-discovery order).
- REQ-CC-PARAMS: `connected_components` mirrors scipy's
  `connected_components(csgraph, directed=True, connection='weak'/'strong',
  return_labels=True)` — the `directed` flag and `connection='strong'` (SCC).
- REQ-MST-REPR: `minimum_spanning_tree(graph)` matches
  `scipy.sparse.csgraph.minimum_spanning_tree` in *representation* — one entry
  per edge, upper-triangular `i < j`, `nnz = n_edges`, sum `= MST weight`.
- REQ-MISSING-FNS: the remaining documented csgraph functions exist
  (`shortest_path`, `floyd_warshall`, `bellman_ford`, `johnson`, `yen`,
  `breadth_first_order`/`depth_first_order`, `…_tree`, `laplacian`,
  `reconstruct_path`, `reverse_cuthill_mckee`, `maximum_flow`,
  `maximum_bipartite_matching`, `structural_rank`).
- REQ-ERR-TYPE: the routines return `Result<_, FerroError>` (CLAUDE.md /
  R-CODE-2) with scipy-matching exception semantics (`ValueError` for non-square,
  `NegativeCycleError` for negative cycles).
- REQ-CONSUMER: a non-test workspace caller (an estimator, e.g. `Isomap` /
  spectral embedding, or the `ferrolearn-python` binding) consumes
  `sparse_graph::*` so it is part of the live translation surface.
- REQ-FERRAY: the routines operate on ferray's sparse / csgraph analog rather
  than `sprs::CsMat` + `ndarray` (R-SUBSTRATE-1).

## Acceptance criteria

All expected values come from the live scipy oracle (R-CHAR-3), never from
ferrolearn. Run from `/tmp`. `G = [[0,1,0,4],[1,0,2,0],[0,2,0,3],[4,0,3,0]]`.

- AC-1 (REQ-DIJ-DIST):
  `python3 -c "import numpy as np,scipy.sparse as sp; from scipy.sparse.csgraph import dijkstra; print(dijkstra(sp.csr_matrix(np.array([[0,1,0,4],[1,0,2,0],[0,2,0,3],[4,0,3,0]],float)), indices=0).tolist())"`
  → `[0.0, 1.0, 3.0, 4.0]`; `dijkstra(&G, 0).distances` equals it element-wise
  (rel ≤ 1e-12, exact here). Disconnected `{0,1}|{2,3}`, `dijkstra(...,indices=0)`
  → `[0.0, 1.0, inf, inf]`; ferrolearn yields `[0,1,INFINITY,INFINITY]`.
- AC-2 (REQ-ALLPAIRS):
  `print(dijkstra(sp.csr_matrix(...)).tolist())` →
  `[[0,1,3,4],[1,0,2,5],[3,2,0,3],[4,5,3,0]]`; `dijkstra_all_pairs(&G)` equals it.
- AC-3 (REQ-DIJ-SIG): `dijkstra(graph, source)` is single-source; scipy
  `dijkstra(csgraph, directed=True, indices=None, return_predecessors=False,
  unweighted=False, limit=inf, min_only=False)` — `directed`/`unweighted`/`limit`/
  `min_only`/`indices=None` have no ferrolearn parameter. A critic pins a FAILING
  test requiring an `unweighted=true` or `limit` path. FAILS until implemented.
- AC-4 (REQ-PRED-SENTINEL):
  `print(dijkstra(sp.csr_matrix(...), indices=0, return_predecessors=True)[1].tolist())`
  → `[-9999, 0, 1, 0]`; ferrolearn `DijkstraResult.predecessors` is
  `[usize::MAX, 0, 1, 0]` (`usize::MAX`, not `-9999`; `usize`, not `int32`).
  A critic pins a FAILING test asserting the source-vertex sentinel equals the
  scipy `-9999` contract. FAILS.
- AC-5 (REQ-CC-LABELS): for `{0,1,2}|{3,4}`,
  `print(connected_components(sp.csr_matrix(...), directed=False))` →
  `(2, [0,0,0,1,1])`; `connected_components(&G).n_components == 2` and `.labels`
  matches `[0,0,0,1,1]` (ascending node-discovery order). For `{0,1}|{2,3}` →
  `(2, [0,0,1,1])`.
- AC-6 (REQ-CC-PARAMS): on the asymmetric directed graph
  `[[0,1,0],[0,0,1],[0,0,0]]`, `connected_components(g, directed=True,
  connection='strong')` → `(3, [2,1,0])` (Tarjan SCC) whereas `connection='weak'`
  → `(1, [0,0,0])`. ferrolearn always returns the weak/undirected answer. A
  critic pins a FAILING `connection='strong'` test. FAILS.
- AC-7 (REQ-MST-REPR — headline):
  `python3 -c "import numpy as np,scipy.sparse as sp; from scipy.sparse.csgraph import minimum_spanning_tree as m; g=sp.csr_matrix(np.array([[0,1,0,4],[1,0,2,0],[0,2,0,3],[4,0,3,0]],float)); print(m(g).toarray(), m(g).nnz, m(g).sum())"`
  → `[[0,1,0,0],[0,0,2,0],[0,0,0,3],[0,0,0,0]] 3 6.0`. ferrolearn
  `minimum_spanning_tree(&G)` returns a SYMMETRIC matrix: `nnz == 6`,
  `sum == 12.0`, with both `[0,1]=1`/`[1,0]=1` etc. populated. A critic pins a
  FAILING test asserting `mst.nnz() == 3` and `total == 6.0` and upper-triangular
  orientation. FAILS until the fix (store one oriented `i<j` entry per kept edge).
- AC-8 (REQ-MISSING-FNS): `scipy.sparse.csgraph` exposes
  `shortest_path, floyd_warshall, bellman_ford, johnson, yen,
  breadth_first_order, depth_first_order, breadth_first_tree, depth_first_tree,
  laplacian, reconstruct_path, reverse_cuthill_mckee, maximum_flow,
  maximum_bipartite_matching, structural_rank` (`__init__.py:15-44`); none has a
  `sparse_graph.rs` symbol. `grep -n "pub fn" sparse_graph.rs` lists only
  `dijkstra`, `dijkstra_all_pairs`, `connected_components`,
  `minimum_spanning_tree`.
- AC-9 (REQ-ERR-TYPE): every public fn returns `Result<_, String>`; scipy raises
  `ValueError` for non-square and `NegativeCycleError` for negative cycles. The
  String error type fails the CLAUDE.md/R-CODE-2 `FerroError` contract.
- AC-10 (REQ-CONSUMER):
  `grep -rn "sparse_graph\|::dijkstra\|::connected_components\|::minimum_spanning_tree" --include=*.rs ferrolearn-*/src | grep -v 'sparse_graph.rs' | grep -v '//!' | grep -v '#\[cfg(test'`
  returns nothing that CALLS the module. `Isomap` (`isomap.rs`) declares its own
  private `fn dijkstra(adj: &[Vec<(usize, f64)>], source)` — it does not consume
  `sparse_graph::dijkstra`.
- AC-11 (REQ-FERRAY): the owned graph computation routes through ferray's sparse /
  csgraph analog, not `sprs::CsMat` + `ndarray::{Array1, Array2}`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-DIJ-DIST (single-source distances) | SHIPPED | impl `pub fn dijkstra in sparse_graph.rs` (`BinaryHeap<Reverse<(OrderedF64, usize)>>` min-heap; relaxation `if new_dist < dist[v] { dist[v] = new_dist; … }`; unreachable left at `f64::INFINITY`) mirrors `scipy.sparse.csgraph.dijkstra`. Live oracle (R-CHAR-3): `dijkstra(G, indices=0)` = `[0,1,3,4]`; disconnected `{0,1}|{2,3}` from src0 = `[0,1,inf,inf]`. ferrolearn matches element-wise. Verification: `cargo test -p ferrolearn-numerical --lib sparse_graph` (`dijkstra_simple_path`, `dijkstra_disconnected` green). NOTE: ships on numerical parity of the *distance vector*; the module-as-surface is gated by REQ-CONSUMER/REQ-FERRAY. |
| REQ-ALLPAIRS (all-pairs distances) | SHIPPED | impl `pub fn dijkstra_all_pairs in sparse_graph.rs` (loops `dijkstra(graph, src)` over all `src`, assigns each result into `result.row_mut(src)`) mirrors `scipy.sparse.csgraph.dijkstra(G)` (= `shortest_path(G, method='D')`). Live oracle: `dijkstra(G)` = `[[0,1,3,4],[1,0,2,5],[3,2,0,3],[4,5,3,0]]`; ferrolearn matches. Verification: `dijkstra_all_pairs_triangle` (green). |
| REQ-CC-LABELS (component count + labels) | SHIPPED | impl `pub fn connected_components in sparse_graph.rs` (builds undirected adjacency `if val != 0.0 { adj[i].push(j); adj[j].push(i); }`, ascending-start BFS `for start in 0..n`, contiguous 0-indexed labels) mirrors `scipy.sparse.csgraph.connected_components(G, directed=False)`. Live oracle: `{0,1,2}|{3,4}` → `(2, [0,0,0,1,1])`; `{0,1}|{2,3}` → `(2, [0,0,1,1])` — scipy labels in ascending node-discovery order, which the `for start in 0..n` BFS reproduces exactly (`n_components` AND label array match). Verification: `connected_components_two_clusters`, `connected_components_single` (green). The `directed`/`connection='strong'` flags are a separate REQ (REQ-CC-PARAMS). |
| REQ-DIJ-SIG (multi-source + flags) | NOT-STARTED | open prereq blocker #1949. `pub fn dijkstra(graph: &CsMat<f64>, source: usize)` is single-source returning a struct; scipy's `dijkstra(csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False, limit=inf, min_only=False)` is multi-source with flags. `indices=None` (all-sources), `directed` (always treats stored entries as directed edges — happens to match for symmetric input but no flag), `unweighted`, `limit`, `min_only` are unimplemented. |
| REQ-PRED-SENTINEL (predecessor `-9999`/int32) | NOT-STARTED | open prereq blocker #1950. `DijkstraResult.predecessors: Array1<usize>` uses `usize::MAX` for the source / no-predecessor; scipy's `return_predecessors=True` uses `-9999` in an `int32` array. Live oracle: `dijkstra(G, indices=0, return_predecessors=True)[1]` = `[-9999, 0, 1, 0]`; ferrolearn = `[usize::MAX, 0, 1, 0]`. Different sentinel AND dtype (R-DEV-3 output contract). |
| REQ-CC-PARAMS (`directed`/`connection='strong'`) | NOT-STARTED | open prereq blocker #1951. `connected_components` hardcodes undirected/weak (edge iff `graph[i,j]!=0 || graph[j,i]!=0`); scipy's `connected_components(csgraph, directed=True, connection='weak'/'strong')` exposes the `directed` flag and strongly-connected components (Tarjan). Live oracle on `[[0,1,0],[0,0,1],[0,0,0]]`: `connection='strong'` → `(3, [2,1,0])` vs `connection='weak'` → `(1, [0,0,0])`; ferrolearn returns only the weak answer. Labels are `usize`, scipy's are `int32`. |
| REQ-MST-REPR (one entry per edge, upper-tri) | SHIPPED | FIXED #1948: `pub fn minimum_spanning_tree in sparse_graph.rs` now pushes ONE oriented triplet `(u, v, w)` per kept edge (dropped the mirror `push((v, u, w))`); since edges are collected with `i<j`, the result is upper-triangular `i<j` with `nnz = n_edges` and entry-sum = MST weight, matching `scipy.sparse.csgraph.minimum_spanning_tree`. Live oracle (4-node) `[[0,1,0,0],[0,0,2,0],[0,0,0,3],[0,0,0,0]]`, `nnz=3`, `sum=6.0` (orientation verified on a 2nd 5-node graph). Pinned by `mst_representation_upper_triangular`; the in-crate `mst_simple` test was corrected to the scipy contract (`nnz==3`, `total==6.0`, lower mirrors `==0`). |
| REQ-MISSING-FNS (rest of csgraph) | NOT-STARTED | open prereq blocker #1952. `scipy.sparse.csgraph` (`__init__.py:15-44`) documents `shortest_path, floyd_warshall, bellman_ford, johnson, yen, breadth_first_order, depth_first_order, breadth_first_tree, depth_first_tree, laplacian, reconstruct_path, reverse_cuthill_mckee, maximum_flow, maximum_bipartite_matching, structural_rank`; `sparse_graph.rs` has only `dijkstra`/`dijkstra_all_pairs`/`connected_components`/`minimum_spanning_tree`. sklearn consumes: `shortest_path`/`dijkstra` (Isomap), `connected_components` (spectral_embedding), `laplacian` (spectral methods) — so these are real downstream prerequisites, not speculative surface. |
| REQ-ERR-TYPE (`FerroError` + scipy exceptions) | SHIPPED (#1953) | FIXED — all four routines (`dijkstra`/`dijkstra_all_pairs`/`connected_components`/`minimum_spanning_tree`) now return `Result<_, FerroError>`; the 7 `Err` sites (non-square `graph`, out-of-bounds `source`, negative-weight) map to `FerroError::InvalidParameter { name, reason }` (messages preserved), mirroring scipy's `ValueError` (non-square/bad source) and `NegativeCycleError` (negative weights). `ferrolearn-core` is a workspace dep (#1961). The 2 in-module `negative_weight_error` assertions updated `.unwrap_err().contains("negative")` → `matches!(Err(FerroError::InvalidParameter { reason, .. }) if reason.contains("negative"))`. Guard `tests/divergence_sparse_graph.rs::sparse_graph_invalid_returns_ferroerror` (out-of-bounds source + negative edge). Completes the `ferrolearn-numerical` error-type sweep (interpolate #1961 / integrate #1975 / distributions #1967 / optimize #1992 / sparse_eig #1983 / sparse_graph #1953). |
| REQ-CONSUMER (non-test production caller) | NOT-STARTED | open prereq blocker #1954. `lib.rs` exposes only `pub mod sparse_graph` (no re-export). `grep -rn "sparse_graph\|::dijkstra\|::connected_components\|::minimum_spanning_tree" --include=*.rs ferrolearn-*/src` returns no CALLER outside the module: the `ferrolearn-cluster` hits are NOT-STARTED doc-comments, and `ferrolearn-decomp/src/isomap.rs` reimplements its OWN private `fn dijkstra(adj: &[Vec<(usize, f64)>], source)` on adjacency lists rather than consuming `sparse_graph::dijkstra` on `CsMat`. S5 grandfathering does not rescue this — `sparse_graph` is an internal scipy.csgraph substrate, not a boundary estimator type; there is no external user and no Python binding for it. Dead translation surface (R-HONEST-3). Fix: route `Isomap`/`spectral_embedding` shortest-path/cc through `sparse_graph` (or fold into ferray per REQ-FERRAY). |
| REQ-FERRAY (ferray substrate) | NOT-STARTED | open prereq blocker #1955. Module operates on `sprs::CsMat<f64>` + `ndarray::{Array1, Array2}` (R-SUBSTRATE-1 wrong substrate). The destination is ferray's sparse / scipy.sparse.csgraph analog. ferray does not yet expose a csgraph layer (R-SUBSTRATE-5: a ferray gap is real work, filed upstream to ferray; the ferrolearn unit is NOT-STARTED on this REQ until ferray ships the sparse-graph surface). |

## Architecture

`sparse_graph.rs` is a flat module of free functions over `sprs::CsMat<f64>`,
plus two result structs (`DijkstraResult { distances: Array1<f64>,
predecessors: Array1<usize> }`, `ConnectedComponentsResult { n_components: usize,
labels: Array1<usize> }`) and one private helper (`OrderedF64`, an `f64` wrapper
implementing `Ord` via `f64::total_cmp` so distances can live in a `BinaryHeap`).
There is no unfitted/Fitted split — these are pure algorithms, not estimators —
which is appropriate, since `scipy.sparse.csgraph` functions are likewise
free functions, not classes.

The four algorithms are textbook-correct on the *values* they compute:

- **`dijkstra`** is lazy-deletion Dijkstra: a min-heap of `(distance, vertex)`
  keyed by `OrderedF64`, skipping stale entries (`if d_u > dist[u] { continue; }`),
  relaxing over `graph.outer_view(u)` (the CSR row = out-edges of `u`). Distances
  match scipy element-wise (REQ-DIJ-DIST SHIPPED). It validates non-negative
  weights up front (`if w < 0.0 { return Err(...) }`) — Dijkstra's precondition;
  scipy steers negative weights to `bellman_ford`/`johnson` (absent here,
  REQ-MISSING-FNS). The `predecessors` field is computed but uses `usize::MAX`
  rather than scipy's `-9999`/`int32` (REQ-PRED-SENTINEL NOT-STARTED), and the
  signature is single-source rather than scipy's flagged multi-source form
  (REQ-DIJ-SIG NOT-STARTED).

- **`dijkstra_all_pairs`** is repeated single-source Dijkstra, the same strategy
  scipy's `shortest_path(method='D')` / `dijkstra(indices=None)` uses for the
  positive-weight case; the `n×n` matrix matches the oracle (REQ-ALLPAIRS
  SHIPPED).

- **`connected_components`** symmetrizes the CSR into adjacency lists
  (`adj[i].push(j); adj[j].push(i)` for every nonzero), then BFS-labels from
  ascending start vertices. This is exactly scipy's `connection='weak'`
  (`directed=False`) component decomposition, and — critically — scipy labels in
  ascending node-discovery order, which the `for start in 0..n` loop reproduces,
  so both `n_components` and the per-vertex label array match (REQ-CC-LABELS
  SHIPPED). The `directed`/`connection='strong'` (Tarjan SCC) surface is absent
  (REQ-CC-PARAMS NOT-STARTED).

- **`minimum_spanning_tree`** is Kruskal: collect each undirected edge once
  (`if i < j && val != 0.0`), sort by weight, union-find with path compression +
  union-by-rank. The *selected edge set* is correct. The defining divergence
  (REQ-MST-REPR, the headline) is purely in the OUTPUT: it pushes both
  `(u, v, w)` and `(v, u, w)` into the triplet list, producing a symmetric matrix
  with `nnz = 2·n_edges` and `sum = 2·(MST weight)`. scipy returns a *directed*
  spanning-tree representation — one entry per edge, upper-triangular (`i < j`)
  for symmetric input. The fix is local and deterministic: emit a single oriented
  triplet `(min(u,v), max(u,v), w)` per `union` success and drop the mirror,
  giving `nnz = 3` / `sum = 6.0` on the canonical 4-node graph.

The two cross-cutting structural facts are REQ-CONSUMER and REQ-FERRAY. The
module has **no caller**: `lib.rs` exposes `pub mod sparse_graph` with no
re-export, and the one crate that genuinely needs single-source shortest paths —
`ferrolearn-decomp`'s `Isomap` — reimplements its own private `dijkstra` over
`Vec<Vec<(usize, f64)>>` adjacency lists rather than calling this module's
`CsMat`-based `dijkstra`. So at baseline `1aa756a40`, `sparse_graph` is dead
translation surface duplicated downstream. And it is on the wrong substrate
(`sprs`/`ndarray` rather than ferray's sparse/csgraph analog, R-SUBSTRATE-1).
Both are NOT-STARTED with filed prereq blockers; the honest call (R-HONEST-3) is
that three numerical REQs ship on impl+oracle while the representation, API, and
integration REQs do not.

## Verification

Commands establishing the claims (run at baseline `1aa756a40`):

- `cargo test -p ferrolearn-numerical --lib sparse_graph` → 7 passed, 0 failed
  (`dijkstra_simple_path`, `dijkstra_disconnected`, `dijkstra_all_pairs_triangle`,
  `connected_components_two_clusters`, `connected_components_single`, `mst_simple`,
  `negative_weight_error`). REQ-MST-REPR is now SHIPPED (#1948): `mst_simple` was
  corrected to the scipy contract (`nnz == 3`, `total == 6.0`, upper-triangular,
  lower mirrors `== 0`) and the failing pin `mst_representation_upper_triangular`
  in `tests/divergence_sparse_graph.rs` is green.
- dijkstra / all-pairs / cc oracle (REQ-DIJ-DIST, REQ-ALLPAIRS, REQ-CC-LABELS,
  R-CHAR-3 — expected from scipy, never from ferrolearn):
  `python3 -c "import numpy as np, scipy.sparse as sp; from scipy.sparse.csgraph import dijkstra, connected_components; g=sp.csr_matrix(np.array([[0,1,0,4],[1,0,2,0],[0,2,0,3],[4,0,3,0]],float)); print(dijkstra(g).tolist()); print(connected_components(g, directed=False))"`
  → `[[0,1,3,4],[1,0,2,5],[3,2,0,3],[4,5,3,0]]` and `(1, [0,0,0,0])`. The 2-cluster
  label oracle (`{0,1,2}|{3,4}` → `(2, [0,0,0,1,1])`) confirms ascending
  discovery-order labelling.
- MST representation oracle (REQ-MST-REPR, headline):
  `python3 -c "import numpy as np, scipy.sparse as sp; from scipy.sparse.csgraph import minimum_spanning_tree as m; g=sp.csr_matrix(np.array([[0,1,0,4],[1,0,2,0],[0,2,0,3],[4,0,3,0]],float)); print(m(g).toarray(), m(g).nnz, m(g).sum())"`
  → `[[0,1,0,0],[0,0,2,0],[0,0,0,3],[0,0,0,0]] 3 6.0`. ferrolearn returns a
  symmetric matrix (`nnz=6`, `sum=12`). A critic pins this as a FAILING `#[test]`
  asserting `mst.nnz() == 3 && total == 6.0` (fails until the mirror push is
  dropped).
- predecessor-sentinel oracle (REQ-PRED-SENTINEL):
  `python3 -c "import numpy as np, scipy.sparse as sp; from scipy.sparse.csgraph import dijkstra; g=sp.csr_matrix(np.array([[0,1,0,4],[1,0,2,0],[0,2,0,3],[4,0,3,0]],float)); print(dijkstra(g, indices=0, return_predecessors=True)[1].tolist())"`
  → `[-9999, 0, 1, 0]`; ferrolearn's `predecessors` is `[usize::MAX, 0, 1, 0]`.
- consumer check (REQ-CONSUMER):
  `grep -rn "sparse_graph\|::dijkstra\|::connected_components\|::minimum_spanning_tree" --include=*.rs ferrolearn-*/src | grep -v 'sparse_graph.rs'`
  returns only NOT-STARTED doc-comments and `isomap.rs`'s OWN private `dijkstra` —
  no production consumer of this module.
