# KD-Tree (sklearn.neighbors.KDTree)

<!--
tier: 3-component
status: draft
baseline-commit: 2013942c
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/neighbors/_kd_tree.pyx.tp      # KDTree(KDTree64); allocate_data/init_node/min_rdist/max_rdist node-bounds specialization
  - sklearn/neighbors/_binary_tree.pxi.tp  # BinaryTree.__init__ (:851-852 leaf_size=40, metric='minkowski'); query (:1089-1142); query_radius (:1201-1258); leaf_size doc/invariance (:245-253)
ferrolearn-module: ferrolearn-neighbors/src/kdtree.rs
parity-ops: KDTree (KDTree.__init__, KDTree.query, KDTree.query_radius)
crosslink-issue: TBD  # NOTE placeholder — replace with the kdtree unit's own tracking issue when the director assigns it (the ferrolearn-neighbors / Layer 3 block).
-->

## Summary

`ferrolearn-neighbors/src/kdtree.rs` mirrors scikit-learn's `KDTree`
(`sklearn/neighbors/_kd_tree.pyx.tp`, a thin specialization of the shared
`BinaryTree` in `_binary_tree.pxi.tp`). It exposes `KdTree::build` (the
constructor analog) and `KdTree::query` (the `k`-nearest-neighbor analog), plus
two free functions `euclidean_distance` and `brute_force_knn`. The tree is the
spatial-index backend consumed (non-test) by the `KdTree` `Algorithm` arm of
`KNeighborsClassifier`/`Regressor` (`knn.rs`), `NearestNeighbors`
(`nearest_neighbors.rs`), and `RadiusNeighbors` (`radius_neighbors.rs`).

Under honest underclaim (R-HONEST-3), the **one behavior that is genuinely
present and oracle-matching** is `query` (k-NN): for a single query point it
returns the `k` nearest neighbors as `(index, distance)` pairs, sorted nearest
distance first, with **true Euclidean distance** matching the live sklearn
1.5.2 oracle to full precision. Everything else diverges from `KDTree`'s
contract:

1. The constructor takes **no `leaf_size`** (sklearn `leaf_size=40`,
   `_binary_tree.pxi.tp:852`) and **no `metric`** (sklearn `metric='minkowski'`
   with `valid_metrics = ['euclidean','l2','minkowski','p','manhattan',
   'cityblock','l1','chebyshev','infinity']`). ferrolearn is **Euclidean-only**
   and builds a one-point-per-node median-split tree, not a leaf-bucketed
   `BinaryTree` (so `leaf_size <= n_points <= 2*leaf_size` node invariant,
   `:251-253`, does not exist).
2. There is **no `query_radius`** (sklearn `_binary_tree.pxi.tp:1201`). The KNN
   estimators that need radius search fall back to a **brute-force** scan in
   `radius_neighbors.rs`/`nearest_neighbors.rs`, bypassing the tree entirely.
3. `query` takes a **single** query row (`&[f64]`) and returns a flat
   `Vec<(usize, f64)>`, not sklearn's batched `(n_query, n_features)` → `(d, i)`
   of shape `(n_query, k)` (`:1125-1131`); there is no `return_distance` /
   `sort_results` / `dualtree` / `breadth_first` toggle (`:1089-1091`).
4. `query` does **not raise on `k > n_samples`**; sklearn raises `ValueError`
   ("k must be less than or equal to the number of training points",
   `:1140-1142`). ferrolearn silently returns `min(k, n_samples)` neighbors.
5. `build` on an **empty array returns an empty tree**; sklearn raises
   `ValueError("X is an empty array")` (`:855-856`).
6. The public surface is **not exposed by `ferrolearn-python`** — `import
   sklearn.neighbors` gives `KDTree`; `import ferrolearn` gives nothing.
7. The crate is on the **`ndarray` substrate** (`use ndarray::Array2`), not
   ferray (R-SUBSTRATE).

`KdTree::build`/`query` are existing pub APIs consumed across `knn.rs`,
`nearest_neighbors.rs`, `radius_neighbors.rs` (grandfathered per S5/R-DEFER-1);
that wiring is the non-test production-consumer surface.

## Algorithm (sklearn — the contract)

### Construction (`BinaryTree.__init__`, `_binary_tree.pxi.tp:851-880`)

`KDTree(X, leaf_size=40, metric='minkowski', sample_weight=None, **kwargs)`
validates `X` is non-empty (`:855`), requires `leaf_size >= 1` (`:861-862`),
resolves the `DistanceMetric` (default Minkowski `p=2` = Euclidean), and
recursively partitions a contiguous `idx_array` into nodes that each hold a
**bucket** of points. A node splits only while `idx_end - idx_start > 2 *
leaf_size` (`:1061`); the split is on the **dimension of greatest spread**
(widest axis), partitioning the index range about its median. For KD-trees, each
node additionally stores axis-aligned **bounds** (`node_bounds[0/1, i_node, :]`,
the per-feature min/max) computed in `init_node64` (`_kd_tree.pyx.tp:73-120`).
The leaf-size invariant `leaf_size <= n_points <= 2*leaf_size` holds except when
`n_samples < leaf_size` (`:251-253`); **`leaf_size` changes performance and tree
shape, not query results** (`:247-250`).

### `query` (k-NN, `_binary_tree.pxi.tp:1089-1199`)

`query(X, k=1, return_distance=True, dualtree=False, breadth_first=False,
sort_results=True)`. For each row of `X` (shape `(n_query, n_features)`):
validates dimension match (`:1136-1138`), raises `ValueError` if
`n_samples < k` (`:1140-1142`), and runs a depth-first search maintaining a
bounded **max-heap** of the `k` best `(reduced_distance, index)`. The node-prune
test uses the **reduced distance** (squared-Euclidean) lower bound
`min_rdist` (`_kd_tree.pyx.tp:123-147`) from the point to the node bounds: a node
is descended only if its `min_rdist` is below the heap's current worst. On
return, results are converted from reduced to true distance and, with
`sort_results=True` (default), each row's `k` neighbors are sorted
**nearest-distance-first** (`:1191`). Returns `(d, i)` of shape `(n_query, k)`.

**Exact ties** (multiple neighbors at identical distance) are **not stably
ordered**: the per-row order is the heap pop / traversal order and depends on
`leaf_size` and node layout. Oracle: querying `(0.5,0.5)` against the four
unit-square corners (all at distance `sqrt(0.5)`) yields index order
`[0,1,2,3]` at `leaf_size=40` but `[0,2,1,3]` at `leaf_size=1` or `2`. The
**distances and the neighbor set are invariant**; only the tie ordering shifts.

### `query_radius` (`_binary_tree.pxi.tp:1201-1258`)

`query_radius(X, r, return_distance=False, count_only=False,
sort_results=False)`. Returns, per query row, all indices within radius `r`
(object array). `count_only=True` returns counts; `return_distance=True` returns
`(ind, dist)`. **Results are unsorted by default** (`:1222`); `sort_results=True`
(requires `return_distance=True`, else `ValueError` `:1258-1259`) sorts each
row's neighbors **by ascending distance** (`:1228-1233`). `count_only` and
`return_distance` together raise `ValueError` (`:1254-1256`).

### Edge cases (live oracle, sklearn 1.5.2, run from /tmp)

- `KDTree(np.zeros((0,2)))` → `ValueError("Found array with 0 sample(s) ...")`.
- `query(..., k=10)` with `n_samples=5` → `ValueError("k must be less than or
  equal to the number of training points")`.
- `query(..., k=0)` → returns empty `(1, 0)` arrays (no error).
- single-point dataset: `KDTree([[1,2]]).query([[1,2]], k=1)` →
  `(array([[0.]]), array([[0]]))`.

## ferrolearn (what exists)

All in `ferrolearn-neighbors/src/kdtree.rs`:

- **`pub struct KdTree { root: Option<Box<KdNode>> }`** — one point per node
  (private `struct KdNode` stores `index`, `split_dim`, `split_val`, `left`,
  `right`). There is **no `leaf_size`**, no leaf bucket, and **no per-node
  bounds**; pruning uses only the single split plane.
- **`pub fn build<F: Float + Send + Sync + 'static>(data: &Array2<F>) -> Self`**
  — recursive median split (`fn build_recursive`) on the widest-spread dimension
  (`fn choose_split_dimension`, max−min spread, matching sklearn's widest-axis
  choice). Converts all data to `f64` internally. **Empty input → empty tree**
  (returns `Self { root: None }`), no error. No `metric`/`leaf_size` arg.
- **`pub fn query<F: Float + Send + Sync + 'static>(&self, data: &Array2<F>,
  query: &[f64], k: usize) -> Vec<(usize, f64)>`** — single query row; depth-first
  `fn search_recursive` maintaining a bounded max-heap (`struct NeighborHeap`,
  `fn try_insert`, `fn worst_distance`, `fn into_sorted`). Returns
  `(index, distance)` pairs **sorted ascending by true Euclidean distance**
  (`NeighborHeap::into_sorted`). The farther-subtree prune is
  `plane_dist < heap.worst_distance()` (`fn search_recursive`, the `if
  plane_dist < ... { if let Some(child) ... }` block) — a 1-D plane distance, the
  correct lower bound for an axis-aligned split. **No `k > n_samples` guard** (the
  heap simply holds `min(k, n_samples)`); no batched query; no
  `return_distance`/`sort_results`/`dualtree` toggle.
- **`pub fn euclidean_distance<F: Float>(a: &[F], b: &[F]) -> F`** — generic
  true-Euclidean distance (sum of squares, then `sqrt`). Consumed non-test by
  `nearest_neighbors.rs` / `radius_neighbors.rs`. (Private `fn
  euclidean_distance_f64` is the `f64` fast path used inside the tree search.)
- **`pub fn brute_force_knn<F: Float + Send + Sync + 'static>(data, query, k)
  -> Vec<(usize, F)>`** — `select_nth_unstable_by` partial sort + final sort;
  the fallback path for the `Algorithm::None` arm and for radius search.

**Consumers (non-test):** `knn.rs` (`SpatialIndex::KdTree(KdTree::build(data))`
under `Algorithm::KdTree`/`Auto`; `kdtree::brute_force_knn` for `None`),
`nearest_neighbors.rs` (same plus `kdtree::euclidean_distance` for its radius
scan), `radius_neighbors.rs` (`KdTree::build`, then a **brute-force** radius
loop using `kdtree::euclidean_distance`). These are existing pub APIs
(grandfathered, S5/R-DEFER-1).

## Requirements

- REQ-1: **`build` + constructor defaults (R-DEV-2).** Match
  `KDTree(X, leaf_size=40, metric='minkowski')` (`_binary_tree.pxi.tp:851-852`):
  a `leaf_size` parameter (default 40, `>=1` validated, `:861-862`) and a
  `metric` parameter (default Minkowski `p=2`). ferrolearn's `build` has neither
  and builds a one-point-per-node tree.
- REQ-2: **`query` k-NN value (R-DEV-1/3).** Match `query` (`:1089`): return the
  `k` nearest neighbors as **true Euclidean distances**, sorted nearest-first
  (`sort_results=True`, `:1191`). ferrolearn's `query` matches the live oracle on
  distances and on the neighbor set/order for a single query row when distances
  are distinct.
- REQ-3: **`query_radius` value (R-DEV-1/3).** Provide `query_radius(X, r, ...)`
  (`:1201`): indices within radius, `count_only`/`return_distance`/`sort_results`
  semantics. ferrolearn's `KdTree` has **no `query_radius`**; radius search is
  brute-force in `radius_neighbors.rs`.
- REQ-4: **Tie-breaking / `sort_results` ordering (R-DEV-1/3).** Match sklearn's
  result contract under exact-distance ties: distances and the neighbor set are
  invariant, tie order is traversal-dependent (`:1191`), and `query_radius`
  default-unsorted vs `sort_results=True` ascending-distance ordering
  (`:1222,1228-1233`). ferrolearn `query` always sorts and exposes no
  `sort_results` toggle; no `query_radius` ordering exists at all.
- REQ-5: **Metric set (R-DEV-2).** Match `valid_metrics`
  (`euclidean`/`l2`/`minkowski`/`p`/`manhattan`/`cityblock`/`l1`/`chebyshev`/
  `infinity`) and the reduced-distance machinery (`_kd_tree.pyx.tp:123-201`).
  ferrolearn is **Euclidean-only** (hardcoded in `euclidean_distance_f64` /
  `search_recursive`).
- REQ-6: **`leaf_size` invariance + node bounds (R-DEV-1).** Match the
  leaf-bucketed `BinaryTree` build: `leaf_size`-bounded leaves
  (`leaf_size <= n_points <= 2*leaf_size`, `:251-253`), per-node axis-aligned
  bounds (`init_node64`, `_kd_tree.pyx.tp:73-120`), and `min_rdist` bound-based
  pruning (`:123-147`) — with results invariant to `leaf_size` (`:247-250`).
  ferrolearn has one-point-per-node, no bounds, no `leaf_size`.
- REQ-7: **Error contract (R-DEV-2).** Match `ValueError` on `k > n_samples`
  (`:1140-1142`) and on empty `X` (`:855-856`). ferrolearn silently clamps `k`
  and returns an empty tree.
- REQ-8: **Batched query shape (R-DEV-3).** Match `query(X)` accepting
  `(n_query, n_features)` and returning `(d, i)` of shape `(n_query, k)`
  (`:1125-1131`), with `return_distance` toggle. ferrolearn's `query` is
  single-row → flat `Vec`.
- REQ-9: **PyO3 binding (R-DEFER-1).** `import sklearn.neighbors` exposes
  `KDTree`; `ferrolearn-python` exposes no `KDTree` shim.
- REQ-10: **ferray substrate (R-SUBSTRATE).** `kdtree.rs` imports
  `ndarray::Array2` + `num_traits::Float`, not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (run from `/tmp`),
never literal-copied from ferrolearn (R-CHAR-3). Fixed dataset
`X = [[0,0],[1,0],[0,1],[1,1],[10,10]]` unless noted.

- AC-1 (REQ-2, present & matching): `KDTree(X).query([[0.1,0.1]], k=2)` →
  distances `[0.14142135623730953, 0.9055385138137417]`, indices `[0, 1]`.
  ferrolearn `KdTree::build(&X).query(&X, &[0.1,0.1], 2)` returns
  `[(0, 0.1414213562373095...), (1, 0.9055385138137417)]` — **matches**.
- AC-2 (REQ-2 distinct-distance order): `KDTree(X).query([[0.5,0.5]], k=4)` →
  all four distances `sqrt(0.5)=0.7071067811865476`, indices `[0,1,2,3]` at the
  default `leaf_size=40`. ferrolearn must return the same four indices (as a set)
  with distance `sqrt(0.5)`.
- AC-3 (REQ-3 pin): `KDTree(X).query_radius([[0.1,0.1]], r=1.5)` → `[[0,1,2,3]]`;
  `return_distance=True` → dist `[0.1414..., 0.9055..., 0.9055..., 1.2728...]`;
  `count_only=True` → `[4]`. ferrolearn `KdTree` cannot express this (no
  `query_radius`).
- AC-4 (REQ-4 pin): `KDTree(X).query_radius([[0.1,0.1]], r=1.5,
  return_distance=True, sort_results=True)` → indices `[0,2,1,3]`, distances
  ascending `[0.1414..., 0.9055..., 0.9055..., 1.2728...]`; default
  (`sort_results=False`) → `[0,1,2,3]`. ferrolearn has no `query_radius`
  ordering. (Also: `query` tie order varies with `leaf_size` — `[0,1,2,3]` at
  40, `[0,2,1,3]` at 1/2 — so a parity test must compare the neighbor SET +
  distances, not a fixed index permutation.)
- AC-5 (REQ-7 pin): `KDTree(X).query([[0,0]], k=10)` → sklearn raises
  `ValueError("k must be less than or equal to the number of training points")`;
  ferrolearn silently returns 5 neighbors. `KDTree(np.zeros((0,2)))` → sklearn
  `ValueError("X is an empty array")`; ferrolearn returns an empty tree.
- AC-6 (REQ-6 leaf_size invariance): `KDTree(X, leaf_size=ls).query([[0.5,0.5]],
  k=4)` returns identical distances and the same neighbor set for `ls in
  {1,2,40}` (results invariant; only tie order shifts). ferrolearn has no
  `leaf_size` to vary.
- AC-7 (REQ-5 metric): `KDTree(X, metric='manhattan').query([[0.1,0.1]], k=1)` →
  distance `0.2` (L1), index `0`; `metric='chebyshev'` → `0.1`. ferrolearn is
  Euclidean-only and cannot express either.
- AC-8 (single point, must match): `KDTree([[1,2]]).query([[1,2]], k=1)` →
  `(0.0, 0)`. ferrolearn `test_build_single_point` already establishes this.

## REQ status table

Binary (R-DEFER-2). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2). Live oracle = installed sklearn 1.5.2, run from `/tmp`. Honest
underclaim (R-HONEST-3): one REQ is SHIPPED (single-row k-NN value, the behavior
that actually exists and matches the oracle); the rest are NOT-STARTED with open
prereq blockers (suggested numbers — the director creates the real issues).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`build` defaults) | NOT-STARTED | open prereq blocker #810. `pub fn build` takes only `data: &Array2<F>` — **no `leaf_size`** (sklearn default 40, `>=1` validated, `_binary_tree.pxi.tp:852,861-862`) and **no `metric`** (sklearn `'minkowski'`, `:852`). Builds one-point-per-node median-split tree, not a leaf-bucketed `BinaryTree`. |
| REQ-2 (`query` k-NN value) | SHIPPED | impl `pub fn query` in `kdtree.rs` (depth-first `fn search_recursive` + bounded max-heap `NeighborHeap::into_sorted`) returns `(index, true-Euclidean-distance)` sorted nearest-first, mirroring `query`/`sort_results=True` (`_binary_tree.pxi.tp:1089,1191`). Non-test consumer: `knn.rs` (`SpatialIndex::KdTree(tree) => ...` under `Algorithm::KdTree`/`Auto`) and `nearest_neighbors.rs`. Live oracle (AC-1): `KDTree(X).query([[0.1,0.1]],k=2)` → dist `[0.14142135623730953, 0.9055385138137417]`, idx `[0,1]`; ferrolearn `query(&X,&[0.1,0.1],2)` matches. Distinct-distance value parity only; tie ordering is REQ-4. |
| REQ-3 (`query_radius`) | NOT-STARTED | open prereq blocker #811. `KdTree` has **no `query_radius`** method (sklearn `_binary_tree.pxi.tp:1201`); `radius_neighbors.rs`/`nearest_neighbors.rs` do a brute-force radius scan via `kdtree::euclidean_distance`, bypassing the tree. Pin: `KDTree(X).query_radius([[0.1,0.1]],r=1.5)` → `[[0,1,2,3]]`, inexpressible on `KdTree`. |
| REQ-4 (tie-break / `sort_results`) | NOT-STARTED | open prereq blocker #812. `query` always sorts and exposes no `sort_results` toggle; `query_radius` (and its default-unsorted vs ascending-sorted ordering, `:1222,1228-1233`) is absent. sklearn tie order is traversal/`leaf_size`-dependent (`query([[0.5,0.5]],k=4)` → `[0,1,2,3]` at leaf_size 40, `[0,2,1,3]` at 1/2). A parity test must compare neighbor SET + distances, not a fixed permutation. |
| REQ-5 (metric set) | NOT-STARTED | open prereq blocker #813. Euclidean-only (`fn euclidean_distance_f64` / `fn search_recursive` hardcode squared-diff sum). sklearn `valid_metrics` = `euclidean/l2/minkowski/p/manhattan/cityblock/l1/chebyshev/infinity` with reduced-distance specialization (`_kd_tree.pyx.tp:123-201`). Pin: `KDTree(X,metric='manhattan').query([[0.1,0.1]],k=1)` → dist `0.2`. |
| REQ-6 (`leaf_size` invariance + bounds) | NOT-STARTED | open prereq blocker #814. No `leaf_size`, no per-node axis-aligned bounds (`init_node64`, `_kd_tree.pyx.tp:73-120`), no `min_rdist` bound-based prune (`:123-147`); prune is a single split-plane distance in `fn search_recursive`. sklearn guarantees `leaf_size <= n_points <= 2*leaf_size` (`:251-253`) with results invariant to `leaf_size` (`:247-250`). |
| REQ-7 (error contract) | NOT-STARTED | open prereq blocker #815. `pub fn query` has **no `k > n_samples` guard** (clamps via heap), and `pub fn build` returns an empty tree on empty input. sklearn raises `ValueError` on both (`:1140-1142`, `:855-856`). Pin: `query([[0,0]],k=10)` on 5 points → sklearn `ValueError`, ferro returns 5. |
| REQ-8 (batched query shape) | NOT-STARTED | open prereq blocker #816. `pub fn query` takes a single `query: &[f64]` → flat `Vec<(usize,f64)>`; sklearn `query(X)` accepts `(n_query, n_features)` → `(d, i)` of shape `(n_query, k)` with `return_distance` toggle (`:1125-1131`). |
| REQ-9 (PyO3 binding) | NOT-STARTED | open prereq blocker #817. `ferrolearn-python/src/` (classifiers/regressors/transformers/clusterers/extras) exposes no `KDTree`; `import ferrolearn` cannot call what `import sklearn.neighbors` provides. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #818. `kdtree.rs` imports `ndarray::Array2` + `num_traits::Float`, not `ferray-core` (R-SUBSTRATE). |

## Architecture

`kdtree.rs` is a single module exposing `pub struct KdTree` plus three free
functions. The tree is a **binary recursive median-split** structure:

1. **Build** (`fn build` → `fn build_recursive`): the dataset is copied to a
   `Vec<Vec<f64>>` of `f64`; `fn choose_split_dimension` picks the
   greatest-spread (max−min) axis — this matches sklearn's "widest axis" choice
   conceptually — then the index range is sorted on that axis and split at its
   **median position**, which becomes the node's stored point; left/right
   subranges recurse. Each node therefore holds exactly **one** point (a
   `leaf_size == 1`-like degenerate compared to sklearn's bucketed
   `BinaryTree`), and stores only `split_dim`/`split_val`, **not** the
   axis-aligned `node_bounds` sklearn's `init_node64` computes
   (`_kd_tree.pyx.tp:73-120`). This is the structural root of REQ-6.

2. **Query** (`fn query` → `fn search_recursive`): a bounded max-heap
   (`struct NeighborHeap`, capacity `k`) tracks the best `(distance, index)`.
   At each node the stored point's true Euclidean distance is `try_insert`-ed,
   the nearer child is descended first (`go_left = query_val <= split_val`), and
   the farther child is descended only if the **split-plane distance**
   `(query_val - split_val).abs()` is below `heap.worst_distance()` (an infinite
   sentinel until the heap fills). This is the correct lower-bound prune for a
   single axis-aligned plane, but it is weaker than sklearn's `min_rdist`
   node-bounds bound (`:123-147`) — correctness is preserved (every closer
   point is still found) but the pruning is the one-plane variant. Results are
   drained sorted-ascending by `NeighborHeap::into_sorted`.

**Distance:** the search uses true Euclidean distance throughout
(`euclidean_distance_f64` = `sqrt` of summed squared diffs), so reported
distances match sklearn's `query` output (which converts reduced→true distance
before returning, `:1191`). The public generic `euclidean_distance` is the
same formula over `F: Float` and is the distance primitive consumed by the
radius-search fallbacks. There is **no reduced-distance fast path** and **no
non-Euclidean metric** (REQ-5).

**Invariants held vs sklearn:** for a single query row with distinct distances,
`query` returns the same neighbor set, in nearest-first order, with
full-precision matching Euclidean distances (AC-1; `test_kdtree_matches_brute_force`
already cross-checks the tree against `brute_force_knn` for `k = 1..=8`).
Single-point and empty-data builds do not panic.

**Invariants NOT held vs sklearn:** constructor `leaf_size`/`metric` (REQ-1);
`query_radius` (REQ-3); `sort_results`/tie-order contract + radius ordering
(REQ-4); non-Euclidean metrics (REQ-5); leaf bucketing + node bounds + `min_rdist`
prune + `leaf_size` result-invariance (REQ-6); `ValueError` on `k > n_samples`
and empty `X` (REQ-7); batched `(n_query, n_features)` query (REQ-8); the PyO3
`KDTree` binding (REQ-9); the ferray substrate (REQ-10).

**On tie-breaking (REQ-4 detail).** sklearn does NOT guarantee ascending-index
tie order; the order of equal-distance neighbors follows heap-pop / traversal
and shifts with `leaf_size` (`[0,1,2,3]` at 40 vs `[0,2,1,3]` at 1/2 for the
unit-square corners). ferrolearn's `NeighborHeap::into_sorted` uses a stable
`sort_by` on distance, so its tie order follows insertion (traversal) order —
which need not equal sklearn's for any given `leaf_size`. The contract a parity
test can assert is therefore **set-of-indices + multiset-of-distances**, not a
fixed permutation; this is itself a sklearn property, documented here so a critic
does not over-pin a spurious index-order divergence.

## Verification

Library crate (green at baseline `2013942c` for the existing — narrower —
contract):
```
cargo test -p ferrolearn-neighbors --lib kdtree
cargo clippy -p ferrolearn-neighbors --all-targets -- -D warnings
cargo fmt --all --check
```
The existing `#[test]`s (`test_build_empty_tree`, `test_build_single_point`,
`test_query_simple`, `test_query_k_neighbors`, `test_kdtree_matches_brute_force`,
`test_brute_force_simple`, `test_euclidean_distance*`) pin ferrolearn's current
single-row Euclidean k-NN behavior and the tree-vs-brute-force agreement; they do
NOT establish the full `KDTree` contract, so they make only REQ-2 SHIPPED.

**Known crate-gauntlet blocker — clippy `collapsible_if` (#819).**
`kdtree.rs:315` (`fn search_recursive`) nests
`if plane_dist < heap.worst_distance() { if let Some(child) = second { ... } }`.
On the workspace MSRV (1.88, raised by the ferray substrate) the let-chain form
is stable, so clippy's `collapsible_if` lint fires under `-D warnings` and wants
it collapsed to `if plane_dist < heap.worst_distance() && let Some(child) =
second { ... }`. This blocks the `cargo clippy --all-targets -- -D warnings`
gate for this crate and must be cleared by an acto-fixer (collapse via let-chain;
lint-only, no behavior change). NO `.rs` edit is made by this doc-author
dispatch.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a
critic should pin (R-CHAR-3 expected values), `X = [[0,0],[1,0],[0,1],[1,1],[10,10]]`:
```
# REQ-2 (present, must stay green): k-NN value parity
python3 -c "import numpy as np; from sklearn.neighbors import KDTree; d,i=KDTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).query(np.array([[0.1,0.1]]),k=2); print(d.tolist(),i.tolist())"  # [[0.1414...,0.9055...]] [[0,1]]
# REQ-3 (query_radius): inexpressible on KdTree
python3 -c "import numpy as np; from sklearn.neighbors import KDTree; print([a.tolist() for a in KDTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).query_radius(np.array([[0.1,0.1]]),r=1.5)])"  # [[0,1,2,3]]
python3 -c "import numpy as np; from sklearn.neighbors import KDTree; print(KDTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).query_radius(np.array([[0.1,0.1]]),r=1.5,count_only=True).tolist())"  # [4]
# REQ-4 (sort_results ascending vs default unsorted)
python3 -c "import numpy as np; from sklearn.neighbors import KDTree; i,d=KDTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).query_radius(np.array([[0.1,0.1]]),r=1.5,return_distance=True,sort_results=True); print([a.tolist() for a in i])"  # [[0,2,1,3]]
# REQ-4 (tie order varies with leaf_size — compare SET, not permutation)
python3 -c "import numpy as np; from sklearn.neighbors import KDTree; X=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]]); print([KDTree(X,leaf_size=ls).query(np.array([[0.5,0.5]]),k=4)[1].tolist() for ls in (1,2,40)])"  # [[[0,2,1,3]],[[0,2,1,3]],[[0,1,2,3]]]
# REQ-5 (metric set): manhattan / chebyshev
python3 -c "import numpy as np; from sklearn.neighbors import KDTree; print(KDTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]]),metric='manhattan').query(np.array([[0.1,0.1]]),k=1)[0].tolist())"  # [[0.2]]
# REQ-7 (k>n_samples and empty raise ValueError)
python3 -c "import numpy as np; from sklearn.neighbors import KDTree; KDTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).query(np.array([[0.,0.]]),k=10)"  # ValueError: k must be <= n training points
python3 -c "import numpy as np; from sklearn.neighbors import KDTree; KDTree(np.zeros((0,2)))"  # ValueError: X is an empty array
# REQ-6 (leaf_size invariance of distances)
python3 -c "import numpy as np; from sklearn.neighbors import KDTree; X=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]]); print([[round(x,6) for x in KDTree(X,leaf_size=ls).query(np.array([[0.5,0.5]]),k=4)[0][0]] for ls in (1,2,40)])"  # all [0.707107]*4
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-neighbors/tests/divergence_kdtree.rs`, asserting the live-sklearn
expected values above and FAILING against current `kdtree.rs`. REQ-2 already
matches and should be guarded by a non-regression pin (set+distance comparison).

## Blockers to open

(Suggested numbers — the director creates the real crosslink issues.)

- #810 — REQ-1 (`build` defaults): no `leaf_size` (default 40, `>=1`,
  `_binary_tree.pxi.tp:852,861-862`) and no `metric` (`'minkowski'`); one-point-
  per-node tree vs leaf-bucketed `BinaryTree`.
- #811 — REQ-3 (`query_radius`): `KdTree` has no `query_radius`
  (`_binary_tree.pxi.tp:1201`); radius search is brute-force in
  `radius_neighbors.rs`/`nearest_neighbors.rs`. Pin: `query_radius([[0.1,0.1]],
  r=1.5)` → `[[0,1,2,3]]`.
- #812 — REQ-4 (tie-break / `sort_results`): `query` always sorts, no
  `sort_results`; no `query_radius` default-unsorted/ascending ordering
  (`:1222,1228-1233`). Parity must compare neighbor SET + distances (tie order is
  `leaf_size`-dependent).
- #813 — REQ-5 (metric set): Euclidean-only; sklearn `valid_metrics`
  euclidean/l2/minkowski/p/manhattan/cityblock/l1/chebyshev/infinity with
  reduced-distance machinery (`_kd_tree.pyx.tp:123-201`). Pin:
  `metric='manhattan'` k=1 → dist `0.2`.
- #814 — REQ-6 (`leaf_size` + bounds): no `leaf_size`, no per-node
  `node_bounds`/`min_rdist` prune (`_kd_tree.pyx.tp:73-147`); sklearn
  `leaf_size <= n_points <= 2*leaf_size`, results invariant to `leaf_size`
  (`:247-253`).
- #815 — REQ-7 (error contract): no `ValueError` on `k > n_samples`
  (`:1140-1142`) or empty `X` (`:855-856`); ferrolearn clamps / returns empty
  tree.
- #816 — REQ-8 (batched query): single-row `&[f64]` → flat `Vec`; sklearn
  `(n_query, n_features)` → `(d, i)` shape `(n_query, k)` + `return_distance`
  (`:1125-1131`).
- #817 — REQ-9 (PyO3 binding): no `ferrolearn-python` `KDTree` shim.
- #818 — REQ-10 (ferray substrate): migrate `kdtree.rs` off `ndarray`/`num-traits`
  to ferray (R-SUBSTRATE).
- #819 — Crate-gauntlet (not a sklearn divergence): clippy `collapsible_if` at
  `kdtree.rs:315` (`if plane_dist < heap.worst_distance() { if let Some(child) =
  second { ... } }` → let-chain collapse; MSRV 1.88). Blocks `cargo clippy
  --all-targets -- -D warnings`; acto-fixer clears it.
