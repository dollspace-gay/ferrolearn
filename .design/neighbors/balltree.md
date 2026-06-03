# Ball Tree (sklearn.neighbors.BallTree)

<!--
tier: 3-component
status: draft
baseline-commit: 8065e7d7
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/neighbors/_ball_tree.pyx.tp    # BallTree(BallTree64); allocate_data/init_node centroid+radius; min_dist/max_dist/min_rdist node-bounds specialization (:76-228); VALID_METRICS (:29-48)
  - sklearn/neighbors/_binary_tree.pxi.tp  # BinaryTree.__init__ (:851-856 empty-X ValueError, :861-862 leaf_size>=1, :851-852 leaf_size=40/metric='minkowski'); query (:1089-1142, :1191); query_radius (:1201-1260); kernel_density / two_point_correlation
ferrolearn-module: ferrolearn-neighbors/src/balltree.rs
parity-ops: BallTree (BallTree.__init__, BallTree.query, BallTree.query_radius)
crosslink-issue: 854
-->

## Summary

`ferrolearn-neighbors/src/balltree.rs` mirrors scikit-learn's `BallTree`
(`sklearn/neighbors/_ball_tree.pyx.tp`, a thin specialization of the shared
`BinaryTree` in `_binary_tree.pxi.tp`). It exposes `BallTree::build` /
`BallTree::build_with_leaf_size` (the constructor analog, `DEFAULT_LEAF_SIZE =
40`), `BallTree::query` (the `k`-nearest-neighbor analog), and
`BallTree::within_radius` (the radius-search analog). The tree is the
spatial-index backend consumed (non-test) by the `BallTree` `Algorithm` arm of
`KNeighborsClassifier`/`Regressor` (`knn.rs`), `NearestNeighbors`
(`nearest_neighbors.rs`), and `RadiusNeighbors` (`radius_neighbors.rs`).

Under honest underclaim (R-HONEST-3), the **behaviors that are genuinely present
and oracle-matching** are:

- `query` (k-NN): for a single query point it returns the `k` nearest neighbors
  as `(index, distance)` pairs, sorted nearest-distance first, with **true
  Euclidean distance** matching the live sklearn 1.5.2 `BallTree` oracle to full
  precision on a tie-free fixture.
- `within_radius`: for a single query point it returns all points within the
  radius as `(index, distance)` pairs (unsorted), matching sklearn's
  `query_radius(..., return_distance=True)` **as a set** (sklearn's default is
  also `sort_results=False`).

Everything else diverges from `BallTree`'s contract:

1. The constructor takes a `leaf_size` (default 40, matching
   `_binary_tree.pxi.tp:851-852`) but **no `metric`** (sklearn `metric='minkowski'`
   `p=2` with a large `VALID_METRICS` set, `_ball_tree.pyx.tp:29-48`). ferrolearn
   is **Euclidean-only** (`fn dist_sq`).
2. `query` takes a **single** query row (`&[f64]`) and returns a flat
   `Vec<(usize, f64)>`, not sklearn's batched `(n_query, n_features)` → `(d, i)`
   of shape `(n_query, k)` (`_binary_tree.pxi.tp:1125-1131`); there is no
   `return_distance` / `sort_results` / `dualtree` / `breadth_first` toggle
   (`:1089-1091`).
3. `query` does **not raise on `k > n_samples`**; sklearn raises `ValueError`
   ("k must be less than or equal to the number of training points",
   `:1140-1142`). ferrolearn silently returns `min(k, n_samples)` neighbors.
4. `build` on an **empty array returns an empty tree**; sklearn raises
   `ValueError("X is an empty array")` (`:855-856`, surfaced through
   `check_array` as "Found array with 0 sample(s)").
5. `within_radius` is single-row and has **no `count_only` / `return_distance` /
   `sort_results`** toggles (sklearn `query_radius`, `:1201-1260`), and its
   `count_only`/`return_distance`/`sort_results` mutual-exclusion `ValueError`s
   (`:1254-1260`) are inexpressible.
6. ferrolearn **lacks** `kernel_density` and `two_point_correlation` (both
   present on sklearn's `BallTree`), and the `get_arrays` / tree-stats
   introspection.
7. The public surface is **not exposed by `ferrolearn-python`** — `import
   sklearn.neighbors` gives `BallTree`; `import ferrolearn` gives nothing.
8. The crate is on the **`ndarray` substrate** (`use ndarray::Array2`), not
   ferray (R-SUBSTRATE).

`BallTree::build` / `build_with_leaf_size` / `query` / `within_radius` are
existing pub APIs consumed across `knn.rs`, `nearest_neighbors.rs`,
`radius_neighbors.rs` (grandfathered per S5/R-DEFER-1); that wiring is the
non-test production-consumer surface.

## Algorithm (sklearn — the contract)

### Construction (`BinaryTree.__init__`, `_binary_tree.pxi.tp:851-896`)

`BallTree(X, leaf_size=40, metric='minkowski', sample_weight=None, **kwargs)`
validates `X` is non-empty (`:855-856` → `ValueError("X is an empty array")`),
requires `leaf_size >= 1` (`:861-862`), resolves the `DistanceMetric` (default
Minkowski `p=2` = Euclidean), and recursively partitions a contiguous
`idx_array` into nodes that each hold a **bucket** of points. A node splits only
while `idx_end - idx_start > 2 * leaf_size`; the split is on the **dimension of
greatest spread** (widest axis), partitioning the index range about its median.
For ball trees, `init_node` (`_ball_tree.pyx.tp:86-145`) stores a per-node
**centroid** (`node_bounds[0, i_node, :]`, the mean of the node's points,
`:111-132`) and a scalar **radius** = the max distance from the centroid to any
point in the node (`:134-142`). The leaf-size invariant `leaf_size <= n_points
<= 2*leaf_size` holds except when `n_samples < leaf_size`; **`leaf_size` changes
performance and tree shape, not query results**.

### `query` (k-NN, `_binary_tree.pxi.tp:1089-1199`)

`query(X, k=1, return_distance=True, dualtree=False, breadth_first=False,
sort_results=True)`. For each row of `X` (shape `(n_query, n_features)`):
validates dimension match (`:1136-1138`), raises `ValueError` if
`n_samples < k` (`:1140-1142`), and runs a depth-first search maintaining a
bounded **max-heap** of the `k` best `(reduced_distance, index)`. The
node-prune test uses the **ball lower bound** `min_dist = max(0, dist(pt,
centroid) - radius)` (`_ball_tree.pyx.tp:148-156`; the reduced/squared form
`min_rdist` at `:186-200`): a node is descended only if its `min_dist` is below
the heap's current worst. On return, results are converted from reduced to true
distance and, with `sort_results=True` (default), each row's `k` neighbors are
sorted **nearest-distance-first** (`:1191`). Returns `(d, i)` of shape
`(n_query, k)`.

**Exact ties** (multiple neighbors at identical distance) are **not stably
ordered**: the per-row order is the heap pop / traversal order and depends on
`leaf_size` and node layout. Oracle: querying `(0.5,0.5)` against the four
unit-square corners (all at distance `sqrt(0.5)`) yields index order
`[0,1,2,3]` at `leaf_size=40` but `[0,2,1,3]` at `leaf_size=1` or `2`. The
**distances and the neighbor set are invariant**; only the tie ordering shifts.

### `query_radius` (`_binary_tree.pxi.tp:1201-1260`)

`query_radius(X, r, return_distance=False, count_only=False,
sort_results=False)`. Returns, per query row, all indices within radius `r`
(object array). `count_only=True` returns counts; `return_distance=True` returns
`(ind, dist)`. **Results are unsorted by default** (`sort_results=False`,
`:1228`); `sort_results=True` (requires `return_distance=True`, else `ValueError`
`:1258-1260`) sorts each row's neighbors by ascending distance. `count_only` and
`return_distance` together raise `ValueError` (`:1254-1256`). The ball-prune
uses `min_dist`/`max_dist` from the node centroid and radius
(`_ball_tree.pyx.tp:148-167`): a node entirely outside `r` is pruned, a node
entirely inside `r` is bulk-included.

### `kernel_density` / `two_point_correlation`

`BallTree` additionally exposes `kernel_density(X, h, kernel='gaussian', ...)`
and `two_point_correlation(X, r, dualtree=False)` (both present on the live
oracle: `hasattr(BallTree, 'kernel_density') == True`, ditto
`two_point_correlation`). ferrolearn has neither.

### Edge cases (live oracle, sklearn 1.5.2, run from /tmp)

- `BallTree(np.empty((0,2)))` → `ValueError("Found array with 0 sample(s)
  (shape=(0, 2)) while a minimum of 1 is required.")`.
- `BallTree([[0,0],[1,1]]).query([[0,0]], k=5)` → `ValueError("k must be less
  than or equal to the number of training points")`.
- `query(..., k=0)` → returns empty `(1, 0)` arrays (no error).

## ferrolearn (what exists)

All in `ferrolearn-neighbors/src/balltree.rs`:

- **`pub struct BallTree`** — a **flat-node** ball tree: `nodes: Vec<Node>`
  (private `struct Node` with `centroid: Vec<f64>`, `radius_sq: f64`,
  `start`/`end` into a permuted `indices` array, and a `NodeKind::Leaf` /
  `NodeKind::Branch { left, right }`), `data: Vec<f64>` (flattened to `f64`),
  `n_features`, `indices`, and `_leaf_size`. This mirrors sklearn's centroid +
  radius node layout (`_ball_tree.pyx.tp:111-142`), with `radius_sq` held as the
  **squared** radius for the all-squared-distance fast path.
- **`pub fn build<F>(data: &Array2<F>) -> Self`** — delegates to
  `build_with_leaf_size` with `DEFAULT_LEAF_SIZE = 40` (matches sklearn's
  `leaf_size=40`, `_binary_tree.pxi.tp:851-852`).
- **`pub fn build_with_leaf_size<F>(data: &Array2<F>, leaf_size: usize) ->
  Self`** — flattens data to `f64`, then `fn build_recursive` computes each
  node's true centroid and squared radius in one pass, splits on the
  **dimension of greatest spread** (max−min per dim) via
  `select_nth_unstable_by` at the median position (`mid = start + count / 2`),
  and recurses while `count > leaf_size`. `leaf_size` is clamped to `>= 1`
  (`leaf_size.max(1)`). **Empty input → empty tree** (returns the
  all-empty `Self`), no error. No `metric` arg.
- **`pub fn query<F>(&self, _data: &Array2<F>, query: &[f64], k: usize) ->
  Vec<(usize, f64)>`** — single query row; depth-first `fn knn_search`
  maintaining a bounded max-heap (`struct KnnCandidate`, `BinaryHeap`). The
  node-prune is `ball_lower_bound_sq(query, centroid, radius_sq) > worst_sq`
  (`fn ball_lower_bound_sq` = `max(0, dist - radius)^2`, the correct
  squared-Euclidean lower bound — the analog of sklearn's `min_rdist`,
  `_ball_tree.pyx.tp:186-200`); the closer child is descended first. Results
  are drained and **sorted ascending by true Euclidean distance**
  (`results.sort_by(...)`). The `_data` parameter is accepted for API
  compatibility but unused (data is stored at build). **No `k > n_samples`
  guard** (the heap simply holds `min(k, n_samples)`); no batched query; no
  `return_distance`/`sort_results`/`dualtree` toggle. `k == 0` and empty tree
  return an empty `Vec`.
- **`pub fn within_radius(&self, query: &[f64], radius: f64) ->
  Vec<(usize, f64)>`** — depth-first `fn radius_search` with the same ball
  prune: a node whose `dist_to_center - node_radius > radius` is skipped, a node
  whose `dist_to_center + node_radius <= radius` is bulk-included, otherwise the
  leaf points are tested against `radius_sq`. Returns `(index, true-distance)`
  pairs **unsorted** (matching sklearn's `sort_results=False` default).

**Consumers (non-test):**
- `knn.rs` (`fn find_neighbors`): `SpatialIndex::BallTree(tree) => tree.query(data,
  &query_f64, k)` under `Algorithm::BallTree`, and `Algorithm::Auto` when
  `n_features > 15` (`fn build_spatial_index`).
- `nearest_neighbors.rs`: `SpatialIndex::BallTree(tree) => tree.query(...)` in
  the k-NN path and `tree.within_radius(...)` in `fn find_radius`.
- `radius_neighbors.rs`: `SpatialIndex::BallTree(tree) => tree.within_radius(
  &query_f64, radius_f64)` for the radius search; `BallTree::build` in the
  index-builder.

These are existing pub APIs (grandfathered, S5/R-DEFER-1).

## Requirements

- REQ-1: **`query` k-NN value (R-DEV-1/3).** Match `query` (`:1089`): return the
  `k` nearest neighbors as **true Euclidean distances**, sorted nearest-first
  (`sort_results=True`, `:1191`). ferrolearn's `query` matches the live `BallTree`
  oracle on distances and on the neighbor set/order for a single query row when
  distances are distinct.
- REQ-2: **Tie-set handling (R-DEV-1/3).** Under exact-distance ties, sklearn's
  distances and neighbor **set** are invariant but the tie order is
  traversal/`leaf_size`-dependent (`[0,1,2,3]` at leaf_size 40, `[0,2,1,3]` at
  1/2). ferrolearn must return the same neighbor set with the same multiset of
  distances; the index permutation among ties is not a fixed contract.
- REQ-3: **`within_radius` value (R-DEV-1/3).** Match `query_radius(X, r,
  return_distance=True)` (`:1201`) **as a set**: the indices within radius `r`
  and their true distances, default-unsorted (`sort_results=False`, `:1228`).
  ferrolearn's `within_radius` matches the live oracle as a set.
- REQ-4: **`leaf_size` + constructor defaults (R-DEV-2).** Match
  `BallTree(X, leaf_size=40, metric='minkowski')` (`_binary_tree.pxi.tp:851-852`).
  ferrolearn has `leaf_size` (default 40, clamped `>=1`) but **no `metric`**
  parameter.
- REQ-5: **`query_radius` full surface (R-DEV-2/3).** Match `query_radius`'s
  `count_only` / `return_distance` / `sort_results` toggles, the
  ascending-distance sort when `sort_results=True` (`:1228-1233`), and the
  mutual-exclusion `ValueError`s (`count_only`+`return_distance`, and
  `sort_results` without `return_distance`, `:1254-1260`). ferrolearn's
  `within_radius` exposes none of these.
- REQ-6: **Metric set (R-DEV-2).** Match `VALID_METRICS`
  (`_ball_tree.pyx.tp:29-48`: euclidean/manhattan/chebyshev/minkowski/haversine/
  mahalanobis/seuclidean/canberra/braycurtis and the boolean metrics) and the
  reduced-distance machinery. ferrolearn is **Euclidean-only** (`fn dist_sq` /
  `fn ball_lower_bound_sq`).
- REQ-7: **Error contract (R-DEV-2).** Match `ValueError` on `k > n_samples`
  (`:1140-1142`) and on empty `X` (`:855-856`). ferrolearn silently clamps `k`
  to `min(k, n_samples)` and returns an empty tree.
- REQ-8: **Batched query shape (R-DEV-3).** Match `query(X)` accepting
  `(n_query, n_features)` and returning `(d, i)` of shape `(n_query, k)`
  (`:1125-1131`), with `return_distance` toggle. ferrolearn's `query` is
  single-row → flat `Vec`.
- REQ-9: **`kernel_density` (R-DEV-3).** Provide `BallTree.kernel_density(X, h,
  kernel=...)` (present on the live oracle). ferrolearn has none.
- REQ-10: **`two_point_correlation` (R-DEV-3).** Provide
  `BallTree.two_point_correlation(X, r, dualtree=False)` (present on the live
  oracle). ferrolearn has none.
- REQ-11: **PyO3 binding (R-DEFER-1).** `import sklearn.neighbors` exposes
  `BallTree`; `ferrolearn-python` exposes no `BallTree` shim.
- REQ-12: **ferray substrate (R-SUBSTRATE).** `balltree.rs` imports
  `ndarray::Array2` + `num_traits::Float`, not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (`from
sklearn.neighbors import BallTree`, run from `/tmp`), never literal-copied from
ferrolearn (R-CHAR-3). Fixed dataset `X = [[0,0],[1,0],[0,1],[1,1],[10,10]]`
unless noted; the tie-free query `[0.2,0.1]` has **strictly distinct** distances
to all five points (verified: `[0.2236, 0.8062, 0.9220, 1.2042, 13.9302]`, all
distinct).

- AC-1 (REQ-1, present & matching): `BallTree(X).query([[0.2,0.1]], k=3)` →
  distances `[0.223606797749979, 0.806225774829855, 0.9219544457292888]`,
  indices `[0, 1, 2]`. ferrolearn `BallTree::build(&X).query(&X, &[0.2,0.1], 3)`
  returns `[(0, 0.223606797749979), (1, 0.806225774829855), (2,
  0.9219544457292888)]` — **matches** to full precision (verified at k=1,2,3).
- AC-2 (REQ-2 tie-set): `BallTree(X, leaf_size=ls).query([[0.5,0.5]], k=4)` → all
  four distances `sqrt(0.5)=0.7071067811865476`; indices `[0,1,2,3]` at
  `leaf_size=40`, `[0,2,1,3]` at `leaf_size=1,2`. ferrolearn returns the four
  indices `{0,1,2,3}` (order `[0,1,2,3]`) all at distance
  `0.7071067811865476` — **set + distances match**; a parity test must compare
  the neighbor SET + distance multiset, not a fixed permutation.
- AC-3 (REQ-3, present & matching as set): `BallTree(X).query_radius([[0.2,0.1]],
  r=1.5, return_distance=True)` → indices `[[0,1,2,3]]`, distances `[[0.2236,
  0.8062, 0.9220, 1.2042]]` (default unsorted). ferrolearn
  `within_radius(&[0.2,0.1], 1.5)` returns `{(0,0.2236), (1,0.8062), (2,0.9220),
  (3,1.2042)}` — **matches as a set**.
- AC-4 (REQ-5 pin): `BallTree(X).query_radius([[0.2,0.1]], r=1.5,
  count_only=True)` → `[4]`; `count_only=True, return_distance=True` →
  `ValueError("count_only and return_distance cannot both be true")`;
  `sort_results=True, return_distance=False` → `ValueError("return_distance must
  be True if sort_results is True")`. ferrolearn `within_radius` cannot express
  `count_only`, `sort_results`, or those `ValueError`s.
- AC-5 (REQ-7 pin): `BallTree([[0,0],[1,1]]).query([[0,0]], k=5)` → sklearn
  raises `ValueError("k must be less than or equal to the number of training
  points")`; ferrolearn silently returns 2 neighbors (`[(0,0.0),
  (1,1.4142135623730951)]`). `BallTree(np.empty((0,2)))` → sklearn
  `ValueError("Found array with 0 sample(s) ...")`; ferrolearn returns an empty
  tree.
- AC-6 (REQ-6 metric): `BallTree(X, metric='manhattan').query([[0.2,0.1]], k=1)`
  → distance `0.30000000000000004` (L1), index `0`. ferrolearn is Euclidean-only
  and cannot express it.
- AC-7 (REQ-9/10 surface): `hasattr(BallTree, 'kernel_density') == True` and
  `hasattr(BallTree, 'two_point_correlation') == True` on the live oracle.
  ferrolearn `BallTree` has neither method.
- AC-8 (single point, must match): `BallTree([[1,2]]).query([[1,2]], k=1)` →
  `(0.0, 0)`. ferrolearn `test_build_single_point` already establishes this.

## REQ status table

Binary (R-DEFER-2). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2). Live oracle = installed sklearn 1.5.2, run from `/tmp`. Honest
underclaim (R-HONEST-3): three REQs are SHIPPED (single-row k-NN value, tie-set
handling, single-row radius value — the behaviors that actually exist and match
the oracle); the rest are NOT-STARTED with open prereq blockers (suggested
numbers — the director creates the real issues).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`query` k-NN value) | SHIPPED | impl `pub fn query` in `balltree.rs` (depth-first `fn knn_search` + bounded max-heap `BinaryHeap<KnnCandidate>`, prune `fn ball_lower_bound_sq`) returns `(index, true-Euclidean-distance)` sorted nearest-first, mirroring `query`/`sort_results=True` (`_binary_tree.pxi.tp:1089,1191`) and the centroid+radius node prune (`_ball_tree.pyx.tp:186-200`). Non-test consumers: `knn.rs` (`SpatialIndex::BallTree(tree) => tree.query(...)` in `fn find_neighbors`) and `nearest_neighbors.rs` (k-NN path). Live oracle (AC-1): `BallTree(X).query([[0.2,0.1]],k=3)` → dist `[0.223606797749979,0.806225774829855,0.9219544457292888]`, idx `[0,1,2]`; ferrolearn `query(&X,&[0.2,0.1],3)` matches to full precision (verified throwaway probe). Distinct-distance value parity; tie handling is REQ-2. |
| REQ-2 (tie-set handling) | SHIPPED | `pub fn query`'s `results.sort_by` on distance returns the full equal-distance neighbor SET with the correct distance multiset; sklearn does NOT guarantee a fixed tie permutation (`[0,1,2,3]` at leaf_size 40 vs `[0,2,1,3]` at 1/2, `_binary_tree.pxi.tp:1191`). Live oracle (AC-2): `BallTree(X,leaf_size=ls).query([[0.5,0.5]],k=4)` → set `{0,1,2,3}`, all dist `0.7071067811865476`; ferrolearn returns the same set, all at `0.7071067811865476` (verified probe). Consumer: same as REQ-1. A parity test compares SET + distance multiset, not a fixed permutation. |
| REQ-3 (`within_radius` value) | SHIPPED | impl `pub fn within_radius` in `balltree.rs` (depth-first `fn radius_search`, centroid+radius prune mirroring `min_dist`/`max_dist`, `_ball_tree.pyx.tp:148-167`) returns the in-radius `(index, true-distance)` set **unsorted**, matching `query_radius(..., return_distance=True)` default `sort_results=False` (`_binary_tree.pxi.tp:1201,1228`). Non-test consumers: `radius_neighbors.rs` (`SpatialIndex::BallTree(tree) => tree.within_radius(...)`) and `nearest_neighbors.rs` (`fn find_radius`). Live oracle (AC-3): `query_radius([[0.2,0.1]],r=1.5,return_distance=True)` → idx `[[0,1,2,3]]`, dist `[[0.2236,0.8062,0.9220,1.2042]]`; ferrolearn `within_radius(&[0.2,0.1],1.5)` matches **as a set** (verified probe). Set-value parity only; toggles/sort/errors are REQ-5. |
| REQ-4 (`leaf_size` + metric defaults) | NOT-STARTED | open prereq blocker #855. `build`/`build_with_leaf_size` provide `leaf_size` (default `DEFAULT_LEAF_SIZE=40`, clamped `>=1` via `leaf_size.max(1)`) — matching `_binary_tree.pxi.tp:851-852` — but there is **no `metric`** parameter (sklearn `'minkowski'`). Euclidean-only build; partial vs the constructor contract, so NOT-STARTED on the `metric` half. |
| REQ-5 (`query_radius` full surface) | NOT-STARTED | open prereq blocker #856. `within_radius` exposes no `count_only`/`return_distance`/`sort_results` toggle, no ascending-distance sort (`_binary_tree.pxi.tp:1228-1233`), and cannot raise the mutual-exclusion `ValueError`s (`:1254-1260`). Pin (AC-4): `query_radius([[0.2,0.1]],r=1.5,count_only=True)` → `[4]`; `count_only+return_distance` → `ValueError`. |
| REQ-6 (metric set) | NOT-STARTED | open prereq blocker #857. Euclidean-only (`fn dist_sq` / `fn ball_lower_bound_sq` hardcode squared-diff sum). sklearn `VALID_METRICS` (`_ball_tree.pyx.tp:29-48`) = euclidean/manhattan/chebyshev/minkowski/haversine/mahalanobis/seuclidean/canberra/braycurtis + boolean metrics with reduced-distance specialization. Pin (AC-6): `BallTree(X,metric='manhattan').query([[0.2,0.1]],k=1)` → dist `0.30000000000000004`. |
| REQ-7 (error contract) | NOT-STARTED | open prereq blocker #858 (same `query`→`Result` consumer-threading class as kdtree #831; thread through `knn.rs`/`nearest_neighbors.rs`/`radius_neighbors.rs`). `pub fn query` has **no `k > n_samples` guard** (clamps via heap), and `pub fn build` returns an empty tree on empty input. sklearn raises `ValueError` on both (`:1140-1142`, `:855-856`). Pin (AC-5): `query([[0,0]],k=5)` on 2 points → sklearn `ValueError`, ferro returns 2; `BallTree(np.empty((0,2)))` → sklearn `ValueError`. |
| REQ-8 (batched query shape) | NOT-STARTED | open prereq blocker #859. `pub fn query` takes a single `query: &[f64]` → flat `Vec<(usize,f64)>`; sklearn `query(X)` accepts `(n_query, n_features)` → `(d, i)` of shape `(n_query, k)` with `return_distance` toggle (`:1125-1131`). |
| REQ-9 (`kernel_density`) | NOT-STARTED | open prereq blocker #860. `BallTree` exposes `kernel_density(X, h, kernel=...)` (live oracle `hasattr == True`); ferrolearn `BallTree` has no such method. |
| REQ-10 (`two_point_correlation`) | NOT-STARTED | open prereq blocker #861. `BallTree` exposes `two_point_correlation(X, r, dualtree=False)` (live oracle `hasattr == True`); ferrolearn `BallTree` has no such method. |
| REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #862. `ferrolearn-python/src/` exposes no `BallTree` (grep `BallTree`/`balltree` → 0 hits); `import ferrolearn` cannot call what `import sklearn.neighbors` provides. |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #863. `balltree.rs` imports `ndarray::Array2` + `num_traits::Float`, not `ferray-core` (R-SUBSTRATE). |

## Architecture

`balltree.rs` is a single module exposing `pub struct BallTree` plus its
build/query/within_radius methods, with private helpers (`struct Node`, `enum
NodeKind`, `struct KnnCandidate`, `fn dist_sq`, `fn ball_lower_bound_sq`, `fn
build_recursive`). The tree is a **flat-node binary recursive median-split ball
tree**:

1. **Build** (`fn build` / `fn build_with_leaf_size` → `fn build_recursive`):
   the dataset is flattened to a `Vec<f64>`; each node computes — in one pass —
   its **true centroid** (mean of its points, matching sklearn
   `_ball_tree.pyx.tp:111-132`) and its **squared radius** (max squared distance
   from centroid to any contained point, the squared form of sklearn's node
   radius, `:134-142`). A node splits while `count > leaf_size` on the
   **dimension of greatest spread** (max−min per dim) via
   `select_nth_unstable_by` at the median (`mid = start + count / 2`); left/right
   subranges recurse. Nodes are stored in a flat `Vec<Node>` with index-based
   child references (`NodeKind::Branch { left, right }`) for cache-friendly
   traversal. `leaf_size` defaults to 40 (`DEFAULT_LEAF_SIZE`) and is clamped to
   `>=1` (`leaf_size.max(1)`) — matching sklearn's `leaf_size=40` / `>=1`
   validation, but sklearn *raises* on `leaf_size < 1` whereas ferrolearn clamps.

2. **k-NN query** (`fn query` → `fn knn_search`): a bounded max-heap
   (`BinaryHeap<KnnCandidate>`, capacity `k`) tracks the best `(dist_sq, index)`.
   The node prune is `ball_lower_bound_sq(query, centroid, radius_sq) >
   worst_sq` — `max(0, dist(pt, centroid) - radius)^2`, the correct
   squared-Euclidean lower bound and the analog of sklearn's `min_rdist`
   (`_ball_tree.pyx.tp:186-200`). The closer child (smaller centroid distance) is
   descended first for tighter pruning. Results are drained, `sqrt`-ed to true
   distance, and `sort_by`-sorted ascending — matching sklearn's reduced→true
   conversion and `sort_results=True` (`:1191-1192`).

3. **Radius query** (`fn within_radius` → `fn radius_search`): the same ball
   geometry drives a three-way test per node — prune if `dist_to_center -
   node_radius > radius` (entirely outside), bulk-include if `dist_to_center +
   node_radius <= radius` (entirely inside, the `min_dist`/`max_dist` pair
   `_ball_tree.pyx.tp:148-167`), otherwise test leaf points against `radius_sq`.
   Results are returned **unsorted**, matching `query_radius`'s
   `sort_results=False` default (`_binary_tree.pxi.tp:1228`).

**Distance:** the search uses squared Euclidean distance internally
(`fn dist_sq`) and converts to true distance (`sqrt`) only at the boundary, so
reported distances match sklearn's `query`/`query_radius` output (which converts
reduced→true distance before returning). There is **no non-Euclidean metric**
(REQ-6) and no reduced/true `DistanceMetric` abstraction.

**Invariants held vs sklearn:** for a single query row with distinct distances,
`query` returns the same neighbor set, in nearest-first order, with
full-precision matching Euclidean distances (AC-1); for exact ties the neighbor
SET and distance multiset match (AC-2, REQ-2); `within_radius` returns the same
in-radius set + distances as `query_radius(..., return_distance=True)` (AC-3).
The in-tree `test_balltree_matches_brute_force` / `test_high_dimensional` /
`test_large_dataset` cross-checks pin the tree against `brute_force_knn` for
`k = 1..=8`. Single-point and empty-data builds do not panic. The centroid +
radius node geometry directly mirrors sklearn's ball-node layout, so the build
is structurally a `BinaryTree`/`BallTree` (unlike `kdtree.rs`, which is
one-point-per-node) — REQ-4 is closer to shipped than its kdtree sibling
(`leaf_size` is present), and is NOT-STARTED only on the missing `metric` arm.

**Invariants NOT held vs sklearn:** constructor `metric` (REQ-4); `query_radius`
toggles / sort / mutual-exclusion errors (REQ-5); non-Euclidean metrics (REQ-6);
`ValueError` on `k > n_samples` and empty `X` (REQ-7); batched `(n_query,
n_features)` query (REQ-8); `kernel_density` (REQ-9); `two_point_correlation`
(REQ-10); the PyO3 `BallTree` binding (REQ-11); the ferray substrate (REQ-12).

**On tie-breaking (REQ-2 detail).** sklearn does NOT guarantee ascending-index
tie order; the order of equal-distance neighbors follows heap-pop / traversal
and shifts with `leaf_size` (`[0,1,2,3]` at 40 vs `[0,2,1,3]` at 1/2 for the
unit-square corners). ferrolearn's `query` uses `sort_by` on distance, so its
tie order follows the heap drain / traversal order — which need not equal
sklearn's for any given `leaf_size`. The contract a parity test can assert is
therefore **set-of-indices + multiset-of-distances**, not a fixed permutation;
this is itself a sklearn property, documented here so a critic does not over-pin
a spurious index-order divergence. ferrolearn happens to match sklearn's
`leaf_size=40` order `[0,1,2,3]` on this fixture, but that coincidence is not the
contract.

## Verification

Library crate (green at baseline `8065e7d7` for the existing — narrower —
contract):
```
cargo test -p ferrolearn-neighbors --lib balltree
cargo clippy -p ferrolearn-neighbors --all-targets -- -D warnings
cargo fmt --all --check
```
The existing `#[test]`s (`test_build_empty`, `test_build_single_point`,
`test_query_simple`, `test_query_k_neighbors`, `test_balltree_matches_brute_force`,
`test_high_dimensional`, `test_custom_leaf_size`, `test_within_radius_basic`,
`test_within_radius_empty`, `test_within_radius_all`,
`test_within_radius_brute_force_comparison`, `test_query_empty_tree`,
`test_query_k_zero`, `test_large_dataset`, `test_bounding_invariant`,
`test_duplicate_points`) pin ferrolearn's current single-row Euclidean k-NN +
radius behavior and the tree-vs-brute-force agreement; they do NOT establish the
full `BallTree` contract, so they make only REQ-1, REQ-2, REQ-3 SHIPPED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a
critic should pin (R-CHAR-3 expected values), `X =
[[0,0],[1,0],[0,1],[1,1],[10,10]]`. The deterministic divergences to pin FIRST
are REQ-7 (k>n / empty `ValueError`) and REQ-6 (metric), since both are
mechanically checkable single-value mismatches; the tie ordering (REQ-2) is NOT
a divergence (set-match) and must not be pinned as one:
```
# REQ-1 (present, must stay green): k-NN value parity (tie-free query)
python3 -c "import numpy as np; from sklearn.neighbors import BallTree; d,i=BallTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).query(np.array([[0.2,0.1]]),k=3); print(d.tolist(),i.tolist())"  # [[0.2236...,0.8062...,0.9220...]] [[0,1,2]]
# REQ-2 (tie SET parity — compare set+distances, NOT permutation)
python3 -c "import numpy as np; from sklearn.neighbors import BallTree; X=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]]); print([BallTree(X,leaf_size=ls).query(np.array([[0.5,0.5]]),k=4)[1].tolist() for ls in (1,2,40)])"  # [[[0,2,1,3]],[[0,2,1,3]],[[0,1,2,3]]]
# REQ-3 (within_radius SET parity)
python3 -c "import numpy as np; from sklearn.neighbors import BallTree; i,d=BallTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).query_radius(np.array([[0.2,0.1]]),r=1.5,return_distance=True); print([a.tolist() for a in i],[a.tolist() for a in d])"  # [[0,1,2,3]] [[0.2236,0.8062,0.9220,1.2042]]
# REQ-5 (query_radius toggles): count_only and mutual-exclusion errors
python3 -c "import numpy as np; from sklearn.neighbors import BallTree; print(BallTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).query_radius(np.array([[0.2,0.1]]),r=1.5,count_only=True).tolist())"  # [4]
# REQ-6 (metric set): manhattan
python3 -c "import numpy as np; from sklearn.neighbors import BallTree; print(BallTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]]),metric='manhattan').query(np.array([[0.2,0.1]]),k=1)[0].tolist())"  # [[0.30000000000000004]]
# REQ-7 (k>n_samples and empty raise ValueError)
python3 -c "import numpy as np; from sklearn.neighbors import BallTree; BallTree(np.array([[0.,0.],[1.,1.]])).query(np.array([[0.,0.]]),k=5)"  # ValueError: k must be <= n training points
python3 -c "import numpy as np; from sklearn.neighbors import BallTree; BallTree(np.empty((0,2)))"  # ValueError: Found array with 0 sample(s)
# REQ-9/10 (kernel_density / two_point_correlation present on the oracle)
python3 -c "from sklearn.neighbors import BallTree; print(hasattr(BallTree,'kernel_density'), hasattr(BallTree,'two_point_correlation'))"  # True True
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-neighbors/tests/divergence_balltree.rs`, asserting the live-sklearn
expected values above and FAILING against current `balltree.rs`. REQ-1, REQ-2,
REQ-3 already match and should be guarded by non-regression pins (set + distance
comparison, not a fixed tie permutation).

## Blockers to open

(Suggested numbers — the director creates the real crosslink issues. #858
shares the `query`→`Result` consumer-threading class with kdtree #831.)

- #855 — REQ-4 (`metric` default): `build` provides `leaf_size` (default 40,
  clamped `>=1`) but no `metric` parameter (sklearn `'minkowski'`,
  `_binary_tree.pxi.tp:851-852`). Euclidean-only build.
- #856 — REQ-5 (`query_radius` surface): `within_radius` has no
  `count_only`/`return_distance`/`sort_results` toggle, no ascending sort
  (`:1228-1233`), no mutual-exclusion `ValueError`s (`:1254-1260`). Pin:
  `query_radius(...,count_only=True)` → `[4]`.
- #857 — REQ-6 (metric set): Euclidean-only; sklearn `VALID_METRICS`
  (`_ball_tree.pyx.tp:29-48`). Pin: `metric='manhattan'` k=1 → dist
  `0.30000000000000004`.
- #858 — REQ-7 (error contract; query→Result consumer-threading, same class as
  kdtree #831): no `ValueError` on `k > n_samples` (`:1140-1142`) or empty `X`
  (`:855-856`); ferrolearn clamps `k` / returns an empty tree. Thread `query`
  through a `Result` and update `knn.rs`/`nearest_neighbors.rs`/
  `radius_neighbors.rs` consumers.
- #859 — REQ-8 (batched query): single-row `&[f64]` → flat `Vec`; sklearn
  `(n_query, n_features)` → `(d, i)` shape `(n_query, k)` + `return_distance`
  (`:1125-1131`).
- #860 — REQ-9 (`kernel_density`): sklearn `BallTree.kernel_density` present;
  ferrolearn has none.
- #861 — REQ-10 (`two_point_correlation`): sklearn
  `BallTree.two_point_correlation` present; ferrolearn has none.
- #862 — REQ-11 (PyO3 binding): no `ferrolearn-python` `BallTree` shim.
- #863 — REQ-12 (ferray substrate): migrate `balltree.rs` off `ndarray`/
  `num-traits` to ferray (R-SUBSTRATE).
