# Nearest Neighbors (sklearn.neighbors.NearestNeighbors)

<!--
tier: 3-component
status: draft
baseline-commit: 6e7e1fbf
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/neighbors/_unsupervised.py   # NearestNeighbors(KNeighborsMixin, RadiusNeighborsMixin, NeighborsBase); __init__ defaults n_neighbors=5/radius=1.0/algorithm='auto'/leaf_size=30/metric='minkowski'/p=2/metric_params=None/n_jobs=None (:132-153); fit → self._fit(X) (:159-176)
  - sklearn/neighbors/_base.py            # KNeighborsMixin.kneighbors (:751-:946): k>0 guard (:808), query_is_train=X is None (:815), n_neighbors>n_samples_fit ValueError (:828-832), self-exclusion sample_range/dup mask (:931-939); RadiusNeighborsMixin.radius_neighbors (default sort_results=False), effective_metric_/n_samples_fit_/_fit_X (:521-605)
ferrolearn-module: ferrolearn-neighbors/src/nearest_neighbors.rs
parity-ops: NearestNeighbors (NearestNeighbors.__init__, .fit, .kneighbors, .radius_neighbors)
crosslink-issue: 864
-->

## Summary

`ferrolearn-neighbors/src/nearest_neighbors.rs` mirrors scikit-learn's
`sklearn.neighbors.NearestNeighbors` (`sklearn/neighbors/_unsupervised.py`), the
unsupervised neighbor-search learner. `_unsupervised.py` is a thin estimator —
all real behavior lives in the shared `KNeighborsMixin.kneighbors` /
`RadiusNeighborsMixin.radius_neighbors` mixins in `sklearn/neighbors/_base.py`.
It exposes the unfitted `NearestNeighbors<F>` (`n_neighbors=5`,
`algorithm=Auto`, `leaf_size=30`) and the fitted `FittedNearestNeighbors<F>`
with `kneighbors`, `radius_neighbors`, `n_samples_fit`, and `shape`.

Under honest underclaim (R-HONEST-3), the **behaviors that are genuinely present
and oracle-matching** are:

- **`kneighbors(x, Some(k))` value (explicit query)** — for a query matrix `x`
  (the sklearn "explicit `X`" path: no self-exclusion), it returns the `k`
  nearest neighbors as an `(n_queries, k)` pair of true-Euclidean distances and
  training indices, sorted nearest-first, matching the live sklearn 1.5.2 oracle
  to full precision on a tie-free fixture.
- **`radius_neighbors(x, radius)` set** — for a query matrix `x` it returns all
  training points within `radius` as `(distances, indices)` per row, matching
  sklearn's `radius_neighbors(X, radius, return_distance=True)` **as a set** and
  matching the `sort_results=True` order (ferrolearn always sorts ascending).
- The **value error guards exist** — `kneighbors` rejects `k == 0`, `k >
  n_train`, and feature-dimension mismatch; `radius_neighbors` rejects a negative
  radius. The *behavior* (raising on overflow) is present; the **exception type
  + message** diverge from sklearn (REQ split below).

Everything else diverges from the `NearestNeighbors` contract:

1. **No `kneighbors(X=None)` self-exclusion path.** ferrolearn's `kneighbors`
   *requires* a query matrix `x`. sklearn's `kneighbors()` (no `X`) queries the
   training data and **excludes each point's own self-match**
   (`query_is_train = X is None`, `_base.py:815`; the `sample_range`/`dup`
   self-removal, `_base.py:931-939`). Passing the training matrix to ferrolearn's
   `kneighbors` returns the self-match (index `i`, distance 0) instead — a
   MISSING-surface divergence (R-DEV-3).
2. **`radius_neighbors` always sorts ascending**, whereas sklearn's default is
   `sort_results=False` → results come back in tree/brute (native) order. The
   neighbor SET matches; the ORDER diverges from the default (R-DEV-3).
3. **Missing constructor params**: `radius` (sklearn default **1.0** — ferrolearn
   has no `radius` field; radius is only a `radius_neighbors` argument),
   `metric='minkowski'`, `p=2`, `metric_params`, `n_jobs` — all ABSENT.
   ferrolearn is **Euclidean-only**.
4. **Error type/message + timing diverge.** `kneighbors(k > n_train)` →
   ferrolearn `FerroError::InvalidParameter`; sklearn `ValueError("Expected
   n_neighbors <= n_samples_fit, ...")` (`_base.py:828-832`). And `fit` rejects
   `n_samples < n_neighbors` (`InsufficientSamples`) **at fit time**, but sklearn
   does NOT error at `fit` — it only errors at `kneighbors` query time.
5. **No `kneighbors_graph` / `radius_neighbors_graph` are NOT-STARTED on the
   estimator** for `X=None` semantics — though `graph.rs` provides the standalone
   `kneighbors_graph`/`radius_neighbors_graph` free functions and the
   `FittedNearestNeighbors::{kneighbors_graph, radius_neighbors_graph}` methods
   (see Architecture; these are consumers of `kneighbors`/`radius_neighbors`).
6. **No PyO3 binding / meta-crate re-export** — `import sklearn.neighbors` gives
   `NearestNeighbors`; `import ferrolearn` gives nothing.
7. The crate is on the **`ndarray` substrate** (`use ndarray::Array2` +
   `num_traits::Float`), not ferray (R-SUBSTRATE).

`NearestNeighbors`/`FittedNearestNeighbors` are existing pub APIs re-exported at
the crate root (`ferrolearn-neighbors/src/lib.rs`: `pub use
nearest_neighbors::{FittedNearestNeighbors, NearestNeighbors}`) and consumed
**non-test** by `graph.rs` (the `kneighbors_graph`/`radius_neighbors_graph` free
functions + the `FittedNearestNeighbors::{kneighbors_graph,
radius_neighbors_graph}` methods). Grandfathered per S5/R-DEFER-1; that wiring is
the non-test production-consumer surface for the value REQs.

## Algorithm (sklearn — the contract)

### Construction (`NearestNeighbors.__init__`, `_unsupervised.py:132-153`)

`NearestNeighbors(*, n_neighbors=5, radius=1.0, algorithm='auto', leaf_size=30,
metric='minkowski', p=2, metric_params=None, n_jobs=None)` — all keyword-only
(`*`). The constructor forwards to `NeighborsBase.__init__`; `fit(X, y=None)`
(`:159-176`) just calls `self._fit(X)`, which validates `X`, resolves
`effective_metric_`/`effective_metric_params_` (`_base.py:521-544`), records
`n_samples_fit_` (`_base.py:600,605`) and `_fit_X`, and builds the chosen spatial
index (ball_tree / kd_tree / brute). `algorithm='auto'` picks kd_tree / ball_tree
/ brute by metric and dimensionality (`_base.py:617-647`).

### `kneighbors` (`KNeighborsMixin.kneighbors`, `_base.py:751-946`)

`kneighbors(X=None, n_neighbors=None, return_distance=True)`:

1. `n_neighbors` defaults to `self.n_neighbors`; **`n_neighbors <= 0` →
   `ValueError("Expected n_neighbors > 0. Got %d")`** (`:808`); a non-integer
   `n_neighbors` → `TypeError` (`:809-814`).
2. **`query_is_train = X is None`** (`:815`). When `X is None`, the query IS the
   training data; ferrolearn-relevant: sklearn then queries **`n_neighbors + 1`**
   neighbors and drops each row's self-column so the result excludes the point
   itself (`:818-820` add 1; the `sample_range`/`dup` self-removal at
   `:931-939`). When `X` is given (explicit), no self-removal — all neighbors
   (possibly including a duplicate of the query) are returned.
3. **`n_neighbors > n_samples_fit_` → `ValueError(f"Expected n_neighbors <=
   n_samples_fit, but n_neighbors = {n_neighbors}, n_samples_fit =
   {n_samples_fit}, n_samples = {X.shape[0]}")`** (`:828-832`).
4. Runs the chosen backend, converts reduced→true distance, **sorts each row
   nearest-first** (`_kneighbors_reduce_func` `argsort`, `:741-746`). Returns
   `(dist, ind)` each shape `(n_queries, n_neighbors)` when
   `return_distance=True`, else just `ind`.

**Live oracle (sklearn 1.5.2, /tmp), `X=[[0,0],[1,0],[0,1],[1,1],[10,10]]`, query
`[[0.2,0.1]]` (tie-free: distances `[0.2236, 0.8062, 0.9220, 1.2042, 13.93]`, all
distinct):**
`NearestNeighbors(n_neighbors=3).fit(X).kneighbors([[0.2,0.1]])` →
`d=[[0.223606797749979, 0.8062257748298549, 0.9219544457292888]]`,
`i=[[0, 1, 2]]`; at `n_neighbors=2` → `d=[[0.2236, 0.8062]]`, `i=[[0, 1]]`.

**Self-exclusion oracle (`X=None`):**
`NearestNeighbors(n_neighbors=2).fit(X).kneighbors()` →
`i=[[1,2],[3,0],[3,0],[1,2],[3,1]]`, `d=[[1.,1.],[1.,1.],[1.,1.],[1.,1.],[12.73,
13.45]]` — each row excludes its own index (row 0 → `{1,2}`, not `{0,...}`).

### `radius_neighbors` (`RadiusNeighborsMixin.radius_neighbors`, `_base.py`)

`radius_neighbors(X=None, radius=None, return_distance=True,
sort_results=False)`. Returns, per query row, all training points within
`radius` as arrays-of-arrays (object dtype) `(distances, indices)`. **Default
`sort_results=False`** → results are returned in tree/brute (native) order, NOT
sorted by distance; `sort_results=True` sorts each row ascending. `X is None`
excludes self (same mechanism as `kneighbors`). Negative `radius` is rejected by
parameter validation.

**Live oracle, sort default (`X=[[10,10],[1,0],[0,1],[0,0],[1,1]]`, query
`[[0.2,0.1]]`, `radius=2.0`):**
`radius_neighbors([[0.2,0.1]])` (default `sort_results=False`) →
`i=[[1,2,3,4]]`, `d=[[0.8062, 0.922, 0.2236, 1.2042]]` (NOT ascending — native
order); `radius_neighbors([[0.2,0.1]], sort_results=True)` → `i=[[3,1,2,4]]`
(ascending). The neighbor SET `{1,2,3,4}` is invariant; only the order differs.

### Error/edge cases (live oracle, sklearn 1.5.2, /tmp)

- `NearestNeighbors(n_neighbors=3).fit([[0,0],[1,1]])` (2 rows) → **NO error at
  fit** (sklearn defers the check to query time).
- `...fit(X).kneighbors([[0.2,0.1]], n_neighbors=100)` → `ValueError("Expected
  n_neighbors <= n_samples_fit, but n_neighbors = 100, n_samples_fit = 5,
  n_samples = 1")`.
- `...kneighbors([[0.2,0.1]], n_neighbors=0)` → `ValueError("Expected n_neighbors
  > 0. Got 0")`.

## ferrolearn (what exists)

All in `ferrolearn-neighbors/src/nearest_neighbors.rs`, generic over
`F: Float (+ Send + Sync + 'static)`; `ndarray` substrate.

- **`pub struct NearestNeighbors<F> { pub n_neighbors: usize, pub algorithm:
  Algorithm, pub leaf_size: usize, _marker }`** — the unfitted estimator. **No
  `radius`/`metric`/`p`/`metric_params`/`n_jobs` fields.**
- **`pub fn new`** sets `n_neighbors=5`, `algorithm=Auto`, `leaf_size=30`
  (matching sklearn `:135-138`). Builder setters `with_n_neighbors` /
  `with_algorithm` / `with_leaf_size`; `impl Default → new()`.
- **`impl Fit<Array2<F>, ()> for NearestNeighbors<F>` / `pub fn fit`** — rejects
  `n_neighbors == 0` (`InvalidParameter`) and `n_samples < n_neighbors`
  (`InsufficientSamples`), then calls `build_spatial_index(self.algorithm, x)`
  and stores `x_train`, `n_neighbors`, the index. The `n_samples < n_neighbors`
  fit-time guard has **no sklearn analog** (sklearn defers to query time).
- **free `fn build_spatial_index`** — `Algorithm::Auto`: `KdTree` if `n_features
  <= 15` else `BallTree`; `KdTree`/`BallTree`/`BruteForce` map directly. (This is
  a simplification of sklearn's metric-aware `auto` rule, `_base.py:617-647`.)
- **`pub struct FittedNearestNeighbors<F> { x_train, n_neighbors, spatial_index }`**.
- **`pub fn kneighbors(&self, x: &Array2<F>, n_neighbors: Option<usize>) ->
  Result<(Array2<F>, Array2<usize>)>`** — `k = n_neighbors.unwrap_or(self.
  n_neighbors)`. Validates `x.ncols() == train_features` (else `ShapeMismatch`),
  `k != 0` (else `InvalidParameter`), `k <= n_train` (else `InvalidParameter`
  with message `"n_neighbors={k} exceeds number of training samples={n_train}"`).
  Per query row, `find_knn` dispatches to `KdTree::query` / `BallTree::query` /
  `kdtree::brute_force_knn`; results fill `(n_queries, k)` distance + index
  arrays. **Sorted nearest-first** (the backends sort). **No `X=None`
  self-exclusion path** (a query matrix is mandatory); **no `return_distance`
  toggle** (always returns both).
- **`pub fn radius_neighbors(&self, x: &Array2<F>, radius: F) ->
  Result<RadiusNeighborResult<F>>`** (`RadiusNeighborResult<F> = Vec<(Vec<F>,
  Vec<usize>)>`). Validates feature match (`ShapeMismatch`) and `radius >= 0`
  (else `InvalidParameter`). Per query row, `find_radius` dispatches to
  `BallTree::within_radius` (ball mode) or `brute_force_radius` (kd/brute mode),
  then **always sorts ascending by distance** (`neighbors.sort_by(...)`).
  Returns `(distances, indices)` per row. **No `sort_results` toggle** (always
  sorts); **no `X=None` self-exclusion; no `count_only`/`return_distance`
  toggles**.
- **`pub fn n_samples_fit(&self) -> usize`** (`x_train.nrows()`, the
  `n_samples_fit_` analog) and **`pub fn shape(&self) -> (usize, usize)`**.
- private **`fn find_knn`** / **`fn find_radius`** (backend dispatch) and free
  **`fn brute_force_radius`**.

**Consumers (non-test).** Crate re-export
(`ferrolearn-neighbors/src/lib.rs`: `pub use nearest_neighbors::{Fitted
NearestNeighbors, NearestNeighbors}`) plus `graph.rs`:
- `fn kneighbors_graph` (free function) builds a transient `NearestNeighbors`
  with `query_k = n_neighbors + 1` and calls `nn.kneighbors(x, Some(query_k))`,
  then `drop_self_neighbors` — emulating sklearn's `include_self=False` /
  `X=None` semantics at the *graph* level.
- `FittedNearestNeighbors::kneighbors_graph` (`graph.rs`) calls
  `self.kneighbors(x, n_neighbors)`; `FittedNearestNeighbors::
  radius_neighbors_graph` calls `self.radius_neighbors(x, radius)`.

These are existing pub APIs (grandfathered, S5/R-DEFER-1) and are the non-test
production consumers that back the value REQs.

## Requirements

- REQ-1: **`kneighbors` value, explicit query (R-DEV-1/3).** Mirror
  `KNeighborsMixin.kneighbors(X, n_neighbors=k)` for an explicit query matrix
  (`_base.py:751`, no self-removal): return `(dist, ind)` of shape `(n_queries,
  k)` with true-Euclidean distances, sorted nearest-first
  (`_base.py:741-746`). ferrolearn's `kneighbors(&x, Some(k))` value-matches the
  live oracle on a tie-free fixture.
- REQ-2: **`radius_neighbors` set (R-DEV-1/3).** Mirror `radius_neighbors(X,
  radius, return_distance=True)` **as a set**: the indices within `radius` and
  their true distances. ferrolearn's `radius_neighbors(&x, radius)` matches the
  oracle as a set (and matches the `sort_results=True` ascending order).
- REQ-3: **`kneighbors` error guards present (R-DEV-2 behavior).** Reject `k ==
  0`, `k > n_train`, and feature mismatch — the *raises-on-overflow* /
  shape-check behavior is present (mirroring the `:808`/`:828-832` checks),
  independent of the exact exception type (REQ-4).
- REQ-4: **`kneighbors` exact `ValueError` type/message + timing (R-DEV-2).**
  Match sklearn's `ValueError("Expected n_neighbors <= n_samples_fit, ...")`
  (`:828-832`) and `ValueError("Expected n_neighbors > 0. Got %d")` (`:808`), AND
  the timing: sklearn does **not** error at `fit` on `n_samples < n_neighbors`
  (defers to query). ferrolearn raises `FerroError::InvalidParameter` /
  `InsufficientSamples` with different messages, and errors at `fit`.
- REQ-5: **`kneighbors(X=None)` self-exclusion (R-DEV-3).** Provide the
  `X is None` path that queries the training data and **excludes each point's own
  self-match** (`:815`, `:931-939`). ferrolearn has no `X=None` path — passing the
  training matrix returns the self-match.
- REQ-6: **`radius_neighbors` `sort_results` default (R-DEV-3).** Match the
  default `sort_results=False` (native/unsorted order) and the `sort_results=True`
  toggle. ferrolearn **always sorts ascending** — matches `sort_results=True`,
  diverges from the default order.
- REQ-7: **Constructor `radius` + metric params (R-DEV-2).** Match `radius=1.0`,
  `metric='minkowski'`, `p=2`, `metric_params=None`, `n_jobs=None`
  (`_unsupervised.py:135-142`). ferrolearn lacks all of these (Euclidean-only; no
  `radius` field).
- REQ-8: **`kneighbors_graph` / `radius_neighbors_graph` on the estimator with
  `X=None` (R-DEV-3).** Match the mixins' `kneighbors_graph` /
  `radius_neighbors_graph` (`include_self`/`X=None` zero-diagonal semantics).
  `graph.rs` provides standalone free functions + `FittedNearestNeighbors`
  methods, but the estimator has no `X=None`/`include_self` path of its own.
- REQ-9: **PyO3 binding + meta-crate re-export (R-DEFER-1).** `import ferrolearn`
  exposes `NearestNeighbors` mirroring `import sklearn`. ferrolearn-python
  exposes no shim and the meta-crate no re-export.
- REQ-10: **ferray substrate (R-SUBSTRATE).** `nearest_neighbors.rs` imports
  `ndarray::Array2` + `num_traits::Float`, not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (`from
sklearn.neighbors import NearestNeighbors`, run from `/tmp`), never literal-copied
from ferrolearn (R-CHAR-3). Primary fixture `X =
[[0,0],[1,0],[0,1],[1,1],[10,10]]`, tie-free query `[0.2,0.1]` (distances
`[0.2236, 0.8062, 0.9220, 1.2042, 13.93]`, all distinct).

- AC-1 (REQ-1, present & matching): `NearestNeighbors(n_neighbors=3).fit(X).
  kneighbors([[0.2,0.1]])` → `d=[[0.223606797749979, 0.8062257748298549,
  0.9219544457292888]]`, `i=[[0,1,2]]`; `n_neighbors=2` → `d=[[0.2236, 0.8062]]`,
  `i=[[0,1]]`. ferrolearn `kneighbors(&xq, Some(3))` returns
  `[0.223606797749979, 0.806225774829855, 0.9219544457292888]` / `[0,1,2]` (and
  `Some(2)` → `[0.2236,0.8062]` / `[0,1]`) — **matches** to full precision
  (verified throwaway probe).
- AC-2 (REQ-2, present & matching as set): on `X2 =
  [[10,10],[1,0],[0,1],[0,0],[1,1]]`, `radius_neighbors([[0.2,0.1]])` (default
  unsorted) → set `{1,2,3,4}`, distances `{0.2236, 0.8062, 0.922, 1.2042}`.
  ferrolearn `radius_neighbors(&xq, 2.0)` returns `i=[3,1,2,4]`,
  `d=[0.2236,0.8062,0.922,1.2042]` — **matches as a set** (and equals sklearn's
  `sort_results=True` order `[3,1,2,4]`).
- AC-3 (REQ-3, present): ferrolearn `kneighbors(&xq, Some(0))` → `Err`,
  `kneighbors(&xq, Some(100))` → `Err`, feature-mismatch query → `Err`; the
  guard behavior is present (in-tree `test_kneighbors_k_zero`,
  `test_kneighbors_k_too_large`, `test_kneighbors_shape_mismatch`).
- AC-4 (REQ-4 pin): `NearestNeighbors(n_neighbors=2).fit(X).kneighbors([[0.2,
  0.1]], n_neighbors=100)` → sklearn `ValueError("Expected n_neighbors <=
  n_samples_fit, but n_neighbors = 100, n_samples_fit = 5, n_samples = 1")`;
  ferrolearn `FerroError::InvalidParameter { reason: "n_neighbors=100 exceeds
  number of training samples=5" }`. `kneighbors(..., n_neighbors=0)` → sklearn
  `ValueError("Expected n_neighbors > 0. Got 0")`; ferrolearn `InvalidParameter`.
  And `NearestNeighbors(n_neighbors=3).fit([[0,0],[1,1]])` → sklearn **no error**;
  ferrolearn `InsufficientSamples`.
- AC-5 (REQ-5 pin): `NearestNeighbors(n_neighbors=2).fit(X).kneighbors()` (no
  `X`) → `i=[[1,2],[3,0],[3,0],[1,2],[3,1]]` (row `i` excludes index `i`).
  ferrolearn has no `X=None` method; `kneighbors(&X, Some(2))` returns
  `i=[[0,1],...]` (row 0 includes self index 0 at distance 0).
- AC-6 (REQ-6 pin): on `X2`, `radius_neighbors([[0.2,0.1]])` default
  (`sort_results=False`) → `i=[[1,2,3,4]]` (native, NOT ascending);
  `sort_results=True` → `i=[[3,1,2,4]]`. ferrolearn always returns the ascending
  `[3,1,2,4]` — matches `sort_results=True`, diverges from the default order.
- AC-7 (REQ-7 defaults): `NearestNeighbors()` → `n_neighbors==5`, `radius==1.0`,
  `algorithm=='auto'`, `leaf_size==30`, `metric=='minkowski'`, `p==2`,
  `metric_params==None`, `n_jobs==None`. ferrolearn `new()` has `n_neighbors==5`,
  `algorithm==Auto`, `leaf_size==30` but **no `radius`/`metric`/`p`/
  `metric_params`/`n_jobs`** — REQ-7 FAILS on the missing fields.
- AC-8 (REQ-9 surface): `import ferrolearn; ferrolearn.NearestNeighbors` →
  AttributeError (no shim). `hasattr` is False; the meta-crate has no re-export.

## REQ status table

Binary (R-DEFER-2). `NearestNeighbors`/`FittedNearestNeighbors` are existing pub
APIs re-exported at the crate root and consumed non-test by `graph.rs` (the
production-consumer surface; grandfathered S5/R-DEFER-1). Cites use symbol
anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed
sklearn 1.5.2, run from `/tmp`. Honest underclaim (R-HONEST-3): three REQs are
SHIPPED (explicit-`X` k-NN value, radius set, the error-guard behavior); the rest
are NOT-STARTED with open prereq blockers (suggested numbers — the director
creates the real issues).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`kneighbors` value, explicit query) | SHIPPED | impl `pub fn kneighbors` in `nearest_neighbors.rs` (per-row `fn find_knn` → `KdTree::query`/`BallTree::query`/`kdtree::brute_force_knn`, fills `(n_queries,k)` arrays, sorted nearest-first) mirrors `KNeighborsMixin.kneighbors` explicit-`X` path (`_base.py:751`, sort `:741-746`). Non-test consumers: `graph.rs` `fn kneighbors_graph` (`nn.kneighbors(x, Some(query_k))`) and `FittedNearestNeighbors::kneighbors_graph` (`self.kneighbors(x, n_neighbors)`). Live oracle (AC-1): `NearestNeighbors(n_neighbors=3).fit(X).kneighbors([[0.2,0.1]])` → dist `[0.223606797749979,0.8062257748298549,0.9219544457292888]`, idx `[0,1,2]`; ferrolearn `kneighbors(&xq,Some(3))` matches to full precision (verified throwaway probe), and `Some(2)` → `[0.2236,0.8062]`/`[0,1]`. Explicit-query (no self-removal) value parity; self-exclusion is REQ-5, error type is REQ-4. |
| REQ-2 (`radius_neighbors` set) | SHIPPED | impl `pub fn radius_neighbors` in `nearest_neighbors.rs` (per-row `fn find_radius` → `BallTree::within_radius`/`fn brute_force_radius`, then `neighbors.sort_by` ascending) returns the in-radius `(distances, indices)` per row, matching `radius_neighbors(X, radius, return_distance=True)` **as a set**. Non-test consumer: `FittedNearestNeighbors::radius_neighbors_graph` (`graph.rs`: `self.radius_neighbors(x, radius)`). Live oracle (AC-2): on `X2=[[10,10],[1,0],[0,1],[0,0],[1,1]]`, `radius_neighbors([[0.2,0.1]])` → set `{1,2,3,4}`, dist `{0.2236,0.8062,0.922,1.2042}`; ferrolearn `radius_neighbors(&xq,2.0)` → `i=[3,1,2,4]`, `d=[0.2236,0.8062,0.922,1.2042]` — matches as a set (verified probe). Set-value parity only; the always-ascending sort vs the `sort_results=False` default is REQ-6. |
| REQ-3 (`kneighbors` error-guard behavior) | SHIPPED | `pub fn kneighbors` rejects `k == 0` (`InvalidParameter`), `k > n_train` (`InvalidParameter`), and feature mismatch (`ShapeMismatch`) — the *raises-on-overflow* / shape-check behavior present in sklearn (`:808`, `:828-832`) is present. Consumers: same as REQ-1; in-tree `test_kneighbors_k_zero`, `test_kneighbors_k_too_large`, `test_kneighbors_shape_mismatch` pin the guards. The exact `ValueError` type/message and fit-vs-query timing are REQ-4 (a distinct, NOT-STARTED requirement). |
| REQ-4 (exact `ValueError` type/message + timing) | NOT-STARTED | open prereq blocker #865. ferrolearn raises `FerroError::InvalidParameter { reason: "n_neighbors={k} exceeds number of training samples={n_train}" }` where sklearn raises `ValueError("Expected n_neighbors <= n_samples_fit, but n_neighbors = {k}, n_samples_fit = {n}, n_samples = {nq}")` (`_base.py:828-832`); `k==0` → ferro `InvalidParameter` vs sklearn `ValueError("Expected n_neighbors > 0. Got 0")` (`:808`). Timing: `fn fit` rejects `n_samples < n_neighbors` (`InsufficientSamples`) at fit time, but sklearn does NOT error at fit (defers to `kneighbors`). Pin (AC-4): `kneighbors(..., n_neighbors=100)` message; `fit(n=3 on 2 rows)` → sklearn OK, ferro errors. |
| REQ-5 (`kneighbors(X=None)` self-exclusion) | NOT-STARTED | open prereq blocker #866. `pub fn kneighbors` requires a query matrix `x` — there is no `X=None` path. sklearn `kneighbors()` (no `X`) sets `query_is_train=X is None` (`:815`) and excludes each point's self-match via the `sample_range`/`dup` mask (`:931-939`). Passing the training matrix to ferrolearn returns the self-match (index `i`, distance 0). `graph.rs` re-implements self-removal at the graph level (`fn drop_self_neighbors`), but the estimator method has no `X=None` surface. Pin (AC-5): `kneighbors()` → `i=[[1,2],[3,0],...]`; ferro `kneighbors(&X,Some(2))` row 0 = `[0,1]`. |
| REQ-6 (`radius_neighbors` `sort_results` default) | NOT-STARTED | open prereq blocker #867. `pub fn radius_neighbors` ALWAYS sorts ascending (`neighbors.sort_by(...)`), matching sklearn `sort_results=True`, but sklearn's DEFAULT is `sort_results=False` → native (tree/brute) order; there is no `sort_results` toggle. Pin (AC-6): on `X2`, `radius_neighbors([[0.2,0.1]])` default → `i=[[1,2,3,4]]` (native, dist `[0.8062,0.922,0.2236,1.2042]` NOT ascending); ferro returns `[3,1,2,4]` (ascending). SET matches, ORDER diverges. |
| REQ-7 (constructor `radius` + metric params) | NOT-STARTED | open prereq blocker #868. `pub struct NearestNeighbors` has `n_neighbors`/`algorithm`/`leaf_size` (matching `:135-138`) but **no `radius` (sklearn 1.0, `:136`)**, **no `metric='minkowski'` (`:139`)**, **no `p=2` (`:140`)**, **no `metric_params` (`:141`)**, **no `n_jobs` (`:142`)** — Euclidean-only; radius is only a `radius_neighbors` argument. Pin (AC-7): `NearestNeighbors().radius == 1.0`, `.metric == 'minkowski'`, `.p == 2`; ferro lacks all. |
| REQ-8 (`kneighbors_graph` / `radius_neighbors_graph` `X=None` on estimator) | NOT-STARTED | open prereq blocker #869. The estimator has no `X=None`/`include_self` graph path. `graph.rs` provides the standalone `pub fn kneighbors_graph`/`radius_neighbors_graph` free functions (zero-diagonal `include_self=False` via `query_k=n+1`+`drop_self_neighbors`) and `FittedNearestNeighbors::{kneighbors_graph, radius_neighbors_graph}` methods, but those do not expose sklearn's `KNeighborsMixin.kneighbors_graph(X=None, mode=...)` method contract on `NearestNeighbors` itself. (graph.rs handles the standalone surface; see `.design/neighbors/graph.md`.) |
| REQ-9 (PyO3 binding + meta-crate re-export) | NOT-STARTED | open prereq blocker #870. No `RsNearestNeighbors` (or equivalent) in `ferrolearn-python/src/` and no meta-crate re-export in `ferrolearn/src/` (both verified absent by `grep -ni nearestneighbors`). `import ferrolearn` cannot construct `NearestNeighbors` nor call `kneighbors`/`radius_neighbors`. NOT-STARTED until the library REQs (esp. #865/#866/#867/#868) land and the shim exposes them. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #871. `nearest_neighbors.rs` imports `ndarray::Array2` + `num_traits::Float` (the wrong substrate); not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`nearest_neighbors.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`NearestNeighbors<F>` (params) → `Fit<Array2<F>, ()>` → `FittedNearestNeighbors<F>`
(the stored `x_train`, `n_neighbors`, and the built `SpatialIndex`). Generic over
`F: Float + Send + Sync + 'static`; every public method returns `Result<_,
FerroError>` (no panics in library code).

**Fit path (`fn fit`).** Validation rejects `n_neighbors == 0`
(`InvalidParameter`) and `n_samples < n_neighbors` (`InsufficientSamples`), then
`build_spatial_index` selects the backend: `Algorithm::Auto` → `KdTree` if
`n_features <= 15` else `BallTree`; `KdTree`/`BallTree`/`BruteForce` map directly
(the private `enum SpatialIndex { None, KdTree, BallTree }`). The
`n_samples < n_neighbors` fit-time guard is a **ferrolearn-only** check — sklearn
defers the `n_neighbors <= n_samples_fit` test to `kneighbors` query time
(`_base.py:828-832`), so a fit that sklearn accepts (e.g. `n_neighbors=3` on 2
rows) is rejected by ferrolearn (REQ-4). The `Auto` rule is also a simplification
of sklearn's metric-aware selection (`_base.py:617-647`).

**k-NN query (`fn kneighbors` → `fn find_knn`).** `k =
n_neighbors.unwrap_or(self.n_neighbors)`; feature-match, `k != 0`, `k <= n_train`
guards (REQ-3); then per query row `find_knn` dispatches to the backend
(`KdTree::query`/`BallTree::query`/`brute_force_knn`), converting each result to
`(index, true-Euclidean-distance)` and filling row `i` of the `(n_queries, k)`
distance + index arrays. The backends return nearest-first, so the output is
sorted ascending — matching sklearn's `argsort` (`_base.py:741-746`). This is the
**explicit-`X`** path only: every query is treated as foreign data, so the self
of a training point (when the training matrix is passed back) is **included** as
a 0-distance neighbor — unlike sklearn's `X=None` self-exclusion (REQ-5).

**Radius query (`fn radius_neighbors` → `fn find_radius`).** Feature-match and
`radius >= 0` guards; then per row `find_radius` dispatches to
`BallTree::within_radius` (ball mode) or `fn brute_force_radius` (kd/brute), and
the row is **always sorted ascending** by distance. This matches sklearn's
`sort_results=True` order but diverges from the `sort_results=False` default
native order (REQ-6); there is no toggle and no `X=None` self-exclusion.

**Distance:** Euclidean throughout (the kd/ball backends and `brute_force_radius`
use `kdtree::euclidean_distance`); there is **no non-Euclidean metric** and no
`metric`/`p`/`metric_params` surface (REQ-7).

**Consumer wiring (`graph.rs`).** The non-test production consumers of this file:
- `pub fn kneighbors_graph` (free function) constructs a transient
  `NearestNeighbors` with `query_k = n_neighbors + 1`, calls `nn.kneighbors(x,
  Some(query_k))`, then `drop_self_neighbors` to emulate sklearn's `X=None` /
  `include_self=False` zero-diagonal graph (`sklearn/neighbors/_graph.py:34-43`).
  This is where the `X=None` self-exclusion that REQ-5 lacks at the *method*
  level is re-implemented at the *graph* level.
- `FittedNearestNeighbors::kneighbors_graph` → `self.kneighbors(x, n_neighbors)`;
  `FittedNearestNeighbors::radius_neighbors_graph` → `self.radius_neighbors(x,
  radius)`. (Note: `graph.rs`'s `radius_neighbors_graph` *free function* consumes
  `FittedRadiusNeighborsClassifier::radius_neighbors`, a different type — see
  `.design/neighbors/graph.md`.)

**Missing fitted attributes vs sklearn (`Attributes`, `_unsupervised.py:74-95`):**
`effective_metric_` (`:76`), `effective_metric_params_` (`:79`), `n_features_in_`
(`:82`), `feature_names_in_` (`:87`). ferrolearn exposes only `n_samples_fit()`
(the `n_samples_fit_` analog) and `shape()`.

**Invariants held vs sklearn:** for an explicit query matrix with distinct
distances, `kneighbors(&x, Some(k))` returns the same neighbor set, nearest-first,
with full-precision Euclidean distances (AC-1); `radius_neighbors(&x, radius)`
returns the same in-radius set + distances as `radius_neighbors(X, radius,
return_distance=True)` (AC-2); the `k==0` / `k>n_train` / feature-mismatch guards
raise (AC-3). The in-tree `test_all_algorithms_agree_kneighbors` cross-checks
brute/kd/ball agreement (set + distance), `test_kneighbors_k1_self_match` /
`test_single_point` pin self-as-nearest, and the radius tests pin sorted-ascending
output.

**Invariants NOT held vs sklearn:** exact `ValueError` type/message + fit-vs-query
timing (REQ-4); `X=None` self-exclusion (REQ-5); `radius_neighbors`
`sort_results=False` default order (REQ-6); constructor `radius`/`metric`/`p`/
`metric_params`/`n_jobs` (REQ-7); the estimator-level `kneighbors_graph`/
`radius_neighbors_graph` `X=None` method contract (REQ-8); the PyO3 binding +
meta-crate re-export (REQ-9); the ferray substrate (REQ-10).

## Verification

Library crate (green at baseline `6e7e1fbf` for the existing — narrower —
contract):
```
cargo test -p ferrolearn-neighbors --lib nearest_neighbors
cargo clippy -p ferrolearn-neighbors --all-targets -- -D warnings
cargo fmt --all --check
```
The existing `#[test]`s (`test_fit_basic`, `test_fit_invalid_k_zero`,
`test_fit_insufficient_samples`, `test_kneighbors_default_k`,
`test_kneighbors_override_k`, `test_kneighbors_k1_self_match`,
`test_kneighbors_shape_mismatch`, `test_kneighbors_k_too_large`,
`test_kneighbors_k_zero`, `test_radius_neighbors_basic`,
`test_radius_neighbors_empty`, `test_radius_neighbors_negative_radius`,
`test_radius_neighbors_shape_mismatch`, `test_radius_neighbors_sorted_by_distance`,
`test_brute_force_algorithm`, `test_kdtree_algorithm`, `test_balltree_algorithm`,
`test_all_algorithms_agree_kneighbors`, `test_radius_neighbors_brute_force`,
`test_radius_neighbors_balltree`, `test_default_matches_new`, `test_f32_support`,
`test_single_point`, `test_radius_zero`) pin ferrolearn's current explicit-query
Euclidean k-NN + always-sorted radius behavior and the cross-backend agreement.
**None compares against the live sklearn oracle**, but the explicit-query value
parity (REQ-1), radius set parity (REQ-2), and error-guard behavior (REQ-3) hold
against the oracle (verified by throwaway probe), so those three are SHIPPED; the
rest are NOT-STARTED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values), `X = [[0,0],[1,0],[0,1],[1,1],[10,10]]`,
`X2 = [[10,10],[1,0],[0,1],[0,0],[1,1]]`. **Pin the deterministic ones FIRST**
(no tie-break ambiguity): REQ-5 (`X=None` self-exclusion — a fixed index set per
row), REQ-6 (radius native-vs-sorted order — a fixed permutation), and REQ-4
(`ValueError` message + fit-no-error timing); REQ-7 (missing params) is a
surface/hasattr check:
```
# REQ-1 (present, must stay green): explicit-query k-NN value parity (tie-free)
python3 -c "import numpy as np; from sklearn.neighbors import NearestNeighbors; d,i=NearestNeighbors(n_neighbors=3).fit(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).kneighbors(np.array([[0.2,0.1]])); print(d.tolist(),i.tolist())"  # [[0.2236...,0.8062...,0.9220...]] [[0,1,2]]
# REQ-2 (present, set parity): radius set
python3 -c "import numpy as np; from sklearn.neighbors import NearestNeighbors; d,i=NearestNeighbors(radius=2.0).fit(np.array([[10.,10.],[1.,0.],[0.,1.],[0.,0.],[1.,1.]])).radius_neighbors(np.array([[0.2,0.1]])); print([a.tolist() for a in i],[a.tolist() for a in d])"  # native [[1,2,3,4]] [[0.8062,0.922,0.2236,1.2042]]
# REQ-4 (exact ValueError + fit defers):
python3 -c "import numpy as np; from sklearn.neighbors import NearestNeighbors; NearestNeighbors(n_neighbors=2).fit(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).kneighbors(np.array([[0.2,0.1]]),n_neighbors=100)"  # ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 100, n_samples_fit = 5, n_samples = 1
python3 -c "import numpy as np; from sklearn.neighbors import NearestNeighbors; NearestNeighbors(n_neighbors=3).fit(np.array([[0.,0.],[1.,1.]])); print('fit OK, no error')"  # fit OK (ferro: InsufficientSamples)
# REQ-5 (X=None self-exclusion):
python3 -c "import numpy as np; from sklearn.neighbors import NearestNeighbors; d,i=NearestNeighbors(n_neighbors=2).fit(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).kneighbors(); print(i.tolist())"  # [[1,2],[3,0],[3,0],[1,2],[3,1]]  (row i excludes index i)
# REQ-6 (radius sort_results default native order):
python3 -c "import numpy as np; from sklearn.neighbors import NearestNeighbors; nn=NearestNeighbors(radius=2.0).fit(np.array([[10.,10.],[1.,0.],[0.,1.],[0.,0.],[1.,1.]])); print([a.tolist() for a in nn.radius_neighbors(np.array([[0.2,0.1]]))[1]], [a.tolist() for a in nn.radius_neighbors(np.array([[0.2,0.1]]),sort_results=True)[1]])"  # [[1,2,3,4]] [[3,1,2,4]]
# REQ-7 (defaults / missing params):
python3 -c "from sklearn.neighbors import NearestNeighbors as N; c=N(); print(c.n_neighbors,c.radius,c.algorithm,c.leaf_size,c.metric,c.p,c.metric_params,c.n_jobs)"  # 5 1.0 auto 30 minkowski 2 None None
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-neighbors/tests/divergence_nearest_neighbors.rs`, asserting the
live-sklearn expected values above and FAILING against current
`nearest_neighbors.rs`. REQ-1, REQ-2, REQ-3 already match and should be guarded
by non-regression pins (REQ-2 compares set + distance, not a fixed permutation,
since ferrolearn's always-ascending order differs from sklearn's default native
order — REQ-6 — and that order difference must NOT be pinned against REQ-2).

ferrolearn-python (REQ-9 binding parity, after the library REQs land):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_neighbors.py -q
```
asserting `ferrolearn.NearestNeighbors` matches `sklearn.neighbors.
NearestNeighbors` on `n_neighbors`/`radius`/`algorithm`/`leaf_size`/`metric`/`p`,
the `kneighbors`/`radius_neighbors` outputs, the `X=None` self-exclusion, and the
exact `ValueError` mapping.

## Blockers to open

(Director creates the real issues; numbers below are SUGGESTIONS continuing the
neighbors layer past the LOF block #845-#853 and the balltree block #855-#863.
#865 is the same error-type/`ValueError`-mapping class as the balltree #858
"query→Result" thread, but here it is the *estimator-level* message/timing
divergence, not a backend `query` signature change.)

- #865 — REQ-4 (exact `ValueError` type/message + timing): `kneighbors(k >
  n_train)` raises `FerroError::InvalidParameter`, not sklearn `ValueError(
  "Expected n_neighbors <= n_samples_fit, ...")` (`_base.py:828-832`); `k==0` →
  `InvalidParameter` vs `ValueError("Expected n_neighbors > 0. Got 0")`
  (`:808`); and `fn fit` rejects `n_samples < n_neighbors` at fit time
  (`InsufficientSamples`), but sklearn defers to query time. Match the message +
  remove the fit-time guard (or align it with sklearn's deferral).
- #866 — REQ-5 (`kneighbors(X=None)` self-exclusion): no `X=None` path; sklearn
  queries the training data and excludes each point's self-match
  (`_base.py:815`, `:931-939`). Add the self-exclusion query surface (the
  `graph.rs` `drop_self_neighbors` logic is the reusable kernel). Pin: `kneighbors()`
  row `i` excludes index `i`.
- #867 — REQ-6 (`radius_neighbors` `sort_results` default): `radius_neighbors`
  always sorts ascending; sklearn's default is `sort_results=False` (native
  order). Add a `sort_results` toggle defaulting to native order. Pin: default →
  `[[1,2,3,4]]`, `sort_results=True` → `[[3,1,2,4]]`.
- #868 — REQ-7 (constructor `radius` + metric params): no `radius` (sklearn 1.0,
  `_unsupervised.py:136`), `metric` (`:139`), `p` (`:140`), `metric_params`
  (`:141`), `n_jobs` (`:142`). Euclidean-only. Add the param surface + a `metric`
  abstraction.
- #869 — REQ-8 (`kneighbors_graph`/`radius_neighbors_graph` `X=None` on the
  estimator): expose sklearn's `KNeighborsMixin.kneighbors_graph(X=None,
  mode=...)` method on `NearestNeighbors` with `include_self`/zero-diagonal
  semantics (graph.rs has the standalone free functions; see
  `.design/neighbors/graph.md`).
- #870 — REQ-9 (PyO3 binding + meta-crate re-export): no `RsNearestNeighbors` in
  `ferrolearn-python` and no meta-crate re-export; expose the estimator +
  `kneighbors`/`radius_neighbors` + `n_samples_fit_` at sklearn parity once
  #865-#868 land.
- #871 — REQ-10 (ferray substrate): migrate `nearest_neighbors.rs` off
  `ndarray`/`num-traits` to `ferray-core` (R-SUBSTRATE).
