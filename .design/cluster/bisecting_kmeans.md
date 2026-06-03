# Bisecting K-Means (sklearn.cluster.BisectingKMeans)

<!--
tier: 3-component
status: draft
baseline-commit: 4c253481
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_bisect_k_means.py     # class _BisectingTree (:24-74); class BisectingKMeans(_BaseKMeans) (:77-531); _parameter_constraints (:208-215); __init__ (:217-243); _inertia_per_cluster (:254-285); _bisect (:287-351); fit (:353-444); predict (:446-476); _predict_recursive (:478-527)
ferrolearn-module: ferrolearn-cluster/src/bisecting_kmeans.rs
parity-ops: BisectingKMeans (.__init__, .fit, .predict, .fit_predict, .transform, .cluster_centers_, .labels_, .inertia_)
crosslink-issue: 1023
-->

## Summary

`ferrolearn-cluster/src/bisecting_kmeans.rs` mirrors scikit-learn's `BisectingKMeans`
(`sklearn/cluster/_bisect_k_means.py`, `class BisectingKMeans(_BaseKMeans)` `:77-531`)
— divisive hierarchical clustering that recursively bisects the selected cluster with
inner 2-means until `n_clusters` leaves are formed. It exposes the unfitted
`BisectingKMeans<F>` (`n_clusters`, `max_iter`, `n_init`, `random_state`,
`bisecting_strategy`), the fitted `FittedBisectingKMeans<F>` (stores
`cluster_centers_`, `labels_`, `inertia_`), the `BisectingStrategy` enum
(`LargestCluster` / `LargestSSE`), a `Predict` impl (nearest of the FINAL leaf
centers), a `Transform` impl (euclidean distance to leaf centers), and a `fit_predict`
convenience mirroring `ClusterMixin.fit_predict`. It is re-exported at the crate root
(`pub use bisecting_kmeans::{BisectingKMeans, BisectingStrategy, FittedBisectingKMeans}`
in `ferrolearn-cluster/src/lib.rs`).

**Under honest assessment (R-HONEST-3), FOUR REQs SHIP through the crate re-export:
the clustering PARTITION (label co-membership up to a permutation) on well-separated
data (REQ-1), the `transform` distance-to-centers CONTRACT (REQ-13), and the
`bisecting_strategy` (REQ-3, fixed iter 126 / #1025) and `n_init` (REQ-4, #1026)
constructor defaults now matching sklearn.** Everything tied to exact VALUES (centers,
inertia, absolute label integers, predict semantics) DIVERGES, for a stack of
compounding reasons that this doc carves out explicitly:

1. **`bisecting_strategy` DEFAULT diverges (R-DEV-2).** sklearn default is
   `bisecting_strategy="biggest_inertia"` (`_bisect_k_means.py:229`) — i.e. split the
   max-SSE leaf. ferrolearn `fn new` defaults to `BisectingStrategy::LargestCluster`
   (`fn new`), which is sklearn's NON-default `"largest_cluster"`. sklearn's
   `"biggest_inertia"` is exactly ferrolearn's `LargestSSE`, so the correct default is
   `LargestSSE`. **This is the single cleanest minimal-fix candidate** (see Blockers).
2. **`n_init` DEFAULT diverges (R-DEV-2).** sklearn `n_init=1` (`:222`); ferrolearn
   `fn new` `n_init=10`. Different default restart count per bisection.
3. **`init` param + k-means++ missing (R-DEV-1/2).** sklearn `init` ∈
   `{"random","k-means++"}` or callable, default `"random"` (`:221`,
   `_init_centroids` call `:313-320`). ferrolearn has NO `init` param — `fn run_2means`
   picks two distinct random sample rows as the two seeds. No `"k-means++"`, no
   `init` constructor surface.
4. **`predict`: tree-descent vs flat nearest-center (R-DEV-1/3).** sklearn `predict`
   → `_predict_recursive` descends the `_BisectingTree`, at each internal node
   assigning to the nearer of the two CHILD centers and recursing (`:478-527`).
   ferrolearn `Predict::predict` assigns to the nearest of the FINAL leaf centers
   (flat, `fn predict`). The two rules coincide on well-separated data but differ in
   general.
5. **`labels_` numbering (tree DFS vs Vec order) (R-DEV-3).** sklearn numbers leaves
   by `iter_leaves()` depth-first left→right order (`:426-428`, `_BisectingTree.iter_leaves`
   `:68-74`). ferrolearn numbers by the order clusters land in its flat
   `Vec<ClusterInfo>` (remove target, push child0, push child1, `fn fit` split loop) —
   a different absolute numbering. The partition matches up-to-permutation; the
   integers do not.
6. **No mean-subtraction for numerical accuracy (R-DEV-1).** sklearn subtracts
   `X_mean` before clustering and adds it back to `cluster_centers_` at the end
   (`:402-404,433-435`). ferrolearn clusters on raw coordinates — affects the float
   precision of centers/inertia.
7. **Inner k-means + `tol` convergence (R-DEV-1).** sklearn's inner solver is
   `_kmeans_single_lloyd`/`_kmeans_single_elkan` with tol-based convergence on the
   center-shift Frobenius norm (`tol` default `1e-4`, `:226,322-330`). ferrolearn
   `fn run_2means` is a basic Lloyd converging on NO-LABEL-CHANGE — no `tol` parameter,
   no `tol` usage.
8. **Best-of-`n_init` tie tolerance (R-DEV-1).** sklearn keeps a run when
   `inertia < best_inertia * (1 - 1e-6)` (`:334`); ferrolearn keeps when `sse < best_sse`
   strict (`fn fit` best loop).
9. **Degenerate-cluster guard (R-DEV-1).** ferrolearn filters candidate clusters to
   those with `>= 2` points and `break`s if none remain (`fn fit` strategy match).
   sklearn always performs exactly `n_clusters - 1` bisections, picking the max-score
   leaf each time with no such guard (`:415-420`).
10. **`random_state` RNG.** sklearn threads numpy's RNG (`check_random_state` →
    `_init_centroids`) per bisection (`:391,313-320`); ferrolearn derives a per-split
    `StdRng` seed (`fn fit` seed derivation). Different RNG streams → exact init /
    labels / centers cannot match sklearn without numpy-RNG parity. This BLOCKS exact
    VALUE parity even after the default fixes.
11. **`sample_weight` missing (R-DEV-1).** sklearn supports it (`fit(..., sample_weight)`
    `:354`, threaded into `_bisect` `:306` and inertia `:438-440`); ferrolearn does not.
12. **`algorithm`/`verbose`/`copy_x` missing (R-DEV-2).** sklearn has `algorithm`
    (`"lloyd"`/`"elkan"`), `verbose`, `copy_x` (`:225,227,228`); ferrolearn has none.
13. **`n_clusters` DEFAULT=8 (R-DEV-2).** sklearn `n_clusters=8` (`:219`); ferrolearn
    `fn new(n_clusters)` requires the argument (a Rust builder, no default).
14. **Error ABI.** sklearn validates params via `_parameter_constraints`
    (`InvalidParameterError`) and `n_samples >= n_clusters` (`ValueError` from
    `_check_params_vs_input`); ferrolearn returns `FerroError::InvalidParameter` /
    `FerroError::InsufficientSamples` — different type/message ABI (R-DEV-2).

There is **NO `ferrolearn-python` binding**: `grep -rn "BisectingKMeans\|RsBisect"
ferrolearn-python/` is EMPTY, so `import ferrolearn` cannot reach `BisectingKMeans`.
`BisectingKMeans` / `BisectingStrategy` / `FittedBisectingKMeans` are existing pub APIs
(grandfathered S5/R-DEFER-1); their only non-test consumer is the crate re-export
(`pub use bisecting_kmeans::{...}` in `ferrolearn-cluster/src/lib.rs`).

## Live oracle probes (sklearn 1.5.2, run from /tmp)

Expected values are from the installed sklearn 1.5.2 oracle, never literal-copied from
ferrolearn (R-CHAR-3). Fixtures: `docs` = the upstream docstring fixture
`[[1,1],[10,1],[3,1],[10,0],[2,1],[10,2],[10,8],[10,9],[10,10]]`
(`_bisect_k_means.py:194-196`); `blobs` = the `make_blobs()` test fixture in
`bisecting_kmeans.rs` (3 points near `(0,0)`, 3 near `(10,10)`, 3 near `(0,10)`, 9×2).

### Probe 1 — docstring fixture: labels_ / predict / cluster_centers_ / inertia_
```
python3 -c "import numpy as np; from sklearn.cluster import BisectingKMeans; \
X=np.array([[1,1],[10,1],[3,1],[10,0],[2,1],[10,2],[10,8],[10,9],[10,10]]); \
m=BisectingKMeans(n_clusters=3, random_state=0).fit(X); \
print(m.labels_.tolist(), m.predict([[0,0],[12,3]]).tolist(), m.cluster_centers_.tolist(), m.inertia_)"
# labels  [0, 2, 0, 2, 0, 2, 1, 1, 1]
# predict [0, 2]
# centers [[2.0, 1.0], [10.0, 9.0], [10.0, 1.0]]
# inertia 6.0
```
**Findings:** sklearn's `labels_` integers follow the `iter_leaves()` DFS tree order
(leaf 0 = `(2,1)` group, leaf 1 = `(10,9)` group, leaf 2 = `(10,1)` group). `predict`
descends the tree (REQ-7). ferrolearn produces the same 3-way PARTITION but with a
DIFFERENT absolute numbering (its flat `Vec` order, REQ-5), DIFFERENT center VALUES
(no mean-subtraction, different inner k-means + RNG, REQ-2/REQ-6), and `inertia_` need
not equal `6.0`. This is why the in-tree tests assert label co-membership
(`labels[0]==labels[1]`) and shapes, never the integer values or center coordinates.

### Probe 2 — well-separated 3-blob PARTITION (the one agreement)
```
python3 -c "import numpy as np; from sklearn.cluster import BisectingKMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=BisectingKMeans(n_clusters=3, random_state=42).fit(X); \
print(m.labels_.tolist(), np.round(m.cluster_centers_,4).tolist())"
# labels  [2, 2, 2, 0, 0, 0, 1, 1, 1]
# centers [[10.0, 10.0667], [0.0, 10.0], [0.0, 0.0667]]
```
sklearn groups samples `{0,1,2}` / `{3,4,5}` / `{6,7,8}`. ferrolearn
`BisectingKMeans::<f64>::new(3).with_random_state(42).fit(blobs)` produces the SAME
3-way partition (co-membership) — verified by the in-tree `test_well_separated_blobs`
asserting `labels[0]==labels[1]==labels[2]`, `labels[3]==labels[4]==labels[5]`,
`labels[6]==labels[7]==labels[8]`, and all three blobs distinct — but with permuted
label integers and (in general) different center VALUES. Partition co-membership
agrees; `labels_` integers / `cluster_centers_` values / `inertia_` do NOT.

### Probe 3 — default constructor surface (the default divergences)
```
python3 -c "from sklearn.cluster import BisectingKMeans; d=BisectingKMeans(); \
print(d.bisecting_strategy, d.n_init, d.init, d.tol, d.algorithm, d.n_clusters, d.max_iter)"
# biggest_inertia 1 random 0.0001 lloyd 8 300
```
sklearn defaults: `bisecting_strategy="biggest_inertia"` (≡ ferrolearn `LargestSSE`),
`n_init=1`, `init="random"`, `tol=1e-4`, `algorithm="lloyd"`, `n_clusters=8`,
`max_iter=300`. ferrolearn `fn new` defaults: `bisecting_strategy=LargestCluster`
(≡ sklearn `"largest_cluster"`, NON-default — REQ-3), `n_init=10` (REQ-4),
`max_iter=300` (matches), no `init`/`tol`/`algorithm`/`copy_x`/`verbose` and no
`n_clusters` default (Rust builder requires the argument — REQ-9).

### Probe 4 — _parameter_constraints (the param ABI)
sklearn `_parameter_constraints` (`_bisect_k_means.py:208-215`) inherits
`_BaseKMeans` (`n_clusters`, `max_iter`, `tol`, `verbose`, `random_state`) and adds
`init ∈ {"k-means++","random"}|callable`, `n_init ∈ [1,∞)`, `copy_x ∈ {bool}`,
`algorithm ∈ {"lloyd","elkan"}`, `bisecting_strategy ∈
{"biggest_inertia","largest_cluster"}`. Invalid values raise `InvalidParameterError`.
ferrolearn rejects `n_clusters==0` / `n_init==0` with
`FerroError::InvalidParameter` and `n_samples < n_clusters` /
`n_samples == 0` with `FerroError::InsufficientSamples` (`fn fit` validation) — a
different error type/message ABI, and no `init`/`algorithm`/`bisecting_strategy`
string validation (the enum is type-safe but not the sklearn surface).

### Probe 5 — transform (euclidean distance to centers)
```
python3 -c "import numpy as np; from sklearn.cluster import BisectingKMeans; \
X=np.array([[1,1],[10,1],[3,1],[10,0],[2,1],[10,2],[10,8],[10,9],[10,10]]); \
m=BisectingKMeans(n_clusters=3, random_state=0).fit(X); \
print(m.transform(X).shape, np.round(m.transform(X)[0],6).tolist())"
# (9, 3)  [1.0, 12.041595, 9.0]   (column j = ||x_i - cluster_centers_[j]||)
```
sklearn `transform` (inherited `_BaseKMeans.transform`) returns the euclidean distance
from each sample to each row of `cluster_centers_`, shape `(n_samples, n_clusters)`.
ferrolearn `Transform::transform` computes exactly the same quantity against its leaf
centers (`fn transform`: `sqrt(squared_euclidean(row, center))`), shape
`(n_samples, n_clusters)`. The CONTRACT (distance-to-centers, shape, column ordering
follows `cluster_centers_` ordering) matches; the VALUES inherit the same center-value
divergence as `cluster_centers_` (REQ-2: no mean-subtraction, RNG, inner algorithm).
On well-separated data where the partition and centers agree to tolerance, the
transform values agree; in general they track the center divergence.

### Probe 6 — non-test consumer
`grep -rn "BisectingKMeans\|RsBisect" ferrolearn-python/` is **EMPTY** — there is no
PyO3 binding, so `import ferrolearn` cannot reach `BisectingKMeans`. `grep -rn
"Bisecting" ferrolearn-cluster/src/` outside `bisecting_kmeans.rs` finds only the crate
re-export (`pub use bisecting_kmeans::{BisectingKMeans, BisectingStrategy,
FittedBisectingKMeans}`) and doc-comment references in `lib.rs`. The sole non-test
consumer of `fit` / `fit_predict` / `predict` / `transform` / the accessors is the
crate re-export.

## Requirements

- REQ-1: **clustering PARTITION up-to-permutation (R-DEV-1).** Mirror
  `BisectingKMeans(n_clusters=k, random_state=s).fit(X)` producing the same grouping of
  samples into leaves on well-separated data. ferrolearn's `fn fit` repeatedly bisects
  the selected cluster with `fn run_2means` and aggregates leaves, recovering the
  correct PARTITION on separable blobs (Probe 2) — but the `labels_` integers (REQ-5),
  `cluster_centers_` VALUES (REQ-2), and `inertia_` (REQ-2) diverge, so this is a
  partition-only claim, not a value-parity claim.
- REQ-2: **`cluster_centers_` / `inertia_` VALUE parity (R-DEV-1).** Mirror sklearn's
  exact center coordinates and total inertia. Diverges because ferrolearn (a) does not
  subtract `X_mean` (`_bisect_k_means.py:402-404,433-435`), (b) uses a basic
  no-label-change Lloyd rather than `_kmeans_single_lloyd`/`_elkan` with `tol`
  convergence (`:322-330`), (c) uses a different RNG (REQ-8), and (d) uses a strict
  `sse < best_sse` rather than `inertia < best*(1-1e-6)` (`:334`). Probe 1: sklearn
  centers `[[2,1],[10,9],[10,1]]`, inertia `6.0`.
- REQ-3: **`bisecting_strategy` DEFAULT = `"biggest_inertia"` (R-DEV-2).** sklearn
  default `bisecting_strategy="biggest_inertia"` (`_bisect_k_means.py:229`), which is
  ferrolearn's `LargestSSE`. ferrolearn `fn new` defaults to `LargestCluster`
  (sklearn's NON-default `"largest_cluster"`). The correct default is `LargestSSE`.
- REQ-4: **`n_init` DEFAULT = 1 (R-DEV-2).** sklearn `n_init=1`
  (`_bisect_k_means.py:222`); ferrolearn `fn new` `n_init=10`. Different default
  restart count.
- REQ-5: **`labels_` integer numbering = `iter_leaves()` DFS order (R-DEV-3).** sklearn
  numbers leaves depth-first left→right over the `_BisectingTree`
  (`_bisect_k_means.py:426-428`, `iter_leaves` `:68-74`); on `docs` =
  `[0,2,0,2,0,2,1,1,1]`. ferrolearn numbers by flat `Vec<ClusterInfo>` order (`fn fit`
  remove-target/push-child0/push-child1) — a different absolute numbering; partition
  matches up-to-permutation (REQ-1), integers do not.
- REQ-6: **mean-subtraction for numerical accuracy (R-DEV-1).** sklearn subtracts
  `X_mean` before clustering and adds it back to `cluster_centers_`
  (`_bisect_k_means.py:402-404,433-435`); ferrolearn clusters on raw coordinates
  (`fn fit`), affecting center/inertia float precision.
- REQ-7: **`predict` tree-descent (R-DEV-1/3).** sklearn `predict` →
  `_predict_recursive` descends the `_BisectingTree`, at each node assigning to the
  nearer of the two CHILD centers (`_bisect_k_means.py:446-527`). ferrolearn
  `Predict::predict` assigns to the nearest of the FINAL leaf centers (flat,
  `fn predict`). Coincides on well-separated data, diverges in general.
- REQ-8: **`random_state` numpy-RNG parity (R-DEV-1).** sklearn threads numpy's RNG via
  `check_random_state` → `_init_centroids` per bisection
  (`_bisect_k_means.py:391,313-320`); ferrolearn derives a per-split `StdRng` seed
  (`fn fit`). Different RNG → exact init/labels/centers cannot match sklearn without a
  numpy-RNG analog; this BLOCKS exact VALUE parity (REQ-2) even after the default
  fixes. Depends on a ferray `random` analog (R-SUBSTRATE-5).
- REQ-9: **constructor surface `init`/`tol`/`algorithm`/`copy_x`/`verbose`/`sample_weight`
  + `n_clusters` default 8 (R-DEV-2).** sklearn `__init__`
  (`_bisect_k_means.py:217-243`) = `n_clusters=8, init="random", n_init=1,
  random_state, max_iter=300, verbose=0, tol=1e-4, copy_x=True, algorithm="lloyd",
  bisecting_strategy="biggest_inertia"`; `fit(X, y, sample_weight)` (`:354`). ferrolearn
  `BisectingKMeans<F>` has `n_clusters/max_iter/n_init/random_state/bisecting_strategy`
  only (`fn new` + builders) — missing `init`, `tol`, `algorithm`, `copy_x`, `verbose`,
  `sample_weight`, and the k-means++ init path; `n_clusters` has no default.
- REQ-10: **error ABI `InvalidParameterError` / `ValueError` (R-DEV-2).** sklearn
  validates via `_parameter_constraints` (`InvalidParameterError`,
  `_bisect_k_means.py:208-215`) and `n_samples >= n_clusters` (`ValueError` from
  `_check_params_vs_input`). ferrolearn returns `FerroError::InvalidParameter` /
  `FerroError::InsufficientSamples` (`fn fit`) — bounds agree, error type/message ABI
  differs.
- REQ-11: **PyO3 binding (R-DEFER-1/3).** No `RsBisectingKMeans` in `ferrolearn-python`
  (`grep -rn "BisectingKMeans\|RsBisect" ferrolearn-python/` EMPTY) — `import
  ferrolearn` cannot reach `BisectingKMeans`.
- REQ-12: **ferray substrate (R-SUBSTRATE).** `bisecting_kmeans.rs` imports
  `ndarray::{Array1, Array2}`, `num_traits::Float`, and `rand::rngs::StdRng`, not
  `ferray-core` / `ferray::linalg` / `ferray::random` (R-SUBSTRATE-1/2).
- REQ-13: **`transform` distance-to-centers contract (R-DEV-3).** sklearn `transform`
  (inherited `_BaseKMeans`) returns euclidean distance from each sample to each
  `cluster_centers_` row, shape `(n_samples, n_clusters)`. ferrolearn
  `Transform::transform` (`fn transform`) computes the same quantity against its leaf
  centers in the same shape and column ordering (Probe 5). The CONTRACT (output
  dtype/shape, distance metric, column-to-center correspondence) matches; the VALUES
  track the `cluster_centers_` divergence (REQ-2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). Fixtures: `docs` =
`[[1,1],[10,1],[3,1],[10,0],[2,1],[10,2],[10,8],[10,9],[10,10]]`; `blobs` =
`make_blobs()` (9×2).

- AC-1 (REQ-1, partition agrees / integers diverge):
  `BisectingKMeans(n_clusters=3, random_state=42).fit(blobs).labels_` → sklearn
  `[2,2,2,0,0,0,1,1,1]`; ferrolearn recovers the same 3-way grouping (`{0,1,2}` /
  `{3,4,5}` / `{6,7,8}`) up to a permutation of the integers.
- AC-2 (REQ-2/REQ-6, diverges): `BisectingKMeans(n_clusters=3, random_state=0).fit(docs)`
  → sklearn `cluster_centers_ = [[2,1],[10,9],[10,1]]`, `inertia_ = 6.0`. ferrolearn
  produces different center VALUES (no mean-subtraction, different inner Lloyd + RNG)
  and need not match `6.0`.
- AC-3 (REQ-5, diverges): `BisectingKMeans(n_clusters=3, random_state=0).fit(docs).labels_`
  → sklearn `[0,2,0,2,0,2,1,1,1]` (DFS leaf order). ferrolearn's flat-`Vec` numbering
  permutes the integers.
- AC-4 (REQ-3, default): `BisectingKMeans().bisecting_strategy` → sklearn
  `"biggest_inertia"` (≡ `LargestSSE`); ferrolearn `fn new` defaults to
  `LargestCluster`.
- AC-5 (REQ-4, default): `BisectingKMeans().n_init` → sklearn `1`; ferrolearn `fn new`
  defaults to `10`.
- AC-6 (REQ-7, diverges in general): `BisectingKMeans(n_clusters=3, random_state=0)
  .fit(docs).predict([[0,0],[12,3]])` → sklearn `[0,2]` via tree-descent. ferrolearn
  uses flat nearest-leaf-center — same on this fixture, divergent where tree-descent
  and flat-nearest disagree.
- AC-7 (REQ-13, contract matches): `BisectingKMeans(n_clusters=3, random_state=0)
  .fit(docs).transform(docs).shape` → sklearn `(9, 3)`, column `j` =
  `||x_i - cluster_centers_[j]||` (row 0 = `[1.0, 12.041595, 9.0]`). ferrolearn
  `Transform::transform` returns the same shape and per-center euclidean distance; the
  values track the `cluster_centers_` divergence (REQ-2).
- AC-8 (REQ-9, surface): `BisectingKMeans().get_params()` exposes
  `init/tol/algorithm/copy_x/verbose/n_clusters(=8)`; ferrolearn `BisectingKMeans<F>`
  has none of `init`/`tol`/`algorithm`/`copy_x`/`verbose` and no `n_clusters` default,
  and `import ferrolearn; ferrolearn.BisectingKMeans` does not exist (no binding,
  REQ-11).

## REQ status table

Binary (R-DEFER-2). `BisectingKMeans` / `BisectingStrategy` / `FittedBisectingKMeans`
are existing pub APIs re-exported at the crate root (the only non-test consumer;
grandfathered S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2, run from
`/tmp`. Honest assessment (R-HONEST-3): **two REQs SHIP** — the clustering PARTITION
up-to-permutation on well-separated data (REQ-1) and the `transform`
distance-to-centers CONTRACT (REQ-13, output shape/metric/ordering) — both through the
crate re-export. `cluster_centers_` / `inertia_` VALUES, `labels_` integers, the
constructor defaults/surface, tree-descent `predict`, and exact RNG parity all
DIVERGE. Blocker numbers below are the real filed issues (#1024–#1034).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (clustering PARTITION up-to-permutation) | SHIPPED | impl `fn fit` (bisect-selected-cluster loop + `fn run_2means` inner 2-means + leaf aggregation) recovers sklearn's grouping for `n_clusters` leaves on well-separated data (Probe 2). Consumer: crate re-export `pub use bisecting_kmeans::{BisectingKMeans, BisectingStrategy, FittedBisectingKMeans}` (`lib.rs`). Guards: in-tree `test_well_separated_blobs`, `test_two_clusters`, `test_largest_sse_strategy` assert co-membership of separable blobs. Verification: `cargo test -p ferrolearn-cluster --lib bisecting` → 20 passed; 0 failed. Underclaim: PARTITION up-to-permutation only — `labels_` INTEGERS (REQ-5), `cluster_centers_`/`inertia_` VALUES (REQ-2), tree-descent `predict` (REQ-7) diverge. |
| REQ-13 (`transform` distance-to-centers CONTRACT) | SHIPPED | impl `Transform::transform` (`fn transform`) returns `sqrt(squared_euclidean(row, center))` for each `(sample, leaf-center)` pair, shape `(n_samples, n_clusters)`, column `j` = distance to `cluster_centers_[j]` — matching sklearn `_BaseKMeans.transform` shape/metric/ordering (Probe 5: `(9,3)`). Consumer: crate re-export (`lib.rs`). Guard: shape asserted via the same fixtures as `cluster_centers_` (`test_cluster_centers_shape`). Underclaim: CONTRACT (shape/metric/column ordering) only — the distance VALUES track the `cluster_centers_` divergence (REQ-2); they agree on well-separated data, diverge where centers diverge. |
| REQ-2 (`cluster_centers_`/`inertia_` value parity) | NOT-STARTED | open prereq blocker #1024. sklearn `docs` → `cluster_centers_ = [[2,1],[10,9],[10,1]]`, `inertia_ = 6.0` (Probe 1/AC-2). ferrolearn diverges: no `X_mean` subtraction (`_bisect_k_means.py:402-404,433-435`, REQ-6), basic no-label-change Lloyd vs `_kmeans_single_lloyd`/`_elkan`+`tol` (`:322-330`, REQ-9), different RNG (REQ-8), strict `sse < best_sse` vs `inertia < best*(1-1e-6)` (`:334`). Gated on REQ-6/REQ-8/REQ-9. |
| REQ-3 (`bisecting_strategy` default = `biggest_inertia`) | SHIPPED | impl `fn new` defaults `bisecting_strategy: BisectingStrategy::LargestSSE` (= sklearn `"biggest_inertia"`, `_bisect_k_means.py:229`). Guard: in-module `test_default_bisecting_strategy_is_largest_sse`. Fixed #1025. |
| REQ-4 (`n_init` default = 1) | SHIPPED | impl `fn new` defaults `n_init: 1` (= sklearn `n_init=1`, `_bisect_k_means.py:222`). Guard: in-module `test_default_n_init_is_one`. Fixed #1026. |
| REQ-5 (`labels_` DFS leaf numbering) | NOT-STARTED | open prereq blocker #1027. sklearn numbers leaves by `iter_leaves()` DFS left→right (`_bisect_k_means.py:426-428`, `:68-74`); `docs` → `[0,2,0,2,0,2,1,1,1]` (AC-3). ferrolearn numbers by flat `Vec<ClusterInfo>` order (`fn fit` remove-target/push-children) — permuted integers (partition matches, REQ-1). Requires a tree structure mirroring `_BisectingTree` to fix the absolute numbering. |
| REQ-6 (mean-subtraction `X_mean`) | NOT-STARTED | open prereq blocker #1028. sklearn subtracts `X_mean` before clustering, adds it back to `cluster_centers_` (`_bisect_k_means.py:402-404,433-435`); ferrolearn `fn fit` clusters on raw coordinates — affects center/inertia float precision (feeds REQ-2). |
| REQ-7 (`predict` tree-descent) | NOT-STARTED | open prereq blocker #1029. sklearn `predict` → `_predict_recursive` descends the `_BisectingTree`, choosing the nearer CHILD center at each node (`_bisect_k_means.py:446-527`); `docs.predict([[0,0],[12,3]])` → `[0,2]` (AC-6). ferrolearn `Predict::predict` (`fn predict`) assigns to the nearest FINAL leaf center (flat) — coincides on well-separated data, diverges in general. Requires the `_BisectingTree` structure (shared with REQ-5). |
| REQ-8 (`random_state` numpy-RNG parity) | NOT-STARTED | open prereq blocker #1030. sklearn threads numpy RNG via `check_random_state` → `_init_centroids` per bisection (`_bisect_k_means.py:391,313-320`); ferrolearn derives a per-split `StdRng` seed (`fn fit`). Different RNG → exact init/labels/centers cannot match without a numpy-RNG analog. Depends on a ferray `random` analog (R-SUBSTRATE-5); blocks exact REQ-2 value parity. |
| REQ-9 (ctor surface `init`/`tol`/`algorithm`/`copy_x`/`verbose`/`sample_weight` + `n_clusters`=8) | NOT-STARTED | open prereq blocker #1031. sklearn `__init__` (`_bisect_k_means.py:217-243`) + `fit(...sample_weight)` (`:354`) expose `init` (incl. k-means++), `tol`, `algorithm`, `copy_x`, `verbose`, `sample_weight`, `n_clusters=8` default. ferrolearn `BisectingKMeans<F>` (`fn new` + builders) has none of these (AC-8); `fn run_2means` has no k-means++ and no `tol` convergence. |
| REQ-10 (error ABI `InvalidParameterError`/`ValueError`) | NOT-STARTED | open prereq blocker #1032. sklearn validates via `_parameter_constraints` (`InvalidParameterError`, `_bisect_k_means.py:208-215`) + `n_samples>=n_clusters` (`ValueError`). ferrolearn `fn fit` returns `FerroError::InvalidParameter` / `FerroError::InsufficientSamples` — bounds agree, type/message ABI differs (surfaces at the binding, REQ-11). |
| REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #1033. `grep -rn "BisectingKMeans\|RsBisect" ferrolearn-python/` is EMPTY — no `RsBisectingKMeans`, so `import ferrolearn` cannot reach `BisectingKMeans` (Probe 6). The only non-test consumer of `fit`/`predict`/`fit_predict`/`transform`/accessors is the crate re-export (`lib.rs`). |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1034. `bisecting_kmeans.rs` imports `ndarray::{Array1, Array2}`, `num_traits::Float`, `rand::rngs::StdRng`; not migrated to `ferray-core` / `ferray::linalg` / `ferray::random` (R-SUBSTRATE-1/2). The RNG migration is entangled with REQ-8. |

## Architecture

`bisecting_kmeans.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`BisectingKMeans<F>` (`n_clusters: usize`, `max_iter: usize`, `n_init: usize`,
`random_state: Option<u64>`, `bisecting_strategy: BisectingStrategy`) →
`Fit<Array2<F>, ()>` → `FittedBisectingKMeans<F>` (private `cluster_centers_:
Array2<F>`, `labels_: Array1<isize>`, `inertia_: F`). The `BisectingStrategy` enum
(`LargestCluster` / `LargestSSE`) mirrors sklearn's `bisecting_strategy ∈
{"largest_cluster","biggest_inertia"}` (`_bisect_k_means.py:145-158,214`) — with the
DEFAULT inverted (REQ-3). Generic over `F: Float + Send + Sync + 'static`; every public
method returns `Result<_, FerroError>` (R-CODE-2). `FittedBisectingKMeans` implements
`Predict<Array2<F>>` (flat nearest-leaf-center) and `Transform<Array2<F>>` (euclidean
distance to leaf centers); a `fit_predict` convenience mirrors `ClusterMixin.fit_predict`.

**Fit path (`fn fit`).** Validates `n_clusters >= 1`, `n_init >= 1`, `n_samples >= 1`,
`n_samples >= n_clusters` (`FerroError`, REQ-10). Initializes a single
`ClusterInfo { indices, center, sse }` over all samples (`fn compute_cluster_stats`),
then loops `while clusters.len() < n_clusters`:
1. **Select** the cluster to bisect — `LargestCluster` → max `indices.len()`,
   `LargestSSE` → max `sse` — but FILTERED to clusters with `>= 2` points, breaking if
   none (`fn fit` strategy match). sklearn instead always runs exactly
   `n_clusters - 1` bisections, picking the max-score leaf via
   `_BisectingTree.get_cluster_to_bisect` with no `>= 2` guard
   (`_bisect_k_means.py:415-420,52-66`) — REQ-1 caveat (degenerate-cluster divergence).
2. **Bisect** — run `fn run_2means` `n_init` times (each seeding two distinct random
   sample rows via a per-split `StdRng`, Lloyd until NO-LABEL-CHANGE or `max_iter`),
   keeping the run with strict `sse < best_sse` (`fn fit` best loop). sklearn's inner
   solver is `_kmeans_single_lloyd`/`_elkan` with `_init_centroids` (incl. k-means++)
   and `tol`-based convergence, keeping `inertia < best*(1-1e-6)`
   (`_bisect_k_means.py:312-337`) — REQ-2/REQ-4/REQ-8/REQ-9.
3. **Replace** the target `ClusterInfo` with its two children
   (`clusters.remove(target_idx)`, push child0, push child1) — REQ-5 numbering origin
   (flat `Vec` order vs sklearn's `_BisectingTree.split` + DFS `iter_leaves`,
   `:40-47,68-74`).
Finally aggregates `cluster_centers_` / `labels_` / `inertia_` by iterating the flat
`Vec<ClusterInfo>` in order. sklearn aggregates by DFS `iter_leaves()` and adds back
`X_mean` (`:422-440`) — REQ-2/REQ-5/REQ-6.

**Predict path (`fn predict`).** Flat nearest-leaf-center: for each sample, argmin over
`squared_euclidean(row, cluster_centers_[c])`. sklearn descends the `_BisectingTree`
choosing the nearer child center at each node (`_predict_recursive`,
`_bisect_k_means.py:478-527`) — REQ-7. The two agree on well-separated data.

**Transform path (`fn transform`).** `sqrt(squared_euclidean(row, center))` for each
`(sample, leaf-center)` → `(n_samples, n_clusters)`, mirroring `_BaseKMeans.transform`
(REQ-13 — contract matches, values track REQ-2).

**Invariants held vs sklearn:** clustering PARTITION (co-membership on separable data,
Probe 2 / AC-1); `cluster_centers_.shape == (n_clusters, n_features)`;
`labels_.len() == n_samples`; labels in `[0, n_clusters)`; `inertia_ >= 0`;
`transform` shape `(n_samples, n_clusters)` + euclidean metric + column-to-center
ordering (REQ-13); predict/transform shape-mismatch error; deterministic for a fixed
`random_state` (`test_reproducibility`); f32 + f64 support.

**Invariants NOT held vs sklearn:** `cluster_centers_`/`inertia_` VALUES (REQ-2);
`bisecting_strategy` default (REQ-3); `n_init` default (REQ-4); `labels_` DFS
numbering (REQ-5); `X_mean` subtraction (REQ-6); tree-descent `predict` (REQ-7);
numpy-RNG init parity (REQ-8); the constructor surface incl. k-means++ / `tol` / `sample_weight`
(REQ-9); the error ABI (REQ-10); the PyO3 binding (REQ-11); the ferray substrate
(REQ-12).

**Consumer wiring.** The only non-test consumer is the crate re-export
(`pub use bisecting_kmeans::{BisectingKMeans, BisectingStrategy, FittedBisectingKMeans}`,
`ferrolearn-cluster/src/lib.rs`). There is no `ferrolearn-python` binding (Probe 6) and
no other in-crate consumer.

## Verification

Library crate (green at baseline `4c253481` for the existing variant behavior):
```
cargo test -p ferrolearn-cluster --lib bisecting     # 20 passed; 0 failed
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The 20 in-tree `#[test]`s (`test_well_separated_blobs`, `test_two_clusters`,
`test_single_cluster`, `test_predict_assigns_correctly`, `test_predict_new_data`,
`test_predict_shape_mismatch`, `test_inertia_non_negative`,
`test_cluster_centers_shape`, `test_largest_sse_strategy`, `test_reproducibility`,
`test_zero_clusters`, `test_k_greater_than_n_samples`, `test_empty_data`,
`test_invalid_n_init`, `test_single_sample`, `test_f32_support`,
`test_identical_points`, `test_labels_in_range`, `test_n_init_picks_best`,
`test_k_equals_n_samples`) pin ferrolearn's current behavior — label co-membership,
shapes, ranges, error edges, reproducibility, f32 support, best-of-`n_init`
monotonicity. **None compares `cluster_centers_` values, `labels_` integers,
`inertia_`, the constructor defaults, or tree-descent `predict` against the live
sklearn `BisectingKMeans` oracle**, so they stay green despite the divergences. In
particular the partition tests assert `labels[0]==labels[1]` co-membership and never
the absolute integers, masking REQ-5.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin REQ-3 (bisecting_strategy default) and
REQ-4 (n_init default) FIRST** — they are isolable, observable-via-`get_params`, and
each is a one-line minimal fix; REQ-2 (center values) is gated on REQ-6/REQ-8/REQ-9 and
cannot pin to exact values until the numpy-RNG analog lands:
```
# REQ-3 / REQ-4 (defaults — observable, isolable)
python3 -c "from sklearn.cluster import BisectingKMeans; d=BisectingKMeans(); \
print(d.bisecting_strategy, d.n_init)"        # biggest_inertia 1  (ferrolearn: LargestCluster, 10)
# REQ-1 (partition) + REQ-5 (label integers) + REQ-2 (centers/inertia)
python3 -c "import numpy as np; from sklearn.cluster import BisectingKMeans; \
X=np.array([[1,1],[10,1],[3,1],[10,0],[2,1],[10,2],[10,8],[10,9],[10,10]]); \
m=BisectingKMeans(n_clusters=3, random_state=0).fit(X); \
print(m.labels_.tolist(), m.cluster_centers_.tolist(), m.inertia_)"
# [0,2,0,2,0,2,1,1,1] [[2.0,1.0],[10.0,9.0],[10.0,1.0]] 6.0
# REQ-7 (tree-descent predict)
python3 -c "import numpy as np; from sklearn.cluster import BisectingKMeans; \
X=np.array([[1,1],[10,1],[3,1],[10,0],[2,1],[10,2],[10,8],[10,9],[10,10]]); \
m=BisectingKMeans(n_clusters=3, random_state=0).fit(X); print(m.predict([[0,0],[12,3]]).tolist())"
# [0, 2]
# REQ-13 (transform contract)
python3 -c "import numpy as np; from sklearn.cluster import BisectingKMeans; \
X=np.array([[1,1],[10,1],[3,1],[10,0],[2,1],[10,2],[10,8],[10,9],[10,10]]); \
m=BisectingKMeans(n_clusters=3, random_state=0).fit(X); \
print(m.transform(X).shape, np.round(m.transform(X)[0],6).tolist())"
# (9, 3) [1.0, 12.041595, 9.0]
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-cluster/tests/divergence_bisecting_kmeans.rs`, asserting the live-sklearn
expected values above and FAILING against current `bisecting_kmeans.rs`. REQ-3 and
REQ-4 pin against constructor defaults (deterministic, no RNG entanglement); REQ-2
(exact center/inertia values) can only be characterized as a partition/tolerance pin
until the numpy-RNG analog (REQ-8) and mean-subtraction (REQ-6) land — pin REQ-3/REQ-4
as the green-after-single-fix targets.

ferrolearn-python (REQ-11 binding parity, after the binding lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_bisecting_kmeans.py -q
```
asserting `ferrolearn.BisectingKMeans` exists and exposes `cluster_centers_` /
`labels_` / `inertia_` / `predict` / `transform` / the sklearn parameter surface,
matching `sklearn.cluster.BisectingKMeans` on the AC fixtures.

## Blockers (to open)

REQ-1 (partition) and REQ-13 (transform contract) SHIP — no blocker. The remaining
NOT-STARTED REQs are the real filed issues #1024–#1034:

- **#1025** — REQ-3 (**cleanest minimal fix**): set `fn new` default
  `bisecting_strategy = BisectingStrategy::LargestSSE` to mirror sklearn
  `"biggest_inertia"` (`_bisect_k_means.py:229`). One-line change, pins green against
  the `get_params`-style default probe with no RNG entanglement.
- **#1026** — REQ-4 (clean minimal fix): set `fn new` default `n_init = 1` to mirror
  sklearn (`_bisect_k_means.py:222`). One-line change, pins green against the default
  probe.
- **#1024** — REQ-2: `cluster_centers_`/`inertia_` value parity — gated on REQ-6
  (mean-subtraction), REQ-8 (numpy-RNG), REQ-9 (inner `_kmeans_single_lloyd`+`tol`).
- **#1027** — REQ-5: `labels_` DFS leaf numbering — introduce a `_BisectingTree` mirror
  so leaves number in `iter_leaves()` DFS order (`_bisect_k_means.py:426-428,68-74`).
- **#1028** — REQ-6: subtract `X_mean` before clustering, add back to centers
  (`_bisect_k_means.py:402-404,433-435`).
- **#1029** — REQ-7: tree-descent `predict` via `_predict_recursive`
  (`_bisect_k_means.py:446-527`) — shares the `_BisectingTree` with REQ-5.
- **#1030** — REQ-8: numpy-RNG parity for `_init_centroids` per bisection
  (`_bisect_k_means.py:313-320,391`) — depends on a ferray `random` analog
  (R-SUBSTRATE-5); blocks exact REQ-2.
- **#1031** — REQ-9: add `init` (incl. k-means++), `tol`, `algorithm`, `copy_x`,
  `verbose`, `sample_weight` ctor/fit surface + `n_clusters=8` default + the
  `inertia < best*(1-1e-6)` tie tolerance (`_bisect_k_means.py:217-243,334,354`).
- **#1032** — REQ-10: `InvalidParameterError` / `ValueError` error ABI
  (`_bisect_k_means.py:208-215`).
- **#1033** — REQ-11: add `RsBisectingKMeans` to `ferrolearn-python` (fit / predict /
  fit_predict / transform / cluster_centers_ / labels_ / inertia_ + parameter surface).
- **#1034** — REQ-12: migrate `bisecting_kmeans.rs` off `ndarray`/`num-traits`/`rand` to
  `ferray-core` / `ferray::linalg` / `ferray::random` (R-SUBSTRATE) — RNG migration
  entangled with REQ-8.
