# AgglomerativeClustering (sklearn.cluster.AgglomerativeClustering)

<!--
tier: 3-component
status: draft
baseline-commit: 75ac7a41
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_agglomerative.py     # class AgglomerativeClustering(ClusterMixin, BaseEstimator) (:781); _parameter_constraints (:934-947); __init__ defaults n_clusters=2/metric='euclidean'/linkage='ward'/distance_threshold=None/compute_full_tree='auto'/compute_distances=False (:949-968); fit -> _validate_data(ensure_min_samples=2) (:989); _fit n_clusters^distance_threshold XOR validation (:1022-1027); ward requires euclidean (:1034-1038); _TREE_BUILDERS dispatch (:720-725, :1040); children_/n_connected_components_/n_leaves_ (:1083-1085); distances_ + n_clusters_ from distance_threshold (:1087-1095); _hc_cut tree-cut + label numbering (:731-775); children_ doc (n_samples-1, 2) full dendrogram (:902-908)
parity-ops: AgglomerativeClustering (.__init__, .fit, .fit_predict, .labels_, .n_clusters_, .children_), ward_tree
crosslink-issue: 962
-->

## Summary

`ferrolearn-cluster/src/agglomerative.rs` mirrors scikit-learn's
`AgglomerativeClustering` (`sklearn/cluster/_agglomerative.py`,
`class AgglomerativeClustering(ClusterMixin, BaseEstimator)` `:781`) —
bottom-up hierarchical clustering that repeatedly merges the two closest
clusters under a linkage criterion until `n_clusters` remain. It exposes the
unfitted `AgglomerativeClustering<F>` (`n_clusters` required, `linkage` default
`Ward`, builder `with_linkage`), the `Linkage` enum (`Ward`/`Complete`/`Average`/
`Single`), the fitted `FittedAgglomerativeClustering<F>` (stores `labels_`,
`n_clusters_`, `children_`, plus accessors `labels()`/`n_clusters()`/`children()`),
a `Fit<Array2<F>, ()>` impl, and a `fit_predict` convenience mirroring
`ClusterMixin.fit_predict`. It is re-exported at the crate root
(`pub use agglomerative::{AgglomerativeClustering, FittedAgglomerativeClustering, Linkage}`
in `ferrolearn-cluster/src/lib.rs`), consumed in-crate by `birch.rs`
(`fn fit` runs `AgglomerativeClustering::new(k).with_linkage(Linkage::Ward)` on
subcluster centers) and `feature_agglomeration.rs` (`fn fit` delegates on `X.T`),
and bound into CPython as `ferrolearn.AgglomerativeClustering` via
`_RsAgglomerativeClustering` (`ferrolearn-python/src/extras.rs`,
`ferrolearn-python/python/ferrolearn/_extras.py`).

**The structural carve-out (#938) — RESOLVED for the unstructured case.** ferrolearn
now builds the dendrogram exactly as sklearn does for the default unstructured
(`connectivity=None`) path, producing bit-exact `children_` (REQ-6) and absolute
`labels_` (REQ-7) for all four linkages:

1. **Full `children_` with internal-node IDs.** `fn full_dendrogram` returns the
   FULL dendrogram, shape `(n_samples-1, 2)`, leaves `0..n-1` and internal-node IDs
   `n+i` for the `i`-th sorted merge (`_agglomerative.py:902-908`,
   `:314`/`:586`). For ward/complete/average it runs the **nearest-neighbour chain**
   (`fn nn_chain`, the Lance–Williams update `fn lance_williams` on **Euclidean**
   distances — NOTE scipy's linkage distances are actual Euclidean, the Ward update
   is the Wishart/Ward form, NOT squared), then a **stable distance sort** and a
   **union-find relabel** (`fn union_find_relabel`) — reproducing
   `scipy.cluster.hierarchy.linkage(X, method, 'euclidean')[:, :2]` bit-for-bit. For
   single it runs **Prim's MST** (`fn mst_single`, mirroring `mst_linkage_core`) +
   `fn single_linkage_relabel` (mirroring `_single_linkage_label`); for single,
   sklearn's pair-column order differs from `scipy.linkage('single')` (R-DEV-7), and
   ferrolearn matches sklearn's OWN `children_`.

2. **`_hc_cut` heap-pop enumeration.** `fn hc_cut` cuts the full tree exactly as
   `_hc_cut` (`_agglomerative.py:731-775`): a negated-id min-heap (`fn heappush_min`/
   `fn heappushpop_min`, a CPython-`heapq`-faithful translation incl. `_siftup`/
   `_siftdown` layout) pops the top-`n_clusters` nodes, then
   `for i, node in enumerate(nodes): label[descendents] = i` numbers each leaf by
   its ancestor's heap position. So on the 3-blob fixture (Probe B) ferrolearn now
   yields sklearn's exact `[2,2,2,1,1,1,0,0,0]`. The consumers `birch.rs` (REQ-3)
   and `feature_agglomeration.rs` (REQ-3/REQ-5) use `labels()` purely as a partition
   and are unaffected by the renumbering (their suites stay green).

The remaining NOT-STARTED surface is the parameters/attributes ferrolearn does
not expose: `metric`, `connectivity`, `memory`; the `compute_full_tree`
partial-tree PERF early-stop (its observable attributes already match the always-
full unstructured path, so only the unimplemented optimisation remains); the
`ensure_min_samples=2` validation; and the ferray substrate. SHIPPED since #938:
the absolute `_hc_cut` `labels_` numbering and the full `children_` dendrogram
(REQ-6/REQ-7); and, this iteration, the `distance_threshold`/`compute_distances`/
`distances_` cut (REQ-9) and the `n_leaves_`/`n_connected_components_` fitted
attributes (REQ-10) — all bit-exact (~1e-12) against the live sklearn/scipy 1.5.2
oracle for the unstructured (`connectivity=None`) case across all four linkages.

## Live oracle probes (sklearn 1.5.2, run from /tmp)

All expected values from the installed sklearn 1.5.2 oracle, never literal-copied
from ferrolearn (R-CHAR-3). ferrolearn `labels_` partitions confirmed against the
in-tree `#[test]`s (co-membership assertions) + the algorithm in `fn agglomerate`.

### Probe A — `labels_` partition, `n_clusters_`, `children_`, `n_leaves_` (2 blobs, ward)

```
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.05,10.05]]); \
m=AgglomerativeClustering(n_clusters=2,linkage='ward').fit(X); \
print(m.labels_.tolist(), m.n_clusters_, m.n_leaves_, m.n_connected_components_, m.children_.shape, m.children_.tolist())"
# labels_ [0,0,0,0,1,1,1,1]   n_clusters_ 2   n_leaves_ 8   n_connected_components_ 1
# children_ shape (7, 2)   children_ [[5,7],[0,3],[1,9],[4,8],[6,11],[2,10],[12,13]]
```
The PARTITION (first 4 together, last 4 together) matches ferrolearn (`test_ward_two_blobs`
asserts exactly this co-membership). But sklearn `children_` has **7 rows = n_samples-1**
with **internal-node IDs >= 8** (`9`, `10`, `11`, `12`, `13`); ferrolearn `children_`
has **6 rows = n_samples - n_clusters** of reused-slot pairs. `n_leaves_` (8) and
`n_connected_components_` (1) are sklearn attributes ferrolearn does not expose.

### Probe B — absolute `labels_` numbering across linkages (3 blobs)

```
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
[print(lk, AgglomerativeClustering(n_clusters=3,linkage=lk).fit(X).labels_.tolist()) for lk in ['ward','complete','average','single']]"
# ward     [2,2,2,1,1,1,0,0,0]
# complete [2,2,2,1,1,1,0,0,0]
# average  [2,2,2,1,1,1,0,0,0]
# single   [0,0,0,1,1,1,2,2,2]
```
The three blobs `{0,1,2}`/`{3,4,5}`/`{6,7,8}` are a clean PARTITION in all four
linkages, and ferrolearn matches that partition (`test_ward_three_blobs` etc. assert
co-membership). But the **absolute integers differ**: sklearn's `_hc_cut` heap
enumeration gives blob `{0,1,2}` the label `2` under ward/complete/average; ferrolearn's
ascending-surviving-slot relabel gives blob `{0,1,2}` the label `0`. Same partition,
permuted numbering — the #938 divergence.

### Probe C — `n_clusters=2` constructor default + docstring example

```
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
X=np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]]); \
print(AgglomerativeClustering().n_clusters, AgglomerativeClustering().fit(X).labels_.tolist())"
# 2   [1,1,1,0,0,0]
```
sklearn `AgglomerativeClustering()` defaults `n_clusters=2` (`__init__` `:951`).
ferrolearn `AgglomerativeClustering::new(n_clusters)` REQUIRES `n_clusters` — no
default (though the PyO3 `_RsAgglomerativeClustering::new` signature does default
`n_clusters=2`, `extras.rs:968`).

### Probe D — single-sample (`ensure_min_samples=2`)

```
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
AgglomerativeClustering(n_clusters=1).fit(np.array([[3.,4.]]))"
# ValueError: Found array with 1 sample(s) (shape=(1, 2)) while a minimum of 2 is required
```
sklearn `fit` calls `_validate_data(X, ensure_min_samples=2)` (`:989`) — **one sample
is rejected**. ferrolearn `fn fit` only rejects `n_samples == 0` and
`n_samples < n_clusters`; with `n_clusters=1` it ACCEPTS a single sample
(`test_single_sample_single_cluster` asserts `labels()[0] == 0`). Divergence.

### Probe E — `linkage='ward'` requires `metric='euclidean'`

```
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
AgglomerativeClustering(linkage='ward',metric='manhattan').fit(np.array([[0.,0.],[1.,1.],[5.,5.]]))"
# ValueError: manhattan was provided as metric. Ward can only work with euclidean distances.
```
sklearn `_fit` raises if `linkage == 'ward' and metric != 'euclidean'` (`:1034-1038`).
ferrolearn has no `metric` parameter at all (Euclidean-only), so this validation path
is unreachable. Missing surface.

### Probe F — `distance_threshold` (+ XOR validation), `compute_distances`/`distances_`

```
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
X=np.array([[0.,0.],[0.1,0.],[10.,10.],[10.1,10.]]); \
print(AgglomerativeClustering(n_clusters=None,distance_threshold=5.0).fit(X).labels_.tolist(), \
      AgglomerativeClustering(n_clusters=2,compute_distances=True).fit(X).distances_.tolist())"
# [0,0,1,1]   distances_ [0.0999..., 0.1, 20.0]
```
sklearn requires `exactly one of n_clusters and distance_threshold` (`_fit`,
`:1022-1027`), derives `n_clusters_ = count_nonzero(distances_ >= threshold) + 1`
(`:1090-1093`), and surfaces merge `distances_` when `compute_distances=True`
(`:1087-1088`). ferrolearn has neither parameter nor `distances_`. Missing surface.

### Probe G — non-test consumers

```
grep -rn "AgglomerativeClustering" ferrolearn-cluster/src/birch.rs ferrolearn-cluster/src/feature_agglomeration.rs \
  ferrolearn-python/src/extras.rs ferrolearn-python/python/ferrolearn/_extras.py
```
THREE real non-test production consumers:
- **in-crate `birch.rs`** — `fn fit` runs `AgglomerativeClustering::<F>::new(k_actual).with_linkage(Linkage::Ward)`
  on `subcluster_centers_` (the global-clustering step, `birch.rs:459`).
- **in-crate `feature_agglomeration.rs`** — `fn fit` runs
  `AgglomerativeClustering::<F>::new(self.n_clusters).with_linkage(map_linkage(...))`
  on `X.T` (`feature_agglomeration.rs:307`).
- **PyO3 binding** — `#[pyclass(name="_RsAgglomerativeClustering")]` (`extras.rs:959`):
  `fn new(n_clusters=2)`, `fn fit` calling
  `ferrolearn_cluster::AgglomerativeClustering::<f64>::new(self.n_clusters)`, `#[getter]
  labels_`; registered via `m.add_class::<extras::RsAgglomerativeClustering>()`
  (`src/lib.rs:67`); wrapped in `class AgglomerativeClustering(_ClusterWrapper)`
  (`python/ferrolearn/_extras.py:432`); exported in `__init__.py`. So
  `import ferrolearn; ferrolearn.AgglomerativeClustering(n_clusters=2).fit(X).labels_`
  is a real consumer. (The binding hard-wires Ward — no `linkage` argument is marshalled.)

## Requirements

- REQ-1: **`labels_` PARTITION up-to-permutation, on separable data (R-DEV-1).** Mirror
  the merge-until-`n_clusters` clustering of `AgglomerativeClustering.fit` (`_fit`,
  `:992-1106`): on well-separated blobs the set of co-membership groups produced by
  `fn agglomerate` matches sklearn's `labels_` partition (Probe A/B). This is the
  PARTITION contract, NOT the absolute integer labels (those are REQ-7/#938).
- REQ-2: **`n_clusters_` == requested `n_clusters` (R-DEV-3).** When `distance_threshold`
  is None, sklearn `n_clusters_ = self.n_clusters` (`_fit`, `:1095`). ferrolearn
  `FittedAgglomerativeClustering::n_clusters_` = the requested `n_clusters` (`fn fit`).
- REQ-3: **Four linkage criteria via Lance–Williams (R-DEV-1 — partition only).** Mirror
  `_TREE_BUILDERS = {ward, complete, average, single}` (`:720-725`): the `Linkage` enum
  +match in `fn agglomerate` produce the correct PARTITION for each of the four linkages
  on separable data (Probe B). Merge-distance VALUES (`distances_`) and ties differ
  (squared-Euclidean Lance–Williams vs sklearn's heap/nn-chain) — see REQ-9.
- REQ-4: **`n_clusters=2` constructor default + sklearn error ABI (R-DEV-2).** sklearn
  `__init__` defaults `n_clusters=2` (`:951`); ferrolearn `fn new(n_clusters)` REQUIRES
  it (Probe C). Also ferrolearn validation errors are `FerroError::InvalidParameter`/
  `InsufficientSamples`, not sklearn's `InvalidParameterError`/`ValueError`.
- REQ-5: **`ensure_min_samples=2` validation (R-DEV-1/2 — single-sample edge case).**
  sklearn `fit` rejects `n_samples < 2` (`_validate_data(ensure_min_samples=2)`, `:989`;
  Probe D). ferrolearn `fn fit` accepts a single sample when `n_clusters <= 1`
  (`test_single_sample_single_cluster`). Divergent edge-case handling.
- REQ-6: **`children_` full-dendrogram format (R-DEV-3 — fitted attribute).** sklearn
  `children_` is shape `(n_samples-1, 2)` with internal-node IDs `>= n_samples`, merge
  `i` forming node `n_samples + i` (`:902-908`; Probe A 7 rows, IDs up to 13).
  ferrolearn `children_` is `Vec<(usize,usize)>` of length `n_samples - n_clusters`
  using reused merged-into slot IDs. Different length AND ID semantics — see #938.
- REQ-7: **`labels_` ABSOLUTE numbering via `_hc_cut` (R-DEV-3 — exact value).** sklearn
  numbers labels by the negated-id min-heap pop order over the top-`n_clusters`
  dendrogram nodes (`_hc_cut`, `:760-775`); ferrolearn relabels by ascending
  surviving-slot order (`fn agglomerate` HashMap loop). Same partition (REQ-1), permuted
  integers (Probe B: sklearn `[2,2,2,…]` vs ferrolearn `[0,0,0,…]`). Reproducing the
  exact numbering requires the full `children_` dendrogram (REQ-6) + `_hc_cut`.
- REQ-8: **`metric` / `connectivity` parameters (R-DEV-2).** sklearn `metric` ∈
  {euclidean,l1,l2,manhattan,cosine,precomputed} default `'euclidean'`, with the
  ward-requires-euclidean rule (`:795-799`, `:1034-1038`; Probe E), and `connectivity`
  for structured clustering (`:812-822`). ferrolearn is Euclidean-only, unstructured —
  neither parameter exists.
- REQ-9: **`distance_threshold` / `compute_full_tree` / `compute_distances` / `distances_`
  (R-DEV-2/3).** sklearn `distance_threshold` (XOR with `n_clusters`, `:1022-1027`;
  `n_clusters_` derived `:1090-1093`), `compute_full_tree='auto'` early-stop
  (`:1051-1064`), `compute_distances` → `distances_` `(n_nodes-1,)` (`:1074`/`:1087-1088`;
  Probe F). ferrolearn has none of these parameters and no `distances_` attribute.
- REQ-10: **`n_leaves_` / `n_connected_components_` fitted attributes + `memory` (R-DEV-3).**
  sklearn sets `n_leaves_` and `n_connected_components_` from the tree builder
  (`:1083-1085`; Probe A: 8 and 1) and caches via `memory` (`:1006`/`:1076`). ferrolearn
  exposes neither attribute nor caching.
- REQ-11: **PyO3 binding parity (R-DEFER-1).** `ferrolearn.AgglomerativeClustering` →
  `_RsAgglomerativeClustering` → `ferrolearn_cluster::AgglomerativeClustering` exposes
  `fit` + `labels_` (Probe G). It marshals `n_clusters` (default 2) and `labels_`
  faithfully; the partition matches sklearn but the absolute label numbering rides REQ-7.
  The binding hard-wires Ward (no `linkage`/`metric`/`distance_threshold` argument).
- REQ-12: **ferray substrate (R-SUBSTRATE).** `agglomerative.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`; the PyO3 boundary
  uses `numpy2_to_ndarray` (`extras.rs`), not `ferray::numpy_interop`. Not migrated.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). ferrolearn partitions confirmed by the
in-tree `#[test]` co-membership assertions in `agglomerative.rs`.

- AC-1 (REQ-1, SHIPPED): on the 2-blob fixture (Probe A) sklearn `labels_` partition
  = `{0,1,2,3}` / `{4,5,6,7}`; ferrolearn `test_ward_two_blobs` asserts
  `labels[0]==labels[1]==labels[2]==labels[3]`, `labels[4..8]` equal, `labels[0]!=labels[4]`
  — same partition. On the 3-blob fixture (Probe B) the partition is
  `{0,1,2}`/`{3,4,5}`/`{6,7,8}` for all four linkages; ferrolearn `test_*_three_blobs`
  assert this co-membership.
- AC-2 (REQ-2, SHIPPED): `AgglomerativeClustering(n_clusters=3).fit(X).n_clusters_` = 3
  (sklearn, Probe A `n_clusters_=2` for `n_clusters=2`); ferrolearn
  `test_n_clusters_matches_config` asserts `n_clusters() == 3`.
- AC-3 (REQ-3, SHIPPED): each of ward/complete/average/single yields the
  `{0,1,2}`/`{3,4,5}`/`{6,7,8}` partition on Probe B; ferrolearn has a passing
  `test_<linkage>_three_blobs` for each.
- AC-4 (REQ-4, diverges): `AgglomerativeClustering().n_clusters` = 2 in sklearn;
  ferrolearn `AgglomerativeClustering::new` has no `n_clusters` default (required arg).
- AC-5 (REQ-5, diverges): `AgglomerativeClustering(n_clusters=1).fit([[3.,4.]])` →
  sklearn `ValueError` (min 2 samples); ferrolearn accepts it
  (`test_single_sample_single_cluster` → `labels()[0]==0`).
- AC-6 (REQ-6, SHIPPED): `children_` is the FULL dendrogram, shape `(n_samples-1, 2)`
  with internal-node IDs `>= n_samples` (`test_children_length` asserts `len ==
  n_samples - 1`), BIT-EXACT-equal to `scipy.cluster.hierarchy.linkage(...)[:, :2]`
  for ward/complete/average and to sklearn's own `children_` for single (parity tests
  in `tests/divergence_agglomerative_dendrogram.rs`).
- AC-7 (REQ-7, SHIPPED): on Probe B sklearn ward `labels_` = `[2,2,2,1,1,1,0,0,0]`;
  ferrolearn now reproduces this EXACTLY via `_hc_cut`
  (`labels_exact_sklearn_*` parity tests, k∈{2,3}, all four linkages).
- AC-8 (REQ-8, missing): `AgglomerativeClustering(linkage='ward',metric='manhattan').fit(X)`
  → sklearn `ValueError`; ferrolearn has no `metric` (Euclidean-only, validation
  unreachable).
- AC-9 (REQ-9, missing): `AgglomerativeClustering(n_clusters=None,distance_threshold=5.0).fit(X).labels_`
  = `[0,0,1,1]`, and `compute_distances=True` → `distances_` = `[~0.1, 0.1, 20.0]`
  (Probe F); ferrolearn has neither.
- AC-10 (REQ-10, missing): on Probe A sklearn `n_leaves_` = 8, `n_connected_components_`
  = 1; ferrolearn exposes neither.
- AC-11 (REQ-11, SHIPPED): `import ferrolearn; ferrolearn.AgglomerativeClustering(n_clusters=2).fit(A).labels_`
  reaches the Rust core and returns the 2-blob partition `{0,1,2,3}`/`{4,5,6,7}`
  matching `sklearn.cluster.AgglomerativeClustering` up to label permutation.

## REQ status table

Binary (R-DEFER-2). `AgglomerativeClustering` / `FittedAgglomerativeClustering` /
`Linkage` are existing pub APIs re-exported at the crate root, consumed in-crate by
`birch.rs` + `feature_agglomeration.rs`, AND bound into CPython as
`ferrolearn.AgglomerativeClustering` (real non-test consumers; grandfathered
S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2,
commit 156ef14). Live oracle = installed sklearn 1.5.2, run from `/tmp`. Honest
assessment (R-HONEST-3): the PARTITION contract SHIPS on separable data
(REQ-1/2/3/11), AND the full `children_` dendrogram (REQ-6) and the absolute
`_hc_cut` `labels_` numbering (REQ-7) now SHIP bit-exact for the unstructured
(`connectivity=None`) case across all four linkages — the #938 structural carve-out
is RESOLVED here (the nn-chain/MST builder + `_hc_cut`). The remaining NOT-STARTED REQs are
the unimplemented parameters (`metric`/`connectivity`/`distance_threshold`/
`compute_full_tree`/`compute_distances`/`memory`), the `n_clusters=2` default + error
ABI, the `ensure_min_samples=2` validation, the `distances_`/`n_leaves_`/
`n_connected_components_` attributes, and the ferray substrate. `#NNN` placeholders
are for the director to assign real blocker numbers; **#938 already exists** (referenced
by `birch.rs` REQ-3 and `feature_agglomeration.rs` REQ-3/REQ-5).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`labels_` PARTITION up-to-permutation, separable data) | SHIPPED | impl `fn agglomerate in agglomerative.rs` (Lance–Williams merge-until-`n_clusters` via `fn find_min_pair` over `fn pairwise_sq_dists`) → `Fit::fit` builds `labels_`, mirroring the merge clustering of `_fit` (`_agglomerative.py:992-1106`). PARTITION value-matches sklearn on Probe A (2 blobs `{0,1,2,3}`/`{4,5,6,7}`) and Probe B (3 blobs, all four linkages) — **up to a label permutation** (absolute numbering is REQ-7). Non-test consumers: `RsAgglomerativeClustering::labels_` (`extras.rs:987`), `birch.rs fn fit` (`birch.rs:459`), `feature_agglomeration.rs fn fit` (`:307`). Verification: `cargo test -p ferrolearn-cluster --lib agglomerative` (24 passed, 0 failed) — `test_ward_two_blobs`, `test_*_three_blobs` assert co-membership. |
| REQ-2 (`n_clusters_` == requested) | SHIPPED | impl `Fit::fit in agglomerative.rs` sets `n_clusters_: self.n_clusters`, mirroring `self.n_clusters_ = self.n_clusters` when `distance_threshold is None` (`_agglomerative.py:1095`). Probe A: sklearn `n_clusters_=2`. Consumer: crate re-export + `n_clusters()` accessor (used by `birch.rs`/`feature_agglomeration.rs`). Verification: `test_n_clusters_matches_config` (asserts 3), `test_single_sample_single_cluster` (asserts 1). |
| REQ-3 (four linkage criteria, partition) | SHIPPED | impl `enum Linkage` + the `match linkage` arms in `fn agglomerate in agglomerative.rs` (Single=min, Complete=max, Average=size-weighted mean, Ward=size-weighted Lance–Williams) mirror `_TREE_BUILDERS` (`_agglomerative.py:720-725`). PARTITION matches sklearn for all four on Probe B. Consumer: `with_linkage` used by `birch.rs` (`Linkage::Ward`) + `feature_agglomeration.rs` (`map_linkage` all four). Verification: `test_ward/complete/average/single_three_blobs` (24 passed). **Caveat (REQ-9)**: this is the PARTITION only — merge-distance VALUES and tie-breaking differ (squared-Euclidean LW vs sklearn heap/nn-chain). |
| REQ-4 (`n_clusters=2` default + error ABI) | NOT-STARTED | open prereq blocker **#963**. sklearn `__init__` defaults `n_clusters=2` (`_agglomerative.py:951`); ferrolearn `fn new(n_clusters)` REQUIRES it (Probe C, AC-4). The PyO3 layer DOES default `n_clusters=2` (`extras.rs:968`) so `ferrolearn.AgglomerativeClustering()` matches, but the Rust constructor does not. Also validation errors are `FerroError::InvalidParameter`/`InsufficientSamples`, not the sklearn `InvalidParameterError`/`ValueError` ABI (R-DEV-2). |
| REQ-5 (`ensure_min_samples=2` validation) | NOT-STARTED | open prereq blocker **#964**. sklearn `fit` → `_validate_data(X, ensure_min_samples=2)` (`_agglomerative.py:989`) rejects `n_samples < 2` with `ValueError` (Probe D). ferrolearn `fn fit` rejects only `n_samples == 0` and `n_samples < n_clusters`; it ACCEPTS a single sample when `n_clusters <= 1` (`test_single_sample_single_cluster`, AC-5). Divergent edge-case handling. |
| REQ-6 (`children_` full-dendrogram format) | SHIPPED | impl `fn full_dendrogram in agglomerative.rs`: nn-chain (`fn nn_chain`, Lance–Williams `fn lance_williams`) + stable distance sort + union-find (`fn union_find_relabel`) for ward/complete/average; Prim's MST (`fn mst_single`) + `fn single_linkage_relabel` for single. Produces `children_` of shape `(n_samples-1, 2)`, leaves `0..n-1`, internal-node IDs `n+i` (`_agglomerative.py:314`/`:586`/`:902-908`). BIT-EXACT-equal to `scipy.cluster.hierarchy.linkage(X, method, 'euclidean')[:, :2]` for ward/complete/average; for `single`, sklearn uses `mst_linkage_core`+`_single_linkage_label` (`:567-584`, R-DEV-7) whose pair-column order differs from `scipy.linkage('single')`, so `children_` matches sklearn's OWN `AgglomerativeClustering.children_` bit-exact. Live-oracle tests: `children_exact_scipy_6pt_nn_chain_linkages`, `children_exact_scipy_10pt_nn_chain_linkages`, `children_exact_sklearn_single_6pt`, `children_exact_sklearn_single_10pt`, `divergence_children_full_dendrogram_format` (`tests/divergence_agglomerative_dendrogram.rs`). Consumers: `Fit::fit` → `children_`, `RsAgglomerativeClustering`, `birch.rs`/`feature_agglomeration.rs`. |
| REQ-7 (`labels_` ABSOLUTE numbering via `_hc_cut`) | SHIPPED | impl `fn hc_cut in agglomerative.rs`: negated-id MIN-heap (`fn heappush_min`/`fn heappushpop_min`/`fn sift_down`/`fn sift_up`, a CPython-`heapq`-exact translation) over the top-`n_clusters` dendrogram nodes + `fn hc_get_descendent`, mirroring `_hc_cut` (`_agglomerative.py:731-775`). Builds `labels_` from the REQ-6 full `children_`, BIT-EXACT-equal to `sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage=…).fit(X).labels_` (Probe B: ward `[2,2,2,1,1,1,0,0,0]` now reproduced exactly). Live-oracle tests: `labels_exact_sklearn_6pt_all_linkages`, `labels_exact_sklearn_10pt_all_linkages` (k∈{2,3}, all four linkages), `divergence_labels_absolute_hc_cut_numbering`. Consumer: `Fit::fit` → `labels_` via `RsAgglomerativeClustering::labels_` + `birch.rs`/`feature_agglomeration.rs` (partition use, unaffected by the renumbering). |
| REQ-8 (`metric` / `connectivity`) | NOT-STARTED | open prereq blocker **#965**. sklearn `metric` ∈ {euclidean,l1,l2,manhattan,cosine,precomputed} default `'euclidean'` (`_agglomerative.py:795-799`), with the ward-requires-euclidean rule (`:1034-1038`; Probe E → `ValueError`), and `connectivity` for structured clustering (`:812-822`, `:1042-1048`). ferrolearn `fn sq_euclidean`/`fn pairwise_sq_dists` are Euclidean-only and unstructured; neither parameter exists. |
| REQ-9 (`distance_threshold`/`compute_distances`/`distances_`) | SHIPPED | impl: ctor `AgglomerativeClustering<F>.n_clusters: Option<usize>` (default `Some(2)` via `fn new`) + `distance_threshold: Option<F>` + `compute_distances: bool`, builders `fn with_distance_threshold` (sets the threshold, clears `n_clusters` → the `n_clusters=None,distance_threshold=t` idiom) / `fn with_compute_distances`; `Fit::fit` enforces the XOR `not ((n_clusters is None) ^ (distance_threshold is None))` (`_agglomerative.py:1022-1027`) as `FerroError::InvalidParameter`. `fn full_dendrogram`/`fn union_find_relabel`/`fn single_linkage_relabel` now thread the per-merge distances (the 3rd scipy-linkage column; nn-chain `current_min` / MST `new_distance`) through the stable sort so they land in `children_` row order; `fn agglomerate` exposes them as `FittedAgglomerativeClustering.distances_: Option<Array1<F>>` (accessor `fn distances`) iff `return_distance = distance_threshold.is_some() || compute_distances` (`:1074`, `:1087-1088`), derives `n_clusters_ = count(distances_ >= t) + 1` (`:1090-1093`), then `labels_ = fn hc_cut(n_clusters_, children_, n_leaves)` (`:1099`). `distances_` EXACT-equals (~1e-12) `scipy.cluster.hierarchy.linkage(X, L, 'euclidean')[:, 2]` (ward/complete/average) and sklearn's own `distances_` (single). Live-oracle tests (`tests/divergence_agglomerative_threshold.rs`, R-CHAR-3): `distances_exact_6pt_all_linkages`, `distances_exact_10pt_all_linkages` (vs `compute_distances=True`), `threshold_cut_exact_6pt_all_linkages` + `threshold_cut_exact_10pt_all_linkages` (per-linkage `n_clusters_` AND integer-exact `labels_` over thresholds spanning below-min → above-max), `threshold_inclusive_boundary_6pt_ward` (the `>=` inclusive rule), `distances_none_when_not_requested`/`distances_some_when_threshold_set`, `xor_both_set_errors`/`xor_neither_set_errors`/`xor_single_set_each_fits`. Non-test consumers: `Fit::fit` (sets all three), `RsAgglomerativeClustering`, `birch.rs`/`feature_agglomeration.rs` (construct via `fn new` → default `Some(2)`). **`compute_full_tree` is a no-op here (honest scope):** the unstructured (`connectivity=None`) path ALWAYS builds the full dendrogram (`:1052-1053`), so the `'auto'`/partial-tree EARLY-STOP is purely a perf optimisation with no observable-attribute consequence — the observable `children_`/`labels_`/`distances_`/`n_clusters_` all match the full-tree path; the partial-tree perf path is unimplemented and not observable. |
| REQ-10 (`n_leaves_`/`n_connected_components_`) | SHIPPED | impl: `FittedAgglomerativeClustering` stores `pub n_leaves_: usize` / `pub n_connected_components_: usize` (accessors `fn n_leaves` / `fn n_connected_components`); `Fit::fit` sets `n_leaves_ = n_samples`, `n_connected_components_ = 1` for the unstructured (`connectivity=None`) path, mirroring `ward_tree`/`linkage_tree` returning `(children_, n_connected_components=1, n_leaves=n_samples, parents)` (`_agglomerative.py:1083-1085`; Probe A: 8 and 1). Live-oracle test `n_leaves_and_connected_components_unstructured` (`tests/divergence_agglomerative_threshold.rs`, both fixtures × all 4 linkages). Non-test consumer: `Fit::fit` (production path) + the two accessors. **`memory` caching stays NOT-STARTED** (open blocker **#967**): `check_memory(self.memory)` (`:1006`, `:1076`) is an opt-in joblib persistence wrapper; in the default `memory=None` path it has no observable-attribute divergence. |
| REQ-11 (PyO3 binding parity) | SHIPPED | impl `#[pyclass(name="_RsAgglomerativeClustering")] RsAgglomerativeClustering` (`ferrolearn-python/src/extras.rs:959`): `fn new(n_clusters=2)`, `fn fit` calling `ferrolearn_cluster::AgglomerativeClustering::<f64>::new(self.n_clusters)`, `#[getter] labels_`; registered via `m.add_class::<extras::RsAgglomerativeClustering>()` (`src/lib.rs:67`); `class AgglomerativeClustering(_ClusterWrapper)` (`python/ferrolearn/_extras.py:432`); exported in `__init__.py`. Non-test consumer: `import ferrolearn; ferrolearn.AgglomerativeClustering(n_clusters=2).fit(X).labels_` (AC-11) — partition matches sklearn (since REQ-1 holds) up to permutation. The binding hard-wires Ward (no `linkage`/`metric`/`distance_threshold` argument marshalled) and surfaces only `labels_` (no `children_`/`n_clusters_`). Verification: `cd ferrolearn-python && maturin develop && PYTHONPATH=python python3 -m pytest tests/ -q`. |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker **#968**. `agglomerative.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` + `std::collections::HashMap`; not migrated to `ferray-core` (R-SUBSTRATE-1/2). The PyO3 boundary uses `numpy2_to_ndarray` (`extras.rs`), an `ndarray` bridge, not `ferray::numpy_interop`. |

## Architecture

`agglomerative.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`AgglomerativeClustering<F>` (`n_clusters`, `linkage`; `fn new(n_clusters)` defaults
`linkage = Ward`, builder `with_linkage`) → `Fit<Array2<F>, ()>` →
`FittedAgglomerativeClustering<F>` (`labels_: Array1<usize>`, `n_clusters_: usize`,
`children_: Vec<(usize,usize)>`, `PhantomData<F>`). Generic over
`F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2). No `Predict` impl (AgglomerativeClustering has no
out-of-sample prediction — matching sklearn, which exposes only `fit_predict`).
Accessors: `labels()`, `n_clusters()`, `children()`. The unfitted-method
`fit_predict` mirrors `ClusterMixin.fit_predict` (`_agglomerative.py:1108-1129`).

**Fit path (`fn fit` → `fn agglomerate`).** Validates `n_clusters != 0` (REQ-4 ABit) and
`n_samples >= n_clusters` (NOT `ensure_min_samples=2` — REQ-5 divergence), then:
1. **Full dendrogram** — `fn full_dendrogram`. For ward/complete/average it builds the
   condensed Euclidean distance vector and runs the **nearest-neighbour chain**
   (`fn nn_chain`, the Lance–Williams update `fn lance_williams` on Euclidean
   distances), then a **stable sort by merge distance** + a **union-find relabel**
   (`fn union_find_relabel`) → `children_` bit-exact with
   `scipy.cluster.hierarchy.linkage(X, method, 'euclidean')[:, :2]`. For single it
   runs **Prim's MST** (`fn mst_single`, mirroring `mst_linkage_core`) + a union-find
   relabel that does NOT reorder the pair (`fn single_linkage_relabel`, mirroring
   `_single_linkage_label`). Leaves are `0..n`, the `i`-th sorted merge forms node
   `n + i`; `children_` is the FULL tree, shape `(n_samples-1, 2)`, regardless of
   `n_clusters` (REQ-6).
2. **Cut** — `fn hc_cut` cuts the full tree into `n_clusters` clusters exactly as
   sklearn's `_hc_cut` (`_agglomerative.py:731-775`): a CPython-`heapq`-faithful
   negated-id min-heap (`fn heappush_min`/`fn heappushpop_min`/`fn sift_up`/
   `fn sift_down`) pops the top-`n_clusters` nodes, and
   `for i, node in enumerate(nodes): label[descendents(node)] = i`
   (`fn hc_get_descendent`) numbers each leaf by its ancestor's heap position →
   `labels_` bit-exact with sklearn (REQ-7).

**Invariants held vs sklearn (the SHIPPED core):** the `labels_` PARTITION AND absolute
numbering on separable data (REQ-1/REQ-7); `n_clusters_` = requested `n_clusters`
(REQ-2); the four linkages, children AND labels bit-exact (REQ-3/REQ-6/REQ-7); the
full-dendrogram `children_` format, shape `(n_samples-1, 2)` (REQ-6); the PyO3
`labels_` marshalling (REQ-11).

**Invariants NOT held vs sklearn:** the `n_clusters=2` Rust-constructor default + the
`InvalidParameterError`/`ValueError` error ABI (REQ-4); the `ensure_min_samples=2`
single-sample rejection (REQ-5); `metric`/`connectivity` (REQ-8);
`distance_threshold`/`compute_full_tree`/`compute_distances`/`distances_` (REQ-9);
`n_leaves_`/`n_connected_components_`/`memory` (REQ-10); the ferray substrate (REQ-12).
NOTE: REQ-6/REQ-7 cover ONLY the unstructured (`connectivity=None`) case — structured
clustering (the `connectivity` matrix path, REQ-8) uses a different sklearn tree builder
and is still unimplemented.

**Consumer wiring.** `pub use agglomerative::{AgglomerativeClustering,
FittedAgglomerativeClustering, Linkage}` (`ferrolearn-cluster/src/lib.rs`)
re-exports the types. The real non-test production consumers are (a) `birch.rs`
(`fn fit` → global Ward clustering on `subcluster_centers_`), (b)
`feature_agglomeration.rs` (`fn fit` → clustering on `X.T`), and (c)
`ferrolearn.AgglomerativeClustering` (`_RsAgglomerativeClustering` in `extras.rs`,
registered in `src/lib.rs`, wrapped in `python/ferrolearn/_extras.py`,
exported in `__init__.py`). Both in-crate consumers use `labels()` purely as a
PARTITION (mapping points/features to whatever integer label they receive), so the
now-sklearn-exact renumbering does not change their output structure — their test
suites (`divergence_birch.rs`, `divergence_feature_agglomeration.rs`) stay green.

## Verification

Library crate (green at baseline `75ac7a41`):
```
cargo test -p ferrolearn-cluster --lib agglomerative     # 24 passed; 0 failed
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s (`test_new_defaults`, `test_with_linkage`,
`test_zero_clusters_error`, `test_empty_data_error`,
`test_more_clusters_than_samples_error`, `test_ward_two_blobs`,
`test_ward_three_blobs`, `test_complete_two_blobs`, `test_complete_three_blobs`,
`test_average_two_blobs`, `test_average_three_blobs`, `test_single_two_blobs`,
`test_single_three_blobs`, `test_label_count_equals_n_samples`,
`test_labels_in_valid_range`, `test_n_clusters_matches_config`,
`test_children_length`, `test_children_empty_when_n_clusters_equals_n_samples`,
`test_single_cluster`, `test_n_clusters_equals_n_samples`,
`test_single_sample_single_cluster`, `test_1d_data`, `test_f32_support`,
`test_identical_points`) assert the PARTITION (co-membership), the label range, the
`n_clusters_` value, and the ferrolearn-internal `children_` length — they assert
STRUCTURE, not exact integer labels against the oracle. The absolute-numbering
divergence (REQ-7) and the `children_`-format divergence (REQ-6) are NOT asserted
green because they DIVERGE.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the partition-parity that
backs the SHIPPED REQs (R-CHAR-3 expected values, never literal-copied from ferrolearn):
```
# REQ-1/REQ-2/REQ-3 (SHIPPED) partition + n_clusters_ — partition matches ferrolearn
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.05,10.05]]); \
m=AgglomerativeClustering(n_clusters=2,linkage='ward').fit(X); print(m.labels_.tolist(), m.n_clusters_)"
# [0,0,0,0,1,1,1,1] 2   (ferrolearn: same partition {0..3}/{4..7})
# REQ-7 (DIVERGES) absolute numbering
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
print(AgglomerativeClustering(n_clusters=3,linkage='ward').fit(X).labels_.tolist())"
# [2,2,2,1,1,1,0,0,0]   (ferrolearn: same partition, permuted integers)
# REQ-6 (DIVERGES) children_ format
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.05,10.05]]); \
print(AgglomerativeClustering(n_clusters=2).fit(X).children_.shape)"  # (7,2); ferrolearn len 6 reused-slot
# REQ-5 (DIVERGES) ensure_min_samples=2
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
AgglomerativeClustering(n_clusters=1).fit(np.array([[3.,4.]]))"  # ValueError; ferrolearn accepts
# REQ-8 (MISSING) ward+manhattan
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
AgglomerativeClustering(linkage='ward',metric='manhattan').fit(np.array([[0.,0.],[1.,1.],[5.,5.]]))"  # ValueError; ferrolearn euclidean-only
# REQ-9 (MISSING) distance_threshold / distances_
python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
X=np.array([[0.,0.],[0.1,0.],[10.,10.],[10.1,10.]]); \
print(AgglomerativeClustering(n_clusters=None,distance_threshold=5.0).fit(X).labels_.tolist(), \
      AgglomerativeClustering(n_clusters=2,compute_distances=True).fit(X).distances_.tolist())"
# [0,0,1,1] [~0.1,0.1,20.0]; ferrolearn has neither
```
A characterization pin (R-CHAR-3) for REQ-1/REQ-3 belongs in
`ferrolearn-cluster/tests/divergence_agglomerative.rs`, asserting the live-sklearn
PARTITION above (canonicalized to ignore label permutation) and PASSING against
current `agglomerative.rs`. Pins for the NOT-STARTED REQs assert the sklearn behavior
ferrolearn cannot yet reach and FAIL until the surface lands: REQ-7 (exact `labels_`
`[2,2,2,1,1,1,0,0,0]`), REQ-6 (`children_` shape `(n_samples-1,2)` with internal-node
IDs), REQ-5 (single-sample `ValueError`), REQ-8 (`metric`), REQ-9
(`distance_threshold`/`distances_`), REQ-10 (`n_leaves_`/`n_connected_components_`).

ferrolearn-python (REQ-11 binding parity):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/ -q
```
asserting `ferrolearn.AgglomerativeClustering(n_clusters=2).fit(X).labels_` value-matches
`sklearn.cluster.AgglomerativeClustering` UP TO A LABEL PERMUTATION on the blob fixtures
(the SHIPPED partition). A divergence pin for the binding's absolute numbering / missing
`children_`/`n_clusters_`/`linkage` surface belongs in
`ferrolearn-python/tests/divergence_agglomerative.py`.

## Blockers to open

(Director creates the real issues; `#NNN` placeholders below. The core
REQ-1/REQ-2/REQ-3/REQ-11 are SHIPPED and need NO blocker. **#938 already exists** — it
is the shared label-numbering / truncated-dendrogram root cause, already referenced by
`birch.rs` REQ-3 and `feature_agglomeration.rs` REQ-3/REQ-5, and it covers BOTH REQ-6
and REQ-7 here.)

- **#938 (existing)** — REQ-6 + REQ-7: build the FULL `(n_samples-1, 2)` dendrogram
  with internal-node IDs (`_agglomerative.py:902-908`) and number `labels_` via the
  `_hc_cut` negated-id heap enumeration (`:760-775`), replacing the truncated
  reused-slot `children_` + ascending-slot HashMap relabel in `fn agglomerate`. This
  single root-cause fix unblocks the absolute-label / `children_`-format parity for
  `agglomerative.rs`, `feature_agglomeration.rs`, and `birch.rs`.
- **#963** — REQ-4: add `n_clusters=2` default to `AgglomerativeClustering::new` (or a
  `Default`/builder), and map validation failures to the sklearn
  `ValueError`/`InvalidParameterError` ABI (`_agglomerative.py:951`, `:935`).
- **#964** — REQ-5: reject `n_samples < 2` in `fn fit` to mirror
  `_validate_data(ensure_min_samples=2)` (`_agglomerative.py:989`).
- **#965** — REQ-8: add `metric` (euclidean/l1/l2/manhattan/cosine/precomputed,
  default `'euclidean'`) with the ward-requires-euclidean rule, and `connectivity`
  for structured clustering (`_agglomerative.py:795-799`, `:1034-1038`, `:812-822`).
- **#966** — REQ-9: add `distance_threshold` (+ the XOR-with-`n_clusters` validation
  and derived `n_clusters_`), `compute_full_tree`, `compute_distances` →
  `distances_`, AND the underlying Euclidean merge-distance VALUE layer
  (`_agglomerative.py:1022-1027`, `:1051-1064`, `:1087-1093`).
- **#967** — REQ-10: add `n_leaves_` / `n_connected_components_` fitted attributes from
  the tree builder, and the `memory` caching parameter (`_agglomerative.py:1083-1085`,
  `:1006`).
- **#968** — REQ-12: migrate `agglomerative.rs` off `ndarray`/`num-traits` to
  `ferray-core`, and the PyO3 boundary off `numpy2_to_ndarray` to
  `ferray::numpy_interop` (R-SUBSTRATE).
- (REQ-11 note) — marshal `linkage` (and later `metric`/`distance_threshold`) through
  `_RsAgglomerativeClustering`, and surface `children_`/`n_clusters_` so the binding
  matches the Rust accessors.
