# AgglomerativeClustering (sklearn.cluster.AgglomerativeClustering)

<!--
tier: 3-component
status: draft
baseline-commit: 75ac7a41
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_agglomerative.py     # class AgglomerativeClustering(ClusterMixin, BaseEstimator) (:781); _parameter_constraints (:934-947); __init__ defaults n_clusters=2/metric='euclidean'/linkage='ward'/distance_threshold=None/compute_full_tree='auto'/compute_distances=False (:949-968); fit -> _validate_data(ensure_min_samples=2) (:989); _fit n_clusters^distance_threshold XOR validation (:1022-1027); ward requires euclidean (:1034-1038); _TREE_BUILDERS dispatch (:720-725, :1040); children_/n_connected_components_/n_leaves_ (:1083-1085); distances_ + n_clusters_ from distance_threshold (:1087-1095); _hc_cut tree-cut + label numbering (:731-775); children_ doc (n_samples-1, 2) full dendrogram (:902-908)
parity-ops: AgglomerativeClustering (.__init__, .fit, .fit_predict, .labels_, .n_clusters_, .children_)
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

**The structural carve-out (the root of #938).** ferrolearn builds the dendrogram
fundamentally differently from sklearn in two coupled ways:

1. **Truncated `children_` with reused-slot IDs vs. the full dendrogram with
   internal-node IDs.** sklearn's `children_` is shape `(n_samples-1, 2)` (the
   FULL dendrogram, regardless of `n_clusters`), where a value `>= n_samples` is an
   internal node and merge `i` forms node `n_samples + i`
   (`_agglomerative.py:902-908`). ferrolearn's `children_` (`FittedAgglomerativeClustering::children_`,
   type `Vec<(usize, usize)>`) is the TRUNCATED merge list of length
   `n_samples - n_clusters` using the **merged-into slot ID** `ci` for the new
   cluster (`fn agglomerate`: `children.push((ci, cj))`, then samples assigned to
   `cj` are redirected to `ci`). Different length AND different ID semantics.

2. **HashMap relabel-by-surviving-root vs. `_hc_cut` heap-pop enumeration.** sklearn
   numbers `labels_` by cutting the full tree: a negated-id min-heap pops the
   top-`n_clusters` dendrogram nodes, and `for i, node in enumerate(nodes):
   label[descendents] = i` (`_hc_cut`, `_agglomerative.py:760-775`) — so the
   absolute label of a sample depends on heap-pop order over internal node IDs.
   ferrolearn relabels by ascending surviving-slot order via a `HashMap` built from
   `active.iter().enumerate()` (`fn agglomerate`, the re-label loop). The
   **PARTITION** (which samples share a cluster) agrees with sklearn on
   well-separated data, but the **ABSOLUTE LABEL NUMBERING differs** — e.g. on the
   3-blob fixture (Probe B) sklearn ward yields `[2,2,2,1,1,1,0,0,0]` while
   ferrolearn's slot-order relabel yields the same partition with permuted integers.
   Reproducing sklearn's exact `labels_` requires first building the full
   `children_` dendrogram and then running `_hc_cut`'s heap enumeration — **not a
   minimal fix**, it is a structural rewrite of both the tree representation and the
   cut. This is the shared root cause that `birch.rs` (REQ-3) and
   `feature_agglomeration.rs` (REQ-3/REQ-5) already gate on as blocker **#938**.

A secondary structural fact: ferrolearn computes linkage VALUES via the
Lance–Williams recurrence on the **squared-Euclidean** distance matrix (`fn agglomerate`,
`Linkage::Ward` arm uses the size-weighted Lance–Williams update on `sq_dists`),
whereas sklearn's `ward_tree`/`linkage_tree` operate on Euclidean distances with a
heap / nearest-neighbor chain (`_TREE_BUILDERS`, `_agglomerative.py:720-725`). For
well-separated data the PARTITION agrees; tie-breaking and the merge-distance VALUES
(which sklearn can surface via `distances_`) differ.

The NOT-STARTED surface is the parameters/attributes ferrolearn does not expose:
`metric`, `connectivity`, `compute_full_tree`, `distance_threshold`,
`compute_distances`, `memory`; the `n_clusters=2` constructor default; the
`ensure_min_samples=2` validation; the `distances_`/`n_connected_components_`/
`n_leaves_` fitted attributes; the sklearn error ABI; the absolute `labels_`
numbering and the full `children_` dendrogram (both #938); and the ferray substrate.

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
- AC-6 (REQ-6, diverges): on Probe A sklearn `children_.shape == (7,2)` with IDs up to
  13; ferrolearn `children().len() == 6` (`n_samples - n_clusters`,
  `test_children_length`) of reused-slot pairs.
- AC-7 (REQ-7, diverges): on Probe B sklearn ward `labels_` = `[2,2,2,1,1,1,0,0,0]`;
  ferrolearn yields the same PARTITION with permuted integers (blob `{0,1,2}` → label 0).
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
assessment (R-HONEST-3): the PARTITION contract genuinely SHIPS on separable data
(REQ-1/2/3/11) through real consumers, but the absolute `labels_` numbering (REQ-7)
and the full `children_` dendrogram (REQ-6) DIVERGE — both rooted in the truncated-tree
+ slot-relabel design, the shared #938 root cause. The remaining NOT-STARTED REQs are
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
| REQ-6 (`children_` full-dendrogram format) | NOT-STARTED | open prereq blocker **#938** (the shared root cause). sklearn `children_` is shape `(n_samples-1, 2)`, internal-node IDs `>= n_samples`, merge `i` forms node `n_samples + i` (`_agglomerative.py:902-908`; Probe A: 7 rows, IDs to 13). ferrolearn `children_` (`FittedAgglomerativeClustering::children_`, built in `fn agglomerate` as `children.push((ci, cj))`) is length `n_samples - n_clusters` of reused merged-into-slot pairs (`test_children_length` asserts `len == n_samples - n_clusters`, a ferrolearn-INTERNAL contract, NOT sklearn's). Different length AND ID semantics — a full-dendrogram rewrite, not a minimal fix. |
| REQ-7 (`labels_` ABSOLUTE numbering via `_hc_cut`) | NOT-STARTED | open prereq blocker **#938** (the shared root cause). sklearn numbers `labels_` by negated-id min-heap pop over the top-`n_clusters` dendrogram nodes + `for i,node in enumerate(nodes): label[descendents]=i` (`_hc_cut`, `_agglomerative.py:760-775`). ferrolearn relabels by ascending surviving-slot order via a `HashMap` from `active.iter().enumerate()` (`fn agglomerate` re-label loop). Probe B: sklearn ward `[2,2,2,1,1,1,0,0,0]` vs ferrolearn same partition with permuted integers (AC-7). Exact numbering requires the full `children_` (REQ-6) then `_hc_cut`. |
| REQ-8 (`metric` / `connectivity`) | NOT-STARTED | open prereq blocker **#965**. sklearn `metric` ∈ {euclidean,l1,l2,manhattan,cosine,precomputed} default `'euclidean'` (`_agglomerative.py:795-799`), with the ward-requires-euclidean rule (`:1034-1038`; Probe E → `ValueError`), and `connectivity` for structured clustering (`:812-822`, `:1042-1048`). ferrolearn `fn sq_euclidean`/`fn pairwise_sq_dists` are Euclidean-only and unstructured; neither parameter exists. |
| REQ-9 (`distance_threshold`/`compute_full_tree`/`compute_distances`/`distances_`) | NOT-STARTED | open prereq blocker **#966**. sklearn `distance_threshold` (XOR with `n_clusters`, `_agglomerative.py:1022-1027`; `n_clusters_` derived `:1090-1093`), `compute_full_tree='auto'` early-stop (`:1051-1064`), `compute_distances` → `distances_` shape `(n_nodes-1,)` (`:1074`, `:1087-1088`; Probe F: dt=5 → `[0,0,1,1]`, `distances_ [~0.1,0.1,20.0]`). ferrolearn `AgglomerativeClustering<F>` has only `n_clusters` + `linkage` — none of these parameters and no `distances_` attribute. The merge-distance VALUES also differ (squared-Euclidean LW vs Euclidean heap/nn-chain), so even a `distances_` accessor would not value-match without REQ-3's value layer. |
| REQ-10 (`n_leaves_`/`n_connected_components_` + `memory`) | NOT-STARTED | open prereq blocker **#967**. sklearn sets `self.n_leaves_`/`self.n_connected_components_` from the tree builder (`_agglomerative.py:1083-1085`; Probe A: 8 and 1) and caches via `memory` (`:1006`, `:1076`). `FittedAgglomerativeClustering` exposes `labels()`/`n_clusters()`/`children()` only — neither attribute, no caching. Missing attributes. |
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
1. **Distance matrix** — `fn pairwise_sq_dists` builds the flat `n × n` row-major
   matrix of **squared** Euclidean distances (`fn sq_euclidean`). Data is converted to
   `f64` upfront for the internal computation.
2. **Merge loop** — while `active.len() > n_clusters`: `fn find_min_pair` scans the
   active clusters for the closest pair `(ci, cj)`; `cj` is removed from `active`,
   `(ci, cj)` is pushed to `children`, and the **Lance–Williams recurrence** updates
   `sq_dists[ci][ck]` for every remaining `ck` per the `Linkage` arm (Single=min,
   Complete=max, Average=size-weighted mean, Ward=`((ni+nk)d_ik + (nj+nk)d_jk -
   nk·d_ij)/(ni+nj+nk)`). `sizes[ci]` accumulates, and every sample assigned to `cj`
   is redirected to `ci`. This is `_TREE_BUILDERS`'s merge clustering (`:720-725`)
   expressed as a dense O(n³)/O(n²) Lance–Williams update rather than sklearn's
   heap (`ward_tree`) / nearest-neighbor chain (`linkage_tree`) on Euclidean distances.
3. **Relabel** — a `HashMap` maps each surviving slot ID in `active.iter().enumerate()`
   order to `0..n_clusters`; `assignment` is mapped through it to `labels_`. This is
   the ascending-slot numbering that DIVERGES from sklearn's `_hc_cut` heap enumeration
   (REQ-7).

**Invariants held vs sklearn (the SHIPPED core):** the `labels_` PARTITION on
separable data (REQ-1 — co-membership matches across all four linkages); `n_clusters_`
= requested `n_clusters` (REQ-2); the four linkage partitions (REQ-3); the PyO3
`labels_` marshalling (REQ-11). The ferrolearn-INTERNAL `children_` length
(`n_samples - n_clusters`, `test_children_length`) is an internal contract, NOT a
sklearn match (sklearn's is `n_samples - 1`).

**Invariants NOT held vs sklearn:** the `n_clusters=2` Rust-constructor default + the
`InvalidParameterError`/`ValueError` error ABI (REQ-4); the `ensure_min_samples=2`
single-sample rejection (REQ-5); the full-dendrogram `children_` format (REQ-6, #938);
the absolute `labels_` numbering via `_hc_cut` (REQ-7, #938); `metric`/`connectivity`
(REQ-8); `distance_threshold`/`compute_full_tree`/`compute_distances`/`distances_`
(REQ-9 — including the merge-distance VALUE divergence); `n_leaves_`/
`n_connected_components_`/`memory` (REQ-10); the ferray substrate (REQ-12).

**Consumer wiring.** `pub use agglomerative::{AgglomerativeClustering,
FittedAgglomerativeClustering, Linkage}` (`ferrolearn-cluster/src/lib.rs:97`)
re-exports the types. The real non-test production consumers are (a) `birch.rs`
(`fn fit` → global Ward clustering on `subcluster_centers_`, `birch.rs:459`), (b)
`feature_agglomeration.rs` (`fn fit` → clustering on `X.T`, `:307`), and (c)
`ferrolearn.AgglomerativeClustering` (`_RsAgglomerativeClustering` in `extras.rs:959`,
registered in `src/lib.rs:67`, wrapped in `python/ferrolearn/_extras.py:432`,
exported in `__init__.py`). Both in-crate consumers inherit the #938 label-numbering
divergence — that is exactly why `birch.rs` REQ-3 and `feature_agglomeration.rs`
REQ-3/REQ-5 cite the partition contract as SHIPPED but the absolute `labels_` VALUE
as NOT-STARTED/#938.

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
