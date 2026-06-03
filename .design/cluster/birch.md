# BIRCH Clustering (sklearn.cluster.Birch)

<!--
tier: 3-component
status: draft
baseline-commit: beb6862e
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_birch.py   # class Birch(ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin, BaseEstimator) (:361-741); __init__ (:491-504); _parameter_constraints (:483-489); _fit (:526-595); _CFNode (:111-263); _CFSubcluster (:266-358); _split_node (:48-108); _global_clustering (:703-738); predict/_predict (:651-679); transform (:681-701); partial_fit (:613-638)
ferrolearn-module: ferrolearn-cluster/src/birch.rs
parity-ops: Birch (.__init__, .fit, .fit_predict, .labels_, .subcluster_centers_, .subcluster_labels_, .predict, .transform, .partial_fit)
crosslink-issue: (cluster-layer tracking; no birch-specific issue yet — director creates the suggested blockers below)
-->

## Summary

`ferrolearn-cluster/src/birch.rs` mirrors scikit-learn's `Birch`
(`sklearn/cluster/_birch.py`, `class Birch(...)` `:361-741`) — the memory-efficient
incremental CF-tree clustering algorithm. It exposes the unfitted `Birch<F>`
(`threshold`, `branching_factor`, `n_clusters`), the fitted `FittedBirch<F>`
(`labels_`, `subcluster_centers_`, `n_clusters_`), a `fit_predict` convenience
mirroring `ClusterMixin.fit_predict`, and the `Fit<Array2<F>, ()>` impl. It is
re-exported at the crate root (`pub use birch::{Birch, FittedBirch}` in
`ferrolearn-cluster/src/lib.rs`) and reaches Python through the **non-test consumer**
`_RsBirch` (`pub struct RsBirch` in `ferrolearn-python/src/extras.rs`), wrapped by the
`Birch` class in `ferrolearn-python/python/ferrolearn/_extras.py`.

**Honest assessment (R-HONEST-3).** Two behaviors VALUE-match the live sklearn 1.5.2
oracle on benign (few-subcluster) fixtures and have a real non-test consumer:
the **mean-based `subcluster_centers_`** (CF centroid `LS/N`) and the
**final `labels_` partition up to a label permutation**. Both are SHIPPED — but ONLY
within the regime where the number of leaf subclusters stays `<= branching_factor`.
The CF data structure ferrolearn implements is a **flat `Vec<ClusteringFeature>`**,
NOT a balanced CF-tree: once the subcluster count would exceed `branching_factor` it
**merges the two closest subclusters** (`build_cf_tree`, the `subclusters.len() >=
branching_factor` branch) instead of splitting a leaf node and growing the tree. The
divergence is structural and large (Probe 4): on a 60-point spread at
`branching_factor=5`, sklearn yields **37** leaf subclusters; ferrolearn caps at
**5**. So `subcluster_centers_` is a value contract only when sklearn itself produces
`<= branching_factor` subclusters. The whole `partial_fit` / `predict` / `transform`
/ `subcluster_labels_` surface and the `compute_labels` / `copy` parameters are
absent.

## Live oracle probes (sklearn 1.5.2, run from /tmp; ferrolearn via throwaway `cargo run --example` probe, since deleted)

All expected values from the installed sklearn 1.5.2 oracle, never literal-copied
from ferrolearn (R-CHAR-3).

### Probe 1 — `subcluster_centers_` + `labels_` VALUE on the blobs fixture (load-bearing)

Fixture `blobs8` = the `make_two_blobs()` test fixture (8×2: four points near origin,
four near `(10,10)`):
```
python3 -c "import numpy as np; from sklearn.cluster import Birch; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.05,10.05]]); \
b=Birch(n_clusters=2,threshold=0.5); print(b.fit_predict(X).tolist()); \
print(np.round(b.subcluster_centers_,6).tolist())"
# sklearn labels_:             [1, 1, 1, 1, 0, 0, 0, 0]
# sklearn subcluster_centers_: [[0.0375, 0.0375], [10.0375, 10.0375]]
```
ferrolearn `Birch::<f64>::new().with_threshold(0.5).with_n_clusters(2).fit(&blobs8)`:
- `subcluster_centers_` → `[[0.037500000000000006, 0.037500000000000006],
  [10.037500000000001, 10.037500000000001]]` — **identical to full f64 precision**
  (`0.0375 = (0+0.1+0+0.05)/4`).
- `labels_` → `[0,0,0,0,1,1,1,1]` — sklearn is `[1,1,1,1,0,0,0,0]`: the SAME partition
  up to a label swap. **Co-membership AGREES; absolute label values differ by a
  permutation** (Agglomerative-Ward label ordering, REQ-3).

### Probe 2 — docstring example (`n_clusters=None`)

The class docstring fixture (`_birch.py:474-480`):
```
python3 -c "import numpy as np; from sklearn.cluster import Birch; \
X=np.array([[0,1],[0.3,1],[-0.3,1],[0,-1],[0.3,-1],[-0.3,-1]]); \
brc=Birch(n_clusters=None); brc.fit(X); print(brc.predict(X).tolist()); \
print(np.round(brc.subcluster_centers_,6).tolist())"
# sklearn predict:             [0, 0, 0, 1, 1, 1]
# sklearn subcluster_centers_: [[0.0, 1.0], [0.0, -1.0]]
```
ferrolearn `Birch::<f64>::new().fit(&X)` (no `n_clusters`): `subcluster_centers_` →
`[[0.0, 1.0], [0.0, -1.0]]` — **identical**; `labels_` → `[0,0,0,1,1,1]`. (NB: sklearn
runs `predict` via `pairwise_distances_argmin` over the centroids, REQ-5; ferrolearn
maps each training point through the subcluster it was inserted into — equal HERE
because each point lands in its own blob, but NOT a general value match — see Probe 5.)

### Probe 3 — moderate make_blobs (still `<= branching_factor` subclusters): VALUE matches

`make_blobs(n_samples=20, centers=3, cluster_std=0.5, random_state=0)`,
`Birch(n_clusters=3, threshold=0.5)`:
```
# sklearn n_subclusters: 10 ; subcluster_centers_[:3] =
#   [[1.0571, 4.6702], [-1.8136, 2.6674], [1.4161, 4.1677]]
# sklearn labels_: [2,1,2,0,0,1,1,0,2,2,0,2,0,0,2,1,1,1,0,2]
```
ferrolearn (default `branching_factor=50` >= 10): `n_subclusters = 10`;
`subcluster_centers_[:3] = [[1.0570514…, 4.6702117…], [-1.8135845…, 2.6673765…],
[1.4160542…, 4.1676703…]]` — **matches sklearn's first three centroids to full f64
precision**; `labels_ = [0,1,0,2,2,1,1,2,0,0,2,0,2,2,0,1,1,1,2,0]` — the SAME
partition as sklearn up to the `0<->2` label swap. **AGREE** (value parity holds
because 10 <= 50; centroid order coincides because insertion order coincides).

### Probe 4 — branching_factor stress (`> branching_factor` subclusters): STRUCTURAL DIVERGENCE

`X = RandomState(1).rand(60,2)*20`, `Birch(n_clusters=None, threshold=1.0,
branching_factor=5)`:
```
# sklearn n_subclusters: 37   (a real CF-tree splits leaf nodes; #leaf subclusters >> branching_factor)
# sklearn subcluster_centers_[0]: [2.318, 1.31393]
```
ferrolearn same call: `n_subclusters = 5`; `subcluster_centers_[0] =
[15.9169…, 6.8190…]`. **DIVERGE** — ferrolearn `build_cf_tree` caps the flat list at
`branching_factor` and merges the two closest CFs whenever a new point would overflow
(the `else` branch with `find_closest_pair` + `merge`), so it can never exceed
`branching_factor` subclusters. sklearn's `_split_node` (`_birch.py:48-108`) splits a
full leaf into two leaves and grows the tree, so leaf-subcluster count is unbounded by
`branching_factor`. This is the root structural divergence: `subcluster_centers_` is a
value contract ONLY when sklearn produces `<= branching_factor` subclusters.

### Probe 5 — `predict` / `transform` / out-of-sample (absent in ferrolearn)

```
python3 -c "import numpy as np; from sklearn.cluster import Birch; \
X=np.array([[0,1],[0.3,1],[-0.3,1],[0,-1],[0.3,-1],[-0.3,-1]]); \
brc=Birch(n_clusters=None).fit(X); print(brc.transform(X)[0].tolist()); print(brc.predict(X).tolist())"
# transform(X)[0] = [0.0, 2.0]   (euclidean_distances(X, subcluster_centers_))
# predict(X)      = [0, 0, 0, 1, 1, 1]   (subcluster_labels_[argmin dist])
```
`FittedBirch<F>` exposes **no `predict` and no `transform`** — there is no
out-of-sample labelling (`pairwise_distances_argmin` over `subcluster_centers_`) and
no distance-to-centroid embedding. ferrolearn's training labels are produced by
threading each point through the subcluster it was *inserted into*, not by argmin over
final centroids, so they can diverge from sklearn's `_predict`-based `labels_` even on
the training set when a point's nearest final centroid differs from its insertion CF.

### Probe 6 — `not_enough_centroids` ConvergenceWarning + `subcluster_labels_`

```
python3 -c "import warnings,numpy as np; from sklearn.cluster import Birch; \
X=np.array([[0.,0.],[0.1,0.],[0.05,0.05]]); \
w=warnings.catch_warnings(record=True); w.__enter__(); warnings.simplefilter('always'); \
b=Birch(n_clusters=5,threshold=1.0).fit(X); \
print(b.subcluster_centers_.shape[0], b.subcluster_labels_.tolist(), b.labels_.tolist(), [str(x.message) for x in w.__exit__ or []])"
# 1 subcluster found, requested 5 -> ConvergenceWarning:
#   "Number of subclusters found (1) by BIRCH is less than (5). Decrease the threshold."
# subcluster_labels_ = [0] ; labels_ = [0,0,0]
```
sklearn detects `len(centroids) < n_clusters` (`_global_clustering`, `:716`), SKIPS the
global clustering step, sets `subcluster_labels_ = np.arange(len(centroids))`, and
emits `ConvergenceWarning`. ferrolearn instead clamps via
`let k_actual = k.min(n_subclusters)` (`fn fit`) and runs Agglomerative with the
clamped `k` — **no warning, and `n_clusters_` is silently the clamped value**, not the
requested one. Also ferrolearn exposes **no `subcluster_labels_`** accessor at all.

### Probe 7 — constructor / defaults / params

```
python3 -c "from sklearn.cluster import Birch; b=Birch(); \
print(b.n_clusters, b.threshold, b.branching_factor, b.compute_labels, b.copy)"
# 3 0.5 50 True True
```
- **`n_clusters` default**: sklearn `n_clusters=3` (`__init__`, `:496`); ferrolearn
  `fn new()` defaults `n_clusters = None` — **different default** (None = "skip global
  clustering" in sklearn). The ferrolearn-python `Birch` wrapper also passes
  `n_clusters=None` default (`_extras.py:441`), so `import ferrolearn; Birch()` does
  NOT mirror `sklearn …; Birch()` (3) — divergent default at the binding too.
- **`n_clusters` type**: sklearn accepts `None`, an `int >= 1`, OR a
  `sklearn.cluster` estimator instance (`_parameter_constraints`, `:486`; estimator
  branch `_global_clustering` `:731-735`). ferrolearn `Option<usize>` — no
  estimator-instance form.
- **`branching_factor`**: sklearn `Interval(Integral, 1, None, closed="neither")` =
  `>= 2` (`:485`); ferrolearn rejects `< 2` (`fn fit`) — **MATCH** on the bound.
- **`threshold`**: sklearn `Interval(Real, 0.0, None, closed="neither")` = `> 0`
  (`:484`); ferrolearn rejects `<= 0` (`fn fit`) — **MATCH** on the bound, but the
  error TYPE diverges (sklearn `InvalidParameterError`; ferrolearn
  `FerroError::InvalidParameter`).
- **missing params**: `compute_labels=True` (`:497`) and `copy=True` (`:498`) are
  absent from `Birch<F>`.

### Probe 8 — non-test consumer (the SHIPPED enabler)

`grep -rn "Birch" ferrolearn-python/src/` → `pub struct RsBirch` + `impl RsBirch`
(`extras.rs:996-1036`) registered via `m.add_class::<extras::RsBirch>()` (`lib.rs:68`)
and wrapped by `class Birch(_ClusterWrapper)` (`_extras.py:440-446`). `RsBirch::fit`
calls `ferrolearn_cluster::Birch::<f64>::new().with_threshold(...).with_n_clusters(...)`
then `.fit(...)`, and `RsBirch::labels_` returns `f.labels()`. **This is a genuine
non-test production consumer** of `Birch::fit` + `FittedBirch::labels` → the
core fit/labels REQs are SHIPPED. NB the binding exposes ONLY `n_clusters`,
`threshold`, `fit`, `labels_` — no `subcluster_centers_`, no `predict`/`transform`.

## Requirements

- REQ-1: **CF subcluster centroid VALUE — `subcluster_centers_` = `LS/N` (R-DEV-1).**
  Mirror the CF centroid `linear_sum_ / n_samples_` read off the leaves
  (`_CFSubcluster.update`, `_birch.py:319`; `subcluster_centers_` `:590-591`).
  ferrolearn `ClusteringFeature::centroid` (`ls[i]/n`) → `subcluster_centers()`
  value-matches sklearn to full f64 precision **when `n_subclusters <=
  branching_factor`** (Probes 1-3). Consumer: `RsBirch` (binding) + crate re-export.
- REQ-2: **CF-tree structure — leaf splitting / `_split_node` (R-DEV-1, the core
  structural divergence).** Mirror the balanced CF-tree: `_CFNode.insert_cf_subcluster`
  (`:196-263`) + `_split_node` (`:48-108`) — a full leaf splits into two and the tree
  grows, so leaf-subcluster count is unbounded by `branching_factor`. ferrolearn
  `build_cf_tree` keeps a FLAT `Vec` capped at `branching_factor` and MERGES the two
  closest CFs on overflow (`find_closest_pair` + `merge`) — diverges structurally once
  `> branching_factor` subclusters would form (Probe 4: 37 vs 5).
- REQ-3: **`labels_` partition VALUE via global Agglomerative-Ward (R-DEV-1/3).**
  Mirror `_global_clustering` (`:703-738`): when `n_clusters` is an int, fit
  `AgglomerativeClustering(n_clusters=n_clusters)` on `subcluster_centers_`, then
  `labels_ = subcluster_labels_[argmin]`. ferrolearn runs `AgglomerativeClustering`
  with `Linkage::Ward` (`fn fit`) and threads training points through their insertion
  CF. The PARTITION matches sklearn up to a label permutation on few-subcluster
  fixtures (Probes 1-3); absolute label VALUES differ (permutation) and the
  out-of-sample path differs (REQ-5).
- REQ-4: **`n_clusters` default `3` + estimator-instance form + error ABI (R-DEV-2).**
  sklearn `n_clusters=3` (`:496`), accepts `None`/`int>=1`/a `sklearn.cluster`
  estimator (`:486`, `:731-735`); raises `InvalidParameterError`. ferrolearn
  `fn new()` defaults `None`, `Option<usize>` (no estimator form), errors with
  `FerroError::InvalidParameter`.
- REQ-5: **`predict` + `transform` out-of-sample (R-DEV-3).** Mirror
  `predict`/`_predict` (`pairwise_distances_argmin(X, subcluster_centers_)` →
  `subcluster_labels_[argmin]`, `:651-679`) and `transform`
  (`euclidean_distances(X, subcluster_centers_)`, `:681-701`). `FittedBirch<F>` has
  NO `predict`/`transform` — no out-of-sample labels, no distance embedding (Probe 5).
- REQ-6: **`subcluster_labels_` fitted attribute (R-DEV-3).** Mirror
  `subcluster_labels_` (the per-subcluster global label, `:723`/`:735`). `FittedBirch`
  exposes `labels()`, `subcluster_centers()`, `n_clusters()` but **no
  `subcluster_labels_`** accessor (Probe 6, Probe 8).
- REQ-7: **`not_enough_centroids` → skip global step + ConvergenceWarning (R-DEV-1).**
  sklearn, when `len(centroids) < n_clusters`, SKIPS global clustering, sets
  `subcluster_labels_ = arange`, warns `ConvergenceWarning` (`:716`/`:722-730`).
  ferrolearn instead CLAMPS `k_actual = k.min(n_subclusters)` and runs Agglomerative
  with the clamped k — no warning, `n_clusters_` silently clamped (Probe 6).
- REQ-8: **`partial_fit` online learning (R-DEV-2).** Mirror `partial_fit`
  (`:613-638`): incremental insertion without rebuilding, `X=None` runs only the
  global step. ferrolearn has no `partial_fit` — only batch `fit`.
- REQ-9: **`compute_labels` + `copy` parameters (R-DEV-2).** sklearn
  `compute_labels=True` (`:497`, gates `labels_` computation `:709`/`:737`) and
  `copy=True` (`:498`, gates in-place overwrite `:534`). Both absent from `Birch<F>`.
- REQ-10: **threshold criterion form (R-DEV-1, equivalence note).** sklearn merges
  iff `sq_radius = new_ss/new_n - ||new_centroid||^2 <= threshold**2`
  (`_CFSubcluster.merge_subcluster`, `:340-342`). ferrolearn
  `would_exceed_threshold` computes the same `variance = new_ss/n - centroid_sq_norm`,
  takes `sqrt`, and compares `radius > threshold` — algebraically EQUIVALENT
  (`r > t  <=>  r^2 > t^2` for `r,t >= 0`), modulo the strict/non-strict boundary
  (sklearn merges on `<=`, ferrolearn splits on `>` → both keep equality as "merge").
- REQ-11: **ferray substrate (R-SUBSTRATE).** `birch.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core` /
  `ferray::linalg`.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). ferrolearn values from a throwaway
`cargo run --example` probe (since deleted). Fixtures: `blobs8` (the `make_two_blobs()`
test fixture, 8×2), `docstring6` (the class-docstring 6×2 fixture), `blobs20` =
`make_blobs(n_samples=20, centers=3, cluster_std=0.5, random_state=0)`,
`spread60` = `RandomState(1).rand(60,2)*20`.

- AC-1 (REQ-1, SHIPPED): `Birch(n_clusters=2,threshold=0.5).fit(blobs8).subcluster_centers_
  == [[0.0375,0.0375],[10.0375,10.0375]]`; ferrolearn `subcluster_centers()` identical
  to full f64. On `blobs20`, sklearn `subcluster_centers_[:3] = [[1.0571,4.6702],
  [-1.8136,2.6674],[1.4161,4.1677]]`; ferrolearn matches to full f64.
- AC-2 (REQ-3, SHIPPED up to permutation): `Birch(n_clusters=3,threshold=0.5).fit_predict(blobs20)`
  → sklearn `[2,1,2,0,0,...]`; ferrolearn `[0,1,0,2,2,...]` — SAME partition, `0<->2`
  relabel. Co-membership matrix identical; absolute labels differ by permutation.
- AC-3 (REQ-2, DIVERGES): `Birch(n_clusters=None,threshold=1.0,branching_factor=5).fit(spread60).subcluster_centers_.shape[0]`
  → sklearn `37`; ferrolearn `5` (capped at `branching_factor`).
- AC-4 (REQ-4): `Birch().n_clusters == 3` in sklearn; ferrolearn `Birch::new().n_clusters
  == None`, and `import ferrolearn; Birch().n_clusters is None` (binding default also 3-divergent).
- AC-5 (REQ-5): `Birch(n_clusters=None).fit(docstring6).transform(docstring6)[0] == [0.,2.]`
  and `.predict(docstring6) == [0,0,0,1,1,1]`; ferrolearn `FittedBirch` has no
  `transform`/`predict` method.
- AC-6 (REQ-7): `Birch(n_clusters=5,threshold=1.0).fit([[0,0],[0.1,0],[0.05,0.05]])`
  → sklearn emits `ConvergenceWarning("Number of subclusters found (1) ... less than
  (5)")`, `subcluster_labels_=[0]`; ferrolearn clamps to `n_clusters_=1` silently.

## REQ status table

Binary (R-DEFER-2). `Birch` / `FittedBirch` are existing pub APIs re-exported at the
crate root with a genuine non-test consumer — `_RsBirch` in `ferrolearn-python`
(Probe 8; grandfathered S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) /
`file:line` (sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2,
run from `/tmp`. Honest underclaim (R-HONEST-3): the mean-based `subcluster_centers_`
and the `labels_` PARTITION value-match the oracle and have a real consumer — but ONLY
in the `n_subclusters <= branching_factor` regime; the CF-tree structure itself
(REQ-2) diverges, and `predict`/`transform`/`partial_fit`/`subcluster_labels_`/
`compute_labels`/`copy` are absent. Suggested blocker numbers — the director creates
the real issues.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`subcluster_centers_` VALUE) | SHIPPED | impl `fn centroid in birch.rs` (CF `ls[i]/n`) → `fn fit` builds `subcluster_centers_`; value-matches sklearn `_CFSubcluster.update` centroid (`_birch.py:319`) read off leaves (`:590-591`) to full f64 (AC-1: `blobs8` `[[0.0375,…],[10.0375,…]]`; `blobs20` first 3 centroids exact). Non-test consumer: `pub struct RsBirch` (`extras.rs`, via `Birch::fit`) + crate re-export. Verification: `cargo test -p ferrolearn-cluster --lib birch` (15 passed) + Probe 1/3. **Caveat (REQ-2): contract holds only when sklearn yields `<= branching_factor` subclusters** — diverges otherwise (AC-3). |
| REQ-2 (CF-tree leaf splitting) | NOT-STARTED | open prereq blocker **#940**. sklearn `_CFNode.insert_cf_subcluster` (`:196-263`) recurses into child nodes and `_split_node` (`:48-108`) splits a full leaf into two leaves so leaf-subcluster count is unbounded by `branching_factor`. ferrolearn `fn build_cf_tree` keeps a FLAT `Vec<ClusteringFeature>` capped at `branching_factor`, MERGING the two closest CFs on overflow (`fn find_closest_pair` + `fn merge`). Pin (AC-3): `spread60`, `branching_factor=5` → sklearn 37 subclusters vs ferrolearn 5. **Root structural divergence — pin FIRST.** |
| REQ-3 (`labels_` partition via global Agglo-Ward) | SHIPPED | impl `fn fit in birch.rs` runs `AgglomerativeClustering::new(k).with_linkage(Linkage::Ward)` on `subcluster_centers_`, mirroring `_global_clustering` int branch (`_birch.py:713-735`, default Agglomerative = Ward). PARTITION value-matches sklearn up to a label permutation on few-subcluster fixtures (AC-2: `blobs20` sklearn `[2,1,2,0,0,…]` vs ferrolearn `[0,1,0,2,2,…]`, same partition). Non-test consumer: `RsBirch::labels_` (`extras.rs`). Verification: `cargo test -p ferrolearn-cluster --lib birch` + Probe 1/3. **Caveat**: absolute label values differ by permutation (Agglo label ordering); gated on REQ-2 in the `> branching_factor` regime and on REQ-5 for out-of-sample. |
| REQ-4 (`n_clusters=3` default + estimator form + error ABI) | NOT-STARTED | open prereq blocker **#941**. sklearn `n_clusters=3` default (`:496`), accepts `None`/`int>=1`/`sklearn.cluster` estimator (`:486`, `:731-735`), raises `InvalidParameterError`. ferrolearn `fn new()` defaults `None`, `Option<usize>` (no estimator-instance form), `FerroError::InvalidParameter` ABI. AC-4: `Birch().n_clusters` = 3 (sklearn) vs `None` (ferro + binding). |
| REQ-5 (`predict` + `transform` out-of-sample) | NOT-STARTED | open prereq blocker **#942**. sklearn `predict`/`_predict` = `pairwise_distances_argmin(X, subcluster_centers_)` → `subcluster_labels_[argmin]` (`:651-679`); `transform` = `euclidean_distances(X, subcluster_centers_)` (`:681-701`). `FittedBirch<F>` has neither — no out-of-sample labelling, no distance embedding (AC-5: `transform(docstring6)[0]=[0,2]`, `predict=[0,0,0,1,1,1]`). Also: ferrolearn `labels_` threads points through their insertion CF, NOT argmin over final centroids, so even training labels can diverge from sklearn's `_predict`-based `labels_`. |
| REQ-6 (`subcluster_labels_` attribute) | NOT-STARTED | open prereq blocker **#943**. sklearn `subcluster_labels_` = global label per subcluster (`:723`/`:735`), the bridge `predict` uses. `FittedBirch` exposes `labels()`/`subcluster_centers()`/`n_clusters()` but no `subcluster_labels_` accessor; `RsBirch` cannot surface it. |
| REQ-7 (`not_enough_centroids` skip + ConvergenceWarning) | NOT-STARTED | open prereq blocker **#944**. sklearn, when `len(centroids) < n_clusters`, SKIPS the global step, `subcluster_labels_ = arange(len)`, warns `ConvergenceWarning` (`:716`/`:722-730`). ferrolearn `fn fit` CLAMPS `k.min(n_subclusters)` and runs Agglomerative on the clamped k — no warning, `n_clusters_` silently clamped (AC-6). |
| REQ-8 (`partial_fit` online learning) | NOT-STARTED | open prereq blocker **#945**. sklearn `partial_fit` (`:613-638`) incrementally inserts without rebuilding (`X=None` → global step only). ferrolearn offers only batch `Fit::fit`; no incremental API. |
| REQ-9 (`compute_labels` + `copy` params) | NOT-STARTED | open prereq blocker **#946**. sklearn `compute_labels=True` (`:497`, gates `labels_` `:709`/`:737`) + `copy=True` (`:498`, gates in-place `:534`). `Birch<F>` (`threshold`/`branching_factor`/`n_clusters`) has neither (Probe 7). |
| REQ-10 (threshold criterion form) | SHIPPED | impl `fn would_exceed_threshold in birch.rs` computes `variance = new_ss/n - centroid_sq_norm`, `radius = sqrt(max(0,variance))`, tests `radius > threshold` — algebraically EQUIVALENT to sklearn `sq_radius <= threshold**2` (`_CFSubcluster.merge_subcluster`, `_birch.py:340-342`; `r>t <=> r^2>t^2` for non-negative) including the negative-variance clamp (sklearn `sqrt(max(0,...))` `:357-358`). Consumer: `fn build_cf_tree` (in-crate) → `fn fit` → `RsBirch`. Verification: `cargo test -p ferrolearn-cluster --lib birch` (`test_threshold_effect_on_subclusters`). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#947**. `birch.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`; not migrated to `ferray-core` / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`birch.rs` follows the unfitted/fitted split (CLAUDE.md naming): `Birch<F>`
(`threshold: F`, `branching_factor: usize`, `n_clusters: Option<usize>`) →
`Fit<Array2<F>, ()>` → `FittedBirch<F>` (private `labels_: Array1<usize>`,
`subcluster_centers_: Array2<F>`, `n_clusters_: usize`). Generic over
`F: Float + Send + Sync + 'static`; every public method returns `Result<_,
FerroError>` (R-CODE-2). No `Predict`/`Transform` trait impl (REQ-5).

**The CF abstraction (`struct ClusteringFeature`).** Holds `n` (count), `ls` (linear
sum `Vec<F>`), `ss` (sum of squared norms), and `point_indices` (original indices —
ferrolearn-specific, used to thread labels back; sklearn does NOT retain point indices,
it re-derives `labels_` via `_predict`, REQ-5). `fn centroid` = `ls[i]/n` (mirrors
`_CFSubcluster.update`, `_birch.py:319`, REQ-1). `fn would_exceed_threshold` is the
radius criterion (REQ-10). `fn merge`/`fn absorb_point` mirror `_CFSubcluster.update`'s
running sums. **There is no `_CFNode` analog** — no tree, no leaf linked-list
(`prev_leaf_`/`next_leaf_`), no `dummy_leaf_`/`root_` (sklearn `_birch.py:111-263`,
`:546-563`).

**Fit path (`fn fit`).** Validates `threshold > 0`, `branching_factor >= 2`,
`n_clusters != Some(0)` (REQ-4/10), handles the empty-input shortcut, then
`build_cf_tree` produces the flat subcluster list. The global step: if
`n_clusters = Some(k)`, fit `AgglomerativeClustering::new(k.min(n_subclusters))`
(Ward) on the centroids and map each point to its subcluster's global label; else each
subcluster is its own cluster (REQ-3). **The `k.min(n_subclusters)` clamp is the REQ-7
divergence** — sklearn warns + skips instead.

**`build_cf_tree` — the structural divergence (REQ-2).** A flat `Vec`: each new point
is absorbed into the nearest CF if it would not exceed `threshold`; else a new CF is
created if `len < branching_factor`; else the two closest CFs are MERGED and the point
starts a new CF — capping the list at `branching_factor`. sklearn instead splits a
full leaf node (`_split_node`, `:48-108`) and grows the balanced tree, so leaf count
is unbounded (Probe 4). The flat list ALSO uses a linear nearest-CF scan
(`distance_to_point` over centroids), where sklearn descends the tree via
`insert_cf_subcluster`'s `np.dot`-based argmin (`:206-209`) — same nearest-CF intent,
O(n·k) vs tree-descent, but the OBSERVABLE divergence is the subcluster count, REQ-2.

**Invariants held vs sklearn:** `subcluster_centers_` VALUE (`LS/N`, REQ-1, AC-1 — in
the `<= branching_factor` regime); `labels_` PARTITION up to permutation (REQ-3, AC-2);
the threshold criterion (REQ-10); the `threshold>0` / `branching_factor>=2` bounds
(Probe 7); `labels()` length = `n_samples`; empty/single-sample edge cases (tests).

**Invariants NOT held vs sklearn:** the CF-tree structure / subcluster count when
`> branching_factor` (REQ-2 — 37 vs 5, AC-3); `n_clusters=3` default + estimator form +
error ABI (REQ-4, AC-4); `predict`/`transform` (REQ-5, AC-5); `subcluster_labels_`
(REQ-6); `not_enough_centroids` warning + skip (REQ-7, AC-6); `partial_fit` (REQ-8);
`compute_labels`/`copy` (REQ-9); the ferray substrate (REQ-11).

**Consumer wiring.** Non-test consumers: the crate re-export (`pub use
birch::{Birch, FittedBirch}`, `ferrolearn-cluster/src/lib.rs`) AND `pub struct RsBirch`
(`ferrolearn-python/src/extras.rs`) — `RsBirch::fit` builds `Birch::<f64>` and calls
`Birch::fit`; `RsBirch::labels_` returns `FittedBirch::labels`; registered via
`m.add_class::<extras::RsBirch>()` (`ferrolearn-python/src/lib.rs`) and wrapped by
`class Birch(_ClusterWrapper)` (`ferrolearn-python/python/ferrolearn/_extras.py`). The
binding surfaces ONLY `n_clusters`/`threshold`/`fit`/`labels_` — not
`subcluster_centers_`, `predict`, or `transform`.

## Verification

Library crate (green at baseline `beb6862e`):
```
cargo test -p ferrolearn-cluster --lib birch     # 15 passed; 0 failed
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The 15 in-tree `#[test]`s (`test_two_clusters_with_n_clusters`, `test_three_clusters`,
`test_subclusters_without_n_clusters`, `test_threshold_effect_on_subclusters`,
`test_subcluster_centers_shape`, `test_empty_data`, `test_single_sample`,
`test_invalid_threshold`, `test_invalid_branching_factor`, `test_invalid_n_clusters`,
`test_f32_support`, `test_labels_in_valid_range`, `test_identical_points`,
`test_default_constructor`, `test_single_cluster`) pin co-membership, shape, range,
default-constructor, and error edges. **None compares `subcluster_centers_` or
`labels_` VALUE against the live sklearn `Birch` oracle**, and none exercises the
`> branching_factor` split path — so they stay green despite REQ-2.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin REQ-2 (CF-tree split) FIRST** — it is the
root structural divergence and the regime boundary for the SHIPPED REQ-1/REQ-3:
```
# REQ-1 (VALUE matches in regime) subcluster_centers_
python3 -c "import numpy as np; from sklearn.cluster import Birch; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.05,10.05]]); \
b=Birch(n_clusters=2,threshold=0.5).fit(X); print(np.round(b.subcluster_centers_,6).tolist())"
# [[0.0375,0.0375],[10.0375,10.0375]]  (ferro: identical to full f64)
# REQ-2 (DIVERGES) subcluster count past branching_factor
python3 -c "import numpy as np; from sklearn.cluster import Birch; \
X=np.random.RandomState(1).rand(60,2)*20; \
b=Birch(n_clusters=None,threshold=1.0,branching_factor=5).fit(X); print(b.subcluster_centers_.shape[0])"
# 37  (ferro: 5 — capped at branching_factor)
# REQ-5 (ABSENT) predict / transform
python3 -c "import numpy as np; from sklearn.cluster import Birch; \
X=np.array([[0,1],[0.3,1],[-0.3,1],[0,-1],[0.3,-1],[-0.3,-1]]); \
b=Birch(n_clusters=None).fit(X); print(b.transform(X)[0].tolist(), b.predict(X).tolist())"
# [0.0,2.0] [0,0,0,1,1,1]  (ferro: no transform/predict)
# REQ-7 (DIVERGES) not_enough_centroids ConvergenceWarning
python3 -W all -c "import numpy as np; from sklearn.cluster import Birch; \
Birch(n_clusters=5,threshold=1.0).fit(np.array([[0.,0.],[0.1,0.],[0.05,0.05]]))"
# ConvergenceWarning: Number of subclusters found (1) by BIRCH is less than (5). Decrease the threshold.
# (ferro: silently clamps n_clusters_=1, no warning)
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-cluster/tests/divergence_birch.rs`, asserting the live-sklearn expected
values above and FAILING against current `birch.rs`. The SHIPPED REQ-1/REQ-3 should
gain a positive value-parity pin (`subcluster_centers_` exact on `blobs8`; `labels_`
partition on `blobs20`) explicitly scoped to the `<= branching_factor` regime, plus a
negative pin for the regime boundary (REQ-2, AC-3) so a future tree fix is detectable.

ferrolearn-python (binding parity — `_RsBirch` already exists, Probe 8):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_birch.py -q
```
asserting `ferrolearn.Birch(n_clusters=2,threshold=0.5).fit(blobs8).labels_` matches
`sklearn.cluster.Birch(...)` co-membership, and pinning the `Birch().n_clusters`
default divergence (3 in sklearn, None in the ferrolearn wrapper, REQ-4/AC-4).

## Blockers to open

(Director creates the real issues; suggested numbers below.)

- **#940** — REQ-2: replace the flat `Vec<ClusteringFeature>` + merge-on-overflow
  (`build_cf_tree`) with a balanced CF-tree — `_CFNode` (subclusters + leaf
  linked-list) + recursive `insert_cf_subcluster` + `_split_node` leaf splitting
  (`_birch.py:48-263`), so leaf-subcluster count grows past `branching_factor`. **The
  root structural divergence and the regime boundary for SHIPPED REQ-1/REQ-3 — pin
  FIRST.**
- **#941** — REQ-4: `n_clusters=3` default; accept the estimator-instance form
  (`sklearn.cluster` model) in addition to `None`/`int`; align the error ABI to the
  sklearn `ValueError`/`InvalidParameterError` contract (`_birch.py:486`, `:731-735`).
  Fix the ferrolearn-python wrapper default (`_extras.py:441`) to `3` as well.
- **#942** — REQ-5: add `predict` (`pairwise_distances_argmin` over
  `subcluster_centers_` → `subcluster_labels_[argmin]`) and `transform`
  (`euclidean_distances(X, subcluster_centers_)`) to `FittedBirch`, and re-derive
  training `labels_` via `_predict` rather than insertion-CF threading
  (`_birch.py:651-738`).
- **#943** — REQ-6: store + expose `subcluster_labels_` on `FittedBirch` (`:723`/`:735`).
- **#944** — REQ-7: detect `n_subclusters < n_clusters` → skip the global step, set
  `subcluster_labels_ = arange`, emit a `ConvergenceWarning` analog instead of
  clamping `k` (`_birch.py:716`/`:722-730`).
- **#945** — REQ-8: add `partial_fit` incremental insertion (depends on #940's tree)
  (`_birch.py:613-638`).
- **#946** — REQ-9: add `compute_labels` (gate `labels_`) and `copy` (gate in-place)
  parameters (`_birch.py:497-498`, `:534`, `:709`/`:737`).
- **#947** — REQ-11: migrate `birch.rs` off `ndarray`/`num-traits` to `ferray-core` /
  `ferray::linalg` (R-SUBSTRATE).
