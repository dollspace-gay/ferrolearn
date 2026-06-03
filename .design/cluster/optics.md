# OPTICS (sklearn.cluster.OPTICS)

<!--
tier: 3-component
status: draft
baseline-commit: beb6862e
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_optics.py   # class OPTICS(ClusterMixin, BaseEstimator) (:36-393); _parameter_constraints (:242-264); __init__ (:266-297); fit (:303-393); compute_optics_graph (:458-668); _set_reach_dist (:671-714); _compute_core_distances_ (:405-438); cluster_optics_dbscan (:726-788); cluster_optics_xi (:810-918); _extend_region (:921-981); _update_filter_sdas (:984-995); _correct_predecessor (:998-1017); _xi_cluster (:1020-1172); _extract_xi_labels (:1175-1201); _validate_size (:396-401)
ferrolearn-module: ferrolearn-cluster/src/optics.rs
parity-ops: OPTICS (.__init__, .fit, .fit_predict, .labels_, .reachability_, .ordering_, .core_distances_, .predecessor_, .cluster_hierarchy_), cluster_optics_xi, cluster_optics_dbscan
crosslink-issue: 1079
-->

## Summary

`ferrolearn-cluster/src/optics.rs` mirrors scikit-learn's `OPTICS`
(`sklearn/cluster/_optics.py`, `class OPTICS(ClusterMixin, BaseEstimator)`
`:36-393`) — Ordering Points To Identify the Clustering Structure, the density
ordering relative of DBSCAN. It exposes the unfitted `OPTICS<F>` (`min_samples`
required; `max_eps`, `xi`, `min_cluster_size` builders), the fitted
`FittedOPTICS<F>` (accessors `ordering()`, `reachability()`, `core_distances()`,
`labels()`, `predecessors()`, `n_clusters()`, plus `extract_clusters(xi)`), and a
`fit_predict` convenience mirroring `ClusterMixin.fit_predict`. It is re-exported
at the crate root (`pub use optics::{FittedOPTICS, OPTICS}` in
`ferrolearn-cluster/src/lib.rs`).

**The reachability-graph math VALUE-matches sklearn on clean/small fixtures and
the per-point `core_distances_` value-matches even on hard fixtures — but the
`ordering_`, `reachability_`, and `labels_` DIVERGE from the live sklearn oracle
on the canonical noisy-blob case (probes below), because the traversal uses a
different seed-selection rule (BinaryHeap vs sklearn's linear-argmin, REQ-2 #1080 —
the prime fixable divergence) and the Xi extraction drops the
`min_cluster_size`/`cluster_hierarchy_` semantics.** Honest assessment (R-HONEST-3):
`core_distances_` value-matches the oracle even on hard fixtures (REQ-1 SHIPPED,
green-guarded); `ordering_` / `reachability_` / `predecessor_` / `labels_` match on
clean/small fixtures (Probe 0) but diverge on noisy data until the REQ-2 traversal
fix lands. `OPTICS` / `FittedOPTICS` are existing pub
APIs (grandfathered per S5/R-DEFER-1); their **only** consumer is the crate
re-export — there is **no `ferrolearn-python` binding**
(`grep -rln OPTICS ferrolearn-python/` is empty, Probe 6) and no other in-crate
consumer.

## Live oracle probes (sklearn 1.5.2, run from /tmp; ferrolearn via a throwaway `cargo run --example` probe, since deleted)

Expected values are from the live installed sklearn 1.5.2 oracle, never
literal-copied from ferrolearn (R-CHAR-3). The `.npy` blob fixture is
`RandomState(0)`: three `0.3·randn(20,2)` Gaussian blobs centred at `(0,0)`,
`(5,5)`, `(10,0)` (60 points).

### Probe 0 — the agreeing small fixtures (where the graph VALUE-matches)

**`three_blobs()`** (the in-tree test fixture, 9×2, `min_samples=2`):
```
python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
m=OPTICS(min_samples=2).fit(X); print(m.ordering_.tolist()); print(m.labels_.tolist()); print(m.predecessor_.tolist())"
# ordering_    [0, 1, 2, 3, 4, 5, 8, 6, 7]
# labels_      [0, 0, 0, 1, 1, 1, 2, 2, 2]
# predecessor_ [-1, 0, 0, 1, 3, 3, 8, 6, 4]
```
ferrolearn `OPTICS::<f64>::new(2).fit(&three_blobs())` → ordering `[0,1,2,3,4,5,8,6,7]`,
labels `[0,0,0,1,1,1,2,2,2]`, predecessor `[-1,0,0,1,3,3,8,6,4]`, reachability/core
identical to full precision — **AGREE** on all four attributes.

**Docstring fixture** (`[[1,2],[2,5],[3,6],[8,7],[8,8],[7,3]]`, `min_samples=2`,
the worked example at `_optics.py:230-236`):
```
# sklearn ordering_ [0, 1, 2, 5, 3, 4]; reachability_ [inf,3.1623,1.4142,4.1231,1.0,5.0]
#         predecessor_ [-1, 0, 1, 5, 3, 2]; labels_ [0, 0, 0, 1, 1, 1]
```
ferrolearn → ordering `[0,1,2,5,3,4]`, reachability `[inf,3.1623,1.4142,4.1231,1.0,5.0]`,
predecessor `[-1,0,1,5,3,2]`, labels `[0,0,0,1,1,1]` — **AGREE** (matches the
sklearn docstring `compute_optics_graph` output `:585-594`).

### Probe 1 — `core_distances_` VALUE parity (the one attribute that matches on hard fixtures)

On the 60-point blob fixture, `min_samples=5`:
```
python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
X=np.load('/tmp/optics_blobs.npy'); m=OPTICS(min_samples=5).fit(X); \
print([round(float(c),4) for c in m.core_distances_[:5]])"
# sklearn core_distances_[:5] = [0.3123, 0.5942, 0.3703, 0.2103, 0.1848]
```
ferrolearn `core_distances()[..5]` = `[0.3123, 0.5942, 0.3703, 0.2103, 0.1848]` —
**AGREE** to 4 dp. The k-NN core distance (`fn core_distance`, distance to the
`(min_samples-1)`-th other point) value-matches sklearn's
`_compute_core_distances_` = `kneighbors(X, min_samples)[0][:, -1]`
(`_optics.py:437`) on this fixture. **But this is the only attribute that
value-matches here** — ordering/reachability/labels diverge (Probes 2–3).

### Probe 2 — `ordering_` / `reachability_` DIVERGE (the core traversal divergence)

Same 60-point fixture, `min_samples=5`:
```
# sklearn ordering_ = [0, 3, 6, 7, 4, 13, 15, 17, 19, 8, 11, 9, 2, 18, 5, 14, 1, 12, 16, 10, 31, 20, 22, ...]
# ferro   ordering_ = [0, 8, 3, 6, 11, 7, 15, 4, 17, 19, 13, 9, 2, 18, 5, 14, 1, 12, 16, 10, 31, 20, 26, ...]
```
The two orderings **diverge from position 1 onward** (`3` vs `8`, then `6,7,4`
vs `3,6,11`). Because `reachability_` is reported in object order but reflects the
traversal, the in-order reachability plots differ too (e.g. ferro position 2 has
`0.248` where sklearn has `0.21`). **Root cause** — the next-point selection rule:
- sklearn picks the next seed by a **linear `argmin` over ALL unprocessed points**
  including those still at `inf` reachability, with **smallest-index tie-break**:
  `point = index[np.argmin(reachability_[index])]` (`compute_optics_graph`,
  `_optics.py:641-642`; the comment `:639-640` "prefer smaller ids on ties,
  possibly np.inf!"). One global pool — there is no separate "restart at the next
  unprocessed point" loop.
- ferrolearn uses a **`BinaryHeap` min-heap of seed entries** with a stale-entry
  skip (`SeedEntry`, `fn update_seeds`, the `while let Some(entry) = seeds.pop()`
  loop in `fn fit`), and restarts components with an outer `for start in
  0..n_samples` loop that appends the next *unseeded* point. On ties and at
  component boundaries the heap order and the "smallest unprocessed index" order
  differ, so the orderings diverge once a point is reachable from two seeds at
  equal distance. (R-DEV-4 does **not** license this: the heap is an
  optimization, but it changes the observable `ordering_`/`reachability_`
  tie-break contract — sklearn's `:53` comment explicitly says it does *not* use
  a heap, precisely to keep this ordering.)

### Probe 3 — `labels_` DIVERGE (Xi extraction + min_cluster_size + cluster_hierarchy_)

Same 60-point fixture, `min_samples=5` (default `min_cluster_size=None`→`min_samples`,
`xi=0.05`):
```
# sklearn labels_: 3 clusters with 11 NOISE points (-1):
#   [0]*20 + [-1,-1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,1] + [2]*20
#   n_noise = 11; cluster_hierarchy_ = [[0,19],[22,30],[20,39],[40,59],[0,59]]
# ferro labels_:  3 clusters, 0 NOISE points:
#   [0]*20 + [1]*20 + [2]*20 ;  n_noise = 0
```
ferrolearn labels **all 60 points** into three contiguous clusters with zero
noise; sklearn carves 11 noise points out of the middle (loosest) blob and
exposes a 5-row `cluster_hierarchy_`. Divergence has three compounding causes:
1. **Different `ordering_`/`reachability_` plot fed to Xi** (Probe 2) → different
   steep areas.
2. **No `cluster_hierarchy_`** — sklearn's `_xi_cluster` returns `(n_clusters, 2)`
   `[start,end]` intervals ordered by `(end, -start)` (`_optics.py:1166-1172`,
   docstring `:191-200`); `_extract_xi_labels` then assigns leaf labels
   (`:1194-1200`). ferrolearn's `fn xi_cluster_extraction` collects intervals but
   **discards them** — `FittedOPTICS` has no `cluster_hierarchy_` accessor, so the
   hierarchy contract (REQ-8) is absent.
3. **`min_cluster_size` semantics differ** — sklearn applies `min_cluster_size`
   *inside* Xi as criterion 3.a (`c_end - c_start + 1 < min_cluster_size →
   continue`, `_optics.py:1155-1156`), so a too-small steep interval is never a
   cluster. ferrolearn ignores size *inside* Xi (`fn xi_cluster_extraction`'s
   `if c_end < c_start + 1` only drops length-1 intervals) and instead does a
   **post-hoc relabel-then-renumber pass** in `fn filter_small_clusters`
   (`optics.rs:706-750` — the collapsible-if site at `:717/:718/:744`): it counts
   final label sizes, relabels sub-`min_size` clusters to `-1`, then renumbers
   contiguously. This is a *different algorithm* with a different fixed point — on
   this fixture it yields 3 full clusters / 0 noise vs sklearn's 11 noise.

### Probe 4 — `cluster_method='dbscan'` extraction is ABSENT

sklearn exposes `cluster_optics_dbscan(*, reachability, core_distances, ordering,
eps)` (`_optics.py:726-788`) and `OPTICS(cluster_method='dbscan', eps=…)`
(`fit` branch `:374-390`):
```
python3 -c "import numpy as np; from sklearn.cluster import compute_optics_graph, cluster_optics_dbscan; \
X=np.load('/tmp/optics_blobs.npy'); o,cd,r,p=compute_optics_graph(X,min_samples=5,max_eps=np.inf,metric='minkowski',p=2,metric_params=None,algorithm='auto',leaf_size=30,n_jobs=None); \
print(cluster_optics_dbscan(reachability=r,core_distances=cd,ordering=o,eps=2.0).tolist()[:6])"
# [0, 0, 0, 0, 0, 0]  (linear-time cumsum extraction, _optics.py:781-787)
```
ferrolearn has **no `cluster_method` parameter, no `eps` parameter, and no
`cluster_optics_dbscan` free function** — only the Xi method is wired into `fit`.

### Probe 5 — defaults / parameter-surface divergences

- **`min_samples` default**: sklearn `min_samples=5` (`_optics.py:269`); ferrolearn
  `fn new(min_samples: usize)` has **no default** — the caller must supply it.
- **`min_samples` float fraction**: sklearn accepts `int>1` *or* `float∈(0,1)`
  (`_parameter_constraints` `:243-246`), normalised by `min_samples = max(2,
  int(min_samples*n))` when `<=1` (`compute_optics_graph` `:598-599`). probe:
  `OPTICS(min_samples=0.1).fit(X)` on 60 pts → effective `6`. ferrolearn
  `min_samples: usize` cannot express a fraction.
- **`metric` / `p` / `metric_params`**: sklearn default `metric='minkowski', p=2`
  (Euclidean) with the full scikit-learn/scipy metric set (`:248-249`,
  `_set_reach_dist` `:702-708`). ferrolearn hard-codes Euclidean (`fn euclidean`)
  — numerically equal to the *default* metric, but no `metric`/`p`/`metric_params`
  params and no `precomputed`/`cityblock`/`cosine`/`minkowski(p≠2)` support.
- **`max_eps`**: sklearn default `np.inf` (`:270`); ferrolearn default
  `F::infinity()` (`fn new`) — **matches**. Validation: sklearn `[0,inf]`
  closed-both (`:247`, accepts `0`); ferrolearn rejects `max_eps <= 0`
  (`fn fit` guard) — **over-rejects `max_eps=0`** and uses
  `FerroError::InvalidParameter`, not the sklearn `InvalidParameterError`/`ValueError`
  ABI.
- **`xi`**: sklearn default `0.05`, range `[0,1]` closed-both (`:253`); ferrolearn
  default `0.05` (`fn new`) but rejects `xi <= 0 || xi >= 1` (`fn fit` /
  `extract_clusters`) — **over-rejects the closed endpoints `0` and `1`**.
- **`predecessor_correction`**: sklearn default `True`, toggleable (`:277`);
  ferrolearn **always applies** predecessor correction (`fn correct_predecessor`
  is unconditional in `fn xi_cluster_extraction`) — no toggle.
- **`min_cluster_size`**: sklearn `None` default, `int>1` or `float∈(0,1]`
  (`:255-259`); ferrolearn `Option<usize>` (`with_min_cluster_size`) — no float
  fraction, and applied via the divergent post-hoc filter (Probe 3).
- **`cluster_method` / `eps`**: sklearn `'xi'`(default)/`'dbscan'` + `eps`
  (`:274-275`); ferrolearn Xi only (Probe 4).
- **`algorithm` / `leaf_size` / `memory` / `n_jobs`**: sklearn NN-backend + caching
  + parallelism knobs (`:279-282`). ferrolearn is brute-force O(n²) only
  (`fn get_neighbors`, `fn core_distance`) — numerically equal to sklearn's
  `algorithm='brute'`, but these params are absent.

### Probe 6 — non-test consumer

`grep -rln "OPTICS\|optics" ferrolearn-python/` is **empty** — there is **no
PyO3 binding**, so `import ferrolearn` cannot reach `OPTICS`.
`grep -rn OPTICS ferrolearn-cluster/src/` outside `optics.rs` finds only the crate
re-export and doc-comment mentions (`lib.rs:18/54/92/114`). The only public entry
points are `fit` / `fit_predict` / the `FittedOPTICS` accessors, and their sole
non-test consumer is the crate re-export (`pub use optics::{FittedOPTICS, OPTICS}`).

## Requirements

- REQ-1: **`core_distances_` VALUE (R-DEV-1).** Mirror `_compute_core_distances_`
  = `kneighbors(X, min_samples)[0][:, -1]` (`_optics.py:405-438`), capped at
  `max_eps` (`:625`). ferrolearn `fn core_distance` value-matches the oracle even
  on the 60-blob fixture (Probe 1) — BUT it is consumed only via the
  `core_distances()` accessor whose sole non-test consumer is the crate re-export
  (Probe 6).
- REQ-2: **`ordering_` VALUE (R-DEV-1) — traversal/tie-break contract.** Mirror the
  linear-`argmin`-over-all-unprocessed seed selection with smallest-index
  tie-break (`compute_optics_graph` `:638-659`, comment `:639-640`/`:53`).
  ferrolearn uses a `BinaryHeap` min-heap + component-restart loop (`fn fit`,
  `fn update_seeds`, `SeedEntry`) → divergent ordering on the 60-blob fixture
  (Probe 2).
- REQ-3: **`reachability_` VALUE (R-DEV-1).** Mirror `_set_reach_dist`
  (`rdist = max(dist, core_dist)`, `np.around` to dtype precision, update on
  improvement) (`_optics.py:671-714`). The per-update arithmetic matches
  (`fn update_seeds` uses `max(core_dist_p, dist)`), but because the *order* of
  updates follows the divergent traversal (REQ-2), the reported `reachability_`
  plot diverges on hard fixtures (Probe 2). ferrolearn also omits the explicit
  `np.around(rdists, decimals=finfo.precision)` rounding (`:711`).
- REQ-4: **`predecessor_` contract & seed sentinel (R-DEV-3).** sklearn
  `predecessor_` is an `ndarray[int]` with seed points = `-1` (`:558-560`,
  `:604-605`). ferrolearn stores `Vec<Option<usize>>` and the `predecessors()`
  accessor returns `&[Option<usize>]` (seed = `None`) — a different output object
  type than sklearn's `-1`-sentinel integer array (R-DEV-3). Values agree on small
  fixtures (Probe 0) but diverge with the traversal on hard ones (REQ-2).
- REQ-5: **`labels_` VALUE parity (R-DEV-1/3 — the core requirement).** Mirror
  `OPTICS.fit().labels_` via `cluster_optics_xi` (`_optics.py:363-372`, `:810-918`).
  ferrolearn's labels diverge on the 60-blob fixture (3 clusters/0 noise vs
  sklearn 3 clusters/11 noise, Probe 3) because the plot diverges (REQ-2/3), the
  Xi `min_cluster_size` criterion is dropped (REQ-7), and the hierarchy/leaf-label
  selection differs (REQ-8).
- REQ-6: **`cluster_method='dbscan'` + `eps` + `cluster_optics_dbscan`
  (R-DEV-2).** sklearn `fit` branch `:374-390` and the free function `:726-788`.
  ferrolearn supports Xi only; no `cluster_method`/`eps` params and no
  `cluster_optics_dbscan` (Probe 4).
- REQ-7: **Xi `min_cluster_size` criterion 3.a (R-DEV-1).** sklearn enforces
  `c_end-c_start+1 < min_cluster_size → continue` *inside* `_xi_cluster`
  (`_optics.py:1155-1156`). ferrolearn applies size filtering *post-hoc* in
  `fn filter_small_clusters` (relabel-to-`-1`-then-renumber, `optics.rs:706-750`)
  — a different algorithm with a different result (Probe 3). (This is the
  collapsible-if clippy site at `optics.rs:717/718/744`.)
- REQ-8: **`cluster_hierarchy_` fitted attribute (R-DEV-3).** sklearn exposes
  `cluster_hierarchy_` = `(n_clusters,2)` `[start,end]` intervals ordered by
  `(end,-start)` (`_optics.py:191-200`, `:1166-1172`). ferrolearn's
  `fn xi_cluster_extraction` discards the intervals; `FittedOPTICS` has no
  `cluster_hierarchy_` accessor.
- REQ-9: **`predecessor_correction` toggle (R-DEV-2).** sklearn `bool`, default
  `True` (`:277`, `:1147-1150`). ferrolearn always applies it; no parameter.
- REQ-10: **parameter surface `metric`/`p`/`metric_params`/`algorithm`/`leaf_size`/
  `memory`/`n_jobs` + `min_samples` default `5` + float-fraction `min_samples`/
  `min_cluster_size` (R-DEV-2).** sklearn `__init__` `:266-297`,
  `_parameter_constraints` `:242-264`. ferrolearn `fn new` requires `min_samples`
  (no default, usize only) and exposes only `max_eps`/`xi`/`min_cluster_size`
  builders (Probe 5).
- REQ-11: **parameter-validation ABI (R-DEV-2).** sklearn `max_eps∈[0,inf]`,
  `xi∈[0,1]`, `min_samples` integer `[2,inf)` — endpoint-inclusive, raising
  `InvalidParameterError` (`_parameter_constraints` `:242-264`); empty/invalid
  shapes raise `ValueError`. ferrolearn over-rejects `max_eps=0`, `xi∈{0,1}`,
  permits `min_samples=1`, and raises `FerroError::InvalidParameter`/
  `InsufficientSamples` (the `fn fit` guards) — different bounds and ABI.
- REQ-12: **PyO3 binding (R-DEFER-1).** No `_RsOPTICS` in `ferrolearn-python`
  (Probe 6) — `import ferrolearn` cannot reach `OPTICS`.
- REQ-13: **ferray substrate (R-SUBSTRATE).** `optics.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float` + `std::collections::{BinaryHeap,
  HashMap}`, not `ferray-core` / `ferray::linalg`.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). Fixtures: `three_blobs` (the in-tree
9×2 test fixture), `docstring` (`[[1,2],[2,5],[3,6],[8,7],[8,8],[7,3]]`),
`blobs60` = `RandomState(0)` three `0.3·randn(20,2)` blobs at `(0,0)/(5,5)/(10,0)`
(`/tmp/optics_blobs.npy`).

- AC-1 (REQ-1, value-matches, no standalone consumer): `OPTICS(min_samples=5).fit(blobs60)`
  → `core_distances_[:5] = [0.3123, 0.5942, 0.3703, 0.2103, 0.1848]`; ferrolearn
  `core_distances()[..5]` identical to 4 dp.
- AC-2 (REQ-2, diverges): `OPTICS(min_samples=5).fit(blobs60).ordering_` →
  sklearn `[0,3,6,7,4,13,15,17,19,8,…]`; ferrolearn `[0,8,3,6,11,7,15,4,17,19,…]`
  — diverges at position 1.
- AC-3 (REQ-5, diverges): `OPTICS(min_samples=5).fit(blobs60).labels_` → sklearn
  has **11** noise points (`(labels_==-1).sum()==11`); ferrolearn has **0**.
- AC-4 (REQ-6): `cluster_optics_dbscan(reachability=r, core_distances=cd,
  ordering=o, eps=2.0)` runs in sklearn (`[0,0,0,0,0,0,…]`); ferrolearn has no such
  function and no `cluster_method='dbscan'`.
- AC-5 (REQ-8): `OPTICS(min_samples=5).fit(blobs60).cluster_hierarchy_` →
  `[[0,19],[22,30],[20,39],[40,59],[0,59]]` in sklearn; ferrolearn `FittedOPTICS`
  has no `cluster_hierarchy_` accessor.
- AC-6 (REQ-10/11): `OPTICS(min_samples=0.1).fit(blobs60)` runs (effective
  `min_samples=6`); `hasattr(OPTICS(),'metric')`/`'p'`/`'algorithm'`/`'cluster_method'`
  True; ferrolearn `OPTICS<F>` has no such params, requires `min_samples`, and
  rejects `xi=0.0`/`max_eps=0.0` that sklearn accepts.
- AC-7 (REQ-12): `import ferrolearn; ferrolearn.OPTICS` raises `AttributeError` —
  no binding.

## REQ status table

Binary (R-DEFER-2). `OPTICS` / `FittedOPTICS` are existing pub APIs re-exported at
the crate root (the only non-test consumer; grandfathered S5/R-DEFER-1). Cites use
symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live
oracle = installed sklearn 1.5.2, run from `/tmp`. Honest assessment (R-HONEST-3):
**REQ-1 (`core_distances_` VALUE) SHIPS** — it value-matches the oracle even on the
hard 60-blob fixture (Probe 1), green-guarded, through the crate re-export consumer
(grandfathered S5/R-DEFER-1). The load-bearing `ordering_`/`reachability_`/`labels_`
match on clean fixtures (Probe 0) but DIVERGE on noisy data (Probe 2/3) — the root
cause is the BinaryHeap traversal (REQ-2 #1080), the prime single-file fixable
divergence; `labels_` additionally needs the in-Xi `min_cluster_size` (REQ-7) and
`cluster_hierarchy_` (REQ-8). Blocker numbers below are the real filed issues.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`core_distances_` VALUE) | SHIPPED | impl `fn core_distance` (distance to the `(min_samples-1)`-th other point within `max_eps`) value-matches `_compute_core_distances_` = `kneighbors(X,min_samples)[0][:,-1]` (`_optics.py:405-438`, cap `:625`) EVEN on the hard 60-blob fixture (Probe 1: `core_distances_[:5]=[0.3123,0.5942,0.3703,0.2103,0.1848]` to 4 dp). Consumer: crate re-export `pub use optics::{FittedOPTICS, OPTICS}` (`lib.rs`, grandfathered S5/R-DEFER-1). Guard: live-oracle `#[test]` in `tests/divergence_optics.rs`. OPTICS is deterministic, so this is genuine value-parity. |
| REQ-2 (`ordering_` traversal value-parity) | SHIPPED | impl `Fit::fit` now selects the next seed by LINEAR ARGMIN over all unprocessed reachability with smallest-index tie-break (single pool, no heap), matching sklearn `compute_optics_graph` (`_optics.py:638-659`, `:53` "we do not employ a heap"). The prior `BinaryHeap` traversal diverged on tie-prone data. Guards: `green_ordering_small10` (sklearn `[0,1,3,9,5,6,7,8,2,4]`; was `[…5,7,8,6…]`) + `green_ordering_three_blobs`/`green_ordering_docstring`. Fixed #1080. |
| REQ-3 (`reachability_` VALUE) | SHIPPED | impl `fn update_seeds` computes `max(core_dist_p, dist)` then rounds via `fn round_to_precision` (`np.around(decimals=np.finfo(dtype).precision)`, round-ties-even, `_optics.py:711`); with the REQ-2 traversal the reported plot value-matches sklearn. Guards: `green_reachability_docstring` + `green_reachability_small10` (noisy tie fixture, live-oracle). Fixed #1080 (bundled). |
| REQ-4 (`predecessor_` contract / `-1` sentinel) | NOT-STARTED | open prereq blocker **#1082**. sklearn `predecessor_` is an int `ndarray` with seeds = `-1` (`:558-560`/`:604-605`); ferrolearn `predecessors()` returns `&[Option<usize>]` (seed = `None`) — a different output object (R-DEV-3). Values agree on `three_blobs`/docstring (Probe 0) but diverge with the traversal (REQ-2). |
| REQ-5 (`labels_` VALUE parity) | NOT-STARTED | open prereq blocker **#1083** (depends on REQ-7 #1085 + REQ-8 #1086). `fit`→`fn xi_cluster_extraction`→`fn filter_small_clusters` derive Xi labels via a dropped in-Xi size criterion (REQ-7) + missing hierarchy leaf-selection (REQ-8). With the REQ-2 traversal fixed, the reachability plot now value-matches, so labels AGREE on small/medium fixtures (blobs60 now matches sklearn 11-noise exactly) — but they DIVERGE on harder data: sklearn's canonical OPTICS test (1500 pts, min_samples=9) → sklearn 21 clusters / 1167 noise vs ferrolearn 19 clusters / 1228 noise. NOT a contract until the in-Xi `min_cluster_size` (#1085) + `cluster_hierarchy_` leaf-selection (#1086) land. |
| REQ-6 (`cluster_method='dbscan'` + `eps` + `cluster_optics_dbscan`) | NOT-STARTED | open prereq blocker **#1084**. sklearn `fit` branch (`_optics.py:374-390`) + free fn `cluster_optics_dbscan` (`:726-788`, linear `cumsum` extraction `:781-787`). ferrolearn has no `cluster_method`/`eps` params and no `cluster_optics_dbscan` (Probe 4, AC-4). |
| REQ-7 (Xi `min_cluster_size` criterion 3.a) | NOT-STARTED | open prereq blocker **#1085**. sklearn enforces size *inside* `_xi_cluster` (`c_end-c_start+1 < min_cluster_size → continue`, `_optics.py:1155-1156`). ferrolearn ignores size inside `fn xi_cluster_extraction` (only drops length-1) and post-filters in `fn filter_small_clusters` (relabel-then-renumber, `optics.rs:706-750`, the collapsible-if clippy site `:717/718/744`) — a different algorithm/fixed point (Probe 3). |
| REQ-8 (`cluster_hierarchy_` attribute) | NOT-STARTED | open prereq blocker **#1086**. sklearn `cluster_hierarchy_` = `(n_clusters,2)` `[start,end]` ordered by `(end,-start)` (`_optics.py:191-200`, `_xi_cluster` `:1166-1172`); `_extract_xi_labels` (`:1175-1201`) derives leaf labels from it. ferrolearn `fn xi_cluster_extraction` collects intervals into `clusters` then **discards** them; `FittedOPTICS` has no accessor. AC-5: sklearn `[[0,19],[22,30],[20,39],[40,59],[0,59]]`. |
| REQ-9 (`predecessor_correction` toggle) | NOT-STARTED | open prereq blocker **#1087**. sklearn `bool` default `True` (`:277`), applied conditionally in `_xi_cluster` (`:1147-1150`). ferrolearn `fn correct_predecessor` is unconditional in `fn xi_cluster_extraction`; no parameter on `OPTICS<F>`. |
| REQ-10 (param surface + `min_samples` default 5 + float fractions) | NOT-STARTED | open prereq blocker **#1088**. sklearn `__init__` (`_optics.py:266-297`) has 14 params incl. `metric`/`p`/`metric_params`/`algorithm`/`leaf_size`/`memory`/`n_jobs` and `min_samples=5` default accepting `float∈(0,1)` (`:243-246`, normalised `:598-599`). ferrolearn `fn new(min_samples: usize)` requires `min_samples` (no default, no fraction) and exposes only `max_eps`/`xi`/`min_cluster_size` builders (Probe 5, AC-6). |
| REQ-11 (validation bounds + ABI) | NOT-STARTED | open prereq blocker **#1089**. sklearn `max_eps∈[0,inf]`, `xi∈[0,1]` (closed-both), `min_samples∈[2,inf)`, raising `InvalidParameterError`/`ValueError` (`_parameter_constraints` `:242-264`). ferrolearn `fn fit` over-rejects `max_eps=0`, `xi∈{0,1}`, permits `min_samples=1`, and raises `FerroError::InvalidParameter`/`InsufficientSamples` — different bounds + ABI (AC-6). |
| REQ-12 (PyO3 binding) | NOT-STARTED | open prereq blocker **#1090**. `grep -rln OPTICS ferrolearn-python/` is EMPTY (Probe 6) — no `_RsOPTICS`, so `import ferrolearn` cannot reach `OPTICS` (AC-7). The only non-test consumer of `fit`/`fit_predict`/the accessors is the crate re-export (`lib.rs`). |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker **#1091**. `optics.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` + `std::collections::{BinaryHeap, HashMap}`; not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`optics.rs` follows the unfitted/fitted split (CLAUDE.md naming): `OPTICS<F>`
(`min_samples`, `max_eps`, `xi`, `min_cluster_size`; `fn new` + `with_*` builders)
→ `Fit<Array2<F>, ()>` → `FittedOPTICS<F>` (private `ordering_`, `reachability_`,
`core_distances_`, `labels_`, `predecessors_`, `min_samples_`; accessors
`ordering()`/`reachability()`/`core_distances()`/`labels()`/`predecessors()`/
`n_clusters()` + `extract_clusters(xi)`). Generic over `F: Float + Send + Sync +
'static`; every public method returns `Result<_, FerroError>` (R-CODE-2). No
`Predict` impl — matching sklearn (OPTICS has no `predict`; the module doc-comment
notes this).

**Fit path (`fn fit`).** Validates `min_samples > 0` (sklearn requires `>= 2`,
REQ-11), `max_eps > 0` (sklearn `[0,inf]`, REQ-11), `xi ∈ (0,1)` (sklearn `[0,1]`,
REQ-11), `n_samples >= 1` (sklearn raises `ValueError` for empty). Then:
1. **Core distances** — `fn core_distance` for every point: brute-force distances
   within `max_eps`, sorted, take the `(min_samples-1)`-th (= sklearn `kneighbors`
   k-th-NN, `_optics.py:437`); `inf` if too few neighbours. **Value-matches
   sklearn** (REQ-1, Probe 1) — equal to `algorithm='brute'`, default metric.
2. **Reachability traversal** — the `for start in 0..n_samples` outer loop seeds an
   unprocessed point, then a `BinaryHeap<SeedEntry>` min-heap expands by smallest
   reachability with `fn update_seeds` (`rdist = max(core_dist_p, dist)`) and a
   stale-entry skip. **Diverges from sklearn's linear-`argmin`-over-all-unprocessed
   + smallest-index tie-break** (`compute_optics_graph` `:638-659`; comment `:53`
   "we do not employ a heap") — the core divergence (REQ-2/3, Probe 2).
3. **Xi extraction** — `fn xi_cluster_extraction` builds the reachability plot
   (with an `inf` tail sentinel, matching `_optics.py:1070`), computes
   steep-up/steep-down/up/down arrays via the `ratio = r[i]/r[i+1]` test (matching
   `:1083-1087`), runs the Figure-19 SDA loop (`fn extend_region`,
   `fn update_filter_sdas`, `fn correct_predecessor` mirror `_extend_region`/
   `_update_filter_sdas`/`_correct_predecessor` `:921-1017`), then emits **leaf
   labels** by greedy all-unassigned interval assignment. It **collects** cluster
   intervals but **discards** them (no `cluster_hierarchy_`, REQ-8) and **omits the
   in-Xi `min_cluster_size` criterion 3.a** (REQ-7).
4. **`min_cluster_size` post-filter** — `fn filter_small_clusters` (`optics.rs:706-750`)
   counts final label sizes, relabels sub-`min_size` clusters to `-1`, renumbers
   contiguously. This is a **post-hoc replacement** for sklearn's in-Xi size
   criterion (REQ-7) and produces a different `labels_` on noisy data (Probe 3).
   This function is the clippy collapsible-if site (`:717/:718/:744`).

**Invariants held vs sklearn:** `core_distances_` VALUE (REQ-1, AC-1 — but
private-accessor / no oracle test); the reachability-graph on small/clean fixtures
(`three_blobs`, docstring — Probe 0 AGREE); `ordering()` is a permutation of
`0..n_samples`; first point in each component has `inf` reachability + `None`
predecessor; `labels()` length = `n_samples`, noise = `-1`; `fit_predict` =
`fit().labels()` (mirrors `ClusterMixin.fit_predict`).

**Invariants NOT held vs sklearn:** `ordering_` traversal/tie-break (REQ-2 — heap
vs linear argmin); `reachability_` plot on hard fixtures (REQ-3); `predecessor_`
output type / `-1` sentinel (REQ-4); `labels_` VALUE (REQ-5 — blobs60 0 vs 11
noise); `cluster_method='dbscan'` + `cluster_optics_dbscan` (REQ-6); in-Xi
`min_cluster_size` (REQ-7); `cluster_hierarchy_` (REQ-8);
`predecessor_correction` toggle (REQ-9); the parameter surface +
`min_samples` default/fraction (REQ-10); validation bounds + ABI (REQ-11); the
PyO3 binding (REQ-12); the ferray substrate (REQ-13).

**Consumer wiring.** The only non-test consumer is the crate re-export
(`pub use optics::{FittedOPTICS, OPTICS}`, `ferrolearn-cluster/src/lib.rs`). There
is no `ferrolearn-python` binding and no other in-crate consumer (Probe 6).

## Verification

Library crate (green at baseline `beb6862e` for the existing behaviour):
```
cargo test -p ferrolearn-cluster --lib optics     # 25 passed; 0 failed
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The 25 in-tree `#[test]`s pin ferrolearn's current behaviour — `ordering` covers
all points / is unique, reachability/core/labels/predecessor lengths,
first-point-`inf`-reachability, isolated-point `inf` core distance, `max_eps`
limiting, `min_cluster_size` filtering, `xi`/`min_samples`/`max_eps`/empty/single
edge cases, and f32 support. **None compares `ordering_`/`reachability_`/`labels_`
VALUE against the live sklearn `OPTICS` oracle on a noisy fixture**, so they stay
green despite the Probe-2/3 divergences. Note `test_invalid_xi_zero`/
`test_invalid_xi_one`/`test_invalid_max_eps_zero` actively assert the REQ-11
over-rejections (sklearn accepts `xi∈{0,1}` and `max_eps=0`) — they must be
corrected when REQ-11 lands (R-HONEST-4).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin REQ-2 (the traversal) FIRST** — it is
the single-file root cause; REQ-3/REQ-5 cascade from it:
```
# Build the blobs60 fixture (RandomState(0), three 0.3*randn(20,2) blobs):
python3 -c "import numpy as np; rng=np.random.RandomState(0); \
X=np.vstack([rng.randn(20,2)*0.3+[0,0],rng.randn(20,2)*0.3+[5,5],rng.randn(20,2)*0.3+[10,0]]); \
np.save('/tmp/optics_blobs.npy',X)"
# REQ-1 (value-matches) core_distances
python3 -c "import numpy as np; from sklearn.cluster import OPTICS; X=np.load('/tmp/optics_blobs.npy'); \
print([round(float(c),4) for c in OPTICS(min_samples=5).fit(X).core_distances_[:5]])"   # [0.3123,0.5942,0.3703,0.2103,0.1848]
# REQ-2 (DIVERGES) ordering
python3 -c "import numpy as np; from sklearn.cluster import OPTICS; X=np.load('/tmp/optics_blobs.npy'); \
print(OPTICS(min_samples=5).fit(X).ordering_.tolist()[:10])"   # [0,3,6,7,4,13,15,17,19,8]; ferro [0,8,3,6,11,7,15,4,17,19]
# REQ-5 (DIVERGES) labels noise count
python3 -c "import numpy as np; from sklearn.cluster import OPTICS; X=np.load('/tmp/optics_blobs.npy'); \
print(int((OPTICS(min_samples=5).fit(X).labels_==-1).sum()))"   # 11; ferro 0
# REQ-8 (ABSENT) cluster_hierarchy_
python3 -c "import numpy as np; from sklearn.cluster import OPTICS; X=np.load('/tmp/optics_blobs.npy'); \
print(OPTICS(min_samples=5).fit(X).cluster_hierarchy_.tolist())"  # [[0,19],[22,30],[20,39],[40,59],[0,59]]
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-cluster/tests/divergence_optics.rs`, asserting the live-sklearn
expected values above and FAILING against current `optics.rs`. (ferrolearn values
in this doc came from a throwaway `cargo run --example optics_probe`, since
deleted.)

ferrolearn-python (REQ-12 binding parity, after #1090 lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_optics.py -q
```
asserting `ferrolearn.OPTICS` exists and exposes `labels_`/`reachability_`/
`ordering_`/`core_distances_`/`predecessor_`/`cluster_hierarchy_` + the sklearn
parameter surface, matching `sklearn.cluster.OPTICS` on the AC fixtures.

## Blockers to open

(Director creates the real issues; numbers are SUGGESTIONS.)

- **#1080** — REQ-2/REQ-3: replace the `BinaryHeap` seed traversal + outer
  component-restart loop with sklearn's single-pool linear-`argmin`-over-all-
  unprocessed + smallest-index tie-break (`compute_optics_graph` `:638-659`),
  and add the `np.around` reachability rounding (`:711`). **The core single-file
  fix — pin FIRST.** REQ-5 unblocks once this + #1083 + #1083 land.
- **#1083** — REQ-5: `labels_` value parity (gated on #1083/#1083/#1083).
- **#1084** — REQ-6: add `cluster_method`/`eps` params + the `cluster_optics_dbscan`
  free function (`_optics.py:726-788`).
- **#1085** — REQ-7: move `min_cluster_size` into `_xi_cluster` as criterion 3.a
  (`_optics.py:1155-1156`); retire the post-hoc `fn filter_small_clusters` (which
  also removes the collapsible-if clippy debt at `optics.rs:717/718/744`).
- **#1086** — REQ-8: expose `cluster_hierarchy_` from `fn xi_cluster_extraction`'s
  collected intervals, ordered `(end,-start)` (`_optics.py:1166-1172`).
- **#1087** — REQ-9: add the `predecessor_correction` bool parameter
  (`_optics.py:277`, `:1147-1150`).
- **#1088** — REQ-10: add `metric`/`p`/`metric_params`/`algorithm`/`leaf_size`/
  `memory`/`n_jobs` params + `min_samples=5` default + float-fraction
  `min_samples`/`min_cluster_size` (`_optics.py:266-297`, `:598-599`).
- **#1089** — REQ-11: `max_eps∈[0,inf]`, `xi∈[0,1]`, `min_samples∈[2,inf)`
  endpoint-inclusive bounds + the sklearn `InvalidParameterError`/`ValueError` ABI
  (`_parameter_constraints` `:242-264`).
- **#1090** — REQ-12: add `_RsOPTICS` to `ferrolearn-python` (fit / fit_predict /
  labels_ / reachability_ / ordering_ / core_distances_ / predecessor_ /
  cluster_hierarchy_ + parameter surface).
- **#1091** — REQ-13: migrate `optics.rs` off `ndarray`/`num-traits` to
  `ferray-core` (R-SUBSTRATE).
- (REQ-1 is SHIPPED — `core_distances_` value-matches the oracle, green-guarded; no blocker.)
- **#1082** — REQ-4: align `predecessor_` to an `isize` `-1`-sentinel array
  matching sklearn's output object (`_optics.py:558-560`).
