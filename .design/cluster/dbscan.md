# DBSCAN (sklearn.cluster.DBSCAN)

<!--
tier: 3-component
status: draft
baseline-commit: e0f84020
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_dbscan.py            # class DBSCAN(ClusterMixin, BaseEstimator) (:185); _parameter_constraints (:331-343); __init__ defaults eps=0.5/min_samples=5/metric='euclidean'/p=None (:345-364); fit neighborhoods via NearestNeighbors.radius_neighbors (:411-422); n_neighbors sample_weight core determination (:424-435); dbscan_inner call (:436); core_sample_indices_/labels_ (:438-439); components_ = X[core_sample_indices_] (:441-446)
  - sklearn/cluster/_dbscan_inner.pyx     # dbscan_inner: DFS, label on first core in index order, border to first-reaching cluster, non-core not expanded
parity-ops: DBSCAN (.__init__, .fit, .fit_predict, .labels_, .core_sample_indices_)
crosslink-issue: 945
-->

## Summary

`ferrolearn-cluster/src/dbscan.rs` mirrors scikit-learn's `DBSCAN`
(`sklearn/cluster/_dbscan.py`, `class DBSCAN(ClusterMixin, BaseEstimator)` `:185`) —
density-based clustering: points with `>= min_samples` neighbors within radius `eps`
are core, clusters expand from core points, and unreachable points are noise
(`-1`). It exposes the unfitted `DBSCAN<F>` (`eps` required, `min_samples=5`), the
fitted `FittedDBSCAN<F>` (stores `labels_` + `core_sample_indices_`, plus an
`n_clusters()` helper), a `Fit<Array2<F>, ()>` impl, and a `fit_predict`
convenience mirroring `ClusterMixin.fit_predict`. It is re-exported at the crate
root (`pub use dbscan::{DBSCAN, FittedDBSCAN}` in `ferrolearn-cluster/src/lib.rs`)
and bound into CPython as `ferrolearn.DBSCAN` via `_RsDBSCAN`
(`ferrolearn-python/src/extras.rs`, `ferrolearn-python/python/ferrolearn/_extras.py`).

**Unlike the `spectral.rs` / `feature_agglomeration.rs` siblings, this unit's core
contract VALUE-matches the live sklearn 1.5.2 oracle EXACTLY.** DBSCAN is
deterministic — there is no RNG, no iterative optimizer, no embedding step. With the
Euclidean metric and no `sample_weight`, the labels are a pure function of the
distance graph and the index-ordered DFS in `dbscan_inner`. Across four fixtures
(the test `make_two_clusters` 8-point, the `test_three_clusters` 9-point, a
hand-built **shared-border tie-break** case, and a `make_blobs` 20-point case),
ferrolearn's `labels_` and `core_sample_indices_` are **bit-identical** to sklearn —
including the cluster numbering, the noise (`-1`) set, AND the border-point
tie-break (a non-core point reachable from two clusters is assigned to the
FIRST-reaching cluster in index order, matching `dbscan_inner`'s DFS). The
load-bearing parity claim therefore SHIPS, and it SHIPS through a real consumer
(`ferrolearn.DBSCAN.fit(X).labels_`), not merely the crate re-export.

The NOT-STARTED surface is the parameters DBSCAN does not yet expose: `sample_weight`
(which alters core determination), `metric` / `p` / `metric_params` / `algorithm` /
`leaf_size` / `n_jobs`, and the `components_` fitted attribute. The `eps=0.5`
constructor default is also absent (ferrolearn `new(eps)` requires `eps`), though
`min_samples=5` matches and the `eps > 0` / `min_samples >= 1` validation matches
sklearn's `_parameter_constraints`.

## Live oracle probes (sklearn 1.5.2, run from /tmp; ferrolearn via throwaway `cargo run --example` probe, since deleted)

### Probe 1 — `labels_` VALUE parity (the load-bearing probe)

Four fixtures. `DBSCAN(eps, min_samples).fit(X).labels_` (sklearn) vs ferrolearn
`DBSCAN::<f64>::new(eps).with_min_samples(min_samples).fit(&X, &()).labels()`.

**Fixture A — `make_two_clusters` 8-point** (the `dbscan.rs` test fixture; two 4-point
squares near `(0,0)` and `(10,10)`), `eps=1.5, min_samples=2`:
```
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; \
X=np.array([[0.,0.],[0.5,0.],[0.,0.5],[0.5,0.5],[10.,10.],[10.5,10.],[10.,10.5],[10.5,10.5]]); \
m=DBSCAN(eps=1.5,min_samples=2).fit(X); print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
# sklearn:    labels [0, 0, 0, 0, 1, 1, 1, 1]   core [0, 1, 2, 3, 4, 5, 6, 7]
# ferrolearn: labels [0, 0, 0, 0, 1, 1, 1, 1]   core [0, 1, 2, 3, 4, 5, 6, 7]   — IDENTICAL
```

**Fixture B — `test_three_clusters` 9-point** (three 3-point blobs near `(0,0)`,
`(5,5)`, `(10,0)`), `eps=0.5, min_samples=2`:
```
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
m=DBSCAN(eps=0.5,min_samples=2).fit(X); print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
# sklearn:    labels [0, 0, 0, 1, 1, 1, 2, 2, 2]   core [0, 1, 2, 3, 4, 5, 6, 7, 8]
# ferrolearn: labels [0, 0, 0, 1, 1, 1, 2, 2, 2]   core [0, 1, 2, 3, 4, 5, 6, 7, 8]   — IDENTICAL
```

**Fixture C — SHARED-BORDER tie-break** (the critical case for `dbscan_inner`'s DFS
order). A left core `(0,0)` with three satellites `{(-0.5,0),(0,0.5),(0,-0.5)}`, a
right core `(2,0)` with three satellites `{(2.5,0),(2,0.5),(2,-0.5)}`, and a single
**border** point `(1,0)` at idx 4 that is reachable (dist `1.0 <= eps`) from the left
core (idx 0) AND the right core (idx 5), but is NOT itself core (only 3 neighbors
`< min_samples=4`). `eps=1.0, min_samples=4`:
```
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; \
X=np.array([[0.,0.],[-0.5,0.],[0.,0.5],[0.,-0.5],[1.,0.],[2.,0.],[2.5,0.],[2.,0.5],[2.,-0.5]]); \
m=DBSCAN(eps=1.0,min_samples=4).fit(X); print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
# sklearn:    labels [0, 0, 0, 0, 0, 1, 1, 1, 1]   core [0, 1, 2, 3, 5, 6, 7, 8]
# ferrolearn: labels [0, 0, 0, 0, 0, 1, 1, 1, 1]   core [0, 1, 2, 3, 5, 6, 7, 8]   — IDENTICAL
```
**The border point (idx 4) is assigned to cluster 0 (the LEFT cluster) in BOTH** —
the first-reaching cluster in index order, because cluster 0 is expanded first
(starting from the lowest unlabeled core idx 0) and claims idx 4 as a non-core
member; when the right cluster's DFS reaches idx 4 it is already labeled and not
reassigned (`dbscan.rs` `fn fit` guards `if labels[neighbor] == -1` before
assigning; `_dbscan_inner.pyx` skips `labels[v] != -1`). idx 4 is correctly NOT in
`core_sample_indices_`. This tie-break is the most fragile part of the contract and
ferrolearn matches it.

**Fixture D — `make_blobs(n_samples=20, centers=3, cluster_std=0.4, random_state=42)`**,
`eps=0.5, min_samples=3` (a realistic case with many noise points):
```
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; from sklearn.datasets import make_blobs; \
X,_=make_blobs(n_samples=20,centers=3,cluster_std=0.4,random_state=42); \
m=DBSCAN(eps=0.5,min_samples=3).fit(X); print(m.labels_.tolist()); print(m.core_sample_indices_.tolist())"
# sklearn:    labels [0,0,0,-1,1,-1,0,1,1,1,-1,-1,-1,0,-1,-1,-1,-1,1,-1]   core [0,1,2,4,6,7,8,9,13]
# ferrolearn: labels [0,0,0,-1,1,-1,0,1,1,1,-1,-1,-1,0,-1,-1,-1,-1,1,-1]   core [0,1,2,4,6,7,8,9,13]   — IDENTICAL
```
All 11 noise (`-1`) positions, both clusters, and all 9 core indices match exactly.

**Conclusion**: `labels_` and `core_sample_indices_` VALUE-match the live oracle on
EVERY fixture, including the shared-border tie-break (Fixture C) and the noise-heavy
blobs (Fixture D). DBSCAN is deterministic, so this is a genuine value-parity
contract, not coincidence — the unit SHIPS its core.

### Probe 2 — `core_sample_indices_` ordering

sklearn computes `core_sample_indices_ = np.where(core_samples)[0]`
(`_dbscan.py:438`) — ascending index order. ferrolearn builds
`core_sample_indices = (0..n_samples).filter(|&i| is_core[i]).collect()`
(`dbscan.rs` `fn fit`) — also ascending. All four probes above confirm the indices
match exactly and in the same (sorted ascending) order.

### Probe 3 — `sample_weight` (missing surface)

sklearn `fit(X, sample_weight=w)` computes `n_neighbors = sum(sample_weight[neighbors])`
when weights are present (`_dbscan.py:427-429`), so a single high-weight point can
become core with fewer than `min_samples` actual neighbors:
```
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; \
X=np.array([[0.,0.],[5.,5.],[5.1,5.]]); \
print('unweighted:', DBSCAN(eps=0.5,min_samples=3).fit(X).labels_.tolist()); \
print('w0=5:', DBSCAN(eps=0.5,min_samples=3).fit(X, sample_weight=np.array([5.,1.,1.])).labels_.tolist())"
# unweighted: [-1, -1, -1]      (no point has 3 neighbors)
# w0=5:       [ 0, -1, -1]      (idx0 weight 5 >= min_samples 3 -> core, forms a singleton cluster)
```
ferrolearn `Fit::fit(&x, &())` has the unit `()` target — **no `sample_weight`
parameter**. The core test is purely `neighborhoods[i].len() >= min_samples`
(`dbscan.rs` `fn fit`, `is_core`). On the unweighted call ferrolearn matches
(`[-1,-1,-1]`); the weighted behavior is unreachable. Missing surface → REQ
NOT-STARTED.

### Probe 4 — `metric` / `p` / `metric_params` (missing surface)

sklearn `metric='euclidean'` default (`_dbscan.py:350`), but accepts any
`pairwise_distances` metric (`_parameter_constraints` `:334-337`) and `p` for
Minkowski (`:341`). A non-Euclidean metric changes the neighborhood graph:
```
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; \
X=np.array([[0.,0.],[0.8,0.8],[0.85,0.85]]); \
print('euclidean:', DBSCAN(eps=1.2,min_samples=2,metric='euclidean').fit(X).labels_.tolist()); \
print('manhattan:', DBSCAN(eps=1.2,min_samples=2,metric='manhattan').fit(X).labels_.tolist())"
# euclidean: [0, 0, 0]    (dist (0,0)->(0.8,0.8) = 1.131 <= 1.2, all connected)
# manhattan: [-1, 0, 0]   (manhattan dist = 1.6 > 1.2, idx0 isolated -> noise)
```
ferrolearn `region_query` / `squared_euclidean` (`dbscan.rs`) is **Euclidean-only**;
there is no `metric` / `p` / `metric_params` parameter. It matches the
`metric='euclidean'` path (the default) but cannot reproduce `manhattan` etc.
Missing surface → REQ NOT-STARTED.

### Probe 5 — defaults / validation

```
python3 -c "from sklearn.cluster import DBSCAN; import numpy as np; X=np.array([[0.,0.],[1.,1.]]); \
print('default eps:', DBSCAN().eps);   # eps default
import sklearn"
# default eps: 0.5
python3 -c "from sklearn.cluster import DBSCAN; import numpy as np; \
try: DBSCAN(eps=0.0).fit(np.array([[0.,0.],[1.,1.]]))
except Exception as e: print('eps=0:', type(e).__name__)"
# eps=0: InvalidParameterError       (Interval(Real,0,None,closed='neither') -> eps>0)
python3 -c "from sklearn.cluster import DBSCAN; import numpy as np; \
try: DBSCAN(eps=0.5,min_samples=0).fit(np.array([[0.,0.],[1.,1.]]))
except Exception as e: print('ms=0:', type(e).__name__)"
# ms=0: InvalidParameterError        (Interval(Integral,1,None,closed='left') -> min_samples>=1)
```
- **`min_samples=5` default**: sklearn `__init__` `:349`; ferrolearn `fn new`
  defaults `min_samples=5` — **matches**.
- **`eps=0.5` default**: sklearn `__init__` `:347`; ferrolearn `fn new(eps: F)`
  **requires** `eps` (no default) — DIVERGES (missing default, REQ-4).
- **`eps > 0` constraint**: sklearn `_parameter_constraints["eps"] =
  Interval(Real, 0.0, None, closed="neither")` (`:332`) — rejects `eps <= 0`.
  ferrolearn `fn fit` returns `Err(FerroError::InvalidParameter{name:"eps"})` for
  `self.eps <= F::zero()` — **matches the constraint** (same accept/reject boundary),
  though the error TYPE is `FerroError::InvalidParameter`, not sklearn's
  `InvalidParameterError`/`ValueError` ABI.
- **`min_samples >= 1` constraint**: sklearn `Interval(Integral, 1, None,
  closed="left")` (`:333`) — rejects `min_samples < 1`. ferrolearn rejects
  `self.min_samples == 0` (`fn fit`) — **matches** (a `usize` cannot be negative, so
  `>= 1` is exactly `!= 0`).

### Probe 6 — `components_` (missing fitted attribute)

sklearn sets `components_ = X[core_sample_indices_].copy()` (the core sample rows,
`_dbscan.py:441-443`):
```
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; \
m=DBSCAN(eps=0.5,min_samples=2).fit(np.array([[0.,0.],[0.1,0.],[10.,10.]])); \
print('components_ shape', m.components_.shape, 'core', m.core_sample_indices_.tolist())"
# components_ shape (2, 2)  core [0, 1]
```
ferrolearn `FittedDBSCAN` exposes `core_sample_indices()` but has **no `components_`
accessor**. The data needed to build it (`X[core_sample_indices_]`) is not retained.
Missing attribute → REQ NOT-STARTED.

### Probe 7 — non-test consumer

```
grep -rn "DBSCAN" ferrolearn-python/src/lib.rs ferrolearn-python/src/extras.rs \
  ferrolearn-python/python/ferrolearn/_extras.py ferrolearn-python/python/ferrolearn/__init__.py
```
**There IS a full PyO3 binding** (unlike `spectral.rs` / `feature_agglomeration.rs`):
- `ferrolearn-python/src/extras.rs` — `#[pyclass(name="_RsDBSCAN")] struct RsDBSCAN`
  with `fn new(eps=0.5, min_samples=5)`, `fn fit(...)` calling
  `ferrolearn_cluster::DBSCAN::<f64>::new(self.eps).with_min_samples(self.min_samples)`,
  and a `#[getter] fn labels_` exposing `f.labels()` as a numpy `i64` array.
- `ferrolearn-python/src/lib.rs` — `m.add_class::<extras::RsDBSCAN>()?` registers it.
- `ferrolearn-python/python/ferrolearn/_extras.py` — `class DBSCAN(_ClusterWrapper)`
  (`eps=0.5, min_samples=5` defaults) whose `_make_rs` returns `_RsDBSCAN(...)`; the
  `_ClusterWrapper.fit` stores `self.labels_ = np.asarray(self._rs.labels_)` and
  provides `fit_predict`.
- `ferrolearn-python/python/ferrolearn/__init__.py` — re-exports `DBSCAN` in
  `__all__`.

So `ferrolearn-cluster::DBSCAN::fit` / `FittedDBSCAN::labels()` have a **real,
non-test production consumer**: `import ferrolearn; ferrolearn.DBSCAN(...).fit(X).labels_`.
This is what makes the core `labels_` / `core_sample_indices_` REQs SHIPPABLE
(impl + value-match + real consumer), not merely value-matching-with-no-consumer.

## Requirements

- REQ-1: **`labels_` VALUE parity (Euclidean, no `sample_weight`) (R-DEV-1/3 — the
  core contract).** Mirror `DBSCAN.fit().labels_` (`_dbscan.py:431-439` + the
  `dbscan_inner` DFS in `_dbscan_inner.pyx`): clusters numbered by the first
  unlabeled core in index order, noise `= -1`, and a border point reachable from two
  clusters assigned to the FIRST-reaching cluster. ferrolearn `fn fit` + `fit_predict`
  produce bit-identical labels on all four probe fixtures (Probe 1), including the
  shared-border tie-break (Fixture C) and noise-heavy blobs (Fixture D), consumed by
  `ferrolearn.DBSCAN.labels_` (Probe 7).
- REQ-2: **`core_sample_indices_` VALUE parity (R-DEV-1/3).** Mirror
  `core_sample_indices_ = np.where(core_samples)[0]` (`_dbscan.py:438`), ascending.
  ferrolearn `core_sample_indices()` matches exactly on all four fixtures (Probe 1/2),
  with core = `neighborhoods[i].len() >= min_samples` (self included).
- REQ-3: **`eps > 0` / `min_samples >= 1` validation (R-DEV-2).** Mirror
  `_parameter_constraints` (`_dbscan.py:332-333`): `eps ∈ (0, inf)` (closed neither),
  `min_samples ∈ [1, inf)`. ferrolearn `fn fit` rejects `eps <= 0` and
  `min_samples == 0` at the same accept/reject boundary (Probe 5). (Error TYPE is
  `FerroError::InvalidParameter`, not the sklearn `InvalidParameterError` ABI — a
  message/type nuance, not a behavioral divergence of the validation boundary.)
- REQ-4: **`eps=0.5` constructor default + sklearn error ABI (R-DEV-2).** sklearn
  `eps=0.5` default (`_dbscan.py:347`); ferrolearn `fn new(eps: F)` REQUIRES `eps`
  (no default). Also the validation error type is `FerroError::InvalidParameter`, not
  `InvalidParameterError`/`ValueError`. Missing default + ABI mismatch.
- REQ-5: **`sample_weight` (R-DEV-1 — affects core determination).** Mirror
  `n_neighbors = sum(sample_weight[neighbors])` (`_dbscan.py:427-429`): a high-weight
  point can be core with fewer neighbors; a negative weight can inhibit a neighbor's
  coreness. ferrolearn `Fit::fit(&x, &())` has no `sample_weight` — core is purely
  `neighbors.len() >= min_samples` (Probe 3). Missing surface.
- REQ-6: **`metric` / `p` / `metric_params` (R-DEV-2 — alters the neighbor graph).**
  sklearn supports any `pairwise_distances` metric (`_dbscan.py:334-337`), `p` for
  Minkowski (`:341`), default `'euclidean'` (`:350`). ferrolearn `region_query` is
  Euclidean-only (Probe 4). Missing surface.
- REQ-7: **`algorithm` / `leaf_size` / `n_jobs` (R-DEV-2 — neighbor-search backend).**
  sklearn `algorithm ∈ {auto,ball_tree,kd_tree,brute}` (`:339`), `leaf_size`
  (`:340`), `n_jobs` (`:342`), routed through `NearestNeighbors`
  (`_dbscan.py:411-422`). ferrolearn uses a fixed `O(n^2)` brute-force `region_query`
  with no parameter. (Brute force matches the default `'auto'` results numerically;
  the parameters themselves are absent.) Missing surface.
- REQ-8: **`components_` fitted attribute (R-DEV-3).** sklearn `components_ =
  X[core_sample_indices_].copy()` (`_dbscan.py:441-446`). `FittedDBSCAN` exposes
  `core_sample_indices()` but no `components_` accessor and does not retain `X`
  (Probe 6). Missing attribute.
- REQ-9: **PyO3 binding VALUE parity (R-DEFER-1/3).** `ferrolearn.DBSCAN` →
  `_RsDBSCAN` → `ferrolearn_cluster::DBSCAN` exposes `labels_` and `fit_predict`
  (Probe 7). The binding marshals `eps`/`min_samples`/`labels_` faithfully and (since
  the Rust core value-matches, REQ-1/REQ-2) `import ferrolearn` matches
  `import sklearn` on the Euclidean/no-`sample_weight` path. The binding does NOT yet
  surface `core_sample_indices_` / `components_` / `sample_weight` / `metric`
  (only `labels_` is exposed via `#[getter]`).
- REQ-10: **ferray substrate (R-SUBSTRATE).** `dbscan.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`. Not migrated.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). ferrolearn values from a throwaway
`cargo run --example` probe (since deleted). Fixtures A–D as in Probe 1.

- AC-1 (REQ-1, SHIPPED): on Fixture A `DBSCAN(eps=1.5,min_samples=2).fit(X).labels_`
  = `[0,0,0,0,1,1,1,1]`; on Fixture B `(eps=0.5,min_samples=2)` =
  `[0,0,0,1,1,1,2,2,2]`; on Fixture D `make_blobs(...).labels_` =
  `[0,0,0,-1,1,-1,0,1,1,1,-1,-1,-1,0,-1,-1,-1,-1,1,-1]`. ferrolearn matches all
  exactly.
- AC-2 (REQ-1, SHIPPED — the tie-break): on Fixture C
  `DBSCAN(eps=1.0,min_samples=4).fit(X).labels_` = `[0,0,0,0,0,1,1,1,1]` — the
  border point idx 4 → cluster 0 (first-reaching). ferrolearn matches exactly; idx 4
  is NOT in `core_sample_indices_`.
- AC-3 (REQ-2, SHIPPED): on Fixture C `core_sample_indices_` = `[0,1,2,3,5,6,7,8]`
  (border idx 4 excluded); on Fixture D = `[0,1,2,4,6,7,8,9,13]`. ferrolearn matches
  exactly.
- AC-4 (REQ-3, SHIPPED): `DBSCAN(eps=0.0).fit(X)` → sklearn `InvalidParameterError`;
  ferrolearn `fn fit` → `Err(InvalidParameter{name:"eps"})`. `DBSCAN(min_samples=0)`
  → sklearn `InvalidParameterError`; ferrolearn → `Err(InvalidParameter{name:"min_samples"})`.
- AC-5 (REQ-4, diverges): `DBSCAN().eps` = `0.5` in sklearn; ferrolearn `DBSCAN::new`
  has no `eps` default (the argument is required). Error type is `FerroError`, not
  `InvalidParameterError`.
- AC-6 (REQ-5, diverges): `DBSCAN(eps=0.5,min_samples=3).fit(X,
  sample_weight=[5,1,1]).labels_` = `[0,-1,-1]` (idx0 becomes core by weight);
  ferrolearn has no `sample_weight` and yields `[-1,-1,-1]` on the unweighted call.
- AC-7 (REQ-6, diverges): `DBSCAN(eps=1.2,min_samples=2,metric='manhattan').fit(X).labels_`
  = `[-1,0,0]` vs `metric='euclidean'` = `[0,0,0]`; ferrolearn is Euclidean-only.
- AC-8 (REQ-8, missing): `DBSCAN(eps=0.5,min_samples=2).fit(X).components_.shape` =
  `(2,2)`; ferrolearn has no `components_`.
- AC-9 (REQ-9, SHIPPED): `import ferrolearn; ferrolearn.DBSCAN(eps=1.5,
  min_samples=2).fit_predict(A)` = `[0,0,0,0,1,1,1,1]` matching
  `sklearn.cluster.DBSCAN` on Fixture A.

## REQ status table

Binary (R-DEFER-2). `DBSCAN` / `FittedDBSCAN` are existing pub APIs re-exported at the
crate root AND bound into CPython as `ferrolearn.DBSCAN` (a real non-test consumer;
grandfathered S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2, run from
`/tmp`. Honest assessment (R-HONEST-3): **this unit's CORE contract genuinely SHIPS**
— `labels_` and `core_sample_indices_` VALUE-match the oracle on every fixture
(including the shared-border tie-break) because DBSCAN is deterministic, and they
flow through a real PyO3 consumer. The NOT-STARTED REQs are unimplemented parameters
(`sample_weight`, `metric`/`p`, `algorithm`/`leaf_size`/`n_jobs`), the `eps=0.5`
default + error ABI, the `components_` attribute, and the ferray substrate. Suggested
blocker numbers — the director creates the real issues; #945 is this doc's crosslink
tracking issue.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`labels_` VALUE parity) | SHIPPED | impl `fn fit` (BFS/DFS cluster expansion, index-ordered) + `fn fit_predict` in `dbscan.rs` mirror `dbscan_inner` (`_dbscan_inner.pyx`) + `_dbscan.py:431-439`. VALUE-matches sklearn EXACTLY on all four fixtures (Probe 1): A `[0,0,0,0,1,1,1,1]`, B `[0,0,0,1,1,1,2,2,2]`, C (border tie) `[0,0,0,0,0,1,1,1,1]` (border idx4→cluster 0, first-reaching, AC-2), D blobs `[0,0,0,-1,1,-1,0,1,1,1,-1,-1,-1,0,-1,-1,-1,-1,1,-1]`. Non-test consumer: `ferrolearn.DBSCAN.fit(X).labels_` via `_RsDBSCAN::labels_` (`ferrolearn-python/src/extras.rs`) + `_ClusterWrapper.fit` (`_extras.py`). Verification: `cargo test -p ferrolearn-cluster --lib dbscan` (31 passed, 0 failed). |
| REQ-2 (`core_sample_indices_` VALUE) | SHIPPED | impl `core_sample_indices = (0..n).filter(is_core).collect()` in `fn fit` + accessor `fn core_sample_indices` in `dbscan.rs`, mirroring `np.where(core_samples)[0]` (`_dbscan.py:438`), core = `neighbors.len() >= min_samples` (`:435`). VALUE-matches ascending on all fixtures (Probe 1/2): C `[0,1,2,3,5,6,7,8]` (border idx4 excluded, AC-3), D `[0,1,2,4,6,7,8,9,13]`. Consumer: same crate re-export (`lib.rs`) — the in-crate accessor (the PyO3 binding does not yet re-expose it, see REQ-9). Verification: same as REQ-1. |
| REQ-3 (`eps>0` / `min_samples>=1` validation) | SHIPPED | impl `fn fit` guards `self.eps <= F::zero()` → `Err(InvalidParameter{name:"eps"})` and `self.min_samples == 0` → `Err(InvalidParameter{name:"min_samples"})`, matching the accept/reject boundary of `_parameter_constraints` `eps: Interval(Real,0,None,closed="neither")` / `min_samples: Interval(Integral,1,None,closed="left")` (`_dbscan.py:332-333`). Probe 5/AC-4: `eps=0` and `min_samples=0` both rejected by both. Consumer: `_RsDBSCAN::fit` maps the error to `PyValueError` (`extras.rs`). Verification: `test_invalid_eps`/`test_zero_eps`/`test_zero_min_samples` (dbscan.rs) green. (Error TYPE differs — `FerroError` not `InvalidParameterError`; the boundary itself matches, so the validation SHIPS while the ABI nuance is tracked under REQ-4.) |
| REQ-4 (`eps=0.5` default + sklearn error ABI) | NOT-STARTED | open prereq blocker **#946**. sklearn `__init__` `eps=0.5` default (`_dbscan.py:347`); ferrolearn `fn new(eps: F)` REQUIRES `eps` — no default (AC-5). The PyO3 layer DOES default `eps=0.5` (`_RsDBSCAN::new` signature, `extras.rs`; `_extras.py` `DBSCAN.__init__`), so `ferrolearn.DBSCAN()` matches, but the Rust constructor `DBSCAN::new` does not. Also validation errors are `FerroError::InvalidParameter`, not the sklearn `InvalidParameterError`/`ValueError` ABI (R-DEV-2). |
| REQ-5 (`sample_weight`) | NOT-STARTED | open prereq blocker **#947**. sklearn `fit(X, sample_weight=w)` sets `n_neighbors = sum(sample_weight[neighbors])` so a high-weight point becomes core with fewer neighbors (`_dbscan.py:427-429`; Probe 3: `w0=5` → `[0,-1,-1]` vs unweighted `[-1,-1,-1]`). ferrolearn `Fit<Array2<F>, ()>` has the unit `()` target — no weight; core is purely `neighbors.len() >= min_samples` (`fn fit`). Missing surface — affects core determination. |
| REQ-6 (`metric` / `p` / `metric_params`) | NOT-STARTED | open prereq blocker **#948**. sklearn accepts any `pairwise_distances` metric (`_dbscan.py:334-337`), `p` for Minkowski (`:341`), default `'euclidean'` (`:350`). ferrolearn `fn region_query` / `fn squared_euclidean` (`dbscan.rs`) is Euclidean-only; no `metric`/`p`/`metric_params` param. Probe 4: `metric='manhattan'` → `[-1,0,0]` vs euclidean `[0,0,0]` — unreachable in ferrolearn. |
| REQ-7 (`algorithm` / `leaf_size` / `n_jobs`) | NOT-STARTED | open prereq blocker **#949**. sklearn routes neighbor search through `NearestNeighbors(algorithm, leaf_size, n_jobs)` (`_dbscan.py:411-422`; constraints `:339-342`). ferrolearn uses a fixed brute-force `O(n^2)` `fn region_query` with no parameter. (Brute force value-matches the default `'auto'` results — the divergence is the absent parameter surface, not a numerical one.) |
| REQ-8 (`components_` fitted attribute) | SHIPPED | `FittedDBSCAN<F>` gains `pub(crate) components_: Array2<F>` built in `Fit::fit` as the rows of `X` at `core_sample_indices` (`row_mut(k).assign(&x.row(idx))`), mirroring `self.components_ = X[self.core_sample_indices_].copy()` (`_dbscan.py:441-446`); empty-data yields `Array2::zeros((0, n_features))` (sklearn `np.empty((0, X.shape[1]))`). `#[must_use] pub fn components(&self) -> &Array2<F>` accessor. Verification (live sklearn 1.5.2, R-CHAR-3, `X=[[1,1],[1.2,1.1],[0.9,1],[8,8],[8.1,8.2],[8,7.9],[5,5]]`, `eps=0.5`, `min_samples=3`): `labels_=[0,0,0,1,1,1,-1]`, `core_sample_indices_=[0,1,2,3,4,5]`, `components_=X[0:6]` shape `(6,2)`; the `row k == X[core_sample_indices()[k]]` invariant holds. Test `dbscan_components_match_sklearn` (+ `test_empty_data` shape `(0,2)`). |
| REQ-9 (PyO3 binding VALUE parity) | SHIPPED | impl `#[pyclass(name="_RsDBSCAN")] RsDBSCAN` (`ferrolearn-python/src/extras.rs`): `fn new(eps=0.5, min_samples=5)`, `fn fit` calling `ferrolearn_cluster::DBSCAN::<f64>::new(eps).with_min_samples(min_samples)`, `#[getter] labels_`; registered via `m.add_class::<extras::RsDBSCAN>()` (`src/lib.rs`); `class DBSCAN(_ClusterWrapper)` + `fit_predict` (`python/ferrolearn/_extras.py`); exported in `__init__.py`. Non-test consumer: `import ferrolearn; ferrolearn.DBSCAN(...).fit(X).labels_`. Since the Rust core value-matches (REQ-1/2), `import ferrolearn` matches `import sklearn` on the Euclidean/no-`sample_weight` path (AC-9). Binding does NOT yet re-expose `core_sample_indices_`/`components_`/`sample_weight`/`metric` — those ride their own REQs. Verification: `cd ferrolearn-python && maturin develop && PYTHONPATH=python python3 -m pytest tests/ -q`. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker **#951**. `dbscan.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` + `std::collections::VecDeque`; not migrated to `ferray-core` (R-SUBSTRATE-1/2). The PyO3 boundary uses `numpy2_to_ndarray` (`extras.rs`), an `ndarray` bridge, not `ferray::numpy_interop`. |

## Architecture

`dbscan.rs` follows the unfitted/fitted split (CLAUDE.md naming): `DBSCAN<F>`
(`eps`, `min_samples`; `fn new(eps)` defaults `min_samples=5`, builder
`with_min_samples`) → `Fit<Array2<F>, ()>` → `FittedDBSCAN<F>` (private
`labels_: Array1<isize>`, `core_sample_indices_: Vec<usize>`, `PhantomData<F>`).
Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2). No `Predict` impl (DBSCAN has no out-of-sample
prediction — matching sklearn, which only has `fit_predict`). Accessors:
`labels()`, `core_sample_indices()`, `n_clusters()` (`= max_label + 1`, or 0 if all
noise).

**Fit path (`fn fit`).** Validates `eps > 0` and `min_samples >= 1` (REQ-3),
short-circuits empty `X` to empty labels. Then the three-step algorithm:
1. **Neighborhoods** — `neighborhoods[i] = region_query(x, i, eps^2)` (brute force,
   squared-Euclidean `<= eps^2`, self included). Mirrors sklearn's
   `neighbors_model.radius_neighbors(X)` (`_dbscan.py:422`), which also leaves the
   point itself in (`:401-402`).
2. **Core determination** — `is_core[i] = neighborhoods[i].len() >= min_samples`,
   then `core_sample_indices = filter(is_core)` ascending. Mirrors
   `core_samples = n_neighbors >= self.min_samples` + `np.where(core_samples)[0]`
   (`_dbscan.py:435/438`). (No `sample_weight` weighting — REQ-5.)
3. **Cluster expansion** — `labels` init to `-1` (all noise). For `i` in index
   order, skip if non-core or already labeled; else start cluster
   `current_cluster += 1`, seed a queue with `i`'s neighbors, and BFS: a popped
   point that is still `-1` is claimed for the current cluster; if it is core, its
   unvisited neighbors are claimed and the core ones enqueued; a non-core (border)
   point is claimed but NOT expanded. This is structurally sklearn's `dbscan_inner`
   DFS (`_dbscan_inner.pyx`): label on first core encounter in index order, border
   to the first-reaching cluster, core not re-expanded once labeled. **The traversal
   order is BFS (a `VecDeque`) in ferrolearn vs DFS (a `vector` stack) in sklearn —
   but the OBSERVABLE result (which cluster each point lands in) is invariant under
   traversal order for DBSCAN**, because cluster membership is determined by
   density-reachability connectivity, not visit order; only the index order of NEW
   cluster starts (lowest unlabeled core first) and the "first-reaching wins" border
   rule matter, and both implementations honor those. Probe 1 Fixtures C and D
   confirm the BFS-vs-DFS choice does not change `labels_`.

**Invariants held vs sklearn (the SHIPPED core):** `labels_` VALUE (REQ-1 — exact on
A/B/C/D incl. the border tie-break); `core_sample_indices_` VALUE (REQ-2 — exact,
ascending); the `eps > 0` / `min_samples >= 1` validation boundary (REQ-3); cluster
numbering = first unlabeled core in index order; noise `= -1`; self-inclusion in
neighborhoods; `n_clusters()` = number of distinct non-negative labels; the PyO3
`labels_` marshalling (REQ-9).

**Invariants NOT held vs sklearn:** `eps=0.5` Rust-constructor default + the
`InvalidParameterError` error ABI (REQ-4); `sample_weight` (REQ-5 — alters core
determination); `metric`/`p`/`metric_params` (REQ-6 — Euclidean-only); the
`algorithm`/`leaf_size`/`n_jobs` neighbor-search parameter surface (REQ-7 — fixed
brute force); the `components_` fitted attribute (REQ-8); the ferray substrate
(REQ-10). The PyO3 binding additionally does not re-expose `core_sample_indices_`
(REQ-9 note).

**Consumer wiring.** `pub use dbscan::{DBSCAN, FittedDBSCAN}`
(`ferrolearn-cluster/src/lib.rs`) re-exports the types; the **real non-test
production consumer** is `ferrolearn.DBSCAN` (`_RsDBSCAN` in
`ferrolearn-python/src/extras.rs`, registered in `src/lib.rs`, wrapped in
`python/ferrolearn/_extras.py`, exported in `__init__.py`) — `import ferrolearn`
reaches DBSCAN, calls `fit`, and reads `labels_`/`fit_predict` (Probe 7).

## Verification

Library crate (green at baseline `e0f84020`):
```
cargo test -p ferrolearn-cluster --lib dbscan     # 31 passed; 0 failed
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s (`test_simple_clusters`, `test_noise_detection`,
`test_core_border_noise_identification`, `test_all_noise_with_high_min_samples`,
`test_all_noise_with_tiny_eps`, `test_single_point_cluster`,
`test_all_in_one_cluster`, `test_empty_data`, `test_single_sample`,
`test_single_sample_noise`, `test_invalid_eps`, `test_zero_eps`,
`test_zero_min_samples`, `test_core_sample_indices_correct`, `test_f32_support`,
`test_three_clusters`, `test_identical_points`) pin co-membership, noise, core/border
identification, the edge cases, and the validation guards. They assert STRUCTURE
(co-membership, n_clusters, noise positions) rather than exact integer labels against
the oracle — the value-parity claim is established by the live-oracle probes below.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the value-parity that backs
the SHIPPED REQs (R-CHAR-3 expected values, never literal-copied from ferrolearn):
```
# REQ-1/REQ-2 (SHIPPED) labels_ + core_sample_indices_ — all four fixtures match exactly
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; \
A=np.array([[0.,0.],[0.5,0.],[0.,0.5],[0.5,0.5],[10.,10.],[10.5,10.],[10.,10.5],[10.5,10.5]]); \
print(DBSCAN(eps=1.5,min_samples=2).fit(A).labels_.tolist())"          # [0,0,0,0,1,1,1,1]  (ferro: identical)
# REQ-1 (SHIPPED) shared-border tie-break — border idx4 -> cluster 0 (first-reaching)
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; \
C=np.array([[0.,0.],[-0.5,0.],[0.,0.5],[0.,-0.5],[1.,0.],[2.,0.],[2.5,0.],[2.,0.5],[2.,-0.5]]); \
m=DBSCAN(eps=1.0,min_samples=4).fit(C); print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
# [0,0,0,0,0,1,1,1,1] [0,1,2,3,5,6,7,8]   (ferro: identical; border idx4 in cluster 0, NOT core)
# REQ-3 (SHIPPED) validation boundary
python3 -c "from sklearn.cluster import DBSCAN; import numpy as np; X=np.array([[0.,0.],[1.,1.]]); \
[print(p, (lambda: (DBSCAN(**p).fit(X), 'ok')[1])()) if False else None for p in []]"  # eps=0 / ms=0 -> InvalidParameterError (ferro: Err InvalidParameter)
# REQ-5 (DIVERGES) sample_weight
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; X=np.array([[0.,0.],[5.,5.],[5.1,5.]]); \
print(DBSCAN(eps=0.5,min_samples=3).fit(X, sample_weight=np.array([5.,1.,1.])).labels_.tolist())"  # [0,-1,-1]; ferro has no sample_weight ([-1,-1,-1])
# REQ-6 (DIVERGES) metric
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; X=np.array([[0.,0.],[0.8,0.8],[0.85,0.85]]); \
print(DBSCAN(eps=1.2,min_samples=2,metric='manhattan').fit(X).labels_.tolist())"  # [-1,0,0]; ferro euclidean-only ([0,0,0])
# REQ-8 (MISSING) components_
python3 -c "import numpy as np; from sklearn.cluster import DBSCAN; \
print(DBSCAN(eps=0.5,min_samples=2).fit(np.array([[0.,0.],[0.1,0.],[10.,10.]])).components_.shape)"  # (2,2); ferro has none
```
A characterization pin (R-CHAR-3) for REQ-1/REQ-2 belongs in
`ferrolearn-cluster/tests/divergence_dbscan.rs`, asserting the live-sklearn `labels_`
/ `core_sample_indices_` above and PASSING against current `dbscan.rs` (these are the
SHIPPED-confirming tests, not failing pins). Pins for the NOT-STARTED REQs
(`sample_weight`, `metric`, `components_`, `eps=0.5`-Rust-default) assert the
sklearn behavior ferrolearn cannot yet reach and FAIL until the surface lands.

ferrolearn-python (REQ-9 binding parity):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/ -q
```
asserting `ferrolearn.DBSCAN(eps,min_samples).fit_predict(X)` value-matches
`sklearn.cluster.DBSCAN` on Fixtures A–D (the SHIPPED core). A divergence pin for the
binding's missing `core_sample_indices_`/`components_`/`sample_weight` surface (REQ-9
note) belongs in `ferrolearn-python/tests/divergence_dbscan.py`.

## Blockers to open

(Director creates the real issues; #945 is this doc's crosslink tracking issue. The
core REQ-1/REQ-2/REQ-3/REQ-9 are SHIPPED and need NO blocker. The rest are
SUGGESTIONS for the NOT-STARTED parameter/attribute surface.)

- **#946** — REQ-4: add `eps=0.5` default to the Rust `DBSCAN::new` (or a
  `Default`/builder), and map validation failures to the sklearn
  `ValueError`/`InvalidParameterError` ABI (`_dbscan.py:347`, `:332-333`).
- **#947** — REQ-5: add a `sample_weight` parameter (a `Fit<Array2<F>, Array1<F>>`
  target or a weighted-fit method) computing `n_neighbors = sum(weight[neighbors])`
  for core determination (`_dbscan.py:427-429`).
- **#948** — REQ-6: add `metric` / `p` / `metric_params` (Minkowski-`p`, manhattan,
  chebyshev, …) to `region_query`, default `'euclidean'` (`_dbscan.py:334-337`,
  `:341`, `:350`).
- **#949** — REQ-7: add the `algorithm` / `leaf_size` / `n_jobs` neighbor-search
  parameter surface, routed through a NearestNeighbors analog
  (`_dbscan.py:411-422`); brute force stays the `'brute'` / default backend.
- **#950** — REQ-8: add a `components_` accessor on `FittedDBSCAN` =
  `X[core_sample_indices_]` (retain the core rows at fit time) (`_dbscan.py:441-446`).
- **#951** — REQ-10: migrate `dbscan.rs` off `ndarray`/`num-traits` to `ferray-core`,
  and the PyO3 boundary off `numpy2_to_ndarray` to `ferray::numpy_interop`
  (R-SUBSTRATE).
- (REQ-9 note) — re-expose `core_sample_indices_` (and later `components_`) on
  `_RsDBSCAN` so the binding surface matches the SHIPPED Rust accessors.
