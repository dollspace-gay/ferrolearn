# HDBSCAN (sklearn.cluster.HDBSCAN)

<!--
tier: 3-component
status: draft
baseline-commit: 0fefae62
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_hdbscan/hdbscan.py   # class HDBSCAN(ClusterMixin, BaseEstimator) (:419); _parameter_constraints (:641-670); __init__ defaults min_cluster_size=5/min_samples=None/cluster_selection_epsilon=0.0/max_cluster_size=None/metric='euclidean'/alpha=1.0/algorithm='auto'/leaf_size=40/cluster_selection_method='eom'/allow_single_cluster=False/store_centers=None (:672-702); fit (:708); core distance = neighbors_distances[:,-1] of min_samples NN, "includes the point itself" (_hdbscan_prims :340-352, docstring :443-446); n_samples==1 raises ValueError (:788); _weighted_cluster_center -> centroids_/medoids_ (:940-982); labels_/probabilities_ attrs (:547-564); n_features_in_/centroids_/medoids_ attrs (:566-590)
  - sklearn/cluster/_hdbscan/_tree.pyx    # condensed tree + EOM/leaf selection; _compute_stability stability = Sum_p (lambda_val - births[parent]) * cluster_size (:240-278); max_lambdas deaths[cluster] (:301-325); get_probabilities result[point] = min(lambda_array[n], max_lambda) / max_lambda (GLOSH membership) (:524-552); tree_to_labels / _do_labelling (:646+)
  - sklearn/cluster/_hdbscan/_reachability.pyx  # mutual_reachability_graph: mr(i,j) = max(core_dist[i], core_dist[j], d(i,j))
  - sklearn/cluster/_hdbscan/_linkage.pyx       # mst_from_data_matrix (Prim over data) / mst_from_mutual_reachability; make_single_linkage (dendrogram)
parity-ops: HDBSCAN (.__init__, .fit, .fit_predict, .labels_, .probabilities_)
crosslink-issue: 1068
-->

## Summary

`ferrolearn-cluster/src/hdbscan.rs` mirrors scikit-learn's `HDBSCAN`
(`sklearn/cluster/_hdbscan/hdbscan.py`, `class HDBSCAN(ClusterMixin, BaseEstimator)`
`:419`) — a hierarchical, density-based clustering that performs DBSCAN over varying
epsilon and integrates the result to find the most-stable clustering, producing
varying-density clusters and a `-1` noise label. It exposes the unfitted `Hdbscan<F>`
(`min_cluster_size`, `min_samples`, `cluster_selection_epsilon`), the fitted
`FittedHdbscan<F>` (stores `labels_` + `probabilities_`, plus an `n_clusters()`
helper), a `Fit<Array2<F>, ()>` impl, and a `fit_predict` convenience mirroring
`ClusterMixin.fit_predict`. It is re-exported at the crate root (`pub use
hdbscan::{FittedHdbscan, Hdbscan}` in `ferrolearn-cluster/src/lib.rs`).

**This unit's load-bearing contract is the `labels_` PARTITION (co-membership up to
permutation) + the noise (`-1`) set on well-separated dense data — and that SHIPS,
VALUE-matching the live sklearn 1.5.2 oracle on all three probe fixtures (Probes 1–3
below).** HDBSCAN is fully deterministic — no RNG, no iterative optimizer — so the
partition is a pure function of the data; the partition match is genuine, not
coincidence. ferrolearn builds the same pipeline sklearn does: per-point core
distances → mutual-reachability MST (Prim, `fn build_mst`) → single-linkage
dendrogram → a condensed tree with an EOM-like (excess-of-mass) stability selection
(`fn extract_clusters`).

**The honest divergences (verified, NOT overclaimed):**

1. **`probabilities_` VALUES DIVERGE.** ferrolearn's membership probability is
   `(child_lambda - birth)/(death - birth)` (`fn extract_clusters`); sklearn's is the
   GLOSH membership `lambda_p / max_lambda_in_cluster` (`_tree.pyx` `get_probabilities`
   `:551-552`). The two formulas differ structurally (sklearn divides by `death`,
   not `death - birth`, and does not subtract `birth`). Probe 1/2 show distinct value
   vectors. NOT-STARTED.
2. **A confirmed core-distance OFF-BY-ONE.** sklearn's core distance for a point is the
   distance to its `min_samples`-th nearest neighbor *including itself* — i.e.
   `sorted_dists[min_samples - 1]` (`_hdbscan_prims` `:351-352`, the docstring states
   `min_samples` "includes the point itself" `:444-446`). ferrolearn's
   `fn compute_core_distances` sorts (self=0 first) and takes `dists[min_samples]` =
   `sorted_dists[min_samples]` — one neighbor TOO FAR. ferrolearn's `min_samples = k`
   reproduces sklearn's `min_samples = k + 1` core distances (Probe 4, decisive). This
   shifts probabilities and can shift the partition near density boundaries.
   NOT-STARTED.
3. **`exact integer labels_` numbering DIVERGE in general** (cluster-index assignment
   order differs from sklearn's `_do_labelling`); only the PARTITION up-to-permutation
   is matched.
4. **Missing params:** `metric`/`metric_params` (euclidean-only), `alpha`, `algorithm`
   (auto/brute/kd_tree/ball_tree), `leaf_size`, `n_jobs`, `max_cluster_size`,
   `allow_single_cluster`, `cluster_selection_method` (`'eom'`/`'leaf'` — ferrolearn
   does EOM-only, no `'leaf'`), `store_centers`. NOT-STARTED.
5. **Missing fitted attributes:** `centroids_`, `medoids_`, `n_features_in_`.
   NOT-STARTED. (Note: sklearn 1.5.2 HDBSCAN has **no** `cluster_persistence_`
   attribute — that is the third-party `scikit-learn-contrib/hdbscan` package, not
   sklearn; Probe 5 confirms. There is therefore nothing to mark NOT-STARTED for
   `cluster_persistence_`.)
6. **`cluster_selection_epsilon` semantics are a simplified approximation** of
   sklearn's DBSCAN-like flat-cut epsilon logic (`_tree.pyx`); ferrolearn's
   `above_epsilon = split_dist > eps` heuristic (`fn extract_clusters`) is not the
   sklearn `epsilon` cut. NOT-STARTED.
7. **Error ABI / `n_samples==1`:** sklearn raises `ValueError`
   ("n_samples=1 while HDBSCAN requires more than one sample", `:788`); ferrolearn
   returns a singleton-noise result for `n_samples=1`, not an error. The
   `min_cluster_size >= 2` validation matches `_parameter_constraints`
   (`Interval(Integral, 2, None, closed="left")`, `:642`) but raises
   `FerroError::InvalidParameter`, not `InvalidParameterError`. NOT-STARTED.
8. **No PyO3 binding.** A grep of `ferrolearn-python/` for `HDBSCAN`/`Hdbscan`/
   `RsHdbscan` returns nothing (Probe 6). The only non-test production consumer is the
   crate re-export `pub use hdbscan::{FittedHdbscan, Hdbscan}` (`lib.rs`) —
   grandfathered (S5 / R-DEFER-1); the PyO3-binding REQ is NOT-STARTED.
9. **ferray substrate.** `hdbscan.rs` imports `ndarray::{Array1, Array2}` +
   `num_traits::Float`, not `ferray-core`. NOT-STARTED.

## Live oracle probes (sklearn 1.5.2, run from /tmp; ferrolearn via a throwaway `tests/zz_hdbscan_probe.rs`, since deleted)

All expected sklearn values are from the installed sklearn 1.5.2 oracle run from
`/tmp` (R-CHAR-3); ferrolearn values are from a throwaway integration test that called
`Hdbscan::<f64>::new()....fit(&x, &())` on the identical fixtures and printed
`labels()` / `probabilities()` (`cargo test ... --nocapture`), then deleted.

### Probe 1 — `two_blobs` (the `hdbscan.rs` 12-point fixture), `min_cluster_size=3`

Two 6-point blobs near `(0,0)` and `(10,10)`.
```
python3 -c "import numpy as np; from sklearn.cluster import HDBSCAN; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05],[-0.05,0.05], \
[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1],[10.05,10.05],[9.95,10.05]]); \
m=HDBSCAN(min_cluster_size=3).fit(X); print(m.labels_.tolist()); print([round(p,3) for p in m.probabilities_])"
# sklearn:    labels [0,0,0,0,0,0,1,1,1,1,1,1]   probs [1.0,0.707,1.0,0.707,1.0,1.0, 1.0,0.707,1.0,0.707,1.0,1.0]
# ferrolearn: labels [0,0,0,0,0,0,1,1,1,1,1,1]   probs [1.0,1.0,  1.0,1.0,  1.0,1.0, 1.0,1.0,  1.0,1.0,  1.0,1.0]
```
**PARTITION + noise: IDENTICAL** (first 6 → cluster A, last 6 → cluster B, no noise;
the integer labels also happen to coincide here, but see Probe 3 for why we only
claim partition-up-to-permutation). **`probabilities_` VALUES: DIVERGE** — sklearn
gives the diagonal corner points `0.707`, ferrolearn gives them `1.0`.

### Probe 2 — `dense_clusters` (the `hdbscan.rs` 20-point fixture), `min_cluster_size=3`, `min_samples=3`

Two 10-point dense blobs near `(0,0)` and `(5,5)`.
```
python3 -c "import numpy as np; from sklearn.cluster import HDBSCAN; \
X=np.array([[0.,0.],[0.05,0.],[0.,0.05],[0.05,0.05],[-0.05,0.],[0.,-0.05],[-0.05,-0.05],[0.03,0.02],[-0.02,0.03],[0.04,-0.01], \
[5.,5.],[5.05,5.],[5.,5.05],[5.05,5.05],[4.95,5.],[5.,4.95],[4.95,4.95],[5.03,5.02],[4.98,5.03],[5.04,4.99]]); \
m=HDBSCAN(min_cluster_size=3,min_samples=3).fit(X); print(m.labels_.tolist()); print([round(p,3) for p in m.probabilities_])"
# sklearn:    labels [0]*10 + [1]*10
#             probs  [0.877,1.0,0.745,0.632,0.632,0.632,0.632,1.0,0.877,1.0, 0.877,1.0,0.745,0.632,0.632,0.632,0.632,1.0,0.877,1.0]
# ferrolearn: labels [0]*10 + [1]*10
#             probs  [1.0,0.824,0.824,0.824,0.824,0.727,0.581,1.0,0.972,1.0, 1.0,0.824,0.824,0.824,0.824,0.727,0.581,1.0,0.972,1.0]
```
**PARTITION + noise: IDENTICAL** (points 0–9 → one cluster, 10–19 → another, no
noise). **`probabilities_` VALUES: DIVERGE** completely.

### Probe 3 — `noise_detection` (the `hdbscan.rs` 14-point fixture), `min_cluster_size=3`

Two 5-point blobs + four scattered outliers at `(50,50)`, `(-50,-50)`, `(100,0)`,
`(0,100)`.
```
python3 -c "import numpy as np; from sklearn.cluster import HDBSCAN; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05], \
[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1],[10.05,10.05], \
[50.,50.],[-50.,-50.],[100.,0.],[0.,100.]]); \
m=HDBSCAN(min_cluster_size=3).fit(X); print(m.labels_.tolist()); print([round(p,3) for p in m.probabilities_])"
# sklearn:    labels [0,0,0,0,0, 1,1,1,1,1, -1,-1,-1,-1]   probs [1,1,1,1,1, 1,1,1,1,1, 0,0,0,0]
# ferrolearn: labels [0,0,0,0,0, 1,1,1,1,1, -1,-1,-1,-1]   probs [1,1,1,1,1, 1,1,1,1,1, 0,0,0,0]
```
**PARTITION + noise set: IDENTICAL** — all four outliers labeled `-1` (prob `0`) by
BOTH, both blobs clustered. On THIS fixture the probability vectors also coincide
(all members at prob `1.0`), because the blobs are tight enough that every member
persists to the cluster death; this is fixture-specific and does NOT generalize
(Probes 1/2). The noise-detection + noise-prob-0 CONTRACT holds.

### Probe 4 — core-distance OFF-BY-ONE (decisive)

sklearn `_hdbscan_prims` takes `neighbors_distances[:, -1]` of the `min_samples`
nearest neighbors, and the docstring states `min_samples` "includes the point itself"
(`:351-352`, `:444-446`) → core distance = `sorted_dists[min_samples - 1]`. ferrolearn
`fn compute_core_distances` sorts with self at index 0 and takes `dists[min_samples]`
→ `sorted_dists[min_samples]`, one neighbor further out.
```
python3 -c "import numpy as np; from sklearn.neighbors import NearestNeighbors; \
X=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[10.,0.]]); ms=3; \
d,_=NearestNeighbors(n_neighbors=ms).fit(X).kneighbors(X,ms); print('sklearn core (ms=3):', d[:,-1].tolist())"
# sklearn core (ms=3): [2.0, 1.0, 1.0, 2.0, 8.0]   = sorted_dists[ms-1]
# ferrolearn fn compute_core_distances ms=3 -> sorted_dists[3] = [3.0, 2.0, 2.0, 3.0, 9.0]   (= sklearn ms=4)
```
And the off-by-one is observable in the output — ferrolearn's two_blobs probabilities
(`min_cluster_size=3`, internal `min_samples=3`) equal sklearn's at `min_samples=4`:
```
python3 -c "import numpy as np; from sklearn.cluster import HDBSCAN; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05],[-0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1],[10.05,10.05],[9.95,10.05]]); \
print('ms=3', [round(p,3) for p in HDBSCAN(min_cluster_size=3,min_samples=3).fit(X).probabilities_]); \
print('ms=4', [round(p,3) for p in HDBSCAN(min_cluster_size=3,min_samples=4).fit(X).probabilities_])"
# ms=3 [1.0,0.707,1.0,0.707,1.0,1.0, ...]
# ms=4 [1.0,1.0,  1.0,1.0,  1.0,1.0, ...]   <- equals ferrolearn's ms=3 output
```
This is the cleanest minimal fix available (change `dists[min_samples]` to
`dists[min_samples - 1]` in `fn compute_core_distances`) — but on its own it does NOT
make probabilities VALUE-match, because the probability FORMULA also diverges
(Probe 1/2). The partition is robust to it on well-separated data (Probes 1–3 still
match the partition).

### Probe 5 — fitted attributes + defaults

```
python3 -c "from sklearn.cluster import HDBSCAN; import numpy as np; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05],[-0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1],[10.05,10.05],[9.95,10.05]]); \
m=HDBSCAN(min_cluster_size=3,store_centers='both').fit(X); \
print('attrs:', [a for a in dir(m) if a.endswith('_') and not a.startswith('_')]); \
print('n_features_in_', m.n_features_in_); print('centroids_', m.centroids_.tolist()); print('medoids_', m.medoids_.tolist())"
# attrs: ['centroids_', 'labels_', 'medoids_', 'n_features_in_', 'probabilities_']   (NO cluster_persistence_)
# n_features_in_ 2
# centroids_ [[0.0261..., 0.05], [10.0261..., 10.05]]
# medoids_   [[0.05, 0.05], [10.05, 10.05]]
python3 -c "from sklearn.cluster import HDBSCAN; h=HDBSCAN(); \
print(h.min_cluster_size, h.min_samples, h.metric, h.alpha, h.cluster_selection_method, h.cluster_selection_epsilon, h.algorithm, h.leaf_size, h.max_cluster_size, h.allow_single_cluster, h.store_centers)"
# 5 None euclidean 1.0 eom 0.0 auto 40 None False None
```
- **Defaults `min_cluster_size=5`, `min_samples=None`, `cluster_selection_epsilon=0.0`**
  — ferrolearn `fn new` matches all three (`:75-81`, `test_default_constructor`).
- **`centroids_`/`medoids_`/`n_features_in_` exist; `cluster_persistence_` does NOT** —
  ferrolearn exposes neither `centroids_`/`medoids_` nor `n_features_in_` (only
  `labels()`, `probabilities()`, `n_clusters()`). The `n_clusters()` helper has no
  sklearn analog (sklearn derives `n_clusters` from `labels_`), but it is not a
  divergence — it is an extra convenience.

### Probe 6 — non-test consumer / PyO3 binding

```
grep -rni "hdbscan" ferrolearn-python/      # -> (empty)
grep -rn  "Hdbscan" ferrolearn-cluster/src/lib.rs
# ferrolearn-cluster/src/lib.rs:108:pub use hdbscan::{FittedHdbscan, Hdbscan};
```
**There is NO PyO3 binding for HDBSCAN** (no `_RsHdbscan`, no registration, no Python
wrapper). The only non-test production consumer is the crate re-export `pub use
hdbscan::{FittedHdbscan, Hdbscan}` (`lib.rs`). `Hdbscan`/`FittedHdbscan` are existing
pub APIs (boundary estimator types) grandfathered under S5 / R-DEFER-1. The
PyO3-binding REQ (REQ-9) is therefore NOT-STARTED.

## Requirements

- REQ-1: **`labels_` PARTITION (co-membership up-to-permutation) + noise set on
  well-separated data (R-DEV-1/3 — the core contract).** Mirror `HDBSCAN.fit().labels_`
  (`hdbscan.py:883` + the EOM selection in `_tree.pyx`): the same points co-cluster,
  the same points are noise (`-1`). ferrolearn `fn fit` (Prim MST → single-linkage
  dendrogram → `fn extract_clusters` EOM selection) produces a partition + noise set
  bit-identical to the oracle on all three fixtures (Probes 1–3). Consumed by
  `pub use hdbscan::{FittedHdbscan, Hdbscan}` (`lib.rs`).
- REQ-2: **`probabilities_` shape/range CONTRACT + noise-prob-0 (R-DEV-3).** Mirror
  `probabilities_` ∈ `[0, 1]`, `probabilities_[noise] == 0` (`hdbscan.py:556-562`).
  ferrolearn `fn extract_clusters` clamps to `[0,1]` and assigns `F::zero()` to noise
  (verified by `test_probabilities_range`; Probes 1–3 all show range + noise-0 held).
- REQ-3: **defaults + `min_cluster_size >= 2` validation (R-DEV-2).** Mirror
  `__init__` defaults `min_cluster_size=5`/`min_samples=None`/
  `cluster_selection_epsilon=0.0` (`:674-676`) and the constraint
  `min_cluster_size: Interval(Integral, 2, None, closed="left")` (`:642`). ferrolearn
  `fn new` defaults match (Probe 5); `fn fit` rejects `min_cluster_size < 2`
  (`test_invalid_min_cluster_size`).
- REQ-4: **`probabilities_` VALUE parity (GLOSH membership) (R-DEV-1).** sklearn
  `result[point] = min(lambda_array[n], max_lambda) / max_lambda` (`_tree.pyx`
  `:551-552`); ferrolearn uses `(child_lambda - birth)/(death - birth)`
  (`fn extract_clusters`). The formulas diverge → values diverge (Probes 1/2).
  Missing.
- REQ-5: **core-distance convention (R-DEV-1 — off-by-one).** sklearn core distance =
  `sorted_dists[min_samples - 1]` (`min_samples` includes self, `:351-352`/`:444-446`);
  ferrolearn `fn compute_core_distances` takes `sorted_dists[min_samples]`, one
  neighbor too far (Probe 4). ferrolearn `min_samples=k` == sklearn `min_samples=k+1`.
  Diverges.
- REQ-6: **exact `labels_` integer numbering (R-DEV-3).** sklearn assigns cluster ids
  in `_do_labelling` order (`_tree.pyx:646+`); ferrolearn assigns ids by condensed
  cluster discovery order (`fn extract_clusters`, `label_counter`). Only the partition
  up-to-permutation is matched, not the exact integers in general. Diverges.
- REQ-7: **`cluster_selection_method` (`'eom'`/`'leaf'`) + `allow_single_cluster` +
  `max_cluster_size` (R-DEV-2).** sklearn supports `'eom'` (default) and `'leaf'`
  selection, `allow_single_cluster`, and `max_cluster_size` (`:666-667`, `:647-650`).
  ferrolearn does an EOM-like selection only (`fn extract_clusters`), with no `'leaf'`
  mode and no param for any of these. Missing surface.
- REQ-8: **`metric`/`metric_params`/`alpha`/`algorithm`/`leaf_size`/`n_jobs`
  (R-DEV-2).** sklearn supports any `pairwise_distances` metric + `'precomputed'`
  (`:651-654`), `alpha` distance scaling (`:656`), `algorithm`
  (auto/brute/kd_tree/ball_tree, `:658-663`), `leaf_size` (`:664`), `n_jobs` (`:665`).
  ferrolearn is euclidean-only with a fixed `O(n^2)` Prim MST (`fn build_mst`,
  `fn sq_euclidean`); none of these params exist. Missing surface.
- REQ-9: **PyO3 binding (R-DEFER-1).** There is NO `ferrolearn.HDBSCAN` binding
  (Probe 6). The non-test consumer is the crate re-export only. Missing.
- REQ-10: **`centroids_`/`medoids_`/`n_features_in_` fitted attributes + `store_centers`
  (R-DEV-3).** sklearn `_weighted_cluster_center` stores `centroids_` (weighted
  average) and `medoids_` (min-total-distance point) when `store_centers` is set
  (`:940-982`), and always stores `n_features_in_` (`:566-567`). ferrolearn exposes
  none (Probe 5). Missing attributes. (No `cluster_persistence_` in sklearn 1.5.2 —
  nothing to add for it.)
- REQ-11: **error ABI + `n_samples==1` (R-DEV-2).** sklearn raises `ValueError` for
  `n_samples==1` (`:788`) and `InvalidParameterError` for constraint violations;
  ferrolearn returns a singleton-noise result for `n_samples==1` (`test_single_point`)
  and raises `FerroError::InvalidParameter` for `min_cluster_size < 2`. The validation
  boundary matches; the error TYPE and the `n_samples==1` behavior diverge.
- REQ-12: **`cluster_selection_epsilon` exact semantics (R-DEV-1).** sklearn applies a
  DBSCAN-like flat-cut epsilon merge (`_tree.pyx`); ferrolearn's `above_epsilon =
  split_dist > eps` heuristic (`fn extract_clusters`) is a simplified approximation,
  not the sklearn epsilon cut. Diverges.
- REQ-13: **ferray substrate (R-SUBSTRATE).** `hdbscan.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`. Not migrated.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). ferrolearn values from a throwaway probe
integration test (since deleted). Fixtures as in Probes 1–3.

- AC-1 (REQ-1, SHIPPED): on `two_blobs` `HDBSCAN(min_cluster_size=3).fit(X).labels_`
  partitions as `{0..5}` / `{6..11}` with no noise; on `dense` `(mcs=3,ms=3)` as
  `{0..9}` / `{10..19}`; on `noise` `(mcs=3)` as `{0..4}` / `{5..9}` with `{10..13}` →
  `-1`. ferrolearn's partition + noise set match each exactly (Probes 1–3).
- AC-2 (REQ-2, SHIPPED): on all three fixtures every `probabilities_[i] ∈ [0,1]`, and
  `probabilities_[i] == 0` for every `labels_[i] == -1` (Probe 3: outliers prob 0).
  ferrolearn holds both (`test_probabilities_range`).
- AC-3 (REQ-3, SHIPPED): `HDBSCAN()` exposes `min_cluster_size=5`, `min_samples=None`,
  `cluster_selection_epsilon=0.0` (Probe 5); ferrolearn `Hdbscan::new()` matches.
  `HDBSCAN(min_cluster_size=1)` → sklearn `InvalidParameterError`; ferrolearn `fit`
  with `min_cluster_size=1` → `Err(InvalidParameter)`.
- AC-4 (REQ-4, diverges): on `two_blobs` `(mcs=3)` sklearn `probabilities_` =
  `[1,0.707,1,0.707,1,1, 1,0.707,1,0.707,1,1]`; ferrolearn = all `1.0`.
- AC-5 (REQ-5, diverges): for `X=[[0,0],[1,0],[2,0],[3,0],[10,0]]`, `min_samples=3`,
  sklearn core distances `[2,1,1,2,8]` (`sorted_dists[ms-1]`); ferrolearn computes
  `[3,2,2,3,9]` (`sorted_dists[ms]`) — off by one neighbor (Probe 4).
- AC-6 (REQ-7, missing): `HDBSCAN(cluster_selection_method='leaf')` yields a finer
  clustering than `'eom'`; ferrolearn has no `cluster_selection_method` param.
- AC-7 (REQ-8, missing): `HDBSCAN(metric='manhattan')` changes the MST/labels;
  ferrolearn is euclidean-only.
- AC-8 (REQ-9, missing): `import ferrolearn; ferrolearn.HDBSCAN` raises
  `AttributeError` (no binding); `import sklearn; sklearn.cluster.HDBSCAN` exists.
- AC-9 (REQ-10, missing): `HDBSCAN(store_centers='both').fit(X)` exposes `centroids_`
  shape `(n_clusters, n_features)` and `medoids_`, plus `n_features_in_` (Probe 5);
  ferrolearn `FittedHdbscan` exposes none.
- AC-10 (REQ-11, diverges): `HDBSCAN().fit(np.array([[0.,0.]]))` → sklearn
  `ValueError`; ferrolearn `fit` on a 1-row array returns labels `[-1]` (no error).

## REQ status table

Binary (R-DEFER-2). `Hdbscan`/`FittedHdbscan` are existing pub APIs re-exported at the
crate root (the boundary estimator types ARE the public API; grandfathered S5 /
R-DEFER-1) — there is **no** PyO3 binding (Probe 6), so the sole non-test consumer is
the crate re-export. Cites use symbol anchors (ferrolearn) / `file:line` (sklearn
1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2, run from `/tmp`. Honest
assessment (R-HONEST-3): **the PARTITION + noise contract genuinely SHIPS** — the
co-membership and noise set VALUE-match the oracle on all three fixtures because
HDBSCAN is deterministic; the probability shape/range/noise-0 contract, the
`min_cluster_size>=2` validation, and (fixed iter 130 / #1070) the core-distance
index also SHIP. The NOT-STARTED REQs are the probability VALUES (different GLOSH
formula, #1069), exact label integers, the missing parameter surface, the missing
fitted attributes, the error ABI, the epsilon semantics, the PyO3 binding, and the
ferray substrate. Blocker numbers below are the real filed issues (#1069–#1078).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`labels_` PARTITION + noise) | SHIPPED | impl `fn fit` (core dists → `fn build_mst` Prim MST → `fn extract_clusters` single-linkage + EOM selection) + `fn fit_predict` in `hdbscan.rs` mirror `HDBSCAN.fit` (`hdbscan.py:883`) + `_tree.pyx` EOM. PARTITION + noise set VALUE-match the oracle EXACTLY on all three fixtures (Probes 1–3): two_blobs `{0..5}/{6..11}`, dense `{0..9}/{10..19}`, noise `{0..4}/{5..9}` + `{10..13}`→`-1`. HDBSCAN is deterministic (no RNG), so this is genuine value-parity of the partition. Non-test consumer: `pub use hdbscan::{FittedHdbscan, Hdbscan}` (`ferrolearn-cluster/src/lib.rs`). Verification: `cargo test -p ferrolearn-cluster --lib hdbscan` (14 passed, 0 failed). |
| REQ-2 (`probabilities_` range + noise-0) | SHIPPED | impl `fn extract_clusters` clamps `prob.clamp(0.0,1.0)` and leaves noise points at `F::zero()` in `hdbscan.rs`, mirroring `probabilities_` ∈ `[0,1]` with noise = `0` (`hdbscan.py:556-562`). Verified by `test_probabilities_range` and Probes 1–3 (every value in range; all four outliers in Probe 3 at prob `0`). Consumer: crate re-export (`lib.rs`). Verification: same as REQ-1. |
| REQ-3 (defaults + `min_cluster_size>=2` validation) | SHIPPED | impl `fn new` defaults `min_cluster_size=5`/`min_samples=None`/`cluster_selection_epsilon=F::zero()` (matches `__init__` `hdbscan.py:674-676`, Probe 5); `fn fit` guards `self.min_cluster_size < 2` → `Err(FerroError::InvalidParameter)`, matching `Interval(Integral, 2, None, closed="left")` (`:642`). Verified by `test_default_constructor`, `test_invalid_min_cluster_size`. Consumer: crate re-export. (Error TYPE differs — `FerroError` not `InvalidParameterError`; tracked under REQ-11.) |
| REQ-4 (`probabilities_` VALUE parity) | NOT-STARTED | open prereq blocker **#1069**. sklearn GLOSH membership `result[point] = min(lambda_array[n], max_lambda) / max_lambda` (`_tree.pyx:551-552`, `deaths = max_lambdas(...)` `:534`); ferrolearn `(child_lambda - birth)/(death - birth)` (`fn extract_clusters`). Different formula → AC-4: two_blobs sklearn `[1,0.707,1,0.707,...]` vs ferrolearn all `1.0`; dense diverges entirely (Probe 2). Partly downstream of REQ-5 (core-distance off-by-one). |
| REQ-5 (core-distance index matches sklearn) | SHIPPED | impl `fn compute_core_distances` now takes `sorted_dists[min_samples-1]` (`min_samples`-NN query includes self at index 0) = sklearn `core_distances = neighbors_distances[:, -1]` (`hdbscan.py:351-352`). Was `dists[min_samples]` (ferrolearn ms=k reproduced sklearn ms=k+1). Guard: `divergence_core_distance_off_by_one_partition` (min_samples-sensitive fixture: ferrolearn ms=3 now == sklearn ms=3). Fixed #1070. (Note: probability VALUE parity still NOT-STARTED #1069 — the GLOSH formula also diverges.) |
| REQ-6 (exact `labels_` integers) | NOT-STARTED | open prereq blocker **#1071**. sklearn assigns cluster ids in `_do_labelling`/`tree_to_labels` order (`_tree.pyx:646+`); ferrolearn assigns by condensed-cluster discovery order (`fn extract_clusters`, `label_counter`). Only the partition up-to-permutation matches (REQ-1); exact integers are not guaranteed for >2 clusters or different traversal orders. |
| REQ-7 (`cluster_selection_method`/`allow_single_cluster`/`max_cluster_size`) | NOT-STARTED | open prereq blocker **#1072**. sklearn `cluster_selection_method: StrOptions({"eom","leaf"})` default `"eom"` (`:666`), `allow_single_cluster` (`:667`), `max_cluster_size` (`:647-650`). ferrolearn `fn extract_clusters` does an EOM-like selection only — no `'leaf'` mode, no param for any of the three. Missing surface (AC-6). |
| REQ-8 (`metric`/`alpha`/`algorithm`/`leaf_size`/`n_jobs`) | NOT-STARTED | open prereq blocker **#1073**. sklearn `metric` + `'precomputed'` (`:651-654`), `alpha` (`:656`), `algorithm` auto/brute/kd_tree/ball_tree (`:658-663`), `leaf_size` (`:664`), `n_jobs` (`:665`). ferrolearn is euclidean-only (`fn sq_euclidean`/`fn mutual_reachability`) with a fixed `O(n^2)` Prim MST (`fn build_mst`); none exist (AC-7). |
| REQ-9 (PyO3 binding) | NOT-STARTED | open prereq blocker **#1074**. `grep -rni hdbscan ferrolearn-python/` is empty (Probe 6) — no `_RsHdbscan`, no registration, no Python wrapper. `import ferrolearn; ferrolearn.HDBSCAN` → AttributeError (AC-8). The sole non-test consumer is the crate re-export (`lib.rs`); `Hdbscan`/`FittedHdbscan` grandfathered as boundary types (S5/R-DEFER-1), so the binding itself is the open work item. |
| REQ-10 (`centroids_`/`medoids_`/`n_features_in_` + `store_centers`) | NOT-STARTED | open prereq blocker **#1075**. sklearn `_weighted_cluster_center` stores `centroids_`/`medoids_` under `store_centers` (`:940-982`), always stores `n_features_in_` (`:566-567`); shapes `(n_clusters, n_features)` (Probe 5, AC-9). `FittedHdbscan` exposes only `labels()`/`probabilities()`/`n_clusters()` — none of these attributes and does not retain `X`. (sklearn 1.5.2 has **no** `cluster_persistence_` — not a divergence.) |
| REQ-11 (error ABI + `n_samples==1`) | NOT-STARTED | open prereq blocker **#1076**. sklearn raises `ValueError("n_samples=1 while HDBSCAN requires more than one sample")` (`hdbscan.py:788`); ferrolearn `fn fit` returns labels `[-1]` for a 1-row array (`test_single_point`, AC-10) — no error. Constraint violations raise `InvalidParameterError` in sklearn vs `FerroError::InvalidParameter` in ferrolearn (the boundary matches, the TYPE does not). |
| REQ-12 (`cluster_selection_epsilon` exact semantics) | NOT-STARTED | open prereq blocker **#1077**. sklearn applies a DBSCAN-like flat-cut epsilon merge in the condensed tree (`_tree.pyx`); ferrolearn's `let above_epsilon = split_dist > eps_f64` heuristic in `fn extract_clusters` is a simplified approximation, not the sklearn epsilon cut. The `cluster_selection_epsilon=0.0` default path (no epsilon) is exercised by Probes 1–3 and matches; non-zero epsilon semantics diverge. |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker **#1078**. `hdbscan.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` (`:46-47`), not `ferray-core` (R-SUBSTRATE-1/2). Not migrated; no PyO3 boundary exists to migrate (REQ-9). |

## Architecture

`hdbscan.rs` follows the unfitted/fitted split (CLAUDE.md naming): `Hdbscan<F>`
(`min_cluster_size: usize`, `min_samples: Option<usize>`, `cluster_selection_epsilon:
F`; `fn new` defaults `5`/`None`/`0`, builders `with_min_cluster_size`/
`with_min_samples`/`with_cluster_selection_epsilon`) → `Fit<Array2<F>, ()>` →
`FittedHdbscan<F>` (private `labels_: Array1<isize>`, `probabilities_: Array1<F>`).
Generic over `F: Float + Send + Sync + 'static`; `fn fit` returns `Result<_,
FerroError>` (R-CODE-2). No `Predict` impl (HDBSCAN labels only the training data —
matching sklearn, which exposes `fit`/`fit_predict` and no out-of-sample `predict`).
Accessors: `labels()`, `probabilities()`, `n_clusters()` (`= max_label + 1`, or 0 if
all noise — a convenience with no sklearn analog, not a divergence).

**Fit path (`fn fit`).** Validates `min_cluster_size >= 2`, `min_samples != 0`,
`cluster_selection_epsilon >= 0` (REQ-3/REQ-11). Short-circuits `n_samples == 0`
(empty result) and `n_samples < min_cluster_size` (all noise). Then the five-step
HDBSCAN pipeline, mirroring `hdbscan.py` `_hdbscan_prims` / `_process_mst` /
`tree_to_labels`:
1. **Core distances** — `fn compute_core_distances` sorts each row's distances and
   takes `dists[min_samples]`. sklearn `_hdbscan_prims` takes the `min_samples`-th NN
   *including self* = `sorted_dists[min_samples-1]` (`:351-352`). **ferrolearn is off
   by one neighbor (REQ-5, Probe 4).** `min_samples` defaults to `min_cluster_size`
   (matches `hdbscan.py:790-792`).
2. **Mutual reachability** — `fn mutual_reachability` = `max(core_dist[i],
   core_dist[j], d(i,j))`, mirroring `_reachability.pyx`
   `mutual_reachability_graph`.
3. **MST** — `fn build_mst` runs Prim's `O(n^2)` over the implicit MR graph. sklearn
   uses Boruvka/Prim over a KD/Ball tree or brute force (`algorithm`, REQ-8); for the
   Euclidean metric the MST is the same tree. No `algorithm`/`leaf_size`/`n_jobs`
   choice.
4. **Single-linkage dendrogram** — `fn extract_clusters` sorts MST edges ascending and
   runs union-find to build the dendrogram (mirrors `_process_mst` → `make_single_linkage`,
   `_linkage.pyx`).
5. **Condensed tree + selection** — `fn extract_clusters` builds a condensed tree
   (children with `>= min_cluster_size` points are real splits; smaller sides "fall
   out" as noise), computes stability `Σ(child_lambda - birth)`, and selects clusters
   bottom-up where a cluster's stability `>=` the sum of its children's (EOM /
   excess-of-mass). The root is never selected. This mirrors `_tree.pyx`
   `_compute_stability` (`:240-278`) + the EOM walk in `tree_to_labels`, **except**:
   ferrolearn's stability omits sklearn's `* cluster_size` weighting (`:273`); there is
   no `'leaf'` mode, `allow_single_cluster`, or `max_cluster_size` (REQ-7); the
   probability is `(child_lambda - birth)/(death - birth)` rather than the GLOSH
   `lambda_p / max_lambda` (`:551-552`, REQ-4); the `cluster_selection_epsilon`
   handling is the heuristic `split_dist > eps` rather than the sklearn flat-cut
   (REQ-12); and exact cluster-id assignment order differs (REQ-6).

**Invariants held vs sklearn (the SHIPPED core):** the `labels_` PARTITION
(co-membership up-to-permutation) + noise set on well-separated data (REQ-1 — exact on
all three fixtures because HDBSCAN is deterministic); `probabilities_` range `[0,1]`
and noise-prob-0 (REQ-2); `min_cluster_size=5`/`min_samples=None`/
`cluster_selection_epsilon=0.0` defaults + the `min_cluster_size >= 2` validation
boundary (REQ-3); the mutual-reachability `max(ci,cj,d)` formula; the Prim MST tree
(euclidean); the single-linkage dendrogram.

**Invariants NOT held vs sklearn:** `probabilities_` VALUES (REQ-4 — GLOSH vs
`(λ-b)/(d-b)`); the core-distance neighbor index (REQ-5 — off by one); exact integer
labels (REQ-6); `cluster_selection_method`/`allow_single_cluster`/`max_cluster_size`
(REQ-7); `metric`/`alpha`/`algorithm`/`leaf_size`/`n_jobs` (REQ-8 — euclidean-only,
fixed Prim); the PyO3 binding (REQ-9 — none exists); `centroids_`/`medoids_`/
`n_features_in_` + `store_centers` (REQ-10); the error ABI + `n_samples==1` raise
(REQ-11); the exact `cluster_selection_epsilon` semantics (REQ-12); the ferray
substrate (REQ-13).

**Consumer wiring.** `pub use hdbscan::{FittedHdbscan, Hdbscan}`
(`ferrolearn-cluster/src/lib.rs`) re-exports the types. Unlike `dbscan.rs` (which has
a full PyO3 binding), HDBSCAN has **no** `ferrolearn.HDBSCAN` consumer — the crate
re-export is the only non-test production consumer, sufficient to grandfather the
boundary estimator types (S5/R-DEFER-1) but leaving REQ-9 NOT-STARTED.

## Verification

Library crate (green at baseline `0fefae62`):
```
cargo test -p ferrolearn-cluster --lib hdbscan     # 14 passed; 0 failed
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s (`test_two_clusters`, `test_dense_clusters`,
`test_noise_detection`, `test_min_cluster_size_effect`, `test_probabilities_range`,
`test_empty_data`, `test_single_point`, `test_too_few_for_cluster`,
`test_invalid_min_cluster_size`, `test_cluster_selection_epsilon`, `test_f32_support`,
`test_identical_points`, `test_n_clusters_accessor`, `test_default_constructor`) pin
the PARTITION (co-membership), the noise set, the probability range, the validation
guard, and the edge cases. They assert STRUCTURE (co-membership, n_clusters, noise
positions, prob∈[0,1]) rather than exact label integers or probability values — the
partition-parity claim is established by the live-oracle probes below.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the partition parity that
backs the SHIPPED REQs (R-CHAR-3 expected values, never literal-copied from
ferrolearn):
```
# REQ-1 (SHIPPED) PARTITION + noise — all three fixtures match (up to permutation)
python3 -c "import numpy as np; from sklearn.cluster import HDBSCAN; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05],[-0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1],[10.05,10.05],[9.95,10.05]]); \
print(HDBSCAN(min_cluster_size=3).fit(X).labels_.tolist())"   # [0,0,0,0,0,0,1,1,1,1,1,1]  (ferro: identical partition)
python3 -c "import numpy as np; from sklearn.cluster import HDBSCAN; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1],[10.05,10.05],[50.,50.],[-50.,-50.],[100.,0.],[0.,100.]]); \
print(HDBSCAN(min_cluster_size=3).fit(X).labels_.tolist())"   # [0,0,0,0,0,1,1,1,1,1,-1,-1,-1,-1]  (ferro: identical)
# REQ-4 (DIVERGES) probabilities VALUES
python3 -c "import numpy as np; from sklearn.cluster import HDBSCAN; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05],[-0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1],[10.05,10.05],[9.95,10.05]]); \
print([round(p,3) for p in HDBSCAN(min_cluster_size=3).fit(X).probabilities_])"  # [1,0.707,1,0.707,1,1,...]; ferro all 1.0
# REQ-5 (DIVERGES) core-distance off-by-one
python3 -c "import numpy as np; from sklearn.neighbors import NearestNeighbors; \
X=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[10.,0.]]); d,_=NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X,3); \
print(d[:,-1].tolist())"  # [2,1,1,2,8] = sorted[ms-1]; ferro fn compute_core_distances -> sorted[ms] = [3,2,2,3,9]
# REQ-10 (MISSING) centroids_/medoids_/n_features_in_ ; (no cluster_persistence_ in sklearn 1.5.2)
python3 -c "import numpy as np; from sklearn.cluster import HDBSCAN; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05],[-0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1],[10.05,10.05],[9.95,10.05]]); \
m=HDBSCAN(min_cluster_size=3,store_centers='both').fit(X); print(m.n_features_in_, m.centroids_.shape, m.medoids_.shape, hasattr(m,'cluster_persistence_'))"  # 2 (2,2) (2,2) False
```
Characterization pins (R-CHAR-3) belong in
`ferrolearn-cluster/tests/divergence_hdbscan.rs`: REQ-1 asserts the live-sklearn
partition (canonicalized — co-membership classes + noise set) and PASSES against
current `hdbscan.rs` (the SHIPPED-confirming test, not a failing pin). Pins for the
NOT-STARTED REQs — REQ-4 (probability VALUES), REQ-5 (core distances `[2,1,1,2,8]`),
REQ-10 (`centroids_`/`medoids_`), REQ-11 (`n_samples==1` raise) — assert the sklearn
behavior ferrolearn cannot yet reach and FAIL until the surface lands.

There is no `ferrolearn-python` verification for this unit — **no PyO3 binding exists**
(REQ-9). When the binding lands, a `ferrolearn-python/tests/divergence_hdbscan.py`
should assert `ferrolearn.HDBSCAN(...).fit(X).labels_` partitions identically to
`sklearn.cluster.HDBSCAN`.

## Blockers to open

(REQ-1/REQ-2/REQ-3 are SHIPPED and need NO blocker. The NOT-STARTED REQs are the
real filed issues #1069–#1078.)

- **#1069** — REQ-4: replace the probability formula with the sklearn GLOSH membership
  `min(lambda_p, max_lambda)/max_lambda` per condensed cluster (`_tree.pyx:524-552`).
- **#1070** — REQ-5 (**cleanest minimal fix**): change `fn compute_core_distances` from
  `dists[min_samples]` to `dists[min_samples - 1]` (with `min_samples` clamped to
  `n`), matching sklearn's "min_samples includes the point itself" convention
  (`_hdbscan_prims:351-352`, docstring `:444-446`).
- **#1071** — REQ-6: assign cluster-id integers in sklearn's `_do_labelling`/
  `tree_to_labels` order so exact `labels_` match (`_tree.pyx:646+`).
- **#1072** — REQ-7: add `cluster_selection_method` (`'eom'`/`'leaf'`),
  `allow_single_cluster`, and `max_cluster_size` (`hdbscan.py:647-650`, `:666-667`).
- **#1073** — REQ-8: add `metric`/`metric_params`/`alpha`/`algorithm`/`leaf_size`/
  `n_jobs` (and `'precomputed'`), default `'euclidean'`/`alpha=1.0`/`algorithm='auto'`
  (`hdbscan.py:651-665`).
- **#1074** — REQ-9: build the `ferrolearn.HDBSCAN` PyO3 binding (`_RsHdbscan` in
  `ferrolearn-python`, registration, Python wrapper, `__init__` export) exposing
  `labels_`/`probabilities_`/`fit_predict` — the real non-test consumer.
- **#1075** — REQ-10: add `centroids_`/`medoids_`/`n_features_in_` accessors +
  `store_centers` to `FittedHdbscan` (retain `X` at fit), mirroring
  `_weighted_cluster_center` (`hdbscan.py:940-982`).
- **#1076** — REQ-11: raise on `n_samples==1` (`hdbscan.py:788`) and map validation
  failures to the sklearn `ValueError`/`InvalidParameterError` ABI.
- **#1077** — REQ-12: implement the sklearn DBSCAN-like `cluster_selection_epsilon`
  flat-cut (`_tree.pyx`) in place of the `split_dist > eps` heuristic.
- **#1078** — REQ-13: migrate `hdbscan.rs` off `ndarray`/`num-traits` to `ferray-core`
  (R-SUBSTRATE); and route any future PyO3 boundary through `ferray::numpy_interop`.
