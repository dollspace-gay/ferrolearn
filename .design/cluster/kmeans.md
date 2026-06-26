# K-Means (sklearn.cluster.KMeans)

<!--
tier: 3-component
status: draft
baseline-commit: 29f2d461
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_kmeans.py            # kmeans_plusplus (:61); _kmeans_plusplus (:164); class KMeans(_BaseKMeans) (:1196); __init__ (:1387-1411) defaults init="k-means++"/n_init="auto"/max_iter=300/tol=1e-4/algorithm="lloyd"; KMeans._check_params_vs_input (:1413-1417) default_n_init=10; _BaseKMeans._check_params_vs_input (:875-908) n_init="auto"+"k-means++" -> _n_init=1; _tolerance (:286-294); _kmeans_single_lloyd (:631+); lloyd inner loop convergence center_shift_tot=(center_shift**2).sum() <= tol (:586-601); post-loop E-step re-run + inertia (:605-625); n_init docstring (:355-364); predict (:1072); fit_transform (:1106); transform (:1130); score=-inertia (:1156-1184)
ferrolearn-module: ferrolearn-cluster/src/kmeans.rs
ferrolearn-python: ferrolearn-python/src/clusterers.rs (RsKMeans, name="_RsKMeans")
parity-ops: kmeans_plusplus, KMeans (.__init__, .fit, .predict, .fit_predict, .transform, .score, .cluster_centers_, .labels_, .inertia_, .n_iter_)
crosslink-issue: 1127
-->

## Summary

`ferrolearn-cluster/src/kmeans.rs` mirrors scikit-learn's `KMeans`
(`sklearn/cluster/_kmeans.py`, `class KMeans(_BaseKMeans)` `:1196`) — Lloyd's
algorithm with greedy k-means++ seeding, multi-start over `n_init` runs, keeping the
lowest-inertia run. It exposes the unfitted `KMeans<F>` (`n_clusters`, `max_iter`,
`tol`, `n_init`, `random_state`), the standalone `kmeans_plusplus` /
`kmeans_plusplus_with_options` helpers, the fitted `FittedKMeans<F>` (private
`cluster_centers_`, `labels_`, `inertia_`, `n_iter_` with accessors), a `Predict` impl
(nearest center), a `Transform` impl (euclidean distance to centers), and a `fit_predict`
convenience mirroring `ClusterMixin.fit_predict`. It is re-exported at the crate root
(`pub use kmeans::{FittedKMeans, KMeans, kmeans_plusplus,
kmeans_plusplus_with_options}` in `ferrolearn-cluster/src/lib.rs`) AND — unlike
the recently-audited `mean_shift` / `affinity_propagation` / `label_*` units — has a
REAL CPython consumer: the PyO3 binding `RsKMeans` (`#[pyclass(name = "_RsKMeans")]`) in
`ferrolearn-python/src/clusterers.rs`, registered in `ferrolearn-python/src/lib.rs`
(`m.add_class::<clusterers::RsKMeans>()`) and wrapped by the sklearn-API class
`ferrolearn.KMeans` (`ferrolearn-python/python/ferrolearn/_clusterers.py`, re-exported in
`__init__.py`). So `import ferrolearn; ferrolearn.KMeans()` reaches this code.

**Under honest assessment (R-HONEST-3), shipped REQs include the clustering PARTITION
up-to-permutation on well-separated data (REQ-1), the
`predict` nearest-center CONTRACT (REQ-2), the `transform` distance-to-centers CONTRACT
(REQ-3), the PyO3 binding marshalling (REQ-4) of `fit`/`predict`/`transform`/
`cluster_centers_`/`labels_`/`inertia_`/`n_iter_`, the `labels_`/`inertia_`↔`cluster_centers_`
consistency via a post-loop E-step (REQ-6, fixed iter 127 / #1037), and the `n_init`
constructor default = 1 (REQ-14, #1045), plus the standalone `kmeans_plusplus` helper
shape (REQ-16).** Everything tied to exact VALUES (centers,
inertia, absolute label integers, `n_iter_`) and several API/structural contracts DIVERGE,
for a stack of compounding reasons carved out below:

1. **`n_init` DEFAULT (R-DEV-2) — REQ-14, FIXED iter 127 / #1045.** sklearn
   `KMeans(n_init="auto")` (`_kmeans.py:1392`), which for the default `init="k-means++"`
   resolves to **1** (`_BaseKMeans._check_params_vs_input` `:886-888`: `n_init=="auto"` &
   `init` str `"k-means++"` → `self._n_init = 1`; docstring `:359-361`). ferrolearn always
   uses k-means++ (no `init` param, REQ-7), so the sklearn-matching default is `1`;
   `fn new` now defaults `n_init = 1` (was `10`). `n_init` is a `pub` field on `KMeans<F>`,
   pinned by `KMeans::<f64>::new(3).n_init == 1` (deterministic, RNG-free). Tracked as the
   standalone REQ-14 (#1045), NOT folded into REQ-9's value parity.
   (The PyO3 `RsKMeans` signature ALSO defaults `n_init=10` (`fn new` `#[pyo3(signature
   = …)]`) and the Python wrapper `ferrolearn.KMeans.__init__` defaults `n_init=10`
   (`_clusterers.py`); both are a separate ferrolearn-python concern folded into REQ-12.)
2. **`tol` is RELATIVE in sklearn (R-DEV-1).** sklearn `_tolerance(X, tol) =
   np.mean(np.var(X, axis=0)) * tol` (`_kmeans.py:286-294`, stored as `self._tol` in
   `_check_params_vs_input` `:883`) — the convergence threshold is scaled by the data
   variance. ferrolearn uses the ABSOLUTE `tol = 1e-4` directly against the centroid shift
   (`fn fit` `max_shift < self.tol`). Different effective convergence threshold (REQ-5).
3. **convergence CRITERION diverges (R-DEV-1).** sklearn Lloyd converges on strict
   label-no-change (`np.array_equal(labels, labels_old)` `:586`) OR
   `center_shift_tot = (center_shift**2).sum() <= tol` — the SUM of squared per-center
   shifts (`:594-595`). ferrolearn converges on `max_shift < tol` where `max_shift` is the
   MAX per-center EUCLIDEAN shift (`fn recompute_centroids_into` returns `max_shift.sqrt()`,
   checked at `fn fit`). Different convergence rule → different `n_iter_`, different stop
   point (REQ-5).
4. **`labels_`/`inertia_` consistency with FINAL centers (R-DEV-1).** sklearn, after the
   inner loop, RE-RUNS the E-step so labels match the final centers, then computes
   `inertia = _inertia(X, sw, centers, labels)` (`_kmeans.py:605-625`) — `labels_` /
   `inertia_` / `cluster_centers_` are all mutually consistent, so
   `fit(X).predict(X) == fit(X).labels_` holds (Probe C). ferrolearn `fn fit` assigns
   labels/inertia to the PRE-recompute centers, then recomputes `new_centers` and
   `std::mem::swap(&mut centers, &mut new_centers)`, storing the POST-swap `centers` as
   `cluster_centers_` but the PRE-swap `labels`/`inertia` (`fn fit` candidate
   construction). So `cluster_centers_` is one M-step AHEAD of `labels_`/`inertia_`, and
   `predict(X)` (which re-assigns to `cluster_centers_`) can disagree with `labels_`. This
   is cleanly pinnable as `fit(X).predict(X) == labels_` (REQ-6).
5. **empty-cluster relocation diverges (R-DEV-1).** sklearn relocates an emptied cluster's
   center to the sample farthest from its assigned center
   (`_relocate_empty_clusters_dense`, invoked inside the Cython `lloyd_iter` step).
   ferrolearn KEEPS the old center for an empty cluster (`fn recompute_centroids_into`
   else-branch). Divergent centers when a cluster empties (REQ-5/REQ-9).
6. **`init` param + k-means++ exactness + `random`/array/callable missing (R-DEV-1/2).**
   sklearn `init ∈ {"k-means++","random"}|callable|array-like`, default `"k-means++"`
   (`_kmeans.py:1391`, `_init_centroids`). ferrolearn now exposes standalone
   `kmeans_plusplus` helpers returning `(centers, indices)` with `sample_weight` and
   `n_local_trials`, but `KMeans` still has NO `init` param — it always uses greedy
   k-means++. The DEFAULT matches (k-means++), but the estimator param surface and the
   `"random"`/array/callable paths are missing, and exact seeded output diverges (numpy
   RNG, REQ-8) (REQ-7/REQ-16).
7. **numpy-RNG parity (R-DEV-1).** sklearn `check_random_state` + numpy RNG; ferrolearn
   `StdRng::seed_from_u64(base_seed + run)` per run (`fn fit`). Exact
   `cluster_centers_`/`labels_`/`inertia_`/`n_iter_` cannot match sklearn without a
   numpy-RNG analog. BLOCKS exact VALUE parity (REQ-8).
8. **`sample_weight` missing (R-DEV-1).** sklearn supports it (`k_means`, `fit`,
   `_inertia`); ferrolearn does not (REQ-10).
9. **`algorithm` (lloyd/elkan) missing (R-DEV-2).** sklearn `algorithm` `:1398`;
   ferrolearn lloyd-only (REQ-10).
10. **`score` method missing (R-DEV-3).** sklearn `KMeans.score(X)` returns `-inertia`
    (`_kmeans.py:1156-1184`). ferrolearn `FittedKMeans` has NO `score` method, the PyO3
    `RsKMeans` has no `score` getter, and the Python `ferrolearn.KMeans` wrapper inherits
    only `TransformerMixin`/`ClusterMixin`/`BaseEstimator` — `ClusterMixin` does NOT define
    `score`, so `score` is absent end-to-end (REQ-11).
11. **`fit_transform` missing (R-DEV-3).** sklearn has both `fit_predict` (`:1047`) and
    `fit_transform` (`:1106`); ferrolearn has `fit_predict` (`fn fit_predict`) but no
    `fit_transform` (REQ-11).
12. **error ABI / validation surface (R-DEV-2).** sklearn validates via
    `_parameter_constraints`/`_check_params_vs_input` (`ValueError` on
    `n_samples < n_clusters`, `InvalidParameterError` on bad params); ferrolearn returns
    `FerroError::InvalidParameter`/`InsufficientSamples` (`fn fit`). Missing
    `n_features_in_`/`feature_names_in_`/`copy_x`/`verbose` (REQ-10).
13. **`labels_` dtype (R-DEV-3).** sklearn `labels_` is `int32` (Probe A). ferrolearn
    stores `Array1<usize>`; the binding marshals to `i64` via `ndarray1_usize_to_numpy`
    (`RsKMeans::predict`/`labels_`) — `int64`, not `int32` (REQ-12 dtype nuance).
14. **ferray substrate (R-SUBSTRATE).** `kmeans.rs` imports `ndarray::{Array1,Array2}`,
    `num_traits::Float`, `rand::{rngs::StdRng,…}`, `rayon`, not `ferray-core` /
    `ferray::linalg` / `ferray::random` (REQ-13).

## Live oracle probes (sklearn 1.5.2, run from /tmp)

Expected values are from the installed sklearn 1.5.2 oracle, never literal-copied from
ferrolearn (R-CHAR-3). `n_init` is set EXPLICITLY in the value probes to control for the
default divergence (REQ-9). Fixture `blobs` = the 9×2 well-separated 3-blob set (3 points
near `(0,0)`, 3 near `(10,10)`, 3 near `(0,10)`) — the `make_blobs()` fixture in
`kmeans.rs`.

### Probe A — well-separated 3-blob: labels_ / centers / inertia_ / n_iter_ / dtype
```
python3 -c "import numpy as np; from sklearn.cluster import KMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=KMeans(n_clusters=3, n_init=10, random_state=42).fit(X); \
print(m.labels_.tolist(), np.round(m.cluster_centers_,4).tolist(), round(m.inertia_,6), m.n_iter_, m.labels_.dtype)"
# labels  [2, 2, 2, 0, 0, 0, 1, 1, 1]
# centers [[10.0, 10.0667], [0.0, 10.0], [0.0, 0.0667]]
# inertia 0.093333
# n_iter  2
# dtype   int32
```
**Findings:** sklearn groups `{0,1,2}` / `{3,4,5}` / `{6,7,8}` (REQ-1, the partition).
ferrolearn `KMeans::<f64>::new(3).with_random_state(42).with_n_init(10).fit(blobs)`
recovers the SAME 3-way grouping — verified by the in-tree `test_well_separated_blobs`
asserting `labels[0]==labels[1]==labels[2]` etc. — but the label INTEGERS are permuted
(different RNG/init order), the `cluster_centers_` VALUES diverge (RNG + empty-cluster +
final-E-step), `inertia_` need not equal `0.093333`, `n_iter_` diverges (different
convergence criterion), and `labels_` is `usize` (binding → `int64`), not sklearn's
`int32` (REQ-12/REQ-13).

### Probe B — default constructor surface (the n_init default divergence)
```
python3 -c "from sklearn.cluster import KMeans; d=KMeans(); \
print(d.init, d.n_init, d.max_iter, d.tol, d.algorithm, d.n_clusters)"
# k-means++ auto 300 0.0001 lloyd 8
```
sklearn defaults: `init="k-means++"`, `n_init="auto"` (resolving to **1** for k-means++
via `_check_params_vs_input` `:886-888`, docstring `:359-361`), `max_iter=300`,
`tol=1e-4`, `algorithm="lloyd"`, `n_clusters=8`. ferrolearn `fn new`: `max_iter=300`
(matches), `tol=1e-4` (absolute, not relative — REQ-5), `n_init=10` (sklearn-equivalent
is `1` for k-means++ — REQ-9), no `init`/`algorithm`/`copy_x`/`verbose`, and no
`n_clusters` default (Rust builder requires the argument). The PyO3 `RsKMeans` signature
(`n_clusters=8, max_iter=300, tol=1e-4, n_init=10, random_state=None`) and the Python
wrapper both default `n_init=10` (REQ-12).

### Probe C — transform shape, argmin==predict, predict==labels_ (consistency)
```
python3 -c "import numpy as np; from sklearn.cluster import KMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=KMeans(n_clusters=3, n_init=10, random_state=42).fit(X); T=m.transform(X); \
print(T.shape, np.round(T[0],6).tolist(), np.array_equal(T.argmin(1), m.predict(X)), np.array_equal(m.predict(X), m.labels_))"
# (9, 3)  [14.189354, 10.0, 0.066667]   argmin==predict True   predict==labels_ True
```
sklearn `transform` returns euclidean distance from each sample to each `cluster_centers_`
row, shape `(n_samples, n_clusters)`; `transform(X).argmin(axis=1) == predict(X)` and
`predict(X) == labels_` (labels match the final centers, REQ-6). ferrolearn
`Transform::transform` (`fn transform`) computes the same metric/shape/column-ordering
(REQ-3), and `Predict::predict` (`fn predict`) is `argmin` over `nearest_center`
(REQ-2 — argmin==predict holds by construction). BUT `predict(X) == labels_` can FAIL in
ferrolearn because `cluster_centers_` is stored one M-step AHEAD of `labels_` (no
post-loop E-step re-run, REQ-6 / divergence #4).

### Probe D — score(X) == -inertia
```
python3 -c "import numpy as np; from sklearn.cluster import KMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=KMeans(n_clusters=3, n_init=10, random_state=42).fit(X); \
print(round(m.score(X),6), round(-m.inertia_,6))"
# -0.093333  -0.093333
```
sklearn `KMeans.score(X)` (`_kmeans.py:1156-1184`) returns `-inertia` of `X` against the
fitted centers. ferrolearn has NO `score` — `FittedKMeans` (`impl FittedKMeans`) exposes
only `cluster_centers`/`labels`/`inertia`/`n_iter`; the PyO3 `RsKMeans` has no `score`
getter; the Python wrapper inherits `TransformerMixin`/`ClusterMixin`/`BaseEstimator` and
`ClusterMixin` has no `score`. Absent end-to-end (REQ-11).

### Probe E — PyO3 binding chain (the real CPython consumer)
- `#[pyclass(name = "_RsKMeans")]` `RsKMeans` in `ferrolearn-python/src/clusterers.rs`
  exposes `new`/`fit`/`predict`/`transform` + getters `cluster_centers_`/`labels_`/
  `inertia_`/`n_iter_`, delegating to `ferrolearn_cluster::KMeans::<f64>` /
  `FittedKMeans::<f64>`.
- Registered: `m.add_class::<clusterers::RsKMeans>()?` in `ferrolearn-python/src/lib.rs`.
- Python export: `from ferrolearn._ferrolearn_rs import _RsKMeans` in
  `ferrolearn-python/python/ferrolearn/_clusterers.py`, wrapped by
  `class KMeans(TransformerMixin, ClusterMixin, BaseEstimator)`, re-exported as `KMeans`
  in `ferrolearn-python/python/ferrolearn/__init__.py` (`from ferrolearn._clusterers
  import KMeans`; `"KMeans"` in `__all__`).
So `import ferrolearn; ferrolearn.KMeans(...).fit(X)` exercises `fn fit` → the binding is a
genuine non-test production consumer of `fit`/`predict`/`transform` and all four getters.

## Requirements

- REQ-1: **clustering PARTITION up-to-permutation (R-DEV-1).** Mirror
  `KMeans(n_clusters=k, n_init=N, random_state=s).fit(X)` producing the same grouping of
  samples into clusters on well-separated data. ferrolearn `fn fit` (k-means++ seed →
  Lloyd assign/recompute loop → best-of-`n_init`) recovers the correct PARTITION on
  separable blobs (Probe A) — but the `labels_` INTEGERS (REQ-9), `cluster_centers_`
  VALUES (REQ-8/REQ-9), `inertia_`, and `n_iter_` diverge. Partition-only, not value
  parity.
- REQ-2: **`predict` nearest-center CONTRACT (R-DEV-3).** Mirror
  `KMeans.predict` = assign each sample to the argmin-distance cluster center
  (`_kmeans.py:1072-1104`, returns `int32` labels). ferrolearn `Predict::predict`
  (`fn predict`, via `fn assign_clusters` / `fn nearest_center`) returns the argmin-squared-
  euclidean center index, so `transform(X).argmin(axis=1) == predict(X)` (Probe C). The
  CONTRACT (nearest-center rule, output shape `(n_samples,)`, shape-mismatch error) matches;
  the absolute label INTEGERS track REQ-9 and the dtype is `usize`/`i64` not `int32`
  (REQ-12/REQ-13).
- REQ-3: **`transform` distance-to-centers CONTRACT (R-DEV-3).** Mirror
  `KMeans.transform` = euclidean distance from each sample to each `cluster_centers_` row,
  shape `(n_samples, n_clusters)` (`_kmeans.py:1130-1154`). ferrolearn
  `Transform::transform` (`fn transform`) returns `sqrt(squared_euclidean(row, center))`
  per `(sample, center)` in the same shape/column ordering (Probe C). CONTRACT matches; the
  distance VALUES track the `cluster_centers_` divergence (REQ-8).
- REQ-4: **PyO3 binding marshalling (R-DEFER-1).** `import ferrolearn; ferrolearn.KMeans`
  reaches `fn fit`/`predict`/`transform` and the `cluster_centers_`/`labels_`/`inertia_`/
  `n_iter_` getters through `RsKMeans` (Probe E). The fit→predict→transform→accessor
  marshalling round-trips arrays/scalars across the PyO3 boundary, preserving shape
  (centers `(k, n_features)`, distances `(n, k)`, labels `(n,)`, inertia/n_iter scalars).
- REQ-5: **convergence criterion + relative `tol` (R-DEV-1).** sklearn converges on strict
  label-no-change OR `center_shift_tot = (center_shift**2).sum() <= self._tol`, where
  `self._tol = mean(var(X)) * tol` (`_kmeans.py:286-294,586-601`). ferrolearn converges on
  `max_shift < self.tol` with `max_shift` the MAX per-center euclidean shift and `tol`
  ABSOLUTE (`fn recompute_centroids_into` + `fn fit`). Different threshold + different
  reduction → different `n_iter_` and stop point.
- REQ-6: **`labels_`/`inertia_` consistency with FINAL centers (R-DEV-1).** sklearn re-runs
  the E-step post-loop so `labels_`/`inertia_` match `cluster_centers_`, guaranteeing
  `fit(X).predict(X) == labels_` (`_kmeans.py:605-625`, Probe C). ferrolearn `fn fit`
  stores `labels`/`inertia` from the assignment to the PRE-swap centers but stores the
  POST-swap centers as `cluster_centers_` — `cluster_centers_` is one M-step ahead, so
  `predict(X)` can disagree with `labels_`. Cleanly pinnable as `fit(X).predict(X) ==
  labels_`.
- REQ-7: **`init` param incl. `random`/array/callable + k-means++ exactness (R-DEV-1/2).**
  sklearn `init ∈ {"k-means++","random"}|callable|array`, default `"k-means++"`
  (`_kmeans.py:1391`, `_init_centroids`/`kmeans_plusplus`). ferrolearn has NO `init` param —
  always greedy k-means++ (`fn kmeans_plusplus_inner`, `n_trials = 2 + floor(ln k)`). The
  DEFAULT matches; the estimator param surface, `"random"`/array/callable, and exact
  k-means++ output (numpy RNG, REQ-8) diverge.
- REQ-8: **`random_state` numpy-RNG parity (R-DEV-1).** sklearn `check_random_state` + numpy
  RNG; ferrolearn `StdRng::seed_from_u64(base_seed + run)` per run (`fn fit`). Different RNG
  → exact `cluster_centers_`/`labels_`/`inertia_`/`n_iter_` cannot match sklearn. Depends on
  a ferray `random` analog (R-SUBSTRATE-5); BLOCKS exact VALUE parity (REQ-9).
- REQ-9: **`cluster_centers_` / `inertia_` / `labels_`-integers / `n_iter_` VALUE parity
  (R-DEV-1).** Mirror sklearn's exact center coordinates, total inertia, absolute label
  integers, and iteration count (Probe A). Diverges because of: numpy-RNG init (REQ-8),
  `n_init` default (REQ-9 itself / observed via REQ defaults), the convergence criterion +
  relative `tol` (REQ-5), the missing post-loop E-step (REQ-6), and empty-cluster
  relocation (sklearn moves an emptied center to the farthest sample;
  ferrolearn keeps the old center, `fn recompute_centroids_into` else-branch).
- REQ-10: **constructor/fit surface `init`/`algorithm`/`copy_x`/`verbose`/`sample_weight` +
  `n_clusters`=8 + error ABI (R-DEV-2).** sklearn `__init__` (`_kmeans.py:1387-1411`) +
  `fit(X, y, sample_weight)` + `_check_params_vs_input` (`InvalidParameterError`/`ValueError`,
  `:875-908,1413-1417`) + `n_features_in_`/`feature_names_in_`. ferrolearn `KMeans<F>`
  (`fn new` + builders) has `n_clusters/max_iter/tol/n_init/random_state` only; `fn fit`
  returns `FerroError::InvalidParameter`/`InsufficientSamples`. Missing
  `init`/`algorithm`/`copy_x`/`verbose`/`sample_weight`, `n_clusters` default, and the
  introspection attrs; different error type/message ABI.
- REQ-11: **`score` + `fit_transform` methods (R-DEV-3).** sklearn `KMeans.score(X)` =
  `-inertia` (`_kmeans.py:1156-1184`, Probe D) and `fit_transform` (`:1106`). ferrolearn
  `FittedKMeans` has neither — only `fit_predict` (`fn fit_predict`); the PyO3 `RsKMeans`
  has no `score`/`fit_transform`; the Python `ferrolearn.KMeans` wrapper's mixins
  (`TransformerMixin`/`ClusterMixin`/`BaseEstimator`) do NOT supply `score`. Absent
  end-to-end.
- REQ-12: **ferrolearn-python signature default `n_init` + `labels_` dtype (R-DEV-2/3,
  R-DEFER-7).** The PyO3 `RsKMeans::new` signature defaults `n_init=10`
  (`#[pyo3(signature = (…, n_init=10, …))]`) and the Python `ferrolearn.KMeans.__init__`
  defaults `n_init=10` (`_clusterers.py`) — both diverge from sklearn's effective `1` for
  k-means++ (mirrors REQ-9 at the binding). The binding marshals `labels_`/`predict` to
  `int64` (`ndarray1_usize_to_numpy`), not sklearn's `int32` (Probe A). This crate is the
  last translation layer (R-DEFER-7); the SEMANTIC default lives in the library `fn new`
  (REQ-9), the binding signature + dtype are the marshalling-layer reflection.
- REQ-13: **ferray substrate (R-SUBSTRATE).** `kmeans.rs` imports `ndarray::{Array1,Array2}`,
  `num_traits::Float`, `rand::{rngs::StdRng, Rng, SeedableRng}`, `rayon::prelude::*`, not
  `ferray-core` / `ferray::linalg` / `ferray::random` (R-SUBSTRATE-1/2). The RNG migration
  is entangled with REQ-8.
- REQ-16: **standalone `kmeans_plusplus` helper (R-DEV-1/2).** sklearn exposes
  `sklearn.cluster.kmeans_plusplus(X, n_clusters, *, sample_weight=None,
  x_squared_norms=None, random_state=None, n_local_trials=None)` (`_kmeans.py:61`) returning
  `(centers, indices)`. ferrolearn exposes `pub fn kmeans_plusplus` plus
  `kmeans_plusplus_with_options` for `sample_weight` and `n_local_trials`, returning the same
  shape contract. Underclaim: no `x_squared_norms` precompute argument and seeded exact
  indices still diverge via `StdRng` vs NumPy `RandomState` (REQ-8).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), `n_init` set
explicitly in value probes, never literal-copied from ferrolearn (R-CHAR-3). Fixture
`blobs` = `make_blobs()` (9×2).

- AC-1 (REQ-1, partition agrees / integers diverge): `KMeans(n_clusters=3, n_init=10,
  random_state=42).fit(blobs).labels_` → sklearn `[2,2,2,0,0,0,1,1,1]`; ferrolearn recovers
  the same 3-way grouping (`{0,1,2}`/`{3,4,5}`/`{6,7,8}`) up to a permutation of the
  integers.
- AC-2 (REQ-2, contract): `KMeans(...).fit(blobs)`: `transform(blobs).argmin(axis=1) ==
  predict(blobs)` (sklearn `True`); ferrolearn `Predict::predict` is argmin over the same
  distances. Shape `(9,)`, shape-mismatch raises.
- AC-3 (REQ-3, contract): `KMeans(n_clusters=3, n_init=10, random_state=42).fit(blobs)
  .transform(blobs).shape` → sklearn `(9, 3)`, column `j` = `||x_i - cluster_centers_[j]||`
  (row 0 = `[14.189354, 10.0, 0.066667]`). ferrolearn returns the same shape + per-center
  euclidean distance; the values track the `cluster_centers_` divergence (REQ-9).
- AC-4 (REQ-4, binding): `import ferrolearn; m=ferrolearn.KMeans(n_clusters=3,
  random_state=42).fit(blobs)` exposes `m.cluster_centers_` `(3,2)`, `m.labels_` `(9,)`,
  `m.inertia_` (float), `m.n_iter_` (int), `m.predict(blobs)` `(9,)`, `m.transform(blobs)`
  `(9,3)` — round-tripping shapes across the PyO3 boundary.
- AC-5 (REQ-14, n_init default — SHIPPED): `KMeans().n_init` resolves to `1` for `init="k-means++"`
  (sklearn `_check_params_vs_input` `:886-888`, docstring `:359-361`). ferrolearn
  `KMeans::<f64>::new(3).n_init == 1` is the target and now passes.
- AC-6 (REQ-6, consistency): sklearn `fit(blobs).predict(blobs) == fit(blobs).labels_` →
  `True` (Probe C, post-loop E-step). ferrolearn must hold `fit(X).predict(X) == labels_`;
  the final E-step now makes this invariant pass.
- AC-7 (REQ-11, score): `KMeans(n_clusters=3, n_init=10, random_state=42).fit(blobs)
  .score(blobs)` → sklearn `-inertia_` (`-0.093333`, Probe D). ferrolearn has no `score`
  (library, binding, or Python wrapper); `fit_transform` is also absent.
- AC-8 (REQ-9 value parity, diverges): `KMeans(n_clusters=3, n_init=10, random_state=42)
  .fit(blobs)` → sklearn `cluster_centers_ = [[10,10.0667],[0,10],[0,0.0667]]`,
  `inertia_ = 0.093333`, `n_iter_ = 2`. ferrolearn produces different center VALUES,
  inertia, label integers, and `n_iter_` (RNG + tol + convergence + empty-cluster +
  no-final-E-step).
- AC-9 (REQ-16, standalone `kmeans_plusplus` helper): sklearn
  `kmeans_plusplus([[1,2]], 1, random_state=0)` → `centers=[[1,2]]`, `indices=[0]`;
  weighted deterministic fixture `X=[[0,0],[10,0]], sample_weight=[0,1], n_clusters=2`
  → `centers=[[10,0],[0,0]]`, `indices=[1,0]`. ferrolearn matches these RNG-independent
  outputs and validates the same core public constraints.

## REQ status table

Binary (R-DEFER-2). `KMeans` / `FittedKMeans` are existing pub APIs re-exported at the
crate root and consumed by the PyO3 binding `RsKMeans` (a REAL non-test CPython consumer;
grandfathered S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn
1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2, run from `/tmp`. Honest
assessment (R-HONEST-3): shipped REQs include the clustering PARTITION up-to-permutation
(REQ-1), the `predict` nearest-center CONTRACT (REQ-2), the `transform` distance-to-centers
CONTRACT (REQ-3), the PyO3 binding marshalling (REQ-4), and standalone
`kmeans_plusplus` helper shape (REQ-16). Exact VALUES, defaults, convergence,
the estimator `init`/`sample_weight`/`algorithm` surface, `score`/`fit_transform`,
numpy-RNG parity, the binding default/dtype, and the ferray substrate still diverge.
Blocker numbers below are the real filed issues (#1036–#1045).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (clustering PARTITION up-to-permutation) | SHIPPED | impl `fn fit` (greedy k-means++ `fn kmeans_plusplus_inner` seed → Lloyd `fn assign_clusters_into`/`fn recompute_centroids_into` loop → best-of-`n_init` lowest-inertia) recovers sklearn's grouping on well-separated data (Probe A). Consumers: PyO3 `RsKMeans::fit` (`clusterers.rs`, registered in `lib.rs`) AND crate re-export `pub use kmeans::{FittedKMeans, KMeans, kmeans_plusplus, kmeans_plusplus_with_options}` (`lib.rs`). Guards: in-tree `test_well_separated_blobs`, `test_two_clusters_on_line`, `test_k_equals_n_samples`, `test_single_cluster` assert co-membership. Verification: `cargo test -p ferrolearn-cluster --lib kmeans` (see below). Underclaim: PARTITION up-to-permutation only — `labels_` INTEGERS, `cluster_centers_`/`inertia_` VALUES, `n_iter_` diverge (REQ-9). |
| REQ-2 (`predict` nearest-center contract) | SHIPPED | impl `Predict::predict` (`fn predict` → `fn assign_clusters`/`fn nearest_center`) returns the argmin-squared-euclidean center index, shape `(n_samples,)`, with a shape-mismatch `FerroError::ShapeMismatch`; `transform(X).argmin(1) == predict(X)` (Probe C). Consumers: PyO3 `RsKMeans::predict` (`ndarray1_usize_to_numpy`) AND crate re-export. Guards: `test_predict_on_new_data`, `test_predict_shape_mismatch`. Underclaim: nearest-center CONTRACT only — label INTEGERS track REQ-9, dtype `usize`/`i64` not `int32` (REQ-12). |
| REQ-3 (`transform` distance-to-centers contract) | SHIPPED | impl `Transform::transform` (`fn transform`) returns `sqrt(squared_euclidean(row, center))` per `(sample, center)`, shape `(n_samples, n_clusters)`, column `j` = distance to `cluster_centers_[j]` — matching sklearn `_BaseKMeans.transform` (`_kmeans.py:1130-1154`) shape/metric/ordering (Probe C: `(9,3)`, row0 `[14.189354,10.0,0.066667]`). Consumers: PyO3 `RsKMeans::transform` (`ndarray2_to_numpy`) AND crate re-export. Guards: `test_transform_distances`, `test_transform_shape`, `test_transform_shape_mismatch`. Underclaim: CONTRACT only — distance VALUES track the `cluster_centers_` divergence (REQ-8/REQ-9). |
| REQ-4 (PyO3 binding marshalling) | SHIPPED | impl `RsKMeans` (`#[pyclass(name="_RsKMeans")]`, `clusterers.rs`) marshals `fit`/`predict`/`transform` + `cluster_centers_`/`labels_`/`inertia_`/`n_iter_` getters over the PyO3 boundary, preserving shapes (centers `(k,nf)`, distances `(n,k)`, labels `(n,)`, scalars). Consumer (the binding IS the CPython consumer): registered `m.add_class::<clusterers::RsKMeans>()` (`ferrolearn-python/src/lib.rs`) → wrapped by `class KMeans(...)` (`_clusterers.py`) → re-exported `KMeans` in `__init__.py` (Probe E). Verification: `maturin develop` + pytest (see below). Underclaim: marshalling/shape contract only — the binding's `n_init=10` signature default + `int64` label dtype diverge (REQ-12); fitted VALUES inherit REQ-9. |
| REQ-5 (convergence criterion + relative tol) | NOT-STARTED | open prereq blocker #1036. sklearn converges on label-no-change OR `(center_shift**2).sum() <= self._tol` with `self._tol = mean(var(X))*tol` (`_kmeans.py:286-294,586-601`); ferrolearn uses absolute `tol` + `max_shift < tol` MAX-euclidean-shift (`fn recompute_centroids_into` + `fn fit`). Different threshold/reduction → different `n_iter_` + stop point. |
| REQ-6 (labels_/inertia_ consistency with final centers) | SHIPPED | impl `Fit::fit` runs a final E-step (`inertia = assign_clusters_into(&mut labels, x, &centers)`) after the Lloyd loop so `labels_`/`inertia_` match the post-swap `cluster_centers_`, mirroring sklearn's post-loop E-step re-run (`_kmeans.py:605-625`). Invariant `fit(X).predict(X) == labels_` now holds. Guard: `pin_req6_predict_equals_labels`. Fixed #1037. |
| REQ-7 (`init` param + random/array/callable + exact k-means++) | PARTIAL | open prereq blocker #1038. sklearn `init ∈ {"k-means++","random"}|callable|array`, default `"k-means++"` (`_kmeans.py:1391`); ferrolearn now exposes standalone `kmeans_plusplus` helpers, but `KMeans` has NO `init` param and always uses greedy k-means++. Default matches; estimator param surface + `random`/array/callable missing; exact seeded output diverges (numpy RNG, REQ-8). |
| REQ-8 (`random_state` numpy-RNG parity) | NOT-STARTED | open prereq blocker #1039. sklearn `check_random_state` + numpy RNG; ferrolearn `StdRng::seed_from_u64(base_seed+run)` (`fn fit`). Different RNG → exact centers/labels/inertia/n_iter cannot match. Depends on a ferray `random` analog (R-SUBSTRATE-5); blocks REQ-9 value parity. |
| REQ-9 (centers/inertia/label-integers/n_iter VALUE parity) | NOT-STARTED | open prereq blocker #1040. sklearn `blobs` → `cluster_centers_=[[10,10.0667],[0,10],[0,0.0667]]`, `inertia_=0.093333`, `n_iter_=2` (Probe A/AC-8). ferrolearn diverges via numpy-RNG (REQ-8), convergence + relative tol (REQ-5), and empty-cluster relocation — sklearn moves an emptied center to the farthest sample (`_relocate_empty_clusters_dense`), ferrolearn keeps the old center (`fn recompute_centroids_into` else-branch). Gated on REQ-5/REQ-8. (REQ-6 post-loop E-step and REQ-14 n_init default are now FIXED — they no longer contribute to this divergence.) |
| REQ-10 (ctor/fit surface init/algorithm/copy_x/verbose/sample_weight + n_clusters=8 + error ABI) | NOT-STARTED | open prereq blocker #1041. sklearn `__init__` (`_kmeans.py:1387-1411`) + `fit(...sample_weight)` + `_check_params_vs_input` (`InvalidParameterError`/`ValueError`, `:875-908`) + `n_features_in_`/`feature_names_in_`. ferrolearn `KMeans<F>` (`fn new`+builders) has `n_clusters/max_iter/tol/n_init/random_state` only; `fn fit` returns `FerroError::InvalidParameter`/`InsufficientSamples` — missing `init`/`algorithm`/`copy_x`/`verbose`/`sample_weight`, `n_clusters` default, introspection attrs; different error ABI. |
| REQ-11 (`score` + `fit_transform`) | NOT-STARTED | open prereq blocker #1042. sklearn `KMeans.score(X)=-inertia` (`_kmeans.py:1156-1184`, Probe D `-0.093333`) + `fit_transform` (`:1106`). ferrolearn `FittedKMeans` has neither (only `fn fit_predict`); PyO3 `RsKMeans` has no `score`/`fit_transform` getter; the Python wrapper's `TransformerMixin`/`ClusterMixin`/`BaseEstimator` do not supply `score`. Absent end-to-end. |
| REQ-12 (ferrolearn-python n_init default + labels_ dtype) | NOT-STARTED | open prereq blocker #1043. PyO3 `RsKMeans::new` signature `n_init=10` (`#[pyo3(signature=(…,n_init=10,…))]`) + Python `ferrolearn.KMeans.__init__` `n_init=10` (`_clusterers.py`) diverge from sklearn's effective `1` for k-means++ (mirrors REQ-9 at the binding, R-DEFER-7). Binding marshals `labels_`/`predict` to `int64` (`ndarray1_usize_to_numpy`), not sklearn `int32` (Probe A). Library fix is REQ-9 `fn new`; this row is the marshalling-layer reflection. |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1044. `kmeans.rs` imports `ndarray::{Array1,Array2}`, `num_traits::Float`, `rand::{rngs::StdRng,Rng,SeedableRng}`, `rayon::prelude::*`; not migrated to `ferray-core`/`ferray::linalg`/`ferray::random` (R-SUBSTRATE-1/2). RNG migration entangled with REQ-8. |
| REQ-14 (`n_init` constructor default = 1) | SHIPPED | impl `fn new` defaults `n_init: 1`, matching sklearn `n_init="auto"` → 1 for the default `init="k-means++"` (`_kmeans.py:886-896`, docstring `:359-361`). Guard: `pin_req14_n_init_default_is_one` (`KMeans::<f64>::new(3).n_init == 1`). Fixed #1045. (The PyO3 binding's `n_init=10` signature default is the separate REQ-12, R-DEFER-7 last layer.) |
| REQ-16 (`kmeans_plusplus` standalone helper) | SHIPPED | impl `pub fn kmeans_plusplus` + `pub fn kmeans_plusplus_with_options` mirror sklearn's public helper shape: `(centers, indices)`, weighted first-center/potential handling, and `n_local_trials=None => 2 + floor(ln(k))`. Consumers: crate re-export. Verification: `tests/divergence_kmeans.rs` pins sklearn's single-sample output, deterministic zero-weight fixture, and public validation constraints; `tests/api_proof.rs` exercises both public functions. Underclaim: no `x_squared_norms` precompute argument, no NumPy-RNG exactness (REQ-8). |

## Architecture

`kmeans.rs` follows the unfitted/fitted split (CLAUDE.md naming): `KMeans<F>`
(`n_clusters: usize`, `max_iter: usize`, `tol: F`, `n_init: usize`, `random_state:
Option<u64>` — all `pub`) → `Fit<Array2<F>, ()>` → `FittedKMeans<F>` (private
`cluster_centers_: Array2<F>`, `labels_: Array1<usize>`, `inertia_: F`, `n_iter_: usize`
with `#[must_use]` accessors). Generic over `F: Float + Send + Sync + 'static`; every
public method returns `Result<_, FerroError>` (R-CODE-2). `FittedKMeans` implements
`Predict<Array2<F>>` (nearest center) and `Transform<Array2<F>>` (euclidean distance to
centers); a `fn fit_predict` convenience mirrors `ClusterMixin.fit_predict`. The
assignment + transform steps parallelize with Rayon above `PARALLEL_WORK_THRESHOLD`
(100k work units). Re-exported `pub use kmeans::{FittedKMeans, KMeans}`
(`ferrolearn-cluster/src/lib.rs`) and bound to CPython via `RsKMeans`
(`ferrolearn-python/src/clusterers.rs`).

**Fit path (`fn fit`).** Validates `n_clusters >= 1`, `n_samples >= 1`,
`n_samples >= n_clusters`, `n_init >= 1` (`FerroError`, REQ-10). For each of `n_init` runs
(seed `base_seed.wrapping_add(run)`, REQ-8):
1. **Seed** — `fn kmeans_plusplus_inner`: first center weighted-random, then each of `k-1`
   remaining centers via greedy k-means++ — at each step sample `n_trials = 2 + floor(ln k)`
   candidates with probability ∝ weighted D(x)^2 and keep the one minimising the weighted
   potential (Arthur–Vassilvitskii + sklearn-style greedy trials). Public wrappers expose
   this as `kmeans_plusplus` / `kmeans_plusplus_with_options` (REQ-16). sklearn
   `_init_centroids` / `kmeans_plusplus` does the same greedy trick but with numpy's RNG
   and supports `random`/array/callable init (REQ-7/REQ-8).
2. **Lloyd loop** (`max_iter`): assign (`fn assign_clusters_into` → `fn nearest_center`,
   accumulating `inertia`), recompute (`fn recompute_centroids_into` — mean of assigned
   samples; EMPTY cluster KEEPS old center, diverging from sklearn's farthest-sample
   relocation, REQ-9), `std::mem::swap(centers, new_centers)`, then converge if
   `max_shift < self.tol` (MAX euclidean shift, ABSOLUTE tol — REQ-5). sklearn instead
   checks label-no-change OR `(center_shift**2).sum() <= self._tol` with relative
   `self._tol = mean(var(X))*tol` (`:286-294,586-601`).
3. **Candidate** — run a final E-step against the converged centers, then store
   `cluster_centers_`, `labels_`, and `inertia_` consistently, mirroring sklearn's
   post-loop E-step (`:605-625`, REQ-6).
Keep the run with the lowest `inertia_` (best-of-`n_init`).

**Predict path (`fn predict`).** Validates feature count, then `fn assign_clusters` →
argmin-squared-euclidean `fn nearest_center` per sample → `Array1<usize>`, mirroring
`KMeans.predict` (`:1072-1104`, REQ-2). The final E-step in `fit` keeps
`fit(X).predict(X) == labels_` consistent (REQ-6).

**Transform path (`fn transform`).** Validates feature count, then
`sqrt(squared_euclidean(row, center))` per `(sample, center)` → `(n_samples, n_clusters)`,
mirroring `_BaseKMeans.transform` (`:1130-1154`, REQ-3 — contract matches, values track
REQ-9).

**Binding (`RsKMeans`).** Mirrors `KMeans::new(n_clusters).with_max_iter/with_tol/
with_n_init/with_random_state`, calls `fit`/`predict`/`transform`, exposes the four
accessors as `#[getter]`s, mapping `FerroError`→`PyValueError` and not-fitted→`PyRuntimeError`
(REQ-4). The signature defaults `n_init=10` (REQ-12) and labels marshal to `int64`
(REQ-12). No `score`/`fit_transform`/`fit_predict` getter (REQ-11), no `init`/`algorithm`/
`sample_weight` (REQ-10).

**Invariants held vs sklearn:** clustering PARTITION (co-membership on separable data,
Probe A / AC-1); `predict` nearest-center rule (`transform.argmin(1)==predict`, AC-2);
`transform` shape `(n,k)` + euclidean metric + column-to-center ordering (AC-3);
`cluster_centers_.shape == (n_clusters, n_features)`; `labels_.len() == n_samples`; labels
in `[0, n_clusters)`; `inertia_ >= 0`; predict/transform shape-mismatch error; deterministic
for a fixed `random_state` (`test_reproducibility_with_seed`); f32 + f64 support; PyO3
round-trip of all four fitted attributes (AC-4); standalone `kmeans_plusplus` shape on
deterministic fixtures (REQ-16).

**Invariants NOT held vs sklearn:** `cluster_centers_`/`inertia_`/`labels_`-integers/
`n_iter_` VALUES (REQ-9); the convergence criterion + relative `tol` (REQ-5);
the `init`/`random`/array/callable estimator surface (REQ-7); numpy-RNG init parity (REQ-8); the
`init`/`algorithm`/`copy_x`/`verbose`/`sample_weight` ctor/fit surface + `n_clusters`=8
default + error ABI + introspection attrs (REQ-10); `score`/`fit_transform` (REQ-11); the
binding `n_init` default + `int32` label dtype (REQ-12); the ferray substrate (REQ-13).

**Consumer wiring.** Non-test consumers: (1) the PyO3 binding `RsKMeans`
(`ferrolearn-python/src/clusterers.rs`, registered in `lib.rs`, wrapped/exported as
`ferrolearn.KMeans`) — a REAL CPython consumer of `fit`/`predict`/`transform`/the four
getters; (2) the crate re-export `pub use kmeans::{FittedKMeans, KMeans,
kmeans_plusplus, kmeans_plusplus_with_options}`
(`ferrolearn-cluster/src/lib.rs`).

## Verification

Library crate:
```
cargo test -p ferrolearn-cluster --lib kmeans     # in-tree KMeans #[test]s
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s (`test_well_separated_blobs`, `test_convergence`,
`test_n_init_picks_best`, `test_kmeans_pp_initialization_deterministic`,
`test_reproducibility_with_seed`, `test_predict_on_new_data`, `test_transform_distances`,
`test_transform_shape`, `test_cluster_centers_shape`, `test_inertia_non_negative`,
`test_k_equals_n_samples`, `test_single_cluster`, `test_single_sample`,
`test_k_greater_than_n_samples`, `test_zero_clusters`, `test_empty_data`,
`test_predict_shape_mismatch`, `test_transform_shape_mismatch`, `test_f32_support`,
`test_two_clusters_on_line`, `test_identical_points`) pin ferrolearn's current behavior —
label co-membership, shapes, ranges, error edges, reproducibility, f32 support,
best-of-`n_init` monotonicity. **None compares `cluster_centers_` values, `labels_`
integers, `inertia_`, `n_iter_`, the `n_init` default, `fit(X).predict(X)==labels_`, the
convergence criterion, or `score` against the live sklearn `KMeans` oracle**, so they stay
green despite the divergences. In particular the partition tests assert
`labels[0]==labels[1]` co-membership and never the absolute integers (masking REQ-9), and
no test exercises the `predict==labels_` consistency invariant (masking REQ-6).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic should
pin (R-CHAR-3 expected values). REQ-14's `n_init` default and REQ-6's
`fit(X).predict(X)==labels_` consistency are already fixed and pinned; REQ-9 exact VALUES
cannot pin until the numpy-RNG analog (REQ-8) lands:
```
# REQ-14 n_init default (observable, isolable, no RNG)
python3 -c "from sklearn.cluster import KMeans; print(KMeans().n_init)"   # auto -> 1 for k-means++
# REQ-6 consistency (no RNG parity needed — must hold for any seed)
python3 -c "import numpy as np; from sklearn.cluster import KMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=KMeans(n_clusters=3, n_init=10, random_state=42).fit(X); print(np.array_equal(m.predict(X), m.labels_))"   # True
# REQ-1 partition + REQ-9 centers/inertia/n_iter
python3 -c "import numpy as np; from sklearn.cluster import KMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=KMeans(n_clusters=3, n_init=10, random_state=42).fit(X); \
print(m.labels_.tolist(), np.round(m.cluster_centers_,4).tolist(), round(m.inertia_,6), m.n_iter_)"
# [2,2,2,0,0,0,1,1,1] [[10,10.0667],[0,10],[0,0.0667]] 0.093333 2
# REQ-11 score
python3 -c "import numpy as np; from sklearn.cluster import KMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=KMeans(n_clusters=3, n_init=10, random_state=42).fit(X); print(round(m.score(X),6), round(-m.inertia_,6))"
# -0.093333 -0.093333
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-cluster/tests/divergence_kmeans.rs`, asserting the live-sklearn expected values
above and FAILING against current `kmeans.rs`. REQ-9 `n_init` and REQ-6 consistency are the
green-after-single-fix targets; REQ-9 exact center/inertia/`n_iter_` values can only be
characterized as a partition/tolerance pin until the numpy-RNG analog (REQ-8),
convergence-criterion (REQ-5), final-E-step (REQ-6), and empty-cluster relocation land.

ferrolearn-python (REQ-4 binding parity; REQ-12 binding default/dtype):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_kmeans.py -q
```
asserting `import ferrolearn; ferrolearn.KMeans(...).fit(blobs)` exposes
`cluster_centers_`/`labels_`/`inertia_`/`n_iter_`/`predict`/`transform` with the right
shapes (REQ-4 green), and pinning `ferrolearn.KMeans().n_init` vs sklearn's effective `1`
and the `int32` label dtype (REQ-12, red until fixed).

## Blockers (to open)

REQ-1 (partition), REQ-2 (predict contract), REQ-3 (transform contract), and REQ-4 (PyO3
binding marshalling) SHIP — no blocker. The remaining NOT-STARTED REQs are the real filed
issues #1036–#1045 (REQ-5 #1036, REQ-6 #1037, REQ-7 #1038, REQ-8 #1039, REQ-9 #1040,
REQ-10 #1041, REQ-11 #1042, REQ-12 #1043, REQ-13 #1044, REQ-14 #1045):

- **#1040** — REQ-9 `n_init` default (**cleanest minimal fix**): set `fn new` default
  `n_init = 1` to mirror sklearn's `n_init="auto"` → `1` for k-means++
  (`_kmeans.py:886-888`, docstring `:359-361`). One-line change; `n_init` is a `pub` field
  so the pin `KMeans::<f64>::new(3).n_init == 1` is deterministic + RNG-free. (Carry the
  same default into the PyO3 `RsKMeans` signature + the Python wrapper — REQ-12.)
- **#1037** — REQ-6 `labels_`/`inertia_` consistency (second-cleanest): re-run the
  assignment against the FINAL `cluster_centers_` before storing `labels_`/`inertia_`
  (mirror the post-loop E-step `_kmeans.py:605-625`) so `fit(X).predict(X) == labels_`.
  Pinnable without RNG parity (must hold for any seed).
- **#1036** — REQ-5 convergence criterion + relative `tol`: scale `tol` by `mean(var(X))`
  (`_tolerance`, `:286-294`) and converge on label-no-change OR `sum(center_shift**2) <=
  tol` (`:586-601`), not `max_shift < tol`.
- **#1038** — REQ-7 `init` param: add `init ∈ {"k-means++","random"}|array|callable`, default
  `"k-means++"` (`:1391`), exposing the missing seeding modes (`_init_centroids`).
- **#1039** — REQ-8 numpy-RNG parity for k-means++ / `random` init (`check_random_state`),
  depends on a ferray `random` analog (R-SUBSTRATE-5); blocks exact REQ-9 value parity.
- **#1040** — REQ-9 `cluster_centers_`/`inertia_`/`labels_`-integers/`n_iter_` value parity —
  gated on REQ-5/REQ-6/REQ-8 + empty-cluster relocation (`_relocate_empty_clusters_dense`).
- **#1041** — REQ-10 ctor/fit surface `init`/`algorithm`/`copy_x`/`verbose`/`sample_weight` +
  `n_clusters`=8 default + `n_features_in_`/`feature_names_in_` + `InvalidParameterError`/
  `ValueError` error ABI (`:1387-1411,875-908`).
- **#1042** — REQ-11 add `score(X) = -inertia` (`_kmeans.py:1156-1184`) + `fit_transform`
  (`:1106`) to `FittedKMeans`, the PyO3 `RsKMeans`, and the Python wrapper.
- **#1043** — REQ-12 ferrolearn-python: set the `RsKMeans` signature + Python wrapper
  `n_init` default to `1` (after REQ-9) and marshal `labels_`/`predict` as `int32`, not
  `int64` (`ndarray1_usize_to_numpy`), to match sklearn's dtype.
- **#1044** — REQ-13 migrate `kmeans.rs` off `ndarray`/`num-traits`/`rand`/`rayon` to
  `ferray-core`/`ferray::linalg`/`ferray::random` (R-SUBSTRATE) — RNG migration entangled
  with REQ-8.
