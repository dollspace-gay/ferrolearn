# Mini-Batch K-Means (sklearn.cluster.MiniBatchKMeans)

<!--
tier: 3-component
status: draft
baseline-commit: cbaca6e5
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_kmeans.py            # class MiniBatchKMeans(_BaseKMeans) (:1687); __init__ (:1890-1920) defaults init="k-means++"/max_iter=100/batch_size=1024/verbose=0/compute_labels=True/tol=0.0/max_no_improvement=10/init_size=None/n_init="auto"/reassignment_ratio=0.01; MiniBatchKMeans._check_params_vs_input (:1922-1951) super().default_n_init=3 + _init_size = 3*batch_size heuristic (:1927-1944); _BaseKMeans._check_params_vs_input (:875-908) n_init="auto"+"k-means++" -> _n_init=1; _mini_batch_step (:1566-1684) low-count reassignment (:1652-1682); _mini_batch_convergence (:1964-2006+) EWA inertia + max_no_improvement early stop; fit (:2046-2199) best-of-n_init init on init_size subsample (:2099-2132) + single mini-batch optimization (:2148-2181) + n_steps_/n_iter_ (:2186-2187) + final compute_labels E-step (:2189-2195); partial_fit (:2202+); transform/predict inherited from _BaseKMeans (:1072,:1130)
ferrolearn-module: ferrolearn-cluster/src/mini_batch_kmeans.rs
ferrolearn-python: ferrolearn-python/src/extras.rs (RsMiniBatchKMeans, name="_RsMiniBatchKMeans")
parity-ops: MiniBatchKMeans (.__init__, .fit, .predict, .transform, .partial_fit, .cluster_centers_, .labels_, .inertia_, .n_iter_, .n_steps_)
crosslink-issue: 1046
-->

## Summary

`ferrolearn-cluster/src/mini_batch_kmeans.rs` mirrors scikit-learn's `MiniBatchKMeans`
(`sklearn/cluster/_kmeans.py`, `class MiniBatchKMeans(_BaseKMeans)` `:1687`) — the online
mini-batch variant of K-Means. It exposes the unfitted `MiniBatchKMeans<F>` (`n_clusters`,
`batch_size`, `max_iter`, `tol`, `n_init`, `random_state`, `init` via the
`MiniBatchKMeansInit` enum — all `pub`), the fitted `FittedMiniBatchKMeans<F>` (private
`cluster_centers_`, `labels_`, `inertia_`, `n_iter_` with `#[must_use]` accessors), a
`Predict` impl (nearest center), a `Transform` impl (euclidean distance to centers), and a
`fn fit_predict` convenience mirroring `ClusterMixin.fit_predict`. It is re-exported at the
crate root (`pub use mini_batch_kmeans::{FittedMiniBatchKMeans, MiniBatchKMeans,
MiniBatchKMeansInit}` in `ferrolearn-cluster/src/lib.rs`) AND — like the sibling-audited
`kmeans` unit — has a REAL CPython consumer: the PyO3 binding `RsMiniBatchKMeans`
(`#[pyclass(name = "_RsMiniBatchKMeans")]`) in `ferrolearn-python/src/extras.rs`, registered
in `ferrolearn-python/src/lib.rs` (`m.add_class::<extras::RsMiniBatchKMeans>()`) and wrapped
by the sklearn-API class `ferrolearn.MiniBatchKMeans`
(`ferrolearn-python/python/ferrolearn/_extras.py`, re-exported in `__init__.py`). So
`import ferrolearn; ferrolearn.MiniBatchKMeans()` reaches this code.

**Note (R-HONEST-3): the `RsMiniBatchKMeans` binding is MARKEDLY THINNER than the sibling
`RsKMeans`** — it exposes only `fit`, `predict`, and the `labels_` getter, with a
constructor signature of just `(n_clusters=8, max_iter=100, random_state=None)`. It does
NOT expose `transform`, `cluster_centers_`, `inertia_`, or `n_iter_`, and does NOT surface
`n_init`/`batch_size`/`tol` constructor params (so the library `fn new` default `n_init=3`
is what the binding silently uses).

**Under honest assessment (R-HONEST-3), FIVE REQs SHIP: the clustering PARTITION
up-to-permutation on well-separated data (REQ-1, via library + crate re-export, and the
`labels_`/`fit`/`predict` slice of it through the binding), the `predict` nearest-center
CONTRACT (REQ-2), the `transform` distance-to-centers CONTRACT (REQ-3, library + crate
re-export only — NOT exposed through the binding), the partial PyO3 binding marshalling
of `fit`/`predict`/`labels_` (REQ-4), and the `n_init` constructor default = 1 (REQ-5,
fixed iter 128 / #1047).** A notable POSITIVE divergence from the sibling
KMeans audit: ferrolearn's `fn fit` computes its final `labels_`/`inertia_` via
`assign_clusters_mb` against the FINAL stored `cluster_centers_`, so the
`fit(X).predict(X) == labels_` consistency invariant ALREADY HOLDS here (no separate fix
needed, unlike KMeans REQ-6 / #1037) — it is folded into REQ-2. Everything tied to exact
VALUES (centers, inertia, absolute label integers, `n_iter_`), the algorithm structure,
several constructor params/attributes, and the binding surface DIVERGE, for the reasons
carved out below:

1. **`n_init` DEFAULT (R-DEV-2) — REQ-5, clean minimal fix.** sklearn
   `MiniBatchKMeans(n_init="auto")` (`_kmeans.py:1903`), which for the default
   `init="k-means++"` resolves to **1** (`_BaseKMeans._check_params_vs_input` `:886-888`:
   `n_init=="auto"` & `init` str `"k-means++"` → `self._n_init = 1`; docstring `:1778-1780`;
   `MiniBatchKMeans._check_params_vs_input` passes `default_n_init=3` `:1923`, but that
   `default_n_init` only applies to `init="random"`/callable, NOT to the default
   `"k-means++"`). ferrolearn `fn new` defaults `n_init = 3` (the module comment at
   `:102-106` claiming "`batch_size`, `max_iter`, and `tol` match scikit-learn ≥ 1.4
   defaults" is CORRECT for those three but the SEPARATE `n_init = 3` is a MIS-TRANSLATION:
   the resolved sklearn default for the default k-means++ init is **1**, not 3 — flag
   R-HONEST-4). `n_init` is a `pub` field on `MiniBatchKMeans<F>`, pinnable by
   `MiniBatchKMeans::<f64>::new(3).n_init == 1` (deterministic, RNG-free) — the cleanest
   minimal fix. (`batch_size=1024`, `max_iter=100`, `tol=0.0` ALL MATCH sklearn `:1895-1900`
   — confirmed Probe A.)
2. **Algorithm STRUCTURE diverges (R-DEV-1).** sklearn runs ONE k-means++ init chosen as the
   best of `n_init` INIT TRIALS, each evaluated on a SUBSAMPLE of `init_size` points
   (`init_size = 3*batch_size` default, `:1929-1944`, validation inertia `:2120-2132`), then
   runs ONE mini-batch optimization over `n_steps = (max_iter * n_samples) // batch_size`
   steps (`:2148`) with EWA-inertia convergence + early stopping via `max_no_improvement`
   consecutive non-improving steps (`_mini_batch_convergence` `:1964-2006+`). ferrolearn
   `fn fit` instead runs `n_init` FULL mini-batch RUNS (each up to `max_iter` steps,
   per-center learning rate `1/count`, `max_shift < tol` convergence) and keeps the
   lowest-inertia run. Different `n_init` semantics (init-trials vs full-runs), different
   convergence (EWA early-stop vs max-shift), and no `init_size` subsampling. NOT-STARTED
   (substantial, RNG-entangled).
3. **Low-count cluster REASSIGNMENT (R-DEV-1).** sklearn, inside `_mini_batch_step`,
   reassigns centers whose `weight_sums` fall below `reassignment_ratio * weight_sums.max()`
   to random observations (`:1652-1682`); ferrolearn `fn update_centers_mini_batch` has NO
   reassignment. NOT-STARTED.
4. **`n_steps_` attribute (R-DEV-3).** sklearn exposes BOTH `n_steps_` (number of minibatches
   processed, `:2186`) AND `n_iter_` (`= ceil(n_steps * batch_size / n_samples)`, `:2187`);
   ferrolearn exposes only `n_iter_` (and computes it as the raw mini-batch step count, not
   sklearn's full-dataset-pass formula). NOT-STARTED.
5. **`partial_fit` (online) (R-DEV-3).** sklearn has `MiniBatchKMeans.partial_fit(X)`
   (`:2202+`) for incremental online updates; ferrolearn has no `partial_fit`. NOT-STARTED.
6. **missing constructor/fit params (R-DEV-2).** `max_no_improvement`, `reassignment_ratio`,
   `init_size`, `compute_labels`, `verbose`, `sample_weight`, `n_clusters`=8 default — all
   absent from `MiniBatchKMeans<F>` / `fn fit`. NOT-STARTED.
7. **numpy-RNG parity (R-DEV-1).** sklearn `check_random_state` + numpy RNG
   (`random_state.randint` for minibatch indices `:2154`, init subsample `:2099`);
   ferrolearn `StdRng::seed_from_u64` per run (`fn fit`). Exact
   `cluster_centers_`/`labels_`/`inertia_` cannot match without a numpy-RNG analog.
   NOT-STARTED.
8. **error ABI / `_check_params_vs_input` (R-DEV-2).** sklearn validates via
   `_parameter_constraints` / `_check_params_vs_input` (`ValueError` /
   `InvalidParameterError`, `:875-908,1922-1951`); ferrolearn returns
   `FerroError::InvalidParameter`/`InsufficientSamples` (`fn fit`). Missing
   `n_features_in_`/`feature_names_in_`; `n_clusters` has no default. NOT-STARTED.
9. **`init` param surface (R-DEV-1/2).** ferrolearn HAS an `init` param
   (`MiniBatchKMeansInit::{KMeansPlusPlus, Random}`, `fn with_init`) and the default matches
   sklearn (`"k-means++"`); BUT the callable/array-like `init` paths and exact k-means++
   output (numpy RNG) diverge. Folded into REQ-2 (default match) / REQ-6 (param surface) /
   REQ-7 (RNG).
10. **ferray substrate (R-SUBSTRATE).** `mini_batch_kmeans.rs` imports
    `ndarray::{Array1,Array2}`, `num_traits::Float`, `rand::{rngs::StdRng,…}`, `rayon`, not
    `ferray-core`/`ferray::linalg`/`ferray::random`. NOT-STARTED.
11. **thin binding surface (R-DEV-3, R-DEFER-7).** `RsMiniBatchKMeans` exposes only
    `fit`/`predict`/`labels_` — no `transform`, `cluster_centers_`, `inertia_`, `n_iter_`
    getters, and no `n_init`/`batch_size`/`tol` ctor params. `labels_`/`predict` marshal to
    `int64` (`ndarray1_usize_to_numpy`), not sklearn's `int32`. NOT-STARTED.

## Live oracle probes (sklearn 1.5.2, run from /tmp)

Expected values are from the installed sklearn 1.5.2 oracle, never literal-copied from
ferrolearn (R-CHAR-3). Fixture `blobs` = the 9×2 well-separated 3-blob set (3 points near
`(0,0)`, 3 near `(10,10)`, 3 near `(0,10)`) — the `make_blobs()` fixture in
`mini_batch_kmeans.rs`.

### Probe A — default constructor surface + n_init resolution
```
python3 -c "from sklearn.cluster import MiniBatchKMeans; import numpy as np; \
d=MiniBatchKMeans(); \
print(d.init, d.n_init, d.batch_size, d.max_iter, d.tol, d.n_clusters, d.max_no_improvement, d.reassignment_ratio, d.init_size, d.compute_labels); \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
print('_n_init=', MiniBatchKMeans(n_clusters=3, random_state=42).fit(X)._n_init)"
# k-means++ auto 1024 100 0.0 8 10 0.01 None True
# _n_init= 1
```
**Findings:** sklearn defaults `init="k-means++"`, `n_init="auto"` (resolving to **1** for
k-means++ via `_check_params_vs_input` `:886-888`, docstring `:1778-1780`),
`batch_size=1024`, `max_iter=100`, `tol=0.0`, `n_clusters=8`, plus
`max_no_improvement=10`/`reassignment_ratio=0.01`/`init_size=None`/`compute_labels=True`.
ferrolearn `fn new`: `batch_size=1024` (matches), `max_iter=100` (matches), `tol=0.0`
(matches), `n_init=3` (sklearn-equivalent is **1** for k-means++ — REQ-5, the
mis-translation flagged at `:102-106`), `init=KMeansPlusPlus` (matches), no
`max_no_improvement`/`reassignment_ratio`/`init_size`/`compute_labels`/`verbose`
(REQ-6), and no `n_clusters` default (Rust builder requires the argument). The PyO3
`RsMiniBatchKMeans` signature `(n_clusters=8, max_iter=100, random_state=None)` exposes
neither `n_init` nor `batch_size`/`tol`, so it silently uses the library `n_init=3` default
(REQ-11).

### Probe B — well-separated 3-blob: labels_ / centers / inertia_ / n_iter_ / n_steps_ / dtype
```
python3 -c "import numpy as np; from sklearn.cluster import MiniBatchKMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=MiniBatchKMeans(n_clusters=3, n_init=3, random_state=42, batch_size=9).fit(X); \
print(m.labels_.tolist(), np.round(m.cluster_centers_,4).tolist(), round(m.inertia_,6), m.n_iter_, m.n_steps_, m.labels_.dtype)"
# labels   [0, 0, 0, 1, 1, 1, 2, 2, 2]
# centers  [[-0.0065, 0.0613], [10.0041, 10.0612], [-0.0043, 9.9957]]
# inertia  0.093797
# n_iter_  20
# n_steps_ 20
# dtype    int32
```
**Findings:** sklearn groups `{0,1,2}` / `{3,4,5}` / `{6,7,8}` (REQ-1, the partition).
ferrolearn `MiniBatchKMeans::<f64>::new(3).with_random_state(42).with_n_init(5)
.with_batch_size(9).fit(blobs)` recovers the SAME 3-way grouping — verified by the in-tree
`test_well_separated_blobs` asserting `labels[0]==labels[1]==labels[2]` etc. — but the label
INTEGERS are permuted (different RNG/init order), the `cluster_centers_` VALUES diverge (RNG
+ structure + reassignment), `inertia_` need not equal `0.093797`, `n_iter_` diverges
(ferrolearn counts mini-batch steps directly; sklearn computes
`ceil(n_steps*batch_size/n_samples)` and ALSO exposes `n_steps_` which ferrolearn lacks —
REQ-9/REQ-8), and `labels_` is `usize` (binding → `int64`), not sklearn's `int32` (REQ-11).

### Probe C — transform shape, argmin==predict, predict==labels_ (consistency)
```
python3 -c "import numpy as np; from sklearn.cluster import MiniBatchKMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=MiniBatchKMeans(n_clusters=3, n_init=3, random_state=42, batch_size=9).fit(X); T=m.transform(X); \
print(T.shape, np.array_equal(T.argmin(1), m.predict(X)), np.array_equal(m.predict(X), m.labels_))"
# (9, 3)  argmin==predict True   predict==labels_ True
```
sklearn `transform` returns euclidean distance from each sample to each `cluster_centers_`
row, shape `(n_samples, n_clusters)`; `transform(X).argmin(axis=1) == predict(X)` and
`predict(X) == labels_` (labels are computed against the final centers via the post-loop
`compute_labels` E-step `:2189-2195`). ferrolearn `Transform::transform` (`fn transform`)
computes the same metric/shape/column-ordering (REQ-3), `Predict::predict` (`fn predict`) is
`argmin` over `assign_clusters_mb` (REQ-2 — argmin==predict holds by construction), AND
`predict(X) == labels_` ALSO HOLDS in ferrolearn because `fn fit` computes its final
`labels_`/`inertia_` via `assign_clusters_mb(x, &centers)` against the SAME final stored
`cluster_centers_` (unlike the sibling KMeans, which needed the #1037 fix) — folded into
REQ-2.

### Probe D — n_steps_ + partial_fit existence (attribute / method surface)
```
python3 -c "from sklearn.cluster import MiniBatchKMeans; m=MiniBatchKMeans(n_clusters=2); \
print('n_steps_ documented:', 'n_steps_' in MiniBatchKMeans.__doc__, 'has partial_fit:', hasattr(m, 'partial_fit'))"
# n_steps_ documented: True   has partial_fit: True
```
sklearn exposes `n_steps_` (`:1816-1819`, `:2186`) and `partial_fit` (`:2202+`).
ferrolearn `FittedMiniBatchKMeans` (`impl FittedMiniBatchKMeans`) exposes only
`cluster_centers`/`labels`/`inertia`/`n_iter` accessors — no `n_steps_` (REQ-8) — and
`MiniBatchKMeans<F>` has no `partial_fit` (REQ-10).

### Probe E — PyO3 binding chain (the real CPython consumer)
- `#[pyclass(name = "_RsMiniBatchKMeans")]` `RsMiniBatchKMeans` in
  `ferrolearn-python/src/extras.rs` exposes `new`/`fit`/`predict` + the `labels_` getter
  ONLY, delegating to `ferrolearn_cluster::MiniBatchKMeans::<f64>` /
  `FittedMiniBatchKMeans::<f64>`. It does NOT expose `transform`, `cluster_centers_`,
  `inertia_`, or `n_iter_`. The `new` signature is `(n_clusters=8, max_iter=100,
  random_state=None)` — no `n_init`/`batch_size`/`tol`.
- Registered: `m.add_class::<extras::RsMiniBatchKMeans>()?` in
  `ferrolearn-python/src/lib.rs`.
- Python export: `from ferrolearn._ferrolearn_rs import _RsMiniBatchKMeans` in
  `ferrolearn-python/python/ferrolearn/_extras.py`, wrapped by
  `class MiniBatchKMeans(_ClusterWrapper)` (its `_make_rs` constructs `_RsMiniBatchKMeans`),
  re-exported as `MiniBatchKMeans` in `ferrolearn-python/python/ferrolearn/__init__.py`
  (`from ferrolearn._extras import MiniBatchKMeans`; `"MiniBatchKMeans"` in `__all__`).
So `import ferrolearn; ferrolearn.MiniBatchKMeans(...).fit(X)` exercises `fn fit` → the
binding is a genuine non-test production consumer of `fit`/`predict`/`labels_` (but not of
`transform`/`cluster_centers_`/`inertia_`/`n_iter_`).

## Requirements

- REQ-1: **clustering PARTITION up-to-permutation (R-DEV-1).** Mirror
  `MiniBatchKMeans(n_clusters=k, n_init=N, random_state=s, batch_size=b).fit(X)` producing
  the same grouping of samples into clusters on well-separated data. ferrolearn `fn fit`
  (k-means++ seed `fn kmeans_plus_plus_mb` → mini-batch assign/update loop over `max_iter`
  steps `fn assign_batch`/`fn update_centers_mini_batch` → best-of-`n_init` lowest-inertia)
  recovers the correct PARTITION on separable blobs (Probe B) — but the `labels_` INTEGERS
  (REQ-7), `cluster_centers_` VALUES (REQ-7/REQ-9), `inertia_`, and `n_iter_` diverge.
  Partition-only, not value parity.
- REQ-2: **`predict` nearest-center CONTRACT + `predict==labels_` consistency (R-DEV-3).**
  Mirror `_BaseKMeans.predict` = assign each sample to the argmin-distance cluster center
  (`_kmeans.py:1072`, returns `int32` labels). ferrolearn `Predict::predict` (`fn predict`
  → `fn assign_clusters_mb`) returns the argmin-squared-euclidean center index, shape
  `(n_samples,)`, with a shape-mismatch `FerroError::ShapeMismatch`, so
  `transform(X).argmin(axis=1) == predict(X)` (Probe C). Additionally `fit(X).predict(X) ==
  labels_` HOLDS because `fn fit` computes final `labels_` via `assign_clusters_mb` against
  the final stored `cluster_centers_` (no separate fix needed, unlike KMeans). CONTRACT
  matches; absolute label INTEGERS track REQ-7, dtype is `usize`/`i64` not `int32` (REQ-11).
- REQ-3: **`transform` distance-to-centers CONTRACT (R-DEV-3).** Mirror
  `_BaseKMeans.transform` = euclidean distance from each sample to each `cluster_centers_`
  row, shape `(n_samples, n_clusters)` (`_kmeans.py:1130`). ferrolearn `Transform::transform`
  (`fn transform`) returns `sqrt(squared_euclidean_mb(row, center))` per `(sample, center)`
  in the same shape/column ordering (Probe C). CONTRACT matches (library + crate re-export);
  NOT exposed through the binding (REQ-11); the distance VALUES track the `cluster_centers_`
  divergence (REQ-7/REQ-9).
- REQ-4: **PyO3 binding marshalling of `fit`/`predict`/`labels_` (R-DEFER-1).**
  `import ferrolearn; ferrolearn.MiniBatchKMeans` reaches `fn fit`/`predict` and the
  `labels_` getter through `RsMiniBatchKMeans` (Probe E). The fit→predict→`labels_`
  marshalling round-trips arrays across the PyO3 boundary, preserving shape (labels `(n,)`).
  Underclaim: the binding marshals ONLY this slice — `transform`/`cluster_centers_`/
  `inertia_`/`n_iter_` and the `n_init`/`batch_size`/`tol` ctor params are NOT exposed
  (REQ-11).
- REQ-5: **`n_init` constructor default = 1 (R-DEV-2).** sklearn `MiniBatchKMeans(n_init=
  "auto")` (`_kmeans.py:1903`) resolves to **1** for the default `init="k-means++"`
  (`_BaseKMeans._check_params_vs_input` `:886-888`, docstring `:1778-1780`; the
  `default_n_init=3` passed at `:1923` applies only to `init="random"`/callable, not to
  k-means++). ferrolearn `fn new` defaults `n_init = 3` — a MIS-TRANSLATION (the module
  comment `:102-106` is correct only about `batch_size`/`max_iter`/`tol`; R-HONEST-4).
  `n_init` is a `pub` field, pinned by `MiniBatchKMeans::<f64>::new(3).n_init == 1`
  (deterministic, RNG-free — the cleanest minimal fix).
- REQ-6: **constructor/fit surface `max_no_improvement`/`reassignment_ratio`/`init_size`/
  `compute_labels`/`verbose`/`sample_weight` + `n_clusters`=8 + `init` callable/array + error
  ABI (R-DEV-2).** sklearn `__init__` (`_kmeans.py:1890-1920`) + `fit(X, y, sample_weight)`
  + `_check_params_vs_input` (`InvalidParameterError`/`ValueError`, `:875-908,1922-1951`) +
  `n_features_in_`/`feature_names_in_`. ferrolearn `MiniBatchKMeans<F>` (`fn new` + builders)
  has `n_clusters/batch_size/max_iter/tol/n_init/random_state/init` only; `fn fit` returns
  `FerroError::InvalidParameter`/`InsufficientSamples`. Missing `max_no_improvement`/
  `reassignment_ratio`/`init_size`/`compute_labels`/`verbose`/`sample_weight`, the `init`
  callable/array paths, `n_clusters` default, the introspection attrs; different error ABI.
- REQ-7: **`random_state` numpy-RNG parity (R-DEV-1).** sklearn `check_random_state` + numpy
  RNG (`random_state.randint` for minibatch indices `:2154` + init subsample `:2099`);
  ferrolearn `StdRng::seed_from_u64(base_seed + run*1_000_003)` per run (`fn fit`). Different
  RNG → exact `cluster_centers_`/`labels_`/`inertia_`/`n_iter_` cannot match. Depends on a
  ferray `random` analog (R-SUBSTRATE-5); BLOCKS exact VALUE parity (REQ-9).
- REQ-8: **algorithm STRUCTURE: init-trials-on-subsample + EWA convergence + early stopping +
  `n_steps_`/`n_iter_` formula (R-DEV-1).** sklearn runs ONE k-means++ init chosen as the
  best of `n_init` INIT TRIALS on an `init_size`-sample subsample (`:2099-2132`,
  `init_size = 3*batch_size` `:1929`), then ONE mini-batch optimization over
  `n_steps = (max_iter*n_samples)//batch_size` (`:2148`) with EWA-inertia convergence +
  `max_no_improvement` early stop (`_mini_batch_convergence` `:1964-2006+`), exposing
  `n_steps_ = i+1` (`:2186`) and `n_iter_ = ceil(n_steps*batch_size/n_samples)` (`:2187`).
  ferrolearn `fn fit` runs `n_init` FULL mini-batch runs (`max_iter` steps each,
  `max_shift < tol` convergence) keeping the lowest-inertia run, exposes only `n_iter_` (raw
  step count), and has no `init_size`/EWA/early-stop/`n_steps_`. Substantial structural
  divergence.
- REQ-9: **`cluster_centers_` / `inertia_` / `labels_`-integers / `n_iter_` VALUE parity
  (R-DEV-1).** Mirror sklearn's exact center coordinates, total inertia, absolute label
  integers, and iteration count (Probe B: `cluster_centers_ =
  [[-0.0065,0.0613],[10.0041,10.0612],[-0.0043,9.9957]]`, `inertia_ = 0.093797`,
  `n_iter_ = 20`, `n_steps_ = 20`). Diverges because of numpy-RNG init (REQ-7), the algorithm
  structure + convergence (REQ-8), the `n_init` default (REQ-5), and missing low-count
  reassignment (REQ-12). Gated on REQ-7/REQ-8.
- REQ-10: **`partial_fit` (online incremental update) (R-DEV-3).** sklearn
  `MiniBatchKMeans.partial_fit(X)` (`_kmeans.py:2202+`) updates the estimate on a single
  mini-batch, accumulating `_counts`/`n_steps_` across calls. ferrolearn has no `partial_fit`
  — `MiniBatchKMeans<F>` offers only `fit`/`fit_predict`. Absent end-to-end.
- REQ-11: **thin binding surface + `labels_` dtype (R-DEV-3, R-DEFER-7).** `RsMiniBatchKMeans`
  (`extras.rs`) exposes ONLY `fit`/`predict`/`labels_` — NO `transform`/`cluster_centers_`/
  `inertia_`/`n_iter_` getters, and its `new` signature `(n_clusters=8, max_iter=100,
  random_state=None)` omits `n_init`/`batch_size`/`tol` (so it uses the library `n_init=3`
  default — REQ-5 at the binding). It marshals `labels_`/`predict` to `int64`
  (`ndarray1_usize_to_numpy`), not sklearn's `int32` (Probe B). This crate is the last
  translation layer (R-DEFER-7).
- REQ-12: **low-count cluster REASSIGNMENT (R-DEV-1).** sklearn `_mini_batch_step`
  (`:1652-1682`) reassigns centers whose `weight_sums` fall below
  `reassignment_ratio * weight_sums.max()` to random observations; ferrolearn
  `fn update_centers_mini_batch` performs no reassignment. Different centers when a cluster
  starves. Contributes to REQ-9 value divergence.
- REQ-13: **ferray substrate (R-SUBSTRATE).** `mini_batch_kmeans.rs` imports
  `ndarray::{Array1,Array2}`, `num_traits::Float`, `rand::{rngs::StdRng,Rng,SeedableRng}`,
  `rayon::prelude::*`, not `ferray-core`/`ferray::linalg`/`ferray::random`
  (R-SUBSTRATE-1/2). The RNG migration is entangled with REQ-7.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), `n_init`/
`batch_size` set explicitly in value probes, never literal-copied from ferrolearn
(R-CHAR-3). Fixture `blobs` = `make_blobs()` (9×2).

- AC-1 (REQ-1, partition agrees / integers diverge): `MiniBatchKMeans(n_clusters=3,
  n_init=3, random_state=42, batch_size=9).fit(blobs).labels_` → sklearn `[0,0,0,1,1,1,2,2,2]`;
  ferrolearn recovers the same 3-way grouping (`{0,1,2}`/`{3,4,5}`/`{6,7,8}`) up to a
  permutation of the integers.
- AC-2 (REQ-2, contract + consistency): `MiniBatchKMeans(...).fit(blobs)`:
  `transform(blobs).argmin(axis=1) == predict(blobs)` (sklearn `True`) AND `predict(blobs)
  == labels_` (sklearn `True`, Probe C); ferrolearn `Predict::predict` is argmin over the
  same distances, and `fn fit` assigns `labels_` against the final centers so both hold.
  Shape `(9,)`, shape-mismatch raises.
- AC-3 (REQ-3, contract): `MiniBatchKMeans(n_clusters=3, n_init=3, random_state=42,
  batch_size=9).fit(blobs).transform(blobs).shape` → sklearn `(9, 3)`, column `j` =
  `||x_i - cluster_centers_[j]||`. ferrolearn returns the same shape + per-center euclidean
  distance; the values track the `cluster_centers_` divergence (REQ-9).
- AC-4 (REQ-4, binding): `import ferrolearn; m=ferrolearn.MiniBatchKMeans(n_clusters=3,
  random_state=42).fit(blobs)` exposes `m.labels_` `(9,)` and `m.predict(blobs)` `(9,)`,
  round-tripping shapes across the PyO3 boundary. (`m.cluster_centers_`/`m.inertia_`/
  `m.n_iter_`/`m.transform` are NOT exposed — REQ-11.)
- AC-5 (REQ-5, n_init default): `MiniBatchKMeans()._n_init` resolves to `1` for
  `init="k-means++"` (sklearn `_check_params_vs_input` `:886-888`, docstring `:1778-1780`;
  Probe A). ferrolearn `MiniBatchKMeans::<f64>::new(3).n_init == 1` is the target; current
  `fn new` gives `3` (deterministic, RNG-free pin — the cleanest minimal fix).
- AC-6 (REQ-8, structure / n_steps_): sklearn `MiniBatchKMeans(n_clusters=3, n_init=3,
  random_state=42, batch_size=9).fit(blobs)` exposes `n_steps_ = 20` and
  `n_iter_ = ceil(20*9/9) = 20` (Probe B/D); the one optimization runs over
  `n_steps = (max_iter*n_samples)//batch_size` steps with EWA/`max_no_improvement` early
  stopping. ferrolearn has no `n_steps_`, runs `n_init` full runs, and counts `n_iter_` as
  raw steps.
- AC-7 (REQ-10, partial_fit): sklearn `MiniBatchKMeans(...).partial_fit(X[:6]).partial_fit
  (X[6:])` incrementally updates `cluster_centers_`/`n_steps_` (`:2202+`). ferrolearn has no
  `partial_fit`.
- AC-8 (REQ-9 value parity, diverges): `MiniBatchKMeans(n_clusters=3, n_init=3,
  random_state=42, batch_size=9).fit(blobs)` → sklearn `cluster_centers_ =
  [[-0.0065,0.0613],[10.0041,10.0612],[-0.0043,9.9957]]`, `inertia_ = 0.093797`,
  `n_iter_ = 20`. ferrolearn produces different center VALUES, inertia, label integers, and
  `n_iter_` (RNG + structure + convergence + no reassignment).

## REQ status table

Binary (R-DEFER-2). `MiniBatchKMeans` / `FittedMiniBatchKMeans` are existing pub APIs
re-exported at the crate root and consumed by the PyO3 binding `RsMiniBatchKMeans` (a REAL
non-test CPython consumer; grandfathered S5/R-DEFER-1). Cites use symbol anchors (ferrolearn)
/ `file:line` (sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2, run
from `/tmp`. Honest assessment (R-HONEST-3): **FIVE REQs SHIP** — the clustering PARTITION
up-to-permutation (REQ-1), the `predict` nearest-center + `predict==labels_` CONTRACT (REQ-2),
the `transform` distance-to-centers CONTRACT (REQ-3), the partial PyO3 binding
marshalling of `fit`/`predict`/`labels_` (REQ-4), and the `n_init` default = 1 (REQ-5, fixed
#1047). Exact VALUES, the algorithm structure, numpy-RNG parity, `partial_fit`, low-count reassignment, the binding
surface/dtype, the missing-params/`n_steps_` surface, and the ferray substrate all DIVERGE.
Blocker numbers below are the real filed issues (#1047–#1055).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (clustering PARTITION up-to-permutation) | SHIPPED | impl `fn fit` (greedy k-means++ `fn kmeans_plus_plus_mb` seed → mini-batch `fn assign_batch`/`fn update_centers_mini_batch` loop → final `fn assign_clusters_mb` → best-of-`n_init` lowest-inertia) recovers sklearn's grouping on well-separated data (Probe B: sklearn `[0,0,0,1,1,1,2,2,2]`). Consumers: PyO3 `RsMiniBatchKMeans::fit` (`extras.rs`, registered in `lib.rs`) AND crate re-export `pub use mini_batch_kmeans::{FittedMiniBatchKMeans, MiniBatchKMeans, MiniBatchKMeansInit}` (`lib.rs`). Guards: in-tree `test_well_separated_blobs`, `test_single_cluster`, `test_identical_points` assert co-membership. Verification: `cargo test -p ferrolearn-cluster --lib mini_batch` (21 passed). Underclaim: PARTITION up-to-permutation only — `labels_` INTEGERS, `cluster_centers_`/`inertia_` VALUES, `n_iter_` diverge (REQ-9). |
| REQ-2 (`predict` nearest-center contract + `predict==labels_`) | SHIPPED | impl `Predict::predict` (`fn predict` → `fn assign_clusters_mb`) returns the argmin-squared-euclidean center index, shape `(n_samples,)`, with a shape-mismatch `FerroError::ShapeMismatch`; `transform(X).argmin(1) == predict(X)`; AND `fit(X).predict(X) == labels_` holds because `fn fit` stores final `labels_` from `assign_clusters_mb(x, &centers)` against the SAME final `cluster_centers_` (Probe C — unlike sibling KMeans, no separate fix needed). Consumers: PyO3 `RsMiniBatchKMeans::predict` (`ndarray1_usize_to_numpy`) AND crate re-export. Guards: `test_predict_on_new_data`, `test_transform_distances_structure`, `test_predict_shape_mismatch_error`. Underclaim: CONTRACT only — label INTEGERS track REQ-7, dtype `usize`/`i64` not `int32` (REQ-11). |
| REQ-3 (`transform` distance-to-centers contract) | SHIPPED | impl `Transform::transform` (`fn transform`) returns `sqrt(squared_euclidean_mb(row, center))` per `(sample, center)`, shape `(n_samples, n_clusters)`, column `j` = distance to `cluster_centers_[j]` — matching sklearn `_BaseKMeans.transform` (`_kmeans.py:1130`) shape/metric/ordering (Probe C: `(9,3)`, argmin==predict). Consumer: crate re-export `pub use mini_batch_kmeans::…` (`lib.rs`). Guards: `test_transform_shape`, `test_transform_distances_structure`, `test_transform_shape_mismatch_error`. Underclaim: CONTRACT only, library + crate re-export — NOT exposed through the PyO3 binding (REQ-11); distance VALUES track the `cluster_centers_` divergence (REQ-7/REQ-9). |
| REQ-4 (PyO3 binding marshalling of fit/predict/labels_) | SHIPPED | impl `RsMiniBatchKMeans` (`#[pyclass(name="_RsMiniBatchKMeans")]`, `extras.rs`) marshals `fit`/`predict` + the `labels_` getter over the PyO3 boundary, preserving label shape `(n,)`. Consumer (the binding IS the CPython consumer): registered `m.add_class::<extras::RsMiniBatchKMeans>()` (`ferrolearn-python/src/lib.rs`) → wrapped by `class MiniBatchKMeans(_ClusterWrapper)` (`_extras.py`) → re-exported `MiniBatchKMeans` in `__init__.py` (Probe E). Verification: `maturin develop` + pytest (see below). Underclaim: marshalling of `fit`/`predict`/`labels_` ONLY — no `transform`/`cluster_centers_`/`inertia_`/`n_iter_` getter, no `n_init`/`batch_size`/`tol` ctor param, `int64` (not `int32`) labels (REQ-11). |
| REQ-5 (`n_init` constructor default = 1) | SHIPPED | impl `fn new` defaults `n_init: 1`, matching sklearn `MiniBatchKMeans(n_init="auto")` → 1 for the default `init="k-means++"` (`_kmeans.py:886-888`; live oracle `_n_init==1`). Guard: `pin_req5_n_init_default_is_one`. Fixed #1047; the mis-translating `fn new` comment (`:102-106`) corrected (R-HONEST-4). (`batch_size=1024`/`max_iter=100`/`tol=0.0` defaults already matched.) |
| REQ-6 (ctor/fit surface max_no_improvement/reassignment_ratio/init_size/compute_labels/verbose/sample_weight + n_clusters=8 + init callable/array + error ABI) | NOT-STARTED | open prereq blocker #1048. sklearn `__init__` (`_kmeans.py:1890-1920`) + `fit(...sample_weight)` + `_check_params_vs_input` (`InvalidParameterError`/`ValueError`, `:875-908,1922-1951`) + `n_features_in_`/`feature_names_in_`. ferrolearn `MiniBatchKMeans<F>` (`fn new`+builders) has `n_clusters/batch_size/max_iter/tol/n_init/random_state/init` only; `fn fit` returns `FerroError::InvalidParameter`/`InsufficientSamples` — missing `max_no_improvement`/`reassignment_ratio`/`init_size`/`compute_labels`/`verbose`/`sample_weight`, the `init` callable/array paths, `n_clusters` default, introspection attrs; different error ABI. |
| REQ-7 (`random_state` numpy-RNG parity) | NOT-STARTED | open prereq blocker #1049. sklearn `check_random_state` + numpy RNG (`random_state.randint` minibatch indices `_kmeans.py:2154`, init subsample `:2099`); ferrolearn `StdRng::seed_from_u64(base_seed+run*1_000_003)` (`fn fit`). Different RNG → exact centers/labels/inertia/n_iter cannot match. Depends on a ferray `random` analog (R-SUBSTRATE-5); blocks REQ-9 value parity. |
| REQ-8 (algorithm STRUCTURE: init-trials-on-subsample + EWA convergence + early stop + n_steps_/n_iter_ formula) | NOT-STARTED | open prereq blocker #1050. sklearn: best-of-`n_init` k-means++ init on an `init_size=3*batch_size` subsample (`_kmeans.py:1929,2099-2132`), then ONE optimization over `n_steps=(max_iter*n_samples)//batch_size` (`:2148`) with EWA-inertia + `max_no_improvement` early stop (`_mini_batch_convergence` `:1964-2006+`), exposing `n_steps_=i+1` (`:2186`) + `n_iter_=ceil(n_steps*batch_size/n_samples)` (`:2187`). ferrolearn `fn fit` runs `n_init` FULL runs (`max_iter` steps, `max_shift<tol`), keeps lowest-inertia, exposes only `n_iter_` (raw step count), no `init_size`/EWA/early-stop/`n_steps_`. |
| REQ-9 (centers/inertia/label-integers/n_iter VALUE parity) | NOT-STARTED | open prereq blocker #1051. sklearn `blobs` → `cluster_centers_=[[-0.0065,0.0613],[10.0041,10.0612],[-0.0043,9.9957]]`, `inertia_=0.093797`, `n_iter_=20`, `n_steps_=20` (Probe B/AC-8). ferrolearn diverges via numpy-RNG (REQ-7), algorithm structure + convergence (REQ-8), `n_init` default (REQ-5), and missing low-count reassignment (REQ-12). Gated on REQ-7/REQ-8. |
| REQ-10 (`partial_fit` online incremental update) | NOT-STARTED | open prereq blocker #1052. sklearn `MiniBatchKMeans.partial_fit(X)` (`_kmeans.py:2202+`) updates centers on a single mini-batch, accumulating `_counts`/`n_steps_` across calls (Probe D). ferrolearn `MiniBatchKMeans<F>` has only `fit`/`fn fit_predict` — no `partial_fit`. Absent end-to-end. |
| REQ-11 (thin binding surface + labels_ dtype) | NOT-STARTED | open prereq blocker #1053. PyO3 `RsMiniBatchKMeans` (`extras.rs`) exposes ONLY `fit`/`predict`/`labels_` — no `transform`/`cluster_centers_`/`inertia_`/`n_iter_` getter; `new` signature `(n_clusters=8, max_iter=100, random_state=None)` omits `n_init`/`batch_size`/`tol` (uses library `n_init=3` — mirrors REQ-5 at the binding). Marshals `labels_`/`predict` to `int64` (`ndarray1_usize_to_numpy`), not sklearn `int32` (Probe B). Library default fix is REQ-5; this row is the marshalling-layer surface (R-DEFER-7 last layer). |
| REQ-12 (low-count cluster reassignment) | NOT-STARTED | open prereq blocker #1054. sklearn `_mini_batch_step` (`_kmeans.py:1652-1682`) reassigns centers with `weight_sums < reassignment_ratio * weight_sums.max()` to random observations; ferrolearn `fn update_centers_mini_batch` performs no reassignment. Contributes to REQ-9 value divergence. |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1055. `mini_batch_kmeans.rs` imports `ndarray::{Array1,Array2}`, `num_traits::Float`, `rand::{rngs::StdRng,Rng,SeedableRng}`, `rayon::prelude::*`; not migrated to `ferray-core`/`ferray::linalg`/`ferray::random` (R-SUBSTRATE-1/2). RNG migration entangled with REQ-7. |

## Architecture

`mini_batch_kmeans.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`MiniBatchKMeans<F>` (`n_clusters: usize`, `batch_size: usize`, `max_iter: usize`, `tol: F`,
`n_init: usize`, `random_state: Option<u64>`, `init: MiniBatchKMeansInit` — all `pub`) →
`Fit<Array2<F>, ()>` → `FittedMiniBatchKMeans<F>` (private `cluster_centers_: Array2<F>`,
`labels_: Array1<usize>`, `inertia_: F`, `n_iter_: usize` with `#[must_use]` accessors).
Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2). `FittedMiniBatchKMeans` implements
`Predict<Array2<F>>` (nearest center) and `Transform<Array2<F>>` (euclidean distance to
centers); a `fn fit_predict` convenience mirrors `ClusterMixin.fit_predict`. The full-dataset
assignment + transform parallelize with Rayon (`into_par_iter`). Re-exported `pub use
mini_batch_kmeans::{FittedMiniBatchKMeans, MiniBatchKMeans, MiniBatchKMeansInit}`
(`ferrolearn-cluster/src/lib.rs`) and bound to CPython via `RsMiniBatchKMeans`
(`ferrolearn-python/src/extras.rs`).

**Fit path (`fn fit`).** Validates `n_clusters >= 1`, `batch_size >= 1`, `n_init >= 1`,
`n_samples >= 1`, `n_samples >= n_clusters` (`FerroError`, REQ-6). For each of `n_init`
FULL runs (seed `base_seed.wrapping_add(run * 1_000_003)`, REQ-7):
1. **Seed** — `fn kmeans_plus_plus_mb` (greedy k-means++, `n_trials = 2 + floor(ln k)`
   candidates per step) or `fn random_init_mb` (Fisher-Yates). sklearn instead chooses the
   best of `n_init` init TRIALS evaluated on an `init_size`-sample subsample
   (`_kmeans.py:2099-2132`, `init_size = 3*batch_size` `:1929`) — different `n_init`
   semantics (REQ-8) — and supports callable/array init (REQ-6).
2. **Mini-batch loop** (`max_iter` steps): sample a batch (`fn sample_batch_indices` —
   without replacement, or all samples shuffled when `batch_size >= n_samples`), assign
   (`fn assign_batch`), update centers in place with per-center learning rate
   `1/center_counts[label]` (`fn update_centers_mini_batch`), converge if `max_shift < tol`
   (MAX euclidean shift). sklearn instead runs ONE optimization over
   `n_steps = (max_iter*n_samples)//batch_size` mini-batches with numpy `randint`-sampled
   batches (`:2154`), EWA-inertia convergence + `max_no_improvement` early stop
   (`_mini_batch_convergence` `:1964-2006+`), and low-count cluster reassignment inside
   `_mini_batch_step` (`:1652-1682`, REQ-12) — none of which ferrolearn has (REQ-8).
3. **Final E-step + candidate** — `(labels, inertia) = assign_clusters_mb(x, &centers)`
   against the FINAL centers, then store `cluster_centers_ = centers`, `labels_ = labels`,
   `inertia_ = inertia`, `n_iter_ = n_iter`. Because `labels_`/`inertia_` are assigned to the
   SAME stored `cluster_centers_`, `fit(X).predict(X) == labels_` holds (REQ-2, no fix needed
   — contrast KMeans #1037). sklearn's post-loop `compute_labels` E-step (`:2189-2195`) does
   the equivalent.
Keep the run with the lowest `inertia_` (best-of-`n_init`). ferrolearn does NOT compute
`n_steps_` and computes `n_iter_` as the raw mini-batch step count, not sklearn's
`ceil(n_steps*batch_size/n_samples)` (REQ-8).

**Predict path (`fn predict`).** Validates feature count, then `fn assign_clusters_mb` →
argmin-squared-euclidean per sample → `Array1<usize>`, mirroring `_BaseKMeans.predict`
(`:1072`, REQ-2). `transform(X).argmin(1) == predict(X)` and `fit(X).predict(X) == labels_`
both hold.

**Transform path (`fn transform`).** Validates feature count, then
`sqrt(squared_euclidean_mb(row, center))` per `(sample, center)` → `(n_samples, n_clusters)`,
mirroring `_BaseKMeans.transform` (`:1130`, REQ-3 — contract matches, values track REQ-9).

**Binding (`RsMiniBatchKMeans`).** Mirrors `MiniBatchKMeans::new(n_clusters).with_max_iter`,
calls `fit`/`predict`, exposes ONLY the `labels_` getter, mapping `FerroError`→`PyValueError`
and not-fitted→`PyRuntimeError`. The signature `(n_clusters=8, max_iter=100,
random_state=None)` omits `n_init`/`batch_size`/`tol` (REQ-11), and there is no
`transform`/`cluster_centers_`/`inertia_`/`n_iter_` exposure (REQ-11). Labels marshal to
`int64` (REQ-11).

**Invariants held vs sklearn:** clustering PARTITION (co-membership on separable data,
Probe B / AC-1); `predict` nearest-center rule (`transform.argmin(1)==predict`, AC-2);
`predict(X) == labels_` consistency (AC-2, final-E-step in `fn fit`); `transform` shape
`(n,k)` + euclidean metric + column-to-center ordering (AC-3);
`cluster_centers_.shape == (n_clusters, n_features)`; `labels_.len() == n_samples`; labels
in `[0, n_clusters)`; `inertia_ >= 0`; predict/transform shape-mismatch error; deterministic
for a fixed `random_state` (`test_reproducibility_with_seed`); f32 + f64 support
(`test_f32_support`); `batch_size`/`max_iter`/`tol` defaults match (Probe A); PyO3
round-trip of `fit`/`predict`/`labels_` (AC-4).

**Invariants NOT held vs sklearn:** `cluster_centers_`/`inertia_`/`labels_`-integers/
`n_iter_` VALUES (REQ-9); the `n_init` default = 1 (REQ-5); the algorithm structure +
convergence + `n_steps_`/`n_iter_` formula (REQ-8); numpy-RNG init parity (REQ-7);
`partial_fit` (REQ-10); low-count cluster reassignment (REQ-12); the
`max_no_improvement`/`reassignment_ratio`/`init_size`/`compute_labels`/`verbose`/
`sample_weight` ctor/fit surface + `n_clusters`=8 default + `init` callable/array + error ABI
+ introspection attrs (REQ-6); the binding surface (`transform`/`cluster_centers_`/
`inertia_`/`n_iter_`/ctor-params absent) + `int32` label dtype (REQ-11); the ferray substrate
(REQ-13).

**Consumer wiring.** Non-test consumers: (1) the PyO3 binding `RsMiniBatchKMeans`
(`ferrolearn-python/src/extras.rs`, registered in `lib.rs`, wrapped/exported as
`ferrolearn.MiniBatchKMeans`) — a REAL CPython consumer of `fit`/`predict`/`labels_`
(NOT `transform`/`cluster_centers_`/`inertia_`/`n_iter_`); (2) the crate re-export
`pub use mini_batch_kmeans::{FittedMiniBatchKMeans, MiniBatchKMeans, MiniBatchKMeansInit}`
(`ferrolearn-cluster/src/lib.rs`).

## Verification

Library crate:
```
cargo test -p ferrolearn-cluster --lib mini_batch     # in-tree MiniBatchKMeans #[test]s (21 passed)
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s (`test_well_separated_blobs`, `test_cluster_centers_shape`,
`test_labels_length`, `test_inertia_non_negative`, `test_n_iter_positive`,
`test_predict_on_new_data`, `test_transform_shape`, `test_transform_distances_structure`,
`test_reproducibility_with_seed`, `test_random_init`, `test_single_cluster`,
`test_zero_clusters_error`, `test_zero_batch_size_error`, `test_too_few_samples_error`,
`test_empty_data_error`, `test_predict_shape_mismatch_error`,
`test_transform_shape_mismatch_error`, `test_f32_support`, `test_large_batch_size`,
`test_n_init_zero_error`, `test_identical_points`) pin ferrolearn's current behavior — label
co-membership, shapes, ranges, error edges, reproducibility, f32 support, large-batch
handling. **None compares `cluster_centers_` values, `labels_` integers, `inertia_`,
`n_iter_`/`n_steps_`, the `n_init` default, the convergence/structure, or `partial_fit`
against the live sklearn `MiniBatchKMeans` oracle**, so they stay green despite the
divergences. In particular the partition tests assert `labels[0]==labels[1]` co-membership
and never the absolute integers (masking REQ-9).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic should pin
(R-CHAR-3 expected values). **Pin REQ-5's `n_init` default (AC-5) FIRST** — `n_init` is a
`pub` field, so the pin is `MiniBatchKMeans::<f64>::new(3).n_init == 1`, deterministic and
RNG-free, a one-line minimal fix in `fn new` (plus correcting the misleading `:102-106`
comment, R-HONEST-4). REQ-9 exact VALUES cannot pin until the numpy-RNG analog (REQ-7) and
the algorithm structure (REQ-8) land:
```
# REQ-5 n_init default (observable, isolable, no RNG)
python3 -c "from sklearn.cluster import MiniBatchKMeans; import numpy as np; \
X=np.zeros((10,2)); print(MiniBatchKMeans(n_clusters=3,random_state=0).fit(X)._n_init)"   # 1 (ferrolearn fn new: 3)
# REQ-2 consistency (no RNG parity needed — must hold for any seed)
python3 -c "import numpy as np; from sklearn.cluster import MiniBatchKMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=MiniBatchKMeans(n_clusters=3, n_init=3, random_state=42, batch_size=9).fit(X); \
print(np.array_equal(m.predict(X), m.labels_))"   # True (already holds in ferrolearn)
# REQ-1 partition + REQ-9 centers/inertia/n_iter/n_steps + REQ-8 n_steps_
python3 -c "import numpy as np; from sklearn.cluster import MiniBatchKMeans; \
X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
m=MiniBatchKMeans(n_clusters=3, n_init=3, random_state=42, batch_size=9).fit(X); \
print(m.labels_.tolist(), np.round(m.cluster_centers_,4).tolist(), round(m.inertia_,6), m.n_iter_, m.n_steps_)"
# [0,0,0,1,1,1,2,2,2] [[-0.0065,0.0613],[10.0041,10.0612],[-0.0043,9.9957]] 0.093797 20 20
# REQ-10 partial_fit
python3 -c "from sklearn.cluster import MiniBatchKMeans; print(hasattr(MiniBatchKMeans(n_clusters=2), 'partial_fit'))"   # True
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-cluster/tests/divergence_mini_batch_kmeans.rs`, asserting the live-sklearn
expected values above and FAILING against current `mini_batch_kmeans.rs`. REQ-5 `n_init` is
the green-after-single-fix target; REQ-9 exact center/inertia/`n_iter_` values can only be
characterized as a partition pin until the numpy-RNG analog (REQ-7) and algorithm structure
(REQ-8) land.

ferrolearn-python (REQ-4 binding parity; REQ-11 binding surface/dtype):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_mini_batch_kmeans.py -q
```
asserting `import ferrolearn; ferrolearn.MiniBatchKMeans(...).fit(blobs)` exposes
`labels_`/`predict` with the right shape (REQ-4 green), and pinning the MISSING
`cluster_centers_`/`inertia_`/`n_iter_`/`transform` attributes + the `int32` label dtype
(REQ-11, red until fixed).

## Blockers (to open)

REQ-1 (partition), REQ-2 (predict contract + consistency), REQ-3 (transform contract), and
REQ-4 (PyO3 binding marshalling of fit/predict/labels_) SHIP — no blocker. The remaining
NINE NOT-STARTED REQs, the real filed issues #1047–#1055:

- **REQ-5 (cleanest minimal fix)** — set `fn new` default `n_init = 1` to mirror sklearn's
  `n_init="auto"` → `1` for k-means++ (`_kmeans.py:886-888`, docstring `:1778-1780`) AND
  correct the misleading `:102-106` comment (R-HONEST-4). One-line change; `n_init` is a
  `pub` field so the pin `MiniBatchKMeans::<f64>::new(3).n_init == 1` is deterministic +
  RNG-free.
- **REQ-6** — add `max_no_improvement`/`reassignment_ratio`/`init_size`/`compute_labels`/
  `verbose`/`sample_weight` ctor/fit params + `init` callable/array + `n_clusters`=8 default
  + `n_features_in_`/`feature_names_in_` + `InvalidParameterError`/`ValueError` error ABI
  (`:1890-1920,875-908,1922-1951`).
- **REQ-7** — numpy-RNG parity (`check_random_state` + `randint`-sampled minibatch indices
  `:2154` and init subsample `:2099`); depends on a ferray `random` analog
  (R-SUBSTRATE-5); blocks exact REQ-9 value parity.
- **REQ-8** — restructure to sklearn's single-optimization shape: best-of-`n_init` init on
  an `init_size=3*batch_size` subsample (`:1929,2099-2132`), `n_steps =
  (max_iter*n_samples)//batch_size` (`:2148`), EWA-inertia convergence + `max_no_improvement`
  early stopping (`_mini_batch_convergence` `:1964-2006+`), and expose `n_steps_ = i+1`
  (`:2186`) + `n_iter_ = ceil(n_steps*batch_size/n_samples)` (`:2187`).
- **REQ-9** — `cluster_centers_`/`inertia_`/`labels_`-integers/`n_iter_`/`n_steps_` value
  parity — gated on REQ-7/REQ-8/REQ-12.
- **REQ-10** — add `partial_fit(X)` (online incremental update, `:2202+`) to
  `MiniBatchKMeans<F>` and the binding.
- **REQ-11** — ferrolearn-python: expose `transform`/`cluster_centers_`/`inertia_`/`n_iter_`
  getters on `RsMiniBatchKMeans`, add `n_init`/`batch_size`/`tol` to the signature (defaulting
  `n_init=1` after REQ-5), and marshal `labels_`/`predict` as `int32`, not `int64`
  (`ndarray1_usize_to_numpy`).
- **REQ-12** — low-count cluster reassignment inside the mini-batch update (`_mini_batch_step`
  `:1652-1682`): reassign centers with `weight_sums < reassignment_ratio * weight_sums.max()`
  to random observations.
- **REQ-13** — migrate `mini_batch_kmeans.rs` off `ndarray`/`num-traits`/`rand`/`rayon` to
  `ferray-core`/`ferray::linalg`/`ferray::random` (R-SUBSTRATE) — RNG migration entangled
  with REQ-7.
