# Isomap (sklearn.manifold.Isomap)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 71fcc81f
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/manifold/_isomap.py  # class Isomap (:130-); __init__(n_neighbors=5, radius=None, n_components=2, eigen_solver="auto", tol=0, max_iter=None, path_method="auto", neighbors_algorithm="auto", n_jobs=None, metric="minkowski", p=2, metric_params=None) (:182-209); _fit_transform (:211-310): nbrs_ = NearestNeighbors(...) (:219-228), kernel_pca_ = KernelPCA(n_components, kernel="precomputed", eigen_solver, tol, max_iter, ...) (:233-240), nbg = kneighbors_graph(nbrs_, n_neighbors, mode="distance") (:242-251) [else radius_neighbors_graph (:253-261)], connected_components fix (:267-297), self.dist_matrix_ = shortest_path(nbg, method=path_method, directed=False) (:299), G = self.dist_matrix_**2; G *= -0.5 (:306-307), self.embedding_ = self.kernel_pca_.fit_transform(G) (:309); reconstruction_error() (:312-335) = sqrt(sum(G_center**2) - sum(evals**2))/n; transform out-of-sample (:386-); fit/fit_transform (:341-384).
  - sklearn/decomposition/_kernel_pca.py  # KernelPCA._fit_transform (:328-): eigh(K) (:350) -> svd_flip(u=eigenvectors_, v=None) DETERMINISTIC sign (:373) -> argsort desc (:376-378) -> drop nonpositive eigenvalues (:382-383); fit_transform: X_transformed = self.eigenvectors_ * np.sqrt(self.eigenvalues_) (:446/477).
  - sklearn/utils/extmath.py  # svd_flip (:848-906), u_based_decision=True branch (:888-896): per eigenvector column, argmax(|u.T|) row, signs = sign(that entry), u *= signs -> max-abs entry of each column made POSITIVE.
ferrolearn-module: ferrolearn-decomp/src/isomap.rs
parity-ops: Isomap
crosslink-issue: 1467
-->

## Summary

`ferrolearn-decomp/src/isomap.rs` mirrors scikit-learn's `Isomap`
(`sklearn/manifold/_isomap.py`, `class Isomap` `:130`): build a k-nearest-neighbor
distance graph, compute all-pairs geodesic (shortest-path) distances, square them,
and apply classical MDS (double-center the `−0.5·D²` kernel, eigendecompose, take
the top `n_components` eigenvectors scaled by `√eigenvalue`) to obtain the
embedding (`fn fit`, `isomap.rs:258`). The exposed surface is the unfitted
`Isomap { n_components, n_neighbors }` (`isomap.rs:54`, default `n_neighbors=5`)
and the fitted `FittedIsomap { embedding_, x_train_, eigenvalues_, eigenvectors_,
geo_sq_row_means_, geo_sq_grand_mean_ }` (`isomap.rs:103`), re-exported at the
crate root (`pub use isomap::{FittedIsomap, Isomap}`, `lib.rs:89`).

**ISOMAP EMBEDDING VALUE PARITY (SHIPPED — `svd_flip` sign convention applied,
RESOLVED #1468).** The geodesic-distance pipeline matches sklearn (REQ-2, SHIPPED):
the kNN graph (`build_knn_graph`, `isomap.rs:159`) + Dijkstra all-pairs shortest
paths (`all_pairs_shortest_paths`, `isomap.rs:228`) reproduce sklearn's
`kneighbors_graph(mode="distance")` + `shortest_path` `dist_matrix_`
(`_isomap.py:242-299`) element-wise on the live 1.5.2 oracle. The MDS kernel
construction also matches: `geo_sq = geodesic²` (`isomap.rs:323`), double-center
`B = −0.5·(D² − rowmean − colmean + grandmean)` (`classical_mds`, `mds.rs:246-251`
= sklearn's `KernelCenterer` of `G = −0.5·D²`, `_isomap.py:306-307`), then
`embedding[i,k] = v_ik · √λ_k` (`mds.rs:264-273` = KernelPCA's
`X = eigenvectors_ · √eigenvalues_`, `_kernel_pca.py:446`). On top of this, `fit`
now applies KernelPCA's `svd_flip` sign convention: per embedding column, the row
of maximum ABSOLUTE value is found and the column is negated so that entry is
POSITIVE (`isomap.rs` `fn fit` per-column max-abs flip = `svd_flip(u=eigenvectors_,
v=None)`, `_kernel_pca.py:373`; `u_based_decision=True`, `extmath.py:888-896`) — a
DETERMINISTIC sign. With the flip applied, the embedding matches sklearn EXACTLY
element-wise (NO sign alignment), verified across n_neighbors {3,4,5},
n_components {1,2,3}, and 3 fixtures (10×3, 12×3, 15×4) in
`tests/divergence_isomap.rs`. REQ-1 (embedding value parity) is therefore SHIPPED;
the FIXABLE sign divergence formerly tracked as `#1468` is RESOLVED.

At baseline `71fcc81f` (with the `svd_flip` fix #1468 landed): the embedding
VALUE parity (REQ-1), the geodesic distance matrix (REQ-2), the structural
shape/determinism/disconnected-graph error (REQ-3), and the error/parameter
contracts (REQ-4) are SHIPPED; the `transform` out-of-sample Nystroem extension
(REQ-5, `#1469`), `radius` mode (REQ-6, `#1470`), `path_method` / `eigen_solver` /
`tol` / `max_iter` (REQ-7, `#1471`), `metric`/`p` (REQ-8, `#1472`),
`reconstruction_error()` + the `kernel_pca_`/`nbrs_`/`dist_matrix_` attr surface
+ degenerate-eigenvalue carve-out (REQ-9, `#1473`), the PyO3 binding (REQ-10,
`#1474`), and the ferray substrate (REQ-11, `#1475`) are NOT-STARTED —
**4 SHIPPED / 7 NOT-STARTED**.

`Isomap` / `FittedIsomap` are existing pub APIs whose non-test consumer is the
crate re-export (`lib.rs:89`, boundary public API, grandfathered S5/R-DEFER-1).
There is **no PyO3 binding** (`grep -rln Isomap ferrolearn-python/` is empty,
REQ-10). A `transform` IS implemented (`impl Transform for FittedIsomap`,
`isomap.rs:388`) but it is a hand-rolled Nystroem extension whose geodesic
approximation uses raw test→train Euclidean distances, NOT sklearn's
`kneighbors`-then-`dist_matrix_` graph linking (`_isomap.py:409-471`) — REQ-5,
NOT-STARTED.

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# REQ-1 (SHIPPED; svd_flip applied, RESOLVED #1468) — the embedding VALUE parity target.
# X is the hard-coded 10x3 fixture (R-CHAR-3). ferrolearn's embedding matches this
# EXACTLY element-wise (same kNN graph + geodesic + centered Gram + v*sqrt(lambda)),
# now with the per-column max-abs-positive svd_flip applied (_kernel_pca.py:373).
python3 -c "import numpy as np; from sklearn.manifold import Isomap
X=np.array([[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]])
m=Isomap(n_components=2,n_neighbors=4).fit(X)
print('embedding_:', np.round(m.embedding_,8).tolist())
print('eigenvalues_:', np.round(m.kernel_pca_.eigenvalues_,6).tolist())"
# -> embedding_: [[-1.43997879,-1.13229201], ... ,[-0.61171047,1.17436347]]
# -> eigenvalues_: [13.807432, 5.052949]
#    distinct eigenvalues => unambiguous columns; with svd_flip applied, ferrolearn == sklearn
#    element-wise AND the sign matches (max-abs entry of each column made positive).

# REQ-2 (SHIPPED) — the geodesic distance matrix parity target (kNN + Dijkstra == kneighbors_graph + shortest_path).
python3 -c "import numpy as np; from sklearn.manifold import Isomap
X=np.array([[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]])
m=Isomap(n_components=2,n_neighbors=4).fit(X)
print('dist_matrix_[0]:', np.round(m.dist_matrix_[0],6).tolist())"
# -> dist_matrix_[0]: [0.0, 1.004988, 2.029683, 3.039633, 1.135782, 1.862794, 2.835908, 3.781824, 2.009975, 2.443451]
#    ferrolearn's all_pairs_shortest_paths (isomap.rs:228) row 0 reproduces this element-wise.

# REQ-9 — sklearn ctor defaults + fitted attrs.
python3 -c "from sklearn.manifold import Isomap as I
m=I(); print('defaults n_neighbors',m.n_neighbors,'radius',m.radius,'n_components',m.n_components,
'eigen_solver',m.eigen_solver,'tol',m.tol,'max_iter',m.max_iter,'path_method',m.path_method,
'neighbors_algorithm',m.neighbors_algorithm,'metric',m.metric,'p',m.p)"
# -> defaults n_neighbors 5 radius None n_components 2 eigen_solver auto tol 0 max_iter None
#    path_method auto neighbors_algorithm auto metric minkowski p 2
```

## Requirements

- REQ-1: **Isomap embedding VALUE parity (classical MDS on geodesic distances +
  svd_flip sign) (SHIPPED; RESOLVED #1468).** sklearn computes
  `G = −0.5·dist_matrix_²` (`_isomap.py:306-307`), KernelPCA double-centers it,
  eigendecomposes, applies `svd_flip(u=eigenvectors_, v=None)` for a DETERMINISTIC
  sign (`_kernel_pca.py:373`; per column the max-abs entry is made positive,
  `extmath.py:888-896`), sorts eigenvalues descending, drops nonpositive ones, and
  returns `embedding_ = eigenvectors_ · √eigenvalues_` (`_kernel_pca.py:446`). The
  parity target on the 10×3 fixture is the Probe REQ-1 `embedding_`. ferrolearn's
  `classical_mds` (`mds.rs:221`) reproduces every step — double-center
  (`mds.rs:246-251`), `eigh_faer` (`mds.rs:254`), sort descending (`mds.rs:256-262`),
  `embedding[i,k] = v_ik · √λ_k` (`mds.rs:264-273`) — and `fn fit` (`isomap.rs:258`)
  now applies the `svd_flip` per-column max-abs-positive flip after `classical_mds`.
  So `ferrolearn.embedding` matches `sklearn.embedding_` EXACTLY element-wise (NO
  sign alignment). The eigenvalues are distinct (`[13.807432, 5.052949]`), so the
  flip is unambiguous; element-wise parity is verified across n_neighbors {3,4,5},
  n_components {1,2,3}, and 3 fixtures (10×3, 12×3, 15×4) in
  `tests/divergence_isomap.rs` (`divergence_embedding_raw_sign` + the
  `green_exact_parity_*` probes), all GREEN.

- REQ-2: **Geodesic distance matrix parity (kNN graph + Dijkstra == sklearn
  `dist_matrix_`) (SHIPPED).** `build_knn_graph` (`isomap.rs:159`) sorts each row's
  Euclidean distances, takes the `k` nearest (`isomap.rs:163-173`), and SYMMETRIZES
  the graph (if `i→j` then `j→i`, keeping the shorter weight, `isomap.rs:175-195`) —
  mirroring sklearn's `kneighbors_graph(nbrs_, n_neighbors, mode="distance")`
  followed by `shortest_path(..., directed=False)` (`_isomap.py:242-251`, `:299`).
  `all_pairs_shortest_paths` (`isomap.rs:228`) runs `dijkstra` (`isomap.rs:199`)
  from every source — sklearn's `shortest_path` with `path_method="auto"` picks
  Dijkstra (or Floyd-Warshall) but converges to the same geodesic. Verified
  element-wise against the live oracle `dist_matrix_[0]` (Probe REQ-2). This
  geodesic matrix is the reused input to REQ-1's `classical_mds` (`isomap.rs:339`).

- REQ-3: **Structural embedding shape + determinism + disconnected-graph error
  (SHIPPED scoped).** `fn fit` (`isomap.rs:258`) returns `FittedIsomap` whose
  `embedding()` (`isomap.rs:123`) is `Array2<f64>` of shape
  `(n_samples, n_components)` (`Array2::zeros((n, n_comp))`, `mds.rs:266`), finite,
  and DETERMINISTIC (the `build_knn_graph` + `dijkstra` + `eigh_faer` path uses no
  RNG, unlike sklearn's `eigen_solver='arpack'` path). When the kNN graph is
  disconnected (some geodesic is `INFINITY`), `fit` returns
  `NumericalInstability { message: "kNN graph is disconnected ..." }`
  (`isomap.rs:309-320`) — sklearn instead `_fix_connected_components` + warns
  (`_isomap.py:267-297`), a behavioral DIVERGENCE flagged under REQ-9. Mirrors the
  `(n_samples, n_components)` output shape of `_fit_transform` (`_isomap.py:309-310`).

- REQ-4: **Error / parameter contracts (SHIPPED scoped).** `fn fit` (`isomap.rs:258`)
  returns `InvalidParameter { name: "n_components" }` for `n_components == 0`
  (`isomap.rs:261-266`) and for `n_components > n_samples` (`isomap.rs:289-297`),
  `InvalidParameter { name: "n_neighbors" }` for `n_neighbors == 0`
  (`isomap.rs:267-272`) and for `n_neighbors >= n_samples` (`isomap.rs:280-288`),
  and `InsufficientSamples { required: 2 }` for `< 2` samples (`isomap.rs:273-279`).
  **FLAG (candidate DIVs):** sklearn validates `n_components >= 1` and
  `n_neighbors >= 1` via `_parameter_constraints` (`_isomap.py:162-180`) and raises
  `InvalidParameterError`/`ValueError` — not the `FerroError` ABI; sklearn does not
  pre-reject `n_neighbors >= n_samples` (it would surface later in the neighbor
  search), so ferrolearn's guard is stricter.

- REQ-5: **`transform` out-of-sample (Nystroem extension) (NOT-STARTED;
  `#1469`).** sklearn's `transform` (`_isomap.py:386-471`) finds each query's
  `n_neighbors` nearest TRAINING points via `nbrs_.kneighbors` (`:411`), forms the
  geodesic kernel `G_X[i,j] = (dist to neighbor + dist_matrix_[neighbor, j])²`
  through the trained graph (`:419-460`), and projects it onto the embedding via the
  KernelPCA Nystroem formula. ferrolearn HAS a `Transform` impl
  (`impl Transform<Array2<f64>> for FittedIsomap`, `isomap.rs:388`) using the stored
  `eigenvalues_`/`eigenvectors_`/`geo_sq_row_means_`/`geo_sq_grand_mean_`
  (`isomap.rs:454-468`), BUT it approximates each test→train geodesic distance with
  the raw test→train EUCLIDEAN distance (`isomap.rs:436`, `:451`) instead of linking
  through the training geodesic graph as sklearn does (`_isomap.py:419-460`)
  (REQ-1's `svd_flip` sign is now applied in `fit`, RESOLVED #1468). Its own test concedes the
  output only "should be correlated" with retraining, not equal
  (`test_isomap_transform_recovers_training`, `isomap.rs:557-568`). No live-oracle
  parity is established → NOT-STARTED.

- REQ-6: **`radius` mode + `radius_neighbors_graph` (NOT-STARTED; `#1470`).**
  sklearn supports `radius`-based neighborhoods: when `radius` is set (and
  `n_neighbors=None`) it builds `radius_neighbors_graph(nbrs_, radius, mode="distance")`
  (`_isomap.py:253-261`) and raises if both `n_neighbors` and `radius` are given
  (`:212-217`). ferrolearn's `Isomap` (`isomap.rs:54`) has no `radius` field — only
  `n_neighbors`.

- REQ-7: **`path_method` (FW/Floyd-Warshall vs D/Dijkstra) + `eigen_solver` / `tol`
  / `max_iter` (NOT-STARTED; `#1471`).** sklearn's `shortest_path(method=path_method,
  directed=False)` selects `"auto"`/`"FW"`/`"D"` (`_isomap.py:299`, default `"auto"`,
  ctor `:191`), and `KernelPCA(eigen_solver=..., tol=..., max_iter=...)`
  (`_isomap.py:236-238`) selects the dense `eigh` vs `arpack`/`randomized` solver.
  ferrolearn hard-codes Dijkstra (`all_pairs_shortest_paths`, `isomap.rs:228`) and
  the dense `eigh_faer` (`mds.rs:254`) — no `path_method` / `eigen_solver` / `tol` /
  `max_iter` fields on `Isomap` (`isomap.rs:54`).

- REQ-8: **`metric` / `p` (non-Euclidean distances) (NOT-STARTED; `#1472`).**
  sklearn's `Isomap(metric="minkowski", p=2, metric_params=None)` (`_isomap.py:194-196`)
  flows `metric`/`p` into `NearestNeighbors` and `kneighbors_graph`
  (`:223-224`, `:246-247`). ferrolearn hard-codes squared Euclidean distance
  (`pairwise_sq_distances`, `isomap.rs:300` → `.sqrt()` in `build_knn_graph`,
  `isomap.rs:167`); there is no `metric`/`p` field on `Isomap` (`isomap.rs:54`).

- REQ-9: **`reconstruction_error()` + `kernel_pca_`/`nbrs_`/`dist_matrix_` fitted-attr
  surface + `neighbors_algorithm` + degenerate-eigenvalue subspace (NOT-STARTED,
  CARVE-OUT; `#1473`).** sklearn's `Isomap` exposes `reconstruction_error()`
  (`_isomap.py:312-335`, `√(Σ G_center² − Σ evals²)/n`) and the fitted attrs
  `embedding_`, `kernel_pca_`, `nbrs_`, `dist_matrix_` (`_isomap.py:228`, `:233`,
  `:299`, `:309`), plus the `neighbors_algorithm="auto"` ctor field (`:192`) and the
  `connected_components` graph-completion fallback for disconnected graphs
  (`:267-297`). `FittedIsomap` exposes only `embedding()` (`isomap.rs:123`) — no
  `reconstruction_error`, no public `dist_matrix_` / `nbrs_` / `kernel_pca_`
  accessors (the eigenvalues/vectors are stored privately for `transform`,
  `isomap.rs:110-117`), no `neighbors_algorithm`, and it ERRORS on disconnected
  graphs rather than completing them (`isomap.rs:309-320`). **CARVE-OUT:** when
  eigenvalues are degenerate (repeated), the corresponding eigenvector subspace is
  rotation-arbitrary even after `svd_flip`, so element-wise parity is structurally
  impossible there; the fixture's distinct eigenvalues avoid this.

- REQ-10: **PyO3 binding (NOT-STARTED; `#1474`).** `import ferrolearn` exposing a
  registered `Isomap` marshalling `fit`/`transform` and `embedding_` — the boundary
  CPython consumer. Absent (`grep -rln Isomap ferrolearn-python/` is empty).

- REQ-11: **ferray substrate (NOT-STARTED; `#1475`).** `isomap.rs` computes on
  `ndarray::Array2<f64>` (`isomap.rs:41`) and eigendecomposes via faer
  (`crate::mds::eigh_faer`, `mds.rs:203` → `faer`), not `ferray-core` arrays /
  `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). Fixture `X` = the 10×3 point set
`[[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],
[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]]`, with
`n_components=2, n_neighbors=4`.

- AC-1 (REQ-1, SHIPPED; RESOLVED #1468): `Isomap::new(2)
  .with_n_neighbors(4).fit(&X).unwrap().embedding()` matches the Probe REQ-1
  `embedding_ = [[-1.43997879,-1.13229201],...,[-0.61171047,1.17436347]]`
  element-wise to < 1e-6 — EXACTLY, with NO sign alignment. `fit` applies the
  per-column max-abs-positive `svd_flip` (`_kernel_pca.py:373`) after
  `classical_mds`, so the signs are deterministic and match sklearn (distinct
  eigenvalues `[13.807432, 5.052949]`). Pinned GREEN by `divergence_embedding_raw_sign`
  + the `green_exact_parity_*` probes in `tests/divergence_isomap.rs`.

- AC-2 (REQ-2, SHIPPED): `Isomap::new(2).with_n_neighbors(4).fit(&X)` produces a
  geodesic distance matrix whose row 0 equals the Probe REQ-2
  `dist_matrix_[0] = [0,1.004988,2.029683,3.039633,1.135782,1.862794,2.835908,
  3.781824,2.009975,2.443451]` to < 1e-6 (kNN graph + Dijkstra ==
  `kneighbors_graph` + `shortest_path`).

- AC-3 (REQ-3, SHIPPED scoped): `Isomap::new(2).with_n_neighbors(3).fit(&grid)
  .unwrap().embedding()` has shape `(9, 2)`, is finite, and is identical across
  runs; a disconnected graph yields `Err(NumericalInstability)`. Pinned by
  `test_isomap_basic_shape` `(9,2)`, `test_isomap_1d` (ncols 1),
  `test_isomap_preserves_ordering` (1D line embedding monotonic).

- AC-4 (REQ-4, SHIPPED scoped): `fit` returns `Err` for `n_components=0`,
  `n_neighbors=0`, `n_neighbors >= n_samples`, `n_components > n_samples`, and
  `n_samples < 2`. Pinned by `test_isomap_invalid_n_components_zero`,
  `test_isomap_invalid_n_neighbors_zero`, `test_isomap_n_neighbors_too_large`,
  `test_isomap_n_components_too_large`, `test_isomap_insufficient_samples`. FLAG:
  sklearn raises `InvalidParameterError`/`ValueError`, not `FerroError`, and does
  not pre-reject `n_neighbors >= n_samples`.

- AC-5 (REQ-5..9, DIVERGES): `Isomap()` defaults `n_neighbors=5, radius=None,
  n_components=2, eigen_solver='auto', tol=0, max_iter=None, path_method='auto',
  neighbors_algorithm='auto', metric='minkowski', p=2` (Probe REQ-9); sklearn
  exposes `transform` (geodesic-graph-linked), `reconstruction_error()`,
  `dist_matrix_`, `kernel_pca_`, `nbrs_`. ferrolearn has a Euclidean-approximation
  `transform` (`isomap.rs:388`), only `embedding()`, no `radius`/`path_method`/
  `eigen_solver`/`metric`/`p` fields, no `reconstruction_error`, and errors on
  disconnected graphs.

- AC-6 (REQ-10/11): no `ferrolearn.Isomap` (no binding, `grep -rln Isomap
  ferrolearn-python/` empty); the module imports `ndarray` + faer
  (`crate::mds::eigh_faer`), not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `Isomap` / `FittedIsomap` are existing pub APIs; the non-test
consumer is the crate re-export (`lib.rs:89`, boundary public API, grandfathered
S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2).
Live oracle = installed sklearn 1.5.2, run from `/tmp`. **ISOMAP EMBEDDING VALUE
PARITY (R-HONEST-3, SHIPPED, RESOLVED #1468):** ferrolearn's geodesic pipeline
(REQ-2) and MDS kernel construction match sklearn, and `fn fit` (`isomap.rs:258`)
now applies KernelPCA's `svd_flip(u=eigenvectors_, v=None)` (`_kernel_pca.py:373`;
max-abs entry of each column made positive, `extmath.py:888-896`) after
`classical_mds`, so the embedding matches sklearn EXACTLY element-wise (NO sign
alignment) — verified across n_neighbors {3,4,5}, n_components {1,2,3}, and 3
fixtures (10×3, 12×3, 15×4) in `tests/divergence_isomap.rs`. The formerly-FIXABLE
sign divergence (#1468) is RESOLVED. REQ-2 is SHIPPED — the kNN+Dijkstra
`dist_matrix_` matches the live oracle element-wise. The least-confident SHIPPED
claim is REQ-3 —
it is STRUCTURAL (shape/determinism + the disconnected-graph error path), and the
disconnected-graph BEHAVIOR diverges from sklearn (ferrolearn errors; sklearn
completes the graph + warns, `_isomap.py:267-297`) — flagged under REQ-9, so REQ-3
is scoped to shape/determinism only. #1467 is this doc's crosslink tracking issue.
Count: **4 SHIPPED (REQ-1,2,3,4) / 7 NOT-STARTED (REQ-5,6,7,8,9,10,11)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (Isomap embedding VALUE parity: classical MDS on geodesic + svd_flip sign) | SHIPPED | RESOLVED #1468 (svd_flip sign convention). sklearn: `G = −0.5·dist_matrix_²` (`_isomap.py:306-307`) → KernelPCA double-centers + eigendecomposes → `svd_flip(u=eigenvectors_, v=None)` for a DETERMINISTIC sign (`_kernel_pca.py:373`; per column, `argmax(|u.T|)` entry made positive, `extmath.py:888-896`) → sort desc → `embedding_ = eigenvectors_·√eigenvalues_` (`_kernel_pca.py:446`). Parity target (Probe REQ-1, R-CHAR-3): `embedding_ = [[-1.43997879,-1.13229201],...,[-0.61171047,1.17436347]]`, `eigenvalues_=[13.807432,5.052949]`. ferrolearn's `classical_mds` (`mds.rs:221`) reproduces double-center (`mds.rs:246-251`), `eigh_faer` (`mds.rs:254`), sort desc (`mds.rs:256-262`), `embedding[i,k]=v_ik·√λ_k` (`mds.rs:264-273`), and `fn fit` (`isomap.rs:258`) then applies the per-column max-abs-positive `svd_flip`. So `ferrolearn.embedding` == `sklearn.embedding_` EXACTLY element-wise (NO sign alignment; distinct eigenvalues ⇒ unambiguous columns). Non-test consumer: crate re-export (`lib.rs:89`). Verification: `cargo test -p ferrolearn-decomp --test divergence_isomap` — `divergence_embedding_raw_sign`, `green_exact_parity_k3_nc2`/`_k5_nc2`/`_nc1`/`_15x4_nc3`/`_12x3_k4`/`_argmax_non_endpoint` (n_neighbors {3,4,5}, n_components {1,2,3}, fixtures 10×3/12×3/15×4) PASS. |
| REQ-2 (geodesic distance matrix: kNN graph + Dijkstra == sklearn `dist_matrix_`) | SHIPPED | `fn build_knn_graph` (`isomap.rs:159`) sorts each row's Euclidean distances (`isomap.rs:165-169`), takes the `k` nearest (`isomap.rs:170-172`), and SYMMETRIZES (`i→j ⇒ j→i`, keeping the shorter weight, `isomap.rs:175-195`) — sklearn `kneighbors_graph(nbrs_, n_neighbors, mode="distance")` + `shortest_path(..., directed=False)` (`_isomap.py:242-251`, `:299`). `fn all_pairs_shortest_paths` (`isomap.rs:228`) runs `fn dijkstra` (`isomap.rs:199`) from every source — sklearn's `shortest_path(method="auto")` (Dijkstra/Floyd-Warshall, same geodesic). Probe REQ-2 (R-CHAR-3): `dist_matrix_[0] = [0,1.004988,2.029683,3.039633,1.135782,1.862794,2.835908,3.781824,2.009975,2.443451]` reproduced element-wise. This geodesic matrix is the reused input to REQ-1's `classical_mds` (`isomap.rs:323`/`339`). Non-test consumer: crate re-export (`lib.rs:89`). Verification: `cargo test -p ferrolearn-decomp isomap` (`test_isomap_preserves_ordering` pins the 1D line geodesic ordering). |
| REQ-3 (structural embedding shape + determinism + disconnected-graph error) | SHIPPED | `fn fit` (`isomap.rs:258`) returns `FittedIsomap` whose `embedding()` (`isomap.rs:123`) is `Array2<f64>` of shape `(n_samples, n_components)` — `Array2::zeros((n, n_comp))` (`mds.rs:266`), finite, DETERMINISTIC (the `build_knn_graph` + `dijkstra` + `eigh_faer` path uses no RNG, unlike sklearn's `eigen_solver='arpack'`). A disconnected kNN graph (some geodesic `INFINITY`) yields `Err(NumericalInstability { message: "kNN graph is disconnected ..." })` (`isomap.rs:309-320`). Mirrors the `(n_samples, n_components)` output of `_fit_transform` (`_isomap.py:309-310`). Non-test consumer: crate re-export (`lib.rs:89`). Verification: `cargo test -p ferrolearn-decomp isomap` → `test_isomap_basic_shape` `(9,2)`, `test_isomap_1d` (ncols 1), `test_isomap_preserves_ordering` PASS. **Scope: SHAPE/finiteness/determinism + the error path; NOT value parity (REQ-1). FLAG: the disconnected-graph BEHAVIOR diverges — sklearn completes the graph + warns (`_isomap.py:267-297`), ferrolearn errors (REQ-9).** |
| REQ-4 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`isomap.rs:258`) returns `Err(InvalidParameter { name: "n_components", reason: "must be at least 1" })` for `n_components==0` (`isomap.rs:261-266`), `Err(InvalidParameter { name: "n_neighbors", reason: "must be at least 1" })` for `n_neighbors==0` (`isomap.rs:267-272`), `Err(InsufficientSamples { required: 2, actual: n, context: "Isomap::fit requires at least 2 samples" })` for `< 2` samples (`isomap.rs:273-279`), `Err(InvalidParameter { name: "n_neighbors", ... })` for `n_neighbors >= n_samples` (`isomap.rs:280-288`), and `Err(InvalidParameter { name: "n_components", ... })` for `n_components > n_samples` (`isomap.rs:289-297`). Non-test consumer: these guards protect every instance reached via the crate re-export (`lib.rs:89`). Verification: `cargo test -p ferrolearn-decomp isomap` (`test_isomap_invalid_n_components_zero`, `_invalid_n_neighbors_zero`, `_n_neighbors_too_large`, `_n_components_too_large`, `_insufficient_samples`) PASS. **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints` (`_isomap.py:162-180`) and raises `InvalidParameterError`/`ValueError`, NOT `FerroError`; sklearn does not pre-reject `n_neighbors >= n_samples`. |
| REQ-5 (`transform` out-of-sample Nystroem extension) | NOT-STARTED | open prereq blocker **#1469**. sklearn `transform` (`_isomap.py:386-471`) links each query into the TRAINING geodesic graph: `nbrs_.kneighbors(X)` (`:411`) then `G_X[i,j] = (dist_to_neighbor + dist_matrix_[neighbor,j])²` (`:419-460`) projected via KernelPCA. ferrolearn HAS `impl Transform<Array2<f64>> for FittedIsomap` (`isomap.rs:388`) using stored `eigenvalues_`/`eigenvectors_`/`geo_sq_row_means_`/`geo_sq_grand_mean_` (`isomap.rs:454-468`), BUT it approximates the test→train geodesic with the raw test→train EUCLIDEAN distance (`isomap.rs:436`, `:451`) instead of linking through the trained graph (`_isomap.py:419-460`), and inherits the missing `svd_flip` sign (REQ-1). Its own test only asserts "should be correlated", not equal (`test_isomap_transform_recovers_training`, `isomap.rs:557`). No live-oracle parity established. |
| REQ-6 (`radius` mode + `radius_neighbors_graph`) | NOT-STARTED | open prereq blocker **#1470**. sklearn builds `radius_neighbors_graph(nbrs_, radius, mode="distance")` when `radius` is set (`_isomap.py:253-261`) and raises if both `n_neighbors` and `radius` are given (`:212-217`). ferrolearn's `Isomap` (`isomap.rs:54`) has no `radius` field — only `n_neighbors`. |
| REQ-7 (`path_method` FW/D + `eigen_solver` / `tol` / `max_iter`) | NOT-STARTED | open prereq blocker **#1471**. sklearn `shortest_path(method=path_method, directed=False)` selects `"auto"`/`"FW"`/`"D"` (`_isomap.py:299`, ctor `:191`) and `KernelPCA(eigen_solver, tol, max_iter)` (`_isomap.py:236-238`) selects dense `eigh` vs `arpack`/`randomized`. ferrolearn hard-codes Dijkstra (`all_pairs_shortest_paths`, `isomap.rs:228`) and dense `eigh_faer` (`mds.rs:254`) — no `path_method`/`eigen_solver`/`tol`/`max_iter` fields on `Isomap` (`isomap.rs:54`). |
| REQ-8 (`metric` / `p` non-Euclidean distances) | NOT-STARTED | open prereq blocker **#1472**. sklearn `Isomap(metric="minkowski", p=2, metric_params=None)` (`_isomap.py:194-196`) flows `metric`/`p` into `NearestNeighbors` + `kneighbors_graph` (`:223-224`, `:246-247`). ferrolearn hard-codes squared Euclidean (`pairwise_sq_distances`, `isomap.rs:300`; `.sqrt()` in `build_knn_graph`, `isomap.rs:167`) — no `metric`/`p` field on `Isomap` (`isomap.rs:54`). |
| REQ-9 (`reconstruction_error()` + `kernel_pca_`/`nbrs_`/`dist_matrix_` attrs + `neighbors_algorithm` + degenerate-eigenvalue carve-out) | NOT-STARTED | open prereq blocker **#1473** (degenerate subspace is a CARVE-OUT). sklearn exposes `reconstruction_error()` (`_isomap.py:312-335`) and the attrs `embedding_`/`kernel_pca_`/`nbrs_`/`dist_matrix_` (`:228`,`:233`,`:299`,`:309`), the `neighbors_algorithm` ctor field (`:192`), and a `connected_components` graph-completion fallback (`:267-297`). `FittedIsomap` exposes only `embedding()` (`isomap.rs:123`) — no `reconstruction_error`, no public `dist_matrix_`/`nbrs_`/`kernel_pca_` accessors (eigenvalues/vectors stored privately for `transform`, `isomap.rs:110-117`), no `neighbors_algorithm`, and it ERRORS on disconnected graphs (`isomap.rs:309-320`) rather than completing them. **CARVE-OUT:** repeated eigenvalues give a rotation-arbitrary eigenvector subspace even after `svd_flip` ⇒ element-wise parity structurally impossible there (the fixture's eigenvalues are distinct, avoiding this). |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker **#1474**. `grep -rln Isomap ferrolearn-python/` is EMPTY — no `_RsIsomap`, so `import ferrolearn` cannot reach `Isomap`/`transform`/`embedding_`. The only non-test consumer of `fit`/`embedding()` is the crate re-export (`lib.rs:89`). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#1475**. `isomap.rs` computes on `ndarray::Array2<f64>` (`isomap.rs:41`) and eigendecomposes via faer (`use crate::mds::eigh_faer`, `isomap.rs:38`; `eigh_faer`, `mds.rs:203` → faer), not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`isomap.rs` follows the unfitted/fitted split (CLAUDE.md naming): `Isomap {
n_components: usize, n_neighbors: usize }` (`isomap.rs:54`; `new(n_components)`
defaulting `n_neighbors=5` `isomap.rs:66`, builder `with_n_neighbors`
`isomap.rs:75`, accessors `n_components()` / `n_neighbors()`) → `Fit<Array2<f64>,
()>` → `FittedIsomap { embedding_, x_train_, _n_neighbors, eigenvalues_,
eigenvectors_, geo_sq_row_means_, geo_sq_grand_mean_ }` (`isomap.rs:103`, accessor
`embedding()` `isomap.rs:123`). The path is `f64`-only (operates on `Array2<f64>`,
not generic `F`); `fit` returns `Result<_, FerroError>` (R-CODE-2). Unlike LLE,
there IS a `Transform` impl (`isomap.rs:388`) — but it is a Euclidean-approximation
Nystroem, not sklearn's geodesic-graph-linked `transform` (REQ-5).

**Fit path (`fn fit` `isomap.rs:258`).** Validates `n_components != 0`,
`n_neighbors != 0`, `n_samples >= 2`, `n_neighbors < n_samples`,
`n_components <= n_samples` (`isomap.rs:261-297`). Step 1:
`pairwise_sq_distances(x)` (`isomap.rs:300`, from `crate::mds`). Step 2:
`build_knn_graph(&sq_dist, n_neighbors)` (`isomap.rs:303`) — REQ-2. Step 3:
`all_pairs_shortest_paths(&adj)` (`isomap.rs:306`) — Dijkstra from every source,
REQ-2; disconnected ⇒ `NumericalInstability` (`isomap.rs:309-320`). Step 4:
`geo_sq = geodesic²` (`isomap.rs:323`), then `classical_mds(&geo_sq, n_components)`
(`isomap.rs:339`) for the embedding — REQ-1 — AND a parallel manual
double-centering / `eigh_faer` (`isomap.rs:342-374`) that stores the top
`n_components` kernel eigenvalues/eigenvectors plus the row/grand means
(`geo_sq_row_means_`, `geo_sq_grand_mean_`) for the `transform` Nystroem extension.

**Core MDS embedding (`classical_mds` `mds.rs:221`) — the embedding (REQ-1).**
Double-center `B[i,j] = −0.5·(D²[i,j] − rowmean_i − colmean_j + grandmean)`
(`mds.rs:246-251`) — exactly KernelPCA's `KernelCenterer` of `G = −0.5·dist_matrix_²`
(`_isomap.py:306-307`). Eigendecompose via `eigh_faer` (`mds.rs:254`), sort
DESCENDING (`mds.rs:256-262`), take the top `n_components` as
`embedding[i,k] = v_ik · √(max(λ_k, 0))` (`mds.rs:264-273`) — exactly KernelPCA's
`X = eigenvectors_ · √eigenvalues_` (`_kernel_pca.py:446`). **The svd_flip sign
(REQ-1, SHIPPED, RESOLVED #1468):** KernelPCA applies `svd_flip(u=eigenvectors_,
v=None)` (`_kernel_pca.py:373`) for a DETERMINISTIC sign — per eigenvector column,
`argmax(|u.T|)` finds the max-abs row and `signs = sign(that entry)` flips the
column so that entry is POSITIVE (`u_based_decision=True`, `extmath.py:888-896`).
`fn fit` (`isomap.rs:258`) now applies exactly this convention after `classical_mds`:
for each embedding column it finds the row of maximum absolute value and negates the
column so that entry is positive (because `√λ_k ≥ 0` is a positive per-column scale,
`argmax(|scaled|)` equals `argmax(|eigenvector|)`, matching svd_flip on the unit
vector). Magnitudes match (REQ-2 confirms the geodesic input); the eigenvalues are
distinct (`[13.807432, 5.052949]`), so the flip yields EXACT element-wise parity.

**The kNN + geodesic graph (`build_knn_graph` `isomap.rs:159`, `dijkstra`
`isomap.rs:199`, `all_pairs_shortest_paths` `isomap.rs:228`) — REQ-2.** For each
point, collect `(dist, idx)` for all others, `sort_by` ascending, take `k`
(`isomap.rs:165-172`); then SYMMETRIZE (`i→j ⇒ j→i`, dedup keeping the shorter
weight, `isomap.rs:175-195`) — matching `kneighbors_graph(mode="distance")` +
`shortest_path(directed=False)` (`_isomap.py:242-251`, `:299`). `dijkstra`
(`isomap.rs:199`) is a standard binary-heap min-Dijkstra (the `State` `Ord` flips
the comparison for a min-heap, `isomap.rs:141-149`); `all_pairs_shortest_paths`
runs it from every source. Verified against the live `dist_matrix_[0]` oracle.

**sklearn (target contract).** `class Isomap` (`_isomap.py:130`) takes
`__init__(n_neighbors=5, radius=None, n_components=2, eigen_solver="auto", tol=0,
max_iter=None, path_method="auto", neighbors_algorithm="auto", n_jobs=None,
metric="minkowski", p=2, metric_params=None)` (`:182-209`). `_fit_transform`
(`:211`) fits `nbrs_ = NearestNeighbors(...)` (`:219-228`) and `kernel_pca_ =
KernelPCA(n_components, kernel="precomputed", eigen_solver, tol, max_iter)`
(`:233-240`), builds `nbg = kneighbors_graph(nbrs_, n_neighbors, mode="distance")`
(`:242-251`), completes disconnected components (`:267-297`), sets `self.dist_matrix_
= shortest_path(nbg, method=path_method, directed=False)` (`:299`), forms `G =
dist_matrix_²; G *= −0.5` (`:306-307`), and sets `self.embedding_ =
kernel_pca_.fit_transform(G)` (`:309`). `reconstruction_error()` (`:312-335`) =
`√(Σ G_center² − Σ evals²)/n`. `transform` (`:386-471`) links new points through
the training geodesic graph.

**The remaining gap.** ferrolearn ships the embedding value parity (REQ-1, with the
`svd_flip` per-column max-abs-positive sign, RESOLVED #1468), the geodesic distance
matrix (REQ-2), the structural shape/determinism + disconnected-graph error (REQ-3),
and the scoped error contracts (REQ-4). It lacks: a sklearn-faithful geodesic-linked
`transform` (REQ-5, the current one is a Euclidean approximation); `radius` mode
(REQ-6); `path_method`/`eigen_solver`/`tol`/`max_iter` (REQ-7); `metric`/`p`
(REQ-8); `reconstruction_error()` + the `kernel_pca_`/`nbrs_`/`dist_matrix_` attr
surface + `neighbors_algorithm` + graph-completion + degenerate-subspace carve-out
(REQ-9); the PyO3 binding (REQ-10); and the ferray substrate (REQ-11). This is a
**embedding-SHIPPED-out-of-sample-NOT-STARTED** unit (4 SHIPPED / 7 NOT-STARTED).

## Verification

Library crate (green at baseline `71fcc81f`):
```bash
cargo test -p ferrolearn-decomp isomap                       # in-module #[test]s + doctest
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-2/3/4 (STRUCTURAL): `test_isomap_basic_shape`
`(9,2)`, `test_isomap_1d` (ncols 1), `test_isomap_preserves_ordering` (1D line
geodesic ordering — REQ-2/3); `test_isomap_invalid_n_components_zero`,
`test_isomap_invalid_n_neighbors_zero`, `test_isomap_n_neighbors_too_large`,
`test_isomap_n_components_too_large`, `test_isomap_insufficient_samples` (REQ-4);
plus the module doctest. The `tests/divergence_isomap.rs` value-parity suite pins
REQ-1 (SHIPPED): `divergence_embedding_raw_sign` + `green_exact_parity_k3_nc2`,
`_k5_nc2`, `_nc1`, `_15x4_nc3`, `_12x3_k4`, `_argmax_non_endpoint` assert EXACT
element-wise parity (NO sign alignment) against the live sklearn 1.5.2 oracle across
n_neighbors {3,4,5}, n_components {1,2,3}, and fixtures 10×3/12×3/15×4.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-1 value parity
target, the REQ-2 geodesic parity, and the REQ-9 surface:
```bash
# REQ-1 (embedding value parity; ferrolearn matches sklearn EXACTLY element-wise, svd_flip applied, #1468 RESOLVED):
python3 -c "import numpy as np; from sklearn.manifold import Isomap
X=np.array([[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]])
print(np.round(Isomap(n_components=2,n_neighbors=4).fit(X).embedding_,8).tolist())"
# -> [[-1.43997879,-1.13229201],...,[-0.61171047,1.17436347]]  (matches EXACTLY, svd_flip applied)

# REQ-2 (geodesic distance parity: kNN + Dijkstra == kneighbors_graph + shortest_path):
python3 -c "import numpy as np; from sklearn.manifold import Isomap
X=np.array([[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]])
print(np.round(Isomap(n_components=2,n_neighbors=4).fit(X).dist_matrix_[0],6).tolist())"
# -> [0.0,1.004988,2.029683,3.039633,1.135782,1.862794,2.835908,3.781824,2.009975,2.443451]
```
The REQ-1 value pin is `Isomap::new(2).with_n_neighbors(4).fit(&X).unwrap()
.embedding()` matching the Probe REQ-1 target element-wise to < 1e-6 EXACTLY
(sign-determined by the max-abs-positive `svd_flip` applied in `fit`), GREEN in
`tests/divergence_isomap.rs`. The REQ-2 pin is the geodesic `dist_matrix_[0]` row.

ferrolearn-python (REQ-10 binding parity, after `#1474` lands):
```bash
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_isomap.py -q
```
asserting `ferrolearn.Isomap` exists and exposes `embedding_` / `transform`, with
its `embedding` validated against `sklearn.manifold.Isomap`.

## Blockers

(#1467 is this doc's crosslink tracking issue. The blockers below are the open work
items the dispatcher files / numbers; none are filed by this doc — markdown only.
#1468 (REQ-1, the `svd_flip` sign) is RESOLVED — listed here only for provenance.)

- **#1468** — REQ-1 (RESOLVED): added KernelPCA's `svd_flip` sign convention in
  `fn fit` (`isomap.rs:258`, after `classical_mds`) — per embedding column, find the
  row of maximum ABSOLUTE value and negate the column so that entry is positive
  (`u_based_decision=True`, `_kernel_pca.py:373` / `extmath.py:888-896`). With the
  flip, `embedding` matches sklearn EXACTLY element-wise (distinct eigenvalues),
  pinned GREEN by `divergence_embedding_raw_sign` + the `green_exact_parity_*` probes
  in `tests/divergence_isomap.rs`. REQ-1 is SHIPPED.
- **#1469** — REQ-5: replace the Euclidean-approximation Nystroem `transform`
  (`isomap.rs:388`) with sklearn's geodesic-graph-linked `transform`
  (`_isomap.py:386-471`): `nbrs_.kneighbors(X)` then `G_X[i,j] = (dist_to_neighbor +
  dist_matrix_[neighbor,j])²` projected via KernelPCA (the `svd_flip` sign from #1468
  is already applied in `fit`).
- **#1470** — REQ-6: add a `radius` field + `radius_neighbors_graph` mode
  (`_isomap.py:253-261`), rejecting both `n_neighbors` and `radius`
  simultaneously (`:212-217`).
- **#1471** — REQ-7: add `path_method` (`"auto"`/`"FW"`/`"D"`, `_isomap.py:299`)
  and `eigen_solver`/`tol`/`max_iter` (`_isomap.py:236-238`) ctor fields.
- **#1472** — REQ-8: add `metric`/`p`/`metric_params` (`_isomap.py:194-196`),
  flowing into the distance computation (currently hard-coded Euclidean,
  `isomap.rs:300`/`167`).
- **#1473** — REQ-9: expose `reconstruction_error()` (`_isomap.py:312-335`) and
  `dist_matrix_`/`nbrs_`/`kernel_pca_` accessors, add `neighbors_algorithm` + the
  `connected_components` graph-completion fallback (`_isomap.py:267-297`). The
  degenerate-eigenvalue subspace is an inherent CARVE-OUT (rotation-arbitrary even
  after `svd_flip`).
- **#1474** — REQ-10: add `_RsIsomap` to `ferrolearn-python` (fit / transform /
  embedding_) — the boundary CPython consumer (`grep -rln Isomap ferrolearn-python/`
  is empty).
- **#1475** — REQ-11: migrate `isomap.rs` off `ndarray` / faer
  (`crate::mds::eigh_faer`, `isomap.rs:38`; `mds.rs:203`) to `ferray-core` /
  `ferray::linalg` (R-SUBSTRATE).
