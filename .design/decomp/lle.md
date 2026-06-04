# Locally Linear Embedding (sklearn.manifold.LocallyLinearEmbedding)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 91124e74
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/manifold/_locally_linear.py  # barycenter_weights (:29-80): per point C = Y[ind] - X[i] (:70), G = C Cᵀ (:71), trace = np.trace(G) (:72), R = reg*trace if trace>0 else reg (:73-76, NO /k), G.flat[::n_neighbors+1] += R (:77), w = solve(G, ones, assume_a="pos") (:78), B[i] = w / Σw (:79). null_space (:125-196): dense path eigh(M, subset_by_index=(k_skip, k+k_skip-1)) (:192-194) then eigen_vectors[:, argsort(abs(eigenvalues))] (:195-196) — eigenvectors UNIT-NORM, NO sqrt-eigenvalue scaling, NO deterministic sign flip; arpack path eigsh(M, k+k_skip, sigma=0.0), eigen_vectors[:, k_skip:] (:173-188). _locally_linear_embedding (:201-): method=="standard" builds W = barycenter_kneighbors_graph (:235), M = (I-W)ᵀ(I-W) (:239-246), null_space(M, n_components, k_skip=1) (:295-301). class LocallyLinearEmbedding(BaseEstimator,...) (:603-): __init__(n_neighbors=5, n_components=2, reg=1e-3, eigen_solver="auto", tol=1e-6, max_iter=100, method="standard", hessian_tol=1e-4, modified_tol=1e-12, neighbors_algorithm="auto", random_state=None, n_jobs=None) (:756-783); method ∈ {standard, hessian, modified, ltsa}; _fit_transform (:785) sets nbrs_, embedding_, reconstruction_error_; transform out-of-sample via barycenter_weights (:851-).
ferrolearn-module: ferrolearn-decomp/src/lle.rs
parity-ops: LocallyLinearEmbedding
crosslink-issue: 1459
-->

## Summary

`ferrolearn-decomp/src/lle.rs` mirrors scikit-learn's `LocallyLinearEmbedding`
(`sklearn/manifold/_locally_linear.py`, `class LocallyLinearEmbedding` `:603`)
for the `method='standard'` arm: find k-nearest neighbors, solve the local
least-squares reconstruction weights `W` (`compute_weights`, `lle.rs:162`),
build `M = (I − W)ᵀ(I − W)`, eigendecompose, and take the bottom
`n_components` eigenvectors (skipping the trivial near-zero eigenvector) as the
embedding (`fn fit`, `lle.rs:272`). The exposed surface is the unfitted
`LLE { n_components, n_neighbors, reg }` (`lle.rs:50`) and the fitted
`FittedLLE { embedding_ }` (`lle.rs:113`), re-exported at the crate root
(`pub use lle::{FittedLLE, LLE}`, `lib.rs:94`).

**STANDARD-LLE VALUE PARITY (SHIPPED, was DIV `#1460`, now RESOLVED).**
ferrolearn's local-covariance regularization now matches sklearn's
`barycenter_weights`: `R = reg * trace if trace > 0 else reg` (`compute_weights`,
`lle.rs:187` — **NO /k**; sklearn `_locally_linear.py:73-76`). With the matched
regularizer, ferrolearn's reconstruction weights `W` match `barycenter_weights`,
`M = (I−W)ᵀ(I−W)` matches, and the bottom `n_components` eigenvectors of `M`
match sklearn's `null_space` selection up to a **per-component sign** (the sign
of each eigenvector is solver-arbitrary — an inherent carve-out, see REQ-7).
The sign-robust `|embedding|` is verified element-wise (sign-aligned to
max-abs-positive) against the live sklearn 1.5.2 oracle to tol `1e-6` across SIX
configurations — different `n_neighbors` (k=3,5), different `reg` (1e-2,1e-1),
`n_components=1`, an 18×3 larger fixture, a 12×5 higher-D fixture, and a
near-degenerate-trace fixture — in `tests/divergence_lle.rs`
(`divergence_standard_lle_embedding_magnitude` + `parity_*`). Hence REQ-1
(value parity) and REQ-2 (weights `W`) are **SHIPPED** (the `#1460` reg fix
landed); REQ-3..5 (the `M` construction + bottom-eigenvector selection, the
structural shape/determinism, and the error/parameter contracts) are SHIPPED
scoped.

At baseline `91124e74` (reg fix `#1460` applied): the standard-LLE `|embedding|`
value parity (REQ-1) and the reconstruction weights `W` (REQ-2) are SHIPPED; the
`M = (I−W)ᵀ(I−W)` + bottom-eigenvector extraction (REQ-3), embedding
shape/determinism (REQ-4), and error/parameter contracts (REQ-5) are SHIPPED
scoped. The non-standard `method` arms (REQ-6), the
`eigen_solver='arpack'`/`random_state`/sign convention (REQ-7), `transform`
out-of-sample (REQ-8), the `reconstruction_error_`/`nbrs_` attrs +
`neighbors_algorithm`/`tol`/`max_iter` surface (REQ-9), the PyO3 binding
(REQ-10), and the ferray substrate (REQ-11) are NOT-STARTED — **5 SHIPPED / 6
NOT-STARTED**.

`LLE` / `FittedLLE` are existing pub APIs whose non-test consumer is the crate
re-export (`lib.rs:94`, boundary public API, grandfathered S5/R-DEFER-1). There
is **no PyO3 binding** (`grep -rln LocallyLinear ferrolearn-python/` is empty;
the `lle` substring match in `classifiers.rs` is the spurious `ndarray::Array1`
token) and **no `transform` / `Transform`** (sklearn's `LocallyLinearEmbedding`
DOES have an out-of-sample `transform` — REQ-8).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# REQ-1 (SHIPPED; was DIV #1460, now RESOLVED) — the |embedding| parity target.
# X is the hard-coded 10x3 fixture (R-CHAR-3). ferrolearn's |embedding| matches this
# to ~1e-8; the per-component SIGN is solver-arbitrary (REQ-7 carve-out).
python3 -c "import numpy as np; from sklearn.manifold import LocallyLinearEmbedding
X=np.array([[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]])
e=LocallyLinearEmbedding(n_components=2,n_neighbors=4,reg=1e-3,method='standard',eigen_solver='dense').fit(X)
print(np.round(e.embedding_,8).tolist())"
# -> [[0.24567158,0.56740396],[0.01227528,0.40123299],[-0.20867548,0.18628911],[-0.4724819,0.10606641],
#     [0.27310581,0.08680629],[0.04109518,-0.08152757],[-0.24079918,-0.14048714],[-0.44342955,-0.37732988],
#     [0.51325342,-0.28842866],[0.27998483,-0.46002551]]
#    With the #1460 reg fix (reg*trace, NO /k), |ferrolearn| == |sklearn| to ~1e-8 element-wise
#    (sign-aligned per column; the per-component SIGN is solver-arbitrary, REQ-7).

# REQ-2 (SHIPPED; was DIV #1460, now RESOLVED) — reconstruction-weights parity on the neighbor block of point 0.
# sklearn AND ferrolearn now both use R = reg*trace (NO /k, _locally_linear.py:73-76 / lle.rs:187). k=4.
python3 -c "import numpy as np; from scipy.linalg import solve
X=np.array([[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]])
d=((X-X[0])**2).sum(1); ind=np.argsort(d)[1:5]; C=X[ind]-X[0]; G=C@C.T; tr=np.trace(G); k=4; reg=1e-3; v=np.ones(k)
Gs=G.copy(); Gs.flat[::k+1]+=reg*tr; ws=solve(Gs,v,assume_a='pos'); ws/=ws.sum()
print('R=reg*trace w:', np.round(ws,8).tolist())
print('trace',tr,'reg*tr',reg*tr)"
# -> R=reg*trace w: [1.21836952, 0.59929776, -1.02925244, 0.21158516]
# -> trace 9.81 reg*tr 0.00981
#    ferrolearn's compute_weights (lle.rs:187, R = reg*trace) now produces these same weights -> same M -> same embedding.

# REQ-9 — sklearn ctor defaults + fitted attrs + has transform.
python3 -c "from sklearn.manifold import LocallyLinearEmbedding as L; import numpy as np
m=L(); print('defaults n_neighbors',m.n_neighbors,'n_components',m.n_components,'reg',m.reg,'method',m.method,
'eigen_solver',m.eigen_solver,'tol',m.tol,'max_iter',m.max_iter,'neighbors_algorithm',m.neighbors_algorithm)
X=np.array([[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]])
e=L(n_components=2,n_neighbors=4).fit(X)
print('attrs:',[a for a in ('embedding_','reconstruction_error_','nbrs_') if hasattr(e,a)],'recon_err',e.reconstruction_error_)
print('has transform', hasattr(e,'transform'))"
# -> defaults n_neighbors 5 n_components 2 reg 0.001 method standard eigen_solver auto tol 1e-06 max_iter 100 neighbors_algorithm auto
# -> attrs: ['embedding_', 'reconstruction_error_', 'nbrs_'] recon_err 0.0001765118939348609
# -> has transform True
```

## Requirements

- REQ-1: **Standard-LLE `|embedding|` VALUE parity, sign-robust (SHIPPED; was DIV
  `#1460`, now RESOLVED).** With `method='standard'` and `eigen_solver='dense'`,
  sklearn returns the bottom eigenvectors of `M = (I−W)ᵀ(I−W)` (`null_space`,
  `_locally_linear.py:192-196`), UNIT-NORM with NO sqrt-eigenvalue scaling. The
  parity target on the 10×3 fixture is the Probe REQ-1 embedding. With the
  matched regularizer (`reg*trace`, NO /k — `lle.rs:187` =
  `_locally_linear.py:73-76`), `|ferrolearn.embedding|` matches
  `|sklearn.embedding_|` to ~1e-8 (the per-component SIGN remains
  solver-arbitrary — REQ-7 carve-out). Pinned by
  `divergence_standard_lle_embedding_magnitude` + the `parity_*` probes in
  `tests/divergence_lle.rs` across n_neighbors/reg/n_components/larger/higher-D
  fixtures.

- REQ-2: **Reconstruction weights `W` (local least-squares + reg + normalize)
  (SHIPPED; was DIV `#1460`, now RESOLVED; coupled to REQ-1).** `compute_weights`
  (`lle.rs:162`) mirrors `barycenter_weights` (`_locally_linear.py:29-80`): per
  point `i`, `Z[j] = x[i] − x[neighbor_j]` (`lle.rs:178`; sklearn
  `C = Y[ind] − X[i]` `:70`), `C = Z Zᵀ` (`lle.rs:183`; sklearn `G = C Cᵀ` `:71`),
  `trace = Σ C_jj` (`lle.rs:186`; sklearn `np.trace(G)` `:72`), add the
  regularizer to the diagonal (`lle.rs:190-192`; sklearn `G.flat[::n+1] += R`
  `:77`), solve `C w = 1` (`lle.rs:194-249` Gaussian elimination; sklearn
  `solve(G, ones, assume_a="pos")` `:78`), normalize `w /= Σw` (`lle.rs:252-257`;
  sklearn `w / np.sum(w)` `:79`). The regularizer now matches: ferrolearn
  `reg_val = reg * trace if trace > 0 else reg` (`lle.rs:187`) = sklearn
  `R = reg * trace if trace > 0 else reg` (`:73-76`, NO /k — the `#1460` fix). The
  entire `compute_weights` path matches `barycenter_weights`; verified via the
  embedding parity (REQ-1).

- REQ-3: **`M = (I−W)ᵀ(I−W)` + bottom-eigenvector extraction, skipping the trivial
  eigenvector (SHIPPED scoped).** `fn fit` (`lle.rs:272`) forms `I − W`
  (`lle.rs:343-349`) and `M = (I−W)ᵀ(I−W)` (`lle.rs:351`) — exactly sklearn's
  `M = (I−W)ᵀ(I−W)` (`_locally_linear.py:239-246`). It eigendecomposes `M`
  (`eigh_faer`, `lle.rs:354`), sorts eigenvalues ASCENDING (`lle.rs:357-362`),
  SKIPS the smallest (trivial ~0) eigenvector and takes the next `n_components`
  RAW into `embedding_` (`lle.rs:367-371`). This is sklearn's `null_space(M,
  n_components, k_skip=1)` dense selection: `eigh(M, subset_by_index=(1,
  n_components))` then ordering by `argsort(abs(eigenvalues))`
  (`_locally_linear.py:192-196`) — the same "drop the bottom trivial eigenvector,
  keep the next `n_components`" rule, eigenvectors taken RAW (unit-norm, no
  sqrt-eigenvalue scaling). **STRUCTURAL claim** (the selection rule + `M`
  construction); the eigenvector VALUES (now matching, REQ-1/REQ-2) depend on `W`.

- REQ-4: **Structural embedding shape + determinism (SHIPPED scoped).** `fn fit`
  (`lle.rs:272`) returns `FittedLLE { embedding_ }` whose `embedding()`
  (`lle.rs:121`) is `Array2<f64>` of shape `(n_samples, n_components)`
  (`Array2::zeros((n, n_comp))`, `lle.rs:366`), finite, and DETERMINISTIC given
  the input (the `find_neighbors` + `eigh_faer` path uses no RNG, unlike sklearn's
  `eigen_solver='arpack'` `random_state` path). Mirrors the
  `(n_samples, n_components)` output shape of `_locally_linear_embedding`
  (`_locally_linear.py:554`).

- REQ-5: **Error / parameter contracts (SHIPPED scoped).** `fn fit` (`lle.rs:272`)
  returns `InvalidParameter { name: "n_components", reason: "must be at least 1" }`
  for `n_components == 0` (`lle.rs:290-295`), `InsufficientSamples { required: 2 }`
  for `< 2` samples (`lle.rs:302-308`), `InvalidParameter` for `n_neighbors == 0`
  (`lle.rs:296-301`), `n_neighbors >= n_samples` (`lle.rs:309-317`; sklearn raises
  `ValueError("Expected n_neighbors <= n_samples...")` `_locally_linear.py:226-230`),
  `n_components >= n_samples` (`lle.rs:319-327`), and `reg < 0` (`lle.rs:328-333`).
  `NumericalInstability` is returned when a local covariance is singular
  (`lle.rs:216-223`). **FLAG (candidate DIVs):** sklearn's bound is `n_components
  > d_in` (output dim ≤ INPUT dim, `_locally_linear.py:222-225`), NOT
  `n_components >= n_samples`; and sklearn raises `ValueError`/`InvalidParameterError`,
  not the `FerroError` ABI.

- REQ-6: **`method ∈ {hessian, modified, ltsa}` (NOT-STARTED; `#1461`).** sklearn
  `_locally_linear_embedding` (`_locally_linear.py:234-`) supports four `method`
  arms — `standard` (`:234`), `hessian` (Hessian eigenmaps, `:248`), `modified`
  (MLLE, `:280`), `ltsa` (local tangent-space alignment) — each building a
  different `M`. ferrolearn implements ONLY `standard`; there is no `method`
  parameter on `LLE`.

- REQ-7: **`eigen_solver='arpack'` + `random_state` + per-component sign convention
  (NOT-STARTED; `#1462`, CARVE-OUT for the sign).** sklearn's
  `null_space` has an `arpack` path (`eigsh(M, k+k_skip, sigma=0.0, ..., v0)`,
  `_locally_linear.py:173-188`) seeded by `random_state`; and even on the dense
  path the per-component SIGN of each eigenvector is solver-arbitrary — sklearn
  applies NO deterministic sign flip, so `embedding_[:, j]` and `−embedding_[:, j]`
  are equally valid. ferrolearn has only the dense `eigh_faer` path, no
  `eigen_solver`/`random_state` field, and no sign convention. Exact per-component
  SIGN parity is structurally impossible (inherent carve-out); the verifiable
  target is the sign-robust `|embedding|` (REQ-1, SHIPPED).

- REQ-8: **`transform` out-of-sample (NOT-STARTED; `#1463`).** sklearn's
  `LocallyLinearEmbedding.transform` (`_locally_linear.py:851-`) embeds NEW points
  by computing their `barycenter_weights` against the fitted `nbrs_` and applying
  them to the training `embedding_`. ferrolearn has NO `Transform` impl and no
  `transform` method — `FittedLLE` exposes only `embedding()` (`lle.rs:121`).

- REQ-9: **`reconstruction_error_` / `nbrs_` / `embedding_` fitted attrs +
  `neighbors_algorithm` / `tol` / `max_iter` (NOT-STARTED; `#1464`).** sklearn's
  `_fit_transform` (`_locally_linear.py:785`) sets `nbrs_` (the fitted
  `NearestNeighbors`), `embedding_`, and `reconstruction_error_` (`squared_error`,
  Probe REQ-9: `0.000177`), and the ctor takes `neighbors_algorithm="auto"`,
  `tol=1e-6`, `max_iter=100` (`:762-783`). ferrolearn exposes only `embedding()`;
  there is no `reconstruction_error_` accessor, no `nbrs_`, no
  `neighbors_algorithm` / `tol` / `max_iter` fields.

- REQ-10: **PyO3 binding (NOT-STARTED; `#1465`).** `import ferrolearn` exposing a
  registered `LocallyLinearEmbedding` marshalling `fit`/`transform` and
  `embedding_`/`reconstruction_error_` — the boundary CPython consumer. Absent
  (`grep -rln LocallyLinear ferrolearn-python/` is empty).

- REQ-11: **ferray substrate (NOT-STARTED; `#1466`).** `lle.rs` computes on
  `ndarray::Array2<f64>` and eigendecomposes via faer (`crate::mds::eigh_faer`,
  `lle.rs:36`/`354`), not `ferray-core` arrays / `ferray::linalg`
  (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). Fixture `X` = the 10×3 point set
`[[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],
[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]]`, with
`n_components=2, n_neighbors=4, reg=1e-3, method='standard', eigen_solver='dense'`.

- AC-1 (REQ-1, SHIPPED; was DIV `#1460`, now RESOLVED): with the matched reg
  formula, `LLE::new(2).with_n_neighbors(4).fit(&X).unwrap().embedding()` matches
  the Probe REQ-1 target `[[0.24567158,0.56740396],...]` in ABSOLUTE VALUE
  (sign-aligned per column) to max abs err < 1e-6 (the per-component SIGN may flip
  — REQ-7). Pinned by `divergence_standard_lle_embedding_magnitude` plus the
  `parity_*` probes (n_neighbors k=3/5, reg 1e-2/1e-1, n_components=1, 18×3,
  12×5, near-degenerate trace) in `tests/divergence_lle.rs`.

- AC-2 (REQ-2, SHIPPED; was DIV `#1460`, now RESOLVED): on the 4-neighbor block of
  point 0, ferrolearn's weights now coincide with sklearn's
  `[1.21836952,0.59929776,-1.02925244,0.21158516]` (`R=reg*trace`, NO /k) to
  ~1e-8; verified via the embedding parity (AC-1).

- AC-3 (REQ-3, SHIPPED scoped): `fn fit` (`lle.rs:272`) forms `M =
  (I−W)ᵀ(I−W)` (`lle.rs:351`), eigendecomposes (`lle.rs:354`), sorts ASCENDING
  (`lle.rs:357-362`), SKIPS the smallest and takes the next `n_components`
  (`lle.rs:367-371`) — the `null_space(M, k, k_skip=1)` selection rule
  (`_locally_linear.py:192-196`). Pinned (structurally) by
  `test_lle_basic_shape` `(9,2)`, `test_lle_1d`, `test_lle_preserves_local_structure`
  (1D line embedding is monotonic — neighbor ordering preserved).

- AC-4 (REQ-4, SHIPPED scoped): `LLE::new(2).with_n_neighbors(3).fit(&grid)
  .unwrap().embedding()` has shape `(9, 2)`, is finite, and is identical across
  runs (deterministic, no RNG). Pinned by `test_lle_basic_shape` `(9,2)`,
  `test_lle_1d` (ncols 1), `test_lle_larger_dataset` `(20,2)`,
  `test_lle_different_n_neighbors` (different `n_neighbors` ⇒ different embedding),
  `green_determinism`, and `green_embedding_columns_centered`
  (`tests/divergence_lle.rs`).

- AC-5 (REQ-5, SHIPPED scoped): `fit` returns `Err` for `n_components=0`,
  `n_neighbors=0`, `n_neighbors >= n_samples`, `n_components >= n_samples`,
  `n_samples < 2`, and `reg < 0`. Pinned by `test_lle_invalid_n_components_zero`,
  `test_lle_invalid_n_neighbors_zero`, `test_lle_n_neighbors_too_large`,
  `test_lle_n_components_too_large`, `test_lle_insufficient_samples`,
  `test_lle_negative_reg`. FLAG: sklearn's bound is `n_components > d_in`
  (`_locally_linear.py:222-225`), and it raises `ValueError`, not `FerroError`.

- AC-6 (REQ-6/7/8/9, DIVERGES): `LocallyLinearEmbedding()` defaults `n_neighbors=5,
  n_components=2, reg=1e-3, method='standard', eigen_solver='auto', tol=1e-6,
  max_iter=100, neighbors_algorithm='auto'` (Probe REQ-9), exposes `embedding_` /
  `reconstruction_error_` / `nbrs_` and `hasattr(transform) == True`; ferrolearn
  has only `standard`, the dense solver, `embedding()`, and no `transform`.

- AC-7 (REQ-10/11): no `ferrolearn.LocallyLinearEmbedding` (no binding, `grep
  -rln LocallyLinear ferrolearn-python/` empty); the module imports `ndarray` +
  faer (`crate::mds::eigh_faer`), not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `LLE` / `FittedLLE` are existing pub APIs; the non-test
consumer is the crate re-export (`lib.rs:94`, boundary public API, grandfathered
S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn
1.5.2). Live oracle = installed sklearn 1.5.2, run from `/tmp`. **STANDARD-LLE
VALUE PARITY (R-HONEST-3, was DIV `#1460`, now RESOLVED):** ferrolearn now
regularizes with `reg*trace` (`lle.rs:187`), matching sklearn `reg*trace`
(`_locally_linear.py:73-76`, NO /k) — so `W` matches `barycenter_weights`, `M`
matches, and the embedding `|values|` match the live oracle to tol `1e-6`
(Probe REQ-1/REQ-2, six fixtures in `tests/divergence_lle.rs`). REQ-1 / REQ-2 are
SHIPPED (the `#1460` reg fix landed). The per-component SIGN of each eigenvector
remains an inherent carve-out (REQ-7, `#1462`): sklearn applies no deterministic
sign flip, so only the sign-robust `|embedding|` (REQ-1) is a parity target.
The least-confident SHIPPED claim is REQ-3 — it covers the `M` construction and
the `null_space` SELECTION RULE (skip the trivial eigenvector, keep the next
`n_components`); the eigenvector VALUES are now pinned by the REQ-1/2 parity
probes. #1459 is this doc's crosslink tracking issue. Count: **5 SHIPPED
(REQ-1..5) / 6 NOT-STARTED (REQ-6..11)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (standard-LLE `\|embedding\|` VALUE parity, sign-robust) | SHIPPED | was DIV `#1460`, now RESOLVED. sklearn `method='standard'`+`eigen_solver='dense'` returns the bottom eigenvectors of `M=(I−W)ᵀ(I−W)` (`null_space`, `_locally_linear.py:192-196`), unit-norm, no sqrt scaling. Parity target (Probe REQ-1, R-CHAR-3): on the 10×3 fixture, `embedding_ = [[0.24567158,0.56740396],...,[0.27998483,-0.46002551]]`. With the matched regularizer `reg*trace` (`lle.rs:187`, NO /k = `:73-76`), `\|ferrolearn.embedding\|` == `\|sklearn.embedding_\|` to ~1e-8, the per-component SIGN being solver-arbitrary (REQ-7). Non-test consumer: crate re-export (`lib.rs:94`). Verification: `cargo test -p ferrolearn-decomp --test divergence_lle` → `divergence_standard_lle_embedding_magnitude` + `parity_neighbors_k3`/`_k5`, `parity_reg_1em2`/`_1em1`, `parity_n_components_1`, `parity_larger_18x3`, `parity_higher_dim_12x5`, `parity_near_degenerate_trace_12x3` (sign-aligned, max-abs err < 1e-6) PASS. |
| REQ-2 (reconstruction weights `W`: local LS + reg + normalize) | SHIPPED | was DIV `#1460`, now RESOLVED (coupled to REQ-1). `fn compute_weights` (`lle.rs:162`) mirrors `barycenter_weights` (`_locally_linear.py:29-80`): `Z[j]=x[i]−x[neighbor_j]` (`lle.rs:178` / `:70`), `C=Z Zᵀ` (`lle.rs:183` / `G=C Cᵀ` `:71`), `trace=Σ C_jj` (`lle.rs:186` / `:72`), add regularizer to the diagonal (`lle.rs:190-192` / `G.flat[::n+1]+=R` `:77`), solve `C w=1` (`lle.rs:194-249` / `solve(G,ones,assume_a="pos")` `:78`), normalize `w/=Σw` (`lle.rs:252-257` / `:79`). The regularizer now MATCHES: `reg_val = reg*trace if trace>0 else reg` (`lle.rs:187`) = `R = reg*trace if trace>0 else reg` (`:73-76`, NO /k — the `#1460` fix). Probe REQ-2 (R-CHAR-3): point-0 block → `[1.21836952,0.59929776,-1.02925244,0.21158516]` on both. Verified via the embedding parity (REQ-1, `tests/divergence_lle.rs`). Non-test consumer: crate re-export (`lib.rs:94`). |
| REQ-3 (`M=(I−W)ᵀ(I−W)` + bottom-eigenvector extraction, skip trivial) | SHIPPED | `fn fit` (`lle.rs:272`) builds `I−W` (`iw[[i,i]]=1.0; iw[[i,j]]-=w[[i,j]]`, `lle.rs:343-349`) then `let m = iw.t().dot(&iw)` (`lle.rs:351`) — exactly sklearn `M=(I−W)ᵀ(I−W)` (`_locally_linear.py:239-246`). Eigendecomposes via `eigh_faer(&m)` (`lle.rs:354`), sorts eigenvalues ASCENDING (`indices.sort_by(... eigenvalues[a].partial_cmp(&eigenvalues[b]) ...)`, `lle.rs:357-362`), SKIPS the smallest and takes the next `n_components` RAW (`indices.iter().skip(1).take(n_comp)`, `lle.rs:367-371`) — sklearn's `null_space(M, n_components, k_skip=1)` dense selection: `eigh(M, subset_by_index=(1, n_components))` ordered by `argsort(abs(eigenvalues))` (`_locally_linear.py:192-196`), eigenvectors RAW/unit-norm with NO sqrt-eigenvalue scaling. Non-test consumer: crate re-export (`lib.rs:94`). Verification: `cargo test -p ferrolearn-decomp lle` → `test_lle_basic_shape` `(9,2)`, `test_lle_1d`, `test_lle_preserves_local_structure` (1D embedding monotonic — neighbor order preserved) PASS. **Scope: the `M` construction + null_space SELECTION RULE; the eigenvector VALUES are pinned by REQ-1/2.** |
| REQ-4 (structural embedding shape + determinism) | SHIPPED | `fn fit` (`lle.rs:272`) returns `FittedLLE { embedding_ }` whose `embedding()` (`lle.rs:121`) is `Array2<f64>` of shape `(n_samples, n_components)` — `let mut embedding = Array2::zeros((n, n_comp))` (`lle.rs:366`), finite, DETERMINISTIC given the input (`find_neighbors` `lle.rs:132` + `eigh_faer` use no RNG, unlike sklearn's `eigen_solver='arpack'` `random_state` path). Mirrors the `(n_samples, n_components)` output of `_locally_linear_embedding` (`_locally_linear.py:554`). Non-test consumer: crate re-export (`lib.rs:94`). Verification: `cargo test -p ferrolearn-decomp lle` (`test_lle_basic_shape` `(9,2)`, `test_lle_1d` ncols 1, `test_lle_larger_dataset` `(20,2)`, `test_lle_different_n_neighbors`) + `--test divergence_lle` (`green_embedding_shape`, `green_determinism`, `green_embedding_columns_centered`) PASS. **Scope: SHAPE/finiteness/determinism, NOT value parity (REQ-1).** |
| REQ-5 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`lle.rs:272`) returns `Err(InvalidParameter { name: "n_components", reason: "must be at least 1" })` for `n_components==0` (`lle.rs:290-295`), `Err(InvalidParameter { name: "n_neighbors", ... })` for `n_neighbors==0` (`lle.rs:296-301`) and for `n_neighbors >= n` (`lle.rs:309-317`, sklearn `ValueError("Expected n_neighbors <= n_samples...")` `_locally_linear.py:226-230`), `Err(InvalidParameter { name: "n_components", ... })` for `n_components >= n` (`lle.rs:319-327`), `Err(InsufficientSamples { required: 2, actual: n, context: "LLE::fit requires at least 2 samples" })` for `< 2` samples (`lle.rs:302-308`), `Err(InvalidParameter { name: "reg", reason: "must be non-negative" })` for `reg < 0` (`lle.rs:328-333`), and `Err(NumericalInstability { ... })` on a singular local covariance (`lle.rs:216-223`). Non-test consumer: these guards protect every instance reached via the crate re-export (`lib.rs:94`). Verification: `cargo test -p ferrolearn-decomp lle` (`test_lle_invalid_n_components_zero`, `_invalid_n_neighbors_zero`, `_n_neighbors_too_large`, `_n_components_too_large`, `_insufficient_samples`, `_negative_reg`) + `--test divergence_lle` (`green_err_*`) PASS. **FLAG (candidate DIVs):** sklearn's output-dim bound is `n_components > d_in` (`_locally_linear.py:222-225`), NOT `n_components >= n_samples`; sklearn raises `ValueError`/`InvalidParameterError`, not the `FerroError` ABI. |
| REQ-6 (`method ∈ {hessian, modified, ltsa}`) | NOT-STARTED | open prereq blocker **#1461**. sklearn `_locally_linear_embedding` builds a different `M` for each `method` — `standard` (`_locally_linear.py:234`), `hessian` Hessian eigenmaps (`:248`), `modified` MLLE (`:280`), `ltsa` (`:380+`). ferrolearn implements ONLY `standard` (`fn fit` `lle.rs:272`) and exposes no `method` parameter on `LLE` (`lle.rs:50`). |
| REQ-7 (`eigen_solver='arpack'` + `random_state` + per-component sign convention) | NOT-STARTED | open prereq blocker **#1462** (sign is a CARVE-OUT). sklearn `null_space` has an `arpack` path `eigsh(M, k+k_skip, sigma=0.0, ..., v0=_init_arpack_v0(..., random_state))` (`_locally_linear.py:173-188`); and even on the dense path the per-component SIGN of each eigenvector is solver-arbitrary — sklearn applies NO deterministic sign flip, so `embedding_[:,j]` and `−embedding_[:,j]` are equivalent. ferrolearn has only the dense `eigh_faer` path (`lle.rs:354`), no `eigen_solver`/`random_state` field, no sign convention. Exact per-component SIGN parity is structurally impossible; the verifiable target is the sign-robust `\|embedding\|` (REQ-1, SHIPPED). |
| REQ-8 (`transform` out-of-sample via barycenter weights) | NOT-STARTED | open prereq blocker **#1463**. sklearn `LocallyLinearEmbedding.transform` (`_locally_linear.py:851-`) embeds NEW points by `barycenter_weights` against the fitted `nbrs_`, then applies them to the training `embedding_` (Probe REQ-9: `hasattr(transform)==True`). ferrolearn has NO `Transform` impl and no `transform` method — `FittedLLE` exposes only `embedding()` (`lle.rs:121`). |
| REQ-9 (`reconstruction_error_`/`nbrs_`/`embedding_` attrs + `neighbors_algorithm`/`tol`/`max_iter`) | NOT-STARTED | open prereq blocker **#1464**. sklearn `_fit_transform` (`_locally_linear.py:785`) sets `nbrs_`, `embedding_`, `reconstruction_error_` (Probe REQ-9: `0.0001765...`), and the ctor takes `neighbors_algorithm="auto"`, `tol=1e-6`, `max_iter=100` (`:756-783`). ferrolearn exposes only `embedding()` (`lle.rs:121`) — NO `reconstruction_error_`, NO `nbrs_`, NO `neighbors_algorithm`/`tol`/`max_iter` fields on `LLE` (`lle.rs:50`). |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker **#1465**. `grep -rln LocallyLinear ferrolearn-python/` is EMPTY (the `lle` substring in `classifiers.rs` is the spurious `ndarray::Array1` token) — no `_RsLocallyLinearEmbedding`, so `import ferrolearn` cannot reach `LocallyLinearEmbedding`/`transform`/`embedding_`/`reconstruction_error_`. The only non-test consumer of `fit`/`embedding()` is the crate re-export (`lib.rs:94`). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#1466**. `lle.rs` computes on `ndarray::Array2<f64>` (`lle.rs:39`) and eigendecomposes via faer (`use crate::mds::eigh_faer`, `lle.rs:36`; `eigh_faer(&m)`, `lle.rs:354` → `faer::Mat::self_adjoint_eigen`), not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`lle.rs` follows the unfitted/fitted split (CLAUDE.md naming): `LLE {
n_components: usize, n_neighbors: usize, reg: f64 }` (`lle.rs:50`; `new(n_components)`
defaulting `n_neighbors=5, reg=1e-3` `lle.rs:64`, builders `with_n_neighbors`
`lle.rs:74` / `with_reg` `lle.rs:81`, accessors `n_components()` / `n_neighbors()`
/ `reg()`) → `Fit<Array2<f64>, ()>` → `FittedLLE { embedding_: Array2<f64> }`
(`lle.rs:113`, accessor `embedding()` `lle.rs:121`). The path is `f64`-only
(operates on `Array2<f64>`, not generic `F`); `fit` returns `Result<_,
FerroError>` (R-CODE-2). There is no `Predict`/`Transform` and no `transform` —
unlike sklearn, which DOES have an out-of-sample `transform` (REQ-8).

**Fit path (`fn fit` `lle.rs:272`).** Validates `n_components != 0`,
`n_neighbors != 0`, `n_samples >= 2`, `n_neighbors < n_samples`,
`n_components < n_samples`, `reg >= 0` (`lle.rs:290-333`). Step 1:
`find_neighbors(x, n_neighbors)` (`lle.rs:132`) — for each point, an O(n·d)
squared-distance pass over all other points, `sort_by` ascending, take the `k`
nearest (`lle.rs:138-150`); sklearn uses `NearestNeighbors(n_neighbors+1)`
dropping self (`_locally_linear.py:216-218`, `barycenter_kneighbors_graph:119`).
Step 2: `compute_weights(x, &neighbors, reg)` (`lle.rs:162`) — REQ-2. Step 3:
`M = (I−W)ᵀ(I−W)` (`lle.rs:343-351`). Step 4: `eigh_faer(&m)` (`lle.rs:354`),
sort ASCENDING, skip the trivial smallest, take the next `n_components`
eigenvectors RAW (`lle.rs:357-371`) — REQ-3.

**Core `compute_weights` (`lle.rs:162`) — the weights (REQ-2).** For each point
`i` with `k` neighbors: `Z[j][f] = x[i][f] − x[neighbor_j][f]` (`lle.rs:174-180`),
`C = Z Zᵀ` (`lle.rs:183`), `trace = Σ_j C_jj` (`lle.rs:186`), `reg_val =
reg·trace` when `trace > 0` else `reg` (`lle.rs:187`) — matching sklearn's
`R = reg*trace if trace>0 else reg` (`_locally_linear.py:73-76`, NO /k), add
`reg_val` to each diagonal (`lle.rs:190-192`), solve `C w = 1` by Gaussian
elimination with partial pivoting (`lle.rs:194-249`; sklearn `solve(G, ones,
assume_a="pos")` `:78`), normalize `w /= Σw` (`lle.rs:252-257`; sklearn
`w / np.sum(w)` `:79`), scatter into the dense `(n,n)` `W` (`lle.rs:260-262`).

**The regularization parity with sklearn (REQ-1/REQ-2, was DIV `#1460`, now
RESOLVED).** sklearn's `barycenter_weights` (`_locally_linear.py:29-80`) computes
`R = reg * trace if trace > 0 else reg` (`:73-76`, no division by `k`), adds it to
each diagonal of `G` (`:77`), and solves `G w = 1`. ferrolearn's `compute_weights`
now uses the identical `reg_val = reg * trace if trace > 0 else reg` (`lle.rs:187`)
— the `#1460` fix removed the previous `/ k`. With the matched regularizer, `W`
matches `barycenter_weights`, `M = (I−W)ᵀ(I−W)` matches, and the
bottom-`n_components` eigenvectors of `M` match sklearn's `null_space` selection
up to the **per-component sign** — an inherent carve-out (REQ-7), because sklearn
applies no deterministic sign flip. So the parity target is the sign-robust
`|embedding|` (REQ-1), verified element-wise (sign-aligned) to tol `1e-6` across
six fixtures in `tests/divergence_lle.rs`.

**sklearn (target contract).** `class LocallyLinearEmbedding` (`_locally_linear.py:603`)
takes `__init__(n_neighbors=5, n_components=2, reg=1e-3, eigen_solver="auto",
tol=1e-6, max_iter=100, method="standard", hessian_tol=1e-4, modified_tol=1e-12,
neighbors_algorithm="auto", random_state=None, n_jobs=None)` (`:756-783`).
`_fit_transform` (`:785`) fits `nbrs_ = NearestNeighbors(n_neighbors+1)` then
`embedding_, reconstruction_error_ = _locally_linear_embedding(...)` (`:795`).
For `method='standard'` (`:234-246`) it builds `W = barycenter_kneighbors_graph`
(barycenter weights, `:235`) and `M = (I−W)ᵀ(I−W)`, then `null_space(M,
n_components, k_skip=1)` (`:125-196`) returns the eigenvectors of `M` skipping
the trivial bottom one. `transform` (`:851-`) embeds new points via
`barycenter_weights` against `nbrs_`.

**The remaining gap.** ferrolearn ships standard-LLE `|embedding|` value parity
(REQ-1) and the reconstruction weights `W` (REQ-2) — the `#1460` reg fix landed —
plus the `M = (I−W)ᵀ(I−W)` construction + bottom-eigenvector selection (REQ-3),
the embedding shape/determinism (REQ-4), and the scoped error contracts (REQ-5).
It lacks: the non-standard `method` arms (REQ-6);
`eigen_solver='arpack'`/`random_state`/sign (REQ-7, sign-carve-out); `transform`
(REQ-8); the `reconstruction_error_`/`nbrs_` attrs +
`neighbors_algorithm`/`tol`/`max_iter` surface (REQ-9); the PyO3 binding (REQ-10);
and the ferray substrate (REQ-11). This is a **standard-LLE-value-parity-SHIPPED**
unit — the LLE pipeline ships with sklearn-matching weights/embedding; the
remaining work is non-standard methods, the arpack solver, out-of-sample
transform, the attribute surface, the binding, and the substrate (5 SHIPPED / 6
NOT-STARTED).

## Verification

Library crate (green at baseline `91124e74`):
```bash
cargo test -p ferrolearn-decomp lle                          # 14/14 in-module + api_proof_lle
cargo test -p ferrolearn-decomp --test divergence_lle        # parity + green-guards
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-3..5 (STRUCTURAL): `test_lle_basic_shape` `(9,2)`,
`test_lle_1d` (ncols 1), `test_lle_preserves_local_structure` (1D line embedding
monotonic), `test_lle_larger_dataset` `(20,2)`, `test_lle_different_n_neighbors`
(REQ-3/4 shape + determinism); `test_lle_invalid_n_components_zero`,
`test_lle_invalid_n_neighbors_zero`, `test_lle_n_neighbors_too_large`,
`test_lle_n_components_too_large`, `test_lle_insufficient_samples`,
`test_lle_negative_reg` (REQ-5); plus the module doctest. The
`tests/divergence_lle.rs` suite pins REQ-1/REQ-2 (VALUE/WEIGHT parity) and
re-pins REQ-4/REQ-5 (green-guards).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-1 value parity,
the REQ-2 weight parity, and the REQ-9 surface:
```bash
# REQ-1 (|embedding| parity; ferrolearn matches in magnitude with the matched reg formula):
python3 -c "import numpy as np; from sklearn.manifold import LocallyLinearEmbedding
X=np.array([[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]])
print(np.round(LocallyLinearEmbedding(n_components=2,n_neighbors=4,reg=1e-3,method='standard',eigen_solver='dense').fit(X).embedding_,8).tolist())"
# -> [[0.24567158,0.56740396],...,[0.27998483,-0.46002551]]  (|.| target; sign solver-arbitrary, REQ-7)

# REQ-2 (weight parity: R=reg*trace, NO /k, on both sklearn and ferrolearn):
python3 -c "import numpy as np; from scipy.linalg import solve
X=np.array([[0,0,0],[1,0.1,0],[2,0.3,0.1],[3,0.2,0],[0.5,1,0.2],[1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2,0],[1.2,2.1,0.1]])
d=((X-X[0])**2).sum(1); ind=np.argsort(d)[1:5]; C=X[ind]-X[0]; G=C@C.T; tr=np.trace(G); k=4; v=np.ones(k)
Gs=G.copy(); Gs.flat[::k+1]+=1e-3*tr; ws=solve(Gs,v,assume_a='pos'); print('sklearn', np.round(ws/ws.sum(),8).tolist())"
# -> sklearn [1.21836952, 0.59929776, -1.02925244, 0.21158516]  (ferrolearn compute_weights now matches)
```
The value pin is `LLE::new(2).with_n_neighbors(4).fit(&X).unwrap().embedding()`
matching the Probe REQ-1 target in ABSOLUTE VALUE to < 1e-6 (sign-robust per
column), pinned by `divergence_standard_lle_embedding_magnitude` and the
`parity_*` probes in `tests/divergence_lle.rs`.

ferrolearn-python (REQ-10 binding parity, after `#1465` lands):
```bash
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_lle.py -q
```
asserting `ferrolearn.LocallyLinearEmbedding` exists and exposes `embedding_` /
`reconstruction_error_` / `transform`, with its `|embedding|` validated against
`sklearn.manifold.LocallyLinearEmbedding` (sign-robust per column).

## Blockers

(#1459 is this doc's crosslink tracking issue. #1460 (the reg `/k` fix for REQ-1
+ REQ-2) is RESOLVED — the formula now matches sklearn `barycenter_weights`. The
blockers below are the remaining open work items; none are filed by this doc —
markdown only.)

- **#1460 (RESOLVED)** — REQ-1 + REQ-2: the local-covariance regularizer in
  `compute_weights` changed from `reg_val = reg * trace / k` to `reg_val = reg *
  trace if trace > 0 else reg` (`lle.rs:187`) to match `barycenter_weights`
  `R = reg * trace` (`_locally_linear.py:73-76`, NO `/k`). After this, `W` matches
  `barycenter_weights`, `M` matches, and the bottom-`n_components` eigenvectors of
  `M` match sklearn's `null_space` selection in ABSOLUTE VALUE (~1e-8) across six
  fixtures; the per-component sign stays solver-arbitrary (REQ-7). REQ-1 / REQ-2
  are SHIPPED.
- **#1461** — REQ-6: implement the `hessian` / `modified` (MLLE) / `ltsa`
  `method` arms (each builds a different `M`, `_locally_linear.py:248-`), gated by
  a `method` parameter on `LLE`.
- **#1462** — REQ-7: add `eigen_solver='arpack'` (shift-invert `eigsh(M,
  k+k_skip, sigma=0.0)`, `_locally_linear.py:173-188`) + `random_state`. The
  per-component SIGN of each eigenvector is an inherent carve-out (sklearn applies
  no deterministic flip) — the parity target is the sign-robust `|embedding|`.
- **#1463** — REQ-8: add out-of-sample `transform` embedding new points via
  `barycenter_weights` against the fitted neighbors (`_locally_linear.py:851-`).
- **#1464** — REQ-9: expose `reconstruction_error_` (the `squared_error`,
  `_locally_linear.py:554/795`) and `nbrs_`, plus `neighbors_algorithm` / `tol` /
  `max_iter` ctor fields.
- **#1465** — REQ-10: add `_RsLocallyLinearEmbedding` to `ferrolearn-python`
  (fit / transform / embedding_ / reconstruction_error_) — the boundary CPython
  consumer.
- **#1466** — REQ-11: migrate `lle.rs` off `ndarray` / faer
  (`crate::mds::eigh_faer`, `lle.rs:36/354`) to `ferray-core` / `ferray::linalg`
  (R-SUBSTRATE).
