# Spectral Embedding (sklearn.manifold.SpectralEmbedding)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: dc2c13c3
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/manifold/_spectral_embedding.py  # def spectral_embedding / _spectral_embedding (:167 wrapper, :300-469); csgraph_laplacian(normed=True, return_diag=True) -> laplacian, dd (:333-335); _set_diag(laplacian, 1) (:347); dense eigh path eigh(laplacian) + diffusion_map.T[:n_components] (:439-440); embedding = embedding / dd (:443, dd = sqrt(degree)); _deterministic_vector_sign_flip(embedding) (:465); drop_first slice embedding[1:n_components].T (:466-467); class SpectralEmbedding(BaseEstimator) (:472); _parameter_constraints affinity/gamma/random_state/eigen_solver/n_neighbors (:139-156); __init__(n_components=2, *, affinity="nearest_neighbors", gamma=None, random_state=None, eigen_solver=None, n_neighbors=None, n_jobs=None) (:159-178); _get_affinity_matrix (:189/:660): precomputed (:680), precomputed_nearest_neighbors 0.5*(A+A.T) (:682-688), nearest_neighbors n_neighbors_ default max(n/10,1) + kneighbors_graph(X, n_neighbors_, include_self=True) + 0.5*(A+A.T) (:689-710), rbf gamma_ default 1/n_features + rbf_kernel(X, gamma=gamma_) (:711-714), callable (:715)
  - sklearn/utils/extmath.py                  # def _deterministic_vector_sign_flip(u): per ROW, flip sign so the max-|entry| is positive
ferrolearn-module: ferrolearn-decomp/src/spectral_embedding.rs
parity-ops: SpectralEmbedding
crosslink-issue: 1443
-->

## Summary

`ferrolearn-decomp/src/spectral_embedding.rs` mirrors scikit-learn's
`SpectralEmbedding` (`sklearn/manifold/_spectral_embedding.py`, `class
SpectralEmbedding(BaseEstimator)` `:472`) and the underlying `spectral_embedding`
function (`:167` wrapper / `_spectral_embedding` `:300-469`) — Laplacian
Eigenmaps non-linear dimensionality reduction. It exposes the unfitted
`SpectralEmbedding { n_components, affinity: Affinity }` with `Affinity::{RBF {
gamma }, NearestNeighbors { n_neighbors } }` (`new(n_components)` defaulting to
`RBF { gamma: 1.0 }`, builder `with_affinity`, accessors `n_components()` /
`affinity()`), and the fitted `FittedSpectralEmbedding { embedding_ }` (accessor
`embedding()`). It is re-exported at the crate root (`pub use
spectral_embedding::{Affinity, FittedSpectralEmbedding, SpectralEmbedding}` in
`ferrolearn-decomp/src/lib.rs`).

**This unit ships the spectral-embedding SHAPE AND its RBF VALUE.** ferrolearn's
`fit` (`fn fit` on `impl Fit<Array2<f64>, ()> for SpectralEmbedding`) builds the
affinity `W` (`fn build_affinity_matrix`), forms the normalized symmetric
Laplacian `L_sym = I − D^{-1/2} W D^{-1/2}` (`fn normalised_laplacian`),
dense-eigendecomposes via `eigh_faer` (`crate::mds::eigh_faer`), sorts
eigenvalues ascending, skips the trivial first eigenvector, copies the next
`n_components` eigenvectors of `L_sym`, **rescales each by `1/dd`** (`dd =
sqrt(degree)`, `degree[i] = Σ_{j≠i} W[i,j]` the off-diagonal row-sum — matching
scipy `csgraph_laplacian(normed=True)`, which IGNORES self-loops, `_spectral_embedding.py:443`),
and applies a **per-column deterministic sign-flip** (`:465`). These two steps
are the value-affecting ones sklearn performs after the dense `eigh`:

| step | sklearn (`_spectral_embedding`) | ferrolearn (`spectral_embedding.rs`) |
|---|---|---|
| Laplacian | `csgraph_laplacian(W, normed=True, return_diag=True)` → `L = I − D^{-1/2}WD^{-1/2}` + `dd = sqrt(degree)` (`:333-335`); degree from the off-diagonal row-sum (self-loops ignored) | `fn normalised_laplacian`: `L_sym = I − D^{-1/2}WD^{-1/2}` — **MATCHES** (REQ-3) |
| eigsolve | dense `eigh(L)` (`:439`), drop trivial, take next `n_components` | `eigh_faer(&L)` ascending, skip first, take next `n_components` — same subspace |
| **/dd rescale** | `if norm_laplacian: embedding = embedding / dd` (`:443`, `dd = sqrt(degree)`) — recovers `u = D^{-1/2}x` | `embedding[[i,k]] = v / dd[i]` (`dd[i] = sqrt(Σ_{j≠i} W[i,j])`, guarded `dd>1e-15`) — **MATCHES** (REQ-1) |
| **sign flip** | `embedding = _deterministic_vector_sign_flip(embedding)` (`:465`) — each column's max-\|entry\| made positive | per-column negate-if-max-abs-entry-is-negative loop — **MATCHES** (REQ-1) |

**Headline finding — the RBF embedding VALUE matches sklearn element-wise** on a
CONNECTED, DISTINCT-eigenvalue, ASYMMETRIC fixture
(`[[0,0],[1.2,0.3],[2.0,1.1],[3.5,0.2],[4.1,2.0]]`, gamma=0.3) for both
`n_components=1` and `n_components=2` (tol 1e-6) — the `L_sym` is shared (REQ-3),
the dense `eigh` recovers the same eigenspace, the `1/dd` rescale recovers `u =
D^{-1/2}x`, and the sign-flip pins the per-column sign (REQ-1). This was the
core value-parity divergence DIV/#1444, now **RESOLVED** (#1444 closed). The fix
keeps the RBF affinity diagonal at `0` (an intermediate that set the diagonal to
`1.0`, matching `rbf_kernel`, was WRONG for the embedding: scipy's
`csgraph_laplacian(normed=True)` ignores self-loops, so degree = off-diagonal
row-sum — diagonal `0` is what reproduces sklearn's `dd`/Laplacian/embedding).

Two carve-outs are explicitly out of scope for REQ-1 value parity (tracked under
REQ-7, blocker **#1446**): (a) the **DEGENERATE eigenvalue** case — a
near-disconnected RBF graph has a repeated `0`-eigenvalue, and faer/arpack pick
different orthonormal null-space bases, so element-wise parity is impossible (no
committed failing test per R-DEFER-3); (b) the **SYMMETRIC-fixture sign-flip ULP
tie** — an antisymmetric eigenvector with `|v[0]| == |v[4]|` is a ULP-level tie
for the sign-flip argmax (faer/numpy can round it oppositely → opposite global
sign), avoided here by using the ASYMMETRIC fixture.

At baseline `dc2c13c3` REQ-1 is **SHIPPED** (the `1/dd` rescale and the sign-flip
are present and element-wise-pinned). The structural shape (REQ-2), the `L_sym`
construction (REQ-3), the RBF off-diagonal affinity (REQ-4), and the scoped error
contracts (REQ-5) are also SHIPPED — 5 SHIPPED / 6 NOT-STARTED (REQ-6..11).

`SpectralEmbedding` / `FittedSpectralEmbedding` / `Affinity` are existing pub
APIs whose only non-test consumer is the crate re-export (the boundary public
API, grandfathered S5/R-DEFER-1). There is **no PyO3 binding** (`grep -rln
SpectralEmbedding ferrolearn-python/` is empty) and **no `Transform`/out-of-sample
projection** (`SpectralEmbedding` is fit-only, mirroring sklearn, which also has
no `transform`).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# REQ-1 (HEADLINE, now SHIPPED) — RBF SpectralEmbedding embedding VALUES (the /dd rescale + sign-flip).
# Fixture: two well-separated triangles (3 pts near origin, 3 near (5,5)), distinct eigenvalues.
python3 -c "import numpy as np; from sklearn.manifold import SpectralEmbedding; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1]]); \
se=SpectralEmbedding(n_components=2, affinity='rbf', gamma=1.0, eigen_solver='arpack', random_state=0); \
print(np.round(se.fit_transform(X),8).tolist())"
# -> [[-0.12247851, -0.0],       [-0.12247851,  0.17923617], [-0.12247851, -0.17923617],
#     [ 0.39230206, -0.0],       [ 0.39230206, -0.47079665], [ 0.39230206,  0.47079665]]
#    The first column SEPARATES the two clusters (-0.122 vs +0.392). The row magnitudes are
#    ~1/dd (dd = sqrt(off-diagonal-row-sum degree) ~ 1.4, see below). ferrolearn now applies
#    BOTH the /dd rescale (:443) and the deterministic sign-flip (:465) — this fixture is the
#    DEGENERATE two-cluster carve-out (repeated 0-eigenvalue, REQ-7/#1446); REQ-1 value parity
#    is pinned on the asymmetric `line5` fixture instead (see green_rbf_embedding_value_parity_*).

# /dd evidence — dd = sqrt(off-diagonal-row-sum degree) is the per-NODE rescale sklearn applies (embedding/dd, :443):
python3 -c "import numpy as np; from sklearn.metrics.pairwise import rbf_kernel; \
from scipy.sparse.csgraph import laplacian as csgraph_laplacian; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1]]); \
A=rbf_kernel(X,gamma=1.0); lap,dd=csgraph_laplacian(A, normed=True, return_diag=True); \
print('dd=sqrt(degree)=', np.round(dd,6).tolist())"
# -> dd=sqrt(degree)= [1.40716, 1.403655, 1.403655, 1.40716, 1.403655, 1.403655]
#    sklearn divides each ROW i of the (transposed) eigenvector block by dd[i] (:443). ferrolearn
#    `fn fit` now applies the same dd[i] = sqrt(off-diagonal row-sum) rescale (REQ-1, SHIPPED).

# sign-flip evidence — _deterministic_vector_sign_flip flips each vector so its max-|entry| is positive:
python3 -c "import numpy as np; from sklearn.utils.extmath import _deterministic_vector_sign_flip; \
u=np.array([[1.0,-3.0,2.0],[-5.0,1.0,4.0]]); \
print('in ', u.tolist()); print('out', _deterministic_vector_sign_flip(u).tolist())"
# -> in  [[1.0, -3.0, 2.0], [-5.0, 1.0, 4.0]]
# -> out [[-1.0, 3.0, -2.0], [5.0, -1.0, -4.0]]
#    row 0 max-|entry| was -3 (col 1) -> whole row negated so it becomes +3; row 1 max-|entry| was
#    -5 (col 0) -> negated so it becomes +5. ferrolearn copies eigenvectors RAW -> sign is whatever
#    faer's eigh returns (arbitrary, basis-dependent), so columns can be sign-flipped vs sklearn.

# REQ-3 — normalized symmetric Laplacian: csgraph_laplacian(normed=True) = I - D^{-1/2}WD^{-1/2},
# with _set_diag(L,1) so the diagonal is 1 where degree>0 (matches ferrolearn fn normalised_laplacian):
python3 -c "import numpy as np; from sklearn.metrics.pairwise import rbf_kernel; \
from scipy.sparse.csgraph import laplacian as csgraph_laplacian; \
X=np.array([[0.,0.],[1.,0.],[5.,5.]]); A=rbf_kernel(X,gamma=1.0); \
L,_=csgraph_laplacian(A, normed=True, return_diag=True); print(np.round(L,6).tolist())"
# -> [[1.0, -0.49793..., -1.6e-11], [-0.49793..., 1.0, -1.7e-11], [-1.6e-11, -1.7e-11, 1.0]]
#    L_ij = -W_ij/sqrt(deg_i*deg_j) off-diagonal, 1 on the diagonal (degree>0). ferrolearn
#    fn normalised_laplacian computes EXACTLY this (l[[i,j]] = 1 - d_inv_sqrt[i]*w*d_inv_sqrt[j]
#    on the diagonal, -d_inv_sqrt[i]*w*d_inv_sqrt[j] off). NOTE: sklearn's rbf_kernel W has
#    diagonal=1.0 (W_ii=1), ferrolearn forces W_ii=0; this perturbs the degree -> see REQ-4/REQ-3 flag.

# REQ-4 — RBF affinity: sklearn rbf_kernel(X, gamma) has diagonal = 1.0; ferrolearn forces W_ii = 0.
python3 -c "import numpy as np; from sklearn.metrics.pairwise import rbf_kernel; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1]]); A=rbf_kernel(X,gamma=1.0); \
print('diag', np.diag(A).tolist(), 'off A[0,1]=', A[0,1])"
# -> diag [1.0, 1.0, 1.0] off A[0,1]= 0.9900498337491681
#    OFF-DIAGONAL exp(-gamma*||xi-xj||^2) MATCHES ferrolearn (fn build_affinity_matrix RBF arm);
#    the DIAGONAL is sklearn rbf_kernel 1.0 vs ferrolearn 0.0; for the EMBEDDING Laplacian the
#    0 diagonal is CORRECT (scipy csgraph_laplacian ignores self-loops). The diagonal-1.0
#    affinity_matrix_ attribute is the separate REQ-9 (NOT-STARTED).

# REQ-8 — DEFAULT affinity is 'nearest_neighbors' (NOT rbf), gamma default = 1/n_features (not 1.0):
python3 -c "from sklearn.manifold import SpectralEmbedding; \
se=SpectralEmbedding(); print('default affinity =', se.affinity, '| default gamma =', se.gamma)"
# -> default affinity = nearest_neighbors | default gamma = None  (gamma=None -> 1/n_features at fit, :712)
#    ferrolearn SpectralEmbedding::new defaults to RBF { gamma: 1.0 } -> different default affinity AND
#    a fixed gamma=1.0 instead of 1/n_features. DIV (REQ-8).

# REQ-6 — kNN affinity GRAPH: sklearn kneighbors_graph(X, k, include_self=True) then 0.5*(A+A.T) (:704-709):
python3 -c "import numpy as np; from sklearn.neighbors import kneighbors_graph; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1]]); \
A=kneighbors_graph(X, 2, include_self=True).toarray(); S=0.5*(A+A.T); print(np.round(S,3).tolist())"
# -> symmetric 0/0.5/1 connectivity WITH self-loops (include_self=True) symmetrized.
#    ferrolearn fn build_affinity_matrix NearestNeighbors arm uses its OWN k-NN graph: NO self-loop,
#    OR-symmetrize (w[i,j]=w[j,i]=1 if j in i's kNN), binary 0/1 -> different graph. DIV (REQ-6).
```

## Requirements

- REQ-1: **RBF embedding VALUE parity on distinct eigenvalues (R-DEV-1, the core
  requirement; SHIPPED, was DIV/#1444 — now RESOLVED).** Mirror
  `spectral_embedding(adjacency, n_components, norm_laplacian=True)`
  (`_spectral_embedding.py:300-469`) for the dense `eigh` path: build `L = I −
  D^{-1/2}WD^{-1/2}` + `dd = sqrt(degree)` (`:333-335`), dense `eigh` (`:439`),
  `embedding = diffusion_map.T[:n_components]` dropping the trivial vector,
  **`embedding = embedding / dd`** (`:443`), then **`embedding =
  _deterministic_vector_sign_flip(embedding)`** (`:465`). ferrolearn `fn fit` now
  takes the next `n_components` eigenvectors of `L_sym`, rescales each row by
  `1/dd` (`dd[i] = sqrt(Σ_{j≠i} W[i,j])`, the degree from the off-diagonal
  row-sum — scipy's `csgraph_laplacian(normed=True)` ignores self-loops, `:443`),
  and applies the per-column deterministic sign-flip (`:465`). The RBF affinity
  diagonal stays at `0` (NOT `1.0`): with `W_ii=0` the degree/`dd`/Laplacian match
  scipy exactly. On the CONNECTED, DISTINCT-eigenvalue, ASYMMETRIC fixture
  `[[0,0],[1.2,0.3],[2.0,1.1],[3.5,0.2],[4.1,2.0]]` (gamma=0.3) the embedding
  matches sklearn element-wise (tol 1e-6) for `n_components` 1 and 2 — pinned by
  `green_rbf_embedding_value_parity_1`/`_2` in
  `tests/divergence_spectral_embedding.rs`. CARVE-OUTS (REQ-7, #1446): the
  degenerate-eigenvalue (repeated-`0`) null-space-basis case and the
  symmetric-fixture sign-flip ULP tie are out of REQ-1 scope.

- REQ-2: **Structural embedding SHAPE (scoped; SHIPPED).** `fit` returns
  `FittedSpectralEmbedding` whose `embedding()` is `(n_samples, n_components)`,
  separates well-separated clusters along the first non-trivial coordinate, and is
  deterministic given the input (no RNG in the dense path). These are
  scale/sign-invariant structural properties that hold independently of the value
  parity (REQ-1). Mirrors the dense-`eigh` SHAPE of `_spectral_embedding`
  (`:439-440`, `embedding[:n_components].T`).

- REQ-3: **Normalized symmetric Laplacian `L_sym` construction (R-DEV-1;
  SHIPPED).** `fn normalised_laplacian` builds `L_sym = I − D^{-1/2}WD^{-1/2}`
  (`:201-210`), with the degree guard `deg ≤ 1e-15 → d_inv_sqrt = 0`, matching
  `csgraph_laplacian(W, normed=True)` (`_spectral_embedding.py:333`) followed by
  `_set_diag(L, 1)` (`:347`) — diagonal `1` where degree `> 0`, off-diagonal
  `−W_ij/sqrt(deg_i·deg_j)` (Probe REQ-3). The degree is the off-diagonal
  row-sum: scipy's `csgraph_laplacian(W, normed=True)` IGNORES the matrix diagonal
  (treats `W` as a graph adjacency, no self-loops), so ferrolearn's `W_ii = 0`
  (REQ-4) is exactly what reproduces scipy's `D` / `dd` / `L_sym` — the embedding
  for which the diagonal MUST be `0` (an intermediate `W_ii=1.0` was wrong; see
  REQ-1, REQ-4). The separate `affinity_matrix_` attribute (which sklearn exposes
  with diagonal `1.0` from `rbf_kernel`) is REQ-9, NOT-STARTED.

- REQ-4: **RBF affinity off-diagonal `exp(−γ·||·||²)` (R-DEV-1; SHIPPED).** `fn
  build_affinity_matrix` RBF arm computes `W_ij = exp(−gamma·||x_i − x_j||²)` for
  `i ≠ j`, matching `rbf_kernel(X, gamma=gamma)` (`_spectral_embedding.py:713`) on
  the OFF-DIAGONAL to f64 precision (Probe REQ-4: `A[0,1] = 0.9900498337491681`).
  The diagonal is correctly `W_ii = 0`: for the EMBEDDING Laplacian, scipy's
  `csgraph_laplacian(normed=True)` ignores self-loops (degree = off-diagonal
  row-sum), so `W_ii=0` is what makes the degree/`dd`/embedding match sklearn
  (REQ-1). (sklearn's `rbf_kernel` DIAGONAL `1.0` only surfaces in the separate
  `affinity_matrix_` fitted attribute, which is REQ-9, NOT-STARTED.) **FLAG (still
  divergent):** sklearn's default `gamma = 1.0 / n_features` (`:712`, `gamma =
  None`), ferrolearn defaults `gamma = 1.0` (REQ-8).

- REQ-5: **Error / parameter contracts (scoped; SHIPPED).** `fn fit` returns
  `InvalidParameter` when `n_components == 0` or `n_components >= n_samples`,
  `InsufficientSamples` when `n_samples < 2`, `InvalidParameter` for kNN
  `n_neighbors == 0` or `n_neighbors >= n_samples`, and `InvalidParameter` for RBF
  `gamma <= 0`. **FLAG (candidate DIVs):** sklearn's `_parameter_constraints`
  (`:139-156`) admit `gamma` `Interval(Real, 0, None, closed="left")` so `gamma ==
  0` is VALID in sklearn (ferrolearn rejects `<= 0`), `n_neighbors` `Interval(
  Integral, 1, None)`, and sklearn raises `InvalidParameterError`/`ValueError`,
  NOT the `FerroError::InvalidParameter` ABI.

- REQ-6: **kNN affinity GRAPH parity (R-DEV-1; NOT-STARTED).** sklearn builds the
  nearest-neighbors affinity via `kneighbors_graph(X, n_neighbors_,
  include_self=True)` then symmetrizes `0.5 * (A + A.T)` (`:703-709`). ferrolearn's
  `fn build_affinity_matrix` NearestNeighbors arm uses its OWN k-NN construction
  (`:157-190`): pairwise squared distances, per-point top-`k` nearest (excluding
  self), then OR-symmetrize `w[i,j] = w[j,i] = 1` — a binary 0/1 graph with NO
  self-loops and OR (not averaged) symmetrization (Probe REQ-6). Different affinity
  matrix → different `L_sym` → different embedding.

- REQ-7: **`eigen_solver` (arpack/lobpcg/amg) + `random_state` + degenerate-
  eigenvalue subspace non-uniqueness + symmetric-fixture sign-tie (R-DEV-2;
  NOT-STARTED, CARVE-OUTS; blocker #1446).** sklearn exposes `eigen_solver ∈
  {'arpack','lobpcg','amg'}` (`:153`) with `random_state` seeding the iterative
  solvers (`:152`); ferrolearn has no `eigen_solver`/`random_state` parameter and
  always uses the dense `eigh_faer` path. TWO carve-outs are excluded from REQ-1
  value parity: (a) for REPEATED eigenvalues (a near-disconnected graph's repeated
  `0`-eigenvalue) the degenerate-subspace basis is rotation-non-unique, so
  faer/arpack pick different null-space bases and element-wise parity is impossible
  (no committed failing test, R-DEFER-3); (b) a SYMMETRIC fixture whose
  antisymmetric eigenvector has `|v[0]| == |v[4]|` is a ULP-level sign-flip argmax
  tie (faer/numpy can round it oppositely → opposite global sign), avoided in the
  REQ-1 pin by using the asymmetric `line5` fixture.

- REQ-8: **Default `affinity='nearest_neighbors'` + `gamma=None → 1/n_features` +
  precomputed/callable affinity (R-DEV-2; NOT-STARTED).** sklearn `__init__`
  defaults `affinity="nearest_neighbors"` (`:163`) and `gamma=None` resolved to
  `1.0 / n_features` at fit (`:712`); it also accepts `affinity ∈
  {'precomputed','precomputed_nearest_neighbors'}` and a callable (`:680-715`).
  ferrolearn `SpectralEmbedding::new` defaults to `RBF { gamma: 1.0 }` (different
  default affinity AND a fixed gamma instead of `1/n_features`), and has no
  precomputed/callable affinity (Probe REQ-8).

- REQ-9: **`affinity_matrix_` / `n_neighbors_` fitted attrs + out-of-sample
  (R-DEV-3; NOT-STARTED).** sklearn exposes the fitted attributes `embedding_`,
  `affinity_matrix_` (`:89`), and `n_neighbors_` (`:103`, set to `max(n/10, 1)`
  when `n_neighbors=None`, `:698-702`). `FittedSpectralEmbedding` exposes only
  `embedding()`; there is no `affinity_matrix_` accessor and no `n_neighbors_`.
  (Both sklearn and ferrolearn are fit-only — neither has `transform`/out-of-sample
  projection — so the out-of-sample part is moot, but the fitted-attr surface
  diverges.)

- REQ-10: **PyO3 binding (R-DEFER-1/3; NOT-STARTED).** `import ferrolearn`
  exposing a registered `SpectralEmbedding` marshalling `fit` and exposing
  `embedding_` — the boundary CPython consumer. Absent (`grep -rln
  SpectralEmbedding ferrolearn-python/` is empty).

- REQ-11: **ferray substrate (R-SUBSTRATE; NOT-STARTED).** `spectral_embedding.rs`
  computes on `ndarray::Array2<f64>` + `Vec<f64>` and decomposes via
  `crate::mds::eigh_faer` (faer), not `ferray-core` arrays / `ferray::linalg`
  (the eigendecomposition) (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). Fixture `tri` = the two-triangle
fixture `[[0,0],[0.1,0],[0,0.1],[5,5],[5.1,5],[5,5.1]]` (distinct eigenvalues).

- AC-1 (REQ-1, SHIPPED — element-wise parity): on the CONNECTED,
  DISTINCT-eigenvalue, ASYMMETRIC fixture `line5 =
  [[0,0],[1.2,0.3],[2.0,1.1],[3.5,0.2],[4.1,2.0]]` with `affinity='rbf',
  gamma=0.3`, sklearn `fit_transform` (eigen_solver='arpack', random_state=0) for
  `n_components=2` →
  `[[-0.5492950,0.5911304],[-0.3247844,-0.0471046],[0.0371155,-0.5640169],
  [0.5056203,0.0536275],[0.7457959,0.6045965]]` and for `n_components=1` →
  `[-0.5492950,-0.3247844,0.0371155,0.5056203,0.7457959]`. ferrolearn's
  `embedding()` matches both element-wise to tol 1e-6 (observed ~1e-15) after the
  `1/dd` rescale (`:443`) + per-column `_deterministic_vector_sign_flip` (`:465`),
  with the RBF diagonal at `0`. Pinned by `green_rbf_embedding_value_parity_2` /
  `_1` in `tests/divergence_spectral_embedding.rs`. CARVE-OUTS (REQ-7, #1446):
  degenerate-eigenvalue null-space basis and symmetric-fixture sign ULP tie are
  out of scope (the asymmetric fixture avoids the latter; the former has no
  committed test per R-DEFER-3).

- AC-2 (REQ-2, SHIPPED): `SpectralEmbedding::new(2).fit(&tri).unwrap().embedding()`
  has shape `(6, 2)` and its first column separates rows 0-2 from rows 3-5 (the two
  triangles); the result is deterministic across runs (no RNG). Pinned by
  `test_spectral_embedding_basic_shape`, `test_spectral_embedding_1d`,
  `test_spectral_embedding_rbf_separates_clusters`,
  `test_spectral_embedding_knn_affinity`, `test_spectral_embedding_larger_dataset`.

- AC-3 (REQ-3, SHIPPED): `fn normalised_laplacian(W)` returns `L_sym` with `L_ii =
  1` (degree `> 0`) and `L_ij = −W_ij/sqrt(deg_i·deg_j)`, matching
  `csgraph_laplacian(W, normed=True)` + `_set_diag(L,1)` (`:333`,`:347`; Probe
  REQ-3). Exercised indirectly by every `fit` test (the embedding is derived from
  it).

- AC-4 (REQ-4, SHIPPED off-diagonal): `fn build_affinity_matrix` RBF arm yields
  `W[0,1] = exp(−1·0.01) = 0.9900498337491681`, matching `rbf_kernel(X,
  gamma=1.0)[0,1]` (Probe REQ-4). Pinned by
  `test_spectral_embedding_rbf_separates_clusters`. The diagonal is `W_ii = 0`
  (correct for the embedding Laplacian — scipy ignores self-loops); the
  diagonal-`1.0` `affinity_matrix_` attribute is the separate REQ-9 (NOT-STARTED).
  FLAG (still divergent): sklearn default `gamma = 1/n_features`, ferrolearn
  `1.0` (REQ-8).

- AC-5 (REQ-5, SHIPPED scoped): `fit` returns `Err` for `n_components=0`,
  `n_components >= n_samples`, `n_samples < 2`, kNN `n_neighbors=0` /
  `n_neighbors >= n_samples`, RBF `gamma <= 0`. Pinned by
  `test_spectral_embedding_invalid_n_components_zero`,
  `test_spectral_embedding_n_components_too_large`,
  `test_spectral_embedding_insufficient_samples`,
  `test_spectral_embedding_knn_n_neighbors_zero`,
  `test_spectral_embedding_knn_too_many_neighbors`,
  `test_spectral_embedding_negative_gamma`. FLAG: sklearn accepts `gamma=0` and
  uses `InvalidParameterError`/`ValueError`, not `FerroError::InvalidParameter`.

- AC-6 (REQ-6, DIVERGES): on `tri` with `n_neighbors=2`, sklearn
  `kneighbors_graph(X, 2, include_self=True)` symmetrized `0.5*(A+A.T)` (Probe
  REQ-6) differs from ferrolearn's binary OR-symmetrized no-self-loop kNN graph.

- AC-7 (REQ-8, DIVERGES): `SpectralEmbedding().affinity == 'nearest_neighbors'`
  and `.gamma is None` (resolved to `1/n_features`) in sklearn (Probe REQ-8);
  `SpectralEmbedding::new(2).affinity()` is `RBF { gamma: 1.0 }`.

- AC-8 (REQ-9/10/11): `hasattr(se, 'affinity_matrix_')` / `'n_neighbors_'` True in
  sklearn; ferrolearn `FittedSpectralEmbedding` exposes only `embedding()`. No
  `ferrolearn.SpectralEmbedding` (no binding). The module imports `ndarray` +
  `eigh_faer`, not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `SpectralEmbedding` / `FittedSpectralEmbedding` / `Affinity`
are existing pub APIs re-exported at the crate root (the only non-test consumer;
grandfathered S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2, run from
`/tmp`. Honest underclaim (R-HONEST-3): the load-bearing RBF embedding VALUE
(REQ-1) is now SHIPPED — element-wise pinned against the live oracle on a
CONNECTED, DISTINCT-eigenvalue, ASYMMETRIC fixture; the least-confident SHIPPED
claim, since parity is scoped to that distinct-eigenvalue/unique-argmax fixture
(the degenerate-eigenvalue and symmetric-fixture sign-tie cases are carved out to
REQ-7, #1446). #1443 is this doc's crosslink tracking issue; the core
value-parity divergence was #1444 — now RESOLVED. Count: 5 SHIPPED (REQ-1..5) /
6 NOT-STARTED (REQ-6..11).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (RBF embedding VALUE parity; `1/dd` rescale + sign-flip) | SHIPPED | `fn fit` takes the next `n_components` eigenvectors of `L_sym`, rescales each row by `1/dd` (`embedding[[i,k]] = if dd[i] > 1e-15 { v / dd[i] } else { v }`, `dd[i] = sqrt(Σ_{j≠i} W[i,j])` the off-diagonal-row-sum degree — scipy `csgraph_laplacian(normed=True)` ignores self-loops, `_spectral_embedding.py:443`), then applies the per-column deterministic sign-flip (negate the column if its max-\|entry\| is negative, `:465`). RBF diagonal stays `0` (a wrong intermediate set it to `1.0`; with `W_ii=0` the degree/`dd`/Laplacian match scipy exactly). Was DIV/#1444 — now RESOLVED. Non-test consumer: re-export `pub use spectral_embedding::{...}` (`lib.rs`, boundary public API, grandfathered S5/R-DEFER-1). Pin (AC-1, R-CHAR-3): `SpectralEmbedding(n_components=2, affinity='rbf', gamma=0.3, eigen_solver='arpack', random_state=0).fit_transform(line5)` (line5 = `[[0,0],[1.2,0.3],[2.0,1.1],[3.5,0.2],[4.1,2.0]]`) → `[[-0.5492950,0.5911304],...,[0.7457959,0.6045965]]`; ferrolearn matches element-wise to ~1e-15 (tol 1e-6) for both `n_components` 1 and 2. Verification: `cargo test -p ferrolearn-decomp --test divergence_spectral_embedding` → `green_rbf_embedding_value_parity_2` / `_1` PASS. CARVE-OUTS (REQ-7, #1446): (a) degenerate eigenvalue (near-disconnected graph → repeated `0`-eigenvalue → faer/arpack pick different null-space bases — no committed failing test, R-DEFER-3); (b) symmetric-fixture sign-flip ULP tie (antisymmetric eigenvector `|v[0]|==|v[4]|` → faer/numpy argmax can differ → opposite global sign; avoided by the asymmetric fixture). |
| REQ-2 (structural embedding SHAPE) | SHIPPED | `fn fit` returns `FittedSpectralEmbedding { embedding_ }` whose `embedding()` is `Array2<f64>` of shape `(n_samples, n_components)` (the `embedding = Array2::zeros((n, n_comp))` allocation + the skip-trivial / take-`n_components` copy), separating well-separated clusters along the first non-trivial coordinate, deterministic (no RNG in the dense path) — scale/sign-INVARIANT structural properties, mirroring the dense-`eigh` SHAPE of `_spectral_embedding` (`_spectral_embedding.py:439-440`). Non-test consumer: crate re-export `pub use spectral_embedding::{Affinity, FittedSpectralEmbedding, SpectralEmbedding}` (`ferrolearn-decomp/src/lib.rs`, boundary public API, grandfathered S5/R-DEFER-1). Verification: `cargo test -p ferrolearn-decomp spectral_embedding` (`test_spectral_embedding_basic_shape` `(8,2)`, `test_spectral_embedding_1d`, `test_spectral_embedding_rbf_separates_clusters` `(c1_mean−c2_mean).abs()>0.01`, `test_spectral_embedding_knn_affinity`, `test_spectral_embedding_larger_dataset`). **Scope (R-HONEST-3): this is the SHAPE/separation/determinism contract, NOT value parity (REQ-1).** |
| REQ-3 (normalized symmetric Laplacian `L_sym`) | SHIPPED | `fn normalised_laplacian` builds `L_sym = I − D^{-1/2}WD^{-1/2}` (diagonal `l[[i,i]] = 1 − d_inv_sqrt[i]·w[[i,i]]·d_inv_sqrt[i]`, off-diagonal `l[[i,j]] = −d_inv_sqrt[i]·w·d_inv_sqrt[j]`) with the degree guard `d_inv_sqrt[i] = if deg > 1e-15 { 1/deg.sqrt() } else { 0 }`, matching `csgraph_laplacian(W, normed=True)` (`_spectral_embedding.py:333`) + `_set_diag(L, 1)` (`:347`) — diagonal `1` where degree `>0`, off-diagonal `−W_ij/sqrt(deg_i·deg_j)` (Probe REQ-3). Non-test consumer: invoked by `fn fit` on every path, reached through the crate re-export. Verification: exercised by every `fit` test (the embedding derives from `L_sym`), and element-wise by `green_rbf_embedding_value_parity_2`/`_1`. The degree is the off-diagonal row-sum: scipy `csgraph_laplacian(normed=True)` IGNORES self-loops, so ferrolearn's `W_ii=0` (REQ-4) is exactly what reproduces scipy's `D`/`dd`/`L_sym` (REQ-1). The diagonal-`1.0` `affinity_matrix_` attribute is the separate REQ-9 (NOT-STARTED). |
| REQ-5 (error / parameter contracts, scoped) | SHIPPED | `fn fit` returns `Err(InvalidParameter { name: "n_components", reason: "must be at least 1" })` for `n_components == 0`, `Err(InvalidParameter { name: "n_components", .. })` for `n_components >= n`, `Err(InsufficientSamples { required: 2, actual: n, context: "SpectralEmbedding::fit requires at least 2 samples" })` for `n < 2`, `Err(InvalidParameter { name: "n_neighbors", .. })` for kNN `n_neighbors == 0` / `>= n`, and `Err(InvalidParameter { name: "gamma", reason: "must be positive" })` for RBF `gamma <= 0`. Non-test consumer: these guards protect every instance reached through the crate re-export (`lib.rs`). Verification: `cargo test -p ferrolearn-decomp spectral_embedding` (`test_spectral_embedding_invalid_n_components_zero`, `_n_components_too_large`, `_insufficient_samples`, `_knn_n_neighbors_zero`, `_knn_too_many_neighbors`, `_negative_gamma`) → green. **FLAG (candidate DIVs):** sklearn `_parameter_constraints` (`:139-156`) admit `gamma == 0` (`Interval(Real, 0, None, closed="left")`) — ferrolearn rejects `<= 0`; sklearn raises `InvalidParameterError`/`ValueError`, not `FerroError::InvalidParameter`. |
| REQ-4 (RBF affinity off-diagonal `exp(−γ·d²)`) | SHIPPED | `fn build_affinity_matrix` RBF arm computes `w[[i,j]] = w[[j,i]] = (-gamma * sq).exp()` for `i ≠ j` (sq = squared Euclidean distance), value-matching `rbf_kernel(X, gamma=gamma)` (`_spectral_embedding.py:713`) on the OFF-DIAGONAL to f64 precision (Probe REQ-4: `W[0,1] = exp(−1·0.01) = 0.9900498337491681`). Non-test consumer: invoked by `fn fit`, reached through the crate re-export. Verification: `test_spectral_embedding_rbf_separates_clusters` + the element-wise `green_rbf_embedding_value_parity_2`/`_1` (which exercise the full RBF→`L_sym`→embedding pipeline). The diagonal is correctly `W_ii = 0` ("Diagonal is 0 (no self-loops)"): scipy's `csgraph_laplacian(normed=True)` ignores self-loops, so `W_ii=0` is what makes the degree/`dd`/embedding match sklearn (REQ-1). sklearn's `rbf_kernel` diagonal `1.0` only surfaces in the separate `affinity_matrix_` attribute (REQ-9, NOT-STARTED). **FLAG (still divergent):** sklearn default `gamma = 1.0/n_features` (`:712`), ferrolearn `1.0` (REQ-8). |
| REQ-6 (kNN affinity GRAPH parity) | NOT-STARTED | open prereq blocker **#1445**. `fn build_affinity_matrix` NearestNeighbors arm (`:157-190`) computes pairwise squared distances, takes each point's top-`k` nearest (excluding self) via `neighbors.sort_by(...).take(k)`, and OR-symmetrizes `w[[i,j]] = w[[j,i]] = 1.0` — a binary 0/1 graph, NO self-loops, OR (not averaged) symmetrization. sklearn builds `kneighbors_graph(X, n_neighbors_, include_self=True)` then `0.5 * (A + A.T)` (`_spectral_embedding.py:703-709`) — WITH self-loops and AVERAGED symmetrization (Probe REQ-6). Different affinity → different `L_sym` → different embedding. |
| REQ-7 (`eigen_solver` arpack/lobpcg/amg + `random_state` + degenerate-subspace + symmetric-fixture sign-tie carve-outs) | NOT-STARTED | open prereq blocker **#1446**. sklearn exposes `eigen_solver ∈ {'arpack','lobpcg','amg'}` (`_spectral_embedding.py:153`) seeded by `random_state` (`:152`); `SpectralEmbedding` has no `eigen_solver`/`random_state` field and always uses `crate::mds::eigh_faer` (dense). **CARVE-OUTS** (out of REQ-1 scope): (a) DEGENERATE eigenvalue — for a near-disconnected graph the repeated-`0` eigenvector basis is rotation-non-unique, so faer/arpack pick different null-space bases and element-wise parity is impossible (no committed failing test, R-DEFER-3); (b) SYMMETRIC-fixture sign-flip ULP tie — an antisymmetric eigenvector with `|v[0]| == |v[4]|` is a ULP-level argmax tie that faer/numpy can round oppositely (→ opposite global sign), avoided in the REQ-1 pin by using the asymmetric fixture. |
| REQ-8 (default `affinity='nearest_neighbors'` + `gamma=None→1/n_features` + precomputed/callable) | NOT-STARTED | open prereq blocker **#1447**. sklearn `__init__` defaults `affinity="nearest_neighbors"` (`_spectral_embedding.py:163`) and `gamma=None` → `1.0/n_features` at fit (`:712`); accepts `affinity ∈ {'precomputed','precomputed_nearest_neighbors'}` + callable (`:680-715`). ferrolearn `SpectralEmbedding::new` defaults `RBF { gamma: 1.0 }` — DIFFERENT default affinity AND a fixed `gamma=1.0` not `1/n_features` (Probe REQ-8); no precomputed/callable affinity (the `Affinity` enum has only `RBF`/`NearestNeighbors`). |
| REQ-9 (`affinity_matrix_` / `n_neighbors_` fitted attrs + out-of-sample) | NOT-STARTED | open prereq blocker **#1448**. sklearn exposes fitted attrs `embedding_`, `affinity_matrix_` (`_spectral_embedding.py:89`), `n_neighbors_` (`:103`, `max(n/10,1)` default when `n_neighbors=None`, `:698-702`). `FittedSpectralEmbedding` exposes only `embedding()`; no `affinity_matrix_` accessor, no `n_neighbors_`. (Both sides are fit-only — no `transform`/out-of-sample — so that part is moot; the fitted-attr surface diverges.) |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker **#1449**. `grep -rln SpectralEmbedding ferrolearn-python/` is EMPTY — no `_RsSpectralEmbedding`, so `import ferrolearn` cannot reach `SpectralEmbedding`. The only non-test consumer of `fit`/`embedding()` is the crate re-export (`lib.rs`). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#1450**. `spectral_embedding.rs` computes on `ndarray::Array2<f64>` + `Vec<f64>` and eigendecomposes via `crate::mds::eigh_faer` (faer), not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`spectral_embedding.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`SpectralEmbedding { n_components: usize, affinity: Affinity }` (`new(n_components)`
defaulting `RBF { gamma: 1.0 }`, builder `with_affinity`, accessors
`n_components()` / `affinity()`) → `Fit<Array2<f64>, ()>` →
`FittedSpectralEmbedding { embedding_: Array2<f64> }` (accessor `embedding()`).
`Affinity` is a `Copy` enum with `RBF { gamma: f64 }` / `NearestNeighbors {
n_neighbors: usize }`. The dense path is `f64`-only (the module operates on
`Array2<f64>`, not generic `F`); every public method returns `Result<_,
FerroError>` (R-CODE-2). There is no `Predict`/`Transform` impl — `SpectralEmbedding`
is fit-only, mirroring sklearn (which also has no `transform`).

**Fit path (`fn fit`).** Validates `n_components != 0`, `n_samples >= 2`,
`n_components < n_samples`, and per-affinity (`n_neighbors ∈ [1, n)`, `gamma > 0`),
then runs the four-step pipeline:
1. `fn build_affinity_matrix` → `W` (RBF `exp(−γ·d²)` off-diagonal, diagonal `0`;
   or the binary OR-symmetrized no-self-loop kNN graph).
2. `fn normalised_laplacian` → `L_sym = I − D^{-1/2}WD^{-1/2}` (REQ-3, matches
   `csgraph_laplacian(normed=True)` + `_set_diag(L,1)`).
3. `crate::mds::eigh_faer(&L)` → `(eigenvalues, eigenvectors)`, sorted ascending
   (re-sorted defensively).
4. Skip the trivial first eigenvector, take the next `n_components` eigenvectors,
   rescale each row by `1/dd` (`dd[i] = sqrt(Σ_{j≠i} W[i,j])`, guarded `dd>1e-15`),
   then apply the per-column deterministic sign-flip into `embedding_` (REQ-1).

**The value match vs sklearn (REQ-1, was DIV/#1444 — now RESOLVED).** sklearn's
`_spectral_embedding` (`_spectral_embedding.py:300-469`) does TWO deterministic
steps after the dense `eigh` (`:439`), and ferrolearn's step 4 now performs BOTH:
- `embedding = embedding / dd` (`:443`) where `dd = sqrt(degree)` and `degree[i] =
  Σ_{j≠i} W[i,j]` is the off-diagonal row-sum (scipy `csgraph_laplacian(...,
  normed=True, return_diag=True)` ignores self-loops, `:333-335`). ferrolearn
  computes `dd[i] = sqrt(Σ_{j≠i} W[i,j])` and divides each eigenvector row by it.
- `embedding = _deterministic_vector_sign_flip(embedding)` (`:465`) — flips each
  vector so its max-\|entry\| is positive, pinning the otherwise basis-arbitrary
  sign. ferrolearn negates each column whose max-\|entry\| is negative.

The RBF affinity diagonal is kept at `0` (a wrong intermediate set it to `1.0`,
matching `rbf_kernel`; but scipy's `csgraph_laplacian(normed=True)` IGNORES the
diagonal, so degree = off-diagonal row-sum — diagonal `0` is what reproduces
sklearn's `dd`/Laplacian/embedding). On DISTINCT-eigenvalue data the dense
`L_sym` and its eigenspace are shared (REQ-3), so the `/dd` rescale and sign-flip
make the embedding match sklearn element-wise to f64 precision (REQ-1). TWO
carve-outs (REQ-7, #1446): for REPEATED eigenvalues the degenerate-subspace basis
is rotation-non-unique (faer/arpack differ, no committed test per R-DEFER-3); a
SYMMETRIC fixture with an antisymmetric `|v[0]|==|v[4]|` eigenvector is a
sign-flip ULP tie — both avoided by the asymmetric pin fixture.

**sklearn (target contract).** `class SpectralEmbedding(BaseEstimator)`
(`_spectral_embedding.py:472`) takes `__init__(n_components=2, *,
affinity="nearest_neighbors", gamma=None, random_state=None, eigen_solver=None,
n_neighbors=None, n_jobs=None)` (`:159-178`) under `_parameter_constraints`
(`:139-156`). `_get_affinity_matrix` (`:660`) builds `W` per `affinity`:
`precomputed` (`:680`), `precomputed_nearest_neighbors` `0.5*(A+A.T)` (`:682-688`),
`nearest_neighbors` `kneighbors_graph(X, n_neighbors_, include_self=True)` then
`0.5*(A+A.T)` with `n_neighbors_` default `max(n/10,1)` (`:689-710`), `rbf`
`rbf_kernel(X, gamma_)` with `gamma_` default `1/n_features` (`:711-714`), or a
callable (`:715`). `fit` (`:718`) calls `spectral_embedding` (`:167`/`:300`):
`csgraph_laplacian(normed=True, return_diag=True)` → `L, dd` (`:333`), `_set_diag(L,
1)` (`:347`), the eigensolver branch (arpack default / lobpcg / amg / dense `eigh`
`:439`), `embedding = diffusion_map.T[:n_components]` dropping the trivial vector,
`embedding = embedding / dd` (`:443`), `_deterministic_vector_sign_flip` (`:465`),
returning `embedding[1:n_components].T` (drop_first, `:466-467`). Fitted attrs:
`embedding_`, `affinity_matrix_` (`:89`), `n_neighbors_` (`:103`).

**The remaining gap.** ferrolearn now matches sklearn on the *RBF embedding
VALUE* (REQ-1, distinct-eigenvalue fixture), the *embedding SHAPE / separation /
determinism* (REQ-2), the *`L_sym` formula* (REQ-3), the *RBF off-diagonal
affinity* (REQ-4), and the *scoped error contracts* (REQ-5). It still DIVERGES
on: the kNN affinity GRAPH (REQ-6, binary/no-self-loop/OR vs
`kneighbors_graph(include_self=True)` + `0.5(A+Aᵀ)`); the `eigen_solver`/
`random_state` surface + degenerate-subspace and symmetric-fixture sign-tie
carve-outs (REQ-7, #1446); the default `affinity='nearest_neighbors'` +
`gamma=1/n_features` + precomputed/callable (REQ-8); the
`affinity_matrix_`/`n_neighbors_` fitted attrs, including the diagonal-`1.0`
`affinity_matrix_` (REQ-9); the PyO3 binding (REQ-10); and the ferray substrate
(REQ-11). This is now a **SHIPPED-VALUE / partial-surface** unit (5 SHIPPED /
6 NOT-STARTED).

## Verification

Library crate (green at baseline `dc2c13c3`):
```bash
cargo test -p ferrolearn-decomp spectral_embedding              # in-module SHAPE/error tests
cargo test -p ferrolearn-decomp --test divergence_spectral_embedding  # 6 passed; 0 failed
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s (`test_spectral_embedding_basic_shape`, `_1d`,
`_rbf_separates_clusters`, `_knn_affinity`, `_invalid_n_components_zero`,
`_n_components_too_large`, `_insufficient_samples`, `_knn_n_neighbors_zero`,
`_getters`, `_knn_too_many_neighbors`, `_negative_gamma`, `_larger_dataset`, plus
the doctest) pin the SHAPE/separation/determinism/error behavior. The
divergence-suite tests `green_rbf_embedding_value_parity_2` / `_1` pin the RBF
embedding VALUE element-wise against the live sklearn oracle (REQ-1) — both PASS
after the #1444 fix (the `/dd` rescale + sign-flip with RBF diagonal `0`).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-1 value pin
(R-CHAR-3 expected values), now GREEN against the fixed `spectral_embedding.rs`:
```bash
# REQ-1 (MATCHES) — RBF embedding values on the connected, distinct-eigenvalue,
# ASYMMETRIC `line5` fixture (gamma=0.3); ferrolearn matches element-wise (tol 1e-6):
python3 -c "import numpy as np; from sklearn.manifold import SpectralEmbedding; \
X=np.array([[0.,0.],[1.2,0.3],[2.,1.1],[3.5,0.2],[4.1,2.]]); \
print(SpectralEmbedding(n_components=2, affinity='rbf', gamma=0.3, eigen_solver='arpack', random_state=0).fit_transform(X).tolist())"
# sklearn -> [[-0.5492950432380824,0.5911304325996134],[-0.32478442062147483,-0.047104629065039066],
#             [0.0371154760143516,-0.564016870974581],[0.5056202555986625,0.053627532619619465],
#             [0.745795887914657,0.6045965227471306]]
# ferrolearn green_rbf_embedding_value_parity_2 / _1 assert this element-wise (~1e-15).

# sign-flip rule (REQ-1, :465) — each vector flipped so its max-|entry| is positive:
python3 -c "import numpy as np; from sklearn.utils.extmath import _deterministic_vector_sign_flip; \
print(_deterministic_vector_sign_flip(np.array([[1.,-3.,2.],[-5.,1.,4.]])).tolist())"
# -> [[-1.0, 3.0, -2.0], [5.0, -1.0, -4.0]]
```
The R-CHAR-3 pin lives in
`ferrolearn-decomp/tests/divergence_spectral_embedding.rs`
(`green_rbf_embedding_value_parity_2`/`_1`), asserting the live-sklearn embedding
above (`SKLEARN_EMBEDDING_2`/`_1`) and PASSING against the fixed
`spectral_embedding.rs`. The DEGENERATE-eigenvalue two-cluster fixture is
intentionally NOT pinned (carve-out, REQ-7/#1446, R-DEFER-3).

ferrolearn-python (REQ-10 binding parity, after #1449 lands):
```bash
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_spectral_embedding.py -q
```
asserting `ferrolearn.SpectralEmbedding` exists and exposes `embedding_` /
`affinity_matrix_` / `n_neighbors_`, matching `sklearn.manifold.SpectralEmbedding`
on the AC fixtures.

## Blockers

(#1443 is this doc's crosslink tracking issue. The core value-parity blocker
#1444 (REQ-1) is RESOLVED — listed below for provenance; the rest are open.)

- **#1444** — REQ-1 (`/dd` rescale + sign-flip): **RESOLVED.** Added `embedding =
  embedding / dd` (`dd = sqrt(off-diagonal-row-sum degree)`,
  `_spectral_embedding.py:443`) and the per-column `_deterministic_vector_sign_flip`
  (flip each vector so its max-\|entry\| is positive, `:465`) to `fn fit`, keeping
  the RBF affinity diagonal at `0` (scipy ignores self-loops). On the connected,
  distinct-eigenvalue, asymmetric fixture this closes the RBF embedding value gap
  to f64 precision; REQ-1 is SHIPPED (`green_rbf_embedding_value_parity_2`/`_1`).
  Carve-outs (degenerate eigenvalue, symmetric-fixture sign tie) tracked under
  REQ-7/#1446.
- **#1445** — REQ-6: replace the binary OR-symmetrized no-self-loop kNN graph
  with `kneighbors_graph(X, n_neighbors_, include_self=True)` + `0.5*(A+A.T)`
  (`_spectral_embedding.py:703-709`).
- **#1446** — REQ-7: add `eigen_solver` (arpack/lobpcg/amg) + `random_state`;
  document the degenerate-eigenvalue subspace carve-out for REQ-1 value parity.
- **#1447** — REQ-8: default `affinity='nearest_neighbors'`; resolve `gamma=None
  → 1/n_features` (`:712`); add precomputed/callable affinity (`:680-715`).
- **#1448** — REQ-9: expose `affinity_matrix_` (`:89`) and `n_neighbors_` (`:103`,
  default `max(n/10,1)`) on `FittedSpectralEmbedding`.
- **#1449** — REQ-10: add `_RsSpectralEmbedding` to `ferrolearn-python` (fit /
  embedding_ / affinity_matrix_ / parameter surface) — the boundary CPython
  consumer.
- **#1450** — REQ-11: migrate `spectral_embedding.rs` off `ndarray` /
  `crate::mds::eigh_faer` (faer) to `ferray-core` / `ferray::linalg` (R-SUBSTRATE).
