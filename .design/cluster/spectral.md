# Spectral Clustering (sklearn.cluster.SpectralClustering)

<!--
tier: 3-component
status: draft
baseline-commit: dc47aa06
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_spectral.py            # class SpectralClustering(ClusterMixin, BaseEstimator) (:379-801); __init__ (:633-666); _parameter_constraints (:606-631); fit affinity construction (:691-732); _spectral_embedding call (:744-751); assign_labels branch (:755-766)
  - sklearn/manifold/_spectral_embedding.py # _spectral_embedding (:300-469); csgraph_laplacian(normed=True) (:333-335); shift-invert eigsh + embedding=embedding/dd (:367-378); eigh path embedding/dd (:439-443); _deterministic_vector_sign_flip + drop_first (:465-469)
ferrolearn-module: ferrolearn-cluster/src/spectral.rs
parity-ops: SpectralClustering (.__init__, .fit, .fit_predict, .labels_, .affinity_matrix_)
crosslink-issue: 928
-->

## Summary

`ferrolearn-cluster/src/spectral.rs` mirrors scikit-learn's `SpectralClustering`
(`sklearn/cluster/_spectral.py`, `class SpectralClustering(ClusterMixin,
BaseEstimator)` `:379-801`) — graph-Laplacian eigenmap clustering. It exposes the
unfitted `SpectralClustering<F>` (`n_clusters` required, `gamma=1.0`, `n_init=10`,
`random_state`), the fitted `FittedSpectralClustering<F>` (stores `labels_`), and a
`fit_predict` convenience mirroring `ClusterMixin.fit_predict`. It is re-exported at
the crate root (`pub use spectral::{FittedSpectralClustering, SpectralClustering}`
in `ferrolearn-cluster/src/lib.rs`).

**This unit has DEEP algorithmic divergence from sklearn, and under honest
underclaim (R-HONEST-3) almost nothing VALUE-matches the oracle.** The pipeline
ferrolearn implements is a *variant* of spectral clustering, not the sklearn
contract:

| step | sklearn | ferrolearn (`spectral.rs`) |
|---|---|---|
| affinity | `pairwise_kernels(X, metric='rbf')` = `exp(-gamma*d^2)` | `affinity_matrix`: `exp(-gamma*||xi-xj||^2)`, diagonal forced to `1.0` |
| Laplacian | `csgraph_laplacian(A, normed=True)` = `L = I − D^{-1/2}AD^{-1/2}` (`_spectral_embedding.py:333`) | `normalized_laplacian`: `D^{-1/2}AD^{-1/2}` — the normalized **adjacency**, NOT `I − …` |
| eigenvectors | **smallest** eigenvalues of `L` (shift-invert eigsh / eigh on `−L`) | **largest** eigenvalues of `D^{-1/2}AD^{-1/2}` (`top_k_eigenvectors`) |
| post-scaling | `embedding = embedding / dd` (recover `u = D^{-1/2}x`, `:378/:443`) then `_deterministic_vector_sign_flip` (`:465`) | `row_normalize`: divide each row by its L2 norm |
| assign_labels | `k_means(maps, …)` (sklearn `_kmeans.k_means`) | `KMeans::new(k).with_n_init(n_init)` (ferrolearn) |

The eigenvector **subspace** is the same in both (the top-`k` of `D^{-1/2}AD^{-1/2}`
are the bottom-`k` of `I − D^{-1/2}AD^{-1/2}`), so on well-separated data both
recover the obvious partition. But the embedding **values diverge** (sklearn scales
rows by `1/dd = 1/sqrt(degree)`; ferrolearn scales rows to unit L2 norm), and the
final labels additionally depend on a *different* KMeans implementation with its own
RNG. So `labels_` is **NOT a value-parity claim** against `SpectralClustering` —
it matches on some fixtures and diverges on others (probes below).

**The only behavior that VALUE-matches the live sklearn 1.5.2 oracle is the RBF
affinity matrix** (`exp(-gamma*d^2)`, equal to `rbf_kernel(X, gamma=gamma)` to full
f64 precision), and that is computed by a **private** helper (`fn affinity_matrix`)
with no public accessor and no non-test consumer — it cannot be SHIPPED on its own.
`SpectralClustering` / `FittedSpectralClustering` are existing pub APIs (grandfathered
per S5/R-DEFER-1); their only consumer is the crate re-export — **there is no
`ferrolearn-python` binding** (`grep -rln SpectralClustering ferrolearn-python/`
is empty) and **no other in-crate consumer**.

## Live oracle probes (sklearn 1.5.2, run from /tmp; ferrolearn via throwaway `cargo run --example` probe, since deleted)

### Probe 1 — label VALUE parity (the load-bearing probe)

**Fixture A — two well-separated blobs** (the `two_blobs()` fixture from
`spectral.rs` tests: 5 points near origin + 5 near `(10,10)`):
```
python3 -c "from sklearn.cluster import SpectralClustering; import numpy as np; \
X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0.,0.1],[10.,10.],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10.,10.1]]); \
print(SpectralClustering(n_clusters=2,gamma=0.1,affinity='rbf',random_state=42,assign_labels='kmeans').fit_predict(X).tolist())"
# sklearn: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
```
ferrolearn `SpectralClustering::<f64>::new(2).with_gamma(0.1).with_random_state(42).fit_predict(&X)`
→ `[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]` — **AGREE** (0 mismatches; same orientation).

**Fixture B — concentric circles** (`make_circles(n_samples=30, factor=0.4,
noise=0.05, random_state=0)`, the canonical non-convex spectral case):
```
# gamma=0.1
sklearn:    [1,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0]
ferrolearn: [1,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0]   # 0 mismatches — AGREE
# gamma=10
sklearn:    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0]
ferrolearn: [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0]   # 2 mismatches (idx 10,19) — DIVERGE
```

**Conclusion**: label parity is **fixture-dependent and NOT guaranteed**. It agrees
on the well-separated blobs and on circles at `gamma=0.1`, but **diverges** at
`gamma=10` (2/30 points differ). Because the embedding values differ and the KMeans
RNG/init differ, agreement is coincidental on benign fixtures, not a contract. Per
the dispatch instruction, "separates well-separated blobs correctly" is NOT a
value-parity basis for SHIPPED; label parity is therefore **NOT-STARTED**.

### Probe 2 — embedding divergence root-cause (the core finding)

On the blobs fixture (`gamma=0.1`), reproducing each side's embedding:
```
sklearn _spectral_embedding(A, n_components=2, drop_first=False):   # the `maps` fed to k_means
  rows 0-4: [ 0.15853,  0.15853]      rows 5-9: [ 0.15853, -0.15853]
ferrolearn (top-2 eigvecs of D^{-1/2}AD^{-1/2}, then row-L2-normalize):
  rows 0-4: [-0.70711, -0.70711]      rows 5-9: [ 0.70711, -0.70711]
```
The two embeddings are **numerically different** — sklearn's row magnitude is
`1/dd ≈ 0.158` (recovered `u = D^{-1/2}x`), ferrolearn's is exactly `1.0` (unit
L2). The **partition structure** (which rows are co-located) is identical, which is
why KMeans recovers the same blobs split — but `embedding != maps`, so the
embedding does NOT value-match `spectral_embedding`. Root cause, precisely:
1. ferrolearn computes `D^{-1/2}AD^{-1/2}` (`normalized_laplacian`), sklearn computes
   `L = I − D^{-1/2}AD^{-1/2}` (`csgraph_laplacian(normed=True)`, `_spectral_embedding.py:333`).
   These are negatives-plus-identity, so eigenvectors of the **largest** eigenvalues
   of the former equal the **smallest** of the latter — the subspace coincides.
2. **Post-scaling diverges**: sklearn does `embedding = embedding / dd`
   (`_spectral_embedding.py:378` & `:443`) to recover `u = D^{-1/2}x`; ferrolearn does
   `row_normalize` (unit L2 per row). These are different transforms of the same
   subspace → different coordinates → KMeans on different point clouds.
3. sklearn applies `_deterministic_vector_sign_flip` (`:465`) for sign determinism;
   ferrolearn has no equivalent (sign comes from `faer`'s `eigh`).
4. The final `k_means`/`KMeans` is a different implementation with its own init
   (k-means++ vs ferrolearn's KMeans init) and RNG — so even an identical embedding
   would not guarantee identical labels (see kmeans.rs, a separate unit).

### Probe 3 — affinity modes
sklearn `affinity ∈ {'rbf','nearest_neighbors','precomputed','precomputed_nearest_neighbors'}
∪ KERNEL_PARAMS` (`'poly','sigmoid','laplacian','chi2',…`) (`_spectral.py:613-619`).
ferrolearn supports **RBF only** (`fn affinity_matrix`, hard-coded `exp(-gamma*sq)`);
`'nearest_neighbors'` (`kneighbors_graph`, `:709-713`), `'precomputed'` (`:720-721`),
and the poly/sigmoid kernels are absent.

### Probe 4 — assign_labels modes
sklearn `assign_labels ∈ {'kmeans','discretize','cluster_qr'}` (`_spectral.py:625`,
branch `:755-766`; `discretize` `:57-189`, `cluster_qr` `:25-54`). ferrolearn
supports **kmeans only** (hard-coded `KMeans` in `fn fit`); `discretize` and
`cluster_qr` are absent.

### Probe 5 — defaults / params / gamma validation
- **`n_clusters` default**: sklearn `n_clusters=8` (`_spectral.py:635`); ferrolearn
  **requires** `n_clusters` (`fn new(n_clusters: usize)`) — no default.
- **RBF affinity VALUE matches**: `rbf_kernel(X, gamma=0.1)[0,1] = 0.9950124791926823`,
  `[0,5] = 2.061153622438558e-09`; ferrolearn `exp(-0.1*sqdist)` →
  `0.9950124791926823` / `2.061153622438558e-09` — **identical to full f64 precision**.
  (sklearn `rbf_kernel` diagonal is `1.0`; ferrolearn forces the diagonal to `1.0`
  explicitly — equal.)
- **gamma validation**: sklearn `_parameter_constraints["gamma"] =
  Interval(Real, 0, None, closed="left")` = `[0.0, inf)` (`_spectral.py:612`). So
  `gamma=0.0` is **accepted** (probe: `SpectralClustering(gamma=0.0).fit_predict(X)`
  runs, RBF→all-ones) and `gamma=-1.0` raises `InvalidParameterError` (probe:
  "The 'gamma' parameter … must be a float in the range [0.0, inf). Got -1.0").
  ferrolearn rejects `gamma <= 0` (`fn fit`, the `self.gamma <= F::zero()` guard) —
  so it **over-rejects `gamma=0`** (which sklearn accepts) and rejects `gamma<0` with
  the wrong error type (`FerroError::InvalidParameter`, not the sklearn
  `InvalidParameterError`/`ValueError` ABI).
- **missing params**: `eigen_solver`, `n_components`, `eigen_tol`, `n_neighbors`,
  `degree`, `coef0`, `kernel_params`, `n_jobs`, `verbose`, `affinity` (`:637-666`)
  are all absent from `SpectralClustering<F>`.
- **missing fitted attribute**: sklearn exposes `affinity_matrix_` and
  `n_features_in_` (`:524-538`); `FittedSpectralClustering` exposes only `labels()`.

### Probe 6 — KMeans RNG/init parity
Even with a matched embedding, sklearn's `k_means` default init is `k-means++` with
its own RNG; ferrolearn's `KMeans` has its own init/RNG (kmeans.rs). Exact label
parity is therefore additionally gated on KMeans parity, a separate translation unit.

### Probe 7 — non-test consumer
`grep -rln "SpectralClustering\|RsSpectral" ferrolearn-python/` is **empty** — there
is **no PyO3 binding**. `grep -rn SpectralClustering ferrolearn-cluster/src/`
outside `spectral.rs` finds only the crate re-export (`lib.rs`). The only public
entry points are `fit` / `fit_predict` / `labels()`, and their sole non-test
consumer is the crate re-export. The value-matching RBF affinity
(`fn affinity_matrix`) is **private** with no accessor.

## Requirements

- REQ-1: **RBF affinity matrix VALUE (R-DEV-1).** Mirror `pairwise_kernels(X,
  metric='rbf')` = `exp(-gamma*||xi-xj||^2)` (`_spectral.py:730`). ferrolearn
  `fn affinity_matrix` value-matches `rbf_kernel(X, gamma=gamma)` to full f64
  precision — BUT it is a private helper with no public accessor / no non-test
  consumer, so it cannot be SHIPPED on its own (R-HONEST-2/R-DEFER-1).
- REQ-2: **`labels_` VALUE parity (R-DEV-1/3 — the core requirement).** Mirror
  `SpectralClustering.fit().labels_` (`_spectral.py:756-766`). ferrolearn's labels
  diverge from the oracle because the embedding diverges (REQ-3) and KMeans diverges
  (REQ-9) — agreement on benign fixtures is coincidental, not a contract.
- REQ-3: **spectral embedding algorithm — `csgraph_laplacian(normed=True)` +
  `embedding/dd` + sign-flip (R-DEV-1, the core algorithmic divergence).** Mirror
  `_spectral_embedding` (`_spectral_embedding.py:300-469`): `L = I −
  D^{-1/2}AD^{-1/2}`, smallest-eigenvalue eigenvectors, `embedding = embedding / dd`,
  `_deterministic_vector_sign_flip`. ferrolearn computes `D^{-1/2}AD^{-1/2}` top-k +
  row-L2-normalize — a different embedding (Probe 2).
- REQ-4: **affinity modes `nearest_neighbors`/`precomputed`/`precomputed_nearest_neighbors`/poly/sigmoid/laplacian (R-DEV-2).**
  sklearn `_spectral.py:709-732`. ferrolearn supports RBF only.
- REQ-5: **assign_labels `discretize` / `cluster_qr` (R-DEV-2).** sklearn
  `_spectral.py:755-766` (`discretize` `:57-189`, `cluster_qr` `:25-54`). ferrolearn
  supports kmeans only.
- REQ-6: **missing params `eigen_solver`/`n_components`/`eigen_tol`/`n_neighbors`/`degree`/`coef0`/`kernel_params`/`n_jobs`/`verbose`/`affinity` (R-DEV-2).**
  sklearn `__init__` `:633-666`. ferrolearn has only `n_clusters`/`gamma`/`n_init`/`random_state`.
- REQ-7: **`n_clusters=8` default + gamma `[0,inf)` validation + `InvalidParameterError` ABI (R-DEV-2).**
  sklearn `n_clusters=8` default (`:635`), `gamma ∈ [0.0,inf)` (`:612`, accepts 0,
  rejects <0 with `InvalidParameterError`). ferrolearn requires `n_clusters`, rejects
  `gamma<=0` (over-rejects 0) with `FerroError::InvalidParameter`.
- REQ-8: **fitted-attribute surface `affinity_matrix_` / `n_features_in_` (R-DEV-3).**
  sklearn `:524-538`. ferrolearn exposes only `labels()`.
- REQ-9: **KMeans assign-labels parity (R-DEV-1).** sklearn `k_means` (k-means++
  init + own RNG, `_spectral.py:756`); ferrolearn `KMeans` is a separate unit with
  its own init/RNG — even a matched embedding would not guarantee identical labels.
- REQ-10: **PyO3 binding (R-DEFER-1/3).** No `_RsSpectralClustering` in
  `ferrolearn-python` — `import ferrolearn` cannot reach `SpectralClustering`.
- REQ-11: **ferray substrate (R-SUBSTRATE).** `spectral.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float` + `ferrolearn_core::NdarrayFaerBackend`,
  not `ferray-core` / `ferray::linalg`.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). ferrolearn values from a throwaway
`cargo run --example` probe (since deleted). Fixtures: `blobs` (the `two_blobs()`
test fixture, 10×2), `circles` = `make_circles(n_samples=30, factor=0.4,
noise=0.05, random_state=0)`.

- AC-1 (REQ-1, value-matches but no consumer): `rbf_kernel(blobs, gamma=0.1)[0,1] =
  0.9950124791926823`, `[0,5] = 2.061153622438558e-09`; ferrolearn `exp(-gamma*sq)`
  → identical to full f64 precision. (Private helper, no accessor → not SHIPPABLE.)
- AC-2 (REQ-2, diverges): `SpectralClustering(n_clusters=2,gamma=10,random_state=42).fit_predict(circles)`
  → sklearn `[1,0,…,0,0,…,1,…,0]`; ferrolearn differs at indices 10,19 (2/30
  mismatches). Label parity FAILS at `gamma=10`.
- AC-3 (REQ-3, diverges): on `blobs gamma=0.1`, sklearn `_spectral_embedding(A,
  n_components=2, drop_first=False)` row magnitude ≈ `0.158` (`= 1/dd`); ferrolearn
  embedding row magnitude `= 1.0` (unit L2) — different coordinates (Probe 2).
- AC-4 (REQ-7): `SpectralClustering(gamma=0.0).fit_predict(blobs)` runs in sklearn;
  ferrolearn `with_gamma(0.0).fit` returns `Err(InvalidParameter)` — over-rejection.
  `SpectralClustering(gamma=-1.0)` → sklearn `InvalidParameterError`; ferrolearn
  `FerroError::InvalidParameter` (different ABI).
- AC-5 (REQ-4/5/6/8/10): `hasattr(SpectralClustering(), 'affinity')` /
  `'eigen_solver'` / `'assign_labels'` True; `SpectralClustering(affinity='nearest_neighbors').fit(X)`
  works in sklearn. ferrolearn `SpectralClustering<F>` has no such params, no
  `affinity_matrix_`, and no `ferrolearn.SpectralClustering` (no binding).

## REQ status table

Binary (R-DEFER-2). `SpectralClustering` / `FittedSpectralClustering` are existing
pub APIs re-exported at the crate root (the only non-test consumer; grandfathered
S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2,
commit 156ef14). Live oracle = installed sklearn 1.5.2, run from `/tmp`. Honest
underclaim (R-HONEST-3): **nothing in this unit value-matches the
`SpectralClustering` contract end-to-end with a non-test consumer** — the RBF
affinity value-matches but is a private helper with no consumer (REQ-1), and the
final `labels_` diverge (REQ-2, the load-bearing parity). Every REQ is NOT-STARTED.
Suggested blocker numbers — the director creates the real issues; #928 is this
doc's crosslink tracking issue, reused for the core embedding divergence (REQ-3).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (RBF affinity VALUE) | NOT-STARTED | open prereq blocker **#929**. impl `fn affinity_matrix` computes `exp(-gamma*||xi-xj||^2)` (diagonal forced `1.0`), VALUE-matching `pairwise_kernels(X, metric='rbf')` (`_spectral.py:730`) to full f64 precision (AC-1: `[0,1]=0.9950124791926823`, `[0,5]=2.061153622438558e-09`). BUT `fn affinity_matrix` is a PRIVATE helper with no public accessor and **no non-test consumer** (no `affinity_matrix_` attribute, REQ-8; no binding, REQ-10) — so it cannot be SHIPPED standalone (R-HONEST-2/R-DEFER-1). Blocker: expose `affinity_matrix_` so the value-matching affinity has a real consumer. |
| REQ-2 (`labels_` VALUE parity) | NOT-STARTED | open prereq blocker **#928** (depends on #930/#937). `fit`/`fit_predict` produce `labels_` via a DIFFERENT pipeline (row-L2-normalized top-k of `D^{-1/2}AD^{-1/2}` + ferrolearn `KMeans`) than sklearn's `_spectral_embedding` + `k_means` (`_spectral.py:756-766`). Pin (AC-2): on `make_circles(…,random_state=0)` at `gamma=10`, sklearn `fit_predict` vs ferrolearn differ at indices 10,19 (2/30). Agreement on blobs / circles@0.1 is COINCIDENTAL (the subspace is shared) — NOT value parity. Gated on REQ-3 (embedding) + REQ-9 (KMeans). |
| REQ-3 (spectral embedding algorithm) | NOT-STARTED | open prereq blocker **#928** (the cleanest single-file core fix). sklearn `_spectral_embedding` (`_spectral_embedding.py:300-469`): `L = csgraph_laplacian(A, normed=True)` = `I − D^{-1/2}AD^{-1/2}` (`:333`), SMALLEST-eigenvalue eigenvectors, `embedding = embedding / dd` (`:378`/`:443`), `_deterministic_vector_sign_flip` (`:465`). ferrolearn `normalized_laplacian` builds `D^{-1/2}AD^{-1/2}` (NO `I−`), `top_k_eigenvectors` takes the LARGEST, `row_normalize` does unit-L2 per row (NOT `/dd`), no sign flip. Pin (AC-3): blobs `gamma=0.1` sklearn embedding row-magnitude ≈ `0.158` (`=1/dd`) vs ferrolearn `1.0` — numerically different coordinates (Probe 2). **The critic should pin REQ-3 FIRST** — it is the root cause; REQ-2 unblocks once the embedding (and #937 KMeans) match. |
| REQ-4 (affinity modes nearest_neighbors/precomputed/poly/sigmoid) | NOT-STARTED | open prereq blocker **#931**. sklearn builds affinity via `kneighbors_graph` (`'nearest_neighbors'`, `:709-713`), passthrough (`'precomputed'`, `:720-721`), or any `pairwise_kernels` metric (`:730`). ferrolearn `fn affinity_matrix` is hard-coded RBF only; no `affinity` parameter. |
| REQ-5 (assign_labels discretize/cluster_qr) | NOT-STARTED | open prereq blocker **#932**. sklearn `assign_labels ∈ {'kmeans','discretize','cluster_qr'}` (`_spectral.py:625`, branch `:755-766`; `discretize` `:57-189`, `cluster_qr` `:25-54`). ferrolearn `fn fit` hard-codes `KMeans`; no `assign_labels` parameter. |
| REQ-6 (missing params eigen_solver/n_components/eigen_tol/n_neighbors/degree/coef0/kernel_params/n_jobs/verbose/affinity) | NOT-STARTED | open prereq blocker **#933**. sklearn `__init__` (`_spectral.py:633-666`) takes 16 params. `SpectralClustering<F>` has only `n_clusters`/`gamma`/`n_init`/`random_state` (`fn new` + builders) — the eigensolver/embedding/affinity/kernel knobs are absent. |
| REQ-7 (`n_clusters=8` default + gamma `[0,inf)` + `InvalidParameterError` ABI) | NOT-STARTED | open prereq blocker **#934**. sklearn `n_clusters=8` default (`:635`); `gamma ∈ [0.0,inf)` (`:612`) — accepts `0.0`, rejects `<0` with `InvalidParameterError`. ferrolearn `fn new(n_clusters)` has no default; `fn fit` rejects `gamma <= 0` — OVER-rejects `gamma=0` (sklearn accepts it, AC-4), and uses `FerroError::InvalidParameter` not the sklearn `ValueError`/`InvalidParameterError` ABI. |
| REQ-8 (fitted attrs `affinity_matrix_`/`n_features_in_`) | NOT-STARTED | open prereq blocker **#935**. sklearn exposes `affinity_matrix_` (`:524-526`) + `n_features_in_` (`:531-532`). `FittedSpectralClustering<F>` stores only private `labels_` (+ `PhantomData`), exposes only `labels()`. No `affinity_matrix_` accessor (so the value-matching affinity has no consumer — REQ-1). |
| REQ-9 (KMeans assign-labels parity) | NOT-STARTED | open prereq blocker **#937** (depends on the kmeans.rs unit). sklearn `k_means(maps, n_clusters, n_init, random_state)` uses k-means++ init with NumPy RNG (`_spectral.py:756`). ferrolearn `KMeans::new(k).with_n_init` is a separate unit with its own init/RNG. Even a matched embedding (REQ-3) would not guarantee identical labels without KMeans parity. |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker **#936**. `grep -rln SpectralClustering ferrolearn-python/` is EMPTY — there is no `_RsSpectralClustering`, so `import ferrolearn` cannot reach `SpectralClustering`. The only non-test consumer of `fit`/`fit_predict`/`labels()` is the crate re-export (`lib.rs`). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#938**. `spectral.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` + `ferrolearn_core::NdarrayFaerBackend` (`eigh`); not migrated to `ferray-core` / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`spectral.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`SpectralClustering<F>` (`n_clusters`, `gamma`, `n_init`, `random_state`) →
`Fit<Array2<F>, ()>` → `FittedSpectralClustering<F>` (private `labels_: Array1<usize>`).
Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2). No `Predict` impl (matching the documented
"no out-of-sample embedding" note).

**Fit path (`fn fit`).** Validates `n_clusters > 0`, `gamma > 0` (over-rejects
`gamma=0` vs sklearn `[0,inf)`, REQ-7), `n_samples >= n_clusters`. Then runs the
five-step pipeline (`fn affinity_matrix` → `fn normalized_laplacian` →
`fn top_k_eigenvectors` → `fn row_normalize` → `KMeans`). The eigendecomposition is
done in `f64` via `NdarrayFaerBackend::eigh` (faer only supports f64), cast back to
`F`. **The pipeline is a spectral-clustering *variant*, not the sklearn
`_spectral_embedding` contract** — see the Summary table and Probe 2:
- `normalized_laplacian` = `D^{-1/2}AD^{-1/2}` (normalized adjacency), NOT sklearn's
  `L = I − D^{-1/2}AD^{-1/2}` (`csgraph_laplacian(normed=True)`, `:333`).
- `top_k_eigenvectors` takes the LARGEST eigenvalues of that adjacency (= smallest
  of `L` — same subspace), but `row_normalize` (unit L2 per row) replaces sklearn's
  `embedding = embedding / dd` (`:378`/`:443`), and there is no
  `_deterministic_vector_sign_flip` (`:465`).
- the final clustering is ferrolearn's `KMeans`, not sklearn's `k_means` (REQ-9).

**Invariants held vs sklearn:** the RBF affinity VALUE (`exp(-gamma*d^2)`, AC-1 —
matches `rbf_kernel`, but private/no-consumer, REQ-1); `labels()` length = `n_samples`;
labels in `[0, n_clusters)`; seed reproducibility (`with_random_state`); the
eigenvector *subspace* (so well-separated blobs cluster correctly — but that is NOT
value parity).

**Invariants NOT held vs sklearn:** `labels_` VALUE (REQ-2 — diverges on
`circles gamma=10`); the embedding VALUE (REQ-3 — `1/dd` vs unit-L2, the core
divergence); affinity modes (REQ-4); assign_labels modes (REQ-5); the parameter
surface (REQ-6); `n_clusters=8` default + gamma `[0,inf)` + error ABI (REQ-7);
fitted attributes `affinity_matrix_`/`n_features_in_` (REQ-8); KMeans parity (REQ-9);
the PyO3 binding (REQ-10); the ferray substrate (REQ-11).

**Consumer wiring.** The only non-test consumer is the crate re-export
(`pub use spectral::{FittedSpectralClustering, SpectralClustering}`,
`ferrolearn-cluster/src/lib.rs`). There is no `ferrolearn-python` binding and no
other in-crate consumer (Probe 7).

## Verification

Library crate (green at baseline `dc47aa06` for the existing variant behavior):
```
cargo test -p ferrolearn-cluster --lib spectral     # 12 passed; 0 failed
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The 12 in-tree `#[test]`s (`test_two_blobs_two_clusters`,
`test_labels_length_matches_n_samples`, `test_labels_in_valid_range`,
`test_single_cluster`, `test_invalid_n_clusters_zero`, `test_invalid_gamma_zero`,
`test_invalid_gamma_negative`, `test_empty_data_error`,
`test_insufficient_samples_error`, `test_n_clusters_equals_n_samples`,
`test_f32_support`, `test_reproducibility_with_seed`) pin ferrolearn's current
**variant** behavior (label co-membership on blobs, length/range, reproducibility,
error edges). **None compares `labels_` or the embedding VALUE against the live
sklearn `SpectralClustering` oracle**, so they stay green despite the divergences;
note `test_invalid_gamma_zero` actively asserts the REQ-7 over-rejection
(`gamma=0.0` errors), which DIVERGES from sklearn (sklearn accepts `gamma=0`).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin REQ-3 (the embedding algorithm) FIRST**
— it is the single-file root cause; REQ-2 (label parity) unblocks once REQ-3 and the
kmeans.rs unit (#937) match:
```
# REQ-1 (value-matches, no consumer) RBF affinity
python3 -c "import numpy as np; from sklearn.metrics.pairwise import rbf_kernel; \
X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0.,0.1],[10.,10.],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10.,10.1]]); \
A=rbf_kernel(X,gamma=0.1); print(A[0,1], A[0,5])"   # 0.9950124791926823 2.061153622438558e-09  (ferro: identical)
# REQ-2 (DIVERGES) labels_ on circles gamma=10
python3 -c "import numpy as np; from sklearn.cluster import SpectralClustering; from sklearn.datasets import make_circles; \
X,_=make_circles(n_samples=30,factor=0.4,noise=0.05,random_state=0); \
print(SpectralClustering(n_clusters=2,gamma=10.,random_state=42).fit_predict(X).tolist())"
# sklearn [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0]; ferro differs at idx 10,19
# REQ-3 (DIVERGES) embedding values
python3 -c "import numpy as np; from sklearn.manifold._spectral_embedding import _spectral_embedding; from sklearn.metrics.pairwise import rbf_kernel; \
X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0.,0.1],[10.,10.],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10.,10.1]]); \
A=rbf_kernel(X,gamma=0.1); m=_spectral_embedding(A,n_components=2,eigen_solver='arpack',random_state=np.random.RandomState(42),drop_first=False); print(m[0].tolist())"
# sklearn maps row0 ~ [0.15853, 0.15853] (=1/dd); ferro row-L2-normalized ~ [-0.70711,-0.70711]
# REQ-7 (DIVERGES) gamma=0 accepted by sklearn
python3 -c "import numpy as np; from sklearn.cluster import SpectralClustering; \
X=np.array([[0.,0.],[1.,1.],[5.,5.],[6.,6.]]); print(SpectralClustering(n_clusters=2,gamma=0.,random_state=0).fit_predict(X).tolist())"  # [0,0,1,0] — runs; ferro Errs (InvalidParameter)
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-cluster/tests/divergence_spectral.rs`, asserting the live-sklearn
expected values above and FAILING against current `spectral.rs`. Note the in-tree
`test_invalid_gamma_zero` must be corrected (or removed) when REQ-7 lands, since
sklearn accepts `gamma=0` (R-HONEST-4).

ferrolearn-python (REQ-10 binding parity, after #936 lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_spectral.py -q
```
asserting `ferrolearn.SpectralClustering` exists and exposes `labels_` /
`affinity_matrix_` / the sklearn parameter surface, matching
`sklearn.cluster.SpectralClustering` on the AC fixtures.

## Blockers to open

(Director creates the real issues; #928 is this doc's crosslink tracking issue,
reused for the core embedding/label divergence REQ-3/REQ-2; the rest are SUGGESTIONS.)

- **#928** — REQ-3 (spectral embedding) / REQ-2 (label parity): replace
  `normalized_laplacian` (`D^{-1/2}AD^{-1/2}`) + `top_k_eigenvectors` (largest) +
  `row_normalize` with sklearn's `_spectral_embedding` contract — `L = I −
  D^{-1/2}AD^{-1/2}`, smallest-eigenvalue eigenvectors, `embedding = embedding/dd`,
  `_deterministic_vector_sign_flip` (`_spectral_embedding.py:333/378/465`). **The
  core single-file fix — pin FIRST.** REQ-2 unblocks once this + #937 (KMeans) land.
- **#929** — REQ-1/REQ-8: expose `affinity_matrix_` on `FittedSpectralClustering` so
  the value-matching RBF affinity (`fn affinity_matrix`) has a real consumer; add
  `n_features_in_` (`_spectral.py:524-538`).
- **#931** — REQ-4: add the `affinity` parameter + `nearest_neighbors`
  (`kneighbors_graph`)/`precomputed`/poly/sigmoid/laplacian construction
  (`_spectral.py:709-732`).
- **#932** — REQ-5: add the `assign_labels` parameter + `discretize` (`:57-189`) /
  `cluster_qr` (`:25-54`) label-assignment branches (`_spectral.py:755-766`).
- **#933** — REQ-6: add `eigen_solver`/`n_components`/`eigen_tol`/`n_neighbors`/
  `degree`/`coef0`/`kernel_params`/`n_jobs`/`verbose` params (`_spectral.py:633-666`).
- **#934** — REQ-7: `n_clusters=8` default; accept `gamma=0` (`[0,inf)`); reject
  `gamma<0` with the sklearn `ValueError`/`InvalidParameterError` ABI (`:612`).
- **#936** — REQ-10: add `_RsSpectralClustering` to `ferrolearn-python` (fit /
  fit_predict / labels_ / affinity_matrix_ + parameter surface).
- **#937** — REQ-9: KMeans assign-labels parity (k-means++ init + RNG) — owned by
  the kmeans.rs unit; REQ-2 depends on it.
- **#938** — REQ-11: migrate `spectral.rs` off `ndarray`/`num-traits`/`NdarrayFaerBackend`
  to `ferray-core` / `ferray::linalg` (R-SUBSTRATE).
