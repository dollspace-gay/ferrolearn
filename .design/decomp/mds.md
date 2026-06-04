# Multidimensional Scaling (sklearn.manifold.MDS)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: b8ff9e61
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/manifold/_mds.py  # _smacof_single (:22-167): SMACOF iterative stress-majorization on a symmetric dissimilarity matrix — random init X = random_state.uniform(size=n*n_components) (:113), Guttman-transform update X = (1/n) B X with B[i,i]+=ratio.sum, B[i,j]=-disparities/dis (:151-155), raw stress = ((dis-disparities)**2).sum()/2 (:147), convergence on relative stress < eps (:160-165), non-metric arm runs IsotonicRegression on the disparities (:130-144). smacof (:187-392): runs _smacof_single n_init times from independent random inits and keeps the lowest-stress result (:348-387), normalized_stress=="auto" -> not metric (:331-332). class MDS(BaseEstimator) (:395-): __init__(n_components=2, *, metric=True, n_init=4, max_iter=300, verbose=0, eps=1e-3, n_jobs=None, random_state=None, dissimilarity="euclidean", normalized_stress="auto"); fit/fit_transform compute dissimilarity_matrix_ (euclidean_distances(X) or X for precomputed) then call smacof; fitted attrs embedding_, stress_, dissimilarity_matrix_, n_iter_.
ferrolearn-module: ferrolearn-decomp/src/mds.rs
parity-ops: MDS
crosslink-issue: 1451
-->

## Summary

`ferrolearn-decomp/src/mds.rs` mirrors the ESTIMATOR NAME of scikit-learn's
`MDS` (`sklearn/manifold/_mds.py`, `class MDS(BaseEstimator)` `:395`) but
implements a DIFFERENT algorithm family member: **classical (metric) MDS /
Principal Coordinates Analysis (PCoA)** — a CLOSED-FORM eigendecomposition of the
double-centred Gram matrix. scikit-learn's `MDS` is **SMACOF** (Scaling by
MAjorizing a COmplicated Function): iterative stress majorization
(`_smacof_single` `:22-167`) from `n_init` independent numpy-`RandomState`
random initial configurations (`smacof` `:348-387`), keeping the lowest-stress
run. These are NOT the same algorithm, so element-wise coordinate parity with
sklearn is structurally IMPOSSIBLE (different method + numpy RNG init + arbitrary
rotation/reflection of the stress minimizer). This is the same heuristic-vs-
reference carve-out class as bayesian_gmm (#1067).

It exposes the unfitted `MDS { n_components: usize, dissimilarity: Dissimilarity }`
with `Dissimilarity::{Euclidean, Precomputed}` (`mds.rs:42`), `new(n_components)`
defaulting to `Euclidean` (`:72`), builder `with_dissimilarity` (`:81`), accessors
`n_components()` / `dissimilarity()`, and the fitted `FittedMDS { embedding_:
Array2<f64>, stress_: f64 }` with accessors `embedding()` (`:117`) / `stress()`
(`:123`). It is re-exported at the crate root (`pub use mds::{Dissimilarity,
FittedMDS, MDS}` in `ferrolearn-decomp/src/lib.rs:95`). Its core helpers
`classical_mds` (`:195`), `eigh_faer` (`:177`) and `pairwise_sq_distances`
(`:133`) are `pub(crate)` and have a SECOND non-test production consumer:
`isomap.rs` (`use crate::mds::{classical_mds, eigh_faer, pairwise_sq_distances}`,
`isomap.rs:38`; `classical_mds(&geo_sq, self.n_components)`, `isomap.rs:339`).

**This unit ships the defining MDS property — DISTANCE PRESERVATION — not
coordinate parity with SMACOF.** `classical_mds` (`:195`) double-centres the
squared-distance matrix `B = -0.5·(D² − rowmean − colmean + grandmean)`
(`:220-225`), eigendecomposes `B` via `eigh_faer` (`:228`, faer self-adjoint
eigen), sorts eigenvalues DESCENDING (`:231-236`), and builds the embedding
`X_k = v_k·sqrt(max(λ_k, 0))` (`:239-248`). By the classical-MDS theorem the
resulting embedding pairwise distances reconstruct the input Euclidean
dissimilarities EXACTLY at full rank (`n_components ≥ rank`) and form the best
low-rank L2 approximation otherwise — that is what ANY MDS algorithm (including
SMACOF) is trying to achieve, and it is verifiable against the input distance
matrix oracle directly (REQ-2). Fit quality is reported via `kruskal_stress`
(`:151`), Kruskal's Stress-1 `sqrt(Σ(d_orig − d_emb)² / Σ d_orig²)`.

At baseline `b8ff9e61` the headline DISTANCE-PRESERVATION claim (REQ-2), the
embedding SHAPE/determinism (REQ-3), the Kruskal Stress-1 computation (REQ-4),
and the scoped error/parameter contracts (REQ-5) are SHIPPED. EXACT SMACOF
coordinate parity (REQ-1) is a NOT-STARTED **carve-out** (different algorithm +
numpy RNG, no committed failing test per R-DEFER-3). The SMACOF algorithm itself
(REQ-6), non-metric MDS (REQ-7), `normalized_stress`/raw-SSR `stress_`/
`max_iter`/`eps` (REQ-8), the sklearn precomputed+attrs+`fit_transform` surface
(REQ-9), the PyO3 binding (REQ-10), and the ferray substrate (REQ-11) are
NOT-STARTED — 4 SHIPPED / 7 NOT-STARTED.

`MDS` / `FittedMDS` / `Dissimilarity` are existing pub APIs whose non-test
consumers are the crate re-export (`lib.rs:95`, boundary public API,
grandfathered S5/R-DEFER-1) and `isomap.rs` (for the `classical_mds` /
`eigh_faer` / `pairwise_sq_distances` helpers). There is **no PyO3 binding**
(`grep -rln MDS ferrolearn-python/` is empty) and **no `fit_transform` /
`Transform`** (sklearn's `MDS` is also fit-only — it has no out-of-sample
`transform`, only `fit_transform`).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# REQ-2 (HEADLINE, SHIPPED) — DISTANCE-PRESERVATION oracle. The defining goal of ANY MDS is to
# reproduce the input pairwise distances. The verifiable oracle is the INPUT distance matrix itself
# (euclidean_distances), NOT sklearn's SMACOF coordinates (a different algorithm). Classical MDS at
# full rank (n_components = data dim) reconstructs the input distances exactly (stress ~ 0).
python3 -c "import numpy as np; from sklearn.metrics import euclidean_distances
X=np.array([[0.,0.],[3.,0.],[0.,4.],[3.,4.]]); D=euclidean_distances(X); print(np.round(D,8).tolist())"
# -> [[0.0, 3.0, 4.0, 5.0], [3.0, 0.0, 5.0, 4.0], [4.0, 5.0, 0.0, 3.0], [5.0, 4.0, 3.0, 0.0]]
#    Replicating ferrolearn's classical_mds (double-centre B = -0.5 J D2 J, eigh, X_k = v_k*sqrt(max(lam_k,0)))
#    at n_components=2 (full rank for this 2D data) gives embedding pairwise distances == this matrix to
#    max abs err 1.78e-15. ferrolearn's classical_mds (mds.rs:195) computes EXACTLY this embedding.

# REQ-1 (CARVE-OUT) — sklearn MDS is SMACOF: DIFFERENT coordinates than classical MDS, and DIFFERENT
# run-to-run (numpy RandomState init + arbitrary rotation/reflection of the stress minimizer).
python3 -c "import numpy as np; from sklearn.manifold import MDS
X=np.array([[0.,0.],[3.,0.],[0.,4.],[3.,4.]])
m0=MDS(n_components=2, random_state=0).fit_transform(X); m1=MDS(n_components=2, random_state=1).fit_transform(X)
print('seed0', np.round(m0,4).tolist()); print('seed1', np.round(m1,4).tolist())"
# -> seed0 [[2.4726, 0.3938], [1.0729, -2.25], [-1.0762, 2.2583], [-2.4693, -0.4021]]
# -> seed1 [[1.3512, -2.093], [-1.6339, -1.8984], [1.6401, 1.8914], [-1.3574, 2.0999]]
#    seed0 != seed1 (RNG init + rotation) and BOTH differ from classical MDS's eigenvector coords.
#    EXACT coordinate parity is structurally impossible -> REQ-1 carve-out (#1452), NOT a parity claim.
#    (Both SMACOF and classical MDS DO preserve the input distances -> that common property is REQ-2.)

# REQ-6 / REQ-8 — sklearn ctor defaults + stress_ DEFINITION (raw sum-of-squared-residuals, /2),
# which is a DIFFERENT quantity than ferrolearn's Kruskal Stress-1.
python3 -c "from sklearn.manifold import MDS; import numpy as np
m=MDS(); print('defaults: n_init', m.n_init, 'max_iter', m.max_iter, 'eps', m.eps, 'metric', m.metric,
      'dissimilarity', m.dissimilarity, 'normalized_stress', m.normalized_stress)
X=np.array([[0.,0.],[3.,0.],[0.,4.],[3.,4.]]); m2=MDS(n_components=2, random_state=0).fit(X)
print('fitted attrs:', [a for a in ('embedding_','stress_','dissimilarity_matrix_','n_iter_') if hasattr(m2,a)])
print('stress_ (raw SSR, NOT Kruskal-1):', m2.stress_)"
# -> defaults: n_init 4 max_iter 300 eps 0.001 metric True dissimilarity euclidean normalized_stress auto
# -> fitted attrs: ['embedding_', 'stress_', 'dissimilarity_matrix_', 'n_iter_']
# -> stress_ (raw SSR, NOT Kruskal-1): 0.0009673767781287632
#    sklearn stress_ = ((dis - disparities)**2).sum()/2 (_mds.py:147), a raw residual sum (units of
#    distance^2, unnormalized). ferrolearn's stress() is Kruskal Stress-1 = sqrt(Sum(d_o-d_e)^2 / Sum d_o^2),
#    a DIFFERENT, dimensionless [0,1] quantity (mds.rs:151). The numbers are not comparable -> REQ-8 flag.
```

## Requirements

- REQ-1: **EXACT coordinate parity with sklearn SMACOF (NOT-STARTED, CARVE-OUT;
  #1452).** sklearn's `MDS.fit_transform` runs `smacof` (`_mds.py:187`), which
  runs `_smacof_single` (`:22-167`) `n_init=4` times from independent random
  inits `X = random_state.uniform(size=n*n_components)` (`:113`) via the Guttman
  transform (`:151-155`) and keeps the lowest-stress result (`:363-365`).
  ferrolearn implements classical MDS (closed-form eigendecomposition,
  `classical_mds` `:195`). The two algorithms produce DIFFERENT coordinates, and
  SMACOF differs run-to-run by the numpy RNG init and an arbitrary
  rotation/reflection of the stress minimizer (Probe REQ-1). Element-wise
  coordinate parity is therefore structurally IMPOSSIBLE; this is excluded from
  scope as a carve-out (no committed failing test, R-DEFER-3). The verifiable
  COMMON property both algorithms target — distance preservation — is REQ-2.

- REQ-2: **Classical-MDS DISTANCE PRESERVATION (R-DEV-1, the headline
  requirement; SHIPPED).** `classical_mds` (`:195`) double-centres the
  squared-distance matrix into `B = -0.5·(D² − rowmean[i] − colmean[j] +
  grandmean)` (`:220-225`), eigendecomposes `B` (`eigh_faer`, `:228`), sorts
  eigenvalues descending (`:231-236`), and emits `X_k = v_k·sqrt(max(λ_k, 0))`
  for the top `n_components` (`:239-248`). By the classical-MDS / PCoA theorem the
  embedding pairwise Euclidean distances reconstruct the input Euclidean
  dissimilarities EXACTLY at full rank (`n_components ≥ rank` of the centred Gram
  matrix) and form the best rank-`n_components` L2 approximation otherwise. This
  is the defining goal of MDS and is verifiable against the INPUT distance matrix
  oracle (Probe REQ-2): on the square fixture `[[0,0],[3,0],[0,4],[3,4]]` the
  input distances `euclidean_distances(X)` are `[[0,3,4,5],...]` and the
  classical-MDS embedding at `n_components=2` reconstructs them to max abs err
  1.78e-15 (stress ≈ 0). Mirrors the OBJECTIVE that sklearn's SMACOF
  (`_smacof_single`) minimizes (the stress between embedding distances and the
  dissimilarities), achieved here in closed form.

- REQ-3: **Structural embedding SHAPE + determinism (scoped; SHIPPED).** `fit`
  (`:271`) returns `FittedMDS` whose `embedding()` is `Array2<f64>` of shape
  `(n_samples, n_components)` (`embedding = Array2::zeros((n, n_comp))` with
  `n_comp = n_components.min(n)`, `:239-240`), finite, and DETERMINISTIC given the
  input (no RNG — the eigendecomposition path is fully deterministic, unlike
  sklearn's RNG-seeded SMACOF). Mirrors the output shape `(n_samples,
  n_components)` of `smacof` (`_mds.py:80-81`, `:287-288`).

- REQ-4: **Kruskal Stress-1 computation (scoped; SHIPPED, DEFINITION FLAG).**
  `kruskal_stress` (`:151`) computes Kruskal's Stress-1 `sqrt(Σ_{i<j}(d_orig −
  d_emb)² / Σ_{i<j} d_orig²)` over the upper triangle (`:155-173`), returning `0`
  when the denominator is `0`. `FittedMDS::stress()` (`:123`) exposes it.
  **FLAG (DIVERGENT DEFINITION):** sklearn's `stress_` is the RAW
  sum-of-squared-residuals `((dis − disparities)**2).sum() / 2` (`_mds.py:147`),
  an unnormalized quantity in distance² units (Probe REQ-6/8: `0.000967...`), NOT
  Kruskal Stress-1. The two `stress`/`stress_` numbers are NOT comparable; sklearn
  only returns Stress-1 under `normalized_stress` with `metric=False` (`:148-149`).
  Matching sklearn's raw `stress_` is REQ-8.

- REQ-5: **Error / parameter contracts (scoped; SHIPPED).** `fit` (`:271`) returns
  `InvalidParameter { name: "n_components", reason: "must be at least 1" }` for
  `n_components == 0` (`:272-277`), `InvalidParameter` for `n_components >
  n_samples` (`:289-297`, `:317-325`), `InsufficientSamples { required: 2 }` for
  `< 2` samples (`:282-288`, `:310-316`), and (Precomputed) `ShapeMismatch` for a
  non-square input (`:301-308`). **FLAG (candidate DIVs):** sklearn's
  `_parameter_constraints` admit `n_components` `Interval(Integral, 1, None)`
  (`:174`) — there is NO upper `n_components <= n_samples` bound in sklearn (SMACOF
  embeds into any dimension), so ferrolearn's `n_components > n_samples` rejection
  is stricter; sklearn raises `InvalidParameterError`/`ValueError`, not the
  `FerroError::InvalidParameter` ABI.

- REQ-6: **SMACOF algorithm — sklearn's ACTUAL metric MDS (R-DEV-2; NOT-STARTED).**
  sklearn's metric MDS is the SMACOF iterative stress-majorization loop
  (`_smacof_single` `:22-167`): random init (`:113`), euclidean distances of the
  current config (`:128`), raw stress (`:147`), Guttman-transform update `X =
  (1/n)·B·X` with `B[i,i] += ratio.sum(axis=1)`, `B[i,j] = -disparities/dis`
  (`:151-155`), convergence on relative stress `< eps` (`:160-165`), wrapped by
  `smacof` running `n_init` independent restarts and keeping the lowest-stress
  result (`:348-387`). ferrolearn has no SMACOF loop, no `n_init`, no
  `random_state`, no Guttman transform — it is closed-form classical MDS. This is
  the big algorithm gap underlying the REQ-1 carve-out.

- REQ-7: **Non-metric MDS `metric=False` (R-DEV-2; NOT-STARTED).** With
  `metric=False`, `_smacof_single` runs an `IsotonicRegression` monotonic
  regression of the disparities against the current distances each iteration
  (`:130-144`, treating `0` dissimilarities as missing), rescaling the disparities
  (`:142-144`). ferrolearn has no `metric` parameter and no isotonic-regression
  path — it computes only metric (classical) MDS.

- REQ-8: **`normalized_stress` + raw-SSR `stress_` + `max_iter`/`eps`
  (R-DEV-2; NOT-STARTED).** sklearn exposes `normalized_stress="auto"` (`→ not
  metric`, `:331-332`), reports `stress_` as the raw residual sum
  `((dis-disparities)**2).sum()/2` (`:147`) or Stress-1 under
  `normalized_stress` (`:148-149`), and is governed by `max_iter=300` / `eps=1e-3`
  convergence (`:160-165`). ferrolearn has no `normalized_stress` / `max_iter` /
  `eps` fields, and `stress()` is Kruskal Stress-1 (a DIFFERENT quantity than
  sklearn's raw `stress_`; Probe REQ-6/8) — matching sklearn's `stress_` value is
  out of scope here.

- REQ-9: **sklearn precomputed SMACOF + fitted attrs + `fit_transform`
  (R-DEV-3; NOT-STARTED).** sklearn's `MDS.fit_transform` exposes
  `dissimilarity_matrix_` (`euclidean_distances(X)`, or `X` for
  `dissimilarity='precomputed'`), `stress_`, `n_iter_`, and `embedding_`, and
  `fit`/`fit_transform` run SMACOF on the dissimilarities. ferrolearn's
  `Precomputed` ENUM VARIANT EXISTS and its classical-MDS fit path IS wired
  (`:300-328`: square check, square the input distances, `classical_mds`), so
  REQ-2 distance-preservation DOES cover the classical precomputed path. But
  ferrolearn exposes only `embedding()` / `stress()` (Kruskal-1) — there is no
  `dissimilarity_matrix_` accessor, no `n_iter_` (no iterations — closed form),
  and the SMACOF-on-precomputed behaviour + the `dissimilarity_matrix_` /
  `n_iter_` attrs + a `fit_transform` method are NOT-STARTED.

- REQ-10: **PyO3 binding (R-DEFER-1/3; NOT-STARTED).** `import ferrolearn`
  exposing a registered `MDS` marshalling `fit`/`fit_transform` and exposing
  `embedding_`/`stress_` — the boundary CPython consumer. Absent (`grep -rln MDS
  ferrolearn-python/` is empty).

- REQ-11: **ferray substrate (R-SUBSTRATE; NOT-STARTED).** `mds.rs` computes on
  `ndarray::Array2<f64>` + `Vec<f64>` and eigendecomposes via faer
  (`eigh_faer` → `faer::Mat::self_adjoint_eigen`, `:177-190`), not `ferray-core`
  arrays / `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). Fixture `sq` = `[[0,0],[3,0],[0,4],
[3,4]]` (a 2D point set whose input distance matrix is `[[0,3,4,5],
[3,0,5,4],[4,5,0,3],[5,4,3,0]]`, the 3-4-5 rectangle), full-rank at
`n_components=2`.

- AC-1 (REQ-1, CARVE-OUT): `MDS(n_components=2, random_state=0).fit_transform(sq)`
  and `random_state=1` give DIFFERENT coordinates (Probe REQ-1: seed0
  `[[2.4726,0.3938],...]` ≠ seed1 `[[1.3512,-2.093],...]`), and both differ from
  the classical-MDS eigenvector coordinates. Exact coordinate parity is impossible
  (different algorithm + RNG); NOT pinned by any test (R-DEFER-3).

- AC-2 (REQ-2, SHIPPED — distance preservation): the classical-MDS embedding of
  `sq` at `n_components=2` (`MDS::new(2).fit(&sq)`) has pairwise Euclidean
  distances equal to the INPUT distance oracle `euclidean_distances(sq) =
  [[0,3,4,5],[3,0,5,4],[4,5,0,3],[5,4,3,0]]` to max abs err < 1e-8 (observed
  ~1.78e-15), i.e. `stress() ≈ 0`. Pinned by `test_mds_preserves_distances` and
  `test_mds_perfect_embedding_low_stress` (`stress() < 0.1`) in `mds.rs`, and the
  collinear best-low-rank case by `test_mds_collinear_data` (`n_components=1`
  recovers evenly-spaced 1D coordinates).

- AC-3 (REQ-3, SHIPPED): `MDS::new(2).fit(&sq).unwrap().embedding()` has shape
  `(4, 2)`, is finite, and is identical across runs (deterministic, no RNG).
  Pinned by `test_mds_basic_embedding_shape` `(4,2)`, `test_mds_1d_embedding`,
  `test_mds_larger_dataset` `(20,2)`, and `conformance_mds`
  (`tests/conformance_wave2.rs:423`, `finite_and_shaped(emb, n, n_components)`).

- AC-4 (REQ-4, SHIPPED scoped): `MDS::new(2).fit(&sq).unwrap().stress()` is `≥ 0`
  and `< 0.1` on the full-rank fixture (Kruskal Stress-1). Pinned by
  `test_mds_stress_non_negative` and `test_mds_perfect_embedding_low_stress`.
  FLAG: this is NOT sklearn's `stress_` (raw SSR `0.000967...`, Probe REQ-6/8) —
  the definitions differ (REQ-8).

- AC-5 (REQ-5, SHIPPED scoped): `fit` returns `Err` for `n_components=0`,
  `n_components > n_samples`, `n_samples < 2`, and (Precomputed) a non-square
  input. Pinned by `test_mds_invalid_n_components_zero`,
  `test_mds_invalid_n_components_too_large`, `test_mds_insufficient_samples`,
  `test_mds_precomputed_not_square`. FLAG: sklearn has no upper `n_components`
  bound and raises `InvalidParameterError`/`ValueError`, not
  `FerroError::InvalidParameter`.

- AC-6 (REQ-6/7/8, DIVERGES): `MDS()` defaults `n_init=4, max_iter=300, eps=1e-3,
  metric=True, normalized_stress='auto'` (Probe REQ-6/8) and runs SMACOF;
  ferrolearn has none of these fields and runs closed-form classical MDS.

- AC-7 (REQ-9, partial): the classical PRECOMPUTED path is wired
  (`test_mds_precomputed` builds a distance matrix and fits `(4,2)`), so REQ-2
  distance preservation covers it; but `hasattr(MDS().fit(sq),
  'dissimilarity_matrix_')` / `'n_iter_')` is True in sklearn while
  `FittedMDS` exposes only `embedding()` / `stress()` — no `dissimilarity_matrix_`,
  no `n_iter_`, no `fit_transform`, no SMACOF.

- AC-8 (REQ-10/11): no `ferrolearn.MDS` (no binding, `grep -rln MDS
  ferrolearn-python/` empty); the module imports `ndarray` + faer
  (`self_adjoint_eigen`), not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `MDS` / `FittedMDS` / `Dissimilarity` are existing pub APIs;
non-test consumers are the crate re-export (`lib.rs:95`, boundary public API,
grandfathered S5/R-DEFER-1) and `isomap.rs` (the `classical_mds` / `eigh_faer` /
`pairwise_sq_distances` helpers, `isomap.rs:38,339`). Cites use symbol anchors
(ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed sklearn
1.5.2, run from `/tmp`. **ALGORITHM DIVERGENCE (R-HONEST-3):** ferrolearn is
classical (metric) MDS / PCoA (closed-form eigendecomposition); sklearn `MDS` is
SMACOF (iterative, numpy-RNG-initialized, multi-restart). EXACT coordinate parity
(REQ-1) is structurally impossible → carve-out (#1452), no committed failing
test (R-DEFER-3). The load-bearing SHIPPED claim is REQ-2 DISTANCE PRESERVATION
(the defining property of any MDS), verified against the INPUT distance matrix
oracle — the least-confident SHIPPED claim only in that its exactness is scoped
to full rank (`n_components ≥ rank`); at lower rank it is the best-L2
approximation, not exact. #1451 is this doc's crosslink tracking issue. Count: 4
SHIPPED (REQ-2..5) / 7 NOT-STARTED (REQ-1, REQ-6..11).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (EXACT SMACOF coordinate parity) | NOT-STARTED | open prereq blocker **#1452** (CARVE-OUT, R-DEFER-3). sklearn `MDS.fit_transform` runs `smacof` (`_mds.py:187`) → `_smacof_single` (`:22-167`): SMACOF iterative stress majorization from `n_init=4` random inits `X = random_state.uniform(size=n*n_components)` (`:113`), Guttman update `X = (1/n)·B·X` (`:151-155`), keeping the lowest-stress run (`:363-365`). ferrolearn is closed-form classical MDS (`classical_mds`, `mds.rs:195`). DIFFERENT algorithm → DIFFERENT coordinates, AND SMACOF differs run-to-run by numpy RNG + arbitrary rotation/reflection (Probe REQ-1: seed0 `[[2.4726,0.3938],...]` ≠ seed1 `[[1.3512,-2.093],...]`). Element-wise parity is structurally IMPOSSIBLE — carved out (same class as bayesian_gmm #1067); no committed failing test (R-DEFER-3). The verifiable common target is REQ-2. |
| REQ-2 (classical-MDS DISTANCE PRESERVATION) | SHIPPED | `classical_mds` (`mds.rs:195`) double-centres `B = -0.5·(sq_dist[[i,j]] − row_means[i] − col_means[j] + grand_mean)` (`:220-225`), eigendecomposes via `eigh_faer` (`:228`), sorts eigenvalues DESCENDING (`indices.sort_by(... eigenvalues[b].partial_cmp(&eigenvalues[a]) ...)`, `:231-236`), and builds `embedding[[i,k]] = eigenvectors[[i,idx]] · eigenvalues[idx].max(0.0).sqrt()` (`:241-247`) — i.e. `X_k = v_k·sqrt(max(λ_k,0))`. By the classical-MDS/PCoA theorem the embedding pairwise distances reconstruct the input Euclidean dissimilarities EXACTLY at full rank (`n_components ≥ rank`), best-L2 otherwise — the defining goal of MDS, and the OBJECTIVE sklearn's SMACOF (`_smacof_single`, `_mds.py:147`) minimizes. Non-test consumers: crate re-export (`lib.rs:95`) AND `isomap.rs` (`classical_mds(&geo_sq, self.n_components)`, `isomap.rs:339`, applying the SAME helper to geodesic distances). Pin (AC-2, R-CHAR-3): on `sq=[[0,0],[3,0],[0,4],[3,4]]`, `euclidean_distances(sq) = [[0,3,4,5],[3,0,5,4],[4,5,0,3],[5,4,3,0]]` (Probe REQ-2); the classical-MDS embedding at `n_components=2` reproduces it to max abs err 1.78e-15 (stress ≈ 0). Verification: `cargo test -p ferrolearn-decomp mds` → `test_mds_preserves_distances`, `test_mds_perfect_embedding_low_stress` (`stress()<0.1`), `test_mds_collinear_data` PASS. **Scope (R-HONEST-3): EXACT at full rank; best-low-rank L2 approximation below rank — NOT coordinate parity with SMACOF (REQ-1).** |
| REQ-3 (structural embedding SHAPE + determinism) | SHIPPED | `fn fit` (`mds.rs:271`) returns `FittedMDS { embedding_, stress_ }` whose `embedding()` (`:117`) is `Array2<f64>` of shape `(n_samples, n_components)` — `let n_comp = n_components.min(n); let mut embedding = Array2::zeros((n, n_comp))` (`:239-240`), finite, and DETERMINISTIC given the input (the `eigh_faer` path uses no RNG, unlike sklearn's RNG-seeded SMACOF). Mirrors the `(n_samples, n_components)` output shape of `smacof` (`_mds.py:80-81`,`:287-288`). Non-test consumers: crate re-export (`lib.rs:95`) and `isomap.rs:339` (`classical_mds` returns the same `(n, n_comp)` embedding). Verification: `cargo test -p ferrolearn-decomp mds` (`test_mds_basic_embedding_shape` `(4,2)`, `test_mds_1d_embedding`, `test_mds_larger_dataset` `(20,2)`) + `conformance_mds` (`tests/conformance_wave2.rs:423`, `finite_and_shaped(emb, x.nrows(), n_components, "MDS.embedding")`) PASS. **Scope: SHAPE/finiteness/determinism, NOT value parity (REQ-1).** |
| REQ-4 (Kruskal Stress-1 computation) | SHIPPED | `fn kruskal_stress` (`mds.rs:151`) computes `sqrt(numerator/denominator)` with `numerator += (d_orig − d_embed)²`, `denominator += d_orig²` over the upper triangle `j in (i+1)..n` (`:155-173`), returning `0.0` when `denominator == 0` (`:169-173`) — Kruskal's Stress-1 `sqrt(Σ(d_o−d_e)² / Σ d_o²)`. Exposed by `FittedMDS::stress()` (`:123`). Non-test consumer: called inside `classical_mds` (`let stress = kruskal_stress(sq_dist, &embedding)`, `:250`), reached through the crate re-export and `isomap.rs` (which discards it as `_stress`, `isomap.rs:339`). Verification: `cargo test -p ferrolearn-decomp mds` (`test_mds_stress_non_negative`, `test_mds_perfect_embedding_low_stress`) PASS. **FLAG (DIVERGENT DEFINITION):** sklearn `stress_` is RAW SSR `((dis-disparities)**2).sum()/2` (`_mds.py:147`), an unnormalized distance² quantity (Probe REQ-6/8: `0.000967...`), NOT Kruskal-1 — the two are not comparable. Matching sklearn's raw `stress_` is REQ-8. |
| REQ-5 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`mds.rs:271`) returns `Err(InvalidParameter { name: "n_components", reason: "must be at least 1" })` for `n_components == 0` (`:272-277`), `Err(InvalidParameter { name: "n_components", reason: format!("n_components ({}) exceeds n_samples ({})", ...) })` for `n_components > n_samples` (`:289-297`, `:317-325`), `Err(InsufficientSamples { required: 2, actual: n, context: "MDS::fit requires at least 2 samples" })` for `< 2` samples (`:282-288`, `:310-316`), and `Err(ShapeMismatch { ... context: "MDS with Precomputed dissimilarity requires a square matrix" })` for a non-square Precomputed input (`:301-308`). Non-test consumer: these guards protect every instance reached through the crate re-export (`lib.rs:95`) and via `isomap.rs`. Verification: `cargo test -p ferrolearn-decomp mds` (`test_mds_invalid_n_components_zero`, `_invalid_n_components_too_large`, `_insufficient_samples`, `_precomputed_not_square`) PASS. **FLAG (candidate DIVs):** sklearn `_parameter_constraints` admit `n_components` `Interval(Integral, 1, None)` (`_mds.py:174`) — NO upper `n_components <= n_samples` bound (SMACOF embeds into any dimension), so ferrolearn is stricter; sklearn raises `InvalidParameterError`/`ValueError`, not `FerroError::InvalidParameter`. |
| REQ-6 (SMACOF algorithm: iterative majorization + n_init + random_state) | NOT-STARTED | open prereq blocker **#1453**. sklearn's metric MDS IS SMACOF (`_smacof_single`, `_mds.py:22-167`): random init (`:113`), Guttman-transform update `X = (1/n)·B·X` with `B[i,i]+=ratio.sum`, `B[i,j]=-disparities/dis` (`:151-155`), convergence on relative stress `< eps` (`:160-165`), wrapped by `smacof` running `n_init=4` independent restarts and keeping the lowest-stress result (`:348-387`). ferrolearn has NO SMACOF loop, no Guttman transform, no `n_init`/`random_state` — `classical_mds` (`mds.rs:195`) is closed-form eigendecomposition. This is the big algorithm gap behind the REQ-1 carve-out. |
| REQ-7 (`metric=False` non-metric MDS via isotonic regression) | NOT-STARTED | open prereq blocker **#1454**. sklearn's `metric=False` arm runs `IsotonicRegression().fit_transform` on the disparities vs current distances each SMACOF iteration (`_mds.py:130-144`, treating `0` dissimilarities as missing) then rescales them (`:142-144`). ferrolearn has no `metric` parameter and no isotonic-regression path — only metric (classical) MDS. |
| REQ-8 (`normalized_stress` + raw-SSR `stress_` + `max_iter`/`eps`) | NOT-STARTED | open prereq blocker **#1455**. sklearn `normalized_stress="auto"` resolves to `not metric` (`_mds.py:331-332`), reports `stress_` as raw `((dis-disparities)**2).sum()/2` (`:147`) or Stress-1 under `normalized_stress` (`:148-149`), governed by `max_iter=300`/`eps=1e-3` convergence (`:160-165`). ferrolearn has no `normalized_stress`/`max_iter`/`eps` fields, and `stress()` is Kruskal Stress-1 — a DIFFERENT quantity than sklearn's raw `stress_` (Probe REQ-6/8: `0.000967...`). |
| REQ-9 (sklearn precomputed SMACOF + `dissimilarity_matrix_`/`n_iter_` attrs + `fit_transform`) | NOT-STARTED | open prereq blocker **#1456**. The classical PRECOMPUTED path IS wired in ferrolearn (`fn fit` `Dissimilarity::Precomputed` arm: square check `:301-308`, square the distances `x.mapv(|v| v*v)` `:327`, `classical_mds` `:331`) and `test_mds_precomputed` fits `(4,2)` — so REQ-2 distance preservation covers the classical precomputed case. But sklearn exposes `dissimilarity_matrix_`, `stress_`, `n_iter_`, `embedding_` and runs SMACOF on the dissimilarities, whereas `FittedMDS` exposes only `embedding()`/`stress()` (Kruskal-1): NO `dissimilarity_matrix_`, NO `n_iter_` (closed form — no iterations), NO `fit_transform` method, and the SMACOF-on-precomputed behaviour is absent. |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker **#1457**. `grep -rln MDS ferrolearn-python/` is EMPTY — no `_RsMDS`, so `import ferrolearn` cannot reach `MDS`/`fit_transform`/`embedding_`/`stress_`. The only non-test consumers of `fit`/`embedding()` are the crate re-export (`lib.rs:95`) and `isomap.rs`. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#1458**. `mds.rs` computes on `ndarray::Array2<f64>` + `Vec<f64>` and eigendecomposes via faer (`eigh_faer` → `faer::Mat::from_fn` + `self_adjoint_eigen(faer::Side::Lower)`, `mds.rs:177-190`), not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`mds.rs` follows the unfitted/fitted split (CLAUDE.md naming): `MDS {
n_components: usize, dissimilarity: Dissimilarity }` (`new(n_components)`
defaulting `Dissimilarity::Euclidean`, builder `with_dissimilarity`, accessors
`n_components()` / `dissimilarity()`) → `Fit<Array2<f64>, ()>` → `FittedMDS {
embedding_: Array2<f64>, stress_: f64 }` (accessors `embedding()` / `stress()`).
`Dissimilarity` is a `Copy` enum with `Euclidean` / `Precomputed` (`:42`). The
path is `f64`-only (operates on `Array2<f64>`, not generic `F`); `fit` returns
`Result<_, FerroError>` (R-CODE-2). There is no `Predict`/`Transform` and no
`fit_transform` — `MDS` is fit-only, mirroring sklearn (which has only
`fit`/`fit_transform`, no out-of-sample `transform`).

**Fit path (`fn fit` `:271`).** Validates `n_components != 0`, then per-mode:
`Euclidean` → require `n_samples >= 2` and `n_components <= n_samples`, build the
squared-distance matrix `pairwise_sq_distances(x)` (`:133`, O(n²d) double loop);
`Precomputed` → require a square matrix, `>= 2` rows, `n_components <= n`, then
square the supplied distances (`x.mapv(|v| v*v)`, `:327`). Both feed
`classical_mds(&sq_dist, n_components)` (`:331`).

**Core `classical_mds` (`:195`) — the algorithm.** (1) Double-centre: compute
`row_means[i]`, `col_means[j]`, `grand_mean` of `D²` (`:203-218`), then `B[i,j] =
-0.5·(D²[i,j] − row_means[i] − col_means[j] + grand_mean)` (`:220-225`) — i.e.
`B = -0.5·J·D²·J` with `J = I − (1/n)·11ᵀ`. (2) `eigh_faer(&B)` (`:228`) →
`(eigenvalues, eigenvectors)` via faer's `self_adjoint_eigen` (`:177-190`). (3)
Sort eigenvalue INDICES descending (`:231-236`). (4) Build the embedding `X_k =
v_k·sqrt(max(λ_k, 0))` for the top `n_comp = n_components.min(n)` (`:239-248`,
clamping negative eigenvalues to `0`). (5) `kruskal_stress(sq_dist, &embedding)`
(`:250`). This is the closed-form solution of the classical-MDS objective: the
top eigenvectors of the double-centred Gram matrix ARE the distance-preserving
coordinates (REQ-2).

**The algorithm divergence vs sklearn (REQ-1 carve-out, #1452).** sklearn's
`MDS` (`_mds.py:395`) is NOT classical MDS — it is SMACOF
(`_smacof_single` `:22-167`): starting from a random configuration `X =
random_state.uniform(size=n*n_components)` (`:113`), it iterates the Guttman
transform `X ← (1/n)·B·X` (`:151-155`) to descend the raw stress
`((dis-disparities)²).sum()/2` (`:147`) until the relative stress change `< eps`
(`:160-165`), and `smacof` (`:187`) repeats this from `n_init=4` independent
random inits keeping the lowest-stress run (`:348-387`). Because SMACOF is an
iterative minimizer of a rotation-invariant stress with RNG init, its output
coordinates are an arbitrary rotation/reflection of one local optimum and DIFFER
run-to-run (Probe REQ-1: `random_state=0` ≠ `random_state=1`) and from classical
MDS's eigenvector coordinates. Exact element-wise parity is therefore IMPOSSIBLE
(same carve-out class as bayesian_gmm #1067, R-DEFER-3). What BOTH algorithms
share — and what is verifiable — is that the embedding distances reproduce the
input dissimilarities (REQ-2), checked against the INPUT distance matrix oracle,
not against SMACOF coordinates.

**Cross-module reuse.** `classical_mds`, `eigh_faer`, and `pairwise_sq_distances`
are `pub(crate)` and consumed by `isomap.rs` (`use crate::mds::{classical_mds,
eigh_faer, pairwise_sq_distances}`, `isomap.rs:38`), which applies classical MDS
to a GEODESIC distance matrix (`classical_mds(&geo_sq, self.n_components)`,
`isomap.rs:339`) — Isomap is "classical MDS on geodesic distances", so this is a
production (non-test) consumer of the exact REQ-2/REQ-3/REQ-4 code paths.

**sklearn (target contract).** `class MDS(BaseEstimator)` (`_mds.py:395`) takes
`__init__(n_components=2, *, metric=True, n_init=4, max_iter=300, verbose=0,
eps=1e-3, n_jobs=None, random_state=None, dissimilarity="euclidean",
normalized_stress="auto")`. `fit_transform` computes `dissimilarity_matrix_ =
euclidean_distances(X)` (or `X` for `dissimilarity='precomputed'`), then calls
`smacof(...)` (`:187`) → `_smacof_single` (`:22`), exposing fitted attrs
`embedding_`, `stress_` (raw SSR or Stress-1), `dissimilarity_matrix_`,
`n_iter_`. `metric=False` adds an `IsotonicRegression` disparity step
(`:130-144`).

**The remaining gap.** ferrolearn ships the DISTANCE-PRESERVATION property
(REQ-2, the defining MDS goal, verified vs the input distance oracle), the
embedding SHAPE/determinism (REQ-3), the Kruskal Stress-1 computation (REQ-4),
and the scoped error contracts (REQ-5). It DIVERGES on / lacks: exact SMACOF
coordinate parity (REQ-1, carve-out #1452 — different algorithm + RNG); the
SMACOF iterative algorithm itself (REQ-6); non-metric MDS (REQ-7); the
`normalized_stress` / raw-SSR `stress_` / `max_iter` / `eps` surface (REQ-8); the
sklearn precomputed-SMACOF + `dissimilarity_matrix_`/`n_iter_` attrs +
`fit_transform` (REQ-9); the PyO3 binding (REQ-10); and the ferray substrate
(REQ-11). This is a **SHIPPED-PROPERTY / partial-surface** unit — the headline
classical-MDS distance preservation ships; SMACOF parity is structurally carved
out (4 SHIPPED / 7 NOT-STARTED).

## Verification

Library crate (green at baseline `b8ff9e61`):
```bash
cargo test -p ferrolearn-decomp mds                          # in-module + conformance_mds
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-2..5: `test_mds_preserves_distances`,
`test_mds_perfect_embedding_low_stress` (`stress()<0.1`), `test_mds_collinear_data`
(REQ-2 distance preservation / best-low-rank); `test_mds_basic_embedding_shape`
`(4,2)`, `test_mds_1d_embedding`, `test_mds_larger_dataset` `(20,2)` (REQ-3
shape); `test_mds_stress_non_negative` (REQ-4); `test_mds_invalid_n_components_zero`,
`test_mds_invalid_n_components_too_large`, `test_mds_insufficient_samples`,
`test_mds_precomputed`, `test_mds_precomputed_not_square`, `test_mds_getters`
(REQ-5/REQ-9-precomputed-path), plus the module doctest. The conformance suite
adds `conformance_mds` (`tests/conformance_wave2.rs:423`) — a SHAPE/finiteness
check (`finite_and_shaped`), NOT value parity (which is impossible vs SMACOF).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-2 distance
oracle (R-CHAR-3 expected values) and the REQ-1 carve-out evidence:
```bash
# REQ-2 (SHIPPED) — input distance matrix oracle; classical MDS at full rank reproduces it to ~1e-15:
python3 -c "import numpy as np; from sklearn.metrics import euclidean_distances
X=np.array([[0.,0.],[3.,0.],[0.,4.],[3.,4.]]); print(np.round(euclidean_distances(X),8).tolist())"
# -> [[0.0, 3.0, 4.0, 5.0], [3.0, 0.0, 5.0, 4.0], [4.0, 5.0, 0.0, 3.0], [5.0, 4.0, 3.0, 0.0]]

# REQ-1 (CARVE-OUT) — sklearn MDS=SMACOF: coords differ run-to-run AND from classical MDS:
python3 -c "import numpy as np; from sklearn.manifold import MDS
X=np.array([[0.,0.],[3.,0.],[0.,4.],[3.,4.]])
print('seed0', np.round(MDS(n_components=2,random_state=0).fit_transform(X),4).tolist())
print('seed1', np.round(MDS(n_components=2,random_state=1).fit_transform(X),4).tolist())"
# -> seed0 [[2.4726,0.3938],...] ; seed1 [[1.3512,-2.093],...]  (different -> no parity, #1452)

# REQ-8 flag — sklearn stress_ is raw SSR (0.000967...), NOT ferrolearn's Kruskal Stress-1:
python3 -c "import numpy as np; from sklearn.manifold import MDS
X=np.array([[0.,0.],[3.,0.],[0.,4.],[3.,4.]]); print(MDS(n_components=2,random_state=0).fit(X).stress_)"
# -> 0.0009673767781287632
```
The REQ-2 distance-preservation pin lives in `mds.rs` (`test_mds_preserves_distances`,
`test_mds_perfect_embedding_low_stress`) — asserting the embedding distances match
the input-distance oracle. NO SMACOF coordinate-parity test exists or should exist
(REQ-1 carve-out, R-DEFER-3 — a different algorithm).

ferrolearn-python (REQ-10 binding parity, after #1457 lands):
```bash
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_mds.py -q
```
asserting `ferrolearn.MDS` exists and exposes `embedding_`/`stress_`, with its
distance-preservation property validated against `euclidean_distances` (NOT
against `sklearn.manifold.MDS` coordinates — the algorithm carve-out).

## Blockers

(#1451 is this doc's crosslink tracking issue. The blockers below are the open
work items; none are filed by this doc — markdown only.)

- **#1452** — REQ-1 (EXACT SMACOF coordinate parity): CARVE-OUT. ferrolearn is
  classical MDS (closed-form eigendecomposition, `classical_mds` `mds.rs:195`);
  sklearn `MDS` is SMACOF (iterative stress majorization + numpy-RNG init +
  multi-restart, `_smacof_single` `_mds.py:22-167`). Different algorithm + RNG +
  arbitrary rotation → exact coordinate parity is structurally impossible; no
  committed failing test (R-DEFER-3, same class as bayesian_gmm #1067). The
  verifiable common target is REQ-2 distance preservation.
- **#1453** — REQ-6: implement the SMACOF loop (random init `_mds.py:113`,
  Guttman transform `:151-155`, `eps`/`max_iter` convergence `:160-165`) with
  `n_init` restarts + `random_state` (`smacof` `:348-387`) — sklearn's actual
  metric MDS.
- **#1454** — REQ-7: add `metric=False` non-metric MDS via `IsotonicRegression`
  on the disparities (`_mds.py:130-144`).
- **#1455** — REQ-8: add `normalized_stress` + sklearn's raw-SSR `stress_`
  definition (`_mds.py:147-149`) + `max_iter`/`eps`.
- **#1456** — REQ-9: add SMACOF-on-precomputed + the fitted attrs
  `dissimilarity_matrix_` / `n_iter_` / `embedding_` + a `fit_transform` method
  matching sklearn `MDS`.
- **#1457** — REQ-10: add `_RsMDS` to `ferrolearn-python` (fit / fit_transform /
  embedding_ / stress_) — the boundary CPython consumer.
- **#1458** — REQ-11: migrate `mds.rs` off `ndarray` / faer (`eigh_faer`,
  `self_adjoint_eigen`) to `ferray-core` / `ferray::linalg` (R-SUBSTRATE).
