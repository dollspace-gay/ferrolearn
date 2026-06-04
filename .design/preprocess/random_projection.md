# RandomProjection (Gaussian + Sparse)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 920e75dd
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/random_projection.py  # johnson_lindenstrauss_min_dim (:64-143): n_components >= 4*log(n_samples)/(eps^2/2 - eps^3/3) (:82,142-143), reject eps not in ]0,1[ (:133), n_samples<=0 (:136). _check_density (:146-153): density=="auto" -> 1/sqrt(n_features) (:148-149), reject <=0 or >1 (:151-152). _gaussian_random_matrix (:166-203): components = rng.normal(loc=0, scale=1/sqrt(n_components), size=(n_components,n_features)) (:200-201) i.e. N(0, 1/n_components) (:171). _sparse_random_matrix (:206-301): s=1/density; entry -sqrt(s)/sqrt(n_components) w.p. 1/(2s)=density/2, 0 w.p. 1-1/s, +sqrt(s)/sqrt(n_components) w.p. density/2 (:213-218); density==1 dense special case rng.binomial(1,0.5)*2-1 scaled 1/sqrt(n_components) (:271-274); else per-COMPONENT n_nonzero_i=rng.binomial(n_features,density) then sample_without_replacement indices then signs, CSR, scaled sqrt(1/density)/sqrt(n_components) (:281-301). BaseRandomProjection ctor (:324-335): __init__(n_components="auto", *, eps=0.1, compute_inverse_components=False, random_state=None); SparseRandomProjection adds density="auto", dense_output=False (:748-765). _compute_inverse_components pinv (:356-361). fit (:363-429): _validate_data float32/float64 (:382-384); n_components=="auto" -> johnson_lindenstrauss_min_dim(n_samples, eps) (:388-391), reject <=0 (:393) or > n_features (:399); else n_components>n_features warns DataDimensionalityWarning (:407-414); components_=_make_random_matrix(n_components_, n_features).astype(X.dtype) (:419-421); inverse_components_ if compute_inverse_components (:423-424); _n_features_out=n_components (:427). inverse_transform (:431-458): X @ inverse_components_.T. transform (SparseRP :810): safe_sparse_dot(X, components_.T, dense_output=self.dense_output).
ferrolearn-module: ferrolearn-preprocess/src/random_projection.rs
parity-ops: GaussianRandomProjection, SparseRandomProjection
crosslink-issue: 1387
-->

## Summary

scikit-learn's `GaussianRandomProjection` and `SparseRandomProjection`
(`random_projection.py:466,607`) reduce dimensionality by multiplying the data by a
RANDOM matrix that preserves pairwise distances in expectation
(Johnson-Lindenstrauss). Gaussian draws entries from `N(0, 1/n_components)`
(`_gaussian_random_matrix:200-201`); Sparse draws Achlioptas/Li `{-sqrt(s)/sqrt(k), 0,
+sqrt(s)/sqrt(k)}` entries with `s=1/density` (`_sparse_random_matrix:213-218,301`).
The free function `johnson_lindenstrauss_min_dim` (`:64`) gives the minimal safe
`n_components` for a target distortion `eps`.

`ferrolearn-preprocess/src/random_projection.rs` ships BOTH transformers over dense
`Array2<F>`: `GaussianRandomProjection<F> { n_components, random_state }` (`new`,
`random_state`) → `FittedGaussianRandomProjection<F> { projection }` (`projection()`),
and `SparseRandomProjection<F> { n_components, density, random_state }` (`new`,
`density`, `random_state`) → `FittedSparseRandomProjection<F> { projection }`
(`projection()`). Both `Fit::fit` (`fn fit in random_projection.rs`) build the
projection matrix with the SAME marginal DISTRIBUTION and scale CONSTANTS as sklearn,
and both `Transform::transform` compute `X @ R` (`x.dot(&projection)`).

This is a **RNG-COUPLED VERIFY-AND-DOCUMENT** unit. The projection matrix is RANDOM:
ferrolearn samples it with the Rust `rand` `SmallRng` PRNG, while sklearn uses numpy's
`RandomState`/Mersenne-Twister stream. **Exact projection-matrix VALUE parity across the
two RNG streams is IMPOSSIBLE** — this is a CARVE-OUT in the SAME class as the
`cluster/kmeans.rs` numpy-RNG carve-outs (#1039). What IS verifiable and SHIPPED: the
projection-matrix **DISTRIBUTION/scale constants** (cited from the sklearn FORMULA as the
oracle, R-CHAR-3), the transform `X @ R` contract, **determinism given a seed**, and the
error contracts. This is a **shipped-partial** unit: **3 SHIPPED** (REQ-1 Gaussian
distribution + transform + determinism, REQ-2 Sparse distribution + transform, REQ-3
scoped error/parameter contracts) / **8 NOT-STARTED** (REQ-4 exact-value RNG-stream
parity CARVE-OUT, REQ-5 Sparse sampling METHOD, REQ-6 `n_components='auto'` +
`johnson_lindenstrauss_min_dim`, REQ-7 `components_` sparse storage + orientation +
`dense_output`, REQ-8 `inverse_transform` + `compute_inverse_components`, REQ-9
`n_components>n_features` warning + sklearn fitted-attr surface, REQ-10 PyO3 binding,
REQ-11 ferray substrate).

## Probes (live sklearn oracle, 1.5.2)

All values below are live output from `python3` against scikit-learn 1.5.2, run from
`/tmp`. Because the per-entry VALUES differ by RNG stream (REQ-4 carve-out), the oracle
here is the DISTRIBUTION/scale **formula** and its CONSTANTS — exactly what ferrolearn
reproduces (R-CHAR-3). They pin the Gaussian scale (REQ-1), the Sparse support/scale and
default density (REQ-2), and the `johnson_lindenstrauss_min_dim` value (NOT-STARTED
REQ-6).

```bash
# PROBE 1 (REQ-1) — Gaussian entries ~ N(0, 1/n_components); scale = 1/sqrt(n_components)
# (_gaussian_random_matrix:200-201, docstring N(0, 1.0/n_components) :171):
python3 -c "import numpy as np; print('gauss scale k=4 =', 1.0/np.sqrt(4))"
#   -> gauss scale k=4 = 0.5
#   ferrolearn `scale = F::one() / F::from(self.n_components).sqrt()` (= 1/sqrt(4) = 0.5),
#   sample N(0,1)*scale -> SAME distribution N(0, 1/4). Empirical var of projection ~ 1/4.

# PROBE 2 (REQ-2) — Sparse nonzero entries are EXACTLY +/- sqrt(1/density)/sqrt(n_components);
# for d=0.5, k=4 -> sqrt(1/0.5)/sqrt(4) = sqrt(2)/2 (_sparse_random_matrix:301):
python3 -c "import numpy as np; d=0.5; k=4; print('sparse support d=0.5 k=4 =', np.sqrt(1/d)/np.sqrt(k))"
#   -> sparse support d=0.5 k=4 = 0.7071067811865476
#   ferrolearn `scale = 1.0/(d*n_components).sqrt()` = 1/sqrt(0.5*4) = 1/sqrt(2) = 0.70710678,
#   nonzero entries +/-scale -> SAME support. (1/sqrt(d*k) == sqrt(1/d)/sqrt(k).)

# PROBE 3 (REQ-2) — default density = 1/sqrt(n_features) (_check_density:148-149); n_features=100 -> 0.1:
python3 -c "import numpy as np; print('default density n_features=100 =', 1/np.sqrt(100))"
#   -> default density n_features=100 = 0.1
#   ferrolearn `density.unwrap_or_else(|| 1.0/(n_features as f64).sqrt())` = 0.1 (== sklearn 'auto').

# PROBE 4 (REQ-6, NOT-STARTED) — johnson_lindenstrauss_min_dim(n_samples, eps)
# = 4*log(n_samples)/(eps^2/2 - eps^3/3) (:82,142-143):
python3 -c "from sklearn.random_projection import johnson_lindenstrauss_min_dim
print('jl(1000, eps=0.1) =', johnson_lindenstrauss_min_dim(1000, eps=0.1))
print('jl(1e6, eps=0.5) =', johnson_lindenstrauss_min_dim(1e6, eps=0.5))"
#   -> jl(1000, eps=0.1) = 5920
#   -> jl(1e6, eps=0.5) = 663
#   ferrolearn has NO johnson_lindenstrauss_min_dim free fn and NO n_components='auto' path.

# PROBE 5 (REQ-4 CARVE-OUT) — sklearn's exact entry VALUES come from numpy RandomState;
# ferrolearn's come from Rust SmallRng -> the two matrices DIFFER element-by-element even
# at the same seed. This is documented as IMPOSSIBLE-to-match, NOT a failing test.
python3 -c "import numpy as np
from sklearn.random_projection import GaussianRandomProjection
g=GaussianRandomProjection(n_components=3, random_state=42).fit(np.ones((10,5)))
print('sklearn components_[0,:] =', np.round(g.components_[0,:],4).tolist())"
#   -> a numpy-RandomState-specific row; ferrolearn's SmallRng row is a DIFFERENT realization
#      of the SAME N(0,1/3) distribution. Distribution matches (REQ-1); exact values do not (REQ-4).
```

## Requirements

- REQ-1: **Gaussian projection DISTRIBUTION + `X @ R` transform contract +
  determinism-given-seed** (HEADLINE, SHIPPED). sklearn's `_gaussian_random_matrix`
  draws `rng.normal(loc=0, scale=1/sqrt(n_components), size=(n_components, n_features))`
  i.e. `N(0, 1/n_components)` (`:200-201`, docstring `:171`); the transformer projects
  via `safe_sparse_dot(X, components_.T)` (`:810` for Sparse, the Gaussian analog being
  `X @ components_.T`). ferrolearn's `fn fit in random_projection.rs` for
  `GaussianRandomProjection` sets `scale = F::one() / F::from(self.n_components).sqrt()`
  (== `1/sqrt(n_components)`), then fills a `(n_features, n_components)` matrix with
  `N(0,1) * scale` via `rand_distr::StandardNormal` + `SmallRng` — the SAME marginal
  distribution `N(0, 1/n_components)`; `fn transform in random_projection.rs` returns
  `x.dot(&projection)` (`X @ R`). ferrolearn stores the matrix in the TRANSPOSE
  orientation `(n_features, n_components)`, which makes `X @ R` transform-EQUIVALENT to
  sklearn's `X @ components_.T` (REQ-7 covers the orientation/storage gap). Determinism:
  a fixed `random_state(seed)` re-seeds `SmallRng::seed_from_u64(seed)`, so identical
  seeds yield identical matrices. **This is DISTRIBUTION + contract + determinism, NOT
  exact-value parity with sklearn's numpy stream** — exact values are the REQ-4
  carve-out. Pinned by the in-module tests. (Probe 1: scale `1/sqrt(4) = 0.5`.)

- REQ-2: **Sparse projection DISTRIBUTION + `X @ R` transform contract** (HEADLINE,
  SHIPPED). sklearn's `_sparse_random_matrix` (with `s = 1/density`) draws each nonzero
  entry as `-sqrt(s)/sqrt(n_components)` w.p. `1/(2s)=density/2`, `0` w.p. `1-1/s`,
  `+sqrt(s)/sqrt(n_components)` w.p. `density/2` (`:213-218`), i.e. nonzero support
  `+/- sqrt(1/density)/sqrt(n_components)` (`:301`), default `density='auto' =
  1/sqrt(n_features)` (`_check_density:148-149`), and a `density==1` dense special case
  giving `+/- 1/sqrt(n_components)` (`:271-274`). ferrolearn's `fn fit in
  random_projection.rs` for `SparseRandomProjection` sets `scale = 1.0/(d *
  n_components).sqrt()` (== `sqrt(1/(d*n_components))` == sklearn's
  `sqrt(1/density)/sqrt(n_components)`), resolves `d = density.unwrap_or(1/sqrt(n_features))`
  (== `'auto'`), and per ENTRY samples a uniform `u`: `u < d/2 -> -scale`, `u < d ->
  +scale`, else `0` — the SAME marginal `{d/2, 1-d, d/2}` distribution with the SAME
  support; `fn transform in random_projection.rs` returns `x.dot(&projection)`. The
  `density==1` case falls out of the same per-entry rule (`scale = 1/sqrt(n_components)`,
  `P(+/-)=0.5` each, `P(0)=0`), matching `:271-274` marginally. **Distributional /
  scale / support parity, NOT exact-value parity** (REQ-4) and NOT the sklearn sampling
  METHOD (REQ-5). Pinned by the in-module tests. (Probe 2: support `1/sqrt(0.5*4) =
  0.70710678`; Probe 3: default density `0.1`.)

- REQ-3: **Error / parameter contracts** (scoped, SHIPPED). `fn fit in
  random_projection.rs` (both transformers) returns `InvalidParameter { name:
  "n_components" }` when `n_components == 0` (mirroring sklearn's `n_components` >= 1
  constraint, `_parameter_constraints` `:314-315`, and `_check_input_size` `n_components
  <= 0` reject `:158-160`), and `InsufficientSamples { required: 1, actual: 0 }` when
  `x.nrows() == 0`. `SparseRandomProjection::fit` additionally rejects `d <= 0 || d > 1`
  with `InvalidParameter { name: "density" }` (mirroring `_check_density` `:151-152`,
  `"Expected density in range ]0, 1]"`). `fn transform in random_projection.rs` returns
  `ShapeMismatch` when `x.ncols()` differs from the fitted feature count; the unfitted
  `Transform for GaussianRandomProjection` / `for SparseRandomProjection` returns
  `InvalidParameter` (the "must fit first" guard, mirroring `check_is_fitted` `:450`).
  Scoped to the contracts ferrolearn enforces over the dense API (sklearn's `eps`-range
  guard and the `n_components <= 0` post-`auto` check are part of REQ-6).

- REQ-4: **EXACT projection-matrix VALUE parity (numpy `RandomState` stream)**
  (NOT-STARTED, CARVE-OUT). sklearn fills the matrix from numpy's `RandomState`
  Mersenne-Twister stream (`rng.normal` `:200`, `rng.binomial` `:283,294`); ferrolearn
  fills it from Rust `rand`'s `SmallRng`. Even at the SAME integer seed the two PRNGs
  emit DIFFERENT number streams, so the projection matrices DIFFER element-by-element
  (Probe 5). Reproducing sklearn's EXACT entries would require a numpy-`RandomState`-bit-
  faithful PRNG (Mersenne-Twister + numpy's normal/binomial transforms) in ferrolearn —
  the SAME class as the `cluster/kmeans.rs` numpy-RNG carve-outs (#1039). The
  DISTRIBUTION, scale, and support ARE matched (REQ-1, REQ-2). Open prereq blocker
  `#1388`. Per R-DEFER-3 there is **NO committed failing test** pinning bit-exact value
  parity — the blocker is the open work item, not a red test.

- REQ-5: **Sparse sampling METHOD (per-row binomial + `sample_without_replacement` vs
  per-entry Bernoulli)** (NOT-STARTED). sklearn's `_sparse_random_matrix` does NOT sample
  each entry independently: for each component row it draws `n_nonzero_i =
  rng.binomial(n_features, density)`, then picks `n_nonzero_i` column indices via
  `sample_without_replacement`, then signs (`:281-301`) — giving a FIXED nonzero count
  per row and a CSR structure. ferrolearn samples each `(i,j)` entry INDEPENDENTLY
  (per-entry Bernoulli with `P(nonzero)=d`), so the per-row nonzero count is random
  (Binomial-distributed) rather than drawn-then-placed. The two procedures are
  MARGINALLY equivalent (same per-entry distribution, REQ-2) but JOINTLY different
  (different row-wise sparsity structure) and RNG-coupled — this distinct mechanism folds
  into the REQ-4 value-parity carve-out but is a separate sampling-procedure gap. Open
  prereq blocker `#1389`.

- REQ-6: **`n_components='auto'` + `eps` + `johnson_lindenstrauss_min_dim` free fn**
  (NOT-STARTED). sklearn defaults `n_components="auto"` and, at `fit`, sets
  `n_components_ = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)`
  (`:388-391`), rejecting the result if `<= 0` (`:393-397`) or `> n_features`
  (`:399-405`); the free function computes `4*log(n_samples)/(eps^2/2 - eps^3/3)`
  (`:82,142-143`) and rejects `eps` outside `]0,1[` (`:133`) / `n_samples <= 0`
  (`:136`). ferrolearn requires an EXPLICIT `n_components: usize` (`new(n_components)`),
  has NO `eps` field, NO `n_components='auto'` path, and NO `johnson_lindenstrauss_min_dim`
  free function. Open prereq blocker `#1390`. (Probe 4: `jl(1000, eps=0.1) = 5920`.)

- REQ-7: **`components_` SPARSE storage (Sparse) + `(n_components, n_features)`
  orientation + `dense_output`** (NOT-STARTED). sklearn stores `components_` as shape
  `(n_components, n_features)` (`:419`), and for `SparseRandomProjection` it is a scipy
  CSR SPARSE matrix (`_sparse_random_matrix` returns `sp.csr_matrix` `:297-301`), with
  transform `safe_sparse_dot(X, components_.T, dense_output=self.dense_output)` (`:810`)
  where `dense_output` (default `False`, `:754`) controls whether the output is dense.
  ferrolearn stores `projection` as a DENSE `Array2<F>` in the TRANSPOSE orientation
  `(n_features, n_components)` (transform-equivalent for `X @ R`), has NO sparse storage
  for `FittedSparseRandomProjection` (the sparse matrix is materialized dense), and NO
  `dense_output` flag. Open prereq blocker `#1391`.

- REQ-8: **`inverse_transform` + `compute_inverse_components`** (NOT-STARTED). sklearn's
  `inverse_transform` (`:431-458`) maps projected data back via `X @
  inverse_components_.T` where `inverse_components_` is `linalg.pinv(components_)`
  (`_compute_inverse_components` `:356-361`), computed eagerly at `fit` when the ctor flag
  `compute_inverse_components=True` (`:423-424`, `:329`) or lazily per call otherwise.
  ferrolearn's `FittedGaussianRandomProjection` / `FittedSparseRandomProjection` expose
  only `projection()` and a `Transform` impl — NO `inverse_transform`, NO pseudo-inverse,
  NO `compute_inverse_components` ctor flag. Open prereq blocker `#1392`.

- REQ-9: **`n_components > n_features` warning + sklearn fitted-attr surface +
  `get_feature_names_out`** (NOT-STARTED). sklearn warns `DataDimensionalityWarning` when
  `n_components > n_features` (`:407-414`), exposes the fitted attributes `n_components_`
  (`:389,416`), `components_` (`:419`), `n_features_in_` (via `_validate_data`), and —
  for Sparse — `density_` (`:788`), plus `get_feature_names_out` (via
  `ClassNamePrefixFeaturesOutMixin`, `_n_features_out = n_components` `:427`). ferrolearn
  emits NO warning when `n_components > n_features`, names the accessor `projection()`
  (not `components_`/`n_components_`/`density_`), and has NO `get_feature_names_out`. Open
  prereq blocker `#1393`.

- REQ-10: **PyO3 binding** (NOT-STARTED). There is no `_RsGaussianRandomProjection` /
  `_RsSparseRandomProjection` CPython binding — `grep -rn
  "RandomProjection\|random_projection" ferrolearn-python/src` finds none — so neither
  transformer is reachable from Python. Open prereq blocker `#1394`.

- REQ-11: **ferray substrate** (NOT-STARTED). Build the projection matrix and compute the
  `X @ R` transform over `ferray-core` arrays + `ferray::random` (the `numpy.random`
  analog) rather than `ndarray::Array2<F>` + `rand`/`rand_distr` `SmallRng` +
  `num_traits::Float` (R-SUBSTRATE-1/2). Notably, `ferray::random` is the substrate that
  would also enable numpy-`RandomState`-faithful sampling (REQ-4). Open prereq blocker
  `#1395`.

## Acceptance criteria

- AC-1 (REQ-1): `GaussianRandomProjection::<f64>::new(5).random_state(42).fit([[..]];
  10x50)` then `transform` yields a `(10, 5)` matrix (`X @ R`); the projection entries
  have empirical variance `~ 1/n_components` (scale `1/sqrt(n_components)`, Probe 1:
  `1/sqrt(4)=0.5`); fitting twice with the SAME seed yields identical output. Pinned by
  `test_gaussian_rp_output_shape`, `test_gaussian_rp_deterministic`,
  `test_gaussian_rp_fit_transform`, `test_gaussian_rp_f32` (in-module). NOT exact-value
  parity with sklearn's numpy stream (REQ-4).

- AC-2 (REQ-2): `SparseRandomProjection::<f64>::new(10).random_state(42).fit(X; 5x100)`
  with default `density=1/sqrt(100)=0.1` produces a matrix that is `>50%` zeros (Probe 3);
  with `density(0.5)`, `n_components=4` the nonzero entries equal `+/- 0.70710678` (Probe
  2, `1/sqrt(0.5*4)`); `transform` yields `X @ R`. Pinned by `test_sparse_rp_output_shape`,
  `test_sparse_rp_sparsity`, `test_sparse_rp_custom_density`, `test_sparse_rp_deterministic`,
  `test_sparse_rp_fit_transform`, `test_sparse_rp_f32` (in-module). Distribution/support,
  NOT exact values (REQ-4) and NOT the sampling method (REQ-5).

- AC-3 (REQ-3): `new(0).fit(X)` → `Err(InvalidParameter{name:"n_components"})`;
  `fit(zeros((0,10)))` → `Err(InsufficientSamples)`; `SparseRandomProjection::new(5)
  .density(0.0).fit(X)` → `Err(InvalidParameter{name:"density"})`; a fitted handle's
  `transform` on a wrong column count → `Err(ShapeMismatch)`; the unfitted
  `*.transform` → `Err(InvalidParameter)`. Pinned by `test_gaussian_rp_zero_components`,
  `test_gaussian_rp_empty_input`, `test_gaussian_rp_shape_mismatch`,
  `test_sparse_rp_zero_components`, `test_sparse_rp_invalid_density`,
  `test_sparse_rp_empty_input`, `test_sparse_rp_shape_mismatch`.

- AC-4 (REQ-4, CARVE-OUT): for a fixed seed, `GaussianRandomProjection(random_state=42)
  .components_` (sklearn, numpy `RandomState`) and ferrolearn's `SmallRng` projection
  DIFFER element-by-element (Probe 5) — exact-value parity is IMPOSSIBLE across the two
  PRNG streams. NOT-STARTED; per R-DEFER-3 there is NO committed failing test pinning
  bit-exact value equality. The DISTRIBUTION matches (AC-1).

- AC-5 (REQ-5): sklearn's `SparseRandomProjection` produces a CSR matrix with a FIXED
  per-row nonzero count `binomial(n_features, density)` placed by
  `sample_without_replacement` (`:281-301`); ferrolearn's per-entry Bernoulli gives a
  random per-row count. Same marginal entry distribution (AC-2), different joint row
  structure.

- AC-6 (REQ-6): `GaussianRandomProjection()` (default `n_components='auto'`) sets
  `n_components_ = johnson_lindenstrauss_min_dim(n_samples, eps=0.1)` (`:388-391`, Probe 4
  `jl(1000)=5920`); ferrolearn requires an explicit `usize` and has no
  `johnson_lindenstrauss_min_dim`.

- AC-7 (REQ-7): `SparseRandomProjection().fit(X).components_` is a scipy CSR sparse matrix
  of shape `(n_components, n_features)` and `transform` honors `dense_output` (`:810`);
  ferrolearn stores a dense `(n_features, n_components)` `Array2<F>` with no
  `dense_output`.

- AC-8 (REQ-8): `rp.inverse_transform(rp.transform(X))` reconstructs via the
  pseudo-inverse of `components_` (`:431-458`); ferrolearn has no `inverse_transform`.

- AC-9 (REQ-9): fitting with `n_components > n_features` raises a
  `DataDimensionalityWarning` (`:407-414`) and a fitted estimator exposes `n_components_`,
  `components_`, `n_features_in_`, `density_` (Sparse), `get_feature_names_out`;
  ferrolearn emits no warning and exposes only `projection()`.

- AC-10 (REQ-10): a CPython `GaussianRandomProjection` / `SparseRandomProjection` binding
  fits/transforms from Python; no such binding exists in `ferrolearn-python`.

- AC-11 (REQ-11): the matrix generation + `X @ R` runs on `ferray-core` arrays +
  `ferray::random` rather than `ndarray` + `rand`/`rand_distr` + `num_traits::Float`.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (Gaussian distribution `N(0,1/n_components)` + `X@R` transform + determinism; HEADLINE) | SHIPPED | impl `fn fit in random_projection.rs` (`GaussianRandomProjection`) sets `let scale = F::one() / F::from(self.n_components).unwrap().sqrt();` (== `1/sqrt(n_components)`) and fills `projection` with `F::from(normal.sample(&mut rng)).unwrap() * scale` where `normal = StandardNormal` — the SAME distribution `N(0, 1/n_components)` as sklearn `rng.normal(loc=0.0, scale=1.0/np.sqrt(n_components), ...)` (`random_projection.py:200-201`, docstring `N(0, 1.0/n_components)` `:171`); `fn transform in random_projection.rs` returns `Ok(x.dot(&self.projection))` (`X @ R`, sklearn `X @ components_.T` `:810`). Determinism: `SmallRng::seed_from_u64(seed)` from `random_state(seed)` → identical matrices for equal seeds. DISTRIBUTION + contract + determinism, NOT exact-value parity (REQ-4 carve-out). Scale live-confirmed (Probe 1: `1/sqrt(4)=0.5`). Non-test consumer: boundary re-export `pub use random_projection::{FittedGaussianRandomProjection, FittedSparseRandomProjection, GaussianRandomProjection, SparseRandomProjection};` at `lib.rs:167-170` (grandfathered S5 / R-DEFER-1 boundary estimator API) + `PipelineTransformer<F> for GaussianRandomProjection` / `FittedPipelineTransformer<F> for FittedGaussianRandomProjection` (`fn fit_pipeline` / `fn transform_pipeline in random_projection.rs`). Verification: `cargo test -p ferrolearn-preprocess` → `test_gaussian_rp_output_shape` (`(10,5)`), `test_gaussian_rp_deterministic` (equal-seed identity), `test_gaussian_rp_fit_transform`, `test_gaussian_rp_f32`. |
| REQ-2 (Sparse distribution support `+/-sqrt(1/(d·n_components))`, probs `{d/2,1-d,d/2}`, default density, `d=1` case; + `X@R` transform; HEADLINE) | SHIPPED | impl `fn fit in random_projection.rs` (`SparseRandomProjection`) sets `let scale = F::from(1.0 / (d * self.n_components as f64).sqrt()).unwrap();` (== `sqrt(1/(d*n_components))` == sklearn `sqrt(1/density)/sqrt(n_components)` `:301`), resolves `let d = self.density.unwrap_or_else(\|\| 1.0 / (n_features as f64).sqrt());` (== `'auto'` `_check_density:148-149`), and per entry samples uniform `u`: `if u < d/2.0 { *v = scale.neg() } else if u < d { *v = scale }` else `0` — the SAME marginal `{d/2, 1-d, d/2}` over support `+/-scale` as sklearn `_sparse_random_matrix:213-218`; the `d==1` case reduces to `+/-1/sqrt(n_components)` w.p. `0.5` each (sklearn dense special case `:271-274`); `fn transform in random_projection.rs` returns `Ok(x.dot(&self.projection))`. Distribution/support/default-density, NOT exact values (REQ-4) and NOT the sampling METHOD (REQ-5). Support live-confirmed (Probe 2: `1/sqrt(0.5*4)=0.70710678`; Probe 3: default `0.1`). Non-test consumer: boundary re-export at `lib.rs:167-170` + `PipelineTransformer<F> for SparseRandomProjection` / `FittedPipelineTransformer<F> for FittedSparseRandomProjection` (`fn fit_pipeline` / `fn transform_pipeline in random_projection.rs`). Verification: `cargo test -p ferrolearn-preprocess` → `test_sparse_rp_output_shape`, `test_sparse_rp_sparsity` (`>50%` zeros at default density), `test_sparse_rp_custom_density`, `test_sparse_rp_deterministic`, `test_sparse_rp_fit_transform`, `test_sparse_rp_f32`. |
| REQ-3 (error / parameter contracts, scoped) | SHIPPED (scoped) | impl `fn fit in random_projection.rs` (both) returns `Err(FerroError::InvalidParameter { name: "n_components", reason: "must be >= 1" })` when `self.n_components == 0` (sklearn `n_components >= 1` constraint `:314-315`, `_check_input_size` reject `:158-160`) and `Err(FerroError::InsufficientSamples { required: 1, actual: 0, .. })` when `x.nrows() == 0`; `SparseRandomProjection::fit` adds `Err(FerroError::InvalidParameter { name: "density", reason: "must be in (0, 1], got {d}" })` when `d <= 0.0 \|\| d > 1.0` (sklearn `_check_density` `"Expected density in range ]0, 1]"` `:151-152`); `fn transform in random_projection.rs` returns `Err(FerroError::ShapeMismatch { .. context: "Fitted{Gaussian,Sparse}RandomProjection::transform" })` when `x.ncols() != self.projection.nrows()`; the unfitted `Transform for {Gaussian,Sparse}RandomProjection` returns `Err(FerroError::InvalidParameter { name, reason: "projection must be fitted before calling transform; use fit() first" })` (mirroring `check_is_fitted` `:450`). Scoped to the dense API (the `eps`-range and post-`auto` `n_components<=0` guards are REQ-6). Non-test consumer: boundary re-export at `lib.rs:167-170`. Verification: `cargo test -p ferrolearn-preprocess` → `test_gaussian_rp_zero_components`, `test_gaussian_rp_empty_input`, `test_gaussian_rp_shape_mismatch`, `test_sparse_rp_zero_components`, `test_sparse_rp_invalid_density`, `test_sparse_rp_empty_input`, `test_sparse_rp_shape_mismatch` green. |
| REQ-4 (EXACT projection-matrix VALUE parity, numpy `RandomState` stream) | NOT-STARTED | carve-out blocker `#1388` (NO committed failing test per R-DEFER-3). `fn fit in random_projection.rs` samples from Rust `rand::rngs::SmallRng` (`SmallRng::seed_from_u64` / `from_os_rng`), whereas sklearn samples from numpy `RandomState`'s Mersenne-Twister stream (`rng.normal` `:200`, `rng.binomial` `:283,294`). The two PRNGs emit DIFFERENT number streams at the same integer seed, so the projection matrices differ element-by-element (Probe 5). Bit-exact value parity needs a numpy-`RandomState`-faithful PRNG — SAME class as the `cluster/kmeans.rs` numpy-RNG carve-outs (#1039). The DISTRIBUTION/scale/support ARE matched (REQ-1, REQ-2). |
| REQ-5 (Sparse sampling METHOD: per-row binomial + `sample_without_replacement` vs per-entry Bernoulli) | NOT-STARTED | open prereq blocker `#1389`. `fn fit in random_projection.rs` (`SparseRandomProjection`) samples each `(i,j)` entry INDEPENDENTLY via per-entry uniform → Bernoulli `P(nonzero)=d`, so the per-row nonzero count is random. sklearn instead draws `n_nonzero_i = rng.binomial(n_features, density)` per component then `sample_without_replacement(n_features, n_nonzero_i)` indices then signs, building a CSR matrix with a FIXED per-row count (`random_projection.py:281-301`). Marginally equivalent (REQ-2), jointly different + RNG-coupled — folds into REQ-4's carve-out as a distinct mechanism. |
| REQ-6 (`n_components='auto'` + `eps` + `johnson_lindenstrauss_min_dim`) | NOT-STARTED | open prereq blocker `#1390`. `GaussianRandomProjection<F> { n_components, random_state }` / `SparseRandomProjection<F> { n_components, density, random_state }` require an EXPLICIT `n_components: usize` (`new(n_components)`), have NO `eps` field, NO `'auto'` path, and there is NO `johnson_lindenstrauss_min_dim` free fn. sklearn defaults `n_components="auto"` → `johnson_lindenstrauss_min_dim(n_samples, eps)` = `4*log(n_samples)/(eps^2/2-eps^3/3)` (`random_projection.py:82,142-143,388-391`), rejecting `<=0` (`:393`) / `>n_features` (`:399`) / `eps` outside `]0,1[` (`:133`). (Probe 4: `jl(1000,eps=0.1)=5920`.) |
| REQ-7 (`components_` SPARSE storage + `(n_components,n_features)` orientation + `dense_output`) | NOT-STARTED | open prereq blocker `#1391`. `FittedSparseRandomProjection<F> { projection: Array2<F> }` stores a DENSE matrix in TRANSPOSE orientation `(n_features, n_components)` (transform-equivalent for `X @ R`) with NO `dense_output`. sklearn stores `components_` as `(n_components, n_features)` (`:419`), a scipy CSR SPARSE matrix for Sparse (`sp.csr_matrix` `:297-301`), and `transform` is `safe_sparse_dot(X, components_.T, dense_output=self.dense_output)` (`:810`, default `dense_output=False` `:754`). |
| REQ-8 (`inverse_transform` + `compute_inverse_components`) | NOT-STARTED | open prereq blocker `#1392`. `FittedGaussianRandomProjection<F>` / `FittedSparseRandomProjection<F>` expose only `projection()` + a `Transform` impl — NO `inverse_transform`, NO pseudo-inverse, NO `compute_inverse_components` ctor flag. sklearn's `inverse_transform` (`:431-458`) is `X @ inverse_components_.T` with `inverse_components_ = linalg.pinv(components_)` (`_compute_inverse_components` `:356-361`), eager at `fit` if `compute_inverse_components=True` (`:423-424`). |
| REQ-9 (`n_components>n_features` warning + `n_components_`/`components_`/`n_features_in_`/`density_` attrs + `get_feature_names_out`) | NOT-STARTED | open prereq blocker `#1393`. `fn fit in random_projection.rs` emits NO warning when `n_components > n_features`; the fitted types name the accessor `projection()` (not `components_`/`n_components_`/`density_`) and provide NO `get_feature_names_out`. sklearn warns `DataDimensionalityWarning` (`:407-414`), and exposes `n_components_` (`:416`), `components_` (`:419`), `n_features_in_`, `density_` (`:788`), `get_feature_names_out` (`ClassNamePrefixFeaturesOutMixin`, `_n_features_out` `:427`). |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker `#1394`. No `_RsGaussianRandomProjection` / `_RsSparseRandomProjection` CPython binding exists — `grep -rn "RandomProjection\|random_projection" ferrolearn-python/src` finds none — so neither transformer is reachable from Python. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker `#1395`. `fn fit in random_projection.rs` builds the matrix on `ndarray::Array2<F>` via `rand::rngs::SmallRng` + `rand_distr::{StandardNormal, Uniform}` + `num_traits::Float`, and `transform` uses `ndarray`'s `x.dot(&projection)` — not `ferray-core` arrays + `ferray::random` (R-SUBSTRATE-1/2). `ferray::random` is also the substrate that would enable the numpy-`RandomState`-faithful sampling of REQ-4. |

## Architecture

**ferrolearn (existing).** `random_projection.rs` exposes two unfitted transformers.
`GaussianRandomProjection<F> { n_components: usize, random_state: Option<u64> }`
(`new(n_components)`, `random_state(seed)`) → fitted
`FittedGaussianRandomProjection<F> { projection: Array2<F> }` (`projection()`).
`SparseRandomProjection<F> { n_components: usize, density: Option<f64>, random_state:
Option<u64> }` (`new(n_components)`, `density(d)`, `random_state(seed)`) → fitted
`FittedSparseRandomProjection<F> { projection: Array2<F> }` (`projection()`). Both
`Fit<Array2<F>, ()>` (`fn fit in random_projection.rs`) reject `n_components == 0`
(`InvalidParameter`) and `x.nrows() == 0` (`InsufficientSamples`); Sparse additionally
resolves `d = density.unwrap_or(1/sqrt(n_features))` and rejects `d <= 0 || d > 1`
(`InvalidParameter`). The matrix is allocated as `Array2::zeros((n_features,
n_components))` (the TRANSPOSE of sklearn's `(n_components, n_features)`, making `X @ R`
transform-equivalent to sklearn's `X @ components_.T`) and filled entry-by-entry:
Gaussian draws `StandardNormal * (1/sqrt(n_components))`; Sparse draws a uniform `u` and
emits `-scale` (`u<d/2`), `+scale` (`u<d`), or `0`, with `scale =
sqrt(1/(d*n_components))`. The PRNG is `SmallRng`, seeded from `random_state` or
`from_os_rng()`. Both `Transform<Array2<F>>` (`fn transform in random_projection.rs`)
check the column count (`ShapeMismatch`) then return `x.dot(&projection)`. The unfitted
`Transform` impls are error stubs (`InvalidParameter`) satisfying the `FitTransform:
Transform` supertrait; `FitTransform` wraps fit→transform; `PipelineTransformer<F>` /
`FittedPipelineTransformer<F>` impls wire both into ferrolearn pipelines. The boundary
re-export at `lib.rs:167-170` (`pub use random_projection::{...}`) plus the pipeline
impls are the non-test production consumers pinning REQ-1 / REQ-2 / REQ-3.

**sklearn (target contract).** `BaseRandomProjection(TransformerMixin, BaseEstimator,
ClassNamePrefixFeaturesOutMixin)` (`random_projection.py:304`) takes
`__init__(n_components="auto", *, eps=0.1, compute_inverse_components=False,
random_state=None)` (`:324-335`); `SparseRandomProjection` adds `density="auto",
dense_output=False` (`:748-765`). `fit` (`:363-429`) validates float32/float64 data
(`:382-384`), resolves `n_components_` — either `johnson_lindenstrauss_min_dim(n_samples,
eps)` when `'auto'` (`:388-391`, rejecting `<=0`/`>n_features`) or the explicit value
(warning `DataDimensionalityWarning` if `> n_features`, `:407-414`) — then builds
`components_ = _make_random_matrix(n_components_, n_features)` of shape `(n_components,
n_features)` (`:419`): Gaussian `rng.normal(loc=0, scale=1/sqrt(n_components))`
(`_gaussian_random_matrix:200-201`), Sparse the Achlioptas/Li CSR matrix
(`_sparse_random_matrix:206-301`, support `+/-sqrt(1/density)/sqrt(n_components)`, probs
`{density/2, 1-density, density/2}`, per-row `binomial`+`sample_without_replacement`
sampling, `density==1` dense special case `:271-274`). If `compute_inverse_components`,
`inverse_components_ = pinv(components_)` (`:423-424,356-361`). `transform`
(SparseRP `:810`) is `safe_sparse_dot(X, components_.T, dense_output=self.dense_output)`;
`inverse_transform` (`:431-458`) is `X @ inverse_components_.T`. The free function
`johnson_lindenstrauss_min_dim` (`:64-143`) computes the safe minimal `n_components`.

**The gap.** ferrolearn matches sklearn on the *projection-matrix DISTRIBUTION, scale,
and support* (REQ-1 Gaussian `N(0,1/n_components)`, REQ-2 Sparse
`+/-sqrt(1/(d*n_components))` with probs `{d/2,1-d,d/2}` and default density
`1/sqrt(n_features)`), the *`X @ R` transform contract*, *determinism given a seed*, and
the scoped *error/parameter contracts* (REQ-3). The HEADLINE carve-out is REQ-4: **exact
projection-matrix VALUE parity is IMPOSSIBLE** — ferrolearn uses Rust `SmallRng`, sklearn
uses numpy `RandomState`; the two PRNG streams diverge bit-for-bit at the same seed (SAME
class as the `cluster/kmeans.rs` numpy-RNG carve-outs #1039). NOT-STARTED with NO
committed failing test per R-DEFER-3 — the blocker is the open work item, not a red test.
The remaining gaps are sampling-procedure / surface / config: the Sparse per-row
`binomial`+`sample_without_replacement` sampling METHOD (REQ-5), `n_components='auto'` +
`eps` + `johnson_lindenstrauss_min_dim` (REQ-6), `components_` CSR sparse storage +
`(n_components,n_features)` orientation + `dense_output` (REQ-7), `inverse_transform` +
`compute_inverse_components` (REQ-8), the `n_components>n_features` warning + sklearn
fitted-attr names + `get_feature_names_out` (REQ-9), the PyO3 binding (REQ-10), and the
non-ferray substrate (REQ-11). This is a **shipped-partial** unit (3 SHIPPED /
8 NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1 Gaussian distribution + transform +
determinism, REQ-2 Sparse distribution + transform, REQ-3 scoped error contracts):

```bash
# Consumer / module wiring check:
grep -n "pub mod random_projection" ferrolearn-preprocess/src/lib.rs        # :114
grep -n "pub use random_projection::" ferrolearn-preprocess/src/lib.rs      # :167 boundary re-export consumer

# REQ-1 / REQ-2 / REQ-3 (in-module tests):
cargo test -p ferrolearn-preprocess random_projection
#   REQ-1: test_gaussian_rp_output_shape ((10,5)), test_gaussian_rp_deterministic (equal-seed identity),
#          test_gaussian_rp_fit_transform, test_gaussian_rp_f32
#   REQ-2: test_sparse_rp_output_shape, test_sparse_rp_sparsity (>50% zeros at default density),
#          test_sparse_rp_custom_density, test_sparse_rp_deterministic, test_sparse_rp_fit_transform,
#          test_sparse_rp_f32
#   REQ-3: test_gaussian_rp_zero_components, test_gaussian_rp_empty_input, test_gaussian_rp_shape_mismatch,
#          test_sparse_rp_zero_components, test_sparse_rp_invalid_density, test_sparse_rp_empty_input,
#          test_sparse_rp_shape_mismatch
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# REQ-1 oracle gate — Gaussian scale constant (DISTRIBUTION oracle, NOT bit-exact values, R-CHAR-3):
python3 -c "import numpy as np; print('gauss scale k=4 =', 1.0/np.sqrt(4))"
#   -> gauss scale k=4 = 0.5
#   ferrolearn `scale = 1/sqrt(n_components)` = 0.5; N(0,1)*scale ~ N(0, 1/4). Empirical var ~ 0.25.

# REQ-2 oracle gate — Sparse support + default density (DISTRIBUTION oracle):
python3 -c "import numpy as np
print('sparse support d=0.5 k=4 =', np.sqrt(1/0.5)/np.sqrt(4))
print('default density n_features=100 =', 1/np.sqrt(100))"
#   -> sparse support d=0.5 k=4 = 0.7071067811865476
#   -> default density n_features=100 = 0.1
#   ferrolearn nonzero entries = +/-1/sqrt(0.5*4) = +/-0.70710678; default d = 1/sqrt(100) = 0.1.

# REQ-4 CARVE-OUT (documented, NO failing test) — sklearn numpy-RandomState values differ from SmallRng:
python3 -c "import numpy as np
from sklearn.random_projection import GaussianRandomProjection
g=GaussianRandomProjection(n_components=3, random_state=42).fit(np.ones((10,5)))
print('sklearn components_[0,:] =', np.round(g.components_[0,:],4).tolist())"
#   -> a numpy-specific realization; ferrolearn's SmallRng row is a DIFFERENT draw of the SAME N(0,1/3).
#   (REQ-4 carve-out #1388; SAME class as cluster/kmeans numpy-RNG #1039; NO committed failing test.)

# REQ-6 oracle gate (NOT-STARTED) — johnson_lindenstrauss_min_dim value:
python3 -c "from sklearn.random_projection import johnson_lindenstrauss_min_dim
print('jl(1000, eps=0.1) =', johnson_lindenstrauss_min_dim(1000, eps=0.1))"
#   -> jl(1000, eps=0.1) = 5920
#   ferrolearn has no johnson_lindenstrauss_min_dim and no n_components='auto'.
```

The in-module `#[test]`s exercise REQ-1 (Gaussian output shape, determinism,
fit_transform, f32), REQ-2 (Sparse output shape, sparsity at default density, custom
density, determinism, fit_transform, f32), and REQ-3 (every error path —
zero-components, empty-input, shape-mismatch, invalid-density, unfitted). No green
ferrolearn command establishes REQ-4 (exact-value RNG-stream parity — carve-out #1388,
NO committed failing test per R-DEFER-3), REQ-5 (Sparse sampling method), or
REQ-6..REQ-11 (`n_components='auto'`/`johnson_lindenstrauss_min_dim`, sparse
storage/`dense_output`, `inverse_transform`, the `n_components>n_features` warning +
sklearn attr surface, PyO3, ferray).

## Blockers

REQ-1 (Gaussian distribution + `X@R` transform + determinism, HEADLINE), REQ-2 (Sparse
distribution + `X@R` transform, HEADLINE), and REQ-3 (scoped error / parameter contracts)
are SHIPPED, with the boundary re-export at `lib.rs:167-170` and the
`PipelineTransformer` / `FittedPipelineTransformer` impls as the grandfathered (S5 /
R-DEFER-1) non-test production consumers.

The headline carve-out is REQ-4: **exact projection-matrix VALUE parity across the numpy
`RandomState` vs Rust `SmallRng` streams is IMPOSSIBLE** — the two PRNGs emit different
number streams at the same seed, so the matrices differ element-by-element. The
DISTRIBUTION, scale, and support ARE matched (REQ-1, REQ-2). This is the SAME class as
the `cluster/kmeans.rs` numpy-RNG carve-outs (#1039). NOT-STARTED with **NO committed
failing test** per R-DEFER-3 — the blocker is the open work item, not a red test.

The NOT-STARTED REQs, filed as `-l blocker` issues against tracking issue #1387
(blockers #1388-#1395):

- `#1388` — REQ-4 (CARVE-OUT): exact projection-matrix VALUE parity (numpy `RandomState`
  Mersenne-Twister stream vs Rust `SmallRng`; `random_projection.py:200,283,294`). SAME
  class as kmeans numpy-RNG #1039. NO committed failing test (R-DEFER-3).
- `#1389` — REQ-5: Sparse per-row `binomial(n_features, density)` +
  `sample_without_replacement` + CSR sampling METHOD vs per-entry Bernoulli (`:281-301`).
- `#1390` — REQ-6: `n_components='auto'` + `eps` + `johnson_lindenstrauss_min_dim` free
  fn (`:64-143,388-391`).
- `#1391` — REQ-7: `components_` CSR sparse storage + `(n_components,n_features)`
  orientation + `dense_output` (`:297-301,419,810`).
- `#1392` — REQ-8: `inverse_transform` + `compute_inverse_components` pinv
  (`:356-361,423-424,431-458`).
- `#1393` — REQ-9: `n_components>n_features` `DataDimensionalityWarning` (`:407-414`) +
  `n_components_`/`components_`/`n_features_in_`/`density_` attrs + `get_feature_names_out`
  (`:416,419,788,427`).
- `#1394` — REQ-10: no PyO3 `GaussianRandomProjection` / `SparseRandomProjection` binding
  in `ferrolearn-python`.
- `#1395` — REQ-11: matrix generation + `X @ R` on `ndarray` + `rand`/`rand_distr` /
  `num_traits`, not `ferray-core` + `ferray::random` (R-SUBSTRATE-1/2; also the enabler
  for REQ-4's numpy-faithful RNG).
