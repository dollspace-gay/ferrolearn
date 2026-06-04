# FactorAnalysis (sklearn.decomposition.FactorAnalysis)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 755ab89d
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_factor_analysis.py  # class FactorAnalysis(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (:42). ctor (:181-204): n_components=None, *, tol=1e-2, copy=True, max_iter=1000, noise_variance_init=None, svd_method="randomized", iterated_power=3, rotation=None, random_state=0. _parameter_constraints (:169-179). fit/_fit (:206-312): _validate_data dtype=float64 (:222); n_components=n_features if None (:228-229); mean_ = X.mean(axis=0) (:231); X -= mean_ (:232); nsqrt = sqrt(n_samples) (:235); llconst = n_features*log(2π) + n_components (:236); var = X.var(axis=0) (:237); psi = np.ones(n_features) init (:239-240) [noise_variance_init overrides :241-248]; SMALL=1e-12 (:252). svd_method="lapack" (:256-264): my_svd = exact linalg.svd, return s[:n_components], Vt[:n_components], squared_norm(s[n_components:]) — DETERMINISTIC. svd_method="randomized" (:266-276): randomized_svd(random_state, n_iter=iterated_power) — RNG. EM loop (:278-297): sqrt_psi = sqrt(psi)+SMALL (:280); s,Vt,unexp_var = my_svd(X/(sqrt_psi*nsqrt)) (:281); s **= 2 (:282); W = sqrt(max(s-1,0))[:,None]*Vt (:284); W *= sqrt_psi (:286); ll = llconst + sum(log(s)) + unexp_var + sum(log(psi)) (:289-290); ll *= -n_samples/2 (:291); loglike.append(ll) (:292); break if (ll - old_ll) < tol (:293); psi = max(var - sum(W**2, axis=0), SMALL) (:297). self.components_ = W (:306, shape (n_components, n_features)); optional _rotate (:307-308); self.noise_variance_ = psi (:309); self.loglike_ = loglike LIST (:310); self.n_iter_ = i+1 (:311). transform (:314-342): Ih = eye(len(components_)) (:333); Wpsi = components_ / noise_variance_ (:337); cov_z = inv(Ih + Wpsi @ components_.T) (:338); X_transformed = (X - mean_) @ Wpsi.T @ cov_z (:339-340). get_covariance (:344-358): cov = components_.T @ components_; cov.flat[diag] += noise_variance_ (:356-357). get_precision (:360-386): matrix-inversion lemma. score_samples/score (:388-426): Gaussian log-likelihood via get_precision + fast_logdet. _rotate (:428-433) / _ortho_rotation (:441-460): varimax/quartimax. _n_features_out = components_.shape[0] (:435-438).
ferrolearn-module: ferrolearn-decomp/src/factor_analysis.rs
parity-ops: FactorAnalysis
crosslink-issue: 1526
-->

## Summary

`ferrolearn-decomp/src/factor_analysis.rs` mirrors scikit-learn's `FactorAnalysis`
(`sklearn/decomposition/_factor_analysis.py`, `class
FactorAnalysis(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator)`
`:42`): the latent linear generative model `X = W·Z + μ + ε`, `Z ~ N(0, I)`,
`ε ~ N(0, diag(ψ))`, fitted by EM maximum likelihood of the loading matrix `W` and
the per-feature noise variance `ψ`. `fn fit` (`pub fn fit` in `factor_analysis.rs`)
now implements scikit-learn's **DETERMINISTIC SVD-based EM** — the `svd_method='lapack'`
branch (`_factor_analysis.py:250-311`) — NOT a random-init posterior-mean EM. The
exposed surface is the unfitted `FactorAnalysis<F> { n_components, max_iter (1000),
tol (1e-3), random_state }` (`struct FactorAnalysis` in `factor_analysis.rs`,
`new`/`with_max_iter`/`with_tol`/`with_random_state`, generic over `F`) and the fitted
`FittedFactorAnalysis<F> { components (n_features, n_components), noise_variance, mean,
n_iter, log_likelihood }` (`struct FittedFactorAnalysis`, accessors `components`/
`noise_variance`/`mean`/`n_iter`/`log_likelihood`, `inverse_transform`), re-exported at
the crate root (`pub use factor_analysis::{FactorAnalysis, FittedFactorAnalysis}`,
`lib.rs:86`) and bound to CPython as `_RsFactorAnalysis`
(`ferrolearn-python/src/extras.rs:1137`, registered
`m.add_class::<extras::RsFactorAnalysis>()` `lib.rs:78`).

**THE HEADLINE — ferrolearn now reaches TIGHT VALUE PARITY with
`sklearn.FactorAnalysis(svd_method='lapack')` (was blocker #1527, now FIXED).** A
critic+fixer cycle replaced the old random-init posterior-mean EM with sklearn's exact
SVD-on-whitened-data EM. The FA log-likelihood depends on `W` only through `W·Wᵀ` (the
implied covariance `Σ = W·Wᵀ + diag(ψ)`), so `noise_variance_`, `Σ`, and the converged
log-likelihood are ROTATION-INVARIANT and now match sklearn to ~1e-13/1e-14; the
loadings `components_` themselves now match sklearn `components_.T` element-wise UP TO
PER-COMPONENT SIGN (faer's right-singular-vector sign vs LAPACK's; sklearn FA applies no
`svd_flip`, so there is no canonical sign to pin), with `transform` scores matching up
to the same sign. The remaining loadings-sign divergence is the only value-parity
carve-out, and it is much narrower than the old "arbitrary rotation" carve-out.

As of this baseline the contract is **SHIPPED-scoped**: the FA VALUE parity
(`noise_variance_`, the implied covariance `W·Wᵀ + diag(ψ)`, the converged
log-likelihood — all rotation-invariant, matching ~1e-13 — plus `components_`/`transform`
matching sklearn `components_.T`/scores up to per-component sign), the SVD-EM algorithm
itself (including sklearn's one-sided convergence `(ll-old_ll)<tol`), the structural
contract (shapes, positivity, finiteness, determinism, error/param contracts,
`inverse_transform`, `PipelineTransformer`, the `_RsFactorAnalysis` binding). NOT-STARTED:
the per-component loadings SIGN carve-out (`#1528`); `svd_method='randomized'` + RNG path
(`#1529`); `rotation` varimax/quartimax (`#1530`); `noise_variance_init` (`#1531`);
`loglike_` LIST attr (`#1532`); `score`/`score_samples` (`#1533`);
`get_covariance`/`get_precision` METHODS (`#1534`); `n_components=None` default + `copy`
+ `n_features_in_` (`#1535`); `tol` DEFAULT 1e-3 vs 1e-2 (`#1536`); `components_`
ORIENTATION transpose (`#1537`); the production `assert_eq!` debug-assert in `transform`
(`#1538`); the ferray substrate (`#1539`). **7 SHIPPED / 12 NOT-STARTED.**

`FactorAnalysis` / `FittedFactorAnalysis` are existing pub APIs whose non-test consumers
are the crate re-export (`lib.rs:86`, boundary public API, grandfathered S5/R-DEFER-1),
the `_RsFactorAnalysis` PyO3 binding (`extras.rs:1137`, registered `lib.rs:78`), and the
`PipelineTransformer`/`FittedPipelineTransformer` impls (`impl PipelineTransformer for
FactorAnalysis` / `impl FittedPipelineTransformer for FittedFactorAnalysis` in
`factor_analysis.rs`). The value-parity claims are pinned by
`tests/divergence_factor_analysis.rs` (13 tests, all green).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# PROBE 1 (the value-parity target) — deterministic svd_method='lapack' on a fresh
# 12x5 fixture with genuine 2-factor structure. VALUES from sklearn (R-CHAR-3), never
# copied from ferrolearn.
python3 -c "
import numpy as np
from sklearn.decomposition import FactorAnalysis
rng=np.random.RandomState(7)
X=rng.randn(12,5) + rng.randn(12,2)@rng.randn(2,5)
m=FactorAnalysis(n_components=2, svd_method='lapack').fit(X)       # default tol=1e-2
print('components_.shape:', m.components_.shape)
print('noise_variance_:', np.round(m.noise_variance_,6).tolist())
print('get_covariance diag:', np.round(np.diag(m.get_covariance()),6).tolist())
print('loglike_[-1]:', round(float(m.loglike_[-1]),6), 'n_iter_:', m.n_iter_,
      'len(loglike_):', len(m.loglike_))
print('score(X):', round(float(m.score(X)),6))
print('components_ row0:', np.round(m.components_[0],6).tolist())"
# -> components_.shape: (2, 5)                          => sklearn (k, p); ferrolearn (p, k) (#1537)
# -> noise_variance_: [1.478876, 0.285627, 1.146431, 0.701773, 0.184197]
#                                                       => ROTATION-INVARIANT, value-parity target
# -> get_covariance diag: [2.344709, 1.428456, 1.183232, 1.956985, 2.526632]
#                                                       => W·Wᵀ + diag(ψ), ROTATION-INVARIANT
# -> loglike_[-1]: -92.425537 n_iter_: 20 len(loglike_): 20
#                                                       => converged ll INVARIANT; loglike_ LIST (#1532)
# -> score(X): -7.702128                                => Gaussian avg log-likelihood (#1533)
# -> components_ row0: [-0.129471, -0.893589, -0.169606, 1.068657, 1.521207]
#                                                       => loadings: match ferro up to per-component SIGN (#1528)

# PROBE 2 (algorithm + tight parity at MATCHED tol=1e-3) — confirms the SVD-EM produces
# the SAME optimum as sklearn when tol is matched (ferrolearn default tol=1e-3).
python3 -c "
import numpy as np
from sklearn.decomposition import FactorAnalysis
rng=np.random.RandomState(7)
X=rng.randn(12,5) + rng.randn(12,2)@rng.randn(2,5)
m=FactorAnalysis(n_components=2, svd_method='lapack', tol=1e-3).fit(X)
print('n_iter_:', m.n_iter_, 'loglike_[-1]:', round(float(m.loglike_[-1]),8))
print('noise_variance_:', [round(v,8) for v in m.noise_variance_])"
# -> n_iter_: 48 loglike_[-1]: -92.34208231
# -> noise_variance_: [1.63300298, 0.17027096, 1.14892366, 0.72079366, 0.1384043]
#    Re-audit cross-check (ferrolearn FactorAnalysis::<f64>::new(2).with_tol(1e-3) on the
#    SAME fixture): noise_variance max diff 2.5e-14, get_covariance() diag max diff
#    7.1e-15, log_likelihood vs loglike_[-1] diff 1.4e-14, n_iter IDENTICAL (48==48),
#    components vs sklearn components_.T after per-component SIGN alignment max diff
#    4.8e-14. (Task-reported fresh-fixture audit: noise 3.3e-15, cov 2.9e-15, ll 7.1e-15,
#    n_iter 36==36, components 1.6e-13 after sign align.) => value parity SHIPPED.

# PROBE 3 (ctor defaults — #1529..#1537).
python3 -c "
from sklearn.decomposition import FactorAnalysis
m=FactorAnalysis()
for p in ['n_components','tol','copy','max_iter','noise_variance_init','svd_method','iterated_power','rotation','random_state']:
    print(f'{p} =', getattr(m,p))"
# -> n_components = None  tol = 0.01  copy = True  max_iter = 1000
# -> noise_variance_init = None  svd_method = randomized  iterated_power = 3
# -> rotation = None  random_state = 0
#    => ferrolearn has n_components/max_iter/tol(1e-3)/random_state only; NO copy,
#       noise_variance_init, svd_method, iterated_power, rotation; tol default 1e-3
#       (not 1e-2); random_state Option<u64> default None (now INERT — lapack EM is
#       deterministic).
```

## Requirements

- REQ-1: **FA VALUE parity with `sklearn.FactorAnalysis(svd_method='lapack')` (SHIPPED;
  was blocker #1527, FIXED).** ferrolearn's `fn fit` (`pub fn fit` in
  `factor_analysis.rs`) implements sklearn's deterministic SVD-EM
  (`_factor_analysis.py:250-311`): the rotation-INVARIANT identifiable quantities —
  `noise_variance_` (ψ), the implied covariance `W·Wᵀ + diag(ψ)` (= sklearn
  `get_covariance()`), and the converged log-likelihood — match sklearn to ~1e-13/1e-14,
  AND the loadings `components_` match sklearn `components_.T` element-wise UP TO
  PER-COMPONENT SIGN, with `transform` scores matching up to the same sign. Re-audit
  cross-check (Probe 2, matched tol=1e-3, fresh 12×5 fixture): `noise_variance_` max diff
  2.5e-14, `get_covariance()` diag max diff 7.1e-15, `log_likelihood` vs `loglike_[-1]`
  diff 1.4e-14, `n_iter` identical (48==48), `components_` after per-component sign
  alignment max diff 4.8e-14. Pinned by `divergence_fa_rotation_invariant_covariance`
  and `divergence_fa_simple_data_loglike` (`tests/divergence_factor_analysis.rs`, both
  green, asserting the rotation-invariant covariance/noise/log-likelihood to a documented
  tolerance). (The former REQ-1 "exact components value parity"/REQ-2 "rotation-invariant
  parity" carve-out is now SHIPPED — only the per-component sign remains, REQ-2 below.)

- REQ-2: **Per-component `components_` SIGN parity (NOT-STARTED, CARVE-OUT; `#1528`).**
  After matching `noise_variance_`/covariance/log-likelihood and aligning each component
  to sklearn `components_.T` by per-component sign, the loadings agree to ~1e-13; without
  sign alignment, individual components may have flipped sign. The cause: faer's thin SVD
  (`factor_analysis_svd_f64`/`_f32` in `factor_analysis.rs`) returns right-singular
  vectors whose sign differs from LAPACK's, and sklearn FA applies NO `svd_flip`
  (`_factor_analysis.py:284`, `W = sqrt(max(s-1,0))[:,None]*Vt`) — there is therefore NO
  deterministic, canonical sign target to match sklearn's exact sign. **CARVE-OUT
  (R-DEFER-3):** the sign is mathematically arbitrary for FA loadings (the model is
  sign-invariant per latent factor); no failing test is asserted (the divergence tests
  reconstruct the sign-invariant covariance, and `carveout_fa_loadings_only_rotation_invariant`
  asserts only the rotation-invariant property structurally).

- REQ-3: **SVD-based EM algorithm = sklearn `svd_method='lapack'` (SHIPPED).** `fn fit`
  (`pub fn fit` in `factor_analysis.rs`) mirrors `_factor_analysis.py:250-311`: `psi =
  ones` init; `llconst = p·ln(2π) + k`; per iteration `sqrt_psi = sqrt(psi) + SMALL`
  (`SMALL = 1e-12`), thin SVD (faer) of `Xc/(sqrt_psi·sqrt(n))`, `s² = s²`, `W =
  sqrt(max(s²-1, 0))·Vt·sqrt_psi`, `unexp_var = ‖s[k..]‖²`, `ll = (llconst + Σ ln(s²) +
  unexp_var + Σ ln(psi))·(-n/2)`, the ONE-SIDED convergence `(ll - old_ll) < tol`, then
  `psi = max(var - Σ W², 1e-12)`. The thin SVD is dispatched to faer for f64/f32 via
  `factor_analysis_svd` (`fn factor_analysis_svd` in `factor_analysis.rs`), a `TypeId`
  dispatcher with `unsafe` transmute mirroring `eigen_dispatch` in `pca.rs`
  (`fn eigen_dispatch`). The old `compute_log_likelihood` + random-init code are GONE.
  Pinned indirectly by all 13 divergence/green-guard tests (the algorithm IS what
  produces the parity). Non-test consumers: re-export `lib.rs:86`, `_RsFactorAnalysis`
  `extras.rs:1137`.

- REQ-4: **Structural: `components` shape `(n_features, n_components)`, transform scores
  shape `(n_samples, n_components)`, positive `noise_variance`, mean shape, `n_iter ≥ 1`,
  finite `log_likelihood`, determinism (SHIPPED).** `fn fit` stores `components` of shape
  `(n_features, n_components)` (`FittedFactorAnalysis.components`), `noise_variance`
  `(n_features,)`, `mean` `(n_features,)`, `n_iter ≥ 1`, a finite scalar `log_likelihood`.
  `transform` (`impl Transform for FittedFactorAnalysis` in `factor_analysis.rs`) returns
  scores `(n_samples, n_components)` via `Σ_z = inv(I + WᵀΨ⁻¹W)`, `β = Σ_z WᵀΨ⁻¹`, `scores
  = (β Xcᵀ)ᵀ` — the SAME posterior-mean formula as sklearn `(X − mean_) @ Wpsi.T @ cov_z`
  (`_factor_analysis.py:337-340`). The EM is now DETERMINISTIC, so determinism is trivial
  (`random_state` is inert). Pinned by `test_fa_fit_returns_fitted` `(4,2)`,
  `test_fa_components_accessor`, `test_fa_transform_shape` `(10,2)`,
  `test_fa_transform_new_data` `(3,1)`, `test_fa_noise_variance_positive`,
  `test_fa_mean_shape`, `test_fa_n_iter_positive`, `test_fa_log_likelihood_finite`,
  `test_fa_reproducible_with_seed`, `test_fa_scores_not_all_zero`, and the
  `green_*` guards in `tests/divergence_factor_analysis.rs`. Non-test consumers:
  re-export `lib.rs:86`, `_RsFactorAnalysis` `extras.rs:1137`.

- REQ-5: **`inverse_transform` (SHIPPED scoped).** `FittedFactorAnalysis::inverse_transform`
  (`pub fn inverse_transform` in `factor_analysis.rs`) maps latent `Z` back to feature
  space as `Z @ Wᵀ + mean` — accounting for ferrolearn's transposed `components` layout
  `(n_features, n_components)`; the column-count guard returns `ShapeMismatch`. sklearn
  `FactorAnalysis` inherits no public `inverse_transform` (it is not a `_BasePCA`
  subclass), so this is a ferrolearn convenience mirroring the generative map `X ≈ W·Z + μ`.
  Pinned by `green_inverse_transform_shape_and_col_mismatch`
  (`tests/divergence_factor_analysis.rs`). Non-test consumers: re-export `lib.rs:86`,
  `_RsFactorAnalysis` `extras.rs:1137`.

- REQ-6: **Error / parameter contracts (SHIPPED scoped).** `fn fit` returns
  `InvalidParameter { name: "n_components" }` for `n_components == 0` and for
  `n_components > n_features`, and `InsufficientSamples { required: 2 }` for `< 2` samples;
  `transform` returns `ShapeMismatch` on a column-count mismatch; `inverse_transform`
  returns `ShapeMismatch` when `z.ncols() != n_components`. Pinned by
  `test_fa_error_zero_components`, `test_fa_error_too_many_components`,
  `test_fa_error_insufficient_samples`, `test_fa_transform_shape_mismatch`, and the
  `green_error_*` guards. **FLAG (candidate DIVs):** sklearn validates via
  `_parameter_constraints` (`_factor_analysis.py:169-179`) raising
  `InvalidParameterError`/`ValueError`, NOT `FerroError`; sklearn allows `n_components=0`
  (`Interval(Integral, 0, None)` `:170`) and `n_components=None` (→ `n_features`
  `:228-229`) which ferrolearn rejects / requires as an explicit `usize`; sklearn does NOT
  pre-reject `n_samples < 2`. Non-test consumers: re-export `lib.rs:86`,
  `_RsFactorAnalysis` `extras.rs:1137`.

- REQ-7: **`PipelineTransformer` integration (SHIPPED scoped).** `FactorAnalysis<F>`
  implements `PipelineTransformer<F>` (`impl PipelineTransformer for FactorAnalysis` in
  `factor_analysis.rs`, `fit_pipeline` delegating to `Fit::fit`) and
  `FittedFactorAnalysis<F>` implements `FittedPipelineTransformer<F>`
  (`transform_pipeline` delegating to `Transform::transform`), so FA can be a transform
  step in a `Pipeline` — the ferrolearn analogue of sklearn's `TransformerMixin` (FA is
  `ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator`
  `_factor_analysis.py:42`). Pinned by `test_fa_pipeline_transformer`. Non-test consumer:
  the `ferrolearn_core::pipeline` machinery (`use ferrolearn_core::pipeline` in
  `factor_analysis.rs`).

- REQ-8: **PyO3 `_RsFactorAnalysis` binding surface (SHIPPED scoped).** `_RsFactorAnalysis`
  (`extras.rs:1137`, registered `m.add_class::<extras::RsFactorAnalysis>()` `lib.rs:78`)
  exposes a `(n_components: usize = 2)` ctor over `ferrolearn_decomp::FittedFactorAnalysis
  <f64>` via the `py_transformer!` macro — the boundary CPython consumer of
  `FactorAnalysis::new` / `fit` / `transform`. **Scope: the binding faithfully marshals the
  Rust fit/transform surface; its scores now match sklearn `transform` up to per-component
  sign (REQ-2); the macro is f64-only and does NOT expose `noise_variance_` / `loglike_` /
  `score` / `get_covariance` getters (those are `#1532`..`#1534`).** Non-test consumer:
  itself (the CPython boundary).

- REQ-9: **`svd_method='randomized'` + RNG path (`iterated_power`) (NOT-STARTED;
  `#1529`).** sklearn's `svd_method="randomized"` (default `_factor_analysis.py:189`)
  drives a `randomized_svd` with an RNG and `n_iter=iterated_power` (`:266-276`).
  ferrolearn implements ONLY the `lapack` (exact, deterministic) branch via faer's thin
  SVD (`factor_analysis_svd` in `factor_analysis.rs`) — there is no `svd_method` field, no
  `iterated_power`, and no randomized path.

- REQ-10: **`rotation` ('varimax' / 'quartimax') (NOT-STARTED; `#1530`).** sklearn's
  `rotation=None` (default) optionally post-rotates the loadings via `_rotate` /
  `_ortho_rotation` (`_factor_analysis.py:307-308`, `:428-460`). ferrolearn's
  `FactorAnalysis<F>` (`struct FactorAnalysis` in `factor_analysis.rs`) has NO `rotation`
  field and applies no post-rotation.

- REQ-11: **`noise_variance_init` (NOT-STARTED; `#1531`).** sklearn's
  `noise_variance_init=None` (default → `np.ones`) optionally seeds the EM with a
  user-supplied per-feature noise prior (`_factor_analysis.py:239-248`), validating its
  length. ferrolearn's `FactorAnalysis<F>` has NO `noise_variance_init` field — `psi` is
  hard-initialised to `ones` (`Array1::from_elem(p, F::one())` in `fn fit`).

- REQ-12: **`loglike_` LIST attribute (NOT-STARTED; `#1532`).** sklearn exposes
  `loglike_` — the per-iteration log-likelihood LIST (`_factor_analysis.py:118`, appended
  each EM step `:292`, stored `:310`; Probe 1 `len = 20`). ferrolearn's
  `FittedFactorAnalysis<F>` keeps only the FINAL scalar `log_likelihood`
  (`FittedFactorAnalysis.log_likelihood`, accessor `log_likelihood()`) — NOT the
  per-iteration vector.

- REQ-13: **`score` / `score_samples` (Gaussian log-likelihood) (NOT-STARTED;
  `#1533`).** sklearn's `FactorAnalysis` exposes `score_samples` (per-sample Gaussian
  log-likelihood under the FA model via `get_precision` + `fast_logdet`,
  `_factor_analysis.py:388-408`) and `score` (their average `:410-426`; Probe 1
  `score(X) = -7.702128`). `FittedFactorAnalysis<F>` exposes neither — it has no
  `get_precision` (`#1534`).

- REQ-14: **`get_covariance` / `get_precision` METHODS (NOT-STARTED; `#1534`).** sklearn's
  `FactorAnalysis.get_covariance` (`_factor_analysis.py:344-358`) reconstructs `cov =
  components_.T @ components_ + diag(noise_variance_)` (Probe 1 diag `[2.344709, 1.428456,
  1.183232, 1.956985, 2.526632]`) and `get_precision` (`:360-386`) its inverse via the
  matrix-inversion lemma. `FittedFactorAnalysis<F>` exposes neither as a METHOD — the
  covariance VALUE matches sklearn (REQ-1) and is computable from the `components()` +
  `noise_variance()` accessors, but there is no `get_covariance`/`get_precision` API.

- REQ-15: **`n_components=None` default + `copy` + `n_features_in_` (NOT-STARTED;
  `#1535`).** sklearn accepts `n_components=None` (→ `n_features`,
  `_factor_analysis.py:228-229`), `copy=True` (`:186`), and exposes the fitted
  `n_features_in_` (`:130-131`). ferrolearn's `FactorAnalysis::new` (`pub fn new` in
  `factor_analysis.rs`) requires an explicit `usize` (no `None` default), has no `copy`
  field (it always copies via `x.to_owned()`), and exposes no `n_features_in_` accessor
  (derivable from `mean().len()`).

- REQ-16: **`tol` DEFAULT (1e-3 vs 1e-2) (NOT-STARTED; `#1536`).** sklearn's `tol`
  defaults to `1e-2` (`_factor_analysis.py:185`; Probe 3). ferrolearn's `tol` defaults to
  `1e-3` (`pub fn new` in `factor_analysis.rs`). The convergence CRITERION now MATCHES
  sklearn (both use the one-sided `(ll - old_ll) < tol`, REQ-3); only the DEFAULT differs,
  which can stop at a different iteration → a slightly different (still valid) optimum
  unless `with_tol(1e-2)` is set. (Probe 1 at default tol=1e-2: `n_iter_=20`; Probe 2 at
  matched tol=1e-3: `n_iter_=48`.)

- REQ-17: **`components_` ORIENTATION (transpose) (NOT-STARTED; `#1537`).** sklearn's
  `components_` has shape `(n_components, n_features)` (`_factor_analysis.py:115`, `:306`;
  Probe 1 `(2, 5)`), whereas ferrolearn's `components` has shape `(n_features,
  n_components)` (`FittedFactorAnalysis.components`; Probe 1 analogue `(5, 2)`) — i.e.
  ferrolearn stores sklearn's `components_.T` (the impl stores `W.T`, preserving the
  established public layout). The `inverse_transform` doc-comment explicitly notes the
  transpose. A consumer comparing `ferrolearn.components` to sklearn `components_`
  array-by-array (R-DEV-3 output-shape contract) sees a transposed matrix.

- REQ-18: **production `assert_eq!` debug-assert in `transform` (NOT-STARTED; `#1538`).**
  `FittedFactorAnalysis::transform` (`impl Transform for FittedFactorAnalysis` in
  `factor_analysis.rs`) ends with `assert_eq!(scores.dim(), (n_samples, k))` — a
  production `assert!`-family macro OUTSIDE `#[cfg(test)]` (R-CODE-2 / R-APG-1: no
  `panic!`-family in library code; library must return `Result<_, FerroError>`). It is a
  self-consistency check that can never fire given the preceding algebra, but it violates
  the no-panic discipline and should be removed or converted to a returned error.

- REQ-19: **ferray substrate (NOT-STARTED; `#1539`).** `factor_analysis.rs` computes on
  `ndarray::{Array1, Array2}` (`use ndarray` in `factor_analysis.rs`), dispatches the SVD
  to `faer` (`factor_analysis_svd_f64`/`_f32`), and uses a hand-rolled Cholesky inverse
  (`fn cholesky_inv`), not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`,
`svd_method='lapack'` for determinism), never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1/REQ-3, SHIPPED): on the Probe 2 fixture at MATCHED tol=1e-3, ferrolearn's
  `noise_variance()`, the reconstructed `componentsᵀ·components + diag(noise_variance)`
  (diag), the converged `log_likelihood()`, and `n_iter()` match sklearn
  `FactorAnalysis(n_components=2, svd_method='lapack', tol=1e-3)` (`noise_variance_ =
  [1.63300298, 0.17027096, 1.14892366, 0.72079366, 0.1384043]`, `loglike_[-1] =
  -92.34208231`, `n_iter_ = 48`) to ~1e-13 (re-audit: noise 2.5e-14, cov diag 7.1e-15, ll
  1.4e-14, n_iter 48==48); `components()` match sklearn `components_.T` after
  per-component sign alignment to ~1e-13 (4.8e-14). Pinned by
  `divergence_fa_rotation_invariant_covariance` and `divergence_fa_simple_data_loglike`.

- AC-2 (REQ-2, NOT-STARTED, CARVE-OUT): individual `components()` columns may carry a sign
  opposite sklearn's `components_.T` (faer vs LAPACK right-singular-vector sign; sklearn
  applies no `svd_flip`). No deterministic sign target exists; no failing test asserts
  this (R-DEFER-3). `carveout_fa_loadings_only_rotation_invariant` asserts only the
  sign-invariant covariance is symmetric.

- AC-3 (REQ-4/5, SHIPPED scoped): `FactorAnalysis::<f64>::new(2).fit(&X, &()).unwrap()
  .components().dim()` is `(n_features, 2)`; `transform(&X)` has shape `(n_samples, 2)`;
  `noise_variance()` entries `> 0`; `mean().len() == n_features`; `n_iter() ≥ 1`;
  `log_likelihood().is_finite()`; two fits with the same seed agree to 1e-15 (trivially —
  deterministic algorithm); `inverse_transform` round-trips the shape. Pinned by
  `test_fa_*` in-module tests + the `green_*` guards.

- AC-4 (REQ-6, SHIPPED scoped): `fit` returns `Err` for `n_components=0`, `n_components >
  n_features`, and `n_samples < 2`; `transform`/`inverse_transform` return `Err` on a
  column-count mismatch. Pinned by `test_fa_error_*` + `green_error_*`. FLAG: sklearn
  raises `InvalidParameterError`/`ValueError` (not `FerroError`), allows
  `n_components=0`/`None`, and does not pre-reject `n_samples < 2`.

- AC-5 (REQ-7/8, SHIPPED scoped): a `Pipeline` with an `FA` transform step transforms
  (`test_fa_pipeline_transformer`); `import ferrolearn; ferrolearn._RsFactorAnalysis(2)`
  exposes `fit`/`transform` over `FittedFactorAnalysis<f64>` (`extras.rs:1137`,
  `lib.rs:78`) — scores matching sklearn `transform` up to per-component sign.

- AC-6 (REQ-9..19, DIVERGES): `FactorAnalysis()` defaults `n_components=None, tol=1e-2,
  copy=True, max_iter=1000, noise_variance_init=None, svd_method="randomized",
  iterated_power=3, rotation=None, random_state=0` (Probe 3,
  `_factor_analysis.py:181-204`); ferrolearn implements only `svd_method='lapack'`, stores
  `components_` as `(n_features, n_components)` (sklearn `components_.T`), defaults
  `tol=1e-3`, and exposes neither `loglike_` (list), `score`/`score_samples`,
  `get_covariance`/`get_precision` methods, nor `n_features_in_`. ferrolearn also carries a
  production `assert_eq!` in `transform` and computes on `ndarray` + faer + a hand-rolled
  Cholesky, not ferray.

## REQ status

Binary (R-DEFER-2). `FactorAnalysis` / `FittedFactorAnalysis` are existing pub APIs; the
non-test consumers are the crate re-export (`lib.rs:86`, boundary public API,
grandfathered S5/R-DEFER-1), the `_RsFactorAnalysis` PyO3 binding (`extras.rs:1137`,
registered `lib.rs:78`), and the `PipelineTransformer` impls (`impl PipelineTransformer
for FactorAnalysis` / `impl FittedPipelineTransformer for FittedFactorAnalysis` in
`factor_analysis.rs`). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn
1.5.2). Live oracle = installed sklearn 1.5.2, run from `/tmp` with `svd_method='lapack'`
for determinism. **THE HEADLINE:** `fn fit` now implements sklearn's deterministic SVD-EM
(`_factor_analysis.py:250-311`); a critic+fixer cycle (was blocker #1527, now FIXED)
brought ferrolearn to TIGHT value parity — `noise_variance_`, the implied covariance
`W·Wᵀ + diag(ψ)`, and the converged log-likelihood (all rotation-invariant) match
sklearn to ~1e-13/1e-14, and `components_`/`transform` match sklearn `components_.T`/scores
UP TO per-component sign. The only value-parity carve-out left is the per-component
loadings SIGN (REQ-2, `#1528`). #1526 is this doc's crosslink tracking issue. Count:
**7 SHIPPED (REQ-1,3,4,5,6,7,8) / 12 NOT-STARTED (REQ-2,9,10,11,12,13,
14,15,16,17,18,19)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (FA value parity vs `svd_method='lapack'`) | SHIPPED | impl `pub fn fit in factor_analysis.rs` implements sklearn's deterministic SVD-EM (`_factor_analysis.py:250-311`). Rotation-INVARIANT `noise_variance_`, implied covariance `W·Wᵀ + diag(ψ)` (= `get_covariance()`), converged log-likelihood match sklearn ~1e-13; `components_`/`transform` match sklearn `components_.T`/scores UP TO per-component SIGN (was blocker #1527, FIXED). Re-audit (Probe 2, matched tol=1e-3, fresh 12×5): noise max diff 2.5e-14, get_covariance() diag max diff 7.1e-15, log_likelihood vs loglike_[-1] diff 1.4e-14, n_iter 48==48, components after sign align 4.8e-14. Non-test consumer: re-export `lib.rs:86`; `_RsFactorAnalysis` `extras.rs:1137`. Verification: `cargo test -p ferrolearn-decomp --test divergence_factor_analysis` → `divergence_fa_rotation_invariant_covariance`, `divergence_fa_simple_data_loglike` PASS. |
| REQ-2 (per-component `components_` SIGN parity) | NOT-STARTED | open prereq blocker **#1528** (CARVE-OUT, R-DEFER-3). faer's thin SVD (`factor_analysis_svd_f64`/`_f32` in `factor_analysis.rs`) returns right-singular vectors whose sign differs from LAPACK's; sklearn FA applies NO `svd_flip` (`_factor_analysis.py:284` `W = sqrt(max(s-1,0))[:,None]*Vt`), so there is NO deterministic canonical sign to match. Loadings agree to ~1e-13 only after per-component sign alignment. No failing test (R-DEFER-3); `carveout_fa_loadings_only_rotation_invariant` asserts only the sign-invariant covariance. |
| REQ-3 (SVD-EM algorithm = sklearn `lapack`) | SHIPPED | impl `pub fn fit in factor_analysis.rs` mirrors `_factor_analysis.py:250-311`: `psi=ones` init; per iteration `sqrt_psi=sqrt(psi)+SMALL` (1e-12), faer thin SVD of `Xc/(sqrt_psi·sqrt(n))`, `s²`, `W=sqrt(max(s²-1,0))·Vt·sqrt_psi`, `unexp_var=‖s[k..]‖²`, `ll=(llconst+Σln(s²)+unexp_var+Σln(psi))·(-n/2)` with `llconst=p·ln(2π)+k`, ONE-SIDED `(ll-old_ll)<tol` (`:293`), `psi=max(var-ΣW², 1e-12)` (`:297`). SVD dispatched f64/f32 via `fn factor_analysis_svd` (TypeId + `unsafe` transmute, mirroring `fn eigen_dispatch in pca.rs`). Old `compute_log_likelihood`/random-init GONE; `with_random_state`/`random_state` retained but INERT. Non-test consumer: re-export `lib.rs:86`; `_RsFactorAnalysis` `extras.rs:1137`. Verification: `cargo test -p ferrolearn-decomp --test divergence_factor_analysis` (13 tests) PASS. |
| REQ-4 (structural: shapes / positive ψ / mean / n_iter / finite ll / determinism) | SHIPPED | `fn fit` stores `components` `(n_features, n_components)` (`FittedFactorAnalysis.components`), `noise_variance` `(n_features,)`, `mean` `(n_features,)`, `n_iter ≥ 1`, finite scalar `log_likelihood`. `transform` (`impl Transform for FittedFactorAnalysis`) returns `(n_samples, n_components)` scores via `Σ_z = inv(I + WᵀΨ⁻¹W)`, `β = Σ_z WᵀΨ⁻¹`, `scores = (β Xcᵀ)ᵀ` — same posterior-mean formula as sklearn `(X − mean_) @ Wpsi.T @ cov_z` (`_factor_analysis.py:337-340`). Deterministic ⇒ determinism trivial. Non-test consumers: re-export `lib.rs:86`, `_RsFactorAnalysis` `extras.rs:1137`. Verification: `cargo test -p ferrolearn-decomp --lib factor_analysis` → `test_fa_fit_returns_fitted` `(4,2)`, `test_fa_components_accessor`, `test_fa_transform_shape` `(10,2)`, `test_fa_transform_new_data` `(3,1)`, `test_fa_noise_variance_positive`, `test_fa_mean_shape`, `test_fa_n_iter_positive`, `test_fa_log_likelihood_finite`, `test_fa_reproducible_with_seed`, `test_fa_scores_not_all_zero` PASS (17/17); + `green_*` guards in the divergence suite. |
| REQ-5 (`inverse_transform` structural round-trip) | SHIPPED | `pub fn inverse_transform in factor_analysis.rs` maps `Z @ Wᵀ + mean` — accounting for the transposed `components` layout `(n_features, n_components)` (doc-comment); column guard `Err(ShapeMismatch)`. Mirrors `X ≈ W·Z + μ` (sklearn `FactorAnalysis` has no public `inverse_transform`; ferrolearn convenience). Non-test consumers: re-export `lib.rs:86`, `_RsFactorAnalysis` `extras.rs:1137`. Verification: `cargo test -p ferrolearn-decomp --test divergence_factor_analysis green_inverse_transform_shape_and_col_mismatch` PASS. |
| REQ-6 (error / parameter contracts, scoped) | SHIPPED | `fn fit` returns `Err(InvalidParameter{name:"n_components", reason:"must be at least 1"})` for `n_components==0`, `Err(InvalidParameter{... "exceeds n_features"})` for `>n_features`, `Err(InsufficientSamples{required:2,...})` for `<2` samples; `transform` `Err(ShapeMismatch)` on column mismatch; `inverse_transform` `Err(ShapeMismatch)` when `z.ncols() != n_components`. Non-test consumers: re-export `lib.rs:86`, `_RsFactorAnalysis` `extras.rs:1137`. Verification: `cargo test -p ferrolearn-decomp factor_analysis` (`test_fa_error_zero_components`, `test_fa_error_too_many_components`, `test_fa_error_insufficient_samples`, `test_fa_transform_shape_mismatch`, `green_error_*`) PASS. **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints` (`_factor_analysis.py:169-179`) raising `InvalidParameterError`/`ValueError` (not `FerroError`); allows `n_components=0` (`:170`) and `n_components=None` → `n_features` (`:228-229`); does NOT pre-reject `n_samples < 2`. |
| REQ-7 (`PipelineTransformer` integration) | SHIPPED | `FactorAnalysis<F>` impls `PipelineTransformer<F>` (`impl PipelineTransformer for FactorAnalysis`, `fit_pipeline` → `Fit::fit`) and `FittedFactorAnalysis<F>` impls `FittedPipelineTransformer<F>` (`transform_pipeline` → `Transform::transform`) — analogue of sklearn's `TransformerMixin` (`_factor_analysis.py:42`). Non-test consumer: the `ferrolearn_core::pipeline` machinery. Verification: `cargo test -p ferrolearn-decomp factor_analysis test_fa_pipeline_transformer` PASS. |
| REQ-8 (PyO3 `_RsFactorAnalysis` binding surface, scoped) | SHIPPED | `_RsFactorAnalysis` (`extras.rs:1137`, registered `m.add_class::<extras::RsFactorAnalysis>()` `lib.rs:78`) exposes a `(n_components: usize = 2)` ctor over `ferrolearn_decomp::FittedFactorAnalysis<f64>` via `py_transformer!` (`extras.rs:1137-1141`) — boundary CPython consumer of `FactorAnalysis::new`/`fit`/`transform`. **Scope: faithful fit/transform marshalling; scores match sklearn `transform` up to per-component sign (REQ-2); f64-only, NO `noise_variance_`/`loglike_`/`score`/`get_covariance` getters (`#1532`..`#1534`).** Verification: `import ferrolearn; ferrolearn._RsFactorAnalysis(2).fit(X).transform(X)` marshals via the `py_transformer!` numpy↔ndarray bridge. |
| REQ-9 (`svd_method='randomized'` + `iterated_power`) | NOT-STARTED | open prereq blocker **#1529**. sklearn `svd_method="randomized"` (default `_factor_analysis.py:189`) → `randomized_svd` with RNG, `n_iter=iterated_power` (`:266-276`). ferrolearn implements ONLY the deterministic `lapack` branch via faer's thin SVD (`fn factor_analysis_svd in factor_analysis.rs`) — no `svd_method`/`iterated_power` field, no randomized path. |
| REQ-10 (`rotation` 'varimax'/'quartimax') | NOT-STARTED | open prereq blocker **#1530**. sklearn `rotation=None` (default) optionally post-rotates loadings via `_rotate`/`_ortho_rotation` (`_factor_analysis.py:307-308`/`:428-460`). ferrolearn `struct FactorAnalysis in factor_analysis.rs` has NO `rotation` field, applies no post-rotation. |
| REQ-11 (`noise_variance_init`) | NOT-STARTED | open prereq blocker **#1531**. sklearn `noise_variance_init=None` (default → `np.ones`) optionally seeds the EM noise prior, length-validated (`_factor_analysis.py:239-248`). ferrolearn has NO `noise_variance_init` field — `psi` hard-inits to `ones` (`Array1::from_elem(p, F::one())` in `pub fn fit`). |
| REQ-12 (`loglike_` LIST attribute) | NOT-STARTED | open prereq blocker **#1532**. sklearn exposes `loglike_` — per-iteration log-likelihood LIST (`_factor_analysis.py:118`, appended `:292`, stored `:310`; Probe 1 `len = 20`). ferrolearn `struct FittedFactorAnalysis in factor_analysis.rs` keeps only the FINAL scalar `log_likelihood` (accessor `log_likelihood()`) — not the vector. |
| REQ-13 (`score` / `score_samples`) | NOT-STARTED | open prereq blocker **#1533**. sklearn `score_samples` (per-sample Gaussian log-likelihood via `get_precision` + `fast_logdet` `_factor_analysis.py:388-408`) + `score` (average `:410-426`; Probe 1 `score(X) = -7.702128`). `struct FittedFactorAnalysis in factor_analysis.rs` exposes neither — no `get_precision` (`#1534`). |
| REQ-14 (`get_covariance` / `get_precision` METHODS) | NOT-STARTED | open prereq blocker **#1534**. sklearn `get_covariance` = `components_.T @ components_ + diag(noise_variance_)` (`_factor_analysis.py:344-358`; Probe 1 diag `[2.344709, 1.428456, 1.183232, 1.956985, 2.526632]`); `get_precision` its inverse via the matrix-inversion lemma (`:360-386`). `struct FittedFactorAnalysis in factor_analysis.rs` exposes neither as a METHOD (the covariance VALUE matches sklearn — REQ-1 — and is computable from `components()` + `noise_variance()`, but there is no `get_covariance`/`get_precision` API). |
| REQ-15 (`n_components=None` default + `copy` + `n_features_in_`) | NOT-STARTED | open prereq blocker **#1535**. sklearn accepts `n_components=None` → `n_features` (`_factor_analysis.py:228-229`), `copy=True` (`:186`), exposes `n_features_in_` (`:130-131`). ferrolearn `pub fn new in factor_analysis.rs` requires explicit `usize` (no `None`), has no `copy` field (always copies `x.to_owned()`), no `n_features_in_` accessor (derivable from `mean().len()`). |
| REQ-16 (`tol` DEFAULT 1e-3 vs 1e-2) | NOT-STARTED | open prereq blocker **#1536**. sklearn `tol=1e-2` (`_factor_analysis.py:185`; Probe 3). ferrolearn `tol=1e-3` (`pub fn new in factor_analysis.rs`). The CRITERION now MATCHES sklearn (both one-sided `(ll-old_ll)<tol`, REQ-3); only the DEFAULT differs → potentially a different stopping iteration unless `with_tol(1e-2)` is set (Probe 1 default tol=1e-2: `n_iter_=20`; Probe 2 tol=1e-3: `n_iter_=48`). |
| REQ-17 (`components_` ORIENTATION transpose) | NOT-STARTED | open prereq blocker **#1537**. sklearn `components_` is `(n_components, n_features)` (`_factor_analysis.py:115`/`:306`; Probe 1 `(2,5)`); ferrolearn `components` is `(n_features, n_components)` (`FittedFactorAnalysis.components`; `(5,2)`) — i.e. sklearn's `components_.T` (impl stores `W.T`, preserving the established public layout). The `inverse_transform` doc-comment notes the transpose. A consumer comparing arrays (R-DEV-3) sees a transposed matrix. |
| REQ-18 (production `assert_eq!` debug-assert in `transform`) | NOT-STARTED | open prereq blocker **#1538** (R-CODE-2 / R-APG-1 flag). `impl Transform for FittedFactorAnalysis in factor_analysis.rs` ends with `assert_eq!(scores.dim(), (n_samples, k))` — a `panic!`-family macro OUTSIDE `#[cfg(test)]` (library must return `Result<_, FerroError>`, never panic). A self-consistency check that cannot fire given the preceding algebra, but it violates the no-panic discipline; remove or convert to a returned error. |
| REQ-19 (ferray substrate) | NOT-STARTED | open prereq blocker **#1539**. `factor_analysis.rs` computes on `ndarray::{Array1, Array2}` (`use ndarray`), dispatches the SVD to `faer` (`factor_analysis_svd_f64`/`_f32`), and uses a hand-rolled Cholesky inverse (`fn cholesky_inv`), not `ferray-core` / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`factor_analysis.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`FactorAnalysis<F> { n_components, max_iter (1000), tol (1e-3), random_state }`
(`struct FactorAnalysis` in `factor_analysis.rs`; `new(n_components)`, builders
`with_max_iter` / `with_tol` / `with_random_state`, accessor `n_components()`; `Default`
→ `new(1)`) → `Fit<Array2<F>, ()>` → `FittedFactorAnalysis<F> { components (n_features,
n_components), noise_variance, mean, n_iter, log_likelihood }` (`struct
FittedFactorAnalysis`, accessors `components()`/`noise_variance()`/`mean()`/`n_iter()`/
`log_likelihood()`, `inverse_transform`). The path is generic over `F: Float + Send +
Sync + 'static` (both f32 and f64); `fit`/`transform`/`inverse_transform` return
`Result<_, FerroError>` (R-CODE-2) — EXCEPT the production `assert_eq!` in `transform`
(REQ-18 flag).

**Fit path (`pub fn fit` in `factor_analysis.rs`) — REQ-1/3/4/6.** Validates
`n_components != 0`, `n_components <= n_features`, `n_samples >= 2` (REQ-6). Step 1:
per-feature `mean` + centering = sklearn `mean_ = mean(X, axis=0); X -= mean_`
(`_factor_analysis.py:231-232`). Step 2: constants `nsqrt = sqrt(n)`, `llconst = p·ln(2π)
+ k` (`:236`), `var = (1/n) Σ Xc²` (`:237`), `psi = ones` (`:239-240`), `SMALL = 1e-12`
(`:252`). Step 3: the DETERMINISTIC SVD-EM loop (`_factor_analysis.py:278-297`), to
`max_iter`: `sqrt_psi = sqrt(psi) + SMALL` (`:280`); whiten `Y = Xc/(sqrt_psi·nsqrt)`
(`:281`); thin SVD of `Y` via `factor_analysis_svd` (the `svd_method='lapack'` `my_svd`,
`:258-264`) → full descending singular-value vector `s` and top-`k` right singular
vectors `Vt`; `s² = s²` (`:282`); `unexp_var = ‖s[k..]‖²` (`:263`); `W =
sqrt(max(s²-1,0))[:,None]·Vt·sqrt_psi` stored transposed as `w[j,c] = W[c,j]` (`:284`,
`:286`); `ll = (llconst + Σ_c ln(s²_c) + unexp_var + Σ_j ln(psi_j))·(-n/2)` (`:289-291`);
ONE-SIDED break `(ll - old_ll) < tol` (`:293`); `psi[j] = max(var_j - Σ_c W[c,j]², SMALL)`
(`:297`). Stores `components = w` (= `W.T`, REQ-17 layout), `noise_variance = psi`,
`mean`, `n_iter`, `log_likelihood = last_ll`. **This is now ALGEBRAICALLY sklearn's
`_fit` lapack branch** — the old random-init posterior-mean EM and `compute_log_likelihood`
are gone; `random_state` is retained for API stability but INERT (the algorithm is
deterministic).

**SVD dispatch (`fn factor_analysis_svd` in `factor_analysis.rs`) — REQ-3/19.** A `TypeId`
dispatcher that reinterprets `&Array2<F>` as `&Array2<f64>`/`&Array2<f32>` (`unsafe`
pointer cast) and `transmute_copy`s the results back, mirroring `fn eigen_dispatch in
pca.rs`. The concrete `factor_analysis_svd_f64`/`_f32` build a `faer::Mat` and call
`thin_svd()`, returning `(s, vt_top)` where `s` is the full descending singular-value
vector (length `min(n, p)`) and `vt_top` the `(k × p)` top-`k` right singular vectors
(row `c` is `V[:, c]ᵀ`). faer's sign convention for the right singular vectors differs
from LAPACK's, and FA applies no `svd_flip`, hence the per-component sign carve-out
(REQ-2).

**Transform (`impl Transform for FittedFactorAnalysis` in `factor_analysis.rs`) —
REQ-4.** Validates the column count (REQ-6), centres, forms `Σ_z = inv(I + WᵀΨ⁻¹W)` via
`cholesky_inv`, `β = Σ_z WᵀΨ⁻¹`, and returns `scores = (β Xcᵀ)ᵀ` of shape `(n_samples,
n_components)`. ALGEBRAICALLY IDENTICAL to sklearn `transform` `(X − mean_) @ Wpsi.T @
cov_z` with `Wpsi = components_/noise_variance_`, `cov_z = inv(Ih + Wpsi @ components_.T)`
(`_factor_analysis.py:333-340`) — so GIVEN the same `W`/`ψ` the scores match sklearn up to
the same per-component sign as the loadings (REQ-1/REQ-2). The trailing
`assert_eq!(scores.dim(), (n_samples, k))` is a production panic-path (REQ-18 flag).

**Internal helper.** `cholesky_inv` (`fn cholesky_inv` in `factor_analysis.rs`) inverts a
small SPD matrix via Cholesky + forward-substitution (regularising a non-positive pivot to
`1e-10`), used in `transform`'s E-step.

**sklearn (target contract).** `class FactorAnalysis(ClassNamePrefixFeaturesOutMixin,
TransformerMixin, BaseEstimator)` (`_factor_analysis.py:42`) takes
`__init__(n_components=None, *, tol=1e-2, copy=True, max_iter=1000,
noise_variance_init=None, svd_method="randomized", iterated_power=3, rotation=None,
random_state=0)` (`:181-204`). `fit`/`_fit` (`:206-312`) centres, inits `psi=ones`, builds
`my_svd` for the chosen `svd_method` (`lapack` exact `:256-264` / `randomized` RNG
`:266-276`), runs the SVD-EM loop (`:278-297`), stores `components_` `(n_components,
n_features)` (`:306`), optional `_rotate` (`:307-308`), `noise_variance_` (`:309`),
`loglike_` list (`:310`), `n_iter_` (`:311`). `transform` (`:314-342`) computes the
posterior mean. `get_covariance` (`:344-358`) / `get_precision` (`:360-386`) reconstruct
`Wᵀ·W + diag(ψ)` and its inverse; `score`/`score_samples` (`:388-426`) the Gaussian
log-likelihood. Fitted attrs: `components_`, `loglike_`, `noise_variance_`, `n_iter_`,
`mean_`, `n_features_in_`.

## Verification

Library-crate gauntlet (live oracle = installed sklearn 1.5.2, run from `/tmp` with
`svd_method='lapack'`):

```bash
cargo test -p ferrolearn-decomp --lib factor_analysis      # 17 in-module structural tests
cargo test -p ferrolearn-decomp --test divergence_factor_analysis  # 13 parity + green-guards
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```

The 17 in-module `#[cfg(test)]` tests pin the STRUCTURAL contract (REQ-4,5,6,7,8): shapes
(`test_fa_fit_returns_fitted`, `test_fa_components_accessor`, `test_fa_transform_shape`,
`test_fa_transform_new_data`, `test_fa_mean_shape`), positivity/finiteness
(`test_fa_noise_variance_positive`, `test_fa_n_iter_positive`,
`test_fa_log_likelihood_finite`, `test_fa_scores_not_all_zero`), determinism
(`test_fa_reproducible_with_seed`), error contracts (`test_fa_error_zero_components`,
`test_fa_error_too_many_components`, `test_fa_error_insufficient_samples`,
`test_fa_transform_shape_mismatch`), and the pipeline (`test_fa_pipeline_transformer`).

The 13 tests in `tests/divergence_factor_analysis.rs` pin the VALUE parity (REQ-1/REQ-3)
and structural guards against the live sklearn LAPACK oracle (R-CHAR-3, never copied from
ferrolearn):

- `divergence_fa_rotation_invariant_covariance` / `divergence_fa_simple_data_loglike` —
  compare ferrolearn's `noise_variance()`, the reconstructed `componentsᵀ·components +
  diag(noise_variance)`, and the converged `log_likelihood()` against the live
  `sklearn.FactorAnalysis(svd_method='lapack')` oracle to a documented tolerance. Both PASS
  (the FIXED SVD-EM matches sklearn well within the asserted bound) — these are the REQ-1
  value-parity pins.
- `carveout_fa_loadings_only_rotation_invariant` — asserts only the sign-invariant implied
  covariance is symmetric, documenting the per-component loadings SIGN carve-out (REQ-2,
  R-DEFER-3); it deliberately does NOT pin `components()` element-wise.
- `green_components_shape_is_features_by_components`, `green_transform_scores_shape`,
  `green_noise_variance_strictly_positive`, `green_log_likelihood_finite_and_n_iter_in_range`,
  `green_determinism_same_seed_identical`, `green_error_zero_components`,
  `green_error_too_many_components`, `green_error_insufficient_samples`,
  `green_error_transform_col_mismatch`, `green_inverse_transform_shape_and_col_mismatch` —
  the structural green-guards (REQ-4/5/6).

Independent re-audit cross-check (Probe 2, fresh 12×5 fixture, matched tol=1e-3): ferrolearn
`FactorAnalysis::<f64>::new(2).with_tol(1e-3)` vs `sklearn.FactorAnalysis(n_components=2,
svd_method='lapack', tol=1e-3)` — `noise_variance` max diff 2.5e-14, `get_covariance()` diag
max diff 7.1e-15, `log_likelihood` vs `loglike_[-1]` diff 1.4e-14, `n_iter` 48==48,
`components()` vs sklearn `components_.T` after per-component sign alignment 4.8e-14. (The
task-reported fresh-fixture audit: noise 3.3e-15, cov 2.9e-15, ll 7.1e-15, n_iter 36==36,
components 1.6e-13.)

## Blockers

- `#1528` — REQ-2, CARVE-OUT (R-DEFER-3): per-component `components_` SIGN — faer's SVD
  right-singular-vector sign differs from LAPACK's; sklearn FA applies no `svd_flip`, so no
  deterministic sign target exists. No failing test.
- `#1529` — REQ-9: `svd_method='randomized'` + RNG path (`iterated_power`); ferrolearn
  implements only the deterministic `lapack` branch.
- `#1530` — REQ-10: `rotation` ('varimax' / 'quartimax').
- `#1531` — REQ-11: `noise_variance_init`.
- `#1532` — REQ-12: `loglike_` per-iteration LIST attribute (ferrolearn keeps only the
  final scalar `log_likelihood`).
- `#1533` — REQ-13: `score` / `score_samples` (Gaussian log-likelihood).
- `#1534` — REQ-14: `get_covariance` / `get_precision` METHODS (the covariance VALUE
  matches via accessors, but no method).
- `#1535` — REQ-15: `n_components=None` default + `copy` + `n_features_in_`.
- `#1536` — REQ-16: `tol` DEFAULT (1e-3 vs 1e-2) — the convergence CRITERION now matches.
- `#1537` — REQ-17: `components_` orientation (ferrolearn `(n_features, n_components)` =
  sklearn `components_.T`).
- `#1538` — REQ-18: production `assert_eq!` debug-assert in `transform` (R-CODE-2 /
  R-APG-1).
- `#1539` — REQ-19: ferray substrate migration.
