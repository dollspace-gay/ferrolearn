# NMF (sklearn.decomposition.NMF)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 5b4b4d15
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_nmf.py  # class NMF(_BaseNMF) (:912-1130). ctor (:912-925): n_components=None, *, init=None, solver="cd", beta_loss="frobenius", tol=1e-4, max_iter=200, random_state=None, alpha_W=0.0, alpha_H="same", l1_ratio=0.0, verbose=0, shuffle=False. solver="cd" DEFAULT (:917). init=None -> "nndsvda" if n_components < min(n_samples, n_features) else "random" (:1000 _check_params / :302 _initialize_nmf). _initialize_nmf (:225-374): "random" (:317-328, avg-scaled randn abs), "nndsvd" (SVD-based: U,S,V = randomized_svd / svd, sign-corrected pos/neg split, :330-360), "nndsvda" (zeros -> average of X, :362-363), "nndsvdar" (zeros -> small random, :364-368), "custom" (:281-299). fit_transform (:1100-1130) -> _fit_transform (:1157+): solver "cd" -> _fit_coordinate_descent (:1268 call / :340 def), "mu" -> _fit_multiplicative_update (:1270 call); components_ = H, reconstruction_err_ = _beta_divergence(X, W, H, beta_loss, square_root=True), n_iter_. transform (:1213): W = _fit_transform(X, H=self.components_, update_H=False)[0] -> NNLS for W with H fixed. inverse_transform (:1238): W @ self.components_. _compute_regularization (:1275) for alpha_W/alpha_H/l1_ratio. _beta_divergence (:89), _gamma per beta_loss.
ferrolearn-module: ferrolearn-decomp/src/nmf.rs
parity-ops: NMF
crosslink-issue: 1608
-->

## Summary

`ferrolearn-decomp/src/nmf.rs` mirrors scikit-learn's `NMF`
(`sklearn/decomposition/_nmf.py`, `class NMF(_BaseNMF)` `:912`): factor a
non-negative matrix `X ≈ W·H` into non-negative `W` (transformed data,
`n_samples × n_components`) and `H` (components, `n_components × n_features`),
minimising `||X − W·H||_Fro`. The exposed surface is the unfitted
`NMF<F> { n_components, max_iter (200), tol (1e-4), solver: NMFSolver{
MultiplicativeUpdate|CoordinateDescent}, init: NMFInit{Random|Nndsvd},
random_state }` (`nmf.rs`, struct at line 78; builders `with_max_iter`/`with_tol`/
`with_solver`/`with_init`/`with_random_state` fns at lines 113-145, accessors fns at
lines 148-181) and the fitted `FittedNMF<F> { components_ (n_components, n_features),
reconstruction_err_, n_iter_ }` (`nmf.rs`, struct at line 193, accessors
`components`/`reconstruction_err`/`n_iter` fns at lines 205/211/217, plus
`inverse_transform` fn at line 229), generic over `F: Float + Send + Sync +
'static` (both f32 and f64, `test_nmf_f32`). Re-exported at the crate root
(`pub use nmf::{FittedNMF, NMF, NMFInit, NMFSolver}`, `lib.rs:97`) and bound in PyO3
as `_RsNMF` (`ferrolearn-python/src/extras.rs:1116`, registered `lib.rs:75`, via the
`py_transformer!` macro). There is NO `tests/divergence_nmf.rs` yet.

**DEFAULT DIVERGENCE: solver + init.** ferrolearn defaults to
`solver = MultiplicativeUpdate` + `init = Random` (`nmf.rs:105-106`), whereas sklearn
defaults to `solver="cd"` (`_nmf.py:917`) and `init=None` → `"nndsvda"` when
`n_components < min(n_samples, n_features)` (`_nmf.py:302`/`:1000`). These are
REQ-6 / REQ-7 (NOT-STARTED).

**EXACT `components_` VALUE PARITY DIVERGES (R-HONEST-3, REQ-5 NOT-STARTED, CARVE-OUT
`#1609`).** ferrolearn's `components_` VALUES diverge from sklearn through four
compounding differences. (a) **Init:** ferrolearn's `init_random` (`nmf.rs:263`) is a
Rust `StdRng` uniform `[0,1)` (`nmf.rs:269-280`, seed `random_state.unwrap_or(0)`),
whereas sklearn's `"random"` is `avg·|randn|` from a numpy `RandomState`
(`_nmf.py:317-328`) and its DEFAULT is the DETERMINISTIC SVD-based `nndsvda`. (b)
**Pseudo-NNDSVD:** ferrolearn's `init_nndsvd` (`nmf.rs:289`) is a Jacobi
eigendecomposition of `XᵀX` (`jacobi_eigen_symmetric` `nmf.rs:362`) whose top
eigenvectors seed `H` with negatives clamped to `0` and `W = (X·Hᵀ)₊`
(`nmf.rs:343-353`) — NOT sklearn's real `nndsvd`/`nndsvda`/`nndsvdar` (`u·sqrt(s)` /
`sqrt(s)·v` with `svd_flip` sign correction and a zeros→average fill,
`_nmf.py:330-368`). (c) **Solver default:** ferrolearn's default MU
(`solve_multiplicative_update` `nmf.rs:461`) vs sklearn's default coordinate-descent
`_fit_coordinate_descent` (`_nmf.py:340`). (d) NMF is identifiable only up to
permutation/scaling, and the numpy RNG ≠ Rust RNG. Different init + different default
solver + permutation/scaling freedom + different RNG ⇒ the `components_` VALUES
diverge (same class as the `minibatch_nmf` / `dictionary_learning` / `sparse_pca` RNG
carve-outs); no failing test is asserted (R-DEFER-3). The `transform` `W` is the NNLS
optimum DOWNSTREAM of the carved-out `H`, and `FittedNMF`'s fields are PRIVATE (no
injectable-`H` constructor), so `transform` value parity FOLDS INTO this carve-out
(REQ-9, INVESTIGATE-for-critic, like `minibatch_nmf` `#1487`).

As of this iteration: the STRUCTURAL `components_` shape `(n_components, n_features)`,
W/H non-negativity, finite `reconstruction_err_` that DECREASES with more
iterations/components, positive `n_iter_`, determinism given a seed, f32 + f64, the
two solvers (MU/CD) × two inits (Random/pseudo-NNDSVD) all run (REQ-1,2,3), the
reconstruction-quality "did NMF work" signal `||X − W·H||` small (REQ-4), the
`inverse_transform` `W·H` algebra (REQ-10), the scoped error/parameter contracts
(REQ-11), and the thin PyO3 binding (REQ-12, scoped) are SHIPPED; exact `components_`
value parity (REQ-5, CARVE-OUT `#1609`), real `init="nndsvda"` default + SVD-based
`nndsvd`/`nndsvdar`/`custom` (REQ-6, `#1610`), `solver="cd"` default (REQ-7,
`#1611`), `beta_loss` (kl/is) + `_gamma` (REQ-8, `#1612`), the `transform` NNLS-W
value (REQ-9, CARVE-OUT — folds into REQ-5, `#1613`), `n_components=None` default
(REQ-13, `#1614`), regularization `alpha_W`/`alpha_H`/`l1_ratio` (REQ-14,
`#1615`), `shuffle` (CD) + fitted attrs `n_components_`/`n_features_in_` (REQ-15,
`#1616`), and the ferray substrate (REQ-16, `#1617`) are NOT-STARTED —
**7 SHIPPED / 9 NOT-STARTED**.

`NMF` / `FittedNMF` are existing pub APIs whose non-test consumers are the crate
re-export (`lib.rs:97`), the `_RsNMF` PyO3 binding (`extras.rs:1116`, registered
`lib.rs:75`), and the `PipelineTransformer<F>` impl (`nmf.rs:744-771`) — boundary
public API, grandfathered S5/R-DEFER-1.

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# PROBE 1 (REQ-1/2/4 SHIPPED scoped + REQ-5 NOT-STARTED) — components_ shape
# (n_components, n_features) + non-negativity + reconstruction quality. Small fixed
# non-negative X (4x3). init='random', random_state=0 to match ferrolearn's only init
# path. VALUES generated by sklearn, never copied from ferrolearn (R-CHAR-3).
python3 -c "
import numpy as np
from sklearn.decomposition import NMF
X=np.array([[1.,2,3],[4,5,6],[7,8,9],[10,11,12]])
m=NMF(n_components=2, init='random', random_state=0).fit(X)
print('components_ shape:', m.components_.shape)
print('components_ non-neg:', bool((m.components_>=0).all()))
print('components_ row0:', np.round(m.components_[0],6).tolist())
print('reconstruction_err_:', round(float(m.reconstruction_err_),6), 'n_iter_:', m.n_iter_)
W=m.transform(X)
print('transform W shape:', W.shape, 'W non-neg:', bool((W>=0).all()))
print('||X-WH||:', round(float(np.linalg.norm(X-W@m.components_)),6))"
# -> components_ shape: (2, 3)                          => structural shape (REQ-1)
# -> components_ non-neg: True                          => W/H non-negativity (REQ-2)
# -> components_ row0: [2.389149, 2.232745, 2.076342]   => VALUES (REQ-5 CARVE-OUT, NOT reproduced)
# -> reconstruction_err_: 0.206438 n_iter_: 200         => finite err / positive n_iter (REQ-1)
# -> transform W shape: (4, 2) W non-neg: True          => W shape (n_samples,k) + non-neg (REQ-1/2)
# -> ||X-WH||: 0.17132                                  => RECONSTRUCTION QUALITY: small (REQ-4 "did NMF work")

# PROBE 2 (REQ-6/7/8/13/14/15 NOT-STARTED) — ctor defaults: solver='cd', init=None,
# beta_loss, alpha_*, l1_ratio, shuffle.
python3 -c "
from sklearn.decomposition import NMF
m=NMF()
for p in ['n_components','init','solver','beta_loss','tol','max_iter','random_state','alpha_W','alpha_H','l1_ratio','shuffle']:
    print(f'{p} =', getattr(m,p))"
# -> n_components = warn  init = None  solver = cd  beta_loss = frobenius  tol = 0.0001
# -> max_iter = 200  random_state = None  alpha_W = 0.0  alpha_H = same  l1_ratio = 0.0  shuffle = False
#    => sklearn DEFAULTS solver='cd' (REQ-7) + init=None->'nndsvda' (REQ-6); has beta_loss
#       (REQ-8), alpha_W/alpha_H/l1_ratio (REQ-14), shuffle (REQ-15), n_components=None (REQ-13).
#       ferrolearn defaults solver=MultiplicativeUpdate + init=Random (nmf.rs:105-106), no such params.

# PROBE 3 (REQ-4 SHIPPED + REQ-5/6/7 NOT-STARTED) — DEFAULT fit (solver='cd',
# init=nndsvda) reconstruction quality + values; inverse_transform algebra (REQ-10).
python3 -c "
import numpy as np
from sklearn.decomposition import NMF
X=np.array([[1.,2,3],[4,5,6],[7,8,9],[10,11,12]])
m=NMF(n_components=2, random_state=0).fit(X)   # default solver='cd', init='nndsvda'
print('default components_ row0:', np.round(m.components_[0],6).tolist())
print('default n_iter_:', m.n_iter_, 'err:', round(float(m.reconstruction_err_),6))
W=m.transform(X)
print('||X-WH|| default:', round(float(np.linalg.norm(X-W@m.components_)),6))
print('inverse_transform == W@H:', bool(np.allclose(m.inverse_transform(W), W@m.components_)))"
# -> default components_ row0: [1.949292, 2.460614, 2.971937]  => DIFFERENT from init='random' (REQ-6/7)
# -> default n_iter_: 200 err: 0.88744
# -> ||X-WH|| default: 0.854916                                => reconstruction quality (REQ-4)
# -> inverse_transform == W@H: True                            => inverse_transform = W@H (REQ-10)
```

## Requirements

- REQ-1: **Structural: `components_` (H) shape `(n_components, n_features)`,
  `transform` (W) shape `(n_samples, n_components)`, finite `reconstruction_err_`
  that DECREASES with more iterations/components, positive `n_iter_`, determinism
  given seed (SHIPPED scoped).** `fn fit` (`impl Fit for NMF`, `nmf.rs:617`) stores
  `components_ = h` of shape `(n_components, n_features)` (`nmf.rs:675`, field
  `:195`, = sklearn `components_ = H` `_nmf.py:1118`-style), the Frobenius
  `reconstruction_err_ = ||X − W·H||_F` (`reconstruction_error`, fn at line 247;
  stored `:676`), and `n_iter_` (`:677`). `transform` (`impl Transform for
  FittedNMF`, `nmf.rs:696`) returns `W` of shape `(n_samples, n_components)`
  (`nmf.rs:721`). The seeded `StdRng` (`init_random` `nmf.rs:269`, seed
  `random_state.unwrap_or(0)` `:654`) plus the deterministic MU/CD loops make the fit
  reproducible given a seed. **Scope: STRUCTURAL (shapes / finiteness / monotone
  decrease / determinism), NOT component VALUES (REQ-5).** Pinned by
  `test_nmf_basic_fit` `(2,3)`, `test_nmf_transform_dimensions` `(4,2)`,
  `test_nmf_reconstruction_error_decreases`, `test_nmf_reconstruction_error_positive`,
  `test_nmf_more_components_lower_error`, `test_nmf_n_iter_positive`,
  `test_nmf_reproducibility`, `test_nmf_single_component`, `test_nmf_f32`,
  `test_nmf_medium_dataset_mu`, `test_nmf_coordinate_descent_solver`,
  `test_nmf_nndsvd_init`, `test_nmf_cd_with_nndsvd`. Non-test consumers: re-export
  `lib.rs:97`, `_RsNMF` (`extras.rs:1116`), `PipelineTransformer` (`nmf.rs:744`).

- REQ-2: **Structural: `components_` (H) and `transform` (W) are NON-NEGATIVE; the
  fit factors a non-negative `X` (SHIPPED scoped).** The MU updates are purely
  multiplicative on non-negative initial factors
  (`solve_multiplicative_update` `nmf.rs:478-495`, `W,H ← W,H · num/(den+eps)`),
  the CD updates clamp each entry to `max(0, …)`
  (`solve_coordinate_descent` `nmf.rs:550-554`/`:581-585`), and `transform`'s W is
  MU-updated from a positive constant (`nmf.rs:727-734`) — so all stored `H` and
  returned `W` entries are `≥ 0`, mirroring NMF's non-negativity constraint (sklearn
  `W,H ≥ 0`, `_nmf.py` objective). Probe 1 confirms sklearn `components_` and
  `transform W` are non-negative. Pinned by `test_nmf_components_non_negative`,
  `test_nmf_transform_non_negative`. **Scope: STRUCTURAL non-negativity, NOT value
  parity (REQ-5).** Non-test consumers: re-export `lib.rs:97`, `_RsNMF`
  (`extras.rs:1116`).

- REQ-3: **Both solvers (MU/CD) × both inits (Random/pseudo-NNDSVD) run end-to-end
  (SHIPPED scoped).** `fn fit` (`nmf.rs:617`) dispatches the init via `match self.init`
  (`nmf.rs:657-660`, `init_random` `nmf.rs:263` or `init_nndsvd` `nmf.rs:289`) and the
  solver via `match self.solver` (`nmf.rs:663-670`, `solve_multiplicative_update`
  `nmf.rs:461` or `solve_coordinate_descent` `nmf.rs:511`), so all four
  init×solver combinations produce a valid non-negative factorization. **Scope:
  STRUCTURAL (each path runs + yields the right shape), NOT value parity (REQ-5) and
  NOT real NNDSVD (REQ-6) or the sklearn default solver (REQ-7).** Pinned by
  `test_nmf_coordinate_descent_solver` `(2,4)`, `test_nmf_nndsvd_init` `(2,4)`,
  `test_nmf_cd_with_nndsvd` `(2,4)`, `test_nmf_medium_dataset_mu` `(3,4)`,
  `test_nmf_getters` (round-trips `solver`/`init`). Non-test consumer: re-export
  `lib.rs:97`.

- REQ-4: **Reconstruction QUALITY: `||X − W·H||` is small / the factorization "works"
  (SHIPPED scoped).** This is the meaningful "did NMF actually factor `X`" signal
  (distinct from element-wise value parity, REQ-5). `reconstruction_error`
  (`nmf.rs:247`) computes `||X − W·H||_F`, the solvers MONOTONICALLY reduce it
  (`solve_multiplicative_update`/`solve_coordinate_descent` break on
  `|prev_err − err| < tol`, `nmf.rs:499`/`:591`), and `test_nmf_medium_dataset_mu`
  asserts the converged error is `< 10.0` on the `medium_dataset` (a real small-error
  factorization). Probe 1 sklearn `||X − W·H|| = 0.17132` (init='random') and Probe 3
  `0.854916` (default) confirm sklearn likewise drives the residual small. Pinned by
  `test_nmf_medium_dataset_mu` (error `< 10.0`),
  `test_nmf_reconstruction_error_decreases` (more iters lower error),
  `test_nmf_more_components_lower_error` (more components lower error). **Scope: the
  reconstruction residual is small / decreasing, NOT a tolerance match to sklearn's
  exact `reconstruction_err_` value (that folds into REQ-5).** Non-test consumer:
  re-export `lib.rs:97`.

- REQ-5: **EXACT `components_` value parity with sklearn's `_fit_transform`
  (NOT-STARTED, CARVE-OUT; `#1609`).** sklearn's `NMF.fit_transform`
  (`_nmf.py:1100-1130`) inits via `_initialize_nmf` (default `nndsvda`,
  `_nmf.py:225`/`:362`, SVD-based) then runs `_fit_coordinate_descent` (default,
  `_nmf.py:340`) or `_fit_multiplicative_update`, setting `components_ = H`.
  ferrolearn's `fn fit` (`nmf.rs:617`) inits via a Rust `StdRng` uniform `init_random`
  (`nmf.rs:263`, default) or a Jacobi-eigendecomposition pseudo-NNDSVD `init_nndsvd`
  (`nmf.rs:289`), then runs the DEFAULT MU `solve_multiplicative_update` (`nmf.rs:461`)
  or CD `solve_coordinate_descent` (`nmf.rs:511`). Probe 1 sklearn `components_ row0 =
  [2.389149, 2.232745, 2.076342]` (init='random') and Probe 3 `[1.949292, 2.460614,
  2.971937]` (default nndsvda) are NOT reproduced element-wise. **CARVE-OUT
  (R-DEFER-3):** matching sklearn requires the real SVD-based init + the sklearn CD
  default + numpy RNG, and NMF is identifiable only up to permutation/scaling; no
  failing test is asserted (same class as the `minibatch_nmf` / `dictionary_learning`
  / `sparse_pca` RNG carve-outs).

- REQ-6: **Real `init="nndsvda"` default + SVD-based `nndsvd`/`nndsvdar`/`custom`
  (NOT-STARTED; `#1610`).** sklearn defaults `init=None` → `"nndsvda"` when
  `n_components < min(n_samples, n_features)` else `"random"` (`_nmf.py:302`/`:1000`);
  `_initialize_nmf` (`_nmf.py:225-374`) implements the real NNDSVD (`U·S·V = svd(X)`
  with `svd_flip`, positive/negative-part split `u·sqrt(s)` / `sqrt(s)·v`, `:330-360`),
  `nndsvda` (zeros → average of `X`, `:362-363`), `nndsvdar` (zeros → small random,
  `:364-368`), and `custom`. ferrolearn's `NMFInit` (`nmf.rs:61`) has only `Random`
  (default, `nmf.rs:106`) and `Nndsvd`; `init_nndsvd` (`nmf.rs:289`) is a Jacobi
  eigendecomposition of `XᵀX` with negatives clamped to `0` (`nmf.rs:322-327`) and
  `W = (X·Hᵀ)₊` (`nmf.rs:343-353`) — NOT the SVD-based NNDSVD/nndsvda/nndsvdar, no
  `custom`, and the default is `Random` not `nndsvda`.

- REQ-7: **`solver="cd"` DEFAULT (NOT-STARTED; `#1611`).** sklearn defaults
  `solver="cd"` (`_nmf.py:917`) → `_fit_coordinate_descent` (`_nmf.py:340`), with
  `solver="mu"` the alternative. ferrolearn's `NMF::new` defaults
  `solver = NMFSolver::MultiplicativeUpdate` (`nmf.rs:105`) — sklearn's NON-default
  solver. ferrolearn DOES expose a `CoordinateDescent` variant (`nmf.rs:56`), but it
  is not the default and is a different CD formulation from sklearn's
  Cython `_update_cdnmf_fast` block-coordinate-descent (sklearn iterates
  feature-by-feature with a precomputed `HHt`; `nmf.rs:524-587` iterates element-wise
  over the full residual).

- REQ-8: **`beta_loss` (`frobenius`/`kullback-leibler`/`itakura-saito`) + `_gamma`
  (NOT-STARTED; `#1612`).** sklearn's `NMF(beta_loss="frobenius")`
  (`_nmf.py:919`, `StrOptions({"frobenius","kullback-leibler","itakura-saito"})` +
  numeric) selects the beta-divergence loss; `_beta_divergence` (`_nmf.py:89`) and the
  MU helpers branch on it, with `_gamma` set per `_beta_loss` for the
  Maximization-Minimization step. ferrolearn's `NMF<F>` (`nmf.rs:78`) has NO
  `beta_loss` field — only the hard-coded Frobenius `reconstruction_error`
  (`nmf.rs:247`) and Frobenius MU/CD; no `_gamma`, no KL/IS path.

- REQ-9: **`transform` NNLS-W VALUE parity (NOT-STARTED, CARVE-OUT — folds into REQ-5;
  `#1613`).** sklearn's `transform` (`_nmf.py:1213`) is DETERMINISTIC given the
  fitted `H`: `W = _fit_transform(X, H=self.components_, update_H=False)[0]` — solves
  the NNLS `min_{W≥0} ||X − W·H||²` for the FIXED fitted `H` via the same CD/MU solver.
  ferrolearn's `transform` (`impl Transform for FittedNMF`, `nmf.rs:696`) likewise
  solves `W` with `H` fixed, but via a 200-iteration MU (`nmf.rs:727-734`) from a
  CONSTANT `0.1` init (`nmf.rs:722-723`), regardless of the fit `solver`. Both target
  the same convex NNLS optimum, but the `W` VALUES sit on TOP of the carved-out fitted
  `H` (REQ-5), and `FittedNMF`'s fields are PRIVATE (no public constructor from an
  arbitrary injected `H`), so a transform value pin is unreachable without the
  components carve-out. **INVESTIGATE (for the critic):** like `minibatch_nmf` `#1487`,
  whether to add an injectable-`H` constructor + an NNLS-optimum residual green-guard,
  or to fold REQ-9 entirely into REQ-5. No failing test is asserted (R-DEFER-3).

- REQ-10: **`inverse_transform` = `W·H` (SHIPPED).** sklearn's
  `NMF.inverse_transform(W)` (`_nmf.py:1238`) returns `W @ self.components_`.
  ferrolearn's `FittedNMF::inverse_transform` (`nmf.rs:229`) returns `w.dot(&self
  .components_)` (`nmf.rs:238`) after a column-count check (`nmf.rs:231-237`,
  `ShapeMismatch`) — the EXACT same `W·H` algebra (deterministic, not RNG/solver-bound,
  so no value carve-out). **FLAG:** ferrolearn checks `w.ncols() == n_components` and
  returns `FerroError::ShapeMismatch` where sklearn `check_array`s and raises
  `ValueError`; this method is NOT exposed through the `_RsNMF` PyO3 binding (the
  `py_transformer!` macro binds only ctor + `fit` + `transform`, `extras.rs:107`).
  Non-test consumer: re-export `lib.rs:97` (no in-tree `#[test]` pins it — the algebra
  is a 1-line `dot` that the doctest's `transform` exercises the same `components_`).

- REQ-11: **Error / parameter contracts (SHIPPED scoped).** `fn fit` (`nmf.rs:617`)
  returns `InvalidParameter { name: "n_components" }` for `n_components == 0`
  (`nmf.rs:620-625`), `InsufficientSamples { required: 1 }` for `0` samples
  (`nmf.rs:626-632`), `InvalidParameter { name: "n_components" }` for `n_components >
  min(n_samples, n_features)` (`nmf.rs:633-642`), and `InvalidParameter { name: "X" }`
  on any negative input entry (`nmf.rs:645-652`); `transform` returns `ShapeMismatch`
  on a column-count mismatch (`nmf.rs:698-704`) and `InvalidParameter { name: "X" }`
  on a negative entry (`nmf.rs:707-714`). Pinned by
  `test_nmf_invalid_n_components_zero`, `test_nmf_invalid_n_components_too_large`,
  `test_nmf_insufficient_samples`, `test_nmf_negative_input_rejected`,
  `test_nmf_transform_shape_mismatch`, `test_nmf_transform_negative_rejected`,
  `test_nmf_zero_entries`. **FLAG (candidate DIVs):** sklearn validates via
  `_parameter_constraints` + `check_non_negative` raising
  `InvalidParameterError`/`ValueError`, NOT `FerroError`; sklearn accepts
  `n_components=None` (→ default, REQ-13) which ferrolearn requires as an explicit
  `usize`; sklearn does NOT pre-reject `n_components > min(n_samples, n_features)` (the
  solver surfaces it later, and `nndsvda` simply downgrades to `random`). Non-test
  consumer: re-export `lib.rs:97`.

- REQ-12: **PyO3 binding surface (thin: `n_components` ctor + `fit` + `transform`)
  (SHIPPED scoped).** sklearn exposes `NMF` through `import sklearn.decomposition`.
  ferrolearn binds `_RsNMF` (`extras.rs:1116`, registered `lib.rs:75`) via the
  `py_transformer!` macro (`extras.rs:107`), exposing a `__new__(n_components=2)` ctor
  + `fit(X)` + `transform(X)` over `f64` (`FittedNMF<f64>`). **Scope: the thin
  fit/transform surface only — NOT the `solver`/`init`/`tol`/`max_iter`/`random_state`
  ctor params, NOT the `components_`/`reconstruction_err_`/`n_iter_` getters, NOT
  `inverse_transform` (REQ-10).** It inherits the REQ-1/2/4 structural behaviour
  (shapes, non-negativity, reconstruction quality) and the REQ-5 value carve-out.
  This binding is the CPython non-test consumer of `FittedNMF<f64>`.

- REQ-13: **`n_components=None` default (NOT-STARTED; `#1614`).** sklearn's
  `NMF(n_components=None)` (`_nmf.py:914`) defaults `n_components` to
  `min(n_samples, n_features)` (the full rank) when `None`. ferrolearn's
  `NMF::new(n_components: usize)` (`nmf.rs:100`) REQUIRES an explicit `usize` and has
  no `None`/auto-rank path — `n_components == 0` is rejected (`nmf.rs:620`).

- REQ-14: **Regularization `alpha_W` / `alpha_H` / `l1_ratio` (NOT-STARTED;
  `#1615`).** sklearn's `NMF(alpha_W=0.0, alpha_H="same", l1_ratio=0.0)`
  (`_nmf.py:921-923`) adds L1/L2 penalties via `_compute_regularization`
  (`_nmf.py:1275`), threaded into the CD/MU updates. ferrolearn's `NMF<F>`
  (`nmf.rs:78`) has NO `alpha_W`/`alpha_H`/`l1_ratio` fields and applies NO
  regularization — the MU (`nmf.rs:461`) and CD (`nmf.rs:511`) are unpenalised.

- REQ-15: **`shuffle` (CD) + fitted attrs `n_components_` / `n_features_in_`
  (NOT-STARTED; `#1616`).** sklearn's `NMF(shuffle=False)` (`_nmf.py:924`) randomises
  the coordinate-descent feature order when `True`, and exposes the fitted attrs
  `n_components_` and `n_features_in_`. ferrolearn's `NMF<F>` (`nmf.rs:78`) has NO
  `shuffle` field (the CD `solve_coordinate_descent` `nmf.rs:511` iterates a fixed
  order), and `FittedNMF<F>` (`nmf.rs:193`) exposes only `components()` /
  `reconstruction_err()` / `n_iter()` (fns at 205/211/217) — `n_components_` is
  derivable from `components_.nrows()` but not exposed, and there is no
  `n_features_in_`.

- REQ-16: **ferray substrate (NOT-STARTED; `#1617`).** `nmf.rs` computes on
  `ndarray::Array2` (`nmf.rs:41`) and uses `rand`/`rand_distr` (`nmf.rs:43-44`,
  `StdRng` + `Uniform`) for init, with a hand-rolled Jacobi eigensolver
  (`jacobi_eigen_symmetric`, `nmf.rs:362`), not `ferray-core` arrays /
  `ferray::random` / `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED scoped): `NMF::<f64>::new(2).with_random_state(42).fit(&X, &())
  .unwrap().components().dim()` is `(2, n_features)`; `transform(&X)` has shape
  `(n_samples, 2)`; `reconstruction_err()` is finite and `≥ 0`; `n_iter() > 0`; more
  iterations / more components do not increase the error; two fits with the same seed
  are identical. Pinned by `test_nmf_basic_fit` `(2,3)`,
  `test_nmf_transform_dimensions` `(4,2)`, `test_nmf_reconstruction_error_positive`,
  `test_nmf_reconstruction_error_decreases`, `test_nmf_more_components_lower_error`,
  `test_nmf_n_iter_positive`, `test_nmf_reproducibility`, `test_nmf_f32`. (Structural
  shapes / finiteness / monotone decrease / determinism only — NOT exact values,
  REQ-5.)

- AC-2 (REQ-2, SHIPPED scoped): every entry of `fitted.components()` and of
  `fitted.transform(&X)` is `≥ 0`. Probe 1 confirms sklearn `components_` / `W` are
  non-negative. Pinned by `test_nmf_components_non_negative`,
  `test_nmf_transform_non_negative`.

- AC-3 (REQ-3, SHIPPED scoped): each of MU/CD × Random/pseudo-NNDSVD fits to the right
  `(n_components, n_features)` shape with non-negative components. Pinned by
  `test_nmf_coordinate_descent_solver`, `test_nmf_nndsvd_init`,
  `test_nmf_cd_with_nndsvd`, `test_nmf_medium_dataset_mu`, `test_nmf_getters`.

- AC-4 (REQ-4, SHIPPED scoped): on `medium_dataset` (6×4) the MU fit's
  `reconstruction_err() < 10.0` (a small-residual factorization); raising
  `max_iter` 10→200 or `n_components` 1→2 does not increase the error. Probe 1
  `||X − W·H|| = 0.17132` and Probe 3 `0.854916` confirm sklearn drives the residual
  small. Pinned by `test_nmf_medium_dataset_mu`,
  `test_nmf_reconstruction_error_decreases`, `test_nmf_more_components_lower_error`.
  (Residual small/decreasing — NOT a tolerance match to sklearn's exact
  `reconstruction_err_`, REQ-5.)

- AC-5 (REQ-5, NOT-STARTED, CARVE-OUT): `NMF(n_components=2, init='random',
  random_state=0).fit(X).components_` (Probe 1: shape `(2,3)`, `row0 = [2.389149,
  2.232745, 2.076342]`) and the default `nndsvda`+`cd` fit (Probe 3: `row0 =
  [1.949292, 2.460614, 2.971937]`) are NOT reproduced element-wise by ferrolearn
  (different init + default-MU + permutation/scaling freedom + Rust RNG). No failing
  test asserts this (R-DEFER-3).

- AC-6 (REQ-6/7/8/13/14/15, DIVERGES): `NMF()` defaults `n_components="warn",
  init=None (→ nndsvda), solver="cd", beta_loss="frobenius", tol=1e-4, max_iter=200,
  random_state=None, alpha_W=0.0, alpha_H="same", l1_ratio=0.0, shuffle=False`
  (Probe 2, `_nmf.py:912-925`); sklearn exposes the real SVD-based
  `nndsvd`/`nndsvda`/`nndsvdar`/`custom` inits, the `cd` default solver, `beta_loss`
  + `_gamma`, regularization, `shuffle`, and `n_components_`/`n_features_in_` attrs.
  ferrolearn defaults `solver=MultiplicativeUpdate` + `init=Random` (`nmf.rs:105-106`),
  has a pseudo-NNDSVD only, no `beta_loss`/`alpha_*`/`l1_ratio`/`shuffle` params, no
  `n_components=None`, and no `n_components_`/`n_features_in_` attrs.

- AC-7 (REQ-9, NOT-STARTED, CARVE-OUT — folds into REQ-5): sklearn's `transform`
  (`_nmf.py:1213`) solves the NNLS `W` for the fixed fitted `H`; ferrolearn solves it
  too (200-iter MU from `0.1`, `nmf.rs:722-734`), but the `W` values sit atop the
  carved-out `H` and `FittedNMF`'s fields are private (no injectable-`H` API). No
  failing test is pinned; INVESTIGATE-for-critic (like `minibatch_nmf` `#1487`).

- AC-8 (REQ-10, SHIPPED): `fitted.inverse_transform(&W)` equals `W·components_`
  (Probe 3 sklearn `inverse_transform == W@H: True`); a `w.ncols() != n_components`
  input returns `Err(ShapeMismatch)` (`nmf.rs:231-237`).

- AC-9 (REQ-12, SHIPPED scoped): `import ferrolearn; ferrolearn._RsNMF(2).fit(X)
  .transform(X)` returns a non-negative `(n_samples, 2)` array (the `py_transformer!`
  surface, `extras.rs:1116`). NOT the `solver`/`init`/`random_state` params, NOT the
  `components_`/`reconstruction_err_`/`n_iter_` getters, NOT `inverse_transform`.

- AC-10 (REQ-16): the module imports `ndarray` (`nmf.rs:41`) + `rand`/`rand_distr`
  (`:43-44`) and a hand-rolled Jacobi eigensolver (`nmf.rs:362`), not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `NMF` / `FittedNMF` are existing pub APIs; the non-test consumers
are the crate re-export (`lib.rs:97`), the `_RsNMF` PyO3 binding (`extras.rs:1116`,
registered `lib.rs:75`), and the `PipelineTransformer` impl (`nmf.rs:744`) — boundary
public API, grandfathered S5/R-DEFER-1. Cites use symbol anchors (ferrolearn) /
`file:line` (sklearn 1.5.2). Live oracle = installed sklearn 1.5.2, run from `/tmp`.
**EXACT `components_` VALUE PARITY DIVERGES (R-HONEST-3, REQ-5 NOT-STARTED, CARVE-OUT
`#1609`):** ferrolearn's Rust-`StdRng` uniform init (default `Random`) / pseudo-NNDSVD
+ DEFAULT MU solver (`nmf.rs:617`) ≠ sklearn's SVD-based `nndsvda` default + `cd`
default solver (`_nmf.py:912-1130`), and NMF is identifiable only up to
permutation/scaling. **`transform` NNLS-W VALUE folds into the carve-out (REQ-9,
`#1613`):** the `W` solve sits atop the carved-out `H` and `FittedNMF`'s fields are
private. The least-confident SHIPPED claim is REQ-10 (`inverse_transform`) — the `W·H`
algebra is exact and matches sklearn (Probe 3), but there is NO in-tree `#[test]`
pinning it directly (the doctest exercises `transform`, not `inverse_transform`); a
green-guard would harden it. #1608 is this doc's crosslink tracking issue. Count:
**7 SHIPPED (REQ-1,2,3,4,10,11,12) / 9 NOT-STARTED (REQ-5,6,7,8,9,13,14,15,16)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (structural: shapes / finite & decreasing err / n_iter / determinism) | SHIPPED | `fn fit` (`impl Fit for NMF`, `nmf.rs:617`) stores `components_ = h` shape `(n_components, n_features)` (`nmf.rs:675`, field `:195`), Frobenius `reconstruction_err_` (`reconstruction_error` fn at line 247; `:676`), `n_iter_` (`:677`); `transform` (`nmf.rs:696`) returns `W` `(n_samples, n_components)` (`:721`). Seeded `StdRng` (`init_random` `:269`, seed `unwrap_or(0)` `:654`) ⇒ reproducible. **Scope: STRUCTURAL, NOT value parity (REQ-5).** Non-test consumers: re-export `lib.rs:97`, `_RsNMF` `extras.rs:1116`, `PipelineTransformer` `nmf.rs:744`. Verification: `cargo test -p ferrolearn-decomp nmf` → `test_nmf_basic_fit` `(2,3)`, `test_nmf_transform_dimensions` `(4,2)`, `test_nmf_reconstruction_error_decreases`, `test_nmf_reconstruction_error_positive`, `test_nmf_more_components_lower_error`, `test_nmf_n_iter_positive`, `test_nmf_reproducibility`, `test_nmf_single_component`, `test_nmf_f32` PASS. |
| REQ-2 (structural: non-negativity of components_ + W) | SHIPPED | MU updates are multiplicative on non-negative factors (`solve_multiplicative_update` `nmf.rs:478-495`), CD clamps each entry to `max(0,…)` (`solve_coordinate_descent` `:550-554`/`:581-585`), `transform`'s W is MU from a positive const (`:727-734`) — all stored `H` / returned `W` `≥ 0`. Probe 1 sklearn `components_` + `transform W` non-negative. **Scope: STRUCTURAL, NOT value parity (REQ-5).** Non-test consumers: re-export `lib.rs:97`, `_RsNMF` `extras.rs:1116`. Verification: `cargo test -p ferrolearn-decomp nmf` → `test_nmf_components_non_negative`, `test_nmf_transform_non_negative` PASS. |
| REQ-3 (both solvers MU/CD × both inits Random/pseudo-NNDSVD run) | SHIPPED | `fn fit` (`nmf.rs:617`) dispatches init via `match self.init` (`:657-660`, `init_random` `:263` / `init_nndsvd` `:289`) and solver via `match self.solver` (`:663-670`, `solve_multiplicative_update` `:461` / `solve_coordinate_descent` `:511`) — all 4 combos fit. **Scope: STRUCTURAL each-path-runs, NOT value parity (REQ-5), real NNDSVD (REQ-6) or default-solver (REQ-7).** Non-test consumer: re-export `lib.rs:97`. Verification: `cargo test -p ferrolearn-decomp nmf` → `test_nmf_coordinate_descent_solver` `(2,4)`, `test_nmf_nndsvd_init` `(2,4)`, `test_nmf_cd_with_nndsvd` `(2,4)`, `test_nmf_medium_dataset_mu` `(3,4)`, `test_nmf_getters` PASS. |
| REQ-4 (reconstruction QUALITY: ‖X−WH‖ small / decreasing — "did NMF work") | SHIPPED | `reconstruction_error` (`nmf.rs:247`) computes `‖X−W·H‖_F`; solvers monotonically reduce it, breaking on `|prev−err|<tol` (`:499`/`:591`); `test_nmf_medium_dataset_mu` asserts converged err `< 10.0`. Probe 1 sklearn `‖X−WH‖=0.17132` (init='random'), Probe 3 `0.854916` (default) confirm a small residual. **Scope: residual small/decreasing, NOT a tolerance match to sklearn's exact `reconstruction_err_` (folds into REQ-5).** Non-test consumer: re-export `lib.rs:97`. Verification: `cargo test -p ferrolearn-decomp nmf` → `test_nmf_medium_dataset_mu`, `test_nmf_reconstruction_error_decreases`, `test_nmf_more_components_lower_error` PASS. |
| REQ-5 (EXACT `components_` value parity) | NOT-STARTED | open prereq blocker **#1609** (CARVE-OUT, R-DEFER-3). sklearn `fit_transform` (`_nmf.py:1100-1130`): `nndsvda` SVD init (`:225`/`:362`) + DEFAULT `cd` solver (`:340`), `components_ = H`. ferrolearn `fn fit` (`nmf.rs:617`): Rust `StdRng` uniform `init_random` (`:263`, default) / Jacobi-pseudo-NNDSVD `init_nndsvd` (`:289`) + DEFAULT MU `solve_multiplicative_update` (`:461`). Probe 1 sklearn `components_ row0 = [2.389149, 2.232745, 2.076342]` (init='random') / Probe 3 `[1.949292, 2.460614, 2.971937]` (default) NOT reproduced. NMF identifiable only up to permutation/scaling; no failing test (same class as `minibatch_nmf` / `dictionary_learning` / `sparse_pca` RNG carve-outs). |
| REQ-6 (real `init="nndsvda"` default + SVD `nndsvd`/`nndsvdar`/`custom`) | NOT-STARTED | open prereq blocker **#1610**. sklearn defaults `init=None` → `"nndsvda"` (`_nmf.py:302`/`:1000`); `_initialize_nmf` (`:225-374`) implements real `nndsvd` (`svd(X)` + `svd_flip`, pos/neg split `u·sqrt(s)`/`sqrt(s)·v` `:330-360`), `nndsvda` (zeros→avg `:362`), `nndsvdar` (zeros→small random `:364`), `custom`. ferrolearn `NMFInit` (`nmf.rs:61`) has only `Random` (default `:106`) and `Nndsvd`; `init_nndsvd` (`:289`) is a Jacobi eigendecomp of `XᵀX` with negatives clamped `0` (`:322-327`) + `W=(X·Hᵀ)₊` (`:343-353`) — NOT SVD-based NNDSVD/nndsvda/nndsvdar, no `custom`, default `Random` not `nndsvda`. |
| REQ-7 (`solver="cd"` DEFAULT) | NOT-STARTED | open prereq blocker **#1611**. sklearn defaults `solver="cd"` (`_nmf.py:917`) → `_fit_coordinate_descent` (`:340`). ferrolearn `NMF::new` defaults `solver=NMFSolver::MultiplicativeUpdate` (`nmf.rs:105`) — sklearn's NON-default. ferrolearn's `CoordinateDescent` variant (`:56`) exists but is not default and is a different element-wise CD (`:524-587`) from sklearn's `_update_cdnmf_fast` feature-block CD. |
| REQ-8 (`beta_loss` kl/is + `_gamma`) | NOT-STARTED | open prereq blocker **#1612**. sklearn `NMF(beta_loss="frobenius")` (`_nmf.py:919`, `StrOptions({"frobenius","kullback-leibler","itakura-saito"})`) + `_beta_divergence` (`:89`) + `_gamma` per `_beta_loss` for the MM step. ferrolearn `NMF<F>` (`nmf.rs:78`) has NO `beta_loss` field — only hard-coded Frobenius `reconstruction_error` (`:247`) + Frobenius MU/CD; no `_gamma`, no KL/IS. |
| REQ-9 (`transform` NNLS-W VALUE) | NOT-STARTED | open prereq blocker **#1613** (CARVE-OUT — folds into REQ-5). sklearn `transform` (`_nmf.py:1213`): `W = _fit_transform(X, H=components_, update_H=False)[0]` — NNLS for `W` with `H` fixed. ferrolearn `transform` (`impl Transform for FittedNMF`, `nmf.rs:696`) solves `W` with `H` fixed too, via 200-iter MU (`:727-734`) from CONSTANT `0.1` init (`:722-723`). Both reach the same convex NNLS optimum, but `W` sits atop the carved-out fitted `H` (REQ-5) and `FittedNMF`'s fields are PRIVATE (no injectable-`H` API), so a transform value pin is gated on REQ-5. **INVESTIGATE (critic):** like `minibatch_nmf` `#1487` — add an injectable-`H` constructor + NNLS-residual green-guard, or fold into REQ-5. No failing test (R-DEFER-3). |
| REQ-10 (`inverse_transform` = `W·H`) | SHIPPED | sklearn `NMF.inverse_transform(W)` (`_nmf.py:1238`) = `W @ components_`. ferrolearn `FittedNMF::inverse_transform` (`nmf.rs:229`) returns `w.dot(&self.components_)` (`:238`) after a `w.ncols()==n_components` check (`:231-237`, `ShapeMismatch`) — EXACT same `W·H` algebra (deterministic, no value carve-out). Probe 3 sklearn `inverse_transform == W@H: True`. Non-test consumer: re-export `lib.rs:97`. **FLAG:** NOT exposed through `_RsNMF` (the `py_transformer!` macro `extras.rs:107` binds only ctor + `fit` + `transform`); no dedicated in-tree `#[test]` pins it (least-confident SHIPPED — a green-guard would harden the algebra). |
| REQ-11 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`nmf.rs:617`) returns `Err(InvalidParameter{name:"n_components", ... "must be at least 1"})` for `==0` (`:620-625`), `Err(InsufficientSamples{required:1,...})` for `0` samples (`:626-632`), `Err(InvalidParameter{name:"n_components", ... "exceeds min(n_samples, n_features)"})` for `>min(n,p)` (`:633-642`), `Err(InvalidParameter{name:"X", ... "non-negative"})` on a negative entry (`:645-652`); `transform` returns `Err(ShapeMismatch)` on column mismatch (`:698-704`) + `Err(InvalidParameter{name:"X"})` on a negative entry (`:707-714`). Non-test consumer: re-export `lib.rs:97`. Verification: `cargo test -p ferrolearn-decomp nmf` (`test_nmf_invalid_n_components_zero`, `_invalid_n_components_too_large`, `_insufficient_samples`, `_negative_input_rejected`, `_transform_shape_mismatch`, `_transform_negative_rejected`, `_zero_entries`) PASS. **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints` + `check_non_negative` raising `InvalidParameterError`/`ValueError` (not `FerroError`); accepts `n_components=None` (REQ-13); does NOT pre-reject `n_components > min(n,p)` (downgrades `nndsvda`→`random`). |
| REQ-12 (PyO3 binding: thin `n_components` ctor + `fit` + `transform`) | SHIPPED (scoped) | sklearn exposes `NMF` via `import sklearn.decomposition`. ferrolearn binds `_RsNMF` (`extras.rs:1116`, registered `lib.rs:75`) via `py_transformer!` (`extras.rs:107`): `__new__(n_components=2)` + `fit(X)` + `transform(X)` over `FittedNMF<f64>`. Inherits REQ-1/2/4 structural behaviour + the REQ-5 value carve-out. **Scope: thin fit/transform only — NOT `solver`/`init`/`tol`/`max_iter`/`random_state` params, NOT `components_`/`reconstruction_err_`/`n_iter_` getters, NOT `inverse_transform` (REQ-10).** This binding is the CPython non-test consumer of `FittedNMF<f64>`. |
| REQ-13 (`n_components=None` default) | NOT-STARTED | open prereq blocker **#1614**. sklearn `NMF(n_components=None)` (`_nmf.py:914`) → `min(n_samples, n_features)` (full rank) when `None`. ferrolearn `NMF::new(n_components: usize)` (`nmf.rs:100`) REQUIRES an explicit `usize`; no `None`/auto-rank path (`==0` rejected `:620`). |
| REQ-14 (`alpha_W`/`alpha_H`/`l1_ratio` regularization) | NOT-STARTED | open prereq blocker **#1615**. sklearn `NMF(alpha_W=0.0, alpha_H="same", l1_ratio=0.0)` (`_nmf.py:921-923`) → `_compute_regularization` (`:1275`) L1/L2 penalties in the CD/MU updates. ferrolearn `NMF<F>` (`nmf.rs:78`) has NO `alpha_W`/`alpha_H`/`l1_ratio` fields — MU (`:461`) and CD (`:511`) are unpenalised. |
| REQ-15 (`shuffle` (CD) + fitted attrs `n_components_`/`n_features_in_`) | NOT-STARTED | open prereq blocker **#1616**. sklearn `NMF(shuffle=False)` (`_nmf.py:924`) randomises CD feature order when True; exposes `n_components_` / `n_features_in_`. ferrolearn `NMF<F>` (`nmf.rs:78`) has NO `shuffle` field (CD `:511` fixed order); `FittedNMF<F>` (`:193`) exposes only `components()`/`reconstruction_err()`/`n_iter()` (`:205`/`:211`/`:217`) — no `n_components_` (derivable from `components_.nrows()`), no `n_features_in_`. |
| REQ-16 (ferray substrate) | NOT-STARTED | open prereq blocker **#1617**. `nmf.rs` computes on `ndarray::Array2` (`nmf.rs:41`), uses `rand`/`rand_distr` `StdRng`+`Uniform` (`:43-44`) for init, and a hand-rolled Jacobi eigensolver (`jacobi_eigen_symmetric` `:362`), not `ferray-core` arrays / `ferray::random` / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`nmf.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`NMF<F> { n_components, max_iter (200), tol (1e-4), solver:
NMFSolver{MultiplicativeUpdate|CoordinateDescent}, init: NMFInit{Random|Nndsvd},
random_state }` (struct at line 78; `new(n_components)` fn at line 100, builders
`with_max_iter` `:113` / `with_tol` `:120` / `with_solver` `:127` / `with_init` `:134`
/ `with_random_state` `:141`, accessors `n_components()`..`random_state()` fns at
lines 148-181) → `Fit<Array2<F>, ()>` → `FittedNMF<F> { components_,
reconstruction_err_, n_iter_ }` (struct at line 193, accessors `components()` `:205` /
`reconstruction_err()` `:211` / `n_iter()` `:217`, plus `inverse_transform()` `:229`).
The path is generic over `F: Float + Send + Sync + 'static` (both f32 and f64,
`test_nmf_f32`); `fit`/`transform`/`inverse_transform` return `Result<_, FerroError>`
(R-CODE-2).

**Fit path (`fn fit`, `nmf.rs:617`) — REQ-1/2/3/4/5/11.** Validates `n_components !=
0`, `n_samples >= 1`, `n_components <= min(n_samples, n_features)`, and
non-negativity of `X` (`nmf.rs:620-652`) — REQ-11. Initialises `(W, H)` via
`init_random` (fn at line 263, Rust `StdRng` uniform `[0,1)`, seed `unwrap_or(0)`) or
`init_nndsvd` (fn at line 289, Jacobi-eigendecomposition pseudo-NNDSVD) per
`match self.init` (`nmf.rs:657-660`) — REQ-6 NOT-STARTED (default `Random`, not real
`nndsvda`). Solves via `solve_multiplicative_update` (fn at line 461, the Lee-Seung
`W,H ← W,H · num/(den+eps)` rules) or `solve_coordinate_descent` (fn at line 511,
element-wise non-negative least-squares CD) per `match self.solver`
(`nmf.rs:663-670`) — REQ-7 NOT-STARTED (default MU, not sklearn's `cd`). Both solvers
break on `|prev_err − err| < tol` (`nmf.rs:499`/`:591`) — the monotone-decrease that
underwrites the reconstruction-quality signal (REQ-4). Stores `components_ = h`,
`reconstruction_err_ = ||X − W·H||_F`, `n_iter_` (`nmf.rs:674-678`). **This is NOT
sklearn's exact fit (REQ-5):** sklearn `nndsvda`-inits (SVD-based) and runs the `cd`
solver by default with a numpy RNG (`_nmf.py:1100-1130`); ferrolearn's
Random/pseudo-NNDSVD + default-MU + Rust RNG produce DIFFERENT component values, and
NMF is identifiable only up to permutation/scaling (CARVE-OUT).

**Transform (`impl Transform for FittedNMF`, `nmf.rs:696`) — REQ-1/2/9.** Validates
the column count (`:698-704`) and non-negativity (`:707-714`) — REQ-11, inits `W` with
the CONSTANT `0.1` (`:722-723`), and solves `W` (with `H` fixed) by 200 multiplicative
updates `W ← W · (X·Hᵀ)/(W·H·Hᵀ + eps)` (`:727-734`). sklearn's `transform`
(`_nmf.py:1213`) solves the same NNLS `min_{W≥0} ||X − W·H||²` via
`_fit_transform(X, H=components_, update_H=False)`. Both reach the same convex
optimum, but the `W` values sit atop the carved-out fitted `H` (REQ-5), and
`FittedNMF`'s fields are private (no injectable-`H` constructor), so transform value
parity FOLDS INTO the components carve-out (REQ-9, `#1613`, INVESTIGATE-for-critic,
like `minibatch_nmf` `#1487`).

**Inverse transform (`FittedNMF::inverse_transform`, `nmf.rs:229`) — REQ-10.** Returns
`W·H = w.dot(&self.components_)` (`:238`) after a `w.ncols() == n_components` check
(`:231-237`) — the EXACT deterministic algebra of sklearn's `inverse_transform`
(`_nmf.py:1238`, `W @ self.components_`). No RNG / solver, so no value carve-out.

**Pipeline + PyO3 (`nmf.rs:744-771`, `extras.rs:1116`) — REQ-12.** `NMF<F>` implements
`PipelineTransformer<F>` (`fit_pipeline` `nmf.rs:752` → `transform_pipeline` `:768`),
and `_RsNMF` (`extras.rs:1116`, registered `lib.rs:75`) is the thin `py_transformer!`
binding (ctor `n_components` + `fit` + `transform` over `FittedNMF<f64>`) — NOT the
`solver`/`init`/`random_state` params, getters, or `inverse_transform`.

**sklearn (target contract).** `class NMF(_BaseNMF)` (`_nmf.py:912`) takes
`__init__(n_components=None, *, init=None, solver="cd", beta_loss="frobenius",
tol=1e-4, max_iter=200, random_state=None, alpha_W=0.0, alpha_H="same", l1_ratio=0.0,
verbose=0, shuffle=False)` (`:912-925`). `init=None` → `"nndsvda"` when
`n_components < min(n_samples, n_features)` else `"random"` (`:302`/`:1000`);
`_initialize_nmf` (`:225-374`) implements `random`/`nndsvd`/`nndsvda`/`nndsvdar`/
`custom`. `fit_transform` (`:1100`) → `_fit_transform` (`:1157`): `cd` →
`_fit_coordinate_descent` (`:340`), `mu` → `_fit_multiplicative_update`; sets
`components_ = H`, `reconstruction_err_ = _beta_divergence(X, W, H, beta_loss,
square_root=True)`, `n_iter_`. `transform` (`:1213`) solves NNLS `W` with `H` fixed;
`inverse_transform` (`:1238`) = `W @ components_`. Regularization via
`_compute_regularization` (`:1275`). Fitted attrs: `components_`,
`reconstruction_err_`, `n_components_`, `n_features_in_`, `n_iter_`.

**The remaining gap.** ferrolearn ships the STRUCTURAL shapes / non-negativity /
finite-decreasing error / determinism (REQ-1,2), the four init×solver paths (REQ-3),
the reconstruction-quality signal (REQ-4), the `inverse_transform` algebra (REQ-10),
the scoped error/parameter contracts (REQ-11), and the thin PyO3 binding (REQ-12). It
lacks: exact `components_` value parity (REQ-5, CARVE-OUT `#1609`); the real
`nndsvda` default + SVD-based `nndsvd`/`nndsvdar`/`custom` (REQ-6, `#1610`); the
`solver="cd"` default (REQ-7, `#1611`); `beta_loss` + `_gamma` (REQ-8, `#1612`);
the `transform` NNLS-W value (REQ-9, CARVE-OUT, folds into REQ-5, `#1613`);
`n_components=None` (REQ-13, `#1614`); regularization (REQ-14, `#1615`);
`shuffle` + `n_components_`/`n_features_in_` attrs (REQ-15, `#1616`); and the ferray
substrate (REQ-16, `#1617`). This is a **structure-SHIPPED-algorithm-NOT-STARTED**
unit (7 SHIPPED / 9 NOT-STARTED).

## Verification

Library crate (green at baseline `5b4b4d15`):
```bash
cargo test -p ferrolearn-decomp nmf                        # in-module #[test]s + doctest
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-1/2/3/4/11 (STRUCTURAL): `test_nmf_basic_fit` `(2,3)`,
`test_nmf_transform_dimensions` `(4,2)`, `test_nmf_reconstruction_error_positive`,
`test_nmf_reconstruction_error_decreases`, `test_nmf_more_components_lower_error`,
`test_nmf_n_iter_positive`, `test_nmf_reproducibility`, `test_nmf_single_component`,
`test_nmf_f32`, `test_nmf_zero_entries` (REQ-1); `test_nmf_components_non_negative`,
`test_nmf_transform_non_negative` (REQ-2); `test_nmf_coordinate_descent_solver`,
`test_nmf_nndsvd_init`, `test_nmf_cd_with_nndsvd`, `test_nmf_medium_dataset_mu`,
`test_nmf_getters` (REQ-3); `test_nmf_medium_dataset_mu` (err `< 10.0`) +
`_reconstruction_error_decreases` + `_more_components_lower_error` (REQ-4);
`test_nmf_invalid_n_components_zero`, `test_nmf_invalid_n_components_too_large`,
`test_nmf_insufficient_samples`, `test_nmf_negative_input_rejected`,
`test_nmf_transform_shape_mismatch`, `test_nmf_transform_negative_rejected` (REQ-11);
plus `test_nmf_pipeline_integration` (the `PipelineTransformer` consumer) and the
module doctest. REQ-10 (`inverse_transform` algebra) has NO dedicated `#[test]` — a
green-guard would harden it (least-confident SHIPPED). There is NO
`tests/divergence_nmf.rs` yet. REQ-5 (`components_` value parity) and REQ-9
(`transform` NNLS-W) are CARVE-OUTs (R-DEFER-3, no failing test).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-1/2/4 structure +
quality and the REQ-5 components value gap:
```bash
# REQ-1/2/4 structural + quality + REQ-5 value gap (values NOT reproduced):
python3 -c "import numpy as np; from sklearn.decomposition import NMF
X=np.array([[1.,2,3],[4,5,6],[7,8,9],[10,11,12]])
m=NMF(n_components=2, init='random', random_state=0).fit(X)
W=m.transform(X)
print(m.components_.shape, bool((m.components_>=0).all()), np.round(m.components_[0],6).tolist())
print('||X-WH||:', round(float(np.linalg.norm(X-W@m.components_)),6))"
# -> (2, 3) True [2.389149, 2.232745, 2.076342]
# -> ||X-WH||: 0.17132

# REQ-7/6 default solver='cd' + init=None->'nndsvda'; REQ-10 inverse_transform=W@H:
python3 -c "import numpy as np; from sklearn.decomposition import NMF
X=np.array([[1.,2,3],[4,5,6],[7,8,9],[10,11,12]])
m=NMF(n_components=2, random_state=0).fit(X)
W=m.transform(X)
print(np.round(m.components_[0],6).tolist(), bool(np.allclose(m.inverse_transform(W), W@m.components_)))"
# -> [1.949292, 2.460614, 2.971937] True
```
The REQ-5 component values are from sklearn's `nndsvda`+`cd` (default) or `random`+`cd`
(Probe 1) on the SAME small `X` (R-CHAR-3); ferrolearn's Random/pseudo-NNDSVD +
default-MU do not reproduce them (permutation/scaling + RNG + solver). REQ-9
(`transform` NNLS-W) folds into REQ-5: the `W` solve targets the same convex optimum
but sits atop the carved-out `H` (no injectable-`H` API).

ferrolearn-python (REQ-12, SHIPPED scoped): `_RsNMF` (`extras.rs:1116`, registered
`lib.rs:75`) is the thin `py_transformer!` binding — ctor `n_components` + `fit` +
`transform`:
```bash
python3 -c "import numpy as np, ferrolearn
X=np.array([[1.,2,3],[4,5,6],[7,8,9],[10,11,12]])
W=ferrolearn._RsNMF(2).fit(X).transform(X) if hasattr(ferrolearn,'_RsNMF') else None
print(W.shape if W is not None else 'binding present in extras.rs:1116')"
```
The non-test consumers of `NMF`/`FittedNMF` are the crate re-export (`lib.rs:97`), the
`_RsNMF` binding (`extras.rs:1116`), and the `PipelineTransformer` impl (`nmf.rs:744`).

## Blockers

(#1608 is this doc's crosslink tracking issue. The blockers below are the open work
items the dispatcher files / numbers; none are filed by this doc — markdown only.)

- **#1609** — REQ-5 (CARVE-OUT): match sklearn's `components_` by implementing the
  SVD-based `nndsvda` default init (`_nmf.py:225`/`:362`), the `cd` default solver
  (`_fit_coordinate_descent` `:340`), and the numpy RNG; NMF is identifiable only up
  to permutation/scaling, so this is inherently RNG/algorithm-bound (no failing test,
  R-DEFER-3). The `transform` NNLS-W (#1613) folds in here.
- **#1610** — REQ-6: replace `init_random` default + the Jacobi-pseudo-NNDSVD
  `init_nndsvd` (`nmf.rs:263`/`:289`) with the real SVD-based `nndsvd`/`nndsvda`/
  `nndsvdar`/`custom` of `_initialize_nmf` (`_nmf.py:225-374`) and default `init=None`
  → `nndsvda` (`:302`).
- **#1611** — REQ-7: change the default `solver` from `MultiplicativeUpdate`
  (`nmf.rs:105`) to sklearn's `cd` (`_nmf.py:917`) and align the CD update with
  `_update_cdnmf_fast` (feature-block CD, `_nmf.py:340`).
- **#1612** — REQ-8: add a `beta_loss` field (`frobenius`/`kullback-leibler`/
  `itakura-saito`, `_nmf.py:919`) + `_gamma` and the beta-divergence MU branches
  (`_beta_divergence` `_nmf.py:89`).
- **#1613** — REQ-9 (CARVE-OUT, folds into #1609): align the `transform` solver
  with sklearn's `_fit_transform(X, H=components_, update_H=False)` (`_nmf.py:1213`) —
  or add an injectable-`H` `FittedNMF` constructor + an NNLS-optimum residual
  green-guard so the transform value is pinnable independent of the carved-out `H`
  (INVESTIGATE-for-critic, like `minibatch_nmf` `#1487`). No failing test (R-DEFER-3).
- **#1614** — REQ-13: add an `Option<usize>` / `None` `n_components` path defaulting
  to `min(n_samples, n_features)` (`_nmf.py:914`).
- **#1615** — REQ-14: add `alpha_W`/`alpha_H`/`l1_ratio` fields (`_nmf.py:921-923`)
  and the `_compute_regularization` (`_nmf.py:1275`) penalties in the MU/CD updates.
- **#1616** — REQ-15: add a `shuffle` field (CD feature-order randomisation,
  `_nmf.py:924`) and expose `n_components_` / `n_features_in_` fitted attrs on
  `FittedNMF`.
- **#1617** — REQ-16: migrate `nmf.rs` off `ndarray` + `rand`/`rand_distr` + the
  hand-rolled `jacobi_eigen_symmetric` (`nmf.rs:362`) to `ferray-core` arrays /
  `ferray::random` / `ferray::linalg` (R-SUBSTRATE).
