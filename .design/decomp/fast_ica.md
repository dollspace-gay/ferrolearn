# FastICA (sklearn.decomposition.FastICA)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 523eaebd
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_fastica.py  # class FastICA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (:368). ctor (:520-543): n_components=None, *, algorithm="parallel", whiten="unit-variance", fun="logcosh", fun_args=None, max_iter=200, tol=1e-4, w_init=None, whiten_solver="svd", random_state=None. _sym_decorrelation (:57-69): W <- (W W.T)^{-1/2} W via linalg.eigh. _ica_def (deflation, :72-104): per-component w, loop gwtx,g_wtx=g(w.T@X); w1=(X*gwtx).mean(axis=1)-g_wtx.mean()*w; _gs_decorrelation Gram-Schmidt (:53); normalise; converge lim=|abs((w1*w).sum())-1| (:96). _ica_par (parallel, :107-136): W=_sym_decorrelation(w_init); loop gwtx,g_wtx=g(W@X); W1=_sym_decorrelation((gwtx@X.T)/p - g_wtx[:,None]*W) (:118); lim=max(abs(abs(einsum"ij,ij->i"(W1,W))-1)) (:123). _logcosh (:141-150): g=tanh(alpha*x), g'=mean(alpha*(1-g^2)), alpha=fun_args["alpha"] default 1.0; _exp (:153-157): g=x*exp(-x^2/2), g'=mean((1-x^2)*exp); _cube (:160-161): g=x^3, g'=mean(3x^2). _fit_transform (:546-692): validate ensure_min_samples=2 (:565); alpha must be in [1,2] (:571-572); n_components=None -> min(n_samples,n_features) (:591-592); center X_mean (:601-602); whiten_solver=='eigh' (:605-619, eigh of XT.dot(X)) OR 'svd' DEFAULT (:620-621, linalg.svd); u*=np.sign(u[0]) (:624); K=(u/d).T[:n_components] (:626, n_components x n_features); X1=K@XT (:628) then X1*=sqrt(n_samples) (:631); w_init=random_state.normal(size=(k,k)) (:638-641); _ica_par/_ica_def (:659-662); n_iter_ (:665); unit-variance S_std rescale W/=S_std.T (:676-681); components_=W@K (:683), mean_=X_mean (:684), whitening_=K (:685), else components_=W (:687); mixing_=linalg.pinv(components_) (:689); _unmixing=W (:690). transform (:736-762): X-=mean_; return X @ components_.T (:762). inverse_transform (:764+): X @ mixing_.T + mean_.
ferrolearn-module: ferrolearn-decomp/src/fast_ica.rs
parity-ops: FastICA
crosslink-issue: 1571
-->

## Summary

`ferrolearn-decomp/src/fast_ica.rs` mirrors scikit-learn's `FastICA`
(`sklearn/decomposition/_fastica.py`, `class FastICA` `:368`): separate a
multivariate signal into additive independent components by maximising
non-Gaussianity (a negentropy approximation), via PCA-whitening followed by a
fixed-point FastICA iteration. The exposed surface is the unfitted `FastICA<F> {
n_components, algorithm: Algorithm{Parallel|Deflation}, fun:
NonLinearity{LogCosh|Exp|Cube}, max_iter (200), tol (1e-4), random_state }`
(`fast_ica.rs`, struct at line 92; `new(n_components)` line 114, builders
`with_algorithm`/`with_fun`/`with_max_iter`/`with_tol`/`with_random_state` lines
126-159, accessor `n_components()` line 162) and the fitted `FittedFastICA<F> {
components (k×k), mixing (n_features×k), mean (n_features), whitening (k×n_features),
n_iter, n_features }` (`fast_ica.rs`, struct at line 183, accessors
`components()`/`mixing()`/`mean()`/`n_iter()` lines 205-229), re-exported at the
crate root (`pub use fast_ica::{Algorithm, FastICA, FittedFastICA, NonLinearity}`,
`lib.rs:87`). A PyO3 binding `_RsFastICA` exists (`ferrolearn-python/src/extras.rs:1108-1113`,
via the `py_transformer!` macro `extras.rs:107-149`) but binds ONLY the
`n_components` ctor + `fit` + `transform`. There is NO `tests/divergence_fast_ica.rs`.

**EXACT `components` / source VALUE PARITY DIVERGES (R-HONEST-3, REQ-4
NOT-STARTED, CARVE-OUT `#1572`).** sklearn's `_fit_transform`
(`_fastica.py:546-692`) draws `w_init = random_state.normal(size=(k,k))`
(`:638-641`) with a numpy `RandomState`, whitens with `whiten_solver='svd'`
(DEFAULT, `:620-621`, with `u *= np.sign(u[0])` sign-fix `:624`), and recovers
sources unique only up to PERMUTATION + SIGN + SCALE. ferrolearn's `fn fit`
(`fast_ica.rs`, impl at line 452) draws `w_init` from a Rust
`Xoshiro256PlusPlus` + `StandardNormal` (`:532-543`, seed
`random_state.unwrap_or(42)`), whitens via covariance `eigh`
(`jacobi_eigen_small` `:363`, = sklearn's NON-default `whiten_solver='eigh'`
option, different sign/order convention from SVD), and is subject to the same
ICA permutation/sign/scale indeterminacy. THREE independent factors — (a) RNG
`w_init` (numpy vs Xoshiro), (b) whitening solver (eigh vs the svd DEFAULT), (c)
ICA identifiability (perm+sign+scale) — make the `components` and source VALUES
diverge element-wise (same class as the minibatch_nmf / lda RNG carve-outs); no
failing test is asserted (R-DEFER-3). The meaningful structural correctness check
is recovery-up-to-perm+sign+scale (abs-correlation ≈ 1 on a known independent-source
mixture), which is the "did ICA work" property, NOT element-wise value parity.

**`components_` ATTRIBUTE SEMANTICS DIVERGE (distinct, REQ-5 NOT-STARTED,
`#1573`).** sklearn stores `self.components_ = np.dot(W, K)`
(`_fastica.py:683`, shape `n_components × n_features` — the FULL unmixing matrix
that maps centered data directly to sources) and exposes `whitening_ = K`
(`:685`) and `_unmixing = W` (`:690`) separately. ferrolearn's `components()`
(`fast_ica.rs:208`) returns `W` (the `k×k` whitened-space unmixing matrix,
`fast_ica.rs:660`), with `whitening` (= `K`) stored as a separate field
(`:663`, NO public accessor). So ferrolearn's `components()` ATTRIBUTE means
`W` whereas sklearn's `components_` means `W @ K` — a different matrix of a
different shape. The `transform` OUTPUT still matches (see REQ-1) because
ferrolearn applies `K` then `W` explicitly; only the stored attribute differs.

As of this iteration: the STRUCTURAL run of whitening + all three nonlinearities
(LogCosh/Exp/Cube) + both algorithms (Parallel/Deflation) producing finite
sources of shape `(n_samples, n_components)`, the `transform` shape, the
mean/mixing/components shapes, the error & parameter contracts (n_components
0/>n_features, n_samples<2, transform feature mismatch), determinism given a
seed, `n_iter ≥ 1`, and the `g(0)=0` nonlinearity sanity (REQ-1,2,3) are SHIPPED
scoped; exact `components`/source value parity (REQ-4, CARVE-OUT `#1572`), the
`components_ = W@K` attribute semantics (REQ-5, `#1573`), `mixing_ =
pinv(components_)` (REQ-6, `#1574`), `whiten_solver` svd-default/eigh +
`whiten='arbitrary-variance'`/`False` + the unit-variance `S_std` rescale (REQ-7,
`#1575`), `fun` as a callable + `fun_args` alpha (REQ-8, `#1576`), the `w_init`
param (REQ-9, `#1577`), `n_components=None` auto-default (REQ-10, `#1578`),
`inverse_transform` (REQ-11, `#1579`), the fitted attrs
`whitening_`/`n_features_in_`/`n_iter_`/`mixing_` naming (REQ-12, `#1580`),
numpy-RNG `w_init` parity (REQ-13, `#1581`), the PyO3 binding surface scope
(REQ-14, `#1582`), and the ferray substrate (REQ-15, `#1583`) are NOT-STARTED —
**3 SHIPPED / 12 NOT-STARTED**.

`FastICA` / `FittedFastICA` are existing pub APIs whose non-test consumers are the
crate re-export (`lib.rs:87`, boundary public API, grandfathered S5/R-DEFER-1),
the `_RsFastICA` PyO3 binding (`extras.rs:1108-1113`), and `PipelineTransformer`
(`fast_ica.rs:713-738`).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# PROBE 1 (REQ-1 structural recovery + REQ-4/5/6 NOT-STARTED) — fitted attr shapes,
# the W@K identity, mixing_ = pinv(components_), and the meaningful ICA-correctness
# property: recovered sources match the TRUE sources up to permutation+sign+scale
# (abs-correlation ≈ 1) on a known mixture of independent non-Gaussian sources.
# VALUES generated by sklearn, never copied from ferrolearn (R-CHAR-3).
python3 -c "
import numpy as np
from sklearn.decomposition import FastICA
rng=np.random.RandomState(0); n=400; t=np.linspace(0,8,n)
s1=np.sign(np.sin(2*t)); s2=np.mod(t,2)-1            # square + sawtooth (independent, non-gaussian)
S=np.c_[s1,s2]+0.02*rng.normal(size=(n,2))
A=np.array([[1.0,0.7],[0.5,1.2]]); X=S.dot(A.T)
ica=FastICA(n_components=2, random_state=0); Sr=ica.fit_transform(X)
print('components_.shape', ica.components_.shape, '(n_components x n_features)')
print('whitening_.shape', ica.whitening_.shape, ' mixing_.shape', ica.mixing_.shape, ' mean_.shape', ica.mean_.shape, ' n_iter_', ica.n_iter_)
print('components_ == _unmixing @ whitening_ (W@K):', bool(np.allclose(ica.components_, ica._unmixing @ ica.whitening_)))
print('mixing_ == pinv(components_):', bool(np.allclose(ica.mixing_, np.linalg.pinv(ica.components_))))
C=np.abs(np.corrcoef(Sr.T,S.T)[:2,2:]); print('abs-corr recovered vs true:', np.round(np.max(C,axis=1),4).tolist())"
# -> components_.shape (2, 2) (n_components x n_features)
# -> whitening_.shape (2, 2)  mixing_.shape (2, 2)  mean_.shape (2,)  n_iter_ 4
# -> components_ == _unmixing @ whitening_ (W@K): True   => sklearn stores W@K, ferrolearn stores W (REQ-5)
# -> mixing_ == pinv(components_): True                  => sklearn mixing_=pinv(W@K), ferrolearn K.T@W.T (REQ-6)
# -> abs-corr recovered vs true: [0.9999, 0.9978]        => recovery up to perm+sign+scale (the real ICA check, REQ-1)

# PROBE 2 (REQ-5 W@K shape divergence with k < n_features) — components_ is k x n_features.
python3 -c "
import numpy as np
from sklearn.decomposition import FastICA
rng=np.random.RandomState(1); X=rng.normal(size=(100,3))**3
ica=FastICA(n_components=2, random_state=0).fit(X)
print('components_.shape', ica.components_.shape, ' whitening_.shape', ica.whitening_.shape, ' _unmixing.shape', ica._unmixing.shape, ' mixing_.shape', ica.mixing_.shape)
print('components_ == W@K:', bool(np.allclose(ica.components_, ica._unmixing @ ica.whitening_)))"
# -> components_.shape (2, 3)  whitening_.shape (2, 3)  _unmixing.shape (2, 2)  mixing_.shape (3, 2)
# -> components_ == W@K: True   => sklearn components_ is (k x n_features); ferrolearn components() is W (k x k)

# PROBE 3 (REQ-7..14 NOT-STARTED) — ctor defaults.
python3 -c "
from sklearn.decomposition import FastICA
m=FastICA()
for p in ['n_components','algorithm','whiten','fun','fun_args','max_iter','tol','w_init','whiten_solver','random_state']:
    print(f'{p} =', getattr(m,p))"
# -> n_components = None  algorithm = parallel  whiten = unit-variance  fun = logcosh
# -> fun_args = None  max_iter = 200  tol = 0.0001  w_init = None  whiten_solver = svd  random_state = None
#    => whiten_solver DEFAULT = 'svd' (ferrolearn uses covariance-eigh, the NON-default option, REQ-7);
#       n_components=None -> min(n_samples,n_features) (ferrolearn requires explicit usize, REQ-10);
#       w_init param (REQ-9); fun callable + fun_args alpha (REQ-8); whiten='unit-variance' S_std rescale (REQ-7).

# PROBE 4 (REQ-7 whiten_solver / REQ-11 inverse_transform) — svd vs eigh agree on THIS data; inverse_transform.
python3 -c "
import numpy as np
from sklearn.decomposition import FastICA
rng=np.random.RandomState(2); X=rng.normal(size=(200,2))**3
a=FastICA(n_components=2,random_state=0,whiten_solver='svd').fit(X)
b=FastICA(n_components=2,random_state=0,whiten_solver='eigh').fit(X)
print('svd vs eigh components_ allclose:', bool(np.allclose(a.components_,b.components_)))
S=a.transform(X); print('inverse_transform recovers X:', bool(np.allclose(a.inverse_transform(S),X,atol=1e-6)))"
# -> svd vs eigh components_ allclose: True   (agree here; differ in sign/order on degenerate spectra)
# -> inverse_transform recovers X: True       => ferrolearn has NO inverse_transform (REQ-11 NOT-STARTED)
```

## Requirements

- REQ-1: **Structural: whitening + all 3 nonlinearities + both algorithms run and
  recover sources of shape `(n_samples, n_components)` up to perm+sign+scale; the
  fitted attr shapes; `n_iter ≥ 1`; determinism given a seed; `g(0)=0`
  (SHIPPED scoped).** `fn fit` (`fast_ica.rs`, impl at line 452) centers
  (`:482-492`), whitens via covariance `eigh` into the whitening matrix `K`
  (`:494-529`), draws a seeded `Xoshiro256PlusPlus` `w_init` (`:532-543`),
  `sym_orthogonalise`s it (`:545`, = sklearn `_sym_decorrelation` `_fastica.py:57`),
  then runs `Algorithm::Parallel` (`:551-595`, mirroring `_ica_par`
  `_fastica.py:107` — symmetric-decorrelated `W1 = (gwtx@X)/n − mean_gp·W`,
  converge on `max|abs(diag(W1·Wᵀ))−1|`) or `Algorithm::Deflation` (`:596-649`,
  mirroring `_ica_def` `_fastica.py:72` — one component at a time + Gram-Schmidt
  `gs_orthogonalise` `:289`). The three nonlinearities `apply_nonlinearity`
  (`:238`) mirror `_logcosh`/`_exp`/`_cube` (`_fastica.py:141`/`:153`/`:160`):
  LogCosh `g=tanh(u)` (alpha=1), Exp `g=u·exp(−u²/2)`, Cube `g=u³`. `transform`
  (`:686`) centers, whitens (`x_white = K@(x−mean)ᵀ`), and unmixes (`sources =
  x_white @ Wᵀ = (x−mean) @ (W@K)ᵀ`) — matching sklearn's `(X−mean) @
  components_.T` OUTPUT (`_fastica.py:762`) despite the different stored
  `components` (REQ-5). Probe 1 confirms recovered sources match the TRUE sources
  up to perm+sign+scale (abs-corr `[0.9999, 0.9978]`). The seeded RNG +
  deterministic eigh make the fit reproducible given a seed.
  **Scope: STRUCTURAL (shape / finiteness / determinism / source-recovery
  correlation), NOT element-wise `components`/source VALUES (REQ-4).** Pinned by
  `test_ica_fit_returns_fitted`, `test_ica_transform_shape`,
  `test_ica_parallel_algorithm`, `test_ica_deflation_algorithm`,
  `test_ica_logcosh`/`_exp`/`_cube`, `test_ica_n_iter_positive`,
  `test_ica_single_component`, `test_ica_sources_not_all_zero`,
  `test_ica_reproducible_with_seed`, `test_ica_nonlinearity_values`. **FLAG for
  the critic:** the in-tree tests assert shapes / finiteness / determinism but NOT
  the meaningful recovery-up-to-perm+sign+scale property (abs-corr ≈ 1 on a known
  independent-source mixture, Probe 1) — that is the real "did ICA work" check and
  is not yet pinned by a test. Non-test consumers: re-export `lib.rs:87`,
  `_RsFastICA` (`extras.rs:1108`).

- REQ-2: **Structural: `mean`/`mixing`/`components` fitted-attr shapes (SHIPPED
  scoped).** `FittedFastICA<F>` (`fast_ica.rs`, struct at line 183) stores `mean`
  of shape `(n_features,)` (`:482-486`/`:662`), `mixing` of shape `(n_features,
  k)` (`mixing = whitening.t().dot(&w.t())` `:657`, = `Kᵀ Wᵀ`; field `:190`),
  `components` (= `W`) of shape `(k, k)` (`:660`), and `whitening` (= `K`) of
  shape `(k, n_features)` (`:663`). Probe 1 confirms sklearn's `mean_` `(2,)`,
  `whitening_` `(2,2)`. **Scope: STRUCTURAL shapes, NOT values (REQ-4) nor the
  `components_=W@K` semantics (REQ-5).** Pinned by `test_ica_mixing_shape`
  `(2,2)`, `test_ica_mean_shape` `len==2`, `test_ica_fit_returns_fitted`
  `components (2,2)`. Non-test consumers: re-export `lib.rs:87`, `_RsFastICA`
  (`extras.rs:1108`).

- REQ-3: **Error / parameter contracts (SHIPPED scoped).** `fn fit`
  (`fast_ica.rs`, impl at line 452) returns `InvalidParameter { name:
  "n_components" }` for `n_components == 0` (`:455-460`) and for `n_components >
  n_features` (`:461-469`), and `InsufficientSamples { required: 2 }` for
  `n_samples < 2` (`:470-476`); `transform` returns `ShapeMismatch` on a
  feature-count mismatch (`:687-693`). Pinned by `test_ica_error_zero_components`,
  `test_ica_error_too_many_components`, `test_ica_error_insufficient_samples`,
  `test_ica_transform_shape_mismatch`. **FLAG (candidate DIVs):** sklearn
  validates `ensure_min_samples=2` (`_fastica.py:565`) raising `ValueError`, NOT
  `FerroError`; sklearn accepts `n_components=None` (→ `min(n_samples,
  n_features)` `:591-592`, REQ-10) which ferrolearn requires as an explicit
  `usize`; sklearn does NOT pre-reject `n_components > n_features` — it clamps with
  a warning (`:593-597`); sklearn also enforces `alpha ∈ [1,2]` (`:571-572`) which
  ferrolearn has no `alpha` param for (REQ-8). Non-test consumers: re-export
  `lib.rs:87`, `_RsFastICA` (`extras.rs:1108`).

- REQ-4: **EXACT `components` / source value parity with sklearn (NOT-STARTED,
  CARVE-OUT; `#1572`).** sklearn's `_fit_transform` (`_fastica.py:546-692`) draws
  `w_init = random_state.normal(size=(k,k))` (`:638-641`, numpy `RandomState`),
  whitens with `whiten_solver='svd'` DEFAULT (`:620-621`, `u *= np.sign(u[0])`
  sign-fix `:624`), and recovers sources unique only up to PERMUTATION + SIGN +
  SCALE. ferrolearn's `fn fit` (`fast_ica.rs`, impl at line 452) draws `w_init`
  from a Rust `Xoshiro256PlusPlus` + `StandardNormal` (`:532-543`, seed
  `unwrap_or(42)`), whitens via covariance `eigh` (`jacobi_eigen_small` `:363` =
  sklearn's NON-default `whiten_solver='eigh'`, different sign/order convention),
  subject to the same indeterminacy. **CARVE-OUT (R-DEFER-3):** matching sklearn
  element-wise requires the numpy RNG `w_init` + the SVD whitening default + a
  fixed perm/sign/scale alignment; no failing test is asserted (same class as the
  minibatch_nmf / lda RNG carve-outs). The transform sources are downstream of the
  same factors → also part of this carve-out.

- REQ-5: **`components_ = W @ K` attribute semantics (NOT-STARTED; `#1573`).**
  sklearn stores `self.components_ = np.dot(W, K)` (`_fastica.py:683`, shape
  `n_components × n_features` — the full unmixing operator on centered data) and
  `self.whitening_ = K` (`:685`), `self._unmixing = W` (`:690`) separately; Probe
  1/2 confirm `components_ == _unmixing @ whitening_` and `components_.shape =
  (k, n_features)`. ferrolearn's `components()` (`fast_ica.rs:208`) returns `W`
  (the `k × k` whitened-space unmixing, `:660`) and stores `whitening` (= `K`) as
  a separate field (`:663`) with NO public accessor. So `components()` means a
  DIFFERENT matrix of a different shape from sklearn's `components_` — although the
  `transform` output still matches (REQ-1), the stored attribute is not the
  sklearn `W@K`.

- REQ-6: **`mixing_ = pinv(components_)` (NOT-STARTED; `#1574`).** sklearn sets
  `self.mixing_ = linalg.pinv(self.components_)` (`_fastica.py:689`, the
  Moore-Penrose pseudo-inverse of `W@K`); Probe 1 confirms `mixing_ ==
  pinv(components_)`. ferrolearn computes `mixing = whitening.t().dot(&w.t())` =
  `Kᵀ Wᵀ` (`fast_ica.rs:657`), which is the TRANSPOSE-product reconstruction (an
  approximate inverse exploiting `K`'s near-orthonormal rows), NOT a numerical
  pseudo-inverse of `components_`. The two coincide only when `K` has orthonormal
  rows and `W` is orthogonal; in general `Kᵀ Wᵀ ≠ pinv(W@K)`.

- REQ-7: **`whiten_solver` svd (DEFAULT) / eigh + `whiten='arbitrary-variance'` /
  `False` + the unit-variance `S_std` rescale (NOT-STARTED; `#1575`).** sklearn
  defaults `whiten_solver='svd'` (`_fastica.py:531`) → `u, d = linalg.svd(XT)`
  (`:620-621`) with the `u *= np.sign(u[0])` sign convention (`:624`), and
  `whiten='unit-variance'` (`:525`) rescales each source to unit variance via
  `S_std` and `W /= S_std.T` (`:676-681`); `whiten='arbitrary-variance'` skips the
  rescale, and `whiten=False` skips whitening entirely (`:632-635`, `components_ =
  W` `:687`). ferrolearn ALWAYS whitens via covariance `eigh`
  (`jacobi_eigen_small` `:363`, `K = eigvec/sqrt(eigval)` `:512-525`) — i.e.
  sklearn's NON-default `whiten_solver='eigh'` path (`:605-619`) — has NO
  `whiten_solver` / `whiten` field, NO SVD path, NO `u*=sign(u[0])`, and NO
  unit-variance `S_std` rescale (`fast_ica.rs:494-529` builds `K` directly with no
  post-hoc source-variance normalisation; `X1 *= sqrt(n_samples)` `_fastica.py:631`
  is also absent — ferrolearn whitens to unit variance via `1/sqrt(eigval)` without
  the `sqrt(n)` scaling).

- REQ-8: **`fun` as a callable + `fun_args` (`alpha`) (NOT-STARTED; `#1576`).**
  sklearn's `fun` accepts a user callable returning `(g, g')`
  (`_fastica.py:580-583`), and `fun_args={'alpha': a}` (`:567`,`:570`) parametrises
  `_logcosh` `g=tanh(alpha·x)`, `g'=mean(alpha·(1−g²))` (`:141-150`) with `alpha ∈
  [1,2]` enforced (`:571-572`). ferrolearn's `NonLinearity` enum
  (`fast_ica.rs:70`) is closed to `{LogCosh, Exp, Cube}` with `alpha` HARD-CODED to
  1 (`apply_nonlinearity` LogCosh `g=tanh(u)` `:246-259`) — no callable variant, no
  `fun_args`, no `alpha` field.

- REQ-9: **`w_init` parameter (NOT-STARTED; `#1577`).** sklearn accepts an explicit
  `w_init` ndarray of shape `(n_components, n_components)`
  (`_fastica.py:530`,`:637-649`) so the caller can fix the initial unmixing
  estimate. ferrolearn's `FastICA<F>` (`fast_ica.rs`, struct at line 92) has NO
  `w_init` field — `w_init` is ALWAYS drawn internally from the seeded
  `Xoshiro256PlusPlus` (`:537-543`); the only knob is `with_random_state`.

- REQ-10: **`n_components=None` auto-default (NOT-STARTED; `#1578`).** sklearn
  accepts `n_components=None` (default, `_fastica.py:522`) → `min(n_samples,
  n_features)` (`:591-592`), and clamps an over-large value with a warning
  (`:593-597`). ferrolearn requires `n_components` as an explicit `usize`
  (`FastICA::new(n_components)` `fast_ica.rs:114`) and hard-rejects `0` and
  `> n_features` (`:455-469`, REQ-3) rather than defaulting/clamping.

- REQ-11: **`inverse_transform` (NOT-STARTED; `#1579`).** sklearn's
  `inverse_transform` (`_fastica.py:764+`) maps sources back to mixed data via
  `X @ mixing_.T + mean_`; Probe 4 confirms it recovers `X`. `FittedFastICA<F>`
  (`fast_ica.rs`, struct at line 183) implements only `Transform`
  (forward unmixing, `:674-707`) — there is NO `inverse_transform` method despite
  `mixing` being stored.

- REQ-12: **Fitted attrs `whitening_` / `n_features_in_` / `n_iter_` + `mixing_`
  naming (NOT-STARTED; `#1580`).** sklearn exposes `components_`, `mixing_`,
  `mean_`, `whitening_`, `n_features_in_`, `n_iter_` as public attributes.
  `FittedFastICA<F>` (`fast_ica.rs`, struct at line 183) exposes accessors
  `components()`/`mixing()`/`mean()`/`n_iter()` (lines 205-229) but `whitening`
  (`:196`) and `n_features` (`:202`) have NO public accessor; there is no
  `n_features_in_`-named accessor, and `components()` returns `W` not `W@K` (REQ-5).

- REQ-13: **numpy-RandomState `w_init` parity (NOT-STARTED; `#1581`).** sklearn
  draws `w_init = check_random_state(self.random_state).normal(size=(k,k))`
  (`_fastica.py:568`,`:638-641`) — the numpy Mersenne-Twister `RandomState`.
  ferrolearn uses a Rust `Xoshiro256PlusPlus` seeded by `random_state.unwrap_or(42)`
  with `rand_distr::StandardNormal` (`fast_ica.rs:533-543`) — a different PRNG and
  a different Gaussian sampler, so even with `whiten_solver='eigh'` + a fixed
  perm/sign/scale the `w_init` draws (and thus the iteration path) diverge.

- REQ-14: **PyO3 binding surface scope (NOT-STARTED; `#1582`).** sklearn's
  `FastICA` exposes the full ctor (`algorithm`/`whiten`/`fun`/`fun_args`/`max_iter`/
  `tol`/`w_init`/`whiten_solver`/`random_state`) + `fit`/`fit_transform`/
  `transform`/`inverse_transform` + the fitted attrs. ferrolearn's `_RsFastICA`
  (`extras.rs:1108-1113`, via `py_transformer!` `extras.rs:107-149`) binds ONLY
  `n_components` (ctor) + `fit` + `transform` — no `algorithm`/`fun`/`max_iter`/
  `tol`/`random_state` ctor args, no `inverse_transform`, and no fitted-attr getters
  (`components_`/`mixing_`/`mean_`/`whitening_`/`n_iter_`).

- REQ-15: **ferray substrate (NOT-STARTED; `#1583`).** `fast_ica.rs` computes on
  `ndarray::{Array1, Array2}` (`fast_ica.rs:50`) and uses `rand` +
  `rand_distr::StandardNormal` + `rand_xoshiro::Xoshiro256PlusPlus`
  (`fast_ica.rs:52-53`,`:533`) for `w_init`, not `ferray-core` arrays /
  `ferray::random` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED scoped): `FastICA::new(2).with_random_state(0).fit(&X)
  .unwrap().transform(&X)` has shape `(n_samples, 2)` and finite entries; both
  `Algorithm::{Parallel, Deflation}` and all three `NonLinearity::{LogCosh, Exp,
  Cube}` run; `n_iter() ≥ 1`; two seeded fits are identical; `g(0)=0`. On a known
  mixture of independent non-Gaussian sources the recovered sources match the true
  sources up to perm+sign+scale (Probe 1: abs-corr `[0.9999, 0.9978]`). Pinned by
  `test_ica_transform_shape` `(50,2)`, `test_ica_parallel_algorithm`,
  `test_ica_deflation_algorithm`, `test_ica_logcosh`/`_exp`/`_cube`,
  `test_ica_n_iter_positive`, `test_ica_single_component` `(50,1)`,
  `test_ica_reproducible_with_seed`, `test_ica_nonlinearity_values`. (Structural /
  recovery-correlation only — NOT element-wise values, REQ-4. FLAG: the abs-corr
  recovery property is not yet pinned by an in-tree test.)

- AC-2 (REQ-2, SHIPPED scoped): `fitted.mixing().dim()` is `(n_features, 2)`,
  `fitted.mean().len()` is `n_features`, `fitted.components().dim()` is `(2,2)`.
  Probe 1 confirms sklearn `mean_` `(2,)`, `whitening_` `(2,2)`. Pinned by
  `test_ica_mixing_shape` `(2,2)`, `test_ica_mean_shape`, `test_ica_fit_returns_fitted`.
  (Structural shapes only — NOT values, REQ-4, nor `components_=W@K` semantics, REQ-5.)

- AC-3 (REQ-3, SHIPPED scoped): `fit` returns `Err` for `n_components=0`,
  `n_components > n_features`, and `n_samples < 2`; `transform` returns `Err` for a
  feature-count mismatch. Pinned by `test_ica_error_zero_components`,
  `test_ica_error_too_many_components`, `test_ica_error_insufficient_samples`,
  `test_ica_transform_shape_mismatch`. FLAG: sklearn raises `ValueError` (not
  `FerroError`), accepts `n_components=None`, clamps `n_components > n_features`
  with a warning (does not reject), and enforces `alpha ∈ [1,2]`.

- AC-4 (REQ-4, NOT-STARTED, CARVE-OUT): `FastICA(n_components=2,
  random_state=0).fit(X).components_` / `fit_transform(X)` is NOT reproduced
  element-wise by ferrolearn (numpy RNG `w_init` + SVD whitening default + ICA
  perm/sign/scale indeterminacy). No failing test asserts this (R-DEFER-3); the
  observable, meaningful check is abs-corr recovery (AC-1).

- AC-5 (REQ-5, NOT-STARTED): sklearn `components_ = W @ K` (Probe 1/2:
  `components_ == _unmixing @ whitening_`, shape `(k, n_features)`); ferrolearn
  `components()` returns `W` (`k×k`) with `whitening` (`K`) stored separately —
  different matrix, different shape (transform OUTPUT still matches, REQ-1).

- AC-6 (REQ-6, NOT-STARTED): sklearn `mixing_ = pinv(components_)` (Probe 1:
  `mixing_ == pinv(components_)`); ferrolearn `mixing = Kᵀ Wᵀ`
  (`fast_ica.rs:657`), a transpose-product reconstruction, not a numerical
  pseudo-inverse of `components_`.

- AC-7 (REQ-7..14, DIVERGES): `FastICA()` defaults `n_components=None,
  algorithm='parallel', whiten='unit-variance', fun='logcosh', fun_args=None,
  max_iter=200, tol=1e-4, w_init=None, whiten_solver='svd', random_state=None`
  (Probe 3, `_fastica.py:520-543`); sklearn exposes `inverse_transform`, the
  `whitening_`/`n_features_in_`/`n_iter_` attrs, a `fun` callable + `fun_args`
  alpha, `w_init`, and the svd/eigh + `whiten` strategies. ferrolearn ALWAYS
  covariance-eigh-whitens (no `whiten_solver`/`whiten`), has a closed
  `NonLinearity` enum (alpha=1, no callable/`fun_args`), no `w_init` param, no
  `n_components=None`, no `inverse_transform`, and the `_RsFastICA` binding exposes
  only `n_components`/`fit`/`transform`.

- AC-8 (REQ-15): the module imports `ndarray` (`fast_ica.rs:50`) +
  `rand`/`rand_distr`/`rand_xoshiro` (`:52-53`), not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `FastICA` / `FittedFastICA` are existing pub APIs; the
non-test consumers are the crate re-export (`lib.rs:87`, boundary public API,
grandfathered S5/R-DEFER-1), the `_RsFastICA` PyO3 binding
(`extras.rs:1108-1113`), and `PipelineTransformer` (`fast_ica.rs:713-738`). Cites
use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle =
installed sklearn 1.5.2, run from `/tmp`.
**EXACT `components`/source VALUE PARITY DIVERGES (R-HONEST-3, REQ-4 NOT-STARTED,
CARVE-OUT `#1572`):** Rust `Xoshiro256PlusPlus` + `StandardNormal` `w_init`
(`fast_ica.rs:532-543`) + covariance-eigh whitening (`:494-529`, sklearn's
NON-default `whiten_solver='eigh'`) + ICA perm/sign/scale indeterminacy ≠
sklearn's numpy `RandomState` `w_init` (`_fastica.py:638-641`) + SVD whitening
DEFAULT (`:620-621`).
**`components_ = W@K` ATTRIBUTE SEMANTICS DIVERGE (distinct, REQ-5 NOT-STARTED,
`#1573`):** sklearn `components_ = np.dot(W, K)` shape `(k, n_features)`
(`_fastica.py:683`); ferrolearn `components()` returns `W` shape `(k, k)`
(`fast_ica.rs:660`,`:208`) with `whitening` (`K`) stored separately — the
transform OUTPUT matches but the stored attribute does not.
The least-confident SHIPPED claim is REQ-1 — it is STRUCTURAL (shape / finiteness
/ determinism), and although Probe 1 confirms the meaningful recovery-up-to-perm+
sign+scale property (abs-corr ≈ 1), NO in-tree test pins that abs-corr property
(the tests assert shapes / finiteness / determinism / `g(0)=0`), which is flagged
for the critic. #1571 is this doc's crosslink tracking issue. Count: **3 SHIPPED
(REQ-1,2,3) / 12 NOT-STARTED (REQ-4,5,6,7,8,9,10,11,12,13,14,15)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (structural: whitening + 3 nonlinearities + 2 algorithms recover sources, shape, n_iter, determinism, g(0)=0) | SHIPPED | `fn fit` (`fast_ica.rs` impl at line 452) centers (`:482-492`), whitens via covariance `eigh` → `K` (`:494-529`), seeds `Xoshiro256PlusPlus` `w_init` (`:532-543`), `sym_orthogonalise` (`:545`, = sklearn `_sym_decorrelation` `_fastica.py:57`), runs `Algorithm::Parallel` (`:551-595`, = `_ica_par` `_fastica.py:107`: `W1=(gwtx@X)/n−mean_gp·W`, converge `max|abs(diag(W1·Wᵀ))−1|`) or `Deflation` (`:596-649`, = `_ica_def` `_fastica.py:72` + Gram-Schmidt `gs_orthogonalise` `:289`); `apply_nonlinearity` (`:238`) = `_logcosh`/`_exp`/`_cube` (`_fastica.py:141`/`:153`/`:160`). `transform` (`:686`) outputs `(x−mean)@(W@K)ᵀ`, matching sklearn `(X−mean)@components_.T` (`_fastica.py:762`). Probe 1 abs-corr recovered vs true `[0.9999, 0.9978]` (perm+sign+scale). **Scope: STRUCTURAL, NOT values (REQ-4).** Non-test consumers: re-export `lib.rs:87`, `_RsFastICA` (`extras.rs:1108`). Verification: `cargo test -p ferrolearn-decomp fast_ica` → `test_ica_transform_shape` `(50,2)`, `_parallel_algorithm`, `_deflation_algorithm`, `_logcosh`/`_exp`/`_cube`, `_n_iter_positive`, `_single_component`, `_reproducible_with_seed`, `_nonlinearity_values` PASS. **FLAG (critic):** abs-corr recovery (the real ICA-correctness property) is NOT pinned by an in-tree test. |
| REQ-2 (structural: mean/mixing/components fitted-attr shapes) | SHIPPED | `FittedFastICA<F>` (`fast_ica.rs` struct at line 183) stores `mean` `(n_features,)` (`:662`), `mixing = Kᵀ Wᵀ` `(n_features, k)` (`:657`), `components = W` `(k, k)` (`:660`), `whitening = K` `(k, n_features)` (`:663`). Probe 1 sklearn `mean_` `(2,)`, `whitening_` `(2,2)`. **Scope: STRUCTURAL shapes, NOT values (REQ-4) nor `components_=W@K` semantics (REQ-5).** Non-test consumers: re-export `lib.rs:87`, `_RsFastICA` (`extras.rs:1108`). Verification: `cargo test -p ferrolearn-decomp fast_ica` → `test_ica_mixing_shape` `(2,2)`, `test_ica_mean_shape`, `test_ica_fit_returns_fitted` PASS. |
| REQ-3 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`fast_ica.rs` impl at line 452) returns `Err(InvalidParameter{name:"n_components", reason:"must be at least 1"})` for `n_components==0` (`:455-460`), `Err(InvalidParameter{name:"n_components", ..."exceeds n_features"})` for `>n_features` (`:461-469`), `Err(InsufficientSamples{required:2,...})` for `n_samples<2` (`:470-476`); `transform` returns `Err(ShapeMismatch)` on feature mismatch (`:687-693`). Non-test consumers: re-export `lib.rs:87`, `_RsFastICA` (`extras.rs:1108`). Verification: `cargo test -p ferrolearn-decomp fast_ica` (`test_ica_error_zero_components`, `_too_many_components`, `_insufficient_samples`, `_transform_shape_mismatch`) PASS. **FLAG (candidate DIVs):** sklearn raises `ValueError` (`_fastica.py:565`) not `FerroError`; accepts `n_components=None` (`:591-592`); CLAMPS `n_components>n_features` with a warning (`:593-597`), not rejects; enforces `alpha∈[1,2]` (`:571-572`). |
| REQ-4 (EXACT `components`/source value parity) | NOT-STARTED | open prereq blocker **#1572** (CARVE-OUT, R-DEFER-3). sklearn `_fit_transform` (`_fastica.py:546-692`): `w_init=random_state.normal(size=(k,k))` (`:638-641`, numpy `RandomState`), `whiten_solver='svd'` DEFAULT (`:620-621`, `u*=sign(u[0])` `:624`), sources unique up to perm+sign+scale. ferrolearn `fn fit` (`fast_ica.rs` impl at line 452): Rust `Xoshiro256PlusPlus`+`StandardNormal` `w_init` (`:532-543`, seed 42), covariance-`eigh` whitening (`:494-529`, = sklearn NON-default `whiten_solver='eigh'`). THREE factors (RNG `w_init` / eigh-vs-svd / ICA perm+sign+scale) diverge the values element-wise. No failing test (same class as minibatch_nmf / lda RNG carve-outs); the observable check is abs-corr recovery (Probe 1, REQ-1). |
| REQ-5 (`components_ = W@K` attribute semantics) | NOT-STARTED | open prereq blocker **#1573**. sklearn `self.components_ = np.dot(W, K)` (`_fastica.py:683`, shape `(k, n_features)`), `whitening_ = K` (`:685`), `_unmixing = W` (`:690`) separate; Probe 1/2 `components_ == _unmixing@whitening_`, shape `(2,3)` for k<n_features. ferrolearn `components()` (`fast_ica.rs:208`) returns `W` (`k×k`, `:660`) with `whitening` (`K`) a separate field (`:663`, no accessor) — different matrix, different shape. Transform OUTPUT still matches (REQ-1) because `K` then `W` are applied explicitly; only the STORED attribute differs. |
| REQ-6 (`mixing_ = pinv(components_)`) | NOT-STARTED | open prereq blocker **#1574**. sklearn `self.mixing_ = linalg.pinv(self.components_)` (`_fastica.py:689`, Moore-Penrose pinv of `W@K`); Probe 1 `mixing_ == pinv(components_)`. ferrolearn `mixing = whitening.t().dot(&w.t())` = `Kᵀ Wᵀ` (`fast_ica.rs:657`), a transpose-product reconstruction (exact inverse only when `K` rows orthonormal + `W` orthogonal), NOT a numerical pseudo-inverse of `components_`. |
| REQ-7 (`whiten_solver` svd-default/eigh + `whiten` strategies + unit-variance `S_std` rescale) | NOT-STARTED | open prereq blocker **#1575**. sklearn `whiten_solver='svd'` DEFAULT (`_fastica.py:531`,`:620-621`, `u*=sign(u[0])` `:624`), `whiten='unit-variance'` rescales `S/=S_std`, `W/=S_std.T` (`:676-681`), `X1*=sqrt(n_samples)` (`:631`); `'arbitrary-variance'`/`False` variants (`:632-635`,`:687`). ferrolearn ALWAYS covariance-`eigh`-whitens (`jacobi_eigen_small` `:363`, `K=eigvec/sqrt(eigval)` `:512-525` = sklearn NON-default `'eigh'` `:605-619`) — NO `whiten_solver`/`whiten` field, NO SVD, NO `u*=sign(u[0])`, NO `S_std` rescale, NO `sqrt(n)` scaling. |
| REQ-8 (`fun` callable + `fun_args` alpha) | NOT-STARTED | open prereq blocker **#1576**. sklearn `fun` accepts a user callable (`_fastica.py:580-583`); `fun_args={'alpha':a}` parametrises `_logcosh` `g=tanh(alpha·x)` (`:141-150`), `alpha∈[1,2]` enforced (`:571-572`). ferrolearn `NonLinearity` enum (`fast_ica.rs:70`) is closed to `{LogCosh,Exp,Cube}` with `alpha` HARD-CODED 1 (`apply_nonlinearity` `:246-259`) — no callable, no `fun_args`, no `alpha`. |
| REQ-9 (`w_init` param) | NOT-STARTED | open prereq blocker **#1577**. sklearn accepts an explicit `w_init` `(k,k)` ndarray (`_fastica.py:530`,`:637-649`). ferrolearn `FastICA<F>` (`fast_ica.rs` struct at line 92) has NO `w_init` field — always drawn from the seeded `Xoshiro256PlusPlus` (`:537-543`); only knob is `with_random_state`. |
| REQ-10 (`n_components=None` auto-default) | NOT-STARTED | open prereq blocker **#1578**. sklearn `n_components=None` (`_fastica.py:522`) → `min(n_samples,n_features)` (`:591-592`), clamps over-large with a warning (`:593-597`). ferrolearn requires explicit `usize` (`FastICA::new` `fast_ica.rs:114`), hard-rejects `0`/`>n_features` (`:455-469`, REQ-3). |
| REQ-11 (`inverse_transform`) | NOT-STARTED | open prereq blocker **#1579**. sklearn `inverse_transform` (`_fastica.py:764+`): `X @ mixing_.T + mean_`; Probe 4 recovers `X`. `FittedFastICA<F>` (`fast_ica.rs` struct at line 183) implements only `Transform` (forward, `:674-707`) — no `inverse_transform` despite `mixing` being stored. |
| REQ-12 (fitted attrs `whitening_`/`n_features_in_`/`n_iter_` + `mixing_` naming) | NOT-STARTED | open prereq blocker **#1580**. sklearn exposes `components_`/`mixing_`/`mean_`/`whitening_`/`n_features_in_`/`n_iter_`. `FittedFastICA<F>` (`fast_ica.rs` struct at line 183) exposes `components()`/`mixing()`/`mean()`/`n_iter()` (`:205-229`) but `whitening` (`:196`) + `n_features` (`:202`) have NO accessor, no `n_features_in_`, and `components()` returns `W` not `W@K` (REQ-5). |
| REQ-13 (numpy-RandomState `w_init` parity) | NOT-STARTED | open prereq blocker **#1581**. sklearn `w_init = check_random_state(random_state).normal(size=(k,k))` (`_fastica.py:568`,`:638-641`, numpy Mersenne-Twister). ferrolearn Rust `Xoshiro256PlusPlus` + `rand_distr::StandardNormal` seeded `unwrap_or(42)` (`fast_ica.rs:533-543`) — different PRNG + different Gaussian sampler ⇒ the `w_init` draws and iteration path diverge even under `whiten_solver='eigh'` + fixed perm/sign/scale. |
| REQ-14 (PyO3 binding surface scope) | NOT-STARTED | open prereq blocker **#1582**. sklearn `FastICA` exposes the full ctor + `fit`/`fit_transform`/`transform`/`inverse_transform` + fitted attrs. ferrolearn `_RsFastICA` (`extras.rs:1108-1113`, via `py_transformer!` `extras.rs:107-149`) binds ONLY `n_components` ctor + `fit` + `transform` — no `algorithm`/`fun`/`max_iter`/`tol`/`random_state` args, no `inverse_transform`, no fitted-attr getters. |
| REQ-15 (ferray substrate) | NOT-STARTED | open prereq blocker **#1583**. `fast_ica.rs` computes on `ndarray::{Array1,Array2}` (`fast_ica.rs:50`) and uses `rand`/`rand_distr::StandardNormal`/`rand_xoshiro::Xoshiro256PlusPlus` (`:52-53`,`:533`) for `w_init`, not `ferray-core` arrays / `ferray::random` (R-SUBSTRATE-1/2). |

## Architecture

`fast_ica.rs` follows the unfitted/fitted split (CLAUDE.md naming): `FastICA<F> {
n_components, algorithm: Algorithm{Parallel|Deflation}, fun:
NonLinearity{LogCosh|Exp|Cube}, max_iter (200), tol (1e-4), random_state }`
(struct at line 92; `new(n_components)` line 114, builders
`with_algorithm`/`with_fun`/`with_max_iter`/`with_tol`/`with_random_state` lines
126-159, accessor `n_components()` line 162) → `Fit<Array2<F>, ()>` →
`FittedFastICA<F> { components (k×k = W), mixing (n_features×k), mean
(n_features), whitening (k×n_features = K), n_iter, n_features }` (struct at line
183, accessors `components()`/`mixing()`/`mean()`/`n_iter()` lines 205-229). The
path is generic over `F: Float + Send + Sync + 'static` (both f32 and f64);
`fit`/`transform` return `Result<_, FerroError>` (R-CODE-2). `Default` is
`new(1)` (`:168-172`).

**Fit path (`fn fit`, impl at line 452) — REQ-1/2/3/4/5/6/7.** Validates
`n_components != 0`, `<= n_features`, `n_samples >= 2`
(`fast_ica.rs:455-476`) — REQ-3. (1) **Centre** (`:482-492`): subtract the
per-feature mean (`mean_ = X_mean`, sklearn `_fastica.py:601-602`). (2) **Whiten**
(`:494-529`): covariance `C = Xcᵀ Xc / n` (`:496`), eigendecompose via
`jacobi_eigen_small` (`:363`, `:500`), sort eigenvalues descending (`:503-508`),
build `K[i,:] = eigvec[:,idx]/sqrt(eigval)` of shape `(k, n_features)`
(`:512-525`) — this is sklearn's NON-default `whiten_solver='eigh'` path
(`_fastica.py:605-619`), NOT the SVD DEFAULT (`:620-621`) and WITHOUT the
`u*=sign(u[0])` sign-fix (`:624`), the unit-variance `S_std` rescale (`:676-681`),
or the `X1*=sqrt(n)` scaling (`:631`) — REQ-7 NOT-STARTED. (3) **w_init** (`:532-543`):
a `k×k` Gaussian from a seeded `Xoshiro256PlusPlus` + `StandardNormal` (seed
`unwrap_or(42)`) — NOT numpy `RandomState.normal` (REQ-13), no `w_init` param
(REQ-9). `sym_orthogonalise` (`:545`, = `_sym_decorrelation` `_fastica.py:57`).
(4) **FastICA iteration**: `Algorithm::Parallel` (`:551-595`) mirrors `_ica_par`
(`_fastica.py:107`) — `W1 = (gwtx@Xw)/n − mean_gp·W`, then symmetric
orthogonalisation, converge on `max|abs(diag(W1·Wᵀ))−1|` (`:579-588`); or
`Algorithm::Deflation` (`:596-649`) mirrors `_ica_def` (`_fastica.py:72`) — one
component at a time with Gram-Schmidt (`gs_orthogonalise` `:289`). The
nonlinearity `apply_nonlinearity` (`:238`) mirrors `_logcosh`/`_exp`/`_cube`
(LogCosh `g=tanh(u)` alpha=1, Exp `g=u·exp(−u²/2)`, Cube `g=u³`). Stores
`components = w` (= `W`, k×k), `mixing = Kᵀ Wᵀ` (`:657`), `mean`, `whitening = K`,
`n_iter`, `n_features` (`:659-666`). **This DIFFERS from sklearn's attribute
storage:** sklearn stores `components_ = W@K` (`_fastica.py:683`, REQ-5),
`mixing_ = pinv(components_)` (`:689`, REQ-6); ferrolearn keeps `W` and `K`
separate and reconstructs `mixing` as `Kᵀ Wᵀ`.

**Transform (`impl Transform for FittedFastICA`, fn at line 686) — REQ-1.**
Validates the feature count (`:687-693` — REQ-3), centers (`:695-700`), whitens
`x_white = K @ (x−mean)ᵀ` (`:702`), then unmixes `sources = x_white @ Wᵀ = (x−mean)
@ (W@K)ᵀ` (`:704`) — which equals sklearn's `(X−mean) @ components_.T`
(`_fastica.py:762`) since sklearn's `components_` IS `W@K`. So the transform
OUTPUT matches sklearn structurally (up to the REQ-4 value carve-out) DESPITE the
different `components` storage (REQ-5). There is NO `inverse_transform` (REQ-11).

**sklearn (target contract).** `class FastICA` (`_fastica.py:368`) takes
`__init__(n_components=None, *, algorithm="parallel", whiten="unit-variance",
fun="logcosh", fun_args=None, max_iter=200, tol=1e-4, w_init=None,
whiten_solver="svd", random_state=None)` (`:520-543`). `_fit_transform`
(`:546-692`) centers, whitens (svd DEFAULT or eigh, `:605-628`), draws `w_init`
(`:638-641`), runs `_ica_par` (`:107`) / `_ica_def` (`:72`), records `n_iter_`
(`:665`), applies the unit-variance `S_std` rescale (`:676-681`), and stores
`components_ = W@K` (`:683`), `mean_` (`:684`), `whitening_ = K` (`:685`),
`mixing_ = pinv(components_)` (`:689`), `_unmixing = W` (`:690`). `transform`
(`:736-762`) returns `(X−mean_) @ components_.T`; `inverse_transform` (`:764+`) is
`X @ mixing_.T + mean_`.

**The remaining gap.** ferrolearn ships the STRUCTURAL whitening + 3 nonlinearities
+ 2 algorithms recovering sources up to perm+sign+scale (REQ-1), the fitted-attr
shapes (REQ-2), and the scoped error & parameter contracts (REQ-3). It lacks:
exact `components`/source value parity (REQ-4, CARVE-OUT `#1572`); the
`components_=W@K` attribute semantics (REQ-5, `#1573`); `mixing_=pinv(components_)`
(REQ-6, `#1574`); `whiten_solver` svd-default/eigh + `whiten` strategies + the
unit-variance `S_std` rescale (REQ-7, `#1575`); the `fun` callable + `fun_args`
alpha (REQ-8, `#1576`); the `w_init` param (REQ-9, `#1577`); `n_components=None`
(REQ-10, `#1578`); `inverse_transform` (REQ-11, `#1579`); the
`whitening_`/`n_features_in_`/`n_iter_` attr surface (REQ-12, `#1580`); numpy-RNG
`w_init` parity (REQ-13, `#1581`); the full PyO3 binding surface (REQ-14, `#1582`);
and the ferray substrate (REQ-15, `#1583`). This is a
**structure-SHIPPED-attribute/algorithm-NOT-STARTED** unit (3 SHIPPED / 12
NOT-STARTED).

## Verification

Library crate (green at baseline `523eaebd`):
```bash
cargo test -p ferrolearn-decomp fast_ica                    # in-module #[test]s + doctest
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-1/2/3 (STRUCTURAL): `test_ica_fit_returns_fitted`
`(2,2)`, `test_ica_transform_shape` `(50,2)`, `test_ica_parallel_algorithm`,
`test_ica_deflation_algorithm`, `test_ica_logcosh`/`_exp`/`_cube`,
`test_ica_n_iter_positive`, `test_ica_single_component` `(50,1)`,
`test_ica_sources_not_all_zero`, `test_ica_reproducible_with_seed`,
`test_ica_nonlinearity_values` (`g(0)=0`), `test_ica_pipeline_transformer` (REQ-1);
`test_ica_mixing_shape` `(2,2)`, `test_ica_mean_shape`, `test_ica_n_components_getter`
(REQ-2); `test_ica_error_zero_components`, `test_ica_error_too_many_components`,
`test_ica_error_insufficient_samples`, `test_ica_transform_shape_mismatch` (REQ-3);
plus the module doctest. There is NO `tests/divergence_fast_ica.rs` yet. **FLAG for
the critic:** no in-tree test pins the meaningful recovery-up-to-perm+sign+scale
property (abs-corr ≈ 1 on a known independent-source mixture, Probe 1) — the real
"did ICA work" check, distinct from the REQ-4 value carve-out. REQ-4 (value parity)
is a CARVE-OUT (R-DEFER-3, no failing test).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-4 value gap,
the REQ-5 `components_=W@K` semantics, and REQ-6 `mixing_=pinv`:
```bash
# REQ-1 recovery + REQ-5/6 attribute semantics (sklearn components_=W@K, mixing_=pinv):
python3 -c "import numpy as np; from sklearn.decomposition import FastICA
rng=np.random.RandomState(0); n=400; t=np.linspace(0,8,n)
S=np.c_[np.sign(np.sin(2*t)), np.mod(t,2)-1]+0.02*rng.normal(size=(n,2))
X=S.dot(np.array([[1.,.7],[.5,1.2]]).T)
ica=FastICA(n_components=2, random_state=0); Sr=ica.fit_transform(X)
C=np.abs(np.corrcoef(Sr.T,S.T)[:2,2:])
print(ica.components_.shape, bool(np.allclose(ica.components_, ica._unmixing@ica.whitening_)),
      bool(np.allclose(ica.mixing_, np.linalg.pinv(ica.components_))), np.round(np.max(C,axis=1),4).tolist())"
# -> (2, 2) True True [0.9999, 0.9978]
#    => components_=W@K, mixing_=pinv(components_); recovery up to perm+sign+scale (NOT element-wise values)
```
The recovery oracle uses a KNOWN independent-source mixture (square + sawtooth);
the abs-corr ≈ 1 property is the observable ICA-correctness check (R-CHAR-3), NOT
copied from ferrolearn.

ferrolearn-python (REQ-14, PARTIAL at baseline): `_RsFastICA`
(`extras.rs:1108-1113`, via `py_transformer!` `extras.rs:107-149`) binds ONLY
`n_components` ctor + `fit` + `transform` — no `algorithm`/`fun`/`max_iter`/`tol`/
`random_state` ctor args, no `inverse_transform`, no fitted-attr getters. The
non-test consumers of `FastICA`/`FittedFastICA` are the crate re-export
(`lib.rs:87`), `_RsFastICA` (`extras.rs:1108`), and `PipelineTransformer`
(`fast_ica.rs:713-738`).

## Blockers

(#1571 is this doc's crosslink tracking issue. The blockers below are the open work
items the dispatcher files / numbers; none are filed by this doc — markdown only.)

- **#1572** — REQ-4 (CARVE-OUT): reach EXACT `components`/source value parity by
  matching sklearn's numpy `RandomState` `w_init` (`_fastica.py:638-641`), the SVD
  whitening DEFAULT (`:620-621`, `u*=sign(u[0])` `:624`), and a fixed
  permutation/sign/scale alignment of the recovered sources; inherently
  RNG/identifiability-bound (no failing test, R-DEFER-3 — the observable check is
  abs-corr recovery).
- **#1573** — REQ-5: store `components_ = W @ K` (`_fastica.py:683`, shape `(k,
  n_features)`) and expose `whitening_ = K` (`:685`) + `_unmixing = W` (`:690`)
  separately, matching sklearn's attribute semantics (ferrolearn currently exposes
  `components() = W`).
- **#1574** — REQ-6: compute `mixing = pinv(components_)` (`_fastica.py:689`, the
  Moore-Penrose pseudo-inverse of `W@K`) instead of the transpose-product `Kᵀ Wᵀ`
  (`fast_ica.rs:657`).
- **#1575** — REQ-7: add a `whiten_solver` field (`svd` DEFAULT via real SVD
  `_fastica.py:620-621` + `u*=sign(u[0])` `:624`; `eigh` `:605-619`), a `whiten`
  field (`unit-variance` `S_std` rescale `:676-681`, `arbitrary-variance`, `False`),
  and the `X1*=sqrt(n)` scaling (`:631`).
- **#1576** — REQ-8: support `fun` as a user callable (`_fastica.py:580-583`) and a
  `fun_args` dict with `alpha∈[1,2]` (`:570-572`) parametrising `_logcosh`
  `g=tanh(alpha·x)` (`:141-150`).
- **#1577** — REQ-9: add an optional `w_init` `(k,k)` matrix param
  (`_fastica.py:530`,`:637-649`) overriding the internally-drawn init.
- **#1578** — REQ-10: accept `n_components=None` → `min(n_samples, n_features)`
  (`_fastica.py:591-592`) and clamp-with-warning instead of rejecting `>n_features`
  (`:593-597`).
- **#1579** — REQ-11: add `inverse_transform` (`_fastica.py:764+`): `X @ mixing_.T
  + mean_`.
- **#1580** — REQ-12: expose `whitening()` / `n_features_in()` accessors and align
  the `components_`/`mixing_`/`n_iter_` attr naming + semantics on
  `FittedFastICA`.
- **#1581** — REQ-13: match sklearn's numpy `RandomState.normal` `w_init` draw
  (`_fastica.py:638-641`) for bit-level RNG parity (prerequisite for #1572).
- **#1582** — REQ-14: extend the `_RsFastICA` PyO3 binding
  (`extras.rs:1108-1113`) beyond `n_components`/`fit`/`transform` — add
  `algorithm`/`fun`/`max_iter`/`tol`/`random_state` ctor args, `inverse_transform`,
  and the fitted-attr getters (`components_`/`mixing_`/`mean_`/`whitening_`/`n_iter_`).
- **#1583** — REQ-15: migrate `fast_ica.rs` off `ndarray` + `rand`/`rand_distr`/
  `rand_xoshiro` to `ferray-core` arrays / `ferray::random` (R-SUBSTRATE).
