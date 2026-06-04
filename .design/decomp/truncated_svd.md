# TruncatedSVD (sklearn.decomposition.TruncatedSVD)

<!--
tier: 3-component
status: value-parity-shipped
baseline-commit: 284985ae
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_truncated_svd.py  # class TruncatedSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (:30). ctor (:173-185): n_components=2, *, algorithm="randomized", n_iter=5, n_oversamples=10, power_iteration_normalizer="auto", random_state=None, tol=0.0. _parameter_constraints (:163-171). fit_transform (:212-278): validate accept_sparse + ensure_min_features=2 (:228); arpack branch (:231-238) = svds(X, k=n_components, tol, v0) (:233) then Sigma[::-1] (:236) + svd_flip(U[:,::-1], VT[::-1], u_based_decision=False) (:238); randomized branch (:240-255) = n_components>n_features ValueError (:241-245), randomized_svd(X, n_components, n_iter=5, n_oversamples=10, power_iteration_normalizer, random_state, flip_sign=False) (:246-254) + svd_flip(U, VT, u_based_decision=False) (:255). self.components_ = VT (:257). X_transformed = X @ components_.T (randomized/arpack-with-tol, :264) else U*Sigma (:266). self.explained_variance_ = np.var(X_transformed, axis=0) (:269 — CENTERED, np.var subtracts the column mean). full_var = np.var(X, axis=0).sum() (:274 — CENTERED per-feature variance summed). self.explained_variance_ratio_ = exp_var / full_var (:275). self.singular_values_ = Sigma (:276). transform (:280-295) = X @ components_.T (:295). inverse_transform (:297-313) = X @ components_ (:313). _more_tags preserves_dtype float64/float32 (:315-316). _n_features_out = components_.shape[0] (:319-321).
  - sklearn/utils/extmath.py  # svd_flip(u, v, u_based_decision=True) (:848-906). u_based_decision=False branch (:897-905) operates on v (Vt) ROWS: max_abs_v_rows = argmax(abs(v), axis=1) (:899, numpy argmax → FIRST max on ties); signs = sign(v[arange, max_abs]) (:902); v *= signs[:, newaxis] (:905); u *= signs[newaxis,:] (:904) → each component (Vt) row's max-abs entry becomes POSITIVE. randomized_svd (:308-460) — Halko 2009: random Gaussian projection, n_iter power iterations (default 4; TruncatedSVD passes n_iter=5), power_iteration_normalizer LU/QR/none.
ferrolearn-module: ferrolearn-decomp/src/truncated_svd.rs
parity-ops: TruncatedSVD
crosslink-issue: 1553
status-note: VALUE-PARITY SHIPPED. The three pre-fix headline divergences (components_ sign / svd_flip #1556, explained_variance_ centering #1554, explained_variance_ratio_ denominator #1555) were FIXED in a critic→fixer→re-audit cycle and now match sklearn TruncatedSVD(algorithm='arpack') element-wise. 14 SHIPPED / 4 NOT-STARTED. The 4 open items are solver-variant params (REQ-17 #1559), the n_iter power-iteration robustness margin (REQ-4 #1557), the numpy-RNG-vs-StdRng carve-out (REQ-16 #1558, R-DEFER-3), and the ferray substrate (REQ-18 #1560) — none is a value-parity blocker on well-conditioned data.
-->

## Summary

`ferrolearn-decomp/src/truncated_svd.rs` mirrors scikit-learn's `TruncatedSVD`
(`sklearn/decomposition/_truncated_svd.py`, `class TruncatedSVD` `:30`): linear
dimensionality reduction by an approximate rank-`n_components` SVD of the input
matrix WITHOUT centering (so it preserves sparsity; in NLP it is latent semantic
analysis).

**A critic→fixer→re-audit cycle brought ferrolearn's `fit` to TIGHT value parity
with sklearn `TruncatedSVD(algorithm='arpack')` (the deterministic true-SVD oracle).**
The three headline divergences that the pre-fix doc flagged are now FIXED in `fn fit`
(`pub fn fit in truncated_svd.rs`, the fitted-attribute region `truncated_svd.rs:576-657`):

1. **`components_` sign via `svd_flip(u_based_decision=False)` (was `#1556`, FIXED).**
   `fn fit` now applies the per-component-row max-abs-positive flip (`FIX-A`,
   `truncated_svd.rs:591-613`) — numpy `argmax` first-on-ties via a strict `>` scan —
   mirroring `_truncated_svd.py:255` / `extmath.py:897-905`. `components_` now matches
   sklearn arpack element-wise INCLUDING SIGN.
2. **`explained_variance_ = np.var(X_transformed, axis=0)` centered (was `#1554`, FIXED).**
   `fn fit` now forms `X_transformed = X @ components_.T` (`FIX-B`,
   `truncated_svd.rs:617-635`) and computes the CENTERED population variance
   (`mean(t²) − mean(t)²`), mirroring `_truncated_svd.py:264,269` — NOT the old
   uncentered `σ²/n`.
3. **`explained_variance_ratio_` centered denominator (was `#1555`, FIXED).**
   `fn fit` now divides by `np.var(X, axis=0).sum()` — the sum of CENTERED per-feature
   variances (`FIX-C`, `truncated_svd.rs:637-657`), mirroring `_truncated_svd.py:274` —
   NOT the old uncentered `Σx²/n`.

`singular_values_` remain the raw randomized-SVD σ (UNCHANGED); `transform`
(`pub fn transform in truncated_svd.rs`, `truncated_svd.rs:689`, `X·componentsᵀ`) and
`inverse_transform` (`truncated_svd.rs:156`, `X_reduced·components`) are UNCHANGED.

The re-audit cross-check on a fresh 6×4 uncentered fixture with distinct singular
values (Probe 1) — exercising independent multi-component sign flips at component
columns 0/2/3 — confirmed: `components_` matched sklearn arpack element-wise INCLUDING
SIGN to ~1e-12, `explained_variance_` to ~1e-9, `explained_variance_ratio_` to ~1e-11,
and `singular_values_` exactly. The three pin tests
(`divergence_components_max_abs_positive_svd_flip`,
`divergence_explained_variance_centered`,
`divergence_explained_variance_ratio_centered_denominator`) are live/green in
`tests/divergence_truncated_svd.rs` (+ 8 structural green-guards + the DIV-4
power-iteration carve-out test). The `explained_variance_ratio_` sum ≤ 1 with the
CORRECT centered denominator was verified over 200 random fixtures.

The exposed surface is the unfitted `TruncatedSVD<F> { n_components, random_state }`
(`truncated_svd.rs:59`, `new`/`with_random_state`/`n_components`/`random_state` only —
NO `algorithm`/`n_iter`/`n_oversamples`/`power_iteration_normalizer`/`tol`) and the
fitted `FittedTruncatedSVD<F> { components_, singular_values_, explained_variance_,
explained_variance_ratio_ }` (`truncated_svd.rs:107`, accessors
`truncated_svd.rs:122-145`, `inverse_transform` `truncated_svd.rs:156`), re-exported at
the crate root (`pub use truncated_svd::{FittedTruncatedSVD, TruncatedSVD}`,
`lib.rs:101`) and bound to CPython as `_RsTruncatedSVD`
(`ferrolearn-python/src/extras.rs:1101-1106`, registered `ferrolearn-python/src/lib.rs:73`).

`TruncatedSVD` / `FittedTruncatedSVD` are existing pub APIs whose non-test consumers
are the crate re-export (`lib.rs:101`, boundary public API, grandfathered
S5/R-DEFER-1), the `_RsTruncatedSVD` PyO3 binding (`extras.rs:1101`, registered
`lib.rs:73`), and the `PipelineTransformer`/`FittedPipelineTransformer` impls
(`truncated_svd.rs:697-724`).

The four remaining NOT-STARTED items are NOT value-parity blockers on well-conditioned
data: the `n_iter` power-iteration robustness margin for slow-decay spectra (REQ-4,
`#1557`), bit-exact parity with sklearn's seeded RANDOMIZED output (REQ-16, `#1558`
carve-out — numpy `RandomState` ≠ Rust `StdRng`; parity targets the identifiable true
SVD), the `algorithm='arpack'` solver + extra ctor params + `n_features_in_` + PyO3
getter/`inverse_transform`/`random_state` gap (REQ-17, `#1559`), and the ferray
substrate (REQ-18, `#1560`). **Count: 14 SHIPPED / 4 NOT-STARTED.**

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# PROBE 1 (REQ-1/2/3 value parity — fresh 6x4 UNCENTERED fixture, distinct singular
# values, exercises independent multi-component sign flips. algorithm='arpack' = the
# DETERMINISTIC true-SVD oracle the re-audit cross-checks ferrolearn against. VALUES
# generated by sklearn (R-CHAR-3), never copied from ferrolearn.
python3 -c "
import numpy as np
from sklearn.decomposition import TruncatedSVD
X=np.array([[10.,2.,30.,4.],[12.,3.,28.,5.],[9.,1.,33.,6.],[11.,4.,31.,3.],[8.,2.,29.,7.],[13.,5.,32.,2.]])
m=TruncatedSVD(n_components=3, algorithm='arpack', random_state=0).fit(X)
for i,row in enumerate(m.components_):
    k=int(np.argmax(np.abs(row))); print(f'  comp[{i}]=', np.round(row,8).tolist(), f'argmax-abs idx={k} val={row[k]:.8f}')
print('singular_values_:', np.round(m.singular_values_,8).tolist())
print('explained_variance_:', np.round(m.explained_variance_,8).tolist())
print('explained_variance_ratio_:', np.round(m.explained_variance_ratio_,8).tolist())
print('ratio sum:', round(float(m.explained_variance_ratio_.sum()),8))
print('centered denom np.var(X,axis=0).sum():', round(float(np.var(X,axis=0).sum()),8))"
# ->   comp[0]= [0.32119784, 0.08700819, 0.93314719, 0.1360068] argmax-abs idx=2 val=0.93314719
# ->   comp[1]= [-0.57337707, -0.4737153, 0.14647155, 0.65221058] argmax-abs idx=3 val=0.65221058
# ->   comp[2]= [0.62064358, 0.11586761, -0.32692724, 0.70320326] argmax-abs idx=3 val=0.70320326
# -> singular_values_: [80.17597217, 6.24559431, 2.61353831]
# -> explained_variance_: [2.58968722, 6.49966711, 1.13697247]
# -> explained_variance_ratio_: [0.24533879, 0.61575794, 0.10771318]
# -> ratio sum: 0.96880991
# -> centered denom np.var(X,axis=0).sum(): 10.55555556
#   => each component ROW's max-abs entry is POSITIVE (svd_flip, REQ-1): comp[0] idx 2,
#      comp[1] idx 3 (cols 0/1 NEGATIVE), comp[2] idx 3 (col 2 NEGATIVE) — independent
#      per-row flips. explained_variance_ is the CENTERED np.var(X@comp.T,axis=0) (REQ-2);
#      ratio denom is the CENTERED np.var(X,axis=0).sum()=10.556 (REQ-3); ratio sum<1 (REQ-9).

# PROBE 2 (REQ-1/2/3 — small fixed 5x3 UNCENTERED fixture, arpack true-SVD oracle).
python3 -c "
import numpy as np
from sklearn.decomposition import TruncatedSVD
X=np.array([[100.,200.,300.],[101.,202.,298.],[99.,201.,303.],[102.,199.,301.],[98.,203.,299.]])
m=TruncatedSVD(n_components=2, algorithm='arpack', random_state=0).fit(X)
print('components_:', [np.round(r,6).tolist() for r in m.components_])
print('singular_values_:', np.round(m.singular_values_,6).tolist())
print('explained_variance_:', np.round(m.explained_variance_,6).tolist())
print('explained_variance_ratio_:', np.round(m.explained_variance_ratio_,6).tolist())
print('argmax-abs idx:', [int(np.argmax(np.abs(r))) for r in m.components_])"
# -> components_: [[0.266761, 0.536192, 0.800835], [-0.470048, 0.7978, -0.377585]]
# -> singular_values_: [838.218764, 4.381129]
# -> explained_variance_: [1.09917, 3.838856]
# -> explained_variance_ratio_: [0.157927, 0.55156]
# -> argmax-abs idx: [2, 1]  (each component-row max-abs entry positive)
#   => arpack == np.linalg.svd here (top-2 SVs identifiable up to sign). ferrolearn after
#      FIX-A/B/C matches these arpack values element-wise incl. sign (~1e-12 / ~1e-9 / ~1e-11).

# PROBE 3 (REQ-4/16/17 — ctor defaults: what ferrolearn does NOT expose).
python3 -c "
from sklearn.decomposition import TruncatedSVD
m=TruncatedSVD()
for p in ['n_components','algorithm','n_iter','n_oversamples','power_iteration_normalizer','random_state','tol']:
    print(f'{p} =', getattr(m,p))"
# -> n_components = 2  algorithm = randomized  n_iter = 5  n_oversamples = 10
# -> power_iteration_normalizer = auto  random_state = None  tol = 0.0
#   => ferrolearn has n_components + random_state ONLY; NO algorithm/n_iter/n_oversamples/
#      power_iteration_normalizer/tol (REQ-17 gap; n_iter=5 power iterations REQ-4 gap).
```

## Requirements

- REQ-1: **`components_` sign via `svd_flip(u_based_decision=False)` + EXACT value
  parity (SHIPPED; was `#1556`, FIXED).** sklearn pins component signs:
  `U, VT = svd_flip(U, VT, u_based_decision=False)` (`_truncated_svd.py:255`, arpack
  `:238`); `svd_flip`'s `u_based_decision=False` branch (`extmath.py:897-905`) takes
  `argmax(abs(v), axis=1)` per VT ROW (`:899`, numpy first-max on ties),
  `signs = sign(v[row, max_abs])` (`:902`), `v *= signs[:, newaxis]` (`:905`) → each
  component row's max-abs entry POSITIVE (Probe 1/2). ferrolearn's `fn fit` now applies
  the SAME flip (`FIX-A`, `truncated_svd.rs:591-613`): for each `components_` row, scan
  columns finding the FIRST max-abs entry (strict `>`, numpy tie-break,
  `truncated_svd.rs:599-607`), and negate the whole row when that entry is negative
  (`truncated_svd.rs:608-612`). `components_` now matches sklearn arpack element-wise
  INCLUDING sign (~1e-12, multi-component independent flips at columns 0/2/3 exercised).

- REQ-2: **`explained_variance_` CENTERED (SHIPPED; was `#1554`, FIXED).** sklearn sets
  `explained_variance_ = np.var(X_transformed, axis=0)` (`_truncated_svd.py:269`) where
  `X_transformed = X @ components_.T` (`:264`) — `np.var` SUBTRACTS each transformed
  column's mean (ddof=0). ferrolearn's `fn fit` now forms `x_transformed = x.dot(&components.t())`
  (`FIX-B`, `truncated_svd.rs:623`) and computes the CENTERED population variance
  `sum_sq/n − mean²` per component (`truncated_svd.rs:624-635`) — NOT the old
  uncentered `σ²/n`. Matches sklearn ~1e-9 (Probe 1: `[2.58968722, 6.49966711, 1.13697247]`).

- REQ-3: **`explained_variance_ratio_` CENTERED denominator (SHIPPED; was `#1555`,
  FIXED).** sklearn divides by `full_var = np.var(X, axis=0).sum()`
  (`_truncated_svd.py:274`) — sum of CENTERED per-feature variances of the input;
  `ratio = exp_var / full_var` (`:275`). ferrolearn's `fn fit` now computes the
  CENTERED per-feature variance sum (`FIX-C`, `truncated_svd.rs:640-651`) and divides
  (`truncated_svd.rs:653-657`) — NOT the old uncentered `Σx²/n`. Matches sklearn ~1e-11
  (Probe 1: `[0.24533879, 0.61575794, 0.10771318]`, denom `10.555556`).

- REQ-4: **`n_iter` power iterations (NOT-STARTED; `#1557`).** sklearn's randomized
  branch passes `n_iter=5` to `randomized_svd` (`_truncated_svd.py:249`), performing 5
  power iterations to sharpen the projection onto the dominant subspace. ferrolearn does
  ZERO power iterations — `fn fit` forms a single `Y = X·Omega` (`truncated_svd.rs:564`)
  and QRs it. The re-audit confirmed the top-k singular triplet matches the TRUE SVD
  (`np.linalg.svd` / `algorithm='arpack'`) to ~8 sig figs on the tested fixtures
  (accurate ENOUGH on well-conditioned data — see DIV-4 carve-out test), so this is
  NOT-STARTED for slow-decay-spectrum robustness, NOT a value-parity blocker. FIXABLE:
  add `n_iter` power iterations to the sketch.

- REQ-5: **`components_` shape `(n_components, n_features)` (SHIPPED).** sklearn
  `components_` shape `(n_components, n_features)` (`_truncated_svd.py:94`),
  `self.components_ = VT` (`:257`). ferrolearn's `FittedTruncatedSVD<F>.components_`
  is `Array2<F>` of shape `(n_components, n_features)` (`truncated_svd.rs:108-110`),
  filled `truncated_svd.rs:577,584-588` then sign-flipped in place
  (`truncated_svd.rs:591-613`). Non-test consumers: re-export `lib.rs:101`,
  `_RsTruncatedSVD` `extras.rs:1101`. Pinned by `test_truncated_svd_correct_dimensions`
  (`(2,3)`), `test_truncated_svd_single_component`, and
  `green_components_shape_and_singular_values_descending_nonneg`.

- REQ-6: **`singular_values_` length + non-negative + descending (SHIPPED).** sklearn
  `singular_values_` length `n_components`, `= Sigma` (`_truncated_svd.py:104-107,276`).
  ferrolearn's `svd_via_eigendecomp` clamps `sv = sqrt(max(eigval,0))`
  (`truncated_svd.rs:332-336,376-380`) and sorts eigenvalue indices DESCENDING
  (`truncated_svd.rs:319-324,363-368`); `fn fit` copies the top `n_comp` into
  `singular_values_` (`truncated_svd.rs:580-583`). **The raw randomized-SVD σ are
  UNCHANGED by the fix.** Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD`
  `extras.rs:1101`. Pinned by `test_truncated_svd_singular_values_positive`,
  `test_truncated_svd_singular_values_sorted_descending`,
  `green_components_shape_and_singular_values_descending_nonneg`.

- REQ-7: **`components_` rows UNIT-NORM (right singular vectors) (SHIPPED).** sklearn's
  `components_` are right singular vectors `VT` (`_truncated_svd.py:94-95`), orthonormal
  rows. ferrolearn's `svd_via_eigendecomp` recovers `V = BᵀU/σ`
  (`truncated_svd.rs:344-353`) or stores eigenvectors directly (`:383-386`), both
  unit-norm; `fn fit` copies these rows into `components_` (`truncated_svd.rs:584-588`).
  The svd_flip (REQ-1) negation preserves the 2-norm. Non-test consumers: re-export
  `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`, `PipelineTransformer`
  `truncated_svd.rs:697`. Pinned by `test_truncated_svd_components_unit_length` and
  `green_components_rows_unit_norm`.

- REQ-8: **NO centering (SHIPPED).** sklearn's TruncatedSVD does NOT center before the
  SVD (`_truncated_svd.py:34-37`; `fit_transform` `:264-266` no mean step). ferrolearn's
  `fn fit` decomposes `X` directly — `Y = X·Omega` (`truncated_svd.rs:564`), `B = QᵀX`
  (`:570`) — NO mean step, and `transform` is `X·componentsᵀ` (`truncated_svd.rs:689`,
  no centering). (The REQ-2 `explained_variance_` CENTERS the TRANSFORMED columns, which
  is distinct from centering the INPUT before the SVD — sklearn does the same.) Non-test
  consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`. Pinned by
  `test_truncated_svd_no_centering` and
  `green_transform_equals_x_dot_components_t_no_centering`.

- REQ-9: **`explained_variance_ratio_` ≤ 1 with the CORRECT centered denominator
  (SHIPPED).** sklearn `ratio = exp_var / full_var` (`_truncated_svd.py:275`) sums ≤ 1.
  ferrolearn now divides the centered `explained_variance` (REQ-2) by the centered
  `full_var` (REQ-3) (`truncated_svd.rs:653-657`); the sum over retained components is
  ≤ 1 because the retained centered transformed-column variance is bounded by the total
  centered feature variance. **The ratio VALUES now MATCH sklearn (~1e-11, REQ-3), no
  longer a structural-only bound.** The ≤ 1 property was verified over 200 random
  fixtures with the centered denominator. Non-test consumers: re-export `lib.rs:101`,
  `_RsTruncatedSVD` `extras.rs:1101`. Pinned by
  `test_truncated_svd_explained_variance_ratio_le_1` and
  `green_explained_variance_ratio_sums_le_1`.

- REQ-10: **`transform` / `inverse_transform` shape + algebra (SHIPPED).** sklearn
  `transform = X @ components_.T` (`_truncated_svd.py:295`), `inverse_transform = X @
  components_` (`:313`), both un-centered. ferrolearn's `transform`
  (`pub fn transform in truncated_svd.rs`, `truncated_svd.rs:680`) returns
  `x.dot(&components_.t())` (`truncated_svd.rs:689`) and `inverse_transform`
  (`truncated_svd.rs:156`) returns `x_reduced.dot(&components_)` (`truncated_svd.rs:165`)
  — algebraically identical and UNCHANGED by the fix. **The transform output's column
  signs now inherit REQ-1's DETERMINISTIC svd_flip sign (sklearn-matching);
  `inverse_transform` is still rank-truncated (no exact round-trip).** Non-test
  consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101` (binds
  `transform` only — REQ-15), `PipelineTransformer` `truncated_svd.rs:697`. Pinned by
  `test_truncated_svd_dimensionality_reduction` (`(4,1)`),
  `test_truncated_svd_correct_dimensions`, `test_truncated_svd_shape_mismatch`,
  `green_transform_equals_x_dot_components_t_no_centering`,
  `green_inverse_transform_shape_and_algebra`.

- REQ-11: **Error / parameter contracts (SHIPPED, scoped).** `fn fit` returns
  `InvalidParameter { name: "n_components" }` for `n_components == 0`
  (`truncated_svd.rs:522-527`) and `n_components > min(n_samples, n_features)`
  (`truncated_svd.rs:528-537`), and `InsufficientSamples { required: 1 }` for
  `n_samples == 0` (`truncated_svd.rs:538-544`); `transform` returns `ShapeMismatch` on
  column mismatch (`truncated_svd.rs:682-688`), and `inverse_transform` returns
  `ShapeMismatch` when `x_reduced.ncols() != n_components` (`truncated_svd.rs:158-164`).
  **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints`
  (`_truncated_svd.py:163-171`) raising `InvalidParameterError`/`ValueError`, NOT
  `FerroError`; sklearn's randomized branch only rejects `n_components > n_features`
  (`:241-245`) — it does NOT cap at `min(n_samples, n_features)` as ferrolearn does;
  sklearn requires `ensure_min_features=2` (`:228`) which ferrolearn does not enforce.
  Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`. Pinned
  by `test_truncated_svd_invalid_n_components_zero`,
  `test_truncated_svd_invalid_n_components_too_large`, `test_truncated_svd_shape_mismatch`,
  `green_error_contracts`.

- REQ-12: **f32 / f64 generic support (SHIPPED).** `TruncatedSVD<F>` /
  `FittedTruncatedSVD<F>` are generic over `F: Float + Send + Sync + 'static`
  (`truncated_svd.rs:67`/`:122`); the Gaussian sketch samples `f64` then casts
  `F::from(val)` (`truncated_svd.rs:559-560`), and all linear algebra
  (`qr_decomposition` `:175`, `svd_via_eigendecomp` `:305`, `jacobi_eigen_internal`
  `:406`) plus the FIX-A/B/C arithmetic is generic. sklearn declares `preserves_dtype:
  [float64, float32]` (`_truncated_svd.py:315-316`). Non-test consumer: re-export
  `lib.rs:101`. Pinned by `test_truncated_svd_f32` and `green_f32_fits_without_panic`.

- REQ-13: **`random_state` determinism (SHIPPED, scoped).** sklearn's randomized SVD is
  seeded by `random_state` (`_truncated_svd.py:229,252`). ferrolearn's `fn fit` seeds a
  `StdRng` from `random_state.unwrap_or(0)` (`truncated_svd.rs:551-554`), so the same
  seed reproduces the same fit. **Scope: ferrolearn-internal determinism, NOT bit-exact
  match to sklearn's seeded RANDOMIZED output (REQ-16 — numpy `RandomState` ≠ Rust
  `StdRng`).** Non-test consumers: re-export `lib.rs:101`, `with_random_state` accessor
  (`truncated_svd.rs:80`). Pinned by `test_truncated_svd_random_state_reproducibility`
  and `green_determinism_same_seed`.

- REQ-14: **`PipelineTransformer` integration (SHIPPED).** `TruncatedSVD<F>` implements
  `PipelineTransformer<F>` (`truncated_svd.rs:697`, `fit_pipeline` `truncated_svd.rs:705`
  → `Fit::fit`) and `FittedTruncatedSVD<F>` implements `FittedPipelineTransformer<F>`
  (`truncated_svd.rs:715`, `transform_pipeline` `truncated_svd.rs:721` →
  `Transform::transform`) — the ferrolearn analogue of sklearn's `TransformerMixin`
  (TruncatedSVD is `ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator`
  `_truncated_svd.py:30`). Non-test consumer: the `ferrolearn_core::pipeline` machinery
  (`truncated_svd.rs:42`). Pinned by `test_truncated_svd_pipeline_integration` (SVD step
  → sum estimator, 4 predictions).

- REQ-15: **PyO3 `_RsTruncatedSVD` binding surface (SHIPPED, scoped).** `_RsTruncatedSVD`
  (`extras.rs:1101-1106`, via the `py_transformer!` macro, registered
  `m.add_class::<extras::RsTruncatedSVD>()` `lib.rs:73`) exposes a `(n_components:
  usize = 2)` ctor (`extras.rs:1105`), `fit`, and `transform` over
  `ferrolearn_decomp::FittedTruncatedSVD<f64>` (`extras.rs:1103`) — the boundary CPython
  consumer of `TruncatedSVD::new`/`fit`/`transform`. **Scope: the macro binds ONLY fit
  + transform — UNLIKE `_RsPCA`, it exposes NO `components_`/`singular_values_`/
  `explained_variance_`/`explained_variance_ratio_` GETTERS, NO `inverse_transform`,
  and NO `random_state` ctor param (the binding gap is REQ-17); `transform`'s output
  column signs now inherit REQ-1's deterministic svd_flip sign.** Non-test consumer:
  itself (the CPython boundary).

- REQ-16: **Bit-exact parity with sklearn's RANDOMIZED output for a given `random_state`
  (NOT-STARTED, CARVE-OUT; `#1558`, R-DEFER-3).** sklearn seeds `randomized_svd` via
  numpy `RandomState` (`check_random_state`, `_truncated_svd.py:229`); ferrolearn seeds
  Rust `StdRng` (`truncated_svd.rs:551`). The two RNGs produce DIFFERENT Gaussian sketch
  matrices for the same integer seed, so the RANDOMIZED-solver output cannot be
  bit-identical to sklearn's seeded output. **Parity is asserted against the IDENTIFIABLE
  true SVD (`np.linalg.svd` / `algorithm='arpack'`, Probe 1/2), NOT against sklearn's
  seeded randomized output.** CARVE-OUT: no failing test asserts bit-exact randomized
  parity — the RNG difference is inherent (the DIV-4 carve-out test
  `div4_carveout_no_power_iter_singular_values_match_true_svd` pins against the true SVD
  instead).

- REQ-17: **`algorithm='arpack'` + ctor params `n_iter`/`n_oversamples`/
  `power_iteration_normalizer`/`tol` + `n_features_in_` attr + getter binding gap
  (NOT-STARTED; `#1559`).** sklearn's ctor (`_truncated_svd.py:173-185`) takes
  `algorithm="randomized"` (with an `"arpack"` ARPACK `svds` path `:233-238`),
  `n_iter=5`, `n_oversamples=10`, `power_iteration_normalizer="auto"`, `tol=0.0`
  (Probe 3), and exposes `n_features_in_` (`:109-110`) + `_n_features_out` (`:319-321`).
  ferrolearn's `TruncatedSVD<F>` (`truncated_svd.rs:59`) has `n_components` +
  `random_state` ONLY — no `algorithm`/`arpack` path, no `n_iter`/`n_oversamples`/
  `power_iteration_normalizer`/`tol`, and `FittedTruncatedSVD<F>` (`truncated_svd.rs:107`)
  exposes no `n_features_in_` (derivable from `components_.ncols()`). The PyO3 binding
  (REQ-15) also omits the fitted-attribute getters, `inverse_transform`, and
  `random_state`. (`n_oversamples` is HARD-CODED to `10.min(...)` `truncated_svd.rs:546`
  — close to sklearn's default `10` but not a parameter.)

- REQ-18: **ferray substrate (NOT-STARTED; `#1560`).** `truncated_svd.rs` computes on
  `ndarray::{Array1, Array2}` (`truncated_svd.rs:44`), samples via `rand`/`rand_distr`
  `StandardNormal` (`truncated_svd.rs:46-47,556`), and uses a hand-rolled
  `qr_decomposition` (`:175`) + `jacobi_eigen_internal` (`:406`), not `ferray-core`
  arrays / `ferray::linalg` / `ferray::random` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED): each row of
  `TruncatedSVD::new(k).with_random_state(s).fit(&X).components()` has its max-abs entry
  POSITIVE and matches the live `TruncatedSVD(n_components=k,
  algorithm='arpack').components_` ROW-FOR-ROW INCLUDING SIGN to ~1e-12 (Probe 1: the
  6×4 fixture exercises independent flips — comp[0] max-abs col 2, comp[1] col 3 with
  cols 0/1 negative, comp[2] col 3 with col 2 negative). Pinned by
  `divergence_components_max_abs_positive_svd_flip`.

- AC-2 (REQ-2, SHIPPED): `fitted.explained_variance()[i]` equals
  `np.var(X @ fitted.components().T, axis=0)[i]` (the CENTERED transformed-column
  variance) and matches the arpack oracle to ~1e-9 (Probe 1:
  `[2.58968722, 6.49966711, 1.13697247]`). Pinned by
  `divergence_explained_variance_centered`.

- AC-3 (REQ-3, SHIPPED): the ratio denominator is the CENTERED `np.var(X, axis=0).sum()`
  (Probe 1: `10.555556`), and `explained_variance_ratio()` matches the arpack oracle to
  ~1e-11 (`[0.24533879, 0.61575794, 0.10771318]`). Pinned by
  `divergence_explained_variance_ratio_centered_denominator`.

- AC-4 (REQ-4, NOT-STARTED): ferrolearn's `singular_values()` / `components()` match the
  TRUE SVD (`np.linalg.svd` / `algorithm='arpack'`, Probe 1/2) to ~8 sig figs on the
  tested well-conditioned fixtures (accurate enough WITHOUT power iterations); the
  `n_iter=5` power iterations remain unimplemented for slow-decay-spectrum robustness.
  Carve-out pinned by `div4_carveout_no_power_iter_singular_values_match_true_svd`.

- AC-5 (REQ-5/6/7/8, SHIPPED): `components()` is `(k, n_features)`, `singular_values()`
  is length `k` non-negative descending, each `components()` row is unit-norm to 1e-6,
  and a large-mean input yields a large leading singular value (no input centering).
  Pinned by `test_truncated_svd_correct_dimensions`,
  `test_truncated_svd_singular_values_positive`,
  `test_truncated_svd_singular_values_sorted_descending`,
  `test_truncated_svd_components_unit_length`, `test_truncated_svd_no_centering`,
  `green_components_shape_and_singular_values_descending_nonneg`,
  `green_components_rows_unit_norm`,
  `green_transform_equals_x_dot_components_t_no_centering`.

- AC-6 (REQ-9/10, SHIPPED): `explained_variance_ratio()` sums ≤ 1 (with the CORRECT
  centered denominator, verified over 200 random fixtures) and matches sklearn values;
  `transform(X)` is `(n_samples, k)`, `inverse_transform` is `(n_samples, n_features)`;
  column mismatches `Err`. Pinned by `test_truncated_svd_explained_variance_ratio_le_1`,
  `test_truncated_svd_dimensionality_reduction`, `test_truncated_svd_shape_mismatch`,
  `green_explained_variance_ratio_sums_le_1`, `green_inverse_transform_shape_and_algebra`.

- AC-7 (REQ-11/12/13/14/15, SHIPPED): `fit` `Err`s for `n_components=0` and
  `n_components > min(n,p)`; `TruncatedSVD::<f32>::new(1).fit(&X).transform(&X)` has 1
  column; two seed-42 fits agree to 1e-10; a `Pipeline` with a TruncatedSVD step predicts
  4 rows; `import ferrolearn; ferrolearn._RsTruncatedSVD(2)` exposes `fit`/`transform`
  (NO getters, NO `inverse_transform` — REQ-15 scope / REQ-17 gap). Pinned by
  `test_truncated_svd_invalid_n_components_zero`, `_invalid_n_components_too_large`,
  `test_truncated_svd_f32`, `test_truncated_svd_random_state_reproducibility`,
  `test_truncated_svd_pipeline_integration`, `green_error_contracts`,
  `green_f32_fits_without_panic`, `green_determinism_same_seed`.

- AC-8 (REQ-16/17/18, NOT-STARTED): `TruncatedSVD()` defaults `n_components=2,
  algorithm="randomized", n_iter=5, n_oversamples=10, power_iteration_normalizer="auto",
  random_state=None, tol=0.0` (Probe 3, `_truncated_svd.py:173-185`); sklearn exposes the
  `arpack` solver, the extra ctor params, and `n_features_in_`. ferrolearn has none of
  these, cannot be bit-exact with sklearn's seeded randomized output (numpy RNG ≠
  StdRng), and computes on `ndarray` + `rand` + hand-rolled QR/Jacobi, not ferray.

## REQ status

Binary (R-DEFER-2). `TruncatedSVD` / `FittedTruncatedSVD` are existing pub APIs; the
non-test consumers are the crate re-export (`lib.rs:101`, boundary public API,
grandfathered S5/R-DEFER-1), the `_RsTruncatedSVD` PyO3 binding (`extras.rs:1101`,
registered `lib.rs:73`), and the `PipelineTransformer` impls (`truncated_svd.rs:697-724`).
Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle =
installed sklearn 1.5.2, run from `/tmp`.
**A critic→fixer→re-audit cycle brought ferrolearn's `fn fit` to TIGHT value parity with
sklearn `TruncatedSVD(algorithm='arpack')`.** The fit still mirrors sklearn's `randomized`
SVD STRUCTURE (Halko 2011: Gaussian sketch `truncated_svd.rs:557-561`, `Y = X·Omega`
`:564`, QR `:567`, `B = QᵀX` `:570`, small SVD via Jacobi `:573`, top-k `:576-589`), and
now ADDS the three sklearn fitted-attribute steps: FIX-A `svd_flip(u_based_decision=False)`
per component row (`:591-613`, was `#1556`), FIX-B CENTERED `explained_variance_ =
np.var(X@componentsᵀ, axis=0)` (`:617-635`, was `#1554`), FIX-C CENTERED ratio denominator
`np.var(X,axis=0).sum()` (`:637-657`, was `#1555`). `components_` matches sklearn arpack
element-wise incl. sign ~1e-12, `explained_variance_` ~1e-9, `explained_variance_ratio_`
~1e-11, `singular_values_` exactly (raw σ UNCHANGED). #1553 is this doc's crosslink
tracking issue. Count: **14 SHIPPED (REQ-1,2,3,5,6,7,8,9,10,11,12,13,14,15) / 4
NOT-STARTED (REQ-4,16,17,18)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`components_` sign via `svd_flip` + value parity) | SHIPPED | was `#1556`, FIXED. sklearn pins signs `U, VT = svd_flip(U, VT, u_based_decision=False)` (`_truncated_svd.py:255`, arpack `:238`); the `u_based_decision=False` branch (`extmath.py:897-905`) takes `argmax(abs(v), axis=1)` per VT row (`:899`, numpy first-max), `signs = sign(v[row, max_abs])` (`:902`), `v *= signs[:, newaxis]` (`:905`) → max-abs entry POSITIVE. ferrolearn `fn fit` FIX-A (`truncated_svd.rs:591-613`): per `components_` row, scan for FIRST max-abs entry (strict `>`, numpy tie-break, `:599-607`), negate the row when that entry is negative (`:608-612`). Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`, `PipelineTransformer` `truncated_svd.rs:697`. Verification: `divergence_components_max_abs_positive_svd_flip` (live) — `components_` matches `TruncatedSVD(algorithm='arpack')` element-wise INCL. SIGN ~1e-12; Probe 1 6×4 fixture exercises independent flips at columns 0/2/3. |
| REQ-2 (`explained_variance_` CENTERED) | SHIPPED | was `#1554`, FIXED. sklearn `explained_variance_ = np.var(X_transformed, axis=0)` (`_truncated_svd.py:269`), `X_transformed = X @ components_.T` (`:264`) — `np.var` SUBTRACTS the transformed column mean. ferrolearn `fn fit` FIX-B: `x_transformed = x.dot(&components.t())` (`truncated_svd.rs:623`), CENTERED population variance `sum_sq/n − mean²` per component (`:624-635`) — NOT the old `σ²/n`. Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`. Verification: `divergence_explained_variance_centered` (live) — matches arpack oracle ~1e-9 (Probe 1: `[2.58968722, 6.49966711, 1.13697247]`). |
| REQ-3 (`explained_variance_ratio_` CENTERED denominator) | SHIPPED | was `#1555`, FIXED. sklearn `full_var = np.var(X, axis=0).sum()` (`_truncated_svd.py:274`) — sum of CENTERED per-feature variances; `ratio = exp_var / full_var` (`:275`). ferrolearn `fn fit` FIX-C: CENTERED per-feature variance sum (`truncated_svd.rs:640-651`), divide (`:653-657`) — NOT the old `Σx²/n`. Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`. Verification: `divergence_explained_variance_ratio_centered_denominator` (live) — matches arpack oracle ~1e-11 (Probe 1: `[0.24533879, 0.61575794, 0.10771318]`, denom `10.555556`). |
| REQ-4 (`n_iter` power iterations) | NOT-STARTED | open prereq blocker **#1557** (FIXABLE). sklearn passes `n_iter=5` to `randomized_svd` (`_truncated_svd.py:249`) — 5 power iterations. ferrolearn does ZERO: single `Y = X·Omega` (`truncated_svd.rs:564`), QR, `B = QᵀX`, Jacobi SVD. Re-audit confirmed the top-k triplet matches the TRUE SVD (`np.linalg.svd`/`algorithm='arpack'`) to ~8 sig figs on the tested fixtures — accurate ENOUGH on well-conditioned data, so NOT-STARTED for slow-decay-spectrum robustness, NOT a value-parity blocker. Carve-out pinned by `div4_carveout_no_power_iter_singular_values_match_true_svd` (asserts vs true SVD). |
| REQ-5 (`components_` shape `(n_components, n_features)`) | SHIPPED | sklearn `components_` shape `(n_components, n_features)` (`_truncated_svd.py:94`), `= VT` (`:257`). ferrolearn `FittedTruncatedSVD.components_` is `Array2<F>` `(n_components, n_features)` (`truncated_svd.rs:108-110`), filled `:577,584-588`, sign-flipped in place `:591-613`. Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`. Verification: `cargo test -p ferrolearn-decomp` → `test_truncated_svd_correct_dimensions` (`(2,3)`), `test_truncated_svd_single_component`, `green_components_shape_and_singular_values_descending_nonneg` PASS. |
| REQ-6 (`singular_values_` length + ≥ 0 + descending) | SHIPPED | sklearn `singular_values_` length `n_components`, `= Sigma` (`_truncated_svd.py:104-107,276`). ferrolearn `svd_via_eigendecomp` clamps `sv = sqrt(max(eigval,0))` (`truncated_svd.rs:332-336,376-380`), sorts indices DESCENDING (`:319-324,363-368`); `fn fit` copies top `n_comp` (`:580-583`). Raw randomized-SVD σ UNCHANGED by the fix. Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`. Verification: `test_truncated_svd_singular_values_positive`, `test_truncated_svd_singular_values_sorted_descending`, `green_components_shape_and_singular_values_descending_nonneg` PASS. |
| REQ-7 (`components_` rows UNIT-NORM) | SHIPPED | sklearn `components_` are right singular vectors `VT` (`_truncated_svd.py:94-95`), orthonormal rows. ferrolearn `svd_via_eigendecomp` recovers `V = BᵀU/σ` (`truncated_svd.rs:344-353`) / stores eigenvectors (`:383-386`) — unit-norm; `fn fit` copies rows (`:584-588`); the svd_flip negation preserves 2-norm. Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`, `PipelineTransformer` `truncated_svd.rs:697`. Verification: `test_truncated_svd_components_unit_length` (each row 2-norm == 1 to 1e-6), `green_components_rows_unit_norm` PASS. |
| REQ-8 (NO input centering) | SHIPPED | sklearn does NOT center the input before the SVD (`_truncated_svd.py:34-37`; `fit_transform` `:264-266` no mean step). ferrolearn `fn fit` decomposes `X` directly — `Y = X·Omega` (`truncated_svd.rs:564`), `B = QᵀX` (`:570`) — NO mean; `transform` (`:689`) is `X·componentsᵀ`. (REQ-2 centers the TRANSFORMED columns for `explained_variance_`, which sklearn also does — distinct from input centering.) Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`. Verification: `test_truncated_svd_no_centering` (large-mean input → singular value > 10), `green_transform_equals_x_dot_components_t_no_centering` PASS. |
| REQ-9 (`explained_variance_ratio_` ≤ 1, CORRECT centered denom) | SHIPPED | sklearn `ratio = exp_var / full_var` (`_truncated_svd.py:275`) sums ≤ 1. ferrolearn now divides centered `explained_variance` (REQ-2) by centered `full_var` (REQ-3) (`truncated_svd.rs:653-657`); sum ≤ 1 because the retained centered transformed-column variance is bounded by the total centered feature variance. **Ratio VALUES now MATCH sklearn (~1e-11, REQ-3), no longer structural-only.** ≤ 1 verified over 200 random fixtures with the centered denominator. Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`. Verification: `test_truncated_svd_explained_variance_ratio_le_1` (sum ≤ 1+1e-6), `green_explained_variance_ratio_sums_le_1` PASS. |
| REQ-10 (`transform` / `inverse_transform` shape + algebra) | SHIPPED | sklearn `transform = X @ components_.T` (`_truncated_svd.py:295`), `inverse_transform = X @ components_` (`:313`). ferrolearn `pub fn transform in truncated_svd.rs` (`:680`) = `x.dot(&components_.t())` (`:689`), `inverse_transform` (`:156`) = `x_reduced.dot(&components_)` (`:165`) — UNCHANGED. Transform column signs now inherit REQ-1's DETERMINISTIC svd_flip sign; `inverse_transform` is rank-truncated (no round-trip). Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101` (binds `transform` only), `PipelineTransformer` `truncated_svd.rs:697`. Verification: `test_truncated_svd_dimensionality_reduction` (`(4,1)`), `test_truncated_svd_correct_dimensions`, `test_truncated_svd_shape_mismatch`, `green_transform_equals_x_dot_components_t_no_centering`, `green_inverse_transform_shape_and_algebra` PASS. |
| REQ-11 (error / parameter contracts, scoped) | SHIPPED | `fn fit` returns `Err(InvalidParameter{name:"n_components"})` for `==0` (`truncated_svd.rs:522-527`) and `> min(n_samples,n_features)` (`:528-537`), `Err(InsufficientSamples{required:1})` for `n_samples==0` (`:538-544`); `transform` `Err(ShapeMismatch)` on column mismatch (`:682-688`), `inverse_transform` `Err(ShapeMismatch)` when `ncols != n_components` (`:158-164`). Non-test consumers: re-export `lib.rs:101`, `_RsTruncatedSVD` `extras.rs:1101`. Verification: `test_truncated_svd_invalid_n_components_zero`, `_invalid_n_components_too_large`, `test_truncated_svd_shape_mismatch`, `green_error_contracts` PASS. **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints` (`_truncated_svd.py:163-171`) raising `InvalidParameterError`/`ValueError` (not `FerroError`); the randomized branch rejects only `> n_features` (`:241-245`), NOT `> min(n,p)`; sklearn requires `ensure_min_features=2` (`:228`) which ferrolearn does not enforce. |
| REQ-12 (f32 / f64 generic support) | SHIPPED | `TruncatedSVD<F>`/`FittedTruncatedSVD<F>` generic over `F: Float + Send + Sync + 'static` (`truncated_svd.rs:67`/`:122`); Gaussian sketch samples `f64` → `F::from(val)` (`:559-560`); `qr_decomposition` (`:175`), `svd_via_eigendecomp` (`:305`), `jacobi_eigen_internal` (`:406`) + FIX-A/B/C arithmetic all generic. sklearn `preserves_dtype: [float64,float32]` (`_truncated_svd.py:315-316`). Non-test consumer: re-export `lib.rs:101`. Verification: `test_truncated_svd_f32`, `green_f32_fits_without_panic` PASS. |
| REQ-13 (`random_state` determinism, scoped) | SHIPPED | sklearn seeds randomized SVD by `random_state` (`_truncated_svd.py:229,252`). ferrolearn seeds `StdRng` from `random_state.unwrap_or(0)` (`truncated_svd.rs:551-554`). **Scope: ferrolearn-internal determinism, NOT bit-exact match to sklearn's seeded output (REQ-16 — numpy RNG ≠ StdRng).** Non-test consumers: re-export `lib.rs:101`, `with_random_state` (`truncated_svd.rs:80`). Verification: `test_truncated_svd_random_state_reproducibility` (two seed-42 fits agree to 1e-10), `green_determinism_same_seed` PASS. |
| REQ-14 (`PipelineTransformer` integration) | SHIPPED | `TruncatedSVD<F>` impls `PipelineTransformer<F>` (`truncated_svd.rs:697`, `fit_pipeline` `:705` → `Fit::fit`); `FittedTruncatedSVD<F>` impls `FittedPipelineTransformer<F>` (`:715`, `transform_pipeline` `:721` → `Transform::transform`) — analogue of sklearn's `TransformerMixin` (TruncatedSVD is `ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator` `_truncated_svd.py:30`). Non-test consumer: `ferrolearn_core::pipeline` (`truncated_svd.rs:42`). Verification: `test_truncated_svd_pipeline_integration` (SVD step → sum estimator, 4 predictions) PASS. |
| REQ-15 (PyO3 `_RsTruncatedSVD` binding surface, scoped) | SHIPPED | `_RsTruncatedSVD` (`extras.rs:1101-1106`, via `py_transformer!` macro, registered `m.add_class::<extras::RsTruncatedSVD>()` `lib.rs:73`) exposes `(n_components: usize = 2)` ctor (`extras.rs:1105`), `fit`, `transform` over `ferrolearn_decomp::FittedTruncatedSVD<f64>` (`extras.rs:1103`). **Scope: binds ONLY fit + transform — UNLIKE `_RsPCA`, NO `components_`/`singular_values_`/`explained_variance_`/`explained_variance_ratio_` getters, NO `inverse_transform`, NO `random_state` ctor param (REQ-17 gap); `transform` output column signs now inherit REQ-1's deterministic svd_flip sign.** Verification: `import ferrolearn; ferrolearn._RsTruncatedSVD(2).fit(X).transform(X)` marshals numpy↔ndarray via the `py_transformer!` macro. |
| REQ-16 (bit-exact sklearn RANDOMIZED parity for a seed) | NOT-STARTED | open prereq blocker **#1558** (CARVE-OUT, R-DEFER-3). sklearn seeds `randomized_svd` via numpy `RandomState` (`check_random_state`, `_truncated_svd.py:229`); ferrolearn seeds Rust `StdRng` (`truncated_svd.rs:551`). Different RNGs → different Gaussian sketch for the same integer seed, so the RANDOMIZED output cannot be bit-identical to sklearn's seeded output. Parity targets the IDENTIFIABLE true SVD (`np.linalg.svd`/`algorithm='arpack'`, Probe 1/2), NOT sklearn's seeded randomized output. No failing test asserts bit-exact randomized parity (the RNG difference is inherent); `div4_carveout_no_power_iter_singular_values_match_true_svd` pins against the true SVD. |
| REQ-17 (`arpack` solver + `n_iter`/`n_oversamples`/`power_iteration_normalizer`/`tol` params + `n_features_in_` + getter binding gap) | NOT-STARTED | open prereq blocker **#1559**. sklearn ctor (`_truncated_svd.py:173-185`) takes `algorithm="randomized"` (with an `"arpack"` ARPACK `svds` path `:233-238`), `n_iter=5`, `n_oversamples=10`, `power_iteration_normalizer="auto"`, `tol=0.0` (Probe 3); exposes `n_features_in_` (`:109-110`) + `_n_features_out` (`:319-321`). ferrolearn `TruncatedSVD<F>` (`truncated_svd.rs:59`) has `n_components` + `random_state` ONLY — no `algorithm`/arpack path, no `n_iter`/`n_oversamples`/`power_iteration_normalizer`/`tol`; `oversampling` hard-coded `10.min(...)` (`:546`); `FittedTruncatedSVD<F>` (`:107`) exposes no `n_features_in_`; the PyO3 binding (REQ-15) omits fitted-attr getters / `inverse_transform` / `random_state`. |
| REQ-18 (ferray substrate) | NOT-STARTED | open prereq blocker **#1560**. `truncated_svd.rs` computes on `ndarray::{Array1, Array2}` (`truncated_svd.rs:44`), samples via `rand`/`rand_distr` `StandardNormal` (`:46-47,556`), and uses hand-rolled `qr_decomposition` (`:175`) + `jacobi_eigen_internal` (`:406`), not `ferray-core` arrays / `ferray::linalg` / `ferray::random` (R-SUBSTRATE-1/2). |

## Architecture

`truncated_svd.rs` follows the unfitted/fitted split (CLAUDE.md naming): `TruncatedSVD<F>
{ n_components, random_state }` (`truncated_svd.rs:59`; `new(n_components)` `:70`,
`with_random_state(seed)` `:80`, accessors `n_components()` `:87` / `random_state()`
`:93`) → `Fit<Array2<F>, ()>` → `FittedTruncatedSVD<F> { components_, singular_values_,
explained_variance_, explained_variance_ratio_ }` (`truncated_svd.rs:107`, accessors
`components()`/`singular_values()`/`explained_variance()`/`explained_variance_ratio()`
`:122-145`, `inverse_transform` `:156`). The path is generic over `F: Float + Send +
Sync + 'static` (both f32 and f64, `test_truncated_svd_f32`);
`fit`/`transform`/`inverse_transform` return `Result<_, FerroError>` (R-CODE-2).

**Fit path (`pub fn fit in truncated_svd.rs`, `:519`) — REQ-1/2/3/4/5/6/7/8/11/13.**
Validates `n_components != 0`, `n_components <= min(n_samples, n_features)`,
`n_samples >= 1` (`truncated_svd.rs:522-544`) — REQ-11. Randomized SVD (Halko 2011): seed
a `StdRng` from `random_state.unwrap_or(0)` (`truncated_svd.rs:551-554` — REQ-13), draw a
Gaussian sketch `Omega` of shape `(n_features, n_random)` with `n_random = n_components +
min(10, n_features − n_components)` (`truncated_svd.rs:546-561`), form `Y = X·Omega`
(`:564`), QR via modified Gram-Schmidt (`qr_decomposition` `:175`), `B = QᵀX` (`:570`),
`svd_via_eigendecomp(B)` (`:573` — Jacobi eigendecomposition of `BBᵀ`/`BᵀB` `:305-402`,
`jacobi_eigen_internal` `:406`), take the top `n_components` singular values + Vt rows
(`:576-589`). **This is sklearn's `randomized_svd` skeleton EXCEPT it does NO power
iterations (REQ-4; sklearn `n_iter=5`), so it relies on the test fixtures being
well-conditioned (the re-audit confirmed ~8-sig-fig top-k agreement with the true SVD).**

The fit then ADDS the three sklearn fitted-attribute steps that the critic→fixer cycle
landed:
- **FIX-A — `svd_flip(u_based_decision=False)` (`:591-613`, was `#1556`, REQ-1).** For
  each `components_` row, scan columns for the FIRST max-abs entry (initialize `j_max=0`,
  update only on strict `>`, `:599-607` — numpy `argmax` tie-break), and if that entry is
  negative, negate the whole row (`:608-612`) so its max-abs entry is positive. This pins
  the otherwise-arbitrary Jacobi eigenvector signs deterministically, mirroring
  `_truncated_svd.py:255` / `extmath.py:897-905` (same convention as `pca.rs`).
- **FIX-B — CENTERED `explained_variance_` (`:617-635`, was `#1554`, REQ-2).** Form
  `x_transformed = x.dot(&components.t())` (`:623`, the FINAL flipped components) and
  compute the CENTERED population variance `sum_sq/n − mean²` per transformed column
  (`:624-635`), matching numpy `np.var(X_transformed, axis=0)` (`_truncated_svd.py:264,269`).
- **FIX-C — CENTERED ratio denominator (`:637-657`, was `#1555`, REQ-3).** Compute
  `full_var = Σ_j var(X[:,j])` (CENTERED per-feature variance summed, `:640-651`) and
  `explained_variance_ratio = explained_variance / full_var` (`:653-657`, guarded for
  `full_var > 0`), matching `np.var(X, axis=0).sum()` (`_truncated_svd.py:274-275`).

`singular_values_` are the raw randomized-SVD σ (`:580-583`), UNCHANGED by the fix.

**Transform (`impl Transform for FittedTruncatedSVD`, `pub fn transform in
truncated_svd.rs` `:680`) — REQ-10/sign-of-1.** Validates the column count (`:682-688`)
then projects `X·componentsᵀ` (`:689`) — algebraically sklearn's `X @ components_.T`
(`_truncated_svd.py:295`), NO centering (REQ-8). The output column signs now inherit
REQ-1's DETERMINISTIC svd_flip sign (sklearn-matching), UNCHANGED otherwise.

**Inverse-transform (`truncated_svd.rs:156`) — REQ-10.** `X_reduced·components_` (`:165`)
= sklearn `X @ components_` (`_truncated_svd.py:313`), NO mean term. Unlike PCA this is
NOT an exact round-trip — the rank truncation discards the complementary subspace.
UNCHANGED by the fix.

**sklearn (target contract).** `class TruncatedSVD(ClassNamePrefixFeaturesOutMixin,
TransformerMixin, BaseEstimator)` (`_truncated_svd.py:30`) takes
`__init__(n_components=2, *, algorithm="randomized", n_iter=5, n_oversamples=10,
power_iteration_normalizer="auto", random_state=None, tol=0.0)` (`:173-185`).
`fit_transform` (`:212-278`) validates with `accept_sparse=["csr","csc"]` +
`ensure_min_features=2` (`:228`), then either the `arpack` branch (`svds` + reverse +
`svd_flip` `:233-238`) or the `randomized` branch (`n_components > n_features` ValueError
`:241-245`, `randomized_svd(n_iter=5, n_oversamples=10, power_iteration_normalizer,
flip_sign=False)` `:246-254`, `svd_flip(U, VT, u_based_decision=False)` `:255`); sets
`components_ = VT` (`:257`), `X_transformed = X @ components_.T` or `U*Sigma` (`:264-266`),
`explained_variance_ = np.var(X_transformed, axis=0)` (`:269`, CENTERED),
`explained_variance_ratio_ = exp_var / np.var(X,axis=0).sum()` (`:274-275`, CENTERED denom),
`singular_values_ = Sigma` (`:276`). `transform` (`:295`) / `inverse_transform` (`:313`)
are uncentered matrix products; `_n_features_out = components_.shape[0]` (`:319-321`).

**Where ferrolearn now stands.** With FIX-A/B/C landed, ferrolearn matches sklearn
`TruncatedSVD(algorithm='arpack')` on the full fitted-attribute surface
(`components_` incl. sign ~1e-12, `explained_variance_` ~1e-9,
`explained_variance_ratio_` ~1e-11, `singular_values_` exactly) on well-conditioned
fixtures. The remaining gaps are NOT value-parity on this regime: the `n_iter` power
iterations for slow-decay-spectrum robustness (REQ-4, `#1557`), the inherent
numpy-RNG-vs-StdRng bit-parity carve-out for the seeded RANDOMIZED solver (REQ-16, `#1558`),
the `arpack` solver path + `n_iter`/`n_oversamples`/`power_iteration_normalizer`/`tol`
ctor params + `n_features_in_` + the PyO3 getter/`inverse_transform`/`random_state`
binding gap (REQ-17, `#1559`), and the ferray substrate (REQ-18, `#1560`). This is now a
**VALUE-PARITY-SHIPPED** unit (14 SHIPPED / 4 NOT-STARTED).

## Verification

Library crate (green at baseline `284985ae`):
```bash
cargo test -p ferrolearn-decomp truncated_svd                  # in-module #[test]s + doctest
cargo test -p ferrolearn-decomp --test divergence_truncated_svd # 3 divergence pins + 8 green-guards + DIV-4 carve-out
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin the STRUCTURAL contract:
`test_truncated_svd_correct_dimensions`, `test_truncated_svd_single_component` (REQ-5);
`test_truncated_svd_singular_values_positive`,
`test_truncated_svd_singular_values_sorted_descending` (REQ-6);
`test_truncated_svd_components_unit_length` (REQ-7); `test_truncated_svd_no_centering`
(REQ-8); `test_truncated_svd_explained_variance_ratio_le_1` (REQ-9);
`test_truncated_svd_dimensionality_reduction`, `test_truncated_svd_shape_mismatch`
(REQ-10); `test_truncated_svd_invalid_n_components_zero`,
`test_truncated_svd_invalid_n_components_too_large`,
`test_truncated_svd_n_components_getter` (REQ-11); `test_truncated_svd_f32` (REQ-12);
`test_truncated_svd_random_state_reproducibility`,
`test_truncated_svd_random_state_getter` (REQ-13);
`test_truncated_svd_pipeline_integration` (REQ-14); plus the module doctest.

The `tests/divergence_truncated_svd.rs` suite (now LIVE/GREEN) pins the VALUE parity and
the structural guards:
- `divergence_components_max_abs_positive_svd_flip` (REQ-1) — `components_` matches
  `TruncatedSVD(algorithm='arpack')` element-wise INCL. SIGN ~1e-12 (multi-component
  independent flips exercised).
- `divergence_explained_variance_centered` (REQ-2) — matches the CENTERED arpack oracle
  ~1e-9.
- `divergence_explained_variance_ratio_centered_denominator` (REQ-3) — matches the
  CENTERED-denominator arpack oracle ~1e-11.
- 8 structural green-guards:
  `green_components_shape_and_singular_values_descending_nonneg` (REQ-5/6),
  `green_components_rows_unit_norm` (REQ-7),
  `green_transform_equals_x_dot_components_t_no_centering` (REQ-8/10),
  `green_inverse_transform_shape_and_algebra` (REQ-10),
  `green_explained_variance_ratio_sums_le_1` (REQ-9, verified over 200 random fixtures),
  `green_error_contracts` (REQ-11), `green_determinism_same_seed` (REQ-13),
  `green_f32_fits_without_panic` (REQ-12).
- `div4_carveout_no_power_iter_singular_values_match_true_svd` (REQ-4/16 carve-out) —
  asserts the top-k singular values match the TRUE SVD (NOT sklearn's seeded randomized
  output), pinning the no-power-iteration accuracy on well-conditioned data.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the deterministic value oracle
uses `algorithm='arpack'` (true SVD):
```bash
# REQ-1/2/3 value parity (fresh 6x4 uncentered fixture, distinct singular values).
python3 -c "import numpy as np; from sklearn.decomposition import TruncatedSVD
X=np.array([[10.,2.,30.,4.],[12.,3.,28.,5.],[9.,1.,33.,6.],[11.,4.,31.,3.],[8.,2.,29.,7.],[13.,5.,32.,2.]])
m=TruncatedSVD(n_components=3, algorithm='arpack', random_state=0).fit(X)
print('components_:', [np.round(r,8).tolist() for r in m.components_])
print('explained_variance_:', np.round(m.explained_variance_,8).tolist())
print('explained_variance_ratio_:', np.round(m.explained_variance_ratio_,8).tolist())
print('singular_values_:', np.round(m.singular_values_,8).tolist())
print('argmax-abs idx:', [int(np.argmax(np.abs(r))) for r in m.components_])"
# -> components_: [[0.32119784, 0.08700819, 0.93314719, 0.1360068], [-0.57337707, -0.4737153, 0.14647155, 0.65221058], [0.62064358, 0.11586761, -0.32692724, 0.70320326]]
# -> explained_variance_: [2.58968722, 6.49966711, 1.13697247]
# -> explained_variance_ratio_: [0.24533879, 0.61575794, 0.10771318]
# -> singular_values_: [80.17597217, 6.24559431, 2.61353831]
# -> argmax-abs idx: [2, 3, 3]  (each component-row max-abs entry positive)
```

ferrolearn-python (REQ-15 binding, present at baseline):
```bash
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -c "import ferrolearn, numpy as np
m=ferrolearn._RsTruncatedSVD(2); m.fit(np.random.rand(5,3)); print(m.transform(np.random.rand(2,3)).shape)"
```
`_RsTruncatedSVD` (`extras.rs:1101`, `lib.rs:73`, via `py_transformer!`) exposes
`(n_components)` ctor + `fit` + `transform` ONLY — its `transform` output now inherits
REQ-1's DETERMINISTIC svd_flip sign; the fitted-attribute getters, `inverse_transform`,
and `random_state` ctor param are the REQ-17 binding gap.

## Blockers

(#1553 is this doc's crosslink tracking issue. The 4 open items below are NOT-STARTED;
none is a value-parity blocker on well-conditioned data — the three pre-fix value
divergences #1556/#1554/#1555 are FIXED. This doc is markdown only and does not file
issues.)

- **#1557** — REQ-4 (FIXABLE): add the `n_iter` power iterations sklearn passes to
  `randomized_svd` (`_truncated_svd.py:249`, `n_iter=5`) to ferrolearn's sketch
  (`truncated_svd.rs:564`) for slow-decay-spectrum robustness. The re-audit confirmed
  top-k accuracy is sufficient on well-conditioned fixtures (~8 sig figs vs the true SVD),
  so this is a robustness margin, not a value-parity blocker.
- **#1558** — REQ-16 (CARVE-OUT, R-DEFER-3): bit-exact parity with sklearn's seeded
  RANDOMIZED output is unreachable (numpy `RandomState` ≠ Rust `StdRng`,
  `truncated_svd.rs:551` vs `_truncated_svd.py:229`); parity is asserted against the
  identifiable true SVD instead (`div4_carveout_no_power_iter_singular_values_match_true_svd`).
  No failing test (inherent RNG difference).
- **#1559** — REQ-17: add the `algorithm` field + the `"arpack"` ARPACK `svds` path
  (`_truncated_svd.py:233-238`), the `n_iter`/`n_oversamples`/`power_iteration_normalizer`/
  `tol` ctor params (`:173-185`), the `n_features_in_` fitted attr (`:109-110`), and the
  missing PyO3 getters / `inverse_transform` / `random_state` ctor param on
  `_RsTruncatedSVD` (`extras.rs:1101`).
- **#1560** — REQ-18: migrate `truncated_svd.rs` off `ndarray` + `rand`/`rand_distr` +
  the hand-rolled `qr_decomposition` (`:175`) / `jacobi_eigen_internal` (`:406`) to
  `ferray-core` arrays / `ferray::linalg` / `ferray::random` (R-SUBSTRATE).
