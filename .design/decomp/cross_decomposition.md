# Cross-decomposition (sklearn.cross_decomposition: PLSRegression / PLSCanonical / CCA / PLSSVD)

<!--
tier: 3-component
status: value-parity-shipped
baseline-commit: 1411f352
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/cross_decomposition/_pls.py  # _get_first_singular_vectors_power_method (:59-115): mode-A seed = FIRST non-constant Y column (:71-74); mode-A weights w = Xᵀu/(uᵀu) (:91), q = Yᵀt/(tᵀt) (:99); mode-B (CCA) X_pinv/Y_pinv precomputed (:78-85), x_weights = X_pinv·y_score (:88-89), y_weights = Y_pinv·x_score (:96-97); convergence ‖Δx_weights‖²<tol (:106-109) + Y.shape[1]==1 single-iter break (:107). _get_first_singular_vectors_svd (:118-125). _center_scale_xy (:128-152): centre X,Y; if scale x_std = X.std(axis=0, ddof=1) (:142), zero-std → 1.0 (:143), X /= x_std (:144); y_std ddof=1 (:145-147). _svd_flip_1d(u, v) (:154-161): idx = argmax(abs(u)) (:158), sign = sign(u[idx]) (:159), u *= sign; v *= sign (:160-161) — per-component sign convention applied after weights finalised, BEFORE scores. svd_flip(U, Vt) (:1105, u_based_decision) for PLSSVD. _PLS base (:186-555): ctor (:215-228) n_components=2, scale=True, deflation_mode="regression", mode="A", algorithm="nipals", max_iter=500, tol=1e-6. fit (:237-403): rank_upper_bound = p (regression) else min(n,p,q) (:294-299); norm_y_weights = deflation_mode=="canonical" (:301); _center_scale_xy (:305); per-component weights then _svd_flip_1d (:354); x_scores = Xk @ x_weights (:357); x_loadings (:365), Xk -= outer(x_scores, x_loadings) (:366); canonical deflate Yk on y_score (:368-371); regression deflate Yk on x_score (:372-375); x_rotations_ = x_weights_ @ pinv(x_loadings_.T @ x_weights_) (:391-393); coef_ = x_rotations_ @ y_loadings_.T (:399), coef_ = (coef_ * y_std).T / x_std (:400); intercept_ = y_mean (:401). transform (:405-450): (X − x_mean)/x_std @ x_rotations_ (:435-438). inverse_transform (:452-503): X @ x_loadings_.T then * x_std + x_mean (:489-492). predict (:505-532): X -= x_mean (NO /x_std, :530); Ypred = X @ coef_.T + intercept_ (:531). PLSRegression (deflation_mode='regression', mode='A'), PLSCanonical (deflation_mode='canonical', mode='A'), CCA (deflation_mode='canonical', mode='B').
  - sklearn/cross_decomposition/_pls.py  # PLSSVD class: fit computes C = X.T @ Y (after _center_scale_xy), U, s, Vt = svd(C, full_matrices=False), svd_flip(U, Vt) (:1105), x_weights_ = U[:, :n_components], y_weights_ = Vt.T[:, :n_components]; transform = (X − x_mean)/x_std @ x_weights_.
ferrolearn-module: ferrolearn-decomp/src/cross_decomposition.rs
parity-ops: PLSRegression, PLSCanonical, CCA, PLSSVD
crosslink-issue: 1618
-->

## Summary

`ferrolearn-decomp/src/cross_decomposition.rs` now has **FULL element-wise value
parity (including sign) with scikit-learn 1.5.2 `sklearn/cross_decomposition/_pls.py`
across ALL FOUR estimators** — `PLSRegression`, `PLSCanonical`, `CCA`, and `PLSSVD`.
A critic→fixer→re-audit cycle (closing #1619/#1620/#1621/#1622) landed the
`_svd_flip_1d` / `svd_flip` per-component sign convention, the CCA `mode='B'`
pseudo-inverse weight path, and the unified mode-A/mode-B NIPALS seed + convergence
criterion. A re-audit cross-check on FRESH fixtures (9×4 X / 9×3 Y, 5 seeds) confirms
all four estimators match the live sklearn 1.5.2 oracle (DEFAULT tol) to machine
epsilon: PLSRegression `x_weights_` 5.6e-16 + `predict` 8.9e-16; PLSCanonical (mode-A)
`x_weights_` 5.6e-16 + `x_scores_` 1.2e-15; CCA (mode-B) `x_weights_` 7.8e-16; PLSSVD
`x_weights_` 1.1e-15. All divergence tests in
`ferrolearn-decomp/tests/divergence_cross_decomposition.rs` are live and green (0
ignored crate-wide).

The exposed surface is four unfitted structs (`PLSRegression<F>`, `PLSCanonical<F>`,
`CCA<F>`, `PLSSVD<F>`) and their `Fitted*` counterparts (`FittedPLSRegression<F>`,
`FittedPLSCanonical<F>`, `FittedCCA<F>`, `FittedPLSSVD<F>`), all re-exported at the
crate root (`pub use cross_decomposition::{CCA, FittedCCA, FittedPLSCanonical,
FittedPLSRegression, FittedPLSSVD, PLSCanonical, PLSRegression, PLSSVD}`,
`lib.rs:79-81`). The three NIPALS estimators funnel through one `fn nipals` kernel
mirroring `_get_first_singular_vectors_power_method` (`_pls.py:59-115`), parameterised
by an internal `NipalsMode` (`Regression`/`Canonical`), `ScoreNorm`
(`None`/`UnitVariance`), and `WeightMode` (`A`/`B`). All four share `fn centre_scale` /
`fn apply_centre_scale`.

The remaining gaps are **feature-surface only** (NOT value divergences): the public
`x_rotations_`/`y_rotations_`/`intercept_` attribute accessors (#1623; the raw-space
`coef_` accessor itself is now SHIPPED via `coefficients()`, #2414), the
`deflation_mode`/`mode`/`algorithm='svd'` constructor params (#1624), the
`inverse_transform` API (#1625), the PyO3 binding for the four estimators (#1626), and
the ferray substrate (#1627). There is **NO PyO3 binding** for any cross-decomposition
estimator (#1626).

Count summary: **17 SHIPPED / 5 NOT-STARTED** (see `## REQ status`). The raw-space
`coef_` orientation (REQ-17a, #2414) and the PLSRegression `n_components = n_features`
bound (REQ-4, #2415) now match sklearn.

## Probes (live sklearn oracle, 1.5.2, DEFAULT tol, run from /tmp)

All expected values generated by sklearn (R-CHAR-3), never copied from ferrolearn.
Fresh fixture (9×4 X / 9×3 Y) distinct from the in-module test fixtures.

```bash
# PROBE 1 (REQ-9/10 — PLSRegression predict + x_weights_, sign included).
python3 -c "
import numpy as np
from sklearn.cross_decomposition import PLSRegression
X=np.array([[1.,2.,3.,1.],[4.,5.,7.,2.],[7.,9.,8.,1.],[10.,11.,14.,3.],
            [13.,15.,16.,2.],[2.,1.,5.,4.],[6.,3.,2.,1.],[8.,12.,9.,5.],[3.,7.,11.,2.]])
Y=np.array([[1.,0.5,2.],[2.,1.2,1.],[3.,1.4,3.],[4.,2.1,2.],[5.,2.6,4.],
            [1.5,0.9,1.],[2.2,1.1,3.],[3.3,1.8,2.],[4.1,2.4,5.]])
m=PLSRegression(n_components=2).fit(X,Y)
print('predict row0:', np.round(m.predict(X)[0],8).tolist())
print('x_weights_ col0:', np.round(m.x_weights_[:,0],8).tolist())"
# -> predict row0: [1.50429937, 0.81156907, 2.40767472]
# -> x_weights_ col0: [0.50690848, 0.59012176, 0.62598241, 0.05427821]
#   => col0 max-abs entry (idx 2, +0.62598241) is POSITIVE => _svd_flip_1d applied.
#      ferrolearn matches predict (8.9e-16) and x_weights_ incl. sign (5.6e-16).

# PROBE 2 (REQ-11/14 — PLSCanonical mode-A x_weights_ + NON-unit score std).
python3 -c "
import numpy as np
from sklearn.cross_decomposition import PLSCanonical
X=np.array([[1.,2.,3.,1.],[4.,5.,7.,2.],[7.,9.,8.,1.],[10.,11.,14.,3.],
            [13.,15.,16.,2.],[2.,1.,5.,4.],[6.,3.,2.,1.],[8.,12.,9.,5.],[3.,7.,11.,2.]])
Y=np.array([[1.,0.5,2.],[2.,1.2,1.],[3.,1.4,3.],[4.,2.1,2.],[5.,2.6,4.],
            [1.5,0.9,1.],[2.2,1.1,3.],[3.3,1.8,2.],[4.1,2.4,5.]])
pc=PLSCanonical(n_components=2).fit(X,Y); t=pc.transform(X)
print('x_weights_ col0:', np.round(pc.x_weights_[:,0],8).tolist())
print('x_score std ddof1:', np.round(t.std(axis=0,ddof=1),6).tolist())"
# -> x_weights_ col0: [0.50690848, 0.59012176, 0.62598241, 0.05427821]
# -> x_score std ddof1: [1.641662, 0.962765]
#   => PLSCanonical scores are NOT unit-variance (std 1.64, 0.96). sklearn does NOT
#      rescale scores; ferrolearn's spurious unit-variance rescaling was REMOVED (#1622).

# PROBE 3 (REQ-13 — CCA mode='B' pseudo-inverse x_weights_ + NON-unit score std).
python3 -c "
import numpy as np
from sklearn.cross_decomposition import CCA
X=np.array([[1.,2.,3.,1.],[4.,5.,7.,2.],[7.,9.,8.,1.],[10.,11.,14.,3.],
            [13.,15.,16.,2.],[2.,1.,5.,4.],[6.,3.,2.,1.],[8.,12.,9.,5.],[3.,7.,11.,2.]])
Y=np.array([[1.,0.5,2.],[2.,1.2,1.],[3.,1.4,3.],[4.,2.1,2.],[5.,2.6,4.],
            [1.5,0.9,1.],[2.2,1.1,3.],[3.3,1.8,2.],[4.1,2.4,5.]])
c=CCA(n_components=2).fit(X,Y); t=c.transform(X)
print('x_weights_ col0:', np.round(c.x_weights_[:,0],8).tolist())
print('x_score std ddof1:', np.round(t.std(axis=0,ddof=1),6).tolist())"
# -> x_weights_ col0: [0.70888014, 0.08291248, 0.68747389, 0.13414216]
# -> x_score std ddof1: [1.41809, 0.655794]
#   => CCA mode-B weights DIFFER from the mode-A PLSCanonical weights above (different
#      structure); scores are NON-unit (1.418, 0.656) — the spurious rescaling was
#      removed. ferrolearn (WeightMode::B + pinv) matches incl. sign at 7.8e-16.

# PROBE 4 (REQ-15/16 — PLSSVD x_weights_ via svd_flip).
python3 -c "
import numpy as np
from sklearn.cross_decomposition import PLSSVD
X=np.array([[1.,2.,3.,1.],[4.,5.,7.,2.],[7.,9.,8.,1.],[10.,11.,14.,3.],
            [13.,15.,16.,2.],[2.,1.,5.,4.],[6.,3.,2.,1.],[8.,12.,9.,5.],[3.,7.,11.,2.]])
Y=np.array([[1.,0.5,2.],[2.,1.2,1.],[3.,1.4,3.],[4.,2.1,2.],[5.,2.6,4.],
            [1.5,0.9,1.],[2.2,1.1,3.],[3.3,1.8,2.],[4.1,2.4,5.]])
s=PLSSVD(n_components=2).fit(X,Y)
print('x_weights_ col0:', np.round(s.x_weights_[:,0],8).tolist())"
# -> x_weights_ col0: [0.50690833, 0.59012195, 0.62598255, 0.05427592]
#   => U[:, :nc] of SVD(XᵀY) after svd_flip(U, Vt); col0 max-abs entry POSITIVE.
#      ferrolearn matches incl. sign at 1.1e-15.

# PROBE 5 (REQ-1/2/3 — ddof=1 scaling + ctor defaults).
python3 -c "
import numpy as np
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD
X=np.array([[1.,2.,3.,1.],[4.,5.,7.,2.],[7.,9.,8.,1.],[10.,11.,14.,3.],[13.,15.,16.,2.]])
print('std X col0 ddof1:', round(float(X[:,0].std(ddof=1)),6), 'ddof0:', round(float(X[:,0].std(ddof=0)),6))
for M in (PLSRegression, PLSCanonical, CCA):
    m=M()
    print(M.__name__,'n_components',m.n_components,'scale',m.scale,'max_iter',m.max_iter,'tol',m.tol,
          'deflation_mode',m.deflation_mode,'mode',m.mode,'algorithm',m.algorithm)
print('PLSSVD n_components',PLSSVD().n_components,'scale',PLSSVD().scale)"
# -> std X col0 ddof1: 4.743416 ddof0: 4.242641
# -> PLSRegression n_components 2 scale True max_iter 500 tol 1e-06 deflation_mode regression mode A algorithm nipals
# -> PLSCanonical  n_components 2 scale True max_iter 500 tol 1e-06 deflation_mode canonical  mode A algorithm nipals
# -> CCA           n_components 2 scale True max_iter 500 tol 1e-06 deflation_mode canonical  mode B algorithm nipals
# -> PLSSVD n_components 2 scale True
#   => ddof=1 (Bessel); ferrolearn centre_scale divides by n-1 (REQ-1 MATCH). ctor
#      defaults scale=True, max_iter=500, tol=1e-6 match (REQ-2/3). deflation_mode/mode/
#      algorithm are sklearn-internal; ferrolearn hard-wires them per struct (REQ-18, #1624).
```

## Requirements

- REQ-1: **`ddof=1` (Bessel) centring + scaling shared by all four estimators
  (SHIPPED).** sklearn `_center_scale_xy` (`_pls.py:128-152`) centres X,Y then, when
  `scale`, divides by `X.std(axis=0, ddof=1)` (`:142`) / `Y.std(axis=0, ddof=1)`
  (`:145`), replacing zero stds with `1.0` (`:143,146`). `fn centre_scale` centres then
  computes `var = Σ(centred)² / n_minus_1` with `n_minus_1 = max(n-1, 1)` — exactly
  `ddof=1` — `sqrt`s it, and replaces near-zero std with `F::one()`. `fn
  apply_centre_scale` applies the stored mean/std to new data.

- REQ-2: **`scale=True` / `scale=False` toggle (SHIPPED).** sklearn `scale=True`
  default (`_pls.py:219`); the `else` branch sets `x_std = ones` (`:149-150`). All four
  ferrolearn ctors default `scale: true` with a `with_scale(bool)` builder;
  `scale=false` makes `centre_scale` return `std = None` so `apply_centre_scale` only
  centres.

- REQ-3: **Constructor defaults `max_iter=500` / `tol=1e-6` / `scale=true` (SHIPPED).**
  sklearn `_PLS.__init__` (`_pls.py:215-228`): `max_iter=500`, `tol=1e-6` (Probe 5). The
  three NIPALS ctors (`PLSRegression::new`/`PLSCanonical::new`/`CCA::new`) set
  `max_iter: 500`, `tol: 1e-6`, `scale: true` with `with_max_iter`/`with_tol`/
  `with_scale` builders. **FLAG (structural):** sklearn's `n_components` defaults to `2`;
  ferrolearn requires `new(n_components: usize)` explicitly (no default).

- REQ-4: **Error / parameter contracts (SHIPPED, scoped).** Every `fit` rejects
  `n_samples_x != n_samples_y` with `ShapeMismatch`, `n_components == 0` with
  `InvalidParameter`, `n_components > max_components` with `InvalidParameter`, and
  `n_samples < 2` with `InsufficientSamples`; every `transform`/`predict` returns
  `ShapeMismatch` on a column mismatch (via `apply_centre_scale`). **FLAG (divergent
  surface, not value):** sklearn validates via `_parameter_constraints`
  (`_pls.py:203-212`) raising `ValueError`, NOT `FerroError`; sklearn does not pre-reject
  `n_samples < 2`. PLSRegression's regression-mode `n_components` upper bound is now
  `rank_upper_bound = p` (= `n_features_x`) ALONE, matching sklearn (`:294`), fixed via
  #2415 (SHIPPED); PLSCanonical/CCA keep `min(n, p, q)`.

- REQ-5: **f32 / f64 generic support (SHIPPED).** All four estimators are generic over
  `F: Float + Send + Sync + 'static`; `svd_dispatch` routes f64 → faer
  `NdarrayFaerBackend::svd`, f32 → f64-roundtrip, other → `svd_via_eigen` Jacobi
  fallback; the NIPALS arithmetic (`fn nipals`) and `fn pinv` are fully generic. sklearn
  validates `dtype=FLOAT_DTYPES`.

- REQ-6: **Fitted shapes (`x_weights_`/`x_loadings_`/`y_loadings_`/`x_scores_`/
  `y_scores_`) (SHIPPED).** sklearn allocates `x_weights_ (p, nc)`, `y_loadings_
  (q, nc)`, `x_loadings_ (p, nc)`, scores `(n, nc)` (`_pls.py:309-314`). `NipalsResult`
  and the `Fitted*` structs allocate the same shapes; PLSSVD stores `x_weights_ (p, nc)`
  / `y_weights_ (q, nc)`.

- REQ-7: **`NipalsMode` deflation distinction (regression vs canonical) (SHIPPED).**
  sklearn deflates X by `Xk -= outer(x_scores, x_loadings)` always (`_pls.py:366`), then
  Y by `outer(x_scores, y_loadings)` for regression (`:372-375`) vs `outer(y_scores,
  y_loadings)` for canonical (`:368-371`). `fn nipals` deflates X identically and
  branches Y on `NipalsMode::Regression` (`Y -= t qᵀ`) vs `Canonical` (`Y -= u cᵀ`);
  PLSRegression passes `Regression`, PLSCanonical/CCA pass `Canonical`.

- REQ-8: **NIPALS seed + convergence criterion matching sklearn (SHIPPED — fixed via
  #1622).** sklearn seeds the power method from the FIRST non-constant Y column
  (`_pls.py:71-74`) and stops on `‖Δx_weights‖² < tol` (`:106-109`), with an immediate
  single-iteration break when `Y.shape[1] == 1` (`:107`). `fn nipals` seeds from the
  first Y column with any `|entry| > eps` and checks convergence on the normalised
  x-weights `‖w − w_old‖² < tol`, breaking immediately when `n_features_y == 1` — unified
  for BOTH mode-A and mode-B. Defaults `max_iter=500`, `tol=1e-6` match sklearn. This
  unification is what makes value parity hold at DEFAULT tol.

- REQ-9: **`PLSRegression::predict` / raw-space `coef_` element-wise value parity
  (SHIPPED).** sklearn `predict` (`_pls.py:530-531`) centres X (NO `/x_std`) then
  `Ypred = X @ coef_.T + intercept_`, with `coef_ = (x_rotations_ @ y_loadings_.T *
  y_std).T / x_std` (`:399-400`). `PLSRegression::fit` builds the internal
  `B = W(PᵀW)⁻¹Qᵀ` then lifts it to the raw-space `coef_ = (B * y_std).T / x_std`
  (`(n_targets, n_features)`, #2414); `FittedPLSRegression::predict` now centres X ONLY
  (no `/x_std`) and computes `X_centred @ coef_.T + y_mean` — a direct transcription of
  sklearn. Re-audit on the fresh fixture matches `predict` at 8.9e-16 (Probe 1: row0
  `[1.50429937, 0.81156907, 2.40767472]`); the raw-space refactor reproduces the prior
  prediction bit-for-bit (live oracle row0 `[1.0001724645730092, 0.5393899142133716]`).

- REQ-10: **`x_weights_`/`x_loadings_`/`y_loadings_`/scores value parity incl. sign for
  the NIPALS estimators (SHIPPED — fixed via #1620/#1622).** sklearn applies
  `_svd_flip_1d(x_weights, y_weights)` per component (`_pls.py:354`, def `:154-161`):
  `idx = argmax(abs(x_weights))`, `sign = sign(x_weights[idx])`, both weights `*= sign`,
  pinning the otherwise-arbitrary NIPALS sign so each component's max-abs `x_weight`
  entry is POSITIVE. `fn nipals` applies the identical `argmax(abs(w))`-based flip to
  `w_final` after the weights are finalised and BEFORE the scores/loadings are derived,
  so the sign propagates consistently to `t`, `p`, `q`, `u`. Re-audit: PLSRegression /
  PLSCanonical `x_weights_` match incl. sign at 5.6e-16, scores at 1.2e-15 (Probes 1-2).

- REQ-11: **`transform` value parity via the rotation (SHIPPED).** sklearn `transform`
  projects `(X − x_mean)/x_std @ x_rotations_` (`_pls.py:435-438`), with `x_rotations_ =
  x_weights_ @ pinv(x_loadings_.T @ x_weights_)` (`:391-393`). `FittedPLSRegression`/
  `FittedPLSCanonical`/`FittedCCA::transform` centre+scale X then multiply by `rotation =
  x_weights_ · (x_loadings_ᵀ x_weights_)⁻¹` (the same `W(PᵀW)⁻¹` rotation, recomputed from
  the stored weights/loadings). With REQ-10's sign convention and the matched weights/
  loadings, the transform output matches sklearn element-wise (Probe 2 PLSCanonical score
  std `[1.641662, 0.962765]` reproduced).

- REQ-12: **`transform_y` accessor (SHIPPED, scoped).** ferrolearn exposes
  `transform_y(y)` on `FittedPLSSVD` (`yc @ y_weights_` — matches sklearn PLSSVD's
  y-projection), `FittedPLSCanonical` and `FittedCCA` (`yc @ y_loadings_`). **FLAG:** the
  canonical/CCA `transform_y` projects Y via `y_loadings_`, whereas sklearn's full
  `transform(Y=…)` uses `y_rotations_` (`_pls.py:447`); the full `y_rotations_`-based
  Y-projection is a feature-surface gap tracked under #1623 (the public-attr blocker).

- REQ-13: **CCA `mode='B'` pseudo-inverse weights value parity (SHIPPED — fixed via
  #1619).** sklearn `CCA` is `mode='B'`: `x_weights = X_pinv @ y_score` (`_pls.py:88-89`),
  `y_weights = Y_pinv @ x_score` (`:96-97`) with the pseudo-inverses precomputed once per
  component (`:78-85`). `fn nipals` now branches on a `WeightMode` enum: `WeightMode::B`
  precomputes `pinv(&xk)` / `pinv(&yk)` per component (the new `fn pinv`, an SVD-based
  Moore-Penrose inverse mirroring scipy `pinv2`) and uses `x_weights = X_pinv·u`,
  `y_weights = Y_pinv·t`; CCA::fit passes `WeightMode::B`, PLSRegression/PLSCanonical pass
  `WeightMode::A`. The spurious unit-variance score-rescaling was REMOVED — sklearn does
  NOT rescale scores, and CCA's scores are NON-unit (Probe 3 std `[1.41809, 0.655794]`).
  Re-audit: CCA `x_weights_` match at 7.8e-16 (Probe 3).

- REQ-14: **PLSCanonical/CCA scores are NON-unit and match sklearn's actual std
  (SHIPPED — reframed via #1622).** The OLD doc claimed a `ScoreNorm::UnitVariance`
  rescaling that forced unit-variance scores; that rescaling was a divergence and was
  REMOVED. sklearn (`_pls.py:356-362`) stores scores WITHOUT any unit-variance rescaling.
  `fn nipals` no longer rescales `t_final`/`u_final` to unit std; `ScoreNorm::UnitVariance`
  now only normalises the y-weights `q` (the `norm_y_weights` behaviour, `_pls.py:301`).
  The resulting PLSCanonical/CCA score stds are non-unit and match sklearn (Probe 2:
  `[1.641662, 0.962765]`; Probe 3: `[1.41809, 0.655794]`).

- REQ-15: **PLSSVD `svd_flip` sign convention (SHIPPED — fixed via #1621).** sklearn
  PLSSVD applies `svd_flip(U, Vt)` (`_pls.py:1105`, `u_based_decision=True`): for each
  column of `U` force the max-abs entry positive, applying the same sign to the paired row
  of `Vt`. `PLSSVD::fit` performs the identical per-column flip on `u`/`vt` after the SVD
  of `XᵀY` and before slicing `x_weights = U[:, :nc]`, `y_weights = Vt.T[:, :nc]`, so the
  PLSSVD weight sign is pinned (Probe 4 col0 max-abs entry positive).

- REQ-16: **PLSSVD `x_weights_`/`y_weights_` value parity incl. sign (SHIPPED).** sklearn
  computes `U, s, Vt = svd(XᵀY)`, `svd_flip(U, Vt)`, `x_weights_ = U[:, :nc]`, `y_weights_
  = Vt.T[:, :nc]`. `PLSSVD::fit` mirrors this exactly: `c = xc.t().dot(&yc)`,
  `svd_dispatch(&c)`, the `svd_flip` of REQ-15, then the same slices. Re-audit: PLSSVD
  `x_weights_` match incl. sign at 1.1e-15 (Probe 4: col0 `[0.50690833, 0.59012195,
  0.62598255, 0.05427592]`).

- REQ-17: **`coef_`/`x_rotations_`/`y_rotations_`/`intercept_` public attr exposure
  (NOT-STARTED — blocker #1623).** sklearn exposes `coef_` of shape `(n_targets,
  n_features)` pre-multiplied by `y_std / x_std` (`_pls.py:399-400`), `intercept_ = y_mean`
  (`:401`), `x_rotations_` (`:391`), `y_rotations_` (`:395`). `FittedPLSRegression` exposes
  `coefficients()` — the INTERNAL pre-scale `(n_features, n_targets)` matrix, NOT sklearn's
  transposed/scaled `coef_` — and has NO `coef_`/`x_rotations_`/`y_rotations_`/`intercept_`
  accessor (the rotation is recomputed inline in `transform`). The canonical/CCA fitted
  structs expose neither. (REQ-9 confirms `predict` is nonetheless correct.)

- REQ-18: **`deflation_mode`/`mode`/`algorithm='svd'` ctor params (NOT-STARTED — blocker
  #1624).** sklearn `_PLS.__init__` exposes `deflation_mode` (`"regression"`/
  `"canonical"`), `mode` (`"A"`/`"B"`), and `algorithm` (`"nipals"`/`"svd"`)
  (`_pls.py:215-228`, `_parameter_constraints:203-212`). ferrolearn HARD-WIRES these per
  struct via the internal `NipalsMode`/`ScoreNorm`/`WeightMode` enums — there is no
  `deflation_mode`/`mode`/`algorithm` ctor field and no `algorithm="svd"` path
  (`_get_first_singular_vectors_svd`, `_pls.py:118-125`) for the NIPALS estimators.
  `n_iter_` IS exposed.

- REQ-19: **`inverse_transform` API (NOT-STARTED — blocker #1625).** sklearn `_PLS`
  exposes `inverse_transform` (`_pls.py:452-503`): `X @ x_loadings_.T * x_std + x_mean`
  (`:489-492`). NONE of the four ferrolearn fitted structs implement
  `inverse_transform`.

- REQ-20: **PyO3 binding surface (NOT-STARTED — blocker #1626).** `grep
  PLSRegression|PLSSVD|PLSCanonical|CCA ferrolearn-python/src` is EMPTY — none of the four
  cross-decomposition estimators is exposed to Python. There is no `_RsPLS*` / `_RsCCA`
  class registered. (R-DEFER-1: the boundary consumer here is the crate re-export
  `lib.rs:79-81`, grandfathered S5.)

- REQ-21: **ferray substrate (NOT-STARTED — blocker #1627).** `cross_decomposition.rs`
  computes on `ndarray::{Array1, Array2}` and uses `ferrolearn_core::backend_faer::
  NdarrayFaerBackend::svd` + hand-rolled `jacobi_eigen_symmetric` / `invert_square` /
  `pinv`, not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`, DEFAULT tol),
never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1/2/3, SHIPPED): `centre_scale(&X, true)` divides centred columns by the
  ddof=1 std (Probe 5: X col-0 std `4.743416`, not ddof0 `4.242641`); all four ctors
  default `scale=true`; the NIPALS three default `max_iter=500`, `tol=1e-6`.

- AC-2 (REQ-4, SHIPPED): every `fit` `Err`s on row mismatch, `n_components==0`,
  `n_components` too large, `n_samples<2`; every `transform`/`predict` `Err`s on column
  mismatch. FLAG: sklearn raises `ValueError`. PLSRegression's bound is now
  `rank_upper_bound = p` matching sklearn (`:294`, #2415); PLSCanonical/CCA use
  `min(n, p, q)`.

- AC-3 (REQ-5/6, SHIPPED): `<f32>` fits transform/predict for all four; fitted matrices
  have sklearn shapes.

- AC-4 (REQ-9/10, SHIPPED): `PLSRegression::new(2).fit(&X,&Y).predict(&X)` equals the
  live `PLSRegression(n_components=2).fit(X,Y).predict(X)` element-wise (re-audit 8.9e-16,
  Probe 1 row0 `[1.50429937, 0.81156907, 2.40767472]`); `x_weights_` matches incl. sign
  (5.6e-16, Probe 1 col0).

- AC-5 (REQ-11/14, SHIPPED): PLSCanonical `transform(X)` matches the live oracle, with
  NON-unit score std `[1.641662, 0.962765]` (Probe 2); the spurious unit-variance
  rescaling is gone.

- AC-6 (REQ-13, SHIPPED): CCA `x_weights_` (mode-B pseudo-inverse) match the live `CCA`
  oracle at 7.8e-16 (Probe 3 col0 `[0.70888014, 0.08291248, 0.68747389, 0.13414216]`),
  distinct from the mode-A PLSCanonical weights; CCA scores are NON-unit (Probe 3
  `[1.41809, 0.655794]`).

- AC-7 (REQ-15/16, SHIPPED): PLSSVD `x_weights_` match the live `PLSSVD` oracle incl. sign
  at 1.1e-15 (Probe 4 col0 `[0.50690833, 0.59012195, 0.62598255, 0.05427592]`).

- AC-8 (REQ-17a SHIPPED; REQ-17b/18/19/20/21 NOT-STARTED): `FittedPLSRegression::
  coefficients()` returns sklearn's scaled/transposed raw-space `coef_` `(n_targets,
  n_features)` (REQ-17a, #2414); ferrolearn still exposes no `x_rotations_` /
  `y_rotations_` / `intercept_` accessor (#1623/REQ-17b), no `deflation_mode`/`mode`/
  `algorithm='svd'` ctor params (#1624), no `inverse_transform` (#1625), no PyO3 binding
  (#1626), and computes on `ndarray` + faer/Jacobi, not ferray (#1627).

## REQ status

Binary (R-DEFER-2). The four estimators + their `Fitted*` are existing pub APIs; the
non-test consumer is the crate re-export (`lib.rs:79-81`, boundary public API,
grandfathered S5/R-DEFER-1). There is NO PyO3 binding (REQ-20). Verification = the
in-module `#[cfg(test)]` suite (structural/shape/error/f32) PLUS
`ferrolearn-decomp/tests/divergence_cross_decomposition.rs` (live element-wise parity vs
the sklearn 1.5.2 oracle, 18 tests, 0 ignored). Cites use symbol anchors (ferrolearn) /
`file:line` (sklearn 1.5.2). Live oracle = installed sklearn 1.5.2, run from `/tmp`.
#1618 is this doc's crosslink tracking issue.

A critic→fixer→re-audit cycle closing #1619 (CCA mode='B'), #1620 (NIPALS
`_svd_flip_1d` sign), #1621 (PLSSVD `svd_flip` sign), and #1622 (NIPALS seed +
convergence criterion + removal of the spurious unit-variance score rescaling) brought
all four estimators to full element-wise value parity (incl. sign) at DEFAULT tol.

Counting the rows below: REQ-1 through REQ-16 are SHIPPED (16 rows; REQ-12 is
SHIPPED-scoped) plus REQ-17a (raw-space `coef_`, #2414) is SHIPPED (17 SHIPPED rows);
REQ-17b and REQ-18 through REQ-21 are NOT-STARTED (5 rows). **Table totals: 17
SHIPPED / 5 NOT-STARTED.** #2414 (raw-space `coef_` orientation) and #2415 (regression
`n_components = n_features` bound) landed in this iteration.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (ddof=1 centring + scaling) | SHIPPED | sklearn `_center_scale_xy` (`_pls.py:128-152`) centres then `x_std = X.std(axis=0, ddof=1)` (`:142`), zero→`1.0` (`:143`); `y_std` ddof=1 (`:145`). ferrolearn `fn centre_scale` centres then `var = Σ(centred)²/n_minus_1` with `n_minus_1 = max(n-1,1)` = ddof=1, `sqrt`, near-zero std → `F::one()`; `apply_centre_scale` reuses stored mean/std. Probe 5: ddof1 `4.743416` ≠ ddof0 `4.242641`. Non-test consumers: the four `fit` impls + re-export `lib.rs:79-81`. Verification: `cargo test -p ferrolearn-decomp cross_decomposition` → `test_centre_scale_helper`, `test_centre_scale_no_scale` PASS. |
| REQ-2 (`scale=True/False` toggle) | SHIPPED | sklearn `scale=True` default (`_pls.py:219`), `else` `x_std=ones` (`:149-150`). ferrolearn ctors default `scale: true`, `with_scale(bool)` builder; `scale=false` → `centre_scale` returns `std=None`, `apply_centre_scale` only centres. Non-test consumer: re-export `lib.rs:79-81`. Verification: `test_plssvd_no_scale`, `test_plsregression_no_scale`, `test_*_builder` PASS. |
| REQ-3 (ctor defaults `max_iter=500`/`tol=1e-6`) | SHIPPED | sklearn `_PLS.__init__` (`_pls.py:215-228`) `max_iter=500`, `tol=1e-6` (Probe 5). ferrolearn NIPALS ctors (`PLSRegression::new`/`PLSCanonical::new`/`CCA::new`) set `max_iter:500`, `tol:1e-6`, `scale:true` + builders. **FLAG:** sklearn `n_components` defaults `2`; ferrolearn requires explicit `new(usize)`. Non-test consumer: re-export `lib.rs:79-81`. Verification: `test_plsregression_builder`, `test_plscanonical_builder`, `test_cca_builder`, `test_plssvd_n_components_getter` PASS. |
| REQ-4 (error / parameter contracts, scoped) | SHIPPED | every `fit` `Err`s on row mismatch (`ShapeMismatch`), `n_components==0` (`InvalidParameter`), `n_components>max` (`InvalidParameter`), `n_samples<2` (`InsufficientSamples`); `transform`/`predict` `Err` on column mismatch. The PLSRegression `n_components` upper bound is now `n_features_x = p` matching sklearn's regression-mode `rank_upper_bound = p` (`_pls.py:294`), fixed via #2415 (PLSCanonical/CCA keep `min(n,p,q)`). **FLAG (surface, not value):** sklearn validates via `_parameter_constraints` (`_pls.py:203-212`) raising `ValueError`; sklearn doesn't pre-reject `n_samples<2`. Non-test consumer: re-export `lib.rs:79-81`. Verification: `tests/divergence_pls_coef_orientation_2414.rs::divergence_pls_n_components_regression_bound` (live, green) + `test_*_invalid_zero_components`, `_too_many_components`, `_row_mismatch`, `_insufficient_samples`, `_transform_shape_mismatch` PASS. |
| REQ-5 (f32 / f64 generic) | SHIPPED | all four generic over `F: Float + Send + Sync + 'static`; `svd_dispatch` f64→faer / f32→roundtrip / other→`svd_via_eigen` Jacobi; `fn nipals` + `fn pinv` generic. sklearn `dtype=FLOAT_DTYPES`. Non-test consumer: re-export `lib.rs:79-81`. Verification: `test_plssvd_f32`, `test_plsregression_f32`, `test_plscanonical_f32`, `test_cca_f32` PASS. |
| REQ-6 (fitted shapes) | SHIPPED | sklearn allocates `x_weights_(p,nc)`/`y_loadings_(q,nc)`/`x_loadings_(p,nc)`/`scores(n,nc)` (`_pls.py:309-314`). `NipalsResult` + `Fitted*` + PLSSVD (`x_weights_(p,nc)`/`y_weights_(q,nc)`) allocate the same. Non-test consumer: re-export `lib.rs:79-81`. Verification: `test_plssvd_x_weights_shape`, `test_plscanonical_scores_shape`, `test_cca_scores_shape`, `test_plsregression_x_scores_shape`, `test_plsregression_coefficients_shape` PASS. |
| REQ-7 (deflation-mode split) | SHIPPED | sklearn deflates X always (`_pls.py:366`), Y by `outer(x_scores,y_loadings)` regression (`:372-375`) vs `outer(y_scores,y_loadings)` canonical (`:368-371`). `fn nipals` deflates X, branches Y on `NipalsMode::Regression` `Y-=tqᵀ` vs `Canonical` `Y-=ucᵀ`; PLSRegression→`Regression`, PLSCanonical/CCA→`Canonical`. Non-test consumer: the three NIPALS `fit` impls. Verification: `test_pls_regression_and_canonical_give_different_scores` PASS. |
| REQ-8 (NIPALS seed + convergence criterion) | SHIPPED | fixed via #1622. sklearn seeds the power method from the FIRST non-constant Y column (`_pls.py:71-74`), stops on `‖Δx_weights‖²<tol` (`:106-109`) + single-iter break when `Y.shape[1]==1` (`:107`). `fn nipals` seeds from the first Y column with any `|entry|>eps`, checks `‖w−w_old‖²<tol`, breaks immediately when `n_features_y==1` — unified for mode-A and mode-B; defaults `max_iter=500`/`tol=1e-6`. Non-test consumer: the three NIPALS `fit` impls + re-export `lib.rs:79-81`. Verification: the full-parity divergence suite below (default-tol parity holds because the stopping rule matches). |
| REQ-9 (`PLSRegression::predict` / raw-space `coef_` value parity) | SHIPPED | sklearn `predict` (`_pls.py:530-531`) centres X (no `/x_std`), `Ypred = X @ coef_.T + intercept_`, `coef_ = (x_rotations_@y_loadings_.T * y_std).T / x_std` (`:399-400`). `PLSRegression::fit` now builds the internal pre-scale `B = W(PᵀW)⁻¹Qᵀ` then lifts it to the raw-space `coef_ = (B * y_std).T / x_std` (shape `(n_targets, n_features)`); `FittedPLSRegression::predict` centres X ONLY (no `/x_std`) and computes `X_centred @ coef_.T + y_mean` — a direct transcription of sklearn (fixed via #2414). Verified bit-identical to the old algebra (live oracle predict row0 `[1.0001724645730092, 0.5393899142133716]` reproduced both ways). Non-test consumer: re-export `lib.rs:79-81`. Verification: `tests/divergence_cross_decomposition.rs` PLSRegression predict-parity test (live, green). |
| REQ-10 (NIPALS weights/scores value parity incl. sign) | SHIPPED | fixed via #1620/#1622. sklearn `_svd_flip_1d(x_weights, y_weights)` per component (`_pls.py:354`, def `:154-161`): `idx=argmax(abs(x_weights))`, both weights `*=sign`. `fn nipals` applies the identical `argmax(abs(w))` flip to `w_final` AFTER finalising the weights and BEFORE deriving `t`/`p`/`q`/`u`, so the sign propagates consistently. Re-audit: PLSRegression/PLSCanonical `x_weights_` 5.6e-16 incl. sign, scores 1.2e-15 (Probes 1-2). Non-test consumer: re-export `lib.rs:79-81`. Verification: `tests/divergence_cross_decomposition.rs` x_weights/x_scores parity tests (live, green). |
| REQ-11 (`transform` value parity via rotation) | SHIPPED | sklearn `transform` = `(X−x_mean)/x_std @ x_rotations_` (`_pls.py:435-438`), `x_rotations_ = x_weights_@pinv(x_loadings_.T@x_weights_)` (`:391-393`). ferrolearn `transform` centres+scales then ×`rotation = W(PᵀW)⁻¹` (recomputed from stored weights/loadings) — same formula. With REQ-10's matched weights/loadings + sign, output matches the oracle (Probe 2 PLSCanonical score std `[1.641662, 0.962765]` reproduced). Non-test consumer: re-export `lib.rs:79-81`. Verification: `tests/divergence_cross_decomposition.rs` transform-parity tests (live, green) + `test_plsregression_transform`, `test_plscanonical_basic_fit_transform`, `test_cca_basic_fit_transform`. |
| REQ-12 (`transform_y` accessor) | SHIPPED scoped | ferrolearn `transform_y` on `FittedPLSSVD` (`yc@y_weights_` — matches sklearn PLSSVD), `FittedPLSCanonical` / `FittedCCA` (`yc@y_loadings_`). **FLAG:** canonical/CCA `transform_y` use `y_loadings_`; sklearn's full `transform(Y=…)` uses `y_rotations_` (`_pls.py:447`). The full `y_rotations_`-based Y-projection is a feature-surface gap tracked under #1623. Non-test consumer: re-export `lib.rs:79-81`. Verification: `test_plssvd_transform_y`, `test_plscanonical_transform_y`, `test_cca_transform_y` PASS. |
| REQ-13 (CCA `mode='B'` pseudo-inverse weights) | SHIPPED | fixed via #1619. sklearn `CCA` is `mode='B'`: `x_weights = X_pinv@y_score` (`_pls.py:88-89`), `y_weights = Y_pinv@x_score` (`:96-97`), pinvs precomputed (`:78-85`). `fn nipals` branches on `WeightMode`: `WeightMode::B` precomputes `pinv(&xk)`/`pinv(&yk)` per component (new `fn pinv`, SVD-based Moore-Penrose ≈ scipy `pinv2`), uses `X_pinv·u`/`Y_pinv·t`; CCA::fit passes `WeightMode::B`. The spurious unit-variance score rescaling was REMOVED. Re-audit: CCA `x_weights_` 7.8e-16 (Probe 3 col0 `[0.70888014, 0.08291248, 0.68747389, 0.13414216]`). Non-test consumer: re-export `lib.rs:79-81`. Verification: `tests/divergence_cross_decomposition.rs` CCA x_weights parity test (live, green). |
| REQ-14 (PLSCanonical/CCA NON-unit scores match sklearn) | SHIPPED | reframed via #1622. The OLD "CCA unit-variance scores" REQ was a divergence: sklearn (`_pls.py:356-362`) stores scores WITHOUT unit-variance rescaling. `fn nipals` no longer rescales `t_final`/`u_final` to unit std; `ScoreNorm::UnitVariance` now only normalises the y-weights `q` (`norm_y_weights`, `_pls.py:301`). PLSCanonical scores have std `[1.641662, 0.962765]` (Probe 2), CCA `[1.41809, 0.655794]` (Probe 3) — both non-unit, matching sklearn. Non-test consumer: re-export `lib.rs:79-81`. Verification: `tests/divergence_cross_decomposition.rs` score-std parity tests (live, green). |
| REQ-15 (PLSSVD `svd_flip` sign convention) | SHIPPED | fixed via #1621. sklearn PLSSVD `svd_flip(U, Vt)` (`_pls.py:1105`, `u_based_decision`): force each column of `U`'s max-abs entry positive, apply to paired `Vt` row. `PLSSVD::fit` performs the identical per-column flip on `u`/`vt` after `svd_dispatch(&c)` and before slicing weights. Probe 4 col0 max-abs entry positive. Non-test consumer: re-export `lib.rs:79-81`. Verification: `tests/divergence_cross_decomposition.rs` PLSSVD sign test (live, green). |
| REQ-16 (PLSSVD `x_weights_`/`y_weights_` value parity incl. sign) | SHIPPED | sklearn: `U,s,Vt=svd(XᵀY)`, `svd_flip(U,Vt)`, `x_weights_=U[:,:nc]`, `y_weights_=Vt.T[:,:nc]`. `PLSSVD::fit` mirrors exactly: `c = xc.t().dot(&yc)`, `svd_dispatch`, the REQ-15 `svd_flip`, then the same slices. Re-audit: PLSSVD `x_weights_` 1.1e-15 incl. sign (Probe 4 col0 `[0.50690833, 0.59012195, 0.62598255, 0.05427592]`). Non-test consumer: re-export `lib.rs:79-81`. Verification: `tests/divergence_cross_decomposition.rs` PLSSVD x_weights parity test (live, green). |
| REQ-17a (`coef_` raw-space + orientation via `coefficients()`) | SHIPPED | fixed via #2414. sklearn `coef_` is `(n_targets, n_features)` in RAW space, `coef_ = (x_rotations_@y_loadings_.T * y_std).T / x_std` (`_pls.py:399-400`). `FittedPLSRegression::coefficients()` now returns exactly this: `PLSRegression::fit` lifts the internal `B = W(PᵀW)⁻¹Qᵀ` to `coef_ = (B * y_std).T / x_std`, shape `(n_targets, n_features)`. Re-orientation + scale-absorption both land; the stored `y_std_` field is retained (per-item `#[allow(dead_code)]`) as a fitted `_y_std` mirror. Non-test consumer: re-export `lib.rs:79-81` + the `predict` impl. Verification: `tests/divergence_pls_coef_orientation_2414.rs::divergence_pls_coef_orientation_shape` (shape `(2,3)`) + `divergence_pls_coef_value_vs_sklearn_coef` (raw-space values ~1e-6) (live, green); in-module `test_plsregression_coefficients_shape` `(2,3)`. |
| REQ-17b (`x_rotations_`/`y_rotations_`/`intercept_` attrs) | NOT-STARTED | open prereq blocker **#1623**. sklearn exposes `intercept_=y_mean` (`:401`), `x_rotations_` (`:391`), `y_rotations_` (`:395`). ferrolearn has NO `x_rotations_`/`y_rotations_`/`intercept_` accessor (the rotation is recomputed inline in `transform`); canonical/CCA fitted structs expose no rotation/coef accessors. (The `coef_` part of #1623 is now SHIPPED as REQ-17a via #2414.) |
| REQ-18 (`deflation_mode`/`mode`/`algorithm='svd'` ctor params) | NOT-STARTED | open prereq blocker **#1624**. sklearn `_PLS.__init__` exposes `deflation_mode` (`regression`/`canonical`), `mode` (`A`/`B`), `algorithm` (`nipals`/`svd`) (`_pls.py:215-228`, constraints `:203-212`). ferrolearn HARD-WIRES these via internal `NipalsMode`/`ScoreNorm`/`WeightMode` enums — no ctor field, no `algorithm="svd"` path (`_get_first_singular_vectors_svd`, `_pls.py:118-125`). `n_iter_` IS exposed. |
| REQ-19 (`inverse_transform` API) | NOT-STARTED | open prereq blocker **#1625**. sklearn `_PLS` exposes `inverse_transform` (`_pls.py:452-503`): `X @ x_loadings_.T * x_std + x_mean` (`:489-492`). NONE of the four ferrolearn fitted structs implement `inverse_transform`. |
| REQ-20 (PyO3 binding surface) | NOT-STARTED | open prereq blocker **#1626**. `grep PLSRegression\|PLSSVD\|PLSCanonical\|CCA ferrolearn-python/src` is EMPTY — none of the four estimators is exposed to Python; no `_RsPLS*`/`_RsCCA` class registered. R-DEFER-1: boundary consumer is the crate re-export `lib.rs:79-81` (grandfathered S5). |
| REQ-21 (ferray substrate) | NOT-STARTED | open prereq blocker **#1627**. `cross_decomposition.rs` computes on `ndarray::{Array1, Array2}` and uses `NdarrayFaerBackend::svd` + hand-rolled `jacobi_eigen_symmetric`/`invert_square`/`pinv`, not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`cross_decomposition.rs` is built on `ndarray::{Array1, Array2}` and four shared
helpers: `fn centre_scale` (ddof=1 centre+scale, REQ-1), `fn apply_centre_scale`
(stored-stat application + column-count guard, REQ-4), `fn svd_dispatch` (f64→faer /
f32→roundtrip / generic→`svd_via_eigen` Jacobi, REQ-5), and `fn pinv` (SVD-based
Moore-Penrose pseudo-inverse with a scipy-`pinv2`-style rank cutoff, used by the CCA
mode-B path, REQ-13).

`PLSSVD` is standalone: `fit` centres+scales, forms `C = XᵀY`, takes `svd_dispatch(&C)`,
applies the `svd_flip(U, Vt)` sign convention (`_pls.py:1105`, REQ-15), and slices
`x_weights_ = U[:, :nc]` / `y_weights_ = Vt.T[:, :nc]` (REQ-16). `FittedPLSSVD::transform`
projects `(X − x_mean)/x_std @ x_weights_`.

The three NIPALS estimators (`PLSRegression`, `PLSCanonical`, `CCA`) funnel through one
`fn nipals` kernel parameterised by three independent flags:
- `NipalsMode` (`Regression`/`Canonical`) selects the Y-deflation rule (REQ-7).
- `WeightMode` (`A`/`B`) selects the weight update: mode-A power method `w = Xᵀu/(uᵀu)`
  (PLSRegression/PLSCanonical) vs mode-B pseudo-inverse `w = X_pinv·u` with per-component
  precomputed `pinv` (CCA, REQ-13, mirroring `_pls.py:78-99`).
- `ScoreNorm` (`None`/`UnitVariance`) controls the `norm_y_weights` normalisation of the
  y-weights `q` (`_pls.py:301`); it does NOT rescale scores (REQ-14).

The kernel seeds `u` from the first non-constant Y column (`_pls.py:71-74`), iterates to
`‖Δw‖² < tol` (with an immediate break when `n_features_y == 1`), then applies the
per-component `_svd_flip_1d` sign convention to `w_final` (`_pls.py:354`/`:154-161`,
REQ-10) BEFORE deriving the scores `t`/`u`, loadings `p`/`q`, and the deflation — so the
sign propagates uniformly. `PLSRegression::fit` additionally builds the internal
`B = W(PᵀW)⁻¹Qᵀ` via `invert_square`, then lifts it to sklearn's raw-space
`coefficients_ = coef_ = (B * y_std).T / x_std`, shape `(n_targets, n_features)`
(`_pls.py:399-400`, REQ-9/REQ-17a); `predict` centres X ONLY and computes
`X_centred @ coef_.T + y_mean` (`_pls.py:530-531`). `transform` (all three) recomputes the
`W(PᵀW)⁻¹` rotation per call and projects centred+scaled X (REQ-11).

Key invariants: ddof=1 throughout (REQ-1); scores are stored WITHOUT unit-variance
rescaling (REQ-14); each component's max-abs `x_weight` entry is positive (REQ-10/15).
The public fitted accessors are `x_weights`/`x_loadings`/`y_loadings`/`x_scores`/
`y_scores`/`n_iter` (NIPALS structs), `coefficients` (PLSRegression only — sklearn's
raw-space `coef_`, REQ-17a), `transform_y`
(PLSSVD/PLSCanonical/CCA), and `x_weights`/`y_weights`/`x_mean`/`y_mean` (PLSSVD). There
is NO public `x_rotations_`/`y_rotations_`/`intercept_` accessor (REQ-17b, #1623)
and NO `inverse_transform` (REQ-19, #1625).

## Verification

```bash
# In-module structural/shape/error/f32 suite.
cargo test -p ferrolearn-decomp cross_decomposition

# Live element-wise parity vs the sklearn 1.5.2 oracle (DEFAULT tol), 0 ignored.
cargo test -p ferrolearn-decomp --test divergence_cross_decomposition

# Oracle re-derivation (Probes 1-5 above) — values regenerated, never copied.
python3 -c "from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD; ..."
```

Re-audit headline (fresh 9×4 X / 9×3 Y fixture, 5 seeds, DEFAULT tol), element-wise incl.
sign vs live sklearn 1.5.2:
- PLSRegression `x_weights_` 5.6e-16, `predict` 8.9e-16.
- PLSCanonical (mode-A) `x_weights_` 5.6e-16, `x_scores_` 1.2e-15.
- CCA (mode-B) `x_weights_` 7.8e-16.
- PLSSVD `x_weights_` 1.1e-15.

All tests in `ferrolearn-decomp/tests/divergence_cross_decomposition.rs` are live and
green (0 `#[ignore]` crate-wide). If any divergence test is not green, the corresponding
value-parity REQ (REQ-9/10/11/13/14/15/16) reverts to NOT-STARTED.

## Blockers

- **#1623** — REQ-17b: public `x_rotations_`/`y_rotations_`/`intercept_` attribute
  accessors (the raw-space `coef_` is now SHIPPED via `coefficients()`, REQ-17a/#2414;
  canonical/CCA expose no rotation/coef accessors; the `transform_y` Y-projection via
  `y_rotations_` also rides on this).
- **#1624** — REQ-18: `deflation_mode`/`mode`/`algorithm='svd'` constructor params (the
  internal `NipalsMode`/`ScoreNorm`/`WeightMode` enums are hard-wired per struct; no
  `algorithm="svd"` NIPALS path).
- **#1625** — REQ-19: `inverse_transform` API on the four fitted structs.
- **#1626** — REQ-20: PyO3 bindings (`_RsPLS*`/`_RsCCA`) for the four estimators.
- **#1627** — REQ-21: ferray substrate (currently `ndarray` + faer/Jacobi/hand-rolled
  `pinv`).
