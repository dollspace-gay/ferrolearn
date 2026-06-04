# PCA (sklearn.decomposition.PCA)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 3c9bb4a7
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_pca.py  # class PCA(_BasePCA) (:121). ctor (:407-423): n_components=None, *, copy=True, whiten=False, svd_solver="auto", tol=0.0, iterated_power="auto", n_oversamples=10, power_iteration_normalizer="auto", random_state=None. _fit (:489) + solver auto-selection (:519-548): "auto" picks covariance_eigh (X.shape[1]<=1000 and X.shape[0]>=10*X.shape[1], :534-535) / full (max(X.shape)<=500 or "mle", :537-538) / randomized (1<=n<0.8*min, :539-540) / full (:543). _fit_full (:551-660) — THE algorithm ferrolearn mirrors. covariance_eigh branch (:593-644): C = X.T @ X (:611); C -= n_samples * mean⊗mean (:612-616); C /= (n_samples-1) (:617); eigenvals, eigenvecs = eigh(C) (:618); flip to descending (:630-631); clip negatives to 0 (:637); explained_variance_ = eigenvals (:638); S = sqrt(eigenvals*(n_samples-1)) (:642); Vt = eigenvecs.T (:643). full branch (:575-591): SVD of centered X, explained_variance_ = S²/(n_samples-1) (:591). svd_flip(U, Vt, u_based_decision=False) (:647) — the deterministic sign convention; components_ = Vt (:649). total_var = sum(explained_variance_) over ALL eigenvalues (:652); explained_variance_ratio_ (:653); singular_values_ = S (:654). n_components postprocess: "mle" (:657-658), 0<n<1.0 cumsum (:659-681), noise_variance_ (:685-688), n_components_ (:691). _fit_truncated (:711-778): arpack/randomized paths with svd_flip(:760,:773).
  - sklearn/utils/extmath.py  # svd_flip(u, v, u_based_decision=True) (:848-906). u_based_decision=False branch (:897-905) operates on Vt ROWS: max_abs_v_rows = argmax(abs(v), axis=1) (:899, numpy argmax → FIRST max on ties); signs = sign(v[row, max_abs]) (:902); v *= signs[:, newaxis] (:905) → each component row's max-abs entry becomes positive. Depends ONLY on the eigenvectors.
  - sklearn/decomposition/_base.py  # _BasePCA. transform (:121-146) → _transform (:148-166): (X − mean_) @ components_.T (+ /sqrt(explained_variance_) if whiten, :157-165). inverse_transform (:168-198): X @ components_ + mean_ (:198) (+ whiten un-scale :192-196). get_covariance (:30-56), get_precision (:58-101), _n_features_out (:200-203). score/score_samples (Gaussian log-likelihood) on PCA via get_precision/noise_variance_.
ferrolearn-module: ferrolearn-decomp/src/pca.rs
parity-ops: PCA
crosslink-issue: 1499
-->

## Summary

`ferrolearn-decomp/src/pca.rs` mirrors scikit-learn's `PCA`
(`sklearn/decomposition/_pca.py`, `class PCA(_BasePCA)` `:121`): linear
dimensionality reduction by projecting centered data onto the directions of
maximum variance (principal components), found by eigendecomposing the sample
covariance matrix. The exposed surface is the unfitted `PCA<F> { n_components }`
(`pca.rs:48`, `new`/`n_components` only — NO `svd_solver`/`whiten`/`copy`/`tol`/
`random_state`) and the fitted `FittedPCA<F> { components_, explained_variance_,
explained_variance_ratio_, mean_, singular_values_ }` (`pca.rs:87`, accessors
`pca.rs:109-135`, `inverse_transform` `pca.rs:145`), re-exported at the crate root
(`pub use pca::{FittedPCA, PCA}`, `lib.rs:98`) and bound to CPython as `_RsPCA`
(`ferrolearn-python/src/transformers.rs:89`, registered `lib.rs:23`).

**ferrolearn's fit EXACTLY MIRRORS sklearn's `covariance_eigh` solver
(`_pca.py:593-644`).** `fn fit` (`pca.rs:369`) computes `mean_`, centers, forms
`cov = X_centeredᵀ·X_centered / (n−1)` (`pca.rs:412-416` — algebraically identical
to sklearn's `C = XᵀX; C -= n·mean⊗mean; C /= n−1` `_pca.py:611-617`),
eigendecomposes via faer `self_adjoint_eigen` (`pca.rs:279-309`), sorts eigenvalues
DESCENDING (`pca.rs:423-428` = sklearn `flip` `_pca.py:630-631`), clips negatives to
0 (`pca.rs:441-445` = `_pca.py:637`), sets `explained_variance_ = eigval`
(`pca.rs:446`), `explained_variance_ratio_ = eigval / total_variance` over ALL
eigenvalues (`pca.rs:430,447-451` = `_pca.py:652-653`), `singular_values_ =
sqrt(eigval·(n−1))` (`pca.rs:453` = `_pca.py:642`), and `components_[k] =
eigenvectors[:,idx]` (`pca.rs:457-459` = `Vt = eigenvecs.T` `_pca.py:643`).

**THE HEADLINE DIVERGENCE — `components_` SIGN CONVENTION (REQ-1 SHIPPED, was
`#1500`, fixed).** sklearn pins eigenvector signs deterministically via
`svd_flip(U, Vt, u_based_decision=False)` (`_pca.py:647`): for each component row,
the max-abs entry is made POSITIVE (`extmath.py:897-905`, depends ONLY on the
eigenvectors). ferrolearn previously had NO `svd_flip` — `fn fit` stored faer's raw
eigenvectors with ARBITRARY signs. As of this iteration `fn fit` (`pca.rs:461-481`)
applies the same per-row max-abs-positive flip (numpy `argmax` first-on-ties via
strict `>`, whole-row negate). Because the rest of the algorithm is exactly
sklearn's `covariance_eigh`, the `components_` / `explained_variance_` /
`explained_variance_ratio_` / `singular_values_` / `transform` now match sklearn
EXACTLY on non-degenerate data (verified element-wise including sign to 1e-6 across
3 fixtures in `tests/divergence_pca.rs`; was #1500, fixed by the critic→fixer cycle).

**DEGENERATE-EIGENVALUE VALUE CARVE-OUT (REQ-2 NOT-STARTED, CARVE-OUT `#1501`).**
On a repeated eigenvalue, faer/LAPACK pick different orthonormal bases for the
eigenspace, so the component VALUES are ambiguous even after the sign flip (same
class as `spectral_embedding`'s degenerate carve-out); the `n_samples < n_features`
rank-deficiency case adds extra ~0 eigenvalues. No failing test is asserted
(R-DEFER-3).

As of this iteration: the components ORTHONORMALITY (REQ-3), the
`explained_variance_` ordering & non-negativity + `explained_variance_ratio_`
summing properties (REQ-4), `singular_values_` non-negativity (REQ-5),
`inverse_transform` round-trip exactness when `n_components == n_features` (REQ-6),
the error/parameter contracts (REQ-7), f32 generic support (REQ-8), `PipelineTransformer`
integration (REQ-9), and the PyO3 `_RsPCA` binding surface (REQ-10) are SHIPPED
scoped, and the `svd_flip` sign convention + exact value parity (REQ-1, was
`#1500`, fixed this iteration) is now SHIPPED. NOT-STARTED: degenerate value
carve-out (REQ-2, `#1501`); `whiten` (REQ-11, `#1502`); `svd_solver` +
`full`-SVD/`randomized`/`arpack` paths (REQ-12, `#1503`); `n_components` as float /
"mle" / None-default (REQ-13, `#1504`); `get_covariance`/`get_precision` (REQ-14,
`#1505`); `score`/`score_samples` + `noise_variance_` (REQ-15, `#1507`);
`n_components_`/`n_features_in_` fitted attrs (REQ-16, `#1508`);
`tol`/`iterated_power`/`n_oversamples`/`random_state`/`copy` ctor params (REQ-17,
`#1509`); the ferray substrate (REQ-18, `#1510`) — **9 SHIPPED / 9 NOT-STARTED**.

`PCA` / `FittedPCA` are existing pub APIs whose non-test consumers are the crate
re-export (`lib.rs:98`, boundary public API, grandfathered S5/R-DEFER-1), the
`_RsPCA` PyO3 binding (`transformers.rs:89`, registered `lib.rs:23`), and the
`PipelineTransformer`/`FittedPipelineTransformer` impls (`pca.rs:509-536`). There is
NO `tests/divergence_pca.rs` yet.

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# PROBE 1 (REQ-1/3/4/5 — fitted attrs + the svd_flip sign convention) — small fixed X
# (10x2). sklearn picks the "full" solver here (max(X.shape)<=500, _pca.py:537-538);
# for n_samples >= 10*n_features it would pick covariance_eigh — both go through the
# SAME svd_flip + ratio/singular postprocess. VALUES generated by sklearn (R-CHAR-3).
python3 -c "
import numpy as np
from sklearn.decomposition import PCA
X=np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],[2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,0.9]])
m=PCA(n_components=2).fit(X)
print('components_:', np.round(m.components_,6).tolist())
print('explained_variance_:', np.round(m.explained_variance_,6).tolist())
print('explained_variance_ratio_:', np.round(m.explained_variance_ratio_,6).tolist())
print('singular_values_:', np.round(m.singular_values_,6).tolist())
print('mean_:', np.round(m.mean_,6).tolist())
for i,row in enumerate(m.components_):
    k=int(np.argmax(np.abs(row)))
    print(f'  comp[{i}] argmax-abs idx={k} value={row[k]:.6f} (positive => svd_flip)')
print('transform row0:', np.round(m.transform(X)[0],6).tolist())"
# -> components_: [[0.677873, 0.735179], [0.735179, -0.677873]]
# -> explained_variance_: [1.284028, 0.049083]
# -> explained_variance_ratio_: [0.963181, 0.036819]   (sums to 1 at n_comp==n_feat)
# -> singular_values_: [3.399448, 0.664643]
# -> mean_: [1.81, 1.91]
# ->   comp[0] argmax-abs idx=1 value=0.735179 (positive => svd_flip)
# ->   comp[1] argmax-abs idx=0 value=0.735179 (positive => svd_flip)
# -> transform row0: [0.82797, 0.175115]
#    => each component ROW's max-abs entry is POSITIVE (svd_flip u_based_decision=False).

# PROBE 2 (REQ-1 — svd_flip(u_based_decision=False) ISOLATED) — eigh / faer return an
# ARBITRARY-sign eigenvector; svd_flip pins the max-abs row entry positive. ferrolearn
# stores the raw faer eigenvector (pca.rs:457-459) and SKIPS this flip => its
# components_ sign DIVERGES whenever faer's sign disagrees. (Cannot run ferrolearn; the
# expected post-flip behaviour is shown.)
python3 -c "
import numpy as np
from sklearn.utils.extmath import svd_flip
Vt_arbitrary = np.array([[-0.677873,-0.735179],[ 0.735179,-0.677873]])
_, Vt_flipped = svd_flip(None, Vt_arbitrary.copy(), u_based_decision=False)
print('arbitrary Vt:', np.round(Vt_arbitrary,6).tolist())
print('flipped  Vt:', np.round(Vt_flipped,6).tolist())"
# -> arbitrary Vt: [[-0.677873, -0.735179], [0.735179, -0.677873]]
# -> flipped  Vt: [[0.677873, 0.735179], [0.735179, -0.677873]]
#    => row 0's max-abs (idx 1) flipped from -0.735179 to +0.735179; row 1 already
#       positive => unchanged. ferrolearn would store EITHER raw sign (faer-dependent).

# PROBE 3 (REQ-11..17 — ctor defaults).
python3 -c "
from sklearn.decomposition import PCA
m=PCA()
for p in ['n_components','copy','whiten','svd_solver','tol','iterated_power','n_oversamples','power_iteration_normalizer','random_state']:
    print(f'{p} =', getattr(m,p))"
# -> n_components = None  copy = True  whiten = False  svd_solver = auto  tol = 0.0
# -> iterated_power = auto  n_oversamples = 10  power_iteration_normalizer = auto
# -> random_state = None
#    => ferrolearn has n_components ONLY; NO copy/whiten/svd_solver/tol/iterated_power/
#       n_oversamples/power_iteration_normalizer/random_state.

# PROBE 4 (REQ-13/14/15/16 — float n_components, noise_variance_, score, get_covariance,
# fitted attrs — all ABSENT in ferrolearn).
python3 -c "
import numpy as np
from sklearn.decomposition import PCA
X=np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],[2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,0.9]])
m=PCA(n_components=1).fit(X)
print('n_components_:', m.n_components_, 'n_features_in_:', m.n_features_in_)
print('noise_variance_:', round(float(m.noise_variance_),6))
print('score (avg loglik):', round(float(m.score(X)),6))
print('get_covariance shape:', m.get_covariance().shape)
mv=PCA(n_components=0.95).fit(X); print('float n_components=0.95 -> n_components_:', mv.n_components_)"
# -> n_components_: 1 n_features_in_: 2
# -> noise_variance_: 0.049083   (mean of discarded eigenvalues, _pca.py:686)
# -> score (avg loglik): -1.355761
# -> get_covariance shape: (2, 2)
# -> float n_components=0.95 -> n_components_: 1   (variance-ratio cumsum, _pca.py:659-681)
#    => ferrolearn exposes none of n_components_/n_features_in_/noise_variance_/score/
#       get_covariance, and requires n_components as an explicit usize.
```

## Requirements

- REQ-1: **`components_` sign via `svd_flip(u_based_decision=False)` + EXACT value
  parity (SHIPPED; was `#1500`, fixed).** sklearn pins eigenvector signs
  deterministically: `U, Vt = svd_flip(U, Vt, u_based_decision=False)`
  (`_pca.py:647`), where `svd_flip` operates on Vt ROWS — `max_abs_v_rows =
  argmax(abs(v), axis=1)` (`extmath.py:899`, numpy `argmax` → FIRST max on ties),
  `signs = sign(v[row, max_abs])` (`:902`), `v *= signs[:, newaxis]` (`:905`) — so
  each component row's max-abs entry becomes POSITIVE (Probe 1/2). ferrolearn's
  `fn fit` now applies the same convention (`pca.rs:461-481`): after filling each
  component row it scans for the max-abs column (strict `>` ⇒ first-on-ties = numpy
  `argmax`) and negates the whole row when that entry is negative. Because the rest
  of the path is exactly sklearn's `covariance_eigh` solver (`_pca.py:593-644`), the
  `components_` and sign-dependent `transform` now match sklearn EXACTLY on
  non-degenerate data. Verification: `tests/divergence_pca.rs`
  `divergence_components_sign_value_parity` / `divergence_transform_sign_value_parity`
  / `divergence_components_sign_convention_max_abs_positive` (all un-ignored) match
  the live sklearn `PCA` oracle element-wise incl. sign to 1e-6 (fixture A 6×3, all 3
  rows flipped); re-audit cross-checks on fresh fixtures B (7×4, per-row-independent
  flips) and C (near-tie) also matched. Consumers: re-export `lib.rs:98`, `_RsPCA`
  `transformers.rs:89`. No algorithm rewrite, no RNG.

- REQ-2: **Degenerate / repeated-eigenvalue + rank-deficient component VALUE
  carve-out (NOT-STARTED, CARVE-OUT; `#1501`).** On a repeated eigenvalue, faer and
  LAPACK (`scipy.linalg.eigh`/`svd`) pick DIFFERENT orthonormal bases for the
  degenerate eigenspace, so the component VALUES are ambiguous even AFTER the REQ-1
  sign flip (only the spanned subspace is well-defined). The `n_samples < n_features`
  rank-deficiency case adds extra eigenvalues near 0 whose eigenvectors are likewise
  basis-ambiguous. This is the same class as `spectral_embedding`'s degenerate
  carve-out (`_spectral_embedding.py` eigenvector ambiguity). **CARVE-OUT
  (R-DEFER-3):** no failing test is asserted — the ambiguity is inherent to the
  eigensolver, not a ferrolearn defect.

- REQ-3: **Components ORTHONORMALITY (unit rows + mutual orthogonality) (SHIPPED
  scoped).** PCA's `components_` are orthonormal eigenvectors of the symmetric
  covariance `C` (sklearn `Vt = eigenvecs.T` from `eigh(C)` `_pca.py:618,643`).
  ferrolearn's `fn fit` (`pca.rs:369`) eigendecomposes the symmetric `cov`
  (`pca.rs:412-416`) via faer `self_adjoint_eigen` (`pca.rs:282`/`:299`, which
  returns orthonormal `U`) and stores each eigenvector column as a `components_` row
  (`pca.rs:457-459`). Pinned by `test_pca_components_orthonormal` (each row unit-norm
  to 1e-8, mutual dot products 0 to 1e-8). **Scope: STRUCTURAL orthonormality, NOT
  exact component VALUES (REQ-1 sign / REQ-2 basis).** Non-test consumers: re-export
  `lib.rs:98`, `_RsPCA` `transformers.rs:89`, `PipelineTransformer` `pca.rs:509`.

- REQ-4: **`explained_variance_` ≥ 0 + DESCENDING; `explained_variance_ratio_`
  sums ≤ 1 (= 1 when `n_components == n_features`) (SHIPPED scoped).** sklearn sorts
  eigenvalues descending (`flip` `_pca.py:630-631`), clips negatives to 0 (`:637`),
  sets `explained_variance_ = eigenvals` (`:638`), and `explained_variance_ratio_ =
  explained_variance_ / sum(explained_variance_)` over ALL eigenvalues
  (`:652-653`). ferrolearn's `fn fit` (`pca.rs:369`) sorts the indices descending by
  eigenvalue (`pca.rs:423-428`), clips negatives to 0 (`pca.rs:441-445`), stores
  `explained_variance_ = eigval_clamped` (`pca.rs:446`), and `explained_variance_ratio_
  = eigval_clamped / total_variance` where `total_variance = sum of ALL eigenvalues`
  (`pca.rs:430,447-451`). Pinned by `test_pca_explained_variance_positive`,
  `test_pca_explained_variance_ratio_sums_le_1`,
  `test_pca_explained_variance_ratio_partial`,
  `test_pca_n_components_equals_n_features` (ratio sum == 1.0 to 1e-8). **Scope:
  ordering / non-negativity / ratio-sum (the MAGNITUDES, which are sign-independent),
  NOT the degenerate-basis VALUES (REQ-2).** Non-test consumers: re-export `lib.rs:98`,
  `_RsPCA` `transformers.rs:89`.

- REQ-5: **`singular_values_` ≥ 0 = `sqrt(explained_variance_·(n−1))` (SHIPPED
  scoped).** sklearn reconstructs `S = sqrt(eigenvals·(n_samples−1))`
  (`_pca.py:642`), `singular_values_ = S` (`:654`). ferrolearn computes
  `singular_values_[k] = (eigval_clamped · n_minus_1).sqrt()` (`pca.rs:453`). Pinned
  by `test_pca_singular_values_positive`. **Scope: the magnitude formula
  (sign-independent), NOT degenerate-basis values (REQ-2).** Non-test consumers:
  re-export `lib.rs:98`, `_RsPCA` `transformers.rs:89`.

- REQ-6: **`inverse_transform` round-trip EXACT when `n_components == n_features`
  (SHIPPED scoped).** sklearn `inverse_transform` (`_base.py:168-198`) is `X @
  components_ + mean_` (`:198`, no-whiten branch); with all components retained,
  `transform` then `inverse_transform` is the identity. ferrolearn's
  `inverse_transform` (`pca.rs:145`) computes `X_reduced·components_ + mean_`
  (`pca.rs:155-160`), the exact algebraic inverse of `transform` `(X−mean)·componentsᵀ`
  (`pca.rs:501`) when `components_` is orthonormal and square. Pinned by
  `test_pca_inverse_transform_roundtrip` (exact to 1e-8 at `n_components ==
  n_features`) and `test_pca_inverse_transform_approx` (bounded error when truncated).
  Note round-trip exactness is SIGN-INVARIANT (the `componentsᵀ·components = I` cancels
  the arbitrary signs), so it is SHIPPED independent of REQ-1. Non-test consumers:
  re-export `lib.rs:98`, `_RsPCA` `transformers.rs:89` (`inverse_transform`
  `transformers.rs:132`).

- REQ-7: **Error / parameter contracts (SHIPPED scoped).** `fn fit` (`pca.rs:369`)
  returns `InvalidParameter { name: "n_components" }` for `n_components == 0`
  (`pca.rs:372-377`) and for `n_components > n_features` (`pca.rs:378-386`), and
  `InsufficientSamples { required: 2 }` for `< 2` samples (`pca.rs:387-393`);
  `transform` returns `ShapeMismatch` on a column-count mismatch (`pca.rs:484-490`),
  and `inverse_transform` returns `ShapeMismatch` when `x_reduced.ncols() !=
  n_components` (`pca.rs:147-153`). Pinned by `test_pca_invalid_n_components_zero`,
  `test_pca_invalid_n_components_too_large`, `test_pca_insufficient_samples`,
  `test_pca_shape_mismatch_transform`, `test_pca_shape_mismatch_inverse_transform`.
  **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints`
  (`_pca.py:380-405`) raising `InvalidParameterError`/`ValueError`, NOT `FerroError`;
  sklearn accepts `n_components=None` (→ `min(X.shape)`, `_pca.py:523-527`) and float
  / "mle" `n_components` (REQ-13) which ferrolearn requires as an explicit `usize`;
  sklearn does not pre-reject `n_samples < 2`. Non-test consumers: re-export
  `lib.rs:98`, `_RsPCA` `transformers.rs:89`.

- REQ-8: **f32 / f64 generic support (SHIPPED scoped).** `PCA<F>` / `FittedPCA<F>`
  are generic over `F: Float + Send + Sync + 'static` (`pca.rs:54`/`:106`); the
  eigensolver dispatches to `faer_eigen_f64` (`pca.rs:279`) / `faer_eigen_f32`
  (`pca.rs:296`) by `TypeId` (`eigen_dispatch` `pca.rs:316-350`), with a Jacobi
  fallback (`jacobi_eigen` `pca.rs:175`) for exotic floats. Pinned by `test_pca_f32`.
  sklearn validates `dtype=[float64, float32]` (`_pca.py:513`). Non-test consumer:
  re-export `lib.rs:98`.

- REQ-9: **`PipelineTransformer` integration (SHIPPED scoped).** `PCA<F>` implements
  `PipelineTransformer<F>` (`pca.rs:509`, `fit_pipeline` `pca.rs:517` delegating to
  `Fit::fit`) and `FittedPCA<F>` implements `FittedPipelineTransformer<F>`
  (`pca.rs:527`, `transform_pipeline` `pca.rs:533` delegating to `Transform::transform`),
  so PCA can be a transform step in a `Pipeline` — the ferrolearn analogue of sklearn's
  `TransformerMixin` (PCA is `ClassNamePrefixFeaturesOutMixin, TransformerMixin`,
  `_base.py:22`). Pinned by `test_pca_pipeline_integration` (PCA step then a sum
  estimator, predicts 4 rows). Non-test consumer: the pipeline machinery itself
  (`ferrolearn_core::pipeline`, `pca.rs:32`).

- REQ-10: **PyO3 `_RsPCA` binding surface (SHIPPED scoped).** `_RsPCA`
  (`transformers.rs:89`, registered `m.add_class::<transformers::RsPCA>()` `lib.rs:23`)
  exposes a `(n_components: usize = 2)` ctor (`transformers.rs:97-104`), `fit`
  (`transformers.rs:106`), `transform` (`transformers.rs:116`), `inverse_transform`
  (`transformers.rs:132`), and getters `components_` (`:148`), `explained_variance_`
  (`:157`), `explained_variance_ratio_` (`:166`), `mean_` (`:178`), `singular_values_`
  (`:187`) over `ferrolearn_decomp::FittedPCA<f64>` — the boundary CPython consumer of
  `PCA::new`/`fit`/`transform`/`inverse_transform` and every `FittedPCA` accessor.
  **Scope: the binding faithfully marshals the Rust surface, but its `components_` /
  `transform` getters inherit REQ-1's arbitrary-sign divergence** — it is NOT
  `n_components` float/"mle", NOT `whiten`/`svd_solver` params, NOT
  `noise_variance_`/`score`/`get_covariance` (those are REQ-11..17). Non-test
  consumer: itself (the CPython boundary).

- REQ-11: **`whiten` (NOT-STARTED; `#1502`).** sklearn's `PCA(whiten=False)`
  (`_pca.py:412`) optionally scales the transform by `1/sqrt(explained_variance_)`
  (`_transform` `_base.py:157-165`) and un-scales it in `inverse_transform`
  (`_base.py:192-196`), de-correlating components to unit variance. ferrolearn's
  `PCA<F>` (`pca.rs:48`) has NO `whiten` field; `transform` (`pca.rs:482`) is the
  plain projection `(X−mean)·componentsᵀ` (`pca.rs:501`, NO whiten scale) and
  `inverse_transform` (`pca.rs:145`) is `X_reduced·components + mean_` (NO whiten
  un-scale).

- REQ-12: **`svd_solver` param + `full`-SVD / `randomized` / `arpack` paths
  (NOT-STARTED; `#1503`).** sklearn's `PCA(svd_solver="auto")` (`_pca.py:413`)
  auto-selects `covariance_eigh` / `full` / `randomized` by data shape
  (`_pca.py:531-543`); `_fit_full`'s `full` branch (`:575-591`) does an SVD of
  centered `X` with `explained_variance_ = S²/(n−1)` (`:591`), and `_fit_truncated`
  (`:711-778`) drives ARPACK `svds` (`:755`) or `randomized_svd` (`:764`).
  ferrolearn implements ONLY the `covariance_eigh`-equivalent eigendecomposition
  (`fn fit` `pca.rs:369`, `eigen_dispatch` `pca.rs:316`) — no `svd_solver` field, no
  SVD-of-X path, no randomized / ARPACK truncated solver.

- REQ-13: **`n_components` as float (variance ratio) / "mle" / None-default
  (NOT-STARTED; `#1504`).** sklearn accepts `n_components=None` (→ `min(X.shape)`,
  `_pca.py:523-527`), a float in `(0, 1)` selecting the smallest count whose
  cumulative `explained_variance_ratio_` exceeds the threshold (`_pca.py:659-681`,
  Probe 4: `0.95 → 1`), and `"mle"` (`_infer_dimension`, `_pca.py:657-658`).
  ferrolearn's `PCA::new(n_components: usize)` (`pca.rs:62`) requires an explicit
  integer count — no float / "mle" / None postprocessing.

- REQ-14: **`get_covariance` / `get_precision` (NOT-STARTED; `#1505`).** sklearn's
  `_BasePCA.get_covariance` (`_base.py:30-56`) reconstructs `cov = components_ᵀ ·
  exp_var_diff · components_ + noise_variance_·I` and `get_precision`
  (`_base.py:58-101`) its inverse via the matrix-inversion lemma (Probe 4:
  `get_covariance` shape `(2,2)`). `FittedPCA<F>` (`pca.rs:87`) exposes neither — it
  has no `noise_variance_` field (REQ-15) on which both depend.

- REQ-15: **`score` / `score_samples` (Gaussian log-likelihood) + `noise_variance_`
  (NOT-STARTED; `#1507`).** sklearn's `PCA` exposes `score_samples`/`score` (the
  per-sample / average Gaussian log-likelihood under the probabilistic-PCA model,
  using `get_precision` and `noise_variance_`; Probe 4: `score = -1.355761`) and the
  fitted `noise_variance_ = mean(explained_variance_[n_components:])` (`_pca.py:686`,
  Probe 4: `0.049083`). `FittedPCA<F>` (`pca.rs:87`) has no `noise_variance_`,
  `score`, or `score_samples` — it never retains the discarded-eigenvalue tail
  (`fn fit` keeps only the top `n_comp`, `pca.rs:438`).

- REQ-16: **Fitted attrs `n_components_` / `n_features_in_` (NOT-STARTED; `#1508`).**
  sklearn exposes `n_components_` (the resolved count after float/"mle"/None
  postprocess, `_pca.py:691`; Probe 4: `1`) and `n_features_in_` (Probe 4: `2`).
  `FittedPCA<F>` (`pca.rs:87`) exposes `components()` / `explained_variance()` /
  `explained_variance_ratio()` / `mean()` / `singular_values()` (`pca.rs:109-135`) —
  no `n_components_` accessor (derivable from `components_.nrows()` but not exposed)
  and no `n_features_in_` (derivable from `mean_.len()` but not exposed).

- REQ-17: **`tol` / `iterated_power` / `n_oversamples` / `random_state` / `copy`
  ctor params (NOT-STARTED; `#1509`).** sklearn's `PCA.__init__` (`_pca.py:407-423`)
  takes `copy=True`, `tol=0.0`, `iterated_power="auto"`, `n_oversamples=10`,
  `power_iteration_normalizer="auto"`, `random_state=None` (Probe 3) — all governing
  the `randomized`/`arpack` truncated solvers (REQ-12) and in-place copy semantics.
  ferrolearn's `PCA<F>` (`pca.rs:48`) has the `n_components` field ONLY — none of
  these parameters exist.

- REQ-18: **ferray substrate (NOT-STARTED; `#1510`).** `pca.rs` computes on
  `ndarray::{Array1, Array2}` (`pca.rs:34`) and eigendecomposes via `faer`
  (`faer_eigen_f64`/`faer_eigen_f32` `pca.rs:279`/`:296`) + a hand-rolled Jacobi
  fallback (`jacobi_eigen` `pca.rs:175`), not `ferray-core` arrays / `ferray::linalg`
  (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED; was `#1500`, fixed): `PCA::new(k).fit(&X).unwrap().components()`
  equals the live sklearn `PCA(n_components=k).fit(X).components_` ROW-FOR-ROW (each
  component row's max-abs entry positive), and `transform(X)` equals sklearn's
  element-wise, INCLUDING sign, to 1e-6 — now that `fn fit` (`pca.rs:461-481`) applies
  `svd_flip(u_based_decision=False)`. Pinned by `tests/divergence_pca.rs`
  (`divergence_components_sign_value_parity` etc., fixture A 6×3 + re-audit fixtures B
  7×4 / C near-tie). PASSES.

- AC-2 (REQ-2, NOT-STARTED, CARVE-OUT): on data with a repeated covariance
  eigenvalue (e.g. isotropic `X`) or `n_samples < n_features`, the component VALUES
  are basis-ambiguous (faer ≠ LAPACK) even after the REQ-1 flip; the spanned subspace
  (and `explained_variance_`) still matches. No failing test asserts element-wise
  value parity (R-DEFER-3).

- AC-3 (REQ-3, SHIPPED scoped): each row of `fitted.components()` is unit-norm and
  rows are mutually orthogonal to 1e-8. Pinned by `test_pca_components_orthonormal`.
  (Orthonormality only — NOT exact values, REQ-1/2.)

- AC-4 (REQ-4, SHIPPED scoped): `fitted.explained_variance()` entries are ≥ 0 and
  descending; `fitted.explained_variance_ratio()` sums ≤ 1, and == 1.0 (to 1e-8)
  when `n_components == n_features`. sklearn oracle (Probe 1):
  `explained_variance_ = [1.284028, 0.049083]`, `ratio = [0.963181, 0.036819]`
  (sums to 1). Pinned by `test_pca_explained_variance_positive`,
  `test_pca_explained_variance_ratio_sums_le_1`, `test_pca_explained_variance_ratio_partial`,
  `test_pca_n_components_equals_n_features`.

- AC-5 (REQ-5, SHIPPED scoped): `fitted.singular_values()` entries are ≥ 0 and equal
  `sqrt(explained_variance · (n−1))`. sklearn oracle (Probe 1): `singular_values_ =
  [3.399448, 0.664643]`. Pinned by `test_pca_singular_values_positive`.

- AC-6 (REQ-6, SHIPPED scoped): with `n_components == n_features`,
  `inverse_transform(transform(X))` reproduces `X` to 1e-8; with fewer components the
  reconstruction error is bounded below the total variance. Pinned by
  `test_pca_inverse_transform_roundtrip`, `test_pca_inverse_transform_approx`.

- AC-7 (REQ-7, SHIPPED scoped): `fit` returns `Err` for `n_components=0`,
  `n_components > n_features`, and `n_samples < 2`; `transform` and `inverse_transform`
  return `Err` for column-count mismatches. Pinned by
  `test_pca_invalid_n_components_zero`, `test_pca_invalid_n_components_too_large`,
  `test_pca_insufficient_samples`, `test_pca_shape_mismatch_transform`,
  `test_pca_shape_mismatch_inverse_transform`. FLAG: sklearn raises
  `InvalidParameterError`/`ValueError` (not `FerroError`), accepts `n_components=None`/
  float/"mle", and does not pre-reject `n_samples < 2`.

- AC-8 (REQ-8/9/10, SHIPPED scoped): `PCA::<f32>::new(1).fit(&X).transform(&X)` has
  1 column (`test_pca_f32`); a `Pipeline` with a `PCA` transform step predicts
  (`test_pca_pipeline_integration`); `import ferrolearn; ferrolearn._RsPCA(2)` exposes
  `fit`/`transform`/`inverse_transform` + `components_`/`explained_variance_`/
  `explained_variance_ratio_`/`mean_`/`singular_values_` getters
  (`transformers.rs:89`, `lib.rs:23`) — with `components_`/`transform` inheriting
  REQ-1's arbitrary sign.

- AC-9 (REQ-11..18, DIVERGES): `PCA()` defaults `n_components=None, copy=True,
  whiten=False, svd_solver="auto", tol=0.0, iterated_power="auto", n_oversamples=10,
  power_iteration_normalizer="auto", random_state=None` (Probe 3, `_pca.py:407-423`);
  sklearn exposes `whiten`, the `full`/`randomized`/`arpack` solvers, float/"mle"/None
  `n_components` (Probe 4: `0.95 → 1`), `get_covariance`/`get_precision`,
  `score`/`score_samples` + `noise_variance_` (Probe 4: `noise_variance_=0.049083`,
  `score=-1.355761`), and `n_components_`/`n_features_in_`. ferrolearn has none of
  these and computes on `ndarray` + `faer`, not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `PCA` / `FittedPCA` are existing pub APIs; the non-test
consumers are the crate re-export (`lib.rs:98`, boundary public API, grandfathered
S5/R-DEFER-1), the `_RsPCA` PyO3 binding (`transformers.rs:89`, registered
`lib.rs:23`), and the `PipelineTransformer` impls (`pca.rs:509-536`). Cites use
symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed
sklearn 1.5.2, run from `/tmp`.
**ferrolearn's fit EXACTLY MIRRORS sklearn's `covariance_eigh` solver
(`_pca.py:593-644`)** — center, `cov = X_centeredᵀX_centered/(n−1)` (`pca.rs:412-416`),
`eigh`, flip-descending, clip-negatives, `explained_variance_`/`ratio`/`singular`
postprocess — EXCEPT the deterministic sign step.
**THE HEADLINE DIVERGENCE — SIGN CONVENTION (REQ-1 SHIPPED, was `#1500`, fixed):**
ferrolearn previously had NO `svd_flip` — `fn fit` stored faer's ARBITRARY-sign
eigenvectors, so `components_` / sign-dependent `transform` diverged from sklearn's
`svd_flip(u_based_decision=False)` (per-row max-abs entry positive, `_pca.py:647`,
`extmath.py:897-905`). As of this iteration `fn fit` (`pca.rs:461-481`) applies the
same per-row max-abs-positive flip, so the values match sklearn EXACTLY on
non-degenerate data (verified element-wise incl. sign to 1e-6 in
`tests/divergence_pca.rs`). **DEGENERATE / RANK-DEFICIENT VALUE CARVE-OUT (REQ-2
NOT-STARTED, `#1501`):** repeated-eigenvalue eigenspaces are basis-ambiguous (faer ≠
LAPACK) even after the flip — same class as `spectral_embedding`'s carve-out, no
failing test (R-DEFER-3). REQ-1 now carries the exact-VALUE parity (the critic
confirmed `components_`/`transform`/`explained_variance_`/`explained_variance_ratio_`/
`singular_values_` match sklearn element-wise to 1e-6 on non-degenerate fixtures);
ferrolearn's `explained_variance_ratio_` correctly divides by the sum of ALL
eigenvalues (`pca.rs:430` = sklearn `total_var = sum(explained_variance_)`
`_pca.py:652`). #1499 is this doc's crosslink tracking issue. Count: **9 SHIPPED
(REQ-1,3,4,5,6,7,8,9,10) / 9 NOT-STARTED (REQ-2,11,12,13,14,15,16,17,18)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`components_` sign via `svd_flip` + EXACT value parity) | SHIPPED | was blocker **#1500** (FIXABLE), now fixed. sklearn `_fit_full` (`_pca.py:551-660`) computes the SAME `covariance_eigh` decomposition as ferrolearn, then pins signs: `U, Vt = svd_flip(U, Vt, u_based_decision=False)` (`:647`); `svd_flip` (`extmath.py:848-906`, `u_based_decision=False` branch `:897-905`) takes `argmax(abs(v), axis=1)` per Vt ROW (`:899`, numpy `argmax` → first-max on ties), `signs = sign(v[row, max_abs])` (`:902`), `v *= signs[:, newaxis]` (`:905`) → each row's max-abs entry POSITIVE. ferrolearn `fn fit` (`pca.rs:461-481`) now applies the same convention: after filling each component row, it finds the max-abs column index (strict `>` ⇒ first-on-ties = numpy `argmax`) and negates the whole row if that entry is negative — so each row's max-abs entry is positive. Because the rest of the path is exactly `covariance_eigh`, the post-flip `components_`/`transform` match sklearn EXACTLY on non-degenerate data. Verification: `cargo test -p ferrolearn-decomp --test divergence_pca` → `divergence_components_sign_value_parity`, `divergence_transform_sign_value_parity`, `divergence_components_sign_convention_max_abs_positive` (all un-ignored) match the live sklearn `PCA` oracle element-wise incl. sign to 1e-6 (fixture A 6×3, all 3 rows were flipped); a re-audit cross-check on fresh fixtures B (7×4, per-row-independent flips at cols 0/2/3) and C (near-tie) matched to 1e-6. Consumers: re-export `lib.rs:98`, `_RsPCA` `transformers.rs:89`. |
| REQ-2 (degenerate / rank-deficient component VALUE carve-out) | NOT-STARTED | open prereq blocker **#1501** (CARVE-OUT, R-DEFER-3). On a repeated covariance eigenvalue, faer's `self_adjoint_eigen` (`pca.rs:282`/`:299`) and LAPACK `eigh`/`svd` (`_pca.py:618`/`:588`) pick DIFFERENT orthonormal bases for the degenerate eigenspace, so the component VALUES are ambiguous even AFTER the REQ-1 sign flip (only the subspace + `explained_variance_` are well-defined); the `n_samples < n_features` rank-deficiency case (`pca.rs:387` allows it once `n_samples >= 2`) adds extra ~0 eigenvalues with basis-ambiguous eigenvectors. Same class as `spectral_embedding`'s degenerate carve-out (`.design/decomp/spectral_embedding.md`). No failing test (R-DEFER-3). |
| REQ-3 (components ORTHONORMALITY) | SHIPPED | PCA `components_` are orthonormal eigenvectors of the symmetric covariance (sklearn `eigh(C)` → `Vt = eigenvecs.T` `_pca.py:618,643`). ferrolearn `fn fit` (`pca.rs:369`) eigendecomposes the symmetric `cov` (`pca.rs:412-416`) via faer `self_adjoint_eigen` (`pca.rs:282`/`:299`, orthonormal `U`) and stores each eigenvector column as a `components_` row (`pca.rs:457-459`). **Scope: STRUCTURAL orthonormality, NOT exact VALUES (REQ-1/2).** Non-test consumers: re-export `lib.rs:98`, `_RsPCA` `transformers.rs:89`, `PipelineTransformer` `pca.rs:509`. Verification: `cargo test -p ferrolearn-decomp pca` → `test_pca_components_orthonormal` (unit rows + mutual orthogonality to 1e-8) PASS. |
| REQ-4 (`explained_variance_` ≥ 0 + descending; `explained_variance_ratio_` sums ≤ 1, = 1 at n_comp==n_feat) | SHIPPED | sklearn flips eigenvalues descending (`_pca.py:630-631`), clips negatives to 0 (`:637`), `explained_variance_ = eigenvals` (`:638`), `explained_variance_ratio_ = explained_variance_ / sum(explained_variance_)` over ALL eigenvalues (`:652-653`). ferrolearn `fn fit` (`pca.rs:369`) sorts indices descending (`pca.rs:423-428`), clips negatives to 0 (`pca.rs:441-445`), `explained_variance_ = eigval_clamped` (`pca.rs:446`), `ratio = eigval_clamped / total_variance` (`total_variance = sum of ALL eigenvalues` `pca.rs:430,447-451`). Probe 1 sklearn `explained_variance_ = [1.284028,0.049083]`, `ratio = [0.963181,0.036819]` (sums 1). **Scope: ordering / non-negativity / ratio-sum (sign-independent magnitudes), NOT degenerate-basis VALUES (REQ-2).** Non-test consumers: re-export `lib.rs:98`, `_RsPCA` `transformers.rs:89`. Verification: `cargo test -p ferrolearn-decomp pca` → `test_pca_explained_variance_positive`, `test_pca_explained_variance_ratio_sums_le_1`, `test_pca_explained_variance_ratio_partial`, `test_pca_n_components_equals_n_features` (ratio sum == 1.0 to 1e-8) PASS. |
| REQ-5 (`singular_values_` ≥ 0 = `sqrt(explained_variance·(n−1))`) | SHIPPED | sklearn `S = sqrt(eigenvals·(n_samples−1))` (`_pca.py:642`), `singular_values_ = S` (`:654`). ferrolearn `singular_values_[k] = (eigval_clamped · n_minus_1).sqrt()` (`pca.rs:453`). Probe 1 sklearn `singular_values_ = [3.399448,0.664643]`. **Scope: the magnitude formula (sign-independent), NOT degenerate-basis values (REQ-2).** Non-test consumers: re-export `lib.rs:98`, `_RsPCA` `transformers.rs:89`. Verification: `cargo test -p ferrolearn-decomp pca` → `test_pca_singular_values_positive` PASS. |
| REQ-6 (`inverse_transform` round-trip EXACT at n_comp==n_feat) | SHIPPED | sklearn `inverse_transform` (`_base.py:168-198`) = `X @ components_ + mean_` (`:198`, no-whiten). ferrolearn `inverse_transform` (`pca.rs:145`) = `X_reduced·components_ + mean_` (`pca.rs:155-160`), the exact algebraic inverse of `transform` `(X−mean)·componentsᵀ` (`pca.rs:501`) when `components_` is orthonormal+square. Round-trip exactness is SIGN-INVARIANT (`componentsᵀ·components = I` cancels arbitrary signs), so SHIPPED independent of REQ-1. Non-test consumers: re-export `lib.rs:98`, `_RsPCA` `transformers.rs:89` (`inverse_transform` `:132`). Verification: `cargo test -p ferrolearn-decomp pca` → `test_pca_inverse_transform_roundtrip` (exact to 1e-8 at n_comp==n_feat), `test_pca_inverse_transform_approx` (bounded truncated error) PASS. |
| REQ-7 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`pca.rs:369`) returns `Err(InvalidParameter{name:"n_components", reason:"must be at least 1"})` for `n_components==0` (`pca.rs:372-377`), `Err(InvalidParameter{name:"n_components", ... "exceeds n_features"})` for `>n_features` (`pca.rs:378-386`), `Err(InsufficientSamples{required:2, ...})` for `<2` samples (`pca.rs:387-393`); `transform` returns `Err(ShapeMismatch)` on column mismatch (`pca.rs:484-490`), `inverse_transform` `Err(ShapeMismatch)` when `ncols != n_components` (`pca.rs:147-153`). Non-test consumers: re-export `lib.rs:98`, `_RsPCA` `transformers.rs:89`. Verification: `cargo test -p ferrolearn-decomp pca` (`test_pca_invalid_n_components_zero`, `_too_large`, `_insufficient_samples`, `_shape_mismatch_transform`, `_shape_mismatch_inverse_transform`) PASS. **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints` (`_pca.py:380-405`) raising `InvalidParameterError`/`ValueError` (not `FerroError`); accepts `n_components=None`/float/"mle"; does NOT pre-reject `n_samples < 2`. |
| REQ-8 (f32 / f64 generic support) | SHIPPED | `PCA<F>`/`FittedPCA<F>` generic over `F: Float + Send + Sync + 'static` (`pca.rs:54`/`:106`); `eigen_dispatch` (`pca.rs:316-350`) routes to `faer_eigen_f64` (`pca.rs:279`)/`faer_eigen_f32` (`pca.rs:296`) by `TypeId`, Jacobi fallback (`jacobi_eigen` `pca.rs:175`). sklearn validates `dtype=[float64,float32]` (`_pca.py:513`). Non-test consumer: re-export `lib.rs:98`. Verification: `cargo test -p ferrolearn-decomp pca` → `test_pca_f32` PASS. |
| REQ-9 (`PipelineTransformer` integration) | SHIPPED | `PCA<F>` impls `PipelineTransformer<F>` (`pca.rs:509`, `fit_pipeline` `pca.rs:517` → `Fit::fit`) and `FittedPCA<F>` impls `FittedPipelineTransformer<F>` (`pca.rs:527`, `transform_pipeline` `pca.rs:533` → `Transform::transform`) — analogue of sklearn's `TransformerMixin` (PCA is `ClassNamePrefixFeaturesOutMixin, TransformerMixin` `_base.py:22`). Non-test consumer: the `ferrolearn_core::pipeline` machinery (`pca.rs:32`). Verification: `cargo test -p ferrolearn-decomp pca` → `test_pca_pipeline_integration` (PCA step → sum estimator, 4 predictions) PASS. |
| REQ-10 (PyO3 `_RsPCA` binding surface, scoped) | SHIPPED | `_RsPCA` (`transformers.rs:89`, registered `m.add_class::<transformers::RsPCA>()` `lib.rs:23`) exposes `(n_components: usize = 2)` ctor (`transformers.rs:97-104`), `fit` (`:106`), `transform` (`:116`), `inverse_transform` (`:132`), and getters `components_` (`:148`), `explained_variance_` (`:157`), `explained_variance_ratio_` (`:166`), `mean_` (`:178`), `singular_values_` (`:187`) over `ferrolearn_decomp::FittedPCA<f64>` — the boundary CPython consumer of `PCA::new`/`fit`/`transform`/`inverse_transform` + all `FittedPCA` accessors. **Scope: faithful marshalling, but `components_`/`transform` getters inherit REQ-1's arbitrary sign**; NOT float/"mle" n_components, NOT `whiten`/`svd_solver`, NOT `noise_variance_`/`score`/`get_covariance` (REQ-11..17). Verification: `import ferrolearn; ferrolearn._RsPCA(2).fit(X)` then `.components_` / `.transform(X)` marshal `numpy2_to_ndarray` ↔ `ndarray2_to_numpy` (`transformers.rs:107`/`:129`). |
| REQ-11 (`whiten`) | NOT-STARTED | open prereq blocker **#1502**. sklearn `PCA(whiten=False)` (`_pca.py:412`) scales `transform` by `1/sqrt(explained_variance_)` (`_base.py:157-165`) and un-scales `inverse_transform` (`_base.py:192-196`). ferrolearn `PCA<F>` (`pca.rs:48`) has NO `whiten` field — `transform` (`pca.rs:482`) is plain `(X−mean)·componentsᵀ` (`pca.rs:501`), `inverse_transform` (`pca.rs:145`) is `X_reduced·components + mean_` (`pca.rs:155-160`), no whiten scaling. |
| REQ-12 (`svd_solver` + full-SVD / randomized / arpack paths) | NOT-STARTED | open prereq blocker **#1503**. sklearn `PCA(svd_solver="auto")` (`_pca.py:413`) auto-selects covariance_eigh/full/randomized (`:531-543`); `_fit_full` full branch (`:575-591`) SVDs centered X (`explained_variance_ = S²/(n−1)` `:591`); `_fit_truncated` (`:711-778`) drives ARPACK `svds` (`:755`) / `randomized_svd` (`:764`). ferrolearn implements ONLY the covariance_eigh-equivalent eigendecomposition (`fn fit` `pca.rs:369`, `eigen_dispatch` `pca.rs:316`) — no `svd_solver` field, no SVD-of-X / randomized / ARPACK path. |
| REQ-13 (`n_components` as float / "mle" / None-default) | NOT-STARTED | open prereq blocker **#1504**. sklearn accepts `n_components=None` (→ `min(X.shape)` `_pca.py:523-527`), float in `(0,1)` via cumulative-ratio cumsum (`_pca.py:659-681`, Probe 4: `0.95 → n_components_ 1`), and `"mle"` (`_infer_dimension` `:657-658`). ferrolearn `PCA::new(n_components: usize)` (`pca.rs:62`) requires an explicit integer — no float / "mle" / None postprocessing. |
| REQ-14 (`get_covariance` / `get_precision`) | NOT-STARTED | open prereq blocker **#1505**. sklearn `_BasePCA.get_covariance` (`_base.py:30-56`) = `components_ᵀ·exp_var_diff·components_ + noise_variance_·I`; `get_precision` (`_base.py:58-101`) its inverse via the matrix-inversion lemma (Probe 4: `get_covariance` shape `(2,2)`). `FittedPCA<F>` (`pca.rs:87`) exposes neither — no `noise_variance_` field (REQ-15) on which both depend. |
| REQ-15 (`score` / `score_samples` + `noise_variance_`) | NOT-STARTED | open prereq blocker **#1507**. sklearn `PCA` exposes `score_samples`/`score` (per-sample / average Gaussian log-likelihood under probabilistic PCA, using `get_precision` + `noise_variance_`; Probe 4: `score = -1.355761`) and fitted `noise_variance_ = mean(explained_variance_[n_components:])` (`_pca.py:686`, Probe 4: `0.049083`). `FittedPCA<F>` (`pca.rs:87`) has no `noise_variance_`/`score`/`score_samples` — `fn fit` keeps only the top `n_comp` (`pca.rs:438`), discarding the eigenvalue tail. |
| REQ-16 (fitted attrs `n_components_` / `n_features_in_`) | NOT-STARTED | open prereq blocker **#1508**. sklearn exposes `n_components_` (resolved count after float/"mle"/None postprocess `_pca.py:691`; Probe 4: `1`) and `n_features_in_` (Probe 4: `2`). `FittedPCA<F>` (`pca.rs:87`) exposes `components()`/`explained_variance()`/`explained_variance_ratio()`/`mean()`/`singular_values()` (`pca.rs:109-135`) — no `n_components_` accessor (derivable from `components_.nrows()`), no `n_features_in_` (derivable from `mean_.len()`). |
| REQ-17 (`tol`/`iterated_power`/`n_oversamples`/`random_state`/`copy` ctor params) | NOT-STARTED | open prereq blocker **#1509**. sklearn `PCA.__init__` (`_pca.py:407-423`) takes `copy=True`, `tol=0.0`, `iterated_power="auto"`, `n_oversamples=10`, `power_iteration_normalizer="auto"`, `random_state=None` (Probe 3) — governing the randomized/arpack truncated solvers (REQ-12) and in-place copy. ferrolearn `PCA<F>` (`pca.rs:48`) has the `n_components` field ONLY — none of these exist. |
| REQ-18 (ferray substrate) | NOT-STARTED | open prereq blocker **#1510**. `pca.rs` computes on `ndarray::{Array1, Array2}` (`pca.rs:34`) and eigendecomposes via `faer` (`faer_eigen_f64`/`faer_eigen_f32` `pca.rs:279`/`:296`) + a hand-rolled Jacobi fallback (`jacobi_eigen` `pca.rs:175`), not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`pca.rs` follows the unfitted/fitted split (CLAUDE.md naming): `PCA<F> {
n_components }` (`pca.rs:48`; `new(n_components)` `pca.rs:62`, accessor
`n_components()` `pca.rs:71`) → `Fit<Array2<F>, ()>` → `FittedPCA<F> { components_,
explained_variance_, explained_variance_ratio_, mean_, singular_values_ }`
(`pca.rs:87`, accessors `components()`/`explained_variance()`/
`explained_variance_ratio()`/`mean()`/`singular_values()` `pca.rs:109-135`,
`inverse_transform` `pca.rs:145`). The path is generic over `F: Float + Send + Sync
+ 'static` (both f32 and f64, `test_pca_f32`); `fit`/`transform`/`inverse_transform`
return `Result<_, FerroError>` (R-CODE-2).

**Fit path (`fn fit` `pca.rs:369`) — REQ-1/2/3/4/5/7.** Validates `n_components != 0`,
`n_components <= n_features`, `n_samples >= 2` (`pca.rs:372-393`) — REQ-7. Step 1:
per-feature `mean` + centering (`pca.rs:398-410`) = sklearn `mean_ = mean(X, axis=0)`
(`_pca.py:567`). Step 2: `cov = X_centeredᵀ·X_centered / (n−1)` (`pca.rs:412-416`) —
algebraically identical to sklearn's `covariance_eigh` `C = XᵀX; C -= n·mean⊗mean; C
/= n−1` (`_pca.py:611-617`). Step 3: `eigen_dispatch` (`pca.rs:316`) → faer
`self_adjoint_eigen` (`pca.rs:282`/`:299`) for f64/f32, Jacobi fallback
(`pca.rs:175`). Step 4: sort eigenvalues DESCENDING (`pca.rs:423-428` = `flip`
`_pca.py:630-631`); compute `total_variance = sum of ALL eigenvalues` (`pca.rs:430` =
`_pca.py:652`); for the top `n_comp`: clip negative eigenvalues to 0 (`pca.rs:441-445`
= `_pca.py:637`), `explained_variance_ = eigval_clamped` (`pca.rs:446` =
`_pca.py:638`), `explained_variance_ratio_ = eigval/total_variance` (`pca.rs:447-451`
= `_pca.py:653`), `singular_values_ = sqrt(eigval·(n−1))` (`pca.rs:453` =
`_pca.py:642`), `components_[k] = eigenvectors[:,idx]` (`pca.rs:457-459` = `Vt =
eigenvecs.T` `_pca.py:643`). **This is sklearn's `covariance_eigh` solver EXCEPT for
the deterministic sign step:** sklearn then runs `svd_flip(U, Vt,
u_based_decision=False)` (`_pca.py:647`) to make each component row's max-abs entry
positive — ferrolearn SKIPS this (REQ-1), so faer's arbitrary signs flow into
`components_`. On a repeated eigenvalue the eigenvector BASIS itself is ambiguous
(REQ-2 carve-out).

**Transform (`impl Transform for FittedPCA` `pca.rs:472`) — REQ-3/sign-of-1.** Centers
`X − mean_` (`pca.rs:493-498`) then projects `(X − mean_)·components_ᵀ`
(`pca.rs:501`) — algebraically sklearn's `_transform` `(X − mean_) @ components_.T`
(`_base.py:148-156`, no-whiten). Because it consumes `components_`, the transform
output's column SIGNS inherit REQ-1's arbitrary sign (the magnitudes/subspace are
correct). No `whiten` scaling (REQ-11).

**Inverse-transform (`pca.rs:145`) — REQ-6.** `X_reduced·components_ + mean_`
(`pca.rs:155-160`) = sklearn `X @ components_ + mean_` (`_base.py:198`). Round-trip
exactness at `n_components == n_features` is SIGN-INVARIANT
(`componentsᵀ·components = I`), so REQ-6 is SHIPPED independent of REQ-1.

**sklearn (target contract).** `class PCA(_BasePCA)` (`_pca.py:121`) takes
`__init__(n_components=None, *, copy=True, whiten=False, svd_solver="auto", tol=0.0,
iterated_power="auto", n_oversamples=10, power_iteration_normalizer="auto",
random_state=None)` (`:407-423`). `_fit` (`:489`) auto-selects the solver by shape
(`:531-543`): `covariance_eigh` (the eigendecomposition of `XᵀX` ferrolearn mirrors,
`:593-644`), `full` (SVD of centered X, `:575-591`), or `randomized`/`arpack`
(`_fit_truncated` `:711-778`). All paths run `svd_flip(u_based_decision=False)`
(`:647`/`:760`/`:773`) for deterministic signs, set `components_ = Vt` (`:649`),
`explained_variance_ratio_` over total variance (`:653`), `singular_values_` (`:654`),
postprocess float/"mle"/None `n_components` (`:657-681`), and compute `noise_variance_`
(`:685-688`) + `n_components_` (`:691`). `_BasePCA` adds `transform`/`inverse_transform`
(`_base.py:121`/`:168`), `get_covariance`/`get_precision` (`:30`/`:58`), and the PCA
subclass adds `score`/`score_samples`.

**The remaining gap.** ferrolearn ships the STRUCTURAL orthonormality (REQ-3),
variance ordering / ratio-sum (REQ-4), singular-value magnitudes (REQ-5), exact
round-trip inverse (REQ-6), the scoped error & parameter contracts (REQ-7), f32
support (REQ-8), pipeline integration (REQ-9), the rich PyO3 binding (REQ-10), and —
as of this iteration — the `svd_flip` sign convention + exact value parity (REQ-1,
was `#1500`, fixed): its fit mirrors sklearn's `covariance_eigh` solver exactly,
including the deterministic sign step. It lacks: the degenerate/rank-deficient VALUE
carve-out (REQ-2, `#1501`); `whiten` (REQ-11, `#1502`); `svd_solver` +
full-SVD/randomized/arpack (REQ-12, `#1503`); float/"mle"/None `n_components`
(REQ-13, `#1504`); `get_covariance`/`get_precision` (REQ-14, `#1505`);
`score`/`score_samples` + `noise_variance_` (REQ-15, `#1507`);
`n_components_`/`n_features_in_` attrs (REQ-16, `#1508`); the
`tol`/`iterated_power`/`n_oversamples`/`random_state`/`copy` ctor params (REQ-17,
`#1509`); and the ferray substrate (REQ-18, `#1510`). This is a
**value-parity-SHIPPED-API-surface-NOT-STARTED** unit (9 SHIPPED / 9 NOT-STARTED),
where REQ-2 is a genuine eigensolver carve-out.

## Verification

Library crate (green at baseline `3c9bb4a7`):
```bash
cargo test -p ferrolearn-decomp pca                          # in-module #[test]s + doctest
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-3..9 (STRUCTURAL):
`test_pca_components_orthonormal` (REQ-3); `test_pca_explained_variance_positive`,
`test_pca_explained_variance_ratio_sums_le_1`,
`test_pca_explained_variance_ratio_partial`, `test_pca_n_components_equals_n_features`
(REQ-4); `test_pca_singular_values_positive` (REQ-5);
`test_pca_inverse_transform_roundtrip`, `test_pca_inverse_transform_approx`,
`test_pca_dimensionality_reduction`, `test_pca_single_component` (REQ-6);
`test_pca_invalid_n_components_zero`, `test_pca_invalid_n_components_too_large`,
`test_pca_insufficient_samples`, `test_pca_shape_mismatch_transform`,
`test_pca_shape_mismatch_inverse_transform`, `test_pca_n_components_getter` (REQ-7);
`test_pca_f32` (REQ-8); `test_pca_pipeline_integration` (REQ-9); plus the module
doctest. REQ-1 (sign / value parity) is now SHIPPED — `tests/divergence_pca.rs` pins
`components_`/`transform` element-wise (incl. sign) against the live sklearn oracle
(3 un-ignored DIV tests + 12 structural green-guards). REQ-2 (degenerate value
parity) is a CARVE-OUT (R-DEFER-3) with no failing test.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-1 sign divergence
and the REQ-4/5 structural magnitudes:
```bash
# REQ-1 sign convention (sklearn svd_flip makes each component row's max-abs entry positive):
python3 -c "import numpy as np; from sklearn.decomposition import PCA
X=np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],[2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,0.9]])
m=PCA(n_components=2).fit(X)
print(np.round(m.components_,6).tolist())
print([int(np.argmax(np.abs(r))) for r in m.components_])"
# -> [[0.677873, 0.735179], [0.735179, -0.677873]]  argmax-abs [1, 0] (both positive)

# REQ-4/5 magnitudes (sign-independent):
python3 -c "import numpy as np; from sklearn.decomposition import PCA
X=np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],[2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,0.9]])
m=PCA(n_components=2).fit(X)
print(np.round(m.explained_variance_,6).tolist(), np.round(m.explained_variance_ratio_,6).tolist(), np.round(m.singular_values_,6).tolist())"
# -> [1.284028, 0.049083] [0.963181, 0.036819] [3.399448, 0.664643]
```
REQ-1's value pin (now that `#1500` has landed) is `PCA::new(k).fit(&X).components()`
matching sklearn's `components_` ROW-FOR-ROW (each row's max-abs entry positive) on
non-degenerate data — green in `tests/divergence_pca.rs`. REQ-2 remains a CARVE-OUT
(no parity test).

ferrolearn-python (REQ-10 binding, present at baseline):
```bash
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -c "import ferrolearn; m=ferrolearn._RsPCA(2)"
```
`_RsPCA` (`transformers.rs:89`, `lib.rs:23`) exposes `(n_components)` ctor +
`fit`/`transform`/`inverse_transform` + `components_`/`explained_variance_`/
`explained_variance_ratio_`/`mean_`/`singular_values_` getters — its `components_` /
`transform` now inherit REQ-1's deterministic `svd_flip` signs; float/"mle"
n_components, `whiten`, `svd_solver`, `noise_variance_`/`score`/`get_covariance` are
REQ-11..17.

## Blockers

(#1499 is this doc's crosslink tracking issue. The blockers below are the open work
items the dispatcher files / numbers; none are filed by this doc — markdown only.
`#1500` (REQ-1, the `svd_flip` sign) was the FIXABLE divergence the critic pinned and
the fixer RESOLVED this iteration — it is now CLOSED.)

- **#1500** — REQ-1 (FIXABLE, RESOLVED): added the sklearn
  `svd_flip(u_based_decision=False)` sign convention (`extmath.py:897-905`) to
  `fn fit` (`pca.rs:461-481`): for each stored `components_` row, find the FIRST
  max-abs entry (numpy `argmax` tie-break, strict `>`) and negate the whole row when
  that entry is negative so the max-abs entry is positive — `components_` /
  `transform` now match sklearn EXACTLY on non-degenerate data, pinned green in
  `tests/divergence_pca.rs`.
- **#1501** — REQ-2 (CARVE-OUT): the repeated-eigenvalue / rank-deficient
  (`n_samples < n_features`) eigenvector BASIS is ambiguous (faer ≠ LAPACK) even after
  the REQ-1 flip; reaching element-wise value parity there is inherently
  eigensolver-bound (no failing test, R-DEFER-3, same class as `spectral_embedding`).
- **#1502** — REQ-11: add a `whiten` field (`_pca.py:412`) scaling `transform` by
  `1/sqrt(explained_variance_)` (`_base.py:157-165`) and un-scaling
  `inverse_transform` (`_base.py:192-196`).
- **#1503** — REQ-12: add a `svd_solver` field (`_pca.py:413`) + the `full` SVD-of-X
  path (`_pca.py:575-591`) and the `randomized`/`arpack` truncated solvers
  (`_fit_truncated` `_pca.py:711-778`), with the auto-selection heuristic
  (`_pca.py:531-543`).
- **#1504** — REQ-13: accept `n_components` as float in `(0,1)` (variance-ratio
  cumsum `_pca.py:659-681`), `"mle"` (`_infer_dimension` `:657-658`), and `None`
  (→ `min(X.shape)` `:523-527`).
- **#1505** — REQ-14: add `get_covariance` (`_base.py:30-56`) and `get_precision`
  (`_base.py:58-101`) on `FittedPCA`, requiring `noise_variance_` (REQ-15).
- **#1507** — REQ-15: retain the discarded-eigenvalue tail as `noise_variance_ =
  mean(explained_variance_[n_components:])` (`_pca.py:686`) and add `score`/
  `score_samples` (Gaussian log-likelihood via `get_precision`).
- **#1508** — REQ-16: expose `n_components_` (`_pca.py:691`) and `n_features_in_`
  fitted attrs on `FittedPCA` (derivable from `components_.nrows()` / `mean_.len()`).
- **#1509** — REQ-17: add the `copy`/`tol`/`iterated_power`/`n_oversamples`/
  `power_iteration_normalizer`/`random_state` ctor params (`_pca.py:407-423`) for the
  truncated solvers (REQ-12).
- **#1510** — REQ-18: migrate `pca.rs` off `ndarray` + `faer` + the hand-rolled
  `jacobi_eigen` (`pca.rs:175`) to `ferray-core` arrays / `ferray::linalg`
  (R-SUBSTRATE).
