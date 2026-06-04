# SparsePCA (sklearn.decomposition.SparsePCA / MiniBatchSparsePCA)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: a7934c59
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_sparse_pca.py  # class _BaseSparsePCA (:24-156): __init__(n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-8, method="lars", n_jobs=None, verbose=False, random_state=None) (:39-60); fit (:62-91) = mean_ = X.mean(axis=0); X = X - mean_; n_components = X.shape[1] if None (:86-89); _fit(...) (:91); transform (:93-123) = X = X - mean_; U = ridge_regression(components_.T, X.T, ridge_alpha, solver="cholesky") (:119-121); return U (:123) — RIDGE, not plain projection (docstring :100-101 "components orthogonality is not enforced ... one cannot use a simple linear projection"); inverse_transform (:125-146) = (X @ components_) + mean_ (:146); _n_features_out = components_.shape[0] (:148-151). class SparsePCA(_BaseSparsePCA) (:159-336): ctor (:279-306) adds U_init/V_init; _fit (:308-336) = dict_learning(X.T, n_components, alpha, tol, max_iter, method, ..., random_state, code_init=V_init.T, dict_init=U_init.T, return_n_iter=True) (:313-326) -> code, dictionary, E, n_iter_; svd_flip(code, dictionary, u_based_decision=True) (:328); components_ = code.T (:329); components_ /= norm (per-row L2, 0->1) (:330-332); n_components_ = len(components_) (:333); error_ = E (:335). class MiniBatchSparsePCA(_BaseSparsePCA) (:339-552): ctor (:488-519) adds callback/batch_size/shuffle/max_no_improvement, tol=1e-3; _fit (:521-552) = MiniBatchDictionaryLearning(...).fit(X.T).transform(X.T).T (:525-545).
  - sklearn/decomposition/_dict_learning.py  # dict_learning (:910-1080): alternating sparse_encode (LARS/CD) + _update_dict; _dict_learning (:554-674) inner loop; alpha = float(regularization) / n_features (per-feature scaling, :120, :141); return_n_iter (:661-673). LARS solver default (method="lars").
  - sklearn/linear_model/_ridge.py  # ridge_regression(X, y, alpha, solver="cholesky") — closed-form (X^T X + alpha I)^-1 X^T y used by _BaseSparsePCA.transform.
  - sklearn/utils/extmath.py  # svd_flip(u, v, u_based_decision=True) — deterministic sign on (code, dictionary).
ferrolearn-module: ferrolearn-decomp/src/sparse_pca.rs
parity-ops: SparsePCA, MiniBatchSparsePCA
crosslink-issue: 1476
-->

## Summary

`ferrolearn-decomp/src/sparse_pca.rs` mirrors scikit-learn's `SparsePCA`
(`sklearn/decomposition/_sparse_pca.py`, `class SparsePCA(_BaseSparsePCA)`
`:159`): find sparse principal components that reconstruct the data by combining
PCA with an L1 (lasso) penalty on the loadings, producing components with many
exact zeros at the cost of explained variance. The exposed surface is the
unfitted `SparsePCA<F> { n_components, alpha (default 1.0), max_iter (1000), tol
(1e-8), random_state }` (`sparse_pca.rs:55`, builders `with_alpha`/`with_max_iter`/
`with_tol`/`with_random_state`, accessors) and the fitted `FittedSparsePCA<F> {
components_ (n_components, n_features), mean_, n_iter_ }` (`sparse_pca.rs:148`),
re-exported at the crate root (`pub use sparse_pca::{FittedSparsePCA, SparsePCA}`,
`lib.rs:99`) and bound to CPython as `_RsSparsePCA` (`extras.rs:1129`, registered
`lib.rs:77`).

**SPARSE-PCA TRANSFORM = RIDGE (R-HONEST-3, REQ-1 SHIPPED, was `#1477`, fixed).**
sklearn's `_BaseSparsePCA.transform` (`_sparse_pca.py:93-123`) does NOT project:
it solves `U = ridge_regression(components_.T, X_centered.T, ridge_alpha=0.01,
solver="cholesky")` (`:119-121`) — a regularized least-squares fit of the
(non-orthonormal) sparse components to the centered data — precisely because
"Sparse PCA components orthogonality is not enforced as in PCA hence one cannot
use a simple linear projection" (docstring `:100-101`). ferrolearn's `transform`
(`impl Transform for FittedSparsePCA`, `sparse_pca.rs`) now computes the same
Ridge form `U = (X − mean_)·Cᵀ·(C·Cᵀ + 0.01·I)⁻¹`, matching the sklearn ridge
oracle on ferrolearn's own fitted components to 1e-6
(`tests/divergence_sparse_pca.rs::divergence_transform_is_ridge_not_projection`,
un-ignored, green). Previously a PLAIN projection (max-abs diff 1.13 on a fixed
`C`/`X`); pinned by the critic (#1477) and fixed by the fixer this iteration.

**EXACT COMPONENTS VALUE PARITY DIVERGES (R-HONEST-3, REQ-4 NOT-STARTED, CARVE-OUT
`#1478`).** sklearn's `SparsePCA._fit` (`_sparse_pca.py:308-336`) runs
`dict_learning(X.T, n_components, alpha, ...)` (`:313-326`) — LARS-solved lasso
dictionary learning seeded from a numpy `RandomState` with per-feature alpha
scaling (`_dict_learning.py:120`/`:141`) — then `svd_flip` (`:328`) and per-row L2
normalization (`:330-332`). ferrolearn fits a PCA/random-init alternating
soft-threshold-lasso + least-squares dictionary (`fn fit`, `sparse_pca.rs:265`;
`soft_threshold`, `sparse_pca.rs:189`) seeded from a Rust `StdRng`. Different
algorithm + different RNG ⇒ the `components_` VALUES diverge (same class as the
cluster / bayesian-GMM RNG carve-outs); no failing test is asserted (R-DEFER-3).

As of this iteration: the ridge `transform` (REQ-1, was `#1477`, fixed), the
structural sparsity / shape / centering / error-decrease / determinism (REQ-2),
the error & parameter contracts (REQ-3), and the PyO3 fit/transform binding
surface (REQ-10) are SHIPPED scoped; exact `dict_learning` components value parity
(REQ-4, `#1478`), `ridge_alpha` exposure (REQ-5, `#1479`), `method` (lars/cd) +
LARS solver (REQ-6, `#1480`), `U_init`/`V_init` + alpha/n_samples scaling (REQ-7,
`#1481`), `MiniBatchSparsePCA` (REQ-8, `#1482`), `inverse_transform` +
`error_`/`n_components_`/`n_features_in_` attrs (REQ-9, `#1483`), and the ferray
substrate (REQ-11, `#1484`) are NOT-STARTED — **4 SHIPPED / 7 NOT-STARTED**.

`SparsePCA` / `FittedSparsePCA` are existing pub APIs whose non-test consumers are
the crate re-export (`lib.rs:99`, boundary public API, grandfathered S5/R-DEFER-1)
and the `_RsSparsePCA` PyO3 binding (`extras.rs:1129`, registered `lib.rs:77`).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# REQ-1 (NOT-STARTED; FIXABLE #1477) — the transform RIDGE formula vs ferrolearn's
# PLAIN projection. C is a fixed NON-orthonormal components matrix (n_components=2,
# n_features=3); X a fixed 3x3 sample matrix (R-CHAR-3). sklearn transform =
# ridge_regression(C.T, Xc.T, 0.01, solver="cholesky"); ferrolearn = Xc @ C.T.
python3 -c "import numpy as np; from sklearn.linear_model import ridge_regression
C=np.array([[0.8,0.6,0.0],[0.0,0.6,0.8]]); X=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
Xc=X-X.mean(axis=0)
U=ridge_regression(C.T, Xc.T, 0.01, solver='cholesky')
print('ridge U (sklearn transform):', np.round(U,6).tolist())
print('plain proj (ferrolearn)    :', np.round(Xc@C.T,6).tolist())
print('max abs diff:', round(float(np.abs(U-Xc@C.T).max()),6))"
# -> ridge U (sklearn transform): [[-3.065693,-3.065693],[0.0,0.0],[3.065693,3.065693]]
# -> plain proj (ferrolearn)    : [[-4.2,-4.2],[0.0,0.0],[4.2,4.2]]
# -> max abs diff: 1.134307   => ferrolearn transform DIVERGES (REQ-1 NOT-STARTED).

# REQ-2 (SHIPPED scoped) — sklearn components_ are STRUCTURALLY SPARSE (exact zeros),
# shape (n_components, n_features); mean centering; error_ vector. NOT value parity.
python3 -c "import numpy as np; from sklearn.decomposition import SparsePCA
X=np.array([[1.,0,0,2,0],[0,3,0,1,0],[2,0,1,0,4],[0,2,3,0,1],[1,1,1,1,1],[3,0,2,1,0],[0,4,0,2,1]])
m=SparsePCA(n_components=2,alpha=1,random_state=0).fit(X)
print('components_ shape:', m.components_.shape)
print('frac exact zeros:', float(np.mean(m.components_==0)))
print('n_iter_:', m.n_iter_, 'error_ len/last:', len(m.error_), round(float(m.error_[-1]),4))
print('mean_:', np.round(m.mean_,6).tolist(), 'n_features_in_:', m.n_features_in_)"
# -> components_ shape: (2, 5)
# -> frac exact zeros: 0.6         => L1 yields exact zeros (structural sparsity)
# -> n_iter_: 8 error_ len/last: 8 15.419   => error_ is a per-iteration vector
# -> mean_: [1.0,1.428571,1.0,1.0,1.0] n_features_in_: 5

# REQ-9 (NOT-STARTED #1483) — sklearn transform output + inverse_transform for a known fit.
python3 -c "import numpy as np; from sklearn.decomposition import SparsePCA
X=np.array([[1.,0,0,2,0],[0,3,0,1,0],[2,0,1,0,4],[0,2,3,0,1],[1,1,1,1,1],[3,0,2,1,0],[0,4,0,2,1]])
m=SparsePCA(n_components=2,alpha=1,random_state=0).fit(X); Xt=m.transform(X)
print('transform shape:', Xt.shape, 'row0:', np.round(Xt[0],6).tolist())
print('inverse_transform row0:', np.round(m.inverse_transform(Xt)[0],6).tolist())"
# -> transform shape: (7, 2) row0: [-1.252944,-1.074542]
# -> inverse_transform row0: [1.581375,0.318674,1.0,1.095936,-0.070251]  ((X @ components_)+mean_)
```

## Requirements

- REQ-1: **`transform` = RIDGE regression (`ridge_alpha=0.01`), matching
  `_BaseSparsePCA.transform` (SHIPPED; was `#1477`, fixed).** sklearn centers
  `X = X − mean_` then computes `U = ridge_regression(components_.T, X.T,
  ridge_alpha, solver="cholesky")` and returns `U` (`_sparse_pca.py:117-123`) — a
  Ridge-regularized least-squares fit of the non-orthonormal sparse components to
  the centered data, NOT a projection (docstring `:100-101`). ferrolearn's
  `transform` (`impl Transform<Array2<F>> for FittedSparsePCA`) now computes the
  closed-form Ridge solution `U = (X − mean_)·Cᵀ·(C·Cᵀ + 0.01·I)⁻¹` (==
  `ridge_regression(Cᵀ, X_centeredᵀ, 0.01, solver="cholesky")ᵀ`) using a
  `ridge_alpha = 0.01` field and `invert_small_symmetric`, returning
  `NumericalInstability` on a singular ridge matrix. The transform FORMULA
  `ridge_regression(componentsᵀ, X_centeredᵀ, alpha=0.01, solver="cholesky")` is
  the oracle (R-CHAR-3); ferrolearn now matches it on its own fitted components to
  1e-6 (and 9.4e-10 on an independent fresh fixture). Was the PLAIN projection
  `x_centered.dot(&self.components_.t())` (Probe REQ-1, max-abs diff 1.13) — the
  critic pinned it (#1477), the fixer added the ridge transform this iteration.

- REQ-2: **Structural: components SPARSITY (L1 → exact zeros), shape
  `(n_components, n_features)`, mean centering, reconstruction error decreases over
  iterations, deterministic given seed (SHIPPED scoped).** `fn fit`
  (`sparse_pca.rs:265`) centers via the per-feature `mean` (`sparse_pca.rs:296-307`,
  = sklearn `mean_ = X.mean(axis=0)` `_sparse_pca.py:83`), runs the
  alternating sparse-coding (`sparse_code_row` with `soft_threshold`,
  `sparse_pca.rs:189`/`:203` — the L1 prox producing exact zeros, mirroring
  `dict_learning`'s lasso `:313`) / dictionary-update loop (`sparse_pca.rs:340-392`),
  and stores `components_` of shape `(n_components, n_features)` (`sparse_pca.rs:150`,
  = sklearn `components_` `_sparse_pca.py:220`). The reconstruction error
  `reconstruction_error_sq` (`sparse_pca.rs:240`) is monitored and the loop breaks on
  relative-change `< tol` (`sparse_pca.rs:388-391`) — the analogue of sklearn's
  per-iteration `error_` vector (`_sparse_pca.py:335`). The seeded `StdRng`
  (`sparse_pca.rs:310-311`) makes the fit DETERMINISTIC. Pinned by
  `test_sparse_pca_components_shape` `(2,4)`, `test_sparse_pca_mean_is_correct`,
  `test_sparse_pca_high_alpha_produces_sparser`. **Scope: STRUCTURAL (sparsity /
  shape / centering / error-decrease / determinism), NOT components value parity
  (REQ-4).** Non-test consumers: re-export `lib.rs:99`, `_RsSparsePCA` `extras.rs:1129`.

- REQ-3: **Error / parameter contracts (SHIPPED scoped).** `fn fit`
  (`sparse_pca.rs:265`) returns `InvalidParameter { name: "n_components" }` for
  `n_components == 0` (`sparse_pca.rs:268-273`) and for `n_components > n_features`
  (`sparse_pca.rs:274-282`), and `InsufficientSamples { required: 2 }` for `< 2`
  samples (`sparse_pca.rs:283-289`). **FLAG (candidate DIVs):** sklearn validates
  via `_parameter_constraints` (`_sparse_pca.py:27-37`: `n_components` in
  `[None, [1,∞)]`, `alpha`/`ridge_alpha`/`tol` in `[0,∞)`, `max_iter` in `[0,∞)`)
  and raises `InvalidParameterError`, NOT `FerroError`; sklearn accepts
  `n_components=None` (→ `n_features`, `_sparse_pca.py:86-89`) which ferrolearn
  requires as an explicit `usize`; sklearn does not pre-reject `n_components >
  n_features` or `n_samples < 2` (LARS/`dict_learning` would surface it later).
  Pinned by `test_sparse_pca_n_components_zero`,
  `test_sparse_pca_n_components_too_large`, `test_sparse_pca_insufficient_samples`,
  `test_sparse_pca_transform_shape_mismatch`. Non-test consumers: re-export
  `lib.rs:99`, `_RsSparsePCA` `extras.rs:1129`.

- REQ-4: **EXACT `components_` value parity with sklearn `dict_learning`
  (NOT-STARTED, CARVE-OUT; `#1478`).** sklearn's `SparsePCA._fit`
  (`_sparse_pca.py:308-336`) runs `dict_learning(X.T, n_components, alpha=alpha,
  tol, max_iter, method, ..., random_state, ...)` (`:313-326`) → LARS-solved lasso
  dictionary learning with per-feature alpha scaling `alpha = regularization /
  n_features` (`_dict_learning.py:120`/`:141`), seeded from a numpy `RandomState`,
  followed by `svd_flip(code, dictionary, u_based_decision=True)` (`:328`),
  `components_ = code.T`, and per-row L2 normalization (`:330-332`). ferrolearn's
  `fn fit` (`sparse_pca.rs:265`) uses a DIFFERENT algorithm — random-init
  (`sparse_pca.rs:310-317`) alternating per-row soft-threshold-lasso coordinate
  descent (`sparse_code_row`, `sparse_pca.rs:203`; `soft_threshold`,
  `sparse_pca.rs:189`) + a least-squares dictionary update `V = (XᵀU)(UᵀU)⁻¹`
  (`sparse_pca.rs:355-369`) with column L2 normalization (`sparse_pca.rs:373-384`),
  seeded from a Rust `StdRng`, with NO svd_flip and NO per-feature alpha scaling.
  Different algorithm + different RNG ⇒ the `components_` VALUES diverge. **CARVE-OUT
  (R-DEFER-3):** matching sklearn requires reimplementing `dict_learning`'s LARS
  solver + numpy RNG semantics + alpha/n_features scaling; no failing test is
  asserted (same class as the cluster / bayesian-GMM RNG carve-outs).

- REQ-5: **`ridge_alpha` parameter exposure (NOT-STARTED; `#1479`).** sklearn's
  `SparsePCA(..., ridge_alpha=0.01, ...)` (`_sparse_pca.py:284`) controls the Ridge
  shrinkage in `transform` (`:119-121`). ferrolearn's `SparsePCA<F>`
  (`sparse_pca.rs:55`) has NO `ridge_alpha` field — only `n_components`, `alpha`,
  `max_iter`, `tol`, `random_state`. Folds with REQ-1: after the ridge transform
  fix, `ridge_alpha=0.01` would be hard-coded; exposing it as a ctor param /
  builder is the remaining surface.

- REQ-6: **`method` (`lars`/`cd`) + `dict_learning` LARS solver parity
  (NOT-STARTED; `#1480`).** sklearn's `SparsePCA(method="lars")`
  (`_sparse_pca.py:287`, default `"lars"`, constraint `StrOptions({"lars","cd"})`
  `:33`) flows into `dict_learning(..., method=self.method, ...)` (`:319`) selecting
  least-angle-regression vs coordinate-descent lasso. ferrolearn hard-codes a
  single hand-rolled soft-threshold coordinate-descent sparse coder
  (`sparse_code_row`, `sparse_pca.rs:203`, fixed `n_cd_iters = 10`
  `sparse_pca.rs:335`) — no `method` field, no LARS path.

- REQ-7: **`U_init`/`V_init` warm restart + alpha/n_samples scaling semantics
  (NOT-STARTED; `#1481`).** sklearn's `SparsePCA(U_init=None, V_init=None)`
  (`_sparse_pca.py:289-290`) seeds `dict_learning(..., code_init=V_init.T,
  dict_init=U_init.T)` (`:311-312`, `:323-324`) for warm restarts, and
  `dict_learning` scales `alpha = regularization / n_features`
  (`_dict_learning.py:120`/`:141`). ferrolearn's `SparsePCA<F>` (`sparse_pca.rs:55`)
  has no `U_init`/`V_init` fields (it always random-inits `V`,
  `sparse_pca.rs:310-317`) and applies `alpha` un-scaled directly in
  `soft_threshold(residual_dot, alpha_f)` (`sparse_pca.rs:233`, `alpha_f`
  `sparse_pca.rs:293`).

- REQ-8: **`MiniBatchSparsePCA` (NOT-STARTED; `#1482`).** sklearn's
  `class MiniBatchSparsePCA(_BaseSparsePCA)` (`_sparse_pca.py:339`) is the online /
  mini-batch variant: `_fit` (`:521-552`) drives a `MiniBatchDictionaryLearning`
  (`:525-541`) over feature batches (`batch_size=3`, `shuffle`, `max_no_improvement`,
  `callback`, `tol=1e-3`). It is a declared route `parity-op` but is ABSENT in
  ferrolearn (`grep -rn MiniBatchSparsePCA ferrolearn-decomp/src/` is empty) — no
  `MiniBatchSparsePCA` type, no online dictionary learner.

- REQ-9: **`inverse_transform` + `error_` / `n_components_` / `n_features_in_`
  fitted attrs (NOT-STARTED; `#1483`).** sklearn exposes `inverse_transform`
  (`_sparse_pca.py:125-146`, `(X @ components_) + mean_` `:146`) and the fitted
  attrs `error_` (per-iteration error vector `:223`/`:335`), `n_components_` (`:226`/
  `:333`), and `n_features_in_` (`:238`). `FittedSparsePCA<F>` (`sparse_pca.rs:148`)
  exposes only `components()` / `mean()` / `n_iter()` accessors
  (`sparse_pca.rs:160-174`) — no `inverse_transform`, no `error_` vector (only the
  scalar loop is monitored, not retained), no `n_components_` / `n_features_in_`
  accessors.

- REQ-10: **PyO3 binding surface (SHIPPED scoped; thin).** `_RsSparsePCA`
  (`extras.rs:1129`, registered `lib.rs:77`) is generated by the `py_transformer!`
  macro (`extras.rs:107-149`), exposing a `(n_components: usize = 2)` ctor, `fit`
  (`extras.rs:127`), and `transform` (`extras.rs:137`) over
  `ferrolearn_decomp::FittedSparsePCA<f64>`. This is the boundary CPython consumer
  of `SparsePCA::new` / `fit` / `transform`. **Scope: the macro exposes ONLY
  `n_components` + `fit` + `transform`** — NOT `alpha`/`max_iter`/`tol`/
  `random_state` ctor params, NOT a `components_` accessor, NOT
  `inverse_transform`; and because the underlying `transform` is the plain
  projection (REQ-1), the binding inherits the transform divergence. The richer
  binding surface is part of REQ-1/REQ-5/REQ-9.

- REQ-11: **ferray substrate (NOT-STARTED; `#1484`).** `sparse_pca.rs` computes on
  `ndarray::{Array1, Array2}` (`sparse_pca.rs:40`) and inverts `UᵀU` with a
  hand-rolled Gauss-Jordan (`invert_small_symmetric`, `sparse_pca.rs:405`), not
  `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED; was `#1477`, fixed): for ferrolearn's own fitted
  `components_ = C` and `mean_ = X.mean(axis=0)`, `transform(X)` equals
  `ridge_regression(C.T, (X−mean).T, 0.01, solver="cholesky")` to < 1e-6 — NOT the
  plain projection `(X−mean) @ C.T` (for the fixed `C =
  [[0.8,0.6,0],[0,0.6,0.8]]`/`X = [[1,2,3],[4,5,6],[7,8,9]]`, ridge `U =
  [[-3.065693,-3.065693],[0,0],[3.065693,3.065693]]` vs proj `[[-4.2,-4.2],[0,0],
  [4.2,4.2]]`, max-abs diff 1.13). Pinned by
  `tests/divergence_sparse_pca.rs::divergence_transform_is_ridge_not_projection`
  (un-ignored, green); fresh-fixture cross-check matched to 9.4e-10. PASSES.

- AC-2 (REQ-2, SHIPPED scoped): `SparsePCA::new(2).with_random_state(0).fit(&X)
  .unwrap().components()` has shape `(2, n_features)`; high `alpha` yields lower
  projected energy than low `alpha`; `mean()` equals the column means; the fit is
  identical across runs with a fixed seed. Pinned by
  `test_sparse_pca_components_shape` `(2,4)`, `test_sparse_pca_mean_is_correct`,
  `test_sparse_pca_high_alpha_produces_sparser`, `test_sparse_pca_basic`. (Structural
  sparsity / shape / centering only — NOT the exact component values, REQ-4.)

- AC-3 (REQ-3, SHIPPED scoped): `fit` returns `Err` for `n_components=0`,
  `n_components > n_features`, and `n_samples < 2`; `transform` returns `Err` for a
  column-count mismatch. Pinned by `test_sparse_pca_n_components_zero`,
  `test_sparse_pca_n_components_too_large`, `test_sparse_pca_insufficient_samples`,
  `test_sparse_pca_transform_shape_mismatch`. FLAG: sklearn raises
  `InvalidParameterError` (not `FerroError`), accepts `n_components=None`, and does
  not pre-reject `n_components > n_features` / `n_samples < 2`.

- AC-4 (REQ-4, NOT-STARTED, CARVE-OUT): `SparsePCA(n_components=2, alpha=1,
  random_state=0).fit(X).components_` (Probe REQ-2: shape `(2,5)`, 60% exact zeros,
  `[[-0.464007,0.885831,0,0,0],[0,0,0,-0.089281,0.996006]]`) is NOT reproduced
  element-wise by ferrolearn (different LARS-`dict_learning` algorithm + numpy RNG +
  alpha/n_features scaling). No failing test asserts this (R-DEFER-3).

- AC-5 (REQ-5..9, DIVERGES): `SparsePCA()` defaults `n_components=None, alpha=1,
  ridge_alpha=0.01, max_iter=1000, tol=1e-8, method="lars", U_init=None,
  V_init=None` (`_sparse_pca.py:279-306`); sklearn exposes `transform` (ridge),
  `inverse_transform` (`(X@components_)+mean_`, Probe REQ-9 `inverse_transform row0
  = [1.581375,0.318674,1.0,1.095936,-0.070251]`), `error_`, `n_components_`,
  `n_features_in_`, and the `MiniBatchSparsePCA` class. ferrolearn has no
  `ridge_alpha`/`method`/`U_init`/`V_init`, no `inverse_transform`, no
  `error_`/`n_components_`/`n_features_in_` accessors, and no `MiniBatchSparsePCA`.

- AC-6 (REQ-10/11): `import ferrolearn` exposes `_RsSparsePCA` with a
  `(n_components)` ctor + `fit` + `transform` (`extras.rs:1129`, registered
  `lib.rs:77`) — but NO `alpha`/`tol`/`random_state` params, NO `components_`
  accessor, NO `inverse_transform`; the module imports `ndarray` (`sparse_pca.rs:40`)
  + hand-rolled Gauss-Jordan inversion, not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `SparsePCA` / `FittedSparsePCA` are existing pub APIs; the
non-test consumers are the crate re-export (`lib.rs:99`, boundary public API,
grandfathered S5/R-DEFER-1) and the `_RsSparsePCA` PyO3 binding (`extras.rs:1129`,
registered `lib.rs:77`). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2). Live oracle = installed sklearn 1.5.2, run from `/tmp`.
**SPARSE-PCA TRANSFORM = RIDGE (R-HONEST-3, REQ-1 SHIPPED, was `#1477`,
fixed):** sklearn's `transform` is a RIDGE regression
(`ridge_regression(components_.T, X_centered.T, ridge_alpha=0.01,
solver="cholesky")`, `_sparse_pca.py:119-121`); ferrolearn now computes the same
Ridge form `(X−mean)·Cᵀ·(C·Cᵀ+0.01·I)⁻¹`, matching the sklearn ridge oracle on its
own fitted components to 1e-6 (was the PLAIN projection, max-abs diff 1.13).
**EXACT COMPONENTS
VALUE PARITY DIVERGES (R-HONEST-3, REQ-4 NOT-STARTED, CARVE-OUT `#1478`):**
ferrolearn's random-init alternating soft-threshold-lasso + LS dictionary
(`sparse_pca.rs:265`) ≠ sklearn's LARS `dict_learning` + svd_flip + numpy RNG +
alpha/n_features scaling (`_sparse_pca.py:308-336`, `_dict_learning.py:120`). The
least-confident SHIPPED claim is REQ-2 — it is STRUCTURAL (sparsity / shape /
centering / error-decrease / determinism), explicitly NOT the component VALUES
(REQ-4); the in-tree tests assert shapes / mean / finiteness, not oracle parity.
#1476 is this doc's crosslink tracking issue. Count: **4 SHIPPED (REQ-1,2,3,10) /
7 NOT-STARTED (REQ-4,5,6,7,8,9,11)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`transform` = RIDGE regression `ridge_alpha=0.01`) | SHIPPED | was blocker **#1477** (FIXABLE), now fixed. sklearn `_BaseSparsePCA.transform` (`_sparse_pca.py:93-123`): `X = X − mean_` (`:117`) then `U = ridge_regression(components_.T, X.T, ridge_alpha, solver="cholesky")` (`:119-121`); return `U` (`:123`) — a Ridge LS fit of the NON-orthonormal sparse components, NOT a projection (docstring `:100-101` "one cannot use a simple linear projection"). ferrolearn `transform` (`impl Transform<Array2<F>> for FittedSparsePCA`) now computes `U = (X−mean_)·Cᵀ·(C·Cᵀ + 0.01·I)⁻¹` (== `ridge_regression(Cᵀ, X_centeredᵀ, 0.01, solver="cholesky")ᵀ`) via the `ridge_alpha=0.01` field + `invert_small_symmetric`, returning `NumericalInstability` on a singular ridge matrix. Verification: `cargo test -p ferrolearn-decomp --test divergence_sparse_pca` → `divergence_transform_is_ridge_not_projection` (un-ignored) matches the sklearn ridge oracle on ferrolearn's own fitted `components_` to 1e-6; an independent fresh-fixture cross-check (7×4, seed 123) matched to 9.4e-10 (ridge vs projection differs by 3.35). Was PLAIN projection (Probe REQ-1, max-abs diff 1.13). Consumers: re-export `lib.rs:99`, `_RsSparsePCA` `extras.rs:1129`. |
| REQ-2 (structural: sparsity / shape / centering / error-decrease / determinism) | SHIPPED | `fn fit` (`sparse_pca.rs:265`) centers via per-feature `mean` (`sparse_pca.rs:296-307` = `mean_ = X.mean(axis=0)` `_sparse_pca.py:83`), runs alternating sparse-coding (`sparse_code_row` `sparse_pca.rs:203` with `soft_threshold` `sparse_pca.rs:189` — the L1 prox producing exact zeros, analogue of `dict_learning`'s lasso `_sparse_pca.py:313`) / dictionary-update (`sparse_pca.rs:340-392`), storing `components_` shape `(n_components, n_features)` (`sparse_pca.rs:150` = `components_` `_sparse_pca.py:220`). `reconstruction_error_sq` (`sparse_pca.rs:240`) monitored; loop breaks on relative-change `< tol` (`sparse_pca.rs:388-391`) — analogue of `error_` `_sparse_pca.py:335`. Seeded `StdRng` (`sparse_pca.rs:310-311`) ⇒ DETERMINISTIC. Probe REQ-2 confirms sklearn structural sparsity (60% exact zeros, shape `(2,5)`). **Scope: STRUCTURAL, NOT value parity (REQ-4).** Non-test consumers: re-export `lib.rs:99`, `_RsSparsePCA` `extras.rs:1129`. Verification: `cargo test -p ferrolearn-decomp sparse_pca` → `test_sparse_pca_components_shape` `(2,4)`, `test_sparse_pca_mean_is_correct`, `test_sparse_pca_high_alpha_produces_sparser`, `test_sparse_pca_basic` PASS. |
| REQ-3 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`sparse_pca.rs:265`) returns `Err(InvalidParameter { name: "n_components", reason: "must be at least 1" })` for `n_components==0` (`sparse_pca.rs:268-273`), `Err(InvalidParameter { name: "n_components", ... "exceeds n_features" })` for `n_components > n_features` (`sparse_pca.rs:274-282`), and `Err(InsufficientSamples { required: 2, actual: n, context: "SparsePCA::fit requires at least 2 samples" })` for `< 2` samples (`sparse_pca.rs:283-289`); `transform` returns `Err(ShapeMismatch)` on column-count mismatch (`sparse_pca.rs:490-496`). Non-test consumers: re-export `lib.rs:99`, `_RsSparsePCA` `extras.rs:1129`. Verification: `cargo test -p ferrolearn-decomp sparse_pca` (`test_sparse_pca_n_components_zero`, `_n_components_too_large`, `_insufficient_samples`, `_transform_shape_mismatch`) PASS. **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints` (`_sparse_pca.py:27-37`) raising `InvalidParameterError` (not `FerroError`); accepts `n_components=None` → `n_features` (`:86-89`); does NOT pre-reject `n_components > n_features` or `n_samples < 2`. |
| REQ-4 (EXACT `components_` value parity with `dict_learning`) | NOT-STARTED | open prereq blocker **#1478** (CARVE-OUT, R-DEFER-3). sklearn `SparsePCA._fit` (`_sparse_pca.py:308-336`): `dict_learning(X.T, n_components, alpha, tol, max_iter, method, ..., random_state, ...)` (`:313-326`) — LARS lasso dictionary learning, per-feature scaling `alpha = regularization / n_features` (`_dict_learning.py:120`/`:141`), numpy `RandomState` init — then `svd_flip(code, dictionary, u_based_decision=True)` (`:328`), `components_ = code.T` (`:329`), per-row L2 norm (`:330-332`). ferrolearn `fn fit` (`sparse_pca.rs:265`): DIFFERENT algorithm — random-init (`sparse_pca.rs:310-317`) per-row soft-threshold CD (`sparse_code_row` `sparse_pca.rs:203`) + LS dict update `V=(XᵀU)(UᵀU)⁻¹` (`sparse_pca.rs:355-369`) + column L2 norm (`sparse_pca.rs:373-384`), Rust `StdRng`, NO svd_flip, NO alpha/n_features scaling. Probe REQ-2 sklearn `components_ = [[-0.464007,0.885831,0,0,0],[0,0,0,-0.089281,0.996006]]` NOT reproduced. No failing test (same class as cluster / bayesian-GMM RNG carve-outs). |
| REQ-5 (`ridge_alpha` param exposure) | NOT-STARTED | open prereq blocker **#1479** (folds with REQ-1). sklearn `SparsePCA(ridge_alpha=0.01)` (`_sparse_pca.py:284`) sets the Ridge shrinkage in `transform` (`:119-121`). ferrolearn `SparsePCA<F>` (`sparse_pca.rs:55`) has no `ridge_alpha` field (only `n_components`/`alpha`/`max_iter`/`tol`/`random_state`); after the ridge transform fix, `ridge_alpha=0.01` is hard-coded — exposing it as ctor/builder is the remaining surface. |
| REQ-6 (`method` lars/cd + LARS solver) | NOT-STARTED | open prereq blocker **#1480**. sklearn `SparsePCA(method="lars")` (`_sparse_pca.py:287`, `StrOptions({"lars","cd"})` `:33`) flows into `dict_learning(..., method=self.method)` (`:319`) selecting LARS vs CD lasso. ferrolearn hard-codes a single soft-threshold CD sparse coder (`sparse_code_row` `sparse_pca.rs:203`, fixed `n_cd_iters=10` `sparse_pca.rs:335`) — no `method` field, no LARS path. |
| REQ-7 (`U_init`/`V_init` warm restart + alpha/n_samples scaling) | NOT-STARTED | open prereq blocker **#1481**. sklearn `SparsePCA(U_init=None, V_init=None)` (`_sparse_pca.py:289-290`) seeds `dict_learning(code_init=V_init.T, dict_init=U_init.T)` (`:311-312`,`:323-324`); `dict_learning` scales `alpha = regularization / n_features` (`_dict_learning.py:120`/`:141`). ferrolearn `SparsePCA<F>` (`sparse_pca.rs:55`) has no `U_init`/`V_init` (always random-inits `V` `sparse_pca.rs:310-317`) and applies `alpha` un-scaled in `soft_threshold(residual_dot, alpha_f)` (`sparse_pca.rs:233`). |
| REQ-8 (`MiniBatchSparsePCA`) | NOT-STARTED | open prereq blocker **#1482**. sklearn `class MiniBatchSparsePCA(_BaseSparsePCA)` (`_sparse_pca.py:339`), `_fit` (`:521-552`) drives `MiniBatchDictionaryLearning` (`:525-541`, `batch_size=3`, `shuffle`, `max_no_improvement`, `tol=1e-3`) over feature batches. Declared route `parity-op` but ABSENT in ferrolearn (`grep -rn MiniBatchSparsePCA ferrolearn-decomp/src/` is empty) — no type, no online dictionary learner. |
| REQ-9 (`inverse_transform` + `error_`/`n_components_`/`n_features_in_` attrs) | NOT-STARTED | open prereq blocker **#1483**. sklearn exposes `inverse_transform` (`_sparse_pca.py:125-146`, `(X @ components_) + mean_` `:146`; Probe REQ-9 `inverse_transform row0 = [1.581375,0.318674,1.0,1.095936,-0.070251]`) and attrs `error_` (`:223`/`:335`), `n_components_` (`:226`/`:333`), `n_features_in_` (`:238`). `FittedSparsePCA<F>` (`sparse_pca.rs:148`) exposes only `components()`/`mean()`/`n_iter()` (`sparse_pca.rs:160-174`) — no `inverse_transform`, no retained `error_` vector, no `n_components_`/`n_features_in_`. |
| REQ-10 (PyO3 binding surface, scoped) | SHIPPED | `_RsSparsePCA` (`extras.rs:1129`, registered `lib.rs:77`) via `py_transformer!` (`extras.rs:107-149`) exposes a `(n_components: usize = 2)` ctor, `fit` (`extras.rs:127`), `transform` (`extras.rs:137`) over `ferrolearn_decomp::FittedSparsePCA<f64>` — the boundary CPython consumer of `SparsePCA::new`/`fit`/`transform`. **Scope: ONLY `n_components` + `fit` + `transform`** — NOT `alpha`/`max_iter`/`tol`/`random_state` ctor params, NOT a `components_` accessor, NOT `inverse_transform`; the bound `transform` inherits REQ-1's plain-projection divergence. Verification: `m.add_class::<extras::RsSparsePCA>()` (`lib.rs:77`); the macro `fit`/`transform` marshal `numpy2_to_ndarray` ↔ `ndarray2_to_numpy` (`extras.rs:129`/`:145`). The richer surface (params, `components_`, `inverse_transform`, ridge transform) is REQ-1/5/9. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#1484**. `sparse_pca.rs` computes on `ndarray::{Array1, Array2}` (`sparse_pca.rs:40`) and inverts `UᵀU` with a hand-rolled Gauss-Jordan (`invert_small_symmetric` `sparse_pca.rs:405`), not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`sparse_pca.rs` follows the unfitted/fitted split (CLAUDE.md naming): `SparsePCA<F>
{ n_components, alpha (default 1.0), max_iter (1000), tol (1e-8), random_state }`
(`sparse_pca.rs:55`; `new(n_components)` `sparse_pca.rs:75`, builders `with_alpha`
`:88` / `with_max_iter` `:95` / `with_tol` `:102` / `with_random_state` `:109`,
accessors `n_components()`/`alpha()`/`max_iter()`/`tol()` `:116-136`) →
`Fit<Array2<F>, ()>` → `FittedSparsePCA<F> { components_, mean_, n_iter_ }`
(`sparse_pca.rs:148`, accessors `components()`/`mean()`/`n_iter()`
`sparse_pca.rs:160-174`). The path is generic over `F: Float + Send + Sync +
'static` (both f32 and f64, `test_sparse_pca_f32`); `fit`/`transform` return
`Result<_, FerroError>` (R-CODE-2).

**Fit path (`fn fit` `sparse_pca.rs:265`) — REQ-2/3/4.** Validates `n_components !=
0`, `n_components <= n_features`, `n_samples >= 2` (`sparse_pca.rs:268-289`) — REQ-3.
Step 1: per-feature `mean` + centering (`sparse_pca.rs:296-307`) = sklearn `mean_ =
X.mean(axis=0)` (`_sparse_pca.py:83-84`). Step 2: random-init dictionary `V`
(n_components, n_features) from a seeded `StdRng` uniform `[-1,1]`, row-normalized
(`sparse_pca.rs:310-330`). Step 3 (outer loop to `max_iter`, `sparse_pca.rs:340`):
(a) fix `V`, solve each row of the code `U` via `sparse_code_row`
(`sparse_pca.rs:344-351`) — `n_cd_iters=10` coordinate-descent passes, each
applying the L1 prox `soft_threshold(residual_dot, alpha) / vk_norm_sq`
(`sparse_pca.rs:233`); (b) fix `U`, update `V = (XᵀU)(UᵀU)⁻¹` via
`invert_small_symmetric` (`sparse_pca.rs:355-369`), then row-normalize
(`sparse_pca.rs:373-384`); break on relative reconstruction-error change `< tol`
(`sparse_pca.rs:388-391`). Stores `components_ = V`, `mean_`, `n_iter_`
(`sparse_pca.rs:394-398`). **This is NOT sklearn's `dict_learning` (REQ-4):**
sklearn runs LARS-solved lasso dictionary learning with per-feature alpha scaling
and `svd_flip` (`_sparse_pca.py:308-336`, `_dict_learning.py:120`); ferrolearn's
random-init alternating soft-threshold-CD + LS-update algorithm + Rust RNG produce
DIFFERENT component values (CARVE-OUT).

**Transform (`impl Transform for FittedSparsePCA`) — REQ-1 (SHIPPED).** Centers
`X − mean_` then solves the closed-form Ridge `U = (X − mean_)·Cᵀ·(C·Cᵀ +
0.01·I)⁻¹` — algebraically `ridge_regression(components_.T, X_centered.T,
ridge_alpha=0.01, solver="cholesky")` (`_sparse_pca.py:119-121`) — to correct for
the NON-orthonormality of the sparse components (docstring `:100-101`). It builds
`M = C·Cᵀ + 0.01·I`, inverts via `invert_small_symmetric` (returning
`NumericalInstability` if singular — unreachable since `M` is SPD), and returns
`(X − mean_)·Cᵀ·M⁻¹`. Matches the sklearn ridge oracle on ferrolearn's own fitted
components to 1e-6. Was the PLAIN projection `x_centered.dot(&components_.t())`
(Probe REQ-1, max-abs diff 1.13) — REQ-1 was `#1477`, fixed this iteration.

**sklearn (target contract).** `class SparsePCA(_BaseSparsePCA)` (`_sparse_pca.py:159`)
takes `__init__(n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000,
tol=1e-8, method="lars", n_jobs=None, U_init=None, V_init=None, verbose=False,
random_state=None)` (`:279-306`). `_BaseSparsePCA.fit` (`:62`) sets `mean_ =
X.mean(axis=0)`, centers, defaults `n_components = n_features` if `None`
(`:86-89`), and calls `_fit` (`:91`). `SparsePCA._fit` (`:308`) runs
`dict_learning(X.T, n_components, alpha, tol, max_iter, method, ...,
random_state, code_init=V_init.T, dict_init=U_init.T, return_n_iter=True)`
(`:313-326`) → `code, dictionary, E, n_iter_`, applies `svd_flip(code, dictionary,
u_based_decision=True)` (`:328`), sets `components_ = code.T` (`:329`), L2-normalizes
each row (`:330-332`), and stores `error_ = E` (`:335`), `n_components_` (`:333`).
`transform` (`:93`) is the Ridge LS fit; `inverse_transform` (`:125`) is
`(X @ components_) + mean_`. `MiniBatchSparsePCA` (`:339`) is the online variant
over `MiniBatchDictionaryLearning` (`:521-552`).

**The remaining gap.** ferrolearn ships the Ridge `transform` (REQ-1, fixed this
iteration), the STRUCTURAL sparsity / shape / centering / error-decrease /
determinism (REQ-2), the scoped error & parameter contracts (REQ-3), and the thin
PyO3 fit/transform binding (REQ-10). It lacks: the exact `dict_learning` components
value parity (REQ-4, CARVE-OUT `#1478`); `ridge_alpha` ctor/builder exposure
(REQ-5, `#1479`); `method` lars/cd + LARS (REQ-6, `#1480`); `U_init`/`V_init` +
alpha/n_features scaling (REQ-7, `#1481`); `MiniBatchSparsePCA` (REQ-8, `#1482`);
`inverse_transform` + `error_`/`n_components_`/`n_features_in_` attrs (REQ-9,
`#1483`); and the ferray substrate (REQ-11, `#1484`). This is a
**transform/structure-SHIPPED-components-NOT-STARTED** unit (4 SHIPPED / 7
NOT-STARTED).

## Verification

Library crate (green at baseline `a7934c59`):
```bash
cargo test -p ferrolearn-decomp sparse_pca                   # in-module #[test]s + doctest
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-2/3 (STRUCTURAL): `test_sparse_pca_basic` `(4,2)`,
`test_sparse_pca_single_component` (ncols 1), `test_sparse_pca_components_shape`
`(2,4)`, `test_sparse_pca_high_alpha_produces_sparser`,
`test_sparse_pca_mean_is_correct`, `test_sparse_pca_f32`,
`test_sparse_pca_builder_methods`, `test_sparse_pca_n_iter_positive` (REQ-2);
`test_sparse_pca_n_components_zero`, `test_sparse_pca_n_components_too_large`,
`test_sparse_pca_insufficient_samples`, `test_sparse_pca_transform_shape_mismatch`
(REQ-3); plus the module doctest. **`tests/divergence_sparse_pca.rs`** now pins
REQ-1: `divergence_transform_is_ridge_not_projection` (the ridge transform matches
the sklearn `ridge_regression` oracle on ferrolearn's own fitted components to
1e-6) + `divergence_transform_ridge_formula_differs_from_projection` (formula
probe) + 9 structural green-guards. REQ-4 (components value parity) remains a
CARVE-OUT (R-DEFER-3) with NO failing test.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-1 transform-ridge
divergence, the REQ-2 structural sparsity, and the REQ-9 inverse_transform surface:
```bash
# REQ-1 (transform RIDGE vs ferrolearn PLAIN projection — DIVERGES, FIXABLE #1477):
python3 -c "import numpy as np; from sklearn.linear_model import ridge_regression
C=np.array([[0.8,0.6,0.0],[0.0,0.6,0.8]]); X=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
Xc=X-X.mean(axis=0); U=ridge_regression(C.T, Xc.T, 0.01, solver='cholesky')
print('ridge:', np.round(U,6).tolist(), 'proj:', np.round(Xc@C.T,6).tolist())"
# -> ridge: [[-3.065693,...],[0,0],[3.065693,...]]  proj: [[-4.2,...],[0,0],[4.2,...]]  (DIFFER)

# REQ-2 (structural sparsity / shape — sklearn components_ have exact zeros):
python3 -c "import numpy as np; from sklearn.decomposition import SparsePCA
X=np.array([[1.,0,0,2,0],[0,3,0,1,0],[2,0,1,0,4],[0,2,3,0,1],[1,1,1,1,1],[3,0,2,1,0],[0,4,0,2,1]])
m=SparsePCA(n_components=2,alpha=1,random_state=0).fit(X)
print(m.components_.shape, float(np.mean(m.components_==0)))"
# -> (2, 5) 0.6
```
REQ-1's value pin (after `#1477` lands) is `FittedSparsePCA{components_=C,
mean_=X.mean(axis=0)}.transform(X)` matching `ridge_regression(C.T, (X−mean).T,
0.01, solver="cholesky")` to < 1e-6. REQ-4 remains a CARVE-OUT (no parity test).

ferrolearn-python (REQ-10 binding, present at baseline):
```bash
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -c "import ferrolearn; from ferrolearn import _RsSparsePCA"
```
`_RsSparsePCA` (`extras.rs:1129`, `lib.rs:77`) exposes `(n_components)` ctor +
`fit` + `transform` only — its `transform` now inherits REQ-1's Ridge form; the
richer surface (params, `components_`, `inverse_transform`) is REQ-5/9.

## Blockers

(#1476 is this doc's crosslink tracking issue. The blockers below are the open work
items the dispatcher files / numbers; none are filed by this doc — markdown only.
`#1477` (REQ-1, the ridge transform) was the FIXABLE divergence the critic pinned
and the fixer RESOLVED this iteration — it is now CLOSED.)

- **#1477** — REQ-1 (FIXABLE, RESOLVED): replaced the plain-projection `transform`
  (`x_centered.dot(&components_.t())`) with sklearn's Ridge LS
  fit `U = ridge_regression(components_.T, X_centered.T, ridge_alpha=0.01,
  solver="cholesky")` (`_sparse_pca.py:119-121`) — closed-form
  `(C·Cᵀ + ridge_alpha·I)⁻¹ · C · X_centeredᵀ` per the Cholesky solver — to correct
  for the non-orthonormality of the sparse components. Pin with a
  `tests/divergence_sparse_pca.rs` against the Probe REQ-1 oracle.
- **#1478** — REQ-4 (CARVE-OUT): reimplement sklearn's `dict_learning` (LARS lasso
  solver, per-feature `alpha / n_features` scaling `_dict_learning.py:120`, numpy
  `RandomState` init, `svd_flip` `_sparse_pca.py:328`) to reach EXACT `components_`
  value parity; inherently RNG/algorithm-bound (no failing test, R-DEFER-3).
- **#1479** — REQ-5: add a `ridge_alpha` field + `with_ridge_alpha` builder
  (default 0.01, `_sparse_pca.py:284`), threading it into the REQ-1 ridge transform
  (currently hard-coded after `#1477`).
- **#1480** — REQ-6: add a `method` field (`lars`/`cd`, `_sparse_pca.py:287`) and a
  LARS sparse-coding path (`dict_learning(method=...)` `:319`).
- **#1481** — REQ-7: add `U_init`/`V_init` warm-restart fields
  (`_sparse_pca.py:289-290`) and the `alpha / n_features` scaling semantics
  (`_dict_learning.py:120`/`:141`).
- **#1482** — REQ-8: add a `MiniBatchSparsePCA` type mirroring the online
  `MiniBatchDictionaryLearning` variant (`_sparse_pca.py:339-552`).
- **#1483** — REQ-9: add `inverse_transform` (`(X @ components_) + mean_`,
  `_sparse_pca.py:146`) and expose `error_` / `n_components_` / `n_features_in_`
  fitted attrs (`_sparse_pca.py:223`/`:226`/`:238`).
- **#1484** — REQ-11: migrate `sparse_pca.rs` off `ndarray` + the hand-rolled
  `invert_small_symmetric` (`sparse_pca.rs:405`) to `ferray-core` / `ferray::linalg`
  (R-SUBSTRATE).
