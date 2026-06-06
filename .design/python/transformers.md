# ferrolearn-python transformers — sklearn StandardScaler + PCA binding shim

<!--
tier: 3-component
status: draft
baseline-commit: be3362243
upstream-paths:
  - sklearn/preprocessing/_data.py     # class StandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) — __init__/_parameter_constraints/fit/transform/inverse_transform; attrs mean_/var_/scale_/n_samples_seen_/n_features_in_
  - sklearn/decomposition/_pca.py       # class PCA(_BasePCA) — __init__/_parameter_constraints/_fit; attrs components_/explained_variance_/explained_variance_ratio_/singular_values_/mean_/n_components_/noise_variance_/n_features_in_
-->

## Summary

`ferrolearn-python/src/transformers.rs` is the PyO3 marshalling shim binding TWO
transformers to CPython: `#[pyclass(name = "_RsStandardScaler")] RsStandardScaler`
(over `ferrolearn_preprocess::FittedStandardScaler<f64>`) and
`#[pyclass(name = "_RsPCA")] RsPCA` (over `ferrolearn_decomp::FittedPCA<f64>`).
`ferrolearn-python/python/ferrolearn/_transformers.py` wraps each as a sklearn
`TransformerMixin`/`BaseEstimator` subclass — `StandardScaler` and `PCA` — so
`import ferrolearn` mirrors `import sklearn.preprocessing.StandardScaler` and
`import sklearn.decomposition.PCA`. They mirror **`sklearn.preprocessing.StandardScaler`**
(`sklearn/preprocessing/_data.py:696`) and **`sklearn.decomposition.PCA`**
(`sklearn/decomposition/_pca.py:121`).

The transformer *correctness* (per-column standardize math; PCA covariance
eigendecomposition + `svd_flip` sign convention) lives DOWN in
`ferrolearn-preprocess/src/standard_scaler.rs` and `ferrolearn-decomp/src/pca.rs`,
each audited by that crate's `//!` REQ status table. THIS unit is the
**sklearn-API marshalling shim** only: constructor parameter ABI, attribute
exposure, method surface, and array coercion across the Python↔Rust boundary
(R-DEV-2 ABI, R-DEV-3 output contract, the `ferray::numpy_interop` boundary
covered by `conversions.md` #2027). Semantic/numerical and missing-knob
divergences are owned by the downstream crates; this doc references their existing
blockers rather than re-filing them, and owns only the binding-level surface.

**Verification model: B (pytest vs sklearn 1.5.2).** Per goal.md §"The
verification model (B)", this unit is verified by
`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` comparing
`import ferrolearn` against the installed `import sklearn` 1.5.2 oracle, plus the
live-sklearn oracle for the constructor-ABI and attribute boundary. As of baseline
`be3362243` the gauntlet is GREEN: **524 passed**. `ferrolearn.StandardScaler()`
and `ferrolearn.PCA(n_components=2)` are both exercised by
`tests/test_check_estimator.py` (`parametrize_with_checks`).

Divergence classes:
1. **api-conformance (SHIPPED core)** — on the DEFAULT parameter path
   (`with_mean=True`/`with_std=True`; `n_components=2`), both transformers expose
   `fit`/`transform`/`inverse_transform` with the right shapes/types plus the
   fitted attributes, and the marshalled VALUES match the live sklearn oracle
   element-wise (StandardScaler `mean_`/`scale_`/`var_`/`n_samples_seen_`; PCA
   `components_`/`explained_variance_`/`explained_variance_ratio_`/`mean_`/
   `singular_values_`, including the `svd_flip` sign). `check_estimator` passes
   for `StandardScaler()` and `PCA(n_components=2)`.
2. **PCA constructor-ABI (NOT-STARTED, R-DEV-2, single-wrapper-fixable — the
   headline)** — `_transformers.py::PCA.__init__(self, *, n_components=2)` makes
   `n_components` keyword-only (so `ferrolearn.PCA(2)` raises `TypeError` whereas
   `sklearn.PCA(2)` works) AND defaults it to `2` instead of sklearn's `None`
   (keep `min(n_samples, n_features)` components). Both are pure Python-wrapper
   constructor fixes (move `n_components` before the `*`; default `None` resolved
   at fit) needing no Rust change.
3. **StandardScaler with_mean/with_std ignored (NOT-STARTED, R-DEV-1/-2)** — the
   wrapper carries `with_mean`/`with_std` params but `fit` always constructs the
   no-arg `_RsStandardScaler()` (which always centers AND scales) and always sets
   `mean_`/`scale_` from the Rust arrays. So `ferrolearn.StandardScaler(with_mean=False)`
   still centers and exposes a non-None `mean_`, whereas sklearn gives `mean_=None`
   + an identity `transform`. Partly wrapper-fixable but the underlying knob is
   owned downstream (blocker #1193).
4. **missing constructor params (NOT-STARTED, R-DEV-2)** — StandardScaler lacks
   `copy`; PCA lacks `copy`/`whiten`/`svd_solver`/`tol`/`iterated_power`/
   `n_oversamples`/`power_iteration_normalizer`/`random_state`. Behavior owned
   downstream (StandardScaler #1193; PCA #1502/#1503/#1509).
5. **missing fitted attrs (SHIPPED #2097, R-DEV-3)** — PCA now exposes
   `n_components_` (int) and `noise_variance_` (float) via the `RsPCA` getters
   over `FittedPCA::n_components_()`/`noise_variance()`, set by
   `_transformers.py::PCA.fit` (downstream prereqs #1505/#1507 SHIPPED).
6. **value parity off the default path (NOT-STARTED, R-DEV-1)** — exact array
   parity for non-default `with_mean`/`with_std`, `whiten`, alternate
   `svd_solver`, and the PCA repeated-eigenvalue sign/basis case is owned
   downstream (StandardScaler #1193; PCA #1501/#1502/#1503).
7. **substrate (NOT-STARTED, R-SUBSTRATE-1)** — the binding round-trips numpy ↔
   `ndarray` via `crate::conversions::*` (rust-numpy), not
   `ferray::numpy_interop`/`ferray-core`; owned by `conversions.md` #2027.

## Upstream reference (sklearn 1.5.2, live oracle = installed sklearn 1.5.2)

Lines stable at tag 1.5.2 / commit 156ef14.

### `sklearn.preprocessing.StandardScaler` (`_data.py:696`)

- **`__init__`** (`_data.py:835`):
  `StandardScaler(self, *, copy=True, with_mean=True, with_std=True)`. ALL params
  are keyword-only (the `*` is first). `_parameter_constraints`
  (`_data.py:829-833`): `"copy": ["boolean"]`, `"with_mean": ["boolean"]`,
  `"with_std": ["boolean"]`.
- **`fit`** (`_data.py:853`) delegates to `partial_fit` (`_data.py:881`), which
  sets `mean_`/`var_`/`scale_`/`n_samples_seen_`/`n_features_in_`. When
  `with_mean=False` AND `with_std=False`, both `mean_` and `var_` are set to
  `None` (`_data.py:993-995`).
- **`transform`** (`_data.py:1027`): dense path subtracts `mean_` only if
  `with_mean`, divides by `scale_` only if `with_std` (`_data.py:1064-1067`) —
  so `with_mean=False`/`with_std=False` is the identity.
- **`inverse_transform`** (`_data.py:1070`).
- **attributes** (`_data.py:756-790`): `scale_` `(n_features,)` or None,
  `mean_` `(n_features,)` or None, `var_` `(n_features,)` or None,
  `n_samples_seen_` int (or array), `n_features_in_` int.

### `sklearn.decomposition.PCA(_BasePCA)` (`_pca.py:121`)

- **`__init__`** (`_pca.py:407-423`):
  `PCA(self, n_components=None, *, copy=True, whiten=False, svd_solver='auto',
  tol=0.0, iterated_power='auto', n_oversamples=10,
  power_iteration_normalizer='auto', random_state=None)`. `n_components` is
  positional-or-keyword (it PRECEDES the `*`), default **None** (keep
  `min(n_samples, n_features)` components); everything after `*` is keyword-only.
  `_parameter_constraints` (`_pca.py:393-405`): `"whiten": ["boolean"]`,
  `"svd_solver": [StrOptions({...})]`, etc.
- **`_fit`** (`_pca.py:489`) dispatches to `_fit_full`/`_fit_truncated`/
  `covariance_eigh`; `svd_flip(U, Vt, u_based_decision=False)` fixes component
  signs (`_pca.py:647`/`:760`/`:773`).
- **attributes** (`_pca.py:691-707`/`:776-801`): `components_`
  `(n_components_, n_features)`, `explained_variance_` `(n_components_,)`,
  `explained_variance_ratio_` `(n_components_,)`, `singular_values_`
  `(n_components_,)`, `mean_` `(n_features,)`, `n_components_` int,
  `noise_variance_` float, `n_features_in_` int.

Live oracle (installed sklearn 1.5.2, run from `/tmp`; R-CHAR-3 — expected values
from sklearn, NEVER from ferrolearn):

```
StandardScaler.__init__ sig -> (self, *, copy=True, with_mean=True, with_std=True)
PCA.__init__ sig            -> (self, n_components=None, *, copy=True, whiten=False,
                                svd_solver='auto', tol=0.0, iterated_power='auto',
                                n_oversamples=10, power_iteration_normalizer='auto',
                                random_state=None)

X = [[0,0],[1,1],[2,4],[3,9]]
StandardScaler().fit(X):  mean_=[1.5,3.5]  scale_=[1.1180339887,3.5]
                          var_=[1.25,12.25]  n_samples_seen_=4  n_features_in_=2
StandardScaler(with_mean=False, with_std=False).fit(X):
                          mean_=None  scale_=None  transform(X) == X (identity)

PCA().n_components -> None ;  PCA(2).n_components -> 2 (positional)
PCA().fit(X).components_.shape -> (2,2) ;  n_components_ -> 2 ;  noise_variance_ -> 0.0
PCA(2).fit(X):  explained_variance_=[17.87568464, 0.12431536]
                explained_variance_ratio_=[0.99309359, 0.00690641]
                singular_values_=[7.3230495, 0.61069312]  mean_=[1.5,3.5]
                components_=[[0.29476487,0.95556982],[0.95556982,-0.29476487]]
```

ferrolearn at baseline `be3362243` (live):
- `StandardScaler().fit(X)` exposes `mean_=[1.5,3.5]`, `scale_=[1.1180339887,3.5]`,
  `var_=[1.25,12.25]`, `n_samples_seen_=4` — MATCHES the oracle default path.
- `StandardScaler(with_mean=False, with_std=False).fit(X)` still gives
  `mean_=[1.5,3.5]` and `transform(X) != X` — DIVERGES (sklearn → None + identity).
- `PCA(2)` raises `TypeError: PCA.__init__() takes 1 positional argument but 2
  were given`; `PCA().n_components == 2` (not None).
- `PCA(n_components=2).fit(X)` `explained_variance_`/`explained_variance_ratio_`/
  `singular_values_`/`mean_`/`components_` (including sign) MATCH the oracle
  element-wise; `hasattr(fitted, 'n_components_')` and `'noise_variance_'` are
  both `False`.

## Requirements

Grouped by estimator (`REQ-SS-*`, `REQ-PCA-*`) plus shared `REQ-CONSUMER`/
`REQ-SUBSTRATE`.

### StandardScaler

- REQ-SS-API-CONFORM: `ferrolearn.StandardScaler` exposes the
  `sklearn.preprocessing.StandardScaler` method surface —
  `fit`/`transform`/`inverse_transform` (bound on `_RsStandardScaler` in
  `transformers.rs`, wrapped in `_transformers.py`) plus `fit_transform`
  (inherited from `TransformerMixin`) — and the fitted attributes `mean_`
  `(n_features,)`, `scale_` `(n_features,)`, `var_` `(n_features,)`,
  `n_samples_seen_` (int), `n_features_in_` (int, set by `_validate_data`), with
  values matching the sklearn oracle element-wise ON THE DEFAULT
  `with_mean=True`/`with_std=True` PATH (`_data.py:756-790` attrs,
  `:1064-1067` transform). `check_estimator(StandardScaler())` passes.
- REQ-SS-WITH-MEAN-STD: `ferrolearn.StandardScaler(with_mean=False)` /
  `(with_std=False)` change the behavior the way sklearn does — `with_mean=False`
  skips centering, `with_std=False` skips scaling, and `with_mean=False` AND
  `with_std=False` make `transform` the identity with `mean_`/`var_` set to None
  (`_data.py:993-995`, `:1064-1067`). [Owned downstream by
  `ferrolearn-preprocess` REQ-6, blocker #1193 — the underlying
  `ferrolearn_preprocess::StandardScaler` always centers+scales; the wrapper
  ignores the params.]
- REQ-SS-COPY: `ferrolearn.StandardScaler` exposes the `copy` constructor param
  (bool, default `True`, `_data.py:835`/`:829`). [Param surface; behavior owned
  downstream #1193.]
- REQ-SS-VALUE-PARITY: `mean_`/`scale_`/`var_`/`n_samples_seen_` match sklearn
  array-by-array on a fixed dataset (R-DEV-1) on the DEFAULT path — the way users
  compare ferrolearn outputs to sklearn. [Default path is SHIPPED; non-default
  `with_mean`/`with_std` values are owned downstream #1193.]

### PCA

- REQ-PCA-API-CONFORM: `ferrolearn.PCA` exposes the
  `sklearn.decomposition.PCA` method surface —
  `fit`/`transform`/`inverse_transform` (bound on `_RsPCA` in `transformers.rs`,
  wrapped in `_transformers.py`) plus `fit_transform` (inherited from
  `TransformerMixin`) — and the fitted attributes `components_`
  `(n_components, n_features)`, `explained_variance_` `(n_components,)`,
  `explained_variance_ratio_` `(n_components,)`, `singular_values_`
  `(n_components,)`, `mean_` `(n_features,)`, with values matching the sklearn
  oracle element-wise INCLUDING the `svd_flip` sign convention, on the DEFAULT
  `svd_solver`/`whiten=False` path (`_pca.py:691-707` attrs, `:647` svd_flip).
  `check_estimator(PCA(n_components=2))` passes.
- REQ-PCA-NCOMP-POSITIONAL: `ferrolearn.PCA` accepts `n_components`
  positionally — `PCA(2)` constructs an estimator with `n_components == 2`,
  matching sklearn `__init__(self, n_components=None, *, ...)` (`_pca.py:407-409`,
  `n_components` before the `*`).
- REQ-PCA-NCOMP-DEFAULT-NONE: `ferrolearn.PCA`'s `n_components` default is `None`
  (keep `min(n_samples, n_features)` components, resolved at fit), matching
  sklearn (`_pca.py:409`; oracle `PCA().n_components` → `None`,
  `PCA().fit(X).components_.shape` → `(2,2)` for the 4×2 X). ferrolearn defaults
  to `2`.
- REQ-PCA-PARAMS: `ferrolearn.PCA` exposes the `copy`/`whiten`/`svd_solver`/`tol`/
  `iterated_power`/`n_oversamples`/`power_iteration_normalizer`/`random_state`
  constructor params (`_pca.py:407-423`/`:393-405`). [Param surface + behavior
  owned downstream: `whiten` `ferrolearn-decomp` REQ-11 #1502; `svd_solver` REQ-12
  #1503; the remaining ctor params REQ-17 #1509.]
- REQ-PCA-ATTRS: `ferrolearn.PCA` exposes `n_components_` (int) and
  `noise_variance_` (float) fitted attrs (`_pca.py:691`/`:686-688`). [SHIPPED
  #2097 — the binding surfaces `FittedPCA::n_components_()`/`noise_variance()`
  (downstream prereqs `ferrolearn-decomp` REQ-16 #1505, REQ-15 #1507 SHIPPED),
  set by `_transformers.py::PCA.fit`.]
- REQ-PCA-VALUE-PARITY: `components_`/`explained_variance_`/
  `explained_variance_ratio_`/`singular_values_`/`mean_` match sklearn
  array-by-array (R-DEV-1), including the deterministic `svd_flip` sign on the
  default solver path. [Default-solver value parity is SHIPPED (downstream
  `ferrolearn-decomp` REQ-1/4/5 confirm element-wise to 1e-6 incl. sign); the
  repeated-eigenvalue/rank-deficient sign-and-basis case is owned downstream
  REQ-2 #1501, and `whiten`/alternate-`svd_solver` value parity downstream
  #1502/#1503.]

### Shared

- REQ-CONSUMER: the binding IS the public API (R-DEFER-1/S5: boundary estimator
  types ARE the public surface, grandfathered existing pub API); its non-test
  production consumers are the Python wrappers `_transformers.py::StandardScaler`/
  `PCA` + the `ferrolearn/__init__.py` re-export, exercised by the pytest gauntlet
  (`test_check_estimator.py`).
- REQ-SUBSTRATE: the binding's array marshalling is on `ferray::numpy_interop`
  producing `ferray-core` arrays, not rust-numpy + `ndarray` (R-SUBSTRATE-1).
  [Owned by `conversions.md` REQ-FERRAY #2027.]

## Acceptance criteria

All expected values come from the live sklearn 1.5.2 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn. The pytest gauntlet
(`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`) is the
end-to-end check (verification model B); rebuild first if the Rust side changed
(`cd ferrolearn-python && maturin develop`).

- AC-SS-API-CONFORM (REQ-SS-API-CONFORM): `test_check_estimator.py`
  (`parametrize_with_checks([..., StandardScaler()])`) passes — the binding
  marshals fit/transform/inverse_transform/attrs. Spot oracle (default path,
  values + shapes):
  `cd /tmp && python3 -c "import numpy as np; from sklearn.preprocessing import StandardScaler; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); m=StandardScaler().fit(X); print(m.mean_.tolist(), [round(v,10) for v in m.scale_], m.var_.tolist(), m.n_samples_seen_, m.n_features_in_)"`
  → `[1.5, 3.5] [1.1180339887, 3.5] [1.25, 12.25] 4 2`. `ferrolearn.StandardScaler().fit(X)`
  exposes the SAME `mean_`/`scale_`/`var_`/`n_samples_seen_` (live-confirmed),
  `n_features_in_ == 2`, and `fit_transform` present.
- AC-SS-WITH-MEAN-STD (REQ-SS-WITH-MEAN-STD): sklearn oracle
  `cd /tmp && python3 -c "import numpy as np; from sklearn.preprocessing import StandardScaler; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); m=StandardScaler(with_mean=False, with_std=False).fit(X); print(m.mean_, m.scale_, np.allclose(m.transform(X), X))"`
  → `None None True`. ferrolearn DIVERGES (live: `mean_=[1.5,3.5]`,
  `transform(X) != X`). A critic pins a FAILING pytest asserting
  `StandardScaler(with_mean=False, with_std=False).fit(X).mean_ is None` and
  `transform` identity. FAILS until the wrapper honors the params AND the
  downstream `ferrolearn_preprocess::StandardScaler` exposes a with_mean/with_std
  knob (blocker #1193).
- AC-SS-COPY (REQ-SS-COPY): sklearn exposes `copy`
  (`cd /tmp && python3 -c "import inspect; from sklearn.preprocessing import StandardScaler; print('copy' in inspect.signature(StandardScaler.__init__).parameters)"`
  → `True`). ferrolearn signature has none (live:
  `(self, *, with_mean=True, with_std=True)`). A critic pins a FAILING pytest
  asserting `'copy' ∈ inspect.signature(ferrolearn.StandardScaler.__init__).parameters`.
  FAILS until the param is added.
- AC-PCA-API-CONFORM (REQ-PCA-API-CONFORM): `test_check_estimator.py`
  (`parametrize_with_checks([..., PCA(n_components=2)])`) passes. Spot oracle
  (default-solver values + sign):
  `cd /tmp && python3 -c "import numpy as np; from sklearn.decomposition import PCA; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); m=PCA(2).fit(X); print([round(v,8) for v in m.explained_variance_], [round(v,8) for v in m.explained_variance_ratio_], [round(v,8) for v in m.singular_values_], np.round(m.components_,8).tolist())"`
  → `[17.87568464, 0.12431536] [0.99309359, 0.00690641] [7.3230495, 0.61069312]
  [[0.29476487, 0.95556982], [0.95556982, -0.29476487]]`. `ferrolearn.PCA(n_components=2).fit(X)`
  matches all four arrays incl. sign (live-confirmed), `mean_ == [1.5,3.5]`,
  `fit_transform` present.
- AC-PCA-NCOMP-POSITIONAL (REQ-PCA-NCOMP-POSITIONAL): sklearn oracle
  `cd /tmp && python3 -c "from sklearn.decomposition import PCA; print(PCA(2).n_components)"`
  → `2`. ferrolearn FAILS:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "from ferrolearn import PCA; PCA(2)"`
  → `TypeError: PCA.__init__() takes 1 positional argument but 2 were given`
  (live-confirmed). A critic pins a FAILING pytest asserting `PCA(2)` constructs
  `n_components == 2`. FAILS until `_transformers.py::PCA.__init__` moves
  `n_components` before the `*`.
- AC-PCA-NCOMP-DEFAULT-NONE (REQ-PCA-NCOMP-DEFAULT-NONE): sklearn oracle
  `cd /tmp && python3 -c "import inspect, numpy as np; from sklearn.decomposition import PCA; print(inspect.signature(PCA.__init__).parameters['n_components'].default, PCA().fit(np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]])).components_.shape)"`
  → `None (2, 2)`. ferrolearn default is `2` (live:
  `inspect.signature(ferrolearn.PCA.__init__)` →
  `(self, *, n_components=2)`). A critic pins a FAILING pytest asserting
  `PCA().n_components is None` and `PCA().fit(X).components_.shape == (2,2)`.
  FAILS until the wrapper defaults to `None` and resolves it at fit.
- AC-PCA-PARAMS (REQ-PCA-PARAMS): sklearn exposes the extra ctor params
  (`cd /tmp && python3 -c "import inspect; from sklearn.decomposition import PCA; ps=inspect.signature(PCA.__init__).parameters; print([p for p in ('copy','whiten','svd_solver','tol','iterated_power','n_oversamples','power_iteration_normalizer','random_state') if p in ps])"`
  → all 8). ferrolearn has NONE. A critic pins FAILING pytests asserting each ∈
  `inspect.signature(ferrolearn.PCA.__init__).parameters`. FAIL until the binding +
  wrapper add them (behavior owned by `ferrolearn-decomp` #1502/#1503/#1509).
- AC-PCA-ATTRS (REQ-PCA-ATTRS): SHIPPED #2097. sklearn oracle on a rank>2
  fixture (so `noise_variance_` is meaningfully non-zero)
  `cd /tmp && python3 -c "import numpy as np; from sklearn.decomposition import PCA; m=PCA(2).fit(np.array([[1.,2.,3.],[4.,5.,7.],[2.,0.,1.],[8.,6.,5.],[3.,3.,2.],[0.,1.,4.]])); print(m.n_components_, m.noise_variance_)"`
  → `2 0.3132465241238894`. `ferrolearn.PCA(n_components=2).fit(X)` now exposes
  `n_components_ == 2` and `noise_variance_` matching the oracle to <1e-9 (via
  `RsPCA::n_components_`/`noise_variance_` over `FittedPCA::n_components_()`/
  `noise_variance()`, set in `_transformers.py::PCA.fit`). Guard
  `tests/divergence_transformers.py::test_pca_n_components_and_noise_variance_match_sklearn`.
- AC-PCA-VALUE-PARITY (REQ-PCA-VALUE-PARITY): the default-solver arrays match
  element-wise (covered by AC-PCA-API-CONFORM). The repeated-eigenvalue
  sign/basis case (`whiten`, alternate `svd_solver`) does NOT — a critic pins a
  FAILING array-by-array pytest on a degenerate fixture; it FAILS until the
  downstream blockers land (`ferrolearn-decomp` REQ-2 #1501, REQ-11 #1502,
  REQ-12 #1503).
- AC-CONSUMER (REQ-CONSUMER):
  `grep -rn "_RsStandardScaler\|_RsPCA" /home/doll/ferrolearn/ferrolearn-python/python/ferrolearn/_transformers.py`
  shows `_transformers.py::StandardScaler` constructs `_RsStandardScaler()` /
  `PCA` constructs `_RsPCA(n_components=...)` and drives fit/transform/
  inverse_transform + reads the fitted attrs; `ferrolearn/__init__.py:12`
  re-exports both; `test_check_estimator.py:31-32` runs `StandardScaler()` and
  `PCA(n_components=2)` through `parametrize_with_checks`. The 524-passing pytest
  exercises the consumer surface.
- AC-SUBSTRATE (REQ-SUBSTRATE): `transformers.rs` head shows
  `use crate::conversions::*` + `use numpy::{PyArray1, PyArray2,
  PyReadonlyArray2}` — the wrong substrate per R-SUBSTRATE-1 (destination
  `ferray::numpy_interop`/`ferray-core`). ferray exposes no `numpy_interop`
  bridge consumable here (R-SUBSTRATE-5). Owned by `conversions.md` #2027.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-SS-API-CONFORM (StandardScaler fit/transform/inverse_transform + attrs, default path) | SHIPPED | impl `RsStandardScaler::fit`/`transform`/`inverse_transform` + getters `mean_`/`scale_` in `transformers.rs` (over `ferrolearn_preprocess::FittedStandardScaler<f64>`, via `fitted.mean()`/`fitted.std()`), wrapped by `StandardScaler` in `_transformers.py` which sets `mean_`/`scale_` from the Rust getters and derives `var_ = scale_**2` + `n_samples_seen_ = X.shape[0]` + `n_features_in_` (via `self._validate_data`), inheriting `fit_transform` from `TransformerMixin` — mirroring `sklearn/preprocessing/_data.py:756-790` (attrs) + `:1027`/`:1070` (methods). Non-test consumer: `_transformers.py::StandardScaler` + `ferrolearn/__init__.py:12` re-export; external users. Verification (model B): `cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` → 524 passed, 0 failed (`test_check_estimator.py:31` runs `StandardScaler()` through `parametrize_with_checks`). Live default-path oracle MATCHES element-wise: `mean_=[1.5,3.5]`, `scale_=[1.1180339887,3.5]`, `var_=[1.25,12.25]`, `n_samples_seen_=4`, `n_features_in_=2`. |
| REQ-SS-WITH-MEAN-STD (with_mean/with_std honored) | SHIPPED (#2037) | FIXED — the downstream prereq #1193 (`ferrolearn-preprocess::StandardScaler` with_mean/with_std/copy builders) was already SHIPPED, so the binding now threads the flags through. `RsStandardScaler::new(with_mean, with_std, copy)` (`transformers.rs`, `#[pyo3(signature=(with_mean=true,with_std=true,copy=true))]`) builds `StandardScaler::new().with_with_mean(..).with_with_std(..).with_copy(..)`; `_transformers.py::StandardScaler.fit` passes the flags and nulls attributes per sklearn (`mean_=array if (with_mean or with_std) else None`, `scale_=array if with_std else None`, `var_=scale_**2 if with_std else None`, `_data.py:1064-1067`). Live oracle (R-CHAR-3) `X=[[1,10],[2,20],[3,30]]`: (T,T) `mean_=[2,20]`/`scale_=[.8165,8.165]`/`var_=[.6667,66.67]`/`t[0]=[-1.2247,-1.2247]`; (T,F) `mean_=[2,20]`/`scale_=None`/`var_=None`/`t[0]=[-1,-10]`; (F,T) `mean_=[2,20]`/`scale_=[.8165,8.165]`/`t[0]=[1.2247,1.2247]`; (F,F) `mean_=None`/`scale_=None`/`var_=None`/`t[0]=[1,10]` — all match. Consumer: `_transformers.py::StandardScaler`. Guards: 4-config parametrized pytest in `tests/divergence_transformers.py` (20 passed). |
| REQ-SS-COPY (copy ctor param) | SHIPPED (#2037) | FIXED — `_transformers.py::StandardScaler.__init__(self, *, copy=True, with_mean=True, with_std=True)` now exposes `copy` (sklearn param order, `_data.py:835`), threaded into `_RsStandardScaler(with_mean, with_std, copy)` → `StandardScaler::with_copy`. sklearn `copy` bool default `True`; non-mutation already held via `_validate_data`/`_ensure_f64`, now the constructor-ABI param is present. Guard: `copy=True` inverse round-trip + signature check in `tests/divergence_transformers.py`. |
| REQ-SS-VALUE-PARITY (mean_/scale_/var_/n_samples_seen_ array parity) | SHIPPED | on the DEFAULT path. `_transformers.py::StandardScaler.fit` marshals `mean_`/`scale_` from `RsStandardScaler::mean_`/`scale_` getters (over `FittedStandardScaler::mean()`/`std()`), and the downstream `ferrolearn-preprocess` REQ-1 is critic-verified bit-identical to the live sklearn oracle (`green_req1_value_match_non_constant`, `green_req1_mean_and_scale_attributes`). Live (R-CHAR-3): ferrolearn `StandardScaler().fit(X)` `mean_`/`scale_`/`var_`/`n_samples_seen_` equal the sklearn oracle element-wise. Non-test consumer: `_transformers.py::StandardScaler` + re-export. (Non-default `with_mean`/`with_std` value parity is REQ-SS-WITH-MEAN-STD, NOT-STARTED #1193.) |
| REQ-PCA-API-CONFORM (PCA fit/transform/inverse_transform + 5 attrs, default solver) | SHIPPED | impl `RsPCA::fit`/`transform`/`inverse_transform` + getters `components_`/`explained_variance_`/`explained_variance_ratio_`/`mean_`/`singular_values_` in `transformers.rs` (over `ferrolearn_decomp::FittedPCA<f64>`), wrapped by `PCA` in `_transformers.py` which copies all five attrs from the Rust getters and sets `n_features_in_` (via `self._validate_data`), inheriting `fit_transform` from `TransformerMixin` — mirroring `sklearn/decomposition/_pca.py:691-707` (attrs) + `:489` (`_fit`), with the `svd_flip` sign convention (`_pca.py:647`) matched downstream. Non-test consumer: `_transformers.py::PCA` + `ferrolearn/__init__.py:12` re-export. Verification (model B): pytest → 524 passed (`test_check_estimator.py:32` runs `PCA(n_components=2)` through `parametrize_with_checks`). Live default-solver oracle MATCHES element-wise incl. sign: `explained_variance_=[17.87568464,0.12431536]`, `explained_variance_ratio_=[0.99309359,0.00690641]`, `singular_values_=[7.3230495,0.61069312]`, `mean_=[1.5,3.5]`, `components_=[[0.29476487,0.95556982],[0.95556982,-0.29476487]]`. |
| REQ-PCA-NCOMP-POSITIONAL (n_components positional ABI) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 constructor ABI; single-wrapper-fixable). sklearn `__init__(self, n_components=None, *, ...)` (`_pca.py:407-409`) makes `n_components` positional-or-keyword — `PCA(2).n_components` → `2`. ferrolearn `_transformers.py::PCA.__init__(self, *, n_components=2)` makes it keyword-only — live: `ferrolearn.PCA(2)` → `TypeError: __init__() takes 1 positional argument but 2 were given`. Single-line Python-wrapper fix: move `n_components` before the `*`. |
| REQ-PCA-NCOMP-DEFAULT-NONE (n_components default None) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 default value; single-wrapper-fixable). sklearn default `n_components=None` keeps `min(n_samples, n_features)` components (`_pca.py:409`; oracle `PCA().n_components` → `None`, `PCA().fit(X).components_.shape` → `(2,2)`). ferrolearn defaults to `2` (`_transformers.py::PCA.__init__` + `RsPCA::new` `#[pyo3(signature = (n_components=2))]`). Wrapper-level fix: default `None` and resolve to `min(n,p)` before constructing `_RsPCA` (the Rust binding needs a usize, so the wrapper resolves None at fit). |
| REQ-PCA-PARAMS (copy/whiten/svd_solver/tol/iterated_power/n_oversamples/power_iteration_normalizer/random_state) | NOT-STARTED | open prereq blockers #1502 (`whiten`, `ferrolearn-decomp` REQ-11), #1503 (`svd_solver`, REQ-12), #1509 (`tol`/`iterated_power`/`n_oversamples`/`power_iteration_normalizer`/`random_state`/`copy`, REQ-17). sklearn `_pca.py:407-423`/`:393-405`. ferrolearn `_transformers.py::PCA.__init__` exposes `n_components` only; `RsPCA::new` takes only `n_components`. The default `svd_solver`/`whiten=False` behavior MATCHES (covariance-eigh, no whitening), so only the param surface + non-default paths are missing — owned downstream; the binding cannot expose what the library lacks. |
| REQ-PCA-ATTRS (n_components_ + noise_variance_) | SHIPPED (#2097) | the downstream prereqs already SHIPPED (`ferrolearn-decomp` REQ-16 #1505 `n_components_`, REQ-15 #1507 `noise_variance_`), so the binding now exposes both. impl: `RsPCA::n_components_` getter marshals `FittedPCA::n_components_()` (`ferrolearn-decomp/src/pca.rs::n_components_`, the row count of `components_`, `_pca.py:691`); `RsPCA::noise_variance_` getter marshals `FittedPCA::noise_variance()` (`pca.rs::noise_variance`, full-spectrum tail mean `mean(sorted_eigenvalues[n_comp..min_dim])` or `0.0` when all kept, `_pca.py:686-688`). Non-test consumer: `_transformers.py::PCA.fit` sets `n_components_ = int(self._rs.n_components_)` + `noise_variance_ = float(self._rs.noise_variance_)`. Live oracle (R-CHAR-3) `X=[[1,2,3],[4,5,7],[2,0,1],[8,6,5],[3,3,2],[0,1,4]]`: `PCA(2)` → `n_components_=2`, `noise_variance_=0.3132465241238894` (NON-ZERO — tail-mean path); `PCA(3)` → `noise_variance_=0.0`. Guard `tests/divergence_transformers.py::test_pca_n_components_and_noise_variance_match_sklearn`. |
| REQ-PCA-VALUE-PARITY (components_/explained_variance_/ratio/singular_values_/mean_ array parity incl. svd_flip sign) | SHIPPED | on the DEFAULT solver path. `_transformers.py::PCA.fit` marshals all five arrays from the `RsPCA` getters (over `FittedPCA`), and the downstream `ferrolearn-decomp` REQ-1/4/5 are critic-verified to MATCH the live sklearn `PCA` oracle element-wise to 1e-6 INCLUDING the per-row `svd_flip(u_based_decision=False)` sign (`tests/divergence_pca.rs`, `_pca.py:647`). Live (R-CHAR-3): ferrolearn `PCA(n_components=2).fit(X)` equals the oracle element-wise (values above). Non-test consumer: `_transformers.py::PCA` + re-export. (The repeated-eigenvalue/rank-deficient sign-and-basis case is owned downstream REQ-2 #1501; `whiten`/alternate-`svd_solver` value parity is owned downstream #1502/#1503.) |
| REQ-CONSUMER (binding IS the public API) | SHIPPED | the binding boundary types ARE the public API (R-DEFER-1/S5: boundary estimator types ARE the public surface; grandfathered existing pub API). Non-test production consumers: `_transformers.py::StandardScaler` constructs `_RsStandardScaler()` and `_transformers.py::PCA` constructs `_RsPCA(n_components=...)`, each calling `fit`/`transform`/`inverse_transform` + reading the fitted-attr getters (`grep -n "_RsStandardScaler\|_RsPCA" python/ferrolearn/_transformers.py`); `ferrolearn/__init__.py:12` re-exports both `StandardScaler` and `PCA`; `test_check_estimator.py:31-32` is the verification consumer + external users. Verification (model B): pytest → 524 passed (both estimators run through `parametrize_with_checks`). |
| REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | open prereq blocker = `conversions.md` REQ-FERRAY #2027. `transformers.rs` marshals via `use crate::conversions::*` + `use numpy::{PyArray1, PyArray2, PyReadonlyArray2}` (rust-numpy) and the conversions produce `ndarray::Array{1,2}` — the WRONG substrate per R-SUBSTRATE-1 (destination `ferray::numpy_interop` + `ferray-core`). ferray exposes no PyO3 numpy-interop bridge yet (R-SUBSTRATE-5). Owned by the conversions unit, surfaced here. |

## Architecture

`transformers.rs` holds two `#[pyclass]` structs, each a THIN shim over a fitted
library type and owning ZERO transformer math:

- **`RsStandardScaler`** wraps `Option<ferrolearn_preprocess::FittedStandardScaler<f64>>`.
  `new()` takes NO params (mirroring `ferrolearn_preprocess::StandardScaler::new`,
  which has no with_mean/with_std knob); `fit` runs the no-arg
  `StandardScaler::<f64>::new()` on the `numpy2_to_ndarray`-coerced X and stores
  the fitted model (mapping `FerroError` → `PyValueError`);
  `transform`/`inverse_transform` and the `mean_`/`scale_` getters delegate to the
  stored `FittedStandardScaler` (`fitted.mean()`/`fitted.std()`), returning
  `PyRuntimeError("not fitted")` before fit.
- **`RsPCA`** stores a `n_components: usize` plus
  `Option<ferrolearn_decomp::FittedPCA<f64>>`. `new`
  (`#[pyo3(signature = (n_components=2))]`) stores the count; `fit` builds
  `PCA::<f64>::new(self.n_components)`, runs it on the coerced X, stores the fitted
  model; `transform`/`inverse_transform` and the five getters
  (`components_`/`explained_variance_`/`explained_variance_ratio_`/`mean_`/
  `singular_values_`) delegate to `FittedPCA`. There is NO `n_components_`/
  `noise_variance_` getter.

`_transformers.py` wraps each as a sklearn-facing estimator:

- **`StandardScaler(TransformerMixin, BaseEstimator)`** —
  `__init__(self, *, with_mean=True, with_std=True)` STORES `with_mean`/`with_std`
  but `fit` always constructs the no-arg `_RsStandardScaler()` and sets
  `mean_`/`scale_` from the Rust getters, `var_ = scale_**2`,
  `n_samples_seen_ = X.shape[0]`. The `with_mean`/`with_std` fallback branches in
  `transform`/`inverse_transform` are gated on `not hasattr(self, "_rs")` and are
  DEAD (`_rs` always present after `fit`/unpickle). So the params are accepted but
  IGNORED — the REQ-SS-WITH-MEAN-STD divergence. No `copy` param (REQ-SS-COPY).
- **`PCA(TransformerMixin, BaseEstimator)`** —
  `__init__(self, *, n_components=2)`: keyword-only `*` (REQ-PCA-NCOMP-POSITIONAL)
  AND default `2` not `None` (REQ-PCA-NCOMP-DEFAULT-NONE). `fit` constructs
  `_RsPCA(n_components=self.n_components)` and copies the five attrs; there are no
  `copy`/`whiten`/`svd_solver`/… params (REQ-PCA-PARAMS) and no `n_components_`/
  `noise_variance_` attrs (REQ-PCA-ATTRS). `__getstate__`/`__setstate__` drop and
  rebuild `_rs` for pickling.

The honest call (R-HONEST-3): the API-CONFORMANCE, the default-path VALUE parity
(both estimators), and the CONSUMER ship on impl + the 524-passing pytest
(`check_estimator` for `StandardScaler()` and `PCA(n_components=2)`) + the live
default-path oracle matching element-wise. Everything else is NOT-STARTED (binary,
R-DEFER-2) and splits into:

- **Single-wrapper-fixable PCA ABI divergences (the headline)**:
  REQ-PCA-NCOMP-POSITIONAL (move `n_components` before `*` — one line) and
  REQ-PCA-NCOMP-DEFAULT-NONE (default `2` → `None`, resolved to `min(n,p)` at
  fit). These are pure Python-wrapper constructor fixes needing no Rust change.
- **Wrapper-surfaced, downstream-owned behavior gaps**:
  REQ-SS-WITH-MEAN-STD (the Rust scaler has no with_mean/with_std knob — #1193),
  REQ-SS-COPY (#1193), REQ-PCA-PARAMS (`whiten` #1502, `svd_solver` #1503, the
  rest #1509), REQ-PCA-ATTRS (`n_components_` #1508, `noise_variance_` #1507). The
  DEFAULTS match (always-center+scale = sklearn's with_mean=True/with_std=True;
  covariance-eigh, whiten=False), so only the param/attr surface + non-default
  paths are missing — and the binding cannot expose what the library crate lacks.
- **Off-default value parity**: REQ-SS-VALUE-PARITY and REQ-PCA-VALUE-PARITY are
  SHIPPED on the DEFAULT path; the non-default `with_mean`/`with_std`,
  `whiten`/alternate-`svd_solver`, and PCA repeated-eigenvalue sign/basis cases
  are owned downstream (#1193; #1501/#1502/#1503).
- **Substrate**: REQ-SUBSTRATE — rust-numpy + `ndarray`, not
  `ferray::numpy_interop` (owned by `conversions.md` #2027).

The least-confident SHIPPED claim is REQ-PCA-VALUE-PARITY: it is SHIPPED only on
the DEFAULT covariance-eigh solver with distinct eigenvalues, where the downstream
`svd_flip` sign is deterministic and verified element-wise to 1e-6. On a
repeated-eigenvalue or rank-deficient fixture, faer/LAPACK pick a different
orthonormal basis and the sign/basis diverges from sklearn — that case is honestly
carved out to the downstream REQ-2 #1501. So this REQ ships the *default-solver,
distinct-eigenvalue* contract, not universal array parity (R-HONEST-3 underclaim).

## Verification

Commands establishing the SHIPPED claims (run at baseline `be3362243`,
verification model B; rebuild first if the Rust side changed:
`cd /home/doll/ferrolearn/ferrolearn-python && maturin develop`):

- **Full pytest gauntlet (REQ-SS-API-CONFORM, REQ-PCA-API-CONFORM,
  REQ-*-VALUE-PARITY default path, REQ-CONSUMER):**
  `cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`
  → `524 passed`. `test_check_estimator.py` runs `StandardScaler()` and
  `PCA(n_components=2)` through sklearn's `parametrize_with_checks`.
- **StandardScaler default-path oracle (REQ-SS-API-CONFORM, REQ-SS-VALUE-PARITY;
  R-CHAR-3):**
  `cd /tmp && python3 -c "import numpy as np; from sklearn.preprocessing import StandardScaler; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); m=StandardScaler().fit(X); print(m.mean_.tolist(), [round(v,10) for v in m.scale_], m.var_.tolist(), m.n_samples_seen_, m.n_features_in_)"`
  → `[1.5, 3.5] [1.1180339887, 3.5] [1.25, 12.25] 4 2`. ferrolearn matches
  element-wise (live).
- **StandardScaler with_mean/with_std oracle (REQ-SS-WITH-MEAN-STD):**
  `cd /tmp && python3 -c "import numpy as np; from sklearn.preprocessing import StandardScaler; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); m=StandardScaler(with_mean=False, with_std=False).fit(X); print(m.mean_, m.scale_, np.allclose(m.transform(X), X))"`
  → `None None True`. ferrolearn DIVERGES (`mean_=[1.5,3.5]`, `transform(X) != X`).
  A critic pins a FAILING pytest; FAIL until #1193 lands in `ferrolearn-preprocess`
  AND the wrapper honors the params.
- **StandardScaler copy oracle (REQ-SS-COPY):**
  `cd /tmp && python3 -c "import inspect; from sklearn.preprocessing import StandardScaler; print('copy' in inspect.signature(StandardScaler.__init__).parameters)"`
  → `True`; ferrolearn signature `(self, *, with_mean=True, with_std=True)` has
  none. FAIL until the param is added (downstream #1193).
- **PCA default-solver oracle (REQ-PCA-API-CONFORM, REQ-PCA-VALUE-PARITY;
  R-CHAR-3):**
  `cd /tmp && python3 -c "import numpy as np; from sklearn.decomposition import PCA; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]]); m=PCA(2).fit(X); print([round(v,8) for v in m.explained_variance_], [round(v,8) for v in m.singular_values_], np.round(m.components_,8).tolist())"`
  → `[17.87568464, 0.12431536] [7.3230495, 0.61069312] [[0.29476487, 0.95556982], [0.95556982, -0.29476487]]`.
  ferrolearn `PCA(n_components=2).fit(X)` matches all arrays incl. sign (live).
- **PCA constructor-ABI oracle (REQ-PCA-NCOMP-POSITIONAL,
  REQ-PCA-NCOMP-DEFAULT-NONE; R-CHAR-3):**
  `cd /tmp && python3 -c "import inspect, numpy as np; from sklearn.decomposition import PCA; print(PCA(2).n_components, inspect.signature(PCA.__init__).parameters['n_components'].default, PCA().fit(np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]])).components_.shape)"`
  → `2 None (2, 2)`. ferrolearn:
  `cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -c "import inspect; from ferrolearn import PCA; print(inspect.signature(PCA.__init__))"`
  → `(self, *, n_components=2)` and `PCA(2)` raises `TypeError`. A critic pins
  FAILING pytests; FAIL until the wrapper moves `n_components` before `*` and
  defaults to `None`.
- **PCA missing-param oracle (REQ-PCA-PARAMS):**
  `cd /tmp && python3 -c "import inspect; from sklearn.decomposition import PCA; ps=inspect.signature(PCA.__init__).parameters; print([p for p in ('copy','whiten','svd_solver','tol','iterated_power','n_oversamples','power_iteration_normalizer','random_state') if p in ps])"`
  → all 8; ferrolearn's signature has none. FAIL until the binding + wrapper add
  them (behavior owned by `ferrolearn-decomp` #1502/#1503/#1509).
- **PCA missing-attr oracle (REQ-PCA-ATTRS):**
  `cd /tmp && python3 -c "import numpy as np; from sklearn.decomposition import PCA; m=PCA(2).fit(np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.]])); print(m.n_components_, m.noise_variance_)"`
  → `2 0.0`. ferrolearn: `hasattr(fitted, 'n_components_')`/`'noise_variance_'` →
  `False` (live). FAIL until `ferrolearn-decomp` #1508/#1507 land.
- **Consumer check (REQ-CONSUMER):**
  `grep -n "_RsStandardScaler\|_RsPCA" /home/doll/ferrolearn/ferrolearn-python/python/ferrolearn/_transformers.py`
  shows `_transformers.py::StandardScaler`/`PCA` construct the `_Rs*` classes and
  drive fit/transform/inverse_transform + attribute reads; `ferrolearn/__init__.py:12`
  re-exports both; the 524-passing pytest exercises them.
- **Substrate check (REQ-SUBSTRATE):** `transformers.rs` head shows
  `use crate::conversions::*` + `use numpy::{PyArray1, PyArray2,
  PyReadonlyArray2}` — the wrong substrate per R-SUBSTRATE-1; ferray exposes no
  `numpy_interop` bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027.
