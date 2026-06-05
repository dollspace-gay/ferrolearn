# ferrolearn-python clusterers — sklearn.cluster.KMeans binding shim

<!--
tier: 3-component
status: draft
baseline-commit: c54770fe7
upstream-paths:
  - sklearn/cluster/_kmeans.py   # class KMeans(_BaseKMeans) — __init__/_parameter_constraints/fit/predict/transform/fit_predict/fit_transform/score
-->

## Summary

`ferrolearn-python/src/clusterers.rs` is the PyO3 marshalling shim binding
`ferrolearn_cluster::KMeans` (an unfitted estimator over `ferrolearn_cluster::FittedKMeans<f64>`)
to CPython as the `#[pyclass(name = "_RsKMeans")]` low-level class, and
`ferrolearn-python/python/ferrolearn/_clusterers.py` wraps it as
`ferrolearn.KMeans` — a `BaseEstimator`/`ClusterMixin`/`TransformerMixin`
subclass — so `import ferrolearn` mirrors `import sklearn.cluster.KMeans`. It
mirrors **`sklearn.cluster.KMeans`** (`sklearn/cluster/_kmeans.py:1196`).

The KMeans *correctness* (k-means++ seeding, Lloyd iteration, convergence,
`inertia_`) lives DOWN in `ferrolearn-cluster/src/kmeans.rs` and is audited by
that crate's routes (REQ status table in `kmeans.rs`'s `//!` doc; blockers
#1036/#1038/#1039/#1044). THIS unit is the **sklearn-API marshalling shim** only:
constructor parameter ABI, attribute exposure, method surface, and array
coercion across the Python↔Rust boundary. Semantic/numerical divergences are
fixed in `ferrolearn-cluster`; this doc references them and owns only the
binding-level surface (R-DEV-2 ABI, R-DEV-3 output contract, the
`ferray::numpy_interop` boundary covered by `conversions.md` #2027).

**Verification model: B (pytest vs sklearn 1.5.2).** Per goal.md §"The
verification model (B)", this unit is verified by
`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` comparing
`import ferrolearn` against the installed `import sklearn` 1.5.2 oracle, plus the
live-sklearn oracle for the constructor-ABI and attribute boundary. As of
baseline `c54770fe7` the gauntlet is GREEN: **518 passed**.
`ferrolearn.KMeans(n_clusters=3, random_state=42, n_init=2)` is exercised by
`tests/test_check_estimator.py` (`parametrize_with_checks`) and
`tests/test_cross_val_score.py`.

Divergence classes:
1. **api-conformance (SHIPPED core)** — `fit`/`predict`/`transform` +
   `fit_predict`/`fit_transform` (mixins) + `cluster_centers_`/`labels_`/
   `inertia_`/`n_iter_`/`n_features_in_` are all exposed with the right
   shapes/types; `check_estimator` + `cross_val_score` pass.
2. **n_clusters-keyword-only (NOT-STARTED, R-DEV-2, single-line fixable)** —
   `_clusterers.py::__init__(self, *, n_clusters=8, ...)` makes EVERY param
   keyword-only, so `ferrolearn.KMeans(2)` raises `TypeError` whereas
   `sklearn.cluster.KMeans(2)` works (`n_clusters` is positional-or-keyword).
3. **n_init-default (NOT-STARTED, R-DEV-2)** — ferrolearn default `n_init=10`;
   sklearn 1.5 default `n_init='auto'` (resolves to 1 for `init='k-means++'`).
4. **missing-params (NOT-STARTED, R-DEV-2)** — sklearn KMeans exposes
   `init`, `verbose`, `copy_x`, `algorithm`; the binding exposes none.
5. **missing-score (NOT-STARTED, R-DEV-3)** — sklearn `KMeans.score(X)` returns
   `-inertia`; `ferrolearn.KMeans` has no `score`.
6. **value-parity-RNG (NOT-STARTED, R-SUBSTRATE-5)** — exact
   `cluster_centers_`/`labels_`/`inertia_` parity requires matching numpy's RNG,
   which the `ferrolearn_cluster` `StdRng` substrate lacks. Deterministic
   STRUCTURE (k centers, valid labels, non-negative inertia) is checkable; the
   exact centroids are not.
7. **substrate (NOT-STARTED, R-SUBSTRATE-1)** — the binding round-trips numpy ↔
   `ndarray` via `crate::conversions::*` (rust-numpy), not
   `ferray::numpy_interop`/`ferray-core`; owned by `conversions.md` #2027.

## Upstream reference (sklearn 1.5.2, live oracle = installed sklearn 1.5.2)

`sklearn.cluster.KMeans` (`sklearn/cluster/_kmeans.py:1196`), lines stable at tag
1.5.2 / commit 156ef14:

- **`__init__`** (`_kmeans.py:1387-1411`):
  `KMeans(n_clusters=8, *, init="k-means++", n_init="auto", max_iter=300,
  tol=1e-4, verbose=0, random_state=None, copy_x=True, algorithm="lloyd")`.
  `n_clusters` is the ONLY positional-or-keyword param (it precedes the `*`);
  everything after `*` is keyword-only.
- **`_parameter_constraints`** (`_kmeans.py:1381-1385`): adds
  `"copy_x": ["boolean"]` and `"algorithm": [StrOptions({"lloyd", "elkan"})]`
  to `_BaseKMeans._parameter_constraints` (which carries `n_clusters`, `init`,
  `n_init`, `max_iter`, `tol`, `verbose`, `random_state`).
- **`fit`** (`_kmeans.py:1437`) — `_validate_data(X, accept_sparse="csr",
  dtype=[np.float64, np.float32], order="C", ...)` then sets
  `cluster_centers_`/`labels_`/`inertia_`/`n_iter_`/`n_features_in_`.
- **`predict`** (`_kmeans.py:1072`, `_BaseKMeans`) — labels of nearest center.
- **`transform`** (`_kmeans.py:1130`) — `euclidean_distances(X, cluster_centers_)`,
  shape `(n_samples, n_clusters)`.
- **`fit_predict`** (`_kmeans.py:1047`) / **`fit_transform`** (`_kmeans.py:1106`).
- **`score`** (`_kmeans.py:1156-1184`) — returns `-scores` (negative
  within-cluster sum of squares = `-inertia`).
- **attributes** (`_kmeans.py:1291-1311`): `cluster_centers_` `(n_clusters,
  n_features)`, `labels_` `(n_samples,)`, `inertia_` float, `n_iter_` int,
  `n_features_in_` int.

Live oracle (installed sklearn 1.5.2, run from `/tmp`; R-CHAR-3 — expected
values from sklearn, NEVER from ferrolearn):

```
sig (self, n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
KMeans(2).n_clusters                                  -> 2          # n_clusters is positional
n_init default                                        -> auto
init default                                          -> k-means++

X=[[0,0],[0.1,0],[5,5],[5.1,5]]; m=KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
m.cluster_centers_.shape -> (2, 2)   m.labels_ -> [1,1,0,0]   m.inertia_ -> 0.01
m.n_iter_ -> 2   m.n_features_in_ -> 2   m.score(X) -> -0.01
```

## Requirements

- REQ-API-CONFORM: `ferrolearn.KMeans` exposes the sklearn.cluster.KMeans method
  surface — `fit`/`predict`/`transform` (bound on `_RsKMeans` in
  `clusterers.rs`, wrapped in `_clusterers.py`) plus `fit_predict`/`fit_transform`
  (inherited from `ClusterMixin`/`TransformerMixin`) — and the fitted attributes
  `cluster_centers_` `(n_clusters, n_features)`, `labels_` `(n_samples,)`,
  `inertia_` (float), `n_iter_` (int), and `n_features_in_` (int, set by
  `_validate_data`), mirroring `_kmeans.py:1291-1311`. The binding marshals the
  estimator API end-to-end; `check_estimator` + `cross_val_score` pass.
- REQ-NCLUSTERS-POSITIONAL: `ferrolearn.KMeans` accepts `n_clusters`
  positionally — `KMeans(2)` constructs an estimator with `n_clusters == 2`,
  matching sklearn `__init__(self, n_clusters=8, *, ...)` (`_kmeans.py:1387-1390`,
  `n_clusters` before the `*`).
- REQ-NINIT-DEFAULT: `ferrolearn.KMeans`'s `n_init` default matches sklearn's
  `n_init='auto'` semantics (`_kmeans.py:1392`, `:1240-1243`): `'auto'` resolves
  to 1 for `init='k-means++'`, 10 for `init='random'`/callable.
- REQ-INIT-PARAM: `ferrolearn.KMeans` exposes the `init` constructor param
  (`{'k-means++','random'}|callable|array`, default `'k-means++'`,
  `_kmeans.py:1391`/`:1211-1232`). [Surface owned here; underlying behavior +
  exact k-means++ in `ferrolearn-cluster` REQ-7, blocker #1038.]
- REQ-ALGORITHM-PARAM: `ferrolearn.KMeans` exposes the `algorithm` param
  (`{'lloyd','elkan'}`, default `'lloyd'`, `_kmeans.py:1277-1289`/`:1384`).
- REQ-COPYX-PARAM: `ferrolearn.KMeans` exposes the `copy_x` param (bool,
  default `True`, `_kmeans.py:1267-1275`/`:1383`).
- REQ-VERBOSE-PARAM: `ferrolearn.KMeans` exposes the `verbose` param (int,
  default `0`, `_kmeans.py:1259-1260`).
- REQ-SCORE: `ferrolearn.KMeans.score(X)` returns `-inertia` (negative
  within-cluster sum of squares), mirroring `_kmeans.py:1156-1184` (returns
  `-scores`), so `KMeans` is usable with `scoring=None`/default-scorer paths.
- REQ-VALUE-PARITY: exact `cluster_centers_`/`labels_`/`inertia_`/`n_iter_`
  match sklearn array-by-array on a fixed `random_state` and dataset (R-DEV-1),
  the way users compare ferrolearn outputs to sklearn. [The deterministic
  STRUCTURE — k centers, valid labels in `0..n_clusters`, non-negative
  monotone-decreasing inertia, `predict == transform.argmin(1)` — is checkable;
  exact centroids require numpy-RNG init parity, owned by `ferrolearn-cluster`
  REQ-8/REQ-5, blockers #1039/#1036.]
- REQ-SUBSTRATE: the binding's array marshalling is on `ferray::numpy_interop`
  producing `ferray-core` arrays, not rust-numpy + `ndarray` (R-SUBSTRATE-1).
  [Owned by `conversions.md` REQ-FERRAY #2027.]
- REQ-CONSUMER: the binding IS the public API (R-DEFER-1/S5: boundary estimator
  types ARE the public surface); its non-test production consumer is the Python
  wrapper `_clusterers.py` + external users, exercised by the pytest gauntlet
  (`test_check_estimator.py`, `test_cross_val_score.py`).

## Acceptance criteria

All expected values come from the live sklearn 1.5.2 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn. The pytest gauntlet
(`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`) is the
end-to-end check (verification model B); rebuild first if the Rust side changed
(`cd ferrolearn-python && maturin develop`).

- AC-API-CONFORM (REQ-API-CONFORM): `test_check_estimator.py`
  (`parametrize_with_checks([..., KMeans(n_clusters=3, random_state=42,
  n_init=2)])`) + `test_cross_val_score.py` pass — the binding marshals
  fit/predict/transform/attrs. Spot oracle (shapes/types on a fixed seed):
  `cd /tmp && python3 -c "import numpy as np; from sklearn.cluster import KMeans; X=np.array([[0.,0.],[0.1,0.],[5.,5.],[5.1,5.]]); m=KMeans(n_clusters=2,n_init=10,random_state=0).fit(X); print(m.cluster_centers_.shape, len(m.labels_), type(m.inertia_).__name__, type(m.n_iter_).__name__, m.n_features_in_)"`
  → `(2, 2) 4 float ... 2`. `ferrolearn.KMeans(n_clusters=2, n_init=10,
  random_state=0).fit(X)` exposes `cluster_centers_.shape == (2,2)`, `labels_`
  of length 4, float `inertia_`, int `n_iter_`, `n_features_in_ == 2`,
  `fit_predict`/`fit_transform` present (confirmed live).
- AC-NCLUSTERS-POSITIONAL (REQ-NCLUSTERS-POSITIONAL): sklearn oracle
  `cd /tmp && python3 -c "from sklearn.cluster import KMeans; print(KMeans(2).n_clusters)"`
  → `2`. ferrolearn FAILS:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "from ferrolearn import KMeans; KMeans(2)"`
  → `TypeError: KMeans.__init__() takes 1 positional argument but 2 were given`
  (live-confirmed). A critic pins a FAILING pytest asserting `KMeans(2)`
  constructs `n_clusters == 2`. FAILS until `_clusterers.py::__init__` moves
  `n_clusters` before the `*`.
- AC-NINIT-DEFAULT (REQ-NINIT-DEFAULT): sklearn oracle
  `cd /tmp && python3 -c "import inspect; from sklearn.cluster import KMeans; print(inspect.signature(KMeans.__init__).parameters['n_init'].default)"`
  → `auto`. ferrolearn signature default is `n_init=10` (live:
  `inspect.signature(ferrolearn.KMeans.__init__)` →
  `(self, *, n_clusters=8, max_iter=300, tol=0.0001, n_init=10, random_state=None)`).
  A critic pins a FAILING pytest asserting `KMeans().n_init == 'auto'`. FAILS
  until the wrapper adopts `n_init='auto'` resolution.
- AC-INIT-PARAM / AC-ALGORITHM-PARAM / AC-COPYX-PARAM / AC-VERBOSE-PARAM
  (REQ-INIT-PARAM, REQ-ALGORITHM-PARAM, REQ-COPYX-PARAM, REQ-VERBOSE-PARAM):
  sklearn exposes them
  (`cd /tmp && python3 -c "import inspect; from sklearn.cluster import KMeans; ps=inspect.signature(KMeans.__init__).parameters; print([p for p in ('init','algorithm','copy_x','verbose') if p in ps])"`
  → `['init', 'algorithm', 'copy_x', 'verbose']`). ferrolearn's signature has
  NONE (live signature above). A critic pins FAILING pytests asserting
  `'init'`/`'algorithm'`/`'copy_x'`/`'verbose'` ∈
  `inspect.signature(ferrolearn.KMeans.__init__).parameters`. FAIL until the
  binding + wrapper add them (underlying behavior in `ferrolearn-cluster`).
- AC-SCORE (REQ-SCORE): sklearn oracle
  `cd /tmp && python3 -c "import numpy as np; from sklearn.cluster import KMeans; X=np.array([[0.,0.],[0.1,0.],[5.,5.],[5.1,5.]]); m=KMeans(n_clusters=2,n_init=10,random_state=0).fit(X); print(round(m.score(X),6), round(-m.inertia_,6))"`
  → `-0.01 -0.01` (`score == -inertia`). ferrolearn FAILS:
  `hasattr(ferrolearn.KMeans().fit(X)-fitted, 'score')` is `False` (live).
  A critic pins a FAILING pytest asserting `KMeans().fit(X).score(X) ==
  -inertia_`. FAILS until `score` is added (mixins don't supply it — sklearn's
  is defined on `_BaseKMeans`).
- AC-VALUE-PARITY (REQ-VALUE-PARITY): structure is checkable, exact values are
  not. Live-confirmed STRUCTURE on `X=[[0,0],[0.1,0],[5,5],[5.1,5]]`,
  `KMeans(n_clusters=2, n_init=10, random_state=0)`: ferrolearn
  `cluster_centers_.shape == (2,2)`, `set(labels_) == {0,1}`, `inertia_ >= 0`,
  `predict(X) == transform(X).argmin(1)`. Exact-value AC (sklearn
  `cluster_centers_ -> [[5.05,5.],[0.05,0.]]`, `inertia_ -> 0.01`): a critic
  pins a FAILING pytest comparing centroids/inertia array-by-array; it FAILS on
  RNG-dependent init until numpy-RNG parity lands in `ferrolearn-cluster`
  (blockers #1039 init RNG, #1036 convergence/tol). Owned downstream; this REQ
  is NOT-STARTED at the binding because the values it marshals are not yet
  sklearn-exact.
- AC-SUBSTRATE (REQ-SUBSTRATE): `clusterers.rs` head shows `use
  crate::conversions::*` + `use numpy::{PyArray1, PyArray2, PyReadonlyArray2}`
  and `use ndarray::...` (via conversions) — the wrong substrate per
  R-SUBSTRATE-1 (destination `ferray::numpy_interop`/`ferray-core`). ferray
  exposes no `numpy_interop` bridge consumable here (R-SUBSTRATE-5). Owned by
  `conversions.md` REQ-FERRAY #2027.
- AC-CONSUMER (REQ-CONSUMER):
  `grep -rn "_RsKMeans\|KMeans" /home/doll/ferrolearn/ferrolearn-python/python/ferrolearn/_clusterers.py`
  shows `_clusterers.py::KMeans` constructs `_RsKMeans` and calls
  `fit`/`predict`/`transform` + reads `cluster_centers_`/`labels_`/`inertia_`/
  `n_iter_`; `ferrolearn/__init__.py` re-exports `KMeans`; `test_check_estimator.py`
  and `test_cross_val_score.py` consume it. The 518-passing pytest exercises the
  consumer surface.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-API-CONFORM (fit/predict/transform + mixins + attrs) | SHIPPED | impl `RsKMeans::fit`/`predict`/`transform` + getters `cluster_centers_`/`labels_`/`inertia_`/`n_iter_` in `clusterers.rs` (over `ferrolearn_cluster::FittedKMeans<f64>`), wrapped by `KMeans` in `_clusterers.py` which sets `cluster_centers_`/`labels_`/`inertia_`/`n_iter_` and `n_features_in_` (via `self._validate_data`) and inherits `fit_predict`/`fit_transform` from `ClusterMixin`/`TransformerMixin` — mirroring `sklearn/cluster/_kmeans.py:1291-1311` (attrs) + `:1047`/`:1072`/`:1106`/`:1130` (methods). Non-test consumer: `_clusterers.py::KMeans` + `ferrolearn/__init__.py` re-export; external users. Verification (model B): `cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` → 518 passed, 0 failed (`test_check_estimator.py` runs `KMeans(n_clusters=3, random_state=42, n_init=2)` through `parametrize_with_checks`; `test_cross_val_score.py` runs it through `cross_val_score`). Live shapes/types match the sklearn oracle (`cluster_centers_.shape (2,2)`, len-4 `labels_`, float `inertia_`, int `n_iter_`, `n_features_in_ == 2`). |
| REQ-NCLUSTERS-POSITIONAL (n_clusters positional ABI) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 constructor ABI). sklearn `__init__(self, n_clusters=8, *, ...)` (`_kmeans.py:1387-1390`) makes `n_clusters` positional-or-keyword — `KMeans(2).n_clusters` → `2`. ferrolearn `_clusterers.py::__init__(self, *, n_clusters=8, ...)` makes EVERY param keyword-only — live: `ferrolearn.KMeans(2)` → `TypeError: __init__() takes 1 positional argument but 2 were given`. Single-line Python-wrapper fix: move `n_clusters` before the `*` in `_clusterers.py::__init__`. |
| REQ-NINIT-DEFAULT (n_init='auto' default) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 default value). sklearn default `n_init='auto'` (`_kmeans.py:1392`; oracle: `signature(...).parameters['n_init'].default` → `auto`), resolving to 1 for `init='k-means++'`, 10 for `init='random'`/callable (`_kmeans.py:1240-1243`, `_check_params_vs_input(..., default_n_init=10)` `:1414`). ferrolearn default `n_init=10` (`_clusterers.py::__init__` + `RsKMeans::new` signature `n_init=10`) — a different default. Wrapper-level fix (adopt `'auto'` + resolve to 1 for the default k-means++ init). |
| REQ-INIT-PARAM (init constructor param) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2). sklearn `init ∈ {'k-means++','random'}|callable|array`, default `'k-means++'` (`_kmeans.py:1391`/`:1211-1232`). The binding exposes NO `init` param (`RsKMeans::new` signature is `n_clusters,max_iter,tol,n_init,random_state`; `_clusterers.py` likewise). Underlying `ferrolearn_cluster::KMeans` always does greedy k-means++ (`fn kmeans_plus_plus in kmeans.rs`), so the DEFAULT matches but non-default inits and the param surface are missing — owned downstream by `ferrolearn-cluster` REQ-7 (blocker #1038); the binding cannot expose what the library lacks. |
| REQ-ALGORITHM-PARAM (algorithm lloyd/elkan) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2). sklearn `algorithm ∈ {'lloyd','elkan'}`, default `'lloyd'` (`_kmeans.py:1277-1289`/`:1384` `_parameter_constraints`). The binding exposes no `algorithm` param. ferrolearn implements Lloyd only (no Elkan in `kmeans.rs`); the default matches but the param + Elkan path are missing — owned downstream by `ferrolearn-cluster`. |
| REQ-COPYX-PARAM (copy_x) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2). sklearn `copy_x` bool, default `True` (`_kmeans.py:1267-1275`/`:1383`). The binding exposes no `copy_x` param. (The wrappers already copy via `_validate_data`/`_ensure_f64`, so the OBSERVABLE non-mutation holds, but the constructor-ABI param is absent.) |
| REQ-VERBOSE-PARAM (verbose) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2). sklearn `verbose` int, default `0` (`_kmeans.py:1259-1260`; in `_BaseKMeans._parameter_constraints`). The binding exposes no `verbose` param. |
| REQ-SCORE (score = -inertia) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-3 output contract). sklearn `KMeans.score(X)` returns `-scores` = `-inertia` (`_kmeans.py:1156-1184`); oracle: `score(X)` → `-0.01` == `-inertia_`. ferrolearn `KMeans` has no `score` (live: `hasattr(fitted, 'score')` → `False`; `ClusterMixin`/`TransformerMixin`/`BaseEstimator` supply none — sklearn's lives on `_BaseKMeans`). Needs a `score` method on the wrapper (or a bound `_RsKMeans` method) returning `-inertia_`. |
| REQ-VALUE-PARITY (exact cluster_centers_/labels_/inertia_/n_iter_) | NOT-STARTED | open prereq blockers #1039 (numpy-RNG init parity) + #1036 (convergence criterion + relative tol), in `ferrolearn-cluster`. The binding faithfully marshals whatever `ferrolearn_cluster::FittedKMeans` produces, but KMeans init is RNG-dependent and `ferrolearn_cluster` uses `StdRng` (`StdRng::seed_from_u64 in kmeans.rs::fit`), not numpy's RNG, so exact centroids/labels diverge from sklearn on the same `random_state` (R-DEV-1 array-by-array parity). Deterministic STRUCTURE is SHIPPED-quality (live: `cluster_centers_.shape == (2,2)`, `set(labels_) == {0,1}`, `inertia_ >= 0`, `predict == transform.argmin(1)`) but exact VALUES are not — so the REQ is NOT-STARTED until the downstream RNG/convergence blockers land. |
| REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | open prereq blocker = `conversions.md` REQ-FERRAY #2027. `clusterers.rs` marshals via `use crate::conversions::*` + `use numpy::{PyArray1, PyArray2, PyReadonlyArray2}` (rust-numpy) and the conversions produce `ndarray::Array{1,2}` — the WRONG substrate per R-SUBSTRATE-1 (destination `ferray::numpy_interop` + `ferray-core`). ferray exposes no PyO3 numpy-interop bridge yet (R-SUBSTRATE-5). Owned by the conversions unit, surfaced here. |
| REQ-CONSUMER (binding IS the public API) | SHIPPED | the binding boundary type IS the public API (R-DEFER-1/S5: boundary estimator types ARE the public surface; grandfathered existing pub API). Non-test production consumers: `_clusterers.py::KMeans` constructs `_RsKMeans(...)` and calls `fit`/`predict`/`transform` + reads `cluster_centers_`/`labels_`/`inertia_`/`n_iter_` (`grep -n "_RsKMeans" python/ferrolearn/_clusterers.py`); `ferrolearn/__init__.py` re-exports `KMeans`; `test_check_estimator.py` + `test_cross_val_score.py` are the verification consumers + external users. Verification (model B): pytest → 518 passed (KMeans exercised in both consumer-suite files). |

## Architecture

`clusterers.rs` is a single `#[pyclass(name = "_RsKMeans")]` struct holding the
five constructor params (`n_clusters`, `max_iter`, `tol`, `n_init`,
`random_state`) plus an `Option<ferrolearn_cluster::FittedKMeans<f64>>`. It is a
THIN shim: `new` (`#[pyo3(signature = (n_clusters=8, max_iter=300, tol=1e-4,
n_init=10, random_state=None))]`) stores params; `fit` builds a
`ferrolearn_cluster::KMeans::<f64>::new(n_clusters).with_max_iter(..).with_tol(..)
.with_n_init(..)[.with_random_state(..)]`, runs it on the `numpy2_to_ndarray`-
coerced X, and stores the fitted model (mapping `FerroError` → `PyValueError`);
`predict`/`transform` and the getters delegate to the stored
`FittedKMeans`, returning `PyRuntimeError("not fitted")` before fit. All
estimator MATH lives in `ferrolearn-cluster/src/kmeans.rs`; this file owns ZERO
algorithm logic — only ABI, attribute exposure, and array coercion (R-DEV-2/-3 +
the `conversions.rs` boundary).

`_clusterers.py::KMeans(TransformerMixin, ClusterMixin, BaseEstimator)` is the
sklearn-facing wrapper: `__init__(self, *, n_clusters=8, max_iter=300, tol=1e-4,
n_init=10, random_state=None)` (the keyword-only `*` is REQ-NCLUSTERS-POSITIONAL's
divergence); `fit` calls `self._validate_data(X, dtype="float64")` (setting
`n_features_in_`), constructs `_RsKMeans`, fits, and copies out
`cluster_centers_`/`labels_`/`inertia_`/`n_iter_`; `predict`/`transform`
`check_is_fitted` + re-validate + (lazily rebuild `_rs` via `_rebuild_rs` after
unpickling); `fit_predict`/`fit_transform` come from the mixins; there is NO
`score` (REQ-SCORE divergence). `__getstate__`/`__setstate__` drop and rebuild
`_rs` for pickling.

The honest call (R-HONEST-3): the API-CONFORMANCE and the CONSUMER ship on impl +
the 518-passing pytest (`check_estimator` + `cross_val_score`). Everything else
is NOT-STARTED (binary, R-DEFER-2) and splits into:

- **Single-wrapper-fixable ABI divergences (the headline)**:
  REQ-NCLUSTERS-POSITIONAL (move `n_clusters` before `*` — one line) and
  REQ-NINIT-DEFAULT (`n_init=10` → `'auto'`). These are pure Python-wrapper
  constructor-ABI fixes that need no library change.
- **Missing-surface divergences**: REQ-INIT-PARAM / REQ-ALGORITHM-PARAM /
  REQ-COPYX-PARAM / REQ-VERBOSE-PARAM (constructor params the binding doesn't
  expose) and REQ-SCORE (a method the wrapper lacks). The default `init`/
  `algorithm` MATCH (k-means++ / Lloyd), so only the param surface + non-default
  paths are missing; `init`/`algorithm`/Elkan behavior is owned downstream by
  `ferrolearn-cluster` (blocker #1038).
- **RNG-blocked value parity**: REQ-VALUE-PARITY — the binding marshals faithfully
  but the underlying `StdRng`-seeded k-means++ does not reproduce numpy's RNG, so
  exact centroids/labels diverge (owned downstream, blockers #1039/#1036). The
  deterministic STRUCTURE is sound; only the exact VALUES are not.
- **Substrate**: REQ-SUBSTRATE — rust-numpy + `ndarray`, not
  `ferray::numpy_interop` (owned by `conversions.md` #2027).

The least-confident SHIPPED claim is REQ-API-CONFORM: it is SHIPPED on the
pytest gauntlet (`check_estimator` passes for `KMeans(n_clusters=3,
random_state=42, n_init=2)`), but `check_estimator` does not assert exact-value
sklearn parity — that's REQ-VALUE-PARITY, honestly NOT-STARTED. So API-CONFORM
ships the *shape/type/method* contract, not the *numerical* one; the two are kept
distinct deliberately (R-HONEST-3 underclaim).

## Verification

Commands establishing the SHIPPED claims (run at baseline `c54770fe7`,
verification model B; rebuild first if the Rust side changed:
`cd /home/doll/ferrolearn/ferrolearn-python && maturin develop`):

- **Full pytest gauntlet (REQ-API-CONFORM, REQ-CONSUMER):**
  `cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`
  → `518 passed`. `test_check_estimator.py` runs `KMeans(n_clusters=3,
  random_state=42, n_init=2)` through sklearn's `parametrize_with_checks`;
  `test_cross_val_score.py` lists `KMeans` in its consumer suites.
- **Constructor-ABI oracle (REQ-NCLUSTERS-POSITIONAL, REQ-NINIT-DEFAULT;
  R-CHAR-3):**
  `cd /tmp && python3 -c "import inspect; from sklearn.cluster import KMeans; print(KMeans(2).n_clusters, inspect.signature(KMeans.__init__).parameters['n_init'].default)"`
  → `2 auto`. ferrolearn:
  `cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -c "import inspect; from ferrolearn import KMeans; print(inspect.signature(KMeans.__init__))"`
  → `(self, *, n_clusters=8, max_iter=300, tol=0.0001, n_init=10, random_state=None)`
  and `KMeans(2)` raises `TypeError`. A critic pins FAILING pytests; FAIL until
  the wrapper moves `n_clusters` before `*` and adopts `n_init='auto'`.
- **Missing-param oracle (REQ-INIT/ALGORITHM/COPYX/VERBOSE-PARAM):**
  `cd /tmp && python3 -c "import inspect; from sklearn.cluster import KMeans; ps=inspect.signature(KMeans.__init__).parameters; print([p for p in ('init','algorithm','copy_x','verbose') if p in ps])"`
  → `['init', 'algorithm', 'copy_x', 'verbose']`; ferrolearn's signature has
  none. FAIL until the binding + wrapper add them (behavior owned by
  `ferrolearn-cluster`, blocker #1038 for `init`).
- **Score oracle (REQ-SCORE):**
  `cd /tmp && python3 -c "import numpy as np; from sklearn.cluster import KMeans; X=np.array([[0.,0.],[0.1,0.],[5.,5.],[5.1,5.]]); m=KMeans(n_clusters=2,n_init=10,random_state=0).fit(X); print(round(m.score(X),6), round(-m.inertia_,6))"`
  → `-0.01 -0.01` (`score == -inertia`). ferrolearn:
  `hasattr(ferrolearn.KMeans(...).fit(X), 'score')` → `False`. FAIL until `score`
  is added.
- **Value-parity oracle (REQ-VALUE-PARITY):**
  `cd /tmp && python3 -c "import numpy as np; from sklearn.cluster import KMeans; X=np.array([[0.,0.],[0.1,0.],[5.,5.],[5.1,5.]]); m=KMeans(n_clusters=2,n_init=10,random_state=0).fit(X); print(m.inertia_, sorted(m.cluster_centers_.tolist()))"`
  → `0.01 [[0.05, 0.0], [5.05, 5.0]]`. ferrolearn structure matches (k=2 centers,
  labels `{0,1}`, `inertia_ >= 0`, `predict == transform.argmin(1)` — live), but
  exact centroids/inertia on the same `random_state` differ (StdRng ≠ numpy RNG).
  A critic pins a FAILING array-by-array pytest; FAIL until #1039/#1036 land in
  `ferrolearn-cluster`.
- **Substrate check (REQ-SUBSTRATE):** `clusterers.rs` head shows
  `use crate::conversions::*` + `use numpy::{PyArray1, PyArray2,
  PyReadonlyArray2}` — the wrong substrate per R-SUBSTRATE-1; ferray exposes no
  `numpy_interop` bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027.
- **Consumer check (REQ-CONSUMER):**
  `grep -n "_RsKMeans" /home/doll/ferrolearn/ferrolearn-python/python/ferrolearn/_clusterers.py`
  shows `_clusterers.py::KMeans` constructs `_RsKMeans` and drives
  fit/predict/transform + attribute reads; `ferrolearn/__init__.py` re-exports
  `KMeans`; the 518-passing pytest exercises it.
