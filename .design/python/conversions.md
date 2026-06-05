# ferrolearn-python conversions — numpy ↔ ndarray PyO3 marshalling bridge

<!--
tier: 3-component
status: draft
baseline-commit: b64f72e5e
upstream-paths:
  - sklearn/utils/validation.py   # check_array / check_X_y / column_or_1d / _check_y — the dtype/label boundary contract
-->

## Summary

`ferrolearn-python/src/conversions.rs` is the numpy ↔ `ndarray` marshalling
bridge used by every estimator binding (`regressors.rs`, `classifiers.rs`,
`transformers.rs`, `clusterers.rs`, `extras.rs`). It is the
**`ferray::numpy_interop` analog** (the PyO3 array bridge on the ferray
substrate, per goal.md R-SUBSTRATE-1): six free functions that copy a read-only
numpy array into an owned `ndarray::Array{1,2}` on the way in
(`numpy2_to_ndarray`, `numpy1_to_ndarray`, `numpy1_to_ndarray_usize`) and
allocate a fresh numpy array from an `ndarray` on the way out
(`ndarray1_to_numpy`, `ndarray2_to_numpy`, `ndarray1_usize_to_numpy`). It owns
no estimator math — it is the data-plane boundary between CPython/numpy and the
Rust library crates, the spot where the numpy dtype/shape contract sklearn
enforces in `sklearn/utils/validation.py` must be preserved across the FFI
(R-CODE-4 / R-CODE-5).

This module has no scikit-learn *function* counterpart — numpy↔C marshalling is
implicit in CPython. Its contract is therefore the **boundary discipline** the
sklearn validation layer (`check_array`/`check_X_y`/`column_or_1d`/`_check_y`)
imposes on data crossing into an estimator: float arrays stay float and keep
their shape; label arrays keep their semantics. The conversions in this file are
the Rust half of that boundary; the Python half lives in the wrappers
(`python/ferrolearn/_*.py`), which call `self._validate_data(..., dtype="float64")`
and (for classifiers) a `LabelEncoder`-equivalent BEFORE the Rust side ever runs.

**Verification model: B (pytest vs sklearn 1.5.2).** Per goal.md §"The
verification model (B)", this unit is verified by
`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` comparing
`import ferrolearn` against the installed `import sklearn` 1.5.2 oracle, plus the
live-sklearn oracle for the label-handling boundary. As of baseline `b64f72e5e`
the gauntlet is GREEN: **513 passed** (after the `validate_data` fix, #2023). The
f64 round-trip and the consumer surface are exercised by every estimator
fit/predict round-trip in that suite.

Divergence classes:
1. **f64 round-trip parity (SHIPPED core)** — `numpy2_to_ndarray`/
   `numpy1_to_ndarray` + `ndarray2_to_numpy`/`ndarray1_to_numpy` preserve shape
   and f64 values exactly (rust-numpy `as_array().to_owned()` copy in,
   `from_array` copy out). The f64 dtype is GUARANTEED upstream in Python
   (`_validate_data(dtype="float64")` + `_ensure_f64`), so the Rust side assumes
   f64 by contract — the coercion is explicit in Python, not a silent Rust cast
   (R-CODE-4).
2. **i64 ↔ usize label marshalling + WHERE encoding happens (the headline)** —
   `numpy1_to_ndarray_usize` does `v as usize` on an i64 array;
   `ndarray1_usize_to_numpy` does `v as i64`. `(-1_i64) as usize` wraps to
   `18446744073709551615` — silent corruption for negative/arbitrary labels IF
   the raw binding were called directly. It is NOT, because the Python
   classifier wrappers (`_classifiers.py::_encode_labels`/`_decode_labels`)
   perform the `LabelEncoder` step (np.unique → contiguous `0..n_classes-1`
   int64; decode via `classes_[y_encoded]`) BEFORE/AFTER the Rust call. So the
   cast is safe under the wrapper-guarded contract (non-negative contiguous
   labels), and the REQ is SHIPPED for that contract — with the raw-binding gap
   (a `_RsLogisticRegression().fit(X, [-1,1])` direct call would wrap) documented
   as the boundary the wrapper must hold.
3. **dtype coverage (NOT-STARTED)** — conversions handles only f64 arrays + i64
   labels. sklearn's `check_array` PRESERVES float32 when `dtype=None`/`"numeric"`
   (oracle confirms); the ferrolearn wrappers force float64 via
   `_validate_data(dtype="float64")` + `np.ascontiguousarray(_, dtype=float64)`,
   so a float32 input is up-cast to float64 — a dtype change vs sklearn
   preserving float32 (R-CODE-5). int32/int64 X likewise become float64.
4. **sparse input (NOT-STARTED)** — sklearn estimators accept `scipy.sparse` X;
   `conversions.rs` only handles dense `PyReadonlyArray{1,2}`. No sparse marshal.
5. **ferray::numpy_interop substrate (NOT-STARTED, R-SUBSTRATE-1)** —
   `conversions.rs` uses `numpy` (rust-numpy) + `ndarray`; the destination is
   `ferray::numpy_interop` (the numpy bridge on the ferray substrate) producing
   `ferray-core` arrays, not `ndarray`.
6. **consumer (SHIPPED)** — every estimator binding imports
   `crate::conversions::*` and calls these on every fit/predict/transform; the
   513-passing pytest exercises the whole consumer surface.

## Upstream reference (sklearn 1.5.2, live oracle = installed sklearn 1.5.2)

The boundary contract conversions must preserve is the dtype/label behavior of
the validation layer (`sklearn/utils/validation.py`, lines stable at tag
1.5.2 / commit 156ef14):

- **`check_array`** (`validation.py:718`). With `dtype="numeric"` and a numeric
  input, the original floating dtype is PRESERVED — float32 stays float32
  (`dtype_orig` logic, `validation.py:854-913`: when the resolved `dtype is
  None` the array keeps its dtype). With an explicit `dtype=np.float64`, the
  array is coerced to float64 (`validation.py:915-942` + the later `astype`).
  This is exactly the lever the ferrolearn wrappers pull: they pass
  `dtype="float64"`, forcing float64 and erasing float32 (divergence class 3).
- **`check_X_y`** (`validation.py:1154`) validates X with `check_array` and y
  with `_check_y`/`column_or_1d`, then `check_consistent_length`
  (`validation.py:436`).
- **`column_or_1d`** (`validation.py:1348`) ravels y to 1-D and, by default
  (`dtype=None`), PRESERVES y's dtype — sklearn does NOT cast labels to a fixed
  integer width. Arbitrary integer, negative, and string labels survive into the
  estimator, which maps them through `LabelEncoder`/`np.unique` to
  `0..n_classes-1` internally and exposes the originals as `classes_`; `predict`
  returns ORIGINAL labels (`_check_y`, `validation.py:1325`).
- **`_check_y`** (`validation.py:1325`) is the y-side of `check_X_y`.

Live oracle (installed sklearn 1.5.2, run from `/tmp`), establishing the label
and dtype boundary conversions must preserve:

```
# negative integer labels — sklearn keeps them; (-1) as usize would wrap
LogisticRegression().fit(X, [-1, 1, -1, 1]).classes_  -> [-1, 1]
LogisticRegression().fit(X, [-1, 1, -1, 1]).predict(X) -> [-1, -1, 1, 1]
np.uint64(np.int64(-1))                                -> 18446744073709551615   # the wrap

# string labels — sklearn keeps them; an i64 marshaller cannot carry them
LogisticRegression().fit(X, ['cat','dog','cat','dog']).classes_   -> ['cat', 'dog']
LogisticRegression().fit(X, ['cat','dog','cat','dog']).predict(X) -> ['cat','cat','dog','dog']

# dtype preservation vs forced float64
check_array(np.array([[1.,2.]], dtype=np.float32), dtype='numeric').dtype -> float32   # preserved
check_array(np.array([[1.,2.]], dtype=np.float32), dtype=np.float64).dtype -> float64   # coerced
```

## Where label encoding happens (the precise finding)

`_classifiers.py` resolves the negative/string/arbitrary-label case in PYTHON,
above the Rust boundary, so `numpy1_to_ndarray_usize`'s `v as usize` only ever
sees non-negative contiguous integers:

- `_encode_labels(y)` (`_classifiers.py`): `classes = np.unique(y)`,
  `label_map = {c: i for i, c in enumerate(classes)}`,
  `y_encoded = np.array([label_map[v] for v in y], dtype=np.int64)` — i.e. a
  `LabelEncoder`-equivalent producing labels in `0..n_classes-1`, int64.
- Every classifier `fit` (`LogisticRegression`, `DecisionTreeClassifier`,
  `RandomForestClassifier`, `KNeighborsClassifier`, `GaussianNB`) calls
  `y_encoded, self.classes_ = _encode_labels(y)` and passes `y_encoded` (not raw
  `y`) into the Rust `fit`.
- Every classifier `predict` calls
  `_decode_labels(np.asarray(self._rs.predict(X)), self.classes_)` —
  `classes_[y_encoded]` — so the ORIGINAL labels come back out, matching sklearn.

Therefore the i64↔usize cast in `conversions.rs` is SAFE **only because the
Python wrapper guarantees non-negative contiguous int64 labels**. The cast is a
real latent hazard at the raw `_Rs*` binding (a direct
`_RsLogisticRegression().fit(X, np.array([-1,1,-1,1]))` would wrap `-1` to
`u64::MAX`), but no production path reaches it un-encoded. This is documented as
the wrapper-guarded contract (REQ-LABEL-MARSHAL SHIPPED) with the boundary the
wrapper must hold; the raw-binding negative/string case is the documented gap,
not a separate NOT-STARTED REQ, because the production consumer (the wrapper)
never violates the contract.

## Requirements

- REQ-F64-ROUNDTRIP: `numpy2_to_ndarray`/`numpy1_to_ndarray` copy a read-only
  numpy f64 array into an owned `Array2<f64>`/`Array1<f64>` preserving shape and
  values exactly; `ndarray2_to_numpy`/`ndarray1_to_numpy` copy an
  `Array2<f64>`/`Array1<f64>` back into a fresh numpy f64 array preserving shape
  and values exactly. The f64 dtype contract is enforced upstream in Python
  (`_validate_data(dtype="float64")` + `_ensure_f64`), so the Rust side assumes
  f64 by contract (R-CODE-4 — explicit Python coercion, no silent Rust cast).
- REQ-LABEL-MARSHAL: `numpy1_to_ndarray_usize` (i64→usize, `v as usize`) and
  `ndarray1_usize_to_numpy` (usize→i64, `v as i64`) marshal class labels across
  the boundary, SAFE under the wrapper-guarded contract that labels are
  non-negative contiguous int64 in `0..n_classes-1`. The `LabelEncoder`-
  equivalent that establishes that contract lives in
  `_classifiers.py::_encode_labels`/`_decode_labels` (np.unique encode /
  `classes_[·]` decode), mirroring sklearn's internal label encoding
  (`_check_y`/`column_or_1d`, `validation.py:1325`/`:1348`). The raw-binding
  negative/string-label case is a documented latent hazard (`(-1) as usize` →
  `u64::MAX`) that no production path reaches un-encoded.
- REQ-DTYPE-COVERAGE: conversions preserve the input numpy dtype where sklearn
  does. sklearn `check_array(dtype='numeric')` PRESERVES float32
  (`validation.py:854-913`); ferrolearn handles only f64 and the wrappers force
  float64 (`dtype="float64"` + `np.ascontiguousarray(_, dtype=float64)`), so
  float32/int input is up-cast to float64 — a dtype change vs sklearn
  (R-CODE-5).
- REQ-SPARSE-INPUT: conversions accept `scipy.sparse` X where sklearn estimators
  do. `conversions.rs` handles only dense `PyReadonlyArray{1,2}`; no sparse
  marshalling exists.
- REQ-FERRAY: conversions are implemented on `ferray::numpy_interop` producing
  `ferray-core` arrays, not rust-numpy + `ndarray` (R-SUBSTRATE-1).
- REQ-CONSUMER: a non-test production consumer uses these conversions on the live
  translation surface — every estimator binding
  (`regressors.rs`/`classifiers.rs`/`transformers.rs`/`clusterers.rs`/
  `extras.rs`) imports `crate::conversions::*` and calls them on every
  fit/predict/transform.

## Acceptance criteria

All expected values come from the live sklearn 1.5.2 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn. The pytest gauntlet
(`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`) is the
end-to-end check (verification model B).

- AC-F64-ROUNDTRIP (REQ-F64-ROUNDTRIP): every estimator fit/predict/transform
  round-trips an f64 array through `numpy*_to_ndarray` → library crate →
  `ndarray*_to_numpy` and the output matches sklearn within tolerance; the
  whole 513-test suite exercises this. Spot oracle (a regressor round-trip,
  shape + value preservation across the boundary):
  `python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; X=np.array([[1.],[2.],[3.]]); y=np.array([2.,4.,6.]); m=LinearRegression().fit(X,y); print(m.predict(np.array([[4.]])).tolist())"`
  → `[8.0]`; `import ferrolearn; ferrolearn.LinearRegression().fit(X,y).predict([[4.]])`
  matches (covered by `tests/` regression round-trip).
- AC-LABEL-MARSHAL (REQ-LABEL-MARSHAL): the wrapper encodes/decodes labels so the
  original labels survive a fit/predict round-trip, matching sklearn. Oracle
  (negative labels):
  `python3 -c "import numpy as np; from sklearn.linear_model import LogisticRegression; X=np.array([[0.],[1.],[2.],[3.]]); y=[-1,1,-1,1]; m=LogisticRegression().fit(X,y); print(m.classes_.tolist(), m.predict(X).tolist())"`
  → `[-1, 1] [-1, -1, 1, 1]`. `ferrolearn.LogisticRegression().fit(X, y)` returns
  `classes_ == [-1, 1]` and `predict` returns `-1`/`1` (NOT `u64::MAX`) BECAUSE
  `_encode_labels` maps `[-1,1]` → `[0,1]` before the Rust `v as usize` and
  `_decode_labels` maps back — verified by the classifier tests in `tests/`. The
  raw-binding hazard: `np.uint64(np.int64(-1))` → `18446744073709551615` shows
  what `v as usize` would do WITHOUT the wrapper guard.
- AC-DTYPE-COVERAGE (REQ-DTYPE-COVERAGE): sklearn preserves float32:
  `python3 -c "import numpy as np; from sklearn.utils import check_array; print(check_array(np.array([[1.,2.]], dtype=np.float32), dtype='numeric').dtype)"`
  → `float32`. ferrolearn up-casts: the wrapper's
  `_validate_data(dtype="float64")` + `np.ascontiguousarray(_, dtype=np.float64)`
  forces float64, and `conversions.rs` only has `PyReadonlyArray2<f64>` — a
  critic pins a FAILING pytest asserting a float32 input keeps float32 through
  the ferrolearn estimator. FAILS until float32 marshalling lands.
- AC-SPARSE-INPUT (REQ-SPARSE-INPUT): sklearn accepts sparse X:
  `python3 -c "import numpy as np, scipy.sparse as sp; from sklearn.linear_model import Ridge; X=sp.csr_matrix(np.eye(3)); y=np.array([1.,2.,3.]); print(Ridge().fit(X,y).coef_.shape)"`
  → `(3,)`. `conversions.rs` has no sparse marshaller (`grep -n "PyReadonly"
  conversions.rs` shows only dense `PyReadonlyArray{1,2}`); a critic pins a
  FAILING pytest passing `scipy.sparse` X to a ferrolearn estimator. FAILS until
  sparse marshalling lands.
- AC-FERRAY (REQ-FERRAY): `conversions.rs` imports `use ndarray::{Array1,
  Array2}` and `use numpy::{PyArray1, PyArray2, PyReadonlyArray1,
  PyReadonlyArray2}` — the WRONG substrate per R-SUBSTRATE-1 (numpy bridge →
  `ferray::numpy_interop`; array type → `ferray-core`). ferray does not yet
  expose a `numpy_interop` PyO3 bridge surface consumable here (R-SUBSTRATE-5).
- AC-CONSUMER (REQ-CONSUMER):
  `grep -rn "conversions::\|numpy2_to_ndarray\|numpy1_to_ndarray\|ndarray.*_to_numpy" /home/doll/ferrolearn/ferrolearn-python/src/*.rs | grep -v 'tests/'`
  shows `regressors.rs`, `classifiers.rs`, `transformers.rs`, `clusterers.rs`,
  and `extras.rs` each `use crate::conversions::*` and call
  `numpy2_to_ndarray(x)` / `numpy1_to_ndarray(y)` / `numpy1_to_ndarray_usize(y)`
  in `fit`, and `ndarray1_to_numpy`/`ndarray2_to_numpy`/`ndarray1_usize_to_numpy`
  in `predict`/`transform`/getters. The 513-passing pytest exercises every one.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-F64-ROUNDTRIP (f64 dtype/shape round-trip) | SHIPPED | impl `pub fn numpy2_to_ndarray`/`numpy1_to_ndarray in conversions.rs` (`x.as_array().to_owned()`) + `pub fn ndarray2_to_numpy`/`ndarray1_to_numpy in conversions.rs` (`PyArray{1,2}::from_array(py, a)`) copy shape + f64 values exactly across the boundary. The f64 dtype contract is enforced UPSTREAM in Python — `_classifiers.py`/`_regressors.py`/`_transformers.py`/`_clusterers.py` call `self._validate_data(..., dtype="float64")` + `np.ascontiguousarray(arr, dtype=np.float64)` before the Rust call — so the Rust side assumes f64 by contract, an explicit Python coercion not a silent Rust cast (R-CODE-4; cf. sklearn `check_array(dtype=np.float64)`, `validation.py:915-942`). Non-test consumer: every estimator binding round-trips arrays through these (e.g. `regressors.rs` `numpy2_to_ndarray(x)` in `fit`, `ndarray1_to_numpy(py, &preds)` in `predict`). Verification (model B): `cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` → 513 passed, 0 failed (every fit/predict/transform round-trips arrays). |
| REQ-LABEL-MARSHAL (i64↔usize labels, wrapper-guarded) | SHIPPED | impl `pub fn numpy1_to_ndarray_usize in conversions.rs` (`y.as_array().mapv(\|v\| v as usize)`) + `pub fn ndarray1_usize_to_numpy in conversions.rs` (`a.mapv(\|v\| v as i64)`) marshal labels, SAFE because the Python classifier wrappers establish the non-negative-contiguous-int64 contract: `_classifiers.py::_encode_labels` does `np.unique(y)` → `0..n_classes-1` int64 (a `LabelEncoder` equivalent) before the Rust `fit`, and `_decode_labels` does `classes_[y_encoded]` after `predict`, mirroring sklearn's internal label encoding (`column_or_1d`/`_check_y`, `validation.py:1348`/`:1325`; sklearn keeps original labels in `classes_`). Live oracle (R-CHAR-3): `LogisticRegression().fit(X,[-1,1,-1,1]).classes_` → `[-1,1]`, `predict` → `[-1,-1,1,1]`; ferrolearn returns the same because `-1` is encoded to `0` BEFORE `v as usize`. Raw-binding hazard documented (`(-1_i64) as usize` = `18446744073709551615`) — no production path reaches the raw `_Rs*` binding un-encoded. Non-test consumer: `classifiers.rs` (`numpy1_to_ndarray_usize(y)` in every classifier `fit`, `ndarray1_usize_to_numpy(py, &preds)` in `predict`) + `clusterers.rs`/`extras.rs` (cluster labels). Verification (model B): pytest classifier tests → green (negative/string labels round-trip via the wrapper). |
| REQ-DTYPE-COVERAGE (preserve float32/int dtype) | NOT-STARTED | blocker issue to be filed by critic. sklearn `check_array(dtype='numeric')` PRESERVES float32 (`validation.py:854-913`; live oracle: `check_array(float32_arr, dtype='numeric').dtype` → `float32`), and preserves int32/int64 where the estimator allows. `conversions.rs` handles ONLY f64 (`numpy2_to_ndarray: PyReadonlyArray2<f64>`) and the wrappers FORCE float64 (`_validate_data(dtype="float64")` + `np.ascontiguousarray(_, dtype=np.float64)`), so a float32 input is up-cast to float64 — a dtype change vs sklearn preserving float32 (R-CODE-5). No f32 marshalling path exists. |
| REQ-SPARSE-INPUT (scipy.sparse X) | NOT-STARTED | blocker issue to be filed by critic. sklearn estimators accept `scipy.sparse` X (live oracle: `Ridge().fit(sp.csr_matrix(np.eye(3)), y).coef_.shape` → `(3,)`). `conversions.rs` only marshals dense `PyReadonlyArray{1,2}` — there is no sparse (CSR/CSC) numpy↔ferrolearn bridge, so sparse input cannot cross the boundary. (Depends on the ferrolearn-sparse `CsrMatrix` surface + a ferray sparse-interop layer.) |
| REQ-FERRAY (ferray::numpy_interop substrate) | NOT-STARTED | blocker issue to be filed by critic. `conversions.rs` imports `use ndarray::{Array1, Array2}` + `use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2}` (rust-numpy) — the WRONG substrate per R-SUBSTRATE-1 (the PyO3 numpy bridge destination is `ferray::numpy_interop`; the array type is `ferray-core`, not `ndarray`). ferray does not yet expose a `numpy_interop` bridge consumable here (R-SUBSTRATE-5: the ferray gap is real work, filed upstream to ferray; this REQ is NOT-STARTED until ferray ships the interop layer). |
| REQ-CONSUMER (non-test estimator-binding callers) | SHIPPED | every estimator binding is a REAL non-test production consumer: `regressors.rs`, `classifiers.rs`, `transformers.rs`, `clusterers.rs`, and `extras.rs` each `use crate::conversions::*` and call `numpy2_to_ndarray(x)`/`numpy1_to_ndarray(y)`/`numpy1_to_ndarray_usize(y)` in `fit` and `ndarray1_to_numpy`/`ndarray2_to_numpy`/`ndarray1_usize_to_numpy` in `predict`/`transform`/getters (e.g. `classifiers.rs` `RsLogisticRegression::fit` → `numpy2_to_ndarray`/`numpy1_to_ndarray_usize`; `RsLogisticRegression::predict` → `ndarray1_usize_to_numpy`). These are the binding shim's data plane — grandfathered existing pub API with genuine downstream consumers (R-DEFER-1/S5). Verification (model B): the consumer grep above + `cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` → 513 passed (the full estimator surface drives these conversions). |

## Architecture

`conversions.rs` is a six-function free-function module with no types — the
data-plane half of the `ferrolearn-python` boundary. The estimator bindings own
control flow (constructor params, fit/predict dispatch, `PyResult` error
mapping); `conversions.rs` owns only the array marshalling:

- **Inbound (numpy → ndarray).** `numpy2_to_ndarray`/`numpy1_to_ndarray` take a
  `PyReadonlyArray{2,1}<f64>` and return an owned `Array{2,1}<f64>` via
  `x.as_array().to_owned()` — a copy, so the borrow on the numpy buffer is
  released and the library crate gets an owned array. `numpy1_to_ndarray_usize`
  takes a `PyReadonlyArray1<i64>` and maps `v as usize` to produce the
  `Array1<usize>` the classifier/cluster library crates consume. The f64 dtype
  is guaranteed by the Python wrapper (`_validate_data(dtype="float64")`), and
  the usize labels are guaranteed non-negative contiguous by the wrapper's
  `_encode_labels` (REQ-F64-ROUNDTRIP + REQ-LABEL-MARSHAL SHIPPED under those
  contracts).
- **Outbound (ndarray → numpy).** `ndarray1_to_numpy`/`ndarray2_to_numpy`
  allocate a fresh `PyArray{1,2}<f64>` via `PyArray::from_array(py, a)` (a copy
  back into a numpy-owned buffer). `ndarray1_usize_to_numpy` maps `v as i64`
  first (since rust-numpy has no native usize numpy dtype), then `from_array` —
  the labels are decoded back to the original space by the wrapper's
  `_decode_labels` (`classes_[y_encoded]`).

The two cross-cutting facts mirror the `csr.md` house pattern. REQ-CONSUMER is
firmly SHIPPED — unlike a leaf utility, these six functions are called by EVERY
estimator binding on EVERY fit/predict/transform; the type-boundary is the
binding shim's reason for existing. REQ-FERRAY is NOT-STARTED — rust-numpy +
`ndarray` is the wrong substrate per R-SUBSTRATE-1 (the destination is
`ferray::numpy_interop` + `ferray-core`), and ferray has no PyO3 interop layer
yet (R-SUBSTRATE-5). The honest call (R-HONEST-3): the f64 round-trip and the
consumer ship on impl + the 513-passing pytest; the label-marshal ships ONLY
under the wrapper-guarded non-negative-contiguous-int64 contract (the
`_encode_labels`/`_decode_labels` layer in `_classifiers.py`, NOT in
`conversions.rs`) — with the raw-binding negative/string hazard documented; and
dtype-coverage, sparse-input, and ferray-substrate do not ship.

The least-confident SHIPPED claim is REQ-LABEL-MARSHAL: it is SHIPPED only
because label encoding happens in the Python wrapper above the Rust cast. The
`v as usize` in `conversions.rs` is, in isolation, a R-CODE-5 cast-hiding hazard;
it is SHIPPED purely on the wrapper guarantee. If a future binding added a
classifier path that called `numpy1_to_ndarray_usize` on un-encoded labels, this
REQ would regress to NOT-STARTED and the `(-1) as usize` wrap would be a live
divergence.

## Verification

Commands establishing the SHIPPED claims (run at baseline `b64f72e5e`,
verification model B):

- **Full pytest gauntlet (REQ-F64-ROUNDTRIP, REQ-LABEL-MARSHAL, REQ-CONSUMER):**
  `cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`
  → `513 passed`. Every estimator fit/predict/transform round-trips arrays
  through these conversions; the classifier tests round-trip labels through
  `_encode_labels`/`_decode_labels` + `numpy1_to_ndarray_usize`/
  `ndarray1_usize_to_numpy`. (Rebuild first if the Rust side changed:
  `cd ferrolearn-python && maturin develop`.)
- **Label oracle (REQ-LABEL-MARSHAL; R-CHAR-3 — expected from sklearn):**
  `python3 -c "import numpy as np; from sklearn.linear_model import LogisticRegression; X=np.array([[0.],[1.],[2.],[3.]]); y=[-1,1,-1,1]; m=LogisticRegression().fit(X,y); print(m.classes_.tolist(), m.predict(X).tolist())"`
  → `[-1, 1] [-1, -1, 1, 1]`. ferrolearn matches because `_encode_labels` maps
  `[-1,1]`→`[0,1]` before `v as usize`. The wrap that the cast WOULD produce
  without the guard: `python3 -c "import numpy as np; print(np.uint64(np.int64(-1)))"`
  → `18446744073709551615`.
- **Dtype oracle (REQ-DTYPE-COVERAGE — sklearn preserves float32):**
  `python3 -c "import numpy as np; from sklearn.utils import check_array; print(check_array(np.array([[1.,2.]], dtype=np.float32), dtype='numeric').dtype, check_array(np.array([[1.,2.]], dtype=np.float32), dtype=np.float64).dtype)"`
  → `float32 float64`. ferrolearn forces float64 (wrapper `dtype="float64"` +
  `conversions.rs` f64-only) — a critic pins a FAILING pytest asserting float32
  survives. FAILS until f32 marshalling lands.
- **Sparse oracle (REQ-SPARSE-INPUT — sklearn accepts sparse X):**
  `python3 -c "import numpy as np, scipy.sparse as sp; from sklearn.linear_model import Ridge; X=sp.csr_matrix(np.eye(3)); print(Ridge().fit(X, np.array([1.,2.,3.])).coef_.shape)"`
  → `(3,)`. `conversions.rs` has no sparse marshaller — a critic pins a FAILING
  pytest passing `scipy.sparse` X to a ferrolearn estimator. FAILS until sparse
  marshalling lands.
- **Consumer check (REQ-CONSUMER):**
  `grep -rn "conversions::\|numpy2_to_ndarray\|numpy1_to_ndarray\|ndarray.*_to_numpy" /home/doll/ferrolearn/ferrolearn-python/src/*.rs | grep -v 'tests/'`
  shows `regressors.rs`/`classifiers.rs`/`transformers.rs`/`clusterers.rs`/
  `extras.rs` each `use crate::conversions::*` and call the marshallers in
  `fit`/`predict`/`transform`/getters — non-test, production, exercised by the
  513-passing pytest.
- **Substrate check (REQ-FERRAY):** `conversions.rs` head shows
  `use ndarray::{Array1, Array2};` and
  `use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};` — the
  wrong substrate per R-SUBSTRATE-1; ferray exposes no `numpy_interop` bridge
  yet (R-SUBSTRATE-5).
