# MaxAbsScaler

<!--
tier: 3-component
status: draft
baseline-commit: aaf48e23
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_data.py  # class MaxAbsScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (:1116); _parameter_constraints {copy:["boolean"]} (:1188); __init__(*, copy=True) (:1190). fit (:1212-1230) resets then -> partial_fit; partial_fit (:1232-1273): _validate_data(accept_sparse=("csr","csc"), dtype=supported_float_dtypes, force_all_finite="allow-nan") (:1252-1257) ALLOWS NaN + sparse; dense max_abs = _array_api._nanmax(abs(X), axis=0) (:1263), sparse min_max_axis(ignore_nan=True) (:1260-1261); first_pass sets n_samples_seen_=X.shape[0] (:1266) else max_abs=maximum(prev,batch) + n_samples_seen_+=batch (:1267-1269); max_abs_ stored (:1271); scale_ = _handle_zeros_in_scale(max_abs, copy=True) (:1272) -> a max_abs of 0 becomes scale_=1. transform (:1275-1306): X /= scale_ (:1305). inverse_transform (:1308-1344): X *= scale_ (:1337). _handle_zeros_in_scale (:88). maxabs_scale(X, *, axis=0, copy=True) free fn (:1351), axis Options{0,1} (:1347).
ferrolearn-module: ferrolearn-preprocess/src/max_abs_scaler.rs
parity-ops: MaxAbsScaler, maxabs_scale
crosslink-issue: 1201
-->

## Summary

scikit-learn's `MaxAbsScaler` (`_data.py:1116`) scales each **feature (column)**
independently by its maximum absolute value so that every column lands in
`[-1, 1]`: it learns `max_abs_ = _nanmax(abs(X), axis=0)` (`:1263`) on `fit`,
derives `scale_ = _handle_zeros_in_scale(max_abs_)` (`:1272`, a `max_abs_` of 0
becomes `scale_ = 1`), and on `transform` applies `X /= scale_` (`:1305`). It
does not center the data (sparsity-preserving). It stores
`max_abs_`/`scale_`/`n_samples_seen_`/`n_features_in_`, allows NaN
(`force_all_finite="allow-nan"`, `:1256`) and sparse CSR/CSC input
(`accept_sparse=("csr","csc")`), supports a `copy` constructor param, an
`inverse_transform` (`X *= scale_`, `:1337`), `partial_fit` streaming
(`:1232`), `get_feature_names_out`/`n_features_in_`, and a `maxabs_scale` free
function with an `axis` argument (`:1351`).

`ferrolearn-preprocess/src/max_abs_scaler.rs` ships a **dense, non-streaming**
estimator with the unfitted/fitted split: `MaxAbsScaler<F>` (a unit
`PhantomData` carrier — NO params; `new()`/`Default`) and
`FittedMaxAbsScaler<F> { max_abs: Array1<F> }` (accessor `max_abs()`,
`inverse_transform`). `impl Fit<Array2<F>,()>` rejects zero rows
(`InsufficientSamples`) and computes per-column `max_abs = max(|x|)` by a fold.
`impl Transform<Array2<F>>` returns `ShapeMismatch` on a column-count mismatch
and applies `x / max_abs` per column, **leaving a `max_abs == 0` column
unchanged**. It also provides an unfitted-`Transform` shim (always errors),
`FitTransform`, and `PipelineTransformer`/`FittedPipelineTransformer` impls.
Non-test consumers: the crate re-export `pub use
max_abs_scaler::{FittedMaxAbsScaler, MaxAbsScaler};`
(`ferrolearn-preprocess/src/lib.rs` line 117), the in-file pipeline impls, and a
PyO3 binding `_RsMaxAbsScaler` (`ferrolearn-python/src/extras.rs` line 1156,
registered `ferrolearn-python/src/lib.rs` line 82) wrapping
`MaxAbsScaler::<f64>::new()` fit/transform.

**Headline finding (document prominently — VERIFY-AND-DOCUMENT, no fixable
divergence): the zero-`max_abs` column handling MATCHES sklearn.** Unlike
`MinMaxScaler`/`StandardScaler` (whose "leave constant column unchanged" branch
diverges from sklearn's map-to-`fr[0]`/`0`), a column with `max_abs == 0` is
necessarily **all-zero**. sklearn computes `scale_ = _handle_zeros_in_scale(0) =
1` (`:1272`,`:88`), so `transform = X / 1 = X` — identity. ferrolearn `continue`s
on that column, also leaving it `= X` — identity. The two coincide for **any**
input: for the fitted (all-zero) column both give `0`; for a post-fit non-zero
input both give `x` unchanged. Live oracle confirms (REQ-2 Probe): all-zero col 0
→ `[0,0,0]` in both; `fit([[0],[0]])` then `transform([[5]])` → `[[5.0]]` in both
(`x/scale_(1)` vs leave-unchanged). The same coincidence holds for
`inverse_transform` (`x*scale_(1) = x` vs leave-unchanged). **No observable
divergence exists on this edge case** — REQ-2 is a SHIPPED green guard, not a
pin. The non-zero per-column value match (REQ-1), `inverse_transform` round-trip
(REQ-3), and the PyO3 marshalling (REQ-4) also SHIP.

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — core per-column max-abs value match (mixed-sign columns, deterministic):
python3 -c "from sklearn.preprocessing import MaxAbsScaler; \
print(MaxAbsScaler().fit_transform([[-3.,1.],[0.,-2.],[2.,4.]]).tolist()); \
m=MaxAbsScaler().fit([[-3.,1.],[0.,-2.],[2.,4.]]); \
print('max_abs_', m.max_abs_.tolist(), 'scale_', m.scale_.tolist())"
# -> [[-1.0, 0.25], [0.0, -0.5], [0.6666666666666666, 1.0]]
#    (col0 max_abs=3 -> [-1, 0, 0.6667]; col1 max_abs=4 -> [0.25, -0.5, 1])
# -> max_abs_ [3.0, 4.0] scale_ [3.0, 4.0]
#    ferrolearn: MaxAbsScaler::<f64>::new().fit(&X,&()).transform(&X) == same, bit-for-bit;
#    max_abs() == [3.0, 4.0] (== max_abs_; note ferrolearn has no separate scale_).

# REQ-1b — all-negative column value-match:
python3 -c "from sklearn.preprocessing import MaxAbsScaler; \
print(MaxAbsScaler().fit_transform([[-5.],[-3.],[-1.]]).tolist())"
# -> [[-1.0], [-0.6], [-0.2]]   (max_abs=5)
#    ferrolearn: same.

# REQ-2 — zero-max_abs (all-zero) column MATCHES sklearn (green guard, NOT a pin):
python3 -c "from sklearn.preprocessing import MaxAbsScaler; \
print('zerocol', MaxAbsScaler().fit_transform([[0.,1.],[0.,2.],[0.,3.]]).tolist()); \
m=MaxAbsScaler().fit([[0.],[0.]]); \
print('fitzero_then5', m.transform([[5.]]).tolist(), 'scale_', m.scale_.tolist())"
# -> zerocol [[0.0, 0.333...], [0.0, 0.666...], [0.0, 1.0]]   (col0 all-zero -> 0)
# -> fitzero_then5 [[5.0]] scale_ [1.0]   (scale_=_handle_zeros_in_scale(0)=1 -> 5/1=5)
#    ferrolearn: leaves max_abs==0 col unchanged -> col0 = [0,0,0] (== input), 5.0 stays 5.0.
#    BOTH give x for any input (x/1 == x == leave-unchanged). MATCH — no divergence.

# REQ-3 — inverse_transform round-trip (incl. zero-max_abs col, also a MATCH):
python3 -c "from sklearn.preprocessing import MaxAbsScaler; \
m=MaxAbsScaler().fit([[-3.,1.],[0.,-2.],[2.,4.]]); \
t=m.transform([[-3.,1.],[0.,-2.],[2.,4.]]); print(m.inverse_transform(t).tolist())"
# -> [[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]]   (inverse(transform(X)) == X)
#    ferrolearn: FittedMaxAbsScaler::inverse_transform applies x*max_abs per col
#    (zero-max_abs col left unchanged -> x*scale_(1)=x in sklearn; both x). MATCH.

# REQ-4 — PyO3 binding (_RsMaxAbsScaler) is a real CPython fit/transform consumer:
#   ferrolearn-python/src/extras.rs line 1156 registers _RsMaxAbsScaler over
#   FittedMaxAbsScaler<f64> with build-block MaxAbsScaler::<f64>::new();
#   lib.rs line 82 add_class. fit() marshals PyReadonlyArray2 -> Array2 -> .fit(&x,&());
#   transform() marshals fitted.transform(&x) -> PyArray2.

# REQ-5 — NaN tolerance (force_all_finite="allow-nan" + _nanmax, :1256,:1263):
python3 -c "import numpy as np; from sklearn.preprocessing import MaxAbsScaler; \
print(MaxAbsScaler().fit_transform(np.array([[1.],[np.nan],[3.]])).tolist())"
# -> [[0.333...], [nan], [1.0]]   (NaN ignored for max_abs -> max_abs=3; NaN passes through)
#    ferrolearn: the fold (acc, v) -> if v > acc { v } else { acc } leaves acc on a NaN
#    compare (NaN > acc is false), so max_abs ignores NaN like _nanmax HERE, BUT there is
#    no force_all_finite contract and NaN handling is not pinned -> NOT-STARTED (do NOT
#    pin a non-finite REJECTION; MaxAbsScaler must ALLOW NaN, :1256).

# REQ-8 — maxabs_scale free fn with axis (:1351, axis Options{0,1} :1347):
python3 -c "from sklearn.preprocessing import maxabs_scale; \
print(maxabs_scale([[1.,2.],[3.,4.]], axis=1).tolist())"
# -> [[0.5, 1.0], [0.75, 1.0]]   (axis=1 scales each ROW by its max abs)
#    ferrolearn: no maxabs_scale free fn (estimator is hard-wired axis=0/per-column).
```

## Requirements

- REQ-1: Per-column max-abs value match — learn `max_abs[j] = max(|x|)` per
  column on `fit` (sklearn `max_abs_ = _nanmax(abs(X), axis=0)`, `:1263`), apply
  `x / max_abs` per column on `transform` (sklearn `X /= scale_`, `:1305`, with
  `scale_ = max_abs` for non-zero columns); reject zero rows; `ShapeMismatch`
  on column-count mismatch; expose a `max_abs()` accessor matching sklearn
  `max_abs_`. Mixed-sign and all-negative columns. Supports `f32` and `f64`
  (REQ-1, REQ-1b Probes).
- REQ-2: **Zero-`max_abs` (all-zero) column → identity** — a column whose
  `max_abs == 0` (necessarily all-zero) must be left unchanged, matching
  sklearn's `scale_ = _handle_zeros_in_scale(0) = 1` (`:1272`,`:88`) → `X / 1 =
  X`. **This MATCHES (verify-and-document, NOT a divergence):** ferrolearn
  `continue`s and sklearn divides by 1 — both are identity for any input,
  including a post-fit non-zero value (REQ-2 Probe: `fit([[0],[0]])` then
  `transform([[5]])` → `[[5.0]]` in both).
- REQ-3: `inverse_transform` (`:1308-1344`) — apply `x * max_abs` per column
  (sklearn `X *= scale_`, `:1337`), round-tripping `inverse(transform(X)) == X`;
  zero-`max_abs` column left unchanged (matches sklearn `x*1 = x`); `ShapeMismatch`
  on column-count mismatch (REQ-3 Probe).
- REQ-4: PyO3 binding (`import ferrolearn` exposes a `MaxAbsScaler` marshalling
  `fit`/`transform` over `FittedMaxAbsScaler<f64>`) — the project boundary
  CPython consumer (REQ-4 Probe).
- REQ-5: NaN tolerance — `fit`/`transform` honoring `force_all_finite="allow-nan"`
  (`:1256`) with `_nanmax` (`:1263`) so NaN entries are ignored when computing
  `max_abs` and pass through `transform` unchanged, as an explicit contract
  (REQ-5 Probe: `[[1],[NaN],[3]]` → `[[0.333],[nan],[1]]`). Must ALLOW NaN, never
  reject it.
- REQ-6: `scale_` / `n_samples_seen_` fitted attribute surface — sklearn stores
  `scale_ = _handle_zeros_in_scale(max_abs_)` (`:1272`) and `n_samples_seen_`
  (`:1266`) alongside `max_abs_`. ferrolearn exposes `max_abs()` but neither
  `scale_` (= `_handle_zeros`-adjusted: 0→1) nor `n_samples_seen_`.
- REQ-7: `partial_fit` (`:1232-1273`) — streaming `max_abs = maximum(prev, batch)`
  (`:1267`) with `n_samples_seen_ += batch_rows` (`:1269`) across calls.
- REQ-8: `maxabs_scale(X, *, axis=0, copy=True)` standalone free function
  including `axis=1` row-scaling (`:1351`, `axis` Options{0,1} `:1347`).
- REQ-9: `copy` constructor parameter + `_parameter_constraints {copy:["boolean"]}`
  — in-place-vs-copy semantics (`__init__` `copy=True`, `:1190`,`:1188`).
  ferrolearn `MaxAbsScaler<F>` has NO params and `transform` always `to_owned()`s.
- REQ-10: Sparse CSR/CSC support — `accept_sparse=("csr","csc")` with
  `min_max_axis(ignore_nan=True)` (`:1260-1261`) and `inplace_column_scale`
  (`:1303`,`:1335`). ferrolearn is dense-only (`Array2<F>`).
- REQ-11: `get_feature_names_out` / `n_features_in_` / `feature_names_in_`
  (OneToOneFeatureMixin one-to-one passthrough; set on `fit`).
- REQ-12: ferray substrate — compute over `ferray-core` arrays / `ferray-ufunc`
  rather than `ndarray::Array2` + `num_traits::Float` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `MaxAbsScaler::<f64>::new().fit(&X).transform(&X)` for
  `X=[[-3,1],[0,-2],[2,4]]` equals `[[-1,0.25],[0,-0.5],[2/3,1]]` and
  `[[-5],[-3],[-1]]` equals `[[-1],[-0.6],[-0.2]]` within ULP tolerance (REQ-1,
  REQ-1b Probes); `max_abs() == [3,4]`; `fit` on `(0,n)` returns
  `Err(InsufficientSamples)`; a column-count mismatch on `transform` returns
  `Err(ShapeMismatch)`. Pinned by an oracle-grounded `#[test]`.
- AC-2 (REQ-2): `MaxAbsScaler::<f64>::new().fit_transform([[0,1],[0,2],[0,3]])`
  has col 0 `== [0,0,0]` (matching sklearn), and `fit([[0],[0]])` then
  `transform([[5]])` `== [[5.0]]` (matching sklearn `x/scale_(1)`). A green guard
  test asserting the MATCH (not a divergence pin).
- AC-3 (REQ-3): `inverse_transform(transform(X)) == X` for the REQ-1 fixture
  (REQ-3 Probe); a column-count mismatch returns `Err(ShapeMismatch)`.
- AC-4 (REQ-4): `python3 -c "import ferrolearn; ..."` resolves the registered
  `_RsMaxAbsScaler`; `.fit(X).transform(X)` matches the REQ-1 Probe.
- AC-5 (REQ-5): `fit_transform` of `[[1],[NaN],[3]]` yields `[[0.333],[NaN],[1]]`
  (NaN ignored for `max_abs`, passes through) as a contracted, pinned behavior.
- AC-6 (REQ-6): a fitted handle exposes `scale_ == [3,4]` (`_handle_zeros`:
  0→1) and `n_samples_seen_ == 3` for the REQ-1 fit.
- AC-7 (REQ-7): two `partial_fit` batches accumulate the global `max_abs` and
  `n_samples_seen_` equals the total row count.
- AC-8 (REQ-8): a free `maxabs_scale(&X, axis, copy)` with `axis=1` reproduces
  `[[0.5,1],[0.75,1]]` for `[[1,2],[3,4]]` (REQ-8 Probe).
- AC-9 (REQ-9): a `copy=false` path is observably in-place; `_parameter_constraints`
  analog validates `copy`.
- AC-10 (REQ-10): a CSR/CSC sparse input scales column-wise without densifying.
- AC-11 (REQ-11): `get_feature_names_out` returns `['x0','x1']` for a 2-feature
  fit; `n_features_in_ == 2`.
- AC-12 (REQ-12): the owned transform computes on `ferray-core` arrays.

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (per-column max-abs value match) | SHIPPED | impl `pub fn fit in max_abs_scaler.rs` (`Fit<Array2<F>,()> for MaxAbsScaler<F>`) rejects 0 rows (`InsufficientSamples`) and sets `max_abs[j]` via the per-column fold `(acc, v) -> if v > acc { v } else { acc }` over `\|x\|`; impl `Transform<Array2<F>> for FittedMaxAbsScaler<F>::transform` divides each non-zero column by `max_abs[j]` (`*v = *v / ma`), returning `ShapeMismatch` when `x.ncols() != n_features`. Mirrors sklearn `max_abs_ = _nanmax(abs(X), axis=0)` (`_data.py:1263`), `scale_ = _handle_zeros_in_scale(max_abs_)` (`:1272`), `X /= scale_` (`:1305`); `max_abs()` accessor mirrors `max_abs_`. Output equals REQ-1 Probe: `[[-1,0.25],[0,-0.5],[2/3,1]]`, REQ-1b `[[-1],[-0.6],[-0.2]]`, `max_abs_ = [3,4]`. Non-test consumers: (a) PyO3 `_RsMaxAbsScaler` (`ferrolearn-python/src/extras.rs` line 1156, registered `ferrolearn-python/src/lib.rs` line 82) — `py_transformer!` marshals `fit`/`transform` over `FittedMaxAbsScaler<f64>` from `MaxAbsScaler::<f64>::new()`; (b) in-file `impl FittedPipelineTransformer<F> for FittedMaxAbsScaler<F>::transform_pipeline` calls `self.transform(x)`; (c) crate re-export `pub use max_abs_scaler::{FittedMaxAbsScaler, MaxAbsScaler};` (`ferrolearn-preprocess/src/lib.rs` line 117), the boundary public API (grandfathered S5/R-DEFER-1). Verification: `cargo test -p ferrolearn-preprocess` (`test_max_abs_scaler_basic`, `test_values_in_range`, `test_negative_values`, `test_shape_mismatch_error`, `test_insufficient_samples_error`, `test_f32_scaler`). |
| REQ-2 (zero-max_abs column → identity, MATCHES sklearn) | SHIPPED | **VERIFY-AND-DOCUMENT — no divergence.** A `max_abs == 0` column is necessarily all-zero. `Transform::transform`'s `if ma == F::zero() { continue }` branch leaves it unchanged; sklearn computes `scale_ = _handle_zeros_in_scale(0) = 1` (`_data.py:1272`,`:88`) and applies `X / 1 = X` (`:1305`). Both are identity for ANY input: the fitted all-zero column → `0`; a post-fit non-zero input → unchanged. REQ-2 Probe oracle-confirmed: `fit_transform([[0,1],[0,2],[0,3]])` col 0 → `[0,0,0]` (both); `fit([[0],[0]]).transform([[5]])` → `[[5.0]]` (both, `5/scale_(1)` vs leave-unchanged). In-module `test_zero_column_unchanged` pins the all-zero fitted-column case and is CORRECT (asserts `0.0` = the matching value), unlike the MinMaxScaler/StandardScaler analogs which pinned a bug. Same consumers as REQ-1. Verification: `cargo test -p ferrolearn-preprocess test_zero_column_unchanged`; oracle REQ-2 Probe. |
| REQ-3 (inverse_transform round-trip) | SHIPPED | impl `pub fn inverse_transform in max_abs_scaler.rs` (`FittedMaxAbsScaler::inverse_transform`) multiplies each non-zero column by `max_abs[j]` (`*v = *v * ma`), leaving `max_abs == 0` columns unchanged, and returns `ShapeMismatch` on a column-count mismatch. Mirrors sklearn `X *= scale_` (`_data.py:1337`) with `scale_ = max_abs_` (non-zero) / `1` (zero-col → both leave `x`). REQ-3 Probe: `inverse_transform(transform(X)) == X` for `X=[[-3,1],[0,-2],[2,4]]`. Non-test consumer: the public method is reachable via the crate re-export boundary (`lib.rs` line 117); it is part of the grandfathered boundary public API surface (S5/R-DEFER-1) exposed to external users. Verification: `cargo test -p ferrolearn-preprocess test_inverse_transform_roundtrip`. Least-confident SHIPPED — `inverse_transform` has no in-tree non-test caller beyond the re-export boundary and is not yet wired to the PyO3 binding (the binding marshals only `fit`/`transform`); it is grandfathered as boundary API and its round-trip is oracle-confirmed. |
| REQ-4 (PyO3 binding) | SHIPPED | `_RsMaxAbsScaler` registered in `ferrolearn-python/src/extras.rs` lines 1156-1162 via `py_transformer!(RsMaxAbsScaler, "_RsMaxAbsScaler", ferrolearn_preprocess::FittedMaxAbsScaler<f64>, (), { ferrolearn_preprocess::MaxAbsScaler::<f64>::new() })`, added in `ferrolearn-python/src/lib.rs` line 82 (`m.add_class::<extras::RsMaxAbsScaler>()?`). The macro generates a `#[pyclass]` whose `fit` marshals `PyReadonlyArray2 -> Array2 -> MaxAbsScaler::<f64>::new().fit(&x_nd, &())` and `transform` marshals `fitted.transform(&x_nd) -> PyArray2` — a real CPython fit/transform consumer of REQ-1's impl. Thin marshalling layer; the numeric behavior it exposes is REQ-1's (and the matching REQ-2 edge case). Verification: `cargo build -p ferrolearn-python`; `python3 -c "import ferrolearn; ..."` round-trip against the REQ-1 Probe. |
| REQ-5 (NaN tolerance: allow-nan + nanmax) | NOT-STARTED | open prereq blocker #1202. ferrolearn's fold `(acc, v) -> if v > acc { v } else { acc }` happens to ignore NaN for `max_abs` (NaN comparisons are false), but there is no `force_all_finite="allow-nan"` contract (`:1256`), no test pinning NaN pass-through, and `transform`'s `x / max_abs` over a NaN entry is unverified against the oracle. REQ-5 Probe (`[[1],[NaN],[3]]` → `[[0.333],[NaN],[1]]`) is not contracted. (Do NOT pin a non-finite REJECTION — MaxAbsScaler must ALLOW NaN.) |
| REQ-6 (scale_ / n_samples_seen_ attributes) | NOT-STARTED | open prereq blocker #1203. `FittedMaxAbsScaler<F>` stores only `max_abs`; it never materializes `scale_ = _handle_zeros_in_scale(max_abs_)` (the `0→1`-adjusted divisor, `:1272`) nor `n_samples_seen_` (`:1266`). The `scale_` accessor and sample-count surface are inaccessible. (`max_abs()` matches `max_abs_` only; `scale_` differs from `max_abs_` exactly on all-zero columns.) |
| REQ-7 (partial_fit / streaming) | NOT-STARTED | open prereq blocker #1204. `Fit::fit` is single-shot; there is no `partial_fit` accumulating `max_abs = maximum(prev, batch)` (`:1267`) and `n_samples_seen_ += batch_rows` (`:1269`) across calls (sklearn `:1232-1273`). |
| REQ-8 (maxabs_scale free fn + axis) | NOT-STARTED | open prereq blocker #1205. No standalone `maxabs_scale` in `max_abs_scaler.rs` or the crate; the estimator is hard-wired to per-column (`axis=0`) scaling with no `axis=1` row path (sklearn free fn `:1351`, `axis` Options{0,1} `:1347`). REQ-8 Probe (`axis=1` → `[[0.5,1],[0.75,1]]`) unavailable. |
| REQ-9 (copy param + _parameter_constraints) | NOT-STARTED | open prereq blocker #1206. `MaxAbsScaler<F>` is a unit `PhantomData` carrier with NO fields; no `copy` param, no in-place path (`transform` always `to_owned()`s), no `_parameter_constraints {copy:["boolean"]}` analog (sklearn `__init__` `:1190`, constraints `:1188`). |
| REQ-10 (sparse CSR/CSC) | NOT-STARTED | open prereq blocker #1207. `Fit`/`Transform` are over dense `Array2<F>` only; no `accept_sparse=("csr","csc")` path, no `min_max_axis(ignore_nan=True)` (`:1260-1261`) or `inplace_column_scale` (`:1303`,`:1335`). MaxAbsScaler is sklearn's flagship sparse-safe scaler; sparse support is a core unmet contract. |
| REQ-11 (get_feature_names_out / n_features_in_) | NOT-STARTED | open prereq blocker #1208. No `n_features_in_`, `feature_names_in_`, or `get_feature_names_out` (OneToOneFeatureMixin one-to-one passthrough set on `fit`). |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1209. Compute path uses `ndarray::Array2` + `num_traits::Float` (`columns_mut`, manual fold), not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `max_abs_scaler.rs` exposes the unfitted/fitted pair.
`MaxAbsScaler<F> { _marker: PhantomData<F> }` is a parameterless carrier
constructed by `new()` (`Default = new()`). `FittedMaxAbsScaler<F> { max_abs:
Array1<F> }` exposes `max_abs()` and `inverse_transform`. `impl Fit<Array2<F>,
()>` rejects `n_samples == 0` (`InsufficientSamples`), then per column `j` folds
`max_abs[j] = fold(0, |acc, v| if v.abs() > acc { v.abs() } else { acc })` over
the column iterator. `impl Transform<Array2<F>> for FittedMaxAbsScaler<F>`
returns `ShapeMismatch` when `x.ncols() != n_features`, clones `x`, and for each
column: **if `max_abs[j] == F::zero()` it `continue`s (leaves the column
unchanged)** — which **matches** sklearn (all-zero column, `scale_ = 1`, identity)
— otherwise sets `*v = *v / max_abs[j]`. `inverse_transform` mirrors this with
`*v = *v * max_abs[j]` (zero-col left unchanged). A second `impl Transform for
MaxAbsScaler<F>` (unfitted) always errors (it exists to satisfy the `FitTransform:
Transform` supertrait); `impl FitTransform` chains `fit` then `transform`; `impl
PipelineTransformer<F>` (`fit_pipeline` boxes the fitted) and `impl
FittedPipelineTransformer<F>` (`transform_pipeline` calls `transform`) provide
pipeline integration. The generic bound `F: Float + Send + Sync + 'static`
supports `f32`/`f64`. The crate re-exports both public types
(`ferrolearn-preprocess/src/lib.rs` line 117), and `ferrolearn-python` registers
`_RsMaxAbsScaler` over `FittedMaxAbsScaler<f64>`.

**sklearn (target contract).** `MaxAbsScaler(OneToOneFeatureMixin,
TransformerMixin, BaseEstimator)` (`:1116`) stores `copy` (`__init__` `:1190`)
under `_parameter_constraints {copy:["boolean"]}` (`:1188`). `fit` (`:1212`)
resets state (`_reset`, `:1196`) and delegates to `partial_fit` (`:1232`), which
`_validate_data(accept_sparse=("csr","csc"), force_all_finite="allow-nan")`
(`:1252-1257`), computes `max_abs = _array_api._nanmax(abs(X), axis=0)` (`:1263`)
for dense (`min_max_axis(..., ignore_nan=True)` for sparse, `:1260-1261`); on the
first pass sets `n_samples_seen_ = X.shape[0]` (`:1266`), else
`max_abs = maximum(self.max_abs_, max_abs)` and `n_samples_seen_ += X.shape[0]`
(`:1267-1269`); stores `max_abs_` (`:1271`) and `scale_ =
_handle_zeros_in_scale(max_abs, copy=True)` (`:1272`). `transform` (`:1275`)
re-validates (`reset=False`, allow-nan) then `X /= scale_` (`:1305`).
`inverse_transform` (`:1308`) does `X *= scale_` (`:1337`). `_handle_zeros_in_scale`
(`:88`) replaces a `scale_` (here `max_abs`) of 0 with 1 — but because a
zero-`max_abs` column is all-zero, dividing by 1 and "leaving unchanged" coincide,
so this introduces **no** divergence from ferrolearn's `continue` (the key
contrast with MinMaxScaler, whose `_handle_zeros` maps a constant column to
`fr[0]` ≠ original). `maxabs_scale` (`:1351`) is the estimator-less free function
with an `axis` argument (`axis` Options{0,1}, `:1347`); `OneToOneFeatureMixin`
provides `get_feature_names_out`.

**The structural gap (no fixable divergence).** ferrolearn's per-column
divide-by-max-abs is an **exact match** for sklearn on dense, finite input,
**including** the zero-`max_abs` edge case (REQ-1, REQ-1b, REQ-2, REQ-3 Probes are
bit-identical / observably identical), and the PyO3 binding (REQ-4) ships. This is
a **verify-and-document** iteration: a full two-oracle audit finds **no behavioral
divergence to pin or fix** on the implemented surface. The remaining gaps are
*contract surface* not yet implemented: explicit NaN tolerance (REQ-5), the
`scale_`/`n_samples_seen_` attributes (REQ-6), `partial_fit` (REQ-7), the
`maxabs_scale` free fn (REQ-8), the `copy` param (REQ-9), sparse CSR/CSC
(REQ-10) — the most consequential miss given MaxAbsScaler's role as sklearn's
sparse-safe scaler — feature-name plumbing (REQ-11), and the ferray substrate
(REQ-12).

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-2, REQ-3, REQ-4):

```bash
# Oracle (REQ-1 Probe) — per-column max-abs, mixed-sign + all-negative:
python3 -c "from sklearn.preprocessing import MaxAbsScaler; \
print(MaxAbsScaler().fit_transform([[-3.,1.],[0.,-2.],[2.,4.]]).tolist()); \
print(MaxAbsScaler().fit_transform([[-5.],[-3.],[-1.]]).tolist())"
#   -> [[-1.0, 0.25], [0.0, -0.5], [0.6666666666666666, 1.0]]
#   -> [[-1.0], [-0.6], [-0.2]]
# ferrolearn equivalent: MaxAbsScaler::<f64>::new().fit(&X,&()).unwrap().transform(&X)

# Oracle (REQ-2 Probe) — zero-max_abs column MATCHES (green guard, not a pin):
python3 -c "from sklearn.preprocessing import MaxAbsScaler; \
print(MaxAbsScaler().fit_transform([[0.,1.],[0.,2.],[0.,3.]]).tolist()); \
m=MaxAbsScaler().fit([[0.],[0.]]); print(m.transform([[5.]]).tolist())"
#   -> [[0.0, 0.333...], [0.0, 0.666...], [0.0, 1.0]]  ;  [[5.0]]
#   ferrolearn: col0 all-zero -> [0,0,0]; transform([[5.]]) -> [[5.0]]. MATCH.

# Oracle (REQ-3 Probe) — inverse_transform round-trip:
python3 -c "from sklearn.preprocessing import MaxAbsScaler; \
m=MaxAbsScaler().fit([[-3.,1.],[0.,-2.],[2.,4.]]); \
print(m.inverse_transform(m.transform([[-3.,1.],[0.,-2.],[2.,4.]])).tolist())"
#   -> [[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]]

# Crate + binding gauntlet:
cargo test -p ferrolearn-preprocess   # incl. test_max_abs_scaler_basic,
                                      #       test_values_in_range,
                                      #       test_negative_values,
                                      #       test_zero_column_unchanged,
                                      #       test_inverse_transform_roundtrip,
                                      #       test_fit_transform_equivalence,
                                      #       test_shape_mismatch_error,
                                      #       test_insufficient_samples_error,
                                      #       test_f32_scaler
cargo build -p ferrolearn-python      # registers _RsMaxAbsScaler (REQ-4)
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s exercise REQ-1, REQ-2, and REQ-3 but are **not
oracle-grounded** (hand-written expected values that happen to match sklearn). To
satisfy R-CHAR-3 the critic should add an oracle-pinned guard asserting the REQ-1
Probe outputs and `max_abs()`, plus a REQ-2 green guard pinning the
`fit([[0],[0]]).transform([[5]]) == [[5.0]]` MATCH (the case where
MinMaxScaler/StandardScaler would diverge). Note `test_zero_column_unchanged`
correctly asserts the matching `0.0` (it is NOT a characterization-of-a-bug,
unlike the MinMaxScaler analog). No currently-green command establishes
REQ-5..REQ-12.

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns
`#`-numbers, replacing the `#<REQn>` placeholders); reference them in the REQ
status table. **No fixable-divergence blocker is filed: this is a
verify-and-document iteration with no behavioral divergence on the implemented
surface (the zero-`max_abs` edge case MATCHES sklearn).**

- #1202 — REQ-5: no `force_all_finite="allow-nan"` contract / pinned NaN
  pass-through; `_nanmax`-equivalence is incidental, not contracted (`:1256`,
  `:1263`).
- #1203 — REQ-6: `FittedMaxAbsScaler` stores only `max_abs`; no `scale_`
  (`_handle_zeros`-adjusted, `:1272`) or `n_samples_seen_` (`:1266`).
- #1204 — REQ-7: no `partial_fit` / streaming `max_abs = maximum(prev,batch)`
  + `n_samples_seen_` accumulation (`:1232-1273`).
- #1205 — REQ-8: no standalone `maxabs_scale` free fn; no `axis=1`
  row-scaling (`:1351`, `:1347`).
- #1206 — REQ-9: no `copy` param, no in-place path, no
  `_parameter_constraints {copy:["boolean"]}` (`:1190`,`:1188`).
- #1207 — REQ-10: no sparse CSR/CSC support
  (`accept_sparse=("csr","csc")`, `min_max_axis`, `inplace_column_scale`,
  `:1260-1261`,`:1303`,`:1335`) — the highest-value miss for this scaler.
- #1208 — REQ-11: no `n_features_in_`/`feature_names_in_`/
  `get_feature_names_out` (OneToOneFeatureMixin).
- #1209 — REQ-12: compute path on `ndarray`/`num_traits`, not ferray
  (R-SUBSTRATE-1/2).
