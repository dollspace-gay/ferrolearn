# MinMaxScaler

<!--
tier: 3-component
status: draft
baseline-commit: 71032cf3
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_data.py  # class MinMaxScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (:291); __init__(feature_range=(0,1), *, copy=True, clip=False) (:455-459); _parameter_constraints {feature_range:[tuple], copy:["boolean"], clip:["boolean"]} (:447-451). fit (:461-487) -> partial_fit; partial_fit (:489-515): _validate_data(force_all_finite="allow-nan") (:490-495) ALLOWS NaN, data_min=_nanmin(X,axis=0) (:498), data_max=_nanmax(X,axis=0) (:499), data_range=data_max-data_min (:507), scale_=(fr[1]-fr[0])/_handle_zeros_in_scale(data_range) (:508-510), min_=fr[0]-data_min*scale_ (:511), stores data_min_/data_max_/data_range_/scale_/min_/n_samples_seen_ (:511-514). transform (:517-547): _validate_data(copy=self.copy, force_writeable, force_all_finite="allow-nan", reset=False) (:531-538); X *= scale_ (:540); X += min_ (:541); if clip: clip(X, fr[0], fr[1], out=X) (:542-543). inverse_transform (:549-587). _handle_zeros_in_scale (:88): a scale (here data_range) of 0 is replaced by 1. minmax_scale(X, feature_range=(0,1), *, axis=0, copy=True) free fn (:589).
ferrolearn-module: ferrolearn-preprocess/src/min_max_scaler.rs
parity-ops: MinMaxScaler, minmax_scale
crosslink-issue: 1169
-->

## Summary

scikit-learn's `MinMaxScaler` (`_data.py:291`) rescales each **feature (column)**
independently to a target range `feature_range` (default `(0, 1)`): it learns
the per-column min/max on `fit` and on `transform` applies the affine map
`X *= scale_; X += min_` where `scale_ = (fr[1]-fr[0]) / _handle_zeros_in_scale(data_range)`
(`:508`) and `min_ = fr[0] - data_min*scale_` (`:511`). It stores
`data_min_/data_max_/data_range_/scale_/min_/n_samples_seen_`, allows NaN
(`force_all_finite="allow-nan"`) with `_nanmin`/`_nanmax`, supports `copy` and
`clip` constructor params, an `inverse_transform`, `partial_fit` (streaming),
`get_feature_names_out`/`n_features_in_`, and a `minmax_scale` free function with
an `axis` argument.

`ferrolearn-preprocess/src/min_max_scaler.rs` ships a **dense, non-streaming**
estimator with the unfitted/fitted split: `MinMaxScaler<F> { feature_range: (F,F) }`
(`new()` = `(0,1)`; `with_feature_range(min,max)` errors if `min >= max`;
`feature_range()` accessor) and `FittedMinMaxScaler<F> { data_min, data_max,
feature_range }` (accessors `data_min()`, `data_max()`, `feature_range()`).
`impl Fit<Array2<F>,()>` rejects zero rows (`InsufficientSamples`) and computes
per-column `data_min`/`data_max` by reduce-min/reduce-max. `impl
Transform<Array2<F>>` computes `(x - min)/(max - min) * range_width + range_min`
per column and returns `ShapeMismatch` on a column-count mismatch. It also
provides an unfitted-`Transform` shim (always errors), `FitTransform`, and
`PipelineTransformer`/`FittedPipelineTransformer` impls. Non-test consumers: the
crate re-export `pub use min_max_scaler::{FittedMinMaxScaler, MinMaxScaler};`
(`lib.rs` line 118), the in-file pipeline impls, and a PyO3 binding
`_RsMinMaxScaler` (`ferrolearn-python/src/extras.rs` line 1148, registered in
`ferrolearn-python/src/lib.rs` line 81) wrapping `MinMaxScaler::<f64>::new()`
(default range) fit/transform.

**Headline finding (document prominently — fixable this iteration):**
**constant / zero-range column handling DIVERGES.** sklearn maps a constant
column to `feature_range[0]` (0 by default), NOT to its original value: for a
constant column `data_range = 0` → `_handle_zeros_in_scale` → 1, so
`scale_ = range_width`, `min_ = fr[0] - data_min*range_width`, and
`transform(data_min) = data_min*range_width + fr[0] - data_min*range_width = fr[0]`.
ferrolearn instead LEAVES the column UNCHANGED (`if span == F::zero() { continue }`,
`transform` in `min_max_scaler.rs`). Live oracle: a constant column maps to
`0.0`, ferrolearn returns the original `5.0` — a DIVERGENCE (REQ-2, the fixable
one). The in-module `test_zero_range_column_unchanged` asserts the WRONG (`5.0`)
behavior and must flip to expect `0.0` (R-HONEST-4). The non-constant value-match
(REQ-1, default + custom range) and the PyO3 marshalling (REQ-7) SHIP.

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — core value match: non-constant columns, default + custom range (deterministic).
python3 -c "from sklearn.preprocessing import MinMaxScaler; \
print('default', MinMaxScaler().fit_transform([[1.,10.],[2.,20.],[3.,30.]]).tolist()); \
print('custom',  MinMaxScaler(feature_range=(-1,1)).fit_transform([[0.],[5.],[10.]]).tolist()); \
m=MinMaxScaler().fit([[1.,10.],[2.,20.],[3.,30.]]); \
print('data_min_', m.data_min_.tolist(), 'data_max_', m.data_max_.tolist())"
# -> default [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
# -> custom  [[-1.0], [0.0], [1.0]]
# -> data_min_ [1.0, 10.0] data_max_ [3.0, 30.0]
#    ferrolearn: MinMaxScaler::<f64>::new()/with_feature_range(-1,1).fit(&X).transform(&X)
#    == same, bit-for-bit; data_min()/data_max() == [1,10]/[3,30].

# REQ-2 — constant / zero-range column DIVERGENCE (the fixable one):
python3 -c "from sklearn.preprocessing import MinMaxScaler; \
print(MinMaxScaler().fit_transform([[5.,1.],[5.,2.],[5.,3.]]).tolist())"
# -> [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]   (constant col 0 -> feature_range[0] = 0.0, NOT 5.0)
#    ferrolearn: leaves constant col 0 at 5.0 -> [[5.0,0.0],[5.0,0.5],[5.0,1.0]]. DIVERGENCE.
# sklearn internal: data_range=0 -> _handle_zeros_in_scale -> 1 (:88);
#   scale_=range_width (:508); min_=fr[0]-data_min*range_width (:511);
#   transform(data_min)=data_min*range_width+fr[0]-data_min*range_width=fr[0] (:540-541).

# REQ-3 — feature_range validation (fr[0] < fr[1]):
python3 -c "from sklearn.preprocessing import MinMaxScaler; \
MinMaxScaler(feature_range=(1,0)).fit([[0.],[1.]])"
# -> ValueError: Minimum of desired feature range must be smaller than maximum. Got (1, 0). (:476-480)
#    ferrolearn: with_feature_range(1.0,0.0) -> Err(InvalidParameter); with_feature_range(1.0,1.0) -> Err. MATCH.
#    (Note: ferrolearn new() hard-codes (0,1); sklearn __init__ default (0,1) needs no separate check.)

# REQ-4 — NaN allowance (force_all_finite="allow-nan" + _nanmin/_nanmax):
python3 -c "import numpy as np; from sklearn.preprocessing import MinMaxScaler; \
print(MinMaxScaler().fit_transform(np.array([[1.],[np.nan],[3.]])).tolist())"
# -> [[0.0], [nan], [1.0]]   (NaN ignored for min/max -> data_min=1,data_max=3; NaN passes through)
#    ferrolearn: reduce-min/reduce-max propagate NaN differently (NaN poisons the column min/max).

# REQ-5 — scale_/min_/data_range_/n_samples_seen_ fitted attributes:
python3 -c "from sklearn.preprocessing import MinMaxScaler; \
m=MinMaxScaler().fit([[1.,10.],[2.,20.],[3.,30.]]); \
print(m.scale_.tolist(), m.min_.tolist(), m.data_range_.tolist(), m.n_samples_seen_)"
# -> [0.5, 0.05] [-0.5, -0.5] [2.0, 20.0] 3
#    ferrolearn FittedMinMaxScaler stores only data_min/data_max/feature_range.

# REQ-6 — inverse_transform round-trip:
python3 -c "from sklearn.preprocessing import MinMaxScaler; \
m=MinMaxScaler().fit([[1.],[2.],[3.]]); print(m.inverse_transform([[0.],[0.5],[1.]]).tolist())"
# -> [[1.0], [2.0], [3.0]]
#    ferrolearn: no inverse_transform.

# REQ-9 — minmax_scale free fn with axis:
python3 -c "from sklearn.preprocessing import minmax_scale; \
print(minmax_scale([[1.,10.],[2.,20.],[3.,30.]], axis=1).tolist())"
# -> [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]   (axis=1 scales each ROW)
#    ferrolearn: no minmax_scale free fn (estimator is hard-wired axis=0/per-column).

# REQ-7 — PyO3 binding (_RsMinMaxScaler) is a real CPython fit/transform consumer:
#   ferrolearn-python/src/extras.rs:1147-1153 registers _RsMinMaxScaler over
#   FittedMinMaxScaler<f64> with build-block MinMaxScaler::<f64>::new() (default range);
#   lib.rs:81 add_class. fit() marshals PyReadonlyArray2 -> Array2 -> .fit(&x,&());
#   transform() marshals fitted.transform(&x) -> PyArray2.
```

## Requirements

- REQ-1: Per-column min-max value match (non-constant columns) — learn
  `data_min`/`data_max` per column on `fit`, apply
  `(x - data_min)/(data_max - data_min) * (fr[1]-fr[0]) + fr[0]` on `transform`
  for the default range `(0,1)` and any custom range (sklearn `scale_`/`min_`
  affine, `:508-511`,`:540-541`); reject zero rows; `ShapeMismatch` on column
  count mismatch; expose `data_min`/`data_max` accessors matching sklearn
  `data_min_`/`data_max_` (REQ-1 Probe). Supports `f32` and `f64`.
- REQ-2: **Constant / zero-range column → `feature_range[0]`** — a column with
  `data_max == data_min` must map to `fr[0]` (0 by default), via
  `_handle_zeros_in_scale` replacing `data_range = 0` with 1 (`:88`,`:508-511`)
  so `transform(data_min) = fr[0]`. **The fixable divergence this iteration**:
  ferrolearn currently `continue`s (leaves the column unchanged), and
  `test_zero_range_column_unchanged` pins the wrong `5.0` result (REQ-2 Probe →
  `0.0`).
- REQ-3: `feature_range` validation — reject `fr[0] >= fr[1]` (sklearn
  "Minimum of desired feature range must be smaller than maximum", `:476-480`)
  with a `FerroError` (REQ-3 Probe: `ValueError`).
- REQ-4: NaN tolerance — `fit` with `force_all_finite="allow-nan"` (`:490-495`)
  and `_nanmin`/`_nanmax` (`:498-499`) so NaN entries are ignored when computing
  `data_min`/`data_max` and pass through `transform` unchanged (REQ-4 Probe:
  `[[0.0],[nan],[1.0]]`).
- REQ-5: Full fitted attribute surface — `scale_`, `min_`, `data_range_`,
  `n_samples_seen_` stored alongside `data_min_`/`data_max_` (`:508-514`)
  (REQ-5 Probe).
- REQ-6: `inverse_transform` (`:549-587`) — invert the affine map round-trip
  (REQ-6 Probe).
- REQ-7: PyO3 binding (`import ferrolearn` exposes a `MinMaxScaler` marshalling
  `fit`/`transform` over default-range `FittedMinMaxScaler<f64>`) — the project
  boundary CPython consumer.
- REQ-8: `partial_fit` (`:489-515`) — streaming min/max accumulation with
  `n_samples_seen_` updated across calls (`data_min = min(prev, batch)` etc.).
- REQ-9: `minmax_scale(X, feature_range=(0,1), *, axis=0, copy=True)` standalone
  free function including `axis=1` row-scaling (`:589`).
- REQ-10: `copy` and `clip` constructor parameters + `_parameter_constraints` —
  in-place-vs-copy semantics (`__init__` `copy=True`, `:455`) and post-transform
  clipping to `feature_range` (`clip=False`, `:542-543`).
- REQ-11: `get_feature_names_out` / `n_features_in_` /`feature_names_in_`
  (OneToOneFeatureMixin one-to-one passthrough; set on `fit`).
- REQ-12: ferray substrate — compute over `ferray-core` arrays / `ferray-ufunc`
  rather than `ndarray::Array2` + `num_traits::Float` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `MinMaxScaler::<f64>::new().fit(&X).transform(&X)` for
  `X=[[1,10],[2,20],[3,30]]` equals `[[0,0],[0.5,0.5],[1,1]]` and
  `with_feature_range(-1,1)` on `[[0],[5],[10]]` equals `[[-1],[0],[1]]` within
  ULP tolerance (REQ-1 Probe); `data_min()==[1,10]`, `data_max()==[3,30]`;
  `fit` on `(0,n)` returns `Err(InsufficientSamples)`; a column-count mismatch on
  `transform` returns `Err(ShapeMismatch)`. Pinned by an oracle-grounded `#[test]`.
- AC-2 (REQ-2): `MinMaxScaler::<f64>::new().fit_transform([[5,1],[5,2],[5,3]])`
  equals `[[0,0],[0,0.5],[0,1]]` (constant col 0 → `0.0`, REQ-2 Probe), NOT the
  current `5.0`; `test_zero_range_column_unchanged` is replaced by an
  oracle-grounded constant-column test asserting `fr[0]`.
- AC-3 (REQ-3): `with_feature_range(1.0,0.0)` and `with_feature_range(1.0,1.0)`
  return `Err(InvalidParameter)` (REQ-3 Probe). (Already holds.)
- AC-4 (REQ-4): `fit_transform` of `[[1],[NaN],[3]]` yields `[[0],[NaN],[1]]`
  (NaN ignored for min/max, passes through) (REQ-4 Probe).
- AC-5 (REQ-5): a fitted handle exposes `scale_==[0.5,0.05]`, `min_==[-0.5,-0.5]`,
  `data_range_==[2,20]`, `n_samples_seen_==3` for the REQ-5 Probe fit.
- AC-6 (REQ-6): `inverse_transform([[0],[0.5],[1]])` after fitting `[[1],[2],[3]]`
  yields `[[1],[2],[3]]` (REQ-6 Probe).
- AC-7 (REQ-7): `python3 -c "import ferrolearn; ..."` resolves the registered
  `_RsMinMaxScaler`; `.fit(X).transform(X)` matches the REQ-1 Probe (default
  range).
- AC-8 (REQ-8): two `partial_fit` batches accumulate the global min/max and
  `n_samples_seen_` equals the total row count.
- AC-9 (REQ-9): a free `minmax_scale(&X, fr, axis, copy)` with `axis=1`
  reproduces `[[0,1],[0,1],[0,1]]` (REQ-9 Probe).
- AC-10 (REQ-10): a `clip` flag clamps out-of-fit-range inputs to `feature_range`;
  a `copy=false` path is observably in-place.
- AC-11 (REQ-11): `get_feature_names_out` returns `['x0','x1']` for a 2-feature
  fit; `n_features_in_ == 2`.
- AC-12 (REQ-12): the owned transform computes on `ferray-core` arrays.

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (per-column min-max value match, non-constant) | SHIPPED | impl `pub fn fit in min_max_scaler.rs` (`Fit<Array2<F>,()> for MinMaxScaler<F>`) rejects 0 rows (`InsufficientSamples`) and sets `data_min[j]`/`data_max[j]` via per-column `reduce(min)`/`reduce(max)`; impl `Transform<Array2<F>> for FittedMinMaxScaler<F>::transform` computes `(*v - min) / span * range_width + range_min` with `range_width = range_max - range_min`, returning `ShapeMismatch` when `x.ncols() != n_features`. Mirrors sklearn affine `scale_ = (fr[1]-fr[0])/data_range` (`_data.py:508`), `min_ = fr[0]-data_min*scale_` (`:511`), `X *= scale_; X += min_` (`:540-541`); `data_min()`/`data_max()` accessors mirror `data_min_`/`data_max_`. Output equals REQ-1 Probe: default `[[0,0],[0.5,0.5],[1,1]]`, custom `(-1,1)` `[[-1],[0],[1]]`, `data_min_=[1,10]`/`data_max_=[3,30]`. Non-test consumers: (a) PyO3 `_RsMinMaxScaler` (`ferrolearn-python/src/extras.rs` line 1148, registered `ferrolearn-python/src/lib.rs` line 81) — `py_transformer!` marshals `fit`/`transform` over `FittedMinMaxScaler<f64>` from `MinMaxScaler::<f64>::new()`; (b) in-file `impl FittedPipelineTransformer<F> for FittedMinMaxScaler<F>::transform_pipeline` calls `self.transform(x)`; (c) crate re-export `pub use min_max_scaler::{FittedMinMaxScaler, MinMaxScaler};` (`ferrolearn-preprocess/src/lib.rs` line 118), the boundary public API (grandfathered S5/R-DEFER-1). Verification: `cargo test -p ferrolearn-preprocess` (`test_min_max_scaler_default_range`, `test_min_max_scaler_custom_range`, `test_fit_transform_equivalence`, `test_shape_mismatch_error`). |
| REQ-2 (constant / zero-range column → feature_range[0]) | SHIPPED | FIXED #1170. `Transform::transform` zero-span branch now sets every element of a constant column to `range_min` (= `feature_range.0`) instead of leaving it unchanged, matching sklearn's `_handle_zeros_in_scale` (data_range 0→1, `_data.py:88`,`:508-511`; `transform(data_min)=fr[0]`). Critic two-round CLEAN vs live oracle: 11 tests in `tests/divergence_min_max_scaler.rs` — constant col → `fr[0]` for default (0.0), `(-1,1)` (-1.0), `(2,5)` (2.0), mixed constant+scaling fixture, negative/zero constant, single-row fit. In-module `test_zero_range_column_unchanged` (pinned the wrong 5.0) replaced by `test_constant_column_maps_to_range_min` (R-HONEST-4). |
| REQ-3 (feature_range validation) | SHIPPED | impl `pub fn with_feature_range in min_max_scaler.rs` returns `Err(FerroError::InvalidParameter { name: "feature_range", .. })` when `range_min >= range_max`, mirroring sklearn's "Minimum of desired feature range must be smaller than maximum" (`_data.py:476-480`). REQ-3 Probe: `(1,0)`/`(1,1)` → error in both. Non-test consumer: the public ctor is exercised through the crate re-export boundary (`lib.rs` line 118) and `MinMaxScaler::<f64>::new()` hard-codes the valid `(0,1)` default consumed by the PyO3 binding. Verification: `cargo test -p ferrolearn-preprocess` (`test_invalid_feature_range`). Least-confident SHIPPED (validation-only ctor; no separate non-test caller of `with_feature_range` beyond tests, but the validated invariant guards every fitted instance, and `new()`'s `(0,1)` is the path the PyO3 + pipeline consumers take). |
| REQ-4 (NaN tolerance: allow-nan + nanmin/nanmax) | NOT-STARTED | open prereq blocker #1171. `fit` computes column min/max via `reduce(\|a,b\| if a<b {a} else {b})`; with IEEE NaN this comparison is `false`, so NaN poisons the running min/max differently from sklearn's `_nanmin`/`_nanmax` which ignore NaN (`:498-499`), and there is no `force_all_finite="allow-nan"` contract (`:490-495`). REQ-4 Probe: sklearn `[[1],[NaN],[3]]` → `data_min=1,data_max=3` → `[[0],[NaN],[1]]`; ferrolearn diverges. |
| REQ-5 (scale_/min_/data_range_/n_samples_seen_) | SHIPPED | `FittedMinMaxScaler<F>` now materializes `pub(crate) data_range_` (= `data_max − data_min`), `scale_` (= `(fr1−fr0)/_handle_zeros(data_range_)`, `_data.py:508`/`:88`), `min_` (= `fr0 − data_min·scale_`, `:511`), and `n_samples_seen_` (= n rows) in `Fit::fit`, with `#[must_use]` getters `scale()`/`min()`/`data_range()`/`n_samples_seen()` (`_data.py:508-514`). `transform`/`inverse_transform` unchanged. Verification (live sklearn 1.5.2, R-CHAR-3): `X=[[1,10],[2,20],[3,30],[5,50]]` default → `scale_=[0.25,0.025]`, `min_=[-0.25,-0.25]`, `data_range_=[4,40]`, `n_samples_seen_=4`; constant col1 + `feature_range=(-1,1)` → `data_range_=[2,0]`, `scale_=[1.0,2.0]` (zero-range col → `(1-(-1))/1=2.0`), `min_=[-2.0,-11.0]`. Tests `min_max_attrs_match_sklearn`, `min_max_scale_handles_zero_range_constant_col`. |
| REQ-6 (inverse_transform) | SHIPPED | `pub fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError>` on `FittedMinMaxScaler<F>` reverses the affine map per column: `x_orig = (x_scaled − fr0)·span/range_width + data_min[j]` where `span = data_max[j]−data_min[j]`, `range_width = fr1−fr0` — the branchless inverse of sklearn `X -= min_; X /= scale_` (`_data.py:549-587`, `scale_=(fr1-fr0)/_handle_zeros(data_range)` `:508`, `min_=fr0-data_min·scale_` `:511`); a constant column (`span==0`) inverts to `data_min[j]`, matching `_handle_zeros`. Shape-guards `n_features` (`ShapeMismatch`). Verification (live sklearn 1.5.2, R-CHAR-3): round-trip `inverse_transform(transform(X))==X` on `X=[[1,10],[2,20],[3,30],[5,50]]`; `feature_range=(-1,1)` `inverse_transform([[0,0],[1,1]])=[[3,30],[5,50]]`; constant col → `data_min`. Tests `min_max_inverse_roundtrip_matches_sklearn`, `min_max_inverse_custom_range_matches_sklearn`, `min_max_inverse_constant_col`, `min_max_inverse_shape_mismatch`. |
| REQ-7 (PyO3 binding) | SHIPPED | `_RsMinMaxScaler` registered in `ferrolearn-python/src/extras.rs` lines 1147-1153 via `py_transformer!(RsMinMaxScaler, "_RsMinMaxScaler", ferrolearn_preprocess::FittedMinMaxScaler<f64>, (), { ferrolearn_preprocess::MinMaxScaler::<f64>::new() })`, added in `ferrolearn-python/src/lib.rs` line 81 (`m.add_class::<extras::RsMinMaxScaler>()?`). The macro (`extras.rs:107-149`) generates a `#[pyclass]` whose `fit` marshals `PyReadonlyArray2 -> numpy2_to_ndarray -> MinMaxScaler::<f64>::new().fit(&x_nd, &())` and `transform` marshals `fitted.transform(&x_nd) -> ndarray2_to_numpy` — a real CPython fit/transform consumer of REQ-1's impl at default range. Thin marshalling layer; the numeric behavior it exposes is REQ-1's (and the REQ-2 divergence). Verification: `cargo build -p ferrolearn-python`; `python3 -c "import ferrolearn; ..."` round-trip against the REQ-1 Probe. |
| REQ-8 (partial_fit / streaming) | NOT-STARTED | open prereq blocker #1174. `Fit::fit` is single-shot; there is no `partial_fit` accumulating `data_min = min(prev, batch)` / `data_max = max(prev, batch)` and `n_samples_seen_ += batch_rows` across calls (sklearn `:489-515`). |
| REQ-9 (minmax_scale free fn + axis) | NOT-STARTED | open prereq blocker #1175. No standalone `minmax_scale` in `min_max_scaler.rs` or the crate; the estimator is hard-wired to per-column (`axis=0`) scaling with no `axis=1` row path (sklearn free fn `:589`). REQ-9 Probe (`axis=1` → `[[0,1],[0,1],[0,1]]`) unavailable. |
| REQ-10 (copy / clip params + _parameter_constraints) | NOT-STARTED | open prereq blocker #1176. `MinMaxScaler<F>` holds only `feature_range`; no `copy` field, no `clip` field, no post-transform clamp to `feature_range` (sklearn `__init__` `:455-459`, `clip` `:542-543`), no `_parameter_constraints` analog. `transform` always `to_owned()`s. |
| REQ-11 (get_feature_names_out / n_features_in_) | NOT-STARTED | open prereq blocker #1177. No `n_features_in_`, `feature_names_in_`, or `get_feature_names_out` (OneToOneFeatureMixin one-to-one passthrough set on `fit`). |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1178. Compute path uses `ndarray::Array2` + `num_traits::Float` (`columns_mut`, manual `reduce`), not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `min_max_scaler.rs` exposes the unfitted/fitted pair.
`MinMaxScaler<F> { pub(crate) feature_range: (F, F) }` is constructed by
`new()` (`(F::zero(), F::one())`), `with_feature_range(range_min, range_max)`
(returns `Err(InvalidParameter)` when `range_min >= range_max`), with
`feature_range()` accessor and `Default = new()`. `FittedMinMaxScaler<F> {
data_min: Array1<F>, data_max: Array1<F>, feature_range: (F,F) }` exposes
`data_min()`, `data_max()`, `feature_range()`. `impl Fit<Array2<F>, ()>` rejects
`n_samples == 0` (`InsufficientSamples`), then per column `j` folds
`data_min[j] = reduce(min)` and `data_max[j] = reduce(max)` over the column
iterator. `impl Transform<Array2<F>> for FittedMinMaxScaler<F>` returns
`ShapeMismatch` when `x.ncols() != n_features`, clones `x`, and for each column
computes `span = max - min`; **if `span == F::zero()` it `continue`s (leaves the
column unchanged)** — the REQ-2 divergence — otherwise sets
`*v = (*v - min) / span * range_width + range_min`. A second `impl Transform for
MinMaxScaler<F>` (unfitted) always errors (it exists to satisfy the
`FitTransform: Transform` supertrait); `impl FitTransform` chains `fit` then
`transform`; `impl PipelineTransformer<F>` (`fit_pipeline` boxes the fitted) and
`impl FittedPipelineTransformer<F>` (`transform_pipeline` calls `transform`)
provide pipeline integration. The generic bound `F: Float + Send + Sync +
'static` supports `f32`/`f64`. The crate re-exports both public types
(`ferrolearn-preprocess/src/lib.rs` line 118), and `ferrolearn-python`
registers `_RsMinMaxScaler` over `FittedMinMaxScaler<f64>` at the default range.

**sklearn (target contract).** `MinMaxScaler(OneToOneFeatureMixin,
TransformerMixin, BaseEstimator)` (`:291`) stores `feature_range`, `copy`, `clip`
(`__init__` `:455-459`) under `_parameter_constraints` (`:447-451`). `fit`
(`:461-487`) resets state and delegates to `partial_fit` (`:489-515`), which
`_validate_data(force_all_finite="allow-nan")` (`:490-495`), computes
`data_min = _nanmin(X, axis=0)` (`:498`), `data_max = _nanmax(X, axis=0)` (`:499`),
`data_range = data_max - data_min` (`:507`),
`scale_ = (fr[1]-fr[0]) / _handle_zeros_in_scale(data_range)` (`:508-510`),
`min_ = fr[0] - data_min*scale_` (`:511`), and stores
`data_min_/data_max_/data_range_/scale_/min_/n_samples_seen_` (`:511-514`).
`transform` (`:517-547`) re-validates (`reset=False`), then `X *= scale_`,
`X += min_`, and `if clip: clip(X, fr[0], fr[1])`. `_handle_zeros_in_scale`
(`:88`) replaces a `data_range` of 0 with 1 — this is precisely what makes a
constant column map to `fr[0]` (REQ-2). `inverse_transform` (`:549-587`) reverses
the affine; `minmax_scale` (`:589`) is the estimator-less free function with an
`axis` argument; `OneToOneFeatureMixin` provides `get_feature_names_out`.

**The structural gap.** ferrolearn's per-column affine is an exact match for
sklearn on **non-constant, finite** columns (REQ-1 Probe is bit-identical for the
default and custom range, and `data_min`/`data_max` match `data_min_`/`data_max_`),
and the PyO3 binding (REQ-7) plus `feature_range` validation (REQ-3) ship.
The one behavior that changes an observable result on well-formed input is the
**constant-column handling** (REQ-2): ferrolearn `continue`s where sklearn maps
to `fr[0]`. Because the divergence is a single `if span == F::zero()` branch and
the `_handle_zeros_in_scale` semantics are well-defined (`data_range 0 → 1`, so
the constant column lands on `fr[0]`), it is the minimal fixable divergence — the
critic pins it (flipping `test_zero_range_column_unchanged` to expect `fr[0]`),
the fixer changes the branch to set the column to `range_min`. The remaining gaps
are *contract surface*: NaN tolerance (REQ-4), the full fitted attribute set
(REQ-5), `inverse_transform` (REQ-6), `partial_fit` (REQ-8), the `minmax_scale`
free fn (REQ-9), `copy`/`clip` (REQ-10), feature-name plumbing (REQ-11), and the
ferray substrate (REQ-12).

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-3, REQ-7):

```bash
# Oracle (REQ-1 Probe) — non-constant per-column min-max, default + custom range:
python3 -c "from sklearn.preprocessing import MinMaxScaler; \
print(MinMaxScaler().fit_transform([[1.,10.],[2.,20.],[3.,30.]]).tolist()); \
print(MinMaxScaler(feature_range=(-1,1)).fit_transform([[0.],[5.],[10.]]).tolist())"
#   -> [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
#   -> [[-1.0], [0.0], [1.0]]
# ferrolearn equivalents: MinMaxScaler::<f64>::new() / with_feature_range(-1.0,1.0)
#   .fit(&X,&()).unwrap().transform(&X)

# REQ-2 divergence (must be pinned as a FAILING oracle test, then fixed):
python3 -c "from sklearn.preprocessing import MinMaxScaler; \
print(MinMaxScaler().fit_transform([[5.,1.],[5.,2.],[5.,3.]]).tolist())"
#   -> [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]  (ferrolearn currently returns 5.0 in col 0)

# Crate + binding gauntlet:
cargo test -p ferrolearn-preprocess   # incl. test_min_max_scaler_default_range,
                                      #       test_min_max_scaler_custom_range,
                                      #       test_invalid_feature_range,
                                      #       test_fit_transform_equivalence,
                                      #       test_shape_mismatch_error
#       (NOTE: test_zero_range_column_unchanged currently asserts the WRONG 5.0;
#        it must flip to expect 0.0 — REQ-2.)
cargo build -p ferrolearn-python      # registers _RsMinMaxScaler (REQ-7)
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s (`test_min_max_scaler_default_range`,
`test_min_max_scaler_custom_range`, `test_invalid_feature_range`,
`test_fit_transform_equivalence`, `test_shape_mismatch_error`) exercise REQ-1 and
REQ-3 but are **not oracle-grounded** (hand-written expected values). To satisfy
R-CHAR-3 the critic should add an oracle-pinned guard asserting the REQ-1 Probe
outputs and `data_min`/`data_max`. `test_zero_range_column_unchanged` is a
**characterization-of-a-bug**: it pins the divergent `5.0` and must flip to the
oracle `0.0` (REQ-2). No currently-green command establishes REQ-2, REQ-4..REQ-6,
or REQ-8..REQ-12.

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns
`#`-numbers, replacing the `#<REQn>` placeholders); reference them in the REQ
status table:

- #1170 — REQ-2: `transform` leaves a constant (`span == 0`) column
  unchanged; sklearn maps it to `feature_range[0]` (`_handle_zeros_in_scale`
  data_range 0→1, `_data.py:88`,`:508-511`). REQ-2 Probe: constant col → `0.0`,
  ferrolearn → original value. **The fixable divergence — pin + flip
  `test_zero_range_column_unchanged` + fix first.**
- #1171 — REQ-4: `fit` reduce-min/max poisons on NaN; no
  `force_all_finite="allow-nan"` / `_nanmin`/`_nanmax` (`:490-499`).
- #1172 — REQ-5: `FittedMinMaxScaler` stores only `data_min`/`data_max`/
  `feature_range`; no `scale_`/`min_`/`data_range_`/`n_samples_seen_` (`:508-514`).
- #1173 — REQ-6: no `inverse_transform` (`:549-587`).
- #1174 — REQ-8: no `partial_fit` / streaming `n_samples_seen_`
  accumulation (`:489-515`).
- #1175 — REQ-9: no standalone `minmax_scale` free fn; no `axis=1`
  row-scaling (`:589`).
- #1176 — REQ-10: no `copy`/`clip` params, no post-transform clamp, no
  `_parameter_constraints` (`:447-459`,`:542-543`).
- #1177 — REQ-11: no `n_features_in_`/`feature_names_in_`/
  `get_feature_names_out` (OneToOneFeatureMixin).
- #1178 — REQ-12: compute path on `ndarray`/`num_traits`, not ferray
  (R-SUBSTRATE-1/2).
