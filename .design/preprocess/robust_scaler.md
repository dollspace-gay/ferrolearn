# RobustScaler

<!--
tier: 3-component
status: draft
baseline-commit: 49b2a788
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_data.py  # class RobustScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (:1445); __init__(*, with_centering=True, with_scaling=True, quantile_range=(25.0,75.0), copy=True, unit_variance=False) (:1562-1575); _parameter_constraints {with_centering:["boolean"], with_scaling:["boolean"], quantile_range:[tuple], copy:["boolean"], unit_variance:["boolean"]} (:1554-1560). fit (:1578): _validate_data(force_all_finite="allow-nan", dtype=FLOAT_DTYPES, accept_sparse="csc") (:1597-1602) ALLOWS NaN; validates 0<=q_min<=q_max<=100 else ValueError (:1604-1606); if with_centering: center_=np.nanmedian(X,axis=0) (:1614) else None (:1616); if with_scaling: quantiles.append(np.nanpercentile(column_data, quantile_range)) per column (:1630), scale_=quantiles[1]-quantiles[0] (:1634), scale_=_handle_zeros_in_scale(scale_, copy=False) -> a scale of 0 becomes 1 (:1635); if unit_variance: scale_ /= norm.ppf(q_max/100)-norm.ppf(q_min/100) (:1636-1638) else scale_=None (:1640). transform (:1644): check_is_fitted; _validate_data(copy=self.copy, force_writeable, reset=False, force_all_finite="allow-nan") (:1658-1666); dense: if with_centering: X -= center_ (:1673); if with_scaling: X /= scale_ (:1675). inverse_transform (:1678): if with_scaling: X *= scale_ (:1706); if with_centering: X += center_ (:1708). _handle_zeros_in_scale (:88): a scale of 0 is replaced by 1. _more_tags allow_nan=True (:1711-1712). robust_scale(X, *, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0,75.0), copy=True, unit_variance=False) free fn (:1719).
ferrolearn-module: ferrolearn-preprocess/src/robust_scaler.rs
parity-ops: RobustScaler, robust_scale
crosslink-issue: 1247
-->

## Summary

scikit-learn's `RobustScaler` (`_data.py:1445`) centers and scales each **feature
(column)** independently using statistics robust to outliers: it removes the
median and divides by the interquartile range (IQR = Q75 − Q25). On `fit` it
stores `center_ = np.nanmedian(X, axis=0)` (`:1614`) and `scale_ = nanpercentile(q_max)
- nanpercentile(q_min)` (`:1630-1634`), then **replaces a zero scale with 1 via
`_handle_zeros_in_scale`** (`:1635`); on `transform` it applies `if with_centering:
X -= center_; if with_scaling: X /= scale_` (`:1672-1675`). It supports
`with_centering`/`with_scaling`/`quantile_range`/`copy`/`unit_variance`
constructor params, NaN tolerance (`force_all_finite="allow-nan"`, `nanmedian`/
`nanpercentile`), sparse CSC/CSR, an `inverse_transform`,
`get_feature_names_out`/`n_features_in_`, and a `robust_scale` free function with
an `axis` argument.

`ferrolearn-preprocess/src/robust_scaler.rs` ships a **dense, fixed-quantile
(25/75), always-center-and-scale** estimator with the unfitted/fitted split:
`RobustScaler<F>` (unit + `PhantomData`; `new()`/`Default`, **no params**) and
`FittedRobustScaler<F> { median: Array1<F>, iqr: Array1<F> }` (accessors
`median()`, `iqr()`). `impl Fit<Array2<F>,()>` rejects zero rows
(`InsufficientSamples`) and per column sorts the values, then computes
`med = quantile_sorted(col, 0.5)`, `q25 = quantile_sorted(col, 0.25)`,
`q75 = quantile_sorted(col, 0.75)` by linear interpolation `idx = q*(n-1)`,
storing `median[j] = med` and `iqr[j] = q75 - q25`. `impl Transform<Array2<F>>`
computes `(x - median)/scale_eff` per column (returns `ShapeMismatch` on a
column-count mismatch), where `scale_eff = if iqr==0 {1} else {iqr}` — a zero-IQR
column is **centered then divided by 1** (#1248 fix; constant column → 0). It also
provides an unfitted-`Transform` shim
(always errors), `FitTransform`, and `PipelineTransformer`/`FittedPipelineTransformer`
impls. Non-test consumers: the crate re-export `pub use robust_scaler::{FittedRobustScaler,
RobustScaler};` (`ferrolearn-preprocess/src/lib.rs` line 124), the in-file pipeline
impls, and a PyO3 binding `_RsRobustScaler` (`ferrolearn-python/src/extras.rs` line
1163, registered `ferrolearn-python/src/lib.rs` line 83) wrapping
`RobustScaler::<f64>::new()` fit/transform.

**Headline finding (FIXED this iteration, #1248):**
**zero-IQR / constant column handling now matches sklearn (DIV-1, REQ-2).** sklearn
maps a constant column to **0**: for `scale_ = 0`, `_handle_zeros_in_scale(scale_)`
sets `scale_ = 1` (`:88`,`:1635`), and because `with_centering=True` by default
`transform` does `X -= center_; X /= 1`, so each entry `(x - median)/1 = 0`.
ferrolearn previously LEFT the column UNCHANGED (`if iqr == F::zero() { continue }`)
— not even centered — but now computes `scale_eff = if iqr==0 {1} else {iqr}` and
always centers, so a constant column `7.0` → `[0,0,0]` and a non-constant zero-IQR
column `[1,1,1,1,9]` → `[0,0,0,0,8]` (Probes B/C), matching the live oracle. The
in-module `test_zero_iqr_column_unchanged` was rewritten to
`test_zero_iqr_column_centered_to_zero` (R-HONEST-4). This is the SAME
`_handle_zeros_in_scale` constant-column pattern fixed for StandardScaler
(#1190/#1191) and MinMaxScaler (#1169/#1170); it reflects through the PyO3
`_RsRobustScaler` binding automatically. The non-constant value-match (REQ-1) and
the PyO3 marshalling (REQ-11) also SHIP.

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — core value match: non-constant columns, default 25/75 IQR (Probe A).
python3 -c "import numpy as np; from sklearn.preprocessing import RobustScaler; \
X=np.array([[1.,10.],[2.,20.],[3.,30.],[100.,40.]]); \
m=RobustScaler().fit(X); \
print('center_', m.center_.tolist(), 'scale_', m.scale_.tolist()); \
print('ft', RobustScaler().fit_transform(X).tolist())"
# -> center_ [2.5, 25.0] scale_ [25.5, 15.0]
# -> ft [[-0.0588235.., -1.0], [-0.0196078.., -0.3333..], [0.0196078.., 0.3333..], [3.8235294.., 1.0]]
#    ferrolearn: RobustScaler::<f64>::new().fit(&X,&()).transform(&X) == same to ~1e-12;
#    median()==[2.5,25.0], iqr()==[25.5,15.0] (== sklearn center_/scale_ on these non-constant cols).

# REQ-1 — quantile linear-interpolation match, even-length column (Probe D):
python3 -c "import numpy as np; from sklearn.preprocessing import RobustScaler; \
m=RobustScaler().fit([[1.],[2.],[3.],[4.]]); \
print('center_', m.center_.tolist(), 'scale_', m.scale_.tolist()); \
print('pct', np.nanpercentile([1.,2.,3.,4.],[25,50,75]).tolist())"
# -> center_ [2.5] scale_ [1.5]
# -> pct [1.75, 2.5, 3.25]   (numpy default 'linear': q25=1.75, q75=3.25, IQR=1.5)
#    ferrolearn quantile_sorted: idx=0.25*(4-1)=0.75 -> 1+(2-1)*0.75=1.75; idx=0.75*3=2.25 ->
#    3+(4-3)*0.25=3.25; median idx=0.5*3=1.5 -> 2+(3-2)*0.5=2.5. EXACT match to numpy 'linear'.

# REQ-2 — zero-IQR / constant column DIVERGENCE (the fixable one, Probe B):
python3 -c "from sklearn.preprocessing import RobustScaler; \
print(RobustScaler().fit_transform([[7.,1.],[7.,2.],[7.,3.]]).tolist()); \
m=RobustScaler().fit([[7.,1.],[7.,2.],[7.,3.]]); \
print('center_', m.center_.tolist(), 'scale_', m.scale_.tolist())"
# -> [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]   (constant col 0 -> 0.0, NOT 7.0)
# -> center_ [7.0, 2.0] scale_ [1.0, 1.0]
#    ferrolearn: leaves constant col 0 at 7.0 -> [[7.0,-1.0],[7.0,0.0],[7.0,1.0]]. DIVERGENCE.
# sklearn internal: scale_=q75-q25=0 -> _handle_zeros_in_scale -> scale_=1 (:88,:1635);
#   transform: X -= center_=7 so x-7=0 (:1673), then X /= scale_=1 (:1675) -> 0.0.

# REQ-2 — NON-constant zero-IQR column diverges even more (Probe C):
python3 -c "from sklearn.preprocessing import RobustScaler; \
print(RobustScaler().fit_transform([[1.],[1.],[1.],[1.],[9.]]).tolist()); \
m=RobustScaler().fit([[1.],[1.],[1.],[1.],[9.]]); \
print('center_', m.center_.tolist(), 'scale_', m.scale_.tolist())"
# -> [[0.0], [0.0], [0.0], [0.0], [8.0]]   (q25=q75=1 -> scale_=1, center_=1 -> (x-1)/1)
# -> center_ [1.0] scale_ [1.0]
#    ferrolearn: iqr=0 -> continue -> UNCHANGED [[1],[1],[1],[1],[9]]. DIVERGENCE (9 stays 9, not 8).

# REQ-4 — quantile_range ctor param (non-default 10/90):
python3 -c "from sklearn.preprocessing import RobustScaler; \
m=RobustScaler(quantile_range=(10.0,90.0)).fit([[1.,10.],[2.,20.],[3.,30.],[100.,40.]]); \
print('scale_', m.scale_.tolist())"
# -> scale_ [62.7, 24.0]   (90th-10th percentile, not the IQR)
#    ferrolearn: quantile_sorted hard-codes 0.25/0.75; no quantile_range param.

# REQ-5 — with_centering / with_scaling ctor params:
python3 -c "from sklearn.preprocessing import RobustScaler; \
X=[[1.,10.],[2.,20.],[3.,30.],[100.,40.]]; \
print('no-center', RobustScaler(with_centering=False).fit_transform(X).tolist()); \
print('no-scale',  RobustScaler(with_scaling=False).fit_transform(X).tolist())"
# -> no-center [[X/scale_, ...]]  (X /= scale_, no median subtraction; center_=None)
# -> no-scale  [[X-center_, ...]] (X -= center_, no IQR division; scale_=None)
#    ferrolearn: RobustScaler has NO params; always centers AND scales.

# REQ-6 — unit_variance ctor param:
python3 -c "from scipy import stats; from sklearn.preprocessing import RobustScaler; \
m=RobustScaler(unit_variance=True).fit([[1.,10.],[2.,20.],[3.,30.],[100.,40.]]); \
print('scale_', m.scale_.tolist(), 'adjust', stats.norm.ppf(0.75)-stats.norm.ppf(0.25))"
# -> scale_ divided by adjust(=1.3489795..) so normal features get unit variance (:1636-1638)
#    ferrolearn: no unit_variance param; raw IQR only.

# REQ-7 — inverse_transform round-trip:
python3 -c "from sklearn.preprocessing import RobustScaler; \
X=[[1.,10.],[2.,20.],[3.,30.],[100.,40.]]; m=RobustScaler().fit(X); \
print(m.inverse_transform(m.transform(X)).tolist())"
# -> recovers X within ULPs (X *= scale_; X += center_, :1706-1708)
#    ferrolearn: FittedRobustScaler has NO inverse_transform method.

# REQ-8 — robust_scale free fn with axis:
python3 -c "from sklearn.preprocessing import robust_scale; \
print(robust_scale([[1.,10.],[2.,20.],[3.,30.],[100.,40.]], axis=1).tolist())"
# -> centers/scales each ROW (axis=1)
#    ferrolearn: no robust_scale free fn; estimator hard-wired axis=0 / per-column.

# REQ-9 — NaN allowance (force_all_finite="allow-nan" + nanmedian/nanpercentile):
python3 -c "import numpy as np; from sklearn.preprocessing import RobustScaler; \
print(RobustScaler().fit_transform(np.array([[1.],[np.nan],[3.],[5.]])).tolist())"
# -> NaN ignored for median/IQR; NaN passes through transform
#    ferrolearn: col.sort_by(partial_cmp) treats NaN as Equal (does NOT skip it); the median/
#    quantile indices then shift -> NaN poisons the statistics, diverging from nanmedian/nanpercentile.

# REQ-11 — PyO3 binding (_RsRobustScaler) is a real CPython fit/transform consumer:
#   ferrolearn-python/src/extras.rs:1163-1169 registers _RsRobustScaler over
#   FittedRobustScaler<f64> with build-block RobustScaler::<f64>::new() (default 25/75);
#   lib.rs:83 add_class. fit() marshals PyReadonlyArray2 -> Array2 -> .fit(&x,&());
#   transform() marshals fitted.transform(&x) -> PyArray2.
```

## Requirements

- REQ-1: Per-column robust scale value match (non-constant, non-zero-IQR columns)
  — learn per-column `median` (`np.nanmedian`, `:1614`) and `iqr = Q75 - Q25`
  (`np.nanpercentile(quantile_range)`, default 25/75, `:1630-1634`) on `fit` via
  numpy-default 'linear' quantile interpolation, apply `(x - median)/iqr` on
  `transform` (sklearn `X -= center_; X /= scale_`, `:1672-1675`); reject zero
  rows (`InsufficientSamples`); `ShapeMismatch` on column-count mismatch; expose
  `median()` matching sklearn `center_` and `iqr()` matching sklearn `scale_` on
  non-constant columns (Probe A, Probe D). Supports `f32` and `f64`.
- REQ-2: **Zero-IQR / constant column → 0** — a column with `iqr == 0` must map
  to `0.0`, because sklearn replaces `scale_` with 1 via `_handle_zeros_in_scale`
  (`:88`,`:1635`) and `transform` subtracts `center_` first: `(x - median)/1 = 0`
  (and `(x - median)` for a non-constant zero-IQR column). **The fixable
  divergence this iteration (DIV-1)**: ferrolearn `continue`s (leaves the column
  completely unchanged, not even centered), and `test_zero_iqr_column_unchanged`
  pins the wrong `7.0` result (Probe B → `0.0`; Probe C non-constant → `[0,0,0,0,8]`).
- REQ-3: Error contracts (scoped) — `InsufficientSamples` on zero rows in `fit`
  and `ShapeMismatch` on a column-count mismatch in `transform`.
- REQ-4: `quantile_range` constructor parameter (`(q_min, q_max)`, default
  `(25.0, 75.0)`, `:1567`) with `0 <= q_min <= q_max <= 100` validation
  (`ValueError`, `:1604-1606`) and per-column `nanpercentile(quantile_range)`
  (`:1630`) (Probe REQ-4: `(10,90)`).
- REQ-5: `with_centering` / `with_scaling` constructor parameters — conditional
  center (`if with_centering: X -= center_`, `:1672-1673`; else `center_=None`,
  `:1616`) and conditional scale (`if with_scaling: X /= scale_`, `:1674-1675`;
  else `scale_=None`, `:1640`), the `*`-only-keyword ctor (`:1562-1575`),
  `_parameter_constraints` (`:1554-1560`) (Probe REQ-5).
- REQ-6: `unit_variance` constructor parameter — when `True`, divide `scale_` by
  `norm.ppf(q_max/100) - norm.ppf(q_min/100)` so normally distributed features get
  unit variance (`:1636-1638`) (Probe REQ-6).
- REQ-7: `inverse_transform` (`:1678`) — apply `X *= scale_; X += center_` per
  column so `inverse_transform(transform(X)) == X` within tolerance;
  `ShapeMismatch` on column-count mismatch (Probe REQ-7).
- REQ-8: `robust_scale(X, *, axis=0, with_centering=True, with_scaling=True,
  quantile_range=(25.0,75.0), copy=True, unit_variance=False)` standalone free
  function including `axis=1` row-scaling (`:1719`).
- REQ-9: NaN tolerance — `fit` with `force_all_finite="allow-nan"` (`:1601`),
  `np.nanmedian` (`:1614`), and `np.nanpercentile` (`:1630`) so NaN entries are
  ignored when computing `center_`/`scale_` and pass through `transform`
  unchanged. **Do NOT reject non-finite input** — `RobustScaler` allows NaN
  (`_more_tags allow_nan=True`, `:1711-1712`) (Probe REQ-9).
- REQ-10: `center_` / `scale_` attribute-name + `_handle_zeros` semantic contract
  — expose fitted attributes under sklearn's names `center_` (= median) and
  `scale_` (= `_handle_zeros_in_scale(IQR)`, i.e. `1` on zero-IQR columns, NOT
  raw `iqr` which ferrolearn stores as `0`) (`:1505-1514`,`:1635`).
- REQ-11: PyO3 binding (`import ferrolearn` exposes a `RobustScaler` marshalling
  `fit`/`transform` over default-quantile `FittedRobustScaler<f64>`) — the project
  boundary CPython consumer.
- REQ-12: `copy` constructor parameter + in-place-vs-copy semantics (`copy=True`,
  `:1568`,`:1661`); `transform` currently always `to_owned()`s.
- REQ-13: Sparse CSC/CSR support — `with_centering=False`-only path
  (`inplace_column_scale`, `:1668-1670`,`:1701-1703`); raise on `with_centering=True`
  over sparse (`:1609-1612`).
- REQ-14: `get_feature_names_out` / `n_features_in_` / `feature_names_in_`
  (OneToOneFeatureMixin one-to-one passthrough; set on `fit`).
- REQ-15: ferray substrate — compute over `ferray-core` arrays / `ferray-ufunc`
  rather than `ndarray::Array2` + `num_traits::Float` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `RobustScaler::<f64>::new().fit(&X,&()).transform(&X)` for
  `X=[[1,10],[2,20],[3,30],[100,40]]` equals
  `[[-0.0588.., -1],[-0.0196.., -0.333..],[0.0196.., 0.333..],[3.8235.., 1]]`
  within ULP tolerance (Probe A); `median()==[2.5,25]`, `iqr()==[25.5,15]`; for
  `[[1],[2],[3],[4]]` `median()==[2.5]`, `iqr()==[1.5]` (quantile 'linear' match,
  Probe D); `fit` on `(0,n)` returns `Err(InsufficientSamples)`; a column-count
  mismatch on `transform` returns `Err(ShapeMismatch)`. Pinned by an
  oracle-grounded `#[test]`.
- AC-2 (REQ-2): `RobustScaler::<f64>::new().fit_transform([[7,1],[7,2],[7,3]])`
  has constant col 0 equal to `[0,0,0]` (Probe B), NOT the current `7.0`; the
  non-constant zero-IQR fit `[[1],[1],[1],[1],[9]]` transforms to `[[0],[0],[0],[0],[8]]`
  (Probe C); `test_zero_iqr_column_unchanged` is replaced by an oracle-grounded
  zero-IQR test asserting `0.0`.
- AC-3 (REQ-3): `fit` on `(0,n)` returns `Err(InsufficientSamples)`; a column-count
  mismatch on `transform` returns `Err(ShapeMismatch)` (pinned by
  `test_insufficient_samples_error`, `test_shape_mismatch_error`).
- AC-4 (REQ-4): `RobustScaler::new().with_quantile_range(10.0,90.0)` on the Probe A
  fixture yields `scale_==[62.7,24.0]`; `with_quantile_range(75.0,25.0)` or any
  out-of-`[0,100]` range returns `Err(InvalidParameter)` (Probe REQ-4).
- AC-5 (REQ-5): `RobustScaler::new().with_centering(false)` divides only and
  `with_scaling(false)` centers only, reproducing the Probe REQ-5 outputs; the
  ctor surface mirrors `__init__(*, with_centering, with_scaling, quantile_range,
  copy, unit_variance)`.
- AC-6 (REQ-6): `unit_variance=true` divides `scale_` by `norm.ppf(0.75)-norm.ppf(0.25)
  == 1.34897..` for the default range, reproducing the Probe REQ-6 `scale_`.
- AC-7 (REQ-7): `inverse_transform(transform(X))` for the Probe A fixture recovers
  `X` within `1e-10`; a column-count mismatch on `inverse_transform` returns
  `Err(ShapeMismatch)` (Probe REQ-7).
- AC-8 (REQ-8): a free `robust_scale(&X, axis, with_centering, with_scaling,
  quantile_range, copy, unit_variance)` with `axis=1` scales each row (Probe REQ-8).
- AC-9 (REQ-9): `fit_transform` of `[[1],[NaN],[3],[5]]` ignores NaN for
  median/IQR and passes NaN through (Probe REQ-9).
- AC-10 (REQ-10): a fitted handle exposes `center_` (= median) and `scale_`
  (= `_handle_zeros_in_scale(IQR)`, `1` on zero-IQR cols, NOT raw `iqr=0`).
- AC-11 (REQ-11): `python3 -c "import ferrolearn; ..."` resolves the registered
  `_RsRobustScaler`; `.fit(X).transform(X)` matches the Probe A output (default
  quantile range).
- AC-12 (REQ-12): a `copy=false` path is observably in-place.
- AC-13 (REQ-13): a CSC matrix with `with_centering=false` scales columns without
  densifying; `with_centering=true` over sparse raises.
- AC-14 (REQ-14): `get_feature_names_out` returns `['x0','x1']` for a 2-feature
  fit; `n_features_in_ == 2`.
- AC-15 (REQ-15): the owned transform computes on `ferray-core` arrays.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (per-column robust scale value match, non-constant) | SHIPPED | impl `pub fn fit in robust_scaler.rs` (`Fit<Array2<F>,()> for RobustScaler<F>`) rejects 0 rows (`InsufficientSamples`) and per column sorts the values then sets `median[j] = quantile_sorted(col, 0.5)` and `iqr[j] = quantile_sorted(col, 0.75) - quantile_sorted(col, 0.25)` via the `quantile_sorted` helper (`idx = q*(n-1)`, linear interpolation `sorted[lo] + (sorted[hi]-sorted[lo])*frac`) — EXACTLY numpy's default 'linear' `nanpercentile` (Probe D: q25=1.75, q75=3.25, median=2.5 for `[1,2,3,4]`); impl `Transform<Array2<F>> for FittedRobustScaler<F>::transform` computes `(*v - med) / iqr` per column, returning `ShapeMismatch` when `x.ncols() != n_features`. Mirrors sklearn `center_=np.nanmedian` (`_data.py:1614`), `scale_=nanpercentile(q_max)-nanpercentile(q_min)` (`:1630-1634`), `X -= center_; X /= scale_` (`:1672-1675`); `median()` mirrors `center_`, `iqr()` mirrors `scale_` on non-constant columns. Output equals Probe A: `center_=[2.5,25]`, `scale_=[25.5,15]`, `ft=[[-0.0588..,-1],[-0.0196..,-0.333..],[0.0196..,0.333..],[3.8235..,1]]`. Non-test consumers: (a) PyO3 `_RsRobustScaler` (`ferrolearn-python/src/extras.rs` line 1163, registered `ferrolearn-python/src/lib.rs` line 83) — `py_transformer!` macro `fit` marshals `numpy2_to_ndarray -> RobustScaler::<f64>::new().fit(&x_nd,&())`, `transform` marshals `fitted.transform(&x_nd) -> ndarray2_to_numpy`; (b) in-file `impl FittedPipelineTransformer<F> for FittedRobustScaler<F>::transform_pipeline` calls `self.transform(x)`; (c) crate re-export `pub use robust_scaler::{FittedRobustScaler, RobustScaler};` (`ferrolearn-preprocess/src/lib.rs` line 124), the boundary public API (grandfathered S5/R-DEFER-1). Verification: `cargo test -p ferrolearn-preprocess` (`test_robust_scaler_basic`, `test_outlier_robustness`, `test_fit_transform_equivalence`). |
| REQ-2 (zero-IQR / constant column → 0) | SHIPPED (closed #1248) | `Transform::transform` now computes `let scale_eff = if iqr == F::zero() { F::one() } else { iqr };` and ALWAYS `*v = (*v - med) / scale_eff` (the `continue` early-skip removed), mirroring sklearn `_handle_zeros_in_scale` (scale 0→1, `_data.py:88`,`:1635`) + center-first (`X -= center_; X /= scale_`, `:1673-1675`). Live oracle (R-CHAR-3): constant col `fit_transform([[7,1],[7,2],[7,3]])` → col 0 `[0,0,0]`; non-constant zero-IQR `fit_transform([[1],[1],[1],[1],[9]])` → `[[0],[0],[0],[0],[8]]`. Guards `divergence_constant_column_centered_to_zero` + `divergence_nonconstant_zero_iqr_centered` PASS; in-module `test_zero_iqr_column_unchanged` rewritten to `test_zero_iqr_column_centered_to_zero` (R-HONEST-4). Reflects through the `_RsRobustScaler` PyO3 binding automatically. Same pattern as StandardScaler #1191 / MinMaxScaler #1170. Two-round critic-verified CLEAN (mixed constant/zero-IQR/normal matrix, f32 path). |
| REQ-3 (InsufficientSamples / ShapeMismatch error contracts) | SHIPPED | `Fit::fit` returns `Err(FerroError::InsufficientSamples { required: 1, actual: 0, context: "RobustScaler::fit" })` when `n_samples == 0`; `Transform::transform` returns `Err(FerroError::ShapeMismatch { .. context: "FittedRobustScaler::transform" })` when `x.ncols() != n_features` (mirrors sklearn `_validate_data(reset=False)` feature-count check, `:1658-1666`). Non-test consumer: the error path guards every fitted instance reached through the crate re-export (`lib.rs` line 124) and the PyO3 binding, which maps `FerroError` → `PyValueError` (`extras.rs` macro line 132/144). Verification: `cargo test -p ferrolearn-preprocess` (`test_insufficient_samples_error`, `test_shape_mismatch_error`). Scoped: ferrolearn's `InsufficientSamples` on 0 rows is a ferrolearn-side guard (sklearn raises on its own validation); the `ShapeMismatch` mirrors sklearn's feature-count check. |
| REQ-4 (quantile_range ctor param) | SHIPPED | `RobustScaler<F>` gains `pub quantile_range: (F, F)` (default `(25.0, 75.0)`) + `with_quantile_range(self, q_min, q_max) -> Result<Self, FerroError>` validating NON-STRICT `0 <= q_min <= q_max <= 100` (mirroring sklearn `if not 0 <= q_min <= q_max <= 100`, `:1604-1606`; `(50,50)` is ACCEPTED → zero IQR → `_handle_zeros` → scale 1; only `q_min > q_max` / out-of-range raise). `Fit::fit` now calls `quantile_sorted` with `quantile_range.0/100` and `.1/100` (`scale_ = Q(q_max) − Q(q_min)`); the median (Q50 = `center_`) is unchanged. Default `(25,75)` byte-identical. Verification (live sklearn 1.5.2, R-CHAR-3, col0 `[1..10]`, col1 `[100..1000]`): default `center_=[5.5,550]`/`scale_=[4.5,450]`; `(10,90)` → `center_=[5.5,550]`, `scale_=[7.2,720]`, `transform([[1,100]])=[[-0.625,-0.625]]`; `(50,50)` accepted → `scale_=[1,1]`. Tests `robust_quantile_range_default_unchanged`, `robust_quantile_range_10_90_matches_sklearn`, `robust_quantile_range_validation_rejects`, `robust_quantile_range_equal_bounds_accepted`. |
| REQ-5 (with_centering / with_scaling ctor params) | SHIPPED | `RobustScaler<F>` gains `pub with_centering`/`with_scaling: bool` (default `true`) + builders `with_with_centering`/`with_with_scaling` (`_data.py:1672-1675`). The flags thread into `FittedRobustScaler` and `Transform::transform` applies them conditionally: `if with_centering { x -= median } if with_scaling { x /= scale_eff }` (`:1616`/`:1640`). Default `(true,true)` byte-identical (regression-guarded via `to_bits`). R-DEV-4: ferrolearn ALWAYS materializes `median`/`iqr` (`center_`/`scale_`) — sklearn sets them to `None` when the flag is `False`; the flags govern transform APPLICATION so transform OUTPUT matches sklearn exactly (the `None`-attr representation is the documented deviation; getters stay `&Array1`). Verification (live sklearn 1.5.2, R-CHAR-3, col0 `[1,3,5,7,9]`, col1 `[100..900]`, `center_=[5,500]`/`scale_=[4,400]`): `(T,T)` `transform(X[:2])=[[-1,-1],[-0.5,-0.5]]`; `(F,T)` `[[0.25,0.25],[0.75,0.75]]` (scale only); `(T,F)` `[[-4,-400],[-2,-200]]` (center only); `(F,F)` identity `[[1,100],[3,300]]`. Tests `robust_with_centering_scaling_default_matches_sklearn`, `robust_with_scaling_false`, `robust_with_centering_false`, `robust_both_false_identity`. |
| REQ-6 (unit_variance ctor param) | NOT-STARTED | open prereq blocker #1251. No `unit_variance` field; `fit` stores raw `iqr` and never divides by `norm.ppf(q_max/100) - norm.ppf(q_min/100)` (sklearn `:1636-1638`), which also requires the normal-distribution inverse-CDF (a `ferray::stats`/`statrs` capability). Probe REQ-6 unavailable. |
| REQ-7 (inverse_transform) | NOT-STARTED | open prereq blocker #1252. `FittedRobustScaler<F>` exposes only `transform` (plus accessors); there is no `inverse_transform` method applying `X *= scale_; X += center_` (sklearn `:1678`,`:1706-1708`). Probe REQ-7 round-trip unavailable. |
| REQ-8 (robust_scale free fn + axis) | SHIPPED | `pub fn robust_scale<F>(x: &Array2<F>, axis: usize, with_centering: bool, with_scaling: bool, quantile_range: (F,F)) -> Result<Array2<F>, FerroError>` — the functional form, by clean delegation to the estimator (`RobustScaler::new().with_with_centering(..).with_with_scaling(..).with_quantile_range(..)?` then fit+transform). `axis=0` (default) column-wise = native; `axis=1` transposes → axis-0 → transposes back (sklearn `_data.py:1845-1848` `if axis==0: s.fit_transform(X) else: s.fit_transform(X.T).T`); `axis∉{0,1}` → `InvalidParameter`. (`unit_variance` #1251 and `copy` are separate REQs, not params here.) Verification (live sklearn 1.5.2, R-CHAR-3, col0 `[1,3,5,7,9]`, col1 `[100..900]`): `axis=0` `[[-1,-1],[-0.5,-0.5],[0,0],[0.5,0.5],[1,1]]`; `with_centering=false` `[[0.25,0.25]..[2.25,2.25]]`; `axis=1` on `[[1,2,3,4,5]]` → `[[-1,-0.5,0,0.5,1]]`; `(10,90)` `[[-0.625..],..]`. Tests `robust_scale_axis0_default_matches_sklearn`, `robust_scale_no_centering_matches_sklearn`, `robust_scale_axis1_matches_sklearn`, `robust_scale_quantile_range_matches_sklearn`, `robust_scale_invalid_axis_errors`. |
| REQ-9 (NaN tolerance: allow-nan + nanmedian/nanpercentile) | NOT-STARTED | open prereq blocker #1254. `fit` does `col.sort_by(\|a,b\| a.partial_cmp(b).unwrap_or(Ordering::Equal))` then `quantile_sorted` over the full column — an IEEE NaN sorts as `Equal` (NOT skipped), shifting the quantile indices and poisoning `median`/`iqr`, diverging from sklearn's `force_all_finite="allow-nan"` (`:1601`) + `np.nanmedian` (`:1614`) / `np.nanpercentile` (`:1630`) which IGNORE NaN. (NOT a rejection bug — RobustScaler ALLOWS NaN, `_more_tags allow_nan=True`, `:1711-1712`.) Probe REQ-9 divergence. |
| REQ-10 (center_ / scale_ attribute names + _handle_zeros semantics) | NOT-STARTED | open prereq blocker #1255. ferrolearn exposes `median()`/`iqr()`, semantically `center_`/`scale_`, but NOT under sklearn's attribute names; and `iqr()` returns raw `Q75-Q25` (= `0` on a zero-IQR column), whereas sklearn's `scale_` is `_handle_zeros_in_scale(IQR)` (= `1` on a zero-IQR column, `:1635`). The attribute-name + `_handle_zeros` contract (`:1505-1514`) is not met. |
| REQ-11 (PyO3 binding) | SHIPPED | `_RsRobustScaler` registered in `ferrolearn-python/src/extras.rs` lines 1163-1169 via `py_transformer!(RsRobustScaler, "_RsRobustScaler", ferrolearn_preprocess::FittedRobustScaler<f64>, (), { ferrolearn_preprocess::RobustScaler::<f64>::new() })`, added in `ferrolearn-python/src/lib.rs` line 83 (`m.add_class::<extras::RsRobustScaler>()?`). The macro (`extras.rs:107-149`) generates a `#[pyclass]` whose `fit` marshals `PyReadonlyArray2 -> numpy2_to_ndarray -> RobustScaler::<f64>::new().fit(&x_nd, &())` and `transform` marshals `fitted.transform(&x_nd) -> ndarray2_to_numpy` — a real CPython fit/transform consumer of REQ-1's impl at the default 25/75 range. Thin marshalling layer; the numeric behavior it exposes is REQ-1's (and the REQ-2 divergence will reflect through it). Verification: `cargo build -p ferrolearn-python`; `python3 -c "import ferrolearn; ..."` round-trip against the Probe A output. |
| REQ-12 (copy param) | NOT-STARTED | open prereq blocker #1256. `RobustScaler<F>` has no `copy` field; `transform` always `to_owned()`s (`let mut out = x.to_owned()`), so there is no in-place-vs-copy choice (sklearn `__init__` `copy=True` `:1568`, `_validate_data(copy=self.copy)` `:1661`). |
| REQ-13 (sparse CSC/CSR) | NOT-STARTED | open prereq blocker #1257. Dense-only (`Array2<F>`); no sparse `with_centering=False` column-scaling path (`inplace_column_scale`, `:1668-1670`,`:1701-1703`) and no raise on `with_centering=True` over sparse (`:1609-1612`). |
| REQ-14 (get_feature_names_out / n_features_in_) | NOT-STARTED | open prereq blocker #1258. No `n_features_in_`, `feature_names_in_`, or `get_feature_names_out` (OneToOneFeatureMixin one-to-one passthrough set on `fit`). |
| REQ-15 (ferray substrate) | NOT-STARTED | open prereq blocker #1259. Compute path uses `ndarray::Array2`/`Array1` + `num_traits::Float` (`column(j)`, `Vec`+`sort_by`, `columns_mut`), not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `robust_scaler.rs` exposes the unfitted/fitted pair.
`RobustScaler<F> { _marker: PhantomData<F> }` is a parameterless unit struct
constructed by `new()` (= `Default`). `FittedRobustScaler<F> { median: Array1<F>,
iqr: Array1<F> }` exposes `median()` and `iqr()`. The free helper
`quantile_sorted(sorted, q)` computes the `q`-th quantile of a sorted slice by
linear interpolation (`idx = q*(n-1)`, `sorted[lo] + (sorted[hi]-sorted[lo])*frac`)
— numpy's default 'linear' method. `impl Fit<Array2<F>, ()>` rejects
`n_samples == 0` (`InsufficientSamples`), then per column `j` collects the column
into a `Vec`, `sort_by(partial_cmp -> Ordering::Equal)`, and sets
`median[j] = quantile_sorted(col, 0.5)`, `iqr[j] = quantile_sorted(col, 0.75) -
quantile_sorted(col, 0.25)`. `impl Transform<Array2<F>> for FittedRobustScaler<F>`
returns `ShapeMismatch` when `x.ncols() != n_features`, clones `x`, and per
column: **if `iqr == F::zero()` it `continue`s (leaves the column COMPLETELY
UNCHANGED — not even centered)** — the REQ-2 divergence — otherwise sets
`*v = (*v - med) / iqr`. A second `impl Transform for RobustScaler<F>` (unfitted)
always errors (it exists to satisfy the `FitTransform: Transform` supertrait);
`impl FitTransform` chains `fit` then `transform`; `impl PipelineTransformer<F>`
(`fit_pipeline` boxes the fitted) and `impl FittedPipelineTransformer<F>`
(`transform_pipeline` calls `transform`) provide pipeline integration. The generic
bound `F: Float + Send + Sync + 'static` supports `f32`/`f64`. The crate re-exports
both public types (`ferrolearn-preprocess/src/lib.rs` line 124), and
`ferrolearn-python` registers `_RsRobustScaler` over `FittedRobustScaler<f64>` at
the default quantile range.

**sklearn (target contract).** `RobustScaler(OneToOneFeatureMixin,
TransformerMixin, BaseEstimator)` (`:1445`) stores `with_centering`,
`with_scaling`, `quantile_range`, `copy`, `unit_variance` (`__init__`
`:1562-1575`) under `_parameter_constraints` (`:1554-1560`). `fit` (`:1578`)
`_validate_data(accept_sparse="csc", dtype=FLOAT_DTYPES,
force_all_finite="allow-nan")` (`:1597-1602`), validates
`0 <= q_min <= q_max <= 100` else `ValueError` (`:1604-1606`), sets
`center_ = np.nanmedian(X, axis=0)` if `with_centering` (`:1614`, else `None`,
`:1616`), and if `with_scaling` builds per-column
`np.nanpercentile(column_data, quantile_range)` (`:1630`), computes
`scale_ = quantiles[1] - quantiles[0]` (`:1634`), then
`scale_ = _handle_zeros_in_scale(scale_, copy=False)` — **replacing a scale of 0
with 1** (`:88`,`:1635`); if `unit_variance` it divides `scale_` by
`norm.ppf(q_max/100) - norm.ppf(q_min/100)` (`:1636-1638`) (else `scale_=None`,
`:1640`). `transform` (`:1644`) re-validates (`reset=False`, `allow-nan`,
`copy=self.copy`) then, dense, `if with_centering: X -= center_` (`:1673`);
`if with_scaling: X /= scale_` (`:1675`). `inverse_transform` (`:1678`) reverses:
`if with_scaling: X *= scale_` (`:1706`); `if with_centering: X += center_`
(`:1708`). `_handle_zeros_in_scale` (`:88`) is precisely what makes a constant
column map to `0` under default `with_centering=True` (REQ-2). `robust_scale`
(`:1719`) is the estimator-less free function with an `axis` argument;
`OneToOneFeatureMixin` provides `get_feature_names_out`; `_more_tags` advertises
`allow_nan=True` (`:1711-1712`).

**The structural gap.** ferrolearn's per-column robust scale is an exact match for
sklearn on **non-constant, non-zero-IQR, finite, default-param** columns (Probe A
and Probe D are value-identical to ~1e-12; the `quantile_sorted` linear
interpolation matches numpy 'linear' `nanpercentile` exactly; `median()`/`iqr()`
match `center_`/`scale_`), and the PyO3 binding (REQ-11) plus the error contracts
(REQ-3) ship. The one behavior that changes an observable result on well-formed
default-param input was the **zero-IQR-column handling** (REQ-2), now FIXED
(#1248): ferrolearn previously `continue`d (left the column unchanged) where
sklearn maps to `0` (constant column) or `(x - median)` (non-constant zero-IQR);
the fix replaces the branch with `scale_eff = if iqr==0 {1} else {iqr}` and always
centers (`(x-median)/scale_eff`), matching the `_handle_zeros_in_scale` semantics
(`scale_` of a zero-IQR feature → `1`, `with_centering` centers first). The
remaining gaps are *contract surface*: `quantile_range` (REQ-4),
`with_centering`/`with_scaling` (REQ-5), `unit_variance` (REQ-6),
`inverse_transform` (REQ-7), the `robust_scale` free fn (REQ-8), NaN tolerance
(REQ-9), the `center_`/`scale_` attribute names + `_handle_zeros` `scale_` (REQ-10),
`copy` (REQ-12), sparse support (REQ-13), feature-name plumbing (REQ-14), and the
ferray substrate (REQ-15).

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-3, REQ-11):

```bash
# Oracle (Probe A) — non-constant per-column robust scale, default 25/75 IQR:
python3 -c "import numpy as np; from sklearn.preprocessing import RobustScaler; \
X=np.array([[1.,10.],[2.,20.],[3.,30.],[100.,40.]]); m=RobustScaler().fit(X); \
print(m.center_.tolist(), m.scale_.tolist()); \
print(RobustScaler().fit_transform(X).tolist())"
#   -> [2.5, 25.0] [25.5, 15.0]
#   -> [[-0.0588.., -1.0], [-0.0196.., -0.333..], [0.0196.., 0.333..], [3.8235.., 1.0]]
# ferrolearn equivalent: RobustScaler::<f64>::new().fit(&X,&()).unwrap().transform(&X)

# Oracle (Probe D) — quantile linear-interpolation match (even-length column):
python3 -c "import numpy as np; from sklearn.preprocessing import RobustScaler; \
m=RobustScaler().fit([[1.],[2.],[3.],[4.]]); print(m.center_.tolist(), m.scale_.tolist()); \
print(np.nanpercentile([1.,2.,3.,4.],[25,50,75]).tolist())"
#   -> [2.5] [1.5]   /   [1.75, 2.5, 3.25]
# ferrolearn quantile_sorted reproduces 1.75/2.5/3.25 (idx=q*(n-1) linear interpolation).

# REQ-2 divergence (must be pinned as a FAILING oracle test, then fixed):
python3 -c "from sklearn.preprocessing import RobustScaler; \
print(RobustScaler().fit_transform([[7.,1.],[7.,2.],[7.,3.]]).tolist()); \
print(RobustScaler().fit_transform([[1.],[1.],[1.],[1.],[9.]]).tolist())"
#   -> [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]   (constant col 0 -> 0.0; ferrolearn returns 7.0)
#   -> [[0.0], [0.0], [0.0], [0.0], [8.0]]      (zero-IQR col -> x-median; ferrolearn unchanged)

# Crate + binding gauntlet:
cargo test -p ferrolearn-preprocess   # incl. test_robust_scaler_basic, test_outlier_robustness,
                                      #       test_fit_transform_equivalence,
                                      #       test_shape_mismatch_error,
                                      #       test_insufficient_samples_error
#       (NOTE: test_zero_iqr_column_unchanged currently asserts the WRONG 7.0;
#        it must flip to expect 0.0 — REQ-2.)
cargo build -p ferrolearn-python      # registers _RsRobustScaler (REQ-11)
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s (`test_robust_scaler_basic`, `test_outlier_robustness`,
`test_fit_transform_equivalence`, `test_shape_mismatch_error`,
`test_insufficient_samples_error`) exercise REQ-1 and REQ-3 but are **not
oracle-grounded** (they assert statistical properties / hand-written values rather
than the live sklearn output). To satisfy R-CHAR-3 the critic should add an
oracle-pinned guard asserting the Probe A / Probe D outputs and `median()`/`iqr()`
against sklearn `center_`/`scale_`. `test_zero_iqr_column_unchanged` is a
**characterization-of-a-bug**: it pins the divergent `7.0` and must flip to the
oracle `0.0` (REQ-2). No currently-green command establishes REQ-2 or
REQ-4..REQ-10, REQ-12..REQ-15.

## Blockers

REQ-2 (DIV-1) was FIXED this iteration; the remaining NOT-STARTED REQs are open
`-l blocker` issues referenced by the REQ status table:

- #1248 — REQ-2 (CLOSED, fixed): `transform` left a zero-IQR column unchanged;
  now `scale_eff = if iqr==0 {1} else {iqr}` and always centers, mirroring
  sklearn `_handle_zeros_in_scale` scale→1 + center-first
  (`_data.py:88`,`:1635`,`:1672-1675`). Constant col → `0.0`; `[1,1,1,1,9]` →
  `[0,0,0,0,8]`. `test_zero_iqr_column_unchanged` rewritten to
  `test_zero_iqr_column_centered_to_zero`. Same pattern as StandardScaler
  #1191 / MinMaxScaler #1170.
- #1249 — REQ-4: `quantile_sorted` hard-codes `0.25`/`0.75`; no `quantile_range`
  ctor param, no `0<=q_min<=q_max<=100` validation (`:1567`,`:1604-1606`,`:1630`).
- #NNN-C — REQ-5: no `with_centering`/`with_scaling` params, no conditional
  center/scale, no `center_=None`/`scale_=None` paths, no `*`-only ctor
  (`:1562-1575`,`:1672-1675`).
- #NNN-D — REQ-6: no `unit_variance` param; no division by
  `norm.ppf(q_max/100)-norm.ppf(q_min/100)` (needs a normal inverse-CDF from
  `ferray::stats`/`statrs`) (`:1636-1638`).
- #NNN-E — REQ-7: no `inverse_transform` (`X *= scale_; X += center_`, `:1678`,
  `:1706-1708`).
- #NNN-F — REQ-8: no standalone `robust_scale` free fn; no `axis=1` row path
  (`:1719`).
- #NNN-G — REQ-9: `fit` sort+`partial_cmp` treats NaN as `Equal` (does not skip);
  no `force_all_finite="allow-nan"` / `nanmedian`/`nanpercentile` (`:1601`,`:1614`,
  `:1630`, `_more_tags allow_nan`, `:1711-1712`).
- #NNN-H — REQ-10: exposes `median()`/`iqr()`, not sklearn `center_`/`scale_`;
  `iqr()` is raw `Q75-Q25` (`0` on zero-IQR cols), not `_handle_zeros_in_scale(IQR)`
  (`1`) (`:1505-1514`,`:1635`).
- #NNN-I — REQ-12: no `copy` param; `transform` always `to_owned()`s (`:1568`,
  `:1661`).
- #NNN-J — REQ-13: dense-only; no sparse `with_centering=False` column scaling /
  raise on `with_centering=True` over sparse (`:1609-1612`,`:1668-1670`,`:1701-1703`).
- #NNN-K — REQ-14: no `n_features_in_`/`feature_names_in_`/`get_feature_names_out`
  (OneToOneFeatureMixin).
- #NNN-L — REQ-15: compute path on `ndarray`/`num_traits`, not ferray
  (R-SUBSTRATE-1/2).
