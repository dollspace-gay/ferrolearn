# StandardScaler

<!--
tier: 3-component
status: draft
baseline-commit: 0106191e
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_data.py  # class StandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (:696); __init__(*, copy=True, with_mean=True, with_std=True) (:835-838); _parameter_constraints {copy:["boolean"], with_mean:["boolean"], with_std:["boolean"]} (:829-833). fit (:853-878) calls _reset() then partial_fit; partial_fit (:880-1025): _validate_data(force_all_finite="allow-nan", dtype=FLOAT_DTYPES, accept_sparse=("csr","csc")) (:914-920) ALLOWS NaN, dense first-pass mean_=0/var_=0 (:986-991), mean_/var_/n_samples_seen_ = _incremental_mean_and_var(X, ..., sample_weight) (:999-1005) -> POPULATION variance (ddof=0); constant_mask=_is_constant_feature(var_, mean_, n_samples_seen_) (:1016-1018); scale_=_handle_zeros_in_scale(sqrt(var_), copy=False, constant_mask) (:1019-1021) -> a var_=0 column has scale_=1. transform (:1027-1068): if with_mean: X -= mean_ (:1064-1065); if with_std: X /= scale_ (:1066-1067). inverse_transform (:1070-1110): if with_std: X *= scale_ (:1106-1107); if with_mean: X += mean_ (:1108-1109). _handle_zeros_in_scale (:88-120): a scale (sqrt(var)) detected near-constant -> 1.0. _is_constant_feature (:72-85). scale(X, *, axis=0, with_mean=True, with_std=True, copy=True) free fn (:133). _more_tags allow_nan=True (:1112-1113).
ferrolearn-module: ferrolearn-preprocess/src/standard_scaler.rs
parity-ops: StandardScaler, scale
crosslink-issue: 1190
-->

## Summary

scikit-learn's `StandardScaler` (`_data.py:696`) standardizes each **feature
(column)** independently to zero mean and unit variance: `z = (x - u) / s` where
`u` is the per-column mean and `s` the per-column standard deviation (biased /
population, `ddof=0`, computed via `_incremental_mean_and_var`). On `fit` it
stores `mean_`, `var_`, `scale_ = _handle_zeros_in_scale(sqrt(var_))`, and
`n_samples_seen_`; on `transform` it applies `if with_mean: X -= mean_; if
with_std: X /= scale_` (`:1064-1067`). It supports `copy`/`with_mean`/`with_std`
constructor params, NaN tolerance (`force_all_finite="allow-nan"`, NaN ignored in
`fit`, passed through in `transform`), `sample_weight`, sparse CSR/CSC,
`partial_fit` (streaming), `inverse_transform`, `get_feature_names_out` /
`n_features_in_`, and a `scale` free function with an `axis` argument.

`ferrolearn-preprocess/src/standard_scaler.rs` ships a **dense, non-streaming,
always-center-and-scale** estimator with the unfitted/fitted split:
`StandardScaler<F>` (unit + `PhantomData`; `new()`/`Default`, **no params** — it
always centers and scales) and `FittedStandardScaler<F> { mean: Array1<F>, std:
Array1<F> }` (accessors `mean()`, `std()`; `inverse_transform`). `impl
Fit<Array2<F>,()>` rejects zero rows (`InsufficientSamples`) and computes per
column `mean = Σx/n` and `var = Σ(x-mean)²/n` (**population variance, ddof=0**),
`std = sqrt(var)`. `impl Transform<Array2<F>>` computes `(x - mean)/std` per
column (returns `ShapeMismatch` on a column-count mismatch), **leaving
zero-std columns unchanged** (`if s == F::zero() { continue }`). It also provides
an unfitted-`Transform` shim (always errors), `FitTransform`, and
`PipelineTransformer`/`FittedPipelineTransformer` impls. Non-test consumers: the
crate re-export `pub use standard_scaler::{FittedStandardScaler, StandardScaler};`
(`ferrolearn-preprocess/src/lib.rs` line 125), the in-file pipeline impls, and a
PyO3 binding `_RsStandardScaler` (`ferrolearn-python/src/transformers.rs` line 12,
registered in `ferrolearn-python/src/lib.rs` line 22) wrapping
`StandardScaler::<f64>::new()` fit / transform / inverse_transform / `mean_`.

**Headline finding (document prominently — fixable this iteration):**
**constant / zero-variance column handling DIVERGES.** sklearn maps a constant
column to **0**, NOT to its original value: for `var_ = 0`,
`_handle_zeros_in_scale(sqrt(0))` sets `scale_ = 1` (`:88-120`,`:1019-1021`), and
because `with_mean=True` by default `transform` does `X -= mean_; X /= 1` so each
entry `(x - mean)/1 = 0` (since `x == mean` for a constant column). ferrolearn
instead LEAVES the column UNCHANGED (`if s == F::zero() { continue }`, `transform`
in `standard_scaler.rs`). Live oracle: constant column `5.0` → `[0.0, 0.0, 0.0]`;
ferrolearn returns the original `5.0` — a DIVERGENCE (REQ-2, the fixable one).
The in-module `test_zero_variance_column_unchanged` asserts the WRONG (`5.0`)
behavior and must flip to expect `0.0` (R-HONEST-4). This is the SAME pattern as
the just-fixed MinMaxScaler constant-column divergence. The non-constant
value-match (REQ-1), `inverse_transform` round-trip (REQ-3), and the PyO3
marshalling (REQ-4) SHIP.

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — core value match: non-constant columns (deterministic, population std).
python3 -c "from sklearn.preprocessing import StandardScaler; \
print(StandardScaler().fit_transform([[1.,10.],[2.,20.],[3.,30.]]).tolist()); \
m=StandardScaler().fit([[1.,10.],[2.,20.],[3.,30.]]); \
print('mean_', m.mean_.tolist(), 'var_', m.var_.tolist(), 'scale_', m.scale_.tolist())"
# -> [[-1.224744871391589, -1.224744871391589], [0.0, 0.0], [1.224744871391589, 1.224744871391589]]
# -> mean_ [2.0, 20.0] var_ [0.6666666666666666, 66.66666666666667] scale_ [0.816496580927726, 8.16496580927726]
#    ferrolearn: StandardScaler::<f64>::new().fit(&X,&()).transform(&X) == same to ~1e-12.
#    NOTE: ferrolearn exposes std()=sqrt(var) (== sklearn scale_ for NON-constant cols);
#          sklearn additionally exposes var_ (=std^2) and scale_=_handle_zeros(sqrt(var))
#          which DIVERGES from raw sqrt(var) ONLY on constant columns (REQ-2/REQ-5).

# REQ-2 — constant / zero-variance column DIVERGENCE (the fixable one):
python3 -c "from sklearn.preprocessing import StandardScaler; \
print(StandardScaler().fit_transform([[1.,5.],[2.,5.],[3.,5.]]).tolist()); \
m=StandardScaler().fit([[1.,5.],[2.,5.],[3.,5.]]); \
print('var_', m.var_.tolist(), 'scale_', m.scale_.tolist(), 'mean_', m.mean_.tolist())"
# -> [[-1.224744871391589, 0.0], [0.0, 0.0], [1.224744871391589, 0.0]]   (constant col 1 -> 0.0, NOT 5.0)
# -> var_ [0.666.., 0.0] scale_ [0.816.., 1.0] mean_ [2.0, 5.0]
#    ferrolearn: leaves constant col 1 at 5.0 -> [[..,5.0],[..,5.0],[..,5.0]]. DIVERGENCE.
# sklearn internal: var_=0 -> _is_constant_feature true (:72-85) -> _handle_zeros_in_scale -> scale_=1 (:88-120);
#   transform: X -= mean_ (:1064-1065) so x-5=0, then X /= scale_=1 (:1066-1067) -> 0.0.

# REQ-3 — inverse_transform round-trip (and constant-col inverse after the fix):
python3 -c "from sklearn.preprocessing import StandardScaler; \
X=[[1.,2.],[3.,4.],[5.,6.]]; m=StandardScaler().fit(X); \
print('rt', m.inverse_transform(m.transform(X)).tolist()); \
mc=StandardScaler().fit([[1.,5.],[2.,5.],[3.,5.]]); t=mc.transform([[1.,5.],[2.,5.],[3.,5.]]); \
print('const fwd', t.tolist(), 'const inv', mc.inverse_transform(t).tolist())"
# -> rt [[1.0000000000000002, 2.0], [3.0, 4.0], [5.0, 6.0]]   (recovers X within ULPs)
# -> const fwd [[..,0.0],[..,0.0],[..,0.0]]  const inv [[..,5.0],[..,5.0],[..,5.0]]
#    Constant-col inverse: sklearn does X *= scale_(=1); X += mean_(=5) -> recovers 5.
#    ferrolearn inverse uses RAW std (=0): x*0+5 = 5 -> ALSO recovers 5 for the scaled value 0.
#    So inverse round-trip is correct EITHER way for the post-fix forward output (0 -> 5);
#    flag for the fixer: align inverse to scale_eff=1 (handle_zeros) for cross-consistency,
#    but the round-trip itself is not a divergence.

# REQ-4 — PyO3 binding (_RsStandardScaler) is a real CPython fit/transform/inverse consumer:
#   ferrolearn-python/src/transformers.rs:12-... registers _RsStandardScaler over
#   FittedStandardScaler<f64> with build StandardScaler::<f64>::new();
#   lib.rs:22 add_class. fit() marshals PyReadonlyArray2 -> Array2 -> .fit(&x,&());
#   transform()/inverse_transform() marshal fitted.{transform,inverse_transform}(&x) -> PyArray2;
#   #[getter] mean_ exposes fitted.mean().

# REQ-5 — var_ / scale_ / n_samples_seen_ fitted attributes + _handle_zeros scale_ semantics:
python3 -c "from sklearn.preprocessing import StandardScaler; \
m=StandardScaler().fit([[1.,5.],[2.,5.],[3.,5.]]); \
print('var_', m.var_.tolist(), 'scale_', m.scale_.tolist(), 'n_samples_seen_', m.n_samples_seen_)"
# -> var_ [0.666.., 0.0] scale_ [0.816.., 1.0] n_samples_seen_ 3
#    ferrolearn FittedStandardScaler stores only mean + std (=sqrt(var)); no var_, no scale_
#    (no _handle_zeros so std stays 0 on constant cols), no n_samples_seen_.

# REQ-6 — with_mean / with_std constructor params:
python3 -c "from sklearn.preprocessing import StandardScaler; \
X=[[1.,10.],[2.,20.],[3.,30.]]; \
print('no-mean', StandardScaler(with_mean=False).fit_transform(X).tolist()); \
print('no-std',  StandardScaler(with_std=False).fit_transform(X).tolist())"
# -> no-mean [[1.224.., ...divide-only...]]  (X /= scale_, no centering)
# -> no-std  [[-1.0, -10.0], [0.0, 0.0], [1.0, 10.0]]  (X -= mean_, no scaling)
#    ferrolearn: StandardScaler has NO params; always centers AND scales.

# REQ-7 — NaN tolerance (force_all_finite="allow-nan" + _incremental_mean_and_var):
python3 -c "import numpy as np; from sklearn.preprocessing import StandardScaler; \
print(StandardScaler().fit_transform(np.array([[1.],[np.nan],[3.]])).tolist())"
# -> [[-1.0], [nan], [1.0]]   (NaN ignored in mean/var -> mean=2,var=1; NaN passes through transform)
#    ferrolearn: the Σ/Σ(x-mean)² fold PROPAGATES NaN -> mean/var become NaN, whole column poisoned.
#    DO NOT pin a non-finite REJECTION: StandardScaler ALLOWS NaN (:918, _more_tags allow_nan (:1112-1113)).

# REQ-8 — scale() free fn with axis:
python3 -c "from sklearn.preprocessing import scale; \
print(scale([[1.,10.],[2.,20.],[3.,30.]], axis=1).tolist())"
# -> standardizes each ROW (axis=1)
#    ferrolearn: no scale() free fn; estimator hard-wired axis=0 / per-column.
```

## Requirements

- REQ-1: Per-column standardize value match (non-constant columns) — learn
  per-column `mean` and population (`ddof=0`) `std = sqrt(Σ(x-mean)²/n)` on `fit`,
  apply `(x - mean)/std` on `transform` (sklearn `mean_`/`scale_`,
  `X -= mean_; X /= scale_`, `:1064-1067`); reject zero rows
  (`InsufficientSamples`); `ShapeMismatch` on column-count mismatch; expose
  `mean()` matching sklearn `mean_` and `std()` matching sklearn `scale_` on
  non-constant columns (REQ-1 Probe). Supports `f32` and `f64`.
- REQ-2: **Constant / zero-variance column → 0** — a column with `var == 0`
  (so `x == mean` everywhere) must map to `0.0`, because sklearn replaces
  `scale_` with 1 via `_handle_zeros_in_scale` (`:88-120`,`:1019-1021`) and
  `transform` subtracts `mean_` first: `(x - mean)/1 = 0`. **The fixable
  divergence this iteration**: ferrolearn `continue`s (leaves the column
  unchanged), and `test_zero_variance_column_unchanged` pins the wrong `5.0`
  result (REQ-2 Probe → `0.0`).
- REQ-3: `inverse_transform` round-trip (`:1070-1110`) — apply `x*std + mean`
  per column so `inverse_transform(transform(X)) == X` within tolerance;
  `ShapeMismatch` on column-count mismatch (REQ-3 Probe).
- REQ-4: PyO3 binding (`import ferrolearn` exposes a `StandardScaler` marshalling
  `fit`/`transform`/`inverse_transform`/`mean_` over `FittedStandardScaler<f64>`)
  — the project boundary CPython consumer.
- REQ-5: Full fitted attribute surface — `var_` (= std²), `scale_`
  (= `_handle_zeros_in_scale(sqrt(var_))`, i.e. `1` on constant columns, NOT raw
  `std`), and `n_samples_seen_` stored alongside `mean_` (`:1013-1023`)
  (REQ-5 Probe).
- REQ-6: `with_mean` / `with_std` constructor parameters + `copy` +
  `_parameter_constraints` — conditional center (`if with_mean: X -= mean_`,
  `:1064`) and conditional scale (`if with_std: X /= scale_`, `:1066`), the
  `*`-only-keyword ctor `__init__(*, copy=True, with_mean=True, with_std=True)`
  (`:835-838`), in-place-vs-copy semantics (REQ-6 Probe).
- REQ-7: NaN tolerance — `fit` with `force_all_finite="allow-nan"` (`:918`) and
  `_incremental_mean_and_var` so NaN entries are ignored when computing
  `mean_`/`var_` and pass through `transform` unchanged (REQ-7 Probe:
  `[[1],[NaN],[3]]` → `[[-1],[NaN],[1]]`). **Do NOT reject non-finite input** —
  `StandardScaler` allows NaN (`_more_tags allow_nan=True`, `:1112-1113`).
- REQ-8: `scale(X, *, axis=0, with_mean=True, with_std=True, copy=True)`
  standalone free function including `axis=1` row-standardization (`:133`).
- REQ-9: `partial_fit` (`:880-1025`) — streaming incremental mean/variance via
  the Chan-Golub-LeVeque update with `n_samples_seen_` accumulated across calls.
- REQ-10: `sample_weight` support — weighted mean/variance in `fit`/`partial_fit`
  (`_check_sample_weight`, `_incremental_mean_and_var(..., sample_weight)`,
  `:923-924`,`:999-1005`).
- REQ-11: Sparse CSR/CSC support — `with_mean=False`-only path that scales
  without densifying (`mean_variance_axis`, `inplace_column_scale`,
  `:940-983`,`:1055-1062`); raise on `with_mean=True` over sparse (`:941-945`).
- REQ-12: `get_feature_names_out` / `n_features_in_` / `feature_names_in_`
  (OneToOneFeatureMixin one-to-one passthrough; set on `fit`).
- REQ-13: ferray substrate — compute over `ferray-core` arrays / `ferray-ufunc`
  rather than `ndarray::Array2` + `num_traits::Float` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `StandardScaler::<f64>::new().fit(&X,&()).transform(&X)` for
  `X=[[1,10],[2,20],[3,30]]` equals
  `[[-1.2247.., -1.2247..],[0,0],[1.2247.., 1.2247..]]` within ULP tolerance
  (REQ-1 Probe); `mean()==[2,20]`, `std()==[0.8165.., 8.1650..]` (== sklearn
  `scale_` on these non-constant cols); `fit` on `(0,n)` returns
  `Err(InsufficientSamples)`; a column-count mismatch on `transform` returns
  `Err(ShapeMismatch)`. Pinned by an oracle-grounded `#[test]`.
- AC-2 (REQ-2): `StandardScaler::<f64>::new().fit_transform([[1,5],[2,5],[3,5]])`
  has constant col 1 equal to `[0,0,0]` (REQ-2 Probe), NOT the current `5.0`;
  `test_zero_variance_column_unchanged` is replaced by an oracle-grounded
  constant-column test asserting `0.0`.
- AC-3 (REQ-3): `inverse_transform(transform(X))` for `X=[[1,2],[3,4],[5,6]]`
  recovers `X` within `1e-10` (REQ-3 Probe); a column-count mismatch on
  `inverse_transform` returns `Err(ShapeMismatch)`.
- AC-4 (REQ-4): `python3 -c "import ferrolearn; ..."` resolves the registered
  `_RsStandardScaler`; `.fit(X).transform(X)` matches the REQ-1 Probe and
  `.inverse_transform(...)` / `.mean_` round-trip.
- AC-5 (REQ-5): a fitted handle exposes `var_==[0.666..,0.0]`,
  `scale_==[0.816..,1.0]` (constant col `scale_=1`, NOT raw `std=0`),
  `n_samples_seen_==3` for the REQ-5 Probe fit.
- AC-6 (REQ-6): `StandardScaler::new().with_mean(false)` divides only and
  `with_std(false)` centers only, reproducing the REQ-6 Probe outputs; the ctor
  surface mirrors `__init__(*, copy, with_mean, with_std)`.
- AC-7 (REQ-7): `fit_transform` of `[[1],[NaN],[3]]` yields `[[-1],[NaN],[1]]`
  (NaN ignored for mean/var, passes through) (REQ-7 Probe).
- AC-8 (REQ-8): a free `scale(&X, axis, with_mean, with_std, copy)` with `axis=1`
  standardizes each row (REQ-8 Probe).
- AC-9 (REQ-9): two `partial_fit` batches accumulate the global mean/variance and
  `n_samples_seen_` equals the total row count (matches a single-shot `fit` on the
  concatenation within tolerance).
- AC-10 (REQ-10): a `sample_weight` vector reproduces sklearn's weighted
  `mean_`/`var_`.
- AC-11 (REQ-11): a CSC matrix with `with_mean=false` scales columns without
  densifying; `with_mean=true` over sparse raises.
- AC-12 (REQ-12): `get_feature_names_out` returns `['x0','x1']` for a 2-feature
  fit; `n_features_in_ == 2`.
- AC-13 (REQ-13): the owned transform computes on `ferray-core` arrays.

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (per-column standardize value match, non-constant) | SHIPPED | impl `pub fn fit in standard_scaler.rs` (`Fit<Array2<F>,()> for StandardScaler<F>`) rejects 0 rows (`InsufficientSamples`) and per column sets `mean[j] = Σx/n` and `std[j] = sqrt(Σ(x-mean)²/n)` (POPULATION variance, ddof=0); impl `Transform<Array2<F>> for FittedStandardScaler<F>::transform` computes `(*v - m) / s` per column, returning `ShapeMismatch` when `x.ncols() != n_features`. Mirrors sklearn `mean_`/`scale_` and `X -= mean_; X /= scale_` (`_data.py:1064-1067`); `mean()` mirrors `mean_`, `std()` mirrors `scale_` on non-constant columns. Output equals REQ-1 Probe: `[[-1.2247..,-1.2247..],[0,0],[1.2247..,1.2247..]]`; `mean_=[2,20]`, `scale_=[0.8165..,8.1650..]`. Non-test consumers: (a) PyO3 `_RsStandardScaler` (`ferrolearn-python/src/transformers.rs` line 12, registered `ferrolearn-python/src/lib.rs` line 22) — `fit` marshals `PyReadonlyArray2 -> numpy2_to_ndarray -> StandardScaler::<f64>::new().fit(&x_nd,&())`, `transform` marshals `fitted.transform(&x_nd) -> ndarray2_to_numpy`; (b) in-file `impl FittedPipelineTransformer<F> for FittedStandardScaler<F>::transform_pipeline` calls `self.transform(x)`; (c) crate re-export `pub use standard_scaler::{FittedStandardScaler, StandardScaler};` (`ferrolearn-preprocess/src/lib.rs` line 125), the boundary public API (grandfathered S5/R-DEFER-1). Verification: `cargo test -p ferrolearn-preprocess` (`test_standard_scaler_zero_mean_unit_variance`, `test_fit_transform_equivalence`, `test_shape_mismatch_error`, `test_insufficient_samples_error`, `test_f32_scaler`). |
| REQ-2 (constant / zero-variance column → 0) | SHIPPED | FIXED #1191. `Transform::transform` zero-variance branch now uses `s_eff = if s==0 {1} else {s}` and always applies `(x - mean)/s_eff`, so a constant column maps to `(x-mean)/1 = 0` matching sklearn `_handle_zeros_in_scale` (`_data.py:88`,`:1019-1021`,`:1064-1067`). `inverse_transform` aligned to `x*s_eff + m` (matching sklearn `X *= scale_`). Critic two-round CLEAN vs live oracle: 9 tests in `tests/divergence_standard_scaler.rs` — constant col → 0 (default, single-row-fit), mixed constant+non-constant fixture, AND inverse non-round-trip on a constant column matches sklearn `x*1+mean` (discriminates the alignment). In-module `test_zero_variance_column_unchanged` replaced by `test_constant_column_maps_to_zero` (R-HONEST-4). |
| REQ-3 (inverse_transform round-trip) | SHIPPED | impl `pub fn inverse_transform in standard_scaler.rs` (`FittedStandardScaler<F>`) applies `*v = *v * s + m` per column, returning `ShapeMismatch` when `x.ncols() != n_features`, mirroring sklearn `X *= scale_; X += mean_` (`_data.py:1106-1109`). REQ-3 Probe: `inverse_transform(transform(X))` recovers `X=[[1,2],[3,4],[5,6]]` within ULPs. Non-test consumer: PyO3 `_RsStandardScaler::inverse_transform` (`ferrolearn-python/src/transformers.rs` line 50) marshals `fitted.inverse_transform(&x_nd)`. Verification: `cargo test -p ferrolearn-preprocess` (`test_inverse_transform_roundtrip`). Caveat (not a divergence): on a constant column ferrolearn's inverse uses raw `std=0` while sklearn uses `scale_=1`; for the scaled value `0` both yield `mean`, so the round-trip is correct either way (flag for the REQ-2 fixer to align inverse to `scale_eff=1` for cross-consistency). |
| REQ-4 (PyO3 binding) | SHIPPED | `_RsStandardScaler` declared `#[pyclass(name = "_RsStandardScaler")]` in `ferrolearn-python/src/transformers.rs` line 12 over `Option<FittedStandardScaler<f64>>`, registered `m.add_class::<transformers::RsStandardScaler>()?` in `ferrolearn-python/src/lib.rs` line 22. `#[pymethods]`: `fit` marshals `numpy2_to_ndarray` → `StandardScaler::<f64>::new().fit(&x_nd,&())`; `transform` and `inverse_transform` marshal `fitted.{transform,inverse_transform}(&x_nd) -> ndarray2_to_numpy`; `#[getter] mean_` exposes `fitted.mean()`. A real CPython fit/transform/inverse consumer of REQ-1/REQ-3's impl. Thin marshalling layer; the numeric behavior it exposes is REQ-1's (and the REQ-2 divergence). Verification: `cargo build -p ferrolearn-python`; `python3 -c "import ferrolearn; ..."` round-trip against the REQ-1 Probe. |
| REQ-5 (var_ / scale_ / n_samples_seen_ + _handle_zeros) | SHIPPED | `FittedStandardScaler<F>` now materializes `pub(crate) var_` (= population variance, ddof=0, stored from the same `var` that `std = sqrt(var)` came from — no double rounding), `scale_` (= `std.mapv(|s| if s==0 {1} else {s})` = `_handle_zeros_in_scale(sqrt(var_))`, `_data.py:88`/`:1019-1021`), and `n_samples_seen_` (= n rows) in `Fit::fit`, with `#[must_use]` getters `var()`/`scale()`/`n_samples_seen()` (`_data.py:1013-1023`). `transform`/`inverse_transform` unchanged (already used the equivalent `s_eff`). Verification (live sklearn 1.5.2, R-CHAR-3, `X=[[1,5],[2,5],[3,5]]` col1 constant): `var_=[0.6666666667,0.0]`, `scale_=[0.816496580927726,1.0]` (scale_[1]==1.0 EXACTLY while std()[1]==0.0 — the `_handle_zeros` distinction), `n_samples_seen_=3`. Tests `standard_scaler_var_scale_nsamples_match_sklearn`, `standard_scaler_scale_differs_from_std_on_constant_col`, `standard_scaler_var_equals_std_squared_nonconstant`. Boundary consumer: `FittedStandardScaler` public API (grandfathered S5). |
| REQ-6 (with_mean / with_std / copy params) | SHIPPED | `StandardScaler<F>` gains `pub with_mean`/`with_std`/`copy: bool` (all default `true`) + builders `with_with_mean`/`with_with_std`/`with_copy` (`_data.py:835-838`). The two flags thread into `FittedStandardScaler` and `Transform::transform` applies them conditionally: `if with_mean { x -= mean } if with_std { x /= scale_ }` (`:1064-1067`); `inverse_transform` mirrors (`if with_std { x *= scale_ } if with_mean { x += mean }`). `copy` is ABI-only (fit never mutates X). Default `(true,true)` is byte-identical to the prior fit (regression-guarded via `to_bits`). R-DEV-4: ferrolearn ALWAYS materializes `mean_`/`var_`/`scale_` (Rust `&Array1` getters cannot be `None`), whereas sklearn sets them to `None` when the flag is `False` — the flags govern transform APPLICATION so transform OUTPUT matches sklearn exactly; the `None`-attr representation is the documented deviation. Verification (live sklearn 1.5.2, R-CHAR-3, `X=[[1,10],[2,20],[3,30]]`): `(T,T)` `[[-1.2247..],[0],[1.2247..]]`; `(F,T)` `[[1.2247..],[2.4495..],[3.6742..]]` (scale only); `(T,F)` `[[-1,-10],[0,0],[1,10]]` (center only); `(F,F)` identity. Tests `standard_scaler_with_mean_std_default_matches_sklearn`, `standard_scaler_with_std_false`, `standard_scaler_with_mean_false`, `standard_scaler_both_false_identity`, `standard_scaler_inverse_roundtrip_each_config`. |
| REQ-7 (NaN tolerance: allow-nan) | NOT-STARTED | open prereq blocker #1194. `fit` computes `mean = col.fold(+)/n` and `var = col.map(\|v\| (v-m)²).fold(+)/n`; an IEEE NaN propagates through both folds, poisoning the whole column's `mean`/`std`, diverging from sklearn's `force_all_finite="allow-nan"` (`:918`) + `_incremental_mean_and_var` which IGNORE NaN. REQ-7 Probe: sklearn `[[1],[NaN],[3]]` → `mean=2,var=1` → `[[-1],[NaN],[1]]`; ferrolearn → NaN column. (NOT a rejection bug — StandardScaler ALLOWS NaN, `_more_tags allow_nan`, `:1112-1113`.) |
| REQ-8 (scale free fn + axis) | NOT-STARTED | open prereq blocker #1195. No standalone `scale` in `standard_scaler.rs` or the crate; the estimator is hard-wired to per-column (`axis=0`) standardization with no `axis=1` row path (sklearn free fn `:133`). REQ-8 Probe (`axis=1` row-standardize) unavailable. |
| REQ-9 (partial_fit / streaming) | NOT-STARTED | open prereq blocker #1196. `Fit::fit` is single-shot two-pass; there is no `partial_fit` performing the Chan-Golub-LeVeque incremental mean/variance update with `n_samples_seen_` accumulated across calls (sklearn `:880-1025`, `_incremental_mean_and_var`). |
| REQ-10 (sample_weight) | NOT-STARTED | open prereq blocker #1197. `fit(&self, x, _y: &())` takes no weights; mean/variance are unweighted. No `_check_sample_weight` analog, no weighted `_incremental_mean_and_var` path (`:923-924`,`:999-1005`). |
| REQ-11 (sparse CSR/CSC) | NOT-STARTED | open prereq blocker #1198. Dense-only (`Array2<F>`); no sparse `with_mean=False` scaling path (`mean_variance_axis`/`inplace_column_scale`, `:940-983`,`:1055-1062`) and no raise on `with_mean=True` over sparse (`:941-945`). |
| REQ-12 (get_feature_names_out / n_features_in_) | NOT-STARTED | open prereq blocker #1199. No `n_features_in_`, `feature_names_in_`, or `get_feature_names_out` (OneToOneFeatureMixin one-to-one passthrough set on `fit`). |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1200. Compute path uses `ndarray::Array2`/`Array1` + `num_traits::Float` (`column(j)`, manual fold, `columns_mut`), not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `standard_scaler.rs` exposes the unfitted/fitted pair.
`StandardScaler<F> { _marker: PhantomData<F> }` is a parameterless unit struct
constructed by `new()` (= `Default`). `FittedStandardScaler<F> { mean: Array1<F>,
std: Array1<F> }` exposes `mean()`, `std()`, and `inverse_transform`. `impl
Fit<Array2<F>, ()>` rejects `n_samples == 0` (`InsufficientSamples`), then per
column `j` computes `m = Σx/n` and `var = Σ(x-m)²/n` (population, ddof=0),
storing `mean[j]=m`, `std[j]=sqrt(var)`. `impl Transform<Array2<F>> for
FittedStandardScaler<F>` returns `ShapeMismatch` when `x.ncols() != n_features`,
clones `x`, and per column: **if `s == F::zero()` it `continue`s (leaves the
column unchanged)** — the REQ-2 divergence — otherwise sets `*v = (*v - m) / s`.
`inverse_transform` applies `*v = *v * s + m` (same `ShapeMismatch` guard). A
second `impl Transform for StandardScaler<F>` (unfitted) always errors (it exists
to satisfy the `FitTransform: Transform` supertrait); `impl FitTransform` chains
`fit` then `transform`; `impl PipelineTransformer<F>` (`fit_pipeline` boxes the
fitted) and `impl FittedPipelineTransformer<F>` (`transform_pipeline` calls
`transform`) provide pipeline integration. The generic bound `F: Float + Send +
Sync + 'static` supports `f32`/`f64`. The crate re-exports both public types
(`ferrolearn-preprocess/src/lib.rs` line 125), and `ferrolearn-python` registers
`_RsStandardScaler` over `FittedStandardScaler<f64>`.

**sklearn (target contract).** `StandardScaler(OneToOneFeatureMixin,
TransformerMixin, BaseEstimator)` (`:696`) stores `copy`, `with_mean`, `with_std`
(`__init__` `:835-838`) under `_parameter_constraints` (`:829-833`). `fit`
(`:853-878`) `_reset()`s then delegates to `partial_fit` (`:880-1025`), which
`_validate_data(force_all_finite="allow-nan", dtype=FLOAT_DTYPES,
accept_sparse=("csr","csc"))` (`:914-920`), and for dense input computes
`mean_, var_, n_samples_seen_ = _incremental_mean_and_var(X, ..., sample_weight)`
(`:999-1005`) — **population variance, ddof=0**. It then derives
`constant_mask = _is_constant_feature(var_, mean_, n_samples_seen_)` (`:1016-1018`)
and `scale_ = _handle_zeros_in_scale(sqrt(var_), copy=False, constant_mask)`
(`:1019-1021`) — replacing the scale of a (near-)constant feature with `1`.
`transform` (`:1027-1068`) re-validates (`reset=False`, `allow-nan`) then
`if with_mean: X -= mean_` (`:1064-1065`); `if with_std: X /= scale_`
(`:1066-1067`). `inverse_transform` (`:1070-1110`) reverses: `if with_std: X *=
scale_`; `if with_mean: X += mean_`. `_handle_zeros_in_scale` (`:88-120`) +
`_is_constant_feature` (`:72-85`) are precisely what make a constant column map to
`0` under default `with_mean=True` (REQ-2). `scale` (`:133`) is the estimator-less
free function with an `axis` argument; `OneToOneFeatureMixin` provides
`get_feature_names_out`; `_more_tags` advertises `allow_nan=True` (`:1112-1113`).

**The structural gap.** ferrolearn's per-column standardize is an exact match for
sklearn on **non-constant, finite, default-param** columns (REQ-1 Probe is
value-identical to ~1e-12; `mean()`/`std()` match `mean_`/`scale_`), and the
`inverse_transform` round-trip (REQ-3) plus the PyO3 binding (REQ-4) ship. The
one behavior that changes an observable result on well-formed default-param input
is the **constant-column handling** (REQ-2): ferrolearn `continue`s where sklearn
maps to `0`. Because the divergence is a single `if s == F::zero()` branch and the
`_handle_zeros_in_scale` semantics are well-defined (`scale_` of a constant
feature → `1`, and `with_mean` centers first so `(x-mean)/1 = 0`), it is the
minimal fixable divergence — the critic pins it (flipping
`test_zero_variance_column_unchanged` to expect `0.0`), the fixer changes the
branch to set the column to `(x - mean) / 1 = 0` (equivalently treat `scale_eff =
1` for zero-variance columns). The remaining gaps are *contract surface*: the full
fitted attribute set incl. `_handle_zeros` `scale_` (REQ-5), `with_mean`/`with_std`/
`copy` params (REQ-6), NaN tolerance (REQ-7), the `scale` free fn (REQ-8),
`partial_fit` (REQ-9), `sample_weight` (REQ-10), sparse support (REQ-11),
feature-name plumbing (REQ-12), and the ferray substrate (REQ-13).

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-3, REQ-4):

```bash
# Oracle (REQ-1 Probe) — non-constant per-column standardize, population std:
python3 -c "from sklearn.preprocessing import StandardScaler; \
print(StandardScaler().fit_transform([[1.,10.],[2.,20.],[3.,30.]]).tolist()); \
m=StandardScaler().fit([[1.,10.],[2.,20.],[3.,30.]]); \
print(m.mean_.tolist(), m.scale_.tolist())"
#   -> [[-1.224744871391589, -1.224744871391589], [0.0, 0.0], [1.224744871391589, 1.224744871391589]]
#   -> [2.0, 20.0] [0.816496580927726, 8.16496580927726]
# ferrolearn equivalent: StandardScaler::<f64>::new().fit(&X,&()).unwrap().transform(&X)

# Oracle (REQ-3 Probe) — inverse round-trip:
python3 -c "from sklearn.preprocessing import StandardScaler; \
X=[[1.,2.],[3.,4.],[5.,6.]]; m=StandardScaler().fit(X); \
print(m.inverse_transform(m.transform(X)).tolist())"
#   -> [[1.0000000000000002, 2.0], [3.0, 4.0], [5.0, 6.0]]

# REQ-2 divergence (must be pinned as a FAILING oracle test, then fixed):
python3 -c "from sklearn.preprocessing import StandardScaler; \
print(StandardScaler().fit_transform([[1.,5.],[2.,5.],[3.,5.]]).tolist())"
#   -> [[-1.224.., 0.0], [0.0, 0.0], [1.224.., 0.0]]  (ferrolearn currently returns 5.0 in col 1)

# Crate + binding gauntlet:
cargo test -p ferrolearn-preprocess   # incl. test_standard_scaler_zero_mean_unit_variance,
                                      #       test_inverse_transform_roundtrip,
                                      #       test_fit_transform_equivalence,
                                      #       test_shape_mismatch_error,
                                      #       test_insufficient_samples_error, test_f32_scaler
#       (NOTE: test_zero_variance_column_unchanged currently asserts the WRONG 5.0;
#        it must flip to expect 0.0 — REQ-2.)
cargo build -p ferrolearn-python      # registers _RsStandardScaler (REQ-4)
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s (`test_standard_scaler_zero_mean_unit_variance`,
`test_inverse_transform_roundtrip`, `test_fit_transform_equivalence`,
`test_shape_mismatch_error`, `test_insufficient_samples_error`, `test_f32_scaler`)
exercise REQ-1 and REQ-3 but are **not oracle-grounded** (they assert statistical
properties / hand-written values rather than the live sklearn output). To satisfy
R-CHAR-3 the critic should add an oracle-pinned guard asserting the REQ-1 Probe
output and `mean()`/`std()` against sklearn `mean_`/`scale_`.
`test_zero_variance_column_unchanged` is a **characterization-of-a-bug**: it pins
the divergent `5.0` and must flip to the oracle `0.0` (REQ-2). No currently-green
command establishes REQ-2 or REQ-5..REQ-13.

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns
`#`-numbers, replacing the `#<REQn>` placeholders); reference them in the REQ
status table:

- #1191 — REQ-2: `transform` leaves a constant (`s == 0`) column unchanged;
  sklearn maps it to `0` (`_handle_zeros_in_scale` scale→1, center first,
  `_data.py:88-120`,`:1019-1021`,`:1064-1067`). REQ-2 Probe: constant col → `0.0`,
  ferrolearn → original value. **The fixable divergence — pin + flip
  `test_zero_variance_column_unchanged` + fix first.**
- #1192 — REQ-5: `FittedStandardScaler` stores only `mean`/`std`; no `var_`,
  no `scale_` (= `_handle_zeros(sqrt(var))`, `1` on constant cols, ≠ raw `std`),
  no `n_samples_seen_` (`:1013-1023`).
- #1193 — REQ-6: no `with_mean`/`with_std`/`copy` params, no conditional
  center/scale, no `*`-only ctor, no `_parameter_constraints` (`:829-838`,
  `:1064-1067`).
- #1194 — REQ-7: `fit` folds propagate NaN; no `force_all_finite="allow-nan"`
  / NaN-ignoring mean/var (`:918`, `_more_tags allow_nan`, `:1112-1113`).
- #1195 — REQ-8: no standalone `scale` free fn; no `axis=1` row path (`:133`).
- #1196 — REQ-9: no `partial_fit` / streaming incremental mean-variance with
  `n_samples_seen_` accumulation (`:880-1025`).
- #1197 — REQ-10: no `sample_weight` (weighted mean/variance, `:923-924`,
  `:999-1005`).
- #1198 — REQ-11: dense-only; no sparse `with_mean=False` column scaling /
  raise on `with_mean=True` over sparse (`:940-983`,`:1055-1062`).
- #1199 — REQ-12: no `n_features_in_`/`feature_names_in_`/
  `get_feature_names_out` (OneToOneFeatureMixin).
- #1200 — REQ-13: compute path on `ndarray`/`num_traits`, not ferray
  (R-SUBSTRATE-1/2).
