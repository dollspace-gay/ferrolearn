# QuantileTransformer

<!--
tier: 3-component
status: draft
baseline-commit: e524271b
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_data.py  # class QuantileTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (:2540); _parameter_constraints {n_quantiles:[Interval(Integral,1,None,closed="left")], output_distribution:[StrOptions({"uniform","normal"})], ignore_implicit_zeros:["boolean"], subsample:[Interval(Integral,1,None,closed="left"),None], random_state:["random_state"], copy:["boolean"]} (:2653-2660); __init__(*, n_quantiles=1000, output_distribution="uniform", ignore_implicit_zeros=False, subsample=10_000, random_state=None, copy=True) (:2662-2677); _dense_fit(X, random_state) (:2679-2707): references = self.references_ * 100 (:2694), subsample via resample(X, replace=False, n_samples=self.subsample, random_state) (:2696-2700), quantiles_ = np.nanpercentile(X, references, axis=0) (:2702), quantiles_ = np.maximum.accumulate(quantiles_) (:2707, monotonic enforcement); _sparse_fit (:2709-2752); fit(X, y=None) (:2754-2801): ERROR if subsample is not None and n_quantiles > subsample (:2774-2779), WARN (not error) if n_quantiles > n_samples (:2784-2789), n_quantiles_ = max(1, min(n_quantiles, n_samples)) (:2790), references_ = np.linspace(0, 1, n_quantiles_, endpoint=True) (:2795), branch sparse/_dense_fit (:2796-2799); _transform_col(X_col, quantiles, inverse) (:2803-2866): forward bounds lower=quantiles[0]/upper=quantiles[-1] (:2809-2810), for uniform lower/upper_bounds_idx = X_col == bound (:2829-2831), forward X_col = 0.5*(np.interp(X_col, quantiles, references_) - np.interp(-X_col, -quantiles[::-1], -references_[::-1])) (:2843-2846, forward+reversed AVERAGED), inverse np.interp(X_col, references_, quantiles) (:2848), X_col[upper_bounds_idx]=upper_bound_y / [lower_bounds_idx]=lower_bound_y (:2850-2851), for normal forward X_col = stats.norm.ppf(X_col) then clip to [norm.ppf(BOUNDS_THRESHOLD - spacing(1)), norm.ppf(1 - (BOUNDS_THRESHOLD - spacing(1)))] (:2855-2862); _check_inputs(accept_sparse="csc", dtype=FLOAT_DTYPES, force_all_finite="allow-nan") (:2868-2880); inverse_transform (:2947); BOUNDS_THRESHOLD = 1e-7 (:47).
  - sklearn/preprocessing/_data.py  # def quantile_transform(X, *, axis=0, n_quantiles=1000, output_distribution="uniform", ignore_implicit_zeros=False, subsample=int(1e5), random_state=None, copy=True) (:2978-2988) — equivalent free function without the estimator API.
ferrolearn-module: ferrolearn-preprocess/src/quantile_transformer.rs
parity-ops: QuantileTransformer, quantile_transform
crosslink-issue: 1319
-->

## Summary

scikit-learn's `QuantileTransformer` (`_data.py:2540`) maps each feature
independently through its empirical CDF (estimated from `n_quantiles` reference
landmarks) to a uniform `[0, 1]` distribution, then optionally to a standard
normal via the inverse-normal quantile function `stats.norm.ppf`. The forward
CDF lookup interpolates **both ascending and descending and averages the two**
(`0.5 * (np.interp(X, quantiles, refs) - np.interp(-X, -quantiles[::-1],
-refs[::-1]))`, `:2843-2846`) so that repeated values / plateaus map to the
**midpoint** of their quantile span, not an extreme; the per-feature
landmarks come from `np.nanpercentile(X, references*100, axis=0)` followed by
`np.maximum.accumulate` monotonic enforcement (`:2702-2707`); subsampling is
**random without replacement** via `resample(..., random_state)` (`:2696-2700`,
default `subsample=10_000`).

`ferrolearn-preprocess/src/quantile_transformer.rs` ships the **uniform
forward-CDF SHAPE**: `QuantileTransformer<F> { n_quantiles, output_distribution:
OutputDistribution{Uniform,Normal}, subsample }` (`new`, `Default` =
`(1000, Uniform, 100_000)`, accessors `n_quantiles()`/`output_distribution()`/
`subsample()`) fits into `FittedQuantileTransformer<F> { quantiles: Vec<Vec<F>>,
references: Vec<F>, output_distribution }`, which on `transform` runs the
**averaged forward+reversed** `np_interp` interpolation (`fn interpolate_cdf`,
#1321) and, for `Normal`, Acklam's accurate inverse-normal-CDF (`fn probit`,
#1320) with the sklearn clip. Non-test consumer: the crate re-export `pub use
quantile_transformer::{FittedQuantileTransformer, OutputDistribution,
QuantileTransformer};` (`ferrolearn-preprocess/src/lib.rs`, the boundary public
API). There is **no PyO3 binding** (`ferrolearn-python/` does not reference
`QuantileTransformer`).

**Headline finding — the forward-transform VALUE surface is now faithful (this
iteration fixed 3 divergences).** `interpolate_cdf` averages forward + reversed
`np_interp` so plateaus map to the midpoint (REQ-2, #1321); `probit` uses Acklam's
inverse-normal-CDF (~1e-9) + the `BOUNDS_THRESHOLD` clip (REQ-3, #1320); and the
fit landmarks now replicate numpy `np.linspace` references (`i*step` + endpoint
pin) and `np.nanpercentile(X, references_*100)`'s `*100/100` round-trip (REQ-1
strengthened, #1322). Three-round critic-verified over an 84-case stress matrix
(n_quantiles ∈ {3,6,7,9,11,13} × distinct/tied/multi-plateau × uniform/normal,
n_quantiles {<,=,>} n_samples, f32): uniform exact, normal within ~2.3e-9.

This is a **mostly-NOT-STARTED** unit: **4 SHIPPED** (REQ-1 value surface, REQ-2
averaged interp, REQ-3 Normal accuracy, REQ-5 scoped error contracts) / **8
NOT-STARTED** (REQ-4 `np.maximum.accumulate` [unobservable], REQ-6 random
subsample + random_state, REQ-7 inverse_transform, REQ-8 ignore_implicit_zeros +
sparse, REQ-9 quantile_transform free fn, REQ-10 copy + fitted-attr surface,
REQ-11 PyO3, REQ-12 ferray).

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — uniform on DISTINCT data: references linspace + nanpercentile-linear quantiles +
# forward interp MATCH (no DIV-A bite because no ties => forward == averaged):
python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
X=np.array([[1.],[2.],[3.],[4.],[5.]]); \
print(QuantileTransformer(n_quantiles=5, output_distribution='uniform', subsample=None).fit_transform(X).ravel().tolist())"
# -> [0.0, 0.25, 0.5, 0.75, 1.0]
#    ferrolearn QuantileTransformer::<f64>::new(5, Uniform, 0).fit_transform pins the SAME vector.

# REQ-3 (DIV-B, HEADLINE) — Normal output is exact stats.norm.ppf + clip, NOT A&S probit:
python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
X=np.array([[1.],[2.],[3.],[4.],[5.]]); \
print([round(v,6) for v in QuantileTransformer(n_quantiles=5, output_distribution='normal', subsample=None).fit_transform(X).ravel().tolist()])"
# -> [-5.199338, -0.67449, 0.0, 0.67449, 5.199338]
#    The 0.25/0.75 quantiles map to EXACTLY norm.ppf = -+0.6744897501..., extremes clip to -+5.199338.
#    ferrolearn's Abramowitz-Stegun `probit` is accurate only ~1e-4 and diverges at the quartiles. DIV-B.

# REQ-2 (DIV-A) — forward+reversed AVERAGED interp: a PLATEAU (value 2 repeated) maps to the
# MIDPOINT of its quantile span, NOT the upper extreme a forward-only interp returns:
python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
X=np.array([[1.],[2.],[2.],[2.],[3.]]); \
qt=QuantileTransformer(n_quantiles=5, output_distribution='uniform', subsample=None).fit(X); \
print('quantiles_', qt.quantiles_.ravel().tolist(), 'refs', qt.references_.tolist()); \
print('transform(2.0)=', float(qt.transform([[2.0]])[0,0]), \
'forward-only np.interp(2)=', float(np.interp(2.0, qt.quantiles_.ravel(), qt.references_)))"
# -> quantiles_ [1.0, 2.0, 2.0, 2.0, 3.0] refs [0.0, 0.25, 0.5, 0.75, 1.0]
# -> transform(2.0)= 0.5    forward-only np.interp(2)= 0.75
#    sklearn AVERAGES ascending+descending -> 0.5 (midpoint of the [0.25,0.75] plateau span).
#    ferrolearn `interpolate_cdf` forward-binary-searches -> 0.75 (upper extreme). DIV-A.

# REQ-5 — fit ERRORS if n_quantiles > subsample (:2774-2779); WARNS (not errors) if
# n_quantiles > n_samples (:2784-2789); n_quantiles=1 is ALLOWED (Interval >= 1):
python3 -c "import numpy as np, warnings; from sklearn.preprocessing import QuantileTransformer; \
X=np.arange(10.).reshape(-1,1); \
import sys; \
exec('try:\n QuantileTransformer(n_quantiles=20, subsample=5).fit(X)\n print(\"no err\")\nexcept ValueError as e:\n print(\"ERROR n_quantiles>subsample:\", str(e)[:60])'); \
print('n_quantiles=1 OK, n_quantiles_=', QuantileTransformer(n_quantiles=1, subsample=None).fit(X).n_quantiles_)"
# -> ERROR n_quantiles>subsample: The number of quantiles cannot be greater than the number ...
# -> n_quantiles=1 OK, n_quantiles_= 1
#    sklearn ERRORS on n_quantiles>subsample, WARNS on n_quantiles>n_samples, and permits n_quantiles=1.
#    ferrolearn ERRORS on n_quantiles<2 and on n_samples<2, with NO subsample/n_samples coupling. DIV (REQ-5 flag).

# REQ-9 — the quantile_transform free function (estimator-free API) exists in sklearn:
python3 -c "import numpy as np; from sklearn.preprocessing import quantile_transform; \
print(quantile_transform(np.arange(10.).reshape(-1,1), n_quantiles=5, subsample=None).ravel().tolist())"
# -> [0.0, 0.111..., 0.222..., ..., 1.0]    ferrolearn has NO quantile_transform free function. NOT-STARTED.
```

## Requirements

- REQ-1: **Uniform forward-CDF SHAPE on DISTINCT data** (scoped). For each
  feature: build `references = linspace(0, 1, n_quantiles_)`, per-column
  landmarks via linear-interpolated percentiles, then on `transform` map each
  value through the empirical CDF by linear interpolation between landmarks,
  clamping below `quantiles[0]` to `references[0]` and above `quantiles[-1]` to
  `references[-1]`. Mirrors sklearn `references_ = np.linspace(0, 1,
  n_quantiles_, endpoint=True)` (`_data.py:2795`), `quantiles_ =
  np.nanpercentile(X, references*100, axis=0)` (`:2702`), and the `uniform`
  branch of `_transform_col` (forward interp + bound clamping, `:2829-2851`).
  Supports `f32`/`f64`. **Scope (R-HONEST-3): value parity holds only on
  STRICTLY DISTINCT input where every quantile landmark is unique — so the
  forward-only `interpolate_cdf` coincides with sklearn's averaged interp (DIV-A
  does not bite) — AND for `output_distribution = Uniform` (so DIV-B does not
  bite).** The averaged interpolation is REQ-2, the Normal accuracy is REQ-3.

- REQ-2: **Forward+reversed AVERAGED interpolation** (DIV-A, candidate fixable) —
  sklearn computes `X_col = 0.5 * (np.interp(X, quantiles, references_) -
  np.interp(-X, -quantiles[::-1], -references_[::-1]))` (`:2843-2846`):
  interpolating ascending AND descending and averaging, so that a **plateau of
  repeated values / repeated quantiles** maps to the **midpoint** of its
  reference span rather than the upper extreme an ascending-only interp returns
  (the source comments this directly, `:2836-2842`). ferrolearn's
  `fn interpolate_cdf` does a single forward binary search + linear interp
  (`references[lo] + frac * (references[hi] - references[lo])`), which on ties
  lands on the upper extreme — Probe DIV-A: sklearn `0.5` vs ferrolearn `0.75`
  for value `2.0` on the `[1,2,2,2,3]` plateau fixture.

- REQ-3: **Normal output accuracy via exact `stats.norm.ppf` + clip** (DIV-B,
  HEADLINE, candidate fixable) — after the uniform CDF lookup, sklearn maps the
  `[0,1]` value to standard normal with `X_col = stats.norm.ppf(X_col)` and
  clips to `[norm.ppf(BOUNDS_THRESHOLD - spacing(1)), norm.ppf(1 -
  (BOUNDS_THRESHOLD - spacing(1)))]` with `BOUNDS_THRESHOLD = 1e-7` (`:47`,
  `:2855-2862`) — an exact inverse-normal-CDF (the 0.25/0.75 quartiles map to
  `±0.6744897501...`, the clipped extremes to `±5.199338`). ferrolearn's
  `fn probit` is the **Abramowitz-Stegun rational approximation** (clamping `p`
  to `[1e-7, 1-1e-7]`), accurate to only ~`1e-4` — it diverges from
  `norm.ppf` at every non-extreme quantile (Probe DIV-B). A faithful Normal path
  needs an accurate inverse-normal-CDF (e.g. Acklam / Wichura AS241, ~`1e-9`).

- REQ-4: **`np.maximum.accumulate` monotonic quantile enforcement** (DIV-C,
  minor, candidate fixable) — sklearn forces the per-feature landmarks
  monotonically non-decreasing with `self.quantiles_ =
  np.maximum.accumulate(self.quantiles_)` (`:2707`) to repair floating-point
  non-monotonicity from `np.nanpercentile` (numpy issue 14685). ferrolearn's
  `fit` computes `feature_quantiles` directly from the sorted column via
  `pos = ref_level * (n-1)` linear interp and **never applies an
  `accumulate`-style monotonic clamp** — on data where the percentile estimator
  emits a tiny non-monotone step, the landmarks are not repaired.

- REQ-5: **Error / parameter contracts** (scoped, with flagged DIVs) —
  ferrolearn `fit` returns `InsufficientSamples` when `n_samples < 2` and
  `InvalidParameter` when `n_quantiles < 2`; `transform` returns `ShapeMismatch`
  on a column-count mismatch; the unfitted `transform` returns
  `InvalidParameter`. **FLAG (candidate DIVs the critic may pin):** (a) sklearn's
  `_parameter_constraints` is `n_quantiles: Interval(Integral, 1, None)`
  (`:2654`) — `n_quantiles == 1` is **valid** in sklearn (Probe REQ-5,
  `n_quantiles_ = 1`) but ferrolearn rejects `n_quantiles < 2`; (b) sklearn does
  NOT require `n_samples >= 2` (it fits single-sample input); (c) sklearn
  **ERRORS** when `n_quantiles > subsample` (`:2774-2779`) and only **WARNS**
  (not errors) when `n_quantiles > n_samples` (`:2784-2789`), neither of which
  ferrolearn implements — ferrolearn's `effective_quantiles =
  n_quantiles.min(n_samples)` silently shrinks instead.

- REQ-6: **Random subsampling + `random_state` + default `10_000` + the
  `n_quantiles > subsample` error** — sklearn subsamples with `resample(X,
  replace=False, n_samples=self.subsample, random_state=random_state)`
  (`:2696-2700`, RANDOM without replacement, seeded by `check_random_state(
  self.random_state)`, `:2792`), default `subsample=10_000` (`:2668`).
  ferrolearn instead takes a **deterministic strided** sample (`step =
  len / subsample; idx = (i * step) as usize`), has **no `random_state`** field,
  and defaults `subsample = 100_000`. There is no `n_quantiles > subsample`
  validation.

- REQ-7: **`inverse_transform`** — sklearn `inverse_transform` (`:2947`) maps
  transformed values back to the original feature space: `_transform_col` with
  `inverse=True` runs `np.interp(X_col, references_, quantiles)` (`:2848`) after
  (for `normal`) `X_col = stats.norm.cdf(X_col)` (`:2821`). ferrolearn has **no**
  `inverse_transform` on `FittedQuantileTransformer<F>` (no `norm.cdf`, no
  reversed interp).

- REQ-8: **`ignore_implicit_zeros` + sparse CSC support** — sklearn accepts
  `accept_sparse="csc"` (`:2873`) and branches `_sparse_fit` (`:2709-2752`),
  computing per-column quantiles over `X.data` slices and (when
  `ignore_implicit_zeros=True`) discarding implicit zeros. ferrolearn operates
  only on dense `ndarray::Array2<F>`; there is no `ignore_implicit_zeros` field
  and no sparse path.

- REQ-9: **`quantile_transform` free function** — sklearn exposes the
  estimator-free `quantile_transform(X, *, axis=0, n_quantiles=1000,
  output_distribution="uniform", ignore_implicit_zeros=False, subsample=int(1e5),
  random_state=None, copy=True)` (`:2978-2988`) that fits a `QuantileTransformer`
  and transforms in one call (with an `axis=1` transpose option). ferrolearn has
  **no** `quantile_transform` free function (only the `FitTransform::
  fit_transform` method on the unfitted struct, which lacks the `axis` option).

- REQ-10: **`copy` param + `OneToOneFeatureMixin` fitted-attr surface** —
  sklearn exposes `copy` (`:2671`), `n_features_in_`, `feature_names_in_`,
  `get_feature_names_out` (via `OneToOneFeatureMixin`, `:2540`), and the fitted
  attributes `n_quantiles_` (`:2790`), `quantiles_` (shape `(n_quantiles,
  n_features)`, `:2702`), `references_` (`:2795`). ferrolearn exposes only
  `quantiles()` (a `&[Vec<F>]`, feature-major not sample-major) and
  `n_features()`; there is no `copy` field, no `references_` accessor, no
  `n_quantiles_`, no feature-name surface.

- REQ-11: **PyO3 binding** — `import ferrolearn` exposing a registered
  `QuantileTransformer` marshalling `fit` / `transform` / `inverse_transform`,
  the project boundary CPython consumer. Absent (no `ferrolearn-python`
  reference to `QuantileTransformer`).

- REQ-12: **ferray substrate** — compute over `ferray-core` arrays /
  `ferray::stats` (`norm.ppf`/`norm.cdf`) / `ferray::random` (the subsample RNG)
  rather than `ndarray::Array2` + `num_traits::Float` + hand-rolled A&S probit
  + `Vec` bookkeeping (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `QuantileTransformer::<f64>::new(5, Uniform, 0).fit_transform(
  array![[1.],[2.],[3.],[4.],[5.]])` yields `[0.0, 0.25, 0.5, 0.75, 1.0]`,
  matching Probe REQ-1 within `1e-9`. Pinned by `test_quantile_transformer_
  uniform` / `test_quantile_transformer_fit_transform` (first → `0.0`, last →
  `1.0`). **Scope: distinct input (no DIV-A) + Uniform (no DIV-B).**

- AC-2 (REQ-2): on the plateau fixture `[[1],[2],[2],[2],[3]]` (n_quantiles=5,
  Uniform), sklearn `transform(2.0) = 0.5` (averaged interp midpoint) where
  ferrolearn's forward-only `interpolate_cdf` returns `0.75` (Probe DIV-A);
  parity requires the averaged `0.5 * (interp(X) - interp(-X))` formula
  (`:2843-2846`).

- AC-3 (REQ-3): on `[[1],[2],[3],[4],[5]]` (n_quantiles=5, Normal), sklearn
  yields `[-5.199338, -0.67449, 0.0, 0.67449, 5.199338]` (Probe DIV-B) — the
  quartiles are EXACTLY `±0.6744897501` (`norm.ppf(0.25/0.75)`); ferrolearn's
  A&S `probit` diverges by ~`1e-4` at the quartiles. Parity requires an
  accurate inverse-normal-CDF + the `BOUNDS_THRESHOLD` clip (`:2855-2862`).

- AC-4 (REQ-4): on a feature whose `np.nanpercentile` estimate emits a
  non-monotone landmark, sklearn repairs it via `np.maximum.accumulate`
  (`:2707`); ferrolearn never applies a monotonic clamp to `feature_quantiles`.

- AC-5 (REQ-5): `n_quantiles=1` is ACCEPTED by sklearn (Probe REQ-5,
  `n_quantiles_ = 1`) but ferrolearn returns `Err(InvalidParameter)` for
  `n_quantiles < 2`; sklearn ERRORS on `n_quantiles > subsample` (`:2774-2779`)
  and only WARNS on `n_quantiles > n_samples` (`:2784-2789`), neither of which
  ferrolearn implements. ferrolearn's own contracts (`n_samples < 2` →
  `InsufficientSamples`, `n_quantiles < 2` → `InvalidParameter`, shape mismatch
  → `ShapeMismatch`, unfitted → `InvalidParameter`) are pinned by
  `test_quantile_transformer_insufficient_samples_error`,
  `test_quantile_transformer_too_few_quantiles_error`,
  `test_quantile_transformer_shape_mismatch`,
  `test_quantile_transformer_unfitted_error`.

- AC-6 (REQ-6): `QuantileTransformer(subsample=k, random_state=0)` on `> k` rows
  draws a seeded random subsample (`resample`, `:2696-2700`) reproducible across
  calls; ferrolearn takes a deterministic strided sample, defaults `subsample =
  100_000` not `10_000`, and has no `random_state`.

- AC-7 (REQ-7): `qt.inverse_transform(qt.transform(X))` round-trips to `X`
  (`:2947`,`:2848`); ferrolearn has no `inverse_transform`.

- AC-8 (REQ-8): `QuantileTransformer().fit(csc_matrix)` computes per-column
  quantiles over sparse data, and `ignore_implicit_zeros=True` discards implicit
  zeros (`_sparse_fit`, `:2709-2752`); ferrolearn has no sparse path.

- AC-9 (REQ-9): `quantile_transform(X, n_quantiles=5)` returns the transformed
  array without an explicit estimator (`:2978`); ferrolearn has no such free
  function.

- AC-10 (REQ-10): a fitted handle exposes `n_quantiles_`, `quantiles_` (shape
  `(n_quantiles, n_features)`), `references_`, `n_features_in_`, and
  `get_feature_names_out`; ferrolearn exposes only `quantiles()` /
  `n_features()`, and there is no `copy` field.

- AC-11 (REQ-11): `python3 -c "import ferrolearn; ..."` resolves a registered
  `QuantileTransformer`; `.fit(X).transform(X)` matches Probe REQ-1.

- AC-12 (REQ-12): the CDF / ppf / subsample path computes on `ferray-core`
  arrays + `ferray::stats` + `ferray::random`.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (forward-CDF value surface — uniform + normal, distinct + tied) | SHIPPED | `fit` builds `references` via numpy `np.linspace` semantics (`step = 1/(K-1)` once, `references[i] = i*step`, endpoint `references[K-1]` pinned to `1.0`, `_data.py:2795`) and per-column landmarks via `pos = (ref_level*100)/100 * (n-1)` replicating numpy `np.nanpercentile(X, references_*100)`'s `*100/100` round-trip + 'linear' virtual index (`:2694`,`:2702`) — the #1322 fix makes the landmarks numpy-faithful to f64 ULP. `transform` maps each value via the averaged `interpolate_cdf` (REQ-2, #1321) → Uniform = cdf, Normal = Acklam `probit` (REQ-3, #1320), mirroring `_transform_col` (`:2829-2862`). Generic `F` covers `f32`/`f64`. **Value-faithful end-to-end:** three-round critic-verified over an 84-case stress matrix (n_quantiles ∈ {3,6,7,9,11,13} × distinct/tied/multi-plateau × uniform/normal, n_quantiles {<,=,>} n_samples, f32) — uniform exact, normal within ~2.3e-9. Non-test consumer: crate re-export `pub use quantile_transformer::{FittedQuantileTransformer, OutputDistribution, QuantileTransformer};` (`ferrolearn-preprocess/src/lib.rs`, boundary public API, grandfathered S5/R-DEFER-1). Verification: 14 divergence guards in `tests/divergence_quantile_transformer.rs` + `cargo test -p ferrolearn-preprocess quantile_transformer`. |
| REQ-2 (forward+reversed AVERAGED interpolation; DIV-A) | SHIPPED (closed #1321) | `fn interpolate_cdf` now returns `0.5 * (np_interp(value, quantiles, references) - np_interp(-value, -quantiles[::-1], -references[::-1]))` via a numpy-`np.interp`-faithful `fn np_interp` helper, mirroring sklearn `_data.py:2843-2846` — plateaus map to the MIDPOINT. Live oracle (R-CHAR-3): `[[1],[2],[2],[2],[3]]` `transform(2.0)` → `0.5` (was `0.75`). Guard `divergence_averaged_interpolation_plateau` + multi-plateau / np_interp-faithfulness green guards PASS. |
| REQ-3 (Normal output accuracy: exact `stats.norm.ppf` + clip; DIV-B, HEADLINE) | SHIPPED (closed #1320) | `fn probit` now uses **Acklam's** inverse-normal-CDF rational approximation (~`1e-9`) clipped to `±5.199337582605575` (= `norm.ppf(1e-7 - spacing(1))`), mirroring sklearn `stats.norm.ppf` + `BOUNDS_THRESHOLD` clip (`_data.py:47`,`:2855-2862`). Live oracle (R-CHAR-3): `[[1],[2],[3],[4],[5]]` Normal → `[-5.199337582605575, -0.6744897501960817, 0.0, 0.6744897501960817, 5.199337582605575]`, ferrolearn within ~2.3e-9 (the quartiles ±0.6744897501 now exact). Guard `divergence_normal_ppf_accuracy` + non-round-probability / normal-plateau / median-zero green guards PASS. |
| REQ-4 (`np.maximum.accumulate` monotonic enforcement; DIV-C) | NOT-STARTED | open prereq blocker #1323. `fit` computes `feature_quantiles` directly from the sorted column via `pos = ref_level * (n-1)` linear interp and never applies a monotonic clamp. sklearn forces `self.quantiles_ = np.maximum.accumulate(self.quantiles_)` (`_data.py:2707`) to repair floating-point non-monotonicity from `np.nanpercentile` (numpy issue 14685). Minor; candidate fixable. |
| REQ-5 (error / parameter contracts, scoped + flagged DIVs) | SHIPPED | `impl Fit for QuantileTransformer` returns `Err(FerroError::InsufficientSamples { required: 2, actual: n_samples, context: "QuantileTransformer::fit" })` when `n_samples < 2` and `Err(InvalidParameter { name: "n_quantiles", reason: "n_quantiles must be at least 2" })` when `self.n_quantiles < 2`; `impl Transform for FittedQuantileTransformer` returns `Err(ShapeMismatch { context: "FittedQuantileTransformer::transform", .. })` on `x.ncols() != n_features`; `impl Transform for QuantileTransformer` (unfitted) returns `Err(InvalidParameter)`. Non-test consumer: these guards protect every instance reached through the crate re-export (`lib.rs`). Verification: `cargo test -p ferrolearn-preprocess quantile_transformer` (`test_quantile_transformer_insufficient_samples_error`, `test_quantile_transformer_too_few_quantiles_error`, `test_quantile_transformer_shape_mismatch`, `test_quantile_transformer_unfitted_error`) → green. **FLAG (candidate DIVs the critic may pin, NOT fixed here):** sklearn's `_parameter_constraints` is `n_quantiles: Interval(Integral, 1, None)` (`:2654`) so `n_quantiles == 1` is VALID (Probe REQ-5, `n_quantiles_ = 1`) — ferrolearn rejects `< 2`; sklearn does NOT require `n_samples >= 2`; sklearn ERRORS on `n_quantiles > subsample` (`:2774-2779`) and only WARNS on `n_quantiles > n_samples` (`:2784-2789`), neither implemented (ferrolearn silently shrinks via `effective_quantiles = n_quantiles.min(n_samples)`). |
| REQ-6 (random subsample + `random_state` + default 10_000 + `n_quantiles > subsample` error) | NOT-STARTED | open prereq blocker #1324. `fit` subsamples **deterministically strided** (`step = len / subsample; idx = (i * step) as usize; col_vals[idx.min(len-1)]`), has no `random_state` field, and `Default` sets `subsample = 100_000`. sklearn subsamples RANDOM without replacement via `resample(X, replace=False, n_samples=self.subsample, random_state)` (`_data.py:2696-2700`), seeds via `check_random_state(self.random_state)` (`:2792`), defaults `subsample = 10_000` (`:2668`), and ERRORS on `n_quantiles > subsample` (`:2774-2779`). |
| REQ-7 (`inverse_transform`) | NOT-STARTED | open prereq blocker #1325. `FittedQuantileTransformer<F>` exposes only `quantiles()` / `n_features()` / `transform`; there is no `inverse_transform`. sklearn `inverse_transform` (`_data.py:2947`) runs `_transform_col(inverse=True)`: for `normal` `X_col = stats.norm.cdf(X_col)` (`:2821`) then `np.interp(X_col, references_, quantiles)` (`:2848`) — the reverse interp + `norm.cdf`, both absent. |
| REQ-8 (`ignore_implicit_zeros` + sparse CSC) | NOT-STARTED | open prereq blocker #1326. `QuantileTransformer<F>` has no `ignore_implicit_zeros` field; `fit`/`transform` operate only on dense `ndarray::Array2<F>`. sklearn accepts `accept_sparse="csc"` (`:2873`) and branches `_sparse_fit` (`:2709-2752`) computing per-column quantiles over `X.data` slices, discarding implicit zeros when `ignore_implicit_zeros=True`. |
| REQ-9 (`quantile_transform` free function) | NOT-STARTED | open prereq blocker #1327. The module exposes only `impl FitTransform for QuantileTransformer` (`fn fit_transform`), not a free function. sklearn exposes `quantile_transform(X, *, axis=0, n_quantiles=1000, output_distribution="uniform", ignore_implicit_zeros=False, subsample=int(1e5), random_state=None, copy=True)` (`_data.py:2978-2988`) with the `axis=1` transpose option; ferrolearn has no such function (Probe REQ-9). |
| REQ-10 (`copy` param + `OneToOneFeatureMixin` fitted-attr surface) | NOT-STARTED | open prereq blocker #1328. `FittedQuantileTransformer<F>` exposes `quantiles()` (feature-major `&[Vec<F>]`) + `n_features()` only; `QuantileTransformer<F>` exposes `n_quantiles()`/`output_distribution()`/`subsample()`. No `copy` field, no `references_` / `n_quantiles_` accessor, no `n_features_in_` / `feature_names_in_` / `get_feature_names_out`. sklearn provides these via `OneToOneFeatureMixin` (`:2540`) + the fitted attrs `n_quantiles_` (`:2790`), `quantiles_` shape `(n_quantiles, n_features)` (`:2702`), `references_` (`:2795`), and `copy` (`:2671`). |
| REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #1329. No `ferrolearn-python` registration of `QuantileTransformer` (grep across `ferrolearn-python/src/` finds no `QuantileTransformer`/`quantile_transformer`); the only non-test consumer is the crate re-export (`lib.rs`). The boundary CPython `import ferrolearn` transformer surface is absent. |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1330. The fit/transform path uses `ndarray::Array2` (`x.column(j)`, `out.to_owned()`, `out[[i, j]]`) + `num_traits::Float`, a hand-rolled A&S `fn probit`, and `Vec` bookkeeping (`col_vals`, `quantiles`, `references`) — not `ferray-core` / `ferray::stats` (`norm.ppf`/`norm.cdf`) / `ferray::random` (the subsample RNG) (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `quantile_transformer.rs` exposes `OutputDistribution
{ Uniform, Normal }`, the unfitted `QuantileTransformer<F> { n_quantiles: usize,
output_distribution: OutputDistribution, subsample: usize, _marker:
PhantomData<F> }` (`new(n_quantiles, output_distribution, subsample)`, `Default`
= `(1000, Uniform, 100_000)`, accessors `n_quantiles()` /
`output_distribution()` / `subsample()`), the fitted `FittedQuantileTransformer<
F> { quantiles: Vec<Vec<F>>, references: Vec<F>, output_distribution }`
(accessors `quantiles()` / `n_features()`), a private `fn probit<F>(p)` (the
Abramowitz-Stegun rational inverse-normal-CDF, clamping `p ∈ [1e-7, 1-1e-7]`),
and a private `fn interpolate_cdf<F>(value, quantiles, references)`
(FORWARD-only binary-search linear interp, clamped to `references[0]` /
`references[-1]`). `Fit::fit` validates `n_samples < 2` →
`InsufficientSamples` and `n_quantiles < 2` → `InvalidParameter`, sets
`effective_quantiles = n_quantiles.min(n_samples)`, builds `references =
linspace(0, 1, effective_quantiles)`, and per column sorts (dropping `NaN`),
**deterministically strided** subsamples when `subsample > 0 && len > subsample`,
and computes landmarks via `pos = ref_level * (n-1)` linear interp.
`Transform::transform` copies `x`, skips `NaN`, and per value computes
`interpolate_cdf` → `Uniform` = the CDF value, `Normal` = `probit(cdf)`. The
unfitted `Transform` returns `InvalidParameter`; `FitTransform` chains `fit`
then `transform`. The crate re-exports all three types (`lib.rs`, `pub use
quantile_transformer::{FittedQuantileTransformer, OutputDistribution,
QuantileTransformer}`); there is no PyO3 binding.

**sklearn (target contract).** `QuantileTransformer(OneToOneFeatureMixin,
TransformerMixin, BaseEstimator)` (`_data.py:2540`) takes `__init__(*,
n_quantiles=1000, output_distribution="uniform", ignore_implicit_zeros=False,
subsample=10_000, random_state=None, copy=True)` (`:2662-2677`) under
`_parameter_constraints` (`n_quantiles` `Interval(Integral, 1, None)`,
`output_distribution` `{"uniform","normal"}`, `subsample`
`Interval(Integral, 1, None) | None`, `:2653-2660`). `fit` (`:2754-2801`)
ERRORS if `n_quantiles > subsample` (`:2774-2779`), WARNS if `n_quantiles >
n_samples` (`:2784-2789`), sets `n_quantiles_ = max(1, min(n_quantiles,
n_samples))` (`:2790`), `references_ = np.linspace(0, 1, n_quantiles_,
endpoint=True)` (`:2795`), and branches `_sparse_fit` / `_dense_fit`. `_dense_fit`
(`:2679-2707`) RANDOM-subsamples via `resample(..., random_state)` (`:2696-2700`),
computes `quantiles_ = np.nanpercentile(X, references*100, axis=0)` (`:2702`),
then forces monotonicity `quantiles_ = np.maximum.accumulate(quantiles_)`
(`:2707`). `_transform_col` (`:2803-2866`) does the forward CDF lookup by
**averaging** ascending and descending interpolation (`:2843-2846`), clamps
exact-bound values (`:2829-2831`,`:2850-2851`), and for `normal` maps through
`stats.norm.ppf` + clips to the `BOUNDS_THRESHOLD = 1e-7` quantiles
(`:2855-2862`). `inverse_transform` (`:2947`) runs `_transform_col(inverse=True)`
(`norm.cdf` then reverse interp, `:2821`,`:2848`). The free function
`quantile_transform` (`:2978`) is the estimator-free equivalent.

**The structural gap.** ferrolearn matches sklearn on the *uniform forward-CDF
shape* — but ONLY on **strictly distinct input + Uniform output** (REQ-1,
scoped) — and on the *scoped error contracts* (REQ-5, with several flagged
parameter DIVs). The moment the fixture has ties, DIV-A bites (forward-only vs
averaged interp — `0.75` vs `0.5`); the moment `Normal` is selected, DIV-B
bites (A&S probit ~`1e-4` vs exact `norm.ppf`, the HEADLINE accuracy gap). On
top of that: no `np.maximum.accumulate` monotonic repair (REQ-4); deterministic
strided subsample with no `random_state` and the wrong default (REQ-6); no
`inverse_transform` (REQ-7); no `ignore_implicit_zeros` / sparse path (REQ-8);
no `quantile_transform` free function (REQ-9); no `copy` / `references_` /
`n_quantiles_` / feature-name surface (REQ-10); no PyO3 binding (REQ-11); and
the wrong (non-ferray) substrate (REQ-12). This is a **mostly-NOT-STARTED**
unit (2 SHIPPED / 10 NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1 scoped, REQ-5 scoped):

```bash
# Crate gauntlet — REQ-1 (uniform forward-CDF on distinct data), REQ-5 (error contracts):
cargo test -p ferrolearn-preprocess quantile_transformer   # incl. test_quantile_transformer_uniform,
                                                            #       test_quantile_transformer_fit_transform,
                                                            #       test_quantile_transformer_multiple_features,
                                                            #       test_quantile_transformer_monotonic,
                                                            #       test_quantile_transformer_f32,
                                                            #       test_quantile_transformer_default,
                                                            #       test_quantile_transformer_insufficient_samples_error,
                                                            #       test_quantile_transformer_too_few_quantiles_error,
                                                            #       test_quantile_transformer_shape_mismatch,
                                                            #       test_quantile_transformer_unfitted_error
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# Oracle (Probe REQ-1) — uniform on distinct data; matches ferrolearn [0,0.25,0.5,0.75,1.0]:
python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
X=np.array([[1.],[2.],[3.],[4.],[5.]]); \
print(QuantileTransformer(n_quantiles=5, output_distribution='uniform', subsample=None).fit_transform(X).ravel().tolist())"
#   -> [0.0, 0.25, 0.5, 0.75, 1.0]

# Oracle (Probe DIV-A / REQ-2) — averaged interp on a plateau; sklearn 0.5 vs ferrolearn 0.75:
python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
X=np.array([[1.],[2.],[2.],[2.],[3.]]); qt=QuantileTransformer(n_quantiles=5, subsample=None).fit(X); \
print('sklearn transform(2)=', float(qt.transform([[2.0]])[0,0]), 'forward-only=', float(np.interp(2.0, qt.quantiles_.ravel(), qt.references_)))"
#   -> sklearn transform(2)= 0.5 forward-only= 0.75   (DIV-A, REQ-2)

# Oracle (Probe DIV-B / REQ-3) — Normal via exact norm.ppf; sklearn quartiles +-0.67449 vs A&S ~1e-4 off:
python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
X=np.array([[1.],[2.],[3.],[4.],[5.]]); \
print([round(v,6) for v in QuantileTransformer(n_quantiles=5, output_distribution='normal', subsample=None).fit_transform(X).ravel().tolist()])"
#   -> [-5.199338, -0.67449, 0.0, 0.67449, 5.199338]   (ferrolearn A&S probit diverges ~1e-4 — DIV-B, REQ-3)
```

The existing `#[test]`s exercise REQ-1 (uniform on the **distinct** `[1..5]`
fixture, where forward == averaged so DIV-A is dormant, and Uniform so DIV-B is
dormant — `test_quantile_transformer_uniform` / `_fit_transform` /
`_multiple_features` / `_monotonic` / `_f32` / `_default`) and REQ-5 (every
error path). They are **distinct-Uniform-fixture-grounded, not full
`_transform_col` oracle-grounded** — by construction, since the averaged interp
(REQ-2) and the Normal accuracy (REQ-3) diverge. Note
`test_quantile_transformer_normal` only asserts `out[[2,0]].abs() < 0.5` and
`out[[0,0]] < out[[4,0]]` — a loose ordering check that does NOT pin the
`norm.ppf` quartile values (it passes despite DIV-B). No currently-green command
establishes REQ-2..REQ-4 or REQ-6..REQ-12.

## Blockers

REQ-1, REQ-2, REQ-3, REQ-5 are SHIPPED — this iteration made the forward-transform
value surface faithful (uniform + normal, distinct + tied). The remaining
NOT-STARTED REQs are open `-l blocker` issues:

- #1321 — REQ-2 (DIV-A, CLOSED/fixed): `interpolate_cdf` now averages forward +
  reversed `np_interp` → plateaus map to the midpoint (`_data.py:2843-2846`).
- #1320 — REQ-3 (DIV-B, CLOSED/fixed): `probit` now uses Acklam's inverse-normal-
  CDF (~1e-9) + `±5.199337582605575` clip (`:2855-2862`).
- #1322 — REQ-1 landmark fix (CLOSED): references via numpy `linspace` `i*step`
  +endpoint-pin and landmark `pos = (ref*100)/100*(n-1)` replicating
  `np.nanpercentile(X, references_*100)` (`:2694`,`:2702`,`:2795`).
- #1323 — REQ-4 (DIV-C): no `np.maximum.accumulate` monotonic repair of the
  per-feature landmarks (`:2707`). Minor; candidate fixable.
- #1324 — REQ-6: deterministic strided subsample, no `random_state`, default
  `100_000` not `10_000`, no `n_quantiles > subsample` error
  (`:2696-2700`,`:2774-2779`).
- #1325 — REQ-7: no `inverse_transform` (`norm.cdf` + reverse interp,
  `:2821`,`:2848`,`:2947`).
- #1326 — REQ-8: no `ignore_implicit_zeros` / sparse CSC path (`_sparse_fit`,
  `:2709-2752`).
- #1327 — REQ-9: no `quantile_transform` free function (`:2978-2988`).
- #1328 — REQ-10: no `copy` field, no `references_` / `n_quantiles_` /
  `n_features_in_` / `get_feature_names_out` surface (`OneToOneFeatureMixin`,
  `:2540`).
- #1329 — REQ-11: no `ferrolearn-python` `QuantileTransformer` binding
  (boundary CPython consumer absent).
- #1330 — REQ-12: fit/transform on `ndarray`/`num_traits`/hand-rolled A&S
  probit/`Vec`, not ferray (`ferray::stats` `norm.ppf`/`norm.cdf`,
  `ferray::random` subsample) (R-SUBSTRATE-1/2).
