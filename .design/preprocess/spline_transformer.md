# SplineTransformer

<!--
tier: 3-component
status: shipped-partial
baseline-commit: ac3c3a60
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_polynomial.py  # class SplineTransformer(TransformerMixin, BaseEstimator) (:580); _parameter_constraints {n_knots:[Interval(Integral,2,None,closed="left")], degree:[Interval(Integral,0,None,closed="left")], knots:[StrOptions({"uniform","quantile"}),"array-like"], extrapolation:[StrOptions({"error","constant","linear","continue","periodic"})], include_bias:["boolean"], order:[StrOptions({"C","F"})], sparse_output:["boolean"]} (:716-726); __init__(n_knots=5, degree=3, *, knots="uniform", extrapolation="constant", include_bias=True, order="C", sparse_output=False) (:728-744); _get_base_knot_positions(X, n_knots, knots, sample_weight) (:735-779): quantile percentiles=100*np.linspace(0,1,n_knots) then np.percentile(X, percentiles, axis=0) / _weighted_percentile (:747-762), uniform x_min/x_max=np.amin/amax(X[mask]) then np.linspace(x_min,x_max,n_knots,endpoint=True) (:764-779); fit(X, y=None, sample_weight=None) (:811-943): n_splines=n_knots+degree-1 non-periodic / n_knots-1 periodic (:873-878), EXTENDED knot vector via Eilers-Marx edge-spacing NOT repeated boundary (:896-923) dist_min=base[1]-base[0]/dist_max=base[-1]-base[-2], knots=r_[linspace(base[0]-degree*dist_min, base[0]-dist_min, degree), base_knots, linspace(base[-1]+dist_max, base[-1]+degree*dist_max, degree)] (:908-923), coef=np.eye(n_splines) + bsplines_=[BSpline.construct_fast(knots[:,i], coef, degree, extrapolate=...)] (:925-940, scipy BSpline design matrix), n_features_out_=n_out-n_features*(1-include_bias) (:942); transform(X) (:945+): bsplines_[i](X[:,i]) design matrix with extrapolation handling (constant=clamp to boundary basis / linear / continue / periodic / error); get_feature_names_out -> "{feat}_sp_{j}" for j in range(n_splines-1+include_bias) (:781-809); fitted attrs bsplines_, n_features_out_, n_features_in_.
ferrolearn-module: ferrolearn-preprocess/src/spline_transformer.rs
parity-ops: SplineTransformer
crosslink-issue: 1331
-->

## Summary

scikit-learn's `SplineTransformer` (`_polynomial.py:580`) expands each input
feature into a B-spline basis (`n_knots + degree - 1` non-periodic columns per
feature) evaluated against a knot vector built by `_get_base_knot_positions`
(`:735-779`) and then **extended on each side by `degree` knots at the EDGE
SPACING** (Eilers & Marx, NOT repeated boundary knots — the source explicitly
rejects the `np.tile`-the-boundary construction at `:898-906`): `dist_min =
base[1]-base[0]`, `dist_max = base[-1]-base[-2]`, `knots = r_[linspace(base[0] -
degree*dist_min, base[0]-dist_min, degree), base_knots, linspace(base[-1] +
dist_max, base[-1]+degree*dist_max, degree)]` (`:908-923`). The basis is the
**scipy `BSpline.construct_fast(knots, eye(n_splines), degree)` design matrix**
(`:925-940`), with `extrapolation` (default `"constant"`) governing out-of-range
behaviour and `include_bias` dropping one column per feature when `False`.

`ferrolearn-preprocess/src/spline_transformer.rs` now ships the **EXTENDED-knot
B-spline SHAPE** (the DIV-1 fix landed): `SplineTransformer<F> { n_knots, degree,
knots: KnotStrategy{Uniform, Quantile} }` (`new`, `Default` = `(5, 3, Uniform)`,
accessors `n_knots()` / `degree()` / `knot_strategy()`) fits into
`FittedSplineTransformer<F> { knot_vectors: Vec<Vec<F>>, degree, n_basis }`
(accessors `knot_vectors()` / `n_basis_per_feature()` / `n_output_features()`).
`fit` computes base interior knots (Uniform = `linspace(min, max, n_knots)`;
Quantile = a hand-rolled linear-interp percentile `pos = frac*(n-1)`) and then
builds the full knot vector by **EXTENDING the base knots with `degree` knots on
each side at the EDGE SPACING** (`dist_min = base[1]-base[0]`, `dist_max =
base[-1]-base[-2]`, `left/right = linspace(...)`) — sklearn's Eilers & Marx
construction (`:908-923`), NOT the clamped/repeated-boundary `np.tile` block the
source rejects. `transform` evaluates a **hand-rolled Cox-de Boor recursion**
(`fn bspline_basis`) over that extended vector, with the degree-0 right-endpoint
handling re-keyed to the base-interval right endpoint `knots[n_basis]`
(closed-right, matching scipy `BSpline.design_matrix`) → `n_basis` columns per
feature whose VALUES now match scipy over the base interval. Non-test consumer:
the crate re-export `pub use spline_transformer::{FittedSplineTransformer,
KnotStrategy, SplineTransformer};` (`ferrolearn-preprocess/src/lib.rs`, the
boundary public API). There is **no PyO3 binding** (`ferrolearn-python/` does not
reference `SplineTransformer`).

**Headline finding — DIV-1 is RESOLVED: the basis VALUES now match scipy over
the base interval.** A fixer rewrote the knot-vector CONSTRUCTION from CLAMPED
(boundary multiplicity `degree+1`, the `np.tile` block sklearn rejects at
`:898-906`) to sklearn's EXTENDED edge-spacing knots (`:908-923`) and re-keyed the
`fn bspline_basis` right-endpoint handling to the base-interval right endpoint
`knots[n_basis]` (closed-right, matching scipy `BSpline.design_matrix`).
Uniform-knot basis VALUE parity is now VERIFIED across degree∈{1,2,3},
multi-feature, dense interior, and both base endpoints — 11 passing value tests in
`tests/divergence_spline_transformer.rs` (`divergence_spline_basis_value_asymmetric_fixture`
+ `probe_a..f`) against the live sklearn 1.5.2 oracle at tol 1e-6
(`x_sp_0(0.0)` now `0.166667`, matching sklearn, where the old clamped basis gave
`1.0`). **Historical context:** the divergence WAS that ferrolearn built clamped
knots and evaluated Cox-de Boor while sklearn extended at the edge spacing and
evaluated scipy `BSpline` — that is now fixed. This is a **mostly-NOT-STARTED**
unit: **3 SHIPPED** (REQ-1 structural dims + B-spline properties; REQ-2 basis
VALUE parity [was DIV-1 headline, now SHIPPED]; REQ-6 scoped error contracts) /
**8 NOT-STARTED** (REQ-3 `extrapolation`, REQ-4 `include_bias`, REQ-5
`np.percentile`-exact quantile knots, REQ-7 `sparse_output`+`order`, REQ-8
`sample_weight`, REQ-9 `get_feature_names_out` + fitted-attr surface, REQ-10 PyO3,
REQ-11 ferray).

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — DIMENSIONS + partition-of-unity HOLD; on the SYMMETRIC n_knots=5/degree=3
# uniform fixture the clamped and extended bases COINCIDENTALLY agree (both 0.166667,
# 0.666667, 0.166667), so this fixture does NOT expose DIV-1:
python3 -c "import numpy as np; from sklearn.preprocessing import SplineTransformer; \
X=np.array([[0.],[0.25],[0.5],[0.75],[1.]]); \
print(np.round(SplineTransformer(n_knots=5, degree=3, knots='uniform').fit_transform(X),6).tolist())"
# -> [[0.166667, 0.666667, 0.166667, 0,0,0,0], [0, 0.166667, 0.666667, 0.166667, 0,0,0],
#     [0,0, 0.166667, 0.666667, 0.166667, 0,0], [0,0,0, 0.166667, 0.666667, 0.166667, 0],
#     [0,0,0,0, 0.166667, 0.666667, 0.166667]]
#    7 columns (5+3-1), each row sums to 1.0 (partition of unity), all >= 0.
#    ferrolearn's clamped Cox-de Boor REPRODUCES this exact matrix on THIS fixture (the
#    in-module test_spline_output_dimensions / _partition_of_unity / _non_negative pin
#    the dims + properties), but NOT the values on the asymmetric fixture below.

# REQ-2 (DIV-1, HEADLINE) — on the ASYMMETRIC n_knots=4/degree=3 fixture the EXTENDED-knot
# basis (sklearn) and the CLAMPED-knot basis (ferrolearn) DIVERGE, most visibly at x=0/x=1:
python3 -c "import numpy as np; from sklearn.preprocessing import SplineTransformer; \
X=np.array([[0.],[0.3],[0.6],[1.0]]); st=SplineTransformer(n_knots=4, degree=3).fit(X); \
print('extended knots:', np.round(st.bsplines_[0].t,6).tolist()); \
print('sklearn basis:', np.round(st.transform(X),6).tolist())"
# -> extended knots: [-1.0, -0.666667, -0.333333, 0.0, 0.333333, 0.666667, 1.0, 1.333333, 1.666667, 2.0]
# -> sklearn basis row0 (x=0): [0.166667, 0.666667, 0.166667, 0.0, 0.0, 0.0]
python3 -c "import numpy as np; from scipy.interpolate import BSpline; \
base=np.linspace(0,1,4); deg=3; \
clamped=np.r_[[base[0]]*deg, base, [base[-1]]*deg]; \
n=len(clamped)-deg-1; bs=BSpline.construct_fast(clamped, np.eye(n), deg, extrapolate=False); \
X=np.array([0.,0.3,0.6,1.0]); print('CLAMPED-knot (ferrolearn-style) knots:', clamped.tolist()); \
print('CLAMPED basis row0 (x=0):', np.round(np.nan_to_num(bs(X).T).T,6).tolist()[0])"
# -> CLAMPED knots: [0,0,0, 0.0, 0.333333, 0.666667, 1.0, 1.0,1.0,1.0]
# -> CLAMPED basis row0 (x=0): [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#    sklearn x=0 -> [0.166667, 0.666667, 0.166667, ...]; ferrolearn-style clamped x=0 -> [1.0, 0,...]. DIV-1.

# REQ-3 — extrapolation default is "constant" (clamp to boundary basis); 'error' raises:
python3 -c "import numpy as np; from sklearn.preprocessing import SplineTransformer; \
X=np.array([[0.],[0.5],[1.]]); st=SplineTransformer(n_knots=3, degree=2, extrapolation='constant').fit(X); \
print('constant extrap @ 2.0:', np.round(st.transform([[2.0]]),6).tolist()); \
import traceback; \
exec('try:\n SplineTransformer(n_knots=3, degree=2, extrapolation=\"error\").fit(X).transform([[2.0]])\n print(\"no err\")\nexcept ValueError as e:\n print(\"error extrap raises:\", str(e)[:50])')"
# -> constant extrap @ 2.0: a clamped boundary basis (NOT zeros);  error extrap raises: X contains values beyond the limits ...
#    ferrolearn has NO extrapolation param; its Cox-de Boor returns whatever the clamped
#    recursion yields out of range (no 'constant'/'linear'/'continue'/'periodic'/'error' switch). NOT-STARTED.

# REQ-4 — include_bias=False drops ONE column per feature (n_splines-1):
python3 -c "import numpy as np; from sklearn.preprocessing import SplineTransformer; \
X=np.array([[0.],[0.25],[0.5],[0.75],[1.]]); \
print('include_bias=True  ncols:', SplineTransformer(n_knots=5,degree=3,include_bias=True).fit_transform(X).shape[1]); \
print('include_bias=False ncols:', SplineTransformer(n_knots=5,degree=3,include_bias=False).fit_transform(X).shape[1])"
# -> include_bias=True ncols: 7 ; include_bias=False ncols: 6
#    ferrolearn always emits n_knots+degree-1 columns; no include_bias field, no column drop. NOT-STARTED.

# REQ-9 — get_feature_names_out emits "{feat}_sp_{j}"; fitted attrs n_features_out_, bsplines_:
python3 -c "import numpy as np; from sklearn.preprocessing import SplineTransformer; \
X=np.array([[0.],[0.25],[0.5],[0.75],[1.]]); st=SplineTransformer(n_knots=5,degree=3).fit(X); \
print('names:', st.get_feature_names_out(['x']).tolist(), 'n_features_out_:', st.n_features_out_)"
# -> names: ['x_sp_0','x_sp_1','x_sp_2','x_sp_3','x_sp_4','x_sp_5','x_sp_6'] n_features_out_: 7
#    ferrolearn exposes knot_vectors()/n_basis_per_feature()/n_output_features() only; no names, no bsplines_. NOT-STARTED.
```

## Requirements

- REQ-1: **Output DIMENSIONS + B-spline structural PROPERTIES** (scoped). For each
  feature, emit `n_basis = n_knots + degree - 1` B-spline basis columns whose
  values over the data range satisfy **partition of unity** (each row sums to 1)
  and **non-negativity** (every basis value `>= 0`). Mirrors sklearn `n_splines =
  n_knots + self.degree - 1` (`_polynomial.py:875`) and the standard B-spline
  basis algebra. ferrolearn's `fit` sets `let n_basis = self.n_knots + self.degree
  - 1;` and `transform` fills `n_features * n_basis` columns via `fn bspline_basis`
  (Cox-de Boor). Supports `f32`/`f64`. **Scope (R-HONEST): these are STRUCTURAL
  PROPERTIES, NOT sklearn VALUE parity — partition-of-unity / non-negativity / the
  column count hold for ANY valid B-spline basis regardless of the knot
  construction. The per-cell VALUE parity against scipy `BSpline` is REQ-2, and
  diverges (DIV-1).**

- REQ-2: **B-spline basis VALUE parity** (DIV-1, HEADLINE) — sklearn builds the
  knot vector by EXTENDING the base knots with `degree` knots on each side at the
  EDGE SPACING (Eilers & Marx) — `dist_min = base_knots[1] - base_knots[0]`,
  `dist_max = base_knots[-1] - base_knots[-2]`, `knots = np.r_[np.linspace(
  base_knots[0] - degree*dist_min, base_knots[0] - dist_min, num=degree),
  base_knots, np.linspace(base_knots[-1] + dist_max, base_knots[-1] +
  degree*dist_max, num=degree)]` (`_polynomial.py:908-923`) — then evaluates the
  scipy `BSpline.construct_fast(knots, np.eye(n_splines), degree)` design matrix
  (`:925-940`). The source EXPLICITLY rejects the repeated-boundary (clamped)
  construction ferrolearn uses, commenting it out at `:898-906` (`# knots =
  np.r_[np.tile(base_knots.min...), base_knots, np.tile(base_knots.max...)]`).
  ferrolearn instead builds a CLAMPED knot vector — `for _ in 0..self.degree {
  full_knots.push(min_val); } ... for _ in 0..self.degree { full_knots.push(
  max_val); }` (boundary multiplicity `degree+1`) — and evaluates a hand-rolled
  `fn bspline_basis` Cox-de Boor recursion. The bases COINCIDE on symmetric uniform
  fixtures (Probe REQ-1) but DIVERGE off-center — Probe REQ-2 (DIV-1): on
  `[[0],[0.3],[0.6],[1.0]]` (n_knots=4, degree=3) sklearn `x_sp_0(0.0) = 0.166667`
  where the clamped basis gives `1.0`.

- REQ-3: **`extrapolation` parameter** (`'error'`/`'constant'`/`'linear'`/
  `'continue'`/`'periodic'`, default `'constant'`) — sklearn's `extrapolation`
  (`_parameter_constraints` `StrOptions({"error","constant","linear","continue",
  "periodic"})`, `:719-721`; default `:734`) governs values outside the base
  interval: `'error'` raises a `ValueError`, `'constant'` clamps to the boundary
  spline value, `'linear'` linearly extrapolates, `'continue'` passes scipy
  `extrapolate=True`, `'periodic'` uses periodic splines (with `n_splines = n_knots
  - 1` and a wrap-around coef, `:877-878`,`:888-895`,`:930-931`). ferrolearn has
  **no `extrapolation` field**; its Cox-de Boor returns whatever the clamped
  recursion produces out of range, with no constant/linear/continue/periodic/error
  switch (Probe REQ-3).

- REQ-4: **`include_bias` parameter** — sklearn `include_bias=True` (default,
  `:739`) keeps all `n_splines` columns; `include_bias=False` drops the last
  spline element inside each feature's data range, so `n_features_out_ = n_out -
  n_features * (1 - include_bias)` (`:942`) and `get_feature_names_out` iterates
  `range(n_splines - 1 + include_bias)` (`:806`). ferrolearn always emits
  `n_basis = n_knots + degree - 1` columns (`transform` writes `self.n_basis` per
  feature); there is **no `include_bias` field and no column drop** (Probe REQ-4:
  sklearn 7 vs 6 columns).

- REQ-5: **`np.percentile`-exact quantile base knots** — sklearn's `'quantile'`
  branch computes `percentiles = 100 * np.linspace(0, 1, n_knots)` then `knots =
  np.percentile(X, percentiles, axis=0)` (`_polynomial.py:747-753`), i.e. numpy's
  default linear-interpolation percentile over the FULL column; the `'uniform'`
  branch is `np.linspace(np.amin(X), np.amax(X), n_knots)` (`:771-777`).
  ferrolearn's `KnotStrategy::Quantile` arm uses a hand-rolled `let frac = i /
  (n_knots-1); let pos = frac * (n-1); col_vals[lo]*(1-f) + col_vals[hi]*f`. While
  this is morally the same linear-index percentile, it has **not been pinned
  against `np.percentile` to ULP**, and the `'uniform'` arm matches `np.linspace`
  only structurally. Folded with REQ-2: the base-knot computation feeds the
  (already-divergent) clamped knot vector, so quantile VALUE parity is moot until
  DIV-1 is resolved.

- REQ-6: **Error / parameter contracts** (scoped, with flagged DIVs) — ferrolearn
  `fit` returns `InsufficientSamples` when `n_samples < 2`, `InvalidParameter`
  when `n_knots < 2`, and `InvalidParameter` when `degree == 0`; `transform`
  returns `ShapeMismatch` on a column-count mismatch; the unfitted `transform`
  returns `InvalidParameter`. **FLAG (candidate DIVs the critic may pin):**
  sklearn's `_parameter_constraints` is `n_knots: Interval(Integral, 2, None,
  closed="left")` (`:717`) — ferrolearn's `n_knots < 2` rejection MATCHES; but
  `degree: Interval(Integral, 0, None, closed="left")` (`:718`) — **`degree == 0`
  is VALID in sklearn** (piecewise-constant splines) yet ferrolearn rejects it; and
  sklearn does NOT require `n_samples >= 2` (it fits on a single sample). sklearn
  raises only the cross-constraint `degree=0 and include_bias=False` error
  (`:326-330`) and an `n_knots >= degree+1`-style completeness check.

- REQ-7: **`sparse_output` + `order` parameters** — sklearn exposes `sparse_output`
  (default `False`, `:726`,`:743`) returning a `scipy.sparse` CSR design matrix via
  `BSpline.design_matrix(..., extrapolate=...)`, and `order` (`{"C","F"}`, default
  `"C"`, `:725`,`:742`) controlling the dense output memory layout. ferrolearn
  emits only a dense row-major `ndarray::Array2<F>`; there is no `sparse_output`
  and no `order` field.

- REQ-8: **`sample_weight`** — sklearn's `fit(X, y=None, sample_weight=None)`
  (`:811`) threads weights into base-knot computation: the `'quantile'` arm uses
  `_weighted_percentile(X, sample_weight, percentile)` (`:756-761`) and the
  `'uniform'` arm disregards zero-weight observations via `mask = sample_weight > 0`
  before `np.amin`/`np.amax` (`:766-770`). ferrolearn's `Fit::fit(x, _y: &())`
  takes **no sample weights** (the `_y` is unit) and computes unweighted
  min/max/percentiles.

- REQ-9: **`get_feature_names_out` + `bsplines_` / `n_features_out_` fitted-attr
  surface** — sklearn emits `f"{input_features[i]}_sp_{j}"` for `j in
  range(n_splines - 1 + include_bias)` (`:806-808`) and exposes the fitted
  attributes `bsplines_` (the per-feature scipy `BSpline` objects, `:940`),
  `n_features_out_` (`:942`), and `n_features_in_` / `feature_names_in_` (via
  `BaseEstimator`). ferrolearn's `FittedSplineTransformer<F>` exposes only
  `knot_vectors()` / `n_basis_per_feature()` / `n_output_features()`; there is **no
  `get_feature_names_out`, no `{feat}_sp_{j}` naming, no `bsplines_` / `n_features_in_`
  surface** (Probe REQ-9).

- REQ-10: **PyO3 binding** — `import ferrolearn` exposing a registered
  `SplineTransformer` marshalling `fit` / `transform`, the project boundary CPython
  consumer. Absent (no `ferrolearn-python` reference to `SplineTransformer`).

- REQ-11: **ferray substrate** — compute the knot vectors and the B-spline design
  matrix over `ferray-core` arrays / `ferray::interpolate` (a faithful `BSpline`)
  rather than `ndarray::Array2` + `num_traits::Float` + a hand-rolled Cox-de Boor
  recursion + `Vec<Vec<F>>` knot bookkeeping (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `SplineTransformer::<f64>::new(5, 3, Uniform).fit_transform(
  array![[0.],[0.25],[0.5],[0.75],[1.]])` yields a `5 x 7` matrix; every row sums
  to `1.0` within `1e-10` (partition of unity) and every entry is `>= -1e-10`
  (non-negativity). Pinned by `test_spline_output_dimensions` (ncols == 7, nrows ==
  5), `test_spline_partition_of_unity`, `test_spline_non_negative`,
  `test_spline_multi_feature` (2 features → 8 cols), `test_spline_degree1`.
  **Scope: STRUCTURAL — these properties hold for any valid B-spline basis; this
  is NOT sklearn value parity (REQ-2).**

- AC-2 (REQ-2): on `[[0],[0.3],[0.6],[1.0]]` (n_knots=4, degree=3), sklearn's
  EXTENDED-knot scipy `BSpline` gives `x_sp_0(0.0) = 0.166667` (extended knots
  `[-1.0, -0.667, -0.333, 0.0, 0.333, 0.667, 1.0, 1.333, 1.667, 2.0]`); ferrolearn
  now reproduces this — its EXTENDED edge-spacing knot construction (`:908-923`)
  + the `knots[n_basis]` right-endpoint re-keying (matching scipy `BSpline`
  `:925-940`) yield the SAME value (previously the old clamped-knot Cox-de Boor
  gave `1.0` on clamped knots `[0,0,0, 0.0, 0.333, 0.667, 1.0, 1.0,1.0,1.0]`).
  PINNED by `divergence_spline_basis_value_asymmetric_fixture` (the full 4x6
  matrix, tol 1e-6) plus `probe_a..f` covering degree∈{1,2,3}, multi-feature,
  dense interior, and both base endpoints in
  `tests/divergence_spline_transformer.rs`.

- AC-3 (REQ-3): `SplineTransformer(extrapolation='constant').transform([[2.0]])`
  on a `[0,1]`-trained transformer returns the clamped boundary basis (NOT zeros);
  `extrapolation='error'` raises `ValueError` (Probe REQ-3). ferrolearn has no
  `extrapolation` switch — its clamped Cox-de Boor neither clamps-to-constant nor
  errors out of range.

- AC-4 (REQ-4): `SplineTransformer(n_knots=5, degree=3, include_bias=False)`
  yields 6 columns per feature (`n_splines - 1`) where `include_bias=True` yields 7
  (Probe REQ-4); ferrolearn always emits 7 with no `include_bias` field.

- AC-5 (REQ-5): on a `'quantile'`-knot fixture, sklearn's base knots equal
  `np.percentile(X, 100*np.linspace(0,1,n_knots), axis=0)` (`:747-753`) to numpy
  ULP; ferrolearn's `pos = frac*(n-1)` linear interp is not pinned against
  `np.percentile`, and feeds the (divergent) clamped vector so the resulting basis
  diverges regardless (REQ-2).

- AC-6 (REQ-6): ferrolearn's contracts (`n_samples < 2` → `InsufficientSamples`,
  `n_knots < 2` → `InvalidParameter`, `degree == 0` → `InvalidParameter`, shape
  mismatch → `ShapeMismatch`, unfitted → `InvalidParameter`) are pinned by
  `test_spline_insufficient_samples_error`, `test_spline_too_few_knots_error`,
  `test_spline_zero_degree_error`, `test_spline_shape_mismatch_error`,
  `test_spline_unfitted_error`. **FLAG: `degree == 0` is VALID in sklearn
  (`Interval(Integral, 0, None)`, `:718`) but ferrolearn rejects it; sklearn does
  not require `n_samples >= 2`.**

- AC-7 (REQ-7): `SplineTransformer(sparse_output=True).transform(X)` returns a
  `scipy.sparse` CSR matrix and `order='F'` returns a Fortran-ordered dense array;
  ferrolearn emits only a dense row-major `Array2<F>`, no `sparse_output`/`order`.

- AC-8 (REQ-8): `SplineTransformer(knots='quantile').fit(X,
  sample_weight=w)` weights the base-knot percentiles (`_weighted_percentile`,
  `:756-761`); ferrolearn's `fit(x, &())` takes no weights.

- AC-9 (REQ-9): a fitted handle exposes `get_feature_names_out(['x'])` →
  `['x_sp_0', ..., 'x_sp_6']`, `n_features_out_ = 7`, and `bsplines_` (Probe REQ-9);
  ferrolearn exposes only `knot_vectors()` / `n_basis_per_feature()` /
  `n_output_features()`.

- AC-10 (REQ-10): `python3 -c "import ferrolearn; ..."` resolves a registered
  `SplineTransformer`; `.fit(X).transform(X)` matches the sklearn oracle.

- AC-11 (REQ-11): the knot-vector + design-matrix path computes on `ferray-core`
  arrays + a `ferray` B-spline rather than `ndarray` + hand-rolled Cox-de Boor.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (output dimensions + B-spline structural properties: partition-of-unity, non-negative) | SHIPPED (scoped) | `Fit::fit for SplineTransformer` sets `let n_basis = self.n_knots + self.degree - 1;` (mirrors sklearn `n_splines = n_knots + self.degree - 1`, `_polynomial.py:875`); `Transform::transform for FittedSplineTransformer` fills `n_features * self.n_basis` columns from `fn bspline_basis` (Cox-de Boor over the clamped knot vector). The structural B-spline properties (partition of unity, non-negativity, column count) HOLD. Non-test consumer: crate re-export `pub use spline_transformer::{FittedSplineTransformer, KnotStrategy, SplineTransformer};` (`ferrolearn-preprocess/src/lib.rs`, boundary public API, grandfathered S5/R-DEFER-1). Verification: `cargo test -p ferrolearn-preprocess spline` (`test_spline_output_dimensions` → 5x7, `test_spline_partition_of_unity` → row sums == 1.0 ± 1e-10, `test_spline_non_negative` → all >= -1e-10, `test_spline_multi_feature` → 2 feats × 4 = 8 cols, `test_spline_degree1`). **SCOPE (R-HONEST): these are STRUCTURAL PROPERTIES, NOT sklearn VALUE parity — they hold for ANY valid B-spline basis irrespective of the clamped-vs-extended knot construction. Per-cell value parity vs scipy `BSpline` is REQ-2 and DIVERGES (DIV-1).** |
| REQ-2 (B-spline basis VALUE parity: extended-spacing knots + scipy `BSpline`; was DIV-1, HEADLINE) | SHIPPED | `Fit::fit for SplineTransformer` now builds the EXTENDED edge-spacing knot vector — `dist_min = base[1] - base[0]; dist_max = base[nb-1] - base[nb-2];` then `left = linspace(base[0] - degree*dist_min, base[0] - dist_min, degree)`, `right = linspace(base[nb-1] + dist_max, base[nb-1] + degree*dist_max, degree)`, `full_knots = [left, base, right]` — mirroring sklearn `_polynomial.py:908-923` (Eilers & Marx; the source rejects the clamped `np.tile` construction at `:898-906`). `fn bspline_basis` re-keys its degree-0 right-endpoint handling to the base-interval right endpoint `base_right = knots[n_basis]` (closed-right: `if x >= base_right { activate the last non-degenerate interval with `knots[i+1] <= base_right` }`), matching scipy `BSpline.design_matrix` (`:925-940`). The bases now reproduce the scipy design matrix over the base interval. Non-test consumer: crate re-export `pub use spline_transformer::{FittedSplineTransformer, KnotStrategy, SplineTransformer};` (`ferrolearn-preprocess/src/lib.rs`, boundary public API). Live oracle (was Probe REQ-2): on `[[0],[0.3],[0.6],[1.0]]` (n_knots=4, degree=3) sklearn `x_sp_0(0.0) = 0.166667` — ferrolearn now MATCHES (the clamped basis formerly gave `1.0`). Verification: 11 value tests in `tests/divergence_spline_transformer.rs` (`divergence_spline_basis_value_asymmetric_fixture` + `probe_a..f`) pin VALUE parity vs the live sklearn 1.5.2 oracle at tol 1e-6 across degree∈{1,2,3}, multi-feature, dense interior, and BOTH base endpoints (x==min, x==max). Was DIV-1 / blocker #1332 (now RESOLVED). |
| REQ-3 (`extrapolation` param: 'constant' default + 'error'/'linear'/'continue'/'periodic') | NOT-STARTED | open prereq blocker #1333. `SplineTransformer<F>` has no `extrapolation` field; out-of-range values fall through the clamped Cox-de Boor with no switch. sklearn's `extrapolation` (`StrOptions({"error","constant","linear","continue","periodic"})`, `_polynomial.py:719-721`, default `"constant"`, `:734`) raises on `'error'`, clamps to the boundary spline on `'constant'`, linearly extrapolates on `'linear'`, passes scipy `extrapolate=True` on `'continue'`, and builds periodic splines (`n_splines = n_knots - 1`, wrap-around coef, `:877-878`,`:888-895`,`:930-931`) on `'periodic'` (Probe REQ-3). |
| REQ-4 (`include_bias` param: drop one column when False) | NOT-STARTED | open prereq blocker #1334. `transform` always writes `self.n_basis = n_knots + degree - 1` columns per feature; `SplineTransformer<F>` has no `include_bias` field and no column-drop path. sklearn drops the last spline element when `include_bias=False`: `n_features_out_ = n_out - n_features * (1 - include_bias)` (`_polynomial.py:942`) and `get_feature_names_out` iterates `range(n_splines - 1 + include_bias)` (`:806`) (Probe REQ-4: 7 vs 6 columns). |
| REQ-5 (`np.percentile`-exact quantile base knots) | NOT-STARTED | open prereq blocker #1335. `KnotStrategy::Quantile` computes `let pos = frac * (n-1); col_vals[lo]*(1-f) + col_vals[hi]*f` (`frac = i/(n_knots-1)`), a hand-rolled linear-index percentile NOT pinned against `np.percentile` to ULP; the `Uniform` arm matches `np.linspace(min, max, n_knots)` only structurally. sklearn uses `percentiles = 100*np.linspace(0,1,n_knots)` then `np.percentile(X, percentiles, axis=0)` (`_polynomial.py:747-753`) / `np.linspace(np.amin(X), np.amax(X), n_knots)` (`:771-777`). The base knots feed the CLAMPED vector (REQ-2/DIV-1), so quantile VALUE parity is moot until DIV-1 is resolved. |
| REQ-6 (error / parameter contracts) | SHIPPED (#1336) | `Fit::fit for SplineTransformer` returns `Err(FerroError::InsufficientSamples { required: 2, .. })` when `n_samples < 2`, and `Err(InvalidParameter { name: "n_knots", .. })` when `self.n_knots < 2`; `Transform::transform for FittedSplineTransformer` returns `Err(ShapeMismatch)` on `x.ncols() != n_features`; unfitted `transform` returns `Err(InvalidParameter)`. FIXED #1336: removed the spurious `degree == 0` rejection — sklearn `_parameter_constraints` `degree: Interval(Integral, 0, None, closed="left")` (`_polynomial.py:718`) ALLOWS `degree=0` (piecewise-constant B-spline); `bspline_basis` already supports degree 0 (the `for d in 1..=degree` build-up loop is empty → returns the degree-0 indicator basis), and the linspace `num==0` edge (no edge-extension knots) was handled. Live oracle (R-CHAR-3): `SplineTransformer(n_knots=3, degree=0).fit([[0],[1],[2],[3],[4]]).transform([[0.5],[2.5]])` → `(2,2)` `[[1,0],[0,1]]`; ferrolearn matches (guard `spline_degree_zero_allowed_matches_sklearn`). CORRECTED the stale flag: the `n_samples >= 2` requirement is NOT a divergence — sklearn ALSO rejects `n_samples=1` (`check_array(ensure_min_samples=2)`, `:830`; live `ValueError: Found array with 1 sample(s) ... minimum of 2 required`), so ferrolearn's `n_samples < 2` check MATCHES. (The sklearn cross-constraint `degree=0 and include_bias=False` is N/A — ferrolearn's `SplineTransformer` exposes no `include_bias` param; that param surface is separately tracked.) |
| REQ-7 (`sparse_output` + `order` params) | NOT-STARTED | open prereq blocker #1337. `transform` returns only a dense row-major `ndarray::Array2<F>` (`Array2::zeros((n_samples, n_out))`); `SplineTransformer<F>` has no `sparse_output` / `order` field. sklearn returns a `scipy.sparse` CSR design matrix when `sparse_output=True` (`:726`,`:743`) and honours `order ∈ {"C","F"}` for the dense layout (`:725`,`:742`). |
| REQ-8 (`sample_weight`) | NOT-STARTED | open prereq blocker #1338. `Fit::fit(&self, x, _y: &())` takes no sample weights (the supervised slot is unit) and computes unweighted min/max/percentiles. sklearn's `fit(X, y=None, sample_weight=None)` (`_polynomial.py:811`) threads weights into base-knot computation via `_weighted_percentile(X, sample_weight, percentile)` for `'quantile'` (`:756-761`) and a `mask = sample_weight > 0` before `np.amin`/`np.amax` for `'uniform'` (`:766-770`). |
| REQ-9 (`get_feature_names_out` + `bsplines_` / `n_features_out_` fitted attrs) | NOT-STARTED | open prereq blocker #1339. `FittedSplineTransformer<F>` exposes only `knot_vectors()` / `n_basis_per_feature()` / `n_output_features()`; there is no `get_feature_names_out`, no `{feat}_sp_{j}` naming, no `bsplines_` / `n_features_out_` / `n_features_in_` surface. sklearn emits `f"{input_features[i]}_sp_{j}"` for `j in range(n_splines - 1 + include_bias)` (`_polynomial.py:806-808`) and exposes `bsplines_` (`:940`), `n_features_out_` (`:942`), `n_features_in_` / `feature_names_in_` (Probe REQ-9). |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #1340. No `ferrolearn-python` registration of `SplineTransformer` (grep across `ferrolearn-python/` finds no `SplineTransformer` / `spline_transformer`); the only non-test consumer is the crate re-export (`lib.rs`). The boundary CPython `import ferrolearn` transformer surface is absent. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #1341. The fit/transform path uses `ndarray::Array2` (`x.column(j)`, `Array2::zeros`, `out[[i, col_offset + k]]`) + `num_traits::Float`, a hand-rolled `fn bspline_basis` Cox-de Boor recursion, and `Vec<Vec<F>>` knot bookkeeping — not `ferray-core` arrays / a `ferray` B-spline (`ferray::interpolate`) (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `spline_transformer.rs` exposes `KnotStrategy {
Uniform, Quantile }`, the unfitted `SplineTransformer<F> { n_knots: usize, degree:
usize, knots: KnotStrategy, _marker: PhantomData<F> }` (`new(n_knots, degree,
knots)`, `Default` = `(5, 3, Uniform)`, accessors `n_knots()` / `degree()` /
`knot_strategy()`), and the fitted `FittedSplineTransformer<F> { knot_vectors:
Vec<Vec<F>>, degree: usize, n_basis: usize }` (accessors `knot_vectors()` /
`n_basis_per_feature()` / `n_output_features()`). `Fit::fit` validates `n_samples
< 2` → `InsufficientSamples`, `n_knots < 2` → `InvalidParameter`, `degree == 0` →
`InvalidParameter`, sets `n_basis = n_knots + degree - 1`, and per column sorts
the values, takes `min`/`max`, computes base interior knots (Uniform =
`linspace(min, max, n_knots)`; Quantile = `pos = frac*(n-1)` linear interp), then
builds the **full knot vector by EXTENDING the base knots with `degree` knots on
each side at the EDGE SPACING** (`dist_min = base[1]-base[0]`, `dist_max =
base[-1]-base[-2]`, `left = linspace(base[0]-degree*dist_min, base[0]-dist_min,
degree)`, `right = linspace(base[-1]+dist_max, base[-1]+degree*dist_max, degree)`,
`full_knots = [left, base, right]`) — sklearn's Eilers & Marx construction
(`:908-923`), NOT the clamped/repeated-boundary `np.tile` block the source rejects
(`:898-906`). `Transform::transform` checks the column count and, per value, calls
the private `fn bspline_basis(x, knots, degree, n_basis)` — a hand-rolled Cox-de
Boor recursion (degree-0 half-open indicators whose right-endpoint handling is
re-keyed to the base-interval right endpoint `knots[n_basis]`, closed-right to
match scipy `BSpline.design_matrix`, then the standard up-degree recurrence) —
writing `n_basis` columns per feature into a dense `Array2<F>`. The unfitted `Transform` returns `InvalidParameter`;
`FitTransform` chains `fit` then `transform`. The crate re-exports all three types
(`lib.rs`, `pub use spline_transformer::{FittedSplineTransformer, KnotStrategy,
SplineTransformer}`); there is no PyO3 binding.

**sklearn (target contract).** `SplineTransformer(TransformerMixin,
BaseEstimator)` (`_polynomial.py:580`) takes `__init__(n_knots=5, degree=3, *,
knots="uniform", extrapolation="constant", include_bias=True, order="C",
sparse_output=False)` (`:728-744`) under `_parameter_constraints` (`n_knots`
`Interval(Integral, 2, None)`, `degree` `Interval(Integral, 0, None)`, `knots`
`{"uniform","quantile"} | array-like`, `extrapolation`
`{"error","constant","linear","continue","periodic"}`, `include_bias`/`sparse_output`
boolean, `order` `{"C","F"}`, `:716-726`). `_get_base_knot_positions` (`:735-779`)
computes base knots: `'quantile'` → `np.percentile(X, 100*np.linspace(0,1,n_knots),
axis=0)` (`:747-753`, or `_weighted_percentile` with `sample_weight`); `'uniform'`
→ `np.linspace(np.amin(X[mask]), np.amax(X[mask]), n_knots)` (`:764-779`). `fit`
(`:811-943`) sets `n_splines = n_knots + degree - 1` (non-periodic, `:875`) and
EXTENDS the base knots by `degree` knots on each side at the **edge spacing**
(Eilers & Marx, `:908-923`) — the source comments out and rejects the
repeated-boundary `np.tile` construction (`:898-906`) — then sets `coef =
np.eye(n_splines)` and `bsplines_ = [BSpline.construct_fast(knots[:,i], coef,
degree, extrapolate=...)]` (`:925-940`, the scipy B-spline design matrix), and
`n_features_out_ = n_out - n_features*(1 - include_bias)` (`:942`). `transform`
(`:945+`) evaluates `bsplines_[i](X[:,i])` (or `BSpline.design_matrix` for
`sparse_output`) with `extrapolation` handling out-of-range values.
`get_feature_names_out` (`:781-809`) emits `f"{feat}_sp_{j}"`.

**The structural gap (DIV-1 now RESOLVED).** ferrolearn matches sklearn on the
*output dimensions* and the *structural B-spline properties* (partition of unity,
non-negativity — REQ-1, scoped), on the *basis VALUES over the base interval*
(REQ-2 — DIV-1, the headline, now FIXED), and on the *scoped error contracts*
(REQ-6, with the `degree == 0` and `n_samples >= 2` parameter DIVs flagged).
**Historical context (DIV-1, what the divergence WAS):** ferrolearn formerly
repeated the boundary knot `degree` times (CLAMPED, multiplicity `degree+1`) and
ran a hand-rolled Cox-de Boor, so the basis VALUES diverged off-center from
sklearn's edge-spacing-extended scipy `BSpline`. The fixer rewrote the knot
construction to sklearn's EXTENDED edge spacing (`:908-923`) and re-keyed the
`fn bspline_basis` right-endpoint handling to `knots[n_basis]` (closed-right,
matching scipy `BSpline.design_matrix`, `:925-940`), so the basis now matches
scipy over the base interval (verified across degree∈{1,2,3}, multi-feature, both
endpoints). The remaining gaps: no `extrapolation` switch (REQ-3); no
`include_bias` column drop (REQ-4); the quantile knots are not `np.percentile`-
pinned (REQ-5); no `sparse_output` / `order` (REQ-7); no `sample_weight` (REQ-8);
no `get_feature_names_out` / `bsplines_` / `n_features_out_` surface (REQ-9); no
PyO3 binding (REQ-10); and the wrong (non-ferray, hand-rolled-Cox-de-Boor)
substrate (REQ-11). This is a **mostly-NOT-STARTED** unit (3 SHIPPED / 8
NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1 scoped, REQ-2 value parity, REQ-6 scoped):

```bash
# Crate gauntlet — REQ-1 (dimensions + partition-of-unity + non-negativity), REQ-2 (basis VALUE parity), REQ-6 (error contracts):
cargo test -p ferrolearn-preprocess spline   # incl. test_spline_output_dimensions,
                                              #       test_spline_partition_of_unity,
                                              #       test_spline_non_negative,
                                              #       test_spline_quantile_knots,
                                              #       test_spline_multi_feature,
                                              #       test_spline_fit_transform,
                                              #       test_spline_degree1,
                                              #       test_spline_default,
                                              #       test_spline_insufficient_samples_error,
                                              #       test_spline_too_few_knots_error,
                                              #       test_spline_zero_degree_error,
                                              #       test_spline_shape_mismatch_error,
                                              #       test_spline_unfitted_error
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# Oracle (Probe REQ-1) — symmetric uniform fixture; structural properties hold, values
# COINCIDENTALLY match (this fixture does NOT expose DIV-1):
python3 -c "import numpy as np; from sklearn.preprocessing import SplineTransformer; \
X=np.array([[0.],[0.25],[0.5],[0.75],[1.]]); \
print(np.round(SplineTransformer(n_knots=5, degree=3, knots='uniform').fit_transform(X),6).tolist())"
#   -> 7 columns, each row sums to 1.0, all >= 0 (ferrolearn reproduces this matrix on THIS fixture)

# REQ-2 value-parity test file (11 live-oracle value tests, was DIV-1/#1332, now GREEN):
cargo test -p ferrolearn-preprocess --test divergence_spline_transformer
#   -> divergence_spline_basis_value_asymmetric_fixture + probe_a..f PASS at tol 1e-6
#      (degree in {1,2,3}, multi-feature, dense interior, both base endpoints)

# Oracle (was Probe REQ-2 / DIV-1, HEADLINE) — asymmetric fixture; ferrolearn's EXTENDED-knot
# basis now MATCHES sklearn (formerly the clamped-knot basis DIVERGED):
python3 -c "import numpy as np; from sklearn.preprocessing import SplineTransformer; \
X=np.array([[0.],[0.3],[0.6],[1.0]]); st=SplineTransformer(n_knots=4, degree=3).fit(X); \
print('sklearn x_sp_0(0.0)=', round(float(st.transform(X)[0,0]),6))"
#   -> sklearn x_sp_0(0.0)= 0.166667    (ferrolearn now matches; old clamped Cox-de Boor gave 1.0)   REQ-2, was DIV-1
```

The in-module `#[test]`s exercise REQ-1 (the universal partition-of-unity /
non-negativity / column-count properties — `test_spline_output_dimensions` /
`_partition_of_unity` / `_non_negative` / `_multi_feature` / `_degree1` /
`_quantile_knots`) and REQ-6 (every error path). REQ-2 (basis VALUE parity, was
DIV-1) is now established by the **11 live-oracle value tests** in
`tests/divergence_spline_transformer.rs`
(`divergence_spline_basis_value_asymmetric_fixture` + `probe_a..f`), which pin the
FULL output matrix against the live sklearn 1.5.2 oracle at tol 1e-6 across
degree∈{1,2,3}, multi-feature, dense interior, and both base endpoints — these are
genuinely **scipy-`BSpline`-value oracle-grounded** (R-CHAR-3). The in-module
`test_spline_quantile_knots` still only asserts the column count + partition of
unity, NOT `np.percentile` value parity (REQ-5). No currently-green command
establishes REQ-3..REQ-5 or REQ-7..REQ-11.

## Blockers

REQ-1 (structural dims + B-spline properties), REQ-2 (basis VALUE parity, was
DIV-1), and REQ-6 (scoped error contracts) are SHIPPED. The remaining NOT-STARTED
REQs are open `-l blocker` issues filed against tracking issue #1331:

- #1332 — REQ-2 (was DIV-1, HEADLINE; now RESOLVED): the knot construction was
  rewritten from CLAMPED (repeated boundary, multiplicity `degree+1`, the `np.tile`
  block sklearn rejects at `_polynomial.py:898-906`) to sklearn's EXTENDED
  edge-spacing knots (`:908-923`), and `fn bspline_basis` re-keyed to the
  base-interval right endpoint `knots[n_basis]` (closed-right, scipy `BSpline`
  design matrix, `:925-940`). Basis VALUES now match scipy over the base interval;
  verified across degree∈{1,2,3}/multi-feature/both endpoints in
  `tests/divergence_spline_transformer.rs`. RESOLVED.
- #1333 — REQ-3: no `extrapolation` param ('constant' default + 'error'/'linear'/
  'continue'/'periodic', `:719-721`,`:734`).
- #1334 — REQ-4: no `include_bias` column drop (`:942`,`:806`).
- #1335 — REQ-5: quantile base knots not pinned to `np.percentile(X,
  100*np.linspace(0,1,n_knots))` (`:747-753`).
- #1336 — REQ-6 param-contract DIVs: sklearn allows `degree == 0` (piecewise-
  constant, `Interval(Integral, 0, None)`, `:718`) and imposes no `n_samples >= 2`
  requirement, yet ferrolearn rejects both.
- #1337 — REQ-7: no `sparse_output` (CSR design matrix) / `order` (`:725-726`,
  `:742-743`).
- #1338 — REQ-8: no `sample_weight` (weighted percentile / zero-weight mask,
  `:756-770`).
- #1339 — REQ-9: no `get_feature_names_out` (`{feat}_sp_{j}`, `:806-808`) /
  `bsplines_` / `n_features_out_` / `n_features_in_` surface (`:940-942`).
- #1340 — REQ-10: no `ferrolearn-python` `SplineTransformer` binding (boundary
  CPython consumer absent).
- #1341 — REQ-11: fit/transform on `ndarray` / `num_traits` / hand-rolled Cox-de
  Boor / `Vec<Vec<F>>`, not ferray (`ferray-core` arrays, a `ferray` B-spline)
  (R-SUBSTRATE-1/2).
```
