# PolynomialFeatures

<!--
tier: 3-component
status: draft
baseline-commit: bad82b9b
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_polynomial.py  # class PolynomialFeatures(TransformerMixin, BaseEstimator) (:99). __init__(degree=2, *, interaction_only=False, include_bias=True, order="C") (:201-207); _parameter_constraints {degree:[Interval(Integral,0,None,left),"array-like"], interaction_only:["boolean"], include_bias:["boolean"], order:[StrOptions({"C","F"})]} (:194-199). _combinations(n_features,min_degree,max_degree,interaction_only,include_bias) (:209-220): comb=combinations if interaction_only else combinations_with_replacement; chain.from_iterable(comb(range(n),i) for i in range(max(1,min_degree), max_degree+1)); if include_bias prepend comb(range(n),0) (empty tuple). powers_ property via np.bincount (:250-264). get_feature_names_out (:266-303). STATEFUL fit (:306-400): _validate_data accept_sparse=True; int degree -> _min_degree=0/_max_degree=degree (:325-333); tuple degree -> (_min_degree,_max_degree) (:334-360); n_output_features_=_num_combinations (:362). transform (:402-...): check_is_fitted; _validate_data(order="F", dtype=FLOAT_DTYPES, reset=False, accept_sparse=("csr","csc")) -> rejects NaN/inf + transform-time n_features_in_ mismatch; dense + sparse CSR/CSC expansion.
ferrolearn-module: ferrolearn-preprocess/src/polynomial_features.rs
parity-ops: PolynomialFeatures
crosslink-issue: 1179
-->

## Summary

scikit-learn's `PolynomialFeatures` (`_polynomial.py:99`) generates a new feature
matrix of all polynomial combinations of the input features up to a specified
degree: for an input row `[a, b]` and `degree=2` it produces
`[1, a, b, a^2, a*b, b^2]` (`:104-105`). It is a **stateful** transformer
(`TransformerMixin, BaseEstimator`): `fit` (`:306`) records `n_features_in_`,
`n_output_features_`, `_min_degree`/`_max_degree`; it exposes a `powers_`
attribute (`:250`) and `get_feature_names_out` (`:266`); `degree` accepts an int
**or** a `(min_degree, max_degree)` tuple (`:111-117`, `:325-360`); it has an
`order` ('C'/'F') output-layout parameter (`:132-134`); and `transform` (`:402`)
validates `X` against `n_features_in_`, rejects non-finite values, and supports
dense **and** sparse (CSR/CSC) input.

`ferrolearn-preprocess/src/polynomial_features.rs` ships a **thin, stateless,
dense-only** transformer: `pub struct PolynomialFeatures<F> { degree: usize,
interaction_only: bool, include_bias: bool }`. Construction is `new(degree,
interaction_only, include_bias)` (errors only on `degree == 0`), with
`default_config()` (degree=2, false, true), `degree()`/`interaction_only()`/
`include_bias()` accessors, and `Default`. A private `feature_combinations`
generates the index-tuples by DFS (interaction_only -> strictly increasing
indices; else non-decreasing -> pure powers), prepending the empty combo for the
bias and sorting by combo length then lexicographically. `impl
Transform<Array2<F>>` (`transform`) errors on zero columns and otherwise
multiplies the selected columns per combo into an `Array2<F>` output. It also
implements `PipelineTransformer`/`FittedPipelineTransformer`. Consumers: the
crate re-export `pub use polynomial_features::PolynomialFeatures` (`lib.rs`) and
the generic pipeline path. There is no PyO3 binding.

**Headline finding (document prominently):** for **integer-degree dense** input,
ferrolearn's polynomial feature **values AND column order are an exact match**
for sklearn — the DFS + length-then-lexicographic sort reproduces sklearn's
`itertools` combination order column-for-column across default, `interaction_only`,
and `include_bias=false` configs (see Probes 1-3: `[1,2,3,4,6,9]`,
`[1,2,3,5,4,6,10,9,15,25]`, the 7-term interaction-only, the no-bias degree-3
interaction-only — all bit-identical). The core dense int-degree transform
therefore SHIPS (REQ-1). The remaining REQs are missing contract surface: the
`(min_degree,max_degree)` tuple degree (REQ-2), the `order` param (REQ-3), the
stateful `fit` + `n_features_in_`/`n_output_features_` + transform-time
feature-count check (REQ-4), `powers_` (REQ-5), `get_feature_names_out` (REQ-6),
sparse support (REQ-7), the **fixable input-validation divergence** —
non-finite (NaN/inf) rejection + zero-samples rejection (REQ-8), the full
constructor + `_parameter_constraints`/`order` (REQ-9), the PyO3 binding
(REQ-10), and the ferray substrate (REQ-11).

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — degree=2, 2 features, full+bias; VALUE + COLUMN ORDER [1,a,b,a^2,a*b,b^2]:
python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
print(PolynomialFeatures(2).fit_transform([[2.,3.]]).tolist()); \
print(PolynomialFeatures(2).fit([[2.,3.]]).powers_.tolist())"
# -> [[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]]
#    [[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]   (powers_; REQ-5)
#    ferrolearn: PolynomialFeatures::<f64>::new(2,false,true).transform([[2,3]]) == [[1,2,3,4,6,9]]

# REQ-1 — 3 features, degree=2: column ORDER is the subtle part (itertools order):
python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
print(PolynomialFeatures(2).fit_transform([[2.,3.,5.]]).tolist()); \
print(PolynomialFeatures(2, interaction_only=True).fit_transform([[2.,3.,5.]]).tolist())"
# -> [[1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 10.0, 9.0, 15.0, 25.0]]  (bias,a,b,c,a^2,ab,ac,b^2,bc,c^2)
#    [[1.0, 2.0, 3.0, 5.0, 6.0, 10.0, 15.0]]                  (interaction_only: bias,a,b,c,ab,ac,bc)
#    ferrolearn new(2,false,true) / new(2,true,true) on [[2,3,5]] == same, same order

# REQ-1 — interaction_only, no bias, degree=3, 3 features; VALUE + ORDER:
python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
print(PolynomialFeatures(3, interaction_only=True, include_bias=False).fit_transform([[2.,3.,5.]]).tolist())"
# -> [[2.0, 3.0, 5.0, 6.0, 10.0, 15.0, 30.0]]  (a,b,c,ab,ac,bc,abc)
#    ferrolearn new(3,true,false) on [[2,3,5]] == same

# REQ-8 — sklearn rejects non-finite NaN/inf (ferrolearn does NOT today -> fixable divergence):
python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
PolynomialFeatures().fit_transform(np.array([[float('nan'),1.]]))"
# -> ValueError: Input X contains NaN.  (_validate_data -> _assert_all_finite, transform :433-435)
python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
PolynomialFeatures().fit_transform(np.array([[float('inf'),1.]]))"
# -> ValueError: Input X contains infinity or a value too large for dtype('float64').

# REQ-8 — sklearn rejects 0 samples (ferrolearn does NOT today -> fixable divergence):
python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
PolynomialFeatures().fit_transform(np.empty((0,2)))"
# -> ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum of 1 is required ...

# REQ-8 / REQ-1 — sklearn rejects 0 features (ferrolearn ALSO errors, with InvalidParameter):
python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
PolynomialFeatures().fit_transform(np.empty((2,0)))"
# -> ValueError: Found array with 0 feature(s) (shape=(2, 0)) while a minimum of 1 is required ...
#    ferrolearn: transform on 0-col input -> Err(FerroError::InvalidParameter) (already matches "raises")

# REQ-2 — degree tuple (min_degree=2, max_degree=3): only degree 2-3 terms (ferrolearn int-only):
python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
print(PolynomialFeatures(degree=(2,3)).fit_transform([[2.,3.]]).tolist()); \
print(PolynomialFeatures(degree=(2,3)).fit([[2.,3.]]).powers_.tolist())"
# -> [[4.0, 6.0, 9.0, 8.0, 12.0, 18.0, 27.0]]   (a^2,ab,b^2,a^3,a^2b,ab^2,b^3 — no bias/linear)
#    [[2,0],[1,1],[0,2],[3,0],[2,1],[1,2],[0,3]]

# REQ-4 — transform-time feature-count consistency check (ferrolearn has none):
python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
m=PolynomialFeatures(2).fit([[2.,3.]]); m.transform([[1.,2.,3.]])"
# -> ValueError: X has 3 features, but PolynomialFeatures is expecting 2 features as input.

# REQ-6 — get_feature_names_out + n_output_features_ / n_features_in_ (ferrolearn has none):
python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
m=PolynomialFeatures(2).fit([[2.,3.]]); print(m.get_feature_names_out().tolist()); \
print(m.n_output_features_, m.n_features_in_)"
# -> ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2'] ; 6 2
```

## Requirements

- REQ-1: Integer-degree dense polynomial feature generation — VALUES and exact
  COLUMN ORDER matching sklearn `_combinations` itertools order
  (`_polynomial.py:209-220`) for the default (`combinations_with_replacement`),
  `interaction_only=True` (`combinations`), and `include_bias=False` configs:
  bias (empty product) -> degree-ascending -> lexicographic-within-degree
  (Probes 1-3), with each output column the product of its selected input
  columns.
- REQ-2: `degree` as a `(min_degree, max_degree)` tuple — generating only terms
  of degree in `[min_degree, max_degree]` (sklearn `:111-117`, `:334-360`;
  Probe 2 `degree=(2,3)` -> `[a^2,ab,b^2,a^3,...]`). ferrolearn accepts a single
  `usize` degree only and always starts at degree 1.
- REQ-3: `order` ('C'/'F') output memory-layout parameter (sklearn `:132-134`,
  `:201`, `_validate_data(order="F", ...)` in `transform` `:434`).
- REQ-4: Stateful `fit(X, y=None)` setting `n_features_in_`,
  `n_output_features_`, `_min_degree`/`_max_degree` (sklearn `:306-400`), plus a
  `transform`-time feature-count consistency check against `n_features_in_`
  (Probe `:402-435`: `X has 3 features, but PolynomialFeatures is expecting 2`).
  ferrolearn is stateless (its `fit_pipeline` just boxes `self`) and records no
  fitted state.
- REQ-5: `powers_` attribute — `ndarray (n_output_features_, n_features_in_)`
  where `powers_[i,j]` is the exponent of input `j` in output `i`, via
  `np.bincount` over the combinations (sklearn `:250-264`; Probe 1 powers).
- REQ-6: `get_feature_names_out(input_features=None)` — string feature names
  `['1','x0','x1','x0^2','x0 x1','x1^2', ...]` from `powers_` (sklearn
  `:266-303`; Probe 6).
- REQ-7: Sparse (CSR/CSC) `transform` support — the `_csr_polynomial_expansion`
  fast path for degree 2-3, CSC<->CSR conversion for degree>=4, empty-matrix
  edge case (sklearn `:402-...`, `_create_expansion` `:38-96`). ferrolearn
  accepts only `Array2<F>`.
- REQ-8: Input validation per `_validate_data` defaults — reject non-finite
  (NaN / +/-inf) input and zero-samples input, matching sklearn's `transform`
  `_assert_all_finite` + min-samples check (Probe REQ-8). ferrolearn rejects
  zero **columns** (already matching "raises") but NOT NaN/inf and NOT
  zero-rows. **This is the single most fixable minimal divergence.**
- REQ-9: Full constructor + `_parameter_constraints` — `__init__(degree=2, *,
  interaction_only=False, include_bias=True, order="C")` with keyword-only args
  and the `Interval(Integral,0,None)`/`"boolean"`/`StrOptions({"C","F"})`
  validation raising `InvalidParameterError` (sklearn `:194-207`). ferrolearn's
  `new` is positional and validates only `degree == 0`.
- REQ-10: PyO3 binding — `import ferrolearn` exposes `PolynomialFeatures`
  mirroring `import sklearn` (the project boundary consumer). None exists.
- REQ-11: ferray substrate — the combination/expansion computes over
  `ferray-core` arrays / `ferray-ufunc` rather than `ndarray::Array2` +
  `num_traits::Float` (R-SUBSTRATE-1/2).

## Acceptance criteria

- AC-1 (REQ-1): `PolynomialFeatures::<f64>::new(2,false,true).transform(&x)` for
  `x=[[2,3]]` equals Probe 1 `[[1,2,3,4,6,9]]`; `new(2,false,true)` /
  `new(2,true,true)` on `[[2,3,5]]` equal Probe 2's two outputs
  (`[[1,2,3,5,4,6,10,9,15,25]]` and `[[1,2,3,5,6,10,15]]`) **in that column
  order**; `new(3,true,false)` on `[[2,3,5]]` equals
  `[[2,3,5,6,10,15,30]]`. Pinned by an **oracle-grounded** `#[test]` whose
  expected values are the live-`sklearn` Probe outputs (R-CHAR-3).
- AC-2 (REQ-2): a `(min_degree,max_degree)` config exists and
  `PolynomialFeatures::degree((2,3)).transform([[2,3]])` equals
  `[[4,6,9,8,12,18,27]]` (Probe 2 tuple). (No representation today.)
- AC-3 (REQ-3): an `order` parameter selects C/F output layout. (Absent.)
- AC-4 (REQ-4): a `fit` produces a fitted handle exposing `n_features_in_` and
  `n_output_features_`; a `transform` with the wrong feature count returns a
  `FerroError` matching sklearn's `X has N features, but ... is expecting M`
  (Probe REQ-4). (Absent — stateless today.)
- AC-5 (REQ-5): a `powers_` accessor returns the Probe-1 exponent matrix
  `[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]`. (Absent.)
- AC-6 (REQ-6): `get_feature_names_out` returns
  `['1','x0','x1','x0^2','x0 x1','x1^2']` (Probe 6). (Absent.)
- AC-7 (REQ-7): a sparse `transform` produces the same nonzero values/order as
  sklearn for a CSR input. (Absent — dense-only.)
- AC-8 (REQ-8): `transform` on input containing NaN or +/-inf returns `Err`
  (matching sklearn `ValueError`), and on a zero-row input returns `Err`
  (matching sklearn min-samples `ValueError`); finite extremes
  (1e308, subnormal, -0.0) are NOT over-rejected. Pinned by oracle-grounded
  `#[test]`s. (Today: NaN/inf and zero-rows pass through silently.)
- AC-9 (REQ-9): the constructor mirrors sklearn keyword-only signature/defaults
  and rejects an out-of-domain `order` / non-boolean flag with an
  `InvalidParameter`-class error. (Today: positional, degree-0-only check.)
- AC-10 (REQ-10): `python3 -c "import ferrolearn; ferrolearn.preprocessing.PolynomialFeatures"`
  resolves and `.fit_transform` matches `sklearn` on Probe 1.
- AC-11 (REQ-11): the owned combination/product path computes on `ferray-core`
  arrays (no `ndarray`/`num_traits` in the compute path).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (int-degree dense values + column order) | SHIPPED | impl `Transform::transform for PolynomialFeatures<F> in polynomial_features.rs` multiplies, per combo, the selected columns (`combo.iter().fold(F::one(), \|acc, &j\| acc * x[[i,j]])`) into `Array2<F>`; column set + ORDER come from `fn feature_combinations in polynomial_features.rs` — bias = empty combo prepended, DFS with `interaction_only ? last_idx+1 : last_idx` start (strictly-increasing vs non-decreasing/pure-powers), then `combos.sort_by(\|a,b\| a.len().cmp(&b.len()).then_with(\|\| a.cmp(b)))`. This reproduces sklearn `_combinations` (`comb=combinations if interaction_only else combinations_with_replacement; chain.from_iterable(comb(range(n),i) for i in range(max(1,min_degree),max_degree+1))`; bias prepend, `_polynomial.py:209-220`) column-for-column. Output equals Probes 1-3 bit-for-bit and in order: `[[1,2,3,4,6,9]]` (2 feat default), `[[1,2,3,5,4,6,10,9,15,25]]` + `[[1,2,3,5,6,10,15]]` (3 feat default / interaction_only), `[[2,3,5,6,10,15,30]]` (deg-3 interaction_only no-bias). Non-test consumers: crate re-export `pub use polynomial_features::PolynomialFeatures;` (`ferrolearn-preprocess/src/lib.rs`) — the boundary public API, grandfathered under S5/R-DEFER-1 — and the generic pipeline path `impl PipelineTransformer<F> / FittedPipelineTransformer<F> for PolynomialFeatures<F> in polynomial_features.rs` whose `transform_pipeline` delegates to `transform`. Verification: `cargo test -p ferrolearn-preprocess` (`test_degree2_two_features_with_bias`, `test_degree2_interaction_only`, `test_no_bias`, `test_degree3_single_feature`) — to satisfy R-CHAR-3 the critic must add an oracle-grounded guard asserting the Probe 1-3 outputs (existing tests are hand-written, not live-derived). |
| REQ-2 (degree tuple / min_degree) | NOT-STARTED | open prereq blocker #1181. `degree` is a single `usize`; `feature_combinations` always starts at degree 1 (DFS seeds `vec![i]`), so a `(min_degree,max_degree)` range with `min_degree>=2` (Probe 2: `[a^2,ab,b^2,a^3,...]`, no bias/linear) is unrepresentable (sklearn `:334-360`). |
| REQ-3 (order C/F param) | NOT-STARTED | open prereq blocker #1182. No `order` field; output is always ndarray default (row-major) layout, with no C/F selection (sklearn `:132-134`, `:201`, `_validate_data(order="F")` `:434`). |
| REQ-4 (stateful fit + n_features_in_/n_output_features_ + transform feature-count check) | NOT-STARTED | open prereq blocker #1183. The struct is stateless; `fit_pipeline` just `Box::new(self.clone())` and records nothing. No `n_features_in_`, no `n_output_features_`, no `_min_degree`/`_max_degree`, and `transform` does NOT reject a feature-count mismatch (Probe REQ-4: sklearn raises `X has 3 features, but ... expecting 2`; sklearn `:306-400`, `:402-435`). |
| REQ-5 (powers_ attribute) | NOT-STARTED | open prereq blocker #1184. No `powers_` accessor / exponent matrix; `feature_combinations` returns index tuples but never the per-input `np.bincount` exponent rows `[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]` (Probe 1; sklearn `:250-264`). Depends on REQ-4 (`powers_` is a fitted attribute). |
| REQ-6 (get_feature_names_out) | NOT-STARTED | open prereq blocker #1185. No feature-name generation; sklearn yields `['1','x0','x1','x0^2','x0 x1','x1^2']` from `powers_` (Probe 6; sklearn `:266-303`). Depends on REQ-5. |
| REQ-7 (sparse CSR/CSC support) | NOT-STARTED | open prereq blocker #1186. `transform` accepts only `Array2<F>`; no CSR/CSC path, no `_csr_polynomial_expansion` fast path (degree 2-3), no CSC<->CSR conversion (degree>=4), no empty-matrix edge case (sklearn `:402-...`, `_create_expansion` `:38-96`). |
| REQ-8 (input validation: non-finite + zero-samples + zero-features) | SHIPPED | FIXED #1180. `transform` now guards, in sklearn `check_array` order, zero-samples (`x.nrows()==0` → `InsufficientSamples`), zero-features (`x.ncols()==0` → `InvalidParameter`), and non-finite NaN/±inf (`x.iter().any(|v| !v.is_finite())` → `InvalidParameter`) — matching sklearn `transform` → `_validate_data` defaults (`_polynomial.py:433-435`). Mirrors converged binarizer/normalizer. Critic two-round CLEAN vs live oracle: 13 tests in `tests/divergence_polynomial_features.rs` — 4 rejection pins (NaN/+inf/-inf/zero-rows) + finite-not-over-rejected guards incl. the polynomial-specific case where a FINITE input (1e308) whose product overflows to inf is correctly ACCEPTED (sklearn validates the input, not the expansion output). |
| REQ-9 (full ctor + _parameter_constraints / order validation) | NOT-STARTED | open prereq blocker #1187. `new(degree, interaction_only, include_bias)` is positional and validates only `degree == 0`; no keyword-only `*`-args, no `order`, no `_parameter_constraints`-style `InvalidParameterError` on a bad `order`/non-boolean (sklearn `:194-207`). |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #1188. No `ferrolearn-python` registration; `import ferrolearn` cannot expose `PolynomialFeatures` (boundary consumer per R-DEFER-1). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #1189. Compute path uses `ndarray::Array2` + `num_traits::Float` + `Array2::zeros`/indexing, not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** A single generic struct `PolynomialFeatures<F>`
(`polynomial_features.rs`) holding three `pub(crate)` config fields (`degree:
usize`, `interaction_only: bool`, `include_bias: bool`) plus a `PhantomData<F>`.
`pub fn new(degree, interaction_only, include_bias)` returns
`Err(FerroError::InvalidParameter)` when `degree == 0` and is otherwise
infallible; `default_config()` yields `(2, false, true)` (matching sklearn's
defaults — `:201-203`); `degree()`/`interaction_only()`/`include_bias()` are
accessors; `impl Default` delegates to `default_config()`. The private
`fn feature_combinations(n_features) -> Vec<Vec<usize>>` prepends the empty combo
for the bias (when `include_bias`), then DFS-generates index tuples up to
`self.degree` — seeding `vec![i]` per feature and extending from `last_idx + 1`
(interaction_only -> strictly increasing, distinct features) or `last_idx`
(non-decreasing -> repeated features = pure powers) — and finally
`sort_by(len, then lexicographic)` so output order is bias -> degree-ascending ->
lexicographic-within-degree. `impl Transform<Array2<F>>` (`transform`) returns
`Err(FerroError::InvalidParameter)` on zero columns, then for each combo folds
the product of the selected input columns into an `Array2<F>` output of shape
`(n_samples, n_combos)`. `PipelineTransformer`/`FittedPipelineTransformer` make
the type usable as a stateless pipeline step (`fit_pipeline` boxes `self.clone()`;
`transform_pipeline` delegates to `transform`). The type is **stateless** — no
unfitted/fitted split, no recorded `n_features_in_`.

**sklearn (target contract).** `PolynomialFeatures(TransformerMixin,
BaseEstimator)` (`:99`) is **stateful**. `__init__(degree=2, *,
interaction_only=False, include_bias=True, order="C")` (`:201-207`) stores four
params with `*`-only kwargs; `_parameter_constraints` (`:194-199`) validates
`degree` as `Interval(Integral,0,None)` or array-like, the flags as boolean, and
`order` as `StrOptions({"C","F"})`. `fit` (`:306-400`) runs
`_validate_data(accept_sparse=True)`, resolves `_min_degree`/`_max_degree` (int
degree -> `0`/`degree`, `:332-333`; tuple -> the pair with non-negative
`min<=max` validation, `:334-360`), and sets `n_output_features_` via
`_num_combinations` (`:362`). The combination order is defined by `_combinations`
(`:209-220`): `comb = combinations if interaction_only else
combinations_with_replacement`, `chain.from_iterable(comb(range(n_features), i)
for i in range(max(1, min_degree), max_degree + 1))`, with `comb(range,0)` (the
empty tuple = bias) prepended when `include_bias`. `powers_` (`:250-264`)
`np.vstack`es a `np.bincount` over those combinations; `get_feature_names_out`
(`:266-303`) renders names from `powers_`. `transform` (`:402-...`) runs
`check_is_fitted`, then `_validate_data(order="F", dtype=FLOAT_DTYPES,
reset=False, accept_sparse=("csr","csc"))` — which **rejects NaN/inf and a
transform-time feature-count mismatch** — and dispatches to a dense product or a
CSR/CSC `_csr_polynomial_expansion` path.

**The structural gap.** ferrolearn's int-degree dense combination order is an
exact match for sklearn's itertools order (Probes 1-3 are bit-identical,
column-for-column), so the core numeric/ordering contract SHIPS (REQ-1). What is
missing is *surface and validation*: the tuple `degree`/`min_degree` (REQ-2), the
`order` param (REQ-3), the entire stateful `fit` + fitted attributes + the
transform-time feature-count guard (REQ-4), `powers_` (REQ-5),
`get_feature_names_out` (REQ-6), sparse support (REQ-7), the fixable non-finite +
zero-samples input validation (REQ-8), the full keyword-only constructor +
`_parameter_constraints`/`order` (REQ-9), the PyO3 binding (REQ-10), and the
ferray substrate (REQ-11). None of these change the int-degree dense numeric
result; REQ-8 is the minimal, self-contained divergence the critic should pin
first (mirroring binarizer's REQ-9 input-validation fix).

## Verification

Commands establishing the single SHIPPED claim (REQ-1):

```bash
# Oracle (REQ-1 Probes) — VALUE + COLUMN ORDER, int-degree dense:
python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
print(PolynomialFeatures(2).fit_transform([[2.,3.]]).tolist())"
#   -> [[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]]
python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
print(PolynomialFeatures(2).fit_transform([[2.,3.,5.]]).tolist()); \
print(PolynomialFeatures(2, interaction_only=True).fit_transform([[2.,3.,5.]]).tolist())"
#   -> [[1.0,2.0,3.0,5.0,4.0,6.0,10.0,9.0,15.0,25.0]]
#      [[1.0,2.0,3.0,5.0,6.0,10.0,15.0]]
python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
print(PolynomialFeatures(3, interaction_only=True, include_bias=False).fit_transform([[2.,3.,5.]]).tolist())"
#   -> [[2.0, 3.0, 5.0, 6.0, 10.0, 15.0, 30.0]]
# ferrolearn equivalents: PolynomialFeatures::<f64>::new(2,false,true)/new(2,true,true)/new(3,true,false).transform(&x)

# Crate gauntlet:
cargo test -p ferrolearn-preprocess        # incl. test_degree2_two_features_with_bias,
                                           #       test_degree2_interaction_only,
                                           #       test_no_bias, test_degree3_single_feature
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s (`test_degree2_two_features_with_bias`,
`test_degree2_interaction_only`, `test_no_bias`, `test_degree1_only_linear`,
`test_multiple_rows`, `test_single_feature_degree2`, `test_degree3_single_feature`,
`test_invalid_degree_zero`, `test_default_config`, `test_pipeline_integration`)
exercise the REQ-1 dense path but are **NOT oracle-grounded** — expected values
are hand-written, not live-derived from sklearn. To satisfy R-CHAR-3 the critic
should add an oracle-pinned green guard asserting the Probe 1-3 outputs
(VALUE + column ORDER), then pin REQ-8 (non-finite + zero-samples) as a FAILING
oracle-grounded test. No currently-green command establishes any of REQ-2..REQ-11.

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns
`#`-numbers); reference them in the REQ status table:

- #1181 — REQ-2: no `(min_degree, max_degree)` tuple degree; combos
  always start at degree 1 (sklearn `:334-360`).
- #1182 — REQ-3: no `order` ('C'/'F') output-layout parameter (sklearn
  `:132-134`, `:201`).
- #1183 — REQ-4: stateless; no `n_features_in_`/`n_output_features_`/
  `_min_degree`/`_max_degree`, no transform-time feature-count guard (sklearn
  `:306-435`).
- #1184 — REQ-5: no `powers_` exponent matrix (sklearn `:250-264`). Depends
  on #1183.
- #1185 — REQ-6: no `get_feature_names_out` (sklearn `:266-303`).
  Depends on #1184.
- #1186 — REQ-7: dense-only; no CSR/CSC `_csr_polynomial_expansion` path
  (sklearn `:402-...`, `:38-96`).
- #1180 — REQ-8: `transform` does not reject NaN/+/-inf or zero-samples
  input (sklearn `_validate_data` defaults, Probe REQ-8). **Most fixable.**
- #1187 — REQ-9: positional `new` with degree-0-only check; no keyword-only
  args, no `order`, no `_parameter_constraints`/`InvalidParameterError` (sklearn
  `:194-207`).
- #1188 — REQ-10: no `ferrolearn-python` registration of `PolynomialFeatures`.
- #1189 — REQ-11: compute path on `ndarray`/`num_traits`, not ferray
  (R-SUBSTRATE-1/2).
```
