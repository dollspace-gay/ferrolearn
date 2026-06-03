# Normalizer

<!--
tier: 3-component
status: draft
baseline-commit: 8f8d2180
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_data.py  # normalize(X, norm="l2", *, axis=1, copy=True, return_norm=False) free fn (:1866): check_array(accept_sparse, copy, dtype=float, force_writeable=True) (:1933-1940) — default force_all_finite=True REJECTS NaN/inf, default min-samples/min-features=1 REJECT empty/zero-feature; dense path (:1962-1969): l1 norms=sum(|X|,axis=1); l2 norms=row_norms(X); max norms=max(|X|,axis=1); norms=_handle_zeros_in_scale(norms) [zeros->1]; X /= norms[:,None]; axis==0 transposes for column-normalize (:1926-1942,:1971-1972); return_norm returns norms (:1974-1975); sparse path inplace_csr_row_normalize_l1/l2 + max via min_max_axis (:1944-1960). class Normalizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (:1980); _parameter_constraints {norm:[StrOptions{l1,l2,max}], copy:["boolean"]} (:2053-2056); __init__(norm='l2', *, copy=True) (:2058-2060); fit validates params + sets n_features_in_ via _validate_data, returns self (:2062-2083); transform(X, copy=None) -> normalize(X, norm=self.norm, axis=1, copy=False) after _validate_data(reset=False) (:2085-2106); _more_tags stateless (:2108-2109). OneToOneFeatureMixin -> get_feature_names_out one-to-one.
ferrolearn-module: ferrolearn-preprocess/src/normalizer.rs
parity-ops: Normalizer, normalize
crosslink-issue: 1139
-->

## Summary

scikit-learn's `Normalizer` (`_data.py:1980`) rescales each **sample (row)** to
unit norm: every row with at least one non-zero component is divided by its L1,
L2, or Max norm so the chosen norm equals 1 (default `norm='l2'` — `:2004-2007`).
A row whose norm is zero is left unchanged (`_handle_zeros_in_scale` maps the
zero divisor to 1 — `:1968`). The estimator is stateless; its `transform`
delegates to the `normalize` free function (`:1866`) with `axis=1`, and a `fit`
that performs only parameter / shape validation. The estimator also supports a
`copy` parameter, fit-time `_parameter_constraints` validation
(`norm ∈ {l1,l2,max}`, `:2053-2056`), `n_features_in_` /
`OneToOneFeatureMixin.get_feature_names_out`, and a sparse-matrix path. The
standalone `normalize` function additionally exposes `axis` (column-normalize)
and `return_norm`.

`ferrolearn-preprocess/src/normalizer.rs` ships a **thin, stateless, dense-only**
transformer: `pub enum NormType { L1, L2 (default), Max }` and
`pub struct Normalizer<F> { norm: NormType }` with `new(norm)`, `l1()/l2()/max()`
constructors, a `norm()` accessor, and `Default` = L2. The only behavior is
`impl Transform<Array2<F>>`: it copies `x`, computes each row's norm
(L1 = Σ|v|, L2 = √Σv², Max = max|v|), leaves the row unchanged if the norm is
zero, otherwise divides each element by the norm — shape-preserving and
infallible (`Ok(...)` always). It also `impl`s `PipelineTransformer<F>` +
`FittedPipelineTransformer<F>` (stateless: `fit_pipeline` boxes `self`,
`transform_pipeline` calls `transform`). There is no `copy` parameter, no
validating `fit`, no `normalize` free function (no `axis`, no `return_norm`),
no `_parameter_constraints` (`NormType` is a closed Rust enum, so an
out-of-domain string is unrepresentable rather than validated), no
`n_features_in_` / feature-name plumbing, no sparse support, and no PyO3
binding.

**Headline finding (document prominently):** ferrolearn's per-row L1/L2/Max
division matches sklearn's dense `normalize` path (`:1962-1969`) **exactly**,
including the default-L2 norm and the zero-row-unchanged behavior — all four
REQ-1 Probe outputs (l1, l2, max, zero-row) are numerically identical to the
ferrolearn `Transform::transform` output (see Probes). The core dense transform
therefore SHIPS (REQ-1), with the in-file `FittedPipelineTransformer` impl plus
the crate re-export as real non-test consumers.

**Transform input validation (REQ-2) was FIXED this iteration (#1140):** sklearn's
`Normalizer.transform` routes through `check_array` with `force_all_finite=True`
and the default min-samples/min-features = 1, so it **raises `ValueError`** on
NaN / ±inf input, on `(0, n)` (zero samples), and on `(n, 0)` (zero features).
ferrolearn's `transform` previously did no validation; it now adds the same three
guards as the converged `binarizer.rs` REQ-9 (samples → features → finite),
returning the matching `FerroError`. The remaining gaps are surface (fit/params,
the `normalize` free fn, copy, feature names, sparse, PyO3, ferray).

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — core value match: row-wise L1/L2/Max + zero-row-unchanged (deterministic).
python3 -c "from sklearn.preprocessing import normalize, Normalizer; \
X=[[-2.,1.,2.],[-1.,0.,1.]]; \
print('l1', normalize(X,norm='l1').tolist()); \
print('l2', normalize(X,norm='l2').tolist()); \
print('max',normalize(X,norm='max').tolist()); \
print('zero',Normalizer().transform([[0.,0.,0.],[3.,4.,0.]]).tolist())"
# -> l1   [[-0.4, 0.2, 0.4], [-0.5, 0.0, 0.5]]
# -> l2   [[-0.6666666666666666, 0.3333333333333333, 0.6666666666666666],
#          [-0.7071067811865475, 0.0, 0.7071067811865475]]
# -> max  [[-1.0, 0.5, 1.0], [-1.0, 0.0, 1.0]]
# -> zero [[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]]   (zero row unchanged; _handle_zeros_in_scale)
#    ferrolearn: Normalizer::<f64>::l1()/l2()/max().transform(&X) == same, bit-for-bit.

# REQ-2 — input validation DIVERGENCE (sklearn raises; ferrolearn returns Ok):
python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer; \
Normalizer().transform(np.array([[float('nan'),1.0]]))"
# -> ValueError: Normalizer does not accept missing values encoded as NaN natively...
python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer; \
Normalizer().transform(np.array([[float('inf'),1.0]]))"
# -> ValueError: Input X contains infinity or a value too large for dtype('float64').
python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer; \
Normalizer().transform(np.empty((0,3)))"
# -> ValueError: Found array with 0 sample(s) (shape=(0, 3)) while a minimum of 1 is required by Normalizer.
python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer; \
Normalizer().transform(np.empty((2,0)))"
# -> ValueError: Found array with 0 feature(s) (shape=(2, 0)) while a minimum of 1 is required by Normalizer.
#    ferrolearn: transform does NO validation -> NaN/inf rows yield NaN/inf, empties pass as Ok.

# REQ-3 — fit-time _parameter_constraints validation (R-DEV-2 exception type):
python3 -c "from sklearn.preprocessing import Normalizer; Normalizer(norm='l3').fit([[1.0,2.0]])"
# -> InvalidParameterError: The 'norm' parameter of Normalizer must be a str among
#    {'l2', 'l1', 'max'}. Got 'l3' instead.   (_parameter_constraints, :2053-2056)

# REQ-4 — normalize free fn: axis=0 (column-normalize) + return_norm:
python3 -c "from sklearn.preprocessing import normalize; \
print(normalize([[1.,2.],[3.,4.]],norm='l1',axis=0).tolist()); \
X,n=normalize([[3.,4.]],norm='l2',return_norm=True); print(X.tolist(), n.tolist())"
# -> [[0.25, 0.3333333333333333], [0.75, 0.6666666666666666]]   (axis=0)
# -> [[0.6, 0.8]] [5.0]                                         (return_norm)

# REQ-6 — n_features_in_ / get_feature_names_out set on fit:
python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer; \
m=Normalizer().fit(np.ones((2,3))); print(m.n_features_in_, m.get_feature_names_out().tolist())"
# -> 3 ['x0', 'x1', 'x2']
```

## Requirements

- REQ-1: Row-wise unit-norm transform — for each row compute the norm
  (L1 = Σ|v| `:1963`; L2 = √Σv² via `row_norms` `:1965`; Max = max|v| `:1967`),
  leave the row unchanged when the norm is zero (`_handle_zeros_in_scale`
  zero→1, `:1968`), otherwise divide each element by the norm (`X /= norms[:,None]`
  `:1969`); default norm L2 (`:2004`); shape-preserving; matching sklearn dense
  `normalize` (`_data.py:1962-1969`) numerically for L1/L2/Max on dense input.
- REQ-2: `transform` input validation per `check_array` — reject non-finite
  (NaN / ±inf, `force_all_finite=True`, `:1933-1940`), zero samples
  (min-samples = 1), and zero features (min-features = 1) with a `FerroError`,
  in sklearn's `check_array` order (REQ-2 Probes: `ValueError`). **The fixable
  divergence this iteration** (mirrors `binarizer.rs` REQ-9).
- REQ-3: validating `fit(X, y=None)` — `_parameter_constraints`
  (`{norm:[StrOptions{l1,l2,max}], copy:["boolean"]}`, `:2053-2056`) raising the
  sklearn-matching error on an out-of-domain `norm` (REQ-3 Probe:
  `InvalidParameterError`), plus the stateless "validation only in `fit`,
  recommend `fit_transform`" contract (`:2033-2036`). (In ferrolearn `NormType`
  is a closed enum, so the string-domain check has no analog yet; a fitted type
  is needed to host `n_features_in_`.)
- REQ-4: `normalize(X, norm, *, axis=1, copy=True, return_norm=False)` standalone
  free function — the estimator-less API (`:1866`), including `axis=0`
  column-normalization (`:1926-1942`, `:1971-1972`) and `return_norm` returning
  the per-axis norms (`:1974-1975`) (REQ-4 Probes).
- REQ-5: `copy` constructor parameter + `transform(X, copy=None)` override — the
  in-place-vs-copy semantics (`__init__` `copy=True`, `:2058`; `transform` copy
  override, `:2085-2106`).
- REQ-6: `n_features_in_` / `feature_names_in_` set on `fit`, and
  `OneToOneFeatureMixin.get_feature_names_out` (one-to-one feature-name
  passthrough) (REQ-6 Probe). Depends on REQ-3 (set during `fit`).
- REQ-7: Sparse-matrix support — CSR `inplace_csr_row_normalize_l1/l2` and the
  Max path via `min_max_axis` operating on `X.data` (`:1944-1960`).
- REQ-8: PyO3 binding (`import ferrolearn` exposes `Normalizer` mirroring
  `import sklearn`) — the project boundary consumer.
- REQ-9: ferray substrate — the transform computes over `ferray-core` arrays /
  `ferray-ufunc` rather than `ndarray::Array2` + `num_traits::Float`
  (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `Normalizer::<f64>::l1()/l2()/max().transform(&X)` for
  `X = [[-2,1,2],[-1,0,1]]` equals the REQ-1 Probe outputs (l1 `[[-0.4,0.2,0.4],
  [-0.5,0,0.5]]`; l2 `[[-0.666..,0.333..,0.666..],[-0.707..,0,0.707..]]`; max
  `[[-1,0.5,1],[-1,0,1]]`) within ULP tolerance, and `l2().transform([[0,0,0],
  [3,4,0]])` leaves the zero row at `[0,0,0]` and yields `[0.6,0.8,0]`; output
  `shape() == X.shape()`. Pinned by an **oracle-grounded** `#[test]` whose
  expected values are the live-`sklearn` Probe outputs (R-CHAR-3).
- AC-2 (REQ-2): `Normalizer::l2().transform(&X)` returns `Err(FerroError)` for
  `X` containing NaN, for `X` containing ±inf, for `X` of shape `(0,3)`, and for
  `X` of shape `(2,0)` (REQ-2 Probes: `ValueError`); finite well-formed input is
  not over-rejected. (No validation exists today.)
- AC-3 (REQ-3): `Normalizer::l2().fit(&X)` returns a fitted handle exposing
  `n_features_in_`; an out-of-domain norm is unrepresentable or rejected with the
  sklearn-matching error (REQ-3 Probe: `InvalidParameterError`).
- AC-4 (REQ-4): a free `normalize(&X, norm, axis, copy, return_norm)` reproduces
  the REQ-4 Probes — `axis=0` column-normalize `[[0.25,0.333..],[0.75,0.666..]]`
  and `return_norm` yielding `([[0.6,0.8]], [5.0])`.
- AC-5 (REQ-5): a `copy` field/param exists and `transform(X, copy=false)` is
  observably in-place. (Neither exists today.)
- AC-6 (REQ-6): `get_feature_names_out` returns `['x0','x1','x2']` for a 3-feature
  fit (REQ-6 Probe); `n_features_in_ == 3` after `fit`.
- AC-7 (REQ-7): a sparse `transform` row-normalizes only stored nonzeros,
  matching dense `normalize` on the same matrix.
- AC-8 (REQ-8): `python3 -c "import ferrolearn; ferrolearn.preprocessing.Normalizer"`
  resolves and `.transform` matches `sklearn` on the REQ-1 Probe.
- AC-9 (REQ-9): the owned transform computes on `ferray-core` arrays (no
  `ndarray`/`num_traits` in the compute path).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (row-wise L1/L2/Max transform) | SHIPPED | impl `Transform::transform for Normalizer<F> in normalizer.rs` — per `row` computes `NormType::L1 => fold(+ \|v\|)`, `L2 => fold(+ v*v).sqrt()`, `Max => fold(max \|v\|)`; `if norm_val == F::zero() { continue }` (zero-row unchanged), else `*v = *v / norm_val`; returns `Ok(out)`. `Default`/`l2()` select `NormType::L2` (`impl Default for Normalizer<F>`, `pub fn l2`). Mirrors sklearn dense `normalize`: `l1: norms=sum(\|X\|,axis=1)` (`_data.py:1963`), `l2: norms=row_norms(X)` (`:1965`), `max: norms=max(\|X\|,axis=1)` (`:1967`), `_handle_zeros_in_scale` zero→1 (`:1968`), `X /= norms[:,None]` (`:1969`), default norm `'l2'` (`:2004`). Output equals all four REQ-1 Probes: l1 `[[-0.4,0.2,0.4],[-0.5,0,0.5]]`, l2 `[[-0.666..,0.333..,0.666..],[-0.707..,0,0.707..]]`, max `[[-1,0.5,1],[-1,0,1]]`, zero-row `[[0,0,0],[0.6,0.8,0]]`. Non-test consumers: (a) in-file `impl FittedPipelineTransformer<F> for Normalizer<F>::transform_pipeline` calls `self.transform(x)` (real pipeline integration, production); (b) crate re-export `pub use normalizer::Normalizer;` (`ferrolearn-preprocess/src/lib.rs` line 119) — the boundary public API, grandfathered under S5/R-DEFER-1. Verification: `cargo test -p ferrolearn-preprocess` (`test_l2_norm_basic`, `test_l1_norm_basic`, `test_max_norm_basic`, `test_zero_row_unchanged`, `test_pipeline_integration`, `test_f32_normalizer`). |
| REQ-2 (transform input validation per check_array) | SHIPPED | FIXED #1140. `Transform::transform` now guards, in sklearn's `check_array` order, zero samples (`x.nrows()==0` → `FerroError::InsufficientSamples`, `validation.py:1084`), zero features (`x.ncols()==0` → `InvalidParameter`, `:1093`), and non-finite NaN/±inf (`x.iter().any(|v| !v.is_finite())` → `InvalidParameter`, `:1063`, force_all_finite=True) — matching sklearn `Normalizer.transform` → `normalize` → `check_array` (`_data.py:1933-1940`). Mirrors the converged `binarizer.rs` REQ-9. Verification (acto-critic two-round, live-oracle, all green): 15 tests in `tests/divergence_normalizer.rs` — 6 rejection pins (NaN/+inf/-inf/zero-samples/zero-features/empty-order) + finite-not-over-rejected guards (zero-NORM-row, 1e308, subnormal, -0.0). The `FittedPipelineTransformer` consumer inherits the validation. |
| REQ-3 (validating fit + parameter constraints) | NOT-STARTED | open prereq blocker #1141. No `Fit` impl, no fitted type, no `_parameter_constraints` analog. `NormType` is a closed Rust enum so an out-of-domain string (`'l3'`) is unrepresentable rather than validated with `InvalidParameterError` (REQ-3 Probe; sklearn `:2053-2056`, `fit` `:2062-2083`). The "validation only in `fit`, recommend `fit_transform`" stateless contract (`:2033-2036`) is therefore absent; `PipelineTransformer::fit_pipeline` boxes `self` without validation. |
| REQ-4 (normalize free function: axis / return_norm) | NOT-STARTED | open prereq blocker #1142. No standalone `normalize` exists in `normalizer.rs` or the crate — only the estimator method path. Missing `axis=0` column-normalize (REQ-4 Probe `[[0.25,0.333..],[0.75,0.666..]]`; sklearn transpose `:1926-1942`,`:1971-1972`) and `return_norm` (REQ-4 Probe `([[0.6,0.8]],[5.0])`; sklearn `:1974-1975`). The estimator is hard-wired to `axis=1`, `return_norm=False`. |
| REQ-5 (copy parameter) | NOT-STARTED | open prereq blocker #1143. The struct holds only `norm: NormType`; no `copy` field, no `copy` constructor arg, no `transform(X, copy)` override (sklearn `__init__` `:2058`, `transform` `:2085-2106`). `transform` always `to_owned()`s, so in-place semantics are unrepresentable. |
| REQ-6 (n_features_in_ / feature names) | NOT-STARTED | open prereq blocker #1144. No `n_features_in_`, no `feature_names_in_`, no `get_feature_names_out` (REQ-6 Probe: `3 ['x0','x1','x2']`; sklearn `OneToOneFeatureMixin` + `_validate_data` in `fit`, `:2082`). Depends on #1141 (set during `fit`). |
| REQ-7 (sparse support) | NOT-STARTED | open prereq blocker #1145. `transform` accepts only `Array2<F>`; no CSR path, no `inplace_csr_row_normalize_l1/l2`, no `min_max_axis`-based Max, no `X.data`-only update (sklearn sparse path `:1944-1960`). |
| REQ-8 (PyO3 binding) | NOT-STARTED | open prereq blocker #1146. No `ferrolearn-python` registration; `import ferrolearn` cannot expose `Normalizer` (boundary consumer per R-DEFER-1). |
| REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker #1147. Compute path uses `ndarray::Array2` + `num_traits::Float` (`rows_mut`, manual fold), not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** Two public items in `normalizer.rs`: an enum
`NormType { L1, L2, Max }` (`#[default] L2`) and a generic struct
`Normalizer<F> { pub(crate) norm: NormType, _marker: PhantomData<F> }`.
Construction is `pub fn new(norm) -> Self` plus the convenience constructors
`l1()/l2()/max()`; `pub fn norm(&self) -> NormType` is the accessor; `impl
Default for Normalizer<F>` yields `Self::new(NormType::L2)` (matching sklearn's
default `'l2'`, `:2004`). The behavior lives in `impl Transform<Array2<F>>` with
`type Output = Array2<F>`, `type Error = FerroError`: it clones `x`, then for
each `row` folds the appropriate norm (L1 = `acc + v.abs()`; L2 =
`(acc + v*v).sqrt()`; Max = running `max(acc, v.abs())`), skips the row when
`norm_val == F::zero()` (zero-row unchanged), and otherwise divides each element
in place by `norm_val`, returning `Ok(out)` — element-row-wise, shape-preserving,
never errors. The generic bound `F: Float + Send + Sync + 'static` supports `f32`
and `f64`. The type is stateless: there is no unfitted/fitted split and no `Fit`
impl. It additionally `impl`s `PipelineTransformer<F>` (`fit_pipeline` boxes
`self.clone()` as a `Box<dyn FittedPipelineTransformer<F>>`) and
`FittedPipelineTransformer<F>` (`transform_pipeline` calls `self.transform(x)`) —
these are the real, non-test production consumers of the transform within the
crate.

**sklearn (target contract).** `Normalizer(OneToOneFeatureMixin,
TransformerMixin, BaseEstimator)` (`:1980`) stores `norm` and `copy` (`:2058-2060`).
`fit` (`:2062-2083`) is validate-only: `@_fit_context` triggers
`_parameter_constraints` (`{norm:[StrOptions{l1,l2,max}], copy:["boolean"]}`,
`:2053-2056`) and `_validate_data` records `n_features_in_`. `transform`
(`:2085-2106`) runs `_validate_data(reset=False, force_writeable=True, copy=copy)`
then delegates to the `normalize` free function with `axis=1` (`:2106`). `normalize`
(`:1866`) does `check_array(accept_sparse, copy, dtype=float, force_writeable=True)`
(`:1933-1940`) — note **no `force_all_finite` override**, so the default
`force_all_finite=True` rejects NaN/inf, and the default min-samples/min-features
= 1 reject empty / zero-feature arrays. For `axis=0` it transposes (column-
normalize, `:1926-1942`,`:1971-1972`). The dense path (`:1962-1969`) computes
`norms` per the norm, `_handle_zeros_in_scale` (zeros → 1, `:1968`), and
`X /= norms[:,None]` (`:1969`); `return_norm` returns `norms` (`:1974-1975`). The
sparse path uses `inplace_csr_row_normalize_l1/l2` and `min_max_axis` for Max
(`:1944-1960`). `OneToOneFeatureMixin` supplies a one-to-one
`get_feature_names_out`.

**The structural gap.** ferrolearn's dense per-row L1/L2/Max division is an exact
match for sklearn's dense `normalize` (all four REQ-1 Probes are numerically
identical, including default-L2 and the zero-row-unchanged
`_handle_zeros_in_scale` semantics). What is missing is *contract surface*: the
`transform` **input validation** (REQ-2 — the fixable divergence: NaN/inf/empty
pass silently where sklearn raises), the validating `fit` + fitted type +
`_parameter_constraints` (REQ-3), the standalone `normalize` free fn with `axis`
and `return_norm` (REQ-4), the `copy` parameter (REQ-5), feature-name plumbing
(REQ-6), the sparse path (REQ-7), the PyO3 binding (REQ-8), and the ferray
substrate (REQ-9). Only REQ-2 changes an observable result on well-formed-shape
input (silent `Ok(NaN)` vs `ValueError`); the rest extend the API to the full
sklearn contract.

## Verification

Commands establishing the single SHIPPED claim (REQ-1):

```bash
# Oracle (REQ-1 Probes) — dense row-wise L1/L2/Max + zero-row-unchanged, sklearn outputs:
python3 -c "from sklearn.preprocessing import normalize, Normalizer; \
X=[[-2.,1.,2.],[-1.,0.,1.]]; \
print(normalize(X,norm='l1').tolist()); \
print(normalize(X,norm='l2').tolist()); \
print(normalize(X,norm='max').tolist()); \
print(Normalizer().transform([[0.,0.,0.],[3.,4.,0.]]).tolist())"
#   -> [[-0.4, 0.2, 0.4], [-0.5, 0.0, 0.5]]
#   -> [[-0.666..., 0.333..., 0.666...], [-0.707..., 0.0, 0.707...]]
#   -> [[-1.0, 0.5, 1.0], [-1.0, 0.0, 1.0]]
#   -> [[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]]
# ferrolearn equivalents: Normalizer::<f64>::l1()/l2()/max().transform(&X)

# Crate gauntlet:
cargo test -p ferrolearn-preprocess        # incl. test_l1_norm_basic, test_l2_norm_basic,
                                           #       test_max_norm_basic, test_zero_row_unchanged,
                                           #       test_pipeline_integration, test_f32_normalizer
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s (`test_l2_norm_basic`, `test_l2_unit_norm_after_transform`,
`test_l1_norm_basic`, `test_l1_unit_norm_after_transform`, `test_max_norm_basic`,
`test_zero_row_unchanged`, `test_negative_values_l2`, `test_default_is_l2`,
`test_multiple_rows_independent`, `test_pipeline_integration`,
`test_f32_normalizer`) exercise the REQ-1 dense path and the pipeline consumer
comprehensively, but are **NOT oracle-grounded** — expected values are
hand-written, not live-derived from sklearn. To satisfy R-CHAR-3 the critic
should add an oracle-pinned green guard asserting the four REQ-1 Probe outputs
(l1 / l2 / max / zero-row). No currently-green command establishes any of
REQ-2..REQ-9; in particular REQ-2's divergence should be pinned as a FAILING
oracle-grounded `#[test]` (NaN/inf/empty/zero-feature `transform` → expected
`Err`, currently `Ok`).

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns
`#`-numbers, replacing the `#<...>` placeholders); reference them in the REQ
status table:

- #1140 — REQ-2: `transform` performs no `check_array`
  validation; NaN/±inf yield `Ok(NaN/inf)` and `(0,n)`/`(n,0)` pass silently
  where sklearn raises `ValueError` (REQ-2 Probes; `_data.py:1933-1940`,
  `:2103-2104`). **The fixable divergence — pin + fix first** (mirror
  `binarizer.rs` REQ-9).
- #1141 — REQ-3: no `Fit` impl, no fitted type, no
  `_parameter_constraints` / `InvalidParameterError` (REQ-3 Probe), no
  fit-only-validation contract (`:2033-2036`, `:2053-2083`).
- #1142 — REQ-4: no standalone `normalize` free fn; missing
  `axis=0` column-normalize and `return_norm` (`:1866`, `:1926-1942`,
  `:1971-1975`).
- #1143 — REQ-5: no `copy` parameter on the constructor or
  `transform`; `transform` always `to_owned()`s, so in-place semantics
  (`:2085-2106`) are unrepresentable.
- #1144 — REQ-6: no `n_features_in_` / `feature_names_in_` /
  `get_feature_names_out` (OneToOneFeatureMixin; set on `fit`, `:2082`). Depends
  on #1141.
- #1145 — REQ-7: dense-only; no CSR path, no
  `inplace_csr_row_normalize_l1/l2`, no `min_max_axis` Max (`:1944-1960`).
- #1146 — REQ-8: no `ferrolearn-python` registration of
  `Normalizer`.
- #1147 — REQ-9: compute path on `ndarray`/`num_traits`, not
  ferray (R-SUBSTRATE-1/2).
