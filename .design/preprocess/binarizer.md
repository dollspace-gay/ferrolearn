# Binarizer

<!--
tier: 3-component
status: draft
baseline-commit: 374d232e
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_data.py  # binarize(X, *, threshold=0.0, copy=True) free fn (:2120-2174): dense path cond=X>threshold; X[cond]=1; X[not_cond]=0 (:2170-2173), STRICT greater-than; sparse path forbids threshold<0 ValueError (:2161-2163), eliminate_zeros (:2168). class Binarizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (:2177); docstring "Values greater than the threshold map to 1, while values less than or equal to the threshold map to 0" (:2180-2182), "below or equal to this are replaced by 0, above it by 1" (:2196-2197); _parameter_constraints {threshold:[Real], copy:["boolean"]} (:2248-2251); __init__(*, threshold=0.0, copy=True) (:2253-2255); fit validates only via _validate_data, sets n_features_in_/feature_names_in_, returns self (:2257-2278); transform(X, copy=None) -> binarize(...) (:2280-2308). OneToOneFeatureMixin -> get_feature_names_out one-to-one. "stateless... recommend fit_transform... parameter validation only in fit" (:2228-2231)
ferrolearn-module: ferrolearn-preprocess/src/binarizer.rs
parity-ops: Binarizer, binarize
crosslink-issue: 1122
-->

## Summary

scikit-learn's `Binarizer` (`_data.py:2177`) sets feature values to 0 or 1 by a
threshold: values **strictly greater** than `threshold` map to 1, values **less
than or equal** map to 0 (default `threshold=0.0`, so only positive values map
to 1 — `:2180-2182`, `:2196-2197`). The estimator is stateless; its `transform`
delegates to the `binarize` free function (`:2120-2174`), whose dense path is
`cond = X > threshold; X[cond] = 1; X[not_cond] = 0` (`:2170-2173`). It also
supports a `copy` parameter, a `fit` that validates parameters and records
`n_features_in_`/`feature_names_in_`, `OneToOneFeatureMixin.get_feature_names_out`,
and a sparse-matrix path that forbids `threshold < 0` (`:2161-2163`).

`ferrolearn-preprocess/src/binarizer.rs` ships a **thin, stateless,
dense-only** transformer: `pub struct Binarizer<F> { pub(crate) threshold: F }`
with `new(threshold)`, a `threshold()` accessor, and `Default` (threshold = 0.0).
The only behavior is `impl Transform<Array2<F>>` whose `transform` is
`x.mapv(|v| if v > self.threshold { F::one() } else { F::zero() })` —
**strict greater-than**, shape-preserving, infallible. There is no `copy`, no
`fit`, no `binarize` free function, no `n_features_in_`/feature-name plumbing, no
sparse support, and no PyO3 binding.

**Headline finding (document prominently):** ferrolearn's `v > threshold`
matches sklearn's `X > threshold` (`:2170`) **exactly** on the dense path,
including the default-threshold (only positives map to 1) and negative-threshold
cases (see Probes — all three live-oracle outputs are bit-identical to the
ferrolearn closure). The core dense transform therefore SHIPS (REQ-1). The
transform's **input-validation** contract also now SHIPS (REQ-9): `transform`
rejects zero-samples, zero-features, and non-finite (NaN/±inf) input in sklearn's
`check_array` order, fixed this iteration (#1123/#1124/#1125). The remaining
divergences are missing surface (`copy`, fit-time *parameter* validation, the
`binarize` free fn, feature names, sparse, PyO3, ferray), not a wrong dense
computation.

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — custom threshold 0.5, strict greater-than; ferrolearn matches via mapv:
python3 -c "from sklearn.preprocessing import Binarizer; \
print(Binarizer(threshold=0.5).transform([[0.4,0.6,0.5],[0.6,0.1,0.2]]).tolist())"
# -> [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]   (0.5 not > 0.5 -> 0; strict)
#    ferrolearn: Binarizer::<f64>::new(0.5).transform(&x) == same

# REQ-1 — default threshold 0.0; only positives map to 1 (0.0 not > 0.0 -> 0):
python3 -c "from sklearn.preprocessing import Binarizer; \
print(Binarizer().transform([[-1.0,0.0,0.5,1.0]]).tolist())"
# -> [[0.0, 0.0, 1.0, 1.0]]
#    ferrolearn: Binarizer::<f64>::default().transform(&x) == same

# REQ-1 — negative threshold -1.0 on dense input (no sparse restriction here):
python3 -c "from sklearn.preprocessing import Binarizer; \
print(Binarizer(threshold=-1.0).transform([[-2.0,-1.0,-0.5,0.0]]).tolist())"
# -> [[0.0, 0.0, 1.0, 1.0]]   (-1.0 not > -1.0 -> 0)
#    ferrolearn: Binarizer::<f64>::new(-1.0).transform(&x) == same

# REQ-6 — sparse + threshold<0 raises ValueError (no analog: ferrolearn dense-only):
python3 -c "from scipy.sparse import csr_matrix; from sklearn.preprocessing import binarize; \
binarize(csr_matrix([[1.0,-2.0]]), threshold=-1.0)"
# -> ValueError: Cannot binarize a sparse matrix with threshold < 0   (:2162-2163)

# REQ-3 — fit-time _parameter_constraints validation (R-DEV-2 exception type):
python3 -c "from sklearn.preprocessing import Binarizer; \
Binarizer(threshold='x').fit([[1.0]])"
# -> InvalidParameterError: The 'threshold' parameter of Binarizer must be an
#    instance of 'float'. Got 'x' instead.   (_parameter_constraints, :2248-2251)
```

## Requirements

- REQ-1: Dense threshold transform — strictly greater-than (`v > threshold` →
  1, else 0), shape-preserving, default threshold 0.0, matching sklearn
  `Binarizer(threshold=t).transform(X)` / `binarize` dense path
  (`_data.py:2170-2173`, `:2196-2197`) numerically on dense input for arbitrary
  (including negative) thresholds.
- REQ-2: `copy` constructor + `transform(X, copy=None)` parameter — in-place vs
  copy semantics (`__init__` `copy=True`, `:2253-2255`; `transform` `copy`
  override, `:2298-2307`).
- REQ-3: `fit(X, y=None)` that validates parameters via `_parameter_constraints`
  (`{threshold:[Real], copy:["boolean"]}`, `:2248-2251`) raising
  `InvalidParameterError` on a non-`Real` threshold (REQ-3 Probe), records
  `n_features_in_`/`feature_names_in_`, and returns `self` (`:2257-2278`); plus the
  stateless "recommend `fit_transform` because validation is only in `fit`"
  contract (`:2228-2231`).
- REQ-4: `binarize(X, *, threshold=0.0, copy=True)` standalone free function — the
  estimator-less API (`:2120-2174`); `@validate_params` rejects a
  closed-`neither` `Interval(Real)` threshold (`:2112-2118`).
- REQ-5: `n_features_in_` / `feature_names_in_` attributes set on fit, and
  `OneToOneFeatureMixin.get_feature_names_out` (one-to-one feature-name passthrough).
- REQ-6: Sparse-matrix support — CSR/CSC path operating on `X.data`, the
  `threshold < 0` → `ValueError` guard (REQ-6 Probe, `:2161-2163`), and
  `eliminate_zeros` (`:2168`).
- REQ-7: PyO3 binding (`import ferrolearn` exposes `Binarizer` mirroring
  `import sklearn`) — the project boundary consumer.
- REQ-8: ferray substrate — the transform computes over `ferray-core` arrays /
  `ferray-ufunc` rather than `ndarray::Array2` + `num_traits::Float` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `Binarizer::<f64>::new(0.5).transform(&x)` for
  `x = [[0.4,0.6,0.5],[0.6,0.1,0.2]]` equals the REQ-1 Probe output
  `[[0,1,0],[1,0,0]]` exactly; `Binarizer::default().transform([[-1,0,0.5,1]])`
  equals `[[0,0,1,1]]`; `Binarizer::new(-1.0).transform([[-2,-1,-0.5,0]])` equals
  `[[0,0,1,1]]`; output `shape() == x.shape()`. Pinned by an **oracle-grounded**
  `#[test]` whose expected values are the live-`sklearn` Probe outputs (R-CHAR-3).
- AC-2 (REQ-2): a `copy` field/param exists and `transform(X, copy=false)` is
  observably in-place. (Neither exists today.)
- AC-3 (REQ-3): `Binarizer::new(t).fit(&X)` validates the threshold and returns a
  fitted handle exposing `n_features_in_`; an invalid threshold yields the
  sklearn-matching error (REQ-3 Probe: `InvalidParameterError`).
- AC-4 (REQ-4): a free `binarize(&X, threshold, copy)` returns the same array as
  the equivalent `Binarizer::new(threshold).transform(&X)`.
- AC-5 (REQ-5): `get_feature_names_out` returns input feature names unchanged;
  `n_features_in_` is set after `fit`.
- AC-6 (REQ-6): a sparse `transform` binarizes only stored nonzeros and raises a
  `FerroError` on `threshold < 0` (REQ-6 Probe: `ValueError`).
- AC-7 (REQ-7): `python3 -c "import ferrolearn; ferrolearn.preprocessing.Binarizer"`
  resolves and `.transform` matches `sklearn` on the REQ-1 Probe.
- AC-8 (REQ-8): the owned transform computes on `ferray-core` arrays (no
  `ndarray`/`num_traits` in the compute path).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (dense strict-greater transform) | SHIPPED | impl `Transform::transform for Binarizer<F> in binarizer.rs` — `let out = x.mapv(\|v\| if v > self.threshold { F::one() } else { F::zero() }); Ok(out)` — strict greater-than, shape-preserving, infallible; `Default` sets threshold `F::zero()` (`impl Default for Binarizer<F> in binarizer.rs`). Mirrors sklearn `binarize` dense path `cond = X > threshold; X[cond]=1; X[not_cond]=0` (`_data.py:2170-2173`) and the docstring contract "values less than or equal to the threshold map to 0" (`:2180-2182`, `:2196-2197`). Output equals all three REQ-1 Probes bit-for-bit: `[[0,1,0],[1,0,0]]` (thr 0.5), `[[0,0,1,1]]` (default), `[[0,0,1,1]]` (thr -1.0). Non-test consumer: crate re-export `pub use binarizer::Binarizer;` (`ferrolearn-preprocess/src/lib.rs` line 106) — the boundary public API of the crate, grandfathered under S5/R-DEFER-1 (existing pub API across prior commits; the estimator type IS the public surface). Verification: `cargo test -p ferrolearn-preprocess` (`test_binarizer_*`) + oracle-grounded green guards in `tests/divergence_binarizer.rs` (`guard_binarizer_*`: threshold 0.5 / default / negative / f32 — bit-identical to live sklearn). |
| REQ-9 (transform input validation per check_array) | SHIPPED | FIXED #1123/#1124/#1125. `Binarizer::transform` now rejects, in sklearn's `check_array` order, zero samples (`x.nrows()==0` → `FerroError::InsufficientSamples`, mirrors `validation.py:1084` min-samples; #1124), zero features (`x.ncols()==0` → `FerroError::InvalidParameter`, mirrors `:1093` min-features; #1125), and non-finite input (`x.iter().any(\|v\| !v.is_finite())` → `FerroError::InvalidParameter`, mirrors `_assert_all_finite` `:1063`, force_all_finite=True; #1123) — matching sklearn `Binarizer.transform`'s `_validate_data` (`_data.py:2301`). Verification (acto-critic, live-oracle, all green): 13 tests in `tests/divergence_binarizer.rs` — 3 non-finite (NaN/+inf/-inf), empty-rows, zero-features all return `Err` matching sklearn `ValueError`; finite extremes (1e308/-0.0/subnormal/(1,1)) NOT over-rejected. Two-round critic-verified CLEAN (validation surface complete vs check_array defaults). |
| REQ-2 (copy parameter) | NOT-STARTED | open prereq blocker #1126. The struct holds only `threshold`; no `copy` field, no `copy` constructor arg, no `transform(X, copy)` override (sklearn `__init__` `:2253-2255`, `transform` `:2298-2307`). `mapv` always allocates a new array, so in-place semantics are unrepresentable today. |
| REQ-3 (fit + parameter validation) | NOT-STARTED | open prereq blocker #1127. No `Fit` impl, no fitted type, no `_parameter_constraints` analog; an invalid threshold cannot be rejected with `InvalidParameterError` (REQ-3 Probe) — ferrolearn never validates and never errors (sklearn `fit` `:2257-2278`, constraints `:2248-2251`). The "recommend `fit_transform` since validation is fit-only" contract (`:2228-2231`) is therefore absent. |
| REQ-4 (binarize free function) | NOT-STARTED | open prereq blocker #1128. No standalone `binarize` function exists in `binarizer.rs` or the crate; only the estimator method path is present (sklearn `binarize` `:2120-2174`). |
| REQ-5 (n_features_in_ / feature names) | NOT-STARTED | open prereq blocker #1129. No `n_features_in_`, no `feature_names_in_`, no `get_feature_names_out` (sklearn `OneToOneFeatureMixin` + `_validate_data` in `fit`, `:2277`). Depends on REQ-3 (these are set during `fit`). |
| REQ-6 (sparse support) | NOT-STARTED | open prereq blocker #1130. `transform` accepts only `Array2<F>`; no CSR/CSC path, no `X.data`-only update, no `threshold < 0` → error guard (REQ-6 Probe: `ValueError`), no `eliminate_zeros` (sklearn sparse path `:2161-2168`). |
| REQ-7 (PyO3 binding) | NOT-STARTED | open prereq blocker #1131. No `ferrolearn-python` registration; `import ferrolearn` cannot expose `Binarizer` (boundary consumer per R-DEFER-1). |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #1132. Compute path uses `ndarray::Array2` + `num_traits::Float` + `Array2::mapv`, not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** A single generic struct `Binarizer<F>`
(`binarizer.rs`) holding one field, `pub(crate) threshold: F`. Construction is
`pub fn new(threshold: F) -> Self`; a `threshold()` accessor returns it; `impl
Default for Binarizer<F>` yields `Self::new(F::zero())` (threshold 0.0, matching
sklearn's default — `:2253`). The only behavior is `impl Transform<Array2<F>>`
with `type Output = Array2<F>`, `type Error = FerroError`, whose `transform`
maps each element with `if v > self.threshold { F::one() } else { F::zero() }`
and wraps the result in `Ok` — element-wise, shape-preserving, never errors.
Generic bound `F: Float + Send + Sync + 'static` supports `f32` and `f64`. The
type is stateless; there is no unfitted/fitted split and no `Fit` impl (contrast
the sibling `StandardScaler`/`FittedStandardScaler` pattern), consistent with
sklearn's "this estimator is stateless and does not need to be fitted"
(`:2228`).

**sklearn (target contract).** `Binarizer(OneToOneFeatureMixin,
TransformerMixin, BaseEstimator)` (`:2177`) stores `threshold` and `copy`
(`:2253-2255`). `transform` (`:2280-2308`) runs `_validate_data(reset=False)`
then delegates to the `binarize` free function (`:2308`); `binarize`
(`:2120-2174`) does `check_array`, then for dense input `cond = X > threshold;
X[cond] = 1; X[not_cond] = 0` (`:2170-2173`) — the load-bearing **strict
greater-than**. For sparse input it guards `threshold < 0` with a `ValueError`
(`:2161-2163`), updates only `X.data` (`:2164-2167`), and calls
`eliminate_zeros` (`:2168`). `fit` (`:2257-2278`) is validate-only: it triggers
`_parameter_constraints` (`{threshold:[Real], copy:["boolean"]}`, `:2248-2251`)
via `@_fit_context`, sets `n_features_in_`/`feature_names_in_` through
`_validate_data`, and returns `self`. `OneToOneFeatureMixin` supplies a
one-to-one `get_feature_names_out`.

**The structural gap.** ferrolearn's dense computation is an exact match for
sklearn's dense `binarize` (the strict `>` and default-0 semantics coincide —
all REQ-1 Probes are bit-identical). What is missing is *surface*: the `copy`
parameter (REQ-2), the validating `fit` + fitted type + `InvalidParameterError`
(REQ-3), the standalone `binarize` free fn (REQ-4), feature-name plumbing
(REQ-5), the entire sparse path including the `threshold < 0` guard (REQ-6), the
PyO3 binding (REQ-7), and the ferray substrate (REQ-8). None of these change the
dense numeric result; they extend the API to the full sklearn contract.

## Verification

Commands establishing the single SHIPPED claim (REQ-1):

```bash
# Oracle (REQ-1 Probes) — dense, strict greater-than, sklearn outputs:
python3 -c "from sklearn.preprocessing import Binarizer; \
print(Binarizer(threshold=0.5).transform([[0.4,0.6,0.5],[0.6,0.1,0.2]]).tolist())"
#   -> [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
python3 -c "from sklearn.preprocessing import Binarizer; \
print(Binarizer().transform([[-1.0,0.0,0.5,1.0]]).tolist())"
#   -> [[0.0, 0.0, 1.0, 1.0]]
python3 -c "from sklearn.preprocessing import Binarizer; \
print(Binarizer(threshold=-1.0).transform([[-2.0,-1.0,-0.5,0.0]]).tolist())"
#   -> [[0.0, 0.0, 1.0, 1.0]]
# ferrolearn equivalents: Binarizer::<f64>::new(0.5)/::default()/::new(-1.0).transform(&x)

# Crate gauntlet:
cargo test -p ferrolearn-preprocess        # incl. test_binarizer_default_threshold,
                                           #       test_binarizer_custom_threshold,
                                           #       test_binarizer_negative_threshold
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s (`test_binarizer_default_threshold`,
`test_binarizer_custom_threshold`, `test_binarizer_all_zeros`,
`test_binarizer_all_ones`, `test_binarizer_negative_threshold`,
`test_binarizer_multiple_rows`, `test_binarizer_preserves_shape`,
`test_binarizer_f32`, `test_output_values_are_zero_or_one`) exercise the REQ-1
dense path comprehensively but are **NOT oracle-grounded** — their expected
values are hand-written, not live-derived from sklearn. To satisfy R-CHAR-3 the
critic should add an oracle-pinned green guard (e.g.
`guard_binarizer_threshold_matches_sklearn_oracle`) asserting the three REQ-1
Probe outputs. No currently-green command establishes any of REQ-2..REQ-8.

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns
`#`-numbers); reference them in the REQ status table:

- #1126 — REQ-2: no `copy` parameter on the constructor or `transform`; `mapv`
  always allocates, so in-place semantics (`:2298-2307`) are unrepresentable.
- #1127 — REQ-3: no `fit`, no fitted type, no `_parameter_constraints`
  validation / `InvalidParameterError` (REQ-3 Probe), no fit-only-validation
  contract (`:2228-2231`, `:2248-2278`).
- #1128 — REQ-4: no standalone `binarize` free function (`:2120-2174`).
- #1129 — REQ-5: no `n_features_in_` / `feature_names_in_` /
  `get_feature_names_out` (OneToOneFeatureMixin; set on `fit`, `:2277`). Depends
  on #1127.
- #1130 — REQ-6: dense-only; no CSR/CSC path, no `threshold < 0` →
  `ValueError` guard (REQ-6 Probe), no `eliminate_zeros` (`:2161-2168`).
- #1131 — REQ-7: no `ferrolearn-python` registration of `Binarizer`.
- #1132 — REQ-8: compute path on `ndarray`/`num_traits`, not ferray
  (R-SUBSTRATE-1/2).
```
