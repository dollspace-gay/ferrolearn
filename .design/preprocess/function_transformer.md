# FunctionTransformer

<!--
tier: 3-component
status: draft
baseline-commit: 8bb833f9
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_function_transformer.py  # _identity(X)=X (:22-24); class FunctionTransformer(TransformerMixin, BaseEstimator) (:27); _parameter_constraints (:141-150); __init__(func=None, inverse_func=None, *, validate=False, accept_sparse=False, check_inverse=True, feature_names_out=None, kw_args=None, inv_kw_args=None) (:152-171); _check_input (reset sets n_features_in_/feature_names_in_ even when validate=False, :173-182); _check_inverse_transform (warns on non-inverse roundtrip, :184-210); fit (:213-235); transform (:237-307); inverse_transform (:309-325); get_feature_names_out (@available_if feature_names_out is not None, :327-373); _transform(X, func, kw_args): func is None -> _identity; return func(X, **kw_args) (:375-379); __sklearn_is_fitted__ -> True (:381-383); _more_tags stateless (:385-386); set_output pandas/polars (:388-416)
ferrolearn-module: ferrolearn-preprocess/src/function_transformer.rs
parity-ops: FunctionTransformer
crosslink-issue: 1111
-->

## Summary

scikit-learn's `FunctionTransformer` (`_function_transformer.py:27`) constructs
a transformer from an **arbitrary array-to-array callable**: `func` is passed
the **whole** `X` and returns the transformed array (`_transform`:
`return func(X, **(kw_args or {}))`, `:375-379`). It is stateless, supports an
`inverse_func`/`inverse_transform` pair, optional input validation
(`validate`/`accept_sparse`), an inverse-consistency sanity check
(`check_inverse` + `fit`), feature-name plumbing (`feature_names_out`,
`get_feature_names_out`, `n_features_in_`/`feature_names_in_`), keyword-argument
forwarding (`kw_args`/`inv_kw_args`), and `set_output` container selection.

`ferrolearn-preprocess/src/function_transformer.rs` ships a **thin, stateless,
element-wise** wrapper: `pub struct FunctionTransformer<F> { func: Box<dyn Fn(F)
-> F + Send + Sync> }` holding a **scalar** closure that is applied to every
element via `Array2::mapv` in `Transform::transform`. There is no `fit`, no
fitted type, no `inverse_func`/`inverse_transform`, no `validate`, no
`feature_names_out`, no `kw_args`, no `func=None` identity default (the caller
must supply a closure), and no PyO3 binding.

**Headline semantic divergence (document prominently):** sklearn's `func` is
**array → array** (`func(X)`); ferrolearn's `func` is **scalar → scalar**
applied element-wise. For element-wise ufuncs (`np.log1p`, `np.sqrt`, `np.abs`,
negation, scalar multiply, clamp) the numeric output **matches** sklearn exactly
(see Probes — `FunctionTransformer(np.log1p)` equals `FunctionTransformer::new(|v|
v.ln_1p())` element-wise). For **whole-array** funcs (column sums, reshaping,
row-wise normalization, any output whose shape differs from the input) the
ferrolearn type **structurally cannot express them** — the closure signature
`Fn(F) -> F` has no access to the array. The forward element-wise path is
SHIPPED-scoped to the ufunc subset; everything else is NOT-STARTED.

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — element-wise ufunc func; ferrolearn matches via mapv:
python3 -c "import numpy as np; from sklearn.preprocessing import FunctionTransformer; \
print(FunctionTransformer(np.log1p).transform(np.array([[0.,1.],[2.,3.]])).tolist())"
# -> [[0.0, 0.6931471805599453], [1.0986122886681096, 1.3862943611198906]]
#    ferrolearn: FunctionTransformer::<f64>::new(|v| v.ln_1p()).transform(&x) == same

# REQ-2 — func=None identity default (no analog: ferrolearn requires a closure):
python3 -c "import numpy as np; from sklearn.preprocessing import FunctionTransformer; \
print(FunctionTransformer().transform(np.array([[0.,1.],[2.,3.]])).tolist())"
# -> [[0.0, 1.0], [2.0, 3.0]]   (func is None -> _identity, :376-377)

# REQ-3 — whole-array func (column sum); ferrolearn structurally cannot express:
python3 -c "import numpy as np; from sklearn.preprocessing import FunctionTransformer; \
print(FunctionTransformer(lambda X: X.sum(axis=0)).transform(np.array([[0.,1.],[2.,3.]])).tolist())"
# -> [2.0, 4.0]   (shape (2,) != input shape (2,2))

# REQ-9 — _parameter_constraints reject non-callable func (R-DEV-2 exception type):
python3 -c "import numpy as np; from sklearn.preprocessing import FunctionTransformer; \
FunctionTransformer(func='nope').transform(np.array([[1.]]))"
# -> TypeError: 'str' object is not callable
```

## Requirements

- REQ-1: Forward element-wise transform — for a scalar ufunc closure, apply it
  to every element of a 2-D input and return an array of the same shape, matching
  sklearn's `FunctionTransformer(<ufunc>).transform(X)` numerically.
- REQ-2: `func=None` identity default and the **array → array** `func` contract
  (`func(X)` receives the whole array; `_identity(X)=X`, `:22-24`, `:375-379`).
- REQ-3: Whole-array `func` support (output shape may differ from input — column
  sums, reshapes, row-normalization).
- REQ-4: `inverse_func` + `inverse_transform` (`:309-325`).
- REQ-5: `validate` / `accept_sparse` input checking (2-D coercion / sparse
  acceptance / exception on failure, `_check_input` + `_validate_data`, `:173-182`).
- REQ-6: `check_inverse` + `fit` — `fit` no-op that runs `_check_input(reset=True)`
  and (when `func` and `inverse_func` both set) `_check_inverse_transform`
  (warns `UserWarning` on non-inverse roundtrip, `:184-210`, `:213-235`); plus
  `__sklearn_is_fitted__ -> True` stateless semantics (`:381-383`).
- REQ-7: `feature_names_out` / `get_feature_names_out` (`'one-to-one'` or
  callable, `@available_if`, `:327-373`) and the `n_features_in_` /
  `feature_names_in_` attributes set on fit (`:176-181`).
- REQ-8: `kw_args` / `inv_kw_args` keyword forwarding to `func` / `inverse_func`
  (`:93-101`, `:379`).
- REQ-9: Constructor parameter surface + `_parameter_constraints` (`:141-171`) —
  the eight named params, `*`-only kwargs, and validation exception types
  (`TypeError`/`InvalidParameterError`) per R-DEV-2.
- REQ-10: PyO3 binding (`import ferrolearn` exposes `FunctionTransformer`
  mirroring `import sklearn`) — the project boundary consumer.
- REQ-11: ferray substrate — `func` over `ferray-core` arrays / `ferray-ufunc`
  rather than `ndarray::Array2` + `num_traits::Float` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `FunctionTransformer::<f64>::new(|v| v.ln_1p()).transform(&x)` for
  `x = [[0,1],[2,3]]` equals the REQ-1 Probe output element-wise to ULP tolerance;
  output `shape() == x.shape()`. Pinned by an oracle-grounded `#[test]`.
- AC-2 (REQ-2): a `FunctionTransformer::default()` / `::identity()` exists whose
  `transform` is the identity, AND the stored `func` can be called as `func(&X)`
  on a whole array. (Neither exists today.)
- AC-3 (REQ-3): a column-sum `func` returns shape `(2,)` from a `(2,2)` input
  (matches REQ-3 Probe). (Impossible under `Fn(F) -> F`.)
- AC-4 (REQ-4): `inverse_transform(transform(X)) == X` for an invertible pair.
- AC-5 (REQ-5): `validate=true` rejects a non-2-D / ragged input with a `FerroError`.
- AC-6 (REQ-6): `fit` returns a fitted handle; with non-inverse `func`/`inverse_func`
  and `check_inverse=true`, a warning is surfaced; `is_fitted() == true` pre-fit.
- AC-7 (REQ-7): `get_feature_names_out` returns input names for `'one-to-one'`;
  `n_features_in_` is set after `fit`.
- AC-8 (REQ-8): `kw_args = {base: 10}` forwarded to a `func(X, base)` changes output.
- AC-9 (REQ-9): constructing with a non-callable / invalid param yields the
  sklearn-matching error type (REQ-9 Probe: `TypeError`).
- AC-10 (REQ-10): `python3 -c "import ferrolearn; ferrolearn.preprocessing.FunctionTransformer"`
  resolves and `.transform` matches `sklearn` on the REQ-1 Probe.
- AC-11 (REQ-11): the owned transform computes on `ferray-core` arrays (no
  `ndarray`/`num_traits` in the compute path).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (element-wise forward transform) | SHIPPED (scoped) | impl `Transform::transform for FunctionTransformer<F> in function_transformer.rs` — `let out = x.mapv(\|v\| (self.func)(v)); Ok(out)` — applies the scalar closure element-wise, shape-preserving, never errors. Mirrors sklearn `_transform` (`_function_transformer.py:375-379`) **only for element-wise ufunc `func`** (output equals the REQ-1 Probe `[[0.0, 0.693…],[1.098…, 1.386…]]` for `\|v\| v.ln_1p()`). Non-test consumer: crate re-export `pub use function_transformer::FunctionTransformer;` (`ferrolearn-preprocess/src/lib.rs` line 114) + the module-doc index entry (lib.rs line 48) — the boundary public API surface of the crate, grandfathered under S5/R-DEFER-1 (existing pub API across prior commits). Verification: oracle Probe REQ-1 vs `mapv` closure; `cargo test -p ferrolearn-preprocess` (`test_ln_function`, `test_sqrt_function`, `test_preserves_shape`) + oracle-grounded green guards `guard_log1p_matches_sklearn_oracle`/`guard_expm1_…`/`guard_sqrt_…`/`guard_log_nan_inf_propagation_…`/`guard_empty_matrix_shape_…` in `tests/divergence_function_transformer.rs` (5 passed, bit-identical to live sklearn). **Caveat: scalar `Fn(F) -> F` contract, NOT sklearn's array `Fn(X) -> X`; correct only on the element-wise/ufunc subset.** |
| REQ-2 (func=None identity + array→array contract) | NOT-STARTED | open prereq blocker #1112. `new` requires a `Func: Fn(F) -> F` closure (no `Default`/`None` path), and the closure is scalar-valued so the whole-array `func(X)` contract (`_identity(X)=X`, `:22-24`; `_transform`, `:375-379`) is unrepresentable. |
| REQ-3 (whole-array func) | NOT-STARTED | open prereq blocker #1113. The stored signature `Box<dyn Fn(F) -> F>` has no access to the array; column-sum/reshape funcs (REQ-3 Probe → shape `(2,)`) cannot be expressed. Requires changing the closure type to `Fn(&Array) -> Result<Array>`. |
| REQ-4 (inverse_func / inverse_transform) | NOT-STARTED | open prereq blocker #1114. No `inverse_func` field, no `inverse_transform` method (sklearn `:309-325`). |
| REQ-5 (validate / accept_sparse) | NOT-STARTED | open prereq blocker #1115. `transform` performs no input validation; no `validate`/`accept_sparse` params; sparse unsupported (`_check_input`/`_validate_data`, `:173-182`). |
| REQ-6 (fit / check_inverse / is_fitted) | NOT-STARTED | open prereq blocker #1116. No `fit`, no fitted type, no `check_inverse`, no `__sklearn_is_fitted__` analog (sklearn `fit` `:213-235`, `_check_inverse_transform` `:184-210`, `:381-383`). |
| REQ-7 (feature_names_out / n_features_in_) | NOT-STARTED | open prereq blocker #1117. No `feature_names_out`, no `get_feature_names_out`, no `n_features_in_`/`feature_names_in_` (sklearn `:327-373`, `:176-181`). |
| REQ-8 (kw_args / inv_kw_args) | NOT-STARTED | open prereq blocker #1118. No keyword-argument forwarding; closure captures its environment instead (sklearn `:93-101`, `:379`). |
| REQ-9 (ctor surface + _parameter_constraints) | NOT-STARTED | open prereq blocker #1119. Only `func` exists; the other seven sklearn params (`inverse_func`, `validate`, `accept_sparse`, `check_inverse`, `feature_names_out`, `kw_args`, `inv_kw_args`) and `_parameter_constraints` validation (REQ-9 Probe → `TypeError`) are absent (R-DEV-2). |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #1120. No `ferrolearn-python` registration; `import ferrolearn` cannot expose this transformer (boundary consumer per R-DEFER-1). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #1121. Compute path uses `ndarray::Array2` + `num_traits::Float` + `Array2::mapv`, not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** A single generic struct `FunctionTransformer<F>`
(`function_transformer.rs`) holding `func: Box<dyn Fn(F) -> F + Send + Sync>`.
Construction is `pub fn new<Func>(func) where Func: Fn(F) -> F + Send + Sync +
'static` — note **no `func=None` default**; a closure is mandatory. The only
behavior is `impl Transform<Array2<F>>` with `type Output = Array2<F>`,
`type Error = FerroError`, whose `transform` calls `x.mapv(|v| (self.func)(v))`
and wraps it in `Ok` — element-wise, shape-preserving, infallible. A `Debug`
impl prints `<fn(F) -> F>` for the opaque closure. Generic bound `F: Float +
Send + Sync + 'static` supports both `f32` and `f64`. The type is stateless;
there is no unfitted/fitted split (contrast the sibling `StandardScaler` /
`FittedStandardScaler` pattern), consistent with sklearn's
`__sklearn_is_fitted__ -> True` (`:381-383`) — but ferrolearn has no `Fit`
impl at all, so there is nothing to be (un)fitted.

**sklearn (target contract).** `FunctionTransformer(TransformerMixin,
BaseEstimator)` (`:27`) stores eight params (`:152-171`). The transform pipeline
is `transform` → `_check_input(reset=False)` → `_transform(X, func, kw_args)`
→ set_output/feature-name plumbing (`:237-307`); `_transform` is the load-bearing
line `return func(X, **(kw_args or {}))` (`:375-379`) — **`func` is called on the
whole array**, defaulting to `_identity` when `None`. `inverse_transform`
(`:309-325`) is the symmetric path through `inverse_func`. `fit` (`:213-235`) is
a near no-op: it runs `_check_input(reset=True)` (which sets
`n_features_in_`/`feature_names_in_` even when `validate=False`, `:176-181`) and,
when both funcs are set and `check_inverse`, calls `_check_inverse_transform`
(`:184-210`) which round-trips a subsample and emits a `UserWarning` if it is
not an identity. `get_feature_names_out` is gated by `@available_if(feature_names_out
is not None)` (`:327`).

**The structural gap.** ferrolearn's `Fn(F) -> F` is a strict subset of
sklearn's `Fn(X) -> X`: every scalar ufunc embeds as an element-wise array func
(so REQ-1 matches), but no array func that reads multiple elements (sums,
reshapes, normalization) or changes shape embeds back into a scalar closure
(REQ-3). A faithful translation must change the stored type to an array-level
closure (`Box<dyn Fn(&Array2<F>) -> Result<Array2<F>, FerroError>>` or a ferray
analog), at which point REQ-2/REQ-4/REQ-8 become expressible. This is the
headline blocker (#1113).

## Verification

Commands establishing the single SHIPPED-scoped claim (REQ-1):

```bash
# Oracle (REQ-1 Probe) — element-wise ufunc, sklearn output:
python3 -c "import numpy as np; from sklearn.preprocessing import FunctionTransformer; \
print(FunctionTransformer(np.log1p).transform(np.array([[0.,1.],[2.,3.]])).tolist())"
#   -> [[0.0, 0.6931471805599453], [1.0986122886681096, 1.3862943611198906]]
# ferrolearn equivalent: FunctionTransformer::<f64>::new(|v| v.ln_1p()).transform(&array![[0.,1.],[2.,3.]])

# Crate gauntlet:
cargo test -p ferrolearn-preprocess        # incl. test_ln_function, test_sqrt_function, test_preserves_shape, test_empty_matrix
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s (`test_identity_function`, `test_sqrt_function`,
`test_ln_function`, `test_negate_function`, `test_constant_function`,
`test_preserves_shape`, `test_clamp_function`, `test_f32_function`,
`test_closure_captures_environment`, `test_empty_matrix`) exercise only the
REQ-1 element-wise path. They are NOT oracle-grounded against sklearn (the
expected values are hand-written, not live-derived) — the critic should add an
oracle-pinned `#[test]` matching the REQ-1 Probe to satisfy R-CHAR-3. No
currently-green command establishes any of REQ-2..REQ-11.

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns
`#`-numbers); reference them in the REQ status table:

- #1112 — REQ-2: no `func=None` identity default; scalar closure
  cannot express `func(X)` whole-array / `_identity` contract.
- #1113 — REQ-3 (headline): stored `Fn(F) -> F` cannot read the array
  or change shape; needs `Fn(&Array) -> Result<Array>` (or ferray analog).
- #1114 — REQ-4: no `inverse_func` / `inverse_transform`.
- #1115 — REQ-5: no `validate` / `accept_sparse` / input checking.
- #1116 — REQ-6: no `fit` / `check_inverse` / `__sklearn_is_fitted__`.
- #1117 — REQ-7: no `feature_names_out` / `get_feature_names_out` /
  `n_features_in_` / `feature_names_in_`.
- #1118 — REQ-8: no `kw_args` / `inv_kw_args` forwarding.
- #1119 — REQ-9: ctor exposes only `func`; missing seven params +
  `_parameter_constraints` validation / exception types (R-DEV-2).
- #1120 — REQ-10: no `ferrolearn-python` registration.
- #1121 — REQ-11: compute path on `ndarray`/`num_traits`, not ferray.
```
