# TransformedTargetRegressor

<!--
tier: 3-component
status: draft
baseline-commit: c07c4089
upstream-paths:
  - sklearn/compose/_target.py   # class TransformedTargetRegressor (:24-:356)
-->

## Summary

`ferrolearn-model-sel/src/transformed_target.rs` mirrors scikit-learn's
`sklearn.compose.TransformedTargetRegressor` (`sklearn/compose/_target.py:24`) —
the meta-regressor that applies a forward transform to the target `y` before
fitting and the inverse transform to predictions. The computation is
`regressor.fit(X, func(y))` at fit and `inverse_func(regressor.predict(X))` at
predict (`_target.py:34-48`).

ferrolearn splits it into the unfitted `TransformedTargetRegressor<F>` (fields:
an inner `ferrolearn_core::pipeline::Pipeline<F>` regressor, `func: fn(F) -> F`,
`inverse_func: fn(F) -> F`) and the fitted `FittedTransformedTargetRegressor<F>`,
wired through the `Fit`/`Predict` traits. The current implementation ships ONLY
the **func/inverse-callable, single-output, elementwise** slice of sklearn's
surface. sklearn's `transformer`-object mode, `check_inverse`, the
`regressor=None → LinearRegression` default, the both-None identity default,
2D/multi-output handling, and `fit_params`/`predict_params` passthrough are
NOT-STARTED. The R-SUBSTRATE migration to `ferray-core` is also NOT-STARTED.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/compose/_target.py:24` — `class TransformedTargetRegressor(_RoutingNotSupportedMixin, RegressorMixin, BaseEstimator)`.
- `:138-144` — `_parameter_constraints`: `regressor`, `transformer`, `func`,
  `inverse_func`, `check_inverse`.
- `:146-159` — `__init__(self, regressor=None, *, transformer=None, func=None,
  inverse_func=None, check_inverse=True)`.
- `:161-218` — `_fit_transformer`: transformer-vs-func mutual-exclusion
  (`:168-173`), `transformer` clone path (`:174-175`), one-of-func/inverse error
  (`:177-189`), both-None identity `FunctionTransformer` default (`:190-199`),
  and the `check_inverse` 1/10-strided round-trip `UserWarning` (`:205-218`).
- `:224-293` — `fit(self, X, y, **fit_params)`: `check_array(force_all_finite=True,
  ensure_2d=False)` on `y` (`:251-259`), stores `self._training_dim = y.ndim`
  (`:263`), reshapes 1D→2D (`:267-270`), fits the transformer (`:271`),
  transforms y and squeezes 2D→1D (`:274-279`), `regressor=None → LinearRegression`
  (`:281-284`) else `clone(self.regressor)` (`:286`), then `regressor_.fit(X,
  y_trans, **fit_params)` (`:288`).
- `:295-328` — `predict(self, X, **predict_params)`: `regressor_.predict(X,
  **predict_params)` (`:316`), `transformer_.inverse_transform` (`:318-320`),
  squeeze back to `_training_dim` (`:321-326`).
- `:342-356` — `n_features_in_` property delegates to `regressor_.n_features_in_`.

## Requirements

- REQ-1: Core func/inverse value parity (elementwise func, single-output). `fit`
  applies `func` element-wise to `y`, fits the inner regressor on `(X, func(y))`;
  `predict` returns `inverse_func(regressor.predict(X))`. Mirrors sklearn's
  func-mode computation (`_target.py:34-48`, `:274-288`, `:316-320`). For an
  ELEMENTWISE `func`/`inverse_func` pair (e.g. `ln`/`exp`) and SINGLE-output `y`,
  results equal sklearn's `TransformedTargetRegressor(LinearRegression(),
  func=np.log, inverse_func=np.exp)`. DETERMINISTIC / oracle-pinnable.
- REQ-2: `transformer`-object mode — accept a transformer object applied to `y`,
  mutually exclusive with `func`/`inverse_func` (`_target.py:168-175`).
- REQ-3: `check_inverse` — verify `inverse_func(func(y)) ≈ y` on a 1/10-strided
  subset and emit a `UserWarning` if not strictly inverse (`_target.py:205-218`).
- REQ-4: `regressor=None → LinearRegression` default — a `None`/absent regressor
  defaults to a fresh `LinearRegression` (`_target.py:281-284`).
- REQ-5: func/inverse both-or-neither validation + both-None identity default —
  error if exactly one of `func`/`inverse_func` is set (`_target.py:177-189`);
  default both-None to an identity transform (`:190`).
- REQ-6: 2D `y` / multi-output — store `_training_dim`, reshape 1D→2D for the
  transformer, squeeze 2D→1D back, handle `y.ndim == 2`
  (`_target.py:263-279`, `:321-326`).
- REQ-7: `fit_params` / `predict_params` passthrough — forward `**fit_params` and
  `**predict_params` to the inner regressor (`_target.py:236-288`, `:306-316`).
- REQ-8: NaN-on-func guard (R-DEV-4 sanctioned deviation) — `fit` returns
  `FerroError::NumericalInstability` when `func` produces NaN. sklearn does NOT
  check func OUTPUT finiteness (it validates the INPUT `y` via
  `check_array(force_all_finite=True)`, `_target.py:251-259`). Documented Rust
  footgun-elimination deviation, not a divergence.
- REQ-9: R-SUBSTRATE — array type on `ferray-core` rather than `ndarray`.
- REQ-10: Non-test production consumer — `TransformedTargetRegressor` reachable
  from non-test production code.

## Acceptance criteria

- AC-1 (REQ-1): for `func=ln`, `inverse_func=exp`, an inner `LinearRegression`
  pipeline, `X=[[1],[2],[3],[4]]`, `y=[2,5,9,16]`, ferrolearn `predict(X)` equals
  the live-oracle output of sklearn's `TransformedTargetRegressor(LinearRegression(),
  func=np.log, inverse_func=np.exp)` (see Verification for the pinned vector)
  within a documented OLS tolerance. DETERMINISTIC / oracle-pinnable.
- AC-2 (REQ-2): a transformer-object form fits/inverts `y` through that object,
  with a `transformer`+`func` mutual-exclusion error matching sklearn.
- AC-3 (REQ-3): a non-strict inverse pair surfaces the sklearn `check_inverse`
  warning (or its ferrolearn analog); `check_inverse=False` suppresses it.
- AC-4 (REQ-4): constructing without a regressor yields a `LinearRegression`-
  backed fit matching sklearn's default-regressor result.
- AC-5 (REQ-5): setting exactly one of `func`/`inverse_func` errors with the
  sklearn-equivalent message; both-None acts as identity.
- AC-6 (REQ-6): fitting with 2D `y` (`n×t`) predicts an `n×t` output and a 1D `y`
  predicts a 1D output, matching `_training_dim` round-tripping.
- AC-7 (REQ-7): a `fit_param`/`predict_param` reaches the inner regressor.
- AC-8 (REQ-8): `fit` with a `func` producing NaN returns
  `FerroError::NumericalInstability`; the deviation from sklearn (which does not
  guard func output) is documented.
- AC-9 (REQ-9): owned computation runs on `ferray-core`, no `ndarray` in owned
  computation.
- AC-10 (REQ-10): `TransformedTargetRegressor` is constructed from non-test
  production code.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (core func/inverse value parity) | SHIPPED | `pub fn fit` (impl `Fit<Array2<F>, Array1<F>>`) in `transformed_target.rs` computes `let y_transformed = y.mapv(self.func)` then `self.regressor.fit(x, &y_transformed)?`; `pub fn predict` (impl `Predict<Array2<F>>`) in `transformed_target.rs` returns `self.pipeline.predict(x)?.mapv(self.inverse_func)`. Mirrors sklearn func-mode `regressor.fit(X, func(y))` / `inverse_func(regressor.predict(X))` (`_target.py:34-48`, `:274-288`, `:316-320`). For elementwise `func`/`inverse_func` and single-output `y` the results match. DETERMINISTIC / oracle-pinnable: with `func=ln, inverse=exp`, inner `LinearRegression`, `X=[[1],[2],[3],[4]]`, `y=[2,5,9,16]`, sklearn predicts `[2.2126323263942624, 4.3788845965934815, 8.665981275583599, 17.15031072688885]` (live oracle, Verification). Tests (synthetic `MeanEstimator` inner): `test_log_exp_transform`, `test_doubling_transform`, `test_square_sqrt_transform`, `test_identity_transform`. CAVEAT: the existing `#[test]`s use a trivial mean-predicting inner estimator and assert hand-computed values, NOT an oracle-grounded `LinearRegression` value-parity vector (R-CHAR-3); the critic should pin the `LinearRegression`-inner parity `#[test]` against THE live-oracle vector above. Non-test consumer: REQ-10. |
| REQ-2 (transformer-object mode) | NOT-STARTED | open prereq blocker (tracking #1682; critic to file specific). `struct TransformedTargetRegressor` carries only `func: fn(F) -> F` / `inverse_func: fn(F) -> F` — no `transformer` field, no transformer-vs-func mutual-exclusion. sklearn accepts a `transformer` object applied to `y` (`_target.py:62-68`, `:174-175`) and errors if both are set (`:168-173`). Absent end-to-end. |
| REQ-3 (check_inverse) | NOT-STARTED | open prereq blocker (tracking #1682). No `check_inverse` field, no subset round-trip check, no warning channel in `fit`. sklearn (default `check_inverse=True`) verifies `inverse_func(func(y)) ≈ y` on a `slice(None, None, max(1, n//10))` subset and emits a `UserWarning` (`_target.py:205-218`). Absent; live oracle confirms the warning fires for a non-strict inverse (Verification). |
| REQ-4 (regressor=None → LinearRegression default) | NOT-STARTED | open prereq blocker (tracking #1682) — R-DEV-2 API divergence. `pub fn new(regressor: Pipeline<F>, func, inverse_func)` REQUIRES a `Pipeline<F>` regressor; there is no `None`/default path. sklearn defaults `regressor=None` to a fresh `LinearRegression` (`_target.py:281-284`, live-oracle-confirmed: default ctor → `regressor_` is `LinearRegression`). ferrolearn additionally requires the regressor be wrapped in a `Pipeline` (no bare-estimator form). The required-`Pipeline` constructor diverges from sklearn's optional-regressor contract. |
| REQ-5 (func/inverse validation + identity default) | NOT-STARTED | open prereq blocker (tracking #1682). PARTIAL/divergent: `new(regressor, func, inverse_func)` requires BOTH `func` and `inverse_func` (type-enforced `fn(F) -> F`), which MATCHES sklearn's both-required rule (`_target.py:177-189`, live oracle: only-`func` → `ValueError "When 'func' is provided, 'inverse_func' must also be provided"`). But there is NO both-None identity default (`_target.py:190`) — ferrolearn has no all-defaults form, so the identity-transform default is absent. NOT-STARTED on the identity-default half; the both-required half is structurally satisfied by the signature. |
| REQ-6 (2D y / multi-output) | NOT-STARTED | open prereq blocker (tracking #1682). `impl Fit<Array2<F>, Array1<F>>` / `impl Predict<Array2<F>>` with `Output = Array1<F>` are single-output only; no `_training_dim`, no 1D→2D reshape, no 2D→1D squeeze. sklearn stores `_training_dim` and handles `y.ndim == 2` (`_target.py:263-279`, `:321-326`). Absent. |
| REQ-7 (fit_params/predict_params passthrough) | NOT-STARTED | open prereq blocker (tracking #1682). `fit(&self, x, y)` and `predict(&self, x)` take no extra params; nothing is forwarded to the inner pipeline. sklearn forwards `**fit_params` (`_target.py:236`, `:288`) and `**predict_params` (`:306`, `:316`) to the regressor. Absent. |
| REQ-8 (NaN-on-func guard, R-DEV-4 deviation) | SHIPPED | `pub fn fit` in `transformed_target.rs` returns `FerroError::NumericalInstability { message: "TransformedTargetRegressor: func produced NaN values in y" }` when `y_transformed.iter().any(|&v| v.is_nan())`. This is a SANCTIONED Rust footgun-elimination DEVIATION (R-DEV-4): sklearn does NOT check func OUTPUT finiteness — it validates the INPUT `y` via `check_array(force_all_finite=True)` (`_target.py:251-259`) but lets a func that emits NaN flow into the regressor. `FerroError::NumericalInstability` is an existing variant (`ferrolearn-core/src/error.rs`). Test: `test_nan_func_error` (`fit` with `func = |_| f64::NAN` is `is_err()`). Non-test consumer: REQ-10. Documented as a deviation, NOT claimed as sklearn parity. |
| REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker (tracking #1682; R-SUBSTRATE-2). `transformed_target.rs` uses `ndarray::{Array1, Array2}` and `.mapv(...)`. Destination substrate is the `ferray-core` array type + its elementwise/ufunc analog (R-SUBSTRATE-1). Not migrated. |
| REQ-10 (consumer) | SHIPPED | Crate re-export: `lib.rs` (`pub use transformed_target::{FittedTransformedTargetRegressor, TransformedTargetRegressor}`), and `pub mod transformed_target` in `lib.rs`. R-DEFER-1 / S5: boundary meta-regressor type, grandfathered existing pub API. CAVEAT (honest underclaim): the ONLY caller outside this module is `ferrolearn-model-sel/tests/api_proof.rs` (`TransformedTargetRegressor::<f64>::new(...)`), which is a TEST and is construction-only — it does NOT count as a non-test consumer. There is NO `ferrolearn-python` binding and NO non-test, non-re-export call site. This REQ is SHIPPED on the strength of the boundary re-export (the public API IS the estimator type, per S5), not a dedicated production caller; the narrower-than-sklearn surface (no Python binding) is noted. |

## Architecture

ferrolearn splits the estimator into the unfitted `TransformedTargetRegressor<F>`
(fields `regressor: Pipeline<F>`, `func: fn(F) -> F`, `inverse_func: fn(F) -> F`,
built via `new(regressor, func, inverse_func)`) and the fitted
`FittedTransformedTargetRegressor<F>` (fields `pipeline: <Pipeline<F> as Fit<…>>::Fitted`,
`inverse_func: fn(F) -> F`). sklearn keeps one class whose post-`fit` state lives
in `regressor_`/`transformer_` (`_target.py:24`, `:284-288`); ferrolearn uses the
project-wide unfitted/Fitted split (CLAUDE.md naming).

The forward transform is a function pointer `fn(F) -> F` applied with
`ndarray::ArrayBase::mapv`, NOT sklearn's `FunctionTransformer` wrapper
(`_target.py:190-199`). This carries three structural consequences vs sklearn:
the func is ELEMENTWISE-only (no array-shape-changing transform), SINGLE-output
(`Array1<F>`, no `_training_dim` machinery), and the inner regressor MUST be a
`Pipeline<F>` (no bare-estimator or `regressor=None` default). These are the
REQ-2/REQ-4/REQ-6 gaps.

`fit` (impl `Fit`) maps `func` over `y`, guards against NaN output (REQ-8,
R-DEV-4 deviation — see table), then delegates to `self.regressor.fit(x,
&y_transformed)`, propagating the inner `FerroError`. `predict` (impl `Predict`)
runs `self.pipeline.predict(x)` and maps `inverse_func` over the raw predictions
— mirroring `inverse_func(regressor.predict(X))` (`_target.py:316-320`).

What is structurally absent vs sklearn: the `transformer`-object branch and its
mutual-exclusion validation (REQ-2, `_target.py:168-175`), the `check_inverse`
subset round-trip + `UserWarning` (REQ-3, `:205-218`), the `regressor=None →
LinearRegression` default (REQ-4, `:281-284`), the both-None identity default
(REQ-5, `:190`), `_training_dim`/2D-`y` handling (REQ-6, `:263-279`), and
`fit_params`/`predict_params` passthrough (REQ-7, `:236-316`). The both-required
`func`/`inverse_func` half of REQ-5 IS satisfied by the type signature.

Invariants: `func`/`inverse_func` are pure `fn(F) -> F` (elementwise);
`y_transformed` is rejected if any element is NaN; predictions have the same row
count as `X`; both `func` and `inverse_func` are always present (type-enforced).

## Verification

Commands establishing the SHIPPED claims (baseline `c07c4089`):

- `cargo test -p ferrolearn-model-sel --lib transformed_target` → 7 passed, 0
  failed (REQ-1 func/inverse mechanism via synthetic inner estimator; REQ-8 NaN
  guard via `test_nan_func_error`).
- REQ-1 value-parity reference for the critic's oracle-grounded pin (live oracle):
  ```
  python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; \
  from sklearn.compose import TransformedTargetRegressor; \
  X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([2.,5.,9.,16.]); \
  tt=TransformedTargetRegressor(regressor=LinearRegression(), func=np.log, inverse_func=np.exp).fit(X,y); \
  print(tt.predict(X).tolist())"
  # -> [2.2126323263942624, 4.3788845965934815, 8.665981275583599, 17.15031072688885]
  ```
  Pin a `#[test]` whose inner pipeline is a real `LinearRegression`, `func=ln`,
  `inverse_func=exp`, expecting THIS vector (R-CHAR-3: from the live oracle, never
  copied from ferrolearn). The existing tests use a trivial mean-predicting inner
  estimator and assert hand-computed values — they pin the func/inverse MECHANISM,
  not the `LinearRegression` value-parity.
- REQ-3 oracle (the warning the gap omits): a non-strict inverse
  (`func=np.log, inverse_func=lambda z: z`) makes sklearn emit
  `UserWarning: "The provided functions or transformer are not strictly inverse
  of each other..."` (`_target.py:210-217`). ferrolearn emits nothing — REQ-3
  NOT-STARTED.
- REQ-4 oracle (the default the gap omits): `TransformedTargetRegressor().fit(X,y)`
  yields `regressor_` of type `LinearRegression` and `transformer_` of type
  `FunctionTransformer` (live oracle). ferrolearn requires an explicit
  `Pipeline<F>` — REQ-4 NOT-STARTED.
- REQ-5 oracle (both-required half, structurally satisfied): only-`func` →
  `ValueError: "When 'func' is provided, 'inverse_func' must also be provided..."`
  (live oracle, `_target.py:185-189`). ferrolearn's `new` requires both at the
  type level — matching the error semantics; the identity-default half is absent.

Commands that establish the NOT-STARTED REQs are absent (no `transformer` field,
no `check_inverse`, no `regressor=None` path, no 2D-`y` `Output`, no params
passthrough, no `ferray-core` usage). Per R-DEFER-2 the table is binary
SHIPPED/NOT-STARTED.

SHIPPED: REQ-1 (core func/inverse value parity — mechanism + tests; the
`LinearRegression` value-parity guard is the critic's pin), REQ-8 (NaN-on-func
guard, R-DEV-4 sanctioned deviation), REQ-10 (boundary re-export consumer;
honest-underclaim caveat — no dedicated non-test caller, no Python binding).
NOT-STARTED (tracking #1682; the critic files per-REQ blockers): REQ-2
(transformer mode), REQ-3 (check_inverse), REQ-4 (regressor=None default), REQ-5
(identity default half), REQ-6 (multi-output), REQ-7 (params passthrough), REQ-9
(ferray substrate).
