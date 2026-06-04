# FeatureUnion

<!--
tier: 3-component
status: draft
baseline-commit: b8555591
upstream-paths:
  - sklearn/pipeline.py        # class FeatureUnion (:1329-:1847)
-->

## Summary

`ferrolearn-model-sel/src/feature_union.rs` mirrors scikit-learn's
`sklearn.pipeline.FeatureUnion` (`sklearn/pipeline.py:1329`) — the transformer
that applies several sub-transformers in parallel to the SAME input and
horizontally concatenates (`hstack`) their outputs in `transformer_list` order.
ferrolearn splits it into the unfitted builder `FeatureUnion<F>` and the fitted
`FittedFeatureUnion<F>`, both wired into the
`PipelineTransformer`/`FittedPipelineTransformer` traits from
`ferrolearn_core::pipeline` so a union is itself composable inside a `Pipeline`.

This file is the actual FeatureUnion implementation referenced by
`ferrolearn-core/src/pipeline.rs` REQ-8 (blocker #366): core declares FeatureUnion
NOT-STARTED at the trait layer and delegates the concrete transformer to
`ferrolearn-model-sel`; this module satisfies that core REQ-8 for the
core-concatenation behavior.

The current implementation ships the **core independent-fit + registration-order
horizontal-concatenation** contract. The remainder of sklearn's
`FeatureUnion` surface — `transformer_weights`, the `'drop'`/`'passthrough'`
sentinels, `get_feature_names_out` name-prefixing, the `y`-optional `fit`
signature, `n_features_in_`/`feature_names_in_` delegation, and the ferray
substrate — is NOT-STARTED.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/pipeline.py:1329` — `class FeatureUnion(TransformerMixin, _BaseComposition)`.
- `:1435-1448` — `__init__(self, transformer_list, *, n_jobs=None,
  transformer_weights=None, verbose=False, verbose_feature_names_out=True)`.
- `:1552-1565` — `_iter`: skips `trans == "drop"` (`:1561`), substitutes a
  `FunctionTransformer(feature_names_out="one-to-one")` for `trans ==
  "passthrough"` (`:1563-1564`), and yields each transformer's weight via
  `get_weight = (self.transformer_weights or {}).get` (`:1558`, `:1565`).
- `:1540-1550` — `_validate_transformer_weights`: a weight key absent from
  `transformer_list` raises `ValueError`.
- `:1643-1688` — `fit(self, X, y=None, **fit_params)`: `y` defaults to `None`;
  fits each transformer independently (`_parallel_func(X, y, _fit_one, ...)`).
- `:1690-1743` — `fit_transform`: fits + transforms each, then `self._hstack(Xs)`.
- `:1770-1810` — `transform(self, X, **params)`: `_transform_one` per transformer
  then `self._hstack(Xs)`.
- `:1812-1821` — `_hstack`: `np.hstack(Xs)` (dense) / `sparse.hstack(...).tocsr()`.
- `:1567-1593` — `get_feature_names_out`: collects each sub-transformer's
  `get_feature_names_out` and delegates to `_add_prefix_for_feature_names_out`.
- `:1595-1641` — `_add_prefix_for_feature_names_out`: prefixes each name as
  `f"{name}__{i}"` when `verbose_feature_names_out` (default True).
- `:1830-1841` — `n_features_in_` / `feature_names_in_` properties delegate to
  `transformer_list[0]`.

## Requirements

- REQ-1: Core independent-fit + registration-order horizontal concatenation.
  `fit` fits every sub-transformer independently on the same `(X, y)`; `transform`
  applies each fitted transformer and `hstack`s the blocks in registration order
  (`np.hstack` order = `transformer_list` order). Mirrors `fit`
  (`pipeline.py:1681`), `transform` (`:1802-1810`), and `_hstack`
  (`:1812-1820`). Output column order = transformer-1 columns, then
  transformer-2 columns, ... DETERMINISTIC / oracle-pinnable.
- REQ-2: `transformer_weights` — each transformer's output block multiplied by
  its weight before concatenation (`_iter` `:1558,:1565`; `_weight_one` applied
  in `_transform_one`), with a `ValueError` for a weight naming a non-existent
  transformer (`:1540-1550`).
- REQ-3: `'drop'`/`'passthrough'` sentinels — a `'drop'` transformer is skipped;
  a `'passthrough'` transformer passes its input columns through unchanged
  (`:1561`, `:1563-1564`).
- REQ-4: `get_feature_names_out` — produce output feature names prefixed with the
  transformer name (`f"{name}__{feat}"`) when `verbose_feature_names_out`
  (`:1567-1593`, `:1608-1616`).
- REQ-5: `y`-optional `fit` — `fit(X, y=None)` per sklearn (`:1643`), so an
  unsupervised union does not require targets.
- REQ-6: Empty-`transformer_list` behavior matches sklearn (sklearn raises
  `ValueError` on an empty list — see Verification).
- REQ-7: Code-quality / clippy gauntlet — `cargo clippy -p ferrolearn-model-sel
  --all-targets -- -D warnings` is green (the `collapsible_if` lint in
  `FittedFeatureUnion::transform` must be resolved).
- REQ-8: R-SUBSTRATE — array type and horizontal concatenation on `ferray-core`
  rather than `ndarray`.
- REQ-9: Non-test production consumer — `FeatureUnion` is reachable from
  non-test production code (crate re-export + `make_union` constructor +
  `PipelineTransformer` composability).

## Acceptance criteria

- AC-1 (REQ-1): for a union of two transformers producing blocks `A`
  (`n×a`) and `B` (`n×b`), `transform(X)` equals the column-concatenation
  `[A | B]` of shape `n×(a+b)`; reordering the registration order reorders the
  output columns. Against the live oracle, a ferrolearn union of sub-transformers
  equivalent to sklearn's `[('s', StandardScaler()), ('m', MinMaxScaler())]`
  produces the same concatenated matrix sklearn does (deterministic;
  oracle-pinnable — see Verification for the sklearn reference matrix).
- AC-2 (REQ-2): a union with `transformer_weights={'a': 2.0}` scales block `a`
  by 2.0 before concatenation, matching sklearn; an unknown weight key raises an
  `InvalidParameter`/`ValueError`-equivalent.
- AC-3 (REQ-3): a union with one transformer set to `'drop'` omits its columns;
  `'passthrough'` emits the input columns unchanged — matching sklearn.
- AC-4 (REQ-4): `get_feature_names_out` returns names prefixed `name__feat`
  matching sklearn's `_add_prefix_for_feature_names_out`.
- AC-5 (REQ-5): `fit(X)` with no `y` succeeds for unsupervised sub-transformers.
- AC-6 (REQ-6): an empty union errors with the sklearn-equivalent exception
  (sklearn: `ValueError`).
- AC-7 (REQ-7): `cargo clippy -p ferrolearn-model-sel --all-targets -- -D
  warnings` exits 0.
- AC-8 (REQ-8): owned concatenation runs on `ferray-core`, no `ndarray` in the
  owned computation.
- AC-9 (REQ-9): `FeatureUnion` is constructed from non-test production code.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (core concat) | SHIPPED | `pub fn fit` in `feature_union.rs` loops `t.fit_pipeline(x, y)?` over each registered `(name, t)` independently (mirrors `_fit_one` per transformer, `pipeline.py:1681`). `FittedFeatureUnion::transform` in `feature_union.rs` calls `t.transform_pipeline(x)?` per fitted transformer, collects the blocks in registration order, and does `ndarray::concatenate(Axis(1), &views)` — mirroring `_hstack` `np.hstack(Xs)` (`pipeline.py:1820`). A row-count consistency check (`part.nrows() != first.nrows()` → `ShapeMismatch`) precedes concat. DETERMINISTIC / oracle-pinnable: a union of sub-transformers equivalent to sklearn's StandardScaler+MinMaxScaler yields sklearn's concatenated matrix (Verification). Non-test consumer: REQ-9. Tests: `test_identity_and_doubler`, `test_three_transformers`, `test_different_width_outputs` pin block order and column counts. CAVEAT: the existing tests assert column counts and synthetic Identity/Doubler values, NOT an oracle-grounded value-parity matrix against live sklearn (model-sel has no `ferrolearn-preprocess` dev-dep); the critic should add the StandardScaler+MinMaxScaler value-parity `#[test]` (expected from the live oracle below). The CONCAT MECHANISM matches; the value-parity guard is the critic's pin. |
| REQ-2 (transformer_weights) | NOT-STARTED | open prereq blocker (tracking #1676; critic to file specific). `struct FeatureUnion` carries only `transformers: Vec<(String, Box<dyn PipelineTransformer<F>>)>` — no `transformer_weights` field, no constructor param, no per-block scaling in `transform`. sklearn multiplies each block by `get_weight(name)` (`pipeline.py:1558,:1565`) and validates keys (`:1540-1550`). Absent end-to-end. |
| REQ-3 (drop/passthrough) | NOT-STARTED | open prereq blocker (tracking #1676). `add(name, transformer)` only accepts a `Box<dyn PipelineTransformer<F>>`; there is no `'drop'`/`'passthrough'` sentinel and `_iter` has no analog. sklearn skips `'drop'` (`pipeline.py:1561`) and substitutes a passthrough `FunctionTransformer` (`:1563-1564`). No support. |
| REQ-4 (get_feature_names_out) | NOT-STARTED | open prereq blocker (tracking #1676). `FeatureUnion`/`FittedFeatureUnion` expose `transformer_names()` and `n_transformers()` but no `get_feature_names_out`; the `PipelineTransformer` trait carries no feature-name method. sklearn prefixes each name `f"{name}__{i}"` (`pipeline.py:1608-1616`). Absent. |
| REQ-5 (y-optional fit) | NOT-STARTED | open prereq blocker (tracking #1676) — R-DEV-2 API divergence. `pub fn fit(&self, x: &Array2<F>, y: &Array1<F>)` REQUIRES `y`; sklearn `fit(X, y=None)` (`pipeline.py:1643`) makes `y` optional so an unsupervised union (the typical case — PCA, scalers) needs no targets. Callers currently must pass a dummy `y`. The required `y: &Array1<F>` signature diverges from sklearn's optional contract. |
| REQ-6 (empty-list behavior) | SHIPPED | `pub fn fit` returns `FerroError::InvalidParameter { name: "transformers", reason: "FeatureUnion must have at least one transformer" }` when `self.transformers.is_empty()`. Matches sklearn: `FeatureUnion([]).fit_transform(...)` raises `ValueError` (`_validate_transformers` `zip(*self.transformer_list)` → "not enough values to unpack", live-oracle-confirmed). `FerroError::InvalidParameter` is the project's `ValueError` analog (R-DEV-2). Test: `test_empty_union_error`. Non-test consumer: REQ-9. DETERMINISTIC / oracle-pinnable. NOTE: sklearn surfaces a bare unpack `ValueError`, not a descriptive one — both are `ValueError`-class; the message text differs (not a contract divergence). |
| REQ-7 (clippy gauntlet) | NOT-STARTED | open prereq blocker (tracking #1676; fix this iteration). `FittedFeatureUnion::transform` has `if let Some(first) = parts.first() { if part.nrows() != first.nrows() { ... } }` — `clippy::collapsible_if` (a let-chain collapse, stable on workspace MSRV 1.88). Confirmed firing: `cargo clippy -p ferrolearn-model-sel --all-targets` warns at `feature_union.rs` ("collapse nested if block" → `if let Some(first) = parts.first() && part.nrows() != first.nrows()`). This blocks `-D warnings`, so the model-sel gauntlet is not green until collapsed. |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker (tracking #1676; R-SUBSTRATE-2). `feature_union.rs` uses `ndarray::{Array1, Array2, Axis}` and `ndarray::concatenate(Axis(1), ...)`. Destination substrate is `ferray-core` array type + its hstack analog (R-SUBSTRATE-1). Not migrated. |
| REQ-9 (consumer) | SHIPPED | Crate re-export: `lib.rs` (`pub use feature_union::{FeatureUnion, FittedFeatureUnion}`). Non-test production constructor: `pub fn make_union` in `helpers.rs` builds a `FeatureUnion::<F>::new()` and `.add("fu{i}", t)` for each input transformer (re-exported `pub use helpers::{make_pipeline, make_union}` in `lib.rs`) — mirrors sklearn's `make_union` (`sklearn/pipeline.py`, the convenience constructor). `FeatureUnion` also `impl PipelineTransformer<F>` (its `fit_pipeline` boxes a `FittedFeatureUnion`), so it composes inside a `Pipeline`. R-DEFER-1 / S5: boundary transformer type, grandfathered existing pub API; no `ferrolearn-python` binding exists for it yet (a narrower-than-sklearn Python surface, noted under REQ-9). |

## Architecture

ferrolearn splits the estimator into the unfitted builder `FeatureUnion<F>`
(field `transformers: Vec<(String, Box<dyn PipelineTransformer<F>>)>`, built via
the chained `.add(name, transformer)`) and the fitted `FittedFeatureUnion<F>`
(field `transformers: Vec<(String, Box<dyn FittedPipelineTransformer<F>>)>`),
matching sklearn's single-class `transformer_list` → post-`fit` updated list
(`pipeline.py:1444`, `_update_transformer_list` `:1823`). sklearn keeps one
class; ferrolearn uses the project-wide unfitted/Fitted split (CLAUDE.md naming).

`fit` short-circuits on an empty list (`InvalidParameter`), then fits each
sub-transformer independently via `fit_pipeline(x, y)`, preserving registration
order. sklearn fits via `_parallel_func(X, y, _fit_one, ...)` (`:1681`) — the
`n_jobs` parallelism is out of scope per goal.md (a performance concern, NOT a
divergence); ferrolearn's serial fit produces the same result.

`FittedFeatureUnion::transform` applies each fitted transformer's
`transform_pipeline(x)`, enforces equal row counts across blocks
(`ShapeMismatch` otherwise — ferrolearn checks this explicitly; sklearn lets
`np.hstack` raise), and horizontally concatenates with
`ndarray::concatenate(Axis(1), &views)` — the column order is
transformer-1's columns, then transformer-2's, matching `np.hstack(Xs)`
(`pipeline.py:1820`). This is the core REQ-1 contract and the deterministic,
oracle-pinnable behavior.

What is structurally absent vs sklearn: the `transformer_weights` field (REQ-2),
the `'drop'`/`'passthrough'` string sentinels (REQ-3, which require `add` to
accept a sentinel union type, not just a boxed transformer), `get_feature_names_out`
name-prefixing (REQ-4, which requires a feature-name method on the
`PipelineTransformer` trait), the `y`-optional `fit` signature (REQ-5), and the
`n_features_in_`/`feature_names_in_` delegation properties (`:1830-1841`).

Invariants: all blocks share the input's row count `n`; output is
`n × Σ block_widths` in registration order; an empty union is rejected at `fit`.

## Verification

Commands establishing the SHIPPED claims (baseline `b8555591`):

- `cargo test -p ferrolearn-model-sel --lib feature_union` → 10 passed, 0 failed
  (REQ-1 block-order/column-count, REQ-6 empty-error).
- Empty-list parity (REQ-6), live oracle:
  `python3 -c "from sklearn.pipeline import FeatureUnion; import numpy as np;
  FeatureUnion([]).fit_transform(np.array([[1.,2.]]))"` →
  `ValueError: not enough values to unpack (expected 2, got 0)`. ferrolearn
  `FeatureUnion::new().fit(...)` → `FerroError::InvalidParameter` (both
  `ValueError`-class; `test_empty_union_error` green).
- Core-concat value reference for the critic's REQ-1 value-parity pin (REQ-1,
  oracle, `X=[[1,2,3],[4,5,6],[7,8,10]]`,
  `FeatureUnion([('s', StandardScaler()), ('m', MinMaxScaler())])`):
  ```
  shape (3, 6)
  [[-1.224744871392 -1.224744871392 -1.162476387438  0.   0.   0.            ]
   [ 0.              0.             -0.116247638744  0.5  0.5  0.428571428571]
   [ 1.224744871392  1.224744871392  1.278724026182  1.   1.   1.            ]]
  ```
  Columns 0-2 = StandardScaler block, columns 3-5 = MinMaxScaler block, in
  registration order — pin a `#[test]` whose expected matrix is THIS live-oracle
  output (R-CHAR-3: never copied from ferrolearn). NOTE: model-sel has no
  `ferrolearn-preprocess` dev-dependency, so the critic either adds one or builds
  sub-transformers reproducing sklearn's scaler formulas; the column-order +
  hstack contract is what is pinned.

Commands that currently FAIL (the NOT-STARTED REQs):

- `cargo clippy -p ferrolearn-model-sel --all-targets -- -D warnings` →
  `error: this `if` statement can be collapsed` at `feature_union.rs`
  (`clippy::collapsible_if`) — REQ-7 NOT-STARTED until collapsed.

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED.
SHIPPED: REQ-1 (core concat — mechanism + tests; value-parity guard is the
critic's pin), REQ-6 (empty-list error), REQ-9 (re-export + `make_union`
consumer + `PipelineTransformer` composability).
NOT-STARTED (tracking #1676; the critic files per-REQ blockers): REQ-2
(transformer_weights), REQ-3 (drop/passthrough), REQ-4 (get_feature_names_out),
REQ-5 (y-optional fit), REQ-7 (collapsible_if clippy gate), REQ-8 (ferray
substrate).
</content>
</invoke>
