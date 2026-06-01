# Typed (Compile-Time) Pipeline

<!--
tier: 3-component
status: draft
baseline-commit: 165f19313a4d7f1c8dd56bc48f096df603bb0b2f
upstream-paths:
  - sklearn/pipeline.py
-->

## Summary

`ferrolearn-core/src/typed_pipeline.rs` is the Rust-idiom **typestate** variant
of scikit-learn's composite-estimator chaining in `sklearn/pipeline.py`. Its
*chaining semantics* mirror `Pipeline._fit`/`Pipeline.predict`: fit each
transformer step, transform the running data, feed the result to the next step,
and let the final estimator predict on the fully transformed data. It is a
sibling of the dynamic-dispatch `Pipeline` (`.design/core/pipeline.md`) and
mirrors the same sklearn contract.

Where the dynamic `Pipeline` constrains all intermediate data to `Array2<f64>`
and dispatches through trait objects, this module encodes the entire step
sequence as a **recursive nested type** — `TypedPipelineStep<Step1,
TypedPipelineStep<Step2, PipelineEnd>>` — so the compiler verifies that each
step's `Output` type matches the next step's `Input` type at **compile time**.
This compile-time type-compatibility guarantee has **no scikit-learn analog**
(Python is dynamically typed); it is documented below as a sanctioned **R-DEV-5**
Rust-idiom deviation, not a divergence. The module is generic over `Input` /
`Output` (no `ndarray` hardcoded — see Architecture), so the numpy-substrate
concern (R-SUBSTRATE) flows through the concrete step types in the estimator
crates, not this file.

Like the dynamic `Pipeline`, this is a SMALL subset. sklearn's `Pipeline`
additionally provides `fit_transform`, pipeline-level
`transform`/`predict_proba`/`decision_function`/`score`,
`named_steps`/`get_params`/`set_params`/`__getitem__`, `'passthrough'` steps,
`memory` caching, `s__p` fit-param/metadata routing, the `make_pipeline`
helper, and `FeatureUnion`. None exist in either pipeline yet; that surface is
identical to the dynamic pipeline's, so the requirements below reference the
SAME shared blockers (#361–#366) rather than filing duplicates.

## Requirements

- REQ-1: Sequential typed fit→transform chaining with a final-estimator
  fit/predict. Fitting fits each transformer on the running intermediate value
  and applies it before the next step (`fit_chain` recurses to the base case
  then fits each step outward), then fits the final estimator on the fully
  transformed data; predicting transforms input through every fitted step then
  calls the final estimator's predict. Mirrors `Pipeline._fit`
  (`sklearn/pipeline.py:383`) + `Pipeline.predict` (`sklearn/pipeline.py:550`).
- REQ-2: Compile-time step type-compatibility guarantee (R-DEV-5 sanctioned
  deviation, no sklearn analog). Each step's associated `Output` type is bound
  to be the next step's `Input` type through the recursive trait bounds, so an
  incompatible chain fails to compile rather than failing at runtime. sklearn
  has only the runtime `_validate_steps` duck-typing check
  (`sklearn/pipeline.py`, intermediate-step `fit`/`transform` requirement);
  the static guarantee is the Rust-idiom addition.
- REQ-3: Pipeline-level non-`predict` apply methods — `fit_transform`
  (`sklearn/pipeline.py:489`), pipeline-level `transform` for a
  transformer-final pipeline (`:863`), `predict_proba` (`:675`),
  `decision_function` (`:731`), `score` (`:961`). NOT implemented: the typed
  pipeline's fitted estimator path exposes only `predict`. (Note: a
  *transform-only* pipeline already exposes `transform`; what is missing is the
  estimator-pipeline's proba/score/decision surface.)
- REQ-4: Introspection / composition surface — `named_steps`
  (`sklearn/pipeline.py:324`), `get_params`/`set_params` with `s__p`
  nested-param addressing, and `__getitem__` integer access + slice
  sub-pipeline (`:298`). NOT implemented: the typed chain has no step-name
  registry or runtime step indexing.
- REQ-5: `'passthrough'`/`None` steps (`sklearn/pipeline.py:251`, `:289`,
  `:471`) and `memory` caching of fitted transformers (`:388`). NOT
  implemented.
- REQ-6: `fit_params` / metadata routing — per-step parameter forwarding via
  `s__p` prefixing and the `MetadataRouter` protocol (`sklearn/pipeline.py:30`,
  fit `:468`). NOT implemented: `fit_step` takes no per-step params.
- REQ-7: `make_pipeline` auto-naming helper (`sklearn/pipeline.py:1220`) —
  builds a pipeline from unnamed estimators with lowercased-type-name step
  names. NOT implemented: the typed builder is unnamed by construction (it is a
  type-level list, not a `(name, obj)` list).
- REQ-8: `FeatureUnion` (`sklearn/pipeline.py:1329`) — parallel transformer fit
  + horizontal concatenation of outputs. NOT implemented: no analog.

## Acceptance criteria

- AC-1: `cargo test -p ferrolearn-core --lib typed_pipeline` is green, including
  `test_single_transformer_with_estimator`,
  `test_two_transformers_with_estimator`,
  `test_three_transformers_with_estimator`, `test_four_step_pipeline`,
  `test_learning_transformer`, `test_learning_transformer_uses_fit_data`, and
  `test_fitted_pipeline_predicts_on_new_data` — pinning REQ-1's sequential
  fit→transform→predict chaining (including that a learning transformer uses
  the training-time parameters at predict time, not test-time).
- AC-2 (REQ-1 chaining oracle): a `StandardScaler → LinearRegression` pipeline
  in sklearn fits each step then the final estimator on the transformed data;
  `python3 -c "from sklearn.pipeline import Pipeline; ..."` confirms the final
  estimator predicts on the transformed running value — the same fit→transform→
  fit-final→predict ordering REQ-1 implements.
- AC-3 (REQ-2): the type-mismatch guarantee is enforced structurally by the
  recursive trait bounds (`Step: TypedTransformStep<Rest::ChainOutput>` and
  `Est: TypedEstimatorStep<Chain::ChainOutput, Target>`); a chain whose step
  output type does not match the next step's input type fails to type-check.
  `test_type_changing_pipeline` exercises a deliberate `Vec<f64> → Vec<i64>`
  type transition that the compiler must thread through correctly.
- AC-4 (REQ-3..8): no green verification exists because the surface is
  unimplemented; each is NOT-STARTED against the shared blocker carried over
  from the dynamic pipeline (#361–#366).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (typed fit→transform chaining + final predict) | SHIPPED | impl `fn fit in Fit for CompletePipeline in typed_pipeline.rs` (`let (fitted_chain, transformed) = self.chain.fit_chain(x)?; let fitted_est = self.estimator.fit_step(&transformed, y)?;`) + the recursive `fn fit_chain in FitTransformChain for TypedPipelineStep in typed_pipeline.rs` (`let (fitted_rest, intermediate) = self.rest.fit_chain(x)?; let fitted_step = self.step.fit_step(&intermediate)?; let output = fitted_step.transform_step(&intermediate)?;`) mirror `Pipeline._fit` `sklearn/pipeline.py:406` (`X, fitted_transformer = fit_transform_one_cached(...)`); `fn predict in Predict for FittedCompletePipeline in typed_pipeline.rs` (`let transformed = self.chain.transform_chain(x)?; self.estimator.predict_step(&transformed)`) mirrors `Pipeline.predict` `sklearn/pipeline.py:550`. Consumer: **public boundary API, no non-test production consumer** — `grep -rn "TypedPipeline\|\.then(\|\.finish(" ferrolearn-*/src/ \| grep -v '#\[cfg(test\|/tests/'` returns only this module's own doc-comments; the only callers are the in-module `#[cfg(test)]` suite and the `tests/api_proof.rs` smoke test (`TypedPipeline::new()`). It is grandfathered existing public API (S5/R-DOC-5) — the typed-pipeline builder types ARE the public surface; classified SHIPPED on that basis, NOT on an invented consumer. Verification: `cargo test -p ferrolearn-core --lib typed_pipeline` → 20 passed, 0 failed. |
| REQ-2 (compile-time type-compatibility, R-DEV-5) | SHIPPED | Enforced structurally by the recursive trait bounds, not by a runtime check: `impl FitTransformChain for TypedPipelineStep in typed_pipeline.rs` requires `Step: TypedTransformStep<Rest::ChainOutput>` (each step's `Input` must be the previous step's `ChainOutput`), and `impl Fit for CompletePipeline in typed_pipeline.rs` requires `Est: TypedEstimatorStep<Chain::ChainOutput, Target>` (the final estimator's `Input` must be the chain's terminal output). A mismatched chain has no satisfying impl and fails to compile. This is the sanctioned R-DEV-5 deviation: sklearn has only the runtime duck-typing `_validate_steps` (intermediate steps must implement `fit`/`transform`); there is no sklearn analog for the static guarantee. There is NO dedicated `trybuild`/compile-fail case pinning a typed-pipeline step-type mismatch (the existing `tests/compile_fail/` and `tests/ui/` cases pin unfitted-`predict`/`transform` on concrete estimators, not chain type-incompatibility) — the guarantee rests on the type-level bounds plus `test_type_changing_pipeline` exercising a real type transition. Honest underclaim: a `trybuild` mismatch case would strengthen this; the structural bound is the current evidence. |
| REQ-3 (fit_transform / pipeline transform / predict_proba / decision_function / score) | NOT-STARTED | open prereq blocker #361 (shared with `.design/core/pipeline.md` REQ-3 — same missing surface). The fitted estimator pipeline exposes only `predict`; sklearn's `fit_transform` (`pipeline.py:489`), `predict_proba` (`:675`), `decision_function` (`:731`), `score` (`:961`) have no typed analog. |
| REQ-4 (named_steps / get_params / set_params / __getitem__) | NOT-STARTED | open prereq blocker #362 (shared). The typed chain is an anonymous type-level list with no `(name, obj)` registry; `named_steps` (`pipeline.py:324`), `s__p` nested params, and `__getitem__` slice sub-pipeline (`:298`) are absent. |
| REQ-5 (passthrough steps + memory caching) | NOT-STARTED | open prereq blocker #363 (shared). No `'passthrough'`/`None` step concept (`pipeline.py:251`, `:289`, `:471`) and no `memory`/`check_memory` transformer caching (`:388`). |
| REQ-6 (fit_params / metadata routing) | NOT-STARTED | open prereq blocker #364 (shared). `TypedTransformStep::fit_step` / `TypedEstimatorStep::fit_step` accept no per-step params; sklearn forwards via `s__p` prefixing and `MetadataRouter` (`pipeline.py:30`, `:468`). |
| REQ-7 (make_pipeline auto-naming) | NOT-STARTED | open prereq blocker #365 (shared). No `make_pipeline` analog; the typed builder has no names at all (a type-level chain), vs sklearn's `_name_estimators` lowercased-type naming (`pipeline.py:1220`). |
| REQ-8 (FeatureUnion) | NOT-STARTED | open prereq blocker #366 (shared). No parallel-fit + hstack composite transformer (`pipeline.py:1329`). |

## Architecture

The module encodes scikit-learn's `Pipeline` step sequence (`sklearn/pipeline.py`)
as a **type-level cons-list** rather than a runtime `self.steps` list of
`(name, estimator)` tuples. There are four typed step traits, each carrying the
data type as a generic parameter / associated type so the compiler can chain
them:

- `TypedTransformStep<Input>` (`{ type Output; type FittedStep; fn fit_step }`)
  / `TypedFittedTransformStep<Input>` (`{ type Output; fn transform_step }`) —
  the unfitted and fitted halves of an intermediate step. This is the typed
  analog of sklearn's "intermediate steps must implement `fit` and `transform`"
  rule (`_validate_steps`). The associated `Output` type is what the compiler
  threads into the next step's `Input`.
- `TypedEstimatorStep<Input, Target>` (`{ type FittedStep; fn fit_step }`) /
  `TypedFittedEstimatorStep<Input>` (`{ type Output; fn predict_step }`) — the
  final step (only needs `fit`/`predict`).

`TransformAdapter<T>` / `EstimatorAdapter<T>` bridge any existing
`Fit`/`Transform`/`Predict` implementor (e.g. a `StandardScaler` or
`LinearRegression`) into these typed traits without modification, so the typed
pipeline composes the same estimators the dynamic pipeline does.

The chain itself is the recursive pair `PipelineEnd` (the nil/identity base
case) and `TypedPipelineStep<Step, Rest>` (cons cell holding one `step` and the
`rest` of the chain). Steps are stored last-added-outermost; the recursive
`FitTransformChain` / `TransformChain` internal traits walk the chain. The base
case `impl FitTransformChain for PipelineEnd` is the identity
(`ChainOutput = Input`), and the recursive `impl FitTransformChain for
TypedPipelineStep` fits `rest` first to obtain the intermediate value, then fits
and transforms `step` on it — the loop body of `sklearn/pipeline.py:392-419`,
expressed as type-level recursion instead of a Python `for` over `self._iter`.

Two complete-pipeline shapes sit on top:

- `CompletePipeline<Chain, Est>` (built by `TypedPipelineBuilder::finish`) holds
  the transformer `chain` plus a final `estimator`. Its `impl Fit<Input, Target>`
  calls `chain.fit_chain` then `estimator.fit_step` on the terminal output
  (`sklearn/pipeline.py:383`); the resulting `FittedCompletePipeline`'s
  `impl Predict<Input>` reruns `transform_chain` then `predict_step`
  (`sklearn/pipeline.py:550`).
- `TransformOnlyPipeline<Chain>` (built by `finish_transform`) is a
  transformer-only pipeline whose `impl Fit<Input, ()>` fits the chain and whose
  fitted form's `impl Transform<Input>` returns the chain's terminal output.
  This is the typed analog of a sklearn transformer-final pipeline's pipeline-
  level `transform` (`:863`) for the unsupervised case; the supervised
  proba/score surface (REQ-3) is still missing.

The builder is `TypedPipeline::new() -> TypedPipelineBuilder<PipelineEnd>`, then
`.then(step)` (prepend a `TypedPipelineStep`) and `.finish(est)` /
`.finish_transform()`. Because the step sequence is a type, an empty-estimator
pipeline is unrepresentable — there is no runtime "missing final estimator"
branch (the dynamic pipeline's REQ-2 fit-time error); `finish` structurally
requires an estimator, `finish_transform` structurally yields a transformer.

**Compile-time type safety (REQ-2, R-DEV-5).** The contract sklearn enforces at
runtime via `_validate_steps` (duck-typing `hasattr(..., "transform")`) is
enforced here at compile time by the trait bounds: the recursive
`impl FitTransformChain for TypedPipelineStep` is `where Step:
TypedTransformStep<Rest::ChainOutput>`, so a step whose `Input` does not equal
the previous step's `Output` has no satisfying impl. This is *why* the module
exists alongside the dynamic pipeline — it trades sklearn's heterogeneous
runtime flexibility for a static guarantee. Per R-DEV-5 this is a sanctioned
Rust-idiom deviation, not a divergence: there is nothing in sklearn to match
because Python cannot express it.

**Substrate.** Unlike `pipeline.rs` (which hardcodes `ndarray::{Array1,
Array2}` and therefore carries blocker #367), `typed_pipeline.rs` is generic over
`Input` / `Output` and imports no array type — `grep -n "ndarray\|Array1\|Array2"
ferrolearn-core/src/typed_pipeline.rs` matches only doc-comment prose. The numpy
substrate concern (R-SUBSTRATE-1/2) therefore does not bind on this file; it
flows through the concrete step types defined in the estimator crates. No
substrate REQ/blocker is filed against this file.

## Verification

- `cargo test -p ferrolearn-core --lib typed_pipeline` → 20 passed, 0 failed.
  Establishes REQ-1 (fit→transform→predict chaining for one/two/three/four
  steps, including `test_learning_transformer_uses_fit_data` proving the fitted
  step uses training-time parameters at predict time) and the structural basis
  of REQ-2 (`test_type_changing_pipeline` threads a `Vec<f64> → Vec<i64>`
  transition through the compiler).
- REQ-1 chaining oracle (installed sklearn 1.5.2):

  ```
  python3 -c "from sklearn.pipeline import Pipeline; \
    from sklearn.preprocessing import StandardScaler; \
    from sklearn.linear_model import LinearRegression; import numpy as np; \
    X=np.array([[1.,2.],[3.,4.],[5.,6.]]); y=np.array([1.,2.,3.]); \
    p=Pipeline([('s',StandardScaler()),('r',LinearRegression())]).fit(X,y); \
    print(p.predict(X).tolist())"
  # -> [1.0, 2.0, 3.0]
  ```

  Confirms sklearn fits each transformer, transforms the running data, fits the
  final estimator on the transformed data, and predicts through the same
  transform chain — the ordering `fit`/`predict` on `CompletePipeline` implement.
- Consumer presence (public-API-only; no non-test production consumer):

  ```
  grep -rn "TypedPipeline\|TypedTransformStep\|TypedEstimatorStep\|\.then(\|\.finish(" \
    ferrolearn-*/src/ | grep -v '#\[cfg(test' | grep -v '/tests/'
  # -> only ferrolearn-core/src/typed_pipeline.rs doc-comments
  #    (the unrelated .finish() hits in scorer.rs / gp_classifier.rs /
  #     function_transformer.rs / newsgroups.rs are tar/builder calls, not this API)
  ```

  The typed pipeline is exercised only by its in-module `#[cfg(test)]` suite and
  `ferrolearn-core/tests/api_proof.rs` (`TypedPipeline::new()`); it is
  grandfathered existing public boundary API (S5/R-DOC-5), classified SHIPPED on
  that basis without inventing a consumer.
- REQ-3..REQ-8 have no green verification because the surface is unimplemented;
  each is NOT-STARTED against the shared blocker (#361–#366) carried over from
  the dynamic pipeline. They will get characterization tests when built.
