# Composite Pipeline

<!--
tier: 3-component
status: draft
baseline-commit: 54a19ef06ffebdf2e596e5db49137820454d50bd
upstream-paths:
  - sklearn/pipeline.py
-->

## Summary

`ferrolearn-core/src/pipeline.rs` is the Rust translation of scikit-learn's
composite-estimator chaining in `sklearn/pipeline.py`. It mirrors a deliberately
minimal subset of the `Pipeline` class: a sequence of named transformer steps
followed by a single final estimator, fit in order (each transformer fit then
applied to the running `Xt`, the estimator fit on the final transformed data),
then a fitted pipeline that transforms input through every fitted transformer
before calling `predict` on the final estimator. Where sklearn uses Python duck
typing and `available_if`, ferrolearn uses four dynamic-dispatch trait objects
(`PipelineTransformer` / `FittedPipelineTransformer` / `PipelineEstimator` /
`FittedPipelineEstimator`) and the typestate split between `Pipeline<F>` and
`FittedPipeline<F>`.

This is a SMALL subset. sklearn's `Pipeline` additionally provides
`fit_transform`, pipeline-level `transform`/`predict_proba`/`decision_function`/
`score`, `named_steps`/`get_params`/`set_params`/`__getitem__`, `'passthrough'`
steps, `memory` caching, `s__p` fit-param/metadata routing, the `make_pipeline`
helper, and `FeatureUnion` — none of which exist in ferrolearn yet. Those are
classified NOT-STARTED below against concrete prereq blockers, not silently
omitted. The data substrate is `ndarray` (not yet ferray).

## Requirements

- REQ-1: Sequential fit→transform chaining with a final-estimator fit/predict.
  Fitting fits each transformer on the running `Xt` and applies it before the next
  step, then fits the final estimator on the fully transformed data; predicting
  transforms input through every fitted transformer then calls the final
  estimator's predict. Mirrors `Pipeline._fit` (`sklearn/pipeline.py:383`) +
  `Pipeline.predict` (`sklearn/pipeline.py:550`).
- REQ-2: A pipeline with no final estimator is rejected at fit. ferrolearn returns
  `FerroError::InvalidParameter`; sklearn raises on a degenerate pipeline — empty
  `steps` raises `ValueError` (unpack of `self.steps[-1]`), and an all-transformer
  pipeline exposes no `predict` (`available_if(_final_estimator_has("predict"))`,
  `sklearn/pipeline.py:549`). ferrolearn collapses both into a fit-time error
  because its `Pipeline` requires the final estimator slot to be set.
- REQ-3: Pipeline-level non-`predict` apply methods — `fit_transform`
  (`sklearn/pipeline.py:489`), `transform` for a transformer-final pipeline
  (`:863`), `predict_proba` (`:675`), `decision_function` (`:731`), `score`
  (`:961`). NOT implemented: ferrolearn's `FittedPipeline` exposes only `predict`.
- REQ-4: Introspection / composition surface — `named_steps`
  (`sklearn/pipeline.py:324`), `get_params`/`set_params` with `s__p` nested-param
  addressing (via `_BaseComposition`), and `__getitem__` integer access + slice
  sub-pipeline (`:298`). NOT implemented: `FittedPipeline` exposes only
  `step_names()`.
- REQ-5: `'passthrough'`/`None` steps (`sklearn/pipeline.py:251`, `:289`, `:471`)
  and `memory` caching of fitted transformers (`:388`). NOT implemented.
- REQ-6: `fit_params` / metadata routing — per-step parameter forwarding via
  `s__p` prefixing and the `MetadataRouter` protocol (`sklearn/pipeline.py:468`,
  `:30`). NOT implemented: `fit_pipeline` takes no per-step params.
- REQ-7: `make_pipeline` auto-naming helper (`sklearn/pipeline.py:1220`) — builds
  a pipeline from unnamed estimators with lowercased-type-name step names. NOT
  implemented: ferrolearn requires explicit step names.
- REQ-8: `FeatureUnion` (`sklearn/pipeline.py:1329`) — parallel transformer fit +
  horizontal concatenation of outputs. SHIPPED (core fit/transform/hstack/
  `get_feature_names_out`): `FeatureUnion`/`FittedFeatureUnion` fit every named
  sub-transformer on the same `X` (`FeatureUnion.fit`, `:1643`) and hstack their
  outputs left-to-right (`FeatureUnion.transform` → `_hstack`, `:1770`/`:1812`),
  with `{name}__x{j}` feature-name prefixing (the `verbose_feature_names_out=True`
  default, `:1567`/`:1608`). NOT-STARTED: `transformer_weights` (`:1369`),
  `'drop'`/`'passthrough'` (`:1530`/`:1563`), `n_jobs` (`:1360`), metadata routing
  (`:1859`), `verbose_feature_names_out=False` (`:1618`). `FeatureUnion::fit` also
  validates transformer-name uniqueness up front, mirroring
  `_validate_transformers` → `_validate_names` (`:1523-1525` →
  `sklearn/utils/metaestimators.py:81-83`): a duplicate name returns
  `FerroError::InvalidParameter` (sklearn's `ValueError` analog) instead of fitting.
- REQ-9: ferray substrate. The pipeline data flow is typed on
  `ndarray::{Array1, Array2}`, the wrong-substrate numpy analog; the destination is
  ferray-core array types (R-SUBSTRATE-1/2). NOT-STARTED (cascading;
  grandfathered-transitional per R-SUBSTRATE-4).

## Acceptance criteria

- AC-1: `cargo test -p ferrolearn-core` is green, including the pipeline unit
  tests `test_pipeline_fit_predict`, `test_pipeline_f32_fit_predict`,
  `test_multiple_transform_steps`, `test_pipeline_estimator_only`,
  `test_fitted_pipeline_step_names`, and `test_pipeline_step_builder` — pinning
  REQ-1's sequential chaining for both `f32` and `f64`.
- AC-2: `test_pipeline_no_estimator_returns_error` and `test_pipeline_default`
  pin REQ-2: fitting a pipeline whose estimator slot is unset returns `Err`.
- AC-3: At least one non-test production estimator in another workspace crate
  implements `PipelineEstimator` and at least one implements
  `PipelineTransformer`, exercising REQ-1's trait surface end-to-end (not just
  test fixtures): `GaussianNB`/`BernoulliNB` and `KernelPCA` respectively.
- AC-4 (REQ-2 oracle): `python3 -c "from sklearn.pipeline import Pipeline; ..."`
  confirms sklearn raises `ValueError` on empty `steps` and `AttributeError`
  ('Pipeline' has no attribute 'predict') on an all-transformer pipeline — i.e.
  a final predictor is mandatory for `.predict`, matching ferrolearn's fit-time
  rejection.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (fit→transform chaining + final predict) | SHIPPED | impl `fn fit in Fit for Pipeline in pipeline.rs` (`current_x = fitted.transform_pipeline(&current_x)?;` then `est.fit_pipeline(&current_x, y)`) mirrors `Pipeline._fit` `sklearn/pipeline.py:406` (`X, fitted_transformer = fit_transform_one_cached(...)`); `fn predict in Predict for FittedPipeline in pipeline.rs` (`current_x = ts.step.transform_pipeline(&current_x)?;` then `self.estimator.1.predict_pipeline(&current_x)`) mirrors `Pipeline.predict` `sklearn/pipeline.py:599` (`for ...: Xt = transform.transform(Xt); return self.steps[-1][1].predict(Xt)`). Non-test consumers (production, outside `#[cfg(test)]`): `impl PipelineEstimator for GaussianNB in gaussian.rs` and `impl PipelineEstimator for BernoulliNB in bernoulli.rs` (each `fn fit_pipeline` delegates to the estimator's own `fit`); `impl PipelineTransformer for KernelPCA in kernel_pca.rs` (`fn fit_pipeline` → `self.fit(x, &())`). Verification: `cargo test -p ferrolearn-core` → 98 passed, 0 failed (lib), pipeline module 29 passed. |
| REQ-2 (no-final-estimator rejected at fit) | SHIPPED | impl `fn fit in Fit for Pipeline in pipeline.rs` (`if self.estimator.is_none() { return Err(FerroError::InvalidParameter { name: "estimator".into(), reason: "pipeline must have a final estimator step".into() }); }`). Oracle: `Pipeline(steps=[]).fit(X,y)` → `ValueError: not enough values to unpack`; `Pipeline([('s',StandardScaler())]).fit(X).predict(X)` → `AttributeError: This 'Pipeline' has no attribute 'predict'` (`available_if` at `sklearn/pipeline.py:549`) — a final predictor is mandatory, matching ferrolearn's fit-time error. Non-test consumer: same production `PipelineEstimator` impls as REQ-1 always supply the estimator slot. Verification: `test_pipeline_no_estimator_returns_error` + `test_pipeline_default` pass. |
| REQ-3 (fit_transform / transform / predict_proba / decision_function / score) | SHIPPED | impl: `fn fit_transform in Pipeline in pipeline.rs` (reuses `Fit::fit` then `fitted.transform_through(x)`), mirroring `Pipeline.fit_transform` (`sklearn/pipeline.py:489`); `fn transform`/`fn predict_proba`/`fn decision_function`/`fn score in FittedPipeline in pipeline.rs` each call the private `fn transform_through` (the shared `Xt = transform.transform(Xt)` loop, `:599-600`/`:719-720`/`:768-769`/`:999-1000`) then delegate to the final estimator — `transform` returns the transformer-prefix output (sklearn raises `AttributeError` on a non-transformer-final `transform`, `_can_transform` `:858`; verified live), `predict_proba` → `predict_proba_pipeline` (`:721`), `decision_function` → `decision_function_pipeline` (`:772`), `score` → `score_pipeline` (`:1004`). The three delegations are new default-`Err` methods on `trait FittedPipelineEstimator` (the `available_if(_final_estimator_has(...))` analog, `:674`/`:731`/`:960`). Non-test production consumer: `impl FittedPipelineEstimator for FittedGaussianNBPipeline in gaussian.rs` overrides `fn predict_proba_pipeline` (→ `self.fitted.predict_proba`) and `fn score_pipeline` (→ maps original float `y` to class indices, `self.fitted.score`); `decision_function_pipeline` keeps the default `Err` (GaussianNB has no `decision_function` in sklearn). Verification (R-CHAR-3, live sklearn 1.5.2 oracle): `gaussian_pipeline_predict_proba_score_match_sklearn` (`ferrolearn-bayes` lib) builds a `StandardScaler`-equivalent transformer + `GaussianNB` pipeline and matches sklearn's `Pipeline.predict_proba` rows (incl. the non-degenerate `[0.3992908671216849, 0.6007091328783151]`, max_relative 1e-9), `.score` (`0.6666666666666666`), and `.transform` (`StandardScaler().fit(X).transform(Xt)`, 1e-12); ferrolearn-core lib tests `test_pipeline_fit_transform_equals_transform`, `test_pipeline_transform_applies_only_transformer_steps`, `test_pipeline_predict_proba_default_is_err`, `test_pipeline_predict_proba_and_score_override`. `cargo test -p ferrolearn-core` 102 lib passed / 0 failed; `cargo test -p ferrolearn-bayes --lib` 116 passed / 0 failed. NOT yet threaded: `sample_weight` on `score` (owned by REQ-6 metadata routing, blocker #364). |
| REQ-4 (named_steps / get_params / set_params / __getitem__) | NOT-STARTED | open prereq blocker #362. `FittedPipeline::step_names` returns only `Vec<&str>`; sklearn's `named_steps` Bunch (`pipeline.py:324`), `s__p` nested params, and `__getitem__` slice sub-pipeline (`:298`) are absent. |
| REQ-5a (passthrough steps) | SHIPPED | `PassthroughTransformer`/`FittedPassthroughTransformer` (no-op identity, `transform_pipeline -> x.clone()`) + `Pipeline::passthrough_step(name)` convenience in `pipeline.rs`; the Rust analog of sklearn's `'passthrough'`/`None` step (`pipeline.py:251`,`:289`). A concrete identity transformer in the chain is observationally equivalent to sklearn's `_iter(filter_passthrough=True)` skip. acto-critic CLEAN (identity exactness, multi-passthrough composition, estimator-after sees raw X, accessor interop, shape/dtype, f32). |
| REQ-5b (memory caching) | NOT-STARTED | open prereq blocker #363. No `memory`/`check_memory`/`fit_transform_one_cached` transformer caching (`pipeline.py:388-390`) — joblib disk-cache substrate, no ferrolearn analog. |
| REQ-6 (fit_params / metadata routing) | NOT-STARTED | open prereq blocker #364. `PipelineEstimator::fit_pipeline` / `PipelineTransformer::fit_pipeline` accept no per-step params; sklearn forwards via `s__p` prefixing and `MetadataRouter` (`pipeline.py:468`, `:30`). |
| REQ-7 (make_pipeline auto-naming) | NOT-STARTED | open prereq blocker #365. No `make_pipeline` analog; ferrolearn `transform_step`/`estimator_step` require explicit names, vs sklearn's `_name_estimators` lowercased-type naming (`pipeline.py:1220`). |
| REQ-8 (FeatureUnion) | SHIPPED | impl: `struct FeatureUnion`/`FittedFeatureUnion in pipeline.rs`; `fn fit in Fit<Array2<F>, ()> for FeatureUnion` fits each named sub-transformer on the SAME `x` via `transformer.fit_pipeline(x, &empty_y)` (mirrors `FeatureUnion.fit` fitting every transformer on `X`, `sklearn/pipeline.py:1643`/`:1681`), recording each output width; `fn transform in Transform<Array2<F>> for FittedFeatureUnion` transforms `x` through each fitted transformer and copies each block into its contiguous column range left-to-right (mirrors `FeatureUnion.transform` → `_hstack` `np.hstack(Xs)`, `pipeline.py:1770`/`:1812`/`:1820`), validating row-alignment (`FerroError::ShapeMismatch` on a mismatched-rows block); `fn get_feature_names_out in FittedFeatureUnion` emits `{name}__x{j}` per block (the `verbose_feature_names_out=True` default prefix, `pipeline.py:1567`/`:1608-1616`; positional `x{j}` = the `OneToOneFeatureMixin` default since the trait objects expose no per-output names). y-handling: `fit_pipeline` requires `Array1<F>`, so the unsupervised sub-transformers get an empty `Array1::zeros(0)` (sklearn passes `y` through, transformers ignore it). Non-test production consumer: the pub API on the `pub mod pipeline` surface — the same boundary the grandfathered `Pipeline`/`FittedPipeline` types sit on (S5; neither `Pipeline` nor `FeatureUnion` is crate-root re-exported in `lib.rs`). Live-oracle verification (R-CHAR-3, sklearn 1.5.2): `test_feature_union_hstack_layout` (the `[ss|mm]` 2-block layout on `[[1,2],[3,4],[5,6]]` → `(3,4)`), `test_feature_union_get_feature_names_out` (`['ss__x0','ss__x1','mm__x0','mm__x1']`), `test_feature_union_single_transformer_width`, `test_feature_union_mixed_widths` (3+1 blocks, oracle StandardScaler+PCA(1) → `(3,4)`), `test_feature_union_empty` (`(n,0)`), `test_feature_union_row_count_consistency`, `test_feature_union_f32`; name-uniqueness validation at fit (mirrors `_validate_names`, `sklearn/utils/metaestimators.py:81-83` reached from `pipeline.py:1523-1525`): duplicate names → `FerroError::InvalidParameter`, pinned by `divergence_feature_union_duplicate_names_must_error` + positive guard `feature_union_unique_names_fit_ok`. NOT-STARTED: `transformer_weights` per-output scaling (`pipeline.py:1369`), `'drop'`/`'passthrough'` sentinels (`:1530`/`:1563`), `n_jobs` (`:1360`), metadata routing (`:1859`), `verbose_feature_names_out=False` (`:1618-1641`), ferray substrate. |
| REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker #367. Pipeline trait surface and data flow are typed on `ndarray::{Array1, Array2}` (wrong substrate); destination is ferray-core (R-SUBSTRATE-1/2, grandfathered-transitional R-SUBSTRATE-4). |

## Architecture

The module mirrors sklearn's `Pipeline` (`sklearn/pipeline.py:57`) with a
typestate split that scikit-learn does not have. sklearn stores a single
`self.steps` list of `(name, estimator)` tuples and dispatches by duck typing
(`hasattr(..., "transform")`) guarded with `available_if`; ferrolearn instead
encodes the transformer/estimator distinction in the type system via four trait
objects, all bounded `Send + Sync` and generic over `F: Float + Send + Sync +
'static` (defaulting to `f64`):

- `PipelineTransformer<F>` / `FittedPipelineTransformer<F>` — the unfitted and
  fitted halves of an intermediate step. `fit_pipeline(x, y) ->
  Box<dyn FittedPipelineTransformer<F>>` then `transform_pipeline(x) ->
  Array2<F>`. This is the typestate analog of sklearn's "intermediate steps must
  implement `fit` and `transform`" rule (`_validate_steps`,
  `sklearn/pipeline.py:250`).
- `PipelineEstimator<F>` / `FittedPipelineEstimator<F>` — the final step.
  `fit_pipeline(x, y) -> Box<dyn FittedPipelineEstimator<F>>` then
  `predict_pipeline(x) -> Array1<F>`. The final-step-only-needs-`fit`/`predict`
  contract of `sklearn/pipeline.py:264`.

`Pipeline<F = f64>` (the unfitted builder) holds `transforms: Vec<TransformStep>`
and `estimator: Option<(String, Box<dyn PipelineEstimator<F>>)>`. The `Option`
is the structural reason REQ-2 holds: an unset slot is unrepresentable as a
fitted pipeline. `new`/`transform_step`/`estimator_step` are the builder; `step`
+ the `PipelineStep` unifying trait (with `TransformerStepWrapper` /
`EstimatorStepWrapper` and the free fns `as_transform_step` / `as_estimator_step`)
provide a single `.step(name, …)` entry point dispatching to the right slot —
this is ferrolearn's Rust-typed substitute for sklearn's heterogeneous
`steps=[(name, obj), …]` list, NOT a `make_pipeline` analog (which auto-NAMES;
REQ-7).

`impl Fit<Array2<F>, Array1<F>> for Pipeline` performs the loop of
`sklearn/pipeline.py:392-419`: fit each transformer on `current_x`, immediately
`transform_pipeline` to advance `current_x`, collect `FittedTransformStep`s, then
fit the final estimator on the fully transformed data (`:471-473`). It returns
`FittedPipeline<F>`, whose `impl Predict<Array2<F>>` reruns the transform chain
and calls `predict_pipeline` on the final estimator (`:599-601`). Intermediate
data is always `Array2<F>` and the prediction is `Array1<F>` — ferrolearn does
not model output-shape polymorphism (multi-output / proba matrices), which is
part of why REQ-3 is unstarted.

Invariant: a `FittedPipeline` always has exactly one final estimator (the type
guarantees it), so unlike sklearn there is no runtime `_final_estimator ==
"passthrough"` branch (`:471`) — passthrough is REQ-5, unstarted.

Production consumers prove the trait surface is real public API, not a test-only
fixture: `ferrolearn-bayes` implements `PipelineEstimator`/`FittedPipelineEstimator`
for `GaussianNB` and `BernoulliNB` (mapping the float `y` to class indices and
back), and `ferrolearn-decomp` implements `PipelineTransformer`/
`FittedPipelineTransformer` for `KernelPCA` (unsupervised, ignoring `y`).

## Verification

- `cargo test -p ferrolearn-core` → 98 passed, 0 failed (lib suite); the
  `pipeline` module's 29 tests (incl. `typed_pipeline`) pass. Establishes
  REQ-1 (fit→transform→predict for f32 and f64) and REQ-2 (no-estimator error).
- REQ-2 oracle (installed sklearn 1.5.2, run outside the unbuilt source tree):

  ```
  python3 -c "from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler; import numpy as np; \
    X=np.array([[1.,2.],[3.,4.]]); y=np.array([0.,1.]); \
    Pipeline(steps=[]).fit(X,y)"
  # -> ValueError: not enough values to unpack (expected 2, got 0)

  python3 -c "...; Pipeline([('s',StandardScaler())]).fit(X).predict(X)"
  # -> AttributeError: This 'Pipeline' has no attribute 'predict'
  ```

  Confirms sklearn mandates a final predictor for `.predict`, matching
  ferrolearn's `InvalidParameter` at fit.
- Non-test consumer presence (grandfathered public API per S5/R-DOC-5, AND a real
  production consumer exists):

  ```
  grep -rn "PipelineEstimator\|PipelineTransformer" ferrolearn-*/src/ \
    | grep -v '#\[cfg(test' | grep -v '/tests/'
  # ferrolearn-bayes/src/gaussian.rs   impl PipelineEstimator for GaussianNB
  # ferrolearn-bayes/src/bernoulli.rs  impl PipelineEstimator for BernoulliNB
  # ferrolearn-decomp/src/kernel_pca.rs impl PipelineTransformer for KernelPCA
  ```

- REQ-3..REQ-9 have no green verification because the surface is unimplemented;
  each is NOT-STARTED against its blocker (#361–#367). They will get
  characterization tests when built.
