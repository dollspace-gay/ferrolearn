//! Dynamic-dispatch pipeline for composing transformers and estimators.
//!
//! A [`Pipeline`] chains zero or more transformer steps followed by a final
//! estimator step. Calling [`Fit::fit`] on a pipeline fits each step in
//! sequence, producing a [`FittedPipeline`] that implements [`Predict`].
//!
//! The pipeline is generic over the float type `F`, supporting both `f32`
//! and `f64` data. All steps in a pipeline must use the same float type.
//! The type parameter defaults to `f64` for backward compatibility.
//!
//! ## REQ status (per `.design/core/pipeline.md`, mirrors `sklearn/pipeline.py` @ 1.5.2)
//!
//! ferrolearn's `Pipeline` is a minimal subset of sklearn's: sequential
//! transformer fit→transform chaining + a single final estimator's fit/predict.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (fit→transform chaining + final predict) | SHIPPED | `Fit for Pipeline` (fit each transformer, transform, fit final estimator) mirrors `Pipeline._fit` (`pipeline.py:406`); `Predict for FittedPipeline` mirrors `Pipeline.predict` (`pipeline.py:599`). Non-test consumers: `impl PipelineEstimator for GaussianNB in gaussian.rs`, `impl PipelineEstimator for BernoulliNB in bernoulli.rs`, `impl PipelineTransformer for KernelPCA in kernel_pca.rs`. (critic: fit-then-transform ≡ sklearn fused fit_transform to ≤1.1e-14 on KernelPCA.) |
//! | REQ-2 (no-final-estimator rejected at fit) | SHIPPED | `Fit for Pipeline` returns `FerroError::InvalidParameter` when the estimator slot is unset; matches sklearn requiring a final predictor for `.predict` (`available_if` at `pipeline.py:549`). |
//! | REQ-3 (fit_transform/transform/predict_proba/decision_function/score) | SHIPPED | `Pipeline::fit_transform` (`Fit::fit` then `transform_through`) mirrors `Pipeline.fit_transform` (`pipeline.py:489`); `FittedPipeline::{transform, predict_proba, decision_function, score}` run the private `transform_through` loop (`pipeline.py:599-600`/`:719-720`/`:768-769`/`:999-1000`) then delegate to the final estimator. `predict_proba`/`decision_function`/`score` forward to the new default-`Err` trait methods `predict_proba_pipeline`/`decision_function_pipeline`/`score_pipeline` on `FittedPipelineEstimator` (the `available_if(_final_estimator_has(...))` analog, `pipeline.py:674`/`:731`/`:960`); `transform` returns the transformer-prefix output (sklearn raises `AttributeError` for a non-transformer-final `transform`, `:858`). Non-test consumer: `impl FittedPipelineEstimator for FittedGaussianNBPipeline in gaussian.rs` overrides `predict_proba_pipeline` (→ `predict_proba`) + `score_pipeline` (→ `score`). Live-oracle verification: `gaussian_pipeline_predict_proba_score_match_sklearn` (StandardScaler+GaussianNB pipeline matches sklearn `predict_proba`/`score`/`transform`) + core `test_pipeline_fit_transform_equals_transform`/`test_pipeline_predict_proba_and_score_override`/`test_pipeline_predict_proba_default_is_err`. |
//! | REQ-4a (named_steps / `__getitem__` int+str+slice) | SHIPPED | `Pipeline::{named_steps, get_step, get_step_by_name, named_step, into_slice}` + `FittedPipeline::{named_steps, get_step, get_step_by_name, named_step}` over the existing `transforms`/`estimator` storage; mirror `Pipeline.named_steps` (`pipeline.py:325` `return Bunch(**dict(self.steps))`), integer/string/slice `Pipeline.__getitem__` (`pipeline.py:298-318`). A step is returned as a `PipelineStepRef`/`FittedPipelineStepRef` enum (the heterogeneous-`(name, obj)`-list analog, since a ferrolearn step is EITHER a `PipelineTransformer` OR a `PipelineEstimator`). `into_slice` consumes `self` (the trait-object steps are not `Clone`, so the new sub-pipeline MOVES the contiguous range, vs sklearn's shallow object-sharing copy `:310`). Non-test consumer: pub API on the grandfathered `Pipeline`/`FittedPipeline` boundary types (S5). Live-oracle verification (R-CHAR-3, sklearn 1.5.2): `test_pipeline_named_steps_match_sklearn`, `test_pipeline_get_step_*`, `test_pipeline_into_slice_*`. |
//! | REQ-4b (get_params / set_params `<step>__<param>` nested protocol) | NOT-STARTED | blocker #362. The `PipelineTransformer`/`PipelineEstimator` trait objects expose NO `get_params`/reflection method, so the `_BaseComposition._get_params`/`_set_params` nested addressing (`pipeline.py:216`/`:237`) is not implementable without first adding a per-step reflection trait (e.g. `fn get_params(&self) -> BTreeMap<String, ParamValue>` on the step traits). Concrete blocker: a `get_params`/reflection method on the step traits. |
//! | REQ-5a (passthrough steps) | SHIPPED | `PassthroughTransformer`/`FittedPassthroughTransformer in pipeline.rs` are a reusable identity transformer (`impl PipelineTransformer` `fit_pipeline` → `Box::new(FittedPassthroughTransformer)`; `transform_pipeline(&self, x) → Ok(x.clone())`), the Rust analog of sklearn's `'passthrough'`/`None` step (`sklearn/pipeline.py:251`/`:266` `_validate_steps` allows it, `:275-290` `_iter(filter_passthrough=True)` skips it so `Xt` passes through, `:337` it stays visible in `named_steps`). ferrolearn types the transformer/estimator split, so the no-op IS a concrete identity transformer placed in the chain — no `filter_passthrough` loop branch needed (the step is genuinely identity). Non-test production consumer: the pub `Pipeline::passthrough_step` builder (the `('name','passthrough')` analog, delegating to `transform_step` with a `PassthroughTransformer`), plus the pub API on the `pub mod pipeline` surface (S5 — same boundary as `Pipeline`/`FeatureUnion`, not crate-root re-exported). Live-oracle (R-CHAR-3, sklearn 1.5.2): `Pipeline([('p','passthrough')]).fit(X).transform(X) == X`; passthrough before/after a transformer == the transformer alone; the step appears in `named_steps`/`steps`. Pinned by `test_passthrough_step_is_identity`, `test_passthrough_before_transformer_is_noop`, `test_passthrough_after_transformer_is_noop`, `test_passthrough_step_appears_in_step_names`, `test_passthrough_transformer_standalone_identity`, `test_passthrough_transformer_f32`. |
//! | REQ-5b (memory caching) | NOT-STARTED | blocker #363. No `memory=`/`check_memory`/`fit_transform_one_cached` transformer caching (`sklearn/pipeline.py:388-390`); requires a joblib disk-cache substrate with no ferrolearn analog yet. |
//! | REQ-6 (fit_params / metadata routing) | NOT-STARTED | blocker #364. |
//! | REQ-7 (make_pipeline auto-naming helper) | NOT-STARTED | blocker #365 (`pipeline.py:1220`). |
//! | REQ-8 (FeatureUnion) | SHIPPED | `FeatureUnion`/`FittedFeatureUnion` in `pipeline.rs`: `impl Fit<Array2<F>, ()> for FeatureUnion` fits each named sub-transformer on the SAME `x` (mirrors `FeatureUnion.fit` fitting every transformer on `X`, `pipeline.py:1643`/`:1681`) recording each output width; the fit also validates transformer-name uniqueness up front (mirrors `_validate_transformers` → `_validate_names`, `pipeline.py:1523-1525` → `sklearn/utils/metaestimators.py:81-83`): a duplicate name returns `FerroError::InvalidParameter` (sklearn's `ValueError: Names provided are not unique` analog) instead of fitting; `impl Transform<Array2<F>> for FittedFeatureUnion` transforms `x` through each and horizontally concatenates the column blocks left-to-right in list order (mirrors `FeatureUnion.transform` → `_hstack`, `pipeline.py:1770`/`:1812` `np.hstack(Xs)`); `FittedFeatureUnion::get_feature_names_out` prefixes each block's positional `x{j}` with `{name}__` (the `verbose_feature_names_out=True` default, `pipeline.py:1567`/`:1608-1616`). Non-test consumer: the pub API on the `pub mod pipeline` surface (S5 — the same boundary the grandfathered `Pipeline`/`FittedPipeline` types live on; neither is crate-root re-exported). Live-oracle (sklearn 1.5.2): `FeatureUnion([('ss',StandardScaler()),('mm',MinMaxScaler())])` on `[[1,2],[3,4],[5,6]]` → `(3,4)` with column blocks `[ss|mm]` and names `['ss__x0','ss__x1','mm__x0','mm__x1']`. NOT-STARTED (no ferrolearn analog yet): `transformer_weights` per-output scaling (`pipeline.py:1369`), the `'drop'`/`'passthrough'` sentinels (`:1530`/`:1563`), `n_jobs` parallelism (`:1360`), metadata routing (`:1859`), `verbose_feature_names_out=False` non-prefixed mode (`:1618-1641`), and the ferray substrate (typed on `ndarray::{Array1,Array2}`). |
//! | REQ-9 (ferray substrate) | NOT-STARTED | blocker #367 — data flow typed on `ndarray::{Array1,Array2}`; cascades (R-SUBSTRATE-4). |
//!
//! acto-critic verdict: NO DIVERGENCE FOUND in the implemented surface (chaining,
//! y-threading, estimator-only predict, and the REQ-3 apply methods
//! — `fit_transform`/`transform`/`predict_proba`/`decision_function`/`score` —
//! all match the live sklearn oracle; `transform` over a non-transformer-final
//! pipeline returns the transformer-prefix output, the structural analog of
//! sklearn's `available_if(_can_transform)` `AttributeError`). Two states only
//! per goal.md R-DEFER-2.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_core::pipeline::{Pipeline, PipelineTransformer, PipelineEstimator};
//! use ferrolearn_core::{Fit, Predict, FerroError};
//! use ndarray::{Array1, Array2};
//!
//! // A trivial identity transformer for demonstration.
//! struct IdentityTransformer;
//!
//! impl PipelineTransformer<f64> for IdentityTransformer {
//!     fn fit_pipeline(
//!         &self,
//!         x: &Array2<f64>,
//!         _y: &Array1<f64>,
//!     ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
//!         Ok(Box::new(FittedIdentity))
//!     }
//! }
//!
//! struct FittedIdentity;
//!
//! impl FittedPipelineTransformer<f64> for FittedIdentity {
//!     fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
//!         Ok(x.clone())
//!     }
//! }
//!
//! // A trivial estimator that predicts the first column.
//! struct FirstColumnEstimator;
//!
//! impl PipelineEstimator<f64> for FirstColumnEstimator {
//!     fn fit_pipeline(
//!         &self,
//!         _x: &Array2<f64>,
//!         _y: &Array1<f64>,
//!     ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
//!         Ok(Box::new(FittedFirstColumn))
//!     }
//! }
//!
//! struct FittedFirstColumn;
//!
//! impl FittedPipelineEstimator<f64> for FittedFirstColumn {
//!     fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
//!         Ok(x.column(0).to_owned())
//!     }
//! }
//!
//! // Build and use the pipeline.
//! use ferrolearn_core::pipeline::FittedPipelineTransformer;
//! use ferrolearn_core::pipeline::FittedPipelineEstimator;
//!
//! let pipeline = Pipeline::new()
//!     .transform_step("scaler", Box::new(IdentityTransformer))
//!     .estimator_step("model", Box::new(FirstColumnEstimator));
//!
//! let x = Array2::<f64>::zeros((5, 3));
//! let y = Array1::<f64>::zeros(5);
//!
//! let fitted = pipeline.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 5);
//! ```

use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::dataset::check_consistent_length;
use crate::error::FerroError;
use crate::traits::{Fit, Predict, Transform};

// ---------------------------------------------------------------------------
// Trait-object interfaces for pipeline steps
// ---------------------------------------------------------------------------

/// An unfitted transformer step that can participate in a [`Pipeline`].
///
/// Implementors must be able to fit themselves on `Array2<F>` data and
/// return a boxed [`FittedPipelineTransformer`].
///
/// The type parameter `F` is the float type (`f32` or `f64`).
pub trait PipelineTransformer<F: Float + Send + Sync + 'static>: Send + Sync {
    /// Fit this transformer on the given data.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if fitting fails.
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError>;
}

/// A fitted transformer step in a [`FittedPipeline`].
///
/// Transforms `Array2<F>` data, producing a new `Array2<F>`.
pub trait FittedPipelineTransformer<F: Float + Send + Sync + 'static>: Send + Sync {
    /// Transform the input data.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if the input shape is incompatible.
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError>;
}

/// An unfitted estimator step that serves as the final step in a [`Pipeline`].
///
/// Implementors must be able to fit themselves on `Array2<F>` data and
/// return a boxed [`FittedPipelineEstimator`].
pub trait PipelineEstimator<F: Float + Send + Sync + 'static>: Send + Sync {
    /// Fit this estimator on the given data.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if fitting fails.
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError>;
}

/// A fitted estimator step in a [`FittedPipeline`].
///
/// Produces `Array1<F>` predictions from `Array2<F>` input.
///
/// The three delegating methods below — `predict_proba_pipeline`,
/// `decision_function_pipeline`, `score_pipeline` — mirror the way sklearn's
/// `Pipeline` forwards to the final estimator's `predict_proba` /
/// `decision_function` / `score` (`sklearn/pipeline.py:675`, `:731`, `:961`).
/// scikit-learn gates each pipeline method on the final estimator actually
/// having the attribute via `available_if(_final_estimator_has(...))`
/// (`sklearn/pipeline.py:674`, `:731`, `:960`); a final estimator that lacks
/// the method raises `AttributeError`. ferrolearn cannot express
/// `available_if` over a trait object, so each method ships a DEFAULT impl that
/// returns [`FerroError::InvalidParameter`] (the closest analog of sklearn's
/// `AttributeError`). A concrete estimator that DOES support the operation
/// overrides the corresponding method.
pub trait FittedPipelineEstimator<F: Float + Send + Sync + 'static>: Send + Sync {
    /// Generate predictions for the input data.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if the input shape is incompatible.
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError>;

    /// Class-probability estimates for the input data, shape
    /// `(n_samples, n_classes)`.
    ///
    /// Mirrors the final-estimator delegation of `Pipeline.predict_proba`
    /// (`sklearn/pipeline.py:721`: `self.steps[-1][1].predict_proba(Xt)`).
    ///
    /// # Errors
    ///
    /// The default implementation returns [`FerroError::InvalidParameter`] —
    /// the analog of sklearn raising `AttributeError` when the final estimator
    /// has no `predict_proba`. Estimators that support probability estimates
    /// override this method.
    fn predict_proba_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let _ = x;
        Err(FerroError::InvalidParameter {
            name: "predict_proba".into(),
            reason: "the final estimator of this pipeline does not support predict_proba".into(),
        })
    }

    /// Confidence scores (decision function) for the input data, shape
    /// `(n_samples, n_classes)` (or `(n_samples,)` for binary, per the
    /// estimator's contract).
    ///
    /// Mirrors the final-estimator delegation of `Pipeline.decision_function`
    /// (`sklearn/pipeline.py:772`: `self.steps[-1][1].decision_function(Xt)`).
    ///
    /// # Errors
    ///
    /// The default implementation returns [`FerroError::InvalidParameter`] —
    /// the analog of sklearn raising `AttributeError` when the final estimator
    /// has no `decision_function`. Estimators that expose a decision function
    /// override this method.
    fn decision_function_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let _ = x;
        Err(FerroError::InvalidParameter {
            name: "decision_function".into(),
            reason: "the final estimator of this pipeline does not support decision_function"
                .into(),
        })
    }

    /// Score the final estimator on `(x, y)`, returning a single scalar
    /// (e.g. mean accuracy for a classifier, R² for a regressor).
    ///
    /// Mirrors the final-estimator delegation of `Pipeline.score`
    /// (`sklearn/pipeline.py:1004`: `self.steps[-1][1].score(Xt, y)`).
    ///
    /// # Errors
    ///
    /// The default implementation returns [`FerroError::InvalidParameter`] —
    /// the analog of sklearn raising `AttributeError` when the final estimator
    /// has no `score`. Estimators that support scoring override this method.
    fn score_pipeline(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        let _ = (x, y);
        Err(FerroError::InvalidParameter {
            name: "score".into(),
            reason: "the final estimator of this pipeline does not support score".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Pipeline (unfitted)
// ---------------------------------------------------------------------------

/// A named transformer step in an unfitted pipeline.
struct TransformStep<F: Float + Send + Sync + 'static> {
    /// Human-readable name for this step.
    name: String,
    /// The unfitted transformer.
    step: Box<dyn PipelineTransformer<F>>,
}

/// A borrowed reference to a single step of an unfitted [`Pipeline`].
///
/// sklearn's `Pipeline.steps` is a flat list of `(name, obj)` tuples where
/// every `obj` is duck-typed; `Pipeline.__getitem__` with an integer or string
/// returns that single `obj` (`sklearn/pipeline.py:298-318`). ferrolearn encodes
/// the transformer/estimator distinction in the type system, so a "step" is
/// EITHER a [`PipelineTransformer`] OR a [`PipelineEstimator`]. This enum is the
/// heterogeneous-step analog: the variant tells the caller which kind of step
/// they reached, mirroring sklearn returning the underlying object.
pub enum PipelineStepRef<'a, F: Float + Send + Sync + 'static> {
    /// A transformer step (an intermediate step of the pipeline).
    Transformer(&'a dyn PipelineTransformer<F>),
    /// The final estimator step.
    Estimator(&'a dyn PipelineEstimator<F>),
}

/// A dynamic-dispatch pipeline that composes transformers and a final estimator.
///
/// Steps are added with [`transform_step`](Pipeline::transform_step) and the
/// final estimator is set with [`estimator_step`](Pipeline::estimator_step).
/// The pipeline implements [`Fit<Array2<F>, Array1<F>>`](Fit) and produces
/// a [`FittedPipeline`] that implements [`Predict<Array2<F>>`](Predict).
///
/// All intermediate data flows as `Array2<F>`. The type parameter defaults
/// to `f64` for backward compatibility.
pub struct Pipeline<F: Float + Send + Sync + 'static = f64> {
    /// Ordered transformer steps.
    transforms: Vec<TransformStep<F>>,
    /// The final estimator step (name + estimator).
    estimator: Option<(String, Box<dyn PipelineEstimator<F>>)>,
}

impl<F: Float + Send + Sync + 'static> Pipeline<F> {
    /// Create a new empty pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrolearn_core::pipeline::Pipeline;
    /// let pipeline = Pipeline::<f64>::new();
    /// ```
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            estimator: None,
        }
    }

    /// Add a named transformer step to the pipeline.
    ///
    /// Transformer steps are applied in the order they are added, before
    /// the final estimator step.
    #[must_use]
    pub fn transform_step(mut self, name: &str, step: Box<dyn PipelineTransformer<F>>) -> Self {
        self.transforms.push(TransformStep {
            name: name.to_owned(),
            step,
        });
        self
    }

    /// Add a named `'passthrough'` (identity no-op) transformer step.
    ///
    /// This is the ergonomic analog of an sklearn `('name', 'passthrough')` step:
    /// a transformer that leaves the running data unchanged but is still a real,
    /// named step (visible in [`step_names`](Pipeline::step_names) /
    /// [`named_steps`](Pipeline::named_steps)). It delegates to
    /// [`transform_step`](Pipeline::transform_step) with a
    /// [`PassthroughTransformer`], so a passthrough step placed anywhere in the
    /// chain is a genuine no-op — fitting skips it and transforming passes `Xt`
    /// through unchanged, mirroring sklearn's `_iter(filter_passthrough=True)`
    /// dropping `'passthrough'` (`sklearn/pipeline.py:289`) while
    /// `named_steps`/`__getitem__` still show it (`:337`).
    #[must_use]
    pub fn passthrough_step(self, name: &str) -> Self {
        self.transform_step(name, Box::new(PassthroughTransformer::<F>::new()))
    }

    /// Set the final estimator step.
    ///
    /// A pipeline must have exactly one estimator step. Setting a new
    /// estimator replaces any previously set estimator.
    #[must_use]
    pub fn estimator_step(mut self, name: &str, estimator: Box<dyn PipelineEstimator<F>>) -> Self {
        self.estimator = Some((name.to_owned(), estimator));
        self
    }

    /// Add a named step to the pipeline using the builder pattern.
    ///
    /// This is a convenience method that accepts either a transformer or
    /// an estimator. The final step added via this method that is an
    /// estimator becomes the pipeline's estimator. This provides the
    /// `Pipeline::new().step("scaler", ...).step("clf", ...)` API.
    #[must_use]
    pub fn step(self, name: &str, step: Box<dyn PipelineStep<F>>) -> Self {
        step.add_to_pipeline(self, name)
    }

    /// Fit the pipeline and return both the [`FittedPipeline`] and the data
    /// after every transformer step has been applied.
    ///
    /// This mirrors `Pipeline.fit_transform` (`sklearn/pipeline.py:489-547`):
    /// `Xt = self._fit(X, y)` fits each transformer on the running `Xt` and
    /// applies it, then the result is the transformed data. sklearn ALSO calls
    /// the final estimator's `fit_transform`/`fit().transform()` when the final
    /// step is itself a transformer (`:540-547`); ferrolearn's final slot is a
    /// non-transformer estimator, so — like its [`FittedPipeline::transform`] —
    /// `fit_transform` returns the data after the transformer prefix, with the
    /// estimator still fit (as in `fit`). The returned `Array2<F>` equals
    /// [`FittedPipeline::transform`] applied to the same `x` (fit-then-transform
    /// ≡ sklearn's fused `fit_transform`, established for REQ-1).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if no estimator step was set
    /// (delegates to [`Fit::fit`]). Propagates any errors from individual step
    /// fitting or transforming.
    pub fn fit_transform(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<(FittedPipeline<F>, Array2<F>), FerroError> {
        let fitted = self.fit(x, y)?;
        let transformed = fitted.transform_through(x)?;
        Ok((fitted, transformed))
    }

    /// Number of steps in the pipeline (transformer steps plus the final
    /// estimator, if set).
    ///
    /// Mirrors `Pipeline.__len__` (`sklearn/pipeline.py:292-296`:
    /// `return len(self.steps)`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.transforms.len() + usize::from(self.estimator.is_some())
    }

    /// Returns `true` if the pipeline has no steps at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the names of all steps (transformers, then the estimator if set)
    /// in pipeline order.
    ///
    /// Mirrors the key ordering of `Pipeline.named_steps`
    /// (`sklearn/pipeline.py:325`: `Bunch(**dict(self.steps))` keyed by step
    /// name in `steps` order).
    #[must_use]
    pub fn step_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.transforms.iter().map(|s| s.name.as_str()).collect();
        if let Some((name, _)) = &self.estimator {
            names.push(name.as_str());
        }
        names
    }

    /// Access every step by its name, in pipeline order, as a
    /// `(name, step)` list.
    ///
    /// This is the trait-object analog of sklearn's `Pipeline.named_steps`,
    /// which returns a `Bunch(**dict(self.steps))` — a name→step mapping
    /// (`sklearn/pipeline.py:325`). Every step (each transformer, then the final
    /// estimator if set) is reachable by its construction name. ferrolearn
    /// returns an ordered `Vec` of `(name, PipelineStepRef)` rather than a hash
    /// map so the pipeline order is preserved and the heterogeneous
    /// transformer/estimator kinds are distinguishable.
    #[must_use]
    pub fn named_steps(&self) -> Vec<(&str, PipelineStepRef<'_, F>)> {
        let mut steps: Vec<(&str, PipelineStepRef<'_, F>)> = self
            .transforms
            .iter()
            .map(|s| {
                (
                    s.name.as_str(),
                    PipelineStepRef::Transformer(s.step.as_ref()),
                )
            })
            .collect();
        if let Some((name, est)) = &self.estimator {
            steps.push((name.as_str(), PipelineStepRef::Estimator(est.as_ref())));
        }
        steps
    }

    /// Look up a single step by name.
    ///
    /// This is the string-key arm of sklearn's `Pipeline.__getitem__`
    /// (`sklearn/pipeline.py:317`: `return self.named_steps[ind]`), which raises
    /// `KeyError` for an unknown name; ferrolearn returns `None` (R-CODE-2: no
    /// panic).
    #[must_use]
    pub fn named_step(&self, name: &str) -> Option<PipelineStepRef<'_, F>> {
        if let Some(ts) = self.transforms.iter().find(|s| s.name == name) {
            return Some(PipelineStepRef::Transformer(ts.step.as_ref()));
        }
        match &self.estimator {
            Some((est_name, est)) if est_name == name => {
                Some(PipelineStepRef::Estimator(est.as_ref()))
            }
            _ => None,
        }
    }

    /// Get the step at position `index` (0-based, transformer steps first then
    /// the final estimator).
    ///
    /// This is the integer arm of sklearn's `Pipeline.__getitem__`
    /// (`sklearn/pipeline.py:313-318`: `name, est = self.steps[ind]; return
    /// est`), which raises `IndexError` out of range; ferrolearn returns `None`
    /// (R-CODE-2: no panic).
    #[must_use]
    pub fn get_step(&self, index: usize) -> Option<PipelineStepRef<'_, F>> {
        let n_transforms = self.transforms.len();
        if index < n_transforms {
            return Some(PipelineStepRef::Transformer(
                self.transforms[index].step.as_ref(),
            ));
        }
        if index == n_transforms
            && let Some((_, est)) = &self.estimator
        {
            return Some(PipelineStepRef::Estimator(est.as_ref()));
        }
        None
    }

    /// Look up a single step by name (alias of [`named_step`](Pipeline::named_step)).
    ///
    /// Provided for symmetry with [`get_step`](Pipeline::get_step); mirrors the
    /// string arm of `Pipeline.__getitem__` (`sklearn/pipeline.py:317`).
    #[must_use]
    pub fn get_step_by_name(&self, name: &str) -> Option<PipelineStepRef<'_, F>> {
        self.named_step(name)
    }

    /// Build a sub-pipeline from the contiguous step range `[start, end)`,
    /// consuming `self`.
    ///
    /// This is the slice arm of sklearn's `Pipeline.__getitem__`
    /// (`sklearn/pipeline.py:307-312`): `pipe[a:b]` returns
    /// `Pipeline(self.steps[a:b], ...)` — a new pipeline over the contiguous
    /// step range. sklearn slicing supports only a step of 1
    /// (`:308-309`, otherwise `ValueError`); a contiguous Rust range is the step-1
    /// analog by construction.
    ///
    /// The sliced steps are addressed in the unified order
    /// (transformer steps `0..n_transforms`, then the estimator at
    /// `n_transforms` if set), matching [`get_step`](Pipeline::get_step). A slice
    /// that includes the estimator index keeps it as the final estimator; a slice
    /// of only transformer indices yields an estimator-less pipeline (valid to
    /// build, errors only at `fit` — mirroring sklearn, where `pipe[:k]` for a
    /// transformer-only range is a `Pipeline` that simply lacks `.predict`).
    ///
    /// # Divergence from sklearn
    ///
    /// sklearn's slice is a SHALLOW copy that shares the underlying estimator
    /// objects with the original pipeline (`sklearn/pipeline.py:303-305`). The
    /// ferrolearn step trait objects are not `Clone`, so this method MOVES the
    /// selected boxed steps into the new pipeline and therefore consumes `self`.
    /// Slicing a [`FittedPipeline`] is NOT implemented for the same reason (the
    /// fitted step trait objects are not `Clone`); it is NOT-STARTED under
    /// blocker #362.
    ///
    /// Out-of-range bounds CLAMP and `start > end` yields an empty pipeline —
    /// Python list-slice semantics, mirroring sklearn `Pipeline.__getitem__`'s
    /// slice arm which slices `self.steps[ind]` (`pipeline.py:307-312`): an
    /// ordinary Python slice never raises on out-of-range bounds (#2235). So
    /// `into_slice(0, 100)` on 3 steps → all 3, `into_slice(5, 100)` → empty,
    /// `into_slice(2, 1)` → empty. This is a TOTAL function (it cannot fail).
    #[must_use]
    pub fn into_slice(self, start: usize, end: usize) -> Pipeline<F> {
        let n_steps = self.len();
        // Python slice clamping: `end` past the length is clamped to the length;
        // a `start >= end` (incl. start past the length) yields an empty range
        // via the `idx >= start && idx < end` filter below.
        let end = end.min(n_steps);

        let Pipeline {
            transforms,
            estimator,
        } = self;
        let n_transforms = transforms.len();

        let mut new_transforms = Vec::new();
        let mut new_estimator = None;
        for (idx, ts) in transforms.into_iter().enumerate() {
            if idx >= start && idx < end {
                new_transforms.push(ts);
            }
        }
        // The estimator (if set) sits at unified index `n_transforms`.
        if let Some(est) = estimator
            && n_transforms >= start
            && n_transforms < end
        {
            new_estimator = Some(est);
        }

        Pipeline {
            transforms: new_transforms,
            estimator: new_estimator,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for Pipeline<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for Pipeline<F> {
    type Fitted = FittedPipeline<F>;
    type Error = FerroError;

    /// Fit the pipeline by fitting each transformer step in order, then
    /// fitting the final estimator on the transformed data.
    ///
    /// Each transformer is fit on the current data, then the data is
    /// transformed before being passed to the next step.
    ///
    /// Before fitting any step, the pipeline validates that `x` and `y` have a
    /// consistent number of samples via
    /// [`check_consistent_length`](crate::dataset::check_consistent_length),
    /// mirroring scikit-learn's `Pipeline.fit`, which runs every step through
    /// input validation (`check_X_y` → `check_consistent_length`,
    /// `sklearn/utils/validation.py:1320`) and rejects `X`/`y` with mismatched
    /// `n_samples` before fitting (`sklearn/pipeline.py:406` `_fit`). A pipeline
    /// therefore rejects inconsistent `X`/`y` up front rather than failing
    /// inside a step's `fit_pipeline`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if no estimator step was set, or
    /// [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`. Propagates any
    /// errors from individual step fitting or transforming.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedPipeline<F>, FerroError> {
        if self.estimator.is_none() {
            return Err(FerroError::InvalidParameter {
                name: "estimator".into(),
                reason: "pipeline must have a final estimator step".into(),
            });
        }

        // sklearn validates X/y sample-count consistency before fitting any
        // step (`check_consistent_length`, `sklearn/utils/validation.py:1320`).
        check_consistent_length(x.nrows(), y.len())?;

        let mut current_x = x.clone();
        let mut fitted_transforms = Vec::with_capacity(self.transforms.len());

        // Fit and transform each transformer step.
        for ts in &self.transforms {
            let fitted = ts.step.fit_pipeline(&current_x, y)?;
            current_x = fitted.transform_pipeline(&current_x)?;
            fitted_transforms.push(FittedTransformStep {
                name: ts.name.clone(),
                step: fitted,
            });
        }

        // Fit the final estimator on the transformed data.
        let (est_name, est) = self.estimator.as_ref().unwrap();
        let fitted_est = est.fit_pipeline(&current_x, y)?;

        Ok(FittedPipeline {
            transforms: fitted_transforms,
            estimator: (est_name.clone(), fitted_est),
        })
    }
}

// ---------------------------------------------------------------------------
// FittedPipeline
// ---------------------------------------------------------------------------

/// A named fitted transformer step.
struct FittedTransformStep<F: Float + Send + Sync + 'static> {
    /// Human-readable name for this step.
    name: String,
    /// The fitted transformer.
    step: Box<dyn FittedPipelineTransformer<F>>,
}

/// A borrowed reference to a single step of a [`FittedPipeline`].
///
/// The fitted analog of [`PipelineStepRef`]: a fitted step is EITHER a
/// [`FittedPipelineTransformer`] (an intermediate step) OR the
/// [`FittedPipelineEstimator`] (the final step). Returned by the
/// `FittedPipeline` `named_steps` / `get_step` / `named_step` accessors, the
/// fitted analog of sklearn's `Pipeline.__getitem__` over a fitted pipeline
/// (`sklearn/pipeline.py:298-318`).
pub enum FittedPipelineStepRef<'a, F: Float + Send + Sync + 'static> {
    /// A fitted transformer step.
    Transformer(&'a dyn FittedPipelineTransformer<F>),
    /// The fitted final estimator step.
    Estimator(&'a dyn FittedPipelineEstimator<F>),
}

/// A fitted pipeline that chains fitted transformers and a fitted estimator.
///
/// Created by calling [`Fit::fit`] on a [`Pipeline`]. Implements
/// [`Predict<Array2<F>>`](Predict), producing `Array1<F>` predictions.
pub struct FittedPipeline<F: Float + Send + Sync + 'static = f64> {
    /// Fitted transformer steps, in order.
    transforms: Vec<FittedTransformStep<F>>,
    /// The fitted estimator (name + estimator).
    estimator: (String, Box<dyn FittedPipelineEstimator<F>>),
}

impl<F: Float + Send + Sync + 'static> FittedPipeline<F> {
    /// Returns the names of all steps (transformers + estimator) in order.
    pub fn step_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.transforms.iter().map(|s| s.name.as_str()).collect();
        names.push(&self.estimator.0);
        names
    }

    /// Number of steps in the fitted pipeline (every transformer step plus the
    /// final estimator).
    ///
    /// Mirrors `Pipeline.__len__` (`sklearn/pipeline.py:292-296`). A
    /// `FittedPipeline` always has exactly one final estimator (the type
    /// guarantees it), so this is never zero.
    #[must_use]
    pub fn len(&self) -> usize {
        self.transforms.len() + 1
    }

    /// Always `false`: a fitted pipeline always has at least its final
    /// estimator step.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Access every fitted step by its name, in pipeline order, as a
    /// `(name, step)` list.
    ///
    /// The fitted analog of sklearn's `Pipeline.named_steps`
    /// (`sklearn/pipeline.py:325`: `Bunch(**dict(self.steps))`) — every fitted
    /// step (each transformer, then the final estimator) is reachable by its
    /// construction name, in pipeline order.
    #[must_use]
    pub fn named_steps(&self) -> Vec<(&str, FittedPipelineStepRef<'_, F>)> {
        let mut steps: Vec<(&str, FittedPipelineStepRef<'_, F>)> = self
            .transforms
            .iter()
            .map(|s| {
                (
                    s.name.as_str(),
                    FittedPipelineStepRef::Transformer(s.step.as_ref()),
                )
            })
            .collect();
        steps.push((
            self.estimator.0.as_str(),
            FittedPipelineStepRef::Estimator(self.estimator.1.as_ref()),
        ));
        steps
    }

    /// Look up a single fitted step by name.
    ///
    /// The fitted analog of the string arm of `Pipeline.__getitem__`
    /// (`sklearn/pipeline.py:317`); returns `None` for an unknown name (R-CODE-2:
    /// no panic, vs sklearn's `KeyError`).
    #[must_use]
    pub fn named_step(&self, name: &str) -> Option<FittedPipelineStepRef<'_, F>> {
        if let Some(ts) = self.transforms.iter().find(|s| s.name == name) {
            return Some(FittedPipelineStepRef::Transformer(ts.step.as_ref()));
        }
        if self.estimator.0 == name {
            return Some(FittedPipelineStepRef::Estimator(self.estimator.1.as_ref()));
        }
        None
    }

    /// Get the fitted step at position `index` (0-based, transformer steps
    /// first then the final estimator).
    ///
    /// The fitted analog of the integer arm of `Pipeline.__getitem__`
    /// (`sklearn/pipeline.py:313-318`); returns `None` out of range (R-CODE-2: no
    /// panic, vs sklearn's `IndexError`).
    #[must_use]
    pub fn get_step(&self, index: usize) -> Option<FittedPipelineStepRef<'_, F>> {
        let n_transforms = self.transforms.len();
        if index < n_transforms {
            return Some(FittedPipelineStepRef::Transformer(
                self.transforms[index].step.as_ref(),
            ));
        }
        if index == n_transforms {
            return Some(FittedPipelineStepRef::Estimator(self.estimator.1.as_ref()));
        }
        None
    }

    /// Look up a single fitted step by name (alias of
    /// [`named_step`](FittedPipeline::named_step)).
    ///
    /// Mirrors the string arm of `Pipeline.__getitem__`
    /// (`sklearn/pipeline.py:317`).
    #[must_use]
    pub fn get_step_by_name(&self, name: &str) -> Option<FittedPipelineStepRef<'_, F>> {
        self.named_step(name)
    }

    /// Run `x` through every fitted transformer step in order, returning the
    /// fully transformed data (the data the final estimator sees).
    ///
    /// This is the shared `for ...: Xt = transform.transform(Xt)` loop of
    /// sklearn's `Pipeline.predict` / `predict_proba` / `decision_function` /
    /// `score` (`sklearn/pipeline.py:599-600`, `:719-720`, `:768-769`,
    /// `:999-1000`), which run the data through every non-final transformer
    /// before delegating to the final estimator.
    ///
    /// # Errors
    ///
    /// Propagates any [`FerroError`] from an individual transformer step.
    fn transform_through(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let mut current_x = x.clone();
        for ts in &self.transforms {
            current_x = ts.step.transform_pipeline(&current_x)?;
        }
        Ok(current_x)
    }

    /// Apply every fitted transformer step to `x`, returning the transformed
    /// data without invoking the final estimator.
    ///
    /// This mirrors `Pipeline.transform` (`sklearn/pipeline.py:863-904`) for the
    /// *transformer-final* case. sklearn gates `transform` on
    /// `_can_transform` (`:858`): it is only available when the final step is
    /// itself a transformer, in which case it runs the data through ALL steps
    /// including the last (`for _, name, transform in self._iter(): Xt =
    /// transform.transform(Xt)`). When the final step is a non-transformer
    /// estimator (e.g. `GaussianNB`), sklearn raises `AttributeError`
    /// (`'Pipeline' has no attribute 'transform'`, verified against the live
    /// 1.5.2 oracle).
    ///
    /// ferrolearn's [`FittedPipeline`] structurally separates the transformer
    /// steps from a single non-transformer estimator slot (the estimator is
    /// reached via [`predict_pipeline`](FittedPipelineEstimator::predict_pipeline),
    /// not `transform_pipeline`). Therefore `transform` applies exactly the
    /// transformer steps and returns the data the final estimator would see —
    /// equivalent to sklearn's transformer-final `transform` over the
    /// transformer prefix. The estimator slot is never a transformer, so there
    /// is no "transform the final step too" branch to mirror.
    ///
    /// # Errors
    ///
    /// Propagates any [`FerroError`] from a transformer step (e.g. a feature
    /// count mismatch).
    pub fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform_through(x)
    }

    /// Transform `x` through every fitted transformer step, then return the
    /// final estimator's class-probability estimates, shape
    /// `(n_samples, n_classes)`.
    ///
    /// Mirrors `Pipeline.predict_proba` (`sklearn/pipeline.py:716-721`): run the
    /// data through every non-final transformer, then
    /// `self.steps[-1][1].predict_proba(Xt)`.
    ///
    /// # Errors
    ///
    /// Propagates transformer-step errors; returns [`FerroError::InvalidParameter`]
    /// if the final estimator does not support `predict_proba` (sklearn's
    /// `AttributeError` analog).
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let xt = self.transform_through(x)?;
        self.estimator.1.predict_proba_pipeline(&xt)
    }

    /// Transform `x` through every fitted transformer step, then return the
    /// final estimator's decision-function scores.
    ///
    /// Mirrors `Pipeline.decision_function` (`sklearn/pipeline.py:767-774`): run
    /// the data through every non-final transformer, then
    /// `self.steps[-1][1].decision_function(Xt)`.
    ///
    /// # Errors
    ///
    /// Propagates transformer-step errors; returns [`FerroError::InvalidParameter`]
    /// if the final estimator does not support `decision_function` (sklearn's
    /// `AttributeError` analog).
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let xt = self.transform_through(x)?;
        self.estimator.1.decision_function_pipeline(&xt)
    }

    /// Transform `x` through every fitted transformer step, then return the
    /// final estimator's score on `(Xt, y)` (e.g. mean accuracy for a
    /// classifier).
    ///
    /// Mirrors `Pipeline.score` (`sklearn/pipeline.py:997-1004`): run the data
    /// through every non-final transformer, then
    /// `self.steps[-1][1].score(Xt, y)`. ferrolearn does not yet thread
    /// `sample_weight` (sklearn's optional third argument, `:961`); that is part
    /// of the metadata-routing surface (REQ-6, blocker #364).
    ///
    /// # Errors
    ///
    /// Propagates transformer-step errors; returns [`FerroError::InvalidParameter`]
    /// if the final estimator does not support `score` (sklearn's
    /// `AttributeError` analog).
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        let xt = self.transform_through(x)?;
        self.estimator.1.score_pipeline(&xt, y)
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedPipeline<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Generate predictions by transforming the input through each fitted
    /// transformer step, then calling predict on the fitted estimator.
    ///
    /// # Errors
    ///
    /// Propagates any errors from transformer or estimator steps.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let current_x = self.transform_through(x)?;
        self.estimator.1.predict_pipeline(&current_x)
    }
}

// ---------------------------------------------------------------------------
// PipelineStep: unified interface for the `.step()` builder method
// ---------------------------------------------------------------------------

/// A trait that unifies transformers and estimators for the
/// [`Pipeline::step`] builder method.
///
/// Implementors of [`PipelineTransformer`] and [`PipelineEstimator`]
/// automatically get a blanket implementation of this trait via the
/// wrapper types [`TransformerStepWrapper`] and [`EstimatorStepWrapper`].
///
/// For convenience, use [`as_transform_step`] and [`as_estimator_step`]
/// to wrap your types.
pub trait PipelineStep<F: Float + Send + Sync + 'static>: Send + Sync {
    /// Add this step to the pipeline under the given name.
    ///
    /// Transformer steps are added as intermediate transform steps.
    /// Estimator steps are set as the final estimator.
    fn add_to_pipeline(self: Box<Self>, pipeline: Pipeline<F>, name: &str) -> Pipeline<F>;
}

/// Wraps a [`PipelineTransformer`] to implement [`PipelineStep`].
///
/// Created by [`as_transform_step`].
pub struct TransformerStepWrapper<F: Float + Send + Sync + 'static>(
    Box<dyn PipelineTransformer<F>>,
);

impl<F: Float + Send + Sync + 'static> PipelineStep<F> for TransformerStepWrapper<F> {
    fn add_to_pipeline(self: Box<Self>, pipeline: Pipeline<F>, name: &str) -> Pipeline<F> {
        pipeline.transform_step(name, self.0)
    }
}

/// Wraps a [`PipelineEstimator`] to implement [`PipelineStep`].
///
/// Created by [`as_estimator_step`].
pub struct EstimatorStepWrapper<F: Float + Send + Sync + 'static>(Box<dyn PipelineEstimator<F>>);

impl<F: Float + Send + Sync + 'static> PipelineStep<F> for EstimatorStepWrapper<F> {
    fn add_to_pipeline(self: Box<Self>, pipeline: Pipeline<F>, name: &str) -> Pipeline<F> {
        pipeline.estimator_step(name, self.0)
    }
}

/// Wrap a [`PipelineTransformer`] as a [`PipelineStep`] for use with
/// [`Pipeline::step`].
///
/// # Examples
///
/// ```
/// use ferrolearn_core::pipeline::{Pipeline, as_transform_step};
/// // Assuming `my_scaler` implements PipelineTransformer<f64>:
/// // let pipeline = Pipeline::new().step("scaler", as_transform_step(my_scaler));
/// ```
pub fn as_transform_step<F: Float + Send + Sync + 'static>(
    t: impl PipelineTransformer<F> + 'static,
) -> Box<dyn PipelineStep<F>> {
    Box::new(TransformerStepWrapper(Box::new(t)))
}

/// Wrap a [`PipelineEstimator`] as a [`PipelineStep`] for use with
/// [`Pipeline::step`].
///
/// # Examples
///
/// ```
/// use ferrolearn_core::pipeline::{Pipeline, as_estimator_step};
/// // Assuming `my_model` implements PipelineEstimator<f64>:
/// // let pipeline = Pipeline::new().step("model", as_estimator_step(my_model));
/// ```
pub fn as_estimator_step<F: Float + Send + Sync + 'static>(
    e: impl PipelineEstimator<F> + 'static,
) -> Box<dyn PipelineStep<F>> {
    Box::new(EstimatorStepWrapper(Box::new(e)))
}

// ---------------------------------------------------------------------------
// PassthroughTransformer: the `'passthrough'` step analog (identity no-op)
// ---------------------------------------------------------------------------

/// A no-op transformer step: fit does nothing and transform returns its input
/// unchanged.
///
/// This is the ferrolearn analog of scikit-learn's `'passthrough'` (and `None`)
/// pipeline step. In sklearn, a `Pipeline` step whose object is the string
/// `'passthrough'` (or `None`) is a transformer that is *skipped* during
/// fit/transform — `_iter(filter_passthrough=True)` drops it
/// (`sklearn/pipeline.py:275-290`), so the running `Xt` passes through unchanged
/// — yet the step is still visible in `named_steps` / `steps` / `__getitem__`
/// (`sklearn/pipeline.py:337`: `"passthrough" if estimator is None else
/// estimator`). The net behavior is identity: `Pipeline([('p','passthrough')])
/// .fit(X).transform(X) == X` (verified against the live 1.5.2 oracle).
///
/// ferrolearn encodes the transformer/estimator distinction in the type system
/// (there is no untyped `steps` list to hold a sentinel string), so rather than a
/// `filter_passthrough` branch in the fit/transform loop, the passthrough step is
/// a concrete, reusable *identity transformer*: its `fit_pipeline` is a no-op and
/// its [`FittedPassthroughTransformer::transform_pipeline`] returns `x.clone()`.
/// Placed anywhere in a [`Pipeline`] it leaves the running data unchanged and
/// still appears in [`Pipeline::step_names`] / [`Pipeline::named_steps`], exactly
/// matching sklearn's observable contract. The ergonomic builder
/// [`Pipeline::passthrough_step`] adds one under a given name (the `('name',
/// 'passthrough')` analog).
///
/// The type parameter `F` is the float type (`f32` or `f64`), defaulting to
/// `f64` to match the rest of this module.
///
/// # Examples
///
/// ```
/// use ferrolearn_core::pipeline::{PassthroughTransformer, FittedPipelineTransformer};
/// use ferrolearn_core::pipeline::PipelineTransformer;
/// use ndarray::{Array1, Array2};
///
/// let p = PassthroughTransformer::<f64>::new();
/// let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = Array1::<f64>::zeros(2);
/// let fitted = p.fit_pipeline(&x, &y).unwrap();
/// // Identity: transform returns the input unchanged.
/// assert_eq!(fitted.transform_pipeline(&x).unwrap(), x);
/// ```
pub struct PassthroughTransformer<F: Float + Send + Sync + 'static = f64> {
    /// `PassthroughTransformer` holds no state; the marker ties the no-op to the
    /// float type `F` so it slots into an `F`-typed [`Pipeline`].
    _marker: core::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> PassthroughTransformer<F> {
    /// Create a new passthrough (identity) transformer.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrolearn_core::pipeline::PassthroughTransformer;
    /// let p = PassthroughTransformer::<f64>::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for PassthroughTransformer<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for PassthroughTransformer<F> {
    /// Fitting a passthrough step does nothing (there are no parameters to learn);
    /// it yields a [`FittedPassthroughTransformer`] whose transform is the
    /// identity. Mirrors sklearn skipping a `'passthrough'` step at fit
    /// (`_iter(filter_passthrough=True)`, `sklearn/pipeline.py:289`), so the
    /// running `Xt` is unaffected.
    fn fit_pipeline(
        &self,
        _x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        Ok(Box::new(FittedPassthroughTransformer::new()))
    }
}

/// The fitted half of a [`PassthroughTransformer`]: an identity transform.
///
/// [`transform_pipeline`](FittedPassthroughTransformer::transform_pipeline)
/// returns its input unchanged, the fitted analog of sklearn's skipped
/// `'passthrough'` step leaving the running `Xt` unchanged
/// (`sklearn/pipeline.py:275-290`).
pub struct FittedPassthroughTransformer<F: Float + Send + Sync + 'static = f64> {
    /// No fitted state; the marker ties the identity transform to `F`.
    _marker: core::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FittedPassthroughTransformer<F> {
    /// Create a new fitted passthrough (identity) transformer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for FittedPassthroughTransformer<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F>
    for FittedPassthroughTransformer<F>
{
    /// Return the input unchanged (identity).
    ///
    /// This is the no-op that makes a passthrough step transparent: the data the
    /// next step (or final estimator) sees is exactly what entered. Matches
    /// sklearn's `'passthrough'` net behavior `Pipeline([('p','passthrough')])
    /// .transform(X) == X` (live 1.5.2 oracle).
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Ok(x.clone())
    }
}

// ---------------------------------------------------------------------------
// FeatureUnion (unfitted)
// ---------------------------------------------------------------------------

/// A composite transformer that fits multiple named sub-transformers on the
/// SAME input and horizontally concatenates their outputs.
///
/// This is the ferrolearn analog of scikit-learn's `sklearn.pipeline.FeatureUnion`
/// (`sklearn/pipeline.py:1329`). Where a [`Pipeline`] chains transformers
/// *sequentially* (each transformer sees the previous one's output),
/// `FeatureUnion` applies every transformer *in parallel* to the same `X`, then
/// concatenates the results column-wise: the output width is the sum of each
/// sub-transformer's output width, and the columns appear left-to-right in the
/// order the transformers were added (mirrors `FeatureUnion.transform` →
/// `_hstack` `np.hstack(Xs)`, `sklearn/pipeline.py:1770`/`:1812`).
///
/// `FeatureUnion` reuses the [`PipelineTransformer`] / [`FittedPipelineTransformer`]
/// trait objects already used by [`Pipeline`], so any transformer usable in a
/// pipeline is usable in a feature union.
///
/// The type parameter `F` is the float type (`f32` or `f64`), defaulting to
/// `f64` to match the rest of this module.
///
/// # Divergence from scikit-learn
///
/// This is the core fit / transform / hstack / `get_feature_names_out` subset.
/// `transformer_weights` (per-transformer output scaling,
/// `sklearn/pipeline.py:1369`), the `'drop'` / `'passthrough'` sentinels
/// (`:1530`/`:1563`), `n_jobs` parallelism (`:1360`), metadata routing (`:1859`),
/// and `verbose_feature_names_out=False` (`:1618`) are NOT implemented (REQ-8
/// NOT-STARTED scope). The data substrate is `ndarray`, not yet ferray.
///
/// # Examples
///
/// ```
/// use ferrolearn_core::pipeline::{
///     FeatureUnion, PipelineTransformer, FittedPipelineTransformer,
/// };
/// use ferrolearn_core::{Transform, FerroError};
/// use ndarray::{Array1, Array2};
///
/// // A transformer that returns its input unchanged.
/// struct Identity;
/// impl PipelineTransformer<f64> for Identity {
///     fn fit_pipeline(
///         &self,
///         _x: &Array2<f64>,
///         _y: &Array1<f64>,
///     ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
///         Ok(Box::new(FittedIdentity))
///     }
/// }
/// struct FittedIdentity;
/// impl FittedPipelineTransformer<f64> for FittedIdentity {
///     fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
///         Ok(x.clone())
///     }
/// }
///
/// use ferrolearn_core::Fit;
/// let union = FeatureUnion::<f64>::new()
///     .with_transformer("a", Box::new(Identity))
///     .with_transformer("b", Box::new(Identity));
/// let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let fitted = union.fit(&x, &()).unwrap();
/// // Two identity transformers → output width 2 + 2 = 4.
/// let out = fitted.transform(&x).unwrap();
/// assert_eq!(out.dim(), (2, 4));
/// assert_eq!(fitted.get_feature_names_out(), vec!["a__x0", "a__x1", "b__x0", "b__x1"]);
/// ```
pub struct FeatureUnion<F: Float + Send + Sync + 'static = f64> {
    /// Ordered named transformers, all fit on the same input.
    transformer_list: Vec<(String, Box<dyn PipelineTransformer<F>>)>,
}

impl<F: Float + Send + Sync + 'static> FeatureUnion<F> {
    /// Create a new empty feature union.
    ///
    /// Sub-transformers are added with
    /// [`with_transformer`](FeatureUnion::with_transformer). An empty union fits
    /// successfully and transforms to a `(n_samples, 0)` matrix (the empty
    /// `np.hstack` analog).
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrolearn_core::pipeline::FeatureUnion;
    /// let union = FeatureUnion::<f64>::new();
    /// assert_eq!(union.n_transformers(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            transformer_list: Vec::new(),
        }
    }

    /// Add a named transformer to the union using the builder pattern.
    ///
    /// Mirrors an entry of sklearn's `transformer_list`
    /// (`sklearn/pipeline.py:1348`). Transformers are applied in the order they
    /// are added; their outputs are concatenated left-to-right.
    #[must_use]
    pub fn with_transformer(mut self, name: &str, t: Box<dyn PipelineTransformer<F>>) -> Self {
        self.transformer_list.push((name.to_owned(), t));
        self
    }

    /// Returns the names of all sub-transformers, in union order.
    ///
    /// Mirrors the key order of sklearn's `named_transformers`
    /// (`sklearn/pipeline.py:1478`: `Bunch(**dict(self.transformer_list))`).
    #[must_use]
    pub fn transformer_names(&self) -> Vec<&str> {
        self.transformer_list
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Number of sub-transformers in the union.
    #[must_use]
    pub fn n_transformers(&self) -> usize {
        self.transformer_list.len()
    }
}

impl<F: Float + Send + Sync + 'static> Default for FeatureUnion<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for FeatureUnion<F> {
    type Fitted = FittedFeatureUnion<F>;
    type Error = FerroError;

    /// Fit every sub-transformer on the SAME input `x`.
    ///
    /// Mirrors `FeatureUnion.fit` (`sklearn/pipeline.py:1643`), which fits each
    /// transformer in `transformer_list` independently on the full `X` (every
    /// transformer sees the same data, unlike the sequential `Pipeline`). The
    /// per-transformer output width is recorded at fit time (by transforming `x`
    /// once) so that `get_feature_names_out` can size each column block.
    ///
    /// # `y` handling
    ///
    /// sklearn's `FeatureUnion` threads `y` to each sub-transformer's `fit`
    /// (`sklearn/pipeline.py:1681`/`_fit_one`), but feature-union transformers are
    /// unsupervised and ignore it. ferrolearn's [`PipelineTransformer::fit_pipeline`]
    /// requires an `Array1<F>` target, so this impl passes an empty
    /// `Array1::zeros(0)` — the union's own `Fit` target type is `()` (it takes no
    /// supervised target), and the empty array is the no-target sentinel handed to
    /// each unsupervised sub-transformer.
    ///
    /// # Errors
    ///
    /// Propagates any [`FerroError`] from an individual sub-transformer's
    /// `fit_pipeline` or its width-probing `transform_pipeline`.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedFeatureUnion<F>, FerroError> {
        // Validate transformer-name uniqueness BEFORE fitting any sub-transformer,
        // mirroring `FeatureUnion._validate_transformers` → `_validate_names`
        // (`sklearn/pipeline.py:1523-1525` → `sklearn/utils/metaestimators.py:81-83`),
        // which sklearn runs on every fit/fit_transform: `if len(set(names)) !=
        // len(names): raise ValueError("Names provided are not unique: {names!r}")`.
        // R-DEV-2 (user-API ABI / exception parity): a duplicate name is a
        // deliberate `ValueError`, so ferrolearn rejects it at fit with the
        // closest analog, `FerroError::InvalidParameter`.
        let names: Vec<&str> = self
            .transformer_list
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        let mut seen = std::collections::HashSet::with_capacity(names.len());
        if !names.iter().all(|name| seen.insert(*name)) {
            return Err(FerroError::InvalidParameter {
                name: "transformer_list".into(),
                reason: format!("Names provided are not unique: {names:?}"),
            });
        }

        // Reject any name containing the reserved `__` separator, mirroring the
        // THIRD clause of `_validate_names`
        // (`sklearn/utils/metaestimators.py:91-95`): `invalid_names = [name for
        // name in names if "__" in name]; if invalid_names: raise
        // ValueError("Estimator names must not contain __: got {0!r}")`. `__` is
        // reserved for the nested-parameter addressing protocol
        // (`<step>__<param>`), so it is forbidden anywhere in a step name (a
        // single `_` is fine). sklearn runs this AFTER the uniqueness clause; we
        // match that order. R-DEV-2 (exception parity): a deliberate `ValueError`,
        // mapped to the closest analog `FerroError::InvalidParameter`. (The MIDDLE
        // clause — names colliding with constructor-arg params,
        // `metaestimators.py:84-90` — has no ferrolearn analog: `FeatureUnion`
        // exposes no `get_params` params, so it is intentionally not mirrored.)
        let invalid_names: Vec<&str> = names
            .iter()
            .copied()
            .filter(|name| name.contains("__"))
            .collect();
        if !invalid_names.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "transformer_list".into(),
                reason: format!("Estimator names must not contain __: got {invalid_names:?}"),
            });
        }

        // FeatureUnion sub-transformers are unsupervised; sklearn passes `y`
        // through but the transformers ignore it (`sklearn/pipeline.py:1681`).
        // The empty target is the no-supervision sentinel for `fit_pipeline`.
        let empty_y: Array1<F> = Array1::zeros(0);

        let mut fitted = Vec::with_capacity(self.transformer_list.len());
        let mut n_features_per = Vec::with_capacity(self.transformer_list.len());

        for (name, transformer) in &self.transformer_list {
            let fitted_t = transformer.fit_pipeline(x, &empty_y)?;
            // Probe the output width once at fit so feature-name prefixing and
            // the hstack column layout know each block's size.
            let out = fitted_t.transform_pipeline(x)?;
            n_features_per.push(out.ncols());
            fitted.push((name.clone(), fitted_t));
        }

        Ok(FittedFeatureUnion {
            fitted,
            n_features_per,
        })
    }
}

// ---------------------------------------------------------------------------
// FittedFeatureUnion
// ---------------------------------------------------------------------------

/// A fitted [`FeatureUnion`]: each named sub-transformer is fitted, and the
/// per-transformer output width is recorded for feature-name prefixing and the
/// horizontal-concatenation column layout.
///
/// Created by calling [`Fit::fit`] on a [`FeatureUnion`]. Implements
/// [`Transform<Array2<F>>`](Transform) producing the horizontally concatenated
/// `Array2<F>`.
pub struct FittedFeatureUnion<F: Float + Send + Sync + 'static = f64> {
    /// Fitted sub-transformers, in union order.
    fitted: Vec<(String, Box<dyn FittedPipelineTransformer<F>>)>,
    /// The output column count of each sub-transformer, in union order
    /// (recorded at fit). The total output width is the sum of these.
    n_features_per: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> FittedFeatureUnion<F> {
    /// Returns the names of all fitted sub-transformers, in union order.
    #[must_use]
    pub fn transformer_names(&self) -> Vec<&str> {
        self.fitted.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Number of fitted sub-transformers in the union.
    #[must_use]
    pub fn n_transformers(&self) -> usize {
        self.fitted.len()
    }

    /// Total output width: the sum of every sub-transformer's output column
    /// count. Equals the number of columns in [`Transform::transform`]'s output.
    #[must_use]
    pub fn n_features_out(&self) -> usize {
        self.n_features_per.iter().sum()
    }

    /// Output feature names, one per output column, in concatenation order.
    ///
    /// For each sub-transformer named `name` with output width `w`, this emits
    /// `"{name}__x0" .. "{name}__x{w-1}"`, then moves on to the next
    /// transformer's block. This mirrors `FeatureUnion.get_feature_names_out`
    /// with the default `verbose_feature_names_out=True`
    /// (`sklearn/pipeline.py:1567`/`:1608-1616`): sklearn prefixes each
    /// sub-transformer's own feature name with `"{name}__"`.
    ///
    /// ferrolearn's [`PipelineTransformer`] trait objects do not expose their own
    /// per-output feature names, so the positional default `x{j}` is used as the
    /// suffix — this is sklearn's `OneToOneFeatureMixin` positional default
    /// (`['x0','x1',...]`), which is exactly what `StandardScaler` /
    /// `MinMaxScaler` and other column-preserving transformers produce. So a union
    /// of two such transformers named `ss`/`mm` over 2-column input yields
    /// `['ss__x0','ss__x1','mm__x0','mm__x1']`, matching the live oracle.
    #[must_use]
    pub fn get_feature_names_out(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.n_features_out());
        for ((name, _), &width) in self.fitted.iter().zip(self.n_features_per.iter()) {
            for j in 0..width {
                names.push(format!("{name}__x{j}"));
            }
        }
        names
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedFeatureUnion<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform `x` through every fitted sub-transformer and horizontally
    /// concatenate the results.
    ///
    /// Mirrors `FeatureUnion.transform` (`sklearn/pipeline.py:1770`): each
    /// transformer transforms the same `x`, then `self._hstack(Xs)`
    /// (`np.hstack`, `:1812`/`:1820`) concatenates the outputs column-wise. The
    /// output has shape `(n_samples, sum_of_widths)` and the columns appear in
    /// transformer order: block 0 is the first transformer's full output, block 1
    /// the second's, and so on. An empty union transforms to a `(n_samples, 0)`
    /// matrix (the empty-`np.hstack` analog).
    ///
    /// # Errors
    ///
    /// Propagates any [`FerroError`] from an individual sub-transformer. Returns
    /// [`FerroError::ShapeMismatch`] if a sub-transformer's output does not have
    /// `n_samples == x.nrows()` rows (the hstack requires row-aligned blocks).
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_rows = x.nrows();

        // Transform `x` through each sub-transformer, collecting the blocks and
        // their widths. Validate each block is row-aligned before any copy.
        let mut blocks: Vec<Array2<F>> = Vec::with_capacity(self.fitted.len());
        let mut total_width = 0usize;
        for (name, transformer) in &self.fitted {
            let block = transformer.transform_pipeline(x)?;
            if block.nrows() != n_rows {
                return Err(FerroError::ShapeMismatch {
                    expected: vec![n_rows, block.ncols()],
                    actual: vec![block.nrows(), block.ncols()],
                    context: format!(
                        "FeatureUnion transformer `{name}` produced {} rows, expected {n_rows} \
                         (every sub-transformer output must be row-aligned for hstack)",
                        block.nrows()
                    ),
                });
            }
            total_width += block.ncols();
            blocks.push(block);
        }

        // Allocate the concatenated output and copy each block into its
        // contiguous column range, left-to-right (bounds-safe: `col_offset` and
        // each block width are derived from the blocks just collected).
        let mut out = Array2::<F>::zeros((n_rows, total_width));
        let mut col_offset = 0usize;
        for block in &blocks {
            let width = block.ncols();
            for r in 0..n_rows {
                for c in 0..width {
                    out[[r, col_offset + c]] = block[[r, c]];
                }
            }
            col_offset += width;
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Test fixtures -------------------------------------------------------

    /// A trivial transformer that doubles all values.
    struct DoublingTransformer;

    impl PipelineTransformer<f64> for DoublingTransformer {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
            Ok(Box::new(FittedDoublingTransformer))
        }
    }

    struct FittedDoublingTransformer;

    impl FittedPipelineTransformer<f64> for FittedDoublingTransformer {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(x.mapv(|v| v * 2.0))
        }
    }

    /// A trivial estimator that sums each row.
    struct SumEstimator;

    impl PipelineEstimator<f64> for SumEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            Ok(Box::new(FittedSumEstimator))
        }
    }

    struct FittedSumEstimator;

    impl FittedPipelineEstimator<f64> for FittedSumEstimator {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            let sums: Vec<f64> = x.rows().into_iter().map(|row| row.sum()).collect();
            Ok(Array1::from_vec(sums))
        }
    }

    // -- f32 test fixtures ---------------------------------------------------

    /// A trivial f32 transformer that doubles all values.
    struct DoublingTransformerF32;

    impl PipelineTransformer<f32> for DoublingTransformerF32 {
        fn fit_pipeline(
            &self,
            _x: &Array2<f32>,
            _y: &Array1<f32>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f32>>, FerroError> {
            Ok(Box::new(FittedDoublingTransformerF32))
        }
    }

    struct FittedDoublingTransformerF32;

    impl FittedPipelineTransformer<f32> for FittedDoublingTransformerF32 {
        fn transform_pipeline(&self, x: &Array2<f32>) -> Result<Array2<f32>, FerroError> {
            Ok(x.mapv(|v| v * 2.0))
        }
    }

    /// A trivial f32 estimator that sums each row.
    struct SumEstimatorF32;

    impl PipelineEstimator<f32> for SumEstimatorF32 {
        fn fit_pipeline(
            &self,
            _x: &Array2<f32>,
            _y: &Array1<f32>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f32>>, FerroError> {
            Ok(Box::new(FittedSumEstimatorF32))
        }
    }

    struct FittedSumEstimatorF32;

    impl FittedPipelineEstimator<f32> for FittedSumEstimatorF32 {
        fn predict_pipeline(&self, x: &Array2<f32>) -> Result<Array1<f32>, FerroError> {
            let sums: Vec<f32> = x.rows().into_iter().map(|row| row.sum()).collect();
            Ok(Array1::from_vec(sums))
        }
    }

    // -- Tests ---------------------------------------------------------------

    #[test]
    fn test_pipeline_fit_predict() {
        let pipeline = Pipeline::new()
            .transform_step("doubler", Box::new(DoublingTransformer))
            .estimator_step("sum", Box::new(SumEstimator));

        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // After doubling: [[2,4,6],[8,10,12]], sums: [12, 30]
        assert_eq!(preds.len(), 2);
        assert!((preds[0] - 12.0).abs() < 1e-10);
        assert!((preds[1] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_f32_fit_predict() {
        let pipeline = Pipeline::<f32>::new()
            .transform_step("doubler", Box::new(DoublingTransformerF32))
            .estimator_step("sum", Box::new(SumEstimatorF32));

        let x = Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0f32, 1.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 2);
        assert!((preds[0] - 12.0).abs() < 1e-5);
        assert!((preds[1] - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_pipeline_step_builder() {
        let pipeline = Pipeline::new()
            .step("doubler", as_transform_step(DoublingTransformer))
            .step("sum", as_estimator_step(SumEstimator));

        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert!((preds[0] - 12.0).abs() < 1e-10);
        assert!((preds[1] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_rejects_inconsistent_x_y() {
        // sklearn's Pipeline.fit validates X/y consistency before fitting any
        // step (check_consistent_length, validation.py:1320): a mismatched
        // n_samples raises ValueError. Live oracle:
        //   from sklearn.pipeline import Pipeline
        //   from sklearn.preprocessing import StandardScaler
        //   from sklearn.naive_bayes import GaussianNB; import numpy as np
        //   p = Pipeline([("s", StandardScaler()), ("c", GaussianNB())])
        //   try: p.fit(np.zeros((3,2)), np.zeros(4)); print("OK")
        //   except ValueError: print("RAISE")          # -> RAISE
        let pipeline = Pipeline::new()
            .transform_step("doubler", Box::new(DoublingTransformer))
            .estimator_step("sum", Box::new(SumEstimator));
        let x = Array2::<f64>::zeros((3, 2));
        let y = Array1::from_vec(vec![0.0, 1.0]); // len 2 != 3 rows
        let result = pipeline.fit(&x, &y);
        assert!(matches!(result, Err(FerroError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_pipeline_accepts_consistent_x_y() -> Result<(), FerroError> {
        // The guard must not reject well-formed X/y (live oracle: same Pipeline
        // with matching shapes -> OK).
        let pipeline = Pipeline::new()
            .transform_step("doubler", Box::new(DoublingTransformer))
            .estimator_step("sum", Box::new(SumEstimator));
        let x =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).map_err(|e| {
                FerroError::InvalidParameter {
                    name: "x".into(),
                    reason: e.to_string(),
                }
            })?;
        let y = Array1::from_vec(vec![0.0, 1.0]);
        let fitted = pipeline.fit(&x, &y)?;
        assert_eq!(fitted.predict(&x)?.len(), 2);
        Ok(())
    }

    #[test]
    fn test_pipeline_no_estimator_returns_error() {
        let pipeline = Pipeline::new().transform_step("doubler", Box::new(DoublingTransformer));

        let x = Array2::<f64>::zeros((2, 3));
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let result = pipeline.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_estimator_only() {
        let pipeline = Pipeline::new().estimator_step("sum", Box::new(SumEstimator));

        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // No transform, just sum: [6, 15]
        assert!((preds[0] - 6.0).abs() < 1e-10);
        assert!((preds[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_fitted_pipeline_step_names() {
        let pipeline = Pipeline::new()
            .transform_step("scaler", Box::new(DoublingTransformer))
            .transform_step("normalizer", Box::new(DoublingTransformer))
            .estimator_step("clf", Box::new(SumEstimator));

        let x = Array2::<f64>::zeros((2, 3));
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let names = fitted.step_names();
        assert_eq!(names, vec!["scaler", "normalizer", "clf"]);
    }

    #[test]
    fn test_multiple_transform_steps() {
        // Two doublers in sequence should quadruple values.
        let pipeline = Pipeline::new()
            .transform_step("double1", Box::new(DoublingTransformer))
            .transform_step("double2", Box::new(DoublingTransformer))
            .estimator_step("sum", Box::new(SumEstimator));

        let x = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // 1.0 * 2 * 2 = 4.0 per element, sum of 2 elements = 8.0
        assert!((preds[0] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_default() {
        let pipeline = Pipeline::<f64>::default();
        let x = Array2::<f64>::zeros((2, 3));
        let y = Array1::from_vec(vec![0.0, 1.0]);
        // Should error because no estimator.
        assert!(pipeline.fit(&x, &y).is_err());
    }

    #[test]
    fn test_pipeline_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        // Pipeline itself is Send+Sync because it only stores
        // Send+Sync trait objects.
        assert_send_sync::<Pipeline<f64>>();
        assert_send_sync::<Pipeline<f32>>();
        assert_send_sync::<FittedPipeline<f64>>();
        assert_send_sync::<FittedPipeline<f32>>();
    }

    // -- REQ-3: fit_transform / transform / predict_proba / decision_function /
    //    score ---------------------------------------------------------------

    /// An estimator that overrides the probability/decision/score delegations,
    /// proving the new default-Err trait methods can be overridden by a real
    /// final estimator (mirrors how `GaussianNB` does so in `gaussian.rs`).
    struct ProbaEstimator;

    impl PipelineEstimator<f64> for ProbaEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            Ok(Box::new(FittedProbaEstimator))
        }
    }

    struct FittedProbaEstimator;

    impl FittedPipelineEstimator<f64> for FittedProbaEstimator {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            // Predict 1.0 when the row sum is positive, else 0.0.
            Ok(Array1::from_iter(
                x.rows()
                    .into_iter()
                    .map(|r| if r.sum() > 0.0 { 1.0 } else { 0.0 }),
            ))
        }

        fn predict_proba_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            // A deterministic two-column "probability" (sigmoid of row sum).
            let mut out = Array2::<f64>::zeros((x.nrows(), 2));
            for (i, r) in x.rows().into_iter().enumerate() {
                let p1 = 1.0 / (1.0 + (-r.sum()).exp());
                out[[i, 0]] = 1.0 - p1;
                out[[i, 1]] = p1;
            }
            Ok(out)
        }

        fn score_pipeline(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64, FerroError> {
            let preds = self.predict_pipeline(x)?;
            let n = y.len();
            if n == 0 {
                return Ok(0.0);
            }
            let correct = preds
                .iter()
                .zip(y.iter())
                .filter(|(p, t)| (**p - **t).abs() < 1e-12)
                .count();
            Ok(correct as f64 / n as f64)
        }
    }

    #[test]
    fn test_pipeline_fit_transform_equals_transform() -> Result<(), FerroError> {
        // fit_transform must return exactly what FittedPipeline::transform
        // returns on the same input (fit-then-transform ≡ fused fit_transform).
        let pipeline = Pipeline::new()
            .transform_step("double1", Box::new(DoublingTransformer))
            .transform_step("double2", Box::new(DoublingTransformer))
            .estimator_step("sum", Box::new(SumEstimator));

        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let (fitted, xt) = pipeline.fit_transform(&x, &y)?;
        // Two doublers quadruple the data.
        let expected = x.mapv(|v| v * 4.0);
        assert_eq!(xt, expected);
        // transform() on the fitted pipeline matches fit_transform's output.
        let xt2 = fitted.transform(&x)?;
        assert_eq!(xt2, expected);
        Ok(())
    }

    #[test]
    fn test_pipeline_transform_applies_only_transformer_steps() -> Result<(), FerroError> {
        // FittedPipeline::transform returns the data the estimator would see —
        // i.e. only the transformer prefix is applied, not the estimator.
        let pipeline = Pipeline::new()
            .transform_step("doubler", Box::new(DoublingTransformer))
            .estimator_step("sum", Box::new(SumEstimator));
        let x = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = Array1::from_vec(vec![0.0, 1.0]);
        let fitted = pipeline.fit(&x, &y)?;
        let xt = fitted.transform(&x)?;
        assert_eq!(xt, x.mapv(|v| v * 2.0));
        Ok(())
    }

    #[test]
    fn test_pipeline_predict_proba_default_is_err() -> Result<(), FerroError> {
        // SumEstimator does not override predict_proba_pipeline → the default
        // Err (sklearn AttributeError analog) fires.
        let pipeline = Pipeline::new()
            .transform_step("doubler", Box::new(DoublingTransformer))
            .estimator_step("sum", Box::new(SumEstimator));
        let x = ndarray::array![[1.0, 1.0]];
        let y = Array1::from_vec(vec![0.0]);
        let fitted = pipeline.fit(&x, &y)?;
        assert!(matches!(
            fitted.predict_proba(&x),
            Err(FerroError::InvalidParameter { .. })
        ));
        assert!(matches!(
            fitted.decision_function(&x),
            Err(FerroError::InvalidParameter { .. })
        ));
        assert!(matches!(
            fitted.score(&x, &y),
            Err(FerroError::InvalidParameter { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_pipeline_predict_proba_and_score_override() -> Result<(), FerroError> {
        // ProbaEstimator overrides the delegations. The transformer doubles the
        // data; the proba estimator sees the doubled rows.
        let pipeline = Pipeline::new()
            .transform_step("doubler", Box::new(DoublingTransformer))
            .estimator_step("clf", Box::new(ProbaEstimator));
        let x = ndarray::array![[1.0], [-2.0]];
        let y = Array1::from_vec(vec![1.0, 0.0]);
        let fitted = pipeline.fit(&x, &y)?;

        // Doubled rows: [2.0], [-4.0]. p1 = sigmoid(row sum).
        let proba = fitted.predict_proba(&x)?;
        assert_eq!(proba.dim(), (2, 2));
        for i in 0..2 {
            assert!((proba.row(i).sum() - 1.0).abs() < 1e-12);
        }
        let p1_row0 = 1.0 / (1.0 + (-2.0f64).exp());
        assert!((proba[[0, 1]] - p1_row0).abs() < 1e-12);

        // Both rows predicted correctly → score 1.0.
        let s = fitted.score(&x, &y)?;
        assert!((s - 1.0).abs() < 1e-12);
        Ok(())
    }

    // -- REQ-4a: named_steps / get_step / get_step_by_name / into_slice -------

    fn is_transformer(r: &PipelineStepRef<'_, f64>) -> bool {
        matches!(r, PipelineStepRef::Transformer(_))
    }
    fn is_estimator(r: &PipelineStepRef<'_, f64>) -> bool {
        matches!(r, PipelineStepRef::Estimator(_))
    }

    #[test]
    fn test_pipeline_named_steps_match_sklearn() {
        // sklearn: Pipeline([('a',StandardScaler()),('b',MinMaxScaler()),
        //                    ('c',GaussianNB())]).named_steps keys order
        //   == ['a', 'b', 'c']  (live oracle, sklearn 1.5.2;
        //   `named_steps = Bunch(**dict(self.steps))`, pipeline.py:325).
        // Every step is reachable by its construction name, in order.
        let pipeline = Pipeline::new()
            .transform_step("a", Box::new(DoublingTransformer))
            .transform_step("b", Box::new(DoublingTransformer))
            .estimator_step("c", Box::new(SumEstimator));

        let named = pipeline.named_steps();
        let names: Vec<&str> = named.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
        // The two transformer steps are transformers; the final is the estimator.
        assert!(is_transformer(&named[0].1));
        assert!(is_transformer(&named[1].1));
        assert!(is_estimator(&named[2].1));
        // step_names() agrees with named_steps() key order.
        assert_eq!(pipeline.step_names(), names);
        // len() counts every step (3), matching sklearn len(pipe)==3.
        assert_eq!(pipeline.len(), 3);
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn test_pipeline_get_step_integer() {
        // sklearn: p[0] -> first step object, p[2] -> last (estimator);
        //   p[10] -> IndexError (live oracle). ferrolearn returns None OOB.
        let pipeline = Pipeline::new()
            .transform_step("a", Box::new(DoublingTransformer))
            .transform_step("b", Box::new(DoublingTransformer))
            .estimator_step("c", Box::new(SumEstimator));

        assert!(matches!(
            pipeline.get_step(0),
            Some(PipelineStepRef::Transformer(_))
        ));
        assert!(matches!(
            pipeline.get_step(1),
            Some(PipelineStepRef::Transformer(_))
        ));
        assert!(matches!(
            pipeline.get_step(2),
            Some(PipelineStepRef::Estimator(_))
        ));
        // Out of range -> None (sklearn raises IndexError).
        assert!(pipeline.get_step(3).is_none());
        assert!(pipeline.get_step(10).is_none());
    }

    #[test]
    fn test_pipeline_get_step_by_name() {
        // sklearn: p['b'] -> the 'b' step; p['nope'] -> KeyError (live oracle).
        let pipeline = Pipeline::new()
            .transform_step("a", Box::new(DoublingTransformer))
            .transform_step("b", Box::new(DoublingTransformer))
            .estimator_step("c", Box::new(SumEstimator));

        assert!(matches!(
            pipeline.get_step_by_name("b"),
            Some(PipelineStepRef::Transformer(_))
        ));
        assert!(matches!(
            pipeline.get_step_by_name("c"),
            Some(PipelineStepRef::Estimator(_))
        ));
        assert!(matches!(
            pipeline.named_step("a"),
            Some(PipelineStepRef::Transformer(_))
        ));
        // Unknown name -> None (sklearn raises KeyError).
        assert!(pipeline.get_step_by_name("nope").is_none());
        assert!(pipeline.named_step("nope").is_none());
    }

    #[test]
    fn test_pipeline_into_slice() -> Result<(), FerroError> {
        // sklearn: p[0:2].steps names == ['a','b'] (a sub-Pipeline of the
        //   contiguous range; pipeline.py:310). p[:1] == ['a']. p[:] == all.
        //   p[1:1] == [] (empty). (live oracle, sklearn 1.5.2.)
        let build = || {
            Pipeline::new()
                .transform_step("a", Box::new(DoublingTransformer))
                .transform_step("b", Box::new(DoublingTransformer))
                .estimator_step("c", Box::new(SumEstimator))
        };

        // [0, 2) -> first two transformer steps, no estimator.
        let sub = build().into_slice(0, 2);
        assert_eq!(sub.step_names(), vec!["a", "b"]);
        assert_eq!(sub.len(), 2);

        // [0, 1) -> just the first step.
        let sub = build().into_slice(0, 1);
        assert_eq!(sub.step_names(), vec!["a"]);

        // [0, 3) -> the whole pipeline (full range), estimator preserved.
        let sub = build().into_slice(0, 3);
        assert_eq!(sub.step_names(), vec!["a", "b", "c"]);

        // [2, 3) -> just the estimator step.
        let sub = build().into_slice(2, 3);
        assert_eq!(sub.step_names(), vec!["c"]);

        // Empty range -> empty pipeline.
        let sub = build().into_slice(1, 1);
        assert!(sub.step_names().is_empty());
        assert!(sub.is_empty());

        Ok(())
    }

    #[test]
    fn test_pipeline_into_slice_clamps_like_python() {
        // sklearn `Pipeline.__getitem__` slices `self.steps[ind]` (Python list
        // slice, `pipeline.py:310`): out-of-range bounds CLAMP, never raise
        // (#2235). Live oracle (sklearn 1.5.2, 2-step pipeline):
        //   p[0:5].steps -> ['a','c'] (clamp); p[2:1] -> []; p[5:100] -> [].
        let build = || {
            Pipeline::new()
                .transform_step("a", Box::new(DoublingTransformer))
                .estimator_step("c", Box::new(SumEstimator))
        };
        // end past len (2) -> clamp to all.
        assert_eq!(build().into_slice(0, 5).step_names(), vec!["a", "c"]);
        // start > end -> empty.
        assert!(build().into_slice(2, 1).is_empty());
        // start past len -> empty.
        assert!(build().into_slice(5, 100).is_empty());
    }

    #[test]
    fn test_pipeline_into_slice_transformer_only_still_fits_estimatorless() -> Result<(), FerroError>
    {
        // A slice dropping the estimator yields an estimator-less pipeline that
        // (like sklearn's transformer-only sub-pipeline) is valid to build but
        // errors at fit (matches REQ-2's no-estimator rejection).
        let pipeline = Pipeline::new()
            .transform_step("a", Box::new(DoublingTransformer))
            .estimator_step("c", Box::new(SumEstimator));
        let sub = pipeline.into_slice(0, 1);
        let x = Array2::<f64>::zeros((2, 2));
        let y = Array1::from_vec(vec![0.0, 1.0]);
        assert!(matches!(
            sub.fit(&x, &y),
            Err(FerroError::InvalidParameter { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_fitted_pipeline_named_steps_and_get_step() -> Result<(), FerroError> {
        // The accessors work on the FITTED pipeline too. Names match
        // construction order (sklearn named_steps on a fitted Pipeline).
        let pipeline = Pipeline::new()
            .transform_step("scaler", Box::new(DoublingTransformer))
            .transform_step("norm", Box::new(DoublingTransformer))
            .estimator_step("clf", Box::new(SumEstimator));
        let x = Array2::<f64>::zeros((2, 3));
        let y = Array1::from_vec(vec![0.0, 1.0]);
        let fitted = pipeline.fit(&x, &y)?;

        let names: Vec<&str> = fitted.named_steps().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["scaler", "norm", "clf"]);
        assert_eq!(fitted.len(), 3);
        assert!(!fitted.is_empty());

        // get_step by integer.
        assert!(matches!(
            fitted.get_step(0),
            Some(FittedPipelineStepRef::Transformer(_))
        ));
        assert!(matches!(
            fitted.get_step(2),
            Some(FittedPipelineStepRef::Estimator(_))
        ));
        assert!(fitted.get_step(3).is_none());

        // get_step_by_name / named_step.
        assert!(matches!(
            fitted.get_step_by_name("norm"),
            Some(FittedPipelineStepRef::Transformer(_))
        ));
        assert!(matches!(
            fitted.named_step("clf"),
            Some(FittedPipelineStepRef::Estimator(_))
        ));
        assert!(fitted.named_step("nope").is_none());
        Ok(())
    }

    // -- REQ-5a: passthrough steps -------------------------------------------

    #[test]
    fn test_passthrough_step_is_identity() -> Result<(), FerroError> {
        // Live oracle (sklearn 1.5.2):
        //   from sklearn.pipeline import Pipeline; import numpy as np
        //   X = np.array([[1.,2.],[3.,4.],[5.,6.]])
        //   p = Pipeline([('p','passthrough')]).fit(X)
        //   np.array_equal(p.transform(X), X)   -> True
        // A pipeline whose only transformer is a passthrough step leaves X
        // unchanged. ferrolearn needs a final estimator slot to fit, so we add a
        // SumEstimator after; transform() (the transformer prefix) must equal X.
        let pipeline = Pipeline::new()
            .passthrough_step("p")
            .estimator_step("sum", Box::new(SumEstimator));
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let fitted = pipeline.fit(&x, &y)?;
        // transform() applies only the (passthrough) transformer prefix -> X.
        assert_eq!(fitted.transform(&x)?, x);
        Ok(())
    }

    #[test]
    fn test_passthrough_before_transformer_is_noop() -> Result<(), FerroError> {
        // Live oracle (sklearn 1.5.2):
        //   Pipeline([('pass','passthrough'),('ss',StandardScaler())]).fit(X)
        //     .transform(X)
        //   == Pipeline([('ss',StandardScaler())]).fit(X).transform(X)   -> True
        // A passthrough BEFORE a real transformer is a no-op: the result equals
        // the transformer alone. ferrolearn analog: a passthrough before a
        // DoublingTransformer == the doubler alone.
        let with_pass = Pipeline::new()
            .passthrough_step("pass")
            .transform_step("dbl", Box::new(DoublingTransformer))
            .estimator_step("sum", Box::new(SumEstimator));
        let without_pass = Pipeline::new()
            .transform_step("dbl", Box::new(DoublingTransformer))
            .estimator_step("sum", Box::new(SumEstimator));
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let a = with_pass.fit(&x, &y)?.transform(&x)?;
        let b = without_pass.fit(&x, &y)?.transform(&x)?;
        assert_eq!(a, b);
        // And it equals the doubler applied to X.
        assert_eq!(a, x.mapv(|v| v * 2.0));
        Ok(())
    }

    #[test]
    fn test_passthrough_after_transformer_is_noop() -> Result<(), FerroError> {
        // Live oracle (sklearn 1.5.2):
        //   Pipeline([('ss',StandardScaler()),('pass','passthrough')]).transform(X)
        //   == Pipeline([('ss',StandardScaler())]).transform(X)   -> True
        // A passthrough AFTER a real transformer is a no-op. ferrolearn analog:
        // doubler then passthrough == doubler alone.
        let with_pass = Pipeline::new()
            .transform_step("dbl", Box::new(DoublingTransformer))
            .passthrough_step("pass")
            .estimator_step("sum", Box::new(SumEstimator));
        let without_pass = Pipeline::new()
            .transform_step("dbl", Box::new(DoublingTransformer))
            .estimator_step("sum", Box::new(SumEstimator));
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let a = with_pass.fit(&x, &y)?.transform(&x)?;
        let b = without_pass.fit(&x, &y)?.transform(&x)?;
        assert_eq!(a, b);
        assert_eq!(a, x.mapv(|v| v * 2.0));
        Ok(())
    }

    #[test]
    fn test_passthrough_step_appears_in_step_names() -> Result<(), FerroError> {
        // Live oracle (sklearn 1.5.2):
        //   p = Pipeline([('p','passthrough'),('ss',StandardScaler())]).fit(X)
        //   list(p.named_steps.keys())  -> ['p', 'ss']
        //   p['p']                      -> 'passthrough'   (still visible)
        // A passthrough step is a real, named step: it shows up in
        // step_names()/named_steps() in order, exactly like sklearn.
        let pipeline = Pipeline::new()
            .passthrough_step("p")
            .transform_step("dbl", Box::new(DoublingTransformer))
            .estimator_step("clf", Box::new(SumEstimator));

        assert_eq!(pipeline.step_names(), vec!["p", "dbl", "clf"]);
        let named: Vec<&str> = pipeline.named_steps().iter().map(|(n, _)| *n).collect();
        assert_eq!(named, vec!["p", "dbl", "clf"]);
        // The passthrough step is a transformer-kind step (reachable by name).
        assert!(matches!(
            pipeline.named_step("p"),
            Some(PipelineStepRef::Transformer(_))
        ));

        // And it survives onto the fitted pipeline's introspection.
        let x = Array2::<f64>::zeros((2, 2));
        let y = Array1::from_vec(vec![0.0, 1.0]);
        let fitted = pipeline.fit(&x, &y)?;
        assert_eq!(fitted.step_names(), vec!["p", "dbl", "clf"]);
        assert!(matches!(
            fitted.named_step("p"),
            Some(FittedPipelineStepRef::Transformer(_))
        ));
        Ok(())
    }

    #[test]
    fn test_passthrough_transformer_standalone_identity() -> Result<(), FerroError> {
        // A standalone PassthroughTransformer: fit_pipeline + transform_pipeline
        // is the identity (the building block the no-op step is made of). This is
        // the pointwise restatement of sklearn's 'passthrough' == identity
        // (Pipeline([('p','passthrough')]).transform(X) == X, live 1.5.2).
        let p = PassthroughTransformer::<f64>::new();
        let x = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = Array1::from_vec(vec![0.0, 1.0]);
        let fitted = p.fit_pipeline(&x, &y)?;
        assert_eq!(fitted.transform_pipeline(&x)?, x);
        // Default constructs the same no-op.
        let fitted2 = PassthroughTransformer::<f64>::default().fit_pipeline(&x, &y)?;
        assert_eq!(fitted2.transform_pipeline(&x)?, x);
        // The fitted half also has a public constructor/Default.
        assert_eq!(
            FittedPassthroughTransformer::<f64>::new().transform_pipeline(&x)?,
            x
        );
        Ok(())
    }

    #[test]
    fn test_passthrough_transformer_f32() -> Result<(), FerroError> {
        // f32 generic support: the identity no-op for f32 data.
        let pipeline = Pipeline::<f32>::new()
            .passthrough_step("p")
            .transform_step("dbl", Box::new(DoublingTransformerF32))
            .estimator_step("sum", Box::new(SumEstimatorF32));
        let x = ndarray::array![[1.0f32, 2.0], [3.0, 4.0]];
        let y = Array1::from_vec(vec![0.0f32, 1.0]);
        let fitted = pipeline.fit(&x, &y)?;
        // passthrough then doubler == doubler alone.
        assert_eq!(fitted.transform(&x)?, x.mapv(|v| v * 2.0));
        Ok(())
    }

    #[test]
    fn test_passthrough_transformer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PassthroughTransformer<f64>>();
        assert_send_sync::<PassthroughTransformer<f32>>();
        assert_send_sync::<FittedPassthroughTransformer<f64>>();
        assert_send_sync::<FittedPassthroughTransformer<f32>>();
    }

    // -- REQ-8: FeatureUnion -------------------------------------------------

    /// A transformer that returns its input columns unchanged (width-preserving,
    /// the OneToOneFeatureMixin shape — like sklearn's StandardScaler).
    struct IdentityTransformer;

    impl PipelineTransformer<f64> for IdentityTransformer {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
            Ok(Box::new(FittedIdentityTransformer))
        }
    }

    struct FittedIdentityTransformer;

    impl FittedPipelineTransformer<f64> for FittedIdentityTransformer {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(x.clone())
        }
    }

    /// A transformer that emits a single column: the row sum (width 1, regardless
    /// of input width). Used to exercise mixed-width hstack blocks.
    struct RowSumTransformer;

    impl PipelineTransformer<f64> for RowSumTransformer {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
            Ok(Box::new(FittedRowSumTransformer))
        }
    }

    struct FittedRowSumTransformer;

    impl FittedPipelineTransformer<f64> for FittedRowSumTransformer {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            let sums: Vec<f64> = x.rows().into_iter().map(|r| r.sum()).collect();
            Array2::from_shape_vec((x.nrows(), 1), sums).map_err(|e| FerroError::InvalidParameter {
                name: "x".into(),
                reason: e.to_string(),
            })
        }
    }

    #[test]
    fn test_feature_union_hstack_layout() -> Result<(), FerroError> {
        // sklearn (live, 1.5.2):
        //   from sklearn.pipeline import FeatureUnion
        //   from sklearn.preprocessing import StandardScaler, MinMaxScaler
        //   import numpy as np
        //   X = np.array([[1.,2.],[3.,4.],[5.,6.]])
        //   fu = FeatureUnion([('ss',StandardScaler()),('mm',MinMaxScaler())]).fit(X)
        //   fu.transform(X).shape        -> (3, 4)
        //   # columns = [ss_col0, ss_col1, mm_col0, mm_col1]  (each transformer's
        //   #   full output, concatenated left-to-right in transformer_list order)
        // The hstack STRUCTURE is what's asserted here: two width-2 identity
        // transformers → a width-4 output whose column blocks are each
        // transformer's full output (here, the unchanged input twice). The block
        // layout (transformer 0's cols, then transformer 1's cols) IS sklearn's
        // _hstack ordering (pipeline.py:1812 np.hstack(Xs)).
        let union = FeatureUnion::<f64>::new()
            .with_transformer("a", Box::new(IdentityTransformer))
            .with_transformer("b", Box::new(IdentityTransformer));
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let fitted = union.fit(&x, &())?;
        let out = fitted.transform(&x)?;

        // Width = sum of widths = 2 + 2 = 4; rows preserved.
        assert_eq!(out.dim(), (3, 4));
        // Block 0 (cols 0..2) = transformer "a"'s output (== x).
        assert_eq!(out.slice(ndarray::s![.., 0..2]).to_owned(), x);
        // Block 1 (cols 2..4) = transformer "b"'s output (== x).
        assert_eq!(out.slice(ndarray::s![.., 2..4]).to_owned(), x);
        Ok(())
    }

    #[test]
    fn test_feature_union_get_feature_names_out() -> Result<(), FerroError> {
        // sklearn (live, 1.5.2): the SAME union as above ->
        //   list(fu.get_feature_names_out())
        //     == ['ss__x0','ss__x1','mm__x0','mm__x1']
        // i.e. each transformer's positional output names ('x0','x1' — the
        // OneToOneFeatureMixin default for StandardScaler/MinMaxScaler) prefixed
        // by '{name}__' (verbose_feature_names_out=True default, pipeline.py:1608).
        // ferrolearn's identity transformers are the width-preserving analog, so
        // the NAMING semantics (prefix + positional x{j}) match exactly.
        let union = FeatureUnion::<f64>::new()
            .with_transformer("ss", Box::new(IdentityTransformer))
            .with_transformer("mm", Box::new(IdentityTransformer));
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let fitted = union.fit(&x, &())?;
        assert_eq!(
            fitted.get_feature_names_out(),
            vec!["ss__x0", "ss__x1", "mm__x0", "mm__x1"]
        );
        // transformer_names() preserves union order; n_transformers/n_features_out.
        assert_eq!(fitted.transformer_names(), vec!["ss", "mm"]);
        assert_eq!(fitted.n_transformers(), 2);
        assert_eq!(fitted.n_features_out(), 4);
        Ok(())
    }

    #[test]
    fn test_feature_union_single_transformer_width() -> Result<(), FerroError> {
        // sklearn (live, 1.5.2):
        //   FeatureUnion([('ss',StandardScaler())]).fit(X).transform(X).shape
        //     -> (3, 2)   (single block == that transformer's width)
        //   get_feature_names_out() -> ['ss__x0','ss__x1']
        let union =
            FeatureUnion::<f64>::new().with_transformer("ss", Box::new(IdentityTransformer));
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let fitted = union.fit(&x, &())?;
        let out = fitted.transform(&x)?;
        assert_eq!(out.dim(), (3, 2));
        assert_eq!(out, x);
        assert_eq!(fitted.get_feature_names_out(), vec!["ss__x0", "ss__x1"]);
        Ok(())
    }

    #[test]
    fn test_feature_union_mixed_widths() -> Result<(), FerroError> {
        // sklearn (live, 1.5.2) — a union whose transformers emit DIFFERENT
        // widths concatenates their blocks correctly. Oracle (StandardScaler
        // keeps 3 cols, PCA(1) emits 1):
        //   X = np.array([[1.,2.,3.],[3.,4.,5.],[5.,6.,7.]])
        //   fu = FeatureUnion([('ss',StandardScaler()),('pca',PCA(1))]).fit(X)
        //   fu.transform(X).shape -> (3, 4)   (3 + 1)
        //   list(fu.get_feature_names_out())
        //     -> ['ss__x0','ss__x1','ss__x2','pca__pca0']
        // ferrolearn analog: a width-3 identity + a width-1 row-sum transformer.
        // The STRUCTURE (block 0 width 3, block 1 width 1; total 4) is sklearn's.
        // (Names: ferrolearn uses the positional x{j} suffix for both blocks —
        // the documented OneToOneFeatureMixin default, since the trait objects
        // expose no per-output names.)
        let union = FeatureUnion::<f64>::new()
            .with_transformer("ident", Box::new(IdentityTransformer))
            .with_transformer("rowsum", Box::new(RowSumTransformer));
        let x = ndarray::array![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]];

        let fitted = union.fit(&x, &())?;
        let out = fitted.transform(&x)?;
        // 3 (identity) + 1 (row sum) = 4 columns.
        assert_eq!(out.dim(), (3, 4));
        // Block 0 == x (identity).
        assert_eq!(out.slice(ndarray::s![.., 0..3]).to_owned(), x);
        // Block 1 == row sums.
        let expected_sums = ndarray::array![[6.0], [12.0], [18.0]];
        assert_eq!(out.slice(ndarray::s![.., 3..4]).to_owned(), expected_sums);
        // Feature names reflect the per-block widths.
        assert_eq!(
            fitted.get_feature_names_out(),
            vec!["ident__x0", "ident__x1", "ident__x2", "rowsum__x0"]
        );
        Ok(())
    }

    #[test]
    fn test_feature_union_empty() -> Result<(), FerroError> {
        // An empty union fits OK and transforms to a (n_samples, 0) matrix — the
        // ferrolearn analog of sklearn's empty-hstack branch
        //   `if not Xs: return np.zeros((X.shape[0], 0))` (pipeline.py:1808).
        // (sklearn's PUBLIC FeatureUnion([]).fit raises at _validate_transformers'
        // `zip(*[])`, a Python-tuple-unpack artifact, not a numerical contract —
        // R-DEV-4: ferrolearn has no such unpack, and the empty-hstack shape is
        // the documented (n, 0) result.)
        let union = FeatureUnion::<f64>::new();
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = union.fit(&x, &())?;
        let out = fitted.transform(&x)?;
        assert_eq!(out.dim(), (2, 0));
        assert!(fitted.get_feature_names_out().is_empty());
        assert_eq!(fitted.n_features_out(), 0);
        Ok(())
    }

    #[test]
    fn test_feature_union_row_count_consistency() -> Result<(), FerroError> {
        // Every sub-output has n_rows == X.nrows(); the hstacked result preserves
        // the row count (live oracle: FeatureUnion outputs have X.shape[0] rows).
        let union = FeatureUnion::<f64>::new()
            .with_transformer("a", Box::new(IdentityTransformer))
            .with_transformer("b", Box::new(RowSumTransformer));
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitted = union.fit(&x, &())?;
        let out = fitted.transform(&x)?;
        assert_eq!(out.nrows(), x.nrows());
        Ok(())
    }

    #[test]
    fn test_feature_union_f32() -> Result<(), FerroError> {
        // f32 generic support: same hstack layout for f32 data.
        let union = FeatureUnion::<f32>::new()
            .with_transformer("a", Box::new(IdentityTransformerF32))
            .with_transformer("b", Box::new(IdentityTransformerF32));
        let x = ndarray::array![[1.0f32, 2.0], [3.0, 4.0]];
        let fitted = union.fit(&x, &())?;
        let out = fitted.transform(&x)?;
        assert_eq!(out.dim(), (2, 4));
        assert_eq!(out.slice(ndarray::s![.., 0..2]).to_owned(), x);
        assert_eq!(out.slice(ndarray::s![.., 2..4]).to_owned(), x);
        Ok(())
    }

    /// f32 identity transformer (width-preserving) for the f32 union test.
    struct IdentityTransformerF32;

    impl PipelineTransformer<f32> for IdentityTransformerF32 {
        fn fit_pipeline(
            &self,
            _x: &Array2<f32>,
            _y: &Array1<f32>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f32>>, FerroError> {
            Ok(Box::new(FittedIdentityTransformerF32))
        }
    }

    struct FittedIdentityTransformerF32;

    impl FittedPipelineTransformer<f32> for FittedIdentityTransformerF32 {
        fn transform_pipeline(&self, x: &Array2<f32>) -> Result<Array2<f32>, FerroError> {
            Ok(x.clone())
        }
    }

    #[test]
    fn test_feature_union_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FeatureUnion<f64>>();
        assert_send_sync::<FeatureUnion<f32>>();
        assert_send_sync::<FittedFeatureUnion<f64>>();
        assert_send_sync::<FittedFeatureUnion<f32>>();
    }
}
