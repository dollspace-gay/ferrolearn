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
//! | REQ-4 (named_steps/get_params/set_params/__getitem__) | NOT-STARTED | blocker #362. Only `step_names()` exists. |
//! | REQ-5 (passthrough steps + memory caching) | NOT-STARTED | blocker #363. |
//! | REQ-6 (fit_params / metadata routing) | NOT-STARTED | blocker #364. |
//! | REQ-7 (make_pipeline auto-naming helper) | NOT-STARTED | blocker #365 (`pipeline.py:1220`). |
//! | REQ-8 (FeatureUnion) | NOT-STARTED | blocker #366 (`pipeline.py:1329`). |
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

use crate::error::FerroError;
use crate::traits::{Fit, Predict};

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
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if no estimator step was set.
    /// Propagates any errors from individual step fitting or transforming.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedPipeline<F>, FerroError> {
        if self.estimator.is_none() {
            return Err(FerroError::InvalidParameter {
                name: "estimator".into(),
                reason: "pipeline must have a final estimator step".into(),
            });
        }

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
}
