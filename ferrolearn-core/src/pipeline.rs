//! Dynamic-dispatch pipeline for composing transformers and estimators.
//!
//! A [`Pipeline`] chains zero or more transformer steps followed by a final
//! estimator step. Calling [`Fit::fit`] on a pipeline fits each step in
//! sequence, producing a [`FittedPipeline`] that implements [`Predict`].
//!
//! The dynamic pipeline constrains all intermediate data to
//! [`ndarray::Array2<f64>`] so that heterogeneous steps can be composed
//! via trait objects. A compile-time typed pipeline (zero-cost, generic)
//! is planned for Phase 3.
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
//! impl PipelineTransformer for IdentityTransformer {
//!     fn fit_pipeline(
//!         &self,
//!         x: &Array2<f64>,
//!         _y: &Array1<f64>,
//!     ) -> Result<Box<dyn FittedPipelineTransformer>, FerroError> {
//!         Ok(Box::new(FittedIdentity))
//!     }
//! }
//!
//! struct FittedIdentity;
//!
//! impl FittedPipelineTransformer for FittedIdentity {
//!     fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
//!         Ok(x.clone())
//!     }
//! }
//!
//! // A trivial estimator that predicts the first column.
//! struct FirstColumnEstimator;
//!
//! impl PipelineEstimator for FirstColumnEstimator {
//!     fn fit_pipeline(
//!         &self,
//!         _x: &Array2<f64>,
//!         _y: &Array1<f64>,
//!     ) -> Result<Box<dyn FittedPipelineEstimator>, FerroError> {
//!         Ok(Box::new(FittedFirstColumn))
//!     }
//! }
//!
//! struct FittedFirstColumn;
//!
//! impl FittedPipelineEstimator for FittedFirstColumn {
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

use crate::error::FerroError;
use crate::traits::{Fit, Predict};

// ---------------------------------------------------------------------------
// Trait-object interfaces for pipeline steps
// ---------------------------------------------------------------------------

/// An unfitted transformer step that can participate in a [`Pipeline`].
///
/// Implementors must be able to fit themselves on `Array2<f64>` data and
/// return a boxed [`FittedPipelineTransformer`].
pub trait PipelineTransformer: Send + Sync {
    /// Fit this transformer on the given data.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if fitting fails.
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer>, FerroError>;
}

/// A fitted transformer step in a [`FittedPipeline`].
///
/// Transforms `Array2<f64>` data, producing a new `Array2<f64>`.
pub trait FittedPipelineTransformer: Send + Sync {
    /// Transform the input data.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if the input shape is incompatible.
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError>;
}

/// An unfitted estimator step that serves as the final step in a [`Pipeline`].
///
/// Implementors must be able to fit themselves on `Array2<f64>` data and
/// return a boxed [`FittedPipelineEstimator`].
pub trait PipelineEstimator: Send + Sync {
    /// Fit this estimator on the given data.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if fitting fails.
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator>, FerroError>;
}

/// A fitted estimator step in a [`FittedPipeline`].
///
/// Produces `Array1<f64>` predictions from `Array2<f64>` input.
pub trait FittedPipelineEstimator: Send + Sync {
    /// Generate predictions for the input data.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if the input shape is incompatible.
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError>;
}

// ---------------------------------------------------------------------------
// Pipeline (unfitted)
// ---------------------------------------------------------------------------

/// A named transformer step in an unfitted pipeline.
struct TransformStep {
    /// Human-readable name for this step.
    name: String,
    /// The unfitted transformer.
    step: Box<dyn PipelineTransformer>,
}

/// A dynamic-dispatch pipeline that composes transformers and a final estimator.
///
/// Steps are added with [`transform_step`](Pipeline::transform_step) and the
/// final estimator is set with [`estimator_step`](Pipeline::estimator_step).
/// The pipeline implements [`Fit<Array2<f64>, Array1<f64>>`](Fit) and produces
/// a [`FittedPipeline`] that implements [`Predict<Array2<f64>>`](Predict).
///
/// All intermediate data flows as `Array2<f64>`.
pub struct Pipeline {
    /// Ordered transformer steps.
    transforms: Vec<TransformStep>,
    /// The final estimator step (name + estimator).
    estimator: Option<(String, Box<dyn PipelineEstimator>)>,
}

impl Pipeline {
    /// Create a new empty pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrolearn_core::pipeline::Pipeline;
    /// let pipeline = Pipeline::new();
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
    pub fn transform_step(mut self, name: &str, step: Box<dyn PipelineTransformer>) -> Self {
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
    pub fn estimator_step(mut self, name: &str, estimator: Box<dyn PipelineEstimator>) -> Self {
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
    pub fn step(self, name: &str, step: Box<dyn PipelineStep>) -> Self {
        step.add_to_pipeline(self, name)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<f64>, Array1<f64>> for Pipeline {
    type Fitted = FittedPipeline;
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
    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<FittedPipeline, FerroError> {
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
struct FittedTransformStep {
    /// Human-readable name for this step.
    name: String,
    /// The fitted transformer.
    step: Box<dyn FittedPipelineTransformer>,
}

/// A fitted pipeline that chains fitted transformers and a fitted estimator.
///
/// Created by calling [`Fit::fit`] on a [`Pipeline`]. Implements
/// [`Predict<Array2<f64>>`](Predict), producing `Array1<f64>` predictions.
pub struct FittedPipeline {
    /// Fitted transformer steps, in order.
    transforms: Vec<FittedTransformStep>,
    /// The fitted estimator (name + estimator).
    estimator: (String, Box<dyn FittedPipelineEstimator>),
}

impl FittedPipeline {
    /// Returns the names of all steps (transformers + estimator) in order.
    pub fn step_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.transforms.iter().map(|s| s.name.as_str()).collect();
        names.push(&self.estimator.0);
        names
    }
}

impl Predict<Array2<f64>> for FittedPipeline {
    type Output = Array1<f64>;
    type Error = FerroError;

    /// Generate predictions by transforming the input through each fitted
    /// transformer step, then calling predict on the fitted estimator.
    ///
    /// # Errors
    ///
    /// Propagates any errors from transformer or estimator steps.
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        let mut current_x = x.clone();

        for ts in &self.transforms {
            current_x = ts.step.transform_pipeline(&current_x)?;
        }

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
pub trait PipelineStep: Send + Sync {
    /// Add this step to the pipeline under the given name.
    ///
    /// Transformer steps are added as intermediate transform steps.
    /// Estimator steps are set as the final estimator.
    fn add_to_pipeline(self: Box<Self>, pipeline: Pipeline, name: &str) -> Pipeline;
}

/// Wraps a [`PipelineTransformer`] to implement [`PipelineStep`].
///
/// Created by [`as_transform_step`].
pub struct TransformerStepWrapper(Box<dyn PipelineTransformer>);

impl PipelineStep for TransformerStepWrapper {
    fn add_to_pipeline(self: Box<Self>, pipeline: Pipeline, name: &str) -> Pipeline {
        pipeline.transform_step(name, self.0)
    }
}

/// Wraps a [`PipelineEstimator`] to implement [`PipelineStep`].
///
/// Created by [`as_estimator_step`].
pub struct EstimatorStepWrapper(Box<dyn PipelineEstimator>);

impl PipelineStep for EstimatorStepWrapper {
    fn add_to_pipeline(self: Box<Self>, pipeline: Pipeline, name: &str) -> Pipeline {
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
/// // Assuming `my_scaler` implements PipelineTransformer:
/// // let pipeline = Pipeline::new().step("scaler", as_transform_step(my_scaler));
/// ```
pub fn as_transform_step(t: impl PipelineTransformer + 'static) -> Box<dyn PipelineStep> {
    Box::new(TransformerStepWrapper(Box::new(t)))
}

/// Wrap a [`PipelineEstimator`] as a [`PipelineStep`] for use with
/// [`Pipeline::step`].
///
/// # Examples
///
/// ```
/// use ferrolearn_core::pipeline::{Pipeline, as_estimator_step};
/// // Assuming `my_model` implements PipelineEstimator:
/// // let pipeline = Pipeline::new().step("model", as_estimator_step(my_model));
/// ```
pub fn as_estimator_step(e: impl PipelineEstimator + 'static) -> Box<dyn PipelineStep> {
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

    impl PipelineTransformer for DoublingTransformer {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer>, FerroError> {
            Ok(Box::new(FittedDoublingTransformer))
        }
    }

    struct FittedDoublingTransformer;

    impl FittedPipelineTransformer for FittedDoublingTransformer {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(x.mapv(|v| v * 2.0))
        }
    }

    /// A trivial estimator that sums each row.
    struct SumEstimator;

    impl PipelineEstimator for SumEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator>, FerroError> {
            Ok(Box::new(FittedSumEstimator))
        }
    }

    struct FittedSumEstimator;

    impl FittedPipelineEstimator for FittedSumEstimator {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            let sums: Vec<f64> = x.rows().into_iter().map(|row| row.sum()).collect();
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
        let pipeline = Pipeline::default();
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
        assert_send_sync::<Pipeline>();
        assert_send_sync::<FittedPipeline>();
    }
}
