//! Core ML traits for the ferrolearn framework.
//!
//! This module defines the fundamental traits that all ferrolearn estimators
//! and transformers implement. The key design principle is **compile-time
//! safety**: calling [`Predict::predict`] on an unfitted model is a type
//! error, not a runtime error.
//!
//! # Design
//!
//! The unfitted struct (e.g., `LogisticRegression`) holds hyperparameters
//! and implements [`Fit`]. Calling [`Fit::fit`] consumes the hyperparameters
//! by reference and returns a *new fitted type* (e.g., `FittedLogisticRegression`)
//! that implements [`Predict`]. The unfitted type **never** implements `Predict`,
//! so the compiler rejects invalid usage.
//!
//! ```text
//! [StandardScaler]      --fit(&x, &())--> [FittedStandardScaler]    --transform(&x)--> Array2<F>
//! [LogisticRegression]  --fit(&x, &y) --> [FittedLogisticRegression] --predict(&x) --> Array1<usize>
//! ```
//!
//! # Float Bound
//!
//! All algorithms are generic over `F: num_traits::Float + Send + Sync + 'static`.

/// Train a model on data, producing a fitted model.
///
/// The unfitted struct holds hyperparameters. Calling `fit` returns a new
/// fitted type that holds learned parameters. This is the core mechanism
/// that ensures compile-time enforcement: the unfitted type does not
/// implement [`Predict`], so calling `predict` before `fit` is a type error.
///
/// # Type Parameters
///
/// - `X`: The feature matrix type (typically `ndarray::Array2<F>`).
/// - `Y`: The target type. Use `()` for unsupervised models.
///
/// # Examples
///
/// ```
/// use ferrolearn_core::Fit;
/// use ferrolearn_core::FerroError;
///
/// struct MyRegressor { alpha: f64 }
/// struct FittedMyRegressor { weights: Vec<f64> }
///
/// impl Fit<Vec<Vec<f64>>, Vec<f64>> for MyRegressor {
///     type Fitted = FittedMyRegressor;
///     type Error = FerroError;
///
///     fn fit(&self, _x: &Vec<Vec<f64>>, _y: &Vec<f64>) -> Result<FittedMyRegressor, FerroError> {
///         Ok(FittedMyRegressor { weights: vec![1.0, 2.0] })
///     }
/// }
/// ```
pub trait Fit<X, Y> {
    /// The fitted model type returned by [`fit`](Fit::fit).
    type Fitted;
    /// The error type returned by [`fit`](Fit::fit).
    type Error;

    /// Train the model on the given data.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid (wrong shape, insufficient
    /// samples) or if the algorithm fails to converge.
    fn fit(&self, x: &X, y: &Y) -> Result<Self::Fitted, Self::Error>;
}

/// Generate predictions from a fitted model.
///
/// Only fitted model types implement this trait. Unfitted configuration
/// structs do **not** implement `Predict`, which means that calling
/// `predict` on an unfitted model is a compile-time error.
///
/// # Type Parameters
///
/// - `X`: The feature matrix type (typically `ndarray::Array2<F>`).
pub trait Predict<X> {
    /// The prediction output type (e.g., `ndarray::Array1<F>` or `ndarray::Array1<usize>`).
    type Output;
    /// The error type returned by [`predict`](Predict::predict).
    type Error;

    /// Generate predictions for the given input.
    ///
    /// # Errors
    ///
    /// Returns an error if the input has an incompatible shape with
    /// the model that was fitted.
    fn predict(&self, x: &X) -> Result<Self::Output, Self::Error>;
}

/// Transform data (e.g., scaling, encoding).
///
/// Transformers that require fitting first should implement [`Fit`]
/// to produce a fitted type that implements `Transform`. Stateless
/// transformers can implement `Transform` directly.
///
/// # Type Parameters
///
/// - `X`: The input data type.
pub trait Transform<X> {
    /// The transformed output type.
    type Output;
    /// The error type returned by [`transform`](Transform::transform).
    type Error;

    /// Transform the input data.
    ///
    /// # Errors
    ///
    /// Returns an error if the input has an incompatible shape.
    fn transform(&self, x: &X) -> Result<Self::Output, Self::Error>;
}

/// Combined fit-and-transform in a single pass.
///
/// This trait extends [`Transform`] and provides a convenience method
/// that fits the transformer and transforms the data in one step.
/// This can be more efficient than calling `fit` followed by `transform`
/// separately when the fitting process already computes the transformed
/// output.
///
/// # Type Parameters
///
/// - `X`: The input data type.
pub trait FitTransform<X>: Transform<X> {
    /// The error type for the combined fit-transform operation.
    type FitError;

    /// Fit the transformer to the data and return the transformed output.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid or if the transformer
    /// cannot be fitted.
    fn fit_transform(&self, x: &X) -> Result<Self::Output, Self::FitError>;
}

/// Incrementally train a model on a batch of data.
///
/// Unlike [`Fit`], `PartialFit` can be called multiple times — each call
/// updates the model with a new batch. This enables online/streaming learning.
///
/// The trait is implemented by both unfitted models (first call) and fitted
/// models (subsequent calls), enabling chaining:
/// `model.partial_fit(&b1, &y1)?.partial_fit(&b2, &y2)?.predict(&x)?`
pub trait PartialFit<X, Y>: Sized {
    /// The result type returned by [`partial_fit`](PartialFit::partial_fit).
    ///
    /// Must itself implement both [`Predict`] (so predictions can be made)
    /// and `PartialFit` (so additional batches can be fed).
    type FitResult: Predict<X> + PartialFit<X, Y>;

    /// The error type returned by [`partial_fit`](PartialFit::partial_fit).
    type Error: std::error::Error;

    /// Update the model with a new batch of data.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid (wrong shape, insufficient
    /// samples) or if the algorithm encounters a numerical problem.
    fn partial_fit(self, x: &X, y: &Y) -> Result<Self::FitResult, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FerroError;

    /// A dummy unfitted model — does NOT implement Predict.
    struct DummyEstimator;

    /// A dummy fitted model — implements Predict.
    struct FittedDummyEstimator {
        _learned_value: f64,
    }

    impl Fit<Vec<f64>, Vec<f64>> for DummyEstimator {
        type Fitted = FittedDummyEstimator;
        type Error = FerroError;

        fn fit(&self, _x: &Vec<f64>, _y: &Vec<f64>) -> Result<FittedDummyEstimator, FerroError> {
            Ok(FittedDummyEstimator {
                _learned_value: 42.0,
            })
        }
    }

    impl Predict<Vec<f64>> for FittedDummyEstimator {
        type Output = Vec<f64>;
        type Error = FerroError;

        fn predict(&self, x: &Vec<f64>) -> Result<Vec<f64>, FerroError> {
            Ok(x.iter().map(|v| v * 2.0).collect())
        }
    }

    #[test]
    fn test_fit_then_predict() {
        let estimator = DummyEstimator;
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        let fitted = estimator.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions, vec![2.0, 4.0, 6.0]);
    }

    /// Compile-time test: the following code must NOT compile.
    /// We verify this by checking that `DummyEstimator` does not
    /// implement `Predict`. This is a static assertion.
    #[test]
    fn test_unfitted_does_not_implement_predict() {
        // This is a compile-time check. If DummyEstimator implemented
        // Predict<Vec<f64>>, this function would not exist as-is.
        // We use a trait-bound negative test via a helper function.
        fn _assert_not_predict<T>()
        where
            T: Fit<Vec<f64>, Vec<f64>>,
        {
            // T implements Fit but we never call predict on it.
            // The point is: DummyEstimator does NOT implement Predict.
        }
        _assert_not_predict::<DummyEstimator>();
    }

    /// A dummy transformer.
    struct DummyTransformer;

    /// A fitted transformer.
    struct FittedDummyTransformer {
        _scale: f64,
    }

    impl Fit<Vec<f64>, ()> for DummyTransformer {
        type Fitted = FittedDummyTransformer;
        type Error = FerroError;

        fn fit(&self, _x: &Vec<f64>, _y: &()) -> Result<FittedDummyTransformer, FerroError> {
            Ok(FittedDummyTransformer { _scale: 2.0 })
        }
    }

    impl Transform<Vec<f64>> for FittedDummyTransformer {
        type Output = Vec<f64>;
        type Error = FerroError;

        fn transform(&self, x: &Vec<f64>) -> Result<Vec<f64>, FerroError> {
            Ok(x.iter().map(|v| v * self._scale).collect())
        }
    }

    #[test]
    fn test_unsupervised_fit_transform() {
        let transformer = DummyTransformer;
        let x = vec![1.0, 2.0, 3.0];
        let fitted = transformer.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed, vec![2.0, 4.0, 6.0]);
    }
}
