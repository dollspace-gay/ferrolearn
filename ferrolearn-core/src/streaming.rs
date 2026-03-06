//! Streaming data adapter for incremental learning.
//!
//! The [`StreamingFitter`] feeds batches from an iterator to a
//! [`PartialFit`](crate::PartialFit) model, enabling online/streaming
//! learning workflows where the full dataset does not fit in memory.
//!
//! # Example
//!
//! ```ignore
//! use ferrolearn_core::streaming::StreamingFitter;
//!
//! // Assume `model` implements PartialFit<Array2<f64>, Array1<f64>>
//! let batches = vec![
//!     (x_batch1, y_batch1),
//!     (x_batch2, y_batch2),
//! ];
//!
//! let fitter = StreamingFitter::new(model).n_epochs(3);
//! let fitted = fitter.fit_batches(batches)?;
//! let predictions = fitted.predict(&x_test)?;
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::traits::PartialFit;

/// Feeds batches from an iterator to a [`PartialFit`] model.
///
/// This adapter collects batches from an iterator and feeds them to a model
/// that implements [`PartialFit`]. Multiple epochs can be specified, causing
/// the batches to be replayed multiple times for convergence.
///
/// # Type Parameters
///
/// - `M`: The model type, which must implement [`PartialFit<X, Y>`].
pub struct StreamingFitter<M> {
    /// The initial (unfitted or partially fitted) model.
    model: M,
    /// Number of passes over the batch iterator.
    n_epochs: usize,
}

impl<M> StreamingFitter<M> {
    /// Create a new `StreamingFitter` wrapping the given model.
    ///
    /// The default number of epochs is 1 (a single pass over the data).
    pub fn new(model: M) -> Self {
        Self { model, n_epochs: 1 }
    }

    /// Set the number of epochs (passes over the data).
    ///
    /// Each epoch replays all batches in order. More epochs can improve
    /// convergence for online learning algorithms.
    ///
    /// # Panics
    ///
    /// This method does not panic, but [`fit_batches`](StreamingFitter::fit_batches)
    /// will return early with the initial model state if `n_epochs` is 0.
    #[must_use]
    pub fn n_epochs(mut self, n_epochs: usize) -> Self {
        self.n_epochs = n_epochs;
        self
    }

    /// Feed all batches to the model, returning the fitted result.
    ///
    /// The batches are collected into a `Vec` so they can be replayed
    /// across multiple epochs. For a single epoch with a non-cloneable
    /// iterator, use [`fit_batches_single_epoch`](StreamingFitter::fit_batches_single_epoch).
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during any `partial_fit` call.
    pub fn fit_batches<X, Y, I>(self, batches: I) -> Result<M::FitResult, M::Error>
    where
        M: PartialFit<X, Y>,
        M::FitResult: PartialFit<X, Y, FitResult = M::FitResult, Error = M::Error>,
        I: IntoIterator<Item = (X, Y)>,
    {
        let batches: Vec<(X, Y)> = batches.into_iter().collect();

        if batches.is_empty() || self.n_epochs == 0 {
            // Feed a zero-length sequence: we need at least one batch.
            // This is inherent to PartialFit requiring at least one call.
            // Return an error if there are no batches at all.
            // However, since we can't construct a FitResult without data,
            // we must have at least one batch.
            return Err(self.no_batches_error());
        }

        // First epoch, first batch: transition from M to M::FitResult.
        let mut batch_iter = batches.iter();
        let (first_x, first_y) = batch_iter.next().unwrap();
        let mut fitted = self.model.partial_fit(first_x, first_y)?;

        // First epoch, remaining batches.
        for (x, y) in batch_iter {
            fitted = fitted.partial_fit(x, y)?;
        }

        // Subsequent epochs.
        for _ in 1..self.n_epochs {
            for (x, y) in &batches {
                fitted = fitted.partial_fit(x, y)?;
            }
        }

        Ok(fitted)
    }

    /// Feed batches from a single-pass iterator to the model.
    ///
    /// Unlike [`fit_batches`](StreamingFitter::fit_batches), this method
    /// does not collect the batches, so it only supports a single epoch.
    /// The `n_epochs` setting is ignored.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during any `partial_fit` call.
    pub fn fit_batches_single_epoch<X, Y, I>(self, batches: I) -> Result<M::FitResult, M::Error>
    where
        M: PartialFit<X, Y>,
        M::FitResult: PartialFit<X, Y, FitResult = M::FitResult, Error = M::Error>,
        I: IntoIterator<Item = (X, Y)>,
    {
        let mut iter = batches.into_iter();

        let (first_x, first_y) = match iter.next() {
            Some(batch) => batch,
            None => return Err(self.no_batches_error()),
        };

        let mut fitted = self.model.partial_fit(&first_x, &first_y)?;

        for (x, y) in iter {
            fitted = fitted.partial_fit(&x, &y)?;
        }

        Ok(fitted)
    }
}

impl<M> StreamingFitter<M> {
    /// Produce a "no batches" error. This is a helper that constructs
    /// the appropriate error when no data is available.
    ///
    /// We use a trick: we need `M::Error`, but we can only get it from
    /// a failed `partial_fit` call. Instead, we construct a simple
    /// sentinel. This requires `M::Error: From<&str>` or similar.
    /// Since we cannot guarantee that, we panic with a descriptive message.
    /// This is acceptable because calling `fit_batches` with zero batches
    /// is a programming error.
    fn no_batches_error<E>(&self) -> E
    where
        E: core::fmt::Display,
    {
        // We cannot generically construct an arbitrary error type.
        // Panicking here is acceptable: zero batches is a precondition violation.
        panic!(
            "StreamingFitter::fit_batches called with zero batches; at least one batch is required"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FerroError;
    use crate::traits::{PartialFit, Predict};

    /// A simple accumulator model for testing streaming fits.
    /// Accumulates the sum of all values seen in each batch.
    #[derive(Clone)]
    struct Accumulator {
        sum: f64,
    }

    impl Accumulator {
        fn new() -> Self {
            Self { sum: 0.0 }
        }
    }

    /// The fitted version of Accumulator.
    #[derive(Clone)]
    struct FittedAccumulator {
        sum: f64,
    }

    impl Predict<Vec<f64>> for FittedAccumulator {
        type Output = f64;
        type Error = FerroError;

        fn predict(&self, x: &Vec<f64>) -> Result<f64, FerroError> {
            // Predict: scale each input by the accumulated sum
            Ok(x.iter().sum::<f64>() + self.sum)
        }
    }

    impl PartialFit<Vec<f64>, Vec<f64>> for Accumulator {
        type FitResult = FittedAccumulator;
        type Error = FerroError;

        fn partial_fit(self, x: &Vec<f64>, _y: &Vec<f64>) -> Result<FittedAccumulator, FerroError> {
            Ok(FittedAccumulator {
                sum: self.sum + x.iter().sum::<f64>(),
            })
        }
    }

    impl PartialFit<Vec<f64>, Vec<f64>> for FittedAccumulator {
        type FitResult = FittedAccumulator;
        type Error = FerroError;

        fn partial_fit(self, x: &Vec<f64>, _y: &Vec<f64>) -> Result<FittedAccumulator, FerroError> {
            Ok(FittedAccumulator {
                sum: self.sum + x.iter().sum::<f64>(),
            })
        }
    }

    #[test]
    fn test_streaming_single_batch() {
        let model = Accumulator::new();
        let fitter = StreamingFitter::new(model);

        let batches = vec![(vec![1.0, 2.0, 3.0], vec![0.0])];

        let fitted = fitter.fit_batches(batches).unwrap();
        // Sum of [1, 2, 3] = 6
        let pred = fitted.predict(&vec![0.0]).unwrap();
        assert!((pred - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_multiple_batches() {
        let model = Accumulator::new();
        let fitter = StreamingFitter::new(model);

        let batches = vec![
            (vec![1.0, 2.0], vec![0.0]),
            (vec![3.0, 4.0], vec![0.0]),
            (vec![5.0], vec![0.0]),
        ];

        let fitted = fitter.fit_batches(batches).unwrap();
        // Sum = 1+2+3+4+5 = 15
        let pred = fitted.predict(&vec![0.0]).unwrap();
        assert!((pred - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_multiple_epochs() {
        let model = Accumulator::new();
        let fitter = StreamingFitter::new(model).n_epochs(3);

        let batches = vec![(vec![1.0, 2.0], vec![0.0]), (vec![3.0], vec![0.0])];

        let fitted = fitter.fit_batches(batches).unwrap();
        // Per epoch sum = 1+2+3 = 6, 3 epochs = 18
        let pred = fitted.predict(&vec![0.0]).unwrap();
        assert!((pred - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_single_epoch_method() {
        let model = Accumulator::new();
        let fitter = StreamingFitter::new(model);

        let batches = vec![(vec![10.0], vec![0.0]), (vec![20.0], vec![0.0])];

        let fitted = fitter.fit_batches_single_epoch(batches).unwrap();
        // Sum = 10 + 20 = 30
        let pred = fitted.predict(&vec![0.0]).unwrap();
        assert!((pred - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_predict_after_fit() {
        let model = Accumulator::new();
        let fitter = StreamingFitter::new(model).n_epochs(1);

        let batches = vec![(vec![5.0], vec![0.0])];

        let fitted = fitter.fit_batches(batches).unwrap();
        // Predict with input [1.0, 2.0]: result = (1+2) + 5 = 8
        let pred = fitted.predict(&vec![1.0, 2.0]).unwrap();
        assert!((pred - 8.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "zero batches")]
    fn test_streaming_empty_batches_panics() {
        let model = Accumulator::new();
        let fitter = StreamingFitter::new(model);

        let batches: Vec<(Vec<f64>, Vec<f64>)> = vec![];
        let _ = fitter.fit_batches(batches);
    }

    #[test]
    fn test_streaming_fitter_builder_pattern() {
        let fitter = StreamingFitter::new(Accumulator::new()).n_epochs(5);
        assert_eq!(fitter.n_epochs, 5);
    }
}
