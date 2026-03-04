//! # ferrolearn-neighbors
//!
//! Nearest neighbor models for the ferrolearn machine learning framework.
//!
//! This crate provides k-nearest neighbors classifiers and regressors with
//! support for both brute-force and KD-tree spatial indexing.
//!
//! # Models
//!
//! - **[`KNeighborsClassifier`]** — Classifies samples by majority vote of the
//!   k nearest training samples.
//! - **[`KNeighborsRegressor`]** — Predicts target values as the (weighted) mean
//!   of the k nearest training samples.
//!
//! # Spatial Indexing
//!
//! - **[`kdtree::KdTree`]** — A KD-Tree for efficient nearest neighbor search
//!   in low-dimensional spaces (d <= 20).
//! - **Brute Force** — Exhaustive search used as fallback for high-dimensional
//!   data or when explicitly requested.
//!
//! # Design
//!
//! Each model follows the compile-time safety pattern:
//!
//! - The unfitted struct (e.g., `KNeighborsClassifier<F>`) holds hyperparameters
//!   and implements [`Fit`](ferrolearn_core::Fit).
//! - Calling `fit()` stores the training data and optionally builds a spatial
//!   index, producing a fitted type (e.g., `FittedKNeighborsClassifier<F>`)
//!   that implements [`Predict`](ferrolearn_core::Predict).
//! - Calling `predict()` on an unfitted model is a compile-time error.
//!
//! # Pipeline Integration
//!
//! Both models implement [`PipelineEstimator`](ferrolearn_core::pipeline::PipelineEstimator)
//! for `f64`, allowing them to be used as the final step in a
//! [`Pipeline`](ferrolearn_core::pipeline::Pipeline).
//!
//! # Float Generics
//!
//! All models are generic over `F: num_traits::Float + Send + Sync + 'static`,
//! supporting both `f32` and `f64`.

pub mod balltree;
pub mod kdtree;
pub mod knn;

// Re-export the main types at the crate root.
pub use knn::{
    Algorithm, FittedKNeighborsClassifier, FittedKNeighborsRegressor, KNeighborsClassifier,
    KNeighborsRegressor, Weights,
};
