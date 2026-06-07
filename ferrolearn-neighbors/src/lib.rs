//! # ferrolearn-neighbors
//!
//! Nearest neighbor models for the ferrolearn machine learning framework.
//!
//! This crate provides k-nearest neighbors and radius-based nearest neighbor
//! classifiers and regressors, plus an unsupervised nearest neighbor search,
//! with support for brute-force, KD-tree, and ball tree spatial indexing.
//!
//! # Models
//!
//! - **[`KNeighborsClassifier`]** — Classifies samples by majority vote of the
//!   k nearest training samples.
//! - **[`KNeighborsRegressor`]** — Predicts target values as the (weighted) mean
//!   of the k nearest training samples.
//! - **[`RadiusNeighborsClassifier`]** — Classifies samples by majority vote of
//!   all training points within a given radius.
//! - **[`RadiusNeighborsRegressor`]** — Predicts target values as the (weighted)
//!   mean of all training points within a given radius.
//! - **[`NearestNeighbors`]** — Unsupervised nearest neighbor search (no labels).
//!
//! # Spatial Indexing
//!
//! - **[`kdtree::KdTree`]** — A KD-Tree for efficient nearest neighbor search
//!   in low-dimensional spaces (d <= 15).
//! - **[`balltree::BallTree`]** — A ball tree for moderate-to-high dimensions.
//! - **Brute Force** — Exhaustive search used as fallback or when explicitly
//!   requested.
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
//! All supervised models implement
//! [`PipelineEstimator`](ferrolearn_core::pipeline::PipelineEstimator),
//! allowing them to be used as the final step in a
//! [`Pipeline`](ferrolearn_core::pipeline::Pipeline).
//!
//! # Float Generics
//!
//! All models are generic over `F: num_traits::Float + Send + Sync + 'static`,
//! supporting both `f32` and `f64`.

pub mod balltree;
pub mod graph;
pub mod introselect;
pub mod kdtree;
pub mod knn;
pub mod local_outlier_factor;
pub mod nearest_centroid;
pub mod nearest_neighbors;
pub mod radius_neighbors;
pub mod sk_kdtree;

pub use graph::{GraphMode, kneighbors_graph, radius_neighbors_graph, sort_graph_by_row_values};

// Re-export the main types at the crate root.
pub use knn::{
    Algorithm, FittedKNeighborsClassifier, FittedKNeighborsRegressor, KNeighborsClassifier,
    KNeighborsRegressor, Weights,
};
pub use local_outlier_factor::{FittedLocalOutlierFactor, LocalOutlierFactor};
pub use nearest_centroid::{FittedNearestCentroid, NearestCentroid};
pub use nearest_neighbors::{FittedNearestNeighbors, NearestNeighbors};
pub use radius_neighbors::{
    FittedRadiusNeighborsClassifier, FittedRadiusNeighborsRegressor, RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
};
