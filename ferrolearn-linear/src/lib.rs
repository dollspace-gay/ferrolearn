//! # ferrolearn-linear
//!
//! Linear models for the ferrolearn machine learning framework.
//!
//! This crate provides implementations of the most common linear models
//! for both regression and classification tasks:
//!
//! - **[`LinearRegression`]** — Ordinary Least Squares via QR decomposition
//! - **[`Ridge`]** — L2-regularized regression via Cholesky decomposition
//! - **[`Lasso`]** — L1-regularized regression via coordinate descent
//! - **[`ElasticNet`]** — Combined L1/L2 regularization via coordinate descent
//! - **[`BayesianRidge`]** — Bayesian Ridge with automatic regularization tuning
//! - **[`HuberRegressor`]** — Robust regression via IRLS with Huber loss
//! - **[`LogisticRegression`]** — Binary and multiclass classification via L-BFGS
//!
//! All models implement the [`ferrolearn_core::Fit`] and [`ferrolearn_core::Predict`]
//! traits, and produce fitted types that implement [`ferrolearn_core::introspection::HasCoefficients`].
//!
//! # Design
//!
//! Each model follows the compile-time safety pattern:
//!
//! - The unfitted struct (e.g., `LinearRegression<F>`) holds hyperparameters
//!   and implements [`Fit`](ferrolearn_core::Fit).
//! - Calling `fit()` produces a new fitted type (e.g., `FittedLinearRegression<F>`)
//!   that implements [`Predict`](ferrolearn_core::Predict).
//! - Calling `predict()` on an unfitted model is a compile-time error.
//!
//! # Pipeline Integration
//!
//! All models implement [`PipelineEstimator`](ferrolearn_core::pipeline::PipelineEstimator)
//! for `f64`, allowing them to be used as the final step in a
//! [`Pipeline`](ferrolearn_core::pipeline::Pipeline).
//!
//! # Float Generics
//!
//! All models are generic over `F: num_traits::Float + Send + Sync + 'static`,
//! supporting both `f32` and `f64`.

pub mod bayesian_ridge;
pub mod elastic_net;
pub mod huber_regressor;
pub mod lasso;
mod linalg;
pub mod linear_regression;
pub mod logistic_regression;
mod optim;
pub mod ridge;
pub mod sgd;

// Re-export the main types at the crate root.
pub use bayesian_ridge::{BayesianRidge, FittedBayesianRidge};
pub use elastic_net::{ElasticNet, FittedElasticNet};
pub use huber_regressor::{FittedHuberRegressor, HuberRegressor};
pub use lasso::{FittedLasso, Lasso};
pub use linear_regression::{FittedLinearRegression, LinearRegression};
pub use logistic_regression::{FittedLogisticRegression, LogisticRegression};
pub use ridge::{FittedRidge, Ridge};
pub use sgd::{FittedSGDClassifier, FittedSGDRegressor, SGDClassifier, SGDRegressor};
