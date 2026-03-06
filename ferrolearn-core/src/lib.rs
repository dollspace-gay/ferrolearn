//! # ferrolearn-core
//!
//! Core traits, error types, dataset abstractions, and pipeline infrastructure
//! for the ferrolearn machine learning framework.
//!
//! This crate defines the foundational abstractions that all other ferrolearn
//! crates depend on:
//!
//! - **[`Fit`]**, **[`Predict`]**, **[`Transform`]**, **[`FitTransform`]** --
//!   the core ML traits with compile-time enforcement that `predict()` cannot
//!   be called on an unfitted model.
//! - **[`FerroError`]** -- the unified error type with rich diagnostic context.
//! - **[`Dataset`]** -- a trait for querying tabular data shape, with
//!   implementations for `ndarray::Array2<f32>` and `ndarray::Array2<f64>`.
//! - **[`pipeline::Pipeline`]** -- a dynamic-dispatch pipeline that composes
//!   transformers and a final estimator (requires `std` feature).
//! - **Introspection traits** -- [`HasCoefficients`], [`HasFeatureImportances`],
//!   [`HasClasses`] for inspecting fitted model internals.
//!
//! # Features
//!
//! - `std` (default) -- Enables `std`-dependent functionality (I/O errors,
//!   pipelines, `std::error::Error` impls).
//! - `faer` (default) -- Enables the [`NdarrayFaerBackend`] using the `faer`
//!   crate for pure-Rust linear algebra.
//! - `blas` -- Enables the [`BLASBackend`](backend_blas::BLASBackend) using
//!   system BLAS/LAPACK via `ndarray-linalg`.
//!
//! # Design Principles
//!
//! ## Compile-Time Safety (AC-3)
//!
//! The unfitted configuration struct (e.g., `LinearRegression`) implements
//! [`Fit`] but **not** [`Predict`]. Calling `fit()` returns a *new fitted
//! type* (e.g., `FittedLinearRegression`) that implements [`Predict`].
//! Attempting to call `predict()` on an unfitted model is a compile error.
//!
//! ## Float Generics (REQ-15)
//!
//! All algorithms are generic over `F: num_traits::Float + Send + Sync + 'static`.
//!
//! ## Error Handling
//!
//! All public functions return `Result<T, FerroError>`. Library code never panics.
//!
//! ## Pluggable Backends (REQ-19)
//!
//! The [`Backend`] trait abstracts linear algebra operations (SVD, QR, Cholesky,
//! etc.), allowing algorithms to be generic over the backend implementation.
//! The default backend [`NdarrayFaerBackend`] delegates to the `faer` crate.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod backend;
#[cfg(feature = "blas")]
pub mod backend_blas;
#[cfg(feature = "faer")]
pub mod backend_faer;
pub mod dataset;
pub mod error;
pub mod introspection;
#[cfg(feature = "std")]
pub mod pipeline;
pub mod streaming;
pub mod traits;
#[cfg(feature = "std")]
pub mod typed_pipeline;

// Re-export the most commonly used items at the crate root.
pub use backend::Backend;
#[cfg(feature = "blas")]
pub use backend_blas::BLASBackend;
#[cfg(feature = "faer")]
pub use backend_faer::NdarrayFaerBackend;
pub use dataset::Dataset;
pub use error::{FerroError, FerroResult};
pub use introspection::{HasClasses, HasCoefficients, HasFeatureImportances};
pub use streaming::StreamingFitter;
pub use traits::{Fit, FitTransform, PartialFit, Predict, Transform};

/// The default linear algebra backend.
///
/// Algorithms generic over [`Backend`] can use `DefaultBackend` as a sensible
/// default that delegates to the `faer` crate for high-performance pure-Rust
/// implementations of SVD, QR, Cholesky, and other decompositions.
#[cfg(feature = "faer")]
pub type DefaultBackend = NdarrayFaerBackend;
