//! # ferrolearn-preprocess
//!
//! Data preprocessing transformers for the ferrolearn machine learning framework.
//!
//! This crate provides standard scalers, encoders, imputers, and feature
//! selection utilities that follow the ferrolearn `Fit`/`Transform` trait
//! pattern.
//!
//! ## Scalers
//!
//! All scalers are generic over `F: Float + Send + Sync + 'static` and implement
//! [`Fit<Array2<F>, ()>`](ferrolearn_core::Fit) (returning a `Fitted*` type) and
//! [`FitTransform<Array2<F>>`](ferrolearn_core::FitTransform). The fitted types
//! implement [`Transform<Array2<F>>`](ferrolearn_core::Transform).
//!
//! - [`StandardScaler`] — zero-mean, unit-variance scaling
//! - [`MinMaxScaler`] — scale features to a given range (default `[0, 1]`)
//! - [`RobustScaler`] — median / IQR-based scaling, robust to outliers
//!
//! ## Encoders
//!
//! - [`OneHotEncoder`] — encode `Array2<usize>` categorical columns as binary columns
//! - [`LabelEncoder`] — map `Array1<String>` labels to integer indices
//!
//! ## Imputers
//!
//! - [`imputer::SimpleImputer`] — fill missing (NaN) values per feature column
//!   using Mean, Median, MostFrequent, or Constant strategy.
//!
//! ## Feature Selection
//!
//! - [`feature_selection::VarianceThreshold`] — remove features with variance
//!   below a configurable threshold.
//! - [`feature_selection::SelectKBest`] — keep the K features with the highest
//!   ANOVA F-scores against class labels.
//! - [`feature_selection::SelectFromModel`] — keep features whose importance
//!   weight (from a pre-fitted model) meets a configurable threshold.
//!
//! ## Pipeline Integration
//!
//! `StandardScaler<f64>`, `MinMaxScaler<f64>`, `RobustScaler<f64>`,
//! `SimpleImputer<f64>`, `VarianceThreshold<f64>`, `SelectKBest<f64>`, and
//! `SelectFromModel<f64>` each implement
//! [`PipelineTransformer`](ferrolearn_core::pipeline::PipelineTransformer)
//! so they can be used as steps inside a
//! [`Pipeline`](ferrolearn_core::pipeline::Pipeline).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_preprocess::StandardScaler;
//! use ferrolearn_core::traits::FitTransform;
//! use ndarray::array;
//!
//! let x = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0]];
//! let scaled = StandardScaler::<f64>::new().fit_transform(&x).unwrap();
//! // scaled columns now have mean ≈ 0 and std ≈ 1
//! ```

pub mod feature_selection;
pub mod imputer;
pub mod label_encoder;
pub mod min_max_scaler;
pub mod one_hot_encoder;
pub mod robust_scaler;
pub mod standard_scaler;

// Re-exports
pub use feature_selection::{
    FittedSelectKBest, FittedVarianceThreshold, ScoreFunc, SelectFromModel, SelectKBest,
    VarianceThreshold,
};
pub use imputer::{FittedSimpleImputer, ImputeStrategy, SimpleImputer};
pub use label_encoder::{FittedLabelEncoder, LabelEncoder};
pub use min_max_scaler::{FittedMinMaxScaler, MinMaxScaler};
pub use one_hot_encoder::{FittedOneHotEncoder, OneHotEncoder};
pub use robust_scaler::{FittedRobustScaler, RobustScaler};
pub use standard_scaler::{FittedStandardScaler, StandardScaler};
