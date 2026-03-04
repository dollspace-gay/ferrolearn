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
//! - [`MaxAbsScaler`] — scale by maximum absolute value so values are in `[-1, 1]`
//! - [`normalizer::Normalizer`] — normalize each sample (row) to unit norm
//! - [`power_transformer::PowerTransformer`] — Yeo-Johnson power transform
//!
//! ## Encoders
//!
//! - [`OneHotEncoder`] — encode `Array2<usize>` categorical columns as binary columns
//! - [`LabelEncoder`] — map `Array1<String>` labels to integer indices
//! - [`ordinal_encoder::OrdinalEncoder`] — map string categories to integers in
//!   order of first appearance
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
//! ## Feature Engineering
//!
//! - [`polynomial_features::PolynomialFeatures`] — generate polynomial and interaction features
//! - [`binarizer::Binarizer`] — threshold features to binary values
//! - [`function_transformer::FunctionTransformer`] — apply a user-provided function element-wise
//!
//! ## Pipeline Integration
//!
//! `StandardScaler<f64>`, `MinMaxScaler<f64>`, `RobustScaler<f64>`,
//! `MaxAbsScaler<f64>`, `Normalizer<f64>`, `PowerTransformer<f64>`,
//! `PolynomialFeatures<f64>`, `SimpleImputer<f64>`, `VarianceThreshold<f64>`,
//! `SelectKBest<f64>`, and `SelectFromModel<f64>` each implement
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

pub mod binarizer;
pub mod column_transformer;
pub mod feature_selection;
pub mod function_transformer;
pub mod imputer;
pub mod label_encoder;
pub mod max_abs_scaler;
pub mod min_max_scaler;
pub mod normalizer;
pub mod one_hot_encoder;
pub mod ordinal_encoder;
pub mod polynomial_features;
pub mod power_transformer;
pub mod robust_scaler;
pub mod standard_scaler;

// Re-exports
pub use binarizer::Binarizer;
pub use column_transformer::{
    ColumnSelector, ColumnTransformer, FittedColumnTransformer, Remainder, make_column_transformer,
};
pub use feature_selection::{
    FittedSelectKBest, FittedVarianceThreshold, ScoreFunc, SelectFromModel, SelectKBest,
    VarianceThreshold,
};
pub use function_transformer::FunctionTransformer;
pub use imputer::{FittedSimpleImputer, ImputeStrategy, SimpleImputer};
pub use label_encoder::{FittedLabelEncoder, LabelEncoder};
pub use max_abs_scaler::{FittedMaxAbsScaler, MaxAbsScaler};
pub use min_max_scaler::{FittedMinMaxScaler, MinMaxScaler};
pub use normalizer::Normalizer;
pub use one_hot_encoder::{FittedOneHotEncoder, OneHotEncoder};
pub use ordinal_encoder::{FittedOrdinalEncoder, OrdinalEncoder};
pub use polynomial_features::PolynomialFeatures;
pub use power_transformer::{FittedPowerTransformer, PowerTransformer};
pub use robust_scaler::{FittedRobustScaler, RobustScaler};
pub use standard_scaler::{FittedStandardScaler, StandardScaler};
