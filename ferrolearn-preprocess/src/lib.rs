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
//! - [`OneHotEncoder`] — encode `Array2<F>` numeric categorical columns as binary columns (per-column sorted-unique `categories_`)
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
//! - [`dummy_feature::add_dummy_feature`] — prepend a constant dummy feature column
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
//!
//! ## REQ status
//!
//! Binary (R-DEFER-2) for the crate-root RE-EXPORT BOUNDARY — this file is the
//! public-API surface, NOT an estimator. Mirrors the `__all__` of six sklearn
//! modules: `preprocessing/__init__.py:30-60`, `feature_selection/__init__.py:27-47`,
//! `feature_extraction/text.py:34-43`, `impute/__init__.py:13`,
//! `random_projection.py:50-54`, `compose/__init__.py:15-20`. Design doc:
//! `.design/preprocess/lib.md`. Per-estimator value parity lives in the sibling
//! routed module docs; this table covers only the boundary surface + the ferray
//! substrate gap. Tracking: #1361.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (re-export boundary) | SHIPPED | the `pub use` block (`:132-202`) surfaces every implemented estimator's unfitted + `Fitted*` pair (plus supporting enums, [`SelectorMixin`], and the `chi2`/`f_classif`/`f_regression`/`r_regression` scoring fns), mirroring the six modules' `__all__`. The surfaced set is the documented subset that is implemented; not-yet-translated names (`HashingVectorizer`, `mutual_info_*`, `johnson_lindenstrauss_min_dim`) are enumerated in the design doc (honest underclaim). Consumers: meta-crate `pub use ferrolearn_preprocess as preprocess;` (`ferrolearn/src/lib.rs:36`) + the `_RsStandardScaler`/`_RsMinMaxScaler`/`_RsMaxAbsScaler`/`_RsRobustScaler`/`_RsPowerTransformer` PyO3 pyclasses (`ferrolearn-python/src/{transformers,extras}.rs`, registered `lib.rs:22,81-84`). Verification: `cargo build -p ferrolearn-preprocess` resolves every re-export; boundary-integrity green-guard `tests/divergence_lib.rs` (fails to compile if any re-export is removed); `cargo test -p ferrolearn-preprocess` green. |
//! | REQ-2 (ferray substrate) | NOT-STARTED | the crate is `ndarray` + `num_traits` across all 37 submodules behind the boundary, not `ferray-core`/`ferray-ufunc` (R-SUBSTRATE-1) — blocker #1362 |
//!
//! `BinaryEncoder`/`FittedBinaryEncoder` (`:163`) is a `category_encoders`-style
//! extension with no sklearn `__all__` analog — an extension of the boundary, not
//! a sklearn-parity item and not a blocker.

pub mod binarizer;
pub mod binary_encoder;
pub mod column_transformer;
pub mod count_vectorizer;
pub mod dummy_feature;
pub mod feature_scoring;
pub mod feature_selection;
pub mod function_transformer;
pub mod generic_univariate_select;
pub mod imputer;
pub mod iterative_imputer;
pub mod kbins_discretizer;
pub mod kernel_centerer;
pub mod knn_imputer;
pub mod label_binarizer;
pub mod label_encoder;
pub mod max_abs_scaler;
pub mod min_max_scaler;
pub mod multi_label_binarizer;
pub mod normalizer;
pub mod one_hot_encoder;
pub mod ordinal_encoder;
pub mod polynomial_features;
pub mod power_transformer;
pub mod quantile_transformer;
pub mod random_projection;
pub mod rfe;
pub mod robust_scaler;
pub mod select_from_model;
pub mod select_percentile;
pub mod selector_mixin;
pub mod sequential_feature_selector;
pub mod spline_transformer;
pub mod standard_scaler;
pub mod stat_selectors;
pub mod target_encoder;
pub mod tfidf;

// Re-exports
pub use binarizer::Binarizer;
pub use column_transformer::{
    ColumnSelector, ColumnTransformer, FittedColumnTransformer, Remainder, make_column_selector,
    make_column_transformer,
};
pub use dummy_feature::add_dummy_feature;
pub use feature_selection::{
    FittedSelectKBest, FittedVarianceThreshold, ScoreFunc, SelectFromModel, SelectKBest,
    VarianceThreshold,
};
pub use function_transformer::FunctionTransformer;
pub use generic_univariate_select::{
    FittedGenericUnivariateSelect, GenericUnivariateMode, GenericUnivariateParam,
    GenericUnivariateSelect,
};
pub use imputer::{
    FittedMissingIndicator, FittedSimpleImputer, ImputeStrategy, MissingIndicator,
    MissingIndicatorFeatures, SimpleImputer,
};
pub use label_encoder::{FittedLabelEncoder, LabelEncoder};
pub use max_abs_scaler::{FittedMaxAbsScaler, MaxAbsScaler, maxabs_scale};
pub use min_max_scaler::{FittedMinMaxScaler, MinMaxScaler, minmax_scale};
pub use normalizer::Normalizer;
pub use one_hot_encoder::{FittedOneHotEncoder, OneHotDrop, OneHotEncoder, OneHotHandleUnknown};
pub use ordinal_encoder::{Categories, FittedOrdinalEncoder, HandleUnknown, OrdinalEncoder};
pub use polynomial_features::{FittedPolynomialFeatures, PolynomialFeatures};
pub use power_transformer::{FittedPowerTransformer, PowerTransformer, power_transform};
pub use robust_scaler::{FittedRobustScaler, RobustScaler};
pub use standard_scaler::{FittedStandardScaler, StandardScaler};

// Phase 3 re-exports
pub use binary_encoder::{BinaryEncoder, FittedBinaryEncoder};
pub use iterative_imputer::{FittedIterativeImputer, InitialStrategy, IterativeImputer};
pub use kbins_discretizer::{BinEncoding, BinStrategy, FittedKBinsDiscretizer, KBinsDiscretizer};
pub use kernel_centerer::{FittedKernelCenterer, KernelCenterer};
pub use knn_imputer::{FittedKNNImputer, KNNImputer, KNNWeights};
pub use quantile_transformer::{
    FittedQuantileTransformer, OutputDistribution, QuantileTransformer, quantile_transform,
};
pub use rfe::{RFE, RFECV};
pub use select_from_model::{FittedSelectFromModelExt, SelectFromModelExt, ThresholdStrategy};
pub use select_percentile::{FittedSelectPercentile, SelectPercentile};
pub use selector_mixin::SelectorMixin;
pub use spline_transformer::{FittedSplineTransformer, KnotStrategy, SplineTransformer};
pub use target_encoder::{FittedTargetEncoder, TargetEncoder};

// Text processing re-exports
pub use count_vectorizer::{CountVectorizer, FittedCountVectorizer};
pub use tfidf::{
    FittedTfidfTransformer, FittedTfidfVectorizer, TfidfNorm, TfidfTransformer, TfidfVectorizer,
};

// Random projection re-exports
pub use random_projection::{
    FittedGaussianRandomProjection, FittedSparseRandomProjection, GaussianRandomProjection,
    SparseRandomProjection,
};

// Newly wired (previously orphaned) re-exports
pub use feature_scoring::{
    chi2, compute_scores_classif, compute_scores_regression, f_classif, f_regression, r_regression,
    r_regression_with_options,
};
pub use label_binarizer::{FittedLabelBinarizer, LabelBinarizer, label_binarize};
pub use multi_label_binarizer::{FittedMultiLabelBinarizer, MultiLabelBinarizer};
pub use sequential_feature_selector::{
    Direction, FittedSequentialFeatureSelector, SequentialFeatureSelector,
};
pub use stat_selectors::{
    FittedSelectFdr, FittedSelectFpr, FittedSelectFwe, SelectFdr, SelectFpr, SelectFwe,
};
