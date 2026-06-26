//! Boundary-integrity green-guard suite for `ferrolearn-preprocess/src/lib.rs`'s
//! re-export boundary (REQ-1 of `.design/preprocess/lib.md`).
//!
//! `lib.rs` is **not** an estimator — it owns no `fit`/`transform` logic. It is
//! the crate's public-API surface: the analog of the `__all__` re-export
//! boundary of six scikit-learn 1.5.2 modules collapsed under one crate root:
//!
//!   * `sklearn.preprocessing`        (`preprocessing/__init__.py:30-60` `__all__`)
//!   * `sklearn.feature_selection`    (`feature_selection/__init__.py:27-47` `__all__`)
//!   * `sklearn.feature_extraction.text` (`text.py:34-43`)
//!   * `sklearn.impute`               (`impute/__init__.py:13` `__all__`)
//!   * `sklearn.random_projection`    (`random_projection.py:50-54` `__all__`)
//!   * `sklearn.compose`              (`compose/__init__.py:15-20` `__all__`)
//!
//! Mirrors the bayes/lib precedent (`ferrolearn-bayes/tests/divergence_lib.rs`,
//! `api_proof` pattern): this file PINS the public surface. The single
//! `use ferrolearn_preprocess::{...}` block below names every re-exported item
//! the design doc claims PRESENT. If any re-export at `lib.rs:106-163` is ever
//! removed or renamed, this test file fails to COMPILE — that compile failure is
//! the boundary's regression guard (AC-1: every name in the `pub use` block
//! resolves to an existing type/function).
//!
//! This is a BOUNDARY test, not a per-estimator value-parity test. Per-estimator
//! `fit`/`transform` ULP parity lives in the sibling routed docs
//! (`.design/preprocess/<estimator>.md`) and their own divergence suites.
//!
//! R-CHAR-3: the PRESENT/ABSENT accounting these `use` statements encode was
//! verified against the LIVE sklearn 1.5.2 `__all__` lists (the probes quoted in
//! `.design/preprocess/lib.md` Probes), not transcribed from the ferrolearn side.

// The compile-time boundary guard. Every name here is a re-export at
// `ferrolearn-preprocess/src/lib.rs:106-163`. Removal => unresolved import =>
// this crate's test build fails. This is intentional and load-bearing.
use ferrolearn_preprocess::{
    // supporting enums for the preprocessing estimators
    BinEncoding,
    BinStrategy,
    // sklearn.preprocessing __all__ (PRESENT estimator classes, each unfitted + Fitted*)
    Binarizer,
    // ferrolearn extension of the boundary (no sklearn __all__ analog)
    BinaryEncoder,
    // sklearn.compose __all__ (PRESENT)
    ColumnSelector,
    ColumnTransformer,
    // sklearn.feature_extraction.text public classes (PRESENT)
    CountVectorizer,
    // sklearn.feature_selection __all__ (PRESENT)
    Direction,
    FittedBinaryEncoder,
    FittedColumnTransformer,
    FittedCountVectorizer,
    // sklearn.random_projection __all__ (PRESENT)
    FittedGaussianRandomProjection,
    // sklearn.impute __all__ (PRESENT; IterativeImputer is experimental in 1.5.2)
    FittedIterativeImputer,
    FittedKBinsDiscretizer,
    FittedKNNImputer,
    FittedKernelCenterer,
    FittedLabelBinarizer,
    FittedLabelEncoder,
    FittedMaxAbsScaler,
    FittedMinMaxScaler,
    FittedMissingIndicator,
    FittedMultiLabelBinarizer,
    FittedOneHotEncoder,
    FittedOrdinalEncoder,
    FittedPowerTransformer,
    FittedQuantileTransformer,
    FittedRobustScaler,
    FittedSelectFdr,
    FittedSelectFpr,
    FittedSelectFromModelExt,
    FittedSelectFwe,
    FittedSelectKBest,
    FittedSelectPercentile,
    FittedSequentialFeatureSelector,
    FittedSimpleImputer,
    FittedSparseRandomProjection,
    FittedSplineTransformer,
    FittedStandardScaler,
    FittedTargetEncoder,
    FittedTfidfTransformer,
    FittedVarianceThreshold,
    FunctionTransformer,
    GaussianRandomProjection,
    ImputeStrategy,
    InitialStrategy,
    IterativeImputer,
    KBinsDiscretizer,
    KNNImputer,
    KNNWeights,
    KernelCenterer,
    KnotStrategy,
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    MissingIndicator,
    MissingIndicatorFeatures,
    MultiLabelBinarizer,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    OutputDistribution,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RFE,
    RFECV,
    Remainder,
    RobustScaler,
    ScoreFunc,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFromModelExt,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
    SimpleImputer,
    SparseRandomProjection,
    SplineTransformer,
    StandardScaler,
    TargetEncoder,
    TfidfNorm,
    TfidfTransformer,
    ThresholdStrategy,
    VarianceThreshold,
    add_dummy_feature,
    // feature-scoring free functions (chi2 / f_classif / f_regression in
    // feature_selection.__all__, plus ferrolearn compute_scores_* helpers)
    chi2,
    compute_scores_classif,
    compute_scores_regression,
    f_classif,
    f_regression,
    make_column_transformer,
    maxabs_scale,
    minmax_scale,
    power_transform,
    quantile_transform,
};

/// Type-level no-op: forces `T` to be named, so removing the corresponding
/// re-export turns this into an unresolved-name compile error. Works for any
/// fully-applied type regardless of its generic arity.
fn name_type<T>() {}

/// Boundary-integrity guard for the six-module `__all__` re-export surface.
///
/// Mirrors the *function* of:
///   * `sklearn/preprocessing/__init__.py:30-60` (`__all__`)
///   * `sklearn/feature_selection/__init__.py:27-47` (`__all__`)
///   * `sklearn/feature_extraction/text.py:34-43`
///   * `sklearn/impute/__init__.py:13` (`__all__`)
///   * `sklearn/random_projection.py:50-54` (`__all__`)
///   * `sklearn/compose/__init__.py:15-20` (`__all__`)
///
/// Each re-exported estimator surfaced at `ferrolearn-preprocess/src/lib.rs`
/// (`:106-163`) is named below. If any `pub use` is removed/renamed the `use`
/// block above + the references here fail to compile, pinning the regression.
///
/// PRESENT/ABSENT accounting verified against the live sklearn 1.5.2 `__all__`
/// (R-CHAR-3); ABSENT names (`GenericUnivariateSelect`,
/// `mutual_info_*`, `r_regression`,
/// `f_oneway`, `SelectorMixin`,
/// `johnson_lindenstrauss_min_dim`, `make_column_selector`,
/// `TransformedTargetRegressor`, `HashingVectorizer`, `TfidfVectorizer`) are
/// deliberately NOT named here — the boundary ships exactly what is implemented.
#[allow(
    clippy::assertions_on_constants,
    reason = "the real assertion is the COMPILE of the use block + name_type refs; \
the assert!(true) only keeps the harness green when every re-export resolves"
)]
#[test]
fn boundary_integrity_six_module_all_surface() {
    // --- preprocessing: unfitted + Fitted* pairs (and single-type transformers) ---
    name_type::<StandardScaler<f64>>();
    name_type::<FittedStandardScaler<f64>>();
    name_type::<MinMaxScaler<f64>>();
    name_type::<FittedMinMaxScaler<f64>>();
    name_type::<MaxAbsScaler<f64>>();
    name_type::<FittedMaxAbsScaler<f64>>();
    name_type::<RobustScaler<f64>>();
    name_type::<FittedRobustScaler<f64>>();
    name_type::<Normalizer<f64>>();
    name_type::<PowerTransformer<f64>>();
    name_type::<FittedPowerTransformer<f64>>();
    name_type::<QuantileTransformer<f64>>();
    name_type::<FittedQuantileTransformer<f64>>();
    name_type::<Binarizer<f64>>();
    name_type::<FunctionTransformer<f64>>();
    name_type::<PolynomialFeatures<f64>>();
    name_type::<SplineTransformer<f64>>();
    name_type::<FittedSplineTransformer<f64>>();
    name_type::<KBinsDiscretizer<f64>>();
    name_type::<FittedKBinsDiscretizer<f64>>();
    name_type::<KernelCenterer<f64>>();
    name_type::<FittedKernelCenterer<f64>>();
    name_type::<TargetEncoder<f64>>();
    name_type::<FittedTargetEncoder<f64>>();
    // preprocessing free functions shipped at the crate root
    let _add_dummy_feature = add_dummy_feature::<f64>;
    let _maxabs_scale = maxabs_scale::<f64>;
    let _minmax_scale = minmax_scale::<f64>;
    let _power_transform = power_transform::<f64>;
    let _quantile_transform = quantile_transform::<f64>;
    name_type::<OneHotEncoder<f64>>();
    name_type::<FittedOneHotEncoder<f64>>();
    name_type::<OrdinalEncoder>();
    name_type::<FittedOrdinalEncoder>();
    name_type::<LabelEncoder>();
    name_type::<FittedLabelEncoder>();
    name_type::<LabelBinarizer>();
    name_type::<FittedLabelBinarizer>();
    name_type::<MultiLabelBinarizer>();
    name_type::<FittedMultiLabelBinarizer>();
    // supporting enums
    name_type::<BinEncoding>();
    name_type::<BinStrategy>();
    name_type::<KnotStrategy>();
    name_type::<OutputDistribution>();
    name_type::<TfidfNorm>();
    name_type::<ImputeStrategy<f64>>();

    // --- feature_selection ---
    name_type::<VarianceThreshold<f64>>();
    name_type::<FittedVarianceThreshold<f64>>();
    name_type::<SelectKBest<f64>>();
    name_type::<FittedSelectKBest<f64>>();
    name_type::<SelectPercentile<f64>>();
    name_type::<FittedSelectPercentile<f64>>();
    name_type::<SelectFromModel<f64>>();
    name_type::<SelectFromModelExt<f64>>();
    name_type::<FittedSelectFromModelExt<f64>>();
    name_type::<SelectFdr<f64>>();
    name_type::<FittedSelectFdr<f64>>();
    name_type::<SelectFpr<f64>>();
    name_type::<FittedSelectFpr<f64>>();
    name_type::<SelectFwe<f64>>();
    name_type::<FittedSelectFwe<f64>>();
    name_type::<RFE<f64>>();
    name_type::<RFECV<f64>>();
    name_type::<SequentialFeatureSelector>();
    name_type::<FittedSequentialFeatureSelector<f64>>();
    name_type::<ScoreFunc>();
    name_type::<ThresholdStrategy>();
    name_type::<Direction>();
    // feature-scoring free functions: name each fn item (coerced to a value)
    let _chi2 = chi2::<f64>;
    let _f_classif = f_classif::<f64>;
    let _f_regression = f_regression::<f64>;
    let _scores_classif = compute_scores_classif::<f64>;
    let _scores_regression = compute_scores_regression::<f64>;

    // --- feature_extraction.text ---
    name_type::<CountVectorizer>();
    name_type::<FittedCountVectorizer>();
    name_type::<TfidfTransformer<f64>>();
    name_type::<FittedTfidfTransformer<f64>>();

    // --- impute (IterativeImputer is experimental in sklearn 1.5.2) ---
    name_type::<SimpleImputer<f64>>();
    name_type::<FittedSimpleImputer<f64>>();
    name_type::<MissingIndicator<f64>>();
    name_type::<FittedMissingIndicator<f64>>();
    name_type::<MissingIndicatorFeatures>();
    name_type::<KNNImputer<f64>>();
    name_type::<FittedKNNImputer<f64>>();
    name_type::<IterativeImputer<f64>>();
    name_type::<FittedIterativeImputer<f64>>();
    name_type::<KNNWeights>();
    name_type::<InitialStrategy>();

    // --- random_projection ---
    name_type::<GaussianRandomProjection<f64>>();
    name_type::<FittedGaussianRandomProjection<f64>>();
    name_type::<SparseRandomProjection<f64>>();
    name_type::<FittedSparseRandomProjection<f64>>();

    // --- compose ---
    name_type::<ColumnTransformer>();
    name_type::<FittedColumnTransformer>();
    name_type::<ColumnSelector>();
    name_type::<Remainder>();
    let _make_ct = make_column_transformer;

    // --- ferrolearn extension (no sklearn __all__ analog) ---
    name_type::<BinaryEncoder<f64>>();
    name_type::<FittedBinaryEncoder<f64>>();

    // If we reached here, every re-exported name resolved. The real assertion is
    // the COMPILE itself (boundary integrity); this keeps the harness green.
    assert!(
        true,
        "boundary integrity holds: all six-module __all__ re-exports resolved"
    );
}
