//! Boundary-integrity green-guard suite for `ferrolearn-preprocess/src/lib.rs`'s
//! re-export boundary (REQ-1 of `.design/preprocess/lib.md`).
//!
//! `lib.rs` is **not** an estimator — it owns no `fit`/`transform` logic. It is
//! the crate's public-API surface: the analog of the `__all__` re-export
//! boundary of six scikit-learn 1.5.2 modules collapsed under one crate root:
//!
//!   * `sklearn.preprocessing`        (`preprocessing/__init__.py:30-60` `__all__`)
//!   * `sklearn.feature_selection`    (`feature_selection/__init__.py:27-47` `__all__`)
//!   * `sklearn.feature_extraction`   (`feature_extraction/__init__.py` `__all__`)
//!   * `sklearn.feature_extraction.text` (`text.py:34-43`)
//!   * `sklearn.feature_extraction.image` (`image.py:23-26` scoped subset)
//!   * `sklearn.impute`               (`impute/__init__.py:13` `__all__`)
//!   * `sklearn.random_projection`    (`random_projection.py:50-54` `__all__`)
//!   * `sklearn.compose`              (`compose/__init__.py:15-20` `__all__`)
//!
//! Mirrors the bayes/lib precedent (`ferrolearn-bayes/tests/divergence_lib.rs`,
//! `api_proof` pattern): this file PINS the public surface. The single
//! `use ferrolearn_preprocess::{...}` block below names every re-exported item
//! the design doc claims PRESENT. If any crate-root re-export is ever
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
// `ferrolearn-preprocess/src/lib.rs`. Removal => unresolved import =>
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
    // sklearn.feature_extraction root/text public classes (PRESENT)
    CountVectorizer,
    DictValue,
    DictVectorizer,
    // sklearn.feature_selection __all__ (PRESENT)
    Direction,
    FeatureHasher,
    FeatureHasherInputType,
    FittedBinaryEncoder,
    FittedColumnTransformer,
    FittedCountVectorizer,
    FittedDictVectorizer,
    // sklearn.random_projection __all__ (PRESENT)
    FittedGaussianRandomProjection,
    FittedGenericUnivariateSelect,
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
    FittedTfidfVectorizer,
    FittedVarianceThreshold,
    FunctionTransformer,
    GaussianRandomProjection,
    GenericUnivariateMode,
    GenericUnivariateParam,
    GenericUnivariateSelect,
    HashingVectorizer,
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
    MaxPatches,
    MinMaxScaler,
    MissingIndicator,
    MissingIndicatorFeatures,
    MultiLabelBinarizer,
    NComponents,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    OutputDistribution,
    PatchExtractor,
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
    SelectorMixin,
    SequentialFeatureSelector,
    SimpleImputer,
    SparseRandomProjection,
    SplineTransformer,
    StandardScaler,
    TargetEncoder,
    TfidfNorm,
    TfidfTransformer,
    TfidfVectorizer,
    ThresholdStrategy,
    VarianceThreshold,
    add_dummy_feature,
    // feature-scoring free functions (chi2 / f_classif / f_regression in
    // feature_selection.__all__, plus ferrolearn compute_scores_* helpers)
    chi2,
    compute_scores_classif,
    compute_scores_regression,
    extract_patches_2d,
    f_classif,
    f_regression,
    grid_to_graph,
    img_to_graph,
    johnson_lindenstrauss_min_dim,
    make_column_selector,
    make_column_transformer,
    maxabs_scale,
    minmax_scale,
    power_transform,
    quantile_transform,
    r_regression,
    r_regression_with_options,
    reconstruct_from_patches_2d,
};

/// Type-level no-op: forces `T` to be named, so removing the corresponding
/// re-export turns this into an unresolved-name compile error. Works for any
/// fully-applied type regardless of its generic arity.
fn name_type<T>() {}

/// Boundary-integrity guard for the covered sklearn `__all__` re-export surface.
///
/// Mirrors the *function* of:
///   * `sklearn/preprocessing/__init__.py:30-60` (`__all__`)
///   * `sklearn/feature_selection/__init__.py:27-47` (`__all__`)
///   * `sklearn/feature_extraction/__init__.py` (`__all__`)
///   * `sklearn/feature_extraction/text.py:34-43`
///   * `sklearn/feature_extraction/image.py:23-26` (scoped subset)
///   * `sklearn/impute/__init__.py:13` (`__all__`)
///   * `sklearn/random_projection.py:50-54` (`__all__`)
///   * `sklearn/compose/__init__.py:15-20` (`__all__`)
///
/// Each re-exported estimator surfaced at `ferrolearn-preprocess/src/lib.rs`
/// is named below. If any `pub use` is removed/renamed the `use`
/// block above + the references here fail to compile, pinning the regression.
///
/// PRESENT/ABSENT accounting verified against the live sklearn 1.5.2 `__all__`
/// (R-CHAR-3); ABSENT names (`mutual_info_*`,
/// `f_oneway`,
/// `TransformedTargetRegressor`) are
/// deliberately NOT named here — the boundary ships exactly what is implemented.
#[allow(
    clippy::assertions_on_constants,
    reason = "the real assertion is the COMPILE of the use block + name_type refs; \
the assert!(true) only keeps the harness green when every re-export resolves"
)]
#[test]
fn boundary_integrity_covered_module_all_surface() {
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
    name_type::<MaxPatches>();
    name_type::<TfidfNorm>();
    name_type::<ImputeStrategy<f64>>();

    // --- feature_selection ---
    name_type::<VarianceThreshold<f64>>();
    name_type::<FittedVarianceThreshold<f64>>();
    name_type::<SelectKBest<f64>>();
    name_type::<FittedSelectKBest<f64>>();
    name_type::<GenericUnivariateSelect<f64>>();
    name_type::<FittedGenericUnivariateSelect<f64>>();
    name_type::<GenericUnivariateMode>();
    name_type::<GenericUnivariateParam>();
    name_type::<&dyn SelectorMixin<f64>>();
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
    let _r_regression = r_regression::<f64>;
    let _r_regression_with_options = r_regression_with_options::<f64>;
    let _scores_classif = compute_scores_classif::<f64>;
    let _scores_regression = compute_scores_regression::<f64>;

    // --- feature_extraction root/text ---
    name_type::<CountVectorizer>();
    name_type::<FittedCountVectorizer>();
    name_type::<DictValue>();
    name_type::<DictVectorizer>();
    name_type::<FittedDictVectorizer>();
    name_type::<FeatureHasher>();
    name_type::<FeatureHasherInputType>();
    name_type::<HashingVectorizer>();
    name_type::<TfidfTransformer<f64>>();
    name_type::<FittedTfidfTransformer<f64>>();
    name_type::<TfidfVectorizer>();
    name_type::<FittedTfidfVectorizer>();

    // --- feature_extraction.image (scoped dense helper subset) ---
    name_type::<PatchExtractor<f64>>();
    let _grid_to_graph = grid_to_graph::<f64>;
    let _img_to_graph = img_to_graph::<f64>;
    let _extract_patches_2d = extract_patches_2d::<f64>;
    let _reconstruct_from_patches_2d = reconstruct_from_patches_2d::<f64>;

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
    name_type::<NComponents>();
    let _johnson_lindenstrauss_min_dim = johnson_lindenstrauss_min_dim::<f64>;

    // --- compose ---
    name_type::<ColumnTransformer>();
    name_type::<FittedColumnTransformer>();
    name_type::<ColumnSelector>();
    name_type::<Remainder>();
    let _make_selector = make_column_selector;
    let _make_ct = make_column_transformer;

    // --- ferrolearn extension (no sklearn __all__ analog) ---
    name_type::<BinaryEncoder<f64>>();
    name_type::<FittedBinaryEncoder<f64>>();

    // If we reached here, every re-exported name resolved. The real assertion is
    // the COMPILE itself (boundary integrity); this keeps the harness green.
    assert!(
        true,
        "boundary integrity holds: all covered sklearn __all__ re-exports resolved"
    );
}
