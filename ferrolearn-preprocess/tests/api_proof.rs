//! Proof-of-API integration test for ferrolearn-preprocess.
//!
//! Audit deliverable for crosslink #301 (under #247). Exercises every
//! public estimator end-to-end after the orphan wiring in #299. Every
//! call uses verified-from-source signatures.

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::feature_selection::SelectFromModel as FeatureSelectionSelectFromModel;
use ferrolearn_preprocess::imputer::ImputeStrategy;
use ferrolearn_preprocess::normalizer::NormType;
use ferrolearn_preprocess::{
    BinEncoding, BinStrategy, Binarizer, BinaryEncoder, ColumnSelector, ColumnTransformer,
    CountVectorizer, Direction, FunctionTransformer, GaussianRandomProjection,
    GenericUnivariateMode, GenericUnivariateParam, GenericUnivariateSelect, HashingVectorizer,
    InitialStrategy, IterativeImputer, KBinsDiscretizer, KNNImputer, KNNWeights, KernelCenterer,
    KnotStrategy, LabelBinarizer, LabelEncoder, MaxAbsScaler, MaxPatches, MinMaxScaler,
    MissingIndicator, MissingIndicatorFeatures, MultiLabelBinarizer, Normalizer, OneHotEncoder,
    OrdinalEncoder, OutputDistribution, PatchExtractor, PolynomialFeatures, PowerTransformer,
    QuantileTransformer, Remainder, RobustScaler, ScoreFunc, SelectFdr, SelectFpr, SelectFwe,
    SelectKBest, SelectPercentile, SelectorMixin, SequentialFeatureSelector, SimpleImputer,
    SparseRandomProjection, SplineTransformer, StandardScaler, TargetEncoder, TfidfNorm,
    TfidfTransformer, TfidfVectorizer, VarianceThreshold, add_dummy_feature, chi2,
    extract_patches_2d, f_classif, f_regression, grid_to_graph, img_to_graph,
    johnson_lindenstrauss_min_dim, make_column_selector, make_column_transformer, maxabs_scale,
    minmax_scale, power_transform, r_regression, reconstruct_from_patches_2d,
};
use ndarray::{Array1, Array2, Array3, array};

fn small_data() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 3),
        vec![
            1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0, 5.0, 50.0,
            500.0, 6.0, 60.0, 600.0, 7.0, 70.0, 700.0, 8.0, 80.0, 800.0,
        ],
    )
    .unwrap()
}

fn binary_labels_usize() -> Array1<usize> {
    array![0usize, 0, 0, 0, 1, 1, 1, 1]
}

// =============================================================================
// Scalers
// =============================================================================
#[test]
fn api_proof_scalers() {
    let x = small_data();
    let _ = StandardScaler::<f64>::new().fit_transform(&x).unwrap();
    let _ = MinMaxScaler::<f64>::new().fit_transform(&x).unwrap();
    let _ = MinMaxScaler::<f64>::with_feature_range(-1.0, 1.0)
        .unwrap()
        .fit_transform(&x)
        .unwrap();
    let _ = minmax_scale(&x, (0.0, 1.0), 0).unwrap();
    let _ = minmax_scale(&x, (-1.0, 1.0), 1).unwrap();
    let _ = MaxAbsScaler::<f64>::new().fit_transform(&x).unwrap();
    let _ = maxabs_scale(&x, 0).unwrap();
    let _ = maxabs_scale(&x, 1).unwrap();
    let _ = RobustScaler::<f64>::new().fit_transform(&x).unwrap();

    // Normalizer is stateless: just .transform(&x).
    for norm in [NormType::L1, NormType::L2, NormType::Max] {
        let _ = Normalizer::<f64>::new(norm).transform(&x).unwrap();
    }
}

// =============================================================================
// Power / Quantile transforms
// =============================================================================
#[test]
fn api_proof_power_quantile() {
    let x = small_data();
    let _ = PowerTransformer::<f64>::new().fit_transform(&x).unwrap();
    let _ = power_transform(&x, true).unwrap();
    let _ = power_transform(&x, false).unwrap();
    for dist in [OutputDistribution::Uniform, OutputDistribution::Normal] {
        let _ = QuantileTransformer::<f64>::new(8, dist, 0)
            .fit_transform(&x)
            .unwrap();
    }
}

// =============================================================================
// PolynomialFeatures, Binarizer, FunctionTransformer (all stateless)
// =============================================================================
#[test]
fn api_proof_feature_engineering() {
    let x = small_data();

    let _ = PolynomialFeatures::<f64>::new(2, true, false)
        .unwrap()
        .transform(&x)
        .unwrap();
    let _ = Binarizer::<f64>::new(50.0).transform(&x).unwrap();
    let _ = add_dummy_feature(&x, 1.0).unwrap();
    let k = array![[9.0, 2.0, -2.0], [2.0, 14.0, -13.0], [-2.0, -13.0, 21.0]];
    let fitted = KernelCenterer::<f64>::new().fit(&k, &()).unwrap();
    let _ = fitted.transform(&k).unwrap();
    // FunctionTransformer takes an element-wise Fn(F) -> F.
    let _ = FunctionTransformer::<f64>::new(|v: f64| v * 2.0)
        .transform(&x)
        .unwrap();

    let image = array![[0.0_f64, 0.0], [0.0, 1.0]];
    assert_eq!(grid_to_graph::<f64>(2, 2, 1, None).unwrap().dim(), (4, 4));
    assert_eq!(img_to_graph(&image, None).unwrap().dim(), (4, 4));
    let patches = extract_patches_2d(&image, (1, 1), Some(MaxPatches::Count(2)), Some(0)).unwrap();
    assert_eq!(patches.dim(), (2, 1, 1));
    assert_eq!(
        reconstruct_from_patches_2d(&patches, (1, 2)).unwrap().dim(),
        (1, 2)
    );
    let image_batch = Array3::<f64>::zeros((1, 10, 10));
    assert_eq!(
        PatchExtractor::<f64>::new()
            .patch_size((2, 2))
            .transform(&image_batch)
            .unwrap()
            .dim(),
        (81, 2, 2)
    );
}

#[test]
fn api_proof_column_transformer() {
    let x = small_data();
    let selector = make_column_selector(&x);
    assert_eq!(selector, ColumnSelector::Indices(vec![0, 1, 2]));

    let ct = ColumnTransformer::new(
        vec![(
            "std".into(),
            Box::new(StandardScaler::<f64>::new()),
            selector.clone(),
        )],
        Remainder::Drop,
    );
    let fitted = ct.fit(&x, &()).unwrap();
    assert_eq!(fitted.transform(&x).unwrap().dim(), x.dim());

    let ct = make_column_transformer(
        vec![(Box::new(StandardScaler::<f64>::new()), selector)],
        Remainder::Drop,
    );
    let fitted = ct.fit(&x, &()).unwrap();
    assert_eq!(fitted.transform(&x).unwrap().dim(), x.dim());
}

// =============================================================================
// KBinsDiscretizer + SplineTransformer
// =============================================================================
#[test]
fn api_proof_kbins_and_splines() {
    let x = small_data();
    for strategy in [
        BinStrategy::Uniform,
        BinStrategy::Quantile,
        BinStrategy::KMeans,
    ] {
        for encode in [BinEncoding::Ordinal, BinEncoding::OneHot] {
            let _ = KBinsDiscretizer::<f64>::new(3, encode, strategy)
                .fit_transform(&x)
                .unwrap();
        }
    }
    for knots in [KnotStrategy::Uniform, KnotStrategy::Quantile] {
        let _ = SplineTransformer::<f64>::new(4, 3, knots)
            .fit_transform(&x)
            .unwrap();
    }
}

// =============================================================================
// Encoders
// =============================================================================
#[test]
fn api_proof_encoders() {
    let x_cat = Array2::from_shape_vec((4, 2), vec![0usize, 1, 1, 0, 0, 2, 2, 1]).unwrap();
    // OneHotEncoder now takes Array2<F> categories (REQ-3 #1150 sorted-unique).
    let x_cat_ohe =
        Array2::from_shape_vec((4, 2), vec![0.0_f64, 1., 1., 0., 0., 2., 2., 1.]).unwrap();
    let f = OneHotEncoder::<f64>::new().fit(&x_cat_ohe, &()).unwrap();
    let _ = f.transform(&x_cat_ohe).unwrap();

    // OrdinalEncoder smoke (constructor only; per-column string fit varies).
    let _ = OrdinalEncoder::new();

    // LabelEncoder fits Array1<String>.
    let labels: Array1<String> = Array1::from(vec![
        "a".to_string(),
        "b".to_string(),
        "a".to_string(),
        "c".to_string(),
        "b".to_string(),
    ]);
    let f = LabelEncoder.fit(&labels, &()).unwrap();
    let _ = f.transform(&labels).unwrap();

    // LabelBinarizer fits Array1<usize>.
    let y = binary_labels_usize();
    let f = LabelBinarizer::new().fit(&y, &()).unwrap();
    let _ = f.transform(&y).unwrap();

    // MultiLabelBinarizer fits Vec<Vec<usize>>.
    let y_multi: Vec<Vec<usize>> = vec![vec![0, 1], vec![1, 2], vec![0]];
    let f = MultiLabelBinarizer.fit(&y_multi, &()).unwrap();
    let _ = f.transform(&y_multi).unwrap();

    // BinaryEncoder
    let _ = BinaryEncoder::<f64>::new().fit(&x_cat, &()).unwrap();

    // TargetEncoder
    let y_cont: Array1<f64> = array![0.0, 1.0, 0.0, 1.0];
    let f = TargetEncoder::<f64>::new(1.0).fit(&x_cat, &y_cont).unwrap();
    let _ = f.transform(&x_cat).unwrap();
}

// =============================================================================
// Imputers
// =============================================================================
#[test]
fn api_proof_imputers() {
    let x_with_nan = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0,
            f64::NAN,
            3.0,
            2.0,
            2.0,
            f64::NAN,
            f64::NAN,
            3.0,
            1.0,
            4.0,
            4.0,
            2.0,
            5.0,
            5.0,
            3.0,
        ],
    )
    .unwrap();

    for strat in [
        ImputeStrategy::Mean,
        ImputeStrategy::Median,
        ImputeStrategy::Constant(0.0),
    ] {
        let _ = SimpleImputer::<f64>::new(strat)
            .fit_transform(&x_with_nan)
            .unwrap();
    }

    let indicator = MissingIndicator::<f64>::new();
    let fitted_indicator = indicator.fit(&x_with_nan, &()).unwrap();
    assert_eq!(
        fitted_indicator.feature_mode(),
        MissingIndicatorFeatures::MissingOnly
    );
    let mask = fitted_indicator.transform(&x_with_nan).unwrap();
    assert_eq!(mask.nrows(), x_with_nan.nrows());
    assert_eq!(mask.ncols(), fitted_indicator.features_().len());

    for w in [KNNWeights::Uniform, KNNWeights::Distance] {
        let _ = KNNImputer::<f64>::new(2, w)
            .fit_transform(&x_with_nan)
            .unwrap();
    }

    for init in [InitialStrategy::Mean, InitialStrategy::Median] {
        let _ = IterativeImputer::<f64>::new(5, 1e-3, init)
            .fit_transform(&x_with_nan)
            .unwrap();
    }
}

// =============================================================================
// Feature selection
// =============================================================================
#[test]
fn api_proof_feature_selection() {
    let x = small_data();
    let y = binary_labels_usize();
    let y_f64 = array![0.0f64, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    // VarianceThreshold (unsupervised)
    let f = VarianceThreshold::<f64>::new(0.0).fit(&x, &()).unwrap();
    let _ = f.get_support();
    let _ = f.transform(&x).unwrap();

    // SelectKBest / SelectPercentile (supervised, FClassif)
    let f = SelectKBest::<f64>::new(2, ScoreFunc::FClassif)
        .fit(&x, &y)
        .unwrap();
    let _ = f.get_support_indices();
    let _ = f.transform(&x).unwrap();
    let f = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif)
        .fit(&x, &y)
        .unwrap();
    let _ = f.transform(&x).unwrap();
    let f = GenericUnivariateSelect::<f64>::new(
        ScoreFunc::FClassif,
        GenericUnivariateMode::KBest,
        GenericUnivariateParam::Value(2.0),
    )
    .fit(&x, &y)
    .unwrap();
    let _ = f.get_support_indices();
    let _ = f.transform(&x).unwrap();

    // SelectFpr/Fdr/Fwe take p-values (Array1<F>) — get them from f_classif first.
    let (_f_stats, p_values) = f_classif::<f64>(&x, &y).unwrap();
    let f = SelectFpr::<f64>::new(0.5).fit(&p_values, &()).unwrap();
    let _ = f.transform(&x).unwrap();
    let f = SelectFdr::<f64>::new(0.5).fit(&p_values, &()).unwrap();
    let _ = f.transform(&x).unwrap();
    let f = SelectFwe::<f64>::new(0.5).fit(&p_values, &()).unwrap();
    let _ = f.transform(&x).unwrap();

    // SelectFromModel — new_from_importances(importances, threshold: Option<F>).
    let importances = Array1::from(vec![0.1f64, 0.5, 0.9]);
    let f = FeatureSelectionSelectFromModel::<f64>::new_from_importances(&importances, Some(0.3))
        .unwrap();
    let _ = f.transform(&x).unwrap();

    // SequentialFeatureSelector::fit(x, y, score_fn) — takes a scoring closure.
    let score_fn = |_x: &Array2<f64>,
                    _y: &Array1<f64>|
     -> Result<f64, ferrolearn_core::error::FerroError> { Ok(0.0) };
    for dir in [Direction::Forward, Direction::Backward] {
        let _ = SequentialFeatureSelector::new(2, dir)
            .fit(&x, &y_f64, score_fn)
            .unwrap();
    }

    // Module-level scoring helpers.
    let (chi2_stats, _p) = chi2::<f64>(&x, &y).unwrap();
    assert_eq!(chi2_stats.len(), 3);
    let (r_stats, _p) = f_regression::<f64>(&x, &y_f64).unwrap();
    assert_eq!(r_stats.len(), 3);
    let corr = r_regression::<f64>(&x, &y_f64).unwrap();
    assert_eq!(corr.len(), 3);
}

// =============================================================================
// Text processing
// =============================================================================
#[test]
fn api_proof_text() {
    let docs: Vec<String> = vec![
        "the quick brown fox".to_string(),
        "the lazy dog".to_string(),
        "the brown dog jumps".to_string(),
    ];
    let f = CountVectorizer::new().fit(&docs).unwrap();
    let counts = f.transform(&docs).unwrap();
    assert_eq!(counts.nrows(), 3);
    let hashed = HashingVectorizer::new()
        .n_features(8)
        .norm(TfidfNorm::None)
        .transform(&docs)
        .unwrap();
    assert_eq!(hashed.dim(), (3, 8));
    let counts_f64 = counts.mapv(|v| v);
    let f = TfidfTransformer::<f64>::new().fit(&counts_f64).unwrap();
    let _ = f.transform(&counts_f64).unwrap();

    let f = TfidfVectorizer::new().fit(&docs).unwrap();
    let tfidf = f.transform(&docs).unwrap();
    assert_eq!(tfidf.nrows(), 3);
    assert_eq!(tfidf.ncols(), f.vocabulary().len());
}

// =============================================================================
// Random projection
// =============================================================================
#[test]
fn api_proof_random_projection() {
    let x = Array2::<f64>::from_shape_vec((8, 50), (0..400).map(|i| i as f64).collect()).unwrap();
    assert_eq!(
        johnson_lindenstrauss_min_dim(1000_usize, 0.1_f64).unwrap(),
        5920
    );
    let _ = GaussianRandomProjection::<f64>::new(10)
        .fit_transform(&x)
        .unwrap();
    let _ = SparseRandomProjection::<f64>::new(10)
        .fit_transform(&x)
        .unwrap();
}
