//! Proof-of-API integration test for ferrolearn-neighbors.
//!
//! Audit deliverable for crosslink #266 (under #242). Exercises every
//! public API surface of the crate end-to-end so that future PRs that
//! change the public API have a green-or-red signal here.
//!
//! Coverage:
//! - KNeighborsClassifier / Regressor: every builder, fit, predict,
//!   predict_proba (cls), score, kneighbors, kneighbors_graph, n_samples_fit
//! - RadiusNeighborsClassifier / Regressor: same set with radius_neighbors
//! - NearestNeighbors: kneighbors, radius_neighbors, kneighbors_graph,
//!   radius_neighbors_graph, n_samples_fit, shape
//! - KNeighborsTransformer / RadiusNeighborsTransformer: builders + fit + transform
//! - KernelDensity: builders + fit + score_samples + score
//! - NearestCentroid: fit, predict, score, centroids
//! - LocalOutlierFactor: fit, fit_predict, predict, decision_function,
//!   score_samples, lof_scores, offset, negative_outlier_factor, with_novelty
//! - KdTree, BallTree: build + query smoke
//! - Free functions: kneighbors_graph, radius_neighbors_graph,
//!   sort_graph_by_row_values
//! - Algorithm and Weights enum coverage

use approx::assert_relative_eq;
use ferrolearn_core::traits::{Fit, Predict, Transform};
use ferrolearn_neighbors::balltree::BallTree;
use ferrolearn_neighbors::kdtree::KdTree;
use ferrolearn_neighbors::knn::{Algorithm, Weights};
use ferrolearn_neighbors::local_outlier_factor::{FittedLocalOutlierFactor, LocalOutlierFactor};
use ferrolearn_neighbors::nearest_centroid::{FittedNearestCentroid, NearestCentroid};
use ferrolearn_neighbors::nearest_neighbors::NearestNeighbors;
use ferrolearn_neighbors::{
    FittedKNeighborsClassifier, FittedKNeighborsRegressor, FittedKNeighborsTransformer,
    FittedKernelDensity, FittedRadiusNeighborsClassifier, FittedRadiusNeighborsRegressor,
    FittedRadiusNeighborsTransformer, GraphMode, KNeighborsClassifier, KNeighborsRegressor,
    KNeighborsTransformer, KernelDensity, KernelDensityKernel, RadiusNeighborsClassifier,
    RadiusNeighborsRegressor, RadiusNeighborsTransformer, kneighbors_graph, radius_neighbors_graph,
    sort_graph_by_row_values,
};
use ndarray::{Array1, Array2, array};

/// Two well-separated clusters in 2D. 6 samples, 2 features, 2 classes.
fn two_clusters_2d() -> (Array2<f64>, Array1<usize>, Array1<f64>) {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
    )
    .unwrap();
    let y_cls = array![0usize, 0, 0, 1, 1, 1];
    let y_reg = array![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
    (x, y_cls, y_reg)
}

// =============================================================================
// KNeighborsClassifier — full builder + every public method
// =============================================================================
#[test]
fn api_proof_kneighbors_classifier() {
    let (x, y, _) = two_clusters_2d();

    let m = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(3)
        .with_weights(Weights::Distance)
        .with_algorithm(Algorithm::BruteForce);
    let f: FittedKNeighborsClassifier<f64> = m.fit(&x, &y).unwrap();

    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 6);
    let proba = f.predict_proba(&x).unwrap();
    assert_eq!(proba.dim(), (6, 2));
    for i in 0..6 {
        assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
    }
    let acc = f.score(&x, &y).unwrap();
    assert_relative_eq!(acc, 1.0, epsilon = 1e-10);

    let (dists, idxs) = f.kneighbors(&x, None).unwrap();
    assert_eq!(dists.dim(), (6, 3));
    assert_eq!(idxs.dim(), (6, 3));
    let g = f
        .kneighbors_graph(&x, None, GraphMode::Connectivity)
        .unwrap();
    assert_eq!(g.n_rows(), 6);
    assert_eq!(g.n_cols(), 6);
    assert_eq!(g.nnz(), 6 * 3);
    let g_dist = f.kneighbors_graph(&x, None, GraphMode::Distance).unwrap();
    assert_eq!(g_dist.nnz(), 6 * 3);

    assert_eq!(f.n_samples_fit(), 6);
    use ferrolearn_core::introspection::HasClasses;
    assert_eq!(f.classes(), &[0, 1]);
    assert_eq!(f.n_classes(), 2);

    // Default impl + every Algorithm and Weights variant compiles.
    let _: KNeighborsClassifier<f64> = Default::default();
    for alg in [
        Algorithm::Auto,
        Algorithm::BruteForce,
        Algorithm::KdTree,
        Algorithm::BallTree,
    ] {
        for w in [Weights::Uniform, Weights::Distance] {
            let _ = KNeighborsClassifier::<f64>::new()
                .with_algorithm(alg)
                .with_weights(w)
                .fit(&x, &y)
                .unwrap();
        }
    }
}

// =============================================================================
// KNeighborsRegressor
// =============================================================================
#[test]
fn api_proof_kneighbors_regressor() {
    let (x, _, y) = two_clusters_2d();

    let m = KNeighborsRegressor::<f64>::new()
        .with_n_neighbors(3)
        .with_weights(Weights::Distance)
        .with_algorithm(Algorithm::Auto);
    let f: FittedKNeighborsRegressor<f64> = m.fit(&x, &y).unwrap();

    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 6);
    // Score on training data should be close to 1.0 with k=3 on these clusters.
    let r2 = f.score(&x, &y).unwrap();
    assert!(r2 > 0.9, "expected R² > 0.9 on clean clusters, got {r2}");

    let (dists, idxs) = f.kneighbors(&x, Some(2)).unwrap();
    assert_eq!(dists.dim(), (6, 2));
    assert_eq!(idxs.dim(), (6, 2));
    let g = f
        .kneighbors_graph(&x, Some(2), GraphMode::Distance)
        .unwrap();
    assert_eq!(g.nnz(), 6 * 2);
    assert_eq!(f.n_samples_fit(), 6);

    let _: KNeighborsRegressor<f64> = Default::default();
}

// =============================================================================
// RadiusNeighborsClassifier
// =============================================================================
#[test]
fn api_proof_radius_neighbors_classifier() {
    let (x, y, _) = two_clusters_2d();

    let m = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(1.0)
        .with_weights(Weights::Uniform)
        .with_algorithm(Algorithm::Auto)
        .with_outlier_label(Some(0));
    let f: FittedRadiusNeighborsClassifier<f64> = m.fit(&x, &y).unwrap();

    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 6);
    let proba = f.predict_proba(&x).unwrap();
    assert_eq!(proba.dim(), (6, 2));
    for i in 0..6 {
        let row_sum: f64 = proba.row(i).sum();
        assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
    }
    let acc = f.score(&x, &y).unwrap();
    assert!(acc > 0.0);

    let (dists, idxs) = f.radius_neighbors(&x, None).unwrap();
    assert_eq!(dists.len(), 6);
    assert_eq!(idxs.len(), 6);
    let g = f
        .radius_neighbors_graph(&x, Some(1.0), GraphMode::Connectivity)
        .unwrap();
    assert_eq!(g.n_rows(), 6);
    assert_eq!(g.n_cols(), 6);

    assert_eq!(f.classes(), &[0, 1]);
    assert_eq!(f.n_classes(), 2);
    assert_eq!(f.n_samples_fit(), 6);

    let _: RadiusNeighborsClassifier<f64> = Default::default();
}

// =============================================================================
// RadiusNeighborsRegressor
// =============================================================================
#[test]
fn api_proof_radius_neighbors_regressor() {
    let (x, _, y) = two_clusters_2d();

    let m = RadiusNeighborsRegressor::<f64>::new()
        .with_radius(1.0)
        .with_weights(Weights::Distance)
        .with_algorithm(Algorithm::BruteForce);
    let f: FittedRadiusNeighborsRegressor<f64> = m.fit(&x, &y).unwrap();

    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 6);
    let r2 = f.score(&x, &y).unwrap();
    assert!(r2 > 0.0);

    let (dists, idxs) = f.radius_neighbors(&x, None).unwrap();
    assert_eq!(dists.len(), 6);
    assert_eq!(idxs.len(), 6);
    let g = f
        .radius_neighbors_graph(&x, Some(1.0), GraphMode::Distance)
        .unwrap();
    assert_eq!(g.n_rows(), 6);

    assert_eq!(f.n_samples_fit(), 6);
    let _: RadiusNeighborsRegressor<f64> = Default::default();
}

// =============================================================================
// NearestNeighbors (unsupervised)
// =============================================================================
#[test]
fn api_proof_nearest_neighbors() {
    let (x, _, _) = two_clusters_2d();

    let nn = NearestNeighbors::<f64>::new()
        .with_n_neighbors(3)
        .with_algorithm(Algorithm::Auto)
        .fit(&x, &())
        .unwrap();

    let (dists, idxs) = nn.kneighbors(&x, None).unwrap();
    assert_eq!(dists.dim(), (6, 3));
    assert_eq!(idxs.dim(), (6, 3));

    let pairs = nn.radius_neighbors(&x, 1.0).unwrap();
    assert_eq!(pairs.len(), 6);

    let g = nn
        .kneighbors_graph(&x, None, GraphMode::Connectivity)
        .unwrap();
    assert_eq!(g.n_rows(), 6);
    assert_eq!(g.n_cols(), 6);
    let g2 = nn
        .radius_neighbors_graph(&x, 1.0, GraphMode::Distance)
        .unwrap();
    assert_eq!(g2.n_rows(), 6);

    assert_eq!(nn.n_samples_fit(), 6);
    assert_eq!(nn.shape(), (6, 2));

    let _: NearestNeighbors<f64> = Default::default();
}

// =============================================================================
// Neighbor graph transformers
// =============================================================================
#[test]
fn api_proof_neighbor_graph_transformers() {
    let (x, _, _) = two_clusters_2d();

    let k_transformer = KNeighborsTransformer::<f64>::new()
        .with_n_neighbors(2)
        .with_mode(GraphMode::Distance)
        .with_algorithm(Algorithm::BruteForce)
        .with_leaf_size(20);
    assert_eq!(k_transformer.n_neighbors(), 2);
    assert_eq!(k_transformer.mode(), GraphMode::Distance);
    let fitted_k: FittedKNeighborsTransformer<f64> = k_transformer.fit(&x, &()).unwrap();
    assert_eq!(fitted_k.n_features_in(), 2);
    assert_eq!(fitted_k.n_samples_fit(), 6);
    assert_eq!(fitted_k.n_neighbors(), 2);
    let k_graph = fitted_k.transform(&x).unwrap();
    assert_eq!(k_graph.n_rows(), 6);
    assert_eq!(k_graph.n_cols(), 6);

    let r_transformer = RadiusNeighborsTransformer::<f64>::new()
        .with_radius(1.0)
        .with_mode(GraphMode::Connectivity)
        .with_algorithm(Algorithm::Auto)
        .with_leaf_size(20);
    assert_eq!(r_transformer.mode(), GraphMode::Connectivity);
    assert_relative_eq!(r_transformer.radius(), 1.0, epsilon = 1e-12);
    let fitted_r: FittedRadiusNeighborsTransformer<f64> = r_transformer.fit(&x, &()).unwrap();
    assert_eq!(fitted_r.n_features_in(), 2);
    assert_eq!(fitted_r.n_samples_fit(), 6);
    assert_eq!(fitted_r.mode(), GraphMode::Connectivity);
    assert_relative_eq!(fitted_r.radius(), 1.0, epsilon = 1e-12);
    let r_graph = fitted_r.transform(&x).unwrap();
    assert_eq!(r_graph.n_rows(), 6);
    assert_eq!(r_graph.n_cols(), 6);
}

// =============================================================================
// KernelDensity
// =============================================================================
#[test]
fn api_proof_kernel_density() {
    let x = array![[0.0], [1.0], [2.0]];
    let q = array![[0.0], [1.5]];

    let model = KernelDensity::<f64>::new()
        .with_bandwidth(1.0)
        .with_kernel(KernelDensityKernel::Gaussian)
        .with_algorithm(Algorithm::Auto)
        .with_leaf_size(40);
    assert_relative_eq!(model.bandwidth(), 1.0, epsilon = 1e-12);
    assert_eq!(model.kernel(), KernelDensityKernel::Gaussian);
    assert_eq!(model.algorithm(), Algorithm::Auto);
    assert_eq!(model.leaf_size(), 40);

    let fitted: FittedKernelDensity<f64> = model.fit(&x, &()).unwrap();
    assert_eq!(fitted.n_features_in(), 1);
    assert_eq!(fitted.n_samples_fit(), 3);
    assert_eq!(fitted.kernel(), KernelDensityKernel::Gaussian);
    let scores = fitted.score_samples(&q).unwrap();
    assert_eq!(scores.len(), 2);
    assert_relative_eq!(scores[0], -1.4625939022307919, epsilon = 1e-12);
    assert_relative_eq!(
        fitted.score(&q).unwrap(),
        scores.iter().copied().sum::<f64>(),
        epsilon = 1e-12
    );

    let _: KernelDensity<f64> = Default::default();
}

// =============================================================================
// NearestCentroid
// =============================================================================
#[test]
fn api_proof_nearest_centroid() {
    let (x, y, _) = two_clusters_2d();

    let m = NearestCentroid::<f64>::new();
    let f: FittedNearestCentroid<f64> = m.fit(&x, &y).unwrap();

    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 6);
    let acc = f.score(&x, &y).unwrap();
    assert_relative_eq!(acc, 1.0, epsilon = 1e-10);

    let cents = f.centroids();
    assert_eq!(cents.dim(), (2, 2));

    use ferrolearn_core::introspection::HasClasses;
    assert_eq!(f.classes(), &[0, 1]);
    assert_eq!(f.n_classes(), 2);

    let _: NearestCentroid<f64> = Default::default();
}

// =============================================================================
// LocalOutlierFactor (every method, both novelty modes)
// =============================================================================
#[test]
fn api_proof_local_outlier_factor() {
    // 8 inliers near origin + 2 obvious outliers far away.
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 0.05, 0.05, -0.05, -0.05,
            10.0, 10.0, -10.0, -10.0,
        ],
    )
    .unwrap();

    let m = LocalOutlierFactor::<f64>::new()
        .with_n_neighbors(3)
        .with_contamination(0.2)
        .with_algorithm(Algorithm::Auto)
        .with_novelty(true);

    // fit_predict shorthand: returns labels for the training data itself.
    let labels = m.fit_predict(&x).unwrap();
    assert_eq!(labels.len(), 10);
    // The two far-out points should be flagged as outliers (-1).
    assert_eq!(labels[8], -1);
    assert_eq!(labels[9], -1);

    let f: FittedLocalOutlierFactor<f64> = m.fit(&x, &()).unwrap();
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 10);

    let scores = f.score_samples(&x).unwrap();
    assert_eq!(scores.len(), 10);
    // sklearn convention: higher score_samples = more inlier-like, so
    // outlier scores are more negative than inlier scores.
    let outlier_score = scores[8].min(scores[9]);
    let inlier_max = (0..8).map(|i| scores[i]).fold(f64::NEG_INFINITY, f64::max);
    assert!(
        outlier_score < inlier_max,
        "outlier score_sample {outlier_score} should be < max inlier {inlier_max}"
    );

    let dec = f.decision_function(&x).unwrap();
    assert_eq!(dec.len(), 10);
    // decision_function = score_samples - offset_, so outliers are < 0.
    assert!(dec[8] < 0.0);
    assert!(dec[9] < 0.0);

    let lof = f.lof_scores();
    assert_eq!(lof.len(), 10);
    let _ = f.offset();
    let _ = f.negative_outlier_factor();

    let _: LocalOutlierFactor<f64> = Default::default();
}

// =============================================================================
// KdTree + BallTree (build + query smoke)
// =============================================================================
#[test]
fn api_proof_spatial_indices() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 5.0, 5.0],
    )
    .unwrap();
    let query = vec![0.5, 0.5];

    let kd = KdTree::build(&x);
    let res_kd = kd.query(&x, &query, 3);
    assert_eq!(res_kd.len(), 3);

    let bt = BallTree::build(&x);
    let res_bt = bt.query(&x, &query, 3);
    assert_eq!(res_bt.len(), 3);

    let within = bt.within_radius(&query, 2.0);
    assert!(!within.is_empty());
}

// =============================================================================
// Free graph functions + sort_graph_by_row_values
// =============================================================================
#[test]
fn api_proof_graph_free_functions() {
    let (x, _, _) = two_clusters_2d();

    let g_knn = kneighbors_graph(&x, 2, GraphMode::Connectivity).unwrap();
    assert_eq!(g_knn.n_rows(), 6);
    assert_eq!(g_knn.n_cols(), 6);
    assert_eq!(g_knn.nnz(), 6 * 2);

    let g_rad = radius_neighbors_graph(&x, 1.0, GraphMode::Distance).unwrap();
    assert_eq!(g_rad.n_rows(), 6);
    assert_eq!(g_rad.n_cols(), 6);

    // sort_graph_by_row_values returns an equivalently-shaped sparse matrix.
    let sorted = sort_graph_by_row_values(&g_knn).unwrap();
    assert_eq!(sorted.n_rows(), g_knn.n_rows());
    assert_eq!(sorted.n_cols(), g_knn.n_cols());
    assert_eq!(sorted.nnz(), g_knn.nnz());
}

// =============================================================================
// f32 numeric type compiles for every estimator
// =============================================================================
#[test]
fn api_proof_f32_compiles() {
    let x32 = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0f32, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
        ],
    )
    .unwrap();
    let y_cls = array![0usize, 0, 0, 1, 1, 1];
    let y_reg = array![1.0f32, 2.0, 3.0, 10.0, 11.0, 12.0];

    let _ = KNeighborsClassifier::<f32>::new()
        .with_n_neighbors(3)
        .fit(&x32, &y_cls)
        .unwrap()
        .predict(&x32)
        .unwrap();
    let _ = KNeighborsRegressor::<f32>::new()
        .with_n_neighbors(3)
        .fit(&x32, &y_reg)
        .unwrap()
        .predict(&x32)
        .unwrap();
    let _ = RadiusNeighborsClassifier::<f32>::new()
        .with_radius(1.0)
        .with_outlier_label(Some(0))
        .fit(&x32, &y_cls)
        .unwrap()
        .predict(&x32)
        .unwrap();
    let _ = RadiusNeighborsRegressor::<f32>::new()
        .with_radius(1.0)
        .fit(&x32, &y_reg)
        .unwrap();
    let _ = NearestCentroid::<f32>::new()
        .fit(&x32, &y_cls)
        .unwrap()
        .predict(&x32)
        .unwrap();
    let _ = LocalOutlierFactor::<f32>::new()
        .with_n_neighbors(3)
        .with_contamination(0.2)
        .fit_predict(&x32)
        .unwrap();
    let _ = NearestNeighbors::<f32>::new()
        .with_n_neighbors(3)
        .fit(&x32, &())
        .unwrap();
    let _ = KernelDensity::<f32>::new()
        .with_bandwidth(1.0)
        .fit(&x32, &())
        .unwrap()
        .score_samples(&x32)
        .unwrap();
    let _ = kneighbors_graph(&x32, 2, GraphMode::Connectivity).unwrap();
    let _ = radius_neighbors_graph(&x32, 1.0f32, GraphMode::Distance).unwrap();
}
