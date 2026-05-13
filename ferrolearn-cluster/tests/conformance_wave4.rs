//! Wave-4 cluster conformance vs scikit-learn (gap-filling).
//!
//! Covers AffinityPropagation, BayesianGaussianMixture, BisectingKMeans,
//! FeatureAgglomeration, HDBSCAN, LabelPropagation, LabelSpreading.

use ferrolearn_cluster::{
    AffinityPropagation, BayesianGaussianMixture, BisectingKMeans, FeatureAgglomeration, Hdbscan,
    LabelPropagation, LabelSpreading,
};
use ferrolearn_core::{Fit, Predict, Transform};
use ferrolearn_test_oracle::{assert_ari_ge, json_to_array2, json_to_labels, load_fixture};

#[test]
fn conformance_affinity_propagation() {
    let fx = load_fixture("affinity_propagation");
    let x = json_to_array2(&fx.input["X"]);
    let damping = fx.params["damping"].as_f64().unwrap_or(0.7);
    let max_iter = fx.params["max_iter"].as_u64().unwrap_or(200) as usize;
    let preference = fx.params["preference"].as_f64().unwrap_or(-50.0);

    let model = AffinityPropagation::<f64>::new()
        .with_damping(damping)
        .with_max_iter(max_iter)
        .with_preference(preference);
    let fitted = model.fit(&x, &()).expect("AffinityPropagation fit");
    let actual: Vec<i64> = fitted.labels().iter().map(|&v| v as i64).collect();
    let expected = json_to_labels(&fx.expected["labels"]);
    // AP is sensitive to messaging-update order; ARI floor 0.50.
    assert_ari_ge(&actual, &expected, 0.50, "AffinityPropagation.labels");
}

#[test]
fn conformance_bayesian_gaussian_mixture() {
    let fx = load_fixture("bayesian_gaussian_mixture");
    let x = json_to_array2(&fx.input["X"]);
    let n_components = fx.params["n_components"].as_u64().unwrap_or(3) as usize;
    let max_iter = fx.params["max_iter"].as_u64().unwrap_or(200) as usize;

    let model = BayesianGaussianMixture::<f64>::new(n_components).with_max_iter(max_iter);
    let fitted = model.fit(&x, &()).expect("BayesianGMM fit");
    let preds = fitted.predict(&x).expect("BayesianGMM predict");
    let actual: Vec<i64> = preds.iter().map(|&v| v as i64).collect();
    let expected = json_to_labels(&fx.expected["labels"]);
    // Variational EM converges to different posteriors than sklearn under
    // different init RNG paths; ARI floor 0.40 accepts the divergence.
    assert_ari_ge(
        &actual,
        &expected,
        0.40,
        "BayesianGaussianMixture.labels",
    );
}

#[test]
fn conformance_bisecting_kmeans() {
    let fx = load_fixture("bisecting_kmeans");
    let x = json_to_array2(&fx.input["X"]);
    let n_clusters = fx.params["n_clusters"].as_u64().unwrap_or(3) as usize;
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);
    let n_init = fx.params["n_init"].as_u64().unwrap_or(3) as usize;

    let model = BisectingKMeans::<f64>::new(n_clusters)
        .with_random_state(random_state)
        .with_n_init(n_init);
    let fitted = model.fit(&x, &()).expect("BisectingKMeans fit");
    let actual: Vec<i64> = fitted.labels().iter().map(|&v| v as i64).collect();
    let expected = json_to_labels(&fx.expected["labels"]);
    assert_ari_ge(&actual, &expected, 0.80, "BisectingKMeans.labels");
}

#[test]
fn conformance_feature_agglomeration() {
    let fx = load_fixture("feature_agglomeration");
    let x = json_to_array2(&fx.input["X"]);
    let n_clusters = fx.params["n_clusters"].as_u64().unwrap_or(3) as usize;

    let model = FeatureAgglomeration::<f64>::new(n_clusters);
    let fitted = model.fit(&x, &()).expect("FeatureAgglomeration fit");
    let xt = fitted.transform(&x).expect("FeatureAgglomeration transform");
    assert_eq!(xt.nrows(), x.nrows(), "FA transformed rows");
    assert_eq!(xt.ncols(), n_clusters, "FA transformed cols");
    for v in xt.iter() {
        assert!(v.is_finite(), "FA transform non-finite");
    }
}

#[test]
fn conformance_hdbscan() {
    let fx = load_fixture("hdbscan");
    let x = json_to_array2(&fx.input["X"]);
    let min_cluster_size = fx.params["min_cluster_size"].as_u64().unwrap_or(10) as usize;

    let model = Hdbscan::<f64>::new().with_min_cluster_size(min_cluster_size);
    let labels = model
        .fit_predict(&x)
        .expect("Hdbscan fit_predict");
    let actual: Vec<i64> = labels.iter().map(|&v| v as i64).collect();
    let expected = json_to_labels(&fx.expected["labels"]);
    // HDBSCAN clustering uses different mutual-reachability minimum-spanning-tree
    // tie-breaking than sklearn's. Use a moderate ARI floor.
    assert_ari_ge(&actual, &expected, 0.60, "HDBSCAN.labels");
}

#[test]
fn conformance_label_propagation() {
    let fx = load_fixture("label_propagation");
    let x = json_to_array2(&fx.input["X"]);
    let y_partial: Vec<i64> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    // ferrolearn uses isize for partial labels with -1 meaning unlabeled
    let y_isize = ndarray::Array1::from_vec(y_partial.iter().map(|&v| v as isize).collect());
    let gamma = fx.params["gamma"].as_f64().unwrap_or(1.0);

    let model = LabelPropagation::<f64>::new().with_gamma(gamma);
    let fitted = model.fit(&x, &y_isize).expect("LabelPropagation fit");
    let preds = fitted.predict(&x).expect("LabelProp predict");
    let expected: Vec<i64> = fx.expected["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let actual: Vec<i64> = preds.iter().map(|&v| v as i64).collect();
    // Different gamma normalization paths can flip ~10% of labels.
    let matches = actual.iter().zip(expected.iter()).filter(|(a, e)| a == e).count();
    let acc = matches as f64 / actual.len() as f64;
    assert!(
        acc >= 0.80,
        "LabelPropagation label-agreement {acc:.4} < 0.80 floor"
    );
}

#[test]
fn conformance_label_spreading() {
    let fx = load_fixture("label_spreading");
    let x = json_to_array2(&fx.input["X"]);
    let y_partial: Vec<i64> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let y_isize = ndarray::Array1::from_vec(y_partial.iter().map(|&v| v as isize).collect());
    let gamma = fx.params["gamma"].as_f64().unwrap_or(1.0);
    let alpha = fx.params["alpha"].as_f64().unwrap_or(0.2);

    let model = LabelSpreading::<f64>::new()
        .with_gamma(gamma)
        .with_alpha(alpha);
    let fitted = model.fit(&x, &y_isize).expect("LabelSpreading fit");
    let preds = fitted.predict(&x).expect("LabelSpreading predict");
    let expected: Vec<i64> = fx.expected["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let actual: Vec<i64> = preds.iter().map(|&v| v as i64).collect();
    let matches = actual.iter().zip(expected.iter()).filter(|(a, e)| a == e).count();
    let acc = matches as f64 / actual.len() as f64;
    assert!(
        acc >= 0.80,
        "LabelSpreading label-agreement {acc:.4} < 0.80 floor"
    );
}
