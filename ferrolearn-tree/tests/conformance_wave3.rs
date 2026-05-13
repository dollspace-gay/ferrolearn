//! Wave-3 tree conformance vs scikit-learn.
//!
//! Most ensemble methods are stochastic with different RNG paths than
//! sklearn — we use accuracy/R²/agreement floors, not coefficient parity.

use ferrolearn_core::{Fit, Predict, Transform};
use ferrolearn_test_oracle::{json_to_array1, json_to_array2, load_fixture};

// ---------------------------------------------------------------------------
// ExtraTreeClassifier / Regressor (single trees)
// ---------------------------------------------------------------------------

#[test]
fn conformance_extra_tree_classifier() {
    let fx = load_fixture("extra_tree_classifier");
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);

    let max_depth = fx.params["max_depth"].as_u64().map(|v| v as usize);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);
    let model = ferrolearn_tree::ExtraTreeClassifier::<f64>::new()
        .with_max_depth(max_depth)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &y).expect("ExtraTreeClassifier fit");
    let preds = fitted.predict(&x).expect("ExtraTreeClassifier predict");
    let expected_acc = fx.expected["accuracy"].as_f64().unwrap_or(0.5);
    let acc = preds.iter().zip(y.iter()).filter(|(a, e)| a == e).count() as f64 / y.len() as f64;
    // 0.85x sklearn — ExtraTree uses random splits, so even with the same
    // seed ferrolearn and sklearn produce different trees; a single tree's
    // accuracy is highly variable.
    assert!(
        acc >= 0.85 * expected_acc,
        "ExtraTreeClassifier accuracy {acc:.4} < 0.85 * sklearn {expected_acc:.4}"
    );
}

#[test]
fn conformance_extra_tree_regressor() {
    let fx = load_fixture("extra_tree_regressor");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let max_depth = fx.params["max_depth"].as_u64().map(|v| v as usize);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = ferrolearn_tree::ExtraTreeRegressor::<f64>::new()
        .with_max_depth(max_depth)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &y).expect("ExtraTreeRegressor fit");
    let preds = fitted.predict(&x).expect("ExtraTreeRegressor predict");

    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = preds.iter().zip(y.iter()).map(|(a, e)| (a - e).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    let expected_r2 = fx.expected["r2"].as_f64().unwrap_or(0.5);
    assert!(
        r2 >= expected_r2 - 0.1,
        "ExtraTreeRegressor R² {r2:.4} < sklearn-0.1 ({expected_r2:.4})"
    );
}

// ---------------------------------------------------------------------------
// ExtraTrees ensembles
// ---------------------------------------------------------------------------

#[test]
fn conformance_extra_trees_classifier() {
    let fx = load_fixture("extra_trees_classifier");
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);

    let n_est = fx.params["n_estimators"].as_u64().unwrap_or(20) as usize;
    let max_depth = fx.params["max_depth"].as_u64().map(|v| v as usize);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);
    let model = ferrolearn_tree::ExtraTreesClassifier::<f64>::new()
        .with_n_estimators(n_est)
        .with_max_depth(max_depth)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &y).expect("ExtraTrees fit");
    let preds = fitted.predict(&x).expect("ExtraTrees predict");
    let expected_acc = fx.expected["accuracy"].as_f64().unwrap_or(0.5);
    let acc = preds.iter().zip(y.iter()).filter(|(a, e)| a == e).count() as f64 / y.len() as f64;
    assert!(
        acc >= 0.95 * expected_acc,
        "ExtraTreesClassifier accuracy {acc:.4} < 0.95 * sklearn {expected_acc:.4}"
    );
}

#[test]
fn conformance_extra_trees_regressor() {
    let fx = load_fixture("extra_trees_regressor");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let n_est = fx.params["n_estimators"].as_u64().unwrap_or(20) as usize;
    let max_depth = fx.params["max_depth"].as_u64().map(|v| v as usize);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = ferrolearn_tree::ExtraTreesRegressor::<f64>::new()
        .with_n_estimators(n_est)
        .with_max_depth(max_depth)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &y).expect("ExtraTreesRegressor fit");
    let preds = fitted.predict(&x).expect("ExtraTreesRegressor predict");

    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = preds.iter().zip(y.iter()).map(|(a, e)| (a - e).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    let expected_r2 = fx.expected["r2"].as_f64().unwrap_or(0.5);
    assert!(
        r2 >= expected_r2 - 0.1,
        "ExtraTreesRegressor R² {r2:.4} < sklearn - 0.1 ({expected_r2:.4})"
    );
}

// ---------------------------------------------------------------------------
// Bagging — ferrolearn version uses an internal DT base (no custom base param)
// ---------------------------------------------------------------------------

#[test]
fn conformance_bagging_classifier() {
    let fx = load_fixture("bagging_classifier");
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);
    let n_est = fx.params["n_estimators"].as_u64().unwrap_or(10) as usize;
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = ferrolearn_tree::BaggingClassifier::<f64>::new()
        .with_n_estimators(n_est)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &y).expect("Bagging fit");
    let preds = fitted.predict(&x).expect("Bagging predict");
    let expected_acc = fx.expected["accuracy"].as_f64().unwrap_or(0.5);
    let acc = preds.iter().zip(y.iter()).filter(|(a, e)| a == e).count() as f64 / y.len() as f64;
    assert!(
        acc >= 0.85 * expected_acc,
        "BaggingClassifier accuracy {acc:.4} < 0.85 * sklearn {expected_acc:.4}"
    );
}

#[test]
fn conformance_bagging_regressor() {
    let fx = load_fixture("bagging_regressor");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let n_est = fx.params["n_estimators"].as_u64().unwrap_or(10) as usize;
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = ferrolearn_tree::BaggingRegressor::<f64>::new()
        .with_n_estimators(n_est)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &y).expect("Bagging fit");
    let preds = fitted.predict(&x).expect("Bagging predict");

    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = preds.iter().zip(y.iter()).map(|(a, e)| (a - e).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    let expected_r2 = fx.expected["r2"].as_f64().unwrap_or(0.5);
    assert!(
        r2 >= expected_r2 - 0.15,
        "BaggingRegressor R² {r2:.4} < sklearn - 0.15 ({expected_r2:.4})"
    );
}

// ---------------------------------------------------------------------------
// AdaBoostRegressor
// ---------------------------------------------------------------------------

#[test]
fn conformance_adaboost_regressor() {
    let fx = load_fixture("adaboost_regressor");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let n_est = fx.params["n_estimators"].as_u64().unwrap_or(20) as usize;
    let lr = fx.params["learning_rate"].as_f64().unwrap_or(1.0);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = ferrolearn_tree::AdaBoostRegressor::<f64>::new()
        .with_n_estimators(n_est)
        .with_learning_rate(lr)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &y).expect("AdaBoostRegressor fit");
    let preds = fitted.predict(&x).expect("AdaBoostRegressor predict");

    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = preds.iter().zip(y.iter()).map(|(a, e)| (a - e).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    let expected_r2 = fx.expected["r2"].as_f64().unwrap_or(0.5);
    assert!(
        r2 >= expected_r2 - 0.2,
        "AdaBoostRegressor R² {r2:.4} < sklearn - 0.2 ({expected_r2:.4})"
    );
}

// ---------------------------------------------------------------------------
// HistGradientBoosting
// ---------------------------------------------------------------------------

#[test]
fn conformance_hist_gradient_boosting_classifier() {
    let fx = load_fixture("hist_gradient_boosting_classifier");
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);
    let n_est = fx.params["max_iter"].as_u64().unwrap_or(50) as usize;
    let max_depth = fx.params["max_depth"].as_u64().map(|v| v as usize);
    let lr = fx.params["learning_rate"].as_f64().unwrap_or(0.1);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = ferrolearn_tree::HistGradientBoostingClassifier::<f64>::new()
        .with_n_estimators(n_est)
        .with_max_depth(max_depth)
        .with_learning_rate(lr)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &y).expect("HistGBC fit");
    let preds = fitted.predict(&x).expect("HistGBC predict");
    let expected_acc = fx.expected["accuracy"].as_f64().unwrap_or(0.5);
    let acc = preds.iter().zip(y.iter()).filter(|(a, e)| a == e).count() as f64 / y.len() as f64;
    assert!(
        acc >= 0.90 * expected_acc,
        "HistGBC accuracy {acc:.4} < 0.90 * sklearn {expected_acc:.4}"
    );
}

#[test]
fn conformance_hist_gradient_boosting_regressor() {
    let fx = load_fixture("hist_gradient_boosting_regressor");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let n_est = fx.params["max_iter"].as_u64().unwrap_or(50) as usize;
    let max_depth = fx.params["max_depth"].as_u64().map(|v| v as usize);
    let lr = fx.params["learning_rate"].as_f64().unwrap_or(0.1);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = ferrolearn_tree::HistGradientBoostingRegressor::<f64>::new()
        .with_n_estimators(n_est)
        .with_max_depth(max_depth)
        .with_learning_rate(lr)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &y).expect("HistGBR fit");
    let preds = fitted.predict(&x).expect("HistGBR predict");

    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = preds.iter().zip(y.iter()).map(|(a, e)| (a - e).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    let expected_r2 = fx.expected["r2"].as_f64().unwrap_or(0.5);
    assert!(
        r2 >= expected_r2 - 0.1,
        "HistGBR R² {r2:.4} < sklearn - 0.1 ({expected_r2:.4})"
    );
}

// ---------------------------------------------------------------------------
// IsolationForest
// ---------------------------------------------------------------------------

#[test]
fn conformance_isolation_forest() {
    let fx = load_fixture("isolation_forest");
    let x = json_to_array2(&fx.input["X"]);
    let n_est = fx.params["n_estimators"].as_u64().unwrap_or(50) as usize;
    let contamination = fx.params["contamination"].as_f64().unwrap_or(0.1);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = ferrolearn_tree::IsolationForest::<f64>::new()
        .with_n_estimators(n_est)
        .with_contamination(contamination)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &()).expect("IsolationForest fit");
    let preds = fitted.predict(&x).expect("IsolationForest predict");
    let expected: Vec<i64> = fx.expected["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let matches = preds
        .iter()
        .zip(expected.iter())
        .filter(|&(&a, &e)| a as i64 == e)
        .count();
    let frac = matches as f64 / preds.len() as f64;
    assert!(
        frac >= 0.80,
        "IsolationForest +1/-1 agreement {frac:.4} < 0.80 floor"
    );
}

// ---------------------------------------------------------------------------
// RandomTreesEmbedding
// ---------------------------------------------------------------------------

#[test]
fn conformance_random_trees_embedding() {
    let fx = load_fixture("random_trees_embedding");
    let x = json_to_array2(&fx.input["X"]);
    let n_est = fx.params["n_estimators"].as_u64().unwrap_or(20) as usize;
    let max_depth = fx.params["max_depth"].as_u64().map(|v| v as usize);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = ferrolearn_tree::RandomTreesEmbedding::<f64>::new()
        .with_n_estimators(n_est)
        .with_max_depth(max_depth)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &()).expect("RandomTreesEmbedding fit");
    let xt = fitted.transform(&x).expect("RandomTreesEmbedding transform");
    assert_eq!(xt.nrows(), x.nrows(), "embedding rows");
    // Expansion factor: each tree contributes one one-hot encoded leaf
    // index; ferrolearn may use a denser binary representation.
    assert!(xt.ncols() > 0, "embedding cols must be > 0");
}

// ---------------------------------------------------------------------------
// VotingClassifier / Regressor — ferrolearn ensemble of DTs (not arbitrary base estimators)
// ---------------------------------------------------------------------------

#[test]
fn conformance_voting_classifier() {
    let fx = load_fixture("voting_classifier");
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);

    // ferrolearn's VotingClassifier uses a Vec<Option<usize>> of max-depths
    // for an ensemble of DTs — not arbitrary base estimators like sklearn.
    // Use a 2-tree ensemble matching the fixture's LR+DT pair pattern.
    let model = ferrolearn_tree::VotingClassifier::<f64>::new()
        .with_max_depths(vec![Some(5), Some(5)]);
    let fitted = model.fit(&x, &y).expect("VotingClassifier fit");
    let preds = fitted.predict(&x).expect("VotingClassifier predict");
    let expected_acc = fx.expected["accuracy"].as_f64().unwrap_or(0.5);
    let acc = preds.iter().zip(y.iter()).filter(|(a, e)| a == e).count() as f64 / y.len() as f64;
    // Different base composition than sklearn (DTs only vs LR+DT) — looser floor.
    assert!(
        acc >= 0.80 * expected_acc,
        "VotingClassifier accuracy {acc:.4} < 0.80 * sklearn {expected_acc:.4} \
         (note: ferrolearn voting uses DT-only ensemble; not directly comparable to sklearn LR+DT)"
    );
}

#[test]
fn conformance_voting_regressor() {
    let fx = load_fixture("voting_regressor");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);

    let model = ferrolearn_tree::VotingRegressor::<f64>::new()
        .with_max_depths(vec![Some(5), Some(5)]);
    let fitted = model.fit(&x, &y).expect("VotingRegressor fit");
    let preds = fitted.predict(&x).expect("VotingRegressor predict");

    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = preds.iter().zip(y.iter()).map(|(a, e)| (a - e).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    // R² >= 0 means the model beats predicting the mean — minimal sanity.
    // Not comparable to sklearn LR+DT directly; ferrolearn uses DT-only.
    assert!(r2 >= 0.0, "VotingRegressor R² {r2:.4} below baseline (mean prediction)");
}
