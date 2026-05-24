//! Conformance tests for ferrolearn-model-sel cross-validation splitters
//! vs scikit-learn.
//!
//! KFold / StratifiedKFold / TimeSeriesSplit are fully deterministic when
//! `shuffle=False`, so the split indices must match sklearn exactly.

use ferrolearn_test_oracle::load_fixture;

fn json_to_usize_vec(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .expect("expected JSON array")
        .iter()
        .map(|v| v.as_u64().expect("expected non-negative integer") as usize)
        .collect()
}

#[test]
fn conformance_kfold() {
    let fx = load_fixture("kfold");
    let n_samples = fx.input["n_samples"].as_u64().unwrap() as usize;
    let n_splits = fx.params["n_splits"].as_u64().unwrap() as usize;
    let shuffle = fx.params["shuffle"].as_bool().unwrap_or(false);

    assert!(
        !shuffle,
        "fixture must use shuffle=false for deterministic comparison"
    );

    let kf = ferrolearn_model_sel::KFold::new(n_splits);
    let folds = kf.split(n_samples);

    let expected_folds = fx.expected["folds"].as_array().unwrap();
    assert_eq!(folds.len(), expected_folds.len(), "fold count");

    for (i, (actual, expected)) in folds.iter().zip(expected_folds.iter()).enumerate() {
        let (actual_train, actual_test) = actual;
        let expected_train = json_to_usize_vec(&expected["train"]);
        let expected_test = json_to_usize_vec(&expected["test"]);
        assert_eq!(
            actual_train.as_slice(),
            expected_train.as_slice(),
            "fold {i} train indices"
        );
        assert_eq!(
            actual_test.as_slice(),
            expected_test.as_slice(),
            "fold {i} test indices"
        );
    }
}

#[test]
fn conformance_stratified_kfold() {
    let fx = load_fixture("stratified_kfold");
    let n_splits = fx.params["n_splits"].as_u64().unwrap() as usize;
    let shuffle = fx.params["shuffle"].as_bool().unwrap_or(false);

    assert!(!shuffle, "fixture must use shuffle=false");

    let y_values: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_values);

    let skf = ferrolearn_model_sel::StratifiedKFold::new(n_splits);
    let folds = skf.split(&y).expect("StratifiedKFold.split");

    let expected_folds = fx.expected["folds"].as_array().unwrap();
    assert_eq!(folds.len(), expected_folds.len(), "fold count");

    for (i, (actual, expected)) in folds.iter().zip(expected_folds.iter()).enumerate() {
        let (actual_train, actual_test) = actual;
        let expected_train = json_to_usize_vec(&expected["train"]);
        let expected_test = json_to_usize_vec(&expected["test"]);
        assert_eq!(
            actual_train.as_slice(),
            expected_train.as_slice(),
            "fold {i} train indices"
        );
        assert_eq!(
            actual_test.as_slice(),
            expected_test.as_slice(),
            "fold {i} test indices"
        );
    }
}

#[test]
fn conformance_time_series_split() {
    let fx = load_fixture("time_series_split");
    let n_samples = fx.input["n_samples"].as_u64().unwrap() as usize;
    let n_splits = fx.params["n_splits"].as_u64().unwrap() as usize;

    let tss = ferrolearn_model_sel::TimeSeriesSplit::new(n_splits);
    let folds = tss.split(n_samples).expect("TimeSeriesSplit.split");

    let expected_folds = fx.expected["folds"].as_array().unwrap();
    assert_eq!(folds.len(), expected_folds.len(), "fold count");

    for (i, (actual, expected)) in folds.iter().zip(expected_folds.iter()).enumerate() {
        let (actual_train, actual_test) = actual;
        let expected_train = json_to_usize_vec(&expected["train"]);
        let expected_test = json_to_usize_vec(&expected["test"]);
        assert_eq!(
            actual_train.as_slice(),
            expected_train.as_slice(),
            "fold {i} train indices"
        );
        assert_eq!(
            actual_test.as_slice(),
            expected_test.as_slice(),
            "fold {i} test indices"
        );
    }
}
