//! Wave-7 model-sel conformance: splitters + dummies.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_model_sel::{
    cross_validation::CrossValidator,
    dummy::{DummyClassifier, DummyClassifierStrategy, DummyRegressor, DummyRegressorStrategy},
    group_splitters::{GroupKFold, GroupShuffleSplit, LeaveOneGroupOut},
    splitters::{LeaveOneOut, LeavePOut, ShuffleSplit},
};
use ferrolearn_test_oracle::{json_to_array1, json_to_array2, load_fixture};

fn json_to_usize_vec(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

fn assert_folds_match(
    actual: &[(Vec<usize>, Vec<usize>)],
    expected: &serde_json::Value,
    label: &str,
) {
    let expected_folds = expected.as_array().unwrap();
    assert_eq!(
        actual.len(),
        expected_folds.len(),
        "{label}: fold count mismatch"
    );
    for (i, ((actual_train, actual_test), expected_fold)) in
        actual.iter().zip(expected_folds.iter()).enumerate()
    {
        let mut expected_train = json_to_usize_vec(&expected_fold["train"]);
        let mut expected_test = json_to_usize_vec(&expected_fold["test"]);
        let mut at = actual_train.clone();
        let mut ate = actual_test.clone();
        at.sort();
        ate.sort();
        expected_train.sort();
        expected_test.sort();
        assert_eq!(
            at.as_slice(),
            expected_train.as_slice(),
            "{label} fold {i} train indices"
        );
        assert_eq!(
            ate.as_slice(),
            expected_test.as_slice(),
            "{label} fold {i} test indices"
        );
    }
}

#[test]
fn conformance_leave_one_out() {
    let fx = load_fixture("leave_one_out");
    let n = fx.input["n_samples"].as_u64().unwrap() as usize;
    let folds = LeaveOneOut.fold_indices(n).expect("LOO split");
    assert_folds_match(&folds, &fx.expected["folds"], "LeaveOneOut");
}

#[test]
fn conformance_leave_p_out() {
    let fx = load_fixture("leave_p_out");
    let n = fx.input["n_samples"].as_u64().unwrap() as usize;
    let p = fx.params["p"].as_u64().unwrap() as usize;
    let folds = LeavePOut::new(p).fold_indices(n).expect("LPO split");
    assert_folds_match(&folds, &fx.expected["folds"], "LeavePOut");
}

#[test]
fn conformance_shuffle_split() {
    let fx = load_fixture("shuffle_split");
    let n = fx.input["n_samples"].as_u64().unwrap() as usize;
    let n_splits = fx.params["n_splits"].as_u64().unwrap() as usize;
    let test_size = fx.params["test_size"].as_f64().unwrap();
    let random_state = fx.params["random_state"].as_u64().unwrap();
    let folds = ShuffleSplit::new(n_splits, test_size)
        .random_state(random_state)
        .fold_indices(n)
        .expect("ShuffleSplit split");
    // Shuffling RNG paths differ between ferrolearn and sklearn — only
    // verify split structure (count + sizes) rather than exact indices.
    let expected_folds = fx.expected["folds"].as_array().unwrap();
    assert_eq!(folds.len(), expected_folds.len(), "ShuffleSplit n_splits");
    let total_size = (test_size * n as f64).round() as usize;
    for (i, (_train, test)) in folds.iter().enumerate() {
        assert_eq!(test.len(), total_size, "ShuffleSplit fold {i} test size");
    }
}

#[test]
fn conformance_group_kfold() {
    let fx = load_fixture("group_kfold");
    let groups_vec = json_to_usize_vec(&fx.input["groups"]);
    let groups = ndarray::Array1::from_vec(groups_vec);
    let n_splits = fx.params["n_splits"].as_u64().unwrap() as usize;
    let folds = GroupKFold::new(n_splits)
        .split(&groups)
        .expect("GroupKFold split");
    // GroupKFold assignment depends on group-size ordering — both libraries
    // should produce the same partition up to fold permutation.
    let expected_folds = fx.expected["folds"].as_array().unwrap();
    assert_eq!(folds.len(), expected_folds.len(), "GroupKFold n_splits");
    for (i, (_train, test)) in folds.iter().enumerate() {
        // Each test fold should consist of complete groups — verify by
        // checking that for each test sample, the same group's samples
        // are all in the test set.
        let test_groups: std::collections::HashSet<usize> =
            test.iter().map(|&i| groups[i]).collect();
        for j in 0..groups.len() {
            if test_groups.contains(&groups[j]) {
                assert!(
                    test.contains(&j),
                    "GroupKFold fold {i}: sample {j} of group {} should be in test",
                    groups[j]
                );
            }
        }
    }
}

#[test]
fn conformance_group_shuffle_split() {
    let fx = load_fixture("group_shuffle_split");
    let groups_vec = json_to_usize_vec(&fx.input["groups"]);
    let groups = ndarray::Array1::from_vec(groups_vec);
    let n_splits = fx.params["n_splits"].as_u64().unwrap() as usize;
    let test_size = fx.params["test_size"].as_f64().unwrap();
    let folds = GroupShuffleSplit::new(n_splits, test_size)
        .split(&groups)
        .expect("GSS split");
    let expected_folds = fx.expected["folds"].as_array().unwrap();
    assert_eq!(folds.len(), expected_folds.len(), "GSS n_splits");
    for (_train, test) in folds.iter() {
        let test_groups: std::collections::HashSet<usize> =
            test.iter().map(|&i| groups[i]).collect();
        // Each fold should keep groups intact.
        for j in 0..groups.len() {
            if test_groups.contains(&groups[j]) {
                assert!(
                    test.contains(&j),
                    "GSS: group {} should be wholly in test or wholly in train",
                    groups[j]
                );
            }
        }
    }
}

#[test]
fn conformance_leave_one_group_out() {
    let fx = load_fixture("leave_one_group_out");
    let groups_vec = json_to_usize_vec(&fx.input["groups"]);
    let groups = ndarray::Array1::from_vec(groups_vec);
    let folds = LeaveOneGroupOut::new()
        .split(&groups)
        .expect("LeaveOneGroupOut split");
    let expected_n = fx.expected["n_folds"].as_u64().unwrap() as usize;
    assert_eq!(folds.len(), expected_n, "LOGO n_folds");
    // Each fold's test set should be exactly one group.
    for (_train, test) in folds.iter() {
        let test_groups: std::collections::HashSet<usize> =
            test.iter().map(|&i| groups[i]).collect();
        assert_eq!(
            test_groups.len(),
            1,
            "LeaveOneGroupOut: test set should contain one group"
        );
    }
}

#[test]
fn conformance_dummy_classifier() {
    let fx = load_fixture("dummy_classifier");
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);

    let model = DummyClassifier::new(DummyClassifierStrategy::MostFrequent);
    let fitted = model.fit(&x, &y).expect("DummyClassifier fit");
    let preds = fitted.predict(&x).expect("DummyClassifier predict");
    let expected: Vec<usize> = fx.expected["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    // most_frequent should be exact.
    for (i, (&a, &e)) in preds.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "DummyClassifier[{i}]");
    }
}

#[test]
fn conformance_dummy_regressor() {
    let fx = load_fixture("dummy_regressor");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let constant_expected = fx.expected["constant"].as_f64().unwrap();

    let model = DummyRegressor::<f64>::new(DummyRegressorStrategy::Mean);
    let fitted = model.fit(&x, &y).expect("DummyRegressor fit");
    let preds = fitted.predict(&x).expect("DummyRegressor predict");
    // All predictions should equal the training-y mean, which is the
    // `constant` field.
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            (p - constant_expected).abs() < 1e-9,
            "DummyRegressor[{i}] {p} != mean {constant_expected}"
        );
    }
}
