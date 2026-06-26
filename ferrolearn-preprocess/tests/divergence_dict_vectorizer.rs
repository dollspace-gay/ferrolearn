//! DictVectorizer divergence tests against live scikit-learn 1.5.2 oracles.
//!
//! Route: `.design/preprocess/dict_vectorizer.md`.
//! Scope: dense Rust output for numeric, string categorical, and iterable
//! string categorical values. Sparse CSR, dtype, `restrict`, Python metadata
//! routing, and arbitrary Python object keys remain residuals.

use approx::assert_abs_diff_eq;
use ferrolearn_preprocess::{DictValue, DictVectorizer};
use ndarray::Array2;
use std::collections::HashMap;

fn sample(items: &[(&str, DictValue)]) -> HashMap<String, DictValue> {
    items
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

fn assert_matrix_close(actual: &Array2<f64>, expected: &[&[f64]]) {
    assert_eq!(actual.nrows(), expected.len());
    assert_eq!(actual.ncols(), expected.first().map_or(0, |row| row.len()));
    for (i, row) in expected.iter().enumerate() {
        for (j, expected_value) in row.iter().enumerate() {
            if expected_value.is_nan() {
                assert!(actual[[i, j]].is_nan());
            } else {
                assert_abs_diff_eq!(actual[[i, j]], *expected_value, epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn numeric_mappings_match_sklearn_oracle() {
    let samples = [
        sample(&[("foo", 1.0.into()), ("bar", 2.0.into())]),
        sample(&[("foo", 3.0.into()), ("baz", 1.0.into())]),
    ];
    let (fitted, actual) = DictVectorizer::new().fit_transform(&samples).unwrap();

    assert_eq!(fitted.get_feature_names_out(), ["bar", "baz", "foo"]);
    assert_matrix_close(&actual, &[&[2.0, 0.0, 1.0], &[0.0, 1.0, 3.0]]);
    assert_eq!(fitted.vocabulary_map()["bar"], 0);
    assert_eq!(fitted.vocabulary_map()["baz"], 1);
    assert_eq!(fitted.vocabulary_map()["foo"], 2);
}

#[test]
fn string_categorical_values_match_sklearn_oracle() {
    let samples = [
        sample(&[("city", "Dubai".into()), ("temperature", 33.0.into())]),
        sample(&[("city", "London".into()), ("temperature", 12.0.into())]),
        sample(&[
            ("city", "San Francisco".into()),
            ("temperature", 18.0.into()),
        ]),
    ];
    let (fitted, actual) = DictVectorizer::new().fit_transform(&samples).unwrap();

    assert_eq!(
        fitted.get_feature_names_out(),
        [
            "city=Dubai",
            "city=London",
            "city=San Francisco",
            "temperature"
        ]
    );
    assert_matrix_close(
        &actual,
        &[
            &[1.0, 0.0, 0.0, 33.0],
            &[0.0, 1.0, 0.0, 12.0],
            &[0.0, 0.0, 1.0, 18.0],
        ],
    );
}

#[test]
fn custom_separator_matches_sklearn_oracle() {
    let samples = [sample(&[("city", "Dubai".into())])];
    let (fitted, actual) = DictVectorizer::new()
        .separator("::")
        .fit_transform(&samples)
        .unwrap();

    assert_eq!(fitted.get_feature_names_out(), ["city::Dubai"]);
    assert_matrix_close(&actual, &[&[1.0]]);
}

#[test]
fn transform_ignores_unseen_keys_and_categories() {
    let train = [
        sample(&[("city", "Dubai".into()), ("temp", 33.0.into())]),
        sample(&[("city", "London".into()), ("temp", 12.0.into())]),
    ];
    let fitted = DictVectorizer::new().fit(&train).unwrap();
    let test = [
        sample(&[
            ("city", "Paris".into()),
            ("temp", 10.0.into()),
            ("extra", 5.0.into()),
        ]),
        sample(&[("city", "Dubai".into())]),
    ];
    let actual = fitted.transform(&test).unwrap();

    assert_eq!(
        fitted.get_feature_names_out(),
        ["city=Dubai", "city=London", "temp"]
    );
    assert_matrix_close(&actual, &[&[0.0, 0.0, 10.0], &[1.0, 0.0, 0.0]]);
}

#[test]
fn iterable_string_values_are_counted() {
    let samples = [sample(&[(
        "tag",
        DictValue::Texts(vec!["x".to_string(), "y".to_string(), "x".to_string()]),
    )])];
    let (fitted, actual) = DictVectorizer::new().fit_transform(&samples).unwrap();

    assert_eq!(fitted.get_feature_names_out(), ["tag=x", "tag=y"]);
    assert_matrix_close(&actual, &[&[2.0, 1.0]]);
}

#[test]
fn none_values_follow_sklearn_nan_path() {
    let samples = [sample(&[
        ("flag", DictValue::from(1.0)),
        ("no", DictValue::from(0.0)),
        ("none", DictValue::None),
    ])];
    let (fitted, actual) = DictVectorizer::new().fit_transform(&samples).unwrap();

    assert_eq!(fitted.get_feature_names_out(), ["flag", "no", "none"]);
    assert_matrix_close(&actual, &[&[1.0, 0.0, f64::NAN]]);
}

#[test]
fn empty_row_is_allowed_but_empty_corpus_errors() {
    let samples = [HashMap::new()];
    let (fitted, actual) = DictVectorizer::new().fit_transform(&samples).unwrap();

    assert!(fitted.get_feature_names_out().is_empty());
    assert_eq!(actual.dim(), (1, 0));
    assert!(DictVectorizer::new().fit_transform(&[]).is_err());
}

#[test]
fn inverse_transform_returns_non_zero_constructed_feature_names() {
    let samples = [
        sample(&[("foo", 1.0.into()), ("bar", 2.0.into())]),
        sample(&[("foo", 3.0.into()), ("baz", 1.0.into())]),
    ];
    let (fitted, actual) = DictVectorizer::new().fit_transform(&samples).unwrap();
    let inverse = fitted.inverse_transform(&actual).unwrap();

    assert_eq!(inverse[0]["bar"], 2.0);
    assert_eq!(inverse[0]["foo"], 1.0);
    assert!(!inverse[0].contains_key("baz"));
    assert_eq!(inverse[1]["baz"], 1.0);
    assert_eq!(inverse[1]["foo"], 3.0);
    assert!(!inverse[1].contains_key("bar"));
}
