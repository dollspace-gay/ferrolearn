//! FeatureHasher divergence tests against live scikit-learn 1.5.2 oracles.
//!
//! Route: `.design/preprocess/feature_hasher.md`.
//! Scope: dense Rust output for dict, pair, and string input modes. Sparse CSR,
//! dtype, Python metadata routing, and non-numeric dict values remain residuals.

use approx::assert_abs_diff_eq;
use ferrolearn_preprocess::{FeatureHasher, FeatureHasherInputType};
use ndarray::Array2;
use std::collections::HashMap;

fn assert_matrix_close(actual: &Array2<f64>, expected: &[&[f64]]) {
    assert_eq!(actual.nrows(), expected.len());
    assert_eq!(actual.ncols(), expected.first().map_or(0, |row| row.len()));
    for (i, row) in expected.iter().enumerate() {
        for (j, expected_value) in row.iter().enumerate() {
            assert_abs_diff_eq!(actual[[i, j]], *expected_value, epsilon = 1e-12);
        }
    }
}

fn dict_sample(items: &[(&str, f64)]) -> HashMap<String, f64> {
    items
        .iter()
        .map(|(name, value)| ((*name).to_string(), *value))
        .collect()
}

#[test]
fn dict_input_unsigned_and_signed_modes_match_sklearn_oracles() {
    let samples = [dict_sample(&[("cat", 2.0), ("dog", 1.0)])];
    let unsigned = FeatureHasher::new()
        .n_features(8)
        .alternate_sign(false)
        .transform_dict(&samples)
        .unwrap();
    let signed = FeatureHasher::new()
        .n_features(8)
        .alternate_sign(true)
        .transform_dict(&samples)
        .unwrap();

    assert_matrix_close(&unsigned, &[&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0]]);
    assert_matrix_close(&signed, &[&[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 2.0]]);
}

#[test]
fn pair_input_matches_dict_oracle() {
    let samples = [vec![("cat".to_string(), 2.0), ("dog".to_string(), 1.0)]];
    let actual = FeatureHasher::new()
        .n_features(8)
        .input_type(FeatureHasherInputType::Pair)
        .alternate_sign(false)
        .transform_pairs(&samples)
        .unwrap();

    assert_matrix_close(&actual, &[&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0]]);
}

#[test]
fn string_input_uses_implied_unit_counts() {
    let samples = [vec![
        "cat".to_string(),
        "cat".to_string(),
        "dog".to_string(),
    ]];
    let actual = FeatureHasher::new()
        .n_features(8)
        .input_type(FeatureHasherInputType::String)
        .alternate_sign(false)
        .transform_strings(&samples)
        .unwrap();

    assert_matrix_close(&actual, &[&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0]]);
}

#[test]
fn string_input_signed_mode_matches_hashing_vectorizer_oracle_tokens() {
    let samples = [vec![
        "the".to_string(),
        "cat".to_string(),
        "sat".to_string(),
    ]];
    let actual = FeatureHasher::new()
        .n_features(8)
        .input_type(FeatureHasherInputType::String)
        .alternate_sign(true)
        .transform_strings(&samples)
        .unwrap();

    assert_matrix_close(&actual, &[&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0]]);
}

#[test]
fn empty_sample_row_is_allowed_but_empty_corpus_errors() {
    let empty_row = [HashMap::new()];
    let actual = FeatureHasher::new()
        .n_features(8)
        .alternate_sign(false)
        .transform_dict(&empty_row)
        .unwrap();

    assert_matrix_close(&actual, &[&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
    assert!(
        FeatureHasher::new()
            .n_features(8)
            .transform_dict(&[])
            .is_err()
    );
}

#[test]
fn invalid_n_features_and_input_type_are_rejected() {
    let samples = [dict_sample(&[("cat", 1.0)])];
    assert!(
        FeatureHasher::new()
            .n_features(0)
            .transform_dict(&samples)
            .is_err()
    );
    assert!(
        FeatureHasher::new()
            .n_features(i32::MAX as usize + 1)
            .fit()
            .is_err()
    );
    assert!(
        FeatureHasher::new()
            .n_features(8)
            .input_type(FeatureHasherInputType::String)
            .transform_dict(&samples)
            .is_err()
    );
}
