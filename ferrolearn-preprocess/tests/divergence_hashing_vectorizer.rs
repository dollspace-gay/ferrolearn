//! HashingVectorizer divergence tests against live scikit-learn 1.5.2 oracles.
//!
//! Route: `.design/preprocess/count_vectorizer.md` REQ-13.
//! Scope: dense Rust output for the default word-analyzer path. Sparse CSR,
//! custom analyzers/tokenizers, and the Python ABI remain documented residuals.

use approx::assert_abs_diff_eq;
use ferrolearn_preprocess::{HashingVectorizer, TfidfNorm};
use ndarray::Array2;

fn docs(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

fn assert_matrix_close(actual: &Array2<f64>, expected: &[&[f64]]) {
    assert_eq!(actual.nrows(), expected.len());
    assert_eq!(actual.ncols(), expected.first().map_or(0, |row| row.len()));
    for (i, row) in expected.iter().enumerate() {
        for (j, expected_value) in row.iter().enumerate() {
            assert_abs_diff_eq!(actual[[i, j]], *expected_value, epsilon = 1e-12);
        }
    }
}

#[test]
fn default_l2_signed_hashing_matches_sklearn_oracle() {
    let actual = HashingVectorizer::new()
        .n_features(8)
        .transform(&docs(&["the cat sat"]))
        .unwrap();

    assert_matrix_close(
        &actual,
        &[&[
            0.0,
            0.0,
            0.0,
            0.0,
            0.5773502691896258,
            0.0,
            -0.5773502691896258,
            0.5773502691896258,
        ]],
    );
}

#[test]
fn unsigned_and_signed_count_modes_match_sklearn_oracles() {
    let docs = docs(&["the cat sat"]);
    let unsigned = HashingVectorizer::new()
        .n_features(8)
        .alternate_sign(false)
        .norm(TfidfNorm::None)
        .transform(&docs)
        .unwrap();
    let signed = HashingVectorizer::new()
        .n_features(8)
        .alternate_sign(true)
        .norm(TfidfNorm::None)
        .transform(&docs)
        .unwrap();

    assert_matrix_close(&unsigned, &[&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]);
    assert_matrix_close(&signed, &[&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0]]);
}

#[test]
fn binary_l1_and_lowercase_controls_match_sklearn_oracles() {
    let binary = HashingVectorizer::new()
        .n_features(8)
        .alternate_sign(false)
        .binary(true)
        .norm(TfidfNorm::None)
        .transform(&docs(&["cat cat dog"]))
        .unwrap();
    let l1 = HashingVectorizer::new()
        .n_features(8)
        .alternate_sign(false)
        .norm(TfidfNorm::L1)
        .transform(&docs(&["cat cat dog"]))
        .unwrap();
    let lowercase = HashingVectorizer::new()
        .n_features(8)
        .alternate_sign(false)
        .norm(TfidfNorm::None)
        .transform(&docs(&["Cat cat"]))
        .unwrap();
    let no_lowercase = HashingVectorizer::new()
        .n_features(8)
        .alternate_sign(false)
        .lowercase(false)
        .norm(TfidfNorm::None)
        .transform(&docs(&["Cat cat"]))
        .unwrap();

    assert_matrix_close(&binary, &[&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]]);
    assert_matrix_close(
        &l1,
        &[&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 2.0 / 3.0]],
    );
    assert_matrix_close(&lowercase, &[&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]);
    assert_matrix_close(&no_lowercase, &[&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]);
}

#[test]
fn default_token_pattern_drops_single_character_tokens() {
    let actual = HashingVectorizer::new()
        .n_features(8)
        .transform(&docs(&["a b c"]))
        .unwrap();

    assert_matrix_close(&actual, &[&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
}

#[test]
fn invalid_n_features_is_rejected() {
    let docs = docs(&["cat"]);
    assert!(
        HashingVectorizer::new()
            .n_features(0)
            .transform(&docs)
            .is_err()
    );
}
