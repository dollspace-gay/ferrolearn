//! Divergence audit: `TfidfVectorizer` vs scikit-learn 1.5.2
//! `sklearn.feature_extraction.text.TfidfVectorizer`.
//!
//! Scope: dense f64 output over the CountVectorizer/TfidfTransformer options
//! implemented by ferrolearn. Sparse CSR, analyzer/tokenizer hooks, n-grams,
//! stop words, dtype, fixed vocabulary, and PyO3 remain tracked gaps.
//!
//! Oracle reproduction (sklearn 1.5.2, run from /tmp):
//! ```text
//! from sklearn.feature_extraction.text import TfidfVectorizer
//! docs = ["cat cat dog", "dog fish", "cat bird"]
//! for kwargs in [{}, {"sublinear_tf": True, "norm": None},
//!                {"use_idf": False}, {"max_features": 2}]:
//!     v = TfidfVectorizer(**kwargs)
//!     print(v.fit_transform(docs).toarray().tolist())
//!     print(v.get_feature_names_out().tolist())
//!     print(getattr(v, "idf_", None).tolist() if hasattr(v, "idf_") else None)
//!     print(v.transform(["cat dog bird", "fish unknown"]).toarray().tolist())
//! ```

use ferrolearn_preprocess::{TfidfNorm, TfidfVectorizer};
use ndarray::{Array2, array};

const TOL: f64 = 1e-12;

fn docs() -> Vec<String> {
    ["cat cat dog", "dog fish", "cat bird"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn transform_docs() -> Vec<String> {
    ["cat dog bird", "fish unknown"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            let diff = (actual[[i, j]] - expected[[i, j]]).abs();
            assert!(
                diff <= TOL,
                "entry ({i},{j}): expected {}, got {}, diff {diff} > {TOL}",
                expected[[i, j]],
                actual[[i, j]]
            );
        }
    }
}

fn assert_idf_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "idf length mismatch");
    for (j, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "idf[{j}]: expected {want}, got {got}"
        );
    }
}

fn assert_vocabulary(actual: &[String], expected: &[&str]) {
    let actual: Vec<&str> = actual.iter().map(String::as_str).collect();
    assert_eq!(actual, expected);
}

#[test]
fn tfidf_vectorizer_default_fit_transform_matches_sklearn() {
    let fitted = TfidfVectorizer::new().fit(&docs()).unwrap();
    assert_vocabulary(fitted.vocabulary(), &["bird", "cat", "dog", "fish"]);
    assert_idf_close(
        fitted.idf().unwrap().as_slice().unwrap(),
        &[
            1.6931471805599454,
            1.2876820724517808,
            1.2876820724517808,
            1.6931471805599454,
        ],
    );

    let actual = fitted.transform(&docs()).unwrap();
    let expected = array![
        [0.0, 0.8944271909999159, 0.4472135954999579, 0.0],
        [0.0, 0.0, 0.6053485081062916, 0.7959605415681652],
        [0.7959605415681652, 0.6053485081062916, 0.0, 0.0],
    ];
    assert_matrix_close(&actual, &expected);
}

#[test]
fn tfidf_vectorizer_transform_matches_sklearn() {
    let fitted = TfidfVectorizer::new().fit(&docs()).unwrap();
    let actual = fitted.transform(&transform_docs()).unwrap();
    let expected = array![
        [
            0.680918560398684,
            0.5178561161676974,
            0.5178561161676974,
            0.0
        ],
        [0.0, 0.0, 0.0, 1.0],
    ];
    assert_matrix_close(&actual, &expected);
}

#[test]
fn tfidf_vectorizer_sublinear_no_norm_matches_sklearn() {
    let actual = TfidfVectorizer::new()
        .sublinear_tf(true)
        .norm(TfidfNorm::None)
        .fit_transform(&docs())
        .unwrap();
    let expected = array![
        [0.0, 2.18023527042932, 1.2876820724517808, 0.0],
        [0.0, 0.0, 1.2876820724517808, 1.6931471805599454],
        [1.6931471805599454, 1.2876820724517808, 0.0, 0.0],
    ];
    assert_matrix_close(&actual, &expected);
}

#[test]
fn tfidf_vectorizer_use_idf_false_matches_sklearn() {
    let fitted = TfidfVectorizer::new().use_idf(false).fit(&docs()).unwrap();
    assert!(fitted.idf().is_none());
    let actual = fitted.transform(&docs()).unwrap();
    let expected = array![
        [0.0, 0.8944271909999159, 0.4472135954999579, 0.0],
        [0.0, 0.0, 0.7071067811865475, 0.7071067811865475],
        [0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
    ];
    assert_matrix_close(&actual, &expected);
}

#[test]
fn tfidf_vectorizer_max_features_matches_sklearn() {
    let fitted = TfidfVectorizer::new().max_features(2).fit(&docs()).unwrap();
    assert_vocabulary(fitted.vocabulary(), &["cat", "dog"]);
    assert_idf_close(
        fitted.idf().unwrap().as_slice().unwrap(),
        &[1.2876820724517808, 1.2876820724517808],
    );

    let actual = fitted.transform(&docs()).unwrap();
    let expected = array![
        [0.8944271909999159, 0.4472135954999579],
        [0.0, 1.0],
        [1.0, 0.0],
    ];
    assert_matrix_close(&actual, &expected);
}

#[test]
fn tfidf_vectorizer_validation_contracts() {
    let empty: Vec<String> = Vec::new();
    assert!(TfidfVectorizer::new().fit(&empty).is_err());

    let fitted = TfidfVectorizer::new().fit(&docs()).unwrap();
    assert!(fitted.transform(&empty).is_err());
    assert!(TfidfVectorizer::new().max_df(0.0).fit(&docs()).is_err());
}
