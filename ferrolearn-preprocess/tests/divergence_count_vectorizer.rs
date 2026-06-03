//! Divergence + green-guard tests for `CountVectorizer` against scikit-learn 1.5.2
//! `sklearn/feature_extraction/text.py` `class CountVectorizer` (`:929`).
//!
//! All expected values come from a LIVE sklearn 1.5.2 oracle run (R-CHAR-3),
//! reproduced in the per-test doc comments, NEVER copied from the ferrolearn side.

use ferrolearn_preprocess::count_vectorizer::CountVectorizer;

fn fit_vocab(docs: &[&str], cv: CountVectorizer) -> Vec<String> {
    let owned: Vec<String> = docs.iter().map(|s| (*s).to_string()).collect();
    let fitted = cv.fit(&owned).unwrap();
    let mut v = fitted.vocabulary().to_vec();
    v.sort();
    v
}

// ===========================================================================
// DIVERGENCE PINS (these MUST FAIL against current ferrolearn)
// ===========================================================================

/// Divergence: ferrolearn's `tokenize` (count_vectorizer.rs:261-272) splits on
/// every non-alphanumeric char and KEEPS length-1 tokens, whereas sklearn's
/// default `token_pattern=r"(?u)\b\w\w+\b"` (text.py:1161, build_tokenizer:350)
/// matches maximal runs of 2+ word chars, DROPPING length-1 tokens.
///
/// LIVE oracle (sklearn 1.5.2):
///   CountVectorizer().fit(['foo a bar']).get_feature_names_out()
///   -> ['bar', 'foo']    (length-1 'a' is dropped)
/// ferrolearn keeps 'a' -> ['a', 'bar', 'foo'].
/// Tracking: see blocker issue filed by critic.
#[test]
fn divergence_tokenizer_drops_length_one_tokens() {
    let vocab = fit_vocab(&["foo a bar"], CountVectorizer::new());
    // sklearn drops length-1 'a'.
    assert_eq!(vocab, vec!["bar".to_string(), "foo".to_string()]);
}

/// Divergence: ferrolearn's `tokenize` (count_vectorizer.rs:268) splits on `_`
/// because `'_'.is_alphanumeric()` is false in Rust, whereas in sklearn's
/// default `token_pattern=r"(?u)\b\w\w+\b"` (text.py:1161) `\w` = `[A-Za-z0-9_]`,
/// so `_` is a WORD char and never a split boundary.
///
/// LIVE oracle (sklearn 1.5.2):
///   CountVectorizer().fit(['a_b cd']).get_feature_names_out()
///   -> ['a_b', 'cd']     ('a_b' is ONE 3-char token; '_' not split)
/// ferrolearn splits on '_' into length-1 'a','b' (kept) plus 'cd'
/// -> ['a', 'b', 'cd'].
/// Tracking: see blocker issue filed by critic.
#[test]
fn divergence_tokenizer_underscore_is_word_char() {
    let vocab = fit_vocab(&["a_b cd"], CountVectorizer::new());
    // sklearn treats 'a_b' as a single token and drops nothing else.
    assert_eq!(vocab, vec!["a_b".to_string(), "cd".to_string()]);
}

/// Divergence: ferrolearn computes `max_df_abs = ceil(max_df * n_docs)` and
/// filters `count <= max_df_abs` (count_vectorizer.rs:135,138), whereas sklearn
/// computes `max_doc_count = max_df * n_doc` as a FLOAT with no rounding
/// (text.py:1379) and filters `dfs <= high` (text.py:1219 in _limit_features),
/// i.e. `df <= max_df * n_doc`.
///
/// LIVE oracle (sklearn 1.5.2), term 'cat' in 2 of 3 docs, max_df=0.5:
///   CountVectorizer(max_df=0.5).fit(['cat dog','cat bird','xx yy'])
///     .get_feature_names_out() sorted -> ['bird', 'dog', 'xx', 'yy']
///   ('cat': 2 <= 0.5*3 = 1.5 is FALSE -> EXCLUDED)
/// ferrolearn: 2 <= ceil(1.5) = 2 is TRUE -> 'cat' INCLUDED.
/// All tokens are length >= 2 so the tokenizer fix does not interfere.
/// Tracking: see blocker issue filed by critic.
#[test]
fn divergence_max_df_float_threshold_no_ceil() {
    let vocab = fit_vocab(
        &["cat dog", "cat bird", "xx yy"],
        CountVectorizer::new().max_df(0.5),
    );
    // sklearn EXCLUDES 'cat' (2/3 > 0.5*3=1.5).
    assert_eq!(
        vocab,
        vec![
            "bird".to_string(),
            "dog".to_string(),
            "xx".to_string(),
            "yy".to_string()
        ]
    );
}

// ===========================================================================
// GREEN GUARDS (SHIPPED behavior; all tokens length >= 2 so unaffected by the
// tokenizer fix). These MUST PASS.
// ===========================================================================

/// REQ-1 default value-match. LIVE oracle (sklearn 1.5.2):
///   CountVectorizer().fit_transform(['the cat sat','the cat sat on the mat'])
///   vocab -> ['cat','mat','on','sat','the']
///   counts -> [[1,0,0,1,1],[1,1,1,1,2]]
#[test]
fn guard_default_value_match() {
    let docs = vec![
        "the cat sat".to_string(),
        "the cat sat on the mat".to_string(),
    ];
    let fitted = CountVectorizer::new().fit(&docs).unwrap();
    let mut vocab = fitted.vocabulary().to_vec();
    vocab.sort();
    assert_eq!(vocab, vec!["cat", "mat", "on", "sat", "the"]);

    let counts = fitted.transform(&docs).unwrap();
    let map = fitted.vocabulary_map();
    // Expected per-term column values from the sklearn oracle, addressed by name
    // (ferrolearn column order is alphabetical, identical to sklearn here).
    let expect_doc0 = [
        ("cat", 1.0),
        ("mat", 0.0),
        ("on", 0.0),
        ("sat", 1.0),
        ("the", 1.0),
    ];
    let expect_doc1 = [
        ("cat", 1.0),
        ("mat", 1.0),
        ("on", 1.0),
        ("sat", 1.0),
        ("the", 2.0),
    ];
    for (term, v) in expect_doc0 {
        assert_eq!(counts[[0, map[term]]], v, "doc0 term {term}");
    }
    for (term, v) in expect_doc1 {
        assert_eq!(counts[[1, map[term]]], v, "doc1 term {term}");
    }
}

/// REQ-3 binary value-match. LIVE oracle (sklearn 1.5.2):
///   CountVectorizer(binary=True).fit_transform(['the the the cat'])
///   vocab -> ['cat','the']; counts -> [[1,1]]
#[test]
fn guard_binary_value_match() {
    let docs = vec!["the the the cat".to_string()];
    let fitted = CountVectorizer::new().binary(true).fit(&docs).unwrap();
    let counts = fitted.transform(&docs).unwrap();
    let map = fitted.vocabulary_map();
    assert_eq!(counts[[0, map["the"]]], 1.0);
    assert_eq!(counts[[0, map["cat"]]], 1.0);
}

/// REQ-4 lowercase value-match. LIVE oracle (sklearn 1.5.2):
///   CountVectorizer().fit(['Hello HELLO hello world']).get_feature_names_out()
///   -> ['hello','world']
#[test]
fn guard_lowercase_value_match() {
    let vocab = fit_vocab(&["Hello HELLO hello world"], CountVectorizer::new());
    assert_eq!(vocab, vec!["hello".to_string(), "world".to_string()]);
}

/// REQ-4 no-lowercase value-match. LIVE oracle (sklearn 1.5.2):
///   CountVectorizer(lowercase=False).fit(['Hello hello world'])
///     .get_feature_names_out() -> ['Hello','hello','world']
#[test]
fn guard_no_lowercase_value_match() {
    let vocab = fit_vocab(
        &["Hello hello world"],
        CountVectorizer::new().lowercase(false),
    );
    assert_eq!(
        vocab,
        vec![
            "Hello".to_string(),
            "hello".to_string(),
            "world".to_string()
        ]
    );
}

/// REQ-5 min_df absolute count. LIVE oracle (sklearn 1.5.2):
///   CountVectorizer(min_df=3).fit(['cat dog','cat bird','cat fish'])
///     .get_feature_names_out() -> ['cat']   ('cat' is the only term in all 3 docs)
#[test]
fn guard_min_df_absolute_count() {
    let vocab = fit_vocab(
        &["cat dog", "cat bird", "cat fish"],
        CountVectorizer::new().min_df(3),
    );
    assert_eq!(vocab, vec!["cat".to_string()]);
}
