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
// ACTIVE DIVERGENCE PINS (#[ignore]d; these FAIL against current ferrolearn)
// ===========================================================================

/// Divergence: `CountVectorizer::fit` (`count_vectorizer.rs:130`) returns
/// `Ok(_)` with an empty vocabulary when every token is shorter than 2 chars,
/// whereas sklearn raises `ValueError("empty vocabulary; perhaps the documents
/// only contain stop words")` in `_count_vocab`
/// (`sklearn/feature_extraction/text.py:1277-1279`).
///
/// LIVE oracle (sklearn 1.5.2):
///   CountVectorizer().fit_transform(["a b c d"])
///   -> ValueError: empty vocabulary; perhaps the documents only contain stop words
/// ferrolearn: Ok, vocabulary() == [] (no error).
///
/// Tracking: #2336
#[test]
#[ignore = "divergence: fit returns Ok empty-vocab instead of ValueError when all tokens <2 chars; tracking #2336"]
fn divergence_empty_vocab_all_single_char() {
    // "a b c d": every token is a single char and dropped by the 2+ char rule,
    // leaving an empty vocabulary. sklearn raises; ferrolearn must too.
    let docs = vec!["a b c d".to_string()];
    let result = CountVectorizer::new().fit(&docs);
    assert!(
        result.is_err(),
        "sklearn raises ValueError(empty vocabulary) at text.py:1278 for docs \
         with only single-char tokens; ferrolearn returned Ok with vocab={:?}",
        result.map(|f| f.vocabulary().to_vec())
    );
}

/// Divergence: `CountVectorizer::fit` (`count_vectorizer.rs:169-174`) returns
/// `Ok(_)` with an empty vocabulary when `min_df` prunes every term, whereas
/// sklearn raises a `ValueError` — either "max_df corresponds to < documents
/// than min_df" (`text.py:1382`) when `max_doc_count < min_doc_count`, or
/// "After pruning, no terms remain..." (`_limit_features`, `text.py:1234`).
///
/// LIVE oracle (sklearn 1.5.2):
///   CountVectorizer(min_df=5).fit(["aa bb", "cc dd"])
///   -> ValueError: max_df corresponds to < documents than min_df
/// ferrolearn: Ok, vocabulary() == [] (no error).
///
/// Tracking: #2337
#[test]
#[ignore = "divergence: fit returns Ok empty-vocab instead of ValueError when min_df prunes all terms; tracking #2337"]
fn divergence_empty_vocab_min_df_prunes_all() {
    // 2 docs, min_df=5: no term appears in >=5 docs, so every term is pruned.
    // sklearn raises (max_df default 1.0 -> max_doc_count=2 < min_doc_count=5).
    let docs = vec!["aa bb".to_string(), "cc dd".to_string()];
    let result = CountVectorizer::new().min_df(5).fit(&docs);
    assert!(
        result.is_err(),
        "sklearn raises ValueError at text.py:1382 when min_df prunes every \
         term; ferrolearn returned Ok with vocab={:?}",
        result.map(|f| f.vocabulary().to_vec())
    );
}

// ===========================================================================
// GREEN GUARDS (formerly-divergent behavior now SHIPPED/fixed; these MUST PASS).
// Kept as regression guards: 2+char rule, '_' word char, max_df float no-ceil.
// ===========================================================================

/// REGRESSION GUARD (was a divergence, fixed under REQ-2 #1217): sklearn's
/// default `token_pattern=r"(?u)\b\w\w+\b"` (text.py:1161) DROPS length-1 tokens.
///
/// LIVE oracle (sklearn 1.5.2):
///   CountVectorizer().fit(['foo a bar']).get_feature_names_out()
///   -> ['bar', 'foo']    (length-1 'a' is dropped)
#[test]
fn guard_tokenizer_drops_length_one_tokens() {
    let vocab = fit_vocab(&["foo a bar"], CountVectorizer::new());
    assert_eq!(vocab, vec!["bar".to_string(), "foo".to_string()]);
}

/// REGRESSION GUARD (was a divergence, fixed under REQ-2 #1217): in sklearn's
/// default `token_pattern` (text.py:1161) `\w = [A-Za-z0-9_]`, so `_` is a WORD
/// char and never a split boundary.
///
/// LIVE oracle (sklearn 1.5.2):
///   CountVectorizer().fit(['a_b cd']).get_feature_names_out()
///   -> ['a_b', 'cd']     ('a_b' is ONE 3-char token; '_' not split)
#[test]
fn guard_tokenizer_underscore_is_word_char() {
    let vocab = fit_vocab(&["a_b cd"], CountVectorizer::new());
    assert_eq!(vocab, vec!["a_b".to_string(), "cd".to_string()]);
}

/// REGRESSION GUARD (was a divergence, fixed under #1218): sklearn computes
/// `max_doc_count = max_df * n_doc` as a FLOAT with no rounding (text.py:1380)
/// and filters `dfs <= high` (text.py:1219), i.e. `df <= max_df * n_doc`.
///
/// LIVE oracle (sklearn 1.5.2), term 'cat' in 2 of 3 docs, max_df=0.5:
///   CountVectorizer(max_df=0.5).fit(['cat dog','cat bird','xx yy'])
///     .get_feature_names_out() sorted -> ['bird', 'dog', 'xx', 'yy']
///   ('cat': 2 <= 0.5*3 = 1.5 is FALSE -> EXCLUDED)
#[test]
fn guard_max_df_float_threshold_no_ceil() {
    let vocab = fit_vocab(
        &["cat dog", "cat bird", "xx yy"],
        CountVectorizer::new().max_df(0.5),
    );
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
// GREEN GUARDS (SHIPPED value/ordering behavior). These MUST PASS.
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

/// REQ-7 max_features top-N with alphabetical tie-break. LIVE oracle (sklearn 1.5.2):
///   CountVectorizer(max_features=3)
///     .fit_transform(['cat cat cat dog dog bird ant','cat dog bird'])
///   vocab -> ['bird','cat','dog']  ('ant' total freq 1, lowest, dropped)
///   counts -> [[1,3,2],[1,1,1]]
/// sklearn breaks ties toward the alphabetically EARLIER term: `_limit_features`
/// (`text.py:1222-1226`) sorts features alphabetically first, then
/// `(-tfs).argsort()` (stable) keeps lower column indices on ties.
#[test]
fn guard_max_features_topn_alpha_tiebreak() {
    let docs = vec![
        "cat cat cat dog dog bird ant".to_string(),
        "cat dog bird".to_string(),
    ];
    let fitted = CountVectorizer::new().max_features(3).fit(&docs).unwrap();
    let mut vocab = fitted.vocabulary().to_vec();
    vocab.sort();
    assert_eq!(vocab, vec!["bird", "cat", "dog"]);
    let counts = fitted.transform(&docs).unwrap();
    let map = fitted.vocabulary_map();
    assert_eq!(counts[[0, map["cat"]]], 3.0);
    assert_eq!(counts[[0, map["dog"]]], 2.0);
    assert_eq!(counts[[0, map["bird"]]], 1.0);
}
