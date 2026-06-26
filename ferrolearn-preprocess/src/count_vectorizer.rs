//! Count vectorizer: convert text documents to a term-count matrix.
//!
//! Tokenizes documents into runs of 2+ word characters (the Rust analog of
//! scikit-learn's default `token_pattern=r"(?u)\b\w\w+\b"`,
//! `sklearn/feature_extraction/text.py:1161`), builds an alphabetically-sorted
//! vocabulary, and produces a term-count matrix of shape `(n_docs, n_vocab)`.
//!
//! Translation target: scikit-learn 1.5.2 `class CountVectorizer` (`text.py:929`).
//! Design: `.design/preprocess/count_vectorizer.md`. Tracking: #1216.
//!
//! `## REQ status`
//!
//! | REQ | Status | Anchor |
//! |---|---|---|
//! | REQ-1 default fit/transform, sorted vocab, count matrix | SHIPPED (scoped: dense) | `CountVectorizer::fit` / `FittedCountVectorizer::transform`; sklearn `_count_vocab` `text.py:1242-1305` |
//! | REQ-2 default token_pattern (drop length-1, `_` word char) | SHIPPED (#1217) | `fn tokenize`; sklearn `text.py:1161`, `build_tokenizer:350` |
//! | REQ-3 binary count clipping | SHIPPED | `FittedCountVectorizer::transform`; sklearn `text.py:1374` |
//! | REQ-4 lowercase toggle | SHIPPED | `fn tokenize`; sklearn `text.py:1157`,`:323` |
//! | REQ-5 max_df/min_df int-vs-float duality + threshold errors | NOT-STARTED (#1219; ceil sub-fix shipped #1218; max_df<min_df + post-prune empty-vocab errors shipped #2337) | `fit` df-filter; sklearn `text.py:1379-1382`,`:1236-1239` |
//! | REQ-6 ngram_range word n-grams | NOT-STARTED (#1220) | sklearn `_word_ngrams` `text.py:242` |
//! | REQ-7 max_features top-N + tie/sort | SHIPPED (scoped) | `fit`; sklearn `_limit_features` `text.py:1222-1227` |
//! | REQ-8 tokenizer/token_pattern/preprocessor/analyzer/strip_accents | NOT-STARTED (#1221) | sklearn `build_analyzer` `text.py:419` |
//! | REQ-9 stop_words | NOT-STARTED (#1222) | sklearn `get_stop_words` `text.py:370` |
//! | REQ-10 fixed vocabulary param + dtype | NOT-STARTED (#1223) | sklearn `_count_vocab` `text.py:1242-1244`,`:1147` |
//! | REQ-11 sparse CSR output | NOT-STARTED (#1224) | sklearn `_count_vocab` `text.py:1299-1304` |
//! | REQ-12 get_feature_names_out contract | NOT-STARTED (#1225) | sklearn `text.py:1455` |
//! | REQ-13 HashingVectorizer | SHIPPED scoped | dense [`HashingVectorizer`] with MurmurHash3 x86-32, `n_features`, `alternate_sign`, `binary`, `norm`, lowercase; tests in `tests/divergence_hashing_vectorizer.rs` |
//! | REQ-14 full 16-param ctor + _parameter_constraints | NOT-STARTED (#1227) | sklearn `text.py:1124-1148` |
//! | REQ-14a empty-vocabulary ValueError parity (post-tokenize + max_df<min_df + post-prune) | SHIPPED (#2336 #2337) | `CountVectorizer::fit` empty-vocab/`max_df`/post-prune `Err(InvalidParameter)`; sklearn `text.py:1277-1279`,`:1381-1382`,`:1236-1239`. Consumer: crate re-export `pub use count_vectorizer::CountVectorizer` (`lib.rs`). |
//! | REQ-15 PyO3 binding | NOT-STARTED (#1228) | `ferrolearn-python/src/transformers.rs` (absent) |

use crate::hashing::{murmurhash3_32_signed, signed_hash_index};
use crate::tfidf::TfidfNorm;
use std::collections::HashMap;

use ferrolearn_core::error::FerroError;
use ndarray::Array2;

// ---------------------------------------------------------------------------
// CountVectorizer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted count vectorizer.
///
/// Tokenizes documents by splitting on non-alphanumeric boundaries, builds a
/// vocabulary sorted alphabetically, and transforms documents into a
/// term-count matrix.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::count_vectorizer::{CountVectorizer, FittedCountVectorizer};
///
/// let docs = vec![
///     "the cat sat".to_string(),
///     "the cat sat on the mat".to_string(),
/// ];
/// let cv = CountVectorizer::new();
/// let fitted = cv.fit(&docs).unwrap();
/// let counts = fitted.transform(&docs).unwrap();
/// assert_eq!(counts.nrows(), 2);
/// assert_eq!(counts.ncols(), fitted.vocabulary().len());
/// ```
#[derive(Debug, Clone)]
pub struct CountVectorizer {
    /// Maximum number of features (vocabulary size). `None` means no limit.
    pub max_features: Option<usize>,
    /// Minimum document frequency (absolute count) for a term to be included.
    pub min_df: usize,
    /// Maximum document frequency as a fraction of total documents.
    /// Terms appearing in more than `max_df * n_docs` documents are excluded.
    pub max_df: f64,
    /// If `true`, all counts are clipped to 0/1 (binary occurrence).
    pub binary: bool,
    /// If `true`, lowercase all tokens before counting.
    pub lowercase: bool,
}

impl CountVectorizer {
    /// Create a new `CountVectorizer` with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_features: None,
            min_df: 1,
            max_df: 1.0,
            binary: false,
            lowercase: true,
        }
    }

    /// Set the maximum number of features.
    #[must_use]
    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    /// Set the minimum document frequency.
    #[must_use]
    pub fn min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set the maximum document frequency as a fraction of total documents.
    #[must_use]
    pub fn max_df(mut self, max_df: f64) -> Self {
        self.max_df = max_df;
        self
    }

    /// Enable or disable binary mode.
    #[must_use]
    pub fn binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// Enable or disable lowercasing.
    #[must_use]
    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Fit the vectorizer on a corpus of documents.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the corpus is empty.
    /// Returns [`FerroError::InvalidParameter`] if `max_df` is not in `(0, 1]`.
    pub fn fit(&self, docs: &[String]) -> Result<FittedCountVectorizer, FerroError> {
        let n_docs = docs.len();
        if n_docs == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "CountVectorizer::fit".into(),
            });
        }
        if self.max_df <= 0.0 || self.max_df > 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "max_df".into(),
                reason: format!("must be in (0, 1], got {}", self.max_df),
            });
        }

        // Build document-frequency counts.
        let mut df_counts: HashMap<String, usize> = HashMap::new();
        for doc in docs {
            let tokens = tokenize(doc, self.lowercase);
            // Unique tokens per document.
            let mut seen = std::collections::HashSet::new();
            for tok in tokens {
                if seen.insert(tok.clone()) {
                    *df_counts.entry(tok).or_insert(0) += 1;
                }
            }
        }

        // Empty-vocabulary error (before df-pruning). sklearn's `_count_vocab`
        // raises `ValueError("empty vocabulary; perhaps the documents only
        // contain stop words")` when the assembled vocabulary is empty
        // (`sklearn/feature_extraction/text.py:1277-1279`). This fires when every
        // token is dropped by the token_pattern (e.g. all length-1 tokens).
        if df_counts.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "vocabulary".into(),
                reason: "empty vocabulary; perhaps the documents only contain stop words".into(),
            });
        }

        // max_df-vs-min_df cross-validation. sklearn computes the document-count
        // bounds (`text.py:1379-1380`) and raises
        // `ValueError("max_df corresponds to < documents than min_df")` when the
        // max_df bound is below the min_df bound (`text.py:1381-1382`). Here
        // `max_df` is a float proportion (bound = `max_df * n_doc`) and `min_df`
        // is an absolute document count (bound = `min_df`).
        let max_df_count = self.max_df * n_docs as f64;
        let min_doc_count = self.min_df as f64;
        if max_df_count < min_doc_count {
            return Err(FerroError::InvalidParameter {
                name: "max_df".into(),
                reason: "max_df corresponds to < documents than min_df".into(),
            });
        }

        // Filter by min_df and max_df.
        //
        // sklearn 1.5.2 computes `max_doc_count = max_df * n_doc` as a FLOAT with
        // NO rounding (`sklearn/feature_extraction/text.py:1379`) and keeps terms
        // with `df <= max_doc_count` (`_limit_features`, `text.py:1219`:
        // `mask &= dfs <= high`). We mirror that exactly: compare the integer
        // document count against the un-rounded float threshold. (Note: sklearn
        // also accepts an integer `max_df` as an absolute count; that int-vs-float
        // duality is a separate gap and is intentionally not implemented here.)
        // (`max_df_count` is computed above for the max_df-vs-min_df check.)
        let mut vocab: Vec<String> = df_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_df && (*count as f64) <= max_df_count)
            .map(|(term, _)| term)
            .collect();
        vocab.sort();

        // Apply max_features: keep the top-N by total corpus frequency.
        if let Some(max_f) = self.max_features
            && vocab.len() > max_f
        {
            // Re-count total frequencies for the remaining terms.
            let mut total_freq: HashMap<String, usize> = HashMap::new();
            for doc in docs {
                let tokens = tokenize(doc, self.lowercase);
                for tok in tokens {
                    if vocab.binary_search(&tok).is_ok() {
                        *total_freq.entry(tok).or_insert(0) += 1;
                    }
                }
            }
            // Sort by descending frequency, then alphabetically for ties.
            vocab.sort_by(|a, b| {
                let fa = total_freq.get(a).unwrap_or(&0);
                let fb = total_freq.get(b).unwrap_or(&0);
                fb.cmp(fa).then_with(|| a.cmp(b))
            });
            vocab.truncate(max_f);
            vocab.sort(); // restore alphabetical order for consistent indexing
        }

        // Post-pruning empty-vocabulary error. sklearn's `_limit_features` raises
        // `ValueError("After pruning, no terms remain. Try a lower min_df or a
        // higher max_df.")` when the df/max_features filter removes every term
        // (`sklearn/feature_extraction/text.py:1236-1239`).
        if vocab.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "vocabulary".into(),
                reason: "After pruning, no terms remain. Try a lower min_df or a higher max_df."
                    .into(),
            });
        }

        // Build vocabulary mapping.
        let vocabulary: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        Ok(FittedCountVectorizer {
            vocabulary,
            sorted_terms: vocab,
            binary: self.binary,
            lowercase: self.lowercase,
        })
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedCountVectorizer
// ---------------------------------------------------------------------------

/// A fitted count vectorizer holding the learned vocabulary.
///
/// Created by calling [`CountVectorizer::fit`].
#[derive(Debug, Clone)]
pub struct FittedCountVectorizer {
    /// Map from term to column index.
    vocabulary: HashMap<String, usize>,
    /// Sorted vocabulary terms (for deterministic column ordering).
    sorted_terms: Vec<String>,
    /// Whether to clip counts to binary.
    binary: bool,
    /// Whether to lowercase tokens.
    lowercase: bool,
}

impl FittedCountVectorizer {
    /// Return the vocabulary as a sorted slice of terms.
    #[must_use]
    pub fn vocabulary(&self) -> &[String] {
        &self.sorted_terms
    }

    /// Return the vocabulary mapping (term -> column index).
    #[must_use]
    pub fn vocabulary_map(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }

    /// Transform documents into a term-count matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if `docs` is empty.
    pub fn transform(&self, docs: &[String]) -> Result<Array2<f64>, FerroError> {
        if docs.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "FittedCountVectorizer::transform".into(),
            });
        }

        let n_docs = docs.len();
        let n_vocab = self.sorted_terms.len();
        let mut matrix = Array2::<f64>::zeros((n_docs, n_vocab));

        for (i, doc) in docs.iter().enumerate() {
            let tokens = tokenize(doc, self.lowercase);
            for tok in tokens {
                if let Some(&col) = self.vocabulary.get(&tok) {
                    if self.binary {
                        matrix[[i, col]] = 1.0;
                    } else {
                        matrix[[i, col]] += 1.0;
                    }
                }
            }
        }

        Ok(matrix)
    }
}

// ---------------------------------------------------------------------------
// HashingVectorizer
// ---------------------------------------------------------------------------

/// Stateless text vectorizer using signed 32-bit MurmurHash3 feature hashing.
///
/// This is the dense Rust analogue of scikit-learn's default word-analyzer
/// `HashingVectorizer` path. It reuses the same tokenization semantics as
/// [`CountVectorizer`] and emits a dense `Array2<f64>` instead of sklearn's CSR
/// sparse matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct HashingVectorizer {
    /// Number of hashed output columns.
    pub n_features: usize,
    /// If `true`, negative MurmurHash3 values contribute with a negative sign.
    pub alternate_sign: bool,
    /// If `true`, non-zero entries are clipped to 1 after duplicate summing.
    pub binary: bool,
    /// If `true`, lowercase all tokens before hashing.
    pub lowercase: bool,
    /// Optional row normalization. sklearn defaults to L2 normalization.
    pub norm: TfidfNorm,
}

impl HashingVectorizer {
    /// Create a new `HashingVectorizer` with sklearn-like defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_features: 1 << 20,
            alternate_sign: true,
            binary: false,
            lowercase: true,
            norm: TfidfNorm::L2,
        }
    }

    /// Set the number of hashed output columns.
    #[must_use]
    pub fn n_features(mut self, n_features: usize) -> Self {
        self.n_features = n_features;
        self
    }

    /// Enable or disable alternating signs.
    #[must_use]
    pub fn alternate_sign(mut self, alternate_sign: bool) -> Self {
        self.alternate_sign = alternate_sign;
        self
    }

    /// Enable or disable binary clipping.
    #[must_use]
    pub fn binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// Enable or disable lowercasing.
    #[must_use]
    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Set the row normalization mode.
    #[must_use]
    pub fn norm(mut self, norm: TfidfNorm) -> Self {
        self.norm = norm;
        self
    }

    /// Stateless fit: validate parameters and return `self`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when `n_features` is outside
    /// sklearn's accepted range `[1, i32::MAX)`.
    pub fn fit(&self, _docs: &[String]) -> Result<Self, FerroError> {
        self.validate()?;
        Ok(self.clone())
    }

    /// Stateless partial fit, matching sklearn's no-op `partial_fit`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when `n_features` is invalid.
    pub fn partial_fit(&self, docs: &[String]) -> Result<Self, FerroError> {
        self.fit(docs)
    }

    /// Transform documents into a dense hashed feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when `n_features` is invalid and
    /// [`FerroError::InsufficientSamples`] when the document slice is empty.
    pub fn transform(&self, docs: &[String]) -> Result<Array2<f64>, FerroError> {
        self.validate()?;
        if docs.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "HashingVectorizer::transform".into(),
            });
        }

        let mut matrix = Array2::<f64>::zeros((docs.len(), self.n_features));
        for (row, doc) in docs.iter().enumerate() {
            for token in tokenize(doc, self.lowercase) {
                let hash = murmurhash3_32_signed(token.as_bytes(), 0);
                let col = signed_hash_index(hash, self.n_features);
                let value = if self.alternate_sign && hash < 0 {
                    -1.0
                } else {
                    1.0
                };
                matrix[[row, col]] += value;
            }
        }

        if self.binary {
            for value in &mut matrix {
                if *value != 0.0 {
                    *value = 1.0;
                }
            }
        }

        normalize_dense_rows(&mut matrix, self.norm);
        Ok(matrix)
    }

    /// Fit and transform in one pass.
    ///
    /// # Errors
    ///
    /// Propagates validation errors from [`Self::transform`].
    pub fn fit_transform(&self, docs: &[String]) -> Result<Array2<f64>, FerroError> {
        self.fit(docs)?.transform(docs)
    }

    fn validate(&self) -> Result<(), FerroError> {
        if self.n_features == 0 || self.n_features >= i32::MAX as usize {
            return Err(FerroError::InvalidParameter {
                name: "n_features".into(),
                reason: "must be in [1, 2147483647)".into(),
            });
        }
        Ok(())
    }
}

impl Default for HashingVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Tokenize a document, matching scikit-learn's default `token_pattern`.
///
/// sklearn 1.5.2 defaults to `token_pattern=r"(?u)\b\w\w+\b"`
/// (`sklearn/feature_extraction/text.py:1161`), which matches maximal runs of
/// 2+ word characters where `\w = [A-Za-z0-9_]` (Unicode-aware via `(?u)`).
/// We therefore treat a char as part of a token iff it is alphanumeric or `_`
/// (`char::is_alphanumeric` is Unicode-aware, the faithful analog of `\w`), and
/// keep only tokens of length >= 2, dropping single-char tokens.
fn tokenize(doc: &str, lowercase: bool) -> Vec<String> {
    let text = if lowercase {
        doc.to_lowercase()
    } else {
        doc.to_string()
    };

    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|s| !s.is_empty() && s.chars().count() >= 2)
        .map(std::string::ToString::to_string)
        .collect()
}

fn normalize_dense_rows(matrix: &mut Array2<f64>, norm: TfidfNorm) {
    match norm {
        TfidfNorm::L1 => {
            for mut row in matrix.rows_mut() {
                let denom: f64 = row.iter().map(|v| v.abs()).sum();
                if denom > 0.0 {
                    for v in &mut row {
                        *v /= denom;
                    }
                }
            }
        }
        TfidfNorm::L2 => {
            for mut row in matrix.rows_mut() {
                let denom = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                if denom > 0.0 {
                    for v in &mut row {
                        *v /= denom;
                    }
                }
            }
        }
        TfidfNorm::None => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_count_vectorizer_basic() {
        let docs = vec![
            "the cat sat".to_string(),
            "the cat sat on the mat".to_string(),
        ];
        let cv = CountVectorizer::new();
        let fitted = cv.fit(&docs).unwrap();
        let counts = fitted.transform(&docs).unwrap();

        assert_eq!(counts.nrows(), 2);
        let vocab = fitted.vocabulary();
        assert!(vocab.contains(&"cat".to_string()));
        assert!(vocab.contains(&"the".to_string()));
        assert!(vocab.contains(&"sat".to_string()));

        // "the" appears once in doc 0, twice in doc 1
        let the_idx = fitted.vocabulary_map()["the"];
        assert_abs_diff_eq!(counts[[0, the_idx]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(counts[[1, the_idx]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_count_vectorizer_binary() {
        let docs = vec!["the the the".to_string()];
        let cv = CountVectorizer::new().binary(true);
        let fitted = cv.fit(&docs).unwrap();
        let counts = fitted.transform(&docs).unwrap();
        // "the" count should be 1 (binary mode)
        assert_abs_diff_eq!(counts[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_count_vectorizer_lowercase() {
        let docs = vec!["Hello HELLO hello".to_string()];
        let cv = CountVectorizer::new();
        let fitted = cv.fit(&docs).unwrap();
        let counts = fitted.transform(&docs).unwrap();
        // All should fold to "hello", count = 3
        assert_eq!(fitted.vocabulary().len(), 1);
        assert_abs_diff_eq!(counts[[0, 0]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_count_vectorizer_no_lowercase() {
        let docs = vec!["Hello hello".to_string()];
        let cv = CountVectorizer::new().lowercase(false);
        let fitted = cv.fit(&docs).unwrap();
        // "Hello" and "hello" are different tokens
        assert_eq!(fitted.vocabulary().len(), 2);
    }

    /// max_features keeps the top-N terms by total corpus frequency.
    ///
    /// LIVE oracle (sklearn 1.5.2):
    ///   CountVectorizer(max_features=3).fit_transform(
    ///       ['cat cat cat dog dog bird ant','cat dog bird'])
    ///   sorted(get_feature_names_out()) -> ['bird','cat','dog']
    ///   ('ant' has corpus frequency 1, the lowest, so it is dropped)
    #[test]
    fn test_count_vectorizer_max_features() {
        let docs = vec![
            "cat cat cat dog dog bird ant".to_string(),
            "cat dog bird".to_string(),
        ];
        let cv = CountVectorizer::new().max_features(3);
        let fitted = cv.fit(&docs).unwrap();
        let mut vocab = fitted.vocabulary().to_vec();
        vocab.sort();
        assert_eq!(vocab, vec!["bird", "cat", "dog"]);
    }

    #[test]
    fn test_count_vectorizer_min_df() {
        let docs = vec![
            "cat dog".to_string(),
            "cat bird".to_string(),
            "cat fish".to_string(),
        ];
        // Only "cat" appears in all 3 docs
        let cv = CountVectorizer::new().min_df(3);
        let fitted = cv.fit(&docs).unwrap();
        assert_eq!(fitted.vocabulary().len(), 1);
        assert_eq!(fitted.vocabulary()[0], "cat");
    }

    #[test]
    fn test_count_vectorizer_max_df() {
        let docs = vec![
            "the cat".to_string(),
            "the dog".to_string(),
            "the bird".to_string(),
        ];
        // "the" appears in 100% of docs. max_df=0.5 should exclude it.
        let cv = CountVectorizer::new().max_df(0.5);
        let fitted = cv.fit(&docs).unwrap();
        assert!(!fitted.vocabulary().contains(&"the".to_string()));
    }

    #[test]
    fn test_count_vectorizer_empty_corpus() {
        let docs: Vec<String> = vec![];
        let cv = CountVectorizer::new();
        assert!(cv.fit(&docs).is_err());
    }

    #[test]
    fn test_count_vectorizer_transform_empty() {
        let docs = vec!["hello world".to_string()];
        let fitted = CountVectorizer::new().fit(&docs).unwrap();
        let empty: Vec<String> = vec![];
        assert!(fitted.transform(&empty).is_err());
    }

    #[test]
    fn test_count_vectorizer_unseen_tokens() {
        let train = vec!["cat dog".to_string()];
        let fitted = CountVectorizer::new().fit(&train).unwrap();
        let test = vec!["fish bird".to_string()];
        let counts = fitted.transform(&test).unwrap();
        // All zeros since no tokens match
        for &v in &counts {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }
}
