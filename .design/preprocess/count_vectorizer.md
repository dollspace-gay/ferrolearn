# CountVectorizer

<!--
tier: 3-component
status: draft
baseline-commit: dc9ba2cc
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/feature_extraction/text.py  # _VectorizerMixin (:205); _word_ngrams (:242); build_preprocessor (:323); build_tokenizer (:350); get_stop_words (:370); build_analyzer (:419); class HashingVectorizer (:562) + its _parameter_constraints (:749) / __init__ (:768) / transform (:859); class CountVectorizer(_VectorizerMixin, BaseEstimator) (:929); _parameter_constraints (:1124-1148); __init__(*, input, encoding, decode_error, strip_accents, lowercase=True, preprocessor, tokenizer, stop_words, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1,1), analyzer="word", max_df=1.0, min_df=1, max_features, vocabulary, binary=False, dtype) (:1150); _limit_features (:1203-1240); _count_vocab (:1242-1305); fit (:1307); fit_transform (:1327-1392) — int-vs-float df at :1379-1380; transform (:1394); get_feature_names_out (:1455)
ferrolearn-module: ferrolearn-preprocess/src/count_vectorizer.rs
parity-ops: CountVectorizer
crosslink-issue: 1216
-->

## Summary

scikit-learn's `CountVectorizer` (`text.py:929`) converts a corpus of raw text
documents into a **sparse CSR** document-term count matrix. It composes a
configurable analysis pipeline (`build_analyzer`, `:419`): decode → preprocess
(`strip_accents` + `lowercase`, `:323`) → tokenize (regex `token_pattern`, the
default `r"(?u)\b\w\w+\b"` matching **2+ word characters**, `:350`) → drop
`stop_words` → assemble word/char n-grams over `ngram_range` (`_word_ngrams`,
`:242`). `fit_transform` (`:1327`) builds the vocabulary, applies the
`max_df`/`min_df` document-frequency filter (where an **int is an absolute
document count and a float is a proportion**, `:1379-1380`), keeps the top
`max_features` terms by corpus frequency (`_limit_features`, `:1203`), and emits
a sorted-vocabulary CSR matrix. The route also names `HashingVectorizer`
(`:562`), a stateless feature-hashing analog with no vocabulary.

`ferrolearn-preprocess/src/count_vectorizer.rs` ships a **dense**, default-path
core: an unfitted `CountVectorizer` struct with five params
(`max_features`/`min_df`/`max_df`/`binary`/`lowercase`), a `fit` that builds an
**alphabetically-sorted** vocabulary with df filtering and top-N selection, a
`FittedCountVectorizer` holding the vocabulary map, and a `transform` producing
an `Array2<f64>` count matrix. The tokenizer (`fn tokenize`) **splits on every
non-word boundary (`\w` = alphanumeric or `_`) and keeps tokens of length ≥ 2** —
matching sklearn's default token pattern `r"(?u)\b\w\w+\b"` (REQ-2 SHIPPED,
#1217). There is no n-gram support,
no custom `token_pattern`/`tokenizer`/`preprocessor`/`analyzer`, no `stop_words`,
no `strip_accents`, no `vocabulary` param, no `dtype`, no sparse output, no
`HashingVectorizer`, no `get_feature_names_out`, and no PyO3 binding. The `min_df`
(`usize`) / `max_df` (`f64`) typing **cannot express** sklearn's int-or-float
duality on either threshold.

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — default fit_transform: vocabulary sorted, count matrix (multi-char tokens):
python3 -c "from sklearn.feature_extraction.text import CountVectorizer; \
cv = CountVectorizer(); X = cv.fit_transform(['the cat sat','the cat sat on the mat']); \
print(list(cv.get_feature_names_out())); print(X.toarray().tolist()); print(type(X).__name__)"
# -> ['cat', 'mat', 'on', 'sat', 'the']
#    [[1, 0, 0, 1, 1], [1, 1, 1, 1, 2]]
#    csr_matrix
#    ferrolearn: CountVectorizer::new().fit(&docs).vocabulary() == ['cat','mat','on','sat','the'];
#    transform(&docs) is the SAME counts but as a DENSE Array2<f64> (not CSR).

# REQ-2 (SHIPPED, #1217) — default token_pattern drops single-char tokens:
python3 -c "from sklearn.feature_extraction.text import CountVectorizer; \
print(list(CountVectorizer().fit(['foo a bar']).get_feature_names_out()))"
# -> ['bar', 'foo']   (default r'(?u)\b\w\w+\b' requires >=2 word chars; 'a' dropped)
#    ferrolearn now matches: tokenize() drops length-1 tokens and treats '_' as a word
#    char (guards divergence_tokenizer_drops_length_one_tokens / _underscore_is_word_char).
#    Residual: a corpus of ALL length-1 tokens yields an empty vocab where sklearn raises
#    ValueError "empty vocabulary" (REQ-14 #1227).

# REQ-5 — max_df=1 as an INT means "in <=1 document" (count), NOT a proportion:
python3 -c "from sklearn.feature_extraction.text import CountVectorizer; \
print(list(CountVectorizer(max_df=1).fit(['the cat','the dog']).get_feature_names_out()))"
# -> ['cat', 'dog']   ('the' is in 2 docs > 1; max_df=1 as int prunes it)
#    ferrolearn: max_df is f64 only -> max_df=1.0 means "<= 1.0*n_docs" (proportion) and prunes
#    NOTHING. The int/proportion duality is structurally unrepresentable.

# REQ-6 — ngram_range=(1,2) emits word bigrams:
python3 -c "from sklearn.feature_extraction.text import CountVectorizer; \
print(list(CountVectorizer(ngram_range=(1,2)).fit(['the cat sat']).get_feature_names_out()))"
# -> ['cat', 'cat sat', 'sat', 'the', 'the cat']   (ferrolearn: unigrams only)

# REQ-13 — HashingVectorizer: fixed-width hashed features, no vocabulary:
python3 -c "from sklearn.feature_extraction.text import HashingVectorizer; \
print(HashingVectorizer(n_features=8).transform(['the cat sat']).shape)"
# -> (1, 8)   (ferrolearn: no HashingVectorizer)
```

## Requirements

- REQ-1: Default-path `fit` + `transform` — build an **alphabetically-sorted**
  vocabulary of whitespace/word tokens (lowercased) and emit a per-document
  term-count matrix whose counts match sklearn's `CountVectorizer().fit_transform`
  on multi-character tokens (REQ-1 Probe counts).
- REQ-2: sklearn-equivalent default tokenization — the default `token_pattern`
  `r"(?u)\b\w\w+\b"` (`:1161`, `build_tokenizer` `:350`): match **2+ word
  characters**, treat `_` as a word char, drop length-1 tokens. (ferrolearn's
  `fn tokenize` splits on non-alphanumeric and keeps length-1 tokens — DIVERGENT.)
- REQ-3: `binary` count clipping (`fit_transform` `:1374-1375`, `X.data.fill(1)`).
- REQ-4: `lowercase` toggle (preprocessor `:323`; default `True`, `:1157`).
- REQ-5: `max_df`/`min_df` int-vs-float duality — int = absolute document count,
  float = proportion of `n_docs` (`fit_transform` `:1379-1380`), plus the
  `max_doc_count < min_doc_count` validation (`:1381-1382`) and the
  "no terms remain" error (`_limit_features` `:1236-1239`).
- REQ-6: `ngram_range` word n-grams (`_word_ngrams` `:242`, default `(1,1)`).
- REQ-7: `max_features` top-N selection by **corpus term frequency** with the
  documented tie/order semantics of `_limit_features` (`:1222-1227`) +
  `_sort_features` (final alphabetical re-sort, `fit_transform` `:1384-1389`).
- REQ-8: Custom analysis hooks — `tokenizer`, `token_pattern`, `preprocessor`,
  `analyzer` (`"word"`/`"char"`/`"char_wb"`/callable), `strip_accents`
  (`build_analyzer` `:419`, `build_preprocessor` `:323`).
- REQ-9: `stop_words` (`"english"` builtin / list / `None`, `get_stop_words`
  `:370`, applied in `_word_ngrams` `:242`).
- REQ-10: Fixed/user-supplied `vocabulary` param + `dtype` (`_count_vocab`
  `fixed_vocab` path `:1242-1244`; `_parameter_constraints` `:1145,:1147`).
- REQ-11: **Sparse CSR output** — sklearn returns `scipy.sparse.csr_matrix`
  (`_count_vocab` `:1299-1304`), not a dense array.
- REQ-12: `get_feature_names_out` exact contract — return the sorted feature
  array as `ndarray` of `str`, with the fitted-state / `input_features` checks
  (`:1455`). (ferrolearn exposes `vocabulary()` / `vocabulary_map()` but not the
  sklearn-shaped accessor.)
- REQ-13: `HashingVectorizer` (`:562`) — stateless `n_features`-wide signed
  feature hashing, `alternate_sign`, no vocabulary (`transform` `:859`).
- REQ-14: Full constructor parameter surface + `_parameter_constraints`
  (`:1124-1148`) — the sixteen named params, `*`-only kwargs, and the validation
  exception types per R-DEV-2 (`input`, `encoding`, `decode_error`,
  `strip_accents`, `preprocessor`, `tokenizer`, `analyzer`, `token_pattern`,
  `ngram_range`, `vocabulary`, `dtype` are all absent from the ferrolearn ctor).
- REQ-15: PyO3 binding (`import ferrolearn` exposes `CountVectorizer` mirroring
  `import sklearn`) — the project boundary consumer.

## Acceptance criteria

- AC-1 (REQ-1): `CountVectorizer::new().fit(&['the cat sat','the cat sat on the
  mat']).vocabulary()` equals `['cat','mat','on','sat','the']` and
  `transform(&docs)` equals the REQ-1 Probe matrix (cell-for-cell, dense). Pinned
  by an oracle-grounded `#[test]`.
- AC-2 (REQ-2): `CountVectorizer::new().fit(&['a b c d e f'])` matches the
  REQ-2 Probe — i.e. yields an **empty vocabulary** (error/empty), NOT a 6-term
  (or `max_features`-truncated) vocab. Today it keeps the single chars (DIVERGENT;
  see `test_count_vectorizer_max_features`).
- AC-3 (REQ-3): `binary(true)` clips every nonzero count to 1.0.
- AC-4 (REQ-4): `lowercase(false)` keeps `"Hello"`/`"hello"` as two terms.
- AC-5 (REQ-5): `max_df=1` (an **int** count) prunes a term present in 2 docs
  (REQ-5 Probe); `min_df=0.5` (a **float** proportion) keeps terms in ≥ 50% of
  docs. Neither is expressible today (`min_df: usize`, `max_df: f64`).
- AC-6 (REQ-6): `ngram_range=(1,2)` on `['the cat sat']` yields the 5-gram vocab
  of the REQ-6 Probe.
- AC-7 (REQ-7): `max_features=k` keeps the k highest-corpus-frequency terms,
  re-sorted alphabetically, matching sklearn on a tie-bearing corpus.
- AC-11 (REQ-11): the output is a CSR sparse structure (or an explicit sparse
  type), not a dense `Array2<f64>`.
- AC-12 (REQ-12): a `get_feature_names_out()`-shaped accessor returns the sorted
  term array and errors when unfitted.
- AC-13 (REQ-13): a `HashingVectorizer` with `n_features=8` transforms
  `['the cat sat']` to width 8 (REQ-13 Probe).
- AC-15 (REQ-15): `python3 -c "import ferrolearn; ferrolearn.feature_extraction.CountVectorizer"`
  resolves and `.fit_transform` matches `sklearn` on the REQ-1 Probe.

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (default fit + transform, sorted vocab, count matrix) | SHIPPED (scoped: multi-char tokens, dense) | impl `pub fn fit in count_vectorizer.rs` builds df counts then `vocab.sort()` (alphabetical) and `FittedCountVectorizer::transform` accumulates `matrix[[i, col]] += 1.0` into `Array2::<f64>::zeros((n_docs, n_vocab))`. Mirrors sklearn `_count_vocab` (`text.py:1242-1305`) + `_sort_features`-via-alphabetical-sort (`fit_transform` `:1384-1389`). On the REQ-1 Probe corpus the vocab `['cat','mat','on','sat','the']` and counts `[[1,0,0,1,1],[1,1,1,1,2]]` match the oracle **for tokens of length ≥ 2** (every token in that corpus). Non-test consumer: crate re-export `pub use count_vectorizer::{CountVectorizer, FittedCountVectorizer};` (`ferrolearn-preprocess/src/lib.rs` line 141) + the "Text processing re-exports" index — the boundary public API of the crate, grandfathered under R-DEFER-1/S5 (existing pub API across prior commits; boundary estimator type). Verification: REQ-1 Probe vs `vocabulary()`/`transform`; `cargo test -p ferrolearn-preprocess` (`test_count_vectorizer_basic`, `test_count_vectorizer_lowercase`). **Caveat: output is DENSE `Array2<f64>`, not CSR (REQ-11). Tokenization now matches sklearn's default token_pattern (REQ-2 SHIPPED, #1217). The `tests/divergence_count_vectorizer.rs` guards are oracle-grounded (R-CHAR-3); the older in-module `#[test]`s remain hand-written except `test_count_vectorizer_max_features`, rewritten against the live oracle.** |
| REQ-2 (sklearn default token_pattern — drop length-1 tokens) | SHIPPED (closed #1217) | impl `fn tokenize in count_vectorizer.rs` now does `text.split(\|c\| !(c.is_alphanumeric() \|\| c == '_')).filter(\|s\| !s.is_empty() && s.chars().count() >= 2)` — `_` is a word char (not a split boundary) and length-1 tokens are dropped, the faithful Rust analog of sklearn's default `token_pattern=r"(?u)\b\w\w+\b"` (`text.py:1161`, `build_tokenizer:350`); `char::is_alphanumeric` is Unicode-aware like `(?u)\w`. Non-test consumer: crate re-export (lib.rs line 141). Verification (R-CHAR-3, live oracle): `['foo a bar']` → `['bar','foo']` (drops length-1 `a`); `['a_b cd']` → `['a_b','cd']` (`a_b` one token); guards in `tests/divergence_count_vectorizer.rs` (`divergence_tokenizer_drops_length_one_tokens`, `divergence_tokenizer_underscore_is_word_char`) PASS; `cargo test -p ferrolearn-preprocess` green. **Residual (REQ-14, #1227): when ALL tokens are length-1 the vocab is empty — sklearn raises an empty-vocabulary `ValueError` (`text.py:1236-1239`) which ferrolearn does not yet emit.** |
| REQ-3 (binary clipping) | SHIPPED | impl `FittedCountVectorizer::transform in count_vectorizer.rs` — `if self.binary { matrix[[i, col]] = 1.0 } else { matrix[[i, col]] += 1.0 }`, mirroring `X.data.fill(1)` (`text.py:1374-1375`). Non-test consumer: same crate re-export (lib.rs line 141). Verification: `cargo test -p ferrolearn-preprocess` `test_count_vectorizer_binary`. |
| REQ-4 (lowercase toggle) | SHIPPED | impl `fn tokenize in count_vectorizer.rs` — `if lowercase { doc.to_lowercase() } else { doc.to_string() }`, threaded from `CountVectorizer::lowercase` (default `true`, mirroring `:1157` / preprocessor `:323`). Non-test consumer: crate re-export (lib.rs line 141). Verification: `cargo test -p ferrolearn-preprocess` `test_count_vectorizer_lowercase`, `test_count_vectorizer_no_lowercase`. **Caveat: Rust `str::to_lowercase` is full-Unicode lowercasing; sklearn lowercases via Python `str.lower()` — agreement on non-ASCII not pinned by oracle.** |
| REQ-5 (max_df/min_df int-vs-float duality + cross-validation) | NOT-STARTED | open prereq blocker #1219 (residual). **The `.ceil()` rounding divergence is now FIXED (closed #1218):** the filter is `(*count as f64) <= self.max_df * n_docs as f64` (no rounding), mirroring sklearn `max_doc_count = max_df * n_doc` (`text.py:1379`) + `dfs <= high` (`_limit_features:1219`); guard `divergence_max_df_float_threshold_no_ceil` PASSES. **Residual (#1219):** `min_df: usize` is always an absolute count and `max_df: f64` is always a proportion; sklearn treats an **int as a count and a float as a proportion on BOTH** thresholds (`:1379-1380`); ferrolearn cannot express `max_df=2` (int) or `min_df=0.5` (float). Also missing: the `max_doc_count < min_doc_count` error (`:1381-1382`) and the "no terms remain after pruning" error (`:1236-1239`). The `max_df ∈ (0,1]` validation is incompatible with sklearn's int-or-float interval. |
| REQ-6 (ngram_range word n-grams) | NOT-STARTED | open prereq blocker #1220. No `ngram_range` field; `tokenize` emits unigrams only; no `_word_ngrams` analog (`text.py:242`). REQ-6 Probe (`['the cat sat']` → 5 grams) unrepresentable. |
| REQ-7 (max_features top-N + tie/sort semantics) | SHIPPED (scoped) | impl `pub fn fit in count_vectorizer.rs` — when `vocab.len() > max_f`, recompute corpus `total_freq`, `sort_by(\|a,b\| fb.cmp(fa).then_with(\|\| a.cmp(b)))`, `truncate(max_f)`, then `vocab.sort()` to restore alphabetical indexing. Mirrors `_limit_features` top-`limit` by `(-tfs).argsort()` (`text.py:1222-1227`) + final `_sort_features` (`:1384-1389`). Non-test consumer: crate re-export (lib.rs line 141). Verification: `cargo test -p ferrolearn-preprocess` `test_count_vectorizer_max_features`. **Caveat: sklearn applies `_sort_features` BEFORE `_limit_features` when `max_features` is set (`:1383-1387`), so its tie-break among equal-frequency terms follows the pre-sorted (alphabetical) index order; ferrolearn ties on `a.cmp(b)` directly — agreement on tie-heavy corpora is NOT oracle-pinned and is a critic candidate. Also scoped by REQ-2 tokenization divergence.** |
| REQ-8 (tokenizer / token_pattern / preprocessor / analyzer / strip_accents) | NOT-STARTED | open prereq blocker #1221. None of the analysis hooks exist: no `tokenizer`, `token_pattern`, `preprocessor`, `analyzer` (`"word"`/`"char"`/`"char_wb"`/callable), or `strip_accents` (`build_analyzer:419`, `build_preprocessor:323`). Tokenization is hard-coded in `fn tokenize`. |
| REQ-9 (stop_words) | NOT-STARTED | open prereq blocker #1222. No `stop_words` field, no English stop-list, no `get_stop_words` (`text.py:370`) / `_word_ngrams` stop filtering (`:242`). |
| REQ-10 (fixed vocabulary param + dtype) | NOT-STARTED | open prereq blocker #1223. No `vocabulary` ctor param (sklearn `_count_vocab` fixed-vocab path `:1242-1244`) and no `dtype` (`:1147`); `FittedCountVectorizer` exposes `vocabulary()`/`vocabulary_map()` but the vocabulary cannot be supplied by the user. |
| REQ-11 (sparse CSR output) | NOT-STARTED | open prereq blocker #1224. `transform` returns a dense `Array2<f64>` (`Array2::<f64>::zeros((n_docs, n_vocab))`); sklearn returns `scipy.sparse.csr_matrix` (`_count_vocab:1299-1304`). No `sprs`-based sparse path. |
| REQ-12 (get_feature_names_out contract) | NOT-STARTED | open prereq blocker #1225. `vocabulary()` returns `&[String]` (sorted) and `vocabulary_map()` returns `&HashMap`, but there is no `get_feature_names_out`-shaped accessor with the fitted-state / `input_features` checks (`text.py:1455`). |
| REQ-13 (HashingVectorizer) | NOT-STARTED | open prereq blocker #1226. No `HashingVectorizer` type anywhere in the crate; sklearn `class HashingVectorizer` (`text.py:562`, `transform:859`) with `n_features`/`alternate_sign`/`norm` is absent (REQ-13 Probe → width 8). |
| REQ-14 (ctor surface + _parameter_constraints) | NOT-STARTED | open prereq blocker #1227. The ferrolearn ctor exposes 5 params (`max_features`, `min_df`, `max_df`, `binary`, `lowercase`); sklearn's 16 (`input`, `encoding`, `decode_error`, `strip_accents`, `preprocessor`, `tokenizer`, `stop_words`, `token_pattern`, `ngram_range`, `analyzer`, `vocabulary`, `dtype` + the 4 present) and `_parameter_constraints` validation / exception types (`:1124-1148`) are absent (R-DEV-2). Note `fit` validates `max_df ∈ (0,1]` only — incompatible with sklearn's int-or-float interval. |
| REQ-15 (PyO3 binding) | NOT-STARTED | open prereq blocker #1228. No `RsCountVectorizer` `#[pyclass]` in `ferrolearn-python/src/transformers.rs`; `import ferrolearn` cannot expose this vectorizer (boundary consumer per R-DEFER-1). |

## Architecture

**ferrolearn (existing).** Two structs in `count_vectorizer.rs`. The unfitted
`CountVectorizer` holds five public fields (`max_features: Option<usize>`,
`min_df: usize`, `max_df: f64`, `binary: bool`, `lowercase: bool`) with
builder-style setters and a `Default`. `fit(&[String]) -> Result<FittedCount­
Vectorizer>` (1) errors on empty corpus and on `max_df ∉ (0,1]`; (2) builds a
`HashMap<String, usize>` of **document** frequencies (unique tokens per doc via a
per-doc `HashSet`); (3) filters by `count >= min_df && (count as f64) <= max_df *
n_docs` (no rounding — #1218 fix); (4) sorts the survivors alphabetically; (5) if `max_features` is set
and exceeded, recomputes corpus term frequencies, sorts by descending frequency
(alphabetical tie-break), truncates, and re-sorts alphabetically; (6) builds a
term→index map. `FittedCountVectorizer` holds the `vocabulary: HashMap`, the
`sorted_terms: Vec<String>`, and the `binary`/`lowercase` flags; `transform`
accumulates a dense `Array2<f64>` of shape `(n_docs, n_vocab)`, ignoring
out-of-vocabulary tokens (test `test_count_vectorizer_unseen_tokens`). Tokenizing
is the free `fn tokenize(doc, lowercase)`: optional `to_lowercase()` then
`split(|c| !(c.is_alphanumeric() || c == '_')).filter(|s| !s.is_empty() &&
s.chars().count() >= 2)` (#1217 fix — `_` is a word char, length-1 tokens dropped,
mirroring `r"(?u)\b\w\w+\b"`).

**sklearn (target contract).** `CountVectorizer(_VectorizerMixin, BaseEstimator)`
(`text.py:929`). The analysis pipeline is assembled by `build_analyzer`
(`:419`): `decode` → `build_preprocessor` (`strip_accents` + `lowercase`,
`:323`) → `build_tokenizer` (regex `token_pattern`, default `r"(?u)\b\w\w+\b"`,
`:350`) → `_word_ngrams` (stop-word drop + n-gram join, `:242`). `_count_vocab`
(`:1242`) streams documents through `analyze(doc)`, accumulating a CSR matrix
with a `defaultdict`-assigned vocabulary. `fit_transform` (`:1327`) then: fills
`X.data` with 1 if `binary`; computes `max_doc_count`/`min_doc_count` from the
**int-or-float** thresholds (`:1379-1380`); validates `max < min` (`:1381`);
`_sort_features` then `_limit_features` (`:1383-1389`) — top `limit` terms by
column sum (`-tfs.argsort()`, `:1222-1227`), erroring if none remain
(`:1236-1239`). `get_feature_names_out` (`:1455`) returns the sorted term array.
`HashingVectorizer` (`:562`) is the vocabulary-free hashing analog.

**The structural gaps.** With REQ-2 (tokenizer, #1217) and the REQ-5 ceil
sub-divergence (#1218) now FIXED, the load-bearing remainders are:
(a) **df-threshold typing** (REQ-5 residual, #1219): `min_df: usize` /
`max_df: f64` cannot model sklearn's int-count-vs-float-proportion duality on
each threshold; the `max_df ∈ (0,1]` validation actively rejects sklearn-valid
int `max_df`.
(b) **dense vs sparse** (REQ-11): the `Array2<f64>` output is not the CSR contract.
The richer analysis surface (n-grams, custom analyzers, stop words, fixed
vocabulary, `HashingVectorizer`) and the PyO3 binding are wholesale absent.

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-3, REQ-4, REQ-7):

```bash
# Oracle (REQ-1 Probe) — default fit_transform on a multi-char corpus:
python3 -c "from sklearn.feature_extraction.text import CountVectorizer; \
cv = CountVectorizer(); X = cv.fit_transform(['the cat sat','the cat sat on the mat']); \
print(list(cv.get_feature_names_out())); print(X.toarray().tolist())"
#   -> ['cat', 'mat', 'on', 'sat', 'the'] ; [[1,0,0,1,1],[1,1,1,1,2]]
# ferrolearn equivalent: CountVectorizer::new().fit(&docs).vocabulary() / .transform(&docs)

# Crate gauntlet:
cargo test -p ferrolearn-preprocess        # incl. test_count_vectorizer_basic, _binary,
                                           # _lowercase, _no_lowercase, _max_features, _min_df,
                                           # _max_df, _empty_corpus, _transform_empty, _unseen_tokens
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s exercise only the default lowercase/whitespace path and
are **hand-written, not oracle-grounded** (expected values are not derived from a
live sklearn call) — the critic should add an oracle-pinned `#[test]` matching
the REQ-1 Probe (R-CHAR-3), and a guard pinning the REQ-2 tokenization
divergence (the `max_features` test currently asserts sklearn-incompatible
behavior). No currently-green command establishes any of REQ-2, REQ-5, REQ-6,
REQ-8..REQ-15.

## Blockers

Two divergences were FIXED this iteration; the rest remain open `-l blocker`
issues referenced by the REQ status table:

- #1217 — REQ-2 (CLOSED, fixed): default tokenizer kept length-1 tokens and
  split on `_`; now mirrors `r"(?u)\b\w\w+\b"` (drops length-1, `_` is a word
  char).
- #1218 — REQ-5 ceil sub-divergence (CLOSED, fixed): `max_df` float threshold
  used `.ceil()`; now compares `(count as f64) <= max_df * n_docs` (no rounding).
- #1219 — REQ-5 residual (OPEN): `min_df: usize` / `max_df: f64` cannot express
  the int-count-vs-float-proportion duality; missing `max < min` and
  "no terms remain" errors; `max_df ∈ (0,1]` validation rejects int `max_df`.
- #1220 — REQ-6: no `ngram_range` / `_word_ngrams`.
- #1221 — REQ-8: no `tokenizer`/`token_pattern`/`preprocessor`/`analyzer`/
  `strip_accents` hooks.
- #1222 — REQ-9: no `stop_words` / English stop-list.
- #1223 — REQ-10: no user-supplied `vocabulary` param, no `dtype`.
- #1224 — REQ-11: dense `Array2<f64>` output, not CSR (`sprs`) sparse.
- #1225 — REQ-12: no `get_feature_names_out`-shaped accessor with
  fitted-state / `input_features` checks.
- #1226 — REQ-13: no `HashingVectorizer`.
- #1227 — REQ-14: ctor exposes 5 of 16 params; missing
  `_parameter_constraints` validation / exception types + empty-vocab error
  (R-DEV-2).
- #1228 — REQ-15: no `ferrolearn-python` registration for `CountVectorizer`.
