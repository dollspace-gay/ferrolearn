# OneHotEncoder

<!--
tier: 3-component
status: draft
baseline-commit: e53ef60f
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_encoders.py  # class OneHotEncoder(_BaseEncoder) (:458). __init__(*, categories='auto', drop=None, sparse_output=True, dtype=np.float64, handle_unknown='error', min_frequency=None, max_categories=None, feature_name_combiner='concat') (:743-762); _parameter_constraints (:728-741). _BaseEncoder._fit (:68-184) sets self.categories_ = _unique(Xi) per column (:99) — sorted UNIQUE actual values, ANY dtype (str/object/number), NOT max+1. _BaseEncoder._transform (:186-) builds X_int via _encode against categories_; handle_unknown='error' raises ValueError "Found unknown categories ... during transform". OneHotEncoder.fit (:958), transform (:985) — returns scipy CSR when sparse_output=True (DEFAULT, :531/:748) else dense ndarray. inverse_transform (:1068), get_feature_names_out (:1187). drop {None,'first','if_binary',array} (:498-516); handle_unknown {'error','ignore','infrequent_if_exist'} (:541); min_frequency/max_categories infrequent grouping (:566-). Fitted attr: categories_ (list of arrays of actual unique values), drop_idx_, infrequent_categories_, n_features_in_, feature_names_in_.
ferrolearn-module: ferrolearn-preprocess/src/one_hot_encoder.rs
parity-ops: OneHotEncoder
crosslink-issue: 1148
-->

## Summary

scikit-learn's `OneHotEncoder` (`_encoders.py:458`) encodes categorical features
as a one-hot array. `fit(X)` learns, per column, `categories_[j] = _unique(Xi)` —
the **sorted unique actual values** of that column (`_BaseEncoder._fit:99`),
accepting **any dtype** (strings, objects, integers, floats). `transform(X)` emits
one binary column per learned category and, by default (`sparse_output=True`,
`:531`/`:748`), returns a **scipy CSR sparse matrix**; with `sparse_output=False`
it returns a dense ndarray. It carries a wide constructor surface (`categories`,
`drop`, `sparse_output`, `dtype`, `handle_unknown`, `min_frequency`,
`max_categories`, `feature_name_combiner`, `:743-762`) and `inverse_transform` /
`get_feature_names_out`.

`ferrolearn-preprocess/src/one_hot_encoder.rs` ships a **simplified, integer-only,
dense** encoder that diverges **structurally** from sklearn. Input is
`Array2<usize>` (non-negative integer indices only); output is a **dense**
`Array2<F>`. The unit struct `OneHotEncoder<F>` (PhantomData) fits into
`FittedOneHotEncoder<F> { n_categories: Vec<usize> }`. The headline divergence:
`fit` sets `n_categories[j] = max(col_j) + 1` (`Fit::fit` in `one_hot_encoder.rs`),
**assuming categories are the contiguous integers `0,1,…,max`** — it does **not**
learn the unique set. For a contiguous `0..max` integer column ferrolearn's dense
output coincides exactly with sklearn `sparse_output=False`; for a non-contiguous
column the two **diverge structurally** (different column count, embedded
always-zero columns). `transform` returns dense `Array2<F>` via a per-column offset
layout; an out-of-range category (`cat >= n_categories[j]`) → `InvalidParameter`,
an ncols mismatch → `ShapeMismatch`. There is no `categories_` attribute, no sparse
output, no `drop` / `handle_unknown` / infrequent-category surface, no
`inverse_transform`, no `get_feature_names_out`, and no PyO3 binding.

**Honest framing (R-HONEST-3).** This is an underclaim-not-overclaim audit. Only
the **in-regime contiguous-`0..max` integer** dense path matches sklearn, and even
then only against the **non-default** `sparse_output=False`. Everything that makes
sklearn's encoder general (unique-set `categories_`, arbitrary/non-contiguous/string
categories, sparse-by-default output, the full param surface, inverse/feature-name
APIs) is **NOT-STARTED** and structural — rewriting `fit` to learn `_unique`
changes the entire encoding, so these are NOT minimal-fixable. This is a
**verify-and-document** iteration: there is no single low-risk minimal divergence
to hand the fixer.

## Probes (live sklearn oracle, 1.5.2; run from /tmp)

```bash
# PROBE 1 (REQ-3 headline) — NON-CONTIGUOUS integer column: sklearn learns the
# unique set, NOT max+1. Categories {2,5,9} → 3 output columns (one per value).
python3 -c "import numpy as np; from sklearn.preprocessing import OneHotEncoder; \
e=OneHotEncoder(sparse_output=False).fit([[2],[5],[9]]); \
print(e.categories_, e.transform([[2],[5],[9]]).shape)"
# -> [array([2, 5, 9])] (3, 3)      (categories_ = _unique(col), _fit:99)
#    ferrolearn fit([[2],[5],[9]]): n_categories=[10], output shape (3,10)
#    (cat 2→col 2, cat 5→col 5, cat 9→col 9; 7 always-zero columns) — STRUCTURAL DIVERGENCE

# PROBE 2 (REQ-1) — CONTIGUOUS 0..max integer column, dense: ferrolearn DOES match.
python3 -c "from sklearn.preprocessing import OneHotEncoder; \
print(OneHotEncoder(sparse_output=False).fit_transform([[0],[1],[2],[1]]).tolist())"
# -> [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0]]
#    ferrolearn fit_transform([[0],[1],[2],[1]]) -> identical dense Array2<f64> — MATCH

# PROBE 3 (REQ-2) — sparse is the DEFAULT: transform returns a scipy CSR matrix.
python3 -c "from sklearn.preprocessing import OneHotEncoder; \
print(type(OneHotEncoder().fit_transform([[0],[1],[2]])))"
# -> <class 'scipy.sparse._csr.csr_matrix'>   (sparse_output=True default, :531/:748)
#    ferrolearn output is always DENSE Array2<F> — DIVERGENCE (no sparse analog)

# PROBE 4 (REQ-4) — handle_unknown='error' (default): unseen category at transform
# raises ValueError, via categories_ membership (NOT a max comparison).
python3 -c "from sklearn.preprocessing import OneHotEncoder; \
e=OneHotEncoder(sparse_output=False).fit([[0],[1]]); e.transform([[2]])"
# -> ValueError: Found unknown categories [np.int64(2)] in column 0 during transform
#    ferrolearn transform([[2]]) after fit([[0],[1]]) -> Err(InvalidParameter
#    "category 2 exceeds max seen during fitting (1)") — TYPE-analog error, different
#    mechanism (cat >= max+1, not set-membership) and message.

# PROBE 5 (REQ-3) — string categories: sklearn encodes strings; ferrolearn cannot
# (Array2<usize> only).
python3 -c "from sklearn.preprocessing import OneHotEncoder; \
e=OneHotEncoder(sparse_output=False).fit([['cat'],['dog'],['cat']]); \
print(e.categories_, e.transform([['cat'],['dog']]).tolist())"
# -> [array(['cat','dog'], dtype=object)] [[1.0,0.0],[0.0,1.0]]
#    ferrolearn: no string input path — structurally unrepresentable.

# PROBE 6 (REQ-6) — get_feature_names_out exists on the fitted estimator.
python3 -c "from sklearn.preprocessing import OneHotEncoder; \
e=OneHotEncoder().fit([[0,5],[1,7]]); print(e.get_feature_names_out().tolist())"
# -> ['x0_0', 'x0_1', 'x1_5', 'x1_7']   (:1187)
#    ferrolearn: no get_feature_names_out.
```

## Requirements

- REQ-1: **Dense one-hot of contiguous `0..max` integer columns.** For an
  `Array2<usize>` whose columns are the contiguous integers `0,1,…,max`, produce a
  dense `Array2<F>` one-hot encoding with a per-column offset layout, matching
  sklearn's `sparse_output=False` dense output (Probe 2). Mirrors
  `_BaseEncoder._transform`'s one-hot construction (`:186-`) **only** under the
  contiguous-integer scope where `max(col)+1` equals `len(_unique(col))`.
- REQ-2: **Sparse-by-default output (`sparse_output=True`).** sklearn `transform`
  returns a scipy CSR matrix by default (`:531`/`:748`, Probe 3); ferrolearn always
  returns a dense `Array2<F>` and has no sparse analog.
- REQ-3: **`categories_` as the sorted unique set (arbitrary / non-contiguous /
  string / float categories).** sklearn `fit` sets `categories_[j] = _unique(Xi)`
  — the actual sorted unique values for any dtype (`_fit:99`, Probes 1 & 5).
  ferrolearn computes `n_categories[j] = max(col)+1`, assuming contiguous
  `0..max` integers; it stores no `categories_` attribute, cannot represent
  non-contiguous integer sets, strings, or floats, and embeds always-zero columns
  for missing intermediate integers (Probe 1). The structural headline divergence.
- REQ-4: **`handle_unknown` parameter and set-membership error contract.** sklearn
  `handle_unknown ∈ {'error'(default),'ignore','infrequent_if_exist'}` (`:541`);
  `'error'` raises `ValueError("Found unknown categories … during transform")` via
  `categories_` membership (Probe 4). ferrolearn errors only on `cat >= max+1`
  (`InvalidParameter`, different mechanism/message), and has no `'ignore'` /
  `'infrequent_if_exist'` modes.
- REQ-5: **`drop` + infrequent-category grouping (`min_frequency` /
  `max_categories`).** sklearn supports `drop ∈ {None,'first','if_binary',array}`
  (`:498-516`) and infrequent grouping (`:566-`). ferrolearn has none of these.
- REQ-6: **`inverse_transform` and `get_feature_names_out`.** sklearn provides
  `inverse_transform` (`:1068`) and `get_feature_names_out` (`:1187`, Probe 6).
  ferrolearn provides neither.
- REQ-7: **Full constructor + `dtype` + `_parameter_constraints` surface.** sklearn's
  keyword-only ctor (`categories`, `drop`, `sparse_output`, `dtype`,
  `handle_unknown`, `min_frequency`, `max_categories`, `feature_name_combiner`,
  `:743-762`) with `_parameter_constraints` validation (`:728-741`). ferrolearn's
  `OneHotEncoder::new()` takes no parameters and validates nothing.
- REQ-8: **PyO3 binding.** `import ferrolearn` exposes `OneHotEncoder` mirroring
  `import sklearn.preprocessing` — the project-boundary consumer.
- REQ-9: **ferray substrate.** The encoder computes over `ferray-core` arrays
  rather than `ndarray::Array2` (R-SUBSTRATE-1/2).

## Acceptance criteria

- AC-1 (REQ-1): `OneHotEncoder::<f64>::new().fit_transform(&array![[0],[1],[2],[1]])`
  equals the Probe 2 oracle `[[1,0,0],[0,1,0],[0,0,1],[0,1,0]]`; multi-column
  contiguous input keeps the per-column offset layout. Pinned by an
  oracle-grounded `#[test]` (R-CHAR-3).
- AC-2 (REQ-2): `transform` can emit a sparse representation when the equivalent
  of `sparse_output=True` is requested; output type matches sklearn's CSR contract
  (Probe 3). Not representable today (always dense).
- AC-3 (REQ-3): fitting `array![[2],[5],[9]]` exposes `categories_ == [[2,5,9]]`
  and transforms to shape `(3,3)` (Probe 1), not `n_categories=[10]` / shape
  `(3,10)`; a string column fits its sorted-unique labels (Probe 5). Not
  representable today.
- AC-4 (REQ-4): unseen-category transform under the `'error'` default fails via
  `categories_` membership with sklearn's "Found unknown categories … during
  transform" message (Probe 4); `'ignore'` yields an all-zero row block. Today
  ferrolearn errors on `cat >= max+1` with a different message and has no `'ignore'`.
- AC-5 (REQ-5): `drop='first'` removes the first category column per feature;
  `min_frequency` groups infrequent categories. Not representable today.
- AC-6 (REQ-6): `inverse_transform` round-trips a one-hot block back to category
  indices; `get_feature_names_out` returns `['x0_0','x0_1',…]` (Probe 6). Not
  representable today.
- AC-7 (REQ-7): the constructor accepts and validates `categories`, `drop`,
  `sparse_output`, `dtype`, `handle_unknown`, `min_frequency`, `max_categories`
  matching `_parameter_constraints` (`:728-741`). Today `new()` takes nothing.
- AC-8 (REQ-8): `python3 -c "import ferrolearn; ferrolearn.preprocessing.OneHotEncoder"`
  resolves and matches `sklearn` on the Probe 2 contiguous case.
- AC-9 (REQ-9): the encoder's owned state/compute uses `ferray-core` (no `ndarray`
  in the compute path).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (dense one-hot of contiguous 0..max integer columns) | SHIPPED (scoped) | impl `fn transform in <Transform<Array2<usize>> for FittedOneHotEncoder> in one_hot_encoder.rs` — zero-fills `Array2<F>` of width `n_output_features()` then sets `out[[i, col_offset + cat]] = F::one()` per element with a per-column `col_offset`, mirroring `_BaseEncoder._transform` one-hot construction (`_encoders.py:186-`). **Scope caveat:** matches sklearn `sparse_output=False` dense output ONLY when each column is the contiguous integers `0..max`, where `max(col)+1 == len(_unique(col))`; outside that regime it diverges (REQ-3). Output equals the Probe 2 oracle `[[1,0,0],[0,1,0],[0,0,1],[0,1,0]]` exactly. `fit` (`fn fit in <Fit<Array2<usize>,()> for OneHotEncoder>`) sets `n_categories[j]=max(col)+1` and rejects 0 rows (`InsufficientSamples`); `n_categories()` / `n_output_features()` introspect the width. Non-test consumer: crate re-export `pub use one_hot_encoder::{FittedOneHotEncoder, OneHotEncoder};` (`ferrolearn-preprocess/src/lib.rs` line 120) — the boundary public API of the crate, grandfathered under S5/R-DEFER-1 (existing pub API; the estimator type IS the surface). Verification: `cargo test -p ferrolearn-preprocess` (`one_hot_encoder::tests::test_one_hot_single_column`, `test_one_hot_multi_column`, `test_fit_transform_equivalence`) + `tests/oracle_tests.rs::test_one_hot_encoder_oracle` (green). |
| REQ-2 (sparse-by-default output) | NOT-STARTED | open prereq blocker #1149. `transform` always returns a dense `Array2<F>` (`type Output = Array2<F>`); sklearn defaults to `sparse_output=True` → scipy CSR (`:531`/`:748`, Probe 3). No sparse output analog (a ferray sparse type, R-SUBSTRATE). Structural — also gated by REQ-9. |
| REQ-3 (categories_ = sorted unique set) | NOT-STARTED | open prereq blocker #1150. `fit` computes `n_categories[j] = max(col).unwrap_or(0) + 1` (`fn fit in one_hot_encoder.rs`), assuming categories are contiguous `0..max` integers; it stores no `categories_` attribute and cannot represent non-contiguous integer sets, strings, or floats. sklearn sets `categories_[j] = _unique(Xi)` (the actual sorted unique values, `_fit:99`) for any dtype — Probe 1 `[[2],[5],[9]]` → `categories_=[[2,5,9]]`, shape `(3,3)` vs ferrolearn `n_categories=[10]`, shape `(3,10)` with 7 always-zero columns; Probe 5 strings unrepresentable. **The structural headline divergence — NOT minimal-fixable** (rewriting `fit` to use the unique set changes the whole encoding/layout). NOT pinned as a committed failing test (R-DEFER-3: structural gaps are NOT-STARTED blockers without a committed failing test). |
| REQ-4 (handle_unknown + set-membership error) | NOT-STARTED | open prereq blocker #1151. `transform` errors only when `cat >= n_categories[j]` → `FerroError::InvalidParameter` ("category {cat} exceeds max seen during fitting ({max})") — a TYPE-analog of sklearn's `ValueError` but a different MECHANISM (max comparison, not `categories_` membership) and MESSAGE; e.g. fit `[[0],[1]]` then transform `[[2]]` errors in BOTH, but ferrolearn would silently accept an in-range-but-unseen value if categories were non-contiguous (Probe 4). sklearn additionally supports `handle_unknown='ignore'` (all-zero block) and `'infrequent_if_exist'` (`:541`); ferrolearn has no such modes (R-DEV-2). Note: ncols-mismatch → `ShapeMismatch` (`fn transform`) is a sound shape guard but distinct from this category contract. |
| REQ-5 (drop + infrequent grouping) | NOT-STARTED | open prereq blocker #1152. No `drop` parameter (`{None,'first','if_binary',array}`, `:498-516`) and no `min_frequency`/`max_categories` infrequent-category grouping (`:566-`). ferrolearn always retains every category column with no dropping or grouping. |
| REQ-6 (inverse_transform + get_feature_names_out) | NOT-STARTED | open prereq blocker #1153. No `inverse_transform` (sklearn `:1068`) and no `get_feature_names_out` (sklearn `:1187`, Probe 6 → `['x0_0','x0_1','x1_5','x1_7']`). `transform_1d` is a single-column convenience, not an inverse. |
| REQ-7 (ctor + dtype + _parameter_constraints) | NOT-STARTED | open prereq blocker #1154. `OneHotEncoder::new()` takes no parameters and validates nothing; sklearn's keyword-only ctor exposes `categories`, `drop`, `sparse_output`, `dtype`, `handle_unknown`, `min_frequency`, `max_categories`, `feature_name_combiner` (`:743-762`) with `_parameter_constraints` validation (`:728-741`). The output dtype `F` is fixed by the generic, not a `dtype` parameter. |
| REQ-8 (PyO3 binding) | NOT-STARTED | open prereq blocker #1155. No `ferrolearn-python` registration of `OneHotEncoder` (no PyO3 binding); `import ferrolearn` cannot expose it (boundary consumer per R-DEFER-1). |
| REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker #1156. State/compute use `ndarray::Array2<usize>` (input) and `ndarray::Array2<F>` (output), not `ferray-core` (R-SUBSTRATE-1/2); a sparse output (REQ-2) would route through ferray's sparse analog. |

## Architecture

**ferrolearn (existing).** Two types in `one_hot_encoder.rs`. The unfitted unit
struct `OneHotEncoder<F>` holds only `PhantomData<F>` (with `new()` and `Default`).
`impl Fit<Array2<usize>, ()> for OneHotEncoder<F>` (`fit`) rejects 0 rows with
`FerroError::InsufficientSamples` (`required:1, actual:0`), then for each column
sets `n_categories[j] = max(col).unwrap_or(0) + 1`, returning `FittedOneHotEncoder<F>
{ n_categories: Vec<usize> }`. `FittedOneHotEncoder` exposes `n_categories() ->
&[usize]` and `n_output_features() -> usize` (= Σ n_categories). `impl
Transform<Array2<usize>> for FittedOneHotEncoder<F>` (`transform`) checks
`x.ncols() == n_categories.len()` (else `ShapeMismatch`), allocates a zero
`Array2<F>` of width `n_output_features()`, and walks each column with a running
`col_offset`, setting `out[[i, col_offset + cat]] = F::one()` and erroring with
`InvalidParameter` when `cat >= n_categories[j]`. A second `impl
Transform<Array2<usize>> for OneHotEncoder<F>` is a supertrait shim (always errors,
"encoder must be fitted first"). `impl FitTransform<Array2<usize>>` fits then
transforms. `transform_1d(&[usize])` is a single-column convenience that wraps the
slice into one column and dispatches to `transform`. Domain: input is
**non-negative integer indices only** (`Array2<usize>`), output is a **dense**
`Array2<F>`.

**sklearn (target contract).** `OneHotEncoder(_BaseEncoder)` (`:458`).
`_BaseEncoder._fit` (`:68-184`) sets `self.categories_ = [_unique(Xi) for each
column]` (`:99`) — the actual sorted unique values, for any dtype (string/object/
number), via `_check_X`'s column-by-column dtype-preserving validation (`:31-66`).
`_BaseEncoder._transform` (`:186-`) builds an integer code matrix `X_int` by
`_encode`-ing each column against `categories_` (raising `ValueError("Found unknown
categories … during transform")` under `handle_unknown='error'`), then `OneHotEncoder.
transform` (`:985`) expands it to one-hot, returning a **scipy CSR** matrix when
`sparse_output=True` (the default, `:531`/`:748`) else a dense ndarray of `dtype`.
The ctor (`:743-762`) is keyword-only with `_parameter_constraints` (`:728-741`).
Fitted attributes include `categories_` (list of arrays), `drop_idx_`,
`infrequent_categories_`, `n_features_in_`, `feature_names_in_`. It also provides
`inverse_transform` (`:1068`) and `get_feature_names_out` (`:1187`).

**The structural gap.** Only the **contiguous-`0..max` integer, dense** regime
coincides (REQ-1 SHIPPED, scoped): when `max(col)+1 == len(_unique(col))` and
sklearn is asked for `sparse_output=False`, the two dense matrices are identical
(Probe 2). Everything else diverges and is **structural, not minimal-fixable**:
the `max+1` heuristic vs the unique-set `categories_` (REQ-3 — the headline; it
re-encodes non-contiguous columns with embedded always-zero columns, Probe 1, and
cannot represent strings/floats, Probe 5); dense vs sparse-by-default output
(REQ-2); the absent param surface, `handle_unknown` modes and set-membership error
contract (REQ-4/REQ-5/REQ-7); the missing `inverse_transform` /
`get_feature_names_out` (REQ-6); the missing PyO3 binding (REQ-8); and the ferray
substrate (REQ-9). Rewriting `fit` to learn `_unique` would change the column
count, layout, and error mechanism wholesale — hence NOT-STARTED blockers rather
than a single fixer-sized diff. There is **no minimal-fixable divergence** this
iteration (verify-and-document).

## Verification

Commands establishing the SHIPPED claim (REQ-1, scoped):

```bash
# Oracle (run from /tmp) — the contiguous-0..max dense path ferrolearn must match:
python3 -c "from sklearn.preprocessing import OneHotEncoder; \
print(OneHotEncoder(sparse_output=False).fit_transform([[0],[1],[2],[1]]).tolist())"
#   -> [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0]]

# Oracle showing the REQ-3 divergence regime (NOT matched — documented, not pinned):
python3 -c "from sklearn.preprocessing import OneHotEncoder; \
e=OneHotEncoder(sparse_output=False).fit([[2],[5],[9]]); \
print(e.categories_, e.transform([[2],[5],[9]]).shape)"
#   -> [array([2,5,9])] (3,3)   (ferrolearn: n_categories=[10], shape (3,10))

# Crate gauntlet:
cargo test -p ferrolearn-preprocess          # incl. one_hot_encoder::tests::
                                             #   test_one_hot_single_column,
                                             #   test_one_hot_multi_column,
                                             #   test_out_of_range_category_error,
                                             #   test_fit_transform_equivalence,
                                             #   test_shape_mismatch_error
                                             # + tests/oracle_tests.rs::
                                             #   test_one_hot_encoder_oracle
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The in-module `#[test]`s and `tests/oracle_tests.rs::test_one_hot_encoder_oracle`
are green (`0 failed`) and exercise the REQ-1 contiguous-integer dense path. To
fully satisfy R-CHAR-3 the critic should add an **oracle-grounded GREEN guard**
whose expected value is the Probe 2 live-sklearn output (`OneHotEncoder(
sparse_output=False).fit_transform([[0],[1],[2],[1]])`), confirming the in-regime
contiguous case bit-for-bit. **Do NOT** add committed failing tests for the
structural NOT-STARTED divergences (REQ-2 sparse, REQ-3 categories_ unique-set):
per R-DEFER-3 a committed failing test must be fixed this iteration, and these are
NOT-STARTED structural blockers. No currently-green command establishes
REQ-2..REQ-9.

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns the
`#`-numbers; placeholders below); reference them in the REQ status table:

- #1149 — REQ-2: dense-only output; no `sparse_output=True` CSR analog
  (sklearn default, `:531`/`:748`, Probe 3). Gated by a ferray sparse type (REQ-9).
- #1150 — REQ-3: `fit` uses `max(col)+1`, not `categories_=_unique(col)`
  (`_fit:99`); cannot represent non-contiguous integer sets (Probe 1) or
  string/float categories (Probe 5). **The structural headline; not minimal-fixable**
  (rewriting `fit` changes the whole encoding) — NOT pinned as a failing test
  (R-DEFER-3).
- #1151 — REQ-4: no `handle_unknown` parameter; the error mechanism is a
  `max+1` comparison (`InvalidParameter`), not `categories_` membership
  (`ValueError`, Probe 4); no `'ignore'` / `'infrequent_if_exist'` modes (R-DEV-2).
- #1152 — REQ-5: no `drop` (`:498-516`) and no `min_frequency`/`max_categories`
  infrequent-category grouping (`:566-`).
- #1153 — REQ-6: no `inverse_transform` (`:1068`) and no
  `get_feature_names_out` (`:1187`, Probe 6).
- #1154 — REQ-7: `new()` exposes no constructor parameters and no
  `_parameter_constraints` validation (`:743-762`, `:728-741`).
- #1155 — REQ-8: no `ferrolearn-python` registration of `OneHotEncoder`.
- #1156 — REQ-9: state/compute on `ndarray::Array2`, not ferray
  (R-SUBSTRATE-1/2).
