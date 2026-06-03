# OrdinalEncoder

<!--
tier: 3-component
status: draft
baseline-commit: 92f9847d
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_encoders.py  # class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder) (:1235); __init__(*, categories='auto', dtype=np.float64, handle_unknown='error', unknown_value=None, encoded_missing_value=np.nan, min_frequency=None, max_categories=None) (:1369-1386). _BaseEncoder._fit (:68-184) -> categories_[j] = _unique(Xi) (sorted unique, :99) for any dtype (int/str/object). dtype default np.float64 (:1262-1263) -> OUTPUT IS FLOAT64. handle_unknown in {'error'(default),'use_encoded_value'} (:1265) + unknown_value (:1274-1276); encoded_missing_value=np.nan (:1283-1284); min_frequency/max_categories infrequent (:1289-1315). transform (:1490+) maps each value to its index in categories_ via _encode; handle_unknown='error' raises ValueError("Found unknown categories ... in column N during transform"). inverse_transform, get_feature_names_out, categories_ / n_features_in_ attributes. _parameter_constraints (:1320-1336).
ferrolearn-module: ferrolearn-preprocess/src/ordinal_encoder.rs
parity-ops: OrdinalEncoder
crosslink-issue: 1157
-->

## Summary

scikit-learn's `OrdinalEncoder` (`_encoders.py:1235`) encodes each categorical
feature column as a single integer-valued column, `0..n_categories-1`, where the
integer assigned to a category is its position in the **sorted unique** category
list for that column (`_BaseEncoder._fit` → `categories_[j] = _unique(Xi)`,
`:99`). It accepts any dtype (int/str/object), and by default emits **float64**
output (`dtype=np.float64`, `:1262`). Unknown categories at `transform` raise a
`ValueError` (`handle_unknown='error'`, default); a rich parameter surface
(`categories`, `dtype`, `handle_unknown='use_encoded_value'` + `unknown_value`,
`encoded_missing_value`, `min_frequency`/`max_categories`) plus
`inverse_transform`, `get_feature_names_out`, and `n_features_in_` round out the
estimator.

`ferrolearn-preprocess/src/ordinal_encoder.rs` ships a **faithful String-only
ordinal encoder**: a unit struct `OrdinalEncoder` and `FittedOrdinalEncoder {
categories: Vec<Vec<String>>, category_to_index: Vec<HashMap<String,usize>> }`.
`impl Fit<Array2<String>, ()>` rejects zero rows with
`FerroError::InsufficientSamples`, then per column collects first-seen unique
categories and calls `unique.sort()` (lexicographic byte order, #344) →
`categories_`, building a category→index map. `impl Transform<Array2<String>> for
FittedOrdinalEncoder` maps each string to its `usize` ordinal index
(ncols-mismatch → `ShapeMismatch`; unknown category → `InvalidParameter`),
returning `Array2<usize>`. `FitTransform` fuses fit+transform; the unfitted
`Transform` impl is a supertrait shim that always errors. There is no
numeric/generic-dtype support, no constructor parameters, no `inverse_transform`,
and no PyO3 binding.

**Headline finding (document prominently).** Unlike the simplified
`OneHotEncoder` sibling, this is a *faithful* ordinal encoder on the string path:
`categories_` (sorted-unique per column) and the ordinal **values** match the
sklearn oracle bit-for-bit (Probe 1: `categories_ == [['bird','cat','dog']]`,
values `[[1,2],[2,0],[1,1],[0,2]]`). Rust `String::sort` agrees with
`np.unique` on string columns (byte order: digits < uppercase < lowercase, Probe
2). Empty-fit rejection also **matches** sklearn (both raise; Probe 4) — no
divergence there. The remaining gaps are structural/surface: the **output
container dtype** (`Array2<usize>` vs sklearn's `np.float64`; values equal,
container differs — R-DEV-3, the single most surgical divergence), String-only
input (no numeric/mixed dtype), and the entire absent parameter/feature surface
(`dtype`, `handle_unknown='use_encoded_value'`, `encoded_missing_value`,
`categories`, infrequent-category folding, `inverse_transform`,
`get_feature_names_out`/`n_features_in_`, ctor + `_parameter_constraints`, PyO3,
ferray substrate).

## Probes (live sklearn oracle, 1.5.2; run from /tmp)

```bash
# PROBE 1 (REQ-1, REQ-3) — string ordinal VALUE-match + categories_ + dtype:
python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
r=OrdinalEncoder().fit_transform([['cat','small'],['dog','large'],['cat','medium'],['bird','small']]); \
print('values:', r.tolist(), 'dtype:', r.dtype); \
import numpy as np; e=OrdinalEncoder().fit([['cat'],['dog'],['cat'],['bird']]); \
print('categories_:', [c.tolist() for c in e.categories_])"
# -> values: [[1.0, 2.0], [2.0, 0.0], [1.0, 1.0], [0.0, 2.0]]  dtype: float64
#    categories_: [['bird', 'cat', 'dog']]
#    ferrolearn: same ordinal VALUES as integers ([[1,2],[2,0],[1,1],[0,2]]),
#    same sorted categories_; container dtype usize vs float64 (REQ-3 / R-DEV-3).

# PROBE 2 (REQ-1) — lexicographic String::sort == np.unique on string dtype:
python3 -c "import numpy as np; \
print(np.unique(['Banana','apple','Apple','banana','10','2','1','Z','a']).tolist())"
# -> ['1', '10', '2', 'Apple', 'Banana', 'Z', 'a', 'apple', 'banana']
#    (byte order: digits < uppercase < lowercase) == Rust String::sort. MATCH.

# PROBE 3 (REQ-2) — unknown category rejected (handle_unknown='error', default):
python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
e=OrdinalEncoder().fit([['cat'],['dog']]); e.transform([['fish']])"
# -> ValueError: Found unknown categories ['fish'] in column 0 during transform
#    ferrolearn: Err(InvalidParameter{reason:"unknown category \"fish\" in column 0"})
#    -- both REJECT; error TYPE maps to FerroError, message nuance (R-DEV-2).

# PROBE 4 (REQ-1) — empty fit: sklearn REJECTS 0 samples (like OneHotEncoder):
python3 -c "import numpy as np; from sklearn.preprocessing import OrdinalEncoder; \
OrdinalEncoder().fit(np.empty((0,2),dtype=object))"
# -> ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum of 1 is required.
#    ferrolearn: Err(InsufficientSamples{required:1, actual:0}) -- MATCH (both reject).

# PROBE 5 (REQ-3) — default output dtype is float64:
python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
r=OrdinalEncoder().fit_transform([['a'],['b']]); print(repr(r), r.dtype)"
# -> array([[0.], [1.]]) float64
#    ferrolearn: Array2<usize> [[0],[1]] -- VALUES equal, container dtype differs (R-DEV-3).
```

## Requirements

- REQ-1: String ordinal fit — for each column, `categories_` = sorted-unique
  categories and a category→ordinal-index map, mirroring sklearn
  `_BaseEncoder._fit` → `categories_[j] = _unique(Xi)` (`:99`) on the string path.
  Rust `String::sort` (lexicographic byte order) matches `np.unique` on string
  dtype (Probe 2). Includes `categories()` / `n_features()` introspection of
  `categories_` / `n_features_in_`. Zero-row input is rejected, matching
  sklearn's `check_array` minimum-samples rejection (Probe 4).
- REQ-2: `transform` (ordinal map) + `fit_transform` — map each category to its
  index in `categories_`, mirroring sklearn `transform` (`_encode` per column,
  `:1490+`), with ncols-mismatch and **unknown-category** rejection under the
  default `handle_unknown='error'` (sklearn raises `ValueError`, Probe 3). The
  encoded ordinal VALUES match the oracle (Probe 1).
- REQ-3: Output dtype + `dtype` parameter — sklearn emits `float64` by default
  and exposes `dtype` (`:1262`, Probe 5); ferrolearn returns `Array2<usize>`. The
  ordinal VALUES are equal; only the output container dtype diverges
  (R-DEV-3 output-contract divergence) and the `dtype` ctor param is absent.
- REQ-4: Numeric / mixed-dtype input — sklearn accepts any dtype (int/str/object)
  and sorts via `np.unique`; ferrolearn is `Array2<String>`-only, so numeric
  columns and the numeric sort order are unrepresentable.
- REQ-5: `handle_unknown='use_encoded_value'` + `unknown_value` — sklearn can map
  unknowns at `transform` to a configurable value instead of raising (`:1265`,
  `:1274`); ferrolearn always rejects unknowns.
- REQ-6: `encoded_missing_value` (NaN) — sklearn encodes missing categories to a
  configurable value, default `np.nan` (`:1283`); ferrolearn has no missing-value
  concept.
- REQ-7: Explicit `categories` parameter — sklearn accepts user-supplied
  per-column category lists instead of `'auto'` (`:1252`); ferrolearn always
  derives categories from the data.
- REQ-8: `min_frequency` / `max_categories` infrequent-category folding — sklearn
  collapses rare categories into a single infrequent bucket (`:1289-1315`);
  ferrolearn has no infrequent-category logic.
- REQ-9: `inverse_transform` — sklearn maps ordinal indices back to categories;
  ferrolearn has no `inverse_transform`.
- REQ-10: `get_feature_names_out` + `n_features_in_` — sklearn exposes both
  (`OneToOneFeatureMixin`); ferrolearn exposes `n_features()` but no
  `get_feature_names_out`.
- REQ-11: Full constructor + `_parameter_constraints` — sklearn's `__init__`
  (`:1369-1386`) and validation surface (`:1320-1336`); ferrolearn's `new()` takes
  no parameters.
- REQ-12: PyO3 binding — `import ferrolearn` exposes `OrdinalEncoder` mirroring
  `import sklearn` (the project-boundary consumer).
- REQ-13: ferray substrate — the encoder computes over `ferray-core` arrays rather
  than `ndarray::Array2<String>` + `std::collections::HashMap` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `OrdinalEncoder::new().fit(&str_2col([("cat","small"),
  ("dog","large"),("cat","medium"),("bird","small")]))` yields `categories()[0]
  == ["bird","cat","dog"]` and `categories()[1] == ["large","medium","small"]`,
  equal to the Probe 1 oracle `categories_`; `n_features() == 2`. Empty-row fit
  returns `Err` matching Probe 4. Pinned by an oracle-grounded `#[test]`
  (R-CHAR-3).
- AC-2 (REQ-2): `fitted.transform(&...)` for the Probe 1 fit equals
  `[[1,2],[2,0],[1,1],[0,2]]` (the oracle values as integers); `fit_transform`
  agrees with separate fit+transform; an unseen category returns `Err`
  (sklearn `ValueError`, Probe 3); a ncols mismatch returns `Err`.
- AC-3 (REQ-3): the encoded VALUES equal the sklearn float64 output cast to
  integer (`[[0.],[1.]]` → `[[0],[1]]`, Probe 5); a `dtype`-parameterized float64
  output container is NOT representable today (`Array2<usize>`).
- AC-4 (REQ-4): a numeric `fit([[10],[2],[1]])` analog yields sorted categories
  `[1,2,10]`. Not representable today (String-only).
- AC-5 (REQ-5): `handle_unknown='use_encoded_value'` maps an unseen category to
  `unknown_value` instead of erroring. Not representable today.
- AC-6 (REQ-6): a missing category encodes to `encoded_missing_value` (NaN).
  Not representable today.
- AC-7 (REQ-7): a user-supplied `categories=[['b','a']]` overrides the
  data-derived sort order. Not representable today.
- AC-8 (REQ-8): `min_frequency` folds rare categories into one bucket. Not
  representable today.
- AC-9 (REQ-9): `inverse_transform([[1,2]])` round-trips to the original
  categories. Not present today.
- AC-10 (REQ-10): `get_feature_names_out()` returns the input feature names and
  `n_features_in_` is exposed. Only `n_features()` exists today.
- AC-11 (REQ-11): the constructor accepts `categories`, `dtype`,
  `handle_unknown`, `unknown_value`, `encoded_missing_value`, `min_frequency`,
  `max_categories` and validates them. `new()` takes none today.
- AC-12 (REQ-12): `python3 -c "import ferrolearn; ferrolearn.preprocessing.OrdinalEncoder"`
  resolves and matches `sklearn` on the Probe 1 case.
- AC-13 (REQ-13): the encoder's owned state/compute uses `ferray-core` (no
  `ndarray` in the compute path).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (string fit → sorted-unique categories_) | SHIPPED | impl `fn fit in <Fit<Array2<String>,()> for OrdinalEncoder> in ordinal_encoder.rs` — per column collects first-seen uniques into a `HashSet<String>`, `unique.sort()` (lexicographic), builds `category_to_index`; rejects 0 rows with `FerroError::InsufficientSamples`. Mirrors sklearn `_BaseEncoder._fit` → `categories_[j] = _unique(Xi)` (`_encoders.py:99`) and `check_array` zero-sample rejection (Probe 4 — both reject). `FittedOrdinalEncoder::categories()` / `n_features()` expose `categories_` / `n_features_in_`. Output equals the oracle: `categories_ == [['bird','cat','dog']]` (Probe 1), and Rust `String::sort` == `np.unique` on string dtype (Probe 2, byte order digits<upper<lower). Non-test consumer: crate re-export `pub use ordinal_encoder::{FittedOrdinalEncoder, OrdinalEncoder};` (`ferrolearn-preprocess/src/lib.rs` line 121) — the boundary public API, grandfathered under R-DEFER-1 (the estimator type IS the surface). Verification: `cargo test -p ferrolearn-preprocess --lib ordinal_encoder` (`test_ordinal_encoder_basic`, `test_single_column`, `test_lexicographic_order`, `test_insufficient_samples_error`, `test_n_features`; 9 passed). |
| REQ-2 (transform + fit_transform, ordinal values + unknown rejection) | SHIPPED | impl `fn transform in <Transform<Array2<String>> for FittedOrdinalEncoder>` — `self.category_to_index[j].get(cat)` per cell, ncols-mismatch → `ShapeMismatch`, unknown → `FerroError::InvalidParameter`; mirrors sklearn `transform` (`_encode` per column, `:1490+`) with `handle_unknown='error'` raising `ValueError` (Probe 3 — both reject). impl `fn fit_transform in <FitTransform<Array2<String>> for OrdinalEncoder>` does `self.fit(x,&())?.transform(x)`. Encoded VALUES equal the oracle: `[[1,2],[2,0],[1,1],[0,2]]` (Probe 1, integer-equal to sklearn's float64 values). Error TYPE maps to `FerroError`; message nuance acceptable (R-DEV-2 — see REQ-3 for the dtype gap, REQ-5 for `use_encoded_value`). Non-test consumer: crate re-export (lib.rs line 121). Verification: `cargo test -p ferrolearn-preprocess --lib ordinal_encoder` (`test_ordinal_encoder_basic`, `test_fit_transform_equivalence`, `test_unknown_category_error`, `test_shape_mismatch_error`, `test_unfitted_transform_error`; 9 passed). |
| REQ-3 (output dtype float64 + `dtype` param) | NOT-STARTED | open prereq blocker #1158. `transform` returns `Array2<usize>` (`type Output = Array2<usize>`); sklearn defaults to `float64` (`:1262`, Probe 5: `array([[0.],[1.]])`) and exposes a `dtype` ctor param. The ordinal VALUES are equal — only the output container dtype diverges (R-DEV-3 output-contract divergence). **Single most surgical divergence**, but structural (the `Output` associated type + a `dtype` param). |
| REQ-4 (numeric / mixed-dtype input) | NOT-STARTED | open prereq blocker #1159. The encoder is `Array2<String>`-only (`impl Fit<Array2<String>, ()>`, `Transform<Array2<String>>`); sklearn accepts int/str/object and sorts via `np.unique`. Numeric columns and the numeric sort order (`[10,2,1]→[1,2,10]`) are structurally unrepresentable today (R-DEV-3). |
| REQ-5 (handle_unknown='use_encoded_value' + unknown_value) | NOT-STARTED | open prereq blocker #1160. `transform` always errors on unknowns (`None => Err(InvalidParameter ...)`); sklearn's `handle_unknown='use_encoded_value'` + `unknown_value` (`:1265`, `:1274`) maps them to a configurable value instead. No such mode in ferrolearn. |
| REQ-6 (encoded_missing_value / NaN) | NOT-STARTED | open prereq blocker #1161. No missing-value concept; sklearn's `encoded_missing_value` (default `np.nan`, `:1283`) is absent. |
| REQ-7 (explicit `categories` param) | NOT-STARTED | open prereq blocker #1162. `fit` always derives categories from data (`'auto'`); sklearn accepts user-supplied per-column lists (`:1252`). No ctor param. |
| REQ-8 (min_frequency / max_categories infrequent folding) | NOT-STARTED | open prereq blocker #1163. No infrequent-category logic; sklearn collapses rare categories (`:1289-1315`). |
| REQ-9 (inverse_transform) | NOT-STARTED | open prereq blocker #1164. No `inverse_transform` method on `FittedOrdinalEncoder`; sklearn maps ordinal indices back to categories. |
| REQ-10 (get_feature_names_out + n_features_in_) | NOT-STARTED | open prereq blocker #1165. `n_features()` exists, but no `get_feature_names_out` (sklearn `OneToOneFeatureMixin`) and no named-feature exposure. |
| REQ-11 (full ctor + _parameter_constraints) | NOT-STARTED | open prereq blocker #1166. `new()` takes no parameters; sklearn's `__init__` (`:1369-1386`) accepts seven params with `_parameter_constraints` validation (`:1320-1336`). |
| REQ-12 (PyO3 binding) | NOT-STARTED | open prereq blocker #1167. No `ferrolearn-python` registration of `OrdinalEncoder` (no PyO3 in the module); `import ferrolearn` cannot expose it (boundary consumer per R-DEFER-1). |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1168. State/compute use `ndarray::Array2<String>` + `std::collections::HashMap`, not `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** Two types in `ordinal_encoder.rs`. The unfitted unit
struct `OrdinalEncoder` (with `new()` and `Default`) carries no state. `impl
Fit<Array2<String>, ()> for OrdinalEncoder` (`fit`) rejects zero rows
(`InsufficientSamples`), then for each column collects first-seen unique
categories via a `HashSet<String>` insert-guard, calls `Vec<String>::sort`
(lexicographic byte order — the `#344` comment notes this matches sklearn), and
builds `category_to_index: HashMap<String,usize>`, returning `FittedOrdinalEncoder
{ categories, category_to_index }`. `FittedOrdinalEncoder` exposes `categories()
-> &[Vec<String>]` and `n_features() -> usize`. `impl Transform<Array2<String>>
for FittedOrdinalEncoder` (`transform`) checks ncols (mismatch →
`ShapeMismatch`), then per cell looks up `category_to_index[j].get(cat)` (unknown
→ `InvalidParameter`), writing the index into an `Array2<usize>`. A second `impl
Transform<Array2<String>> for OrdinalEncoder` is the supertrait shim required by
`FitTransform: Transform` — it always errors. `impl FitTransform<Array2<String>>
for OrdinalEncoder` (`fit_transform`) fits then transforms. The domain is fixed:
input `String`, output `usize`.

**sklearn (target contract).** `OrdinalEncoder(OneToOneFeatureMixin,
_BaseEncoder)` (`:1235`). `__init__` (`:1369-1386`) takes `categories='auto'`,
`dtype=np.float64`, `handle_unknown='error'`, `unknown_value=None`,
`encoded_missing_value=np.nan`, `min_frequency=None`, `max_categories=None`, with
`_parameter_constraints` (`:1320-1336`). `fit` delegates to `_BaseEncoder._fit`
(`:68-184`): `categories_[j] = _unique(Xi)` (`np.unique`-sorted, `:99`) for any
dtype, with infrequent-category folding when `min_frequency`/`max_categories` are
set. `transform` (`:1490+`) encodes each value to its index in `categories_` via
`_encode`, raising `ValueError("Found unknown categories ... in column N during
transform")` under `handle_unknown='error'` or substituting `unknown_value` under
`'use_encoded_value'`, and applies `encoded_missing_value` for missing entries;
output is cast to `dtype` (float64 by default). `inverse_transform` maps codes
back to categories, and `get_feature_names_out` / `n_features_in_` come from
`OneToOneFeatureMixin` / `_BaseEncoder`.

**The structural gap.** On the non-empty string path the two coincide exactly —
sorted-unique `categories_`, the forward category→ordinal map, the
unknown-category rejection, and zero-row rejection all match the oracle
(REQ-1/REQ-2 SHIPPED; Probes 1-4). What differs is: the **output container dtype**
(ferrolearn `Array2<usize>` vs sklearn `float64` — REQ-3, the most surgical gap,
values are already equal); the **input dtype** (String-only vs any dtype — REQ-4);
and the entire absent parameter/feature surface — `handle_unknown='use_encoded_value'`
+ `unknown_value` (REQ-5), `encoded_missing_value` (REQ-6), explicit `categories`
(REQ-7), infrequent-category folding (REQ-8), `inverse_transform` (REQ-9),
`get_feature_names_out` (REQ-10), the full constructor (REQ-11), the PyO3 binding
(REQ-12), and the ferray substrate (REQ-13).

## Verification

Commands establishing the SHIPPED claims (REQ-1/REQ-2):

```bash
# Oracle (run from /tmp) — the string path ferrolearn must match:
python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
r=OrdinalEncoder().fit_transform([['cat','small'],['dog','large'],['cat','medium'],['bird','small']]); \
print('values:', r.tolist()); \
import numpy as np; e=OrdinalEncoder().fit([['cat'],['dog'],['cat'],['bird']]); \
print('categories_:', [c.tolist() for c in e.categories_])"
#   -> values: [[1.0,2.0],[2.0,0.0],[1.0,1.0],[0.0,2.0]]  categories_: [['bird','cat','dog']]
python3 -c "import numpy as np; \
print(np.unique(['Banana','apple','Apple','banana','10','2','1']).tolist())"
#   -> ['1','10','2','Apple','Banana','apple','banana']  (== Rust String::sort)
python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
e=OrdinalEncoder().fit([['cat'],['dog']]); e.transform([['fish']])"
#   -> ValueError: Found unknown categories ['fish'] in column 0 during transform

# Crate gauntlet:
cargo test -p ferrolearn-preprocess --lib ordinal_encoder
#   -> 9 passed; 0 failed (test_ordinal_encoder_basic, test_fit_transform_equivalence,
#      test_unknown_category_error, test_shape_mismatch_error, test_insufficient_samples_error,
#      test_unfitted_transform_error, test_single_column, test_n_features, test_lexicographic_order)
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The nine in-module `#[test]`s are green (`0 failed`) and exercise the REQ-1/REQ-2
string path. To fully satisfy R-CHAR-3 the critic should add oracle-pinned guards
whose expected values are the Probe 1 live-sklearn outputs (the in-module
expectations are hand-written, though they coincide with the oracle). No
currently-green command establishes REQ-3..REQ-13.

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns
`#`-numbers, replacing the `#<...>` placeholders in the REQ status table):

- #1158 — REQ-3: `transform` returns `Array2<usize>`; sklearn defaults to
  `float64` and exposes a `dtype` param (`:1262`, Probe 5). Values equal, container
  differs (R-DEV-3). **Most fixable / surgical divergence** — change the `Output`
  associated type (and add a `dtype` selector) without touching the encode logic.
- #1159 — REQ-4: `Array2<String>`-only; no numeric/mixed-dtype input or
  the `np.unique` numeric sort order (R-DEV-3).
- #1160 — REQ-5: no `handle_unknown='use_encoded_value'` + `unknown_value`
  mode; unknowns always error (`:1265`, `:1274`).
- #1161 — REQ-6: no `encoded_missing_value` / NaN handling (`:1283`).
- #1162 — REQ-7: no explicit `categories` param; always `'auto'` (`:1252`).
- #1163 — REQ-8: no `min_frequency` / `max_categories` infrequent
  folding (`:1289-1315`).
- #1164 — REQ-9: no `inverse_transform`.
- #1165 — REQ-10: no `get_feature_names_out`; `n_features_in_` only via
  `n_features()`.
- #1166 — REQ-11: `new()` takes no parameters; missing the seven-param
  `__init__` + `_parameter_constraints` (`:1369-1386`, `:1320-1336`).
- #1167 — REQ-12: no `ferrolearn-python` registration of `OrdinalEncoder`.
- #1168 — REQ-13: state/compute on `ndarray` + `HashMap`, not ferray
  (R-SUBSTRATE-1/2).
