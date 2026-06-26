# DictVectorizer

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 31903385
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/feature_extraction/__init__.py       # __all__ includes DictVectorizer
  - sklearn/feature_extraction/_dict_vectorizer.py # class DictVectorizer; fit; fit_transform; transform; inverse_transform; get_feature_names_out; restrict
ferrolearn-module: ferrolearn-preprocess/src/dict_vectorizer.rs
parity-ops: DictVectorizer
crosslink-issue: none
-->

## Summary

scikit-learn's `DictVectorizer` turns feature-value mappings into vectors. Numeric
values keep their original feature name, string values become one-hot
`feature=value` columns, iterable string values are counted, missing features
transform to zero, and unseen transform-time features are ignored. sklearn can
return sparse CSR or dense arrays and also supports dtype control, arbitrary
Python mapping objects, `restrict`, metadata routing, and exact Python exception
types.

`ferrolearn-preprocess/src/dict_vectorizer.rs` ships a scoped dense analogue:
`DictVectorizer`, `FittedDictVectorizer`, and `DictValue` over
`HashMap<String, DictValue>` samples. It pins sorted feature names, dense numeric
values, string categorical one-hot values, iterable string counts, transform
ignore behavior, empty-row behavior, `None` as NaN, and inverse transform against
live sklearn 1.5.2 oracles. Residual gaps remain for sparse CSR, dtype, `sort=False`,
`restrict`, arbitrary Python object keys, PyO3, metadata routing, and exact
exception classes/messages.

## Probes

```bash
python3 - <<'PY'
from sklearn.feature_extraction import DictVectorizer
cases = [
    ("numeric", [{"foo": 1.0, "bar": 2.0}, {"foo": 3.0, "baz": 1.0}]),
    ("categorical", [
        {"city": "Dubai", "temperature": 33.0},
        {"city": "London", "temperature": 12.0},
        {"city": "San Francisco", "temperature": 18.0},
    ]),
]
for name, data in cases:
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(data)
    print(name, dv.get_feature_names_out().tolist(), X.tolist())
PY
# numeric -> ['bar', 'baz', 'foo'] ; [[2,0,1],[0,1,3]]
# categorical -> ['city=Dubai','city=London','city=San Francisco','temperature']
#                [[1,0,0,33],[0,1,0,12],[0,0,1,18]]
```

## Requirements

- REQ-1: `DictVectorizer` is a public crate-root type corresponding to
  `sklearn.feature_extraction.DictVectorizer`.
- REQ-2: Numeric mapping values produce dense columns named by the original key,
  sorted by feature name by default.
- REQ-3: String values become one-hot categorical feature names using the
  separator (`feature=value` by default).
- REQ-4: Iterable string values count repeated categorical values.
- REQ-5: Transform ignores unseen feature names and unseen categorical values.
- REQ-6: Empty sample rows are allowed, empty sample sequences error.
- REQ-7: `inverse_transform` returns non-zero constructed feature names.
- REQ-8: Sparse CSR, dtype control, `sort=False`, `restrict`, PyO3, metadata
  routing, arbitrary Python object keys, and exact Python error ABI match sklearn.

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 public name | SHIPPED | `pub use dict_vectorizer::{DictValue, DictVectorizer, FittedDictVectorizer};` in `ferrolearn-preprocess/src/lib.rs`; boundary guard `tests/divergence_lib.rs`; surface inventory entry. |
| REQ-2 numeric mappings | SHIPPED scoped | `DictValue::Number` keeps the original key and stores the numeric value. Oracle guard `numeric_mappings_match_sklearn_oracle`. |
| REQ-3 string categorical values | SHIPPED scoped | `DictValue::Text` creates `feature{separator}value` with value 1. Oracle guards `string_categorical_values_match_sklearn_oracle` and `custom_separator_matches_sklearn_oracle`. |
| REQ-4 iterable strings | SHIPPED scoped | `DictValue::Texts` emits one entry per string and sums repeats. Oracle guard `iterable_string_values_are_counted`. |
| REQ-5 transform unseen ignore | SHIPPED scoped | `FittedDictVectorizer::transform` only writes entries present in fitted `vocabulary`. Oracle guard `transform_ignores_unseen_keys_and_categories`. |
| REQ-6 empty behavior / None | SHIPPED scoped | Empty row yields shape `(1, 0)`; empty corpus errors; `DictValue::None` maps to NaN. Oracle guards `empty_row_is_allowed_but_empty_corpus_errors` and `none_values_follow_sklearn_nan_path`. |
| REQ-7 inverse transform | SHIPPED scoped | `inverse_transform` returns non-zero constructed feature names and values. Oracle guard `inverse_transform_returns_non_zero_constructed_feature_names`. |
| REQ-8 full sklearn contract | NOT-STARTED | Dense `Array2<f64>` only; no sparse CSR, dtype, `sort=False`, `restrict`, arbitrary Python object keys, PyO3, metadata routing, or exact sklearn exception ABI. |

## Verification

```bash
cargo test -p ferrolearn-preprocess --test divergence_dict_vectorizer
cargo test -p ferrolearn-preprocess --test divergence_lib
cargo test -p ferrolearn-preprocess --test api_proof api_proof_text
cargo test -p ferrolearn-preprocess --test conformance_surface_coverage
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```
