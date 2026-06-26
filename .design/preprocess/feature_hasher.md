# FeatureHasher

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 2af9b762
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/feature_extraction/__init__.py  # __all__ includes FeatureHasher
  - sklearn/feature_extraction/_hash.py     # class FeatureHasher; __init__; fit; transform
ferrolearn-module: ferrolearn-preprocess/src/feature_hasher.rs
parity-ops: FeatureHasher
crosslink-issue: none
-->

## Summary

scikit-learn's `FeatureHasher` implements the hashing trick for symbolic
features. It accepts `input_type="dict"`, `"pair"`, or `"string"`, hashes feature
names with signed 32-bit MurmurHash3 seed 0, maps columns by
`abs(hash) % n_features`, optionally flips signs via `alternate_sign`, and
returns a sparse CSR matrix. It is stateless; `fit` validates parameters.

`ferrolearn-preprocess/src/feature_hasher.rs` ships a scoped dense analogue:
`FeatureHasher` plus `FeatureHasherInputType`, with `transform_dict`,
`transform_pairs`, and `transform_strings` returning `Array2<f64>`. The
hash/sign/value behavior is pinned against live sklearn 1.5.2 oracles. Residual
gaps remain for sparse CSR output, dtype control, Python estimator protocol,
metadata routing, byte-string handling, categorical dict values, and exact
exception classes/messages.

## Probes

```bash
python3 - <<'PY'
from sklearn.feature_extraction import FeatureHasher
cases = [
    ("dict_altfalse", FeatureHasher(n_features=8, alternate_sign=False, input_type="dict"), [{"cat": 2.0, "dog": 1.0}]),
    ("dict_alttrue", FeatureHasher(n_features=8, alternate_sign=True, input_type="dict"), [{"cat": 2.0, "dog": 1.0}]),
    ("pair_altfalse", FeatureHasher(n_features=8, alternate_sign=False, input_type="pair"), [[("cat", 2.0), ("dog", 1.0)]]),
    ("string_altfalse", FeatureHasher(n_features=8, alternate_sign=False, input_type="string"), [["cat", "cat", "dog"]]),
    ("string_alttrue", FeatureHasher(n_features=8, alternate_sign=True, input_type="string"), [["the", "cat", "sat"]]),
]
for name, fh, data in cases:
    X = fh.transform(data)
    print(name, X.shape, X.toarray().tolist())
PY
# dict_altfalse -> [[0,0,0,0,0,1,0,2]]
# dict_alttrue  -> [[0,0,0,0,0,-1,0,2]]
# pair_altfalse -> [[0,0,0,0,0,1,0,2]]
# string_altfalse -> [[0,0,0,0,0,1,0,2]]
# string_alttrue -> [[0,0,0,0,1,0,-1,1]]
```

## Requirements

- REQ-1: `FeatureHasher` is a public crate-root type corresponding to
  `sklearn.feature_extraction.FeatureHasher`.
- REQ-2: Dict input hashes `HashMap<String, f64>` feature values with sklearn's
  MurmurHash3 column and `alternate_sign` semantics.
- REQ-3: Pair input hashes duplicate-capable `(String, f64)` feature pairs with
  the same semantics.
- REQ-4: String input hashes feature names with implied value `1.0`.
- REQ-5: Empty corpus errors, while empty sample rows produce all-zero rows.
- REQ-6: Sparse CSR output, dtype selection, Python estimator protocol, metadata
  routing, byte strings, and non-numeric categorical values match sklearn.

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 public name | SHIPPED | `pub use feature_hasher::{FeatureHasher, FeatureHasherInputType};` in `ferrolearn-preprocess/src/lib.rs`; boundary guard `tests/divergence_lib.rs`; surface inventory entry. |
| REQ-2 dict input | SHIPPED scoped | `transform_dict` accumulates `HashMap<String, f64>` values into dense columns via shared MurmurHash3 helpers. `tests/divergence_feature_hasher.rs::dict_input_unsigned_and_signed_modes_match_sklearn_oracles` pins sklearn rows. |
| REQ-3 pair input | SHIPPED scoped | `transform_pairs` accepts `Vec<(String, f64)>` per sample and sums duplicates/collisions. Oracle guard `pair_input_matches_dict_oracle`. |
| REQ-4 string input | SHIPPED scoped | `transform_strings` applies implied value 1. Oracle guards `string_input_uses_implied_unit_counts` and `string_input_signed_mode_matches_hashing_vectorizer_oracle_tokens`. |
| REQ-5 empty input behavior | SHIPPED scoped | `empty_output` rejects zero samples and allows empty rows; guard `empty_sample_row_is_allowed_but_empty_corpus_errors`. Error type is Rust `FerroError`, not sklearn `ValueError`. |
| REQ-6 full sklearn contract | NOT-STARTED | Dense `Array2<f64>` instead of CSR; no dtype parameter, PyO3 class, metadata routing, byte-string pass-through, or categorical string-value expansion such as dict value `"x"` becoming `feature=x`. |

## Verification

```bash
cargo test -p ferrolearn-preprocess --test divergence_feature_hasher
cargo test -p ferrolearn-preprocess --test divergence_lib
cargo test -p ferrolearn-preprocess --test api_proof api_proof_text
cargo test -p ferrolearn-preprocess --test conformance_surface_coverage
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```
