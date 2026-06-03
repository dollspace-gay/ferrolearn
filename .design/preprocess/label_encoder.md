# LabelEncoder

<!--
tier: 3-component
status: draft
baseline-commit: ac356557
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_label.py  # class LabelEncoder(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None) (:34); fit(y): y=column_or_1d(y, warn=True); self.classes_=_unique(y); return self (:84-99) — NO empty rejection (_unique([]) -> empty classes_). fit_transform(y): self.classes_, y = _unique(y, return_inverse=True); return y (:101-116). transform(y): check_is_fitted; y=column_or_1d(y, dtype=classes_.dtype); if _num_samples(y)==0: return np.array([]); return _encode(y, uniques=classes_) (:118-137) — _encode raises ValueError("y contains previously unseen labels: ...") on unseen (utils/_encode.py). inverse_transform(y): check_is_fitted; if _num_samples(y)==0 return np.array([]); diff=setdiff1d(y, arange(len(classes_))); if len(diff): raise ValueError("y contains previously unseen labels: %s") ; return classes_[y] (:139-162). _more_tags X_types ["1dlabels"] (:164-165). _unique (utils/_encode.py) returns np.unique-sorted unique values. Accepts ANY hashable+comparable dtype (numbers/strings/object); 1D via column_or_1d; classes_ is an ndarray sorted by np.unique. NO n_features_in_ / get_feature_names_out (target encoder, auto_wrap_output_keys=None).
ferrolearn-module: ferrolearn-preprocess/src/label_encoder.rs
parity-ops: LabelEncoder
crosslink-issue: 1133
-->

## Summary

scikit-learn's `LabelEncoder` (`_label.py:34`) encodes target labels (`y`, not
`X`) into integers `0..n_classes-1`. `fit(y)` sets `classes_ = _unique(y)` (sorted
unique, `:98`); `transform(y)` maps each label to its index in `classes_` via
`_encode` (`:137`); `inverse_transform(y)` maps indices back to labels
(`:162`); `fit_transform(y)` does both in one `_unique(..., return_inverse=True)`
pass (`:115`). It accepts **any hashable, comparable dtype** (numbers, strings,
object), is strictly 1-D (`column_or_1d`), and exposes a single fitted attribute
`classes_` (an ndarray sorted by `np.unique`). It is a target encoder — it has
**no** `n_features_in_` / `get_feature_names_out` (Probe 8).

`ferrolearn-preprocess/src/label_encoder.rs` ships a **String-only** label
encoder: a unit struct `LabelEncoder` and `FittedLabelEncoder { classes:
Vec<String>, label_to_index: HashMap<String,usize> }`. `impl Fit<Array1<String>,
()>` rejects **empty** input with `FerroError::InsufficientSamples`, else sets
`classes` = sorted-unique labels (Rust `String::sort` = lexicographic byte order)
and builds the label→index map. `FittedLabelEncoder::transform` maps label→`usize`
(unknown label → `FerroError::InvalidParameter` "unknown label");
`inverse_transform(Array1<usize>)` maps index→label (out-of-range → `InvalidParameter`).
`FitTransform` fits then transforms; the unfitted `Transform` impl is a
supertrait shim that always errors. There is no numeric/generic-dtype support and
no PyO3 binding.

**Headline finding (document prominently):** on the **non-empty, string** path,
ferrolearn matches the sklearn oracle exactly — sorted-unique `classes_`,
label→int forward map, int→label inverse, and `fit_transform` (Probes 4b, 5, 6).
**Empty-fit parity (REQ-5) was FIXED this iteration (#1134):** the `if x.is_empty()`
→ `InsufficientSamples` guard was removed so `fit([])` now yields an empty
`FittedLabelEncoder` matching sklearn `_unique([])` (Probe 1). The remaining gaps
are structural
(numeric/generic dtype — String-only can't represent the `[10,2,1]→[1,2,10]`
numeric-sort contract, Probe 4) or surface (PyO3 binding, ferray substrate) plus
two R-DEV-2 error-contract divergences (unseen-label message; unfitted-transform
should be a `NotFittedError` analog, Probe 7).

## Probes (live sklearn oracle, 1.5.2; run from /tmp)

```bash
# PROBE 1 (REQ-5) — empty fit: sklearn returns empty classes_, NO error:
python3 -c "from sklearn.preprocessing import LabelEncoder; \
le=LabelEncoder().fit([]); print(repr(le.classes_))"
# -> array([], dtype=float64)            (_unique([]) -> empty; fit :98)
#    ferrolearn: LabelEncoder.fit(&empty) -> Err(InsufficientSamples) — DIVERGENCE

# PROBE 2 (REQ-3) — empty transform / inverse_transform return empty:
python3 -c "from sklearn.preprocessing import LabelEncoder; \
le=LabelEncoder().fit(['a','b']); print(repr(le.transform([])), repr(le.inverse_transform([])))"
# -> array([], dtype=float64) array([], dtype=float64)   (:134-135, :155-156)
#    ferrolearn: transform(empty)->Ok(empty), inverse_transform(empty)->Ok(empty) — MATCH (value)

# PROBE 3 (REQ-6) — unseen label on transform raises ValueError:
python3 -c "from sklearn.preprocessing import LabelEncoder; \
le=LabelEncoder().fit(['a','b']); le.transform(['u'])"
# -> ValueError: y contains previously unseen labels: np.str_('u')   (_encode, :137)
#    ferrolearn: Err(InvalidParameter{reason:"unknown label \"u\""}) — TYPE analog, MESSAGE differs

# PROBE 3b (REQ-2) — inverse_transform out-of-range raises ValueError:
python3 -c "from sklearn.preprocessing import LabelEncoder; \
le=LabelEncoder().fit(['a','b']); le.inverse_transform([5])"
# -> ValueError: y contains previously unseen labels: [5]            (:159-160)
#    ferrolearn: Err(InvalidParameter{reason:"index 5 is out of range ..."}) — TYPE analog

# PROBE 4 (REQ-4) — numeric labels sort numerically; ferrolearn (String-only) cannot:
python3 -c "from sklearn.preprocessing import LabelEncoder; \
print(LabelEncoder().fit([10,2,1]).classes_.tolist())"
# -> [1, 2, 10]                            (np.unique numeric sort)
python3 -c "from sklearn.preprocessing import LabelEncoder; \
print(LabelEncoder().fit(['10','2','1']).classes_.tolist())"
# -> ['1', '10', '2']                      (the String analog ferrolearn DOES match)

# PROBE 5 (REQ-1) — string fit + transform + inverse round-trip (the SHIPPED path):
python3 -c "from sklearn.preprocessing import LabelEncoder; \
le=LabelEncoder().fit(['cat','dog','cat','bird']); \
print(le.classes_.tolist(), le.transform(['cat','dog','cat','bird']).tolist(), \
le.inverse_transform([1,2,1,0]).tolist())"
# -> ['bird','cat','dog'] [1,2,1,0] ['cat','dog','cat','bird']

# PROBE 6 (REQ-3) — fit_transform:
python3 -c "from sklearn.preprocessing import LabelEncoder; \
print(LabelEncoder().fit_transform(['foo','bar','foo','baz']).tolist())"
# -> [2, 0, 2, 1]                          (bar<baz<foo)

# PROBE 7 (REQ-6) — transform on UNFITTED estimator raises NotFittedError:
python3 -c "from sklearn.preprocessing import LabelEncoder; LabelEncoder().transform(['a'])"
# -> NotFittedError: This LabelEncoder instance is not fitted yet. Call 'fit' ...
#    ferrolearn: unfitted Transform shim -> Err(InvalidParameter ...) — TYPE differs

# PROBE 8 (REQ-7) — LabelEncoder has NO n_features_in_ / get_feature_names_out:
python3 -c "from sklearn.preprocessing import LabelEncoder; le=LabelEncoder().fit(['a','b']); \
print('n_features_in_' in dir(le), 'get_feature_names_out' in dir(le))"
# -> False False                           (target encoder; auto_wrap_output_keys=None)
```

## Requirements

- REQ-1: String fit — `classes_` = sorted-unique labels, built into a label→index
  map, mirroring sklearn `fit(y): self.classes_ = _unique(y)` (`:98`) on the
  string/non-empty path. String sort is lexicographic, matching sklearn's
  `np.unique` on string dtype (Probe 4b: `['10','2','1'] → ['1','10','2']`).
- REQ-2: `inverse_transform` — map integer indices back to labels
  (`classes_[y]`, `:162`), with an out-of-range guard mirroring sklearn's
  `setdiff1d(y, arange(len(classes_)))` → error (`:158-160`, Probe 3b).
- REQ-3: `transform` (label→int) and `fit_transform` (fit then transform) on the
  string/non-empty path, mirroring `transform` `_encode(y, uniques=classes_)`
  (`:137`) and `fit_transform` `_unique(..., return_inverse=True)` (`:115`); plus
  `classes()` / `n_classes()` introspection of `classes_`.
- REQ-4: Numeric / generic hashable+comparable dtype support — sklearn accepts
  any such dtype and sorts by `np.unique` (Probe 4a: `[10,2,1] → [1,2,10]`);
  ferrolearn is `Array1<String>`-only and cannot represent numeric labels or the
  numeric sort order.
- REQ-5: Empty-fit parity — sklearn `fit([])` returns empty `classes_` with NO
  error (`_unique([])`, Probe 1); ferrolearn rejects empty input with
  `FerroError::InsufficientSamples` (the `if x.is_empty()` guard in `fit`).
- REQ-6: Error-contract parity (R-DEV-2) — the unseen-label message ("y contains
  previously unseen labels", `_encode` / `:160`, Probe 3) and the unfitted-transform
  exception **type** (`NotFittedError`, `check_is_fitted` at `:131`, Probe 7);
  ferrolearn uses `FerroError::InvalidParameter` for both with a different message.
- REQ-7: PyO3 binding — `import ferrolearn` exposes `LabelEncoder` mirroring
  `import sklearn` (the project-boundary consumer). NOTE: LabelEncoder has NO
  `n_features_in_` / `get_feature_names_out` (Probe 8), so the binding need not
  expose them.
- REQ-8: ferray substrate — the encoder computes over `ferray-core` arrays rather
  than `ndarray::Array1<String>` + `std::collections::HashMap` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `LabelEncoder::new().fit(&str_arr(["cat","dog","cat","bird"]))`
  yields `classes() == ["bird","cat","dog"]` and `n_classes() == 3`, equal to the
  Probe 5 oracle output; `fit(["10","2","1"])` yields `["1","10","2"]` (Probe 4b).
  Pinned by an oracle-grounded `#[test]` (R-CHAR-3).
- AC-2 (REQ-2): `fitted.inverse_transform(&array![1,2,1,0])` for the Probe 5 fit
  equals `["cat","dog","cat","bird"]`; `inverse_transform(&array![5])` returns
  `Err` (Probe 3b: sklearn `ValueError`).
- AC-3 (REQ-3): `fitted.transform(&["cat","dog","cat","bird"])` equals `[1,2,1,0]`
  (Probe 5); `fit_transform(["foo","bar","foo","baz"])` equals `[2,0,2,1]`
  (Probe 6); `transform(empty)` and `inverse_transform(empty)` return `Ok(empty)`
  (Probe 2).
- AC-4 (REQ-4): a numeric `fit([10,2,1])` analog yields `classes_ == [1,2,10]`
  (Probe 4a). Not representable today (String-only).
- AC-5 (REQ-5): `LabelEncoder::new().fit(&empty)` returns `Ok` with empty
  `classes_` matching sklearn (Probe 1). Today it returns
  `Err(InsufficientSamples)`.
- AC-6 (REQ-6): the unseen-label error message contains "previously unseen
  labels" (Probe 3) and the unfitted-transform error is a `NotFittedError` analog
  (Probe 7). Today both are `InvalidParameter` with different messages.
- AC-7 (REQ-7): `python3 -c "import ferrolearn; ferrolearn.preprocessing.LabelEncoder"`
  resolves and round-trips against `sklearn` on the Probe 5 case.
- AC-8 (REQ-8): the encoder's owned state/compute uses `ferray-core` (no
  `ndarray` in the compute path).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (string fit → sorted-unique classes_) | SHIPPED | impl `fn fit in <Fit<Array1<String>,()> for LabelEncoder> in label_encoder.rs` — collects into a `HashSet<String>`, `unique.sort()` (lexicographic), builds `label_to_index`; non-empty path mirrors sklearn `fit(y): self.classes_ = _unique(y)` (`_label.py:98`). `FittedLabelEncoder::classes()` / `n_classes()` expose `classes_`. Output equals the oracle: `['bird','cat','dog']` (Probe 5) and `['1','10','2']` for string-of-numeric (Probe 4b) bit-for-bit. Non-test consumer: crate re-export `pub use label_encoder::{FittedLabelEncoder, LabelEncoder};` (`ferrolearn-preprocess/src/lib.rs` line 116) — the boundary public API of the crate, grandfathered under S5/R-DEFER-1 (existing pub API; the estimator type IS the surface). Verification: `cargo test -p ferrolearn-preprocess` (`test_label_encoder_basic`, `test_single_class`) + `tests/oracle_tests.rs::test_label_encoder_oracle` (green). |
| REQ-2 (inverse_transform) | SHIPPED | impl `fn inverse_transform in FittedLabelEncoder in label_encoder.rs` — `classes[idx].clone()` per index with an `if idx >= n_classes` out-of-range guard returning `FerroError::InvalidParameter`; mirrors sklearn `return self.classes_[y]` (`:162`) and the `setdiff1d` out-of-range → error guard (`:158-160`, Probe 3b). Round-trip `[1,2,1,0] → ['cat','dog','cat','bird']` matches Probe 5; out-of-range `[5]` errors (sklearn `ValueError`, error TYPE maps to `FerroError`, R-DEV-2 acceptable analog — see REQ-6 for the message gap). Non-test consumer: crate re-export (lib.rs line 116). Verification: `cargo test -p ferrolearn-preprocess` (`test_inverse_transform_roundtrip`, `test_inverse_transform_out_of_range`). |
| REQ-3 (transform + fit_transform) | SHIPPED | impl `fn transform in <Transform<Array1<String>> for FittedLabelEncoder>` — `self.label_to_index.get(label)` per element, mirrors sklearn `_encode(y, uniques=classes_)` (`:137`); impl `fn fit_transform in <FitTransform<Array1<String>> for LabelEncoder>` does `self.fit(x,&())?.transform(x)`, mirrors sklearn `_unique(..., return_inverse=True)` (`:115`). Outputs match the oracle: transform `[1,2,1,0]` (Probe 5), fit_transform `[2,0,2,1]` (Probe 6); empty transform / inverse_transform return `Ok(empty)` matching sklearn's empty-returns-empty (`:134-135`, `:155-156`, Probe 2). Non-test consumer: crate re-export (lib.rs line 116). Verification: `cargo test -p ferrolearn-preprocess` (`test_label_encoder_basic`, `test_fit_transform_equivalence`, `test_unknown_label_error`). |
| REQ-4 (numeric/generic dtype) | NOT-STARTED | open prereq blocker #1135. The encoder is `Array1<String>`-only (`impl Fit<Array1<String>, ()>`, `Transform<Array1<String>>`); there is no numeric or generic hashable+comparable label type. sklearn accepts any such dtype and sorts via `np.unique` (Probe 4a: `[10,2,1] → [1,2,10]`); the String analog gives lexicographic `['1','10','2']`, so the numeric-sort contract is structurally unrepresentable today. Structural divergence (R-DEV-3 label ordering). |
| REQ-5 (empty-fit parity) | SHIPPED | FIXED #1134. The `if x.is_empty()` → `InsufficientSamples` guard was REMOVED from `fit`; empty input now flows through the HashSet→sort→map path yielding an empty `FittedLabelEncoder` (`classes = []`, empty map), matching sklearn `fit([])` → empty `classes_` (`_label.py:98`, `_unique([])`; Probe 1). Verification (acto-critic, live-oracle, green): `divergence_empty_fit_succeeds` + post-empty-fit guards `green_empty_fit_then_empty_transform_ok` / `_transform_unseen_rejected` / `_empty_inverse_ok` / `_inverse_oob_rejected` in `tests/divergence_label_encoder.rs`; in-module `test_empty_fit_yields_empty_classes` (R-HONEST-4: replaced the prior `test_empty_input_error` that pinned the divergent behavior). Two-round critic-verified CLEAN. |
| REQ-6 (error-contract parity, R-DEV-2) | NOT-STARTED | open prereq blocker #1136. (a) Unseen-label transform returns `FerroError::InvalidParameter{reason:"unknown label \"u\""}`; sklearn raises `ValueError("y contains previously unseen labels: ...")` (`_encode`, Probe 3) — error TYPE maps acceptably to `FerroError`, but the MESSAGE/contract differs (R-DEV-2). (b) Unfitted `transform` returns `InvalidParameter`; sklearn's `check_is_fitted` (`:131`) raises `NotFittedError` (Probe 7) — ferrolearn has no `NotFittedError` analog here. |
| REQ-7 (PyO3 binding) | NOT-STARTED | open prereq blocker #1137. No `ferrolearn-python` registration of `LabelEncoder` (grep of `ferrolearn-python/` for `LabelEncoder` is empty); `import ferrolearn` cannot expose it (boundary consumer per R-DEFER-1). Note: LabelEncoder has NO `n_features_in_` / `get_feature_names_out` (Probe 8), so no such attribute marshalling is owed. |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #1138. State/compute use `ndarray::Array1<String>` + `std::collections::HashMap`, not `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** Two types in `label_encoder.rs`. The unfitted unit
struct `LabelEncoder` (with `new()` and `Default`) carries no state. `impl
Fit<Array1<String>, ()> for LabelEncoder` (`fit`) rejects empty input
(`InsufficientSamples`), then collects labels into a `HashSet<String>`, sorts the
result with `Vec<String>::sort` (lexicographic byte order), and builds the
`label_to_index: HashMap<String,usize>`, returning `FittedLabelEncoder { classes,
label_to_index }`. `FittedLabelEncoder` exposes `classes() -> &[String]`,
`n_classes() -> usize`, and `inverse_transform(&Array1<usize>) ->
Result<Array1<String>, FerroError>` (out-of-range index → `InvalidParameter`).
`impl Transform<Array1<String>> for FittedLabelEncoder` (`transform`) maps each
label through `label_to_index` (unknown → `InvalidParameter`). A second `impl
Transform<Array1<String>> for LabelEncoder` is a supertrait shim required by the
`FitTransform: Transform` bound — it always errors. `impl
FitTransform<Array1<String>> for LabelEncoder` (`fit_transform`) fits then
transforms. Generic over nothing: the label domain is fixed to `String`.

**sklearn (target contract).** `LabelEncoder(TransformerMixin, BaseEstimator,
auto_wrap_output_keys=None)` (`:34`). `fit(y)` (`:84-99`) does `y =
column_or_1d(y, warn=True); self.classes_ = _unique(y)` — `_unique`
(`utils/_encode.py`) is the `np.unique`-sorted unique values, with NO
empty-rejection (the source of the REQ-5 divergence). `fit_transform`
(`:101-116`) fuses fit+transform via `_unique(..., return_inverse=True)`.
`transform` (`:118-137`) calls `check_is_fitted` (→ `NotFittedError` if unfit),
coerces to `classes_.dtype`, returns `np.array([])` for empty input, else
`_encode(y, uniques=classes_)` (→ `ValueError` on unseen labels).
`inverse_transform` (`:139-162`) check-is-fitted, returns `[]` for empty, guards
out-of-range via `setdiff1d(y, arange(len(classes_)))` (→ `ValueError`), then
indexes `classes_[y]`. The sole fitted attribute is `classes_` (an ndarray sorted
by `np.unique`); there is intentionally NO `n_features_in_` /
`get_feature_names_out` because it is a **target** encoder (`auto_wrap_output_keys
=None`, Probe 8).

**The structural gap.** On the non-empty string path the two coincide exactly —
sorted-unique `classes_`, the forward label→int map, the int→label inverse, and
`fit_transform` all match the oracle (REQ-1/2/3 SHIPPED). What differs is: the
**empty-fit** behavior (ferrolearn errors, sklearn does not — REQ-5, the minimal
fix); the **label domain** (String-only vs any hashable+comparable, so numeric
sort `[10,2,1]→[1,2,10]` is unrepresentable — REQ-4); the **error contracts**
(unseen-label message + the `NotFittedError` type — REQ-6, R-DEV-2); the missing
PyO3 binding (REQ-7); and the ferray substrate (REQ-8).

## Verification

Commands establishing the SHIPPED claims (REQ-1/2/3):

```bash
# Oracle (run from /tmp) — the string/non-empty path ferrolearn must match:
python3 -c "from sklearn.preprocessing import LabelEncoder; \
le=LabelEncoder().fit(['cat','dog','cat','bird']); \
print(le.classes_.tolist(), le.transform(['cat','dog','cat','bird']).tolist(), \
le.inverse_transform([1,2,1,0]).tolist())"
#   -> ['bird','cat','dog'] [1,2,1,0] ['cat','dog','cat','bird']
python3 -c "from sklearn.preprocessing import LabelEncoder; \
print(LabelEncoder().fit_transform(['foo','bar','foo','baz']).tolist())"
#   -> [2, 0, 2, 1]
python3 -c "from sklearn.preprocessing import LabelEncoder; \
print(LabelEncoder().fit(['10','2','1']).classes_.tolist())"
#   -> ['1', '10', '2']   (lexicographic; the String analog ferrolearn matches)

# Crate gauntlet:
cargo test -p ferrolearn-preprocess          # incl. test_label_encoder_basic,
                                             #       test_inverse_transform_roundtrip,
                                             #       test_fit_transform_equivalence,
                                             #       test_unknown_label_error,
                                             #       test_inverse_transform_out_of_range,
                                             #       test_single_class,
                                             #       oracle_tests::test_label_encoder_oracle
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The in-module `#[test]`s and `tests/oracle_tests.rs::test_label_encoder_oracle`
are green (`0 failed`) and exercise the REQ-1/2/3 string path. To fully satisfy
R-CHAR-3 the critic should add oracle-pinned guards whose expected values are the
Probe 5/6 live-sklearn outputs (the in-module expectations are hand-written). No
currently-green command establishes REQ-4..REQ-8; in particular
`test_empty_input_error` PINS the REQ-5 divergence as *current* behavior (it
asserts `fit(empty).is_err()`, the opposite of sklearn) and must flip when REQ-5
is fixed.

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns
`#`-numbers); reference them in the REQ status table:

- #1135 — REQ-4: `Array1<String>`-only; no numeric/generic hashable+comparable
  label support, so the `np.unique` numeric-sort contract (`[10,2,1]→[1,2,10]`,
  Probe 4a) is unrepresentable (R-DEV-3 label ordering).
- #1134 — REQ-5: `fit` has an `if x.is_empty()` → `InsufficientSamples`
  guard with no sklearn counterpart; sklearn `fit([])` yields empty `classes_`
  with no error (Probe 1, `:98`). **Most fixable minimal divergence** — relax the
  guard to produce an empty `FittedLabelEncoder`.
- #1136 — REQ-6: unseen-label error message diverges from sklearn's "y
  contains previously unseen labels" (`_encode`, Probe 3); unfitted-transform
  returns `InvalidParameter` rather than a `NotFittedError` analog
  (`check_is_fitted`, `:131`, Probe 7) (R-DEV-2).
- #1137 — REQ-7: no `ferrolearn-python` registration of `LabelEncoder`.
- #1138 — REQ-8: state/compute on `ndarray` + `HashMap`, not ferray
  (R-SUBSTRATE-1/2).
```
