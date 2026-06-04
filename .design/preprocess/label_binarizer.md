# LabelBinarizer

<!--
tier: 3-component
status: draft
baseline-commit: af5b0730
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_label.py  # class LabelBinarizer(TransformerMixin, BaseEstimator) (:180); __init__(*, neg_label=0, pos_label=1, sparse_output=False) (:263); fit(y) (:269) — neg_label>=pos_label raise (:283-287), sparse+(pos_label==0 or neg_label!=0) raise (:289-294), y_type_=type_of_target(y) (:296), reject multioutput (:298-301), reject 0 samples (:302-303), classes_=unique_labels(y) (:306); fit_transform=fit(y).transform(y) (:309-329); transform(y) (:331) delegates to label_binarize(y, classes=classes_, pos_label, neg_label, sparse_output) (:357-363); inverse_transform(Y, threshold=None) (:365) — threshold defaults (pos_label+neg_label)/2 (:399-400), multiclass -> _inverse_binarize_multiclass (:402-403), else _inverse_binarize_thresholding (:404-407). Module fn label_binarize(y, *, classes, neg_label=0, pos_label=1, sparse_output=False) (:430): reject 0 samples (:497-498), neg>=pos raise (:499-504), sparse constraint (:506-512), pos_label==0 switch (:514-517), n_classes==1 binary -> all-neg_label single column (:532-538), n_classes>=3 -> y_type="multiclass" (:539-540), unknown labels SILENTLY IGNORED via np.isin/y_seen (:556-559), CSR build (:563), dense toarray + neg_label fill (:575-583), binary collapse to single last column (:592-596). _inverse_binarize_multiclass multiclass branch = classes.take(y.argmax(axis=1), mode="clip") (:641); _inverse_binarize_thresholding STRICT y > threshold (:667), then classes[y.ravel()] (:679). type_of_target/unique_labels drive label-type/multilabel handling; classes_ is an ndarray (int dtype on int input, object otherwise).
ferrolearn-module: ferrolearn-preprocess/src/label_binarizer.rs
parity-ops: LabelBinarizer, label_binarize
crosslink-issue: 1238
-->

## Summary

scikit-learn's `LabelBinarizer` (`_label.py:180`) converts a 1-D vector of class
labels into a **(samples x classes) binary indicator matrix** using a one-vs-rest
scheme, with a special **single-column** encoding for the binary (exactly two
classes) and single-class cases. `__init__(*, neg_label=0, pos_label=1,
sparse_output=False)` (`:263`); `fit(y)` (`:269`) validates `neg_label <
pos_label` (`:283-287`), records `y_type_ = type_of_target(y)` (`:296`) and
`classes_ = unique_labels(y)` (`:306`); `transform(y)` (`:331`) delegates to the
module-level `label_binarize(...)` (`:357-363`), which **silently ignores unknown
labels** (`:556-559`), encodes a **binary** problem as one column with `pos_label`
on the second sorted class (`:531`, `:592-596`), encodes a **single class** as an
all-`neg_label` column (`:532-538`), and one-hot-encodes the multiclass case;
`inverse_transform(Y, threshold=None)` (`:365`) uses `argmax` for multiclass
(`:641`) and a **strict** `Y > threshold` for the binary/thresholding path
(`:667`).

`ferrolearn-preprocess/src/label_binarizer.rs` ships a **`usize`-only**, dense
core: a unit struct `LabelBinarizer` and `FittedLabelBinarizer { classes:
Vec<usize> }`. `impl Fit<Array1<usize>, ()>::fit` rejects empty input with
`FerroError::InsufficientSamples`, else sets `classes` = sorted+deduped unique.
`impl Transform<Array1<usize>>::transform -> Array2<f64>` emits an all-zero column
when `k == 1` (#1240), a single column when `k == 2` (`1.0` on the second class),
and a one-hot `(n, k)` matrix otherwise, **silently ignoring any label not seen
during fit** (#1239, leaving an all-zero row). `FittedLabelBinarizer::inverse_transform`
argmax-decodes the multiclass case and thresholds the binary single column at
**strict `> 0.5`** (#1241). There is no `neg_label`/`pos_label`/`sparse_output`
ctor param, no `label_binarize` free function, no CSR path, no generic label type,
and no PyO3 binding.

**Headline finding (document prominently).** On the **known-label, default
(`neg_label=0`, `pos_label=1`) int path** the multiclass one-hot values (REQ-2),
the binary single-column values (REQ-3), and the multiclass `inverse_transform`
argmax (REQ-7) coincide exactly with the oracle, as does sorted-unique `classes_`
(REQ-1). Three **fixable behavioral divergences** were FIXED this iteration:
(DIV-1, REQ-4, #1239) `transform` of an unknown label now silently leaves an
all-zero row (was an error, `:556-559`); (DIV-2, REQ-5, #1240) the single-class
(`k == 1`) case now emits an all-`0` column (was all-`1.0`, `:532-538`); (DIV-3,
REQ-6, #1241) `inverse_transform` now uses a STRICT `> 0.5` (was `>= 0.5`,
`:667`). The remaining gaps are surface/structural: the
`neg_label`/`pos_label`/`sparse_output`
ctor params (REQ-8/REQ-9), the `label_binarize` free function (REQ-10), the
generic label domain + `type_of_target`/multilabel input handling (REQ-11), and
the PyO3 binding (REQ-12). Unlike `MultiLabelBinarizer`, the empty-input edge is
**consistent**: sklearn `label_binarize` also rejects 0 samples (`:497-498`,
`:302-303`), matching ferrolearn's `InsufficientSamples`.

## Probes (live sklearn oracle, 1.5.2; run from /tmp)

```bash
# PROBE A (REQ-1, REQ-2, REQ-7) — multiclass one-hot + sorted classes_:
python3 -c "from sklearn.preprocessing import LabelBinarizer; \
lb=LabelBinarizer(); print(lb.fit_transform([0,1,2,1]).tolist()); print(lb.classes_.tolist())"
# -> [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
#    [0, 1, 2]
#    ferrolearn: LabelBinarizer::new().fit(&y,&()).classes() == [0,1,2];
#    fitted.transform(&y) is the SAME indicator as a DENSE Array2<f64>;
#    fitted.inverse_transform(&that) round-trips to [0,1,2,1] (argmax).

# PROBE B (REQ-3) — binary (exactly 2 classes) -> SINGLE column, pos_label on 2nd class:
python3 -c "from sklearn.preprocessing import LabelBinarizer; \
print(LabelBinarizer().fit_transform([0,1,0,1]).tolist())"
# -> [[0], [1], [0], [1]]
#    ferrolearn: k==2 branch -> out[[i,0]] = 1.0 iff idx==1 (the 2nd sorted class). MATCHES.

# PROBE C (REQ-4) — UNKNOWN label is SILENTLY IGNORED -> all-zero row, NO error/warning:
python3 -c "from sklearn.preprocessing import LabelBinarizer; \
lb=LabelBinarizer().fit([0,1,2]); print(lb.transform([0,3]).tolist())"
# -> [[1, 0, 0], [0, 0, 0]]   (label 3 not in classes_ -> contributes nothing)
#    ferrolearn: fitted.transform(&array![0,3]) -> Err(InvalidParameter "unknown label 3
#    not seen during fit") — DIVERGENCE (errors instead of leaving an all-zero row).

# PROBE D (REQ-5) — SINGLE class (k==1) -> all-neg_label (default 0) single column:
python3 -c "from sklearn.preprocessing import LabelBinarizer; \
print(LabelBinarizer().fit_transform([5,5,5]).tolist())"
# -> [[0], [0], [0]]   (np.zeros((n,1)); Y += neg_label, :532-538)
#    ferrolearn: k==1 falls into the ELSE/one-hot branch -> Array2::zeros((3,1));
#    out[[i,0]] = 1.0 for the single class -> [[1],[1],[1]] — DIVERGENCE (all-1, not all-0).

# PROBE E (REQ-6) — inverse_transform binary STRICT threshold (> not >=) at exactly 0.5:
python3 -c "import numpy as np; from sklearn.preprocessing import LabelBinarizer; \
lb=LabelBinarizer().fit([0,1]); \
print(lb.inverse_transform(np.array([[0.5]])).tolist(), lb.inverse_transform(np.array([[0.6]])).tolist())"
# -> [0] [1]   (0.5 > 0.5 is FALSE -> class[0]; 0.6 > 0.5 -> class[1], :667)
#    ferrolearn: y[[i,0]] >= 0.5 -> at exactly 0.5 picks classes[1] == 1 — DIVERGENCE
#    (ferrolearn returns class 1 at 0.5; sklearn returns class 0).
```

## Requirements

- REQ-1: `fit` discovers sorted-unique `classes_` — mirror sklearn `classes_ =
  unique_labels(y)` (`:306`) on the all-int label path (PROBE A: `[0,1,2,1] ->
  classes_ == [0,1,2]`).
- REQ-2: `transform` of **known** labels in the **multiclass** (k >= 3) case emits
  the dense one-hot indicator — `Y[i, j] == pos_label` iff `classes_[j] == y[i]`
  (`label_binarize` `:552-563` then `toarray()` `:575-577`); on the known-label
  default path the indicator **values** match the oracle (PROBE A matrix).
- REQ-3: `transform` of the **binary** (exactly 2 classes) case emits a **single
  column** with `pos_label` (default 1) on the **second** sorted class, `neg_label`
  (default 0) on the first (`:531`, `:592-596`, PROBE B). ferrolearn's `k == 2`
  branch matches on the default `pos/neg_label` known-label path.
- REQ-4: `transform` of an **unknown** label — sklearn **silently ignores** it
  (`y_in_classes = np.isin(y, classes); y_seen = y[y_in_classes]`, `:556-559`),
  leaving an all-`neg_label` (all-zero default) row with NO error and NO warning
  (PROBE C). ferrolearn instead returns `FerroError::InvalidParameter` (transform
  `k==2` and `else` branches). HEADLINE behavioral divergence (DIV-1).
- REQ-5: `transform` of a **single class** (k == 1) — sklearn returns an
  all-`neg_label` (default 0) single column (`Y = np.zeros((len(y),1)); Y +=
  neg_label`, `:532-538`, PROBE D). ferrolearn routes k == 1 into the multiclass
  `else` branch and emits an all-`1.0` column. HEADLINE behavioral divergence
  (DIV-2).
- REQ-6: `inverse_transform` **binary thresholding** — sklearn uses a **STRICT**
  `y = (Y > threshold)` (`:667`, default `threshold = (pos_label+neg_label)/2 =
  0.5`, `:399-400`) then `classes[y.ravel()]` (`:679`), so exactly-0.5 maps to
  `classes[0]` (PROBE E). ferrolearn thresholds at `>= 0.5`, so exactly-0.5 maps to
  `classes[1]`. HEADLINE behavioral divergence (DIV-3).
- REQ-7: `inverse_transform` **multiclass** argmax — sklearn `classes.take(
  Y.argmax(axis=1), mode="clip")` (`:641`). ferrolearn's `else` branch computes the
  per-row first-argmax and indexes `classes`, matching on the multiclass path
  (PROBE A round-trip).
- REQ-8: `neg_label` / `pos_label` constructor params + their validation —
  `__init__(*, neg_label=0, pos_label=1, ...)` (`:263`), `neg_label >= pos_label`
  raise (`:283-287`), and the dense `pos_label==0` switch + `neg_label != 0` fill
  (`:514-517`, `:579-583`). ferrolearn has no such params (hardcoded 0/1).
- REQ-9: `sparse_output` CSR output + its constraint — `sparse_output` keeps the
  CSR matrix (`:584-585`), with the `pos_label==0 or neg_label!=0` rejection guard
  (`:289-294`, `:506-512`). ferrolearn always emits a dense `Array2<f64>`; no
  `sparse_output` field, no `sprs` path.
- REQ-10: the module-level `label_binarize(y, *, classes, neg_label=0, pos_label=1,
  sparse_output=False)` free function (`:430`) — the standalone parity op that
  `transform` delegates to (`:357-363`), callable without a fitted estimator.
  Absent in ferrolearn (no free function; logic is inlined into `transform`).
- REQ-11: arbitrary label types (strings, etc.) + `type_of_target` / multilabel-
  indicator input handling — sklearn `classes_` carries object dtype on non-int
  labels, records `y_type_ = type_of_target(y)` (`:296`), and `label_binarize`
  handles the `multilabel-indicator` input branch (`:543-550`, `:564-569`).
  ferrolearn is `Array1<usize>`-only (R-DEV-3).
- REQ-12: PyO3 binding — `import ferrolearn` exposes `LabelBinarizer` mirroring
  `import sklearn` (the project-boundary consumer). Absent.

## Acceptance criteria

- AC-1 (REQ-1): `LabelBinarizer::new().fit(&array![0,1,2,1], &()).classes()` equals
  `[0,1,2]` and `n_classes()` equals `3`, matching the PROBE A oracle `classes_`.
  Pinned by an oracle-grounded `#[test]` (R-CHAR-3).
- AC-2 (REQ-2): `fitted.transform(&array![0,1,2,1])` equals
  `[[1,0,0],[0,1,0],[0,0,1],[0,1,0]]` cell-for-cell (dense), matching PROBE A.
- AC-3 (REQ-3): `LabelBinarizer::new().fit(&array![0,1,0,1],&())?.transform(...)`
  equals `[[0],[1],[0],[1]]` (single column, 1.0 on the second sorted class),
  matching PROBE B.
- AC-4 (REQ-4): `fitted.transform(&array![0,3])` for a `fit([0,1,2])` returns
  `Ok([[1,0,0],[0,0,0]])` (label 3 ignored, all-zero row) — matching PROBE C. Today
  it returns `Err(InvalidParameter)` (DIVERGENT; pinned by
  `test_transform_unknown_label_error`).
- AC-5 (REQ-5): `LabelBinarizer::new().fit(&array![5,5,5],&())?.transform(...)`
  equals `[[0],[0],[0]]` (all-`neg_label`), matching PROBE D. Today it returns
  `[[1],[1],[1]]` (DIVERGENT; `test_single_class` checks only SHAPE `(3,1)`, so the
  wrong values are currently unpinned).
- AC-6 (REQ-6): `fitted.inverse_transform(&array![[0.5]])` for a `fit([0,1])`
  returns `[0]` (PROBE E: strict `0.5 > 0.5` is False), while `[[0.6]]` returns
  `[1]`. Today exactly-0.5 returns `[1]` (DIVERGENT; the `>= 0.5` branch).
- AC-7 (REQ-7): `fitted.inverse_transform(&fitted.transform(&array![0,1,2,1]))`
  round-trips to `[0,1,2,1]` (argmax-clip), matching the PROBE A multiclass path.
- AC-8 (REQ-8): a `LabelBinarizer` constructed with `neg_label=-1, pos_label=1`
  produces a `{-1,+1}` indicator (`:579-583`), and `neg_label=1, pos_label=1` is
  rejected (`:283-287`). Not expressible today.
- AC-9 (REQ-9): the output is a CSR sparse structure when `sparse_output` is set
  (with the `pos_label==0`/`neg_label!=0` guard), not a dense `Array2<f64>`.
- AC-10 (REQ-10): a free `label_binarize(&y, &classes, ...)` produces the PROBE A/B
  matrices without constructing a fitted estimator. No such function today.
- AC-11 (REQ-11): a string-label `fit(["a","b","a"])` analog yields `classes_ ==
  ["a","b"]` and a multilabel-indicator input is binarized per `y_type_`. Not
  representable today (`usize`-only).
- AC-12 (REQ-12): `python3 -c "import ferrolearn; ferrolearn.preprocessing.LabelBinarizer"`
  resolves and `.fit_transform` matches `sklearn` on PROBE A.

## `## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (fit → sorted-unique classes_) | SHIPPED | impl `fn fit in <Fit<Array1<usize>,()> for LabelBinarizer> in label_binarizer.rs` — `let mut classes: Vec<usize> = y.iter().copied().collect(); classes.sort_unstable(); classes.dedup();`, the faithful `usize` analog of sklearn `classes_ = unique_labels(y)` (`_label.py:306`). `FittedLabelBinarizer::classes()` / `n_classes()` expose `classes_`. On PROBE A the discovered `classes_ == [0,1,2]` matches the oracle. Non-test consumer: crate re-export `pub use label_binarizer::{FittedLabelBinarizer, LabelBinarizer};` (`ferrolearn-preprocess/src/lib.rs`) — the boundary public API of the crate, grandfathered under R-DEFER-1/S5 (existing pub API across prior commits; the estimator type IS the surface). Verification: PROBE A vs `.classes()`; `cargo test -p ferrolearn-preprocess` (`test_fit_discovers_sorted_classes`, `test_non_contiguous_classes`). **Empty-input edge is CONSISTENT here: ferrolearn `InsufficientSamples` matches sklearn `y has 0 samples` raise (`:302-303`, `:497-498`) — unlike the MultiLabelBinarizer case, no sub-divergence.** |
| REQ-2 (transform multiclass one-hot values) | SHIPPED (scoped: known labels, k>=3, default neg/pos_label, dense) | impl `fn transform in <Transform<Array1<usize>> for FittedLabelBinarizer> in label_binarizer.rs` — the `else` (k != 2) branch builds `class_to_idx: HashMap<usize,usize>` then sets `out[[i, idx]] = 1.0` into `Array2::zeros((n, k))`, mirroring `label_binarize` multiclass CSR + `toarray()` (`_label.py:552-577`). On PROBE A the indicator values `[[1,0,0],[0,1,0],[0,0,1],[0,1,0]]` match the oracle. Non-test consumer: crate re-export (lib.rs). Verification: PROBE A vs `.transform()`; `cargo test -p ferrolearn-preprocess` (`test_multiclass_transform_indicator_matrix`, `test_non_contiguous_classes`). **Caveat: output is DENSE `Array2<f64>` not CSR (REQ-9); only the default `pos_label=1`/`neg_label=0` is expressible (REQ-8); the unknown-label sub-path DIVERGES (REQ-4); k==1 falls into this branch but DIVERGES (REQ-5) — this row is scoped strictly to k>=3 known-label default-label values.** |
| REQ-3 (transform binary single column, pos_label on 2nd class) | SHIPPED (scoped: known labels, default neg/pos_label) | impl `fn transform in <Transform<Array1<usize>> for FittedLabelBinarizer>` — the `k == 2` branch emits `Array2::zeros((n, 1))` with `out[[i, 0]] = if *idx == 1 { 1.0 } else { 0.0 }`, putting `1.0` on the **second** sorted class, mirroring sklearn's binary collapse to the last sorted column (`_label.py:531`, `:592-596`). On PROBE B `fit_transform([0,1,0,1]) == [[0],[1],[0],[1]]` matches the oracle. Non-test consumer: crate re-export (lib.rs). Verification: PROBE B vs `.transform()`; `cargo test -p ferrolearn-preprocess` (`test_binary_transform_single_column`, `test_inverse_transform_binary` round-trip). **Caveat: only the default `pos_label=1`/`neg_label=0` is expressible (REQ-8); unknown-label sub-path DIVERGES (REQ-4).** |
| REQ-4 (transform unknown-label: ignore, all-zero row) | SHIPPED (closed #1239) | impl `fn transform` — both branches now `if let Some(&idx) = class_to_idx.get(&label) { ... }`, SKIPPING any unseen label and leaving the row's cells at their zero-init value, mirroring sklearn `label_binarize` `np.isin` ignore (`_label.py:556-559`). Live oracle (R-CHAR-3): `fit([0,1,2]).transform([0,3])` → `Ok([[1,0,0],[0,0,0]])` (3 ignored). Guard `divergence_transform_unknown_label_silently_ignored` PASS; in-module `test_transform_unknown_label_error` rewritten to `test_transform_unknown_label_ignored`. Consumer: re-export lib.rs. **No warning emitted (no log facade).** |
| REQ-5 (transform single-class k==1 → all-zero column) | SHIPPED (closed #1240) | impl `fn transform` — added a `k == 1` arm returning `Array2::zeros((n, 1))` (all-`0.0` = all-neg_label single column), mirroring sklearn `n_classes==1 → np.zeros((len(y),1)); Y += neg_label` (`_label.py:532-538`). Live oracle: `fit_transform([5,5,5])` → `[[0],[0],[0]]`. Guard `divergence_single_class_all_zero_column` PASS; in-module `test_single_class` tightened to assert all-0.0 values (was shape-only). Consumer: re-export lib.rs. |
| REQ-6 (inverse_transform binary STRICT threshold) | SHIPPED (closed #1241) | impl `fn inverse_transform` — binary (`k == 2`) branch now `if y[[i, 0]] > 0.5` (STRICT, was `>= 0.5`), mirroring sklearn `_inverse_binarize_thresholding` `y > threshold` (default 0.5) (`_label.py:667`,`:679`), so exactly-0.5 → `classes[0]`. Live oracle: `fit([0,1]).inverse_transform([[0.5]])` → `[0]`, `[[0.6]]` → `[1]`. Guard `divergence_binary_inverse_strict_threshold_at_half` PASS. Consumer: re-export lib.rs. **The k==1 inverse path was verified to MATCH the oracle (sklearn `_inverse_binarize_thresholding` len(classes)==1 → repeat(classes[0]); ferrolearn argmax of all-zero → classes[0]) — green edge, no blocker.** |
| REQ-7 (inverse_transform multiclass argmax) | SHIPPED (scoped: multiclass k != 2) | impl `fn inverse_transform in FittedLabelBinarizer` — the `else` (k != 2) branch computes the per-row first-argmax (`best_v = f64::NEG_INFINITY; if v > best_v { ... }`) and returns `self.classes[best_j]`, mirroring sklearn `classes.take(Y.argmax(axis=1), mode="clip")` (`_label.py:641`). On PROBE A the round-trip `inverse_transform(transform([0,1,2,1])) == [0,1,2,1]`. Non-test consumer: crate re-export (lib.rs). Verification: PROBE A round-trip; `cargo test -p ferrolearn-preprocess` (`test_inverse_transform_multiclass`, `test_roundtrip_multiclass_non_contiguous`, `test_inverse_transform_shape_mismatch`). **Caveat: scoped to the multiclass path; the binary single-column inverse DIVERGES at the threshold boundary (REQ-6).** |
| REQ-8 (neg_label/pos_label ctor params + validation) | NOT-STARTED | open prereq blocker #1242. `LabelBinarizer` is a zero-field unit struct; `pos_label`/`neg_label` are hardcoded to `1.0`/`0.0` in `transform`/`inverse_transform`. sklearn `__init__(*, neg_label=0, pos_label=1, ...)` (`:263`) lets the user pick the encoding, validates `neg_label < pos_label` (`:283-287`), and applies the `pos_label==0` switch + `neg_label != 0` fill in the dense path (`:514-517`, `:579-583`). No analog params, no validation. |
| REQ-9 (sparse_output CSR + constraint) | NOT-STARTED | open prereq blocker #1243. `transform` returns a dense `Array2<f64>` (`Array2::zeros((n, k))` / `(n, 1)`); sklearn returns a `scipy.sparse.csr_matrix` and keeps it CSR when `sparse_output=True` (`label_binarize` `:563`, `:584-585`), with the `pos_label==0 or neg_label!=0` rejection guard (`:289-294`, `:506-512`). No `sparse_output` field, no `sprs`-based path. |
| REQ-10 (label_binarize free function) | NOT-STARTED | open prereq blocker #1244. The standalone module-level `label_binarize(y, *, classes, neg_label=0, pos_label=1, sparse_output=False)` (`_label.py:430`) — the parity op `LabelBinarizer.transform` delegates to (`:357-363`) and a public sklearn API in its own right — has no ferrolearn counterpart; the binarization logic is inlined into `FittedLabelBinarizer::transform` and is not callable without a fitted estimator. |
| REQ-11 (arbitrary label types + type_of_target/multilabel input) | NOT-STARTED | open prereq blocker #1245. The impl is `Array1<usize>`-only (`impl Fit<Array1<usize>, ()>`, `Transform<Array1<usize>>`, `classes: Vec<usize>`). sklearn accepts any label type (`classes_ = unique_labels(y)`, object dtype on non-int), records `y_type_ = type_of_target(y)` (`:296`), and `label_binarize` handles the `multilabel-indicator` input branch (`:543-550`, `:564-569`) — all structurally unrepresentable here (R-DEV-3). |
| REQ-12 (PyO3 binding) | NOT-STARTED | open prereq blocker #1246. No `LabelBinarizer` `#[pyclass]` in `ferrolearn-python/src/` (grep of `ferrolearn-python/src/` for `LabelBinarizer`/`label_binarizer` is empty); `import ferrolearn` cannot expose this transformer (boundary consumer per R-DEFER-1). |

## Architecture

**ferrolearn (existing).** Two types in `label_binarizer.rs`. The unfitted unit
struct `LabelBinarizer` (with `new()` and `Default`) carries no state — no
`neg_label` / `pos_label` / `sparse_output` params. `impl Fit<Array1<usize>, ()>
for LabelBinarizer::fit` (1) rejects empty `y` with
`FerroError::InsufficientSamples`; (2) `sort_unstable()` + `dedup()` the labels
into `classes: Vec<usize>`, returning `FittedLabelBinarizer { classes }`.
`FittedLabelBinarizer` exposes `classes() -> &[usize]`, `n_classes() -> usize`, and
two transform directions. Forward, via `impl Transform<Array1<usize>>`:
`transform(&Array1<usize>) -> Result<Array2<f64>, FerroError>` builds a
`class_to_idx: HashMap<usize,usize>`; the `k == 2` branch fills a single column
(`1.0` on the second class), the `else` (k != 2, **including k == 1**) branch fills
a one-hot `(n, k)` matrix — both **error `InvalidParameter`** on any unseen label.
Inverse: `inverse_transform(&Array2<f64>) -> Result<Array1<usize>, FerroError>`
guards the column count (`ShapeMismatch`: `expected_cols = if k == 2 { 1 } else {
k }`), then the `k == 2` branch thresholds at `>= 0.5` and the `else` branch takes
the per-row first-argmax. The label domain is fixed to `usize`.

**sklearn (target contract).** `LabelBinarizer(TransformerMixin, BaseEstimator)`
(`:180`). `__init__(*, neg_label=0, pos_label=1, sparse_output=False)` (`:263`).
`fit(y)` (`:269`): validates `neg_label < pos_label` (`:283-287`) and the sparse
constraint (`:289-294`), records `y_type_ = type_of_target(y)` (`:296`), rejects
multioutput (`:298-301`) and 0 samples (`:302-303`), sets `classes_ =
unique_labels(y)` (`:306`). `transform(y)` (`:331`) delegates to `label_binarize(y,
classes=classes_, pos_label, neg_label, sparse_output)` (`:357-363`).
`label_binarize` (`:430`): rejects 0 samples (`:497-498`); on `y_type == "binary"`
with `n_classes == 1` returns an all-`neg_label` single column (`:532-538`); with
`len(classes) >= 3` switches to `"multiclass"` (`:539-540`); for binary/multiclass,
**ignores unknown labels** (`np.isin`, `:556-559`), builds a CSR matrix (`:563`),
`toarray()`s + fills `neg_label` and applies the `pos_label==0` switch unless
`sparse_output` (`:575-585`), and collapses the binary case to the single last
column (`:592-596`). `inverse_transform(Y, threshold=None)` (`:365`): default
`threshold = (pos_label+neg_label)/2` (`:399-400`); multiclass →
`_inverse_binarize_multiclass = classes.take(Y.argmax(axis=1), mode="clip")`
(`:641`); else `_inverse_binarize_thresholding`: STRICT `Y > threshold` (`:667`)
then `classes[y.ravel()]` (`:679`).

**The structural gaps.** On the **known-label default-encoding int path** the two
coincide: sorted-unique `classes_` (REQ-1), multiclass one-hot values (REQ-2),
binary single-column values (REQ-3), and multiclass inverse argmax (REQ-7) all
match the oracle. The three load-bearing **behavioral** divergences were FIXED
this iteration: (DIV-1, REQ-4, #1239) unknown-label handling — now ignores
(`:556-559`); (DIV-2, REQ-5, #1240) single-class output — now all-`0` (`:532-538`);
(DIV-3, REQ-6, #1241) inverse binary threshold — now strict `>` (`:667`). The
remaining gaps are structural/surface: the
`neg_label`/`pos_label`/`sparse_output` ctor params + validation (REQ-8/REQ-9), the
`label_binarize` free function (REQ-10), the generic label domain +
`type_of_target`/multilabel-indicator input handling (REQ-11), and the PyO3 binding
(REQ-12). The empty-`y` fit edge is consistent (both raise) and does NOT generate a
blocker.

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-2, REQ-3, REQ-7 — known-label
default-encoding int path):

```bash
# Oracle (PROBE A, run from /tmp) — the multiclass + classes_ + argmax path:
python3 -c "from sklearn.preprocessing import LabelBinarizer; \
lb=LabelBinarizer(); print(lb.fit_transform([0,1,2,1]).tolist()); print(lb.classes_.tolist())"
#   -> [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]] ; [0, 1, 2]
# Oracle (PROBE B) — binary single-column:
python3 -c "from sklearn.preprocessing import LabelBinarizer; \
print(LabelBinarizer().fit_transform([0,1,0,1]).tolist())"   # -> [[0],[1],[0],[1]]
# ferrolearn equivalent: LabelBinarizer::new().fit(&y,&()).classes() == [0,1,2];
#   fitted.transform(&array![0,1,2,1]) == the PROBE A indicator (dense Array2<f64>);
#   fitted.transform(&array![0,1,0,1]) == [[0],[1],[0],[1]];
#   fitted.inverse_transform(transform(...)) round-trips (argmax).

# Crate gauntlet:
cargo test -p ferrolearn-preprocess        # incl. test_fit_discovers_sorted_classes,
                                           # test_multiclass_transform_indicator_matrix,
                                           # test_binary_transform_single_column,
                                           # test_inverse_transform_multiclass,
                                           # test_inverse_transform_binary,
                                           # test_roundtrip_multiclass_non_contiguous,
                                           # test_non_contiguous_classes,
                                           # test_inverse_transform_shape_mismatch
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The in-module `#[test]`s exercise the `usize` known-label default-encoding path and
are **hand-written, not oracle-grounded** (expected values are not derived from a
live sklearn call) — the critic should add oracle-pinned `#[test]`s matching
PROBE A/B (R-CHAR-3). Two existing tests/areas must change when the headline
divergences are fixed: `test_transform_unknown_label_error` (asserts an unknown
label `.is_err()` — opposite of sklearn's silent-ignore, REQ-4) must flip to assert
an all-zero row; and `test_single_class` (asserts only `mat.shape() == [3,1]`)
under-pins REQ-5 — a value-checking test asserting an all-`0.0` column must be added
(it would currently FAIL against the all-`1.0` impl, R-HONEST-4). No currently-green
command establishes REQ-4, REQ-5, REQ-6, or REQ-8..REQ-12.

## Blockers

The three headline divergences (DIV-1/DIV-2/DIV-3) were FIXED this iteration; the
structural gaps are open `-l blocker` issues referenced by the REQ status table:

- #1239 — REQ-4 (DIV-1, CLOSED/fixed): `transform` errored on an unknown label;
  now SILENTLY ignores it (`if let Some(&idx) = class_to_idx.get(&label)`),
  leaving an all-zero row, mirroring sklearn `np.isin` (`_label.py:556-559`).
- #1240 — REQ-5 (DIV-2, CLOSED/fixed): single-class (k == 1) emitted an all-`1.0`
  column; now a `k == 1` arm returns `Array2::zeros((n,1))` (all-0), mirroring
  sklearn `np.zeros((n,1)); Y += neg_label` (`:532-538`).
- #1241 — REQ-6 (DIV-3, CLOSED/fixed): `inverse_transform` binary thresholded at
  `>= 0.5`; now STRICT `> 0.5`, mirroring sklearn `y > threshold` (`:667`).
- #1242 — REQ-8: no `neg_label`/`pos_label` ctor params or `neg_label < pos_label`
  validation; sklearn `__init__(*, neg_label=0, pos_label=1, ...)` (`:263`,
  `:283-287`, `:579-583`).
- #NNN-E — REQ-9: dense `Array2<f64>` output, not CSR (`sprs`); no `sparse_output`
  param or its constraint (`label_binarize:563`, `:584-585`, `:289-294`).
- #NNN-F — REQ-10: no module-level `label_binarize` free function (`:430`); logic is
  inlined into `transform` and not independently callable.
- #NNN-G — REQ-11: `Array1<usize>`-only; no arbitrary label domain, no `y_type_ =
  type_of_target` (`:296`), no `multilabel-indicator` input handling (`:543-550`)
  (R-DEV-3).
- #NNN-H — REQ-12: no `ferrolearn-python` registration of `LabelBinarizer`
  (boundary consumer per R-DEFER-1).

Note (NOT a blocker): the empty-`y` fit edge is CONSISTENT — ferrolearn
`InsufficientSamples` matches sklearn's `y has 0 samples` raise (`:302-303`,
`:497-498`), unlike the MultiLabelBinarizer case.
