# MultiLabelBinarizer

<!--
tier: 3-component
status: draft
baseline-commit: 24b9c1c4
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_label.py  # class MultiLabelBinarizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None) (:688); _parameter_constraints {classes, sparse_output} (:751-754); __init__(*, classes=None, sparse_output=False) (:756); fit(y) (:761) — classes=sorted(set(chain.from_iterable(y))) when classes is None (:779), dup-check raise when classes set (:780-785), dtype=int if all int else object (:788-790); fit_transform(y) (:794) — classes-set path delegates fit().transform() (:811-812), else optimized defaultdict path (:814-835); transform(y) (:837) -> _build_cache (:863) -> _transform (:869); _transform collects unknown labels and emits warnings.warn("unknown class(es) {sorted} will be ignored") — NEVER raises (:889-902), builds CSR (:905-907), toarray() unless sparse_output (:858-859); inverse_transform(yt) (:909) — ValueError if yt.shape[1] != len(classes_) (:925-930), ValueError "Expected only 0s and 1s in label indicator" if any value not in {0,1} via setdiff1d (:941-947), returns list of tuples [tuple(classes_.compress(row)) for row in yt] (:948); _more_tags X_types ["2dlabels"] (:950-951). Labels: any orderable+hashable object; classes_ is an ndarray with int dtype if all-int else object.
ferrolearn-module: ferrolearn-preprocess/src/multi_label_binarizer.rs
parity-ops: MultiLabelBinarizer
crosslink-issue: 1229
-->

## Summary

scikit-learn's `MultiLabelBinarizer` (`_label.py:688`) converts an iterable of
label sets into a **(samples x classes) binary indicator matrix** where
`y_indicator[i, j] == 1` iff `classes_[j]` is in sample `i`. `fit(y)` (`:761`)
sets `classes_ = sorted(set(chain.from_iterable(y)))` when `classes is None`
(`:779`); `transform(y)` (`:837`) builds the indicator, **collecting and ignoring
unknown labels with a `warnings.warn(...)`** (`:889-902`) and never raising;
`inverse_transform(yt)` (`:909`) **strictly validates** the matrix is all 0s/1s
(`:941-947`) and returns a **list of tuples** (`:948`). Labels may be any
orderable+hashable object (ints, strings, tuples), and `classes_` carries `int`
dtype when all labels are int else `object` (`:788`). The constructor
(`:756`) takes `classes=None` (an explicit ordering, with a duplicate-rejection
guard `:780-785`) and `sparse_output=False` (CSR when set).

`ferrolearn-preprocess/src/multi_label_binarizer.rs` ships a **`usize`-only**,
dense core: a unit struct `MultiLabelBinarizer` and `FittedMultiLabelBinarizer {
classes: Vec<usize> }`. `impl Fit<Vec<Vec<usize>>, ()>::fit` rejects empty input
with `FerroError::InsufficientSamples`, else sets `classes` = sorted+deduped
unique of all labels (`classes.sort_unstable(); classes.dedup()`).
`FittedMultiLabelBinarizer::transform` builds a class→index `HashMap` and emits a
dense `Array2<f64>` multi-hot matrix, **skipping any label not seen during fit**
(#1230 fix — mirrors sklearn ignore, `:889-902`).
`inverse_transform(&Array2<f64>)` checks column count (`ShapeMismatch`), then
**validates the matrix is all-0/1** (`Err(InvalidParameter)` otherwise, #1231 fix,
`:941-947`) and includes a class iff its cell `== 1.0`, returning
`Vec<Vec<usize>>`. There is no `classes` ctor param, no `sparse_output`, no
generic/non-int label type, no optimized `fit_transform`, and no PyO3 binding.

**Headline finding (document prominently).** On the **known-label int path** the
two coincide exactly: sorted-unique `classes_` and the multi-hot indicator
**values** match the oracle (REQ-1, REQ-2 SHIPPED). The two **most-fixable
divergences** were behavioral and are now FIXED: (REQ-3, #1230) `transform` of an
*unknown* label now ignores it (was an error); and (REQ-4, #1231)
`inverse_transform` now strictly validates 0/1 and selects exact-1 cells (was a
`>= 0.5` threshold), matching sklearn (`:941-947`). The remaining gaps are
surface/structural (`classes` param, CSR, generic label dtype, optimized
`fit_transform`, PyO3 binding) plus the empty-`y` edge (#1237).

## Probes (live sklearn oracle, 1.5.2; run from /tmp)

```bash
# PROBE A (REQ-1, REQ-2) — int fit_transform: sorted classes_, multi-hot values match:
python3 -c "from sklearn.preprocessing import MultiLabelBinarizer; \
mlb=MultiLabelBinarizer(); \
print(mlb.fit_transform([[0,2],[1],[0,1,2]]).tolist()); print(mlb.classes_.tolist())"
# -> [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
#    [0, 1, 2]
#    ferrolearn: MultiLabelBinarizer::new().fit(&y).classes() == [0,1,2];
#    fitted.transform(&y) is the SAME indicator but as a DENSE Array2<f64> (not CSR).

# PROBE B (REQ-3) — UNKNOWN label is IGNORED with a warning, NEVER raised:
python3 -c "import warnings; from sklearn.preprocessing import MultiLabelBinarizer; \
mlb=MultiLabelBinarizer().fit([[0,1]])
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always'); r=mlb.transform([[0,5]])
    print('result', r.tolist()); print('warning', str(w[0].message))"
# -> result [[1, 0]]
#    warning unknown class(es) [5] will be ignored
#    ferrolearn: fitted.transform(&vec![vec![0,5]]) -> Err(InvalidParameter "unknown label 5
#    not seen during fit") — DIVERGENCE (errors instead of ignore-with-warning).

# PROBE C (REQ-4) — inverse_transform STRICTLY validates 0/1 -> ValueError:
python3 -c "import numpy as np; from sklearn.preprocessing import MultiLabelBinarizer; \
mlb=MultiLabelBinarizer().fit([[0,1,2]])
try:
    mlb.inverse_transform(np.array([[0.4,0.6,0.5]]))
except ValueError as e: print('ValueError:', e)"
# -> ValueError: Expected only 0s and 1s in label indicator. Also got [0.4 0.5 0.6]
#    ferrolearn: fitted.inverse_transform(&array![[0.4,0.6,0.5]]) -> Ok(vec![vec![1,2]])
#    (0.4<0.5 dropped, 0.6/0.5>=0.5 kept) — DIVERGENCE (thresholds & succeeds; sklearn RAISES).

# PROBE D (REQ-4) — inverse_transform returns a list of TUPLES (not list of lists):
python3 -c "import numpy as np; from sklearn.preprocessing import MultiLabelBinarizer; \
mlb=MultiLabelBinarizer().fit([[0,1,2]]); print(mlb.inverse_transform(np.array([[1,0,1]])))"
# -> [(0, 2)]   (tuple per row; ferrolearn returns Vec<Vec<usize>> i.e. vec![vec![0,2]])

# PROBE E (REQ-5) — `classes` ctor param fixes column ordering, skips y-scan:
python3 -c "from sklearn.preprocessing import MultiLabelBinarizer; \
mlb=MultiLabelBinarizer(classes=[2,1,0]); \
print(mlb.fit_transform([[0,2]]).tolist(), mlb.classes_.tolist())"
# -> [[1, 0, 1]] [2, 1, 0]   (column order is the supplied [2,1,0]; ferrolearn always sorts)
```

## Requirements

- REQ-1: `fit` discovers sorted-unique `classes_` — mirror sklearn
  `classes = sorted(set(chain.from_iterable(y)))` (`:779`) on the all-int label
  path (PROBE A: `[[0,2],[1],[0,1,2]] -> classes_ == [0,1,2]`).
- REQ-2: `transform` of **known** labels emits the dense multi-hot indicator —
  `y_indicator[i, j] == 1` iff `classes_[j] in y[i]` (`_transform` `:869-907`
  then `toarray()` `:858-859`); on the known-label path the indicator **values**
  match the oracle (PROBE A matrix).
- REQ-3: `transform` of **unknown** labels — sklearn **collects and ignores** them
  with a `warnings.warn("unknown class(es) {sorted} will be ignored")` and never
  raises (`:889-902`, PROBE B). ferrolearn instead returns
  `FerroError::InvalidParameter` (`:170-176`). HEADLINE behavioral divergence.
- REQ-4: `inverse_transform` 0/1 validation + return contract — sklearn raises
  `ValueError("Expected only 0s and 1s in label indicator. ...")` on any value
  ∉ `{0,1}` (`setdiff1d`, `:941-947`, PROBE C) and returns a **list of tuples**
  `[tuple(classes_.compress(row)) for row in yt]` (`:948`, PROBE D). ferrolearn
  instead **thresholds at `>= 0.5`** (`:96`) — accepting arbitrary floats — and
  returns `Vec<Vec<usize>>`. HEADLINE behavioral divergence.
- REQ-5: `classes` constructor param — an explicit class ordering that fixes
  column order and skips the `y` scan, with the duplicate-rejection guard
  (`:780-785`, PROBE E). ferrolearn has no such param; `classes_` is always the
  sorted discovered set.
- REQ-6: `sparse_output` — CSR output when set (`_transform` returns
  `sp.csr_matrix`, kept sparse when `sparse_output` `:858-859`, `:905-907`).
  ferrolearn always emits a dense `Array2<f64>`.
- REQ-7: Arbitrary orderable+hashable label types (strings, tuples) and
  `classes_` `object` dtype — sklearn accepts any such label and sets
  `dtype = int if all int else object` (`:788`, doc example `:727-731`).
  ferrolearn is `Vec<Vec<usize>>`-only (R-DEV-3).
- REQ-8: Optimized `fit_transform` — sklearn's `classes is None` path
  (`:814-835`) builds the column mapping in a single `defaultdict`-driven pass
  rather than `fit` then `transform`. ferrolearn has no `FitTransform` impl for
  this estimator (no default-trait fusion either).
- REQ-9: PyO3 binding — `import ferrolearn` exposes `MultiLabelBinarizer`
  mirroring `import sklearn` (the project-boundary consumer). Absent.

## Acceptance criteria

- AC-1 (REQ-1): `MultiLabelBinarizer::new().fit(&vec![vec![0,2],vec![1],vec![0,1,2]],
  &()).classes()` equals `[0,1,2]` and `n_classes()` equals `3`, matching the
  PROBE A oracle `classes_`. Pinned by an oracle-grounded `#[test]` (R-CHAR-3).
- AC-2 (REQ-2): `fitted.transform(&y)` for the PROBE A fit equals
  `[[1,0,1],[0,1,0],[1,1,1]]` cell-for-cell (dense), matching the oracle indicator.
- AC-3 (REQ-3): `fitted.transform(&vec![vec![0,5]])` for a `fit([[0,1]])` returns
  `Ok([[1,0]])` (ignoring `5`) and surfaces the unknown class — matching PROBE B.
  Today it returns `Err(InvalidParameter)` (DIVERGENT; pinned by
  `test_transform_unknown_label_error`).
- AC-4 (REQ-4): `fitted.inverse_transform(&array![[0.4,0.6,0.5]])` returns `Err`
  (PROBE C: sklearn `ValueError`) instead of the current `Ok([[1,2]])`; on a valid
  0/1 matrix `[[1,0,1]]` it returns the PROBE D label sets (`[0,2]`). Today the
  0.5-threshold path is DIVERGENT (pinned by `test_inverse_threshold`).
- AC-5 (REQ-5): a `MultiLabelBinarizer` constructed with `classes=[2,1,0]`
  produces `classes_ == [2,1,0]` and the PROBE E column order. Not expressible
  today (no `classes` param).
- AC-6 (REQ-6): the output is a CSR sparse structure when `sparse_output` is set,
  not a dense `Array2<f64>`.
- AC-7 (REQ-7): a string-label `fit([["sci-fi","thriller"],["comedy"]])` analog
  yields `classes_ == ["comedy","sci-fi","thriller"]` (doc example `:727-731`).
  Not representable today (`usize`-only).
- AC-8 (REQ-8): a fused `fit_transform(&y)` produces the PROBE A matrix in a
  single pass equivalent to `fit(&y)?.transform(&y)`.
- AC-9 (REQ-9): `python3 -c "import ferrolearn; ferrolearn.preprocessing.MultiLabelBinarizer"`
  resolves and `.fit_transform` matches `sklearn` on PROBE A.

## `## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (fit → sorted-unique classes_) | SHIPPED | impl `fn fit in <Fit<Vec<Vec<usize>>,()> for MultiLabelBinarizer> in multi_label_binarizer.rs` — `let mut classes = y.iter().flatten().copied().collect(); classes.sort_unstable(); classes.dedup();`, the faithful `usize` analog of sklearn `classes = sorted(set(itertools.chain.from_iterable(y)))` (`_label.py:779`). `FittedMultiLabelBinarizer::classes()` / `n_classes()` expose `classes_`. On PROBE A the discovered `classes_ == [0,1,2]` matches the oracle bit-for-bit. Non-test consumer: crate re-export `pub use multi_label_binarizer::{FittedMultiLabelBinarizer, MultiLabelBinarizer};` (`ferrolearn-preprocess/src/lib.rs` line 155) — the boundary public API of the crate, grandfathered under R-DEFER-1/S5 (existing pub API across prior commits; the estimator type IS the surface). Verification: PROBE A vs `.classes()`; `cargo test -p ferrolearn-preprocess` (`test_fit_discovers_sorted_classes`, `test_non_contiguous_classes`). **Caveat (NOT-STARTED #1237): ferrolearn rejects empty `y` with `InsufficientSamples`; sklearn `sorted(set(chain.from_iterable([])))` yields empty `classes_` with no error — a minor empty-input sub-divergence, analogous to the LabelEncoder REQ-5 case.** |
| REQ-2 (transform → dense multi-hot, known labels) | SHIPPED (scoped: known labels, dense) | impl `fn transform in <Transform<Vec<Vec<usize>>> for FittedMultiLabelBinarizer> in multi_label_binarizer.rs` — builds `class_to_idx: HashMap<usize,usize>` then sets `out[[i, idx]] = 1.0` into `Array2::zeros((n, k))`, mirroring sklearn `_transform` (`_label.py:869-907`) + `toarray()` (`:858-859`). On PROBE A the indicator **values** `[[1,0,1],[0,1,0],[1,1,1]]` match the oracle. Non-test consumer: crate re-export (lib.rs line 155). Verification: PROBE A vs `.transform()`; `cargo test -p ferrolearn-preprocess` (`test_transform_multi_hot`, `test_empty_label_set`, `test_duplicate_labels_in_input`). **Caveat: output is DENSE `Array2<f64>`, not CSR (REQ-6). The UNKNOWN-label sub-path DIVERGES (REQ-3): ferrolearn errors where sklearn ignores-with-warning — this row is scoped strictly to the known-label indicator values.** |
| REQ-3 (transform unknown-label: ignore, no error) | SHIPPED (closed #1230) | impl `fn transform in <Transform<Vec<Vec<usize>>> for FittedMultiLabelBinarizer>` — `if let Some(&idx) = class_to_idx.get(&label) { out[[i, idx]] = 1.0; }` SKIPS unknown labels and always returns `Ok`, mirroring sklearn `_transform` collect-and-ignore (`_label.py:889-902`). Live oracle (R-CHAR-3): `fit([[0,1]]).transform([[0,5]])` → `Ok([[1.0,0.0]])` (5 skipped); `fit([[0,1,2]]).transform([[2,9],[1]])` → `[[0,0,1],[0,1,0]]`. Guards `divergence_req3_transform_unknown_label_ignored` + `divergence_req3_transform_unknown_label_multi_sample` PASS; in-module `test_transform_unknown_label_error` rewritten to `test_transform_unknown_label_ignored` (R-HONEST-4). Consumer: re-export lib.rs:155. **R-DEV note: sklearn's `warnings.warn("unknown class(es) ... will be ignored")` is intentionally NOT emitted — the crate has no `log`/`tracing` facade; the behavioral indicator parity is the contract.** |
| REQ-4 (inverse_transform 0/1 validation) | SHIPPED (closed #1231) | impl `fn inverse_transform` — after the `ShapeMismatch` guard, validates `find(\|&&v\| v != 0.0 && v != 1.0)` → `Err(InvalidParameter "Expected only 0s and 1s in label indicator...")`, then includes a class iff cell `== 1.0` (was `>= 0.5`), mirroring sklearn `np.setdiff1d(yt,[0,1])` raise + exact-1 selection (`_label.py:941-947`). Scoped `#[allow(clippy::float_cmp, reason=...)]` on the method (indicator is exact 0/1). Live oracle (R-CHAR-3): `inverse_transform([[0.4,0.6,0.5]])` → `Err` (sklearn ValueError); `inverse_transform([[1,0,1]])` → `vec![vec![0,2]]` (sklearn `[(0,2)]`). Guard `divergence_req4_inverse_rejects_non_01` PASS; in-module `test_inverse_threshold` rewritten to `test_inverse_rejects_non_01` (R-HONEST-4). Consumer: re-export lib.rs:155. **Scoped: ferrolearn returns `Vec<Vec<usize>>`, sklearn returns a list of tuples — a representationally-faithful return-type difference, out of scope (no separate blocker).** |
| REQ-5 (`classes` ctor param) | NOT-STARTED | open prereq blocker #1232. `MultiLabelBinarizer` is a zero-field unit struct; there is no `classes` parameter. sklearn `__init__(*, classes=None, ...)` (`:756`) lets the user fix an explicit class ordering (skipping the `y` scan, `:787`) with a duplicate-rejection guard (`:780-785`, PROBE E: `classes=[2,1,0]` → that exact column order). `classes_` in ferrolearn is always the sorted discovered set. |
| REQ-6 (sparse_output CSR) | NOT-STARTED | open prereq blocker #1233. `transform` returns a dense `Array2<f64>` (`Array2::zeros((n, k))`); sklearn returns a `scipy.sparse.csr_matrix` and keeps it CSR when `sparse_output=True` (`_transform` `:905-907`, `:858-859`). No `sparse_output` field, no `sprs`-based path. |
| REQ-7 (arbitrary orderable+hashable labels + object dtype) | NOT-STARTED | open prereq blocker #1234. The impl is `Vec<Vec<usize>>`-only (`impl Fit<Vec<Vec<usize>>, ()>`, `Transform<Vec<Vec<usize>>>`, `classes: Vec<usize>`). sklearn accepts any orderable+hashable label (strings, tuples) and sets `dtype = int if all int else object` (`:788`); the string example `[["sci-fi","thriller"],["comedy"]] → ["comedy","sci-fi","thriller"]` (`:727-731`) is structurally unrepresentable (R-DEV-3). |
| REQ-8 (optimized fit_transform) | NOT-STARTED | open prereq blocker #1235. There is no `FitTransform` impl for `MultiLabelBinarizer` (and no default-trait fusion), so the optimized single-pass `defaultdict` path sklearn uses when `classes is None` (`fit_transform` `:814-835`) has no analog. Callers must `fit(&y)?.transform(&y)` manually (two passes). |
| REQ-9 (PyO3 binding) | NOT-STARTED | open prereq blocker #1236. No `RsMultiLabelBinarizer` `#[pyclass]` in `ferrolearn-python/src/` (grep of `ferrolearn-python/src/` for `MultiLabelBinarizer`/`multi_label` is empty); `import ferrolearn` cannot expose this transformer (boundary consumer per R-DEFER-1). |

## Architecture

**ferrolearn (existing).** Two types in `multi_label_binarizer.rs`. The unfitted
unit struct `MultiLabelBinarizer` (with `new()` and `Default`) carries no state —
no `classes` / `sparse_output` params. `impl Fit<Vec<Vec<usize>>, ()> for
MultiLabelBinarizer::fit` (1) rejects empty `y` with
`FerroError::InsufficientSamples`; (2) flattens all labels, `sort_unstable()`,
`dedup()` into `classes: Vec<usize>`, returning `FittedMultiLabelBinarizer {
classes }`. `FittedMultiLabelBinarizer` exposes `classes() -> &[usize]`,
`n_classes() -> usize`, `inverse_transform(&Array2<f64>) ->
Result<Vec<Vec<usize>>, FerroError>` (column-count `ShapeMismatch` guard, then an
all-0/1 validation `Err` and exact-`== 1.0` cell selection, #1231), and — via
`impl Transform<Vec<Vec<usize>>>` —
`transform(&Vec<Vec<usize>>) -> Result<Array2<f64>, FerroError>`, which builds a
`class_to_idx: HashMap<usize,usize>`, fills a dense `Array2::zeros((n, k))`, and
**skips any unseen label** (#1230, mirrors sklearn ignore). The label
domain is fixed to `usize`.

**sklearn (target contract).** `MultiLabelBinarizer(TransformerMixin,
BaseEstimator, auto_wrap_output_keys=None)` (`:688`). `__init__(*, classes=None,
sparse_output=False)` (`:756`). `fit(y)` (`:761`): when `classes is None`,
`classes_ = sorted(set(chain.from_iterable(y)))` (`:779`); when supplied, it
rejects duplicates (`:780-785`) and uses the given order; `dtype = int if all int
else object` (`:788`). `fit_transform(y)` (`:794`) delegates to `fit().transform()`
when `classes` is set (`:811-812`), else runs an optimized `defaultdict`-driven
single pass (`:814-835`). `transform` (`:837`) → `_build_cache` (`:863`) →
`_transform` (`:869`): per sample it maps known labels to column indices and
**adds any `KeyError` label to an `unknown` set**, then — if non-empty — emits
`warnings.warn("unknown class(es) {sorted} will be ignored")` (`:889-902`),
building a CSR matrix (`:905-907`) and `toarray()`-ing it unless `sparse_output`
(`:858-859`). `inverse_transform(yt)` (`:909`): `check_is_fitted`, raise on column
mismatch (`:925-930`), **raise `ValueError` on any value ∉ {0,1}** via
`np.setdiff1d(yt, [0,1])` (`:941-947`), then return `[tuple(classes_.compress(row))
for row in yt]` (`:948`).

**The structural gaps.** On the **known-label int path** the two coincide exactly
— sorted-unique `classes_` and the multi-hot indicator values both match the
oracle (REQ-1/2 SHIPPED). The two load-bearing **behavioral** divergences are now
FIXED: (a) **unknown-label handling** (REQ-3, #1230): ferrolearn now ignores
unknown labels (`:889-902`); (b) **inverse 0/1 validation** (REQ-4, #1231):
ferrolearn now rejects non-0/1 and selects exact-1 cells (`:941-947`) — the
`Vec<Vec<usize>>`-vs-tuple return type is a faithful, out-of-scope difference. The
remaining gaps are structural/surface: the `classes` ctor param (REQ-5, #1232),
CSR output (REQ-6, #1233), the generic orderable+hashable label domain
(REQ-7, #1234), the optimized `fit_transform` (REQ-8, #1235), the PyO3 binding
(REQ-9, #1236), and the empty-`y` fit edge (#1237).

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-2 — known-label int path):

```bash
# Oracle (PROBE A, run from /tmp) — the int path ferrolearn must match:
python3 -c "from sklearn.preprocessing import MultiLabelBinarizer; \
mlb=MultiLabelBinarizer(); \
print(mlb.fit_transform([[0,2],[1],[0,1,2]]).tolist()); print(mlb.classes_.tolist())"
#   -> [[1, 0, 1], [0, 1, 0], [1, 1, 1]] ; [0, 1, 2]
# ferrolearn equivalent: MultiLabelBinarizer::new().fit(&y, &()).classes() == [0,1,2];
#   fitted.transform(&y) == the same indicator (dense Array2<f64>).

# Crate gauntlet:
cargo test -p ferrolearn-preprocess        # incl. test_fit_discovers_sorted_classes,
                                           # test_transform_multi_hot, test_empty_label_set,
                                           # test_duplicate_labels_in_input,
                                           # test_non_contiguous_classes,
                                           # test_inverse_transform_roundtrip
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The in-module `#[test]`s exercise the `usize` known-label path and are
**hand-written, not oracle-grounded** (expected values are not derived from a live
sklearn call) — the critic should add an oracle-pinned `#[test]` matching PROBE A
(R-CHAR-3). Two existing tests PIN divergent behavior and must flip when the
headline divergences are fixed: `test_transform_unknown_label_error` (asserts an
unknown label is `.is_err()` — opposite of sklearn's ignore-with-warning, REQ-3)
and `test_inverse_threshold` (asserts `inverse_transform([[0.4,0.6,0.5]])` returns
`Ok(vec![vec![1,2]])` — the exact case where sklearn RAISES, REQ-4, R-HONEST-4).
No currently-green command establishes REQ-3..REQ-9.

## Blockers

REQ-3 + REQ-4 (the two headline divergences) were FIXED this iteration; the
remaining NOT-STARTED REQs are open `-l blocker` issues referenced by the REQ
status table:

- #1230 — REQ-3 (CLOSED, fixed): `transform` ERRORED on an unknown label
  (`InvalidParameter`, `:170-176`); now SKIPS unknown labels and returns `Ok`,
  mirroring sklearn collect-and-ignore (`_label.py:889-902`). The Python warning
  has no Rust analog (no log facade) and is intentionally not emitted.
- #1231 — REQ-4 (CLOSED, fixed): `inverse_transform` 0.5-thresholded arbitrary
  floats; now validates the matrix is all-0/1 (`Err` otherwise) and includes a
  class iff cell `== 1.0`, mirroring sklearn `setdiff1d(yt,[0,1])` (`:941-947`).
  The `Vec<Vec<usize>>`-vs-list-of-tuples return type is out of scope (faithful).
- #1237 — REQ-1 edge: empty-`y` fit errors `InsufficientSamples`; sklearn yields
  empty `classes_` with no error (`:779`).
- #1232 — REQ-5: no `classes` ctor param; sklearn `__init__(*, classes=None)`
  (`:756`) fixes column ordering with a duplicate guard (`:780-785`, PROBE E).
- #1233 — REQ-6: dense `Array2<f64>` output, not CSR (`sprs`); no `sparse_output`
  param (`_transform:905-907`, `:858-859`).
- #1234 — REQ-7: `Vec<Vec<usize>>`-only; no arbitrary orderable+hashable label
  domain or `object`-dtype `classes_` (`:788`, string example `:727-731`)
  (R-DEV-3).
- #1235 — REQ-8: no `FitTransform` impl / optimized single-pass `fit_transform`
  (`:814-835`).
- #1236 — REQ-9: no `ferrolearn-python` registration of `MultiLabelBinarizer`
  (boundary consumer per R-DEFER-1).
