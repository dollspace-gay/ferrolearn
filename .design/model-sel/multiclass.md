# OneVsRestClassifier / OneVsOneClassifier (multiclass meta-estimators)

<!--
tier: 3-component
status: draft
baseline-commit: 2d7e3ff6613bc739fed17d59b9b38d448826aa45
upstream-paths:
  - sklearn/multiclass.py            # OneVsRestClassifier (:196), OvR fit (:325), predict (:476-512), predict_proba (:514-552), decision_function (:554-582); OneVsOneClassifier (:665), OvO fit (:756), _fit_ovo_binary (:626), predict (:918-939), decision_function (:941-986); _predict_binary (:103); OutputCodeClassifier (:1025)
  - sklearn/utils/multiclass.py      # _ovr_decision_function (:520-562)
-->

## Summary

`ferrolearn-model-sel/src/multiclass.rs` mirrors scikit-learn's multiclass
meta-estimators from `sklearn/multiclass.py`: `OneVsRestClassifier` (`:196`,
one binary classifier per class, class-vs-rest) and `OneVsOneClassifier`
(`:665`, one binary classifier per class pair, majority vote). Both wrap a base
binary estimator — expressed as a `PipelineFactory` CLOSURE
(`Box<dyn Fn() -> Pipeline<f64> + Send + Sync>`) rather than a wrapped
estimator object, the same sanctioned R-DEV-7 Rust idiom used in
`grid_search.rs`/`calibration.rs`, NOT a bug.

ferrolearn ships the **OvR strategy end-to-end** for binary-multiclass `y`: one
binary estimator per sorted class, `decision_function` = per-class scores,
sorted `classes_`, and `predict` = column-argmax. The OvR predict TIE-BREAK is
**LAST-on-tie**, and this MATCHES sklearn (verified live below — sklearn's
`predict` overwrite pattern `argmaxima[maxima==pred]=i` also picks the LATER
class index, NOT `np.argmax` first-on-tie). The OvO fit MECHANIC ships (the
`K*(K-1)/2` pairwise estimators, sorted `classes_`, the per-pair filter/label).

The central DETERMINISTIC FIXABLE divergence is **OvO predict**:
ferrolearn votes with PURE integer counts and `max_by_key` (last-on-tie) and has
NO confidence term, whereas sklearn computes
`Y = _ovr_decision_function(predictions, confidences, n_classes)` (=
`votes + sum_of_confidences/(3*(|sum|+1))`) then `argmax` — so on a VOTE TIE
sklearn breaks by summed confidence and can pick a DIFFERENT class
(oracle-verified below: vote tie `[1,1,1]`, confidences favoring class 0 →
sklearn picks class 0, ferrolearn picks class 2). The critic pins this
(#1812). The rest are missing-feature/architectural blockers: OvO
`decision_function` (#1813), OvR/OvO `predict_proba` (#1814),
`OutputCodeClassifier` (#1815), `sample_weight`/`n_jobs`/`fit_params` (#1816),
multilabel OvR (#1817), and the `ferray` substrate (#1818).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### `OneVsRestClassifier` (`sklearn/multiclass.py:196`)
- `:196-201` — `class OneVsRestClassifier(MultiOutputMixin, ClassifierMixin,
  MetaEstimatorMixin, BaseEstimator)`.
- `:316` — `__init__(self, estimator, *, n_jobs=None, verbose=0)`.
- `:362-365` — `fit`: `self.label_binarizer_ =
  LabelBinarizer(sparse_output=True)`; `Y = label_binarizer_.fit_transform(y)`;
  `self.classes_ = self.label_binarizer_.classes_` (sorted unique).
- `:370-382` — one `_fit_binary` per column of `Y` (one estimator per class,
  class-vs-rest); the binary target column is 1 for the class, 0 otherwise.
- `:492-500` — `predict` (multiclass branch): `maxima.fill(-inf)`;
  `for i, e in enumerate(self.estimators_): pred = _predict_binary(e, X);
  np.maximum(maxima, pred, out=maxima); argmaxima[maxima == pred] = i`. The
  overwrite-on-equality assignment means the LATER estimator index wins a tie →
  **LAST-on-tie** (NOT `np.argmax` first-on-tie). Returns
  `self.classes_[argmaxima]`.
- `:514-552` — `predict_proba`: `Y = np.array([e.predict_proba(X)[:, 1] for e
  in estimators_]).T`; for single-label multiclass `Y /= np.sum(Y, axis=1)[:,
  None]` (row-normalize to sum 1).
- `:554-582` — `decision_function`: per-class `est.decision_function(X)` scores,
  `(n_samples, n_classes)`.

### `OneVsOneClassifier` (`sklearn/multiclass.py:665`)
- `:665` — `class OneVsOneClassifier(MetaEstimatorMixin, ClassifierMixin,
  BaseEstimator)`. `:748` — `__init__(self, estimator, *, n_jobs=None)`.
- `:795-799` — `fit`: `self.classes_ = np.unique(y)`; raises `ValueError` if
  `len(self.classes_) == 1`.
- `:800-820` — `n_classes*(n_classes-1)/2` `_fit_ovo_binary` estimators over
  `for i in range(n_classes) for j in range(i+1, n_classes)` (pairs `(i, j)`,
  `i < j` by sorted-class index).
- `:626-645` — `_fit_ovo_binary(estimator, X, y, i, j, ...)`: filter
  `cond = (y==i) | (y==j)`; `y_binary[y == i] = 0`, `y_binary[y == j] = 1` —
  **class `i` (lower) → 0, class `j` (higher) → 1** (verified live below).
- `:918-939` — `predict`: `Y = self.decision_function(X)`; multiclass branch
  `return self.classes_[Y.argmax(axis=1)]` (`np.argmax` → FIRST-on-tie, but on
  the `votes+confidence` continuous `Y` ties are essentially impossible).
- `:941-986` — `decision_function`: `predictions` (per-pair `est.predict`, 0/1)
  + `confidences` (per-pair `_predict_binary`); `Y =
  _ovr_decision_function(predictions, confidences, len(self.classes_))`.

### `_ovr_decision_function` (`sklearn/utils/multiclass.py:520-562`)
- `:540-550` — `votes`/`sum_of_confidences` zeros `(n, K)`; for `k` over pairs
  `(i<j)`: `sum_of_confidences[:, i] -= conf[:, k]`,
  `sum_of_confidences[:, j] += conf[:, k]`,
  `votes[pred[:, k]==0, i] += 1`, `votes[pred[:, k]==1, j] += 1`.
- `:559-562` — `transformed = sum_of_confidences / (3*(|sum_of_confidences|+1))`
  ∈ `(-1/3, 1/3)`; returns `votes + transformed`. The `1/3` bound guarantees the
  confidence term BREAKS VOTE TIES without overriding a ≥1-vote difference.
  ⇒ OvO predict = `argmax(votes + transformed_confidences)`.

### `_predict_binary` (`sklearn/multiclass.py:103-112`)
- regressor → `predict(X)`; else `decision_function(X).ravel()`, falling back to
  `predict_proba(X)[:, 1]` for proba-only estimators. The per-binary confidence.

### `OutputCodeClassifier` (`sklearn/multiclass.py:1025`)
- A THIRD multiclass strategy (error-correcting output codes) that ferrolearn
  lacks entirely.

## Requirements

R-DEV mental test applied per REQ ("numerical/API contract" → MATCH;
"Cython/CPython footgun" → deviate; "missing feature" → NOT-STARTED with a
blocker).

- REQ-OVR-PREDICT (OvR predict — column-argmax, LAST-on-tie): ferrolearn's
  `impl Predict for FittedOneVsRestClassifier` runs `decision_function` then
  per-row `iter().enumerate().max_by(partial_cmp)` → Rust `Iterator::max_by`
  returns the LAST element among equals → on a score tie the HIGHEST class index
  wins. sklearn `predict` (`:492-500`) overwrites `argmaxima[maxima==pred]=i`
  for ascending `i` → the LATER (highest) index also wins. **MATCH** (R-DEV-1 —
  tie-breaking is an explicit MATCH item; verified live: a 3-way `[0.5,0.5,0.5]`
  decision-score tie → both pick class 2). Do NOT assume `np.argmax`
  first-on-tie — sklearn OvR uses the overwrite pattern = LAST. SHIPPED.
- REQ-OVR-MECH (OvR fit + decision_function + classes_): one binary estimator
  per sorted unique class (`unique_classes`), binary target `1.0` for the class
  / `0.0` else, fresh pipeline per class; `decision_function` =
  `(n_samples, n_classes)` of per-class scores; `classes()` sorted. Mirrors
  sklearn `fit` (`:362-382`) + `decision_function` (`:554-582`) + `classes_`
  (`:365`). **MATCH** (R-DEV-1/R-DEV-3 — structural/output contract). NOTE:
  ferrolearn's binary labels are `{1.0, 0.0}` and `decision_function` uses the
  binary `predict` score (vs sklearn's `decision_function`) — both monotone in
  the positive-class confidence, so argmax-equivalent; an R-DEV-7-adjacent
  representation choice, NOT a divergence. SHIPPED.
- REQ-OVO-DECISION (OvO predict = argmax(votes + confidence) — DETERMINISTIC
  FIXABLE DIVERGENCE): ferrolearn `impl Predict for FittedOneVsOneClassifier`
  accumulates PURE integer `votes` (per pair, `preds > 0.5 → cls_i` else
  `cls_j`) and picks `max_by_key(votes)` (LAST-on-tie), with NO confidence term.
  sklearn `predict` (`:935-939`) does `argmax(decision_function(X))` where
  `decision_function` = `_ovr_decision_function` =
  `votes + sum_of_confidences/(3*(|sum|+1))` (`utils/multiclass.py:520-562`,
  `multiclass.py:983`). On a VOTE TIE sklearn breaks by summed confidence;
  ferrolearn picks the last class index → DIVERGENCE. **MATCH-class fixable**
  (R-DEV-1 — tie-breaking + decision-function contract): port
  `_ovr_decision_function`. (The i/j vote convention is internally consistent —
  ferrolearn trains cls_i as the POSITIVE/1 class and votes cls_i on
  `preds > 0.5`, whereas sklearn maps cls_i → 0 and votes cls_i on `pred == 0`;
  both consistently attribute the high-confidence pair outcome to the same
  underlying class, so the integer vote counts agree. The ONLY behavioral gap is
  the missing confidence tie-break.) NOT-STARTED (#1812).
- REQ-OVO-DECISION-FN (OvO `decision_function` method): ABSENT —
  `FittedOneVsOneClassifier` exposes only `classes()`/`n_estimators()`/`predict`,
  no `decision_function`. sklearn exposes `decision_function` (`:941-986`)
  returning `votes + transformed_confidences`, `(n_samples, n_classes)`.
  **MATCH-intent / missing-feature** (tied to REQ-OVO-DECISION — the fix likely
  adds this method and routes `predict` through it). NOT-STARTED (#1813).
- REQ-PREDICT-PROBA (OvR/OvO predict_proba): ABSENT on both. sklearn OvR
  `predict_proba` (`:514-552`) stacks per-estimator positive-class probabilities
  and row-NORMALIZES for single-label multiclass; OvO has no `predict_proba` by
  default. **MATCH-intent / missing-feature**. NOT-STARTED (#1814).
- REQ-OUTPUT-CODE (`OutputCodeClassifier`): MISSING estimator — the
  error-correcting output-code strategy (`:1025`) has no ferrolearn analog.
  **MATCH-intent / missing-feature** (a follow-up acto-builder ships it; do NOT
  build it in this doc-author pass). NOT-STARTED (#1815).
- REQ-SAMPLE-WEIGHT / REQ-NJOBS / REQ-FIT-PARAMS: ABSENT. sklearn threads
  `**fit_params` to each sub-estimator (`:325`, `:756`), parallelizes the
  sub-problems with `n_jobs` (`:370`, `:804`), and (via the base estimator)
  honors `sample_weight`. ferrolearn `fit(x, y)` takes only `(x, y)`, no
  weight/jobs/params channel. **MATCH-intent / missing-feature**. NOT-STARTED
  (#1816).
- REQ-MULTILABEL (OvR multilabel-indicator `y`): sklearn OvR supports a
  multilabel-indicator `Y` (binary-relevance, `LabelBinarizer`, `:362-382`;
  `predict` non-multiclass branch `:501-512`). ferrolearn accepts only a
  `Array1<usize>` label vector (binary-multiclass), rejecting `< 2` classes;
  no indicator-matrix path. **MATCH-intent / missing-feature**. NOT-STARTED
  (#1817).
- REQ-X-1 (R-SUBSTRATE ndarray→ferray-core): production code imports
  `use ndarray::{Array1, Array2}` (array type); destination substrate is
  `ferray-core` (R-SUBSTRATE-1). NOT-STARTED (#1818).
- REQ-X-2 (non-test production consumer): the boundary meta-estimator types
  `OneVsRestClassifier`/`OneVsOneClassifier`/`FittedOneVsRestClassifier`/
  `FittedOneVsOneClassifier` are the public API (S5 / R-DEFER-1) and are
  re-exported from `lib.rs`. SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side). The oracle is the
installed sklearn 1.5.2 (run from a directory OUTSIDE the source clone — the
clone at `/home/doll/scikit-learn` is unbuilt; `cd /tmp` first).

- AC-OVR-PREDICT (REQ-OVR-PREDICT — SHIPPED, LAST-on-tie MATCHES): construct a
  decision-score TIE via a base estimator whose `decision_function` returns an
  identical constant for every binary problem → a 3-way tie; confirm BOTH
  sklearn and ferrolearn pick the HIGHEST class index.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multiclass import OneVsRestClassifier
  from sklearn.base import BaseEstimator, ClassifierMixin
  class ConstScore(BaseEstimator, ClassifierMixin):
      def __init__(self, val=0.0): self.val=val
      def fit(self,X,y): self.classes_=np.unique(y); return self
      def decision_function(self,X): return np.full(X.shape[0], 0.5)
      def predict(self,X): return np.zeros(X.shape[0])
  X=np.array([[0.],[1.],[2.],[3.],[4.],[5.]]); y=np.array([0,0,1,1,2,2])
  clf=OneVsRestClassifier(ConstScore()).fit(X,y)
  print('df row:', clf.decision_function(X[:1]).tolist())
  print('predict on tie:', clf.predict(X[:1]).tolist())"
  # -> df row: [[0.5, 0.5, 0.5]]
  # -> predict on tie: [2]   (LAST class index)
  ```
  ferrolearn `FittedOneVsRestClassifier::predict` over a tied decision row picks
  `classes[2]` (Rust `max_by` returns LAST among equals). VERIFIED MATCH →
  SHIPPED. The critic may pin this exact LAST-on-tie selection as an
  oracle-grounded `#[test]` with a constant-score fixture.
- AC-OVR-MECH (REQ-OVR-MECH — SHIPPED, mechanic): K classes → K estimators,
  sorted `classes_`, decision_function shape `(n_samples, K)`. Oracle:
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multiclass import OneVsRestClassifier
  from sklearn.svm import SVC
  X=np.array([[10,10.],[8,10],[-5,5.5],[-5.4,5.5],[-20,-20],[-15,-20]])
  y=np.array([0,0,1,1,2,2])
  clf=OneVsRestClassifier(SVC()).fit(X,y)
  print('n_est:', len(clf.estimators_), 'classes_:', clf.classes_.tolist(),
        'df shape:', clf.decision_function(X[:3]).shape)"
  # -> n_est: 3 classes_: [0, 1, 2] df shape: (3, 3)
  ```
  ferrolearn `test_ovr_fit_predict_three_classes`/`test_ovr_decision_function_shape`
  assert `n_estimators()==3`, `classes()==[0,1,2]`, decision_function `(6,3)`.
- AC-OVO-DECISION (REQ-OVO-DECISION — DIVERGENCE, the critic's pin): build a
  VOTE TIE where the confidence-correction picks a class DIFFERENT from
  last-on-tie. Reproduce `_ovr_decision_function` on hand-built
  predictions/confidences for 3 classes (pairs `(0,1),(0,2),(1,2)`): a `[1,1,1]`
  vote tie with confidences favoring class 0.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.utils.multiclass import _ovr_decision_function
  predictions = np.array([[0,1,0]])   # votes: class0(k0), class2(k1), class1(k2) -> 1-1-1 tie
  confidences = np.array([[-1.0,-1.0,0.0]])
  Y = _ovr_decision_function(predictions, confidences, 3)
  print('Y:', [round(v,4) for v in Y[0].tolist()])
  print('sklearn OvO predict argmax:', int(Y.argmax(axis=1)[0]))"
  # -> Y: [1.2222, 0.8333, 0.8333]
  # -> sklearn OvO predict argmax: 0
  ```
  sklearn picks class 0 (confidence-corrected). ferrolearn accumulates the SAME
  `votes = [1,1,1]` and `max_by_key` returns class 2 (LAST among equal counts) —
  with NO confidence term. DIVERGENCE. NOT-STARTED (#1812). The critic pins a
  FAILING `#[test]` driving `FittedOneVsOneClassifier::predict` (via a fixture
  whose three pairwise classifiers produce this exact vote pattern) and asserts
  the sklearn-correct class 0.
- AC-OVO-CONVENTION (REQ-OVO-DECISION supporting fact — the i/j map): sklearn
  `_fit_ovo_binary` maps the LOWER class to 0 and HIGHER to 1.
  ```
  cd /tmp && python3 -c "
  import numpy as np, sklearn.multiclass as M
  from sklearn.svm import SVC
  orig=M._fit_binary; cap={}
  def spy(est,X,y,fit_params,classes=None):
      cap['ybin']=np.array(y); cap['classes']=classes; return orig(est,X,y,fit_params,classes)
  M._fit_binary=spy
  y=np.array([0,0,1,1,2,2]); X=np.arange(12).reshape(6,2).astype(float)
  M._fit_ovo_binary(SVC(),X,y,0,2,fit_params={})
  print('pair(0,2) y_binary=', cap['ybin'].tolist(), 'classes=', cap['classes'])"
  # -> pair(0,2) y_binary= [0, 0, 1, 1] classes= [0, 2]   (class i=0 -> 0, j=2 -> 1)
  ```
  ferrolearn trains the FLIPPED labels (cls_i → 1.0, cls_j → 0.0) AND votes cls_i
  on `preds > 0.5`, so the net vote attribution to the underlying class matches;
  the integer vote counts agree. Only the missing confidence tie-break (#1812)
  diverges — NOT the convention.
- AC-OVO-DECISION-FN (REQ-OVO-DECISION-FN — ABSENT): sklearn exposes a
  `decision_function` on OvO; ferrolearn has none.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multiclass import OneVsOneClassifier
  from sklearn.svm import SVC
  from sklearn.datasets import load_iris
  X,y=load_iris(return_X_y=True)
  clf=OneVsOneClassifier(SVC()).fit(X,y)
  print('df shape:', clf.decision_function(X[:5]).shape)"
  # -> df shape: (5, 3)
  ```
  `grep -n "fn decision_function" ferrolearn-model-sel/src/multiclass.rs` shows
  ONLY the OvR `decision_function` — no method on `FittedOneVsOneClassifier`.
  NOT-STARTED (#1813).
- AC-PREDICT-PROBA (REQ-PREDICT-PROBA — ABSENT): sklearn OvR `predict_proba`
  normalizes per-class probs.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multiclass import OneVsRestClassifier
  from sklearn.svm import SVC
  from sklearn.datasets import load_iris
  X,y=load_iris(return_X_y=True)
  clf=OneVsRestClassifier(SVC(probability=True)).fit(X,y)
  p=clf.predict_proba(X[:3]); print(p.shape, p.sum(axis=1).round(6).tolist())"
  # -> (3, 3) [1.0, 1.0, 1.0]
  ```
  ferrolearn has no `predict_proba` on either OvR or OvO. NOT-STARTED (#1814).
- AC-OUTPUT-CODE (REQ-OUTPUT-CODE — MISSING): the estimator exists in sklearn.
  ```
  cd /tmp && python3 -c "
  from sklearn.multiclass import OutputCodeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import make_classification
  X,y=make_classification(n_samples=100,n_features=4,n_informative=2,n_redundant=0,random_state=0,shuffle=False)
  clf=OutputCodeClassifier(RandomForestClassifier(random_state=0),random_state=0).fit(X,y)
  print('n_est:', len(clf.estimators_), 'code_book shape:', clf.code_book_.shape)"
  # -> n_est: 3 code_book shape: (2, 3)
  ```
  `grep -n "OutputCode" ferrolearn-model-sel/src/multiclass.rs` finds nothing.
  NOT-STARTED (#1815).
- AC-SAMPLE-WEIGHT (REQ-SAMPLE-WEIGHT/NJOBS/FIT-PARAMS — ABSENT): sklearn OvR/OvO
  accept `n_jobs` and thread `**fit_params`.
  ```
  cd /tmp && python3 -c "
  import inspect
  from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
  print('OvR init:', list(inspect.signature(OneVsRestClassifier.__init__).parameters)[1:])
  print('OvO init:', list(inspect.signature(OneVsOneClassifier.__init__).parameters)[1:])"
  # -> OvR init: ['estimator', 'n_jobs', 'verbose']
  # -> OvO init: ['estimator', 'n_jobs']
  ```
  ferrolearn `new(make_pipeline)` takes only the factory; `fit(x, y)` only `(x,
  y)`. No `n_jobs`/`verbose`/`sample_weight`/`fit_params`. NOT-STARTED (#1816).
- AC-X-2 (REQ-X-2 — SHIPPED): `grep -n "pub use multiclass" lib.rs` shows the
  re-export of `OneVsRestClassifier, OneVsOneClassifier,
  FittedOneVsRestClassifier, FittedOneVsOneClassifier`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-OVR-PREDICT (OvR column-argmax, LAST-on-tie) | SHIPPED | impl `impl Predict for FittedOneVsRestClassifier` in `multiclass.rs`: `fn predict` calls `self.decision_function(x)?` then per row `row.iter().enumerate().max_by(\|(_, a), (_, b)\| a.partial_cmp(b).unwrap_or(Equal)).map(\|(k, _)\| k)` → Rust `Iterator::max_by` returns the LAST element among equals → on a score tie the HIGHEST class index wins; maps to `self.classes[best_k]`. sklearn `predict` (`sklearn/multiclass.py:492-500`) fills `maxima` with `-inf` then `np.maximum(maxima, pred, out=maxima); argmaxima[maxima == pred] = i` for ascending `i` → the LATER (highest) index also wins on equality (NOT `np.argmax` first-on-tie). LIVE ORACLE (AC-OVR-PREDICT): a 3-way decision-score tie `[0.5,0.5,0.5]` (constant-score base estimator) → sklearn `predict == [2]` (LAST class index); ferrolearn `max_by` likewise selects `classes[2]`. VERIFIED MATCH. Non-test consumer: REQ-X-2 (boundary type + `lib.rs` re-export). |
| REQ-OVR-MECH (OvR fit + decision_function + classes_) | SHIPPED | impl `pub fn fit in multiclass.rs` (`OneVsRestClassifier`): `unique_classes(y)` (sorted/dedup), rejects `< 2` classes (`FerroError::InvalidParameter`), then per class builds `y_binary = if l == cls { 1.0 } else { 0.0 }` and fits a fresh `(self.make_pipeline)()` → `FittedOneVsRestClassifier { estimators, classes }`. `pub fn decision_function in multiclass.rs` returns `Array2 (n_samples, n_classes)` with column `k` = estimator `k`'s `predict` scores; `pub fn classes`/`pub fn n_estimators` accessors. Mirrors sklearn `fit` (one estimator per `LabelBinarizer` column, `sklearn/multiclass.py:362-382`), `decision_function` (`:554-582`), `classes_` sorted (`:365`). LIVE ORACLE (AC-OVR-MECH): `OneVsRestClassifier(SVC())` on 3 classes → `n_est==3`, `classes_==[0,1,2]`, df shape `(3,3)`; ferrolearn `test_ovr_fit_predict_three_classes`/`test_ovr_decision_function_shape` assert the same. R-DEV-7-adjacent NOTE: ferrolearn's binary labels are `{1.0,0.0}` and `decision_function` uses the binary `predict` score (vs sklearn's `decision_function`) — both monotone in positive-class confidence, hence argmax-equivalent; a representation choice, not a divergence. Non-test consumer: REQ-X-2. |
| REQ-OVO-DECISION (OvO predict = argmax(votes + confidence) — DETERMINISTIC FIXABLE DIVERGENCE) | NOT-STARTED | open prereq blocker #1812. impl `impl Predict for FittedOneVsOneClassifier` in `multiclass.rs`: per pair `(cls_i, cls_j, est)`, `if preds[s] > 0.5 { votes[s][idx_i] += 1 } else { votes[s][idx_j] += 1 }`, then `sample_votes.iter().enumerate().max_by_key(\|&(_, v)\| *v).map(\|(k, _)\| k)` → Rust `max_by_key` returns the LAST index among equal vote counts, with NO confidence term. sklearn `predict` (`sklearn/multiclass.py:935-939`) is `argmax(decision_function(X))` where `decision_function` (`:983`) returns `_ovr_decision_function(predictions, confidences, K)` = `votes + sum_of_confidences/(3*(|sum_of_confidences|+1))` (`sklearn/utils/multiclass.py:540-562`). On a VOTE TIE sklearn breaks by summed confidence; ferrolearn picks the last class index. LIVE ORACLE (AC-OVO-DECISION): `_ovr_decision_function([[0,1,0]], [[-1,-1,0]], 3)` = `[1.2222, 0.8333, 0.8333]` → sklearn argmax **0**; the identical `votes=[1,1,1]` under ferrolearn's `max_by_key` → class **2**. DETERMINISTIC FIXABLE: port `_ovr_decision_function`; the critic pins a FAILING `#[test]` driving `FittedOneVsOneClassifier::predict` on a constructed vote-tie asserting class 0. (The i/j vote convention is self-consistent — AC-OVO-CONVENTION — so the integer vote counts agree; only the confidence tie-break diverges.) |
| REQ-OVO-DECISION-FN (OvO `decision_function` method) | NOT-STARTED | open prereq blocker #1813. `impl FittedOneVsOneClassifier in multiclass.rs` exposes only `pub fn classes`/`pub fn n_estimators` plus the `Predict::predict` impl — there is NO `decision_function` method. sklearn `decision_function` (`sklearn/multiclass.py:941-986`) stacks per-pair `predictions` + `confidences` and returns `_ovr_decision_function(...)` = `votes + transformed_confidences`, shape `(n_samples, n_classes)`. LIVE ORACLE (AC-OVO-DECISION-FN): `OneVsOneClassifier(SVC()).fit(iris).decision_function(X[:5]).shape == (5, 3)`; `grep -n "fn decision_function" multiclass.rs` finds only the OvR method. Tied to REQ-OVO-DECISION (the fix likely adds this method and routes `predict` through it). |
| REQ-PREDICT-PROBA (OvR/OvO predict_proba) | NOT-STARTED | open prereq blocker #1814. Neither `FittedOneVsRestClassifier` nor `FittedOneVsOneClassifier` defines `predict_proba` (`grep -n "predict_proba" multiclass.rs` is empty). sklearn OvR `predict_proba` (`sklearn/multiclass.py:514-552`) stacks `e.predict_proba(X)[:, 1]` per estimator and row-NORMALIZES (`Y /= np.sum(Y, axis=1)[:, None]`) for single-label multiclass; OvO has no default `predict_proba`. LIVE ORACLE (AC-PREDICT-PROBA): `OneVsRestClassifier(SVC(probability=True)).fit(iris).predict_proba(X[:3])` shape `(3,3)`, rows sum to 1.0; ferrolearn cannot produce probabilities at all. Absent end-to-end. |
| REQ-OUTPUT-CODE (`OutputCodeClassifier`) | NOT-STARTED | open prereq blocker #1815. MISSING ESTIMATOR. `multiclass.rs` defines no output-code strategy (`grep -n "OutputCode" multiclass.rs` is empty); `lib.rs` re-exports only the OvR/OvO types. sklearn `OutputCodeClassifier` (`sklearn/multiclass.py:1025`) fits `int(n_classes * code_size)` binary estimators against a random code book and predicts via `pairwise_distances_argmin` to the nearest code. LIVE ORACLE (AC-OUTPUT-CODE): `OutputCodeClassifier(RandomForestClassifier(...), random_state=0).fit(...)` → `len(estimators_)==3`, `code_book_.shape==(2,3)`. A follow-up acto-builder ships this (NOT this doc-author pass). |
| REQ-SAMPLE-WEIGHT / REQ-NJOBS / REQ-FIT-PARAMS (weights, parallelism, fit-param threading) | NOT-STARTED | open prereq blocker #1816. impl `pub fn new in multiclass.rs` (both estimators) takes ONLY `make_pipeline: PipelineFactory`; `pub fn fit in multiclass.rs` takes only `(x: &Array2<f64>, y: &Array1<usize>)`. There is no `n_jobs`/`verbose` constructor param, no `sample_weight`, no `**fit_params` channel. sklearn `OneVsRestClassifier.__init__(estimator, *, n_jobs=None, verbose=0)` (`sklearn/multiclass.py:316`) and `OneVsOneClassifier.__init__(estimator, *, n_jobs=None)` (`:748`), both parallelizing the sub-problems with `Parallel(n_jobs=...)` (`:370`, `:804`) and threading `**fit_params` to each sub-estimator (`fit(X, y, **fit_params)`, `:325`/`:756`). LIVE ORACLE (AC-SAMPLE-WEIGHT): OvR init params `['estimator','n_jobs','verbose']`, OvO `['estimator','n_jobs']`; ferrolearn has none of these channels. Absent end-to-end. |
| REQ-MULTILABEL (OvR multilabel-indicator y) | NOT-STARTED | open prereq blocker #1817. impl `pub fn fit in multiclass.rs` (`OneVsRestClassifier`) takes `y: &Array1<usize>` (a single label vector) and `unique_classes`/rejects `< 2` classes — there is NO multilabel-indicator (2D binary `Y`) path. sklearn OvR (`sklearn/multiclass.py:362-382`) uses `LabelBinarizer(sparse_output=True)` and supports an indicator matrix `y` (binary-relevance), with `predict` returning `label_binarizer_.inverse_transform(indicator)` in the non-multiclass branch (`:501-512`) and a `multilabel_` property (`:584-587`). ferrolearn is binary-multiclass only. Absent end-to-end. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker #1818. Production code in `multiclass.rs` imports `use ndarray::{Array1, Array2}` and operates on `Array2<f64>`/`Array1<usize>`/`Array1<f64>` (e.g. `fn select_rows` builds `Array2::from_shape_vec`, `fn decision_function` returns `Array2<f64>`). Per R-SUBSTRATE-1 the destination is `ferray-core`; `ndarray` is the wrong substrate. Not migrated (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | Crate re-export: `lib.rs` `pub use multiclass::{FittedOneVsOneClassifier, FittedOneVsRestClassifier, OneVsOneClassifier, OneVsRestClassifier};` (and `pub mod multiclass;`). Per S5 / R-DEFER-1 the boundary meta-estimator types ARE the public API and are grandfathered (existing pub surface). CAVEAT (honest underclaim): `grep -rn "OneVsRestClassifier\|OneVsOneClassifier" ferrolearn-*/src/ \| grep -v 'tests\|multiclass.rs'` finds ONLY the `lib.rs` re-export — there is NO dedicated non-test internal caller and NO `ferrolearn-python` binding yet. SHIPPED on the boundary re-export per S5, not a dedicated production caller; the missing Python binding is noted. The base estimator is supplied as a `PipelineFactory` CLOSURE (an R-DEV-7 Rust idiom for sklearn's wrapped-`estimator` pattern) — noted, not pinned. |

## Architecture

ferrolearn splits each meta-estimator into an unfitted/Fitted pair (CLAUDE.md
naming): `OneVsRestClassifier { make_pipeline: PipelineFactory }` →
`FittedOneVsRestClassifier { estimators: Vec<FittedPipeline<f64>>, classes:
Vec<usize> }`, and `OneVsOneClassifier { make_pipeline }` →
`FittedOneVsOneClassifier { estimators: Vec<(usize, usize, FittedPipeline<f64>)>,
classes: Vec<usize> }`. sklearn keeps a single class whose post-`fit` state is
`estimators_` + `classes_` (+ `label_binarizer_`/`pairwise_indices_`).

**The base-estimator representation is an R-DEV-7 deviation.** sklearn wraps an
`estimator` object and `clone`s it per sub-problem (`:371`, `:805`); ferrolearn
takes a `make_pipeline: Box<dyn Fn() -> Pipeline<f64> + Send + Sync>` closure
called once per sub-problem to produce a fresh binary pipeline. This is the
sanctioned Rust analog (same idiom as `grid_search.rs`/`calibration.rs`) — noted,
not pinned.

**OvR (REQ-OVR-MECH, REQ-OVR-PREDICT — SHIPPED).** `fit` collects sorted unique
classes (`unique_classes`), rejects `< 2` classes, and trains one binary pipeline
per class with target `1.0` for the class / `0.0` otherwise (sklearn's
class-vs-rest, `:362-382`). `decision_function` returns `(n_samples, n_classes)`
of per-class scores (sklearn `:554-582`). `predict` runs `decision_function` then
per-row `max_by(partial_cmp)`. The TIE-BREAK is LAST-on-tie, and this MATCHES
sklearn's overwrite pattern `argmaxima[maxima==pred]=i` (`:499`), which also picks
the LATER class index — verified live (`[0.5,0.5,0.5]` → class 2). This is the
key correctness check: sklearn OvR is NOT `np.argmax` first-on-tie.

**OvO fit MECHANIC (part of REQ-OVO-DECISION — SHIPPED portion).** `fit` builds
`K*(K-1)/2` pairwise estimators over `(cls_i < cls_j)`, filters to samples of
either class, labels cls_i as `1.0` / cls_j as `0.0`, and stores `(cls_i, cls_j,
FittedPipeline)`. sklearn `_fit_ovo_binary` (`:626-645`) maps the LOWER class to
0 and HIGHER to 1 (verified live, AC-OVO-CONVENTION) — the OPPOSITE label
assignment — but ferrolearn ALSO flips the predict-time vote (`preds > 0.5 →
cls_i`), so the net vote attribution to the underlying class is consistent and
the integer vote counts agree.

**The OvO predict DIVERGENCE (REQ-OVO-DECISION, #1812) is the central
deterministic FIXABLE gap.** ferrolearn's `predict` tallies PURE integer votes and
`max_by_key` (LAST-on-tie) with NO confidence term. sklearn's `predict` is
`argmax(decision_function)` where `decision_function` = `_ovr_decision_function`
= `votes + sum_of_confidences/(3*(|sum|+1))` (`utils/multiclass.py:540-562`,
`multiclass.py:983`). The transformed confidence ∈ `(-1/3, 1/3)` BREAKS VOTE TIES
without overriding a ≥1-vote difference. On a `[1,1,1]` vote tie with confidences
`[-1,-1,0]` (pairs `(0,1),(0,2),(1,2)`), sklearn picks class 0 while ferrolearn
picks class 2 — oracle-verified, deterministic. The fix ports
`_ovr_decision_function` (likely as a new OvO `decision_function`, REQ-OVO-DECISION-FN
#1813, with `predict` routed through it); the critic pins a failing test on the
constructed vote-tie.

What is structurally ABSENT vs sklearn: OvO `decision_function`
(REQ-OVO-DECISION-FN, #1813), OvR/OvO `predict_proba` (REQ-PREDICT-PROBA, #1814),
`OutputCodeClassifier` (REQ-OUTPUT-CODE, #1815),
`sample_weight`/`n_jobs`/`verbose`/`fit_params` (REQ-SAMPLE-WEIGHT/NJOBS/FIT-PARAMS,
#1816), multilabel-indicator OvR (REQ-MULTILABEL, #1817), and the `ferray`
substrate (REQ-X-1, #1818). SHIPPED: OvR mechanic + LAST-on-tie predict
(REQ-OVR-MECH, REQ-OVR-PREDICT), the boundary re-export (REQ-X-2).

Invariants: `y.len() == X.nrows()` (`FerroError::ShapeMismatch`); `>= 2` classes
(`FerroError::InvalidParameter`); `classes` sorted ascending; OvR
`decision_function` is `(n_samples, n_classes)`; OvR predict and OvO predict both
LAST-on-tie (OvR matches sklearn; OvO diverges from sklearn's confidence
tie-break).

## Verification

Commands establishing the SHIPPED claims (baseline
`2d7e3ff6613bc739fed17d59b9b38d448826aa45`). The oracle is the installed sklearn
1.5.2 (`cd /tmp` — the source clone at `/home/doll/scikit-learn` is unbuilt):

- `cargo test -p ferrolearn-model-sel --lib multiclass` → **12 passed, 0 failed**
  (`multiclass::tests::{test_ovr_fit_predict_three_classes,
  test_ovr_decision_function_shape, test_ovr_shape_mismatch,
  test_ovr_single_class_fails, test_ovr_two_classes,
  test_ovo_fit_predict_three_classes, test_ovo_n_estimators_four_classes,
  test_ovo_shape_mismatch, test_ovo_single_class_fails, test_ovo_two_classes,
  test_unique_classes, test_unique_classes_single}`).
- REQ-OVR-PREDICT SHIPPED oracle (live sklearn — LAST-on-tie MATCHES, R-CHAR-3):
  the AC-OVR-PREDICT snippet (constant-score base estimator) prints
  `df row: [[0.5, 0.5, 0.5]]` and `predict on tie: [2]`; ferrolearn `max_by` over
  a tied decision row likewise selects `classes[2]` (LAST). The critic may pin
  this as an oracle-grounded `#[test]`.
- REQ-OVR-MECH SHIPPED oracle (live sklearn): AC-OVR-MECH — `OneVsRestClassifier(
  SVC())` on 3 classes → `n_est==3`, `classes_==[0,1,2]`, df shape `(3,3)`;
  matched by `test_ovr_fit_predict_three_classes`/`test_ovr_decision_function_shape`.
- REQ-OVO-DECISION DIVERGENCE oracle (#1812 — the pin): AC-OVO-DECISION —
  `_ovr_decision_function([[0,1,0]], [[-1,-1,0]], 3) = [1.2222, 0.8333, 0.8333]`
  → sklearn argmax **0**; ferrolearn's identical `votes=[1,1,1]` under `max_by_key`
  → class **2**. The critic pins a FAILING `#[test]` driving
  `FittedOneVsOneClassifier::predict` on a constructed vote-tie asserting class 0.
  Supporting: AC-OVO-CONVENTION shows `_fit_ovo_binary` maps lower→0/higher→1 (the
  convention is internally consistent; only the confidence tie-break diverges).
- REQ-OVO-DECISION-FN ABSENT oracle (#1813): AC-OVO-DECISION-FN —
  `OneVsOneClassifier(SVC()).fit(iris).decision_function(X[:5]).shape == (5,3)`;
  ferrolearn has no `decision_function` on `FittedOneVsOneClassifier`.
- REQ-PREDICT-PROBA ABSENT oracle (#1814): AC-PREDICT-PROBA —
  `OneVsRestClassifier(SVC(probability=True)).fit(iris).predict_proba(X[:3])`
  shape `(3,3)`, rows sum to 1.0; no ferrolearn `predict_proba`.
- REQ-OUTPUT-CODE MISSING oracle (#1815): AC-OUTPUT-CODE —
  `OutputCodeClassifier(...).fit(...)` → `len(estimators_)==3`,
  `code_book_.shape==(2,3)`; no ferrolearn symbol.
- REQ-SAMPLE-WEIGHT/NJOBS/FIT-PARAMS ABSENT oracle (#1816): AC-SAMPLE-WEIGHT —
  OvR init `['estimator','n_jobs','verbose']`, OvO `['estimator','n_jobs']`;
  ferrolearn `new(make_pipeline)`/`fit(x, y)` expose none.
- REQ-MULTILABEL ABSENT (#1817): `fit` takes `y: &Array1<usize>` only; no
  indicator-matrix path (sklearn `LabelBinarizer`, `multiclass.py:362-382`).
- REQ-X-2 consumer: `grep -n "pub use multiclass" ferrolearn-model-sel/src/lib.rs`
  shows `pub use multiclass::{FittedOneVsOneClassifier, FittedOneVsRestClassifier,
  OneVsOneClassifier, OneVsRestClassifier};`. `grep -rn
  "OneVsRestClassifier\|OneVsOneClassifier" ferrolearn-*/src/ | grep -v
  'tests\|multiclass.rs'` shows ONLY the re-export (no dedicated internal caller,
  no Python binding — honest underclaim).
- REQ-X-1 substrate: `grep -n "ndarray" ferrolearn-model-sel/src/multiclass.rs`
  shows `use ndarray::{Array1, Array2}` — wrong substrate, migration owed (#1818).

SHIPPED: REQ-OVR-PREDICT (OvR LAST-on-tie — VERIFIED MATCH vs sklearn's overwrite
pattern, NOT first-on-tie), REQ-OVR-MECH (OvR fit/decision_function/classes_),
REQ-X-2 (boundary re-export consumer; no dedicated caller / no Python binding —
honest underclaim). NOT-STARTED: REQ-OVO-DECISION (votes+confidence tie-break,
#1812 — DETERMINISTIC FIXABLE), REQ-OVO-DECISION-FN (#1813),
REQ-PREDICT-PROBA (#1814), REQ-OUTPUT-CODE (#1815),
REQ-SAMPLE-WEIGHT/NJOBS/FIT-PARAMS (#1816), REQ-MULTILABEL (#1817),
REQ-X-1 (ferray substrate, #1818).

Per R-DEFER-2 every REQ is binary SHIPPED/NOT-STARTED. The DETERMINISTIC FIXABLE
divergence the critic should pin as a FAILING test is **REQ-OVO-DECISION** (#1812
— OvO predict on a constructed VOTE TIE: sklearn's `votes + transformed_confidences`
argmax picks class 0, ferrolearn's pure-vote `max_by_key` picks class 2). The fix
ports `_ovr_decision_function` (adding the OvO `decision_function`, #1813, and
routing `predict` through it). REQ-OVR-PREDICT is verified SHIPPED (LAST-on-tie
MATCHES sklearn) — NOT a pin. The i/j vote convention (AC-OVO-CONVENTION) is
internally consistent — NOT a separate divergence. The remaining NOT-STARTED REQs
are missing-feature/architectural (#1813/#1814/#1815/#1816/#1817) or substrate
(#1818) — blockers, not pins.

Least-confident SHIPPED claim: REQ-OVR-PREDICT — the LAST-on-tie MATCH is verified
only on an EXACT decision-score tie (the constant-score fixture). For nearly-tied
floating-point scores the two argmax reductions can still diverge if rounding
makes one side's `maxima == pred` equality hold where the other's strict
comparison does not; the verified surface is the exact-tie regime, and ferrolearn
uses `predict` scores while sklearn uses `decision_function` scores (argmax-
equivalent under monotonicity, REQ-OVR-MECH note), so a base estimator whose two
score channels are non-monotone relative to each other could surface a difference.
