# SelfTrainingClassifier (semi-supervised self-training meta-estimator)

<!--
tier: 3-component
status: draft
baseline-commit: 1fe9a1bf5b1d2f39d8d6d80f3bc4f09f6ffb2d31
upstream-paths:
  - sklearn/semi_supervised/_self_training.py   # SelfTrainingClassifier (:39), _parameter_constraints (:160-169), fit (:191-299), predict (:301-322), delegated methods (:324-417)
-->

## Summary

`ferrolearn-model-sel/src/self_training.rs` mirrors scikit-learn's
`SelfTrainingClassifier` (`sklearn/semi_supervised/_self_training.py:39`): a
semi-supervised meta-estimator that wraps a base classifier, repeatedly refits it
on the labeled subset, and pseudo-labels confident unlabeled samples until no new
labels are added or `max_iter` is reached.

ferrolearn ships the **self-training ITERATION SHAPE** — refit-on-labeled,
predict-unlabeled, pseudo-label-confident, iterate-to-fixpoint-or-`max_iter` — for
the `criterion='threshold'` case. But the SHIPPED surface is **rigorously narrow**:
it is **BINARY-ONLY** (labels hard-coded `{0, 1}` derived from a SINGLE score
interpreted as `P(class 1)`, not `predict_proba` argmax over `classes_`), its
`predict` returns **RAW SCORES** rather than class labels, and two deterministic
contract facets are INVERTED from sklearn — the selection comparison (`>=` vs
strict `>`) and the threshold validation interval (`(0, 1]` vs `[0, 1)`). It does
NOT ship multi-class, `criterion='k_best'`, `predict`-returns-labels, the
`transduction_`/`labeled_iter_`/`n_iter_`/`termination_condition_` attribute
contract, `max_iter=0`/`max_iter=None`, the `predict_proba`/`decision_function`/
`predict_log_proba`/`score` delegation surface, or the `ferray-core` substrate.

The base estimator is expressed as a `FitFn` CLOSURE (`Box<dyn Fn(&Array2<f64>,
&Array1<usize>) -> Result<PredictFn, FerroError> + Send + Sync>`, shared with
`calibration.rs`) rather than a wrapped/`clone`d estimator object — the sanctioned
R-DEV-7 Rust idiom, NOT a bug.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### `SelfTrainingClassifier` (`sklearn/semi_supervised/_self_training.py:39`)
- `:160-169` — `_parameter_constraints`: `threshold` ∈
  `Interval(Real, 0.0, 1.0, closed="left")` (`:164`) ⇒ **`[0, 1)`: 0.0 ACCEPTED,
  1.0 REJECTED** (verified live); `criterion` ∈ `StrOptions({"threshold",
  "k_best"})` (`:165`); `k_best` ∈ `Interval(Integral, 1, None, closed="left")`
  (`:166`); `max_iter` ∈ `Interval(Integral, 0, None, closed="left")` OR `None`
  (`:167`) ⇒ **`max_iter=0` and `max_iter=None` both VALID**.
- `:171-185` — `__init__(self, base_estimator, threshold=0.75,
  criterion="threshold", k_best=10, max_iter=10, verbose=False)`. The base
  estimator must implement `fit` (`:163`) and is used through `predict_proba`
  (`:256`).
- `:224` — `has_label = y != -1`: the unlabeled sentinel is `-1`.
- `:226-227` — `if np.all(has_label): warnings.warn("y contains no unlabeled
  samples", UserWarning)` — all-labeled input is a WARNING, not an error.
- `:229-239` — `criterion == "k_best"` and `k_best > n_unlabeled` ⇒ a
  `UserWarning` ("k_best is larger than the amount of unlabeled samples").
- `:241-245` — `transduction_ = np.copy(y)`; `labeled_iter_ = full_like(y, -1);
  labeled_iter_[has_label] = 0`; `n_iter_ = 0` (STARTS AT 0).
- `:247-288` — the self-training loop: `while not np.all(has_label) and
  (max_iter is None or n_iter_ < max_iter)`. The `np.all(has_label)` guard is
  evaluated BEFORE the first iteration ⇒ all-labeled-initially ⇒ loop body never
  runs ⇒ `n_iter_` stays 0. Inside: `n_iter_ += 1` (`:250`); refit
  `base_estimator_.fit(X[has_label], transduction_[has_label])` (`:251-253`);
  `prob = base_estimator_.predict_proba(X[~has_label])` (`:256`);
  `pred = classes_[np.argmax(prob, axis=1)]` (`:257`) — **MULTI-CLASS via argmax
  over `classes_`**; `max_proba = np.max(prob, axis=1)` (`:258`).
- `:261-262` — `criterion == "threshold"` ⇒ `selected = max_proba > self.threshold`
  — **STRICT `>`**: a sample with `max_proba == threshold` is NOT selected
  (verified live).
- `:263-269` — `criterion == "k_best"` ⇒ select the `k_best` highest-`max_proba`
  via `np.argpartition`.
- `:272-277` — map `selected` to original indices, write
  `transduction_[selected_full] = pred[selected]`, `has_label[selected_full] =
  True`, `labeled_iter_[selected_full] = n_iter_`.
- `:279-282` — if `selected_full.shape[0] == 0` ⇒
  `termination_condition_ = "no_change"; break`.
- `:290-293` — after the loop: `if n_iter_ == max_iter:
  termination_condition_ = "max_iter"`; `if np.all(has_label):
  termination_condition_ = "all_labeled"`.
- `:295-298` — a FINAL `base_estimator_.fit(X[has_label], transduction_[has_label])`
  always runs (even for `max_iter=0`), then `classes_ = base_estimator_.classes_`.
- `:301-322` — `predict(X)` returns `base_estimator_.predict(X)` — **class
  LABELS** (verified live: `predict(...)` → `[1]`, a class label).
- `:324-417` — `predict_proba` / `decision_function` / `predict_log_proba` /
  `score`, each `available_if`-guarded and delegating to `base_estimator_`.
- Attributes: `transduction_`, `labeled_iter_`, `n_iter_`,
  `termination_condition_`, `classes_`, `base_estimator_`.

## Requirements

R-DEV mental test applied per REQ ("numerical/API/structural contract" → MATCH;
"Cython/CPython footgun / RNG" → deviate; "missing feature / inverted contract" →
NOT-STARTED with a blocker). Per R-DEFER-2 / R-HONEST-2 classification is binary
(SHIPPED / NOT-STARTED).

- REQ-SELFTRAIN-LOOP (self-training iteration mechanic — binary, threshold
  criterion): ferrolearn `pub fn fit` separates labeled (`y != UNLABELED`) and
  unlabeled, then loops `for _iter in 0..max_iter`: gather labeled indices, refit
  the base via `(self.fit_fn)(&x_labeled, &y_labeled)`, predict on the unlabeled
  rows, pseudo-label every confident unlabeled sample, and break early on
  empty-unlabeled or zero-new-labels. This is the same ITERATION SHAPE as sklearn's
  `while`-loop (refit-on-labeled `:251-253` → predict-unlabeled `:256` →
  pseudo-label-confident `:272-277` → terminate-on-no-change `:279-282`) for
  `criterion='threshold'`, BINARY. **MATCH on the iteration shape only**
  (R-DEV-1 structural contract). SHIPPED for the loop MECHANIC; the selection
  COMPARISON (REQ-THRESHOLD-STRICT), threshold VALIDATION (REQ-THRESHOLD-BOUND),
  multi-class (REQ-MULTICLASS), the `n_iter`/attribute contract (REQ-ATTRS) and
  the `predict` output (REQ-PREDICT-LABELS) all diverge and are pinned separately.
  Honest scope: this REQ covers ONLY the refit→predict→pseudo-label→iterate
  skeleton; end-to-end transduction equality with sklearn is NOT claimed.
- REQ-THRESHOLD-STRICT (selection `>` vs `>=` — DETERMINISTIC FIXABLE DIVERGENCE):
  ferrolearn selects with `if max_prob >= self.threshold` (`fn fit`, the
  pseudo-label loop). sklearn uses `selected = max_proba > self.threshold`
  (`:262`, **STRICT `>`**). A sample whose `max_proba` is EXACTLY equal to
  `threshold` is SELECTED by ferrolearn but NOT by sklearn. **DEVIATE —
  deterministic FIXABLE** (R-DEV-1: the selection comparison is a numerical
  contract; the `==threshold` boundary is RNG-free). The critic pins a FAILING
  `#[test]` constructing a base whose score gives `max_prob == threshold` for an
  unlabeled sample and asserting it is LEFT unlabeled. Single-spot fixable
  (`>=` → `>`). NOT-STARTED (#1841).
- REQ-THRESHOLD-BOUND (threshold validation `[0,1)` vs `(0,1]` — DETERMINISTIC
  FIXABLE DIVERGENCE): ferrolearn rejects `self.threshold <= 0.0 ||
  self.threshold > 1.0` ⇒ accepts the interval `(0, 1]` (REJECTS 0.0, ALLOWS 1.0).
  sklearn `Interval(Real, 0.0, 1.0, closed="left")` (`:164`) accepts `[0, 1)`
  (ALLOWS 0.0, REJECTS 1.0) — the OPPOSITE endpoints (both verified live:
  `threshold=0.0` ACCEPTED, `threshold=1.0` → `InvalidParameterError`). **DEVIATE —
  deterministic FIXABLE** (R-DEV-2: a parameter-constraint contract). The critic
  pins a FAILING `#[test]`: `threshold(0.0)` should be ACCEPTED, `threshold(1.0)`
  should ERROR. Single-spot fixable (the two comparison operators). NOT-STARTED
  (#1842).
- REQ-MULTICLASS (multi-class via `predict_proba` argmax + `classes_`): ferrolearn
  pseudo-labels are BINARY-ONLY — `let predicted_label = if prob >= 0.5 { 1 } else
  { 0 }` from a SINGLE score (`fn fit`), with no `classes_`. sklearn `pred =
  self.base_estimator_.classes_[np.argmax(prob, axis=1)]` (`:257`) handles any
  number of classes through `predict_proba`'s `(n, n_classes)` matrix.
  **MATCH-intent / missing-feature — architectural** (needs a `predict_proba`-shaped
  base output + a `classes_` channel). NOT-STARTED (#1843).
- REQ-CRITERION-KBEST (`criterion='k_best'` + `k_best` param + over-budget
  warning): ABSENT — `SelfTrainingClassifier` has only `threshold`/`max_iter`
  builders, no `criterion`/`k_best`. sklearn selects the `k_best`
  highest-`max_proba` via `np.argpartition` (`:263-269`) and warns when `k_best >
  n_unlabeled` (`:229-239`). **MATCH-intent / missing-feature.** NOT-STARTED
  (#1844).
- REQ-PREDICT-LABELS (`predict` returns class labels): ferrolearn `impl
  Predict<Array2<f64>> for FittedSelfTrainingClassifier` returns
  `Array1<f64>` RAW SCORES (`(self.predict_fn)(x)`). sklearn `predict(X)` returns
  `base_estimator_.predict(X)` — class LABELS (`:322`, verified live: `predict(...)
  → [1]`). **DEVIATE — architectural output-contract divergence**, coupled to
  REQ-MULTICLASS (labels require the `classes_`/argmax machinery). NOT-STARTED
  (#1845).
- REQ-ATTRS (`transduction_` / `labeled_iter_` / `n_iter_` /
  `termination_condition_` semantics): ferrolearn exposes `transduced_labels()`
  (the `usize` working-label vector, including unfilled `UNLABELED` sentinels) and
  `n_iter()` — but `n_iter` is incremented at the TOP of every `for`-loop
  iteration (`n_iter += 1`), so all-labeled-initially yields `n_iter == 1`,
  whereas sklearn's `n_iter_` STARTS AT 0 and is incremented INSIDE the `while`
  AFTER the `np.all(has_label)` guard (`:247-250`), giving `n_iter_ == 0` for
  all-labeled input (verified live: all-labeled ⇒ `n_iter_=0`,
  `termination='all_labeled'`). There is NO `labeled_iter_` and NO
  `termination_condition_`. **DEVIATE — `n_iter` semantics + missing attributes.**
  NOT-STARTED (#1846).
- REQ-MAX-ITER (`max_iter=0` / `max_iter=None`): ferrolearn `max_iter: usize`
  REJECTS `0` (`if self.max_iter == 0 ⇒ InvalidParameter`) and has no `None`
  (unlimited) path. sklearn accepts `max_iter=0` — the `while` never runs, only the
  final fit (`:295`) executes, `termination='max_iter'` (verified live: `n_iter_=0`,
  `transduction` unchanged) — and `max_iter=None` for unlimited iteration (`:167`,
  `:248`). **DEVIATE / missing-feature.** NOT-STARTED (#1847).
- REQ-PROBA-DELEGATION (`predict_proba` / `decision_function` /
  `predict_log_proba` / `score` delegate to base): ABSENT —
  `FittedSelfTrainingClassifier` exposes only `Predict::predict` (raw scores),
  `transduced_labels()`, `n_iter()`. sklearn delegates each of these to
  `base_estimator_` under `available_if` (`:324-417`). **MATCH-intent /
  missing-feature.** NOT-STARTED (#1848).
- REQ-X-1 (R-SUBSTRATE ndarray→ferray-core): production code imports
  `use ndarray::{Array1, Array2}` and operates on `Array2<f64>`/`Array1<usize>`;
  `fn select_rows` builds `Array2`. The destination substrate is `ferray-core`
  (R-SUBSTRATE-1). NOT-STARTED (#1849).
- REQ-X-2 (non-test production consumer): the boundary meta-estimator types
  `SelfTrainingClassifier` / `FittedSelfTrainingClassifier` (+ the `UNLABELED`
  sentinel const) are the public API (S5 / R-DEFER-1) and are re-exported from
  `lib.rs`. SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side). The oracle is the
installed sklearn 1.5.2; run from `/tmp` (the source clone at
`/home/doll/scikit-learn` is the read-only cite tree, not built).

- AC-SELFTRAIN-LOOP (REQ-SELFTRAIN-LOOP — SHIPPED mechanic, binary threshold
  case): a separable binary problem (feature ≈ class) self-trains so the
  confident unlabeled samples are pseudo-labeled.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.base import BaseEstimator, ClassifierMixin
  from sklearn.semi_supervised import SelfTrainingClassifier
  class ColZero(ClassifierMixin, BaseEstimator):
      def fit(self, X, y):
          self.classes_=np.unique(y); self.n_features_in_=X.shape[1]; return self
      def predict_proba(self, X):
          s=np.clip(np.asarray(X)[:,0],0,1); return np.column_stack([1-s, s])
      def predict(self, X): return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
  X=np.array([[0.0],[0.1],[0.9],[1.0],[0.05],[0.95]])
  y=np.array([0,0,1,1,-1,-1])
  m=SelfTrainingClassifier(ColZero(), threshold=0.7).fit(X,y)
  print('transduction_:', m.transduction_.tolist())   # idx4->0, idx5->1
  "
  # -> transduction_: [0, 0, 1, 1, 0, 1]
  ```
  ferrolearn `feature0_fit_fn` self-training (`test_self_training_pseudo_labels_assigned`,
  `test_self_training_iterative_labeling`) pseudo-labels the same confident
  unlabeled samples (e.g. score 0.05 → 0, 0.95 → 1). SHIPPED on the iteration
  SHAPE only — the comparison/validation/multi-class/predict facets diverge below.
- AC-THRESHOLD-STRICT (REQ-THRESHOLD-STRICT — DETERMINISTIC FIXABLE, the critic's
  pin): a base giving an unlabeled sample `max_proba` EXACTLY equal to the
  threshold is NOT selected by sklearn (strict `>`).
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.base import BaseEstimator, ClassifierMixin
  from sklearn.semi_supervised import SelfTrainingClassifier
  class ExactProba(ClassifierMixin, BaseEstimator):
      def fit(self, X, y):
          self.classes_=np.array([0,1]); self.n_features_in_=X.shape[1]; return self
      def predict_proba(self, X):
          return np.array([[0.25,0.75] if r[0]==5.0 else [0.9,0.1] for r in np.asarray(X)])
      def predict(self, X): return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
  X=np.array([[0.],[1.],[5.]]); y=np.array([0,0,-1])
  m=SelfTrainingClassifier(ExactProba(), threshold=0.75).fit(X,y)
  print('transduction_:', m.transduction_.tolist(), 'termination:', m.termination_condition_)
  "
  # -> transduction_: [0, 0, -1] termination: no_change
  #    (unlabeled idx 2 has max_proba == 0.75 == threshold => NOT selected: strict >)
  ```
  ferrolearn `if max_prob >= self.threshold` (`fn fit`) WOULD select the
  `max_prob == 0.75 == threshold` sample (a score of 0.75 ⇒ `prob=0.75`,
  `max_prob=max(0.75, 0.25)=0.75`, `0.75 >= 0.75` true) ⇒ pseudo-labels it `1`,
  diverging from sklearn's `-1`. The critic pins a FAILING `#[test]` asserting the
  `== threshold` unlabeled sample stays `UNLABELED`. DETERMINISTIC / single-spot
  fixable (`>=` → `>`).
- AC-THRESHOLD-BOUND (REQ-THRESHOLD-BOUND — DETERMINISTIC FIXABLE, the critic's
  pin): the valid threshold INTERVAL is inverted.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.linear_model import LogisticRegression
  from sklearn.semi_supervised import SelfTrainingClassifier
  X=np.array([[0.],[1.],[2.],[3.]]); y=np.array([0,1,-1,-1])
  for t in (0.0, 1.0):
      try:
          SelfTrainingClassifier(LogisticRegression(), threshold=t).fit(X,y)
          print(f'threshold={t} -> ACCEPTED')
      except Exception as e:
          print(f'threshold={t} -> {type(e).__name__}')
  "
  # -> threshold=0.0 -> ACCEPTED
  # -> threshold=1.0 -> InvalidParameterError
  ```
  ferrolearn does the OPPOSITE: `threshold(0.0)` ⇒ `InvalidParameter`
  ("must be in (0, 1]"), `threshold(1.0)` ⇒ ACCEPTED (its
  `test_self_training_invalid_threshold` even asserts `0.0` is an error, and
  `test_self_training_high_threshold_no_labels` USES `threshold(1.0)`). The critic
  pins a FAILING `#[test]`: `threshold(0.0).fit(...)` should SUCCEED,
  `threshold(1.0).fit(...)` should ERROR. DETERMINISTIC / single-spot fixable
  (the two boundary operators).
- AC-MULTICLASS (REQ-MULTICLASS — ABSENT): sklearn pseudo-labels across `>2`
  classes via `predict_proba` argmax.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.linear_model import LogisticRegression
  from sklearn.semi_supervised import SelfTrainingClassifier
  X=np.array([[0.],[0.1],[5.],[5.1],[10.],[10.1],[5.05]])
  y=np.array([0,0,1,1,2,2,-1])
  m=SelfTrainingClassifier(LogisticRegression()).fit(X,y)
  print('classes_:', m.classes_.tolist(), 'transduction_:', m.transduction_.tolist())
  "
  # -> classes_: [0, 1, 2] transduction_: [0, 0, 1, 1, 2, 2, 1]
  ```
  ferrolearn pseudo-labels are `if prob >= 0.5 { 1 } else { 0 }` — only `{0,1}`
  exist; a 3-class problem is inexpressible. NOT-STARTED (#1843).
- AC-PREDICT-LABELS (REQ-PREDICT-LABELS — DIVERGENCE): sklearn `predict` returns
  class LABELS.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.linear_model import LogisticRegression
  from sklearn.semi_supervised import SelfTrainingClassifier
  X=np.array([[0.],[0.1],[5.],[5.1],[10.],[10.1],[5.05]])
  y=np.array([0,0,1,1,2,2,-1])
  m=SelfTrainingClassifier(LogisticRegression()).fit(X,y)
  print('predict([[5.05]]):', m.predict(np.array([[5.05]])).tolist())
  "
  # -> predict([[5.05]]): [1]   (a class LABEL)
  ```
  ferrolearn `FittedSelfTrainingClassifier::predict` returns
  `(self.predict_fn)(x)` — RAW `f64` SCORES, NOT labels
  (`test_self_training_predict` asserts `scores ≈ [0.2, 0.8]`, the raw column-0
  values). NOT-STARTED (#1845).
- AC-ATTRS (REQ-ATTRS — n_iter semantics + missing attrs): all-labeled-initially
  gives `n_iter_ == 0` and `termination_condition_ == 'all_labeled'` in sklearn.
  ```
  cd /tmp && python3 -c "
  import warnings; warnings.simplefilter('ignore')
  import numpy as np
  from sklearn.linear_model import LogisticRegression
  from sklearn.semi_supervised import SelfTrainingClassifier
  X=np.array([[0.],[0.1],[5.],[5.1]]); y=np.array([0,0,1,1])
  m=SelfTrainingClassifier(LogisticRegression()).fit(X,y)
  print('all-labeled n_iter_:', m.n_iter_, 'termination:', m.termination_condition_)
  print('has labeled_iter_:', hasattr(m,'labeled_iter_'), 'has termination_condition_:', hasattr(m,'termination_condition_'))
  "
  # -> all-labeled n_iter_: 0 termination: all_labeled
  # -> has labeled_iter_: True has termination_condition_: True
  ```
  ferrolearn `test_self_training_all_labeled` asserts `n_iter() == 1` (the
  `for`-loop runs once and increments `n_iter` before the empty-unlabeled break) —
  vs sklearn's `0`. There is no `labeled_iter_`/`termination_condition_`.
  NOT-STARTED (#1846).
- AC-MAX-ITER (REQ-MAX-ITER — ABSENT): sklearn accepts `max_iter=0` and
  `max_iter=None`.
  ```
  cd /tmp && python3 -c "
  import warnings; warnings.simplefilter('ignore')
  import numpy as np
  from sklearn.linear_model import LogisticRegression
  from sklearn.semi_supervised import SelfTrainingClassifier
  X=np.array([[0.],[0.1],[5.],[5.1],[10.],[10.1],[5.05]])
  y=np.array([0,0,1,1,2,2,-1])
  m0=SelfTrainingClassifier(LogisticRegression(), max_iter=0).fit(X,y)
  print('max_iter=0 n_iter_:', m0.n_iter_, 'termination:', m0.termination_condition_, 'transduction:', m0.transduction_.tolist())
  mN=SelfTrainingClassifier(LogisticRegression(), max_iter=None).fit(X,y)
  print('max_iter=None n_iter_:', mN.n_iter_)
  "
  # -> max_iter=0 n_iter_: 0 termination: max_iter transduction: [0, 0, 1, 1, 2, 2, -1]
  # -> max_iter=None n_iter_: 1
  ```
  ferrolearn `if self.max_iter == 0 ⇒ InvalidParameter` ("must be >= 1") and
  `max_iter: usize` has no `None`. `test_self_training_invalid_max_iter` asserts
  `max_iter(0)` is an error. NOT-STARTED (#1847).
- AC-PROBA-DELEGATION (REQ-PROBA-DELEGATION — ABSENT): sklearn exposes
  `predict_proba`/`decision_function`/`score` delegating to the base.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.linear_model import LogisticRegression
  from sklearn.semi_supervised import SelfTrainingClassifier
  X=np.array([[0.],[0.1],[5.],[5.1]]); y=np.array([0,0,1,-1])
  m=SelfTrainingClassifier(LogisticRegression()).fit(X,y)
  print('predict_proba shape:', m.predict_proba(X).shape, 'has score:', hasattr(m,'score'))
  "
  # -> predict_proba shape: (4, 2) has score: True
  ```
  `FittedSelfTrainingClassifier` exposes only `predict` (raw scores),
  `transduced_labels()`, `n_iter()` — no proba/decision/score delegation.
  NOT-STARTED (#1848).
- AC-X-2 (REQ-X-2 — SHIPPED): `grep -n "pub use self_training"
  ferrolearn-model-sel/src/lib.rs` shows the re-export
  `pub use self_training::{FittedSelfTrainingClassifier, SelfTrainingClassifier,
  UNLABELED};`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-SELFTRAIN-LOOP (self-training iteration mechanic — binary, threshold criterion) | SHIPPED | impl `pub fn fit in self_training.rs`: builds `labeled_mask = y != UNLABELED`, loops `for _iter in 0..max_iter` refitting `(self.fit_fn)(&x_labeled, &y_labeled)` on the labeled subset, predicts on the unlabeled rows (`pred_fn(&x_unlabeled)`), pseudo-labels every confident unlabeled sample (`labels[global_i] = predicted_label; labeled_mask[global_i] = true`), and breaks on empty-unlabeled (`unlabeled_idx.is_empty()`) or zero new labels (`new_labels_count == 0`). Mirrors sklearn's `while`-loop refit-on-labeled (`sklearn/semi_supervised/_self_training.py:251-253`) → predict-unlabeled (`:256`) → pseudo-label-confident (`:272-277`) → terminate-on-no-change (`:279-282`) for `criterion='threshold'`, BINARY. LIVE ORACLE (AC-SELFTRAIN-LOOP): a separable column-0 base with `threshold=0.7` pseudo-labels the unlabeled samples (`transduction_==[0,0,1,1,0,1]`); ferrolearn `test_self_training_pseudo_labels_assigned`/`test_self_training_iterative_labeling` pseudo-label the same confident rows. SHIPPED on the iteration SHAPE only — the comparison (#1841), validation (#1842), multi-class (#1843), `n_iter`/attrs (#1846) and `predict` output (#1845) facets diverge; end-to-end transduction equality is NOT claimed. Non-test consumer: REQ-X-2. |
| REQ-THRESHOLD-STRICT (selection `>` vs `>=` — DETERMINISTIC FIXABLE DIVERGENCE) | NOT-STARTED | open prereq blocker #1841. impl `pub fn fit in self_training.rs`: `if max_prob >= self.threshold { ... }` — INCLUSIVE `>=`. sklearn `selected = max_proba > self.threshold` (`sklearn/semi_supervised/_self_training.py:262`) — STRICT `>`. A sample with `max_proba == threshold` is SELECTED by ferrolearn, NOT by sklearn. LIVE ORACLE (AC-THRESHOLD-STRICT): a base giving an unlabeled sample `max_proba == 0.75 == threshold` ⇒ sklearn `transduction_==[0,0,-1]`, `termination='no_change'` (NOT pseudo-labeled); ferrolearn's `>=` pseudo-labels it `1`. DETERMINISTIC FIXABLE (the `==threshold` boundary is RNG-free): the critic pins a FAILING `#[test]` asserting the `==threshold` unlabeled sample stays `UNLABELED`. Single-spot fixable (`>=` → `>`). |
| REQ-THRESHOLD-BOUND (threshold validation `[0,1)` vs `(0,1]` — DETERMINISTIC FIXABLE DIVERGENCE) | NOT-STARTED | open prereq blocker #1842. impl `pub fn fit in self_training.rs`: `if self.threshold <= 0.0 \|\| self.threshold > 1.0 ⇒ InvalidParameter` — accepts `(0, 1]` (REJECTS 0.0, ALLOWS 1.0). sklearn `Interval(Real, 0.0, 1.0, closed="left")` (`sklearn/semi_supervised/_self_training.py:164`) accepts `[0, 1)` (ALLOWS 0.0, REJECTS 1.0) — the INVERTED endpoints. LIVE ORACLE (AC-THRESHOLD-BOUND): `threshold=0.0` → ACCEPTED, `threshold=1.0` → `InvalidParameterError`; ferrolearn `test_self_training_invalid_threshold` asserts `0.0` is an ERROR and `test_self_training_high_threshold_no_labels` USES `threshold(1.0)`. DETERMINISTIC FIXABLE: the critic pins a FAILING `#[test]` — `threshold(0.0)` SUCCEEDS, `threshold(1.0)` ERRORS. Single-spot fixable (the two boundary operators). |
| REQ-MULTICLASS (multi-class via predict_proba argmax + classes_) | NOT-STARTED | open prereq blocker #1843. impl `pub fn fit in self_training.rs`: `let predicted_label = if prob >= 0.5 { 1 } else { 0 }` from a SINGLE clamped score — BINARY-ONLY `{0,1}`, no `classes_`. sklearn `pred = self.base_estimator_.classes_[np.argmax(prob, axis=1)]` (`sklearn/semi_supervised/_self_training.py:257`) handles any `n_classes` via the `(n, n_classes)` `predict_proba` matrix. LIVE ORACLE (AC-MULTICLASS): a 3-class problem ⇒ `classes_==[0,1,2]`, `transduction_==[0,0,1,1,2,2,1]`; ferrolearn cannot express class 2 (only `{0,1}` are reachable). ARCHITECTURAL — needs a `predict_proba`-shaped base output + a `classes_` channel. Absent end-to-end. |
| REQ-CRITERION-KBEST (`criterion='k_best'` + `k_best` param + over-budget warning) | NOT-STARTED | open prereq blocker #1844. `SelfTrainingClassifier` has only `.threshold(f64)`/`.max_iter(usize)` builders — NO `criterion`/`k_best` fields (`grep -n "criterion\|k_best" self_training.rs` is empty). sklearn selects the `k_best` highest-`max_proba` via `np.argpartition(-max_proba, n_to_select)` (`sklearn/semi_supervised/_self_training.py:263-269`) and warns when `k_best > n_unlabeled` (`:229-239`). LIVE ORACLE: `SelfTrainingClassifier(LogisticRegression(), criterion='k_best', k_best=1)` selects one sample per round; ferrolearn has no such mode. Absent end-to-end. |
| REQ-PREDICT-LABELS (`predict` returns class labels) | NOT-STARTED | open prereq blocker #1845. impl `impl Predict<Array2<f64>> for FittedSelfTrainingClassifier in self_training.rs`: `fn predict(&self, x) -> Array1<f64>` returns `(self.predict_fn)(x)` — RAW `f64` SCORES. sklearn `predict(X)` returns `self.base_estimator_.predict(X)` — class LABELS (`sklearn/semi_supervised/_self_training.py:322`). LIVE ORACLE (AC-PREDICT-LABELS): `m.predict([[5.05]])==[1]` (a class label); ferrolearn `test_self_training_predict` asserts `scores ≈ [0.2, 0.8]` (raw column-0 values). ARCHITECTURAL output-contract divergence, COUPLED to REQ-MULTICLASS (labels need the `classes_`/argmax machinery). Absent end-to-end. |
| REQ-ATTRS (`transduction_` / `labeled_iter_` / `n_iter_` / `termination_condition_` semantics) | NOT-STARTED | open prereq blocker #1846. `FittedSelfTrainingClassifier in self_training.rs` exposes `transduced_labels()` (the `usize` working-label vector incl. unfilled `UNLABELED` sentinels) and `n_iter()`, but `n_iter += 1` is at the TOP of the `for`-loop ⇒ all-labeled-initially yields `n_iter == 1`. sklearn `n_iter_` STARTS AT 0 and increments INSIDE the `while` AFTER the `np.all(has_label)` guard (`sklearn/semi_supervised/_self_training.py:245-250`) ⇒ all-labeled ⇒ `n_iter_ == 0`. No `labeled_iter_`, no `termination_condition_`. LIVE ORACLE (AC-ATTRS): all-labeled ⇒ `n_iter_==0`, `termination=='all_labeled'`, `labeled_iter_`/`termination_condition_` present; ferrolearn `test_self_training_all_labeled` asserts `n_iter()==1`. DEVIATE — `n_iter` semantics + missing attributes. Absent end-to-end. |
| REQ-MAX-ITER (`max_iter=0` / `max_iter=None`) | NOT-STARTED | open prereq blocker #1847. impl `pub fn fit in self_training.rs`: `if self.max_iter == 0 ⇒ InvalidParameter` ("must be >= 1"); `max_iter: usize` has no `None`. sklearn `Interval(Integral, 0, None, closed="left")` OR `None` (`sklearn/semi_supervised/_self_training.py:167`) ⇒ `max_iter=0` runs only the final fit (`:295`), `max_iter=None` iterates unlimited (`:248`). LIVE ORACLE (AC-MAX-ITER): `max_iter=0` ⇒ `n_iter_=0`, `termination='max_iter'`, `transduction` unchanged; `max_iter=None` ⇒ `n_iter_=1`; ferrolearn `test_self_training_invalid_max_iter` asserts `max_iter(0)` is an error. DEVIATE / missing-feature. Absent end-to-end. |
| REQ-PROBA-DELEGATION (`predict_proba` / `decision_function` / `predict_log_proba` / `score` delegate to base) | NOT-STARTED | open prereq blocker #1848. `FittedSelfTrainingClassifier in self_training.rs` exposes only `impl Predict::predict` (raw scores), `transduced_labels()`, `n_iter()` — no proba/decision/log-proba/score surface (`grep -n "predict_proba\|decision_function\|fn score" self_training.rs` is empty). sklearn delegates each to `base_estimator_` under `available_if` (`sklearn/semi_supervised/_self_training.py:324-417`). LIVE ORACLE (AC-PROBA-DELEGATION): `m.predict_proba(X).shape==(4,2)`, `hasattr(m,'score')==True`; ferrolearn has neither. Absent end-to-end. |
| REQ-X-1 (R-SUBSTRATE ndarray→ferray-core) | NOT-STARTED | open prereq blocker #1849. Production code in `self_training.rs` imports `use ndarray::{Array1, Array2}` and operates on `Array2<f64>`/`Array1<usize>`/`Array1<f64>`; `fn select_rows` builds `Array2::from_shape_vec`. Per R-SUBSTRATE-1 the destination array type is `ferray-core`, not `ndarray`. Not migrated (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | Crate re-export: `ferrolearn-model-sel/src/lib.rs` `pub mod self_training;` + `pub use self_training::{FittedSelfTrainingClassifier, SelfTrainingClassifier, UNLABELED};`. Per S5 / R-DEFER-1 the boundary meta-estimator types ARE the public API and are grandfathered (existing pub surface). CAVEAT (honest underclaim): `grep -rn "SelfTrainingClassifier\|self_training" ferrolearn-*/src/ \| grep -v 'self_training.rs\|tests'` finds ONLY the `lib.rs` re-export + `//!` doc-table line — NO dedicated non-test internal CALLER and NO `ferrolearn-python` binding yet. SHIPPED on the boundary re-export per S5, not a dedicated production caller; the missing Python binding is noted. The base estimator is a `FitFn` CLOSURE (R-DEV-7 idiom for sklearn's wrapped/`clone`d `base_estimator`) — noted, not pinned. |

## Architecture

ferrolearn splits the estimator into an unfitted/Fitted pair (CLAUDE.md naming):
`SelfTrainingClassifier { fit_fn: FitFn, threshold: f64, max_iter: usize }`
(defaults `threshold = 0.75`, `max_iter = 10`, matching sklearn's `:174`/`:177`)
→ `FittedSelfTrainingClassifier { predict_fn: PredictFn, transduced_labels:
Array1<usize>, n_iter: usize }`. sklearn keeps a single `SelfTrainingClassifier`
(`:39`) whose post-`fit` state is `base_estimator_` + `transduction_` +
`labeled_iter_` + `n_iter_` + `termination_condition_` + `classes_`. The
`UNLABELED` sentinel const is `usize::MAX` — ferrolearn's `usize`-target analog of
sklearn's `-1` (`:224`), since `usize` cannot hold a negative sentinel.

**The base-estimator representation is an R-DEV-7 deviation.** sklearn wraps a
`base_estimator` object and `clone`s it each iteration (`:215`, `:251`);
ferrolearn takes a `fit_fn: FitFn` closure (`Box<dyn Fn(&Array2<f64>,
&Array1<usize>) -> Result<PredictFn, FerroError> + Send + Sync>`, shared with
`calibration.rs`) called once per iteration to fit the base and return a
`PredictFn` — noted, not pinned.

**The self-training loop (REQ-SELFTRAIN-LOOP — SHIPPED) is a faithful translation
of the iteration SHAPE for the binary `criterion='threshold'` case.** `fit`
separates labeled/unlabeled by the `UNLABELED` sentinel, refits on the labeled
subset, predicts on the unlabeled rows, pseudo-labels every confident sample, and
breaks on empty-unlabeled or zero-new-labels — exactly sklearn's
refit→predict→select→terminate cadence (`:251-282`). Verified against the live
oracle on a separable column-0 base: the confident unlabeled samples are
pseudo-labeled to the same `{0,1}` classes.

**BUT the SHIPPED surface is rigorously NARROW, and several contract facets are
INVERTED or ABSENT vs sklearn.** The two DETERMINISTIC FIXABLE divergences the
critic should pin are:
- **REQ-THRESHOLD-STRICT (#1841):** ferrolearn selects with `max_prob >=
  threshold` where sklearn uses STRICT `max_proba > threshold` (`:262`). At
  `max_proba == threshold` ferrolearn pseudo-labels, sklearn does not — oracle-
  verified, RNG-free, single-spot fixable (`>=` → `>`).
- **REQ-THRESHOLD-BOUND (#1842):** ferrolearn validates `threshold ∈ (0, 1]`
  (rejects 0.0, allows 1.0) where sklearn's `_parameter_constraints` is
  `[0, 1)` (allows 0.0, rejects 1.0, `:164`) — the INVERTED endpoints, oracle-
  verified, single-spot fixable.

The remaining gaps are architectural / missing-feature blockers, all rooting in
ferrolearn being **binary-only, score-returning, threshold-criterion-only**:
multi-class via `predict_proba` argmax + `classes_` (REQ-MULTICLASS, #1843);
`criterion='k_best'` + `k_best` + over-budget warning (REQ-CRITERION-KBEST,
#1844); `predict`-returns-LABELS (REQ-PREDICT-LABELS, #1845, coupled to
multi-class); the `transduction_`/`labeled_iter_`/`n_iter_`/
`termination_condition_` attribute contract incl. the `n_iter` off-by-one for
all-labeled input (REQ-ATTRS, #1846); `max_iter=0`/`max_iter=None`
(REQ-MAX-ITER, #1847); the `predict_proba`/`decision_function`/
`predict_log_proba`/`score` delegation surface (REQ-PROBA-DELEGATION, #1848); and
the `ferray-core` substrate (REQ-X-1, #1849). SHIPPED: the binary
threshold-criterion loop mechanic (REQ-SELFTRAIN-LOOP) and the boundary re-export
(REQ-X-2).

Invariants: `y.len() == x.nrows()` (`FerroError::ShapeMismatch`); `>= 1` labeled
sample (`FerroError::InsufficientSamples`); `threshold ∈ (0, 1]` and `max_iter >=
1` (`FerroError::InvalidParameter` — both INVERTED/narrower vs sklearn, see
REQ-THRESHOLD-BOUND/REQ-MAX-ITER); pseudo-labels are `{0, 1}`; unfilled unlabeled
samples retain `UNLABELED` in `transduced_labels()`; `predict` returns raw `f64`
scores (NOT labels).

## Verification

Commands establishing the SHIPPED claims (baseline
`1fe9a1bf5b1d2f39d8d6d80f3bc4f09f6ffb2d31`). The oracle is the installed sklearn
1.5.2 (`cd /tmp`; the source clone at `/home/doll/scikit-learn` is the read-only
cite tree):

- `cargo test -p ferrolearn-model-sel --lib self_training` → the in-file
  `self_training::tests` pass (`test_self_training_all_labeled`,
  `test_self_training_pseudo_labels_assigned`,
  `test_self_training_high_threshold_no_labels`,
  `test_self_training_no_labeled_samples`, `test_self_training_shape_mismatch`,
  `test_self_training_invalid_threshold`, `test_self_training_invalid_max_iter`,
  `test_self_training_predict`, `test_self_training_max_iter_respected`,
  `test_self_training_iterative_labeling`).
- REQ-SELFTRAIN-LOOP SHIPPED oracle (live sklearn, separable column-0 base,
  R-CHAR-3): AC-SELFTRAIN-LOOP — `SelfTrainingClassifier(ColZero(), threshold=0.7)
  .fit(X, y)` pseudo-labels the confident unlabeled rows
  (`transduction_==[0,0,1,1,0,1]`); ferrolearn `test_self_training_pseudo_labels_assigned`
  /`test_self_training_iterative_labeling` MATCH on the same confident rows.
  SHIPPED on the loop SHAPE only.
- REQ-THRESHOLD-STRICT DETERMINISTIC FIXABLE oracle (#1841 — the critic's pin):
  AC-THRESHOLD-STRICT — a base giving an unlabeled sample `max_proba == 0.75 ==
  threshold` ⇒ sklearn `transduction_==[0,0,-1]`, `termination=='no_change'` (NOT
  selected, strict `>`); ferrolearn's `>=` pseudo-labels it `1`. The critic pins a
  FAILING `#[test]` asserting the `==threshold` unlabeled sample stays
  `UNLABELED`.
- REQ-THRESHOLD-BOUND DETERMINISTIC FIXABLE oracle (#1842 — the critic's pin):
  AC-THRESHOLD-BOUND — `threshold=0.0` → ACCEPTED, `threshold=1.0` →
  `InvalidParameterError`; ferrolearn does the OPPOSITE (`0.0` errors, `1.0`
  accepted). The critic pins a FAILING `#[test]`: `threshold(0.0).fit(...)`
  SUCCEEDS, `threshold(1.0).fit(...)` ERRORS.
- REQ-MULTICLASS ABSENT oracle (#1843): AC-MULTICLASS — a 3-class problem ⇒
  `classes_==[0,1,2]`, `transduction_==[0,0,1,1,2,2,1]`; ferrolearn pseudo-labels
  only `{0,1}`.
- REQ-PREDICT-LABELS DIVERGENCE oracle (#1845): AC-PREDICT-LABELS —
  `m.predict([[5.05]])==[1]` (a class LABEL); ferrolearn `predict` returns raw
  `f64` scores (`test_self_training_predict` ⇒ `[0.2, 0.8]`).
- REQ-ATTRS DIVERGENCE oracle (#1846): AC-ATTRS — all-labeled ⇒ `n_iter_==0`,
  `termination=='all_labeled'`, `labeled_iter_`/`termination_condition_` present;
  ferrolearn `test_self_training_all_labeled` asserts `n_iter()==1` and has neither
  attribute.
- REQ-MAX-ITER ABSENT oracle (#1847): AC-MAX-ITER — `max_iter=0` ⇒ `n_iter_=0`,
  `termination='max_iter'`, transduction unchanged; `max_iter=None` ⇒ `n_iter_=1`;
  ferrolearn rejects `max_iter(0)` and has no `None`.
- REQ-CRITERION-KBEST ABSENT oracle (#1844):
  `SelfTrainingClassifier(LogisticRegression(), criterion='k_best', k_best=1)`
  selects per round; ferrolearn has no `criterion`/`k_best`.
- REQ-PROBA-DELEGATION ABSENT oracle (#1848): AC-PROBA-DELEGATION —
  `m.predict_proba(X).shape==(4,2)`, `hasattr(m,'score')==True`; ferrolearn
  exposes only `predict`/`transduced_labels()`/`n_iter()`.
- REQ-X-1 substrate (#1849): `grep -n "ndarray" ferrolearn-model-sel/src/
  self_training.rs` shows `use ndarray::{Array1, Array2}` — wrong substrate,
  migration owed.
- REQ-X-2 consumer: `grep -n "pub use self_training" ferrolearn-model-sel/src/
  lib.rs` shows `pub use self_training::{FittedSelfTrainingClassifier,
  SelfTrainingClassifier, UNLABELED};`. `grep -rn "SelfTrainingClassifier"
  ferrolearn-*/src/ | grep -v 'self_training.rs\|tests'` shows the re-export + a
  `//!` doc line only (no dedicated internal caller, no Python binding — honest
  underclaim).

SHIPPED (2): REQ-SELFTRAIN-LOOP (binary `criterion='threshold'` self-training
iteration SHAPE — refit→predict→pseudo-label→iterate, VERIFIED against the live
oracle on a separable column-0 base; the comparison/validation/multi-class/attrs/
predict facets diverge), REQ-X-2 (boundary re-export consumer; no dedicated caller
/ no Python binding — honest underclaim). NOT-STARTED (9): REQ-THRESHOLD-STRICT
(#1841 — DETERMINISTIC FIXABLE), REQ-THRESHOLD-BOUND (#1842 — DETERMINISTIC
FIXABLE), REQ-MULTICLASS (#1843 — architectural), REQ-CRITERION-KBEST (#1844),
REQ-PREDICT-LABELS (#1845 — architectural, label-output), REQ-ATTRS (#1846 —
n_iter semantics + missing attrs), REQ-MAX-ITER (#1847), REQ-PROBA-DELEGATION
(#1848), REQ-X-1 (#1849 — ferray substrate).

Per R-DEFER-2 every REQ is binary SHIPPED/NOT-STARTED. The DETERMINISTIC FIXABLE
divergences the critic should pin as FAILING tests are **REQ-THRESHOLD-STRICT**
(#1841 — `max_prob >= threshold` vs sklearn STRICT `max_proba > threshold` `:262`;
a `==threshold` unlabeled sample is wrongly pseudo-labeled; single-spot fixable
`>=` → `>`) and **REQ-THRESHOLD-BOUND** (#1842 — threshold interval `(0, 1]` vs
sklearn `[0, 1)` `:164`; `0.0` should be accepted and `1.0` rejected, the inverse
of ferrolearn; single-spot fixable). Everything else is architectural / missing:
REQ-MULTICLASS (#1843) + REQ-PREDICT-LABELS (#1845) need the `predict_proba`
argmax + `classes_` + label-output machinery; REQ-ATTRS (#1846), REQ-MAX-ITER
(#1847), REQ-CRITERION-KBEST (#1844), REQ-PROBA-DELEGATION (#1848) are missing
parameters/attributes/surface; REQ-X-1 (#1849) is the ferray substrate. The
SHIPPED surface is one narrow strip: the binary, score-returning,
threshold-criterion iteration loop.

Least-confident SHIPPED claim: REQ-SELFTRAIN-LOOP — SHIPPED rests on the
ITERATION SHAPE (refit→predict→pseudo-label→iterate) matching the live oracle's
confident-sample pseudo-labeling on a binary separable base, NOT on end-to-end
transduction equality: the `>=`-vs-`>` selection (#1841) and `(0,1]`-vs-`[0,1)`
validation (#1842) deterministically diverge at the boundary, the labels are
binary-only (#1843), and the `n_iter` count is off-by-one for all-labeled input
(#1846). The honest reading is "the binary threshold-criterion self-training
skeleton ships; its comparison/validation boundaries are inverted and everything
multi-class/label/attribute is absent" — a follow-up critic pins #1841/#1842 and
builders reconcile the architectural REQs.
