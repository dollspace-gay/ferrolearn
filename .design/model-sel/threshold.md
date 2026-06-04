# FixedThresholdClassifier & TunedThresholdClassifierCV

<!--
tier: 3-component
status: draft
baseline-commit: 9cadb9673ecddc6661f3967ea7983696902f8053
upstream-paths:
  - sklearn/model_selection/_classification_threshold.py   # FixedThresholdClassifier (:233), TunedThresholdClassifierCV (:619), _threshold_scores_to_class_labels (:57)
-->

## Summary

`ferrolearn-model-sel/src/threshold.rs` mirrors scikit-learn's two
decision-threshold meta-classifiers in
`sklearn/model_selection/_classification_threshold.py`:
`FixedThresholdClassifier` (`:233`) and `TunedThresholdClassifierCV` (`:619`).
Both wrap a binary base estimator that produces a per-sample score and convert
that score to a hard class label by comparing it to a decision threshold —
`FixedThresholdClassifier` with a user-supplied constant threshold,
`TunedThresholdClassifierCV` with a threshold selected by cross-validation to
maximise a scoring metric.

ferrolearn ships the **core thresholding comparison** (`score >= threshold`)
end-to-end for both estimators, and the **structural CV threshold search** for
`TunedThresholdClassifierCV`. It does NOT ship sklearn's `threshold='auto'`
resolution, arbitrary `pos_label`/`classes_`, the estimator-wrapping +
`response_method` dispatch (ferrolearn takes a score-producing closure instead —
a Rust idiom, R-DEV-7), or `TunedThresholdClassifierCV`'s default behaviour
(`scoring='balanced_accuracy'`, `thresholds=100` auto grid, StratifiedKFold CV,
`refit`, `store_cv_results`). The R-SUBSTRATE migration to `ferray-core` is also
NOT-STARTED.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### Shared base
- `:57-66` — `_threshold_scores_to_class_labels(y_score, threshold, classes, pos_label)`:
  the thresholding kernel. With `pos_label=None` the label map is `np.array([0, 1])`
  (`:60`); otherwise it is `[neg_label_idx, pos_label_idx]` derived from `classes`
  (`:62-64`). The label is `classes[map[(y_score >= threshold).astype(int)]]`
  (`:66`) — a `>=` comparison, so a score exactly equal to the threshold maps to
  the positive class.
- `:69-230` — `BaseThresholdClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator)`:
  wraps `estimator` + `response_method="auto"` (`:105-107`); `fit` requires
  `type_of_target(y) == "binary"` else `ValueError` (`:145-149`); `classes_`
  delegates to `estimator_.classes_` (`:160-163`); `_get_response_method`
  resolves `"auto"` to `["predict_proba", "decision_function"]` (`:109-115`);
  exposes `predict_proba`/`predict_log_proba`/`decision_function` via
  `available_if` passthrough to the fitted estimator (`:165-217`).

### FixedThresholdClassifier (`:233`)
- `:321-325` — `_parameter_constraints`: `threshold` is `StrOptions({"auto"})` or
  `Real`; `pos_label` is `[Real, str, "boolean", None]`.
- `:327-337` — `__init__(self, estimator, *, threshold="auto", pos_label=None,
  response_method="auto")` (default `threshold="auto"`, `pos_label=None`).
- `:339-361` — `_fit`: `self.estimator_ = clone(self.estimator).fit(X, y, ...)`.
- `:363-392` — `predict`: gets `y_score` via `_get_response_values_binary` over
  the resolved response method (`:377-383`); `threshold == "auto"` resolves to
  `0.5` for `predict_proba`, `0.0` for `decision_function` (`:385-388`); then
  `_threshold_scores_to_class_labels(y_score, decision_threshold, self.classes_,
  self.pos_label)` (`:390-391`).

### TunedThresholdClassifierCV (`:619`)
- `:782-799` — `_parameter_constraints`: `scoring` is a scorer name / callable /
  mapping; `thresholds` is `Interval(Integral, 1, None)` or array-like; `cv` is a
  cv-object / `"prefit"` / a float in (0,1); `refit`/`store_cv_results` boolean.
- `:801-821` — `__init__(self, estimator, *, scoring="balanced_accuracy",
  response_method="auto", thresholds=100, cv=None, refit=True, n_jobs=None,
  random_state=None, store_cv_results=False)`.
- `:823-940` — `_fit`: `cv = check_cv(self.cv, y=y, classifier=True)` (default
  `None` → 5-fold **StratifiedKFold**, `:858`); per split, `_fit_and_score_over_thresholds`
  fits a cloned classifier on the train fold and computes a score CURVE over
  candidate thresholds via `_CurveScorer` (`:892-906`); for an integer
  `thresholds`, the candidate grid is `np.linspace(min_threshold, max_threshold,
  num=self.thresholds)` spanning the observed score range (`:921-924`); the
  per-fold curves are combined by `_mean_interpolated_score` (interpolating each
  fold's curve onto a common grid and averaging across folds, `:591-616`,
  `:928-930`); `best_idx = objective_scores.argmax()` (`:931`) sets
  `best_score_`/`best_threshold_`; `store_cv_results` saves the grid + mean scores
  (`:934-938`). When `refit` (default) the final `estimator_` is trained on the
  whole set (`:878-890`).
- `:439-515` — `_CurveScorer._score`: for an integer `thresholds`, builds the
  per-fold grid as `np.linspace(np.min(y_score), np.max(y_score), self._thresholds)`
  (`:498-501`) — the grid is the score range, NOT a fixed `[0,1]`.
- `:942-966` — `predict`: `_threshold_scores_to_class_labels(y_score,
  self.best_threshold_, self.classes_, pos_label)`.

## Requirements

- REQ-1: **FixedThresholdClassifier predict thresholding parity (DETERMINISTIC).**
  `FittedFixedThresholdClassifier::predict` maps each score `s` to `1` if
  `s >= threshold` else `0`, mirroring sklearn's
  `_threshold_scores_to_class_labels` `(y_score >= threshold)` for the binary
  `{0,1}`/`pos_label=1` case (`:60`, `:66`). Includes the `>=` edge: a score
  exactly equal to the threshold maps to the positive class. Oracle-pinnable.
- REQ-2: **`threshold='auto'` resolution.** sklearn defaults `threshold="auto"`,
  resolving to `0.5` for a `predict_proba` response and `0.0` for a
  `decision_function` response (`:385-388`); ferrolearn REQUIRES an explicit
  `threshold: f64`. Feature gap.
- REQ-3: **`pos_label` / arbitrary `classes_`.** sklearn supports an arbitrary
  positive label and class ordering, mapping the thresholded `{0,1}` index back
  through `classes` (`:62-66`, `:259-262`); ferrolearn hard-codes binary `{0,1}`
  (`predict` returns `usize` 0/1, no `classes_`). Feature gap.
- REQ-4: **estimator-wrapping + `response_method` dispatch / closure idiom
  (R-DEV-7).** sklearn wraps an `estimator` and dispatches over
  `response_method='auto'` (`predict_proba`→`decision_function`, `:109-115`,
  `:377`); ferrolearn takes a `FitScoreFn` CLOSURE that returns a `ScoreFn`, so
  the response-method choice lives in the caller's closure. The closure is the
  sanctioned Rust analog of the wrapped-estimator pattern (R-DEV-7), but the
  observable `response_method`/`predict_proba`/`predict_log_proba` surface is not
  exposed. `decision_function` passthrough IS present.
- REQ-5: **TunedThresholdClassifierCV CV threshold-selection structure.**
  `TunedThresholdClassifierCV::fit` runs k-fold CV, collects out-of-fold scores,
  and picks the candidate threshold maximising a scoring closure — structurally
  the same shape as sklearn's argmax-over-CV-scores selection (`:931`). KFold is
  unshuffled by default, so the selection is deterministic given the data (no RNG
  in the default path). BUT the fold-combination differs: ferrolearn pools all
  folds into a single OOF score vector and scores each threshold ONCE globally,
  whereas sklearn scores a curve per fold and `_mean_interpolated_score`-averages
  across folds (`:591-616`, `:928-930`). Structural-parity candidate; the
  per-fold-mean combination is part of the REQ-6 default gap.
- REQ-6: **TunedThresholdClassifierCV default-behaviour parity.** sklearn's
  defaults — `scoring='balanced_accuracy'` (`:805`), `thresholds=100` auto grid
  over the observed score range (`:921-924`, `:498-501`), `cv=None` → 5-fold
  **StratifiedKFold** (`:858`), `refit=True`, `store_cv_results`, plus the
  per-fold mean-interpolated scoring — none hold in ferrolearn: `scoring` and
  `cv` are required arguments, the default grid is a fixed `[0.0, 0.05, …, 1.0]`
  (not the score range), the splitter is plain unshuffled KFold (not stratified),
  `cv_results` exposure is partial, and there is no `response_method`. Feature
  gap.
- REQ-7: **R-SUBSTRATE.** `threshold.rs` is on `ndarray`, not `ferray-core`.
  Feature gap.
- REQ-8: **Non-test production consumer.** Re-export only (S5).

## Acceptance criteria

- AC-1 (REQ-1): with scores `[0.1,0.6,0.4,0.9,0.5]` and `threshold=0.5`,
  `FittedFixedThresholdClassifier::predict` yields `[0,1,0,1,1]` — matching the
  live oracle `FixedThresholdClassifier(est, threshold=0.5,
  response_method='decision_function').predict(X)` over an estimator whose
  `decision_function` returns column 0. The score `0.5 == threshold` row maps to
  class 1 (the `>=` edge). DETERMINISTIC / oracle-pinnable.
- AC-2 (REQ-2): a `FixedThresholdClassifier` constructed without an explicit
  threshold (the `"auto"` default) resolves to 0.5 (proba) / 0.0 (decision) — no
  ferrolearn surface accepts an absent/`"auto"` threshold.
- AC-3 (REQ-3): a `FixedThresholdClassifier` over `pos_label`-shifted or
  string-labelled `y` returns labels in the original class space — ferrolearn's
  `predict` returns only `usize` `{0,1}`.
- AC-4 (REQ-4): `predict_proba`/`predict_log_proba`/`response_method` selection is
  observable on the sklearn estimator; ferrolearn exposes only
  `decision_function` (raw score passthrough) and the closure-supplied score.
- AC-5 (REQ-5): for a separable column-0 scorer with `y=[0,0,0,1,1,1]` and grid
  `[0.1,0.3,0.5,0.7,0.9]`, `best_threshold()` selects the score-maximising
  threshold; the structural "argmax over candidate thresholds across CV folds"
  matches sklearn's `objective_scores.argmax()` shape, while acknowledging the
  pooled-OOF vs per-fold-mean-interpolation difference.
- AC-6 (REQ-6): on a realistic dataset
  `TunedThresholdClassifierCV(est)` (all defaults) selects a `best_threshold_`
  off a `linspace(min_score, max_score, 100)` grid via balanced-accuracy on
  StratifiedKFold — ferrolearn requires explicit `scoring`/`cv`, uses a fixed
  `[0,1]` step-0.05 grid, and plain KFold.
- AC-7 (REQ-7): owned computation runs on `ferray-core`, no `ndarray`.
- AC-8 (REQ-8): the estimators are constructed from non-test production code.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (FixedThresholdClassifier predict parity) | SHIPPED | impl `pub fn predict` on `FittedFixedThresholdClassifier` in `threshold.rs` returns `scores.mapv(\|s\| if s >= self.threshold { 1usize } else { 0 })`. Mirrors sklearn `_threshold_scores_to_class_labels` `classes[map[(y_score >= threshold).astype(int)]]` for the `pos_label=None` → `[0,1]` binary case (`sklearn/model_selection/_classification_threshold.py:60`, `:66`); the `>=` comparison means a score exactly equal to the threshold → class 1, identical to ferrolearn's `>=`. DETERMINISTIC / oracle-pinnable: scores `[0.1,0.6,0.4,0.9,0.5]`, `threshold=0.5` → live oracle `FixedThresholdClassifier(est, threshold=0.5, response_method='decision_function').predict(X)` `[0,1,0,1,1]` (the `0.5==0.5` row → 1; Verification). The `decision_function` passthrough `pub fn decision_function` returns the raw score, matching sklearn's `decision_function` (`:201-217`). Test: `fixed_threshold_basic`. CAVEAT (R-CHAR-3): the existing `#[test]` derives its expected vector by hand and does not exercise the `s == threshold` edge — the critic should pin an oracle-grounded `#[test]` on `[…,0.5]`/`t=0.5` expecting `[0,1,0,1,1]` from the live call above. Non-test consumer: REQ-8. |
| REQ-2 (`threshold='auto'` resolution) | NOT-STARTED | open prereq blocker (tracking #1736; critic to file specific). `FixedThresholdClassifier::new(fit_fn, threshold: f64)` REQUIRES an explicit `f64`; there is no `"auto"`/`Option` path and no `response_method` to resolve against. sklearn defaults `threshold="auto"`, resolving to `0.5` for `predict_proba` and `0.0` for `decision_function` (`sklearn/.../_classification_threshold.py:331`, `:385-388`). Live oracle: `FixedThresholdClassifier(LogisticRegression()).fit(X,y).predict(X)` uses the 0.5 proba auto-threshold; ferrolearn cannot express the unset/`"auto"` state. Absent end-to-end. |
| REQ-3 (`pos_label` / arbitrary `classes_`) | NOT-STARTED | open prereq blocker (tracking #1736). `predict` returns `Array1<usize>` of `{0,1}`; there is no `pos_label`, no `classes_`, no remapping of the thresholded index. sklearn maps `(y_score >= threshold)` through `[neg_label_idx, pos_label_idx]` into the estimator's `classes` (`:62-66`) and accepts `pos_label` `[Real, str, "boolean", None]` (`:324`), defaulting to 1 only for `{-1,1}`/`{0,1}` targets (`:259-262`). Absent end-to-end. |
| REQ-4 (estimator-wrapping + response_method / closure idiom, R-DEV-7) | NOT-STARTED | open prereq blocker (tracking #1736). ferrolearn takes a `FitScoreFn` closure (`Box<dyn Fn(&Array2, &Array1<usize>) -> Result<ScoreFn, _>>`) returning a `ScoreFn`; the response-method selection lives in the caller's closure, which is the sanctioned Rust analog (R-DEV-7) of sklearn's wrapped `estimator` + `response_method` dispatch (`:105-115`, `:377-383`). HOWEVER the observable contract sklearn exposes — `predict_proba`/`predict_log_proba` passthrough (`:165-199`), `response_method='auto'` (predict_proba→decision_function) selection, and the `clone(estimator).fit` wrapping — is NOT present: `FittedFixedThresholdClassifier` exposes only `decision_function` (raw score) and `predict`. SHIPPED-adjacent on the score-passthrough (`decision_function` mirrors `:201-217`), but the proba surface + response-method dispatch are absent, so the REQ is NOT-STARTED on the wrapped-estimator/response_method contract. The closure idiom itself is sound (R-DEV-7), pinned here as a deliberate divergence, not a bug. |
| REQ-5 (TunedThresholdClassifierCV CV threshold-selection structure) | SHIPPED | impl `pub fn fit` on `TunedThresholdClassifierCV` in `threshold.rs`: builds folds via `KFold::new(self.cv).fold_indices(n)`, fits the base on each train fold and scores the test fold into a pooled out-of-fold vector `oof_scores`, then for each candidate `thr` computes `oof_scores.mapv(\|s\| if s >= thr { 1 } else { 0 })`, scores via the `scoring` closure, and keeps the argmax (`if score > best_score`); refits the base on the full set with the chosen threshold. This mirrors sklearn's structural "pick the threshold maximising the CV objective" (`objective_scores.argmax()`, `sklearn/.../_classification_threshold.py:931`) and the refit-on-whole-set path (`:878-890`). DETERMINISTIC: `KFold::new` defaults `shuffle: false` (`cross_validation.rs`), so `fold_indices` is contiguous and RNG-free → selection is deterministic given the data; no `random_state`/RNG carve-out applies to the default path. Test: `tuned_threshold_picks_best` (separable `y=[0,0,0,1,1,1]`, grid `[0.1,0.3,0.5,0.7,0.9]` → `best_threshold()==0.5`), `tuned_threshold_rejects_cv1`, `tuned_threshold_shape_mismatch`. CAVEAT / honest underclaim: ferrolearn pools ALL folds into ONE OOF vector and scores each threshold once globally; sklearn scores a per-fold curve and combines them with `_mean_interpolated_score` (interpolate each fold onto a common grid, average across folds, `:591-616`, `:928-930`). So the selection MECHANISM (argmax over a CV-derived score per threshold) is shipped, but the fold-combination ARITHMETIC diverges — folded into REQ-6. SHIPPED on the structural argmax-over-CV selection only. Non-test consumer: REQ-8. |
| REQ-6 (TunedThresholdClassifierCV default-behaviour parity) | NOT-STARTED | open prereq blocker (tracking #1736; critic to file specifics). Multiple defaults diverge: (a) **scoring** — sklearn defaults `scoring='balanced_accuracy'` (`:805`); ferrolearn requires an explicit `ThresholdScoring` closure (no default metric). (b) **thresholds grid** — sklearn `thresholds=100` builds `np.linspace(min_score, max_score, 100)` over the OBSERVED score range (`:921-924`, `:498-501`); ferrolearn defaults a FIXED `(0..=20).map(\|i\| i*0.05)` = `[0,0.05,…,1.0]` grid (`new`), wrong whenever scores fall outside `[0,1]` (e.g. `decision_function` logits). (c) **cv** — sklearn `cv=None` → 5-fold **StratifiedKFold** (`:858`); ferrolearn uses plain unshuffled `KFold` and requires an explicit `cv: usize`. (d) **fold combination** — per-fold curve + `_mean_interpolated_score` (`:591-616`) vs ferrolearn's single pooled OOF vector (REQ-5 caveat). (e) **store_cv_results / cv_results_** — ferrolearn exposes `cv_scores()`/`thresholds()`/`best_score()` accessors but no `store_cv_results` toggle or `cv_results_` dict shape. (f) **response_method / predict_proba** — absent (REQ-4). Live oracle: `TunedThresholdClassifierCV(RandomForestClassifier(random_state=0)).fit(...)` finds `best_threshold_ ≈ 0.342` off the auto grid + balanced_accuracy + StratifiedKFold (docstring `:769`) — unreproducible in ferrolearn. Absent end-to-end. |
| REQ-7 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker (tracking #1736; R-SUBSTRATE-2/3). `threshold.rs` imports `ndarray::{Array1, Array2}` and operates on `Array2<f64>`/`Array1<f64>`/`Array1<usize>`; `subset_rows`/`subset_rows_1d` build `ndarray` arrays. Destination substrate is the `ferray-core` array type (R-SUBSTRATE-1). Not migrated. |
| REQ-8 (consumer) | SHIPPED | Crate re-export: `lib.rs` `pub use threshold::{FitScoreFn, FittedFixedThresholdClassifier, FittedTunedThresholdClassifierCV, FixedThresholdClassifier, ThresholdScoring, TunedThresholdClassifierCV}` (and `pub mod threshold`). R-DEFER-1 / S5: boundary meta-estimator types ARE the public API; existing pub surface grandfathered. CAVEAT (honest underclaim): `grep` finds NO non-test, non-re-export caller and NO `ferrolearn-python` binding for either estimator (the only `*Threshold*` matches outside `threshold.rs` are the `lib.rs` re-export lines). SHIPPED on the boundary re-export per S5, not a dedicated production caller; the missing Python binding is noted. |

## Architecture

ferrolearn splits each estimator into an unfitted/Fitted pair (CLAUDE.md naming).
`FixedThresholdClassifier { fit_fn: FitScoreFn, threshold: f64 }` fits to
`FittedFixedThresholdClassifier { scorer: ScoreFn, threshold: f64 }`;
`TunedThresholdClassifierCV { fit_fn, cv: usize, thresholds: Vec<f64>, scoring }`
fits to `FittedTunedThresholdClassifierCV { scorer, best_threshold, best_score,
cv_scores, thresholds }`. sklearn keeps a single class each
(`FixedThresholdClassifier`, `TunedThresholdClassifierCV`) whose post-`fit` state
is `estimator_` (+ `best_threshold_`/`best_score_`/`cv_results_` for the tuned
variant).

The central design divergence is the **score source**. sklearn wraps an
`estimator` and selects a `response_method` (`predict_proba`→`decision_function`,
`:109-115`) to produce `y_score`; ferrolearn instead takes a `FitScoreFn` closure
— `Box<dyn Fn(&Array2, &Array1<usize>) -> Result<ScoreFn, FerroError>>` — that
trains the base model and returns a `ScoreFn` mapping `X` to per-sample scores.
The caller therefore owns the response-method decision; the closure is the
sanctioned Rust analog (R-DEV-7) of the wrapped-estimator pattern, mirroring the
calibration/self-training factory shape elsewhere in this crate. The observable
cost is REQ-4 (no `predict_proba`/`predict_log_proba`, no `response_method`
surface) and REQ-3 (no `pos_label`/`classes_` remapping — `predict` returns
`usize` `{0,1}` directly).

The thresholding kernel is shared and faithful: both fitted types' `predict`
compute `scores.mapv(|s| if s >= threshold { 1 } else { 0 })`, exactly sklearn's
`(y_score >= threshold)` with `pos_label=None`'s `[0,1]` label map (`:60`, `:66`)
— including the `>=` edge (REQ-1). `decision_function` on both returns the raw
`ScoreFn` output, matching sklearn's passthrough (`:201-217`).

`TunedThresholdClassifierCV::fit` is structurally a CV-argmax threshold search:
validate `cv >= 2` / shape / `n >= cv` (returning `FerroError::InvalidParameter`/
`ShapeMismatch`/`InsufficientSamples`), build folds via `KFold::new(cv)`
(unshuffled → deterministic, no RNG), accumulate a single pooled out-of-fold
score vector, then for each candidate threshold score the OOF predictions with the
`scoring` closure and keep the argmax (`:931` analog), refitting the base on the
whole set with the chosen threshold (`:878-890` analog). It diverges from sklearn
in the fold-combination arithmetic (single pooled OOF vector vs per-fold curves
combined by `_mean_interpolated_score`, `:591-616`) and in every default (REQ-6):
`scoring`/`cv` are required, the default grid is a fixed `[0,0.05,…,1.0]` rather
than `linspace(min_score, max_score, 100)`, and the splitter is plain KFold not
StratifiedKFold.

What is structurally absent vs sklearn: `threshold='auto'` resolution (REQ-2,
`:385-388`); `pos_label`/arbitrary `classes_` (REQ-3, `:62-66`, `:259-262`);
estimator-wrapping + `response_method`/`predict_proba` (REQ-4, `:105-217`); the
default scoring/grid/StratifiedKFold/refit/store_cv_results behaviour and the
per-fold mean-interpolated combination (REQ-6); and the `ferray` substrate
(REQ-7). The `score >= threshold` comparison (REQ-1) and the CV argmax-selection
structure (REQ-5) ARE shipped.

Invariants: `cv >= 2`; `y.len() == X.nrows()`; `n >= cv`; the candidate grid is
non-empty (defaults when empty); `best_threshold` is one of the candidate
thresholds; predictions have the same row count as `X`; the thresholding
comparison is `>=` (score equal to threshold → positive class).

## Verification

Commands establishing the SHIPPED claims (baseline
`9cadb9673ecddc6661f3967ea7983696902f8053`):

- `cargo test -p ferrolearn-model-sel --lib threshold` → the 5 `threshold::tests`
  pass (`fixed_threshold_basic`, `tuned_threshold_picks_best`,
  `tuned_threshold_default_grid`, `tuned_threshold_rejects_cv1`,
  `tuned_threshold_shape_mismatch`).
- REQ-1 oracle (FixedThresholdClassifier predict + `>=` edge, live oracle):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import FixedThresholdClassifier
  from sklearn.base import BaseEstimator, ClassifierMixin
  class ColZero(ClassifierMixin, BaseEstimator):
      def fit(self, X, y):
          self.classes_=np.array([0,1]); self.n_features_in_=X.shape[1]; return self
      def decision_function(self, X): return np.asarray(X)[:,0]
  X=np.array([[0.1,9.],[0.6,9.],[0.4,9.],[0.9,9.],[0.5,9.]]); y=np.array([0,1,0,1,1])
  clf=FixedThresholdClassifier(ColZero().fit(X,y), threshold=0.5,
      response_method='decision_function').fit(X,y)
  print(clf.predict(X).tolist())"
  # -> [0, 1, 0, 1, 1]   (the score-0.5 == threshold-0.5 row -> class 1)
  ```
  Pin an oracle-grounded `#[test]` building a column-0 `FitScoreFn`, scores
  `[0.1,0.6,0.4,0.9,0.5]`, `threshold=0.5`, expecting `[0,1,0,1,1]` (R-CHAR-3:
  from the live oracle, never copied from ferrolearn).
- REQ-2 oracle (the auto-threshold the gap omits, live oracle):
  `FixedThresholdClassifier(est)` with no explicit threshold defaults to
  `threshold="auto"` → 0.5 for `predict_proba`, 0.0 for `decision_function`
  (`sklearn/.../_classification_threshold.py:385-388`); ferrolearn requires an
  explicit `f64` — REQ-2 NOT-STARTED.
- REQ-4 oracle (the response_method / proba surface the gap omits, live oracle):
  `FixedThresholdClassifier(LogisticRegression(), response_method='predict_proba')`
  exposes `predict_proba`; ferrolearn has only `decision_function` + closure score
  — REQ-4 NOT-STARTED.
- REQ-5 oracle (TunedThresholdClassifierCV structural selection, live oracle):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import TunedThresholdClassifierCV
  from sklearn.base import BaseEstimator, ClassifierMixin
  class ColZero(ClassifierMixin, BaseEstimator):
      def fit(self, X, y):
          self.classes_=np.array([0,1]); self.n_features_in_=X.shape[1]; return self
      def predict_proba(self, X):
          s=np.asarray(X)[:,0]; return np.column_stack([1-s, s])
  X=np.array([[0.1,0.],[0.2,0.],[0.4,0.],[0.6,0.],[0.8,0.],[0.9,0.]])
  y=np.array([0,0,0,1,1,1])
  m=TunedThresholdClassifierCV(ColZero(), thresholds=[0.1,0.3,0.5,0.7,0.9],
      scoring='accuracy', cv=2, store_cv_results=True).fit(X,y)
  print(round(float(m.best_threshold_),3))"
  ```
  confirms sklearn ALSO selects the separating threshold by argmax over candidate
  thresholds; ferrolearn `tuned_threshold_picks_best` asserts `best_threshold()
  == 0.5` on the same setup. NOTE the fold-combination differs (REQ-5 caveat): the
  numeric `best_threshold_` can diverge when folds disagree because sklearn
  mean-interpolates per-fold curves while ferrolearn pools the OOF vector — that
  arithmetic divergence is REQ-6, pinnable as a failing `#[test]`.
- REQ-6 oracle (the default behaviour the gap omits, live oracle):
  `TunedThresholdClassifierCV(RandomForestClassifier(random_state=0)).fit(...)`
  (all defaults) finds `best_threshold_ ≈ 0.342` off `linspace(min_score,
  max_score, 100)` via balanced_accuracy on StratifiedKFold (docstring `:769`);
  ferrolearn requires explicit `scoring`/`cv`, defaults a fixed `[0,1]` step-0.05
  grid, and uses plain KFold — REQ-6 NOT-STARTED.

Commands that establish the NOT-STARTED REQs are absent: no `"auto"`/`Option`
threshold (REQ-2), no `pos_label`/`classes_` remap (REQ-3), no
`predict_proba`/`response_method` (REQ-4), no balanced_accuracy default / auto
score-range grid / StratifiedKFold / per-fold mean-interpolation (REQ-6), no
`ferray-core` usage (REQ-7). Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED.

SHIPPED: REQ-1 (FixedThresholdClassifier `>=` thresholding parity, including the
`s == threshold` edge), REQ-5 (TunedThresholdClassifierCV structural
argmax-over-CV threshold selection; deterministic unshuffled KFold; pooled-OOF vs
per-fold-mean-interpolation divergence noted → REQ-6), REQ-8 (boundary re-export
consumer; no dedicated non-test caller, no Python binding — honest underclaim).
NOT-STARTED (tracking #1736; the critic files per-REQ blockers): REQ-2
(`threshold='auto'`), REQ-3 (`pos_label`/`classes_`), REQ-4 (estimator-wrapping +
`response_method`/`predict_proba`; closure idiom is the sanctioned R-DEV-7
analog), REQ-6 (Tuned default scoring/grid/StratifiedKFold/refit/store_cv_results
+ mean-interpolated fold combination), REQ-7 (ferray substrate).
