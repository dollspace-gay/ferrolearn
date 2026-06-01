# Logistic Regression with Cross-Validated C Selection

<!--
tier: 3-component
status: draft
baseline-commit: 3b2bfcbc13a3ddee347a862e090d9525c5052715
upstream-paths:
  - sklearn/linear_model/_logistic.py
-->

## Summary
`LogisticRegressionCV<F>` mirrors scikit-learn's `LogisticRegressionCV`
(`sklearn/linear_model/_logistic.py:1464`): it selects the regularization
strength `C` by k-fold cross-validation over a grid of candidate `C` values,
scoring each fold's held-out split by accuracy, picking the `C` with the highest
mean (sum) accuracy across folds, then refitting `LogisticRegression` on the full
dataset at that `C` (`refit=True`). The Rust type owns only the L2 / lbfgs /
`refit=True` / integer-fold / accuracy-scoring slice of sklearn's surface; the
remaining constructor knobs (penalty variants, solver variants, `scoring`,
`l1_ratios`, the full set of fitted attributes, the sklearn `StratifiedKFold`
partition) are NOT-STARTED with filed blockers.

## Requirements
- REQ-1: Default `Cs` grid is 10 log-spaced values from `1e-4` to `1e4`,
  matching sklearn `np.logspace(-4, 4, 10)`.
- REQ-2: For each candidate `C`, run k-fold cross-validation, score each fold on
  its held-out split, average (sum) the scores across folds per `C`, select the
  `C` with the maximal mean score, then refit `LogisticRegression` on all data at
  that best `C` (`refit=True` semantics).
- REQ-3: `predict`, `predict_proba`, `predict_log_proba`, and
  `decision_function` delegate to the inner refitted `LogisticRegression`.
- REQ-4: `fit_intercept` (inherited from the inner `LogisticRegression`),
  `HasCoefficients` (`coef_`/`intercept_`), and `HasClasses`
  (`classes_`/`n_classes`) are exposed on the fitted estimator.
- REQ-5: The cross-validation partition matches sklearn's default
  `StratifiedKFold(5)` fold membership.
- REQ-6: The default scoring metric is accuracy, matching sklearn's
  `scoring=None` (which resolves to `LinearClassifierMixin.score` =
  `accuracy_score`).
- REQ-7: Expose the sklearn fitted attributes `C_`, `Cs_`, `scores_`,
  `coefs_paths_`, `n_iter_`.
- REQ-8: Accept `Cs` as an integer (count) that expands to
  `np.logspace(-4, 4, Cs)`.
- REQ-9: Support `refit=False` (average `C_` across folds rather than refit at a
  single best `C`).
- REQ-10: Support the `scoring` parameter (arbitrary scorer, not only accuracy).
- REQ-11: Accept `cv` as a splitter object, not only an integer fold count.
- REQ-12: Support `l1_ratios` (elasticnet path) and its best-index selection.
- REQ-13: Support `penalty` ∈ {`l1`, `elasticnet`, `None`} (inherited gap).
- REQ-14: Support `solver` variants (inherited gap).
- REQ-15: Support `multi_class='ovr'` explicitly (inherited gap).
- REQ-16: Support `class_weight` (inherited gap).
- REQ-17: Support `sample_weight` in `fit` (inherited gap).
- REQ-18: Support `random_state` / `n_jobs` (inherited gap).
- REQ-19: Run on the ferray substrate (inherited gap).

## Acceptance criteria
- AC-1 (REQ-1): the default `cs` field equals `np.logspace(-4, 4, 10).tolist()`
  to within `1e-12` (live oracle: max abs diff `3.4e-13`).
- AC-2 (REQ-2): `best_c()` is always a member of `cs_evaluated()`, and the
  selected `C` is the `argmax` over `cv_scores()`; `cv_scores()` has length
  `cs.len()`.
- AC-3 (REQ-3): `predict` agrees with `argmax(predict_proba)`; `predict_proba`
  rows sum to 1.
- AC-4 (REQ-4): `coefficients().len()` equals `n_features`; `classes()` is the
  sorted unique label set.
- AC-5 (REQ-6): the per-fold score equals `correct / count` (accuracy), matching
  sklearn `scoring=None`.
- AC-6 (REQ-5): on a fixed dataset, `best_c()` and `coef_` match sklearn's
  `LogisticRegressionCV.C_`/`coef_` (currently blocked — see REQ-5/REQ-2 notes).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (Cs grid) | SHIPPED | impl `pub fn new in logistic_regression_cv.rs` builds `cs.push(base.powf(F::from(-4.0 + i*8/9)))` for `i in 0..10`, mirroring sklearn `sklearn/linear_model/_logistic.py:276` (`Cs = np.logspace(-4, 4, Cs)` with default `Cs=10`). Non-test consumer: `pub fn fit in logistic_regression_cv.rs` iterates `self.cs` to drive CV. Verification: live oracle `np.logspace(-4,4,10)` vs the Rust formula → max abs diff `3.4e-13`. |
| REQ-2 (CV select + refit) | SHIPPED | impl `Fit for LogisticRegressionCV in logistic_regression_cv.rs`: per `c`, sums fold accuracy (`total_correct/total_count`), tracks `best_c` by strict `>` (`np.argmax` first-max), then `let inner = lr.fit(x, y)?` refits at `best_c`. Mirrors sklearn `_logistic.py:2082` (`best_index = scores.sum(axis=0).argmax()`) + the post-loop refit (`_logistic.py:2099`). Non-test consumer: `pub fn predict / predict_proba in logistic_regression_cv.rs` read `self.inner`. NOTE: exact `C_`/`coef_` parity is bounded by the LBFGS stopping gap (analog of #412, see `logistic_regression.rs` REQ-1) AND by the fold-partition divergence (REQ-5, #456); the selection *mechanism* is shipped. |
| REQ-3 (predict / proba) | SHIPPED | impl `Predict for FittedLogisticRegressionCV in logistic_regression_cv.rs` plus `pub fn predict_proba / predict_log_proba / decision_function in logistic_regression_cv.rs` all delegate to `self.inner` (`FittedLogisticRegression`), mirroring sklearn `LogisticRegressionCV` inheriting `LinearClassifierMixin.predict` / `predict_proba`. Non-test consumer: re-exported `pub use logistic_regression_cv::{...} in lib.rs`; the example in the module `//!` doc-comment calls `fitted.predict(&x)`. |
| REQ-4 (fit_intercept / coef / classes) | SHIPPED | impl `HasCoefficients for FittedLogisticRegressionCV` (`coefficients`/`intercept` → `self.inner`) and `HasClasses for FittedLogisticRegressionCV` (`classes`/`n_classes` → `self.inner`) in `logistic_regression_cv.rs`. The intercept term is fit by the inner `LogisticRegression` (sklearn default `fit_intercept=True`). Non-test consumer: `pub use {FittedLogisticRegressionCV, LogisticRegressionCV} in lib.rs` exposes these traits at the crate boundary. |
| REQ-5 (StratifiedKFold parity) | NOT-STARTED | open prereq blocker #456. `fn stratified_kfold_split in logistic_regression_cv.rs` assigns sample `i` of each class to test fold `i % k` within that class; sklearn `StratifiedKFold` (`cv=None`, `_logistic.py:1477`) partitions each class into `k` near-equal *contiguous* chunks (with an optional shuffle/`random_state`). Different fold membership changes per-`C` accuracy and can change the selected `C`. |
| REQ-6 (accuracy scoring) | SHIPPED | impl `Fit for LogisticRegressionCV in logistic_regression_cv.rs` scores each fold as `total_correct / total_count` (accuracy), mirroring sklearn `scoring=None` → `log_reg.score(...)` (`_logistic.py:800-801`) which is `LinearClassifierMixin.score` → `accuracy_score` (`sklearn/base.py:738-764`). Non-test consumer: the `cv_scores` accumulated here drive `best_c` selection in the same `fit`. Verification: per-fold score formula is accuracy by construction. |
| REQ-7 (C_/Cs_/scores_/coefs_paths_/n_iter_) | NOT-STARTED | open prereq blocker #457. The fitted struct exposes only `best_c()` (≈ `C_`), `cv_scores()`, `cs_evaluated()` (≈ `Cs_`); sklearn additionally exposes `scores_`/`coefs_paths_` dicts keyed by class, the full `C_` array, and `n_iter_` (`_logistic.py:2033-2058`). `n_iter_` is also unavailable on the inner `FittedLogisticRegression` (#450). |
| REQ-8 (Cs as int) | NOT-STARTED | open prereq blocker #458. `cs: Vec<F>` only; there is no integer-`Cs` constructor that expands to `np.logspace(-4, 4, Cs)` (`_logistic.py:275-276`). |
| REQ-9 (refit=False) | NOT-STARTED | open prereq blocker #459. `fit` always refits at a single `best_c`; sklearn `refit=False` averages `C_` across folds via `best_indices` (`_logistic.py:2124-2149`). |
| REQ-10 (scoring param) | NOT-STARTED | open prereq blocker #460. No `scoring` parameter; accuracy is hardcoded. sklearn accepts any scorer (`_logistic.py:1523-1528`, `789-805`). |
| REQ-11 (cv splitter object) | NOT-STARTED | open prereq blocker #461. `cv: usize` only; sklearn accepts any splitter or CV iterable, defaulting to `StratifiedKFold(5)` (`_logistic.py:1477`). |
| REQ-12 (l1_ratios / elasticnet) | NOT-STARTED | open prereq blocker #462. No `l1_ratios` grid; sklearn selects a best `l1_ratio` via `best_index // len(Cs_)` (`_logistic.py:2088-2090`). |
| REQ-13 (penalty l1/none) | NOT-STARTED | open prereq blocker #442 (inherited via the inner `LogisticRegression`). Only L2 is implemented. |
| REQ-14 (solver variants) | NOT-STARTED | open prereq blocker #443 (inherited). Only lbfgs is implemented. |
| REQ-15 (multi_class='ovr') | NOT-STARTED | open prereq blocker #444 (inherited). The inner estimator hardcodes binary=sigmoid / >2=multinomial with no `multi_class` param. |
| REQ-16 (class_weight) | NOT-STARTED | open prereq blocker #445 (inherited). |
| REQ-17 (sample_weight) | NOT-STARTED | open prereq blocker #451 (inherited). |
| REQ-18 (random_state / n_jobs) | NOT-STARTED | open prereq blocker #452 (inherited). |
| REQ-19 (ferray substrate) | NOT-STARTED | open prereq blocker #453 (inherited). The module computes on `ndarray::{Array1, Array2}` and the inner `crate::optim::lbfgs` (R-SUBSTRATE). |

## Architecture
The unfitted estimator is `pub struct LogisticRegressionCV<F>` in
`logistic_regression_cv.rs` with fields `cs: Vec<F>`, `cv: usize`,
`max_iter: usize`, `tol: F`. `pub fn new` constructs the default 10-point
log-spaced `cs` grid (REQ-1) and `cv = 5` (matching sklearn's default
`StratifiedKFold(5)`, `_logistic.py:1477`), with builder setters
`with_cs`/`with_cv`/`with_max_iter`/`with_tol`.

`Fit for LogisticRegressionCV` validates shape, non-empty `cs`, `cv >= 2`,
`n_samples >= cv`, and `>= 2` classes (returning `FerroError::ShapeMismatch`/
`InvalidParameter`/`InsufficientSamples`). It then, for each candidate `C`,
loops `fold in 0..cv` calling `fn stratified_kfold_split` (the `i % k`
within-class assignment, REQ-5/#456), extracts rows/labels via `fn select_rows`/
`fn select_elements`, fits an inner `LogisticRegression::new().with_c(c)...`,
predicts on the held-out split, and accumulates accuracy. A fold whose training
split degenerates to a single class (or whose inner fit/predict errors) is
treated as a failed `C` scored `0` — note this differs from sklearn, which would
still score the fold from the multinomial fit; it is a consequence of the
`i % k` partition (#456). The best `C` is chosen by strict `>` (first-max, like
`np.argmax`, `_logistic.py:2082`), then `LogisticRegression` is refit on all data
at `best_c` (REQ-2, `refit=True`).

The fitted type `pub struct FittedLogisticRegressionCV<F>` wraps
`inner: FittedLogisticRegression<F>`, `best_c`, `cv_scores`, `cs_evaluated`. All
prediction (`Predict`, `predict_proba`, `predict_log_proba`,
`decision_function`) and introspection (`HasCoefficients`, `HasClasses`)
delegate to `inner`. The full sklearn fitted-attribute surface
(`scores_`/`coefs_paths_`/`n_iter_`/the `C_` array) is not stored (REQ-7/#457).

**Invariant**: `best_c() ∈ cs_evaluated()` and `cv_scores().len() == cs.len()`.
**Parity boundary**: even with the selection mechanism correct, exact `C_`/
`coef_` agreement with sklearn is gated by (a) the LBFGS stopping bound the inner
refit inherits (analog of #412, documented in `logistic_regression.rs` REQ-1)
and (b) the fold-partition divergence (REQ-5/#456). Under-claim accordingly.

## Verification
- `cargo test -p ferrolearn-linear` — exercises `test_default_constructor`
  (REQ-1: `cs.len() == 10`, `cv == 5`), `test_binary_cv`/`test_best_c_in_cs`
  (REQ-2: `best_c ∈ cs`), `test_multiclass_cv`/`test_has_coefficients`
  (REQ-3/REQ-4), `test_stratified_kfold_split`/`test_stratified_kfold_uneven`
  (the current `i % k` partition behaviour, REQ-5 mechanism).
- Cs-grid oracle (REQ-1):
  `python3 -c "import numpy as np; print(np.logspace(-4,4,10).tolist())"`
  vs the Rust `10^(-4 + i*8/9)` formula → max abs diff `3.4e-13` (< AC-1's
  `1e-12`).
- Scoring oracle (REQ-6): sklearn `scoring=None` resolves to
  `accuracy_score` (`sklearn/base.py:764`); the Rust per-fold score is
  `correct / count`, accuracy by construction.
- AC-6 (REQ-5 end-to-end `C_`/`coef_` parity vs
  `LogisticRegressionCV(cv=5).fit(...)`) is **not** currently green and is the
  open work behind blocker #456 (compounded by #412); this is why REQ-5 is
  NOT-STARTED and REQ-2's parity claim is explicitly under-claimed to the
  *mechanism*.
