# Logistic Regression

<!--
tier: 3-component
status: draft
baseline-commit: fd8e0244b6e37f4f2043953038bc6e16f06120c6
upstream-paths:
  - sklearn/linear_model/_logistic.py
  - sklearn/linear_model/_linear_loss.py
-->

## Summary

`ferrolearn-linear/src/logistic_regression.rs` mirrors scikit-learn's
`sklearn.linear_model.LogisticRegression` for its default solver only
(`solver='lbfgs'`, `penalty='l2'`). It fits a regularized logistic classifier
by L-BFGS: the binary branch minimizes the sigmoid log-loss and the multiclass
branch (`n_classes >= 3`) minimizes the multinomial softmax cross-entropy —
matching the lbfgs default that sklearn selects for `multi_class='auto'`. The
penalty, intercept handling, and `C` convention follow sklearn's canonical
objective `C * sum(pointwise_loss) + 0.5 * ||w||^2` (intercept unpenalized).

## Requirements

- REQ-1: Binary L-BFGS L2 fit producing `coef_`/`intercept_` that minimize
  `C * sum_i logloss(y_i, sigmoid(X w + b)) + 0.5 * ||w||^2`, with the intercept
  excluded from the penalty (sklearn objective, `_logistic.py:360`).
- REQ-2: Multiclass (`n_classes >= 3`) multinomial fit — softmax cross-entropy
  with L2 penalty, matching sklearn's lbfgs `multi_class='auto'` -> multinomial.
- REQ-3: `predict` returns argmax over class probabilities mapped back to the
  original (sorted-unique) class labels, not internal indices.
- REQ-4: `predict_proba` — sigmoid (binary, columns `[1-p, p]`) or softmax
  (multiclass), each row normalized to sum to 1.
- REQ-5: `decision_function` — raw `X @ coef + intercept` (binary) / per-class
  pre-softmax scores (multiclass).
- REQ-6: `fit_intercept` honored, including `false` (intercept fixed at 0).
- REQ-7: `C` inverse-regularization-strength parameter; smaller `C` => stronger
  penalty; `C <= 0` rejected.
- REQ-8: `HasCoefficients` / `HasClasses` introspection; `classes_` in sorted
  order, `coef_`/`weight_matrix` in sklearn `(n_classes, n_features)` orientation.
- REQ-9: `penalty` parameter — `'l1'`, `'elasticnet'`, `None`.
- REQ-10: `solver` parameter and non-lbfgs solvers (`liblinear`, `newton-cg`,
  `newton-cholesky`, `sag`, `saga`).
- REQ-11: `multi_class` parameter — explicit `'ovr'` one-vs-rest scheme.
- REQ-12: `class_weight` (`dict` / `'balanced'`).
- REQ-13: `dual` formulation.
- REQ-14: `intercept_scaling`.
- REQ-15: `l1_ratio` (elastic-net mixing).
- REQ-16: `warm_start`.
- REQ-17: `n_iter_` fitted attribute.
- REQ-18: `sample_weight` in `fit`.
- REQ-19: `random_state` / `n_jobs`.
- REQ-20: ferray substrate (array/linalg/optim on `ferray-*`, not `ndarray` +
  `crate::optim::lbfgs`).

## Acceptance criteria

- AC-1 (REQ-1): on `X=[[1,2],[2,3],[3,4],[5,6],[6,7],[7,8]]`, `y=[0,0,0,1,1,1]`,
  `C=1.0`, `max_iter=1000`, `tol=1e-4`, ferrolearn `coef`/`intercept` agree with
  `sklearn.linear_model.LogisticRegression(solver='lbfgs')` to within the shared
  L-BFGS tolerance (observed `~1e-3`: ferrolearn `[0.70450, 0.70452]`, intercept
  `-6.3406` vs sklearn `[0.70468, 0.70451]`, intercept `-6.3411`).
- AC-2 (REQ-1): the intercept parameter does not appear in the L2 penalty term
  of the objective.
- AC-3 (REQ-2): on three separable clusters (`n_classes=3`), `weight_matrix` has
  shape `(3, n_features)` and `predict` recovers the cluster labels.
- AC-4 (REQ-3): `classes()` equals the sorted unique labels; `predict` outputs
  values drawn from those labels (no index collapse, cf. #368).
- AC-5 (REQ-4): every row of `predict_proba` sums to 1 (epsilon `1e-10`); binary
  columns are `[1-sigmoid(z), sigmoid(z)]`.
- AC-6 (REQ-5): binary `decision_function` equals `X @ coef + intercept`.
- AC-7 (REQ-6): `with_fit_intercept(false)` yields `intercept == 0`.
- AC-8 (REQ-7): `C <= 0` returns `FerroError::InvalidParameter`.
- AC-9 (REQ-8): `n_classes()` and `classes()` reflect the fitted label set.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (binary L2 lbfgs fit) | SHIPPED | impl `fn fit_binary in logistic_regression.rs` builds the closure `objective` with `loss = sum_i logloss` (no `/n`; the dead `let _ = n_f`) and `loss + reg/2 * ||w||^2` where `reg = 1/C` — algebraically `1/C` times sklearn's canonical `C*sum(pointwise_loss) + 0.5*||w||^2` (`sklearn/linear_model/_logistic.py:360` `# C * sum(pointwise_loss) + penalty`), same minimizer. Intercept `b` is excluded from `reg_loss` (penalty on `w` only). Non-test consumer: `ferrolearn-python/src/classifiers.rs` (`RsLogisticRegression::fit` -> `ferrolearn_linear::LogisticRegression::<f64>::new().with_c(..).fit(..)`). Also `logistic_regression_cv.rs` (`fn fit` -> `LogisticRegression::<F>::new().with_c(c)`). Verification: `cargo run --example` harness coef `[0.70450, 0.70452]` int `-6.3406` vs sklearn `[0.70468, 0.70451]` int `-6.3411` (~1e-3, shared tol). |
| REQ-2 (multinomial softmax fit) | SHIPPED | impl `fn fit_multinomial in logistic_regression.rs` — one-hot targets, `softmax_2d(X W^T + b)`, `loss = -sum y_onehot * log p + reg/2 ||W||^2`, gradient `diff.t().dot(x) + reg*W`. Matches sklearn lbfgs `multi_class='auto'` -> multinomial for `n_classes >= 3` (`_logistic.py:959-965` docstring; selected at runtime). Non-test consumer: `logistic_regression_cv.rs` `fn fit` (calls `LogisticRegression::fit` which routes to `fit_multinomial` when `n_classes == 3` in `test_multiclass_cv`-exercised paths) and `RsLogisticRegression::fit` (multiclass via Python). Verification: `cargo test -p ferrolearn-linear --lib logistic` `test_multiclass_classification`, `test_multiclass_predict_proba` green; live oracle multi `coef_` shape `(3,2)`. |
| REQ-3 (predict argmax -> labels) | SHIPPED | impl `fn predict in logistic_regression.rs` (`Predict for FittedLogisticRegression`) argmaxes `predict_proba` columns then maps `predictions[i] = self.classes[best_class]`, restoring original labels from the sorted-unique `classes` vector. Non-test consumer: `RsLogisticRegression::predict` (`classifiers.rs`) returns these to Python; `FittedLogisticRegressionPipeline::predict_pipeline` re-floats them. Verification: `test_binary_classification`, `test_multiclass_classification` green; sklearn `classes_=[0,1,2]`. |
| REQ-4 (predict_proba normalized) | SHIPPED | impl `fn predict_proba in logistic_regression.rs` — binary path `probs[i,0]=1-sigmoid(z)`, `probs[i,1]=sigmoid(z)`; multiclass path `softmax_2d(X W^T + b)`. `softmax_2d` subtracts row-max then normalizes by the row sum. Non-test consumer: `RsLogisticRegression::predict_proba` (`classifiers.rs`); `FittedLogisticRegressionCV::predict_proba` (`logistic_regression_cv.rs`). Verification: `test_binary_predict_proba`, `test_multiclass_predict_proba` assert row-sums == 1 (eps 1e-10); oracle multi row0 sum `0.99999...`. |
| REQ-5 (decision_function) | SHIPPED | impl `fn decision_function in logistic_regression.rs` — binary returns `X @ coef + intercept`, multiclass returns the pre-softmax score matrix. Non-test consumer: `FittedLogisticRegressionCV::decision_function` (`logistic_regression_cv.rs`) delegates to the inner fitted model. KNOWN SHAPE DIVERGENCE: ferrolearn returns `(n_samples, 1)` for binary; sklearn returns `(n_samples,)` (oracle `decision_function(X).shape == (6,)`). Values are correct; the dropped trailing axis is the only gap (the impl doc-comment itself flags this as a deliberate 2-D uniformity choice — tracked as a presentation divergence, not pinned here). |
| REQ-6 (fit_intercept incl false) | SHIPPED | impl `LogisticRegression::with_fit_intercept` + both `fit_binary`/`fit_multinomial` branch on `self.fit_intercept`: when false, `b = 0`, the param vector omits the bias slot, and gradient skips `grad_b`. Non-test consumer: `RsLogisticRegression::new`/`fit` thread `fit_intercept` from the Python signature `(c, max_iter, tol, fit_intercept=true)`. Verification: `test_no_intercept` asserts `intercept() == 0` (eps 1e-10). |
| REQ-7 (C regularization) | SHIPPED | impl `LogisticRegression::with_c`; `fit_binary`/`fit_multinomial` set `reg = 1/C` and reject `C <= 0` (`FerroError::InvalidParameter { name: "C" }`), matching sklearn's `Interval(Real, 0, None, closed="right")` constraint (`_logistic.py:1108`). Non-test consumer: `RsLogisticRegression` (`with_c(self.c)`), `LogisticRegressionCV` (per-C sweep refits `with_c(c)`). Verification: `test_invalid_c` (C=0 and C=-1 both error). |
| REQ-8 (HasCoefficients/HasClasses) | SHIPPED | impl `HasCoefficients`/`HasClasses for FittedLogisticRegression in logistic_regression.rs`. `weight_matrix()` is `(n_classes, n_features)` (binary `(1, n_features)`) — same orientation as sklearn `coef_` (`_logistic.py:1012` `coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)`); NO transpose. `classes()` is the sorted-unique label vector, `n_classes()` its length. Non-test consumer: `RsLogisticRegression::classes_`/`coef_` getters (`classifiers.rs`) expose these to Python. Verification: `test_has_coefficients`, `test_has_classes` green. NOTE: the `HasCoefficients::coefficients()` introspection trait returns a flat `(n_features,)` vector (row 0 of `weight_matrix`), an internal-trait convention distinct from the 2-D sklearn `coef_`; the 2-D sklearn shape is `weight_matrix()`. |
| REQ-9 (penalty l1/elasticnet/none) | NOT-STARTED | open prereq blocker #442. Only L2 is implemented (`reg/2 * ||w||^2` hardcoded); no `penalty` parameter on `LogisticRegression`. |
| REQ-10 (solver variants + param) | NOT-STARTED | open prereq blocker #443. `LbfgsOptimizer` is hardcoded; no `solver` field; `liblinear`/`newton-cg`/`newton-cholesky`/`sag`/`saga` absent. |
| REQ-11 (multi_class='ovr'/param) | NOT-STARTED | open prereq blocker #444. Strategy is implicit (binary=sigmoid, `>=3`=multinomial); no `multi_class` parameter and no explicit OvR scheme. |
| REQ-12 (class_weight) | SHIPPED | impl `enum ClassWeight` (`Balanced`/`Dict`) + `LogisticRegression::with_class_weight` + `fn effective_sample_weights in logistic_regression.rs`, which computes the per-class multiplier (`'balanced'` = `n_samples/(n_classes*bincount)` per `sklearn/utils/class_weight.py:73`; dict per `:77-83`, missing classes -> 1.0) and folds it into the per-sample weight `sw_i * class_weight[y_i]` exactly as sklearn (`_logistic.py:312-313`). Non-test consumer: `ferrolearn-python/src/classifiers.rs` (`RsLogisticRegression::new`/`fit` thread `class_weight` from `_classifiers.py::LogisticRegression`). Verification: `divergence_logistic_weights.rs` `class_weight_balanced_matches_oracle`/`class_weight_dict_matches_oracle`/`class_weight_balanced_equals_equivalent_sample_weight`/`class_weight_dict_composes_with_sample_weight` + pytest `test_logreg_class_weight_*` — live oracle balanced coef diff < 1e-5, dict diff < 1e-5, balanced≡equivalent sample_weight < 1e-6. |
| REQ-13 (dual) | NOT-STARTED | open prereq blocker #446. No `dual` parameter (liblinear-only feature). |
| REQ-14 (intercept_scaling) | NOT-STARTED | open prereq blocker #447. No `intercept_scaling` parameter. |
| REQ-15 (l1_ratio) | NOT-STARTED | open prereq blocker #448. No elastic-net support, so no `l1_ratio`. |
| REQ-16 (warm_start) | NOT-STARTED | open prereq blocker #449. Each `fit` starts from `x0 = zeros`; no warm-start. |
| REQ-17 (n_iter_) | SHIPPED | impl `FittedLogisticRegression.n_iter` field (set from `LbfgsOptimizer::minimize_reporting` in both `fit_binary`/`fit_multinomial`) + getter `fn n_iter in logistic_regression.rs`. Mirrors sklearn's `n_iter_` (`_logistic.py:1375-1376`, scipy `OptimizeResult.nit` analog); shape `(1,)` on the binary/multinomial lbfgs path. R-DEV-7: ferrolearn's L-BFGS ≠ scipy's, so the honest count (positive int `<= max_iter`, deterministic) is shipped, NOT asserted `== sklearn`. Non-test consumer: `RsLogisticRegression::n_iter_` getter -> `_classifiers.py` sets `n_iter_ = np.array([n], dtype=int32)`. Verification: `divergence_logistic_weights.rs` `n_iter_contract` + pytest `test_logreg_n_iter_contract` (shape `(1,)`, dtype int32, `1 <= n <= max_iter`, deterministic). |
| REQ-18 (sample_weight) | SHIPPED | impl `fn fit_with_sample_weight in logistic_regression.rs` threads per-sample weights into the logloss + gradient of both branches (`loss -= w_i*logloss_i`, `grad += w_i*(p-y)·x`), mirroring sklearn (`_logistic.py:302-313`, `:451` `l2_reg_strength = 1/(C*sw_sum)` — ferrolearn omits the `/sw_sum` divisor on BOTH loss and penalty, so weighting only the loss terms preserves the same minimizer). `Fit::fit` delegates `None` (byte-identical to the prior unweighted fit). Negative weights rejected (`InvalidParameter`). Non-test consumer: `RsLogisticRegression::fit(x,y,sample_weight)` -> `_classifiers.py::LogisticRegression.fit(X,y,sample_weight)`. Verification: `divergence_logistic_weights.rs` `binary_sample_weight_matches_oracle`/`multiclass_sample_weight_matches_oracle`/`integer_sample_weight_equals_row_duplication`/`sample_weight_none_is_byte_identical_to_unweighted` + pytest `test_logreg_sample_weight_*` — live oracle weighted coef diff < 1e-6 (binary), predict_proba < 5e-3 (3-class), integer-weight≡row-dup. |
| REQ-19 (random_state/n_jobs) | SHIPPED | impl `LogisticRegression.random_state`/`n_jobs` fields + `with_random_state`/`with_n_jobs` in `logistic_regression.rs`. Documented no-ops on the deterministic lbfgs path (sklearn `_logistic.py:1112`/`:1121`; `random_state` only feeds sag/saga/liblinear shuffling, `n_jobs` is a threading knob) — R-DEV-7 (observable contract preserved: API surface + `get_params`/`clone` parity, result unchanged). Non-test consumer: `RsLogisticRegression` ctor stores both; `_classifiers.py::LogisticRegression` exposes them via `get_params`. Verification: `divergence_logistic_weights.rs` `random_state_n_jobs_are_noops_on_lbfgs` + pytest `test_logreg_random_state_n_jobs_noop_and_get_params` (coef unchanged; `get_params`/`clone` round-trip). |
| REQ-20 (ferray substrate) | NOT-STARTED | open prereq blocker #453. Module uses `ndarray::{Array1, Array2}` and `crate::optim::lbfgs::LbfgsOptimizer`, not the ferray array/linalg/optim substrate (R-SUBSTRATE-2). |

## Architecture

Two structs, mirroring the naming convention:

- `LogisticRegression<F>` (unfitted) — the constructor-parameter struct holding
  `c`, `max_iter`, `tol`, `fit_intercept`. `new()` defaults `C=1.0`, `tol=1e-4`,
  `fit_intercept=true` — matching sklearn `__init__` (`_logistic.py:1129`) except
  `max_iter` defaults to `1000` here vs sklearn's `100` (a benign convergence
  margin, not a contract divergence; both report the same fitted attrs at
  convergence). The struct exposes only the four lbfgs-relevant parameters;
  sklearn's remaining 9 constructor parameters are the subject of REQ-9..REQ-19.

- `FittedLogisticRegression<F>` (fitted) — stores `weight_matrix`
  `(n_classes, n_features)` (binary `(1, n_features)`) and `intercept_vec`
  `(n_classes,)` as the sklearn-shaped attributes, plus a flat `coefficients`
  `(n_features,)` / scalar `intercept` for the `HasCoefficients` trait, the
  sorted `classes` vector, and an `is_binary` flag selecting the sigmoid vs
  softmax prediction path.

`Fit::fit` validates shapes (`ShapeMismatch`), `C > 0`
(`InvalidParameter`), non-empty input, and `>= 2` distinct classes
(`InsufficientSamples`), then dispatches to `fit_binary` (`n_classes == 2`) or
`fit_multinomial` (`n_classes >= 3`).

**Objective and the `C` convention (REQ-1, REQ-7).** sklearn's canonical
logistic objective is `C * sum_i pointwise_loss_i + 0.5 * ||w||^2`
(`_logistic.py:356-362`), which sklearn implements as
`mean_i loss_i + 0.5 * l2_reg_strength * ||w||^2` with
`l2_reg_strength = 1/(C * sw_sum)` and `sw_sum = n_samples` for the unweighted
case (`_logistic.py:451`, `_linear_loss.py:172-175`). ferrolearn minimizes the
canonical form scaled by `1/C`: `sum_i loss_i + (1/(2C)) * ||w||^2`. Since
`sum_i loss_i + (1/(2C))||w||^2 = n * [mean_i loss_i + (1/(2 C n))||w||^2]`,
ferrolearn's objective is a positive-scalar (`n`, then `C`) multiple of
sklearn's, so the minimizer — and hence `coef_`/`intercept_` — is identical. The
`let _ = n_f` lines in both branches are deliberate dead code documenting the
#334 fix (a prior pass divided by `n`, making the effective regularization `n x`
too strong). The intercept enters `reg_loss` in neither branch, so the penalty
is on `w` only — matching sklearn (the intercept is the last param slot,
excluded from `l2_penalty`).

**Numerical stability.** `sigmoid` uses the sign-split form to avoid overflow;
`softmax_2d` subtracts the per-row max before exponentiating; the log-loss
clips probabilities to `[1e-15, 1 - 1e-15]`.

**Prediction surface.** `predict_proba` -> sigmoid/softmax; `predict` argmaxes
those probabilities and maps the winning column back through `classes`;
`decision_function` returns raw scores; `predict_log_proba` delegates to
`crate::log_proba`. `decision_function` keeps a 2-D `(n_samples, 1)` shape for
binary where sklearn returns `(n_samples,)` (REQ-5 note).

**Consumers.** The unfitted estimator and its fitted form are public boundary
API (R-DEFER-1 grandfathered). Real non-test consumers: the PyO3 binding
`RsLogisticRegression` (`ferrolearn-python/src/classifiers.rs`) registered in
`ferrolearn-python/src/lib.rs`, and `LogisticRegressionCV`
(`logistic_regression_cv.rs`) which refits `LogisticRegression` per candidate
`C` and at the selected best `C`. Pipeline integration is provided via
`PipelineEstimator` / `FittedPipelineEstimator`.

## Verification

Commands establishing the SHIPPED claims:

- `cargo test -p ferrolearn-linear --lib logistic` — 25 passed, 0 failed.
  Pins: `test_binary_classification`, `test_binary_predict_proba` (REQ-1/4),
  `test_multiclass_classification`, `test_multiclass_predict_proba` (REQ-2/4),
  `test_has_classes` (REQ-3/8), `test_no_intercept` (REQ-6), `test_invalid_c`
  (REQ-7), `test_has_coefficients` (REQ-8), `test_softmax_2d`, `test_sigmoid`.
- Live sklearn oracle (REQ-1, REQ-8 orientation):
  `python3 -c "import numpy as np; from sklearn.linear_model import LogisticRegression; X=np.array([[1.,2.],[2.,3.],[3.,4.],[5.,6.],[6.,7.],[7.,8.]]); y=np.array([0,0,0,1,1,1]); m=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,tol=1e-4).fit(X,y); print(m.coef_.shape, m.coef_, m.intercept_, m.decision_function(X).shape)"`
  -> `coef_` `(1,2)` `[[0.70468, 0.70451]]`, `intercept_` `[-6.3411]`,
  `decision_function` shape `(6,)`. ferrolearn (library, same input):
  `coef [0.70450, 0.70452]`, `intercept -6.3406` — agreement to the shared
  L-BFGS tolerance (`~1e-3`). The `(6,)` vs `(6,1)` shape gap is the REQ-5 note.
- Multiclass oracle (REQ-2):
  `LogisticRegression(C=10.0,solver='lbfgs',max_iter=2000).fit(Xm,ym)` over three
  separable clusters yields `coef_` shape `(3,2)`, `classes_=[0,1,2]`,
  `predict_proba` rows summing to 1 — matching `weight_matrix` `(3,2)` and the
  green `test_multiclass_*` cases.

Any of REQ-9..REQ-20 is NOT-STARTED: the corresponding sklearn behavior has no
implementing symbol in `logistic_regression.rs`, with blockers #442..#453 filed.
