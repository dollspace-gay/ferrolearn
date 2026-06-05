# CalibratedClassifierCV (sigmoid / isotonic calibrators) & calibration_curve

<!--
tier: 3-component
status: draft
baseline-commit: ae6bdb4904c9d7ae62c045d298c172173df451c6
upstream-paths:
  - sklearn/calibration.py   # CalibratedClassifierCV (:66), _fit_calibrator (:632), _CalibratedClassifier (:680), _sigmoid_calibration (:770), _SigmoidCalibration (:872), calibration_curve (:937)
  - sklearn/isotonic.py      # IsotonicRegression(out_of_bounds="clip") — the isotonic calibrator
-->

## Summary

`ferrolearn-model-sel/src/calibration.rs` mirrors scikit-learn's probability
calibration from `sklearn/calibration.py`: `CalibratedClassifierCV` (`:66`) and
its two calibrators — Platt's sigmoid (`_sigmoid_calibration` `:770`,
`_SigmoidCalibration` `:872`) and isotonic regression
(`IsotonicRegression(out_of_bounds="clip")`, fitted via `_fit_calibrator` `:632`).
It wraps a base classifier — expressed as a `FitFn` CLOSURE rather than a wrapped
estimator object (a sanctioned R-DEV-7 Rust idiom, NOT a bug) — and calibrates the
raw decision scores it produces.

ferrolearn ships the **sigmoid (Platt) calibrator end-to-end**: its Newton's-method
fit reaches the SAME global minimum of the SAME convex objective as sklearn's
L-BFGS-B fit, so the calibrated PROBABILITIES match the live oracle to ~7e-9 on
well-conditioned data (verified below). The cross-validation fit/predict scaffold,
the OOF score collection, and the `Predict` impl ship for the BINARY case.

Everything else is NOT-STARTED with a filed blocker. The **isotonic** calibrator
uses block-MIDPOINT breakpoints and DIVERGES from sklearn's actual-X-threshold
interpolation by up to 0.18 on calibrated probabilities (deterministic — the
critic pins it, #1800). The **default behaviour diverges architecturally**:
sklearn's DEFAULT `ensemble=True` averages K per-fold calibrated classifiers
(ferrolearn does only the `ensemble=False` single-calibrator path, #1801); `cv=None`
uses StratifiedKFold for a classifier (ferrolearn uses plain `kfold_indices`,
#1802); `predict_proba` returns `(n_samples, n_classes)` via one-vs-rest with
normalization and `classes_` (ferrolearn is BINARY-only, returns a single
`Array1<f64>`, #1803). `calibration_curve` (`:937`) is ABSENT entirely (#1804).
`sample_weight` (#1805), the `max|F|>=30` rescaling (#1806), the `method='sigmoid'`/
`cv=None` defaults (#1807), and the ferray substrate (#1808) are all NOT-STARTED.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### `CalibratedClassifierCV` (`sklearn/calibration.py:66`)
- `:66` — `class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin,
  BaseEstimator)`.
- `:106-111` — docstring: `method : {'sigmoid', 'isotonic'}, default='sigmoid'`.
- `:149-166` — docstring: `ensemble : bool, default=True`. If `True`, the final
  estimator is an ensemble of `n_cv` (classifier, calibrator) pairs and the output
  is the AVERAGE predicted probabilities; if `False`, `cross_val_predict` gives
  unbiased OOF predictions for a SINGLE calibrator and the base is refit on all data.
- `:268-281` — `__init__(self, estimator=None, *, method="sigmoid", cv=None,
  n_jobs=None, ensemble=True)`.
- `:283-294` — `_get_estimator`: default base estimator is `LinearSVC(random_state=0)`.
- `:356-357` — `self.classes_ = LabelEncoder().fit(y).classes_`.
- `:390-401` — for an integer `cv`, raises `ValueError` if `np.any(np.unique(y,
  return_counts=True)[1] < n_folds)` — at least one class has fewer than `n_folds`
  examples.
- `:409` — `cv = check_cv(self.cv, y, classifier=True)` → 5-fold **StratifiedKFold**
  when `cv=None` for a classifier.
- `:411-426` — `ensemble=True` path: per `(train, test)` split,
  `_fit_classifier_calibrator_pair(clone(estimator), ...)` fits the base on `train`
  and a calibrator on the `test`-fold predictions; the K pairs accumulate in
  `self.calibrated_classifiers_`.
- `:427-465` — `ensemble=False` path: `cross_val_predict` yields OOF `predictions`,
  ONE calibrator is fit on all OOF (`_fit_calibrator`), and `this_estimator.fit(X,
  y)` refits the base on the FULL data — a SINGLE (estimator, calibrator) pair.
- `:474-500` — `predict_proba`: `mean_proba = mean over self.calibrated_classifiers_
  of calibrated_classifier.predict_proba(X)` — AVERAGES the K calibrated classifiers,
  shape `(n_samples, n_classes)`.

### `_fit_calibrator` / `_CalibratedClassifier` (`:632`, `:680`)
- `:664-674` — `Y = label_binarize(y, classes=classes)`; for each class column
  `this_pred`, fit `IsotonicRegression(out_of_bounds="clip")` (method `"isotonic"`)
  or `_SigmoidCalibration()` (method `"sigmoid"`) on `(this_pred, Y[:, class_idx])`.
- `:709-765` — `_CalibratedClassifier.predict_proba`: applies the per-class
  calibrators to the base predictions; for `n_classes == 2`, `proba[:, 0] = 1.0 -
  proba[:, 1]`; otherwise NORMALIZES rows to sum to 1 (`proba / sum(proba, axis=1)`,
  uniform on the all-zero edge); clips `(1, 1+1e-5] → 1`.

### `_sigmoid_calibration` / `_SigmoidCalibration` (`:770`, `:872`)
- `:798-815` — `F = predictions`; if `max(|F|) >= max_abs_prediction_threshold`
  (default 30), `F = F / scale_constant` with `scale_constant = max(|F|)` — rescaling
  for conditioning; A is unscaled at the end (`:868-869`).
- `:820-829` — priors `prior0 = #{y<=0}`, `prior1 = n - prior0`; targets
  `T[y>0] = (prior1 + 1) / (prior1 + 2)`, `T[y<=0] = 1 / (prior0 + 2)`.
- `:831-863` — minimises `HalfBinomialLoss` with `raw_prediction = -(A*F + B)` (so
  `P = expit(-(A*F+B)) = 1/(1+exp(A*F+B))`) via
  `scipy.optimize.minimize(method="L-BFGS-B", jac=True, gtol=1e-6, ftol=64*eps)`
  from `AB0 = [0, log((prior0+1)/(prior1+1))]`. The objective is CONVEX → the global
  minimum (hence the calibrated probabilities) is unique regardless of init/optimizer.
- `:869` — returns `(A / scale_constant, B)`.
- `:910-924` — `_SigmoidCalibration.predict(T) = expit(-(a_*T + b_))`.

### `calibration_curve` (`:937`)
- `:937-944` — `calibration_curve(y_true, y_prob, *, pos_label=None, n_bins=5,
  strategy="uniform")`.
- `:1024-1043` — bins `y_prob` into `n_bins` (uniform `linspace(0,1,n_bins+1)` or
  quantile `percentile`), then `prob_true = bincount(binids, weights=y_true) /
  bincount(binids)` and `prob_pred = bincount(binids, weights=y_prob) /
  bincount(binids)` over the NON-EMPTY bins. Returns `(prob_true, prob_pred)`.

### `IsotonicRegression(out_of_bounds="clip")` (`sklearn/isotonic.py`)
- PAV on `(X, y)`; builds `f_ = interp1d(X_thresholds_, y_thresholds_,
  kind="linear", bounds_error=False)` where `X_thresholds_` are the UNIQUE sorted X
  values (ties averaged via `_make_unique`) and `y_thresholds_` the pooled PAV means;
  out-of-bounds clips to the endpoint y. Verified live: for the fixture below,
  `X_thresholds_ = [0,1,1.5,3,3.5,5]`, `y_thresholds_ = [0,0,0.5,0.5,1,1]` — the
  breakpoints are ACTUAL X values, NOT block midpoints.

## Requirements

Each REQ carries the goal.md mental test (MATCH = numerical/API contract → match
sklearn; DEVIATE = Rust analog / footgun).

- REQ-SIGMOID (Platt scaling calibrated probabilities — DETERMINISTIC): `fit_sigmoid`
  fits `P = sigmoid(a*f + b)` by Newton's method with Platt targets `t_pos =
  (n_pos+1)/(n_pos+2)`, `t_neg = 1/(n_neg+2)`; sklearn `_sigmoid_calibration` (`:770`)
  fits `P = expit(-(A*f+B))` with the SAME targets (`:827-829`) by L-BFGS-B. The
  objective is CONVEX, so both reach the SAME unique minimum; ferrolearn's
  `(a,b) = (-A, -B)` is a sign reparam yielding IDENTICAL probabilities. **MATCH**
  (R-DEV-1 — calibrated probabilities are the numerical contract). Verified against
  the live oracle: max abs prob diff 6.7e-9 on `|F|<30` data. SHIPPED.
- REQ-ISOTONIC (isotonic calibrated probabilities — DETERMINISTIC, DIVERGENT):
  `fit_isotonic` runs PAV and uses each block's MIDPOINT `(lo+hi)/2` as the
  breakpoint score with the pooled mean as the probability, then `isotonic_lookup`
  clamps + linearly interpolates between midpoints. sklearn
  `IsotonicRegression(out_of_bounds="clip")` interpolates over the ACTUAL unique
  sorted X-thresholds (`:670`). **MATCH-intent / DEVIATE-actual** (R-DEV-1 — the
  calibrated probability IS the contract; ferrolearn computes a different one).
  Verified DIVERGENT: max abs prob diff 0.18 at query points. NOT-STARTED (#1800).
- REQ-ENSEMBLE (ensemble=True default — K averaged calibrators): sklearn's DEFAULT
  `ensemble=True` (`:275`, `:411-426`) fits K (classifier, calibrator) pairs and
  AVERAGES their `predict_proba` (`:474-500`); ferrolearn `fit` does only the
  `ensemble=False` path — ONE calibrator on aggregated OOF + base refit on full data.
  **MATCH-intent / gap** (R-DEV-1 — the DEFAULT observable behaviour differs;
  K-averaging is unimplemented). NOT-STARTED (#1801).
- REQ-STRATIFIED-CV (cv=None ⇒ StratifiedKFold for a classifier): sklearn
  `check_cv(self.cv, y, classifier=True)` (`:409`) gives StratifiedKFold; ferrolearn
  `kfold_indices(n, cv)` is NON-stratified, NON-shuffled, consecutive folds. This
  changes OOF fold membership. **MATCH-intent / gap** (R-DEV-1). NOT-STARTED (#1802).
- REQ-MULTICLASS (predict_proba `(n_samples, n_classes)` + one-vs-rest +
  normalization + `classes_`): sklearn's `_CalibratedClassifier.predict_proba`
  (`:709-765`) fits per-class calibrators on `label_binarize`d columns and normalizes
  rows to sum to 1; `classes_` (`:357`). ferrolearn is BINARY-only: `predict` returns
  a single `Array1<f64>` of positive-class probability, no `classes_`, no
  normalization, no negative-class column. **MATCH-intent / gap** (R-DEV-3 — output
  object contract). NOT-STARTED (#1803).
- REQ-CALIBRATION-CURVE (`calibration_curve` function): sklearn `:937` bins `y_prob`
  and returns `(prob_true, prob_pred)` per non-empty bin. ABSENT in ferrolearn.
  **MATCH-intent / gap** (a follow-up acto-builder ships it; do NOT build it in this
  doc-author pass). NOT-STARTED (#1804).
- REQ-SAMPLE-WEIGHT (sample_weight): sklearn `_sigmoid_calibration` weights the
  priors/targets (`:821-823`) and `IsotonicRegression.fit` accepts `sample_weight`;
  ferrolearn `fit_sigmoid`/`fit_isotonic` have no weight channel. **MATCH-intent /
  gap** (R-DEV-1). NOT-STARTED (#1805).
- REQ-SIGMOID-RESCALE (max|F|>=30 rescaling): sklearn rescales `F` by `max(|F|)`
  when `max(|F|) >= 30` for L-BFGS-B conditioning, then unscales A (`:811-815`,
  `:868-869`); ferrolearn has no rescaling. For `|F|<30` results coincide (REQ-SIGMOID
  verifies this); for very large scores the optimiser conditioning differs (the convex
  minimum is still the same in exact arithmetic). **MATCH-intent / gap** (R-DEV-1 —
  numerical robustness). NOT-STARTED (#1806).
- REQ-METHOD-DEFAULT (default method / cv): sklearn defaults `method="sigmoid"`
  (`:272`) and `cv=None` → 5-fold (`:273`); ferrolearn `new(fit_fn, method, cv)`
  requires an explicit `CalibrationMethod` and `cv: usize`. **MATCH-intent / gap**
  (R-DEV-2 — constructor defaults). NOT-STARTED (#1807).
- REQ-X-1 (R-SUBSTRATE): production code imports `ndarray::{Array1, Array2}`;
  destination substrate is `ferray-core` (R-SUBSTRATE-1). NOT-STARTED (#1808).
- REQ-X-2 (non-test production consumer): the boundary estimator types
  `CalibratedClassifierCV`/`CalibrationMethod`/`FittedCalibratedClassifierCV` are the
  public API (S5 / R-DEFER-1) and are re-exported from `lib.rs`. SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side). The oracle is the
installed sklearn 1.5.2 package (run from a directory OUTSIDE the source clone — the
clone at `/home/doll/scikit-learn` is unbuilt; `cd /tmp` first).

- AC-SIGMOID (REQ-SIGMOID — SHIPPED, the calibrated probabilities match): isolate the
  calibrator from the base estimator (construct scores/labels directly) and compare
  `fit_sigmoid` probabilities to sklearn's `expit(-(A*f+B))`:
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.calibration import _sigmoid_calibration
  from scipy.special import expit
  scores=np.array([-2.,-1.,0.5,1.,2.,3.,-1.5,0.2,2.5,-0.7])
  labels=np.array([0,0,0,1,1,1,0,1,1,0])
  A,B=_sigmoid_calibration(scores,labels)
  print(np.round(expit(-(A*scores+B)),6).tolist())"
  # -> [0.098873, 0.218482, 0.532053, 0.644747, 0.822198, 0.921766, 0.149037, 0.462024, 0.880687, 0.270133]
  ```
  ferrolearn's `sigmoid_fn(a*f + b)` over `(a,b)` from `fit_sigmoid(scores, labels)`
  reproduces these to MAX ABS DIFF 6.66e-9 (well within ~1e-4). VERIFIED MATCH →
  SHIPPED. The critic may pin this exact vector as an oracle-grounded `#[test]`.
- AC-ISOTONIC (REQ-ISOTONIC — DIVERGENCE, the pin): the same isolated-calibrator
  comparison against `IsotonicRegression(out_of_bounds="clip")`:
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.isotonic import IsotonicRegression
  s=np.array([0.,1.,2.,3.,4.,5.,2.5,1.5,3.5]); y=np.array([0,0,1,0,1,1,0,1,1],dtype=float)
  ir=IsotonicRegression(out_of_bounds='clip').fit(s,y)
  q=np.array([-1.,0.5,1.2,2.2,2.7,3.3,4.4,6.])
  print(np.round(ir.predict(q),6).tolist())"
  # sklearn -> [0.0, 0.0, 0.2, 0.5, 0.5, 0.8, 1.0, 1.0]
  ```
  ferrolearn `isotonic_lookup(fit_isotonic(s,y), q)` → `[0.0, 0.0, 0.08, 0.48, 0.68,
  0.92, 1.0, 1.0]` — MAX ABS DIFF 0.18. DIVERGENCE. NOT-STARTED (#1800). The critic
  pins a failing `#[test]` asserting the sklearn probabilities at these query points.
- AC-ENSEMBLE (REQ-ENSEMBLE — DIVERGENCE): the DEFAULT produces K calibrated
  classifiers:
  ```
  cd /tmp && python3 -c "
  from sklearn.calibration import CalibratedClassifierCV
  from sklearn.naive_bayes import GaussianNB
  from sklearn.datasets import make_classification
  X,y=make_classification(n_samples=200,n_features=4,n_redundant=0,random_state=42)
  m=CalibratedClassifierCV(GaussianNB(),cv=3).fit(X,y)
  print(len(m.calibrated_classifiers_))"   # -> 3 (ensemble=True default)
  ```
  ferrolearn `fit` produces a SINGLE (predict_fn, calibrator); no K-averaging. Gap.
  NOT-STARTED (#1801).
- AC-MULTICLASS (REQ-MULTICLASS — DIVERGENCE): `predict_proba` shape + normalization:
  ```
  cd /tmp && python3 -c "
  from sklearn.calibration import CalibratedClassifierCV
  from sklearn.naive_bayes import GaussianNB
  from sklearn.datasets import make_classification
  X,y=make_classification(n_samples=300,n_features=6,n_redundant=0,n_classes=3,n_informative=4,random_state=0)
  m=CalibratedClassifierCV(GaussianNB(),cv=3).fit(X,y)
  p=m.predict_proba(X[:2]); print(p.shape, p.sum(axis=1).tolist())"   # -> (2, 3) [1.0, 1.0]
  ```
  ferrolearn returns a single `Array1<f64>` (positive-class prob), no `(n,n_classes)`
  matrix, no `classes_`. Gap. NOT-STARTED (#1803).
- AC-CALIBRATION-CURVE (REQ-CALIBRATION-CURVE — ABSENT): the function exists in
  sklearn:
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.calibration import calibration_curve
  yt=np.array([0,0,0,0,1,1,1,1,1]); yp=np.array([0.1,0.2,0.3,0.4,0.65,0.7,0.8,0.9,1.])
  pt,pp=calibration_curve(yt,yp,n_bins=3); print(pt.tolist(), pp.tolist())"
  # -> [0.0, 0.5, 1.0] [0.2, 0.525, 0.85]
  ```
  ferrolearn has no `calibration_curve`. Gap. NOT-STARTED (#1804).
- AC-X-2 (REQ-X-2 — SHIPPED): `grep -n "pub use calibration" lib.rs` shows the
  re-export of `CalibratedClassifierCV, CalibrationMethod, FittedCalibratedClassifierCV`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-SIGMOID (Platt calibrated probabilities, DETERMINISTIC) | SHIPPED | impl `fn fit_sigmoid in calibration.rs` computes Platt targets `t_pos = (n_pos+1)/(n_pos+2)`, `t_neg = 1/(n_neg+2)` and minimises the NLL of `P = sigmoid_fn(a*f + b)` by Newton's method (init `a=b=0`, `max_iter=100`, `tol=1e-8`, 2x2 Hessian solve). This is sklearn's `_sigmoid_calibration` (`sklearn/calibration.py:770`) with the SAME targets (`:827-829`) and the SAME convex `HalfBinomialLoss` objective on `raw_prediction = -(A*f+B)` → `P = expit(-(A*f+B))` (`:838`); ferrolearn `(a,b) = (-A,-B)` is a sign reparam (`:924` `expit(-(a_*T+b_))`). Because the objective is CONVEX the global minimum — hence the calibrated PROBABILITIES — is unique regardless of optimizer (Newton vs L-BFGS-B). LIVE ORACLE (AC-SIGMOID): scores `[-2,-1,.5,1,2,3,-1.5,.2,2.5,-.7]`, labels `[0,0,0,1,1,1,0,1,1,0]` → sklearn `expit(-(A*f+B))` `[0.098873,0.218482,0.532053,0.644747,0.822198,0.921766,0.149037,0.462024,0.880687,0.270133]`; ferrolearn `sigmoid_fn(a*f+b)` reproduces these to MAX ABS DIFF **6.66e-9**. `Calibrator::Sigmoid` is applied at predict via `fn transform in calibration.rs` (`scores.mapv(|f| sigmoid_fn(a*f + b))`). Tests: `test_fit_sigmoid_basic`, `test_sigmoid_fn_properties`, `test_fit_sigmoid_empty`. Honest underclaim: only `\|F\|<30` is verified — the large-score rescaling is REQ-SIGMOID-RESCALE (#1806); and the path is wired only into the `ensemble=False` binary fit. Non-test consumer: REQ-X-2 (boundary type + `lib.rs` re-export). |
| REQ-ISOTONIC (isotonic calibrated probabilities, DETERMINISTIC) | NOT-STARTED | open prereq blocker #1800. DIVERGENCE. impl `fn fit_isotonic in calibration.rs` runs PAV (merge while `prev_mean > next_mean`) and emits, per block, a breakpoint at the block MIDPOINT `f64::midpoint(lo, hi)` with the pooled mean as probability; `fn isotonic_lookup in calibration.rs` clamps to endpoints and linearly interpolates between MIDPOINTS. sklearn fits `IsotonicRegression(out_of_bounds="clip")` (`sklearn/calibration.py:670`) which interpolates over the ACTUAL unique sorted X-thresholds with pooled PAV means (`X_thresholds_`/`y_thresholds_`), clipping out-of-bounds. LIVE ORACLE (AC-ISOTONIC): `s=[0,1,2,3,4,5,2.5,1.5,3.5]`, `y=[0,0,1,0,1,1,0,1,1]`, query `[-1,.5,1.2,2.2,2.7,3.3,4.4,6]` → sklearn `[0,0,0.2,0.5,0.5,0.8,1,1]` (`X_thresholds_=[0,1,1.5,3,3.5,5]`) vs ferrolearn `[0,0,0.08,0.48,0.68,0.92,1,1]` (midpoints `[0,1,2.25,3.5,4,5]`) — MAX ABS DIFF **0.18**. DETERMINISTIC, FIXABLE: the critic pins a failing `#[test]` on the calibrated probabilities at the query points. Existing tests (`test_fit_isotonic_basic`, `test_isotonic_lookup_*`) only check monotonicity/endpoints/internal interpolation, not parity with sklearn's breakpoints. |
| REQ-ENSEMBLE (ensemble=True default — K averaged calibrators) | NOT-STARTED | open prereq blocker #1801. ARCHITECTURAL. impl `pub fn fit in calibration.rs` collects OOF `(score, label)` pairs over `kfold_indices`, fits ONE calibrator (`fit_sigmoid`/`fit_isotonic`) on the aggregated OOF, refits the base on FULL data, and returns a SINGLE `FittedCalibratedClassifierCV { predict_fn, calibrator }` — exactly sklearn's `ensemble=False` path (`sklearn/calibration.py:427-465`). sklearn's DEFAULT is `ensemble=True` (`:275`), which fits K per-fold (classifier, calibrator) pairs (`:411-426`) and AVERAGES their `predict_proba` (`:474-500`). ferrolearn has neither the `ensemble` parameter nor the K-averaging. LIVE ORACLE (AC-ENSEMBLE): `CalibratedClassifierCV(GaussianNB(), cv=3)` → `len(calibrated_classifiers_) == 3`; ferrolearn produces one. The DEFAULT observable behaviour differs end-to-end. |
| REQ-STRATIFIED-CV (cv=None ⇒ StratifiedKFold for a classifier) | NOT-STARTED | open prereq blocker #1802. DIVERGENCE. impl `pub fn fit in calibration.rs` calls `fn kfold_indices in calibration.rs` (NON-stratified, NON-shuffled: `base = n/k`, first `n%k` folds get `+1`, consecutive slices). sklearn `cv = check_cv(self.cv, y, classifier=True)` (`sklearn/calibration.py:409`) uses 5-fold **StratifiedKFold** for a classifier when `cv=None`. Non-stratified folds change OOF fold membership, hence the OOF score/label pairs the single calibrator is fit on. ferrolearn additionally requires an explicit `cv: usize` (no `cv=None` default — REQ-METHOD-DEFAULT). The `< n_folds`-per-class fold-feasibility guard (`:390-401`) is also absent (ferrolearn guards only `n_samples < cv`). |
| REQ-MULTICLASS (predict_proba (n_samples, n_classes) + OvR + normalization + classes_) | NOT-STARTED | open prereq blocker #1803. ARCHITECTURAL. `impl Predict for FittedCalibratedClassifierCV` in `calibration.rs` returns `Array1<f64>` — a SINGLE column of positive-class probability via `self.calibrator.transform(&raw_scores)`. There is no `classes_`, no per-class calibrator vector, no `(n_samples, n_classes)` matrix, and no normalization. sklearn `_fit_calibrator` (`sklearn/calibration.py:664-674`) fits one calibrator per `label_binarize`d class column and `_CalibratedClassifier.predict_proba` (`:709-765`) returns `(n_samples, n_classes)` with `proba[:,0] = 1 - proba[:,1]` for binary and row-normalization for multiclass; `classes_` is set at `:357`. LIVE ORACLE (AC-MULTICLASS): 3-class `predict_proba` shape `(2, 3)`, rows sum to 1.0; ferrolearn cannot represent multiclass at all. Absent end-to-end. |
| REQ-CALIBRATION-CURVE (`calibration_curve` function) | NOT-STARTED | open prereq blocker #1804. MISSING FUNCTION. `calibration.rs` defines no `calibration_curve`; `grep -n "calibration_curve" ferrolearn-model-sel/src/` finds only the route's `parity_ops`, no Rust symbol. sklearn `calibration_curve(y_true, y_prob, *, pos_label=None, n_bins=5, strategy="uniform")` (`sklearn/calibration.py:937`) bins `y_prob` (uniform/quantile) and returns `(prob_true, prob_pred)` = per-non-empty-bin mean of `y_true` and mean of `y_prob` (`:1024-1043`). LIVE ORACLE (AC-CALIBRATION-CURVE): `calibration_curve([0,0,0,0,1,1,1,1,1], [.1,.2,.3,.4,.65,.7,.8,.9,1.], n_bins=3)` → `([0,0.5,1], [0.2,0.525,0.85])`. A follow-up acto-builder ships this (NOT this doc-author pass). |
| REQ-SAMPLE-WEIGHT (sample_weight channel) | NOT-STARTED | open prereq blocker #1805. impl `fn fit_sigmoid`/`fn fit_isotonic in calibration.rs` take only `(scores, labels)` — no weight argument. sklearn `_sigmoid_calibration(predictions, y, sample_weight=None, ...)` weights the priors `prior0/prior1` (`sklearn/calibration.py:821-823`) and `HalfBinomialLoss` (`:842`), and `IsotonicRegression.fit(X, y, sample_weight)` weights PAV; `_fit_calibrator` threads `sample_weight` through (`:673`). Absent end-to-end. |
| REQ-SIGMOID-RESCALE (max\|F\|>=30 rescaling) | NOT-STARTED | open prereq blocker #1806. impl `fn fit_sigmoid in calibration.rs` runs Newton on the raw `scores` with no rescaling. sklearn rescales `F = F / max(\|F\|)` when `max(\|F\|) >= max_abs_prediction_threshold=30` for L-BFGS-B conditioning, then unscales `A = A/scale_constant` (`sklearn/calibration.py:811-815`, `:868-869`). For `\|F\|<30` the two coincide (REQ-SIGMOID verifies max diff 6.66e-9); for very large scores the optimiser conditioning differs (exact-arithmetic minimum is unchanged, but the numerical paths can diverge). Could be folded into REQ-SIGMOID; pinned separately as the edge gap. |
| REQ-METHOD-DEFAULT (default method='sigmoid' / cv=None) | NOT-STARTED | open prereq blocker #1807. API-shape. `pub fn new in calibration.rs` is `new(fit_fn, method: CalibrationMethod, cv: usize)` — both `method` and `cv` are MANDATORY with no defaults. sklearn `__init__(estimator=None, *, method="sigmoid", cv=None, n_jobs=None, ensemble=True)` (`sklearn/calibration.py:268-281`) defaults `method="sigmoid"`, `cv=None` (→5-fold), `ensemble=True`. The no-method / no-cv constructor forms are unrepresentable. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker #1808. Production code in `calibration.rs` imports `use ndarray::{Array1, Array2}` and operates on `Array2<f64>`/`Array1<f64>`/`Array1<usize>` (e.g. `fn select_rows` builds `Array2::from_shape_vec`). Per R-SUBSTRATE-1 the destination is `ferray-core`; `ndarray` is the wrong substrate. Not migrated. |
| REQ-X-2 (non-test production consumer) | SHIPPED | Crate re-export: `lib.rs` `pub use calibration::{CalibratedClassifierCV, CalibrationMethod, FittedCalibratedClassifierCV};` (and `pub mod calibration;`). Per S5 / R-DEFER-1 the boundary meta-estimator types ARE the public API and are grandfathered (existing pub surface). CAVEAT (honest underclaim): `grep -rn "CalibratedClassifierCV" ferrolearn-*/src/ \| grep -v 'tests\|calibration.rs'` finds ONLY the `lib.rs` re-export — there is NO dedicated non-test internal caller and NO `ferrolearn-python` binding. SHIPPED on the boundary re-export per S5, not a dedicated production caller; the missing Python binding is noted. The base estimator is supplied as a `FitFn` CLOSURE (an R-DEV-7 Rust idiom for sklearn's wrapped-`estimator` pattern) — noted, not pinned. |

## Architecture

ferrolearn splits the estimator into an unfitted/Fitted pair (CLAUDE.md naming):
`CalibratedClassifierCV { fit_fn: FitFn, method: CalibrationMethod, cv: usize }`
fits to `FittedCalibratedClassifierCV { predict_fn: PredictFn, calibrator:
Calibrator }`. sklearn keeps a single `CalibratedClassifierCV` whose post-`fit`
state is `calibrated_classifiers_` (a LIST — length K for the default ensemble) and
`classes_`.

**The base-estimator representation is the central R-DEV-7 deviation.** sklearn
wraps an `estimator` object and resolves its `decision_function`/`predict_proba`
(`:336-340`, `:616-620`); ferrolearn instead takes a `FitFn = Box<dyn Fn(&Array2,
&Array1<usize>) -> Result<PredictFn>>` closure that trains the base and returns a
`PredictFn = Box<dyn Fn(&Array2) -> Result<Array1<f64>>>` producing RAW decision
scores. The closure is the sanctioned Rust analog of the wrapped-estimator pattern
(R-DEV-7, the same idiom used in `threshold.rs`/self-training) — noted as a
deliberate divergence, NOT a bug, and NOT pinned.

**The fit scaffold** (`pub fn fit`) validates `cv >= 2`
(`FerroError::InvalidParameter`), `n_samples >= cv`
(`FerroError::InsufficientSamples`), and `y.len() == n_samples`
(`FerroError::ShapeMismatch`); builds folds via `kfold_indices(n, cv)`
(non-stratified — REQ-STRATIFIED-CV, #1802); per fold fits the base on the train rows
(`select_rows`/`select_elements`) and collects OOF `(score, label)` pairs in fold
order; fits ONE calibrator on the aggregated OOF; and refits the base on the FULL
data. This is precisely sklearn's `ensemble=False` path (`:427-465`) — but sklearn's
DEFAULT is `ensemble=True` with K-averaging (REQ-ENSEMBLE, #1801), so the default
observable behaviour diverges architecturally.

**The calibrators.** `Calibrator::Sigmoid { a, b }` (REQ-SIGMOID) is the SHIPPED
piece: `fit_sigmoid`'s Newton minimum equals sklearn's L-BFGS-B minimum of the same
convex objective, so the calibrated probabilities match to ~7e-9 (oracle-verified).
`Calibrator::Isotonic { mapping }` (REQ-ISOTONIC, #1800) DIVERGES: ferrolearn's PAV
emits block-MIDPOINT breakpoints, whereas sklearn's `IsotonicRegression` interpolates
over the actual unique sorted X-thresholds — up to 0.18 difference at query points
(oracle-verified, deterministic → the critic pins it).

**The predict path** (`impl Predict for FittedCalibratedClassifierCV`) runs the base
`predict_fn` to raw scores then `calibrator.transform` to a SINGLE `Array1<f64>` of
positive-class probability. This is BINARY-only: no `classes_`, no `(n_samples,
n_classes)` matrix, no one-vs-rest normalization (REQ-MULTICLASS, #1803), unlike
sklearn's `predict_proba` (`:474-500`, `:709-765`).

What is structurally absent vs sklearn: the `ensemble=True` K-averaging default
(REQ-ENSEMBLE), StratifiedKFold (REQ-STRATIFIED-CV), multiclass `predict_proba` +
`classes_` (REQ-MULTICLASS), `calibration_curve` (REQ-CALIBRATION-CURVE),
`sample_weight` (REQ-SAMPLE-WEIGHT), the `max|F|>=30` rescaling (REQ-SIGMOID-RESCALE),
the `method`/`cv` constructor defaults (REQ-METHOD-DEFAULT), and the `ferray`
substrate (REQ-X-1). The sigmoid calibrated probabilities (REQ-SIGMOID) and the
boundary re-export (REQ-X-2) ARE shipped.

Invariants: `cv >= 2`; `y.len() == X.nrows()`; `n_samples >= cv`; calibrated
probabilities lie in `[0, 1]`; the OOF folds form a disjoint covering partition of
`0..n_samples` (consecutive, non-shuffled).

## Verification

Commands establishing the SHIPPED claims (baseline
`ae6bdb4904c9d7ae62c045d298c172173df451c6`). The oracle is the installed sklearn
1.5.2 (`cd /tmp` — the source clone at `/home/doll/scikit-learn` is unbuilt):

- `cargo test -p ferrolearn-model-sel --lib calibration` → **20 passed, 0 failed**
  (`calibration::tests::{test_fit_sigmoid_*, test_sigmoid_fn_properties,
  test_fit_isotonic_*, test_isotonic_lookup_*, test_calibrated_classifier_*,
  test_calibration_method_eq, test_kfold_indices_*, test_select_*}`).
- REQ-SIGMOID SHIPPED oracle (live sklearn — the calibrated probabilities MATCH,
  R-CHAR-3): the AC-SIGMOID snippet prints sklearn `expit(-(A*f+B))` =
  `[0.098873, 0.218482, 0.532053, 0.644747, 0.822198, 0.921766, 0.149037, 0.462024,
  0.880687, 0.270133]`; ferrolearn `sigmoid_fn(a*f+b)` over `fit_sigmoid(scores,
  labels)` reproduces these to MAX ABS DIFF **6.66e-9**. The critic may pin this exact
  vector as an oracle-grounded `#[test]` on the isolated `fit_sigmoid`.
- REQ-ISOTONIC DIVERGENCE oracle (#1800 — the pin): the AC-ISOTONIC snippet prints
  sklearn `IsotonicRegression(out_of_bounds="clip").predict(q)` =
  `[0.0, 0.0, 0.2, 0.5, 0.5, 0.8, 1.0, 1.0]`; ferrolearn `isotonic_lookup` gives
  `[0.0, 0.0, 0.08, 0.48, 0.68, 0.92, 1.0, 1.0]` — MAX ABS DIFF **0.18** (midpoint
  breakpoints `[0,1,2.25,3.5,4,5]` vs sklearn `X_thresholds_=[0,1,1.5,3,3.5,5]`). The
  critic pins a failing `#[test]` asserting the sklearn probabilities at `q`.
- REQ-ENSEMBLE DIVERGENCE oracle (#1801): AC-ENSEMBLE — `CalibratedClassifierCV(
  GaussianNB(), cv=3)` → `len(calibrated_classifiers_) == 3` (K averaged); ferrolearn
  produces one calibrator (ensemble=False).
- REQ-MULTICLASS DIVERGENCE oracle (#1803): AC-MULTICLASS — 3-class `predict_proba`
  shape `(2, 3)`, rows sum to 1.0; ferrolearn returns a single `Array1<f64>`.
- REQ-CALIBRATION-CURVE ABSENT oracle (#1804): AC-CALIBRATION-CURVE — `calibration_curve(
  [0,0,0,0,1,1,1,1,1], [.1,.2,.3,.4,.65,.7,.8,.9,1.], n_bins=3)` → `([0,0.5,1],
  [0.2,0.525,0.85])`; no ferrolearn symbol exists.
- REQ-X-2 consumer: `grep -n "pub use calibration" ferrolearn-model-sel/src/lib.rs`
  shows `pub use calibration::{CalibratedClassifierCV, CalibrationMethod,
  FittedCalibratedClassifierCV};`. `grep -rn "CalibratedClassifierCV"
  ferrolearn-*/src/ | grep -v 'tests\|calibration.rs'` shows ONLY the re-export (no
  dedicated internal caller, no Python binding — honest underclaim).
- REQ-X-1 substrate: `grep -n "ndarray" ferrolearn-model-sel/src/calibration.rs` shows
  `use ndarray::{Array1, Array2}` — wrong substrate, migration owed (#1808).

SHIPPED: REQ-SIGMOID (Platt calibrated probabilities — VERIFIED MATCH to 6.66e-9),
REQ-X-2 (boundary re-export consumer; no dedicated caller / no Python binding —
honest underclaim). NOT-STARTED: REQ-ISOTONIC (block-midpoint divergence, #1800 —
DETERMINISTIC FIXABLE), REQ-ENSEMBLE (ensemble=True K-averaging default, #1801 —
architectural), REQ-STRATIFIED-CV (non-stratified folds, #1802), REQ-MULTICLASS
(binary-only, #1803 — architectural), REQ-CALIBRATION-CURVE (missing function,
#1804), REQ-SAMPLE-WEIGHT (#1805), REQ-SIGMOID-RESCALE (max|F|>=30, #1806),
REQ-METHOD-DEFAULT (#1807), REQ-X-1 (ferray substrate, #1808).

Per R-DEFER-2 every REQ is binary SHIPPED/NOT-STARTED. The DETERMINISTIC FIXABLE
divergence the critic should pin as a FAILING test is **REQ-ISOTONIC** (#1800 —
calibrated probabilities at query points vs `IsotonicRegression(out_of_bounds="clip")`).
REQ-SIGMOID is verified SHIPPED (the calibrated probabilities match the live oracle
to 6.66e-9) — NOT a pin. The remaining NOT-STARTED REQs are architectural/missing
(ensemble #1801, stratified cv #1802, multiclass #1803, calibration_curve #1804,
sample_weight #1805) or edge/API-shape/substrate (#1806/#1807/#1808) — blockers,
not pins.

Least-confident SHIPPED claim: REQ-SIGMOID — it is verified SHIPPED only for
`|F| < 30` well-conditioned scores; the moment scores exceed the
`max_abs_prediction_threshold=30` the absence of sklearn's rescaling
(REQ-SIGMOID-RESCALE, #1806) can make the optimiser conditioning — and thus the
converged probabilities — diverge, so the "shipped" surface is the `|F|<30` regime,
not the full Platt contract.
