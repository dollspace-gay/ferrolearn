# Classification & Ranking Metrics (sklearn.metrics classification + ranking-curve functions)

<!--
tier: 3-component
status: draft
baseline-commit: 2df4489e
upstream: scikit-learn 1.5.2 (commit 156ef14); `class_likelihood_ratios` tracked from local sklearn mirror commit f1cc4e7
upstream-paths:
  - sklearn/metrics/_classification.py   # accuracy_score (:169); confusion_matrix (:257); multilabel_confusion_matrix (:431); cohen_kappa_score (:649); jaccard_score (:752); matthews_corrcoef (:940); zero_one_loss (:1037); f1_score (:1132); fbeta_score (:1325); precision_recall_fscore_support (:1614); precision_score (:2056); recall_score (:2236); class_likelihood_ratios (local mirror :2240); balanced_accuracy_score (:2407); classification_report (:2508); hamming_loss (:2739); log_loss (:2843); hinge_loss (:2998); brier_score_loss (:3151); d2_log_loss_score (:3289)
  - sklearn/metrics/_ranking.py          # auc (:52); average_precision_score (:127); det_curve (:282); roc_auc_score (:420); precision_recall_curve (:879); roc_curve (:1053); top_k_accuracy_score (:1891)
ferrolearn-module: ferrolearn-metrics/src/classification.rs
parity-ops: accuracy_score, precision_score, recall_score, f1_score, fbeta_score, jaccard_score, precision_recall_fscore_support, confusion_matrix, multilabel_confusion_matrix, matthews_corrcoef, cohen_kappa_score, class_likelihood_ratios, balanced_accuracy_score, hamming_loss, zero_one_loss, classification_report, log_loss, hinge_loss, brier_score_loss, d2_log_loss_score, roc_auc_score, roc_curve, precision_recall_curve, det_curve, average_precision_score, auc, top_k_accuracy_score, calibration_curve, d2_brier_score
crosslink-issue: 806  # tracking issue for the classification-metrics unit (Layer 1 — ferrolearn-metrics block)
-->

## Summary

`ferrolearn-metrics/src/classification.rs` mirrors the classification-metric
functions of scikit-learn's `sklearn/metrics/_classification.py` plus the binary
ranking-curve functions hosted in `sklearn/metrics/_ranking.py` (`roc_curve`,
`precision_recall_curve`, `auc`, `average_precision_score`, `det_curve`,
`roc_auc_score`, `top_k_accuracy_score`). It exposes **30** public functions and
two public enums (`Average`, `ClassLikelihoodUndefined`).

Twenty-eight of the 30 functions have a direct scikit-learn counterpart; **one
(`d2_brier_score`) has NO sklearn 1.5.2 analog** — `from sklearn.metrics import
d2_brier_score` raises `ImportError` — and
`class_likelihood_ratios_with_options` is the Rust options-bearing companion for
sklearn's keyword parameters. `calibration_curve` mirrors
`sklearn.calibration.calibration_curve` (`sklearn/calibration.py:937`), which
lives outside the two routed upstream files.

Under honest underclaim (R-HONEST-3): a **core subset value-matches the live
sklearn 1.5.2 oracle on its supported signature** — `accuracy_score`,
`confusion_matrix` (default labels), `auc`, `roc_curve` arrays
(`drop_intermediate=False` equivalent), `precision_recall_curve` arrays,
`average_precision_score` (step integration, binary), `roc_auc_score` (binary),
`balanced_accuracy_score`, `matthews_corrcoef`, `jaccard_score`, `fbeta_score`,
`brier_score_loss` (default `pos_label`), `hinge_loss` (binary),
`top_k_accuracy_score` (normalize default), `hamming_loss`, `zero_one_loss`,
`multilabel_confusion_matrix`, the precision/recall/F1 family, and
`d2_log_loss_score`, plus `class_likelihood_ratios` including `labels`,
`sample_weight`, and undefined replacement options — within their narrower
(binary/integer-label, mostly no-`sample_weight`) contract. But **most functions
DIVERGE on at least one
parameter or edge** the oracle exercises:

1. **`log_loss` clips probabilities to `1e-15`** (hard-coded `EPS` constant);
   sklearn 1.5.2 clips to `np.finfo(y_pred.dtype).eps = 2.22e-16`
   (`_classification.py:2951,2962`). For a sample whose true-class probability is
   exactly `0`, ferrolearn returns `-ln(1e-15) = 34.5388`, sklearn
   `-ln(2.22e-16) = 36.0437` — a fully deterministic value divergence.
2. **`cohen_kappa_score` has no `weights` parameter** (always unweighted);
   sklearn's `weights='linear'`/`'quadratic'` (`_classification.py:649`) give
   materially different values.
3. **`roc_curve` does not implement `drop_intermediate`** (default `True` in
   sklearn, `_ranking.py:1053`); ferrolearn always keeps every distinct-score
   point — i.e. it reproduces sklearn's `drop_intermediate=False` arrays, so it
   diverges on the **default** call whenever collinear points exist.
4. **`det_curve` keeps the prepended `(0,0)`/`+inf` ROC endpoint**; sklearn's
   `det_curve` (`_ranking.py:282`) drops it, yielding fewer points.
5. **`calibration_curve` bins by `floor(prob·n_bins)`**; sklearn bins by
   `np.searchsorted(bins[1:-1], prob)` (`sklearn/calibration.py:1036`), so a
   value on a bin boundary (e.g. `0.5` with `n_bins=2`) lands in a different bin.

Pervasive signature gaps (every affected function): **`sample_weight`** (absent
on most classification metrics; present for `class_likelihood_ratios` through
`class_likelihood_ratios_with_options`), **`zero_division`** (absent on precision/recall/F1/fbeta/jaccard —
ferrolearn hard-codes the `0.0` convention via `safe_div`, which matches sklearn's
**default** but not `zero_division=1`/`np.nan`), **`labels`** (absent on
confusion_matrix/log_loss/hinge/precision_recall_fscore_support/etc.),
**`pos_label`** (absent on brier/precision/recall/curves — ferrolearn hard-codes
positive class = label `1`), **`normalize`** (absent on accuracy/log_loss/top_k —
present only on `zero_one_loss`), and **multiclass `roc_auc_score`**
(`multi_class='ovr'/'ovo'` + `average`, `_ranking.py:420`) which ferrolearn
cannot express because it takes a 1-D `Array1<f64>` score, not a
`(n_samples, n_classes)` probability matrix. The label substrate is
`Array1<usize>` (sklearn accepts arbitrary hashable labels via `LabelEncoder`).

All 28 functions + `Average` are existing pub APIs re-exported at the crate root
(`lib.rs`: `pub use classification::{ … }`); that re-export is the non-test
production-consumer surface and the functions are grandfathered under
S5/R-DEFER-1. (The in-crate `scorer.rs` does **not** yet wire any classification
metric — its classification-scorer registry is NOT-STARTED, blocked on #783 — so
`scorer.rs` is not a consumer here.) The existing `#[test]`s pin only
ferrolearn's narrower behavior and do not establish sklearn parity.

## Algorithm (sklearn — the contract)

### Count-based metrics (precision / recall / F / jaccard / support)

`precision_recall_fscore_support` (`_classification.py:1614`) is the engine.
Per-class TP/FP/FN come from `multilabel_confusion_matrix` (`:431`); the per-class
ratios are `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, and
`fbeta = (1+β²)·P·R / (β²·P + R)`. A 0/0 ratio is resolved by **`zero_division`**
(default `0.0`, with an `UndefinedMetricWarning`; also accepts `1.0` and
`np.nan`). `average` (`'binary'` default with `pos_label=1`, plus `'micro'`,
`'macro'`, `'weighted'`, `'samples'`, or `None`) selects the reduction:
`'micro'` sums TP/FP/FN globally first; `'macro'` is the unweighted per-class
mean; `'weighted'` weights per-class scores by true-class support.
`precision_score` (`:2056`), `recall_score` (`:2236`), `f1_score` (`:1132`),
`fbeta_score` (`:1325`), and `jaccard_score` (`:752`, ratio `TP/(TP+FP+FN)`) all
delegate to this engine. `classification_report` (`:2508`) formats the per-class
table + `accuracy` + `macro avg` + `weighted avg` rows (default `digits=2`).

### Confusion matrices

`confusion_matrix` (`:257`): `C[i,j]` = count of true `i` predicted `j` over
`labels` (default = sorted union); supports `sample_weight` and
`normalize='true'/'pred'/'all'` (row/col/grand-total normalization).
`multilabel_confusion_matrix` (`:431`): per-class `[[TN,FP],[FN,TP]]` stack.

### Agreement / correlation

`matthews_corrcoef` (`:940`): Gorodkin multiclass MCC from the confusion matrix;
**zero denominator → returns `0.0`**. `cohen_kappa_score` (`:649`):
`(po − pe)/(1 − pe)` with optional `weights='linear'/'quadratic'` reweighting the
disagreement matrix. `balanced_accuracy_score` (`:2407`): mean per-class recall;
`adjusted=True` rescales by chance `1/n_classes` to `(score − chance)/(1 − chance)`.

### Loss metrics

`accuracy_score` (`:169`): fraction correct (`normalize=False` → raw count),
`sample_weight`. `hamming_loss` (`:2739`) = `1 − accuracy` (single-label).
`zero_one_loss` (`:1037`): misclassification count/fraction. `log_loss`
(`:2843`): `−Σ y·ln(clip(p, eps, 1−eps))`, **`eps = np.finfo(y_pred.dtype).eps`
(`:2951`)**, with `labels`, `sample_weight`, and `normalize`. `hinge_loss`
(`:2998`): `mean(max(0, 1 − decision·sign))` (binary) or the multiclass
crammer-singer form via `pred_decision` 2-D. `brier_score_loss` (`:3151`):
`mean((p − y)²)` over the `pos_label`-encoded target. `d2_log_loss_score`
(`:3289`): `1 − logloss(model)/logloss(prior)`.

### Ranking curves & areas (`_ranking.py`)

`roc_curve` (`:1053`): FPR/TPR at descending thresholds, prepends `(0,0)` with
threshold `+inf`, and by **default `drop_intermediate=True`** drops collinear
interior points. `precision_recall_curve` (`:879`): precision/recall at
descending thresholds, appends the `(precision=1, recall=0)` sentinel;
`thresholds` has one fewer element. `det_curve` (`:282`): `(FPR, FNR, thresholds)`
**without** the ROC `(0,0)`/`+inf` endpoint. `auc` (`:52`): trapezoidal area.
`average_precision_score` (`:127`): **step integration**
`−Σ diff(recall)·precision[:-1]` (`:236`) — NOT trapezoidal. `roc_auc_score`
(`:420`): binary AUC, or multiclass `multi_class='ovr'/'ovo'` with
`average='macro'/'weighted'` over a `(n_samples, n_classes)` proba matrix.
`top_k_accuracy_score` (`:1891`): fraction (or count, `normalize=False`) whose
true label is among the top-`k` scores.

## ferrolearn (what exists)

All public functions live in `ferrolearn-metrics/src/classification.rs`. Label
inputs are `&Array1<usize>`; scores/probabilities are `&Array1<F>`/`&Array1<f64>`
or `&Array2<f64>`. Public enums are `Average { Binary, Macro, Micro, Weighted }`
(no `Samples`, no `None`/per-class output) and
`ClassLikelihoodUndefined` (Rust modeling of `replace_undefined_by`). The 0/0
convention for the averaged metrics is the private `fn safe_div` (returns `0.0`)
— equivalent to sklearn's **default** `zero_division=0.0` but not configurable.

- **`pub fn accuracy_score(y_true, y_pred)`** — fraction correct; **no
  `normalize`, no `sample_weight`** (matches sklearn default value).
- **`pub fn precision_score` / `recall_score` / `f1_score`
  `(y_true, y_pred, average: Average)`** — delegate to `fn aggregate_metric` /
  `fn aggregate_recall` / `fn aggregate_f1` over `fn per_class_counts`. `Binary`
  requires exactly 2 classes and uses **index 1** as positive (no `pos_label`);
  no `zero_division`/`sample_weight`. Micro/macro/weighted value-match the oracle.
- **`pub fn fbeta_score(y_true, y_pred, beta: f64, average)`** — `beta`
  **required positional** (sklearn keyword-only, no default); rejects `beta ≤ 0`.
- **`pub fn jaccard_score(y_true, y_pred, average)`** — per-class
  `TP/(TP+FP+FN)`; no `pos_label`/`zero_division`/`sample_weight`.
- **`pub fn precision_recall_fscore_support(y_true, y_pred, beta, average)`** —
  returns `(f64, f64, f64, usize)`; **support is the total sample count**, not
  per-class; no `labels`/`zero_division`/`sample_weight`/`warn_for`.
- **`pub fn confusion_matrix(y_true, y_pred)`** — sorted-union labels only; **no
  `labels` reordering, no `sample_weight`, no `normalize`**.
- **`pub fn class_likelihood_ratios(y_true, y_pred)`** and
  **`pub fn class_likelihood_ratios_with_options(...)`** — binary LR+/LR-
  values match sklearn, including explicit `[negative, positive]` labels,
  `sample_weight`, and undefined-ratio replacement. Still typed on
  `Array1<usize>` rather than arbitrary sklearn labels.
- **`pub fn multilabel_confusion_matrix(y_true, y_pred)`** — `(n_classes, 2, 2)`
  `[[TN,FP],[FN,TP]]` stack; sorted-union classes; no `labels`/`sample_weight`.
- **`pub fn matthews_corrcoef(y_true, y_pred)`** — Gorodkin MCC; **zero
  denominator → `Ok(0.0)`** (matches sklearn); no `sample_weight`.
- **`pub fn cohen_kappa_score(y_true, y_pred)`** — unweighted only; **no
  `weights` parameter** (diverges from sklearn linear/quadratic); no `labels`/
  `sample_weight`.
- **`pub fn balanced_accuracy_score(y_true, y_pred, adjusted: bool)`** —
  mean per-class recall over classes with nonzero support; `adjusted` rescales by
  `1/counted`; no `sample_weight`. Value-matches the oracle.
- **`pub fn hamming_loss(y_true, y_pred)`** = `1 − accuracy_score`; **single-label
  only**; no `sample_weight`.
- **`pub fn zero_one_loss(y_true, y_pred, normalize: bool)`** — has `normalize`;
  no `sample_weight`.
- **`pub fn log_loss(y_true, y_prob: &Array2<f64>)`** — **clamps to a hard-coded
  `EPS = 1e-15`** (diverges from `np.finfo(dtype).eps = 2.22e-16`); always
  normalized (no `normalize`); no `labels`/`sample_weight`; labels index columns.
- **`pub fn hinge_loss(y_true, pred_decision: &Array1<f64>)`** — binary only
  (maps `0→−1`, `1→+1`); **no multiclass 2-D `pred_decision`**; no `labels`/
  `sample_weight`. Binary value-matches.
- **`pub fn brier_score_loss(y_true, y_prob: &Array1<f64>)`** — `mean((y−p)²)`;
  **no `pos_label`** (positive class hard-coded to `1`); no `sample_weight`.
  Default-`pos_label` value matches.
- **`pub fn d2_log_loss_score(y_true, y_prob: &Array2<f64>)`** — `1 −
  logloss(model)/logloss(prior)`; inherits log_loss's `EPS` divergence; no
  `sample_weight`/`labels`.
- **`pub fn d2_brier_score(y_true, y_prob)`** — **no sklearn 1.5.2 analog**
  (`ImportError`); out-of-scope invention.
- **`pub fn roc_curve(y_true, y_score) -> CurveResult<F>`** — prepends `(0,0)`/
  `+inf`, one point per distinct score; **no `drop_intermediate`** (reproduces
  `drop_intermediate=False`); no `pos_label`/`sample_weight`; rejects labels > 1.
- **`pub fn precision_recall_curve(y_true, y_score)`** — appends `(P=1, R=0)`;
  `thresholds` one shorter; arrays value-match the oracle; no `pos_label`/
  `sample_weight`.
- **`pub fn det_curve(y_true, y_score)`** — `roc_curve` with `fnr = 1 − tpr`,
  **keeping the prepended `(0,0)`/`+inf` endpoint** (sklearn drops it); no
  `pos_label`/`sample_weight`.
- **`pub fn auc(x, y) -> Result<F>`** — trapezoidal; value-matches.
- **`pub fn average_precision_score(y_true, y_score) -> Result<F>`** — **step
  integration** `Σ |R_{k−1}−R_k|·P_{k−1}` over the PR curve; binary value-matches
  the sklearn step AP (`_ranking.py:236`); no `pos_label`/`sample_weight`/
  multiclass `average`.
- **`pub fn roc_auc_score(y_true, y_score: &Array1<f64>) -> Result<f64>`** —
  **binary only**; **no `multi_class='ovr'/'ovo'`** path (cannot take a
  `(n_samples, n_classes)` proba matrix); no `sample_weight`/`max_fpr`.
- **`pub fn top_k_accuracy_score(y_true, y_score: &Array2<f64>, k: usize)`** —
  fraction in top-`k`; **no `normalize` (always fraction), no `labels`,
  no `sample_weight`**.
- **`pub fn calibration_curve(y_true, y_prob, n_bins) -> (Array1<F>, Array1<F>)`**
  — uniform bins via **`floor(prob·n_bins)`** (diverges from sklearn's
  `searchsorted(bins[1:-1], prob)` on bin boundaries); **no `strategy`
  (`'quantile'`), no `pos_label`**; omits empty bins.
- **`pub fn classification_report(y_true, y_pred) -> String`** — per-class +
  accuracy + macro/weighted rows; fixed `digits=4` formatting; no `labels`/
  `target_names`/`sample_weight`/`output_dict`.

**Internal helpers:** `fn check_same_length`, `fn unique_classes`,
`fn per_class_counts`, `fn safe_div` (the `0.0` zero-division), `fn aggregate_metric`/
`fn aggregate_recall`/`fn aggregate_f1`, `fn validate_binary_scores`,
`fn sort_by_score_desc`, `fn resolve_class_likelihood_labels`,
`fn validate_class_likelihood_replacements`.

**Consumers (non-test):** crate re-export (`lib.rs`: `pub use classification::{
Average, ClassLikelihoodUndefined, accuracy_score, … d2_log_loss_score }` — all
30 fns + the two enums).
These are existing pub APIs (grandfathered, S5/R-DEFER-1). `scorer.rs` does NOT
yet consume any classification metric (its classification-scorer registry is
NOT-STARTED, blocked on #783). No `ferrolearn-python` binding exposes the
classification metrics (a binding gap, REQ-17).

## Requirements

- REQ-1: **precision/recall/F1/fbeta/jaccard parity (R-DEV-1/2).** Match
  `precision_score` (`:2056`) / `recall_score` (`:2236`) / `f1_score` (`:1132`) /
  `fbeta_score` (`:1325`) / `jaccard_score` (`:752`): `average='binary'` default
  with **`pos_label`**, configurable **`zero_division` (`0.0`/`1.0`/`np.nan`)**,
  `'samples'`/`None` averaging, `labels`, `sample_weight`. (Micro/macro/weighted
  values + default-`zero_division=0.0` already match.)
- REQ-2: **`accuracy_score` parity (R-DEV-1/2).** Match `:169`: `normalize=False`
  count, `sample_weight`. (Default fraction already matches.)
- REQ-3: **`confusion_matrix` parity (R-DEV-1/2/3).** Match `:257`: **`labels`
  reordering/subsetting**, `sample_weight`, `normalize='true'/'pred'/'all'`.
  (Default sorted-union counts already match.)
- REQ-4: **`multilabel_confusion_matrix` parity (R-DEV-1/2).** Match `:431`:
  `labels`, `sample_weight`, `samplewise`. (Default per-class stack matches.)
- REQ-5: **`matthews_corrcoef` parity (R-DEV-1/2).** Match `:940`: `sample_weight`.
  (Value + zero-denominator→`0.0` already match.)
- REQ-6: **`cohen_kappa_score` parity (R-DEV-1/2).** Match `:649`: **`weights`
  (`None`/`'linear'`/`'quadratic'`)**, `labels`, `sample_weight`. (Unweighted
  value matches; weighted diverges.)
- REQ-7: **`balanced_accuracy_score` parity (R-DEV-1/2).** Match `:2407`:
  `sample_weight`. (`adjusted=False`/`True` values already match.)
- REQ-8: **`log_loss` parity (R-DEV-1/5).** Match `:2843`: **clip to
  `np.finfo(dtype).eps = 2.22e-16` not `1e-15`** (`:2951,2962`), `labels`,
  `normalize`, `sample_weight`. Deterministic value divergence on prob-0/1.
- REQ-9: **`hinge_loss` parity (R-DEV-1/2).** Match `:2998`: **multiclass 2-D
  `pred_decision`**, `labels`, `sample_weight`. (Binary value matches.)
- REQ-10: **`brier_score_loss` parity (R-DEV-1/2).** Match `:3151`: **`pos_label`**,
  `sample_weight`. (Default-`pos_label` value matches.)
- REQ-11: **`roc_auc_score` parity (R-DEV-1/2/3).** Match `:420`: **multiclass
  `multi_class='ovr'/'ovo'`** over a `(n_samples, n_classes)` proba matrix,
  `average`, `sample_weight`, `max_fpr`. (Binary value matches.)
- REQ-12: **`roc_curve` parity (R-DEV-1/3).** Match `:1053`: **`drop_intermediate`
  (default `True`)**, `pos_label`, `sample_weight`. (Arrays match the
  `drop_intermediate=False` case.)
- REQ-13: **`precision_recall_curve` parity (R-DEV-1/3).** Match `:879`:
  `pos_label`, `sample_weight`, `drop_intermediate`. (Default arrays + endpoint
  convention already match.)
- REQ-14: **`det_curve` parity (R-DEV-1/3).** Match `:282`: **drop the
  prepended `(0,0)`/`+inf` ROC endpoint**, `pos_label`, `sample_weight`.
- REQ-15: **`average_precision_score` parity (R-DEV-1/2).** Match `:127,236`:
  `pos_label`, `sample_weight`, multiclass `average`. (Binary **step** value
  already matches sklearn's `−Σ diff(recall)·precision[:-1]`.)
- REQ-16: **`top_k_accuracy_score` parity (R-DEV-1/2).** Match `:1891`:
  **`normalize=False` count**, `labels`, `sample_weight`. (Default fraction matches.)
- REQ-17: **`calibration_curve` parity (R-DEV-1).** Match
  `sklearn/calibration.py:937`: **`searchsorted` binning not `floor`** (`:1036`),
  `strategy='quantile'`, `pos_label`. Deterministic boundary divergence.
- REQ-18: **`classification_report` parity (R-DEV-2/3).** Match `:2508`:
  `labels`, `target_names`, `digits` (default 2, ferro 4), `output_dict`,
  `sample_weight`, **per-class support** (ferro support is total).
- REQ-19: **`precision_recall_fscore_support` parity (R-DEV-1/3).** Match `:1614`:
  **per-class `support`** (ferro returns total `n`), `zero_division`, `labels`,
  `sample_weight`. (Aggregated p/r/f values match.)
- REQ-20: **`d2_log_loss_score` parity (R-DEV-1).** Match `:3289`: `sample_weight`,
  `labels`; inherits the REQ-8 `EPS` divergence via `log_loss`.
- REQ-21: **`d2_brier_score` has no sklearn 1.5.2 analog.** `from sklearn.metrics
  import d2_brier_score` → `ImportError`. Out-of-scope invention (goal.md "we
  translate, not innovate"); not a translation unit.
- REQ-22: **Arbitrary-label substrate (R-DEV-1/3).** Every function takes
  `Array1<usize>` labels; sklearn accepts arbitrary hashable labels via
  `LabelEncoder` and surfaces them as `classes_`/`pos_label`.
- REQ-23: **PyO3 binding (R-DEFER-1).** `import sklearn.metrics` exposes these
  classification metrics; `ferrolearn-python` exposes no shim.
- REQ-24: **ferray substrate (R-SUBSTRATE).** `classification.rs` imports
  `ndarray::{Array1, Array2, Array3}` + `num_traits::Float`, not `ferray-core`.
- REQ-25: **`class_likelihood_ratios` parity (R-DEV-1/2).** Match local mirror
  `_classification.py:2240`: binary LR+/LR- values, `[negative, positive]`
  `labels`, `sample_weight`, and `replace_undefined_by` replacement ranges.
  SHIPPED for `Array1<usize>` labels via `ClassLikelihoodUndefined`; arbitrary
  hashable labels remain covered by REQ-22.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (run from `/tmp`),
never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-8 pin, cleanest): `log_loss([0,1], [[0,1],[1,0]], labels=[0,1])`
  must equal sklearn **36.0437** (`−ln(np.finfo(float64).eps)`). ferrolearn
  clamps to `1e-15` and returns **34.5388**. FAILS.
- AC-2 (REQ-6 pin): `cohen_kappa_score([0,1,2,2,1],[0,2,2,1,1],weights='linear')`
  must equal sklearn **0.5** and `weights='quadratic'` **0.642857…**; ferrolearn
  (unweighted, `0.375`) cannot express `weights`. FAILS.
- AC-3 (REQ-12 pin): `roc_curve([0,0,0,0,1,1,1,1],[0.1..0.8])` default must
  return thresholds **`[inf, 0.8, 0.5, 0.1]`** (len 4, `drop_intermediate=True`).
  ferrolearn returns **`[inf, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]`** (len 9).
  FAILS.
- AC-4 (REQ-17 pin): `calibration_curve([0,1,0,1],[0.5,0.5,0.0,1.0],n_bins=2)`
  must equal sklearn `frac_pos=[0.3333, 1.0]`, `mean_pred=[0.3333, 1.0]` (the
  two `0.5` samples land in bin **0** via `searchsorted`). ferrolearn's
  `floor(0.5·2)=1` puts them in bin **1**, giving different bins. FAILS.
- AC-5 (REQ-14 pin): `det_curve([0,0,1,1],[0.1,0.4,0.35,0.8])` must equal sklearn
  `fpr=[0.5,0.5,0.0]`, `fnr=[0.0,0.5,0.5]`, `th=[0.35,0.4,0.8]` (3 points, no
  `(0,0)`/`+inf`). ferrolearn keeps the prepended endpoint (`fpr[0]=0`,
  `fnr[0]=1`, `th[0]=+inf`), giving an extra point. FAILS.
- AC-6 (REQ-11 pin): `roc_auc_score(y3, proba3x3, multi_class='ovr')` must return
  a scalar OVR-macro AUC; ferrolearn cannot accept a `(n_samples, n_classes)`
  proba matrix — signature-blocked. FAILS.
- AC-7 (REQ-1 pin): `precision_score([0,0],[1,1], zero_division=1)` exercises the
  configurable convention (this exact case is still `0.0` because TP=0,FP=2 is a
  genuine `0/2`); a true 0/0 (`precision_score([1,1],[0,0],zero_division=1)` for
  the absent class via per-class output) yields `1.0` under `zero_division=1`,
  `np.nan` under `np.nan`. ferrolearn's `safe_div` hard-codes `0.0`. FAILS.
- AC-8 (REQ-1 `pos_label`): `recall_score([0,1],[0,1], pos_label=0)` reports the
  negative class; ferrolearn's `Binary` is hard-wired to index 1. FAILS.
- AC-9 (value-correct baselines, must stay green): `accuracy_score([0,1,2,1,0],
  [0,1,2,0,0]) == 0.8`; `roc_auc_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.75`;
  `average_precision_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.8333…` (step);
  `matthews_corrcoef([0,1,1,0],[0,1,0,0]) == 0.5773…`;
  `balanced_accuracy_score([0,0,1,1,2,2],[0,1,1,1,2,0],adjusted=True) == 0.5`;
  `precision_recall_curve` arrays `p=[0.5,0.667,0.5,1,1] r=[1,1,0.5,0.5,0]`;
  `confusion_matrix` default counts; `f1_score(...,Micro) == accuracy` for
  balanced classes — all already match sklearn and bound the correctness present.
- AC-10 (REQ-25, shipped): `class_likelihood_ratios([1]*3+[0]*17,
  [1]*2+[0]*10+[1]*8)` must return `LR+ = 34/24`, `LR- = 17/27`; with
  `sample_weight=[1]*15+[0]*5`, it must return `24/9` and `12/27`.

## REQ status table

Binary (R-DEFER-2). All 30 functions + `Average`/`ClassLikelihoodUndefined` are existing pub APIs
re-exported at the crate root (`lib.rs`); that re-export is the non-test
production-consumer surface (grandfathered S5/R-DEFER-1). Cites use symbol anchors
(ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed sklearn
1.5.2, run from `/tmp`. Honest underclaim (R-HONEST-3): a function is **SHIPPED**
only where its supported signature value-matches the oracle on the contract it
exposes AND has a non-test consumer + tests + symbol anchor; otherwise
**NOT-STARTED** with an open prereq blocker.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (precision/recall/f1/fbeta/jaccard) | NOT-STARTED | open prereq blocker #807. `pub fn precision_score`/`recall_score`/`f1_score`/`fbeta_score`/`jaccard_score` micro/macro/weighted **value-match** the oracle (`precision_score([0,1,1,0,1],[0,1,0,1,1])==2/3`; macro 3-class `0.4444`) and `safe_div` reproduces sklearn's **default** `zero_division=0.0`; but **no `zero_division` arg (`1.0`/`np.nan`), no `pos_label` (`Binary`=index 1), no `labels`/`'samples'`/`None`, no `sample_weight`** (`_classification.py:2056,2236,1132,1325,752`). `fbeta`'s `beta` is required-positional (sklearn keyword-only). |
| REQ-2 (`accuracy_score`) | NOT-STARTED | open prereq blocker #808. `pub fn accuracy_score` default fraction matches (`[0,1,2,1,0]/[0,1,2,0,0]==0.8`); **no `normalize=False` count, no `sample_weight`** (`:169`). |
| REQ-3 (`confusion_matrix`) | NOT-STARTED | open prereq blocker #809. `pub fn confusion_matrix` default sorted-union counts match; **no `labels` reorder/subset, no `sample_weight`, no `normalize='true'/'pred'/'all'`** (`:257`). Pin: `labels=[2,0,1]` reorders rows/cols; `normalize='true'` row-normalizes. |
| REQ-4 (`multilabel_confusion_matrix`) | NOT-STARTED | open prereq blocker #810. `pub fn multilabel_confusion_matrix` default `[[TN,FP],[FN,TP]]` stack matches; no `labels`/`sample_weight`/`samplewise` (`:431`). |
| REQ-5 (`matthews_corrcoef`) | NOT-STARTED | open prereq blocker #811. `pub fn matthews_corrcoef` value + **zero-denominator→`0.0`** match the oracle (`[0,1,1,0]/[0,1,0,0]==0.5774`; all-pred-same→`0.0`); **no `sample_weight`** (`:940`). |
| REQ-6 (`cohen_kappa_score`) | NOT-STARTED | open prereq blocker #812. `pub fn cohen_kappa_score` unweighted value matches (`0.375`); **no `weights` (`'linear'`/`'quadratic'`)** → diverges (sklearn `0.5`/`0.6429`), no `labels`/`sample_weight` (`:649`). Pin: `weights='linear'`→`0.5`, ferro inexpressible. |
| REQ-7 (`balanced_accuracy_score`) | NOT-STARTED | open prereq blocker #813. `pub fn balanced_accuracy_score` `adjusted=False`/`True` values match (`0.6667`/`0.5`); **no `sample_weight`** (`:2407`). |
| REQ-8 (`log_loss`) | NOT-STARTED | open prereq blocker #814. `pub fn log_loss` **clamps to `EPS=1e-15`**; sklearn clips to `np.finfo(dtype).eps=2.22e-16` (`:2951,2962`). Pin: `log_loss([0,1],[[0,1],[1,0]])` → sklearn **36.0437**, ferro **34.5388**. No `labels`/`normalize`/`sample_weight`. |
| REQ-9 (`hinge_loss`) | NOT-STARTED | open prereq blocker #815. `pub fn hinge_loss` binary value matches (`0.55`); **no multiclass 2-D `pred_decision`**, no `labels`/`sample_weight` (`:2998`). |
| REQ-10 (`brier_score_loss`) | NOT-STARTED | open prereq blocker #816. `pub fn brier_score_loss` default-`pos_label` value matches (`0.0375`); **no `pos_label`** (positive class hard-wired to `1`), no `sample_weight` (`:3151`). Pin: `pos_label=0`→`0.6875`, ferro inexpressible. |
| REQ-11 (`roc_auc_score`) | NOT-STARTED | open prereq blocker #817. `pub fn roc_auc_score` binary value matches (`[0,0,1,1]/[0.1,0.4,0.35,0.8]==0.75`); **no multiclass `multi_class='ovr'/'ovo'`** path — takes `Array1<f64>`, not a `(n_samples,n_classes)` proba matrix; no `average`/`sample_weight`/`max_fpr` (`:420`). |
| REQ-12 (`roc_curve`) | NOT-STARTED | open prereq blocker #818. `pub fn roc_curve` arrays match the **`drop_intermediate=False`** case + `(0,0)`/`+inf` prepend; **no `drop_intermediate` (sklearn default `True`)**, no `pos_label`/`sample_weight` (`:1053`). Pin: 8-sample monotone → sklearn `[inf,0.8,0.5,0.1]` (len 4), ferro len 9. |
| REQ-13 (`precision_recall_curve`) | NOT-STARTED | open prereq blocker #819. `pub fn precision_recall_curve` default arrays + `(P=1,R=0)` sentinel + `len(thresholds)=len(p)−1` **value-match** the oracle (`p=[0.5,0.667,0.5,1,1] r=[1,1,0.5,0.5,0]`); **no `pos_label`/`sample_weight`/`drop_intermediate`** (`:879`). |
| REQ-14 (`det_curve`) | NOT-STARTED | open prereq blocker #820. `pub fn det_curve` = `roc_curve` with `fnr=1−tpr`, **keeping the prepended `(0,0)`/`+inf` endpoint** sklearn drops (`:282`); no `pos_label`/`sample_weight`. Pin: `[0,0,1,1]/[0.1,0.4,0.35,0.8]` → sklearn 3 points `fpr=[0.5,0.5,0]`, ferro has the extra `+inf`/`(0,1)` point. |
| REQ-15 (`average_precision_score`) | NOT-STARTED | open prereq blocker #821. `pub fn average_precision_score` binary **step** value matches sklearn `−Σ diff(recall)·precision[:-1]` (`:236`) — `0.8333` and `0.8875` confirmed; **no `pos_label`/`sample_weight`/multiclass `average`** (`:127`). |
| REQ-16 (`top_k_accuracy_score`) | NOT-STARTED | open prereq blocker #822. `pub fn top_k_accuracy_score` default fraction matches; **no `normalize=False` count, no `labels`/`sample_weight`** (`:1891`). |
| REQ-17 (`calibration_curve`) | NOT-STARTED | open prereq blocker #823. `pub fn calibration_curve` bins via **`floor(prob·n_bins)`**; sklearn uses `searchsorted(bins[1:-1], prob)` (`calibration.py:1036`). Pin: `[0.5,0.5,0.0,1.0]`,`n_bins=2` → boundary `0.5` lands in bin 0 (sklearn) vs bin 1 (ferro). No `strategy='quantile'`/`pos_label`; mirrors `sklearn/calibration.py:937` (outside routed upstream). |
| REQ-18 (`classification_report`) | NOT-STARTED | open prereq blocker #824. `pub fn classification_report` emits per-class + accuracy + macro/weighted rows but **fixed `digits=4` (sklearn default 2), total support per row, no `target_names`/`labels`/`output_dict`/`sample_weight`** (`:2508`). |
| REQ-19 (`precision_recall_fscore_support`) | NOT-STARTED | open prereq blocker #825. `pub fn precision_recall_fscore_support` aggregated p/r/f values match but **`support` is total `n`, not per-class** (sklearn returns per-class arrays for `average=None`); `beta` required-positional; no `zero_division`/`labels`/`sample_weight` (`:1614`). |
| REQ-20 (`d2_log_loss_score`) | NOT-STARTED | open prereq blocker #826. `pub fn d2_log_loss_score` `1−logloss(model)/logloss(prior)` structure matches but **inherits the REQ-8 `EPS=1e-15` divergence** via `log_loss`; no `sample_weight`/`labels` (`:3289`). |
| REQ-21 (`d2_brier_score`) | NOT-STARTED | open prereq blocker #827. `pub fn d2_brier_score` has **no sklearn 1.5.2 analog** — `from sklearn.metrics import d2_brier_score` → `ImportError`. Out-of-scope invention (goal.md "we translate, not innovate"); cannot be SHIPPED against a non-existent oracle. |
| REQ-22 (arbitrary-label substrate) | NOT-STARTED | open prereq blocker #828. Every function takes `&Array1<usize>` labels; sklearn accepts arbitrary hashable labels via `LabelEncoder` and exposes `pos_label`/`classes_` over them (`_classification.py` validation path). |
| REQ-23 (PyO3 binding) | NOT-STARTED | open prereq blocker #829. `ferrolearn-python` exposes no classification-metric shim; `import ferrolearn` cannot call what `import sklearn.metrics` provides. |
| REQ-24 (ferray substrate) | NOT-STARTED | open prereq blocker #830. `classification.rs` imports `ndarray::{Array1, Array2, Array3}` + `num_traits::Float`, not `ferray-core` (R-SUBSTRATE). |
| REQ-25 (`class_likelihood_ratios`) | SHIPPED | `pub fn class_likelihood_ratios` + `pub fn class_likelihood_ratios_with_options` mirror local mirror `_classification.py:2240`: binary-only LR+/LR-, sorted default labels or explicit `[negative, positive]`, `sample_weight`, and `replace_undefined_by` via `ClassLikelihoodUndefined`. Verification: `tests/divergence_class_likelihood_ratios.rs` pins sklearn's `34/24`, `17/27`, weighted `24/9`, `12/27`, undefined replacements, and nonbinary errors; crate-root re-export is the production consumer. |

## Architecture

`classification.rs` is a flat module of free functions plus the public `Average`
enum; there are no fitted/unfitted types — these are stateless metrics. Labels
are `Array1<usize>` throughout (the substrate divergence, REQ-22), scores/
probabilities are `Array1<F>`/`Array1<f64>` or `Array2<f64>`. Five families:

1. **Count-based** (`precision_score`, `recall_score`, `f1_score`, `fbeta_score`,
   `jaccard_score`, `precision_recall_fscore_support`, `classification_report`,
   `balanced_accuracy_score`): all route through `fn per_class_counts` (TP/FP/FN
   per class) and `fn aggregate_metric`/`fn aggregate_recall`/`fn aggregate_f1`,
   with `fn safe_div` supplying the **hard-coded `zero_division=0.0`** (REQ-1) and
   `Average::Binary` hard-wiring the positive class to **index 1** (no `pos_label`,
   REQ-1). The micro/macro/weighted values match the oracle; the divergences are
   configurability (`zero_division`, `pos_label`, `'samples'`/`None`, `labels`,
   `sample_weight`) and `precision_recall_fscore_support`'s total-vs-per-class
   `support` (REQ-19) and `classification_report`'s `digits=4` (REQ-18).
2. **Confusion matrices / likelihood ratios** (`confusion_matrix`,
   `multilabel_confusion_matrix`, `class_likelihood_ratios`):
   sorted-union labels via `fn unique_classes`; `confusion_matrix` uses
   `partition_point` to index. Diverge on `labels`/`normalize`/`sample_weight`
   (REQ-3/4). `class_likelihood_ratios_with_options` uses its own weighted
   binary 2x2 fold so LR+/LR- can support sklearn's labels/sample-weight/
   undefined-replacement surface without changing the existing matrix API.
3. **Agreement / correlation** (`matthews_corrcoef`, `cohen_kappa_score`):
   confusion-matrix-derived. `matthews_corrcoef`'s **zero-denominator→`0.0`**
   convention matches sklearn (REQ-5, value-correct). `cohen_kappa_score` is
   unweighted-only — the **missing `weights`** is its consequential divergence
   (REQ-6).
4. **Loss metrics** (`accuracy_score`, `hamming_loss`, `zero_one_loss`,
   `log_loss`, `hinge_loss`, `brier_score_loss`, `d2_log_loss_score`,
   `d2_brier_score`): single-pass folds. `log_loss`'s **`EPS=1e-15` clamp** is the
   single cleanest deterministic value divergence (REQ-8: prob-0 → ferro `34.5388`
   vs sklearn `36.0437`), and it propagates into `d2_log_loss_score` (REQ-20).
   `hinge_loss` is binary-only (REQ-9), `brier_score_loss` hard-wires `pos_label=1`
   (REQ-10), and `d2_brier_score` has no sklearn analog (REQ-21).
5. **Ranking curves & areas** (`roc_curve`, `precision_recall_curve`, `det_curve`,
   `auc`, `average_precision_score`, `roc_auc_score`, `top_k_accuracy_score`,
   `calibration_curve`): `fn validate_binary_scores` + `fn sort_by_score_desc`
   feed the curve walkers. `precision_recall_curve` arrays and
   `average_precision_score` (**step** integration `Σ|R_{k−1}−R_k|·P_{k−1}`,
   matching sklearn `−Σ diff(recall)·precision[:-1]`) and `auc` (trapezoidal) and
   binary `roc_auc_score` are value-correct. The structural divergences are
   `roc_curve`'s **missing `drop_intermediate`** (REQ-12 — ferro = the
   `drop_intermediate=False` arrays), `det_curve`'s **retained `(0,0)`/`+inf`
   endpoint** (REQ-14), `calibration_curve`'s **`floor`-vs-`searchsorted` binning**
   (REQ-17), and `roc_auc_score`'s **absent multiclass `multi_class`** path
   (REQ-11, signature-blocked by the 1-D score input).

**Invariants held vs sklearn (supported signature):** accuracy fraction;
precision/recall/F1/fbeta/jaccard micro/macro/weighted values + default
`zero_division=0.0`; default `confusion_matrix`/`multilabel_confusion_matrix`
counts; MCC value + zero-denom→`0.0`; balanced-accuracy `adjusted` values;
binary `hinge_loss`/`brier_score_loss`(default pos_label)/`roc_auc_score`;
`precision_recall_curve` arrays + endpoint; `auc` trapezoid; binary **step**
`average_precision_score`; `top_k_accuracy_score` default fraction; `hamming_loss`/
`zero_one_loss`; `class_likelihood_ratios` values/options.
**Invariants NOT held vs sklearn:** `log_loss` clip eps (REQ-8);
`cohen_kappa_score` `weights` (REQ-6); `roc_curve` `drop_intermediate` (REQ-12);
`det_curve` endpoint (REQ-14); `calibration_curve` binning (REQ-17); multiclass
`roc_auc_score` (REQ-11); `pos_label` (REQ-1/10); `zero_division` configurability
(REQ-1/19); `normalize` on accuracy/log_loss/top_k (REQ-2/8/16); `labels`/
`sample_weight` (most remaining metrics); per-class `support` (REQ-18/19);
arbitrary labels (REQ-22).

**Function inventory vs sklearn:** 28 of 30 have a direct counterpart in
`_classification.py`/`_ranking.py`; `calibration_curve` mirrors
`sklearn/calibration.py:937` (outside the two routed files); **`d2_brier_score`
has no sklearn 1.5.2 counterpart** (REQ-21), and
`class_likelihood_ratios_with_options` is the Rust options companion for
sklearn's keyword parameters. `precision_recall_fscore_support` exists but still
has per-class output gaps; `dcg`/`ndcg` live in `ranking.rs`.

## Verification

Library crate (green at baseline `2df4489e` for the existing — narrower —
contract):
```
cargo test -p ferrolearn-metrics --lib classification   # 53 passed; 0 failed
cargo test -p ferrolearn-metrics --test divergence_class_likelihood_ratios
cargo clippy -p ferrolearn-metrics --all-targets -- -D warnings
cargo fmt --all --check
```
The existing `#[test]`s (and `#[cfg(kani)]` proofs for accuracy/precision/recall/
f1/log_loss range + confusion-matrix non-negativity) pin only ferrolearn's
narrower behavior; they do NOT establish sklearn parity, so they make no REQ
SHIPPED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the deterministic
divergences a critic should pin first (R-CHAR-3 expected values):
```
# REQ-8 (log_loss clip eps — CLEANEST): sklearn 36.0437 vs ferro 34.5388
python3 -c "import numpy as np; from sklearn.metrics import log_loss; print(log_loss([0,1],np.array([[0.,1.],[1.,0.]]),labels=[0,1]))"  # 36.04365338911715

# REQ-12 (roc_curve drop_intermediate default True): sklearn len 4 vs ferro len 9
python3 -c "from sklearn.metrics import roc_curve; import numpy as np; print(roc_curve([0,0,0,0,1,1,1,1],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])[2].tolist())"  # [inf,0.8,0.5,0.1]

# REQ-14 (det_curve drops (0,0)/inf endpoint): sklearn 3 pts vs ferro 4 (extra +inf/(0,1))
python3 -c "from sklearn.metrics import det_curve; print([a.tolist() for a in det_curve([0,0,1,1],[0.1,0.4,0.35,0.8])])"  # fpr [0.5,0.5,0.0], fnr [0.0,0.5,0.5], th [0.35,0.4,0.8]

# REQ-17 (calibration_curve searchsorted vs floor): boundary 0.5 -> bin 0 (sklearn) vs bin 1 (ferro)
python3 -c "from sklearn.calibration import calibration_curve; print([a.tolist() for a in calibration_curve([0,1,0,1],[0.5,0.5,0.0,1.0],n_bins=2)])"  # [[0.3333,1.0],[0.3333,1.0]]

# REQ-6 (cohen_kappa weights): sklearn linear 0.5 / quadratic 0.6429 vs ferro 0.375 (unweighted only)
python3 -c "from sklearn.metrics import cohen_kappa_score as c; print(c([0,1,2,2,1],[0,2,2,1,1]), c([0,1,2,2,1],[0,2,2,1,1],weights='linear'), c([0,1,2,2,1],[0,2,2,1,1],weights='quadratic'))"  # 0.375 0.5 0.642857

# REQ-11 (roc_auc multiclass): ferro cannot take (n,k) proba matrix
python3 -c "import numpy as np; from sklearn.metrics import roc_auc_score; print(roc_auc_score([0,1,2,2,1,0],np.array([[.7,.2,.1],[.1,.8,.1],[.1,.2,.7],[.2,.2,.6],[.3,.5,.2],[.6,.3,.1]]),multi_class='ovr'))"  # 1.0

# REQ-1 (zero_division / pos_label): ferro safe_div hard-codes 0.0; Binary=index 1
python3 -c "import numpy as np; from sklearn.metrics import precision_score; print(precision_score([0,0],[1,1],zero_division=1))"  # 0.0 (TP=0,FP=2 genuine 0/2)
python3 -c "from sklearn.metrics import recall_score; print(recall_score([0,1],[0,1],pos_label=0))"  # 1.0 (negative-class recall)

# AC-9 baselines that must stay green (value-correct today):
python3 -c "from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score,matthews_corrcoef,balanced_accuracy_score as ba; print(accuracy_score([0,1,2,1,0],[0,1,2,0,0]), roc_auc_score([0,0,1,1],[0.1,0.4,0.35,0.8]), average_precision_score([0,0,1,1],[0.1,0.4,0.35,0.8]), matthews_corrcoef([0,1,1,0],[0,1,0,0]), ba([0,0,1,1,2,2],[0,1,1,1,2,0],adjusted=True))"  # 0.8 0.75 0.8333 0.5774 0.5
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-metrics/tests/divergence_classification.rs`, asserting the
live-sklearn expected values above and FAILING against current
`classification.rs`. Every REQ is NOT-STARTED; each carries an open prereq blocker.

## Blockers to open

- #807 — REQ-1 (precision/recall/f1/fbeta/jaccard): no `zero_division`/`pos_label`/
  `labels`/`'samples'`/`None`/`sample_weight`; `fbeta` `beta` positional
  (`_classification.py:2056,2236,1132,1325,752`).
- #808 — REQ-2 (`accuracy_score`): no `normalize=False`/`sample_weight` (`:169`).
- #809 — REQ-3 (`confusion_matrix`): no `labels`/`sample_weight`/`normalize` (`:257`).
- #810 — REQ-4 (`multilabel_confusion_matrix`): no `labels`/`sample_weight`/
  `samplewise` (`:431`).
- #811 — REQ-5 (`matthews_corrcoef`): no `sample_weight` (`:940`).
- #812 — REQ-6 (`cohen_kappa_score`): no `weights` (`'linear'`/`'quadratic'`)/
  `labels`/`sample_weight` (`:649`). Pin: `weights='linear'`→`0.5`, ferro `0.375`.
- #813 — REQ-7 (`balanced_accuracy_score`): no `sample_weight` (`:2407`).
- #814 — REQ-8 (`log_loss`): clamps `EPS=1e-15` vs `np.finfo(dtype).eps=2.22e-16`
  (`:2951,2962`); no `labels`/`normalize`/`sample_weight`. Pin: prob-0 →
  sklearn `36.0437`, ferro `34.5388`.
- #815 — REQ-9 (`hinge_loss`): no multiclass 2-D `pred_decision`/`labels`/
  `sample_weight` (`:2998`).
- #816 — REQ-10 (`brier_score_loss`): no `pos_label`/`sample_weight` (`:3151`).
  Pin: `pos_label=0`→`0.6875`.
- #817 — REQ-11 (`roc_auc_score`): no multiclass `multi_class='ovr'/'ovo'`/
  `average`/`sample_weight`/`max_fpr`; 1-D score signature blocks proba matrix
  (`:420`).
- #818 — REQ-12 (`roc_curve`): no `drop_intermediate` (default `True`)/`pos_label`/
  `sample_weight` (`:1053`). Pin: 8-sample monotone → sklearn 4 thresholds,
  ferro 9.
- #819 — REQ-13 (`precision_recall_curve`): no `pos_label`/`sample_weight`/
  `drop_intermediate` (`:879`).
- #820 — REQ-14 (`det_curve`): keeps the `(0,0)`/`+inf` ROC endpoint sklearn drops;
  no `pos_label`/`sample_weight` (`:282`).
- #821 — REQ-15 (`average_precision_score`): no `pos_label`/`sample_weight`/
  multiclass `average` (`:127,236`).
- #822 — REQ-16 (`top_k_accuracy_score`): no `normalize=False`/`labels`/
  `sample_weight` (`:1891`).
- #823 — REQ-17 (`calibration_curve`): `floor` binning vs `searchsorted`
  (`calibration.py:1036`); no `strategy='quantile'`/`pos_label`. Pin: `0.5`,
  `n_bins=2` → bin 0 (sklearn) vs bin 1 (ferro).
- #824 — REQ-18 (`classification_report`): `digits=4` vs default 2; total support
  per row; no `labels`/`target_names`/`output_dict`/`sample_weight` (`:2508`).
- #825 — REQ-19 (`precision_recall_fscore_support`): total `support` not per-class;
  `beta` positional; no `zero_division`/`labels`/`sample_weight` (`:1614`).
- #826 — REQ-20 (`d2_log_loss_score`): inherits the `EPS=1e-15` divergence via
  `log_loss`; no `sample_weight`/`labels` (`:3289`).
- #827 — REQ-21 (`d2_brier_score`): no sklearn 1.5.2 analog (`ImportError`);
  out-of-scope invention (goal.md). Decide remove-or-document.
- #828 — REQ-22 (arbitrary-label substrate): every function takes
  `Array1<usize>`; sklearn accepts arbitrary hashable labels via `LabelEncoder`.
- #829 — REQ-23: no `ferrolearn-python` classification-metric binding.
- #830 — REQ-24: migrate `classification.rs` off `ndarray`/`num-traits` to the
  ferray substrate (R-SUBSTRATE).
