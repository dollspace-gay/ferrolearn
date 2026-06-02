# AdaBoost Regressor (AdaBoostRegressor — AdaBoost.R2)

<!--
tier: 3-component
status: draft
baseline-commit: 7394a119
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/ensemble/_weight_boosting.py   # BaseWeightBoosting (:63); _parameter_constraints (:70, n_estimators>=1, learning_rate in (0,inf)); BaseWeightBoosting.fit (:111, sample_weight init 1/n :143-146, clip eps :164, per-iter _boost :169, stop on error==0 :180, normalize :202); AdaBoostRegressor (:971); __init__ defaults (:1094, n_estimators=50/learning_rate=1.0/loss='linear'); _parameter_constraints loss StrOptions{linear,square,exponential} (:1091); _validate_estimator default DecisionTreeRegressor(max_depth=3) (:1115); AdaBoostRegressor._boost (:1117) — weighted bootstrap random_state.choice(replace=True, p=sample_weight) (:1162-1167), error_vect=|y_pred-y| (:1176), error_max normalize (:1181-1183), square/exponential loss (:1185-1188), estimator_error=weighted-avg-loss (:1191), perfect-fit return (:1193-1195), error>=0.5 discard (:1197-1201), beta=err/(1-err) (:1203), estimator_weight=lr*log(1/beta) (:1206), sample-weight update beta**((1-err)*lr) skipping last iter (:1208-1211); _get_median_predict (:1215, sort :1220, stable_cumsum CDF :1223, cdf>=0.5*total argmax :1224-1225); predict (:1232, weighted median :1252); feature_importances_ (:277, sum weight*clf.fi / norm :300-308)
ferrolearn-module: ferrolearn-tree/src/adaboost_regressor.rs
parity-ops: AdaBoostRegressor
crosslink-issue: 702
-->

## Summary

`ferrolearn-tree/src/adaboost_regressor.rs` mirrors scikit-learn's
`sklearn.ensemble.AdaBoostRegressor` (`_weight_boosting.py:971`) — the
**AdaBoost.R2** algorithm (Drucker 1997) over `DecisionTreeRegressor(max_depth=3)`
base learners. Each boosting round (1) draws a weighted sample of the rows,
(2) fits a regression tree, (3) computes a normalized per-sample regression loss
(linear / square / exponential), (4) forms `beta = error/(1-error)` and an
`estimator_weight = learning_rate * log(1/beta)`, and (5) reweights samples by
`beta` raised to a loss-dependent exponent. Prediction is the **weighted median**
of the per-tree outputs (`_get_median_predict`, `:1215`).

ferrolearn re-implements this natively. The module ships the unfitted
`AdaBoostRegressor<F>` + `FittedAdaBoostRegressor<F>`, an `AdaBoostLoss` enum
(`Linear`/`Square`/`Exponential`), `fn new` + `with_*` builders + `Default`,
`Fit`/`Predict`, a `score` (R²), `estimators()`/`estimator_weights()`/
`n_features()` accessors, `HasFeatureImportances`, the private `fn weighted_median`
+ `fn resample_weighted`, and the `PipelineEstimator`/`FittedPipelineEstimator`
adapters. The per-tree build correctness (best split, leaf values) is **inherited
from the oracle-verified `decision_tree.rs`** (`.design/tree/decision_tree.md`)
via `build_regression_tree_with_feature_subset`.

**Two divergence classes drive the REQ split (R-HONEST-3 — underclaim):**

1. **Headline numerical divergence (R-DEV-1) — missing `* learning_rate` in the
   sample-weight exponent.** In `fit`, the reweight is `weights[i] *= beta^(1 -
   loss_i)`. sklearn (`_weight_boosting.py:1209-1211`) does `sample_weight *=
   beta ** ((1.0 - error_vect) * learning_rate)` — ferrolearn **omits the
   `* learning_rate` factor**. The two agree only when `learning_rate == 1.0`
   (the default), so the default path is unaffected, but for any non-unit
   `learning_rate` the reweighted distribution diverges, which feeds the next
   round's weighted-average loss → `beta` → `estimator_weight`, making it
   **prediction-affecting** even within ferrolearn's own deterministic stream.
   This is the single-file fixer's target (REQ-5, blocker #703).

2. **RNG / resampling boundary — end-to-end prediction parity is INFEASIBLE.**
   sklearn fits each tree on a **numpy weighted bootstrap-with-replacement** draw
   (`random_state.choice(np.arange(n), size=n, replace=True, p=sample_weight)`,
   `:1162-1167`) — a fresh stochastic draw every iteration off the numpy MT19937
   stream. ferrolearn fits each tree on a **deterministic systematic resample**
   (`resample_weighted`: build the cumulative-weight CDF, walk it at a fixed
   `step = total/n` from `step/2`) and **uses no RNG at all** despite storing a
   `random_state`. These are different sampling procedures off incompatible
   streams; the exact per-round tree — and therefore the end-to-end prediction at
   a given `random_state` — cannot bit-match sklearn. This is the same
   documented boundary class as the random-forest bootstrap and SGD shuffle
   (`.design/tree/random_forest.md`). ferrolearn's fit is, however, **fully
   deterministic and reproducible** (no RNG → identical output every call),
   which is the pinnable contract (REQ-7, boundary blocker #706).

The *deterministic, structural* parts — defaults/param surface, the three loss
normalizations, the `beta`/`estimator_weight` formula, and the weighted-median
predict — match sklearn term-for-term and are SHIPPED. The lr-exponent reweight
and the numpy-parity/end-to-end parity are NOT-STARTED with concrete blockers.

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `get_params()`, sklearn 1.5.2)

`AdaBoostRegressor().get_params()` →
`{'estimator': None, 'learning_rate': 1.0, 'loss': 'linear', 'n_estimators': 50,
'random_state': None}`. Base estimator default (when `estimator is None`):
`DecisionTreeRegressor(max_depth=3)` (`_validate_estimator`, `:1115`).
Constraints (`:70`, `:1091`): `n_estimators` integer `>= 1`, `learning_rate`
real in `(0, inf)`, `loss in {'linear', 'square', 'exponential'}`.

**Defaults ferrolearn matches** (`fn new`): `n_estimators=50`,
`learning_rate=1.0`, `loss=Linear`, `max_depth=Some(3)` (the `DecisionTreeRegressor(max_depth=3)`
base), `random_state=None`. Validation in `fit` rejects `n_estimators==0`
(`InvalidParameter`) and `learning_rate <= 0` (`InvalidParameter`), mirroring the
constraint intervals.

**Surface differences (not divergences for the default path, noted for honesty):**
ferrolearn exposes `max_depth` directly as a hyperparameter rather than wrapping a
configurable `estimator`; sklearn's pluggable `estimator` (any `fit`/`predict`
learner) is not generic in ferrolearn — the base learner is always a regression
tree (R-DEV-7 implementation choice for the default; a pluggable-base-estimator
gap is REQ-6). sklearn also exposes fitted attributes `estimator_errors_` and
`estimators_`/`estimator_weights_`; ferrolearn exposes `estimators()` and
`estimator_weights()` but **not** an `estimator_errors_` analog (REQ-6).

### fit loop (`BaseWeightBoosting.fit`, `:111`)

sklearn: init `sample_weight = 1/n` (`:143-146`); each iteration clip to machine
eps (`:164`), call `_boost`, store `estimator_weights_[iboost]`/
`estimator_errors_[iboost]`, **stop if `estimator_error == 0`** (`:180`),
normalize `sample_weight /= sum` unless last iteration (`:200-202`).

ferrolearn (`fit`): init `weights = 1/n_samples` uniform; loop `n_estimators`
times: `resample_weighted` → tree build → predictions → losses → weighted-avg
loss → `beta` → `estimator_weight` → reweight → normalize. Matches the uniform
init, the perfect-fit early stop (`max_error <= eps` keeps the tree with weight
1.0 and breaks — sklearn returns weight `1.0` error `0.0` on perfect fit
`:1193-1195` then `fit` breaks on `error==0` `:180`), and the `avg_loss >= 0.5`
discard/stop (`:1197-1201`). One benign ordering detail: sklearn skips the
sample-weight update on the LAST iteration (`if not iboost == n_estimators-1`,
`:1208`); ferrolearn always updates then normalizes — benign because the final
round's weights are never consumed (no subsequent tree).

### _boost — losses, beta, estimator_weight (`:1117`)

- **Weighted bootstrap** (`:1162-1167`): `random_state.choice(arange(n),
  size=n, replace=True, p=sample_weight)`. ferrolearn: `resample_weighted`
  (deterministic systematic resample, no RNG). **Resampling boundary — REQ-7.**
- **Per-sample error** (`:1176`): `error_vect = |y_pred - y|`. ferrolearn:
  `abs_errors[i] = |y[i] - preds[i]|`. Match.
- **Normalize by max** (`:1181-1183`): `error_max = max`; if nonzero,
  `error_vect /= error_max`. ferrolearn: `normalised = e / max_error` with a
  `max_error <= eps` perfect-fit guard. Match.
- **Loss** (`:1185-1188`): `square → e**2`; `exponential → 1 - exp(-e)`;
  linear leaves `e`. ferrolearn `AdaBoostLoss::{Linear,Square,Exponential}`:
  `normalised`, `normalised*normalised`, `1 - (-normalised).exp()`. **Match
  term-for-term (REQ-2).**
- **Average loss** (`:1191`): `estimator_error = (masked_sample_weight *
  masked_error_vector).sum()`. Because `sample_weight` is renormalized to sum 1
  each round, this equals the weight-normalized mean. ferrolearn computes the
  explicit weighted mean `sum(w*l)/sum(w)` — algebraically identical when weights
  sum to 1, and robust when they do not. Match.
- **beta** (`:1203`): `beta = estimator_error / (1 - estimator_error)`.
  ferrolearn: `avg_loss / (1 - avg_loss).max(eps)`. Match (REQ-3).
- **estimator_weight** (`:1206`): `learning_rate * log(1/beta)`. ferrolearn:
  `(1/beta.max(eps)).ln() * learning_rate`. Match (REQ-3).
- **Sample-weight update** (`:1208-1211`): `sample_weight[mask] *= beta **
  ((1 - error_vect) * learning_rate)`. ferrolearn: `weights[i] *= beta^(1 -
  loss_i)` — **MISSING `* learning_rate` in the exponent (REQ-5, blocker #703).**

### predict — weighted median (`_get_median_predict`, `:1215`)

sklearn: collect per-estimator predictions, `argsort` each row, take
`stable_cumsum` of the `estimator_weights_` in sorted order, find the first index
where `weight_cdf >= 0.5 * weight_cdf[:, -1]` (the total), and return that
estimator's prediction (`:1220-1230`). ferrolearn `fn weighted_median`:
sort `(value, weight)` pairs by value, accumulate weight, return the first value
where `cumulative >= total_weight/2`. Structurally identical CDF-crossing rule
(REQ-4). Given a fixed set of trees + weights this is deterministic and
intra-ferrolearn pinnable; tie-break on equal cumulative weight matches the
"first index reaching half" convention.

### feature_importances_ (`:277`)

sklearn property: `sum(weight * clf.feature_importances_ for weight, clf in
zip(estimator_weights_, estimators_)) / estimator_weights_.sum()` (`:300-308`).
ferrolearn: `aggregate_tree_importances(estimators, None, Some(estimator_weights),
n_features)` (in `decision_tree.rs`), the weighted sum of per-tree importances
normalized to sum 1, exposed via `HasFeatureImportances`. Match in form (REQ-8);
the per-tree importance itself is inherited from oracle-verified `decision_tree.rs`.

## ferrolearn (what exists)

- **Unfitted**: `pub struct AdaBoostRegressor<F>` (public fields `n_estimators`,
  `learning_rate`, `max_depth`, `random_state`, `loss`); `fn new`; `with_*`
  builders (`with_n_estimators`, `with_learning_rate`, `with_max_depth`,
  `with_random_state`, `with_loss`); `Default`.
- **Enum**: `pub enum AdaBoostLoss { Linear, Square, Exponential }`.
- **Fitted**: `pub struct FittedAdaBoostRegressor<F>` (`estimators:
  Vec<Vec<Node<F>>>`, `estimator_weights: Vec<F>`, `n_features`,
  `feature_importances: Array1<F>`).
- **Traits**: `Fit<Array2<F>, Array1<F>>`; `Predict<Array2<F>>`;
  `HasFeatureImportances<F>`; `PipelineEstimator<F>` (unfitted) /
  `FittedPipelineEstimator<F>` (fitted).
- **Methods**: `fn estimators`, `fn estimator_weights`, `fn n_features`,
  `fn score` (R² via crate `r2_score`).
- **Internal helpers**: `fn weighted_median`, `fn resample_weighted`.
- **Build delegation** (from `decision_tree.rs`, oracle-verified):
  `build_regression_tree_with_feature_subset`, `traverse`,
  `aggregate_tree_importances`, `Node<F>`.
- **Consumers (non-test)**: crate re-export (`ferrolearn-tree/src/lib.rs`
  `pub use adaboost_regressor::{AdaBoostLoss, AdaBoostRegressor,
  FittedAdaBoostRegressor}`); pipeline adapters (`fit_pipeline` →
  `Box<dyn FittedPipelineEstimator>`; `predict_pipeline`). **There is NO PyO3
  binding for the regressor** — `ferrolearn-python` exposes only
  `RsAdaBoostClassifier` (`extras.rs`), not an `AdaBoostRegressor` shim; recorded
  honestly (a binding-registration gap, REQ-9). The pipeline `PipelineEstimator`
  impl is the production (non-test) consumer that satisfies R-DEFER-1 for the
  fitted/unfitted surface; these are grandfathered existing pub APIs (S5).

## Requirements

- REQ-1: **Param surface + defaults (R-DEV-2).** `n_estimators=50`,
  `learning_rate=1.0`, `loss='linear'`, base `max_depth=3`, `random_state=None`
  match `AdaBoostRegressor().get_params()` / `_validate_estimator` (`:1094`,
  `:1115`); `n_estimators>=1` and `learning_rate>0` validated (`:70`). Surface
  gaps: pluggable `estimator` (REQ-6), `estimator_errors_` attribute (REQ-6).
- REQ-2: **AdaBoostLoss normalization (R-DEV-1).** Linear/Square/Exponential map
  `e/error_max`, `(e/error_max)^2`, `1 - exp(-e/error_max)` exactly as
  `_boost` `:1185-1188`.
- REQ-3: **beta + estimator_weight formula (R-DEV-1).** `beta =
  error/(1-error)` (`:1203`) and `estimator_weight = learning_rate * log(1/beta)`
  (`:1206`). Deterministic given the per-round loss.
- REQ-4: **Weighted-median predict (R-DEV-1/3).** Sort per-tree predictions, take
  the cumulative `estimator_weights` CDF, return the first value with `cdf >= 0.5
  * total` (`_get_median_predict`, `:1215`). Deterministic given a fixed
  ensemble; intra-ferrolearn pinnable.
- REQ-5: **Sample-weight reweight WITH learning_rate (R-DEV-1, HEADLINE).**
  `sample_weight *= beta ** ((1 - error_vect) * learning_rate)` (`:1209-1211`).
  ferrolearn omits the `* learning_rate` factor — observable for
  `learning_rate != 1.0`, prediction-affecting through the round-to-round
  weight cascade.
- REQ-6: **Pluggable base `estimator` + `estimator_errors_` (R-DEV-2/3).**
  sklearn boosts any `fit`/`predict` learner and exposes `estimator_errors_`;
  ferrolearn hardwires a regression tree and stores no per-round error vector.
- REQ-7: **Weighted bootstrap = numpy `choice(replace=True, p=sample_weight)`
  / end-to-end prediction parity (RNG + resampling boundary).** sklearn
  `:1162-1167` draws a stochastic weighted bootstrap off numpy MT19937 each
  round; ferrolearn uses a deterministic systematic resample with NO RNG. Exact
  per-round tree and end-to-end prediction parity at a `random_state` are
  INFEASIBLE (documented boundary).
- REQ-8: **`feature_importances_` = normalized weighted sum (R-DEV-3).**
  `sum(weight * clf.fi) / sum(weight)` (`:300-308`); per-tree importance
  inherited from oracle-verified `decision_tree.rs`.
- REQ-9: **PyO3 binding for the regressor (R-DEFER-1).** sklearn exposes
  `AdaBoostRegressor` through `import sklearn.ensemble`; ferrolearn has a
  classifier shim (`RsAdaBoostClassifier`) but no regressor shim.
- REQ-10: **ferray substrate (R-SUBSTRATE).** Imports `ndarray`/`num-traits`,
  not `ferray-core`/`ferray::random`.

## Acceptance criteria

- AC-1: live `AdaBoostRegressor().get_params()` equals the REQ-1 defaults for the
  params ferrolearn exposes (`n_estimators=50`, `learning_rate=1.0`,
  `loss='linear'`); base `max_depth=3`; surface gaps enumerated.
- AC-2: for each loss, the normalized per-sample loss equals sklearn's
  `_boost` transform on a fixed `error_vect` (e.g. `e=[0,0.5,1.0]`,
  `error_max=1.0`: linear `[0,0.5,1]`, square `[0,0.25,1]`, exponential
  `[0, 1-e^-0.5, 1-e^-1]`).
- AC-3: on a fixed `(loss_vector, weight_vector)`, `beta = err/(1-err)` and
  `estimator_weight = lr*log(1/beta)` match the closed-form sklearn values to
  1e-12.
- AC-4: `weighted_median` over fixed `(value, weight)` pairs returns the first
  value whose cumulative weight reaches half the total (matches
  `_get_median_predict` CDF rule); covered by `test_weighted_median_*`.
- AC-5 (REQ-5 pin, R-CHAR-3): with `learning_rate=0.5` on a fixed dataset, the
  post-round normalized `sample_weight` must equal numpy
  `w * beta**((1-e)*0.5)` (renormalized), computed by a live sklearn-derived
  reference — ferrolearn currently produces `w * beta**(1-e)` and FAILS until the
  fixer adds `* learning_rate`.
- AC-6: `random_state` reproducibility — two `fit` calls with the same params
  produce identical `predict` (covered by
  `test_adaboost_regressor_reproducibility`; trivially holds since fit uses no
  RNG). End-to-end equality to sklearn at a seed is NOT asserted (boundary).
- AC-7: `feature_importances_` sums to 1 when any tree splits and equals the
  normalized weighted sum of per-tree importances.

## REQ status table

Binary (R-DEFER-2). `AdaBoostRegressor`/`FittedAdaBoostRegressor`/`AdaBoostLoss`
are boundary estimator types re-exported at the crate root and wired into the
pipeline adapters (the non-test production-consumer surface; existing pub APIs,
grandfathered per S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) /
`file:line` (sklearn 1.5.2). Verification green at baseline `7394a119`
(`cargo test -p ferrolearn-tree --lib adaboost_regressor`: 18 passed, 0 failed).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (param surface + defaults) | SHIPPED (gaps flagged) | `fn new` on `AdaBoostRegressor` (`n_estimators=50`, `learning_rate=F::one()`, `loss=AdaBoostLoss::Linear`, `max_depth=Some(3)`, `random_state=None`) + `fit` validation (`n_estimators==0`→`InvalidParameter`, `learning_rate<=0`→`InvalidParameter`) match `AdaBoostRegressor.__init__` (`_weight_boosting.py:1094`), `_validate_estimator` default `DecisionTreeRegressor(max_depth=3)` (`:1115`), and `_parameter_constraints` (`:70`). Consumer: crate re-export (`lib.rs`) + pipeline adapter. Tests: `test_adaboost_regressor_default`, `test_adaboost_regressor_zero_estimators`, `test_adaboost_regressor_invalid_learning_rate`. Verification: `cd /tmp && python3 -c "from sklearn.ensemble import AdaBoostRegressor; print(AdaBoostRegressor().get_params())"` → `learning_rate=1.0, loss='linear', n_estimators=50, random_state=None`. Surface gaps (pluggable `estimator`, `estimator_errors_`) → blocker #705. |
| REQ-2 (AdaBoostLoss normalization) | SHIPPED | `losses` map in `fit` (`AdaBoostLoss::Linear` → `e/max_error`, `Square` → `(e/max_error)^2`, `Exponential` → `1 - (-e/max_error).exp()`) mirrors `_boost` `:1185-1188` (`masked_error_vector **= 2` / `1.0 - np.exp(-masked_error_vector)`). Consumer: pipeline `fit_pipeline`; crate re-export of `AdaBoostLoss`. Tests: `test_adaboost_regressor_square_loss`, `test_adaboost_regressor_exponential_loss`. |
| REQ-3 (beta + estimator_weight) | SHIPPED | `let beta = avg_loss / (F::one()-avg_loss).max(eps)` mirrors `_weight_boosting.py:1203`; `let est_weight = (F::one()/beta.max(eps)).ln() * self.learning_rate` mirrors `:1206`. Deterministic given the per-round loss. Consumer: `estimator_weights()` accessor + pipeline `predict_pipeline` (weights drive the median). Tests: `test_adaboost_regressor_simple`, `test_adaboost_regressor_two_features`. |
| REQ-4 (weighted-median predict) | SHIPPED | `fn weighted_median` (sort by value, accumulate weight, return first value with `cumulative >= total/2`) mirrors `_get_median_predict` (`:1215`, sort `:1220`, `stable_cumsum` CDF `:1223`, `weight_cdf >= 0.5*total` argmax `:1224-1225`); `FittedAdaBoostRegressor::predict` builds `(value, weight)` per tree and calls it. Consumer: pipeline `predict_pipeline`; `score`. Tests: `test_weighted_median_basic`, `test_weighted_median_unequal_weights`, `test_weighted_median_single`, `test_weighted_median_empty`, `test_adaboost_regressor_perfect_fit`. Deterministic given a fixed ensemble. |
| REQ-5 (reweight WITH learning_rate) | NOT-STARTED | open prereq blocker #703. The reweight in `fit` is `weights[i] = weights[i] * beta.powf(F::one() - losses[i])` — exponent is `(1 - loss_i)`, MISSING the `* learning_rate` factor sklearn applies at `_weight_boosting.py:1209-1211` (`beta ** ((1.0 - masked_error_vector) * self.learning_rate)`). Identical only when `learning_rate==1.0`; for any other value the round-to-round reweighted distribution — and thus `beta`/`estimator_weight`/prediction — diverges. HEADLINE single-file fixer target. |
| REQ-6 (pluggable estimator + estimator_errors_) | NOT-STARTED | open prereq blocker #705. ferrolearn hardwires a regression tree (`build_regression_tree_with_feature_subset`) instead of sklearn's generic `estimator` (`:1115`), and stores no `estimator_errors_` vector (sklearn `estimator_errors_`, `:154`/`:177`). |
| REQ-7 (weighted bootstrap / end-to-end parity) | NOT-STARTED | open prereq blocker #706 (documented RNG + resampling boundary). `fn resample_weighted` is a DETERMINISTIC systematic resample (cumulative-weight CDF walked at fixed `step=total/n` from `step/2`, no RNG) and `random_state` is stored-but-unused; sklearn `:1162-1167` draws a stochastic numpy weighted bootstrap (`random_state.choice(replace=True, p=sample_weight)`) every round. Different sampling procedure off an incompatible (numpy MT19937) stream → exact per-round tree and end-to-end prediction parity at a seed INFEASIBLE. ferrolearn fit is deterministic/reproducible (`test_adaboost_regressor_reproducibility`), which is the pinnable contract. |
| REQ-8 (feature_importances_ normalized weighted sum) | SHIPPED | `fit` calls `aggregate_tree_importances(&estimators, None, Some(&estimator_weights), n_features)` (`decision_tree.rs`, oracle-verified) — weighted sum of per-tree importances normalized to sum 1, mirroring `feature_importances_` `:300-308` (`sum(weight*clf.fi)/norm`); exposed via `HasFeatureImportances::feature_importances`. Consumer: `HasFeatureImportances` trait impl + pipeline. Tests: per-tree importance correctness inherited from `decision_tree.rs` (`.design/tree/decision_tree.md`); aggregation exercised by `test_adaboost_regressor_simple`. |
| REQ-9 (PyO3 regressor binding) | NOT-STARTED | open prereq blocker #707. `ferrolearn-python` registers `RsAdaBoostClassifier` (`extras.rs:616`) but no `AdaBoostRegressor` shim; `import ferrolearn` cannot construct the regressor sklearn exposes at `sklearn.ensemble.AdaBoostRegressor`. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #708. `adaboost_regressor.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`/`ferray::random` (R-SUBSTRATE). |

## Architecture

`AdaBoostRegressor<F>` is the unfitted boundary type (public fields
`n_estimators`/`learning_rate`/`max_depth`/`random_state`/`loss` + `with_*`
builders + `Default`). `fit` validates shapes
(`ShapeMismatch`/`InsufficientSamples`/`InvalidParameter`), initializes uniform
`weights = 1/n_samples`, and runs the boost loop: (1) `resample_weighted`
produces deterministic systematic-resample indices, (2)
`build_regression_tree_with_feature_subset` (from `decision_tree.rs`) fits a tree
on those indices over all features, (3) per-sample absolute errors and
`max_error` are computed with an `eps` perfect-fit guard (keep tree, weight 1.0,
break), (4) the `AdaBoostLoss` normalization produces `losses`, (5) the
weighted-average loss gates a `>= 0.5` discard/stop, (6) `beta` and
`estimator_weight` are formed, (7) **the reweight** (REQ-5 divergence) multiplies
each weight by `beta^(1 - loss_i)` — sklearn uses `beta^((1 - loss_i) *
learning_rate)` — then normalizes. After the loop,
`aggregate_tree_importances` builds the weighted-normalized
`feature_importances`.

`FittedAdaBoostRegressor<F>` stores `estimators: Vec<Vec<Node<F>>>`,
`estimator_weights`, `n_features`, `feature_importances`. `predict` traverses
each tree (`decision_tree::traverse`) to a leaf `value`, pairs it with the
tree's `estimator_weight`, and returns `weighted_median` of those pairs per row —
the sklearn weighted-median rule. `score` is R² via the crate `r2_score`.
Pipeline integration is provided by `PipelineEstimator`/`FittedPipelineEstimator`.

**Invariants held:** uniform weight init; perfect-fit and `>=0.5` early stops;
weighted-median CDF-crossing rule; feature-count guard on `predict`. **Invariant
NOT held vs sklearn:** the reweight exponent omits `* learning_rate` (REQ-5).
**Boundary:** the per-round training subset is a deterministic systematic
resample, not sklearn's numpy weighted bootstrap — end-to-end parity at a seed is
out of reach (REQ-7), but ferrolearn's fit is RNG-free and thus exactly
reproducible.

## Verification

Library crate (green at baseline `7394a119`):
```
cargo test -p ferrolearn-tree --lib adaboost_regressor   # 18 passed; 0 failed
cargo clippy -p ferrolearn-tree --all-targets -- -D warnings
cargo fmt --all --check
```
Live sklearn oracle (installed 1.5.2, run from `/tmp`):
```
# REQ-1 defaults
python3 -c "from sklearn.ensemble import AdaBoostRegressor; print(AdaBoostRegressor().get_params())"
# REQ-5 headline divergence — the reweight exponent (lr=0.5):
#   sklearn: w *= beta**((1-e)*0.5);  ferrolearn: w *= beta**(1-e)
python3 -c "import numpy as np; beta=0.3; e=np.array([0.0,0.5,1.0]); lr=0.5; print('sklearn exp', beta**((1-e)*lr)); print('ferro  exp', beta**(1-e))"
# REQ-2 loss normalization (error_max=1.0)
python3 -c "import numpy as np; e=np.array([0.0,0.5,1.0]); print('linear',e); print('square',e**2); print('exp',1-np.exp(-e))"
```
The NOT-STARTED REQs (5, 6, 7, 9, 10) have no green verification by construction —
each carries an open prereq blocker. REQ-1/2/3/4/8 are verified by the in-crate
`#[test]`s named in the status table plus the live `get_params()` /
closed-form-loss comparisons (deterministic). A characterization pin for REQ-5
(R-CHAR-3, AC-5) belongs in `ferrolearn-tree/tests/divergence_adaboost_regressor.rs`:
fit with `learning_rate=0.5`, assert the post-round normalized `sample_weight`
equals the numpy `w*beta**((1-e)*0.5)` reference — FAILS until the fixer adds the
`* learning_rate` factor at the reweight in `fit`.

## Blockers to open

- #703 — REQ-5 (HEADLINE): sample-weight reweight in `fit` uses exponent
  `(1 - loss_i)`, missing the `* learning_rate` factor sklearn applies
  (`_weight_boosting.py:1209-1211`, `beta ** ((1.0 - error_vect) *
  learning_rate)`). Prediction-affecting for `learning_rate != 1.0`. Single-file
  fixer target.
- #705 — REQ-6: no pluggable base `estimator` (regression tree hardwired,
  `:1115`) and no `estimator_errors_` fitted attribute (sklearn `:154`/`:177`).
- #706 — REQ-7: weighted-bootstrap / end-to-end prediction parity — ferrolearn's
  `resample_weighted` is a deterministic systematic resample with `random_state`
  unused; sklearn draws a numpy weighted bootstrap (`random_state.choice(...,
  replace=True, p=sample_weight)`, `:1162-1167`) off MT19937. Documented RNG +
  resampling boundary (like random-forest bootstrap / SGD shuffle); exact
  parity INFEASIBLE. ferrolearn fit remains deterministic/reproducible.
- #707 — REQ-9: no PyO3 `AdaBoostRegressor` shim in `ferrolearn-python`
  (only `RsAdaBoostClassifier`, `extras.rs:616`).
- #708 — REQ-10: migrate `adaboost_regressor.rs` off `ndarray`/`num-traits` to
  the ferray substrate (R-SUBSTRATE).
