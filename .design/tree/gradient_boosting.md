# Gradient Boosting (GradientBoostingClassifier / GradientBoostingRegressor)

<!--
tier: 3-component
status: draft
baseline-commit: 15cb050d
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/ensemble/_gb.py   # BaseGradientBoosting (:354); _parameter_constraints (:357-367, criterion StrOptions{friedman_mse,squared_error} :361, validation_fraction :365, n_iter_no_change :366); __init__ (:373); _init_raw_predictions (:87); _update_terminal_regions (:129) — apply :184, HalfSquaredError no-op :186/:155-157, HalfBinomialLoss Newton sum(w(y-p))/sum(w p(1-p)) :191-206, HalfMultinomialLoss (K-1)/K * num/den :208-225, generic else fit_intercept_only :241-247, leaf replace tree.value[leaf,0,0]=update :250-259, raw += lr*tree.value.take(terminal_regions) :262-264; set_huber_delta weighted_percentile(abserr,100*quantile) :267-272; _init_state init_ DummyClassifier(prior)/DummyRegressor(quantile=0.5 for AbsoluteError|Huber, mean for squared) :535-547; _fit_stage :428 — set_huber_delta :443-449, neg_gradient = -loss.gradient :454-458, DecisionTreeRegressor(criterion=self.criterion friedman_mse, splitter=best, max_depth, min_samples_*) :471-483, subsample mask :485-487, tree.fit(neg_g) :490-492, _update_terminal_regions :496-507; fit :616; _fit_stages :808 (early stopping n_iter_no_change/validation_fraction/tol :840-845/:924-945, subsample mask :855-862); GradientBoostingClassifier (:1117), __init__ defaults loss='log_loss'/learning_rate=0.1/n_estimators=100/subsample=1.0/criterion='friedman_mse'/max_depth=3 (:1451-1495); GradientBoostingRegressor (:1723), __init__ defaults loss='squared_error'/.../alpha=0.9 (:2051-2097)
  - sklearn/_loss/loss.py   # HalfSquaredError (:516); AbsoluteError.fit_intercept_only weighted median (:565-574); HuberLoss.fit_intercept_only median + weighted-mean clipped term (:694-710); HalfBinomialLoss (:886); HalfMultinomialLoss.fit_intercept_only log-prior (:955/:1010); _weighted_percentile import (:25)
ferrolearn-module: ferrolearn-tree/src/gradient_boosting.rs
parity-ops: GradientBoostingClassifier, GradientBoostingRegressor
crosslink-issue: 733
-->

## Summary

`ferrolearn-tree/src/gradient_boosting.rs` mirrors scikit-learn's
`sklearn.ensemble.GradientBoostingRegressor` (`_gb.py:1723`) and
`GradientBoostingClassifier` (`_gb.py:1117`) — the forward stage-wise additive
model that, each round, fits a regression tree to the loss's negative gradient
(pseudo-residuals) and adds `learning_rate * tree.value` to the running raw
predictions (`_fit_stage`, `:428`).

**HEADLINE divergence (R-DEV-1) — ferrolearn omits the terminal-region
leaf-value update.** sklearn does NOT add the regression tree's mean-residual
leaves directly. After fitting `DecisionTreeRegressor(criterion='friedman_mse')`
to the negative gradient, sklearn calls `_update_terminal_regions`
(`:496-507`, `:129`) which **replaces** each leaf's value with the
**loss-optimal** line-search value (`argmin_x loss(y, raw_old + x*tree.value)`,
`:149-151`) — the weighted median for `AbsoluteError` (`loss.py:565-574`), the
median+clipped-mean for `HuberLoss` (`loss.py:694-710`), and a single
Newton-Raphson step `Σw(y-p)/Σw·p(1-p)` for `HalfBinomialLoss` (`:191-206`) /
`(K-1)/K · Σw·neg_g / Σw·p(1-p)` for `HalfMultinomialLoss` (`:208-225`) — and
only then does `raw_prediction[:, k] += learning_rate * tree.value.take(...)`
(`:262-264`). ferrolearn's two fit loops instead add the regression tree's raw
**mean-residual** leaf directly (`f_vals[i] += lr * leaf_value`, GBR fit loop /
GBC `fit_binary` / `fit_multiclass`), with **no `_update_terminal_regions`
analog at all**.

**Consequence (the REQ split).** The mean residual IS the L2-optimal leaf
(`HalfSquaredError` update is the identity, `:155-157`/`:186`), so ferrolearn
matches sklearn **exactly** for `GradientBoostingRegressor(loss='squared_error')`
— **live-verified end-to-end** (see REQ-4 / Verification).

**SHIPPED (the terminal-region update).** ferrolearn now performs the
`_update_terminal_regions` line-search before applying `lr*leaf` in all three fit
loops: `group_samples_by_leaf` buckets the in-bag samples by leaf, then a per-loss
helper replaces the leaf — `lad_leaf_value` (median), `huber_leaf_value`
(median + clipped-mean, with `huber_stage_delta`), `binary_newton_leaf` and
`multiclass_newton_leaf` (Newton step via `safe_divide`). `LeastSquares` keeps the
identity (mean-residual leaf untouched) so the REQ-4 linchpin stays exact. This is
the ONE coherent builder-scale change — REQ-5/REQ-6/REQ-7 below, now SHIPPED.

ferrolearn ships the unfitted `GradientBoostingRegressor<F>` /
`GradientBoostingClassifier<F>` + their `Fitted*` types, the `RegressionLoss`
(`LeastSquares`/`Lad`/`Huber`) and `ClassificationLoss` (`LogLoss`) enums,
`fn new` + `with_*` builders + `Default`, `Fit`/`Predict`, `score`,
`predict_proba`/`predict_log_proba`/`decision_function` (classifier),
`HasClasses`/`HasFeatureImportances`, and `PipelineEstimator`/
`FittedPipelineEstimator` adapters. The per-round tree build is delegated to the
oracle-verified `build_regression_tree_with_feature_subset` from
`decision_tree.rs` (`.design/tree/decision_tree.md`).

**Consumers (non-test, R-DEFER-1).** Crate re-exports (`ferrolearn-tree/src/lib.rs`:
`pub use gradient_boosting::{ClassificationLoss, FittedGradientBoostingClassifier,
FittedGradientBoostingRegressor, GradientBoostingClassifier,
GradientBoostingRegressor, RegressionLoss}`); the PyO3 bindings
`RsGradientBoostingRegressor` (`ferrolearn-python/src/extras.rs:336`,
registered `lib.rs:36`) and `RsGradientBoostingClassifier`
(`extras.rs:670`, registered `lib.rs:57`); and the pipeline adapters. These
boundary estimator types are existing pub APIs, grandfathered per S5/R-DEFER-1.

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `get_params()`, sklearn 1.5.2)

`GradientBoostingRegressor().get_params()` (relevant keys) →
`loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0,
criterion='friedman_mse', max_depth=3, min_samples_split=2, min_samples_leaf=1,
alpha=0.9` (`__init__`, `:2051-2097`).
`GradientBoostingClassifier().get_params()` → `loss='log_loss',
learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
max_depth=3, min_samples_split=2, min_samples_leaf=1` (`:1451-1495`).
Constraints (`:357-367`): `criterion in {'friedman_mse', 'squared_error'}`
(`:361`).

**Defaults ferrolearn matches** (`fn new`, both estimators): `n_estimators=100`,
`learning_rate=0.1`, `max_depth=Some(3)`, `min_samples_split=2`,
`min_samples_leaf=1`, `subsample=1.0`; GBR `loss=LeastSquares` /
`huber_alpha=0.9`; GBC `loss=LogLoss`. Validation in `fit` rejects
`n_samples==0` (`InsufficientSamples`), `y.len() != n_samples`
(`ShapeMismatch`), `n_estimators==0` / `learning_rate<=0` /
`subsample not in (0,1]` (`InvalidParameter`), and GBC `n_classes<2`
(`InvalidParameter`).

### init / prior (`_init_state`, `:535-547`; `_init_raw_predictions`, `:87`)

sklearn: `init_` is `DummyClassifier(strategy="prior")` (classifier),
`DummyRegressor(strategy="quantile", quantile=0.5)` = median (`AbsoluteError`/
`HuberLoss`), else `DummyRegressor(strategy="mean")`. ferrolearn (GBR fit):
`init = mean(y)` for `LeastSquares`, `median_f(y)` for `Lad`/`Huber` — **matches**
(`_init_state:542-547`). GBC: binary `init = log(p/(1-p))` (clipped log-odds of
the positive-class prior), multiclass `init[k] = log(p_k)` (log-prior). sklearn's
binary prior raw-prediction is the log-odds via `HalfBinomialLoss.link`; the
multiclass raw-prediction is `log(p_k)` (softmax-invariant under a constant
shift). The init priors **match** in form (REQ-3).

### per-round stage (`_fit_stage`, `:428`)

For each round (and each of `K` trees per round in the multiclass case, `:466`):
1. (Huber only) `set_huber_delta` = `_weighted_percentile(|y-raw|, 100*quantile)`
   (`:443-449`, `:267-272`).
2. `neg_gradient = -loss.gradient(y, raw)` (`:454-458`): for squared error
   `y - raw`; for absolute error `sign(y - raw)`; for Huber the clipped residual;
   for binomial `y - expit(raw)`; for multinomial `y_onehot_k - softmax_k`.
3. Fit `DecisionTreeRegressor(criterion=self.criterion='friedman_mse',
   splitter='best', max_depth, min_samples_split, min_samples_leaf,
   min_weight_fraction_leaf, min_impurity_decrease, max_features, max_leaf_nodes,
   ccp_alpha)` to `neg_gradient` (`:471-492`).
4. **`_update_terminal_regions` (`:496-507`, `:129`) — REPLACE each leaf with the
   loss-optimal value, THEN add to raw:**
   - `HalfSquaredError`: no-op; leaves stay the mean residual (`:155-157`/`:186`).
   - `AbsoluteError` / `HuberLoss` (generic `else`, `:241-247`):
     `update = loss.fit_intercept_only(y_true = y[idx] - raw[idx,k], sw)`. For
     `AbsoluteError` this is the **weighted median** of the leaf's residuals
     (`loss.py:565-574`); for `HuberLoss`, `median + average(sign(d)·min(δ,|d|))`
     over the leaf (`loss.py:694-710`).
   - `HalfBinomialLoss` (`:191-206`): `update = Σw·neg_g / Σw·p(1-p)` (Newton
     step, `p = y - neg_g = expit(raw)`).
   - `HalfMultinomialLoss` (`:208-225`): `update = (K-1)/K · Σw·neg_g /
     Σw·p(1-p)`.
   - Then `tree.value[leaf,0,0] = update` (`:259`) and
     `raw[:, k] += learning_rate * tree.value.take(terminal_regions)`
     (`:262-264`).

### criterion (friedman_mse vs MSE)

sklearn's GB trees use `criterion='friedman_mse'` (`:472`, `:361`); ferrolearn's
`build_regression_tree_with_feature_subset` uses MSE. For **split selection** the
Friedman improvement criterion is a strictly increasing function of the MSE
reduction of a single binary split, so both choose the **same split/threshold**
→ identical tree **structure** at `subsample=1.0`. Only the node *impurity*
reported and hence `feature_importances_` may differ in magnitude (REQ-8).

### subsample / RNG (`_fit_stages` mask, `:855-862`; `_fit_stage` mask, `:485-487`)

`subsample=1.0` (the default) → fully **deterministic**: no RNG draw is taken in
either fit loop (ferrolearn's `subsample_size == n_samples` short-circuits the
sample to `0..n_samples`). `subsample<1.0` draws a bootstrap mask — sklearn via
its numpy-MT `random_state`, ferrolearn via `StdRng` + `rand::seq::index::sample`
— an **RNG-boundary** divergence (the numpy-MT-vs-StdRng precedent, REQ-9).

### missing surface (`__init__`, `:1451-1495`/`:2051-2097`)

ferrolearn exposes neither: `n_iter_no_change`/`validation_fraction`/`tol`
(early stopping, `:840-945`), an explicit `criterion` param (`:472`), `ccp_alpha`
(`:482`), `max_features` (`:479`), `min_impurity_decrease` (`:478`),
`max_leaf_nodes`, `min_weight_fraction_leaf` (`:477`), the `init` estimator
(`:538`), nor `staged_predict`/`staged_decision_function`/`apply`. GBR's `alpha`
(quantile loss) and `loss='quantile'` are absent (REQ-10).

## ferrolearn (what exists)

- **Unfitted**: `pub struct GradientBoostingRegressor<F>` /
  `GradientBoostingClassifier<F>` (public fields `n_estimators`,
  `learning_rate: f64`, `max_depth`, `min_samples_split`, `min_samples_leaf`,
  `subsample`, `loss`, GBR `huber_alpha`, `random_state`); `fn new`; `with_*`
  builders; `Default`.
- **Enums**: `pub enum RegressionLoss { LeastSquares, Lad, Huber }`,
  `pub enum ClassificationLoss { LogLoss }`.
- **Fitted**: `pub struct FittedGradientBoostingRegressor<F>` (`init: F`,
  `learning_rate`, `trees: Vec<Vec<Node<F>>>`, `n_features`,
  `feature_importances`); `pub struct FittedGradientBoostingClassifier<F>`
  (`classes: Vec<usize>`, `init: Vec<F>`, `learning_rate`, `trees:
  Vec<Vec<Vec<Node<F>>>>`, `n_features`, `feature_importances`).
- **Traits**: GBR `Fit<Array2<F>, Array1<F>>` + `Predict`; GBC `Fit<Array2<F>,
  Array1<usize>>` + `Predict`; `HasFeatureImportances<F>` (both); `HasClasses`
  (GBC); `PipelineEstimator`/`FittedPipelineEstimator` (both, GBC via
  `FittedGbcPipelineAdapter`).
- **Methods**: GBR `fn init`, `fn learning_rate`, `fn trees`, `fn n_features`,
  `fn score` (R²). GBC `fn init`, `fn learning_rate`, `fn trees`, `fn n_features`,
  `fn score` (mean accuracy), `fn predict_proba`, `fn predict_log_proba`,
  `fn decision_function`; private `fn fit_binary`, `fn fit_multiclass`.
- **Internal helpers**: `fn sigmoid`, `fn softmax_matrix`, `fn median_f`,
  `fn quantile_f`, `fn compute_regression_residuals`.
- **Build delegation** (from `decision_tree.rs`, oracle-verified):
  `build_regression_tree_with_feature_subset`, `traverse`,
  `compute_feature_importances`, `Node<F>`.
- **Consumers (non-test)**: crate re-export (`lib.rs`); PyO3 bindings
  `RsGradientBoostingRegressor` (`extras.rs:336`, `lib.rs:36`) and
  `RsGradientBoostingClassifier` (`extras.rs:670`, `lib.rs:57`) — each
  constructs the estimator with `n_estimators`/`learning_rate`/`max_depth`/
  `random_state` and calls `fit`/`predict`; the pipeline adapters.

## Requirements

- REQ-1: **Param surface + numeric defaults (R-DEV-2).** GBR `n_estimators=100`,
  `learning_rate=0.1`, `max_depth=Some(3)`, `min_samples_split=2`,
  `min_samples_leaf=1`, `subsample=1.0`, `loss=LeastSquares`, `huber_alpha=0.9`;
  GBC same + `loss=LogLoss` — match `__init__` (`:2051-2097`/`:1451-1495`).
  `fit` validates shapes/empty/`n_estimators`/`learning_rate`/`subsample`/
  `n_classes`. Surface gaps (`criterion`, `ccp_alpha`, `max_features`, early
  stopping, `init`, `alpha`/`loss='quantile'`, `staged_*`) → REQ-8/REQ-10.
- REQ-2: **Negative-gradient pseudo-residuals per loss (R-DEV-1).** GBR
  `compute_regression_residuals` — `y - f` (`LeastSquares`), `sign(y-f)` (`Lad`),
  clipped residual (`Huber`); GBC binary `y - sigmoid(f)`, multiclass
  `y_onehot_k - softmax_k`. Match `loss.gradient` (`_fit_stage:454-458`).
- REQ-3: **init prior (R-DEV-1).** GBR `init = mean(y)` (`LeastSquares`) /
  `median_f(y)` (`Lad`/`Huber`) matches `_init_state:542-547`; GBC binary
  log-odds `log(p/(1-p))`, multiclass log-prior `log(p_k)` match the
  `DummyClassifier(prior)` raw-prediction form.
- REQ-4: **`GradientBoostingRegressor(loss='squared_error')` end-to-end parity
  (R-DEV-1, the linchpin).** Because the L2 terminal-region update is the
  identity (`:155-157`), ferrolearn's mean-residual leaf == sklearn's optimal
  leaf, so the whole tree+boost framework matches sklearn array-by-array.
  Deterministic at `subsample=1.0`. Live-verified.
- REQ-5: **`AbsoluteError`/LAD terminal-region median update (R-DEV-1,
  HEADLINE).** sklearn replaces each leaf with the **weighted median** of the
  leaf's residuals (`_update_terminal_regions` generic branch `:241-247` +
  `AbsoluteError.fit_intercept_only` `loss.py:565-574`). ferrolearn adds the
  mean-residual leaf unchanged → diverges from round 1.
- REQ-6: **`Huber` terminal-region update (R-DEV-1).** sklearn leaf =
  `median + average(sign(d)·min(δ,|d|))` over the leaf (`loss.py:694-710`);
  ferrolearn adds the clipped-residual mean leaf unchanged → diverges.
- REQ-7: **`LogLoss` Newton terminal-region update — binary + multiclass
  (R-DEV-1, HEADLINE).** sklearn leaf = `Σw(y-p)/Σw·p(1-p)` (binomial,
  `:191-206`) / `(K-1)/K · Σw·neg_g/Σw·p(1-p)` (multinomial, `:208-225`);
  ferrolearn adds the mean-residual leaf unchanged → diverges from round 1.
- REQ-8: **friedman_mse criterion + feature_importances (R-DEV-1/3).** GB trees
  use `criterion='friedman_mse'` (`:472`); ferrolearn uses MSE. Split selection
  is identical (Friedman improvement monotone in MSE reduction for one split) so
  tree structure matches, but node impurity / `feature_importances_` magnitudes
  may differ.
- REQ-9: **subsample bootstrap RNG (R-DEV-1).** `subsample<1.0` draws a mask;
  ferrolearn's `StdRng` differs from sklearn's numpy-MT `random_state`
  (RNG-boundary). At `subsample=1.0` (default) both are deterministic and agree.
- REQ-10: **Missing param surface + early stopping + staged API (R-DEV-2/3).**
  `n_iter_no_change`/`validation_fraction`/`tol` (`:840-945`), explicit
  `criterion`, `ccp_alpha`, `max_features`, `min_impurity_decrease`,
  `max_leaf_nodes`, `min_weight_fraction_leaf`, the `init` estimator,
  `staged_predict`/`staged_decision_function`, GBR `alpha`/`loss='quantile'`.
- REQ-11: **PyO3 binding fidelity (R-DEFER-1).** `RsGradientBoostingRegressor`
  (`extras.rs:336`) / `RsGradientBoostingClassifier` (`extras.rs:670`) expose
  only `n_estimators`/`learning_rate`/`max_depth`/`random_state`; no `loss`,
  `subsample`, `min_samples_*`, `predict_proba`/`decision_function`,
  `feature_importances_`, `classes_`.
- REQ-12: **ferray substrate (R-SUBSTRATE).** `gradient_boosting.rs` imports
  `ndarray`/`num-traits`/`rand`, not `ferray-core`/`ferray::random`.

## Acceptance criteria

- AC-1 (REQ-1): live `GradientBoostingRegressor().get_params()` /
  `GradientBoostingClassifier().get_params()` equal the REQ-1 defaults for the
  exposed params; covered by `test_gbr_default_trait`, `test_gbc_default_trait`,
  and the validation tests (`test_gbr_zero_estimators`,
  `test_gbr_invalid_learning_rate`, `test_gbr_invalid_subsample`,
  `test_gbc_single_class`, shape/empty).
- AC-2 (REQ-2, R-CHAR-3): `compute_regression_residuals` equals
  `-loss.gradient(y, raw)` on fixed `(y, f)` — `y-f` / `sign(y-f)` / clipped
  (covered by `test_regression_residuals_*`); GBC binary residual `y - σ(f)`.
- AC-3 (REQ-3): GBR `fitted.init()` equals `np.mean(y)` (L2) /
  `np.median(y)` (LAD/Huber); GBC binary `init()[0] == log(p/(1-p))`.
- AC-4 (REQ-4, R-CHAR-3, linchpin): on a fixed `(X, y)`,
  `GradientBoostingRegressor(loss=LeastSquares, n_estimators, learning_rate,
  max_depth).fit(X,y).predict(X)` equals live
  `GradientBoostingRegressor(loss='squared_error', subsample=1.0)` to ~1e-8.
  **Live-verified GREEN** (see Verification) on two datasets.
- AC-5 (REQ-5 pin, R-CHAR-3): on a skewed-within-leaf `(X, y)`,
  `GradientBoostingRegressor(loss=Lad)` `predict` must equal live
  `GradientBoostingRegressor(loss='absolute_error')`; ferrolearn's mean-leaf
  output diverges → FAILS until the weighted-median terminal-region update lands.
- AC-6 (REQ-6 pin, R-CHAR-3): `GradientBoostingRegressor(loss=Huber)` `predict`
  must equal live `loss='huber'`; FAILS until the median+clipped-mean leaf update
  lands.
- AC-7 (REQ-7 pin, R-CHAR-3): `GradientBoostingClassifier()` `decision_function`
  (binary and `K>2`) must equal live sklearn; ferrolearn's mean-leaf raw scores
  diverge → FAILS until the Newton terminal-region update lands.
- AC-8 (REQ-9): two `fit` calls with identical params produce identical
  `predict` (`test_gbr_reproducibility`, `test_gbc_reproducibility`); the
  `subsample<1.0` divergence vs numpy-MT is an RNG-boundary (not pinnable
  array-by-array).
- AC-9 (REQ-11, R-CHAR-3): `import ferrolearn`'s GB binding exposes `loss`,
  `subsample`, `predict_proba`, `feature_importances_`, `classes_` — currently
  absent → pytest divergence FAILS until the binding is extended.

## REQ status table

Binary (R-DEFER-2). `GradientBoostingRegressor`/`GradientBoostingClassifier`
(+ their `Fitted*` and the `RegressionLoss`/`ClassificationLoss` enums) are
boundary estimator types re-exported at the crate root, exposed through the
`RsGradientBoosting*` PyO3 bindings, and wired into the pipeline adapters (the
non-test production-consumer surface; existing pub APIs, grandfathered per
S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2). Verification green at baseline `15cb050d`
(`cargo test -p ferrolearn-tree --lib gradient_boosting::`).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (param surface + numeric defaults) | SHIPPED (gaps flagged) | `fn new` on both estimators (`n_estimators=100`, `learning_rate=0.1`, `max_depth=Some(3)`, `min_samples_split=2`, `min_samples_leaf=1`, `subsample=1.0`; GBR `loss=LeastSquares`/`huber_alpha=0.9`, GBC `loss=LogLoss`) + `fit` validation (`InsufficientSamples`/`ShapeMismatch`/`InvalidParameter` for empty/shape/`n_estimators==0`/`learning_rate<=0`/`subsample∉(0,1]`/GBC `n_classes<2`) match `__init__` (`_gb.py:2051-2097`/`:1451-1495`). Consumer: crate re-export (`lib.rs`) + `RsGradientBoostingRegressor`/`RsGradientBoostingClassifier` (`extras.rs:336`/`:670`) + pipeline adapters. Tests: `test_gbr_default_trait`, `test_gbc_default_trait`, `test_gbr_zero_estimators`, `test_gbr_invalid_learning_rate`, `test_gbr_invalid_subsample`, `test_gbc_zero_estimators`, `test_gbc_invalid_learning_rate`, `test_gbc_single_class`, `test_g*_shape_mismatch_fit`, `test_g*_empty_data`. Verification: live `get_params()` (see Verification). Gaps (`criterion`/`ccp_alpha`/`max_features`/early-stop/`init`/`alpha`) → REQ-8/REQ-10. |
| REQ-2 (negative-gradient pseudo-residuals per loss) | SHIPPED | `compute_regression_residuals` returns `y-f` (`LeastSquares`), `sign(y-f)` (`Lad`), clipped residual via `quantile_f` δ (`Huber`); GBC `fit_binary` residual `yi - sigmoid(f_vals[i])`, `fit_multiclass` `yi_k - probs[k][i]` (softmax). Mirrors `-loss.gradient` (`_fit_stage:454-458`) — squared `y-raw`, absolute `sign`, binomial `y-expit`, multinomial `y_onehot-softmax`. Consumer: feeds the per-round tree build in all three fit loops, reached via crate re-export + PyO3 `fit`. Tests: `test_regression_residuals_least_squares`, `test_regression_residuals_lad`, `test_regression_residuals_huber`. NOTE: residuals are the GRADIENT only; the leaf VALUE that consumes them is the L2 mean (REQ-5/6/7 divergence). |
| REQ-3 (init prior) | SHIPPED | GBR `init = sum(y)/n` (`LeastSquares`) / `median_f(y)` (`Lad`/`Huber`) in `fit` mirrors `_init_state:542-547` (`DummyRegressor(strategy='mean')` / `quantile=0.5`); GBC `fit_binary` `init_val = (p_clipped/(1-p_clipped)).ln()` (log-odds prior), `fit_multiclass` `init[k] = (cnt/n).max(eps).ln()` (log-prior) mirror the `DummyClassifier(strategy='prior')` raw-prediction. Consumer: stored as `Fitted*.init`, consumed by `predict`/`predict_proba`/`decision_function` + PyO3 `predict`. Tests: covered indirectly by `test_gbr_simple_least_squares` (init=mean=3.0), `test_gbc_binary_simple`; `fitted.init()` accessor exercised. |
| REQ-4 (GBR squared_error end-to-end parity, LINCHPIN) | SHIPPED | The L2 `_update_terminal_regions` is the identity (`_gb.py:155-157`/`:186`), so ferrolearn's mean-residual leaf (`f_vals[i] += lr*value` in the GBR `fit` loop) equals sklearn's optimal leaf, validating the entire init→residual→tree→shrinkage→predict framework array-by-array. Deterministic at `subsample=1.0`. Consumer: `FittedGradientBoostingRegressor::predict` + `RsGradientBoostingRegressor.predict` (`extras.rs:336`). LIVE-VERIFIED (R-CHAR-3): on `X=[[1..8]], y=[1,1,1,1,5,5,5,5]`, `n_estimators=5, lr=0.1, max_depth=1` ferrolearn `predict` == sklearn `[2.18098×4, 3.81902×4]` exactly; on a generic 30×3 random dataset (`n_estimators=20, max_depth=3`) ferrolearn `[-0.5098813394742213, 0.06836662653713442, -0.7685960570544793, -0.7685960570544793, -1.0872496275174468]` == sklearn `[-0.50988134, 0.06836663, -0.76859606, -0.76859606, -1.08724963]` to printed precision. Tests: `test_gbr_simple_least_squares` (in-crate, directional); the array-by-array oracle pin belongs in `ferrolearn-tree/tests/divergence_gradient_boosting.rs` (AC-4). |
| REQ-5 (LAD terminal-region median update, HEADLINE) | SHIPPED | The GBR `fit` loop now runs the terminal-region line-search before `f_vals[i] += lr*value`: `group_samples_by_leaf` buckets the in-bag samples by leaf, then for `RegressionLoss::Lad` `lad_leaf_value` replaces each leaf with `median(y[idx]-f_vals[idx])` (`AbsoluteError.fit_intercept_only`, `_loss/loss.py:565-574` → `_update_terminal_regions` generic `else` `_gb.py:241-247`). Companion fix: `compute_regression_residuals` Lad branch now matches sklearn's `CyAbsoluteError` gradient tie convention (`+1 if y >= f else -1`; the zero residual contributes `+1`, not `0`) so the leaf grouping matches sklearn's tree structure. Consumer: GBR `fit` (reached via crate re-export `lib.rs` + `RsGradientBoostingRegressor` `extras.rs:336`) → `FittedGradientBoostingRegressor::predict`. Tests: `divergence_lad_median_terminal_region` (live oracle `predict=[0.729×3,1.0×5]`, GREEN), `test_lad_leaf_value_median`, `test_group_samples_by_leaf`, `test_regression_residuals_lad` (tie convention). |
| REQ-6 (Huber terminal-region update) | SHIPPED | The GBR `fit` loop runs `huber_stage_delta` once per stage (`_weighted_percentile(|y-f|, 100*alpha)` over the in-bag set, `set_huber_delta` `_gb.py:267-272` via `weighted_percentile_uniform` matching `utils/stats.py:_weighted_percentile`), then `huber_leaf_value` replaces each leaf with `median(d) + mean(sign(d-median)·min(δ,|d-median|))` over the leaf residuals `d = y[idx]-f_vals[idx]` (`HuberLoss.fit_intercept_only` `_loss/loss.py:694-710`). Consumer: GBR `fit` → `FittedGradientBoostingRegressor::predict` (crate re-export + `RsGradientBoostingRegressor`). Live-verified: `loss='huber', alpha=0.9, n_estimators=3, lr=0.1, max_depth=2` on `y=[0,0,0,10,1,1,1,20]` → `predict=[0.729×3, 1.60975×4, 6.149]` matches sklearn to 1e-4 (covered by `test_gbr_huber_loss` + the ephemeral oracle check). |
| REQ-7 (LogLoss Newton terminal-region update, binary + multiclass, HEADLINE) | SHIPPED | GBC `fit_binary` runs `binary_newton_leaf` (= `Σ(y-p)/Σ p(1-p)` over the leaf, `np.average` form, `HalfBinomialLoss` branch `_gb.py:191-206`) and `fit_multiclass` runs `multiclass_newton_leaf` (= `(K-1)/K · Σ neg_g/Σ p(1-p)`, `HalfMultinomialLoss` branch `:208-225`) to replace each leaf before `f_vals += lr*value`; both go through `safe_divide` (`_safe_divide` `_gb.py:66-78`, `|den|<1e-150 → 0`). Consumer: GBC `fit` → `predict`/`predict_proba`/`decision_function` (crate re-export `lib.rs` + `RsGradientBoostingClassifier` `extras.rs:670`). Live-verified: binary `predict_proba[:,1]=[0.297947…×4, 0.702052…×4]` (`divergence_logloss_newton_terminal_region_binary`, GREEN); multiclass `predict_proba` row0 `[0.626013,0.186993,0.186993]` matches sklearn exactly (the `decision_function` raw scores differ only by the softmax-invariant `ln(K)` init-prior offset, REQ-3). Tests: `divergence_logloss_newton_terminal_region_binary`, `test_binary_newton_leaf_value` (incl. zero-Hessian guard). |
| REQ-8 (friedman_mse criterion + feature_importances) | NOT-STARTED | open prereq blocker #737. GB trees use `criterion='friedman_mse'` (`_fit_stage:472`, constraint `:361`); ferrolearn's `build_regression_tree_with_feature_subset` uses MSE. SPLIT selection is identical (Friedman improvement is monotone in single-split MSE reduction → same threshold, same structure at `subsample=1.0`), but the node-impurity used for `compute_feature_importances` differs, so `feature_importances_` MAGNITUDES diverge from sklearn (ordering typically preserved). ferrolearn exposes no explicit `criterion` param. |
| REQ-9 (subsample bootstrap RNG) | NOT-STARTED | open prereq blocker #738 (RNG-boundary). At `subsample<1.0` the GBR/GBC fit loops draw `rand::seq::index::sample(&mut StdRng, ...)`; sklearn draws its bootstrap mask from numpy-MT `random_state` (`_fit_stages:855-862`). Different PRNG streams → different per-round subsamples → no array-by-array parity (the documented numpy-MT-vs-StdRng boundary). At `subsample=1.0` (the default) both short-circuit to all samples and agree — that path is covered by REQ-4. ferrolearn reproducibility within its own RNG holds (`test_gbr_reproducibility`, `test_gbc_reproducibility`). |
| REQ-10 (missing param surface + early stopping + staged API) | NOT-STARTED | open prereq blocker #739. ferrolearn exposes none of: `n_iter_no_change`/`validation_fraction`/`tol` early stopping (`_fit_stages:840-945`), explicit `criterion` (`:472`), `ccp_alpha` (`:482`), `max_features` (`:479`), `min_impurity_decrease` (`:478`), `max_leaf_nodes`, `min_weight_fraction_leaf` (`:477`), the pluggable `init` estimator (`_init_state:538`), `staged_predict`/`staged_decision_function`, nor GBR `alpha`/`loss='quantile'` (PinballLoss, `_init_state:544`). |
| REQ-11 (PyO3 binding fidelity) | NOT-STARTED | open prereq blocker #740. `RsGradientBoostingRegressor` (`extras.rs:336`, signature `(n_estimators=100, learning_rate=0.1, max_depth=Some(3), random_state=None)`) and `RsGradientBoostingClassifier` (`extras.rs:670`, same signature, `y: i64`) expose no `loss`, `subsample`, `min_samples_*`, no `predict_proba`/`predict_log_proba`/`decision_function`, no `feature_importances_`/`classes_`; `import ferrolearn` cannot reach the LAD/Huber/multiclass-probability surface sklearn exposes. |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #741. `gradient_boosting.rs` imports `ndarray::{Array1, Array2}`, `num_traits::{Float, FromPrimitive, ToPrimitive}`, and `rand`/`rand::rngs::StdRng`, not `ferray-core`/`ferray::random` (R-SUBSTRATE). |

## Architecture

`GradientBoostingRegressor<F>` / `GradientBoostingClassifier<F>` are the unfitted
boundary types (public hyperparameter fields + `with_*` builders + `Default`).
`fit` validates, computes the init prior, and runs the forward stage-wise loop.

**GBR `fit`:** `init = mean(y)` (L2) / `median_f(y)` (LAD/Huber); each round
(1) `compute_regression_residuals` (REQ-2), (2) optional subsample
(`StdRng`, REQ-9), (3) `build_regression_tree_with_feature_subset` on the
residuals (MSE, REQ-8), (4) **terminal-region line-search**: for `Lad`/`Huber`,
`group_samples_by_leaf` then replace each leaf with `lad_leaf_value` /
`huber_leaf_value` (`LeastSquares` is the identity — leaf untouched), then
`f_vals[i] += lr*value` (REQ-5/REQ-6, SHIPPED). After the loop,
`compute_feature_importances` (REQ-8) builds the
normalized `feature_importances`. `FittedGradientBoostingRegressor::predict`
sums `init + Σ lr*leaf` over the stored trees.

**GBC `fit`** dispatches on `n_classes`: `fit_binary` (single tree sequence on
log-odds residuals `y - σ(f)`, init log-odds prior) or `fit_multiclass`
(`K` trees per round on `y_onehot_k - softmax_k`, init log-prior). Both now run
the Newton terminal-region update (`binary_newton_leaf` / `multiclass_newton_leaf`
via `group_samples_by_leaf` + `safe_divide`) to replace each leaf before
`f_vals += lr*value` (REQ-7, SHIPPED).
`predict`/`predict_proba`/`decision_function`/`predict_log_proba` traverse the
stored trees and apply `sigmoid` (binary) / `softmax` (multiclass) to the
cumulative raw score.

**Invariants held:** default-matching param surface; correct per-loss negative
gradients (incl. the LAD `+1`-at-tie convention); correct init priors;
L2 identity-update parity (REQ-4); **the `_update_terminal_regions` line-search
leaf update for Lad/Huber/LogLoss (REQ-5/6/7) — group leaf samples → compute the
loss-optimal leaf value (median / median+clipped-mean / Newton step) → replace →
apply `lr*leaf`, with the L2 identity preserved**; deterministic `subsample=1.0`
path; own-RNG reproducibility; feature-count guard on predict.
**Invariants NOT held vs sklearn:** (b) MSE instead of
friedman_mse impurity → `feature_importances_` magnitude divergence (REQ-8);
(c) `StdRng` vs numpy-MT at `subsample<1.0` (REQ-9, boundary); (d) missing early
stopping / `criterion` / `ccp_alpha` / `max_features` / `init` / `alpha` / staged
API (REQ-10); (e) thin PyO3 surface (REQ-11); (f) wrong substrate (REQ-12).

## Verification

Library crate (green at baseline `15cb050d`):
```
cargo test -p ferrolearn-tree --lib gradient_boosting::
cargo clippy -p ferrolearn-tree --all-targets -- -D warnings
cargo fmt --all --check
```
Live sklearn oracle (installed 1.5.2, run from `/tmp`):
```
# REQ-1 defaults:
python3 -c "from sklearn.ensemble import GradientBoostingRegressor as R, GradientBoostingClassifier as C; print({k:R().get_params()[k] for k in ['loss','learning_rate','n_estimators','max_depth','subsample','criterion','alpha']}); print({k:C().get_params()[k] for k in ['loss','learning_rate','n_estimators','max_depth','subsample','criterion']})"
#   -> GBR loss='squared_error',lr=0.1,n=100,max_depth=3,subsample=1.0,criterion='friedman_mse',alpha=0.9
#   -> GBC loss='log_loss', ...

# REQ-4 LINCHPIN — GBR squared_error end-to-end parity (deterministic, subsample=1.0):
python3 -c "import numpy as np; from sklearn.ensemble import GradientBoostingRegressor; X=np.arange(1,9.).reshape(-1,1); y=np.array([1,1,1,1,5,5,5,5.]); print(np.round(GradientBoostingRegressor(loss='squared_error',n_estimators=5,learning_rate=0.1,max_depth=1).fit(X,y).predict(X),8).tolist())"
#   -> [2.18098, 2.18098, 2.18098, 2.18098, 3.81902, 3.81902, 3.81902, 3.81902]
#   ferrolearn predict == this exactly (VERIFIED this iteration via an ephemeral
#   integration test; on a generic 30x3 dataset both agree to ~1e-7).

# REQ-5 LAD divergence (skewed leaves, median != mean):
python3 -c "import numpy as np; from sklearn.ensemble import GradientBoostingRegressor; X=np.arange(1,9.).reshape(-1,1); y=np.array([0,0,0,10,1,1,1,20.]); print(GradientBoostingRegressor(loss='absolute_error',n_estimators=3,learning_rate=1.0,max_depth=1).fit(X,y).predict(X).tolist())"
#   -> [0,0,0,1,1,1,1,1]  (sklearn weighted-median leaves); ferrolearn L2-mean leaves diverge.

# REQ-7 LogLoss divergence:
python3 -c "import numpy as np; from sklearn.ensemble import GradientBoostingClassifier; X=np.array([[1,2],[2,3],[3,3],[4,4],[5,6],[6,7],[7,8],[8,9.]]); y=np.array([0,0,0,0,1,1,1,1]); print(np.round(GradientBoostingClassifier(n_estimators=5,learning_rate=0.1,max_depth=1).fit(X,y).decision_function(X).ravel(),6).tolist())"
#   -> [-0.85709×4, 0.85709×4]  (Newton leaves); ferrolearn mean-leaf raw scores diverge.
```
REQ-1/2/3/4 are verified by the in-crate `#[test]`s named in the status table
plus the live `get_params()` / array-by-array L2 comparison (deterministic).
The NOT-STARTED REQs (5, 6, 7, 8, 9, 10, 11, 12) have no green verification by
construction — each carries an open prereq blocker. Characterization pins for
REQ-4/5/6/7 (R-CHAR-3, AC-4/5/6/7) belong in
`ferrolearn-tree/tests/divergence_gradient_boosting.rs`: assert L2 GBR `predict`
equals live `loss='squared_error'` (a SHIPPED guard); assert LAD `predict`
equals live `loss='absolute_error'`, Huber equals `loss='huber'`, and GBC
`decision_function` equals live sklearn — each of the latter three FAILS until
the terminal-region update lands.

## Blockers to open

- #734 — REQ-5 (HEADLINE, R-DEV-1): GBR `fit` loop adds the regression tree's
  MEAN-residual leaf directly (`f_vals[i] += lr*value`) instead of sklearn's
  weighted-MEDIAN terminal-region update (`_update_terminal_regions` generic
  branch `_gb.py:241-247` → `AbsoluteError.fit_intercept_only` `loss.py:565-574`).
  Part of the ONE coherent builder-scale terminal-region change in
  `gradient_boosting.rs` (with #735/#736).
- #735 — REQ-6 (R-DEV-1): no Huber terminal-region update `median +
  average(sign(d)·min(δ,|d|))` (`loss.py:694-710`, δ from `_gb.py:267-272`).
  Coherent with #734/#736.
- #736 — REQ-7 (HEADLINE, R-DEV-1): GBC `fit_binary`/`fit_multiclass` add the
  mean-residual leaf instead of the Newton step `Σw(y-p)/Σw·p(1-p)` (binomial,
  `_gb.py:191-206`) / `(K-1)/K · Σw·neg_g/Σw·p(1-p)` (multinomial, `:208-225`).
  Coherent with #734/#735 (group leaf samples → loss-optimal leaf → replace →
  `lr*leaf`).
- #737 — REQ-8: GB trees use MSE not `criterion='friedman_mse'` (`_gb.py:472`,
  `:361`); split selection matches (Friedman monotone in MSE for one split) but
  node impurity → `feature_importances_` magnitudes diverge; no `criterion` param.
- #738 — REQ-9 (RNG-boundary): `subsample<1.0` draws `StdRng` +
  `rand::seq::index::sample` vs sklearn numpy-MT `random_state`
  (`_fit_stages:855-862`); no array-by-array parity. `subsample=1.0` deterministic
  path agrees (REQ-4).
- #739 — REQ-10: missing `n_iter_no_change`/`validation_fraction`/`tol` early
  stopping (`_fit_stages:840-945`), explicit `criterion`, `ccp_alpha`,
  `max_features`, `min_impurity_decrease`, `max_leaf_nodes`,
  `min_weight_fraction_leaf`, pluggable `init`, `staged_predict`/
  `staged_decision_function`, GBR `alpha`/`loss='quantile'`.
- #740 — REQ-11: `RsGradientBoostingRegressor`/`RsGradientBoostingClassifier`
  (`extras.rs:336`/`:670`) expose no `loss`/`subsample`/`min_samples_*`/
  `predict_proba`/`decision_function`/`feature_importances_`/`classes_`.
- #741 — REQ-12: migrate `gradient_boosting.rs` off `ndarray`/`num-traits`/`rand`
  to the ferray substrate (R-SUBSTRATE).
