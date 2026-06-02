# CART Decision Trees (DecisionTreeClassifier / DecisionTreeRegressor)

<!--
tier: 3-component
status: draft
baseline-commit: 1037bcd6
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/tree/_classes.py        # BaseDecisionTree (:95), DecisionTreeClassifier (:698), DecisionTreeRegressor (:1093), CRITERIA_CLF/CRITERIA_REG (:71-83), feature_importances_ (:671)
  - sklearn/tree/_criterion.pyx     # Gini (:695), Entropy (:622), MSE (:1090), FriedmanMSE, MAE, Poisson (:1580), impurity_improvement (:188)
  - sklearn/tree/_splitter.pyx      # BestSplitter / node_split_best (:293), FEATURE_THRESHOLD (:33), midpoint threshold (:479)
  - sklearn/tree/_tree.pyx          # DepthFirstTreeBuilder / BestFirstTreeBuilder (max_leaf_nodes), ccp pruning
ferrolearn-module: ferrolearn-tree/src/decision_tree.rs
parity-ops: DecisionTreeClassifier, DecisionTreeRegressor
crosslink-issue: 658
-->

## Summary

`ferrolearn-tree/src/decision_tree.rs` mirrors scikit-learn's
`sklearn.tree.DecisionTreeClassifier` (`sklearn/tree/_classes.py:698`) and
`sklearn.tree.DecisionTreeRegressor` (`sklearn/tree/_classes.py:1093`) — the
**CART** estimators. sklearn delegates all numerics to the Cython
`Tree`/`Splitter`/`Criterion` stack (`_tree.pyx`, `_splitter.pyx`,
`_criterion.pyx`); ferrolearn re-implements that stack natively. The module
ships `ClassificationCriterion {Gini, Entropy}`, `RegressionCriterion {Mse}`, a
flat-`Vec<Node<F>>` tree, the unfitted `DecisionTreeClassifier<F>` /
`FittedDecisionTreeClassifier<F>` and `DecisionTreeRegressor<F>` /
`FittedDecisionTreeRegressor<F>` pairs, a recursive depth-first best-split
builder (`fn build_classification_tree`, `fn build_regression_tree` in
`decision_tree.rs`), `Fit`/`Predict`, `predict_proba`/`predict_log_proba`,
`feature_importances_`, and crate-public tree builders consumed by the ensemble
crates (`random_forest.rs`, `extra_tree.rs`, etc.).

The builder **appears algorithmically close** for the shipped subset (gini, MSE,
the `<=` midpoint split, weighted impurity decrease), but **no oracle-pinned
divergence test exists** — `ferrolearn-tree` has only loose unit tests
(`#[test] test_classifier_simple_binary`, `test_gini_impurity_balanced`, …)
asserting prediction labels and impurity at hand-computed points, none comparing
against a live `sklearn` `tree_.node_count` / root split / `feature_importances_`
/ `predict_proba` array. Under R-CHAR-1/R-CHAR-3 a numerical-contract REQ cannot
be SHIPPED until a test pins it against the live oracle. **The estimator surface
REQs (constructor types, the gini/MSE impurity helpers, predict/predict_proba
plumbing, fitted-attribute accessors that are wired and consumed) are SHIPPED**;
**every node-for-node numerical-parity REQ and every missing-parameter REQ is
NOT-STARTED** with a concrete blocker. This doc flags the lines the critic
should pin first (see Blockers and the top-3 suspected divergences).

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `inspect.signature`, sklearn 1.5.2)

`DecisionTreeClassifier.__init__`: `criterion='gini'`, `splitter='best'`,
`max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`,
`min_weight_fraction_leaf=0.0`, `max_features=None`, `random_state=None`,
`max_leaf_nodes=None`, `min_impurity_decrease=0.0`, `class_weight=None`,
`ccp_alpha=0.0`, `monotonic_cst=None` (`_classes.py:946`).
`DecisionTreeRegressor.__init__`: same block with `criterion='squared_error'`
and no `class_weight` (`_classes.py:1317`).

### Criteria (`_criterion.pyx`)

- **Classification** — `{"gini" (default), "entropy", "log_loss"}`
  (`CRITERIA_CLF`, `_classes.py:71`; `"log_loss"` and `"entropy"` BOTH map to
  `_criterion.Entropy`, `_classes.py:73-74`).
  - Gini: `node_impurity = 1 − Σ_c (count_c / N_t)²` (`_criterion.pyx:695`,
    `Gini`).
  - Entropy / log_loss: `node_impurity = −Σ_c p_c · ln(p_c)` where
    `p_c = count_c / N_t`, with `0·ln 0 = 0` skipped — **natural log (base e),
    NOT log₂** (`_criterion.pyx:655`, `entropy -= count_k * log(count_k)`).
- **Regression** — `{"squared_error" (default), "friedman_mse",
  "absolute_error", "poisson"}` (`CRITERIA_REG`, `_classes.py:76`).
  - MSE / squared_error: `impurity = sq_sum/N_t − (sum/N_t)²`
    (`_criterion.pyx:1094`).
  - friedman_mse: Friedman's variant that scores the split by
    `n_L·n_R/(n_L+n_R) · (mean_L − mean_R)²` (`FriedmanMSE`).
  - absolute_error / MAE: L1 around the **median** of each node (`MAE`).
  - poisson: half-Poisson deviance,
    `mean(y·log(y/ŷ) + ŷ − y)` (`_criterion.pyx:1580`, `Poisson`).

### Best-split builder (`_splitter.pyx::node_split_best`, `:293`)

For `splitter='best'`, at each node, for each candidate feature: sort the
node's samples by that feature, scan adjacent pairs, and for each split point
compute the **weighted impurity improvement**
`N_t/N · (impurity_parent − N_tR/N_t·imp_R − N_tL/N_t·imp_L)`
(`_criterion.pyx:188-222`, `impurity_improvement`). Selection rules:

- A feature is **constant** (skipped) when
  `feature_values[end-1] <= feature_values[start] + FEATURE_THRESHOLD`,
  `FEATURE_THRESHOLD = 1e-7` (`_splitter.pyx:33,404`). Equal adjacent values are
  skipped as split points (`feature_values[p] <= feature_values[p-1] +
  FEATURE_THRESHOLD`).
- Threshold = **midpoint**: `feature_values[p_prev]/2.0 + feature_values[p]/2.0`
  (`_splitter.pyx:479`), clamped to `feature_values[p_prev]` if FP rounding
  makes the midpoint `>= feature_values[p]` (`_splitter.pyx:488`).
- Split is rejected unless `n_left >= min_samples_leaf` AND
  `n_right >= min_samples_leaf` (`_splitter.pyx:451`) and the weighted child
  counts meet `min_weight_leaf = min_weight_fraction_leaf · N` (`:470`).
- sklearn keeps the split with the largest **proxy** improvement and breaks
  ties by feature order with a `>=` reservoir comparison; features are visited
  in a randomized order driven by `random_state` (`_splitter.pyx` Fisher-Yates).
  Even with `splitter='best'`, feature visitation order is RNG-permuted, so the
  *chosen* split among equal-improvement candidates is `random_state`-dependent.

### Stopping / pruning (`_tree.pyx`)

Depth-first `DepthFirstTreeBuilder` (default) stops a node when `n <
min_samples_split`, `depth >= max_depth`, the node is pure
(`impurity <= EPSILON`), or `n < 2·min_samples_leaf`. `max_leaf_nodes`
(when set) switches to `BestFirstTreeBuilder` (frontier expanded by best
improvement). `min_impurity_decrease` rejects a split whose *weighted* decrease
is below the threshold. `ccp_alpha > 0` triggers minimal cost-complexity
pruning after the tree is built (`_tree.pyx::ccp_pruning_path`).

### `max_features`

`{None, "sqrt", "log2", int, float}` (`_classes.py:941` constraints). At each
node `max_features_` features are drawn (without replacement, RNG-permuted) and
only those are searched. `None` ⇒ all features. `max_features_` is the resolved
integer, exposed as a fitted attribute.

### Fitted attributes

`tree_` (the node arrays), `feature_importances_` (normalized total weighted
impurity decrease per feature, `_classes.py:671`,
`tree_.compute_feature_importances()` then divided by the sum),
`n_features_in_`, `max_features_`; classifier also `classes_`, `n_classes_`,
`n_outputs_`. `predict` returns the leaf majority (classifier) / mean
(regressor); `predict_proba` returns leaf class frequencies (normalized to sum
1); `predict_log_proba = log(predict_proba)`.

### Live oracle anchors (sklearn 1.5.2, the critic's first pin targets)

Classifier, `X = [[1,2],[2,3],[3,3],[5,6],[6,7],[7,8],[1.5,5],[6.5,2],[3,1]]`,
`y=[0,0,0,1,1,1,2,2,0]`, `DecisionTreeClassifier(random_state=0)`:
`tree_.node_count = 7`; root split `feature=1, threshold=5.5`;
`feature_importances_ = [0.18461538461538465, 0.8153846153846153]`;
`classes_=[0,1,2]`, `n_classes_=3`, `max_features_=2`, `n_features_in_=2`;
`predict = [0,0,0,1,1,1,2,2,0]`; `predict_proba[:3] = [[1,0,0],[1,0,0],[1,0,0]]`.

Regressor, `X=[[1]..[8]]`, `y=[1.0,1.2,0.9,1.1,5.0,5.2,4.9,5.1]`,
`DecisionTreeRegressor(random_state=0)`: `tree_.node_count = 15`; root split
`feature=0, threshold=4.5`; `predict = y` (each sample its own leaf);
`feature_importances_ = [1.0]`.

## ferrolearn (what exists)

`decision_tree.rs` ships:

- `enum ClassificationCriterion { Gini, Entropy, LogLoss }` and
  `enum RegressionCriterion { Mse, FriedmanMse, AbsoluteError, Poisson }` — the
  full sklearn criterion sets (`CRITERIA_CLF`/`CRITERIA_REG`, `_classes.py:71-81`).
  `LogLoss` is the `entropy` alias; the regression criteria are dispatched
  through `fn regression_node_impurity` / `fn regression_leaf_value` /
  `fn find_best_regression_split`.
- `struct DecisionTreeClassifier<F>` with `max_depth: Option<usize>`,
  `min_samples_split: usize`, `min_samples_leaf: usize`,
  `criterion: ClassificationCriterion` and builder methods `new`,
  `with_max_depth`, `with_min_samples_split`, `with_min_samples_leaf`,
  `with_criterion`; `DecisionTreeRegressor<F>` analogous. **Only 4 of the 13
  sklearn constructor params exist**; `max_features`, `random_state`,
  `max_leaf_nodes`, `min_impurity_decrease`, `class_weight`, `ccp_alpha`,
  `min_weight_fraction_leaf`, `splitter`, `monotonic_cst` are absent at the
  estimator surface.
- `enum Node<F> { Split{...}, Leaf{...} }`, flat `Vec<Node<F>>`.
- `fn build_classification_tree` / `fn build_regression_tree` — recursive
  depth-first builder; `fn find_best_classification_split` /
  `fn find_best_regression_split` — the best-split scan with `<=`-midpoint
  thresholds; `fn gini_impurity`, `fn entropy_impurity` (uses `.ln()`),
  `fn mse_for_indices`, `fn compute_impurity`.
- `fn predict`, `fn predict_proba`, `fn predict_log_proba`, `fn score`,
  `fn feature_importances` (via `HasFeatureImportances`), `fn classes`/
  `n_classes` (via `HasClasses`), `fn nodes`, `fn n_features`.
- crate-public `fn compute_feature_importances`, `fn aggregate_tree_importances`,
  `fn traverse`, and the forest builders
  `build_classification_tree_per_split_features` /
  `build_regression_tree_per_split_features` (per-split RNG feature subsampling)
  / `..._with_feature_subset`.

**Non-test production consumers** (R-DEFER-1 / R-DEFER-5 grandfathering):
the types are re-exported in `ferrolearn-tree/src/lib.rs`
(`pub use decision_tree::{ClassificationCriterion, DecisionTreeClassifier,
DecisionTreeRegressor, FittedDecisionTreeClassifier, FittedDecisionTreeRegressor,
Node, RegressionCriterion}`); the ensemble crates consume the internal builders
(`random_forest.rs` calls `build_classification_tree_per_split_features`,
`compute_feature_importances`, `traverse`; `extra_tree.rs`, `bagging.rs`,
`adaboost*.rs`, `gradient_boosting.rs` similarly); and the PyO3 binding wraps
`DecisionTreeClassifier` (`RsDecisionTreeClassifier` in
`ferrolearn-python/src/classifiers.rs`).

## Requirements

- REQ-1 (criteria — formulas): classification `gini = 1−Σp²` (default) and
  `entropy = −Σp·ln p` (natural log); regression `squared_error` (default);
  PLUS the missing `log_loss` alias and `friedman_mse`/`absolute_error`/
  `poisson` regression criteria, each matching sklearn `_criterion.pyx`.
- REQ-2 (best-split builder, `splitter='best'`): for each candidate feature,
  weighted impurity improvement `N_t/N·(parent − N_tL/N_t·imp_L −
  N_tR/N_t·imp_R)`; midpoint threshold `(x[p_prev]+x[p])/2`; `<=` goes left;
  constant-feature skip at `FEATURE_THRESHOLD=1e-7`; tie-break + RNG feature
  order matching sklearn node-for-node.
- REQ-3 (stopping / pruning params): `max_depth`, `min_samples_split`,
  `min_samples_leaf`, `min_weight_fraction_leaf` (#664), `min_impurity_decrease`
  (#662) present; `max_leaf_nodes` (best-first, #661), `ccp_alpha`
  (cost-complexity pruning, #663) still absent, all with sklearn semantics.
- REQ-4 (`max_features`): `{None, "sqrt", "log2", int, float}` resolved to
  `max_features_`, per-node RNG subsampling (`random_state`-dependent boundary).
- REQ-5 (fitted attributes): `tree_` structure (`node_count`, per-node
  feature/threshold), `feature_importances_` (normalized weighted impurity
  decrease), `classes_`/`n_classes_` (classifier), `n_features_in_`,
  `max_features_`.
- REQ-6 (predict / predict_proba / predict_log_proba + multiclass): leaf
  majority/mean; leaf class frequencies; element-wise log; multiclass and
  multi-output classification.
- REQ-7 (`class_weight` + `random_state` determinism): classifier
  `class_weight={None, "balanced", dict}` reweighting; documented
  `random_state` boundary for feature-subsampling / tie order (the SGD / libsvm
  RNG-boundary precedent).
- REQ-8 (ferray substrate, R-SUBSTRATE): module computes on `ferray-core`
  arrays, not `ndarray`.

## Acceptance criteria

- AC-1: gini/entropy/MSE on a fixed node equal sklearn's `_criterion` node
  impurity within 1e-12; `log_loss` alias and friedman_mse/MAE/poisson exist and
  match.
- AC-2: on the classifier oracle set, `node_count == 7`, root split
  `(feature=1, threshold=5.5)`; on the regressor oracle set `node_count == 15`,
  root `(feature=0, threshold=4.5)` — node-for-node against
  `DecisionTreeClassifier/Regressor(random_state=0)`.
- AC-3: setting `min_impurity_decrease`, `max_leaf_nodes`, `ccp_alpha` produces
  trees identical to sklearn with the same params.
- AC-4: `max_features='sqrt'` with a fixed `random_state` reproduces sklearn's
  `max_features_` and the chosen split set.
- AC-5: `feature_importances_` equals sklearn within 1e-12
  (`[0.18461538…, 0.81538461…]` on the oracle classifier set).
- AC-6: `predict_proba` equals sklearn array-for-array; multiclass labels in
  `classes_` order.
- AC-7: `class_weight='balanced'` matches sklearn; identical `random_state`
  yields identical trees.
- AC-8: the module's owned computation uses `ferray-core` array types.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (criteria — gini/entropy/MSE + log_loss/friedman_mse/absolute_error/poisson) | SHIPPED (alt criteria; gini/entropy/MSE oracle pin still tracked by #659) | impl: `enum ClassificationCriterion { Gini, Entropy, LogLoss }` and `enum RegressionCriterion { Mse, FriedmanMse, AbsoluteError, Poisson }` in `decision_tree.rs`. `log_loss` is an alias for `entropy` — both route to `fn entropy_impurity` via `fn compute_impurity` (matching `CRITERIA_CLF`, `_classes.py:73`). Regression criteria are dispatched in `fn regression_node_impurity` (node impurity), `fn regression_leaf_value` (mean for MSE/FriedmanMSE/Poisson, **median** via `fn median_value` for AbsoluteError — `MAE.node_value`, `_criterion.pyx:1419`), and `fn find_best_regression_split` (per-criterion split score): FriedmanMSE proxy `diff²/(n_L·n_R·n)`, `diff = n_R·sum_L − n_L·sum_R` (`_criterion.pyx:1557-1574`); AbsoluteError via `fn mae_for_indices` (`(1/n)Σ|y−median|`, `_criterion.pyx:1450-1472`); Poisson via `fn poisson_deviance_for_indices` (`(1/n)Σ y·ln(y/mean)`, `0·ln0=0`, `+∞` when child sum ≤ ε, `_criterion.pyx:1671-1708`). Poisson `y` guard in `DecisionTreeRegressor::fit` rejects negative y / Σy≤0 (`_classes.py:267-277`). **Production consumers** (R-DEFER-1): the new variants are reachable via the grandfathered `with_criterion` builder on the re-exported `DecisionTreeRegressor`/`DecisionTreeClassifier` (`lib.rs` `pub use`), and `compute_impurity` (both finders) is consumed by `random_forest.rs`/`extra_tree.rs`/`bagging.rs`. **Smoke tests** (R-CHAR-3, live sklearn 1.5.2 oracle): `test_classifier_log_loss_is_entropy_alias` (log_loss == entropy, entropy root `(1,5.5)` + fi `[0.13795,0.86205]`), `test_regressor_friedman_mse_oracle` / `test_regressor_poisson_oracle` (root `(0,4.5)`, mean leaves `[1.1,1.1,1.0,1.0,5.1,5.1,5.0,5.0]`), `test_regressor_absolute_error_median_leaves_oracle` (median leaves `[1.0,1.1,1.1,1.1,5.0,5.1,5.1,5.1]`), `test_regressor_poisson_rejects_non_positive_y`, `test_regressor_squared_error_unchanged` (MSE byte-identical). The dedicated gini/entropy/MSE node-impurity oracle pin remains tracked under #659 (critic-owned). |
| REQ-2 (best-split builder node-for-node) | NOT-STARTED (partial — #660 closed) | impl `fn find_best_classification_split`/`fn find_best_regression_split` in `decision_tree.rs` compute weighted impurity decrease and `(x[i]+x[i+1])/2` midpoint with `<=`-left. The `FEATURE_THRESHOLD=1e-7` constant-feature band IS now shipped (`fn feature_threshold`, used in BOTH finders: feature skipped when sorted spread `feat_max <= feat_min + threshold_band`, split point skipped when `x[next] <= x[idx] + threshold_band`, mirroring `_splitter.pyx:33,405`), and `make_classification_leaf` argmax now picks the lowest index on count ties (sklearn `np.argmax`) — pinned by `divergence_clf_feature_threshold_band` (node_count==1, predict all-0 on the 3e-8-spread set). STILL open: no RNG feature permutation / strict `>` tie-break vs sklearn's RNG-ordered reservoir (remaining blocker #659/#670, pinned RED by `divergence_clf_tiebreak_random_state`); the `node_count==7`/root `(1,5.5)` oracle is pinned green by `clf_tree_structure_oracle`. |
| REQ-3 (stopping/pruning params) | NOT-STARTED (partial — #662 + #664 closed; #661 max_leaf_nodes / #663 ccp_alpha still open) | impl: `max_depth`, `min_samples_split`, `min_samples_leaf` are wired (`fn build_classification_tree` stop check `n < params.min_samples_split \|\| depth >= max_depth`) and unit-tested (`test_classifier_max_depth_1`, `test_classifier_min_samples_leaf`). **`min_impurity_decrease` (#662) SHIPPED**: `pub min_impurity_decrease: F` + `#[must_use] with_min_impurity_decrease` on BOTH `DecisionTreeClassifier`/`DecisionTreeRegressor` (default `0.0`, mirroring `_classes.py:946`/`:1317`); the gate lives in `struct ImpurityGate` (threaded through `fn build_classification_tree`/`fn build_regression_tree`), rejecting a split when the tree-normalized improvement `N_t/N·(parent − N_tL/N_t·imp_L − N_tR/N_t·imp_R)` satisfies `improvement + EPSILON < min_impurity_decrease` (`ImpurityGate::rejects`, `_tree.pyx:284`; `EPSILON = F::epsilon()` = `np.finfo('double').eps`, `_tree.pyx:63`). The improvement is recovered from the finder's stored `best_impurity_decrease`: `/N` for gini/entropy/MSE/MAE/poisson, `/N_t` for friedman_mse whose finder score IS `FriedmanMSE.impurity_improvement = diff²/(n_L·n_R·n_t)` (`_criterion.pyx:1573`). The finders' accept gate was relaxed `> 0` → `>= 0` so zero-improvement splits the default `0.0` gate accepts are returned (sklearn's `>=`-semantics). **`min_weight_fraction_leaf` (#664) SHIPPED**: `pub min_weight_fraction_leaf: F` + `#[must_use] with_min_weight_fraction_leaf` on BOTH (default `0.0`); folded ONCE at fit into the effective per-child minimum `max(min_samples_leaf, ceil(min_weight_fraction_leaf · N))` (`fn effective_min_samples_leaf`, uniform-weight `min_weight_leaf = min_weight_fraction_leaf · N` of `_classes.py:371`, child-reject `weighted_n < min_weight_leaf` of `_splitter.pyx:470`), fed as `params.min_samples_leaf` into the existing split-finder child-size gate. **Production consumers** (R-DEFER-1): both params are reachable via the new `with_*` builders on the re-exported `DecisionTreeClassifier`/`DecisionTreeRegressor` (`lib.rs` `pub use`); the gate/fold are invoked from `DecisionTreeClassifier::fit`/`DecisionTreeRegressor::fit`; the forest builders pass `ImpurityGate::disabled` (byte-identical, no `min_impurity_decrease` exposure). **Smoke tests** (R-CHAR-3, live sklearn 1.5.2 oracle, 9×2 set): `test_classifier_min_impurity_decrease_default_node_count_7` (node_count 7, predict `[0,0,0,1,1,1,2,2,0]`), `_0_2_node_count_3` (mid=0.2 → 3, `[0,0,0,1,1,1,0,0,0]`), `_0_5_node_count_1` (mid=0.5 → 1, all-0), `test_classifier_min_weight_fraction_leaf_0_25_node_count_5` (mwfl=0.25 → 5, `[0,0,0,1,1,1,0,0,0]`), `test_effective_min_samples_leaf_fold`. STILL absent: `max_leaf_nodes` (best-first builder, #661), `ccp_alpha` (cost-complexity pruning, #663). The regressor finder retains the `> 0` accept (so the zero-improvement `min_impurity_decrease` boundary for regression remains pre-existing divergence #660/#3); the classifier accept is `>= 0`. open prereq blockers #661 (max_leaf_nodes), #663 (ccp_alpha). |
| REQ-4 (max_features resolution + subsampling) | NOT-STARTED | impl: per-split sampling exists internally (`max_features_per_split` in `ClassificationData`, used by `build_classification_tree_per_split_features`) but `DecisionTreeClassifier`/`Regressor` expose NO `max_features` param and NO `max_features_` fitted attribute; the `{"sqrt","log2",float}` resolution does not exist on the estimator. open prereq blocker #665. |
| REQ-5 (fitted attributes) | NOT-STARTED (partial) | impl: `fn feature_importances` (`HasFeatureImportances`, consumed by `random_forest.rs` `aggregate_tree_importances` and the PyO3 binding), `fn classes`/`n_classes` (`HasClasses`), `fn nodes`, `fn n_features` are SHIPPED at the API level. But `feature_importances_` normalization is NOT oracle-pinned (`test_classifier_feature_importances` only asserts `sum==1` and `[0]>0`, not the `[0.1846…, 0.8153…]` values), no `max_features_`, and `nodes()` is not pinned to sklearn's `tree_` node count/order. open prereq blocker #666 (feature_importances_ numeric pin), #665 (max_features_). |
| REQ-6 (predict / predict_proba / multiclass) | NOT-STARTED (partial) | impl `fn predict`/`fn predict_proba`/`fn predict_log_proba` in `decision_tree.rs` exist, consumed by the PyO3 binding (`RsDecisionTreeClassifier`) and the pipeline adapter (`FittedClassifierPipelineAdapter::predict_pipeline`). Multiclass label prediction is unit-tested (`test_classifier_multiclass`). But `predict_proba` leaf-frequency arrays are NOT oracle-pinned (`test_classifier_predict_proba` only checks row sums == 1), and multi-output (2-D `y`) is unsupported (`Fit<Array2<F>, Array1<usize>>` only). open prereq blocker #667 (predict_proba pin), #668 (multi-output). |
| REQ-7 (class_weight + random_state determinism) | NOT-STARTED | impl: no `class_weight` field on `DecisionTreeClassifier`; no `random_state` field (RNG only enters via the forest builders' `seed: u64`). The estimator's split search is deterministic (no feature permutation), so it can NOT reproduce sklearn's `random_state`-dependent tie behavior. open prereq blocker #669 (class_weight), #670 (random_state determinism boundary, à la the SGD / libsvm-CV RNG boundary). |
| REQ-8 (ferray substrate) | NOT-STARTED | impl: module imports `use ndarray::{Array1, Array2}` and computes entirely on `ndarray` — the wrong substrate per R-SUBSTRATE-1. No `ferray-core` usage. open prereq blocker #671 (migrate decision_tree.rs to ferray-core). |

## Architecture

The fitted estimators store the tree as a flat `Vec<Node<F>>` (index 0 = root,
`enum Node<F> { Split { feature, threshold, left, right, impurity_decrease,
n_samples }, Leaf { value, class_distribution, n_samples } }`) — a denser layout
than sklearn's struct-of-arrays `tree_` (`_tree.pyx`), but observationally
equivalent if node count and per-node splits match. Traversal
(`fn traverse_tree`) sends `sample[feature] <= threshold` left, matching
sklearn's `X[i, feature] <= threshold` convention (`_tree.pyx` apply).

The builder is recursive depth-first (`fn build_classification_tree`,
`fn build_regression_tree`), structurally the `DepthFirstTreeBuilder`
(`_tree.pyx`) but WITHOUT the `max_leaf_nodes` best-first variant,
`min_impurity_decrease` gate, or `ccp_alpha` post-pruning. The split-finder
(`fn find_best_classification_split`) recomputes impurity from incrementally
maintained class counts (classifier) / running sums (regressor) — the same
O(n_features · n log n) sort-scan sklearn uses — and returns
`best_score * n` as the stored `impurity_decrease`, where `best_score` is the
*per-node-normalized* decrease `parent − Σ (n_child/n)·imp_child`. NOTE: sklearn
normalizes the importance by `N_t/N_total` at importance-aggregation time
(`compute_feature_importances`), whereas ferrolearn folds `* n` into the node
and divides by the global sum in `fn compute_feature_importances` — the critic
must confirm these produce the identical normalized vector
(`_classes.py:671`); the suspected mismatch is documented as a top-3 divergence.

Class labels are mapped to dense `0..n_classes` indices at fit
(`y_mapped`), and `classes` holds the sorted unique original labels — mirroring
sklearn's `classes_` ordering (`LabelEncoder`-style). The pure-node /
single-class early stop (`class_counts … count() <= 1`) and the regressor's
`parent_mse <= F::epsilon()` early stop mirror sklearn's `EPSILON` purity
check but the threshold constant differs (machine epsilon vs sklearn's
`EPSILON = 1e-7`), a candidate divergence on near-constant targets.

## Verification

Commands that would establish each SHIPPED claim (none of the numerical REQs
are SHIPPED until pinned):

```bash
# API-level (currently green — proves predict/predict_proba/accessors run):
cargo test -p ferrolearn-tree decision_tree

# Oracle pins the critic must ADD (currently absent → REQs NOT-STARTED):
#   classifier node_count/root split/feature_importances_/predict_proba vs
python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
X=np.array([[1,2],[2,3],[3,3],[5,6],[6,7],[7,8],[1.5,5],[6.5,2],[3,1]]); \
y=np.array([0,0,0,1,1,1,2,2,0]); c=DecisionTreeClassifier(random_state=0).fit(X,y); \
print(c.tree_.node_count, c.tree_.feature[0], c.tree_.threshold[0], c.feature_importances_.tolist())"
#   regressor root split / predict vs DecisionTreeRegressor(random_state=0)
```

The existing `#[cfg(test)] mod tests` asserts label correctness and hand-checked
gini/entropy at balanced/pure counts (`test_gini_impurity_balanced` → 0.5,
`test_entropy_balanced` → `ln 2`), satisfying API smoke but NOT R-CHAR-3
(no live-oracle expected value, no `node_count`/`feature_importances_` array
pin). Until those land, the numerical REQs stay NOT-STARTED.

## Blockers to open

(Suggested sequential numbers from the next free issue #659; the user creates
them.)

- **#659** — Blocker for REQ-1 of decision_tree: add `log_loss` criterion alias
  (classification) and `friedman_mse`/`absolute_error`/`poisson` regression
  criteria; pin gini/entropy/MSE against live `_criterion.pyx`.
- **#660** — Blocker for REQ-2 of decision_tree: best-split builder does not
  match sklearn node-for-node (no `FEATURE_THRESHOLD=1e-7` constant-feature
  band, no RNG feature-order tie-break); pin `node_count`/root split against the
  oracle.
- **#661** — Blocker for REQ-3 of decision_tree: add `max_leaf_nodes` (best-first
  builder) — extends existing issue #51.
- **#662** — Blocker for REQ-3 of decision_tree: add `min_impurity_decrease`
  split gate. **CLOSED** — `pub min_impurity_decrease` + `with_min_impurity_decrease`
  on both estimators; `ImpurityGate::rejects` applies `improvement + EPSILON <
  threshold` (`_tree.pyx:284`) in both build loops.
- **#663** — Blocker for REQ-3 of decision_tree: add `ccp_alpha` cost-complexity
  pruning — extends existing issue #49.
- **#664** — Blocker for REQ-3 of decision_tree: add `min_weight_fraction_leaf`.
  **CLOSED** — `pub min_weight_fraction_leaf` + `with_min_weight_fraction_leaf`
  on both estimators; `fn effective_min_samples_leaf` folds
  `ceil(min_weight_fraction_leaf · N)` into the per-child leaf gate
  (`_classes.py:371`, `_splitter.pyx:470`).
- **#665** — Blocker for REQ-4/REQ-5 of decision_tree: add `max_features`
  (`{None,"sqrt","log2",int,float}`) param + `max_features_` fitted attribute on
  the estimator.
- **#666** — Blocker for REQ-5 of decision_tree: pin `feature_importances_`
  values (`[0.1846…, 0.8153…]`) against the oracle; verify the `*n` /
  global-normalize scheme equals sklearn's `N_t/N`-weighted normalization.
- **#667** — Blocker for REQ-6 of decision_tree: pin `predict_proba` leaf
  frequency arrays against the oracle.
- **#668** — Blocker for REQ-6 of decision_tree: add multi-output (2-D `y`)
  classification/regression support.
- **#669** — Blocker for REQ-7 of decision_tree: add `class_weight`
  (`{None,"balanced",dict}`) — extends existing issue #54 (sample_weight).
- **#670** — Blocker for REQ-7 of decision_tree: add `random_state` +
  `splitter='random'`; document the RNG-determinism boundary — extends existing
  issue #50.
- **#671** — Blocker for REQ-8 of decision_tree (R-SUBSTRATE): migrate
  `decision_tree.rs` owned computation from `ndarray` to `ferray-core`.

## Top 3 suspected divergences (critic's first pins)

1. **Constant-feature / tie threshold band + RNG tie-break (REQ-2).** ferrolearn
   skips a split point only on exact equality `x[idx]==x[next]`
   (`find_best_classification_split`), whereas sklearn treats a feature as
   constant within `FEATURE_THRESHOLD=1e-7` (`_splitter.pyx:404`) and visits
   features in a `random_state`-permuted order, breaking improvement ties
   differently. On the oracle classifier set the root is `(feature=1,
   threshold=5.5)`; ferrolearn's deterministic feature-0-first scan may pick a
   different feature when feature 0 and feature 1 give equal improvement →
   different `node_count`. **Pin `node_count==7` and root `(1, 5.5)` first.**

2. **`feature_importances_` normalization (REQ-5).** ferrolearn stores
   `best_score * n` per split (per-node decrease × node sample count) and divides
   by the global sum (`fn compute_feature_importances`). sklearn computes
   `Σ_nodes (N_t/N_total)·node_decrease` then normalizes
   (`_classes.py:671`). These coincide only if the per-node weighting is the
   same — ferrolearn omits the `/N_total` on the parent-impurity term inside the
   decrease. Expected oracle vector `[0.18461538…, 0.81538461…]`; confirm
   ferrolearn matches to 1e-12. **High-suspicion mismatch.**

3. **MSE/purity early-stop epsilon + best-score `> 0` gate (REQ-2/REQ-3).** The
   regressor stops at `parent_mse <= F::epsilon()` (≈2.2e-16 for f64) vs
   sklearn's `EPSILON=1e-7` purity threshold, and both finders accept a split
   only when `best_score > F::zero()` (strict positive) — sklearn accepts any
   improvement above `-INFINITY` subject to `min_impurity_decrease` (default
   0.0, a `>=` gate inside the builder). On near-constant or tiny-improvement
   data this changes whether a node splits → different `node_count`
   (regressor oracle expects `node_count==15`). Pin the regressor tree shape.
