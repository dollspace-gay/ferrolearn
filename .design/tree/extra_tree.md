# Extremely Randomized Trees (ExtraTreeClassifier / ExtraTreeRegressor)

<!--
tier: 3-component
status: draft
baseline-commit: ee767063
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/tree/_classes.py        # ExtraTreeClassifier (:1424), ExtraTreeRegressor (:1698); both extend DecisionTree* with splitter='random'; ExtraTreeClassifier.__init__ defaults criterion='gini'/splitter='random'/max_features='sqrt' (:1564-1572), ExtraTreeRegressor.__init__ defaults criterion='squared_error'/splitter='random'/max_features=1.0 (:1838-1846)
  - sklearn/tree/_splitter.pyx      # RandomSplitter / node_split_random (:674), find_min_max (:949), FEATURE_THRESHOLD constant skip (:773), rand_uniform threshold draw (:785), threshold==max -> min guard (:791), partition (:795)
  - sklearn/tree/_criterion.pyx     # Gini (:695), Entropy (:622), LogLoss alias, MSE (:1090), FriedmanMSE, MAE, Poisson (:1580) — shared with decision_tree.rs
  - sklearn/utils/_random.pxd       # rand_uniform(low, high, state) = (high-low)*rand_double + low  (:52)
ferrolearn-module: ferrolearn-tree/src/extra_tree.rs
parity-ops: ExtraTreeClassifier, ExtraTreeRegressor
crosslink-issue: none yet (658-area: decision_tree #658, criteria #659)
-->

## Summary

`ferrolearn-tree/src/extra_tree.rs` mirrors scikit-learn's
`sklearn.tree.ExtraTreeClassifier` (`sklearn/tree/_classes.py:1424`) and
`sklearn.tree.ExtraTreeRegressor` (`sklearn/tree/_classes.py:1698`) — the
**single-tree** extremely-randomized variants. In sklearn each is a thin
subclass of `DecisionTreeClassifier`/`DecisionTreeRegressor` whose only
difference is `splitter='random'` (the `RandomSplitter`,
`_splitter.pyx::node_split_random`, `:674`): for each of `max_features`
randomly-selected candidate features, a **single random threshold** is drawn
uniformly in `[min, max]` of that feature in the node, and the best split
**among those random candidates** is taken — versus `BestSplitter` which scans
every midpoint. The classifier defaults `max_features='sqrt'`; the regressor
defaults `max_features=1.0` (all features). Everything else — the
criteria, the tree object, predict/predict_proba, feature_importances_ — is
inherited unchanged from the CART base classes.

ferrolearn re-implements this natively. The module ships the unfitted
`ExtraTreeClassifier<F>` / `ExtraTreeRegressor<F>` and their `Fitted*`
counterparts, a recursive random-threshold builder
(`fn build_extra_classification_tree`, `fn build_extra_regression_tree` in
`extra_tree.rs`), `Fit`/`Predict`, `predict_proba`/`predict_log_proba`,
`feature_importances_`, and crate-public ensemble builders
(`fn build_extra_classification_tree_for_ensemble`,
`fn build_extra_regression_tree_for_ensemble`) consumed by
`extra_trees_ensemble.rs`. The criteria types (`ClassificationCriterion`,
`RegressionCriterion`), `Node<F>`, `TreeParams`, `fn traverse` and
`fn compute_feature_importances` are **shared with `decision_tree.rs`**.

**ExtraTree is inherently RNG-driven.** The split threshold is a random draw
seeded by `random_state`; numpy's MT19937 stream (sklearn) and Rust's `StdRng`
(ferrolearn) cannot bit-match, so **exact node-for-node parity at a given
`random_state` is a documented RNG boundary** — the same boundary already
accepted for the SGD shuffle, the libsvm cross-validation fold RNG, and
decision_tree's RNG-ordered tie-break. The **deterministic** contract (criteria
formulas, the param/default surface, predict structure, feature_importances
normalization, the random-threshold *distribution and bounds*, error paths) is
the shippable/pinnable part; exact threshold values are not.

**Compile-forced gap (top divergence — director needs a one-line fix).**
`decision_tree.rs` recently gained `ClassificationCriterion::LogLoss` (and
`RegressionCriterion::{FriedmanMse, AbsoluteError, Poisson}`). extra_tree.rs's
**private** `fn compute_impurity` matches only `Gini` and `Entropy` →
`error[E0004]: non-exhaustive patterns: ClassificationCriterion::LogLoss not
covered` at `extra_tree.rs:798`. **The workspace does not compile.** This is the
prerequisite blocker for *every* REQ below (nothing can be verified until the
crate builds). The fix is the same as decision_tree's
(`Entropy | LogLoss => entropy_impurity`, `decision_tree.rs:1012`); the
regression finders here are MSE-only and do not pattern-match
`RegressionCriterion`, so the regressor does not have the same compile gap but
silently ignores friedman_mse/absolute_error/poisson (REQ-1).

This doc adapts to the **existing** code: under R-HONEST-3 no oracle pins exist
yet and the estimator is RNG-driven, so the API-surface REQs (constructor types
+ defaults, predict/predict_proba plumbing, fitted-attribute accessors that are
wired and consumed) are SHIPPED, while every node-for-node numerical-parity REQ,
the LogLoss compile gap, the regression-criterion gap, and the constant-feature
band are NOT-STARTED with concrete blockers.

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `inspect.signature`, sklearn 1.5.2)

`ExtraTreeClassifier.__init__` (`_classes.py:1564`): `criterion='gini'`,
`splitter='random'`, `max_depth=None`, `min_samples_split=2`,
`min_samples_leaf=1`, `min_weight_fraction_leaf=0.0`, `max_features='sqrt'`,
`random_state=None`, `max_leaf_nodes=None`, `min_impurity_decrease=0.0`,
`class_weight=None`, `ccp_alpha=0.0`, `monotonic_cst=None`. Live:
`ExtraTreeClassifier().max_features == 'sqrt'`, `.splitter == 'random'`,
`.criterion == 'gini'`.

`ExtraTreeRegressor.__init__` (`_classes.py:1838`): same block with
`criterion='squared_error'`, `splitter='random'`, `max_features=1.0` (i.e. all
features), and no `class_weight`. Live: `ExtraTreeRegressor().max_features ==
1.0`, `.criterion == 'squared_error'`.

**Note the asymmetry**: ExtraTreeClassifier defaults `max_features='sqrt'` (NOT
`None` like DecisionTreeClassifier), while ExtraTreeRegressor defaults to all
features. ferrolearn matches both: `ExtraTreeClassifier::new()` →
`MaxFeatures::Sqrt`, `ExtraTreeRegressor::new()` → `MaxFeatures::All`.

### Random-split builder (`_splitter.pyx::node_split_random`, `:674`)

For each candidate feature `f` drawn from the random `max_features` subset
(reservoir-style draw over not-yet-constant features):

1. `partitioner.find_min_max(f, &min, &max)` (`:949`) — node-local min/max of
   the feature over the current sample set.
2. **Constant-feature skip**: `if max <= min + FEATURE_THRESHOLD` (`:773`,
   `FEATURE_THRESHOLD = 1e-7`), the feature is marked constant and skipped.
3. **Threshold draw**: `threshold = rand_uniform(min, max, random_state)`
   (`:785`). `rand_uniform(low, high) = (high - low) * rand_double(state) +
   low` (`utils/_random.pxd:52`) — uniform in `[low, high)`.
4. **Boundary guard**: `if threshold == max: threshold = min` (`:791`) — so the
   threshold is always in `[min, max)`, guaranteeing a non-trivial partition.
5. `partition_samples(threshold)` splits left `x <= threshold` / right
   `x > threshold`; reject if either side `< min_samples_leaf` (`:795`).
6. The split's `improvement` (impurity decrease) is computed by the criterion;
   the **best among the random candidates** is kept. The search continues until
   at least one valid partition is found, even if it inspects more than
   `max_features` features (sklearn docstring note, `_classes.py`).

Crucially each feature gets **one** random threshold (not a scan) — that single
draw is the whole randomization.

### ferrolearn random-threshold builder (what exists)

`fn find_random_classification_split` / `fn find_random_regression_split` in
`extra_tree.rs`:
- Sample `k = min(max_features_n, n_features)` features without replacement via
  `rand::seq::index::sample` (`fn rand_sample_indices`).
- For each: compute node-local `feat_min`/`feat_max`; **skip if
  `feat_min >= feat_max`** (no `FEATURE_THRESHOLD` epsilon band — divergence,
  REQ-2); draw `threshold = fn random_threshold(rng, min, max)` =
  `min + u*(max-min)` where `u = next_u64 / u64::MAX ∈ [0, 1)` (same
  distribution shape as `rand_uniform`, different RNG stream).
- Partition `x <= threshold` left; reject if either side `< min_samples_leaf`.
- Keep the candidate with the largest impurity decrease
  (`parent_impurity − weighted_child_impurity`); return `None` if no positive
  decrease, in which case the node becomes a leaf.

Divergences vs sklearn's `node_split_random`: (1) no `FEATURE_THRESHOLD` band
(uses `>=` exact equality); (2) no `threshold == max → min` boundary guard
(ferrolearn's `u < 1` makes `threshold < max` almost surely, but ties at the
max sample value still partition differently); (3) numpy MT19937 vs Rust StdRng
stream; (4) ferrolearn does not continue past `max_features` to guarantee a
valid partition — it leafs instead. All of these are subsumed by the RNG
boundary for *exact* parity but (1) and (4) are also observable on
*deterministic* edge inputs and are pinned separately.

### Criteria (`_criterion.pyx`, shared with decision_tree)

ExtraTree inherits the exact same `CRITERIA_CLF`/`CRITERIA_REG` as the CART
base classes. Classification: `gini` (`:695`), `entropy` (`:622`), `log_loss`
(alias of entropy). Regression: `squared_error`/MSE (`:1090`), `friedman_mse`,
`absolute_error` (median leaves), `poisson` (`:1580`). ferrolearn's
`ClassificationCriterion`/`RegressionCriterion` (shared from `decision_tree.rs`)
carry all of these variants, but extra_tree.rs's private `fn compute_impurity`
covers only `{Gini, Entropy}` (the LogLoss compile gap, REQ-1), and the
regression finders are hard-wired to MSE/variance — `friedman_mse`,
`absolute_error`, `poisson` are silently ignored even when set via
`with_criterion` (REQ-1).

### Fitted attributes

sklearn: `classes_`, `n_classes_`, `max_features_`, `feature_importances_`,
`n_features_in_`, `tree_`. ferrolearn exposes `fn classes`/`fn n_classes`
(`HasClasses`), `fn feature_importances` (`HasFeatureImportances`,
normalized to sum 1 via `fn compute_feature_importances`), `fn nodes`,
`fn n_features`. No `max_features_` fitted attribute; `feature_importances_`
values are not oracle-pinned.

## ferrolearn (what exists)

- **Unfitted**: `pub struct ExtraTreeClassifier<F>` / `pub struct
  ExtraTreeRegressor<F>` with public fields `max_depth`, `min_samples_split`,
  `min_samples_leaf`, `max_features`, `criterion`, `random_state`; builder
  setters `with_*`; `Default`/`fn new`.
- **Fitted**: `pub struct FittedExtraTreeClassifier<F>` (nodes, classes,
  n_features, feature_importances) / `pub struct FittedExtraTreeRegressor<F>`.
- **Traits**: `Fit<Array2<F>, Array1<usize>>` (clf) / `Fit<Array2<F>,
  Array1<F>>` (reg); `Predict`; `HasFeatureImportances`; `HasClasses` (clf);
  `PipelineEstimator`/`FittedPipelineEstimator`.
- **Methods**: `fn predict_proba`, `fn predict_log_proba`, `fn score` (clf);
  `fn score` (reg, R²); `fn nodes`, `fn n_features`.
- **Builder internals**: `fn build_extra_classification_tree`,
  `fn build_extra_regression_tree`, `fn find_random_classification_split`,
  `fn find_random_regression_split`, `fn random_threshold`,
  `fn resolve_max_features`, `fn compute_impurity` (clf, Gini/Entropy only),
  `fn gini_impurity`, `fn entropy_impurity`.
- **Ensemble surface** (`pub(crate)`): `fn
  build_extra_classification_tree_for_ensemble`,
  `fn build_extra_regression_tree_for_ensemble` — consumed by
  `extra_trees_ensemble.rs`.
- **Consumers**: crate re-export (`lib.rs` `pub use extra_tree::{...}`); PyO3
  binding `RsExtraTreeClassifier` (`ferrolearn-python/src/extras.rs`, uses
  `ExtraTreeClassifier::<f64>::new()` + `FittedExtraTreeClassifier<f64>`);
  ensemble builders (`extra_trees_ensemble.rs`).

## Requirements

- REQ-1: **Criteria.** `criterion` accepts the shared `ClassificationCriterion
  {Gini, Entropy, LogLoss}` and `RegressionCriterion {Mse, FriedmanMse,
  AbsoluteError, Poisson}`; impurity is computed by the matching sklearn formula
  (gini `:695`, entropy `:622`, log_loss = entropy alias, MSE `:1090`,
  friedman_mse, MAE-median, poisson `:1580`). The crate must compile (the
  classifier `compute_impurity` match must be exhaustive over the shared enum).
- REQ-2: **Random-split builder.** For each of `max_features` randomly selected
  candidate features, draw a single uniform random threshold in
  `[min, max]` of the feature in the node (`splitter='random'`,
  `node_split_random`, `:674`), apply the `FEATURE_THRESHOLD=1e-7`
  constant-feature band (`:773`) and the `threshold==max→min` guard (`:791`),
  reject children `< min_samples_leaf`, and keep the best impurity decrease
  among the random candidates.
- REQ-3: **Param surface & defaults.** Constructor params + defaults match
  sklearn: ExtraTreeClassifier `criterion='gini'`, `max_features='sqrt'`,
  `splitter='random'`; ExtraTreeRegressor `criterion='squared_error'`,
  `max_features=1.0`; both `max_depth=None`, `min_samples_split=2`,
  `min_samples_leaf=1`, `random_state=None`. Plus `min_weight_fraction_leaf`,
  `max_leaf_nodes`, `min_impurity_decrease`, `class_weight`, `ccp_alpha`,
  `monotonic_cst` (inherited from the CART base classes).
- REQ-4: **`max_features` resolution.** `{int, float, 'sqrt', 'log2', None}` →
  concrete `k`; the search guarantees at least one valid partition even past
  `max_features`; the inferred value is exposed as `max_features_`.
- REQ-5: **Fitted attributes.** `classes_`, `n_classes_`,
  `feature_importances_` (normalized to sum 1), `max_features_`, `tree_`/nodes.
- REQ-6: **predict / predict_proba.** Class-label and target prediction via
  leaf traversal; `predict_proba` leaf-frequency arrays summing to 1;
  `predict_log_proba`; multi-output `y`.
- REQ-7: **`random_state` determinism (RNG boundary).** Fitting with a fixed
  `random_state` is reproducible run-to-run; exact node-for-node parity with
  sklearn at a given seed is the documented RNG boundary (numpy MT19937 vs Rust
  StdRng) and is NOT a parity requirement.
- REQ-8: **ferray substrate.** Owned computation on `ferray-core`, not
  `ndarray`/`rand` (R-SUBSTRATE-1).

## Acceptance criteria

- AC-1: `cargo build -p ferrolearn-tree` succeeds (exhaustive `compute_impurity`
  match). Setting `with_criterion(RegressionCriterion::Poisson)` produces
  poisson-deviance splits / mean leaves matching `ExtraTreeRegressor` semantics
  (or is rejected for negative y).
- AC-2: On a deterministic 2-class set where one feature is constant within band
  `1e-8` (below FEATURE_THRESHOLD), the constant feature is skipped exactly as
  sklearn's `node_split_random` (`tree_.node_count` and root feature match a
  live `ExtraTreeClassifier(random_state=0).fit(...)`).
- AC-3: `inspect.signature(ExtraTreeClassifier)` defaults and the
  `ExtraTreeRegressor` defaults equal the ferrolearn `new()` field values
  (`max_features='sqrt'` clf / `1.0` reg verified live).
- AC-4: `resolve_max_features('sqrt', n)` etc. match
  `max(1, int(...))`/`ceil(sqrt)` semantics and the fitted `max_features_`.
- AC-5: `feature_importances_` on a fixed-seed fit matches
  `ExtraTreeClassifier(random_state=s).fit(...).feature_importances_` within a
  documented tolerance OR is declared RNG-boundary.
- AC-6: `predict_proba` rows sum to 1 and equal sklearn's leaf frequencies on a
  fixed-seed fit (or RNG-boundary).
- AC-7: Two `fit` calls with the same `random_state` produce identical trees
  (`test_extra_classifier_deterministic`, `test_extra_regressor_deterministic`).
- AC-8: `rg "use ndarray|use rand" extra_tree.rs` is empty (ferray substrate).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (criteria — gini/entropy/log_loss + mse/friedman_mse/absolute_error/poisson) | SHIPPED | Classifier: `fn compute_impurity` covers `{Gini, Entropy \| LogLoss}` (log_loss = entropy alias, mirroring `decision_tree.rs`), so the crate compiles (#680 closed). Regression: `RegressionData` carries a `criterion: RegressionCriterion` field threaded from `ExtraTreeRegressor::fit`; `fn build_extra_regression_tree` computes the leaf value via `fn regression_leaf_value` (median for `AbsoluteError` via `fn median_value`, mean for MSE/FriedmanMSE/Poisson) and `fn find_random_regression_split` scores the random candidate threshold per-criterion (variance for MSE, the Friedman improvement proxy for FriedmanMSE, L1 `fn mae_for_indices` for AbsoluteError, half-deviance `fn poisson_deviance_for_indices` for Poisson) — mirroring `decision_tree.rs`'s `fn regression_leaf_value`/`fn regression_node_impurity`/`fn find_best_regression_split`. Poisson rejects negative y / Σy≤0 in `ExtraTreeRegressor::fit` (mirror `decision_tree.rs`). MSE path kept numerically identical (default). **Verified**: `pin3_reg_absolute_error_differs_from_mse_same_seed` + `pin3_reg_absolute_error_yields_median_leaf` (`tests/divergence_extra_tree.rs`) green — `AbsoluteError` yields median leaf 1.0 distinct from MSE's mean 2.333 at a fixed `random_state` (#681 closed). **Note**: the ensemble path (`fn build_extra_regression_tree_for_ensemble`) still hard-pins `RegressionCriterion::Mse` (preserving `ExtraTreesRegressor`'s existing behavior); per-criterion ensemble support spans `extra_trees_ensemble.rs` and is out of this single-file fix's scope. |
| REQ-2 (random-split builder — bounds/band/guard) | NOT-STARTED | impl `fn find_random_classification_split`/`fn find_random_regression_split` in `extra_tree.rs` draw one uniform threshold per candidate feature via `fn random_threshold` (`min + u*(max-min)`, `u ∈ [0,1)`) and keep the best impurity decrease — the *distribution/bounds* mirror `rand_uniform` (`utils/_random.pxd:52`) / `node_split_random` (`_splitter.pyx:674`). But the constant-feature test uses exact `feat_min >= feat_max` instead of sklearn's `max <= min + FEATURE_THRESHOLD` (`_splitter.pyx:773`, `1e-7` band), there is no `threshold==max → min` boundary guard (`:791`), and ferrolearn leafs rather than continuing past `max_features` to guarantee a valid partition. None of these is oracle-pinned (existing tests assert only label separation / node count on hand inputs, not a live `ExtraTree*` `tree_`). open prereq blockers #682 (FEATURE_THRESHOLD band + threshold==max guard), #683 (deterministic-input split-structure oracle pin once crate compiles). |
| REQ-3 (param surface & defaults) | SHIPPED | impl: `pub struct ExtraTreeClassifier<F>`/`ExtraTreeRegressor<F>` fields + `fn new` defaults in `extra_tree.rs` — classifier `max_features = MaxFeatures::Sqrt`, `criterion = Gini`; regressor `max_features = MaxFeatures::All`, `criterion = Mse`; both `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `random_state=None`. Matches sklearn `_classes.py:1564` (clf) / `:1838` (reg). **Verified live**: `ExtraTreeClassifier().max_features == 'sqrt'`, `ExtraTreeRegressor().max_features == 1.0`, both `splitter == 'random'`. Non-test consumer: crate re-export (`lib.rs` `pub use extra_tree::{ExtraTreeClassifier, ExtraTreeRegressor, ...}`) + PyO3 binding `RsExtraTreeClassifier` (`ferrolearn-python/src/extras.rs`, `ExtraTreeClassifier::<f64>::new()`). Tests: `test_classifier_default`, `test_regressor_default`, `test_classifier_builder_methods`, `test_regressor_builder_methods`. **Caveat**: `min_weight_fraction_leaf`, `max_leaf_nodes`, `min_impurity_decrease`, `class_weight`, `ccp_alpha`, `monotonic_cst` are absent (tracked under #684); the params that exist match sklearn. (Verification blocked until #680 lets the crate compile.) |
| REQ-4 (max_features resolution) | NOT-STARTED (partial) | impl `fn resolve_max_features` maps `Sqrt → ceil(sqrt(n))`, `Log2 → ceil(log2(n)).max(1)`, `All → n`, `Fixed(k)`, `Fraction(f) → ceil(n*f)`, clamped to `[1, n]`. sklearn uses `max(1, int(sqrt(n)))` / `max(1, int(log2(n)))` / `max(1, int(f*n))` (`_classes.py` `_validate_params`) — `ceil` vs `int(...)` is a potential off-by-one divergence (e.g. n=2: ceil(sqrt 2)=2 vs int=1), not oracle-pinned. No `max_features_` fitted attribute; no "continue past max_features until a valid partition" guarantee. open prereq blockers #685 (resolve_max_features ceil-vs-int oracle pin), #686 (max_features_ fitted attr + valid-partition guarantee). |
| REQ-5 (fitted attributes) | SHIPPED (partial) | impl: `fn feature_importances` (`HasFeatureImportances`, normalized sum-1 via `fn compute_feature_importances`, consumed by `extra_trees_ensemble.rs` importance aggregation + the PyO3 binding), `fn classes`/`fn n_classes` (`HasClasses`), `fn nodes`, `fn n_features` are wired and consumed (re-export + ensemble + binding). Tests: `test_extra_classifier_classes`, `test_extra_classifier_feature_importances`, `test_extra_regressor_feature_importances`. **But** `feature_importances_` is asserted only `sum==1` + `[0]>[1]`, not pinned to a live `ExtraTree*.feature_importances_` (RNG-dependent → AC-5 boundary), and there is no `max_features_`. open prereq blockers #686 (max_features_), #687 (feature_importances_ numeric pin / RNG-boundary classification). (Accessors blocked from running until #680.) |
| REQ-6 (predict / predict_proba) | SHIPPED (partial) | impl `fn predict` (clf: leaf majority class; reg: leaf mean), `fn predict_proba` (leaf class-frequency, rows sum to 1), `fn predict_log_proba`, in `extra_tree.rs`, via `fn traverse` shared from `decision_tree.rs`. Consumed by PyO3 binding `RsExtraTreeClassifier` (`extras.rs`) and the pipeline adapter (`FittedExtraTreeClassifierPipelineAdapter::predict_pipeline`). Tests: `test_extra_classifier_predict_proba` (rows sum to 1), `test_extra_classifier_simple_binary`, `test_extra_regressor_simple`. **But** `predict_proba` arrays are not pinned to sklearn leaf frequencies (RNG-dependent), and multi-output `y` (`Fit<Array2<F>, Array1<_>>` only) is unsupported. open prereq blockers #688 (predict_proba pin / RNG-boundary), #689 (multi-output). (Blocked from running until #680.) |
| REQ-7 (random_state determinism — RNG boundary) | SHIPPED | impl: `random_state: Option<u64>` seeds `StdRng::seed_from_u64` in both `fit` impls (`extra_tree.rs`); same seed ⇒ identical RNG stream ⇒ identical tree. Tests: `test_extra_classifier_deterministic`, `test_extra_regressor_deterministic` (two fits at the same seed produce equal predictions). **Documented RNG boundary**: exact node-for-node parity with sklearn at a given `random_state` is INFEASIBLE — numpy MT19937 (`rand_uniform`, `utils/_random.pxd:52`) vs Rust `StdRng` are different streams, the same boundary accepted for the SGD shuffle, libsvm CV folds, and decision_tree's RNG tie-break (#670). This REQ ships *reproducibility*, not *cross-impl bit-parity*. Consumer: `with_random_state` on the re-exported types + the PyO3 binding's `random_state` kwarg. (Blocked from running until #680.) |
| REQ-8 (ferray substrate) | NOT-STARTED | impl: module imports `use ndarray::{Array1, Array2}` and `use rand::{SeedableRng, ...}` / `StdRng` — the wrong substrate per R-SUBSTRATE-1 (array → `ferray-core`, RNG → `ferray::random`). No `ferray` usage. open prereq blocker #690 (migrate extra_tree.rs array ops to `ferray-core` and the random-threshold draw to `ferray::random`, jointly with `decision_tree.rs` #671 and `extra_trees_ensemble.rs`). |

## Architecture

`extra_tree.rs` deliberately reuses the CART substrate from `decision_tree.rs`:
the `Node<F>` enum (Split/Leaf, flat `Vec<Node<F>>`, index-0 root), `TreeParams`
(max_depth/min_samples_split/min_samples_leaf), `fn traverse` (leaf lookup),
`fn compute_feature_importances` (normalized weighted impurity decrease), and
the criterion enums. The **only** algorithmic divergence from
`decision_tree.rs` is the split *finder*: `fn find_random_classification_split`/
`fn find_random_regression_split` draw a single random threshold per candidate
feature (`fn random_threshold`) instead of scanning sorted midpoints — exactly
mirroring sklearn's `RandomSplitter` vs `BestSplitter` distinction
(`_splitter.pyx::node_split_random:674` vs `node_split_best:293`).

The `Fitted*` structs and all trait impls are structurally identical to the
decision-tree counterparts; predict/predict_proba/feature_importances are
literally the same traversal code, differing only in which `nodes` vector was
produced. The ensemble crate (`extra_trees_ensemble.rs`) calls
`fn build_extra_*_for_ensemble` directly (passing a per-tree `feature_indices`
subset and a seeded `StdRng`), which is why those builders are `pub(crate)`.

The shared criterion enum is the source of the compile gap: because
`decision_tree.rs` owns `ClassificationCriterion` and added `LogLoss`, every
`match` on it elsewhere must be exhaustive. extra_tree.rs's private
`fn compute_impurity` was not updated, so the crate fails to build — the
prerequisite for any verification (#680).

## Verification

Once #680 lands (crate compiles), the SHIPPED claims are established by:

```
cargo test -p ferrolearn-tree            # extra_tree unit tests (clf/reg)
cargo clippy -p ferrolearn-tree --all-targets -- -D warnings
cargo fmt --all --check
```

Live sklearn oracle anchors (for the NOT-STARTED parity REQs once a critic pins
them — RNG-boundary REQs get a *distribution*/*structure* pin, not a bit pin):

```
python3 -c "from sklearn.tree import ExtraTreeClassifier; print(ExtraTreeClassifier().max_features, ExtraTreeClassifier().splitter)"   # sqrt random  (REQ-3, AC-3)
python3 -c "from sklearn.tree import ExtraTreeRegressor; print(ExtraTreeRegressor().max_features)"                                    # 1.0          (REQ-3, AC-3)
python3 -c "from sklearn.tree import ExtraTreeClassifier; ... constant-band X; print(m.tree_.node_count, m.tree_.feature[0])"          # REQ-2 / AC-2
```

PyO3 path: `cd ferrolearn-python && maturin develop && PYTHONPATH=python python3
-m pytest tests/ -q` comparing `ferrolearn.ExtraTreeClassifier` against
`sklearn.tree.ExtraTreeClassifier` on the *deterministic* surface (defaults,
error paths, predict shape) — never on exact fitted-tree values (RNG boundary).

Until #680 lands, **all** verification commands fail (the crate does not
compile); every REQ classified SHIPPED above is SHIPPED in *structure*
(impl + consumer + test present) but its test is currently un-runnable — the
compile fix is the gate.

## Blockers to open

- **#680** (top priority, unblocks the crate): LogLoss arm missing in
  `fn compute_impurity` (`extra_tree.rs:798`) → `E0004` non-exhaustive match →
  workspace does not compile. One-line fix `Entropy | LogLoss => entropy_impurity`
  (mirror `decision_tree.rs:1012`).
- **#681**: regression criteria — `friedman_mse`/`absolute_error`/`poisson` are
  silently ignored (finders are MSE-only); add per-criterion dispatch + the
  poisson non-negative-y guard (mirror `decision_tree.rs`).
- **#682**: random-split builder — `FEATURE_THRESHOLD=1e-7` constant band
  (`_splitter.pyx:773`) and `threshold==max → min` guard (`:791`) missing.
- **#683**: deterministic-input split-structure oracle pin vs live `ExtraTree*`
  (once crate compiles).
- **#684**: missing params `min_weight_fraction_leaf`, `max_leaf_nodes`,
  `min_impurity_decrease`, `class_weight`, `ccp_alpha`, `monotonic_cst`.
- **#685**: `resolve_max_features` `ceil` vs sklearn `int(...)` off-by-one pin.
- **#686**: `max_features_` fitted attribute + "continue past max_features until
  a valid partition" guarantee.
- **#687**: `feature_importances_` numeric pin or RNG-boundary classification.
- **#688**: `predict_proba` leaf-frequency pin or RNG-boundary classification.
- **#689**: multi-output `y` support.
- **#690**: ferray substrate migration (array → `ferray-core`, RNG →
  `ferray::random`), jointly with decision_tree #671.

## Top 3 suspected divergences (director / critic's first pins)

1. **LogLoss compile gap (#680) — blocks everything.** `fn compute_impurity` in
   extra_tree.rs is non-exhaustive over the shared `ClassificationCriterion`
   after decision_tree's LogLoss addition; `cargo build -p ferrolearn-tree`
   fails with `E0004` at `extra_tree.rs:798`. The director needs the one-line
   `Entropy | LogLoss => entropy_impurity` fix to restore the build.
2. **Regression criteria ignored (#681).** `with_criterion(FriedmanMse |
   AbsoluteError | Poisson)` compiles but the finders compute plain MSE/variance
   — friedman_mse/MAE-median/poisson are silently dropped, diverging from
   `ExtraTreeRegressor(criterion=...)`.
3. **Constant-feature band + boundary guard (#682).** ferrolearn skips a feature
   only on exact `min >= max`, not sklearn's `max <= min + 1e-7`, and lacks the
   `threshold==max → min` guard — observable on near-constant features even
   before invoking the RNG boundary.
