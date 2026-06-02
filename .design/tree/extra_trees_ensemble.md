# Extra-Trees Ensembles (ExtraTreesClassifier / ExtraTreesRegressor)

<!--
tier: 3-component
status: draft
baseline-commit: 95a167e6
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/ensemble/_forest.py     # ForestClassifier.predict (:883, SOFT vote = argmax of mean predict_proba, :904-907); ForestClassifier.predict_proba (:922, mean of per-tree predict_proba, :962-963); ForestRegressor.predict (:1042, mean of per-tree predict, :1081); feature_importances_ (BaseForest, mean over trees then /sum); _generate_sample_indices (:128, randint(0,n,n_bootstrap)); _generate_unsampled_indices (:140); ExtraTreesClassifier (:1914), ExtraTreesClassifier.__init__ (:2224, criterion='gini' :2228, max_features='sqrt' :2233, bootstrap=False :2236); ExtraTreesRegressor (:2285), ExtraTreesRegressor.__init__ (:2564, criterion='squared_error' :2568, max_features=1.0 :2573, bootstrap=False)
  - sklearn/tree/_classes.py        # the ExtraTree* base learner built per estimator (splitter='random') — shared via extra_tree.rs
ferrolearn-module: ferrolearn-tree/src/extra_trees_ensemble.rs
parity-ops: ExtraTreesClassifier, ExtraTreesRegressor
crosslink-issue: 678
-->

## Summary

`ferrolearn-tree/src/extra_trees_ensemble.rs` mirrors scikit-learn's
`sklearn.ensemble.ExtraTreesClassifier` (`_forest.py:1914`) and
`ExtraTreesRegressor` (`_forest.py:2285`) — forests of **extremely randomized
trees**. Each member is an `ExtraTree*` base learner (`splitter='random'`: one
random threshold drawn per random candidate feature at each node), and unlike
`RandomForest*`, **bootstrap sampling is disabled by default** (`bootstrap=False`,
`_forest.py:2236`): every tree sees all rows, so the ensemble's randomness comes
solely from the random split thresholds + random feature subsets. Aggregation is
shared with `RandomForest*` because both subclass `ForestClassifier` /
`ForestRegressor`: the classifier predicts via **soft voting** — `argmax` of the
*mean* of per-tree `predict_proba` (`predict`, `:904-907`; `predict_proba`,
`:962-963`); the regressor predicts the *mean* of per-tree outputs (`:1081`).
`feature_importances_` is the normalized mean of per-tree importances.

ferrolearn re-implements this natively. The module ships unfitted
`ExtraTreesClassifier<F>` / `ExtraTreesRegressor<F>` and their `Fitted*`
counterparts, reuses `MaxFeatures` + `fn resolve_max_features` (from
`random_forest.rs`) + `fn make_tree_params`, parallel (`rayon`) per-tree build
delegating to `extra_tree.rs`'s crate-public
`fn build_extra_classification_tree_for_ensemble` /
`fn build_extra_regression_tree_for_ensemble`, an optional
`bootstrap` toggle (default `false`) and `n_jobs`, `Fit`/`Predict`,
`predict_proba`/`predict_log_proba`/`score` (clf) and `score` (reg),
`feature_importances_`, and pipeline adapters. The per-tree build correctness
(random-threshold splitter, criteria, leaf values, importances) is **inherited
from `extra_tree.rs`** (`.design/tree/extra_tree.md`).

**An extra-trees forest is inherently RNG-driven.** The random split thresholds
(and the optional bootstrap draw) are seeded by `random_state`; numpy's MT19937
stream (sklearn) and Rust's `StdRng` (`next_u64`, ferrolearn) cannot bit-match,
so **exact ensemble-for-ensemble parity at a given `random_state` is a documented
RNG boundary** — the same boundary already accepted for `extra_tree.rs`'s random
threshold, `random_forest.rs`'s bootstrap, the SGD shuffle, and the libsvm
CV-fold RNG. The **deterministic** contract — the param/default surface, and the
*aggregation logic* given a fixed set of trees (soft-vote mean→argmax, regressor
mean, importance mean-normalization) — is the shippable/pinnable part; the exact
tree ensemble at a seed is not.

**Top divergence the director must see — the classifier votes HARD, not SOFT.**
`FittedExtraTreesClassifier::predict` (`predict` in `extra_trees_ensemble.rs`)
tallies **per-tree predicted labels** (`votes[class_idx] += 1`, one vote per
tree) and returns the `argmax` of those integer vote counts — a **hard majority
vote**. sklearn `ExtraTreesClassifier.predict` (inherited from
`ForestClassifier.predict`, `:904-907`) returns the `argmax` of the **mean of
per-tree `predict_proba`** — a **soft vote** weighted by leaf class fractions.
These differ whenever trees emit non-degenerate leaf distributions: a live oracle
on 200 shallow (`max_depth=2`) samples shows `predict == soft-argmax` (True) but
`predict == hard-majority` (False), with the two voting schemes disagreeing on
**54/200** rows. ferrolearn's own `predict_proba` *does* average per-tree
distributions correctly (soft), so its `predict` is internally inconsistent with
its `predict_proba` — the fix is to make `predict` `argmax` over `predict_proba`
(REQ-3). This is the **same divergence class** as `random_forest.rs`'s just-fixed
#670 hard-vote bug, NOT an RNG-boundary artifact.

This doc adapts to the **existing** code. Under R-HONEST-3 no oracle pins exist
yet and the ensemble is RNG-driven, so: the param/default REQ for the params
ferrolearn *has* is SHIPPED with the absent params flagged; the deterministic
aggregation REQs are SHIPPED for the regressor mean / `predict_proba` mean /
importance normalization but the classifier `predict` soft-vote REQ is
NOT-STARTED (hard-vote divergence); and every numpy-parity, missing-param, oob,
class_weight, regressor-criterion, and substrate REQ is NOT-STARTED with a
concrete blocker.

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `get_params()`, sklearn 1.5.2)

`ExtraTreesClassifier` (`_forest.py:1914`, `__init__` `:2224`):
`n_estimators=100`, `criterion='gini'`, `max_depth=None`,
`min_samples_split=2`, `min_samples_leaf=1`, `min_weight_fraction_leaf=0.0`,
`max_features='sqrt'`, `max_leaf_nodes=None`, `min_impurity_decrease=0.0`,
`bootstrap=False`, `oob_score=False`, `n_jobs=None`, `random_state=None`,
`verbose=0`, `warm_start=False`, `class_weight=None`, `ccp_alpha=0.0`,
`max_samples=None`, `monotonic_cst=None`.

`ExtraTreesRegressor` (`_forest.py:2285`, `__init__` `:2564`): same block with
`criterion='squared_error'` (`:2568`), `max_features=1.0` (`:2573`, = all
features), and no `class_weight`.

**The asymmetry vs RandomForest** is the `bootstrap=False` default (`:2236`) —
extra-trees do not bootstrap unless explicitly asked. ferrolearn matches this:
`ExtraTreesClassifier::new()` / `ExtraTreesRegressor::new()` both set
`bootstrap = false`.

**The defaults ferrolearn matches:** `n_estimators=100`, `max_depth=None`,
`min_samples_split=2`, `min_samples_leaf=1`, `bootstrap=false`,
`random_state=None`, `n_jobs=None`; clf `max_features='sqrt'` →
`MaxFeatures::Sqrt`, `criterion='gini'` → `ClassificationCriterion::Gini`; reg
`max_features=1.0` → `MaxFeatures::All`.

**Params PRESENT in ferrolearn** (unlike `random_forest.rs`): `bootstrap`
(`with_bootstrap`, a genuine toggle, default `false`); `n_jobs` (`with_n_jobs`,
a `rayon` thread-pool count); `criterion` on the classifier (`with_criterion`).

**Params ABSENT in ferrolearn** (REQ-1 flags each): `criterion` on the
**regressor** (no `with_criterion` — `ExtraTreesRegressor` always uses MSE, and
even the classifier's `criterion` flows only to the per-tree build);
`max_samples` (bootstrap, when on, always draws exactly `n_samples`);
`oob_score`/`oob_decision_function_`/`oob_prediction_` (no OOB at all);
`class_weight` / `'balanced_subsample'` (clf has no weighting);
`max_leaf_nodes`, `min_impurity_decrease`, `min_weight_fraction_leaf`,
`ccp_alpha`, `monotonic_cst` (tree params not threaded through, even though
`extra_tree.rs`/`decision_tree.rs` support several of them);
`warm_start`, `verbose` (Python ergonomics, R-DEV-4 — not divergences).

### Optional bootstrap sampling (`bootstrap=False` default, `_generate_sample_indices` `:128`)

`ExtraTreesClassifier`/`Regressor` default `bootstrap=False`: each tree is fit on
the **entire** dataset. When `bootstrap=True`, sklearn draws
`random_instance.randint(0, n_samples, n_samples_bootstrap)` (`:133`) indices
uniformly with replacement (`n_samples_bootstrap = n_samples` by default).

ferrolearn (`fn build_single_classification_tree` /
`fn build_single_regression_tree`): when `bootstrap` is `false` (the default) the
indices are `(0..n_samples).collect()` — all rows, matching sklearn's
no-bootstrap path. When `bootstrap` is `true`, draws exactly `n_samples` indices
via `(rng.next_u64() as usize) % n_samples` per draw (`StdRng` seeded from a
per-tree seed). Divergences in the bootstrap path: (1) numpy MT19937 vs `StdRng`
(RNG boundary, REQ-2); (2) modulo reduction is slightly non-uniform vs numpy's
rejection-sampled `randint` (subsumed by the RNG boundary); (3) no `max_samples`
support — the bootstrap size is fixed at `n_samples`. The **default**
(no-bootstrap) path is deterministic w.r.t. the sample set and matches sklearn
exactly.

### Per-tree fit (each member is an `ExtraTree*`)

Each estimator is an `ExtraTree*` (`splitter='random'`): at each node, for each
of `max_features` random candidate features a single random threshold is drawn
uniformly in `[min, max]` and the best random candidate is kept. ferrolearn seeds
per-tree masters sequentially from `random_state` (for thread-order determinism)
then dispatches `tree_seeds.par_iter()` to
`build_extra_classification_tree_for_ensemble` /
`build_extra_regression_tree_for_ensemble` (in `extra_tree.rs`), each given a
fresh `StdRng::seed_from_u64(seed)`. The tree-build *correctness* is inherited
from `extra_tree.rs` (REQ-2b) — which itself carries the random-threshold
divergences (FEATURE_THRESHOLD band, `threshold==max→min` guard) and RNG boundary
documented in `.design/tree/extra_tree.md`. Note: the ensemble regression builder
hard-pins `RegressionCriterion::Mse` (per `extra_tree.md` REQ-1's note), so the
regressor `criterion` is structurally fixed (REQ-1).

### Aggregation (shared with RandomForest via Forest* base classes)

- **Classifier `predict`** (`ForestClassifier.predict`, `:904-907`):
  `classes_.take(argmax(predict_proba, axis=1))` — **soft vote**.
  `predict_proba` (`:962-963`) = `(sum over trees of tree.predict_proba(X)) /
  n_estimators`; each tree's `predict_proba` is the per-leaf class fraction.
  ferrolearn's `fn predict_proba` matches this (sums per-tree leaf
  `class_distribution`, divides by `n_trees`). ferrolearn's `fn predict` does
  **not** — it hard-votes one label per tree via `votes[class_idx] += 1` /
  `max_by_key` and returns `classes[winner]` (REQ-3).
- **Regressor `predict`** (`ForestRegressor.predict`, `:1081`): `(sum over trees
  of tree.predict(X)) / n_estimators` — plain mean. ferrolearn matches
  (`sum / n_trees_f` over leaf `value`). Deterministic given a fixed forest,
  pinnable intra-ferrolearn (REQ-4).

### feature_importances_ (mean over trees then normalize)

sklearn (`BaseForest.feature_importances_`): gather `tree.feature_importances_`
for every tree with `tree_.node_count > 1`, take `np.mean(..., axis=0)`, then
divide by the sum (returns zeros if all trees are stumps). ferrolearn (`fit`,
both): `sum` the per-tree `compute_feature_importances` across **all** trees then
divide by the total. Sum-then-normalize and mean-then-normalize are
**algebraically identical** when every tree is counted (stumps contribute an
all-zero vector), so the values match — EXCEPT sklearn excludes single-node trees
from the mean and returns zeros via an `if not all_importances` guard when every
tree is a stump (observable only in the all-stumps edge case). REQ-5 ships the
normalization and pins the edge case.

### oob_score_ / oob_decision_function_ / oob_prediction_

When `oob_score=True` (only valid if `bootstrap=True`), sklearn scores each
sample using only the trees for which it was out-of-bag
(`_generate_unsampled_indices`, `:140`) and exposes `oob_score_` +
`oob_decision_function_` (clf) / `oob_prediction_` (reg). ferrolearn has **no OOB
machinery** (REQ-6, NOT-STARTED).

### class_weight / 'balanced_subsample' (clf)

sklearn supports `class_weight in {None, 'balanced', 'balanced_subsample', dict,
list-of-dict}`. ferrolearn's extra-trees classifier has **no** class weighting
(REQ-7, NOT-STARTED).

## ferrolearn (what exists)

- **Unfitted**: `pub struct ExtraTreesClassifier<F>` (public fields
  `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`,
  `max_features`, `bootstrap`, `criterion`, `random_state`, `n_jobs`) / `pub
  struct ExtraTreesRegressor<F>` (same minus `criterion`); builder setters
  `with_n_estimators`/`with_max_depth`/`with_min_samples_split`/
  `with_min_samples_leaf`/`with_max_features`/`with_bootstrap`/`with_criterion`
  (clf only)/`with_random_state`/`with_n_jobs`; `Default`/`fn new`.
- **Fitted**: `pub struct FittedExtraTreesClassifier<F>` (trees, classes,
  n_features, feature_importances) / `pub struct FittedExtraTreesRegressor<F>`.
- **Reused enum/helper**: `MaxFeatures` (imported from `random_forest.rs`);
  `fn resolve_max_features`; `fn make_tree_params`.
- **Traits**: `Fit<Array2<F>, Array1<usize>>` (clf) / `Fit<Array2<F>,
  Array1<F>>` (reg); `Predict`; `HasFeatureImportances`; `HasClasses` (clf);
  `PipelineEstimator`/`FittedPipelineEstimator`.
- **Methods**: `fn predict_proba`, `fn predict_log_proba`, `fn score` (clf);
  `fn score` (reg, R²); `fn trees`, `fn n_features`, `fn n_estimators`.
- **Per-tree dispatch**: `fn build_single_classification_tree` /
  `fn build_single_regression_tree` (do the optional bootstrap draw, then call
  `extra_tree.rs`'s `fn build_extra_classification_tree_for_ensemble` /
  `fn build_extra_regression_tree_for_ensemble`).
- **Build delegation** (from `decision_tree.rs` / `extra_tree.rs`):
  `Node<F>`, `TreeParams`, `ClassificationCriterion`, `fn traverse`,
  `fn compute_feature_importances`.
- **Consumers**: crate re-export (`lib.rs` `pub use extra_trees_ensemble::{
  ExtraTreesClassifier, ExtraTreesRegressor, FittedExtraTreesClassifier,
  FittedExtraTreesRegressor}`); PyO3 bindings `RsExtraTreesClassifier` /
  `RsExtraTreesRegressor` (`ferrolearn-python/src/extras.rs`, use
  `ExtraTreesClassifier::<f64>::new()` / `ExtraTreesRegressor::<f64>::new()` +
  the `Fitted*` types, registered in `ferrolearn-python/src/lib.rs`); the
  pipeline adapters.

## Requirements

- REQ-1: **Param surface + defaults (R-DEV-2).** `n_estimators=100`,
  `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`,
  `bootstrap=false`, `random_state=None`, `n_jobs=None`, clf
  `max_features='sqrt'`+`criterion='gini'`, reg `max_features=1.0` match
  sklearn's `get_params()`. ABSENT: regressor `criterion`, `max_samples`,
  `oob_score`, `class_weight`, `max_leaf_nodes`, `min_impurity_decrease`,
  `min_weight_fraction_leaf`, `ccp_alpha`, `monotonic_cst`.
- REQ-2: **Optional bootstrap sampling.** `bootstrap=False` default → all rows
  per tree (matches sklearn exactly); `bootstrap=True` → with-replacement
  `n_samples` draws. `max_samples` sizing absent. RNG boundary (numpy MT vs
  StdRng) for the bootstrap-on path.
- REQ-2b: **Per-tree fit** — each member is an `ExtraTree*` (random-threshold
  splitter) on the (optionally bootstrapped) sample with `max_features` per-split
  random subsampling; correctness inherited from `extra_tree.rs`. RNG boundary
  for the exact ensemble.
- REQ-3: **Classifier aggregation = SOFT vote.** `predict` is `argmax` of the
  *mean of per-tree `predict_proba`* (sklearn `:904-907`), NOT a hard
  per-tree-label majority. `FittedExtraTreesClassifier::predict` must route
  through `predict_proba` and take the per-row lowest-index argmax mapped to
  `classes_[idx]` (matches `np.argmax` tie-break).
- REQ-4: **Classifier `predict_proba` + regressor `predict` = mean.** clf
  `predict_proba` = mean of per-tree class distributions (`:962-963`); reg
  `predict` = mean of per-tree outputs (`:1081`). Deterministic given the forest.
- REQ-5: **`feature_importances_` = normalized mean of per-tree importances**,
  with the all-stumps edge case returning zeros.
- REQ-6: **`oob_score_` / `oob_decision_function_` / `oob_prediction_`**
  (`:140`).
- REQ-7: **`class_weight` (+ `'balanced_subsample'`)** for the classifier.
- REQ-8: **Regressor `criterion`** — `ExtraTreesRegressor` should accept
  `criterion in {squared_error, friedman_mse, absolute_error, poisson}`
  (`:2568`); ferrolearn has no `with_criterion` and the ensemble builder
  hard-pins MSE.
- REQ-9: **`random_state` determinism** — same seed → identical forest
  (ferrolearn reproducibility, NOT numpy parity). RNG boundary.
- REQ-10: **ferray substrate (R-SUBSTRATE).** Imports `ndarray`/`rand`/`rayon`,
  not `ferray-core`/`ferray::random`.

## Acceptance criteria

- AC-1: live `ExtraTreesClassifier().get_params()` /
  `ExtraTreesRegressor().get_params()` equal the defaults in REQ-1 for the params
  ferrolearn exposes (including `bootstrap=False`); absent params are enumerated.
- AC-2: with `bootstrap=False` (default) each tree sees `(0..n_samples)`;
  toggling `with_bootstrap(true)` draws `n_samples` with-replacement indices
  (`test_ensemble_classifier_with_bootstrap`, `_regressor_with_bootstrap`).
- AC-3: on 200 shallow (`max_depth=2`) samples, sklearn `predict` equals
  `classes_.take(argmax(predict_proba))` (soft) and differs from a hard
  per-tree-label majority on `>0` rows — establishing the divergence class
  (live: 54/200). ferrolearn `predict` must agree with `argmax(predict_proba)`
  on the same fixed forest.
- AC-4: regressor `predict` equals `mean` of `t.predict(X)` over
  `t in estimators_` (sklearn `:1081`); ferrolearn matches to 1e-12 on a fixed
  intra-ferrolearn forest.
- AC-5: `predict_proba` rows sum to 1 and equal the mean of per-tree
  distributions on a fixed forest.
- AC-6: `feature_importances_` sums to 1 (or is all-zeros when every tree is a
  stump) and equals the normalized mean of per-tree importances.
- AC-7: `random_state` reproducibility — two `fit` calls with the same seed
  produce identical `predict` (`test_ensemble_classifier_deterministic` /
  `test_ensemble_regressor_deterministic`).

## REQ status table

Binary (R-DEFER-2). `ExtraTreesClassifier`/`ExtraTreesRegressor` are boundary
estimator types re-exported at the crate root and registered as PyO3
`RsExtraTreesClassifier`/`RsExtraTreesRegressor` (S5/R-DEFER-1 non-test consumer
surface). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (param surface + defaults) | SHIPPED (with gaps flagged) | `fn new` on `ExtraTreesClassifier` (`MaxFeatures::Sqrt`, `ClassificationCriterion::Gini`, `n_estimators=100`, `min_samples_split=2`, `min_samples_leaf=1`, `max_depth=None`, `bootstrap=false`, `n_jobs=None`) and `ExtraTreesRegressor` (`MaxFeatures::All`, `bootstrap=false`) match `get_params()` (`_forest.py:2224`/`:2228`/`:2233`/`:2236`, `:2564`/`:2568`/`:2573`). Consumer: `RsExtraTreesClassifier::new` / `RsExtraTreesRegressor::new` (PyO3, `extras.rs`) + crate re-export (`lib.rs`). Tests: `test_ensemble_classifier_default`, `test_ensemble_regressor_default`, `test_ensemble_classifier_builder`, `test_ensemble_regressor_builder`. Verification: `cd /tmp && python3 -c "from sklearn.ensemble import ExtraTreesClassifier as C,ExtraTreesRegressor as R; print(C().get_params(), R().get_params())"` (matches the exposed subset incl. `bootstrap=False`). ABSENT params (regressor `criterion`, `max_samples`, `oob_score`, `class_weight`, `max_leaf_nodes`, `min_impurity_decrease`, `min_weight_fraction_leaf`, `ccp_alpha`, `monotonic_cst`) → open prereq blocker #691. |
| REQ-2 (optional bootstrap sampling) | SHIPPED (default path) / NOT-STARTED (bootstrap-on parity + max_samples) | Default `bootstrap=false`: `fn build_single_classification_tree`/`fn build_single_regression_tree` use `(0..n_samples).collect()` — matches sklearn's whole-dataset path (`_forest.py:1683-1684`) exactly. Consumer: PyO3 `fit`; tests `test_ensemble_classifier_no_bootstrap`, `test_ensemble_regressor_no_bootstrap`. **But** the `bootstrap=true` branch draws `next_u64() % n_samples` (StdRng), diverging from numpy `randint(0,n,n_bootstrap)` (`_forest.py:128`) — RNG boundary — and there is no `max_samples` sizing. open prereq blockers #692 (max_samples sizing), #693 (bootstrap-on RNG boundary, documented like extra_tree #683-class). |
| REQ-2b (per-tree fit = ExtraTree base learner) | SHIPPED (correctness inherited) | `fit` calls `build_extra_classification_tree_for_ensemble` / `build_extra_regression_tree_for_ensemble` (`extra_tree.rs`, `.design/tree/extra_tree.md`) per seed with `resolve_max_features` per-split subsets and a fresh `StdRng`. Consumer: PyO3 `fit`. Tests: `test_ensemble_classifier_simple`, `test_ensemble_regressor_simple`. Exact tree ensemble at a seed = documented RNG boundary (#693); the base-learner random-threshold divergences (FEATURE_THRESHOLD band, `threshold==max→min`) are owned/tracked in `.design/tree/extra_tree.md` (#682). |
| REQ-3 (classifier SOFT vote) | NOT-STARTED | open prereq blocker #694. `FittedExtraTreesClassifier::predict` (`extra_trees_ensemble.rs`) tallies one hard label per tree (`votes[class_idx] += 1`, `max_by_key`, returns `classes[winner]`) — a hard majority vote. sklearn `predict` (inherited `ForestClassifier.predict`, `_forest.py:904-907`) = `classes_.take(argmax(predict_proba))` = SOFT vote. Live oracle (200 shallow `max_depth=2` samples): `soft==predict` True, `hard==predict` False, **54/200** rows differ. Same divergence class as `random_forest.rs` #670 (just fixed). ferrolearn's own `predict_proba` already averages correctly (soft), so `predict` is internally inconsistent with `predict_proba`; the fix routes `predict` through `argmax(predict_proba)`. |
| REQ-4 (predict_proba mean / regressor mean) | SHIPPED | `fn predict_proba` sums per-tree leaf `class_distribution` then `/ n_trees_f` (mirrors `_forest.py:962-963`); `FittedExtraTreesRegressor::predict` = `sum / n_trees_f` over leaf `value` (mirrors `_forest.py:1081`). Consumer: PyO3 regressor `predict`; clf `predict_proba` getter; pipeline adapters. Tests: `test_ensemble_classifier_predict_proba` (rows sum to 1), `test_ensemble_regressor_simple`, `test_ensemble_regressor_constant_target` (mean leaf value). Deterministic given the forest. |
| REQ-5 (feature_importances_ normalized mean) | SHIPPED (stump edge pinnable) | `fit` (both) accumulates `compute_feature_importances` over trees then divides by the sum (algebraically equal to sklearn's mean-then-normalize); `HasFeatureImportances::feature_importances`. Consumer: PyO3 `RsExtraTreesClassifier`/`RsExtraTreesRegressor` getters + crate re-export. Tests: `test_ensemble_classifier_feature_importances`, `test_ensemble_regressor_feature_importances` (sum==1, dominant feature). All-stumps zeros-return divergence → open prereq blocker #695. |
| REQ-6 (oob_score_ / oob_decision_function_ / oob_prediction_) | NOT-STARTED | open prereq blocker #696. No OOB machinery; sklearn `_generate_unsampled_indices` (`_forest.py:140`) + `_set_oob_score_and_attributes` absent. |
| REQ-7 (class_weight + balanced_subsample) | NOT-STARTED | open prereq blocker #697. Extra-trees classifier has no `class_weight`; `class_weight=None` (`_forest.py:2224` init) constraint is not threaded through. |
| REQ-8 (regressor criterion) | NOT-STARTED | open prereq blocker #698. `ExtraTreesRegressor` has no `with_criterion` and `fn build_single_regression_tree` → `build_extra_regression_tree_for_ensemble` hard-pins `RegressionCriterion::Mse` (per `.design/tree/extra_tree.md` REQ-1 note); sklearn defaults `criterion='squared_error'` but accepts friedman_mse/absolute_error/poisson (`_forest.py:2568`). |
| REQ-9 (random_state determinism) | SHIPPED | per-tree seeds derived sequentially from `StdRng::seed_from_u64(random_state)` in `fit` (both); same seed → identical forest. Consumer: PyO3 `random_state` kwarg. Tests: `test_ensemble_classifier_deterministic`, `test_ensemble_regressor_deterministic`. This is ferrolearn reproducibility; exact numpy-MT parity is the RNG boundary (#693). |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #699. Imports `ndarray`/`rand`/`rayon`, not `ferray-core`/`ferray::random` (R-SUBSTRATE). |

## Architecture

The module has two parallel estimator pairs reusing `MaxFeatures` +
`fn resolve_max_features` + `fn make_tree_params`. `ExtraTreesClassifier<F>` /
`ExtraTreesRegressor<F>` are the unfitted boundary types (public fields +
`with_*` builders + `Default`). `fit` (both) validates shapes
(`ShapeMismatch`/`InsufficientSamples`/`InvalidParameter` for `n_estimators==0`),
generates per-tree `u64` seeds sequentially from a `StdRng` master (or
`rand::rng()` when `random_state` is `None`), then `par_iter`s the seeds —
optionally inside a `rayon::ThreadPoolBuilder` sized by `n_jobs`. Each tree gets
a `rng = StdRng::seed_from_u64(seed)`, computes `indices` (`(0..n_samples)` by
default, or a with-replacement bootstrap draw when `bootstrap` is `true`), and
delegates the recursive random-threshold build to `extra_tree.rs`.
`FittedExtraTreesClassifier<F>` stores `trees: Vec<Vec<Node<F>>>` + `classes` +
`feature_importances`; `predict` traverses each tree (`decision_tree::traverse`)
and — divergence REQ-3 — hard-votes, whereas `predict_proba` correctly averages
per-leaf `class_distribution`. `FittedExtraTreesRegressor<F>::predict` averages
per-tree leaf `value`.

Invariants held: `predict_proba` rows sum to 1 (divide by `n_trees`);
`feature_importances` sums to 1 when any tree splits; feature-count guards on
`predict`/`predict_proba`. Invariant NOT held vs sklearn: `predict ==
argmax(predict_proba)` (the hard-vs-soft gap, REQ-3).

The exact ensemble at a `random_state` is NOT reproducible against sklearn
because the random thresholds (`extra_tree.rs`) and the optional bootstrap
(`next_u64() % n`) use `StdRng`, not numpy MT19937 — the documented RNG boundary,
consistent with extra_tree/random_forest/SGD.

## Verification

Library crate (green at baseline `95a167e6` except REQ-3's pending pin):
```
cargo test -p ferrolearn-tree --lib extra_trees_ensemble   # ensemble unit tests
cargo clippy -p ferrolearn-tree --all-targets -- -D warnings
cargo fmt --all --check
```
Live sklearn oracle (installed 1.5.2, run from `/tmp`):
```
# REQ-1 defaults (note bootstrap=False)
python3 -c "from sklearn.ensemble import ExtraTreesClassifier as C, ExtraTreesRegressor as R; print(C().get_params()); print(R().get_params())"
# REQ-3 soft-vote divergence class (establishes predict==soft, predict!=hard, 54/200 differ)
python3 -c "import numpy as np; from sklearn.ensemble import ExtraTreesClassifier as C; rng=np.random.RandomState(7); X=rng.randn(200,8); y=(X[:,0]+0.5*rng.randn(200)>0).astype(int); m=C(n_estimators=10,max_depth=2,random_state=1).fit(X,y); p=m.predict_proba(X); soft=m.classes_[p.argmax(1)]; hard=np.array([np.bincount([int(t.predict(X[i:i+1].astype(np.float32))[0]) for t in m.estimators_],minlength=2).argmax() for i in range(len(X))]); print('soft==predict',bool(np.all(soft==m.predict(X))),'hard==predict',bool(np.all(hard==m.predict(X))),'differ',int((soft!=hard).sum()))"
```
The NOT-STARTED REQs (3, 6, 7, 8, 10) and the bootstrap-on/max_samples half of
REQ-2 have no green verification by construction — each carries an open prereq
blocker. REQ-2b/4/5/9 and the default-bootstrap half of REQ-2 are verified by the
in-crate `#[test]`s named above (deterministic, or ferrolearn-internal
reproducibility); REQ-1 by the live `get_params()` comparison. A characterization
pin for REQ-3 (R-CHAR-3) belongs in
`ferrolearn-tree/tests/divergence_extra_trees.rs` and/or
`ferrolearn-python/tests/divergence_ensemble.py` once the fixer routes `predict`
through `predict_proba` (mirroring the `random_forest.rs` #670 fix).

## Blockers to open

- #691 — REQ-1: missing constructor params on `extra_trees_ensemble.rs`
  (regressor `criterion`, `max_samples`, `oob_score`, `class_weight`,
  `max_leaf_nodes`, `min_impurity_decrease`, `min_weight_fraction_leaf`,
  `ccp_alpha`, `monotonic_cst`) vs sklearn `_forest.py:2224`/`:2564`.
- #692 — REQ-2: `max_samples` bootstrap sizing absent (`_forest.py:2244`).
- #693 — REQ-2/REQ-2b/REQ-9: exact ensemble parity at `random_state` is a
  numpy-MT vs StdRng RNG boundary (documented like extra_tree/random_forest/SGD,
  NOT a fixable divergence); covers the `bootstrap=true` `next_u64() % n` draw.
- #694 — REQ-3 (TOP): `FittedExtraTreesClassifier::predict` hard-votes
  (`votes[class_idx] += 1` / `max_by_key`); sklearn soft-votes (`argmax` of mean
  `predict_proba`, `_forest.py:904-907`). Same class as `random_forest.rs` #670;
  fix routes `predict` through `predict_proba`. Live: 54/200 rows differ.
- #695 — REQ-5: `feature_importances_` should exclude single-node trees and
  return zeros when all trees are stumps.
- #696 — REQ-6: `oob_score_` / `oob_decision_function_` / `oob_prediction_`
  (`_forest.py:140`) not implemented.
- #697 — REQ-7: classifier `class_weight` / `'balanced_subsample'` not threaded
  through the extra-trees forest.
- #698 — REQ-8: `ExtraTreesRegressor` has no `with_criterion`; ensemble builder
  hard-pins `RegressionCriterion::Mse` (`_forest.py:2568`).
- #699 — REQ-10: migrate `extra_trees_ensemble.rs` off `ndarray`/`rand`/`rayon`
  to the ferray substrate (R-SUBSTRATE), jointly with `extra_tree.rs` #690.

## Top 3 suspected divergences (director / critic's first pins)

1. **Classifier hard-vote (#694) — the headline.** `predict` tallies one label
   per tree and returns the argmax of integer vote counts; sklearn returns
   `argmax(mean predict_proba)` (SOFT, `_forest.py:904-907`). 54/200 rows differ
   on a live `max_depth=2` oracle. Identical to `random_forest.rs`'s fixed #670;
   the fix is the one-liner routing `predict` through the already-correct
   `predict_proba`.
2. **Regressor criterion hard-pinned to MSE (#698).** `ExtraTreesRegressor` has
   no `with_criterion`; the ensemble builder calls
   `build_extra_regression_tree_for_ensemble` which pins
   `RegressionCriterion::Mse`, so `friedman_mse`/`absolute_error`/`poisson`
   (`_forest.py:2568`) are unreachable.
3. **Missing params + no OOB/class_weight (#691/#696/#697).** `max_samples`,
   `oob_score`+`oob_*_`, and `class_weight` are absent from the surface; with
   `bootstrap=False` the default, OOB is moot by default but still part of the
   contract when bootstrap is enabled.
