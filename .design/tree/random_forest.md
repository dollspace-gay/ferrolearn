# Random Forests (RandomForestClassifier / RandomForestRegressor)

<!--
tier: 3-component
status: draft
baseline-commit: 0db5e880
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/ensemble/_forest.py     # BaseForest (:211); BaseForest.fit (:332); _get_n_samples_bootstrap (:95); _generate_sample_indices (:128, randint(0,n,n_bootstrap)); _generate_unsampled_indices (:140); feature_importances_ (:653, mean over node_count>1 trees then /sum, :684); ForestClassifier (:740); ForestClassifier.predict (:883, argmax of predict_proba — SOFT vote, docstring :887); ForestClassifier.predict_proba (:922, mean of tree predict_proba, :962); ForestClassifier._set_oob_score_and_attributes (:806); ForestRegressor (:1006); ForestRegressor.predict (:1042, mean of tree predict, :1081); RandomForestClassifier (:1170, criterion='gini'/max_features='sqrt'); RandomForestRegressor (:1555, criterion='squared_error'/max_features=1.0)
  - sklearn/tree/_classes.py        # the base DecisionTree built per estimator (criteria, predict_proba leaf fractions, feature_importances_) — shared via decision_tree.rs
ferrolearn-module: ferrolearn-tree/src/random_forest.rs
parity-ops: RandomForestClassifier, RandomForestRegressor
crosslink-issue: 669
-->

## Summary

`ferrolearn-tree/src/random_forest.rs` mirrors scikit-learn's
`sklearn.ensemble.RandomForestClassifier` (`_forest.py:1170`) and
`RandomForestRegressor` (`_forest.py:1555`) — Breiman bagged ensembles of CART
trees. Each tree is fit on a **bootstrap** sample of the rows (with replacement,
`_generate_sample_indices`, `:128`) using **per-split random feature
subsampling** of size `max_features` (`RandomForestClassifier` defaults
`'sqrt'`; `RandomForestRegressor` defaults `1.0` = all features). Aggregation:
the classifier predicts via **soft voting** — `argmax` of the *mean* of the
per-tree `predict_proba` (`predict`, `:904-907`; `predict_proba`, `:962`); the
regressor predicts the *mean* of per-tree outputs (`:1081`).
`feature_importances_` is the normalized **mean** of per-tree importances over
trees with `node_count > 1` (`:684`).

ferrolearn re-implements this natively. The module ships unfitted
`RandomForestClassifier<F>` / `RandomForestRegressor<F>` and their `Fitted*`
counterparts, a `MaxFeatures` enum + `fn resolve_max_features`, parallel
(`rayon`) per-tree bootstrap + build delegating to `decision_tree.rs`'s
crate-public `fn build_classification_tree_per_split_features` /
`fn build_regression_tree_per_split_features`, `Fit`/`Predict`,
`predict_proba`/`predict_log_proba`/`score` (clf) and `score` (reg),
`feature_importances_`, and pipeline adapters. The per-tree build correctness
(criteria, best split, FEATURE_THRESHOLD band, leaf values, feature importances)
is **inherited from `decision_tree.rs`, which is oracle-verified**
(`.design/tree/decision_tree.md`).

**A forest is inherently RNG-driven.** The bootstrap draw and the per-split
feature subset are seeded by `random_state`; numpy's MT19937 stream (sklearn,
`randint`) and Rust's `StdRng` (ferrolearn, `next_u64() % n`) cannot bit-match,
so **exact ensemble-for-ensemble parity at a given `random_state` is a
documented RNG boundary** — the same boundary already accepted for SGD's
shuffle, the libsvm CV-fold RNG, and decision_tree's `splitter='random'`. The
**deterministic** contract — the param/default surface, and the *aggregation
logic* given a fixed set of trees (soft-vote mean→argmax, regressor mean,
importance mean-normalization) — is the shippable/pinnable part; the exact tree
ensemble at a seed is not.

**Top divergence the director must see — the classifier votes HARD, not SOFT.**
`FittedRandomForestClassifier::predict` (`predict` in `random_forest.rs`)
tallies **per-tree predicted labels** (`votes[class_idx] += 1`, one vote per
tree) and returns the `argmax` of those integer vote counts — a **hard majority
vote**. sklearn `RandomForestClassifier.predict` (`:904-907`) returns the
`argmax` of the **mean of per-tree `predict_proba`** — a **soft vote** weighted
by leaf class fractions. These differ whenever trees emit non-degenerate leaf
distributions: a live oracle on 200 shallow (`max_depth=2`) samples shows
`predict == soft-argmax` (True) but `predict == hard-majority` (False), with the
two voting schemes disagreeing on **30/200** rows. ferrolearn's own
`predict_proba` *does* average per-tree distributions correctly (soft), so its
`predict` is internally inconsistent with its `predict_proba` — the fix is to
make `predict` `argmax` over `predict_proba` (REQ-4). This is a genuine
numerical divergence, NOT an RNG-boundary artifact.

This doc adapts to the **existing** code. Under R-HONEST-3 no oracle pins exist
yet and the ensemble is RNG-driven, so: the param/default REQ for the params
ferrolearn *has* is SHIPPED with the absent params flagged; the deterministic
aggregation REQs are SHIPPED for the regressor mean / `predict_proba` mean /
importance normalization but the classifier `predict` soft-vote REQ is
NOT-STARTED (hard-vote divergence); and every numpy-parity, missing-param, oob,
class_weight, and substrate REQ is NOT-STARTED with a concrete blocker.

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `get_params()`, sklearn 1.5.2)

`RandomForestClassifier` (`_forest.py:1170`): `n_estimators=100`,
`criterion='gini'`, `max_depth=None`, `min_samples_split=2`,
`min_samples_leaf=1`, `min_weight_fraction_leaf=0.0`, `max_features='sqrt'`,
`max_leaf_nodes=None`, `min_impurity_decrease=0.0`, `bootstrap=True`,
`oob_score=False`, `n_jobs=None`, `random_state=None`, `verbose=0`,
`warm_start=False`, `class_weight=None`, `ccp_alpha=0.0`, `max_samples=None`,
`monotonic_cst=None`.

`RandomForestRegressor` (`_forest.py:1555`): same block with
`criterion='squared_error'`, `max_features=1.0`, and no `class_weight`.

**The defaults ferrolearn matches:** `n_estimators=100`, `max_depth=None`,
`min_samples_split=2`, `min_samples_leaf=1`, `random_state=None`; clf
`max_features='sqrt'` → `MaxFeatures::Sqrt`, `criterion='gini'` →
`ClassificationCriterion::Gini`; reg `max_features=1.0` → `MaxFeatures::All`.

**Params ABSENT in ferrolearn** (REQ-1 flags each): `criterion` on the
regressor (the regressor has no `with_criterion` — always MSE); `bootstrap`
(ferrolearn always bootstraps, no toggle); `max_samples` (bootstrap is always
`n_samples` draws); `oob_score`/`oob_decision_function_` (no OOB at all);
`class_weight` / `'balanced_subsample'` (clf has no weighting);
`max_leaf_nodes`, `min_impurity_decrease`, `min_weight_fraction_leaf`,
`ccp_alpha`, `monotonic_cst` (tree params not threaded through, even though
`decision_tree.rs` supports several of them); `n_jobs` (ferrolearn always uses
`rayon`'s global pool — a R-DEV-7 implementation choice, not an observable-API
gap); `warm_start`, `verbose` (Python ergonomics, R-DEV-4 — not divergences).

### Bootstrap sampling (`_generate_sample_indices`, `:128`)

`n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, max_samples)`
(`:95`): `None` → `n_samples`; `int` → that count (must be `<= n_samples`);
`float` → `max(round(n_samples * max_samples), 1)`. Then
`random_instance.randint(0, n_samples, n_samples_bootstrap, dtype=int32)`
(`:133`) draws indices uniformly with replacement.

ferrolearn (`fit`): always draws exactly `n_samples` indices via
`(bootstrap_rng.next_u64() as usize) % n_samples` per draw (`StdRng` seeded from
a per-tree seed). Divergences: (1) numpy MT19937 vs `StdRng` (RNG boundary,
REQ-2); (2) modulo reduction introduces a slight non-uniformity vs numpy's
rejection-sampled `randint` (subsumed by the RNG boundary); (3) no
`max_samples` support — bootstrap size is fixed at `n_samples` (REQ-1/REQ-2).

### Per-tree fit (`_parallel_build_trees`, `:154`)

Each estimator is a full `DecisionTree*` fit on `X[sample_indices]` with
`max_features` random candidate features evaluated at each split. ferrolearn
seeds per-tree masters sequentially from `random_state` (for thread-order
determinism) then dispatches `tree_seeds.par_iter()` to
`build_classification_tree_per_split_features` /
`build_regression_tree_per_split_features` (in `decision_tree.rs`), which draw a
fresh `max_features_n`-feature subset at each split node from a derived
`split_seed`. The tree-build *correctness* is inherited from the
oracle-verified `decision_tree.rs` (REQ-3); only the RNG stream and the bootstrap
diverge from sklearn (RNG boundary).

### Aggregation

- **Classifier `predict`** (`:904-907`): `classes_.take(argmax(predict_proba, axis=1))`
  — **soft vote**. `predict_proba` (`:962-963`) = `(sum over trees of
  tree.predict_proba(X)) / n_estimators`; each tree's `predict_proba` is the
  per-leaf class fraction (`tree.tree_.value`). ferrolearn's `predict_proba`
  matches this (averages per-tree leaf `class_distribution`, falls back to a
  one-hot at the leaf's `value` when no distribution is stored, divides by
  `n_trees`). ferrolearn's `predict` does **not** — it hard-votes (REQ-4).
- **Regressor `predict`** (`:1081`): `(sum over trees of tree.predict(X)) /
  n_estimators` — plain mean. ferrolearn matches (`sum / n_trees_f`). This is
  deterministic given a fixed forest and pinnable intra-ferrolearn (REQ-5).

### feature_importances_ (`:653`, mean over trees, `:684`)

sklearn: gather `tree.feature_importances_` for every tree with
`tree_.node_count > 1`, take `np.mean(..., axis=0)`, then divide by the sum
(`all_importances / np.sum(all_importances)`). ferrolearn (`fit`): `sum` the
per-tree `compute_feature_importances` across **all** trees then divide by the
total. Sum-then-normalize and mean-then-normalize are **algebraically identical**
when every tree is counted, so the values match — EXCEPT sklearn **excludes
single-node (root-only) trees** from the mean, while ferrolearn includes them
(their importance vector is all-zero, so they do not change the normalized
result either — the divergence is observable only in the all-trees-are-stumps
edge case, where sklearn returns zeros via the `if not all_importances` guard,
`:681`). REQ-6 ships the normalization and pins the edge case.

### oob_score_ / oob_decision_function_ (`:806`, `:1110`)

When `oob_score=True`, sklearn scores each sample using only the trees for which
it was out-of-bag (`_generate_unsampled_indices`, `:140`) and exposes
`oob_score_` + `oob_decision_function_` (clf) / `oob_prediction_` (reg).
ferrolearn has **no OOB machinery** (REQ-7, NOT-STARTED).

### class_weight / 'balanced_subsample' (clf)

sklearn supports `class_weight in {None, 'balanced', 'balanced_subsample',
dict, list-of-dict}`; `'balanced_subsample'` recomputes weights on each
bootstrap draw. ferrolearn's forest classifier has **no** class weighting (REQ-8,
NOT-STARTED) even though `decision_tree.rs` ships a `ClassWeight` enum — it is
not threaded through `random_forest.rs`.

## ferrolearn (what exists)

- **Unfitted**: `pub struct RandomForestClassifier<F>` (public fields
  `n_estimators`, `max_depth`, `max_features`, `min_samples_split`,
  `min_samples_leaf`, `random_state`, `criterion`) / `pub struct
  RandomForestRegressor<F>` (same minus `criterion`); builder setters `with_*`;
  `Default`/`fn new`.
- **Fitted**: `pub struct FittedRandomForestClassifier<F>` (trees, classes,
  n_features, feature_importances) / `pub struct FittedRandomForestRegressor<F>`.
- **Enum/helper**: `pub enum MaxFeatures {Sqrt, Log2, All, Fixed, Fraction}`;
  `fn resolve_max_features`; `fn make_tree_params`.
- **Traits**: `Fit<Array2<F>, Array1<usize>>` (clf) / `Fit<Array2<F>,
  Array1<F>>` (reg); `Predict`; `HasFeatureImportances`; `HasClasses` (clf);
  `PipelineEstimator`/`FittedPipelineEstimator`.
- **Methods**: `fn predict_proba`, `fn predict_log_proba`, `fn score` (clf);
  `fn score` (reg, R²); `fn trees`, `fn n_features`.
- **Build delegation** (from `decision_tree.rs`):
  `fn build_classification_tree_per_split_features`,
  `fn build_regression_tree_per_split_features`,
  `fn compute_feature_importances`, `fn traverse`, `Node<F>`.
- **Consumers**: crate re-export (`lib.rs` `pub use random_forest::{…,
  RandomForestClassifier, RandomForestRegressor}`); PyO3 bindings
  `RsRandomForestClassifier` (`ferrolearn-python/src/classifiers.rs`, uses
  `RandomForestClassifier::<f64>::new()` + `FittedRandomForestClassifier<f64>`,
  registered in `ferrolearn-python/src/lib.rs`) and `RsRandomForestRegressor`
  (`ferrolearn-python/src/extras.rs`); the pipeline adapters.

## Requirements

- REQ-1: **Param surface + defaults (R-DEV-2).** `n_estimators=100`,
  `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`,
  `random_state=None`, clf `max_features='sqrt'`+`criterion='gini'`, reg
  `max_features=1.0` match sklearn's `get_params()`. ABSENT: regressor
  `criterion`, `bootstrap`, `max_samples`, `oob_score`, `class_weight`,
  `max_leaf_nodes`, `min_impurity_decrease`, `min_weight_fraction_leaf`,
  `ccp_alpha`, `monotonic_cst`.
- REQ-2: **Bootstrap sampling** — with-replacement `n_samples` draws,
  `max_samples` sizing, `bootstrap` toggle. RNG boundary (numpy MT vs StdRng).
- REQ-3: **Per-tree fit** — each tree = a CART tree on a bootstrap sample with
  `max_features` per-split random subsampling; correctness inherited from the
  oracle-verified `decision_tree.rs`. RNG boundary for the exact ensemble.
- REQ-4: **Classifier aggregation = SOFT vote.** `predict` is `argmax` of
  the *mean of per-tree `predict_proba`* (sklearn `:904-907`), NOT a hard
  per-tree-label majority. `FittedRandomForestClassifier::predict` routes
  through `predict_proba` and takes the per-row lowest-index argmax mapped to
  `classes_[idx]` (matches `np.argmax` tie-break).
- REQ-5: **Classifier `predict_proba` + regressor `predict` = mean.** clf
  `predict_proba` = mean of per-tree class distributions (`:962`); reg
  `predict` = mean of per-tree outputs (`:1081`). Deterministic given the forest.
- REQ-6: **`feature_importances_` = normalized mean of per-tree importances**
  (`:684`), with the all-stumps edge case returning zeros (`:681`).
- REQ-7: **`oob_score_` / `oob_decision_function_`** (`:806`, `:1110`).
- REQ-8: **`class_weight` (+ `'balanced_subsample'`)** for the classifier.
- REQ-9: **`random_state` determinism** — same seed → identical forest
  (ferrolearn reproducibility, NOT numpy parity). RNG boundary.
- REQ-10: **ferray substrate (R-SUBSTRATE).** Imports `ndarray`/`rand`, not
  `ferray-core`/`ferray::random`.

## Acceptance criteria

- AC-1: live `RandomForestClassifier().get_params()` /
  `RandomForestRegressor().get_params()` equal the defaults in REQ-1 for the
  params ferrolearn exposes; absent params are enumerated.
- AC-2: on 200 shallow (`max_depth=2`) samples, sklearn `predict` equals
  `classes_.take(argmax(predict_proba))` (soft) and differs from a hard
  per-tree-label majority on `>0` rows — establishing the divergence class
  (live: 30/200). ferrolearn `predict` must agree with `argmax(predict_proba)`
  on the same fixed forest.
- AC-3: regressor `predict` equals `mean` of `t.predict(X)` over
  `t in estimators_` (sklearn `:1081`); ferrolearn matches to 1e-12 on a fixed
  intra-ferrolearn forest.
- AC-4: `predict_proba` rows sum to 1 and equal the mean of per-tree
  distributions on a fixed forest.
- AC-5: `feature_importances_` sums to 1 (or is all-zeros when every tree is a
  stump) and equals the normalized mean of per-tree importances.
- AC-6: `random_state` reproducibility — two `fit` calls with the same seed
  produce identical `predict` (already covered by
  `test_forest_classifier_reproducibility` / `_regressor_reproducibility`).

## REQ status table

Binary (R-DEFER-2). `RandomForestClassifier`/`RandomForestRegressor` are
boundary estimator types re-exported at the crate root and registered as PyO3
`RsRandomForestClassifier`/`RsRandomForestRegressor` (S5/R-DEFER-1 non-test
consumer surface). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (param surface + defaults) | SHIPPED (with gaps flagged) | `fn new` on `RandomForestClassifier` (`MaxFeatures::Sqrt`, `ClassificationCriterion::Gini`, `n_estimators=100`, `min_samples_split=2`, `min_samples_leaf=1`, `max_depth=None`) and `RandomForestRegressor` (`MaxFeatures::All`) match `get_params()` (`_forest.py:1170`,`:1555`). Consumer: `RsRandomForestClassifier::new` / `RsRandomForestRegressor::new` (PyO3) + crate re-export. Verification: `cd /tmp && python3 -c "import sklearn; from sklearn.ensemble import RandomForestClassifier as C,RandomForestRegressor as R; print(C().get_params(), R().get_params())"` (matches the exposed subset). ABSENT params (regressor `criterion`, `bootstrap`, `max_samples`, `oob_score`, `class_weight`, `max_leaf_nodes`, `min_impurity_decrease`, `min_weight_fraction_leaf`, `ccp_alpha`) → open prereq blocker #672. |
| REQ-2 (bootstrap sampling) | NOT-STARTED | open prereq blocker #673. `fit` draws exactly `n_samples` via `next_u64() % n_samples` (StdRng), with no `max_samples` sizing and no `bootstrap=False` toggle; numpy `randint(0,n,n_bootstrap)` (`_forest.py:133`) is a different stream (RNG boundary) AND ferrolearn omits `max_samples`/`bootstrap`. |
| REQ-3 (per-tree fit) | SHIPPED (correctness inherited) | `fit` calls `build_classification_tree_per_split_features` / `build_regression_tree_per_split_features` (`decision_tree.rs`, oracle-verified per `.design/tree/decision_tree.md`) on the bootstrap sample with `resolve_max_features` per-split subsets. Consumer: PyO3 `fit`. Tests: `test_forest_classifier_simple`, `test_forest_regressor_simple`, `test_forest_regressor_max_features_strategies`. Exact tree ensemble at a seed = documented RNG boundary (#674). |
| REQ-4 (classifier SOFT vote) | SHIPPED | closed #670. `FittedRandomForestClassifier::predict` (`random_forest.rs`) now routes through `predict_proba` and returns `self.classes[argmax_row]` with the lowest-index argmax tie-break (matches `np.argmax`), mirroring sklearn `predict` (`_forest.py:904-907`) = `classes_.take(argmax(predict_proba, axis=1))` (SOFT). Pin `divergence_predict_is_soft_vote_argmax_of_proba` (`tests/divergence_random_forest.rs`) green: 0/200 rows diverge from `argmax(predict_proba)`. |
| REQ-5 (predict_proba mean / regressor mean) | SHIPPED | `fn predict_proba` averages per-tree leaf `class_distribution` then `/n_trees` (mirrors `_forest.py:962`); `FittedRandomForestRegressor::predict` = `sum / n_trees_f` (mirrors `_forest.py:1081`). Consumer: PyO3 regressor `predict`; clf `predict_proba` getter; pipeline adapters. Tests: `test_forest_regressor_simple` (mean separation), `test_forest_regressor_reproducibility`. Deterministic given the forest. |
| REQ-6 (feature_importances_ normalized mean) | SHIPPED (stump edge pinnable) | `fit` accumulates `compute_feature_importances` over trees then divides by the sum (algebraically equal to sklearn's mean-then-normalize, `_forest.py:684`); `HasFeatureImportances::feature_importances`. Consumer: `RsRandomForestClassifier`/`extras.rs` getters. Tests: `test_forest_classifier_feature_importances`, `test_forest_regressor_feature_importances`. All-stumps zeros-return divergence (`_forest.py:681`) → open prereq blocker #676. |
| REQ-7 (oob_score_ / oob_decision_function_) | NOT-STARTED | open prereq blocker #677. No OOB machinery; sklearn `_generate_unsampled_indices` (`_forest.py:140`) + `_set_oob_score_and_attributes` (`:806`,`:1110`) absent. |
| REQ-8 (class_weight + balanced_subsample) | NOT-STARTED | open prereq blocker #678. Forest classifier has no `class_weight`; `decision_tree.rs`'s `ClassWeight` enum is not threaded through `random_forest.rs`. |
| REQ-9 (random_state determinism) | SHIPPED | per-tree seeds derived sequentially from `StdRng::seed_from_u64(random_state)` in `fit`; same seed → identical forest. Consumer: PyO3 `random_state`. Tests: `test_forest_classifier_reproducibility`, `test_forest_regressor_reproducibility`. This is ferrolearn reproducibility; exact numpy-MT parity is the RNG boundary (#674). |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #679. Imports `ndarray`/`rand`/`rayon`, not `ferray-core`/`ferray::random` (R-SUBSTRATE). |

## Architecture

The module has two parallel estimator pairs sharing `MaxFeatures` +
`fn resolve_max_features` + `fn make_tree_params`. `RandomForestClassifier<F>`
/ `RandomForestRegressor<F>` are the unfitted boundary types (public fields +
`with_*` builders + `Default`). `fit` (both) validates shapes
(`ShapeMismatch`/`InsufficientSamples`/`InvalidParameter` for `n_estimators==0`),
generates per-tree `u64` seeds sequentially from a `StdRng` master (or
`rand::rng()` when `random_state` is `None`), then `par_iter`s the seeds: each
tree gets a `bootstrap_rng = StdRng::seed_from_u64(seed)`, draws `n_samples`
bootstrap indices, derives a `split_seed`, and delegates the recursive build to
`decision_tree.rs`. `FittedRandomForestClassifier<F>` stores `trees:
Vec<Vec<Node<F>>>` + `classes` + `feature_importances`; `predict` traverses each
tree (`decision_tree::traverse`) and — divergence REQ-4 — hard-votes, whereas
`predict_proba` correctly averages per-leaf `class_distribution`.
`FittedRandomForestRegressor<F>::predict` averages per-tree leaf `value`.

Invariants held: `predict_proba` rows sum to 1 (divide by `n_trees`);
`feature_importances` sums to 1 when any tree splits; feature-count guards on
`predict`/`predict_proba`. Invariant NOT held vs sklearn: `predict ==
argmax(predict_proba)` (the hard-vs-soft gap, REQ-4).

The exact ensemble at a `random_state` is NOT reproducible against sklearn
because the bootstrap (`next_u64() % n`) and per-split feature draws use
`StdRng`, not numpy MT19937 — the documented RNG boundary, consistent with
SGD/extra_tree/decision_tree `splitter='random'`.

## Verification

Library crate (green at baseline `0db5e880`):
```
cargo test -p ferrolearn-tree --lib random_forest   # 27 passed; 0 failed
cargo clippy -p ferrolearn-tree --all-targets -- -D warnings
cargo fmt --all --check
```
Live sklearn oracle (installed 1.5.2, run from `/tmp`):
```
# REQ-1 defaults
python3 -c "import sklearn; from sklearn.ensemble import RandomForestClassifier as C, RandomForestRegressor as R; print(C().get_params()); print(R().get_params())"
# REQ-4 soft-vote divergence class (establishes predict==soft, predict!=hard, 30/200 differ)
python3 -c "import numpy as np; from sklearn.ensemble import RandomForestClassifier as C; rng=np.random.RandomState(7); X=rng.randn(200,8); y=(X[:,0]+0.5*rng.randn(200)>0).astype(int); m=C(n_estimators=10,max_depth=2,random_state=1).fit(X,y); p=m.predict_proba(X); soft=m.classes_[p.argmax(1)]; hard=np.array([np.bincount([int(t.predict(X[i:i+1].astype(np.float32))[0]) for t in m.estimators_],minlength=2).argmax() for i in range(len(X))]); print('soft==predict',np.all(soft==m.predict(X)),'hard==predict',np.all(hard==m.predict(X)),'differ',int((soft!=hard).sum()))"
```
The NOT-STARTED REQs (2, 4, 7, 8, 10) have no green verification by
construction — each carries an open prereq blocker. REQ-3/5/6/9 are verified by
the in-crate `#[test]`s named above (deterministic, or ferrolearn-internal
reproducibility); REQ-1 by the live `get_params()` comparison. A characterization
pin for REQ-4 (R-CHAR-3) belongs in
`ferrolearn-tree/tests/divergence_random_forest.rs` and/or
`ferrolearn-python/tests/divergence_ensemble.py` once the fixer routes `predict`
through `predict_proba`.

## Blockers to open

- #672 — REQ-1: missing constructor params on `random_forest.rs`
  (regressor `criterion`, `bootstrap`, `max_samples`, `oob_score`,
  `class_weight`, `max_leaf_nodes`, `min_impurity_decrease`,
  `min_weight_fraction_leaf`, `ccp_alpha`) vs sklearn `_forest.py:1170`/`:1555`.
- #673 — REQ-2: bootstrap sampling — `max_samples` sizing + `bootstrap=False`
  toggle absent; `next_u64() % n` vs numpy `randint` (`_forest.py:128`).
- #674 — REQ-3/REQ-9: exact ensemble parity at `random_state` is a numpy-MT
  vs StdRng RNG boundary (documented like SGD/extra_tree, NOT a fixable
  divergence).
- #675 — REQ-4: `FittedRandomForestClassifier::predict` hard-votes; sklearn
  soft-votes (`argmax` of mean `predict_proba`, `_forest.py:904-907`).
- #676 — REQ-6: `feature_importances_` should exclude single-node trees and
  return zeros when all trees are stumps (`_forest.py:681`).
- #677 — REQ-7: `oob_score_` / `oob_decision_function_` / `oob_prediction_`
  (`_forest.py:140`,`:806`,`:1110`) not implemented.
- #678 — REQ-8: classifier `class_weight` / `'balanced_subsample'` not threaded
  through the forest.
- #679 — REQ-10: migrate `random_forest.rs` off `ndarray`/`rand`/`rayon` to the
  ferray substrate (R-SUBSTRATE).
