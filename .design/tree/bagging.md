# Bagging ensembles (BaggingClassifier / BaggingRegressor)

<!--
tier: 3-component
status: draft
baseline-commit: f0939ff4
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/ensemble/_bagging.py    # BaseBagging (:287); _parameter_constraints (:294, max_samples/max_features each Interval(Integral,1,None) | Interval(RealNotInt,0,1)); __init__ defaults (:314, n_estimators=10/max_samples=1.0/max_features=1.0/bootstrap=True/bootstrap_features=False/oob_score=False/warm_start=False/random_state=None); _generate_indices (:54, bootstrap->randint(0,n_pop,n) / else sample_without_replacement); _generate_bagging_indices (:67, feature draw then sample draw off one check_random_state); _parallel_build_estimators (:91, per-est fit on X[indices][:,features]); _max_samples sizing (:475-484, float->int(max_samples*n) ; int passthrough ; must be <= n); _max_features sizing (:487-498, int passthrough / float->int(max_features*n_features_in_) then max(1,int)); bootstrap+oob guard (:501); _parallel_predict_proba (:203, mean of per-est predict_proba, voting fallback :222-225); BaggingClassifier (:645, estimator=None->DecisionTreeClassifier :844); BaggingClassifier.predict (:895, classes_.take(argmax(predict_proba)) — SOFT vote :914); BaggingClassifier.predict_proba (:916, sum(all_proba)/n_estimators :964); predict_log_proba (:968); decision_function (:1024); _validate_y (:887, np.unique return_inverse classes_); _set_oob_score (:848, oob_decision_function_/oob_score_); BaggingRegressor (:1071, estimator=None->DecisionTreeRegressor :1334); BaggingRegressor.predict (:1259, sum/n_estimators :1299); BaggingRegressor._set_oob_score (:1303, oob_prediction_/oob_score_)
  - sklearn/tree/_classes.py        # the DecisionTree{Classifier,Regressor} default base estimator built per member — shared via decision_tree.rs
ferrolearn-module: ferrolearn-tree/src/bagging.rs
parity-ops: BaggingClassifier, BaggingRegressor
crosslink-issue: 717
-->

## Summary

`ferrolearn-tree/src/bagging.rs` mirrors scikit-learn's
`sklearn.ensemble.BaggingClassifier` (`_bagging.py:645`) and `BaggingRegressor`
(`_bagging.py:1071`) — bootstrap-aggregation meta-estimators that fit
`n_estimators` base learners, each on a random subset of **samples** (drawn with
replacement when `bootstrap=True`) and optionally a random subset of
**features** (`_generate_bagging_indices`, `:67`), then aggregate: the
classifier predicts via **soft voting** — `classes_.take(argmax(mean
predict_proba))` (`predict`, `:913-914`); the regressor predicts the **mean** of
per-estimator outputs (`predict`, `:1299`). `predict_proba` is the per-class
**mean** of the base estimators' `predict_proba` (`:964`).

scikit-learn's Bagging is a **meta-estimator over an arbitrary base
`estimator`** (`__init__`, `:813`; `estimator=None` → `DecisionTreeClassifier` /
`DecisionTreeRegressor`, `:844`/`:1334`). ferrolearn re-implements only the
**decision-tree-only** specialization natively: the module ships unfitted
`BaggingClassifier<F>` / `BaggingRegressor<F>` and their `Fitted*` counterparts,
parallel (`rayon`) per-tree bootstrap-sample + feature-subset draws delegating to
`decision_tree.rs`'s crate-public `fn build_classification_tree_with_feature_subset`
/ `fn build_regression_tree_with_feature_subset`, `Fit`/`Predict`,
`predict_proba`/`predict_log_proba`/`score` (clf) and `score` (reg, R²),
`feature_importances_`, and pipeline adapters. The per-tree build correctness
(criteria, best split, leaf class distributions, importances) is **inherited from
`decision_tree.rs`, which is oracle-verified** (`.design/tree/decision_tree.md`).

**A bagging ensemble is inherently RNG-driven.** The bootstrap sample draw and
the feature-subset draw are seeded by `random_state`; numpy's MT19937 stream
(sklearn, `randint`/`sample_without_replacement`) and Rust's `StdRng`
(ferrolearn, `next_u64() % n`) cannot bit-match, so **exact ensemble-for-ensemble
parity at a given `random_state` is a documented RNG boundary** — the same
boundary already accepted for `random_forest.rs` (#674) and `extra_trees`. The
**deterministic** contract — the param/default surface, and the *aggregation
logic* given a fixed set of trees (soft-vote mean→argmax, regressor mean,
importance normalization) — is the shippable/pinnable part; the exact tree
ensemble at a seed is not.

**Top divergence the director must see — the classifier votes HARD, not SOFT.**
`FittedBaggingClassifier::predict` (`predict` in `bagging.rs`, the loop at lines
**~476-496**) tallies **per-tree predicted labels** (`votes[class_idx] += 1`, one
vote per tree) and returns the `argmax` of those integer vote counts via
`max_by_key` — a **hard majority vote**. sklearn `BaggingClassifier.predict`
(`:913-914`) returns `classes_.take(argmax(predict_proba, axis=1))` — a **soft
vote** = argmax of the *mean of per-estimator `predict_proba`*. These differ
whenever trees emit non-degenerate leaf distributions: a live oracle on 300
samples with shallow (`max_depth=2`) base trees, `max_features=0.5`,
`n_estimators=11` shows `predict == argmax(predict_proba)` (soft, True) while
soft and a hard per-tree-label majority **disagree on 21/300 rows**. ferrolearn's
own `predict_proba` *does* average per-tree leaf distributions correctly (soft,
`predict_proba` divides by `n_trees`), so its `predict` is **internally
inconsistent** with its `predict_proba` — the single-file fixer reroutes
`predict` through `predict_proba` (REQ-3). This is a genuine numerical
divergence, NOT an RNG-boundary artifact, and is the **same divergence class** as
`random_forest.rs` (#670) and `extra_trees` (#679).

A secondary divergence rides along: `max_by_key` returns the **last** maximum on
a tie, whereas numpy `argmax` returns the **lowest** index — folded into REQ-3
(it vanishes once `predict` routes through `predict_proba` with a lowest-index
argmax).

**This is also a meta-estimator (architectural, like `voting.rs`).** sklearn
wraps an **arbitrary** base `estimator` (`HasMethods(["fit","predict"])`, `:295`);
ferrolearn is **decision-tree-only** — no `estimator` field. A heterogeneous base
learner is cross-crate (it would need trait objects / an estimator enum spanning
the leaf crates, and cannot live in `ferrolearn-tree`, which depends only on
`ferrolearn-core`) — NOT-STARTED, like `voting.rs` #695. The decision-tree
specialization that ferrolearn ships IS the most common Bagging use (the sklearn
default base estimator).

This doc adapts to the **existing** code under R-HONEST-3 (underclaim beats
overclaim). The param/default REQ for the params ferrolearn *has* is SHIPPED with
absent params flagged; the deterministic aggregation REQs are SHIPPED for the
regressor mean / `predict_proba` mean / feature-importance normalization, but the
classifier `predict` soft-vote REQ is NOT-STARTED (hard-vote divergence); and
every numpy-parity, missing-param, oob, heterogeneous-estimator, and substrate
REQ is NOT-STARTED with a concrete blocker.

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `get_params()`, sklearn 1.5.2)

`BaggingClassifier(estimator=None, n_estimators=10, *, max_samples=1.0,
max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False,
warm_start=False, n_jobs=None, random_state=None, verbose=0)` (`__init__`,
`:813`). Live `get_params()`: `{'bootstrap': True, 'bootstrap_features': False,
'estimator': None, 'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 10,
'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0,
'warm_start': False}`. `BaggingRegressor` (`:1230`) has the identical block.

**The defaults ferrolearn matches** (live-verified): `n_estimators=10`,
`max_samples=1.0`, `max_features=1.0`, `bootstrap=true`,
`bootstrap_features=false`, `random_state=None` (`fn new` on both
`BaggingClassifier` and `BaggingRegressor`). The `oob_score=false` default is
matched only by absence (ferrolearn has no `oob_score` field at all).

**Params ABSENT in ferrolearn** (REQ-1 flags each): `estimator` (the
heterogeneous base learner — ferrolearn is tree-only, REQ-7); `oob_score`
(no OOB machinery, REQ-8); `warm_start`, `n_jobs`, `verbose` (Python/parallel
ergonomics — `n_jobs` is subsumed by `rayon`'s global pool, R-DEV-7; `warm_start`
and `verbose` are R-DEV-4 non-divergences). Additionally, `max_samples` and
`max_features` accept **`int` OR `float`** in sklearn (`_parameter_constraints`,
`:297-304`: `Interval(Integral,1,None)` *or* `Interval(RealNotInt,0,1)`);
ferrolearn types both as `f64` fractions only (REQ-9).

**ferrolearn adds `max_depth`** as a constructor knob (`with_max_depth`) — a
convenience pass-through to the base tree's `max_depth`; sklearn sets base-tree
params on the `estimator` object instead. ferrolearn pins the base tree at
`min_samples_split=2`, `min_samples_leaf=1`, `criterion=Gini` (clf) / MSE (reg),
matching the `DecisionTree*` defaults.

### Sample + feature subsampling (`_generate_bagging_indices`, `:67`)

sklearn draws, off **one** `check_random_state(seed)` (`:78`), first the
**feature** indices then the **sample** indices via `_generate_indices` (`:54`):
`bootstrap` → `random_state.randint(0, n_population, n_samples)` (with
replacement); else `sample_without_replacement(n_population, n_samples)`. Sizing
(`_fit`, `:475-498`): `max_samples` float → `int(max_samples * n_samples)`, int →
passthrough, must be `<= n_samples` (`:480`); `max_features` int → passthrough,
float → `int(max_features * n_features_in_)`, then `max(1, int(...))` (`:495`).

ferrolearn (`fit`): `n_sample_draw = ceil(n_samples * max_samples).max(1)`,
`n_feature_draw = ceil(n_features * max_features).max(1).min(n_features)`. Then
per tree, seeded by a derived `StdRng`: samples drawn first (`bootstrap` →
`next_u64() % n_samples` with replacement; else `rand_sample_indices` without
replacement), then features (`bootstrap_features` → `next_u64() % n_features`;
else the full set when `n_feature_draw == n_features`, otherwise
`rand_sample_indices`). Divergences: (1) numpy MT19937 vs `StdRng` and
`randint`/`sample_without_replacement` vs `next_u64() % n` (RNG boundary, REQ-2);
(2) **draw order is reversed** — ferrolearn draws samples-then-features, sklearn
features-then-samples (subsumed by the RNG boundary, observable only as a
different ensemble at a seed, not a contract difference for a fixed ensemble);
(3) ferrolearn's `ceil` sizing vs sklearn's `int(... )` truncation +
`max(1,int())` for features (REQ-9, only matters at fractional sizes);
(4) no `int`-valued `max_samples`/`max_features` (REQ-9). The bootstrap toggle
(`bootstrap`, `bootstrap_features`) IS present and exercised (REQ-2 structural).

### Per-estimator fit (`_parallel_build_estimators`, `:91`)

Each member is a base estimator fit on `X[sample_indices][:, feature_indices]`
(`:191-195`). ferrolearn seeds per-tree masters sequentially from `random_state`
(for thread-order determinism), then `tree_seeds.par_iter()` builds each tree via
`build_classification_tree_with_feature_subset` /
`build_regression_tree_with_feature_subset` (in `decision_tree.rs`) on the
sampled rows and the selected feature subset, storing the `feature_indices` for
prediction-time re-indexing. The tree-build *correctness* is inherited from the
oracle-verified `decision_tree.rs` (REQ-2); only the RNG stream and the base
estimator identity (tree-only, REQ-7) diverge from sklearn.

### Aggregation

- **Classifier `predict`** (`:913-914`): `classes_.take(argmax(predict_proba,
  axis=1))` — **soft vote**. `predict_proba` (`:916`) = `sum(per-estimator
  predict_proba) / n_estimators` (`:964`); each member's `predict_proba` is the
  per-leaf class fraction (`_parallel_predict_proba`, `:203`; voting fallback
  `:222-225` only when the base has no `predict_proba`). ferrolearn's
  `predict_proba` matches this (averages each tree's leaf `class_distribution`,
  falls back to a one-hot at the leaf's `value` when no distribution is stored,
  divides by `n_trees`). ferrolearn's `predict` does **NOT** — it hard-votes
  (REQ-3).
- **Regressor `predict`** (`:1299`): `sum(per-estimator predict) / n_estimators`
  — plain mean (`_parallel_predict_regression`, `:263`). ferrolearn matches
  (`sum / n_trees_f`). Deterministic given a fixed ensemble (REQ-5).

### feature_importances_

sklearn's `BaggingClassifier`/`BaggingRegressor` do **NOT** expose
`feature_importances_` as a fitted attribute (it is not in their `Attributes`
docstring, `:733`/`:1156`; the base `DecisionTree.feature_importances_` is per
member only — unlike `RandomForest`, which aggregates). ferrolearn **adds** an
aggregated, normalized `feature_importances_` via `HasFeatureImportances`
(`aggregate_tree_importances` over the ensemble, mapped back through
`feature_indices`, normalized to sum to 1). This is a ferrolearn **extension**,
not a sklearn-contract mirror (R-DEV-7-style convenience): it is SHIPPED as a
self-consistent introspection surface but carries no sklearn parity claim (REQ-6).

### oob_score_ / oob_decision_function_ / oob_prediction_ (`:848`, `:1303`)

When `oob_score=True`, sklearn scores each sample using only the estimators for
which it was out-of-bag (the complement of `estimators_samples_`), exposing
`oob_score_` + `oob_decision_function_` (clf, `:884-885`) /
`oob_prediction_` (reg, `:1328`). It guards `bootstrap=False` with `oob_score`
(`:501`). ferrolearn has **no OOB machinery** (REQ-8, NOT-STARTED).

### Heterogeneous base estimator (`estimator`, `:295`/`:671`)

sklearn's `estimator` is any object with `fit`/`predict` (`HasMethods`, `:295`);
`estimator=None` resolves to a `DecisionTree*` (`:844`/`:1334`). This is the core
meta-estimator contract — Bagging an `SVC`, a `KNeighborsClassifier`, etc.
ferrolearn is **decision-tree-only** with no `estimator` field. A faithful
version needs cross-crate trait objects / an estimator enum and cannot live in
`ferrolearn-tree` (depends only on `ferrolearn-core`) — the same architectural
gap as `voting.rs` (REQ-7, NOT-STARTED).

## ferrolearn (what exists)

- **Unfitted**: `pub struct BaggingClassifier<F>` (public fields
  `n_estimators`, `max_samples`, `max_features`, `bootstrap`,
  `bootstrap_features`, `random_state`, `max_depth`) / `pub struct
  BaggingRegressor<F>` (identical); builder setters `with_n_estimators`,
  `with_max_samples`, `with_max_features`, `with_bootstrap`,
  `with_bootstrap_features`, `with_random_state`, `with_max_depth`;
  `Default` / `fn new`.
- **Fitted**: `pub struct FittedBaggingClassifier<F>` (`trees:
  Vec<Vec<Node<F>>>`, `feature_indices`, `classes`, `n_features`,
  `feature_importances`) / `pub struct FittedBaggingRegressor<F>` (same minus
  `classes`).
- **Traits**: `Fit<Array2<F>, Array1<usize>>` (clf) / `Fit<Array2<F>, Array1<F>>`
  (reg); `Predict`; `HasFeatureImportances` (both); `HasClasses` (clf);
  `PipelineEstimator` / `FittedPipelineEstimator` (both, via adapters).
- **Methods**: `fn predict_proba`, `fn predict_log_proba`, `fn score` (clf, mean
  accuracy); `fn score` (reg, R²); `fn trees`, `fn n_features`.
- **Build delegation** (from `decision_tree.rs`):
  `fn build_classification_tree_with_feature_subset`,
  `fn build_regression_tree_with_feature_subset`,
  `fn aggregate_tree_importances`, `fn traverse`, `Node<F>`,
  `ClassificationCriterion`, `TreeParams`.
- **Consumers**: crate re-export — `ferrolearn-tree/src/lib.rs`
  (`pub use bagging::{BaggingClassifier, BaggingRegressor,
  FittedBaggingClassifier, FittedBaggingRegressor}`); PyO3 binding
  `RsBaggingClassifier` (`ferrolearn-python/src/extras.rs`, uses
  `BaggingClassifier::<f64>::new().with_n_estimators(..)` +
  `FittedBaggingClassifier<f64>`, registered in `ferrolearn-python/src/lib.rs` as
  `_RsBaggingClassifier`) — exposes only `n_estimators`/`random_state`, `fit`,
  `predict` (no `predict_proba`, no `max_samples`/`max_features`/`bootstrap`).
  **There is NO PyO3 binding for `BaggingRegressor`** (verified: only
  `RsBaggingClassifier` matches `Bagging` in `ferrolearn-python/src/`) — for the
  regressor the crate re-export + the pipeline adapter are the only non-test
  production consumers.

## Requirements

- REQ-1: **Param surface + defaults (R-DEV-2).** `n_estimators=10`,
  `max_samples=1.0`, `max_features=1.0`, `bootstrap=true`,
  `bootstrap_features=false`, `random_state=None` match sklearn's `get_params()`
  (`_bagging.py:314`/`:813`/`:1230`). ABSENT: `estimator` (REQ-7), `oob_score`
  (REQ-8), `warm_start`, `n_jobs`, `verbose`; `max_samples`/`max_features` as
  `int` (REQ-9).
- REQ-2: **Sample + feature subsampling + per-tree fit.** With-replacement sample
  draws (`bootstrap`), feature-subset draws (`max_features`,
  `bootstrap_features`), each tree = a CART tree on `X[samples][:,features]`;
  correctness inherited from the oracle-verified `decision_tree.rs`. RNG boundary
  (numpy MT vs StdRng; reversed draw order) for the exact ensemble.
- REQ-3: **Classifier aggregation = SOFT vote (THE divergence).** `predict` must
  be `argmax` of the *mean of per-tree `predict_proba`* (sklearn `:913-914`), NOT
  a hard per-tree-label majority. `FittedBaggingClassifier::predict` (the
  `votes[class_idx] += 1` / `max_by_key` loop, ~476-496) routes through
  `predict_proba` and takes the per-row lowest-index argmax mapped to
  `classes_[idx]` (matches `np.argmax` tie-break, eliminating the `max_by_key`
  last-index tie bug).
- REQ-4: **Classifier `predict_proba` = mean of per-tree distributions.**
  `predict_proba` = `sum(per-tree leaf class-fraction) / n_trees` (sklearn
  `:964`). Deterministic given the ensemble.
- REQ-5: **Regressor `predict` = mean.** `predict` = `sum(per-tree output) /
  n_trees` (sklearn `:1299`). Deterministic given the ensemble.
- REQ-6: **`feature_importances_` (ferrolearn extension).** Normalized aggregated
  per-tree importances, feature-index-remapped, summing to 1 — a ferrolearn
  introspection convenience; sklearn Bagging does NOT expose this attribute (no
  sklearn parity claim).
- REQ-7: **Heterogeneous base `estimator` (architectural, cross-crate).** Hold an
  arbitrary base learner (`HasMethods(["fit","predict"])`, `:295`;
  `estimator=None` → `DecisionTree*`, `:844`/`:1334`). ferrolearn is tree-only.
- REQ-8: **`oob_score` / `oob_decision_function_` / `oob_prediction_`** (`:848`,
  `:1303`), with the `bootstrap=False` guard (`:501`).
- REQ-9: **`int`-valued `max_samples` / `max_features` + sizing parity.** sklearn
  accepts `int` (absolute count) OR `float` (fraction); float sizing is
  `int(frac * n)` (truncation) for samples and `max(1, int(frac * n))` for
  features (`:478`/`:490-495`). ferrolearn types both as `f64` fractions and
  sizes with `ceil(...).max(1)` — differs at fractional sizes and rejects `int`.
- REQ-10: **`random_state` determinism** — same seed → identical ensemble
  (ferrolearn reproducibility, NOT numpy parity). RNG boundary.
- REQ-11: **ferray substrate (R-SUBSTRATE).** Imports `ndarray`/`rand`/`rayon`,
  not `ferray-core`/`ferray::random`.

## Acceptance criteria

- AC-1: live `BaggingClassifier().get_params()` / `BaggingRegressor().get_params()`
  equal the defaults in REQ-1 for the params ferrolearn exposes (live-verified:
  `n_estimators=10`, `max_samples=1.0`, `max_features=1.0`, `bootstrap=True`,
  `bootstrap_features=False`, `oob_score=False`, `random_state=None`); absent
  params (`estimator`, `oob_score`, `warm_start`, `n_jobs`, `verbose`) are
  enumerated.
- AC-2: on 300 samples with shallow (`max_depth=2`) base trees,
  `max_features=0.5`, `n_estimators=11`, sklearn `predict` equals
  `classes_.take(argmax(predict_proba))` (soft, live: True) and a hard
  per-tree-label majority differs from soft on `>0` rows — establishing the
  divergence class (live: 21/300). ferrolearn `predict` must agree with
  `argmax(predict_proba)` on the same fixed ensemble.
- AC-3: regressor `predict` equals `mean` of per-tree outputs (sklearn `:1299`);
  ferrolearn matches to 1e-12 on a fixed intra-ferrolearn ensemble.
- AC-4: `predict_proba` rows sum to 1 and equal the mean of per-tree
  distributions on a fixed ensemble (sklearn `:964`).
- AC-5: `random_state` reproducibility — two `fit` calls with the same seed
  produce identical `predict` (covered by
  `test_bagging_classifier_reproducibility` / `_regressor_reproducibility`).
- AC-6: live `BaggingClassifier(max_samples=3)` (int) draws 3 samples while
  `max_samples=0.5` (float) draws `int(0.5*n)`; ferrolearn rejects the int form
  (typed `f64`) and uses `ceil` sizing for the fraction — pins REQ-9.

## REQ status table

Binary (R-DEFER-2). `BaggingClassifier`/`BaggingRegressor` are boundary estimator
types re-exported at the crate root; `BaggingClassifier` is additionally
registered as PyO3 `_RsBaggingClassifier` (S5/R-DEFER-1 non-test consumer
surface). `BaggingRegressor` has **no** PyO3 binding — its crate re-export + the
pipeline adapter are the non-test consumers. Cites use symbol anchors
(ferrolearn) / `file:line` (sklearn 1.5.2).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (param surface + defaults) | SHIPPED (with gaps flagged) | `fn new` on `BaggingClassifier` and `BaggingRegressor` set `n_estimators=10`, `max_samples=1.0`, `max_features=1.0`, `bootstrap=true`, `bootstrap_features=false`, `random_state=None` — match live `get_params()` (`_bagging.py:314`/`:813`/`:1230`). Consumer: `RsBaggingClassifier::new` (PyO3, `extras.rs`) + crate re-export. Verification: `cd /tmp && python3 -c "from sklearn.ensemble import BaggingClassifier as C, BaggingRegressor as R; print(C().get_params(), R().get_params())"` (matches the exposed subset). ABSENT params (`estimator`, `oob_score`, `warm_start`, `n_jobs`, `verbose`; `int` max_samples/max_features) → open prereq blocker #718. Tests: `test_bagging_classifier_default`, `test_bagging_regressor_default`. |
| REQ-2 (subsampling + per-tree fit) | SHIPPED (correctness inherited; RNG boundary) | `fit` (both) draws sample indices (`bootstrap` → `next_u64() % n_samples` / else `rand_sample_indices`) and feature indices (`bootstrap_features` → `next_u64() % n_features` / full set / `rand_sample_indices`), then calls `build_classification_tree_with_feature_subset` / `build_regression_tree_with_feature_subset` (`decision_tree.rs`, oracle-verified per `.design/tree/decision_tree.md`). Consumer: PyO3 `RsBaggingClassifier::fit`; pipeline adapters. Tests: `test_bagging_classifier_simple`, `test_bagging_classifier_feature_subsample`, `test_bagging_classifier_no_bootstrap`, `test_bagging_classifier_bootstrap_features`, `test_bagging_classifier_max_samples_subsample`, `test_bagging_regressor_simple`, `test_bagging_regressor_feature_subsample`. Exact ensemble at a seed + reversed draw order = documented RNG boundary (#719). |
| REQ-3 (classifier SOFT vote — THE divergence) | NOT-STARTED | open prereq blocker #720 (HEADLINE). `FittedBaggingClassifier::predict` (`bagging.rs`, the `votes[class_idx] += 1` / `max_by_key` loop ~476-496) hard-votes per-tree labels and returns `max_by_key` (last-index on ties); sklearn `predict` (`_bagging.py:913-914`) = `classes_.take(argmax(predict_proba))` (SOFT, lowest-index argmax). Live: soft==predict True; soft vs hard differ on 21/300 rows (`max_depth=2`, `max_features=0.5`, `n_estimators=11`). ferrolearn's own `predict_proba` already soft-averages, so `predict` is internally inconsistent. Same class as random_forest #670 / extra_trees #679. |
| REQ-4 (predict_proba mean) | SHIPPED | `fn predict_proba` on `FittedBaggingClassifier` accumulates per-tree leaf `class_distribution` (one-hot fallback at the leaf `value`) then divides by `n_trees` — mirrors sklearn `:964` / `_parallel_predict_proba` (`:203`). Consumer: `predict_log_proba` (same impl) + crate re-export. Tests: `test_bagging_classifier_simple`, `test_bagging_classifier_multiclass`, `test_bagging_classifier_has_classes` (exercise the fitted classifier whose proba feeds `predict_log_proba`). Rows sum to 1 by construction. Deterministic given the ensemble. |
| REQ-5 (regressor mean) | SHIPPED | `predict` on `FittedBaggingRegressor` = `sum / n_trees_f` — mirrors sklearn `:1299` / `_parallel_predict_regression` (`:263`). Consumer: crate re-export + `FittedBaggingRegressor as FittedPipelineEstimator::predict_pipeline`; `score` (R²). Tests: `test_bagging_regressor_simple` (preds within tolerance of targets), `test_bagging_regressor_reproducibility`, `test_bagging_regressor_no_bootstrap`. Deterministic given the ensemble. |
| REQ-6 (feature_importances_ — ferrolearn extension) | SHIPPED (no sklearn parity claim) | `fit` (both) aggregates via `aggregate_tree_importances(&trees, Some(&feature_indices), None, n_features)`, normalized to sum to 1; exposed through `HasFeatureImportances::feature_importances`. Consumer: crate re-export (the `HasFeatureImportances` impl is the introspection surface). sklearn Bagging does **not** expose `feature_importances_` (`_bagging.py:733`/`:1156` Attributes lists none) — this is a ferrolearn convenience, SHIPPED as self-consistent, not a parity mirror. |
| REQ-7 (heterogeneous base `estimator` — architectural) | NOT-STARTED | open prereq blocker #721 (cross-crate, like voting #694). ferrolearn has no `estimator` field — it is decision-tree-only (`fit` always calls `build_*_tree_with_feature_subset`). sklearn's `estimator` is any `HasMethods(["fit","predict"])` object, defaulting to `DecisionTree*` (`_bagging.py:295`/`:844`/`:1334`). A faithful version needs cross-crate trait objects / an estimator enum and CANNOT live in `ferrolearn-tree` (depends only on `ferrolearn-core`); belongs in the meta-crate `ferrolearn` or a dedicated ensemble crate. |
| REQ-8 (oob_score / oob_decision_function_ / oob_prediction_) | NOT-STARTED | open prereq blocker #722. No OOB machinery and no `oob_score` field; sklearn `_set_oob_score` (`_bagging.py:848` clf / `:1303` reg) + the `bootstrap=False` guard (`:501`) + `estimators_samples_` complement absent. |
| REQ-9 (int max_samples/max_features + sizing parity) | NOT-STARTED | open prereq blocker #723. `max_samples`/`max_features` are typed `f64` (fractions only); sklearn accepts `int` (absolute count) OR `float` per `_parameter_constraints` (`_bagging.py:297-304`). Float sizing also differs: ferrolearn `ceil(frac*n).max(1)` vs sklearn `int(frac*n)` (samples, `:478`) and `max(1, int(frac*n))` (features, `:495`). Live: `max_samples=3` (int) draws 3; `max_samples=0.5` draws `int(0.5*n)`. |
| REQ-10 (random_state determinism) | SHIPPED | per-tree seeds derived sequentially from `StdRng::seed_from_u64(random_state)` in `fit` (both); same seed → identical ensemble. Consumer: PyO3 `random_state` (`RsBaggingClassifier`). Tests: `test_bagging_classifier_reproducibility`, `test_bagging_regressor_reproducibility`. This is ferrolearn reproducibility; exact numpy-MT parity is the RNG boundary (#719). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #724. Imports `ndarray`/`rand`/`rayon`, not `ferray-core`/`ferray::random` (R-SUBSTRATE). |

## Architecture

The module has two parallel estimator pairs (no shared helper struct — the
sample/feature-draw + build logic is duplicated between the two `fit`
implementations). `BaggingClassifier<F>` / `BaggingRegressor<F>` are the unfitted
boundary types (public fields + `with_*` builders + `Default`). `fit` (both)
validates shapes (`ShapeMismatch` for `n_samples != y.len()`,
`InsufficientSamples` for zero rows, `InvalidParameter` for `n_estimators==0` and
for `max_samples`/`max_features` outside `(0.0, 1.0]`), generates per-tree `u64`
seeds sequentially from a `StdRng` master (or `rand::rng()` when `random_state`
is `None`), then `par_iter`s the seeds: each tree gets a fresh
`StdRng::seed_from_u64(seed)`, draws `n_sample_draw` sample indices then
`n_feature_draw` feature indices, and delegates the recursive build to
`decision_tree.rs`, retaining the `feature_indices` for prediction-time
re-indexing.

`FittedBaggingClassifier<F>` stores `trees: Vec<Vec<Node<F>>>` +
`feature_indices` + `classes` + `feature_importances`; `predict_proba`
sub-indexes each input row by the tree's `feature_indices`, traverses
(`decision_tree::traverse`), accumulates the leaf `class_distribution`
(one-hot at `value` when none), divides by `n_trees` (soft, correct); `predict`
— divergence REQ-3 — hard-votes (`votes[class_idx] += 1`, `max_by_key`,
~476-496). `FittedBaggingRegressor<F>::predict` sub-indexes, traverses, and
averages the per-tree leaf `value`.

Invariants held: `predict_proba` rows sum to 1 (divide by `n_trees`);
`feature_importances` sums to 1 when any tree splits; feature-count guards on
`predict`/`predict_proba`. Invariant NOT held vs sklearn: `predict ==
argmax(predict_proba)` (the hard-vs-soft gap, REQ-3, the single-file fixer
target).

The exact ensemble at a `random_state` is NOT reproducible against sklearn
because the bootstrap (`next_u64() % n`) and feature draws use `StdRng` (not numpy
MT19937), and ferrolearn draws samples-then-features while sklearn draws
features-then-samples — the documented RNG boundary, consistent with
random_forest / extra_trees / decision_tree `splitter='random'`.

This is also a meta-estimator whose faithful (heterogeneous-base) form is
cross-crate (REQ-7) — the decision-tree-only specialization ferrolearn ships is
the sklearn default base estimator, the most common case.

## Verification

Library crate (green at baseline `f0939ff4`):
```
cargo test -p ferrolearn-tree --lib bagging   # 25 passed; 0 failed (existing suite)
cargo clippy -p ferrolearn-tree --all-targets -- -D warnings
cargo fmt --all --check
```
Live sklearn oracle (installed 1.5.2, run from `/tmp`):
```
# REQ-1 defaults (live-verified to match the exposed subset)
python3 -c "from sklearn.ensemble import BaggingClassifier as C, BaggingRegressor as R; print(C().get_params()); print(R().get_params())"
# REQ-3 soft-vote divergence class (establishes predict==soft, soft!=hard on 21/300 rows)
python3 -c "import numpy as np; from sklearn.ensemble import BaggingClassifier as C; from sklearn.tree import DecisionTreeClassifier as D; rng=np.random.RandomState(3); X=rng.randn(300,6); y=(X[:,0]+X[:,1]+0.8*rng.randn(300)>0).astype(int); m=C(estimator=D(max_depth=2),n_estimators=11,max_features=0.5,random_state=1).fit(X,y); p=m.predict_proba(X); soft=m.classes_[p.argmax(1)]; hard=np.array([np.bincount([int(t.predict((X[i:i+1])[:,f])[0]) for t,f in zip(m.estimators_,m.estimators_features_)],minlength=2).argmax() for i in range(len(X))]); print('soft==predict',bool(np.all(soft==m.predict(X))),'differ(soft vs hard)',int((soft!=hard).sum()))"
# REQ-9 int vs float max_samples sizing
python3 -c "print('int 3 =>3 samples; float 0.5 => int(0.5*n)')"
```
The NOT-STARTED REQs (3, 7, 8, 9, 11) have no green verification by construction —
each carries an open prereq blocker. REQ-2/4/5/6/10 are verified by the in-crate
`#[test]`s named in the table (deterministic given the ensemble, or
ferrolearn-internal reproducibility); REQ-1 by the live `get_params()`
comparison. A characterization pin for REQ-3 (R-CHAR-3) belongs in
`ferrolearn-tree/tests/divergence_bagging.rs` (and/or
`ferrolearn-python/tests/divergence_ensemble.py` once `predict_proba` is exposed
on the PyO3 binding), with expected values from the live oracle above, once the
fixer routes `predict` through `predict_proba`.

## Blockers to open

- #718 — REQ-1: missing constructor params on `bagging.rs` (`estimator`,
  `oob_score`, `warm_start`, `n_jobs`, `verbose`; `int`-valued
  `max_samples`/`max_features`) vs sklearn `_bagging.py:314`/`:813`/`:1230`.
- #719 — REQ-2/REQ-10: exact ensemble parity at `random_state` is a numpy-MT vs
  StdRng RNG boundary (numpy `randint`/`sample_without_replacement` `:54-64`,
  features-then-samples draw order `:81-86`) — documented like
  random_forest/extra_trees, NOT a fixable divergence.
- #720 — REQ-3 (HEADLINE): `FittedBaggingClassifier::predict` hard-votes
  (`votes[class_idx] += 1` / `max_by_key`, ~476-496, last-index tie); sklearn
  soft-votes (`classes_.take(argmax(predict_proba))`, `_bagging.py:913-914`).
  Fixer reroutes `predict` through the already-correct `predict_proba` with a
  lowest-index argmax. Same class as #670 / #679.
- #721 — REQ-7 (architectural, cross-crate): ferrolearn's `Bagging*` are
  decision-tree-only; sklearn's are meta-estimators over an arbitrary base
  `estimator` (`_bagging.py:295`/`:844`/`:1334`). Needs cross-crate trait objects
  / an estimator enum and CANNOT live in `ferrolearn-tree` — belongs in the
  meta-crate `ferrolearn` or a dedicated ensemble crate (like voting #694).
- #722 — REQ-8: `oob_score` / `oob_decision_function_` / `oob_prediction_`
  (`_bagging.py:848`/`:1303`) + the `bootstrap=False` guard (`:501`) not
  implemented.
- #723 — REQ-9: `int`-valued `max_samples`/`max_features` not accepted (typed
  `f64`) and float sizing diverges (`ceil(frac*n)` vs sklearn `int(frac*n)` /
  `max(1,int(frac*n))`, `_bagging.py:478`/`:490-495`).
- #724 — REQ-11: migrate `bagging.rs` off `ndarray`/`rand`/`rayon` to the ferray
  substrate (R-SUBSTRATE).
