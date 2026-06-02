# Voting ensembles (VotingClassifier / VotingRegressor)

<!--
tier: 3-component
status: draft
baseline-commit: 8c8921be
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/ensemble/_voting.py    # _BaseVoting (:52); _BaseVoting.fit (:82, clones each named estimator, fits on full X via _fit_single_estimator :104); _predict (:78, stacks per-estimator predict -> (n_samples, n_estimators)); _weights_not_none (:71); VotingClassifier (:199); __init__(estimators, *, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False) (:338); fit (:363, LabelEncoder le_/classes_ :416-418, y_type guard :399-414); predict (:425, soft=argmax(predict_proba) :440 / hard=apply_along_axis bincount-argmax weighted :443-448, then le_.inverse_transform :450); _collect_probas (:454); _check_voting (:458, raises AttributeError for hard :460); predict_proba (:465 @available_if soft-only, np.average weighted :480); transform (:485, soft->stacked/flattened proba, hard->stacked labels); get_feature_names_out (:517); VotingRegressor (:555); __init__(estimators, *, weights=None, n_jobs=None, verbose=False) (:646); fit (:660, column_or_1d :694); predict (:699, np.average of _predict weighted :716); transform (:718)
  - sklearn/ensemble/_base.py      # _BaseHeterogeneousEnsemble._validate_estimators (named (str, estimator) tuples, 'drop' handling); _fit_single_estimator
ferrolearn-module: ferrolearn-tree/src/voting.rs
parity-ops: VotingClassifier, VotingRegressor
crosslink-issue: 693
-->

## Summary

**The central finding: this is not the same estimator.** scikit-learn's
`VotingClassifier` (`_voting.py:199`) and `VotingRegressor` (`:555`) are
**meta-estimators** over a user-supplied **heterogeneous** list of named
`(str, estimator)` tuples (`estimators`, a *required* positional argument,
`:338`/`:646`) — e.g. `[('lr', LogisticRegression()), ('rf',
RandomForestClassifier()), ('gnb', GaussianNB())]`. They clone and fit each
named estimator on the full dataset (`_BaseVoting.fit`, `:82`-`:128`), then
aggregate: the classifier toggles between **hard** voting (per-estimator label
`bincount`-argmax, `:443`-`:448`) and **soft** voting (argmax of the
weight-averaged `predict_proba`, `:440`) via the `voting` parameter, with
per-estimator `weights` (`:71`); the regressor takes the weight-averaged mean
of per-estimator predictions (`:716`).

`ferrolearn-tree/src/voting.rs` instead ships a **homogeneous decision-tree
ensemble**: `VotingClassifier<F>`/`VotingRegressor<F>` build a `Vec` of
`DecisionTree{Classifier,Regressor}` keyed by a `Vec<Option<usize>>` of
`max_depths` (`with_max_depths`) sharing one `min_samples_split` /
`min_samples_leaf` / `criterion`. There is **no `estimators` list, no `voting`
hard/soft toggle, no `weights`, no `le_`, no `named_estimators_`, no
`transform`/`flatten_transform`.** The classifier always **hard-votes**
(`votes[class_idx] += 1`, `max_by_key`, in `predict`) and always exposes an
ungated `predict_proba` (averaging per-tree proba); sklearn does the opposite —
`predict` follows the `voting` flag and `predict_proba` is *gated behind
`voting='soft'`* (`_check_voting` raises `AttributeError` for hard, `:458`-`:463`;
live-confirmed). The regressor averages per-tree outputs.

**This is NOT an R-DEV-7 "Rust analog" deviation — the observable contract is
not preserved.** Under R-DEV-7 a different implementation is legitimate only when
the observable contract (constructor surface, fitted attributes, output shapes,
the `voting`/`weights`/`estimators` semantics users rely on) is held. Here the
constructor surface (`estimators` required vs absent), the fitted attributes
(`le_`/`classes_`/`named_estimators_`/`estimators_` vs `trees`), and the
aggregation modes (hard/soft toggle + weights vs hard-only no-weights) all
differ. A faithful translation must hold **heterogeneous, cross-crate**
estimators (trait objects or an estimator enum spanning
`LogisticRegression`/`GaussianNB`/`RandomForest…`/`KNeighbors…`) and therefore
**cannot live in `ferrolearn-tree`** — `ferrolearn-tree` depends only on
`ferrolearn-core` (verified: its `Cargo.toml` has no `ferrolearn-linear` /
`ferrolearn-neighbors` / `ferrolearn-bayes` dependency, and adding one would
invert the documented dependency order in `goal.md`). The meta-estimator must
live in a crate that can depend on the leaf estimator crates (the meta-crate
`ferrolearn`, or a dedicated ensemble crate). **This is the architectural
blocker (#694), and it is the headline.**

This doc adapts to the **existing** code under R-HONEST-3 (underclaim beats
overclaim). The aggregation **math that is actually present** and matches
sklearn *for the homogeneous, `weights=None` case* is SHIPPED with deterministic,
verifiable claims; everything that constitutes the *core meta-estimator
contract* (the `estimators` list, `voting` toggle, `weights`, `transform`,
`le_`/`named_estimators_`, ferray substrate) is NOT-STARTED with a concrete
blocker.

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `get_params()`, sklearn 1.5.2)

`VotingClassifier(estimators, *, voting='hard', weights=None, n_jobs=None,
flatten_transform=True, verbose=False)` (`:338`). Live:
`{'estimators': [], 'flatten_transform': True, 'n_jobs': None, 'verbose': False,
'voting': 'hard', 'weights': None}`. `estimators` is a **required positional**
list of `(str, estimator)` tuples; a member may be set to the string `'drop'`.

`VotingRegressor(estimators, *, weights=None, n_jobs=None, verbose=False)`
(`:646`). Live: `{'estimators': [], 'n_jobs': None, 'verbose': False,
'weights': None}` (no `voting`, no `flatten_transform`).

ferrolearn exposes **none** of these. Its surface is `with_max_depths(Vec<Option
<usize>>)` + `with_min_samples_split` + `with_min_samples_leaf` + (clf)
`with_criterion` — a decision-tree hyperparameter grid, not an estimator list.

### fit (`_BaseVoting.fit`, `:82`)

`_validate_estimators` unzips the `(name, est)` tuples; `weights`, if given,
must match `len(estimators)` (`:87`-`:91`). Each non-`'drop'` estimator is
**cloned** and fit on the **full** `(X, y)` by `_fit_single_estimator` (`:104`).
Fitted clones land in `self.estimators_`; `named_estimators_` is a `Bunch`
keyed by name (`:117`-`:123`). The **classifier** additionally fits a
`LabelEncoder` (`le_`, `:416`), sets `classes_ = le_.classes_` (`:417`), and fits
the members on label-encoded `y` (`:418`); it guards `y_type` against
continuous/multilabel targets (`:399`-`:414`). The **regressor** runs
`column_or_1d(y)` (`:694`).

ferrolearn `fit` (both): trains one `DecisionTree*` per `max_depths` entry on
the full `(X, y)`, no cloning of arbitrary estimators, no `LabelEncoder` (it
collects `classes` by `sort_unstable` + `dedup` of the raw `usize` labels), no
`y_type` guard, no `named_estimators_`.

### Classifier aggregation (`predict`, `:425`)

- **hard** (default): `predictions = self._predict(X)` stacks per-estimator
  labels into `(n_samples, n_estimators)` (`_predict`, `:78`-`:80`), then per row
  `np.argmax(np.bincount(x, weights=self._weights_not_none))` (`:443`-`:448`) —
  the most-voted label, **lowest index on a tie** (numpy `bincount`+`argmax`;
  live-confirmed `argmax(bincount([0,1])) == 0`), weighted by `weights`. The
  result is mapped back through `le_.inverse_transform` (`:450`).
- **soft**: `np.argmax(self.predict_proba(X), axis=1)` (`:440`), i.e. argmax of
  the weight-averaged per-estimator `predict_proba` (`:480`-`:482`).

ferrolearn `predict` (`predict` on `FittedVotingClassifier`): always the
**hard** path — `votes[class_idx] += 1` per tree (uniform, no weights), then
`max_by_key(count)`. Rust's `Iterator::max_by_key` returns the **last** maximum
on a tie, but because votes are accumulated in ascending `classes` order the
selected index is the highest-count class with — note — the **last** index on a
tie, the **opposite** of numpy's lowest-index `argmax(bincount)`. (This tie
divergence is real but only observable on exact vote ties; it is folded into the
NOT-STARTED `voting`/`weights` REQ rather than claimed SHIPPED, per R-HONEST-3.)

### Classifier `predict_proba` (`:465`) and the `voting` gate

`predict_proba` is decorated `@available_if(_check_voting)` (`:465`):
`_check_voting` **raises `AttributeError`** when `voting='hard'` (`:458`-`:462`;
live-confirmed). When available (soft), it returns `np.average(_collect_probas
(X), axis=0, weights=self._weights_not_none)` (`:480`) — the weight-averaged
mean of per-estimator `predict_proba`, shape `(n_samples, n_classes)`.

ferrolearn `predict_proba` (`predict_proba` on `FittedVotingClassifier`) is
**always present** (no `voting` gate) and is the **uniform** mean of per-tree
`predict_proba` (`sum / n_trees`). For the homogeneous, `weights=None`,
`voting='soft'` case this *formula* matches sklearn's `np.average(..., axis=0)`
with uniform weights.

### Regressor aggregation (`predict`, `:699`)

`np.average(self._predict(X), axis=1, weights=self._weights_not_none)` (`:716`)
— the weight-averaged mean of per-estimator predictions. ferrolearn `predict`
(`predict` on `FittedVotingRegressor`) is the **uniform** mean (`sum /
n_trees_f`); matches sklearn for `weights=None`.

### transform / flatten_transform (`:485`, `:718`)

`VotingClassifier.transform` (`:485`): soft → stacked per-estimator proba
(`(n_estimators, n_samples, n_classes)`), flattened to `(n_samples,
n_estimators*n_classes)` when `flatten_transform=True` (`:508`-`:512`); hard →
stacked labels `(n_samples, n_estimators)` (`:515`). `VotingRegressor.transform`
(`:718`): stacked per-estimator predictions `(n_samples, n_estimators)`.
ferrolearn has **no `transform`** (it is not a `Transform` impl), and no
`flatten_transform`.

## ferrolearn (what exists)

- **Unfitted**: `pub struct VotingClassifier<F>` (public fields `max_depths:
  Vec<Option<usize>>`, `min_samples_split`, `min_samples_leaf`, `criterion`);
  `pub struct VotingRegressor<F>` (same minus `criterion`); `with_max_depths` /
  `with_min_samples_split` / `with_min_samples_leaf` / (clf) `with_criterion`;
  `Default` / `fn new` (default `max_depths = [Some(2), Some(4), Some(6), None]`).
- **Fitted**: `pub struct FittedVotingClassifier<F>` (`trees:
  Vec<FittedDecisionTreeClassifier<F>>`, `classes: Vec<usize>`);
  `pub struct FittedVotingRegressor<F>` (`trees:
  Vec<FittedDecisionTreeRegressor<F>>`).
- **Traits**: `Fit<Array2<F>, Array1<usize>>` (clf) / `Fit<Array2<F>, Array1<F>>`
  (reg); `Predict`; `HasClasses` (clf); `PipelineEstimator` /
  `FittedPipelineEstimator` (both, via adapters).
- **Methods**: `fn n_estimators`, `fn predict_proba`, `fn predict_log_proba`,
  `fn score` (clf, mean accuracy); `fn n_estimators`, `fn score` (reg, R²).
- **Build delegation** (from `decision_tree.rs`): `DecisionTreeClassifier` /
  `DecisionTreeRegressor` + their `Fitted*` + `ClassificationCriterion`. Per-tree
  correctness is inherited from the oracle-verified `decision_tree.rs`
  (`.design/tree/decision_tree.md`).
- **Consumers**: crate re-export — `ferrolearn-tree/src/lib.rs`
  (`pub use voting::{FittedVotingClassifier, FittedVotingRegressor,
  VotingClassifier, VotingRegressor}`). There is **no PyO3 binding** for the
  Voting types (verified: no `Voting` symbol in `ferrolearn-python/src/`), so the
  crate re-export + the pipeline adapters are the only non-test production
  consumers.

## Requirements

- REQ-1: **Heterogeneous `estimators=[(name, est)]` list (THE headline,
  architectural).** Hold a user-supplied list of named, *different* fitted
  estimators (cross-crate: linear, bayes, neighbors, forest…), clone + fit each
  on full `(X, y)` (`_BaseVoting.fit`, `:82`), expose `estimators_` /
  `named_estimators_` (`:117`). Requires cross-crate trait objects / an estimator
  enum in a crate that may depend on the leaf crates — **not `ferrolearn-tree`**.
- REQ-2: **`voting` hard/soft toggle (classifier).** `voting='hard'` →
  `bincount`-argmax of per-estimator labels (`:443`); `voting='soft'` → argmax
  of averaged proba (`:440`); default `'hard'`. ferrolearn is hard-only.
- REQ-3: **`predict_proba` gated behind `voting='soft'`.** `_check_voting`
  raises `AttributeError` for `voting='hard'` (`:458`-`:462`); ferrolearn always
  exposes it.
- REQ-4: **Per-estimator `weights`.** Weight the label `bincount` (hard), the
  proba average (soft, `:480`), and the regressor average (`:716`);
  `len(weights) == len(estimators)` validation (`:87`). ferrolearn has no weights.
- REQ-5: **`transform` / `flatten_transform` (TransformerMixin).** clf soft →
  stacked/flattened proba, hard → stacked labels (`:485`); reg → stacked
  predictions (`:718`). ferrolearn has no `Transform` impl.
- REQ-6: **`le_` / `classes_` via `LabelEncoder` + `named_estimators_` + y_type
  guard.** Label-encode `y` (`:416`-`:418`), `inverse_transform` on predict
  (`:450`), guard continuous/multilabel targets (`:399`-`:414`), name-keyed
  `Bunch` (`:117`). ferrolearn uses raw `usize` labels + `sort`/`dedup`, no
  encoder, no name keying, no guard.
- REQ-7: **Classifier hard-vote aggregation math (homogeneous, weights=None).**
  Per-row majority of per-estimator labels (the `voting='hard'`, `weights=None`
  case of `:443`-`:448`). PRESENT in ferrolearn `predict` — but the tie-break
  differs (numpy lowest-index vs Rust `max_by_key` last-index), folded into
  REQ-2/REQ-4.
- REQ-8: **Soft `predict_proba` = uniform mean of per-estimator proba
  (weights=None).** sklearn `np.average(..., axis=0)` with uniform weights
  (`:480`). PRESENT in ferrolearn `predict_proba` (`sum / n_trees`). Deterministic,
  pinnable.
- REQ-9: **Regressor mean (weights=None).** sklearn `np.average(_predict, axis=1)`
  uniform (`:716`). PRESENT in ferrolearn `predict` (`sum / n_trees_f`).
  Deterministic, pinnable.
- REQ-10: **ferray substrate (R-SUBSTRATE).** Imports `ndarray`, not
  `ferray-core`.

## Acceptance criteria

- AC-1: live `VotingClassifier(estimators=[]).get_params()` /
  `VotingRegressor(estimators=[]).get_params()` show `estimators`, `voting`
  (clf), `weights`, `flatten_transform` (clf), `n_jobs`, `verbose` — none of
  which ferrolearn exposes; ferrolearn's surface is `max_depths` +
  `min_samples_*` + `criterion`.
- AC-2: live `VotingClassifier(..., voting='hard').predict_proba` raises
  `AttributeError`; ferrolearn `FittedVotingClassifier::predict_proba` always
  returns a value (gate absent).
- AC-3: for a fixed homogeneous decision-tree set, ferrolearn `predict` (clf)
  equals the per-row majority of the per-tree labels, **except** on exact ties,
  where it selects the last-index class while numpy `argmax(bincount)` selects
  the lowest-index class (REQ-7 tie divergence, folded into REQ-2/REQ-4).
- AC-4: for a fixed homogeneous set, ferrolearn `predict_proba` rows sum to 1
  and equal `mean` of per-tree `predict_proba` (the `weights=None` soft case of
  `_voting.py:480`); ferrolearn matches to 1e-12 intra-ferrolearn.
- AC-5: for a fixed homogeneous set, ferrolearn regressor `predict` equals
  `mean` of per-tree `predict(X)` (the `weights=None` case of `:716`); matches to
  1e-12 intra-ferrolearn.

## REQ status table

Binary (R-DEFER-2). `VotingClassifier`/`VotingRegressor` are crate-root
re-exported boundary types (S5/R-DEFER-1 non-test consumer surface is the
`lib.rs` re-export + the pipeline adapters; there is **no** PyO3 binding). Cites
use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Under R-HONEST-3,
only the deterministic aggregation math actually present is claimed SHIPPED; the
meta-estimator contract is NOT-STARTED.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (heterogeneous `estimators` list — THE headline) | NOT-STARTED | open prereq blocker #694 (**architectural, cross-crate**). ferrolearn has no `estimators` field at all — it keys on `max_depths: Vec<Option<usize>>` and builds homogeneous `DecisionTree*` (`fit` on `VotingClassifier`/`VotingRegressor`). sklearn requires a list of named heterogeneous `(str, estimator)` tuples (`_voting.py:338`/`:646`, `_BaseVoting.fit:82`). A faithful version needs cross-crate trait objects / an estimator enum and CANNOT live in `ferrolearn-tree` (its `Cargo.toml` depends only on `ferrolearn-core`; depending on `ferrolearn-linear`/`-neighbors`/`-bayes` inverts the `goal.md` dependency order) — it belongs in the meta-crate `ferrolearn` or a dedicated ensemble crate. |
| REQ-2 (`voting` hard/soft toggle) | NOT-STARTED | open prereq blocker #695. ferrolearn `predict` is hard-only (`votes[class_idx] += 1` + `max_by_key` in `predict`); no `voting` param. sklearn toggles soft=`argmax(predict_proba)` (`:440`) vs hard=`bincount`-argmax (`:443`). Also pins the tie-break divergence (Rust `max_by_key` last-index vs numpy `argmax(bincount)` lowest-index, REQ-7). |
| REQ-3 (`predict_proba` gated behind `voting='soft'`) | NOT-STARTED | open prereq blocker #696. ferrolearn `predict_proba` is always available; sklearn `_check_voting` raises `AttributeError` when `voting='hard'` (`_voting.py:458`-`:462`, live-confirmed). |
| REQ-4 (per-estimator `weights`) | NOT-STARTED | open prereq blocker #697. No `weights` field; sklearn weights the `bincount` (hard), the proba average (`:480`), and the regressor average (`:716`), with `len(weights)==len(estimators)` validation (`:87`-`:91`). |
| REQ-5 (`transform` / `flatten_transform`) | NOT-STARTED | open prereq blocker #698. No `Transform` impl and no `flatten_transform`; sklearn `transform` returns stacked/flattened proba or labels (clf `:485`) / stacked predictions (reg `:718`). |
| REQ-6 (`le_`/`classes_`/`named_estimators_` + y_type guard) | NOT-STARTED | open prereq blocker #699. ferrolearn collects `classes` by `sort_unstable`+`dedup` of raw `usize` labels (`fit`), with no `LabelEncoder` `le_`, no `inverse_transform` (`:450`), no name-keyed `named_estimators_` (`:117`), and no continuous/multilabel `y_type` guard (`:399`-`:414`). |
| REQ-7 (classifier hard-vote math, homogeneous + weights=None) | SHIPPED (tie-break divergence pinned in REQ-2) | `predict` on `FittedVotingClassifier` tallies per-tree labels (`votes[class_idx] += 1`) and returns the majority class — mirrors the `voting='hard'`, `weights=None` path of sklearn `_voting.py:443`-`:448` (`argmax(bincount)`). Consumer: crate re-export (`lib.rs`) + `FittedVotingClassifierPipelineAdapter::predict_pipeline`. Tests: `test_voting_classifier_fit_predict`, `test_voting_classifier_multiclass`, `test_voting_classifier_f32`. Verification: `cargo test -p ferrolearn-tree --lib voting` (19 passed, 0 failed). **Tie-break differs** (Rust `max_by_key` last-index vs numpy lowest-index) → folded into #695. |
| REQ-8 (soft `predict_proba` = uniform mean, weights=None) | SHIPPED | `predict_proba` on `FittedVotingClassifier` accumulates per-tree `predict_proba` then divides by `n_trees` — equals sklearn `np.average(_collect_probas, axis=0)` with uniform weights (`_voting.py:480`). Consumer: `predict_log_proba` (same impl) + crate re-export. Tests: `test_voting_classifier_fit_predict`, `test_voting_classifier_multiclass` (exercise the fitted classifier whose proba feeds `predict_log_proba`). Rows sum to 1 by construction (divide by `n_trees`). NOTE: in sklearn this method is gated to `voting='soft'` (REQ-3) — the *formula* is SHIPPED, the *gating contract* is NOT-STARTED. |
| REQ-9 (regressor mean, weights=None) | SHIPPED | `predict` on `FittedVotingRegressor` = `sum / n_trees_f` — equals sklearn `np.average(_predict, axis=1)` with uniform weights (`_voting.py:716`). Consumer: crate re-export + `FittedVotingRegressor as FittedPipelineEstimator::predict_pipeline`; `score` (R²). Tests: `test_voting_regressor_fit_predict`, `test_voting_regressor_averaging` (single unlimited tree overfits to 1e-10), `test_voting_regressor_f32`. Deterministic given the tree set. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #700. `voting.rs` imports `ndarray::{Array1, Array2}`, not `ferray-core` (R-SUBSTRATE). |

## Architecture

The module has two parallel estimator pairs. `VotingClassifier<F>` /
`VotingRegressor<F>` are unfitted boundary types: public hyperparameter fields
(`max_depths`, `min_samples_split`, `min_samples_leaf`, clf `criterion`) +
`with_*` builders + `Default`/`fn new`. `fit` (both) validates shapes
(`ShapeMismatch` for `nrows != y.len()`, `InsufficientSamples` for zero rows,
`InvalidParameter` for empty `max_depths`), then trains one
`DecisionTree{Classifier,Regressor}` per `max_depths` entry on the **full**
`(X, y)` (no bootstrap — diversity comes only from depth). The classifier
collects `classes` by `sort_unstable` + `dedup`.

`FittedVotingClassifier<F>` stores `trees` + `classes`; `predict` hard-votes
(`votes[class_idx] += 1`, `max_by_key`), `predict_proba` uniform-averages
per-tree proba, `predict_log_proba` logs it, `score` is mean accuracy.
`FittedVotingRegressor<F>` stores `trees`; `predict` uniform-averages, `score`
is R². Both expose `n_estimators`. Pipeline integration: `VotingClassifier`'s
adapter maps `usize` labels to `F`; `FittedVotingRegressor` *is* a
`FittedPipelineEstimator` directly.

**Contract held vs sklearn (homogeneous, weights=None only):** the hard-vote
majority formula (REQ-7), the uniform soft-proba mean (REQ-8), the uniform
regressor mean (REQ-9) — all deterministic given the tree set and pinnable
intra-ferrolearn.

**Contract NOT held vs sklearn:** the *identity of the estimator*. There is no
`estimators` list (REQ-1, architectural), no `voting` toggle (REQ-2), no proba
gating (REQ-3), no `weights` (REQ-4), no `transform`/`flatten_transform`
(REQ-5), no `le_`/`named_estimators_`/y_type guard (REQ-6). The hard-vote
tie-break also differs (numpy lowest-index `argmax(bincount)` vs Rust
last-index `max_by_key`).

**Why REQ-1 cannot be fixed in `ferrolearn-tree`.** A faithful meta-estimator
holds heterogeneous fitted estimators from *other* crates
(`LogisticRegression`, `GaussianNB`, `KNeighborsClassifier`,
`RandomForestClassifier`, …). `ferrolearn-tree`'s `Cargo.toml` depends only on
`ferrolearn-core`; per `goal.md`'s dependency order, the leaf estimator crates
(linear, bayes, neighbors) sit *beside* tree, not below it, and the composing
layer is the meta-crate `ferrolearn` (or a new ensemble crate that may depend on
all of them). A trait-object / estimator-enum design is therefore a cross-crate
change — the headline architectural blocker.

## Verification

Library crate (green at baseline `8c8921be`):
```
cargo test -p ferrolearn-tree --lib voting   # 19 passed; 0 failed
cargo clippy -p ferrolearn-tree --all-targets -- -D warnings
cargo fmt --all --check
```
Live sklearn oracle (installed 1.5.2, run from `/tmp`):
```
# REQ-1 surface (estimators required; voting/weights/flatten_transform present)
python3 -c "from sklearn.ensemble import VotingClassifier as C, VotingRegressor as R; print(C(estimators=[]).get_params()); print(R(estimators=[]).get_params())"
# REQ-3 predict_proba gated behind voting='soft'
python3 -c "import numpy as np; from sklearn.ensemble import VotingClassifier; from sklearn.tree import DecisionTreeClassifier as D; c=VotingClassifier(estimators=[('a',D())],voting='hard').fit(np.array([[0.],[1.]]),np.array([0,1]));
try: c.predict_proba; print('available (WRONG)')
except AttributeError as e: print('raises', type(e).__name__)"
# REQ-7 hard-vote tie-break: numpy argmax(bincount) is lowest-index
python3 -c "import numpy as np; print('lowest-index tie:', int(np.argmax(np.bincount([0,1]))))"
```
The NOT-STARTED REQs (1, 2, 3, 4, 5, 6, 10) have no green verification by
construction — each carries an open prereq blocker. REQ-7/8/9 are verified by
the in-crate `#[test]`s named in the table (deterministic given the tree set,
`weights=None`, homogeneous). A characterization pin for the REQ-7 tie-break
divergence and the REQ-3 gating divergence (R-CHAR-3) belongs in a future
`ferrolearn-tree/tests/divergence_voting.rs` (or in the composing crate once
REQ-1 lands), with expected values from the live oracle above.

## Blockers to open

- #694 — REQ-1 (HEADLINE, architectural): ferrolearn's `Voting*` are
  homogeneous decision-tree ensembles keyed by `max_depths`; sklearn's are
  meta-estimators over a required heterogeneous `estimators=[(name, est)]` list
  (`_voting.py:338`/`:646`, `_BaseVoting.fit:82`). A faithful translation needs
  cross-crate trait objects / an estimator enum and CANNOT live in
  `ferrolearn-tree` (depends only on `ferrolearn-core`) — belongs in the
  meta-crate `ferrolearn` or a dedicated ensemble crate.
- #695 — REQ-2: no `voting` hard/soft toggle (`predict` is hard-only,
  `_voting.py:440`/`:443`); also the hard-vote tie-break divergence (Rust
  `max_by_key` last-index vs numpy `argmax(bincount)` lowest-index, `:445`).
- #696 — REQ-3: `predict_proba` not gated behind `voting='soft'`; sklearn
  `_check_voting` raises `AttributeError` for `voting='hard'`
  (`_voting.py:458`-`:462`).
- #697 — REQ-4: no per-estimator `weights` (hard `bincount` weights `:445`, soft
  proba average `:480`, regressor average `:716`; `len(weights)==len(estimators)`
  validation `:87`-`:91`).
- #698 — REQ-5: no `transform` / `flatten_transform` (`TransformerMixin`;
  `_voting.py:485` clf, `:718` reg).
- #699 — REQ-6: no `LabelEncoder` `le_` / `classes_` decode-on-predict
  (`_voting.py:416`-`:418`, `:450`), no `named_estimators_` `Bunch` (`:117`), no
  continuous/multilabel `y_type` guard (`:399`-`:414`).
- #700 — REQ-10: migrate `voting.rs` off `ndarray` to the ferray substrate
  (R-SUBSTRATE).
