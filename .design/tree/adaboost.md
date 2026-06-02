# AdaBoost Classifier (AdaBoostClassifier — SAMME / SAMME.R)

<!--
tier: 3-component
status: draft
baseline-commit: 7d9078c1
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/ensemble/_weight_boosting.py   # BaseWeightBoosting (:63); _parameter_constraints (:70, n_estimators>=1, learning_rate in (0,inf)); BaseWeightBoosting.fit (:111, sample_weight init 1/n :143-146, clip eps :164, per-iter _boost :169, stop on error==0 :180, normalize :200-202); _samme_proba (:318-336); AdaBoostClassifier (:339); _parameter_constraints algorithm StrOptions{SAMME,SAMME.R} (:493-498); __init__ defaults (:501-517, n_estimators=50/learning_rate=1.0/algorithm='SAMME.R'); _validate_estimator default DecisionTreeClassifier(max_depth=1) (:519-547), SAMME.R deprecation FutureWarning (:526-534); _boost dispatch (:554-597); _boost_real SAMME.R (:601-658) — weighted fit :605, predict_proba :607, incorrect :616, estimator_error :619, perfect-fit return :622-623, y_coding :634-635, xlogy estimator_weight :644-649, exp reweight :652-656; _boost_discrete SAMME (:660-708) — weighted fit :664, predict :666, incorrect :673, estimator_error :676, perfect-fit return :679-680, worse-than-random discard :685-693, estimator_weight :696-698, log+alpha*incorrect reweight :701-706; predict (:710-732, binary pred>0 :729-730, argmax :732); decision_function (:768-813, SAMME.R sum _samme_proba :796-798, SAMME np.where(correct,w,-w/(K-1)) :799-807, /sum(weights) :809, binary collapse :810-812); predict_proba (:891-917, softmax of decision/(K-1) via _compute_proba_from_decision :872-889); feature_importances_ (:277-308, sum weight*clf.fi / norm :300-308)
ferrolearn-module: ferrolearn-tree/src/adaboost.rs
parity-ops: AdaBoostClassifier
crosslink-issue: 708
-->

## Summary

`ferrolearn-tree/src/adaboost.rs` mirrors scikit-learn's
`sklearn.ensemble.AdaBoostClassifier` (`_weight_boosting.py:339`) — the
multi-class **AdaBoost-SAMME** family (Zhu et al. 2009) over decision-tree
**stumps** (`DecisionTreeClassifier(max_depth=1)`, `_validate_estimator`,
`:521`). Two variants are dispatched by the `AdaBoostAlgorithm` enum
(`_boost`, `:554-597`):

- **SAMME** (discrete) — `_boost_discrete` (`:660-708`): weighted error → an
  `estimator_weight = lr*(log((1-err)/err) + log(K-1))` → exponential reweight
  of the misclassified samples; predict by a per-class weighted vote.
- **SAMME.R** (real) — `_boost_real` (`:601-658`): uses the base learner's
  class **probabilities**, a `xlogy`-based reweight, equal estimator weights,
  and a `_samme_proba` (`:318-336`) log-probability vote.

ferrolearn re-implements this natively. The module ships the unfitted
`AdaBoostClassifier<F>` + `FittedAdaBoostClassifier<F>`, the
`AdaBoostAlgorithm` enum (`Samme`/`SammeR`), `fn new` + `with_*` builders +
`Default`, `Fit`/`Predict`, `score` (mean accuracy), `predict_proba`,
`predict_log_proba`, `decision_function`, `HasClasses`,
`HasFeatureImportances`, the private `fn resample_weighted`, and the
`PipelineEstimator`/`FittedPipelineEstimator` adapters. The per-stump build
correctness (best split, leaf class, leaf distribution) is inherited from the
oracle-verified `decision_tree.rs` (`.design/tree/decision_tree.md`) via
`build_classification_tree_with_feature_subset`.

**Three divergence classes drive the REQ split (R-HONEST-3 — underclaim):**

1. **HEADLINE structural divergence (R-DEV-1) — resample-vs-weighted base fit,
   NOT a boundary.** sklearn fits each base stump on the **weighted** data
   directly: `estimator.fit(X, y, sample_weight=sample_weight)`
   (`_boost_discrete:664`, `_boost_real:605`), with **no RNG** in the
   classifier path (the bootstrap RNG is regressor-only). ferrolearn instead
   fits an **unweighted** stump on a **deterministic systematic resample** of
   the rows (`resample_weighted` at `adaboost.rs`, called in `fit_samme` /
   `fit_samme_r`) and never passes `sample_weight` to the builder. These
   produce different stumps → different per-round error/alpha → different
   reweighting → **different end-to-end predictions**. Because sklearn's
   classifier path is itself **deterministic** (no `random_state.choice`),
   end-to-end parity is *achievable* once ferrolearn fits weighted stumps —
   unlike the regressor's stochastic-bootstrap boundary
   (`.design/tree/adaboost_regressor.md` REQ-7). A weighted classification
   builder substrate **already exists**: `ClassificationData.sample_weight:
   Option<&[F]>` is oracle-verified for `class_weight` (decision-tree REQ-7),
   but `build_classification_tree_with_feature_subset` hard-codes
   `sample_weight: None` (`decision_tree.rs:2448`). This is a **2-file change**
   (a new weighted builder entry point in `decision_tree.rs` + the
   `fit_samme`/`fit_samme_r` call sites), not single-file. REQ-6, blocker #710.

2. **Missing perfect-fit `estimator_weight = 1.0` guard (R-DEV-1) — single-file
   fixer-able.** sklearn returns `(sample_weight, 1.0, 0.0)` the instant
   `estimator_error <= 0` (`_boost_discrete:679-680`, `_boost_real:622-623`),
   and `fit` then breaks on `error == 0` (`:180`). ferrolearn's `fit_samme` has
   **no such guard**: a perfect stump (`err == 0`) flows into the alpha formula
   as `alpha = lr*(ln((1-0).max(eps)/0.max(eps)) + ln(K-1)) = lr*(ln(1/1e-10) +
   ln(K-1)) ≈ 23*lr + lr*ln(K-1)` instead of `1.0`, and boosting continues
   instead of stopping. Fixable entirely inside `fit_samme` (and a matching
   guard for the SAMME.R reweight). REQ-7, blocker #711.

3. **Default `algorithm` — DELIBERATE R-DEV-6 deviation.** ferrolearn's
   `fn new` defaults `algorithm = AdaBoostAlgorithm::Samme`; sklearn 1.5.2's
   live default is `'SAMME.R'` (`__init__:507`, confirmed by
   `AdaBoostClassifier().get_params()` → `{'algorithm': 'SAMME.R', ...}`).
   sklearn **itself** deprecated `'SAMME.R'` in 1.4 with a `FutureWarning`
   ("The SAMME.R algorithm (the default) is deprecated and will be removed in
   1.6", `_validate_estimator:526-534`) and removed it in 1.6, making `SAMME`
   the sole, default option. This is the R-DEV-6 case ("sklearn is wrong by
   their own admission"): defaulting to `SAMME` ships the behavior sklearn is
   converging on. The "match the literal 1.5.2 default string" REQ is therefore
   classified NOT-STARTED honestly (REQ-1b, blocker #712) while the rest of the
   param surface is SHIPPED.

The *deterministic, structural* parts that do NOT depend on the base-fit
divergence — the param surface (modulo the algorithm-default deviation), the
SAMME `estimator_weight`/reweight formulas, and the worse-than-random stop —
match sklearn term-for-term and are SHIPPED. The predict-family
(`decision_function`/`predict`/`predict_proba`) carries an **independent**
formula divergence (no `-w/(K-1)` term, no `/sum(weights)`, no softmax) that is
NOT-STARTED on its own. The weighted-base-fit (REQ-6, #709) and the
perfect-fit guard (REQ-7, #710) are now SHIPPED; the SAMME.R reweight,
end-to-end parity, PyO3 binding, and ferray substrate remain NOT-STARTED with
concrete blockers.

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `get_params()`, sklearn 1.5.2)

`AdaBoostClassifier().get_params()` →
`{'algorithm': 'SAMME.R', 'estimator': None, 'learning_rate': 1.0,
'n_estimators': 50, 'random_state': None}`. Base estimator default (when
`estimator is None`): `DecisionTreeClassifier(max_depth=1)`
(`_validate_estimator`, `:521`). Constraints (`:70`, `:493-498`): `n_estimators`
integer `>= 1`, `learning_rate` real in `(0, inf)`, `algorithm in {'SAMME',
'SAMME.R'}`.

**Defaults ferrolearn matches** (`fn new`): `n_estimators=50`,
`learning_rate=1.0`, base stump `max_depth=1`, `random_state=None`. Validation
in `fit` rejects `n_estimators==0` (`InvalidParameter`), `learning_rate <= 0`
(`InvalidParameter`), `n_classes < 2` (`InvalidParameter`), and shape/empty
inputs (`ShapeMismatch`/`InsufficientSamples`), mirroring the constraint
intervals.

**Deliberate default deviation (R-DEV-6, REQ-1b):** ferrolearn `algorithm`
defaults to `Samme`; sklearn 1.5.2 defaults to `'SAMME.R'`
(`__init__:507`) — but 1.5.2 deprecates `'SAMME.R'` with a `FutureWarning`
(`:526-534`) and 1.6 removes it. ferrolearn ships sklearn's destination
behavior.

**Surface differences (not divergences for the default path, noted for
honesty):** ferrolearn exposes the stump via the hardwired tree builder rather
than a pluggable `estimator`; sklearn's pluggable `estimator` is not generic in
ferrolearn (REQ-9). ferrolearn does not expose `estimator_errors_` (sklearn
`:154`/`:177`, REQ-9). `learning_rate` is stored as `f64` (not the generic `F`)
on `AdaBoostClassifier<F>` — benign for the default `f64` instantiation and for
`f32` (the value is `F::from(self.learning_rate)`-converted at the top of
`fit_samme`/`fit_samme_r`), but it is an inconsistency with the generic
contract noted under REQ-1.

### fit loop (`BaseWeightBoosting.fit`, `:111`)

sklearn: init `sample_weight = 1/n` (`:143-146`); each iteration clip to machine
eps (`:164`), call `_boost`, store `estimator_weights_[iboost]`/
`estimator_errors_[iboost]`, **stop if `estimator_error == 0`** (`:180`),
stop on non-finite/non-positive weight sum (`:185-198`), normalize
`sample_weight /= sum` unless the last iteration (`:200-202`).

ferrolearn (`fit` → `fit_samme`/`fit_samme_r`): init `weights = 1/n_samples`
uniform (matches `:143-146`); loop `n_estimators` times: `resample_weighted`
→ stump build → predictions/probabilities → weighted error → alpha → reweight →
normalize. **Divergences:** (a) the base fit uses a resample instead of weighted
training (REQ-6); (b) no perfect-fit `err==0 → weight 1.0, stop` (REQ-7);
(c) ferrolearn always reweights+normalizes every round including the last —
benign, the final round's weights are never consumed.

### _boost_discrete — SAMME (`:660-708`)

- **Weighted base fit** (`:664`): `estimator.fit(X, y,
  sample_weight=sample_weight)` — deterministic, NO bootstrap, NO RNG.
  ferrolearn: `build_classification_tree_with_feature_subset(... indices ...)`
  on `resample_weighted(&weights, n_samples)` with the builder's
  `sample_weight: None`. **HEADLINE structural divergence — REQ-6, blocker
  #710.**
- **Weighted error** (`:673-676`): `incorrect = y_predict != y`;
  `estimator_error = mean(average(incorrect, weights=sample_weight))` =
  `sum(w_i * [pred_i != y_i]) / sum(w)`. ferrolearn computes `weighted_error =
  sum(w_i over misclassified)` then `err = weighted_error / weight_sum`.
  Same quantity (REQ-3).
- **Perfect-fit** (`:679-680`): `if estimator_error <= 0: return sample_weight,
  1.0, 0.0`. ferrolearn: **absent** — REQ-7, blocker #711.
- **Worse-than-random stop** (`:685-693`): `if estimator_error >= 1 - 1/K:` pop
  the estimator; raise if it was the only one, else terminate. ferrolearn:
  `if err >= 1 - 1/K { if estimators.is_empty() { push tree, weight 1.0 } break
  }` — same threshold; ferrolearn *keeps* the worst-case first stump (with
  weight 1.0) rather than raising, a benign Rust-side robustness choice that
  still terminates boosting (REQ-4).
- **estimator_weight** (`:696-698`): `lr * (log((1-err)/err) + log(K-1))`.
  ferrolearn: `lr*((1-err).max(eps)/err.max(eps)).ln() + lr*(K-1).ln()` —
  algebraically identical (the `.max(eps)` only matters at the err==0 boundary
  that REQ-7 owns). **Match (REQ-2).**
- **Reweight** (`:701-706`): `sample_weight = exp(log(sample_weight) +
  estimator_weight * incorrect * (sample_weight > 0))` = `w_i * exp(alpha)` for
  misclassified positive-weight samples, unchanged otherwise. ferrolearn:
  `if preds[i] != y_mapped[i] { weights[i] *= alpha.exp() }`. **Match (REQ-2).**

### _boost_real — SAMME.R (`:601-658`)

- **Weighted base fit** (`:605`) + `predict_proba` (`:607`). Same headline
  resample divergence (REQ-6).
- **estimator_error** (`:619`): from `incorrect = argmax(proba) != y`. The
  ferrolearn SAMME.R path does **not** compute or use an `estimator_error`.
- **y_coding + xlogy reweight** (`:634-649`): `y_codes = [-1/(K-1), 1]`;
  `estimator_weight = -lr * (K-1)/K * sum_k xlogy(y_coding_k, proba_k)` (i.e.
  `-lr*(K-1)/K * sum_k y_coding_k * log(proba_k)`); then (not last iter)
  `sample_weight *= exp(estimator_weight)` for positive weights (`:652-656`).
  ferrolearn `fit_samme_r` instead reweights `weights[i] *= exp(-factor *
  log(p_{y_i}))` with `factor = lr*(K-1)/K` — i.e. it uses **only the
  true-class term** `-(K-1)/K*lr*log(p_correct)`, dropping the
  `+ (K-1)/K*lr * (1/(K-1)) * sum_{k != y} log(p_k)` contribution from the
  `-1/(K-1)` codes (`xlogy(-1/(K-1), proba_k)` over the wrong classes). The two
  agree only when `K == 2` (the wrong-class sum is a single term and the coding
  collapses) or when the off-class probabilities are degenerate. For `K > 2`
  the per-round reweight diverges from sklearn. **REQ-8, blocker #713.**

### predict / decision_function / predict_proba (`:710-917`)

sklearn `decision_function` (`:768-813`):
- **SAMME** (`:799-807`): `pred = sum_t np.where(predict_t == class, w_t,
  -w_t/(K-1))`, then `pred /= sum(weights)` (`:809`); binary (`:810-812`)
  collapses to `pred[:,1] - pred[:,0]` (after negating col 0). So each stump
  contributes `+w` to its predicted class **and `-w/(K-1)` to every other
  class**, and the whole thing is **normalized by `sum(weights)`**.
- **SAMME.R** (`:796-798`): `pred = sum_t _samme_proba(est_t, K, X)` where
  `_samme_proba = (K-1)*(log p - mean_k log p_k)` (`:318-336`), `/sum(weights)`
  (all weights 1).

`predict` (`:710-732`): binary → `classes_.take(pred > 0)`; multiclass →
`classes_.take(argmax(pred))`. `predict_proba` (`:891-917`):
`softmax(decision/(K-1))` for multiclass, `softmax([-d, d]/2)` for binary
(`_compute_proba_from_decision`, `:872-889`).

ferrolearn `decision_function`:
- **SAMME**: adds only `estimator_weights[t]` to the predicted class
  (`out[[i, class_idx]] += w_t`); **no `-w/(K-1)` term, no `/sum(weights)`, no
  binary collapse** — returns a `(n_samples, K)` non-negative vote matrix. The
  **argmax agrees** with sklearn's for predict (the `-w/(K-1)` term is a uniform
  per-class offset that does not change the argmax, and normalization is
  monotone), so `predict` argmax-parity holds for SAMME multiclass; but the
  **decision_function values themselves diverge** (sign convention, magnitude,
  shape — sklearn returns `(n,)` for binary). **REQ-5, blocker #714.**
- **SAMME.R**: accumulates `(K-1)*(log p_k - mean log p)` per stump
  (`_samme_proba` form) — **matches sklearn's summand** (`:334-336`), modulo the
  `/sum(weights)` (a positive monotone scale that does not affect argmax). So
  the SAMME.R decision summand is correct in form (subject to REQ-6 producing
  the right stumps).

ferrolearn `predict`: argmax of the vote/accumulator. ferrolearn `predict_proba`
SAMME normalizes the vote vector to sum 1 (NOT sklearn's
`softmax(decision/(K-1))`); SAMME.R applies a plain `softmax(accumulated)` (NOT
`softmax(accumulated/(K-1))`). **predict_proba values diverge from sklearn —
REQ-5, blocker #714.**

### feature_importances_ (`:277-308`)

sklearn property: `sum(weight * clf.feature_importances_) /
estimator_weights_.sum()` (`:300-308`). ferrolearn:
`aggregate_tree_importances(&estimators, None, Some(&estimator_weights),
n_features)` (in `decision_tree.rs`) — the weighted sum of per-stump importances
normalized to sum 1, exposed via `HasFeatureImportances`. Match in form (REQ-10);
the per-stump importance is inherited from oracle-verified `decision_tree.rs`.
For SAMME.R, ferrolearn weights are all `1.0`, matching sklearn's all-ones
SAMME.R weights.

## ferrolearn (what exists)

- **Unfitted**: `pub struct AdaBoostClassifier<F>` (public fields
  `n_estimators`, `learning_rate: f64`, `algorithm`, `random_state`); `fn new`;
  `with_*` builders (`with_n_estimators`, `with_learning_rate`,
  `with_algorithm`, `with_random_state`); `Default`.
- **Enum**: `pub enum AdaBoostAlgorithm { Samme, SammeR }`.
- **Fitted**: `pub struct FittedAdaBoostClassifier<F>` (`classes: Vec<usize>`,
  `estimators: Vec<Vec<Node<F>>>`, `estimator_weights: Vec<F>`, `n_features`,
  `n_classes`, `algorithm`, `feature_importances: Array1<F>`).
- **Traits**: `Fit<Array2<F>, Array1<usize>>`; `Predict<Array2<F>>`;
  `HasClasses`; `HasFeatureImportances<F>`; `PipelineEstimator<F>` /
  `FittedPipelineEstimator<F>`.
- **Methods**: `fn score` (mean accuracy), `fn predict_proba`,
  `fn predict_log_proba`, `fn decision_function`; private `fn predict_samme`,
  `fn predict_samme_r`, `fn fit_samme`, `fn fit_samme_r`.
- **Internal helper**: `fn resample_weighted` (deterministic systematic
  resample, no RNG).
- **Build delegation** (from `decision_tree.rs`, oracle-verified):
  `build_classification_tree_with_feature_subset` (currently `sample_weight:
  None`, `decision_tree.rs:2448`), `traverse`, `aggregate_tree_importances`,
  `Node<F>`.
- **Consumers (non-test)**: crate re-export (`ferrolearn-tree/src/lib.rs`:
  `pub use adaboost::{AdaBoostAlgorithm, AdaBoostClassifier,
  FittedAdaBoostClassifier}`); the **PyO3 binding** `RsAdaBoostClassifier`
  (`ferrolearn-python/src/extras.rs:616`) — constructs
  `AdaBoostClassifier::<f64>::new().with_n_estimators(..).with_learning_rate(..)`
  and calls `fit`/`predict`; the pipeline adapters
  (`PipelineEstimator::fit_pipeline` / `FittedPipelineEstimator::predict_pipeline`).
  These boundary estimator types are existing pub APIs, grandfathered per
  S5/R-DEFER-1; the PyO3 shim is the R-DEFER-1 non-test production consumer.

## Requirements

- REQ-1: **Param surface + numeric defaults (R-DEV-2).** `n_estimators=50`,
  `learning_rate=1.0`, base stump `max_depth=1`, `random_state=None` match
  `AdaBoostClassifier().get_params()` / `_validate_estimator` (`:507`, `:521`);
  `n_estimators>=1`, `learning_rate>0`, `n_classes>=2`, shape/empty validated
  (`:70`, `:493-498`). `learning_rate` stored as `f64` not `F` (noted; benign).
- REQ-1b: **Literal 1.5.2 `algorithm` default (R-DEV-6 deviation).** sklearn
  1.5.2 default is `'SAMME.R'` (`:507`); ferrolearn defaults to `Samme`. This is
  a DELIBERATE deviation tracking sklearn's own 1.4 deprecation / 1.6 removal of
  SAMME.R (`:526-534`). Classified NOT-STARTED against the literal-match
  criterion, documented as intentional.
- REQ-2: **SAMME estimator_weight + reweight formulas (R-DEV-1).**
  `alpha = lr*(log((1-err)/err) + log(K-1))` (`:696-698`) and `w_i *=
  exp(alpha)` for misclassified positive-weight samples (`:701-706`).
  Deterministic given the per-round error.
- REQ-3: **SAMME weighted error (R-DEV-1).** `err = sum(w_i*[pred_i!=y_i]) /
  sum(w)` equals `mean(average(incorrect, weights=sample_weight))` (`:676`).
- REQ-4: **Error threshold / stop conditions (R-DEV-1).** Worse-than-random
  stop at `err >= 1 - 1/K` (`:685-693`); ferrolearn keeps the first stump with
  weight 1.0 and breaks (vs sklearn raising on the empty ensemble — benign
  robustness choice).
- REQ-5: **predict / decision_function / predict_proba contract (R-DEV-1/3).**
  SAMME `decision_function = sum_t where(correct, w, -w/(K-1)) / sum(w)`, binary
  collapse to `(n,)` (`:799-813`); `predict_proba = softmax(decision/(K-1))`
  (`:872-917`). ferrolearn drops the `-w/(K-1)` term and the `/sum(w)`
  normalization, returns `(n,K)` for binary, and `predict_proba` normalizes the
  raw vote (SAMME) / plain-softmax (SAMME.R, no `/(K-1)`). `predict` argmax
  agrees for SAMME (uniform offsets / monotone scaling), but
  `decision_function`/`predict_proba` **values** diverge.
- REQ-6: **Weighted base-estimator fit (R-DEV-1, HEADLINE).** sklearn fits each
  stump on the weighted data (`estimator.fit(X, y, sample_weight=sample_weight)`,
  `_boost_discrete:664` / `_boost_real:605`) deterministically with no RNG;
  ferrolearn fits an unweighted stump on a deterministic systematic resample
  (`resample_weighted`, builder `sample_weight: None`). Prediction-affecting.
  2-file change (weighted builder in `decision_tree.rs` + the
  `fit_samme`/`fit_samme_r` call sites).
- REQ-7: **Perfect-fit `estimator_weight = 1.0` guard (R-DEV-1).** sklearn
  returns `1.0`/stops when `estimator_error <= 0` (`:679-680`, `:622-623`);
  ferrolearn has no guard and computes `alpha ≈ 23*lr + lr*log(K-1)`,
  continuing to boost. Single-file fixer target inside `fit_samme`.
- REQ-8: **SAMME.R `_boost_real` reweight + `_samme_proba` (R-DEV-1).** sklearn
  reweights via the full `y_coding`/`xlogy` term `exp(-lr*(K-1)/K * sum_k
  y_coding_k log p_k)` (`:634-656`); ferrolearn uses only the true-class term
  `exp(-lr*(K-1)/K * log p_correct)`, dropping the `-1/(K-1)` off-class
  contributions (correct only for `K == 2`). The `_samme_proba` *predict*
  summand `(K-1)*(log p - mean log p)` matches in form.
- REQ-9: **Pluggable base `estimator` + `estimator_errors_` (R-DEV-2/3).**
  sklearn boosts any `predict_proba`/`predict` learner and exposes
  `estimator_errors_`; ferrolearn hardwires a stump and stores no per-round
  error vector.
- REQ-10: **`feature_importances_` = normalized weighted sum (R-DEV-3).**
  `sum(weight*clf.fi)/sum(weight)` (`:300-308`); per-stump importance inherited
  from oracle-verified `decision_tree.rs`.
- REQ-11: **End-to-end predict parity vs live `AdaBoostClassifier(
  algorithm='SAMME')` (R-DEV-1).** Achievable once REQ-6 (weighted fit) and
  REQ-7 (perfect-fit guard) land, because sklearn's SAMME path is fully
  deterministic (no `random_state.choice`). Blocked on REQ-6/REQ-7.
- REQ-12: **PyO3 binding fidelity (R-DEFER-1).** `RsAdaBoostClassifier`
  (`extras.rs:616`) exposes only `n_estimators`/`learning_rate`/`random_state`;
  no `algorithm` arg, no `predict_proba`/`decision_function`, no `estimator`.
- REQ-13: **ferray substrate (R-SUBSTRATE).** Imports `ndarray`/`num-traits`,
  not `ferray-core`/`ferray::random`.

## Acceptance criteria

- AC-1: live `AdaBoostClassifier().get_params()` equals the REQ-1 defaults for
  the params ferrolearn exposes (`n_estimators=50`, `learning_rate=1.0`,
  `random_state=None`); base stump `max_depth=1`.
- AC-1b (REQ-1b): `AdaBoostClassifier().get_params()['algorithm'] == 'SAMME.R'`
  while ferrolearn `Default::default().algorithm == Samme` — the documented
  R-DEV-6 deviation; covered by `test_adaboost_default_trait`.
- AC-2 (REQ-2, R-CHAR-3): on a fixed `(err, K, lr)`, `alpha = lr*(log((1-err)/
  err) + log(K-1))` matches the closed-form sklearn value to 1e-12 (e.g.
  `err=0.25, K=3, lr=1.0 → log(3) + log(2) = 1.7917...`); the reweight
  multiplies a misclassified sample's weight by `exp(alpha)`.
- AC-3 (REQ-3): `err = sum(w_i*[pred!=y])/sum(w)` equals numpy
  `np.mean(np.average(incorrect, weights=w))` on a fixed `(incorrect, w)`.
- AC-4 (REQ-4): `err >= 1 - 1/K` terminates boosting (no further stumps added).
- AC-5 (REQ-5 pin, R-CHAR-3): with a fixed fitted ensemble, ferrolearn's
  `decision_function` row must equal sklearn's `sum_t where(correct, w,
  -w/(K-1))/sum(w)` (and binary collapse to `(n,)`); ferrolearn currently
  diverges → FAILS until the predict-family is corrected.
- AC-6 (REQ-7 pin, R-CHAR-3): a dataset a single stump fits perfectly
  (`err==0`) must yield `estimator_weight == 1.0` and stop boosting (matches
  `_boost_discrete:679-680` + `fit:180`); ferrolearn currently produces
  `alpha ≈ 23*lr` → FAILS until the guard lands.
- AC-7 (REQ-8 pin, K>2, R-CHAR-3): the post-round SAMME.R `sample_weight` must
  equal the numpy `exp(-lr*(K-1)/K * sum_k y_coding_k log p_k)` reference;
  ferrolearn's true-class-only update diverges for `K>2` → FAILS.
- AC-8 (REQ-11 pin, R-CHAR-3): `predict` on a fixed `(X, y, random_state)` must
  equal live `AdaBoostClassifier(algorithm='SAMME', n_estimators=N,
  learning_rate=lr).fit(X,y).predict(X)` — FAILS until REQ-6 + REQ-7 land.
- AC-9 (REQ-10): `feature_importances_` sums to 1 when any stump splits and
  equals the normalized weighted sum of per-stump importances.
- AC-10: `random_state` reproducibility — two `fit` calls with identical params
  produce identical `predict` (covered by `test_adaboost_reproducibility`;
  trivially holds since fit uses no RNG).

## REQ status table

Binary (R-DEFER-2). `AdaBoostClassifier`/`FittedAdaBoostClassifier`/
`AdaBoostAlgorithm` are boundary estimator types re-exported at the crate root,
exposed through the `RsAdaBoostClassifier` PyO3 binding, and wired into the
pipeline adapters (the non-test production-consumer surface; existing pub APIs,
grandfathered per S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) /
`file:line` (sklearn 1.5.2). Verification green at baseline `7d9078c1`
(`cargo test -p ferrolearn-tree --lib adaboost::`: 26 passed, 0 failed).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (param surface + numeric defaults) | SHIPPED (gaps flagged) | `fn new` on `AdaBoostClassifier` (`n_estimators=50`, `learning_rate=1.0`, base stump `max_depth=Some(1)` via `stump_params` in `fit_samme`/`fit_samme_r`, `random_state=None`) + `fit` validation (`n_estimators==0`/`learning_rate<=0`/`n_classes<2` → `InvalidParameter`; `ShapeMismatch`/`InsufficientSamples`) match `AdaBoostClassifier.__init__` (`_weight_boosting.py:501-517`), `_validate_estimator` default `DecisionTreeClassifier(max_depth=1)` (`:521`), `_parameter_constraints` (`:70`, `:493-498`). Consumer: crate re-export (`lib.rs`) + `RsAdaBoostClassifier` (`extras.rs:616`) + pipeline adapter. Tests: `test_adaboost_default_trait`, `test_adaboost_zero_estimators`, `test_adaboost_invalid_learning_rate`, `test_adaboost_negative_learning_rate`, `test_adaboost_single_class`, `test_adaboost_shape_mismatch_fit`, `test_adaboost_empty_data`. Verification: `python3 -c "from sklearn.ensemble import AdaBoostClassifier; print(AdaBoostClassifier().get_params())"` → `learning_rate=1.0, n_estimators=50, random_state=None`. Note: `learning_rate` typed `f64` not `F` (benign). Surface gaps (pluggable `estimator`, `estimator_errors_`) → REQ-9. |
| REQ-1b (literal 1.5.2 algorithm default) | NOT-STARTED | open prereq blocker #712 (DELIBERATE R-DEV-6 deviation, NOT a fixer target). sklearn 1.5.2 default `algorithm='SAMME.R'` (`:507`, live `get_params()`); ferrolearn `Default::default().algorithm == AdaBoostAlgorithm::Samme` (`fn new`). sklearn deprecated SAMME.R in 1.4 (`FutureWarning`, `_validate_estimator:526-534`) and removed it in 1.6 — ferrolearn defaults to the surviving algorithm. Documented as intentional; the blocker records the rationale, not a fix obligation. |
| REQ-2 (SAMME estimator_weight + reweight) | SHIPPED | `let alpha = lr*((1-err).max(eps)/err.max(eps)).ln() + lr*(K-1).ln()` in `fit_samme` mirrors `_boost_discrete:696-698` (`learning_rate*(log((1-err)/err)+log(n_classes-1))`); the reweight `if preds[i]!=y_mapped[i] { weights[i] *= alpha.exp() }` mirrors `:701-706` (`exp(log(w)+alpha*incorrect*(w>0))`). Deterministic given the per-round error. Consumer: `estimator_weights` drives `predict_samme`/`decision_function`; `RsAdaBoostClassifier.predict`. Tests: `test_adaboost_samme_binary_simple`, `test_adaboost_samme_multiclass`, `test_adaboost_samme_learning_rate_effect`. |
| REQ-3 (SAMME weighted error) | SHIPPED | `fit_samme` accumulates `weighted_error += weights[i]` over misclassified samples, then `err = weighted_error / weight_sum`, equal to sklearn `np.mean(np.average(incorrect, weights=sample_weight))` (`_boost_discrete:676`). Consumer: feeds `alpha` (REQ-2) and the stop test (REQ-4). Tests: `test_adaboost_samme_binary_simple` (perfect-separation err drives correct stumps). |
| REQ-4 (error threshold / stop conditions) | SHIPPED | `if err >= F::one() - F::one()/F::from(n_classes) { ... break }` in `fit_samme` mirrors `_boost_discrete:685` (`estimator_error >= 1 - 1/n_classes`); ferrolearn keeps the first stump (weight 1.0) and breaks rather than raising on the empty ensemble (`:687-692`) — benign robustness choice, boosting still terminates. Consumer: bounds the `estimators` vector consumed by `predict`. Tests: `test_adaboost_samme_multiclass`, `test_adaboost_4_classes`. |
| REQ-5 (predict / decision_function / predict_proba contract) | NOT-STARTED | open prereq blocker #714. `decision_function` (SAMME) adds only `out[[i,class_idx]] += estimator_weights[t]` — MISSING the `-w/(K-1)` off-class term and the `/sum(weights)` normalization sklearn applies (`:799-809`), and returns `(n_samples,K)` for binary instead of collapsing to `(n,)` (`:810-812`). `predict_proba` (SAMME) normalizes the raw vote instead of `softmax(decision/(K-1))` (`:872-917`); (SAMME.R) applies plain `softmax(accumulated)` without the `/(K-1)` scale. `predict` argmax agrees for SAMME (uniform offsets / monotone scaling don't move the argmax), but `decision_function`/`predict_proba` VALUES diverge from sklearn. |
| REQ-6 (weighted base-estimator fit, HEADLINE) | SHIPPED | `fit_samme`/`fit_samme_r` now call `build_weighted_classification_tree_with_feature_subset` (new `pub(crate)` fn in `decision_tree.rs`) passing the live round `&weights` over ALL samples, threading `ClassificationData.sample_weight: Some(sample_weight)`, `min_weight_leaf: F::zero()`, and `indices = 0..n_samples` (no resampling) — mirroring sklearn's deterministic weighted fit `estimator.fit(X, y, sample_weight=sample_weight)` (`_boost_discrete:664`, `_boost_real:605`), NO RNG. The dead systematic-resample helper `resample_weighted` is removed. The weighted node-count/gini/leaf-distribution path is the oracle-verified `class_weight` path (decision-tree REQ-7). Consumer: `fit_samme`/`fit_samme_r` (non-test), reached via crate re-export + `RsAdaBoostClassifier` (`extras.rs`). Test: `divergence_weighted_fit_round2_predict` (n_estimators=2 SAMME `predict` == live sklearn `[0,0,0,0,0,0,1,1]`) now GREEN. Closes #709. |
| REQ-7 (perfect-fit estimator_weight=1.0 guard) | SHIPPED | `fit_samme` now guards `if err <= eps { estimators.push(tree); estimator_weights.push(F::one()); break; }` BEFORE the worse-than-random check, mirroring sklearn's `if estimator_error <= 0: return sample_weight, 1.0, 0.0` (`_boost_discrete:679-680`) + `fit` stop on `error == 0` (`fit:180`). `fit_samme_r` adds the matching guard: it computes the weighted `argmax(proba) != y` error (`_boost_real:616-619`) and on `err <= eps` pushes the stump (weight `1.0`, the SAMME.R constant) and breaks (`_boost_real:622-623`). Consumer: `fit_samme`/`fit_samme_r` (non-test) via crate re-export + `RsAdaBoostClassifier`. Test: `divergence_perfect_fit_estimator_weight_one` (single perfect stump weight == 1.0, was ≈23.0258) now GREEN. Closes #710. |
| REQ-8 (SAMME.R _boost_real reweight + _samme_proba) | NOT-STARTED | open prereq blocker #713. `fit_samme_r` reweights `weights[i] *= exp(-factor * p_correct.ln())` with `factor = lr*(K-1)/K` — the true-class term ONLY. sklearn's `_boost_real:634-649` uses the full `y_coding=[-1/(K-1),1]` / `xlogy` weight `exp(-lr*(K-1)/K * sum_k y_coding_k * log p_k)`, including the `-1/(K-1)` off-class contributions (`:644-656`). Equal only for `K==2`; diverges for `K>2`. The `_samme_proba` PREDICT summand `(K-1)*(log p_k - mean_k log p_k)` (`predict_samme_r`/`decision_function`) matches the sklearn form (`:334-336`) but is fed by resample-built stumps (REQ-6) and lacks the `/sum(weights)` scale (immaterial to argmax). |
| REQ-9 (pluggable estimator + estimator_errors_) | NOT-STARTED | open prereq blocker #715. ferrolearn hardwires a stump (`build_classification_tree_with_feature_subset`) instead of sklearn's generic `estimator` (`_validate_estimator:521`), and stores no `estimator_errors_` vector (sklearn `:154`/`:177`). |
| REQ-10 (feature_importances_ normalized weighted sum) | SHIPPED | `fit_samme`/`fit_samme_r` call `aggregate_tree_importances(&estimators, None, Some(&estimator_weights), n_features)` (`decision_tree.rs`, oracle-verified) — weighted sum of per-stump importances normalized to sum 1, mirroring `feature_importances_` `:300-308` (`sum(weight*clf.fi)/norm`); exposed via `HasFeatureImportances::feature_importances`. SAMME.R weights are all `1.0` matching sklearn's all-ones SAMME.R weights. Consumer: `HasFeatureImportances` impl + pipeline. Tests: per-stump importance correctness inherited from `decision_tree.rs` (`.design/tree/decision_tree.md`); aggregation exercised by `test_adaboost_many_features`. |
| REQ-11 (end-to-end predict parity vs SAMME) | NOT-STARTED | open prereq blocker #716 (depends on #710 + #711). sklearn's SAMME path is fully deterministic (no `random_state.choice` — the RNG only seeds the base estimator, irrelevant to a deterministic stump build), so once ferrolearn fits WEIGHTED stumps (REQ-6) and adds the perfect-fit guard (REQ-7), end-to-end `predict` parity with live `AdaBoostClassifier(algorithm='SAMME')` is ACHIEVABLE (unlike the regressor's stochastic-bootstrap boundary). Currently the resample-built stumps make per-round trees diverge → no parity. |
| REQ-12 (PyO3 binding fidelity) | NOT-STARTED | open prereq blocker #717. `RsAdaBoostClassifier` (`extras.rs:616`, signature `(n_estimators=50, learning_rate=1.0, random_state=None)`) exposes no `algorithm` parameter, no `predict_proba`/`predict_log_proba`/`decision_function`, no pluggable `estimator`; `import ferrolearn` cannot reach the SAMME.R variant or the probability/decision surface sklearn exposes. |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #718. `adaboost.rs` imports `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}`, not `ferray-core`/`ferray::random` (R-SUBSTRATE). |

## Architecture

`AdaBoostClassifier<F>` is the unfitted boundary type (public fields
`n_estimators`/`learning_rate: f64`/`algorithm`/`random_state` + `with_*`
builders + `Default`). `fit` validates shapes/params, derives the sorted unique
`classes` and a `y_mapped` (label → index), and dispatches on `algorithm` to
`fit_samme` or `fit_samme_r`.

`fit_samme` runs the SAMME loop: init uniform `weights = 1/n_samples`; each round
(1) `resample_weighted` produces deterministic systematic-resample indices, (2)
`build_classification_tree_with_feature_subset` fits an **unweighted** stump on
those indices (REQ-6 divergence — sklearn fits weighted), (3) compute
per-sample predictions + the normalized weighted error `err`, (4) the
worse-than-random stop at `err >= 1 - 1/K` (REQ-4), (5) `alpha = lr*(log((1-err)/
err)+log(K-1))` (REQ-2; **no perfect-fit `err==0 → 1.0` guard**, REQ-7), (6)
reweight misclassified positive-weight samples by `exp(alpha)` (REQ-2) and
normalize. `fit_samme_r` is analogous but uses leaf class distributions, an
equal `1.0` estimator weight per stump, and a true-class-only log-prob reweight
(REQ-8 divergence). After the loop, `aggregate_tree_importances` builds the
weighted-normalized `feature_importances` (REQ-10).

`FittedAdaBoostClassifier<F>` stores `classes`, `estimators: Vec<Vec<Node<F>>>`,
`estimator_weights`, `n_features`, `n_classes`, `algorithm`,
`feature_importances`. `predict` traverses each stump (`decision_tree::traverse`)
and either votes by `estimator_weight` per class (SAMME) or accumulates
`(K-1)*(log p - mean log p)` (SAMME.R), then argmaxes — argmax-correct for SAMME
but with a `decision_function`/`predict_proba` value divergence (REQ-5).
`score` is mean accuracy via the crate `mean_accuracy`. Pipeline integration is
provided by `PipelineEstimator`/`FittedPipelineEstimator`.

**Invariants held:** uniform weight init; SAMME alpha/reweight formula;
worse-than-random stop; weight renormalization; feature-count guard on predict;
all-ones SAMME.R weights into `feature_importances`. **Invariants NOT held vs
sklearn:** (a) the base stump is resample-built, not weighted (REQ-6 — the
headline, prediction-affecting, 2-file fix; end-to-end parity achievable once
fixed because the SAMME path is deterministic); (b) no perfect-fit `1.0` guard
(REQ-7 — single-file); (c) `decision_function`/`predict_proba` formulas omit the
`-w/(K-1)` term, the `/sum(weights)` normalization, the binary collapse, and the
`/(K-1)` softmax scaling (REQ-5); (d) SAMME.R reweight drops the off-class
`y_coding` terms (REQ-8, wrong for `K>2`).

## Verification

Library crate (green at baseline `7d9078c1`):
```
cargo test -p ferrolearn-tree --lib adaboost::      # 26 passed; 0 failed
cargo clippy -p ferrolearn-tree --all-targets -- -D warnings
cargo fmt --all --check
```
Live sklearn oracle (installed 1.5.2, run from `/tmp`):
```
# REQ-1 / REQ-1b defaults (algorithm='SAMME.R' is the R-DEV-6 deviation):
python3 -c "from sklearn.ensemble import AdaBoostClassifier; print(AdaBoostClassifier().get_params())"
# REQ-2 SAMME alpha closed form (err=0.25, K=3, lr=1.0):
python3 -c "import numpy as np; err=0.25; K=3; print(np.log((1-err)/err)+np.log(K-1))"   # 1.7917594...
# REQ-7 perfect-fit guard: sklearn returns estimator_weight 1.0 on err==0
#   (_boost_discrete:679-680); ferrolearn computes ~23*lr+lr*log(K-1).
# REQ-8 SAMME.R reweight (K=3): sklearn full y_coding/xlogy vs ferrolearn true-class only:
python3 -c "import numpy as np; from scipy.special import xlogy; K=3; lr=1.0; p=np.array([0.2,0.3,0.5]); yc=np.array([-1/(K-1),1.0]); coding=yc.take((np.arange(K)==2).astype(int)); print('sklearn', -lr*(K-1)/K*xlogy(coding,p).sum()); print('ferro  ', -lr*(K-1)/K*np.log(p[2]))"
# REQ-11 end-to-end SAMME parity (deterministic — achievable after #710/#711):
python3 -c "import numpy as np; from sklearn.ensemble import AdaBoostClassifier; from sklearn.datasets import load_iris; X,y=load_iris(return_X_y=True); print(AdaBoostClassifier(algorithm='SAMME',n_estimators=10).fit(X,y).predict(X)[:10])"
```
The NOT-STARTED REQs (1b, 5, 6, 7, 8, 9, 11, 12, 13) have no green verification
by construction — each carries an open prereq blocker. REQ-1/2/3/4/10 are
verified by the in-crate `#[test]`s named in the status table plus the live
`get_params()` / closed-form-alpha comparisons (deterministic). Characterization
pins for REQ-5/REQ-7/REQ-8/REQ-11 (R-CHAR-3, AC-5/AC-6/AC-7/AC-8) belong in
`ferrolearn-tree/tests/divergence_adaboost.rs`: assert `decision_function`
matches the sklearn `where(correct,w,-w/(K-1))/sum(w)` form; assert a perfectly
fittable stump yields `estimator_weight == 1.0` and stops; assert the SAMME.R
post-round `sample_weight` matches the numpy `xlogy` reference for `K>2`; assert
`predict` equals live `AdaBoostClassifier(algorithm='SAMME')` — each FAILS until
the respective fix/builder lands.

## Blockers to open

- #710 — REQ-6 (HEADLINE, R-DEV-1, NOT a boundary): base stump is resample-built
  unweighted (`fit_samme`/`fit_samme_r` → `build_classification_tree_with_feature_subset`
  with `resample_weighted` and `sample_weight: None`, `decision_tree.rs:2448`)
  instead of sklearn's deterministic WEIGHTED fit `estimator.fit(X, y,
  sample_weight=sample_weight)` (`_weight_boosting.py:664`/`:605`). 2-file change:
  add a weighted classification builder in `decision_tree.rs` (the
  `ClassificationData.sample_weight` path is oracle-verified for `class_weight`)
  + switch the two `fit_*` call sites. Prediction-affecting; gates REQ-11.
- #711 — REQ-7 (single-file fixer): no perfect-fit `estimator_error <= 0 →
  estimator_weight 1.0, stop` guard in `fit_samme` (sklearn `_boost_discrete:679-680`,
  `fit:180`); ferrolearn computes `alpha ≈ 23*lr + lr*log(K-1)` and keeps
  boosting.
- #712 — REQ-1b (DELIBERATE R-DEV-6 deviation, NOT a fix obligation): ferrolearn
  default `algorithm = Samme` vs sklearn 1.5.2 `'SAMME.R'` (`:507`). Records the
  rationale (sklearn deprecated SAMME.R in 1.4 `FutureWarning` `:526-534`, removed
  it in 1.6). Closed by documentation, not a code change.
- #713 — REQ-8: SAMME.R reweight in `fit_samme_r` uses only the true-class term
  `exp(-lr*(K-1)/K * log p_correct)` instead of sklearn's full `y_coding`/`xlogy`
  weight `exp(-lr*(K-1)/K * sum_k y_coding_k log p_k)` (`_boost_real:634-656`).
  Diverges for `K>2`.
- #714 — REQ-5: `decision_function`/`predict_proba` omit the SAMME `-w/(K-1)`
  off-class term, the `/sum(weights)` normalization, the binary `(n,)` collapse
  (`:799-813`), and the `softmax(decision/(K-1))` proba transform (`:872-917`).
  predict argmax (SAMME) still agrees; values diverge.
- #715 — REQ-9: no pluggable base `estimator` (stump hardwired, `:521`) and no
  `estimator_errors_` fitted attribute (sklearn `:154`/`:177`).
- #716 — REQ-11: end-to-end SAMME predict parity — achievable but blocked on
  #710 (weighted fit) + #711 (perfect-fit guard); sklearn's SAMME path is
  deterministic.
- #717 — REQ-12: `RsAdaBoostClassifier` (`extras.rs:616`) exposes no `algorithm`
  arg, no `predict_proba`/`decision_function`, no pluggable `estimator`.
- #718 — REQ-13: migrate `adaboost.rs` off `ndarray`/`num-traits` to the ferray
  substrate (R-SUBSTRATE).
