# ClassifierChain / RegressorChain / OutputCodeClassifier (chained + output-code meta-estimators)

<!--
tier: 3-component
status: draft
baseline-commit: fb0860d86ed1bdda80368926c092e96f708c2404
upstream-paths:
  - sklearn/multioutput.py            # _BaseChain (:625), fit (:700-814), _get_predictions/un-permute (:659-697), predict (:816-829); ClassifierChain (:832), __init__ (:983-990), chain_method (:773-781), predict_proba (:1035-1048), classes_; RegressorChain (:1106)
  - sklearn/multiclass.py             # OutputCodeClassifier (:1025), __init__ (:1135), code_size constraint (:1130), fit (:1170-1220) n_estimators=int(n_classes*code_size) (:1189) / code_book_ (:1193-1199), predict (:1222-1245)
-->

## Summary

`ferrolearn-model-sel/src/chain.rs` mirrors three scikit-learn meta-estimators:
the chained multi-output estimators `ClassifierChain` / `RegressorChain` built on
the `_BaseChain` base (`sklearn/multioutput.py:625`), and the error-correcting
output-code multi-class classifier `OutputCodeClassifier`
(`sklearn/multiclass.py:1025`). The chain strategy fits one estimator per target
in chain order, augmenting each link's features with the PRIOR targets, and
un-permutes `predict` back to original column order. The output-code strategy
reduces a `K`-class problem to one binary problem per code-matrix COLUMN and
predicts by nearest code row (euclidean argmin). The base estimator is expressed
as a `PipelineFactory` CLOSURE (`Box<dyn Fn() -> Pipeline<f64> + Send + Sync>`)
rather than a wrapped/`clone`d estimator object — the sanctioned R-DEV-7 Rust
idiom shared with `multiclass.rs`/`multioutput.rs`/`grid_search.rs`, NOT a bug.

ferrolearn ships the **chain fit/predict core end-to-end** for the default
`cv=None` case (true-Y-during-fit, predicted-during-predict, un-permute to
original column order — verified END-TO-END below against the live oracle,
including a non-identity explicit `order`) and the OutputCodeClassifier
fit/predict MECHANIC. The unit has ONE **deterministic FIXABLE divergence** the
critic should pin: the OutputCodeClassifier `n_codes` formula
(REQ-OCC-NCODES, #1830) — ferrolearn `ceil(code_size*k).max(2)` vs sklearn
`int(code_size*k)` (FLOOR, no `max(2)`), giving `5` vs `4` at `code_size=1.5,
k=3`. The OutputCodeClassifier code VALUES + raw-vs-thresholded predict
(REQ-OCC-PREDICT, #1831) and exact code book (REQ-OCC-RNG, #1836) are
architectural / R-DEFER-3 RNG carve-outs. The remaining gaps are missing-feature
blockers: `cv` leakage-free chain features (#1832), `order='random'` (#1833),
`chain_method` proba/decision cascade (#1834), `predict_proba`/`classes_`
(#1835), `sample_weight`/`n_jobs`/`fit_params` (#1837), and the `ferray`
substrate (#1838).

## Upstream reference (scikit-learn 1.5.2, tag 1.5.2)

### `_BaseChain` (`sklearn/multioutput.py:625`)
- `:700-814` — `fit(self, X, Y, **fit_params)`. `:723-734` resolve `order_`:
  default `np.array(range(Y.shape[1]))` (`:728-729`); `order='random'` ⇒
  `random_state.permutation(Y.shape[1])` (`:730-732`); a list must be a
  permutation of `range(n_targets)` else `ValueError("invalid order")` (`:733`).
- `:736` — `self.estimators_ = [clone(self.base_estimator) for _ in
  range(Y.shape[1])]` — one fresh clone per target.
- `:738-744` (cv=None, default) — `Y_pred_chain = Y[:, self.order_]`;
  `X_aug = np.hstack((X, Y_pred_chain))` — the augmented feature matrix is `X`
  followed by the TRUE targets in chain order.
- `:783-795` — for each `chain_idx`, fit estimator `order_[chain_idx]` on
  `X_aug[:, : (X.shape[1] + chain_idx)]` (i.e. `X` + the first `chain_idx`
  prior targets) against `y = Y[:, self.order_[chain_idx]]` (`:789`). With
  `cv=None` the prior-target features are the TRUE `Y` columns.
- `:797-812` (cv set) — after fitting link `chain_idx`, the prior-target FEATURE
  column for later links is REPLACED by `cross_val_predict(...)` (leakage-free)
  rather than the true `Y` (`:799-812`). `cv=None` skips this entirely.
- `:773-781` — `chain_method` (ClassifierChain only) selects the response method
  (`predict`/`predict_proba`/`decision_function`) used for the CASCADE features;
  `RegressorChain` defaults to `"predict"`.
- `:659-697` — `_get_predictions(X, output_method)` (the `predict` body):
  `Y_output_chain`/`Y_feature_chain` zeros `(n, K)`; for each `chain_idx`,
  `X_aug = hstack((X, Y_feature_chain[:, :chain_idx]))` — features are the link's
  OWN PRIOR PREDICTIONS (`:671`, `:677`), `Y_feature_chain[:, chain_idx] =
  feature_predictions` (`:684`), `Y_output_chain[:, chain_idx] =
  output_predictions` (`:691`). UN-PERMUTE: `inv_order[self.order_] =
  arange(len(order_))`; `Y_output = Y_output_chain[:, inv_order]` (`:693-695`)
  ⇒ output is in ORIGINAL column order. `predict` (`:816-829`) calls it with
  `output_method="predict"`.

### `ClassifierChain` (`sklearn/multioutput.py:832`)
- `:983-990` — `__init__(self, base_estimator, *, order=None, cv=None,
  chain_method="predict", random_state=None, verbose=False)`.
- `:1035-1048` — `predict_proba` = `_get_predictions(X,
  output_method="predict_proba")`; `predict_log_proba` (`:1050ish`); `classes_`
  from the per-link estimators.

### `RegressorChain` (`sklearn/multioutput.py:1106`)
- `__init__(self, base_estimator, *, order=None, cv=None, random_state=None,
  verbose=False)` — NO `chain_method`.

### `OutputCodeClassifier` (`sklearn/multiclass.py:1025`)
- `:1130` — `code_size` parameter constraint
  `Interval(Real, 0.0, None, closed="neither")` ⇒ `code_size <= 0` raises an
  `InvalidParameterError` at parameter validation (verified live).
- `:1135` — `__init__(self, estimator, *, code_size=1.5, random_state=None,
  n_jobs=None)`.
- `:1183-1188` — `self.classes_ = np.unique(y)`; raises `ValueError` if
  `n_classes == 0`.
- `:1189` — `n_estimators = int(n_classes * self.code_size)` — **FLOOR via
  `int()`, NO `max(2)`**.
- `:1193-1199` — `self.code_book_ = random_state.uniform(size=(n_classes,
  n_estimators))`; `code_book_[code_book_ > 0.5] = 1.0`; then **if the base
  estimator `hasattr(decision_function)`: `code_book_[code_book_ != 1] = -1.0`
  (codes `{-1,1}`) ELSE `code_book_[code_book_ != 1] = 0.0` (codes `{0,1}` for
  predict-only bases)** — verified live: a `RandomForestClassifier` base (no
  `decision_function`) yields `{0.0, 1.0}`.
- `:1208-1213` — `Parallel(_fit_binary(estimator, X, Y[:, i], ...))` for each
  code COLUMN `i` — one binary estimator per column.
- `:1222-1245` — `predict`: `Y = np.array([_predict_binary(e, X) for e in
  estimators_], order="F", dtype=float64).T` — the RAW per-binary predictions;
  `pred = pairwise_distances_argmin(Y, self.code_book_, metric="euclidean")`
  (euclidean distance from raw `Y` to each code row, argmin = FIRST-on-tie);
  `return self.classes_[pred]`.

## Requirements

R-DEV mental test applied per REQ ("numerical/API/structural contract" → MATCH;
"Cython/CPython footgun / RNG" → deviate; "missing feature" → NOT-STARTED with a
blocker). Per R-DEFER-2 / R-HONEST-2 classification is binary
(SHIPPED / NOT-STARTED).

- REQ-CHAIN-CORE (chain fit/predict, default cv=None — true-Y-during-fit,
  predicted-during-predict, un-permute to original column order): ferrolearn
  `fn fit_chain` (shared by both estimators) resolves `chain_order = order or
  0..n_targets`, and for each `t` in chain order fits a fresh
  `factory()` pipeline on `inputs = hcat(x, chained)` against `y.column(t)`,
  then APPENDS the TRUE `y[[i, t]]` to `chained` — matching sklearn `cv=None`
  `X_aug = hstack((X, Y[:, order_]))` (`:738-744`, `:789-795`). `impl Predict for
  Fitted{Classifier,Regressor}Chain` sequentially predicts, feeds PRIOR
  PREDICTIONS (`preds[i]`) as appended chain features, and writes each link to
  `out[[i, self.order[step]]]` — the materialized analog of sklearn's
  `Y_feature_chain` cascade + `Y_output_chain[:, inv_order]` un-permute
  (`:659-695`). **MATCH** (R-DEV-1/R-DEV-3 — structural/output contract; the
  un-permute to original column order is the defining behavior). Verified
  END-TO-END below against the live oracle (`RegressorChain(LinearRegression(),
  cv=None)` on a deterministic two-output problem): the predicted matrix equals
  sklearn column-by-column and is in ORIGINAL column order. SHIPPED for BOTH
  `ClassifierChain` and `RegressorChain`.
- REQ-CHAIN-ORDER (explicit `order` list + un-permute respects it): ferrolearn
  `.order(Vec<usize>)` builder; `fit_chain` validates `o.len() == n_targets`
  (`FerroError::InvalidParameter`) and processes targets in that order, with
  `predict` writing `out[[i, order[step]]]` so a non-identity order is
  un-permuted back to original columns. sklearn `order=[...]` (`:724`, `:733`)
  with `inv_order` (`:693-695`). **MATCH** (R-DEV-3). Verified live with
  `order=[1,0]`: the predicted matrix still equals `Y` in ORIGINAL column order.
  SHIPPED. (NOTE: ferrolearn does NOT enforce that the explicit list is a
  PERMUTATION of `0..n_targets` — only its LENGTH — whereas sklearn raises
  `ValueError("invalid order")` on a non-permutation (`:733`); a duplicate/
  out-of-range index would alias/overflow a column. Length-only validation, a
  minor validation-gap note within the SHIPPED happy path, not the un-permute
  behavior itself.)
- REQ-OCC-MECH (OutputCodeClassifier fit/predict mechanic — per-column binary
  estimators, nearest-code euclidean argmin): ferrolearn `fn fit` builds sorted
  unique `classes`, rejects `len != n` (ShapeMismatch) / `n == 0`
  (InsufficientSamples) / `< 2` classes (InvalidParameter), generates an
  `(k, n_codes)` code matrix, and fits one binary pipeline per code COLUMN;
  `impl Predict for FittedOutputCodeClassifier` thresholds each binary `predict`
  to a vote and picks the class whose code ROW minimises euclidean distance
  (`if d < best_dist` ⇒ FIRST-on-tie). Mirrors sklearn per-column `_fit_binary`
  (`:1208-1213`) + `pairwise_distances_argmin(Y, code_book_, "euclidean")`
  (FIRST-on-tie, `:1244-1245`). **MATCH on the mechanic** (R-DEV-3 — one
  estimator per column + euclidean-argmin first-on-tie). SHIPPED for the
  STRUCTURE only; the code-COUNT (REQ-OCC-NCODES), code-VALUES + raw-vs-
  thresholded distance (REQ-OCC-PREDICT) and the exact code book (REQ-OCC-RNG)
  diverge — see below. (Honest scope: this REQ covers the per-column-fit +
  euclidean-argmin SKELETON; the numeric n_codes/code-value facets are pinned
  separately.)
- REQ-OCC-NCODES (OutputCodeClassifier `n_codes` formula — DETERMINISTIC FIXABLE
  DIVERGENCE): ferrolearn `let n_codes = ((self.code_size * k as f64).ceil() as
  usize).max(2)` — CEIL with a floor of 2. sklearn `n_estimators = int(n_classes
  * self.code_size)` (`:1189`) — FLOOR via `int()`, NO `max(2)`. At
  `code_size=1.5, k=3`: ferrolearn `ceil(4.5)=5`, sklearn `int(4.5)=4`.
  **DEVIATE — deterministic FIXABLE** (R-DEV-2: a numeric-contract footgun, not
  RNG-bound — the COUNT is independent of the code book's random values). The
  critic pins a FAILING `#[test]` on `n_estimators()` asserting `4`. ALSO:
  sklearn raises `InvalidParameterError` for `code_size <= 0` (constraint `:1130`,
  verified live); ferrolearn's `.max(2)` MASKS small/zero/negative `code_size`
  (e.g. `code_size=0` ⇒ `n_codes=2` instead of an error) — a coupled validation
  facet of the same divergence. NOT-STARTED (#1830).
- REQ-OCC-PREDICT (OutputCodeClassifier code VALUES + raw-vs-thresholded
  predict): ferrolearn codes are `{-1.0, +1.0}` UNCONDITIONALLY
  (`code[[i,j]] = if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 }`), and
  `predict` thresholds each binary output to a `{-1.0, +1.0}` VOTE
  (`if preds[i] > 0.5 { 1.0 } else { -1.0 }`) before the euclidean distance.
  sklearn chooses code values BY BASE TYPE: predict-only bases ⇒ codes `{0,1}`
  with `code_book_[!=1]=0` (`:1199`), decision_function bases ⇒ `{-1,1}`
  (`:1197`), and the distance is to the RAW `_predict_binary` `Y` (NOT a
  thresholded vote, `:1239-1244`). With ferrolearn's `Pipeline` base (a
  predict-only `{0,1}` output), sklearn would use codes `{0,1}` on raw `Y`,
  whereas ferrolearn uses codes `{-1,1}` on thresholded `{-1,1}` votes — a
  DIFFERENT distance geometry. **DEVIATE — architectural**, and COUPLED to the
  R-DEFER-3 RNG carve-out (the exact code book can't match SmallRng vs numpy, so
  end-to-end prediction equality is not testable; only the n_codes COUNT is a
  deterministic pin). NOT-STARTED (#1831).
- REQ-CHAIN-CV (cv-based leakage-free chain features): sklearn `cv` ⇒ the
  prior-target FEATURE for later links comes from `cross_val_predict` (`:797-812`)
  instead of the true `Y`. ferrolearn `fit_chain` ALWAYS appends the true
  `y[[i, t]]` (cv=None semantics only) — there is no `cv` parameter. **MATCH-intent
  / missing-feature.** NOT-STARTED (#1832).
- REQ-CHAIN-RANDOM-ORDER (`order='random'` + `random_state` permutation):
  ferrolearn supports only the DEFAULT (`0..n_targets`) and an explicit list;
  there is no `random_state` field and no permutation path. sklearn
  `order='random'` ⇒ `random_state.permutation` (`:730-732`). **MATCH-intent /
  missing-feature.** NOT-STARTED (#1833).
- REQ-CHAIN-METHOD (ClassifierChain `chain_method` proba/decision cascade):
  ferrolearn cascades the binary `predict` output ONLY (no `chain_method`).
  sklearn ClassifierChain selects `predict`/`predict_proba`/`decision_function`
  for the cascade FEATURES (`:773-781`, `chain_method_`). **MATCH-intent /
  missing-feature.** NOT-STARTED (#1834).
- REQ-CHAIN-PROBA (ClassifierChain `predict_proba`/`predict_log_proba`/`classes_`):
  ABSENT — `FittedClassifierChain` exposes only `n_targets()`/`order()` and
  `Predict::predict`. sklearn `predict_proba` (`:1035-1048`),
  `predict_log_proba`, `classes_`. **MATCH-intent / missing-feature.**
  NOT-STARTED (#1835).
- REQ-OCC-RNG (OutputCodeClassifier exact code book): ferrolearn fills the code
  matrix from `SmallRng` (`seed_from_u64`); sklearn from
  `RandomState.uniform > 0.5`. The exact random code book CANNOT match
  bit-for-bit. **DEVIATE — R-DEFER-3 RNG carve-out** (blocker, NO failing test).
  The STRUCTURAL shipped part is seed-determinism-across-runs (a fixed
  `random_state(seed)` reproduces the same code matrix within ferrolearn).
  NOT-STARTED (#1836).
- REQ-SAMPLE-WEIGHT-NJOBS (`sample_weight` / `n_jobs` / `**fit_params`): ABSENT
  — `fit` takes only `(x, y)`; constructors take only the factory (+ `code_size`/
  `random_state` for OCC). sklearn threads `**fit_params` and parallelizes with
  `n_jobs` (chain `:794`; OCC `:1208`). **MATCH-intent / missing-feature.**
  NOT-STARTED (#1837).
- REQ-X-1 (R-SUBSTRATE ndarray+rand→ferray-core): production code imports
  `use ndarray::{Array1, Array2}` and `rand::rngs::SmallRng`; the destination
  substrate is `ferray-core` (R-SUBSTRATE-1). NOT-STARTED (#1838).
- REQ-X-2 (non-test production consumer): the boundary meta-estimator types
  `ClassifierChain`/`RegressorChain`/`OutputCodeClassifier` (+ their Fitted
  forms) are the public API (S5 / R-DEFER-1) and are re-exported from `lib.rs`.
  SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side). The oracle is the
installed sklearn 1.5.2; run from `/tmp` (the source clone at
`/home/doll/scikit-learn` is the read-only cite tree, not built).

- AC-CHAIN-CORE (REQ-CHAIN-CORE — SHIPPED, verified END-TO-END with a
  deterministic base): use `LinearRegression` (exact, no-noise) so the predicted
  `(n_samples, n_targets)` matrix is matched element-for-element; `cv=None`
  matches ferrolearn's true-Y-during-fit. The KEY adversarial check is original
  COLUMN ORDER (un-permute), not chain order.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multioutput import RegressorChain
  from sklearn.linear_model import LinearRegression
  X = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.],[0.,2.],[2.,2.]])
  # y0 = 2x0+3x1+1 ; y1 = -1x0+4x1   (exact)
  Y = np.column_stack([2*X[:,0]+3*X[:,1]+1.0, -1*X[:,0]+4*X[:,1]])
  rc = RegressorChain(LinearRegression(), cv=None).fit(X, Y)
  P = rc.predict(X)
  print('order_:', list(rc.order_))                # -> [0, 1]
  print('shape:', P.shape, 'matches Y exactly:', np.allclose(P, Y))
  "
  # -> order_: [0, 1]
  # -> shape: (6, 2) matches Y exactly: True
  ```
  ferrolearn's `fit_chain` (`cv=None` true-Y append) + `Fitted*Chain::predict`
  (cascade prior predictions, write `out[[i, order[step]]]`) MATCHES: the
  predicted matrix equals `Y` in original column order.
- AC-CHAIN-ORDER (REQ-CHAIN-ORDER — SHIPPED, non-identity un-permute):
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multioutput import RegressorChain
  from sklearn.linear_model import LinearRegression
  X = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.],[0.,2.],[2.,2.]])
  Y = np.column_stack([2*X[:,0]+3*X[:,1]+1.0, -1*X[:,0]+4*X[:,1]])
  rc = RegressorChain(LinearRegression(), order=[1,0], cv=None).fit(X, Y)
  P = rc.predict(X)
  print('order_:', list(rc.order_), 'matches Y in ORIGINAL order:', np.allclose(P, Y))
  "
  # -> order_: [1, 0] matches Y in ORIGINAL order: True
  ```
  ferrolearn `ClassifierChain::new(factory()).order(vec![1,0])` /
  `RegressorChain` writes `out[[i, order[step]]]`, so the explicit non-identity
  order is un-permuted back to original columns; `classifier_chain_explicit_order`
  asserts `fitted.order() == &[1, 0]`.
- AC-OCC-NCODES (REQ-OCC-NCODES — DIVERGENCE, the critic's pin): the n_codes
  COUNT is deterministic (independent of the random code values).
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multiclass import OutputCodeClassifier
  from sklearn.ensemble import RandomForestClassifier
  X = np.repeat(np.arange(3),3).reshape(-1,1).astype(float)
  y = np.repeat(np.arange(3),3)
  occ = OutputCodeClassifier(RandomForestClassifier(random_state=0), code_size=1.5, random_state=0).fit(X,y)
  print('n_estimators:', len(occ.estimators_), 'code_book shape:', occ.code_book_.shape)
  print('code values (predict-only base):', sorted(set(occ.code_book_.ravel().tolist())))
  try:
      OutputCodeClassifier(RandomForestClassifier(), code_size=0.0).fit(X,y)
  except Exception as e:
      print('code_size=0 ->', type(e).__name__)
  "
  # -> n_estimators: 4 code_book shape: (3, 4)
  # -> code values (predict-only base): [0.0, 1.0]
  # -> code_size=0 -> InvalidParameterError
  ```
  ferrolearn `OutputCodeClassifier::new(factory()).code_size(1.5).random_state(0)
  .fit(...).n_estimators()` returns `((1.5*3).ceil()).max(2) == 5`, NOT `4`.
  The critic pins a FAILING `#[test]` asserting `n_estimators() == 4`.
  ferrolearn's code values are `{-1,1}` (not `{0,1}`) and `code_size=0` returns
  `n_codes=2` instead of an error (REQ-OCC-PREDICT/validation facets, #1830/#1831).
- AC-OCC-MECH (REQ-OCC-MECH — SHIPPED mechanic, structure only): per-column
  binary estimators + euclidean nearest-code argmin (first-on-tie).
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multiclass import OutputCodeClassifier
  from sklearn.ensemble import RandomForestClassifier
  X = np.repeat(np.arange(3),3).reshape(-1,1).astype(float)
  y = np.repeat(np.arange(3),3)
  occ = OutputCodeClassifier(RandomForestClassifier(random_state=0), code_size=2.0, random_state=0).fit(X,y)
  P = occ.predict(X)
  print('classes_:', occ.classes_.tolist(), 'n_est:', len(occ.estimators_))
  print('predict in classes:', set(P.tolist()) <= {0,1,2})
  "
  # -> classes_: [0, 1, 2] n_est: 6
  # -> predict in classes: True
  ```
  ferrolearn `output_code_basic_shapes` asserts `classes() == [0,1,2]`,
  `n_estimators() >= 2`, every prediction in `{0,1,2}`, `predict().len() == 9`.
  The euclidean-argmin first-on-tie (`if d < best_dist`) mirrors sklearn
  `pairwise_distances_argmin`. (END-TO-END label equality is NOT testable here:
  the code book diverges — REQ-OCC-RNG #1836 — and the code VALUES + n_codes
  diverge — #1830/#1831.)
- AC-CHAIN-CV (REQ-CHAIN-CV — ABSENT): sklearn `cv` injects `cross_val_predict`
  features.
  ```
  cd /tmp && python3 -c "
  import inspect
  from sklearn.multioutput import RegressorChain
  print('RegressorChain init:', list(inspect.signature(RegressorChain.__init__).parameters)[1:])
  "
  # -> RegressorChain init: ['base_estimator', 'order', 'cv', 'random_state', 'verbose']
  ```
  ferrolearn `RegressorChain::new(factory())` + `.order(...)` expose NO `cv`
  channel; `fit_chain` always appends true `y`. NOT-STARTED (#1832).
- AC-CHAIN-RANDOM-ORDER (REQ-CHAIN-RANDOM-ORDER — ABSENT): sklearn
  `order='random'` permutes via `random_state`.
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from sklearn.multioutput import RegressorChain
  from sklearn.linear_model import LinearRegression
  X = np.zeros((6,2)); Y = np.zeros((6,3))
  rc = RegressorChain(LinearRegression(), order='random', random_state=0).fit(X, Y)
  print('order_ (random):', list(rc.order_))
  "
  # -> order_ (random): [2, 0, 1]   (a permutation, seed-dependent)
  ```
  ferrolearn has no `random_state`/`order='random'` path. NOT-STARTED (#1833).
- AC-CHAIN-PROBA (REQ-CHAIN-PROBA / REQ-CHAIN-METHOD — ABSENT): sklearn
  ClassifierChain exposes `predict_proba`/`classes_`/`chain_method`.
  ```
  cd /tmp && python3 -c "
  import inspect, numpy as np
  from sklearn.multioutput import ClassifierChain
  from sklearn.linear_model import LogisticRegression
  print('CC init:', list(inspect.signature(ClassifierChain.__init__).parameters)[1:])
  X = np.array([[0.],[1.],[2.],[3.],[4.],[5.]]); Y = np.array([[0,0],[0,1],[0,1],[1,0],[1,1],[1,1]])
  cc = ClassifierChain(LogisticRegression()).fit(X, Y)
  print('predict_proba shape:', cc.predict_proba(X).shape, 'classes_:', [c.tolist() for c in cc.classes_])
  "
  # -> CC init: ['base_estimator', 'order', 'cv', 'chain_method', 'random_state', 'verbose']
  # -> predict_proba shape: (6, 2) classes_: [[0, 1], [0, 1]]
  ```
  `grep -n "predict_proba\|chain_method\|fn classes" ferrolearn-model-sel/src/chain.rs`
  is empty for `FittedClassifierChain`. NOT-STARTED (#1834/#1835).
- AC-OCC-INIT (REQ-SAMPLE-WEIGHT-NJOBS — ABSENT): sklearn OCC accepts `n_jobs`.
  ```
  cd /tmp && python3 -c "
  import inspect
  from sklearn.multiclass import OutputCodeClassifier
  print('OCC init:', list(inspect.signature(OutputCodeClassifier.__init__).parameters)[1:])
  "
  # -> OCC init: ['estimator', 'code_size', 'random_state', 'n_jobs']
  ```
  ferrolearn `OutputCodeClassifier::new` exposes only the factory + `code_size` +
  `random_state`; no `n_jobs`/`sample_weight`/`fit_params`. NOT-STARTED (#1837).
- AC-X-2 (REQ-X-2 — SHIPPED): `grep -n "pub use chain" ferrolearn-model-sel/src/
  lib.rs` shows the re-export of `ClassifierChain, FittedClassifierChain,
  FittedOutputCodeClassifier, FittedRegressorChain, OutputCodeClassifier,
  RegressorChain`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-CHAIN-CORE (chain fit/predict, cv=None: true-Y fit, predicted predict, un-permute to original col order) | SHIPPED | impl `fn fit_chain in chain.rs` (shared by both estimators): `chain_order = order or 0..n_targets`; for each `&t in &chain_order` fits `factory()` on `inputs = hcat(x, chained)` against `y.column(t)`, then `new_chained[[i, chained.ncols()]] = y[[i, t]]` — appends the TRUE training target (cv=None). `impl Predict<Array2<f64>> for FittedRegressorChain`/`FittedClassifierChain`: per `(step, est)`, `inputs = hcat(x, chained)`, `preds = est.predict(&inputs)?`, append `preds[i]` to `chained`, write `out[[i, self.order[step]]] = preds[i]` — cascades PRIOR PREDICTIONS and writes via `order[step]` ⇒ UN-PERMUTES to original column order. Mirrors sklearn `_BaseChain.fit` cv=None `X_aug=hstack((X, Y[:, order_]))` (`sklearn/multioutput.py:738-744`, `:789-795`) + `_get_predictions` cascade `Y_feature_chain` + `Y_output_chain[:, inv_order]` (`:659-695`). LIVE ORACLE (AC-CHAIN-CORE): `RegressorChain(LinearRegression(), cv=None).fit(X,Y).predict(X)` shape `(6,2)`, `order_==[0,1]`, `np.allclose(P, Y)==True` (exact column-by-column recovery of two distinct no-noise outputs IN ORIGINAL COLUMN ORDER). VERIFIED MATCH. Non-test consumer: REQ-X-2 (boundary type + `lib.rs` re-export). |
| REQ-CHAIN-ORDER (explicit `order` list + un-permute respects it) | SHIPPED | impl `.order(Vec<usize>)` builder on both `ClassifierChain`/`RegressorChain`; `fn fit_chain` validates `o.len() != n_targets` ⇒ `FerroError::InvalidParameter` and processes `chain_order = o.clone()`; `predict` writes `out[[i, self.order[step]]]`. Mirrors sklearn `order=[...]` (`sklearn/multioutput.py:724`, `:733`) + `inv_order` un-permute (`:693-695`). LIVE ORACLE (AC-CHAIN-ORDER): `RegressorChain(LinearRegression(), order=[1,0], cv=None)` ⇒ `order_==[1,0]` yet `np.allclose(P, Y)==True` in ORIGINAL column order; ferrolearn `classifier_chain_explicit_order` asserts `order()==&[1,0]`. VALIDATION-GAP NOTE: ferrolearn checks the list LENGTH only, NOT that it is a permutation of `0..n_targets` (sklearn raises `ValueError("invalid order")`, `:733`) — a duplicate/out-of-range index would alias a column; happy-path SHIPPED, the permutation check is the honest underclaim. Non-test consumer: REQ-X-2. |
| REQ-OCC-MECH (OutputCodeClassifier mechanic — per-column binary fit + euclidean nearest-code argmin, first-on-tie) | SHIPPED | impl `fn fit in chain.rs` (`OutputCodeClassifier`): sorted/dedup `classes`, rejects `y.len()!=n` (ShapeMismatch) / `n==0` (InsufficientSamples) / `< 2` classes (InvalidParameter); builds `(k, n_codes)` `code` matrix; `for j in 0..n_codes` fits one `(self.make_pipeline)()` on the per-column binary labels. `impl Predict for FittedOutputCodeClassifier`: per est `votes[[i,j]]`, then per sample picks `if d < best_dist` (euclidean to each code ROW, FIRST-on-tie) ⇒ `classes[best]`. Mirrors sklearn `_fit_binary` per code column (`sklearn/multiclass.py:1208-1213`) + `pairwise_distances_argmin(Y, code_book_, "euclidean")` (`:1244-1245`, first-on-tie). LIVE ORACLE (AC-OCC-MECH): `OutputCodeClassifier(RandomForestClassifier(), code_size=2.0, random_state=0)` ⇒ `classes_==[0,1,2]`, predictions ⊆ `{0,1,2}`; ferrolearn `output_code_basic_shapes` asserts the same structure. SHIPPED for the STRUCTURE only — the n_codes COUNT (REQ-OCC-NCODES), code VALUES (REQ-OCC-PREDICT) and exact code book (REQ-OCC-RNG) diverge; END-TO-END label equality is NOT claimed. Non-test consumer: REQ-X-2. |
| REQ-OCC-NCODES (n_codes formula — DETERMINISTIC FIXABLE DIVERGENCE) | NOT-STARTED | open prereq blocker #1830. impl `fn fit in chain.rs`: `let n_codes = ((self.code_size * k as f64).ceil() as usize).max(2)` — CEIL + floor-of-2. sklearn `n_estimators = int(n_classes * self.code_size)` (`sklearn/multiclass.py:1189`) — FLOOR via `int()`, NO `max(2)`. At `code_size=1.5, k=3`: ferrolearn `ceil(4.5)=5`, sklearn `int(4.5)=4`. LIVE ORACLE (AC-OCC-NCODES): `OutputCodeClassifier(RandomForestClassifier(), code_size=1.5, random_state=0).fit(X,y); len(estimators_)==4`, `code_book_.shape==(3,4)`; ferrolearn `n_estimators()==5`. DETERMINISTIC FIXABLE (count is RNG-independent): the critic pins a FAILING `#[test]` asserting `n_estimators()==4`. COUPLED VALIDATION FACET: sklearn raises `InvalidParameterError` for `code_size<=0` (constraint `:1130`), but ferrolearn's `.max(2)` returns `n_codes=2` (no error) — also pinnable. |
| REQ-OCC-PREDICT (code VALUES {-1,1} vs {0,1}-by-base + raw-vs-thresholded distance) | NOT-STARTED | open prereq blocker #1831. impl `fn fit in chain.rs`: `code[[i,j]] = if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 }` — codes `{-1,+1}` UNCONDITIONALLY; `predict` thresholds `votes[[i,j]] = if preds[i] > 0.5 { 1.0 } else { -1.0 }` before the euclidean distance. sklearn picks code values BY BASE: predict-only ⇒ `code_book_[!=1]=0` ⇒ `{0,1}` (`sklearn/multiclass.py:1199`), decision_function ⇒ `{-1,1}` (`:1197`); distance is to the RAW `_predict_binary` `Y`, NOT a thresholded vote (`:1239-1244`). With ferrolearn's predict-only `Pipeline` base, sklearn would use `{0,1}` on raw `Y` (verified live: code values `[0.0, 1.0]`); ferrolearn uses `{-1,1}` on thresholded votes — different distance geometry. ARCHITECTURAL, COUPLED to the R-DEFER-3 RNG carve-out (#1836) — the exact code book can't match, so end-to-end prediction equality is not testable; only the n_codes COUNT (#1830) is a deterministic pin. Absent end-to-end. |
| REQ-CHAIN-CV (cv leakage-free chain features) | NOT-STARTED | open prereq blocker #1832. impl `fn fit_chain in chain.rs` ALWAYS appends `y[[i, t]]` (true target) to `chained` — there is NO `cv` parameter on either estimator. sklearn `cv` ⇒ the prior-target FEATURE for later links is `cross_val_predict(base_estimator, X_aug[:, :col_idx], y, cv)` (`sklearn/multioutput.py:797-812`), leakage-free. LIVE ORACLE (AC-CHAIN-CV): `RegressorChain.__init__` params include `'cv'`; ferrolearn `new(factory())`/`.order(...)` expose no cv channel. Absent end-to-end. |
| REQ-CHAIN-RANDOM-ORDER (`order='random'` + random_state) | NOT-STARTED | open prereq blocker #1833. impl `fn fit_chain in chain.rs` supports only the DEFAULT `(0..n_targets)` and an explicit `Vec<usize>`; `ClassifierChain`/`RegressorChain` have no `random_state` field, no permutation path. sklearn `order='random'` ⇒ `self.order_ = random_state.permutation(Y.shape[1])` (`sklearn/multioutput.py:730-732`). LIVE ORACLE (AC-CHAIN-RANDOM-ORDER): `RegressorChain(..., order='random', random_state=0).order_` is a seed-dependent permutation; ferrolearn has neither knob. Absent end-to-end. |
| REQ-CHAIN-METHOD (ClassifierChain chain_method proba/decision cascade) | NOT-STARTED | open prereq blocker #1834. impl `impl Predict for FittedClassifierChain in chain.rs` cascades the binary `predict` output ONLY (`preds = est.predict(&inputs)?`, append `preds[i]`); there is no `chain_method`. sklearn ClassifierChain selects `predict`/`predict_proba`/`decision_function` for the cascade FEATURES via `chain_method_` (`sklearn/multioutput.py:773-781`, `_get_predictions` `chain_method` `:668-684`). LIVE ORACLE (AC-CHAIN-PROBA): `ClassifierChain.__init__` includes `'chain_method'`; ferrolearn has no such parameter. Absent end-to-end. |
| REQ-CHAIN-PROBA (ClassifierChain predict_proba / predict_log_proba / classes_) | NOT-STARTED | open prereq blocker #1835. `FittedClassifierChain in chain.rs` exposes only `n_targets()`/`order()` + `impl Predict::predict` — NO `predict_proba`/`predict_log_proba`/`classes()` (`grep -n "predict_proba\|fn classes" chain.rs` is empty for the chain types). sklearn `predict_proba = _get_predictions(X, "predict_proba")` (`sklearn/multioutput.py:1035-1048`), `predict_log_proba`, `classes_`. LIVE ORACLE (AC-CHAIN-PROBA): `ClassifierChain(LogisticRegression()).fit(X,Y).predict_proba(X)` shape `(6,2)`, `classes_==[[0,1],[0,1]]`; ferrolearn cannot produce probabilities. Absent end-to-end. |
| REQ-OCC-RNG (OutputCodeClassifier exact code book) | NOT-STARTED | open prereq blocker #1836. impl `fn fit in chain.rs` fills `code` from `SmallRng` (`SmallRng::seed_from_u64(seed)` / `from_os_rng`), `rng.random::<f64>() < 0.5`. sklearn fills `code_book_` from `RandomState.uniform(size=(k, n))` thresholded at `> 0.5` (`sklearn/multiclass.py:1193-1194`). The exact random code book CANNOT match bit-for-bit (different PRNG). R-DEFER-3 RNG carve-out (blocker, NO failing test). The STRUCTURAL shipped facet: a fixed `.random_state(seed)` reproduces the same code matrix across ferrolearn runs (seed-determinism), pinned by `output_code_basic_shapes` using `.random_state(7)`. Exact cross-library code-book equality: out of scope (R-DEFER-3). |
| REQ-SAMPLE-WEIGHT-NJOBS (sample_weight / n_jobs / **fit_params) | NOT-STARTED | open prereq blocker #1837. impl `fn fit in chain.rs` (all three estimators) takes only `(x, y)`; `ClassifierChain::new`/`RegressorChain::new` take only the factory, `OutputCodeClassifier::new` the factory (+ `.code_size`/`.random_state` builders). No `sample_weight`/`n_jobs`/`**fit_params` channel. sklearn threads `**routed_params.estimator.fit` (chain `sklearn/multioutput.py:791-795`) and parallelizes OCC with `Parallel(n_jobs=self.n_jobs)` (`sklearn/multiclass.py:1208`); OCC `__init__` has `n_jobs` (`:1135`). LIVE ORACLE (AC-OCC-INIT): `OutputCodeClassifier.__init__` params `['estimator','code_size','random_state','n_jobs']`; ferrolearn lacks `n_jobs`/weights/fit-params. Absent end-to-end. |
| REQ-X-1 (R-SUBSTRATE ndarray+rand→ferray-core) | NOT-STARTED | open prereq blocker #1838. Production code in `chain.rs` imports `use ndarray::{Array1, Array2}` and `use rand::rngs::SmallRng` and operates on `Array2<f64>`/`Array1<usize>` (`fn hcat` builds `Array2::<f64>::zeros`, `predict` returns `Array2<f64>`/`Array1<usize>`). Per R-SUBSTRATE-1 the destination array type is `ferray-core`, not `ndarray`. Not migrated (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | Crate re-export: `ferrolearn-model-sel/src/lib.rs` `pub mod chain;` + `pub use chain::{ClassifierChain, FittedClassifierChain, FittedOutputCodeClassifier, FittedRegressorChain, OutputCodeClassifier, RegressorChain};`. Per S5 / R-DEFER-1 the boundary meta-estimator types ARE the public API and are grandfathered (existing pub surface). CAVEAT (honest underclaim): `grep -rn "ClassifierChain\|RegressorChain\|OutputCodeClassifier" ferrolearn-*/src/ \| grep -v 'chain.rs\|tests'` finds the `lib.rs` re-export plus a `multiclass.rs` DOC-table cross-reference only — NO dedicated non-test internal CALLER and NO `ferrolearn-python` binding yet. SHIPPED on the boundary re-export per S5, not a dedicated production caller; the missing Python binding is noted. The base estimator is a `PipelineFactory` CLOSURE (R-DEV-7 idiom for sklearn's wrapped/`clone`d-`estimator`) — noted, not pinned. |

## Architecture

ferrolearn splits each meta-estimator into an unfitted/Fitted pair (CLAUDE.md
naming): `ClassifierChain { make_pipeline: PipelineFactory, order: Option<Vec<usize>> }`
→ `FittedClassifierChain { estimators: Vec<FittedPipeline<f64>>, order: Vec<usize> }`,
the STRUCTURALLY IDENTICAL `RegressorChain`/`FittedRegressorChain`, and
`OutputCodeClassifier { make_pipeline, code_size: f64, random_state: Option<u64> }`
→ `FittedOutputCodeClassifier { estimators, code: Array2<f64>, classes: Vec<usize> }`.
sklearn keeps a single `_BaseChain` base (`:625`) whose post-`fit` state is
`order_` + `estimators_` (+ `chain_method_`/`classes_` on ClassifierChain), and a
single `OutputCodeClassifier` (`:1025`) with `classes_` + `code_book_` +
`estimators_`. ferrolearn's two chain types are byte-for-byte the same
`fit_chain`/`predict` logic differing only in type names — a deliberate
duplication of the sklearn base/subclass split.

**The base-estimator representation is an R-DEV-7 deviation.** sklearn wraps a
`base_estimator`/`estimator` object and `clone`s it per link/column (chain `:736`,
OCC `_fit_binary` `:1208`); ferrolearn takes a `make_pipeline: Box<dyn Fn() ->
Pipeline<f64> + Send + Sync>` closure called once per link/column to produce a
fresh binary pipeline (same idiom as `multiclass.rs`/`multioutput.rs`) — noted,
not pinned.

**The chain core (REQ-CHAIN-CORE, REQ-CHAIN-ORDER — SHIPPED) is a faithful 1:1
translation of the cv=None path.** `fit_chain` resolves `chain_order`, and for
each `t` fits a fresh pipeline on `hcat(x, chained)` against `y.column(t)`, then
appends the TRUE `y[[i, t]]` — exactly sklearn's `X_aug = hstack((X, Y[:,
order_]))` (`:738-744`). `Fitted*Chain::predict` cascades the link's OWN PRIOR
PREDICTIONS (`preds[i]`) as appended features and writes `out[[i, order[step]]]`
— the materialized analog of sklearn's `Y_feature_chain` cascade
(`:671`,`:677`,`:684`) plus `Y_output_chain[:, inv_order]` un-permute
(`:693-695`). Verified END-TO-END against the live oracle (`LinearRegression`,
two distinct no-noise outputs): the predicted matrix equals sklearn's
column-by-column IN ORIGINAL COLUMN ORDER for both the default order and a
non-identity `order=[1,0]`. **There is NO un-permute bug** — ferrolearn returns
original column order, not chain order — so the adversarial check passes. The
ONLY chain caveat is a validation underclaim: ferrolearn checks `order.len()`
only, not that the list is a permutation of `0..n_targets` (sklearn raises
`ValueError("invalid order")`, `:733`).

**The OutputCodeClassifier mechanic (REQ-OCC-MECH — SHIPPED structure) ships, but
three numeric/RNG facets diverge.** `fit` builds sorted unique `classes`, an
`(k, n_codes)` code matrix, and one binary pipeline per COLUMN; `predict`
thresholds each binary output to a vote and picks the euclidean-nearest code row
(`if d < best_dist`, FIRST-on-tie) — matching sklearn's per-column `_fit_binary`
(`:1208-1213`) + `pairwise_distances_argmin(..., "euclidean")` (`:1244`,
first-on-tie). The DETERMINISTIC FIXABLE divergence is **REQ-OCC-NCODES (#1830):**
`n_codes = ceil(code_size*k).max(2)` vs sklearn `int(code_size*k)` (FLOOR, no
`max(2)`) — `5` vs `4` at `code_size=1.5, k=3` (oracle-verified, RNG-independent;
the critic pins `n_estimators()==4`). Coupled to it, ferrolearn's `.max(2)`
MASKS the `code_size<=0` `InvalidParameterError` sklearn raises (constraint
`:1130`). **REQ-OCC-PREDICT (#1831)** is architectural: ferrolearn uses code
values `{-1,1}` UNCONDITIONALLY on THRESHOLDED `{-1,1}` votes, while sklearn picks
`{0,1}` for predict-only bases (`:1199`, the case matching ferrolearn's Pipeline)
or `{-1,1}` for decision_function bases (`:1197`) and computes distance to the
RAW `_predict_binary` `Y` (`:1239-1244`) — a different distance geometry, and
coupled to the **REQ-OCC-RNG (#1836)** R-DEFER-3 carve-out (SmallRng vs numpy
`RandomState.uniform` ⇒ the exact code book can't match, so end-to-end label
equality is NOT testable; only the n_codes COUNT is a deterministic pin).

What is structurally ABSENT vs sklearn (missing-feature blockers): `cv`
leakage-free chain features (REQ-CHAIN-CV, #1832), `order='random'` +
`random_state` (REQ-CHAIN-RANDOM-ORDER, #1833), ClassifierChain `chain_method`
cascade (REQ-CHAIN-METHOD, #1834), ClassifierChain `predict_proba`/
`predict_log_proba`/`classes_` (REQ-CHAIN-PROBA, #1835), `sample_weight`/
`n_jobs`/`**fit_params` (REQ-SAMPLE-WEIGHT-NJOBS, #1837), and the `ferray`
substrate (REQ-X-1, #1838). SHIPPED: the cv=None chain core + explicit order +
un-permute (REQ-CHAIN-CORE, REQ-CHAIN-ORDER), the OutputCodeClassifier mechanic
(REQ-OCC-MECH), and the boundary re-export (REQ-X-2).

Invariants: chain — `y.nrows() == x.nrows()` (`FerroError::ShapeMismatch`);
`n_targets >= 1` (`FerroError::InvalidParameter`); `order.len() == n_targets`;
`predict` returns `(n_samples, n_targets)` in ORIGINAL column order; each link's
features are `x` ++ the prior links' (true Y at fit, predictions at predict).
OutputCodeClassifier — `y.len() == x.nrows()`; `n_samples >= 1`; `>= 2` classes;
`classes` sorted; `n_codes` columns each fit a binary pipeline; `predict` is the
euclidean-nearest code row (first-on-tie).

## Verification

Commands establishing the SHIPPED claims (baseline
`fb0860d86ed1bdda80368926c092e96f708c2404`). The oracle is the installed sklearn
1.5.2 (`cd /tmp`; the source clone at `/home/doll/scikit-learn` is the read-only
cite tree):

- `cargo test -p ferrolearn-model-sel --lib chain` → the in-file tests
  `chain::tests::{classifier_chain_fits_and_predicts_shape,
  regressor_chain_fits_and_predicts_shape, classifier_chain_explicit_order,
  output_code_basic_shapes}` (shape/order/structure pins).
- REQ-CHAIN-CORE SHIPPED oracle (live sklearn, deterministic LinearRegression
  base, R-CHAR-3): AC-CHAIN-CORE — `RegressorChain(LinearRegression(), cv=None)
  .fit(X,Y).predict(X)` shape `(6,2)`, `order_==[0,1]`, `np.allclose(P, Y)==True`
  (exact column-by-column recovery IN ORIGINAL COLUMN ORDER). ferrolearn
  `fit_chain` (true-Y append) + `predict` (cascade prior predictions, write
  `out[[i, order[step]]]`) MATCHES.
- REQ-CHAIN-ORDER SHIPPED oracle: AC-CHAIN-ORDER — `RegressorChain(...,
  order=[1,0], cv=None)` ⇒ `order_==[1,0]` yet `np.allclose(P, Y)==True` in
  ORIGINAL column order (un-permute verified for a non-identity order);
  `classifier_chain_explicit_order` asserts `order()==&[1,0]`. (Underclaim:
  permutation-validity check is length-only.)
- REQ-OCC-MECH SHIPPED oracle: AC-OCC-MECH — `OutputCodeClassifier(
  RandomForestClassifier(), code_size=2.0, random_state=0)` ⇒ `classes_==[0,1,2]`,
  predictions ⊆ `{0,1,2}`; `output_code_basic_shapes` asserts the same structure +
  euclidean-argmin first-on-tie. (Structure only — no end-to-end label equality.)
- REQ-OCC-NCODES DETERMINISTIC FIXABLE oracle (#1830 — the critic's pin):
  AC-OCC-NCODES — `OutputCodeClassifier(RandomForestClassifier(), code_size=1.5,
  random_state=0).fit(X,y); len(estimators_)==4`, `code_book_.shape==(3,4)`,
  code values `[0.0, 1.0]`, `code_size=0 → InvalidParameterError`; ferrolearn
  `n_estimators()==ceil(4.5).max(2)==5`. The critic pins a FAILING `#[test]` on
  `n_estimators()` asserting `4` (and optionally the `code_size<=0` error mask).
- REQ-OCC-PREDICT architectural divergence (#1831): code values `{-1,1}` vs
  sklearn `{0,1}`-by-base + thresholded-vs-raw distance — coupled to the
  REQ-OCC-RNG (#1836) carve-out; no end-to-end pin.
- REQ-CHAIN-CV ABSENT oracle (#1832): AC-CHAIN-CV — `RegressorChain.__init__`
  params include `'cv'`; ferrolearn always appends true `y`. REQ-CHAIN-RANDOM-ORDER
  ABSENT (#1833): AC-CHAIN-RANDOM-ORDER — `order='random'` gives a seed-dependent
  permutation; ferrolearn lacks the knob.
- REQ-CHAIN-METHOD / REQ-CHAIN-PROBA ABSENT oracle (#1834/#1835): AC-CHAIN-PROBA —
  `ClassifierChain.__init__` includes `'chain_method'`;
  `ClassifierChain(LogisticRegression()).fit(X,Y).predict_proba(X)` shape `(6,2)`,
  `classes_==[[0,1],[0,1]]`; ferrolearn `FittedClassifierChain` exposes neither.
- REQ-SAMPLE-WEIGHT-NJOBS ABSENT oracle (#1837): AC-OCC-INIT —
  `OutputCodeClassifier.__init__` params `['estimator','code_size','random_state',
  'n_jobs']`; ferrolearn lacks `n_jobs`/weights/fit-params.
- REQ-OCC-RNG R-DEFER-3 carve-out (#1836): `chain.rs` uses `SmallRng` vs sklearn
  `RandomState.uniform` — exact code book not matchable; seed-determinism within
  ferrolearn is the shipped facet.
- REQ-X-2 consumer: `grep -n "pub use chain" ferrolearn-model-sel/src/lib.rs`
  shows `pub use chain::{ClassifierChain, FittedClassifierChain,
  FittedOutputCodeClassifier, FittedRegressorChain, OutputCodeClassifier,
  RegressorChain};`. `grep -rn "ClassifierChain\|RegressorChain\|
  OutputCodeClassifier" ferrolearn-*/src/ | grep -v 'chain.rs\|tests'` shows the
  re-export + a `multiclass.rs` doc-table cross-reference only (no dedicated
  internal caller, no Python binding — honest underclaim).
- REQ-X-1 substrate: `grep -n "ndarray\|SmallRng" ferrolearn-model-sel/src/chain.rs`
  shows `use ndarray::{Array1, Array2}` + `use rand::rngs::SmallRng` — wrong
  substrate, migration owed (#1838).

SHIPPED (4): REQ-CHAIN-CORE (cv=None chain fit/predict — VERIFIED END-TO-END
column-by-column vs sklearn LinearRegression base, ORIGINAL column order),
REQ-CHAIN-ORDER (explicit non-identity order un-permute — verified live),
REQ-OCC-MECH (per-column binary fit + euclidean nearest-code argmin — structure),
REQ-X-2 (boundary re-export consumer; no dedicated caller / no Python binding —
honest underclaim). NOT-STARTED (9): REQ-OCC-NCODES (#1830 — DETERMINISTIC
FIXABLE), REQ-OCC-PREDICT (#1831 — architectural/RNG-coupled), REQ-CHAIN-CV
(#1832), REQ-CHAIN-RANDOM-ORDER (#1833), REQ-CHAIN-METHOD (#1834),
REQ-CHAIN-PROBA (#1835), REQ-OCC-RNG (#1836 — R-DEFER-3 carve-out),
REQ-SAMPLE-WEIGHT-NJOBS (#1837), REQ-X-1 (#1838 — ferray substrate).

Per R-DEFER-2 every REQ is binary SHIPPED/NOT-STARTED. The DETERMINISTIC FIXABLE
divergence the critic should pin as a FAILING test is **REQ-OCC-NCODES** (#1830 —
`OutputCodeClassifier.n_estimators()` is `ceil(code_size*k).max(2)==5` where
sklearn `int(code_size*k)==4` at `code_size=1.5, k=3`; the COUNT is
RNG-independent, so an oracle-grounded `#[test]` asserting `4` fails today; the
`code_size<=0` `.max(2)` mask is a coupled facet). REQ-CHAIN-CORE / REQ-CHAIN-ORDER
are verified SHIPPED (chain core + un-permute MATCH the live oracle, original
column order) — NOT pins. REQ-OCC-PREDICT (#1831) and REQ-OCC-RNG (#1836) are
architectural / R-DEFER-3 RNG carve-outs (the exact code book + raw-vs-thresholded
predict can't match SmallRng-vs-numpy) — blockers, not pins. The remaining
NOT-STARTED REQs (#1832/#1833/#1834/#1835/#1837) are missing features and
(#1838) substrate — blockers, not pins.

Least-confident SHIPPED claim: REQ-OCC-MECH — SHIPPED rests on the STRUCTURAL
mechanic (per-column binary fit + euclidean nearest-code argmin first-on-tie)
matching the live oracle's `classes_`/shape/membership, NOT end-to-end label
equality: the n_codes COUNT (#1830), code VALUES (#1831) and exact code book
(#1836) all diverge, so two of three numeric facets of the strategy are
NOT-STARTED. The honest reading is "the reduction skeleton ships; its numerics
diverge" — a follow-up critic pins #1830 and a builder reconciles #1831 within
the #1836 RNG carve-out.
