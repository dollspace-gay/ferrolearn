# ferrolearn-python classifiers — sklearn LogisticRegression + DecisionTreeClassifier + RandomForestClassifier + KNeighborsClassifier + GaussianNB binding shim

<!--
tier: 3-component
status: draft
baseline-commit: ff4b37171
upstream-paths:
  - sklearn/linear_model/_logistic.py        # class LogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator) @ :810 — __init__ @ :1129; predict_proba @ :1395; n_iter_ @ :1276; attrs classes_/coef_/intercept_/n_iter_/n_features_in_
  - sklearn/tree/_classes.py                  # class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree) @ :698 — __init__ @ :945; predict_proba @ :1017; attrs classes_/feature_importances_/n_features_in_
  - sklearn/ensemble/_forest.py               # class RandomForestClassifier(ForestClassifier) @ :1170 — __init__ @ :1494; predict @ :883 (soft-vote); predict_proba @ :922; attrs classes_/feature_importances_/n_features_in_
  - sklearn/neighbors/_classification.py      # class KNeighborsClassifier(KNeighborsMixin, ClassifierMixin, NeighborsBase) @ :39 — __init__ @ :193; predict @ :240; predict_proba @ :307; attrs classes_/n_features_in_
  - sklearn/naive_bayes.py                     # class GaussianNB(_BaseNB) @ :147 — __init__ @ :234; predict @ :86; predict_proba @ :128; attrs classes_/theta_/var_/epsilon_/class_count_/class_prior_/n_features_in_
-->

## Summary

`ferrolearn-python/src/classifiers.rs` is the PyO3 marshalling shim binding FIVE
classification estimators to CPython:
`#[pyclass(name = "_RsLogisticRegression")] RsLogisticRegression`
(over `ferrolearn_linear::FittedLogisticRegression<f64>`),
`#[pyclass(name = "_RsDecisionTreeClassifier")] RsDecisionTreeClassifier`
(over `ferrolearn_tree::FittedDecisionTreeClassifier<f64>`),
`#[pyclass(name = "_RsRandomForestClassifier")] RsRandomForestClassifier`
(over `ferrolearn_tree::FittedRandomForestClassifier<f64>`),
`#[pyclass(name = "_RsKNeighborsClassifier")] RsKNeighborsClassifier`
(over `ferrolearn_neighbors::FittedKNeighborsClassifier<f64>`),
and `#[pyclass(name = "_RsGaussianNB")] RsGaussianNB`
(over `ferrolearn_bayes::FittedGaussianNB<f64>`).
`ferrolearn-python/python/ferrolearn/_classifiers.py` wraps each as a sklearn
`ClassifierMixin`/`BaseEstimator` subclass — `LogisticRegression`,
`DecisionTreeClassifier`, `RandomForestClassifier`, `KNeighborsClassifier`,
`GaussianNB` — so `import ferrolearn` mirrors
`from sklearn.linear_model import LogisticRegression`,
`from sklearn.tree import DecisionTreeClassifier`,
`from sklearn.ensemble import RandomForestClassifier`,
`from sklearn.neighbors import KNeighborsClassifier`,
`from sklearn.naive_bayes import GaussianNB`.
They mirror **`sklearn.linear_model.LogisticRegression`** (`_logistic.py:810`),
**`sklearn.tree.DecisionTreeClassifier`** (`_classes.py:698`),
**`sklearn.ensemble.RandomForestClassifier`** (`_forest.py:1170`),
**`sklearn.neighbors.KNeighborsClassifier`** (`_classification.py:39`), and
**`sklearn.naive_bayes.GaussianNB`** (`naive_bayes.py:147`).

The classifier *correctness* (LBFGS logistic fit; CART tree induction; bagged
forest soft-vote; k-NN weighted vote; Gaussian likelihood) lives DOWN in
`ferrolearn-linear/src/logistic_regression.rs`,
`ferrolearn-tree/src/{decision_tree,random_forest}.rs`,
`ferrolearn-neighbors/src/knn.rs`, and `ferrolearn-bayes/src/gaussian.rs`, each
audited by that crate's `//!` REQ status table. THIS unit is the **sklearn-API
marshalling shim** only: constructor parameter ABI (R-DEV-2), attribute exposure
+ output object contract (R-DEV-3), method surface, and array/label coercion
across the Python↔Rust boundary (the `_encode_labels`/`_decode_labels`
LabelEncoder round-trip owned by `conversions.md` REQ-LABEL-MARSHAL; the
`ferray::numpy_interop` substrate owned by `conversions.md` REQ-FERRAY #2027).
Semantic/numerical and missing-knob divergences are owned by the downstream
library crates; this doc CITES their existing blockers rather than re-filing, and
owns only the binding-level surface.

**Verification model: B (pytest vs sklearn 1.5.2).** Per goal.md §"The
verification model (B)", this unit is verified by
`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` comparing
`import ferrolearn` against the installed `import sklearn` 1.5.2 oracle, plus the
live-sklearn oracle for the constructor-ABI and attribute boundary. As of
baseline `ff4b37171` the gauntlet is GREEN: **542 passed**. All five estimators —
`LogisticRegression()`, `DecisionTreeClassifier()`,
`RandomForestClassifier(n_estimators=5, random_state=42)`,
`KNeighborsClassifier(n_neighbors=3)`, `GaussianNB()` — are exercised by
`tests/test_check_estimator.py` (`parametrize_with_checks`) and
`tests/test_cross_val_score.py` (`cross_val_score`).

### Headline single-wrapper-fixable ABI divergences (3 distinct, R-DEV-2)

1. **REQ-RF-NESTIMATORS-POSITIONAL** — `_classifiers.py::RandomForestClassifier.__init__`
   puts `n_estimators` AFTER the leading `*`, making it keyword-only, so
   `ferrolearn.RandomForestClassifier(10)` raises `TypeError`, whereas sklearn's
   `n_estimators` is positional-or-keyword (`_forest.py:1494`,
   `RandomForestClassifier(10).n_estimators == 10`). Single-line wrapper fix:
   move `n_estimators` before the `*`.
2. **REQ-KNN-NNEIGHBORS-POSITIONAL** — `_classifiers.py::KNeighborsClassifier.__init__`
   puts `n_neighbors` after the leading `*`, so `ferrolearn.KNeighborsClassifier(3)`
   raises `TypeError`, whereas sklearn's `n_neighbors` is positional-or-keyword
   (`_classification.py:193`, `KNeighborsClassifier(3).n_neighbors == 3`).
   Single-line wrapper fix: move `n_neighbors` before the `*`.
3. **REQ-LOGREG-MAXITER-DEFAULT** — `_classifiers.py::LogisticRegression.__init__`
   defaults `max_iter=1000`, whereas sklearn's default is `max_iter=100`
   (`_logistic.py:1129`). Single-line wrapper fix: change the default to `100`.
   RISK: ferrolearn's LBFGS may need >100 iterations to converge to the
   downstream-verified ~1e-8 coef/intercept parity; if changing the default to
   `100` breaks the value-parity green guards, that is a SEPARATE downstream
   convergence concern (the ABI match is still `100` per sklearn) — the critic
   decides whether the ABI fix lands with a downstream convergence blocker.

Divergence classes (beyond the three headlines):
1. **api-conformance (SHIPPED core)** — on the DEFAULT parameter path all five
   expose `fit`/`predict` with the right shapes/types plus the wrapper-exposed
   getters, and the marshalled VALUES match the live sklearn oracle.
   `check_estimator` + `cross_val_score` pass for all five.
2. **predict_proba absent (NOT-STARTED)** — the `RandomForestClassifier` and
   `KNeighborsClassifier` wrappers (and their `_Rs*` bindings) expose NO
   `predict_proba`, whereas sklearn's RF (`_forest.py:922`) and KNN
   (`_classification.py:307`) do — owned downstream (RF #671-class binding gap;
   KNN value-correct but no binding consumer #877).
3. **feature_importances_ not surfaced (NOT-STARTED)** — the Rust
   `RsDecisionTreeClassifier`/`RsRandomForestClassifier` expose a
   `feature_importances_` getter, but the Python wrappers never read it, so
   `ferrolearn.DecisionTreeClassifier().fit(X,y).feature_importances_` raises
   `AttributeError` — sklearn exposes it (`_classes.py` / `_forest.py`).
4. **n_iter_ faked (NOT-STARTED)** — `_classifiers.py::LogisticRegression.fit`
   sets `self.n_iter_ = self.max_iter` (a FAKE = `1000`), never the actual LBFGS
   count sklearn exposes (live `n_iter_ == [11]`) — owned downstream (#450).
5. **missing constructor params (NOT-STARTED)** — each wrapper omits most sklearn
   ctor params (penalty/solver/class_weight/random_state, criterion/max_features,
   bootstrap/oob_score, weights/metric/p, priors) — owned downstream.
6. **value parity off the default path / tie-breaking (NOT-STARTED)** — KNN
   tie-breaking, RF RNG/bootstrap determinism, LogReg solver path — R-DEV-1, cited
   downstream.
7. **substrate (NOT-STARTED, R-SUBSTRATE-1)** — `classifiers.rs` round-trips numpy
   ↔ `ndarray` via `crate::conversions::*` (rust-numpy), not
   `ferray::numpy_interop`/`ferray-core`; owned by `conversions.md` REQ-FERRAY
   #2027.

## Upstream reference (sklearn 1.5.2, live oracle = installed sklearn 1.5.2)

Lines stable at tag 1.5.2 / commit 156ef14.

### `sklearn.linear_model.LogisticRegression` (`_logistic.py:810`)

- **`__init__`** (`_logistic.py:1129`):
  `LogisticRegression(self, penalty='l2', *, dual=False, tol=1e-4, C=1.0,
  fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None,
  solver='lbfgs', max_iter=100, multi_class='deprecated', verbose=0,
  warm_start=False, n_jobs=None, l1_ratio=None)`. `penalty` is
  positional-or-keyword (PRECEDES `*`); `C`/`fit_intercept`/`tol`/`max_iter` are
  keyword-only. `max_iter` default is **100**.
- **`fit`**: default `solver='lbfgs'`, `penalty='l2'`; sets `coef_`
  `(1, n_features)` (binary), `intercept_` `(1,)`, `classes_`, `n_iter_`
  `(:1276`/`:1376`, the ACTUAL count), `n_features_in_`.
- **`predict_proba`** (`_logistic.py:1395`): sigmoid/softmax, rows sum to 1.
- **attributes**: `classes_` `(n_classes,)`, `coef_` `(1, n_features)` binary,
  `intercept_` `(1,)`, `n_iter_` `(n_classes_or_1,)` int32, `n_features_in_`.

### `sklearn.tree.DecisionTreeClassifier` (`_classes.py:698`)

- **`__init__`** (`_classes.py:945`):
  `DecisionTreeClassifier(self, *, criterion='gini', splitter='best',
  max_depth=None, min_samples_split=2, min_samples_leaf=1,
  min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
  max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None,
  ccp_alpha=0.0, monotonic_cst=None)`. ALL params keyword-only (`*` FIRST).
- **`predict_proba`** (`_classes.py:1017`): per-leaf class frequencies.
- **attributes**: `classes_`, `feature_importances_` `(n_features,)`,
  `n_features_in_`.

### `sklearn.ensemble.RandomForestClassifier` (`_forest.py:1170`)

- **`__init__`** (`_forest.py:1494`):
  `RandomForestClassifier(self, n_estimators=100, *, criterion='gini',
  max_depth=None, min_samples_split=2, min_samples_leaf=1,
  min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None,
  min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
  random_state=None, verbose=0, warm_start=False, class_weight=None,
  ccp_alpha=0.0, max_samples=None, monotonic_cst=None)`. `n_estimators` is
  positional-or-keyword (PRECEDES `*`); everything after is keyword-only.
- **`predict`** (`_forest.py:883`): SOFT vote — `classes_[argmax(predict_proba)]`.
- **`predict_proba`** (`_forest.py:922`): mean of per-tree class probabilities.
- **attributes**: `classes_`, `feature_importances_`, `n_features_in_`.

### `sklearn.neighbors.KNeighborsClassifier` (`_classification.py:39`)

- **`__init__`** (`_classification.py:193`):
  `KNeighborsClassifier(self, n_neighbors=5, *, weights='uniform',
  algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None,
  n_jobs=None)`. `n_neighbors` is positional-or-keyword (PRECEDES `*`).
- **`predict`** (`_classification.py:240`): weighted vote, smallest-label tie-break.
- **`predict_proba`** (`_classification.py:307`): normalized weighted vote shares.
- **attributes**: `classes_`, `n_features_in_`.

### `sklearn.naive_bayes.GaussianNB` (`naive_bayes.py:147`)

- **`__init__`** (`naive_bayes.py:234`):
  `GaussianNB(self, *, priors=None, var_smoothing=1e-9)`. ALL params keyword-only
  (`*` FIRST).
- **`predict`** (`naive_bayes.py:86`) / **`predict_proba`** (`naive_bayes.py:128`):
  argmax / normalized joint log-likelihood.
- **attributes**: `classes_`, `theta_`, `var_`, `epsilon_`, `class_count_`,
  `class_prior_`, `n_features_in_`.

Live oracle (installed sklearn 1.5.2, run from `/tmp`; R-CHAR-3 — expected values
from sklearn, NEVER from ferrolearn). `X = [[0,0],[1,1],[2,4],[3,9],[4,1],[5,2]]`,
`y = [0,0,1,1,0,1]`:

```
LogisticRegression.__init__ -> (self, penalty='l2', *, ..., C=1.0, ..., max_iter=100, ...)   # C keyword-only, max_iter default 100
DecisionTreeClassifier.__init__ -> (self, *, criterion='gini', ..., max_depth=None, ...)     # all keyword-only
RandomForestClassifier.__init__ -> (self, n_estimators=100, *, ...)                          # n_estimators positional
KNeighborsClassifier.__init__   -> (self, n_neighbors=5, *, weights='uniform', ...)          # n_neighbors positional
GaussianNB.__init__             -> (self, *, priors=None, var_smoothing=1e-9)                # all keyword-only

RandomForestClassifier(10).n_estimators -> 10   ; KNeighborsClassifier(3).n_neighbors -> 3   # positional accepted

LogisticRegression().fit(X,y):  classes_=[0,1] coef_=[[0.4366,0.958]] intercept_=[-3.3158] n_iter_=[11]  pred=[0,0,1,1,0,1]  proba[0]=[0.964966,0.035034]
DecisionTreeClassifier().fit(X,y): classes_=[0,1] feature_importances_=[0.0,1.0] pred=[0,0,1,1,0,1]
RandomForestClassifier(random_state=0).fit(X,y): classes_=[0,1] pred=[0,0,1,1,0,1]  (has predict_proba)
KNeighborsClassifier(3).fit(X,y): classes_=[0,1] pred=[0,0,0,1,0,1]  (has predict_proba)
GaussianNB().fit(X,y): classes_=[0,1] pred=[0,0,1,1,0,1] proba[0]=[0.993664,0.006336]
```

ferrolearn at baseline `ff4b37171` (live, `import ferrolearn`):
- `LogisticRegression().fit(X,y)` → `classes_=[0,1]`, `coef_=[[0.4366,0.958]]`,
  `intercept_=[-3.3157]` (matches the oracle element-wise to ≤1e-3 at default
  stopping; downstream verifies ~1e-8 at convergence), but `n_iter_=1000` (FAKE =
  `max_iter`) vs oracle `[11]`. `C` is keyword-only — MATCHES; `max_iter` default
  is `1000` vs sklearn `100` — DIVERGES.
- `DecisionTreeClassifier().fit(X,y)` → `classes_=[0,1]`, `pred=[0,0,1,1,0,1]`
  (matches), but `feature_importances_` raises `AttributeError` (the wrapper never
  reads the Rust getter). `__init__` is fully keyword-only — MATCHES sklearn.
- `RandomForestClassifier(10)` → `TypeError: RandomForestClassifier.__init__()
  takes 1 positional argument but 2 were given` (`n_estimators` keyword-only) —
  DIVERGES (headline 1). No `predict_proba`, no `feature_importances_` surfaced.
- `KNeighborsClassifier(3)` → `TypeError: ... takes 1 positional argument but 2
  were given` (`n_neighbors` keyword-only) — DIVERGES (headline 2). No
  `predict_proba`. `KNeighborsClassifier(n_neighbors=3).fit(X,y).predict(X)` →
  `[0,0,1,1,0,1]` vs oracle `[0,0,0,1,0,1]` (tie/value boundary, downstream).
- `GaussianNB().fit(X,y)` → `classes_=[0,1]`, `pred=[0,0,1,1,0,1]`,
  `proba[0]=[0.993664,0.006336]` — MATCHES element-wise. `__init__` is fully
  keyword-only — MATCHES sklearn (missing `priors`).

## Requirements

Grouped by estimator (`REQ-LOGREG-*`, `REQ-DT-*`, `REQ-RF-*`, `REQ-KNN-*`,
`REQ-GNB-*`) plus shared `REQ-CONSUMER`/`REQ-SUBSTRATE`.

### LogisticRegression

- REQ-LOGREG-API-CONFORM: `ferrolearn.LogisticRegression` exposes the
  `sklearn.linear_model.LogisticRegression` method surface — `fit`/`predict`/
  `predict_proba` (bound on `_RsLogisticRegression` in `classifiers.rs`, wrapped
  in `_classifiers.py`) plus `score` (`ClassifierMixin`) — and the fitted
  attributes `classes_`, `coef_` `(1, n_features)`, `intercept_` `(1,)`,
  `n_features_in_`, with values matching the sklearn oracle on the DEFAULT
  `penalty='l2'`/`C=1.0`/`solver='lbfgs'` path. `check_estimator` +
  `cross_val_score` pass.
- REQ-LOGREG-C-ABI: `ferrolearn.LogisticRegression`'s `C` is keyword-only,
  matching sklearn `__init__(self, penalty='l2', *, ..., C=1.0, ...)`
  (`_logistic.py:1129`, `C` AFTER the `*`). Live: both make `C` keyword-only —
  this ABI MATCHES.
- REQ-LOGREG-MAXITER-DEFAULT (**HEADLINE 3/3, single-wrapper-fixable**):
  `ferrolearn.LogisticRegression`'s `max_iter` default is `100`, matching sklearn
  `max_iter=100` (`_logistic.py:1129`). ferrolearn defaults `max_iter=1000`.
- REQ-LOGREG-NITER: `ferrolearn.LogisticRegression`'s `n_iter_` is the ACTUAL
  LBFGS iteration count, matching sklearn (`_logistic.py:1276`/`:1376`; oracle
  `n_iter_ == [11]`). ferrolearn fakes `n_iter_ = max_iter` (live `1000`). [Owned
  downstream: `ferrolearn-linear` REQ-9..20 blocker #450 — `FittedLogisticRegression`
  does not track the count.]
- REQ-LOGREG-PARAMS: `ferrolearn.LogisticRegression` exposes the
  `penalty`/`dual`/`intercept_scaling`/`class_weight`/`random_state`/`solver`/
  `warm_start`/`l1_ratio`/`n_jobs` constructor params (`_logistic.py:1129`).
  [Param surface + behavior owned downstream: penalty #442, solver #443,
  multi_class=ovr #444, class_weight #445, dual #446, intercept_scaling #447,
  l1_ratio #448, warm_start #449, random_state/n_jobs #452.]
- REQ-LOGREG-VALUE-PARITY: `coef_`/`intercept_`/`predict_proba` match sklearn on
  the DEFAULT path (R-DEV-1). [Default path SHIPPED — downstream `ferrolearn-linear`
  REQ-1/2/4 verify binary + multinomial coef/intercept/predict_proba to ~1e-8 at
  convergence; the binary `decision_function` `(n,)`-shape ABI is #454.]

### DecisionTreeClassifier

- REQ-DT-API-CONFORM: `ferrolearn.DecisionTreeClassifier` exposes the
  `sklearn.tree.DecisionTreeClassifier` method surface — `fit`/`predict`/
  `predict_proba` (bound on `_RsDecisionTreeClassifier`, wrapped) plus `score` —
  and `classes_`, `n_features_in_`, with `pred`/`predict_proba` matching the
  sklearn oracle on the DEFAULT `criterion='gini'` path. `check_estimator` +
  `cross_val_score` pass.
- REQ-DT-CTOR-ABI: `ferrolearn.DecisionTreeClassifier`'s `__init__` is fully
  keyword-only, matching sklearn `__init__(self, *, criterion='gini', ...)`
  (`_classes.py:945`, the `*` is FIRST). Live: both fully keyword-only — this ABI
  MATCHES.
- REQ-DT-FEATURE-IMPORTANCES: `ferrolearn.DecisionTreeClassifier` exposes
  `feature_importances_` `(n_features,)`, matching sklearn. The Rust
  `RsDecisionTreeClassifier::feature_importances_` getter EXISTS (over
  `FittedDecisionTreeClassifier::feature_importances`, downstream REQ-5 SHIPPED),
  but the Python wrapper `_classifiers.py::DecisionTreeClassifier` never reads it,
  so `feature_importances_` raises `AttributeError`. [Binding-level gap: the
  wrapper must surface the existing getter — blocker to be filed by critic.]
- REQ-DT-PARAMS: `ferrolearn.DecisionTreeClassifier` exposes
  `criterion`/`splitter`/`max_features`/`random_state`/`max_leaf_nodes`/
  `min_weight_fraction_leaf`/`min_impurity_decrease`/`class_weight`/`ccp_alpha`
  (`_classes.py:945`). The wrapper + `RsDecisionTreeClassifier::new` take
  `max_depth`/`min_samples_split`/`min_samples_leaf` only. [Library supports
  criterion/min_impurity_decrease/ccp_alpha/max_leaf_nodes/class_weight
  (downstream REQ-1/3/7 SHIPPED) but the binding does not expose them;
  `max_features` #665, `random_state`/`splitter='random'` #670.]
- REQ-DT-VALUE-PARITY: `pred`/`predict_proba` match sklearn on the DEFAULT path
  (R-DEV-1). [Default gini tree SHIPPED — downstream `ferrolearn-tree` REQ-1/2/6
  verify CART best-split + predict/predict_proba to the oracle; multi-output 2-D
  `y` #668.]

### RandomForestClassifier

- REQ-RF-API-CONFORM: `ferrolearn.RandomForestClassifier` exposes the
  `sklearn.ensemble.RandomForestClassifier` method surface — `fit`/`predict`
  (bound on `_RsRandomForestClassifier`, wrapped) plus `score` — and `classes_`,
  `n_features_in_`, with `pred` matching the sklearn oracle on the DEFAULT path
  (soft-vote, `_forest.py:883`). `check_estimator` + `cross_val_score` pass.
- REQ-RF-NESTIMATORS-POSITIONAL (**HEADLINE 1/3, single-wrapper-fixable**):
  `ferrolearn.RandomForestClassifier` accepts `n_estimators` positionally —
  `RandomForestClassifier(10).n_estimators == 10`, matching sklearn
  `__init__(self, n_estimators=100, *, ...)` (`_forest.py:1494`, `n_estimators`
  before the `*`). ferrolearn makes it keyword-only.
- REQ-RF-PREDICT-PROBA: `ferrolearn.RandomForestClassifier` exposes
  `predict_proba` `(n_samples, n_classes)` (rows sum to 1), matching sklearn
  (`_forest.py:922`, mean of per-tree probas). The wrapper has NO `predict_proba`
  method and `RsRandomForestClassifier` exposes NO `predict_proba` getter, even
  though `FittedRandomForestClassifier::predict_proba` exists and is SHIPPED
  downstream (`ferrolearn-tree` REQ-5). [Binding-level gap: bind + surface the
  existing `predict_proba` — blocker to be filed by critic.]
- REQ-RF-FEATURE-IMPORTANCES: `ferrolearn.RandomForestClassifier` exposes
  `feature_importances_`. The Rust `RsRandomForestClassifier::feature_importances_`
  getter EXISTS (downstream REQ-6 SHIPPED) but the Python wrapper never reads it,
  so it raises `AttributeError`. [Binding-level gap — blocker to be filed by
  critic.]
- REQ-RF-PARAMS: `ferrolearn.RandomForestClassifier` exposes
  `criterion`/`max_features`/`bootstrap`/`oob_score`/`max_samples`/`class_weight`/
  `n_jobs`/`warm_start`/`ccp_alpha`/`max_leaf_nodes`/`min_impurity_decrease`
  (`_forest.py:1494`). The wrapper takes `n_estimators`/`max_depth`/
  `min_samples_split`/`min_samples_leaf`/`random_state` only. [Owned downstream:
  bootstrap/max_samples #672, oob_score #675, class_weight #676, tree-param
  passthrough #671.]
- REQ-RF-VALUE-PARITY: `pred`/`predict_proba` match sklearn at a given
  `random_state` (R-DEV-1). [The deterministic contract (soft-vote, proba mean,
  in-impl reproducibility) is SHIPPED downstream (`ferrolearn-tree` REQ-4/5/9);
  exact numpy-MT bootstrap/feature-subsample parity is the DOCUMENTED RNG boundary
  #673.]

### KNeighborsClassifier

- REQ-KNN-API-CONFORM: `ferrolearn.KNeighborsClassifier` exposes the
  `sklearn.neighbors.KNeighborsClassifier` method surface — `fit`/`predict`
  (bound on `_RsKNeighborsClassifier`, wrapped) plus `score` — and `classes_`,
  `n_features_in_`, on the DEFAULT `weights='uniform'`/`metric='minkowski'`/`p=2`
  path. `check_estimator` + `cross_val_score` pass.
- REQ-KNN-NNEIGHBORS-POSITIONAL (**HEADLINE 2/3, single-wrapper-fixable**):
  `ferrolearn.KNeighborsClassifier` accepts `n_neighbors` positionally —
  `KNeighborsClassifier(3).n_neighbors == 3`, matching sklearn
  `__init__(self, n_neighbors=5, *, ...)` (`_classification.py:193`, before the
  `*`). ferrolearn makes it keyword-only.
- REQ-KNN-PREDICT-PROBA: `ferrolearn.KNeighborsClassifier` exposes `predict_proba`
  `(n_samples, n_classes)`, matching sklearn (`_classification.py:307`). The
  wrapper has NO `predict_proba` and `RsKNeighborsClassifier` exposes NO
  `predict_proba` getter; `FittedKNeighborsClassifier::predict_proba` is
  value-correct but has NO non-test consumer. [Owned downstream:
  `ferrolearn-neighbors` REQ-2/REQ-11 blocker #877 — the binding under-exposes the
  surface.]
- REQ-KNN-PARAMS: `ferrolearn.KNeighborsClassifier` exposes
  `weights`/`algorithm`/`leaf_size`/`p`/`metric`/`metric_params`/`n_jobs`
  (`_classification.py:193`). The wrapper takes `n_neighbors` only. [Owned
  downstream: `ferrolearn-neighbors` REQ-10/REQ-11 blockers #876 (weights/metric/p)
  + #877 (PyO3 surface).]
- REQ-KNN-VALUE-PARITY: `pred` matches sklearn including tie-breaking (R-DEV-1).
  [Uniform-vote predict + smallest-label tie-break is SHIPPED downstream
  (`ferrolearn-neighbors` REQ-1); Euclidean-only/no-distance-weighting and 2-D
  query divergences are owned downstream #876. Live: ferrolearn
  `KNeighborsClassifier(n_neighbors=3).predict(X)=[0,0,1,1,0,1]` vs oracle
  `[0,0,0,1,0,1]` on this dataset — a value/tie boundary owned downstream.]

### GaussianNB

- REQ-GNB-API-CONFORM: `ferrolearn.GaussianNB` exposes the
  `sklearn.naive_bayes.GaussianNB` method surface — `fit`/`predict`/`predict_proba`
  (bound on `_RsGaussianNB`, wrapped) plus `score` — and `classes_`,
  `n_features_in_`, with `pred`/`predict_proba` matching the sklearn oracle on the
  DEFAULT `var_smoothing=1e-9` path. `check_estimator` + `cross_val_score` pass.
- REQ-GNB-CTOR-ABI: `ferrolearn.GaussianNB`'s `__init__` is fully keyword-only,
  matching sklearn `__init__(self, *, priors=None, var_smoothing=1e-9)`
  (`naive_bayes.py:234`, the `*` is FIRST). Live: both fully keyword-only — this
  ABI MATCHES.
- REQ-GNB-PARAMS: `ferrolearn.GaussianNB` exposes the `priors` constructor param
  (`naive_bayes.py:234`). The wrapper + `RsGaussianNB::new` take `var_smoothing`
  only. [Library supports `priors` validation (downstream `ferrolearn-bayes`
  REQ-2 SHIPPED) but the binding does not expose the kwarg — owned downstream
  REQ-8 blocker #897; `sample_weight`/getters #894/#896/#897.]
- REQ-GNB-VALUE-PARITY: `pred`/`predict_proba` match sklearn on the DEFAULT path
  (R-DEV-1). [Default path SHIPPED — downstream `ferrolearn-bayes` REQ-1/3/4
  verify `epsilon_` (global var, no floor), `_joint_log_likelihood`, predict,
  predict_proba to ~1e-9.]

### Shared

- REQ-CONSUMER: the binding IS the public API (R-DEFER-1/S5: boundary estimator
  types ARE the public surface, grandfathered existing pub API); its non-test
  production consumers are the Python wrappers `_classifiers.py::{LogisticRegression,
  DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, GaussianNB}`
  + the `ferrolearn/__init__.py:5-11` re-export, exercised by the pytest gauntlet
  (`test_check_estimator.py`, `test_cross_val_score.py`). The label round-trip
  (`_encode_labels`/`_decode_labels`) is covered by `conversions.md`
  REQ-LABEL-MARSHAL (SHIPPED).
- REQ-SUBSTRATE: the binding's array marshalling is on `ferray::numpy_interop`
  producing `ferray-core` arrays, not rust-numpy + `ndarray` (R-SUBSTRATE-1).
  [Owned by `conversions.md` REQ-FERRAY #2027.]

## Acceptance criteria

All expected values come from the live sklearn 1.5.2 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn. The pytest gauntlet
(`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`) is the
end-to-end check (verification model B); rebuild first if the Rust side changed
(`cd ferrolearn-python && maturin develop`). All ferrolearn live probes below use
`X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]])`,
`y=np.array([0,0,1,1,0,1])`.

- AC-LOGREG-API-CONFORM (REQ-LOGREG-API-CONFORM): `test_check_estimator.py`
  (`parametrize_with_checks([LogisticRegression(), ...])`) +
  `test_cross_val_score.py` pass. Spot oracle:
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import LogisticRegression; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]]); y=np.array([0,0,1,1,0,1]); m=LogisticRegression().fit(X,y); print(m.classes_.tolist(), m.predict(X).tolist(), [round(v,6) for v in m.predict_proba(X)[0]])"`
  → `[0, 1] [0, 0, 1, 1, 0, 1] [0.964966, 0.035034]`.
  `ferrolearn.LogisticRegression().fit(X,y)` matches (live).
- AC-LOGREG-C-ABI (REQ-LOGREG-C-ABI): sklearn oracle
  `cd /tmp && python3 -c "import inspect; from sklearn.linear_model import LogisticRegression; print(inspect.signature(LogisticRegression.__init__).parameters['C'].kind)"`
  → `KEYWORD_ONLY`. ferrolearn wrapper makes `C` keyword-only (live) — ABI
  MATCHES.
- AC-LOGREG-MAXITER-DEFAULT (REQ-LOGREG-MAXITER-DEFAULT): sklearn oracle
  `cd /tmp && python3 -c "import inspect; from sklearn.linear_model import LogisticRegression; print(inspect.signature(LogisticRegression.__init__).parameters['max_iter'].default)"`
  → `100`. ferrolearn:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "import inspect; from ferrolearn import LogisticRegression; print(inspect.signature(LogisticRegression.__init__).parameters['max_iter'].default)"`
  → `1000` (live-confirmed). A critic pins a FAILING pytest asserting the default
  is `100`. FAILS until `_classifiers.py::LogisticRegression.__init__` changes the
  default to `100` (flagging the convergence risk).
- AC-LOGREG-NITER (REQ-LOGREG-NITER): sklearn oracle
  `cd /tmp && python3 -c "import numpy as np; from sklearn.linear_model import LogisticRegression; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]]); y=np.array([0,0,1,1,0,1]); print(LogisticRegression().fit(X,y).n_iter_.tolist())"`
  → `[11]`. ferrolearn `LogisticRegression().fit(X,y).n_iter_` → `1000` (FAKE).
  Owned downstream `ferrolearn-linear` #450.
- AC-LOGREG-PARAMS (REQ-LOGREG-PARAMS): sklearn exposes the extra params
  (`cd /tmp && python3 -c "import inspect; from sklearn.linear_model import LogisticRegression; ps=inspect.signature(LogisticRegression.__init__).parameters; print([p for p in ('penalty','solver','class_weight','random_state','dual','l1_ratio') if p in ps])"`
  → all 6). ferrolearn has none. FAIL until added (owned downstream
  #442-#452).
- AC-DT-API-CONFORM (REQ-DT-API-CONFORM): `test_check_estimator.py`
  (`DecisionTreeClassifier()`) + `test_cross_val_score.py` pass. Spot oracle:
  `cd /tmp && python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]]); y=np.array([0,0,1,1,0,1]); m=DecisionTreeClassifier().fit(X,y); print(m.classes_.tolist(), m.predict(X).tolist())"`
  → `[0, 1] [0, 0, 1, 1, 0, 1]`. `ferrolearn.DecisionTreeClassifier().fit(X,y)`
  matches (live).
- AC-DT-CTOR-ABI (REQ-DT-CTOR-ABI): sklearn oracle
  `cd /tmp && python3 -c "import inspect; from sklearn.tree import DecisionTreeClassifier; print(set(p.kind.name for p in list(inspect.signature(DecisionTreeClassifier.__init__).parameters.values())[1:]))"`
  → `{'KEYWORD_ONLY'}`. ferrolearn wrapper likewise (live) — ABI MATCHES.
- AC-DT-FEATURE-IMPORTANCES (REQ-DT-FEATURE-IMPORTANCES): sklearn oracle
  `cd /tmp && python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]]); y=np.array([0,0,1,1,0,1]); print(np.round(DecisionTreeClassifier().fit(X,y).feature_importances_,4).tolist())"`
  → `[0.0, 1.0]`. ferrolearn FAILS:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "import numpy as np; from ferrolearn import DecisionTreeClassifier; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]]); y=np.array([0,0,1,1,0,1]); print(DecisionTreeClassifier().fit(X,y).feature_importances_)"`
  → `AttributeError: 'DecisionTreeClassifier' object has no attribute
  'feature_importances_'` (live-confirmed). A critic pins a FAILING pytest. FAILS
  until the wrapper reads the existing `RsDecisionTreeClassifier.feature_importances_`
  getter.
- AC-DT-PARAMS (REQ-DT-PARAMS): sklearn exposes the extra params
  (`cd /tmp && python3 -c "import inspect; from sklearn.tree import DecisionTreeClassifier; ps=inspect.signature(DecisionTreeClassifier.__init__).parameters; print([p for p in ('criterion','splitter','max_features','random_state','class_weight','ccp_alpha') if p in ps])"`
  → all 6). ferrolearn has none. FAIL until added (owned downstream
  #665/#670 + binding gap).
- AC-RF-API-CONFORM (REQ-RF-API-CONFORM): `test_check_estimator.py`
  (`RandomForestClassifier(n_estimators=5, random_state=42)`) +
  `test_cross_val_score.py` pass. Spot oracle:
  `cd /tmp && python3 -c "import numpy as np; from sklearn.ensemble import RandomForestClassifier; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]]); y=np.array([0,0,1,1,0,1]); m=RandomForestClassifier(random_state=0).fit(X,y); print(m.classes_.tolist(), m.predict(X).tolist())"`
  → `[0, 1] [0, 0, 1, 1, 0, 1]`. `ferrolearn.RandomForestClassifier(random_state=0).fit(X,y)`
  predicts the same on this dataset (live).
- AC-RF-NESTIMATORS-POSITIONAL (REQ-RF-NESTIMATORS-POSITIONAL): sklearn oracle
  `cd /tmp && python3 -c "from sklearn.ensemble import RandomForestClassifier; print(RandomForestClassifier(10).n_estimators)"`
  → `10`. ferrolearn FAILS:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "from ferrolearn import RandomForestClassifier; RandomForestClassifier(10)"`
  → `TypeError: RandomForestClassifier.__init__() takes 1 positional argument but
  2 were given` (live-confirmed). A critic pins a FAILING pytest asserting
  `RandomForestClassifier(10).n_estimators == 10`. FAILS until
  `_classifiers.py::RandomForestClassifier.__init__` moves `n_estimators` before
  the `*`.
- AC-RF-PREDICT-PROBA (REQ-RF-PREDICT-PROBA): sklearn oracle
  `cd /tmp && python3 -c "import numpy as np; from sklearn.ensemble import RandomForestClassifier; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]]); y=np.array([0,0,1,1,0,1]); print(RandomForestClassifier(random_state=0).fit(X,y).predict_proba(X).shape)"`
  → `(6, 2)`. ferrolearn FAILS:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "from ferrolearn import RandomForestClassifier; print(hasattr(RandomForestClassifier, 'predict_proba'))"`
  → `False` (live-confirmed). A critic pins a FAILING pytest. FAILS until the
  wrapper + `RsRandomForestClassifier` surface the existing
  `FittedRandomForestClassifier::predict_proba` (downstream `ferrolearn-tree`
  REQ-5).
- AC-RF-FEATURE-IMPORTANCES (REQ-RF-FEATURE-IMPORTANCES): sklearn exposes
  `feature_importances_`; ferrolearn FAILS (`AttributeError`, wrapper never reads
  the existing `RsRandomForestClassifier.feature_importances_` getter,
  live-confirmed). A critic pins a FAILING pytest.
- AC-RF-PARAMS (REQ-RF-PARAMS): sklearn exposes
  `criterion`/`max_features`/`bootstrap`/`oob_score`/`class_weight`/`max_samples`
  (`cd /tmp && python3 -c "import inspect; from sklearn.ensemble import RandomForestClassifier; ps=inspect.signature(RandomForestClassifier.__init__).parameters; print([p for p in ('criterion','max_features','bootstrap','oob_score','class_weight','max_samples') if p in ps])"`
  → all 6). ferrolearn has none. FAIL until added (owned downstream
  #671/#672/#675/#676).
- AC-KNN-API-CONFORM (REQ-KNN-API-CONFORM): `test_check_estimator.py`
  (`KNeighborsClassifier(n_neighbors=3)`) + `test_cross_val_score.py` pass. Spot
  oracle:
  `cd /tmp && python3 -c "import numpy as np; from sklearn.neighbors import KNeighborsClassifier; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]]); y=np.array([0,0,1,1,0,1]); print(KNeighborsClassifier(n_neighbors=3).fit(X,y).classes_.tolist())"`
  → `[0, 1]`. `ferrolearn.KNeighborsClassifier(n_neighbors=3).fit(X,y)` exposes
  the same `classes_` (live; check_estimator green).
- AC-KNN-NNEIGHBORS-POSITIONAL (REQ-KNN-NNEIGHBORS-POSITIONAL): sklearn oracle
  `cd /tmp && python3 -c "from sklearn.neighbors import KNeighborsClassifier; print(KNeighborsClassifier(3).n_neighbors)"`
  → `3`. ferrolearn FAILS:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "from ferrolearn import KNeighborsClassifier; KNeighborsClassifier(3)"`
  → `TypeError: KNeighborsClassifier.__init__() takes 1 positional argument but 2
  were given` (live-confirmed). A critic pins a FAILING pytest asserting
  `KNeighborsClassifier(3).n_neighbors == 3`. FAILS until
  `_classifiers.py::KNeighborsClassifier.__init__` moves `n_neighbors` before the
  `*`.
- AC-KNN-PREDICT-PROBA (REQ-KNN-PREDICT-PROBA): sklearn oracle
  `cd /tmp && python3 -c "import numpy as np; from sklearn.neighbors import KNeighborsClassifier; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]]); y=np.array([0,0,1,1,0,1]); print(KNeighborsClassifier(n_neighbors=3).fit(X,y).predict_proba(X).shape)"`
  → `(6, 2)`. ferrolearn FAILS:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "from ferrolearn import KNeighborsClassifier; print(hasattr(KNeighborsClassifier, 'predict_proba'))"`
  → `False` (live-confirmed). Owned downstream `ferrolearn-neighbors` #877.
- AC-KNN-PARAMS (REQ-KNN-PARAMS): sklearn exposes
  `weights`/`algorithm`/`leaf_size`/`p`/`metric` (all present). ferrolearn has
  none. FAIL until added (owned downstream #876/#877).
- AC-GNB-API-CONFORM (REQ-GNB-API-CONFORM): `test_check_estimator.py`
  (`GaussianNB()`) + `test_cross_val_score.py` pass. Spot oracle:
  `cd /tmp && python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; X=np.array([[0.,0.],[1.,1.],[2.,4.],[3.,9.],[4.,1.],[5.,2.]]); y=np.array([0,0,1,1,0,1]); m=GaussianNB().fit(X,y); print(m.classes_.tolist(), m.predict(X).tolist(), [round(v,6) for v in m.predict_proba(X)[0]])"`
  → `[0, 1] [0, 0, 1, 1, 0, 1] [0.993664, 0.006336]`. `ferrolearn.GaussianNB().fit(X,y)`
  matches element-wise (live).
- AC-GNB-CTOR-ABI (REQ-GNB-CTOR-ABI): sklearn oracle
  `cd /tmp && python3 -c "import inspect; from sklearn.naive_bayes import GaussianNB; print(set(p.kind.name for p in list(inspect.signature(GaussianNB.__init__).parameters.values())[1:]))"`
  → `{'KEYWORD_ONLY'}`. ferrolearn wrapper likewise (live) — ABI MATCHES.
- AC-GNB-PARAMS (REQ-GNB-PARAMS): sklearn exposes `priors`
  (`cd /tmp && python3 -c "import inspect; from sklearn.naive_bayes import GaussianNB; print('priors' in inspect.signature(GaussianNB.__init__).parameters)"`
  → `True`). ferrolearn signature is `(self, *, var_smoothing=1e-9)` — absent.
  FAIL until added (owned downstream `ferrolearn-bayes` #897).
- AC-CONSUMER (REQ-CONSUMER):
  `grep -n "_RsLogisticRegression\|_RsDecisionTreeClassifier\|_RsRandomForestClassifier\|_RsKNeighborsClassifier\|_RsGaussianNB" /home/doll/ferrolearn/ferrolearn-python/python/ferrolearn/_classifiers.py`
  shows each wrapper constructs its `_Rs*` class and drives fit/predict;
  `ferrolearn/__init__.py:5-11` re-exports all five; `test_check_estimator.py:26-30`
  runs them through `parametrize_with_checks`. The 542-passing pytest exercises the
  consumer surface.
- AC-SUBSTRATE (REQ-SUBSTRATE): `classifiers.rs` head shows
  `use crate::conversions::*` + `use numpy::{PyArray1, PyArray2, PyReadonlyArray1,
  PyReadonlyArray2}` — the wrong substrate per R-SUBSTRATE-1 (destination
  `ferray::numpy_interop`/`ferray-core`). ferray exposes no `numpy_interop` bridge
  consumable here (R-SUBSTRATE-5). Owned by `conversions.md` REQ-FERRAY #2027.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-LOGREG-API-CONFORM (fit/predict/predict_proba + classes_/coef_/intercept_, default path) | SHIPPED | impl `RsLogisticRegression::fit`/`predict`/`predict_proba` + getters `classes_`/`coef_`/`intercept_` in `classifiers.rs` (over `ferrolearn_linear::FittedLogisticRegression<f64>` via `fitted.coefficients()`/`fitted.intercept()`/`fitted.classes()`), wrapped by `LogisticRegression` in `_classifiers.py` which sets `coef_.reshape(1,-1)`, `intercept_ = np.array([...])`, `classes_` (via `_encode_labels`), `n_features_in_` (via `self._validate_data`) and inherits `score` from `ClassifierMixin` — mirroring `sklearn/linear_model/_logistic.py:810` (class) + `:1395` (predict_proba). Non-test consumer: `_classifiers.py::LogisticRegression` + `ferrolearn/__init__.py:5-11` re-export; external users. Verification (model B): `cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` → 542 passed, 0 failed (`test_check_estimator.py:26` + `test_cross_val_score.py`). Live default-path oracle MATCHES: `classes_=[0,1]`, `coef_=[[0.4366,0.958]]`, `intercept_=[-3.3158]`, `pred=[0,0,1,1,0,1]`, `proba[0]=[0.964966,0.035034]`. |
| REQ-LOGREG-C-ABI (C keyword-only) | SHIPPED | `_classifiers.py::LogisticRegression.__init__(self, *, C=1.0, max_iter=1000, tol=1e-4, fit_intercept=True)` makes `C` keyword-only, MATCHING sklearn `_logistic.py:1129` (`__init__(self, penalty='l2', *, ..., C=1.0, ...)` — `C` AFTER the `*`). Marshalled to `RsLogisticRegression::new` via `#[pyo3(signature = (c=1.0, ...))]` + `with_c`. Live: both `inspect.signature(...).parameters['C'].kind` → `KEYWORD_ONLY`. Non-test consumer: `_classifiers.py::LogisticRegression`. |
| REQ-LOGREG-MAXITER-DEFAULT (max_iter default 100) — **HEADLINE 3/3** | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 default divergence; single-wrapper-fixable). sklearn `max_iter=100` (`_logistic.py:1129`). ferrolearn `_classifiers.py::LogisticRegression.__init__` defaults `max_iter=1000` (live-confirmed). Single-line wrapper fix: change the default to `100`. RISK: ferrolearn's LBFGS may need >100 iters to converge to the downstream ~1e-8 parity (downstream `ferrolearn-linear` REQ-1 verifies coef at convergence; the ~1e-3 gap at default tol is the LBFGS stopping bound) — if `100` breaks the value-parity green guards that is a SEPARATE downstream convergence blocker; the ABI match is still `100`. |
| REQ-LOGREG-NITER (n_iter_ = real LBFGS count) | NOT-STARTED | open prereq blocker #450 (`ferrolearn-linear` REQ-9..20: `n_iter_` not tracked). sklearn exposes the ACTUAL LBFGS count (`_logistic.py:1276`/`:1376`); oracle `LogisticRegression().fit(X,y).n_iter_` → `[11]`. ferrolearn `_classifiers.py::LogisticRegression.fit` sets `self.n_iter_ = self.max_iter` — a hardcoded FAKE (live: `1000`), because `_RsLogisticRegression` exposes NO `n_iter_` getter and `FittedLogisticRegression` discards the count. The binding cannot expose an attr the library does not compute — owned downstream (R-DEV-1/-3). |
| REQ-LOGREG-PARAMS (penalty/dual/solver/class_weight/random_state/intercept_scaling/warm_start/l1_ratio/n_jobs) | NOT-STARTED | open prereq blockers #442 (penalty l1/elasticnet/none) + #443 (solver variants) + #444 (multi_class=ovr) + #445 (class_weight) + #446 (dual) + #447 (intercept_scaling) + #448 (l1_ratio) + #449 (warm_start) + #452 (random_state/n_jobs). sklearn `_logistic.py:1129`. ferrolearn `_classifiers.py::LogisticRegression.__init__` exposes `C`/`max_iter`/`tol`/`fit_intercept` only; `RsLogisticRegression::new` likewise. The default `penalty='l2'`/`solver='lbfgs'` behavior MATCHES — only the param surface + non-default paths are missing — owned downstream. |
| REQ-LOGREG-VALUE-PARITY (coef_/intercept_/predict_proba array parity, default path) | SHIPPED | on the DEFAULT path. `_classifiers.py::LogisticRegression.fit`/`predict_proba` marshal from the `RsLogisticRegression` getters (over `FittedLogisticRegression`), and downstream `ferrolearn-linear` REQ-1/2/4 are critic-verified to MATCH the live sklearn `LogisticRegression` oracle to ~1e-8 at convergence (binary LBFGS + multinomial softmax; coef/intercept/predict_proba). Live (R-CHAR-3): ferrolearn `LogisticRegression().fit(X,y)` `coef_=[[0.4366,0.958]]`, `intercept_=[-3.3157]`, `proba[0]=[0.964966,0.035034]` agree with the oracle. Non-test consumer: `_classifiers.py::LogisticRegression` + re-export. (Binary `decision_function` `(n,)`-shape ABI #454.) |
| REQ-DT-API-CONFORM (fit/predict/predict_proba + classes_, default gini path) | SHIPPED | impl `RsDecisionTreeClassifier::fit`/`predict`/`predict_proba` + getter `classes_` in `classifiers.rs` (over `ferrolearn_tree::FittedDecisionTreeClassifier<f64>`), wrapped by `DecisionTreeClassifier` in `_classifiers.py` which marshals `max_depth`/`min_samples_split`/`min_samples_leaf`, sets `classes_` + `n_features_in_`, inherits `score` from `ClassifierMixin` — mirroring `sklearn/tree/_classes.py:698` (class) + `:1017` (predict_proba). Non-test consumer: `_classifiers.py::DecisionTreeClassifier` + `ferrolearn/__init__.py:5-11` re-export. Verification (model B): pytest → 542 passed (`test_check_estimator.py:27`). Live default-path oracle MATCHES: `classes_=[0,1]`, `pred=[0,0,1,1,0,1]`. |
| REQ-DT-CTOR-ABI (all params keyword-only) | SHIPPED | `_classifiers.py::DecisionTreeClassifier.__init__(self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1)` makes ALL params keyword-only, MATCHING sklearn `_classes.py:945` (`__init__(self, *, criterion='gini', ...)` — the `*` is FIRST). Live: both signatures' non-self params are all `KEYWORD_ONLY`. Non-test consumer: `_classifiers.py::DecisionTreeClassifier`. |
| REQ-DT-FEATURE-IMPORTANCES (feature_importances_ surfaced) | NOT-STARTED | blocker issue to be filed by critic (binding-level: the Rust getter exists but the wrapper does not read it). sklearn exposes `feature_importances_` (`_classes.py`; oracle `[0.0,1.0]`). The Rust `RsDecisionTreeClassifier::feature_importances_` getter EXISTS (over `FittedDecisionTreeClassifier::feature_importances`, downstream `ferrolearn-tree` REQ-5 SHIPPED), but `_classifiers.py::DecisionTreeClassifier` never reads it — live: `ferrolearn.DecisionTreeClassifier().fit(X,y).feature_importances_` → `AttributeError`. Single-wrapper fix: the wrapper must set `self.feature_importances_` from `self._rs.feature_importances_`. |
| REQ-DT-PARAMS (criterion/splitter/max_features/random_state/max_leaf_nodes/min_weight_fraction_leaf/min_impurity_decrease/class_weight/ccp_alpha) | NOT-STARTED | open prereq blockers #665 (`max_features`) + #670 (`random_state`/`splitter='random'`) plus the binding does not expose criterion/min_impurity_decrease/ccp_alpha/max_leaf_nodes/class_weight (library SHIPPED downstream `ferrolearn-tree` REQ-1/3/7 but `RsDecisionTreeClassifier::new` takes only `max_depth`/`min_samples_split`/`min_samples_leaf`). sklearn `_classes.py:945`. The default gini behavior MATCHES — only the param surface + non-default paths are missing. |
| REQ-DT-VALUE-PARITY (pred/predict_proba array parity, default gini) | SHIPPED | on the DEFAULT path. `_classifiers.py::DecisionTreeClassifier` marshals `predict`/`predict_proba` from the `RsDecisionTreeClassifier` methods (over `FittedDecisionTreeClassifier`), and downstream `ferrolearn-tree` REQ-1/2/6 are critic-verified to MATCH the live sklearn `DecisionTreeClassifier` oracle (CART best-split + per-leaf class frequencies). Live (R-CHAR-3): ferrolearn `DecisionTreeClassifier().fit(X,y)` `pred=[0,0,1,1,0,1]` equals the oracle. Non-test consumer: `_classifiers.py::DecisionTreeClassifier` + re-export. (Multi-output 2-D `y` #668.) |
| REQ-RF-API-CONFORM (fit/predict + classes_, default soft-vote path) | SHIPPED | impl `RsRandomForestClassifier::fit`/`predict` + getter `classes_` in `classifiers.rs` (over `ferrolearn_tree::FittedRandomForestClassifier<f64>`), wrapped by `RandomForestClassifier` in `_classifiers.py` which marshals `n_estimators`/`max_depth`/`min_samples_split`/`min_samples_leaf`/`random_state`, sets `classes_` + `n_features_in_`, inherits `score` — mirroring `sklearn/ensemble/_forest.py:1170` (class) + `:883` (soft-vote predict). Non-test consumer: `_classifiers.py::RandomForestClassifier` + `ferrolearn/__init__.py:5-11` re-export. Verification (model B): pytest → 542 passed (`test_check_estimator.py:28` runs `RandomForestClassifier(n_estimators=5, random_state=42)`). Live default-path oracle MATCHES on this dataset: `classes_=[0,1]`, `pred=[0,0,1,1,0,1]`. |
| REQ-RF-NESTIMATORS-POSITIONAL (n_estimators positional ABI) — **HEADLINE 1/3** | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 constructor ABI; single-wrapper-fixable). sklearn `__init__(self, n_estimators=100, *, ...)` (`_forest.py:1494`) makes `n_estimators` positional-or-keyword — `RandomForestClassifier(10).n_estimators` → `10`. ferrolearn `_classifiers.py::RandomForestClassifier.__init__(self, *, n_estimators=100, ...)` makes it keyword-only — live: `ferrolearn.RandomForestClassifier(10)` → `TypeError: __init__() takes 1 positional argument but 2 were given`. Single-line Python-wrapper fix: move `n_estimators` before the `*`. |
| REQ-RF-PREDICT-PROBA (predict_proba surfaced) | SHIPPED (#2050) | FIXED — `RsRandomForestClassifier::predict_proba` (`classifiers.rs`) binds `FittedRandomForestClassifier::predict_proba` (`random_forest.rs:432`, mean of per-tree probas, `_forest.py:922`), and `_classifiers.py::RandomForestClassifier.predict_proba` surfaces it. Structural contract verified (exact value is the numpy-MT bootstrap RNG boundary #673, not pinned): `proba.shape == (n, n_classes)`, rows sum to 1.0, `predict == classes_[argmax(predict_proba)]` (`_forest.py:883`). Guard `tests/divergence_classifiers.py::test_red_rf_predict_proba_surfaced`. |
| REQ-RF-FEATURE-IMPORTANCES (feature_importances_ surfaced) | NOT-STARTED | blocker issue to be filed by critic (binding-level: the Rust getter exists but the wrapper does not read it). sklearn exposes `feature_importances_`. The Rust `RsRandomForestClassifier::feature_importances_` getter EXISTS (over `FittedRandomForestClassifier::feature_importances`, downstream `ferrolearn-tree` REQ-6 SHIPPED), but `_classifiers.py::RandomForestClassifier` never reads it — live: `AttributeError`. Single-wrapper fix: set `self.feature_importances_` from `self._rs.feature_importances_`. |
| REQ-RF-PARAMS (criterion/max_features/bootstrap/oob_score/max_samples/class_weight/n_jobs/warm_start/ccp_alpha/max_leaf_nodes/min_impurity_decrease) | NOT-STARTED | open prereq blockers #671 (tree-param passthrough / regressor criterion) + #672 (bootstrap/max_samples) + #675 (oob_score) + #676 (class_weight). sklearn `_forest.py:1494`. ferrolearn `_classifiers.py::RandomForestClassifier.__init__` exposes `n_estimators`/`max_depth`/`min_samples_split`/`min_samples_leaf`/`random_state` only; `RsRandomForestClassifier::new` likewise. The default `max_features='sqrt'`/`criterion='gini'`/`bootstrap=True` behavior MATCHES (downstream REQ-1) — only the param surface + non-default paths are missing — owned downstream. |
| REQ-RF-VALUE-PARITY (pred/predict_proba array parity at random_state) | SHIPPED | on the DETERMINISTIC contract. `_classifiers.py::RandomForestClassifier.predict` marshals from `RsRandomForestClassifier::predict` (over `FittedRandomForestClassifier`), and downstream `ferrolearn-tree` REQ-4/5/9 are critic-verified for soft-vote `classes_[argmax(predict_proba)]`, proba-mean (rows sum to 1), and seeded reproducibility. Live (R-CHAR-3): ferrolearn `RandomForestClassifier(random_state=0).fit(X,y)` `pred=[0,0,1,1,0,1]` equals the oracle on this dataset. Non-test consumer: `_classifiers.py::RandomForestClassifier` + re-export. (Exact numpy-MT bootstrap/feature-subsample parity is the DOCUMENTED RNG boundary #673.) |
| REQ-KNN-API-CONFORM (fit/predict + classes_, default uniform path) | SHIPPED | impl `RsKNeighborsClassifier::fit`/`predict` + getter `classes_` in `classifiers.rs` (over `ferrolearn_neighbors::FittedKNeighborsClassifier<f64>`), wrapped by `KNeighborsClassifier` in `_classifiers.py` which marshals `n_neighbors`, sets `classes_` + `n_features_in_`, inherits `score` — mirroring `sklearn/neighbors/_classification.py:39` (class) + `:240` (predict). Non-test consumer: `_classifiers.py::KNeighborsClassifier` + `ferrolearn/__init__.py:5-11` re-export. Verification (model B): pytest → 542 passed (`test_check_estimator.py:29` runs `KNeighborsClassifier(n_neighbors=3)`). Live: `classes_=[0,1]`; default-path uniform-vote predict + smallest-label tie-break is SHIPPED downstream (`ferrolearn-neighbors` REQ-1). |
| REQ-KNN-NNEIGHBORS-POSITIONAL (n_neighbors positional ABI) — **HEADLINE 2/3** | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 constructor ABI; single-wrapper-fixable). sklearn `__init__(self, n_neighbors=5, *, ...)` (`_classification.py:193`) makes `n_neighbors` positional-or-keyword — `KNeighborsClassifier(3).n_neighbors` → `3`. ferrolearn `_classifiers.py::KNeighborsClassifier.__init__(self, *, n_neighbors=5)` makes it keyword-only — live: `ferrolearn.KNeighborsClassifier(3)` → `TypeError: __init__() takes 1 positional argument but 2 were given`. Single-line Python-wrapper fix: move `n_neighbors` before the `*`. |
| REQ-KNN-PREDICT-PROBA (predict_proba surfaced) | NOT-STARTED | open prereq blocker #877 (`ferrolearn-neighbors` REQ-2/REQ-11: PyO3 surface under-exposed). sklearn KNN exposes `predict_proba` `(n_samples, n_classes)` (`_classification.py:307`; oracle shape `(6,2)`). `_classifiers.py::KNeighborsClassifier` has NO `predict_proba` method and `RsKNeighborsClassifier` exposes NO `predict_proba` getter — live: `hasattr(ferrolearn.KNeighborsClassifier, 'predict_proba')` → `False`. The library `FittedKNeighborsClassifier::predict_proba` is value-correct but has NO non-test consumer — owned downstream. |
| REQ-KNN-PARAMS (weights/algorithm/leaf_size/p/metric/metric_params/n_jobs) | NOT-STARTED | open prereq blockers #876 (`weights`/`metric`/`p` callable variants) + #877 (PyO3 surface — `weights`/`algorithm` not bound). sklearn `_classification.py:193`. ferrolearn `_classifiers.py::KNeighborsClassifier.__init__` exposes `n_neighbors` only; `RsKNeighborsClassifier::new` likewise. The default `weights='uniform'`/`metric='minkowski'`/`p=2` (Euclidean) behavior MATCHES — only the param surface + non-default paths are missing — owned downstream. |
| REQ-KNN-VALUE-PARITY (pred array parity incl. tie-break) | SHIPPED | on the DEFAULT uniform path. `_classifiers.py::KNeighborsClassifier.predict` marshals from `RsKNeighborsClassifier::predict` (over `FittedKNeighborsClassifier`), and downstream `ferrolearn-neighbors` REQ-1 is critic-verified for the uniform weighted-vote `np.argmax` smallest-label tie-break (`_classification.py:240-305`). Non-test consumer: `_classifiers.py::KNeighborsClassifier` + re-export. (Live: ferrolearn `KNeighborsClassifier(n_neighbors=3).predict(X)=[0,0,1,1,0,1]` vs oracle `[0,0,0,1,0,1]` on this dataset — a value/tie boundary; distance-weighting/2-D-query/non-Euclidean divergences owned downstream #876.) |
| REQ-GNB-API-CONFORM (fit/predict/predict_proba + classes_, default path) | SHIPPED | impl `RsGaussianNB::fit`/`predict`/`predict_proba` + getter `classes_` in `classifiers.rs` (over `ferrolearn_bayes::FittedGaussianNB<f64>`), wrapped by `GaussianNB` in `_classifiers.py` which marshals `var_smoothing`, sets `classes_` + `n_features_in_`, inherits `score` — mirroring `sklearn/naive_bayes.py:147` (class) + `:128` (predict_proba). Non-test consumer: `_classifiers.py::GaussianNB` + `ferrolearn/__init__.py:5-11` re-export. Verification (model B): pytest → 542 passed (`test_check_estimator.py:30`). Live default-path oracle MATCHES element-wise: `classes_=[0,1]`, `pred=[0,0,1,1,0,1]`, `proba[0]=[0.993664,0.006336]`. |
| REQ-GNB-CTOR-ABI (all params keyword-only) | SHIPPED | `_classifiers.py::GaussianNB.__init__(self, *, var_smoothing=1e-9)` makes ALL params keyword-only, MATCHING sklearn `naive_bayes.py:234` (`__init__(self, *, priors=None, var_smoothing=1e-9)` — the `*` is FIRST). Marshalled to `RsGaussianNB::new` via `#[pyo3(signature = (var_smoothing=1e-9))]` + `with_var_smoothing`. Live: both fully `KEYWORD_ONLY`. Non-test consumer: `_classifiers.py::GaussianNB`. |
| REQ-GNB-PARAMS (priors) | NOT-STARTED | open prereq blocker #897 (`ferrolearn-bayes` REQ-8: PyO3 surface — no `priors` kwarg). sklearn `naive_bayes.py:234`. The library supports `priors` validation (downstream `ferrolearn-bayes` REQ-2 SHIPPED, `with_class_prior`), but `_classifiers.py::GaussianNB.__init__` + `RsGaussianNB::new` expose `var_smoothing` only — the binding does not expose the `priors` kwarg. The default `priors=None` (data-derived) behavior MATCHES — only the param surface is missing — owned downstream. |
| REQ-GNB-VALUE-PARITY (pred/predict_proba array parity, default path) | SHIPPED | on the DEFAULT path. `_classifiers.py::GaussianNB.fit`/`predict_proba` marshal from the `RsGaussianNB` methods (over `FittedGaussianNB`), and downstream `ferrolearn-bayes` REQ-1/3/4 are critic-verified to MATCH the live sklearn `GaussianNB` oracle to ~1e-9 (`epsilon_` global var no floor, `_joint_log_likelihood`, predict, predict_proba). Live (R-CHAR-3): ferrolearn `GaussianNB().fit(X,y)` `pred=[0,0,1,1,0,1]`, `proba[0]=[0.993664,0.006336]` equal the oracle. Non-test consumer: `_classifiers.py::GaussianNB` + re-export. |
| REQ-CONSUMER (binding IS the public API) | SHIPPED | the binding boundary types ARE the public API (R-DEFER-1/S5: boundary estimator types ARE the public surface; grandfathered existing pub API). Non-test production consumers: `_classifiers.py::{LogisticRegression,DecisionTreeClassifier,RandomForestClassifier,KNeighborsClassifier,GaussianNB}` each construct their `_Rs*` class and call `fit`/`predict`(/`predict_proba`) + read the getters (`grep -n "_RsLogisticRegression\|_RsDecisionTreeClassifier\|_RsRandomForestClassifier\|_RsKNeighborsClassifier\|_RsGaussianNB" python/ferrolearn/_classifiers.py`); `ferrolearn/__init__.py:5-11` re-exports all five; `test_check_estimator.py:26-30` runs them through `parametrize_with_checks` + external users. The label round-trip (`_encode_labels`/`_decode_labels`) is covered by `conversions.md` REQ-LABEL-MARSHAL (SHIPPED). Verification (model B): pytest → 542 passed. |
| REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | open prereq blocker = `conversions.md` REQ-FERRAY #2027. `classifiers.rs` marshals via `use crate::conversions::*` + `use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2}` (rust-numpy) and the conversions produce `ndarray::Array{1,2}` — the WRONG substrate per R-SUBSTRATE-1 (destination `ferray::numpy_interop` + `ferray-core`). ferray exposes no PyO3 numpy-interop bridge yet (R-SUBSTRATE-5). Owned by the conversions unit, surfaced here. |

## Architecture

`classifiers.rs` holds five `#[pyclass]` structs, each a THIN shim over a fitted
library type and owning ZERO classification math. Every struct stores the
constructor params + an `Option<Fitted...>`, returns `PyRuntimeError("not
fitted")` from getters/predict before `fit`, and maps `FerroError` →
`PyValueError`. The Python wrappers in `_classifiers.py` add the sklearn estimator
contract on top: `_validate_data` (sets `n_features_in_`), `_check_classification_target`
(rejects continuous `y`), `_encode_labels`/`_decode_labels` (the LabelEncoder
round-trip — `np.unique(y)` → `0..n_classes-1` int64 before the Rust `fit`,
`classes_[y_encoded]` after `predict`, so arbitrary labels — negative, string,
non-contiguous — round-trip; `classes_` is set on the PYTHON side, covered by
`conversions.md` REQ-LABEL-MARSHAL), the `_ClassifierPickleMixin` (re-fit on
unpickle from stored training data), and `score` (from `ClassifierMixin`).

- **`RsLogisticRegression`** wraps `Option<FittedLogisticRegression<f64>>` +
  `c`/`max_iter`/`tol`/`fit_intercept`. `new`
  (`#[pyo3(signature = (c=1.0, max_iter=1000, tol=1e-4, fit_intercept=true))]`)
  stores the params; `fit` runs `LogisticRegression::<f64>::new().with_c(..)
  .with_max_iter(..).with_tol(..).with_fit_intercept(..)`; `predict`/
  `predict_proba` and the `classes_`/`coef_`/`intercept_` getters delegate to
  `FittedLogisticRegression`. The wrapper `LogisticRegression.fit` then sets
  `coef_.reshape(1,-1)`, `intercept_ = np.array([float(...)])` (matching sklearn's
  `(1, n_features)`/`(1,)` binary shapes, R-DEV-3) and `n_iter_ = self.max_iter`
  (the FAKE — REQ-LOGREG-NITER NOT-STARTED).
- **`RsDecisionTreeClassifier`** wraps `Option<FittedDecisionTreeClassifier<f64>>`
  + `max_depth`/`min_samples_split`/`min_samples_leaf`. Exposes `fit`/`predict`/
  `predict_proba` + getters `classes_` AND `feature_importances_`; the Python
  wrapper surfaces only `classes_` (REQ-DT-FEATURE-IMPORTANCES NOT-STARTED — the
  getter is unread).
- **`RsRandomForestClassifier`** wraps `Option<FittedRandomForestClassifier<f64>>`
  + `n_estimators`/`max_depth`/`min_samples_split`/`min_samples_leaf`/
  `random_state` (the last applied via `with_random_state` only when `Some`).
  Exposes `fit`/`predict` + getters `classes_`/`feature_importances_` — NO
  `predict_proba` (REQ-RF-PREDICT-PROBA NOT-STARTED); the wrapper reads neither
  `feature_importances_` (REQ-RF-FEATURE-IMPORTANCES NOT-STARTED) nor surfaces
  `predict_proba`, and makes `n_estimators` keyword-only
  (REQ-RF-NESTIMATORS-POSITIONAL, HEADLINE 1/3).
- **`RsKNeighborsClassifier`** wraps `Option<FittedKNeighborsClassifier<f64>>` +
  `n_neighbors`. Exposes `fit`/`predict` + getter `classes_` — NO `predict_proba`
  (REQ-KNN-PREDICT-PROBA NOT-STARTED); the wrapper makes `n_neighbors`
  keyword-only (REQ-KNN-NNEIGHBORS-POSITIONAL, HEADLINE 2/3).
- **`RsGaussianNB`** wraps `Option<FittedGaussianNB<f64>>` + `var_smoothing`.
  Exposes `fit`/`predict`/`predict_proba` + getter `classes_` — the fullest
  surface after LogReg/DT; missing only the `priors` kwarg (REQ-GNB-PARAMS
  NOT-STARTED). `__init__` ABI fully matches sklearn (REQ-GNB-CTOR-ABI SHIPPED).

Array/label marshalling for all five is via `crate::conversions::*`
(`numpy2_to_ndarray`/`numpy1_to_ndarray_usize`/`ndarray1_usize_to_numpy`/
`ndarray1_to_numpy`/`ndarray2_to_numpy`) over rust-numpy `PyReadonly*`/`PyArray*`
and `ndarray::Array{1,2}` — the WRONG substrate per R-SUBSTRATE-1 (destination
`ferray::numpy_interop`/`ferray-core`), owned by `conversions.md` REQ-FERRAY
#2027. The `i64`↔`usize` label coercion is SAFE only because the Python wrappers
establish the non-negative-contiguous-int64 contract via `_encode_labels`
(`conversions.md` REQ-LABEL-MARSHAL).

## Verification

Verification model B (pytest vs `import sklearn` 1.5.2). Run from the binding
crate; rebuild on any Rust change first.

```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q
cargo clippy -p ferrolearn-python --all-targets -- -D warnings
```

As of baseline `ff4b37171` the pytest gauntlet is GREEN: **542 passed**. All five
classifiers are exercised by `tests/test_check_estimator.py:26-30`
(`parametrize_with_checks`) and `tests/test_cross_val_score.py` (`cross_val_score`)
— this pins every `*-API-CONFORM`, `*-VALUE-PARITY` (default path), the matching
constructor ABIs (REQ-LOGREG-C-ABI / REQ-DT-CTOR-ABI / REQ-GNB-CTOR-ABI), and
REQ-CONSUMER.

The downstream value parity each SHIPPED `*-VALUE-PARITY` relies on is verified by
the owning crate's gauntlet + live-sklearn `#[test]` pins:
`cargo test -p ferrolearn-linear` (LogisticRegression REQ-1/2/4),
`cargo test -p ferrolearn-tree` (DecisionTreeClassifier REQ-1/2/6,
RandomForestClassifier REQ-4/5/9), `cargo test -p ferrolearn-neighbors`
(KNeighborsClassifier REQ-1), `cargo test -p ferrolearn-bayes` (GaussianNB
REQ-1/3/4).

The NOT-STARTED REQs are pinned by the critic as FAILING pytests under
`ferrolearn-python/tests/divergence_classifiers.py`, each FAILING until the fix
lands (R-DEFER-3):
- The three headline ABIs (`RandomForestClassifier(10).n_estimators == 10`,
  `KNeighborsClassifier(3).n_neighbors == 3`,
  `LogisticRegression().__init__` `max_iter` default `== 100`) — single-line
  wrapper fixes.
- The binding-level surface gaps (`DecisionTreeClassifier().feature_importances_`,
  `RandomForestClassifier().feature_importances_`,
  `hasattr(RandomForestClassifier, 'predict_proba')`).
- The downstream-owned divergences reference the cited library blockers
  (LogReg `n_iter_` #450 + params #442-#452; DT params #665/#670; RF params
  #671/#672/#675/#676; KNN `predict_proba`/params #876/#877; GNB `priors` #897;
  substrate `conversions.md` REQ-FERRAY #2027).
