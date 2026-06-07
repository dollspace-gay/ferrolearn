# ferrolearn-python extras — the ~40-estimator sklearn binding shim

<!--
tier: 3-component
status: draft
baseline-commit: 8aa19d6c1c1933c9e9ba449dc40f9dccd9dafab1
upstream-paths:
  - sklearn/ensemble/_forest.py            # RandomForestRegressor / ExtraTreesRegressor / ExtraTreesClassifier
  - sklearn/ensemble/_gb.py                # GradientBoosting{Regressor,Classifier}
  - sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py  # HistGradientBoosting{Regressor,Classifier}
  - sklearn/ensemble/_weight_boosting.py   # AdaBoostClassifier
  - sklearn/ensemble/_bagging.py           # BaggingClassifier
  - sklearn/linear_model/_bayes.py         # BayesianRidge / ARDRegression
  - sklearn/linear_model/_huber.py         # HuberRegressor
  - sklearn/linear_model/_quantile.py      # QuantileRegressor
  - sklearn/linear_model/_omp.py           # OrthogonalMatchingPursuit
  - sklearn/linear_model/_least_angle.py   # Lars / LassoLars
  - sklearn/linear_model/_ridge.py         # RidgeClassifier
  - sklearn/svm/_classes.py                # LinearSVC
  - sklearn/discriminant_analysis.py       # QuadraticDiscriminantAnalysis
  - sklearn/naive_bayes.py                 # MultinomialNB / BernoulliNB / ComplementNB
  - sklearn/neighbors/_regression.py       # KNeighborsRegressor
  - sklearn/neighbors/_nearest_centroid.py # NearestCentroid
  - sklearn/kernel_ridge.py                # KernelRidge
  - sklearn/cluster/_kmeans.py             # MiniBatchKMeans
  - sklearn/cluster/_dbscan.py             # DBSCAN
  - sklearn/cluster/_agglomerative.py      # AgglomerativeClustering
  - sklearn/cluster/_birch.py              # Birch
  - sklearn/mixture/_gaussian_mixture.py   # GaussianMixture
  - sklearn/decomposition/_incremental_pca.py  # IncrementalPCA
  - sklearn/decomposition/_truncated_svd.py    # TruncatedSVD
  - sklearn/decomposition/_fastica.py          # FastICA
  - sklearn/decomposition/_nmf.py              # NMF
  - sklearn/decomposition/_kernel_pca.py       # KernelPCA
  - sklearn/decomposition/_sparse_pca.py       # SparsePCA / MiniBatchSparsePCA
  - sklearn/decomposition/_factor_analysis.py  # FactorAnalysis
  - sklearn/preprocessing/_data.py             # MinMaxScaler / MaxAbsScaler / RobustScaler / PowerTransformer
  - sklearn/kernel_approximation.py            # Nystroem / RBFSampler
-->

## Summary

`ferrolearn-python/src/extras.rs` is the LARGEST PyO3 marshalling shim in the
binding crate: it binds **~40 estimators** (the "extras" surface — everything
beyond the 12 originally-bound estimators) across `ferrolearn_linear`,
`ferrolearn_tree`, `ferrolearn_neighbors`, `ferrolearn_bayes`,
`ferrolearn_cluster`, `ferrolearn_decomp`, `ferrolearn_preprocess`, and
`ferrolearn_kernel`. Three declarative macros — `py_regressor!`,
`py_classifier!`, `py_transformer!` — generate the common
`#[pyclass]`/`fit`/`predict`/`transform` shells; estimators with extra state
(ensembles carrying `random_state`, clusterers exposing `labels_`) are
hand-written `#[pyclass]` blocks. `ferrolearn-python/python/ferrolearn/_extras.py`
wraps each `_Rs*` pyclass as a sklearn mixin subclass
(`_RegressorWrapper`/`_ClassifierWrapper`/`_ClusterWrapper`/`_TransformerWrapper`
over `RegressorMixin`/`ClassifierMixin`/`ClusterMixin`/`TransformerMixin` +
`BaseEstimator`), so `import ferrolearn` mirrors the corresponding
`from sklearn.<module> import <Est>`. The `_ClassifierWrapper` performs a
`LabelEncoder`-style round-trip (`_encode` → `np.unique` + `np.searchsorted`,
decode via `self.classes_[y_enc]`) so arbitrary label dtypes map onto the Rust
`usize`-label classifier core.

This unit is a **thin marshalling shim only**: constructor ABI, method/attribute
surface, label encoding, and numpy↔ndarray array coercion across the
Python↔Rust boundary. The estimator *correctness* (the math, the full
hyperparameter surface, the fitted attributes ferrolearn omits) lives DOWN in
the eight library crates, each audited by its own `//!` REQ status table. Per
the goal statement (§"Semantic/numerical bugs are fixed DOWN in the library
crate"), value/method/param divergences are owned by those pre-existing audited
crates and referenced generically here ("owned downstream by `<crate>`") rather
than re-filed; this doc owns only the binding-level surface and the three
binding/ABI-level fixable divergences.

**Verification model: B (pytest vs sklearn 1.5.2).** Per goal.md §"The
verification model (B)", this unit is verified by
`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q` comparing
`import ferrolearn` against the installed `import sklearn` 1.5.2 oracle, plus the
live-sklearn oracle for the constructor-ABI boundary. Pins land under
`ferrolearn-python/tests/divergence_<category>.py`; rebuild
(`cd ferrolearn-python && maturin develop`) before pytest sees a Rust change.

Divergence classes (three are binding-level FIXABLE, headlined; the rest are
owned downstream):

1. **api-conformance (SHIPPED, per category)** — on the DEFAULT parameter path,
   the regressors expose `fit`/`predict`, classifiers `fit`/`predict` (+ label
   round-trip), clusterers `fit` + `labels_`, transformers `fit`/`transform`,
   with the right marshalled shapes/types and (on the deterministic default
   path) values matching the live sklearn oracle to the downstream-verified
   tolerance.
2. **ctor-abi-positional (NOT-STARTED — HEADLINE 1)** — 16 wrappers make the
   primary hyperparameter keyword-only while sklearn makes it
   positional-or-keyword (R-DEV-2). Single-wrapper-class-fixable per estimator.
3. **module-root `#![allow(non_snake_case)]` (NOT-STARTED — HEADLINE 2)** —
   `extras.rs:5` carries a crate-root-level `#![allow]` (R-CODE-3/R-APG-1).
4. **phase-framing (NOT-STARTED — HEADLINE 3)** — `extras.rs:1` `//!` and
   `_extras.py:1` docstring use "Phase 2 binding expansion" deferral framing
   (R-DEFER-4).
5. **decomp n_components default (NOT-STARTED, R-DEV-2)** — 5 decomp
   transformers default `n_components=2` vs sklearn `None`/`'warn'`.
6. **missing methods / params / off-default value parity / RNG parity
   (NOT-STARTED)** — owned downstream by the library crates.
7. **substrate (NOT-STARTED, R-SUBSTRATE-1)** — owned by `conversions.md` #2027.

## Upstream reference (sklearn 1.5.2, live oracle = installed sklearn 1.5.2)

Lines stable at tag 1.5.2 / commit 156ef14. The headline ABI divergence is the
**kind** of the primary hyperparameter. The following 16 sklearn classes make
their primary hyperparameter `POSITIONAL_OR_KEYWORD` (live-verified from `/tmp`,
R-CHAR-3), so `sklearn.X(val)` works while `ferrolearn.X(val)` raises
`TypeError`:

| sklearn class | upstream `file:line` | primary param | sklearn kind |
|---|---|---|---|
| `RandomForestRegressor` | `ensemble/_forest.py:1245` | `n_estimators=100` | POSITIONAL_OR_KEYWORD |
| `ExtraTreesRegressor` | `ensemble/_forest.py:1788` | `n_estimators=100` | POSITIONAL_OR_KEYWORD |
| `KNeighborsRegressor` | `neighbors/_regression.py:25` | `n_neighbors=5` | POSITIONAL_OR_KEYWORD |
| `RidgeClassifier` | `linear_model/_ridge.py:1228` | `alpha=1.0` | POSITIONAL_OR_KEYWORD |
| `KernelRidge` | `kernel_ridge.py:16` | `alpha=1.0` | POSITIONAL_OR_KEYWORD |
| `MiniBatchKMeans` | `cluster/_kmeans.py:1745` | `n_clusters=8` | POSITIONAL_OR_KEYWORD |
| `AgglomerativeClustering` | `cluster/_agglomerative.py:801` | `n_clusters=2` | POSITIONAL_OR_KEYWORD |
| `DBSCAN` | `cluster/_dbscan.py:165` | `eps=0.5` | POSITIONAL_OR_KEYWORD |
| `GaussianMixture` | `mixture/_gaussian_mixture.py:455` | `n_components=1` | POSITIONAL_OR_KEYWORD |
| `TruncatedSVD` | `decomposition/_truncated_svd.py:29` | `n_components=2` | POSITIONAL_OR_KEYWORD |
| `FastICA` | `decomposition/_fastica.py:330` | `n_components=None` | POSITIONAL_OR_KEYWORD |
| `NMF` | `decomposition/_nmf.py:1183` | `n_components='warn'` | POSITIONAL_OR_KEYWORD |
| `IncrementalPCA` | `decomposition/_incremental_pca.py:20` | `n_components=None` | POSITIONAL_OR_KEYWORD |
| `KernelPCA` | `decomposition/_kernel_pca.py:30` | `n_components=None` | POSITIONAL_OR_KEYWORD |
| `SparsePCA` | `decomposition/_sparse_pca.py:159` | `n_components=None` | POSITIONAL_OR_KEYWORD |
| `FactorAnalysis` | `decomposition/_factor_analysis.py:37` | `n_components=None` | POSITIONAL_OR_KEYWORD |

Estimators whose primary hyperparameter is `KEYWORD_ONLY` in sklearn too — the
ferrolearn wrapper MATCHES, no divergence (live-verified): `MultinomialNB.alpha`,
`BernoulliNB.alpha`, `ComplementNB.alpha`, `GradientBoosting{Regressor,Classifier}`
(first non-keyword-only param is `loss`/`n_estimators` → keyword-only),
`HistGradientBoosting{Regressor,Classifier}.max_iter`, `AdaBoostClassifier.n_estimators`,
`DecisionTreeRegressor.max_depth`, `ExtraTreeClassifier.max_depth`,
`Birch.threshold`. The four hand-written linear regressors
(`BayesianRidge`/`ARDRegression`/`HuberRegressor`/`QuantileRegressor`) and the
no-knob transformers (`MinMaxScaler`/`MaxAbsScaler`/`RobustScaler`/
`PowerTransformer`/`Nystroem`/`RBFSampler`) and `NearestCentroid`/`QDA`/
`LinearSVC` are not in the positional-fix set.

**`AdaBoostClassifier` / `BaggingClassifier` — NOT a clean positional divergence.**
sklearn `AdaBoostClassifier.__init__(self, estimator=None, *, n_estimators=50,
...)` (`ensemble/_weight_boosting.py:328`) and `BaggingClassifier.__init__(self,
estimator=None, n_estimators=10, ...)` (`ensemble/_bagging.py:592`) both put
`estimator` first; `n_estimators` is keyword-only (Ada) /
positional-or-keyword-but-second (Bagging). The ferrolearn wrappers expose
`n_estimators` first. Because sklearn's first positional slot is `estimator`
(which ferrolearn cannot accept — there is no base-estimator pluggability),
`Bagging(10)` would set `estimator=10` in sklearn, so the positional contract is
not comparable; these are excluded from the positional-ABI fix and folded into
the missing-params class (no `estimator` knob — owned downstream).

**Decomp `n_components` default divergence (R-DEV-2, live-verified):**

| transformer | sklearn default (`file:line`) | ferrolearn default |
|---|---|---|
| `IncrementalPCA` | `None` (`_incremental_pca.py:20`) | `2` |
| `FastICA` | `None` (`_fastica.py:330`) | `2` |
| `KernelPCA` | `None` (`_kernel_pca.py:30`) | `2` |
| `SparsePCA` | `None` (`_sparse_pca.py:159`) | `2` |
| `FactorAnalysis` | `None` (`_factor_analysis.py:37`) | `2` |
| `NMF` | `'warn'`→`None` (`_nmf.py:1183`) | `2` |
| `TruncatedSVD` | `2` (`_truncated_svd.py:29`) | `2` (MATCHES) |

Live oracle (installed sklearn 1.5.2, run from `/tmp`; R-CHAR-3 — values from
sklearn, NEVER from ferrolearn):

```
RandomForestRegressor.n_estimators -> POSITIONAL_OR_KEYWORD   (15 more identical, table above)
MultinomialNB.alpha / BernoulliNB.alpha / ComplementNB.alpha -> KEYWORD_ONLY   (ferrolearn matches)
AdaBoostClassifier.n_estimators -> KEYWORD_ONLY ; BaggingClassifier first param -> estimator
FastICA/KernelPCA/SparsePCA/FactorAnalysis/IncrementalPCA n_components default -> None ; NMF -> 'warn' ; TruncatedSVD -> 2
```

ferrolearn at baseline `8aa19d6` (live, model B build): `import ferrolearn` OK;
`ferrolearn.RandomForestRegressor(50)` → `TypeError:
RandomForestRegressor.__init__() takes 1 positional argument but 2 were given`
(the headline divergence); `ferrolearn.FastICA().n_components` → `2` (vs sklearn
`None`).

## Estimator catalog

40 estimators (route `parity_ops` lists 42 names; `ExtraTreesClassifier`
appears once, `BaggingClassifier`/`NearestCentroid` once — the Rust file defines
40 `#[pyclass]` blocks, all registered in `lib.rs:27-88`). "Pos-fix" = in the
16-estimator REQ-CTOR-ABI-POSITIONAL set.

| # | ferrolearn estimator | Rust `_Rs*` class (`extras.rs`) | underlying `ferrolearn_*` fitted type | exposed methods/getters | divergence class |
|---|---|---|---|---|---|
| 1 | `BayesianRidge` | `RsBayesianRidge` (macro) | `ferrolearn_linear::FittedBayesianRidge` | fit/predict | regressor (ABI matches; no `coef_`) |
| 2 | `ARDRegression` | `RsARDRegression` (macro) | `ferrolearn_linear::FittedARDRegression` | fit/predict | regressor |
| 3 | `HuberRegressor` | `RsHuberRegressor` (macro) | `ferrolearn_linear::FittedHuberRegressor` | fit/predict | regressor |
| 4 | `QuantileRegressor` | `RsQuantileRegressor` (macro) | `ferrolearn_linear::FittedQuantileRegressor` | fit/predict | regressor |
| 5 | `DecisionTreeRegressor` | `RsDecisionTreeRegressor` (macro) | `ferrolearn_tree::FittedDecisionTreeRegressor` | fit/predict | regressor (ABI matches) |
| 6 | `RandomForestRegressor` | `RsRandomForestRegressor` (hand) | `ferrolearn_tree::FittedRandomForestRegressor` | fit/predict | regressor + **Pos-fix** + RNG |
| 7 | `ExtraTreesRegressor` | `RsExtraTreesRegressor` (hand) | `ferrolearn_tree::FittedExtraTreesRegressor` | fit/predict | regressor + **Pos-fix** + RNG |
| 8 | `GradientBoostingRegressor` | `RsGradientBoostingRegressor` (hand) | `ferrolearn_tree::FittedGradientBoostingRegressor` | fit/predict | regressor (ABI matches) + RNG |
| 9 | `HistGradientBoostingRegressor` | `RsHistGradientBoostingRegressor` (hand) | `ferrolearn_tree::FittedHistGradientBoostingRegressor` | fit/predict | regressor (ABI matches) + RNG |
| 10 | `KNeighborsRegressor` | `RsKNeighborsRegressor` (hand, #2147) | `ferrolearn_neighbors::FittedKNeighborsRegressor` | fit/predict | regressor (full ctor surface SHIPPED; `weights`/`algorithm` wired) |
| 11 | `KernelRidge` | `RsKernelRidge` (macro) | `ferrolearn_kernel::FittedKernelRidge` | fit/predict | regressor + **Pos-fix** |
| 11b | `OrthogonalMatchingPursuit` | `RsOrthogonalMatchingPursuit` (hand, #2172) | `ferrolearn_linear::FittedOMP` | fit/predict/`coef_`/`intercept_` | regressor (full ctor surface SHIPPED; `precompute` accepted+ignored; coef_/intercept_ ~1e-12) |
| 11c | `Lars` | `RsLars` (hand, #2174) | `ferrolearn_linear::FittedLars` | fit/predict/`coef_`/`intercept_` | regressor (full ctor surface SHIPPED; `precompute`/`fit_path`/`eps`/`verbose`/`copy_X`/`random_state` accepted+ignored; `jitter` NotImpl; coef_/intercept_ ~1e-6) |
| 11d | `LassoLars` | `RsLassoLars` (hand, #2174) | `ferrolearn_linear::FittedLassoLars` | fit/predict/`coef_`/`intercept_` | regressor (full ctor surface SHIPPED; `alpha` positional-first; `positive`/`jitter` NotImpl; coef_/intercept_ ~1e-6) |
| 11e | `RANSACRegressor` | `RsRANSACRegressor` (hand, #2178) | `ferrolearn_linear::FittedRANSACRegressor<FittedLinearRegression>` | fit/predict/`coef_`/`intercept_`/`inlier_mask_` | regressor over default LinearRegression base (full ctor surface SHIPPED; `estimator`/`loss='squared_error'`/`stop_*`/`is_*_valid` NotImpl; coef_/inlier_mask_ match on well-separated data, RNG caveat #2118) |
| 12 | `RidgeClassifier` | `RsRidgeClassifier` (macro) | `ferrolearn_linear::FittedRidgeClassifier` | fit/predict | classifier + **Pos-fix** |
| 13 | `LinearSVC` | `RsLinearSVC` (macro) | `ferrolearn_linear::FittedLinearSVC` | fit/predict | classifier (ABI matches) |
| 14 | `QuadraticDiscriminantAnalysis` | `RsQDA` (macro) | `ferrolearn_linear::FittedQDA` | fit/predict | classifier (ABI matches) |
| 15 | `MultinomialNB` | `RsMultinomialNB` (macro) | `ferrolearn_bayes::FittedMultinomialNB` | fit/predict | classifier (ABI matches) |
| 16 | `BernoulliNB` | `RsBernoulliNB` (macro) | `ferrolearn_bayes::FittedBernoulliNB` | fit/predict | classifier (ABI matches) |
| 17 | `ComplementNB` | `RsComplementNB` (macro) | `ferrolearn_bayes::FittedComplementNB` | fit/predict | classifier (ABI matches) |
| 18 | `ExtraTreeClassifier` | `RsExtraTreeClassifier` (macro) | `ferrolearn_tree::FittedExtraTreeClassifier` | fit/predict | classifier (ABI matches) |
| 19 | `ExtraTreesClassifier` | `RsExtraTreesClassifier` (hand) | `ferrolearn_tree::FittedExtraTreesClassifier` | fit/predict | classifier (ABI matches; `n_estimators` kw-only both) + RNG |
| 20 | `AdaBoostClassifier` | `RsAdaBoostClassifier` (hand) | `ferrolearn_tree::FittedAdaBoostClassifier` | fit/predict | classifier (ABI matches) + RNG |
| 21 | `GradientBoostingClassifier` | `RsGradientBoostingClassifier` (hand) | `ferrolearn_tree::FittedGradientBoostingClassifier` | fit/predict | classifier (ABI matches) + RNG |
| 22 | `HistGradientBoostingClassifier` | `RsHistGradientBoostingClassifier` (hand) | `ferrolearn_tree::FittedHistGradientBoostingClassifier` | fit/predict | classifier (ABI matches) + RNG |
| 23 | `BaggingClassifier` | `RsBaggingClassifier` (hand) | `ferrolearn_tree::FittedBaggingClassifier` | fit/predict | classifier (no `estimator` knob — downstream) + RNG |
| 24 | `NearestCentroid` | `RsNearestCentroid` (macro, no params) | `ferrolearn_neighbors::FittedNearestCentroid` | fit/predict | classifier (no knobs) |
| 25 | `MiniBatchKMeans` | `RsMiniBatchKMeans` (hand) | `ferrolearn_cluster::FittedMiniBatchKMeans` | fit/predict/`labels_` | clusterer + **Pos-fix** + RNG |
| 26 | `DBSCAN` | `RsDBSCAN` (hand) | `ferrolearn_cluster::FittedDBSCAN` | ctor(`metric`/`p`)/fit(`sample_weight`)/`labels_`/`core_sample_indices_`/`components_` | clusterer + **Pos-fix** |
| 27 | `AgglomerativeClustering` | `RsAgglomerativeClustering` (hand) | `ferrolearn_cluster::FittedAgglomerativeClustering` | fit/`labels_` | clusterer + **Pos-fix** |
| 28 | `Birch` | `RsBirch` (hand) | `ferrolearn_cluster::FittedBirch` | fit/`labels_` | clusterer (ABI matches) |
| 28b | `FeatureAgglomeration` | `RsFeatureAgglomeration` (hand, #943) | `ferrolearn_cluster::FittedFeatureAgglomeration` | fit/transform/inverse_transform/`labels_`/`n_clusters_`/`children_`/`distances_`/`n_leaves_`/`n_connected_components_` | transformer+clusterer (full value parity SHIPPED; arbitrary-callable pooling/`distance_threshold` NotImpl #941) |
| 29 | `GaussianMixture` | `RsGaussianMixture` (hand) | `ferrolearn_cluster::FittedGaussianMixture` | fit/predict | mixture + **Pos-fix** + RNG |
| 30 | `IncrementalPCA` | `RsIncrementalPCA` (macro) | `ferrolearn_decomp::FittedIncrementalPCA` | fit/transform | transformer + **Pos-fix** + n_comp-default |
| 31 | `TruncatedSVD` | `RsTruncatedSVD` (macro) | `ferrolearn_decomp::FittedTruncatedSVD` | fit/transform | transformer + **Pos-fix** (default matches) |
| 32 | `FastICA` | `RsFastICA` (macro) | `ferrolearn_decomp::FittedFastICA` | fit/transform | transformer + **Pos-fix** + n_comp-default + RNG |
| 33 | `NMF` | `RsNMF` (macro) | `ferrolearn_decomp::FittedNMF` | fit/transform | transformer + **Pos-fix** + n_comp-default + RNG |
| 34 | `KernelPCA` | `RsKernelPCA` (macro) | `ferrolearn_decomp::FittedKernelPCA` | fit/transform | transformer + **Pos-fix** + n_comp-default |
| 35 | `SparsePCA` | `RsSparsePCA` (macro) | `ferrolearn_decomp::FittedSparsePCA` | fit/transform | transformer + **Pos-fix** + n_comp-default |
| 36 | `FactorAnalysis` | `RsFactorAnalysis` (macro) | `ferrolearn_decomp::FittedFactorAnalysis` | fit/transform | transformer + **Pos-fix** + n_comp-default |
| 37 | `MinMaxScaler` | `RsMinMaxScaler` (macro, no params) | `ferrolearn_preprocess::FittedMinMaxScaler` | fit/transform | transformer (no knobs) |
| 38 | `MaxAbsScaler` | `RsMaxAbsScaler` (macro, no params) | `ferrolearn_preprocess::FittedMaxAbsScaler` | fit/transform | transformer (no knobs) |
| 39 | `RobustScaler` | `RsRobustScaler` (macro, no params) | `ferrolearn_preprocess::FittedRobustScaler` | fit/transform | transformer (no knobs) |
| 40 | `PowerTransformer` | `RsPowerTransformer` (macro, no params) | `ferrolearn_preprocess::FittedPowerTransformer` | fit/transform | transformer (no knobs) |
| 41 | `Nystroem` | `RsNystroem` (macro, no params) | `ferrolearn_kernel::FittedNystroem` | fit/transform | transformer (no knobs) |
| 42 | `RBFSampler` | `RsRBFSampler` (macro, no params) | `ferrolearn_kernel::FittedRBFSampler` | fit/transform | transformer (no knobs) |

Missing fitted-attribute surface (all owned downstream): regressors expose NO
`coef_`/`feature_importances_`; classifiers NO `predict_proba`/`decision_function`;
transformers NO `inverse_transform`/`components_`; clusterers NO
`cluster_centers_`/`n_clusters_`/`children_`. The Python wrappers set only
`n_features_in_`, `classes_` (classifiers), `labels_` (clusterers).

## Requirements

Grouped by category per the scale-management directive (one API-CONFORM +
VALUE-PARITY row per category, estimators listed in Evidence), plus the
cross-cutting binding REQs and the three HEADLINE fixable divergences.

### Per-category API conformance (SHIPPED on the default path)

- REQ-REGRESSOR-API-CONFORM: each of the 11 regressors (`BayesianRidge`,
  `ARDRegression`, `HuberRegressor`, `QuantileRegressor`, `DecisionTreeRegressor`,
  `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`,
  `HistGradientBoostingRegressor`, `KNeighborsRegressor`, `KernelRidge`) exposes
  `fit(X, y)`/`predict(X)` (bound on its `_Rs*` class, wrapped by
  `_RegressorWrapper`) plus `score` (inherited from `RegressorMixin`) + sets
  `n_features_in_`, returning a 1-D float64 prediction array of the right shape.
- REQ-CLASSIFIER-API-CONFORM: each of the 13 classifiers (`RidgeClassifier`,
  `LinearSVC`, `QuadraticDiscriminantAnalysis`, `MultinomialNB`, `BernoulliNB`,
  `ComplementNB`, `ExtraTreeClassifier`, `ExtraTreesClassifier`,
  `AdaBoostClassifier`, `GradientBoostingClassifier`,
  `HistGradientBoostingClassifier`, `BaggingClassifier`, `NearestCentroid`)
  exposes `fit`/`predict` + `score` (from `ClassifierMixin`), with the
  `_ClassifierWrapper` `_encode`/decode round-trip mapping arbitrary label dtypes
  to/from the Rust `usize`-label core, exposing `classes_` (sorted unique labels)
  + `n_features_in_`, and `predict` returning labels in the original dtype.
- REQ-CLUSTERER-API-CONFORM: each of the 5 clusterers (`MiniBatchKMeans`,
  `DBSCAN`, `AgglomerativeClustering`, `Birch`, `GaussianMixture`) exposes
  `fit(X)` and the `labels_` attribute (a `labels_` getter on the `_Rs*` class
  for the four cluster types; `GaussianMixture` exposes `fit`/`predict`), with
  `MiniBatchKMeans`/`GaussianMixture` additionally exposing `predict`, plus
  `fit_predict` (from `ClusterMixin` / hand-written), exposing `n_features_in_`.
- REQ-TRANSFORMER-API-CONFORM: each of the 13 transformers (`IncrementalPCA`,
  `TruncatedSVD`, `FastICA`, `NMF`, `KernelPCA`, `SparsePCA`, `FactorAnalysis`,
  `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `PowerTransformer`, `Nystroem`,
  `RBFSampler`) exposes `fit(X)`/`transform(X)` (bound on its `_Rs*` class,
  wrapped by `_TransformerWrapper`) plus `fit_transform` (from
  `TransformerMixin`), exposing `n_features_in_` and returning a 2-D float64
  array.

### Per-category default-path value parity (SHIPPED, owned downstream)

- REQ-REGRESSOR-VALUE-PARITY: on the deterministic default path the marshalled
  predictions of the deterministic regressors match the live sklearn oracle to
  the downstream-verified tolerance. [Owned downstream: `ferrolearn_linear`,
  `ferrolearn_tree`, `ferrolearn_neighbors`, `ferrolearn_kernel` REQ tables.]
- REQ-CLASSIFIER-VALUE-PARITY: on the deterministic default path the decoded
  label predictions match the live sklearn oracle. [Owned downstream:
  `ferrolearn_linear`, `ferrolearn_bayes`, `ferrolearn_tree` REQ tables.]
- REQ-CLUSTERER-VALUE-PARITY: on the deterministic default path the `labels_`
  partition matches sklearn up to a label permutation (clustering invariance).
  [Owned downstream: `ferrolearn_cluster` REQ table.]
- REQ-TRANSFORMER-VALUE-PARITY: on the deterministic default path the
  transformed output matches the live sklearn oracle to the downstream-verified
  tolerance (sign/permutation invariance where the decomposition is sign-free).
  [Owned downstream: `ferrolearn_decomp`, `ferrolearn_preprocess`,
  `ferrolearn_kernel` REQ tables.]

### HEADLINE fixable binding-level divergences

- REQ-CTOR-ABI-POSITIONAL (**HEADLINE 1**): the 16 wrappers whose sklearn
  primary hyperparameter is positional-or-keyword
  (`RandomForestRegressor`/`n_estimators`, `ExtraTreesRegressor`/`n_estimators`,
  `KNeighborsRegressor`/`n_neighbors`, `RidgeClassifier`/`alpha`,
  `KernelRidge`/`alpha`, `MiniBatchKMeans`/`n_clusters`,
  `AgglomerativeClustering`/`n_clusters`, `DBSCAN`/`eps`,
  `GaussianMixture`/`n_components`, `TruncatedSVD`/`n_components`,
  `FastICA`/`n_components`, `NMF`/`n_components`, `IncrementalPCA`/`n_components`,
  `KernelPCA`/`n_components`, `SparsePCA`/`n_components`,
  `FactorAnalysis`/`n_components`) accept that param POSITIONALLY —
  `ferrolearn.X(val)` constructs an estimator with the param set to `val`,
  matching sklearn. Today every `_extras.py` `__init__` is
  `def __init__(self, *, <param>=<default>)` (leading `*`), so
  `ferrolearn.X(val)` raises `TypeError`. Fix: move the primary param before the
  `*` in each affected wrapper `__init__` (Python-side; the `#[pyo3(signature)]`
  on the `_Rs*` class already accepts it by name).
- REQ-MODULE-ALLOW (**HEADLINE 2**): `extras.rs` carries NO module/crate-root
  `#![allow(..)]` (R-CODE-3/R-APG-1). Today `extras.rs:5` has
  `#![allow(non_snake_case)]`. All `#[pyclass]` field names are snake_case and
  type names are PascalCase, so the lint appears unneeded and the attribute is
  likely removable outright (or, if any generated name trips it, replaceable with
  a per-item `#[allow(non_snake_case, reason="...")]` per R-APG-3).
- REQ-PHASE-FRAMING (**HEADLINE 3**): neither `extras.rs:1` `//!` nor
  `_extras.py:1` docstring uses `Phase \d+` deferral framing (R-DEFER-4). Today
  both say "Phase 2 binding expansion" / "Phase-2 binding wrappers". Fix: reword
  to describe the surface (the ~40-estimator extras binding) without phase
  framing, and add a `## REQ status` summary in the `//!`.

### Binding-level fitted-attribute surface (SHIPPED)

- REQ-DISCRETE-NB-FITTED-ATTRS (#2103): `MultinomialNB`/`BernoulliNB`/
  `ComplementNB` expose the four `_BaseDiscreteNB` fitted attributes sklearn
  defines — `feature_log_prob_` (shape `(n_classes, n_features)`),
  `class_log_prior_` (`(n_classes,)`), `feature_count_` (`(n_classes,
  n_features)`), `class_count_` (`(n_classes,)`) — matching the live sklearn
  oracle to atol 1e-7. The Rust fitted types already compute all four
  (`ferrolearn_bayes` REQ-1/3/4 SHIPPED); this is a binding-surface exposure, not
  a library-math change, so it is owned HERE (the binding) rather than downstream.
  Impl: a second `#[pymethods]` block per `_Rs*NB` class (pyo3
  `multiple-pymethods`) with four `#[getter]`s; the `py_classifier!` macro stays
  unchanged. Consumer: `_extras.py::_DiscreteNBWrapper.fit`. For `ComplementNB`,
  `feature_log_prob_` is the `-logged` complement weight (positive values) — this
  is exactly what sklearn exposes (`naive_bayes.py:1041`), not a bug.

- REQ-RANSAC-BINDING (#2178): `RANSACRegressor` is bound over the default
  `LinearRegression` base — `RsRANSACRegressor` (hand-written `#[pyclass]`,
  `extras.rs`) over `ferrolearn_linear::RANSACRegressor<f64,
  LinearRegression<f64>>` (the base is FIXED to `LinearRegression::<f64>::new()`,
  sklearn's `estimator=None`→`LinearRegression()` default `fit_intercept=True`,
  `_ransac.py:380`). The constructor mirrors sklearn's full surface
  `(estimator=None, *, min_samples=None, residual_threshold=None,
  is_data_valid=None, is_model_valid=None, max_trials=100, max_skips=np.inf,
  stop_n_inliers=np.inf, stop_score=np.inf, stop_probability=0.99,
  loss='absolute_error', random_state=None)` (`_ransac.py:288-315`). The
  `_extras.py::RANSACRegressor` wrapper resolves `min_samples` BEFORE the ABI
  (`None`→`n_features+1`, int≥1→as-is, float∈[0,1]→`ceil(min_samples*n_samples)`,
  `_ransac.py:382-397`), threads `residual_threshold`/`max_trials`/`random_state`,
  and rejects the unsupported surface: a non-default `estimator` /
  `loss='squared_error'` (or callable, ransac.rs REQ-8 #516) / non-default
  `max_skips`/`stop_*` (ransac.rs REQ-7 #515) / `is_data_valid`/`is_model_valid`
  (ransac.rs REQ-11 #519) → `NotImplementedError`; out-of-range
  `min_samples`/`max_trials`/`residual_threshold` → `ValueError` (sklearn
  `InvalidParameterError`). Fitted attrs `coef_`/`intercept_`/`inlier_mask_`/
  `n_features_in_` are surfaced; `coef_`/`intercept_` are recovered EXACTLY from
  the refit base via affine probe predictions (`intercept_=predict(0)`,
  `coef_[j]=predict(e_j)−intercept_`) because the core keeps the refit base
  private (ransac.rs REQ-10 NOT-STARTED #518; no `n_trials_`/`estimator_` faked).
  Verification (model B): `tests/divergence_extras.py::test_ransac_*` (14 cases,
  live sklearn 1.5.2 oracle on WELL-SEPARATED outlier data so the best inlier set
  is unique → `coef_`/`intercept_`/`inlier_mask_` MATCH sklearn `atol 1e-2`
  despite the RNG-substrate caveat #2118). Impl: `extras.rs` `RsRANSACRegressor`
  (`fit`/`predict`/`coef_`/`intercept_`/`inlier_mask_`). Non-test consumer
  (R-DEFER-1): `_extras.py::RANSACRegressor` (`_make_rs`/`fit`/`predict`),
  `__init__.py` re-export, `lib.rs` `m.add_class::<extras::RsRANSACRegressor>()`.

### Downstream-owned divergence REQs (NOT-STARTED)

- REQ-DECOMP-NCOMPONENTS-DEFAULT: the 5 decomp transformers
  (`IncrementalPCA`/`FastICA`/`KernelPCA`/`SparsePCA`/`FactorAnalysis`, plus
  `NMF`) default `n_components` to sklearn's value (`None` / `'warn'`→`None`)
  rather than the ferrolearn hardcoded `2` (R-DEV-2). [`TruncatedSVD` default `2`
  MATCHES sklearn.] The `None`/`'warn'` auto-`n_components` behavior is owned by
  `ferrolearn_decomp`; the default literal is set in both `extras.rs` macro
  invocations and `_extras.py`.
- REQ-MISSING-METHODS: regressors expose `coef_`/`feature_importances_`;
  classifiers expose `predict_proba`/`decision_function`; transformers expose
  `inverse_transform`/`components_`; clusterers expose `cluster_centers_`/
  `children_`/`n_clusters_` — matching sklearn. [Owned downstream by the eight
  library crates; the binding cannot expose attrs/methods the library does not
  compute.]
- REQ-MISSING-PARAMS: each estimator exposes sklearn's FULL constructor surface,
  not the thin subset bound here (e.g. `RandomForestRegressor` lacks
  `criterion`/`max_features`/`bootstrap`/`oob_score`/`n_jobs`/...;
  `BaggingClassifier` lacks the `estimator` base-estimator knob; the no-knob
  scalers lack `feature_range`/`quantile_range`/`method`; `Nystroem`/`RBFSampler`
  lack `kernel`/`gamma`/`n_components`). [Owned downstream by the eight library
  crates.]
- REQ-VALUE-PARITY-RNG: the stochastic estimators
  (`RandomForestRegressor`/`ExtraTreesRegressor`/`ExtraTreesClassifier`,
  `GradientBoosting*`/`HistGradientBoosting*`, `BaggingClassifier`,
  `AdaBoostClassifier`, `MiniBatchKMeans`, `GaussianMixture`, `FastICA`, `NMF`)
  reproduce sklearn outputs under a shared `random_state` — which requires the
  numpy RNG (Mersenne-Twister/PCG64) stream, i.e. `ferray::random` (R-SUBSTRATE-5).
  [Owned downstream / upstream in ferray; the `_Rs*` classes pass `random_state`
  as `Option<u64>` to a non-numpy RNG, so seeded streams will not match sklearn.]
- REQ-CONSUMER: the binding IS the public API; its non-test production consumers
  are the `_extras.py` wrappers, the `ferrolearn/__init__.py` re-export, the
  `lib.rs` `add_class` registrations, and the head-to-head bench harness.
- REQ-SUBSTRATE: the binding's array marshalling is on `ferray::numpy_interop`
  producing `ferray-core` arrays, not rust-numpy + `ndarray` (R-SUBSTRATE-1).
  [Owned by `conversions.md` REQ-FERRAY #2027.]

## Acceptance criteria

All expected values come from the live sklearn 1.5.2 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn. The pytest gauntlet
(`cd ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`) is the
end-to-end check (model B); rebuild first if the Rust side changed
(`cd ferrolearn-python && maturin develop`).

- AC-REGRESSOR-API-CONFORM (REQ-REGRESSOR-API-CONFORM): for each of the 11
  regressors, `cd ferrolearn-python && PYTHONPATH=python python3 -c "import
  numpy as np, ferrolearn as f; X=np.random.RandomState(0).randn(20,3);
  y=np.random.RandomState(1).randn(20); m=f.<Est>().fit(X,y); p=m.predict(X);
  assert p.shape==(20,) and m.n_features_in_==3"` succeeds. The `_RegressorWrapper`
  surface is exercised by `tests/divergence_regressors.py`.
- AC-CLASSIFIER-API-CONFORM (REQ-CLASSIFIER-API-CONFORM): string-label round-trip
  — `cd ferrolearn-python && PYTHONPATH=python python3 -c "import numpy as np,
  ferrolearn as f; X=np.random.RandomState(0).randn(30,4);
  y=np.array(['a','b','c']*10); m=f.<Est>().fit(X,y);
  assert set(m.predict(X)) <= {'a','b','c'} and list(m.classes_)==['a','b','c']"`
  succeeds. Exercised by `tests/divergence_classifiers.py`.
- AC-CLUSTERER-API-CONFORM (REQ-CLUSTERER-API-CONFORM):
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "import numpy as np,
  ferrolearn as f; X=np.random.RandomState(0).randn(30,2);
  m=f.<Est>().fit(X); assert m.labels_.shape==(30,)"` succeeds for the four
  cluster types; `MiniBatchKMeans`/`GaussianMixture` additionally satisfy
  `m.predict(X).shape==(30,)`. Exercised by `tests/divergence_clusterers.py`.
- AC-TRANSFORMER-API-CONFORM (REQ-TRANSFORMER-API-CONFORM):
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "import numpy as np,
  ferrolearn as f; X=np.abs(np.random.RandomState(0).randn(20,4));
  Xt=f.<Est>().fit_transform(X); assert Xt.ndim==2 and Xt.shape[0]==20"`
  succeeds. Exercised by `tests/divergence_transformers.py`.
- AC-CTOR-ABI-POSITIONAL (REQ-CTOR-ABI-POSITIONAL): sklearn oracle —
  `cd /tmp && python3 -c "import inspect; from sklearn.ensemble import
  RandomForestRegressor as R; print(inspect.signature(R.__init__).parameters['n_estimators'].kind.name)"`
  → `POSITIONAL_OR_KEYWORD` (and the 15 others, table above). ferrolearn FAILS —
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "import ferrolearn as f;
  f.RandomForestRegressor(50)"` → `TypeError: RandomForestRegressor.__init__()
  takes 1 positional argument but 2 were given` (live-confirmed at `8aa19d6`). A
  critic pins a PARAMETRIZED FAILING pytest over the 16 (e.g.
  `assert f.RandomForestRegressor(50).n_estimators == 50`, `f.DBSCAN(0.3).eps ==
  0.3`, `f.TruncatedSVD(3).n_components == 3`). FAILS until each affected
  `_extras.py` `__init__` moves its primary param before the `*`.
- AC-MODULE-ALLOW (REQ-MODULE-ALLOW): `grep -n "#!\[allow" /home/doll/ferrolearn/ferrolearn-python/src/extras.rs`
  → `5:#![allow(non_snake_case)]` (present = R-CODE-3 violation). After removal,
  `cargo clippy -p ferrolearn-python --all-targets -- -D warnings` stays green
  (confirming the allow was unneeded) and the grep returns empty. A critic pins
  this as a gate check.
- AC-PHASE-FRAMING (REQ-PHASE-FRAMING):
  `grep -niE "phase [0-9]" /home/doll/ferrolearn/ferrolearn-python/src/extras.rs
  /home/doll/ferrolearn/ferrolearn-python/python/ferrolearn/_extras.py` →
  `extras.rs:1` and `_extras.py:1` match (R-DEFER-4 violation). After rewording,
  the grep returns empty.
- AC-DECOMP-NCOMPONENTS-DEFAULT (REQ-DECOMP-NCOMPONENTS-DEFAULT): sklearn oracle
  `cd /tmp && python3 -c "import inspect; from sklearn.decomposition import
  FastICA; print(inspect.signature(FastICA.__init__).parameters['n_components'].default)"`
  → `None` (NMF → `'warn'`). ferrolearn:
  `cd ferrolearn-python && PYTHONPATH=python python3 -c "import ferrolearn as f;
  print(f.FastICA().n_components)"` → `2` (live-confirmed). A critic pins a
  FAILING pytest; the auto-`n_components` behavior is owned by `ferrolearn_decomp`.
- AC-DISCRETE-NB-FITTED-ATTRS (REQ-DISCRETE-NB-FITTED-ATTRS): live sklearn
  1.5.2 oracle — for each of `MultinomialNB`/`BernoulliNB`/`ComplementNB`, fit
  BOTH `ferrolearn.<Est>()` and `sklearn.naive_bayes.<Est>()` on the same integer
  count matrix `X`/labels `y` and assert
  `np.testing.assert_allclose(getattr(fl, a), getattr(sk, a), atol=1e-7)` for
  `a in {feature_log_prob_, class_log_prior_, feature_count_, class_count_}`.
  Pinned by `tests/divergence_extras.py::test_{multinomial,bernoulli,complement}_discrete_nb_fitted_attrs_match_sklearn`.
- AC-CONSUMER (REQ-CONSUMER):
  `grep -n "_Rs" /home/doll/ferrolearn/ferrolearn-python/python/ferrolearn/_extras.py`
  shows each wrapper constructs its `_Rs*` class; `ferrolearn/__init__.py:16`
  re-exports all ~40; `lib.rs:27-88` registers every `_Rs*` via `add_class`; the
  bench harness (`ferrolearn-bench/src/bin/harness.rs`) drives them head-to-head
  vs sklearn. The pytest gauntlet exercises the consumer surface.
- AC-SUBSTRATE (REQ-SUBSTRATE): `extras.rs:7-10` shows `use crate::conversions::*`
  + `use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2}` + `use
  ndarray::Array1` — the wrong substrate per R-SUBSTRATE-1. Owned by
  `conversions.md` #2027.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-REGRESSOR-API-CONFORM (fit/predict, 11 regressors) | SHIPPED | `py_regressor!` macro in `extras.rs` generates `fit(x: PyReadonlyArray2, y: PyReadonlyArray1<f64>)`/`predict` for `RsBayesianRidge`/`RsARDRegression`/`RsHuberRegressor`/`RsQuantileRegressor`/`RsDecisionTreeRegressor`/`RsKNeighborsRegressor`/`RsKernelRidge`; hand-written `RsRandomForestRegressor`/`RsExtraTreesRegressor`/`RsGradientBoostingRegressor`/`RsHistGradientBoostingRegressor` mirror the same shape (`fit`/`predict` over `FittedRandomForestRegressor` etc.). Wrapped by `_RegressorWrapper.fit`/`predict` in `_extras.py` (sets `n_features_in_ = X.shape[1]`, inherits `score` from `RegressorMixin`). Mirrors the sklearn regressor `fit`/`predict` contract across `ensemble/_forest.py`, `_gb.py`, `_hist_gradient_boosting/`, `linear_model/_bayes.py`/`_huber.py`/`_quantile.py`, `neighbors/_regression.py`, `kernel_ridge.py`, `tree`. Non-test consumer: `_extras.py::_RegressorWrapper` subclasses + `ferrolearn/__init__.py:16` re-export + `lib.rs:27-41` registration + bench harness. Verification (model B): `tests/divergence_regressors.py` + AC probe (predict shape `(n,)`, `n_features_in_`). |
| REQ-CLASSIFIER-API-CONFORM (fit/predict + LabelEncoder, 13 classifiers) | SHIPPED | `py_classifier!` macro generates `fit(x, y: PyReadonlyArray1<i64>)` (decoding via `numpy1_to_ndarray_usize`) / `predict → PyArray1<i64>` for `RsRidgeClassifier`/`RsLinearSVC`/`RsQDA`/`RsMultinomialNB`/`RsBernoulliNB`/`RsComplementNB`/`RsExtraTreeClassifier`/`RsNearestCentroid`; hand-written `RsExtraTreesClassifier`/`RsAdaBoostClassifier`/`RsGradientBoostingClassifier`/`RsHistGradientBoostingClassifier`/`RsBaggingClassifier` mirror it. Wrapped by `_ClassifierWrapper.fit`/`predict` in `_extras.py`: `_encode(y)` (`np.unique`+`np.searchsorted`) sets `classes_`, fit on encoded `y_enc`, `predict` decodes via `self.classes_[y_enc]` — the LabelEncoder round-trip mapping arbitrary dtypes to the Rust `usize`-label core. Mirrors the sklearn classifier `fit`/`predict`/`classes_` contract across `linear_model/_ridge.py`, `svm/_classes.py`, `discriminant_analysis.py`, `naive_bayes.py`, `ensemble/*`, `tree`, `neighbors/_nearest_centroid.py`. Non-test consumer: `_extras.py::_ClassifierWrapper` subclasses + `__init__.py:16` + `lib.rs:44-62`. Verification (model B): `tests/divergence_classifiers.py` + string-label AC probe. |
| REQ-CLUSTERER-API-CONFORM (fit + labels_, 5 clusterers incl. GMM) | SHIPPED | hand-written `RsMiniBatchKMeans`/`RsDBSCAN`/`RsAgglomerativeClustering`/`RsBirch` each expose `fit(x: PyReadonlyArray2)` (over `Fit::fit(&x_nd, &())`) + a `#[getter] fn labels_` returning `PyArray1<i64>` from `f.labels()`; `RsMiniBatchKMeans` additionally exposes `predict`; `RsGaussianMixture` exposes `fit`/`predict` (over `FittedGaussianMixture`). Wrapped by `_ClusterWrapper.fit` (sets `self.labels_ = np.asarray(self._rs.labels_)`, `n_features_in_`) + `fit_predict`; `GaussianMixture` is a hand wrapper with `fit`/`predict`/`fit_predict`. Mirrors `cluster/_kmeans.py`/`_dbscan.py`/`_agglomerative.py`/`_birch.py` + `mixture/_gaussian_mixture.py` (`labels_`/`predict`). Non-test consumer: `_extras.py::_ClusterWrapper` subclasses + `GaussianMixture` + `__init__.py:16` + `lib.rs:65-69`. Verification (model B): `tests/divergence_clusterers.py` + `labels_` shape AC. |
| REQ-TRANSFORMER-API-CONFORM (fit/transform, 13 transformers) | SHIPPED | `py_transformer!` macro generates `fit(x: PyReadonlyArray2)` (over `Fit::fit(&x_nd, &())`) / `transform → PyArray2<f64>` for `RsIncrementalPCA`/`RsTruncatedSVD`/`RsFastICA`/`RsNMF`/`RsKernelPCA`/`RsSparsePCA`/`RsFactorAnalysis`/`RsMinMaxScaler`/`RsMaxAbsScaler`/`RsRobustScaler`/`RsPowerTransformer`/`RsNystroem`/`RsRBFSampler`. Wrapped by `_TransformerWrapper.fit`/`transform` in `_extras.py` (sets `n_features_in_`, inherits `fit_transform` from `TransformerMixin`). Mirrors the sklearn transformer `fit`/`transform` contract across `decomposition/*`, `preprocessing/_data.py`, `kernel_approximation.py`. Non-test consumer: `_extras.py::_TransformerWrapper` subclasses + `__init__.py:16` + `lib.rs:72-88`. Verification (model B): `tests/divergence_transformers.py` + `fit_transform` 2-D AC. |
| REQ-REGRESSOR-VALUE-PARITY (default-path predict parity) | SHIPPED | on the DETERMINISTIC default path. The `_Rs*` regressors are thin shims over the fitted library types; predict parity for the deterministic regressors (`BayesianRidge`/`ARDRegression`/`HuberRegressor`/`QuantileRegressor`/`DecisionTreeRegressor`/`KNeighborsRegressor`/`KernelRidge`) is critic-verified against the live sklearn oracle in the owning crates' REQ tables — owned downstream by `ferrolearn_linear`, `ferrolearn_tree`, `ferrolearn_neighbors`, `ferrolearn_kernel` (pre-existing audited crates). Non-test consumer: `_extras.py::_RegressorWrapper`. (Seeded-RNG ensemble parity is REQ-VALUE-PARITY-RNG, NOT-STARTED.) |
| REQ-CLASSIFIER-VALUE-PARITY (default-path label parity) | SHIPPED | on the DETERMINISTIC default path. The decoded label predictions of the deterministic classifiers (`RidgeClassifier`/`LinearSVC`/`QuadraticDiscriminantAnalysis`/`MultinomialNB`/`BernoulliNB`/`ComplementNB`/`ExtraTreeClassifier`/`NearestCentroid`) match sklearn after the `_encode`/decode round-trip; verified in the owning crates — owned downstream by `ferrolearn_linear`, `ferrolearn_bayes`, `ferrolearn_tree` (pre-existing audited crates). Non-test consumer: `_extras.py::_ClassifierWrapper`. (Seeded-RNG ensemble classifiers → REQ-VALUE-PARITY-RNG.) |
| REQ-CLUSTERER-VALUE-PARITY (default-path partition parity) | SHIPPED | on the DETERMINISTIC default path. `labels_` for the deterministic clusterers (`DBSCAN`/`AgglomerativeClustering`/`Birch`) matches sklearn's partition up to label permutation; verified in `ferrolearn_cluster` (pre-existing audited crate). Non-test consumer: `_extras.py::_ClusterWrapper`. (`MiniBatchKMeans`/`GaussianMixture` seeded parity → REQ-VALUE-PARITY-RNG.) |
| REQ-DBSCAN-BINDING-SURFACE (`sample_weight` + `core_sample_indices_` + `components_` + `metric`/`p`) | SHIPPED | FIXED #2190: `RsDBSCAN.fit` (`extras.rs`) gains `#[pyo3(signature = (x, sample_weight=None))]` and chains `.with_sample_weight(numpy1_to_ndarray(w))` onto `ferrolearn_cluster::DBSCAN::<f64>::new(self.eps).with_min_samples(self.min_samples)` when `Some` (the weighted core-determination path, `cluster/_dbscan.py:370,427-435`); a wrong-length weight surfaces as the core's `FerroError::ShapeMismatch` → `PyValueError` (mirroring `_check_sample_weight`'s `ValueError`). New `#[getter] fn core_sample_indices_ → PyArray1<i64>` (`np.where(core_samples)[0]`, `_dbscan.py:438`) and `#[getter] fn components_ → PyArray2<f64>` (`X[core_sample_indices_].copy()`, `_dbscan.py:441-446`) surface `f.core_sample_indices()`/`f.components()`. EXTENDED #2193: `RsDBSCAN::new` ctor gains `metric="euclidean".to_string()` + `p=None` (`_dbscan.py:345-363`); `fn fit` resolves the lowercased metric string via the module-level `fn resolve_dbscan_metric(metric, p) -> PyResult<DbscanMetric<f64>>` (`euclidean`/`l2`→`Euclidean`, `manhattan`/`l1`/`cityblock`→`Manhattan`, `chebyshev`→`Chebyshev`, `minkowski`→`Minkowski(p.unwrap_or(2.0))`; unknown→`PyValueError`, matching sklearn's `StrOptions(set(_VALID_METRICS)|{"precomputed"})` `InvalidParameterError ⊂ ValueError`, `_dbscan.py:334-337`) and chains `.with_metric(resolved)` onto the builder — CONSUMING `ferrolearn_cluster::dbscan::DbscanMetric`/`with_metric` (core commit 485c06fcd, `dbscan.rs`). `p` is the Minkowski order, IGNORED for every non-Minkowski metric (#2192, `_dbscan.py:411-418` — only `Minkowski` carries `p`, `with_p` is NOT called on the other branches); a non-positive/NaN Minkowski `p` surfaces from `Fit::fit` as `FerroError::InvalidParameter` → `PyValueError`. `_extras.py::DBSCAN` ctor is now `(eps=0.5, *, min_samples=5, metric='euclidean', p=None)` (`_dbscan.py:345-363`), storing the new attrs + threading them through `_make_rs(metric=self.metric, p=self.p)` (so `BaseEstimator.get_params`/`clone`/`set_params` round-trip eps/min_samples/metric/p); it OVERRIDES `fit(self, X, y=None, sample_weight=None)` (threading the weight, `_dbscan.py:370`) + `fit_predict(X, y=None, sample_weight=None)` (`_dbscan.py:450`), setting `self.labels_`/`core_sample_indices_`/`components_`/`n_features_in_`; the `None`-weight + `metric='euclidean'` path stays byte-identical to the prior `labels_`-only fit. Non-test consumer: `import ferrolearn; ferrolearn.DBSCAN(eps, min_samples=k, metric=..., p=...).fit(X, sample_weight=w)` exposing all three fitted attrs. Live-oracle verification (model B, R-CHAR-3): `tests/divergence_dbscan_sample_weight.py` (9 pass) — unweighted+default `labels_`/`core_sample_indices_`/`components_` element-wise == sklearn; promote (`w=[5,1,1]` isolated→core) / demote (`w=[0.5;4]`, `min_samples=4`→all noise) cases; the #2189 pairwise-sum boundary (ten `0.1` weights: sequential fold `0.999…`<1 would demote-all, numpy pairwise `==1.0`→all core, matches the oracle THROUGH the binding); negative-weight inhibition; wrong-length `sample_weight` → `ValueError` (no panic, R-CODE-2); `fit_predict(X, sample_weight=w)`; `clone()`/`get_params`/`set_params`/default `eps=0.5`/`min_samples=5`. PLUS `tests/test_dbscan_metric.py` (20 pass, #2193) — per-metric `labels_`+`core_sample_indices_` parity for `euclidean`/`l2`, `manhattan`/`l1`/`cityblock`, `chebyshev`, `minkowski`-p∈{1,2,3} on metric-DISCRIMINATING fixtures (manhattan/chebyshev flip the partition vs euclidean); `minkowski p=1 == manhattan`, `p=2 == euclidean` (both == sklearn); default `metric='euclidean'` (no kwarg) unchanged; `metric='euclidean', p=3` → euclidean (p IGNORED #2192, fixture discriminates vs minkowski-3) matches sklearn; unknown `metric='nope'` → `ValueError` (both sides); non-positive Minkowski `p∈{0,-1}` → `ValueError` (both sides); manhattan + `sample_weight`; `clone`/`get_params`(incl. metric/p)/`set_params(metric='manhattan').fit`. DOCUMENTED DIVERGENCE (R-DEV-3): ferrolearn `metric='minkowski', p=None` → `p=2` (documented intent `_dbscan.py:243-246`, fits == euclidean) whereas the LIVE sklearn 1.5.2 oracle raises `TypeError` (forwards `p=None` to `NearestNeighbors`, `None < 1`) — pinned both-sides in `test_minkowski_p_none_resolves_to_p2`. NOT-STARTED follow-on: `metric_params` (per-metric kwargs), `metric='precomputed'`, callable metric (no ferrolearn distance-matrix/closure path), `algorithm`/`leaf_size`/`n_jobs` (neighbor-search knobs, identical result) stay unexposed. |
| REQ-TRANSFORMER-VALUE-PARITY (default-path transform parity) | SHIPPED | on the DETERMINISTIC default path. Transformed output for the deterministic transformers (`TruncatedSVD`/`KernelPCA`/`SparsePCA`/`FactorAnalysis`/`IncrementalPCA`/`MinMaxScaler`/`MaxAbsScaler`/`RobustScaler`/`PowerTransformer`/`Nystroem`/`RBFSampler`) matches the live sklearn oracle (sign/permutation invariance where applicable); verified in `ferrolearn_decomp`, `ferrolearn_preprocess`, `ferrolearn_kernel` (pre-existing audited crates). Non-test consumer: `_extras.py::_TransformerWrapper`. (`FastICA`/`NMF` seeded init → REQ-VALUE-PARITY-RNG.) |
| REQ-CTOR-ABI-POSITIONAL (16 positional primaries) — HEADLINE 1 | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 constructor ABI; single-wrapper-class-fixable per estimator). sklearn makes the primary hyperparameter `POSITIONAL_OR_KEYWORD` for all 16 (live, table in Upstream reference): `RandomForestRegressor`/`ExtraTreesRegressor` (`n_estimators`), `KNeighborsRegressor` (`n_neighbors`), `RidgeClassifier`/`KernelRidge` (`alpha`), `MiniBatchKMeans`/`AgglomerativeClustering` (`n_clusters`), `DBSCAN` (`eps`), `GaussianMixture` (`n_components`), `TruncatedSVD`/`FastICA`/`NMF`/`IncrementalPCA`/`KernelPCA`/`SparsePCA`/`FactorAnalysis` (`n_components`). Every `_extras.py` `__init__` is `def __init__(self, *, <param>=<default>)` (leading `*`), so `ferrolearn.X(val)` raises `TypeError` (live: `f.RandomForestRegressor(50)` → `TypeError: __init__() takes 1 positional argument but 2 were given`) while `sklearn.X(val)` works. Fix: move the primary param before the `*` in each of the 16 `_extras.py` wrappers (the `_Rs*` `#[pyo3(signature)]` already accepts it by name). |
| REQ-MODULE-ALLOW (no module-root `#![allow]`) — HEADLINE 2 | NOT-STARTED | blocker issue to be filed by critic (R-CODE-3/R-APG-1). `extras.rs:5` carries `#![allow(non_snake_case)]` at MODULE ROOT. All `#[pyclass]` field names (`n_estimators`, `max_depth`, `n_clusters`, ...) are snake_case and type/Python names PascalCase, so the lint appears unneeded — likely removable outright. Fix (Rust-side): delete line 5 and confirm `cargo clippy -p ferrolearn-python --all-targets -- -D warnings` stays green; if any generated item trips the lint, replace with a scoped `#[allow(non_snake_case, reason="...")]` per R-APG-3. The anti-pattern-gate (R-APG-1) flags this construct. |
| REQ-PHASE-FRAMING (no Phase-N deferral framing) — HEADLINE 3 | NOT-STARTED | blocker issue to be filed by critic (R-DEFER-4). `extras.rs:1-3` `//!` says "Additional PyO3 bindings (Phase 2 binding expansion)" and `_extras.py:1-3` docstring says "Phase-2 binding wrappers"; `grep -niE "phase [0-9]"` matches both. Fix: reword to describe the ~40-estimator extras binding surface without phase framing, and add a `## REQ status` summary to the `//!` per goal.md §"every routed file has a `## REQ status` table". |
| REQ-DECOMP-NCOMPONENTS-DEFAULT (n_components default) | NOT-STARTED | open prereq owned downstream by `ferrolearn_decomp` (auto-`n_components` for `None`/`'warn'`). sklearn defaults `n_components=None` for `IncrementalPCA`/`FastICA`/`KernelPCA`/`SparsePCA`/`FactorAnalysis` and `'warn'`→`None` for `NMF` (live; `decomposition/*` `file:line` in Upstream reference); ferrolearn hardcodes `2` in both the `py_transformer!` macro invocations (`extras.rs`) and `_extras.py` (`def __init__(self, *, n_components=2)`). Live: `f.FastICA().n_components` → `2` vs sklearn `None`. The binding cannot synthesize the `None`-auto behavior the library does not implement; the literal `2` is the binding default, but the auto-rank logic is owned downstream. (`TruncatedSVD` default `2` MATCHES sklearn — no divergence.) |
| REQ-DISCRETE-NB-FITTED-ATTRS (feature_log_prob_/class_log_prior_/feature_count_/class_count_) | SHIPPED | FIXED #2103. The three discrete-NB classifiers (`MultinomialNB`/`BernoulliNB`/`ComplementNB`) expose the four `_BaseDiscreteNB` fitted attributes sklearn defines (`naive_bayes.py:880-892` feature counts/log-prob, `:580-602` `_update_class_log_prior`, `ComplementNB._update_feature_log_prob` `:1032-1042` for the `-logged` complement weight). Impl: a SECOND `#[pymethods] impl Rs{Multinomial,Bernoulli,Complement}NB` block in `extras.rs` (enabled by `pyo3` feature `multiple-pymethods` in `Cargo.toml`) adds four `#[getter]`s each — `feature_log_prob_`/`feature_count_` → `ndarray2_to_numpy(py, fitted.feature_log_prob()/.feature_count())`, `class_log_prior_`/`class_count_` → `ndarray1_to_numpy`; the `py_classifier!` macro and all 18 invocations stay UNCHANGED. The Rust fitted types already compute all four (`ferrolearn_bayes::FittedMultinomialNB::feature_log_prob`/`class_log_prior`/`feature_count`/`class_count`, `FittedBernoulliNB::*`, `FittedComplementNB::*` — pre-existing audited, `ferrolearn-bayes` REQ-1/3/4 SHIPPED). `class_count()` (all three) and `ComplementNB::class_log_prior()` return owned `Array1`, bound to a local before marshalling; no `unwrap`/`expect`/`panic` (not-fitted → `PyRuntimeError`). Non-test production consumer: `_extras.py::_DiscreteNBWrapper.fit` (the three NB wrappers subclass it) sets `self.{feature_log_prob_,class_log_prior_,feature_count_,class_count_} = np.array(self._rs.<attr>)`. Verification (model B): `tests/divergence_extras.py::test_{multinomial,bernoulli,complement}_discrete_nb_fitted_attrs_match_sklearn` fit BOTH ferrolearn and the live sklearn 1.5.2 oracle on fixtures and `np.testing.assert_allclose` all four attrs (atol 1e-7). |
| REQ-MISSING-METHODS (coef_/predict_proba/inverse_transform/cluster_centers_) | NOT-STARTED | open prereq owned downstream by the eight library crates. The `_Rs*` classes expose ONLY `fit`/`predict` (regressors/classifiers), `fit`/`transform` (transformers), `fit` + `labels_` (+`predict` for MiniBatchKMeans/GMM) — no `coef_`/`feature_importances_` (regressors), no `predict_proba`/`decision_function` (classifiers), no `inverse_transform`/`components_` (transformers), no `cluster_centers_`/`children_`/`n_clusters_` (clusterers). The discrete-NB fitted attrs (`feature_log_prob_`/`class_log_prior_`/`feature_count_`/`class_count_`) are now SHIPPED separately (REQ-DISCRETE-NB-FITTED-ATTRS, #2103). The remaining wrappers set only `n_features_in_`/`classes_`/`labels_`. sklearn exposes these across all routed upstream files. The binding cannot expose attrs/methods the fitted library types do not compute — owned downstream (`ferrolearn_linear`/`_tree`/`_bayes`/`_neighbors`/`_cluster`/`_decomp`/`_preprocess`/`_kernel`). |
| REQ-MISSING-PARAMS (full constructor surface) | NOT-STARTED | open prereq owned downstream by the eight library crates. Each `_Rs*` constructor binds a THIN subset of sklearn's params (e.g. `RsRandomForestRegressor` → `n_estimators`/`max_depth`/`min_samples_split`/`min_samples_leaf`/`random_state` vs sklearn's full `criterion`/`max_features`/`bootstrap`/`oob_score`/`n_jobs`/... ; `RsBaggingClassifier` lacks the `estimator` base-estimator knob; the no-knob scalers/kernels lack `feature_range`/`quantile_range`/`method`/`kernel`/`gamma`/`n_components`). sklearn's full surface is in the routed upstream `__init__`s. The binding cannot expose params the library builders (`with_*`) do not accept — owned downstream. |
| REQ-KNR-CTOR-SURFACE (KNeighborsRegressor full constructor) | SHIPPED | FIXED #2147. `ferrolearn.KNeighborsRegressor` now exposes sklearn's full `(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)` surface (`sklearn/neighbors/_regression.py:178-189`), where `weights='distance'` CHANGES predictions (inverse-distance-weighted neighbor-target average, `_regression.py:43-45`). Impl: the thin `py_regressor!(RsKNeighborsRegressor, ... (n_neighbors: usize = 5) ...)` invocation was replaced by a HAND-WRITTEN `#[pyclass(name = "_RsKNeighborsRegressor")]` in `extras.rs` mirroring `RsKNeighborsClassifier` (`classifiers.rs`): `new` takes all 7 result/ABI params (`metric_params` is wrapper-validated), `fit(x, y: PyReadonlyArray1<f64>)` maps `weights→Weights::{Uniform,Distance}` and `algorithm→Algorithm::{Auto,BruteForce,KdTree}` (`ball_tree→Auto`), rejects `p!=2.0`/`metric∉{minkowski,euclidean}` with `PyNotImplementedError(... NOT-STARTED #876)`, builds `ferrolearn_neighbors::KNeighborsRegressor::<f64>::new().with_n_neighbors().with_weights().with_algorithm()` (`knn.rs:1160-1175`), and `predict→PyArray1<f64>` — same fit/predict ABI the macro emitted. `leaf_size`/`n_jobs` are stored as per-field `#[allow(dead_code, reason="...ABI-parity no-op...")]` knobs (`_regression.py:184,188`). The Rust neighbors core was UNTOUCHED (capability pre-existing, audited `ferrolearn_neighbors` REQ table). Non-test production consumer: `_extras.py::KNeighborsRegressor(_RegressorPickleMixin, RegressorMixin, BaseEstimator)` — full shim mirroring `_classifiers.py::KNeighborsClassifier` (`__init__` stores all 8 params; `_make_rs` rejects callable `weights`/non-None `metric_params` with `NotImplementedError #876` then constructs `_RsKNeighborsRegressor`; `fit`/`predict` validate float64, set `n_features_in_`, `_store_training_data` for pickle-rebuild; inherits `RegressorMixin.score`); `ferrolearn/__init__.py:41,100` re-export; `lib.rs:42` registration (name `_RsKNeighborsRegressor` preserved, registration unchanged). Verification (model B): `tests/divergence_extras.py::test_knr_*` (14 cases, live sklearn 1.5.2 oracle, R-CHAR-3) — `weights={uniform,distance}` predict parity (rtol/atol 1e-9), distance-weighting CHANGES the prediction, `algorithm∈{auto,brute,kd_tree,ball_tree}` identical predict, `get_params`/`set_params`/`clone` 8-param round-trip, unsupported `p=3`/`metric='manhattan'`/callable-weights/`metric_params` raise `NotImplementedError` (sklearn oracle accepts → explicit honest divergence #876), pickle round-trip preserves distance predictions. (REQ-MISSING-PARAMS remains NOT-STARTED for the OTHER estimators whose library builders lack the `with_*` knobs.) |
| REQ-OMP-CTOR-SURFACE (OrthogonalMatchingPursuit full constructor + fitted attrs) | SHIPPED | FIXED #2172. `ferrolearn.OrthogonalMatchingPursuit` now mirrors `sklearn.linear_model.OrthogonalMatchingPursuit` (`sklearn/linear_model/_omp.py:645-753`), exposing sklearn's full keyword-only constructor `(n_nonzero_coefs=None, tol=None, fit_intercept=True, precompute='auto')` (`_omp.py:742-753`, `_parameter_constraints` `:735-740`) plus the fitted `coef_` (shape `(n_features,)`) / `intercept_` (scalar) / `n_features_in_` attrs (`_omp.py:814-815`). Impl: hand-written `#[pyclass(name = "_RsOrthogonalMatchingPursuit")]` in `extras.rs` (`new` takes the 4 params; threads `n_nonzero_coefs`/`tol` into the core builder ONLY when `Some`, preserving the core's `None`/`None` default `max(int(0.1*n_features),1)` path, `omp.rs` Fit mirroring `_omp.py:785`; `fit(x, y: PyReadonlyArray1<f64>)` / `predict → PyArray1<f64>`; `#[getter] coef_`/`intercept_` over `HasCoefficients`; no `unwrap`/`expect`/`panic` — not-fitted → `PyRuntimeError`). `precompute` (`'auto'`/`True`/`False`, `_omp.py:739`) is a Gram-matrix speed knob that does NOT change the OMP solution (`_omp.py:791-813`); the core never uses a Gram path, so the binding ACCEPTS any value and ignores it (held on the `_extras.py` wrapper for `get_params`/`clone`). The Rust OMP core was UNTOUCHED (capability pre-existing, audited `ferrolearn_linear` omp.rs REQ-1/2/5 SHIPPED; coef_/intercept_ match sklearn ~1e-12). `n_iter_`/`n_nonzero_coefs_` stay NOT-STARTED — the core does not compute them (`omp.rs` REQ-7 #491). Non-test production consumer: `_extras.py::OrthogonalMatchingPursuit(_RegressorPickleMixin, RegressorMixin, BaseEstimator)` — `__init__` stores the 4 sklearn-default params, `_make_rs` constructs `_RsOrthogonalMatchingPursuit`, `fit` validates float64 + sets `n_features_in_`/`coef_`/`intercept_` + `_store_training_data` for pickle-rebuild, `predict`, inherits `RegressorMixin.score`; `ferrolearn/__init__.py` re-exports `OrthogonalMatchingPursuit`; `lib.rs` registers `RsOrthogonalMatchingPursuit`. Verification (model B): `tests/divergence_extras.py::test_omp_*` (9 tests, live sklearn 1.5.2 oracle, R-CHAR-3) — coef_/intercept_/predict parity across 5 `n_nonzero_coefs` (atol 1e-10), default-`None` 0.1·n_features path, `tol` path, `fit_intercept` True/False, `precompute='auto'/False/True` accepted+result-invariant, 4-param keyword-only ctor ABI/get_params/clone matching the sklearn key set, `score`/`n_features_in_`, pickle round-trip. |
| REQ-LARS-CTOR-SURFACE (Lars full constructor + fitted attrs) | SHIPPED | FIXED #2174. `ferrolearn.Lars` mirrors `sklearn.linear_model.Lars` (`sklearn/linear_model/_least_angle.py:922-1068`), exposing sklearn's full keyword-only constructor `(fit_intercept=True, verbose=False, precompute='auto', n_nonzero_coefs=500, eps=np.finfo(float).eps, copy_X=True, fit_path=True, jitter=None, random_state=None)` (`_least_angle.py:1047-1058`, `_parameter_constraints` `:1032-1042`) plus the fitted `coef_` (shape `(n_features,)`) / `intercept_` (scalar) / `n_features_in_` attrs (`_least_angle.py:1125`). Impl: hand-written `#[pyclass(name = "_RsLars")]` in `extras.rs` (`new` takes `n_nonzero_coefs`/`fit_intercept`; `fit(x, y: PyReadonlyArray1<f64>)` → `ferrolearn_linear::Lars::<f64>::new().with_n_nonzero_coefs().with_fit_intercept()` (`lars.rs`); `predict → PyArray1<f64>`; `#[getter] coef_`/`intercept_` over `HasCoefficients`; no `unwrap`/`expect`/`panic` — not-fitted → `PyRuntimeError`). Only `fit_intercept`/`n_nonzero_coefs` change the supported result and are threaded into the core; the other seven sklearn params are accept-and-ignore on the supported path — `verbose`/`copy_X` (diagnostics/copy), `precompute` ('auto'/bool/ndarray/None — Gram-matrix speed knob, `_get_gram` `_least_angle.py:1070-1079`, no result change), `eps` (Cholesky floor `:1037`), `fit_path` (final `coef_` identical, `:1133-1151`), `random_state` (only consumed by `jitter`). The Rust LARS core was UNTOUCHED (capability pre-existing, audited `ferrolearn_linear` lars.rs REQ-1/3/4 SHIPPED; coef_/intercept_ match sklearn ~1e-6). `jitter != None` (seeded gaussian noise on y, `:1170-1175`, R-SUBSTRATE-5) → honest `NotImplementedError` (#2174). `n_nonzero_coefs` (default 500) is an UPPER bound capped at `n_features` (sklearn `max_features`); the wrapper passes `min(n_nonzero_coefs, n_features)` to the core (which errors above `n_features`), reproducing sklearn's cap. Non-test production consumer: `_extras.py::Lars(_RegressorPickleMixin, RegressorMixin, BaseEstimator)` — `__init__` stores the 9 sklearn-default params; `_validate` mirrors `_parameter_constraints` (n_nonzero_coefs>=1 / eps>=0 / precompute set → `ValueError`; jitter non-None → `NotImplementedError`); `_make_rs` clamps + constructs `_RsLars`; `fit` validates float64 + sets `n_features_in_`/`coef_`/`intercept_` + `_store_training_data` for pickle-rebuild; `predict`; inherits `RegressorMixin.score`; `ferrolearn/__init__.py` re-exports `Lars`; `lib.rs` registers `RsLars`. Verification (model B): `tests/divergence_extras.py::test_lars_*` (live sklearn 1.5.2 oracle, R-CHAR-3) — coef_/intercept_/predict parity per `n_nonzero_coefs∈{1,2,3,5}` (atol 1e-6) + default-500-capped-at-n_features path, fit_intercept True/False, precompute='auto'/False/True/None accepted+result-invariant, 9-param keyword-only ctor ABI matching the sklearn signature, get_params/clone key-set parity, n_nonzero_coefs<1/bad-precompute → ValueError matching sklearn, jitter!=None → NotImplementedError (sklearn fits → honest divergence #2174), score/n_features_in_, pickle round-trip, copy_X non-bool → ValueError (#2177). NOT-STARTED (Rust LARS **core**, not the binding; `xfail`-pinned in `divergence_lars.py`): deep-path `coef_` numerical blowup when active features ≳ n_features (#2175) and collinear/duplicate-column handling (#2176 — sklearn `lars_path` drops the tied atom and fits; ferrolearn errors on a singular LARS Gram). The binding faithfully surfaces the core; these are pre-existing core-LARS limitations, separate units. |
| REQ-LASSOLARS-CTOR-SURFACE (LassoLars full constructor + fitted attrs) | SHIPPED | FIXED #2174. `ferrolearn.LassoLars` mirrors `sklearn.linear_model.LassoLars` (`sklearn/linear_model/_least_angle.py:1212-1388`), exposing sklearn's full constructor `(alpha=1.0, *, fit_intercept=True, verbose=False, precompute='auto', max_iter=500, eps=..., copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None)` — `alpha` positional-or-keyword FIRST, rest keyword-only (`_least_angle.py:1363-1376`, `_parameter_constraints` `:1353-1359`) — plus the fitted `coef_`/`intercept_`/`n_features_in_` attrs. Impl: hand-written `#[pyclass(name = "_RsLassoLars")]` in `extras.rs` (`new` takes `alpha`/`max_iter`/`fit_intercept`; `fit` → `ferrolearn_linear::LassoLars::<f64>::new().with_alpha().with_max_iter().with_fit_intercept()` (`lars.rs`); `predict → PyArray1<f64>`; `#[getter] coef_`/`intercept_`; no panic — not-fitted → `PyRuntimeError`). `alpha`/`max_iter`/`fit_intercept` thread into the core; `verbose`/`precompute`/`eps`/`copy_X`/`fit_path`/`random_state` are the same accept-and-ignore set as `Lars`. The Rust LARS-Lasso core was UNTOUCHED (audited `ferrolearn_linear` lars.rs REQ-2/3/4 SHIPPED; coef_/intercept_ match sklearn ~1e-6). `positive=True` (`:1357`, non-negative-coefficient constraint — a DIFFERENT optimization the Rust core lacks, `lars.rs` has no `positive` builder) → honest `NotImplementedError` (NOT-STARTED #2174); `positive=False` (default) is the supported path. `jitter != None` → `NotImplementedError` (R-SUBSTRATE-5; #2174). Non-test production consumer: `_extras.py::LassoLars(_RegressorPickleMixin, RegressorMixin, BaseEstimator)` — `__init__` stores the 11 params (alpha positional first); `_validate` mirrors `_parameter_constraints` (alpha>=0 / max_iter>=0 / positive boolean / eps>=0 / precompute set → `ValueError`; positive=True & jitter non-None → `NotImplementedError`); `_make_rs` constructs `_RsLassoLars`; `fit` validates float64 + sets `n_features_in_`/`coef_`/`intercept_` + `_store_training_data`; `predict`; inherits `RegressorMixin.score`; `ferrolearn/__init__.py` re-exports `LassoLars`; `lib.rs` registers `RsLassoLars`. Verification (model B): `tests/divergence_extras.py::test_lasso_lars_*` + `test_lars_score_*`/`test_lars_pickle_*` (live sklearn 1.5.2 oracle, R-CHAR-3) — coef_/intercept_/predict parity per `alpha∈{0.1,0.5,1.0,2.0}` (atol 1e-6), `alpha` positional-first ABI matching sklearn, fit_intercept True/False, get_params/clone key-set parity, negative-alpha → ValueError matching sklearn, positive=True / jitter!=None → NotImplementedError (sklearn fits → honest divergence #2174), score/n_features_in_, pickle round-trip, copy_X non-bool → ValueError (#2177). NOT-STARTED (Rust LARS **core**, not the binding; `xfail`-pinned): collinear/duplicate-column handling (#2176 — sklearn drops the tied atom; ferrolearn errors on a singular LARS Gram). The binding faithfully surfaces the core. |
| REQ-IFOREST-BINDING (IsolationForest outlier detector) | SHIPPED | FIXED #2180. `ferrolearn.IsolationForest` mirrors `sklearn.ensemble.IsolationForest` (`sklearn/ensemble/_iforest.py:221-248`), exposing sklearn's full keyword-only constructor `(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)` plus the `OutlierMixin` method surface — `predict` (±1, `_iforest.py:357-378`), `score_samples` (∈[-1,0], `_iforest.py:412-451`), `decision_function` (`=score_samples-offset_`, `_iforest.py:380-410`), `fit_predict` — and the fitted `offset_` (`_iforest.py:344/353`) / `n_features_in_` attrs. Impl: hand-written `#[pyclass(name = "_RsIsolationForest")]` in `extras.rs` over `ferrolearn_tree::IsolationForest<f64>`/`FittedIsolationForest<f64>` (`new` takes resolved `n_estimators`/`max_samples: usize`/`contamination: Option<f64>`/`random_state`; `fit(x)` → `IsolationForest::new().with_n_estimators().with_max_samples().{with_contamination_auto()|with_contamination(c)}` (isolation_forest.rs); `predict→PyArray1<i64>` mapping the core `Array1<isize>`; `score_samples`/`decision_function`→`PyArray1<f64>`; `#[getter] offset_`/`n_features_in_` over `offset()`/`n_features()`; no `unwrap`/`expect`/`panic` — not-fitted → `PyRuntimeError`). The Rust isolation-forest core was UNTOUCHED (capability pre-existing, audited `ferrolearn_tree` isolation_forest.rs REQ-1..9 SHIPPED). The `_extras.py::IsolationForest(_RegressorPickleMixin, OutlierMixin, BaseEstimator)` wrapper resolves `max_samples` ('auto'→`min(256,n)`, int→as-is, float→`int(f*n)`, `_iforest.py:303-318`) and `contamination` ('auto'→None ABI / float→`c`, `_iforest.py:341-353`) BEFORE the ABI; `_validate_static` mirrors `_parameter_constraints` (`_iforest.py:199-219`): `n_estimators<1`/`contamination∉(0,0.5]`/`max_samples` int`<1`/float∉(0,1]/bad-string → `ValueError`; `max_features!=1.0`/`bootstrap=True`/`warm_start=True` → honest `NotImplementedError` (core has only the full-feature, with-replacement subsample path, isolation_forest.rs REQ-7a/7b NOT-STARTED #728/#729); `n_jobs`/`verbose` accept-and-ignore. **RNG-substrate divergence (#2118 / isolation_forest.rs RNG-boundary #730, R-SUBSTRATE-5):** the core's subsample+split draws use `StdRng`, NOT numpy MT19937, so for a GIVEN `random_state` the exact `score_samples` values DIFFER from sklearn (verified: max\|Δ\|≈0.09, predict agreement≈0.89 on 45×2). Exact score parity is NOT asserted (would be fragile); the STRUCTURAL contract IS asserted and matches the live sklearn oracle. Non-test production consumer: `_extras.py::IsolationForest` (`_make_rs`/`fit`/`predict`/`score_samples`/`decision_function`/`fit_predict` + pickle-refit) + `ferrolearn/__init__.py` re-export + `lib.rs` `m.add_class::<extras::RsIsolationForest>()`. Verification (model B): `tests/divergence_extras.py::test_iforest_*` (16 tests, live sklearn 1.5.2 oracle, R-CHAR-3) — get_params/clone key-set parity, predict∈{-1,+1}/shape, score_samples∈[-1,0], decision_function==score-offset_, predict==where(df<0,-1,1), contamination='auto'→offset_=-0.5 (matches oracle), contamination float→offset_=percentile, anomaly ranking (outliers<inliers, both ferrolearn AND oracle), n_features_in_, fit_predict==fit().predict(), max_samples 'auto'/int/float resolution, invalid params→ValueError (matching oracle), max_features/bootstrap/warm_start→NotImplementedError, n_jobs/verbose accept-ignore, not-fitted→NotFittedError, pickle round-trip, AND an explicit honest pin that exact score_samples DIVERGE from sklearn (RNG substrate). |
| REQ-FEATAGGLOM-BINDING (FeatureAgglomeration transformer + dendrogram attrs, REQ-10 #943) | SHIPPED | FIXED #943. `ferrolearn.FeatureAgglomeration` mirrors `sklearn.cluster.FeatureAgglomeration` (`sklearn/cluster/_agglomerative.py:1296-1346` + `sklearn/cluster/_feature_agglomeration.py:22-92`), exposing the keyword-defaulted constructor `(n_clusters=2, *, metric=None, memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', pooling_func=np.mean, distance_threshold=None, compute_distances=False)` (`_agglomerative.py:1296`, `n_clusters` positional-or-keyword), the `TransformerMixin` `fit`/`transform`/`fit_transform` surface PLUS `inverse_transform`, and the dendrogram fitted attrs `labels_`/`n_clusters_`/`children_`/`distances_`/`n_leaves_`/`n_connected_components_`/`n_features_in_` delegated from the inner clustering over `X.T` (`_agglomerative.py:1083-1095/1338-1340`). Impl: hand-written `#[pyclass(name = "_RsFeatureAgglomeration")]` in `extras.rs` over `ferrolearn_cluster::FeatureAgglomeration<f64>`/`FittedFeatureAgglomeration<f64>` (`new` takes `n_clusters: usize`/`linkage: String`/`pooling: String`/`compute_distances: bool`; `fit(x)` maps the linkage/pooling strings to `AgglomerativeLinkage::{Ward,Complete,Average,Single}`/`PoolingFunc::{Mean,Max}` then builds `FeatureAgglomeration::new(n_clusters).with_linkage().with_pooling_func().with_compute_distances()`; `transform`/`inverse_transform`→`PyArray2<f64>` via `ndarray2_to_numpy`; `#[getter] labels_`→`PyArray1<i64>` (`ndarray1_usize_to_numpy(fitted.labels())`); `children_`→`(n_features-1, 2)` `PyArray2<i64>` built from `fitted.children()`; `distances_`→`Option<PyArray1<f64>>` (None when `compute_distances=False`); `n_clusters_`/`n_leaves_`/`n_connected_components_`/`n_features_in_`→`usize`; no `unwrap`/`expect`/`panic` — not-fitted → `PyRuntimeError`, FerroError → `PyValueError`, unknown linkage/pooling string → `PyValueError`). The Rust feature-agglomeration core was UNTOUCHED (capability pre-existing/audited, `ferrolearn_cluster` feature_agglomeration.rs REQ-1..9 SHIPPED — `fit`/`transform`/`inverse_transform`/`labels`/`children`/`distances`/`n_leaves`/`n_connected_components`). The `_extras.py::FeatureAgglomeration(_TransformerWrapper)` wrapper resolves `pooling_func` BEFORE the ABI — the strings `'mean'`/`'max'` OR the numpy callables `np.mean`/`np.max` map to the enum; ANY OTHER callable raises `NotImplementedError` (arbitrary-callable pooling, `_agglomerative.py:1291` `[callable]`, REQ-7 NOT-STARTED #941); an unknown string raises `ValueError` (sklearn `InvalidParameterError ⊂ ValueError`). The core-unsupported params `metric`/`connectivity`/`distance_threshold` raise `NotImplementedError` when set NON-default (honest gap-surfacing, esp. `distance_threshold` which would change the cut, REQ-6 NOT-STARTED #941); `memory`/`compute_full_tree='auto'` are accept-and-ignore. `fit` sets the dendrogram attrs from the `_Rs*` getters. **VALUE parity (NOT just structural):** `labels_` integer-EXACT, `transform`/`inverse_transform`/`fit_transform` ≈ sklearn (atol 1e-9), `children_` int-equal, `distances_` ≈ sklearn (atol 1e-9), all across {ward,complete,average,single} × {mean,max} — because the core delegates to the bit-exact `AgglomerativeClustering._hc_cut` over `X.T`. Non-test production consumer (R-DEFER-1): `_extras.py::FeatureAgglomeration` (`_make_rs`/`fit`/`transform`/`inverse_transform` + the dendrogram-attr exposure) + `ferrolearn/__init__.py` re-export (`FeatureAgglomeration` in the import block + `__all__`) + `lib.rs` `m.add_class::<extras::RsFeatureAgglomeration>()`. Verification (model B): `tests/test_feature_agglomeration.py` (21 tests, live sklearn 1.5.2 oracle, R-CHAR-3) — `labels_`/`transform`/`inverse_transform` parity per linkage×pooling (8 param-pairs), `children_`/`distances_`/`n_leaves_`/`n_connected_components_`/`n_clusters_` per linkage, `distances_` None when not computed (sklearn omits the attr), `fit_transform == fit().transform() ==` sklearn, `'mean'`/`np.mean` and `'max'`/`np.max` equivalent, `get_params`/`set_params` round-trip + default `n_clusters == 2`, `n_clusters` positional, arbitrary-callable pooling → `NotImplementedError`, unknown pooling string → `ValueError`, `distance_threshold` → `NotImplementedError`, 1-feature `X` → `ValueError` (matching sklearn `ensure_min_features=2`). REQ-6 (`distance_threshold`/`metric`/`connectivity`/`compute_full_tree`) and REQ-7 (arbitrary-callable `pooling_func`) remain NOT-STARTED in `ferrolearn_cluster` feature_agglomeration.rs (open prereq blocker #941); the binding honestly rejects them rather than silently ignoring. |
| REQ-VALUE-PARITY-RNG (seeded stochastic parity) | NOT-STARTED | open prereq owned downstream / upstream in ferray (R-SUBSTRATE-5: numpy-RNG stream). The stochastic estimators (`RandomForestRegressor`/`ExtraTreesRegressor`/`ExtraTreesClassifier`/`GradientBoosting*`/`HistGradientBoosting*`/`BaggingClassifier`/`AdaBoostClassifier`/`MiniBatchKMeans`/`GaussianMixture`/`FastICA`/`NMF`) pass `random_state: Option<u64>` to a non-numpy RNG, so a shared `random_state` will NOT reproduce sklearn's bootstrap/init draws (sklearn uses numpy's Mersenne-Twister/PCG64). Bit-exact seeded parity needs `ferray::random` mirroring numpy's stream — owned in ferray and the owning library crates; until then seeded outputs diverge. (Default deterministic-path API conformance is SHIPPED above.) | The stochastic estimators (`RandomForestRegressor`/`ExtraTreesRegressor`/`ExtraTreesClassifier`/`GradientBoosting*`/`HistGradientBoosting*`/`BaggingClassifier`/`AdaBoostClassifier`/`MiniBatchKMeans`/`GaussianMixture`/`FastICA`/`NMF`) pass `random_state: Option<u64>` to a non-numpy RNG, so a shared `random_state` will NOT reproduce sklearn's bootstrap/init draws (sklearn uses numpy's Mersenne-Twister/PCG64). Bit-exact seeded parity needs `ferray::random` mirroring numpy's stream — owned in ferray and the owning library crates; until then seeded outputs diverge. (Default deterministic-path API conformance is SHIPPED above.) |
| REQ-CONSUMER (binding IS the public API) | SHIPPED | the binding boundary types ARE the public API (R-DEFER-1/S5: boundary estimator types ARE the public surface; grandfathered existing pub API across prior commits). Non-test production consumers: `_extras.py` `_RegressorWrapper`/`_ClassifierWrapper`/`_ClusterWrapper`/`_TransformerWrapper` subclasses + `GaussianMixture` (each constructs its `_Rs*` class via `_make_rs` and drives fit/predict/transform — `grep -n "_Rs" python/ferrolearn/_extras.py`); `ferrolearn/__init__.py:16` re-exports all ~40 (`__all__`); `lib.rs:27-88` registers every `_Rs*` via `m.add_class::<extras::Rs*>()`; the head-to-head bench harness (`ferrolearn-bench/src/bin/harness.rs`, `ferrolearn-bench/sklearn_bench.py`) drives them vs sklearn. Verification (model B): pytest `tests/divergence_*.py` + external users. |
| REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | open prereq blocker = `conversions.md` REQ-FERRAY #2027. `extras.rs:7-10` marshals via `use crate::conversions::*` + `use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2}` + `use ndarray::Array1` (rust-numpy → `ndarray::Array{1,2}`) — the WRONG substrate per R-SUBSTRATE-1 (destination `ferray::numpy_interop` + `ferray-core`). ferray exposes no PyO3 numpy-interop bridge consumable here yet (R-SUBSTRATE-5). Owned by the conversions unit, surfaced here. |

## Architecture

`extras.rs` is built from three declarative `macro_rules!` shells plus
hand-written `#[pyclass]` blocks for estimators carrying extra state:

- **`py_regressor!`** (`extras.rs`) emits a `#[pyclass]` with the declared
  hyperparameter fields + `fitted: Option<$fitted_path>`, a `#[new]` with a
  `#[pyo3(signature = (...))]` defaulting each field, a `fit(x:
  PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>)` that builds the unfitted
  model from `$build_block`, coerces X/y via `numpy2_to_ndarray`/
  `numpy1_to_ndarray`, calls `Fit::fit`, and maps `FerroError → PyValueError`;
  and `predict → PyArray1<f64>` returning `PyRuntimeError("not fitted")` before
  fit. Used for the 7 hyperparameter-simple regressors.
- **`py_classifier!`** is identical except `y: PyReadonlyArray1<i64>` decoded via
  `numpy1_to_ndarray_usize` and `predict → PyArray1<i64>` — the Rust core works
  in `usize` labels; the Python `_ClassifierWrapper._encode`/decode handles
  arbitrary label dtypes.
- **`py_transformer!`** emits `fit(x: PyReadonlyArray2<f64>)` over
  `Fit::fit(&x_nd, &())` and `transform → PyArray2<f64>`.
- **Hand-written blocks** cover estimators with `random_state: Option<u64>`
  (the ensembles, `MiniBatchKMeans`, `GaussianMixture`) — applied conditionally
  via `if let Some(s) = self.random_state { m = m.with_random_state(s); }` — and
  the clusterers, which expose a `#[getter] fn labels_` over `f.labels()` instead
  of a supervised `predict`. `RsDBSCAN` maps its signed `i64` noise labels via
  `lbls.mapv(|v| v as i64)`; the others use `ndarray1_usize_to_numpy`.

`_extras.py` mirrors this with four base wrappers over sklearn mixins
(`_RegressorWrapper`/`_ClassifierWrapper`/`_ClusterWrapper`/`_TransformerWrapper`
+ `BaseEstimator`), each `_make_rs()`-constructing its `_Rs*` class with
keyword args. EVERY wrapper `__init__` is `def __init__(self, *, ...)` —
keyword-only — which is the source of REQ-CTOR-ABI-POSITIONAL (the 16 cases
where sklearn's primary param is positional-or-keyword). The `_ClassifierWrapper`
adds the `_encode` round-trip (`np.unique` + `np.searchsorted` → `classes_`;
decode via `self.classes_[y_enc]`). `GaussianMixture` is hand-written (sklearn
places it in `sklearn.mixture`, fit/predict/labels_ style).

Two binding-hygiene defects sit at the file head, independent of the math:
`extras.rs:5` `#![allow(non_snake_case)]` (module-root allow, R-CODE-3 —
REQ-MODULE-ALLOW) and the "Phase 2 binding expansion" / "Phase-2 binding
wrappers" framing in `extras.rs:1` + `_extras.py:1` (R-DEFER-4 —
REQ-PHASE-FRAMING). Both are fixed entirely within this unit (no downstream
dependency); REQ-MODULE-ALLOW is Rust-side, REQ-PHASE-FRAMING is doc-comment +
docstring.

All estimator *correctness* — full param surface, fitted attributes, seeded RNG
parity, decomp auto-`n_components` — lives DOWN in the eight library crates,
each with its own audited `//!` REQ status table; this shim references those
generically rather than re-filing per-estimator blockers.

## Verification

Model B (pytest vs `import sklearn` 1.5.2). Rebuild before pytest sees a Rust
change:

```bash
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/ -q          # divergence_{regressors,classifiers,clusterers,transformers}.py
cargo clippy -p ferrolearn-python --all-targets -- -D warnings
```

The SHIPPED API-conformance claims are established by the per-category
`tests/divergence_*.py` pins + the AC probes above. The three HEADLINE
NOT-STARTED REQs are established by their live oracle commands (run from `/tmp`,
R-CHAR-3):

```bash
# REQ-CTOR-ABI-POSITIONAL — sklearn accepts positional, ferrolearn does not
cd /tmp && python3 -c "import inspect; from sklearn.ensemble import RandomForestRegressor as R; print(inspect.signature(R.__init__).parameters['n_estimators'].kind.name)"   # POSITIONAL_OR_KEYWORD
cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -c "import ferrolearn as f; f.RandomForestRegressor(50)"   # TypeError (live)

# REQ-MODULE-ALLOW — module-root allow present
grep -n "#!\[allow" /home/doll/ferrolearn/ferrolearn-python/src/extras.rs    # 5:#![allow(non_snake_case)]

# REQ-PHASE-FRAMING — phase framing present
grep -niE "phase [0-9]" /home/doll/ferrolearn/ferrolearn-python/src/extras.rs /home/doll/ferrolearn/ferrolearn-python/python/ferrolearn/_extras.py

# REQ-DECOMP-NCOMPONENTS-DEFAULT — default 2 vs None
cd /tmp && python3 -c "import inspect; from sklearn.decomposition import FastICA; print(inspect.signature(FastICA.__init__).parameters['n_components'].default)"   # None
cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -c "import ferrolearn as f; print(f.FastICA().n_components)"   # 2
```

Each NOT-STARTED REQ closes only when its fix lands AND the pinned pytest goes
green (R-DEFER-3). The downstream-owned REQs (value/method/param/RNG/substrate)
close in their owning crates' iterations; this binding doc references them rather
than re-filing per-estimator blockers (S8 won't-fix-on-noise).
