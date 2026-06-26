# sklearn vs ferrolearn Gap Report

Generated 2026-06-25 and updated 2026-06-26 from the current worktree.

This replaces the stale 2026-03-25 report. The old report materially
understated the current ferrolearn surface: many items it marked missing now
exist in source, including `CategoricalNB`, radius neighbors, `LocalOutlierFactor`,
`NearestCentroid`, covariance estimators, Gaussian processes, neural-network
estimators, many linear CV estimators, ExtraTrees/Bagging/Voting, and much of
model selection.

## Evidence and Scope

- sklearn baseline: official stable scikit-learn API reference, version 1.9.0:
  <https://scikit-learn.org/stable/api/index.html>.
- Local oracle runtime: `sklearn 1.5.2` is installed in this workspace and is
  the version cited by most current `divergence_*` tests.
- Local sklearn source mirror: `.sklearn-ref/scikit-learn` at commit `f1cc4e7`.
- ferrolearn workspace: `Cargo.toml` lists 22 workspace members.
- Test evidence: the current tree contains 346 `tests/divergence_*.rs` files.

The exact API gap list below was produced by parsing the sklearn 1.9.0 API
index for public classes/functions and comparing it with current public Rust
symbols in `ferrolearn-*/src`. It is an exact-name/public-surface comparison,
not a crate-path compatibility check: a present item may live in a different
ferrolearn crate/module than sklearn's Python module. Obvious aliases are
normalized (`HDBSCAN`/`Hdbscan`, `TSNE`/`Tsne`, `LocallyLinearEmbedding`/`LLE`,
`KDTree`/`KdTree`, `LinearDiscriminantAnalysis`/`LDA`,
`QuadraticDiscriminantAnalysis`/`QDA`). Presence does not prove behavioral
parity.

## Summary

ferrolearn is now broad but still not sklearn-parity complete.

- Scoped ML-facing sklearn API gaps: 51 exact public items missing across the
  modules listed below.
- Whole sklearn infrastructure areas are not counted in that exact gap count: callbacks,
  frozen estimators, full `sklearn.base` estimator protocol, `sklearn.utils`,
  display/plotting classes, metadata routing, and Python object compatibility.
- Existing Rust implementations frequently expose a narrower or Rust-idiomatic
  API: fewer constructor parameters, fewer fitted attributes, different error
  types/messages, less sparse/object/string/dataframe support, less `sample_weight`
  and `multioutput` support, and incomplete `get_feature_names_out` /
  `feature_names_in_` behavior.
- Many exact-value differences are documented as boundaries around numpy RNG,
  BLAS/LAPACK summation order, scipy solver choices, or deliberate Rust API
  choices. These are still parity gaps when the target is sklearn behavior.

## Exact sklearn API Gaps

These names are present in sklearn 1.9.0 and absent from ferrolearn's current
public Rust surface after the alias normalization above.

| sklearn module | Missing public items |
|---|---|
| `sklearn.linear_model` | `LarsCV`, `LassoLarsCV`, `TheilSenRegressor` |
| `sklearn.tree` | `export_graphviz`, `plot_tree` |
| `sklearn.ensemble` | `StackingClassifier`, `StackingRegressor` |
| `sklearn.neighbors` | `KNeighborsTransformer`, `KernelDensity`, `NeighborhoodComponentsAnalysis`, `RadiusNeighborsTransformer` |
| `sklearn.cluster` | `SpectralBiclustering`, `SpectralCoclustering` |
| `sklearn.decomposition` | `MiniBatchDictionaryLearning`, `MiniBatchSparsePCA`, `SparseCoder`, `dict_learning`, `dict_learning_online` |
| `sklearn.feature_selection` | `mutual_info_classif`, `mutual_info_regression` |
| `sklearn.feature_extraction` | `DictVectorizer`, `FeatureHasher` |
| `sklearn.feature_extraction.text` | `HashingVectorizer` |
| `sklearn.metrics` | `confusion_matrix_at_thresholds`, `metric_at_thresholds`, `ConfusionMatrixDisplay`, `DetCurveDisplay`, `PrecisionRecallDisplay`, `PredictionErrorDisplay`, `RocCurveDisplay` |
| `sklearn.model_selection` | `LearningCurveDisplay`, `ValidationCurveDisplay` |
| `sklearn.calibration` | `CalibrationDisplay` |
| `sklearn.kernel_approximation` | `AdditiveChi2Sampler`, `PolynomialCountSketch`, `SkewedChi2Sampler` |
| `sklearn.gaussian_process.kernels` | `CompoundKernel`, `ExpSineSquared`, `Exponentiation`, `Hyperparameter`, `RationalQuadratic` |
| `sklearn.datasets` | `fetch_20newsgroups_vectorized`, `fetch_lfw_pairs`, `fetch_lfw_people`, `fetch_olivetti_faces`, `fetch_rcv1`, `fetch_species_distributions`, `load_sample_image`, `load_sample_images` |
| `sklearn.inspection` | `DecisionBoundaryDisplay`, `PartialDependenceDisplay` |

The following scoped modules had no exact public-item miss in this pass:
`sklearn.svm`, `sklearn.naive_bayes`, `sklearn.mixture`, `sklearn.cross_decomposition`,
`sklearn.discriminant_analysis`, `sklearn.pipeline`, `sklearn.preprocessing`,
`sklearn.impute`, `sklearn.random_projection`, `sklearn.compose`,
`sklearn.feature_extraction.image`, `sklearn.semi_supervised`,
`sklearn.metrics.pairwise`, `sklearn.manifold`, `sklearn.kernel_ridge`,
`sklearn.gaussian_process`, `sklearn.covariance`, `sklearn.neural_network`, `sklearn.dummy`,
`sklearn.multiclass`, `sklearn.multioutput`, and `sklearn.isotonic`. This means
only that names exist; it is not a value- or contract-parity claim.

## Whole Areas Outside the Exact Gap Count

These are sklearn public areas with no full ferrolearn equivalent or no Python
protocol equivalent:

- `sklearn.callback`: `AutoPropagatedCallback`, `CallbackContext`,
  `CallbackSupportMixin`, `FitCallback`, `ProgressBar`, `ScoringMonitor`,
  `ScoringMonitorLog`, `with_callbacks`.
- `sklearn.frozen`: `FrozenEstimator`.
- `sklearn.exceptions`: sklearn's public warning/error class family
  (`ConvergenceWarning`, `NotFittedError`, `UndefinedMetricWarning`, and
  related classes) has no full Rust or Python compatibility equivalent.
- `sklearn.experimental`: import gates such as `enable_halving_search_cv` and
  `enable_iterative_imputer` do not exist; ferrolearn exposes the corresponding
  implemented APIs directly.
- `sklearn.base`: the Python `BaseEstimator` protocol, mixins, `clone`, and
  `is_classifier`/`is_regressor`/`is_clusterer`/`is_outlier_detector` style
  runtime discovery are not fully mirrored by Rust traits.
- `sklearn.utils`: only a small validation subset exists in `ferrolearn-core`;
  broad utilities like tags, indexing, resampling, random-state compatibility,
  `estimator_html_repr`, metadata routing, and estimator checks are absent.
- Display and plotting APIs are broadly absent beyond the exact display names
  listed above.

## Behavioral and Contract Gaps by Area

The source and tests document many narrower gaps. Important recurring themes:

| ferrolearn area | Current parity gaps and inaccuracies |
|---|---|
| Core / pipeline | Rust typestate gives stronger compile-time guarantees but is not sklearn's Python estimator protocol. Full `BaseEstimator` behavior, `get_params`/`set_params`/`clone`, metadata routing, estimator tags, HTML representation, and broad `check_estimator` compatibility are absent. Pipeline and `FeatureUnion` behavior has separate divergence tests for slicing and duplicate/dunder names. |
| Linear / SVM / discriminant | Many estimator names exist, but exact sklearn helper functions and several estimator variants are still missing (see API table). Known gaps include LDA binary `decision_function` shape, OMP Gram/precompute/CV/multi-output/n_iter surfaces, solver-option differences, `sample_weight`/`class_weight` coverage gaps, libsvm probability RNG/value differences, and Rust `FerroError` ABI instead of sklearn `ValueError`/`InvalidParameterError`/warnings. |
| Tree / ensemble | Stacking and tree export/plot helpers are absent. Implementations still differ in AdaBoost decision/probability/SAMME.R/base-estimator behavior, Voting's heterogeneous estimator/weight/transform surface, random and missing-value routing details, HGB bin threshold and missing-direction behavior, RNG exactness, and sklearn visualization/export attributes. |
| Neighbors | Missing transformer, density, and metric-learning estimators remain. Present estimators still have gaps around `X=None` self-query behavior, `sort_results`, `include_self`, constructor surfaces (`radius`, `metric`, `p`, `metric_params`, `n_jobs`), sparse/precomputed distances, exact error messages, and some tie/order behavior. |
| Naive Bayes | All five estimator names exist, but base/discrete sklearn conveniences are incomplete: `coef_`/`intercept_` properties, shared `_count`/`_update_feature_log_prob` style internals, exact prior/alpha edge semantics, `sample_weight`, warning/error ABI, and some `partial_fit` edge contracts remain narrower than sklearn. |
| Cluster / mixture / semi-supervised | Exact missing items include biclustering/coclustering. Present estimators have documented non-parity around BIRCH's CF-tree splitting and online API, BisectingKMeans centers/inertia/label numbering/tree-descent prediction, KMeans/MiniBatchKMeans defaults and RNG/local-optimum details, BayesianGMM pruning, Agglomerative full dendrogram/label numbering, FeatureAgglomeration inverse-transform shape behavior, HDBSCAN/OPTICS boundary precision, and semi-supervised zero-row/probability edge cases. |
| Decomposition / manifold / cross-decomposition | Missing decomposition helper functions and sparse/dictionary online variants remain. Present estimators still have gaps such as PCA `svd_solver` override/ARPACK/MLE/parameter surface, repeated-eigenbasis exactness, rank-deficient score precision, PLS fitted attributes/constructor modes/inverse transform/PyO3 bindings, LDA topic-model RNG initialization, and manifold helper-function/API differences. |
| Preprocess / impute / feature extraction / feature selection | This remains one of the biggest contract-parity areas. Exact missing names include mutual-information scoring and dict/hash/text extraction APIs. Common gaps include dense-only implementations where sklearn supports sparse, limited string/object/mixed dtype handling, incomplete feature-name plumbing, reduced Python/PyO3 exposure, image helper color/sparse/Python ABI gaps, random-projection RNG and component-orientation differences, KBins k-means/local-optimum edge cases, spline/quantile/power-transform edge cases, TF-IDF unsmoothed idf behavior, and degenerate feature-scoring semantics. |
| Metrics / scoring / pairwise | Many metrics now exist, but missing display, threshold, consensus, chunked pairwise, and several pairwise-kernel functions remain. Regression metrics are documented as mostly 1D/unweighted; `sample_weight`, `multioutput`, keyword/default surfaces, exact validation exceptions, and PyO3 exposure are incomplete. Scorer utilities are present but not full sklearn scoring/protocol parity. |
| Model selection / compose / calibration / multiclass / multioutput | Many names exist, but some exact sklearn public names are still absent (display helpers remain). Several implementations use Rust closures or simplified wrappers rather than sklearn's estimator-cloning protocol. RNG exactness, result-table parity, threshold/calibration displays, and some group/splitter edge semantics remain documented divergence areas. |
| Kernel / GP / covariance / neural | Core estimator names exist, but GP kernels are incomplete (`RationalQuadratic`, `ExpSineSquared`, `Exponentiation`, `CompoundKernel`, `Hyperparameter`). Kernel approximation is missing three sklearn estimators. Covariance and neural crates have public surfaces but many fitted-state items are still in conformance exclusions or have narrower solver/optimizer/attribute contracts than sklearn. |
| Datasets / fetch | Many toy/generator/fetch names exist, but 8 sklearn dataset APIs are absent. The newly surfaced `make_sparse_coded_signal`, `make_biclusters`, and `make_checkerboard` are scoped dense generators with sklearn-shaped outputs and structural guards; exact stochastic values remain RNG-substrate gaps. Fetcher signatures are narrower than sklearn: common sklearn options such as `return_X_y`, `as_frame`, `download_if_missing`, shuffle/random-state controls, retry/delay knobs, and some large-network value parity checks are absent or unverified. |
| Sparse / numerical / IO | `ferrolearn-sparse` is closer to scipy sparse than sklearn, but not full scipy parity; helper gaps remain. `ferrolearn-numerical` targets scipy-like primitives and should not be counted as sklearn estimator parity. `ferrolearn-io` adds JSON/MessagePack/ONNX/PMML-style facilities, but these are not sklearn API equivalents. |
| Python bindings | `ferrolearn-python` exposes only a subset of Rust estimators. Many Rust implementations have no Python wrapper, and wrappers generally do not provide full sklearn estimator protocol compatibility. |

## Stale or Inaccurate Documentation Claims

These documentation issues should be corrected separately from implementation
work:

- Root `README.md` still says the workspace has 14 crates, but current
  `Cargo.toml` lists 22 members.
- Root `README.md` says MSRV is 1.85, while `Cargo.toml` sets
  `rust-version = "1.88"`.
- The previous `GAP-REPORT.md` listed many now-present APIs as missing,
  including `CategoricalNB`, radius neighbors, `NearestNeighbors`,
  `LocalOutlierFactor`, `NearestCentroid`, many linear CV estimators,
  covariance, kernel/GP, neural-network, ensemble, and model-selection items.
- Several crate README parity summaries are optimistic relative to current
  source/test caveats. They should point to this report or to the relevant
  `tests/divergence_*.rs` files instead of claiming broad exact parity.
- Some test files retain historical "divergence" wording even after a fix has
  made the pin a green guard. Treat `divergence_*` filenames as evidence
  locations, not proof that the current assertion is failing.

## Evidence Inventory

Current divergence-test count by crate:

| Crate | `divergence_*.rs` files |
|---|---:|
| `ferrolearn-bayes` | 8 |
| `ferrolearn-cluster` | 30 |
| `ferrolearn-core` | 3 |
| `ferrolearn-covariance` | 6 |
| `ferrolearn-datasets` | 5 |
| `ferrolearn-decomp` | 39 |
| `ferrolearn-kernel` | 10 |
| `ferrolearn-linear` | 68 |
| `ferrolearn-metrics` | 15 |
| `ferrolearn-model-sel` | 31 |
| `ferrolearn-neighbors` | 17 |
| `ferrolearn-neural` | 5 |
| `ferrolearn-numerical` | 7 |
| `ferrolearn-preprocess` | 75 |
| `ferrolearn-sparse` | 4 |
| `ferrolearn-tree` | 18 |

The highest-signal files for future parity work are:

- API surface inventories and exclusions:
  `ferrolearn-*/tests/conformance/_surface_inventory.toml` and
  `ferrolearn-*/tests/conformance/_surface_exclusions.toml`.
- Source-level requirement tables in implementation files such as
  `ferrolearn-preprocess/src/lib.rs`, `ferrolearn-metrics/src/regression.rs`,
  `ferrolearn-cluster/src/birch.rs`, `ferrolearn-cluster/src/bisecting_kmeans.rs`,
  `ferrolearn-decomp/src/pca.rs`, `ferrolearn-decomp/src/cross_decomposition.rs`,
  `ferrolearn-neighbors/src/nearest_neighbors.rs`, and
  `ferrolearn-tree/src/adaboost.rs`.
- Current Python bindings in `ferrolearn-python/src/lib.rs` and wrapper modules.

## Recommended Next Work

1. Fix stale user-facing docs first: root `README.md` crate count, MSRV, and
   broad parity claims.
2. Convert the exact API gap table into tracked issues by module, separating
   estimators from standalone helpers/displays.
3. Add a generated parity-inventory script so this report can be refreshed
   repeatably from sklearn docs and Rust public symbols.
4. For behavioral parity, prioritize areas where present APIs return different
   values rather than merely narrower Rust surfaces: metrics edge cases,
   preprocessing degeneracies, tree/ensemble prediction behavior, neighbor
   query semantics, and cluster/decomposition RNG or solver differences.
