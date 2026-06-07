//! PyO3 bindings for the extended estimator surface (~40 estimators beyond the
//! core regressor/classifier/transformer/clusterer bindings), so `import ferrolearn`
//! mirrors the breadth of `import sklearn` and the head-to-head bench can exercise
//! the full ferrolearn surface against scikit-learn.
//!
//! ## REQ status
//!
//! ~40 `sklearn` estimator binding shims (via the `py_regressor!`/`py_classifier!`/
//! `py_transformer!` macros + hand-written pyclasses) over
//! `ferrolearn_linear`/`ferrolearn_tree`/`ferrolearn_neighbors`/`ferrolearn_bayes`/
//! `ferrolearn_cluster`/`ferrolearn_decomp`/`ferrolearn_preprocess`/`ferrolearn_kernel`,
//! wrapped by the `_extras.py` mixin wrappers (`_RegressorWrapper`/`_ClassifierWrapper`/
//! `_ClusterWrapper`/`_TransformerWrapper` + a LabelEncoder-equivalent `_encode`).
//! This unit owns the sklearn-API marshalling surface only (constructor ABI,
//! method/attribute exposure, array coercion); the estimator MATH lives downstream
//! in the eight respective crates (pre-existing audited). Verification model B:
//! pytest comparing `import ferrolearn` against `import sklearn` 1.5.2. Design doc:
//! `.design/python/extras.md` (17 REQs). Every REQ is BINARY (R-DEFER-2): SHIPPED
//! or NOT-STARTED (with a concrete blocker). Verified via
//! `tests/divergence_extras.py` + the per-category divergence suites (595 pytest pass).
//!
//! **15 SHIPPED / 5 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-REGRESSOR-API-CONFORM (fit/predict, 11 regressors) | SHIPPED | `py_regressor!` macro + hand-written `Rs*Regressor` expose fit/predict, wrapped by `_extras.py::_RegressorWrapper` (+ `n_features_in_`/`score`). Mirrors the sklearn regressor fit/predict contract across `ensemble`/`linear_model`/`neighbors`/`kernel_ridge`/`tree`. |
//! | REQ-CLASSIFIER-API-CONFORM (fit/predict + LabelEncoder, 13 classifiers) | SHIPPED | `py_classifier!` macro + hand-written classifiers expose fit/predict, wrapped by `_extras.py::_ClassifierWrapper` whose `_encode` (np.unique+searchsorted) sets `classes_` and round-trips arbitrary label dtypes through the Rust `usize`-label core. |
//! | REQ-CLUSTERER-API-CONFORM (fit + labels_, 5 clusterers incl. GMM) | SHIPPED | `RsMiniBatchKMeans`/`RsDBSCAN`/`RsAgglomerativeClustering`/`RsBirch` expose fit + `labels_` (+ predict for MiniBatchKMeans), `RsGaussianMixture` fit/predict; wrapped by `_extras.py::_ClusterWrapper` (+ `fit_predict`). |
//! | REQ-DBSCAN-BINDING-SURFACE (`sample_weight` + `core_sample_indices_` + `components_` + `metric`/`p`) | SHIPPED | FIXED #2190: `RsDBSCAN.fit` takes `#[pyo3(signature = (x, sample_weight=None))]` and chains `.with_sample_weight(numpy1_to_ndarray(w))` (weighted core determination, `cluster/_dbscan.py:370,427-435`; wrong-length → `ShapeMismatch`→`PyValueError`); new `#[getter] core_sample_indices_ → PyArray1<i64>` (`_dbscan.py:438`) + `#[getter] components_ → PyArray2<f64>` (`_dbscan.py:441-446`). EXTENDED #2193: `RsDBSCAN::new` ctor gains `metric="euclidean".to_string()` + `p=None` (`_dbscan.py:345-363`); `fn fit` resolves the lowercased metric string via `resolve_dbscan_metric` (`euclidean`/`l2`→Euclidean, `manhattan`/`l1`/`cityblock`→Manhattan, `chebyshev`→Chebyshev, `minkowski`→`Minkowski(p.unwrap_or(2.0))`; unknown→`PyValueError`, matching sklearn's `InvalidParameterError ⊂ ValueError`) and chains `.with_metric(resolved)` onto the builder (consuming `ferrolearn_cluster::dbscan::DbscanMetric`/`with_metric`, core commit 485c06fcd). `p` IGNORED for non-Minkowski metrics (#2192); non-positive/NaN Minkowski `p`→`InvalidParameter`→`PyValueError`. `_extras.py::DBSCAN` overrides `fit(X, y=None, sample_weight=None)` + `fit_predict(... sample_weight=None)`, exposing `labels_`/`core_sample_indices_`/`components_`; ctor `(eps=0.5, *, min_samples=5, metric='euclidean', p=None)` threads metric/p through `_make_rs` (so `get_params`/`clone`/`set_params` round-trip them). Non-test consumer: `ferrolearn.DBSCAN(..., metric='manhattan', p=...).fit(X, sample_weight=w)`. Live oracle: `tests/divergence_dbscan_sample_weight.py` (9 pass) + `tests/test_dbscan_metric.py` (20 pass) — euclidean/l2/manhattan(+aliases)/chebyshev/minkowski-p∈{1,2,3} `labels_`+`core_sample_indices_` parity, p1==manhattan, p2==euclidean, euclidean-ignores-p (#2192), unknown-metric-`ValueError`, non-positive-p-`ValueError`, metric+sample_weight, clone/get_params/set_params. DOCUMENTED DIVERGENCE: ferrolearn `metric='minkowski', p=None`→p=2 (documented intent, fits) vs sklearn `TypeError`. `metric_params`/`precomputed`/callable/`algorithm`/`leaf_size`/`n_jobs` stay NOT-STARTED. See `.design/python/extras.md`. |
//! | REQ-TRANSFORMER-API-CONFORM (fit/transform, 13 transformers) | SHIPPED | `py_transformer!` macro exposes fit/transform for the decomp/preprocess/kernel transformers, wrapped by `_extras.py::_TransformerWrapper` (+ `fit_transform`). |
//! | REQ-REGRESSOR-VALUE-PARITY (default-path predict parity) | SHIPPED | deterministic default path: the deterministic regressors' predict parity is verified downstream in `ferrolearn_linear`/`_tree`/`_neighbors`/`_kernel` REQ tables. (Seeded-RNG ensembles → REQ-VALUE-PARITY-RNG.) |
//! | REQ-CLASSIFIER-VALUE-PARITY (default-path label parity) | SHIPPED | deterministic default path: decoded label predictions of the deterministic classifiers match sklearn after the `_encode`/decode round-trip; verified downstream in `ferrolearn_linear`/`_bayes`/`_tree`. |
//! | REQ-CLUSTERER-VALUE-PARITY (default-path partition parity) | SHIPPED | deterministic default path: `labels_` for `DBSCAN`/`AgglomerativeClustering`/`Birch` match sklearn's partition up to label permutation; verified in `ferrolearn_cluster`. |
//! | REQ-TRANSFORMER-VALUE-PARITY (default-path transform parity) | SHIPPED | deterministic default path: transformed output for the deterministic transformers matches the sklearn oracle (sign/permutation invariance where applicable); verified in `ferrolearn_decomp`/`_preprocess`/`_kernel`. |
//! | REQ-CTOR-ABI-POSITIONAL (16 positional primaries) | SHIPPED | FIXED #2055: the 16 `_extras.py` wrappers whose primary hyperparameter sklearn makes positional-or-keyword now place it before the `*` (`RandomForestRegressor`/`ExtraTreesRegressor` `n_estimators`, `KNeighborsRegressor` `n_neighbors`, `RidgeClassifier`/`KernelRidge` `alpha`, `MiniBatchKMeans`/`AgglomerativeClustering` `n_clusters`, `DBSCAN` `eps`, `GaussianMixture`/`TruncatedSVD`/`FastICA`/`NMF`/`IncrementalPCA`/`KernelPCA`/`SparsePCA`/`FactorAnalysis` `n_components`). Parametrized guard `test_red_extras_primary_param_positional` (16 cases). |
//! | REQ-MODULE-ALLOW (no module-root `#![allow]`) | SHIPPED | FIXED #2056: the module-root `#![allow(non_snake_case)]` was removed (it was dead — all field names are snake_case); `cargo clippy -p ferrolearn-python --all-targets -- -D warnings` stays green (R-CODE-3/R-APG-1). |
//! | REQ-PHASE-FRAMING (no Phase-N deferral framing) | SHIPPED | the `//!` (this header) and `_extras.py` docstring were reworded to describe the extras binding surface without "Phase N" framing (R-DEFER-4), and this `## REQ status` table was added. |
//! | REQ-DECOMP-NCOMPONENTS-DEFAULT (n_components default) | NOT-STARTED | the 5-6 decomp transformers hardcode `n_components=2` vs sklearn `None` (`IncrementalPCA`/`FastICA`/`KernelPCA`/`SparsePCA`/`FactorAnalysis`) / `'warn'`→None (`NMF`); the `None`-auto-rank behavior is owned downstream by `ferrolearn_decomp`. (`TruncatedSVD` default 2 MATCHES.) |
//! | REQ-DISCRETE-NB-FITTED-ATTRS (feature_log_prob_/class_log_prior_/feature_count_/class_count_) | SHIPPED | FIXED #2103: `MultinomialNB`/`BernoulliNB`/`ComplementNB` expose the four `_BaseDiscreteNB` fitted attrs (sklearn/naive_bayes.py:880-892/580-602/1032-1042). A SECOND `#[pymethods] impl Rs{Multinomial,Bernoulli,Complement}NB` block (pyo3 `multiple-pymethods`) adds four `#[getter]`s each over the Rust fitted types (`FittedMultinomialNB` multinomial.rs:489/499/509/520, `FittedBernoulliNB` bernoulli.rs:520/530/540/551, `FittedComplementNB` complement.rs:535/586/545/556); `py_classifier!` macro + its 18 invocations UNCHANGED. ComplementNB `feature_log_prob_` is the `-logged` complement weight (positive) per sklearn. Non-test consumer: `_extras.py::_DiscreteNBWrapper.fit` sets `self.{feature_log_prob_,class_log_prior_,feature_count_,class_count_}` from `self._rs.*` (the three NB wrappers subclass it). Verification (model B): `tests/divergence_extras.py::test_{multinomial,bernoulli,complement}_discrete_nb_fitted_attrs_match_sklearn` (live sklearn 1.5.2 oracle, atol 1e-7). |
//! | REQ-BERNOULLI-BINARIZE-NONE (`binarize=None` skips binarization) | SHIPPED | FIXED #2104: `_RsBernoulliNB` now types `binarize: Option<f64> = Some(0.0)` (was `f64 = 0.0`) and builds via `BernoulliNB::with_binarize_option(binarize)` (bernoulli.rs), so pyo3 maps Python `None`→`Option::None` (skips binarization, sklearn/naive_bayes.py:1076,1156,1179,1185) and a float→`Some(f64)`. `feature_count_` then accumulates raw X. Non-test consumer: `_extras.py::BernoulliNB.__init__(binarize=0.0)` passes `self.binarize` straight through (default 0.0 unchanged). Verification (model B): `tests/divergence_extras.py::test_red_bernoulli_binarize_none_feature_count_matches_sklearn` (live sklearn 1.5.2 oracle, atol 1e-9). |
//! | REQ-IFOREST-BINDING (#2180) | SHIPPED | `RsIsolationForest` (hand `#[pyclass]`) over `ferrolearn_tree::IsolationForest<f64>`/`FittedIsolationForest<f64>` exposes the `OutlierMixin` surface — `predict` (±1, `_iforest.py:357-378`), `score_samples` (∈[-1,0], `_iforest.py:412-451`), `decision_function` (`=score-offset_`, `_iforest.py:380-410`) — plus `offset_` (`_iforest.py:344/353`) and `n_features_in_`. The `_extras.py::IsolationForest(_RegressorPickleMixin, OutlierMixin, BaseEstimator)` wrapper mirrors sklearn's full ctor (`_iforest.py:221-233`), resolves `max_samples` ('auto'→`min(256,n)`, int→as-is, float→`int(f*n)`, `_iforest.py:303-318`) and `contamination` ('auto'→`with_contamination_auto()`, float→`with_contamination(c)`, `_iforest.py:341-353`) BEFORE the ABI, and rejects the unsupported surface (`max_features!=1.0`/`bootstrap=True`/`warm_start=True` → NotImplementedError, isolation_forest.rs REQ-7a/7b #728/#729; bad `n_estimators`/`contamination`/`max_samples` → ValueError); `n_jobs`/`verbose` accept-and-ignore. RNG-substrate caveat #2118 (isolation_forest.rs StdRng vs numpy MT19937 #730): exact `score_samples` DIVERGE from sklearn for a given seed; the STRUCTURAL contract matches (asserted vs the live oracle). Non-test consumer: `_extras.py::IsolationForest` + `lib.rs` `add_class` + `__init__.py` re-export. Verification: `tests/divergence_extras.py::test_iforest_*` (16, live sklearn 1.5.2 oracle). |
//! | REQ-FEATAGGLOM-BINDING (#943) | SHIPPED | `RsFeatureAgglomeration` (hand `#[pyclass]`) over `ferrolearn_cluster::FeatureAgglomeration<f64>`/`FittedFeatureAgglomeration<f64>` exposes `fit`/`transform`/`inverse_transform` (`_feature_agglomeration.py:32-92`) + the dendrogram getters `labels_` (1-D int), `n_clusters_`, `children_` ((n-1,2) int), `distances_` (1-D f64 or None), `n_leaves_`, `n_connected_components_`, `n_features_in_` — delegated from the inner clustering over `X.T` (`_agglomerative.py:1083-1095/1339`). `fit` maps `linkage`/`pooling` strings → `AgglomerativeLinkage`/`PoolingFunc`; an unknown string → `PyValueError`; FerroError → `PyValueError`; not-fitted → `PyRuntimeError` (no panic). The `_extras.py::FeatureAgglomeration(_TransformerWrapper)` wrapper mirrors sklearn's ctor (`_agglomerative.py:1296`, `n_clusters=2` positional), resolves `pooling_func` ('mean'/'max' OR np.mean/np.max → enum; other callable → NotImplementedError REQ-7 #941) BEFORE the ABI, rejects core-unsupported `metric`/`connectivity`/`distance_threshold` (non-default) → NotImplementedError REQ-6 #941, and sets the dendrogram attrs post-fit. VALUE-EXACT vs sklearn: `labels_` int-exact, `transform`/`inverse_transform`/`children_`/`distances_` ≈1e-9 across {ward,complete,average,single}×{mean,max}. Non-test consumer: `_extras.py::FeatureAgglomeration` + `lib.rs` `add_class` + `__init__.py` re-export. Verification: `tests/test_feature_agglomeration.py` (21, live sklearn 1.5.2 oracle). |
//! | REQ-RANSAC-BINDING (#2178) | SHIPPED | `RsRANSACRegressor` (hand `#[pyclass]`) over `ferrolearn_linear::RANSACRegressor<f64, LinearRegression<f64>>` (default `LinearRegression()` base, `_ransac.py:380`) exposes `fit`/`predict`/`coef_`/`intercept_`/`inlier_mask_`. The `_extras.py::RANSACRegressor` wrapper mirrors sklearn's full ctor (`_ransac.py:288-315`), resolves `min_samples` (None→`n_features+1`, float→`ceil`) before the ABI, and rejects the unsupported surface (`estimator`/`loss='squared_error'`/`stop_*`/`is_*_valid` → NotImplementedError; bad `min_samples`/`max_trials`/`residual_threshold` → ValueError). `coef_`/`intercept_` recovered EXACTLY via affine probe predictions (core keeps the refit base private, ransac.rs REQ-10 #518; no `n_trials_`/`estimator_` faked). Non-test consumer: `_extras.py::RANSACRegressor` + `lib.rs` `add_class` + `__init__.py` re-export. Verification: `tests/divergence_extras.py::test_ransac_*` (14, live sklearn 1.5.2 oracle on well-separated data; coef_/inlier_mask_ MATCH `atol 1e-2`, RNG caveat #2118). |
//! | REQ-MISSING-METHODS (coef_/predict_proba/inverse_transform/cluster_centers_) | NOT-STARTED | the `Rs*` classes expose only fit/predict(/transform/labels_) — no `coef_`/`feature_importances_`, `predict_proba`/`decision_function`, `inverse_transform`/`components_`, `cluster_centers_`/`children_`. The binding cannot expose attrs the fitted library types do not compute — owned downstream by the eight crates. |
//! | REQ-MISSING-PARAMS (full constructor surface) | NOT-STARTED | each `Rs*` constructor binds a thin subset of sklearn's params (e.g. `RsRandomForestRegressor` lacks `criterion`/`max_features`/`bootstrap`/`oob_score`; `RsBaggingClassifier` lacks the `estimator` knob). Owned downstream by the eight crates. (`KNeighborsRegressor` is the exception: full surface SHIPPED, REQ-KNR-CTOR-SURFACE.) |
//! | REQ-KNR-CTOR-SURFACE (KNeighborsRegressor full constructor) | SHIPPED | FIXED #2147: hand-written `RsKNeighborsRegressor` (replacing the `n_neighbors`-only `py_regressor!` invocation) mirrors `RsKNeighborsClassifier` and exposes sklearn's full `(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)` surface (`neighbors/_regression.py:178-189`); `weights='distance'` changes predictions and matches the live sklearn 1.5.2 oracle. `fit(x, y: f64)`/`predict→PyArray1<f64>` keep the macro ABI. Validation is metric-aware (FIXED #2148): `metric='minkowski'` requires `p==2` (else `PyNotImplementedError #876`); `metric='euclidean'` accepts ANY `p` (sklearn ignores p for euclidean, `_base.py:526-538`); other metrics `PyNotImplementedError #876`; an invalid `weights` STRING raises `ValueError` (sklearn `InvalidParameterError ⊂ ValueError`, FIXED #2149); callable-weights/`metric_params` reject with `NotImplementedError #876`. Non-test consumer: `_extras.py::KNeighborsRegressor` full shim (`_make_rs`/`fit`/`predict` + `_RegressorPickleMixin`). Verification: `tests/divergence_extras.py::test_knr_*` (14 cases) + `tests/divergence_knr_ctor_surface.py` (#2148/#2149 pins). |
//! | REQ-VALUE-PARITY-RNG (seeded stochastic parity) | NOT-STARTED | the stochastic estimators (RF/ET/GB/HistGB/Bagging/AdaBoost/MiniBatchKMeans/GMM/FastICA/NMF) pass `random_state: Option<u64>` to a non-numpy RNG, so a shared `random_state` will not reproduce sklearn's numpy-MT/PCG64 draws (R-SUBSTRATE-5; needs `ferray::random`). |
//! | REQ-CONSUMER (binding IS the public API) | SHIPPED | non-test consumers: the `_extras.py` wrappers + `GaussianMixture` construct their `_Rs*` class via `_make_rs` and drive fit/predict/transform; `ferrolearn/__init__.py` re-exports all ~40; `lib.rs` registers every `_Rs*`; the head-to-head bench drives them vs sklearn (595 pytest pass). |
//! | REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | marshals via `crate::conversions::*` (rust-numpy + `ndarray`), not `ferray::numpy_interop`/`ferray-core` (R-SUBSTRATE-1); ferray exposes no numpy bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027. |

use crate::conversions::*;
use ferrolearn_core::{Fit, Predict, Transform};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// ===========================================================================
// Linear regressors (extras)
// ===========================================================================

macro_rules! py_regressor {
    (
        $cls_name:ident, $py_name:literal, $fitted_path:path,
        ($($field:ident : $ty:ty = $default:expr),* $(,)?),
        $build_block:block
    ) => {
        #[pyclass(name = $py_name)]
        pub struct $cls_name {
            $($field: $ty,)*
            fitted: Option<$fitted_path>,
        }

        #[pymethods]
        impl $cls_name {
            #[new]
            #[pyo3(signature = ($($field = $default),*))]
            fn new($($field: $ty),*) -> Self {
                Self { $($field,)* fitted: None }
            }

            fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
                $(let $field = self.$field.clone();)*
                let x_nd = numpy2_to_ndarray(x);
                let y_nd = numpy1_to_ndarray(y);
                let model = $build_block;
                let fitted = model.fit(&x_nd, &y_nd)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                self.fitted = Some(fitted);
                Ok(())
            }

            fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>)
                -> PyResult<Bound<'py, PyArray1<f64>>>
            {
                let fitted = self.fitted.as_ref()
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
                let x_nd = numpy2_to_ndarray(x);
                let preds = fitted.predict(&x_nd)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(ndarray1_to_numpy(py, &preds))
            }
        }
    };
}

macro_rules! py_classifier {
    (
        $cls_name:ident, $py_name:literal, $fitted_path:path,
        ($($field:ident : $ty:ty = $default:expr),* $(,)?),
        $build_block:block
    ) => {
        #[pyclass(name = $py_name)]
        pub struct $cls_name {
            $($field: $ty,)*
            fitted: Option<$fitted_path>,
        }

        #[pymethods]
        impl $cls_name {
            #[new]
            #[pyo3(signature = ($($field = $default),*))]
            fn new($($field: $ty),*) -> Self {
                Self { $($field,)* fitted: None }
            }

            fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
                $(let $field = self.$field.clone();)*
                let x_nd = numpy2_to_ndarray(x);
                let y_nd = numpy1_to_ndarray_usize(y);
                let model = $build_block;
                let fitted = model.fit(&x_nd, &y_nd)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                self.fitted = Some(fitted);
                Ok(())
            }

            fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>)
                -> PyResult<Bound<'py, PyArray1<i64>>>
            {
                let fitted = self.fitted.as_ref()
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
                let x_nd = numpy2_to_ndarray(x);
                let preds = fitted.predict(&x_nd)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(ndarray1_usize_to_numpy(py, &preds))
            }
        }
    };
}

macro_rules! py_transformer {
    (
        $cls_name:ident, $py_name:literal, $fitted_path:path,
        ($($field:ident : $ty:ty = $default:expr),* $(,)?),
        $build_block:block
    ) => {
        #[pyclass(name = $py_name)]
        pub struct $cls_name {
            $($field: $ty,)*
            fitted: Option<$fitted_path>,
        }

        #[pymethods]
        impl $cls_name {
            #[new]
            #[pyo3(signature = ($($field = $default),*))]
            fn new($($field: $ty),*) -> Self {
                Self { $($field,)* fitted: None }
            }

            fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
                $(let $field = self.$field.clone();)*
                let x_nd = numpy2_to_ndarray(x);
                let model = $build_block;
                let fitted = model.fit(&x_nd, &())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                self.fitted = Some(fitted);
                Ok(())
            }

            fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>)
                -> PyResult<Bound<'py, PyArray2<f64>>>
            {
                let fitted = self.fitted.as_ref()
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
                let x_nd = numpy2_to_ndarray(x);
                let xt = fitted.transform(&x_nd)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(ndarray2_to_numpy(py, &xt))
            }
        }
    };
}

// ===========================================================================
// Linear regressors
// ===========================================================================

// BayesianRidge (#2161): hand-written pyclass (replacing the thin
// `py_regressor!` invocation) so the binding can surface sklearn's
// `compute_score` constructor param (`sklearn/linear_model/_bayes.py:198`), the
// `sample_weight` fit argument (`_bayes.py:217`), the `predict(return_std=True)`
// path (`_bayes.py:343-371`), and the fitted `n_iter_`/`scores_`/`sigma_`
// attributes. The MATH lives in `ferrolearn_linear::BayesianRidge`
// (`fit_with_sample_weight`, `with_compute_score`, `predict_with_std`,
// `n_iter`/`scores`/`sigma_full`); this is the thin marshalling shim and the
// non-test production consumer of the new core API (R-DEFER-1).
#[pyclass(name = "_RsBayesianRidge")]
pub struct RsBayesianRidge {
    max_iter: usize,
    tol: f64,
    compute_score: bool,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedBayesianRidge<f64>>,
}

#[pymethods]
impl RsBayesianRidge {
    #[new]
    #[pyo3(signature = (max_iter=300, tol=1e-3, compute_score=false, fit_intercept=true))]
    fn new(max_iter: usize, tol: f64, compute_score: bool, fit_intercept: bool) -> Self {
        Self {
            max_iter,
            tol,
            compute_score,
            fit_intercept,
            fitted: None,
        }
    }

    // sklearn's `fit(X, y, sample_weight=None)` (`_bayes.py:217`). A `None`
    // sample_weight is byte-identical to the unweighted fit; a non-None weight
    // is rescaled via `_rescale_data` inside the Rust core (`_bayes.py:254-256`).
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::BayesianRidge::<f64>::new()
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_compute_score(self.compute_score)
            .with_fit_intercept(self.fit_intercept);
        let sw = sample_weight.as_ref().map(|w| numpy1_to_ndarray(w.clone()));
        let fitted = model
            .fit_with_sample_weight(&x_nd, &y_nd, sw.as_ref())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    // sklearn's `predict(X, return_std=False)` (`_bayes.py:343`). With
    // `return_std=False` returns the mean array; with `return_std=True` returns
    // a `(mean, std)` tuple (`_bayes.py:367-371`).
    #[pyo3(signature = (x, return_std=false))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        return_std: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        if return_std {
            let (mean, std) = fitted
                .predict_with_std(&x_nd)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let mean_py = ndarray1_to_numpy(py, &mean);
            let std_py = ndarray1_to_numpy(py, &std);
            Ok((mean_py, std_py).into_pyobject(py)?.into_any())
        } else {
            let preds = fitted
                .predict(&x_nd)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(ndarray1_to_numpy(py, &preds).into_any())
        }
    }

    // sklearn `n_iter_` (`_bayes.py:316` `self.n_iter_ = iter_ + 1`): the actual
    // number of EM iterations to reach the stopping criterion.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_iter())
    }

    // sklearn `scores_` (`_bayes.py:283/302/330`): the per-iteration log
    // marginal likelihood (length n_iter_+1), populated only when
    // compute_score=True; otherwise an empty array.
    #[getter]
    fn scores_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let scores = Array1::from_vec(fitted.scores().to_vec());
        Ok(ndarray1_to_numpy(py, &scores))
    }

    // sklearn `sigma_` (`_bayes.py:333-337`): the full (n_features, n_features)
    // posterior covariance matrix of the weights.
    #[getter]
    fn sigma_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.sigma_full()))
    }

    // sklearn `coef_` / `intercept_` (`_bayes.py:96-101`).
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }

    #[getter]
    fn alpha_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.alpha())
    }

    #[getter]
    fn lambda_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.lambda())
    }
}

// ARDRegression (#2163): hand-written pyclass (replacing the thin
// `py_regressor!` invocation) so the binding can surface sklearn's
// `compute_score` constructor param (`sklearn/linear_model/_bayes.py:587`), the
// `predict(X, return_std=True)` path (`_bayes.py:761-791`), and the fitted
// `n_iter_`/`scores_`/`sigma_` attributes. The MATH lives in
// `ferrolearn_linear::ARDRegression` (`with_compute_score`, `predict_with_std`,
// `n_iter`/`scores`/`sigma_full`); this is the thin marshalling shim and the
// non-test production consumer of the new core API (R-DEFER-1). `sigma_` is the
// kept-feature `(n_kept, n_kept)` covariance, exactly as sklearn exposes it
// (`_bayes.py:727`).
#[pyclass(name = "_RsARDRegression")]
pub struct RsARDRegression {
    max_iter: usize,
    tol: f64,
    compute_score: bool,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedARDRegression<f64>>,
}

#[pymethods]
impl RsARDRegression {
    #[new]
    #[pyo3(signature = (max_iter=300, tol=1e-3, compute_score=false, fit_intercept=true))]
    fn new(max_iter: usize, tol: f64, compute_score: bool, fit_intercept: bool) -> Self {
        Self {
            max_iter,
            tol,
            compute_score,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::ARDRegression::<f64>::new()
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_compute_score(self.compute_score)
            .with_fit_intercept(self.fit_intercept);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    // sklearn's `predict(X, return_std=False)` (`_bayes.py:761`). With
    // `return_std=False` returns the mean array; with `return_std=True` returns
    // a `(mean, std)` tuple computed over the KEPT columns (`_bayes.py:787-790`).
    #[pyo3(signature = (x, return_std=false))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        return_std: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        if return_std {
            let (mean, std) = fitted
                .predict_with_std(&x_nd)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let mean_py = ndarray1_to_numpy(py, &mean);
            let std_py = ndarray1_to_numpy(py, &std);
            Ok((mean_py, std_py).into_pyobject(py)?.into_any())
        } else {
            let preds = fitted
                .predict(&x_nd)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(ndarray1_to_numpy(py, &preds).into_any())
        }
    }

    // sklearn `n_iter_` (`_bayes.py:716` `self.n_iter_ = iter_ + 1`).
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_iter())
    }

    // sklearn `scores_` (`_bayes.py:695-704`): the per-iteration ARD objective
    // (length n_iter_), populated only when compute_score=True; otherwise empty.
    #[getter]
    fn scores_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let scores = Array1::from_vec(fitted.scores().to_vec());
        Ok(ndarray1_to_numpy(py, &scores))
    }

    // sklearn `sigma_` (`_bayes.py:727`): the kept-feature (n_kept, n_kept)
    // posterior covariance matrix of the weights.
    #[getter]
    fn sigma_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.sigma_full()))
    }

    // sklearn `coef_` / `intercept_` (`_bayes.py:490`, `:510`).
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }

    // sklearn `alpha_` (`_bayes.py:493`): estimated noise precision.
    #[getter]
    fn alpha_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.alpha())
    }

    // sklearn `lambda_` (`_bayes.py:496`): per-feature weight precisions.
    #[getter]
    fn lambda_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.lambda()))
    }
}

// HuberRegressor (#500/#501): hand-written pyclass (replacing the thin
// `py_regressor!` invocation) so the binding can surface sklearn's `warm_start`
// constructor param (`sklearn/linear_model/_huber.py:265`), the `sample_weight`
// fit argument (`_huber.py:277`), and the fitted `coef_`/`intercept_`/`scale_`
// attributes the warm-start refit reuses. The MATH lives in
// `ferrolearn_linear::HuberRegressor` (`fit_with_sample_weight` +
// `with_warm_start_state`); this is the thin marshalling shim and the
// non-test production consumer of the new core API (R-DEFER-1).
#[pyclass(name = "_RsHuberRegressor")]
pub struct RsHuberRegressor {
    epsilon: f64,
    alpha: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    warm_start: bool,
    fitted: Option<ferrolearn_linear::FittedHuberRegressor<f64>>,
}

#[pymethods]
impl RsHuberRegressor {
    #[new]
    #[pyo3(signature = (epsilon=1.35, alpha=0.0001, max_iter=100, tol=1e-5,
                        fit_intercept=true, warm_start=false))]
    fn new(
        epsilon: f64,
        alpha: f64,
        max_iter: usize,
        tol: f64,
        fit_intercept: bool,
        warm_start: bool,
    ) -> Self {
        Self {
            epsilon,
            alpha,
            max_iter,
            tol,
            fit_intercept,
            warm_start,
            fitted: None,
        }
    }

    // sklearn's `fit(X, y, sample_weight=None)` (`_huber.py:277`). When
    // `warm_start` is set AND a prior fit exists, the previously fitted
    // `(coef_, intercept_, scale_)` seed the optimizer (sklearn `_huber.py:308`
    // `if self.warm_start and hasattr(self, "coef_")`) — the R-DEV-4 stand-in
    // is `with_warm_start_state`, fed from `self.fitted`.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);

        let mut model = ferrolearn_linear::HuberRegressor::<f64>::new()
            .with_epsilon(self.epsilon)
            .with_alpha(self.alpha)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_fit_intercept(self.fit_intercept)
            .with_warm_start(self.warm_start);

        if self.warm_start
            && let Some(prev) = &self.fitted
        {
            use ferrolearn_core::introspection::HasCoefficients;
            model = model.with_warm_start_state(
                prev.coefficients().to_owned(),
                prev.intercept(),
                prev.scale(),
            );
        }

        let sw = sample_weight.as_ref().map(|w| numpy1_to_ndarray(w.clone()));
        let fitted = model
            .fit_with_sample_weight(&x_nd, &y_nd, sw.as_ref())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }

    #[getter]
    fn scale_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.scale())
    }

    // sklearn `n_iter_` (`sklearn/linear_model/_huber.py:342` `self.n_iter_ =
    // opt_res.nit`): the number of optimizer iterations the fit ran for. The
    // MATH (and the R-DEV-7 honest-count caveat) lives in
    // `FittedHuberRegressor::n_iter`; this getter marshals it as a Python int.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_iter())
    }
}

// QuantileRegressor (#507/#508): hand-written pyclass (replacing the thin
// `py_regressor!` invocation) so the binding can surface sklearn's `solver`
// (`sklearn/linear_model/_quantile.py:134`, default `"highs"`) and
// `solver_options` (`_quantile.py:135`, default `None`) constructor params and
// the fitted `n_iter_` attribute (`_quantile.py:300` `self.n_iter_ =
// result.nit`). The MATH (LP build, two-phase simplex, the pivot count, and the
// solver/solver_options validation) lives in
// `ferrolearn_linear::QuantileRegressor` (`with_solver`/`with_solver_options`,
// `FittedQuantileRegressor::n_iter`); this is the thin marshalling shim and the
// non-test production consumer of the new core API (R-DEFER-1).
//
// Solver validity mirrors sklearn's `_parameter_constraints["solver"]`
// (`_quantile.py:114-124`): the HiGHS family + "revised simplex" reach the same
// LP vertex; an invalid string OR "interior-point" (removed in scipy>=1.11) is
// rejected by the core as `InvalidParameter`, marshalled here to `ValueError`
// (matching sklearn's `InvalidParameterError`/`ValueError`). `solver_options`
// is `[dict, None]` in sklearn (`_quantile.py:125`) — ANY dict is valid and
// sklearn fits successfully. The core ACCEPTS (and ignores) any `solver_options`
// (they are HiGHS tuning knobs that do not change the LP optimum, #2168), so the
// binding stores them for `get_params`/`clone()` round-trip and passes them
// straight through to the solve.
#[pyclass(name = "_RsQuantileRegressor")]
pub struct RsQuantileRegressor {
    quantile: f64,
    alpha: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    solver: String,
    // sklearn's `solver_options` is `[dict, None]` (`_quantile.py:125`). Any
    // dict is accepted; the options are HiGHS tuning knobs that do not change the
    // LP optimum (#2168). Stored for `get_params`/`clone()` round-trip.
    solver_options: Option<std::collections::HashMap<String, f64>>,
    fitted: Option<ferrolearn_linear::FittedQuantileRegressor<f64>>,
}

#[pymethods]
impl RsQuantileRegressor {
    #[new]
    #[pyo3(signature = (quantile=0.5, alpha=1.0, max_iter=10000, tol=1e-6,
                        fit_intercept=true, solver="highs".to_string(),
                        solver_options=None))]
    fn new(
        quantile: f64,
        alpha: f64,
        max_iter: usize,
        tol: f64,
        fit_intercept: bool,
        solver: String,
        solver_options: Option<std::collections::HashMap<String, f64>>,
    ) -> Self {
        Self {
            quantile,
            alpha,
            max_iter,
            tol,
            fit_intercept,
            solver,
            solver_options,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::QuantileRegressor::<f64>::new()
            .with_quantile(self.quantile)
            .with_alpha(self.alpha)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_fit_intercept(self.fit_intercept)
            .with_solver(self.solver.clone())
            .with_solver_options(self.solver_options.clone());
        let fitted = model.fit(&x_nd, &y_nd).map_err(quantile_fit_err)?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }

    // sklearn `n_iter_` (`_quantile.py:300` `self.n_iter_ = result.nit`): the
    // number of solver iterations. R-DEV-7: ferrolearn's value is the honest
    // two-phase primal-simplex pivot count, NOT scipy's HiGHS iteration count,
    // so it is a positive int <= max_iter and deterministic, but not == sklearn.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_iter())
    }
}

// Map `QuantileRegressor::fit` errors to the sklearn-equivalent Python
// exception. An invalid `solver` (incl. "interior-point") mirrors sklearn's
// `InvalidParameterError`/`ValueError` (`_quantile.py:114`/`:196`); everything
// else is a `ValueError`. (`solver_options` is now accepted by the core for any
// dict, #2168, so it never produces an error here.)
fn quantile_fit_err(e: ferrolearn_core::error::FerroError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// OrthogonalMatchingPursuit (#2172): hand-written pyclass (rather than a thin
// `py_regressor!` invocation) so the binding surfaces sklearn's full keyword-only
// constructor `(n_nonzero_coefs=None, tol=None, fit_intercept=True,
// precompute='auto')` (`sklearn/linear_model/_omp.py:742-753`,
// `_parameter_constraints` `:735-740`) — including the two `Option`-typed stopping
// criteria — AND the fitted `coef_`/`intercept_` getters (sklearn `coef_` shape
// `(n_features,)` for the single-target fit, `intercept_` scalar, `_omp.py:814-815`).
// The greedy-Cholesky OMP MATH lives in
// `ferrolearn_linear::OrthogonalMatchingPursuit` (`with_n_nonzero_coefs`/
// `with_tol`/`with_fit_intercept`, `omp.rs`); this is the thin marshalling shim and
// the non-test production consumer of that core API (R-DEFER-1). coef_/intercept_
// match the live sklearn oracle to ~1e-12 (`ferrolearn_linear` omp.rs REQ-1/2/5
// SHIPPED).
//
// `precompute` (sklearn `'auto'`/`True`/`False`, `_omp.py:739`) is purely a
// Gram-matrix speed optimization: `_pre_fit` either passes a precomputed
// `X.T@X`/`X.T@y` to `orthogonal_mp_gram` or runs `orthogonal_mp` directly, but
// BOTH compute the IDENTICAL OMP solution (`_omp.py:791-813`). ferrolearn's core
// never uses a Gram path, so its result is identical under every `precompute`
// setting; the binding therefore ACCEPTS any `precompute` value (string or bool)
// and ignores it. It is a `BaseEstimator` param held on the `_extras.py` wrapper
// for `get_params`/`clone()` round-trip, so the Rust class need not store it. The
// Rust core exposes no `n_iter_`/`n_nonzero_coefs_` (sklearn `_omp.py:785-789`/
// `:792`) — those stay NOT-STARTED (`ferrolearn_linear` omp.rs REQ-7 #491), so no
// getter is faked for them.
#[pyclass(name = "_RsOrthogonalMatchingPursuit")]
pub struct RsOrthogonalMatchingPursuit {
    n_nonzero_coefs: Option<usize>,
    tol: Option<f64>,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedOMP<f64>>,
}

#[pymethods]
impl RsOrthogonalMatchingPursuit {
    // `precompute` is accepted (any string/bool, `_omp.py:739`) and ignored — it
    // is a Gram-matrix speed knob that does NOT change the OMP solution
    // (`_omp.py:791-813`); the `get_params`/`clone` round-trip lives on the Python
    // wrapper. `_extras.py` always passes the sklearn default `'auto'`.
    #[new]
    #[pyo3(signature = (n_nonzero_coefs=None, tol=None, fit_intercept=true, precompute=None))]
    fn new(
        n_nonzero_coefs: Option<usize>,
        tol: Option<f64>,
        fit_intercept: bool,
        precompute: Option<pyo3::Py<pyo3::PyAny>>,
    ) -> Self {
        let _ = precompute;
        Self {
            n_nonzero_coefs,
            tol,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let mut model = ferrolearn_linear::OrthogonalMatchingPursuit::<f64>::new()
            .with_fit_intercept(self.fit_intercept);
        // sklearn applies `n_nonzero_coefs`/`tol` only when set; the `None`/`None`
        // default (`max(int(0.1 * n_features), 1)`) is handled INSIDE the core
        // (`omp.rs` Fit, mirroring `_omp.py:785`), so only thread the Some values
        // — leaving both unset preserves the core default path.
        if let Some(n) = self.n_nonzero_coefs {
            model = model.with_n_nonzero_coefs(n);
        }
        if let Some(t) = self.tol {
            model = model.with_tol(t);
        }
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }

    // sklearn `coef_` (`_omp.py:814` `self.coef_ = coef_.T`): the (n_features,)
    // parameter vector for the single-target fit.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    // sklearn `intercept_` (`_omp.py:815` `self._set_intercept(...)`): the scalar
    // independent term for the single-target fit.
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }
}

// Lars (#2174): hand-written pyclass (rather than a thin `py_regressor!`
// invocation) so the binding surfaces sklearn's full keyword-only constructor
// `(fit_intercept=True, verbose=False, precompute='auto', n_nonzero_coefs=500,
// eps=np.finfo(float).eps, copy_X=True, fit_path=True, jitter=None,
// random_state=None)` (`sklearn/linear_model/_least_angle.py:1047-1058`,
// `_parameter_constraints` `:1032-1042`) AND the fitted `coef_` (shape
// `(n_features,)`) / `intercept_` (scalar) getters (sklearn `coef_path[:, -1]`
// reduced to `coef_`, `_least_angle.py:1125-1131`; `intercept_` via
// `_set_intercept`). The equiangular-homotopy LARS MATH lives in
// `ferrolearn_linear::Lars` (`with_n_nonzero_coefs`/`with_fit_intercept`,
// `lars.rs`); this is the thin marshalling shim and the non-test production
// consumer of that core API (R-DEFER-1). coef_/intercept_ match the live sklearn
// oracle to ~1e-6 (`ferrolearn_linear` lars.rs REQ-1/3/4 SHIPPED).
//
// Of sklearn's 9 ctor params only `fit_intercept`/`n_nonzero_coefs` change the
// supported result, so only those two are threaded into the Rust core. The other
// seven are accepted (validated to sklearn's `_parameter_constraints` set in the
// `_extras.py` wrapper) and ignored on the supported path:
//   - `verbose`/`copy_X` — diagnostics / input-copy knobs: never affect coef_.
//   - `precompute` ('auto'/True/False/ndarray/None) — a Gram-matrix speed knob;
//     `_get_gram` (`_least_angle.py:1070-1079`) precomputes `X.T@X` but the LARS
//     path is identical with or without it. The core never uses a Gram path, so
//     every `precompute` yields the same fit.
//   - `eps` — the Cholesky regularization floor (`_least_angle.py:1037`); only a
//     numerical-stability guard, does not change the supported result.
//   - `fit_path` — when False sklearn skips storing the full `coef_path_` but the
//     final `coef_` is IDENTICAL (`_least_angle.py:1133-1151`); accepted.
//   - `random_state` — only consumed BY `jitter` (seeds the gaussian noise,
//     `_least_angle.py:1170-1175`); a no-op when `jitter is None`.
// `jitter` (a non-None Real >= 0) ADDS scaled gaussian noise to `y` before the
// fit (`_least_angle.py:1170-1175`), which DOES change coef_. The Rust core cannot
// replicate numpy's seeded normal draws (R-SUBSTRATE-5), so a non-None `jitter`
// raises `NotImplementedError` in the `_extras.py` wrapper (#2174-tracked); the
// `None` default is the normal supported path. The wrapper holds all 9 params for
// `get_params`/`clone` round-trip, so this Rust class need only store the two that
// reach the core.
#[pyclass(name = "_RsLars")]
pub struct RsLars {
    n_nonzero_coefs: usize,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedLars<f64>>,
}

#[pymethods]
impl RsLars {
    // sklearn's `n_nonzero_coefs=500` (`_least_angle.py:1053`) is an UPPER bound on
    // the active set; the path stops at `min(n_nonzero_coefs, n_features)`
    // (`_least_angle.py` lars_path `max_features`). The `_extras.py` wrapper clamps
    // it to `min(n_nonzero_coefs, n_features)` before the ABI so the core (which
    // errors when `n_nonzero_coefs > n_features`) sees a valid bound that yields
    // the same support as sklearn's cap.
    #[new]
    #[pyo3(signature = (n_nonzero_coefs, fit_intercept=true))]
    fn new(n_nonzero_coefs: usize, fit_intercept: bool) -> Self {
        Self {
            n_nonzero_coefs,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::Lars::<f64>::new()
            .with_n_nonzero_coefs(self.n_nonzero_coefs)
            .with_fit_intercept(self.fit_intercept);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }

    // sklearn `coef_` (`_least_angle.py:1125` `self.coef_[k] = coef_path[:, -1]`):
    // the `(n_features,)` parameter vector at the final LARS knot.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    // sklearn `intercept_` (`LinearModel._set_intercept`): the scalar independent
    // term (0.0 when `fit_intercept=False`).
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }
}

// LassoLars (#2174): hand-written pyclass so the binding surfaces sklearn's full
// constructor `(alpha=1.0, *, fit_intercept=True, verbose=False,
// precompute='auto', max_iter=500, eps=np.finfo(float).eps, copy_X=True,
// fit_path=True, positive=False, jitter=None, random_state=None)`
// (`sklearn/linear_model/_least_angle.py:1363-1376`, `_parameter_constraints`
// `:1353-1359`) — note `alpha` is positional-or-keyword FIRST, the rest
// keyword-only — AND the fitted `coef_`/`intercept_` getters. The LARS-Lasso
// homotopy MATH lives in `ferrolearn_linear::LassoLars`
// (`with_alpha`/`with_max_iter`/`with_fit_intercept`, `lars.rs`); this is the
// thin marshalling shim and the non-test production consumer of that core API
// (R-DEFER-1). coef_/intercept_ match the live sklearn oracle to ~1e-6
// (`ferrolearn_linear` lars.rs REQ-2/3/4 SHIPPED).
//
// `alpha`/`max_iter`/`fit_intercept` are threaded into the core. The accept-and-
// ignore params are the same as `Lars` (`verbose`/`precompute`/`eps`/`copy_X`/
// `fit_path`/`random_state`), validated in the `_extras.py` wrapper. `jitter` is
// `NotImplementedError` for non-None (seeded gaussian noise, R-SUBSTRATE-5).
// `positive` (`_least_angle.py:1357`/`:1374`) constrains all coefficients to be
// non-negative — a DIFFERENT optimization the Rust core does NOT implement
// (`lars.rs` has no `positive` builder), so `positive=True` raises
// `NotImplementedError` in the wrapper (NOT-STARTED #2174); `positive=False`
// (default) is the normal supported path.
#[pyclass(name = "_RsLassoLars")]
pub struct RsLassoLars {
    alpha: f64,
    max_iter: usize,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedLassoLars<f64>>,
}

#[pymethods]
impl RsLassoLars {
    #[new]
    #[pyo3(signature = (alpha=1.0, max_iter=500, fit_intercept=true))]
    fn new(alpha: f64, max_iter: usize, fit_intercept: bool) -> Self {
        Self {
            alpha,
            max_iter,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::LassoLars::<f64>::new()
            .with_alpha(self.alpha)
            .with_max_iter(self.max_iter)
            .with_fit_intercept(self.fit_intercept);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }

    // sklearn `coef_` (`_least_angle.py:1125`): the `(n_features,)` Lasso-LARS
    // parameter vector interpolated at `alpha`.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    // sklearn `intercept_` (`LinearModel._set_intercept`): the scalar independent
    // term (0.0 when `fit_intercept=False`).
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        use ferrolearn_core::introspection::HasCoefficients;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }
}

// RANSACRegressor (#2178): hand-written pyclass (rather than a thin
// `py_regressor!` invocation) so the binding surfaces sklearn's RANSAC
// constructor surface (`sklearn/linear_model/_ransac.py:288-315`,
// `_parameter_constraints` `:260-286`) over the DEFAULT `LinearRegression` base
// (`_ransac.py:380`, `estimator=None` → `LinearRegression()`) AND the fitted
// `coef_`/`intercept_` (from the refit base estimator, `_ransac.py:597-605`) and
// `inlier_mask_` (`_ransac.py:201`) getters. The RANSAC MATH (sampling loop, MAD
// threshold, inlier classification, n_inliers→R² selection, single final refit)
// lives in `ferrolearn_linear::RANSACRegressor<f64, LinearRegression<f64>>`
// (`ransac.rs`, REQ-1..6/9 SHIPPED); this is the thin marshalling shim and the
// non-test production consumer of that core API (R-DEFER-1).
//
// The base estimator is FIXED to `ferrolearn_linear::LinearRegression::<f64>::new()`
// — sklearn's default `fit_intercept=True` LinearRegression (`_ransac.py:380`).
// The core `RANSACRegressor<F, E>` is generic over `E`, but the Rust core has no
// estimator-pluggability across the Python ABI, so only the default base is bound
// (the `_extras.py` wrapper rejects a non-LinearRegression `estimator` with
// `NotImplementedError`). `min_samples` is typed `Option<usize>`: the wrapper
// resolves sklearn's `None`→`n_features+1` and `float`→`ceil(min_samples*n_samples)`
// BEFORE the ABI (it has X at fit time), so the core sees only an integer count or
// `None` (which the core itself defaults to `n_features+1`, `ransac.rs` `fit`).
// `residual_threshold` is `Option<f64>` (None → the core's MAD-of-y default,
// `ransac.rs` REQ-2). `random_state` is `Option<u64>` threaded to the core's
// `StdRng` seed (RNG-substrate caveat #2118: the Rust Fisher-Yates RNG ≠ numpy
// MT19937, so for a GIVEN seed the drawn subsets differ from sklearn — but on
// well-separated data the best inlier set is UNIQUE, so the refit coef_ converges
// to sklearn's regardless of RNG, given sufficient `max_trials`).
//
// `n_trials_`/`n_skips_*`/`estimator_` are NOT exposed: the Rust core does not
// track the trial count (ransac.rs REQ-7 NOT-STARTED #515) nor surface the refit
// base as a wrapped fitted type (REQ-10 NOT-STARTED #518), so no getter is faked
// for them. `loss='squared_error'` and the non-default stop_*/max_skips knobs are
// rejected by the `_extras.py` wrapper (`NotImplementedError`, ransac.rs REQ-7/8).
#[pyclass(name = "_RsRANSACRegressor")]
pub struct RsRANSACRegressor {
    // Resolved integer count or None (the wrapper resolves float/None before the
    // ABI; None lets the core default to n_features+1).
    min_samples: Option<usize>,
    residual_threshold: Option<f64>,
    max_trials: usize,
    random_state: Option<u64>,
    // Number of features seen at fit (needed to size the coef_ probe; the core
    // fitted type exposes no n_features accessor — ransac.rs REQ-10 NOT-STARTED).
    n_features: usize,
    fitted: Option<
        ferrolearn_linear::FittedRANSACRegressor<ferrolearn_linear::FittedLinearRegression<f64>>,
    >,
}

impl RsRANSACRegressor {
    /// Recover the refit base estimator's intercept as `predict(0_vector)`.
    /// The base is a `FittedLinearRegression` whose `predict` is the exact affine
    /// map `X·coef + intercept` (linear_regression.rs REQ-2), so a zero row yields
    /// the intercept exactly. The core keeps the base private (ransac.rs REQ-10
    /// NOT-STARTED #518); this reads it through the only public surface (`predict`).
    fn recover_intercept(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let zero = ndarray::Array2::<f64>::zeros((1, self.n_features));
        let pred = fitted
            .predict(&zero)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(pred[0])
    }

    /// Recover the refit base estimator's coefficient vector as
    /// `coef_[j] = predict(e_j) − intercept_`, probing one unit row per feature.
    /// Exact (to float round-off) because the base `predict` is affine.
    fn recover_coef(&self) -> PyResult<Array1<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let intercept = self.recover_intercept()?;
        // One probe row per feature: row j is the unit vector e_j.
        let mut probe = ndarray::Array2::<f64>::zeros((self.n_features, self.n_features));
        for j in 0..self.n_features {
            probe[[j, j]] = 1.0;
        }
        let preds = fitted
            .predict(&probe)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let coef = preds.mapv(|p| p - intercept);
        Ok(coef)
    }
}

#[pymethods]
impl RsRANSACRegressor {
    #[new]
    #[pyo3(signature = (min_samples=None, residual_threshold=None, max_trials=100,
                        random_state=None))]
    fn new(
        min_samples: Option<usize>,
        residual_threshold: Option<f64>,
        max_trials: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            min_samples,
            residual_threshold,
            max_trials,
            random_state,
            n_features: 0,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        self.n_features = x_nd.ncols();
        // The default base estimator: sklearn's `LinearRegression()` with
        // `fit_intercept=True` (`_ransac.py:380`).
        let base = ferrolearn_linear::LinearRegression::<f64>::new();
        let mut model =
            ferrolearn_linear::RANSACRegressor::new(base).with_max_trials(self.max_trials);
        if let Some(ms) = self.min_samples {
            model = model.with_min_samples(ms);
        }
        if let Some(t) = self.residual_threshold {
            model = model.with_residual_threshold(t);
        }
        if let Some(seed) = self.random_state {
            model = model.with_random_state(seed);
        }
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }

    // sklearn `estimator_.coef_` (`_ransac.py:602-605`, the refit base estimator's
    // coefficients): the `(n_features,)` vector of the LinearRegression fitted on
    // the best inlier set.
    //
    // The Rust core's `FittedRANSACRegressor` keeps the refit base estimator
    // PRIVATE and exposes no `estimator_`/`coefficients()` accessor (ransac.rs
    // REQ-10 NOT-STARTED #518; the manifest forbids touching ransac.rs). The base
    // IS a `FittedLinearRegression`, whose `predict` is the EXACT affine map
    // `X·coef + intercept` (linear_regression.rs `Predict`, REQ-2). So we recover
    // the coefficients EXACTLY (to float round-off) from probe predictions:
    // `intercept_ = predict(0)`, `coef_[j] = predict(e_j) − intercept_`. This is
    // not a refit — it reads the already-fitted base through its only public
    // surface. (When ransac.rs ships REQ-10, swap this for a direct getter.)
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self.recover_coef()?;
        Ok(ndarray1_to_numpy(py, &coef))
    }

    // sklearn `estimator_.intercept_` (`_ransac.py:602-605`): the scalar
    // independent term of the refit base LinearRegression. Recovered as
    // `predict(0_vector)` (see `coef_`).
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.recover_intercept()
    }

    // sklearn `inlier_mask_` (`_ransac.py:201` / `:589`): the boolean mask of the
    // winning subset model (True == inlier, `ransac.rs` REQ-5).
    #[getter]
    fn inlier_mask_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(PyArray1::from_slice(py, fitted.inlier_mask()))
    }
}

// IsolationForest (#2180): hand-written pyclass (rather than a thin
// `py_transformer!`/`py_regressor!` invocation) so the binding surfaces the
// sklearn `OutlierMixin` method surface — `predict` (±1, `_iforest.py:357-378`),
// `score_samples` (`_iforest.py:412-451`), `decision_function`
// (`_iforest.py:380-410`) — over the SAME fitted handle, plus the fitted
// `offset_` (`_iforest.py:344/353`) and `n_features_in_` attributes. The anomaly
// MATH (isolation-tree build, `c(n)` average path length, the `-2^(-mean/c)`
// score, the `offset_` percentile) lives in
// `ferrolearn_tree::IsolationForest`/`FittedIsolationForest` (isolation_forest.rs,
// REQ-1..9 SHIPPED); this is the thin marshalling shim and the non-test
// production consumer of that core API (R-DEFER-1).
//
// Constructor surface: the wrapper resolves sklearn's `max_samples`
// (`'auto'`→`min(256,n)`, int→as-is, float→`int(f*n)`, `_iforest.py:303-318`)
// and `contamination` (`'auto'`→`with_contamination_auto()`, float→
// `with_contamination(f)`, `_iforest.py:341-353`) BEFORE this ABI (the core takes
// a `usize` `max_samples` and the `Contamination` enum). `n_estimators`/
// `max_samples`/`random_state` thread through directly.
//
// RNG-substrate caveat (#2118/isolation_forest.rs RNG-boundary #730): the Rust
// core's subsample + split draws use `StdRng`, NOT numpy's MT19937, so for a
// GIVEN `random_state` the trees (and thus the exact `score_samples` values)
// DIFFER from sklearn. The STRUCTURAL contract still holds and matches sklearn:
// `predict ∈ {-1,+1}`, `score_samples ∈ [-1,0]`,
// `decision_function == score_samples − offset_`, `predict == −1 where df<0 else
// +1`, `offset_ == −0.5` for `contamination='auto'`, and the anomaly RANKING
// (injected far outliers score lower than inliers). Exact score parity is the
// RNG-substrate limitation, NOT a binding bug.
#[pyclass(name = "_RsIsolationForest")]
pub struct RsIsolationForest {
    n_estimators: usize,
    // Resolved integer count (the wrapper resolves 'auto'/float→int before the
    // ABI). The core re-clamps to `min(max_samples, n_samples)` internally
    // (isolation_forest.rs `fit`), matching sklearn's int-max_samples warn-clamp
    // (`_iforest.py:307-314`).
    max_samples: usize,
    // None == sklearn `contamination='auto'`; Some(f) == the float path.
    contamination: Option<f64>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedIsolationForest<f64>>,
}

#[pymethods]
impl RsIsolationForest {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_samples=256, contamination=None,
                        random_state=None))]
    fn new(
        n_estimators: usize,
        max_samples: usize,
        contamination: Option<f64>,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_estimators,
            max_samples,
            contamination,
            random_state,
            fitted: None,
        }
    }

    // sklearn `fit(X, y=None)` (`_iforest.py:269`): `y` is ignored. The wrapper
    // passes only `X`; the core's `Fit<Array2<f64>, ()>` takes unit `y`.
    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let mut model = ferrolearn_tree::IsolationForest::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_max_samples(self.max_samples);
        model = match self.contamination {
            // sklearn `contamination='auto'` (`_iforest.py:341-345`, offset_=-0.5).
            None => model.with_contamination_auto(),
            // sklearn `contamination=<float>` (`_iforest.py:353`, offset_=percentile).
            Some(c) => model.with_contamination(c),
        };
        if let Some(seed) = self.random_state {
            model = model.with_random_state(seed);
        }
        let fitted = model
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    // sklearn `predict` (`_iforest.py:357-378`): +1 inlier / -1 outlier. The core
    // `Predict::Output` is `Array1<isize>`; marshalled to a numpy int64 array.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let preds_i64 = preds.mapv(|p| p as i64);
        Ok(PyArray1::from_array(py, &preds_i64))
    }

    // sklearn `score_samples` (`_iforest.py:412-451`): the opposite of the paper
    // anomaly score, in `[-1, 0]` (higher == more normal).
    fn score_samples<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let scores = fitted
            .score_samples(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &scores))
    }

    // sklearn `decision_function` (`_iforest.py:380-410`):
    // `score_samples(X) - offset_`; `< 0` == outlier.
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let df = fitted
            .decision_function(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &df))
    }

    // sklearn `offset_` (`_iforest.py:344/353`): the decision-threshold offset
    // (-0.5 for `contamination='auto'`, else the percentile of train scores).
    #[getter]
    fn offset_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.offset())
    }

    // sklearn `n_features_in_` (set by `_validate_data` at fit): the number of
    // features seen during fit. The core fitted type exposes it as `n_features()`.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_features())
    }
}

// ===========================================================================
// Tree regressors
// ===========================================================================

py_regressor!(
    RsDecisionTreeRegressor, "_RsDecisionTreeRegressor",
    ferrolearn_tree::FittedDecisionTreeRegressor<f64>,
    (max_depth: Option<usize> = None, min_samples_split: usize = 2,
     min_samples_leaf: usize = 1),
    {
        ferrolearn_tree::DecisionTreeRegressor::<f64>::new()
            .with_max_depth(max_depth).with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
    }
);

#[pyclass(name = "_RsRandomForestRegressor")]
pub struct RsRandomForestRegressor {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedRandomForestRegressor<f64>>,
}

#[pymethods]
impl RsRandomForestRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, min_samples_split=2,
                        min_samples_leaf=1, random_state=None))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let mut m = ferrolearn_tree::RandomForestRegressor::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_min_samples_split(self.min_samples_split)
            .with_min_samples_leaf(self.min_samples_leaf);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsExtraTreesRegressor")]
pub struct RsExtraTreesRegressor {
    n_estimators: usize,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedExtraTreesRegressor<f64>>,
}

#[pymethods]
impl RsExtraTreesRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, random_state=None))]
    fn new(n_estimators: usize, max_depth: Option<usize>, random_state: Option<u64>) -> Self {
        Self {
            n_estimators,
            max_depth,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let mut m = ferrolearn_tree::ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsGradientBoostingRegressor")]
pub struct RsGradientBoostingRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedGradientBoostingRegressor<f64>>,
}

#[pymethods]
impl RsGradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=Some(3), random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_estimators,
            learning_rate,
            max_depth,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let mut m = ferrolearn_tree::GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsHistGradientBoostingRegressor")]
pub struct RsHistGradientBoostingRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedHistGradientBoostingRegressor<f64>>,
}

#[pymethods]
impl RsHistGradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=None, random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_estimators,
            learning_rate,
            max_depth,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let mut m = ferrolearn_tree::HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }
}

// ---------------------------------------------------------------------------
// KNeighborsRegressor (#2147): hand-written pyclass exposing the FULL sklearn
// constructor surface (`neighbors/_regression.py:178-189`:
// `(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2,
//  metric='minkowski', metric_params=None, n_jobs=None)`). Replaces the thin
// `py_regressor!(... (n_neighbors: usize = 5) ...)` invocation. `weights`
// CHANGES PREDICTIONS (inverse-distance-weighted average of neighbor targets);
// the Rust core already supports it (`knn.rs:1160-1175` `with_n_neighbors`/
// `with_weights`/`with_algorithm`). Mirrors `RsKNeighborsClassifier`
// (classifiers.rs) but `fit(X, y: f64)` over continuous targets and `predict`
// returns `PyArray1<f64>` — same fit/predict ABI the `py_regressor!` macro
// emitted, so `_extras.py`'s plumbing is unchanged.
#[pyclass(name = "_RsKNeighborsRegressor")]
pub struct RsKNeighborsRegressor {
    n_neighbors: usize,
    weights: String,
    algorithm: String,
    // `leaf_size`/`n_jobs` are stored for ABI parity with sklearn
    // (`neighbors/_regression.py:184,188`) but are search-perf / threading
    // knobs only — they do NOT change the predicted result, so they are
    // accepted and held without affecting the fitted model.
    #[allow(
        dead_code,
        reason = "ABI-parity no-op: leaf_size affects only tree-build perf, not the result (sklearn neighbors/_regression.py:184)"
    )]
    leaf_size: usize,
    p: f64,
    metric: String,
    #[allow(
        dead_code,
        reason = "ABI-parity no-op: n_jobs is a threading knob, not a result-affecting param (sklearn neighbors/_regression.py:188)"
    )]
    n_jobs: Option<i64>,
    fitted: Option<ferrolearn_neighbors::FittedKNeighborsRegressor<f64>>,
}

#[pymethods]
impl RsKNeighborsRegressor {
    #[new]
    #[pyo3(signature = (
        n_neighbors=5,
        weights="uniform".to_string(),
        algorithm="auto".to_string(),
        leaf_size=30,
        p=2.0,
        metric="minkowski".to_string(),
        n_jobs=None,
    ))]
    fn new(
        n_neighbors: usize,
        weights: String,
        algorithm: String,
        leaf_size: usize,
        p: f64,
        metric: String,
        n_jobs: Option<i64>,
    ) -> Self {
        Self {
            n_neighbors,
            weights,
            algorithm,
            leaf_size,
            p,
            metric,
            n_jobs,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        use ferrolearn_neighbors::{Algorithm, Weights};

        // weights: only 'uniform'/'distance' are supported in the Rust core
        // (`knn.rs::Weights`); callable weights are NOT-STARTED (#876).
        // 'distance' weights neighbor targets by inverse distance — it CHANGES
        // the prediction (sklearn/neighbors/_regression.py:43-45).
        let weights = match self.weights.as_str() {
            "uniform" => Weights::Uniform,
            "distance" => Weights::Distance,
            other => {
                // An invalid weights STRING is a parameter-value error, not an
                // unsupported-feature error: sklearn raises InvalidParameterError
                // (a ValueError subclass). Callable weights are intercepted in
                // the Python shim (NotImplementedError #876) and never reach here.
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "weights={other:?} is invalid (expected 'uniform' or 'distance')"
                )));
            }
        };

        // algorithm: 'auto'/'brute'/'kd_tree'/'ball_tree' all produce the SAME
        // predict (search strategy only). The Rust core has no exact
        // 'ball_tree'-distinct regressor variant, so it maps to Auto (identical
        // result), mirroring `RsKNeighborsClassifier`.
        let algorithm = match self.algorithm.as_str() {
            "auto" => Algorithm::Auto,
            "brute" => Algorithm::BruteForce,
            "kd_tree" => Algorithm::KdTree,
            "ball_tree" => Algorithm::Auto,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid algorithm: {other:?} (expected one of \
                     'auto', 'brute', 'kd_tree', 'ball_tree')"
                )));
            }
        };

        // sklearn validates `p` against Interval(Real, 0, None, closed="right")
        // == (0, inf] UNCONDITIONALLY at fit, BEFORE the metric is consulted
        // (sklearn/neighbors/_base.py:400), so p <= 0 (and NaN) is an
        // InvalidParameterError (a ValueError subclass) for ANY metric (#2151).
        if self.p <= 0.0 || self.p.is_nan() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "p={} is invalid (must be in the range (0, inf])",
                self.p
            )));
        }

        // The Rust core is Euclidean-only. sklearn consumes `p` ONLY when
        // metric == "minkowski" (sklearn/neighbors/_base.py:526-538); 'euclidean'
        // and its alias 'l2' ARE the Euclidean (p=2) distance and accept any
        // valid p (p is ignored). metric == "minkowski" requires p == 2; other p
        // (true Minkowski) and all non-Euclidean metrics are NOT-STARTED (#876).
        match self.metric.as_str() {
            "euclidean" | "l2" => {}
            "minkowski" => {
                if self.p != 2.0 {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                        "p={} not supported for metric='minkowski' (Euclidean-only, \
                         p=2; Minkowski-p NOT-STARTED #876)",
                        self.p
                    )));
                }
            }
            other => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "metric={other:?} not supported (only minkowski(p=2)/euclidean/l2; \
                     NOT-STARTED #876)"
                )));
            }
        }

        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_neighbors::KNeighborsRegressor::<f64>::new()
            .with_n_neighbors(self.n_neighbors)
            .with_weights(weights)
            .with_algorithm(algorithm);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }
}

py_regressor!(
    RsKernelRidge, "_RsKernelRidge",
    ferrolearn_kernel::FittedKernelRidge<f64>,
    (alpha: f64 = 1.0),
    {
        ferrolearn_kernel::KernelRidge::<f64>::new().with_alpha(alpha)
    }
);

// ===========================================================================
// Linear classifiers
// ===========================================================================

py_classifier!(
    RsRidgeClassifier, "_RsRidgeClassifier",
    ferrolearn_linear::FittedRidgeClassifier<f64>,
    (alpha: f64 = 1.0, fit_intercept: bool = true),
    {
        ferrolearn_linear::RidgeClassifier::<f64>::new()
            .with_alpha(alpha).with_fit_intercept(fit_intercept)
    }
);

py_classifier!(
    RsLinearSVC, "_RsLinearSVC",
    ferrolearn_linear::FittedLinearSVC<f64>,
    (c: f64 = 1.0, max_iter: usize = 1000, tol: f64 = 1e-4),
    {
        ferrolearn_linear::LinearSVC::<f64>::new()
            .with_c(c).with_max_iter(max_iter).with_tol(tol)
    }
);

py_classifier!(
    RsQDA, "_RsQDA",
    ferrolearn_linear::FittedQDA<f64>,
    (reg_param: f64 = 0.0),
    {
        ferrolearn_linear::QDA::<f64>::new().with_reg_param(reg_param)
    }
);

// ===========================================================================
// Bayes (extras)
// ===========================================================================

py_classifier!(
    RsMultinomialNB, "_RsMultinomialNB",
    ferrolearn_bayes::FittedMultinomialNB<f64>,
    (alpha: f64 = 1.0, fit_prior: bool = true),
    {
        ferrolearn_bayes::MultinomialNB::<f64>::new()
            .with_alpha(alpha).with_fit_prior(fit_prior)
    }
);

// Discrete-NB fitted attributes (#2103): the four `_BaseDiscreteNB` attrs
// sklearn exposes (`feature_log_prob_`/`class_log_prior_`/`feature_count_`/
// `class_count_`, sklearn/naive_bayes.py:885-892/580-602/880-883). Added as a
// SECOND `#[pymethods]` block (requires pyo3 `multiple-pymethods`) so the
// `py_classifier!` macro stays UNCHANGED. The Rust fitted type already computes
// all four (FittedMultinomialNB, multinomial.rs:489/499/509/520).
#[pymethods]
impl RsMultinomialNB {
    #[getter]
    fn feature_log_prob_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.feature_log_prob()))
    }

    #[getter]
    fn class_log_prior_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.class_log_prior()))
    }

    #[getter]
    fn feature_count_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.feature_count()))
    }

    #[getter]
    fn class_count_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let cc = fitted.class_count();
        Ok(ndarray1_to_numpy(py, &cc))
    }
}

py_classifier!(
    RsBernoulliNB, "_RsBernoulliNB",
    ferrolearn_bayes::FittedBernoulliNB<f64>,
    (alpha: f64 = 1.0, fit_prior: bool = true, binarize: Option<f64> = Some(0.0)),
    {
        // sklearn `binarize : float or None, default=0.0` (naive_bayes.py:1076,1156);
        // `None` SKIPS binarization (naive_bayes.py:1179,1185). pyo3 maps Python
        // `None` -> `Option::None` and a float -> `Some(f64)`, so both
        // `_RsBernoulliNB(binarize=None)` and `_RsBernoulliNB(binarize=0.5)` work.
        ferrolearn_bayes::BernoulliNB::<f64>::new()
            .with_alpha(alpha).with_fit_prior(fit_prior).with_binarize_option(binarize)
    }
);

// Discrete-NB fitted attributes (#2103): FittedBernoulliNB computes all four
// (bernoulli.rs:520/530/540/551). Second `#[pymethods]` block keeps the
// `py_classifier!` macro UNCHANGED. sklearn/naive_bayes.py:880-892/580-602.
#[pymethods]
impl RsBernoulliNB {
    #[getter]
    fn feature_log_prob_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.feature_log_prob()))
    }

    #[getter]
    fn class_log_prior_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.class_log_prior()))
    }

    #[getter]
    fn feature_count_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.feature_count()))
    }

    #[getter]
    fn class_count_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let cc = fitted.class_count();
        Ok(ndarray1_to_numpy(py, &cc))
    }
}

py_classifier!(
    RsComplementNB, "_RsComplementNB",
    ferrolearn_bayes::FittedComplementNB<f64>,
    (alpha: f64 = 1.0, fit_prior: bool = true, norm: bool = false),
    {
        ferrolearn_bayes::ComplementNB::<f64>::new()
            .with_alpha(alpha).with_fit_prior(fit_prior).with_norm(norm)
    }
);

// Discrete-NB fitted attributes (#2103): FittedComplementNB computes all four
// (complement.rs:535/586/545/556). `feature_log_prob_` is the `-logged`
// complement weight (positive) — this is what sklearn exposes
// (sklearn/naive_bayes.py:1032-1042). `class_log_prior()` and `class_count()`
// return OWNED `Array1<F>` here, so bind to a local before marshalling.
#[pymethods]
impl RsComplementNB {
    #[getter]
    fn feature_log_prob_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.feature_log_prob()))
    }

    #[getter]
    fn class_log_prior_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let clp = fitted.class_log_prior();
        Ok(ndarray1_to_numpy(py, &clp))
    }

    #[getter]
    fn feature_count_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.feature_count()))
    }

    #[getter]
    fn class_count_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let cc = fitted.class_count();
        Ok(ndarray1_to_numpy(py, &cc))
    }
}

// ===========================================================================
// Tree classifiers (extras)
// ===========================================================================

py_classifier!(
    RsExtraTreeClassifier, "_RsExtraTreeClassifier",
    ferrolearn_tree::FittedExtraTreeClassifier<f64>,
    (max_depth: Option<usize> = None, min_samples_split: usize = 2,
     min_samples_leaf: usize = 1),
    {
        ferrolearn_tree::ExtraTreeClassifier::<f64>::new()
            .with_max_depth(max_depth).with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
    }
);

#[pyclass(name = "_RsExtraTreesClassifier")]
pub struct RsExtraTreesClassifier {
    n_estimators: usize,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedExtraTreesClassifier<f64>>,
}

#[pymethods]
impl RsExtraTreesClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, random_state=None))]
    fn new(n_estimators: usize, max_depth: Option<usize>, random_state: Option<u64>) -> Self {
        Self {
            n_estimators,
            max_depth,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut m = ferrolearn_tree::ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsAdaBoostClassifier")]
pub struct RsAdaBoostClassifier {
    n_estimators: usize,
    learning_rate: f64,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedAdaBoostClassifier<f64>>,
}

#[pymethods]
impl RsAdaBoostClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=50, learning_rate=1.0, random_state=None))]
    fn new(n_estimators: usize, learning_rate: f64, random_state: Option<u64>) -> Self {
        Self {
            n_estimators,
            learning_rate,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut m = ferrolearn_tree::AdaBoostClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsGradientBoostingClassifier")]
pub struct RsGradientBoostingClassifier {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedGradientBoostingClassifier<f64>>,
}

#[pymethods]
impl RsGradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=Some(3), random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_estimators,
            learning_rate,
            max_depth,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut m = ferrolearn_tree::GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsHistGradientBoostingClassifier")]
pub struct RsHistGradientBoostingClassifier {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedHistGradientBoostingClassifier<f64>>,
}

#[pymethods]
impl RsHistGradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=None, random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_estimators,
            learning_rate,
            max_depth,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut m = ferrolearn_tree::HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsBaggingClassifier")]
pub struct RsBaggingClassifier {
    n_estimators: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedBaggingClassifier<f64>>,
}

#[pymethods]
impl RsBaggingClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=10, random_state=None))]
    fn new(n_estimators: usize, random_state: Option<u64>) -> Self {
        Self {
            n_estimators,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut m =
            ferrolearn_tree::BaggingClassifier::<f64>::new().with_n_estimators(self.n_estimators);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

py_classifier!(
    RsNearestCentroid,
    "_RsNearestCentroid",
    ferrolearn_neighbors::FittedNearestCentroid<f64>,
    (),
    { ferrolearn_neighbors::NearestCentroid::<f64>::new() }
);

// ===========================================================================
// Cluster (extras) — these don't fit the supervised pattern; predict is fitted.labels_
// ===========================================================================

#[pyclass(name = "_RsMiniBatchKMeans")]
pub struct RsMiniBatchKMeans {
    n_clusters: usize,
    max_iter: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_cluster::FittedMiniBatchKMeans<f64>>,
}

#[pymethods]
impl RsMiniBatchKMeans {
    #[new]
    #[pyo3(signature = (n_clusters=8, max_iter=100, random_state=None))]
    fn new(n_clusters: usize, max_iter: usize, random_state: Option<u64>) -> Self {
        Self {
            n_clusters,
            max_iter,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let mut m = ferrolearn_cluster::MiniBatchKMeans::<f64>::new(self.n_clusters)
            .with_max_iter(self.max_iter);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_usize_to_numpy(py, f.labels()))
    }
}

// DBSCAN metric/p surface (#2193): map the lowercased sklearn metric STRING to
// `ferrolearn_cluster::dbscan::DbscanMetric<f64>`, threading the Minkowski order
// `p` (`p=None` -> `p=2`, `sklearn/cluster/_dbscan.py:243-246/354`). sklearn's
// `_parameter_constraints["metric"]` is `StrOptions(set(_VALID_METRICS) |
// {"precomputed"})` (`_dbscan.py:334-337`): only the aliases sklearn itself
// recognizes are accepted; an unknown string is an `InvalidParameterError` (a
// `ValueError` subclass), surfaced here as `PyValueError`. The aliases mirror
// `sklearn.metrics.pairwise._VALID_METRICS`: `euclidean`/`l2` -> Euclidean,
// `manhattan`/`l1`/`cityblock` -> Manhattan, `chebyshev` -> Chebyshev,
// `minkowski` -> Minkowski(p). `p` is IGNORED for every non-Minkowski metric
// (`_dbscan.py:411-418`, #2192) so `with_p` is NOT called on those branches;
// only `Minkowski` carries `p`. The metrics ferrolearn cannot compute
// (`metric_params` kwargs, `precomputed`, callable, the spatial metrics like
// `cosine`/`haversine`) are out of scope (NOT-STARTED, see `.design/python/extras.md`).
fn resolve_dbscan_metric(
    metric: &str,
    p: Option<f64>,
) -> PyResult<ferrolearn_cluster::dbscan::DbscanMetric<f64>> {
    use ferrolearn_cluster::dbscan::DbscanMetric;
    // sklearn's `metric` constraint is `StrOptions(set(_VALID_METRICS) |
    // {"precomputed"})` (`_dbscan.py:334-337`) — EXACT (case-sensitive) string
    // membership over lowercase names. So we match the raw string WITHOUT
    // `to_lowercase()`: `'Euclidean'`/`'L2'` are rejected by sklearn
    // (InvalidParameterError) and must be rejected here too (#2194).
    match metric {
        "euclidean" | "l2" => Ok(DbscanMetric::Euclidean),
        "manhattan" | "l1" | "cityblock" => Ok(DbscanMetric::Manhattan),
        "chebyshev" => Ok(DbscanMetric::Chebyshev),
        // To SELECT Minkowski-with-`p` the core requires
        // `with_metric(Minkowski(p))`, NOT `with_p` alone (#2192); construct the
        // variant directly with the order. The strict-positivity / NaN check on
        // `p` is enforced downstream by `Fit::fit` (-> `InvalidParameter` ->
        // `PyValueError`).
        //
        // `p=None` with `metric='minkowski'` is REJECTED, matching the LIVE
        // sklearn 1.5.2 oracle (R-CHAR-3, #2193): `DBSCAN(metric='minkowski',
        // p=None).fit` forwards `p=None` to `NearestNeighbors`, which raises
        // (`TypeError: None < 1`). We do NOT silently resolve to `p=2` (the
        // docstring's stated default) — the live behavior is to FAIL, and the
        // oracle, not the docstring, is the contract. ferrolearn raises a
        // `ValueError` (its param-error convention) requiring an explicit `p`.
        "minkowski" => match p {
            Some(pval) => Ok(DbscanMetric::Minkowski(pval)),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "metric='minkowski' requires an explicit p (p=None is rejected; \
                 sklearn raises TypeError for this combination)",
            )),
        },
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "metric == {other:?}, must be one of 'euclidean'/'l2', \
             'manhattan'/'l1'/'cityblock', 'chebyshev', or 'minkowski'."
        ))),
    }
}

#[pyclass(name = "_RsDBSCAN")]
pub struct RsDBSCAN {
    eps: f64,
    min_samples: usize,
    // sklearn `metric` (`_dbscan.py:350`, default `"euclidean"`): the neighbor
    // distance metric STRING, resolved to a `DbscanMetric<f64>` at fit time.
    metric: String,
    // sklearn `p` (`_dbscan.py:354`, default `None`): the Minkowski order, used
    // ONLY by `metric='minkowski'` and IGNORED otherwise (#2192). `None` -> 2.0
    // for Minkowski.
    p: Option<f64>,
    fitted: Option<ferrolearn_cluster::FittedDBSCAN<f64>>,
}

#[pymethods]
impl RsDBSCAN {
    #[new]
    #[pyo3(signature = (eps=0.5, min_samples=5, metric="euclidean".to_string(), p=None))]
    fn new(eps: f64, min_samples: usize, metric: String, p: Option<f64>) -> Self {
        Self {
            eps,
            min_samples,
            metric,
            p,
            fitted: None,
        }
    }

    // sklearn's `DBSCAN.fit(X, y=None, sample_weight=None)`
    // (`sklearn/cluster/_dbscan.py:370`). A `None` sample_weight is the
    // unweighted `len >= min_samples` core path; a `Some(w)` weight chains the
    // core builder `.with_sample_weight(w)` (the weighted `sum(w[neighbors]) >=
    // min_samples` path, `_dbscan.py:427-435`). A wrong-length `sample_weight`
    // surfaces from the core as a `FerroError::ShapeMismatch` -> `PyValueError`
    // (mirroring `_check_sample_weight`'s `ValueError`, `_dbscan.py:397`).
    //
    // The `metric`/`p` ctor params resolve to a `DbscanMetric<f64>` chained via
    // `.with_metric(...)` (`_dbscan.py:411-422`, neighbor test `dist <= eps`):
    // an unknown metric string is rejected as `PyValueError` (sklearn's
    // `InvalidParameterError`), a non-positive/NaN Minkowski `p` surfaces from
    // `Fit::fit` as `InvalidParameter` -> `PyValueError`. `p` is IGNORED for the
    // non-Minkowski metrics (#2192), matching sklearn.
    #[pyo3(signature = (x, sample_weight=None))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let resolved = resolve_dbscan_metric(&self.metric, self.p)?;
        let mut m = ferrolearn_cluster::DBSCAN::<f64>::new(self.eps)
            .with_min_samples(self.min_samples)
            .with_metric(resolved);
        if let Some(w) = sample_weight {
            m = m.with_sample_weight(numpy1_to_ndarray(w));
        }
        let fitted = m
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let lbls = f.labels();
        let arr: Array1<i64> = lbls.mapv(|v| v as i64);
        Ok(PyArray1::from_array(py, &arr))
    }

    /// Indices of the core samples, mirroring sklearn's
    /// `core_sample_indices_ = np.where(core_samples)[0]`
    /// (`sklearn/cluster/_dbscan.py:438`); a 1-D ascending int array.
    #[getter]
    fn core_sample_indices_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let arr: Array1<i64> = f.core_sample_indices().iter().map(|&i| i as i64).collect();
        Ok(PyArray1::from_array(py, &arr))
    }

    /// Coordinates of the core samples (rows of `X` at `core_sample_indices_`),
    /// shape `(n_core, n_features)`, mirroring sklearn's
    /// `components_ = X[core_sample_indices_].copy()`
    /// (`sklearn/cluster/_dbscan.py:441-446`).
    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(PyArray2::from_array(py, f.components()))
    }
}

#[pyclass(name = "_RsAgglomerativeClustering")]
pub struct RsAgglomerativeClustering {
    n_clusters: usize,
    fitted: Option<ferrolearn_cluster::FittedAgglomerativeClustering<f64>>,
}

#[pymethods]
impl RsAgglomerativeClustering {
    #[new]
    #[pyo3(signature = (n_clusters=2))]
    fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let m = ferrolearn_cluster::AgglomerativeClustering::<f64>::new(self.n_clusters);
        let fitted = m
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_usize_to_numpy(py, f.labels()))
    }
}

#[pyclass(name = "_RsBirch")]
pub struct RsBirch {
    n_clusters: Option<usize>,
    threshold: f64,
    fitted: Option<ferrolearn_cluster::FittedBirch<f64>>,
}

#[pymethods]
impl RsBirch {
    #[new]
    #[pyo3(signature = (n_clusters=None, threshold=0.5))]
    fn new(n_clusters: Option<usize>, threshold: f64) -> Self {
        Self {
            n_clusters,
            threshold,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let mut m = ferrolearn_cluster::Birch::<f64>::new().with_threshold(self.threshold);
        if let Some(n) = self.n_clusters {
            m = m.with_n_clusters(n);
        }
        let fitted = m
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_usize_to_numpy(py, f.labels()))
    }
}

#[pyclass(name = "_RsGaussianMixture")]
pub struct RsGaussianMixture {
    n_components: usize,
    max_iter: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_cluster::FittedGaussianMixture<f64>>,
}

#[pymethods]
impl RsGaussianMixture {
    #[new]
    #[pyo3(signature = (n_components=1, max_iter=100, random_state=None))]
    fn new(n_components: usize, max_iter: usize, random_state: Option<u64>) -> Self {
        Self {
            n_components,
            max_iter,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let mut m = ferrolearn_cluster::GaussianMixture::<f64>::new(self.n_components)
            .with_max_iter(self.max_iter);
        if let Some(s) = self.random_state {
            m = m.with_random_state(s);
        }
        let fitted = m
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

// FeatureAgglomeration (#943): hand-written `#[pyclass]` over
// `ferrolearn_cluster::FeatureAgglomeration<f64>`/`FittedFeatureAgglomeration<f64>`.
// FeatureAgglomeration is BOTH a transformer (transform/inverse_transform via
// sklearn's `AgglomerationTransform`/`TransformerMixin`,
// `sklearn/cluster/_feature_agglomeration.py:22-92`) AND exposes the hierarchical
// `labels_`/`children_`/`distances_`/`n_leaves_`/`n_connected_components_` fitted
// attributes delegated from the inner clustering over `X.T`
// (`sklearn/cluster/_agglomerative.py:1338-1340`). The Rust core is
// pre-existing/audited (`ferrolearn_cluster` feature_agglomeration.rs REQ-1..9
// SHIPPED); this is the thin marshalling shim and the non-test production
// consumer of `fit`/`transform`/`inverse_transform`/`labels()`/`children()`/etc.
//
// The wrapper resolves `linkage` ('ward'/'complete'/'average'/'single') and
// `pooling` ('mean'/'max') to the closed `AgglomerativeLinkage`/`PoolingFunc`
// enums BEFORE the ABI; an unrecognized string is a `ValueError` (sklearn's
// `InvalidParameterError ⊂ ValueError`, `_agglomerative.py:1290-1291`). The
// arbitrary-callable `pooling_func` (sklearn `[callable]`, `:1291`) and the
// `metric`/`memory`/`connectivity`/`compute_full_tree`/`distance_threshold`
// params the core lacks are handled in the `_extras.py` wrapper (REQ-7/#941).
#[pyclass(name = "_RsFeatureAgglomeration")]
pub struct RsFeatureAgglomeration {
    n_clusters: usize,
    linkage: String,
    pooling: String,
    compute_distances: bool,
    fitted: Option<ferrolearn_cluster::FittedFeatureAgglomeration<f64>>,
}

#[pymethods]
impl RsFeatureAgglomeration {
    #[new]
    #[pyo3(signature = (n_clusters=2, linkage="ward".to_string(),
                        pooling="mean".to_string(), compute_distances=false))]
    fn new(n_clusters: usize, linkage: String, pooling: String, compute_distances: bool) -> Self {
        Self {
            n_clusters,
            linkage,
            pooling,
            compute_distances,
            fitted: None,
        }
    }

    // sklearn `fit(X, y=None)` (`_agglomerative.py:1322`): `y` ignored. Maps the
    // `linkage`/`pooling` strings to the core enums; an unknown string is a
    // `ValueError` (sklearn `InvalidParameterError ⊂ ValueError`).
    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        use ferrolearn_cluster::{AgglomerativeLinkage, PoolingFunc};
        let linkage = match self.linkage.as_str() {
            "ward" => AgglomerativeLinkage::Ward,
            "complete" => AgglomerativeLinkage::Complete,
            "average" => AgglomerativeLinkage::Average,
            "single" => AgglomerativeLinkage::Single,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "linkage == {other:?}, must be one of 'ward', 'complete', 'average', 'single'."
                )));
            }
        };
        let pooling = match self.pooling.as_str() {
            "mean" => PoolingFunc::Mean,
            "max" => PoolingFunc::Max,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "pooling_func == {other:?}, only 'mean'/np.mean and 'max'/np.max are \
                     supported (arbitrary-callable pooling is NOT-STARTED #941)."
                )));
            }
        };
        let x_nd = numpy2_to_ndarray(x);
        let model = ferrolearn_cluster::FeatureAgglomeration::<f64>::new(self.n_clusters)
            .with_linkage(linkage)
            .with_pooling_func(pooling)
            .with_compute_distances(self.compute_distances);
        let fitted = model
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    // sklearn `AgglomerationTransform.transform`
    // (`_feature_agglomeration.py:32-64`): pool features per cluster, output
    // shape `(n_samples, n_clusters)`, columns ordered by ascending label index.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let xt = fitted
            .transform(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &xt))
    }

    // sklearn `AgglomerationTransform.inverse_transform`
    // (`_feature_agglomeration.py:66-92`): broadcast each cluster's pooled value
    // back to every member feature, output shape `(n_samples, n_features)`.
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        xred: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let xred_nd = numpy2_to_ndarray(xred);
        let xi = fitted
            .inverse_transform(&xred_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &xi))
    }

    // sklearn `labels_` (`_agglomerative.py:1339`, delegated from the inner
    // AgglomerativeClustering over `X.T`): per-feature cluster assignment, 1-D
    // int array of length `n_features`.
    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_usize_to_numpy(py, fitted.labels()))
    }

    // sklearn `n_clusters_` (`_agglomerative.py:1083`): number of feature clusters.
    #[getter]
    fn n_clusters_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_clusters())
    }

    // sklearn `children_` (`_agglomerative.py:1086`): the merge history, an
    // `(n_features - 1, 2)` int array.
    #[getter]
    fn children_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let children = fitted.children();
        let n = children.len();
        let mut arr = Array2::<i64>::zeros((n, 2));
        for (i, &(a, b)) in children.iter().enumerate() {
            arr[[i, 0]] = a as i64;
            arr[[i, 1]] = b as i64;
        }
        Ok(PyArray2::from_array(py, &arr))
    }

    // sklearn `distances_` (`_agglomerative.py:1093-1095`): per-merge linkage
    // distances, a 1-D f64 array, or `None` if `compute_distances` was not set.
    #[getter]
    fn distances_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.distances().map(|d| ndarray1_to_numpy(py, d)))
    }

    // sklearn `n_leaves_` (`_agglomerative.py:1090`): number of leaves in the
    // inner hierarchical tree (== n_features for the unstructured path).
    #[getter]
    fn n_leaves_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_leaves())
    }

    // sklearn `n_connected_components_` (`_agglomerative.py:1089`): always 1 for
    // the unstructured `connectivity=None` path.
    #[getter]
    fn n_connected_components_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_connected_components())
    }

    // sklearn `n_features_in_` (set by `_validate_data` at fit): number of
    // features seen during fit. The core fitted type exposes it as `n_features()`.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_features())
    }
}

// ===========================================================================
// Decomp (extras)
// ===========================================================================

py_transformer!(
    RsIncrementalPCA, "_RsIncrementalPCA",
    ferrolearn_decomp::FittedIncrementalPCA<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::IncrementalPCA::<f64>::new(n_components) }
);

py_transformer!(
    RsTruncatedSVD, "_RsTruncatedSVD",
    ferrolearn_decomp::FittedTruncatedSVD<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::TruncatedSVD::<f64>::new(n_components) }
);

py_transformer!(
    RsFastICA, "_RsFastICA",
    ferrolearn_decomp::FittedFastICA<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::FastICA::<f64>::new(n_components) }
);

py_transformer!(
    RsNMF, "_RsNMF",
    ferrolearn_decomp::FittedNMF<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::NMF::<f64>::new(n_components) }
);

py_transformer!(
    RsKernelPCA, "_RsKernelPCA",
    ferrolearn_decomp::FittedKernelPCA<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::KernelPCA::<f64>::new(n_components) }
);

py_transformer!(
    RsSparsePCA, "_RsSparsePCA",
    ferrolearn_decomp::FittedSparsePCA<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::SparsePCA::<f64>::new(n_components) }
);

py_transformer!(
    RsFactorAnalysis, "_RsFactorAnalysis",
    ferrolearn_decomp::FittedFactorAnalysis<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::FactorAnalysis::<f64>::new(n_components) }
);

// ===========================================================================
// Preprocess (extras)
// ===========================================================================

py_transformer!(
    RsMinMaxScaler,
    "_RsMinMaxScaler",
    ferrolearn_preprocess::FittedMinMaxScaler<f64>,
    (),
    { ferrolearn_preprocess::MinMaxScaler::<f64>::new() }
);

py_transformer!(
    RsMaxAbsScaler,
    "_RsMaxAbsScaler",
    ferrolearn_preprocess::FittedMaxAbsScaler<f64>,
    (),
    { ferrolearn_preprocess::MaxAbsScaler::<f64>::new() }
);

py_transformer!(
    RsRobustScaler,
    "_RsRobustScaler",
    ferrolearn_preprocess::FittedRobustScaler<f64>,
    (),
    { ferrolearn_preprocess::RobustScaler::<f64>::new() }
);

py_transformer!(
    RsPowerTransformer,
    "_RsPowerTransformer",
    ferrolearn_preprocess::FittedPowerTransformer<f64>,
    (),
    { ferrolearn_preprocess::PowerTransformer::<f64>::new() }
);

py_transformer!(
    RsNystroem,
    "_RsNystroem",
    ferrolearn_kernel::FittedNystroem<f64>,
    (),
    { ferrolearn_kernel::Nystroem::<f64>::new() }
);

py_transformer!(
    RsRBFSampler,
    "_RsRBFSampler",
    ferrolearn_kernel::FittedRBFSampler<f64>,
    (),
    { ferrolearn_kernel::RBFSampler::<f64>::new() }
);
