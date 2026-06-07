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
//! **14 SHIPPED / 5 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-REGRESSOR-API-CONFORM (fit/predict, 11 regressors) | SHIPPED | `py_regressor!` macro + hand-written `Rs*Regressor` expose fit/predict, wrapped by `_extras.py::_RegressorWrapper` (+ `n_features_in_`/`score`). Mirrors the sklearn regressor fit/predict contract across `ensemble`/`linear_model`/`neighbors`/`kernel_ridge`/`tree`. |
//! | REQ-CLASSIFIER-API-CONFORM (fit/predict + LabelEncoder, 13 classifiers) | SHIPPED | `py_classifier!` macro + hand-written classifiers expose fit/predict, wrapped by `_extras.py::_ClassifierWrapper` whose `_encode` (np.unique+searchsorted) sets `classes_` and round-trips arbitrary label dtypes through the Rust `usize`-label core. |
//! | REQ-CLUSTERER-API-CONFORM (fit + labels_, 5 clusterers incl. GMM) | SHIPPED | `RsMiniBatchKMeans`/`RsDBSCAN`/`RsAgglomerativeClustering`/`RsBirch` expose fit + `labels_` (+ predict for MiniBatchKMeans), `RsGaussianMixture` fit/predict; wrapped by `_extras.py::_ClusterWrapper` (+ `fit_predict`). |
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
//! | REQ-MISSING-METHODS (coef_/predict_proba/inverse_transform/cluster_centers_) | NOT-STARTED | the `Rs*` classes expose only fit/predict(/transform/labels_) — no `coef_`/`feature_importances_`, `predict_proba`/`decision_function`, `inverse_transform`/`components_`, `cluster_centers_`/`children_`. The binding cannot expose attrs the fitted library types do not compute — owned downstream by the eight crates. |
//! | REQ-MISSING-PARAMS (full constructor surface) | NOT-STARTED | each `Rs*` constructor binds a thin subset of sklearn's params (e.g. `RsRandomForestRegressor` lacks `criterion`/`max_features`/`bootstrap`/`oob_score`; `RsBaggingClassifier` lacks the `estimator` knob). Owned downstream by the eight crates. (`KNeighborsRegressor` is the exception: full surface SHIPPED, REQ-KNR-CTOR-SURFACE.) |
//! | REQ-KNR-CTOR-SURFACE (KNeighborsRegressor full constructor) | SHIPPED | FIXED #2147: hand-written `RsKNeighborsRegressor` (replacing the `n_neighbors`-only `py_regressor!` invocation) mirrors `RsKNeighborsClassifier` and exposes sklearn's full `(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)` surface (`neighbors/_regression.py:178-189`); `weights='distance'` changes predictions and matches the live sklearn 1.5.2 oracle. `fit(x, y: f64)`/`predict→PyArray1<f64>` keep the macro ABI. Validation is metric-aware (FIXED #2148): `metric='minkowski'` requires `p==2` (else `PyNotImplementedError #876`); `metric='euclidean'` accepts ANY `p` (sklearn ignores p for euclidean, `_base.py:526-538`); other metrics `PyNotImplementedError #876`; an invalid `weights` STRING raises `ValueError` (sklearn `InvalidParameterError ⊂ ValueError`, FIXED #2149); callable-weights/`metric_params` reject with `NotImplementedError #876`. Non-test consumer: `_extras.py::KNeighborsRegressor` full shim (`_make_rs`/`fit`/`predict` + `_RegressorPickleMixin`). Verification: `tests/divergence_extras.py::test_knr_*` (14 cases) + `tests/divergence_knr_ctor_surface.py` (#2148/#2149 pins). |
//! | REQ-VALUE-PARITY-RNG (seeded stochastic parity) | NOT-STARTED | the stochastic estimators (RF/ET/GB/HistGB/Bagging/AdaBoost/MiniBatchKMeans/GMM/FastICA/NMF) pass `random_state: Option<u64>` to a non-numpy RNG, so a shared `random_state` will not reproduce sklearn's numpy-MT/PCG64 draws (R-SUBSTRATE-5; needs `ferray::random`). |
//! | REQ-CONSUMER (binding IS the public API) | SHIPPED | non-test consumers: the `_extras.py` wrappers + `GaussianMixture` construct their `_Rs*` class via `_make_rs` and drive fit/predict/transform; `ferrolearn/__init__.py` re-exports all ~40; `lib.rs` registers every `_Rs*`; the head-to-head bench drives them vs sklearn (595 pytest pass). |
//! | REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | marshals via `crate::conversions::*` (rust-numpy + `ndarray`), not `ferray::numpy_interop`/`ferray-core` (R-SUBSTRATE-1); ferray exposes no numpy bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027. |

use crate::conversions::*;
use ferrolearn_core::{Fit, Predict, Transform};
use ndarray::Array1;
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

py_regressor!(
    RsBayesianRidge, "_RsBayesianRidge",
    ferrolearn_linear::FittedBayesianRidge<f64>,
    (max_iter: usize = 300, tol: f64 = 1e-3, fit_intercept: bool = true),
    {
        ferrolearn_linear::BayesianRidge::<f64>::new()
            .with_max_iter(max_iter).with_tol(tol).with_fit_intercept(fit_intercept)
    }
);

py_regressor!(
    RsARDRegression, "_RsARDRegression",
    ferrolearn_linear::FittedARDRegression<f64>,
    (max_iter: usize = 300, tol: f64 = 1e-3, fit_intercept: bool = true),
    {
        ferrolearn_linear::ARDRegression::<f64>::new()
            .with_max_iter(max_iter).with_tol(tol).with_fit_intercept(fit_intercept)
    }
);

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

py_regressor!(
    RsQuantileRegressor, "_RsQuantileRegressor",
    ferrolearn_linear::FittedQuantileRegressor<f64>,
    (quantile: f64 = 0.5, alpha: f64 = 1.0, max_iter: usize = 10000,
     tol: f64 = 1e-6, fit_intercept: bool = true),
    {
        ferrolearn_linear::QuantileRegressor::<f64>::new()
            .with_quantile(quantile).with_alpha(alpha).with_max_iter(max_iter)
            .with_tol(tol).with_fit_intercept(fit_intercept)
    }
);

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

#[pyclass(name = "_RsDBSCAN")]
pub struct RsDBSCAN {
    eps: f64,
    min_samples: usize,
    fitted: Option<ferrolearn_cluster::FittedDBSCAN<f64>>,
}

#[pymethods]
impl RsDBSCAN {
    #[new]
    #[pyo3(signature = (eps=0.5, min_samples=5))]
    fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let m = ferrolearn_cluster::DBSCAN::<f64>::new(self.eps).with_min_samples(self.min_samples);
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
