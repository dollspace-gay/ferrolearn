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
//! **12 SHIPPED / 5 NOT-STARTED.**
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
//! | REQ-MISSING-METHODS (coef_/predict_proba/inverse_transform/cluster_centers_) | NOT-STARTED | the `Rs*` classes expose only fit/predict(/transform/labels_) — no `coef_`/`feature_importances_`, `predict_proba`/`decision_function`, `inverse_transform`/`components_`, `cluster_centers_`/`children_`. The binding cannot expose attrs the fitted library types do not compute — owned downstream by the eight crates. |
//! | REQ-MISSING-PARAMS (full constructor surface) | NOT-STARTED | each `Rs*` constructor binds a thin subset of sklearn's params (e.g. `RsRandomForestRegressor` lacks `criterion`/`max_features`/`bootstrap`/`oob_score`; `RsBaggingClassifier` lacks the `estimator` knob). Owned downstream by the eight crates. |
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

py_regressor!(
    RsHuberRegressor, "_RsHuberRegressor",
    ferrolearn_linear::FittedHuberRegressor<f64>,
    (epsilon: f64 = 1.35, alpha: f64 = 0.0001, max_iter: usize = 100,
     tol: f64 = 1e-5, fit_intercept: bool = true),
    {
        ferrolearn_linear::HuberRegressor::<f64>::new()
            .with_epsilon(epsilon).with_alpha(alpha).with_max_iter(max_iter)
            .with_tol(tol).with_fit_intercept(fit_intercept)
    }
);

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

py_regressor!(
    RsKNeighborsRegressor, "_RsKNeighborsRegressor",
    ferrolearn_neighbors::FittedKNeighborsRegressor<f64>,
    (n_neighbors: usize = 5),
    {
        ferrolearn_neighbors::KNeighborsRegressor::<f64>::new()
            .with_n_neighbors(n_neighbors)
    }
);

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

py_classifier!(
    RsBernoulliNB, "_RsBernoulliNB",
    ferrolearn_bayes::FittedBernoulliNB<f64>,
    (alpha: f64 = 1.0, fit_prior: bool = true, binarize: f64 = 0.0),
    {
        ferrolearn_bayes::BernoulliNB::<f64>::new()
            .with_alpha(alpha).with_fit_prior(fit_prior).with_binarize(binarize)
    }
);

py_classifier!(
    RsComplementNB, "_RsComplementNB",
    ferrolearn_bayes::FittedComplementNB<f64>,
    (alpha: f64 = 1.0, fit_prior: bool = true, norm: bool = false),
    {
        ferrolearn_bayes::ComplementNB::<f64>::new()
            .with_alpha(alpha).with_fit_prior(fit_prior).with_norm(norm)
    }
);

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
