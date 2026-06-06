//! PyO3 bindings for classification models.
//!
//! ## REQ status
//!
//! Five `sklearn` classifier binding shims: `#[pyclass]`
//! `RsLogisticRegression`/`RsDecisionTreeClassifier`/`RsRandomForestClassifier`/
//! `RsKNeighborsClassifier`/`RsGaussianNB` over
//! `ferrolearn_linear`/`ferrolearn_tree`/`ferrolearn_neighbors`/`ferrolearn_bayes`,
//! wrapped by the matching `ferrolearn.*` classes (`ClassifierMixin`/`BaseEstimator`,
//! with a LabelEncoder-equivalent `_encode_labels`/`_decode_labels`) in
//! `python/ferrolearn/_classifiers.py`. This unit owns the sklearn-API marshalling
//! surface only (constructor ABI, attribute exposure, method surface, array
//! coercion); the classifier MATH lives downstream in the respective crates.
//! Verification model B: pytest comparing `import ferrolearn` against
//! `import sklearn` 1.5.2 (live oracle; `sklearn/linear_model/_logistic.py`,
//! `tree/_classes.py`, `ensemble/_forest.py`, `neighbors/_classification.py`,
//! `naive_bayes.py`). Design doc: `.design/python/classifiers.md` (28 REQs).
//! Every REQ is BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete
//! blocker). Verified via `tests/divergence_classifiers.py` +
//! `tests/test_check_estimator.py` (553 pytest pass).
//!
//! **21 SHIPPED / 7 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-LOGREG-API-CONFORM (fit/predict/predict_proba + classes_/coef_/intercept_) | SHIPPED | `RsLogisticRegression::*` + getters, wrapped by `_classifiers.py::LogisticRegression` (coef_ reshaped `(1,-1)`, intercept_ `(1,)`, classes_ via `_encode_labels`, + `n_features_in_`/`score`) — mirroring `_logistic.py:810`/`:1395`. Live default-path oracle matches. |
//! | REQ-LOGREG-C-ABI (C keyword-only) | SHIPPED | `LogisticRegression.__init__(self, *, C=1.0, ...)` matches sklearn `_logistic.py:1129` (`C` after the `*`). |
//! | REQ-LOGREG-MAXITER-DEFAULT (max_iter default 100) | SHIPPED | FIXED #2049: wrapper default `max_iter=100` matching sklearn `_logistic.py:1129` (was 1000); critic verified the LBFGS solver still converges + predicts accuracy 1.0 at 100 iters. Guard `test_red_logreg_max_iter_default`. |
//! | REQ-LOGREG-VALUE-PARITY (coef_/intercept_/predict_proba parity, default path) | SHIPPED | default path: downstream `ferrolearn-linear` REQ-1/2/4 match the sklearn `LogisticRegression` oracle ~1e-8 at convergence (binary LBFGS + multinomial softmax). (decision_function shape #454.) |
//! | REQ-LOGREG-NITER (n_iter_ = real LBFGS count) | NOT-STARTED | `fit` sets `n_iter_ = self.max_iter` (faked) vs sklearn actual (`_logistic.py:1276`, e.g. `[11]`); `_RsLogisticRegression` exposes no `n_iter_` getter — downstream #450. |
//! | REQ-LOGREG-PARAMS (penalty/dual/solver/class_weight/random_state/intercept_scaling/warm_start/l1_ratio/n_jobs) | NOT-STARTED | the wrapper exposes `C`/`max_iter`/`tol`/`fit_intercept` only; sklearn `_logistic.py:1129`. Default l2/lbfgs MATCHES — downstream #442-#452. |
//! | REQ-DT-API-CONFORM (fit/predict/predict_proba + classes_, default gini) | SHIPPED | `RsDecisionTreeClassifier::*` + getter, wrapped by `_classifiers.py::DecisionTreeClassifier` — mirroring `tree/_classes.py:698`/`:1017`. Live default-path oracle matches. |
//! | REQ-DT-CTOR-ABI (all params keyword-only) | SHIPPED | `DecisionTreeClassifier.__init__(self, *, max_depth=None, ...)` matches sklearn `_classes.py:945` (the `*` is first). |
//! | REQ-DT-FEATURE-IMPORTANCES (feature_importances_ surfaced) | SHIPPED | FIXED #2047: the wrapper `fit` now reads the pre-existing Rust getter `self.feature_importances_ = np.array(self._rs.feature_importances_)` — `(n_features,)` summing to 1.0 matching sklearn. Guard `test_red_dt_feature_importances_exposed`. |
//! | REQ-DT-VALUE-PARITY (pred/predict_proba parity, default gini) | SHIPPED | default path: downstream `ferrolearn-tree` REQ-1/2/6 match the sklearn `DecisionTreeClassifier` oracle (CART best-split + per-leaf frequencies). (Multi-output #668.) |
//! | REQ-DT-PARAMS (criterion/splitter/max_features/random_state/max_leaf_nodes/min_impurity_decrease/class_weight/ccp_alpha) | NOT-STARTED | the wrapper exposes `max_depth`/`min_samples_split`/`min_samples_leaf` only; sklearn `_classes.py:945`. Default gini MATCHES — downstream #665/#670. |
//! | REQ-RF-API-CONFORM (fit/predict + classes_, default soft-vote) | SHIPPED | `RsRandomForestClassifier::*` + getter, wrapped by `_classifiers.py::RandomForestClassifier` — mirroring `ensemble/_forest.py:1170`/`:883`. Live default-path oracle matches. |
//! | REQ-RF-NESTIMATORS-POSITIONAL (n_estimators positional ABI) | SHIPPED | FIXED #2045: `_classifiers.py::RandomForestClassifier.__init__(self, n_estimators=100, *, ...)` moves `n_estimators` before the `*`, so `ferrolearn.RandomForestClassifier(10).n_estimators == 10` matching sklearn `_forest.py:1494`. Guard `test_red_rf_n_estimators_positional`. |
//! | REQ-RF-FEATURE-IMPORTANCES (feature_importances_ surfaced) | SHIPPED | FIXED #2048: the wrapper `fit` now reads the pre-existing Rust getter (over `FittedRandomForestClassifier::feature_importances`) — `(n_features,)` summing to 1.0. Guard `test_red_rf_feature_importances_exposed`. |
//! | REQ-RF-VALUE-PARITY (pred/predict_proba parity at random_state) | SHIPPED | deterministic contract: downstream `ferrolearn-tree` REQ-4/5/9 verify soft-vote argmax + proba-mean + seeded reproducibility. (Exact numpy-MT bootstrap parity is the documented RNG boundary #673.) |
//! | REQ-RF-PREDICT-PROBA (predict_proba surfaced) | SHIPPED | FIXED #2050: `RsRandomForestClassifier::predict_proba` binds the existing `FittedRandomForestClassifier::predict_proba` (soft-vote mean of per-tree class probabilities, `random_forest.rs:432`), surfaced by `_classifiers.py::RandomForestClassifier.predict_proba` — `(n_samples, n_classes)` rows summing to 1.0, with `predict == classes_[argmax(predict_proba)]` matching sklearn `_forest.py:922`/`:883`. Guard `test_red_rf_predict_proba_surfaced`. |
//! | REQ-RF-PARAMS (criterion/max_features/bootstrap/oob_score/max_samples/class_weight/n_jobs/warm_start/ccp_alpha/max_leaf_nodes/min_impurity_decrease) | NOT-STARTED | the wrapper exposes `n_estimators`/`max_depth`/`min_samples_split`/`min_samples_leaf`/`random_state` only; sklearn `_forest.py:1494`. Default sqrt/gini/bootstrap MATCHES — downstream #671/#672/#675/#676. |
//! | REQ-KNN-API-CONFORM (fit/predict + classes_, default uniform) | SHIPPED | `RsKNeighborsClassifier::*` + getter, wrapped by `_classifiers.py::KNeighborsClassifier` — mirroring `neighbors/_classification.py:39`/`:240`. |
//! | REQ-KNN-NNEIGHBORS-POSITIONAL (n_neighbors positional ABI) | SHIPPED | FIXED #2046: `_classifiers.py::KNeighborsClassifier.__init__(self, n_neighbors=5)` drops the keyword-only `*`, so `ferrolearn.KNeighborsClassifier(3).n_neighbors == 3` matching sklearn `_classification.py:193`. Guard `test_red_knn_n_neighbors_positional`. |
//! | REQ-KNN-VALUE-PARITY (pred parity incl. tie-break) | SHIPPED | default uniform path: downstream `ferrolearn-neighbors` REQ-1 verifies the uniform weighted-vote smallest-label tie-break. (Distance-weighting/2-D-query/non-Euclidean divergences downstream #876.) |
//! | REQ-KNN-PREDICT-PROBA (predict_proba surfaced) | SHIPPED | FIXED #2051: `RsKNeighborsClassifier::predict_proba` binds the existing `FittedKNeighborsClassifier::predict_proba` (normalized weighted class-vote shares, `knn.rs:487`), surfaced by `_classifiers.py::KNeighborsClassifier.predict_proba` — `(n_samples, n_classes)` rows summing to 1.0, with `predict == classes_[argmax(predict_proba)]` matching sklearn `_classification.py:307`. Guard `test_red_knn_predict_proba_surfaced`. |
//! | REQ-KNN-PARAMS (weights/algorithm/leaf_size/p/metric/metric_params/n_jobs) | NOT-STARTED | the wrapper exposes `n_neighbors` only; sklearn `_classification.py:193`. Default uniform/minkowski/p=2 MATCHES — downstream #876/#877. |
//! | REQ-GNB-API-CONFORM (fit/predict/predict_proba + classes_, default path) | SHIPPED | `RsGaussianNB::*` + getter, wrapped by `_classifiers.py::GaussianNB` — mirroring `naive_bayes.py:147`/`:128`. Live default-path oracle matches element-wise. |
//! | REQ-GNB-CTOR-ABI (all params keyword-only) | SHIPPED | `GaussianNB.__init__(self, *, var_smoothing=1e-9)` matches sklearn `naive_bayes.py:234` (the `*` is first). |
//! | REQ-GNB-VALUE-PARITY (pred/predict_proba parity, default path) | SHIPPED | default path: downstream `ferrolearn-bayes` REQ-1/3/4 match the sklearn `GaussianNB` oracle ~1e-9 (`epsilon_` global var, `_joint_log_likelihood`, predict_proba). |
//! | REQ-GNB-PARAMS (priors) | NOT-STARTED | the wrapper + `RsGaussianNB::new` expose `var_smoothing` only; sklearn `naive_bayes.py:234` has `priors`. Default data-derived priors MATCH — downstream #897. |
//! | REQ-CONSUMER (binding IS the public API) | SHIPPED | non-test consumers: the five `_classifiers.py` wrappers construct their `_Rs*` class and drive fit/predict(/predict_proba) + getter reads; `ferrolearn/__init__.py` re-exports all five; `test_check_estimator.py` runs each through `parametrize_with_checks` (553 pytest pass). Label round-trip covered by `conversions.md` REQ-LABEL-MARSHAL. |
//! | REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | marshals via `crate::conversions::*` (rust-numpy + `ndarray`), not `ferray::numpy_interop`/`ferray-core` (R-SUBSTRATE-1); ferray exposes no numpy bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027. |

use crate::conversions::*;
use ferrolearn_core::{Fit, HasClasses, HasCoefficients, HasFeatureImportances, Predict};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// LogisticRegression
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsLogisticRegression")]
pub struct RsLogisticRegression {
    c: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedLogisticRegression<f64>>,
}

#[pymethods]
impl RsLogisticRegression {
    #[new]
    #[pyo3(signature = (c=1.0, max_iter=1000, tol=1e-4, fit_intercept=true))]
    fn new(c: f64, max_iter: usize, tol: f64, fit_intercept: bool) -> Self {
        Self {
            c,
            max_iter,
            tol,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let model = ferrolearn_linear::LogisticRegression::<f64>::new()
            .with_c(self.c)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
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
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let proba = fitted
            .predict_proba(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &proba))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let classes = fitted.classes();
        let arr = ndarray::Array1::from_vec(classes.iter().map(|&c| c as i64).collect());
        Ok(PyArray1::from_array(py, &arr))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }
}

// ---------------------------------------------------------------------------
// DecisionTreeClassifier
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsDecisionTreeClassifier")]
pub struct RsDecisionTreeClassifier {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    fitted: Option<ferrolearn_tree::FittedDecisionTreeClassifier<f64>>,
}

#[pymethods]
impl RsDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (max_depth=None, min_samples_split=2, min_samples_leaf=1))]
    fn new(max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize) -> Self {
        Self {
            max_depth,
            min_samples_split,
            min_samples_leaf,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let model = ferrolearn_tree::DecisionTreeClassifier::<f64>::new()
            .with_max_depth(self.max_depth)
            .with_min_samples_split(self.min_samples_split)
            .with_min_samples_leaf(self.min_samples_leaf);
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
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let proba = fitted
            .predict_proba(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &proba))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let classes = fitted.classes();
        let arr = ndarray::Array1::from_vec(classes.iter().map(|&c| c as i64).collect());
        Ok(PyArray1::from_array(py, &arr))
    }

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.feature_importances()))
    }
}

// ---------------------------------------------------------------------------
// RandomForestClassifier
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsRandomForestClassifier")]
pub struct RsRandomForestClassifier {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedRandomForestClassifier<f64>>,
}

#[pymethods]
impl RsRandomForestClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None))]
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

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut model = ferrolearn_tree::RandomForestClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_min_samples_split(self.min_samples_split)
            .with_min_samples_leaf(self.min_samples_leaf);
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
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let proba = fitted
            .predict_proba(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &proba))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let classes = fitted.classes();
        let arr = ndarray::Array1::from_vec(classes.iter().map(|&c| c as i64).collect());
        Ok(PyArray1::from_array(py, &arr))
    }

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.feature_importances()))
    }
}

// ---------------------------------------------------------------------------
// KNeighborsClassifier
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsKNeighborsClassifier")]
pub struct RsKNeighborsClassifier {
    n_neighbors: usize,
    fitted: Option<ferrolearn_neighbors::FittedKNeighborsClassifier<f64>>,
}

#[pymethods]
impl RsKNeighborsClassifier {
    #[new]
    #[pyo3(signature = (n_neighbors=5))]
    fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let model = ferrolearn_neighbors::KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(self.n_neighbors);
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
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let proba = fitted
            .predict_proba(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &proba))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let classes = fitted.classes();
        let arr = ndarray::Array1::from_vec(classes.iter().map(|&c| c as i64).collect());
        Ok(PyArray1::from_array(py, &arr))
    }
}

// ---------------------------------------------------------------------------
// GaussianNB
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsGaussianNB")]
pub struct RsGaussianNB {
    var_smoothing: f64,
    fitted: Option<ferrolearn_bayes::FittedGaussianNB<f64>>,
}

#[pymethods]
impl RsGaussianNB {
    #[new]
    #[pyo3(signature = (var_smoothing=1e-9))]
    fn new(var_smoothing: f64) -> Self {
        Self {
            var_smoothing,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let model =
            ferrolearn_bayes::GaussianNB::<f64>::new().with_var_smoothing(self.var_smoothing);
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
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let proba = fitted
            .predict_proba(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &proba))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let classes = fitted.classes();
        let arr = ndarray::Array1::from_vec(classes.iter().map(|&c| c as i64).collect());
        Ok(PyArray1::from_array(py, &arr))
    }
}
