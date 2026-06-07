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
//! **24 SHIPPED / 5 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-LOGREG-API-CONFORM (fit/predict/predict_proba + classes_/coef_/intercept_) | SHIPPED | `RsLogisticRegression::*` + getters, wrapped by `_classifiers.py::LogisticRegression` (coef_ reshaped `(1,-1)`, intercept_ `(1,)`, classes_ via `_encode_labels`, + `n_features_in_`/`score`) — mirroring `_logistic.py:810`/`:1395`. Live default-path oracle matches. |
//! | REQ-LOGREG-C-ABI (C keyword-only) | SHIPPED | `LogisticRegression.__init__(self, *, C=1.0, ...)` matches sklearn `_logistic.py:1129` (`C` after the `*`). |
//! | REQ-LOGREG-MAXITER-DEFAULT (max_iter default 100) | SHIPPED | FIXED #2049: wrapper default `max_iter=100` matching sklearn `_logistic.py:1129` (was 1000); critic verified the LBFGS solver still converges + predicts accuracy 1.0 at 100 iters. Guard `test_red_logreg_max_iter_default`. |
//! | REQ-LOGREG-VALUE-PARITY (coef_/intercept_/predict_proba parity, default path) | SHIPPED | default path: downstream `ferrolearn-linear` REQ-1/2/4 match the sklearn `LogisticRegression` oracle ~1e-8 at convergence (binary LBFGS + multinomial softmax). (decision_function shape #454.) |
//! | REQ-LOGREG-NITER (n_iter_ = real LBFGS count) | SHIPPED | FIXED #450/#2169: `RsLogisticRegression::n_iter_` getter binds the real `FittedLogisticRegression::n_iter` (from `LbfgsOptimizer::minimize_reporting`); `_classifiers.py::LogisticRegression.fit` sets `n_iter_ = np.array([n], dtype=int32)` (shape `(1,)`, matching sklearn `_logistic.py:1376`). R-DEV-7: honest count (positive, `<= max_iter`, deterministic), not the literal scipy value. Guard `test_logreg_n_iter_contract`. |
//! | REQ-LOGREG-PARAMS-WEIGHTING (class_weight/random_state/n_jobs + sample_weight) | SHIPPED | FIXED #445/#451/#452/#2169: ctor gains `class_weight` (`None`/`'balanced'`/dict, re-keyed onto encoded labels), `random_state`, `n_jobs`; `fit(X,y,sample_weight=None)` threads per-sample weights. `class_weight`/`sample_weight` reach the Rust core's `with_class_weight`/`fit_with_sample_weight`; `random_state`/`n_jobs` are stored no-ops (deterministic lbfgs). sklearn `_logistic.py:1129`. Guards `test_logreg_sample_weight_*`/`test_logreg_class_weight_*`/`test_logreg_random_state_n_jobs_noop_and_get_params` (live oracle). NOTE: multiclass `coef_` exposure still collapses to `(1,n_features)` (separate #2170). |
//! | REQ-LOGREG-PARAMS-REMAINING (penalty/dual/solver/intercept_scaling/warm_start/l1_ratio) | NOT-STARTED | the wrapper still exposes no `penalty`/`dual`/`solver`/`intercept_scaling`/`warm_start`/`l1_ratio`; sklearn `_logistic.py:1129`. Default l2/lbfgs MATCHES — downstream #442-#444/#446-#449. |
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
//! | REQ-KNN-PARAMS (weights/algorithm/leaf_size/p/metric/metric_params/n_jobs) | SHIPPED (surfaced subset) | FIXED #2138: `RsKNeighborsClassifier` gains `weights`/`algorithm`/`leaf_size`/`p`/`metric`/`n_jobs` (`#[pyo3(signature = (n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, p=2.0, metric="minkowski", n_jobs=None))]`), surfaced by `_classifiers.py::KNeighborsClassifier.__init__` matching sklearn `_classification.py:193` (n_neighbors positional, rest keyword-only). `fit` maps `weights`→`Weights::{Uniform,Distance}`, `algorithm`→`Algorithm::{Auto,BruteForce,KdTree}` (`ball_tree`→`Auto`, identical result); `leaf_size`/`n_jobs` are ABI no-ops (perf/threading only). The Euclidean-only restriction is validated metric-aware (FIXED #2148): `metric='minkowski'` requires `p==2` (else `NotImplementedError #876`); `metric='euclidean'` accepts ANY `p` (sklearn consumes `p` only for minkowski, `_base.py:526-538`); other metrics raise `NotImplementedError #876`. An invalid `weights` STRING raises `ValueError` (matching sklearn `InvalidParameterError ⊂ ValueError`, FIXED #2149); callable weights raise `NotImplementedError #876`; `metric_params` is wrapper-validated `is None`. Guards `test_knn_weights_distance_*`/`test_knn_algorithm_same_result`/`test_knn_get_params_clone`/`test_knn_unsupported_*`/`test_knn_euclidean_metric_ignores_p`/`test_knn_invalid_weights_string_is_valueerror`. NOT-STARTED: callable-weights / Minkowski-p≠2 / custom-metric (#876). |
//! | REQ-GNB-API-CONFORM (fit/predict/predict_proba + classes_, default path) | SHIPPED | `RsGaussianNB::*` + getter, wrapped by `_classifiers.py::GaussianNB` — mirroring `naive_bayes.py:147`/`:128`. Live default-path oracle matches element-wise. |
//! | REQ-GNB-CTOR-ABI (all params keyword-only) | SHIPPED | `GaussianNB.__init__(self, *, var_smoothing=1e-9)` matches sklearn `naive_bayes.py:234` (the `*` is first). |
//! | REQ-GNB-FITTED-ATTRS (theta_/var_/class_prior_/class_count_/epsilon_ surfaced) | SHIPPED | FIXED #2102: `RsGaussianNB::{theta_,var_,class_prior_,class_count_,epsilon_}` getters bind the pre-existing `FittedGaussianNB` accessors (`gaussian.rs:549`/:557`/:565`/:572`/:584`); consumer `_classifiers.py::GaussianNB.fit` assigns all five after `classes_`, mirroring sklearn `naive_bayes.py` `theta_`/`var_`/`class_prior_`/`class_count_`/`epsilon_`. Guard `test_gaussiannb_fitted_attrs_match_sklearn` (live oracle, R-CHAR-3). |
//! | REQ-GNB-VALUE-PARITY (pred/predict_proba parity, default path) | SHIPPED | default path: downstream `ferrolearn-bayes` REQ-1/3/4 match the sklearn `GaussianNB` oracle ~1e-9 (`epsilon_` global var, `_joint_log_likelihood`, predict_proba). |
//! | REQ-GNB-PARAMS (priors) | NOT-STARTED | the wrapper + `RsGaussianNB::new` expose `var_smoothing` only; sklearn `naive_bayes.py:234` has `priors`. Default data-derived priors MATCH — downstream #897. |
//! | REQ-CONSUMER (binding IS the public API) | SHIPPED | non-test consumers: the five `_classifiers.py` wrappers construct their `_Rs*` class and drive fit/predict(/predict_proba) + getter reads; `ferrolearn/__init__.py` re-exports all five; `test_check_estimator.py` runs each through `parametrize_with_checks` (553 pytest pass). Label round-trip covered by `conversions.md` REQ-LABEL-MARSHAL. |
//! | REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | marshals via `crate::conversions::*` (rust-numpy + `ndarray`), not `ferray::numpy_interop`/`ferray-core` (R-SUBSTRATE-1); ferray exposes no numpy bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027. |

use crate::conversions::*;
use ferrolearn_core::{Fit, HasClasses, HasFeatureImportances, Predict};
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
    // `class_weight` is normalized by the Python wrapper into either the string
    // "balanced" or a list of (class_label_as_i64, weight) pairs; `None` ==
    // uniform. Stored so it threads into the Rust core's `with_class_weight`.
    class_weight_mode: Option<ClassWeightArg>,
    // `random_state`/`n_jobs` are stored for sklearn `get_params`/`clone` parity
    // only — they are documented no-ops on the deterministic lbfgs path
    // (sklearn `_logistic.py:1112`/`:1121`; mirrored downstream by
    // `LogisticRegression::with_random_state`/`with_n_jobs`).
    #[allow(
        dead_code,
        reason = "ABI/get_params-parity no-op: random_state does not affect the deterministic lbfgs solver (sklearn _logistic.py:1112)"
    )]
    random_state: Option<u64>,
    #[allow(
        dead_code,
        reason = "ABI/get_params-parity no-op: n_jobs is a threading knob, not a result-affecting param (sklearn _logistic.py:1121)"
    )]
    n_jobs: Option<i64>,
    fitted: Option<ferrolearn_linear::FittedLogisticRegression<f64>>,
}

/// Normalized `class_weight` argument crossing the PyO3 boundary.
#[derive(Clone)]
enum ClassWeightArg {
    /// sklearn `class_weight='balanced'`.
    Balanced,
    /// sklearn `class_weight={label: weight}`, as `(label_i64, weight)` pairs.
    Dict(Vec<(i64, f64)>),
}

#[pymethods]
impl RsLogisticRegression {
    #[new]
    #[pyo3(signature = (c=1.0, max_iter=1000, tol=1e-4, fit_intercept=true, class_weight=None, random_state=None, n_jobs=None))]
    fn new(
        c: f64,
        max_iter: usize,
        tol: f64,
        fit_intercept: bool,
        class_weight: Option<Bound<'_, PyAny>>,
        random_state: Option<u64>,
        n_jobs: Option<i64>,
    ) -> PyResult<Self> {
        let class_weight_mode = parse_class_weight(class_weight.as_ref())?;
        Ok(Self {
            c,
            max_iter,
            tol,
            fit_intercept,
            class_weight_mode,
            random_state,
            n_jobs,
            fitted: None,
        })
    }

    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, i64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let sw_nd = sample_weight.map(numpy1_to_ndarray);
        let mut model = ferrolearn_linear::LogisticRegression::<f64>::new()
            .with_c(self.c)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_fit_intercept(self.fit_intercept);
        if let Some(seed) = self.random_state {
            model = model.with_random_state(seed);
        }
        if let Some(jobs) = self.n_jobs {
            model = model.with_n_jobs(jobs);
        }
        if let Some(cw) = &self.class_weight_mode {
            // The Rust core keys `class_weight` by the encoded usize label
            // (the Python wrapper passes already-encoded contiguous labels
            // 0..n_classes via `_encode_labels`, so the i64 labels are >= 0).
            let core_cw = match cw {
                ClassWeightArg::Balanced => {
                    ferrolearn_linear::logistic_regression::ClassWeight::Balanced
                }
                ClassWeightArg::Dict(pairs) => {
                    let mapped: Vec<(usize, f64)> = pairs
                        .iter()
                        .map(|&(lbl, w)| (lbl.max(0) as usize, w))
                        .collect();
                    ferrolearn_linear::logistic_regression::ClassWeight::Dict(mapped)
                }
            };
            model = model.with_class_weight(core_cw);
        }
        let fitted = model
            .fit_with_sample_weight(&x_nd, &y_nd, sw_nd.as_ref())
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

    /// Coefficient matrix (sklearn `coef_`, `_logistic.py`): shape
    /// `(1, n_features)` for binary and `(n_classes, n_features)` for multiclass.
    /// Binds the core's full `weight_matrix()` (which already shapes the binary
    /// weight vector to `(1, n_features)`), NOT the flat first-class
    /// `coefficients()` — fixing the multiclass `(1, n_features)` collapse (#2170).
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.weight_matrix()))
    }

    /// Intercept vector (sklearn `intercept_`, `_logistic.py`): shape `(1,)` for
    /// binary and `(n_classes,)` for multiclass. Binds the core's per-class
    /// `intercept_vec()`, NOT the scalar first-class `intercept()` (#2170).
    #[getter]
    fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.intercept_vec()))
    }

    /// Number of L-BFGS iterations performed during `fit` (sklearn `n_iter_`,
    /// `_logistic.py:1376`). Exposed as a scalar; the Python wrapper presents it
    /// as `np.array([n_iter])` to match sklearn's `(1,)` shape on the
    /// binary/multinomial lbfgs path. R-DEV-7: the count is honest (positive,
    /// `<= max_iter`, deterministic), not asserted equal to scipy's.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_iter())
    }
}

/// Parse a Python `class_weight` argument (`None` / `'balanced'` / `dict`) into
/// the normalized [`ClassWeightArg`]. Mirrors sklearn's accepted values
/// (`_logistic.py:1111` `[dict, StrOptions({"balanced"}), None]`).
fn parse_class_weight(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<ClassWeightArg>> {
    let Some(obj) = obj else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    // String form: only 'balanced' is valid.
    if let Ok(s) = obj.extract::<String>() {
        if s == "balanced" {
            return Ok(Some(ClassWeightArg::Balanced));
        }
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "class_weight={s:?} is invalid (expected 'balanced', a dict, or None)"
        )));
    }
    // Dict form: {label: weight}. Labels are the already-encoded i64 class
    // indices passed by the Python wrapper.
    if let Ok(dict) = obj.cast::<pyo3::types::PyDict>() {
        let mut pairs: Vec<(i64, f64)> = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let label: i64 = k.extract().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "class_weight dict keys must be integer class labels",
                )
            })?;
            let weight: f64 = v.extract().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "class_weight dict values must be numeric weights",
                )
            })?;
            pairs.push((label, weight));
        }
        return Ok(Some(ClassWeightArg::Dict(pairs)));
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "class_weight must be None, 'balanced', or a {class: weight} dict",
    ))
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
    weights: String,
    algorithm: String,
    // `leaf_size`/`n_jobs` are stored for ABI parity with sklearn
    // (`neighbors/_classification.py:193`) but are search-perf / threading
    // knobs only — they do NOT change the predicted result, so they are
    // accepted and held without affecting the fitted model.
    #[allow(
        dead_code,
        reason = "ABI-parity no-op: leaf_size affects only tree-build perf, not the result (sklearn neighbors/_classification.py:193)"
    )]
    leaf_size: usize,
    p: f64,
    metric: String,
    #[allow(
        dead_code,
        reason = "ABI-parity no-op: n_jobs is a threading knob, not a result-affecting param (sklearn neighbors/_classification.py:193)"
    )]
    n_jobs: Option<i64>,
    fitted: Option<ferrolearn_neighbors::FittedKNeighborsClassifier<f64>>,
}

#[pymethods]
impl RsKNeighborsClassifier {
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

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        use ferrolearn_neighbors::{Algorithm, Weights};

        // weights: only 'uniform'/'distance' are supported in the Rust core
        // (`knn.rs::Weights`); callable weights are NOT-STARTED (#876).
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
        // predict/predict_proba (search strategy only). The Rust core has no
        // exact 'ball_tree'-distinct classifier variant, so it maps to Auto
        // (identical result).
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
        let y_nd = numpy1_to_ndarray_usize(y);
        let model = ferrolearn_neighbors::KNeighborsClassifier::<f64>::new()
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

    /// Per-class feature means (sklearn `theta_`, `naive_bayes.py:171`;
    /// `ferrolearn_bayes::FittedGaussianNB::theta` at `gaussian.rs:549`),
    /// shape `(n_classes, n_features)`.
    #[getter]
    fn theta_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.theta()))
    }

    /// Per-class epsilon-smoothed feature variances (sklearn `var_`,
    /// `naive_bayes.py:202`; `FittedGaussianNB::var` at `gaussian.rs:557`),
    /// shape `(n_classes, n_features)`.
    #[getter]
    fn var_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.var()))
    }

    /// Empirical class priors `class_count_ / class_count_.sum()` (sklearn
    /// `class_prior_`, `naive_bayes.py:176`; `FittedGaussianNB::class_prior`
    /// at `gaussian.rs:584` — returns an owned `Array1`), shape `(n_classes,)`.
    #[getter]
    fn class_prior_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let cp = fitted.class_prior();
        Ok(ndarray1_to_numpy(py, &cp))
    }

    /// Number of training samples seen per class (sklearn `class_count_`,
    /// `naive_bayes.py:173`; `FittedGaussianNB::class_count` at `gaussian.rs:572`
    /// — returns an owned `Array1`), shape `(n_classes,)`.
    #[getter]
    fn class_count_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let cc = fitted.class_count();
        Ok(ndarray1_to_numpy(py, &cc))
    }

    /// Absolute additive variance smoothing `var_smoothing * max(var(X))`
    /// (sklearn `epsilon_`, `naive_bayes.py:431`; `FittedGaussianNB::epsilon`
    /// at `gaussian.rs:565`).
    #[getter]
    fn epsilon_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.epsilon())
    }
}
