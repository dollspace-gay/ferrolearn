//! PyO3 bindings for regression models.
//!
//! ## REQ status
//!
//! The `sklearn.linear_model` `LinearRegression`/`Ridge`/`Lasso`/`ElasticNet`
//! binding shims: `#[pyclass]` `RsLinearRegression`/`RsRidge`/`RsLasso`/`RsElasticNet`
//! over `ferrolearn_linear::{LinearRegression,Ridge,Lasso,ElasticNet}`, wrapped by
//! the matching `ferrolearn.*` classes (`RegressorMixin`/`BaseEstimator`) in
//! `python/ferrolearn/_regressors.py`. This unit owns the sklearn-API marshalling
//! surface only (constructor ABI, attribute exposure, method surface, array
//! coercion); the regressor MATH lives downstream in `ferrolearn-linear`.
//! Verification model B: pytest comparing `import ferrolearn` against
//! `import sklearn` 1.5.2 (live oracle; `sklearn/linear_model/_base.py` +
//! `_ridge.py` + `_coordinate_descent.py`). Design doc:
//! `.design/python/regressors.md` (20 REQs). Every REQ is BINARY (R-DEFER-2):
//! SHIPPED or NOT-STARTED (with a concrete blocker). Verified via
//! `tests/divergence_regressors.py` + `tests/test_check_estimator.py` +
//! `tests/test_cross_val_score.py` (542 pytest pass).
//!
//! **13 SHIPPED / 7 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-LINREG-API-CONFORM (fit/predict + coef_/intercept_) | SHIPPED | `RsLinearRegression::fit`/`predict` + getters `coef_`/`intercept_`, wrapped by `_regressors.py::LinearRegression` (+ `n_features_in_`, `score` from `RegressorMixin`) — mirroring `_base.py:582`. Live default-path oracle matches element-wise. |
//! | REQ-LINREG-FIT-INTERCEPT-ABI (fit_intercept keyword-only) | SHIPPED | `LinearRegression.__init__(self, *, fit_intercept=True)` matches sklearn `_base.py:568` (the `*` is first). The only estimator whose constructor ABI already matched sklearn (Ridge/Lasso/ElasticNet diverged on `alpha`). |
//! | REQ-LINREG-VALUE-PARITY (coef_/intercept_ array parity) | SHIPPED | default path: marshalled from the Rust getters; downstream `ferrolearn-linear` REQ-1/5 match the sklearn oracle ≤1e-8 (full-rank/rank-deficient/underdetermined min-norm OLS). (`positive=True` NNLS #371; multi-output #372.) |
//! | REQ-LINREG-PARAMS (copy_X/n_jobs/positive) | NOT-STARTED | the wrapper exposes `fit_intercept` only; sklearn `_base.py:568-579`. Default full-rank OLS MATCHES, only the param surface is missing — downstream #371/#374. |
//! | REQ-RIDGE-API-CONFORM (fit/predict + coef_/intercept_, default cholesky) | SHIPPED | `RsRidge::fit`/`predict` + getters, wrapped by `_regressors.py::Ridge` (marshals `alpha` via `with_alpha`) — mirroring `_ridge.py:914`. Live default-path oracle matches element-wise. |
//! | REQ-RIDGE-ALPHA-POSITIONAL (alpha positional ABI) | SHIPPED | FIXED #2040: `_regressors.py::Ridge.__init__(self, alpha=1.0, *, fit_intercept=True)` moves `alpha` before the `*`, so `ferrolearn.Ridge(0.5).alpha == 0.5` matching sklearn `_ridge.py:893`. Guard `test_red_ridge_alpha_positional`. |
//! | REQ-RIDGE-VALUE-PARITY (coef_/intercept_ array parity, default cholesky) | SHIPPED | default path: downstream `ferrolearn-linear` REQ-1/5 match the sklearn `Ridge` oracle ≤1e-8 across alpha∈{0.1,1,10,100} (closed-form Cholesky, unpenalized intercept). (Per-target alpha #385, multi-output #384, alt solvers downstream.) |
//! | REQ-RIDGE-PARAMS (copy_X/max_iter/tol/solver/positive/random_state) | NOT-STARTED | the wrapper exposes `alpha`/`fit_intercept` only; sklearn `_ridge.py:893-912`. Default cholesky MATCHES, only the param surface is missing — downstream #386/#387/#388/#390. |
//! | REQ-LASSO-API-CONFORM (fit/predict + coef_/intercept_, default cyclic) | SHIPPED | `RsLasso::fit`/`predict` + getters, wrapped by `_regressors.py::Lasso` (marshals `alpha`/`max_iter`/`tol`) — mirroring `_coordinate_descent.py:932`. Live default-path coef matches the downstream-verified converged tolerance. |
//! | REQ-LASSO-ALPHA-POSITIONAL (alpha positional ABI) | SHIPPED | FIXED #2041: `_regressors.py::Lasso.__init__(self, alpha=1.0, *, ...)` moves `alpha` before the `*`, so `ferrolearn.Lasso(0.1).alpha == 0.1` matching sklearn `_coordinate_descent.py:1310`. Guard `test_red_lasso_alpha_positional`. |
//! | REQ-LASSO-VALUE-PARITY (coef_/intercept_ + support set, default cyclic) | SHIPPED | default converged path: downstream `ferrolearn-linear` REQ-1/4/6 match the sklearn `Lasso` oracle ≤1e-6 for converged coef/intercept + exact-zero support (cyclic CD, `l1_reg=α·n`). (Dual-gap stopping #412, positive/selection='random' downstream.) |
//! | REQ-LASSO-NITER (n_iter_ = real CD count) | NOT-STARTED | `_regressors.py::Lasso.fit` sets `n_iter_ = self.max_iter` (faked 1000) vs sklearn actual (`_coordinate_descent.py:1103`, e.g. 89); `_RsLasso` exposes no `n_iter_` getter, `FittedLasso` discards the count — downstream #411. |
//! | REQ-LASSO-PARAMS (precompute/copy_X/warm_start/positive/random_state/selection) | NOT-STARTED | the wrapper exposes `alpha`/`max_iter`/`tol`/`fit_intercept` only; sklearn `_coordinate_descent.py:1310-1322`. Default cyclic MATCHES — downstream #407/#408/#409/#410. |
//! | REQ-ELASTICNET-API-CONFORM (fit/predict + coef_/intercept_, default cyclic) | SHIPPED | `RsElasticNet::fit`/`predict` + getters, wrapped by `_regressors.py::ElasticNet` (marshals `alpha`/`l1_ratio`/`max_iter`/`tol`) — mirroring `_coordinate_descent.py:932`. Live default-path coef matches the downstream-verified converged tolerance. |
//! | REQ-ELASTICNET-ALPHA-POSITIONAL (alpha positional ABI) | SHIPPED | FIXED #2042: `_regressors.py::ElasticNet.__init__(self, alpha=1.0, *, l1_ratio=0.5, ...)` moves ONLY `alpha` before the `*` (l1_ratio stays keyword-only), so `ferrolearn.ElasticNet(0.1).alpha == 0.1` matching sklearn `_coordinate_descent.py:898`. Guard `test_red_elasticnet_alpha_positional`. |
//! | REQ-ELASTICNET-VALUE-PARITY (coef_/intercept_ + support set, default cyclic) | SHIPPED | default converged path: downstream `ferrolearn-linear` REQ-1/4/5 match the sklearn `ElasticNet` oracle <1e-5 over the (alpha,l1_ratio) grid (`l1_reg=α·l1_ratio·n`, `l2_reg=α·(1−l1_ratio)·n`), incl. l1_ratio=1↔Lasso / 0↔L2. (Dual-gap #412 downstream.) |
//! | REQ-ELASTICNET-NITER (n_iter_ = real CD count) | NOT-STARTED | `_regressors.py::ElasticNet.fit` sets `n_iter_ = self.max_iter` (faked 1000) vs sklearn actual (`_coordinate_descent.py:1103`, e.g. 58); `_RsElasticNet` exposes no `n_iter_` getter — downstream #417. |
//! | REQ-ELASTICNET-PARAMS (precompute/copy_X/warm_start/positive/random_state/selection) | NOT-STARTED | the wrapper exposes `alpha`/`l1_ratio`/`max_iter`/`tol`/`fit_intercept` only; sklearn `_coordinate_descent.py:898-912`. Default cyclic MATCHES — downstream #407/#408/#409/#410. |
//! | REQ-CONSUMER (binding IS the public API) | SHIPPED | non-test consumers: `_regressors.py::{LinearRegression,Ridge,Lasso,ElasticNet}` construct their `_Rs*` class and drive fit/predict + coef_/intercept_ reads; `ferrolearn/__init__.py` re-exports all four; `test_check_estimator.py` + `test_cross_val_score.py` exercise them (542 pytest pass). |
//! | REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | marshals via `crate::conversions::*` (rust-numpy + `ndarray`), not `ferray::numpy_interop`/`ferray-core` (R-SUBSTRATE-1); ferray exposes no numpy bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027. |

use crate::conversions::*;
use ferrolearn_core::{Fit, HasCoefficients, Predict};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// LinearRegression
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsLinearRegression")]
pub struct RsLinearRegression {
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedLinearRegression<f64>>,
}

#[pymethods]
impl RsLinearRegression {
    #[new]
    #[pyo3(signature = (fit_intercept=true))]
    fn new(fit_intercept: bool) -> Self {
        Self {
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::LinearRegression::<f64>::new()
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
// Ridge
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsRidge")]
pub struct RsRidge {
    alpha: f64,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedRidge<f64>>,
}

#[pymethods]
impl RsRidge {
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_intercept=true))]
    fn new(alpha: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::Ridge::<f64>::new()
            .with_alpha(self.alpha)
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
// Lasso
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsLasso")]
pub struct RsLasso {
    alpha: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedLasso<f64>>,
}

#[pymethods]
impl RsLasso {
    #[new]
    #[pyo3(signature = (alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=true))]
    fn new(alpha: f64, max_iter: usize, tol: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            max_iter,
            tol,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::Lasso::<f64>::new()
            .with_alpha(self.alpha)
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
// ElasticNet
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsElasticNet")]
pub struct RsElasticNet {
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedElasticNet<f64>>,
}

#[pymethods]
impl RsElasticNet {
    #[new]
    #[pyo3(signature = (alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=true))]
    fn new(alpha: f64, l1_ratio: f64, max_iter: usize, tol: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            l1_ratio,
            max_iter,
            tol,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::ElasticNet::<f64>::new()
            .with_alpha(self.alpha)
            .with_l1_ratio(self.l1_ratio)
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
