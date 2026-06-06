//! PyO3 bindings for transformer models.
//!
//! ## REQ status
//!
//! The `sklearn.preprocessing.StandardScaler` + `sklearn.decomposition.PCA`
//! binding shims: `#[pyclass] RsStandardScaler` over
//! `ferrolearn_preprocess::StandardScaler` and `#[pyclass] RsPCA` over
//! `ferrolearn_decomp::PCA`, wrapped by `ferrolearn.StandardScaler` / `ferrolearn.PCA`
//! (`TransformerMixin`/`BaseEstimator`) in `python/ferrolearn/_transformers.py`.
//! This unit owns the sklearn-API marshalling surface only (constructor ABI,
//! attribute exposure, method surface, array coercion); the transformer MATH
//! lives downstream in `ferrolearn-preprocess/src/standard_scaler.rs` and
//! `ferrolearn-decomp/src/pca.rs`. Verification model B: pytest comparing
//! `import ferrolearn` against `import sklearn` 1.5.2 (live oracle;
//! `sklearn/preprocessing/_data.py` + `sklearn/decomposition/_pca.py`). Design
//! doc: `.design/python/transformers.md` (12 REQs). Every REQ is BINARY
//! (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker). Verified via
//! `tests/divergence_transformers.py` + `tests/test_check_estimator.py`
//! (534 pytest pass).
//!
//! **9 SHIPPED / 3 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-SS-API-CONFORM (fit/transform/inverse_transform + attrs, default path) | SHIPPED | `RsStandardScaler::fit`/`transform`/`inverse_transform` + getters `mean_`/`scale_`, wrapped by `_transformers.py::StandardScaler` which reads `var_ = np.array(self._rs.var_)` (FIXED #2086; was the buggy `scale_**2`) + `n_samples_seen_` + `n_features_in_` (`self._validate_data`) and inherits `fit_transform` — mirroring `_data.py:756-790`/`:1027`/`:1070`. Live default-path oracle matches element-wise. |
//! | REQ-SS-VALUE-PARITY (mean_/scale_/var_/n_samples_seen_ array parity) | SHIPPED | default path: marshals `mean_`/`scale_`/`var_` from the Rust getters (over `FittedStandardScaler::mean()`/`scale()`/`var()`); downstream `ferrolearn-preprocess` REQ-1/5 is bit-identical to the sklearn oracle. FIXED #2087: `RsStandardScaler::scale_` getter now marshals `FittedStandardScaler::scale()` — the `_handle_zeros_in_scale`-clamped value (`1.0` on a constant column), what sklearn exposes as `scale_` (`_data.py:1019-1020`, helper `:88`) — NOT the raw `std()` (`0.0` on a constant column), so on `X=[[1,5],[2,5],[3,5]]` `scale_=[0.816496580927726,1.0]` matching sklearn. FIXED #2086: `RsStandardScaler::var_` getter exposes the TRUE population variance from `FittedStandardScaler::var()` (`0.0` on a constant column), so the wrapper reads `var_ = np.array(self._rs.var_)` instead of the buggy `scale_**2` — sklearn's `var_` is the raw variance computed BEFORE `_handle_zeros_in_scale` clamps `scale_=1.0` on a constant column (`_data.py:1013-1023`), so `var_=[0.6666666666666666,0.0]` while `scale_**2=[0.6666666666666666,1.0]`. Consumer: `_transformers.py::StandardScaler.fit`. Guards `tests/divergence_transformers.py::test_red_standardscaler_scale_constant_column_matches_oracle` + `::test_red_standardscaler_var_constant_column_matches_oracle`. (Non-default with_mean/with_std parity is REQ-SS-WITH-MEAN-STD.) |
//! | REQ-PCA-API-CONFORM (fit/transform/inverse_transform + 5 attrs, default solver) | SHIPPED | `RsPCA::fit`/`transform`/`inverse_transform` + getters `components_`/`explained_variance_`/`explained_variance_ratio_`/`mean_`/`singular_values_`, wrapped by `_transformers.py::PCA` (+ `n_features_in_`), inheriting `fit_transform` — mirroring `_pca.py:691-707`/`:489` with the `svd_flip` sign (`_pca.py:647`). Live default-solver oracle matches element-wise incl. sign. |
//! | REQ-PCA-VALUE-PARITY (components/variance/singular/mean parity incl. svd_flip sign) | SHIPPED | default solver: all five arrays marshalled from the `RsPCA` getters; downstream `ferrolearn-decomp` REQ-1/4/5 match the sklearn oracle ≤1e-6 incl. per-row `svd_flip(u_based_decision=False)` sign. (Repeated-eigenvalue/rank-deficient case downstream #1501; whiten/alt-solver #1502/#1503.) |
//! | REQ-PCA-NCOMP-POSITIONAL (n_components positional ABI) | SHIPPED | FIXED #2035: `_transformers.py::PCA.__init__(self, n_components=2)` drops the keyword-only `*`, so `ferrolearn.PCA(2).n_components == 2` matching sklearn `__init__(self, n_components=None, *, ...)` (`_pca.py:407`). Guard `test_red_pca_n_components_positional`. |
//! | REQ-PCA-NCOMP-DEFAULT-NONE (n_components default None) | SHIPPED | FIXED #2036: wrapper default `n_components=None` stored verbatim (`PCA().n_components is None` matching `_pca.py:409`), resolved at fit time to `min(X.shape)` before constructing `_RsPCA` (and the same resolution in `__setstate__`); an explicit int passes through. Guard `test_red_pca_n_components_default_none`. |
//! | REQ-SS-WITH-MEAN-STD (with_mean/with_std honored) | SHIPPED | FIXED #2037: `RsStandardScaler` now carries `with_mean`/`with_std`/`copy` (`#[pyo3(signature = (with_mean=true, with_std=true, copy=true))]`) and `fit` threads them into the downstream scaler via `StandardScaler::<f64>::new().with_with_mean(..).with_with_std(..).with_copy(..)` (`ferrolearn-preprocess` REQ-6, #1193), so `transform` honors them. The wrapper `_transformers.py::StandardScaler.fit` nulls the fitted attrs to match sklearn: `mean_ = array if (with_mean or with_std) else None`, `scale_`/`var_ = array if with_std else None` (`_data.py:993-995`,`:1022-1023`). Live oracle `X=[[1,10],[2,20],[3,30]]` (R-CHAR-3): (T,T) `transform[0]=[-1.224745,-1.224745]`; (T,F) `scale_=var_=None`, `transform[0]=[-1,-10]`; (F,T) `mean_=[2,20]`, `transform[0]=[1.224745,1.224745]`; (F,F) `mean_=scale_=var_=None`, `transform[0]=[1,10]`. Consumer: `_transformers.py::StandardScaler`. Verified `tests/divergence_transformers.py::test_red_standardscaler_*`. |
//! | REQ-SS-COPY (copy ctor param) | SHIPPED | FIXED #2037: `_transformers.py::StandardScaler.__init__(self, *, copy=True, with_mean=True, with_std=True)` (sklearn param order, `_data.py:835`) stores `self.copy` and threads it into `_RsStandardScaler(self.with_mean, self.with_std, self.copy)` → `StandardScaler::with_copy` (ABI-only downstream: `fit`/`transform` operate on owned copies, so non-mutation holds either way). Verified `tests/divergence_transformers.py::test_red_standardscaler_copy_roundtrip`. |
//! | REQ-PCA-PARAMS (copy/whiten/svd_solver/tol/iterated_power/n_oversamples/power_iteration_normalizer/random_state) | NOT-STARTED | the wrapper exposes `n_components` only; sklearn `_pca.py:407-423`. Default `svd_solver`/`whiten=False` MATCHES, so only the param surface + non-default paths are missing — owned downstream #1502/#1503/#1509. |
//! | REQ-PCA-ATTRS (n_components_ + noise_variance_) | NOT-STARTED | `_RsPCA` has no `n_components_`/`noise_variance_` getter; `FittedPCA` discards the eigenvalue tail. sklearn `_pca.py:691`/`:686-688`. The binding cannot expose attrs the library does not compute — downstream #1508/#1507. |
//! | REQ-CONSUMER (binding IS the public API) | SHIPPED | non-test consumers: `_transformers.py::StandardScaler` constructs `_RsStandardScaler()` and `::PCA` constructs `_RsPCA(...)`, driving fit/transform/inverse_transform + attr reads; `ferrolearn/__init__.py` re-exports both; `test_check_estimator.py` runs each through `parametrize_with_checks` (534 pytest pass). |
//! | REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | marshals via `crate::conversions::*` (rust-numpy + `ndarray`), not `ferray::numpy_interop`/`ferray-core` (R-SUBSTRATE-1); ferray exposes no numpy bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027. |

use crate::conversions::*;
use ferrolearn_core::{Fit, Transform};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsStandardScaler")]
pub struct RsStandardScaler {
    fitted: Option<ferrolearn_preprocess::FittedStandardScaler<f64>>,
    with_mean: bool,
    with_std: bool,
    copy: bool,
}

#[pymethods]
impl RsStandardScaler {
    #[new]
    #[pyo3(signature = (with_mean=true, with_std=true, copy=true))]
    fn new(with_mean: bool, with_std: bool, copy: bool) -> Self {
        Self {
            fitted: None,
            with_mean,
            with_std,
            copy,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let model = ferrolearn_preprocess::StandardScaler::<f64>::new()
            .with_with_mean(self.with_mean)
            .with_with_std(self.with_std)
            .with_copy(self.copy);
        let fitted = model
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

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
        let result = fitted
            .transform(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &result))
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let result = fitted
            .inverse_transform(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &result))
    }

    #[getter]
    fn mean_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.mean()))
    }

    /// Per-column scale used to divide during `transform` — the
    /// `_handle_zeros_in_scale`-clamped value, `1.0` on a constant
    /// (zero-variance) column. Mirrors sklearn `StandardScaler.scale_`, which is
    /// `_handle_zeros_in_scale(sqrt(var_))` (`sklearn/preprocessing/_data.py:1019-1020`,
    /// helper at `:88`) — NOT the raw std (`0.0` on a constant column). Marshals
    /// `FittedStandardScaler::scale()` (downstream REQ-5, #1192), NOT `std()`.
    #[getter]
    fn scale_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.scale()))
    }

    /// True population variance (ddof=0), `0.0` on a constant (zero-variance)
    /// column — mirrors sklearn `StandardScaler.var_`, which is the raw variance
    /// computed BEFORE `_handle_zeros_in_scale` clamps `scale_` to 1.0 there
    /// (`sklearn/preprocessing/_data.py:1013-1023`). This is NOT `scale_**2`:
    /// on a constant column `scale_=1.0` so `scale_**2=1.0`, whereas the true
    /// `var_=0.0`. Marshals `FittedStandardScaler::var()` (downstream REQ-5,
    /// #1192).
    #[getter]
    fn var_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.var()))
    }
}

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsPCA")]
pub struct RsPCA {
    n_components: usize,
    fitted: Option<ferrolearn_decomp::FittedPCA<f64>>,
}

#[pymethods]
impl RsPCA {
    #[new]
    #[pyo3(signature = (n_components=2))]
    fn new(n_components: usize) -> Self {
        Self {
            n_components,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let model = ferrolearn_decomp::PCA::<f64>::new(self.n_components);
        let fitted = model
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

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
        let result = fitted
            .transform(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &result))
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let result = fitted
            .inverse_transform(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &result))
    }

    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.components()))
    }

    #[getter]
    fn explained_variance_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.explained_variance()))
    }

    #[getter]
    fn explained_variance_ratio_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.explained_variance_ratio()))
    }

    #[getter]
    fn mean_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.mean()))
    }

    #[getter]
    fn singular_values_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.singular_values()))
    }
}
