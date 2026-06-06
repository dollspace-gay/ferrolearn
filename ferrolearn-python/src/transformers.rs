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
//! doc: `.design/python/transformers.md` (14 REQs). Every REQ is BINARY
//! (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker). Verified via
//! `tests/divergence_transformers.py` + `tests/test_check_estimator.py`
//! (534 pytest pass).
//!
//! **13 SHIPPED / 2 NOT-STARTED.**
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
//! | REQ-PCA-WHITEN (whiten ctor param) | SHIPPED (#2098) | FIXED — the downstream `PCA::with_whiten` builder (`ferrolearn-decomp/src/pca.rs:180`, REQ-11 #1502) already ships sklearn's whitening (`Transform::transform` divides each column by `sqrt(explained_variance_)`, `inverse_transform` re-multiplies; `_base.py:157-165`/`:192-196`), so the binding now threads the param. `RsPCA` carries `whiten: bool` (`#[pyo3(signature = (n_components=2, whiten=false))]`) and `fit` builds `PCA::<f64>::new(self.n_components).with_whiten(self.whiten)`. Non-test consumer: `_transformers.py::PCA.__init__(self, n_components=None, *, whiten=False)` stores `self.whiten` and `fit` passes `whiten=self.whiten` into `_RsPCA(...)` (and `__setstate__` rebuilds with `whiten=getattr(self, "whiten", False)`). Live oracle (R-CHAR-3) `X=[[1,2,3],[4,5,7],[2,0,1],[8,6,5],[3,3,2],[0,1,4]]`, `PCA(n_components=2, whiten=True).fit_transform(X)` matches sklearn element-wise to 1e-6; `whiten=False` is byte-identical to the default path. Guard `tests/divergence_transformers.py::test_pca_whiten_transform_matches_sklearn`. |
//! | REQ-PCA-PARAMS (copy/svd_solver/tol/iterated_power/n_oversamples/power_iteration_normalizer/random_state) | NOT-STARTED | the wrapper exposes `n_components`+`whiten` only; sklearn `_pca.py:407-423`. The remaining params' default `svd_solver` MATCHES, so only the param surface + non-default paths are missing — owned downstream #1503/#1509. (`whiten` is now SHIPPED, see REQ-PCA-WHITEN.) |
//! | REQ-PCA-ATTRS (n_components_ + noise_variance_) | SHIPPED (#2097) | FIXED — the downstream prereqs (`ferrolearn-decomp` REQ-16 #1505 `n_components_`, REQ-15 #1507 `noise_variance_`) already SHIPPED, so the binding now exposes both. `RsPCA::n_components_` getter marshals `FittedPCA::n_components_()` (`ferrolearn-decomp/src/pca.rs:283`); `RsPCA::noise_variance_` getter marshals `FittedPCA::noise_variance()` (`pca.rs:304`) — the full-spectrum tail mean `mean(sorted_eigenvalues[n_comp..min_dim])` or `0.0` when all kept (`_pca.py:686-688`). Consumer: `_transformers.py::PCA.fit` sets `n_components_ = int(self._rs.n_components_)` + `noise_variance_ = float(self._rs.noise_variance_)`. Live oracle (R-CHAR-3) `X=[[1,2,3],[4,5,7],[2,0,1],[8,6,5],[3,3,2],[0,1,4]]`: `PCA(2)` → `n_components_=2`, `noise_variance_=0.3132465241238894` (non-zero); `PCA(3)` → `noise_variance_=0.0`. Verified `tests/divergence_transformers.py::test_pca_n_components_and_noise_variance_match_sklearn`. |
//! | REQ-PCA-SCORE (score + score_samples Gaussian log-likelihood) | SHIPPED (#2099) | the downstream prereq already SHIPPED (`ferrolearn-decomp` REQ-15 #1507): `FittedPCA::score_samples` (`ferrolearn-decomp/src/pca.rs:484`) is the per-sample log-likelihood `-0.5*sum(Xr*(Xr@precision),1) - 0.5*(n_features*log(2pi) - logdet(precision))`, and `FittedPCA::score` (`pca.rs:533`) is its mean, both matching the live sklearn 1.5.2 oracle to 1e-6. The binding now surfaces them: `RsPCA::score_samples` (returns a `(n_samples,)` numpy array via `ndarray1_to_numpy`) + `RsPCA::score` (returns `f64`), each mapping `FerroError -> PyValueError`. Mirrors sklearn `PCA.score_samples`/`PCA.score` (`sklearn/decomposition/_pca.py:805-853`). Non-test consumer: `_transformers.py::PCA.score_samples` + `_transformers.py::PCA.score` (`check_is_fitted` + `_validate_data(reset=False)` + `_ensure_f64`, rebuilding `_rs` as `__setstate__` does if absent). Live oracle (R-CHAR-3) `X=[[1,2,3],[4,5,7],[2,0,1],[8,6,5],[3,3,2],[0,1,4]]`, `PCA(n_components=2)`: `score(X)=-5.358925927374854`, `score_samples(X)=[-4.6972115,-5.3570670,-5.8335549,-5.7184482,-5.4976041,-5.0496698]`. Guard `tests/divergence_transformers.py::test_pca_score_and_score_samples_match_sklearn`. |
//! | REQ-PCA-COVARIANCE-PRECISION (get_covariance + get_precision) | SHIPPED (#2100) | the downstream prereq already SHIPPED (`ferrolearn-decomp` REQ-14 #1505): `FittedPCA::get_covariance` (`ferrolearn-decomp/src/pca.rs:328`) is the INFALLIBLE data covariance `components_ᵀ·diag(exp_var_diff)·components_ + noise_variance_·I`, and `FittedPCA::get_precision` (`pca.rs:390`) is its eigendecomposed inverse (returns a `Result`), both matching the live sklearn 1.5.2 oracle to 1e-6. The binding now surfaces them: `RsPCA::get_covariance` (returns a `(n_features, n_features)` numpy array via `ndarray2_to_numpy`, no error mapping — infallible) + `RsPCA::get_precision` (same shape, mapping `FerroError -> PyValueError`). Neither takes an `X` (the model's own data covariance / its inverse). Mirrors sklearn `PCA.get_covariance`/`PCA.get_precision` (`sklearn/decomposition/_base.py:30-101`). Non-test consumer: `_transformers.py::PCA.get_covariance` + `_transformers.py::PCA.get_precision` (`check_is_fitted` + `_ensure_rs`, no `_validate_data` since there is no `X` — matching sklearn's `get_covariance(self)`/`get_precision(self)` signatures). Live oracle (R-CHAR-3) `X=[[1,2,3],[4,5,7],[2,0,1],[8,6,5],[3,3,2],[0,1,4]]`, `PCA(n_components=2)`: `get_covariance()` diag `[8.0, 5.3666667, 4.6666667]`; `get_precision()` diag `[0.7432854, 2.0460427, 0.7745159]`. Guard `tests/divergence_transformers.py::test_pca_get_covariance_and_precision_match_sklearn`. |
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
    whiten: bool,
    fitted: Option<ferrolearn_decomp::FittedPCA<f64>>,
}

#[pymethods]
impl RsPCA {
    #[new]
    #[pyo3(signature = (n_components=2, whiten=false))]
    fn new(n_components: usize, whiten: bool) -> Self {
        Self {
            n_components,
            whiten,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let model = ferrolearn_decomp::PCA::<f64>::new(self.n_components).with_whiten(self.whiten);
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

    /// The resolved number of retained components, `n_components_` (int).
    /// Marshals `FittedPCA::n_components_()` (`ferrolearn-decomp/src/pca.rs`,
    /// REQ-16), the row count of `components_`. Mirrors sklearn
    /// `PCA.n_components_` (`sklearn/decomposition/_pca.py:691`).
    #[getter]
    fn n_components_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_components_())
    }

    /// Estimated noise covariance of the probabilistic-PCA model,
    /// `noise_variance_` (float). Marshals `FittedPCA::noise_variance()`
    /// (`ferrolearn-decomp/src/pca.rs`, REQ-15) — the mean of the discarded tail
    /// eigenvalues `mean(explained_variance_[n_components_:min(n, p)])`, or `0.0`
    /// when all components are kept. Mirrors sklearn `PCA.noise_variance_`
    /// (`sklearn/decomposition/_pca.py:686-688`).
    #[getter]
    fn noise_variance_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.noise_variance())
    }

    /// Per-sample Gaussian log-likelihood under the fitted PCA model,
    /// `score_samples(X)` (shape `(n_samples,)`). Marshals
    /// `FittedPCA::score_samples()` (`ferrolearn-decomp/src/pca.rs:484`, REQ-15
    /// #1507) — `-0.5 * sum(Xr * (Xr @ precision), axis=1) - 0.5 * (n_features *
    /// log(2*pi) - logdet(precision))` with `precision = get_precision()`.
    /// Mirrors sklearn `PCA.score_samples` (`sklearn/decomposition/_pca.py:805-830`).
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
        let result = fitted
            .score_samples(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &result))
    }

    /// Average Gaussian log-likelihood of all samples under the fitted PCA model,
    /// `score(X)` (float) — the mean of `score_samples(X)`. Marshals
    /// `FittedPCA::score()` (`ferrolearn-decomp/src/pca.rs:533`, REQ-15 #1507).
    /// Mirrors sklearn `PCA.score` (`sklearn/decomposition/_pca.py:832-853`).
    fn score(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let result = fitted
            .score(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(result)
    }

    /// Data covariance `(n_features, n_features)` of the generative
    /// (probabilistic-PCA) model, `get_covariance()`. Marshals
    /// `FittedPCA::get_covariance()` (`ferrolearn-decomp/src/pca.rs:328`, REQ-14
    /// #1505) — `components_ᵀ · diag(exp_var_diff) · components_ +
    /// noise_variance_ · I`. Takes no `X` (the model's own data covariance);
    /// infallible (returns `Array2` directly). Mirrors sklearn
    /// `PCA.get_covariance` (`sklearn/decomposition/_base.py:30-56`).
    fn get_covariance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let cov = fitted.get_covariance();
        Ok(ndarray2_to_numpy(py, &cov))
    }

    /// Data precision matrix `(n_features, n_features)` (inverse of the data
    /// covariance) of the generative model, `get_precision()`. Marshals
    /// `FittedPCA::get_precision()` (`ferrolearn-decomp/src/pca.rs:390`, REQ-14
    /// #1505) — the eigendecomposed inverse of `get_covariance()`. Takes no `X`;
    /// fallible (`FerroError -> PyValueError`, e.g. ill-conditioned covariance).
    /// Mirrors sklearn `PCA.get_precision` (`sklearn/decomposition/_base.py:58-101`).
    fn get_precision<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let prec = fitted
            .get_precision()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &prec))
    }
}
