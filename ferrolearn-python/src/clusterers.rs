//! PyO3 bindings for clustering models.
//!
//! ## REQ status
//!
//! The `sklearn.cluster.KMeans` binding shim: `#[pyclass(name = "_RsKMeans")]`
//! over `ferrolearn_cluster::KMeans`, wrapped by `ferrolearn.KMeans`
//! (`BaseEstimator`/`ClusterMixin`/`TransformerMixin`) in
//! `python/ferrolearn/_clusterers.py`. This unit owns the sklearn-API
//! marshalling surface only (constructor ABI, attribute exposure, method
//! surface, array coercion); the KMeans MATH (k-means++ seeding, Lloyd
//! iteration, convergence, `inertia_`) lives in `ferrolearn-cluster/src/kmeans.rs`
//! (blockers #1036/#1038/#1039/#1044). Verification model B: pytest comparing
//! `import ferrolearn` against `import sklearn` 1.5.2 (live oracle;
//! `sklearn/cluster/_kmeans.py:1196`). Design doc: `.design/python/clusterers.md`
//! (11 REQs). Every REQ is BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a
//! concrete blocker). Verified via `tests/divergence_clusterers.py`,
//! `tests/test_check_estimator.py`, and `tests/test_cross_val_score.py`
//! (524 pytest pass).
//!
//! **4 SHIPPED / 7 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-API-CONFORM (fit/predict/transform + mixins + attrs) | SHIPPED | `RsKMeans::fit`/`predict`/`transform` + getters `cluster_centers_`/`labels_`/`inertia_`/`n_iter_` (over `FittedKMeans<f64>`), wrapped by `_clusterers.py::KMeans` which also sets `n_features_in_` (`self._validate_data`) and inherits `fit_predict`/`fit_transform` from the mixins — mirroring `_kmeans.py:1291-1311` + `:1047`/`:1072`/`:1106`/`:1130`. `check_estimator` + `cross_val_score` pass; live shapes/types match the sklearn oracle. |
//! | REQ-NCLUSTERS-POSITIONAL (n_clusters positional ABI) | SHIPPED | FIXED #2029: `_clusterers.py::__init__(self, n_clusters=8, *, ...)` moves `n_clusters` before the `*`, so `ferrolearn.KMeans(2).n_clusters == 2` matching sklearn `__init__(self, n_clusters=8, *, ...)` (`_kmeans.py:1387`). Guard `test_n_clusters_positional_matches_sklearn`. |
//! | REQ-NINIT-DEFAULT (n_init='auto' default) | SHIPPED | FIXED #2030: wrapper default `n_init='auto'` stored verbatim (so `KMeans().n_init == 'auto'` matching `_kmeans.py:1392`), resolved at fit time to `1` for the k-means++ default (`_kmeans.py:1240-1243`); an explicit int passes through unchanged. Guard `test_n_init_default_matches_sklearn`. |
//! | REQ-INIT-PARAM (init constructor param) | NOT-STARTED | no `init` param; sklearn `init ∈ {'k-means++','random'}|callable|array` (`_kmeans.py:1391`). Default k-means++ MATCHES (the library always does greedy k-means++), so only the param surface + non-default inits are missing — owned downstream by `ferrolearn-cluster` (blocker #1038). Binding blocker #2031. |
//! | REQ-ALGORITHM-PARAM (algorithm lloyd/elkan) | NOT-STARTED | no `algorithm` param; sklearn `{'lloyd','elkan'}` default `'lloyd'` (`_kmeans.py:1384`). Lloyd default MATCHES; the param + Elkan path are missing (Lloyd-only in `kmeans.rs`). Blocker #2031. |
//! | REQ-COPYX-PARAM (copy_x) | NOT-STARTED | no `copy_x` param; sklearn bool default `True` (`_kmeans.py:1383`). The wrappers already copy via `_validate_data`, so non-mutation holds, but the constructor-ABI param is absent. Blocker #2031. |
//! | REQ-VERBOSE-PARAM (verbose) | NOT-STARTED | no `verbose` param; sklearn int default `0` (`_kmeans.py:1259`). Blocker #2031. |
//! | REQ-SCORE (score = -inertia) | NOT-STARTED | no `score` method; sklearn `KMeans.score(X)` returns `-inertia` (`_kmeans.py:1156-1184`; the mixins supply none — sklearn's lives on `_BaseKMeans`). Blocker #2032. |
//! | REQ-VALUE-PARITY (exact centers/labels/inertia) | NOT-STARTED | the binding marshals faithfully, but `ferrolearn_cluster` seeds k-means++ with `StdRng`, not numpy's RNG, so exact centroids/labels diverge on the same `random_state` (R-SUBSTRATE-5). Deterministic STRUCTURE (k centers, valid labels, non-negative inertia, `predict == transform.argmin(1)`, RNG-robust label parity on separated blobs) is verified; exact VALUES are not. Blocker #2033 (downstream #1039/#1036). |
//! | REQ-SUBSTRATE (ferray::numpy_interop) | NOT-STARTED | marshals via `crate::conversions::*` (rust-numpy + `ndarray`), not `ferray::numpy_interop`/`ferray-core` (R-SUBSTRATE-1); ferray exposes no numpy bridge (R-SUBSTRATE-5). Owned by `conversions.md` #2027. |
//! | REQ-CONSUMER (binding IS the public API) | SHIPPED | non-test consumers: `_clusterers.py::KMeans` constructs `_RsKMeans` and drives fit/predict/transform + attribute reads; `ferrolearn/__init__.py` re-exports `KMeans`; `test_check_estimator.py` + `test_cross_val_score.py` exercise it (524 pytest pass). |

use crate::conversions::*;
use ferrolearn_core::{Fit, Predict, Transform};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsKMeans")]
pub struct RsKMeans {
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    n_init: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_cluster::FittedKMeans<f64>>,
}

#[pymethods]
impl RsKMeans {
    #[new]
    #[pyo3(signature = (n_clusters=8, max_iter=300, tol=1e-4, n_init=10, random_state=None))]
    fn new(
        n_clusters: usize,
        max_iter: usize,
        tol: f64,
        n_init: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_clusters,
            max_iter,
            tol,
            n_init,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let mut model = ferrolearn_cluster::KMeans::<f64>::new(self.n_clusters)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_n_init(self.n_init);
        if let Some(seed) = self.random_state {
            model = model.with_random_state(seed);
        }
        let fitted = model
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

    #[getter]
    fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.cluster_centers()))
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_usize_to_numpy(py, fitted.labels()))
    }

    #[getter]
    fn inertia_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.inertia())
    }

    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_iter())
    }
}
