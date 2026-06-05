//! Conversion utilities between ndarray and numpy arrays.
//!
//! ## REQ status
//!
//! The numpy↔ndarray PyO3 marshalling bridge for the ferrolearn-python binding
//! shim — the `ferray::numpy_interop` analog. Verification model B: pytest
//! comparing `import ferrolearn` against `import sklearn` 1.5.2 (the dtype/shape
//! boundary contract mirrors `sklearn/utils/validation.py` `check_array`). Design
//! doc: `.design/python/conversions.md` (6 REQs). Every REQ is BINARY (R-DEFER-2):
//! SHIPPED or NOT-STARTED (with a concrete blocker). Verified via
//! `ferrolearn-python/tests/divergence_conversions.py` (518 pytest pass).
//!
//! **3 SHIPPED / 3 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-F64-ROUNDTRIP (f64 dtype/shape preservation) | SHIPPED | `numpy{1,2}_to_ndarray` + `ndarray{1,2}_to_numpy` preserve f64 shape/values exactly (rust-numpy copy); estimator fit/predict/transform round-trips return float64 arrays matching sklearn ≤1e-8. The f64 dtype is enforced upstream in Python (`_validate_data(dtype="float64")`), R-CODE-4. Guards `*_round_trip` in `divergence_conversions.py`. |
//! | REQ-LABEL-MARSHAL (i64↔usize labels, wrapper-guarded) | SHIPPED | `numpy1_to_ndarray_usize` (`v as usize`) / `ndarray1_usize_to_numpy` (`v as i64`) only ever see non-negative contiguous ints because `_classifiers.py::_encode_labels` runs a LabelEncoder-equivalent (`np.unique`→`0..n`) before the Rust `fit` and `_decode_labels` restores originals on `predict`. Verified vs sklearn for negative `[-1,1]`, string `['cat','dog']`, and non-contiguous `[0,2]` labels (classes_ + predict match). |
//! | REQ-CONSUMER (production consumer) | SHIPPED | every estimator binding (`regressors.rs`/`classifiers.rs`/`transformers.rs`/`clusterers.rs`/`extras.rs`) marshals through these fns; 518 pytest pass. |
//! | REQ-DTYPE-COVERAGE (float32/int preservation) | NOT-STARTED | only f64 arrays + i64 labels; the Python wrapper up-casts float32→float64 (sklearn preserves float32 dtype). R-CODE-5 narrowing. Blocker #2025. |
//! | REQ-SPARSE-INPUT (scipy.sparse X) | NOT-STARTED | dense numpy only; sklearn estimators accept `scipy.sparse` X. Blocker #2026. |
//! | REQ-FERRAY (ferray::numpy_interop substrate) | NOT-STARTED | `rust-numpy` + `ndarray` vs the ferray numpy-bridge analog (R-SUBSTRATE-1). Blocker #2027. |

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Convert a numpy 2D array (read-only) to an ndarray `Array2<f64>`.
pub fn numpy2_to_ndarray(x: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    x.as_array().to_owned()
}

/// Convert a numpy 1D array (read-only) to an ndarray `Array1<f64>`.
pub fn numpy1_to_ndarray(x: PyReadonlyArray1<'_, f64>) -> Array1<f64> {
    x.as_array().to_owned()
}

/// Convert a numpy 1D array of i64 to an ndarray `Array1<usize>` (for class labels).
pub fn numpy1_to_ndarray_usize(y: PyReadonlyArray1<'_, i64>) -> Array1<usize> {
    y.as_array().mapv(|v| v as usize)
}

/// Convert an ndarray `Array1<f64>` to a numpy 1D array.
pub fn ndarray1_to_numpy<'py>(py: Python<'py>, a: &Array1<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_array(py, a)
}

/// Convert an ndarray `Array2<f64>` to a numpy 2D array.
pub fn ndarray2_to_numpy<'py>(py: Python<'py>, a: &Array2<f64>) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_array(py, a)
}

/// Convert an ndarray `Array1<usize>` to a numpy 1D array of i64.
pub fn ndarray1_usize_to_numpy<'py>(
    py: Python<'py>,
    a: &Array1<usize>,
) -> Bound<'py, PyArray1<i64>> {
    let converted = a.mapv(|v| v as i64);
    PyArray1::from_array(py, &converted)
}
