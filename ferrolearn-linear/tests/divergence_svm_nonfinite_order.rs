//! Validation-ORDER parity (#2270) for the batch-6 non-finite guards added to
//! the libsvm-backed kernel SVMs (`SVC`/`SVR` in `svm.rs`, `NuSVC`/`NuSVR` in
//! `nu_svm.rs`).
//!
//! scikit-learn 1.5.2's `BaseLibSVM.fit` validates the data through
//! `self._validate_data(X, y, dtype=np.float64, …)` (`sklearn/svm/_base.py:190`),
//! which routes to `sklearn.utils.validation.check_X_y`. `check_X_y` runs
//! `check_array(X, force_all_finite=True)` — the **X-finiteness** check —
//! BEFORE `check_consistent_length(X, y)` — the **X/y length** check. So when an
//! input has BOTH a non-finite cell in `X` AND a mismatched `y` length, sklearn
//! raises the finiteness `ValueError("Input X contains NaN.")`, NOT a
//! length/consistency error.
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, NOT
//! copied from ferrolearn). With `X = [[1, NaN], [2, 2], [3, 3]]` (3 rows, a NaN
//! at [0,1]) and `y = [0, 1]` (length 2 ≠ 3 rows):
//!
//! ```text
//! $ cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.utils.validation import check_X_y
//! X=np.array([[1.,np.nan],[2.,2.],[3.,3.]]); y=np.array([0,1])
//! try: check_X_y(X,y)
//! except Exception as e: print(type(e).__name__, str(e)[:30])"
//! ValueError Input X contains NaN.
//! $ # and through the full estimator entry points:
//! $ cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.svm import SVC, SVR, NuSVC, NuSVR
//! X=np.array([[1.,np.nan],[2.,2.],[3.,3.]]); y=np.array([0,1])
//! for est in (SVC(),NuSVC()):  est.fit(X,y)          # -> ValueError Input X contains NaN.
//! for est in (SVR(),NuSVR()):  est.fit(X,y.astype(float))  # -> ValueError Input X contains NaN."
//! ```
//!
//! Every one of `SVC`/`SVR`/`NuSVC`/`NuSVR` raises `ValueError("Input X contains
//! NaN.")` — the **finiteness** error — on this input.
//!
//! FIXED (#2270): each of `SVC::fit`/`SVR::fit` (`svm.rs`) and
//! `NuSVC::fit`/`NuSVR::fit` (`nu_svm.rs`) now runs the X-finiteness guard (and
//! the `y`-finiteness guard for the regressors) BEFORE the X/y length
//! (`ShapeMismatch`) check, matching `check_X_y`'s
//! `check_array(force_all_finite=True)`-then-`check_consistent_length` order. So
//! on the BOTH-bad input the four fits now return the finiteness
//! `InvalidParameter` (sklearn's `ValueError("Input X contains NaN.")`), NOT a
//! `ShapeMismatch`. The 25 batch-6 non-finite tests all use a WELL-SHAPED `y`, so
//! none of them exercises this ordering; these four pins do.
//!
//! Tracking: #2270.

use ferrolearn_core::Fit;
use ferrolearn_core::error::FerroError;
use ferrolearn_linear::nu_svm::{NuSVC, NuSVR};
use ferrolearn_linear::svm::{LinearKernel, SVC, SVR};
use ndarray::{Array1, Array2, array};

/// 3-row X with a NaN at [0,1].
fn xc_nan() -> Array2<f64> {
    Array2::from_shape_vec((3, 2), vec![1.0, f64::NAN, 2.0, 2.0, 3.0, 3.0])
        .expect("valid 3x2 shape")
}

/// length-2 labels (≠ 3 rows) → triggers ferrolearn's ShapeMismatch.
fn yc_short() -> Array1<usize> {
    array![0usize, 1]
}

/// Assert the fit raised the **finiteness** rejection (sklearn's
/// `ValueError("Input X contains NaN.")`), i.e. `InvalidParameter` on `X`
/// mentioning NaN/infinity — NOT a `ShapeMismatch`.
fn assert_finite_err_not_shape<T>(res: Result<T, FerroError>) {
    match res {
        Err(FerroError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "X", "sklearn reports the X-finiteness error");
            assert!(
                reason.contains("NaN") || reason.contains("infinity"),
                "reason should mention NaN/infinity, got: {reason}"
            );
        }
        Err(FerroError::ShapeMismatch { .. }) => panic!(
            "DIVERGENCE #2270: ferrolearn returned ShapeMismatch, but sklearn's \
             check_X_y validates X-finiteness BEFORE X/y length, so it raises \
             ValueError(\"Input X contains NaN.\")"
        ),
        Err(other) => panic!("expected finiteness InvalidParameter, got {other:?}"),
        Ok(_) => panic!("expected a rejection, fit returned Ok"),
    }
}

/// `SVC::fit` runs the X-finiteness guard BEFORE the X/y length
/// (`ShapeMismatch`) check, matching sklearn's `check_X_y`
/// (`sklearn/svm/_base.py:190`) which validates X-finiteness first.
/// Input: `X=[[1,NaN],[2,2],[3,3]]`, `y=[0,1]`.
/// sklearn raises `ValueError("Input X contains NaN.")`; ferrolearn returns the
/// finiteness `InvalidParameter` (NOT `ShapeMismatch`).
/// Tracking: #2270
#[test]
fn svc_nonfinite_x_precedes_length_check() {
    let res = SVC::<f64, LinearKernel>::new(LinearKernel).fit(&xc_nan(), &yc_short());
    assert_finite_err_not_shape(res);
}

/// `SVR::fit` runs the X-/y-finiteness guards BEFORE the X/y length
/// (`ShapeMismatch`) check; sklearn's `check_X_y` validates finiteness first.
/// Input: `X=[[1,NaN],[2,2],[3,3]]`, `y=[0.0,1.0]`.
/// sklearn raises `ValueError("Input X contains NaN.")`; ferrolearn returns the
/// finiteness `InvalidParameter` (NOT `ShapeMismatch`).
/// Tracking: #2270
#[test]
fn svr_nonfinite_x_precedes_length_check() {
    let res = SVR::<f64, LinearKernel>::new(LinearKernel).fit(&xc_nan(), &array![0.0f64, 1.0]);
    assert_finite_err_not_shape(res);
}

/// `NuSVC::fit` runs the X-finiteness guard BEFORE the X/y length
/// (`ShapeMismatch`) check; sklearn's `check_X_y` validates X-finiteness first.
/// Input: `X=[[1,NaN],[2,2],[3,3]]`, `y=[0,1]`.
/// sklearn raises `ValueError("Input X contains NaN.")`; ferrolearn returns the
/// finiteness `InvalidParameter` (NOT `ShapeMismatch`).
/// Tracking: #2270
#[test]
fn nusvc_nonfinite_x_precedes_length_check() {
    let res = NuSVC::<f64, LinearKernel>::new(LinearKernel).fit(&xc_nan(), &yc_short());
    assert_finite_err_not_shape(res);
}

/// `NuSVR::fit` runs the X-/y-finiteness guards BEFORE the X/y length
/// (`ShapeMismatch`) check; sklearn's `check_X_y` validates finiteness first.
/// Input: `X=[[1,NaN],[2,2],[3,3]]`, `y=[0.0,1.0]`.
/// sklearn raises `ValueError("Input X contains NaN.")`; ferrolearn returns the
/// finiteness `InvalidParameter` (NOT `ShapeMismatch`).
/// Tracking: #2270
#[test]
fn nusvr_nonfinite_x_precedes_length_check() {
    let res = NuSVR::<f64, LinearKernel>::new(LinearKernel).fit(&xc_nan(), &array![0.0f64, 1.0]);
    assert_finite_err_not_shape(res);
}
