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
//! DIVERGENCE: ferrolearn's batch-6 guard fires the X-finiteness check AFTER the
//! X/y length (`ShapeMismatch`) check at each fit entry:
//!   - `SVC::fit`   — `svm.rs:2027` (ShapeMismatch) precedes `svm.rs:2051` (finite)
//!   - `SVR::fit`   — `svm.rs:3288` (ShapeMismatch) precedes `svm.rs:3320` (finite)
//!   - `NuSVC::fit` — `nu_svm.rs:253` (ShapeMismatch) precedes `nu_svm.rs:272` (finite)
//!   - `NuSVR::fit` — `nu_svm.rs:671` (ShapeMismatch) precedes `nu_svm.rs:690` (finite)
//! so on the BOTH-bad input ferrolearn returns `FerroError::ShapeMismatch` where
//! sklearn raises the finiteness `ValueError` — an exception-class divergence
//! (R-DEV-2 exception/ABI parity), the SAME validation-ORDER category pinned for
//! the batch-5 CV estimators. The 25 batch-6 non-finite tests all use a
//! WELL-SHAPED `y`, so none of them exercises this ordering.
//!
//! Tracking: #2270.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::Fit;
use ferrolearn_linear::nu_svm::{NuSVC, NuSVR};
use ferrolearn_linear::svm::{LinearKernel, SVC, SVR};
use ndarray::{array, Array1, Array2};

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

/// Divergence: `SVC::fit` checks X/y length (`svm.rs:2027`, ShapeMismatch)
/// BEFORE X-finiteness (`svm.rs:2051`); sklearn's `check_X_y`
/// (`sklearn/svm/_base.py:190`) checks X-finiteness first.
/// Input: `X=[[1,NaN],[2,2],[3,3]]`, `y=[0,1]`.
/// sklearn raises `ValueError("Input X contains NaN.")`; ferrolearn returns
/// `ShapeMismatch`.
/// Tracking: #2270
#[test]
#[ignore = "divergence: SVC non-finite-X guard ordered after y-length check; tracking #2270"]
fn svc_nonfinite_x_precedes_length_check() {
    let res = SVC::<f64, LinearKernel>::new(LinearKernel).fit(&xc_nan(), &yc_short());
    assert_finite_err_not_shape(res);
}

/// Divergence: `SVR::fit` checks X/y length (`svm.rs:3288`) BEFORE X-finiteness
/// (`svm.rs:3320`); sklearn checks X-finiteness first.
/// Input: `X=[[1,NaN],[2,2],[3,3]]`, `y=[0.0,1.0]`.
/// sklearn raises `ValueError("Input X contains NaN.")`; ferrolearn returns
/// `ShapeMismatch`.
/// Tracking: #2270
#[test]
#[ignore = "divergence: SVR non-finite-X guard ordered after y-length check; tracking #2270"]
fn svr_nonfinite_x_precedes_length_check() {
    let res = SVR::<f64, LinearKernel>::new(LinearKernel).fit(&xc_nan(), &array![0.0f64, 1.0]);
    assert_finite_err_not_shape(res);
}

/// Divergence: `NuSVC::fit` checks X/y length (`nu_svm.rs:253`) BEFORE
/// X-finiteness (`nu_svm.rs:272`); sklearn checks X-finiteness first.
/// Input: `X=[[1,NaN],[2,2],[3,3]]`, `y=[0,1]`.
/// sklearn raises `ValueError("Input X contains NaN.")`; ferrolearn returns
/// `ShapeMismatch`.
/// Tracking: #2270
#[test]
#[ignore = "divergence: NuSVC non-finite-X guard ordered after y-length check; tracking #2270"]
fn nusvc_nonfinite_x_precedes_length_check() {
    let res = NuSVC::<f64, LinearKernel>::new(LinearKernel).fit(&xc_nan(), &yc_short());
    assert_finite_err_not_shape(res);
}

/// Divergence: `NuSVR::fit` checks X/y length (`nu_svm.rs:671`) BEFORE
/// X-finiteness (`nu_svm.rs:690`); sklearn checks X-finiteness first.
/// Input: `X=[[1,NaN],[2,2],[3,3]]`, `y=[0.0,1.0]`.
/// sklearn raises `ValueError("Input X contains NaN.")`; ferrolearn returns
/// `ShapeMismatch`.
/// Tracking: #2270
#[test]
#[ignore = "divergence: NuSVR non-finite-X guard ordered after y-length check; tracking #2270"]
fn nusvr_nonfinite_x_precedes_length_check() {
    let res = NuSVR::<f64, LinearKernel>::new(LinearKernel).fit(&xc_nan(), &array![0.0f64, 1.0]);
    assert_finite_err_not_shape(res);
}
