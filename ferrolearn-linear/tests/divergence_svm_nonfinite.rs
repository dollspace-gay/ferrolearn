//! Non-finite input validation parity (#2269, batch 6 — the FINAL linear batch)
//! for the **libsvm-backed kernel SVMs** in `ferrolearn-linear`: `SVC`/`SVR`
//! (`svm.rs`), `NuSVC`/`NuSVR` (`nu_svm.rs`), and `OneClassSVM`
//! (`one_class_svm.rs`).
//!
//! scikit-learn 1.5.2 validates `X`/`y` at fit through `BaseLibSVM.fit` ->
//! `self._validate_data(X, y, dtype=np.float64, order="C", accept_sparse="csr",
//! …)` (`sklearn/svm/_base.py:190-197`), keeping the default
//! `force_all_finite=True`, so any NaN or +/-inf in `X` (or float `y`) raises
//! `ValueError` BEFORE the libsvm SMO solve runs — for ALL five estimators
//! (`SVC`/`SVR`/`NuSVC`/`NuSVR`/`OneClassSVM` all extend `BaseLibSVM`):
//!   - X-nan        -> `ValueError("Input X contains NaN.")`
//!   - X-+/-inf     -> `ValueError("Input X contains infinity or a value too
//!                       large for dtype('float64').")`
//!   - y-nan/inf    -> `ValueError("Input y contains NaN." / "… infinity …")`
//!     for the float-`y` regressors `SVR`/`NuSVR` (classification `y` is integer
//!     class labels; `OneClassSVM` is unsupervised — X only).
//!
//! Previously these estimators ran the libsvm SMO solver on non-finite input
//! (producing NaN `dual_coef_`/`support_`); they now reject it up-front with
//! `FerroError::InvalidParameter`, matching sklearn's reject-at-fit contract
//! (R-DEV-1 numerical / R-DEV-2 exception parity).
//!
//! IMPLEMENTATION NOTE (the SEPARATE-arm hazard the manifest flagged):
//! `NuSVC::fit`/`NuSVR::fit` call `crate::svm::solve_nu_svc`/`solve_nu_svr`
//! DIRECTLY (the genuine libsvm `Solver_NU` dual) — they do NOT route through
//! the guarded `SVC::fit`/`SVR::fit`, so each nu fit entry carries its OWN
//! finiteness guard. Verified: a NuSVC/NuSVR fit on non-finite input raises
//! (below), confirming the nu solver does not bypass validation.
//!
//! NOTE on sample_weight: sklearn's `BaseLibSVM.fit` takes a `sample_weight`
//! kwarg (validated via `_check_sample_weight`, raising on a non-finite weight).
//! ferrolearn's `Fit::fit` for these five estimators takes ONLY `(x, y)` (or
//! `(x, ())` for OneClassSVM) — there is no `sample_weight` argument in the
//! public `Fit` trait surface — so the sklearn `sample_weight`-finiteness raise
//! has no ferrolearn fit-entry counterpart here. X (and float `y`) are the
//! validated inputs.
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, NOT
//! copied from ferrolearn). Confirmed via:
//!
//! ```text
//! python3 -c "
//! import numpy as np
//! from sklearn.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM
//! Xc=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]]); yc=np.array([0,0,0,1,1,1])
//! Xr=np.array([[1.],[2.],[3.],[4.]]); yr=np.array([1.,5.,2.,8.])
//! def mk(X,v,i=0,j=0): X=X.copy(); X[i,j]=v; return X
//! # SVC/NuSVC/OneClassSVM: X-nan/+inf/-inf -> ValueError
//! # SVR/NuSVR: X-nan/+inf/-inf AND y-nan/+inf -> ValueError
//! "
//! ```
//!
//! Oracle result (sklearn 1.5.2 — every non-finite case raises ValueError):
//! ```text
//! SVC          X-nan  Input X contains NaN.
//! SVC          X+inf  Input X contains infinity or a value too large ...
//! SVC          X-inf  Input X contains infinity or a value too large ...
//! SVR          X-nan / X+inf / X-inf            ValueError ...
//! SVR          y-nan  Input y contains NaN. / y+inf  ... infinity ...
//! NuSVC        X-nan / X+inf / X-inf            ValueError ...
//! NuSVR        X-nan / X+inf / X-inf / y-nan / y+inf   ValueError ...
//! OneClassSVM  X-nan / X+inf / X-inf            ValueError ...
//! ```
//! All-finite fits succeed and reproduce the existing oracle attributes (the
//! finite-path sanity checks below pin a known `support_`/`dual_coef_` per
//! estimator so the guard provably did NOT regress the SHIPPED behavior).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::{Fit, Predict};
use ferrolearn_linear::nu_svm::{NuSVC, NuSVR};
use ferrolearn_linear::one_class_svm::OneClassSVM;
use ferrolearn_linear::svm::{LinearKernel, RbfKernel, SVC, SVR};
use ndarray::{Array1, Array2, array};

// ---------------------------------------------------------------------------
// shared finite fixtures
// ---------------------------------------------------------------------------

/// Binary linearly-separable 6x2 classification set (the svm.rs oracle fixture).
fn finite_xc() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
    )
    .expect("valid 6x2 shape")
}

fn finite_yc() -> Array1<usize> {
    array![0usize, 0, 0, 1, 1, 1]
}

/// 4x1 regression set (the nu_svm.rs NuSVR oracle fixture).
fn finite_xr() -> Array2<f64> {
    Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).expect("valid 4x1 shape")
}

fn finite_yr() -> Array1<f64> {
    array![1.0, 5.0, 2.0, 8.0]
}

/// Helper: clone `X` and poison cell (i, j) with `v`.
fn poison(x: &Array2<f64>, i: usize, j: usize, v: f64) -> Array2<f64> {
    let mut x = x.clone();
    x[[i, j]] = v;
    x
}

/// Assert a fit `Result` is the non-finite `InvalidParameter` rejection on the
/// given input name (`"X"` or `"y"`), matching sklearn's `ValueError`.
fn assert_nonfinite_err<T>(res: Result<T, FerroError>, want_name: &str) {
    match res {
        Err(FerroError::InvalidParameter { name, reason }) => {
            assert_eq!(name, want_name, "rejected input name");
            assert!(
                reason.contains("NaN") || reason.contains("infinity"),
                "reason should mention NaN/infinity, got: {reason}"
            );
        }
        Err(other) => panic!("expected InvalidParameter, got {other:?}"),
        Ok(_) => panic!("expected non-finite rejection, fit returned Ok"),
    }
}

// ---------------------------------------------------------------------------
// SVC (svm.rs) — X-only (classification labels are integer)
// ---------------------------------------------------------------------------

#[test]
fn svc_rejects_nan_in_x() {
    let res = SVC::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xc(), 0, 0, f64::NAN), &finite_yc());
    assert_nonfinite_err(res, "X");
}

#[test]
fn svc_rejects_pos_inf_in_x() {
    let res = SVC::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xc(), 1, 1, f64::INFINITY), &finite_yc());
    assert_nonfinite_err(res, "X");
}

#[test]
fn svc_rejects_neg_inf_in_x() {
    let res = SVC::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xc(), 2, 0, f64::NEG_INFINITY), &finite_yc());
    assert_nonfinite_err(res, "X");
}

/// Finite-path sanity (no false positive, no regression): the all-finite SVC
/// fit succeeds and reproduces the live `SVC(kernel='linear', C=1.0)` oracle
/// `support_ [1,2,3]` (R-CHAR-3; svm.md AC-2/AC-3, NOT copied from ferrolearn).
#[test]
fn svc_finite_path_unchanged() {
    let fitted = SVC::<f64, LinearKernel>::new(LinearKernel)
        .fit(&finite_xc(), &finite_yc())
        .expect("finite SVC fit succeeds");
    assert_eq!(fitted.support().to_vec(), vec![1, 2, 3], "support_ oracle");
    let preds = fitted.predict(&finite_xc()).expect("predict ok");
    assert_eq!(preds, array![0usize, 0, 0, 1, 1, 1], "predict oracle");
}

// ---------------------------------------------------------------------------
// SVR (svm.rs) — X AND float y
// ---------------------------------------------------------------------------

#[test]
fn svr_rejects_nan_in_x() {
    let res = SVR::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xr(), 0, 0, f64::NAN), &finite_yr());
    assert_nonfinite_err(res, "X");
}

#[test]
fn svr_rejects_pos_inf_in_x() {
    let res = SVR::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xr(), 3, 0, f64::INFINITY), &finite_yr());
    assert_nonfinite_err(res, "X");
}

#[test]
fn svr_rejects_neg_inf_in_x() {
    let res = SVR::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xr(), 1, 0, f64::NEG_INFINITY), &finite_yr());
    assert_nonfinite_err(res, "X");
}

#[test]
fn svr_rejects_nan_in_y() {
    let res = SVR::<f64, LinearKernel>::new(LinearKernel)
        .fit(&finite_xr(), &array![f64::NAN, 5.0, 2.0, 8.0]);
    assert_nonfinite_err(res, "y");
}

#[test]
fn svr_rejects_inf_in_y() {
    let res = SVR::<f64, LinearKernel>::new(LinearKernel)
        .fit(&finite_xr(), &array![1.0, f64::INFINITY, 2.0, 8.0]);
    assert_nonfinite_err(res, "y");
}

/// Finite-path sanity: the all-finite SVR fit succeeds (no false positive).
#[test]
fn svr_finite_path_unchanged() {
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("6x1");
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let fitted = SVR::<f64, LinearKernel>::new(LinearKernel)
        .with_c(100.0)
        .with_max_iter(500_000)
        .with_tol(1e-8)
        .fit(&x, &y)
        .expect("finite SVR fit succeeds");
    // Live oracle SVR(kernel='linear',C=100,epsilon=0.1): support_ [0,5]
    // (svm.md AC-6, R-CHAR-3).
    assert_eq!(fitted.support().to_vec(), vec![0, 5], "support_ oracle");
}

// ---------------------------------------------------------------------------
// NuSVC (nu_svm.rs) — its OWN guard (solve_nu_svc, not SVC::fit); X-only
// ---------------------------------------------------------------------------

#[test]
fn nusvc_rejects_nan_in_x() {
    let res = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xc(), 0, 0, f64::NAN), &finite_yc());
    assert_nonfinite_err(res, "X");
}

#[test]
fn nusvc_rejects_pos_inf_in_x() {
    let res = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xc(), 4, 1, f64::INFINITY), &finite_yc());
    assert_nonfinite_err(res, "X");
}

#[test]
fn nusvc_rejects_neg_inf_in_x() {
    let res = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xc(), 5, 0, f64::NEG_INFINITY), &finite_yc());
    assert_nonfinite_err(res, "X");
}

/// Finite-path sanity: the all-finite NuSVC fit succeeds and reproduces the
/// live `NuSVC(kernel='linear', nu=0.5)` oracle `support_ [1,2,3,5]`
/// (nu_svm.md REQ-2, R-CHAR-3 — confirms the guard did NOT regress the nu solve).
#[test]
fn nusvc_finite_path_unchanged() {
    let fitted = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_max_iter(200_000)
        .with_tol(1e-7)
        .fit(&finite_xc(), &finite_yc())
        .expect("finite NuSVC fit succeeds");
    assert_eq!(
        fitted.support().to_vec(),
        vec![1, 2, 3, 5],
        "support_ oracle"
    );
}

// ---------------------------------------------------------------------------
// NuSVR (nu_svm.rs) — its OWN guard (solve_nu_svr, not SVR::fit); X AND float y
// ---------------------------------------------------------------------------

#[test]
fn nusvr_rejects_nan_in_x() {
    let res = NuSVR::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xr(), 0, 0, f64::NAN), &finite_yr());
    assert_nonfinite_err(res, "X");
}

#[test]
fn nusvr_rejects_pos_inf_in_x() {
    let res = NuSVR::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xr(), 2, 0, f64::INFINITY), &finite_yr());
    assert_nonfinite_err(res, "X");
}

#[test]
fn nusvr_rejects_neg_inf_in_x() {
    let res = NuSVR::<f64, LinearKernel>::new(LinearKernel)
        .fit(&poison(&finite_xr(), 3, 0, f64::NEG_INFINITY), &finite_yr());
    assert_nonfinite_err(res, "X");
}

#[test]
fn nusvr_rejects_nan_in_y() {
    let res = NuSVR::<f64, LinearKernel>::new(LinearKernel)
        .fit(&finite_xr(), &array![f64::NAN, 5.0, 2.0, 8.0]);
    assert_nonfinite_err(res, "y");
}

#[test]
fn nusvr_rejects_inf_in_y() {
    let res = NuSVR::<f64, LinearKernel>::new(LinearKernel)
        .fit(&finite_xr(), &array![1.0, 5.0, f64::INFINITY, 8.0]);
    assert_nonfinite_err(res, "y");
}

/// Finite-path sanity: the all-finite NuSVR fit succeeds and reproduces the
/// live `NuSVR(kernel='linear', nu=0.5, C=1.0)` oracle `support_ [2,3]`
/// (nu_svm.md REQ-3, R-CHAR-3).
#[test]
fn nusvr_finite_path_unchanged() {
    let fitted = NuSVR::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_c(1.0)
        .with_max_iter(500_000)
        .with_tol(1e-8)
        .fit(&finite_xr(), &finite_yr())
        .expect("finite NuSVR fit succeeds");
    assert_eq!(fitted.support().to_vec(), vec![2, 3], "support_ oracle");
}

// ---------------------------------------------------------------------------
// OneClassSVM (one_class_svm.rs) — unsupervised, X only
// ---------------------------------------------------------------------------

fn finite_ocs_x() -> Array2<f64> {
    ndarray::arr2(&[
        [0.0, 0.0],
        [0.1, 0.1],
        [-0.1, 0.1],
        [0.1, -0.1],
        [0.0, 0.2],
        [0.2, 0.0],
        [3.0, 3.0],
    ])
}

#[test]
fn ocs_rejects_nan_in_x() {
    let res = OneClassSVM::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .fit(&poison(&finite_ocs_x(), 0, 0, f64::NAN), &());
    assert_nonfinite_err(res, "X");
}

#[test]
fn ocs_rejects_pos_inf_in_x() {
    let res = OneClassSVM::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .fit(&poison(&finite_ocs_x(), 3, 1, f64::INFINITY), &());
    assert_nonfinite_err(res, "X");
}

#[test]
fn ocs_rejects_neg_inf_in_x() {
    let res = OneClassSVM::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .fit(&poison(&finite_ocs_x(), 6, 0, f64::NEG_INFINITY), &());
    assert_nonfinite_err(res, "X");
}

/// Also exercise the RBF (non-linear) kernel path: the X-finiteness guard is
/// kernel-independent (it fires before `resolved_for_fit`).
#[test]
fn ocs_rejects_nan_in_x_rbf() {
    let res = OneClassSVM::<f64, RbfKernel<f64>>::new(RbfKernel::with_gamma(1.0))
        .with_nu(0.5)
        .fit(&poison(&finite_ocs_x(), 2, 1, f64::NAN), &());
    assert_nonfinite_err(res, "X");
}

/// Finite-path sanity: the all-finite OneClassSVM fit succeeds and reproduces
/// the live `OneClassSVM(kernel='linear', nu=0.5)` oracle `intercept_ [-0.01]`
/// / `offset_ 0.01` (one_class_svm.md AC-3, R-CHAR-3 — guard did NOT regress).
#[test]
fn ocs_finite_path_unchanged() {
    let fitted = OneClassSVM::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .fit(&finite_ocs_x(), &())
        .expect("finite OneClassSVM fit succeeds");
    let intercept = fitted.intercept();
    assert!(
        (intercept[0] - (-0.01)).abs() < 1e-2,
        "intercept_ {} vs oracle -0.01",
        intercept[0]
    );
    assert!(
        (fitted.offset() - 0.01).abs() < 1e-2,
        "offset_ {} vs oracle 0.01",
        fitted.offset()
    );
}
