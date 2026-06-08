//! Non-finite input validation parity for the SGD `partial_fit` entry points
//! that the #2263 batch-4 sweep MISSED.
//!
//! The batch-4 sweep (`divergence_linear_nonfinite_batch4.rs`) added the
//! `x.iter().any(|v| !v.is_finite())` guard only to the `Fit::fit` /
//! `fit_with_sample_weight` entries of `SGDClassifier` / `SGDRegressor`. It did
//! NOT touch the four `PartialFit::partial_fit` entries:
//!   - `impl PartialFit for SGDClassifier`        (`sgd.rs`, unfitted initial call)
//!   - `impl PartialFit for FittedSGDClassifier`  (`sgd.rs`, incremental call)
//!   - `impl PartialFit for SGDRegressor`         (`sgd.rs`, unfitted initial call)
//!   - `impl PartialFit for FittedSGDRegressor`   (`sgd.rs`, incremental call)
//!
//! sklearn `SGDClassifier.partial_fit` / `SGDRegressor.partial_fit` validate
//! their inputs through the SAME `_validate_data(...)` with the default
//! `force_all_finite=True` as `fit`:
//!   - classifier: `partial_fit` (`_stochastic_gradient.py:844`) →
//!     `_partial_fit` (`:581`) → `self._validate_data(X, y, accept_sparse="csr",
//!     ...)` (`:596`, force_all_finite defaults True).
//!   - regressor: `partial_fit` (`_stochastic_gradient.py:1523`) →
//!     `_partial_fit` (`:1462`) → `self._validate_data(X, y, ...)` (`:1476`).
//!
//! So a NaN / +/-inf in X (or y for the regressor) raises
//! `ValueError("Input X contains NaN.")` / `"Input y contains NaN."` BEFORE any
//! kernel step.
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, never
//! copied from ferrolearn):
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.linear_model import SGDClassifier, SGDRegressor
//! Xg = np.array([[1.,1.],[2.,2.],[3.,1.],[5.,5.],[6.,5.],[7.,6.]])
//! yc = np.array([0,0,0,1,1,1]); yr = np.array([1.,2.,3.,4.,5.,6.])
//! Xn = Xg.copy(); Xn[0,0]=np.nan
//! SGDClassifier().partial_fit(Xn, yc, classes=[0,1]) # -> ValueError: Input X contains NaN.
//! SGDRegressor().partial_fit(Xn, yr)                 # -> ValueError: Input X contains NaN.
//! yn = yr.copy(); yn[0]=np.nan
//! SGDRegressor().partial_fit(Xg, yn)                 # -> ValueError: Input y contains NaN.
//! "
//! ```
//!
//! Oracle result (sklearn 1.5.2, every case raises):
//! ```text
//! SGDClassifier.partial_fit NaN X   ValueError: Input X contains NaN.
//! SGDRegressor.partial_fit  NaN X   ValueError: Input X contains NaN.
//! SGDRegressor.partial_fit  NaN y   ValueError: Input y contains NaN.
//! ```
//!
//! ferrolearn ACTUAL (current main): `SGDClassifier::partial_fit` /
//! `SGDRegressor::partial_fit` (and their `FittedSGD*` incremental counterparts)
//! call `validate_clf_params` / `validate_reg_params` (which check only
//! shape/eta0/alpha/l1_ratio/validation_fraction — NO finiteness) or no
//! parameter validation at all, then run the SGD kernel on NaN/Inf data,
//! returning `Ok(_)` with NaN `coef_` / `intercept_` instead of `Err`.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::{Fit, PartialFit};
use ferrolearn_linear::{SGDClassifier, SGDRegressor};
use ndarray::{Array1, Array2, array};

fn finite_x() -> Array2<f64> {
    array![
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [7.0, 6.0]
    ]
}

fn finite_yc() -> Array1<usize> {
    array![0usize, 0, 0, 1, 1, 1]
}

fn finite_yr() -> Array1<f64> {
    array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
}

/// `true` iff the error is the non-finite `InvalidParameter` for `X` or `y`
/// (the sklearn `ValueError` analog used by the batch-4 `Fit` guards).
fn is_nonfinite_xy_err(res: &Result<impl Sized, FerroError>) -> bool {
    matches!(
        res,
        Err(FerroError::InvalidParameter { name, reason })
            if (name == "X" || name == "y") && reason.contains("NaN or infinity")
    )
}

fn nan_in_x() -> Array2<f64> {
    let mut x = finite_x();
    x[[0, 0]] = f64::NAN;
    x
}

fn pinf_in_x() -> Array2<f64> {
    let mut x = finite_x();
    x[[1, 1]] = f64::INFINITY;
    x
}

// ---------------------------------------------------------------------------
// Divergence: SGDClassifier::partial_fit (unfitted initial call)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `<SGDClassifier as PartialFit>::partial_fit`
/// diverges from sklearn `SGDClassifier.partial_fit`
/// (`_stochastic_gradient.py:844` → `_partial_fit` `:581` →
/// `_validate_data(... force_all_finite=True)` `:596`) for NaN/+inf in X.
/// sklearn raises `ValueError: Input X contains NaN.`; ferrolearn returns
/// `Ok(FittedSGDClassifier)` (the guard lives only on the `Fit::fit` entry).
/// Tracking: #2264
#[test]
fn sgd_classifier_partial_fit_rejects_non_finite_x() {
    for x in [nan_in_x(), pinf_in_x()] {
        let res = SGDClassifier::<f64>::new().partial_fit(&x, &finite_yc());
        assert!(
            is_nonfinite_xy_err(&res),
            "SGDClassifier::partial_fit must reject non-finite X like sklearn"
        );
    }
}

// ---------------------------------------------------------------------------
// Divergence: FittedSGDClassifier::partial_fit (incremental call)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `<FittedSGDClassifier as PartialFit>::partial_fit`
/// diverges from sklearn `SGDClassifier.partial_fit`
/// (`_stochastic_gradient.py:844` → `_partial_fit` `:581` →
/// `_validate_data` `:596`) for NaN in X on an incremental update.
/// sklearn raises `ValueError: Input X contains NaN.`; ferrolearn returns `Ok`.
/// Tracking: #2264
#[test]
fn sgd_classifier_incremental_partial_fit_rejects_non_finite_x() {
    let fitted = SGDClassifier::<f64>::new()
        .fit(&finite_x(), &finite_yc())
        .expect("finite initial fit");
    let res = fitted.partial_fit(&nan_in_x(), &finite_yc());
    assert!(
        is_nonfinite_xy_err(&res),
        "FittedSGDClassifier::partial_fit must reject non-finite X like sklearn"
    );
}

// ---------------------------------------------------------------------------
// Divergence: SGDRegressor::partial_fit (unfitted initial call), X and y
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `<SGDRegressor as PartialFit>::partial_fit`
/// diverges from sklearn `SGDRegressor.partial_fit`
/// (`_stochastic_gradient.py:1523` → `_partial_fit` `:1462` →
/// `_validate_data(... force_all_finite=True)` `:1476`) for NaN/+inf in X.
/// sklearn raises `ValueError: Input X contains NaN.`; ferrolearn returns `Ok`.
/// Tracking: #2264
#[test]
fn sgd_regressor_partial_fit_rejects_non_finite_x() {
    for x in [nan_in_x(), pinf_in_x()] {
        let res = SGDRegressor::<f64>::new().partial_fit(&x, &finite_yr());
        assert!(
            is_nonfinite_xy_err(&res),
            "SGDRegressor::partial_fit must reject non-finite X like sklearn"
        );
    }
}

/// Divergence: ferrolearn's `<SGDRegressor as PartialFit>::partial_fit`
/// diverges from sklearn `SGDRegressor.partial_fit`
/// (`_stochastic_gradient.py:1523` → `_partial_fit` `:1462` →
/// `_validate_data` `:1476`) for NaN in the float target y.
/// sklearn raises `ValueError: Input y contains NaN.`; ferrolearn returns `Ok`.
/// Tracking: #2264
#[test]
fn sgd_regressor_partial_fit_rejects_non_finite_y() {
    let mut y = finite_yr();
    y[0] = f64::NAN;
    let res = SGDRegressor::<f64>::new().partial_fit(&finite_x(), &y);
    assert!(
        is_nonfinite_xy_err(&res),
        "SGDRegressor::partial_fit must reject non-finite y like sklearn"
    );
}

// ---------------------------------------------------------------------------
// Divergence: FittedSGDRegressor::partial_fit (incremental call)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `<FittedSGDRegressor as PartialFit>::partial_fit`
/// diverges from sklearn `SGDRegressor.partial_fit`
/// (`_stochastic_gradient.py:1523` → `_partial_fit` `:1462` →
/// `_validate_data` `:1476`) for NaN in X on an incremental update.
/// sklearn raises `ValueError: Input X contains NaN.`; ferrolearn returns `Ok`.
/// Tracking: #2264
#[test]
fn sgd_regressor_incremental_partial_fit_rejects_non_finite_x() {
    let fitted = SGDRegressor::<f64>::new()
        .fit(&finite_x(), &finite_yr())
        .expect("finite initial fit");
    let res = fitted.partial_fit(&nan_in_x(), &finite_yr());
    assert!(
        is_nonfinite_xy_err(&res),
        "FittedSGDRegressor::partial_fit must reject non-finite X like sklearn"
    );
}
