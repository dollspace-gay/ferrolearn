//! Spillover gaps in the #2256 non-finite validation sweep for
//! `LinearRegression`. The sweep added a finite-check at the entry of
//! `LinearRegression::fit_with_sample_weight` (the 1-D `Fit<Array2,Array1>`
//! path) and at `Lasso`/`ElasticNet` `Fit::fit`, but it does NOT cover two
//! observable scikit-learn-validated inputs:
//!
//!   1. The SEPARATE multi-output `Fit<Array2<F>, Array2<F>>` arm
//!      (`linear_regression.rs`, the `FittedMultiOutputLinearRegression`
//!      producer) has its own fit body that routes around
//!      `fit_with_sample_weight` and therefore has NO finite-check — it
//!      centers and solves on non-finite X/Y and returns `Ok` with NaN coef.
//!      sklearn validates regardless of output dimensionality:
//!      `LinearRegression.fit` calls `self._validate_data(X, y, ...,
//!      multi_output=True, ...)` with the default `force_all_finite=True`
//!      (`sklearn/linear_model/_base.py:609`), so `check_array` raises
//!      `ValueError` on any NaN/+-inf in X OR the 2-D Y BEFORE the solve.
//!
//!   2. `sample_weight` is never checked for finiteness. The #2256 guard only
//!      runs `x.iter().any(!is_finite)` / `y.iter().any(!is_finite)`; a
//!      non-finite `sample_weight` flows straight into the weighted centering
//!      and √w rescaling and returns `Ok`. sklearn validates sample_weight via
//!      `_check_sample_weight` → `check_array(..., input_name="sample_weight")`
//!      (`sklearn/utils/validation.py:2050`), default `force_all_finite=True`,
//!      so a NaN/inf weight raises `ValueError`.
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, NOT
//! copied from ferrolearn):
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.linear_model import LinearRegression
//! X=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
//! Y2=np.array([[1.,1.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([1.,2.,3.,4.])
//! Xn=X.copy(); Xn[0,0]=np.nan
//! LinearRegression().fit(Xn, Y2)            # ValueError: Input X contains NaN.
//! Yn=Y2.copy(); Yn[1,0]=np.nan
//! LinearRegression().fit(X, Yn)             # ValueError: Input y contains NaN.
//! Yi=Y2.copy(); Yi[2,1]=np.inf
//! LinearRegression().fit(X, Yi)             # ValueError: Input y contains infinity ...
//! LinearRegression().fit(X, y, sample_weight=[1,1,1,np.nan]) # ValueError: Input sample_weight contains NaN.
//! LinearRegression().fit(X, y, sample_weight=[1,1,1,np.inf]) # ValueError: Input sample_weight contains infinity ...
//! "
//! ```
//! Result: every line above raises `ValueError`; ferrolearn returns `Ok`.

use ferrolearn_core::Fit;
use ferrolearn_core::error::FerroError;
use ferrolearn_linear::LinearRegression;
use ndarray::{Array1, Array2, array};

fn finite_x() -> Array2<f64> {
    array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
}

fn finite_y2() -> Array2<f64> {
    array![[1.0, 1.0], [2.0, 1.0], [3.0, 2.0], [4.0, 2.0]]
}

// ---------------------------------------------------------------------------
// Gap 1: multi-output `Fit<Array2, Array2>` arm accepts non-finite.
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's multi-output `LinearRegression::fit`
/// (`Fit<Array2<f64>, Array2<f64>>` in `linear_regression.rs`, the
/// `FittedMultiOutputLinearRegression` arm at the `fn fit` ~line 537) diverges
/// from `sklearn/linear_model/_base.py:609`
/// (`self._validate_data(..., multi_output=True, ...)`, `force_all_finite=True`)
/// for NaN in X with a 2-D target.
/// sklearn raises `ValueError: Input X contains NaN.`; ferrolearn returns
/// `Ok(..)` (NaN coef) — the entry finite-check lives only in
/// `fit_with_sample_weight`, which this arm does not call.
/// Tracking: #2257
#[test]
fn linreg_multioutput_rejects_nan_in_x_like_sklearn() {
    let mut x = finite_x();
    x[[0, 0]] = f64::NAN;
    let y2 = finite_y2();
    let res: Result<_, FerroError> =
        Fit::<Array2<f64>, Array2<f64>>::fit(&LinearRegression::<f64>::new(), &x, &y2);
    assert!(
        matches!(res, Err(FerroError::InvalidParameter { .. })),
        "sklearn raises ValueError on NaN in X for the 2-D multi-output fit; \
         ferrolearn returned {res:?}"
    );
}

/// Divergence: same multi-output arm vs `sklearn/linear_model/_base.py:609`
/// for NaN in the 2-D target Y. sklearn raises
/// `ValueError: Input y contains NaN.`; ferrolearn returns `Ok(..)`.
/// Tracking: #2257
#[test]
fn linreg_multioutput_rejects_nan_in_y2_like_sklearn() {
    let x = finite_x();
    let mut y2 = finite_y2();
    y2[[1, 0]] = f64::NAN;
    let res: Result<_, FerroError> =
        Fit::<Array2<f64>, Array2<f64>>::fit(&LinearRegression::<f64>::new(), &x, &y2);
    assert!(
        matches!(res, Err(FerroError::InvalidParameter { .. })),
        "sklearn raises ValueError on NaN in the 2-D Y; ferrolearn returned {res:?}"
    );
}

/// Divergence: same multi-output arm vs `sklearn/linear_model/_base.py:609`
/// for +inf in the 2-D target Y. sklearn raises
/// `ValueError: Input y contains infinity ...`; ferrolearn returns `Ok(..)`.
/// Tracking: #2257
#[test]
fn linreg_multioutput_rejects_inf_in_y2_like_sklearn() {
    let x = finite_x();
    let mut y2 = finite_y2();
    y2[[2, 1]] = f64::INFINITY;
    let res: Result<_, FerroError> =
        Fit::<Array2<f64>, Array2<f64>>::fit(&LinearRegression::<f64>::new(), &x, &y2);
    assert!(
        matches!(res, Err(FerroError::InvalidParameter { .. })),
        "sklearn raises ValueError on +inf in the 2-D Y; ferrolearn returned {res:?}"
    );
}

// ---------------------------------------------------------------------------
// Gap 2: sample_weight non-finite is not validated.
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `LinearRegression::fit_with_sample_weight`
/// (`linear_regression.rs`) diverges from `_check_sample_weight` →
/// `check_array(..., input_name="sample_weight")`
/// (`sklearn/utils/validation.py:2050`, `force_all_finite=True`) for a
/// `sample_weight` containing NaN. sklearn raises
/// `ValueError: Input sample_weight contains NaN.`; ferrolearn returns
/// `Ok(..)` — the #2256 guard checks X and y only, never `sample_weight`.
/// Tracking: #2258
#[test]
fn linreg_rejects_nan_sample_weight_like_sklearn() {
    let x = finite_x();
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
    let w: Array1<f64> = array![1.0, 1.0, 1.0, f64::NAN];
    let res = LinearRegression::<f64>::new().fit_with_sample_weight(&x, &y, Some(&w));
    assert!(
        matches!(res, Err(FerroError::InvalidParameter { .. })),
        "sklearn raises ValueError on NaN in sample_weight; ferrolearn returned {res:?}"
    );
}

/// Divergence: same sample_weight path vs `sklearn/utils/validation.py:2050`
/// for an inf weight. sklearn raises
/// `ValueError: Input sample_weight contains infinity ...`; ferrolearn returns
/// `Ok(..)`.
/// Tracking: #2258
#[test]
fn linreg_rejects_inf_sample_weight_like_sklearn() {
    let x = finite_x();
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
    let w: Array1<f64> = array![1.0, 1.0, 1.0, f64::INFINITY];
    let res = LinearRegression::<f64>::new().fit_with_sample_weight(&x, &y, Some(&w));
    assert!(
        matches!(res, Err(FerroError::InvalidParameter { .. })),
        "sklearn raises ValueError on inf in sample_weight; ferrolearn returned {res:?}"
    );
}
