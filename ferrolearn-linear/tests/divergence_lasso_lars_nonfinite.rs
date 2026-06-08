//! Divergence: `LassoLars` non-finite input validation gap (batch-2 #2259 sweep).
//!
//! The #2259 batch-2 sweep added `force_all_finite`-equivalent guards to
//! `Lars::fit` (`lars.rs`), but the SEPARATE `LassoLars::fit` impl in the SAME
//! file (`lars.rs:656`) was left untouched. `LassoLars::fit` calls only
//! `validate_input` (shape/sample-count, `lars.rs:250`) and then runs the LARS
//! homotopy path — it never rejects NaN/+/-inf in X or y.
//!
//! scikit-learn 1.5.2 `LassoLars` extends `Lars` but OVERRIDES `fit`
//! (`sklearn/linear_model/_least_angle.py:1698`), which calls
//! `self._validate_data(X, y, force_writeable=True, y_numeric=True)`
//! (`_least_angle.py:1726`) with the default `force_all_finite=True`, so
//! `check_array` raises `ValueError` on any non-finite X/y BEFORE the path runs.
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, NOT
//! copied from ferrolearn):
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.linear_model import LassoLars
//! X=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[2.,1.],[4.,3.]]); y=np.array([1.,2.,3.,4.,1.5,2.5])
//! # NaN/+inf/-inf in X, NaN/inf in y -> ValueError, e.g.:
//! #   LassoLars Xnan  ValueError: Input X contains NaN.
//! #   LassoLars Xpinf ValueError: Input X contains infinity or a value too large ...
//! #   LassoLars ynan  ValueError: Input y contains NaN.
//! "
//! ```
//! Oracle result: every non-finite case raises `ValueError`; the finite fit is
//! `coef_=[0.45909091, 0.01363636]`, `intercept_=0.5954545455` (alpha=0.1).
//!
//! ferrolearn `LassoLars::fit` returns `Ok(..)` with NaN coefficients for these
//! inputs instead of rejecting — a reject-at-fit contract divergence
//! (R-DEV-1 / R-DEV-2 exception parity).
//!
//! Tracking: #2260.

use ferrolearn_core::Fit;
use ferrolearn_core::error::FerroError;
use ferrolearn_linear::LassoLars;
use ndarray::{Array1, Array2, array};

fn finite_xy() -> (Array2<f64>, Array1<f64>) {
    let x: Array2<f64> = array![
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [2.0, 1.0],
        [4.0, 3.0]
    ];
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 1.5, 2.5];
    (x, y)
}

/// `true` iff the error is a non-finite `InvalidParameter` for `X` or `y` (the
/// sklearn `ValueError` analog the batch-2 sweep emits for the sibling
/// estimators).
fn is_nonfinite_xy_err(res: &Result<impl Sized, FerroError>) -> bool {
    matches!(
        res,
        Err(FerroError::InvalidParameter { name, reason })
            if (name == "X" || name == "y") && reason.contains("NaN or infinity")
    )
}

fn nonfinite_xy_cases() -> Vec<(Array2<f64>, Array1<f64>, &'static str)> {
    let (x, y) = finite_xy();
    let mut x_nan = x.clone();
    x_nan[[0, 0]] = f64::NAN;
    let mut x_pinf = x.clone();
    x_pinf[[1, 1]] = f64::INFINITY;
    let mut x_ninf = x.clone();
    x_ninf[[2, 0]] = f64::NEG_INFINITY;
    let mut y_nan = y.clone();
    y_nan[0] = f64::NAN;
    let mut y_inf = y.clone();
    y_inf[1] = f64::INFINITY;
    vec![
        (x_nan, y.clone(), "NaN in X"),
        (x_pinf, y.clone(), "+inf in X"),
        (x_ninf, y.clone(), "-inf in X"),
        (x.clone(), y_nan, "NaN in y"),
        (x, y_inf, "inf in y"),
    ]
}

/// Divergence: ferrolearn `LassoLars::fit` (`lars.rs:656`) accepts non-finite
/// X/y, while sklearn `LassoLars.fit` (`_least_angle.py:1698`, validated at
/// `_least_angle.py:1726` with `force_all_finite=True`) raises `ValueError`.
/// sklearn raises for all five cases; ferrolearn returns `Ok` (NaN coef).
/// Tracking: #2260.
#[test]
fn lasso_lars_rejects_non_finite_input_like_sklearn() {
    let model = LassoLars::<f64>::new().with_alpha(0.1);
    for (x, y, tag) in nonfinite_xy_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "LassoLars: {tag} must be rejected like sklearn (got {res:?})"
        );
    }
}
