//! Non-finite input validation parity (batch 2 of the #2259 sweep) for the
//! `ferrolearn-linear` estimators `Ridge`, `Lars`, `OrthogonalMatchingPursuit`
//! (OMP), and `HuberRegressor`.
//!
//! scikit-learn 1.5.2 validates `X`/`y` at fit through `_validate_data(...)`
//! with the default `force_all_finite=True`:
//!   - `Ridge.fit`                       → `sklearn/linear_model/_ridge.py:1242`
//!     (also `sample_weight` via `_check_sample_weight`, default
//!     `force_all_finite=True`)
//!   - `Lars.fit`                        → `sklearn/linear_model/_least_angle.py:1183`
//!   - `OrthogonalMatchingPursuit.fit`   → `sklearn/linear_model/_omp.py:772`
//!   - `HuberRegressor.fit`              → `sklearn/linear_model/_huber.py:297`
//!     (also `sample_weight` via `_check_sample_weight`, `_huber.py:306`)
//!
//! `check_array` raises `ValueError` on any NaN or +/-inf in `X` or `y` (and on
//! a non-finite `sample_weight` for Ridge/Huber) BEFORE the solver runs.
//! Previously ferrolearn accepted non-finite input and produced NaN `coef_`;
//! these estimators now reject it with `FerroError::InvalidParameter`, matching
//! sklearn's reject-at-fit contract (R-DEV-1 / R-DEV-2 exception parity).
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, NOT
//! copied from ferrolearn). The following script confirms every non-finite case
//! raises `ValueError` and every finite case fits, for all four estimators
//! (plus non-finite `sample_weight` for Ridge and Huber):
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.linear_model import Ridge, Lars, OrthogonalMatchingPursuit, HuberRegressor
//! X=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[2.,1.],[4.,3.]]); y=np.array([1.,2.,3.,4.,1.5,2.5])
//! w=np.array([1.,2.,1.,1.,1.,1.])
//! # each non-finite X/y -> ValueError; finite -> fits.
//! # Ridge/Huber: non-finite sample_weight -> ValueError too.
//! "
//! ```
//!
//! Oracle result (abbreviated):
//! ```text
//! <Estimator> Xnan  ValueError: Input X contains NaN.
//! <Estimator> Xinf  ValueError: Input X contains infinity or a value too large ...
//! <Estimator> Xninf ValueError: Input X contains infinity or a value too large ...
//! <Estimator> ynan  ValueError: Input y contains NaN.
//! <Estimator> yinf  ValueError: Input y contains infinity or a value too large ...
//! <Estimator> finite NO-RAISE
//! Ridge wnan/winf  ValueError: Input sample_weight contains NaN / infinity ...
//! Huber wnan/winf  ValueError: Input sample_weight contains NaN / infinity ...
//! ```
//!
//! Known-fit oracle coefficients (finite, no-regression sanity):
//! ```text
//! Ridge(alpha=1.0)               coef_=[0.4003795066, 0.0740037951] intercept_=0.569259962
//! Lars(n_nonzero_coefs=1)        coef_=[0.0, 0.1666666667]          intercept_=1.6666666667
//! OrthogonalMatchingPursuit(n=1) coef_=[0.0, 0.3823529412]          intercept_=0.8039215686
//! HuberRegressor(epsilon=1.35)   coef_=[0.5, 0.0]                   intercept_=0.5
//! ```

use ferrolearn_core::Fit;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_linear::{HuberRegressor, Lars, OrthogonalMatchingPursuit, Ridge};
use ndarray::{Array1, Array2, array};

/// Shared finite design: full-rank, well-conditioned, fits cleanly under every
/// estimator.
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

/// `true` iff the error is the non-finite `InvalidParameter` for `X` or `y`
/// (the sklearn `ValueError` analog).
fn is_nonfinite_xy_err(res: &Result<impl Sized, FerroError>) -> bool {
    matches!(
        res,
        Err(FerroError::InvalidParameter { name, reason })
            if (name == "X" || name == "y") && reason.contains("NaN or infinity")
    )
}

/// `true` iff the error is the non-finite `InvalidParameter` for `sample_weight`.
fn is_nonfinite_sw_err(res: &Result<impl Sized, FerroError>) -> bool {
    matches!(
        res,
        Err(FerroError::InvalidParameter { name, reason })
            if name == "sample_weight" && reason.contains("NaN or infinity")
    )
}

/// Build the five non-finite `(X, y)` perturbations: NaN/+inf/-inf in X, NaN/inf
/// in y — exactly the cases the oracle confirms raise.
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

// ---------------------------------------------------------------------------
// Ridge (single-output Fit<Array2, Array1>) + sample_weight
// ---------------------------------------------------------------------------

#[test]
fn ridge_rejects_non_finite_input_like_sklearn() {
    // Oracle: Ridge().fit raises ValueError for NaN/+inf/-inf in X, NaN/inf in y
    // (`_ridge.py:1242`, force_all_finite=True). ferrolearn must reject all five.
    let model = Ridge::<f64>::new().with_alpha(1.0);
    for (x, y, tag) in nonfinite_xy_cases() {
        let res = model.fit(&x, &y);
        assert!(is_nonfinite_xy_err(&res), "Ridge: {tag} must be rejected");
    }
}

#[test]
fn ridge_rejects_non_finite_sample_weight_like_sklearn() {
    // Oracle: Ridge().fit(X, y, sample_weight=w) raises ValueError for a
    // non-finite weight (`_check_sample_weight`, default force_all_finite=True).
    let (x, y) = finite_xy();
    let model = Ridge::<f64>::new().with_alpha(1.0);

    let mut w_nan = Array1::<f64>::ones(6);
    w_nan[0] = f64::NAN;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, Some(&w_nan))),
        "Ridge: NaN sample_weight must be rejected"
    );

    let mut w_inf = Array1::<f64>::ones(6);
    w_inf[1] = f64::INFINITY;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, Some(&w_inf))),
        "Ridge: inf sample_weight must be rejected"
    );
}

#[test]
fn ridge_multioutput_rejects_non_finite_input() {
    // The SEPARATE multi-output arm Fit<Array2, Array2>::fit does not delegate to
    // fit_with_sample_weight; it must enforce the same reject-at-fit contract
    // (sklearn `_ridge.py:1242`, multi_output=True).
    let (x, _) = finite_xy();
    let y2: Array2<f64> = array![
        [1.0, 0.5],
        [2.0, 1.5],
        [3.0, 2.5],
        [4.0, 3.5],
        [1.5, 1.0],
        [2.5, 2.0]
    ];
    let model = Ridge::<f64>::new().with_alpha(1.0);

    let mut x_nan = x.clone();
    x_nan[[0, 0]] = f64::NAN;
    assert!(
        is_nonfinite_xy_err(&Fit::<Array2<f64>, Array2<f64>>::fit(&model, &x_nan, &y2)),
        "Ridge multi-output: NaN in X must be rejected"
    );

    let mut y_inf = y2.clone();
    y_inf[[2, 1]] = f64::INFINITY;
    assert!(
        is_nonfinite_xy_err(&Fit::<Array2<f64>, Array2<f64>>::fit(&model, &x, &y_inf)),
        "Ridge multi-output: inf in Y must be rejected"
    );
}

#[test]
fn ridge_finite_input_fits_matches_sklearn_oracle() {
    // No false positive + no regression: finite input fits and reproduces the
    // live sklearn 1.5.2 coefficients (R-CHAR-3).
    let (x, y) = finite_xy();
    let fitted = Ridge::<f64>::new()
        .with_alpha(1.0)
        .fit(&x, &y)
        .expect("finite Ridge fit must succeed");
    // Oracle: Ridge(alpha=1.0) coef_=[0.4003795066, 0.0740037951],
    // intercept_=0.569259962.
    let coef = fitted.coefficients();
    approx::assert_relative_eq!(coef[0], 0.400_379_506_6, epsilon = 1e-7);
    approx::assert_relative_eq!(coef[1], 0.074_003_795_1, epsilon = 1e-7);
    approx::assert_relative_eq!(fitted.intercept(), 0.569_259_962, epsilon = 1e-7);
}

// ---------------------------------------------------------------------------
// Lars
// ---------------------------------------------------------------------------

#[test]
fn lars_rejects_non_finite_input_like_sklearn() {
    // Oracle: Lars().fit raises ValueError for NaN/+inf/-inf in X, NaN/inf in y
    // (`_least_angle.py:1183`, force_all_finite=True).
    let model = Lars::<f64>::new().with_n_nonzero_coefs(1);
    for (x, y, tag) in nonfinite_xy_cases() {
        let res = model.fit(&x, &y);
        assert!(is_nonfinite_xy_err(&res), "Lars: {tag} must be rejected");
    }
}

#[test]
fn lars_finite_input_fits_matches_sklearn_oracle() {
    let (x, y) = finite_xy();
    let fitted = Lars::<f64>::new()
        .with_n_nonzero_coefs(1)
        .fit(&x, &y)
        .expect("finite Lars fit must succeed");
    // Oracle: Lars(n_nonzero_coefs=1) coef_=[0.0, 0.1666666667],
    // intercept_=1.6666666667.
    let coef = fitted.coefficients();
    approx::assert_relative_eq!(coef[0], 0.0, epsilon = 1e-6);
    approx::assert_relative_eq!(coef[1], 0.166_666_666_7, epsilon = 1e-6);
    approx::assert_relative_eq!(fitted.intercept(), 1.666_666_666_7, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// OrthogonalMatchingPursuit (OMP)
// ---------------------------------------------------------------------------

#[test]
fn omp_rejects_non_finite_input_like_sklearn() {
    // Oracle: OrthogonalMatchingPursuit().fit raises ValueError for NaN/+inf/-inf
    // in X, NaN/inf in y (`_omp.py:772`, force_all_finite=True).
    let model = OrthogonalMatchingPursuit::<f64>::new().with_n_nonzero_coefs(1);
    for (x, y, tag) in nonfinite_xy_cases() {
        let res = model.fit(&x, &y);
        assert!(is_nonfinite_xy_err(&res), "OMP: {tag} must be rejected");
    }
}

#[test]
fn omp_finite_input_fits_matches_sklearn_oracle() {
    let (x, y) = finite_xy();
    let fitted = OrthogonalMatchingPursuit::<f64>::new()
        .with_n_nonzero_coefs(1)
        .fit(&x, &y)
        .expect("finite OMP fit must succeed");
    // Oracle: OrthogonalMatchingPursuit(n_nonzero_coefs=1)
    // coef_=[0.0, 0.3823529412], intercept_=0.8039215686.
    let coef = fitted.coefficients();
    approx::assert_relative_eq!(coef[0], 0.0, epsilon = 1e-9);
    approx::assert_relative_eq!(coef[1], 0.382_352_941_2, epsilon = 1e-9);
    approx::assert_relative_eq!(fitted.intercept(), 0.803_921_568_6, epsilon = 1e-9);
}

// ---------------------------------------------------------------------------
// HuberRegressor + sample_weight
// ---------------------------------------------------------------------------

#[test]
fn huber_rejects_non_finite_input_like_sklearn() {
    // Oracle: HuberRegressor().fit raises ValueError for NaN/+inf/-inf in X,
    // NaN/inf in y (`_huber.py:297`, force_all_finite=True).
    let model = HuberRegressor::<f64>::new();
    for (x, y, tag) in nonfinite_xy_cases() {
        let res = model.fit(&x, &y);
        assert!(is_nonfinite_xy_err(&res), "Huber: {tag} must be rejected");
    }
}

#[test]
fn huber_rejects_non_finite_sample_weight_like_sklearn() {
    // Oracle: HuberRegressor().fit(X, y, sample_weight=w) raises ValueError for a
    // non-finite weight (`_check_sample_weight`, `_huber.py:306`, default
    // force_all_finite=True).
    let (x, y) = finite_xy();
    let model = HuberRegressor::<f64>::new();

    let mut w_nan = Array1::<f64>::ones(6);
    w_nan[0] = f64::NAN;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, Some(&w_nan))),
        "Huber: NaN sample_weight must be rejected"
    );

    let mut w_inf = Array1::<f64>::ones(6);
    w_inf[1] = f64::NEG_INFINITY;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, Some(&w_inf))),
        "Huber: -inf sample_weight must be rejected"
    );
}

#[test]
fn huber_finite_input_fits_matches_sklearn_oracle() {
    let (x, y) = finite_xy();
    let fitted = HuberRegressor::<f64>::new()
        .fit(&x, &y)
        .expect("finite Huber fit must succeed");
    // Oracle: HuberRegressor(epsilon=1.35) coef_=[0.5, 0.0], intercept_=0.5.
    // (in-repo L-BFGS reaches the same convex minimum; loose tol per R-DEV-7.)
    let coef = fitted.coefficients();
    approx::assert_relative_eq!(coef[0], 0.5, epsilon = 1e-3);
    approx::assert_relative_eq!(coef[1], 0.0, epsilon = 1e-3);
    approx::assert_relative_eq!(fitted.intercept(), 0.5, epsilon = 1e-3);
}
