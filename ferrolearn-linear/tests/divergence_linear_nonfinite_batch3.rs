//! Non-finite input validation parity (batch 3 of the #2261 sweep) for the
//! remaining `ferrolearn-linear` estimators: `BayesianRidge`, `ARDRegression`,
//! the GLM family (`PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`),
//! and `RidgeClassifier`. Continues #2256 (batch 1) / #2259 (batch 2).
//!
//! scikit-learn 1.5.2 validates `X`/`y` at fit through `_validate_data(...)`
//! with the default `force_all_finite=True`:
//!   - `BayesianRidge.fit`     → `sklearn/linear_model/_bayes.py:238-239`
//!     (also `sample_weight` via `_check_sample_weight`, `_bayes.py:244`)
//!   - `ARDRegression.fit`     → `sklearn/linear_model/_bayes.py:624-631`
//!     (no `sample_weight` — `fit(X, y)`, `_bayes.py:606`)
//!   - `PoissonRegressor` / `GammaRegressor` / `TweedieRegressor` `.fit`
//!     → `sklearn/linear_model/_glm/glm.py:189-196` (also `sample_weight` via
//!     `_check_sample_weight`, `glm.py:211`); all three route through the
//!     shared `_GeneralizedLinearRegressor.fit`.
//!   - `RidgeClassifier.fit`   → `sklearn/linear_model/_ridge.py:1291-1298`
//!     (`_prepare_data`; also `sample_weight` via `_check_sample_weight`,
//!     `_ridge.py:1305`). `y` is the class label (binarized), not a numeric
//!     target, so only `X` (and `sample_weight`) carry the finiteness check.
//!
//! `check_array` raises `ValueError` on any NaN or +/-inf in `X` or `y` (and on
//! a non-finite `sample_weight`) BEFORE the solver runs. Previously ferrolearn
//! accepted non-finite input and produced NaN `coef_`; these estimators now
//! reject it with `FerroError::InvalidParameter`, matching sklearn's
//! reject-at-fit contract (R-DEV-1 / R-DEV-2 exception parity).
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, NOT
//! copied from ferrolearn). Confirmed every non-finite case raises `ValueError`
//! and every finite case fits, for all six estimators (plus non-finite
//! `sample_weight` for BayesianRidge / the GLM family / RidgeClassifier):
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.linear_model import (BayesianRidge, ARDRegression,
//!     PoissonRegressor, GammaRegressor, TweedieRegressor, RidgeClassifier)
//! X=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[2.,1.],[4.,3.]])
//! y=np.array([1.,2.,3.,4.,1.5,2.5]); yc=np.array([0,0,1,1,0,1])
//! # each non-finite X/y -> ValueError; finite -> fits.
//! "
//! ```
//!
//! Oracle result (abbreviated): every estimator raises
//! `ValueError: Input X contains NaN.` / `... contains infinity ...` for NaN /
//! +inf / -inf in X; `Input y contains NaN.` / `... infinity ...` for NaN / inf
//! in y (the float-y regressors); `Input sample_weight contains NaN / infinity`
//! for a non-finite weight; and fits cleanly on the finite design.
//!
//! Known-fit oracle coefficients (finite, no-regression sanity, R-CHAR-3):
//! ```text
//! BayesianRidge()             coef_=[0.49999942, 4.4e-07]  intercept_=0.50000035
//! ARDRegression()             coef_=[0.49999997, 0.0]      intercept_=0.50000013
//! PoissonRegressor()          coef_=[0.14342527, 0.04106484] intercept_=0.08399124
//! GammaRegressor()            coef_=[0.12558468, 0.05607879] intercept_=0.07550799
//! TweedieRegressor(power=1.5) coef_=[0.13460467, 0.04950582] intercept_=0.07619382
//! RidgeClassifier()           coef_=[0.51233397, -0.09487666] intercept_=-1.49905123
//! ```

use ferrolearn_core::Fit;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_linear::{
    ARDRegression, BayesianRidge, GammaRegressor, PoissonRegressor, RidgeClassifier,
    TweedieRegressor,
};
use ndarray::{Array1, Array2, array};

/// Shared finite design: full-rank, well-conditioned, all-positive `y` so the
/// log-link GLMs (Gamma needs `y > 0`) fit cleanly under every estimator.
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

/// The five non-finite `(X, y)` perturbations the oracle confirms raise:
/// NaN/+inf/-inf in X, NaN/inf in y.
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

/// The three non-finite `X` perturbations (for the `usize`-target classifier,
/// whose `y` is finite by type): NaN/+inf/-inf in X.
fn nonfinite_x_only_cases() -> Vec<(Array2<f64>, &'static str)> {
    let (x, _) = finite_xy();
    let mut x_nan = x.clone();
    x_nan[[0, 0]] = f64::NAN;
    let mut x_pinf = x.clone();
    x_pinf[[1, 1]] = f64::INFINITY;
    let mut x_ninf = x.clone();
    x_ninf[[2, 0]] = f64::NEG_INFINITY;
    vec![
        (x_nan, "NaN in X"),
        (x_pinf, "+inf in X"),
        (x_ninf, "-inf in X"),
    ]
}

// ---------------------------------------------------------------------------
// BayesianRidge (Fit<Array2, Array1> + fit_with_sample_weight)
// ---------------------------------------------------------------------------

#[test]
fn bayesian_ridge_rejects_non_finite_input_like_sklearn() {
    // Oracle: BayesianRidge().fit raises ValueError for NaN/+inf/-inf in X,
    // NaN/inf in y (`_bayes.py:238-239`, force_all_finite=True).
    let model = BayesianRidge::<f64>::new();
    for (x, y, tag) in nonfinite_xy_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "BayesianRidge: {tag} must be rejected, got {res:?}"
        );
    }
}

#[test]
fn bayesian_ridge_rejects_non_finite_sample_weight_like_sklearn() {
    // Oracle: BayesianRidge().fit(X, y, sample_weight=w) raises ValueError for a
    // non-finite weight (`_check_sample_weight`, `_bayes.py:244`).
    let (x, y) = finite_xy();
    let model = BayesianRidge::<f64>::new();

    let mut w_nan = Array1::<f64>::ones(6);
    w_nan[0] = f64::NAN;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, Some(&w_nan))),
        "BayesianRidge: NaN sample_weight must be rejected"
    );

    let mut w_inf = Array1::<f64>::ones(6);
    w_inf[1] = f64::INFINITY;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, Some(&w_inf))),
        "BayesianRidge: inf sample_weight must be rejected"
    );
}

#[test]
fn bayesian_ridge_finite_input_fits_matches_sklearn_oracle() {
    // No false positive + no regression: finite input fits and reproduces the
    // live sklearn 1.5.2 coefficients (R-CHAR-3).
    let (x, y) = finite_xy();
    let fitted = BayesianRidge::<f64>::new()
        .fit(&x, &y)
        .expect("finite BayesianRidge fit must succeed");
    // Oracle: BayesianRidge() coef_=[0.49999942, 4.4e-07], intercept_=0.50000035.
    let coef = fitted.coefficients();
    approx::assert_relative_eq!(coef[0], 0.499_999_42, epsilon = 1e-6);
    approx::assert_relative_eq!(coef[1], 0.0, epsilon = 1e-5);
    approx::assert_relative_eq!(fitted.intercept(), 0.500_000_35, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// ARDRegression (Fit<Array2, Array1>; no sample_weight)
// ---------------------------------------------------------------------------

#[test]
fn ard_rejects_non_finite_input_like_sklearn() {
    // Oracle: ARDRegression().fit raises ValueError for NaN/+inf/-inf in X,
    // NaN/inf in y (`_bayes.py:624-631`, force_all_finite=True). ARD has no
    // sample_weight (`_bayes.py:606`).
    let model = ARDRegression::<f64>::new();
    for (x, y, tag) in nonfinite_xy_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "ARDRegression: {tag} must be rejected, got {res:?}"
        );
    }
}

#[test]
fn ard_finite_input_fits_matches_sklearn_oracle() {
    let (x, y) = finite_xy();
    let fitted = ARDRegression::<f64>::new()
        .fit(&x, &y)
        .expect("finite ARDRegression fit must succeed");
    // Oracle: ARDRegression() coef_=[0.49999997, 0.0], intercept_=0.50000013.
    let coef = fitted.coefficients();
    approx::assert_relative_eq!(coef[0], 0.499_999_97, epsilon = 1e-6);
    approx::assert_relative_eq!(coef[1], 0.0, epsilon = 1e-5);
    approx::assert_relative_eq!(fitted.intercept(), 0.500_000_13, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// PoissonRegressor (Fit<Array2, Array1> + fit_with_sample_weight)
// ---------------------------------------------------------------------------

#[test]
fn poisson_rejects_non_finite_input_like_sklearn() {
    // Oracle: PoissonRegressor().fit raises ValueError for NaN/+inf/-inf in X,
    // NaN/inf in y (`glm.py:189-196`, force_all_finite=True). The finiteness
    // guard precedes the per-family y-domain check.
    let model = PoissonRegressor::<f64>::new();
    for (x, y, tag) in nonfinite_xy_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "PoissonRegressor: {tag} must be rejected, got {res:?}"
        );
    }
}

#[test]
fn poisson_rejects_non_finite_sample_weight_like_sklearn() {
    // Oracle: PoissonRegressor().fit(X, y, sample_weight=w) raises ValueError for
    // a non-finite weight (`_check_sample_weight`, `glm.py:211`).
    let (x, y) = finite_xy();
    let model = PoissonRegressor::<f64>::new();

    let mut w_nan = Array1::<f64>::ones(6);
    w_nan[0] = f64::NAN;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, &w_nan)),
        "PoissonRegressor: NaN sample_weight must be rejected"
    );

    let mut w_inf = Array1::<f64>::ones(6);
    w_inf[1] = f64::NEG_INFINITY;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, &w_inf)),
        "PoissonRegressor: -inf sample_weight must be rejected"
    );
}

#[test]
fn poisson_finite_input_fits_matches_sklearn_oracle() {
    let (x, y) = finite_xy();
    let fitted = PoissonRegressor::<f64>::new()
        .fit(&x, &y)
        .expect("finite PoissonRegressor fit must succeed");
    // Oracle: PoissonRegressor() coef_=[0.14342527, 0.04106484],
    // intercept_=0.08399124 (default alpha=1.0, log link). ferrolearn's IRLS
    // reaches the same convex optimum as sklearn's lbfgs to solver tolerance
    // (~1e-4, R-DEV-7), so a 1e-3 tolerance pins the no-regression sanity.
    let coef = fitted.coefficients();
    approx::assert_relative_eq!(coef[0], 0.143_425_27, epsilon = 1e-3);
    approx::assert_relative_eq!(coef[1], 0.041_064_84, epsilon = 1e-3);
    approx::assert_relative_eq!(fitted.intercept(), 0.083_991_24, epsilon = 1e-3);
}

// ---------------------------------------------------------------------------
// GammaRegressor (Fit<Array2, Array1> + fit_with_sample_weight; y > 0)
// ---------------------------------------------------------------------------

#[test]
fn gamma_rejects_non_finite_input_like_sklearn() {
    // Oracle: GammaRegressor().fit raises ValueError for NaN/+inf/-inf in X,
    // NaN/inf in y (`glm.py:189-196`, force_all_finite=True). All `y` here are
    // strictly positive (Gamma domain) except the injected non-finite entry,
    // which the finiteness guard rejects before the domain check.
    let model = GammaRegressor::<f64>::new();
    for (x, y, tag) in nonfinite_xy_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "GammaRegressor: {tag} must be rejected, got {res:?}"
        );
    }
}

#[test]
fn gamma_rejects_non_finite_sample_weight_like_sklearn() {
    let (x, y) = finite_xy();
    let model = GammaRegressor::<f64>::new();

    let mut w_nan = Array1::<f64>::ones(6);
    w_nan[2] = f64::NAN;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, &w_nan)),
        "GammaRegressor: NaN sample_weight must be rejected"
    );

    let mut w_inf = Array1::<f64>::ones(6);
    w_inf[3] = f64::INFINITY;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, &w_inf)),
        "GammaRegressor: inf sample_weight must be rejected"
    );
}

#[test]
fn gamma_finite_input_fits_matches_sklearn_oracle() {
    let (x, y) = finite_xy();
    let fitted = GammaRegressor::<f64>::new()
        .fit(&x, &y)
        .expect("finite GammaRegressor fit must succeed");
    // Oracle: GammaRegressor() coef_=[0.12558468, 0.05607879],
    // intercept_=0.07550799 (default alpha=1.0, log link, y > 0). IRLS reaches
    // the same convex optimum as sklearn's lbfgs to ~1e-4 (R-DEV-7).
    let coef = fitted.coefficients();
    approx::assert_relative_eq!(coef[0], 0.125_584_68, epsilon = 1e-3);
    approx::assert_relative_eq!(coef[1], 0.056_078_79, epsilon = 1e-3);
    approx::assert_relative_eq!(fitted.intercept(), 0.075_507_99, epsilon = 1e-3);
}

// ---------------------------------------------------------------------------
// TweedieRegressor (Fit<Array2, Array1> + fit_with_sample_weight; power=1.5)
// ---------------------------------------------------------------------------

#[test]
fn tweedie_rejects_non_finite_input_like_sklearn() {
    // Oracle: TweedieRegressor(power=1.5).fit raises ValueError for NaN/+inf/-inf
    // in X, NaN/inf in y (`glm.py:189-196`, force_all_finite=True).
    let model = TweedieRegressor::<f64>::new().with_power(1.5);
    for (x, y, tag) in nonfinite_xy_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "TweedieRegressor: {tag} must be rejected, got {res:?}"
        );
    }
}

#[test]
fn tweedie_rejects_non_finite_sample_weight_like_sklearn() {
    let (x, y) = finite_xy();
    let model = TweedieRegressor::<f64>::new().with_power(1.5);

    let mut w_nan = Array1::<f64>::ones(6);
    w_nan[4] = f64::NAN;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, &w_nan)),
        "TweedieRegressor: NaN sample_weight must be rejected"
    );

    let mut w_inf = Array1::<f64>::ones(6);
    w_inf[5] = f64::INFINITY;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &y, &w_inf)),
        "TweedieRegressor: inf sample_weight must be rejected"
    );
}

#[test]
fn tweedie_finite_input_fits_matches_sklearn_oracle() {
    let (x, y) = finite_xy();
    let fitted = TweedieRegressor::<f64>::new()
        .with_power(1.5)
        .fit(&x, &y)
        .expect("finite TweedieRegressor fit must succeed");
    // Oracle: TweedieRegressor(power=1.5) coef_=[0.13460467, 0.04950582],
    // intercept_=0.07619382 (default alpha=1.0, log link for power>0). IRLS
    // reaches the same convex optimum as sklearn's lbfgs to ~1e-4 (R-DEV-7).
    let coef = fitted.coefficients();
    approx::assert_relative_eq!(coef[0], 0.134_604_67, epsilon = 1e-3);
    approx::assert_relative_eq!(coef[1], 0.049_505_82, epsilon = 1e-3);
    approx::assert_relative_eq!(fitted.intercept(), 0.076_193_82, epsilon = 1e-3);
}

// ---------------------------------------------------------------------------
// RidgeClassifier (Fit<Array2, Array1<usize>> + fit_with_sample_weight)
// y is the class label (finite by type) — only X / sample_weight are checked.
// ---------------------------------------------------------------------------

#[test]
fn ridge_classifier_rejects_non_finite_x_like_sklearn() {
    // Oracle: RidgeClassifier().fit raises ValueError for NaN/+inf/-inf in X
    // (`_ridge.py:1291-1298`, force_all_finite=True). RidgeClassifier solves the
    // binarized indicator targets directly (NOT via the #2259-guarded Ridge::fit),
    // so the X guard is owned by RidgeClassifier::fit_with_sample_weight.
    let yc: Array1<usize> = array![0, 0, 1, 1, 0, 1];
    let model = RidgeClassifier::<f64>::new();
    for (x, tag) in nonfinite_x_only_cases() {
        let res = model.fit(&x, &yc);
        assert!(
            is_nonfinite_xy_err(&res),
            "RidgeClassifier: {tag} must be rejected, got {res:?}"
        );
    }
}

#[test]
fn ridge_classifier_rejects_non_finite_sample_weight_like_sklearn() {
    // Oracle: RidgeClassifier().fit(X, y, sample_weight=w) raises ValueError for
    // a non-finite weight (`_check_sample_weight`, `_ridge.py:1305`).
    let (x, _) = finite_xy();
    let yc: Array1<usize> = array![0, 0, 1, 1, 0, 1];
    let model = RidgeClassifier::<f64>::new();

    let mut w_nan = Array1::<f64>::ones(6);
    w_nan[0] = f64::NAN;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &yc, Some(&w_nan))),
        "RidgeClassifier: NaN sample_weight must be rejected"
    );

    let mut w_inf = Array1::<f64>::ones(6);
    w_inf[1] = f64::INFINITY;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &yc, Some(&w_inf))),
        "RidgeClassifier: inf sample_weight must be rejected"
    );
}

#[test]
fn ridge_classifier_finite_input_fits_matches_sklearn_oracle() {
    let (x, _) = finite_xy();
    let yc: Array1<usize> = array![0, 0, 1, 1, 0, 1];
    let fitted = RidgeClassifier::<f64>::new()
        .fit(&x, &yc)
        .expect("finite RidgeClassifier fit must succeed");
    // Oracle: RidgeClassifier() coef_=[0.51233397, -0.09487666],
    // intercept_=-1.49905123 (binary {-1,+1} indicator, alpha=1.0). coef_matrix
    // is (n_features, n_targets); binary => single column 0.
    let coef = fitted.coef_matrix();
    approx::assert_relative_eq!(coef[[0, 0]], 0.512_333_97, epsilon = 1e-6);
    approx::assert_relative_eq!(coef[[1, 0]], -0.094_876_66, epsilon = 1e-6);
    approx::assert_relative_eq!(fitted.intercept_vec()[0], -1.499_051_23, epsilon = 1e-6);
}
