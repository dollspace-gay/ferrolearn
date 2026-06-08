//! Non-finite input validation parity (batch 5 of the #2265 sweep) for the
//! `ferrolearn-linear` cross-validated estimators `RidgeCV`, `LassoCV`,
//! `ElasticNetCV`, and `LogisticRegressionCV`.
//!
//! scikit-learn 1.5.2 validates `X`/`y` at fit through `_validate_data(...)`
//! keeping the default `force_all_finite=True`, so any NaN or +/-inf in `X`
//! (or float `y`) raises `ValueError` BEFORE the alpha/C grid is built or the
//! CV folds are split — NOT deep inside a per-fold fit:
//!   - `RidgeCV.fit` → `_RidgeGCV.fit` `_validate_data` (`_ridge.py:2087`,
//!     `cv=None` default); the `cv=Some(k)` path routes `GridSearchCV` →
//!     `Ridge.fit` `_validate_data` (`_ridge.py:1242`).
//!   - `LassoCV.fit` / `ElasticNetCV.fit` → `LinearModelCV.fit` `_validate_data`
//!     (`_coordinate_descent.py:1619`/`:1644`; `check_X_params`/`check_y_params`
//!     do not set `force_all_finite=False`, so the default `True` applies).
//!   - `LogisticRegressionCV.fit` → `_validate_data` (`_logistic.py:1868`); `y`
//!     is class labels (integer), so only `X` is finiteness-checked.
//!
//! Previously these CV estimators ran the alpha-grid / k-fold path on non-finite
//! input (producing NaN selections / `coef_`); they now reject it up-front with
//! `FerroError::InvalidParameter`, matching sklearn's reject-at-fit contract
//! (R-DEV-1 / R-DEV-2 exception parity).
//!
//! NOTE on sample_weight: sklearn's CV `fit` methods take a `sample_weight`
//! kwarg (validated via `_check_sample_weight`, raising on a non-finite weight).
//! ferrolearn's `Fit::fit` for all four estimators takes ONLY `(x, y)` — there
//! is no `sample_weight` argument in the public `Fit` trait surface — so the
//! sklearn `sample_weight`-finiteness raise has no ferrolearn fit-entry
//! counterpart here. X (and float `y`) are the validated inputs.
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, NOT
//! copied from ferrolearn). Confirmed via:
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV
//! # X-nan/+inf/-inf -> ValueError for all four; y-nan/+inf -> ValueError for the
//! # three float-y regressors. All-finite fits: RidgeCV alpha_=0.1,
//! # LassoCV alpha_=0.01, ElasticNetCV alpha_=0.01/l1_ratio_=0.5, LogRegCV C_=[0.1].
//! "
//! ```
//!
//! Oracle result (sklearn 1.5.2, abbreviated — every non-finite case raises):
//! ```text
//! RidgeCV        X-nan ValueError: Input X contains NaN.
//! RidgeCV        X+inf/X-inf ValueError: Input X contains infinity ...
//! RidgeCV        y-nan/y+inf ValueError: Input y contains NaN / infinity ...
//! LassoCV        X-nan/X+inf/X-inf / y-nan/y+inf  ValueError ...
//! ElasticNetCV   X-nan/X+inf/X-inf / y-nan/y+inf  ValueError ...
//! LogRegCV       X-nan/X+inf/X-inf                ValueError ...
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::{Fit, Predict};
use ferrolearn_linear::{ElasticNetCV, LassoCV, LogisticRegressionCV, RidgeCV};
use ndarray::{Array1, Array2, array};

/// Shared finite design matrix (n=10, 3 features) with enough samples for the
/// default-ish CV fold counts used below.
fn finite_x() -> Array2<f64> {
    array![
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0],
        [2.0, 1.0, 0.0],
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 3.0],
        [3.0, 1.0, 2.0],
        [1.0, 1.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.5, 1.5, 2.5],
        [2.0, 2.0, 0.0],
    ]
}

/// Float regression targets matching `finite_x` (the regressor oracle fixture).
fn finite_yr() -> Array1<f64> {
    array![2.0, 1.0, 3.0, 2.5, 1.0, 5.0, 2.0, 2.5, 2.2, 3.5]
}

/// Two well-separated clusters for `LogisticRegressionCV` (n=10, 2 classes).
fn finite_xc() -> Array2<f64> {
    array![
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [1.5, 1.5],
        [8.0, 8.0],
        [8.0, 9.0],
        [9.0, 8.0],
        [9.0, 9.0],
        [8.5, 8.5],
    ]
}

/// Integer class labels for `LogisticRegressionCV` (`Array1<usize>`).
fn finite_yc() -> Array1<usize> {
    array![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1]
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

/// The three non-finite-`X` perturbations (NaN/+inf/-inf), shared across the
/// regressors (3-feature `finite_x`).
fn nonfinite_x_cases() -> Vec<(Array2<f64>, &'static str)> {
    let x = finite_x();
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

/// The three non-finite-`X` perturbations for the 2-feature classifier design.
fn nonfinite_xc_cases() -> Vec<(Array2<f64>, &'static str)> {
    let x = finite_xc();
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

/// Non-finite float-`y` perturbations (NaN/inf), for the regressors.
fn nonfinite_yr_cases() -> Vec<(Array1<f64>, &'static str)> {
    let mut y_nan = finite_yr();
    y_nan[0] = f64::NAN;
    let mut y_inf = finite_yr();
    y_inf[1] = f64::INFINITY;
    vec![(y_nan, "NaN in y"), (y_inf, "inf in y")]
}

// ---------------------------------------------------------------------------
// RidgeCV (Fit<Array2, Array1<f64>>) — X + y validated.
// ---------------------------------------------------------------------------

#[test]
fn ridge_cv_rejects_non_finite_x_like_sklearn() {
    // Oracle: RidgeCV(alphas=[0.1,1.0,10.0]).fit raises ValueError for NaN/+inf/
    // -inf in X (`_RidgeGCV.fit` `_validate_data`, `_ridge.py:2087`).
    let model = RidgeCV::<f64>::new().with_alphas(vec![0.1, 1.0, 10.0]);
    let y = finite_yr();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &y);
        assert!(is_nonfinite_xy_err(&res), "RidgeCV: {tag} must be rejected");
    }
}

#[test]
fn ridge_cv_rejects_non_finite_y_like_sklearn() {
    // Oracle: RidgeCV(...).fit raises ValueError for NaN/inf in y.
    let model = RidgeCV::<f64>::new().with_alphas(vec![0.1, 1.0, 10.0]);
    let x = finite_x();
    for (y, tag) in nonfinite_yr_cases() {
        let res = model.fit(&x, &y);
        assert!(is_nonfinite_xy_err(&res), "RidgeCV: {tag} must be rejected");
    }
}

#[test]
fn ridge_cv_rejects_non_finite_x_kfold_path_like_sklearn() {
    // The k-fold path (`cv=Some(k)`) must ALSO reject up-front, before the fold
    // split (sklearn `GridSearchCV` → per-fold `Ridge.fit` `_validate_data`,
    // `_ridge.py:1242`; ferrolearn validates at the CV entry for the clean error).
    let model = RidgeCV::<f64>::new()
        .with_alphas(vec![0.1, 1.0, 10.0])
        .with_cv(3);
    let y = finite_yr();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "RidgeCV(cv=3): {tag} must be rejected"
        );
    }
}

#[test]
fn ridge_cv_all_finite_unchanged() {
    // No false positive: all-finite input still fits, and the GCV alpha
    // selection is unchanged. Oracle (sklearn 1.5.2):
    //   RidgeCV(alphas=[0.1,1.0,10.0]).fit(X, y).alpha_ == 0.1
    // (GCV LOO selection over this fixture).
    let model = RidgeCV::<f64>::new().with_alphas(vec![0.1, 1.0, 10.0]);
    let x = finite_x();
    let y = finite_yr();
    let fitted = model.fit(&x, &y).expect("finite RidgeCV fit must succeed");
    assert!(
        (fitted.best_alpha() - 0.1).abs() < 1e-9,
        "RidgeCV alpha_ regression: expected 0.1, got {}",
        fitted.best_alpha()
    );
    // Predictions are finite (the fit ran on clean data).
    let preds = fitted.predict(&x).expect("predict must succeed");
    assert!(preds.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// LassoCV (Fit<Array2, Array1<f64>>) — X + y validated.
// ---------------------------------------------------------------------------

#[test]
fn lasso_cv_rejects_non_finite_x_like_sklearn() {
    // Oracle: LassoCV(...).fit raises ValueError for NaN/+inf/-inf in X
    // (`LinearModelCV.fit` `_validate_data`, `_coordinate_descent.py:1619`).
    let model = LassoCV::<f64>::new()
        .with_alphas(vec![0.01, 0.1, 1.0])
        .with_cv(3);
    let y = finite_yr();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &y);
        assert!(is_nonfinite_xy_err(&res), "LassoCV: {tag} must be rejected");
    }
}

#[test]
fn lasso_cv_rejects_non_finite_y_like_sklearn() {
    let model = LassoCV::<f64>::new()
        .with_alphas(vec![0.01, 0.1, 1.0])
        .with_cv(3);
    let x = finite_x();
    for (y, tag) in nonfinite_yr_cases() {
        let res = model.fit(&x, &y);
        assert!(is_nonfinite_xy_err(&res), "LassoCV: {tag} must be rejected");
    }
}

#[test]
fn lasso_cv_rejects_non_finite_x_auto_grid_like_sklearn() {
    // The auto-alpha-grid path (no explicit alphas) must also reject up-front,
    // before `compute_alpha_max` would otherwise propagate NaN.
    let model = LassoCV::<f64>::new().with_n_alphas(5).with_cv(3);
    let y = finite_yr();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "LassoCV(auto-grid): {tag} must be rejected"
        );
    }
}

#[test]
fn lasso_cv_all_finite_unchanged() {
    // Oracle (sklearn 1.5.2): LassoCV(alphas=[0.01,0.1,1.0],cv=3).fit(X,y).alpha_
    // == 0.01.
    let model = LassoCV::<f64>::new()
        .with_alphas(vec![0.01, 0.1, 1.0])
        .with_cv(3);
    let x = finite_x();
    let y = finite_yr();
    let fitted = model.fit(&x, &y).expect("finite LassoCV fit must succeed");
    assert!(
        (fitted.best_alpha() - 0.01).abs() < 1e-9,
        "LassoCV alpha_ regression: expected 0.01, got {}",
        fitted.best_alpha()
    );
    let preds = fitted.predict(&x).expect("predict must succeed");
    assert!(preds.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// ElasticNetCV (Fit<Array2, Array1<f64>>) — X + y validated.
// ---------------------------------------------------------------------------

#[test]
fn elastic_net_cv_rejects_non_finite_x_like_sklearn() {
    // Oracle: ElasticNetCV(...).fit raises ValueError for NaN/+inf/-inf in X
    // (`LinearModelCV.fit` `_validate_data`, `_coordinate_descent.py:1644`).
    let model = ElasticNetCV::<f64>::new().with_n_alphas(5).with_cv(3);
    let y = finite_yr();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "ElasticNetCV: {tag} must be rejected"
        );
    }
}

#[test]
fn elastic_net_cv_rejects_non_finite_y_like_sklearn() {
    let model = ElasticNetCV::<f64>::new().with_n_alphas(5).with_cv(3);
    let x = finite_x();
    for (y, tag) in nonfinite_yr_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "ElasticNetCV: {tag} must be rejected"
        );
    }
}

#[test]
fn elastic_net_cv_all_finite_unchanged() {
    // Oracle (sklearn 1.5.2):
    //   ElasticNetCV(alphas=[0.01,0.1,1.0],l1_ratio=0.5,cv=3).fit(X,y)
    //   -> alpha_ == 0.01, l1_ratio_ == 0.5.
    // ferrolearn auto-generates the alpha grid (no explicit-alphas API); with the
    // default single l1_ratio=0.5 the CV selection still picks a small alpha for
    // this near-linear fixture. We assert the fit succeeds and l1_ratio_ == 0.5
    // (the default, byte-identical to today) and predictions are finite.
    let model = ElasticNetCV::<f64>::new().with_n_alphas(20).with_cv(3);
    let x = finite_x();
    let y = finite_yr();
    let fitted = model
        .fit(&x, &y)
        .expect("finite ElasticNetCV fit must succeed");
    assert!(
        (fitted.best_l1_ratio() - 0.5).abs() < 1e-12,
        "ElasticNetCV l1_ratio_ regression: expected 0.5, got {}",
        fitted.best_l1_ratio()
    );
    assert!(
        fitted.best_alpha() > 0.0,
        "ElasticNetCV alpha_ must be positive"
    );
    let preds = fitted.predict(&x).expect("predict must succeed");
    assert!(preds.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// LogisticRegressionCV (Fit<Array2, Array1<usize>>) — X only (y is labels).
// ---------------------------------------------------------------------------

#[test]
fn logistic_regression_cv_rejects_non_finite_x_like_sklearn() {
    // Oracle: LogisticRegressionCV(Cs=[0.1,1.0,10.0],cv=2).fit raises ValueError
    // for NaN/+inf/-inf in X (`_validate_data`, `_logistic.py:1868`). y is
    // integer class labels (never non-finite).
    let model = LogisticRegressionCV::<f64>::new()
        .with_cs(vec![0.1, 1.0, 10.0])
        .with_cv(2);
    let yc = finite_yc();
    for (x, tag) in nonfinite_xc_cases() {
        let res = model.fit(&x, &yc);
        assert!(
            is_nonfinite_xy_err(&res),
            "LogisticRegressionCV: {tag} must be rejected"
        );
    }
}

#[test]
fn logistic_regression_cv_all_finite_unchanged() {
    // Oracle (sklearn 1.5.2):
    //   LogisticRegressionCV(Cs=[0.1,1.0,10.0],cv=2).fit(Xc,yc)
    //   classifies the two clusters perfectly; classes_ == [0, 1].
    let model = LogisticRegressionCV::<f64>::new()
        .with_cs(vec![0.1, 1.0, 10.0])
        .with_cv(2);
    let x = finite_xc();
    let yc = finite_yc();
    let fitted = model
        .fit(&x, &yc)
        .expect("finite LogisticRegressionCV fit must succeed");
    assert!(fitted.best_c() > 0.0, "best_c must be a positive candidate");
    let preds = fitted.predict(&x).expect("predict must succeed");
    let correct = preds.iter().zip(yc.iter()).filter(|(p, a)| p == a).count();
    assert_eq!(
        correct, 10,
        "separable clusters must classify perfectly, got {correct}/10"
    );
}
