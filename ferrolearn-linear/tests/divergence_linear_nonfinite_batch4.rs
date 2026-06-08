//! Non-finite input validation parity (batch 4 of the #2263 sweep) for the
//! `ferrolearn-linear` estimators `LogisticRegression`, `LinearDiscriminantAnalysis`
//! (LDA), `SGDClassifier`, `SGDRegressor`, `LinearSVC`, and `LinearSVR`.
//!
//! scikit-learn 1.5.2 validates `X`/`y` at fit through `_validate_data(...)`
//! with the default `force_all_finite=True`:
//!   - `LogisticRegression.fit`        → `sklearn/linear_model/_logistic.py:1223`
//!     (also `sample_weight` via `_check_sample_weight`, `_logistic.py:303`)
//!   - `LinearDiscriminantAnalysis.fit`→ `sklearn/discriminant_analysis.py:589`
//!     (no `sample_weight`; `y` integer labels)
//!   - `SGDClassifier.fit`             → `sklearn/linear_model/_stochastic_gradient.py:1476`
//!     (also `sample_weight` via `_check_sample_weight`, `:1501`; `y` integer)
//!   - `SGDRegressor.fit`              → `sklearn/linear_model/_stochastic_gradient.py:1476`
//!     (also `y` float + `sample_weight`)
//!   - `LinearSVC.fit`                 → `sklearn/svm/_classes.py:302` (`y` integer)
//!   - `LinearSVR.fit`                 → `sklearn/svm/_classes.py:302` (`y` float)
//!
//! `check_array` raises `ValueError` on any NaN or +/-inf in `X` or `y` (and on
//! a non-finite `sample_weight` where the estimator takes one) BEFORE the solver
//! runs. Previously ferrolearn accepted non-finite input and produced NaN
//! `coef_`/`intercept_`; these estimators now reject it with
//! `FerroError::InvalidParameter`, matching sklearn's reject-at-fit contract
//! (R-DEV-1 / R-DEV-2 exception parity).
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, NOT
//! copied from ferrolearn). The following script confirms every non-finite case
//! raises `ValueError` and every finite case fits:
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
//! from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
//! from sklearn.svm import LinearSVC, LinearSVR
//! Xg = np.array([[1.,1.],[2.,2.],[3.,1.],[5.,5.],[6.,5.],[7.,6.]])
//! yc = np.array([0,0,0,1,1,1]); yr = np.array([1.,2.,3.,4.,5.,6.])
//! def mkX(v): X=Xg.copy(); X[0,0]=v; return X
//! # LogisticRegression / SGDClassifier / LinearSVC : X-nan/inf, sample_weight-nan
//! # LDA : X-nan/inf
//! # SGDRegressor / LinearSVR : X-nan/inf, y-nan/inf, sample_weight-nan
//! "
//! ```
//!
//! Oracle result (sklearn 1.5.2, abbreviated — every case below raises):
//! ```text
//! LogReg X-nan/X+inf/X-inf  ValueError: Input X contains NaN / infinity ...
//! LogReg sw-nan/sw+inf      ValueError: Input sample_weight contains NaN / infinity ...
//! LDA    X-nan/X+inf/X-inf  ValueError: Input X contains NaN / infinity ...
//! SGDClf X-nan/X+inf        ValueError: Input X contains NaN / infinity ...
//! SGDClf sw-nan            ValueError: Input sample_weight contains NaN ...
//! SGDReg X-nan/y-nan/y+inf  ValueError: Input X/y contains NaN / infinity ...
//! SGDReg sw-nan            ValueError: Input sample_weight contains NaN ...
//! LinSVC X-nan/X+inf/X-inf  ValueError: Input X contains NaN / infinity ...
//! LinSVR X-nan/X+inf/y-nan/y+inf  ValueError: Input X/y contains NaN / infinity ...
//! LinSVR sw -- N/A (ferrolearn LinearSVR::fit takes no sample_weight arg)
//! ```
//!
//! NOTE: ferrolearn's `LinearSVC::fit`/`LinearSVR::fit` take no `sample_weight`
//! argument (the public `Fit` signature has none), so the sklearn
//! `sample_weight`-finiteness check has no ferrolearn fit-entry counterpart for
//! those two; only X (+ y for the regressor) are validated. The sklearn
//! `sample_weight` raises for `LinearSVC`/`LinearSVR` are listed in the oracle
//! for completeness but are not testable at this fit signature.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::{Fit, Predict};
use ferrolearn_linear::lda::LDA;
use ferrolearn_linear::{LinearSVC, LinearSVR, LogisticRegression, SGDClassifier, SGDRegressor};
use ndarray::{Array1, Array2, array};

/// Shared finite design: two well-separated clusters, fits cleanly under every
/// classifier/regressor in this batch.
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

/// Integer class labels for the classifiers (`Array1<usize>`).
fn finite_yc() -> Array1<usize> {
    array![0usize, 0, 0, 1, 1, 1]
}

/// Float regression targets for the regressors (`Array1<f64>`).
fn finite_yr() -> Array1<f64> {
    array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
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

/// The three non-finite-`X` perturbations (NaN/+inf/-inf), shared across every
/// estimator since they all validate `X`.
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

/// Non-finite float-`y` perturbations (NaN/inf), for the regressors.
fn nonfinite_yr_cases() -> Vec<(Array1<f64>, &'static str)> {
    let mut y_nan = finite_yr();
    y_nan[0] = f64::NAN;
    let mut y_inf = finite_yr();
    y_inf[1] = f64::INFINITY;
    vec![(y_nan, "NaN in y"), (y_inf, "inf in y")]
}

// ---------------------------------------------------------------------------
// LogisticRegression (Fit<Array2, Array1<usize>>) + sample_weight
// ---------------------------------------------------------------------------

#[test]
fn logreg_rejects_non_finite_x_like_sklearn() {
    // Oracle: LogisticRegression().fit raises ValueError for NaN/+inf/-inf in X
    // (`_logistic.py:1223`, force_all_finite=True). y is integer-labeled.
    let model = LogisticRegression::<f64>::new();
    let yc = finite_yc();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &yc);
        assert!(
            is_nonfinite_xy_err(&res),
            "LogisticRegression: {tag} must be rejected"
        );
    }
}

#[test]
fn logreg_rejects_non_finite_sample_weight_like_sklearn() {
    // Oracle: LogisticRegression().fit(X, y, sample_weight=w) raises ValueError
    // for a non-finite weight (`_check_sample_weight`, `_logistic.py:303`).
    let x = finite_x();
    let yc = finite_yc();
    let model = LogisticRegression::<f64>::new();

    let mut w_nan = Array1::<f64>::ones(6);
    w_nan[0] = f64::NAN;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &yc, Some(&w_nan))),
        "LogisticRegression: NaN sample_weight must be rejected"
    );

    let mut w_inf = Array1::<f64>::ones(6);
    w_inf[3] = f64::INFINITY;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &yc, Some(&w_inf))),
        "LogisticRegression: inf sample_weight must be rejected"
    );
}

#[test]
fn logreg_finite_input_fits_no_false_positive() {
    // No regression: the finite path still fits and yields finite parameters.
    let x = finite_x();
    let yc = finite_yc();
    let fitted = LogisticRegression::<f64>::new()
        .fit(&x, &yc)
        .expect("finite input must fit");
    let preds = fitted.predict(&x).expect("predict on finite input");
    assert_eq!(preds.len(), 6);
}

// ---------------------------------------------------------------------------
// LinearDiscriminantAnalysis (Fit<Array2, Array1<usize>>; no sample_weight)
// ---------------------------------------------------------------------------

#[test]
fn lda_rejects_non_finite_x_like_sklearn() {
    // Oracle: LinearDiscriminantAnalysis().fit raises ValueError for NaN/+inf/-inf
    // in X (`discriminant_analysis.py:589`, force_all_finite=True). y integer.
    let model = LDA::<f64>::new(Some(1));
    let yc = finite_yc();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &yc);
        assert!(is_nonfinite_xy_err(&res), "LDA: {tag} must be rejected");
    }
}

#[test]
fn lda_finite_input_fits_no_false_positive() {
    let x = finite_x();
    let yc = finite_yc();
    let fitted = LDA::<f64>::new(Some(1))
        .fit(&x, &yc)
        .expect("finite input must fit");
    let preds = fitted.predict(&x).expect("predict on finite input");
    assert_eq!(preds.len(), 6);
}

// ---------------------------------------------------------------------------
// SGDClassifier (Fit<Array2, Array1<usize>>) + sample_weight
// ---------------------------------------------------------------------------

#[test]
fn sgd_classifier_rejects_non_finite_x_like_sklearn() {
    // Oracle: SGDClassifier().fit raises ValueError for NaN/+inf/-inf in X
    // (`_stochastic_gradient.py:1476`, force_all_finite=True). y integer.
    let model = SGDClassifier::<f64>::new();
    let yc = finite_yc();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &yc);
        assert!(
            is_nonfinite_xy_err(&res),
            "SGDClassifier: {tag} must be rejected"
        );
    }
}

#[test]
fn sgd_classifier_rejects_non_finite_sample_weight_like_sklearn() {
    // Oracle: SGDClassifier().fit(X, y, sample_weight=w) raises ValueError for a
    // non-finite weight (`_check_sample_weight`, `_stochastic_gradient.py:1501`).
    let x = finite_x();
    let yc = finite_yc();
    let model = SGDClassifier::<f64>::new();

    let mut w_nan = Array1::<f64>::ones(6);
    w_nan[0] = f64::NAN;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &yc, &w_nan)),
        "SGDClassifier: NaN sample_weight must be rejected"
    );

    let mut w_inf = Array1::<f64>::ones(6);
    w_inf[2] = f64::INFINITY;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &yc, &w_inf)),
        "SGDClassifier: inf sample_weight must be rejected"
    );
}

#[test]
fn sgd_classifier_finite_input_fits_no_false_positive() {
    let x = finite_x();
    let yc = finite_yc();
    let fitted = SGDClassifier::<f64>::new()
        .fit(&x, &yc)
        .expect("finite input must fit");
    let preds = fitted.predict(&x).expect("predict on finite input");
    assert_eq!(preds.len(), 6);
}

// ---------------------------------------------------------------------------
// SGDRegressor (Fit<Array2, Array1<f64>>) + y + sample_weight
// ---------------------------------------------------------------------------

#[test]
fn sgd_regressor_rejects_non_finite_x_and_y_like_sklearn() {
    // Oracle: SGDRegressor().fit raises ValueError for NaN/+inf/-inf in X AND for
    // NaN/inf in y (`_stochastic_gradient.py:1476`, force_all_finite=True).
    let model = SGDRegressor::<f64>::new();
    let yr = finite_yr();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &yr);
        assert!(
            is_nonfinite_xy_err(&res),
            "SGDRegressor: {tag} must be rejected"
        );
    }
    let x = finite_x();
    for (y, tag) in nonfinite_yr_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "SGDRegressor: {tag} must be rejected"
        );
    }
}

#[test]
fn sgd_regressor_rejects_non_finite_sample_weight_like_sklearn() {
    // Oracle: SGDRegressor().fit(X, y, sample_weight=w) raises ValueError for a
    // non-finite weight (`_check_sample_weight`, `_stochastic_gradient.py:1501`).
    let x = finite_x();
    let yr = finite_yr();
    let model = SGDRegressor::<f64>::new();

    let mut w_nan = Array1::<f64>::ones(6);
    w_nan[0] = f64::NAN;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &yr, &w_nan)),
        "SGDRegressor: NaN sample_weight must be rejected"
    );

    let mut w_inf = Array1::<f64>::ones(6);
    w_inf[1] = f64::INFINITY;
    assert!(
        is_nonfinite_sw_err(&model.fit_with_sample_weight(&x, &yr, &w_inf)),
        "SGDRegressor: inf sample_weight must be rejected"
    );
}

#[test]
fn sgd_regressor_finite_input_fits_no_false_positive() {
    let x = finite_x();
    let yr = finite_yr();
    let fitted = SGDRegressor::<f64>::new()
        .fit(&x, &yr)
        .expect("finite input must fit");
    let preds = fitted.predict(&x).expect("predict on finite input");
    assert!(preds.iter().all(|v| v.is_finite()), "finite preds");
}

// ---------------------------------------------------------------------------
// LinearSVC (Fit<Array2, Array1<usize>>; no sample_weight at this signature)
// ---------------------------------------------------------------------------

#[test]
fn linsvc_rejects_non_finite_x_like_sklearn() {
    // Oracle: LinearSVC().fit raises ValueError for NaN/+inf/-inf in X
    // (`svm/_classes.py:302`, force_all_finite=True). y integer.
    let model = LinearSVC::<f64>::new();
    let yc = finite_yc();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &yc);
        assert!(
            is_nonfinite_xy_err(&res),
            "LinearSVC: {tag} must be rejected"
        );
    }
}

#[test]
fn linsvc_finite_input_fits_no_false_positive() {
    let x = finite_x();
    let yc = finite_yc();
    let fitted = LinearSVC::<f64>::new()
        .fit(&x, &yc)
        .expect("finite input must fit");
    let preds = fitted.predict(&x).expect("predict on finite input");
    assert_eq!(preds.len(), 6);
}

// ---------------------------------------------------------------------------
// LinearSVR (Fit<Array2, Array1<f64>>; no sample_weight at this signature)
// ---------------------------------------------------------------------------

#[test]
fn linsvr_rejects_non_finite_x_and_y_like_sklearn() {
    // Oracle: LinearSVR().fit raises ValueError for NaN/+inf/-inf in X AND for
    // NaN/inf in y (`svm/_classes.py:302`, force_all_finite=True).
    let model = LinearSVR::<f64>::new();
    let yr = finite_yr();
    for (x, tag) in nonfinite_x_cases() {
        let res = model.fit(&x, &yr);
        assert!(
            is_nonfinite_xy_err(&res),
            "LinearSVR: {tag} must be rejected"
        );
    }
    let x = finite_x();
    for (y, tag) in nonfinite_yr_cases() {
        let res = model.fit(&x, &y);
        assert!(
            is_nonfinite_xy_err(&res),
            "LinearSVR: {tag} must be rejected"
        );
    }
}

#[test]
fn linsvr_finite_input_fits_no_false_positive() {
    let x = finite_x();
    let yr = finite_yr();
    let fitted = LinearSVR::<f64>::new()
        .fit(&x, &yr)
        .expect("finite input must fit");
    let preds = fitted.predict(&x).expect("predict on finite input");
    assert!(preds.iter().all(|v| v.is_finite()), "finite preds");
}
