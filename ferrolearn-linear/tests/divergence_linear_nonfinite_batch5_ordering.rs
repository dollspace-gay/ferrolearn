//! Validation-ORDERING divergences left uncovered by batch-5 of the #2265 sweep
//! (`divergence_linear_nonfinite_batch5.rs`). The batch-5 tests only feed inputs
//! that are large enough for the fold split (`n_samples >= cv`) and have >= 2
//! classes, so they never exercise the case where ferrolearn's fold-count /
//! class-count guards fire BEFORE the non-finite-X guard.
//!
//! scikit-learn 1.5.2 validates finiteness inside `_validate_data(...)` at the
//! very TOP of `fit`, BEFORE `check_cv` / the fold split and BEFORE any
//! class-count processing:
//!   - `LinearModelCV.fit` (LassoCV/ElasticNetCV): `_validate_data`
//!     (`_coordinate_descent.py:1619`/`:1644`) runs at fit-entry; `cv =
//!     check_cv(self.cv)` only at `_coordinate_descent.py:1730`. So a NaN in X
//!     with `n_samples < cv` raises `ValueError("Input X contains NaN.")`, NOT a
//!     fold-count error.
//!   - `LogisticRegressionCV.fit`: `_validate_data` (`_logistic.py:1868`) runs
//!     before `self.classes_ = ...` and the fold split, so a NaN in X with
//!     `n_samples < cv` OR with a single class still raises the NaN error first.
//!   - `RidgeCV(cv=k)` is the MIRROR case: sklearn routes the `cv=Some(k)` path
//!     through `GridSearchCV(KFold(n_splits=k))`, whose split-count check fires
//!     BEFORE the per-fold `Ridge.fit` `_validate_data`, so for `n_samples < k`
//!     sklearn raises `"Cannot have number of splits n_splits=k greater than the
//!     number of samples ..."` — NOT the NaN error. ferrolearn checks finiteness
//!     up-front (before the cv branch) so it raises the NaN error first here,
//!     diverging in the OPPOSITE direction.
//!
//! ferrolearn instead checks `n_samples < cv` (LassoCV/ElasticNetCV/LogRegCV)
//! and `classes.len() < 2` (LogRegCV) BEFORE the non-finite guard, so it returns
//! the WRONG error variant (`InsufficientSamples`) for the three CD/logistic
//! estimators; and RidgeCV's up-front finiteness guard returns the NaN error
//! where sklearn returns the split-count error.
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — NEVER copied from ferrolearn):
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.linear_model import LassoCV, ElasticNetCV, LogisticRegressionCV, RidgeCV
//! X = np.array([[1.,2.],[3.,4.],[5.,6.]]); Xn = X.copy(); Xn[0,0]=np.nan
//! y = np.array([1.,2.,3.])
//! LassoCV(alphas=[0.1,1.0], cv=5).fit(Xn, y)         # ValueError: Input X contains NaN.
//! ElasticNetCV(n_alphas=5, cv=5).fit(Xn, y)          # ValueError: Input X contains NaN.
//! LogisticRegressionCV(Cs=[0.1,1.0],cv=5).fit(Xn,[0,1,0])  # ValueError: Input X contains NaN.
//! X4 = np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]]); X4[0,0]=np.nan
//! LogisticRegressionCV(Cs=[0.1,1.0],cv=2).fit(X4,[0,0,0,0]) # ValueError: Input X contains NaN.
//! RidgeCV(alphas=[0.1,1.0], cv=5).fit(X, y)          # ValueError: Cannot have number of splits ...
//! "
//! ```
//! Oracle results (sklearn 1.5.2, confirmed live):
//! ```text
//! LassoCV       n<cv NaN  -> ValueError: Input X contains NaN.
//! ElasticNetCV  n<cv NaN  -> ValueError: Input X contains NaN.
//! LogRegCV      n<cv NaN  -> ValueError: Input X contains NaN.
//! LogRegCV      1class NaN-> ValueError: Input X contains NaN.
//! RidgeCV(cv=5) n<cv NaN  -> ValueError: Cannot have number of splits n_splits=5 ...
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::Fit;
use ferrolearn_linear::{ElasticNetCV, LassoCV, LogisticRegressionCV, RidgeCV};
use ndarray::{array, Array1, Array2};

/// `true` iff the error is the non-finite `InvalidParameter` for `X` (the
/// sklearn `ValueError: Input X contains NaN.` analog the batch-5 guard emits).
fn is_nonfinite_x_err(res: &Result<impl Sized, FerroError>) -> bool {
    matches!(
        res,
        Err(FerroError::InvalidParameter { name, reason })
            if name == "X" && reason.contains("NaN or infinity")
    )
}

/// 3-sample design with a NaN in `X[0,0]`; `n=3 < cv=5`.
fn x3_nan() -> Array2<f64> {
    let mut x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    x[[0, 0]] = f64::NAN;
    x
}

// ---------------------------------------------------------------------------
// #2267 — LassoCV / ElasticNetCV / LogisticRegressionCV check the fold-count
// (and class-count) BEFORE finiteness. sklearn validates finiteness FIRST.
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn `LassoCV::fit` checks `n_samples < cv`
/// (`lasso_cv.rs:322`) BEFORE the non-finite guard (`lasso_cv.rs:342`).
/// sklearn `LinearModelCV.fit` runs `_validate_data`
/// (`_coordinate_descent.py:1619`) — finiteness — before `cv = check_cv(...)`
/// (`_coordinate_descent.py:1730`). Input: `n=3 < cv=5`, NaN in `X`.
/// sklearn raises `ValueError: Input X contains NaN.`; ferrolearn returns
/// `InsufficientSamples`.
/// Tracking: #2267
#[test]
#[ignore = "divergence: LassoCV validates n<cv before non-finite X; tracking #2267"]
fn lasso_cv_validates_finiteness_before_fold_count_like_sklearn() {
    let x = x3_nan();
    let y: Array1<f64> = array![1.0, 2.0, 3.0];
    let res = LassoCV::<f64>::new()
        .with_alphas(vec![0.1, 1.0])
        .with_cv(5)
        .fit(&x, &y);
    assert!(
        is_nonfinite_x_err(&res),
        "LassoCV n<cv + NaN must raise the non-finite-X error first (sklearn \
         `Input X contains NaN.`), got {res:?}"
    );
}

/// Divergence: ferrolearn `ElasticNetCV::fit` checks `n_samples < cv`
/// (`elastic_net_cv.rs:352`) BEFORE the non-finite guard
/// (`elastic_net_cv.rs:379`). sklearn `LinearModelCV.fit` `_validate_data`
/// (`_coordinate_descent.py:1644`) runs first. Input: `n=3 < cv=5`, NaN in `X`.
/// sklearn raises `ValueError: Input X contains NaN.`; ferrolearn returns
/// `InsufficientSamples`.
/// Tracking: #2267
#[test]
#[ignore = "divergence: ElasticNetCV validates n<cv before non-finite X; tracking #2267"]
fn elastic_net_cv_validates_finiteness_before_fold_count_like_sklearn() {
    let x = x3_nan();
    let y: Array1<f64> = array![1.0, 2.0, 3.0];
    let res = ElasticNetCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(5)
        .fit(&x, &y);
    assert!(
        is_nonfinite_x_err(&res),
        "ElasticNetCV n<cv + NaN must raise the non-finite-X error first \
         (sklearn `Input X contains NaN.`), got {res:?}"
    );
}

/// Divergence: ferrolearn `LogisticRegressionCV::fit` checks `n_samples < cv`
/// (`logistic_regression_cv.rs:337`) BEFORE the non-finite guard
/// (`logistic_regression_cv.rs:369`). sklearn `_validate_data`
/// (`_logistic.py:1868`) runs before `check_cv` / the fold split. Input:
/// `n=3 < cv=5`, NaN in `X`. sklearn raises `ValueError: Input X contains
/// NaN.`; ferrolearn returns `InsufficientSamples`.
/// Tracking: #2267
#[test]
#[ignore = "divergence: LogisticRegressionCV validates n<cv before non-finite X; tracking #2267"]
fn logistic_regression_cv_validates_finiteness_before_fold_count_like_sklearn() {
    let x = x3_nan();
    let yc: Array1<usize> = array![0usize, 1, 0];
    let res = LogisticRegressionCV::<f64>::new()
        .with_cs(vec![0.1, 1.0])
        .with_cv(5)
        .fit(&x, &yc);
    assert!(
        is_nonfinite_x_err(&res),
        "LogisticRegressionCV n<cv + NaN must raise the non-finite-X error \
         first (sklearn `Input X contains NaN.`), got {res:?}"
    );
}

/// Divergence: ferrolearn `LogisticRegressionCV::fit` checks
/// `classes.len() < 2` (`logistic_regression_cv.rs:349`) BEFORE the non-finite
/// guard (`logistic_regression_cv.rs:369`). sklearn `_validate_data`
/// (`_logistic.py:1868`) runs before `self.classes_` is computed
/// (`_logistic.py:1232`-style label encoding). Input: `n=4 >= cv=2`, a SINGLE
/// class, NaN in `X`. sklearn raises `ValueError: Input X contains NaN.`;
/// ferrolearn returns `InsufficientSamples` ("requires at least 2 distinct
/// classes").
/// Tracking: #2267
#[test]
#[ignore = "divergence: LogisticRegressionCV validates single-class before non-finite X; tracking #2267"]
fn logistic_regression_cv_validates_finiteness_before_class_count_like_sklearn() {
    let mut x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    x[[0, 0]] = f64::NAN;
    let yc: Array1<usize> = array![0usize, 0, 0, 0];
    let res = LogisticRegressionCV::<f64>::new()
        .with_cs(vec![0.1, 1.0])
        .with_cv(2)
        .fit(&x, &yc);
    assert!(
        is_nonfinite_x_err(&res),
        "LogisticRegressionCV single-class + NaN must raise the non-finite-X \
         error first (sklearn `Input X contains NaN.`), got {res:?}"
    );
}

// ---------------------------------------------------------------------------
// #2268 — RidgeCV(cv=k) MIRROR case: ferrolearn raises non-finite X up-front;
// sklearn raises the n_splits>n_samples split error first.
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn `RidgeCV::fit` checks finiteness up-front
/// (`ridge_cv.rs:293`) BEFORE the cv-mode branch, so for `cv=Some(k)` with
/// `n_samples < k` it returns the non-finite-X error. sklearn routes the
/// `cv=Some(k)` path through `GridSearchCV(KFold(n_splits=k))` whose
/// split-count check fires before the per-fold `Ridge.fit` `_validate_data`
/// (`_ridge.py:1242`), so sklearn raises `ValueError: Cannot have number of
/// splits n_splits=5 greater than the number of samples ...` — a `cv`-named
/// error, NOT the non-finite-X error. Input: `n=3 < cv=5`, NaN in `X`.
/// ferrolearn must surface the fold-count (`cv`) error to match sklearn's
/// ordering here, but returns `InvalidParameter { name: "X", ... }`.
/// Tracking: #2268
#[test]
#[ignore = "divergence: RidgeCV(cv=k) raises non-finite X before n_splits>n; tracking #2268"]
fn ridge_cv_kfold_raises_split_count_before_finiteness_like_sklearn() {
    let x = x3_nan();
    let y: Array1<f64> = array![1.0, 2.0, 3.0];
    let res = RidgeCV::<f64>::new()
        .with_alphas(vec![0.1, 1.0])
        .with_cv(5)
        .fit(&x, &y);
    // sklearn raises the split-count (`cv`) error first, NOT the NaN error.
    // ferrolearn returns the NaN error (over-prioritized finiteness).
    let is_split_count_err = matches!(
        &res,
        Err(FerroError::InsufficientSamples { context, .. })
            if context.contains("folds")
    ) || matches!(
        &res,
        Err(FerroError::InvalidParameter { name, .. }) if name == "cv"
    );
    assert!(
        is_split_count_err,
        "RidgeCV(cv=5) n<cv + NaN must raise the split-count (cv) error first \
         like sklearn (`Cannot have number of splits ...`), got {res:?}"
    );
}
