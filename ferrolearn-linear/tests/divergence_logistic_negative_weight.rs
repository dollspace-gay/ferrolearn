//! Divergence: `LogisticRegression::fit_with_sample_weight` rejects negative
//! `sample_weight` entries, whereas scikit-learn 1.5.2 ACCEPTS them and fits.
//!
//! This is the same class of divergence as huber #2159: sklearn's
//! `LogisticRegression.fit` validates `sample_weight` via `_check_sample_weight`
//! WITHOUT `only_non_negative=True`
//! (`sklearn/linear_model/_logistic.py:303`:
//!   `sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)`),
//! and `_check_sample_weight` only runs `check_non_negative` when
//! `only_non_negative` is set (`sklearn/utils/validation.py:2062-2063`). So
//! sklearn imposes NO non-negativity constraint: a negative weight is accepted
//! and the model fits.
//!
//! ferrolearn instead raises `FerroError::InvalidParameter` for any negative
//! `sample_weight` entry
//! (`ferrolearn-linear/src/logistic_regression.rs:310-315`:
//!   `if sw.iter().any(|&w| w < F::zero()) { return Err(InvalidParameter ... ) }`).
//!
//! Tracking: #2171

use ferrolearn_core::HasCoefficients;
use ferrolearn_linear::LogisticRegression;
use ndarray::{Array1, Array2, array};

/// ferrolearn's L-BFGS is a different implementation than scipy's, so we assert
/// agreement to the shared optimizer tolerance, not ULP equality
/// (goal.md R-DEV-1/R-DEV-7). The divergence under test is the REJECTION itself,
/// not a numeric gap.
const LBFGS_TOL: f64 = 5e-3;

/// Divergence: ferrolearn's `fit_with_sample_weight` returns `Err` for a
/// negative `sample_weight` entry; sklearn 1.5.2 fits and returns coefficients.
///
/// Live oracle (sklearn 1.5.2, `LogisticRegression(C=1.0, max_iter=5000,
/// tol=1e-9).fit(X, y, sample_weight=w)` with `w[3] = -1.0`):
///   coef_      = [0.6577080996682715, 0.6577080996682715]
///   intercept_ = [-3.7224631097545546]
///
/// sklearn ACCEPTS + fits → expected: ferrolearn returns `Ok` with matching
/// coef. ferrolearn REJECTS → this test FAILS at the `.expect()`.
#[test]
fn divergence_negative_sample_weight_rejected_binary() {
    let x: Array2<f64> = Array2::from_shape_vec(
        (8, 2),
        vec![
            1., 1., 1., 2., 2., 1., 2., 2., // class 0
            5., 5., 5., 6., 6., 5., 6., 6., // class 1
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
    // One negative weight (row 3). sklearn accepts this.
    let w: Array1<f64> = array![1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0];

    let fitted = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(5000)
        .with_tol(1e-9)
        .fit_with_sample_weight(&x, &y, Some(&w))
        // sklearn does NOT reject this; ferrolearn does. The expect pins the
        // divergence: it panics on the current `InvalidParameter` Err.
        .expect("sklearn 1.5.2 accepts negative sample_weight (#2171); ferrolearn must too");

    let coef = fitted.coefficients();
    assert!(
        (coef[0] - 0.6577080996682715).abs() < LBFGS_TOL,
        "coef0 {} vs oracle 0.6577081",
        coef[0]
    );
    assert!(
        (coef[1] - 0.6577080996682715).abs() < LBFGS_TOL,
        "coef1 {} vs oracle 0.6577081",
        coef[1]
    );
    assert!(
        (fitted.intercept() - (-3.7224631097545546)).abs() < 5e-2,
        "intercept {} vs oracle -3.7224631",
        fitted.intercept()
    );
}
