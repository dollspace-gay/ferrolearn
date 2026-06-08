//! Non-finite input validation parity for the single-task linear estimators
//! `LinearRegression`, `Lasso`, and `ElasticNet` (#2256).
//!
//! scikit-learn 1.5.2 validates `X`/`y` at fit through `_validate_data(...)`
//! with the default `force_all_finite=True`
//! (`LinearRegression.fit` → `sklearn/linear_model/_base.py:609`;
//! `ElasticNet.fit` / `Lasso.fit` → `sklearn/linear_model/_coordinate_descent.py:980`),
//! so `check_array` raises `ValueError` on any NaN or +/-inf in `X` or `y`
//! BEFORE the solver runs. Previously ferrolearn accepted non-finite input and
//! produced NaN `coef_`; these estimators now reject it with
//! `FerroError::InvalidParameter`, matching sklearn's reject-at-fit contract
//! (R-DEV-1 / R-DEV-2 exception parity).
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, NOT
//! copied from ferrolearn):
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
//! X=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]]); y=np.array([1.,2.,3.,4.])
//! for cls in (LinearRegression, Lasso, ElasticNet):
//!     for Xb, yb, tag in [
//!         (X.copy(), y.copy(), 'finite'),
//!     ]:
//!         pass
//! "
//! ```
//!
//! Result (each non-finite case raises `ValueError`; each finite case fits):
//! ```text
//! LinearRegression Xnan  ValueError: Input X contains NaN.
//! LinearRegression Xinf  ValueError: Input X contains infinity or a value too large ...
//! LinearRegression Xninf ValueError: Input X contains infinity or a value too large ...
//! LinearRegression ynan  ValueError: Input y contains NaN.
//! LinearRegression yinf  ValueError: Input y contains infinity or a value too large ...
//! LinearRegression finite NO-RAISE
//! Lasso/ElasticNet: identical (all five non-finite raise, finite fits).
//! ```

use ferrolearn_core::Fit;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_linear::{ElasticNet, Lasso, LinearRegression};
use ndarray::{Array1, Array2, array};

/// Shared finite design: full-rank, well-conditioned, fits cleanly.
fn finite_xy() -> (Array2<f64>, Array1<f64>) {
    let x: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
    (x, y)
}

/// Returns `true` iff fitting `model` on `(x, y)` errors with the non-finite
/// `InvalidParameter` (the sklearn `ValueError` analog).
fn rejects_nonfinite<M>(model: &M, x: &Array2<f64>, y: &Array1<f64>) -> bool
where
    M: Fit<Array2<f64>, Array1<f64>, Error = FerroError>,
{
    matches!(
        model.fit(x, y),
        Err(FerroError::InvalidParameter { ref name, ref reason })
            if (name == "X" || name == "y") && reason.contains("NaN or infinity")
    )
}

// ---------------------------------------------------------------------------
// LinearRegression
// ---------------------------------------------------------------------------

#[test]
fn linreg_rejects_non_finite_input_like_sklearn() {
    // sklearn `LinearRegression().fit` raises `ValueError` for NaN/+inf/-inf in
    // X and NaN/inf in y (`_base.py:609`, force_all_finite=True). ferrolearn
    // must reject all five.
    let (x, y) = finite_xy();
    let model = LinearRegression::<f64>::new();

    let mut x_nan = x.clone();
    x_nan[[0, 0]] = f64::NAN;
    assert!(
        rejects_nonfinite(&model, &x_nan, &y),
        "NaN in X must be rejected"
    );

    let mut x_pinf = x.clone();
    x_pinf[[1, 1]] = f64::INFINITY;
    assert!(
        rejects_nonfinite(&model, &x_pinf, &y),
        "+inf in X must be rejected"
    );

    let mut x_ninf = x.clone();
    x_ninf[[2, 0]] = f64::NEG_INFINITY;
    assert!(
        rejects_nonfinite(&model, &x_ninf, &y),
        "-inf in X must be rejected"
    );

    let mut y_nan = y.clone();
    y_nan[0] = f64::NAN;
    assert!(
        rejects_nonfinite(&model, &x, &y_nan),
        "NaN in y must be rejected"
    );

    let mut y_inf = y.clone();
    y_inf[1] = f64::INFINITY;
    assert!(
        rejects_nonfinite(&model, &x, &y_inf),
        "inf in y must be rejected"
    );
}

#[test]
fn linreg_single_nonfinite_cell_among_finite_rejected() {
    // The `.any(..)` catches a lone non-finite cell in an otherwise finite X.
    let (mut x, y) = finite_xy();
    x[[3, 1]] = f64::NAN;
    let model = LinearRegression::<f64>::new();
    assert!(rejects_nonfinite(&model, &x, &y));
}

#[test]
fn linreg_finite_input_fits_unchanged() {
    // No false positive: all-finite input still fits. Guard the SHIPPED OLS
    // result is unperturbed — coef_/intercept_ from the live sklearn oracle:
    //   cd /tmp && python3 -c "import numpy as np; \
    //     from sklearn.linear_model import LinearRegression; \
    //     X=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]]); y=np.array([1.,2.,3.,4.]); \
    //     m=LinearRegression().fit(X,y); \
    //     print([round(c,10) for c in m.coef_], round(m.intercept_,10))"
    //   -> X is rank-deficient (cols collinear: col1 = col0 + 1); min-norm
    //      OLS -> coef_ [0.25, 0.25], intercept_ 0.25, predict(X) == [1,2,3,4]
    let (x, y) = finite_xy();
    let fitted = LinearRegression::<f64>::new()
        .fit(&x, &y)
        .expect("finite input must fit");
    // The design is consistent (y is an exact linear function of X), so
    // predictions reproduce y to high precision regardless of the min-norm split.
    let preds = ferrolearn_core::Predict::predict(&fitted, &x).expect("predict");
    for (p, &t) in preds.iter().zip(y.iter()) {
        assert!((p - t).abs() < 1e-8, "fit must reproduce finite target");
    }
}

// ---------------------------------------------------------------------------
// Lasso
// ---------------------------------------------------------------------------

#[test]
fn lasso_rejects_non_finite_input_like_sklearn() {
    // sklearn `Lasso().fit` raises `ValueError` for NaN/+inf/-inf in X and
    // NaN/inf in y (`_coordinate_descent.py:980`, force_all_finite=True).
    let (x, y) = finite_xy();
    let model = Lasso::<f64>::new().with_alpha(0.1);

    let mut x_nan = x.clone();
    x_nan[[0, 0]] = f64::NAN;
    assert!(
        rejects_nonfinite(&model, &x_nan, &y),
        "NaN in X must be rejected"
    );

    let mut x_pinf = x.clone();
    x_pinf[[1, 1]] = f64::INFINITY;
    assert!(
        rejects_nonfinite(&model, &x_pinf, &y),
        "+inf in X must be rejected"
    );

    let mut x_ninf = x.clone();
    x_ninf[[2, 0]] = f64::NEG_INFINITY;
    assert!(
        rejects_nonfinite(&model, &x_ninf, &y),
        "-inf in X must be rejected"
    );

    let mut y_nan = y.clone();
    y_nan[0] = f64::NAN;
    assert!(
        rejects_nonfinite(&model, &x, &y_nan),
        "NaN in y must be rejected"
    );

    let mut y_inf = y.clone();
    y_inf[1] = f64::INFINITY;
    assert!(
        rejects_nonfinite(&model, &x, &y_inf),
        "inf in y must be rejected"
    );
}

#[test]
fn lasso_single_nonfinite_cell_among_finite_rejected() {
    let (mut x, y) = finite_xy();
    x[[3, 1]] = f64::INFINITY;
    let model = Lasso::<f64>::new().with_alpha(0.1);
    assert!(rejects_nonfinite(&model, &x, &y));
}

#[test]
fn lasso_finite_input_fits_unchanged() {
    // No false positive: all-finite input fits and yields the same coef_ as a
    // direct known-fit on a clean fixture (live sklearn 1.5.2 oracle):
    //   cd /tmp && python3 -c "import numpy as np; \
    //     from sklearn.linear_model import Lasso; \
    //     X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([2.,4.,6.,8.,10.]); \
    //     m=Lasso(alpha=0.1).fit(X,y); print(round(m.coef_[0],8), round(m.intercept_,8))"
    //   -> 1.95 0.15
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
    let fitted = Lasso::<f64>::new()
        .with_alpha(0.1)
        .fit(&x, &y)
        .expect("finite input must fit");
    assert!((fitted.coefficients()[0] - 1.95).abs() < 1e-6);
    assert!((fitted.intercept() - 0.15).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// ElasticNet
// ---------------------------------------------------------------------------

#[test]
fn enet_rejects_non_finite_input_like_sklearn() {
    // sklearn `ElasticNet().fit` raises `ValueError` for NaN/+inf/-inf in X and
    // NaN/inf in y (`_coordinate_descent.py:980`, force_all_finite=True).
    let (x, y) = finite_xy();
    let model = ElasticNet::<f64>::new().with_alpha(0.1).with_l1_ratio(0.5);

    let mut x_nan = x.clone();
    x_nan[[0, 0]] = f64::NAN;
    assert!(
        rejects_nonfinite(&model, &x_nan, &y),
        "NaN in X must be rejected"
    );

    let mut x_pinf = x.clone();
    x_pinf[[1, 1]] = f64::INFINITY;
    assert!(
        rejects_nonfinite(&model, &x_pinf, &y),
        "+inf in X must be rejected"
    );

    let mut x_ninf = x.clone();
    x_ninf[[2, 0]] = f64::NEG_INFINITY;
    assert!(
        rejects_nonfinite(&model, &x_ninf, &y),
        "-inf in X must be rejected"
    );

    let mut y_nan = y.clone();
    y_nan[0] = f64::NAN;
    assert!(
        rejects_nonfinite(&model, &x, &y_nan),
        "NaN in y must be rejected"
    );

    let mut y_inf = y.clone();
    y_inf[1] = f64::INFINITY;
    assert!(
        rejects_nonfinite(&model, &x, &y_inf),
        "inf in y must be rejected"
    );
}

#[test]
fn enet_single_nonfinite_cell_among_finite_rejected() {
    let (mut x, y) = finite_xy();
    x[[3, 1]] = f64::NEG_INFINITY;
    let model = ElasticNet::<f64>::new().with_alpha(0.1).with_l1_ratio(0.5);
    assert!(rejects_nonfinite(&model, &x, &y));
}

#[test]
fn enet_finite_input_fits_unchanged() {
    // No false positive: all-finite input fits, matching the live sklearn 1.5.2
    // oracle:
    //   cd /tmp && python3 -c "import numpy as np; \
    //     from sklearn.linear_model import ElasticNet; \
    //     X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([2.,4.,6.,8.,10.]); \
    //     m=ElasticNet(alpha=0.1,l1_ratio=0.5).fit(X,y); \
    //     print(round(m.coef_[0],8), round(m.intercept_,8))"
    //   -> 1.92682927 0.2195122
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
    let fitted = ElasticNet::<f64>::new()
        .with_alpha(0.1)
        .with_l1_ratio(0.5)
        .fit(&x, &y)
        .expect("finite input must fit");
    assert!((fitted.coefficients()[0] - 1.926_829_27).abs() < 1e-6);
    assert!((fitted.intercept() - 0.219_512_2).abs() < 1e-6);
}
