//! Divergence: `RadiusNeighbors{Classifier,Regressor}::fit` rejects `radius == 0`,
//! but sklearn's `radius` parameter constraint is `Interval(Real, 0, None,
//! closed="both")` (`sklearn/neighbors/_base.py:397`:
//! `"radius": [Interval(Real, 0, None, closed="both"), None]`), so `radius == 0`
//! is a VALID parameter — only a NEGATIVE radius is rejected by
//! `estimator._validate_params()` (`sklearn/base.py:1466`).
//!
//! ferrolearn rejects `self.radius <= F::zero()` in both fit bodies
//! (`radius_neighbors.rs:335` classifier, `:863` regressor), i.e. it
//! over-rejects `radius == 0`. This surfaces TWO divergences:
//!
//!   (A) `radius=0` + FINITE X: sklearn fits successfully (returns a fitted
//!       estimator); ferrolearn returns `Err(InvalidParameter { name: "radius" })`.
//!
//!   (B) `radius=0` + NaN X: since `radius=0` passes sklearn's param validation,
//!       sklearn proceeds to `_validate_data` and raises the FINITENESS
//!       `ValueError("Input X contains NaN.")`. ferrolearn's `radius <= 0` param
//!       check fires FIRST (the #2273 reorder placed it above the finiteness
//!       guard), so ferrolearn returns `Err(InvalidParameter { name: "radius" })`
//!       — the WRONG error category. This is a param-ordering grid case the
//!       #2273 reorder enshrined incorrectly (it ordered a NON-EXISTENT param
//!       constraint before finiteness).
//!
//! Live sklearn 1.5.2 oracle (run from `/tmp`, R-CHAR-3 — NEVER copied from
//! ferrolearn):
//! ```text
//! import numpy as np
//! from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor
//! Xf = np.array([[0.,0.],[1.,1.],[5.,5.]]); yc=[0,0,1]; yr=[0.,0.,1.]
//! RadiusNeighborsClassifier(radius=0.0).fit(Xf, yc)   # -> fitted, classes_=[0 1]
//! RadiusNeighborsRegressor(radius=0.0).fit(Xf, yr)    # -> fitted (no error)
//! print(RadiusNeighborsClassifier._parameter_constraints['radius'][0].left,
//!       RadiusNeighborsClassifier._parameter_constraints['radius'][0].closed)
//!   # -> 0 both
//! Xn = np.array([[np.nan,0.],[1.,1.],[5.,5.]])
//! RadiusNeighborsClassifier(radius=0.0).fit(Xn, yc)
//!   # -> ValueError: Input X contains NaN.   (NOT a radius param error)
//! ```

use ferrolearn_core::Fit;
use ferrolearn_core::error::FerroError;
use ferrolearn_neighbors::{RadiusNeighborsClassifier, RadiusNeighborsRegressor};
use ndarray::{Array1, Array2, array};

fn x_finite() -> Array2<f64> {
    array![[0.0f64, 0.0], [1.0, 1.0], [5.0, 5.0]]
}
fn x_nan() -> Array2<f64> {
    array![[f64::NAN, 0.0], [1.0, 1.0], [5.0, 5.0]]
}

/// (A) classifier: `radius=0` + finite X — sklearn fits; ferrolearn errors.
/// sklearn `RadiusNeighborsClassifier(radius=0.0).fit(Xf, yc)` returns a fitted
/// estimator (`radius` interval `[0, None)` closed="both",
/// `sklearn/neighbors/_base.py:397`). ferrolearn rejects `radius <= 0`
/// (`radius_neighbors.rs:335`).
/// Tracking: #2275
#[test]
#[ignore = "divergence: sklearn radius=0 is a VALID param (Interval closed=both, _base.py:397); ferrolearn rejects radius<=0; tracking #2275"]
fn radius_classifier_zero_finite_fits_in_sklearn() {
    let y: Array1<usize> = array![0usize, 0, 1];
    let res = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(0.0)
        .fit(&x_finite(), &y);
    // sklearn returns a fitted estimator; ferrolearn must NOT reject radius=0.
    assert!(
        res.is_ok(),
        "sklearn fits RadiusNeighborsClassifier(radius=0.0) on finite X \
         (radius interval is closed-both at 0, _base.py:397); ferrolearn \
         returned {res:?}"
    );
}

/// (A) regressor: `radius=0` + finite X — sklearn fits; ferrolearn errors.
/// Tracking: #2275
#[test]
#[ignore = "divergence: sklearn radius=0 is a VALID param (Interval closed=both, _base.py:397); ferrolearn rejects radius<=0; tracking #2275"]
fn radius_regressor_zero_finite_fits_in_sklearn() {
    let y: Array1<f64> = array![0.0f64, 0.0, 1.0];
    let res = RadiusNeighborsRegressor::<f64>::new()
        .with_radius(0.0)
        .fit(&x_finite(), &y);
    assert!(
        res.is_ok(),
        "sklearn fits RadiusNeighborsRegressor(radius=0.0) on finite X; \
         ferrolearn returned {res:?}"
    );
}

/// (B) classifier: `radius=0` + NaN X — sklearn raises the FINITENESS error
/// (radius=0 passes param validation), ferrolearn raises a `radius` param error
/// (the #2273 reorder put the spurious `radius <= 0` check before finiteness).
/// sklearn: `ValueError("Input X contains NaN.")` =>
/// `FerroError::InvalidParameter { name: "X", .. }`. ferrolearn returns
/// `InvalidParameter { name: "radius", .. }`.
/// Tracking: #2275
#[test]
#[ignore = "divergence: radius=0 valid in sklearn so NaN X gives finiteness err (name=X); ferrolearn gives radius param err; tracking #2275"]
fn radius_classifier_zero_nan_is_finiteness_error_not_radius() {
    let y: Array1<usize> = array![0usize, 0, 1];
    let err = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(0.0)
        .fit(&x_nan(), &y)
        .expect_err("either error path; sklearn raises ValueError on NaN");
    // sklearn names the DATA (finiteness ValueError), not `radius` — because
    // radius=0 is a valid param and never reaches `InvalidParameterError`.
    match err {
        FerroError::InvalidParameter { name, .. } => assert_eq!(
            name, "X",
            "sklearn raises the finiteness ValueError(Input X contains NaN.) for \
             radius=0 + NaN (radius=0 is a valid param, _base.py:397); ferrolearn \
             named `{name}` (the spurious radius<=0 check fired first)"
        ),
        other => panic!("expected InvalidParameter, got {other:?}"),
    }
}
