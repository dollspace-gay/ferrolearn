//! Divergence pin: `RANSACRegressor::with_min_samples(0)` (the integer-count
//! path) vs the live scikit-learn 1.5.2 oracle.
//!
//! sklearn constrains `min_samples` to `Interval(Integral, 1, None,
//! closed="left")` (`sklearn/linear_model/_ransac.py:263`), so `min_samples=0`
//! is rejected at parameter validation with `InvalidParameterError`
//! (a subclass of `ValueError`) BEFORE any fitting happens:
//!
//! ```text
//! python3 -c "import numpy as np; \
//!   from sklearn.linear_model import RANSACRegressor, LinearRegression; \
//!   X=np.arange(1,11).reshape(-1,1).astype(float); y=2*X.ravel()+1.0; \
//!   import sklearn; print(sklearn.__version__); \
//!   RANSACRegressor(LinearRegression(), min_samples=0).fit(X,y)"
//! # -> 1.5.2
//! # -> sklearn.utils._param_validation.InvalidParameterError:
//! #    The 'min_samples' parameter of RANSACRegressor must be an int in the
//! #    range [1, inf) ... Got 0 instead.
//! ```
//!
//! ferrolearn (`ferrolearn-linear/src/ransac.rs:420`) resolves
//! `Some(MinSamples::Count(k)) => k.max(1)`, which SILENTLY coerces a count of 0
//! to 1 and fits successfully — it never raises. sklearn rejects the parameter;
//! ferrolearn accepts and runs. The observable contract diverges: the same
//! configuration that is a hard error in sklearn is a successful fit in
//! ferrolearn (goal.md R-DEV-2 exception/defaults parity).
//!
//! This test asserts the sklearn behavior (an error). It FAILS against the
//! current ferrolearn implementation, which returns `Ok`.

use ferrolearn_core::traits::Fit;
use ferrolearn_linear::LinearRegression;
use ferrolearn_linear::ransac::RANSACRegressor;
use ndarray::{Array1, Array2};

/// Divergence: `RANSACRegressor::fit` accepts `min_samples = Count(0)` where
/// sklearn raises `InvalidParameterError` (`sklearn/linear_model/_ransac.py:263`,
/// `Interval(Integral, 1, None, closed="left")`).
///
/// sklearn (oracle, 1.5.2): `min_samples=0` -> InvalidParameterError (ValueError
/// subclass), parameter rejected before fit.
/// ferrolearn (`ransac.rs:420`): `Count(k) => k.max(1)` -> 0 silently becomes 1,
/// `fit` returns `Ok`.
///
/// Tracking: #2251
#[test]
fn ransac_min_samples_count_zero_must_error() {
    let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
    let y = Array1::from(
        (1..=10)
            .map(|i| 2.0 * f64::from(i) + 1.0)
            .collect::<Vec<_>>(),
    );

    // sklearn's `Interval(Integral, 1, None)` rejects min_samples=0 with
    // InvalidParameterError. ferrolearn's `Count(0).max(1)` accepts it.
    let model = RANSACRegressor::new(LinearRegression::<f64>::new())
        .with_min_samples(0)
        .with_random_state(0);
    let result = model.fit(&x, &y);

    assert!(
        result.is_err(),
        "min_samples=0: sklearn raises InvalidParameterError \
         (sklearn/linear_model/_ransac.py:263); ferrolearn's Count(0).max(1) \
         silently coerces to 1 and returns Ok"
    );
}
