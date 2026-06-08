//! Validation-ORDER pins: invalid constructor param + non-finite X for the
//! `ferrolearn-neighbors` estimators vs the LIVE scikit-learn 1.5.2 oracle.
//!
//! The #2272 non-finite guards were placed FIRST inside each `fit` body, ABOVE
//! the parameter-constraint checks (`n_neighbors == 0`, `shrink_threshold <=
//! 0`). sklearn does the OPPOSITE: the `@_fit_context` decorator calls
//! `estimator._validate_params()` BEFORE the wrapped `fit` method body that
//! performs `_validate_data` finiteness checking
//! (`sklearn/base.py:1466`: `estimator._validate_params()` then
//! `:1472`: `return fit_method(...)`). So for a doubly-degenerate input
//! (invalid param AND NaN/inf in X) sklearn raises `InvalidParameterError`
//! naming the PARAMETER, not the data-finiteness `ValueError`.
//!
//! ferrolearn returns `InvalidParameter { name: "X", reason: "Input X contains
//! NaN or infinity." }` (the finiteness guard fires first) where sklearn names
//! the offending constructor parameter. This is the inverse of the SVM
//! finiteness-vs-length ordering divergence pinned in #2270.
//!
//! Live sklearn 1.5.2 oracle (run from `/tmp`):
//! ```text
//! import numpy as np
//! from sklearn.neighbors import (KNeighborsClassifier, NearestNeighbors,
//!     LocalOutlierFactor, NearestCentroid)
//! X = np.array([[0.,0.],[1.,1.],[2.,2.]]); X[0,0] = np.nan
//! y = np.array([0,1,0])
//!
//! KNeighborsClassifier(n_neighbors=0).fit(X, y)
//!   -> InvalidParameterError: The 'n_neighbors' parameter of
//!      KNeighborsClassifier must be ...   (NOT "Input X contains NaN.")
//! NearestNeighbors(n_neighbors=0).fit(X)
//!   -> InvalidParameterError: The 'n_neighbors' parameter of NearestNeighbors ...
//! LocalOutlierFactor(n_neighbors=0).fit(X)
//!   -> InvalidParameterError: The 'n_neighbors' parameter of LocalOutlierFactor ...
//! NearestCentroid(shrink_threshold=0.0).fit(X, y)
//!   -> InvalidParameterError: The 'shrink_threshold' parameter of NearestCentroid ...
//! NearestCentroid(shrink_threshold=-1.0).fit(X, y)
//!   -> InvalidParameterError: The 'shrink_threshold' parameter of NearestCentroid ...
//! ```
//!
//! Every expected behavior is the LIVE sklearn oracle (goal.md R-CHAR-3),
//! NEVER copied from the ferrolearn side.

use ferrolearn_core::Fit;
use ferrolearn_core::error::FerroError;
use ferrolearn_neighbors::{
    KNeighborsClassifier, LocalOutlierFactor, NearestCentroid, NearestNeighbors,
};
use ndarray::{Array1, Array2, array};

/// 3-point fixture with a NaN in X[0,0] (so the data is non-finite).
fn x_nan() -> Array2<f64> {
    array![[f64::NAN, 0.0], [1.0, 1.0], [2.0, 2.0]]
}

fn y3() -> Array1<usize> {
    array![0usize, 1, 0]
}

/// Divergence: `KNeighborsClassifier::fit` with `n_neighbors == 0` AND NaN in X
/// returns the X-finiteness error, but sklearn validates the PARAMETER first.
/// sklearn (`sklearn/base.py:1466` `_validate_params()` before the `fit` body)
/// raises `InvalidParameterError` naming `n_neighbors`; ferrolearn's finiteness
/// guard (knn.rs:794, ABOVE the `n_neighbors == 0` check at knn.rs:811) returns
/// `InvalidParameter { name: "X" }`.
/// Tracking: #2273
#[test]
fn knn_classifier_param_before_finiteness_order() {
    let err = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(0)
        .fit(&x_nan(), &y3())
        .expect_err("n_neighbors=0 + NaN X must error");
    // sklearn names the PARAMETER (n_neighbors), not the data (X).
    match err {
        FerroError::InvalidParameter { name, .. } => assert_eq!(
            name, "n_neighbors",
            "sklearn raises InvalidParameterError on n_neighbors (base.py:1466 \
             _validate_params runs before _validate_data finiteness); \
             ferrolearn returned name={name:?}"
        ),
        other => panic!("expected InvalidParameter(n_neighbors), got {other:?}"),
    }
}

/// Divergence: `NearestNeighbors::fit` with `n_neighbors == 0` AND NaN in X.
/// sklearn raises `InvalidParameterError` on `n_neighbors`
/// (`sklearn/base.py:1466`); ferrolearn's finiteness guard
/// (nearest_neighbors.rs:244) fires before the `n_neighbors == 0` check and
/// returns `InvalidParameter { name: "X" }`.
/// Tracking: #2273
#[test]
fn nearest_neighbors_param_before_finiteness_order() {
    let err = NearestNeighbors::<f64>::new()
        .with_n_neighbors(0)
        .fit(&x_nan(), &())
        .expect_err("n_neighbors=0 + NaN X must error");
    match err {
        FerroError::InvalidParameter { name, .. } => assert_eq!(
            name, "n_neighbors",
            "sklearn raises InvalidParameterError on n_neighbors before finiteness; \
             ferrolearn returned name={name:?}"
        ),
        other => panic!("expected InvalidParameter(n_neighbors), got {other:?}"),
    }
}

/// Divergence: `LocalOutlierFactor::fit` with `n_neighbors == 0` AND NaN in X.
/// sklearn raises `InvalidParameterError` on `n_neighbors`
/// (`sklearn/base.py:1466`); ferrolearn's finiteness guard
/// (local_outlier_factor.rs:352) fires first and returns
/// `InvalidParameter { name: "X" }`.
/// Tracking: #2273
#[test]
fn lof_param_before_finiteness_order() {
    let err = LocalOutlierFactor::<f64>::new()
        .with_n_neighbors(0)
        .fit(&x_nan(), &())
        .expect_err("n_neighbors=0 + NaN X must error");
    match err {
        FerroError::InvalidParameter { name, .. } => assert_eq!(
            name, "n_neighbors",
            "sklearn raises InvalidParameterError on n_neighbors before finiteness; \
             ferrolearn returned name={name:?}"
        ),
        other => panic!("expected InvalidParameter(n_neighbors), got {other:?}"),
    }
}

/// Divergence: `NearestCentroid::fit` with `shrink_threshold == 0` (invalid,
/// `closed="neither"`) AND NaN in X. sklearn validates the param first
/// (`sklearn/base.py:1466`) and raises `InvalidParameterError` naming
/// `shrink_threshold`; ferrolearn's finiteness guard (nearest_centroid.rs:185)
/// fires before the `shrink_threshold <= 0` check (nearest_centroid.rs:210) and
/// returns `InvalidParameter { name: "X" }`.
/// Tracking: #2273
#[test]
fn nearest_centroid_param_before_finiteness_order() {
    let err = NearestCentroid::<f64>::new()
        .with_shrink_threshold(0.0)
        .fit(&x_nan(), &y3())
        .expect_err("shrink_threshold=0 + NaN X must error");
    match err {
        FerroError::InvalidParameter { name, .. } => assert_eq!(
            name, "shrink_threshold",
            "sklearn raises InvalidParameterError on shrink_threshold before \
             finiteness; ferrolearn returned name={name:?}"
        ),
        other => panic!("expected InvalidParameter(shrink_threshold), got {other:?}"),
    }
}
