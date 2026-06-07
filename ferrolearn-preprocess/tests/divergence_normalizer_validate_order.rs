//! ACToR divergence pin: `FittedNormalizer::transform` validation ORDER vs
//! scikit-learn 1.5.2 `Normalizer.transform`.
//!
//! sklearn `Normalizer.transform` calls
//! `self._validate_data(X, ..., reset=False)` (`sklearn/preprocessing/_data.py:2104`).
//! Inside `_validate_data` the order is fixed: `check_array(X, ...)` runs FIRST
//! (`sklearn/base.py:633`) — which enforces `force_all_finite=True`
//! (NaN/±inf), `ensure_min_samples=1`, `ensure_min_features=1` — and ONLY AFTER
//! that does `self._check_n_features(X, reset=False)` run
//! (`sklearn/base.py:654`), which compares the column count against the fitted
//! `n_features_in_`.
//!
//! Therefore, when a held-out transform input has BOTH a non-finite value
//! (or zero samples) AND a column count that differs from the fitted
//! `n_features_in_`, sklearn reports the `check_array` error (NaN/inf/samples),
//! NOT the feature-count error.
//!
//! ferrolearn's `FittedNormalizer::transform`
//! (`ferrolearn-preprocess/src/normalizer.rs:264-273`) inverts this order: it
//! checks `x.ncols() != self.n_features_in_` FIRST and returns
//! `FerroError::ShapeMismatch` before ever validating finiteness/samples (which
//! only happens inside the delegated `normalize` call). So for a NaN-bearing,
//! wrong-column input it returns `ShapeMismatch` where sklearn raises the NaN
//! `ValueError`.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::Normalizer;
use ndarray::array;

/// Divergence: `FittedNormalizer::transform` diverges from sklearn 1.5.2
/// `Normalizer.transform` validation ordering
/// (`sklearn/preprocessing/_data.py:2104` -> `sklearn/base.py:633` check_array
/// BEFORE `:654` `_check_n_features`).
///
/// Input: `fit(np.ones((2,3)))` then `transform([[nan, 1.0]])` (2 cols, NaN).
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer
/// m=Normalizer(norm='l2').fit(np.ones((2,3)))
/// m.transform(np.array([[float('nan'),1.0]]))"
/// ```
/// -> `ValueError: Input X contains NaN.` (a check_array finite error — which
///    ferrolearn maps to `FerroError::InvalidParameter`, per the REQ-2 mapping
///    used by `Normalizer::transform` / `normalize`).
///
/// ferrolearn returns `Err(FerroError::ShapeMismatch { .. })` instead, because
/// the column-count guard fires before any finite validation.
///
/// Tracking: #2207
#[test]
#[ignore = "divergence: FittedNormalizer::transform checks n_features before finite/samples, inverting sklearn check_array-then-_check_n_features order; tracking #2207"]
fn divergence_fitted_transform_finite_checked_before_n_features() {
    let x_fit = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
    let fitted = Normalizer::<f64>::l2().fit(&x_fit, &()).unwrap();

    // 2 columns (fit saw 3) AND a NaN. sklearn's check_array runs first and
    // raises the NaN error, which ferrolearn maps to InvalidParameter.
    let x = array![[f64::NAN, 1.0]];
    let result = fitted.transform(&x);

    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "sklearn check_array (force_all_finite) runs before _check_n_features, so \
         a NaN+wrong-column transform input must report the finite error \
         (FerroError::InvalidParameter); ferrolearn returned {result:?}"
    );
}
