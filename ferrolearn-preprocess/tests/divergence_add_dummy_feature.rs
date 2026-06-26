//! Divergence audit: ferrolearn `add_dummy_feature` vs scikit-learn 1.5.2
//! `sklearn/preprocessing/_data.py` `add_dummy_feature` (`:2594`).
//!
//! The shipped Rust surface covers the dense `Array2` path. sklearn's sparse
//! CSR/CSC/COO preserving path remains out of scope for this helper.

use ferrolearn_core::error::FerroError;
use ferrolearn_preprocess::add_dummy_feature;
use ndarray::{Array2, array};

/// GREEN: default dummy value `1.0` is prepended as the first column and the
/// original columns shift right. Live sklearn test oracle:
/// `add_dummy_feature([[1,0],[0,1],[0,1]])`
/// -> `[[1,1,0],[1,0,1],[1,0,1]]`.
#[test]
fn green_default_value_matches_sklearn() {
    let x = array![[1.0_f64, 0.0], [0.0, 1.0], [0.0, 1.0]];
    let out = add_dummy_feature(&x, 1.0).unwrap();
    let sk = array![[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]];
    assert_eq!(out.shape(), &[3, 3]);
    for (a, b) in out.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-12, "got {a}, sklearn {b}");
    }
}

/// GREEN: custom `value` fills the dummy feature column.
/// Live oracle:
/// `add_dummy_feature([[2.,3.],[4.,5.]], value=2.5)`
/// -> `[[2.5,2.,3.],[2.5,4.,5.]]`.
#[test]
fn green_custom_value_matches_sklearn() {
    let x = array![[2.0_f64, 3.0], [4.0, 5.0]];
    let out = add_dummy_feature(&x, 2.5).unwrap();
    let sk = array![[2.5, 2.0, 3.0], [2.5, 4.0, 5.0]];
    for (a, b) in out.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-12, "got {a}, sklearn {b}");
    }
}

/// GREEN: the helper returns a new array and leaves the input matrix unchanged.
#[test]
fn green_input_is_not_modified() {
    let x = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let original = x.clone();
    let _ = add_dummy_feature(&x, -1.0).unwrap();
    assert_eq!(x, original);
}

/// GREEN: zero samples map to the crate's `InsufficientSamples`, matching
/// sklearn's `check_array` minimum-sample rejection.
#[test]
fn green_zero_rows_errors() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let err = add_dummy_feature(&x, 1.0).unwrap_err();
    assert!(matches!(err, FerroError::InsufficientSamples { .. }));
}

/// GREEN: zero features are rejected, matching sklearn `check_array`'s default
/// minimum-feature validation.
#[test]
fn green_zero_features_errors() {
    let x: Array2<f64> = Array2::zeros((2, 0));
    let err = add_dummy_feature(&x, 1.0).unwrap_err();
    assert!(matches!(err, FerroError::InvalidParameter { .. }));
}

/// GREEN: dense input validation rejects NaN/Inf before adding the feature,
/// matching sklearn `check_array(..., dtype=FLOAT_DTYPES)` defaults.
#[test]
fn green_non_finite_input_errors() {
    let x = array![[1.0_f64, f64::NAN], [2.0, 3.0]];
    let err = add_dummy_feature(&x, 1.0).unwrap_err();
    assert!(matches!(err, FerroError::InvalidParameter { .. }));

    let x = array![[1.0_f64, f64::INFINITY], [2.0, 3.0]];
    let err = add_dummy_feature(&x, 1.0).unwrap_err();
    assert!(matches!(err, FerroError::InvalidParameter { .. }));
}

/// GREEN: sklearn does not validate `value` through `check_array`; it is only
/// used to fill the prepended column. Keep that behavior for NaN values.
#[test]
fn green_non_finite_dummy_value_is_allowed() {
    let x = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let out = add_dummy_feature(&x, f64::NAN).unwrap();
    assert!(out[[0, 0]].is_nan());
    assert!(out[[1, 0]].is_nan());
    assert!((out[[0, 1]] - 1.0).abs() < 1e-12);
    assert!((out[[1, 2]] - 4.0).abs() < 1e-12);
}
