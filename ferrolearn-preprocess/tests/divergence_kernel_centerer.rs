//! Divergence audit: ferrolearn `KernelCenterer` vs scikit-learn 1.5.2
//! `sklearn/preprocessing/_data.py` `class KernelCenterer` (`:2421`).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::KernelCenterer;
use ndarray::{Array2, array};

/// GREEN: sklearn doc example. Fitting stores column means `[3, 1, 2]` and
/// grand mean `2`, then double-centers the square training kernel.
#[test]
fn green_doc_example_fit_transform_matches_sklearn() {
    let k = array![
        [9.0_f64, 2.0, -2.0],
        [2.0, 14.0, -13.0],
        [-2.0, -13.0, 21.0]
    ];
    let out = KernelCenterer::<f64>::new().fit_transform(&k).unwrap();
    let sk = array![[5.0, 0.0, -5.0], [0.0, 14.0, -14.0], [-5.0, -14.0, 19.0]];
    for (a, b) in out.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-12, "got {a}, sklearn {b}");
    }
}

/// GREEN: fitted attributes match sklearn's `K_fit_rows_` and `K_fit_all_`.
#[test]
fn green_fit_attributes_match_sklearn() {
    let k = array![
        [9.0_f64, 2.0, -2.0],
        [2.0, 14.0, -13.0],
        [-2.0, -13.0, 21.0]
    ];
    let fitted = KernelCenterer::<f64>::new().fit(&k, &()).unwrap();
    let sk_rows = [3.0_f64, 1.0, 2.0];
    for (got, sk) in fitted.k_fit_rows().iter().zip(sk_rows.iter()) {
        assert!((got - sk).abs() < 1e-12, "got {got}, sklearn {sk}");
    }
    assert!((fitted.k_fit_all() - 2.0).abs() < 1e-12);
    assert_eq!(fitted.n_features_in(), 3);
}

/// GREEN: rectangular prediction kernels use the training column means and each
/// prediction row mean (`_data.py:2563-2567`).
#[test]
fn green_rectangular_transform_matches_sklearn_formula() {
    let k_fit = array![
        [9.0_f64, 2.0, -2.0],
        [2.0, 14.0, -13.0],
        [-2.0, -13.0, 21.0]
    ];
    let fitted = KernelCenterer::<f64>::new().fit(&k_fit, &()).unwrap();
    let k_pred = array![[10.0, -1.0, 4.0], [-2.0, 7.0, 8.0]];
    let out = fitted.transform(&k_pred).unwrap();
    let sk = array![
        [14.0 / 3.0, -13.0 / 3.0, -1.0 / 3.0],
        [-22.0 / 3.0, 11.0 / 3.0, 11.0 / 3.0],
    ];
    for (a, b) in out.iter().zip(sk.iter()) {
        assert!((a - b).abs() < 1e-12, "got {a}, sklearn {b}");
    }
}

/// GREEN: `fit` requires a square kernel matrix.
#[test]
fn green_fit_non_square_errors() {
    let k = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let err = KernelCenterer::<f64>::new().fit(&k, &()).unwrap_err();
    assert!(matches!(err, FerroError::InvalidParameter { .. }));
}

/// GREEN: transform requires the prediction-kernel column count to match the
/// fitted training sample count.
#[test]
fn green_transform_ncols_mismatch_errors() {
    let k_fit = array![[1.0_f64, 2.0], [2.0, 1.0]];
    let fitted = KernelCenterer::<f64>::new().fit(&k_fit, &()).unwrap();
    let k_bad = array![[1.0, 2.0, 3.0]];
    let err = fitted.transform(&k_bad).unwrap_err();
    assert!(matches!(err, FerroError::ShapeMismatch { .. }));
}

/// GREEN: check-array style validation rejects empty and non-finite matrices.
#[test]
fn green_validation_errors() {
    let k: Array2<f64> = Array2::zeros((0, 0));
    let err = KernelCenterer::<f64>::new().fit(&k, &()).unwrap_err();
    assert!(matches!(err, FerroError::InsufficientSamples { .. }));

    let k = array![[1.0_f64, f64::NAN], [2.0, 3.0]];
    let err = KernelCenterer::<f64>::new().fit(&k, &()).unwrap_err();
    assert!(matches!(err, FerroError::InvalidParameter { .. }));
}
