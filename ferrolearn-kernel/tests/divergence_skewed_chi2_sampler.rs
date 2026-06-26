//! Green-guard + documented-divergence tests for
//! `ferrolearn-kernel::SkewedChi2Sampler` against scikit-learn 1.5.2
//! `sklearn.kernel_approximation.SkewedChi2Sampler`
//! (`sklearn/kernel_approximation.py:425-582`).
//!
//! The transform formula is pinned in the in-crate unit test with fixed
//! sklearn component arrays. This integration file covers the public fitted
//! surface and validation boundaries. Exact `random_weights_` /
//! `random_offset_` values remain an RNG-substrate gap, the same class of
//! carve-out documented for `RBFSampler`.

use ferrolearn_core::{Fit, Transform};
use ferrolearn_kernel::SkewedChi2Sampler;
use ndarray::array;

#[test]
fn green_skewed_chi2_public_fit_transform_surface() {
    // sklearn 1.5.2 source: fit samples random_weights_ shape
    // `(n_features, n_components)` and random_offset_ shape `(n_components,)`;
    // transform returns `(n_samples, n_components)` scaled by sqrt(2/n).
    let x = array![[0.0, 1.0], [2.0, 3.0]];
    let fitted = SkewedChi2Sampler::<f64>::new()
        .with_skewedness(0.5)
        .with_n_components(4)
        .with_random_state(42)
        .fit(&x, &())
        .unwrap();

    assert_eq!(fitted.n_features_in(), 2);
    assert_eq!(fitted.n_components(), 4);
    assert_eq!(fitted.random_weights().dim(), (2, 4));
    assert_eq!(fitted.random_offset().len(), 4);
    for &offset in fitted.random_offset() {
        assert!(offset >= 0.0);
        assert!(offset < 2.0 * std::f64::consts::PI);
    }

    let z = fitted.transform(&x).unwrap();
    assert_eq!(z.dim(), (2, 4));
    let scale = (2.0_f64 / 4.0).sqrt();
    for &value in &z {
        assert!(value.is_finite());
        assert!(value.abs() <= scale + 1e-12);
    }
}

#[test]
fn green_skewed_chi2_validation_boundaries() {
    let x = array![[0.0, 1.0], [2.0, 3.0]];

    // sklearn parameter constraints require n_components >= 1.
    assert!(
        SkewedChi2Sampler::<f64>::new()
            .with_n_components(0)
            .fit(&x, &())
            .is_err()
    );

    let fitted = SkewedChi2Sampler::<f64>::new()
        .with_skewedness(0.5)
        .with_n_components(3)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();

    // sklearn transform rejects values <= -skewedness:
    // "X may not contain entries smaller than -skewedness."
    assert!(fitted.transform(&array![[-0.5, 0.0]]).is_err());
    assert!(fitted.transform(&array![[-0.6, 0.0]]).is_err());

    // The fitted transformer also validates feature count, matching sklearn's
    // `validate_data(..., reset=False)` fitted-shape check.
    assert!(fitted.transform(&array![[0.0, 1.0, 2.0]]).is_err());
}

#[test]
fn green_skewed_chi2_random_state_reproducible_in_rust_substrate() {
    let x = array![[0.0, 1.0], [2.0, 3.0]];
    let fitted_a = SkewedChi2Sampler::<f64>::new()
        .with_n_components(4)
        .with_random_state(7)
        .fit(&x, &())
        .unwrap();
    let fitted_b = SkewedChi2Sampler::<f64>::new()
        .with_n_components(4)
        .with_random_state(7)
        .fit(&x, &())
        .unwrap();

    assert_eq!(fitted_a.random_weights(), fitted_b.random_weights());
    assert_eq!(fitted_a.random_offset(), fitted_b.random_offset());
}
