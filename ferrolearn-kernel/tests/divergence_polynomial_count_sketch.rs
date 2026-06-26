//! Green-guard + documented-divergence tests for
//! `ferrolearn-kernel::PolynomialCountSketch` against scikit-learn 1.5.2
//! `sklearn.kernel_approximation.PolynomialCountSketch`
//! (`sklearn/kernel_approximation.py:40-236`).
//!
//! The dense transform formula is pinned in the in-crate unit test with fixed
//! sklearn hash arrays. Exact `indexHash_` / `bitHash_` values remain an
//! RNG-substrate gap because ferrolearn uses Xoshiro while sklearn uses numpy
//! `RandomState`.

use ferrolearn_core::{Fit, Transform};
use ferrolearn_kernel::PolynomialCountSketch;
use ndarray::{Array2, array};

#[test]
fn green_polynomial_count_sketch_public_fit_transform_surface() {
    // sklearn source: fit creates indexHash_ and bitHash_ with shape
    // `(degree, n_features [+ bias])`; transform returns
    // `(n_samples, n_components)`.
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let fitted = PolynomialCountSketch::<f64>::new()
        .with_gamma(0.5)
        .with_degree(3)
        .with_coef0(2.0)
        .with_n_components(6)
        .with_random_state(42)
        .fit(&x, &())
        .unwrap();

    assert_eq!(fitted.n_features_in(), 2);
    assert_eq!(fitted.degree(), 3);
    assert_eq!(fitted.n_components(), 6);
    assert_eq!(fitted.index_hash().dim(), (3, 3));
    assert_eq!(fitted.bit_hash().dim(), (3, 3));
    for &index in fitted.index_hash() {
        assert!(index < 6);
    }
    for &sign in fitted.bit_hash() {
        assert!(sign == -1.0 || sign == 1.0);
    }

    let z = fitted.transform(&x).unwrap();
    assert_eq!(z.dim(), (2, 6));
    for &value in &z {
        assert!(value.is_finite());
    }
}

#[test]
fn green_polynomial_count_sketch_validation_boundaries() {
    let x = array![[1.0, 2.0]];

    // sklearn parameter constraints require finite gamma >= 0, degree >= 1,
    // finite coef0, and n_components >= 1.
    assert!(
        PolynomialCountSketch::<f64>::new()
            .with_gamma(-1.0)
            .fit(&x, &())
            .is_err()
    );
    assert!(
        PolynomialCountSketch::<f64>::new()
            .with_degree(0)
            .fit(&x, &())
            .is_err()
    );
    assert!(
        PolynomialCountSketch::<f64>::new()
            .with_coef0(f64::INFINITY)
            .fit(&x, &())
            .is_err()
    );
    assert!(
        PolynomialCountSketch::<f64>::new()
            .with_n_components(0)
            .fit(&x, &())
            .is_err()
    );

    // sklearn validate_data rejects zero samples, zero features, non-finite
    // input, and fitted feature-count mismatches.
    assert!(
        PolynomialCountSketch::<f64>::new()
            .fit(&Array2::<f64>::zeros((0, 2)), &())
            .is_err()
    );
    assert!(
        PolynomialCountSketch::<f64>::new()
            .fit(&Array2::<f64>::zeros((2, 0)), &())
            .is_err()
    );
    assert!(
        PolynomialCountSketch::<f64>::new()
            .fit(&array![[f64::NAN, 1.0]], &())
            .is_err()
    );

    let fitted = PolynomialCountSketch::<f64>::new()
        .with_n_components(4)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();
    assert!(fitted.transform(&array![[1.0, 2.0, 3.0]]).is_err());
    assert!(fitted.transform(&array![[f64::NAN, 2.0]]).is_err());
    assert!(fitted.transform(&Array2::<f64>::zeros((0, 2))).is_err());
}

#[test]
fn green_polynomial_count_sketch_random_state_reproducible_in_rust_substrate() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let fitted_a = PolynomialCountSketch::<f64>::new()
        .with_degree(3)
        .with_n_components(5)
        .with_random_state(7)
        .fit(&x, &())
        .unwrap();
    let fitted_b = PolynomialCountSketch::<f64>::new()
        .with_degree(3)
        .with_n_components(5)
        .with_random_state(7)
        .fit(&x, &())
        .unwrap();

    assert_eq!(fitted_a.index_hash(), fitted_b.index_hash());
    assert_eq!(fitted_a.bit_hash(), fitted_b.bit_hash());
}
