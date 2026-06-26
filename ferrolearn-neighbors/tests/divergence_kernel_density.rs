//! Green guards for `KernelDensity` against scikit-learn's
//! `sklearn.neighbors.KernelDensity`.
//!
//! Expected values come from the live sklearn 1.5.2 oracle in this workspace:
//!
//! ```text
//! import numpy as np
//! from sklearn.neighbors import KernelDensity
//! X = np.array([[0.0], [1.0], [2.0]])
//! Q = np.array([[0.0], [1.5]])
//! kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(X)
//! kde.score_samples(Q), kde.score(Q)
//! # [-1.4625939022307919, -1.2805560178145314], -2.7431499200453233
//! X2 = np.array([[0., 0.], [1., 0.], [0., 1.]])
//! Q2 = np.array([[0., 0.], [1., 1.]])
//! kde2 = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X2)
//! kde2.score_samples(Q2), kde2.score(Q2)
//! # [-1.31065022773568, -2.7915713182780513], -4.1022215460137312
//! ```
//!
//! Remaining sklearn contract gaps are documented in `GAP-REPORT.md`: no
//! sample weights, non-Gaussian kernels, string bandwidth rules, tree tolerance
//! controls, sparse/precomputed distances, `sample`, Python attributes, or exact
//! sklearn exception type/message parity.

use approx::assert_relative_eq;
use ferrolearn_core::Fit;
use ferrolearn_neighbors::{Algorithm, KernelDensity};
use ndarray::array;

#[test]
fn green_kernel_density_gaussian_score_samples_1d_match_sklearn() {
    let x = array![[0.0], [1.0], [2.0]];
    let q = array![[0.0], [1.5]];
    let fitted = KernelDensity::<f64>::new()
        .with_bandwidth(1.0)
        .fit(&x, &())
        .unwrap();

    let scores = fitted.score_samples(&q).unwrap();

    assert_relative_eq!(scores[0], -1.4625939022307919, epsilon = 1e-12);
    assert_relative_eq!(scores[1], -1.2805560178145314, epsilon = 1e-12);
    assert_relative_eq!(
        fitted.score(&q).unwrap(),
        -2.7431499200453233,
        epsilon = 1e-12
    );
}

#[test]
fn green_kernel_density_gaussian_score_samples_2d_bandwidth_match_sklearn() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let q = array![[0.0, 0.0], [1.0, 1.0]];
    let fitted = KernelDensity::<f64>::new()
        .with_bandwidth(0.5)
        .with_algorithm(Algorithm::KdTree)
        .fit(&x, &())
        .unwrap();

    let scores = fitted.score_samples(&q).unwrap();

    assert_eq!(fitted.n_features_in(), 2);
    assert_eq!(fitted.n_samples_fit(), 3);
    assert_relative_eq!(scores[0], -1.31065022773568, epsilon = 1e-12);
    assert_relative_eq!(scores[1], -2.7915713182780513, epsilon = 1e-12);
    assert_relative_eq!(
        fitted.score(&q).unwrap(),
        -4.102_221_546_013_731,
        epsilon = 1e-12
    );
}

#[test]
fn green_kernel_density_validation_boundaries_are_clean_errors() {
    let x = array![[0.0], [1.0], [2.0]];

    assert!(
        KernelDensity::<f64>::new()
            .with_bandwidth(0.0)
            .fit(&x, &())
            .is_err(),
        "sklearn rejects bandwidth=0 before fitting"
    );
    assert!(
        KernelDensity::<f64>::new()
            .fit(&array![[f64::NAN]], &())
            .is_err(),
        "sklearn rejects NaN training data"
    );

    let fitted = KernelDensity::<f64>::new().fit(&x, &()).unwrap();
    assert!(
        fitted.score_samples(&array![[f64::NAN]]).is_err(),
        "sklearn rejects NaN query data"
    );
    assert!(
        fitted.score_samples(&array![[0.0, 1.0]]).is_err(),
        "sklearn rejects query feature-count mismatch"
    );
}
