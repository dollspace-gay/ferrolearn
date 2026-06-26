//! Public `ClassicalMDS` parity against sklearn 1.9 `_classical_mds.py`.
//!
//! The installed local sklearn runtime is 1.5.2 and does not expose this public
//! estimator, so the expected values below are generated from the 1.9 source
//! algorithm: pairwise distances, double centering, scipy `linalg.eigh`, reverse
//! descending eigen order, and sklearn `svd_flip(U, None)`.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::{ClassicalMDS, Dissimilarity};
use ndarray::{Array1, Array2, array};

fn fixture_x() -> Array2<f64> {
    array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0], [1.0, 2.0]]
}

fn expected_distances() -> Array2<f64> {
    array![
        [0.0, 3.0, 4.0, 5.0, 2.236_067_977_5],
        [3.0, 0.0, 5.0, 4.0, 2.828_427_124_746],
        [4.0, 5.0, 0.0, 3.0, 2.236_067_977_5],
        [5.0, 4.0, 3.0, 0.0, 2.828_427_124_746],
        [
            2.236_067_977_5,
            2.828_427_124_746,
            2.236_067_977_5,
            2.828_427_124_746,
            0.0,
        ],
    ]
}

fn expected_embedding() -> Array2<f64> {
    array![
        [2.0, -1.4],
        [2.0, 1.6],
        [-2.0, -1.4],
        [-2.0, 1.6],
        [1.443_289_932_013e-15, -0.4],
    ]
}

fn expected_eigenvalues() -> Array1<f64> {
    array![15.999_999_999_999_979, 9.199_999_999_999_976]
}

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
    assert_eq!(actual.dim(), expected.dim());
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            let a = actual[[i, j]];
            let e = expected[[i, j]];
            assert!(
                (a - e).abs() <= tol,
                "[{i},{j}] got {a}, expected {e}, diff {} > {tol}",
                (a - e).abs()
            );
        }
    }
}

#[test]
fn classical_mds_euclidean_matches_sklearn_source_oracle() {
    let fitted = ClassicalMDS::new(2).fit(&fixture_x(), &()).unwrap();

    assert_matrix_close(fitted.embedding(), &expected_embedding(), 1e-9);
    assert_matrix_close(fitted.dissimilarity_matrix(), &expected_distances(), 1e-10);

    assert_eq!(fitted.eigenvalues().len(), 2);
    for (actual, expected) in fitted
        .eigenvalues()
        .iter()
        .zip(expected_eigenvalues().iter())
    {
        assert!((actual - expected).abs() <= 1e-9);
    }
}

#[test]
fn classical_mds_precomputed_matches_euclidean_path() {
    let distances = expected_distances();
    let fitted = ClassicalMDS::new(2)
        .with_metric(Dissimilarity::Precomputed)
        .fit(&distances, &())
        .unwrap();

    assert_matrix_close(fitted.embedding(), &expected_embedding(), 1e-9);
    assert_matrix_close(fitted.dissimilarity_matrix(), &distances, 1e-12);
}

#[test]
fn classical_mds_fit_transform_returns_embedding() {
    let embedding = ClassicalMDS::new(2).fit_transform(&fixture_x()).unwrap();
    assert_matrix_close(&embedding, &expected_embedding(), 1e-9);
}

#[test]
fn classical_mds_validates_public_contract() {
    assert!(ClassicalMDS::new(0).fit(&fixture_x(), &()).is_err());

    let nonsquare = array![[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]];
    assert!(
        ClassicalMDS::new(2)
            .with_metric(Dissimilarity::Precomputed)
            .fit(&nonsquare, &())
            .is_err()
    );

    let nonsymmetric = array![[0.0, 1.0, 2.0], [9.0, 0.0, 3.0], [2.0, 3.0, 0.0]];
    assert!(
        ClassicalMDS::new(2)
            .with_metric(Dissimilarity::Precomputed)
            .fit(&nonsymmetric, &())
            .is_err()
    );
}

#[test]
fn classical_mds_components_clamp_to_sample_count() {
    let fitted = ClassicalMDS::new(10).fit(&fixture_x(), &()).unwrap();
    assert_eq!(fitted.embedding().dim(), (5, 5));
    assert_eq!(fitted.eigenvalues().len(), 5);
}
