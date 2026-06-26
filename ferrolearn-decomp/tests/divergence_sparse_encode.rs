//! Divergence guard for `ferrolearn_decomp::sparse_encode` against
//! scikit-learn 1.5.2 `sklearn.decomposition.sparse_encode`.
//!
//! Scope: OMP, threshold, and lasso-cd sparse coding on dense f64 arrays.
//! sklearn's lars/lasso_lars, positive-code constraints, precomputed Gram/cov,
//! init, and n_jobs paths remain tracked gaps.

use ferrolearn_decomp::{DictTransformAlgorithm, sparse_encode};
use ndarray::{Array2, array};

fn fixture() -> (Array2<f64>, Array2<f64>) {
    let x = array![[1.0, 0.5, -1.0], [0.0, 1.0, 2.0], [2.0, -1.0, 0.5],];
    let dictionary = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.6, 0.8, 0.0],
    ];
    (x, dictionary)
}

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
    assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            let diff = (actual[[i, j]] - expected[[i, j]]).abs();
            assert!(
                diff <= tol,
                "entry ({i},{j}): expected {}, got {}, diff {diff} > {tol}",
                expected[[i, j]],
                actual[[i, j]]
            );
        }
    }
}

#[test]
fn sparse_encode_omp_matches_sklearn() {
    let (x, dictionary) = fixture();
    let actual = sparse_encode(
        &x,
        &dictionary,
        DictTransformAlgorithm::Omp,
        Some(2),
        None,
        1000,
    )
    .expect("OMP sparse_encode should succeed");

    // sklearn 1.5.2 oracle:
    // sparse_encode(X, dictionary, algorithm="omp", n_nonzero_coefs=2)
    let expected = array![
        [1.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
        [2.0, -1.0, 0.0, 0.0],
    ];
    assert_matrix_close(&actual, &expected, 1e-12);
}

#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle threshold output is pinned at f64 precision"
)]
#[test]
fn sparse_encode_threshold_matches_sklearn() {
    let (x, dictionary) = fixture();
    let actual = sparse_encode(
        &x,
        &dictionary,
        DictTransformAlgorithm::Threshold,
        None,
        Some(0.4),
        1000,
    )
    .expect("threshold sparse_encode should succeed");

    // sklearn 1.5.2 oracle:
    // sparse_encode(X, dictionary, algorithm="threshold", alpha=0.4)
    let expected = array![
        [0.6, 0.09999999999999998, -0.6, 0.6],
        [0.0, 0.6, 1.6, 0.4],
        [1.6, -0.6, 0.09999999999999998, 0.0],
    ];
    assert_matrix_close(&actual, &expected, 1e-12);
}

#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle lasso-cd output is pinned at f64 precision"
)]
#[test]
fn sparse_encode_lasso_cd_matches_sklearn() {
    let (x, dictionary) = fixture();
    let actual = sparse_encode(
        &x,
        &dictionary,
        DictTransformAlgorithm::LassoCd,
        None,
        Some(0.2),
        1000,
    )
    .expect("lasso-cd sparse_encode should succeed");

    // sklearn 1.5.2 oracle:
    // sparse_encode(X, dictionary, algorithm="lasso_cd", alpha=0.2, max_iter=1000)
    let expected = array![
        [0.500016926659445, 0.0, -0.8, 0.49998984400433283],
        [0.0, 0.8, 1.8, 0.0],
        [1.8, -0.8, 0.3, 0.0],
    ];
    assert_matrix_close(&actual, &expected, 1e-4);
}

#[test]
fn sparse_encode_validation_contracts() {
    let (x, dictionary) = fixture();

    assert!(
        sparse_encode(
            &x,
            &dictionary.slice(ndarray::s![.., ..2]).to_owned(),
            DictTransformAlgorithm::Omp,
            Some(2),
            None,
            1000,
        )
        .is_err()
    );
    assert!(
        sparse_encode(
            &x,
            &dictionary,
            DictTransformAlgorithm::Omp,
            Some(0),
            None,
            1000,
        )
        .is_err()
    );
    assert!(
        sparse_encode(
            &x,
            &dictionary,
            DictTransformAlgorithm::Threshold,
            None,
            Some(-1.0),
            1000,
        )
        .is_err()
    );
}
