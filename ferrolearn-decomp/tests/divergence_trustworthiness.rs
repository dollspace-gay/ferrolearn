//! Public helper parity for `ferrolearn_decomp::trustworthiness`.
//!
//! Oracle runtime: sklearn 1.5.2 in this workspace.
//!
//! Python oracle:
//! ```python
//! import numpy as np
//! from sklearn.manifold import trustworthiness
//! X = np.arange(5).reshape(-1, 1)
//! E = np.array([[0], [2], [4], [1], [3]])
//! print(trustworthiness(X, E, n_neighbors=1))  # 0.19999999999999996
//! X7 = np.array([[0.,0.],[1.1,0.2],[0.2,1.3],[2.2,0.1],
//!                [2.1,2.4],[3.4,3.1],[4.2,4.6]])
//! E7 = np.array([[0.,0.],[0.93,0.11],[0.12,0.83],[1.84,0.17],
//!                [1.69,1.94],[3.27,2.86],[3.91,4.27]])
//! for k in [1, 2, 3]:
//!     print(k, trustworthiness(X7, E7, n_neighbors=k))
//! # 1 0.8857142857142857
//! # 2 0.9591836734693877
//! # 3 0.9523809523809523
//! ```

use approx::assert_relative_eq;
use ferrolearn_decomp::trustworthiness;
use ndarray::array;

#[test]
fn trustworthiness_identity_and_affine_match_sklearn() {
    let x = array![
        [0.0, 0.0],
        [1.0, 0.3],
        [2.0, 1.1],
        [3.0, 0.7],
        [4.0, 2.0],
        [5.0, 2.4],
    ];
    let affine = x.mapv(|v| 5.0 + v / 10.0);

    assert_relative_eq!(trustworthiness(&x, &x, 2).unwrap(), 1.0, epsilon = 1e-15);
    assert_relative_eq!(
        trustworthiness(&x, &affine, 2).unwrap(),
        1.0,
        epsilon = 1e-15
    );
}

#[test]
fn trustworthiness_small_shuffle_matches_sklearn() {
    let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
    let embedded = array![[0.0], [2.0], [4.0], [1.0], [3.0]];

    assert_relative_eq!(
        trustworthiness(&x, &embedded, 1).unwrap(),
        0.19999999999999996,
        epsilon = 1e-15
    );
}

#[test]
fn trustworthiness_rank_penalties_match_sklearn() {
    let x = array![
        [0.0, 0.0],
        [1.1, 0.2],
        [0.2, 1.3],
        [2.2, 0.1],
        [2.1, 2.4],
        [3.4, 3.1],
        [4.2, 4.6],
    ];
    let embedded = array![
        [0.0, 0.0],
        [0.93, 0.11],
        [0.12, 0.83],
        [1.84, 0.17],
        [1.69, 1.94],
        [3.27, 2.86],
        [3.91, 4.27],
    ];

    assert_relative_eq!(
        trustworthiness(&x, &embedded, 1).unwrap(),
        0.8857142857142857,
        epsilon = 1e-15
    );
    assert_relative_eq!(
        trustworthiness(&x, &embedded, 2).unwrap(),
        0.9591836734693877,
        epsilon = 1e-15
    );
    assert_relative_eq!(
        trustworthiness(&x, &embedded, 3).unwrap(),
        0.9523809523809523,
        epsilon = 1e-15
    );
}

#[test]
fn trustworthiness_validation_errors() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
    let embedded = x.clone();

    let err = trustworthiness(&x, &embedded, 0).expect_err("k=0 must be rejected");
    assert!(format!("{err}").contains("n_neighbors"));

    let err = trustworthiness(&x, &embedded, 2).expect_err("k >= n/2 must be rejected");
    assert!(format!("{err}").contains("n_neighbors"));

    let short = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let err = trustworthiness(&x, &short, 1).expect_err("row mismatch must be rejected");
    assert!(format!("{err}").contains("Shape mismatch"));
}
