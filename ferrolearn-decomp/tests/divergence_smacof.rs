//! Public helper parity for `ferrolearn_decomp::smacof` against
//! `sklearn.manifold.smacof`.
//!
//! The fixed-init path is deterministic and matches sklearn element-wise. The
//! default random-init path remains a documented RNG carve-out because
//! ferrolearn does not use numpy `RandomState`.

use ferrolearn_decomp::smacof;
use ndarray::{Array2, array};

fn fixed_init() -> Array2<f64> {
    array![[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.05]]
}

fn assert_emb_eq(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
    assert_eq!(actual.dim(), expected.dim(), "embedding shape mismatch");
    let (rows, cols) = expected.dim();
    for i in 0..rows {
        for j in 0..cols {
            let a = actual[[i, j]];
            let e = expected[[i, j]];
            assert!(
                (a - e).abs() <= tol,
                "embedding [{i},{j}] = {a}, expected {e} (|diff| = {}, tol {tol})",
                (a - e).abs()
            );
        }
    }
}

/// Green guard: public `smacof(D, init=X0, n_init=1)` matches sklearn's helper
/// return tuple on the metric, precomputed-dissimilarity path.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// from sklearn.manifold import smacof
/// smacof(D, metric=True, init=X0, n_init=1, normalized_stress=False,
///        max_iter=300, eps=1e-3, return_n_iter=True)
///   -> stress = 3.148219331054871, n_iter = 13,
///      embedding = [[-3.333717200034, -1.658330631573],
///                   [-0.431085112947, -0.700165295708],
///                   [-0.786750476780,  2.465105803376],
///                   [ 4.551552789761, -0.106609876095]]
/// ```
#[test]
fn public_smacof_fixed_init_precomputed_matches_sklearn() {
    let d: Array2<f64> = array![
        [0.0, 2.0, 5.0, 9.0],
        [2.0, 0.0, 3.0, 4.0],
        [5.0, 3.0, 0.0, 6.0],
        [9.0, 4.0, 6.0, 0.0],
    ];
    let sk_emb: Array2<f64> = array![
        [-3.333_717_200_034, -1.658_330_631_573],
        [-0.431_085_112_947, -0.700_165_295_708],
        [-0.786_750_476_780, 2.465_105_803_376],
        [4.551_552_789_761, -0.106_609_876_095],
    ];

    let (embedding, stress, n_iter) =
        smacof(&d, 2, Some(&fixed_init()), 1, 300, 1e-3, false, None).unwrap();

    assert_emb_eq(&embedding, &sk_emb, 1e-6);
    assert!((stress - 3.148_219_331_054_871).abs() <= 1e-6);
    assert_eq!(n_iter, 13);
}

#[test]
fn public_smacof_validates_shape_and_components() {
    let non_square = array![[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]];
    assert!(smacof(&non_square, 2, None, 1, 300, 1e-3, false, None).is_err());

    let d = array![[0.0, 1.0], [1.0, 0.0]];
    assert!(smacof(&d, 0, None, 1, 300, 1e-3, false, None).is_err());

    let init = array![[0.1], [0.2]];
    assert!(smacof(&d, 0, Some(&init), 1, 300, 1e-3, false, None).is_err());
}
