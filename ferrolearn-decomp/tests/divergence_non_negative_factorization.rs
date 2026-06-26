//! Divergence guard for `ferrolearn_decomp::non_negative_factorization` against
//! scikit-learn 1.5.2 `sklearn.decomposition.non_negative_factorization`.
//!
//! Scope: deterministic Frobenius-loss path with `solver="cd"`,
//! `init="nndsvd"`, no regularization, `update_H=true`, and `shuffle=false`.

use ferrolearn_decomp::{NMF, NMFInit, NMFSolver, non_negative_factorization};
use ndarray::{Array2, array};

fn fixture_12x6() -> Array2<f64> {
    array![
        [2.744, 3.576, 3.014, 2.724, 2.118, 3.229],
        [2.188, 4.459, 4.818, 1.917, 3.959, 2.644],
        [2.84, 4.628, 0.355, 0.436, 0.101, 4.163],
        [3.891, 4.35, 4.893, 3.996, 2.307, 3.903],
        [0.591, 3.2, 0.717, 4.723, 2.609, 2.073],
        [1.323, 3.871, 2.281, 2.842, 0.094, 3.088],
        [3.06, 3.085, 4.719, 3.409, 1.798, 2.185],
        [3.488, 0.301, 3.334, 3.353, 1.052, 0.645],
        [1.577, 1.819, 2.851, 2.193, 4.942, 0.51],
        [1.044, 0.807, 3.266, 1.266, 2.332, 1.222],
        [0.795, 0.552, 3.282, 0.691, 0.983, 1.844],
        [4.105, 0.486, 4.19, 0.48, 4.882, 2.343],
    ]
}

fn reconstruction_error(x: &Array2<f64>, w: &Array2<f64>, h: &Array2<f64>) -> f64 {
    let wh = w.dot(h);
    x.iter()
        .zip(wh.iter())
        .map(|(actual, reconstructed)| {
            let diff = actual - reconstructed;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle values are pinned at full f64 precision"
)]
#[test]
fn divergence_non_negative_factorization_cd_nndsvd() {
    let x = fixture_12x6();
    let (w, h, n_iter) = non_negative_factorization(
        &x,
        3,
        NMFInit::Nndsvd,
        NMFSolver::CoordinateDescent,
        200,
        1e-4,
        Some(0),
    )
    .expect("standalone NMF helper should fit the non-negative fixture");

    assert_eq!(w.dim(), (12, 3), "W shape");
    assert_eq!(h.dim(), (3, 6), "H shape");
    assert_eq!(n_iter, 151, "n_iter must match sklearn");

    // sklearn 1.5.2 oracle:
    // non_negative_factorization(X, n_components=3, init="nndsvd",
    // solver="cd", max_iter=200, tol=1e-4, random_state=0)
    let sk_w_row0 = [0.7208617466878319, 1.2958572601871108, 0.7663393327001597];
    let sk_h_row0 = [
        1.6531963561786642,
        0.03253488093304504,
        2.504223159640572,
        0.07478966190269389,
        2.2876705631651566,
        0.6092353143399691,
    ];
    for (k, &expected) in sk_w_row0.iter().enumerate() {
        let actual = w[[0, k]];
        assert!(
            (actual - expected).abs() <= 1e-6,
            "W[0][{k}]: sklearn {expected}, ferrolearn {actual}"
        );
    }
    for (j, &expected) in sk_h_row0.iter().enumerate() {
        let actual = h[[0, j]];
        assert!(
            (actual - expected).abs() <= 1e-6,
            "H[0][{j}]: sklearn {expected}, ferrolearn {actual}"
        );
    }

    let expected_reconstruction = 5.513563243249451_f64;
    let actual_reconstruction = reconstruction_error(&x, &w, &h);
    assert!(
        (actual_reconstruction - expected_reconstruction).abs() <= 1e-6,
        "reconstruction norm: sklearn {expected_reconstruction}, ferrolearn {actual_reconstruction}"
    );
}

#[test]
fn standalone_helper_matches_nmf_fit_metadata() {
    use ferrolearn_core::traits::Fit;

    let x = fixture_12x6();
    let fitted = NMF::<f64>::new(3)
        .with_init(NMFInit::Nndsvd)
        .with_solver(NMFSolver::CoordinateDescent)
        .with_max_iter(200)
        .with_tol(1e-4)
        .with_random_state(0)
        .fit(&x, &())
        .expect("NMF::fit should succeed on fixture");
    let (w, h, n_iter) = non_negative_factorization(
        &x,
        3,
        NMFInit::Nndsvd,
        NMFSolver::CoordinateDescent,
        200,
        1e-4,
        Some(0),
    )
    .expect("helper should succeed on fixture");

    assert_eq!(&h, fitted.components());
    assert_eq!(n_iter, fitted.n_iter());
    assert!((reconstruction_error(&x, &w, &h) - fitted.reconstruction_err()).abs() < 1e-12);
}

#[test]
fn standalone_helper_reuses_nmf_validation_contracts() {
    let x = fixture_12x6();

    assert!(
        non_negative_factorization(
            &x,
            0,
            NMFInit::Nndsvd,
            NMFSolver::CoordinateDescent,
            200,
            1e-4,
            Some(0),
        )
        .is_err()
    );
    assert!(
        non_negative_factorization(
            &array![[1.0, -1.0], [2.0, 3.0]],
            1,
            NMFInit::Nndsvd,
            NMFSolver::CoordinateDescent,
            200,
            1e-4,
            Some(0),
        )
        .is_err()
    );
}
