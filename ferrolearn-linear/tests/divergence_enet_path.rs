//! Public helper parity for `ferrolearn_linear::enet_path` against the live
//! scikit-learn 1.5.2 `sklearn.linear_model.enet_path` oracle.
//!
//! `enet_path` solves the raw dense design path: no intercept is fitted and no
//! centering is performed. This mirrors sklearn's helper contract; callers that
//! need intercept behavior center data before calling the helper.

use ferrolearn_core::error::FerroError;
use ferrolearn_linear::{EnetPathOptions, enet_path};
use ndarray::{Array1, Array2, array};

fn fixture() -> (Array2<f64>, Array1<f64>) {
    (
        array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]],
        array![3.0, 2.5, 7.1, 6.0, 11.2],
    )
}

/// Live sklearn 1.5.2 oracle:
/// `enet_path(X, y, l1_ratio=0.5, alphas=[0.3, 0.1], max_iter=5000,
/// tol=1e-8, return_n_iter=True)` returns alphas `[0.3, 0.1]`, coef path
/// `[[0.71460827, 0.6592252], [1.29642622, 1.37033601]]`, dual gaps
/// `[8.81684610e-08, 8.81606436e-08]`, and n_iters `[161, 158]`.
#[test]
fn enet_path_explicit_alphas_matches_sklearn() {
    let (x, y) = fixture();
    let result = enet_path(
        &x,
        &y,
        EnetPathOptions::default()
            .with_l1_ratio(0.5)
            .with_alphas(vec![0.3, 0.1])
            .with_max_iter(5000)
            .with_tol(1e-8),
    )
    .unwrap();

    assert_eq!(result.alphas().as_slice().unwrap(), &[0.3, 0.1]);
    assert_eq!(result.coefficients().dim(), (2, 2));
    assert!((result.coefficients()[[0, 0]] - 0.71460827).abs() < 1e-6);
    assert!((result.coefficients()[[1, 0]] - 1.29642622).abs() < 1e-6);
    assert!((result.coefficients()[[0, 1]] - 0.6592252).abs() < 1e-6);
    assert!((result.coefficients()[[1, 1]] - 1.37033601).abs() < 1e-6);
    assert!((result.dual_gaps()[0] - 8.81684610e-08).abs() < 1e-7);
    assert!((result.dual_gaps()[1] - 8.81606436e-08).abs() < 1e-7);
    assert_eq!(result.n_iters(), &[161, 158]);
}

/// sklearn sorts an explicit alpha vector descending before solving the path.
#[test]
fn enet_path_sorts_explicit_alphas_like_sklearn() {
    let (x, y) = fixture();
    let result = enet_path(
        &x,
        &y,
        EnetPathOptions::default()
            .with_l1_ratio(0.5)
            .with_alphas(vec![0.1, 0.3])
            .with_max_iter(5000)
            .with_tol(1e-8),
    )
    .unwrap();

    assert_eq!(result.alphas().as_slice().unwrap(), &[0.3, 0.1]);
}

/// Live sklearn 1.5.2 oracle:
/// `enet_path(X, y, l1_ratio=0.5, n_alphas=4, eps=1e-2, max_iter=5000,
/// tol=1e-8, return_n_iter=True)` returns generated alphas
/// `[44.36, 9.55707228, 2.05900881, 0.4436]` and the coefficient path below.
#[test]
fn enet_path_generated_alpha_grid_matches_sklearn() {
    let (x, y) = fixture();
    let result = enet_path(
        &x,
        &y,
        EnetPathOptions::default()
            .with_l1_ratio(0.5)
            .with_n_alphas(4)
            .with_eps(1e-2)
            .with_max_iter(5000)
            .with_tol(1e-8),
    )
    .unwrap();

    let want_alphas = [44.36, 9.55707228, 2.05900881, 0.4436];
    for (got, want) in result.alphas().iter().zip(want_alphas) {
        assert!((*got - want).abs() < 1e-6, "alpha {got} vs {want}");
    }

    let want_coefs = [
        [0.0, 0.6227204, 0.81564517, 0.74160114],
        [0.0, 0.68451392, 1.03949892, 1.25623589],
    ];
    for (feature, want_row) in want_coefs.iter().enumerate() {
        for (alpha_idx, &want) in want_row.iter().enumerate() {
            let got = result.coefficients()[[feature, alpha_idx]];
            assert!(
                (got - want).abs() < 1e-6,
                "coef[{feature},{alpha_idx}] {got} vs {want}"
            );
        }
    }
    assert_eq!(result.n_iters(), &[1, 24, 65, 129]);
}

/// sklearn cannot generate an automatic alpha grid when `l1_ratio=0`; callers
/// must supply explicit alphas for the pure-L2 path.
#[test]
fn enet_path_l1_ratio_zero_generated_grid_errors_like_sklearn() {
    let (x, y) = fixture();
    let err = enet_path(
        &x,
        &y,
        EnetPathOptions::default()
            .with_l1_ratio(0.0)
            .with_n_alphas(4),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        FerroError::InvalidParameter {
            name,
            ..
        } if name == "l1_ratio"
    ));
}

#[test]
fn enet_path_negative_l1_ratio_errors() {
    let (x, y) = fixture();
    let err = enet_path(
        &x,
        &y,
        EnetPathOptions::default()
            .with_l1_ratio(-0.1)
            .with_alphas(vec![0.1]),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        FerroError::InvalidParameter {
            name,
            ..
        } if name == "l1_ratio"
    ));
}
