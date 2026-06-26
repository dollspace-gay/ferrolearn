//! Public helper parity for `ferrolearn_linear::lasso_path` against the live
//! scikit-learn 1.5.2 `sklearn.linear_model.lasso_path` oracle.
//!
//! `lasso_path` solves the raw dense design path: no intercept is fitted and no
//! centering is performed. This mirrors sklearn's helper contract; callers that
//! need intercept behavior center data before calling the helper.

use ferrolearn_core::error::FerroError;
use ferrolearn_linear::{LassoPathOptions, lasso_path};
use ndarray::{Array1, Array2, array};

fn fixture() -> (Array2<f64>, Array1<f64>) {
    (
        array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]],
        array![3.0, 2.5, 7.1, 6.0, 11.2],
    )
}

/// Live sklearn 1.5.2 oracle:
/// `lasso_path(X, y, alphas=[0.3, 0.1], max_iter=5000, tol=1e-8,
/// return_n_iter=True)` returns alphas `[0.3, 0.1]`, coef path
/// `[[0.60555573, 0.61481499], [1.40555539, 1.41481464]]`, dual gaps
/// `[8.39223219e-08, 8.64990852e-08]`, and n_iters `[215, 147]`.
#[test]
fn lasso_path_explicit_alphas_matches_sklearn() {
    let (x, y) = fixture();
    let result = lasso_path(
        &x,
        &y,
        LassoPathOptions::default()
            .with_alphas(vec![0.3, 0.1])
            .with_max_iter(5000)
            .with_tol(1e-8),
    )
    .unwrap();

    assert_eq!(result.alphas().as_slice().unwrap(), &[0.3, 0.1]);
    assert_eq!(result.coefficients().dim(), (2, 2));
    assert!((result.coefficients()[[0, 0]] - 0.60555573).abs() < 1e-6);
    assert!((result.coefficients()[[1, 0]] - 1.40555539).abs() < 1e-6);
    assert!((result.coefficients()[[0, 1]] - 0.61481499).abs() < 1e-6);
    assert!((result.coefficients()[[1, 1]] - 1.41481464).abs() < 1e-6);
    assert!((result.dual_gaps()[0] - 8.39223219e-08).abs() < 1e-7);
    assert!((result.dual_gaps()[1] - 8.64990852e-08).abs() < 1e-7);
    assert_eq!(result.n_iters(), &[215, 147]);
}

/// sklearn sorts an explicit alpha vector descending before solving the path.
#[test]
fn lasso_path_sorts_explicit_alphas_like_sklearn() {
    let (x, y) = fixture();
    let result = lasso_path(
        &x,
        &y,
        LassoPathOptions::default()
            .with_alphas(vec![0.1, 0.3])
            .with_max_iter(5000)
            .with_tol(1e-8),
    )
    .unwrap();

    assert_eq!(result.alphas().as_slice().unwrap(), &[0.3, 0.1]);
}

/// Live sklearn 1.5.2 oracle:
/// `lasso_path(X, y, n_alphas=4, eps=1e-2, max_iter=5000, tol=1e-8,
/// return_n_iter=True)` returns generated alphas
/// `[22.18, 4.77853614, 1.0295044, 0.2218]` and the coefficient path below.
#[test]
fn lasso_path_generated_alpha_grid_matches_sklearn() {
    let (x, y) = fixture();
    let result = lasso_path(
        &x,
        &y,
        LassoPathOptions::default()
            .with_n_alphas(4)
            .with_eps(1e-2)
            .with_max_iter(5000)
            .with_tol(1e-8),
    )
    .unwrap();

    let want_alphas = [22.18, 4.77853614, 1.0295044, 0.2218];
    for (got, want) in result.alphas().iter().zip(want_alphas) {
        assert!((*got - want).abs() < 1e-6, "alpha {got} vs {want}");
    }

    let want_coefs = [
        [0.0, 0.39821607, 0.57178238, 0.6091761],
        [0.0, 1.19821577, 1.37178204, 1.40917576],
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
    assert_eq!(result.n_iters(), &[1, 215, 187, 166]);
}

#[test]
fn lasso_path_negative_alpha_errors() {
    let (x, y) = fixture();
    let err = lasso_path(
        &x,
        &y,
        LassoPathOptions::default().with_alphas(vec![0.1, -0.1]),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        FerroError::InvalidParameter {
            name,
            ..
        } if name == "alphas"
    ));
}
