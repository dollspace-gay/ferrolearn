//! Public helper parity for `ferrolearn_linear::lars_path` against the live
//! scikit-learn 1.5.2 `sklearn.linear_model.lars_path` oracle.
//!
//! `lars_path` solves the raw dense design path: no intercept is fitted and no
//! centering is performed. This mirrors sklearn's helper contract; callers that
//! need intercept behavior center data before calling the helper.

use ferrolearn_core::error::FerroError;
use ferrolearn_linear::{LarsPathMethod, LarsPathOptions, lars_path};
use ndarray::{Array1, Array2, array};

fn fixture() -> (Array2<f64>, Array1<f64>) {
    (
        array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]],
        array![3.0, 2.5, 7.1, 6.0, 11.2],
    )
}

/// Live sklearn 1.5.2 oracle:
/// `lars_path(X, y, max_iter=2, method="lar", return_n_iter=True)` returns
/// alphas `[22.18, 13.38, 0.0]`, active `[1, 0]`, coefficient path
/// `[[0.0, 0.0, 0.61944444], [0.0, 0.8, 1.41944444]]`, and `n_iter=2`.
#[test]
fn lars_path_lar_matches_sklearn() {
    let (x, y) = fixture();
    let result = lars_path(&x, &y, LarsPathOptions::default().with_max_iter(2)).unwrap();

    let want_alphas = [22.18, 13.38, 0.0];
    for (got, want) in result.alphas().iter().zip(want_alphas) {
        assert!((*got - want).abs() < 1e-8, "alpha {got} vs {want}");
    }
    assert_eq!(result.active(), &[1, 0]);
    assert_eq!(result.coefficients().dim(), (2, 3));

    let want_coefs = [[0.0, 0.0, 0.61944444], [0.0, 0.8, 1.41944444]];
    for (feature, want_row) in want_coefs.iter().enumerate() {
        for (alpha_idx, &want) in want_row.iter().enumerate() {
            let got = result.coefficients()[[feature, alpha_idx]];
            assert!(
                (got - want).abs() < 1e-8,
                "coef[{feature},{alpha_idx}] {got} vs {want}"
            );
        }
    }
    assert_eq!(result.n_iter(), 2);
}

/// On this fixture sklearn's LARS and LARS-Lasso paths match exactly.
#[test]
fn lars_path_lasso_matches_sklearn() {
    let (x, y) = fixture();
    let result = lars_path(
        &x,
        &y,
        LarsPathOptions::default()
            .with_max_iter(2)
            .with_method(LarsPathMethod::Lasso),
    )
    .unwrap();

    let want_alphas = [22.18, 13.38, 0.0];
    for (got, want) in result.alphas().iter().zip(want_alphas) {
        assert!((*got - want).abs() < 1e-8, "alpha {got} vs {want}");
    }
    assert_eq!(result.active(), &[1, 0]);
    assert!((result.coefficients()[[0, 2]] - 0.61944444).abs() < 1e-8);
    assert!((result.coefficients()[[1, 2]] - 1.41944444).abs() < 1e-8);
    assert_eq!(result.n_iter(), 2);
}

/// Live sklearn 1.5.2 oracle:
/// `lars_path(X, y, alpha_min=0.5, method="lasso", return_n_iter=True)`
/// interpolates the final knot to alpha `0.5` with final coefficients
/// `[0.5962963, 1.3962963]`.
#[test]
fn lars_path_alpha_min_interpolates_like_sklearn() {
    let (x, y) = fixture();
    let result = lars_path(
        &x,
        &y,
        LarsPathOptions::default()
            .with_alpha_min(0.5)
            .with_method(LarsPathMethod::Lasso),
    )
    .unwrap();

    let want_alphas = [22.18, 13.38, 0.5];
    for (got, want) in result.alphas().iter().zip(want_alphas) {
        assert!((*got - want).abs() < 1e-8, "alpha {got} vs {want}");
    }
    assert!((result.coefficients()[[0, 2]] - 0.5962963).abs() < 1e-7);
    assert!((result.coefficients()[[1, 2]] - 1.3962963).abs() < 1e-7);
    assert_eq!(result.n_iter(), 2);
}

#[test]
fn lars_path_negative_alpha_min_errors() {
    let (x, y) = fixture();
    let err = lars_path(&x, &y, LarsPathOptions::default().with_alpha_min(-0.1)).unwrap_err();

    assert!(matches!(
        err,
        FerroError::InvalidParameter {
            name,
            ..
        } if name == "alpha_min"
    ));
}
