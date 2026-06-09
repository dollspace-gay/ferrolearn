//! Divergence pins for `ferrolearn-decomp` `PLSRegression` vs scikit-learn 1.5.2
//! (`sklearn/cross_decomposition/_pls.py`).
//!
//! Two distinct, audited divergences, each pinned by a FAILING `#[test]`:
//!
//! 1. `coefficients()` vs sklearn `coef_` — ORIENTATION + VALUE-SPACE
//!    (#2414). sklearn `coef_` is `(n_targets, n_features)` and is expressed in
//!    RAW (un-scaled) space (`_pls.py:399-400`,
//!    `coef_ = (x_rotations_ @ y_loadings_.T * y_std).T / x_std`), so
//!    `predict(X_raw) = X_raw @ coef_.T + intercept_` (`_pls.py:531`).
//!    ferrolearn's `coefficients_` is `(n_features_x, n_features_y)` (the
//!    TRANSPOSE shape) and lives in the CENTRED+SCALED space (the `x_std`/`y_std`
//!    factors are applied separately in `predict`,
//!    `cross_decomposition.rs:1448-1457`). The public `coefficients()` accessor
//!    therefore returns neither the orientation nor the values of sklearn's
//!    documented `coef_` attribute.
//!
//! 2. `fit` `n_components` upper bound (#2415). sklearn's regression-mode bound
//!    is `rank_upper_bound = p` (= `n_features_x`) ALONE (`_pls.py:294`);
//!    ferrolearn uses `min(n_features_x, n_features_y, n_samples)`
//!    (`cross_decomposition.rs:1360`), so it REJECTS `n_components` values that
//!    sklearn accepts whenever `n_features_y < n_features_x`.
//!
//! All expected values come from the live sklearn 1.5.2 oracle, run from `/tmp`
//! (R-CHAR-3); none are literal-copied from the ferrolearn side.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::cross_decomposition::PLSRegression;
use ndarray::{Array2, array};

/// 5x3 X, 5x2 Y deterministic fixture (shared with the existing cross-decomp
/// divergence suite).
fn fixture() -> (Array2<f64>, Array2<f64>) {
    let x = array![
        [1., 2., 3.],
        [4., 5., 7.],
        [7., 9., 8.],
        [10., 11., 14.],
        [13., 15., 16.]
    ];
    let y = array![[1., 0.5], [2., 1.2], [3., 1.4], [4., 2.1], [5., 2.6]];
    (x, y)
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
}

// ===========================================================================
// DIV #2414: coefficients() vs sklearn coef_ — orientation + value-space
// ===========================================================================

/// Divergence: ferrolearn `FittedPLSRegression::coefficients()`
/// (`cross_decomposition.rs:1306`, field `coefficients_` shape
/// `(n_features_x, n_features_y) = (3, 2)`) does NOT match the ORIENTATION of
/// sklearn's documented `coef_` attribute, which is
/// `(n_targets, n_features) = (2, 3)` (`sklearn/cross_decomposition/_pls.py:400`:
/// `self.coef_ = (self.coef_ * self._y_std).T / self._x_std`).
///
/// sklearn 1.5.2 `coef_` shape on this fixture is `(2, 3)`; ferrolearn's
/// `coefficients()` is `(3, 2)`. The public accessor advertises the wrong shape.
/// Tracking: #2414
#[test]
fn divergence_pls_coef_orientation_shape() {
    let (x, y) = fixture();
    let fitted = PLSRegression::<f64>::new(2)
        .fit(&x, &y)
        .expect("PLSRegression fit");
    let coef = fitted.coefficients();

    // Live sklearn 1.5.2 oracle (R-CHAR-3, run from /tmp):
    //   PLSRegression(n_components=2).fit(X, Y).coef_.shape == (2, 3)
    let sk_coef_shape = (2_usize, 3_usize);

    assert_eq!(
        coef.dim(),
        sk_coef_shape,
        "ferrolearn coefficients() shape {:?} != sklearn coef_ shape {:?} \
         (sklearn coef_ is (n_targets, n_features), _pls.py:400)",
        coef.dim(),
        sk_coef_shape
    );
}

/// Divergence: ferrolearn `coefficients()` does NOT match sklearn `coef_`
/// element-wise either. sklearn `coef_` is in RAW (un-scaled) space
/// (`_pls.py:399-400`: `coef_ = (x_rotations_ @ y_loadings_.T * y_std).T / x_std`),
/// while ferrolearn's `coefficients_` is in the centred+scaled space (the
/// `x_std`/`y_std` factors are applied separately in `predict`,
/// `cross_decomposition.rs:1448-1457`). So even transposing ferrolearn's matrix
/// does not yield sklearn's `coef_`.
///
/// Oracle (sklearn 1.5.2): `PLSRegression(n_components=2).fit(X, Y).coef_` =
/// `[[0.12481816, 0.13591311, 0.05773859], [0.04427986, 0.01921339, 0.0957523]]`.
/// We compare against the transpose of ferrolearn's `coefficients()` to give the
/// accessor every chance (orientation-only would still pass the value check) —
/// it FAILS on value because the scaling is not absorbed.
/// Tracking: #2414
#[test]
fn divergence_pls_coef_value_vs_sklearn_coef() {
    let (x, y) = fixture();
    let fitted = PLSRegression::<f64>::new(2)
        .fit(&x, &y)
        .expect("PLSRegression fit");
    let coef = fitted.coefficients();

    // Live sklearn 1.5.2 oracle (R-CHAR-3): PLSRegression(2).fit(X,Y).coef_,
    // shape (n_targets=2, n_features=3).
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_coef: Array2<f64> = array![
        [0.12481816, 0.13591311, 0.05773859],
        [0.04427986, 0.01921339, 0.0957523]
    ];

    // `coefficients()` now returns sklearn's documented `coef_` directly:
    // shape (n_targets, n_features) = (2, 3) in RAW (un-scaled) space
    // (`_pls.py:399-400`). Compare element-wise WITHOUT transposing — the fix
    // both re-orients AND absorbs the y_std/x_std scaling.
    assert_eq!(
        coef.dim(),
        sk_coef.dim(),
        "ferrolearn coefficients() shape {:?} != sklearn coef_ shape {:?}",
        coef.dim(),
        sk_coef.dim()
    );
    let ferro = coef.to_owned();
    let diff = max_abs_diff(&ferro, &sk_coef);
    assert!(
        diff < 1e-6,
        "ferrolearn coefficients() diverges from sklearn coef_ by {diff:e} \
         (sklearn coef_ is raw space (B * y_std).T / x_std, _pls.py:399-400); \
         ferro={ferro:?} sklearn={sk_coef:?}"
    );
}

// ===========================================================================
// DIV #2415: PLSRegression n_components upper bound (regression mode == p)
// ===========================================================================

/// Divergence: ferrolearn `PLSRegression::fit` (`cross_decomposition.rs:1360`)
/// rejects `n_components` above `min(n_features_x, n_features_y, n_samples)`,
/// but sklearn's REGRESSION-mode bound is `rank_upper_bound = p`
/// (= `n_features_x`) ALONE (`sklearn/cross_decomposition/_pls.py:294`:
/// `rank_upper_bound = p if self.deflation_mode == "regression" else ...`).
///
/// With `p=3`, `q=1` (single target), `n=5`: sklearn ACCEPTS `n_components=3`
/// (verified live: fits, `coef_.shape == (1, 3)`); ferrolearn REJECTS it because
/// its bound is `min(3, 1, 5) = 1`. ferrolearn returns `Err` where sklearn fits.
/// Tracking: #2415
#[test]
fn divergence_pls_n_components_regression_bound() {
    // p=3 features, q=1 target. sklearn regression rank_upper_bound = p = 3.
    let x = array![
        [1., 2., 3.],
        [4., 5., 7.],
        [7., 9., 8.],
        [10., 11., 14.],
        [13., 15., 16.]
    ];
    let y = array![[1.], [2.], [3.], [4.], [5.]];

    // Live sklearn 1.5.2 oracle (R-CHAR-3): PLSRegression(n_components=3).fit(X,Y)
    // SUCCEEDS (does not raise) for q=1, p=3 (rank_upper_bound = p = 3).
    let result = PLSRegression::<f64>::new(3).fit(&x, &y);

    assert!(
        result.is_ok(),
        "ferrolearn PLSRegression(n_components=3).fit rejected a fit that sklearn \
         accepts (sklearn regression rank_upper_bound = p = 3, _pls.py:294); \
         got Err: {:?}",
        result.err()
    );
}
