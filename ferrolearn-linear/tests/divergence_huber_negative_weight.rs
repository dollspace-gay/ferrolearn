//! Divergence pin: `HuberRegressor::fit_with_sample_weight` rejects negative
//! `sample_weight`, but scikit-learn 1.5.2 ACCEPTS them and returns a fitted
//! model.
//!
//! sklearn's `HuberRegressor.fit` (`sklearn/linear_model/_huber.py:306`) passes
//! `sample_weight` through `_check_sample_weight`
//! (`sklearn/utils/validation.py`), which validates length and dtype but
//! imposes NO non-negativity constraint — negative weights flow straight into
//! `_huber_loss_and_gradient` (`sklearn/linear_model/_huber.py:18`). The fit
//! converges and produces `coef_`/`intercept_`/`scale_` like any other run.
//!
//! ferrolearn (`huber_regressor.rs:682-687`) instead rejects ANY negative
//! weight with `FerroError::InvalidParameter`, so a call sklearn answers raises
//! an error. This is a public-contract divergence introduced by REQ-11.
//!
//! Expected values are from the live sklearn 1.5.2 oracle (R-CHAR-3), NOT copied
//! from ferrolearn. The companion in-tree test `sample_weight_negative_errors`
//! (`divergence_huber_fit.rs:382`) asserts the OPPOSITE (that the error is
//! correct) and so masks this divergence.

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_linear::HuberRegressor;
use ndarray::{Array1, Array2};

/// Divergence: ferrolearn's `HuberRegressor::fit_with_sample_weight` diverges
/// from `sklearn/linear_model/_huber.py:306` for a `sample_weight` containing a
/// negative entry.
///
/// Input: `X = RandomState(0).randn(12, 2)`,
/// `y = X @ [2, -1] + 1 + 0.3·randn`, `y[3] += 8`, `w = ones; w[0] = -1`.
///
/// sklearn returns `coef_ ≈ [1.94291, -1.07860]`, `intercept_ ≈ 0.99998`,
/// `scale_ ≈ 0.18822` (a normal, non-degenerate fit).
/// ferrolearn returns `Err(InvalidParameter { name: "sample_weight" })`.
///
/// Tracking: #2159
#[test]
fn divergence_huber_negative_sample_weight_accepted_by_sklearn() {
    // Live sklearn 1.5.2 oracle (generated via
    //   HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=200, tol=1e-5)
    //     .fit(X, y, sample_weight=w)).
    #[rustfmt::skip]
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            1.764052345967664, 0.4001572083672233,
            0.9787379841057392, 2.240893199201458,
            1.8675579901499675, -0.977277879876411,
            0.9500884175255894, -0.1513572082976979,
            -0.10321885179355784, 0.41059850193837233,
            0.144043571160878, 1.454273506962975,
            0.7610377251469934, 0.12167501649282841,
            0.44386323274542566, 0.33367432737426683,
            1.4940790731576061, -0.20515826376580087,
            0.31306770165090136, -0.8540957393017248,
            -2.5529898158340787, 0.6536185954403606,
            0.8644361988595057, -0.7421650204064419,
        ],
    )
    .unwrap();
    let y = Array1::from(vec![
        4.808873870764388,
        0.2802730666303912,
        5.726121415366779,
        10.995378888341126,
        0.8427975587820493,
        0.27462126632886635,
        2.446884661510233,
        1.6675008939972364,
        3.926980685791979,
        1.8859922021363493,
        -4.863971871906364,
        3.5179421088566474,
    ]);
    let mut w = Array1::from_elem(12, 1.0_f64);
    w[0] = -1.0;

    // sklearn 1.5.2 oracle (NOT copied from ferrolearn).
    let sk_coef = [1.9429070298186077, -1.0786019953027477];
    let sk_intercept = 0.9999795053711648;
    let sk_scale = 0.18821574392686072;

    let fitted = HuberRegressor::<f64>::new()
        .with_alpha(1e-4)
        .with_epsilon(1.35)
        .with_max_iter(200)
        .with_tol(1e-5)
        .fit_with_sample_weight(&x, &y, Some(&w))
        .expect(
            "sklearn 1.5.2 accepts negative sample_weight and returns a fitted \
             model; ferrolearn must too (it currently errors)",
        );

    let coef = HasCoefficients::coefficients(&fitted);
    let intercept = HasCoefficients::intercept(&fitted);
    for (i, &sk) in sk_coef.iter().enumerate() {
        assert!(
            (coef[i] - sk).abs() < 1e-3,
            "neg-weight coef[{i}]: sklearn={sk:.6}, ferrolearn={:.6}",
            coef[i]
        );
    }
    assert!(
        (intercept - sk_intercept).abs() < 1e-3,
        "neg-weight intercept: sklearn={sk_intercept:.6}, ferrolearn={intercept:.6}"
    );
    assert!(
        (fitted.scale() - sk_scale).abs() < 1e-3,
        "neg-weight scale_: sklearn={sk_scale:.6}, ferrolearn={:.6}",
        fitted.scale()
    );
}
