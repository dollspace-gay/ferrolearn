//! Divergence pin for `ferrolearn_linear::ard::ARDRegression` in the
//! `n_samples < n_features` regime.
//!
//! scikit-learn's `ARDRegression.fit` selects the posterior-covariance update
//! method based on the sample/feature ratio
//! (`sklearn/linear_model/_bayes.py:670-674`):
//!
//! ```text
//! update_sigma = (
//!     self._update_sigma
//!     if n_samples >= n_features
//!     else self._update_sigma_woodbury
//! )
//! ```
//!
//! When `n_samples < n_features` sklearn uses `_update_sigma_woodbury`
//! (`_bayes.py:732-748`), which inverts the well-conditioned
//! `(n_samples, n_samples)` matrix `eye/alpha_ + (Xk/lambda) @ Xk.T` via the
//! Woodbury identity. ferrolearn's `fit` ALWAYS calls `update_sigma`
//! (`ferrolearn-linear/src/ard.rs:502`, `:609`), which is the
//! `_update_sigma` DIRECT branch (`_bayes.py:750-759`): it inverts the
//! `(n_features, n_features)` matrix `diag(lambda) + alpha * Xk^T Xk`. In the
//! `n_samples < n_features` regime that Gram block is rank-deficient
//! (`rank(Xk^T Xk) <= n_samples < n_features`), so the inverted matrix is
//! ill-conditioned; although the two formulas are equal in EXACT arithmetic,
//! the direct branch accumulates floating-point error that diverges the EM
//! trajectory — sklearn uses the Woodbury branch precisely to avoid this.
//!
//! Result: ferrolearn's `coef_` and `n_iter_` diverge from sklearn in the
//! `n_samples < n_features` regime.
//!
//! Expected values are from the LIVE sklearn 1.5.2 oracle (NOT copied from
//! ferrolearn — R-CHAR-3):
//!
//! ```text
//! python3 -c "import numpy as np; from sklearn.linear_model import ARDRegression; \
//!   X=np.array([[1.,-2.,0.5,3.,-1.,2.,0.,1.5],[2.,1.,-1.,0.,2.,-1.,1.,-0.5], \
//!     [-1.,3.,2.,1.,0.,0.5,-2.,2.],[0.5,-1.,1.,-2.,3.,1.,0.,-1.], \
//!     [3.,0.,-0.5,1.,-2.,0.,2.,0.5]]); \
//!   y=4.0*X[:,0]-3.0*X[:,3]; m=ARDRegression().fit(X,y); \
//!   print(m.n_iter_, m.coef_.tolist())"
//! # -> 300 [3.208360333621716, 0.0, 0.0, -1.3299476770370697, 0.0, 0.0, 0.0, 0.0]
//! ```
//!
//! Tracking: #2164 (n_samples < n_features Woodbury branch).

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::ard::ARDRegression;
use ndarray::{Array2, array};

/// `n_samples < n_features` (5 samples, 8 features): ferrolearn's `coef_` and
/// `n_iter_` must match the live sklearn 1.5.2 oracle. sklearn uses
/// `_update_sigma_woodbury` here; ferrolearn always uses the direct
/// `_update_sigma` branch, so the EM trajectory (and therefore `coef_` and
/// `n_iter_`) diverges.
#[test]
fn divergence_ard_woodbury_n_lt_p() {
    // sklearn 1.5.2 live oracle (see module doc): n_samples=5 < n_features=8.
    const SK_N_ITER: usize = 300;
    const SK_COEF: [f64; 8] = [
        3.208_360_333_621_716,
        0.0,
        0.0,
        -1.329_947_677_037_069_7,
        0.0,
        0.0,
        0.0,
        0.0,
    ];

    let x = Array2::from_shape_vec(
        (5, 8),
        vec![
            1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.0, 1.5, //
            2.0, 1.0, -1.0, 0.0, 2.0, -1.0, 1.0, -0.5, //
            -1.0, 3.0, 2.0, 1.0, 0.0, 0.5, -2.0, 2.0, //
            0.5, -1.0, 1.0, -2.0, 3.0, 1.0, 0.0, -1.0, //
            3.0, 0.0, -0.5, 1.0, -2.0, 0.0, 2.0, 0.5,
        ],
    )
    .unwrap();
    // y depends only on features 0 and 3.
    let y = array![
        4.0 * 1.0 - 3.0 * 3.0,
        4.0 * 2.0 - 3.0 * 0.0,
        4.0 * -1.0 - 3.0 * 1.0,
        4.0 * 0.5 - 3.0 * -2.0,
        4.0 * 3.0 - 3.0 * 1.0,
    ];

    let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();

    // n_iter_ must match sklearn's Woodbury trajectory.
    assert_eq!(
        fitted.n_iter(),
        SK_N_ITER,
        "n_iter_ diverges in n<p regime: ferrolearn={}, sklearn={SK_N_ITER} \
         (missing Woodbury branch, #2164)",
        fitted.n_iter(),
    );

    // coef_ must match sklearn's Woodbury trajectory element-wise.
    let coef = fitted.coefficients();
    for i in 0..8 {
        assert!(
            (coef[i] - SK_COEF[i]).abs() <= 1e-6 * SK_COEF[i].abs().max(1.0),
            "coef_[{i}] diverges in n<p regime: ferrolearn={}, sklearn={} \
             (missing Woodbury branch, #2164)",
            coef[i],
            SK_COEF[i],
        );
    }
}
