//! Divergence pins for `LinearSVR` against the live scikit-learn 1.5.2 oracle
//! (`sklearn/svm/_classes.py` class `LinearSVR`, `sklearn/svm/_base.py`
//! `_fit_liblinear`, commit 156ef14).
//!
//! `LinearSVR` is DETERMINISTIC for these inputs (liblinear's coordinate
//! descent on a strictly convex objective with `random_state` unused for the
//! shuffle on tiny fixed data, and a tight `tol`/large `max_iter` so the
//! optimum is reached), so exact `coef_`/`intercept_` parity is testable.
//!
//! THE CORE DIVERGENCE. sklearn's liblinear minimizes the L2-regularized
//! epsilon-insensitive primal with the data term scaled by `C` (the SUMMED
//! loss, no `1/n` averaging):
//!
//! ```text
//!   min_w  0.5*||w||^2  +  C * sum_i  max(0, |y_i - w.x_i| - epsilon)
//! ```
//!
//! (`sklearn/svm/_base.py` `_fit_liblinear`; objective documented in the
//! `LinearSVR` docstring `_classes.py:386-394` "The strength of the
//! regularization is inversely proportional to C"). ferrolearn instead scales
//! the data-term gradient by `C / n_samples` (`linear_svr.rs` `fn fit`, the
//! `self.c / n_f * ...` terms), i.e. it minimizes
//! `0.5*||w||^2 + (C/n) * sum_i loss`. The effective C is too small by a factor
//! of `n_samples`, so the optimum is shifted for every C and — critically — the
//! `coef_` DEPENDENCE ON C is flattened relative to liblinear.
//!
//! ferrolearn's API always fits an intercept (there is no `fit_intercept`
//! parameter — `fn fit` unconditionally updates a scalar `b`), so every oracle
//! below is taken with sklearn `fit_intercept=True` to match ferrolearn's
//! actual behavior. The exact python invocation that produced each constant is
//! recorded in a comment (goal.md R-CHAR-3; values are NEVER copied from the
//! ferrolearn side).
//!
//! Tracking: #607 (coef_ C-dependence / C-vs-C/n solver crux + basic parity),
//! #609 (epsilon default 0.0 vs 0.1).

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::linear_svr::{LinearSVR, LinearSVRLoss};
use ndarray::{Array1, Array2, array};

/// Divergence (#609, REQ-3): sklearn's `LinearSVR` constructor defaults
/// `epsilon=0.0` (`sklearn/svm/_classes.py:378` "epsilon : float, default=0.0"
/// and `:522` `epsilon=0.0`). ferrolearn's `LinearSVR::new` sets
/// `epsilon = 0.1` (`linear_svr.rs` `fn LinearSVR::new`,
/// `epsilon: F::from(0.1).unwrap()`). A default-parameter ABI divergence
/// (R-DEV-2): `LinearSVR::new()` fits with a wider insensitivity tube than
/// `sklearn.svm.LinearSVR()`.
///
/// The expected value is the sklearn `file:line` symbolic constant (the
/// constructor default at `_classes.py:522`), NOT a value copied from
/// ferrolearn.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "from sklearn.svm import LinearSVR; print(LinearSVR().epsilon)"
/// # 0.0
/// ```
#[test]
fn linear_svr_default_epsilon() {
    // sklearn constructor default `epsilon=0.0` (_classes.py:378, :522).
    const SK_EPSILON_DEFAULT: f64 = 0.0;

    let m = LinearSVR::<f64>::new();
    assert!(
        (m.epsilon - SK_EPSILON_DEFAULT).abs() < 1e-18,
        "default epsilon must be sklearn's 0.0 (_classes.py:522), got {} \
         (ferrolearn's LinearSVR::new sets 0.1)",
        m.epsilon
    );
}

/// Divergence (#607, REQ-1/REQ-7 — the crux): liblinear's `coef_` tracks `C`
/// because the objective scales the SUMMED loss by `C`
/// (`sklearn/svm/_base.py` `_fit_liblinear`; `0.5*||w||^2 + C*sum loss`).
/// ferrolearn scales by `C / n_samples` (`linear_svr.rs` `fn fit`, the
/// `self.c / n_f` data-term gradient), which FLATTENS the C-dependence: with
/// `n_samples = 4`, the effective regularization at `C=0.1` and `C=1.0` is
/// `C/4` apart, so ferrolearn's two fits land close together while liblinear's
/// move from `coef [0.58]` to `coef [1.2941]`.
///
/// ferrolearn always fits an intercept, so the oracle uses
/// `fit_intercept=True` to match. We pin BOTH absolute `coef_` values; at least
/// one fails because ferrolearn's `C/n` cannot reproduce the C-scaling.
///
/// Oracle (live sklearn 1.5.2, fit_intercept=True to match ferrolearn's API):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
///   X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([1.,3.,2.,5.]); \
///   print([LinearSVR(epsilon=0.,C=c,fit_intercept=True,max_iter=200000,tol=1e-10) \
///          .fit(X,y).coef_.tolist() for c in (0.1,1.0)])"
/// # [[0.5800000000000001], [1.2941176470588234]]
/// # intercepts: C=0.1 -> [0.26], C=1.0 -> [-0.17647058823529413]
/// ```
#[test]
fn linear_svr_coef_c_dependence() {
    // Live sklearn 1.5.2, X=[[1],[2],[3],[4]], y=[1,3,2,5], epsilon=0,
    // fit_intercept=True (matches ferrolearn's always-intercept API).
    const SK_COEF_C_0_1: f64 = 0.5800000000000001; // C=0.1
    const SK_COEF_C_1_0: f64 = 1.2941176470588234; // C=1.0

    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y: Array1<f64> = array![1.0, 3.0, 2.0, 5.0];

    let fit_c01 = LinearSVR::<f64>::new()
        .with_epsilon(0.0)
        .with_c(0.1)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();
    let coef_c01 = fit_c01.coefficients()[0];

    let fit_c10 = LinearSVR::<f64>::new()
        .with_epsilon(0.0)
        .with_c(1.0)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();
    let coef_c10 = fit_c10.coefficients()[0];

    // liblinear's C-spread: coef moves from 0.58 to 1.2941, a gap of ~0.714.
    // ferrolearn's C/n scaling collapses this gap (the effective C is /4).
    let sk_spread = (SK_COEF_C_1_0 - SK_COEF_C_0_1).abs();
    let ferro_spread = (coef_c10 - coef_c01).abs();

    // Pin BOTH absolute coef values against the liblinear oracle. At least one
    // FAILS under the C/n scaling.
    assert!(
        (coef_c01 - SK_COEF_C_0_1).abs() < 1e-3,
        "C=0.1 coef: sklearn (liblinear) {SK_COEF_C_0_1}, ferrolearn {coef_c01} \
         (gap {:.4}). ferrolearn's C/n scaling shifts the optimum. \
         C-spread sklearn={sk_spread:.4} vs ferrolearn={ferro_spread:.4}.",
        (coef_c01 - SK_COEF_C_0_1).abs()
    );
    assert!(
        (coef_c10 - SK_COEF_C_1_0).abs() < 1e-3,
        "C=1.0 coef: sklearn (liblinear) {SK_COEF_C_1_0}, ferrolearn {coef_c10} \
         (gap {:.4}). ferrolearn's C/n scaling shifts the optimum. \
         C-spread sklearn={sk_spread:.4} vs ferrolearn={ferro_spread:.4}.",
        (coef_c10 - SK_COEF_C_1_0).abs()
    );
}

/// Divergence (#607, REQ-1): a fixed-config `LinearSVR` fit must reproduce the
/// live `sklearn.svm.LinearSVR` `coef_`/`intercept_` to a meaningful tolerance
/// (1e-3 — liblinear's dual CD and a CORRECT primal CD converge to the same
/// optimum). This FAILS because of the `C/n` mis-scaling (`linear_svr.rs`
/// `fn fit`) and the fixed-step (`step = 0.01`) sub-gradient descent that does
/// not converge to liblinear's optimum.
///
/// To keep this pin ISOLATED from the epsilon-default divergence (#609), the
/// sklearn oracle uses `epsilon=0.1` — matching ferrolearn's current default —
/// so ONLY the solver/scaling divergence is exercised here. C=1.0 and the
/// always-on intercept also match ferrolearn's defaults.
///
/// The 1e-3 tolerance cleanly separates the C/n divergence (a LARGE gap, the
/// effective C is off by a factor of `n_samples`) from any sub-ULP
/// solver-trajectory noise — the epsilon-insensitive loss is non-smooth, but
/// the strictly convex `0.5*||w||^2` regularizer makes the minimizer unique,
/// and liblinear at `tol=1e-10`/`max_iter=200000` is stable to ~1e-8.
///
/// Oracle (live sklearn 1.5.2, epsilon=0.1 to match ferrolearn's default):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
///   X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([2.,4.,6.,8.,10.]); \
///   m=LinearSVR(epsilon=0.1,C=1.0,fit_intercept=True,max_iter=200000,tol=1e-10) \
///     .fit(X,y); print(m.coef_.tolist(), m.intercept_.tolist())"
/// # coef [1.9499999999856916] intercept [0.15000000007154177]
/// ```
#[test]
fn linear_svr_coef_parity() {
    // Live sklearn 1.5.2: X=[[1]..[5]], y=[2,4,6,8,10], epsilon=0.1 (==
    // ferrolearn default), C=1.0, fit_intercept=True. Stable optimum.
    const SK_COEF: f64 = 1.9499999999856916;
    const SK_INTERCEPT: f64 = 0.15000000007154177;

    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y: Array1<f64> = array![2.0, 4.0, 6.0, 8.0, 10.0];

    // ferrolearn defaults: C=1.0, epsilon=0.1, loss=EpsilonInsensitive. Give it
    // a generous iteration budget + tight tol so it has every chance to reach
    // its own optimum — the divergence is the OBJECTIVE (C/n), not the budget.
    let fitted = LinearSVR::<f64>::new()
        .with_epsilon(0.1)
        .with_c(1.0)
        .with_loss(LinearSVRLoss::EpsilonInsensitive)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();

    let coef = fitted.coefficients()[0];
    let intercept = fitted.intercept();

    assert!(
        (coef - SK_COEF).abs() < 1e-3,
        "coef parity: sklearn (liblinear) {SK_COEF}, ferrolearn {coef} \
         (gap {:.4}). ferrolearn minimizes 0.5||w||^2 + (C/n)*sum loss; \
         liblinear minimizes 0.5||w||^2 + C*sum loss.",
        (coef - SK_COEF).abs()
    );
    assert!(
        (intercept - SK_INTERCEPT).abs() < 1e-3,
        "intercept parity: sklearn (liblinear) {SK_INTERCEPT}, \
         ferrolearn {intercept} (gap {:.4}).",
        (intercept - SK_INTERCEPT).abs()
    );
}
