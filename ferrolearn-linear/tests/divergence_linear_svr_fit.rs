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
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::linear_svr::{DualMode, LinearSVR, LinearSVRLoss};
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

/// Divergence (#610, REQ-4): the `squared_epsilon_insensitive` (L2) loss must
/// reproduce the live `sklearn.svm.LinearSVR(loss='squared_epsilon_insensitive')`
/// `coef_`/`intercept_`. liblinear's `L2R_L2LOSS_SVR_DUAL` solver
/// (`sklearn/svm/src/liblinear/linear.cpp:1078-1081`, selected by
/// `_get_liblinear_solver_type`, `_base.py:1015-1016`) minimizes
/// `0.5·‖w‖² + C·Σ max(0,|r|−ε)²` with the dual CD using `lambda = 0.5/C`,
/// `upper_bound = +inf` — and crucially NO `1/n` averaging. A correct ferrolearn
/// `SquaredEpsilonInsensitive` branch (`linear_svr.rs fn fit`, `(half/c, inf)`)
/// converges to the same optimum.
///
/// Oracle (live sklearn 1.5.2, fit_intercept=True to match ferrolearn's API):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
///   X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([2.,4.,6.,8.,10.]); \
///   m=LinearSVR(loss='squared_epsilon_insensitive',epsilon=0.1,C=1.0, \
///     fit_intercept=True,max_iter=200000,tol=1e-10).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist(), m.predict([[1.5]]).tolist())"
/// # coef [1.8912820512820514] intercept [0.28205128205128216] predict(1.5) [3.1189743589743593]
/// ```
#[test]
fn linear_svr_squared_loss() {
    // Live sklearn 1.5.2: X=[[1]..[5]], y=2x, loss='squared_epsilon_insensitive',
    // epsilon=0.1, C=1.0, fit_intercept=True, max_iter=200000, tol=1e-10.
    const SK_COEF: f64 = 1.8912820512820514;
    const SK_INTERCEPT: f64 = 0.28205128205128216;
    const SK_PREDICT_1_5: f64 = 3.1189743589743593;

    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y: Array1<f64> = array![2.0, 4.0, 6.0, 8.0, 10.0];

    let fitted = LinearSVR::<f64>::new()
        .with_loss(LinearSVRLoss::SquaredEpsilonInsensitive)
        .with_epsilon(0.1)
        .with_c(1.0)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();

    let coef = fitted.coefficients()[0];
    let intercept = fitted.intercept();

    assert!(
        (coef - SK_COEF).abs() < 1e-3,
        "squared-loss coef parity: sklearn (liblinear L2R_L2LOSS_SVR_DUAL) \
         {SK_COEF}, ferrolearn {coef} (gap {:.4}).",
        (coef - SK_COEF).abs()
    );
    assert!(
        (intercept - SK_INTERCEPT).abs() < 1e-3,
        "squared-loss intercept parity: sklearn {SK_INTERCEPT}, \
         ferrolearn {intercept} (gap {:.4}).",
        (intercept - SK_INTERCEPT).abs()
    );

    // predict([[1.5]]) = 1.5*coef + intercept against the live oracle.
    let x_new = Array2::from_shape_vec((1, 1), vec![1.5]).unwrap();
    let pred = fitted.predict(&x_new).unwrap()[0];
    assert!(
        (pred - SK_PREDICT_1_5).abs() < 1e-3,
        "squared-loss predict(1.5) parity: sklearn {SK_PREDICT_1_5}, \
         ferrolearn {pred} (gap {:.4}).",
        (pred - SK_PREDICT_1_5).abs()
    );
}

/// Divergence (#608, REQ-2): `predict(X) = X @ coef_ + intercept_` must match the
/// live `sklearn.svm.LinearSVR.predict` on held-out rows for an oracle-matched
/// fit. With the default (epsilon-insensitive) loss, `epsilon=0.1`, `C=1.0`,
/// `fit_intercept=True`, the live oracle yields `coef [1.95]`, `intercept [0.15]`
/// (cf. `linear_svr_coef_parity`), so `predict([[2.5]]) = 2.5*1.95 + 0.15`.
/// `fn predict in linear_svr.rs` computes `x.dot(&coefficients) + intercept`,
/// mirroring `LinearModel.predict` (`X @ coef_.T + intercept_`).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
///   X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([2.,4.,6.,8.,10.]); \
///   m=LinearSVR(epsilon=0.1,C=1.0,fit_intercept=True,max_iter=200000,tol=1e-10) \
///     .fit(X,y); print(m.predict([[2.5]]).tolist())"
/// # [5.025000000054111]
/// ```
#[test]
fn linear_svr_predict() {
    // Live sklearn 1.5.2: same fit as linear_svr_coef_parity (eps=0.1, C=1.0).
    const SK_PREDICT_2_5: f64 = 5.025000000054111;

    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y: Array1<f64> = array![2.0, 4.0, 6.0, 8.0, 10.0];

    let fitted = LinearSVR::<f64>::new()
        .with_epsilon(0.1)
        .with_c(1.0)
        .with_loss(LinearSVRLoss::EpsilonInsensitive)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();

    let x_new = Array2::from_shape_vec((1, 1), vec![2.5]).unwrap();
    let pred = fitted.predict(&x_new).unwrap()[0];
    assert!(
        (pred - SK_PREDICT_2_5).abs() < 1e-3,
        "predict(2.5) parity: sklearn (liblinear) {SK_PREDICT_2_5}, \
         ferrolearn {pred} (gap {:.4}). predict = X@coef + intercept.",
        (pred - SK_PREDICT_2_5).abs()
    );
}

/// Structural (#613, REQ-8): `FittedLinearSVR::n_iter()` exposes the dual-CD
/// outer-iteration count, mirroring `sklearn.svm.LinearSVR.n_iter_`
/// (`sklearn/svm/_classes.py:467-468`, set to `n_iter_.max().item()` at
/// `_classes.py:603`; the liblinear iteration count from `_base.py:1215, :1247`).
/// ferrolearn's dual CD is a distinct implementation, so the exact count need
/// NOT match liblinear's bookkeeping — `n_iter_` is a STRUCTURAL attribute. The
/// invariant `1 <= n_iter <= max_iter` is what sklearn guarantees (the warning
/// at `_base.py:1234-1238` fires precisely when `n_iter >= max_iter`).
#[test]
fn linear_svr_n_iter() {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y: Array1<f64> = array![2.0, 4.0, 6.0, 8.0, 10.0];

    const MAX_ITER: usize = 200_000;
    let fitted = LinearSVR::<f64>::new()
        .with_epsilon(0.1)
        .with_c(1.0)
        .with_max_iter(MAX_ITER)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();

    let n_iter = fitted.n_iter();
    assert!(
        (1..=MAX_ITER).contains(&n_iter),
        "n_iter must satisfy 1 <= n_iter <= max_iter ({MAX_ITER}), got {n_iter} \
         (mirrors sklearn LinearSVR.n_iter_, _classes.py:603)"
    );
}

/// #612 (REQ-6): sklearn's `dual` defaults to `"auto"`, and for the default
/// `epsilon_insensitive` loss `_validate_dual_parameter` resolves to the dual
/// solver (`_classes.py:13-29`; the loss is dual-only, solver type 13,
/// `_base.py:1015`). ferrolearn models this as `DualMode` with default `Auto`;
/// `with_dual(Auto)` and `with_dual(True)` must produce the SAME `coef_` as the
/// default fit AND match the live oracle — `dual` is a solver-path choice on a
/// strongly-convex objective (R-DEV-7), so the minimizer is invariant.
///
/// Oracle (live sklearn 1.5.2, X=[[1]..[5]], y=2x, eps=0.1, C=1.0):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
///   X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([2.,4.,6.,8.,10.]); \
///   print([LinearSVR(epsilon=0.1,C=1.0,dual=d,fit_intercept=True, \
///          max_iter=200000,tol=1e-10).fit(X,y).coef_.tolist() \
///          for d in ('auto', True)])"
/// # dual='auto' -> coef [1.9499999999701896], dual=True -> coef [1.9499999999871787]
/// ```
#[test]
fn linear_svr_dual_auto_true() {
    // Live sklearn 1.5.2: dual in {'auto', True} both give coef ~1.95 (same as
    // the default fit pinned by linear_svr_coef_parity).
    const SK_COEF: f64 = 1.9499999999856916;

    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y: Array1<f64> = array![2.0, 4.0, 6.0, 8.0, 10.0];

    let build = |dual: DualMode| {
        LinearSVR::<f64>::new()
            .with_epsilon(0.1)
            .with_c(1.0)
            .with_dual(dual)
            .with_max_iter(200_000)
            .with_tol(1e-10)
            .fit(&x, &y)
            .unwrap()
    };

    // Default fit (DualMode::Auto is the constructor default).
    let default_fit = LinearSVR::<f64>::new()
        .with_epsilon(0.1)
        .with_c(1.0)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();
    let coef_default = default_fit.coefficients()[0];

    let coef_auto = build(DualMode::Auto).coefficients()[0];
    let coef_true = build(DualMode::True).coefficients()[0];

    // Structural: Auto and True equal the default fit byte-for-byte (same solver
    // path — Auto resolves to True, both are the existing dual CD).
    assert_eq!(
        coef_auto, coef_default,
        "dual=Auto must be byte-identical to the default fit (Auto->True->dual CD)"
    );
    assert_eq!(
        coef_true, coef_default,
        "dual=True must be byte-identical to the default fit (the existing dual CD)"
    );

    // Oracle: both match live sklearn's coef ~1.95.
    assert!(
        (coef_auto - SK_COEF).abs() < 1e-3,
        "dual=Auto coef parity: sklearn {SK_COEF}, ferrolearn {coef_auto}"
    );
    assert!(
        (coef_true - SK_COEF).abs() < 1e-3,
        "dual=True coef parity: sklearn {SK_COEF}, ferrolearn {coef_true}"
    );
}

/// #612 (REQ-6): `dual=False` with the default `epsilon_insensitive` loss is an
/// unsupported `(loss, dual)` combination — sklearn's
/// `_get_liblinear_solver_type` has no `{False: …}` entry for that loss
/// (`_base.py:1015`) and raises `ValueError("Unsupported set of arguments…")`
/// (`_base.py:1047`). ferrolearn must return `Err` (FerroError::InvalidParameter,
/// the R-DEV-2 analog of sklearn's ValueError).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
///   X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([2.,4.,6.,8.,10.]); \
///   LinearSVR(dual=False).fit(X,y)"
/// # ValueError: Unsupported set of arguments: The combination of penalty='l2'
/// #   and loss='epsilon_insensitive' ... is not supported
/// ```
#[test]
fn linear_svr_dual_false_eps_rejected() {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y: Array1<f64> = array![2.0, 4.0, 6.0, 8.0, 10.0];

    // Default loss is EpsilonInsensitive; dual=False is unsupported -> Err.
    let result = LinearSVR::<f64>::new()
        .with_dual(DualMode::False)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y);
    assert!(
        result.is_err(),
        "dual=False with epsilon_insensitive loss must return Err (sklearn \
         raises ValueError 'Unsupported set of arguments', _base.py:1047)"
    );
}

/// #612 (REQ-6): `dual=False` with `squared_epsilon_insensitive` is SUPPORTED —
/// sklearn selects primal solver type 11 (`_base.py:1016`), while `dual=True`
/// selects dual type 12. Both minimize the SAME strongly-convex objective
/// `0.5·‖w‖² + C·Σ max(0,|r|−ε)²` and reach the SAME unique optimum, so
/// ferrolearn's dual CD (R-DEV-7: implementation differs, observable coef
/// matches) reproduces sklearn's `dual=False` coef.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
///   X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([2.,4.,6.,8.,10.]); \
///   m=LinearSVR(dual=False,loss='squared_epsilon_insensitive',epsilon=0.1, \
///     C=1.0,fit_intercept=True,max_iter=200000,tol=1e-10).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// # coef [1.8912820512820514] intercept [0.28205128205128216]
/// ```
#[test]
fn linear_svr_dual_false_squared() {
    // Live sklearn 1.5.2: dual=False, loss='squared_epsilon_insensitive'.
    // Identical to the dual=True squared fit (same optimum).
    const SK_COEF: f64 = 1.8912820512820514;
    const SK_INTERCEPT: f64 = 0.28205128205128216;

    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y: Array1<f64> = array![2.0, 4.0, 6.0, 8.0, 10.0];

    let fitted = LinearSVR::<f64>::new()
        .with_dual(DualMode::False)
        .with_loss(LinearSVRLoss::SquaredEpsilonInsensitive)
        .with_epsilon(0.1)
        .with_c(1.0)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();

    let coef = fitted.coefficients()[0];
    let intercept = fitted.intercept();

    assert!(
        (coef - SK_COEF).abs() < 1e-3,
        "dual=False squared-loss coef parity: sklearn {SK_COEF}, \
         ferrolearn {coef} (gap {:.4}). R-DEV-7: dual CD reaches liblinear's \
         primal optimum.",
        (coef - SK_COEF).abs()
    );
    assert!(
        (intercept - SK_INTERCEPT).abs() < 1e-3,
        "dual=False squared-loss intercept parity: sklearn {SK_INTERCEPT}, \
         ferrolearn {intercept} (gap {:.4}).",
        (intercept - SK_INTERCEPT).abs()
    );
}

/// #614 (REQ-9): sklearn constrains `tol` strictly positive
/// (`"tol": [Interval(Real, 0.0, None, closed="neither")]`, `_classes.py:508`),
/// so `tol <= 0` raises a `ValueError` at validation. ferrolearn must return
/// `Err` (FerroError::InvalidParameter) for `tol = 0` (and negative).
///
/// Oracle (live sklearn 1.5.2 — constraint at `_classes.py:508`):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
///   X=np.array([[1.],[2.],[3.]]); y=np.array([1.,2.,3.]); \
///   LinearSVR(tol=0.0).fit(X,y)"
/// # InvalidParameterError: The 'tol' parameter ... must be a float in the
/// #   range (0.0, inf). Got 0.0 instead.
/// ```
#[test]
fn linear_svr_tol_validation() {
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let y: Array1<f64> = array![1.0, 2.0, 3.0];

    // tol = 0 is rejected (interval is open at 0).
    let zero = LinearSVR::<f64>::new().with_tol(0.0).fit(&x, &y);
    assert!(
        zero.is_err(),
        "tol=0.0 must return Err (sklearn Interval open at 0, _classes.py:508)"
    );

    // Negative tol is rejected too.
    let neg = LinearSVR::<f64>::new().with_tol(-1e-4).fit(&x, &y);
    assert!(
        neg.is_err(),
        "negative tol must return Err (sklearn Interval(Real,0,None,'neither'))"
    );
}

/// #614 (REQ-9): sklearn sets the standard `n_features_in_` fitted attribute
/// (number of columns of the `X` seen in `fit`, via `_validate_data`,
/// `_classes.py:569-576`). ferrolearn's `FittedLinearSVR::n_features_in()` must
/// return the number of features (`X.ncols()`).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
///   X=np.zeros((6,3)); X[:,0]=np.arange(6); y=np.arange(6.); \
///   print(LinearSVR().fit(X,y).n_features_in_)"
/// # 3
/// ```
#[test]
fn linear_svr_n_features_in() {
    // X has p=3 features; n_features_in_ must be 3 (sklearn _classes.py:569-576).
    const P: usize = 3;
    let mut x = Array2::<f64>::zeros((6, P));
    for i in 0..6 {
        x[[i, 0]] = i as f64;
    }
    let y: Array1<f64> = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    let fitted = LinearSVR::<f64>::new()
        .with_max_iter(5000)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        fitted.n_features_in(),
        P,
        "n_features_in must equal X.ncols() = {P} (sklearn n_features_in_, \
         _classes.py:569-576)"
    );
}
