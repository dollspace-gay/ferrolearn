//! Divergence pins for `LinearSVC` against the live scikit-learn 1.5.2 oracle
//! (`sklearn/svm/_classes.py` class `LinearSVC` at `_classes.py:32`,
//! `sklearn/svm/_base.py` `_fit_liblinear` at `_base.py:1052`, commit 156ef14).
//!
//! `LinearSVC` is DETERMINISTIC for these tiny fixed inputs: liblinear's
//! coordinate descent on the strictly-convex L2-regularized squared-hinge
//! primal, with `random_state` only seeding a shuffle that is immaterial at a
//! tight `tol` / large `max_iter`, reaches the unique optimum. So exact
//! `coef_` / `intercept_` / `decision_function` parity is testable.
//!
//! THE CORE DIVERGENCE (#618). liblinear minimizes the L2-regularized
//! squared-hinge primal with the data term scaled by `C` (the SUMMED loss, no
//! `1/n` averaging):
//!
//! ```text
//!   min_w   0.5*||w||^2  +  C * sum_i  max(0, 1 - y_i*(w.x_i + b))^2
//! ```
//!
//! (the `C` argument passed verbatim to `liblinear.train_wrap`,
//! `sklearn/svm/_base.py:1215-1228`; objective documented in the `LinearSVC`
//! docstring, `_classes.py:32`). ferrolearn instead scales the data-term
//! gradient/Hessian by `C / n_samples`
//! (`ferrolearn-linear/src/linear_svc.rs` `fn solve_binary_primal`, the
//! `c / n_f * ...` terms with `n_f = n_samples`), i.e. it minimizes
//! `0.5*||w||^2 + (C/n) * sum_i loss`. The effective C is too small by a factor
//! of `n_samples`, so the optimum is shifted for every C and — critically — the
//! `coef_` DEPENDENCE ON C is flattened relative to liblinear.
//!
//! THE SHAPE DIVERGENCE (#619). For the binary case sklearn's
//! `LinearClassifierMixin.decision_function` collapses the single-column score
//! matrix to 1-D `(n_samples,)`
//! (`sklearn/linear_model/_base.py:365`:
//! `return xp.reshape(scores, (-1,)) if scores.shape[1] == 1 else scores`).
//! ferrolearn returns `Array2` shape `(n_samples, 1)`
//! (`ferrolearn-linear/src/linear_svc.rs` `fn decision_function`, the
//! `Array2::<F>::zeros((n_samples, 1))` binary branch). The PER-SAMPLE VALUES
//! also diverge because they are downstream of the wrong (`C/n`) optimum.
//!
//! ferrolearn's `LinearSVC` API ALWAYS fits an intercept (there is no
//! `fit_intercept` parameter — `fn solve_binary_primal` unconditionally updates
//! a scalar `b`), so every oracle below uses sklearn `fit_intercept=True` to
//! match ferrolearn's actual behavior. The exact python invocation that
//! produced each constant is recorded in a comment (goal.md R-CHAR-3; values are
//! NEVER copied from the ferrolearn side).
//!
//! Tracking: #618 (coef_ C-dependence + basic coef_/intercept_ parity — the
//! C-vs-C/n solver crux), #619 (binary decision_function shape `(n,)` + value).

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::linear_svc::{LinearSVC, LinearSVCLoss};
use ndarray::{Array1, Array2, array};

/// The fixed binary 8x2 well-separated set used across the pins (AC-1).
fn binary_set() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
    (x, y)
}

/// Divergence (#618, REQ-1/REQ-10 — the crux): liblinear's `coef_` tracks `C`
/// because the objective scales the SUMMED loss by `C`
/// (`sklearn/svm/_base.py:1215-1228`; `0.5*||w||^2 + C*sum loss`, the `C`
/// passed straight to `liblinear.train_wrap`). ferrolearn scales by
/// `C / n_samples` (`linear_svc.rs` `fn solve_binary_primal`, the `c / n_f`
/// gradient/Hessian terms), which FLATTENS the C-dependence: with
/// `n_samples = 8`, the effective regularization at `C=0.1` vs `C=1.0` is
/// pulled together while liblinear's `coef_` moves from `0.07847` to `0.12835`
/// (and the intercept from `-0.48286` to `-1.19438`).
///
/// ferrolearn always fits an intercept, so the oracle uses `fit_intercept=True`
/// to match. We pin BOTH absolute `coef_[0]` values; at least one fails because
/// ferrolearn's `C/n` cannot reproduce the C-scaling.
///
/// Oracle (live sklearn 1.5.2, fit_intercept=True to match ferrolearn's API):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   print([(LinearSVC(C=c,loss='squared_hinge',fit_intercept=True,max_iter=200000,tol=1e-10) \
///          .fit(X,y).coef_[0,0]) for c in (0.1,1.0)])"
/// # [0.0784651864625997, 0.12835213611984458]
/// # intercepts: C=0.1 -> -0.4828626859236904, C=1.0 -> -1.1943776585907158
/// ```
#[test]
fn linear_svc_coef_c_dependence() {
    // Live sklearn 1.5.2, binary 8x2 set, squared_hinge, fit_intercept=True
    // (matches ferrolearn's always-intercept API).
    const SK_COEF_C_0_1: f64 = 0.0784651864625997; // C=0.1
    const SK_COEF_C_1_0: f64 = 0.12835213611984458; // C=1.0

    let (x, y) = binary_set();

    let fit_c01 = LinearSVC::<f64>::new()
        .with_c(0.1)
        .with_loss(LinearSVCLoss::SquaredHinge)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();
    let coef_c01 = fit_c01.coefficients()[0];

    let fit_c10 = LinearSVC::<f64>::new()
        .with_c(1.0)
        .with_loss(LinearSVCLoss::SquaredHinge)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();
    let coef_c10 = fit_c10.coefficients()[0];

    let sk_spread = (SK_COEF_C_1_0 - SK_COEF_C_0_1).abs();
    let ferro_spread = (coef_c10 - coef_c01).abs();

    // Pin BOTH absolute coef values against the liblinear oracle. At least one
    // FAILS under the C/n scaling.
    assert!(
        (coef_c01 - SK_COEF_C_0_1).abs() < 1e-2,
        "C=0.1 coef: sklearn (liblinear) {SK_COEF_C_0_1}, ferrolearn {coef_c01} \
         (gap {:.5}). ferrolearn's C/n scaling shifts the optimum. \
         C-spread sklearn={sk_spread:.5} vs ferrolearn={ferro_spread:.5}.",
        (coef_c01 - SK_COEF_C_0_1).abs()
    );
    assert!(
        (coef_c10 - SK_COEF_C_1_0).abs() < 1e-2,
        "C=1.0 coef: sklearn (liblinear) {SK_COEF_C_1_0}, ferrolearn {coef_c10} \
         (gap {:.5}). ferrolearn's C/n scaling shifts the optimum. \
         C-spread sklearn={sk_spread:.5} vs ferrolearn={ferro_spread:.5}.",
        (coef_c10 - SK_COEF_C_1_0).abs()
    );
}

/// Divergence (#618, REQ-1): a fixed-config `LinearSVC` fit (default
/// `squared_hinge`, `C=1.0`) must reproduce the live `sklearn.svm.LinearSVC`
/// `coef_` / `intercept_` to a meaningful tolerance (1e-2 — liblinear's CD and
/// a CORRECT primal/dual CD converge to the same optimum; the C/n gap is large,
/// well above solver noise). This FAILS because of the `C/n` mis-scaling
/// (`linear_svc.rs` `fn solve_binary_primal`).
///
/// Both `coef_` columns are equal by the symmetry of the 8x2 set; we pin
/// `coef_[0]`, `coef_[1]`, and `intercept_`.
///
/// Oracle (live sklearn 1.5.2, fit_intercept=True to match ferrolearn's API):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   m=LinearSVC(C=1.0,loss='squared_hinge',fit_intercept=True,max_iter=200000,tol=1e-10) \
///     .fit(X,y); print(m.coef_.tolist(), m.intercept_.tolist())"
/// # coef [[0.12835213611984458, 0.12835213611984475]] intercept [-1.1943776585907158]
/// ```
#[test]
fn linear_svc_coef_parity() {
    // Live sklearn 1.5.2: binary 8x2 set, squared_hinge, C=1.0,
    // fit_intercept=True, max_iter=200000, tol=1e-10. Stable optimum.
    const SK_COEF_0: f64 = 0.12835213611984458;
    const SK_COEF_1: f64 = 0.12835213611984475;
    const SK_INTERCEPT: f64 = -1.1943776585907158;

    let (x, y) = binary_set();

    let fitted = LinearSVC::<f64>::new()
        .with_c(1.0)
        .with_loss(LinearSVCLoss::SquaredHinge)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();

    let coef = fitted.coefficients();
    let coef0 = coef[0];
    let coef1 = coef[1];
    let intercept = fitted.intercept();

    assert!(
        (coef0 - SK_COEF_0).abs() < 1e-2,
        "coef_[0] parity: sklearn (liblinear) {SK_COEF_0}, ferrolearn {coef0} \
         (gap {:.5}). ferrolearn minimizes 0.5||w||^2 + (C/n)*sum loss; \
         liblinear minimizes 0.5||w||^2 + C*sum loss (_base.py:1215-1228).",
        (coef0 - SK_COEF_0).abs()
    );
    assert!(
        (coef1 - SK_COEF_1).abs() < 1e-2,
        "coef_[1] parity: sklearn (liblinear) {SK_COEF_1}, ferrolearn {coef1} \
         (gap {:.5}).",
        (coef1 - SK_COEF_1).abs()
    );
    assert!(
        (intercept - SK_INTERCEPT).abs() < 1e-2,
        "intercept parity: sklearn (liblinear) {SK_INTERCEPT}, \
         ferrolearn {intercept} (gap {:.5}).",
        (intercept - SK_INTERCEPT).abs()
    );
}

/// Divergence (#619, REQ-2): binary `decision_function` shape + values.
///
/// SHAPE: sklearn's `LinearClassifierMixin.decision_function` collapses the
/// single-column binary score matrix to 1-D `(n_samples,)`
/// (`sklearn/linear_model/_base.py:365`:
/// `return xp.reshape(scores, (-1,)) if scores.shape[1] == 1 else scores`;
/// live: `m.decision_function(X).shape == (8,)`). ferrolearn returns `Array2`
/// shape `(8, 1)` (`linear_svc.rs` `fn decision_function`, the
/// `Array2::<F>::zeros((n_samples, 1))` binary branch).
///
/// VALUES: even reading ferrolearn's single column, the per-sample scores
/// diverge because they are `X @ w + b` with the WRONG (`C/n`) optimum (#618).
/// We assert the shape-stable per-sample decision (ferrolearn column 0) matches
/// sklearn's 1-D `(8,)` values element-by-element. The shape contract `(8, 1)`
/// vs `(8,)` is best resolved at the Rust API (`fn decision_function` returning
/// a 1-D `Array1` for the binary case, mirroring sklearn's `.ravel()`); the
/// PyO3 binding-ABI fix (cf. #600/#581) only matters once the Rust side is 1-D.
///
/// Oracle (live sklearn 1.5.2, fit_intercept=True to match ferrolearn's API):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   m=LinearSVC(C=1.0,loss='squared_hinge',fit_intercept=True,max_iter=200000,tol=1e-10) \
///     .fit(X,y); print(m.decision_function(X).shape, m.decision_function(X).tolist())"
/// # (8,) [-0.9376733863510265, -0.8093212502311817, -0.8093212502311818,
/// #       -0.6809691141113371, 0.8592565193267989, 0.9876086554466437,
/// #       0.9876086554466437, 1.1159607915664884]
/// ```
#[test]
fn linear_svc_decision_function() {
    // Live sklearn 1.5.2: binary 8x2 set, squared_hinge, C=1.0,
    // fit_intercept=True. decision_function(X) is 1-D shape (8,).
    const SK_DF: [f64; 8] = [
        -0.9376733863510265,
        -0.8093212502311817,
        -0.8093212502311818,
        -0.6809691141113371,
        0.8592565193267989,
        0.9876086554466437,
        0.9876086554466437,
        1.1159607915664884,
    ];

    let (x, y) = binary_set();

    let fitted = LinearSVC::<f64>::new()
        .with_c(1.0)
        .with_loss(LinearSVCLoss::SquaredHinge)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();

    let df = fitted.decision_function(&x).unwrap();

    // Per-sample value parity (ferrolearn's binary column 0 vs sklearn's 1-D
    // (8,) decision values). FAILS because the values come from the wrong
    // (C/n) optimum. The (8,1) vs (8,) shape divergence is documented above and
    // tracked separately under #619.
    for (i, &sk) in SK_DF.iter().enumerate() {
        let fl = df[[i, 0]];
        assert!(
            (fl - sk).abs() < 1e-2,
            "decision_function[{i}]: sklearn (liblinear, 1-D (8,)) {sk}, \
             ferrolearn (col 0 of (8,1)) {fl} (gap {:.5}). Values diverge \
             from the C/n optimum (#618); the (8,1) vs (8,) shape diverges \
             from sklearn's .ravel() (_base.py:365, #619).",
            (fl - sk).abs()
        );
    }
}
