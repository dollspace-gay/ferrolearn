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

use ferrolearn_core::introspection::{HasClasses, HasCoefficients};
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::linear_svc::{
    ClassWeight, DualMode, LinearSVC, LinearSVCLoss, LinearSVCPenalty,
};
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

    // Binary decision_function is now 1-D `(8,)` (sklearn ravels the single
    // column, `linear_model/_base.py:365`). `as_binary()` returns the raveled
    // `Array1`; a multiclass fit would return `None` here.
    let binary = df
        .as_binary()
        .expect("binary decision_function is 1-D (n,)");
    assert_eq!(binary.len(), 8, "binary decision_function shape is (8,)");

    // Per-sample value parity (ferrolearn's 1-D (8,) decision vs sklearn's 1-D
    // (8,) decision values).
    for (i, &sk) in SK_DF.iter().enumerate() {
        let fl = binary[i];
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

/// Pin (#620, REQ-3 — predict + classes_): certifies that `predict` and
/// `classes_` reproduce the live `sklearn.svm.LinearSVC` oracle on the binary
/// 8x2 set. The predicted labels are downstream of the liblinear-parity fit
/// (REQ-1), and `classes_ = np.unique(y)` (`sklearn/svm/_classes.py:311`).
/// This pin should PASS — a green pin certifies REQ-3 SHIPPED.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   m=LinearSVC(C=1.0,loss='squared_hinge',fit_intercept=True,max_iter=200000,tol=1e-10).fit(X,y); \
///   print(m.predict(X).tolist(), m.classes_.tolist())"
/// # [0, 0, 0, 0, 1, 1, 1, 1] [0, 1]
/// ```
#[test]
fn linear_svc_predict_parity() {
    // Live sklearn 1.5.2: binary 8x2 set, squared_hinge, C=1.0,
    // fit_intercept=True, max_iter=200000, tol=1e-10.
    const SK_PREDICT: [usize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    const SK_CLASSES: [usize; 2] = [0, 1];

    let (x, y) = binary_set();

    let fitted = LinearSVC::<f64>::new()
        .with_c(1.0)
        .with_loss(LinearSVCLoss::SquaredHinge)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();
    assert_eq!(
        preds.len(),
        SK_PREDICT.len(),
        "predict length: sklearn {}, ferrolearn {}",
        SK_PREDICT.len(),
        preds.len()
    );
    for (i, &sk) in SK_PREDICT.iter().enumerate() {
        assert_eq!(
            preds[i], sk,
            "predict[{i}]: sklearn (liblinear) {sk}, ferrolearn {}",
            preds[i]
        );
    }

    assert_eq!(
        fitted.classes(),
        &SK_CLASSES,
        "classes_ mismatch: sklearn np.unique(y) {SK_CLASSES:?}, ferrolearn {:?}",
        fitted.classes()
    );
}

/// Pin (#621, REQ-4 — hinge-loss optimum): certifies that a `loss='hinge'` fit
/// reproduces the live `sklearn.svm.LinearSVC(loss='hinge')` `coef_` /
/// `intercept_` on the binary 8x2 set. The dual CD solves the true hinge
/// optimum (`U = C`, `diag = 0`, solver type 3, `linear.cpp:849-858`), so this
/// pin should PASS — a green pin certifies REQ-4 SHIPPED.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   m=LinearSVC(loss='hinge',C=1.0,fit_intercept=True,max_iter=200000,tol=1e-10).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// # coef [[0.15384615383852776, 0.15384615383915584]] intercept [-1.4615384615168394]
/// ```
#[test]
fn linear_svc_hinge_coef_parity() {
    // Live sklearn 1.5.2: binary 8x2 set, loss='hinge', C=1.0,
    // fit_intercept=True, max_iter=200000, tol=1e-10.
    const SK_COEF_0: f64 = 0.15384615383852776;
    const SK_COEF_1: f64 = 0.15384615383915584;
    const SK_INTERCEPT: f64 = -1.4615384615168394;

    let (x, y) = binary_set();

    let fitted = LinearSVC::<f64>::new()
        .with_c(1.0)
        .with_loss(LinearSVCLoss::Hinge)
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
        "hinge coef_[0]: sklearn (liblinear, U=C diag=0) {SK_COEF_0}, \
         ferrolearn {coef0} (gap {:.5}).",
        (coef0 - SK_COEF_0).abs()
    );
    assert!(
        (coef1 - SK_COEF_1).abs() < 1e-2,
        "hinge coef_[1]: sklearn (liblinear) {SK_COEF_1}, ferrolearn {coef1} \
         (gap {:.5}).",
        (coef1 - SK_COEF_1).abs()
    );
    assert!(
        (intercept - SK_INTERCEPT).abs() < 1e-2,
        "hinge intercept: sklearn (liblinear) {SK_INTERCEPT}, \
         ferrolearn {intercept} (gap {:.5}).",
        (intercept - SK_INTERCEPT).abs()
    );
}

/// Pin (#627, REQ-11 — n_iter_ / n_features_in_ / tol validation). These
/// accessors DO NOT EXIST on `FittedLinearSVC` yet, so this pin FAILS (compile
/// error on `n_features_in()` / `n_iter()`) until the fixer adds them — that
/// compile failure IS the real REQ-11 gap.
///
/// - `n_features_in_` is DETERMINISTIC: oracle `n_features_in_ == 2` for the
///   8x2 set (`sklearn/svm/_classes.py:302` via `_validate_data`).
/// - `n_iter_` is shuffle-path dependent: sklearn's liblinear shuffles `index`
///   each sweep (live: squared_hinge n_iter_=6, hinge n_iter_=560), while
///   ferrolearn sweeps natural order, so the COUNT legitimately differs (the
///   documented RNG-path boundary, cf. SGD). sklearn exposes
///   `n_iter_ = n_iter_.max().item()` (`_classes.py:338`). We DO NOT pin an
///   exact value — only that the accessor exists and is bounded.
/// - tol validation: `_parameter_constraints` is
///   `tol: Interval(Real, 0.0, None, closed="neither")` (`_classes.py:237`), so
///   `tol <= 0` raises ValueError. ferrolearn must error on `tol <= 0`.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   m=LinearSVC(C=1.0,loss='squared_hinge',fit_intercept=True,max_iter=200000,tol=1e-10).fit(X,y); \
///   print(m.n_features_in_, m.n_iter_)"
/// # 2 6
/// ```
#[test]
fn linear_svc_attrs_and_tol_validation() {
    let (x, y) = binary_set();

    let fitted = LinearSVC::<f64>::new()
        .with_c(1.0)
        .with_loss(LinearSVCLoss::SquaredHinge)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();

    // n_features_in_ is deterministic (oracle == 2 for the 8x2 set).
    assert_eq!(
        fitted.n_features_in(),
        2,
        "n_features_in_: sklearn (_validate_data, _classes.py:302) 2, \
         ferrolearn {}",
        fitted.n_features_in()
    );

    // n_iter_ is shuffle-path dependent (sklearn liblinear shuffles `index`
    // each sweep; ferrolearn sweeps natural order). Pin only existence +
    // boundedness, not an exact value (the documented RNG-path boundary).
    let k = fitted.n_iter();
    assert!(
        (1..=200_000).contains(&k),
        "n_iter_ must be bounded in [1, max_iter]; got {k}"
    );

    // tol <= 0 must raise (Interval(Real, 0.0, None, closed='neither'),
    // _classes.py:237).
    assert!(
        LinearSVC::<f64>::new().with_tol(0.0).fit(&x, &y).is_err(),
        "tol=0.0 must be rejected (closed='neither', _classes.py:237)"
    );
    assert!(
        LinearSVC::<f64>::new().with_tol(-1.0).fit(&x, &y).is_err(),
        "tol=-1.0 must be rejected (closed='neither', _classes.py:237)"
    );
}

/// PIN 1 (#622, REQ-5 — l1-penalty coef_/intercept_ parity). Certifies the NEW
/// l1 solver (`fn solve_binary_l1r_l2`, liblinear `solve_l1r_l2_svc`,
/// `linear.cpp:1467`, solver type 5 via `_get_liblinear_solver_type`,
/// `sklearn/svm/_base.py:1014`) reaches liblinear's l1 optimum on the binary
/// 8x2 set. This is a DISTINCT optimum from the l2 fit: the l2 intercept is
/// -1.19438, the l1 intercept is -1.20796 — same `coef_` to 1e-2 here but a
/// different objective (`‖w‖₁ + C·Σ max(0,1−yf)²`).
///
/// NOTE: `LinearSVC(penalty='l1')` defaults `random_state=None` and liblinear
/// shuffles the coordinate order, so the converged `coef_` jitters at ~1e-7..1e-9
/// run-to-run. We pin with tolerance 1e-2 — well above the jitter, far below the
/// l1-vs-wrong-objective gap.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   m=LinearSVC(penalty='l1',loss='squared_hinge',dual=False,C=1.0,\
///     fit_intercept=True,max_iter=200000,tol=1e-10).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// # coef [[0.12831858, 0.12831858]] intercept [-1.20796460]  (jitters ~1e-7)
/// ```
#[test]
fn linear_svc_l1_penalty_coef_parity() {
    // Live sklearn 1.5.2: l1, squared_hinge, dual=False, C=1.0,
    // fit_intercept=True. Run-to-run-stable to ~1e-7 (random shuffle).
    const SK_COEF: f64 = 0.12831858;
    const SK_INTERCEPT: f64 = -1.20796460;

    let (x, y) = binary_set();

    let fitted = LinearSVC::<f64>::new()
        .with_penalty(LinearSVCPenalty::L1)
        .with_loss(LinearSVCLoss::SquaredHinge)
        .with_dual(DualMode::False)
        .with_c(1.0)
        .with_max_iter(200_000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .unwrap();

    let coef = fitted.coefficients();
    let coef0 = coef[0];
    let coef1 = coef[1];
    let intercept = fitted.intercept();

    assert!(
        (coef0 - SK_COEF).abs() < 1e-2,
        "l1 coef_[0]: sklearn (liblinear solve_l1r_l2_svc, type 5) {SK_COEF}, \
         ferrolearn {coef0} (gap {:.6}).",
        (coef0 - SK_COEF).abs()
    );
    assert!(
        (coef1 - SK_COEF).abs() < 1e-2,
        "l1 coef_[1]: sklearn (liblinear) {SK_COEF}, ferrolearn {coef1} \
         (gap {:.6}).",
        (coef1 - SK_COEF).abs()
    );
    assert!(
        (intercept - SK_INTERCEPT).abs() < 1e-2,
        "l1 intercept_: sklearn (liblinear) {SK_INTERCEPT}, ferrolearn \
         {intercept} (gap {:.6}). (Distinct from the l2 optimum -1.19438.)",
        (intercept - SK_INTERCEPT).abs()
    );
}

/// PIN 2 (#625, REQ-8 — unsupported penalty×loss×dual combinations reject).
/// `fn liblinear_solver_type` mirrors `_get_liblinear_solver_type`
/// (`sklearn/svm/_base.py:995-1049`). The `multi_class='ovr'` slice of
/// `_solver_type_dict` (`_base.py:1013-1014`) is:
///
/// ```text
///   hinge:         { l2: { True: 3 } }
///   squared_hinge: { l1: { False: 5 }, l2: { False: 2, True: 1 } }
/// ```
///
/// Combos with no entry raise `ValueError` in sklearn → ferrolearn `Err`.
///
/// Oracle matrix (live sklearn 1.5.2 `.fit`, binary 8x2 set; ERR = ValueError,
/// OK = fits):
/// ```text
///   l1 + hinge          (any dual)  -> ERR  "penalty='l1' and loss='hinge' is not supported"
///   l1 + squared_hinge  dual=True   -> ERR  "... are not supported when dual=True"
///   l2 + hinge          dual=False  -> ERR  "... are not supported when dual=False"
///   l2 + squared_hinge  dual=True   -> OK   (solver 1)
///   l2 + squared_hinge  dual=False  -> OK   (solver 2)
///   l2 + hinge          dual=True   -> OK   (solver 3)
///   l1 + squared_hinge  dual=False  -> OK   (solver 5)
/// python3 -c "import numpy as np,warnings; from sklearn.svm import LinearSVC; \
///   X=...; y=...; \
///   [print(p,l,d, ...try LinearSVC(penalty=p,loss=l,dual=d).fit(X,y)...) ...]"
/// ```
#[test]
fn linear_svc_dual_penalty_rejects() {
    let (x, y) = binary_set();

    let base = || {
        LinearSVC::<f64>::new()
            .with_c(1.0)
            .with_max_iter(200_000)
            .with_tol(1e-10)
    };

    // --- Unsupported combos: sklearn raises ValueError; ferrolearn must Err. ---

    // l1 + hinge (any dual): no `hinge` entry has an `l1` penalty (_base.py:1013).
    assert!(
        base()
            .with_penalty(LinearSVCPenalty::L1)
            .with_loss(LinearSVCLoss::Hinge)
            .with_dual(DualMode::True)
            .fit(&x, &y)
            .is_err(),
        "l1+hinge,dual=True must reject (penalty='l1' and loss='hinge' unsupported)"
    );
    assert!(
        base()
            .with_penalty(LinearSVCPenalty::L1)
            .with_loss(LinearSVCLoss::Hinge)
            .with_dual(DualMode::False)
            .fit(&x, &y)
            .is_err(),
        "l1+hinge,dual=False must reject (penalty='l1' and loss='hinge' unsupported)"
    );

    // l1 + squared_hinge + dual=True: l1 requires dual=False (_base.py:1014).
    assert!(
        base()
            .with_penalty(LinearSVCPenalty::L1)
            .with_loss(LinearSVCLoss::SquaredHinge)
            .with_dual(DualMode::True)
            .fit(&x, &y)
            .is_err(),
        "l1+squared_hinge,dual=True must reject (l1 has no dual solver)"
    );

    // l2 + hinge + dual=False: hinge requires dual=True (_base.py:1013).
    assert!(
        base()
            .with_penalty(LinearSVCPenalty::L2)
            .with_loss(LinearSVCLoss::Hinge)
            .with_dual(DualMode::False)
            .fit(&x, &y)
            .is_err(),
        "l2+hinge,dual=False must reject (hinge has no primal solver)"
    );

    // --- Supported combos: sklearn fits; ferrolearn must be Ok. ---

    // l2 + squared_hinge + dual=True (solver 1).
    assert!(
        base()
            .with_penalty(LinearSVCPenalty::L2)
            .with_loss(LinearSVCLoss::SquaredHinge)
            .with_dual(DualMode::True)
            .fit(&x, &y)
            .is_ok(),
        "l2+squared_hinge,dual=True must fit (solver type 1)"
    );
    // l2 + squared_hinge + dual=False (solver 2).
    assert!(
        base()
            .with_penalty(LinearSVCPenalty::L2)
            .with_loss(LinearSVCLoss::SquaredHinge)
            .with_dual(DualMode::False)
            .fit(&x, &y)
            .is_ok(),
        "l2+squared_hinge,dual=False must fit (solver type 2)"
    );
    // l2 + hinge + dual=True (solver 3).
    assert!(
        base()
            .with_penalty(LinearSVCPenalty::L2)
            .with_loss(LinearSVCLoss::Hinge)
            .with_dual(DualMode::True)
            .fit(&x, &y)
            .is_ok(),
        "l2+hinge,dual=True must fit (solver type 3)"
    );
    // l1 + squared_hinge + dual=False (solver 5).
    assert!(
        base()
            .with_penalty(LinearSVCPenalty::L1)
            .with_loss(LinearSVCLoss::SquaredHinge)
            .with_dual(DualMode::False)
            .fit(&x, &y)
            .is_ok(),
        "l1+squared_hinge,dual=False must fit (solver type 5)"
    );
}

/// PIN 3 (#625, REQ-8 — dual='auto' resolution + l2 dual-invariance).
///
/// (a) l2+squared_hinge is dual-INVARIANT: the live oracle's
///     `LinearSVC(dual=True).coef_` == `LinearSVC(dual=False).coef_` (both fit
///     the same `0.5·‖w‖² + C·Σ L`). ferrolearn must agree across `DualMode::True`
///     and `DualMode::False` (within 1e-6) AND both match the live l2 oracle.
/// (b) `DualMode::Auto` on the 8x2 set (n_samples=8 ≥ n_features=2) resolves to
///     the primal-preferring path (`_validate_dual_parameter('auto',
///     'squared_hinge','l2','ovr',X) == False`, live oracle), and — since l2 is
///     dual-invariant — must reach the SAME l2 optimum.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   from sklearn.svm._classes import _validate_dual_parameter; \
///   X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   mt=LinearSVC(penalty='l2',loss='squared_hinge',dual=True,C=1.0,\
///       fit_intercept=True,max_iter=200000,tol=1e-10).fit(X,y); \
///   mf=LinearSVC(penalty='l2',loss='squared_hinge',dual=False,C=1.0,\
///       fit_intercept=True,max_iter=200000,tol=1e-10).fit(X,y); \
///   print(mt.coef_.tolist(), mt.intercept_.tolist()); \
///   print(mf.coef_.tolist(), mf.intercept_.tolist()); \
///   print(_validate_dual_parameter('auto','squared_hinge','l2','ovr',X))"
/// # dual=True  coef [[0.12835213612349905, 0.12835213612515478]] intercept [-1.1943776585903603]
/// # dual=False coef [[0.12835213611984458, 0.12835213611984475]] intercept [-1.1943776585907158]
/// # _validate_dual_parameter(...) == False
/// ```
#[test]
fn linear_svc_dual_auto_and_invariance() {
    // Live sklearn 1.5.2 l2/squared_hinge optimum (dual-invariant to ~3e-12).
    const SK_COEF_0: f64 = 0.12835213611984458;
    const SK_COEF_1: f64 = 0.12835213611984475;
    const SK_INTERCEPT: f64 = -1.1943776585907158;

    let (x, y) = binary_set();

    let fit_with = |dual: DualMode| {
        LinearSVC::<f64>::new()
            .with_penalty(LinearSVCPenalty::L2)
            .with_loss(LinearSVCLoss::SquaredHinge)
            .with_dual(dual)
            .with_c(1.0)
            .with_max_iter(200_000)
            .with_tol(1e-10)
            .fit(&x, &y)
            .unwrap()
    };

    let dual_true = fit_with(DualMode::True);
    let dual_false = fit_with(DualMode::False);
    let dual_auto = fit_with(DualMode::Auto);

    // (a) dual-invariance: ferrolearn dual=True and dual=False agree (1e-6).
    assert!(
        (dual_true.coefficients()[0] - dual_false.coefficients()[0]).abs() < 1e-6
            && (dual_true.coefficients()[1] - dual_false.coefficients()[1]).abs() < 1e-6
            && (dual_true.intercept() - dual_false.intercept()).abs() < 1e-6,
        "l2/squared_hinge must be dual-invariant: dual=True coef={:?} int={}, \
         dual=False coef={:?} int={}",
        dual_true.coefficients(),
        dual_true.intercept(),
        dual_false.coefficients(),
        dual_false.intercept()
    );

    // Both match the live l2 oracle (1e-2).
    for (label, f) in [("dual=True", &dual_true), ("dual=False", &dual_false)] {
        assert!(
            (f.coefficients()[0] - SK_COEF_0).abs() < 1e-2
                && (f.coefficients()[1] - SK_COEF_1).abs() < 1e-2
                && (f.intercept() - SK_INTERCEPT).abs() < 1e-2,
            "{label} must match live l2 oracle coef [[{SK_COEF_0}, {SK_COEF_1}]] \
             intercept [{SK_INTERCEPT}]; got coef={:?} intercept={}",
            f.coefficients(),
            f.intercept()
        );
    }

    // (b) dual='auto' on 8x2 (n_samples=8 >= n_features=2) resolves to the
    // primal-preferring path (_validate_dual_parameter -> False, live oracle)
    // and reaches the same dual-invariant l2 optimum.
    assert!(
        (dual_auto.coefficients()[0] - SK_COEF_0).abs() < 1e-2
            && (dual_auto.coefficients()[1] - SK_COEF_1).abs() < 1e-2
            && (dual_auto.intercept() - SK_INTERCEPT).abs() < 1e-2,
        "dual=Auto must match live l2 oracle coef [[{SK_COEF_0}, {SK_COEF_1}]] \
         intercept [{SK_INTERCEPT}]; got coef={:?} intercept={}",
        dual_auto.coefficients(),
        dual_auto.intercept()
    );
}

/// The fixed imbalanced binary 8x2 set for the #626 class_weight pins
/// (class0 count = 6, class1 count = 2 — imbalanced so class_weight bites).
fn imbalanced_binary_set() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.0, 1.5, 8.0, 8.0, 9.0, 9.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 0, 0, 1, 1];
    (x, y)
}

/// PIN 1 (#626, REQ-9 — binary class_weight, three DISTINCT optima). Certifies
/// `fn fit`'s binary `(cp, cn)` wiring (`cp = C·weights[idx(classes[1])]`,
/// `cn = C·weights[idx(classes[0])]`, mirroring
/// `train_one(&sub_prob, param, &model_->w[0], weighted_C[1], weighted_C[0], ...)`,
/// `sklearn/svm/src/liblinear/linear.cpp:2551`) plus `fn compute_class_weight`
/// (`sklearn.utils.compute_class_weight`, called at `_base.py:1179`). The three
/// class_weight settings produce THREE DISTINCT `coef_`/`intercept_` optima on
/// the imbalanced (6:2) set; the test asserts each matches the live oracle
/// within 1e-2 AND asserts the None vs Explicit intercept gap exceeds 0.02 — so
/// if class_weight were ignored (uniform C, `cp==cn`), the test would FAIL.
///
/// Oracle (live sklearn 1.5.2, squared_hinge, dual=True, C=1.0,
/// fit_intercept=True, max_iter=200000, tol=1e-10):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[1.5,1.5],[2.,1.5],[8.,8.],[9.,9.]]); \
///   y=np.array([0,0,0,0,0,0,1,1]); \
///   [print(repr(cw), LinearSVC(C=1.0,loss='squared_hinge',dual=True,fit_intercept=True,\
///      max_iter=200000,tol=1e-10,class_weight=cw).fit(X,y).coef_.tolist(), \
///      LinearSVC(C=1.0,loss='squared_hinge',dual=True,fit_intercept=True,max_iter=200000,\
///      tol=1e-10,class_weight=cw).fit(X,y).intercept_.tolist()) \
///    for cw in [None,'balanced',{0:1.0,1:5.0}]]"
/// # None        coef [[0.1005644741506319, 0.15957404219365467]]  int [-1.26346130747511]
/// # balanced    coef [[0.09936888941409959, 0.16666283617680763]] int [-1.2132032194284366]
/// # {0:1,1:5}   coef [[0.11058720550456055, 0.17164468740379785]] int [-1.2954689964234851]
/// ```
#[test]
fn linear_svc_class_weight_binary_parity() {
    // Live sklearn 1.5.2: imbalanced 8x2 set (counts 6:2), squared_hinge,
    // dual=True, C=1.0, fit_intercept=True, max_iter=200000, tol=1e-10.
    const SK_NONE_COEF: [f64; 2] = [0.1005644741506319, 0.15957404219365467];
    const SK_NONE_INT: f64 = -1.26346130747511;
    const SK_BAL_COEF: [f64; 2] = [0.09936888941409959, 0.16666283617680763];
    const SK_BAL_INT: f64 = -1.2132032194284366;
    const SK_EXP_COEF: [f64; 2] = [0.11058720550456055, 0.17164468740379785];
    const SK_EXP_INT: f64 = -1.2954689964234851;

    let (x, y) = imbalanced_binary_set();

    let fit_with = |cw: ClassWeight<f64>| {
        LinearSVC::<f64>::new()
            .with_c(1.0)
            .with_loss(LinearSVCLoss::SquaredHinge)
            .with_dual(DualMode::True)
            .with_max_iter(200_000)
            .with_tol(1e-10)
            .with_class_weight(cw)
            .fit(&x, &y)
            .unwrap()
    };

    let none = fit_with(ClassWeight::None);
    let balanced = fit_with(ClassWeight::Balanced);
    let explicit = fit_with(ClassWeight::Explicit(vec![(0, 1.0), (1, 5.0)]));

    // Each fit matches its live oracle (1e-2).
    let check = |label: &str,
                 f: &ferrolearn_linear::linear_svc::FittedLinearSVC<f64>,
                 sk_coef: &[f64; 2],
                 sk_int: f64| {
        let c = f.coefficients();
        assert!(
            (c[0] - sk_coef[0]).abs() < 1e-2
                && (c[1] - sk_coef[1]).abs() < 1e-2
                && (f.intercept() - sk_int).abs() < 1e-2,
            "{label}: sklearn coef [[{}, {}]] int [{sk_int}]; ferrolearn coef {:?} int {} \
             (gaps coef0 {:.5}, coef1 {:.5}, int {:.5})",
            sk_coef[0],
            sk_coef[1],
            c,
            f.intercept(),
            (c[0] - sk_coef[0]).abs(),
            (c[1] - sk_coef[1]).abs(),
            (f.intercept() - sk_int).abs()
        );
    };
    check("None", &none, &SK_NONE_COEF, SK_NONE_INT);
    check("Balanced", &balanced, &SK_BAL_COEF, SK_BAL_INT);
    check("Explicit{0:1,1:5}", &explicit, &SK_EXP_COEF, SK_EXP_INT);

    // The three optima are MUTUALLY DISTINCT: if class_weight were ignored
    // (uniform C, cp==cn), None and Explicit would coincide. The live oracle's
    // None vs Explicit intercept gap is |−1.26346 − (−1.29547)| ≈ 0.032.
    assert!(
        (none.intercept() - explicit.intercept()).abs() > 0.02,
        "None vs Explicit intercept must DIFFER (>0.02) — proves class_weight \
         is applied, not ignored: None int {}, Explicit int {} (gap {:.5})",
        none.intercept(),
        explicit.intercept(),
        (none.intercept() - explicit.intercept()).abs()
    );
    assert!(
        (none.intercept() - balanced.intercept()).abs() > 0.02,
        "None vs Balanced intercept must DIFFER (>0.02): None int {}, Balanced int {} (gap {:.5})",
        none.intercept(),
        balanced.intercept(),
        (none.intercept() - balanced.intercept()).abs()
    );
}

/// PIN 2 (#626, REQ-9 — balanced formula). Certifies `fn compute_class_weight`'s
/// `Balanced` branch computes `n_samples / (n_classes · count_c)` per class
/// (`sklearn/utils/class_weight.py:66-74`, called at `_base.py:1179`). On the
/// imbalanced 8x2 set (counts class0=6, class1=2, n_classes=2): class0 weight =
/// `8/(2·6) = 0.6666…`, class1 weight = `8/(2·2) = 2.0`. So `ClassWeight::Balanced`
/// must produce the SAME fit as `ClassWeight::Explicit([(0, 0.6666…), (1, 2.0)])`
/// to a tight 1e-6 (same per-class C scaling → same optimum).
///
/// Oracle (live sklearn 1.5.2): `compute_class_weight('balanced', classes=[0,1],
/// y=...)` returns `[0.6666666666666666, 2.0]`, and the `class_weight='balanced'`
/// fit equals the `class_weight={0:0.6666666666666666, 1:2.0}` fit:
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   from sklearn.utils.class_weight import compute_class_weight; \
///   X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[1.5,1.5],[2.,1.5],[8.,8.],[9.,9.]]); \
///   y=np.array([0,0,0,0,0,0,1,1]); \
///   print(compute_class_weight('balanced', classes=np.array([0,1]), y=y).tolist()); \
///   mb=LinearSVC(C=1.0,loss='squared_hinge',dual=True,fit_intercept=True,max_iter=200000,\
///       tol=1e-10,class_weight='balanced').fit(X,y); \
///   me=LinearSVC(C=1.0,loss='squared_hinge',dual=True,fit_intercept=True,max_iter=200000,\
///       tol=1e-10,class_weight={0:0.6666666666666666,1:2.0}).fit(X,y); \
///   print(np.allclose(mb.coef_, me.coef_, atol=1e-9), np.allclose(mb.intercept_, me.intercept_, atol=1e-9))"
/// # [0.6666666666666666, 2.0]
/// # True True
/// ```
#[test]
fn linear_svc_class_weight_balanced_formula() {
    let (x, y) = imbalanced_binary_set();

    let fit_with = |cw: ClassWeight<f64>| {
        LinearSVC::<f64>::new()
            .with_c(1.0)
            .with_loss(LinearSVCLoss::SquaredHinge)
            .with_dual(DualMode::True)
            .with_max_iter(200_000)
            .with_tol(1e-10)
            .with_class_weight(cw)
            .fit(&x, &y)
            .unwrap()
    };

    // balanced weights per the live oracle: class0 = 8/(2*6), class1 = 8/(2*2).
    let balanced = fit_with(ClassWeight::Balanced);
    let explicit_balanced = fit_with(ClassWeight::Explicit(vec![
        (0, 0.666_666_666_666_666_6),
        (1, 2.0),
    ]));

    let cb = balanced.coefficients();
    let ce = explicit_balanced.coefficients();
    assert!(
        (cb[0] - ce[0]).abs() < 1e-6
            && (cb[1] - ce[1]).abs() < 1e-6
            && (balanced.intercept() - explicit_balanced.intercept()).abs() < 1e-6,
        "Balanced must equal Explicit[(0,8/(2·6)),(1,8/(2·2))]=[(0,0.6667),(1,2.0)] \
         (compute_class_weight 'balanced', class_weight.py:66-74): \
         Balanced coef {cb:?} int {}, Explicit coef {ce:?} int {} \
         (gaps coef0 {:.8}, coef1 {:.8}, int {:.8})",
        balanced.intercept(),
        explicit_balanced.intercept(),
        (cb[0] - ce[0]).abs(),
        (cb[1] - ce[1]).abs(),
        (balanced.intercept() - explicit_balanced.intercept()).abs()
    );
}

/// PIN 3 (#626, REQ-9 — multiclass OvR `Cn = base-C` subtlety). The OvR contract
/// is `Cp = C·weights[k]`, `Cn = C` (the negative rest is UNWEIGHTED, the base
/// `C`), mirroring
/// `train_one(&sub_prob, param, w, weighted_C[i], param->C, ...)`
/// (`sklearn/svm/src/liblinear/linear.cpp:2571`). On a moderately-overlapping
/// 3-class set, weighting ONLY class 2 (`{0:1, 1:1, 2:5}`) measurably shifts
/// class 2's OvR `coef_`/`intercept_` (its positive group is up-weighted), while
/// classes 0 and 1 are LEFT BIT-IDENTICAL to the `None` fit — because their
/// negative rest (which includes class-2 samples) is the unweighted base `C`,
/// so up-weighting class 2 cannot leak into their subproblems. This is the exact
/// `Cn = base-C` behavior the pin certifies.
///
/// This is the "class_weight measurably shifts the multiclass optimum" branch:
/// the live oracle's class-2 row moves from coef `[-0.11756, 0.71436]` /
/// int -1.20484 (None) to coef `[-0.11494, 0.78267]` / int -1.24905
/// ({0:1,1:5→2:5}), confirming the OvR per-class `(cp,cn)` is threaded.
/// ferrolearn's per-class `coef_`/`intercept_` (`weight_vectors()`/`intercepts()`,
/// matching the oracle's per-class `coef_` rows) are pinned row-by-row within
/// 1e-2 for the Explicit class_weight.
///
/// Oracle (live sklearn 1.5.2, squared_hinge, dual=True, C=1.0,
/// fit_intercept=True, max_iter=500000, tol=1e-10, classes_=[0,1,2]):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
///   X=np.array([[0.,0.],[0.5,0.5],[0.,1.],[1.,0.],[0.3,0.7],[3.,0.],[3.5,0.5],\
///     [3.,1.],[2.7,0.3],[1.5,3.],[2.,3.5],[1.0,3.]]); \
///   y=np.array([0,0,0,0,0,1,1,1,1,2,2,2]); \
///   m=LinearSVC(C=1.0,loss='squared_hinge',dual=True,fit_intercept=True,max_iter=500000,\
///     tol=1e-10,class_weight={0:1.0,1:1.0,2:5.0}).fit(X,y); \
///   print(m.coef_.tolist()); print(m.intercept_.tolist())"
/// # coef [[-0.7277376501497536, -0.44707056917992416],
/// #       [0.7027143330071364, -0.3649784839404613],
/// #       [-0.1149375302371442, 0.7826703935798498]]
/// # int  [1.2859544208859206, -1.0055610724797077, -1.2490465741017618]
/// # (None fit class-2 row: coef [-0.11756204021108131, 0.7143621095823389] int -1.2048447007468126
/// #  — class-2 row SHIFTS, classes 0/1 stay bit-identical: Cn=base-C subtlety.)
/// ```
#[test]
fn linear_svc_class_weight_multiclass_ovr() {
    // Live sklearn 1.5.2: 3-class moderately-overlapping 12x2 set,
    // class_weight={0:1,1:1,2:5}, squared_hinge, dual=True, C=1.0,
    // fit_intercept=True. Per-class coef_ rows (classes_=[0,1,2]).
    const SK_COEF: [[f64; 2]; 3] = [
        [-0.7277376501497536, -0.44707056917992416],
        [0.7027143330071364, -0.3649784839404613],
        [-0.1149375302371442, 0.7826703935798498],
    ];
    const SK_INT: [f64; 3] = [1.2859544208859206, -1.0055610724797077, -1.2490465741017618];
    // None fit class-2 row (proves the shift the Cn=base-C path produces).
    const SK_NONE_CLASS2_COEF: [f64; 2] = [-0.11756204021108131, 0.7143621095823389];

    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.3, 0.7, 3.0, 0.0, 3.5, 0.5, 3.0, 1.0, 2.7,
            0.3, 1.5, 3.0, 2.0, 3.5, 1.0, 3.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2];

    let fit_with = |cw: ClassWeight<f64>| {
        LinearSVC::<f64>::new()
            .with_c(1.0)
            .with_loss(LinearSVCLoss::SquaredHinge)
            .with_dual(DualMode::True)
            .with_max_iter(500_000)
            .with_tol(1e-10)
            .with_class_weight(cw)
            .fit(&x, &y)
            .unwrap()
    };

    let explicit = fit_with(ClassWeight::Explicit(vec![(0, 1.0), (1, 1.0), (2, 5.0)]));
    assert_eq!(explicit.classes(), &[0, 1, 2], "classes_ must be [0,1,2]");

    // Per-class OvR coef_/intercept_ rows match the oracle (1e-2). The per-class
    // weight vectors are exposed via weight_vectors()/intercepts().
    let wv = explicit.weight_vectors();
    let ints = explicit.intercepts();
    assert_eq!(wv.len(), 3, "must have 3 OvR weight vectors");
    for k in 0..3 {
        assert!(
            (wv[k][0] - SK_COEF[k][0]).abs() < 1e-2
                && (wv[k][1] - SK_COEF[k][1]).abs() < 1e-2
                && (ints[k] - SK_INT[k]).abs() < 1e-2,
            "OvR class {k} coef/int: sklearn coef {:?} int {}; ferrolearn coef [{}, {}] int {} \
             (gaps coef0 {:.5}, coef1 {:.5}, int {:.5})",
            SK_COEF[k],
            SK_INT[k],
            wv[k][0],
            wv[k][1],
            ints[k],
            (wv[k][0] - SK_COEF[k][0]).abs(),
            (wv[k][1] - SK_COEF[k][1]).abs(),
            (ints[k] - SK_INT[k]).abs()
        );
    }

    // The Cn=base-C subtlety bites: with ONLY class 2 up-weighted, class 2's row
    // must SHIFT away from the unweighted None fit (its positive group penalty
    // grew), while classes 0/1 are unaffected (their negative rest is base C).
    // The class-2 coef[1] moves from ~0.7144 (None) to ~0.7827 (weighted).
    let none = fit_with(ClassWeight::None);
    let none_wv = none.weight_vectors();
    assert!(
        (none_wv[2][1] - SK_NONE_CLASS2_COEF[1]).abs() < 1e-2,
        "None fit class-2 coef[1]: sklearn {}, ferrolearn {} (gap {:.5})",
        SK_NONE_CLASS2_COEF[1],
        none_wv[2][1],
        (none_wv[2][1] - SK_NONE_CLASS2_COEF[1]).abs()
    );
    assert!(
        (wv[2][1] - none_wv[2][1]).abs() > 0.02,
        "class 2 row must SHIFT when class 2 is up-weighted (Cn=base-C, the OvR \
         positive group is the only thing weighted): weighted coef[1] {}, None coef[1] {} \
         (gap {:.5} — proves cp is threaded, not ignored)",
        wv[2][1],
        none_wv[2][1],
        (wv[2][1] - none_wv[2][1]).abs()
    );
    // Classes 0 and 1 stay (near-)identical to None: weighting class 2 does NOT
    // leak into their subproblems because their negative rest is base C.
    assert!(
        (wv[0][0] - none_wv[0][0]).abs() < 1e-3 && (wv[1][0] - none_wv[1][0]).abs() < 1e-3,
        "classes 0/1 must be unaffected by class-2 weighting (Cn=base-C): \
         class0 coef0 weighted {} vs None {}, class1 coef0 weighted {} vs None {}",
        wv[0][0],
        none_wv[0][0],
        wv[1][0],
        none_wv[1][0]
    );
}
