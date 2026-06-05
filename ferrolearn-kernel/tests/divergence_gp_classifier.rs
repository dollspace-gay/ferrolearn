//! Green-guard + divergence tests for
//! `ferrolearn-kernel::gp_classifier::GaussianProcessClassifier` against
//! scikit-learn 1.5.2 `sklearn.gaussian_process.GaussianProcessClassifier`
//! (`sklearn/gaussian_process/_gpc.py`).
//!
//! Translation unit #1930. All oracle expected values below were produced by a
//! live `sklearn` 1.5.2 call run from `/tmp` with `optimizer=None` (so the
//! kernel is FIXED and both sides use the SAME un-tuned hyperparameters) —
//! NEVER copied from ferrolearn's own output (R-CHAR-3). The exact oracle
//! command is recorded next to each constant block.
//!
//! Binary fixture (matches `.design/kernel/gp_classifier.md` ACs):
//!   X  = [[0],[0.5],[1],[3],[3.5],[4]]
//!   y  = [0, 0, 0, 1, 1, 1]
//!   Xs = [[0.5],[2.0],[3.5]]
//!   kernel = RBF(length_scale=1.0)
//!
//! Multi-class fixture:
//!   X  = [[0],[0.5],[1],[4],[4.5],[5],[8],[8.5],[9]]
//!   y  = [0,0,0,1,1,1,2,2,2]
//!   kernel = RBF(length_scale=1.0)
//!
//! GREEN GUARDS (must PASS — the deterministic SHIPPED slice, optimizer=None):
//!   - `green_binary_predict_unambiguous` (REQ-2) non-boundary decisions vs oracle
//!   - `green_binary_log_marginal`        (REQ-3) binary LML value vs oracle
//!   - `green_binary_pi_hat_shape`        (REQ-1) posterior mode pi_ shape vs oracle
//!   - `green_binary_score`               (REQ-4) accuracy vs oracle
//!   - `green_binary_classes`             (REQ-5) classes ordering vs oracle
//!   - `green_multiclass_classes`         (REQ-5) classes ordering vs oracle
//!
//! RED PINS (must FAIL now — single-file-fixable divergences; the fixer makes
//! them green so they are intentionally un-ignored):
//!   - `divergence_binary_predict_proba_squashing` (REQ-9) — ferrolearn squashes
//!     the predictive latent (f_bar*, var*) with the MacKay probit instead of
//!     sklearn's 5-term LAMBDAS/COEFS erf approximation (`_gpc.py:324-331`).
//!   - `divergence_binary_predict_latent_sign` (REQ-2) — ferrolearn's binary
//!     `predict` is argmax-of-`predict_proba`; sklearn's is the LATENT-SIGN
//!     decision `np.where(f_star > 0, classes_[1], classes_[0])` (`_gpc.py:291`).
//!     At a query point with f_star slightly > 0 these DISAGREE (sklearn → class
//!     1, ferrolearn → class 0).
//!   - `divergence_multiclass_lml_mean_vs_sum` (REQ-12) — ferrolearn SUMS the
//!     per-binary LMLs; sklearn MEANS them for n_classes>2 (`_gpc.py:743-749`).
//!
//! Live oracle (run from /tmp):
//! ```text
//! python3 -c "
//! import numpy as np
//! from sklearn.gaussian_process import GaussianProcessClassifier as GPC
//! from sklearn.gaussian_process.kernels import RBF
//! X=np.array([[0.],[0.5],[1.],[3.],[3.5],[4.]]); y=np.array([0,0,0,1,1,1])
//! Xs=np.array([[0.5],[2.0],[3.5]])
//! m=GPC(kernel=RBF(1.0), optimizer=None).fit(X,y)
//! print(m.predict(Xs).tolist())               # [0, 1, 1]  (latent sign)
//! print(np.argmax(m.predict_proba(Xs),axis=1).tolist()) # [0, 0, 1] (argmax proba)
//! print(m.predict_proba(Xs)[:,1].tolist())     # see ORACLE_POS below
//! print(repr(m.log_marginal_likelihood_value_))# -3.525884756588723
//! print(m.score(X,y))                          # 1.0
//! print(m.classes_.tolist())                   # [0, 1]
//! be=m.base_estimator_
//! print(be.kernel_(be.X_train_, Xs).T.dot(be.y_train_-be.pi_).tolist())
//! #  f_star = [-0.848.., +6.9e-17, +0.848..]  middle is slightly > 0
//! X3=np.array([[0.],[0.5],[1.],[4.],[4.5],[5.],[8.],[8.5],[9.]])
//! y3=np.array([0,0,0,1,1,1,2,2,2])
//! m3=GPC(kernel=RBF(1.0), optimizer=None).fit(X3,y3)
//! print(repr(m3.log_marginal_likelihood_value_))  # -5.246903194983737 (MEAN)
//! print([est.log_marginal_likelihood() for est in m3.base_estimator_.estimators_])
//! print(m3.classes_.tolist())                  # [0, 1, 2]
//! "
//! ```

use ferrolearn_core::{Fit, Predict};
use ferrolearn_kernel::{GaussianProcessClassifier, RBFKernel};
use ndarray::{Array1, Array2};

/// Binary fixture design `X = [[0],[0.5],[1],[3],[3.5],[4]]`.
fn x_binary() -> Array2<f64> {
    Array2::from_shape_vec((6, 1), vec![0.0, 0.5, 1.0, 3.0, 3.5, 4.0]).unwrap()
}

/// Binary fixture labels `y = [0,0,0,1,1,1]`.
fn y_binary() -> Array1<usize> {
    Array1::from_vec(vec![0usize, 0, 0, 1, 1, 1])
}

/// Query points `Xs = [[0.5],[2.0],[3.5]]`.
fn xs_binary() -> Array2<f64> {
    Array2::from_shape_vec((3, 1), vec![0.5, 2.0, 3.5]).unwrap()
}

/// Multi-class fixture design `X = [[0],[0.5],[1],[4],[4.5],[5],[8],[8.5],[9]]`.
fn x_multiclass() -> Array2<f64> {
    Array2::from_shape_vec((9, 1), vec![0.0, 0.5, 1.0, 4.0, 4.5, 5.0, 8.0, 8.5, 9.0]).unwrap()
}

/// Multi-class labels `y = [0,0,0,1,1,1,2,2,2]`.
fn y_multiclass() -> Array1<usize> {
    Array1::from_vec(vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2])
}

fn fit_binary() -> ferrolearn_kernel::FittedGaussianProcessClassifier<f64> {
    GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)))
        .fit(&x_binary(), &y_binary())
        .expect("binary fit")
}

// ---------------------------------------------------------------------------
// GREEN GUARDS (deterministic SHIPPED behaviors, optimizer=None) — must PASS
// ---------------------------------------------------------------------------

/// REQ-2 / AC-2: the UNAMBIGUOUS binary class decisions match the oracle.
///
/// At the two query points with |f_star| ≈ 0.848 (well away from the decision
/// boundary), the latent-sign decision and the argmax-of-proba decision agree,
/// so ferrolearn matches the oracle there regardless of the squash/decision-rule
/// divergence. Oracle: `m.predict(Xs) == [0, 1, 1]`; rows 0 and 2 are 0 and 1.
/// (Row 1, f_star ≈ +6.9e-17, is the boundary case pinned RED separately.)
#[test]
fn green_binary_predict_unambiguous() {
    let fitted = fit_binary();
    let pred = fitted.predict(&xs_binary()).unwrap();
    assert_eq!(
        pred[0], 0,
        "Xs[0] (f_star≈-0.848) should decide class 0 like sklearn"
    );
    assert_eq!(
        pred[2], 1,
        "Xs[2] (f_star≈+0.848) should decide class 1 like sklearn"
    );
}

/// REQ-3 / AC-3: binary log marginal likelihood value matches the oracle.
///
/// Oracle: `m.log_marginal_likelihood_value_ == -3.525884756588723`
/// (`_gpc.py:751-753`). The design doc claims algebraic equality (diff 0.0).
#[test]
fn green_binary_log_marginal() {
    const ORACLE_LML: f64 = -3.525884756588723;
    let fitted = fit_binary();
    let lml = fitted.log_marginal_likelihood();
    assert!(
        (lml - ORACLE_LML).abs() < 1e-6,
        "binary LML diverges: ferrolearn={lml}, sklearn={ORACLE_LML}"
    );
}

/// REQ-1 / AC-1: converged posterior mode shape matches the oracle's
/// `base_estimator_.pi_`.
///
/// Oracle pi_ = [0.3156024029, 0.2997621127, 0.326954663,
///               0.673045337, 0.7002378873, 0.6843975971].
///
/// `pi_hat` is a private field, so it cannot be read directly. The latent mode
/// is the input to both the (green-guarded) binary LML and the predict decision;
/// a wrong mode would shift the -3.5258847566 LML. This test additionally pins
/// the qualitative pi_ < 0.5 / > 0.5 split the oracle exhibits, via the
/// training-point predict_proba columns.
#[test]
fn green_binary_pi_hat_shape() {
    let fitted = fit_binary();
    let proba = fitted.predict_proba(&x_binary()).unwrap();
    for i in 0..3 {
        assert!(
            proba[[i, 1]] < 0.5,
            "row {i}: class-0 training point should have P(class=1) < 0.5"
        );
    }
    for i in 3..6 {
        assert!(
            proba[[i, 1]] > 0.5,
            "row {i}: class-1 training point should have P(class=1) > 0.5"
        );
    }
}

/// REQ-4 / AC-4: training-set accuracy matches the oracle `m.score(X,y) == 1.0`.
#[test]
fn green_binary_score() {
    const ORACLE_SCORE: f64 = 1.0;
    let fitted = fit_binary();
    let score = fitted.score(&x_binary(), &y_binary()).unwrap();
    assert!(
        (score - ORACLE_SCORE).abs() < 1e-12,
        "binary score diverges: ferrolearn={score}, sklearn={ORACLE_SCORE}"
    );
}

/// REQ-5 / AC-5: `classes()` ordering matches the oracle `m.classes_ == [0, 1]`.
#[test]
fn green_binary_classes() {
    let fitted = fit_binary();
    assert_eq!(
        fitted.classes(),
        &[0usize, 1],
        "binary classes must match sklearn `m.classes_`"
    );
}

/// REQ-5 / AC-5: multi-class `classes()` ordering matches `m.classes_ == [0,1,2]`.
#[test]
fn green_multiclass_classes() {
    let fitted = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)))
        .fit(&x_multiclass(), &y_multiclass())
        .unwrap();
    assert_eq!(
        fitted.classes(),
        &[0usize, 1, 2],
        "multi-class classes must match sklearn `m.classes_`"
    );
}

// ---------------------------------------------------------------------------
// RED PINS (single-file-fixable divergences) — must FAIL now, fixer greens
// ---------------------------------------------------------------------------

/// REQ-9 / AC-9 DIVERGENCE: `predict_proba` squashing.
///
/// ferrolearn's `predict_binary_proba` (gp_classifier.rs ~519-527) squashes the
/// predictive latent `(f_bar*, var*)` with the MacKay probit
/// `sigmoid(f_bar*/sqrt(1+pi*var*/8))` and FALSELY comments that this is
/// "the formulation used in scikit-learn's GaussianProcessClassifier".
/// sklearn (`_gpc.py:324-331`) uses the 5-term LAMBDAS/COEFS error-function
/// approximation of Williams & Barber:
///   alpha = 1/(2*var*); gamma = LAMBDAS*f_bar*;
///   integrals = sqrt(pi/alpha)*erf(gamma*sqrt(alpha/(alpha+LAMBDAS^2)))
///               /(2*sqrt(var* * 2*pi));
///   pi* = (COEFS*integrals).sum() + 0.5*COEFS.sum().
///
/// The predictive `f_bar*`/`var*` MATCH sklearn (they feed both squashes); only
/// the squash diverges, so this is a single-file fix.
///
/// Oracle positive-class column `m.predict_proba(Xs)[:,1]`:
///   [0.3217472091761806, 0.49999999500016656, 0.6782527808241525]
/// ferrolearn's MacKay squash gives `[0.31915674.., 0.5, 0.68084325..]` — a
/// ~2.6e-3 gap at rows 0 and 2, far above 1e-6. Row 1 is ~0.5 under both squashes
/// (f_bar*≈0), so it does NOT discriminate the squash — rows 0 and 2 do.
///
/// Tracking: see filed `-l blocker` issue. RED now; fixer makes it green.
#[test]
fn divergence_binary_predict_proba_squashing() {
    // Oracle positive-class probabilities, sklearn 1.5.2, optimizer=None.
    const ORACLE_POS: [f64; 3] = [0.3217472091761806, 0.49999999500016656, 0.6782527808241525];
    let fitted = fit_binary();
    let proba = fitted.predict_proba(&xs_binary()).unwrap();

    // Guard the shared-input premise: the f_bar*≈0 middle point lands at ~0.5
    // under EITHER squash, confirming the predictive mean/variance feed both
    // (i.e. the divergence is the squash, not an f_bar*/var* mismatch).
    assert!(
        (proba[[1, 1]] - 0.5).abs() < 1e-6,
        "middle query point should be ~0.5 (f_bar*≈0) under either squash; got {}",
        proba[[1, 1]]
    );

    // The divergence: rows 0 and 2 must match sklearn's LAMBDAS/COEFS values.
    for i in [0usize, 2] {
        assert!(
            (proba[[i, 1]] - ORACLE_POS[i]).abs() < 1e-6,
            "predict_proba squash diverges at row {i}: ferrolearn P(class=1)={}, \
             sklearn LAMBDAS/COEFS={} (MacKay probit gap)",
            proba[[i, 1]],
            ORACLE_POS[i]
        );
    }
}

/// REQ-2 DIVERGENCE: binary `predict` decision rule (latent sign vs argmax-proba).
///
/// sklearn's binary `predict` does `np.where(f_star > 0, classes_[1],
/// classes_[0])` (`_gpc.py:291`) — a SIGN test on the latent predictive mean,
/// NOT an argmax of `predict_proba`. ferrolearn's `Predict::predict`
/// (gp_classifier.rs) argmaxes `predict_proba`'s columns. The design doc (REQ-2)
/// claims these always agree "because the squash is monotone and `[1−π, π]`
/// crosses at `π=0.5 ⟺ f_bar*=0`" — but that is FALSE for sklearn's actual
/// `predict` AND for the LAMBDAS/COEFS squash: the squash does not cross exactly
/// at f_star=0.
///
/// At Xs[1]=[2.0], f_star = +6.938893903907228e-17 (slightly POSITIVE), so
/// sklearn's `predict` → class 1, while its `predict_proba(Xs)[1,1] =
/// 0.49999999500016656` is < 0.5, so argmax → class 0. The whole vector:
///   oracle `m.predict(Xs)` = [0, 1, 1]
///   oracle argmax(predict_proba) = [0, 0, 1]
/// ferrolearn returns the argmax form [0, 0, 1], diverging at row 1.
///
/// Single-file fix: the binary branch must decide by the latent sign of f_bar*,
/// not by argmax of the squashed proba.
///
/// Tracking: see filed `-l blocker` issue. RED now; fixer makes it green.
#[test]
fn divergence_binary_predict_latent_sign() {
    // Oracle `m.predict(Xs)` — the latent-sign decision (NOT argmax-of-proba).
    const ORACLE_PREDICT: [usize; 3] = [0, 1, 1];
    let fitted = fit_binary();
    let pred = fitted.predict(&xs_binary()).unwrap();
    assert_eq!(
        pred.to_vec(),
        ORACLE_PREDICT.to_vec(),
        "binary predict must match sklearn's latent-sign `np.where(f_star>0, \
         classes_[1], classes_[0])` (_gpc.py:291); ferrolearn argmaxes \
         predict_proba and diverges at the f_star≈+0 boundary point"
    );
}

/// REQ-12 / AC-12 DIVERGENCE: multi-class LML mean-vs-sum.
///
/// sklearn's `GaussianProcessClassifier.log_marginal_likelihood` returns the
/// MEAN of the per-binary LMLs for n_classes>2
/// (`np.mean([... ])`, `_gpc.py:743-749`). ferrolearn's
/// `FittedGaussianProcessClassifier::log_marginal_likelihood` SUMS them
/// (`.fold(F::zero(), |a,b| a+b)`).
///
/// Oracle (3-class fixture): `m.log_marginal_likelihood_value_` =
///   -5.246903194983737 (MEAN of [-5.24569279925462, -5.24932398644197,
///   -5.245692799254621]); ferrolearn returns the SUM = -15.740709584951212,
///   i.e. ~3x the oracle mean for 3 balanced classes.
///
/// Tracking: see filed `-l blocker` issue. RED now; fixer makes it green.
#[test]
fn divergence_multiclass_lml_mean_vs_sum() {
    const ORACLE_LML_MEAN: f64 = -5.246903194983737;
    let fitted = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)))
        .fit(&x_multiclass(), &y_multiclass())
        .unwrap();
    let lml = fitted.log_marginal_likelihood();
    assert!(
        (lml - ORACLE_LML_MEAN).abs() < 1e-5,
        "multi-class LML diverges: ferrolearn={lml} (SUM), \
         sklearn={ORACLE_LML_MEAN} (MEAN of per-binary LMLs)"
    );
}

// ---------------------------------------------------------------------------
// RE-AUDIT GREEN GUARDS (unit #1930 re-audit of fixes #1931 / #1932 / #1933 /
// #1939). These pin the *fixed* behavior across a RANGE — not just the original
// single fixtures — so a regression in any of the four fixes turns one RED.
//
// A SEPARATE oracle dataset/kernel from the fixtures above (RBF(1.5) on
// X=[[0],[1],[2],[3],[4],[5]]) is used so these are not re-pins of the same
// numbers. Oracle (run from /tmp, sklearn 1.5.2, optimizer=None):
// ```text
// python3 -c "
// import numpy as np
// from sklearn.gaussian_process import GaussianProcessClassifier as GPC
// from sklearn.gaussian_process.kernels import RBF
// X=np.array([[0.],[1.],[2.],[3.],[4.],[5.]]); y=np.array([0,0,0,1,1,1])
// m=GPC(kernel=RBF(1.5), optimizer=None).fit(X,y)
// Xs=np.array([[5.0],[0.0],[2.5],[100.0],[2.0]])
// print(m.predict_proba(Xs).tolist())
// print(m.predict(Xs).tolist())
// # near-boundary fixture:
// Xs2=np.array([[2.5],[2.49],[2.51]])
// print(m.predict(Xs2).tolist())            # [1, 0, 1]
// "
// ```
// ---------------------------------------------------------------------------

/// Re-audit fixture: `X = [[0],[1],[2],[3],[4],[5]]`, `y = [0,0,0,1,1,1]`,
/// `kernel = RBF(length_scale=1.5)`. Distinct from the `RBF(1.0)` fixture above.
fn x_audit() -> Array2<f64> {
    Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()
}

fn y_audit() -> Array1<usize> {
    Array1::from_vec(vec![0usize, 0, 0, 1, 1, 1])
}

fn fit_audit() -> ferrolearn_kernel::FittedGaussianProcessClassifier<f64> {
    GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.5)))
        .fit(&x_audit(), &y_audit())
        .expect("audit binary fit")
}

/// #1931 RE-AUDIT (REQ-9): the 5-term LAMBDAS/COEFS erf approximation must match
/// sklearn's `predict_proba` (`_gpc.py:324-331`) ACROSS THE PROBABILITY RANGE,
/// not just at one point. The query points stress, in order:
///   row 0 `[5.0]`   — HIGH-confidence class 1 (coincident with a class-1 train
///                     point), proba P(class=1) ≈ 0.6475;
///   row 1 `[0.0]`   — HIGH-confidence class 0 (coincident with a class-0 train
///                     point), proba P(class=1) ≈ 0.3525;
///   row 2 `[2.5]`   — decision boundary, P(class=1) ≈ 0.5;
///   row 3 `[100.0]` — FAR from all training data (huge predictive variance):
///                     sklearn's erf integrals → 0 so P(class=1) → 0.5. This
///                     stresses `alpha = 1/(2·var)` as var→∞ and the var floor;
///                     ferrolearn must NOT produce NaN/Inf here;
///   row 4 `[2.0]`   — COINCIDES with a training point (predictive variance ≈ 0):
///                     stresses the `var <= 0` floor the #1931 fix added;
///                     ferrolearn must NOT produce NaN/Inf and must match sklearn.
///
/// Oracle `m.predict_proba(Xs)[:,1]` (sklearn 1.5.2, optimizer=None, RBF(1.5)):
///   [0.647507760095209, 0.3524922299051241, 0.4999999950001666,
///    0.49999999500016656, 0.4335062776756615]
/// Element-wise tolerance 1e-6 (sklearn-vs-ferrolearn agree to ~1e-13 in
/// practice).
#[test]
fn green_audit_predict_proba_range() {
    // Oracle positive-class column, RBF(1.5), optimizer=None.
    const ORACLE_POS: [f64; 5] = [
        0.647507760095209,
        0.3524922299051241,
        0.4999999950001666,
        0.49999999500016656,
        0.4335062776756615,
    ];
    let fitted = fit_audit();
    let xs = Array2::from_shape_vec((5, 1), vec![5.0, 0.0, 2.5, 100.0, 2.0]).unwrap();
    let proba = fitted.predict_proba(&xs).unwrap();

    for i in 0..5 {
        // No NaN/Inf anywhere — especially the far (row 3) and coincident
        // (rows 0,1,4) points that stress the var floor and alpha=1/(2var).
        assert!(
            proba[[i, 0]].is_finite() && proba[[i, 1]].is_finite(),
            "row {i}: predict_proba produced non-finite value [{}, {}]",
            proba[[i, 0]],
            proba[[i, 1]]
        );
        // Rows sum to 1 (1 - pi_star, pi_star).
        let s = proba[[i, 0]] + proba[[i, 1]];
        assert!((s - 1.0).abs() < 1e-12, "row {i}: proba sums to {s}, not 1");
        // Match sklearn's LAMBDAS/COEFS erf approximation element-wise.
        assert!(
            (proba[[i, 1]] - ORACLE_POS[i]).abs() < 1e-6,
            "row {i}: predict_proba erf-approx diverges: ferrolearn P(class=1)={}, \
             sklearn LAMBDAS/COEFS={}",
            proba[[i, 1]],
            ORACLE_POS[i]
        );
    }
}

/// #1931 RE-AUDIT — far-point determinism guard.
///
/// At a point arbitrarily far from training data (`x = 1e6`), sklearn's erf
/// approximation saturates the predictive probability to exactly the
/// "no information" value `0.5 + tiny` (the erf integrals → 0 as var→∞).
/// ferrolearn must reproduce a finite value ≈ 0.5, NOT NaN/Inf from the
/// `alpha = 1/(2·var)` term. Oracle:
/// ```text
/// m.predict_proba(np.array([[1e6]]))[:,1]  # -> 0.49999999500016656
/// ```
#[test]
fn green_audit_predict_proba_far_point_no_nan() {
    const ORACLE_FAR_POS: f64 = 0.49999999500016656;
    let fitted = fit_audit();
    let xs = Array2::from_shape_vec((1, 1), vec![1.0e6]).unwrap();
    let proba = fitted.predict_proba(&xs).unwrap();
    assert!(
        proba[[0, 1]].is_finite(),
        "far point produced non-finite P(class=1)={}",
        proba[[0, 1]]
    );
    assert!(
        (proba[[0, 1]] - ORACLE_FAR_POS).abs() < 1e-6,
        "far point: ferrolearn P(class=1)={}, sklearn={ORACLE_FAR_POS}",
        proba[[0, 1]]
    );
}

/// #1932 RE-AUDIT (REQ-2): binary `predict` decides by the SIGN of the latent
/// predictive mean f_star (`np.where(f_star > 0, classes_[1], classes_[0])`,
/// `_gpc.py:291`), NOT argmax of the squashed `predict_proba`. This guard pins a
/// query point where the two rules DISAGREE so it cannot pass by accident:
///   at `x = 2.5`, f_star = +5.55e-17 (slightly POSITIVE) ⇒ sklearn predict →
///   class 1, while `predict_proba(2.5)[1] = 0.49999999500016656 < 0.5` ⇒
///   argmax → class 0.
/// Flanking points `2.49` (f_star < 0 ⇒ class 0) and `2.51` (f_star > 0 ⇒
/// class 1) confirm the sign crossing.
/// Oracle `m.predict([[2.5],[2.49],[2.51]])` = `[1, 0, 1]`;
/// `argmax(predict_proba)` would give `[0, 0, 1]`.
#[test]
fn green_audit_predict_latent_sign_boundary() {
    const ORACLE_PREDICT: [usize; 3] = [1, 0, 1];
    let fitted = fit_audit();
    let xs = Array2::from_shape_vec((3, 1), vec![2.5, 2.49, 2.51]).unwrap();
    let pred = fitted.predict(&xs).unwrap();
    assert_eq!(
        pred.to_vec(),
        ORACLE_PREDICT.to_vec(),
        "binary predict must use the latent sign of f_star (_gpc.py:291), not \
         argmax of predict_proba: at x=2.5 (f_star=+5.55e-17>0) sklearn → class 1 \
         while proba[1]<0.5 would argmax → class 0"
    );
    // Discriminating cross-check: at x=2.5 the proba argmax is class 0, proving
    // the decision was made on the latent sign, not on the squashed proba.
    let proba = fitted.predict_proba(&xs).unwrap();
    assert!(
        proba[[0, 1]] < 0.5,
        "premise: at x=2.5 P(class=1)={} must be < 0.5 so argmax-of-proba would \
         pick class 0; the latent-sign rule still picks class 1",
        proba[[0, 1]]
    );
}

/// #1939 RE-AUDIT (REQ-13): multi-class OvR `predict_proba` now that the #1931
/// squash is fixed. sklearn fits one binary class-c-vs-rest GP per class, takes
/// each one's class-1 probability, stacks them and NORMALIZES rows to sum to 1
/// (`OneVsRestClassifier.predict_proba` over `_gpc.py:779-807`). ferrolearn does
/// the same (`raw[i,c] = binary_c.proba`, then row-normalize). This guard checks
/// the full `n_samples × n_classes` matrix element-wise against the oracle,
/// including inter-cluster ambiguous query points (rows 3, 4).
///
/// 3-class fixture `X=[[0],[0.5],[1],[4],[4.5],[5],[8],[8.5],[9]]`,
/// `y=[0,0,0,1,1,1,2,2,2]`, `RBF(1.0)`, `optimizer=None`.
/// `Xs = [[0.0],[4.5],[9.0],[2.5],[6.5]]`. Oracle `m.predict_proba(Xs)`:
///   row0 [0.49454661367581776, 0.2526814432944745,  0.2527719430297078]
///   row1 [0.24259248670282157, 0.5148150265942707,  0.2425924867029077]
///   row2 [0.25277194302977124, 0.252681443294453,   0.49454661367577574]
///   row3 [0.3483909236677405,  0.3483909336812822,  0.3032181426509773]
///   row4 [0.3032181426509635,  0.34839093368128915, 0.3483909236677474]
#[test]
fn green_audit_multiclass_ovr_predict_proba() {
    #[rustfmt::skip]
    const ORACLE: [[f64; 3]; 5] = [
        [0.49454661367581776, 0.2526814432944745,  0.2527719430297078],
        [0.24259248670282157, 0.5148150265942707,  0.2425924867029077],
        [0.25277194302977124, 0.252681443294453,   0.49454661367577574],
        [0.3483909236677405,  0.3483909336812822,  0.3032181426509773],
        [0.3032181426509635,  0.34839093368128915, 0.3483909236677474],
    ];
    let fitted = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)))
        .fit(&x_multiclass(), &y_multiclass())
        .unwrap();
    let xs = Array2::from_shape_vec((5, 1), vec![0.0, 4.5, 9.0, 2.5, 6.5]).unwrap();
    let proba = fitted.predict_proba(&xs).unwrap();
    assert_eq!(proba.dim(), (5, 3));
    for i in 0..5 {
        let s: f64 = (0..3).map(|c| proba[[i, c]]).sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "row {i}: OvR proba sums to {s}, not 1"
        );
        for c in 0..3 {
            assert!(
                (proba[[i, c]] - ORACLE[i][c]).abs() < 1e-5,
                "multi-class OvR predict_proba diverges at [{i},{c}]: \
                 ferrolearn={}, sklearn={}",
                proba[[i, c]],
                ORACLE[i][c]
            );
        }
    }
}

/// #1933 RE-AUDIT (REQ-12): 3-class `log_marginal_likelihood` returns the MEAN
/// of the per-binary LMLs (`np.mean([...])`, `_gpc.py:743-749`). Confirms the
/// already-pinned mean behavior stays green on the same fixture.
/// Oracle `m.log_marginal_likelihood_value_` = -5.246903194983737, the mean of
/// per-binary `[-5.24569279925462, -5.24932398644197, -5.245692799254621]`.
#[test]
fn green_audit_multiclass_lml_is_mean() {
    const ORACLE_LML_MEAN: f64 = -5.246903194983737;
    #[rustfmt::skip]
    const PER_BINARY: [f64; 3] = [-5.24569279925462, -5.24932398644197, -5.245692799254621];
    // Re-derive the mean from the oracle per-binary list (independent of the
    // ferrolearn aggregate) to anchor the expected value (R-CHAR-3).
    let derived_mean = (PER_BINARY[0] + PER_BINARY[1] + PER_BINARY[2]) / 3.0;
    assert!(
        (derived_mean - ORACLE_LML_MEAN).abs() < 1e-9,
        "oracle self-consistency: mean of per-binary LMLs = {derived_mean}, \
         expected {ORACLE_LML_MEAN}"
    );
    let fitted = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)))
        .fit(&x_multiclass(), &y_multiclass())
        .unwrap();
    let lml = fitted.log_marginal_likelihood();
    assert!(
        (lml - ORACLE_LML_MEAN).abs() < 1e-5,
        "3-class LML must be the MEAN: ferrolearn={lml}, sklearn={ORACLE_LML_MEAN}"
    );
    // Guard against a SUM regression: the sum would be ~3x as negative.
    let implied_sum = ORACLE_LML_MEAN * 3.0;
    assert!(
        (lml - implied_sum).abs() > 1.0,
        "LML must be the MEAN, not the SUM ({implied_sum}); got {lml}"
    );
}
