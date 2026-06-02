//! Divergence pins for the SGD family (`SGDClassifier` / `SGDRegressor`)
//! against the live scikit-learn 1.5.2 oracle
//! (`sklearn/linear_model/_stochastic_gradient.py`,
//! `sklearn/linear_model/_sgd_fast.pyx.tp`, commit 156ef14).
//!
//! # Why these tests are single-sample (the RNG-shuffle barrier)
//!
//! Multi-sample, multi-epoch SGD shuffles the sample order each epoch.
//! sklearn permutes with numpy's `our_rand_r`
//! (`_sgd_fast.pyx.tp:579-580`); ferrolearn permutes with
//! `rand::rngs::StdRng` (`ferrolearn-linear/src/sgd.rs` `train_binary_sgd` /
//! `train_regressor_sgd`, `indices.shuffle(&mut rng)`). The two PRNGs produce
//! different permutations from the same seed, so a multi-sample fitted-weight
//! trajectory is NOT cross-impl comparable.
//!
//! Every behavioral pin below therefore uses **n_samples = 1**: a shuffle of a
//! single element is the identity in BOTH implementations, so the resulting
//! `coef_`/`intercept_` after a fixed `max_iter` are fully deterministic and
//! expose the exact per-step eta (schedule + t0) and the exact L2 update rule
//! WITHOUT any RNG contamination. The classifier-default and loss-formula pins
//! are deterministic by construction (no fit, or value-grid).
//!
//! Every expected value is produced by RUNNING scikit-learn 1.5.2 (the live
//! oracle), never copied from the ferrolearn side (goal.md R-CHAR-3). The exact
//! python invocation is recorded above each block of oracle constants.

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::sgd::{
    Hinge, LearningRateSchedule, Loss, Penalty, RegressorLoss, SGDClassifier, SGDRegressor,
};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// REQ-7 / #527 — `optimal` learning-rate schedule omits the `t0` offset.
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:565-570,592`
//   `typw = np.sqrt(1.0 / np.sqrt(alpha))`
//   `initial_eta0 = typw / max(1.0, loss.dloss(1.0, -typw))`
//   `optimal_init = 1.0 / (initial_eta0 * alpha)`
//   `eta = 1.0 / (alpha * (optimal_init + t - 1))`
// so the FIRST sample (t=1) sees `eta = 1/(alpha*optimal_init) = initial_eta0`.
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `compute_lr`
//   `LearningRateSchedule::Optimal => F::one() / (alpha * t_f)`
// — the `optimal_init` offset is absent, so the first sample (t=1) sees
//   `eta = 1/(alpha*1)` (= 100 for alpha=0.01) instead of `initial_eta0` (≈0.76).
//
// Single sample, alpha=0.01, epsilon_insensitive loss (dloss ∈ {-1,0,1},
// bounded so the divergence is in eta, not in an exploding gradient).
// ---------------------------------------------------------------------------

/// Oracle invocation:
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// m=SGDRegressor(loss='epsilon_insensitive',epsilon=0.1,penalty='l2',alpha=0.01,l1_ratio=0.15, \
///   learning_rate='optimal',max_iter=5,tol=None,shuffle=False,fit_intercept=True,random_state=0) \
///   .fit(np.array([[2.0,-1.0]]),np.array([3.0])); print(m.coef_.tolist(), m.intercept_.tolist())"
/// ```
/// -> coef [5.6143854881471125, -2.8071927440735562], intercept [2.9900296495184233]
///
/// ferrolearn's `compute_lr(Optimal)` returns `1/(alpha*t)`, so at alpha=0.01
/// the first step uses eta=100 instead of sklearn's ≈0.76 — the fit diverges
/// to roughly `[-40.0, 20.0]`, far from the sklearn oracle.
#[test]
fn sgd_optimal_schedule_t0_offset() {
    const SK_COEF0: f64 = 5.6143854881471125;
    const SK_COEF1: f64 = -2.8071927440735562;
    const SK_INTERCEPT: f64 = 2.9900296495184233;

    let x = Array2::from_shape_vec((1, 2), vec![2.0, -1.0]).unwrap();
    let y = Array1::from_vec(vec![3.0]);

    // ferrolearn's Optimal schedule ignores eta0, but validate requires eta0>0.
    let model = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::EpsilonInsensitive(0.1))
        .with_learning_rate(LearningRateSchedule::Optimal)
        .with_eta0(0.5)
        .with_alpha(0.01)
        .with_max_iter(5)
        .with_tol(-1.0) // disable convergence early-exit (tol=None analog)
        .with_random_state(0);

    let fitted = model.fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    let intercept = fitted.intercept();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-9
            && (coef[1] - SK_COEF1).abs() < 1e-9
            && (intercept - SK_INTERCEPT).abs() < 1e-9,
        "optimal schedule t0 offset diverges: sklearn coef=[{SK_COEF0}, {SK_COEF1}] \
         intercept={SK_INTERCEPT}; ferrolearn coef={coef:?} intercept={intercept}"
    );
}

// ---------------------------------------------------------------------------
// REQ-4 / #525 — L2 penalty `max(0, ...)` clamp on the `wscale` shrink.
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:632-635`
//   `w.scale(max(0, 1.0 - ((1.0 - l1_ratio) * eta * alpha)))`
// With `penalty='l2'` (l1_ratio forced to 0.0, `:559`) the shrink factor is
// `max(0, 1 - eta*alpha)`. When `eta*alpha > 1` the factor is CLAMPED to 0,
// zeroing the weight vector before the gradient add.
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `train_regressor_sgd`
//   `weights[j] = weights[j] - eta * (grad * xi[j] + alpha * weights[j])`
// i.e. `w_new = (1 - eta*alpha) * w - eta*grad*x`, which has NO `max(0, ·)`
// clamp: with `eta*alpha > 1` the shrink factor `(1 - eta*alpha)` goes NEGATIVE
// (a sign-flipping blow-up) instead of being clamped to 0.
//
// constant schedule => eta=eta0 is IDENTICAL in both impls, so the only
// difference is the clamp. eta0=1.0, alpha=2.0 => 1-eta*alpha = -1 < 0.
// (Note: for the default penalty='l2' WITHOUT the clamp the two update forms
//  are algebraically identical `(1-eta*alpha)w - eta*g*x`; the clamp is the
//  genuine observable divergence.)
// ---------------------------------------------------------------------------

/// Oracle invocation:
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// m=SGDRegressor(loss='epsilon_insensitive',epsilon=0.1,penalty='l2',alpha=2.0,l1_ratio=0.15, \
///   learning_rate='constant',eta0=1.0,max_iter=3,tol=None,shuffle=False,fit_intercept=True,random_state=0) \
///   .fit(np.array([[2.0,-1.0]]),np.array([3.0])); print(m.coef_.tolist(), m.intercept_.tolist())"
/// ```
/// -> coef [2.0, -1.0], intercept [1.0]
///
/// ferrolearn's inline `(1-eta*alpha)*w` (= -1 * w here) sign-flips and
/// amplifies instead of clamping to zero, yielding coef ≈ [6.0, -3.0].
#[test]
fn sgd_l2_wscale_clamp() {
    const SK_COEF0: f64 = 2.0;
    const SK_COEF1: f64 = -1.0;
    const SK_INTERCEPT: f64 = 1.0;

    let x = Array2::from_shape_vec((1, 2), vec![2.0, -1.0]).unwrap();
    let y = Array1::from_vec(vec![3.0]);

    let model = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::EpsilonInsensitive(0.1))
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(1.0)
        .with_alpha(2.0)
        .with_max_iter(3)
        .with_tol(-1.0)
        .with_random_state(0);

    let fitted = model.fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    let intercept = fitted.intercept();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-9
            && (coef[1] - SK_COEF1).abs() < 1e-9
            && (intercept - SK_INTERCEPT).abs() < 1e-9,
        "L2 wscale max(0,.) clamp diverges: sklearn coef=[{SK_COEF0}, {SK_COEF1}] \
         intercept={SK_INTERCEPT}; ferrolearn coef={coef:?} intercept={intercept}"
    );
}

// ---------------------------------------------------------------------------
// REQ-9 / #529 — `SGDClassifier` default learning-rate / eta0 / power_t.
//
// sklearn site: `sklearn/linear_model/_stochastic_gradient.py:1242-1244`
//   `SGDClassifier.__init__` defaults: `learning_rate="optimal"`, `eta0=0.0`,
//   `power_t=0.5`.
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `SGDClassifier::new`
//   `learning_rate: LearningRateSchedule::InvScaling, eta0: 0.01, power_t: 0.25`.
// ---------------------------------------------------------------------------

/// Oracle invocation:
/// ```text
/// python3 -c "from sklearn.linear_model import SGDClassifier; m=SGDClassifier(); \
///   print(m.learning_rate, m.eta0, m.power_t)"
/// ```
/// -> `optimal 0.0 0.5`
///
/// ferrolearn `SGDClassifier::new()` reports `InvScaling`, `0.01`, `0.25`.
#[test]
fn sgd_classifier_default_learning_rate() {
    // sklearn defaults (live oracle).
    const SK_ETA0: f64 = 0.0;
    const SK_POWER_T: f64 = 0.5;

    let clf = SGDClassifier::<f64>::new();

    let is_optimal = matches!(clf.learning_rate, LearningRateSchedule::Optimal);
    assert!(
        is_optimal
            && (clf.eta0 - SK_ETA0).abs() < 1e-12
            && (clf.power_t - SK_POWER_T).abs() < 1e-12,
        "SGDClassifier defaults diverge: sklearn learning_rate='optimal' eta0={SK_ETA0} \
         power_t={SK_POWER_T}; ferrolearn learning_rate={:?} eta0={} power_t={}",
        clf.learning_rate,
        clf.eta0,
        clf.power_t
    );
}

// ---------------------------------------------------------------------------
// REQ-1 — Hinge `gradient` at the margin boundary z = y*p = threshold.
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:222-226`
//   `cdef double z = p * y`
//   `if z <= self.threshold:` (NON-strict)
//   `    return -y`
//   `return 0.0`
// At z == threshold (=1.0) sklearn returns `dloss = -y` (NONZERO).
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `impl Loss for Hinge` `gradient`
//   `if margin < F::one() { -y_true } else { F::zero() }` (STRICT `<`)
// At margin == 1.0 ferrolearn returns `0.0` — the boundary sample produces NO
// gradient, so the SGD step it would have triggered in sklearn is dropped.
//
// (The four classifier losses Log, ModifiedHuber, SquaredError were verified
//  against the live `_sgd_fast` oracle and match to f64 precision — only Hinge
//  diverges, at the z==threshold boundary, so only this one is pinned.)
// ---------------------------------------------------------------------------

/// Oracle invocation:
/// ```text
/// python3 -c "from sklearn.linear_model._sgd_fast import Hinge; h=Hinge(1.0); \
///   print(h.py_dloss(1.0,1.0), h.py_dloss(-1.0,-1.0))"
/// ```
/// where `py_dloss(p, y)`: at (p=1, y=1) z=1 -> dloss=-1.0; at (p=-1, y=-1)
/// z=1 -> dloss=1.0. ferrolearn `Hinge.gradient` returns 0.0 in both cases.
#[test]
fn sgd_hinge_gradient_boundary() {
    // sklearn dloss at z == threshold == 1.0 (live oracle).
    const SK_DLOSS_YPOS: f64 = -1.0; // y=1, p=1
    const SK_DLOSS_YNEG: f64 = 1.0; // y=-1, p=-1

    let h = Hinge;
    // ferrolearn signature is gradient(y_true, y_pred).
    let g_pos = <Hinge as Loss<f64>>::gradient(&h, 1.0, 1.0);
    let g_neg = <Hinge as Loss<f64>>::gradient(&h, -1.0, -1.0);

    assert!(
        (g_pos - SK_DLOSS_YPOS).abs() < 1e-12 && (g_neg - SK_DLOSS_YNEG).abs() < 1e-12,
        "Hinge gradient at margin boundary z=1 diverges: sklearn dloss=({SK_DLOSS_YPOS}, \
         {SK_DLOSS_YNEG}); ferrolearn=({g_pos}, {g_neg})"
    );
}

// ---------------------------------------------------------------------------
// REQ-5 / #526 — `penalty='l1'` via the Tsuruoka cumulative-penalty
// truncated gradient.
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:560-561,656-658,750-778`
//   `elif penalty_type == L1: l1_ratio = 1.0`            (`:560-561`)
//   `u += (l1_ratio * eta * alpha)`                      (`:657`)
//   `l1penalty(w, q, x_ind, xnnz, u)`                    (`:658`)
//   where `l1penalty` (`:750-778`) pushes each touched coordinate toward 0 by
//   `(u ± q[idx])` (clamped not to cross 0) and accumulates the applied penalty
//   in `q[idx]`. With penalty='l1', eff=1.0 so the L2 shrink factor is
//   `max(0, 1 - 0) = 1` (no multiplicative L2 decay).
//
// Single sample, squared_error loss, constant schedule (eta=eta0 identical in
// both impls), so the only moving parts are the gradient add and the L1
// truncation — fully deterministic.
// ---------------------------------------------------------------------------

/// Oracle invocation:
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// X=np.array([[2.0,-1.0]]); y=np.array([3.0]); \
/// m=SGDRegressor(loss='squared_error',penalty='l1',alpha=0.1,l1_ratio=0.15, \
///   learning_rate='constant',eta0=0.1,max_iter=3,tol=None,random_state=0, \
///   fit_intercept=True,shuffle=False).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// ```
/// -> coef [0.9204, -0.4452], intercept [0.4752]
///
/// (l1_ratio=0.15 is ignored under penalty='l1' since eff is forced to 1.0;
///  it is passed only to confirm ferrolearn also ignores it.)
#[test]
fn sgd_l1_truncated_gradient() {
    const SK_COEF0: f64 = 0.9204;
    const SK_COEF1: f64 = -0.4452;
    const SK_INTERCEPT: f64 = 0.4752;

    let x = Array2::from_shape_vec((1, 2), vec![2.0, -1.0]).unwrap();
    let y = Array1::from_vec(vec![3.0]);

    let model = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::SquaredError)
        .with_penalty(Penalty::L1)
        .with_l1_ratio(0.15)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.1)
        .with_alpha(0.1)
        .with_max_iter(3)
        .with_tol(-1.0) // disable convergence early-exit (tol=None analog)
        .with_random_state(0);

    let fitted = model.fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    let intercept = fitted.intercept();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-9
            && (coef[1] - SK_COEF1).abs() < 1e-9
            && (intercept - SK_INTERCEPT).abs() < 1e-9,
        "l1 truncated gradient diverges: sklearn coef=[{SK_COEF0}, {SK_COEF1}] \
         intercept={SK_INTERCEPT}; ferrolearn coef={coef:?} intercept={intercept}"
    );
}

// ---------------------------------------------------------------------------
// REQ-5 / #526 — `penalty='elasticnet'` with a non-default `l1_ratio`.
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:632-635,656-658`
//   ElasticNet keeps the user `l1_ratio` (no override at `:558-561`), so BOTH
//   the L2 shrink `max(0, 1 - (1-l1_ratio)*eta*alpha)` (`:635`) AND the L1
//   truncation `u += l1_ratio*eta*alpha; l1penalty(...)` (`:657-658`) fire.
//
// Single sample, squared_error, constant schedule, l1_ratio=0.3 (non-default).
// ---------------------------------------------------------------------------

/// Oracle invocation:
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// X=np.array([[2.0,-1.0]]); y=np.array([3.0]); \
/// m=SGDRegressor(loss='squared_error',penalty='elasticnet',alpha=0.1,l1_ratio=0.3, \
///   learning_rate='constant',eta0=0.1,max_iter=3,tol=None,random_state=0, \
///   fit_intercept=True,shuffle=False).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// ```
/// -> coef [0.9234070529999999, -0.457234953], intercept [0.4712037]
#[test]
fn sgd_elasticnet_l1_ratio() {
    const SK_COEF0: f64 = 0.9234070529999999;
    const SK_COEF1: f64 = -0.457234953;
    const SK_INTERCEPT: f64 = 0.4712037;

    let x = Array2::from_shape_vec((1, 2), vec![2.0, -1.0]).unwrap();
    let y = Array1::from_vec(vec![3.0]);

    let model = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::SquaredError)
        .with_penalty(Penalty::ElasticNet)
        .with_l1_ratio(0.3)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.1)
        .with_alpha(0.1)
        .with_max_iter(3)
        .with_tol(-1.0)
        .with_random_state(0);

    let fitted = model.fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    let intercept = fitted.intercept();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-9
            && (coef[1] - SK_COEF1).abs() < 1e-9
            && (intercept - SK_INTERCEPT).abs() < 1e-9,
        "elasticnet (l1_ratio=0.3) diverges: sklearn coef=[{SK_COEF0}, {SK_COEF1}] \
         intercept={SK_INTERCEPT}; ferrolearn coef={coef:?} intercept={intercept}"
    );
}
