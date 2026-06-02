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
    ClassifierLoss, Hinge, LearningRateSchedule, Loss, Penalty, RegressorLoss, SGDClassifier,
    SGDRegressor,
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

// ---------------------------------------------------------------------------
// REQ-5 / #526 (re-audit, parent #522) — `l1_ratio` is NOT validated to the
// closed interval [0, 1].
//
// sklearn site: `sklearn/linear_model/_stochastic_gradient.py:2018` (regressor)
//   and `:1217` (classifier):
//     `"l1_ratio": [Interval(Real, 0, 1, closed="both")],`
//   This `_parameter_constraints` entry is enforced by `@_fit_context`
//   on EVERY `.fit()` regardless of `penalty`, so `l1_ratio=2.0` (or any value
//   outside [0,1]) raises `InvalidParameterError` even under `penalty='l2'`.
//   (Live oracle: SGDRegressor(penalty='l2', l1_ratio=2.0).fit(...) -> raises
//    InvalidParameterError; same for penalty='elasticnet' and for l1_ratio<0.)
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `validate_reg_params`
//   (`:1564-1599`) / `validate_clf_params` (`:835-870`) check only `eta0` and
//   `alpha` — there is NO `l1_ratio` range check anywhere, so `with_l1_ratio(2.0)`
//   is silently accepted and `fit` returns `Ok` (eff is clamped only implicitly
//   by `effective_l1_ratio`, which for `penalty='l2'` ignores l1_ratio entirely).
//
// Fully deterministic: this is a parameter-validation contract, no RNG / no fit
// trajectory involved — `fit` either errors (sklearn) or returns Ok (ferrolearn).
// ---------------------------------------------------------------------------

/// Oracle invocation:
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// X=np.array([[2.0,-1.0]]); y=np.array([3.0]); \
/// SGDRegressor(penalty='l2',l1_ratio=2.0,alpha=0.1,eta0=0.1,learning_rate='constant', \
///   max_iter=3,tol=None,random_state=0,shuffle=False).fit(X,y)"
/// ```
/// -> raises `sklearn.utils._param_validation.InvalidParameterError`
///    ("The 'l1_ratio' parameter ... must be a float in the range [0, 1]").
/// Also raises for `penalty='elasticnet'` and for `l1_ratio=-0.5` (live oracle).
///
/// ferrolearn `SGDRegressor::with_l1_ratio(2.0).fit(...)` returns `Ok` — no
/// `InvalidParameter` error is produced — so this assertion (expecting an `Err`)
/// FAILS against the current implementation.
#[test]
fn sgd_l1_ratio_out_of_range_rejected() {
    let x = Array2::from_shape_vec((1, 2), vec![2.0, -1.0]).unwrap();
    let y = Array1::from_vec(vec![3.0]);

    // sklearn rejects l1_ratio outside [0, 1] regardless of penalty.
    let model = SGDRegressor::<f64>::new()
        .with_penalty(Penalty::L2)
        .with_l1_ratio(2.0)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.1)
        .with_alpha(0.1)
        .with_max_iter(3)
        .with_tol(-1.0)
        .with_random_state(0);

    let result = model.fit(&x, &y);
    assert!(
        result.is_err(),
        "l1_ratio=2.0 out of [0,1]: sklearn raises InvalidParameterError, \
         ferrolearn accepted it (fit returned Ok)"
    );

    // Negative l1_ratio is likewise rejected by sklearn.
    let model_neg = SGDRegressor::<f64>::new()
        .with_penalty(Penalty::ElasticNet)
        .with_l1_ratio(-0.5)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.1)
        .with_alpha(0.1)
        .with_max_iter(3)
        .with_tol(-1.0)
        .with_random_state(0);

    assert!(
        model_neg.fit(&x, &y).is_err(),
        "l1_ratio=-0.5 out of [0,1]: sklearn raises InvalidParameterError, \
         ferrolearn accepted it (fit returned Ok)"
    );
}

// ---------------------------------------------------------------------------
// REQ-12 / #532 — `shuffle=false` deterministic MULTI-sample, MULTI-epoch
// kernel parity against the live sklearn oracle.
//
// sklearn site: `sklearn/linear_model/_stochastic_gradient.py:107` (default
//   `shuffle=True`), constraint `"shuffle": ["boolean"]` (`:89`); kernel gate
//   `sklearn/linear_model/_sgd_fast.pyx.tp:579-581`:
//     `if shuffle:`
//     `    dataset.shuffle(seed)`
//     `for i in range(n_samples):`
//   so with `shuffle=False` the samples are visited in index order `0..n-1`
//   every epoch (no permutation), and ALL other kernel logic (schedule,
//   penalty, gradient) is unchanged.
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `train_regressor_sgd`
//   `if hyper.shuffle { indices.shuffle(&mut rng); }` — when `shuffle=false`,
//   `indices` stays `0..n-1` each epoch, matching sklearn exactly.
//
// This is the FIRST multi-sample, multi-epoch cross-impl parity test: the RNG
// barrier that forced every other pin to n_samples=1 is removed by `shuffle=
// false`, so the FULL update kernel (L2 shrink + gradient add + elasticnet L1
// truncation, applied across 4 samples × 5 epochs) is validated against the
// oracle. To isolate the kernel from the still-divergent early-stopping logic
// (#530) both sides disable convergence: ferrolearn `with_tol(0.0)`
// (`abs(prev_loss-epoch_loss) < 0.0` is never true, so all max_iter epochs
// run) and sklearn `tol=None` (no early stop). `learning_rate='constant'`
// removes the t/t0 schedule subtleties so any mismatch is purely the update
// rule.
// ---------------------------------------------------------------------------

/// Oracle invocations (live scikit-learn 1.5.2):
/// ```text
/// # penalty='l2'
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.]]); y=np.array([1.,2.,3.,4.]); \
/// m=SGDRegressor(loss='squared_error',penalty='l2',alpha=0.01, \
///   learning_rate='constant',eta0=0.01,max_iter=5,tol=None,shuffle=False, \
///   random_state=0,fit_intercept=True).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// # -> [0.5103165909636498, 0.42319810364130317] [0.16255331549195393]
///
/// # penalty='elasticnet', l1_ratio=0.3
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.]]); y=np.array([1.,2.,3.,4.]); \
/// m=SGDRegressor(loss='squared_error',penalty='elasticnet',l1_ratio=0.3, \
///   alpha=0.01,learning_rate='constant',eta0=0.01,max_iter=5,tol=None, \
///   shuffle=False,random_state=0,fit_intercept=True).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// # -> [0.5102136050112174, 0.4230749783888256] [0.16265294456399926]
/// ```
#[test]
fn sgd_shuffle_false_multisample_kernel_parity() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

    // ----- penalty='l2' (live oracle) -----
    const SK_L2_COEF0: f64 = 0.5103165909636498;
    const SK_L2_COEF1: f64 = 0.42319810364130317;
    const SK_L2_INTERCEPT: f64 = 0.16255331549195393;

    let model_l2 = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::SquaredError)
        .with_penalty(Penalty::L2)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.01)
        .with_alpha(0.01)
        .with_max_iter(5)
        .with_tol(0.0) // abs(prev-cur) < 0.0 is never true -> all 5 epochs run
        .with_shuffle(false) // index order 0..n-1 each epoch, matches sklearn
        .with_random_state(0);

    let fitted_l2 = model_l2.fit(&x, &y).unwrap();
    let coef_l2 = fitted_l2.coefficients();
    let intercept_l2 = fitted_l2.intercept();

    assert!(
        (coef_l2[0] - SK_L2_COEF0).abs() < 1e-7
            && (coef_l2[1] - SK_L2_COEF1).abs() < 1e-7
            && (intercept_l2 - SK_L2_INTERCEPT).abs() < 1e-7,
        "shuffle=false L2 multi-sample kernel diverges: sklearn coef=[{SK_L2_COEF0}, \
         {SK_L2_COEF1}] intercept={SK_L2_INTERCEPT}; ferrolearn coef={coef_l2:?} \
         intercept={intercept_l2}"
    );

    // ----- penalty='elasticnet', l1_ratio=0.3 (live oracle) -----
    const SK_EN_COEF0: f64 = 0.5102136050112174;
    const SK_EN_COEF1: f64 = 0.4230749783888256;
    const SK_EN_INTERCEPT: f64 = 0.16265294456399926;

    let model_en = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::SquaredError)
        .with_penalty(Penalty::ElasticNet)
        .with_l1_ratio(0.3)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.01)
        .with_alpha(0.01)
        .with_max_iter(5)
        .with_tol(0.0)
        .with_shuffle(false)
        .with_random_state(0);

    let fitted_en = model_en.fit(&x, &y).unwrap();
    let coef_en = fitted_en.coefficients();
    let intercept_en = fitted_en.intercept();

    assert!(
        (coef_en[0] - SK_EN_COEF0).abs() < 1e-7
            && (coef_en[1] - SK_EN_COEF1).abs() < 1e-7
            && (intercept_en - SK_EN_INTERCEPT).abs() < 1e-7,
        "shuffle=false elasticnet(l1_ratio=0.3) multi-sample kernel diverges: sklearn \
         coef=[{SK_EN_COEF0}, {SK_EN_COEF1}] intercept={SK_EN_INTERCEPT}; ferrolearn \
         coef={coef_en:?} intercept={intercept_en}"
    );
}

// ---------------------------------------------------------------------------
// REQ-10 / #530 — convergence stop rule: `best_loss` + `sumloss` +
// `n_iter_no_change` consecutive-non-improving epochs (NOT a first-epoch
// mean-loss delta). Now pinnable cross-impl via `shuffle=false`.
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:688-707`
//   `if tol > -INFINITY and sumloss > best_loss - tol * train_count:`
//   `    no_improvement_count += 1`
//   `else:`
//   `    no_improvement_count = 0`
//   `if sumloss < best_loss:`
//   `    best_loss = sumloss`
//   `if no_improvement_count >= n_iter_no_change:`   (default 5)
//   `    ... else: break`
//   `sumloss` is the SUM of per-sample losses over the epoch (`:669`,
//   `sumloss / train_count` is only the *printed* average); `best_loss` tracks
//   the running minimum. So convergence requires `n_iter_no_change` (=5)
//   CONSECUTIVE epochs whose `sumloss` fails to beat `best_loss - tol*n`.
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `train_regressor_sgd:1560-1562`
//   `if (prev_loss - epoch_loss).abs() < hyper.tol { break; }`
//   on the MEAN epoch_loss (`epoch_loss /= n`, `:1557`). This breaks on the
//   FIRST epoch whose mean-loss delta drops below tol — a different criterion
//   that stops at a DIFFERENT epoch, yielding different `coef_`/`intercept_`.
//
// Setup (deterministic on both sides): 4 samples / 2 features, squared_error,
// penalty='l2', `learning_rate='constant'` (eta=eta0 every step, removes the
// t/t0 schedule subtleties so the divergence is PURELY the stop rule),
// alpha=0.01, eta0=0.01, default tol=1e-3, default n_iter_no_change=5,
// `shuffle=False` (index order 0..n-1 each epoch in BOTH impls). max_iter=1000
// (large enough that the stop rule, not the cap, decides when to stop).
//
// Stop-epoch divergence: sklearn runs 49 epochs (`m.n_iter_` below); ferrolearn's
// mean-delta rule stops at epoch 45, so the final weights differ by ~1.6e-2.
// ---------------------------------------------------------------------------

/// Oracle invocation (live scikit-learn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.]]); y=np.array([1.,2.,3.,4.]); \
/// m=SGDRegressor(loss='squared_error',penalty='l2',alpha=0.01, \
///   learning_rate='constant',eta0=0.01,max_iter=1000,tol=1e-3,shuffle=False, \
///   random_state=0,fit_intercept=True).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist(), m.n_iter_)"
/// ```
/// -> coef [0.8037686404055491, 0.16059017315681692]
///    intercept [0.12903834217696583]   n_iter_ 49
///
/// ferrolearn's first-epoch mean-delta stop rule halts at epoch 45 with
/// coef ~[0.78798, 0.17604] intercept ~0.13199 — a ~1.6e-2 weight divergence
/// driven entirely by the different convergence criterion.
#[test]
fn sgd_convergence_n_iter_no_change() {
    const SK_COEF0: f64 = 0.8037686404055491;
    const SK_COEF1: f64 = 0.16059017315681692;
    const SK_INTERCEPT: f64 = 0.12903834217696583;

    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

    // constant schedule => eta=eta0 every step in both impls; tol=1e-3 default,
    // n_iter_no_change=5 default (ferrolearn hardcodes 5). shuffle=false makes
    // the visit order 0..n-1 identical to sklearn.
    let model = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::SquaredError)
        .with_penalty(Penalty::L2)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.01)
        .with_alpha(0.01)
        .with_max_iter(1000)
        .with_tol(1e-3)
        .with_shuffle(false)
        .with_random_state(0);

    let fitted = model.fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    let intercept = fitted.intercept();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-7
            && (coef[1] - SK_COEF1).abs() < 1e-7
            && (intercept - SK_INTERCEPT).abs() < 1e-7,
        "convergence stop rule diverges: sklearn (best_loss/sumloss/n_iter_no_change, \
         49 epochs) coef=[{SK_COEF0}, {SK_COEF1}] intercept={SK_INTERCEPT}; ferrolearn \
         (first-epoch mean-delta, ~45 epochs) coef={coef:?} intercept={intercept}"
    );
}

// ---------------------------------------------------------------------------
// REQ-8 / #528 — adaptive learning-rate schedule: divide `eta` by 5 on the
// SAME `no_improvement_count >= n_iter_no_change` machinery as convergence (and
// only while `eta > 1e-6`), resetting the count. Now pinnable via `shuffle=
// false`.
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:697-707`
//   `if no_improvement_count >= n_iter_no_change:`
//   `    if learning_rate == ADAPTIVE and eta > 1e-6:`
//   `        eta = eta / 5`
//   `        no_improvement_count = 0`
//   `    else: ... break`
//   where `no_improvement_count` is driven by the training-loss branch at
//   `:690-693` (`sumloss > best_loss - tol*train_count`). So in adaptive mode
//   the convergence break is REPLACED by an eta/=5 step; the fit keeps running
//   (here to `n_iter_ = 80`) instead of stopping.
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `train_regressor_sgd:1560-1577`
//   - convergence check `if (prev_loss-epoch_loss).abs() < tol { break; }` fires
//     FIRST and still applies under the adaptive schedule (sklearn removes it);
//   - the adaptive arm divides `current_eta` by 2 (NOT 5) and only when
//     `epoch_loss >= prev_loss` for 5 CONSECUTIVE epochs (a different trigger
//     than sklearn's `sumloss > best_loss - tol*n`).
//   On this monotone-decreasing-loss data the `>= prev_loss` trigger never
//   fires, so ferrolearn never divides eta and instead early-stops at epoch ~45
//   via the (sklearn-absent) convergence rule.
//
// Same data / params as #530 but `learning_rate='adaptive'` (eta0=0.01). Both
// sides deterministic under shuffle=false; the eta trajectory (and stop epoch)
// diverge -> different `coef_`. sklearn 80 epochs vs ferrolearn ~45.
// ---------------------------------------------------------------------------

/// Oracle invocation (live scikit-learn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.]]); y=np.array([1.,2.,3.,4.]); \
/// m=SGDRegressor(loss='squared_error',penalty='l2',alpha=0.01, \
///   learning_rate='adaptive',eta0=0.01,max_iter=1000,tol=1e-3,shuffle=False, \
///   random_state=0,fit_intercept=True).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist(), m.n_iter_)"
/// ```
/// -> coef [0.8065190275590332, 0.15336844797680402]
///    intercept [0.12731338963662575]   n_iter_ 80
///
/// ferrolearn's adaptive arm (÷2 on a never-firing `>=prev_loss` 5-epoch trigger,
/// plus the sklearn-absent mean-delta early stop) halts at epoch ~45 with
/// coef ~[0.78798, 0.17604] intercept ~0.13199 — a ~2e-2 weight divergence from
/// the ÷5 / n_iter_no_change schedule mismatch.
#[test]
fn sgd_adaptive_schedule_divisor() {
    const SK_COEF0: f64 = 0.8065190275590332;
    const SK_COEF1: f64 = 0.15336844797680402;
    const SK_INTERCEPT: f64 = 0.12731338963662575;

    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

    // adaptive schedule, eta0=0.01, tol=1e-3 default, n_iter_no_change=5 default.
    // shuffle=false => visit order 0..n-1 identical to sklearn.
    let model = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::SquaredError)
        .with_penalty(Penalty::L2)
        .with_learning_rate(LearningRateSchedule::Adaptive)
        .with_eta0(0.01)
        .with_alpha(0.01)
        .with_max_iter(1000)
        .with_tol(1e-3)
        .with_shuffle(false)
        .with_random_state(0);

    let fitted = model.fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    let intercept = fitted.intercept();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-7
            && (coef[1] - SK_COEF1).abs() < 1e-7
            && (intercept - SK_INTERCEPT).abs() < 1e-7,
        "adaptive schedule diverges: sklearn (eta/=5 on n_iter_no_change, 80 epochs) \
         coef=[{SK_COEF0}, {SK_COEF1}] intercept={SK_INTERCEPT}; ferrolearn (eta/=2 on \
         >=prev_loss 5-epoch trigger + mean-delta early stop, ~45 epochs) coef={coef:?} \
         intercept={intercept}"
    );
}

// ---------------------------------------------------------------------------
// REQ-11 / #531 — `fit_intercept=False` skips the intercept update.
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:639-644`
//   `if fit_intercept == 1:`
//   `    intercept_update = update`
//   `    if one_class: ...`
//   `    if intercept_update != 0:`
//   `        intercept += intercept_update * intercept_decay`
//   so with `fit_intercept=0` the intercept update is NEVER executed; the
//   intercept stays at its init value `0`. `fit_intercept` defaults to `True`
//   (`_stochastic_gradient.py`), constraint `["boolean"]` at `:86`.
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `train_regressor_sgd` /
//   `train_binary_sgd` — the intercept update is gated
//   `if hyper.fit_intercept { *intercept = *intercept - eta * grad; }`. The
//   intercept enters training as `0` (`b = F::zero()` in the regressor `Fit::fit`
//   / `fit_ova`), so with `fit_intercept=false` it remains EXACTLY `0`.
//
// Multi-sample / multi-epoch, `shuffle=false`, constant schedule, `tol=None`,
// so the full update kernel is deterministic and cross-impl comparable. The
// `coef_` must match sklearn's `fit_intercept=False` `coef_` to ULP precision,
// and BOTH intercepts must be exactly `0.0` (no leakage from the init).
// ---------------------------------------------------------------------------

/// Oracle invocation (live scikit-learn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.]]); y=np.array([1.,2.,3.,4.]); \
/// m=SGDRegressor(loss='squared_error',penalty='l2',alpha=0.01, \
///   learning_rate='constant',eta0=0.01,max_iter=5,tol=None,shuffle=False, \
///   fit_intercept=False,random_state=0).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// ```
/// -> coef [0.5326796739094939, 0.44573604649819804]   intercept [0.0]
///
/// Compare against the `fit_intercept=True` companion
/// (`sgd_shuffle_false_multisample_kernel_parity`, intercept
/// `0.16255331549195393`): turning the flag off zeroes the intercept and shifts
/// the coefficients onto the through-origin fit, exactly as sklearn does.
#[test]
fn sgd_fit_intercept_false() {
    const SK_COEF0: f64 = 0.5326796739094939;
    const SK_COEF1: f64 = 0.44573604649819804;
    const SK_INTERCEPT: f64 = 0.0;

    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

    let model = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::SquaredError)
        .with_penalty(Penalty::L2)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.01)
        .with_alpha(0.01)
        .with_max_iter(5)
        .with_tol(-1.0) // disable convergence early-exit (tol=None analog)
        .with_shuffle(false) // index order 0..n-1 each epoch, matches sklearn
        .with_fit_intercept(false)
        .with_random_state(0);

    let fitted = model.fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    let intercept = fitted.intercept();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-9 && (coef[1] - SK_COEF1).abs() < 1e-9,
        "fit_intercept=false coef diverges: sklearn coef=[{SK_COEF0}, {SK_COEF1}]; \
         ferrolearn coef={coef:?}"
    );
    // The intercept must be EXACTLY 0.0 on both sides (no init leakage).
    assert_eq!(
        intercept, SK_INTERCEPT,
        "fit_intercept=false must leave the intercept at exactly 0.0 (sklearn \
         intercept_ == 0.0); ferrolearn intercept={intercept}"
    );
}

// ---------------------------------------------------------------------------
// REQ-2 / #523 — `squared_hinge` classifier loss
// (`SquaredHinge(threshold=1.0)`).
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:248-258`
//   `cdef double z = self.threshold - p * y`
//   loss:  `z * z if z > 0 else 0.0`
//   dloss: `-2 * y * z if z > 0 else 0.0`
//   registry `sklearn/linear_model/_stochastic_gradient.py:511`
//     `"squared_hinge": (SquaredHinge, 1.0)`.
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `impl Loss for SquaredHinge`,
//   wired through `enum ClassifierLoss::SquaredHinge` in `dispatch_train_binary`.
//
// 2-class, shuffle=false, constant schedule (eta=eta0 identical both impls),
// tol=None (no early stop), so the full multi-sample / multi-epoch update
// kernel is deterministic and cross-impl comparable.
// ---------------------------------------------------------------------------

/// Oracle invocation (live scikit-learn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDClassifier; \
/// X=np.array([[1.,2.],[2.,1.],[8.,9.],[9.,8.]]); y=np.array([0,0,1,1]); \
/// m=SGDClassifier(loss='squared_hinge',penalty='l2',alpha=0.01, \
///   learning_rate='constant',eta0=0.01,max_iter=5,tol=None,shuffle=False, \
///   fit_intercept=True,random_state=0).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// ```
/// -> coef [0.0569485774276016, 0.09335170687740356]
///    intercept [-0.20237316143907]
#[test]
fn sgd_squared_hinge_loss() {
    const SK_COEF0: f64 = 0.0569485774276016;
    const SK_COEF1: f64 = 0.09335170687740356;
    const SK_INTERCEPT: f64 = -0.20237316143907;

    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 8.0, 9.0, 9.0, 8.0]).unwrap();
    let y = Array1::from_vec(vec![0_usize, 0, 1, 1]);

    let model = SGDClassifier::<f64>::new()
        .with_loss(ClassifierLoss::SquaredHinge)
        .with_penalty(Penalty::L2)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.01)
        .with_alpha(0.01)
        .with_max_iter(5)
        .with_tol(-1.0) // disable convergence early-exit (tol=None analog)
        .with_shuffle(false)
        .with_random_state(0);

    let fitted = model.fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    let intercept = fitted.intercept();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-9
            && (coef[1] - SK_COEF1).abs() < 1e-9
            && (intercept - SK_INTERCEPT).abs() < 1e-9,
        "squared_hinge loss diverges: sklearn coef=[{SK_COEF0}, {SK_COEF1}] \
         intercept={SK_INTERCEPT}; ferrolearn coef={coef:?} intercept={intercept}"
    );
}

// ---------------------------------------------------------------------------
// REQ-2 / #523 — `perceptron` classifier loss (`Hinge(threshold=0.0)`).
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:216-226`
//   `cdef double z = p * y`
//   loss:  `self.threshold - z if z <= self.threshold else 0.0`  (= max(0,-z))
//   dloss: `-y if z <= self.threshold else 0.0`
//   registry `sklearn/linear_model/_stochastic_gradient.py:512`
//     `"perceptron": (Hinge, 0.0)`.
// (Note: `SGDClassifier(loss='perceptron')` is the oracle here, NOT the
//  standalone `Perceptron` estimator, which differs in defaults.)
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `impl Loss for Perceptron`,
//   wired through `enum ClassifierLoss::Perceptron` in `dispatch_train_binary`.
// ---------------------------------------------------------------------------

/// Oracle invocation (live scikit-learn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDClassifier; \
/// X=np.array([[1.,2.],[2.,1.],[8.,9.],[9.,8.]]); y=np.array([0,0,1,1]); \
/// m=SGDClassifier(loss='perceptron',penalty='l2',alpha=0.01, \
///   learning_rate='constant',eta0=0.01,max_iter=5,tol=None,shuffle=False, \
///   fit_intercept=True,random_state=0).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// ```
/// -> coef [0.009957048471181063, 0.009961042575429069]
///    intercept [-0.04]
#[test]
fn sgd_perceptron_loss() {
    const SK_COEF0: f64 = 0.009957048471181063;
    const SK_COEF1: f64 = 0.009961042575429069;
    const SK_INTERCEPT: f64 = -0.04;

    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 8.0, 9.0, 9.0, 8.0]).unwrap();
    let y = Array1::from_vec(vec![0_usize, 0, 1, 1]);

    let model = SGDClassifier::<f64>::new()
        .with_loss(ClassifierLoss::Perceptron)
        .with_penalty(Penalty::L2)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.01)
        .with_alpha(0.01)
        .with_max_iter(5)
        .with_tol(-1.0)
        .with_shuffle(false)
        .with_random_state(0);

    let fitted = model.fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    let intercept = fitted.intercept();

    assert!(
        (coef[0] - SK_COEF0).abs() < 1e-9
            && (coef[1] - SK_COEF1).abs() < 1e-9
            && (intercept - SK_INTERCEPT).abs() < 1e-9,
        "perceptron loss diverges: sklearn coef=[{SK_COEF0}, {SK_COEF1}] \
         intercept={SK_INTERCEPT}; ferrolearn coef={coef:?} intercept={intercept}"
    );
}

// ---------------------------------------------------------------------------
// REQ-3 / #524 — `squared_epsilon_insensitive` regressor loss.
//
// sklearn site: `sklearn/linear_model/_sgd_fast.pyx.tp:375-387`
//   loss:  `ret = |y - p| - epsilon; ret*ret if ret > 0 else 0`
//   dloss: `z = y - p;
//           -2*(z-epsilon) if z > epsilon;
//            2*(-z-epsilon) if z < -epsilon;
//            else 0`
//   registry `sklearn/linear_model/_stochastic_gradient.py:1405`
//     `"squared_epsilon_insensitive": (SquaredEpsilonInsensitive, DEFAULT_EPSILON)`
//   with `DEFAULT_EPSILON = 0.1`.
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs`
//   `impl Loss for SquaredEpsilonInsensitive`, wired through
//   `enum RegressorLoss::SquaredEpsilonInsensitive(eps)` in
//   `dispatch_train_regressor`.
//
// Two oracle fits: a single-sample fit (exact per-step eta, no RNG) AND a
// multi-sample / multi-epoch shuffle=false fit (validates the loss over epochs).
// Both: constant schedule, tol=None, fully deterministic.
// ---------------------------------------------------------------------------

/// Oracle invocations (live scikit-learn 1.5.2):
/// ```text
/// # single sample, eta0=0.05
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// X=np.array([[2.0,-1.0]]); y=np.array([3.0]); \
/// m=SGDRegressor(loss='squared_epsilon_insensitive',epsilon=0.1,penalty='l2', \
///   alpha=0.01,learning_rate='constant',eta0=0.05,max_iter=5,tol=None, \
///   shuffle=False,fit_intercept=True,random_state=0).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// # -> [0.9558857922397863, -0.47794289611989316] [0.478752180393125]
///
/// # multi-sample, shuffle=False, eta0=0.01
/// python3 -c "import numpy as np; from sklearn.linear_model import SGDRegressor; \
/// X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.]]); y=np.array([1.,2.,3.,4.]); \
/// m=SGDRegressor(loss='squared_epsilon_insensitive',epsilon=0.1,penalty='l2', \
///   alpha=0.01,learning_rate='constant',eta0=0.01,max_iter=5,tol=None, \
///   shuffle=False,fit_intercept=True,random_state=0).fit(X,y); \
///   print(m.coef_.tolist(), m.intercept_.tolist())"
/// # -> [0.5631419328099845, 0.41545070758814734] [0.16944283314514064]
/// ```
#[test]
fn sgd_squared_epsilon_insensitive_loss() {
    // ----- single sample (live oracle) -----
    const SK_S_COEF0: f64 = 0.9558857922397863;
    const SK_S_COEF1: f64 = -0.47794289611989316;
    const SK_S_INTERCEPT: f64 = 0.478752180393125;

    let xs = Array2::from_shape_vec((1, 2), vec![2.0, -1.0]).unwrap();
    let ys = Array1::from_vec(vec![3.0]);

    let model_s = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::SquaredEpsilonInsensitive(0.1))
        .with_penalty(Penalty::L2)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.05)
        .with_alpha(0.01)
        .with_max_iter(5)
        .with_tol(-1.0)
        .with_shuffle(false)
        .with_random_state(0);

    let fitted_s = model_s.fit(&xs, &ys).unwrap();
    let coef_s = fitted_s.coefficients();
    let intercept_s = fitted_s.intercept();

    assert!(
        (coef_s[0] - SK_S_COEF0).abs() < 1e-9
            && (coef_s[1] - SK_S_COEF1).abs() < 1e-9
            && (intercept_s - SK_S_INTERCEPT).abs() < 1e-9,
        "squared_epsilon_insensitive (single sample) diverges: sklearn coef=[{SK_S_COEF0}, \
         {SK_S_COEF1}] intercept={SK_S_INTERCEPT}; ferrolearn coef={coef_s:?} \
         intercept={intercept_s}"
    );

    // ----- multi-sample, shuffle=false (live oracle) -----
    const SK_M_COEF0: f64 = 0.5631419328099845;
    const SK_M_COEF1: f64 = 0.41545070758814734;
    const SK_M_INTERCEPT: f64 = 0.16944283314514064;

    let xm = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let ym = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

    let model_m = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::SquaredEpsilonInsensitive(0.1))
        .with_penalty(Penalty::L2)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.01)
        .with_alpha(0.01)
        .with_max_iter(5)
        .with_tol(-1.0)
        .with_shuffle(false)
        .with_random_state(0);

    let fitted_m = model_m.fit(&xm, &ym).unwrap();
    let coef_m = fitted_m.coefficients();
    let intercept_m = fitted_m.intercept();

    assert!(
        (coef_m[0] - SK_M_COEF0).abs() < 1e-9
            && (coef_m[1] - SK_M_COEF1).abs() < 1e-9
            && (intercept_m - SK_M_INTERCEPT).abs() < 1e-9,
        "squared_epsilon_insensitive (multi-sample shuffle=false) diverges: sklearn \
         coef=[{SK_M_COEF0}, {SK_M_COEF1}] intercept={SK_M_INTERCEPT}; ferrolearn \
         coef={coef_m:?} intercept={intercept_m}"
    );
}

// ---------------------------------------------------------------------------
// REQ-9b / #544 (parent #522) — loss `epsilon` is NOT validated to [0, inf).
//
// sklearn site: `sklearn/linear_model/_stochastic_gradient.py:2024` (regressor)
//   and `:1219` (BaseSGDRegressor):
//     `"epsilon": [Interval(Real, 0, None, closed="left")],`
//   This `_parameter_constraints` entry is enforced by `@_fit_context` on EVERY
//   `.fit()`, so a negative `epsilon` raises `InvalidParameterError` (boundary 0
//   is valid — closed="left"). Default `epsilon=0.1`.
//
// ferrolearn site: `ferrolearn-linear/src/sgd.rs` `validate_reg_params` — the
//   epsilon lives inside the `RegressorLoss::{Huber,EpsilonInsensitive,
//   SquaredEpsilonInsensitive}(F)` variants; before #544 it was never range-
//   checked, so `with_loss(RegressorLoss::EpsilonInsensitive(-0.5))` was
//   silently accepted and `fit` returned `Ok`.
//
// Fully deterministic: a parameter-validation contract, no RNG / no fit
// trajectory — `fit` either errors (sklearn) or (pre-fix) returns Ok.
// ---------------------------------------------------------------------------

/// Oracle invocation:
/// ```text
/// python3 -c "from sklearn.linear_model import SGDRegressor; import numpy as np; \
///   SGDRegressor(loss='epsilon_insensitive', epsilon=-0.5).fit( \
///     np.array([[1.,2.],[2.,1.]]), np.array([1.,2.]))"
/// ```
/// -> raises `sklearn.utils._param_validation.InvalidParameterError`
///    ("The 'epsilon' parameter ... must be a float in the range [0.0, inf)").
/// Same for `loss='huber'` and `loss='squared_epsilon_insensitive'` with a
/// negative `epsilon`. A NON-negative epsilon (0.0, 0.1) fits OK (live oracle).
#[test]
fn sgd_epsilon_negative_rejected() {
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();
    let y = Array1::from_vec(vec![1.0, 2.0]);

    // sklearn rejects a negative epsilon on every loss that carries one.
    for loss in [
        RegressorLoss::EpsilonInsensitive(-0.5),
        RegressorLoss::Huber(-0.5),
        RegressorLoss::SquaredEpsilonInsensitive(-0.5),
    ] {
        let model = SGDRegressor::<f64>::new()
            .with_loss(loss)
            .with_learning_rate(LearningRateSchedule::Constant)
            .with_eta0(0.1)
            .with_max_iter(3)
            .with_tol(-1.0)
            .with_shuffle(false)
            .with_random_state(0);
        assert!(
            model.fit(&x, &y).is_err(),
            "epsilon=-0.5 ({loss:?}) out of [0, inf): sklearn raises \
             InvalidParameterError, ferrolearn accepted it (fit returned Ok)"
        );
    }

    // Boundary 0.0 is valid (closed="left") and 0.1 (the default) fits OK.
    for eps in [0.0_f64, 0.1] {
        let model = SGDRegressor::<f64>::new()
            .with_loss(RegressorLoss::EpsilonInsensitive(eps))
            .with_learning_rate(LearningRateSchedule::Constant)
            .with_eta0(0.1)
            .with_max_iter(3)
            .with_tol(-1.0)
            .with_shuffle(false)
            .with_random_state(0);
        assert!(
            model.fit(&x, &y).is_ok(),
            "epsilon={eps} is in [0, inf) (closed-left boundary 0 valid); \
             sklearn fits OK but ferrolearn rejected it"
        );
    }
}
