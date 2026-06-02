//! Stochastic Gradient Descent (SGD) linear models.
//!
//! This module provides [`SGDClassifier`] and [`SGDRegressor`], two linear
//! models trained using stochastic gradient descent. Both support online /
//! streaming learning via the [`PartialFit`] trait and a range of configurable
//! loss functions and learning-rate schedules.
//!
//! # Classifier
//!
//! ```
//! use ferrolearn_linear::sgd::{SGDClassifier, ClassifierLoss};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 2.0, 3.0, 3.0, 1.0,
//!     8.0, 7.0, 9.0, 8.0, 7.0, 9.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let model = SGDClassifier::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! # Regressor
//!
//! ```
//! use ferrolearn_linear::sgd::{SGDRegressor, RegressorLoss};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0];
//!
//! let model = SGDRegressor::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 4);
//! ```
//!
//! ## REQ status (per `.design/linear/sgd.md`, mirrors `sklearn/linear_model/_stochastic_gradient.py` + `_sgd_fast.pyx.tp` @ 1.5.2, commit 156ef14)
//!
//! Parity is framed on the deterministic schedule/loss/penalty math + defaults;
//! random-shuffle full-fit weight parity is out of scope (cross-PRNG boundary,
//! `_sgd_fast.pyx.tp:579-580` vs `StdRng`). Two states only per R-DEFER-2.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (classifier losses hinge/log/modified_huber/squared_error incl. Hinge boundary) | SHIPPED | `impl Loss for Hinge/LogLoss/ModifiedHuber/SquaredError`. Hinge `gradient` now uses the NON-strict boundary `margin <= 1` matching `_sgd_fast.pyx.tp:224` (`if z <= threshold: return -y`). Consumer: `fn dispatch_train_binary` -> `Fit for SGDClassifier` -> `impl PipelineEstimator for SGDClassifier`. Tests: `test_hinge_loss_*`, divergence `sgd_hinge_gradient_boundary`. Closed #539. |
//! | REQ-2 (squared_hinge, perceptron) | SHIPPED | `pub struct SquaredHinge` (`loss = (1-py)^2 if >0 else 0`, `gradient = -2y(1-py)`, `_sgd_fast.pyx.tp:248-258` with `threshold=1.0`, `_stochastic_gradient.py:511`) + `pub struct Perceptron` (`Hinge(threshold=0.0)`: `loss = max(0,-py)`, `gradient = -y if py<=0 else 0`, `_sgd_fast.pyx.tp:216-226`, `_stochastic_gradient.py:512`); `enum ClassifierLoss::{SquaredHinge,Perceptron}` wired in `fn dispatch_train_binary`. Consumer: `fn dispatch_train_binary` -> `fn fit_ova` -> `Fit for SGDClassifier` -> `impl PipelineEstimator for SGDClassifier`. Tests: divergence `sgd_squared_hinge_loss` (live oracle coef `[0.0569485774276016, 0.09335170687740356]` intercept `-0.20237316143907`), `sgd_perceptron_loss` (live oracle coef `[0.009957048471181063, 0.009961042575429069]` intercept `-0.04`). Closes #523. |
//! | REQ-3 (regressor losses incl. squared_epsilon_insensitive) | SHIPPED | `pub struct SquaredEpsilonInsensitive<F> { epsilon }` (`loss = max(0,|y-p|-eps)^2`, `gradient = -2(z-eps) if z>eps; 2(-z-eps) if z<-eps; else 0` for `z=y-p`, `_sgd_fast.pyx.tp:375-387`, `_stochastic_gradient.py:1405` default `epsilon=0.1`) + `enum RegressorLoss::SquaredEpsilonInsensitive(F)` wired in `fn dispatch_train_regressor`. Consumer: `fn dispatch_train_regressor` -> `Fit for SGDRegressor` -> `impl PipelineEstimator for SGDRegressor`. Test: divergence `sgd_squared_epsilon_insensitive_loss` (live oracle single-sample coef `[0.9558857922397863, -0.47794289611989316]` intercept `0.478752180393125`, multi-sample shuffle=false coef `[0.5631419328099845, 0.41545070758814734]` intercept `0.16944283314514064`). Closes #524. |
//! | REQ-4 (L2 penalty = clamped wscale shrink) | SHIPPED | `fn train_binary_sgd`/`train_regressor_sgd` apply `shrink = max(0, 1 - eta*alpha)` then `w = w*shrink - eta*grad*x`, mirroring `w.scale(max(0, 1-eta*alpha))` (`_sgd_fast.pyx.tp:632-635`); intercept unregularized. Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `PipelineEstimator`. Test: divergence `sgd_l2_wscale_clamp`. Closed #525. |
//! | REQ-5 (l1/elasticnet + l1_ratio) | SHIPPED | `enum Penalty {L2,L1,ElasticNet}` + `pub penalty`/`pub l1_ratio` fields on `SGDClassifier`/`SGDRegressor` with `fn with_penalty`/`fn with_l1_ratio` builders (defaults `L2`/`0.15`, `_stochastic_gradient.py:1231-1256`). `fn train_binary_sgd`/`train_regressor_sgd` derive `eff` via `fn effective_l1_ratio` (`L2->0`, `L1->1`, `ElasticNet->l1_ratio`, `_sgd_fast.pyx.tp:558-561`), apply the L2 shrink `max(0, 1-(1-eff)*eta*alpha)` BEFORE the gradient add (`:632-635`), then the Tsuruoka cumulative-penalty L1 truncation with fit-persistent scalar `u` and per-feature `q` AFTER (`:656-658,750-778`, `wscale=1`). Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `PipelineEstimator`. Tests: divergence `sgd_l1_truncated_gradient` (live oracle coef [0.9204,-0.4452]), `sgd_elasticnet_l1_ratio` (l1_ratio=0.3, coef [0.92340705,-0.45723495]). Closed #526. NOTE (partial_fit+l1): `u`/`q` are scoped per `train_*_sgd` call, so they persist across the epochs of a single `fit` (the parity-critical path) but reset per `partial_fit` call. This MATCHES sklearn, which re-allocates `q=np.zeros(...)`/`u=0.0` at the top of every `_plain_sgd` call (`_sgd_fast.pyx.tp:551-556`) and only carries `t_` across `partial_fit` (`_stochastic_gradient.py` re-invokes `_plain_sgd` per call). The full `fit` path is exact. |
//! | REQ-6 (constant + invscaling schedules) | SHIPPED | `fn compute_lr`: `Constant => eta0`, `InvScaling => eta0 / t^power_t` (`_sgd_fast.pyx.tp:479,593-594`). Consumer: per-step in `fn train_binary_sgd`/`train_regressor_sgd`. Tests: `test_constant_lr`, `test_invscaling_lr`. |
//! | REQ-7 (optimal schedule t0 offset) | SHIPPED | `fn compute_lr` Optimal arm now `1/(alpha*(optimal_init + t - 1))` with `optimal_init` from `fn optimal_init` (`typw=sqrt(1/sqrt(alpha))`, `e0=typw/max(1,|gradient(1,-typw)|)`, `optimal_init=1/(e0*alpha)`), mirroring `_sgd_fast.pyx.tp:565-570,592`. Computed once per fit before the epoch loop. Consumer: `fn train_*_sgd`. Tests: `test_optimal_lr`, `test_optimal_init_matches_oracle`, divergence `sgd_optimal_schedule_t0_offset`. Closed #527. |
//! | REQ-8 (adaptive /5 + n_iter_no_change trigger) | SHIPPED | `fn convergence_tail` (shared by `fn train_binary_sgd`/`train_regressor_sgd`) divides `current_eta` by 5 (not 2) when `no_improve_count >= n_iter_no_change` AND the schedule is `Adaptive` AND `eta > 1e-6`, resetting the count — the SAME `best_loss`/`sumloss > best_loss - tol*n` machinery as convergence (`_sgd_fast.pyx.tp:697-707`). The old `>= prev_loss` 5-epoch `/2` trigger is deleted. Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `PipelineEstimator`. Test: divergence `sgd_adaptive_schedule_divisor` (live oracle coef `[0.8065190275590332, 0.15336844797680402]` intercept `0.12731338963662575`, n_iter_ 80). Closes #528. |
//! | REQ-9 (default params per estimator) | SHIPPED (classifier defaults) | `SGDClassifier::new` now sets `learning_rate=Optimal, eta0=0.0, power_t=0.5` (`_stochastic_gradient.py:1242-1244`); `fn schedule_requires_eta0` gates the `eta0>0` validation to constant/invscaling/adaptive (`_stochastic_gradient.py:149-153`). Consumer: `Fit for SGDClassifier`. Tests: divergence `sgd_classifier_default_learning_rate`, `test_sgd_classifier_default`, `test_sgd_classifier_optimal_eta0_zero_ok`. Closed #529. Remaining missing fields (`fit_intercept`, `epsilon`, `early_stopping`, `validation_fraction`, `average`, `warm_start`, `class_weight`, `C`) tracked under their own blockers. (`penalty`/`l1_ratio` shipped under REQ-5; `shuffle` under REQ-12; `n_iter_no_change` folded into REQ-10.) |
//! | REQ-10 (convergence best_loss/n_iter_no_change/sumloss) | SHIPPED | `fn convergence_tail` (shared by `fn train_binary_sgd`/`train_regressor_sgd`) tracks `best_loss` (running min, init `+inf`) and increments `no_improve_count` when `tol_active && sumloss > best_loss - tol*n_samples`, resetting otherwise; breaks once `no_improve_count >= hyper.n_iter_no_change` (non-adaptive), exactly mirroring `_sgd_fast.pyx.tp:688-707`. `sumloss` is now the SUM of per-sample losses over the epoch (the `/= n_samples` mean division is removed, `_sgd_fast.pyx.tp:597`); `tol_active = hyper.tol > -inf` encodes sklearn's `tol=None -> -INFINITY` disable (`:690`). The per-sample gradient is clipped to `[-1e12, 1e12]` via `fn max_dloss` before the update (`_sgd_fast.pyx.tp:546,613-620`). `n_iter_no_change` is now a settable `pub` field (default 5) with `fn with_n_iter_no_change` on both estimators, threaded through `SGDHyper`/`clf_hyper`/`reg_hyper` (`_stochastic_gradient.py` `n_iter_no_change=5`). Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `PipelineEstimator`. Test: divergence `sgd_convergence_n_iter_no_change` (live oracle coef `[0.8037686404055491, 0.16059017315681692]` intercept `0.12903834217696583`, n_iter_ 49). Closes #530. |
//! | REQ-11 (fit_intercept) | SHIPPED | `pub fit_intercept: bool` field on `SGDClassifier`/`SGDRegressor` + `fn with_fit_intercept` builders (default `true`, sklearn `_stochastic_gradient.py` `fit_intercept=True`, constraint `["boolean"]` at `:86`), threaded through `SGDHyper.fit_intercept` + `fn clf_hyper`/`reg_hyper`. `fn train_binary_sgd`/`train_regressor_sgd` gate the intercept update: `if hyper.fit_intercept { *intercept = *intercept - eta * grad; }`, mirroring `if fit_intercept == 1: intercept_update = update; ... intercept += intercept_update * intercept_decay` (`_sgd_fast.pyx.tp:639-644`, `intercept_decay=1` on the standard path). When `false` the intercept is never modified and stays at its init value `0` (`b = F::zero()` before training in `fn fit_ova`/regressor `Fit::fit`), so `coef_` matches sklearn and `intercept_` is exactly `0`. Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `PipelineEstimator`. Test: divergence `sgd_fit_intercept_false` (live oracle coef `[0.5326796739094939, 0.44573604649819804]`, intercept exactly `0.0`). Closes #531. |
//! | REQ-12 (shuffle flag) | SHIPPED | `pub shuffle: bool` field on `SGDClassifier`/`SGDRegressor` + `fn with_shuffle` builders (default `true`, `_stochastic_gradient.py:107` `shuffle=True`, constraint `["boolean"]` at `:89`), threaded through `SGDHyper.shuffle` + `fn clf_hyper`/`reg_hyper`. `fn train_binary_sgd`/`train_regressor_sgd` gate the per-epoch shuffle: `if hyper.shuffle { indices.shuffle(&mut rng); }`, mirroring `if shuffle: dataset.shuffle(seed)` (`_sgd_fast.pyx.tp:579-580`); when off, `indices` stays `0..n-1` each epoch matching sklearn's no-shuffle index order (`:581` `for i in range(n_samples)`). Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `PipelineEstimator`. Tests: divergence `sgd_shuffle_false_multisample_kernel_parity` (4-sample/2-feature/5-epoch L2 oracle coef `[0.5103165909636498, 0.42319810364130317]` intercept `0.16255331549195393`; elasticnet l1_ratio=0.3 oracle coef `[0.5102136050112174, 0.4230749783888256]` intercept `0.16265294456399926`). Closes #532. This `shuffle=false` parity ALSO validates REQ-4/REQ-5/REQ-6 (L2 shrink + elasticnet truncated gradient + constant schedule) over MULTIPLE samples and epochs against the live oracle — previously only single-sample. |
//! | REQ-13 (early_stopping + validation_fraction) | NOT-STARTED | blocker #533. No validation split / score callback. |
//! | REQ-14 (average / ASGD) | NOT-STARTED | blocker #534. No averaged coef/intercept (`_sgd_fast.pyx.tp:646-654`). |
//! | REQ-15 (class_weight) | NOT-STARTED | blocker #535. No per-class `weight_pos`/`weight_neg`/`sample_weight` (`_sgd_fast.pyx.tp:599-602,630`). |
//! | REQ-16 (partial_fit semantics) | SHIPPED | `fn partial_fit (PartialFit for SGDClassifier/FittedSGDClassifier/SGDRegressor/FittedSGDRegressor)` sets `max_iter=1` and carries `self.t` (`_stochastic_gradient.py:581-674`). Consumer: `PartialFit` trait (`ferrolearn-core`). Tests: `test_sgd_*_partial_fit*`. |
//! | REQ-17 (multiclass one-vs-all) | SHIPPED | `fn fit_ova` (one binary per class) + `fn predict` argmax (`_stochastic_gradient.py:788-844`). Consumer: `Fit for SGDClassifier` -> `PipelineEstimator`. Test: `test_sgd_classifier_multiclass`. |
//! | REQ-18 (SGDOneClassSVM) | NOT-STARTED | blocker #536 (builder). Estimator absent (`_stochastic_gradient.py:2084-2278`). |
//! | REQ-19 (anti-pattern cleanup) | NOT-STARTED | blocker #537. `fn compute_lr`'s `_Phantom` arm now returns `eta0` (unreach macro removed); the loss/lr kernels still use `F::from(..).unwrap()` constants (R-CODE-2) tracked here. |
//! | REQ-20 (ferray substrate migration) | NOT-STARTED | blocker #538. Still `ndarray` + `StdRng` (R-SUBSTRATE-1). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, PartialFit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::SeedableRng;
use rand::seq::SliceRandom;

// ---------------------------------------------------------------------------
// Loss functions
// ---------------------------------------------------------------------------

/// A loss function for SGD optimization.
///
/// Provides the loss value and its gradient with respect to the prediction.
pub trait Loss<F: Float>: Clone + Send + Sync {
    /// Compute the loss for a single sample.
    fn loss(&self, y_true: F, y_pred: F) -> F;

    /// Compute the gradient of the loss with respect to `y_pred`.
    fn gradient(&self, y_true: F, y_pred: F) -> F;
}

/// Hinge loss for linear SVM-style classification.
///
/// `L(y, p) = max(0, 1 - y * p)` where `y in {-1, +1}`.
#[derive(Debug, Clone, Copy)]
pub struct Hinge;

impl<F: Float> Loss<F> for Hinge {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let margin = y_true * y_pred;
        if margin < F::one() {
            F::one() - margin
        } else {
            F::zero()
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        // sklearn `Hinge.dloss` uses a NON-strict boundary at the threshold
        // (`_sgd_fast.pyx.tp:224`: `if z <= self.threshold: return -y`), so at
        // the exact margin `z == 1` the gradient is `-y`, not `0`.
        let margin = y_true * y_pred;
        if margin <= F::one() {
            -y_true
        } else {
            F::zero()
        }
    }
}

/// Squared hinge loss for (quadratically penalized) linear SVM classification.
///
/// `L(y, p) = max(0, 1 - y * p)^2` where `y in {-1, +1}`. This is sklearn's
/// `SquaredHinge(threshold=1.0)` (`_sgd_fast.pyx.tp:232-258`); the
/// `squared_hinge` classifier loss maps to it (`_stochastic_gradient.py:511`).
#[derive(Debug, Clone, Copy)]
pub struct SquaredHinge;

impl<F: Float> Loss<F> for SquaredHinge {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        // `_sgd_fast.pyx.tp:248-252`: `z = threshold - p*y; z*z if z > 0 else 0`
        // with `threshold = 1.0` (`_stochastic_gradient.py:511`).
        let z = F::one() - y_pred * y_true;
        if z > F::zero() { z * z } else { F::zero() }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        // `_sgd_fast.pyx.tp:254-258`: `z = threshold - p*y; -2*y*z if z > 0 else 0`.
        let z = F::one() - y_pred * y_true;
        if z > F::zero() {
            -F::from(2.0).unwrap_or_else(|| F::one() + F::one()) * y_true * z
        } else {
            F::zero()
        }
    }
}

/// Perceptron loss for linear classification.
///
/// `L(y, p) = max(0, -y * p)` where `y in {-1, +1}`. This is sklearn's
/// `Hinge(threshold=0.0)` (`_sgd_fast.pyx.tp:200-226`); the `perceptron`
/// classifier loss maps to it (`_stochastic_gradient.py:512`). The existing
/// [`Hinge`] hardcodes `threshold = 1.0`, so this is a separate type.
#[derive(Debug, Clone, Copy)]
pub struct Perceptron;

impl<F: Float> Loss<F> for Perceptron {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        // `_sgd_fast.pyx.tp:216-220`: `z = p*y; threshold - z if z <= threshold
        // else 0` with `threshold = 0.0` (`_stochastic_gradient.py:512`), i.e.
        // `max(0, -z)`.
        let z = y_pred * y_true;
        if z <= F::zero() { -z } else { F::zero() }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        // `_sgd_fast.pyx.tp:222-226`: `z = p*y; -y if z <= threshold else 0`
        // with `threshold = 0.0`.
        let z = y_pred * y_true;
        if z <= F::zero() { -y_true } else { F::zero() }
    }
}

/// Log loss (logistic regression / cross-entropy).
///
/// `L(y, p) = log(1 + exp(-y * p))` where `y in {-1, +1}`.
#[derive(Debug, Clone, Copy)]
pub struct LogLoss;

impl<F: Float> Loss<F> for LogLoss {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let z = y_true * y_pred;
        if z > F::from(18.0).unwrap() {
            (-z).exp()
        } else if z < F::from(-18.0).unwrap() {
            -z
        } else {
            (F::one() + (-z).exp()).ln()
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let z = y_true * y_pred;
        let exp_nz = if z > F::from(18.0).unwrap() {
            (-z).exp()
        } else if z < F::from(-18.0).unwrap() {
            F::from(1e18).unwrap()
        } else {
            (-z).exp()
        };
        -y_true * exp_nz / (F::one() + exp_nz)
    }
}

/// Squared error loss for regression.
///
/// `L(y, p) = 0.5 * (y - p)^2`.
#[derive(Debug, Clone, Copy)]
pub struct SquaredError;

impl<F: Float> Loss<F> for SquaredError {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let diff = y_true - y_pred;
        F::from(0.5).unwrap() * diff * diff
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        y_pred - y_true
    }
}

/// Modified Huber loss for classification.
///
/// Smooth approximation to hinge with quadratic behaviour near the margin:
///
/// ```text
/// L(y, p) = max(0, 1 - y*p)^2   if y*p >= -1
///         = -4 * y * p            otherwise
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ModifiedHuber;

impl<F: Float> Loss<F> for ModifiedHuber {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let z = y_true * y_pred;
        if z >= -F::one() {
            let margin = F::one() - z;
            if margin > F::zero() {
                margin * margin
            } else {
                F::zero()
            }
        } else {
            -F::from(4.0).unwrap() * z
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let z = y_true * y_pred;
        if z >= -F::one() {
            if z < F::one() {
                F::from(-2.0).unwrap() * y_true * (F::one() - z)
            } else {
                F::zero()
            }
        } else {
            -F::from(4.0).unwrap() * y_true
        }
    }
}

/// Huber loss for robust regression.
///
/// `L(y, p) = 0.5 * (y - p)^2` if `|y - p| <= epsilon`, else
/// `epsilon * (|y - p| - 0.5 * epsilon)`.
#[derive(Debug, Clone, Copy)]
pub struct Huber<F> {
    /// Threshold parameter for switching from quadratic to linear loss.
    pub epsilon: F,
}

impl<F: Float + Send + Sync> Loss<F> for Huber<F> {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let diff = y_true - y_pred;
        let abs_diff = diff.abs();
        if abs_diff <= self.epsilon {
            F::from(0.5).unwrap() * diff * diff
        } else {
            self.epsilon * (abs_diff - F::from(0.5).unwrap() * self.epsilon)
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let diff = y_pred - y_true;
        let abs_diff = diff.abs();
        if abs_diff <= self.epsilon {
            diff
        } else if diff > F::zero() {
            self.epsilon
        } else {
            -self.epsilon
        }
    }
}

/// Epsilon-insensitive loss for support vector regression.
///
/// `L(y, p) = max(0, |y - p| - epsilon)`.
#[derive(Debug, Clone, Copy)]
pub struct EpsilonInsensitive<F> {
    /// Insensitivity margin.
    pub epsilon: F,
}

impl<F: Float + Send + Sync> Loss<F> for EpsilonInsensitive<F> {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let diff = (y_true - y_pred).abs();
        if diff > self.epsilon {
            diff - self.epsilon
        } else {
            F::zero()
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let diff = y_pred - y_true;
        if diff > self.epsilon {
            F::one()
        } else if diff < -self.epsilon {
            -F::one()
        } else {
            F::zero()
        }
    }
}

/// Squared epsilon-insensitive loss for support vector regression.
///
/// `L(y, p) = max(0, |y - p| - epsilon)^2`. This is sklearn's
/// `SquaredEpsilonInsensitive` (`_sgd_fast.pyx.tp:364-388`); the
/// `squared_epsilon_insensitive` regressor loss maps to it
/// (`_stochastic_gradient.py:1405`, default `epsilon = DEFAULT_EPSILON = 0.1`).
#[derive(Debug, Clone, Copy)]
pub struct SquaredEpsilonInsensitive<F> {
    /// Insensitivity margin.
    pub epsilon: F,
}

impl<F: Float + Send + Sync> Loss<F> for SquaredEpsilonInsensitive<F> {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        // `_sgd_fast.pyx.tp:375-377`: `ret = |y - p| - epsilon;
        // ret*ret if ret > 0 else 0`.
        let ret = (y_true - y_pred).abs() - self.epsilon;
        if ret > F::zero() {
            ret * ret
        } else {
            F::zero()
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        // `_sgd_fast.pyx.tp:379-387`: `z = y - p;
        // -2*(z-epsilon) if z > epsilon; 2*(-z-epsilon) if z < -epsilon; else 0`.
        let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());
        let z = y_true - y_pred;
        if z > self.epsilon {
            -two * (z - self.epsilon)
        } else if z < -self.epsilon {
            two * (-z - self.epsilon)
        } else {
            F::zero()
        }
    }
}

// ---------------------------------------------------------------------------
// Learning rate schedules
// ---------------------------------------------------------------------------

/// Learning rate schedule for SGD.
#[derive(Debug, Clone, Copy)]
pub enum LearningRateSchedule<F> {
    /// Fixed learning rate `eta0` throughout training.
    Constant,
    /// Optimal schedule: `eta = 1 / (alpha * t)`.
    Optimal,
    /// Inverse scaling: `eta = eta0 / t^power_t`.
    InvScaling,
    /// Adaptive: starts at `eta0`, halved when loss fails to decrease for
    /// 5 consecutive epochs. Stops when `eta < 1e-6`.
    Adaptive,
    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<F>),
}

/// Compute the learning rate for a given step.
///
/// `optimal_init` is the `t0` offset of the `optimal` schedule, derived once
/// per fit from `alpha` and the loss's `dloss(1, -typw)` bound (see
/// [`optimal_init`]). It is ignored by the other schedules.
fn compute_lr<F: Float>(
    schedule: &LearningRateSchedule<F>,
    eta0: F,
    alpha: F,
    power_t: F,
    optimal_init: F,
    t: usize,
) -> F {
    let t_f = F::from(t.max(1)).unwrap_or_else(F::one);
    match schedule {
        LearningRateSchedule::Constant => eta0,
        // sklearn `_sgd_fast.pyx.tp:592`: `eta = 1/(alpha*(optimal_init+t-1))`,
        // so the first sample (t=1) sees `eta = 1/(alpha*optimal_init) = e0`.
        LearningRateSchedule::Optimal => F::one() / (alpha * (optimal_init + t_f - F::one())),
        LearningRateSchedule::InvScaling => eta0 / t_f.powf(power_t),
        LearningRateSchedule::Adaptive => eta0,
        // `_Phantom` is an uninhabited marker arm; fall back to `eta0` rather
        // than aborting (R-APG-1 forbids the unreach macro in production).
        LearningRateSchedule::_Phantom(_) => eta0,
    }
}

/// Compute the `optimal` schedule's `t0` offset `optimal_init`.
///
/// Mirrors `_sgd_fast.pyx.tp:565-570`:
/// `typw = sqrt(1/sqrt(alpha))`,
/// `initial_eta0 = typw / max(1, dloss(1, -typw))`,
/// `optimal_init = 1/(initial_eta0 * alpha)`.
///
/// sklearn calls `loss.dloss(1.0, -typw)` where the cython signature is
/// `dloss(self, y, p)` (y first, p second), so `y = 1.0`, `p = -typw`. The
/// ferrolearn signature is `gradient(y_true, y_pred)`, mapping to
/// `gradient(1.0, -typw)`; its absolute value matches `max(1.0, dloss(...))`.
/// Returns `1.0` when `alpha == 0` (the schedule is unused / guarded upstream).
fn optimal_init<F, L>(loss_fn: &L, alpha: F) -> F
where
    F: Float,
    L: Loss<F>,
{
    if alpha <= F::zero() {
        return F::one();
    }
    let typw = (F::one() / alpha.sqrt()).sqrt();
    let dloss = loss_fn.gradient(F::one(), -typw).abs();
    let initial_eta0 = typw / dloss.max(F::one());
    F::one() / (initial_eta0 * alpha)
}

// ---------------------------------------------------------------------------
// Penalty (regularization term)
// ---------------------------------------------------------------------------

/// Regularization penalty for SGD.
///
/// Mirrors sklearn's `penalty` parameter
/// (`_stochastic_gradient.py:997-1012`). The effective `l1_ratio` passed to the
/// kernel is derived per `_sgd_fast.pyx.tp:558-561`: `l2 -> 0.0`, `l1 -> 1.0`,
/// `elasticnet -> user l1_ratio`.
#[derive(Debug, Clone, Copy)]
pub enum Penalty {
    /// L2 (ridge) penalty — multiplicative `wscale` shrink only (the default).
    L2,
    /// L1 (lasso) penalty — Tsuruoka cumulative-penalty truncated gradient.
    L1,
    /// Elastic-net — convex mix of L2 and L1 controlled by `l1_ratio`.
    ElasticNet,
}

/// Compute the effective `l1_ratio` for the truncated-gradient kernel.
///
/// Mirrors `_sgd_fast.pyx.tp:558-561`: `L2 -> 0.0`, `L1 -> 1.0`,
/// `ElasticNet -> user l1_ratio`.
fn effective_l1_ratio<F: Float>(penalty: Penalty, l1_ratio: F) -> F {
    match penalty {
        Penalty::L2 => F::zero(),
        Penalty::L1 => F::one(),
        Penalty::ElasticNet => l1_ratio,
    }
}

/// The `MAX_DLOSS` gradient clip bound (`_sgd_fast.pyx.tp:546`, `1e12`).
///
/// sklearn clips `dloss` to `[-MAX_DLOSS, MAX_DLOSS]` before forming the update
/// `update = -eta * dloss` (`_sgd_fast.pyx.tp:613-620`) to avoid numerical
/// instabilities. Falls back to `F::max_value()` (an even looser, never-active
/// bound) if `1e12` is not representable in `F` — so the clamp is always a safe
/// no-op widening rather than a panic.
#[inline]
fn max_dloss<F: Float>() -> F {
    F::from(1e12_f64).unwrap_or_else(F::max_value)
}

/// Shared SGD epoch-end convergence / adaptive-eta tail.
///
/// Mirrors `_sgd_fast.pyx.tp:688-707` exactly. `epoch_sumloss` is the SUM of
/// per-sample losses over the epoch (`:597`, NOT the printed mean `sumloss /
/// train_count`). The criterion increments `no_improve_count` whenever
/// `sumloss > best_loss - tol * train_count` (an epoch that fails to beat the
/// running minimum by at least `tol*n`) and resets it otherwise; `best_loss`
/// tracks the running minimum. Once `no_improve_count >= n_iter_no_change`,
/// under the adaptive schedule (`eta > 1e-6`) `eta` is divided by 5 and the
/// count reset (`:699-701`); otherwise the caller breaks (convergence, `:702`).
///
/// Returns `true` iff the epoch loop should `break` (convergence). The two
/// branches are mutually exclusive, exactly as upstream: adaptive decays eta
/// and keeps running, non-adaptive (or eta already `<= 1e-6`) stops.
#[allow(clippy::too_many_arguments, reason = "mirrors the upstream epoch tail")]
#[inline]
fn convergence_tail<F: Float>(
    epoch_sumloss: F,
    best_loss: &mut F,
    no_improve_count: &mut usize,
    current_eta: &mut F,
    tol_active: bool,
    tol: F,
    n_samples: usize,
    n_iter_no_change: usize,
    adaptive: bool,
) -> bool {
    // `_sgd_fast.pyx.tp:690-693`: training-loss branch (early_stopping=False).
    let n = F::from(n_samples).unwrap_or_else(F::zero);
    if tol_active && epoch_sumloss > *best_loss - tol * n {
        *no_improve_count += 1;
    } else {
        *no_improve_count = 0;
    }
    // `:694-695`: track the running minimum.
    if epoch_sumloss < *best_loss {
        *best_loss = epoch_sumloss;
    }
    // `:698-707`: convergence break OR adaptive eta/=5.
    if *no_improve_count >= n_iter_no_change {
        // `:699`: `if learning_rate == ADAPTIVE and eta > 1e-6`.
        let eta_floor = F::from(1e-6_f64).unwrap_or_else(F::zero);
        let divisor = F::from(5.0_f64).unwrap_or_else(F::one);
        if adaptive && *current_eta > eta_floor {
            *current_eta = *current_eta / divisor;
            *no_improve_count = 0;
            false
        } else {
            true
        }
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Classifier loss enum
// ---------------------------------------------------------------------------

/// Available loss functions for [`SGDClassifier`].
#[derive(Debug, Clone, Copy)]
pub enum ClassifierLoss {
    /// Hinge loss (linear SVM).
    Hinge,
    /// Squared hinge loss (quadratically penalized SVM,
    /// `_stochastic_gradient.py:511`).
    SquaredHinge,
    /// Perceptron loss (`Hinge(threshold=0.0)`,
    /// `_stochastic_gradient.py:512`).
    Perceptron,
    /// Log loss (logistic regression).
    Log,
    /// Squared error loss.
    SquaredError,
    /// Modified Huber loss.
    ModifiedHuber,
}

/// Available loss functions for [`SGDRegressor`].
#[derive(Debug, Clone, Copy)]
pub enum RegressorLoss<F> {
    /// Squared error loss (default).
    SquaredError,
    /// Huber loss with the given epsilon.
    Huber(F),
    /// Epsilon-insensitive loss with the given epsilon.
    EpsilonInsensitive(F),
    /// Squared epsilon-insensitive loss with the given epsilon
    /// (`_stochastic_gradient.py:1405`, `_sgd_fast.pyx.tp:364-388`).
    SquaredEpsilonInsensitive(F),
}

// ---------------------------------------------------------------------------
// SGDClassifier
// ---------------------------------------------------------------------------

/// Stochastic Gradient Descent classifier.
///
/// Supports binary classification via a decision boundary and multiclass
/// classification via one-vs-all decomposition.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_linear::sgd::SGDClassifier;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0, 2.0, 3.0, 3.0, 1.0,
///     8.0, 7.0, 9.0, 8.0, 7.0, 9.0,
/// ]).unwrap();
/// let y = array![0, 0, 0, 1, 1, 1];
///
/// let clf = SGDClassifier::<f64>::new();
/// let fitted = clf.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SGDClassifier<F> {
    /// The loss function to use.
    pub loss: ClassifierLoss,
    /// The learning rate schedule.
    pub learning_rate: LearningRateSchedule<F>,
    /// Initial learning rate.
    pub eta0: F,
    /// Regularization strength (`alpha`).
    pub alpha: F,
    /// Regularization penalty (`l2`/`l1`/`elasticnet`). Defaults to `L2`.
    pub penalty: Penalty,
    /// Elastic-net mixing parameter; only used when `penalty == ElasticNet`.
    /// Defaults to `0.15` (sklearn default).
    pub l1_ratio: F,
    /// Maximum number of passes over the training data.
    pub max_iter: usize,
    /// Convergence tolerance. Training stops when the loss improvement
    /// is below this threshold.
    pub tol: F,
    /// Optional random seed for sample shuffling.
    pub random_state: Option<u64>,
    /// Power parameter for inverse scaling schedule.
    pub power_t: F,
    /// Whether to shuffle the training data after each epoch. Defaults to
    /// `true` (sklearn `SGDClassifier(shuffle=True)`,
    /// `_stochastic_gradient.py:107`).
    pub shuffle: bool,
    /// Number of consecutive epochs with no loss improvement (beyond `tol`)
    /// before convergence triggers, or — under the `adaptive` schedule — before
    /// `eta` is divided by 5. Defaults to `5` (sklearn
    /// `_stochastic_gradient.py` `n_iter_no_change=5`, `_sgd_fast.pyx.tp:698`).
    pub n_iter_no_change: usize,
    /// Whether to fit (update) the intercept. Defaults to `true` (sklearn
    /// `SGDClassifier(fit_intercept=True)`, `_stochastic_gradient.py:104`,
    /// constraint `["boolean"]` at `:86`). When `false` the intercept is never
    /// updated and stays at its init value `0` (`_sgd_fast.pyx.tp:639-644`: the
    /// intercept update is gated on `if fit_intercept == 1`).
    pub fit_intercept: bool,
}

impl<F: Float> SGDClassifier<F> {
    /// Create a new `SGDClassifier` with default settings.
    ///
    /// Defaults match scikit-learn's `SGDClassifier.__init__`
    /// (`_stochastic_gradient.py:1231-1256`): `loss = Hinge`,
    /// `learning_rate = Optimal`, `eta0 = 0.0`, `alpha = 0.0001`,
    /// `penalty = L2`, `l1_ratio = 0.15`, `max_iter = 1000`, `tol = 1e-3`,
    /// `power_t = 0.5`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            loss: ClassifierLoss::Hinge,
            learning_rate: LearningRateSchedule::Optimal,
            eta0: F::from(0.0).unwrap_or_else(F::zero),
            alpha: F::from(0.0001).unwrap_or_else(F::zero),
            penalty: Penalty::L2,
            l1_ratio: F::from(0.15).unwrap_or_else(F::zero),
            max_iter: 1000,
            tol: F::from(1e-3).unwrap_or_else(F::zero),
            random_state: None,
            power_t: F::from(0.5).unwrap_or_else(F::zero),
            shuffle: true,
            n_iter_no_change: 5,
            fit_intercept: true,
        }
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: ClassifierLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the learning rate schedule.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: LearningRateSchedule<F>) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the initial learning rate.
    #[must_use]
    pub fn with_eta0(mut self, eta0: F) -> Self {
        self.eta0 = eta0;
        self
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the regularization penalty (`l2`/`l1`/`elasticnet`).
    #[must_use]
    pub fn with_penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }

    /// Set the elastic-net mixing parameter (`l1_ratio`, used only when
    /// `penalty == ElasticNet`).
    #[must_use]
    pub fn with_l1_ratio(mut self, l1_ratio: F) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    /// Set the maximum number of epochs.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the power parameter for inverse scaling.
    #[must_use]
    pub fn with_power_t(mut self, power_t: F) -> Self {
        self.power_t = power_t;
        self
    }

    /// Set whether the training data is shuffled after each epoch.
    ///
    /// Mirrors sklearn's `shuffle` parameter (default `True`,
    /// `_stochastic_gradient.py:107`). With `false` the samples are visited in
    /// index order `0..n-1` every epoch (`_sgd_fast.pyx.tp:579-581`), making the
    /// fit fully deterministic and cross-impl comparable to sklearn.
    #[must_use]
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the number of consecutive non-improving epochs before convergence
    /// (or, under the `adaptive` schedule, before `eta` is divided by 5).
    ///
    /// Mirrors sklearn's `n_iter_no_change` parameter (default `5`,
    /// `_stochastic_gradient.py`); the epoch-end stop rule at
    /// `_sgd_fast.pyx.tp:698` triggers once `no_improvement_count` reaches it.
    #[must_use]
    pub fn with_n_iter_no_change(mut self, n_iter_no_change: usize) -> Self {
        self.n_iter_no_change = n_iter_no_change;
        self
    }

    /// Set whether the intercept (bias) term is fit.
    ///
    /// Mirrors sklearn's `fit_intercept` parameter (default `True`,
    /// `_stochastic_gradient.py:104`, constraint `["boolean"]` at `:86`). With
    /// `false` the intercept update is skipped every step
    /// (`_sgd_fast.pyx.tp:639-644`: `if fit_intercept == 1: ... intercept += ...`)
    /// and the intercept stays at its init value `0`.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for SGDClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract hyperparameter bundle from an `SGDClassifier`.
fn clf_hyper<F: Float>(clf: &SGDClassifier<F>) -> SGDHyper<F> {
    SGDHyper {
        learning_rate: clf.learning_rate,
        eta0: clf.eta0,
        alpha: clf.alpha,
        max_iter: clf.max_iter,
        tol: clf.tol,
        random_state: clf.random_state,
        power_t: clf.power_t,
        penalty: clf.penalty,
        l1_ratio: clf.l1_ratio,
        shuffle: clf.shuffle,
        n_iter_no_change: clf.n_iter_no_change,
        fit_intercept: clf.fit_intercept,
    }
}

/// Internal hyperparameter bundle shared between Fit and PartialFit paths.
#[derive(Debug, Clone)]
struct SGDHyper<F> {
    learning_rate: LearningRateSchedule<F>,
    eta0: F,
    alpha: F,
    max_iter: usize,
    tol: F,
    random_state: Option<u64>,
    power_t: F,
    /// Regularization penalty (l2/l1/elasticnet).
    penalty: Penalty,
    /// Elastic-net mixing parameter (only meaningful for `ElasticNet`).
    l1_ratio: F,
    /// Whether to shuffle the sample order each epoch (`_sgd_fast.pyx.tp:579`).
    shuffle: bool,
    /// Number of consecutive non-improving epochs before convergence /
    /// adaptive-eta decay triggers (`_stochastic_gradient.py` default 5,
    /// `_sgd_fast.pyx.tp:698`).
    n_iter_no_change: usize,
    /// Whether to fit (update) the intercept each step. When `false` the
    /// intercept update is skipped (`_sgd_fast.pyx.tp:639-644`).
    fit_intercept: bool,
}

/// Train a single binary classifier via SGD, updating `weights` and
/// `intercept` in place. `y_binary` must be in `{-1, +1}`.
///
/// Returns the cumulative loss and the step counter after training.
fn train_binary_sgd<F, L>(
    x: &Array2<F>,
    y_binary: &Array1<F>,
    weights: &mut Array1<F>,
    intercept: &mut F,
    loss_fn: &L,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> (F, usize)
where
    F: Float + ScalarOperand + Send + Sync + 'static,
    L: Loss<F>,
{
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut t = initial_t;
    // Epoch-end convergence state, mirroring `_sgd_fast.pyx.tp:525,532-534`:
    // `best_loss = INFINITY`, `no_improvement_count = 0`. `current_eta` carries
    // the adaptive-schedule eta (`eta = eta / 5` decay, `:700`).
    let mut best_loss = F::infinity();
    let mut current_eta = hyper.eta0;
    let mut no_improve_count: usize = 0;
    // `tol = None` upstream becomes `-INFINITY`, disabling the stop rule
    // (`tol > -INFINITY` is false, `:690`). ferrolearn encodes that as a
    // finite/-inf `tol`; the criterion is active iff `tol > -inf`.
    let tol_active = hyper.tol > F::neg_infinity();
    let max_dloss = max_dloss::<F>();
    let mut indices: Vec<usize> = (0..n_samples).collect();
    // `optimal_init` (the `optimal` schedule's t0 offset) depends on the loss
    // and alpha, so it is computed once per fit, before the epoch loop
    // (`_sgd_fast.pyx.tp:565-570`).
    let opt_init = optimal_init(loss_fn, hyper.alpha);

    // Effective l1_ratio from the penalty (`_sgd_fast.pyx.tp:558-561`):
    // `L2 -> 0.0`, `L1 -> 1.0`, `ElasticNet -> user l1_ratio`.
    let eff = effective_l1_ratio(hyper.penalty, hyper.l1_ratio);
    let apply_l1 = matches!(hyper.penalty, Penalty::L1 | Penalty::ElasticNet);
    // Tsuruoka cumulative-penalty state. `u` (scalar) accumulates the total L1
    // penalty applied so far; `q` (per-feature) records how much penalty has
    // actually been applied to each weight. Both persist for the WHOLE fit —
    // allocated once before the epoch loop, mirroring `q = np.zeros(...)` and
    // `u = 0.0` allocated once per `_plain_sgd` call (`_sgd_fast.pyx.tp:551-556`).
    let mut u = F::zero();
    let mut q: Array1<F> = Array1::zeros(n_features);

    // Build the RNG for shuffling.
    let mut rng = match hyper.random_state {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_os_rng(),
    };

    let mut total_loss = F::zero();

    for _epoch in 0..hyper.max_iter {
        // sklearn shuffles the sample order each epoch only when `shuffle` is
        // set (`_sgd_fast.pyx.tp:579-580`: `if shuffle: dataset.shuffle(seed)`).
        // With `shuffle == false` the indices stay `0..n-1` every epoch, exactly
        // matching sklearn's no-shuffle index order (`:581 for i in range(n)`).
        if hyper.shuffle {
            indices.shuffle(&mut rng);
        }
        let mut epoch_loss = F::zero();

        for &i in &indices {
            t += 1;

            let eta = match hyper.learning_rate {
                LearningRateSchedule::Adaptive => current_eta,
                _ => compute_lr(
                    &hyper.learning_rate,
                    hyper.eta0,
                    hyper.alpha,
                    hyper.power_t,
                    opt_init,
                    t,
                ),
            };

            // Compute prediction: w^T x_i + b.
            let mut y_pred = *intercept;
            let xi = x.row(i);
            for j in 0..n_features {
                y_pred = y_pred + weights[j] * xi[j];
            }

            // Clip the gradient to `[-MAX_DLOSS, MAX_DLOSS]` before forming the
            // update, matching `_sgd_fast.pyx.tp:613-620`.
            let grad = loss_fn
                .gradient(y_binary[i], y_pred)
                .max(-max_dloss)
                .min(max_dloss);
            // `sumloss` is the SUM (not mean) of per-sample losses over the
            // epoch (`_sgd_fast.pyx.tp:597`), used by the `best_loss` stop rule.
            epoch_loss = epoch_loss + loss_fn.loss(y_binary[i], y_pred);

            // L2 shrink: scale the whole weight vector by the CLAMPED factor
            // `max(0, 1 - (1-eff)*eta*alpha)` BEFORE the gradient add, mirroring
            // `w.scale(max(0, 1 - (1-l1_ratio)*eta*alpha))`
            // (`_sgd_fast.pyx.tp:632-635`). For pure L2 (`eff=0`) this is
            // `max(0, 1-eta*alpha)`; for L1 (`eff=1`) it is `max(0, 1) = 1`
            // (no L2 shrink); for elasticnet the `(1-eff)` weakens the L2 part.
            let shrink = (F::one() - (F::one() - eff) * eta * hyper.alpha).max(F::zero());
            for j in 0..n_features {
                weights[j] = weights[j] * shrink;
            }
            // Gradient add `w.add(x, -eta*grad)` (`_sgd_fast.pyx.tp:637-638`).
            for j in 0..n_features {
                weights[j] = weights[j] - eta * grad * xi[j];
            }
            // The intercept update is gated on `fit_intercept` and is NOT
            // regularized (`intercept_decay=1`, `_sgd_fast.pyx.tp:639-644`:
            // `if fit_intercept == 1: intercept_update = update; ...
            // intercept += intercept_update * intercept_decay`). When
            // `fit_intercept` is false the intercept is never modified and
            // stays at its init value `0` (`intercept` enters this fn as `0`).
            if hyper.fit_intercept {
                *intercept = *intercept - eta * grad;
            }

            // L1 cumulative penalty (Tsuruoka truncated gradient), applied AFTER
            // the gradient add only for L1/ElasticNet (`_sgd_fast.pyx.tp:656-658`,
            // `l1penalty` at `:750-778` with `wscale = 1`).
            if apply_l1 {
                u = u + eff * eta * hyper.alpha;
                for j in 0..n_features {
                    let z = weights[j];
                    if weights[j] > F::zero() {
                        weights[j] = (weights[j] - (u + q[j])).max(F::zero());
                    } else if weights[j] < F::zero() {
                        weights[j] = (weights[j] + (u - q[j])).min(F::zero());
                    }
                    q[j] = q[j] + (weights[j] - z);
                }
            }
        }

        // `epoch_loss` is now the epoch `sumloss` (no mean division). The
        // epoch-end stop rule mirrors `_sgd_fast.pyx.tp:688-707` exactly:
        // the criterion compares `sumloss` to `best_loss - tol * train_count`.
        total_loss = epoch_loss;

        if convergence_tail(
            epoch_loss,
            &mut best_loss,
            &mut no_improve_count,
            &mut current_eta,
            tol_active,
            hyper.tol,
            n_samples,
            hyper.n_iter_no_change,
            matches!(hyper.learning_rate, LearningRateSchedule::Adaptive),
        ) {
            break;
        }
    }

    (total_loss, t)
}

/// Fitted SGD classifier.
///
/// Holds the learned weight vectors and intercepts. For binary problems
/// there is a single weight vector; for multiclass problems there is one
/// per class (one-vs-all).
///
/// Implements [`Predict`] and [`PartialFit`] to support both inference and
/// online learning.
#[derive(Debug, Clone)]
pub struct FittedSGDClassifier<F> {
    /// Weight matrix: one row per binary sub-problem.
    /// Binary: shape `(1, n_features)`, multiclass: `(n_classes, n_features)`.
    weight_matrix: Vec<Array1<F>>,
    /// Intercept vector, one per sub-problem.
    intercepts: Vec<F>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Number of features the model was trained on.
    n_features: usize,
    /// The loss function used during training.
    loss: ClassifierLoss,
    /// Hyperparameters for continued training via `partial_fit`.
    hyper: SGDHyper<F>,
    /// Global step counter across all partial_fit calls.
    t: usize,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<usize>>
    for SGDClassifier<F>
{
    type Fitted = FittedSGDClassifier<F>;
    type Error = FerroError;

    /// Fit the SGD classifier on the given data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have mismatched
    /// sample counts.
    /// Returns [`FerroError::InsufficientSamples`] if fewer than 2 classes
    /// are present.
    /// Returns [`FerroError::InvalidParameter`] if `eta0` or `alpha` are
    /// not positive.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedSGDClassifier<F>, FerroError> {
        validate_clf_params(
            x,
            y,
            &self.learning_rate,
            self.eta0,
            self.alpha,
            self.l1_ratio,
        )?;

        let n_features = x.ncols();
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "SGDClassifier requires at least 2 distinct classes".into(),
            });
        }

        let hyper = clf_hyper(self);
        let loss_enum = self.loss;

        let (weight_matrix, intercepts, t) =
            fit_ova(x, y, &classes, n_features, &loss_enum, &hyper, 0)?;

        Ok(FittedSGDClassifier {
            weight_matrix,
            intercepts,
            classes,
            n_features,
            loss: loss_enum,
            hyper,
            t,
        })
    }
}

/// Whether a learning-rate schedule requires `eta0 > 0`.
///
/// Mirrors sklearn `_more_validate_params` (`_stochastic_gradient.py:149-153`):
/// `eta0 > 0` is enforced only for `constant`/`invscaling`/`adaptive`; the
/// `optimal` schedule derives its own initial rate and accepts `eta0 == 0`.
fn schedule_requires_eta0<F: Float>(schedule: &LearningRateSchedule<F>) -> bool {
    matches!(
        schedule,
        LearningRateSchedule::Constant
            | LearningRateSchedule::InvScaling
            | LearningRateSchedule::Adaptive
    )
}

/// Validate classifier input shapes and parameters.
fn validate_clf_params<F: Float>(
    x: &Array2<F>,
    y: &Array1<usize>,
    schedule: &LearningRateSchedule<F>,
    eta0: F,
    alpha: F,
    l1_ratio: F,
) -> Result<(), FerroError> {
    let n_samples = x.nrows();
    if n_samples != y.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "y length must match number of samples in X".into(),
        });
    }
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "SGDClassifier requires at least one sample".into(),
        });
    }
    if schedule_requires_eta0(schedule) && eta0 <= F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "eta0".into(),
            reason: "must be positive".into(),
        });
    }
    if alpha < F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "alpha".into(),
            reason: "must be non-negative".into(),
        });
    }
    if l1_ratio < F::zero() || l1_ratio > F::one() {
        return Err(FerroError::InvalidParameter {
            name: "l1_ratio".into(),
            reason: "must be in the range [0, 1]".into(),
        });
    }
    Ok(())
}

/// Result type for one-vs-all training: (weight_matrix, intercepts, step_counter).
type OvaResult<F> = (Vec<Array1<F>>, Vec<F>, usize);

/// Train one-vs-all binary classifiers, returning per-class weights, intercepts,
/// and the cumulative step counter.
fn fit_ova<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
    classes: &[usize],
    n_features: usize,
    loss_enum: &ClassifierLoss,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> Result<OvaResult<F>, FerroError> {
    let n_classes = classes.len();
    let mut weight_matrix: Vec<Array1<F>> = Vec::with_capacity(n_classes);
    let mut intercepts: Vec<F> = Vec::with_capacity(n_classes);
    let mut global_t = initial_t;

    if n_classes == 2 {
        // Single binary problem: class[0] -> -1, class[1] -> +1.
        let y_binary: Array1<F> = y.mapv(|label| {
            if label == classes[1] {
                F::one()
            } else {
                -F::one()
            }
        });
        let mut w = Array1::<F>::zeros(n_features);
        let mut b = F::zero();
        let (_, t) =
            dispatch_train_binary(x, &y_binary, &mut w, &mut b, loss_enum, hyper, global_t);
        global_t = t;
        weight_matrix.push(w);
        intercepts.push(b);
    } else {
        // One-vs-all: one binary problem per class.
        for &cls in classes {
            let y_binary: Array1<F> =
                y.mapv(|label| if label == cls { F::one() } else { -F::one() });
            let mut w = Array1::<F>::zeros(n_features);
            let mut b = F::zero();
            let (_, t) =
                dispatch_train_binary(x, &y_binary, &mut w, &mut b, loss_enum, hyper, global_t);
            global_t = t;
            weight_matrix.push(w);
            intercepts.push(b);
        }
    }

    Ok((weight_matrix, intercepts, global_t))
}

/// Train one-vs-all using existing weight vectors (for partial_fit).
#[allow(clippy::too_many_arguments)]
fn partial_fit_ova<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
    classes: &[usize],
    weight_matrix: &mut [Array1<F>],
    intercepts: &mut [F],
    loss_enum: &ClassifierLoss,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> usize {
    let n_classes = classes.len();
    let mut global_t = initial_t;

    if n_classes == 2 {
        let y_binary: Array1<F> = y.mapv(|label| {
            if label == classes[1] {
                F::one()
            } else {
                -F::one()
            }
        });
        let (_, t) = dispatch_train_binary(
            x,
            &y_binary,
            &mut weight_matrix[0],
            &mut intercepts[0],
            loss_enum,
            hyper,
            global_t,
        );
        global_t = t;
    } else {
        for (idx, &cls) in classes.iter().enumerate() {
            let y_binary: Array1<F> =
                y.mapv(|label| if label == cls { F::one() } else { -F::one() });
            let (_, t) = dispatch_train_binary(
                x,
                &y_binary,
                &mut weight_matrix[idx],
                &mut intercepts[idx],
                loss_enum,
                hyper,
                global_t,
            );
            global_t = t;
        }
    }

    global_t
}

/// Dispatch to the appropriate typed loss training function.
fn dispatch_train_binary<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y_binary: &Array1<F>,
    w: &mut Array1<F>,
    b: &mut F,
    loss_enum: &ClassifierLoss,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> (F, usize) {
    match loss_enum {
        ClassifierLoss::Hinge => train_binary_sgd(x, y_binary, w, b, &Hinge, hyper, initial_t),
        ClassifierLoss::SquaredHinge => {
            train_binary_sgd(x, y_binary, w, b, &SquaredHinge, hyper, initial_t)
        }
        ClassifierLoss::Perceptron => {
            train_binary_sgd(x, y_binary, w, b, &Perceptron, hyper, initial_t)
        }
        ClassifierLoss::Log => train_binary_sgd(x, y_binary, w, b, &LogLoss, hyper, initial_t),
        ClassifierLoss::SquaredError => {
            train_binary_sgd(x, y_binary, w, b, &SquaredError, hyper, initial_t)
        }
        ClassifierLoss::ModifiedHuber => {
            train_binary_sgd(x, y_binary, w, b, &ModifiedHuber, hyper, initial_t)
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedSGDClassifier<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// For binary classification, uses `sign(w^T x + b)`.
    /// For multiclass, returns the class whose one-vs-all score is highest.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::<usize>::zeros(n_samples);

        if self.classes.len() == 2 {
            // Binary: single weight vector.
            let scores = x.dot(&self.weight_matrix[0]) + self.intercepts[0];
            for i in 0..n_samples {
                predictions[i] = if scores[i] >= F::zero() {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
            }
        } else {
            // Multiclass: one-vs-all, pick highest score.
            for i in 0..n_samples {
                let xi = x.row(i);
                let mut best_class = 0;
                let mut best_score = F::neg_infinity();
                for (c, w) in self.weight_matrix.iter().enumerate() {
                    let score = xi.dot(w) + self.intercepts[c];
                    if score > best_score {
                        best_score = score;
                        best_class = c;
                    }
                }
                predictions[i] = self.classes[best_class];
            }
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> PartialFit<Array2<F>, Array1<usize>>
    for FittedSGDClassifier<F>
{
    type FitResult = FittedSGDClassifier<F>;
    type Error = FerroError;

    /// Incrementally train the classifier on a new batch of data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have mismatched
    /// sizes or `x` has the wrong number of features.
    fn partial_fit(
        mut self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedSGDClassifier<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        // Use a single-epoch hyper for partial_fit.
        let mut hyper = self.hyper.clone();
        hyper.max_iter = 1;

        let t = partial_fit_ova(
            x,
            y,
            &self.classes,
            &mut self.weight_matrix,
            &mut self.intercepts,
            &self.loss,
            &hyper,
            self.t,
        );
        self.t = t;

        Ok(self)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> PartialFit<Array2<F>, Array1<usize>>
    for SGDClassifier<F>
{
    type FitResult = FittedSGDClassifier<F>;
    type Error = FerroError;

    /// Initial call to `partial_fit` on an unfitted classifier.
    ///
    /// Equivalent to `fit` but with a single epoch, enabling subsequent
    /// incremental calls.
    ///
    /// # Errors
    ///
    /// Same as [`Fit::fit`].
    fn partial_fit(
        self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedSGDClassifier<F>, FerroError> {
        validate_clf_params(
            x,
            y,
            &self.learning_rate,
            self.eta0,
            self.alpha,
            self.l1_ratio,
        )?;

        let n_features = x.ncols();
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "SGDClassifier requires at least 2 distinct classes".into(),
            });
        }

        let mut hyper = clf_hyper(&self);
        hyper.max_iter = 1;
        let loss_enum = self.loss;

        let (weight_matrix, intercepts, t) =
            fit_ova(x, y, &classes, n_features, &loss_enum, &hyper, 0)?;

        Ok(FittedSGDClassifier {
            weight_matrix,
            intercepts,
            classes,
            n_features,
            loss: loss_enum,
            hyper: clf_hyper(&self),
            t,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedSGDClassifier<F>
{
    /// Returns the coefficient vector for the first (or only) binary classifier.
    fn coefficients(&self) -> &Array1<F> {
        &self.weight_matrix[0]
    }

    /// Returns the intercept for the first (or only) binary classifier.
    fn intercept(&self) -> F {
        self.intercepts[0]
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for SGDClassifier<F>
where
    F: Float + ToPrimitive + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedSGDClassifierPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration that converts predictions to float.
struct FittedSGDClassifierPipeline<F>(FittedSGDClassifier<F>)
where
    F: Float + Send + Sync + 'static;

// Safety: inner type fields are Send + Sync.
unsafe impl<F> Send for FittedSGDClassifierPipeline<F> where F: Float + Send + Sync + 'static {}
unsafe impl<F> Sync for FittedSGDClassifierPipeline<F> where F: Float + Send + Sync + 'static {}

impl<F> FittedPipelineEstimator<F> for FittedSGDClassifierPipeline<F>
where
    F: Float + ToPrimitive + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

// ---------------------------------------------------------------------------
// SGDRegressor
// ---------------------------------------------------------------------------

/// Stochastic Gradient Descent regressor.
///
/// Supports several loss functions for regression, trained using stochastic
/// gradient descent with configurable learning rate schedules.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_linear::sgd::SGDRegressor;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = array![2.0, 4.0, 6.0, 8.0];
///
/// let model = SGDRegressor::<f64>::new();
/// let fitted = model.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SGDRegressor<F> {
    /// The loss function to use.
    pub loss: RegressorLoss<F>,
    /// The learning rate schedule.
    pub learning_rate: LearningRateSchedule<F>,
    /// Initial learning rate.
    pub eta0: F,
    /// Regularization strength (`alpha`).
    pub alpha: F,
    /// Regularization penalty (`l2`/`l1`/`elasticnet`). Defaults to `L2`.
    pub penalty: Penalty,
    /// Elastic-net mixing parameter; only used when `penalty == ElasticNet`.
    /// Defaults to `0.15` (sklearn default).
    pub l1_ratio: F,
    /// Maximum number of passes over the training data.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Optional random seed for sample shuffling.
    pub random_state: Option<u64>,
    /// Power parameter for inverse scaling schedule.
    pub power_t: F,
    /// Whether to shuffle the training data after each epoch. Defaults to
    /// `true` (sklearn `SGDRegressor(shuffle=True)`,
    /// `_stochastic_gradient.py:2038`).
    pub shuffle: bool,
    /// Number of consecutive epochs with no loss improvement (beyond `tol`)
    /// before convergence triggers, or — under the `adaptive` schedule — before
    /// `eta` is divided by 5. Defaults to `5` (sklearn
    /// `_stochastic_gradient.py` `n_iter_no_change=5`, `_sgd_fast.pyx.tp:698`).
    pub n_iter_no_change: usize,
    /// Whether to fit (update) the intercept. Defaults to `true` (sklearn
    /// `SGDRegressor(fit_intercept=True)`, `_stochastic_gradient.py:2031`,
    /// constraint `["boolean"]` at `:86`). When `false` the intercept is never
    /// updated and stays at its init value `0` (`_sgd_fast.pyx.tp:639-644`: the
    /// intercept update is gated on `if fit_intercept == 1`).
    pub fit_intercept: bool,
}

impl<F: Float> SGDRegressor<F> {
    /// Create a new `SGDRegressor` with default settings.
    ///
    /// Defaults match scikit-learn's `SGDRegressor.__init__`
    /// (`_stochastic_gradient.py:2042-2068`): `loss = SquaredError`,
    /// `learning_rate = InvScaling`, `eta0 = 0.01`, `alpha = 0.0001`,
    /// `penalty = L2`, `l1_ratio = 0.15`, `max_iter = 1000`, `tol = 1e-3`,
    /// `power_t = 0.25`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            loss: RegressorLoss::SquaredError,
            n_iter_no_change: 5,
            fit_intercept: true,
            shuffle: true,
            penalty: Penalty::L2,
            l1_ratio: F::from(0.15).unwrap_or_else(F::zero),
            learning_rate: LearningRateSchedule::InvScaling,
            eta0: F::from(0.01).unwrap(),
            alpha: F::from(0.0001).unwrap(),
            max_iter: 1000,
            tol: F::from(1e-3).unwrap(),
            random_state: None,
            power_t: F::from(0.25).unwrap(),
        }
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: RegressorLoss<F>) -> Self {
        self.loss = loss;
        self
    }

    /// Set the learning rate schedule.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: LearningRateSchedule<F>) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the initial learning rate.
    #[must_use]
    pub fn with_eta0(mut self, eta0: F) -> Self {
        self.eta0 = eta0;
        self
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the regularization penalty (`l2`/`l1`/`elasticnet`).
    #[must_use]
    pub fn with_penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }

    /// Set the elastic-net mixing parameter (`l1_ratio`, used only when
    /// `penalty == ElasticNet`).
    #[must_use]
    pub fn with_l1_ratio(mut self, l1_ratio: F) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    /// Set the maximum number of epochs.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the power parameter for inverse scaling.
    #[must_use]
    pub fn with_power_t(mut self, power_t: F) -> Self {
        self.power_t = power_t;
        self
    }

    /// Set whether the training data is shuffled after each epoch.
    ///
    /// Mirrors sklearn's `shuffle` parameter (default `True`,
    /// `_stochastic_gradient.py:2038`). With `false` the samples are visited in
    /// index order `0..n-1` every epoch (`_sgd_fast.pyx.tp:579-581`), making the
    /// fit fully deterministic and cross-impl comparable to sklearn.
    #[must_use]
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the number of consecutive non-improving epochs before convergence
    /// (or, under the `adaptive` schedule, before `eta` is divided by 5).
    ///
    /// Mirrors sklearn's `n_iter_no_change` parameter (default `5`,
    /// `_stochastic_gradient.py`); the epoch-end stop rule at
    /// `_sgd_fast.pyx.tp:698` triggers once `no_improvement_count` reaches it.
    #[must_use]
    pub fn with_n_iter_no_change(mut self, n_iter_no_change: usize) -> Self {
        self.n_iter_no_change = n_iter_no_change;
        self
    }

    /// Set whether the intercept (bias) term is fit.
    ///
    /// Mirrors sklearn's `fit_intercept` parameter (default `True`,
    /// `_stochastic_gradient.py:2031`, constraint `["boolean"]` at `:86`). With
    /// `false` the intercept update is skipped every step
    /// (`_sgd_fast.pyx.tp:639-644`: `if fit_intercept == 1: ... intercept += ...`)
    /// and the intercept stays at its init value `0`.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for SGDRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract hyperparameter bundle from an `SGDRegressor`.
fn reg_hyper<F: Float>(reg: &SGDRegressor<F>) -> SGDHyper<F> {
    SGDHyper {
        learning_rate: reg.learning_rate,
        eta0: reg.eta0,
        alpha: reg.alpha,
        max_iter: reg.max_iter,
        tol: reg.tol,
        random_state: reg.random_state,
        power_t: reg.power_t,
        penalty: reg.penalty,
        l1_ratio: reg.l1_ratio,
        shuffle: reg.shuffle,
        n_iter_no_change: reg.n_iter_no_change,
        fit_intercept: reg.fit_intercept,
    }
}

/// Train a single regressor via SGD, updating `weights` and `intercept`
/// in place. Returns the final loss and step counter.
fn train_regressor_sgd<F, L>(
    x: &Array2<F>,
    y: &Array1<F>,
    weights: &mut Array1<F>,
    intercept: &mut F,
    loss_fn: &L,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> (F, usize)
where
    F: Float + ScalarOperand + Send + Sync + 'static,
    L: Loss<F>,
{
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut t = initial_t;
    // Epoch-end convergence state, mirroring `_sgd_fast.pyx.tp:525,532-534`:
    // `best_loss = INFINITY`, `no_improvement_count = 0`. `current_eta` carries
    // the adaptive-schedule eta (`eta = eta / 5` decay, `:700`).
    let mut best_loss = F::infinity();
    let mut current_eta = hyper.eta0;
    let mut no_improve_count: usize = 0;
    // `tol = None` upstream becomes `-INFINITY`, disabling the stop rule
    // (`tol > -INFINITY` is false, `:690`). ferrolearn encodes that as a
    // finite/-inf `tol`; the criterion is active iff `tol > -inf`.
    let tol_active = hyper.tol > F::neg_infinity();
    let max_dloss = max_dloss::<F>();
    let mut indices: Vec<usize> = (0..n_samples).collect();
    // `optimal_init` (the `optimal` schedule's t0 offset) depends on the loss
    // and alpha, so it is computed once per fit, before the epoch loop
    // (`_sgd_fast.pyx.tp:565-570`).
    let opt_init = optimal_init(loss_fn, hyper.alpha);

    // Effective l1_ratio from the penalty (`_sgd_fast.pyx.tp:558-561`):
    // `L2 -> 0.0`, `L1 -> 1.0`, `ElasticNet -> user l1_ratio`.
    let eff = effective_l1_ratio(hyper.penalty, hyper.l1_ratio);
    let apply_l1 = matches!(hyper.penalty, Penalty::L1 | Penalty::ElasticNet);
    // Tsuruoka cumulative-penalty state (`u` scalar, `q` per-feature), allocated
    // once and persisting for the whole fit (`_sgd_fast.pyx.tp:551-556`).
    let mut u = F::zero();
    let mut q: Array1<F> = Array1::zeros(n_features);

    let mut rng = match hyper.random_state {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_os_rng(),
    };

    let mut total_loss = F::zero();

    for _epoch in 0..hyper.max_iter {
        // sklearn shuffles the sample order each epoch only when `shuffle` is
        // set (`_sgd_fast.pyx.tp:579-580`: `if shuffle: dataset.shuffle(seed)`).
        // With `shuffle == false` the indices stay `0..n-1` every epoch, exactly
        // matching sklearn's no-shuffle index order (`:581 for i in range(n)`).
        if hyper.shuffle {
            indices.shuffle(&mut rng);
        }
        let mut epoch_loss = F::zero();

        for &i in &indices {
            t += 1;

            let eta = match hyper.learning_rate {
                LearningRateSchedule::Adaptive => current_eta,
                _ => compute_lr(
                    &hyper.learning_rate,
                    hyper.eta0,
                    hyper.alpha,
                    hyper.power_t,
                    opt_init,
                    t,
                ),
            };

            let xi = x.row(i);
            let mut y_pred = *intercept;
            for j in 0..n_features {
                y_pred = y_pred + weights[j] * xi[j];
            }

            // Clip the gradient to `[-MAX_DLOSS, MAX_DLOSS]` before forming the
            // update, matching `_sgd_fast.pyx.tp:613-620`.
            let grad = loss_fn
                .gradient(y[i], y_pred)
                .max(-max_dloss)
                .min(max_dloss);
            // `sumloss` is the SUM (not mean) of per-sample losses over the
            // epoch (`_sgd_fast.pyx.tp:597`), used by the `best_loss` stop rule.
            epoch_loss = epoch_loss + loss_fn.loss(y[i], y_pred);

            // L2 shrink: clamped multiplicative factor
            // `max(0, 1 - (1-eff)*eta*alpha)` applied to the whole weight vector
            // BEFORE the gradient add (`_sgd_fast.pyx.tp:632-635`); for pure L2
            // (`eff=0`) this is `max(0, 1-eta*alpha)`, for L1 (`eff=1`) it is 1.
            let shrink = (F::one() - (F::one() - eff) * eta * hyper.alpha).max(F::zero());
            for j in 0..n_features {
                weights[j] = weights[j] * shrink;
            }
            // Gradient add `w.add(x, -eta*grad)` (`_sgd_fast.pyx.tp:637-638`).
            for j in 0..n_features {
                weights[j] = weights[j] - eta * grad * xi[j];
            }
            // The intercept update is gated on `fit_intercept` and is NOT
            // regularized (`_sgd_fast.pyx.tp:639-644`: `if fit_intercept == 1:
            // intercept_update = update; ... intercept += intercept_update *
            // intercept_decay`). When `fit_intercept` is false the intercept is
            // never modified and stays at its init value `0` (`intercept` enters
            // this fn as `0`).
            if hyper.fit_intercept {
                *intercept = *intercept - eta * grad;
            }

            // L1 cumulative penalty (Tsuruoka truncated gradient), applied AFTER
            // the gradient add only for L1/ElasticNet (`_sgd_fast.pyx.tp:656-658`,
            // `l1penalty` at `:750-778` with `wscale = 1`).
            if apply_l1 {
                u = u + eff * eta * hyper.alpha;
                for j in 0..n_features {
                    let z = weights[j];
                    if weights[j] > F::zero() {
                        weights[j] = (weights[j] - (u + q[j])).max(F::zero());
                    } else if weights[j] < F::zero() {
                        weights[j] = (weights[j] + (u - q[j])).min(F::zero());
                    }
                    q[j] = q[j] + (weights[j] - z);
                }
            }
        }

        // `epoch_loss` is now the epoch `sumloss` (no mean division). The
        // epoch-end stop rule mirrors `_sgd_fast.pyx.tp:688-707` exactly.
        total_loss = epoch_loss;

        if convergence_tail(
            epoch_loss,
            &mut best_loss,
            &mut no_improve_count,
            &mut current_eta,
            tol_active,
            hyper.tol,
            n_samples,
            hyper.n_iter_no_change,
            matches!(hyper.learning_rate, LearningRateSchedule::Adaptive),
        ) {
            break;
        }
    }

    (total_loss, t)
}

/// Dispatch regressor training to the appropriate typed loss function.
fn dispatch_train_regressor<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    w: &mut Array1<F>,
    b: &mut F,
    loss_enum: &RegressorLoss<F>,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> (F, usize) {
    match loss_enum {
        RegressorLoss::SquaredError => {
            train_regressor_sgd(x, y, w, b, &SquaredError, hyper, initial_t)
        }
        RegressorLoss::Huber(eps) => {
            train_regressor_sgd(x, y, w, b, &Huber { epsilon: *eps }, hyper, initial_t)
        }
        RegressorLoss::EpsilonInsensitive(eps) => train_regressor_sgd(
            x,
            y,
            w,
            b,
            &EpsilonInsensitive { epsilon: *eps },
            hyper,
            initial_t,
        ),
        RegressorLoss::SquaredEpsilonInsensitive(eps) => train_regressor_sgd(
            x,
            y,
            w,
            b,
            &SquaredEpsilonInsensitive { epsilon: *eps },
            hyper,
            initial_t,
        ),
    }
}

/// Fitted SGD regressor.
///
/// Holds the learned weight vector and intercept. Implements [`Predict`]
/// and [`PartialFit`] to support both inference and online learning.
#[derive(Debug, Clone)]
pub struct FittedSGDRegressor<F> {
    /// Learned weight vector (one per feature).
    weights: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
    /// Number of features the model was trained on.
    n_features: usize,
    /// The loss function used during training.
    loss: RegressorLoss<F>,
    /// Hyperparameters for continued training.
    hyper: SGDHyper<F>,
    /// Global step counter.
    t: usize,
}

/// Validate regressor input shapes and parameters.
fn validate_reg_params<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    schedule: &LearningRateSchedule<F>,
    eta0: F,
    alpha: F,
    l1_ratio: F,
) -> Result<(), FerroError> {
    let n_samples = x.nrows();
    if n_samples != y.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "y length must match number of samples in X".into(),
        });
    }
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "SGDRegressor requires at least one sample".into(),
        });
    }
    if schedule_requires_eta0(schedule) && eta0 <= F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "eta0".into(),
            reason: "must be positive".into(),
        });
    }
    if alpha < F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "alpha".into(),
            reason: "must be non-negative".into(),
        });
    }
    if l1_ratio < F::zero() || l1_ratio > F::one() {
        return Err(FerroError::InvalidParameter {
            name: "l1_ratio".into(),
            reason: "must be in the range [0, 1]".into(),
        });
    }
    Ok(())
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<F>>
    for SGDRegressor<F>
{
    type Fitted = FittedSGDRegressor<F>;
    type Error = FerroError;

    /// Fit the SGD regressor on the given data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have mismatched
    /// sample counts.
    /// Returns [`FerroError::InvalidParameter`] if `eta0` or `alpha` are
    /// invalid.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedSGDRegressor<F>, FerroError> {
        validate_reg_params(
            x,
            y,
            &self.learning_rate,
            self.eta0,
            self.alpha,
            self.l1_ratio,
        )?;

        let n_features = x.ncols();
        let hyper = reg_hyper(self);
        let mut w = Array1::<F>::zeros(n_features);
        let mut b = F::zero();

        let (_, t) = dispatch_train_regressor(x, y, &mut w, &mut b, &self.loss, &hyper, 0);

        Ok(FittedSGDRegressor {
            weights: w,
            intercept: b,
            n_features,
            loss: self.loss,
            hyper,
            t,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedSGDRegressor<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `X @ weights + intercept`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let preds = x.dot(&self.weights) + self.intercept;
        Ok(preds)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> PartialFit<Array2<F>, Array1<F>>
    for FittedSGDRegressor<F>
{
    type FitResult = FittedSGDRegressor<F>;
    type Error = FerroError;

    /// Incrementally train the regressor on a new batch of data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have mismatched
    /// sizes or `x` has the wrong number of features.
    fn partial_fit(
        mut self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedSGDRegressor<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let mut hyper = self.hyper.clone();
        hyper.max_iter = 1;

        let (_, t) = dispatch_train_regressor(
            x,
            y,
            &mut self.weights,
            &mut self.intercept,
            &self.loss,
            &hyper,
            self.t,
        );
        self.t = t;

        Ok(self)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> PartialFit<Array2<F>, Array1<F>>
    for SGDRegressor<F>
{
    type FitResult = FittedSGDRegressor<F>;
    type Error = FerroError;

    /// Initial call to `partial_fit` on an unfitted regressor.
    ///
    /// Equivalent to `fit` but with a single epoch.
    ///
    /// # Errors
    ///
    /// Same as [`Fit::fit`].
    fn partial_fit(
        self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedSGDRegressor<F>, FerroError> {
        validate_reg_params(
            x,
            y,
            &self.learning_rate,
            self.eta0,
            self.alpha,
            self.l1_ratio,
        )?;

        let n_features = x.ncols();
        let mut hyper = reg_hyper(&self);
        hyper.max_iter = 1;
        let mut w = Array1::<F>::zeros(n_features);
        let mut b = F::zero();

        let (_, t) = dispatch_train_regressor(x, y, &mut w, &mut b, &self.loss, &hyper, 0);

        Ok(FittedSGDRegressor {
            weights: w,
            intercept: b,
            n_features,
            loss: self.loss,
            hyper: reg_hyper(&self),
            t,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedSGDRegressor<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.weights
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for SGDRegressor<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedSGDRegressor<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // Loss function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hinge_loss_correct_side() {
        let h = Hinge;
        // y=1, pred=2 => margin=2 >= 1 => loss=0
        assert!((Loss::<f64>::loss(&h, 1.0, 2.0) - 0.0).abs() < 1e-10);
        assert!((Loss::<f64>::gradient(&h, 1.0, 2.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hinge_loss_wrong_side() {
        let h = Hinge;
        // y=1, pred=-0.5 => margin=-0.5 < 1 => loss=1.5
        assert!((Loss::<f64>::loss(&h, 1.0, -0.5) - 1.5).abs() < 1e-10);
        assert!((Loss::<f64>::gradient(&h, 1.0, -0.5) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_log_loss_zero_pred() {
        let l = LogLoss;
        // y=1, pred=0 => loss=log(1+exp(0))=log(2)
        let loss = Loss::<f64>::loss(&l, 1.0, 0.0);
        assert!((loss - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_loss_large_correct() {
        let l = LogLoss;
        // y=1, pred=20 => very small loss
        let loss = Loss::<f64>::loss(&l, 1.0, 20.0);
        assert!(loss < 1e-5);
    }

    #[test]
    fn test_squared_error_loss() {
        let s = SquaredError;
        assert!((Loss::<f64>::loss(&s, 3.0, 1.0) - 2.0).abs() < 1e-10);
        assert!((Loss::<f64>::gradient(&s, 3.0, 1.0) - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_modified_huber_loss() {
        let mh = ModifiedHuber;
        // y=1, pred=2 => z=2 >= 1 => loss=0
        assert!((Loss::<f64>::loss(&mh, 1.0, 2.0)).abs() < 1e-10);
        // y=1, pred=0.5 => z=0.5 => loss=(1-0.5)^2=0.25
        assert!((Loss::<f64>::loss(&mh, 1.0, 0.5) - 0.25).abs() < 1e-10);
        // y=1, pred=-2 => z=-2 < -1 => loss=-4*(-2)=8
        assert!((Loss::<f64>::loss(&mh, 1.0, -2.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss_quadratic_region() {
        let h = Huber { epsilon: 1.0_f64 };
        // |y - p| = 0.5 <= 1.0 => quadratic
        assert!((Loss::<f64>::loss(&h, 1.0, 0.5) - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss_linear_region() {
        let h = Huber { epsilon: 1.0_f64 };
        // |y - p| = 3 > 1 => linear: 1*(3 - 0.5) = 2.5
        assert!((Loss::<f64>::loss(&h, 3.0, 0.0) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_insensitive_inside() {
        let ei = EpsilonInsensitive { epsilon: 0.1_f64 };
        // |y - p| = 0.05 <= 0.1 => loss=0
        assert!((Loss::<f64>::loss(&ei, 1.0, 0.95)).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_insensitive_outside() {
        let ei = EpsilonInsensitive { epsilon: 0.1_f64 };
        // |y - p| = 0.5 > 0.1 => loss=0.4
        assert!((Loss::<f64>::loss(&ei, 1.0, 0.5) - 0.4).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Learning rate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_lr() {
        let lr: LearningRateSchedule<f64> = LearningRateSchedule::Constant;
        // optimal_init is ignored by the Constant schedule.
        assert!((compute_lr(&lr, 0.1, 0.01, 0.25, 1.0, 100) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_optimal_lr() {
        // sklearn `_sgd_fast.pyx.tp:592`: `eta = 1/(alpha*(optimal_init+t-1))`.
        // At t=1 this equals `initial_eta0 = 1/(alpha*optimal_init)`. With
        // alpha=0.01 and Hinge `dloss(1,-typw)` (= -1, |·|=1), the live oracle
        // gives `typw = sqrt(1/sqrt(0.01)) = 3.1622776601683795` and
        // `optimal_init = 1/(typw*0.01) = 31.62277660168379` (computed by
        // `_sgd_fast`'s init block), so eta@t=1 == typw.
        let lr: LearningRateSchedule<f64> = LearningRateSchedule::Optimal;
        const OPTIMAL_INIT: f64 = 31.62277660168379;
        const ETA_T1: f64 = 3.1622776601683795; // = typw = initial_eta0
        assert!((compute_lr(&lr, 0.0, 0.01, 0.5, OPTIMAL_INIT, 1) - ETA_T1).abs() < 1e-9);
        // At t=10: eta = 1/(0.01*(31.62277660168379 + 9)).
        let expected_t10 = 1.0 / (0.01 * (OPTIMAL_INIT + 9.0));
        assert!((compute_lr(&lr, 0.0, 0.01, 0.5, OPTIMAL_INIT, 10) - expected_t10).abs() < 1e-10);
    }

    #[test]
    fn test_optimal_init_matches_oracle() {
        // optimal_init derivation matches the live sklearn oracle (Hinge):
        // python3 -c "import numpy as np; from sklearn.linear_model._sgd_fast \
        //   import Hinge; a=0.01; typw=np.sqrt(1/np.sqrt(a)); \
        //   e0=typw/max(1.0,abs(Hinge(1.0).py_dloss(1.0,-typw))); print(1/(e0*a))"
        // -> 31.62277660168379
        const SK_OPTIMAL_INIT: f64 = 31.62277660168379;
        let got = optimal_init(&Hinge, 0.01_f64);
        assert!((got - SK_OPTIMAL_INIT).abs() < 1e-9, "got {got}");
        // alpha == 0 returns 1.0 (schedule guarded / unused upstream).
        assert!((optimal_init(&Hinge, 0.0_f64) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_invscaling_lr() {
        let lr: LearningRateSchedule<f64> = LearningRateSchedule::InvScaling;
        // eta = 0.1 / 10^0.5 = 0.1 / 3.162... ~= 0.0316...
        let result = compute_lr(&lr, 0.1, 0.01, 0.5, 1.0, 10);
        let expected = 0.1 / 10.0_f64.sqrt();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_lr_returns_eta0() {
        let lr: LearningRateSchedule<f64> = LearningRateSchedule::Adaptive;
        assert!((compute_lr(&lr, 0.05, 0.01, 0.25, 1.0, 100) - 0.05).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // SGDClassifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sgd_classifier_binary() {
        // Well-separated clusters centered near origin for SGD stability.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                -2.0, -2.0, -1.5, -2.0, -2.0, -1.5, -1.5, -1.5, 2.0, 2.0, 1.5, 2.0, 2.0, 1.5, 1.5,
                1.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let clf = SGDClassifier::<f64>::new()
            .with_loss(ClassifierLoss::Log)
            .with_random_state(42)
            .with_max_iter(1000)
            .with_eta0(0.01);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "expected >= 6 correct, got {correct}");
    }

    #[test]
    fn test_sgd_classifier_log_loss() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = SGDClassifier::<f64>::new()
            .with_loss(ClassifierLoss::Log)
            .with_random_state(42)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 4, "expected >= 4 correct, got {correct}");
    }

    #[test]
    fn test_sgd_classifier_multiclass() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0,
                0.0, 5.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let clf = SGDClassifier::<f64>::new()
            .with_random_state(42)
            .with_max_iter(1000)
            .with_eta0(0.01);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(
            correct >= 6,
            "expected >= 6 correct for multiclass, got {correct}"
        );
    }

    #[test]
    fn test_sgd_classifier_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length
        let clf = SGDClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_sgd_classifier_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];
        let clf = SGDClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_invalid_eta0() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];
        // sklearn enforces `eta0 > 0` only for constant/invscaling/adaptive
        // (`_stochastic_gradient.py:149-153`); under the default `optimal`
        // schedule `eta0 = 0.0` is valid, so use `constant` to hit the reject.
        let clf = SGDClassifier::<f64>::new()
            .with_learning_rate(LearningRateSchedule::Constant)
            .with_eta0(0.0);
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_optimal_eta0_zero_ok() {
        // Default `optimal` schedule with `eta0 = 0.0` (sklearn default) must
        // NOT be rejected by validation (`_stochastic_gradient.py:149-153`).
        let x = Array2::from_shape_vec((4, 1), vec![-2.0, -1.0, 1.0, 2.0]).unwrap_or_default();
        let y = array![0, 0, 1, 1];
        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        assert!((clf.eta0 - 0.0).abs() < 1e-12);
        assert!(clf.fit(&x, &y).is_ok());
    }

    #[test]
    fn test_sgd_classifier_invalid_alpha() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = SGDClassifier::<f64>::new().with_alpha(-1.0);
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_has_coefficients() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_sgd_classifier_partial_fit() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let fitted = clf.partial_fit(&x, &y).unwrap();
        let fitted = fitted.partial_fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_sgd_classifier_partial_fit_chain() {
        // Test the chaining pattern:
        // model.partial_fit(&b1, &y1)?.partial_fit(&b2, &y2)?.predict(&x)?
        let x1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y1 = array![0, 0, 1, 1];
        let x2 =
            Array2::from_shape_vec((4, 2), vec![0.5, 0.5, 1.5, 1.5, 7.5, 7.5, 8.5, 8.5]).unwrap();
        let y2 = array![0, 0, 1, 1];

        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let preds = clf
            .partial_fit(&x1, &y1)
            .unwrap()
            .partial_fit(&x2, &y2)
            .unwrap()
            .predict(&x1)
            .unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_sgd_classifier_partial_fit_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let fitted = clf.partial_fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let y_bad = array![0, 1];
        assert!(fitted.partial_fit(&x_bad, &y_bad).is_err());
    }

    #[test]
    fn test_sgd_classifier_modified_huber() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = SGDClassifier::<f64>::new()
            .with_loss(ClassifierLoss::ModifiedHuber)
            .with_random_state(42)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_sgd_classifier_squared_error_loss() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = SGDClassifier::<f64>::new()
            .with_loss(ClassifierLoss::SquaredError)
            .with_random_state(42)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_sgd_classifier_pipeline() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let fitted = clf.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_sgd_classifier_constant_lr() -> Result<(), FerroError> {
        let x = Array2::from_shape_vec((4, 1), vec![-2.0, -1.0, 1.0, 2.0]).unwrap_or_default();
        let y = array![0, 0, 1, 1];

        // The `constant` schedule requires `eta0 > 0`; the default is now 0.0
        // (sklearn `optimal` default), so set it explicitly.
        let clf = SGDClassifier::<f64>::new()
            .with_learning_rate(LearningRateSchedule::Constant)
            .with_eta0(0.01)
            .with_random_state(42)
            .with_max_iter(200);
        let fitted = clf.fit(&x, &y)?;
        assert_eq!(fitted.predict(&x)?.len(), 4);
        Ok(())
    }

    #[test]
    fn test_sgd_classifier_f32() {
        let x = Array2::from_shape_vec((4, 1), vec![-2.0f32, -1.0, 1.0, 2.0]).unwrap();
        let y = array![0_usize, 0, 1, 1];

        let clf = SGDClassifier::<f32>::new()
            .with_random_state(42)
            .with_max_iter(200);
        let fitted = clf.fit(&x, &y).unwrap();
        assert_eq!(fitted.predict(&x).unwrap().len(), 4);
    }

    // -----------------------------------------------------------------------
    // SGDRegressor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sgd_regressor_basic() {
        // y = 2*x + 1
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = SGDRegressor::<f64>::new()
            .with_random_state(42)
            .with_max_iter(2000)
            .with_eta0(0.01)
            .with_alpha(0.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Check rough accuracy.
        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert!(
                (*p - actual).abs() < 2.0,
                "prediction {p} too far from {actual}"
            );
        }
    }

    #[test]
    fn test_sgd_regressor_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0]; // Wrong length
        let model = SGDRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_regressor_predict_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_sgd_regressor_invalid_eta0() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let model = SGDRegressor::<f64>::new().with_eta0(-0.1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_regressor_has_coefficients() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_sgd_regressor_partial_fit() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let fitted = model.partial_fit(&x, &y).unwrap();
        let fitted = fitted.partial_fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_sgd_regressor_partial_fit_chain() {
        let x1 = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y1 = array![2.0, 4.0, 6.0];
        let x2 = Array2::from_shape_vec((3, 1), vec![4.0, 5.0, 6.0]).unwrap();
        let y2 = array![8.0, 10.0, 12.0];

        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let preds = model
            .partial_fit(&x1, &y1)
            .unwrap()
            .partial_fit(&x2, &y2)
            .unwrap()
            .predict(&x1)
            .unwrap();
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_sgd_regressor_partial_fit_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let fitted = model.partial_fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let y_bad = array![1.0, 2.0];
        assert!(fitted.partial_fit(&x_bad, &y_bad).is_err());
    }

    #[test]
    fn test_sgd_regressor_huber_loss() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = SGDRegressor::<f64>::new()
            .with_loss(RegressorLoss::Huber(1.35))
            .with_random_state(42)
            .with_max_iter(500);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.predict(&x).unwrap().len(), 4);
    }

    #[test]
    fn test_sgd_regressor_epsilon_insensitive() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = SGDRegressor::<f64>::new()
            .with_loss(RegressorLoss::EpsilonInsensitive(0.1))
            .with_random_state(42)
            .with_max_iter(500);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.predict(&x).unwrap().len(), 4);
    }

    #[test]
    fn test_sgd_regressor_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_sgd_regressor_f32() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![2.0f32, 4.0, 6.0, 8.0]);

        let model = SGDRegressor::<f32>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.predict(&x).unwrap().len(), 4);
    }

    #[test]
    fn test_sgd_regressor_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        let model = SGDRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);
        let clf = SGDClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_default() {
        // sklearn `SGDClassifier()` defaults (live oracle / `:1242-1244`):
        // learning_rate='optimal', eta0=0.0, alpha=0.0001, power_t=0.5.
        let clf = SGDClassifier::<f64>::default();
        assert!(matches!(clf.learning_rate, LearningRateSchedule::Optimal));
        assert!((clf.eta0 - 0.0).abs() < 1e-12);
        assert!((clf.alpha - 0.0001).abs() < 1e-12);
        assert!((clf.power_t - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_sgd_regressor_default() {
        let model = SGDRegressor::<f64>::default();
        assert!(model.eta0 > 0.0);
        assert!(model.alpha >= 0.0);
    }
}
