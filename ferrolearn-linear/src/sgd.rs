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
//! | REQ-9 (default params per estimator) | SHIPPED (classifier defaults) | `SGDClassifier::new` now sets `learning_rate=Optimal, eta0=0.0, power_t=0.5` (`_stochastic_gradient.py:1242-1244`); `fn schedule_requires_eta0` gates the `eta0>0` validation to constant/invscaling/adaptive (`_stochastic_gradient.py:149-153`). Consumer: `Fit for SGDClassifier`. Tests: divergence `sgd_classifier_default_learning_rate`, `test_sgd_classifier_default`, `test_sgd_classifier_optimal_eta0_zero_ok`. Closed #529. `epsilon` is validated to `[0, inf)` in `fn validate_reg_params` (`RegressorLoss::{Huber,EpsilonInsensitive,SquaredEpsilonInsensitive}(e)` reject `e < 0` with `FerroError::InvalidParameter { name: "epsilon" }`, mirroring `_stochastic_gradient.py:2024` `"epsilon": [Interval(Real, 0, None, closed="left")]`; test `sgd_epsilon_negative_rejected`, closes #544). Remaining missing fields (`fit_intercept`, `early_stopping`, `validation_fraction`, `average`, `warm_start`, `class_weight`, `C`) tracked under their own blockers. (`penalty`/`l1_ratio` shipped under REQ-5; `shuffle` under REQ-12; `n_iter_no_change` folded into REQ-10.) |
//! | REQ-10 (convergence best_loss/n_iter_no_change/sumloss) | SHIPPED | `fn convergence_tail` (shared by `fn train_binary_sgd`/`train_regressor_sgd`) tracks `best_loss` (running min, init `+inf`) and increments `no_improve_count` when `tol_active && sumloss > best_loss - tol*n_samples`, resetting otherwise; breaks once `no_improve_count >= hyper.n_iter_no_change` (non-adaptive), exactly mirroring `_sgd_fast.pyx.tp:688-707`. `sumloss` is now the SUM of per-sample losses over the epoch (the `/= n_samples` mean division is removed, `_sgd_fast.pyx.tp:597`); `tol_active = hyper.tol > -inf` encodes sklearn's `tol=None -> -INFINITY` disable (`:690`). The per-sample gradient is clipped to `[-1e12, 1e12]` via `fn max_dloss` before the update (`_sgd_fast.pyx.tp:546,613-620`). `n_iter_no_change` is now a settable `pub` field (default 5) with `fn with_n_iter_no_change` on both estimators, threaded through `SGDHyper`/`clf_hyper`/`reg_hyper` (`_stochastic_gradient.py` `n_iter_no_change=5`). Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `PipelineEstimator`. Test: divergence `sgd_convergence_n_iter_no_change` (live oracle coef `[0.8037686404055491, 0.16059017315681692]` intercept `0.12903834217696583`, n_iter_ 49). Closes #530. |
//! | REQ-11 (fit_intercept) | SHIPPED | `pub fit_intercept: bool` field on `SGDClassifier`/`SGDRegressor` + `fn with_fit_intercept` builders (default `true`, sklearn `_stochastic_gradient.py` `fit_intercept=True`, constraint `["boolean"]` at `:86`), threaded through `SGDHyper.fit_intercept` + `fn clf_hyper`/`reg_hyper`. `fn train_binary_sgd`/`train_regressor_sgd` gate the intercept update: `if hyper.fit_intercept { *intercept = *intercept - eta * grad; }`, mirroring `if fit_intercept == 1: intercept_update = update; ... intercept += intercept_update * intercept_decay` (`_sgd_fast.pyx.tp:639-644`, `intercept_decay=1` on the standard path). When `false` the intercept is never modified and stays at its init value `0` (`b = F::zero()` before training in `fn fit_ova`/regressor `Fit::fit`), so `coef_` matches sklearn and `intercept_` is exactly `0`. Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `PipelineEstimator`. Test: divergence `sgd_fit_intercept_false` (live oracle coef `[0.5326796739094939, 0.44573604649819804]`, intercept exactly `0.0`). Closes #531. |
//! | REQ-12 (shuffle flag) | SHIPPED | `pub shuffle: bool` field on `SGDClassifier`/`SGDRegressor` + `fn with_shuffle` builders (default `true`, `_stochastic_gradient.py:107` `shuffle=True`, constraint `["boolean"]` at `:89`), threaded through `SGDHyper.shuffle` + `fn clf_hyper`/`reg_hyper`. `fn train_binary_sgd`/`train_regressor_sgd` gate the per-epoch shuffle: `if hyper.shuffle { indices.shuffle(&mut rng); }`, mirroring `if shuffle: dataset.shuffle(seed)` (`_sgd_fast.pyx.tp:579-580`); when off, `indices` stays `0..n-1` each epoch matching sklearn's no-shuffle index order (`:581` `for i in range(n_samples)`). Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `PipelineEstimator`. Tests: divergence `sgd_shuffle_false_multisample_kernel_parity` (4-sample/2-feature/5-epoch L2 oracle coef `[0.5103165909636498, 0.42319810364130317]` intercept `0.16255331549195393`; elasticnet l1_ratio=0.3 oracle coef `[0.5102136050112174, 0.4230749783888256]` intercept `0.16265294456399926`). Closes #532. This `shuffle=false` parity ALSO validates REQ-4/REQ-5/REQ-6 (L2 shrink + elasticnet truncated gradient + constant schedule) over MULTIPLE samples and epochs against the live oracle — previously only single-sample. |
//! | REQ-13 (early_stopping + validation_fraction + n_iter_no_change-on-val-score) | SHIPPED (for the verifiable logic; the validation-split SELECTION is numpy-RNG-coupled so full fitted-coef parity is NOT oracle-verifiable — same barrier as `shuffle`) | `pub early_stopping: bool` (default `false`) + `pub validation_fraction: F` (default `0.1` via `cst`) fields on `SGDClassifier`/`SGDRegressor` + `fn with_early_stopping`/`with_validation_fraction` builders (`_stochastic_gradient.py:114-115`, constraints `["boolean"]`/`Interval(Real, 0, 1, closed="neither")` at `:524-525`), threaded through `SGDHyper.early_stopping`/`validation_fraction` (`fn clf_hyper`/`reg_hyper`; one-class hardwires `false`/`0.1`). `fn validate_validation_fraction` rejects `validation_fraction` outside the OPEN interval `(0, 1)` (`FerroError::InvalidParameter`, called from `fn validate_clf_params`/`validate_reg_params`). When `early_stopping`, the `Fit` path (`fn fit_ova` for the classifier, `SGDRegressor::fit_with_sample_weight` for the regressor) splits the data via `fn make_validation_split` BEFORE the kernel: a seeded (`StdRng::seed_from_u64(random_state.unwrap_or(0))`) hold-out of `fn validation_count` (`ceil(validation_fraction*n)`, clamped to `[1,n-1]`) samples — STRATIFIED per class for the classifier (mirrors `StratifiedShuffleSplit`, `:280-287`; the multiclass split is computed ONCE on the full `y` and SHARED across OvA subproblems, `:796`), plain `ShuffleSplit` for the regressor — returning `Err` on an empty train/val subset (`:295-307`). The kernel trains on the TRAIN subset only (sklearn does NOT refit on full data) and, at epoch-end, when early stopping, scores the CURRENT (weights,intercept) on the held-out val set via `fn convergence_tail_score` with `best_score` init `-inf`: `if tol_active && score < best_score + tol { no_improve++ } else { 0 }; if score > best_score { best_score = score }` then the SHARED `no_improve >= n_iter_no_change` adaptive-÷5/break tail (`_sgd_fast.pyx.tp:678-707`). The val score is `fn r2_score` for the regressor (`1 - SS_res/SS_tot`, `SS_tot==0` -> `1.0`/`0.0` edge) and `fn binary_accuracy` for each classifier subproblem (relabeled `{-1,+1}` target, `decision>=0 -> +1` tie convention), mirroring `_ValidationScoreCallback.__call__` = `est.score(X_val,y_val)` (R²/accuracy, `_stochastic_gradient.py:79`, `fit_binary`'s `classes=[-1,1]`+`y_i` callback at `:451-454`). `early_stopping=false` leaves the training-loss `fn convergence_tail` path byte-identical (the 25 prior divergence tests stay green). Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `impl PipelineEstimator`. VERIFIED DETERMINISTICALLY: `fn r2_score`/`binary_accuracy` against the live `sklearn.metrics.r2_score`/`accuracy_score` oracle in `mod tests` (`test_validation_r2_matches_sklearn` -> `0.8887362637362637`, `test_validation_r2_constant_y_edge_cases` -> `1.0`/`0.0`, `test_validation_binary_accuracy_matches_sklearn` -> `0.75`); the `(0,1)` constraint (divergence `sgd_validation_fraction_invalid`); the behavioral early-stop (divergence `sgd_early_stopping_stops_early` — `early_stopping=true` yields a finite model DIFFERENT from `early_stopping=false`; `sgd_early_stopping_classifier_valid`). NOT VERIFIABLE (honest): the validation-subset SELECTION uses numpy Mersenne-Twister (`ShuffleSplit`/`StratifiedShuffleSplit`) whereas ferrolearn uses `StdRng`, so the exact held-out indices — and hence the full fitted `coef_` under early stopping — are NOT cross-impl reproducible (`_stochastic_gradient.py:284-287`, the SAME PRNG barrier as `shuffle`, `_sgd_fast.pyx.tp:579-580`). `_sgd_fast.pyx.tp:678-689` / `_stochastic_gradient.py:63-79,257-310,524-525`. Closes #533. NOTE: early stopping on `partial_fit` is OFF (sklearn raises `early_stopping should be False with partial_fit`, `:147-148`); the kernel receives `val_set=None` there. |
//! | REQ-14 (average / ASGD) | SHIPPED | `pub average: usize` field on `SGDClassifier`/`SGDRegressor` (default `0` = OFF) + `fn with_average` builders (sklearn `average=True`≡`1`, `average=N`≡`N`, `average=False`≡`0`; `_stochastic_gradient.py:1256,2068`), threaded through `SGDHyper.average` + `fn clf_hyper`/`reg_hyper` (one-class path hardwires `0`). `fn train_binary_sgd`/`train_regressor_sgd` allocate `average_coef`/`average_intercept` before the epoch loop and, AFTER the weight/intercept update + L1 truncation, when `hyper.average > 0 && t >= hyper.average`, accumulate the DIRECT running mean `avg += (current - avg) / (t - average + 1)` — the plain-array equivalent of sklearn's lazy `w.add_average(..., t - average + 1)` / `average_intercept += (intercept - average_intercept) / (t - average + 1)` (`_sgd_fast.pyx.tp:646-654`); the accumulator is passive (does NOT alter the live trajectory). FINALIZE: at fit-end, `if hyper.average > 0 && hyper.average <= t { weights = average_coef; intercept = average_intercept; }`, mirroring `if self.average > 0: if self.average <= self.t_ - 1: coef_ = average_coef` (`_stochastic_gradient.py:834-836`); `t` (= `n_iter_ * n_samples`, `initial_t=0`) equals sklearn's `self.t_ - 1` (sklearn inits `self.t_ = 1`). `average=0` skips both blocks, leaving the trajectory byte-identical (the 22 prior divergence tests stay green). Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `PipelineEstimator`. Tests: divergence `sgd_average_from_start` (`SGDRegressor(average=True)` live oracle coef `[0.42614902504529534, 0.3665230497098742]` intercept `0.14648807826338486`), `sgd_average_threshold` (`SGDRegressor(average=20)`, begins mid-run, oracle coef `[0.5042444287230554, 0.41888001003992603]` intercept `0.16902090306985734`), `sgd_average_classifier` (`SGDClassifier(average=True)` oracle coef `[0.11902998815794437, 0.060826180676538694]` intercept `-0.10666666666666665`). `_sgd_fast.pyx.tp:646-654` / `_stochastic_gradient.py:834-836`. Closes #534. NOTE: averaging on the `partial_fit` path is OFF (`average` not yet carried into the `partial_fit_ova` hyper, which sets `max_iter=1`) — the full `fit` path (the parity-critical one) is exact; partial_fit ASGD state carry-over is a follow-up. |
//! | REQ-15 (class_weight + sample_weight) | SHIPPED | `pub enum ClassWeight<F> {None,Balanced,Explicit(Vec<(usize,F)>)}` + `pub class_weight` field on `SGDClassifier` (default `ClassWeight::None`) with `fn with_class_weight`; `fn compute_class_weight` returns the expanded per-class weights (`None->1.0`; `Balanced-> n_samples/(n_classes*count_c)`; `Explicit->1.0 default, override by label`) faithful to `sklearn.utils.compute_class_weight` (`sklearn/utils/class_weight.py:63-81`, `_stochastic_gradient.py:624`). `fn fit_with_sample_weight` on `SGDClassifier` AND `SGDRegressor` validates `sample_weight.len()==n_samples` (else `ShapeMismatch`); `Fit::fit` delegates with `ones(n)` (byte-identical default path — the 17 prior divergence tests stay green). `fn fit_ova` builds the per-subproblem per-sample weight `w_i = class_weight_for_sample(i) * sample_weight[i]` with the sklearn OvA mapping (binary `pos=expanded[1]`/`neg=expanded[0]`, `_stochastic_gradient.py:765-766`; multiclass class k `pos=expanded[k]`/`neg=1.0`, `:816`) and passes `&[F]` into `fn train_binary_sgd`. The kernel scales ONLY the gradient term `g = grad * sample_w[i]` (`update *= class_weight*sample_weight`, `_sgd_fast.pyx.tp:630`): the weight data term `w[j]*shrink - eta*g*x[j]` and the (gated) intercept gradient term `-eta*g` use `g`; the L2 shrink (`:632-635`), L1 truncation (`:656-658`), one-class `-2*eta*alpha` offset (`:642`) and the unweighted `sumloss` (`:597`) are UNSCALED. `fn train_regressor_sgd` mirrors the same scaling (`class_weight=1` for regression). Consumer: `Fit for SGDClassifier`/`SGDRegressor` -> `PipelineEstimator`; `fit_with_sample_weight` consumed by `Fit::fit`. Tests: divergence `sgd_class_weight_balanced` (oracle coef `[0.4806667587635881, 0.4620316761984426]` intercept `-1.2811684177087947`), `sgd_class_weight_explicit` (`{0:1.0,1:3.0}` coef `[0.5705300651778317, 0.5660417632427646]` intercept `-1.7542279278451731`), `sgd_sample_weight` (coef `[0.25648548424261425, 0.7995046753090618]` intercept `-1.221373410658307`), `sgd_class_weight_balanced_multiclass` (class-0 coef `[-0.586000112348521, -0.369263665877338]` + argmax preds), `sgd_regressor_sample_weight` (coef `[0.9425558668838198, 1.3974216923953962]` intercept `0.7259434415390171`). `_sgd_fast.pyx.tp:599-602,630` / `_stochastic_gradient.py:624,765-766,816`. Closes #535. NOTE: `class_weight`/`sample_weight` on the `partial_fit` path are uniform `1.0` (no `class_weight`/`sample_weight` arg on `PartialFit` yet) — tracked under the partial_fit surface, not this REQ. |
//! | REQ-16 (partial_fit semantics) | SHIPPED | `fn partial_fit (PartialFit for SGDClassifier/FittedSGDClassifier/SGDRegressor/FittedSGDRegressor)` sets `max_iter=1` and carries `self.t` (`_stochastic_gradient.py:581-674`). Consumer: `PartialFit` trait (`ferrolearn-core`). Tests: `test_sgd_*_partial_fit*`. |
//! | REQ-17 (multiclass one-vs-all) | SHIPPED | `fn fit_ova` (one binary per class) + `fn predict` argmax (`_stochastic_gradient.py:788-844`). Consumer: `Fit for SGDClassifier` -> `PipelineEstimator`. Test: `test_sgd_classifier_multiclass`. |
//! | REQ-18 (SGDOneClassSVM) | SHIPPED | `pub struct SGDOneClassSVM<F>` (`nu`/`fit_intercept`/`max_iter`/`tol`/`shuffle`/`learning_rate`/`eta0`/`power_t`/`random_state`/`n_iter_no_change` + `new`/`#[must_use]` builders, defaults `_stochastic_gradient.py:2245-2281`) with `fn fit_one_class` + `impl Fit<Array2<F>, ()> for SGDOneClassSVM` (X-only fit, `y` ignored, `_stochastic_gradient.py:2554`): builds `y = ones(n)`, `alpha = nu/2` (`:2588`), `penalty = L2`, `l1_ratio = 0`, `one_class = true` (`:2262-2289,2312`), inits the SGD intercept `b = 1` (offset init 0 -> `1 - 0`, `:2238,2325`), calls the reused `fn train_binary_sgd` Hinge kernel, then stores `coef_ = w`, `offset_ = 1 - b` (`:2377`). The one-class intercept term lives in `fn train_binary_sgd`: when `hyper.one_class` the gated intercept update gains `- 2*eta*alpha` (`intercept_update = -eta*grad - 2*eta*alpha`), mirroring `_sgd_fast.pyx.tp:641-642` (`if one_class: intercept_update -= 2.*eta*alpha`); `pub one_class: bool` was added to `SGDHyper` (default `false` via `fn clf_hyper`/`reg_hyper`, leaving the clf/reg intercept update byte-identical — the existing 15 divergence tests stay green). `pub struct FittedSGDOneClassSVM<F>` exposes `coef()`/`offset()`/`decision_function()` (`X·coef_ - offset_`, `:2622`)/`score_samples()` (`+ offset_ = X·coef_`, `:2639`) and `impl Predict<Array2<F>>` returning `Array1<isize>` of `+1`/`-1` (`(decision >= 0) ? +1 : -1`, `:2655-2657`). Consumer: `pub use sgd::{SGDOneClassSVM, FittedSGDOneClassSVM}` from `ferrolearn-linear/src/lib.rs` (the grandfathered public-API boundary, matching `SGDClassifier`/`SGDRegressor`). Tests: divergence `sgd_one_class_svm_decision` (live oracle nu=0.5/eta0=0.05/constant/max_iter=10/shuffle=false: coef `[0.009883660184666337, 0.009883660184666337]`, offset `1.1102230246251565e-16`, 1e-7) and `sgd_one_class_svm_predict` (nu=0.8/eta0=0.1/max_iter=15: coef `[0.20020636453962284, 0.12292535592963398]`, offset `0.10000000000000009`, predict `[1,-1,1,-1]`). `_stochastic_gradient.py:2084-2668` / `_sgd_fast.pyx.tp:639-644`. Closes #536. |
//! | REQ-19 (anti-pattern cleanup) | SHIPPED | `fn compute_lr`'s `_Phantom` arm returns `eta0` (the `unreachable!()` macro was removed earlier), and every production `F::from(<f64 literal>).unwrap()` / `F::from(<literal>).unwrap_or_else(|| ...)` constant-construction site is now `fn cst<F: Float>(x: f64) -> F { F::from(x).unwrap_or_else(F::zero) }` (a private module-level infallible-for-f32/f64 constant helper, defined after the imports). 23 call sites replaced: LogLoss `18.0`/`-18.0`/`1e18` (`_sgd_fast.pyx.tp:267-283`), SquaredError/Huber `0.5` (`:291-295,315-331`), ModifiedHuber `4.0`/`-2.0` (`:178-194`), SquaredHinge/SquaredEpsilonInsensitive/intercept/one-class `2.0` (`:254-258,379-387,641-642,2588`), and the `SGDClassifier`/`SGDRegressor`/`SGDOneClassSVM` `::new` defaults (`0.0`/`0.0001`/`0.15`/`1e-3`/`0.5`/`0.25`/`0.01`/`0.01`/`0.5`, `_stochastic_gradient.py:1242-1256,2042-2068,2245-2281`). No numeric literal changed -> byte-identical for f32/f64; all 25 `divergence_sgd_fit` + full lib/doctest suites stay green. No production panicking constant-conversion remains outside `#[cfg(test)]` in `sgd.rs` (verified by grep). Per R-APG-1 / R-CODE-2. The runtime `F::from(<usize>)` conversions (`t`, `n_samples`, `num_iter`, `count`, `from_usize`) and the deliberately-non-zero-fallback constants (`max_dloss` `1e12`->`F::max_value`, `eta_floor` `1e-6`, `divisor` `5.0`) already used `unwrap_or_else` and were already gate-compliant. Closes #537. |
//! | REQ-20 (ferray substrate migration) | NOT-STARTED | blocker #538. Still `ndarray` + `StdRng` (R-SUBSTRATE-1). |
//! | REQ-21 (non-finite input rejected) | SHIPPED | All three SGD fit entries reject any NaN/+/-inf in their float inputs BEFORE the SGD kernel with `FerroError::InvalidParameter`, mirroring sklearn's `_validate_data(force_all_finite=True)` (`_stochastic_gradient.py:1476` clf/reg base, `:2392` one-class) + `_check_sample_weight` (`:1501`) → `ValueError("Input X contains NaN.")` / `"Input y contains NaN."` / `"... contains infinity ..."`. `SGDClassifier::fit_with_sample_weight` checks X + `sample_weight` (`y: Array1<usize>` finite by type); `SGDRegressor::fit_with_sample_weight` checks X + y (`Array1<F>`) + `sample_weight`; the SEPARATE `SGDOneClassSVM::fit_one_class` arm (X-only fit, no y/sample_weight) checks X. `Fit::fit` delegates to the `fit_with_sample_weight` entries with unit weights, so the guard covers the default path too. `.iter().any(|v| !v.is_finite())` catches NaN and Inf; finite paths byte-identical. Verified vs the live sklearn 1.5.2 oracle (R-CHAR-3): NaN/+inf/-inf in X for SGDClassifier/SGDRegressor, NaN/inf in y + sample_weight for SGDRegressor, NaN in sample_weight for SGDClassifier all raise `ValueError` (`tests/divergence_linear_nonfinite_batch4.rs::sgd_*`). Non-test consumer: the existing `Fit for SGDClassifier`/`SGDRegressor` + `pub use sgd::{...}` boundary. (#2263) |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, PartialFit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::SeedableRng;
use rand::seq::SliceRandom;

/// Convert an `f64` literal constant to `F`. The conversion is infallible for
/// the supported real types (`f32`/`f64`); the `F::zero()` fallback is
/// unreachable for those and exists only to keep the call non-panicking
/// (no `.unwrap()` in production, per the anti-pattern gate R-APG-1 / R-CODE-2).
#[inline]
fn cst<F: Float>(x: f64) -> F {
    F::from(x).unwrap_or_else(F::zero)
}

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
            -cst::<F>(2.0) * y_true * z
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
        if z > cst(18.0) {
            (-z).exp()
        } else if z < cst(-18.0) {
            -z
        } else {
            (F::one() + (-z).exp()).ln()
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let z = y_true * y_pred;
        let exp_nz = if z > cst(18.0) {
            (-z).exp()
        } else if z < cst(-18.0) {
            cst(1e18)
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
        cst::<F>(0.5) * diff * diff
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
            -cst::<F>(4.0) * z
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let z = y_true * y_pred;
        if z >= -F::one() {
            if z < F::one() {
                cst::<F>(-2.0) * y_true * (F::one() - z)
            } else {
                F::zero()
            }
        } else {
            -cst::<F>(4.0) * y_true
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
            cst::<F>(0.5) * diff * diff
        } else {
            self.epsilon * (abs_diff - cst::<F>(0.5) * self.epsilon)
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
        let two = cst::<F>(2.0);
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
// Class weights
// ---------------------------------------------------------------------------

/// Per-class weighting strategy for [`SGDClassifier`].
///
/// Mirrors sklearn's `class_weight` parameter
/// (`_stochastic_gradient.py` constraint `[dict, "balanced", None]`); the
/// expanded per-class weights are computed by [`compute_class_weight`] following
/// `sklearn.utils.compute_class_weight` semantics and fed into the per-sample
/// `update *= class_weight * sample_weight` scaling (`_sgd_fast.pyx.tp:630`).
#[derive(Debug, Clone)]
pub enum ClassWeight<F> {
    /// Uniform weights (all classes weighted `1.0`). The default.
    None,
    /// Balanced weights `n_samples / (n_classes * count_c)` per class `c`,
    /// matching `sklearn.utils.compute_class_weight("balanced", ...)`
    /// (`class_weight.py:73`).
    Balanced,
    /// Explicit class-label -> weight map. Classes absent from the map default
    /// to `1.0`, matching the dict branch of `compute_class_weight`
    /// (`class_weight.py:77-81`).
    Explicit(Vec<(usize, F)>),
}

/// Compute the expanded per-class weight vector aligned to `classes`
/// (sorted ascending, matching sklearn's `classes_`).
///
/// Faithful to `sklearn.utils.compute_class_weight`
/// (`sklearn/utils/class_weight.py:63-81`):
/// - `None` -> all `1.0` (`:63-65`).
/// - `Balanced` -> `n_samples / (n_classes * count_c)` per class `c`,
///   where `count_c` is the number of samples with label `c` (`:66-74`).
/// - `Explicit(map)` -> `1.0` default, overridden by the map entries matched by
///   class label (`:75-81`).
///
/// `classes` is the sorted unique label set; `y` is the per-sample label array.
fn compute_class_weight<F: Float>(cw: &ClassWeight<F>, classes: &[usize], y: &[usize]) -> Vec<F> {
    match cw {
        ClassWeight::None => vec![F::one(); classes.len()],
        ClassWeight::Balanced => {
            // `recip_freq = len(y) / (n_classes * bincount(y_ind))`
            // (`class_weight.py:73`), indexed per class.
            let n_samples = F::from(y.len()).unwrap_or_else(F::zero);
            let n_classes = F::from(classes.len()).unwrap_or_else(F::one);
            classes
                .iter()
                .map(|&c| {
                    let count = y.iter().filter(|&&label| label == c).count();
                    let count_f = F::from(count).unwrap_or_else(F::one);
                    if count_f > F::zero() {
                        n_samples / (n_classes * count_f)
                    } else {
                        F::one()
                    }
                })
                .collect()
        }
        ClassWeight::Explicit(map) => classes
            .iter()
            .map(|&c| {
                map.iter()
                    .find(|(label, _)| *label == c)
                    .map_or_else(F::one, |(_, w)| *w)
            })
            .collect(),
    }
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

/// Coefficient of determination `R^2 = 1 - SS_res / SS_tot` of a linear model
/// `(weights, intercept)` on `(x_val, y_val)`.
///
/// Mirrors `sklearn.metrics.r2_score` for the dense single-output, uniformly
/// weighted case (`RegressorMixin.score` -> `r2_score`, the regressor
/// `_ValidationScoreCallback.__call__`, `_stochastic_gradient.py:79`):
/// `SS_res = sum((y - y_pred)^2)`, `SS_tot = sum((y - mean(y))^2)`,
/// `R^2 = 1 - SS_res/SS_tot`. The degenerate `SS_tot == 0` (constant `y_val`)
/// case follows sklearn's `_metrics/_regression.py` convention: a perfect
/// `SS_res == 0` scores `1.0`, otherwise `0.0`
/// (`r2_score` `nonzero_denominator`/`nonzero_numerator` branch). Returns `0.0`
/// for an empty validation set (no information).
#[must_use]
fn r2_score<F: Float>(
    weights: &Array1<F>,
    intercept: F,
    x_val: &Array2<F>,
    y_val: &Array1<F>,
) -> F {
    let n = y_val.len();
    if n == 0 {
        return F::zero();
    }
    let n_f = F::from(n).unwrap_or_else(F::one);
    let mean = y_val.iter().fold(F::zero(), |acc, &v| acc + v) / n_f;
    let mut ss_res = F::zero();
    let mut ss_tot = F::zero();
    let n_features = weights.len();
    for i in 0..n {
        let xi = x_val.row(i);
        let mut pred = intercept;
        for j in 0..n_features {
            pred = pred + weights[j] * xi[j];
        }
        let res = y_val[i] - pred;
        ss_res = ss_res + res * res;
        let dev = y_val[i] - mean;
        ss_tot = ss_tot + dev * dev;
    }
    if ss_tot > F::zero() {
        F::one() - ss_res / ss_tot
    } else if ss_res > F::zero() {
        // constant `y_val` but imperfect prediction -> R^2 = 0 (sklearn).
        F::zero()
    } else {
        // constant `y_val` and perfect prediction -> R^2 = 1 (sklearn).
        F::one()
    }
}

/// Binary classification accuracy of a linear decision on `(x_val, y_val)` where
/// `y_val` is the relabeled `{-1, +1}` target.
///
/// Mirrors the classifier `_ValidationScoreCallback.__call__`
/// (`_stochastic_gradient.py:79`, `est.score(X_val, y_val)` ->
/// `ClassifierMixin.score` -> `accuracy_score`) for one One-vs-All binary
/// subproblem: the callback is built with `classes = np.array([-1, 1])` and the
/// relabeled binary target `y_i` (`fit_binary`, `:451-454`), so the score is the
/// fraction of validation samples whose binary decision `sign(w·x + b)` matches
/// the relabeled label. The decision uses the `>= 0 -> +1` tie convention
/// matching [`FittedSGDClassifier::predict`] (`scores[i] >= 0 -> classes[1]`).
/// Returns `0.0` for an empty validation set.
#[must_use]
fn binary_accuracy<F: Float>(
    weights: &Array1<F>,
    intercept: F,
    x_val: &Array2<F>,
    y_val: &Array1<F>,
) -> F {
    let n = y_val.len();
    if n == 0 {
        return F::zero();
    }
    let n_features = weights.len();
    let mut correct = 0usize;
    for i in 0..n {
        let xi = x_val.row(i);
        let mut decision = intercept;
        for j in 0..n_features {
            decision = decision + weights[j] * xi[j];
        }
        let pred = if decision >= F::zero() {
            F::one()
        } else {
            -F::one()
        };
        // `y_val[i]` is the relabeled `{-1, +1}` target; a positive product
        // means `pred` and the label agree.
        if pred * y_val[i] > F::zero() {
            correct += 1;
        }
    }
    F::from(correct).unwrap_or_else(F::zero) / F::from(n).unwrap_or_else(F::one)
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

/// Score-based SGD epoch-end convergence / adaptive-eta tail (early stopping).
///
/// Mirrors the `if early_stopping:` branch of `_sgd_fast.pyx.tp:678-707`. The
/// validation `score` (R^2 for the regressor, binary accuracy for the
/// classifier subproblem) replaces the training loss, and — because a HIGHER
/// score is better — `best_score` is initialized to `-inf` and the criterion
/// flips sense relative to [`convergence_tail`]: `no_improvement_count`
/// increments when `score < best_score + tol` (the epoch fails to beat the best
/// score so far by at least `tol`, `:682-685`) and resets otherwise; `best_score`
/// tracks the running MAXIMUM (`:686-687`). The shared
/// `no_improvement_count >= n_iter_no_change` tail (`:698-707`, adaptive eta/=5
/// or break) is identical to the loss path.
///
/// Returns `true` iff the epoch loop should `break` (convergence).
#[allow(clippy::too_many_arguments, reason = "mirrors the upstream epoch tail")]
#[inline]
fn convergence_tail_score<F: Float>(
    score: F,
    best_score: &mut F,
    no_improve_count: &mut usize,
    current_eta: &mut F,
    tol_active: bool,
    tol: F,
    n_iter_no_change: usize,
    adaptive: bool,
) -> bool {
    // `_sgd_fast.pyx.tp:682-685`: validation-score branch (early_stopping=True).
    if tol_active && score < *best_score + tol {
        *no_improve_count += 1;
    } else {
        *no_improve_count = 0;
    }
    // `:686-687`: track the running maximum.
    if score > *best_score {
        *best_score = score;
    }
    // `:698-707`: convergence break OR adaptive eta/=5 (shared with the loss
    // path — byte-identical logic).
    if *no_improve_count >= n_iter_no_change {
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
    /// Per-class weighting strategy. Defaults to [`ClassWeight::None`] (uniform).
    /// Mirrors sklearn's `class_weight` parameter (default `None`); the expanded
    /// weights scale the per-sample gradient term via
    /// `update *= class_weight * sample_weight` (`_sgd_fast.pyx.tp:599-602,630`).
    pub class_weight: ClassWeight<F>,
    /// Averaged-SGD (ASGD) threshold. `0` disables averaging (the default,
    /// matching sklearn `average=False`); `1` averages from the first step
    /// (sklearn `average=True`); `N > 1` begins averaging once the global step
    /// counter `t >= N` (sklearn `average=N`). The averaged weights/intercept
    /// replace the plain ones at fit-end when `average <= self.t_ - 1`
    /// (`_sgd_fast.pyx.tp:646-654`, `_stochastic_gradient.py:834-836`).
    pub average: usize,
    /// Whether to stop training early based on a held-out validation score.
    /// Defaults to `false` (sklearn `SGDClassifier(early_stopping=False)`,
    /// `_stochastic_gradient.py:114`, constraint `["boolean"]` at `:524`). When
    /// `true`, [`validation_fraction`](Self::validation_fraction) of the training
    /// data is held out (stratified) as a validation set and the epoch-end
    /// convergence rule uses the validation accuracy of each One-vs-All binary
    /// subproblem instead of the training loss (`_sgd_fast.pyx.tp:678-687`).
    pub early_stopping: bool,
    /// Fraction of the training data held out as the validation set when
    /// [`early_stopping`](Self::early_stopping) is `true`. Defaults to `0.1`
    /// (sklearn `validation_fraction=0.1`, `_stochastic_gradient.py:115`). Must
    /// lie in the open interval `(0, 1)`
    /// (constraint `Interval(Real, 0, 1, closed="neither")` at `:525`).
    pub validation_fraction: F,
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
            eta0: cst(0.0),
            alpha: cst(0.0001),
            penalty: Penalty::L2,
            l1_ratio: cst(0.15),
            max_iter: 1000,
            tol: cst(1e-3),
            random_state: None,
            power_t: cst(0.5),
            shuffle: true,
            n_iter_no_change: 5,
            fit_intercept: true,
            class_weight: ClassWeight::None,
            average: 0,
            early_stopping: false,
            validation_fraction: cst(0.1),
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

    /// Set the per-class weighting strategy.
    ///
    /// Mirrors sklearn's `class_weight` parameter (default `None`). The expanded
    /// per-class weights (via [`compute_class_weight`]) scale only the
    /// gradient-derived part of each per-sample update
    /// (`_sgd_fast.pyx.tp:599-602,630`); the L2 shrink, L1 truncation and the
    /// one-class offset are left unscaled.
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight<F>) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Set the averaged-SGD (ASGD) threshold.
    ///
    /// Mirrors sklearn's `average` parameter (default `False`,
    /// `_stochastic_gradient.py:1256`). `0` disables averaging (the plain SGD
    /// trajectory, byte-identical to the unaveraged kernel); `with_average(1)`
    /// is sklearn `average=True` (average from the first step); `with_average(N)`
    /// is sklearn `average=N` (begin averaging once the global step counter
    /// `t >= N`). The running mean of the post-update weights/intercept replaces
    /// the plain `coef_`/`intercept_` at fit-end when `average <= self.t_ - 1`
    /// (`_sgd_fast.pyx.tp:646-654`, `_stochastic_gradient.py:834-836`).
    #[must_use]
    pub fn with_average(mut self, average: usize) -> Self {
        self.average = average;
        self
    }

    /// Enable or disable early stopping on a held-out validation score.
    ///
    /// Mirrors sklearn's `early_stopping` parameter (default `False`,
    /// `_stochastic_gradient.py:114`, constraint `["boolean"]` at `:524`). When
    /// enabled, [`with_validation_fraction`](Self::with_validation_fraction) of
    /// the data is held out (stratified per class) and each One-vs-All binary
    /// subproblem's epoch-end convergence is driven by its validation accuracy
    /// rather than the training loss (`_sgd_fast.pyx.tp:678-687`).
    #[must_use]
    pub fn with_early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }

    /// Set the fraction of the training data held out for early-stopping
    /// validation.
    ///
    /// Mirrors sklearn's `validation_fraction` parameter (default `0.1`,
    /// `_stochastic_gradient.py:115`, constraint
    /// `Interval(Real, 0, 1, closed="neither")` at `:525`). Only used when
    /// [`early_stopping`](Self::early_stopping) is `true`; validated to the open
    /// interval `(0, 1)` at fit time.
    #[must_use]
    pub fn with_validation_fraction(mut self, validation_fraction: F) -> Self {
        self.validation_fraction = validation_fraction;
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
        one_class: false,
        average: clf.average,
        early_stopping: clf.early_stopping,
        validation_fraction: clf.validation_fraction,
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
    /// Whether this is a one-class SVM fit. When `true` the (gated) intercept
    /// update gains the extra `- 2*eta*alpha` term, mirroring the `if one_class`
    /// branch in `_sgd_fast.pyx.tp:641-642`
    /// (`intercept_update -= 2. * eta * alpha`). `false` for the standard
    /// classifier/regressor paths, leaving their intercept update byte-identical.
    one_class: bool,
    /// Averaged-SGD (ASGD) threshold. `0` disables averaging (the default,
    /// byte-identical to the plain SGD trajectory). `N > 0` begins accumulating
    /// the running mean of the post-update weights/intercept once the global step
    /// counter `t >= N`, mirroring `if 0 < average <= t` (`_sgd_fast.pyx.tp:646`).
    /// sklearn `average=True` maps to `N = 1`; `average=N` maps to `N`.
    average: usize,
    /// Whether to use early stopping on a held-out validation score. When
    /// `true` the epoch-end convergence rule scores the current weights on the
    /// validation set (R^2 / accuracy) instead of the training loss
    /// (`_sgd_fast.pyx.tp:678-687`, `_stochastic_gradient.py:114`). The
    /// validation set itself is split off in the `Fit` path BEFORE the kernel
    /// and passed in separately; this flag only selects the score-based
    /// epoch-end branch. `false` (the default) leaves the training-loss
    /// convergence path byte-identical.
    early_stopping: bool,
    /// Fraction of the training data held out as the validation set when
    /// `early_stopping` is `true` (`_stochastic_gradient.py:115`, default `0.1`,
    /// constraint `Interval(Real, 0, 1, closed="neither")` at `:525`). Carried
    /// for validation/documentation; the actual split happens in the `Fit` path.
    validation_fraction: F,
}

/// Train a single binary classifier via SGD, updating `weights` and
/// `intercept` in place. `y_binary` must be in `{-1, +1}`.
///
/// `sample_w[i]` is the per-sample weight `class_weight_i * sample_weight_i`
/// for sample `i` (`_sgd_fast.pyx.tp:599-602,630`). It scales ONLY the
/// gradient-derived part of the update (`update *= class_weight * sample_weight`
/// at `:630`); the L2 shrink (`:632-635`), the L1 truncation (`:656-658`) and
/// the one-class `-2*eta*alpha` offset (`:642`) are left unscaled. An all-ones
/// `sample_w` (the default `fit` path) reproduces the byte-identical unweighted
/// behaviour. `sample_w.len()` must equal `x.nrows()`.
///
/// Returns the cumulative loss and the step counter after training.
#[allow(
    clippy::too_many_arguments,
    reason = "threads the per-sample weight vector"
)]
fn train_binary_sgd<F, L>(
    x: &Array2<F>,
    y_binary: &Array1<F>,
    weights: &mut Array1<F>,
    intercept: &mut F,
    loss_fn: &L,
    hyper: &SGDHyper<F>,
    initial_t: usize,
    sample_w: &[F],
    val_set: Option<(&Array2<F>, &Array1<F>)>,
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
    // the adaptive-schedule eta (`eta = eta / 5` decay, `:700`). When early
    // stopping is active `best_score = -INFINITY` instead (higher score is
    // better — `_sgd_fast.pyx.tp:533`).
    let mut best_loss = F::infinity();
    let mut best_score = F::neg_infinity();
    // Early stopping uses the score branch only when a validation set was split
    // off in the `Fit` path (`val_set.is_some()`). The relabeled `{-1,+1}`
    // binary validation target is scored with accuracy (`binary_accuracy`,
    // `_stochastic_gradient.py:451-454,79`).
    let early_stopping = hyper.early_stopping && val_set.is_some();
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

    // Averaged-SGD (ASGD) accumulators (`_sgd_fast.pyx.tp:646-654`). When
    // `hyper.average > 0`, once the global step `t >= average` we maintain the
    // running mean of the POST-update weights/intercept. This is the DIRECT
    // running-mean form of sklearn's lazy `w.add_average` (a wscale optimization
    // that is mathematically identical for plain arrays): with
    // `num_iter = t - average + 1` (= 1 at the first averaged step) and
    // `mu = 1/num_iter`, `avg += (current - avg) * mu`. The accumulator is a
    // PASSIVE observer — it never feeds back into the live `weights`/`intercept`.
    let mut average_coef: Array1<F> = Array1::zeros(n_features);
    let mut average_intercept = F::zero();

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
            // Per-sample weight scaling: `update *= class_weight * sample_weight`
            // (`_sgd_fast.pyx.tp:630`). `update = -eta*dloss`, so scaling the
            // update by `w_i` is equivalent to scaling the (clipped) gradient
            // `dloss` by `w_i` BEFORE forming both the weight data term and the
            // (gated) intercept gradient term. This multiplies ONLY the
            // gradient-derived part; the L2 shrink, L1 truncation and the
            // one-class offset below are unaffected. `g` is the scaled gradient.
            let g = grad * sample_w[i];
            // `sumloss` is the SUM (not mean) of per-sample losses over the
            // epoch (`_sgd_fast.pyx.tp:597`), computed from the UNWEIGHTED loss
            // (the weight only multiplies `update`, not `loss.loss(y, p)`).
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
            // Gradient add `w.add(x, update)` with the scaled `update = -eta*g`
            // (`_sgd_fast.pyx.tp:637-638`); `g` is the sample-weighted gradient.
            for j in 0..n_features {
                weights[j] = weights[j] - eta * g * xi[j];
            }
            // The intercept update is gated on `fit_intercept` and is NOT
            // regularized (`intercept_decay=1`, `_sgd_fast.pyx.tp:639-644`:
            // `if fit_intercept == 1: intercept_update = update; if one_class:
            // intercept_update -= 2.*eta*alpha; intercept += intercept_update *
            // intercept_decay`). `update = -eta*g` is the SCALED update (sklearn
            // sets `intercept_update = update` at `:640`, after `update *=
            // class_weight*sample_weight` at `:630`), so the standard path is
            // `intercept -= eta*g`. For the one-class SVM the extra
            // `- 2*eta*alpha` term is added (`:641-642`) and is NOT scaled by the
            // sample weight. When `fit_intercept` is false the intercept is never
            // modified and stays at its init value (`0` clf/reg, `1` one-class).
            if hyper.fit_intercept {
                let two = cst::<F>(2.0);
                let mut intercept_update = -eta * g;
                if hyper.one_class {
                    intercept_update = intercept_update - two * eta * hyper.alpha;
                }
                *intercept = *intercept + intercept_update;
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

            // ASGD running-mean accumulation (`_sgd_fast.pyx.tp:646-654`:
            // `if 0 < average <= t: w.add_average(..., t - average + 1);
            // average_intercept += (intercept - average_intercept) /
            // (t - average + 1)`). Performed AFTER the weight/intercept update +
            // L1 truncation, so `weights`/`intercept` hold their final post-step
            // values for this sample. `t` here is the SAME 1-based global step
            // the schedule used above. `num_iter = t - average + 1` is `>= 1`
            // whenever `t >= average`.
            if hyper.average > 0 && t >= hyper.average {
                let num_iter = t - hyper.average + 1;
                let num_iter_f = F::from(num_iter).unwrap_or_else(F::one);
                let mu = F::one() / num_iter_f;
                for j in 0..n_features {
                    average_coef[j] = average_coef[j] + (weights[j] - average_coef[j]) * mu;
                }
                average_intercept = average_intercept + (*intercept - average_intercept) * mu;
            }
        }

        // `epoch_loss` is now the epoch `sumloss` (no mean division).
        total_loss = epoch_loss;

        // Epoch-end stop rule (`_sgd_fast.pyx.tp:678-707`). When early stopping
        // is active, score the CURRENT weights/intercept on the held-out
        // validation set (binary accuracy of the relabeled `{-1,+1}` target,
        // `_stochastic_gradient.py:79`) and run the score-based branch
        // (`best_score` init `-inf`, higher is better, `:678-687`); otherwise
        // the training-loss branch (`sumloss` vs `best_loss`, `:688-695`).
        let should_break = if let (true, Some((x_val, y_val))) = (early_stopping, val_set) {
            let score = binary_accuracy(weights, *intercept, x_val, y_val);
            convergence_tail_score(
                score,
                &mut best_score,
                &mut no_improve_count,
                &mut current_eta,
                tol_active,
                hyper.tol,
                hyper.n_iter_no_change,
                matches!(hyper.learning_rate, LearningRateSchedule::Adaptive),
            )
        } else {
            convergence_tail(
                epoch_loss,
                &mut best_loss,
                &mut no_improve_count,
                &mut current_eta,
                tol_active,
                hyper.tol,
                n_samples,
                hyper.n_iter_no_change,
                matches!(hyper.learning_rate, LearningRateSchedule::Adaptive),
            )
        };
        if should_break {
            break;
        }
    }

    // ASGD finalize: select the averaged weights/intercept when averaging was
    // enabled AND enough steps were taken (`_stochastic_gradient.py:834-836`:
    // `if self.average > 0: if self.average <= self.t_ - 1: coef_ =
    // average_coef`). Here `t` is the returned step counter `= n_iter_ *
    // n_samples` (`initial_t = 0` on the full-fit path), which equals sklearn's
    // `self.t_ - 1` (sklearn inits `self.t_ = 1`, then `self.t_ += n_iter_ *
    // n_samples`). So `average <= self.t_ - 1` maps to `hyper.average <= t`.
    if hyper.average > 0 && hyper.average <= t {
        for j in 0..n_features {
            weights[j] = average_coef[j];
        }
        *intercept = average_intercept;
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
        // Delegate to the sample-weighted path with a uniform `ones(n)` weight
        // vector, so the default `fit` behaviour is byte-identical to the
        // unweighted kernel (`_check_sample_weight` returns `ones` when
        // `sample_weight=None`, `_stochastic_gradient.py:627`).
        let sample_weight = Array1::<F>::from_elem(x.nrows(), F::one());
        self.fit_with_sample_weight(x, y, &sample_weight)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> SGDClassifier<F> {
    /// Fit the SGD classifier with explicit per-sample weights.
    ///
    /// Mirrors `SGDClassifier.fit(X, y, sample_weight=...)`. The per-sample
    /// weight scales ONLY the gradient-derived part of each update
    /// (`update *= class_weight * sample_weight`, `_sgd_fast.pyx.tp:630`); the
    /// L2 shrink, L1 truncation and one-class offset are unscaled. The
    /// `class_weight` field (via [`compute_class_weight`],
    /// `_stochastic_gradient.py:624`) is combined multiplicatively per sample.
    ///
    /// [`Fit::fit`] delegates here with a uniform `ones(n)` weight vector.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x`/`y`/`sample_weight` have
    /// mismatched sample counts.
    /// Returns [`FerroError::InsufficientSamples`] if fewer than 2 classes
    /// are present.
    /// Returns [`FerroError::InvalidParameter`] if `eta0` or `alpha` are
    /// invalid.
    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
        sample_weight: &Array1<F>,
    ) -> Result<FittedSGDClassifier<F>, FerroError> {
        validate_clf_params(
            x,
            y,
            &self.learning_rate,
            self.eta0,
            self.alpha,
            self.l1_ratio,
            self.validation_fraction,
        )?;

        let n_samples = x.nrows();
        if sample_weight.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![sample_weight.len()],
                context: "sample_weight length must match number of samples in X".into(),
            });
        }

        // Non-finite input validation (#2263). sklearn `SGDClassifier.fit`
        // -> `self._validate_data(X, y, ...)` (`_stochastic_gradient.py:1476`)
        // keeps the default `force_all_finite=True`, so `check_array` rejects any
        // NaN or +/-inf in X with a `ValueError("Input X contains NaN.")` /
        // `"... contains infinity ..."` BEFORE the SGD kernel. sklearn also
        // validates `sample_weight` via `_check_sample_weight` (default
        // `force_all_finite=True`, `_stochastic_gradient.py:1501`). `y` is
        // `Array1<usize>` (integer class labels), finite by type, so only X +
        // sample_weight need the runtime check. `.iter().any(|v| !v.is_finite())`
        // rejects both NaN and Inf (bounds-safe, no panic, R-CODE-2); the finite
        // path is byte-identical. This is the SGDClassifier fit entry (`Fit::fit`
        // delegates here with unit weights).
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }
        if sample_weight.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "sample_weight".into(),
                reason: "Input sample_weight contains NaN or infinity.".into(),
            });
        }

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

        // Expanded per-class weights (`_stochastic_gradient.py:624`), aligned to
        // the sorted `classes` (= sklearn `classes_`).
        let expanded = compute_class_weight(&self.class_weight, &classes, &y.to_vec());
        let sw = sample_weight.to_vec();

        let (weight_matrix, intercepts, t) = fit_ova(
            x, y, &classes, n_features, &loss_enum, &hyper, 0, &expanded, &sw,
        )?;

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
/// Validate `validation_fraction` to the OPEN interval `(0, 1)`.
///
/// Mirrors sklearn `_parameter_constraints["validation_fraction"]`
/// (`_stochastic_gradient.py:525`, `Interval(Real, 0, 1, closed="neither")`):
/// the bounds are both EXCLUSIVE, so `0.0` and `1.0` are invalid. sklearn
/// validates this unconditionally (it is a constructor constraint), independent
/// of `early_stopping`.
fn validate_validation_fraction<F: Float>(validation_fraction: F) -> Result<(), FerroError> {
    if validation_fraction <= F::zero() || validation_fraction >= F::one() {
        return Err(FerroError::InvalidParameter {
            name: "validation_fraction".into(),
            reason: "must be in the open interval (0, 1)".into(),
        });
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "threads each validated parameter"
)]
fn validate_clf_params<F: Float>(
    x: &Array2<F>,
    y: &Array1<usize>,
    schedule: &LearningRateSchedule<F>,
    eta0: F,
    alpha: F,
    l1_ratio: F,
    validation_fraction: F,
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
    validate_validation_fraction(validation_fraction)?;
    Ok(())
}

/// Result type for one-vs-all training: (weight_matrix, intercepts, step_counter).
type OvaResult<F> = (Vec<Array1<F>>, Vec<F>, usize);

/// Number of validation samples for an early-stopping split: `ceil` of
/// `validation_fraction * n` clamped to `[1, n-1]` so that BOTH the train and
/// the validation subset are non-empty.
///
/// sklearn delegates the count to `ShuffleSplit`/`StratifiedShuffleSplit`, which
/// use `ceil(test_size * n)` and raise if either subset is empty
/// (`_stochastic_gradient.py:295-307`). The exact sample SELECTION is numpy-RNG
/// coupled and not cross-impl reproducible (the same barrier as `shuffle`); only
/// the count + non-emptiness are reproduced here.
fn validation_count<F: Float>(validation_fraction: F, n: usize) -> usize {
    let n_f = F::from(n).unwrap_or_else(F::zero);
    let raw = (validation_fraction * n_f).ceil();
    let n_val = raw.to_usize().unwrap_or(1).max(1);
    n_val.min(n.saturating_sub(1))
}

/// Build a seeded, optionally stratified train/validation index partition for
/// early stopping.
///
/// Returns `(train_idx, val_idx)`. The first `n_val` entries of a seeded random
/// permutation form the validation set; for the classifier (`stratify = Some`)
/// the permutation is built per class so the validation set is proportional per
/// class, mirroring sklearn's `StratifiedShuffleSplit`
/// (`_stochastic_gradient.py:280-287`); for the regressor a plain `ShuffleSplit`
/// permutation is used. The RNG is `StdRng::seed_from_u64(random_state ?? 0)`.
///
/// The SELECTION is intentionally NOT identical to sklearn (numpy's
/// Mersenne-Twister permutation differs from `StdRng`); only the deterministic
/// contract — a valid, seeded, stratified-for-classifier, non-empty split — is
/// reproduced. Returns `None` if either subset would be empty (sklearn raises a
/// `ValueError`, `_stochastic_gradient.py:295-307`).
fn make_validation_split<F: Float>(
    n: usize,
    validation_fraction: F,
    random_state: Option<u64>,
    stratify: Option<&[usize]>,
) -> Option<(Vec<usize>, Vec<usize>)> {
    if n < 2 {
        return None;
    }
    let n_val = validation_count(validation_fraction, n);
    if n_val == 0 || n_val >= n {
        return None;
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(random_state.unwrap_or(0));

    let mut val_mask = vec![false; n];
    match stratify {
        Some(labels) => {
            // Per-class proportional hold-out (`StratifiedShuffleSplit`). For each
            // class, shuffle its member indices and take `round(frac * count)`
            // (at least 1 when the class has >= 2 members) into validation.
            let mut classes: Vec<usize> = labels.to_vec();
            classes.sort_unstable();
            classes.dedup();
            for &c in &classes {
                let mut members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
                members.shuffle(&mut rng);
                let count = members.len();
                let frac_f = F::from(count).unwrap_or_else(F::zero) * validation_fraction;
                let mut take = frac_f.round().to_usize().unwrap_or(0);
                if take == 0 && count >= 2 {
                    take = 1;
                }
                take = take.min(count.saturating_sub(1)).min(count);
                for &idx in members.iter().take(take) {
                    val_mask[idx] = true;
                }
            }
        }
        None => {
            // Plain shuffle hold-out (`ShuffleSplit`): first `n_val` of a seeded
            // permutation.
            let mut perm: Vec<usize> = (0..n).collect();
            perm.shuffle(&mut rng);
            for &idx in perm.iter().take(n_val) {
                val_mask[idx] = true;
            }
        }
    }

    let val_idx: Vec<usize> = (0..n).filter(|&i| val_mask[i]).collect();
    let train_idx: Vec<usize> = (0..n).filter(|&i| !val_mask[i]).collect();
    if val_idx.is_empty() || train_idx.is_empty() {
        return None;
    }
    Some((train_idx, val_idx))
}

/// Gather the rows of `x` indexed by `idx` into a fresh `Array2`.
fn gather_rows<F: Float>(x: &Array2<F>, idx: &[usize]) -> Array2<F> {
    let n_features = x.ncols();
    let mut out = Array2::<F>::zeros((idx.len(), n_features));
    for (r, &i) in idx.iter().enumerate() {
        let src = x.row(i);
        for j in 0..n_features {
            out[[r, j]] = src[j];
        }
    }
    out
}

/// Gather the entries of `v` indexed by `idx` into a fresh `Array1`.
fn gather<F: Float>(v: &Array1<F>, idx: &[usize]) -> Array1<F> {
    Array1::from_iter(idx.iter().map(|&i| v[i]))
}

/// Train one-vs-all binary classifiers, returning per-class weights, intercepts,
/// and the cumulative step counter.
///
/// `expanded_class_weight[k]` is the weight of `classes[k]` from
/// [`compute_class_weight`] (`_stochastic_gradient.py:624`). `sample_weight[i]`
/// is the user per-sample weight. For each binary subproblem the per-sample
/// weight passed to the kernel is `class_weight_for_sample(i) * sample_weight[i]`
/// where `class_weight_for_sample(i)` is `pos_weight` if sample `i` is the
/// positive class else `neg_weight`, with the sklearn OvA mapping:
/// binary (`_fit_binary`, `:765-766`) `pos = expanded[1]`, `neg = expanded[0]`;
/// multiclass class `k` (`_fit_multiclass`, `:816`) `pos = expanded[k]`,
/// `neg = 1.0`.
#[allow(clippy::too_many_arguments, reason = "threads class + sample weights")]
fn fit_ova<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
    classes: &[usize],
    n_features: usize,
    loss_enum: &ClassifierLoss,
    hyper: &SGDHyper<F>,
    initial_t: usize,
    expanded_class_weight: &[F],
    sample_weight: &[F],
) -> Result<OvaResult<F>, FerroError> {
    let n_classes = classes.len();
    let mut weight_matrix: Vec<Array1<F>> = Vec::with_capacity(n_classes);
    let mut intercepts: Vec<F> = Vec::with_capacity(n_classes);
    let mut global_t = initial_t;

    // Early-stopping validation split. Computed ONCE over the full multiclass
    // labels (so the hold-out is stratified per class and SHARED by every OvA
    // subproblem, exactly as sklearn precomputes the mask in `_fit_multiclass`,
    // `_stochastic_gradient.py:796`, and reuses it for each binary fit). The
    // split is stratified (`StratifiedShuffleSplit`, `:280-281`). When the split
    // is infeasible (too few samples) it returns `None`, and an empty validation
    // set raises (`:295-307`).
    let split = if hyper.early_stopping {
        match make_validation_split(
            x.nrows(),
            hyper.validation_fraction,
            hyper.random_state,
            Some(&y.to_vec()),
        ) {
            Some(s) => Some(s),
            None => {
                return Err(FerroError::InvalidParameter {
                    name: "validation_fraction".into(),
                    reason: "early_stopping split led to an empty train or validation set; \
                             increase the number of samples or change validation_fraction"
                        .into(),
                });
            }
        }
    } else {
        None
    };

    // Closure: run one OvA binary subproblem (relabel + per-sample weights +
    // optional validation slice + kernel call). `pos`/`neg` are the class-weight
    // mappings for this subproblem.
    let run_subproblem =
        |cls: usize, pos_weight: F, neg_weight: F, w: &mut Array1<F>, b: &mut F, t0: usize| {
            let y_binary: Array1<F> =
                y.mapv(|label| if label == cls { F::one() } else { -F::one() });
            // Per-sample weight = class_weight_for_sample(i) * sample_weight[i]
            // (`_sgd_fast.pyx.tp:599-602,630`). `y_binary[i] > 0` selects pos.
            let sample_w_full: Vec<F> = (0..x.nrows())
                .map(|i| {
                    let cw = if y_binary[i] > F::zero() {
                        pos_weight
                    } else {
                        neg_weight
                    };
                    cw * sample_weight[i]
                })
                .collect();

            if let Some((train_idx, val_idx)) = &split {
                // Train on the train subset only; score on the held-out
                // validation subset (relabeled `{-1,+1}` target, accuracy —
                // `_stochastic_gradient.py:451-454,79`). sklearn does NOT refit
                // on the full data; the train-subset weights are final
                // (`_ValidationScoreCallback` only reads weights, never writes
                // back to the live fit).
                let x_tr = gather_rows(x, train_idx);
                let y_tr = gather(&y_binary, train_idx);
                let sw_tr: Vec<F> = train_idx.iter().map(|&i| sample_w_full[i]).collect();
                let x_val = gather_rows(x, val_idx);
                let y_val = gather(&y_binary, val_idx);
                dispatch_train_binary(
                    &x_tr,
                    &y_tr,
                    w,
                    b,
                    loss_enum,
                    hyper,
                    t0,
                    &sw_tr,
                    Some((&x_val, &y_val)),
                )
            } else {
                dispatch_train_binary(
                    x,
                    &y_binary,
                    w,
                    b,
                    loss_enum,
                    hyper,
                    t0,
                    &sample_w_full,
                    None,
                )
            }
        };

    if n_classes == 2 {
        // Single binary problem: class[0] -> -1, class[1] -> +1.
        // OvA weight mapping (`_fit_binary`, `_stochastic_gradient.py:765-766`):
        // pos_weight = expanded[1], neg_weight = expanded[0].
        let pos_weight = expanded_class_weight[1];
        let neg_weight = expanded_class_weight[0];
        let mut w = Array1::<F>::zeros(n_features);
        let mut b = F::zero();
        let (_, t) = run_subproblem(classes[1], pos_weight, neg_weight, &mut w, &mut b, global_t);
        global_t = t;
        weight_matrix.push(w);
        intercepts.push(b);
    } else {
        // One-vs-all: one binary problem per class. Multiclass mapping
        // (`_fit_multiclass`, `_stochastic_gradient.py:816`): for class k,
        // pos_weight = expanded[k], neg_weight = 1.0.
        for (k, &cls) in classes.iter().enumerate() {
            let pos_weight = expanded_class_weight[k];
            let neg_weight = F::one();
            let mut w = Array1::<F>::zeros(n_features);
            let mut b = F::zero();
            let (_, t) = run_subproblem(cls, pos_weight, neg_weight, &mut w, &mut b, global_t);
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
    // `partial_fit` does not (yet) carry `class_weight`/`sample_weight`, so the
    // per-sample weight is uniform `1.0` — the all-ones path is byte-identical to
    // the pre-weighting kernel (`update *= 1*1`, `_sgd_fast.pyx.tp:630`).
    let sample_w: Vec<F> = vec![F::one(); x.nrows()];

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
            &sample_w,
            None,
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
                &sample_w,
                None,
            );
            global_t = t;
        }
    }

    global_t
}

/// Dispatch to the appropriate typed loss training function.
///
/// `sample_w[i] = class_weight_i * sample_weight_i` is the per-sample weight
/// (`_sgd_fast.pyx.tp:599-602,630`), forwarded verbatim to the kernel.
#[allow(
    clippy::too_many_arguments,
    reason = "threads the per-sample weight vector"
)]
fn dispatch_train_binary<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y_binary: &Array1<F>,
    w: &mut Array1<F>,
    b: &mut F,
    loss_enum: &ClassifierLoss,
    hyper: &SGDHyper<F>,
    initial_t: usize,
    sample_w: &[F],
    val_set: Option<(&Array2<F>, &Array1<F>)>,
) -> (F, usize) {
    match loss_enum {
        ClassifierLoss::Hinge => train_binary_sgd(
            x, y_binary, w, b, &Hinge, hyper, initial_t, sample_w, val_set,
        ),
        ClassifierLoss::SquaredHinge => train_binary_sgd(
            x,
            y_binary,
            w,
            b,
            &SquaredHinge,
            hyper,
            initial_t,
            sample_w,
            val_set,
        ),
        ClassifierLoss::Perceptron => train_binary_sgd(
            x,
            y_binary,
            w,
            b,
            &Perceptron,
            hyper,
            initial_t,
            sample_w,
            val_set,
        ),
        ClassifierLoss::Log => train_binary_sgd(
            x, y_binary, w, b, &LogLoss, hyper, initial_t, sample_w, val_set,
        ),
        ClassifierLoss::SquaredError => train_binary_sgd(
            x,
            y_binary,
            w,
            b,
            &SquaredError,
            hyper,
            initial_t,
            sample_w,
            val_set,
        ),
        ClassifierLoss::ModifiedHuber => train_binary_sgd(
            x,
            y_binary,
            w,
            b,
            &ModifiedHuber,
            hyper,
            initial_t,
            sample_w,
            val_set,
        ),
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
            self.validation_fraction,
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

        // Initial `partial_fit` does not carry per-sample/class weights here, so
        // the expanded class weights and sample weights are uniform `1.0` — the
        // all-ones path is byte-identical to the pre-weighting kernel.
        let expanded: Vec<F> = vec![F::one(); classes.len()];
        let sw: Vec<F> = vec![F::one(); x.nrows()];

        let (weight_matrix, intercepts, t) = fit_ova(
            x, y, &classes, n_features, &loss_enum, &hyper, 0, &expanded, &sw,
        )?;

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
    /// Averaged-SGD (ASGD) threshold. `0` disables averaging (the default,
    /// matching sklearn `average=False`); `1` averages from the first step
    /// (sklearn `average=True`); `N > 1` begins averaging once the global step
    /// counter `t >= N` (sklearn `average=N`). The averaged weights/intercept
    /// replace the plain ones at fit-end when `average <= self.t_ - 1`
    /// (`_sgd_fast.pyx.tp:646-654`, `_stochastic_gradient.py:834-836`).
    pub average: usize,
    /// Whether to stop training early based on a held-out validation score.
    /// Defaults to `false` (sklearn `SGDRegressor(early_stopping=False)`,
    /// `_stochastic_gradient.py:114`, constraint `["boolean"]` at `:524`). When
    /// `true`, [`validation_fraction`](Self::validation_fraction) of the training
    /// data is held out as a validation set and the epoch-end convergence rule
    /// uses the validation `R^2` instead of the training loss
    /// (`_sgd_fast.pyx.tp:678-687`).
    pub early_stopping: bool,
    /// Fraction of the training data held out as the validation set when
    /// [`early_stopping`](Self::early_stopping) is `true`. Defaults to `0.1`
    /// (sklearn `validation_fraction=0.1`, `_stochastic_gradient.py:115`). Must
    /// lie in the open interval `(0, 1)`
    /// (constraint `Interval(Real, 0, 1, closed="neither")` at `:525`).
    pub validation_fraction: F,
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
            average: 0,
            fit_intercept: true,
            shuffle: true,
            penalty: Penalty::L2,
            l1_ratio: cst(0.15),
            learning_rate: LearningRateSchedule::InvScaling,
            eta0: cst(0.01),
            alpha: cst(0.0001),
            max_iter: 1000,
            tol: cst(1e-3),
            random_state: None,
            power_t: cst(0.25),
            early_stopping: false,
            validation_fraction: cst(0.1),
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

    /// Set the averaged-SGD (ASGD) threshold.
    ///
    /// Mirrors sklearn's `average` parameter (default `False`,
    /// `_stochastic_gradient.py:2068`). `0` disables averaging (the plain SGD
    /// trajectory, byte-identical to the unaveraged kernel); `with_average(1)`
    /// is sklearn `average=True` (average from the first step); `with_average(N)`
    /// is sklearn `average=N` (begin averaging once the global step counter
    /// `t >= N`). The running mean of the post-update weights/intercept replaces
    /// the plain `coef_`/`intercept_` at fit-end when `average <= self.t_ - 1`
    /// (`_sgd_fast.pyx.tp:646-654`, `_stochastic_gradient.py:834-836`).
    #[must_use]
    pub fn with_average(mut self, average: usize) -> Self {
        self.average = average;
        self
    }

    /// Enable or disable early stopping on a held-out validation score.
    ///
    /// Mirrors sklearn's `early_stopping` parameter (default `False`,
    /// `_stochastic_gradient.py:114`, constraint `["boolean"]` at `:524`). When
    /// enabled, [`with_validation_fraction`](Self::with_validation_fraction) of
    /// the data is held out and the epoch-end convergence is driven by the
    /// validation `R^2` rather than the training loss
    /// (`_sgd_fast.pyx.tp:678-687`).
    #[must_use]
    pub fn with_early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }

    /// Set the fraction of the training data held out for early-stopping
    /// validation.
    ///
    /// Mirrors sklearn's `validation_fraction` parameter (default `0.1`,
    /// `_stochastic_gradient.py:115`, constraint
    /// `Interval(Real, 0, 1, closed="neither")` at `:525`). Only used when
    /// [`early_stopping`](Self::early_stopping) is `true`; validated to the open
    /// interval `(0, 1)` at fit time.
    #[must_use]
    pub fn with_validation_fraction(mut self, validation_fraction: F) -> Self {
        self.validation_fraction = validation_fraction;
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
        one_class: false,
        average: reg.average,
        early_stopping: reg.early_stopping,
        validation_fraction: reg.validation_fraction,
    }
}

/// Train a single regressor via SGD, updating `weights` and `intercept`
/// in place. Returns the final loss and step counter.
///
/// `sample_w[i]` is the per-sample weight `sample_weight[i]`, scaling ONLY the
/// gradient-derived part of each update (`update *= class_weight * sample_weight`
/// with `class_weight = 1` for regression, `_sgd_fast.pyx.tp:599-602,630`); the
/// L2 shrink and L1 truncation are unscaled. An all-ones `sample_w` (the default
/// `fit` path) reproduces the byte-identical unweighted behaviour.
#[allow(
    clippy::too_many_arguments,
    reason = "threads the per-sample weight vector"
)]
fn train_regressor_sgd<F, L>(
    x: &Array2<F>,
    y: &Array1<F>,
    weights: &mut Array1<F>,
    intercept: &mut F,
    loss_fn: &L,
    hyper: &SGDHyper<F>,
    initial_t: usize,
    sample_w: &[F],
    val_set: Option<(&Array2<F>, &Array1<F>)>,
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
    // the adaptive-schedule eta (`eta = eta / 5` decay, `:700`). Under early
    // stopping `best_score = -INFINITY` (higher score is better, `:533`).
    let mut best_loss = F::infinity();
    let mut best_score = F::neg_infinity();
    // Early stopping uses the validation-R^2 branch only when a validation set
    // was split off in the `Fit` path (`_stochastic_gradient.py:79`,
    // `RegressorMixin.score` -> `r2_score`).
    let early_stopping = hyper.early_stopping && val_set.is_some();
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

    // Averaged-SGD (ASGD) accumulators (`_sgd_fast.pyx.tp:646-654`). Direct
    // running-mean form of sklearn's lazy `w.add_average` (mathematically
    // identical for plain arrays): once `t >= average`, `avg += (current - avg)
    // / (t - average + 1)`. A passive observer — never fed back into the live
    // `weights`/`intercept` trajectory.
    let mut average_coef: Array1<F> = Array1::zeros(n_features);
    let mut average_intercept = F::zero();

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
            // Per-sample weight scaling: `update *= class_weight * sample_weight`
            // with `class_weight = 1` for regression (`_sgd_fast.pyx.tp:630`).
            // `g` is the sample-weighted gradient, scaling ONLY the gradient term.
            let g = grad * sample_w[i];
            // `sumloss` is the SUM (not mean) of per-sample losses over the
            // epoch (`_sgd_fast.pyx.tp:597`), computed from the UNWEIGHTED loss.
            epoch_loss = epoch_loss + loss_fn.loss(y[i], y_pred);

            // L2 shrink: clamped multiplicative factor
            // `max(0, 1 - (1-eff)*eta*alpha)` applied to the whole weight vector
            // BEFORE the gradient add (`_sgd_fast.pyx.tp:632-635`); for pure L2
            // (`eff=0`) this is `max(0, 1-eta*alpha)`, for L1 (`eff=1`) it is 1.
            let shrink = (F::one() - (F::one() - eff) * eta * hyper.alpha).max(F::zero());
            for j in 0..n_features {
                weights[j] = weights[j] * shrink;
            }
            // Gradient add `w.add(x, -eta*g)` with the scaled `g`
            // (`_sgd_fast.pyx.tp:637-638`).
            for j in 0..n_features {
                weights[j] = weights[j] - eta * g * xi[j];
            }
            // The intercept update is gated on `fit_intercept` and is NOT
            // regularized (`_sgd_fast.pyx.tp:639-644`: `if fit_intercept == 1:
            // intercept_update = update; ... intercept += intercept_update *
            // intercept_decay`). `update = -eta*g` is the SCALED update. When
            // `fit_intercept` is false the intercept is never modified and stays
            // at its init value `0` (`intercept` enters this fn as `0`).
            if hyper.fit_intercept {
                *intercept = *intercept - eta * g;
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

            // ASGD running-mean accumulation (`_sgd_fast.pyx.tp:646-654`),
            // AFTER the weight/intercept update + L1 truncation so
            // `weights`/`intercept` are the final post-step values for this
            // sample. `t` is the SAME 1-based global step the schedule used;
            // `num_iter = t - average + 1` is `>= 1` whenever `t >= average`.
            if hyper.average > 0 && t >= hyper.average {
                let num_iter = t - hyper.average + 1;
                let num_iter_f = F::from(num_iter).unwrap_or_else(F::one);
                let mu = F::one() / num_iter_f;
                for j in 0..n_features {
                    average_coef[j] = average_coef[j] + (weights[j] - average_coef[j]) * mu;
                }
                average_intercept = average_intercept + (*intercept - average_intercept) * mu;
            }
        }

        // `epoch_loss` is now the epoch `sumloss` (no mean division).
        total_loss = epoch_loss;

        // Epoch-end stop rule (`_sgd_fast.pyx.tp:678-707`). When early stopping
        // is active, score the CURRENT weights on the held-out validation set
        // (R^2, `_stochastic_gradient.py:79`) and run the score-based branch
        // (`best_score` init `-inf`, higher is better, `:678-687`); otherwise the
        // training-loss branch (`sumloss` vs `best_loss`, `:688-695`).
        let should_break = if let (true, Some((x_val, y_val))) = (early_stopping, val_set) {
            let score = r2_score(weights, *intercept, x_val, y_val);
            convergence_tail_score(
                score,
                &mut best_score,
                &mut no_improve_count,
                &mut current_eta,
                tol_active,
                hyper.tol,
                hyper.n_iter_no_change,
                matches!(hyper.learning_rate, LearningRateSchedule::Adaptive),
            )
        } else {
            convergence_tail(
                epoch_loss,
                &mut best_loss,
                &mut no_improve_count,
                &mut current_eta,
                tol_active,
                hyper.tol,
                n_samples,
                hyper.n_iter_no_change,
                matches!(hyper.learning_rate, LearningRateSchedule::Adaptive),
            )
        };
        if should_break {
            break;
        }
    }

    // ASGD finalize (`_stochastic_gradient.py:834-836`: averaged coef/intercept
    // chosen when `average <= self.t_ - 1`). `t` here equals sklearn's
    // `self.t_ - 1` on the full-fit path (`initial_t = 0`, sklearn inits
    // `self.t_ = 1` then adds `n_iter_ * n_samples`), so the condition is
    // `hyper.average <= t`.
    if hyper.average > 0 && hyper.average <= t {
        for j in 0..n_features {
            weights[j] = average_coef[j];
        }
        *intercept = average_intercept;
    }

    (total_loss, t)
}

/// Dispatch regressor training to the appropriate typed loss function.
///
/// `sample_w[i] = sample_weight[i]` is forwarded verbatim to the kernel
/// (`_sgd_fast.pyx.tp:630`, `class_weight = 1` for regression).
#[allow(
    clippy::too_many_arguments,
    reason = "threads the per-sample weight vector"
)]
fn dispatch_train_regressor<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    w: &mut Array1<F>,
    b: &mut F,
    loss_enum: &RegressorLoss<F>,
    hyper: &SGDHyper<F>,
    initial_t: usize,
    sample_w: &[F],
    val_set: Option<(&Array2<F>, &Array1<F>)>,
) -> (F, usize) {
    match loss_enum {
        RegressorLoss::SquaredError => train_regressor_sgd(
            x,
            y,
            w,
            b,
            &SquaredError,
            hyper,
            initial_t,
            sample_w,
            val_set,
        ),
        RegressorLoss::Huber(eps) => train_regressor_sgd(
            x,
            y,
            w,
            b,
            &Huber { epsilon: *eps },
            hyper,
            initial_t,
            sample_w,
            val_set,
        ),
        RegressorLoss::EpsilonInsensitive(eps) => train_regressor_sgd(
            x,
            y,
            w,
            b,
            &EpsilonInsensitive { epsilon: *eps },
            hyper,
            initial_t,
            sample_w,
            val_set,
        ),
        RegressorLoss::SquaredEpsilonInsensitive(eps) => train_regressor_sgd(
            x,
            y,
            w,
            b,
            &SquaredEpsilonInsensitive { epsilon: *eps },
            hyper,
            initial_t,
            sample_w,
            val_set,
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
#[allow(
    clippy::too_many_arguments,
    reason = "threads each validated parameter"
)]
fn validate_reg_params<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    schedule: &LearningRateSchedule<F>,
    eta0: F,
    alpha: F,
    l1_ratio: F,
    loss: &RegressorLoss<F>,
    validation_fraction: F,
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
    // sklearn `_stochastic_gradient.py:2024`:
    // `"epsilon": [Interval(Real, 0, None, closed="left")]` — epsilon must be
    // `>= 0` (a negative epsilon raises `InvalidParameterError`). ferrolearn
    // carries epsilon inside the loss variant, so the faithful equivalent is to
    // reject a negative epsilon on the variants that have one (SquaredError has
    // none, boundary 0 is valid — closed-left).
    let epsilon = match loss {
        RegressorLoss::Huber(e)
        | RegressorLoss::EpsilonInsensitive(e)
        | RegressorLoss::SquaredEpsilonInsensitive(e) => Some(*e),
        RegressorLoss::SquaredError => None,
    };
    if let Some(e) = epsilon
        && e < F::zero()
    {
        return Err(FerroError::InvalidParameter {
            name: "epsilon".into(),
            reason: "must be in the range [0, inf)".into(),
        });
    }
    validate_validation_fraction(validation_fraction)?;
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
        // Delegate to the sample-weighted path with a uniform `ones(n)` weight
        // vector — byte-identical to the unweighted kernel
        // (`_check_sample_weight` -> ones, `_stochastic_gradient.py:627`).
        let sample_weight = Array1::<F>::from_elem(x.nrows(), F::one());
        self.fit_with_sample_weight(x, y, &sample_weight)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> SGDRegressor<F> {
    /// Fit the SGD regressor with explicit per-sample weights.
    ///
    /// Mirrors `SGDRegressor.fit(X, y, sample_weight=...)`. The per-sample weight
    /// scales ONLY the gradient-derived part of each update
    /// (`update *= class_weight * sample_weight` with `class_weight = 1` for
    /// regression, `_sgd_fast.pyx.tp:630`); the L2 shrink and L1 truncation are
    /// unscaled. [`Fit::fit`] delegates here with a uniform `ones(n)` vector.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x`/`y`/`sample_weight` have
    /// mismatched sample counts.
    /// Returns [`FerroError::InvalidParameter`] if `eta0` or `alpha` are
    /// invalid.
    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: &Array1<F>,
    ) -> Result<FittedSGDRegressor<F>, FerroError> {
        validate_reg_params(
            x,
            y,
            &self.learning_rate,
            self.eta0,
            self.alpha,
            self.l1_ratio,
            &self.loss,
            self.validation_fraction,
        )?;

        let n_samples = x.nrows();
        if sample_weight.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![sample_weight.len()],
                context: "sample_weight length must match number of samples in X".into(),
            });
        }

        // Non-finite input validation (#2263). sklearn `SGDRegressor.fit`
        // -> `self._validate_data(X, y, ...)` (`_stochastic_gradient.py:1476`,
        // the shared `BaseSGD` base path) keeps the default
        // `force_all_finite=True`, so `check_array` rejects any NaN or +/-inf in
        // X OR y with a `ValueError("Input X contains NaN.")` /
        // `"Input y contains NaN."` / `"... contains infinity ..."` BEFORE the
        // SGD kernel. sklearn also validates `sample_weight` via
        // `_check_sample_weight` (default `force_all_finite=True`,
        // `_stochastic_gradient.py:1501`). `y` is `Array1<F>` (float targets), so
        // X, y, AND sample_weight all need the runtime check.
        // `.iter().any(|v| !v.is_finite())` rejects both NaN and Inf (bounds-safe,
        // no panic, R-CODE-2); the finite path is byte-identical. This is the
        // SGDRegressor fit entry (`Fit::fit` delegates here with unit weights).
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "Input y contains NaN or infinity.".into(),
            });
        }
        if sample_weight.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "sample_weight".into(),
                reason: "Input sample_weight contains NaN or infinity.".into(),
            });
        }

        let n_features = x.ncols();
        let hyper = reg_hyper(self);
        let mut w = Array1::<F>::zeros(n_features);
        let mut b = F::zero();
        let sw = sample_weight.to_vec();

        // Early-stopping validation split (`ShuffleSplit`,
        // `_stochastic_gradient.py:282-287`). The hold-out is split off BEFORE
        // training; the kernel trains on the train subset and scores its R^2 on
        // the held-out validation subset each epoch. sklearn does NOT refit on
        // the full data — the train-subset weights are final.
        let t = if hyper.early_stopping {
            let (train_idx, val_idx) = make_validation_split(
                n_samples,
                hyper.validation_fraction,
                hyper.random_state,
                None,
            )
            .ok_or_else(|| FerroError::InvalidParameter {
                name: "validation_fraction".into(),
                reason: "early_stopping split led to an empty train or validation set; \
                         increase the number of samples or change validation_fraction"
                    .into(),
            })?;
            let x_tr = gather_rows(x, &train_idx);
            let y_tr = gather(y, &train_idx);
            let sw_tr: Vec<F> = train_idx.iter().map(|&i| sw[i]).collect();
            let x_val = gather_rows(x, &val_idx);
            let y_val = gather(y, &val_idx);
            let (_, t) = dispatch_train_regressor(
                &x_tr,
                &y_tr,
                &mut w,
                &mut b,
                &self.loss,
                &hyper,
                0,
                &sw_tr,
                Some((&x_val, &y_val)),
            );
            t
        } else {
            let (_, t) =
                dispatch_train_regressor(x, y, &mut w, &mut b, &self.loss, &hyper, 0, &sw, None);
            t
        };

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
        // `partial_fit` carries no per-sample weight here; uniform `1.0`.
        let sample_w: Vec<F> = vec![F::one(); x.nrows()];

        let (_, t) = dispatch_train_regressor(
            x,
            y,
            &mut self.weights,
            &mut self.intercept,
            &self.loss,
            &hyper,
            self.t,
            &sample_w,
            None,
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
            &self.loss,
            self.validation_fraction,
        )?;

        let n_features = x.ncols();
        let mut hyper = reg_hyper(&self);
        hyper.max_iter = 1;
        let mut w = Array1::<F>::zeros(n_features);
        let mut b = F::zero();
        // Initial `partial_fit` carries no per-sample weight here; uniform `1.0`.
        let sample_w: Vec<F> = vec![F::one(); x.nrows()];

        let (_, t) =
            dispatch_train_regressor(x, y, &mut w, &mut b, &self.loss, &hyper, 0, &sample_w, None);

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
// SGDOneClassSVM
// ---------------------------------------------------------------------------

/// Linear One-Class SVM trained by Stochastic Gradient Descent.
///
/// Mirrors scikit-learn's
/// [`SGDOneClassSVM`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDOneClassSVM.html)
/// (`_stochastic_gradient.py:2084-2668`). It solves the linear One-Class SVM
/// primal via the same SGD kernel as [`SGDClassifier`], with the targets fixed
/// to `y = ones(n)`, the Hinge loss (`threshold = 1`), the L2 penalty, and
/// `alpha = nu / 2` (`_stochastic_gradient.py:2479,2588`). The SGD intercept
/// `b` relates to the One-Class offset `rho` by `offset_ = 1 - b`
/// (`_stochastic_gradient.py:2325,2377`), and the per-sample intercept update
/// gains an extra `- 2*eta*alpha` term (`_sgd_fast.pyx.tp:641-642`).
///
/// The decision function is `decision_function(X) = X · coef_ - offset_`
/// (`_stochastic_gradient.py:2622`); `predict` returns `+1` (inlier) where the
/// decision is `>= 0` and `-1` (outlier) otherwise (`:2655-2657`).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_linear::sgd::SGDOneClassSVM;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((4, 2), vec![
///     -1.0, -1.0, -2.0, -1.0, 1.0, 1.0, 2.0, 1.0,
/// ]).unwrap();
///
/// let model = SGDOneClassSVM::<f64>::new()
///     .with_learning_rate(ferrolearn_linear::sgd::LearningRateSchedule::Constant)
///     .with_eta0(0.05)
///     .with_max_iter(10)
///     .with_shuffle(false);
/// let fitted = model.fit(&x, &()).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// assert_eq!(preds.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct SGDOneClassSVM<F> {
    /// The `nu` parameter — an upper bound on the fraction of training errors
    /// and a lower bound on the fraction of support vectors. Must be in
    /// `(0, 1]`. Defaults to `0.5` (`_stochastic_gradient.py:2098-2102,2247`).
    pub nu: F,
    /// Whether to fit (update) the intercept. Defaults to `true`
    /// (`_stochastic_gradient.py:2104-2105,2248`).
    pub fit_intercept: bool,
    /// Maximum number of passes over the training data. Defaults to `1000`
    /// (`_stochastic_gradient.py:2107,2249`).
    pub max_iter: usize,
    /// Convergence tolerance. Defaults to `1e-3`
    /// (`_stochastic_gradient.py:2113,2250`). Set to `F::neg_infinity()` to
    /// disable the early-stop rule (the analog of sklearn's `tol=None`,
    /// `_stochastic_gradient.py:2310`).
    pub tol: F,
    /// Whether to shuffle the training data after each epoch. Defaults to
    /// `true` (`_stochastic_gradient.py:2118,2251`).
    pub shuffle: bool,
    /// The learning rate schedule. Defaults to `Optimal`
    /// (`_stochastic_gradient.py:2132,2254`).
    pub learning_rate: LearningRateSchedule<F>,
    /// Initial learning rate for the `constant`/`invscaling`/`adaptive`
    /// schedules. Defaults to `0.0` (`_stochastic_gradient.py:2145,2255`).
    pub eta0: F,
    /// Power parameter for the inverse-scaling schedule. Defaults to `0.5`
    /// (`_stochastic_gradient.py:2151,2256`).
    pub power_t: F,
    /// Optional random seed for sample shuffling
    /// (`_stochastic_gradient.py:2125`).
    pub random_state: Option<u64>,
    /// Number of consecutive non-improving epochs before convergence (or, under
    /// the `adaptive` schedule, before `eta` is divided by 5). Defaults to `5`
    /// (`_stochastic_gradient.py:2278`).
    pub n_iter_no_change: usize,
}

impl<F: Float> SGDOneClassSVM<F> {
    /// Create a new `SGDOneClassSVM` with default settings.
    ///
    /// Defaults match scikit-learn's `SGDOneClassSVM.__init__`
    /// (`_stochastic_gradient.py:2245-2281`): `nu = 0.5`,
    /// `fit_intercept = true`, `max_iter = 1000`, `tol = 1e-3`,
    /// `shuffle = true`, `learning_rate = Optimal`, `eta0 = 0.0`,
    /// `power_t = 0.5`, `n_iter_no_change = 5`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nu: cst(0.5),
            fit_intercept: true,
            max_iter: 1000,
            tol: cst(1e-3),
            shuffle: true,
            learning_rate: LearningRateSchedule::Optimal,
            eta0: cst(0.0),
            power_t: cst(0.5),
            random_state: None,
            n_iter_no_change: 5,
        }
    }

    /// Set the `nu` parameter (upper bound on the fraction of training errors).
    #[must_use]
    pub fn with_nu(mut self, nu: F) -> Self {
        self.nu = nu;
        self
    }

    /// Set whether the intercept (bias) term is fit.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
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

    /// Set whether the training data is shuffled after each epoch.
    #[must_use]
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
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

    /// Set the power parameter for inverse scaling.
    #[must_use]
    pub fn with_power_t(mut self, power_t: F) -> Self {
        self.power_t = power_t;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the number of consecutive non-improving epochs before convergence.
    #[must_use]
    pub fn with_n_iter_no_change(mut self, n_iter_no_change: usize) -> Self {
        self.n_iter_no_change = n_iter_no_change;
        self
    }

    /// Fit the linear One-Class SVM on `x` (the X-only fit shape).
    ///
    /// This is the inherent entry point mirroring sklearn's `fit(X)`
    /// (`_stochastic_gradient.py:2554-2600`). The [`Fit`] trait impl with a
    /// unit target `()` delegates here.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `nu` is not in `(0, 1]`
    ///   (`_stochastic_gradient.py:2236`,
    ///   `Interval(Real, 0.0, 1.0, closed="right")`).
    /// - [`FerroError::InvalidParameter`] if `eta0` is not positive for the
    ///   `constant`/`invscaling`/`adaptive` schedules.
    /// - [`FerroError::InsufficientSamples`] if `x` has no rows.
    pub fn fit_one_class(&self, x: &Array2<F>) -> Result<FittedSGDOneClassSVM<F>, FerroError>
    where
        F: Send + Sync + ScalarOperand + 'static,
    {
        // `nu` constraint: `Interval(Real, 0.0, 1.0, closed="right")`, i.e.
        // `0 < nu <= 1` (`_stochastic_gradient.py:2236`).
        if self.nu <= F::zero() || self.nu > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "nu".into(),
                reason: "must be in the range (0, 1]".into(),
            });
        }
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "SGDOneClassSVM requires at least one sample".into(),
            });
        }

        // Non-finite input validation (#2263) — SEPARATE SGD arm. sklearn
        // `SGDOneClassSVM.fit` -> `self._validate_data(X, None, ...)`
        // (`_stochastic_gradient.py:2392`) keeps the default
        // `force_all_finite=True`, so `check_array` rejects any NaN or +/-inf in
        // X with a `ValueError("Input X contains NaN.")` / `"... contains
        // infinity ..."` BEFORE the SGD kernel. This is an X-only fit (no `y`, no
        // `sample_weight` argument), so X is the only runtime check.
        // `.iter().any(|v| !v.is_finite())` rejects both NaN and Inf (bounds-safe,
        // no panic, R-CODE-2); the finite path is byte-identical.
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }

        // `eta0 > 0` is required for the constant/invscaling/adaptive schedules
        // (mirrors `_more_validate_params`, `_stochastic_gradient.py:149-153`);
        // the `optimal` schedule accepts `eta0 == 0`.
        if schedule_requires_eta0(&self.learning_rate) && self.eta0 <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "eta0".into(),
                reason: "must be positive".into(),
            });
        }

        let n_features = x.ncols();
        // sklearn: `alpha = self.nu / 2` (`_stochastic_gradient.py:2588`),
        // `penalty="l2"`, `l1_ratio=0`, `loss="hinge"` (`:2262-2265`).
        let two = cst::<F>(2.0);
        let alpha = self.nu / two;
        let hyper = SGDHyper {
            learning_rate: self.learning_rate,
            eta0: self.eta0,
            alpha,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            power_t: self.power_t,
            penalty: Penalty::L2,
            l1_ratio: F::zero(),
            shuffle: self.shuffle,
            n_iter_no_change: self.n_iter_no_change,
            fit_intercept: self.fit_intercept,
            one_class: true,
            // sklearn's `SGDOneClassSVM` has no `average` parameter — averaging is
            // always off on the one-class path (`_stochastic_gradient.py:2245-2281`).
            average: 0,
            // The one-class SVM exposes no `early_stopping`/`validation_fraction`
            // (`_stochastic_gradient.py:2245-2281`); the early-stop score branch
            // is always off, leaving the one-class trajectory byte-identical.
            early_stopping: false,
            validation_fraction: cst(0.1),
        };

        // `y = np.ones(n_samples)` (`_stochastic_gradient.py:2289`).
        let y_ones: Array1<F> = Array1::from_elem(n_samples, F::one());
        let mut w = Array1::<F>::zeros(n_features);
        // The One-Class offset is initialized to 0, so the SGD intercept starts
        // at `b = 1 - offset_ = 1` (`_stochastic_gradient.py:2238,2325`). This
        // differs from the classifier/regressor paths, which start at `b = 0`.
        let mut b = F::one();
        // One-Class SVM fit has no per-sample weighting here; the per-sample
        // weight is uniform `1.0` (byte-identical to the pre-weighting kernel).
        let sample_w: Vec<F> = vec![F::one(); n_samples];

        let (_, _t) = train_binary_sgd(
            x, &y_ones, &mut w, &mut b, &Hinge, &hyper, 0, &sample_w, None,
        );

        // `offset_ = 1 - intercept` (`_stochastic_gradient.py:2377`).
        let offset = F::one() - b;

        Ok(FittedSGDOneClassSVM {
            coef: w,
            offset,
            n_features,
        })
    }
}

impl<F: Float> Default for SGDOneClassSVM<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, ()> for SGDOneClassSVM<F> {
    type Fitted = FittedSGDOneClassSVM<F>;
    type Error = FerroError;

    /// Fit the linear One-Class SVM. The target `y` is ignored (present for API
    /// consistency, mirroring sklearn's `fit(X, y=None)`,
    /// `_stochastic_gradient.py:2554`); the fit uses `y = ones(n)` internally.
    ///
    /// # Errors
    ///
    /// See [`SGDOneClassSVM::fit_one_class`].
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedSGDOneClassSVM<F>, FerroError> {
        self.fit_one_class(x)
    }
}

/// Fitted linear One-Class SVM.
///
/// Holds the learned weight vector `coef_` and the One-Class offset `offset_`
/// (`_stochastic_gradient.py:2177-2182`). Implements [`Predict`] (returning
/// `+1`/`-1` inlier/outlier labels) and exposes [`decision_function`],
/// [`score_samples`], [`coef`], and [`offset`].
///
/// [`decision_function`]: FittedSGDOneClassSVM::decision_function
/// [`score_samples`]: FittedSGDOneClassSVM::score_samples
/// [`coef`]: FittedSGDOneClassSVM::coef
/// [`offset`]: FittedSGDOneClassSVM::offset
#[derive(Debug, Clone)]
pub struct FittedSGDOneClassSVM<F> {
    /// Weight vector (`coef_`, shape `(n_features,)`).
    coef: Array1<F>,
    /// The One-Class offset (`offset_`), a scalar: `offset_ = 1 - intercept`.
    offset: F,
    /// Number of features the model was trained on.
    n_features: usize,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> FittedSGDOneClassSVM<F> {
    /// The learned weight vector (`coef_`).
    #[must_use]
    pub fn coef(&self) -> &Array1<F> {
        &self.coef
    }

    /// The One-Class offset (`offset_`).
    ///
    /// Satisfies `decision_function = score_samples - offset_`
    /// (`_stochastic_gradient.py:2182`).
    #[must_use]
    pub fn offset(&self) -> F {
        self.offset
    }

    /// Signed distance to the separating hyperplane:
    /// `decision_function(X) = X · coef_ - offset_`
    /// (`_stochastic_gradient.py:2622`).
    ///
    /// Positive for an inlier, negative for an outlier.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }
        Ok(x.dot(&self.coef) - self.offset)
    }

    /// Raw scoring function of the samples:
    /// `score_samples(X) = decision_function(X) + offset_ = X · coef_`
    /// (`_stochastic_gradient.py:2639`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn score_samples(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        Ok(self.decision_function(x)? + self.offset)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedSGDOneClassSVM<F>
{
    type Output = Array1<isize>;
    type Error = FerroError;

    /// Return labels (`+1` inlier, `-1` outlier) for the given feature matrix.
    ///
    /// Mirrors `_stochastic_gradient.py:2655-2657`:
    /// `y = (decision_function(X) >= 0); y[y == 0] = -1`, i.e. `+1` where the
    /// decision is `>= 0` and `-1` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let decisions = self.decision_function(x)?;
        Ok(decisions.mapv(|d| if d >= F::zero() { 1 } else { -1 }))
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
    // Early-stopping validation-score helpers (REQ-13)
    //
    // These pin the DETERMINISTIC per-epoch validation score against the live
    // sklearn 1.5.2 oracle (`sklearn.metrics.r2_score` / `accuracy_score`, the
    // regressor/classifier `_ValidationScoreCallback`,
    // `_stochastic_gradient.py:79`). The validation-set SELECTION is numpy-RNG
    // coupled and NOT verified here (same barrier as `shuffle`); the SCORE math
    // on a GIVEN (weights, intercept, val set) is fully deterministic.
    // -----------------------------------------------------------------------

    #[test]
    fn test_validation_r2_matches_sklearn() {
        // Oracle:
        //   python3 -c "import numpy as np; from sklearn.metrics import r2_score; \
        //     X=np.array([[1.,2.],[3.,1.],[0.,4.]]); w=np.array([1.,-2.]); b=0.5; \
        //     y=np.array([-2.,4.,-7.]); print(r2_score(y, X@w+b))"
        //   -> 0.8887362637362637
        let weights = array![1.0_f64, -2.0];
        let x_val = array![[1.0_f64, 2.0], [3.0, 1.0], [0.0, 4.0]];
        let y_val = array![-2.0_f64, 4.0, -7.0];
        let got = r2_score(&weights, 0.5, &x_val, &y_val);
        assert!(
            (got - 0.8887362637362637).abs() < 1e-12,
            "r2 {got} != sklearn 0.8887362637362637"
        );
    }

    #[test]
    fn test_validation_r2_constant_y_edge_cases() {
        // sklearn `r2_score` with SS_tot == 0 (constant y_val): perfect const
        // prediction -> 1.0, imperfect -> 0.0.
        //   python3 -c "from sklearn.metrics import r2_score; \
        //     print(r2_score([5,5,5],[5,5,5]), r2_score([5,5,5],[4,4,4]))"
        //   -> 1.0 0.0
        let weights = array![0.0_f64, 0.0];
        let x = array![[1.0_f64, 2.0], [3.0, 1.0], [0.0, 4.0]];
        let y_const = array![5.0_f64, 5.0, 5.0];
        // weights=0, intercept=5 -> all predictions 5 -> perfect -> 1.0.
        let got_perfect = r2_score(&weights, 5.0, &x, &y_const);
        assert!(
            (got_perfect - 1.0).abs() < 1e-12,
            "perfect-const r2 {got_perfect} != 1.0"
        );
        // weights=0, intercept=4 -> all predictions 4, y all 5 -> imperfect -> 0.0.
        let got_imperfect = r2_score(&weights, 4.0, &x, &y_const);
        assert!(
            got_imperfect.abs() < 1e-12,
            "imperfect-const r2 {got_imperfect} != 0.0"
        );
    }

    #[test]
    fn test_validation_binary_accuracy_matches_sklearn() {
        // Oracle:
        //   python3 -c "import numpy as np; from sklearn.metrics import accuracy_score; \
        //     X=np.array([[1.,1.],[2.,2.],[0.,0.],[3.,1.]]); w=np.array([1.,1.]); b=-3.; \
        //     dec=X@w+b; pred=np.where(dec>=0,1.,-1.); y=np.array([1.,1.,-1.,1.]); \
        //     print(accuracy_score(y,pred))"
        //   -> 0.75  (dec=[-1,1,-3,1] -> pred=[-1,1,-1,1], y=[1,1,-1,1] -> 3/4)
        let weights = array![1.0_f64, 1.0];
        let x_val = array![[1.0_f64, 1.0], [2.0, 2.0], [0.0, 0.0], [3.0, 1.0]];
        let y_val = array![1.0_f64, 1.0, -1.0, 1.0];
        let got = binary_accuracy(&weights, -3.0, &x_val, &y_val);
        assert!((got - 0.75).abs() < 1e-12, "accuracy {got} != sklearn 0.75");
    }

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
