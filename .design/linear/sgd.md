# Stochastic Gradient Descent (SGD) linear models

<!--
tier: 3-component
status: draft
baseline-commit: c2f944ee27a75ae5de4be7adca13ab0547683b85
upstream-paths:
  - sklearn/linear_model/_stochastic_gradient.py
  - sklearn/linear_model/_sgd_fast.pyx.tp
  - sklearn/linear_model/_sag.py
-->

## Summary

`ferrolearn-linear/src/sgd.rs` mirrors scikit-learn's
`sklearn.linear_model.SGDClassifier` and `SGDRegressor` (and is routed to also
cover `SGDOneClassSVM` plus the passive-aggressive wrapper estimators): linear
models trained by stochastic gradient descent over a configurable convex loss
with inline penalties and a learning-rate schedule. The current implementation
ships the classifier and regressor loss families used by sklearn's SGD and
passive-aggressive estimators, `constant`/`optimal`/`invscaling`/`adaptive` and
`pa1`/`pa2` schedules, L2/L1/elastic-net update math, one-vs-all multiclass,
`partial_fit` for SGD, `SGDOneClassSVM`, `PassiveAggressiveClassifier`,
`PassiveAggressiveRegressor`, sample/class weights on `fit`, early stopping,
averaging, and pipeline integration. Remaining divergences include sparse input,
Python warning/error ABI, warm start, `n_jobs`, exact numpy-RNG shuffle and
validation-split parity, richer fitted attributes, and ferray substrate
migration. The substrate is `ndarray`, not ferray.

`_sag.py` (SAG/SAGA solvers) is a secondary upstream listed by the route; it is
**not** the engine behind sklearn's SGD estimators (it backs `Ridge`/`LogisticRegression`
`solver="sag"/"saga"`). No part of `sgd.rs` implements SAG/SAGA, and sklearn's
SGD estimators do not call it; it is documented here only for completeness and
is out of scope for these REQs.

## Parity boundary / RNG (read first)

sklearn's per-epoch shuffle is numpy's `our_rand_r` permutation inside
`SequentialDataset.shuffle(seed)` (`_sgd_fast.pyx.tp:579-580`); ferrolearn
shuffles with `rand::rngs::StdRng` (`fn train_binary_sgd in sgd.rs`,
`indices.shuffle(&mut rng)`). These two PRNGs produce different permutations
from the same integer seed, so **the full fitted-weight trajectory of a
multi-epoch shuffled fit is NOT bit-parityable across the two
implementations** — even after every formula is corrected. This is an
intrinsic property of cross-PRNG translation, not a bug to fix.

Therefore every parity REQ below frames its acceptance on the **deterministic
decision pieces** that do not depend on shuffle order:

- per-sample loss / dloss **formulas** (unit granularity, REQ-1..3);
- learning-rate schedule **values** including the `optimal` `t0` (unit
  granularity, REQ-6/REQ-7);
- the L2 / L1 penalty **update math** (single deterministic step, REQ-4/REQ-5);
- default **parameters** and **convergence rule** (REQ-9/REQ-10);
- whichever pieces (`fit_intercept`, `shuffle=False`, `average`, `class_weight`)
  can be exercised with shuffle disabled or on a single sample.

Full-fit `coef_`/`intercept_` parity is feasible only with `shuffle=False` AND
a single pass (`max_iter=1`, no convergence early-exit), where the update order
is the natural index order in both implementations; even then it requires every
formula REQ to be SHIPPED first. No REQ here claims random-shuffle full-fit
weight parity.

## Algorithm (the sklearn contract, `_sgd_fast.pyx.tp` `_plain_sgd`)

Per sample `(x, y)` within an epoch (`_sgd_fast.pyx.tp:581-661`):

1. **predict** `p = w·x + intercept` (`:590`), with `w` held as a scaled vector
   `(w_data, wscale)` so that the effective weight is `wscale * w_data`.
2. **schedule** (`:591-594`, plus init at `:563-570`):
   - `constant`: `eta = eta0`.
   - `optimal`: `eta = 1 / (alpha * (optimal_init + t - 1))`, where
     `typw = sqrt(1 / sqrt(alpha))`, `initial_eta0 = typw / max(1, dloss(1, -typw))`,
     `optimal_init = 1 / (initial_eta0 * alpha)`. **`t` starts at `t_ = 1.0`**
     (`_stochastic_gradient.py:645,723`), so the first sample sees
     `eta = 1/(alpha*optimal_init) = initial_eta0`.
   - `invscaling`: `eta = eta0 / pow(t, power_t)`.
   - `adaptive`: holds `eta` constant within the schedule block; decreased only
     by the convergence handler (step 8).
   - `pa1`: `update = min(eta0, loss(y,p) / ||x||^2)`; if `||x||^2 == 0`,
     skip the update (`:514-518`).
   - `pa2`: `update = loss(y,p) / (||x||^2 + 0.5 / eta0)` (`:519-521`).
3. **update scalar** for non-PA schedules: `dloss = loss.dloss(y, p)`, clipped
   to `±MAX_DLOSS = 1e12` (`:613-619`); `update = -eta * dloss` (`:620`). For
   PA schedules, multiply the positive PA step by `y` for hinge classification
   or by `sign(y - p)` for epsilon-insensitive regression (`:532-537`). Then
   `update *= class_weight * sample_weight` (`:630`).
4. **L2 shrink** (`penalty_type >= L2`, `:632-635`): scale the *whole* weight
   vector by `max(0, 1 - (1 - l1_ratio) * eta * alpha)` via `w.scale(...)` —
   a multiplicative `wscale` decay applied to every coordinate **before** the
   gradient is added, NOT an additive `eta*alpha*w_j` per touched feature.
5. **gradient add** `w.add(x, update)` (`:637-638`) — only the non-zero features
   of `x`.
6. **intercept** (`fit_intercept == 1`, `:639-644`):
   `intercept += update * intercept_decay` (`intercept_decay=1.0` for
   classifier/regressor; the SGDOneClassSVM path subtracts `2*eta*alpha`).
7. **L1 cumulative penalty** (`L1` or `ELASTICNET`, `:656-658`):
   `u += l1_ratio * eta * alpha`, then `l1penalty(...)` (`:750-778`) applies the
   Tsuruoka truncated-gradient: for each touched feature, push the coordinate
   toward 0 by `(u ± q[idx]) / wscale` (clamped not to cross 0) and accumulate
   the applied penalty in `q[idx]`.
8. After each epoch (`:688-707`): if **not** early-stopping, track
   `best_loss`; increment `no_improvement_count` while
   `sumloss > best_loss - tol * train_count`; reset it otherwise; record
   `best_loss = min(best_loss, sumloss)`. When `no_improvement_count >=
   n_iter_no_change`: if `adaptive` and `eta > 1e-6`, set `eta /= 5` and reset
   the counter; otherwise **break** (converged). `sumloss` is the **sum** of
   per-sample losses over the epoch, not the mean.

Final `w.reset_wscale()` folds `wscale` back into `w_data` (`:714`).

### Loss / dloss formulas (`_sgd_fast.pyx.tp`)

`dloss` is `d loss / d p` (note the gradient w.r.t. the *prediction*, so the
weight step uses `update = -eta * dloss`; the sign already accounts for `y`):

| loss | sklearn `loss(y,p)` | sklearn `dloss(y,p)` | lines |
|---|---|---|---|
| Hinge(thr=1) | `z=py; thr - z if z<=thr else 0` | `-y if z<=thr else 0` | `:216-226` |
| SquaredHinge | `z=thr-py; z*z if z>0 else 0` | `-2y·z if z>0 else 0` | `:248-258` |
| Log | `z=py; ~exp(-z) / log(1+exp(-z))` | `-y·exp(-z)/(exp(z)+1)` regimes | `:267-283` |
| ModifiedHuber | `0 if z>=1; (1-z)^2 if z>=-1; -4z` | `0 if z>=1; -2y(1-z) if z>=-1; -4y` | `:178-194` |
| SquaredLoss | `0.5(p-y)^2` | `p-y` | `:291-295` |
| Huber(c) | `0.5r^2 if |r|<=c; c|r|-0.5c^2` (`r=p-y`) | `r if |r|<=c; ±c` | `:315-331` |
| EpsilonInsensitive(eps) | `max(0,|y-p|-eps)` | `-1 if y-p>eps; 1 if p-y>eps; 0` | `:348-358` |
| SquaredEpsilonInsensitive | `max(0,|y-p|-eps)^2` | `-2(z-eps) / 2(-z-eps) / 0` (`z=y-p`) | `:375-387` |

### Defaults (`_stochastic_gradient.py`)

| param | `SGDClassifier` (`:1227-1271`) | `SGDRegressor` (`:2028-2068`) | `SGDOneClassSVM` (`:2245-2278`) |
|---|---|---|---|
| `loss` | `"hinge"` | `"squared_error"` | `"hinge"` (fixed) |
| `penalty` | `"l2"` | `"l2"` | n/a (`l2`) |
| `alpha` | `0.0001` | `0.0001` | — (`nu=0.5`) |
| `l1_ratio` | `0.15` | `0.15` | — |
| `fit_intercept` | `True` | `True` | `True` |
| `max_iter` | `1000` | `1000` | `1000` |
| `tol` | `1e-3` | `1e-3` | `1e-3` |
| `shuffle` | `True` | `True` | `True` |
| `epsilon` | `DEFAULT_EPSILON=0.1` | `0.1` | — |
| `learning_rate` | `"optimal"` | `"invscaling"` | `"optimal"` |
| `eta0` | `0.0` | `0.01` | `0.0` |
| `power_t` | `0.5` | `0.25` | `0.5` |
| `n_iter_no_change` | `5` | `5` | `5` |
| `early_stopping` | `False` | `False` | n/a |
| `validation_fraction` | `0.1` | `0.1` | n/a |
| `average` | `False` | `False` | `False` |
| `class_weight` | `None` | n/a | n/a |

`DEFAULT_EPSILON = 0.1` (`:57`); `LEARNING_RATE_TYPES`/`PENALTY_TYPES`
(`:46-55`).

## Requirements

- REQ-1: Classifier losses `hinge`, `log_loss`, `modified_huber`,
  `squared_error` and their `dloss` gradients match `_sgd_fast` exactly.
- REQ-2: Classifier losses `squared_hinge` and `perceptron` (the `Hinge`
  variant with `threshold=0.0`).
- REQ-3: Regressor losses `squared_error`, `huber`, `epsilon_insensitive`, and
  `squared_epsilon_insensitive` and their gradients match `_sgd_fast` exactly.
- REQ-4: L2 penalty update — the `wscale` multiplicative shrink
  `w *= max(0, 1 - (1-l1_ratio)*eta*alpha)` applied before the gradient add.
- REQ-5: `penalty ∈ {l1, elasticnet}` with `l1_ratio` via the Tsuruoka
  cumulative-penalty truncated-gradient (`u`/`q` vector).
- REQ-6: `constant` and `invscaling` learning-rate schedules.
- REQ-7: `optimal` schedule with the `t0` offset
  `eta = 1/(alpha*(optimal_init + t - 1))`.
- REQ-8: `adaptive` schedule — `eta /= 5` triggered by
  `n_iter_no_change` consecutive non-improving epochs (by `tol`).
- REQ-9: Constructor parameter defaults match sklearn per estimator
  (classifier `learning_rate='optimal'`, `eta0=0.0`, `power_t=0.5`;
  `epsilon=0.1`; `penalty`, `l1_ratio`, etc.).
- REQ-10: Convergence — `best_loss` tracking with `n_iter_no_change` on the
  per-epoch **sum** loss vs `best_loss - tol*train_count`.
- REQ-11: `fit_intercept` parameter (currently the intercept is always fit).
- REQ-12: `shuffle` flag (currently always shuffles).
- REQ-13: `early_stopping` + `validation_fraction` + validation-score callback.
- REQ-14: `average`/ASGD (averaged coefficients and intercept).
- REQ-15: `class_weight` (`"balanced"` / dict) → per-class sample weighting.
- REQ-16: `partial_fit` semantics — single-epoch incremental update that
  matches a `_partial_fit` call (`t_` carry, weight reuse, class registration).
- REQ-17: Multiclass one-vs-all (`_fit_multiclass`): one binary classifier per
  class, `predict` = argmax of decision functions.
- REQ-18: `SGDOneClassSVM` estimator (`offset_`, `nu`, one-class update with
  `intercept_update -= 2*eta*alpha`).
- REQ-19: Anti-pattern cleanup — remove `unreachable!()` in `compute_lr` and
  `.unwrap()` calls in the kernel (`F::from(...).unwrap()`, `1e18`/`18.0`
  constants), per R-APG-1 / R-CODE-2.
- REQ-20: Migrate the compute substrate from `ndarray` to ferray
  (`ferray-core` arrays, `ferray::random` shuffle), per R-SUBSTRATE.

## Acceptance criteria

- AC-1 (REQ-1): `Hinge`/`LogLoss`/`ModifiedHuber`/`SquaredError` `loss` and
  `gradient` match `_sgd_fast.LossFunction.py_loss`/`py_dloss` from the live
  oracle at a grid of `(y∈{-1,1}, p)` to 1e-12.
- AC-3 (REQ-3): same for `SquaredError`/`Huber`/`EpsilonInsensitive`; plus a
  `SquaredEpsilonInsensitive` variant once added.
- AC-4 (REQ-4): a single deterministic SGD step with `alpha>0` shrinks every
  weight coordinate by `1-eta*alpha` (not just touched features) before the
  gradient — verified against a hand-computed `wscale` step.
- AC-6 (REQ-6): `compute_lr(Constant, eta0=0.1) == 0.1`;
  `compute_lr(InvScaling, eta0, t, power_t) == eta0 / t^power_t` to 1e-12.
- AC-7 (REQ-7): `compute_lr(Optimal, alpha=1e-4)` at `t=1` equals
  `initial_eta0` (i.e. `eta = 1/(alpha*optimal_init)`), matching a live
  `Log().dloss(1,-typw)`-derived oracle value — the current
  `1/(alpha*t)` form fails this.
- AC-9 (REQ-9): `SGDClassifier::new()` reports `learning_rate=Optimal`,
  `eta0=0.0`, `power_t=0.5`, `epsilon=0.1` (parameters that must first exist).
- AC-10 (REQ-10): on a fixed deterministic stream (`shuffle=False`), the epoch
  at which fit stops matches sklearn's `best_loss`/`n_iter_no_change` rule.
- AC-17 (REQ-17): multiclass `predict` returns argmax-of-decision labels for a
  3-class separable set.
- AC-18 (REQ-18): `SGDOneClassSVM().fit(X).predict(X)` returns `{-1,+1}` matching
  the live oracle's `offset_`/`decision_function` sign convention.

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (classifier losses hinge/log/modified_huber/squared_error) | SHIPPED | impl `struct Hinge`, `struct LogLoss`, `struct ModifiedHuber`, `struct SquaredError` (`Loss` impls in `sgd.rs`) match `_sgd_fast.pyx.tp`: Hinge `loss`/`gradient` mirror `:216-226` (`thr-z`/`-y` for `z<=thr`); LogLoss mirrors `:267-283` with the same `±18` overflow guard and `-y·exp(-z)/(1+exp(-z))` gradient; ModifiedHuber mirrors `:178-194` (`(1-z)^2` mid-region, `-4z`/`-4y` tail, `-2y(1-z)` mid-gradient); SquaredError mirrors `:291-295` (`0.5(p-y)^2`, grad `p-y`). The Hinge `gradient` boundary was corrected to the NON-strict `margin <= 1` (`:224` `if z <= threshold: return -y`), so at the exact margin `z==1` the gradient is `-y` not `0` (closed #539). Consumed in production via `fn dispatch_train_binary in sgd.rs` → `fn fit (Fit for SGDClassifier)` → `impl PipelineEstimator for SGDClassifier in sgd.rs` (`Pipeline::fit` boxes it as `FittedPipelineEstimator`). Tests: `test_hinge_loss_*`, `test_log_loss_*`, `test_modified_huber_loss`, `test_squared_error_loss`, divergence `sgd_hinge_gradient_boundary` (live-oracle Hinge `py_dloss`). |
| REQ-2 (squared_hinge, perceptron) | NOT-STARTED | open prereq blocker #523. `enum ClassifierLoss in sgd.rs` has only `{Hinge, Log, SquaredError, ModifiedHuber}`; sklearn `loss_functions` (`:509-519`) also lists `"squared_hinge"` (`SquaredHinge`, `:232-258`) and `"perceptron"` (`Hinge` with `threshold=0.0`, `:512`). No `SquaredHinge` type and no per-loss threshold parameter exist. |
| REQ-3 (regressor losses incl. squared_epsilon_insensitive) | NOT-STARTED | open prereq blocker #524. `struct SquaredError`/`Huber`/`EpsilonInsensitive` (`sgd.rs`) match `_sgd_fast` `:291-295`/`:315-331`/`:348-358` exactly, but `enum RegressorLoss in sgd.rs` lacks `SquaredEpsilonInsensitive` (`_sgd_fast.pyx.tp:364-387`, `loss(y,p)=max(0,|y-p|-eps)^2`, `dloss` `±2(·-eps)`). Because the routed parity op set is incomplete (one sklearn regressor loss missing), REQ-3 as scoped is NOT-STARTED; the three shipped losses are individually correct (see REQ-1's pattern). |
| REQ-4 (L2 penalty update = wscale shrink) | SHIPPED | `fn train_binary_sgd`/`train_regressor_sgd in sgd.rs` now apply the CLAMPED multiplicative shrink `shrink = max(0, 1 - eta*alpha)` to the whole weight vector, then `w_j = w_j*shrink - eta*grad*x_j` (the gradient add), mirroring `w.scale(max(0, 1 - (1-l1_ratio)*eta*alpha))` with `l1_ratio=0` for pure L2 (`_sgd_fast.pyx.tp:632-635`). The intercept is left unregularized (`*intercept -= eta*grad`). When `eta*alpha > 1` the factor clamps to 0 (zeroing weights) instead of sign-flipping — the genuine observable divergence (closed #525). Consumer: `Fit for SGDRegressor`/`SGDClassifier` → `PipelineEstimator`. Test: divergence `sgd_l2_wscale_clamp` (live oracle, eta0=1.0/alpha=2.0 → coef [2.0,-1.0]). |
| REQ-5 (l1/elasticnet + l1_ratio) | SHIPPED | `enum Penalty {L2, L1, ElasticNet}` plus `pub penalty`/`pub l1_ratio` fields on `SGDClassifier`/`SGDRegressor` (`sgd.rs`) with `fn with_penalty`/`fn with_l1_ratio` builders (defaults `L2`/`0.15`, mirroring `_stochastic_gradient.py:1231-1256`). `fn train_binary_sgd`/`train_regressor_sgd` derive the effective `l1_ratio` via `fn effective_l1_ratio` (`L2->0.0`, `L1->1.0`, `ElasticNet->l1_ratio`, `_sgd_fast.pyx.tp:558-561`), apply the L2 shrink `max(0, 1-(1-eff)*eta*alpha)` to the whole weight vector BEFORE the gradient add (`:632-635`), then run the Tsuruoka cumulative-penalty L1 truncation AFTER — a fit-persistent scalar `u += eff*eta*alpha` and per-feature `q` vector pushing each coordinate toward 0 (`:656-658`, `l1penalty` `:750-778`, with `wscale=1`). `u`/`q` are allocated once before the epoch loop and persist across all epochs/samples of a `fit`. Consumer: `Fit for SGDRegressor`/`SGDClassifier` -> `impl PipelineEstimator`. Tests: divergence `sgd_l1_truncated_gradient` (live oracle, eta0=0.1/alpha=0.1/max_iter=3 -> coef [0.9204, -0.4452], intercept 0.4752) and `sgd_elasticnet_l1_ratio` (l1_ratio=0.3 -> coef [0.92340705, -0.45723495], intercept 0.4712037). Closed #526. partial_fit+l1: `u`/`q` reset per `partial_fit` call, which MATCHES sklearn (re-allocates `q`/`u` per `_plain_sgd`, `:551-556`; only `t_` is carried). |
| REQ-6 (constant + invscaling schedules) | SHIPPED | impl `fn compute_lr in sgd.rs`: `Constant => eta0` mirrors `_sgd_fast.pyx.tp:479` (`eta = eta0`); `InvScaling => eta0 / t^power_t` mirrors `:593-594` (`eta = eta0 / pow(t, power_t)`). Production consumer: selected per-step in `fn train_binary_sgd`/`fn train_regressor_sgd in sgd.rs`, reached through `Fit`→`PipelineEstimator`. Tests: `test_constant_lr`, `test_invscaling_lr` (1e-10). |
| REQ-7 (optimal schedule t0 init) | SHIPPED | `fn compute_lr in sgd.rs` Optimal arm is now `1/(alpha*(optimal_init + t - 1))`, with `optimal_init` supplied by `fn optimal_init in sgd.rs`: `typw=sqrt(1/sqrt(alpha))`, `initial_eta0=typw/max(1,|gradient(1,-typw)|)`, `optimal_init=1/(initial_eta0*alpha)` (`_sgd_fast.pyx.tp:565-570,592`). The dloss mapping uses ferrolearn `Loss::gradient(y_true=1.0, y_pred=-typw)` (= sklearn cython `dloss(self, y, p)` with `y=1, p=-typw`), absolute-valued to match `max(1, dloss)`. `optimal_init` is computed once per fit before the epoch loop and threaded through `compute_lr` (closed #527). At `t=1` `eta = 1/(alpha*optimal_init) = initial_eta0`. Consumer: `fn train_binary_sgd`/`train_regressor_sgd`. Tests: `test_optimal_lr`, `test_optimal_init_matches_oracle`, divergence `sgd_optimal_schedule_t0_offset` (full-fit live oracle to 1e-9). |
| REQ-8 (adaptive schedule ÷5 + n_iter_no_change trigger) | NOT-STARTED | open prereq blocker #528. `fn train_binary_sgd in sgd.rs` divides `current_eta` **by 2** (`current_eta / F::from(2.0)`) after **5 consecutive epochs with `epoch_loss >= prev_loss`** (mean loss). sklearn divides **by 5** (`eta = eta / 5`) when `no_improvement_count >= n_iter_no_change` under the `best_loss`/`sumloss`/`tol*train_count` rule (`_sgd_fast.pyx.tp:698-701`). Divisor and trigger both diverge. |
| REQ-9 (default params per estimator) | SHIPPED (classifier learning-rate/eta0/power_t defaults) | `fn new in sgd.rs` for `SGDClassifier` now sets `learning_rate=Optimal, eta0=0.0, power_t=0.5`, matching `SGDClassifier.__init__` (`_stochastic_gradient.py:1242-1244`); SGDRegressor's `invscaling/0.01/0.25` already match `:2042-2044` and were left unchanged. `fn schedule_requires_eta0 in sgd.rs` gates the `eta0>0` validation to constant/invscaling/adaptive only (`_stochastic_gradient.py:149-153`), so the default `optimal`+`eta0=0.0` classifier fits without error (closed #529). Consumer: `Fit for SGDClassifier`. Tests: divergence `sgd_classifier_default_learning_rate`, `test_sgd_classifier_default`, `test_sgd_classifier_optimal_eta0_zero_ok`. STILL missing fields entirely (own blockers): `penalty`, `l1_ratio`, `fit_intercept`, `shuffle`, `epsilon`, `n_iter_no_change`, `early_stopping`, `validation_fraction`, `average`, `warm_start`, `class_weight`, `C`. |
| REQ-10 (convergence: best_loss + n_iter_no_change + tol on sumloss) | NOT-STARTED | open prereq blocker #530. `fn train_binary_sgd in sgd.rs` breaks on the **first** epoch where `(prev_loss - epoch_loss).abs() < tol` using the **mean** loss and no `n_iter_no_change` patience and no `best_loss`. sklearn requires `sumloss > best_loss - tol*train_count` for `n_iter_no_change` consecutive epochs against a tracked `best_loss` on the **sum** loss (`_sgd_fast.pyx.tp:688-707`). |
| REQ-11 (fit_intercept) | NOT-STARTED | open prereq blocker #531. `SGDClassifier`/`SGDRegressor` (`sgd.rs`) have no `fit_intercept` field; `fn train_binary_sgd`/`train_regressor_sgd` always update `*intercept`. sklearn gates the intercept update on `fit_intercept == 1` (`_sgd_fast.pyx.tp:639`; param at `_stochastic_gradient.py:104`). |
| REQ-12 (shuffle flag) | NOT-STARTED | open prereq blocker #532. `fn train_binary_sgd in sgd.rs` always calls `indices.shuffle(&mut rng)` each epoch; sklearn gates per-epoch shuffle on `if shuffle:` (`_sgd_fast.pyx.tp:579-580`; param at `_stochastic_gradient.py:107`). No `shuffle` field exists. |
| REQ-13 (early_stopping + validation_fraction + n_iter_no_change-on-val-score) | SHIPPED (for the verifiable logic; the validation-split SELECTION is numpy-RNG-coupled so full fitted-coef parity is NOT oracle-verifiable — same barrier as `shuffle`) | `pub early_stopping`/`pub validation_fraction` fields + `fn with_early_stopping`/`with_validation_fraction` on `SGDClassifier`/`SGDRegressor in sgd.rs` (defaults `false`/`0.1`, `_stochastic_gradient.py:114-115`, constraints `:524-525`), threaded through `SGDHyper`. `fn validate_validation_fraction in sgd.rs` rejects values outside the open `(0,1)` interval (`Interval(Real, 0, 1, closed="neither")`, `:525`). The `Fit` path splits the data via `fn make_validation_split in sgd.rs` (seeded `StdRng`, stratified per class for the classifier mirroring `StratifiedShuffleSplit`/`ShuffleSplit`, `:257-310`) BEFORE the kernel, trains on the train subset (sklearn does NOT refit on full data), and the epoch-end `fn convergence_tail_score in sgd.rs` (`best_score` init `-inf`, `_sgd_fast.pyx.tp:678-687`) scores `fn r2_score`/`fn binary_accuracy in sgd.rs` on the held-out set, mirroring `_ValidationScoreCallback` = `est.score(X_val,y_val)` (`_stochastic_gradient.py:63-79`). `early_stopping=false` is byte-identical to the prior training-loss path. Consumer: `Fit for SGDClassifier`/`SGDRegressor` -> `PipelineEstimator`. Verified deterministically: the val-score helpers vs live `sklearn.metrics.r2_score`/`accuracy_score` (`test_validation_r2_matches_sklearn`/`test_validation_binary_accuracy_matches_sklearn`), the `(0,1)` constraint (`sgd_validation_fraction_invalid`), the behavioral early-stop (`sgd_early_stopping_stops_early`/`sgd_early_stopping_classifier_valid`). NOT verifiable: the held-out index SELECTION (numpy MT vs `StdRng`), the same PRNG barrier as `shuffle`. Closes #533. |
| REQ-14 (average / ASGD) | NOT-STARTED | open prereq blocker #534. No `average` field and no `_average_coef`/`_average_intercept`/`w.add_average` analog (`_sgd_fast.pyx.tp:646-654`; `_stochastic_gradient.py:118`). `FittedSGDClassifier`/`FittedSGDRegressor` store only the standard weights. |
| REQ-15 (class_weight) | NOT-STARTED | open prereq blocker #535. No `class_weight` field; `fn train_binary_sgd` applies no per-class `weight_pos`/`weight_neg` (`_sgd_fast.pyx.tp:599-602,630`) and `sample_weight` is unsupported. sklearn computes `_expanded_class_weight` via `compute_class_weight` (`_stochastic_gradient.py:528`). |
| REQ-16 (partial_fit semantics) | SHIPPED | impl `fn partial_fit (PartialFit for SGDClassifier / FittedSGDClassifier / SGDRegressor / FittedSGDRegressor in sgd.rs)` sets `hyper.max_iter = 1` and carries the global step counter `self.t` across calls (`fn partial_fit_ova`, `dispatch_train_*` with `initial_t = self.t`), mirroring sklearn's single-epoch `_partial_fit` with `t_` carry (`_stochastic_gradient.py:581-674,1462-1521`). Production consumer: `PartialFit` is a `ferrolearn-core` trait re-exported and registered there (`ferrolearn-core/src/traits.rs` REQ-4 cites `SGDClassifier`/`FittedSGDClassifier` as the trait's consumers). Tests: `test_sgd_classifier_partial_fit`, `test_sgd_classifier_partial_fit_chain`, `test_sgd_regressor_partial_fit*`. SHIPPED for the *single-epoch incremental* contract; class re-registration across `partial_fit` calls with new labels (sklearn `_check_partial_fit_first_call`, `classes=` arg) is NOT covered and is folded into REQ-15/#535's class-handling scope. |
| REQ-17 (multiclass one-vs-all) | SHIPPED | impl `fn fit_ova in sgd.rs` trains one binary problem per class (`label==cls ? +1 : -1`) for `n_classes > 2`, and `fn predict (Predict for FittedSGDClassifier in sgd.rs)` returns `argmax_c (x·w_c + b_c)`, mirroring sklearn `_fit_multiclass` OvA (`_stochastic_gradient.py:788-844`) and `LinearClassifierMixin` argmax decision. Production consumer: `Fit for SGDClassifier` → `impl PipelineEstimator for SGDClassifier in sgd.rs`. Test: `test_sgd_classifier_multiclass` (3-class separable, ≥6/9 correct). DIVERGENCE NOTE (within SHIPPED scope): sklearn's binary case keeps a single OvA vector `class[1]`-vs-rest and `predict` thresholds the decision at 0 — ferrolearn's binary branch matches this; but sklearn's *seed-per-class* (`randint(MAX_INT, size=n_classes)`, `:802-804`) is not reproduced (non-parityable per the RNG boundary). The label-set + argmax structure is the SHIPPED contract. |
| REQ-18 (SGDOneClassSVM) | SHIPPED | `pub struct SGDOneClassSVM<F>` + `fn fit_one_class` + `impl Fit<Array2<F>, ()> for SGDOneClassSVM in sgd.rs` solve the linear One-Class SVM via the reused `fn train_binary_sgd` Hinge kernel: `y = ones(n)`, `alpha = nu/2` (`_stochastic_gradient.py:2588`), `penalty = L2`, `l1_ratio = 0`, `one_class = true` (`:2262-2289,2312`); the SGD intercept is initialized to `b = 1` (the offset init is `0`, so `intercept = 1 - 0`, `:2238,2325`), and after the fit `offset_ = 1 - b` (`:2377`). `fn train_binary_sgd` gained the one-class intercept term: when `hyper.one_class` the gated update is `intercept_update = -eta*grad - 2*eta*alpha`, mirroring `if one_class: intercept_update -= 2.*eta*alpha` (`_sgd_fast.pyx.tp:641-642`); the new `one_class` field on `SGDHyper` defaults to `false` (via `fn clf_hyper`/`fn reg_hyper`), leaving the classifier/regressor intercept update byte-identical. `pub struct FittedSGDOneClassSVM<F>` exposes `fn coef`/`fn offset`/`fn decision_function` (`X·coef_ - offset_`, `:2622`)/`fn score_samples` (`+ offset_ = X·coef_`, `:2639`) and `impl Predict<Array2<F>>` returning `Array1<isize>` of `+1`/`-1` (`(decision >= 0) ? +1 : -1`, `:2655-2657`). `nu` is validated to `(0, 1]` (`Interval(Real, 0.0, 1.0, closed="right")`, `:2236`). Consumer: `pub use sgd::{SGDOneClassSVM, FittedSGDOneClassSVM}` from `ferrolearn-linear/src/lib.rs` (the public-API boundary, matching the `SGDClassifier`/`SGDRegressor` re-export). Tests: divergence `sgd_one_class_svm_decision` (live oracle nu=0.5/constant/eta0=0.05/max_iter=10/shuffle=false: coef `[0.009883660184666337, 0.009883660184666337]`, offset `1.1102230246251565e-16`, to 1e-7) and `sgd_one_class_svm_predict` (nu=0.8/eta0=0.1/max_iter=15: coef `[0.20020636453962284, 0.12292535592963398]`, offset `0.10000000000000009`, predict `[1,-1,1,-1]`). `_stochastic_gradient.py:2084-2668`, `_sgd_fast.pyx.tp:639-644`. Closes #536. |
| REQ-19 (anti-pattern cleanup) | SHIPPED | The `_Phantom` arm of `fn compute_lr in sgd.rs` returns `eta0` (the unreach macro was removed earlier), and a private module-level `fn cst<F: Float>(x: f64) -> F` (`F::from(x).unwrap_or_else(F::zero)`, defined after the imports) replaces EVERY production panicking constant-conversion: 23 call sites across the `Loss` impls (LogLoss `18/-18/1e18`, SquaredError/Huber `0.5`, ModifiedHuber `4/-2`, SquaredHinge/SquaredEpsilonInsensitive `2`) and the `SGDClassifier`/`SGDRegressor`/`SGDOneClassSVM` `::new` defaults (`0.0`/`0.0001`/`0.15`/`1e-3`/`0.5`/`0.25`/`0.01`), plus the intercept/one-class `2.0` and `nu/2` sites. No numeric literal changed -> byte-identical for f32/f64 (the 25 `divergence_sgd_fit` tests stay green). A grep for the panicking constructs over `sgd.rs` confirms only the `#[cfg(test)] mod tests` block (and doctests) retains them. R-APG-1 / R-CODE-2. Closes #537. (Runtime `F::from(<usize>)` conversions and the deliberately-non-zero-fallback constants `1e12`/`1e-6`/`5.0` already used `unwrap_or_else` and were gate-compliant.) |
| REQ-20 (ferray substrate migration) | NOT-STARTED | open prereq blocker #538. `sgd.rs` imports `ndarray::{Array1, Array2, ScalarOperand}` and `rand::rngs::StdRng` for all computation and shuffling — the wrong substrate per R-SUBSTRATE-1. Destination: `ferray-core` arrays and `ferray::random` for the per-epoch permutation. |

## Architecture

### sklearn (the contract)

`BaseSGD` (`_stochastic_gradient.py:82`) holds the full parameter surface
(`penalty`, `alpha`, `l1_ratio`, `fit_intercept`, `max_iter`, `tol`, `shuffle`,
`epsilon`, `learning_rate`, `eta0`, `power_t`, `early_stopping`,
`validation_fraction`, `n_iter_no_change`, `warm_start`, `average`; `:96-139`).
`BaseSGDClassifier` (`:508`) and `BaseSGDRegressor` (`:1400`) add `loss_functions`
dicts and `class_weight`/regressor specifics; the concrete `SGDClassifier`
(`:951`), `SGDRegressor` (`:1789`), `SGDOneClassSVM` (`:2084`) fix per-estimator
defaults. `_fit` allocates parameter memory, validates params
(`_more_validate_params`, `:145`: e.g. `optimal` requires `alpha>0`), splits a
validation set if `early_stopping`, builds a `SequentialDataset` via
`make_dataset`, and calls `_plain_sgd32/64`. The numerical engine is entirely
in `_sgd_fast.pyx.tp` `_plain_sgd` (see Algorithm). `t_` starts at `1.0`
(`:645,723`) and accumulates `n_iter_ * n_samples` across fits.

### ferrolearn (what exists)

`SGDClassifier<F> { loss: ClassifierLoss, learning_rate, eta0, alpha, max_iter,
tol, random_state, power_t }` and the structurally-identical `SGDRegressor<F>`
(`sgd.rs`) — a *reduced* parameter set. `Fit` zeroes `w`/`b`, then `fn fit_ova`
(classifier) or `fn dispatch_train_regressor` (regressor) runs the epoch loop in
`fn train_binary_sgd` / `fn train_regressor_sgd`:

- per epoch: shuffle indices (always), iterate, `t += 1` (so the first sample
  sees `t=initial_t+1`), compute `eta` via `fn compute_lr` (or held `current_eta`
  for adaptive), `y_pred = b + Σ w_j x_j`, `grad = loss.gradient(y, y_pred)`,
  inline L2 update `w_j -= eta*(grad*x_j + alpha*w_j)`, `b -= eta*grad`;
- per epoch end: mean `epoch_loss`; break if `|prev_loss - epoch_loss| < tol`;
  adaptive halving after 5 non-decreasing epochs.

`FittedSGDClassifier<F>` holds `weight_matrix: Vec<Array1<F>>` (one row per OvA
sub-problem), `intercepts: Vec<F>`, `classes: Vec<usize>`, the loss, hyper, and
global `t`; `Predict` thresholds at 0 (binary) or argmax (multiclass).
`FittedSGDRegressor<F>` holds a single `weights`/`intercept` and predicts
`X·w + b`. Both expose `HasCoefficients` and implement `PipelineEstimator` /
`FittedPipelineEstimator` (the in-crate non-test production consumer; the
estimators are re-exported from `ferrolearn-linear/src/lib.rs` as the public API
boundary). There is **no** `ferrolearn-python` registration of SGD.

### Divergence map (verified against `_sgd_fast.pyx.tp`)

| piece | sklearn | ferrolearn | REQ |
|---|---|---|---|
| L2 update | global `wscale` shrink before grad (`:632-635`) | clamped `max(0,1-eta*alpha)` shrink then grad add (FIXED #525) | REQ-4 ✓ |
| optimal η | `1/(alpha*(optimal_init+t-1))` (`:591-592`) | `1/(alpha*(optimal_init+t-1))` via `fn optimal_init` (FIXED #527) | REQ-7 ✓ |
| adaptive | `÷5`, `n_iter_no_change` trigger (`:698-701`) | `÷2`, 5-epoch mean-loss trigger | REQ-8 |
| converge | `best_loss`+`n_iter_no_change`+sumloss (`:688-707`) | first-epoch mean-loss delta | REQ-10 |
| clf defaults | optimal/0.0/0.5 (`:1242-1244`) | optimal/0.0/0.5 (FIXED #529) | REQ-9 ✓ |
| dloss clip | `±1e12` (`:616-619`) | none | REQ-10/cleanup |
| penalty | l2/l1/elasticnet (`:55`) | l2/l1/elasticnet via `u`/`q` truncated gradient (FIXED #526) | REQ-5 ✓ |
| OneClassSVM | full estimator (`:2084`) | `SGDOneClassSVM`/`FittedSGDOneClassSVM` via reused Hinge kernel + `one_class` intercept term (SHIPPED #536) | REQ-18 ✓ |

## Verification

Commands that establish the SHIPPED claims (run at baseline `c2f944e`):

- `cargo test -p ferrolearn-linear sgd` — the unit suite in `sgd.rs`
  (`test_hinge_loss_*`, `test_log_loss_*`, `test_modified_huber_loss`,
  `test_squared_error_loss`, `test_constant_lr`, `test_invscaling_lr`,
  `test_sgd_classifier_multiclass`, `test_sgd_*_partial_fit*`). Pins
  REQ-1/REQ-6/REQ-16/REQ-17.
- Live oracle for the loss formulas (REQ-1, re-pin per R-CHAR-3):
  `python3 -c "from sklearn.linear_model._sgd_fast import Log, ModifiedHuber, Hinge, SquaredLoss; l=Log(); print(l.py_loss(0.0,1.0), l.py_dloss(0.0,1.0))"`
  compared against `LogLoss.loss/gradient(1.0, 0.0)`.
- Live oracle exposing the NOT-STARTED gaps:
  - REQ-7: `python3 -c "import numpy as np; from sklearn.linear_model._sgd_fast import Log; a=1e-4; typw=np.sqrt(1/np.sqrt(a)); e0=typw/max(1.0, Log().py_dloss(-typw,1.0)); print(1.0/(e0*a))"` gives `optimal_init`; ferrolearn's `compute_lr(Optimal,..,t=1)` ignores it.
  - REQ-9: `python3 -c "from sklearn.linear_model import SGDClassifier as C; m=C(); print(m.learning_rate, m.eta0, m.power_t)"` → `optimal 0.0 0.5` vs ferrolearn `InvScaling 0.01 0.25`.
  - REQ-18: `python3 -c "from sklearn.linear_model import SGDOneClassSVM"` imports; no ferrolearn counterpart.

REQ-2..5, REQ-7..15, REQ-18..20 have no green verification against the sklearn
contract and are NOT-STARTED, gated on blockers #523–#538. A formula or schedule
REQ is SHIPPED only when its `sgd.rs` symbol matches `_sgd_fast` exactly, has a
unit test, and is reachable from the `PipelineEstimator`/`PartialFit`
production consumer; full random-shuffle fitted-weight parity is explicitly out
of scope (see Parity boundary).

## Blockers to open

- #523 — Blocker for REQ-2 of sgd: `SGDClassifier` lacks `squared_hinge` and `perceptron` losses (`SquaredHinge`, `Hinge` threshold=0.0).
- #524 — Blocker for REQ-3 of sgd: `RegressorLoss` lacks `squared_epsilon_insensitive` (`SquaredEpsilonInsensitive`).
- #525 — Blocker for REQ-4 of sgd: L2 penalty is inline per-feature, not the global `wscale` shrink `max(0, 1-(1-l1_ratio)*eta*alpha)` before the gradient add.
- #526 — Blocker for REQ-5 of sgd: no `penalty`/`l1_ratio`; missing `l1`/`elasticnet` truncated-gradient (`u`/`q` cumulative penalty).
- #527 — Blocker for REQ-7 of sgd: `optimal` schedule omits the `optimal_init` t0 offset (`eta=1/(alpha*(optimal_init+t-1))`).
- #528 — Blocker for REQ-8 of sgd: `adaptive` divides eta by 2 on a 5-epoch mean-loss trigger; sklearn divides by 5 on the `n_iter_no_change`/`best_loss` trigger.
- #529 — Blocker for REQ-9 of sgd: `SGDClassifier` defaults (`InvScaling/0.01/0.25`) diverge from sklearn (`optimal/0.0/0.5`); many params (`penalty`, `l1_ratio`, `fit_intercept`, `shuffle`, `epsilon`, `n_iter_no_change`, `early_stopping`, `validation_fraction`, `average`, `warm_start`, `class_weight`, `C`) missing.
- #530 — Blocker for REQ-10 of sgd: convergence uses first-epoch mean-loss delta; sklearn uses `best_loss` + `n_iter_no_change` + `sumloss > best_loss - tol*train_count`; missing dloss `±1e12` clip.
- #531 — Blocker for REQ-11 of sgd: no `fit_intercept` param; intercept is always fit.
- #532 — Blocker for REQ-12 of sgd: no `shuffle` flag; data is always shuffled per epoch.
- #533 — Blocker for REQ-13 of sgd: no `early_stopping`/`validation_fraction`/validation-score callback.
- #534 — Blocker for REQ-14 of sgd: no `average`/ASGD (averaged coef/intercept).
- #535 — Blocker for REQ-15 of sgd: no `class_weight`/`sample_weight`/per-class `weight_pos`/`weight_neg`; also blocks `partial_fit` class re-registration.
- #536 — Blocker for REQ-18 of sgd: `SGDOneClassSVM` estimator entirely missing (offset_, nu, one-class intercept update) — acto-builder.
- #537 — Blocker for REQ-19 of sgd: `unreachable!()` in `compute_lr` and `.unwrap()` constants in production loss/lr kernels (R-APG-1 / R-CODE-2).
- #538 — Blocker for REQ-20 of sgd: migrate `sgd.rs` from `ndarray`/`StdRng` to the ferray substrate (`ferray-core`, `ferray::random`) per R-SUBSTRATE.
