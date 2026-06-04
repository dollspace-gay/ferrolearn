# MLPClassifier / MLPRegressor — Multi-layer Perceptron

<!--
tier: 3-component
status: draft
baseline-commit: 0d8690a4163e7f7d4725e89aae676ed1eec1092b
upstream-paths:
  - sklearn/neural_network/_multilayer_perceptron.py
  - sklearn/neural_network/_stochastic_optimizers.py
  - sklearn/neural_network/_base.py
-->

## Summary

`ferrolearn-neural/src/mlp.rs` mirrors scikit-learn's `sklearn.neural_network.MLPClassifier`
(`_multilayer_perceptron.py:762`) and `MLPRegressor` (`:1261`): feedforward neural networks
trained by mini-batch gradient descent with backpropagation. ferrolearn exposes the unfitted
builders `MLPClassifier<F>`/`MLPRegressor<F>`, the fitted `FittedMLPClassifier<F>`/
`FittedMLPRegressor<F>`, an `Activation` enum (`Relu`/`Tanh`/`Logistic`/`Identity`) and a
`Solver<F>` enum (`Sgd`/`Adam`).

This doc audits the CURRENT `mlp.rs` against the sklearn 1.5.2 contract. The estimator is
**RNG-coupled**: fitted weights/predictions depend on random weight init (Glorot/He
`_init_coef`, `_multilayer_perceptron.py:409-424`), the per-epoch minibatch shuffle, and the
early-stopping `train_test_split` — all driven by numpy's `RandomState` (Mersenne-Twister),
which cannot bit-match Rust's `StdRng`. Exact post-fit values are therefore an **R-DEFER-3
carve-out** (NOT-STARTED, no failing test — the test would be unwinnable). The
**deterministic** pieces — activation functions, output transforms, the forward pass given
fixed weights, the loss formulas, and the optimizer update equations — ARE oracle-pinnable.
The critic pins these by constructing a fitted net with KNOWN weights (or comparing the math
directly) against the live sklearn 1.5.2 oracle.

The audit finds the deterministic *forward/activation/output* surface matches sklearn, but
several deterministic *training* details diverge (Adam epsilon placement, SGD nesterov,
the L2 term in the reported loss), several constructor params are missing, several fitted
attributes are absent, `lbfgs` is unsupported, and a `clippy::collapsible_if` lint blocks the
crate gauntlet. There is **no non-test production consumer** (no `ferrolearn-python`
registration; `lib.rs` only re-exports), so even the deterministic-correct REQs cannot claim a
strict R-HONEST-2 SHIPPED. This doc under-claims by design (R-HONEST-3).

## Upstream cite

scikit-learn 1.5.2 (commit 156ef14):
- `_base.py` — `ACTIVATIONS` (`identity`/`tanh`/`logistic`=`expit`/`relu`=`max(X,0)`/`softmax`)
  `_base.py:69-75`; `DERIVATIVES` `:148-153`; `squared_loss` (`mean/2`) `:156-172`;
  `log_loss` `:175-200`; `binary_log_loss` `:203-228`; `LOSS_FUNCTIONS` `:231-235`.
- `_multilayer_perceptron.py` — `_parameter_constraints` `:64-97`; abstract `__init__`
  signature `:99-150`; `_forward_pass` `:161-184`; `_forward_pass_fast` `:186-221`;
  `_compute_loss_grad` (`grad = X.T@delta + alpha*W`, then `/= n_samples`) `:223-235`;
  `_backprop` (loss + `0.5*alpha*sum(W^2)/n_samples`; `deltas[last]=activations[-1]-y`)
  `:287-364`; `_initialize` / `out_activation_` selection `:366-407`; `_init_coef`
  (Glorot uniform, factor 2 for logistic else 6, bias drawn from same uniform) `:409-424`;
  `_fit_stochastic` (split / batching / convergence) `:549-702`;
  `_update_no_improvement_count` `:704-731`; `MLPClassifier.__init__` defaults `:1041-1067`;
  `predict` / `_predict` / `predict_proba` `:1149-1255`; `MLPRegressor` `:1261`.
- `_stochastic_optimizers.py` — `SGDOptimizer._get_updates` (momentum + nesterov, default
  `nesterov=True`) `:168-194`; `SGDOptimizer.iteration_ends` (`invscaling`) `:137-150`;
  `trigger_stopping` (`adaptive` lr/5) `:152-166`; `AdamOptimizer._get_updates`
  (`lr_t = lr * sqrt(1-b2^t)/(1-b1^t)`, `upd = -lr_t * m/(sqrt(v)+eps)`) `:255-287`.

## Requirements

- REQ-1 (activations): `relu`/`tanh`/`logistic`/`identity` match `ACTIVATIONS`
  (`_base.py:69-75`): `relu=max(0,x)`, `tanh`, `logistic=expit`, `identity=x`. Deterministic.
- REQ-2 (output transform + loss): classifier uses softmax (multiclass) / logistic (binary)
  output with cross-entropy loss; regressor uses identity output with squared loss
  (`mean/2`); loss adds the L2 term `0.5*alpha*sum(W^2)/n_samples` (`_backprop`,
  `_multilayer_perceptron.py:330-338`; `_base.py:172`, `:200`, `:226`). Deterministic.
- REQ-3 (forward pass / predict): given fixed weights/biases, classifier `predict_proba`/
  `predict` and regressor `predict` equal sklearn `_forward_pass_fast`
  (`_multilayer_perceptron.py:186-221`); `predict_proba` rows sum to 1; binary expands
  `(n,1)`→`[1-p, p]` (`:1252-1253`); `classes_` is sorted-unique; `predict` returns the
  argmax label. Deterministic.
- REQ-4 (optimizer update formulas): Adam and SGD update rules match
  `_stochastic_optimizers.py` given fixed gradients — Adam
  `lr_t = lr*sqrt(1-b2^t)/(1-b1^t)`, `upd = -lr_t*m/(sqrt(v)+eps)` (`:278-287`); SGD with
  momentum AND nesterov (default `nesterov=True`, `:168-194`). Deterministic.
- REQ-5 (exact fitted weights/predictions after fit): random init + minibatch shuffle +
  early-stopping split, numpy MT vs Rust `StdRng`. **RNG carve-out (R-DEFER-3)** — also
  covers the bias-init divergence (sklearn draws biases from the same uniform; ferrolearn
  zeros) and the Glorot factor (sklearn 2 for logistic else 6; ferrolearn always 6). NO
  failing test; the meaningful signal is that the net LEARNS.
- REQ-6 (solver coverage): sklearn `{lbfgs, sgd, adam}` (`_parameter_constraints`,
  `_multilayer_perceptron.py:70`); ferrolearn `Solver` has only `Sgd`/`Adam`. `lbfgs` absent.
- REQ-7 (SGD learning-rate schedule + nesterov): sklearn `learning_rate` ∈
  `{constant, invscaling, adaptive}` + `power_t`, `momentum`, `nesterovs_momentum`
  (`_stochastic_optimizers.py:137-194`). ferrolearn SGD: constant lr + momentum only, no
  schedule, no nesterov.
- REQ-8 (convergence criterion): sklearn stops when loss/val-score fails to improve by `tol`
  for `n_iter_no_change` (default 10) consecutive epochs (`_update_no_improvement_count`
  + `_fit_stochastic:664`); ferrolearn stops the FIRST epoch `|prev_loss - loss| < tol`
  (effectively a 1-epoch / different criterion). Divergence; `n_iter_no_change` is not a param.
- REQ-9 (missing constructor params): vs `MLPClassifier.__init__`
  (`_multilayer_perceptron.py:1041-1067`) ferrolearn lacks `learning_rate` (schedule),
  `learning_rate_init` (folded into `Solver`), `power_t`, `shuffle`, `warm_start`,
  `n_iter_no_change`, `momentum`/`nesterovs_momentum` (SGD-only, no nesterov), `beta_1`/
  `beta_2`/`epsilon` exposed only inside `Solver::Adam`, `max_fun`, `verbose`.
- REQ-10 (fitted attributes): sklearn exposes `coefs_`, `intercepts_`, `loss_`,
  `loss_curve_`, `best_loss_`, `n_iter_`, `t_`, `classes_`, `n_layers_`, `n_outputs_`,
  `out_activation_`, `validation_scores_`/`best_validation_score_` (`_initialize`,
  `_multilayer_perceptron.py:366-407`). ferrolearn exposes `classes()`/`n_classes()`
  (classifier) and `n_layers()`; weights/biases are private; the rest are absent.
- REQ-11 (predict_proba): softmax normalization (multiclass) and binary logistic with
  `[1-p, p]` expansion match sklearn given fixed weights (`predict_proba`,
  `_multilayer_perceptron.py:1232-1255`). Deterministic.
- REQ-12 (collapsible_if lint): `mlp.rs` `train_network`'s
  `if early_stopping { if let (Some(xv), Some(yv)) = ... }` triggers
  `clippy::collapsible_if` (let-chain collapse, stable from Rust 1.88) which fails the
  `-D warnings` gauntlet.
- REQ-13 (substrate): the unit must compute on the ferray substrate (`ferray-core` array +
  `ferray::random` sampling), not `ndarray` + `rand`/`rand_distr` (R-SUBSTRATE-1).
- REQ-14 (non-test production consumer): a real consumer (the canonical `ferrolearn-python`
  registration) must exist so the estimator is reachable as `import ferrolearn` surface
  (R-DEFER-1 / R-HONEST-2).

## Acceptance criteria

- AC-1 (REQ-1): `activate_inplace` on a fixed vector equals `ACTIVATIONS[name]` from a live
  sklearn call for relu/tanh/logistic/identity within f64 ULP tolerance (deterministic; must
  have a failing characterization test until met).
- AC-2 (REQ-2): the multiclass cross-entropy, binary cross-entropy, and squared (`mean/2`)
  losses computed in `train_network` equal `LOSS_FUNCTIONS[...]` on fixed inputs; the L2 term
  `0.5*alpha*sum(W^2)/n_samples` is present in the REPORTED loss (currently it is NOT — see
  REQ-2 diagnostic).
- AC-3 (REQ-3/REQ-11): given `coefs_`/`intercepts_` copied from a live sklearn-fitted model,
  `FittedMLPClassifier::predict_proba` / `FittedMLPRegressor::predict` equal
  `m.predict_proba` / `m.predict` on a fixed `X` within f64 ULP tolerance; `predict_proba`
  rows sum to 1; binary returns shape `(n,2)` ordered `[1-p, p]`.
- AC-4 (REQ-4): one Adam update from a fixed gradient + state equals sklearn's
  `AdamOptimizer._get_updates` (`-lr_t*m/(sqrt(v)+eps)`); one SGD-with-nesterov update equals
  `SGDOptimizer._get_updates` with `nesterov=True`. (Both currently DIVERGE — see diagnostics.)
- AC-5 (REQ-5): on a linearly separable 2-class problem with a fixed seed, train accuracy
  beats chance and the loss curve decreases (learning signal). Exact weights NOT compared.
- AC-6 (REQ-6): `MLPClassifier(solver='lbfgs').fit(...)` works in sklearn; ferrolearn has no
  `lbfgs` `Solver` variant.
- AC-7 (REQ-8): `MLPRegressor(n_iter_no_change=...)` exists in sklearn; ferrolearn stops on a
  single `|Δloss| < tol`, not after `n_iter_no_change` consecutive non-improving epochs.
- AC-8 (REQ-9/REQ-10): the listed sklearn `__init__` params and fitted attributes are
  enumerated; ferrolearn lacks them.
- AC-9 (REQ-12): `cargo clippy -p ferrolearn-neural --all-targets -- -D warnings` fails on
  `collapsible_if` at the `train_network` early-stopping block.
- AC-10 (REQ-13): `cargo tree -p ferrolearn-neural` shows `ndarray`/`rand`/`rand_distr`, no
  ferray crates in `mlp.rs`'s owned computation path.
- AC-11 (REQ-14): `grep -rn "MLPClassifier\|MLPRegressor" --include=*.rs` outside `src/mlp.rs`
  shows only `src/lib.rs` (re-export) and `tests/` — no non-test consumer.

## REQ status

Classification key: **DET** = deterministic (oracle-pinnable, the critic SHOULD pin a failing
characterization test from the live oracle); **RNG** = R-DEFER-3 carve-out (blocker, NO
failing test); **GAP** = feature/divergence (blocker + failing test if deterministic); **CQ**
= code-quality.

All REQs below are **NOT-STARTED**. Even the deterministic-correct ones (REQ-1, REQ-3, REQ-11)
fail R-HONEST-2 SHIPPED because (a) there is **no non-test production consumer** (REQ-14) and
(b) there is **no live-oracle characterization test** in the crate yet — the existing tests
check shape/range/learning only, never value-parity against sklearn (R-CHAR-1/R-CHAR-3).
Under-claim over overclaim (R-HONEST-3).

| REQ | Class | Status | Evidence |
|---|---|---|---|
| REQ-1 (activations) | DET | NOT-STARTED | `fn activate_inplace in mlp.rs` computes `Relu = max(0,x)`, `Tanh = tanh`, `Logistic = stable_sigmoid` (= `expit`), `Identity = no-op` — math matches `ACTIVATIONS` (`_base.py:69-75`) exactly. BUT: no value-parity test against the live oracle (existing `test_activate_*` are self-referential, expected values literal-computed in Rust, violating R-CHAR-3), and no non-test consumer (REQ-14). Blocker: needs a live-sklearn characterization test + a consumer. Pin under #1710. |
| REQ-2 (output + loss) | DET | NOT-STARTED | `train_network in mlp.rs` computes multiclass cross-entropy `-Σ y·ln(softmax)`, binary cross-entropy, and regression `Σ(diff²)/(2n)` (= `_base.py:172/200/226`) — formulas match. **Divergence:** the L2 penalty `0.5*alpha*Σ(W²)/n_samples` is added to the GRADIENT in `fn backward` (`+ alpha*W`) but is NOT added to the REPORTED loss, whereas sklearn adds it in `_backprop` (`_multilayer_perceptron.py:334-338`). The reported `loss_` / convergence value therefore diverges. Blocker: add the L2 term to the reported loss; pin a live-oracle loss-value test. #1710. |
| REQ-3 (forward / predict) | DET | NOT-STARTED | `fn forward_output in mlp.rs` (consumed by `FittedMLPClassifier::predict_proba` and `FittedMLPRegressor::predict`) computes `z = a@W + b`, hidden activation, then softmax/logistic/identity output — matches `_forward_pass_fast` (`_multilayer_perceptron.py:186-221`); verified numerically (relu→logistic on fixed weights → `[0.6661893, 0.4056446]` matching scipy `expit`). `predict` argmax + sorted-unique `classes` matches `_label_binarizer.inverse_transform`. NOT-STARTED: no value-parity test vs the live oracle, no non-test consumer (REQ-14). #1710. |
| REQ-4 (optimizer formulas) | GAP/DET | NOT-STARTED | **Adam DIVERGES:** `fn apply_adam_update in mlp.rs` uses `m_hat=m/(1-b1^t)`, `v_hat=v/(1-b2^t)`, `upd=-lr*m_hat/(sqrt(v_hat)+eps)`, placing `eps` outside the v bias-correction; sklearn uses `lr_t=lr*sqrt(1-b2^t)/(1-b1^t)`, `upd=-lr_t*m/(sqrt(v)+eps)` (`_stochastic_optimizers.py:278-287`). Verified numerically: t=1, g=[0.5,-0.2] gives sklearn `[-9.99999e-4, 9.99998e-4]` vs ferrolearn `[-9.99999e-4, 9.99999e-4]` — differ at ~1e-9 (and grow with eps relative magnitude). **SGD DIVERGES:** `fn apply_sgd_update` implements plain momentum `v=μv-lr·g; W+=v` with NO nesterov, but sklearn defaults `nesterovs_momentum=True` and applies the double-update (`:188-194`). Blockers: align Adam eps/bias-correction; add nesterov. Failing tests from fixed gradients. #1710. |
| REQ-5 (exact fitted weights) | RNG | NOT-STARTED | `fn fit` (classifier/regressor) seeds `StdRng` (`seed_from_u64` / `from_os_rng`) and inits via `fn xavier_init` (Glorot uniform, **factor always 6**) with **zero biases**; sklearn `_init_coef` uses factor 2 for `logistic` else 6 AND draws biases from the same uniform (`_multilayer_perceptron.py:412-424`). numpy MT vs Rust `StdRng` cannot bit-match. **R-DEFER-3 carve-out: blocker only, NO failing test.** The structural init divergences (zero bias, factor) ARE deterministic-documentable but the post-fit values remain RNG-coupled. Determinism-default divergence to DOCUMENT: ferrolearn `random_state=None` → `from_os_rng` (non-deterministic, like sklearn `None`). #1710. |
| REQ-6 (solver coverage: lbfgs) | GAP | NOT-STARTED | `enum Solver in mlp.rs` has only `Sgd`/`Adam`; sklearn allows `{lbfgs, sgd, adam}` (`_parameter_constraints`, `_multilayer_perceptron.py:70`; `_fit_lbfgs`, `:502-547`). `lbfgs` (L-BFGS-B full-batch quasi-Newton) is absent. Blocker: implement an `lbfgs` solver path (depends on a quasi-Newton optimizer in `ferrolearn-numerical`). #1710. |
| REQ-7 (lr schedule + nesterov) | GAP | NOT-STARTED | `Solver::Sgd { learning_rate, momentum }` is constant-lr momentum only. sklearn supports `learning_rate ∈ {constant, invscaling, adaptive}` with `power_t` (`iteration_ends`/`trigger_stopping`, `_stochastic_optimizers.py:137-166`) and `nesterovs_momentum` (default True). None present. Blocker: add `learning_rate` schedule + `power_t` + nesterov. #1710. |
| REQ-8 (convergence criterion) | GAP | NOT-STARTED | `train_network in mlp.rs` returns as soon as `|prev_loss - epoch_loss| < tol` for a SINGLE epoch and counts early-stopping patience at a hardcoded `patience = 10`; sklearn requires `_no_improvement_count > n_iter_no_change` (default 10) CONSECUTIVE non-improving epochs measured against `best_loss_`/`best_validation_score_` with the `+ tol` slack (`_update_no_improvement_count`, `_fit_stochastic:664`). `n_iter_no_change` is not a ferrolearn param. Diverges (1-epoch Δloss vs 10-epoch no-improvement). Blocker: add `n_iter_no_change` + best-loss tracking. #1710. |
| REQ-9 (missing ctor params) | GAP | NOT-STARTED | vs `MLPClassifier.__init__` (`_multilayer_perceptron.py:1041-1067`), ferrolearn lacks: `learning_rate` (schedule), `learning_rate_init` (folded into `Solver`), `power_t`, `shuffle` (always shuffles), `warm_start`, `n_iter_no_change`, `momentum`/`nesterovs_momentum` (SGD-only; no nesterov), `beta_1`/`beta_2`/`epsilon` (only inside `Solver::Adam`, not top-level), `max_fun`, `verbose`. Also `batch_size`/`tol`/`alpha`/`max_iter`/`hidden_layer_sizes`/`activation`/`solver`/`early_stopping`/`validation_fraction` ARE present with matching defaults (verified: solver=adam, lr_init=0.001, max_iter=200, tol=1e-4, alpha=1e-4, batch=auto). Blocker: expose the missing R-DEV-2 params. #1710. |
| REQ-10 (fitted attributes) | GAP | NOT-STARTED | ferrolearn exposes `classes()`/`n_classes()` (`HasClasses`) and `n_layers()`; the layer weights/biases are PRIVATE (`layers: Vec<LayerParams<F>>`). sklearn exposes `coefs_`, `intercepts_`, `loss_`, `loss_curve_`, `best_loss_`, `n_iter_`, `t_`, `n_outputs_`, `out_activation_`, `validation_scores_`/`best_validation_score_` (`_initialize`, `_multilayer_perceptron.py:366-407`). Absent: all of the above except `classes_`/`n_layers_`. Blocker: expose `coefs_`/`intercepts_`/`loss_`/`loss_curve_`/`n_iter_`/`out_activation_` etc. (R-DEV-1). #1710. |
| REQ-11 (predict_proba) | DET | NOT-STARTED | `fn predict_proba in mlp.rs` returns softmax rows (multiclass) and binary `[1-p, p]` (= `_multilayer_perceptron.py:1252-1253`); rows sum to 1 (verified). Same status as REQ-3: math matches but no live-oracle value test and no non-test consumer (REQ-14). #1710. |
| REQ-12 (collapsible_if lint) | CQ | NOT-STARTED | `train_network in mlp.rs` has `if early_stopping { if let (Some(xv), Some(yv)) = (&x_val, &y_val) { ... } }` which `clippy::collapsible_if` flags (let-chain collapse, suggested `if early_stopping && let (Some(xv), Some(yv)) = ...`). Confirmed: `cargo clippy -p ferrolearn-neural --all-targets` emits the warning (toolchain 1.95.0), so the `-D warnings` gauntlet fails. Blocker: collapse the nested `if`/`if let`. #1710. |
| REQ-13 (ferray substrate) | GAP | NOT-STARTED | `ferrolearn-neural/Cargo.toml` depends on `ndarray`, `rand`, `rand_distr`, `rand_xoshiro`; `mlp.rs` imports `use ndarray::{Array1, Array2, ...}`, `use rand::SeedableRng`, `use rand::seq::SliceRandom` and computes entirely on `ndarray` arrays with `StdRng`. Destination substrate is `ferray-core` (array) + `ferray::random` (sampling) per R-SUBSTRATE-1. No ferray usage. Blocker: migrate array type to `ferray-core` and RNG to `ferray::random`; this is where the RNG carve-out (REQ-5) is resolved or re-pinned. #1710. |
| REQ-14 (non-test consumer) | GAP | NOT-STARTED | The only references to `MLPClassifier`/`MLPRegressor`/`FittedMLP*` outside `src/mlp.rs` are `ferrolearn-neural/src/lib.rs` (re-export, line ~19) and `#[cfg(test)]` callers — verified via `grep -rn ... | grep -v src/mlp.rs | grep -v /tests/ | grep -v src/lib.rs` → empty. There is **no `ferrolearn-python` registration** of the neural crate (`grep -rn neural ferrolearn-python/src/` is empty). Per R-HONEST-2 / R-DEFER-1 a SHIPPED estimator needs a real non-test consumer; the canonical one is the PyO3 `import ferrolearn` registration (a Layer-6 task per R-DEFER-7, gated on the binding crate). #1710. |

## RNG carve-out vs deterministic classification

Per R-DEFER-3, RNG-stream divergences get a blocker but **NO failing characterization test**
(numpy `RandomState`/Mersenne-Twister vs Rust `StdRng` cannot bit-match). What MUST still
match — and gets a **failing characterization test from the critic** (R-CHAR-1, expected
values from the LIVE sklearn 1.5.2 oracle per R-CHAR-3) — is the deterministic math.

**RNG carve-outs (blocker only, NO failing test) — REQ-5:**
- Post-fit `coefs_`/`intercepts_` and predictions after `fit` — depend on the random init
  (`_init_coef` uniform, `_multilayer_perceptron.py:418-421`), the per-epoch minibatch
  shuffle (`_fit_stochastic:617-621`), and the early-stopping `train_test_split`
  (`:588-594`). Cannot bit-match.
- The structural init divergences (ferrolearn zeros the biases vs sklearn's uniform draw;
  ferrolearn uses Glorot factor 6 even for `logistic` vs sklearn's factor 2) are *documentable*
  and feed REQ-5; they shift the init distribution but the post-fit values remain RNG-coupled,
  so they are carried with REQ-5 (no failing value test on fitted weights).
- Determinism-default to DOCUMENT (not bit-match): ferrolearn `random_state=None` →
  `StdRng::from_os_rng` (non-deterministic), matching sklearn `None`
  (`check_random_state`, `_multilayer_perceptron.py:454`). A `random_state=Some(seed)` run is
  reproducible within ferrolearn but NOT equal to sklearn's `random_state=seed` stream.
- The meaningful signal for REQ-5 is a LEARNING test: on a linearly separable 2-class problem
  with a fixed seed, train accuracy beats chance and the loss decreases (the existing
  `test_classifier_binary_basic` / `test_regressor_linear_fit` exercise this, but with
  Rust-side thresholds, not an oracle).

**Deterministic — REQs that MUST get a failing characterization test from the critic
(expected values from LIVE sklearn 1.5.2):**
- REQ-1 activations: value parity vs `ACTIVATIONS[name]` on a fixed vector.
- REQ-2 loss formulas + the missing L2 term in the reported loss.
- REQ-3 / REQ-11 forward pass and `predict_proba` given `coefs_`/`intercepts_` copied from a
  live sklearn-fitted model (the canonical way to make an RNG-coupled estimator's forward
  pass deterministic-testable).
- REQ-4 Adam update (eps/bias-correction divergence) and SGD nesterov update, both from a
  FIXED gradient + optimizer state — fully deterministic, must fail until fixed.
- REQ-12 the clippy lint (mechanical, fails the `-D warnings` gauntlet today).

## Architecture

`mlp.rs` defines unfitted builders `MLPClassifier<F>` / `MLPRegressor<F>` (public fields
`hidden_layer_sizes`, `activation`, `solver`, `max_iter`, `tol`, `batch_size`, `random_state`,
`early_stopping`, `validation_fraction`, `alpha`) with `#[must_use]` chained `with_*` setters
and `new()` defaults matching sklearn where present (solver Adam lr=0.001, max_iter=200,
tol=1e-4, alpha=1e-4, hidden `[100]`, relu). The fitted `FittedMLPClassifier<F>` holds
`layers: Vec<LayerParams<F>>` (private weights/biases), `hidden_activation`, `classes`,
`is_binary`; `FittedMLPRegressor<F>` holds `layers` + `hidden_activation`. `Fit` produces the
fitted struct; `Predict` and the inherent `predict_proba`/`predict_log_proba` (classifier)
give the post-fit surface. Both estimators also implement `PipelineEstimator`/
`FittedPipelineEstimator` for the core pipeline (still a test-only consumer).

The free functions partition the algorithm: `activate_inplace` / `activation_derivative` /
`stable_sigmoid` / `softmax_rows` (= `_base.py` ACTIVATIONS/DERIVATIVES); `forward` (cached,
training) / `forward_output` (cacheless, predict) (= `_forward_pass` / `_forward_pass_fast`);
`backward` (= `_backprop` + `_compute_loss_grad`, with `grad_w = inᵀ@δ/n + alpha·W`);
`apply_sgd_update` / `apply_adam_update` (= the two optimizers); `train_network` (= `_fit` +
`_fit_stochastic`); `init_layers` / `xavier_init` (= `_init_coef`, but zero-bias, factor-6);
`build_classification_target` (= `LabelBinarizer.transform`).

Structural divergences from the sklearn class: the reported loss omits the L2 term (REQ-2);
Adam's eps/bias-correction placement differs (REQ-4); SGD has no nesterov and no lr schedule
(REQ-4/REQ-7); the convergence criterion is single-epoch Δloss not `n_iter_no_change`
consecutive (REQ-8); `lbfgs` is absent (REQ-6); biases init to zero and the Glorot factor is
always 6 (REQ-5); many constructor params and fitted attributes are missing (REQ-9/REQ-10).
The substrate is `ndarray` + `rand`/`rand_distr` + `StdRng`, not ferray (REQ-13). Generic over
`F: Float + Send + Sync + ScalarOperand + 'static`, supporting f32 and f64 per CLAUDE.md.

## Verification

Commands that would establish SHIPPED claims (NONE is green for a SHIPPED classification
today — every REQ is NOT-STARTED, lacking a live-oracle value test and/or a non-test consumer):

- `cargo test -p ferrolearn-neural --lib` — 49 lib tests pass, but they assert shape/range/
  learning only (e.g. `test_classifier_binary_basic` checks `correct >= 6`,
  `test_softmax_rows` checks rows sum to 1). None pins a VALUE against sklearn, so they do not
  establish REQ-1..REQ-4/REQ-11 (R-CHAR-1/R-CHAR-3 require live-sklearn expected values).
- Live oracle, REQ-1 (activations):
  `python3 -c "from sklearn.neural_network._base import ACTIVATIONS; import numpy as np; x=np.array([[-2.,-.5,0.,1.5]]); [print(k, (lambda a: (ACTIVATIONS[k](a), a)[1].tolist())(x.copy())) for k in ['relu','tanh','logistic','identity']]"`.
- Live oracle, REQ-3/REQ-11 (forward / predict_proba): fit a sklearn MLP, copy `m.coefs_`,
  `m.intercepts_`, `m.out_activation_` into a `FittedMLPClassifier`, and compare
  `predict_proba` (verified the relu→logistic forward on fixed weights →
  `[0.6661893, 0.4056446]` matches scipy `expit`).
- Live oracle, REQ-4 (Adam): `AdamOptimizer._get_updates([g])` on a fixed `g` →
  `upd = -lr*sqrt(1-b2^t)/(1-b1^t)*m/(sqrt(v)+eps)`; verified t=1 g=[0.5,-0.2] →
  sklearn `[-9.99999e-4, 9.99998e-4]` vs ferrolearn `[-9.99999e-4, 9.99999e-4]` (diverges).
- Defaults (REQ-9):
  `python3 -c "from sklearn.neural_network import MLPClassifier as M; m=M(); print(m.solver, m.learning_rate_init, m.learning_rate, m.max_iter, m.tol, m.alpha, m.batch_size, m.n_iter_no_change, m.momentum, m.nesterovs_momentum)"`
  → `adam 0.001 constant 200 0.0001 0.0001 auto 10 0.9 True`.
- Lint (REQ-12): `cargo clippy -p ferrolearn-neural --all-targets` emits
  `clippy::collapsible_if` at `mlp.rs` `train_network` (early-stopping block); fails under
  `-D warnings`.
- Substrate (REQ-13): `cargo tree -p ferrolearn-neural | grep -E 'ndarray|rand|ferray'`.
- Consumer (REQ-14):
  `grep -rn "MLPClassifier\|MLPRegressor" --include=*.rs | grep -v src/mlp.rs | grep -v src/lib.rs | grep -v /tests/`
  is empty; `grep -rn neural ferrolearn-python/src/` is empty.

Because no command above is green in a way that satisfies impl + non-test consumer +
live-sklearn-pinned value test simultaneously, every REQ is NOT-STARTED.
