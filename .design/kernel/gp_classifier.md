# Gaussian Process Classification

<!--
tier: 3-component
status: draft
baseline-commit: 0d808ab95
upstream-paths:
  - sklearn/gaussian_process/_gpc.py        # _BinaryGaussianProcessClassifierLaplace (:37), GaussianProcessClassifier (:487); LAMBDAS/COEFS (:31-37), binary fit (:172), predict (:270), predict_proba (:293), log_marginal_likelihood (:335), _posterior_mode (:414); GPC __init__ (:659), fit (:683), predict (:757), predict_proba (:779)
-->

## Summary

`ferrolearn-kernel/src/gp_classifier.rs` mirrors scikit-learn's
`sklearn.gaussian_process.GaussianProcessClassifier` (GPC) and its internal
`_BinaryGaussianProcessClassifierLaplace` — a probabilistic classifier that
models the latent decision function as a Gaussian Process and approximates the
non-Gaussian (Bernoulli/logistic) posterior by a Gaussian via the **Laplace
approximation** (Rasmussen & Williams Alg. 3.1/3.2/5.1). ferrolearn exposes the
unfitted `GaussianProcessClassifier<F>` (constructor `new`/`default_rbf`,
builders `max_iter`/`tol`) and `FittedGaussianProcessClassifier<F>`
(`predict` via the `Predict` trait, `predict_proba`, `predict_log_proba`,
`log_marginal_likelihood`, `classes`, `score`), driving the `GPKernel` family
documented in `gp_kernels.md`. Multi-class is one-vs-rest (one binary GP per
class), as in sklearn's default.

The **fixed-kernel, binary correctness slice** — the Laplace posterior mode
`f̂`/`π̂`, the latent `predict` sign decision, the `log_marginal_likelihood`
VALUE, `score`, and `classes` ordering — is value-exact against the live sklearn
1.5.2 oracle run with `optimizer=None`, and has a real production consumer (the
boundary estimator type itself, re-exported from `lib.rs`; there is NO Python
binding for GPC — the consumer is the Rust estimator API, which IS the public
surface per R-DEFER-1/S5). Those REQs are SHIPPED.

Everything that depends on the probability-squashing approximation, kernel
tuning, the convergence criterion's downstream API, multi-class probability
aggregation, the constructor surface, or the ferray substrate diverges and is
NOT-STARTED. The two headlines:

1. **`predict_proba` squashing is the WRONG approximation (DETERMINISTIC,
   FIXABLE — the headline divergence).** ferrolearn's `predict_binary_proba`
   computes the predictive latent mean `f̄* = K*·(y − π̂)` and variance
   `var* = k(x*,x*) − vᵀv` (`v = L⁻¹·sqrt(W)·k*`) — both MATCH sklearn — then
   squashes with the **MacKay probit** `sigmoid(f̄* / sqrt(1 + π·var*/8))`
   (`fn predict_binary_proba`), and a code comment FALSELY claims this "is the
   formulation used in scikit-learn's GaussianProcessClassifier" (R-HONEST-4
   flag: it is NOT). sklearn (`_gpc.py:324-331`) instead uses the **5-term
   LAMBDAS/COEFS error-function** approximation of Williams & Barber:
   `LAMBDAS=[0.41,0.4,0.37,0.44,0.39]` (`:31`),
   `COEFS=[-1854.82,3516.90,221.29,128.12,-2010.49]` (`:32-34`),
   `alpha = 1/(2·var*)`, `gamma = LAMBDAS·f̄*`,
   `integrals = sqrt(π/alpha)·erf(gamma·sqrt(alpha/(alpha+LAMBDAS²)))/(2·sqrt(var*·2π))`,
   `π* = (COEFS·integrals).sum() + 0.5·COEFS.sum()`. These give DIFFERENT
   probabilities (oracle below: MacKay `[0.319, 0.681]` vs sklearn
   `[0.322, 0.678]` at `Xs=[[0.5],[3.5]]`). The fix needs `erf` (available via
   the existing `statrs` dependency — `statrs::function::erf::erf` — or
   `ferray::stats`).
2. **Hyperparameter optimization ABSENT (the headline blocker).** Like GPR,
   ferrolearn's `fit_binary_gpc` uses the kernel's initial hyperparameters and
   NEVER optimizes; sklearn's default `optimizer="fmin_l_bfgs_b"` tunes `theta`
   on the LML (`_gpc.py:215-254`). So `GaussianProcessClassifier::new(kernel)` ≠
   sklearn's DEFAULT GPC (which tunes); it equals sklearn only with
   `optimizer=None`. Depends on gp_kernels `eval_gradient` #1912 + `bounds`
   #1913 + an L-BFGS-B optimizer.

The remaining gaps: the `_posterior_mode` convergence criterion + W-clamp differ
from sklearn (benign at the values checked, but the API and the W-clamp are
documented), the multi-class LML aggregation (sklearn MEANS the per-binary LMLs,
ferrolearn SUMS them), the binary `predict_proba` normalization (sklearn returns
the LAMBDAS/COEFS `π*` un-renormalized; ferrolearn's binary uses
`[1−π, π]` which DOES sum to 1, but the per-class OvR normalization differs from
sklearn's `OneVsRestClassifier` path), `multi_class="one_vs_one"`, the full
constructor surface, the `log_marginal_likelihood(theta, eval_gradient)` API,
and the ferray array/linalg/erf substrate (`ndarray` + hand-rolled cholesky +
`statrs`).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/gaussian_process/_gpc.py`
  - module constants `LAMBDAS` `:31`, `COEFS` `:32-34` — the 5-term erf
    approximation coefficients for the logistic sigmoid.
  - `class _BinaryGaussianProcessClassifierLaplace` `:37` — `BaseEstimator`.
    - `__init__` `:153-170` — `kernel=None` (→ `1.0*RBF(1.0)`),
      `optimizer="fmin_l_bfgs_b"`, `n_restarts_optimizer=0`,
      `max_iter_predict=100`, `warm_start=False`, `copy_X_train=True`,
      `random_state=None`.
    - `fit` `:172-268` — `LabelEncoder` (`:200-202`); errors if `classes_.size>2`
      (`:203-207`) or `==1` (`:208-213`); optimizer loop `:215-254`
      (`obj_func = -log_marginal_likelihood(theta, eval_gradient=True)`,
      `_constrained_optimization` over `kernel_.theta`/`kernel_.bounds`,
      `n_restarts_optimizer` log-uniform restarts); else stores the LML at the
      fixed theta (`:255-258`); precomputes the posterior mode (`:262-266`).
    - `predict` `:270-291` — `K_star = kernel_(X_train_, X)`;
      `f_star = K_star.T @ (y_train_ − pi_)` (Alg. 3.2 line 4, `:289`);
      `return np.where(f_star > 0, classes_[1], classes_[0])` (`:291`) — a sign
      decision on the LATENT mean, NOT an argmax of `predict_proba`.
    - `predict_proba` `:293-333` — `f_star = K_star.T @ (y_train_ − pi_)`;
      `v = solve(L_, W_sr_[:,None]*K_star)` (`:313`);
      `var_f_star = kernel_.diag(X) − einsum("ij,ij->j", v, v)` (`:315`); then the
      LAMBDAS/COEFS erf integral (`:324-331`) → `π*`;
      `return vstack((1−π*, π*)).T` (`:333`).
    - `log_marginal_likelihood` `:335-412` — `theta=None` → precomputed value
      (`:367-370`); else rebuild `K` at theta, `_posterior_mode`,
      `Z` (`:385`); `eval_gradient` → Alg. 5.1 gradient (`:390-412`, needs
      `kernel(X, eval_gradient=True)` `:379`).
    - `_posterior_mode` `:414-470` — Newton loop, max `max_iter_predict`;
      `pi = expit(f)`, `W = pi*(1−pi)`, `W_sr = sqrt(W)` (`:438-441`),
      `B = I + W_sr·K·W_sr`, `L = cholesky(B, lower=True)` (`:443-444`),
      `b = W*f + (y − pi)` (`:446`),
      `a = b − W_sr·cho_solve((L,True), W_sr_K·b)` (`:448`), `f = K·a` (`:450`);
      convergence on the approximate LML
      `lml = -0.5·aᵀf − Σlog1p(exp(−(2y−1)·f)) − Σlog diag(L)` (`:454-458`),
      `if lml − lml_old < 1e-10: break` (`:462`). **NO W-clamp.**
    - `_constrained_optimization` `:472-484` — `scipy.optimize.minimize(method=
      "L-BFGS-B", jac=True)` on `obj_func` over `bounds`.
  - `class GaussianProcessClassifier` `:487` — `ClassifierMixin`, `BaseEstimator`.
    - `_parameter_constraints` `:647-657` / `__init__` `:659-680` —
      `kernel=None`, `optimizer="fmin_l_bfgs_b"`, `n_restarts_optimizer=0`,
      `max_iter_predict=100`, `warm_start=False`, `copy_X_train=True`,
      `random_state=None`, `multi_class="one_vs_rest"`, `n_jobs=None`.
    - `fit` `:683-755` — rejects `CompoundKernel` (`:699-700`); `classes_ =
      np.unique(y)` (`:721`); errors if `n_classes_==1` (`:723-728`); for
      `n_classes_>2` wraps in `OneVsRestClassifier` (`:730-733`) or
      `OneVsOneClassifier` (`:734-737`); the multi-class LML is the **MEAN** of
      the per-binary LMLs (`:743-749`), the binary LML is the single value
      (`:750-753`).
    - `predict` `:757-777` — delegates to `base_estimator_.predict` (the latent
      sign for binary; the OvR/OvO argmax for multi-class via `multiclass.py`).
    - `predict_proba` `:779-807` — rejects `one_vs_one` (`:795-800`); delegates to
      `base_estimator_.predict_proba`; the OvR aggregation
      (`multiclass.py:550-551`) normalizes `Y /= Y.sum(axis=1)`.
  - `OneVsRestClassifier.predict_proba` `multiclass.py:515-551` — stacks the
    per-class positive-class probabilities and normalizes rows to sum 1
    (`:550-551`).

## Requirements

- REQ-1: Binary Laplace posterior mode `f̂`/`π̂`. `fit_binary_gpc` runs the
  Newton iteration `π = sigmoid(f)`, `W = diag(π(1−π))`,
  `B = I + sqrt(W)·K·sqrt(W)`, `L = cholesky(B)`, `b = Wf + (y−π)`,
  `a = b − sqrt(W)·L⁻ᵀL⁻¹(sqrt(W)·K·b)`, `f = K·a` — mirroring
  `_gpc.py:438-450` (Alg. 3.1). The converged `f̂`/`π̂` match sklearn's `pi_`
  under `optimizer=None`. (Deterministic / oracle-pinnable.)
- REQ-2: Binary `predict` (latent-sign decision). For the binary case `predict`
  selects `classes_[1]` when the latent predictive mean is positive, else
  `classes_[0]` — sklearn does `np.where(f_star > 0, classes_[1], classes_[0])`
  (`_gpc.py:291`). ferrolearn argmaxes `predict_proba`'s two columns, which
  agrees with the sign of `f̄*` because the squash is a monotone increasing
  function of `f̄*` and `[1−π, π]` crosses at `π=0.5 ⟺ f̄*=0`. (Deterministic /
  oracle-pinnable.)
- REQ-3: Binary `log_marginal_likelihood` VALUE at the fitted theta.
  `binary_log_marginal_likelihood` computes
  `-0.5·f̂ᵀ(y−π̂) + Σ[yᵢlog πᵢ+(1−yᵢ)log(1−πᵢ)] − Σlog Lᵢᵢ`, which is
  ALGEBRAICALLY EQUAL to sklearn's
  `Z = -0.5·aᵀf − Σlog1p(exp(−(2y−1)f)) − Σlog diag(L)` (`_gpc.py:454-458`)
  at convergence (the Bernoulli term `-log(1+exp(-(2y-1)f))` is the log-sigmoid
  of the signed margin = `y log π + (1-y) log(1-π)` for `π=sigmoid(f)`; and
  `aᵀf = f̂ᵀ(y−π̂)` since `a = K⁻¹f̂ = y−π̂` at convergence). Oracle-verified
  equal to 0.0 difference. (Deterministic / oracle-pinnable; the theta-arg +
  gradient API is REQ-9.)
- REQ-4: `score` = mean accuracy (`ClassifierMixin.score`).
  `FittedGaussianProcessClassifier::score` returns
  `mean_accuracy(predict(X), y)` — sklearn's `ClassifierMixin.score` default on
  `predict`. (Deterministic / oracle-pinnable.)
- REQ-5: `classes` ordering. `classes` returns the sorted unique class labels
  (`sort_unstable` + `dedup` in `Fit::fit`), matching sklearn's
  `classes_ = np.unique(y)` (`_gpc.py:721`) / the `LabelEncoder` sorted order
  (`:200-202`). (Deterministic / oracle-pinnable.)
- REQ-6: `max_iter_predict` default = 100. `new` sets `max_iter = 100`, matching
  `__init__(max_iter_predict=100)` (`_gpc.py:159`, `:665`). (Deterministic /
  oracle-pinnable.)
- REQ-7: `n_classes==1` / single-class error. `Fit::fit` returns
  `InvalidParameter` when `classes.len() < 2`, mirroring sklearn's `ValueError`
  "requires 2 or more distinct classes" (`_gpc.py:723-728`; binary-class
  `:208-213`). (Deterministic / oracle-pinnable — R-DEV-2 ABI.)
- REQ-8: Production consumer. `GaussianProcessClassifier` /
  `FittedGaussianProcessClassifier` are re-exported from `lib.rs` as the public
  boundary estimator type — the GPC API itself (no Python binding for GP exists).
  (Non-test consumer; grandfathered boundary type per S5.)
- REQ-9: `predict_proba` LAMBDAS/COEFS squashing. `predict_proba` must squash the
  predictive latent `(f̄*, var*)` with sklearn's 5-term error-function
  approximation `π* = (COEFS·integrals).sum() + 0.5·COEFS.sum()` using
  `LAMBDAS`/`COEFS` and `erf` (`_gpc.py:324-331`). ferrolearn's
  `predict_binary_proba` instead applies the **MacKay probit**
  `sigmoid(f̄*/sqrt(1+π·var*/8))` and its comment FALSELY claims that is sklearn's
  formulation (R-HONEST-4). The predictive mean/variance MATCH; only the squash
  diverges. Fix: replace the MacKay block with the LAMBDAS/COEFS erf integral
  (`statrs::function::erf::erf` / `ferray::stats`). (Deterministic / oracle-
  pinnable — the headline FIXABLE divergence.)
- REQ-10: Hyperparameter optimization (default `optimizer="fmin_l_bfgs_b"`).
  `fit` with the default optimizer must run L-BFGS-B on `-log_marginal_likelihood`
  (with `n_restarts_optimizer` restarts) to tune `kernel_.theta`
  (`_gpc.py:215-254`), using the kernel's `eval_gradient` (`:379`) and `bounds`.
  ferrolearn's `fit_binary_gpc` never optimizes — the kernel keeps its initial
  hyperparameters; there is no `optimizer`/`n_restarts_optimizer` surface.
  (Depends on gp_kernels `eval_gradient` #1912 + `bounds` #1913 + an L-BFGS-B
  optimizer.)
- REQ-11: `_posterior_mode` convergence criterion + W-clamp. sklearn iterates
  until the approximate-LML change `lml − lml_old < 1e-10` (`_gpc.py:454-462`),
  max `max_iter_predict=100`, with NO clamp on `W`. ferrolearn iterates until the
  max latent-f change `< tol` (default `1e-6`, no sklearn analog), max `max_iter`,
  and CLAMPS `W = π(1−π)` to `≥ 1e-12` (`fn fit_binary_gpc`, `if w_val < 1e-12`)
  — sklearn does not clamp. The converged `f̂`/`π̂` still match the oracle at the
  fixtures tested (REQ-1), so this is a benign implementation difference at those
  values; documented per R-DEV-1. The `tol` parameter and the W-clamp are
  surface/behavioral divergences that must be reconciled (sklearn's hard-coded
  `1e-10` LML criterion, no `tol`). (Deterministic / oracle-pinnable — the
  critic verifies `f̂`/`π̂` parity across a wider fixture set; any mismatch beyond
  tolerance reclassifies the W-clamp/criterion as a value divergence.)
- REQ-12: Multi-class LML aggregation. For `n_classes_>2`, sklearn returns the
  **MEAN** of the per-binary LMLs
  (`np.mean([est.log_marginal_likelihood() ...])`, `_gpc.py:743-749`).
  ferrolearn's `log_marginal_likelihood` **SUMS** the per-binary LMLs
  (`fold(F::zero(), |a,b| a+b)`). Oracle (3-class fixture): sklearn mean
  `-5.2469`, ferrolearn sum `-15.7407` — a value divergence. (Deterministic /
  oracle-pinnable.)
- REQ-13: OvR `predict_proba` normalization. For multi-class, sklearn's
  `OneVsRestClassifier.predict_proba` stacks the per-class positive
  probabilities and normalizes rows `Y /= Y.sum(axis=1)`
  (`multiclass.py:550-551`). ferrolearn does the same row-normalization in
  `predict_proba` (multi-class branch), but it builds each column from the
  DIVERGENT MacKay squash (REQ-9), so the multi-class probabilities differ from
  sklearn even after normalization. The normalization STEP matches; its inputs do
  not. (Deterministic / oracle-pinnable once REQ-9 lands.)
- REQ-14: `multi_class="one_vs_one"`. sklearn supports `multi_class="one_vs_one"`
  (`OneVsOneClassifier`, `_gpc.py:734-737`) and rejects `predict_proba` in that
  mode (`:795-800`). ferrolearn only does one-vs-rest; there is no `multi_class`
  selector. (Deterministic / oracle-pinnable once supported.)
- REQ-15: `log_marginal_likelihood(theta, eval_gradient)` API. Evaluate the LML
  at an ARBITRARY `theta` (rebuilding `K`) and optionally return the Alg. 5.1
  gradient (`_gpc.py:335-412`, needs `kernel(X, eval_gradient=True)` `:379`).
  ferrolearn's `log_marginal_likelihood(&self)` takes no arguments and evaluates
  only at the fitted theta — no theta argument, no gradient. (Depends on #1912.)
- REQ-16: Constructor surface (`optimizer`, `n_restarts_optimizer`,
  `warm_start`, `copy_X_train`, `random_state`, `multi_class`, `n_jobs`,
  `kernel=None` default). sklearn's `__init__` (`_gpc.py:659-680`) carries all of
  these; `kernel=None → 1.0*RBF(1.0)` (`:187-190`). ferrolearn's `new` requires
  an explicit kernel and exposes only `max_iter`/`tol` (`tol` has no sklearn
  analog). (R-DEV-2 ABI.)
- REQ-17: ferray substrate (R-SUBSTRATE-1). Array type → `ferray-core`; Cholesky
  / triangular solves → `ferray::linalg` (the `scipy.linalg.cholesky`/`cho_solve`/
  `solve` analog sklearn uses at `_gpc.py:444,448,313`); `erf` → `ferray::stats`
  (for the REQ-9 fix) — instead of `ndarray` + the hand-rolled `cholesky_gpc`/
  `forward_solve_gpc`/`backward_solve_gpc` + `statrs`.

## Acceptance criteria

All live-oracle commands use sklearn `optimizer=None` (R-CHAR-3) so both sides
use the SAME un-tuned kernel; otherwise sklearn's default tuner makes the
comparison invalid (that gap is REQ-10). Binary fixture: `X=[[0],[0.5],[1],[3],
[3.5],[4]]`, `y=[0,0,0,1,1,1]`, `Xs=[[0.5],[2.0],[3.5]]`, `RBF(1.0)`. Multi-class
fixture: `X=[[0],[0.5],[1],[4],[4.5],[5],[8],[8.5],[9]]`,
`y=[0,0,0,1,1,1,2,2,2]`, `RBF(1.0)`.

- AC-1 (REQ-1): ferrolearn's converged `π̂` equals the live oracle's
  `base_estimator_.pi_` = `[0.3156024029, 0.2997621127, 0.326954663,
  0.673045337, 0.7002378873, 0.6843975971]` to ~1e-8 (binary fixture). In-crate
  `predict_proba_class_0_near_0` pins the qualitative shape.
- AC-2 (REQ-2): ferrolearn `fit(X,y).predict(Xs)` equals the live oracle's
  `predict(Xs)` = `[0, 1, 1]` (binary fixture). In-crate `fit_predict_binary`/
  `fit_predict_binary_2d`.
- AC-3 (REQ-3): ferrolearn `log_marginal_likelihood()` (binary) equals the live
  oracle's `m.log_marginal_likelihood_value_` = `-3.5258847566` to ~1e-8 (binary
  fixture; algebraic-equality verified, diff 0.0). In-crate
  `log_marginal_likelihood_binary_finite_and_negative`/
  `log_marginal_likelihood_prefers_separable_data`.
- AC-4 (REQ-4): ferrolearn `score(X,y)` equals `m.score(X,y)` = `1.0` (binary
  fixture).
- AC-5 (REQ-5): ferrolearn `classes()` equals `m.classes_.tolist()` = `[0, 1]`
  (binary) / `[0, 1, 2]` (multi-class). In-crate
  `classes_accessor_returns_sorted_labels`/`non_contiguous_labels`.
- AC-6 (REQ-6): `GaussianProcessClassifier::new(kernel)` reports `max_iter ==
  100`, matching `GPC().max_iter_predict == 100` (`_gpc.py:665`). In-crate
  `builder_pattern`.
- AC-7 (REQ-7): live oracle `GPC(kernel=RBF(1.0),optimizer=None).fit(X,[0,0])`
  raises `ValueError` "requires 2 or more distinct classes"; ferrolearn's
  `fit(x, all-same-class)` returns `Err(InvalidParameter)`. In-crate
  `fit_rejects_single_class`.
- AC-8 (REQ-8): `lib.rs` re-exports `GaussianProcessClassifier`/
  `FittedGaussianProcessClassifier` (line 70) and documents it in the crate
  `//!` (line 21); the classifier fits and predicts through `kernel.compute` in
  `fn fit_binary_gpc`/`fn predict_binary_proba` (the boundary public API). No
  `ferrolearn-python` GP binding exists (`grep` confirms none).
- AC-9 (REQ-9): live oracle (binary fixture) `m.predict_proba(Xs)` =
  `[[0.6782527908, 0.3217472092], [0.500000005, 0.499999995],
  [0.3217472192, 0.6782527808]]`. ferrolearn's MacKay squash yields the positive-
  class column `[0.3191567403, 0.5, 0.6808432597]` (≠ sklearn's
  `[0.3217472092, 0.499999995, 0.6782527808]`) — the pinned divergence. The
  predictive `f̄*`/`var*` MATCH (they feed both squashes). Fix: in
  `predict_binary_proba` replace the `kappa`/`sigmoid` block with
  `alpha = 1/(2·var*)`, `gamma = LAMBDAS·f̄*`,
  `integrals = sqrt(π/alpha)·erf(gamma·sqrt(alpha/(alpha+LAMBDAS²)))/(2·sqrt(var*·2π))`,
  `π* = (COEFS·integrals).sum() + 0.5·COEFS.sum()`, with `LAMBDAS`/`COEFS` from
  `_gpc.py:31-34` and `erf` from `statrs`/`ferray::stats`.
- AC-10 (REQ-10): `GPC(kernel=RBF(1.0)).fit(X,y)` (DEFAULT optimizer) yields a
  TUNED `m.kernel_.theta` (`length_scale` ≠ 1.0) and a higher LML than the
  `optimizer=None` value; ferrolearn's `fit` leaves the kernel at
  `length_scale=1.0`. Oracle-pinnable once the optimizer + #1912/#1913 land.
- AC-11 (REQ-11): ferrolearn's `π̂` matches the oracle's `pi_` (AC-1) DESPITE the
  different convergence criterion (max-Δf `< 1e-6` vs LML-Δ `< 1e-10`) and the
  W-clamp (`≥ 1e-12`, absent in sklearn). The critic sweeps multiple fixtures
  (well-separated, near-overlapping, multi-d) and asserts `π̂` parity to ~1e-7;
  any fixture exceeding tolerance reclassifies the W-clamp/criterion as a value
  divergence requiring the criterion swap to `lml − lml_old < 1e-10` and removal
  of the clamp.
- AC-12 (REQ-12): live oracle (multi-class fixture)
  `m.log_marginal_likelihood_value_` = `-5.2469031950` (the MEAN of per-binary
  `[-5.2457, -5.2493, -5.2457]`); ferrolearn's multi-class
  `log_marginal_likelihood()` returns the SUM `-15.7407095850` — the pinned
  divergence. Fix: divide the fold by `binary_models.len()` in the multi-class
  branch (sklearn means only for `n_classes_>2`; binary stays the single value).
- AC-13 (REQ-13): live oracle (multi-class fixture) `m.predict_proba(X)[0]` =
  `[0.83.../0.49454.../...]`-style normalized OvR row that sums to 1; ferrolearn's
  multi-class `predict_proba` row also sums to 1 (in-crate
  `predict_proba_multiclass` checks `|row_sum − 1| < 1e-8`) but its column values
  derive from the MacKay squash and so diverge element-wise from sklearn until
  REQ-9 lands. The normalization step matches; the inputs do not.
- AC-14 (REQ-14): `GPC(multi_class="one_vs_one").fit(X3, y3)` (3-class) predicts
  via OvO and `predict_proba` raises `ValueError` "does not support predicting
  probability"; ferrolearn has no `multi_class` selector (always OvR).
  Oracle-pinnable once supported.
- AC-15 (REQ-15): `m.log_marginal_likelihood(np.log([2.0]), eval_gradient=True)`
  returns `(value, gradient)` at an arbitrary theta; ferrolearn's
  `log_marginal_likelihood()` takes no arguments and returns a scalar at the
  fitted theta only. Oracle-pinnable once #1912 lands.
- AC-16 (REQ-16): `GPC()` (no kernel) defaults to `1.0*RBF(1.0)` and exposes
  `optimizer`/`n_restarts_optimizer`/`warm_start`/`copy_X_train`/`random_state`/
  `multi_class`/`n_jobs`; ferrolearn `new` requires an explicit kernel and
  exposes only `max_iter`/`tol` (`tol` has no sklearn analog —
  `max_iter_predict`'s criterion is the hard-coded `1e-10`).
- AC-17 (REQ-17): no `ndarray`/hand-rolled `cholesky_gpc`/`statrs` in the owned
  computation; arrays/linalg/erf route through ferray.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (binary posterior mode f̂/π̂) | SHIPPED | `fn fit_binary_gpc` runs the Newton loop `let pi = f.mapv(sigmoid); let w = pi(1-pi) clamped; let b = I + sqrt(w)·K·sqrt(w); let l = cholesky_gpc(&b)?; b_vec = w*f+(y-pi); a = b_vec - sqrt(w)·backward_solve(forward_solve(sqrt(w)·K·b_vec)); f_new = K·a` — mirrors `_gpc.py:438-450` (Alg. 3.1). Live oracle (`optimizer=None`, `RBF(1.0)`, binary fixture): `base_estimator_.pi_ = [0.3156024029,0.2997621127,0.326954663,0.673045337,0.7002378873,0.6843975971]`. Non-test consumer: `lib.rs` re-export (REQ-8). In-crate `fit_predict_binary`/`predict_proba_class_0_near_0`. Verification: `cargo test -p ferrolearn-kernel --lib gp_classifier`. Deterministic / oracle-pinnable. |
| REQ-2 (binary latent-sign predict) | SHIPPED | FIXED #1932 (the doc-author's prior argmax-of-proba reasoning was WRONG, R-HONEST-4): binary `Predict::predict` now decides by the SIGN of the latent posterior mean `f̄* = K*·(y−π̂)` (via `predict_binary_latent_mean`), matching sklearn `np.where(f_star > 0, classes_[1], classes_[0])` (`_gpc.py:289-291`) — argmax-of-proba and latent-sign DISAGREE near the boundary (at `f̄*=+6.9e-17` proba[1]=0.4999999950<0.5 → argmax class 0, but sklearn class 1). Pinned by `divergence_binary_predict_latent_sign` (was `[0,0,1]` vs sklearn `[0,1,1]`). Deterministic / oracle-pinnable. |
| REQ-3 (binary LML value) | SHIPPED | `fn binary_log_marginal_likelihood` computes `-0.5·f̂ᵀ(y-π̂) + Σ[yᵢlnπᵢ+(1-yᵢ)ln(1-πᵢ)] - Σln Lᵢᵢ` — ALGEBRAICALLY EQUAL to sklearn's `Z = -0.5·aᵀf - Σlog1p(exp(-(2y-1)f)) - Σlog diag(L)` (`_gpc.py:454-458`) at convergence (Bernoulli log-lik identity + `aᵀf = f̂ᵀ(y-π̂)`). Oracle-verified diff = 0.0; `m.log_marginal_likelihood_value_ = -3.5258847566` (binary fixture). Non-test consumer: REQ-8 (`log_marginal_likelihood` is a public method). In-crate `log_marginal_likelihood_binary_finite_and_negative`/`log_marginal_likelihood_prefers_separable_data`. Deterministic / oracle-pinnable. (theta-arg + gradient = REQ-15, NOT-STARTED.) |
| REQ-4 (score / mean accuracy) | SHIPPED | `FittedGaussianProcessClassifier::score` does `let preds = self.predict(x)?; Ok(crate::mean_accuracy(&preds, y))` — sklearn's `ClassifierMixin.score` default (accuracy on `predict`). Live oracle (binary fixture): `m.score(X,y) = 1.0`. Non-test consumer: REQ-8 (`score` is a public method). Verification: covered by the fit/predict path tests. Deterministic / oracle-pinnable. |
| REQ-5 (classes ordering) | SHIPPED | `Fit::fit` builds `classes` by `y.iter().copied().collect()` then `sort_unstable()` + `dedup()`; `FittedGaussianProcessClassifier::classes` returns the sorted slice — matches `classes_ = np.unique(y)` (`_gpc.py:721`) / `LabelEncoder` sorted order (`:200-202`). Live oracle: `[0,1]` (binary), `[0,1,2]` (multi-class). Non-test consumer: REQ-8. In-crate `classes_accessor_returns_sorted_labels`/`non_contiguous_labels`. Deterministic / oracle-pinnable. |
| REQ-6 (max_iter_predict default 100) | SHIPPED | `GaussianProcessClassifier::new` sets `max_iter: 100` and the field doc records `Default: 100` — matches `__init__(max_iter_predict=100)` (`_gpc.py:159`, `:665`). Non-test consumer: REQ-8 (`new`/`default_rbf` are the constructors). In-crate `builder_pattern`/`converges_with_few_iterations`. Deterministic / oracle-pinnable. |
| REQ-7 (n_classes==1 error) | SHIPPED | `Fit::fit` returns `Err(FerroError::InvalidParameter { name: "y", reason: "need at least 2 classes ..." })` when `classes.len() < 2` — mirrors sklearn's `ValueError` "requires 2 or more distinct classes" (`_gpc.py:723-728`). Live oracle: `GPC(...).fit(X,[0,0])` raises that `ValueError`. Non-test consumer: REQ-8. In-crate `fit_rejects_single_class`. Deterministic / oracle-pinnable (R-DEV-2 ABI). |
| REQ-8 (production consumer) | SHIPPED | `lib.rs` re-exports `pub use gp_classifier::{FittedGaussianProcessClassifier, GaussianProcessClassifier}` (line 70) and documents `GaussianProcessClassifier` in the crate `//!` (line 21). The estimator IS the public boundary type: it holds `kernel: Box<dyn GPKernel<F>>` and drives `kernel.compute` in `fn fit_binary_gpc`/`fn predict_binary_proba`. There is NO `ferrolearn-python` GP binding (confirmed: `grep -rn GaussianProcess ferrolearn-python/` → no matches) — per R-DEFER-1/S5 the boundary estimator type is the public API; this REQ ships on the re-export, NOT a Python registration. Verification: `cargo test -p ferrolearn-kernel --lib gp_classifier`. |
| REQ-9 (predict_proba LAMBDAS/COEFS squash) | SHIPPED | FIXED #1931: `fn predict_binary_proba` now squashes the predictive `(f̄*, var*)` with sklearn's 5-term erf approximation `alpha=1/(2·var*)`, `gamma=LAMBDAS·f̄*`, `integrals=sqrt(π/alpha)·erf(gamma·sqrt(alpha/(alpha+LAMBDAS²)))/(2·sqrt(var*·2π))`, `π*=(COEFS·integrals).sum()+0.5·COEFS.sum()` with module consts `LAMBDAS=[0.41,0.4,0.37,0.44,0.39]`/`COEFS` verbatim from `_gpc.py:31-34`,`:324-331` (via `statrs` `erf`, f64, `var*<=0` floored to `f64::MIN_POSITIVE`), replacing the wrong MacKay probit (and its false comment, R-HONEST-4). Oracle-verified element-wise (~1e-13) across high/low/boundary/far(var→∞)/coincident(var≈0) points, no NaN/Inf. Pinned by `divergence_binary_predict_proba_squashing`. Deterministic / oracle-pinnable. |
| REQ-10 (hyperparameter optimization) | NOT-STARTED | open prereq blockers #1912 (`eval_gradient`/`dK/dθ`) + #1913 (`theta`/`bounds`); plus a missing L-BFGS-B optimizer. `fn fit_binary_gpc` does a single fixed-kernel Newton loop with NO optimizer — the kernel keeps its initial hyperparameters; there is no `optimizer`/`n_restarts_optimizer` field. sklearn's default `optimizer="fmin_l_bfgs_b"` (`_gpc.py:157`,`:663`) runs `_constrained_optimization` on `-log_marginal_likelihood(theta, eval_gradient=True)` over `kernel_.theta`/`kernel_.bounds` with `n_restarts_optimizer` restarts (`:215-254`). Requires the kernel's `eval_gradient` (gp_kernels.md / #1912) and `bounds` (gp_kernels.md / #1913), neither exposed by `GPKernel`. Consequence: `new(kernel)` ≠ sklearn's DEFAULT GPC — they agree only under `optimizer=None` (which every SHIPPED AC pins). Deterministic in result / oracle-pinnable once #1912/#1913 + an optimizer land. |
| REQ-11 (posterior-mode convergence + W-clamp) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-1). `fn fit_binary_gpc` converges on `max_i |f_new_i - f_i| < tol` (default `tol=1e-6`, no sklearn analog) and CLAMPS `W=π(1-π)` to `≥ 1e-12` (`if w_val < 1e-12 { 1e-12 }`). sklearn converges on the approximate-LML change `lml - lml_old < 1e-10` (`_gpc.py:454-462`) with NO W-clamp. The converged `π̂` matches the oracle at the binary fixture (REQ-1, AC-11), so the gap is benign THERE; but the criterion is not sklearn's, `tol` has no upstream counterpart, and the W-clamp is an un-sklearn behavior that perturbs near-degenerate `W`. NOT-STARTED until the criterion is `lml - lml_old < 1e-10` and the clamp removed (or the critic proves clamp/criterion never changes `π̂` beyond tolerance across the fixture sweep, in which case this becomes a documented benign deviation closed by an explanatory note + criterion alignment). Deterministic / oracle-pinnable. |
| REQ-12 (multi-class LML aggregation) | SHIPPED | FIXED #1933: `FittedGaussianProcessClassifier::log_marginal_likelihood` now returns the MEAN of the per-binary LMLs (`sum / binary_models.len()`), matching sklearn's `np.mean([...])` for `n_classes_>2` (`_gpc.py:743-749`); for binary (1 model) mean==value so REQ-3 stays green. Oracle (3-class fixture) `-5.2469031950` (was sum `-15.7407095850`). Pinned by `divergence_multiclass_lml_mean_vs_sum`. Deterministic / oracle-pinnable. |
| REQ-13 (OvR predict_proba normalization) | SHIPPED | CLOSED by the REQ-9 squash fix #1931: `predict_proba` (multi-class branch) row-normalizes `raw[[i,c]] /= row_sum` (matching `OneVsRestClassifier.predict_proba`'s `Y /= Y.sum(axis=1)`), and now that each column is the correct LAMBDAS/COEFS positive probability, the full `n×n_classes` matrix matches sklearn's OvR `predict_proba` (`_gpc.py:779-807`) element-wise (~1e-13, rows sum to 1). Guard `green_audit_multiclass_ovr_predict_proba`. Deterministic / oracle-pinnable. |
| REQ-14 (multi_class one_vs_one) | NOT-STARTED | blocker issue to be filed by critic. ferrolearn's `Fit::fit` always builds one-vs-rest (`binary_models`, one per class); there is no `multi_class` selector and no OvO path. sklearn supports `multi_class="one_vs_one"` (`OneVsOneClassifier`, `_gpc.py:734-737`) and rejects `predict_proba` in that mode (`:795-800`). Live oracle: `GPC(multi_class="one_vs_one").fit(X3,y3)` predicts via OvO; `predict_proba` raises `ValueError`. Needs an OvO decomposition + the `multi_class` param. Deterministic / oracle-pinnable once supported. |
| REQ-15 (LML theta-arg + gradient API) | NOT-STARTED | open prereq blocker #1912 (kernel `eval_gradient`). `FittedGaussianProcessClassifier::log_marginal_likelihood(&self)` takes no arguments and evaluates only at the fitted theta. sklearn's `log_marginal_likelihood(theta=None, eval_gradient=False, clone_kernel=True)` (`_gpc.py:335-412`) rebuilds `K` at `theta` and with `eval_gradient` returns the Alg. 5.1 gradient (`:390-412`), needing `kernel(X, eval_gradient=True)` (`:379`, #1912). The VALUE at the fitted theta IS shipped (REQ-3); the theta-evaluation + gradient is the gap REQ-10's optimizer consumes. Deterministic / oracle-pinnable once #1912 lands. |
| REQ-16 (constructor surface) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 ABI). `GaussianProcessClassifier::new(kernel)` REQUIRES an explicit kernel and exposes only `max_iter`/`tol` builders. sklearn's `__init__` (`_gpc.py:659-680`) defaults `kernel=None → 1.0*RBF(1.0)` (`:187-190`) and carries `optimizer`, `n_restarts_optimizer`, `warm_start`, `copy_X_train`, `random_state`, `multi_class`, `n_jobs`. ferrolearn has `default_rbf()` (a bare `RBF(1.0)`, NOT sklearn's `Constant*RBF` default) but no `None`-kernel path and none of those params; `tol` has NO sklearn analog (sklearn's posterior-mode criterion is the hard-coded `1e-10`). Live oracle: `GPC().kernel_` is `1**2 * RBF(length_scale=1)`. Deterministic / oracle-pinnable on the default kernel + param surface. |
| REQ-17 (ferray substrate) | NOT-STARTED | blocker issue to be filed by critic (R-SUBSTRATE-1). `gp_classifier.rs` imports `ndarray::{Array1, Array2}` and hand-rolls `cholesky_gpc`/`forward_solve_gpc`/`backward_solve_gpc`/`mat_vec_mul`; the REQ-9 fix additionally needs `erf` (currently the crate would reach for `statrs`). Destination: `ferray-core` (array type), `ferray::linalg` (the `scipy.linalg.cholesky`/`cho_solve`/`solve` analog sklearn uses at `_gpc.py:444,448,313`), `ferray::stats` (the `scipy.special.erf` analog, `_gpc.py:13`/`:328`). Not migrated. |

## Architecture

ferrolearn splits the estimator into the unfitted `GaussianProcessClassifier<F>`
(fields `kernel: Box<dyn GPKernel<F>>`, `max_iter: usize`, `tol: F`) and the
fitted `FittedGaussianProcessClassifier<F>` (`classes: Vec<usize>`,
`binary_models: Vec<FittedBinaryGPC<F>>`). Each `FittedBinaryGPC` stores
`x_train`, the Laplace mode `f_hat`, `pi_hat = sigmoid(f_hat)`, the binary labels
`y_binary`, the Cholesky factor `l_factor` of `B = I + sqrt(W)·K·sqrt(W)`, and
the `kernel`. This mirrors sklearn's two estimators — the public
`GaussianProcessClassifier` (`_gpc.py:487`, the `ClassifierMixin` wrapper that
does OvR/OvO) and the internal `_BinaryGaussianProcessClassifierLaplace`
(`:37`, which holds `pi_`/`W_sr_`/`L_`/`X_train_`/`y_train_`). ferrolearn's
`binary_models` is the OvR fan-out; sklearn delegates that to
`OneVsRestClassifier`/`OneVsOneClassifier` from `multiclass.py`. The builder API
is method-chained where sklearn uses keyword constructor args.

The numerical core is `fn fit_binary_gpc` (the Newton/Laplace loop, Alg. 3.1,
`_gpc.py:438-450`), `fn binary_log_marginal_likelihood` (Alg. 5.1 value), and
`fn predict_binary_proba` (Alg. 3.2 — predictive mean/variance + the squash).
`fit_binary_gpc` iterates `π = sigmoid(f)`, `W = π(1−π)` (clamped to `≥ 1e-12`,
the un-sklearn W-clamp of REQ-11), `B = I + sqrt(W)·K·sqrt(W)`,
`L = cholesky_gpc(B)`, `b = Wf + (y−π)`,
`a = b − sqrt(W)·backward_solve(forward_solve(sqrt(W)·K·b))`, `f = K·a`, breaking
on `max_i|Δf| < tol` (vs sklearn's LML-Δ `< 1e-10`, REQ-11). The hand-rolled
lower Cholesky + forward/backward substitution are the Rust analog of sklearn's
`scipy.linalg.cholesky`/`cho_solve`/`solve` (`_gpc.py:444,448,313`) — the ferray
substrate gap (REQ-17).

`predict_binary_proba` computes the predictive latent mean `f̄* = K*·(y−π̂)`
(`_gpc.py:312`) and, per test point, `v = forward_solve(L, sqrt(W)·k*)`,
`var* = k(x*,x*) − vᵀv` (`:313-315`) — both numerically matching sklearn — then
DIVERGES at the squash: it applies the MacKay probit instead of sklearn's
LAMBDAS/COEFS erf approximation (`:324-331`). This is the headline FIXABLE
divergence (REQ-9): the inputs to the squash are correct, only the squash
function is wrong, and the in-code comment misattributes the MacKay form to
sklearn (R-HONEST-4).

The structural gaps mirror `gaussian_process.md`/`gp_kernels.md`'s missing
surface: REQ-10 (tuning) needs the kernel's `eval_gradient` (#1912) and `bounds`
(#1913) plus an L-BFGS-B optimizer; REQ-15 (LML at arbitrary theta with gradient)
needs `eval_gradient` (#1912). So the un-tuned posterior is the slice that ships,
and every SHIPPED acceptance criterion pins sklearn `optimizer=None` to compare
like-for-like. The multi-class LML MEAN-vs-SUM (REQ-12) and the binary
probability squash (REQ-9) are entirely local divergences. The `one_vs_one`
mode (REQ-14) and the wider constructor surface (REQ-16) are absent abstractions.

Invariants: `B = I + sqrt(W)·K·sqrt(W)` is symmetric PD (Cholesky succeeds, else
`FerroError::NumericalInstability`); `predict_*` require
`X.ncols() == x_train.ncols()` (else `ShapeMismatch`); `fit` rejects
`n_samples < 2` (`InsufficientSamples`), `y.len() != n_samples` (`ShapeMismatch`),
and `< 2` distinct classes (`InvalidParameter`, REQ-7); `var*` is clipped to `≥ 0`
(`.max(F::zero())`, matching sklearn's predictive-variance floor); per-row
probabilities sum to 1 (binary `[1−π,π]`; multi-class explicit renormalization).

## Verification

Commands establishing the SHIPPED claims (run at baseline `0d808ab95`):

- `cargo test -p ferrolearn-kernel --lib gp_classifier` → all green
  (REQ-1/2: `fit_predict_binary`/`fit_predict_binary_2d`/
  `predict_proba_class_0_near_0`/`fit_predict_multiclass`/`f32_fit_predict`;
  REQ-3: `log_marginal_likelihood_binary_finite_and_negative`/
  `log_marginal_likelihood_prefers_separable_data`; REQ-5:
  `classes_accessor_returns_sorted_labels`/`non_contiguous_labels`; REQ-6:
  `builder_pattern`/`converges_with_few_iterations`; REQ-7:
  `fit_rejects_single_class`; error paths: `fit_rejects_insufficient_samples`/
  `fit_rejects_mismatched_y`/`predict_rejects_wrong_features`; kernels:
  `fit_with_matern`/`fit_with_sum_kernel`/`fit_with_product_kernel`).
- Fixed-kernel oracle (REQ-1..5, R-CHAR-3 — expected from sklearn `optimizer=None`,
  NEVER copied from ferrolearn), binary fixture `X=[[0],[0.5],[1],[3],[3.5],[4]]`,
  `y=[0,0,0,1,1,1]`, `Xs=[[0.5],[2.0],[3.5]]`, `RBF(1.0)`:
  `python3 -c "import numpy as np; from sklearn.gaussian_process import GaussianProcessClassifier as GPC; from sklearn.gaussian_process.kernels import RBF; X=np.array([[0.],[0.5],[1.],[3.],[3.5],[4.]]); y=np.array([0,0,0,1,1,1]); Xs=np.array([[0.5],[2.],[3.5]]); m=GPC(kernel=RBF(1.0),optimizer=None).fit(X,y); print(m.predict(Xs).tolist()); print(m.predict_proba(Xs).tolist()); print(m.log_marginal_likelihood_value_); print(m.score(X,y)); print(m.base_estimator_.pi_.tolist())"`
  → `predict=[0,1,1]`, `pi_=[0.3156,0.2998,0.3270,0.6730,0.7002,0.6844]`,
  `LML=-3.5258847566`, `score=1.0`. A critic pins `fit().predict`/`f̂`/
  `log_marginal_likelihood`/`score`/`classes` to the live oracle.

Open divergences pinned as FAILING tests (NOT-STARTED, oracle expected values):

- REQ-9 (predict_proba squash, FIXABLE — headline): live oracle (binary fixture)
  positive-class column `[0.3217472092, 0.499999995, 0.6782527808]`; ferrolearn's
  MacKay squash gives `[0.3191567403, 0.5, 0.6808432597]` — element-wise
  mismatch. The predictive `f̄*`/`var*` match (verified by replicating sklearn's
  `f_star`/`var_f_star` and feeding both squashes). Fixable entirely in this file
  (replace MacKay with LAMBDAS/COEFS erf).
- REQ-12 (multi-class LML): live oracle (3-class fixture)
  `log_marginal_likelihood_value_ = -5.2469031950` (MEAN); ferrolearn returns the
  SUM `-15.7407095850`. Fixable entirely in this file.
- REQ-10 (optimization): `GPC(kernel=RBF(1.0)).fit(X,y)` (DEFAULT optimizer) tunes
  `kernel_.theta` and raises the LML; ferrolearn is fixed at the initial kernel.
  Pinned once #1912/#1913 + an optimizer land.
- REQ-11 (convergence + W-clamp): the critic sweeps fixtures asserting `π̂` parity
  to the oracle's `pi_`; at the binary fixture they match, so the test pins the
  criterion/clamp surface (and reclassifies to a value divergence on any fixture
  where they don't).
- REQ-13 (OvR proba): gated on REQ-9 — multi-class `predict_proba` rows sum to 1
  but diverge element-wise until the squash is corrected.
- REQ-14 (one_vs_one): `GPC(multi_class="one_vs_one").fit(X3,y3)` predicts via OvO
  and `predict_proba` raises; ferrolearn has no `multi_class`.
- REQ-15 (LML theta/gradient): `m.log_marginal_likelihood(np.log([2.0]),
  eval_gradient=True)` returns `(value, gradient)`; ferrolearn's API takes no
  arguments. Pinned once #1912 lands.
- REQ-16 (constructor): `GPC().kernel_` is `1**2 * RBF(length_scale=1)` (the
  default kernel) and exposes the full param surface; ferrolearn requires an
  explicit kernel and exposes only `max_iter`/`tol`.

### Deterministic-vs-blocked split

- **Deterministic / oracle-pinnable NOW (SHIPPED):** REQ-1 (posterior mode),
  REQ-2 (binary latent-sign predict), REQ-3 (binary LML value), REQ-4 (score),
  REQ-5 (classes ordering), REQ-6 (max_iter default), REQ-7 (n_classes==1 error),
  REQ-8 (consumer) — all under sklearn `optimizer=None`, binary (or, for
  REQ-2/4/5, multi-class `predict`/`classes`).
- **Deterministic / oracle-pinnable but NOT-STARTED (local divergence —
  fixable in this file):** REQ-9 (predict_proba LAMBDAS/COEFS squash — the
  headline), REQ-11 (posterior-mode convergence + W-clamp), REQ-12 (multi-class
  LML mean-vs-sum), REQ-13 (OvR proba, gated on REQ-9), REQ-16 (constructor
  surface).
- **Dependency-blocked (NOT-STARTED):** REQ-10 (hyperparameter optimization —
  needs #1912/#1913 + an L-BFGS-B optimizer, the headline blocker), REQ-14
  (one_vs_one decomposition), REQ-15 (LML theta-arg + gradient — needs #1912).
- **Substrate-blocked (NOT-STARTED):** REQ-17 (ferray array/linalg/erf substrate
  — R-SUBSTRATE-1; the REQ-9 fix's `erf` should route through `ferray::stats`).

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED: REQ-1..9, REQ-12,
REQ-13 (impl + non-test boundary consumer + green verification under
`optimizer=None`; this iteration FIXED REQ-9 predict_proba LAMBDAS/COEFS #1931,
REQ-2 binary predict latent-sign #1932, REQ-12 multi-class LML mean #1933, and
REQ-13 OvR predict_proba closed by the REQ-9 squash fix). NOT-STARTED (the
optimization REQs reference existing #1912/#1913): REQ-10 (hyperparameter
optimization #1912/#1913 + optimizer #1934 — the headline blocker), REQ-11
(posterior-mode convergence + W-clamp #1935), REQ-14 (one_vs_one #1937), REQ-15
(LML theta-arg + gradient #1912/#1936), REQ-16 (constructor surface #1938),
REQ-17 (ferray substrate #1940).
