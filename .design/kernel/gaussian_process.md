# Gaussian Process Regression

<!--
tier: 3-component
status: draft
baseline-commit: 8b75937c4
upstream-paths:
  - sklearn/gaussian_process/_gpr.py        # class GaussianProcessRegressor (:26), __init__ (:200), fit (:222), predict (:363), sample_y (:494), log_marginal_likelihood (:533)
-->

## Summary

`ferrolearn-kernel/src/gaussian_process.rs` mirrors scikit-learn's
`sklearn.gaussian_process.GaussianProcessRegressor` — a Bayesian nonparametric
regressor that, given a covariance kernel `k`, computes `K = k(X,X) + alpha·I`,
Cholesky-factors `K = L Lᵀ`, solves `alpha_vec = K⁻¹ y`, and predicts a posterior
mean `K(X*,X)·alpha_vec` with a predictive variance
`diag(k(X*,X*)) − ‖L⁻¹ K(X,X*)‖²` (Rasmussen & Williams Alg. 2.1,
`_gpr.py:341-360`/`:438-490`). ferrolearn exposes the unfitted
`GaussianProcessRegressor<F>` (constructor `new`/`default_rbf`, builders `alpha`/
`normalize_y`/`n_restarts_optimizer`) and the `FittedGaussianProcessRegressor<F>`
(`predict` via the `Predict` trait, `predict_with_std`, `sample_y`,
`log_marginal_likelihood`, `score`), driving the `GPKernel` family documented in
`gp_kernels.md`.

The **fixed-kernel, single-output, `normalize_y=False` correctness slice** is
value-exact against the live sklearn 1.5.2 oracle run with `optimizer=None`, and
has a real production consumer (the boundary estimator type itself, re-exported
from `lib.rs`; there is NO Python binding for GP — the consumer is the Rust
estimator API, which IS the public surface per R-DEFER-1/S5). Those REQs are
SHIPPED: `fit`+`predict` posterior mean, `predict_with_std` predictive variance,
`log_marginal_likelihood` *value* at the fitted theta, `score` (R²), and the
`alpha=1e-10` default. These are deterministic and oracle-pinnable now (using
sklearn `optimizer=None` so both sides use the SAME un-tuned kernel).

Everything that depends on tuning, target scaling, multi-output, the prior path,
or numpy's RNG diverges and is NOT-STARTED. The headline:

1. **`normalize_y` is missing the std scaling (DETERMINISTIC, FIXABLE — the
   headline divergence).** sklearn divides `y` by the population std `np.std(y)`
   (with `_handle_zeros_in_scale` guarding `std=0`) in `fit` (`_gpr.py:268-273`)
   and *rescales the predictive mean by `_y_train_std` and the variance by
   `_y_train_std²`* in `predict` (`:443`, `:484`). ferrolearn's `fit` only
   computes `y_mean = mean(y)` and centers (`y − y_mean`) — NO std — and `predict`/
   `predict_with_std`/`sample_y`/`log_marginal_likelihood` only ADD the mean back
   (no `·y_std` rescale). So `normalize_y=true` diverges from sklearn.
2. **Hyperparameter optimization ABSENT (the headline blocker).** sklearn's
   default `optimizer="fmin_l_bfgs_b"` runs L-BFGS-B on the negative
   log-marginal-likelihood (with `n_restarts_optimizer` restarts) to tune the
   kernel's `theta` (`:292-340`), using the kernel's `eval_gradient` (`:576`) and
   `bounds`. ferrolearn's `fit` NEVER optimizes — `n_restarts_optimizer` is stored
   but unused; it uses the kernel's initial hyperparameters as-is. So
   `GaussianProcessRegressor::new(kernel)` ≠ sklearn's DEFAULT GPR (which tunes);
   it equals sklearn only with `optimizer=None`.

The remaining gaps: single-output only (no `(n, n_targets)` `y` / `n_targets`),
no GP-prior prediction on an unfitted model, the `log_marginal_likelihood(y)`
API shape (cannot evaluate at arbitrary `theta` or return the gradient), the
`sample_y` RNG substrate (`Xoshiro256Plus` ≠ numpy `multivariate_normal`), the
absent `return_cov`/`copy_X_train`/`random_state` constructor surface, and the
ferray array/linalg substrate (`ndarray` + hand-rolled `cholesky`).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/gaussian_process/_gpr.py`
  - `class GaussianProcessRegressor` `:26` — `RegressorMixin`, `MultiOutputMixin`.
  - `_parameter_constraints` `:189-198` / `__init__` `:200-219` —
    `kernel=None` (→ `C(1.0, "fixed") * RBF(1.0, "fixed")`, `:238-241`),
    `alpha=1e-10`, `optimizer="fmin_l_bfgs_b"`, `n_restarts_optimizer=0`,
    `normalize_y=False`, `copy_X_train=True`, `n_targets=None`,
    `random_state=None`.
  - `fit` `:222-361` — clones the kernel; `multi_output=True` validation (`:251`);
    `normalize_y` branch `:267-278` (`_y_train_mean = np.mean(y, axis=0)`;
    `_y_train_std = _handle_zeros_in_scale(np.std(y, axis=0))`;
    `y = (y - mean) / std`); the `else` sets `mean=0, std=1` (`:275-278`);
    optimizer loop `:292-333` (`obj_func` = `-log_marginal_likelihood(theta,
    eval_gradient=True)`, `_constrained_optimization` over `kernel_.theta` /
    `kernel_.bounds`, `n_restarts_optimizer` log-uniform restarts);
    `K = kernel_(X); K[diag] += alpha` (`:342-343`); `L = cholesky(K, lower=True)`
    (`:345`); `alpha_ = cho_solve((L, lower), y_train)` (`:356`).
  - `predict` `:363-492` — `return_std`/`return_cov` mutually exclusive (`:398`);
    UNFITTED → GP PRIOR (`:410-436`: `y_mean = zeros`, `y_cov = kernel(X)` /
    `y_var = kernel.diag(X)`); FITTED POSTERIOR (`:437-492`):
    `K_trans = kernel_(X, X_train_); y_mean = K_trans @ alpha_` (`:439-440`),
    `y_mean = _y_train_std * y_mean + _y_train_mean` (`:443`);
    `V = solve_triangular(L_, K_trans.T, lower=True)` (`:450`);
    `y_var = kernel_.diag(X) - einsum("ij,ji->i", V.T, V)` (`:470-471`),
    negatives clipped to 0 with a `warnings.warn` (`:475-481`),
    `y_var = outer(y_var, _y_train_std**2)` (`:484`), `return y_mean, sqrt(y_var)`.
  - `sample_y` `:494-531` — `rng = check_random_state(random_state)` (default
    `random_state=0`); `y_mean, y_cov = self.predict(X, return_cov=True)`;
    `y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T` (`:520-522`).
  - `log_marginal_likelihood` `:533-648` — signature `(theta=None,
    eval_gradient=False, clone_kernel=True)`; `theta=None` → returns the
    precomputed `log_marginal_likelihood_value_` (`:564-567`); else rebuilds
    `K` at `theta`, Cholesky, `alpha`, and computes
    `-0.5·yᵀα − sum(log diag L) − n/2·log(2π)` (`:605-609`); `eval_gradient`
    → the `0.5·trace((ααᵀ − K⁻¹)·K_gradient)` gradient (`:611-643`).
  - `_constrained_optimization` `:650-674` — `scipy.optimize.minimize(method=
    "L-BFGS-B")` on `obj_func` over `bounds`.

## Requirements

- REQ-1: Fixed-kernel posterior mean (`normalize_y=False`). `fit` builds
  `K + alpha·I`, Cholesky-factors it, solves `alpha_vec = K⁻¹ y`, and `predict`
  returns `K(X*,X)·alpha_vec + y_mean` (`y_mean=0` here) — mirroring
  `_gpr.py:342-356` (`K[diag]+=alpha`, `cho_solve`) and `:439-440` (`K_trans @
  alpha_`). Matches sklearn run with `optimizer=None` (same un-tuned kernel).
  (Deterministic / oracle-pinnable.)
- REQ-2: Predictive variance / std. `predict_with_std` returns
  `(mean, sqrt(diag(k(X*,X*)) − sum(v²)))` with `v = L⁻¹ K(X,X*)`, negatives
  clipped to 0 — mirroring `_gpr.py:450-490` (`solve_triangular`,
  `einsum("ij,ji->i", V.T, V)`, negative-clip). Matches sklearn `optimizer=None`,
  `normalize_y=False`. (Deterministic / oracle-pinnable.)
- REQ-3: Log-marginal-likelihood VALUE at the fitted theta.
  `log_marginal_likelihood(y)` computes `-0.5·y_centeredᵀ·alpha_vec −
  sum(log diag L) − n/2·log(2π)` — equal to sklearn's
  `log_marginal_likelihood_value_` at the kernel's (un-tuned) theta
  (`_gpr.py:605-609`, `:335-337`). (Deterministic / oracle-pinnable; the
  theta-argument + gradient API is REQ-8.)
- REQ-4: `score` = R² (`RegressorMixin.score`). `FittedGaussianProcessRegressor::
  score` returns `r2_score(predict(X), y)` — sklearn's `RegressorMixin.score`
  default on `predict`. (Deterministic / oracle-pinnable.)
- REQ-5: `alpha` default = 1e-10. `new` sets `alpha = 1e-10`, matching
  `__init__(alpha=1e-10)` (`_gpr.py:204`). (Deterministic / oracle-pinnable.)
- REQ-6: Production consumer. `GaussianProcessRegressor` /
  `FittedGaussianProcessRegressor` are re-exported from `lib.rs` as the public
  boundary estimator type — the GP regressor API itself (no Python binding for
  GP exists). (Non-test consumer; grandfathered boundary type per S5.)
- REQ-7: `normalize_y` std scaling. With `normalize_y=true`, `fit` must divide
  `y` by the population std `_y_train_std = _handle_zeros_in_scale(np.std(y))`
  (`_gpr.py:270-273`) and `predict`/`predict_with_std`/`sample_y`/
  `log_marginal_likelihood` must rescale the mean by `·_y_train_std` (`:443`) and
  the variance by `·_y_train_std²` (`:484`). ferrolearn only centers by the mean
  (NO std). (Deterministic / oracle-pinnable — the fixable divergence.)
- REQ-8: Hyperparameter optimization (default `optimizer="fmin_l_bfgs_b"`).
  `fit` with the default optimizer must run L-BFGS-B on `-log_marginal_likelihood`
  (with `n_restarts_optimizer` restarts) to tune `kernel_.theta`
  (`_gpr.py:292-333`), using the kernel's `eval_gradient` (`:576`) and `bounds`.
  ferrolearn never optimizes (`n_restarts_optimizer` stored but unused).
  (Deterministic in result; depends on gp_kernels `eval_gradient` #1912 +
  `bounds` #1913 + an L-BFGS-B optimizer.)
- REQ-9: `log_marginal_likelihood(theta, eval_gradient)` API. Evaluate the LML at
  an ARBITRARY `theta` (rebuilding `K` at that theta) and optionally return the
  gradient `0.5·trace((ααᵀ − K⁻¹)·K_gradient)` (`_gpr.py:533-648`). ferrolearn's
  `log_marginal_likelihood(&self, y)` takes `y` and only evaluates at the fitted
  theta — no theta argument, no gradient. (Depends on #1912.)
- REQ-10: Multi-output `y` / `n_targets`. Support `y` of shape `(n, n_targets)`
  and the `n_targets` constructor param (`_gpr.py:251` `multi_output=True`,
  `:418` prior `n_targets`). ferrolearn is single-output (`y: Array1<F>`).
  (Deterministic / oracle-pinnable once supported.)
- REQ-11: GP-prior prediction on an unfitted model. `predict` before `fit`
  returns the GP prior (`y_mean=0`, `std=sqrt(kernel.diag(X))`,
  `cov=kernel(X)`) (`_gpr.py:410-436`). ferrolearn has no prior path (the type
  system goes `Fit → Fitted`; an unfitted estimator cannot predict). (R-DEV-3.)
- REQ-12: `sample_y` numpy-compatible RNG. sklearn draws via
  `check_random_state(random_state).multivariate_normal(y_mean, y_cov, n_samples)`
  (default `random_state=0`) (`_gpr.py:518-522`). ferrolearn uses
  `Xoshiro256Plus` + `StandardNormal` + a `mean + L_post·z` draw. Exact sample-
  value parity is numpy-RNG-substrate-blocked (R-SUBSTRATE-5) — the posterior
  MEAN/COV it draws from ARE deterministic (REQ-1/REQ-2). (RNG carve-out.)
- REQ-13: `return_cov` / `predict` covariance output. sklearn's `predict(...,
  return_cov=True)` returns the full posterior covariance `kernel_(X) − VᵀV`
  (`_gpr.py:454-465`) and rejects `return_std and return_cov` (`:398`).
  ferrolearn's `predict` returns the mean only; `predict_with_std` returns std;
  there is no covariance accessor. (Deterministic / oracle-pinnable.)
- REQ-14: Constructor surface (`copy_X_train`, `random_state`, `optimizer`,
  `n_targets`, `kernel=None` default). sklearn's `__init__` (`_gpr.py:200-219`)
  carries `copy_X_train=True`, `random_state=None`, `optimizer="fmin_l_bfgs_b"`,
  `n_targets=None`, and `kernel=None → C(1.0,"fixed")*RBF(1.0,"fixed")`.
  ferrolearn requires an explicit kernel (no `None` default), and has no
  `copy_X_train`/`random_state`/`optimizer`/`n_targets`. (R-DEV-2 ABI.)
- REQ-15: ferray substrate (R-SUBSTRATE-1). Array type → `ferray-core`;
  Cholesky / triangular solves → `ferray::linalg` (the `scipy.linalg.cholesky`/
  `cho_solve`/`solve_triangular` analog); RNG → `ferray::random` — instead of
  `ndarray` + the hand-rolled `cholesky`/`forward_solve`/`backward_solve` +
  `rand_xoshiro`/`rand_distr`.

## Acceptance criteria

All live-oracle commands use sklearn `optimizer=None` (R-CHAR-3) so both sides
use the SAME un-tuned kernel; otherwise sklearn's default tuner makes the
comparison invalid (that gap is REQ-8). Fixtures use
`X=[[0],[1],[2],[3],[4]]`, `y=[0,1,4,9,16]`, `Xs=[[0.5],[2.5]]`, `RBF(1.0)`,
`alpha=1e-10`.

- AC-1 (REQ-1): ferrolearn `fit(X,y).predict(Xs)` equals
  `GPR(kernel=RBF(1.0), alpha=1e-10, optimizer=None).fit(X,y).predict(Xs)` =
  `[0.0996264781, 5.8146715565]` element-wise to ~1e-9 (live oracle).
- AC-2 (REQ-2): ferrolearn `predict_with_std(Xs).1` equals the live oracle's
  `predict(Xs, return_std=True)[1]` = `[0.1184472918, 0.0900419083]` to ~1e-8.
  At training points the std is ≈ 0 (near-interpolation); the
  `predict_with_std_basic`/`predict_with_std_far_away` in-crate tests pin the
  qualitative shape.
- AC-3 (REQ-3): ferrolearn `log_marginal_likelihood(y)` equals the live oracle's
  `m.log_marginal_likelihood_value_` = `-138.9976379006` to ~1e-8 (same fixture,
  `optimizer=None`).
- AC-4 (REQ-4): ferrolearn `score(X,y)` equals `m.score(X,y)` = `1.0` (perfect
  near-interpolation at `alpha=1e-10`) to ~1e-9.
- AC-5 (REQ-5): `GaussianProcessRegressor::new(kernel)` reports `alpha == 1e-10`,
  matching `GPR().alpha == 1e-10` (`_gpr.py:204`).
- AC-6 (REQ-6): `lib.rs` re-exports `GaussianProcessRegressor`/
  `FittedGaussianProcessRegressor`; the regressor fits and predicts through
  `kernel.compute` in `fn fit`/`fn predict` (the boundary public API).
- AC-7 (REQ-7): live oracle with `normalize_y=True, optimizer=None`:
  `m._y_train_std == np.std(y) == 5.89915248` and
  `m.predict(Xs, return_std=True)` = `([-0.0697541470, 5.8539706737],
  [0.6987386353, 0.5311709469])`. ferrolearn `normalize_y(true)` centers by
  `y_mean=6.0` but does NOT divide by `5.899`, so its mean and (un-rescaled) std
  diverge from those values — the pinned divergence. Fix: in `fit` compute
  `y_std = pop_std(y)` (with the `std=0 → 1` guard), divide `y_centered` by it,
  store it, and multiply the predictive mean by `y_std` / variance by `y_std²` in
  `predict`/`predict_with_std`/`sample_y`/`log_marginal_likelihood`.
- AC-8 (REQ-8): `GPR(kernel=RBF(1.0)).fit(X,y)` (DEFAULT optimizer) yields a
  TUNED `m.kernel_.theta` (`length_scale` ≠ 1.0) and a higher
  `log_marginal_likelihood_value_` than the `optimizer=None` value; ferrolearn's
  `fit` leaves the kernel at `length_scale=1.0` regardless of
  `n_restarts_optimizer`. Oracle-pinnable once the optimizer + #1912/#1913 land.
- AC-9 (REQ-9): `m.log_marginal_likelihood(theta=np.log([2.0]), eval_gradient=
  True)` returns `(value, gradient)` at an arbitrary theta; ferrolearn's
  `log_marginal_likelihood(y)` cannot accept a theta or return a gradient.
  Oracle-pinnable once #1912 lands.
- AC-10 (REQ-10): `GPR(optimizer=None).fit(X, Y2)` with `Y2` of shape `(5,2)`
  predicts a `(2,2)` mean; ferrolearn's `fit` only accepts `Array1<F>`.
  Oracle-pinnable once multi-output lands.
- AC-11 (REQ-11): `GPR(kernel=RBF(1.0)).predict(Xs, return_std=True)` on an
  UNFITTED model returns `(zeros, [1.0, 1.0])` (the prior, `kernel.diag=1`);
  ferrolearn cannot predict before `fit`. Oracle-pinnable once a prior path
  exists.
- AC-12 (REQ-12): N/A — RNG carve-out; no value-parity test. ferrolearn's
  `Xoshiro256Plus` draw cannot equal numpy `multivariate_normal`; the posterior
  mean/cov it samples from (AC-1/AC-2) is the slice that ships.
- AC-13 (REQ-13): `m.predict(Xs, return_cov=True)[1]` is a `(2,2)` covariance and
  `predict(return_std=True, return_cov=True)` raises `RuntimeError`; ferrolearn
  has no covariance output and no mutual-exclusion guard. Oracle-pinnable.
- AC-14 (REQ-14): `GPR()` (no kernel) defaults to `C(1.0,"fixed")*RBF(1.0,
  "fixed")`, and exposes `copy_X_train`/`random_state`/`n_targets`/`optimizer`;
  ferrolearn `new` requires an explicit kernel and exposes none of these.
- AC-15 (REQ-15): no `ndarray`/hand-rolled `cholesky`/`rand_xoshiro` in the owned
  computation; arrays/linalg/RNG route through ferray.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (fixed-kernel posterior mean) | SHIPPED | `Fit::fit` in `gaussian_process.rs` does `let mut k_mat = self.kernel.compute(x, x); k_mat[[i,i]] += self.alpha; let l = cholesky(&k_mat)?; let alpha_vec = cholesky_solve(&l, &y_centered)`, and `Predict::predict` does `k_star.dot(&self.alpha_vec).mapv(\|v\| v + self.y_mean)` — mirrors `_gpr.py:342-356` (`K[diag]+=alpha`, `cho_solve`) and `:439-440` (`K_trans @ alpha_`). Live oracle (`optimizer=None`, `RBF(1.0)`, `alpha=1e-10`, `Xs=[[0.5],[2.5]]`): `predict = [0.0996264781, 5.8146715565]`. Non-test consumer: `lib.rs` re-exports `pub use gaussian_process::{FittedGaussianProcessRegressor, GaussianProcessRegressor}` — the boundary estimator API (no Python binding for GP; the Rust estimator IS the public surface per R-DEFER-1/S5). In-crate `fit_predict_basic`/`fit_predict_interpolation`/`single_sample`/`constant_target`. Verification: `cargo test -p ferrolearn-kernel --lib gaussian_process` (21 passed). Deterministic / oracle-pinnable. |
| REQ-2 (predictive variance / std) | SHIPPED | `FittedGaussianProcessRegressor::predict_with_std` solves `L v = K*ᵀ` by forward substitution, then `var = diag − sum(v²)` with `if var[col] < 0 { var[col] = 0 }` and `std = var.sqrt()` — mirrors `_gpr.py:450-490` (`solve_triangular(L_, K_trans.T)`, `einsum("ij,ji->i", V.T, V)`, negative-clip, `sqrt`). Live oracle (`optimizer=None`): `predict(Xs, return_std=True)[1] = [0.1184472918, 0.0900419083]`. Non-test consumer: same boundary estimator (REQ-6). In-crate `predict_with_std_basic`/`predict_with_std_far_away`/`predict_with_std_variance_nonnegative`. Deterministic / oracle-pinnable. (sklearn ALSO emits a `warnings.warn` on negative-clip `:477` — a sub-finding folded into REQ-13's output-contract surface; ferrolearn clips silently, but the clipped VALUES match.) |
| REQ-3 (LML value at fitted theta) | SHIPPED | `FittedGaussianProcessRegressor::log_marginal_likelihood` computes `-0.5·y_centered.dot(&alpha_vec) − sum(l_factor[[i,i]].ln()) − 0.5·n·ln(2π)` — equals sklearn's `log_marginal_likelihood_value_` at the (un-tuned) kernel theta (`_gpr.py:605-609`, computed in fit `:335-337`). Live oracle (`optimizer=None`): `log_marginal_likelihood_value_ = -138.9976379006`. Non-test consumer: same boundary estimator (REQ-6); `log_marginal_likelihood` is a public method on `FittedGaussianProcessRegressor`. In-crate `log_marginal_likelihood_is_finite`/`log_marginal_likelihood_prefers_right_scale`. Deterministic / oracle-pinnable. (The theta-argument + gradient API is REQ-9, NOT-STARTED.) |
| REQ-4 (score / R²) | SHIPPED | `FittedGaussianProcessRegressor::score` does `let preds = self.predict(x)?; Ok(crate::r2_score(&preds, y))` — sklearn's `RegressorMixin.score` default (R² on `predict`). Live oracle (`optimizer=None`): `m.score(X,y) = 1.0` (perfect near-interpolation at `alpha=1e-10`). Non-test consumer: same boundary estimator (REQ-6); `score` is a public method. Verification: covered by the fit/predict path tests. Deterministic / oracle-pinnable. |
| REQ-5 (alpha default 1e-10) | SHIPPED | `GaussianProcessRegressor::new` sets `alpha: F::from(1e-10).unwrap()` and the field doc-comment records `Default: 1e-10` — matches `__init__(alpha=1e-10)` (`_gpr.py:204`). Live oracle: `GPR().alpha == 1e-10`. Non-test consumer: same boundary estimator (REQ-6); `new`/`default_rbf` are the public constructors. In-crate `fit_predict_interpolation`/`f32_fit_predict` rely on the default/builder. Deterministic / oracle-pinnable. |
| REQ-6 (production consumer) | SHIPPED | `lib.rs` re-exports `pub use gaussian_process::{FittedGaussianProcessRegressor, GaussianProcessRegressor}` (line 69) and documents `GaussianProcessRegressor` in the crate `//!` (line 19). The estimator IS the public boundary type: `GaussianProcessRegressor` holds `kernel: Box<dyn GPKernel<F>>` and drives `kernel.compute` in `fn fit` and `fn predict`; `gp_kernels.md` REQ-7 cites this file as the kernel consumer. There is NO `ferrolearn-python` GP binding (confirmed: `grep -rn GaussianProcess ferrolearn-python/` → no matches) — per R-DEFER-1/S5 the boundary estimator type is the public API and its consumers are external Rust users; this REQ ships on the re-export, NOT on a Python registration. Verification: `cargo test -p ferrolearn-kernel --lib gaussian_process`. |
| REQ-7 (normalize_y std scaling) | SHIPPED | FIXED #1921: `Fit::fit` now stores `y_std` = population std (ddof=0, `_handle_zeros_in_scale` guard `std<=0 → 1`) and computes `y_centered = (y - y_mean)/y_std`; `predict`/`predict_with_std`/`sample_y`/`log_marginal_likelihood` rescale by `y_std`/`y_std²` (`predict` mean `·y_std + y_mean` `_gpr.py:443`, variance `·y_std²` `:484`). When `normalize_y=false`, `y_std=1`/`y_mean=0` so the path is byte-identical (the 5 normalize_y=False guards stay green). Oracle-verified at `normalize_y=True` (mean/std/LML/score + constant-y std=0 guard) vs the live sklearn 1.5.2: `predict(Xs, return_std=True) = ([-0.0698,5.8540],[0.6987,0.5312])` for the pin dataset, and a second non-zero-mean/non-unit-std dataset. Pinned by `divergence_normalize_y_std_scaling` + 5 `green_normalize_y_*` guards in `tests/divergence_gaussian_process.rs`. Deterministic / oracle-pinnable. |
| REQ-8 (hyperparameter optimization) | NOT-STARTED | open prereq blockers #1912 (`eval_gradient`/`dK/dθ`) + #1913 (`theta`/`bounds`/`Hyperparameter`); plus a missing L-BFGS-B optimizer. `Fit::fit` does a SINGLE fixed-kernel `K + alpha·I` Cholesky solve with NO optimizer loop — `n_restarts_optimizer` is stored on the struct but never read after construction, and the kernel's initial hyperparameters are used as-is. sklearn's default `optimizer="fmin_l_bfgs_b"` (`_gpr.py:205`) runs `_constrained_optimization` on `-log_marginal_likelihood(theta, eval_gradient=True)` over `kernel_.theta`/`kernel_.bounds` with `n_restarts_optimizer` log-uniform restarts (`:292-333`), then sets the tuned theta. This requires the kernel's `eval_gradient` (gp_kernels.md REQ-8 / blocker #1912) and `bounds` (gp_kernels.md REQ-9 / blocker #1913), neither of which the `GPKernel` trait exposes (`grep` confirms only `fn compute`/`fn diagonal` in the trait, `gp_kernels.rs`). Consequence: `GaussianProcessRegressor::new(kernel)` ≠ sklearn's DEFAULT GPR — they agree only under sklearn `optimizer=None` (which is why every SHIPPED AC pins it). Live oracle: `GPR(kernel=RBF(1.0)).fit(X,y).kernel_.theta` is tuned away from `length_scale=1.0`. Deterministic in result / oracle-pinnable once #1912/#1913 + an optimizer land. |
| REQ-9 (LML theta-arg + gradient API) | NOT-STARTED | open prereq blocker #1912 (kernel `eval_gradient`). `FittedGaussianProcessRegressor::log_marginal_likelihood(&self, y: &Array1<F>)` takes `y` and evaluates ONLY at the stored fitted theta (via `alpha_vec`); it has no `theta` argument (cannot rebuild `K` at arbitrary hyperparameters) and returns no gradient. sklearn's `log_marginal_likelihood(theta=None, eval_gradient=False, clone_kernel=True)` (`_gpr.py:533-648`) rebuilds `K` at `theta`, and with `eval_gradient` returns the `0.5·trace((ααᵀ − K⁻¹)·K_gradient)` gradient (`:611-643`) — which needs `kernel(X, eval_gradient=True)` (`:576`, blocker #1912). The VALUE at the fitted theta IS shipped (REQ-3); the theta-evaluation + gradient is the gap and is the objective REQ-8's optimizer consumes. Deterministic / oracle-pinnable once #1912 lands. |
| REQ-10 (multi-output y / n_targets) | NOT-STARTED | blocker issue to be filed by critic. `Fit<Array2<F>, Array1<F>>` accepts `y: &Array1<F>` (single-output); `alpha_vec`/`y_mean` are 1-D scalars. sklearn validates `multi_output=True` (`_gpr.py:251`), supports `y` of shape `(n, n_targets)`, the `n_targets` constructor param, and per-target `_y_train_mean`/`_y_train_std`/`alpha_` (`:269-278`, `:587-609`). Live oracle: `GPR(optimizer=None).fit(X, Y2)` with `Y2.shape=(5,2)` predicts a `(2,2)` mean; ferrolearn cannot accept a 2-D `y`. Needs a multi-output fit/predict path and per-column normalization. Deterministic / oracle-pinnable once supported. |
| REQ-11 (GP-prior unfitted predict) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-3 output contract). ferrolearn's type system is `GaussianProcessRegressor --fit--> FittedGaussianProcessRegressor`; only the FITTED type implements `Predict`, so an unfitted model cannot predict. sklearn's `predict` checks `if not hasattr(self, "X_train_")` and returns the GP PRIOR (`y_mean = zeros`, `y_var = kernel.diag(X)` → `std`, `y_cov = kernel(X)`) (`_gpr.py:410-436`). Live oracle: `GPR(kernel=RBF(1.0)).predict(Xs, return_std=True)` (UNFITTED) = `(zeros, [1.0, 1.0])`. A prior path would require predicting from an unfitted estimator (or a `prior_predict` method). Deterministic / oracle-pinnable once a prior path exists. |
| REQ-12 (sample_y numpy RNG) | NOT-STARTED | open prereq: numpy-compatible `ferray::random` (R-SUBSTRATE-5 carve-out — NO value-parity test). `FittedGaussianProcessRegressor::sample_y` builds the posterior cov `K** − VᵀV` (+1e-10 jitter), Cholesky-factors it to `l_post`, and draws `out[[i,s]] = mean[i] + sum(l_post[[i,j]]·z[j])` with `z` from `Xoshiro256Plus::seed_from_u64` + `StandardNormal`. sklearn draws via `check_random_state(random_state).multivariate_normal(y_mean, y_cov, n_samples)` (default `random_state=0`) (`_gpr.py:518-522`). Exact sample-value parity is structurally unachievable across `Xoshiro256Plus` ↔ numpy Mersenne-Twister `multivariate_normal` until `ferray::random` ships the numpy analog. The posterior MEAN and COVARIANCE the draw is built from ARE deterministic and oracle-pinnable (REQ-1/REQ-2). Carve-out: blocker, no failing value test. (Also: ferrolearn's `random_state: Option<u64>` default-reseeds from the OS when `None`; sklearn defaults `random_state=0`, a reproducible default — a sub-divergence under this REQ.) |
| REQ-13 (return_cov / covariance output) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-3 output contract). `Predict::predict` returns the mean only; `predict_with_std` returns `(mean, std)`; there is NO posterior-covariance accessor and NO `return_cov`/`return_std` mutual-exclusion guard. sklearn's `predict(X, return_std=False, return_cov=False)` returns `y_cov = kernel_(X) − VᵀV` (`_gpr.py:454-465`), raises `RuntimeError` if both flags set (`:398-401`), and emits a `warnings.warn` when clipping negative variance (`:477`). Live oracle: `m.predict(Xs, return_cov=True)[1]` is a `(2,2)` matrix. ferrolearn would need a covariance method (the `sample_y` path already computes `K** − VᵀV` internally, so the math exists but is not exposed) + the negative-variance warning. Deterministic / oracle-pinnable. |
| REQ-14 (constructor surface) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 ABI). `GaussianProcessRegressor::new(kernel)` REQUIRES an explicit kernel and exposes only `alpha`/`normalize_y`/`n_restarts_optimizer` builders. sklearn's `__init__` (`_gpr.py:200-219`) defaults `kernel=None → C(1.0,"fixed")*RBF(1.0,"fixed")` (`:238-241`) and carries `optimizer="fmin_l_bfgs_b"`, `copy_X_train=True`, `n_targets=None`, `random_state=None`. ferrolearn has `default_rbf()` (an `RBF(1.0)`, NOT the sklearn `Constant*RBF` "fixed" default) but no `None`-kernel path, no `copy_X_train` (it always `x.clone()`s — matching `copy_X_train=True`, but the param is absent), no `optimizer` selector, no `random_state`/`n_targets`. Live oracle: `GPR().kernel_` is `1**2 * RBF(length_scale=1)`. Deterministic / oracle-pinnable on the default kernel + param surface. |
| REQ-15 (ferray substrate) | NOT-STARTED | blocker issue to be filed by critic (R-SUBSTRATE-1). `gaussian_process.rs` imports `ndarray::{Array1, Array2}`, `rand::SeedableRng`, `rand_distr::{Distribution, StandardNormal}`, `rand_xoshiro::Xoshiro256Plus`, and hand-rolls `cholesky`/`forward_solve`/`backward_solve`/`cholesky_solve`. Destination: `ferray-core` (array type), `ferray::linalg` (the `scipy.linalg.cholesky`/`cho_solve`/`solve_triangular` analog sklearn uses at `_gpr.py:345,356,450`), `ferray::random` (the `numpy.random.multivariate_normal` analog for `sample_y`). Not migrated. The numpy-compatible RNG (REQ-12) and this REQ are linked — `ferray::random` is the prerequisite for `sample_y` value parity. |

## Architecture

ferrolearn splits the estimator into the unfitted `GaussianProcessRegressor<F>`
(fields `kernel: Box<dyn GPKernel<F>>`, `alpha: F`, `normalize_y: bool`,
`n_restarts_optimizer: usize`) and the fitted
`FittedGaussianProcessRegressor<F>` (`x_train`, `l_factor` = lower Cholesky `L`,
`alpha_vec` = `K⁻¹y`, `y_mean`, `kernel`), matching sklearn's pre-/post-`fit`
attribute split — sklearn keeps one class and sets `X_train_`/`L_`/`alpha_`/
`_y_train_mean`/`_y_train_std`/`kernel_` on `self` in `fit`
(`_gpr.py:289-356`). The builder API is method-chained where sklearn uses keyword
constructor args (`_gpr.py:200-219`).

The numerical core is `fn fit` (`gaussian_process.rs`), `fn predict`,
`fn predict_with_std`, `fn sample_y`, and `fn log_marginal_likelihood`. `fit`
computes `K = kernel.compute(X,X)`, adds `alpha` to the diagonal, factors
`L = cholesky(K)` (the hand-rolled lower Cholesky), and solves
`alpha_vec = cholesky_solve(L, y_centered)` (`forward_solve` then
`backward_solve`) — the Rust analog of sklearn's `scipy.linalg.cholesky` +
`cho_solve` (`_gpr.py:345-356`). This fixed-kernel solve is the whole of
ferrolearn's `fit`: there is NO optimizer loop, so `n_restarts_optimizer` is
inert and the kernel keeps its initial hyperparameters (REQ-8). `predict`/
`predict_with_std` reuse `L` and `alpha_vec`; the predictive variance solves
`L v = K*ᵀ` by forward substitution and subtracts `sum(v²)` from
`kernel.diagonal(X*)`, clipping negatives (sklearn `:470-481`).

The structural gaps mirror `gp_kernels.md`'s missing surface: REQ-8 (tuning)
needs the kernel's `eval_gradient` (#1912) and `bounds` (#1913) plus an L-BFGS-B
optimizer; REQ-9 (LML at arbitrary theta with gradient) needs `eval_gradient`
(#1912). So the un-tuned posterior is the slice that ships, and every SHIPPED
acceptance criterion pins sklearn `optimizer=None` to compare like-for-like.
The `normalize_y` divergence (REQ-7) is NOT a kernel dependency — it is entirely
local: `fit` stores only `y_mean` (population mean) and centers, and every
predict/sample/LML path only adds the mean back; the population-std divide
(`fit`) and the `·y_std`/`·y_std²` rescales (`predict`) sklearn applies
(`_gpr.py:268-273`, `:443`, `:484`) are simply absent. With `normalize_y=False`
sklearn's `_y_train_std=1`/`_y_train_mean=0`, so ferrolearn (which also leaves
`y_mean=0` in that branch) matches exactly — which is why the SHIPPED slice is
the `normalize_y=False` case.

Single-output is baked into the type (`Fit<Array2<F>, Array1<F>>`,
`Predict<Array2<F>, Output = Array1<F>>`) — multi-output (REQ-10), the GP-prior
unfitted path (REQ-11), `return_cov` (REQ-13), and the wider constructor surface
(REQ-14) are absent abstractions, not divergent values.

Invariants: `K + alpha·I` is symmetric PD for `alpha > 0` (Cholesky succeeds, else
`FerroError::NumericalInstability` — sklearn raises `LinAlgError` reworded to the
"increase alpha" message, `_gpr.py:346-353`); `predict`/`predict_with_std`/
`sample_y` require `X.ncols() == x_train.ncols()` (else `ShapeMismatch`); `fit`
rejects `n_samples < 1` (`InsufficientSamples`) and `y.len() != n_samples`
(`ShapeMismatch`); predictive variance is non-negative (clipped). At training
points with small `alpha` the posterior near-interpolates (std ≈ 0).

## Verification

Commands establishing the SHIPPED claims (run at baseline `8b75937c4`):

- `cargo test -p ferrolearn-kernel --lib gaussian_process` → 21 passed, 0 failed
  (REQ-1: `fit_predict_basic`/`fit_predict_interpolation`/`single_sample`/
  `constant_target`/`multivariate_2d`/`f32_fit_predict`; REQ-2:
  `predict_with_std_basic`/`predict_with_std_far_away`/
  `predict_with_std_variance_nonnegative`; REQ-3:
  `log_marginal_likelihood_is_finite`/`log_marginal_likelihood_prefers_right_scale`;
  error paths: `fit_rejects_mismatched_y`/`predict_rejects_wrong_features`).
- Fixed-kernel oracle (REQ-1..4, R-CHAR-3 — expected from sklearn `optimizer=None`,
  NEVER copied from ferrolearn), `X=[[0],[1],[2],[3],[4]]`, `y=[0,1,4,9,16]`,
  `Xs=[[0.5],[2.5]]`, `RBF(1.0)`, `alpha=1e-10`:
  `python3 -c "import numpy as np; from sklearn.gaussian_process import GaussianProcessRegressor as GPR; from sklearn.gaussian_process.kernels import RBF; X=np.array([[0.],[1.],[2.],[3.],[4.]]); y=np.array([0.,1.,4.,9.,16.]); Xs=np.array([[0.5],[2.5]]); m=GPR(kernel=RBF(1.0),alpha=1e-10,optimizer=None).fit(X,y); print(m.predict(Xs,return_std=True)); print(m.log_marginal_likelihood_value_); print(m.score(X,y))"`
  → mean `[0.0996264781, 5.8146715565]`, std `[0.1184472918, 0.0900419083]`,
  LML `-138.9976379006`, score `1.0`. A critic pins these as Rust `#[test]`s
  comparing `fit().predict`/`predict_with_std`/`log_marginal_likelihood`/`score`
  to the live oracle.

Open divergences pinned as FAILING tests (NOT-STARTED, oracle expected values):

- REQ-7 (normalize_y std): live oracle `normalize_y=True, optimizer=None` →
  `_y_train_std = 5.89915248`, `predict(Xs, return_std=True) =
  ([-0.0697541470, 5.8539706737], [0.6987386353, 0.5311709469])`. ferrolearn's
  `normalize_y(true).fit(...).predict_with_std(Xs)` centers by `y_mean=6.0` but
  omits the `/5.899` and `·5.899`/`·5.899²` rescales — element-wise mismatch is
  the pinned divergence. (Fixable entirely in this file.)
- REQ-8 (optimization): `GPR(kernel=RBF(1.0)).fit(X,y)` (DEFAULT optimizer) tunes
  `kernel_.theta` away from `length_scale=1.0` and raises the LML; ferrolearn's
  result is fixed at the initial kernel. Pinned once #1912/#1913 + an optimizer
  land (no green test possible before then — NOT-STARTED).
- REQ-9 (LML theta/gradient): `m.log_marginal_likelihood(np.log([2.0]),
  eval_gradient=True)` returns `(value, gradient)`; ferrolearn's API takes `y`
  and returns a scalar at the fitted theta only. Pinned once #1912 lands.
- REQ-10 (multi-output): `GPR(optimizer=None).fit(X, Y2)` (`Y2.shape=(5,2)`)
  predicts `(2,2)`; ferrolearn's `Fit` accepts only `Array1<F>`.
- REQ-11 (prior predict): unfitted `GPR(kernel=RBF(1.0)).predict(Xs,
  return_std=True) = (zeros, [1.0, 1.0])`; ferrolearn cannot predict pre-`fit`.
- REQ-13 (return_cov): `m.predict(Xs, return_cov=True)[1]` is `(2,2)`;
  `predict(return_std=True, return_cov=True)` raises `RuntimeError`; ferrolearn
  exposes neither.
- REQ-14 (constructor): `GPR().kernel_` is `1**2 * RBF(length_scale=1)` (the
  `Constant*RBF` "fixed" default); ferrolearn requires an explicit kernel.

The RNG carve-out (REQ-12) gets a blocker but **NO failing value test**
(R-DEFER-3): ferrolearn's `Xoshiro256Plus` posterior draw cannot equal numpy
`multivariate_normal` (default `random_state=0`) until `ferray::random` ships the
numpy analog — so exact `sample_y`-value parity is structurally unachievable and
is documented, not tested. The posterior mean/cov the draw is built from (REQ-1/
REQ-2) is the slice that ships regardless.

### Deterministic-vs-blocked split

- **Deterministic / oracle-pinnable NOW (SHIPPED):** REQ-1 (posterior mean),
  REQ-2 (predictive variance/std), REQ-3 (LML value), REQ-4 (score/R²), REQ-5
  (alpha default), REQ-6 (consumer) — all under sklearn `optimizer=None`,
  `normalize_y=False`, single-output.
- **Deterministic / oracle-pinnable but NOT-STARTED (missing feature or local
  divergence):** REQ-7 (normalize_y std — local fix, no kernel dep), REQ-8
  (optimization — needs #1912/#1913 + optimizer), REQ-9 (LML theta/gradient —
  needs #1912), REQ-10 (multi-output), REQ-11 (prior predict), REQ-13
  (return_cov), REQ-14 (constructor surface).
- **Substrate-blocked carve-outs (NOT-STARTED):** REQ-12 (sample_y numpy RNG —
  R-SUBSTRATE-5, NO value-parity test), REQ-15 (ferray array/linalg/random
  substrate — R-SUBSTRATE-1).

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED: REQ-1..7 (impl +
non-test boundary consumer + green verification under `optimizer=None`; REQ-7
normalize_y std scaling FIXED #1921). NOT-STARTED (the critic files per-REQ
`-l blocker` issues; the optimization REQ references existing #1912/#1913):
REQ-8 (hyperparameter optimization #1912/#1913 + optimizer #1922 — the headline
blocker), REQ-9 (LML theta-arg + gradient #1912/#1923), REQ-10 (multi-output/
n_targets), REQ-11 (GP-prior unfitted predict), REQ-12 (sample_y numpy RNG —
carve-out, no test), REQ-13 (return_cov / covariance output), REQ-14 (constructor
surface), REQ-15 (ferray substrate).
