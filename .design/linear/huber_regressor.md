# Huber Regressor

<!--
tier: 3-component
status: draft
baseline-commit: 0573632231b565fd230151d43969909d96ba3aed
upstream-paths:
  - sklearn/linear_model/_huber.py
-->

## Summary

`HuberRegressor<F>` in `ferrolearn-linear` mirrors scikit-learn's
`sklearn.linear_model.HuberRegressor` — an L2-regularized linear regressor that
is robust to outliers by minimizing the Huber loss (quadratic for small
residuals, linear beyond a threshold). scikit-learn jointly optimizes the
parameter vector `[coef, intercept, scale]` with `scipy.optimize.minimize(method="L-BFGS-B")`
(`sklearn/linear_model/_huber.py:325`), where the scale `sigma` is part of the
objective and bounded strictly positive. The ferrolearn implementation instead
fits via **Iteratively Reweighted Least Squares (IRLS)** with **no scale
parameter**, which is a material algorithmic divergence: its fitted coefficients
do not match sklearn on outlier data, and it exposes neither `scale_` nor
`outliers_`.

## Requirements

- REQ-1: Fit produces `coef_` / `intercept_` / `scale_` by jointly minimizing the
  scale-aware Huber objective, matching sklearn's `HuberRegressor.fit` on outlier
  data.
- REQ-2: `epsilon` constructor parameter, default `1.35`, controlling the
  quadratic/linear threshold.
- REQ-3: `alpha` L2 penalty applied to the coefficients only (intercept and scale
  not penalized).
- REQ-4: The scale `sigma` is jointly estimated as part of the optimized parameter
  vector and bounded strictly positive.
- REQ-5: `outliers_` boolean mask set where `|y - X·coef - intercept| > scale * epsilon`.
- REQ-6: `predict` computes `X · coef + intercept`.
- REQ-7: `fit_intercept` parameter honored, and the fitted model exposes
  coefficients and intercept via `HasCoefficients`.
- REQ-8: Fitted model exposes the `scale_` attribute.
- REQ-9: Fitted model exposes `n_iter_` (number of optimizer iterations).
- REQ-10: `warm_start` constructor parameter, reusing previously fitted
  parameters as the optimizer's starting point.
- REQ-11: `sample_weight` support in `fit`.
- REQ-12: Computation lives on the ferray substrate (array type, linear algebra)
  rather than `ndarray` plus hand-rolled Cholesky/Gaussian solvers.

## Acceptance criteria

- AC-1 (REQ-1): On `X = randn(50,3); y = X@[1,2,-1] + 0.1·randn(50); y[:5]+=20`,
  fitted `coef_` matches sklearn within `1e-3` (sklearn-deterministic L-BFGS
  convergence). Currently FAILS — see Verification.
- AC-2 (REQ-2): `HuberRegressor::<f64>::new().epsilon == 1.35`; `epsilon <= 1.0`
  is rejected with `FerroError::InvalidParameter`.
- AC-3 (REQ-3): With `fit_intercept = true`, increasing `alpha` shrinks `coef_`
  toward zero but does not shrink the intercept by the penalty.
- AC-4 (REQ-4): A `scale_` value `> 0` is produced and is part of the optimized
  parameter vector. Currently no scale parameter exists.
- AC-5 (REQ-5): `outliers_[i]` equals `|residual_i| > scale_ * epsilon` and
  matches sklearn's boolean mask on the AC-1 dataset. Currently no `outliers_`.
- AC-6 (REQ-6): `predict(X)` returns `X · coef + intercept` with length `n_samples`.
- AC-7 (REQ-7): `fit_intercept = false` yields intercept `0.0`; `coefficients()`
  has length `n_features`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (joint L-BFGS fit matches sklearn) | SHIPPED | closes #495. `fn fit` in `huber_regressor.rs` now minimizes the scale-aware Huber objective `fn huber_loss_and_gradient` (a line-by-line translation of `sklearn/linear_model/_huber.py:18` `_huber_loss_and_gradient`) over `[coef, intercept, log_sigma]` with `crate::optim::lbfgs::LbfgsOptimizer` — the joint optimization sklearn performs at `sklearn/linear_model/_huber.py:325`. The scale is reparameterized `sigma = exp(log_sigma)` (chain-rule gradient) to honor sklearn's `sigma >= eps*10` bound (`_huber.py:323`) on the unconstrained in-repo solver; the intercept is a fit parameter, NOT mean-centering. A Huber-IRLS warm start (`fn irls_warm_start`) seeds the solve near the convex minimum so it converges. Non-test consumer: `ferrolearn-python/src/extras.rs` `RsHuberRegressor`. Live oracle on the outlier dataset — sklearn `coef_=[0.990185,1.979965,-0.999142], intercept_=0.009502, scale_=0.081961`; ferrolearn now matches to 1e-3 (`coef=[0.990185,1.979964,-0.999141], intercept=0.009503, scale=0.081961`). Verification: `cargo test -p ferrolearn-linear --test divergence_huber_fit` (`divergence_huber_fit_outlier_dataset`, `divergence_huber_fit_not_convergence_max_iter`, `parity_huber_fit_outlier_dataset_epsilon_1_5`, all green). |
| REQ-2 (epsilon default 1.35) | SHIPPED | impl `pub fn new` in `huber_regressor.rs` sets `epsilon: F::from(1.35)`, mirroring sklearn `sklearn/linear_model/_huber.py:262` (`epsilon=1.35`). `fit` rejects `epsilon <= 1.0` (`FerroError::InvalidParameter`), tighter than sklearn's `[1, inf)` constraint (`_huber.py:251`) but covering the same intent. Non-test consumer: `ferrolearn-python/src/extras.rs` `RsHuberRegressor` (`HuberRegressor::<f64>::new().with_epsilon(epsilon)`). Verification: `cargo test -p ferrolearn-linear --lib huber` (`test_default_constructor`, `test_epsilon_too_small_error`). |
| REQ-3 (alpha L2 on coef only) | SHIPPED | impl `fn weighted_ridge_solve` in `huber_regressor.rs` adds `alpha` only to the `n_features × n_features` normal-equation diagonal (`xtwx[[i,i]] += alpha`); the intercept is recovered separately via mean-centering in `fit` (`*ym - xm.dot(&w)`), so it is never penalized. Mirrors sklearn `sklearn/linear_model/_huber.py:111` (`grad[:n_features] += alpha * 2.0 * w`) and `:124` (`loss += alpha * np.dot(w, w)`) — penalty on `w` only, scale/intercept excluded. Non-test consumer: `RsHuberRegressor` (`.with_alpha(alpha)`). Verification: `cargo test -p ferrolearn-linear --lib huber` (`test_l2_regularization_shrinks_coefficients`). |
| REQ-4 (scale_ jointly estimated + bounded >0) | SHIPPED | closes #496. `sigma` is the last element of the optimized parameter vector in `fn fit` / `fn huber_loss_and_gradient`, reparameterized `sigma = exp(log_sigma)` and floored at `f64::EPSILON * 10.0` inside the objective + on extraction, mirroring sklearn `sklearn/linear_model/_huber.py:317` (init `1`) and `:323` (`bounds[-1][0] = np.finfo(np.float64).eps * 10`). Surfaced on `FittedHuberRegressor` as the `scale` field and the `scale()` accessor. Live oracle `scale_ ≈ 0.081961`; ferrolearn matches to 1e-3. Non-test consumer: `RsHuberRegressor`. Verification: `divergence_huber_fit_outlier_dataset` (scale assertion), `test_scale_positive`. |
| REQ-5 (outliers_ mask) | SHIPPED | closes #497. `fn fit` sets `outliers = residual.mapv(|r| r > band)` where `residual = |y - X·coef - intercept|` and `band = scale * epsilon`, a direct translation of sklearn `sklearn/linear_model/_huber.py:350-351` (`self.outliers_ = residual > self.scale_ * self.epsilon`). Surfaced on `FittedHuberRegressor` as the `outliers` field and the `outliers()` accessor. Verification: `divergence_huber_fit_outlier_dataset` (flags the 5 injected outliers), `parity_huber_fit_outlier_dataset_epsilon_1_5` (outlier count == sklearn's 11), `test_outliers_mask_length_and_band`. |
| REQ-6 (predict) | SHIPPED | impl `Predict for FittedHuberRegressor` (`fn predict`) in `huber_regressor.rs` returns `x.dot(&self.coefficients) + self.intercept`, mirroring sklearn's `LinearModel._decision_function` contract (`HuberRegressor` inherits `LinearModel`, `_huber.py:128`). Non-test consumer: `ferrolearn-python/src/extras.rs` `RsHuberRegressor` (the `py_regressor!` macro wires `predict` through `FittedHuberRegressor<f64>`). Verification: `cargo test -p ferrolearn-linear --lib huber` (`test_predict_length`, `test_predict_feature_mismatch`). |
| REQ-7 (fit_intercept / HasCoefficients) | SHIPPED | impl `fn with_fit_intercept` + the mean-centering branch in `fit` (`if self.fit_intercept { ... }`, else intercept `F::zero()`) and impl `HasCoefficients for FittedHuberRegressor` (`fn coefficients`, `fn intercept`) in `huber_regressor.rs`, mirroring sklearn `sklearn/linear_model/_huber.py:344` (`if self.fit_intercept: self.intercept_ = parameters[-2] else self.intercept_ = 0.0`) and the `coef_`/`intercept_` attributes (`:178-188`). Non-test consumer: `RsHuberRegressor` (`.with_fit_intercept(fit_intercept)`; coefficients/intercept surfaced via the `py_regressor!` wrapper). Verification: `cargo test -p ferrolearn-linear --lib huber` (`test_no_intercept`, `test_has_coefficients_length`). |
| REQ-8 (scale_ attribute) | SHIPPED | closes #498. `FittedHuberRegressor` carries the `scale` field set from `parameters[-1].exp()` (floored), exposed via `pub fn scale` — mirroring sklearn `self.scale_ = parameters[-1]` (`sklearn/linear_model/_huber.py:343`). Shipped together with REQ-4. Verification: `test_scale_positive`, the `scale_` assertions in `divergence_huber_fit_outlier_dataset` / `parity_huber_fit_outlier_dataset_epsilon_1_5`. |
| REQ-9 (n_iter_) | NOT-STARTED | open prereq blocker #499. The IRLS loop in `fit` does not retain its iteration count; sklearn records `self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)` (`sklearn/linear_model/_huber.py:342`). |
| REQ-10 (warm_start) | NOT-STARTED | open prereq blocker #500. No `warm_start` parameter on `HuberRegressor`; sklearn's `warm_start` reuses `[coef_, intercept_, scale_]` as the optimizer seed (`sklearn/linear_model/_huber.py:265`, `:308`). |
| REQ-11 (sample_weight) | NOT-STARTED | open prereq blocker #501. `fit(&self, x, y)` has no weight argument; sklearn threads `sample_weight` through `_huber_loss_and_gradient` (`sklearn/linear_model/_huber.py:306`, `:18`). |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #502. The module is built on `ndarray::{Array1, Array2}` with hand-rolled `cholesky_solve`/`gaussian_solve`; per R-SUBSTRATE the destination is `ferray-core` arrays and `ferray::linalg`. |

## Architecture

**Unfitted type.** `HuberRegressor<F>` holds `epsilon`, `alpha`, `max_iter`,
`tol`, `fit_intercept` (`pub struct HuberRegressor<F>` in `huber_regressor.rs`),
constructed via `new` (defaults `epsilon=1.35, alpha=1e-4, max_iter=100,
tol=1e-5, fit_intercept=true`) and builder setters `with_epsilon`,
`with_alpha`, `with_max_iter`, `with_tol`, `with_fit_intercept`. The parameter
names and defaults mirror sklearn's `__init__` (`sklearn/linear_model/_huber.py:259-274`)
except sklearn additionally has `warm_start` (REQ-10) and ferrolearn enforces
`epsilon > 1.0` rather than `epsilon >= 1.0` (sklearn `_parameter_constraints`,
`_huber.py:251`).

**Fitted type.** `FittedHuberRegressor<F>` stores `coefficients: Array1<F>`,
`intercept: F`, `scale: F` (sklearn `scale_`) and `outliers: Array1<bool>`
(sklearn `outliers_`), with `scale()` / `outliers()` accessors
(`sklearn/linear_model/_huber.py:343-351`). `n_iter_` (REQ-9) is still absent —
the in-repo `LbfgsOptimizer` returns only the final parameter vector, not its
iteration count.

**Optimization — joint L-BFGS (parity).** `fn fit` now mirrors sklearn: it
minimizes `fn huber_loss_and_gradient` — a translation of
`_huber_loss_and_gradient` (`sklearn/linear_model/_huber.py:18`) — over the joint
vector `w = [coef..., intercept?, log_sigma]` with `crate::optim::lbfgs`. The
per-sample loss is
`n·sigma + Σ_inlier r²/sigma + Σ_outlier (2·epsilon·|r| − sigma·epsilon²) + alpha·||coef||²`,
split at `|r| > epsilon·sigma` (`_huber.py:67`), `alpha` on `coef` only. The
scale is reparameterized `sigma = exp(log_sigma)` (chain-rule gradient
`∂L/∂log_sigma = sigma·∂L/∂sigma`) to honour sklearn's `sigma >= eps*10` bound
(`_huber.py:323`) on the unconstrained in-repo optimizer; `sigma` is also floored
in the objective so the solve stays stable as `sigma → 0` on near-perfect fits.
The intercept is a fit parameter optimized jointly, NOT recovered by
mean-centering. A few Huber-IRLS reweighting steps (`fn irls_warm_start`) seed
the optimizer near the convex minimum — robustly (the Huber weights down-weight
outliers, so a plain OLS warm start's poisoning is avoided) — which the slower
in-repo L-BFGS needs to converge on poorly-scaled or heavy-outlier targets
(sklearn's bounded L-BFGS-B does not). The old IRLS/`weighted_ridge_solve` path
is removed. **Limitation:** the in-repo `LbfgsOptimizer` is not scipy's
L-BFGS-B; on pathological tiny single-feature datasets with a perfect inlier line
plus a large outlier it can still stall — tracked as the optimizer-quality
blocker #504 (related REQ-12 / #502).

**Pipeline integration.** `PipelineEstimator`/`FittedPipelineEstimator` impls
wrap `fit`/`predict` for `ferrolearn-core` pipelines — internal plumbing, not a
sklearn-surface requirement.

**Consumers.** `HuberRegressor`/`FittedHuberRegressor` are re-exported from
`ferrolearn-linear/src/lib.rs` and consumed by the PyO3 binding
`ferrolearn-python/src/extras.rs` (`RsHuberRegressor`, the `_RsHuberRegressor`
class), which is the production marshalling path that backs
`ferrolearn.HuberRegressor` (`ferrolearn-python/python/ferrolearn/_extras.py`).
Per R-DEFER-5/S5 the boundary estimator type IS the public API; this binding is
the non-test production consumer for the SHIPPED REQs.

## Verification

Commands establishing the SHIPPED claims and the REQ-1 divergence:

- `cargo test -p ferrolearn-linear --lib huber` — 21 huber unit tests pass
  (constructor defaults, validation errors, intercept handling, predict,
  `HasCoefficients`, L2 shrink, pipeline). Pins REQ-2, REQ-3, REQ-6, REQ-7.
- Live sklearn oracle (REQ-1 / AC-1), on outlier data:
  ```
  python3 -c "from sklearn.linear_model import HuberRegressor; import numpy as np; \
    rng=np.random.RandomState(0); X=rng.randn(50,3); y=X@np.array([1,2,-1])+0.1*rng.randn(50); \
    y[:5]+=20; m=HuberRegressor().fit(X,y); print(m.coef_.tolist(), m.intercept_, m.scale_)"
  # -> [0.9902, 1.9800, -0.9991]  0.00950  0.08196   (n_iter_=15)
  ```
  Fitting the SAME `X,y` through `HuberRegressor::<f64>::new().fit(&x,&y)` yields
  `coef = [1.7419, 2.3571, -0.6434]`, `intercept = 1.7489` — a parity failure of
  ~0.75 on `coef[0]` and ~1.74 on the intercept. REQ-1 is therefore NOT-STARTED
  (blocker #495); the remaining scale/attribute REQs (#496–#499) are prerequisites
  of a faithful joint-optimization rewrite.

A faithful REQ-1 fix replaces IRLS with the scale-aware L-BFGS objective of
`_huber_loss_and_gradient`, pinned as a `#[test]` whose expected `coef_`/`scale_`
come from the live oracle above (R-CHAR-3), and surfaced through the
`RsHuberRegressor` binding via pytest-vs-sklearn.
