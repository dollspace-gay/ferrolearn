# Bayesian Ridge Regression

<!--
tier: 3-component
status: draft
baseline-commit: 0ae1dd834d683d91a81d30f868af0c0eff5fe62c
upstream-paths:
  - sklearn/linear_model/_bayes.py
-->

## Summary

This module mirrors scikit-learn's `sklearn.linear_model.BayesianRidge`
(`sklearn/linear_model/_bayes.py:26`): a Bayesian formulation of ridge
regression that infers the noise precision `alpha_` and the weight precision
`lambda_` from the data by evidence maximization (MacKay 1992 / Tipping 2001
update rules), rather than using a fixed regularization strength. The
ferrolearn type is `BayesianRidge<F>` with fitted type `FittedBayesianRidge<F>`
in `bayesian_ridge.rs`.

The current ferrolearn implementation captures the *shape* of the algorithm
(center, iterate `coef_`/`alpha_`/`lambda_`, expose `predict`/`HasCoefficients`)
but **diverges numerically from sklearn** in three contract-bearing ways: it
omits the four Gamma-prior hyperparameters that enter the update equations, it
approximates the effective-degrees-of-freedom term `gamma_` with a
Cholesky-diagonal trace instead of sklearn's exact SVD eigenvalue sum, and its
`alpha_init` default is `1.0` instead of sklearn's `1/Var(y)`. Those are pinned
NOT-STARTED below.

## Requirements

- REQ-1: Iterative Bayesian fit — `coef_`, `intercept_`, `alpha_`, `lambda_`
  produced by the MacKay/Tipping evidence-maximization update equations
  *including* the Gamma hyperprior terms `2*alpha_1`, `2*alpha_2`, `2*lambda_1`,
  `2*lambda_2`, with `gamma_` the exact SVD eigenvalue sum, matching sklearn
  within tolerance.
- REQ-2: Constructor exposes `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2`
  (default `1e-6` each) per sklearn's `__init__` and `_parameter_constraints`.
- REQ-3: `alpha_init` defaults to `1/(Var(y)+eps)` when unset (sklearn `None`
  sentinel); `lambda_init` defaults to `1.0`.
- REQ-4: `predict` returns the posterior-mean prediction `X @ coef_ + intercept_`.
- REQ-5: `fit_intercept` centering and `HasCoefficients` (`coef_`/`intercept_`)
  introspection.
- REQ-6: `compute_score` / `scores_` — log marginal likelihood per iteration.
- REQ-7: `n_iter_` fitted attribute (actual iteration count at convergence).
- REQ-8: `predict(return_std=True)` and the full `sigma_` posterior covariance
  matrix giving the predictive standard deviation.
- REQ-9: `sample_weight` support (sklearn rescales X, y via `_rescale_data`).
- REQ-10: ferray substrate — array type `ferray-core`, SVD via `ferray::linalg`
  rather than `ndarray` + hand-rolled Cholesky.

## Acceptance criteria

- AC-1: On `X=[[1],[2],[3],[4],[5]]`, `y=[3,5,7,9,11]`, sklearn yields
  `alpha_ ≈ 2.000e6`, `lambda_ ≈ 0.2500004`, `coef_ ≈ [1.99999997]`,
  `intercept_ ≈ 1.00000008`, `n_iter_ = 5`. ferrolearn's `alpha()` must match
  `alpha_` within relative `1e-6` (it currently returns the clamp ceiling
  `1e10`). (REQ-1)
- AC-2: `BayesianRidge` constructed with `alpha_1=alpha_2=lambda_1=lambda_2=1e-6`
  reproduces AC-1; with those set to `0` it reproduces the no-prior limit.
  (REQ-2)
- AC-3: A `BayesianRidge` fit with `alpha_init` unset uses `1/(Var(y)+eps)` as
  the iteration-0 `alpha_` (for the AC-1 data, `1/Var(y)=0.125`), not `1.0`.
  (REQ-3)
- AC-4: `predict` length equals `n_samples` and equals `X @ coef_ + intercept_`.
  (REQ-4)
- AC-5: With `fit_intercept=false` the intercept is exactly `0.0`;
  `coefficients()` length equals `n_features`. (REQ-5)

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (evidence-max fit w/ hyperpriors) | NOT-STARTED | open prereq blocker #464. The M-step in `fn fit` of `bayesian_ridge.rs` computes `new_alpha = (n_f - gamma) / sse` and `new_lambda = gamma / w_norm_sq` — it omits sklearn's `+ 2*alpha_1` / `+ 2*alpha_2` (`_bayes.py:307`) and `+ 2*lambda_1` / `+ 2*lambda_2` (`_bayes.py:306`). It also computes `gamma` via `alpha * xtx[[i,i]] * sd_new[i]` (a Cholesky-diagonal trace approximation in `fn fit`) instead of sklearn's exact `np.sum((alpha_*eigen_vals_)/(lambda_ + alpha_*eigen_vals_))` (`_bayes.py:305`), and converges on relative change in `alpha`/`lambda` rather than sklearn's `np.sum(np.abs(coef_old_ - coef_)) < tol` (`_bayes.py:310`). Oracle: `python3 -c "from sklearn.linear_model import BayesianRidge; import numpy as np; X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([3.,5.,7.,9.,11.]); m=BayesianRidge().fit(X,y); print(m.alpha_, m.lambda_)"` → `alpha_=2000000.99..., lambda_=0.25000037...`; the ferrolearn update (emulated) yields `alpha=1e10` (clamp ceiling), a > 3-order-of-magnitude divergence in `alpha_`. |
| REQ-2 (alpha_1/alpha_2/lambda_1/lambda_2 params) | NOT-STARTED | open prereq blocker #465. `struct BayesianRidge<F>` in `bayesian_ridge.rs` has only `{max_iter, tol, alpha_init, lambda_init, fit_intercept}`; sklearn's `__init__` declares `alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6` (`_bayes.py:192-195`) and constrains them in `_parameter_constraints` (`_bayes.py:175-178`). Missing params ⇒ the REQ-1 update equations cannot be expressed. |
| REQ-3 (alpha_init default = 1/Var(y)) | NOT-STARTED | open prereq blocker #466. `fn new` in `bayesian_ridge.rs` sets `alpha_init = F::one()` (i.e. `1.0`) unconditionally; sklearn defaults `alpha_init=None` and, when `None`, sets `alpha_ = 1.0 / (np.var(y) + eps)` (`_bayes.py:266-269`). For the AC-1 data this is `0.125` vs ferrolearn's `1.0`, changing the EM trajectory. (ferrolearn's field is a non-optional `F`, so the `None` sentinel itself is also absent.) |
| REQ-4 (predict posterior mean) | SHIPPED | impl `fn predict in bayesian_ridge.rs` (`impl Predict for FittedBayesianRidge`) computes `x.dot(&self.coefficients) + self.intercept`, mirroring sklearn's `_decision_function` path (`_bayes.py:365`, the `return_std=False` branch). Non-test consumer: `ferrolearn-python/src/extras.rs` `RsBayesianRidge` (`py_regressor!` macro wraps `FittedBayesianRidge<f64>` and exposes `predict` to CPython), surfaced as `ferrolearn.BayesianRidge` in `ferrolearn-python/python/ferrolearn/_extras.py`. Verification: `cargo test -p ferrolearn-linear bayesian` (`test_predict_length`, `test_predict_feature_mismatch`). |
| REQ-5 (fit_intercept / HasCoefficients) | SHIPPED | impl `fn fit in bayesian_ridge.rs` centers X, y when `fit_intercept` and recovers `intercept = y_mean - x_mean·w` (mirrors sklearn `_set_intercept`, `_bayes.py:339`); `impl HasCoefficients for FittedBayesianRidge` (`fn coefficients`, `fn intercept`) exposes `coef_`/`intercept_`. Non-test consumer: `RsBayesianRidge` in `ferrolearn-python/src/extras.rs` (constructed with `.with_fit_intercept(fit_intercept)`, `.with_max_iter`, `.with_tol`), exposing the fitted coefficients/intercept across the PyO3 boundary. Verification: `cargo test -p ferrolearn-linear bayesian` (`test_no_intercept`, `test_has_coefficients_length`). |
| REQ-6 (compute_score / scores_) | NOT-STARTED | open prereq blocker #467. No `compute_score` field, no `scores_` attribute, no `_log_marginal_likelihood` analog; sklearn computes it per iteration (`_bayes.py:396-426`, accumulated at `:302`/`:330`). |
| REQ-7 (n_iter_) | NOT-STARTED | open prereq blocker #468. `FittedBayesianRidge` stores no iteration count; sklearn sets `self.n_iter_ = iter_ + 1` (`_bayes.py:316`). The ferrolearn loop variable is `_iter` (unused). |
| REQ-8 (predict return_std / sigma_) | NOT-STARTED | open prereq blocker #469. `FittedBayesianRidge.sigma` is only the *diagonal* of `(alpha·XᵀX + lambda·I)⁻¹` from `fn bayesian_ridge_solve`; sklearn's `sigma_` is the full `(n_features, n_features)` covariance `(1/alpha_)·Vh.T·(Vh/(eigen_vals_+lambda_/alpha_))` (`_bayes.py:333-337`) used by `predict(return_std=True)` to form `y_std = sqrt((X·sigma_·X).sum(1) + 1/alpha_)` (`_bayes.py:369-370`). `fn predict` has no `return_std` path. |
| REQ-9 (sample_weight) | NOT-STARTED | open prereq blocker #470. `fn fit` takes only `(x, y)`; sklearn accepts `sample_weight` and rescales via `_rescale_data` (`_bayes.py:254-256`). |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #471. The module uses `ndarray` (`Array1`/`Array2`) and a hand-rolled `fn cholesky_solve` / `fn cholesky_diag_inv` rather than `ferray-core` arrays and `ferray::linalg` SVD (sklearn uses `scipy.linalg.svd`, `_bayes.py:287`). R-SUBSTRATE-2. |

## Architecture

**Types.** Unfitted `BayesianRidge<F>` (`bayesian_ridge.rs`) holds
`{max_iter, tol, alpha_init, lambda_init, fit_intercept}` with builder setters
(`with_max_iter`, `with_tol`, `with_alpha_init`, `with_lambda_init`,
`with_fit_intercept`) and `Default`/`new` giving `max_iter=300`, `tol=1e-3`,
`alpha_init=1.0`, `lambda_init=1.0`, `fit_intercept=true`. Fitted
`FittedBayesianRidge<F>` holds `{coefficients, intercept, alpha, lambda, sigma}`
where `sigma` is a *vector* (the covariance diagonal), accessed via `alpha()`,
`lambda()`, `sigma()`, plus `HasCoefficients`.

**sklearn shape.** sklearn's `fit` (`_bayes.py:217`) preprocesses (center via
`_preprocess_data`, optional `_rescale_data` for weights), takes
`U, S, Vh = svd(X)` with `eigen_vals_ = S**2` (`_bayes.py:287-288`), then loops
(`_bayes.py:291`): `_update_coef_` solves the posterior mean from the SVD
factors (`_bayes.py:373-394`); `gamma_` is the exact eigenvalue sum
(`_bayes.py:305`); `lambda_`/`alpha_` are updated with the Gamma-prior terms
(`_bayes.py:306-307`); convergence is `sum|coef_old - coef| < tol`
(`_bayes.py:310`). After the loop sklearn recomputes `coef_` once more, builds
the full `sigma_` (`_bayes.py:333-337`), and sets the intercept
(`_bayes.py:339`).

**ferrolearn shape.** `fn fit` centers (when `fit_intercept`), forms `XᵀX`
explicitly, then loops calling `fn bayesian_ridge_solve` — which solves
`(alpha·XᵀX + lambda·I) w = alpha·Xᵀy` by `fn cholesky_solve` and returns
`diag((alpha·XᵀX + lambda·I)⁻¹)` via `fn cholesky_diag_inv`. This is the
*normal-equation* route, not the SVD route; it is mathematically equivalent for
`coef_` given the same `alpha`/`lambda`, but the `gamma`, hyperprior, and
`alpha_init`-default divergences (REQ-1/2/3) drive `alpha`/`lambda` — and hence
`coef_` on regularization-sensitive data — away from sklearn. The clamp
`[1e-10, 1e10]` in `fn fit` masks the `alpha_` divergence as a saturated value
(AC-1: ferrolearn `alpha → 1e10` vs sklearn `2.0e6`).

**Invariants preserved.** Centering/intercept recovery (REQ-5), prediction as
`X·coef_ + intercept_` (REQ-4), and the `≥2`-sample / shape / positive-init
validation guards in `fn fit` all hold and match sklearn's observable contract
for those facets.

## Verification

Commands establishing the SHIPPED claims (REQ-4, REQ-5):

```bash
cargo test -p ferrolearn-linear bayesian
# test_predict_length, test_predict_feature_mismatch (REQ-4),
# test_no_intercept, test_has_coefficients_length (REQ-5)
```

Live sklearn oracle pinning the NOT-STARTED divergences (REQ-1/2/3):

```bash
python3 -c "from sklearn.linear_model import BayesianRidge; import numpy as np; \
X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([3.,5.,7.,9.,11.]); \
m=BayesianRidge().fit(X,y); \
print('coef', m.coef_.tolist(), 'intercept', m.intercept_, \
      'alpha', m.alpha_, 'lambda', m.lambda_, 'n_iter', m.n_iter_, \
      '1/var(y)', 1.0/np.var(y))"
# -> coef [1.99999997...] intercept 1.00000008 alpha 2000000.99... lambda 0.25000037...
#    n_iter 5  1/var(y) 0.125
```

The `alpha_` oracle (`2.0e6`) vs the ferrolearn EM result (clamped `1e10`)
establishes that REQ-1 is NOT-STARTED: no current Rust test compares `alpha()`
or `lambda()` against the sklearn oracle, and such a test would fail. When the
hyperprior terms, the exact-`gamma_` SVD formula, and the `1/Var(y)`
`alpha_init` default land (blockers #464/#465/#466), REQ-1/2/3 flip to SHIPPED
with a characterization test pinning AC-1 against this oracle.
