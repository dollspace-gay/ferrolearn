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

The ferrolearn implementation now matches sklearn's SVD-based MacKay
evidence-maximization loop (closed #464/#465/#466): `fn fit` takes the thin SVD
of the centered design via `ferray::linalg::svd`, iterates the exact `gamma_`
eigenvalue sum and the Gamma-hyperprior `alpha_`/`lambda_` updates, converges on
`sum(|coef_old - coef|) < tol`, and seeds `alpha_` from `1/(Var(y)+eps)` when
`alpha_init` is `None`. REQ-1/2/3, the SVD facet of REQ-10, and now REQ-6/7/8/9
(closed #2161) are SHIPPED below: `with_compute_score` + `fn scores` (the exact
`_log_marginal_likelihood`), `n_iter` (`iter_ + 1`), the full `(n_features,
n_features)` `sigma_full` + `predict_with_std`, and
`fn fit_with_sample_weight` (sqrt-weight rescaling). The remaining
array-type migration off `ndarray` stays NOT-STARTED (REQ-10 array facet,
tracked by #471).

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
  `intercept_ ≈ 1.00000008`, `n_iter_ = 5`. The SVD/MacKay rewrite now matches
  the converged `coef_`/`alpha_`/`lambda_` (no clamp ceiling); the parity is
  exercised on the regularization-sensitive 30×5 design in
  `divergence_bayesian_ridge_fit_coef_alpha_lambda`. (REQ-1)
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
| REQ-1 (evidence-max fit w/ hyperpriors) | SHIPPED | closed #464. `fn fit` for `BayesianRidge` (`bayesian_ridge.rs`) runs the MacKay/Tipping loop: per-iteration `update_coef` posterior mean (`_bayes.py:373-394`), exact `gamma = sum((alpha*eigen_vals)/(lambda + alpha*eigen_vals))` (`_bayes.py:305`), `lambda = (gamma + 2*lambda_1) / (sum(coef^2) + 2*lambda_2)` (`_bayes.py:306`), `alpha = (n - gamma + 2*alpha_1) / (rmse + 2*alpha_2)` (`_bayes.py:307`), converging on `sum(|coef_old - coef|) < tol` (`_bayes.py:310`), with a final `update_coef` after the loop (`_bayes.py:322`). Non-test consumer: `RsBayesianRidge` in `ferrolearn-python/src/extras.rs` (`py_regressor!` over `FittedBayesianRidge<f64>`). Verified by `divergence_bayesian_ridge_fit_coef_alpha_lambda` (coef_ to ~1e-3, alpha_/lambda_ to ~1e-2 vs the live oracle SK_COEF/SK_ALPHA/SK_LAMBDA) plus `oracle_bayesian_ridge_y_scaled_var_init` and `oracle_bayesian_ridge_random_state_7`. |
| REQ-2 (alpha_1/alpha_2/lambda_1/lambda_2 params) | SHIPPED | closed #465. `struct BayesianRidge<F>` (`bayesian_ridge.rs`) gains fields `alpha_1, alpha_2, lambda_1, lambda_2` (default `1e-6` in `fn new`) with setters `with_alpha_1`/`with_alpha_2`/`with_lambda_1`/`with_lambda_2`, mirroring `_bayes.py:192-195` and `_parameter_constraints` (`_bayes.py:175-178`). Consumed in the M-step of `fn fit` (`lambda` / `alpha` updates). Verified by `test_default_constructor` / `test_builder_setters`. |
| REQ-3 (alpha_init default = 1/Var(y)) | SHIPPED | closed #466. `alpha_init` is now `Option<F>` (default `None`, the sklearn sentinel) on `struct BayesianRidge`; `fn fit` sets `alpha = 1 / (var(y) + eps)` when `None` (`_bayes.py:266-269`) via `fn variance`, and `lambda_init: Option<F>` defaults to `1.0` (`_bayes.py:270-271`). The y-scaled `oracle_bayesian_ridge_y_scaled_var_init` case (where `1/Var(y) ≈ 2.65e-6`) exercises this init and matches the oracle. |
| REQ-4 (predict posterior mean) | SHIPPED | impl `fn predict in bayesian_ridge.rs` (`impl Predict for FittedBayesianRidge`) computes `x.dot(&self.coefficients) + self.intercept`, mirroring sklearn's `_decision_function` path (`_bayes.py:365`, the `return_std=False` branch). Non-test consumer: `ferrolearn-python/src/extras.rs` `RsBayesianRidge` (`py_regressor!` macro wraps `FittedBayesianRidge<f64>` and exposes `predict` to CPython), surfaced as `ferrolearn.BayesianRidge` in `ferrolearn-python/python/ferrolearn/_extras.py`. Verification: `cargo test -p ferrolearn-linear bayesian` (`test_predict_length`, `test_predict_feature_mismatch`). |
| REQ-5 (fit_intercept / HasCoefficients) | SHIPPED | impl `fn fit in bayesian_ridge.rs` centers X, y when `fit_intercept` and recovers `intercept = y_mean - x_mean·w` (mirrors sklearn `_set_intercept`, `_bayes.py:339`); `impl HasCoefficients for FittedBayesianRidge` (`fn coefficients`, `fn intercept`) exposes `coef_`/`intercept_`. Non-test consumer: `RsBayesianRidge` in `ferrolearn-python/src/extras.rs` (constructed with `.with_fit_intercept(fit_intercept)`, `.with_max_iter`, `.with_tol`), exposing the fitted coefficients/intercept across the PyO3 boundary. Verification: `cargo test -p ferrolearn-linear bayesian` (`test_no_intercept`, `test_has_coefficients_length`). |
| REQ-6 (compute_score / scores_) | SHIPPED | closed #2161. `with_compute_score` on `struct BayesianRidge` (default `false`, `_bayes.py:198`); `fn fit_with_sample_weight` calls `fn log_marginal_likelihood` — the exact `_bayes.py:396-426` LML (Gamma-hyperprior terms `lambda_1*log λ − lambda_2*λ + alpha_1*log α − alpha_2*α` plus `0.5*(p·log λ + n·log α − α·rmse − λ·‖coef‖² + logdet_sigma − n·log 2π)`, with `logdet_sigma = −Σ log(λ + α·eigen_vals)` for `n > p`) — per iteration (`:302`) plus once post-loop (`:330`), stored as `scores` (length `n_iter+1`) with getter `fn scores`. Non-test consumer: `RsBayesianRidge::scores_` getter in `ferrolearn-python/src/extras.rs` → `_extras.py::BayesianRidge.scores_`. Verified by `divergence_bayesian_ridge_scores_ac1`/`_30x5_final` (Rust, final LML to 1e-6 vs live oracle) + `test_bayesian_ridge_scores_matches_sklearn` (pytest). |
| REQ-7 (n_iter_) | SHIPPED | closed #2161. `FittedBayesianRidge.n_iter` set to `last_iter + 1` in `fn fit_with_sample_weight` (sklearn `self.n_iter_ = iter_ + 1`, `_bayes.py:316`); getter `fn n_iter`. Non-test consumer: `RsBayesianRidge::n_iter_` getter (`extras.rs`) → `_extras.py::BayesianRidge.n_iter_`. Verified by `divergence_bayesian_ridge_n_iter` (== 5, the live-oracle value) + `test_bayesian_ridge_n_iter_matches_sklearn` (pytest, == sklearn). |
| REQ-8 (predict return_std / sigma_) | SHIPPED | closed #2161. `FittedBayesianRidge.sigma_full` is the full `(n_features, n_features)` covariance `(1/alpha_)·Vhᵀ·(Vh/(eigen_vals_+lambda_/alpha_))` (`_bayes.py:333-337`), getter `fn sigma_full` (the `fn sigma` diagonal is preserved); `fn predict_with_std` returns `(mean, sqrt((X·sigma_·X).sum(1) + 1/alpha_))` (`_bayes.py:367-371`). Non-test consumer: `RsBayesianRidge::predict(return_std=True)` + `sigma_` getter (`extras.rs`) → `_extras.py::BayesianRidge.predict`/`sigma_`. Verified by `divergence_bayesian_ridge_return_std_ac1` (Rust) + `test_bayesian_ridge_return_std_matches_sklearn`/`_sigma_full_matches_sklearn` (pytest, to rtol 1e-5). |
| REQ-9 (sample_weight) | SHIPPED | closed #2161. `fn fit_with_sample_weight(x, y, Option<&Array1<F>>)` rescales centered `(X, y)` by `sqrt(sample_weight)` via `fn rescale_data` (sklearn `_rescale_data`, `_bayes.py:254-256`), with weighted centering offsets via `fn weighted_means`; `Fit::fit` delegates `None` (byte-identical, verified by `test_fit_with_none_sample_weight_matches_fit`). Non-test consumer: `RsBayesianRidge::fit(x, y, sample_weight=None)` (`extras.rs`) → `_extras.py::BayesianRidge.fit`. Verified by `divergence_bayesian_ridge_sample_weight` (Rust, coef_/intercept_/alpha_/lambda_ vs live oracle) + `test_bayesian_ridge_sample_weight_matches_sklearn` (pytest). |
| REQ-10 (ferray substrate) | SHIPPED (SVD) | `fn fit` now computes the design-matrix SVD through `ferray::linalg::svd` (`ferray-linalg/src/decomp/svd.rs:40`) in `fn svd_thin`, bridging ndarray↔ferray at that boundary (R-SUBSTRATE-4), replacing the hand-rolled Cholesky kernels — mirroring sklearn's `scipy.linalg.svd(X, full_matrices=False)` (`_bayes.py:287`). The remaining array-type migration off `ndarray` (`Array1`/`Array2`) to `ferray-core` is tracked by #471. |

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

**ferrolearn shape.** `fn fit` delegates to `fn fit_with_sample_weight`, which
centers (when `fit_intercept`, using `fn weighted_means` under a
`sample_weight`), optionally rescales `(X, y)` by `sqrt(sample_weight)` via
`fn rescale_data` (REQ-9), takes the thin SVD `X = U S Vᵀ` through `fn svd_thin`
(`ferray::linalg::svd`), and runs the MacKay loop: per-iteration `fn update_coef`
posterior mean, the exact `gamma_` eigenvalue sum, and the Gamma-hyperprior
`alpha_`/`lambda_` updates, converging on `sum(|coef_old - coef|) < tol`. When
`compute_score` is set it accumulates `fn log_marginal_likelihood` per iteration
plus once post-loop into `scores` (REQ-6); it records `n_iter = last_iter + 1`
(REQ-7); after the loop it recomputes `coef_`, builds the full `(n_features,
n_features)` `sigma_full` (whose diagonal is the preserved `sigma` vector,
REQ-8), and sets the intercept. `fn predict_with_std` then forms the predictive
std from `sigma_full` (REQ-8). This is the SVD route end-to-end — the old
normal-equation/`bayesian_ridge_solve`/`[1e-10, 1e10]`-clamp implementation
described in earlier drafts has been fully replaced (the `alpha_` clamp no
longer exists).

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
