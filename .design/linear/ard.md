# Automatic Relevance Determination Regression

<!--
tier: 3-component
status: draft
baseline-commit: 92d2f91e280a05a1b88e808401549cecad5f249a
upstream-paths:
  - sklearn/linear_model/_bayes.py
-->

## Summary

This module mirrors scikit-learn's `sklearn.linear_model.ARDRegression`
(`sklearn/linear_model/_bayes.py:433`): a Bayesian linear regression with a
per-feature weight-precision prior (`lambda_i`). Features whose precision
exceeds `threshold_lambda` are pruned (their coefficients driven to zero),
giving automatic relevance / feature selection. The estimate is obtained by an
iterative evidence-maximization loop. The ferrolearn type is `ARDRegression<F>`
with fitted type `FittedARDRegression<F>` in `ard.rs`.

The ferrolearn implementation already carries the constructor surface
(`max_iter`, `tol`, `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2`,
`threshold_lambda`, `fit_intercept`) and — contrary to its stale module
doc-comment, which writes the update equations *without* the Gamma hyperprior
terms — the actual `fn fit` update equations DO include the
`2*alpha_1`/`2*alpha_2`/`2*lambda_1`/`2*lambda_2` terms (so this is NOT a
#464-class hyperprior-omission divergence). However, the fit loop diverges from
sklearn in three structural ways that break 2D parity (REQ-1/2/3 NOT-STARTED):
it seeds `alpha = 1.0` instead of `1/(Var(y)+eps)`, it solves the *full*
feature system every iteration instead of masking pruned columns via
`keep_lambda`, and it converges on a relative `alpha`/`lambda` delta rather than
sklearn's `sum(|coef_old - coef_|) < tol`. `predict` and the
`fit_intercept`/`HasCoefficients` surface (REQ-4/5) are SHIPPED.

## Requirements

- REQ-1: Iterative ARD fit — `coef_`, `intercept_`, `alpha_`, `lambda_`
  produced by the evidence-maximization update equations including the Gamma
  hyperprior terms (`2*alpha_1`, `2*alpha_2`, `2*lambda_1`, `2*lambda_2`), with
  per-iteration `keep_lambda` column masking, init `alpha_ = 1/(Var(y)+eps)`,
  and convergence on `sum(|coef_old - coef_|) < tol`, matching sklearn within
  tolerance on multi-feature designs.
- REQ-2: Init `alpha_ = 1/(Var(y)+eps)` and `lambda_ = ones(n_features)`
  (sklearn `_bayes.py:658-659`); constructor exposes `lambda_1`, `lambda_2`
  (default `1e-6`) and `threshold_lambda` (default `1e4`).
- REQ-3: `threshold_lambda` pruning applied *per iteration* via
  `keep_lambda = lambda_ < threshold_lambda`, dropping pruned columns from the
  solve and zeroing their coefficients (sklearn `_bayes.py:691-692`), with the
  default threshold `1e4`.
- REQ-4: `predict` returns the posterior-mean prediction `X @ coef_ + intercept_`.
- REQ-5: `fit_intercept` centering and `HasCoefficients` (`coef_`/`intercept_`)
  introspection.
- REQ-6: `compute_score` / `scores_` — value of the objective (log marginal
  likelihood) at each iteration (sklearn `_bayes.py:695-704`).
- REQ-7: `n_iter_` fitted attribute (actual iteration count at convergence,
  sklearn `_bayes.py:716`).
- REQ-8: `predict(return_std=True)` and the full `sigma_` posterior covariance
  matrix giving the predictive standard deviation (sklearn `_bayes.py:727`,
  `_bayes.py:761`).
- REQ-9: ferray substrate — array type `ferray-core`, linear algebra
  (`pinvh`/Cholesky) via `ferray::linalg` rather than `ndarray` plus the
  hand-rolled Cholesky in `fn ard_solve`.

## Acceptance criteria

- AC-1: On `X=[[1],[2],[3],[4],[5]]`, `y=[3,5,7,9,11]`, sklearn yields
  `coef_ ≈ [1.99999997]`, `intercept_ ≈ 1.00000008`, `alpha_ ≈ 2.000001e6`,
  `lambda_ ≈ [0.2500004]`, `n_iter_ = 5`. The single-feature case matches; on
  the 2-feature design `X=[[1,100],...,[6,600]]`, `y=[2,4,...,12]` sklearn
  yields `coef_ ≈ [0.01019, 0.01990]`, `alpha_ ≈ 2.500001e6`,
  `lambda_ ≈ [48.80, 2500.005]`, while ferrolearn returns
  `coef ≈ [0.0, 0.01990]`, `alpha ≈ 2.009e6`, `lambda ≈ [493087, 2475]` —
  feature 0 is wrongly pruned. (REQ-1)
- AC-2: After fit, `lambda_` is seeded such that the first solve uses
  `alpha_ = 1/(Var(y)+eps)`; ferrolearn currently hardcodes `alpha = F::one()`,
  which on the 2D design alters the pruning trajectory. (REQ-2)
- AC-3: A feature whose `lambda_i` crosses `threshold_lambda` mid-loop is
  removed from the solve for all subsequent iterations and its coefficient is
  exactly `0.0`; ferrolearn keeps every feature in the solve and prunes only
  once after the loop, so a feature transiently exceeding the threshold is
  irrecoverably zeroed. (REQ-3)
- AC-4: `fitted.predict(X)` equals `X @ coef_ + intercept_` and has length
  `n_samples`. (REQ-4)
- AC-5: With `fit_intercept=false`, `intercept_ == 0.0`;
  `coefficients()`/`intercept()` expose `coef_`/`intercept_`. (REQ-5)

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (iterative ARD fit) | NOT-STARTED | open prereq blocker #474. The hyperprior terms ARE present (`fn fit` in `ard.rs`: `new_alpha = (n_f - gamma_sum + two * self.alpha_1) / (sse + two * self.alpha_2)` and `new_lambda[i] = (gamma[i] + two * self.lambda_1) / (wi_sq + two * self.lambda_2)`), but the loop solves the full system every iteration (`fn ard_solve` over all `n_features`) with no `keep_lambda` masking, seeds `alpha = F::one()`, and converges on a relative `alpha`/`lambda` delta. Live oracle: sklearn 2D `coef_=[0.01019,0.01990]`, `lambda_=[48.80,2500.005]`; ferrolearn `coef=[0.0,0.01990]`, `lambda=[493087,2475]` — feature 0 wrongly pruned. |
| REQ-2 (init `alpha_=1/Var(y)`, params) | NOT-STARTED | open prereq blocker #475. Constructor params `lambda_1`/`lambda_2`/`threshold_lambda` ARE present (`struct ARDRegression`, `fn new` defaults `lambda_1=lambda_2=1e-6`, `threshold_lambda=1e4`), but `fn fit` hardcodes `let mut alpha = F::one()` rather than sklearn's `1.0 / (np.var(y) + eps)` (`sklearn/linear_model/_bayes.py:658`). |
| REQ-3 (per-iter `threshold_lambda` pruning) | NOT-STARTED | open prereq blocker #476. `fn fit` applies pruning ONCE after the loop (`if lambda[i] > self.threshold_lambda { w[i] = F::zero() }`), not the per-iteration `keep_lambda = lambda_ < self.threshold_lambda; coef_[~keep_lambda] = 0` that also removes columns from the solve (`sklearn/linear_model/_bayes.py:691-692`). |
| REQ-4 (predict) | SHIPPED | impl `fn predict in ard.rs` (`let preds = x.dot(&self.coefficients) + self.intercept`) mirrors sklearn `LinearModel._decision_function` used by `ARDRegression.predict` (`sklearn/linear_model/_bayes.py:761`, `X @ coef_ + intercept_`). Non-test consumer: `ferrolearn-python/src/extras.rs` (`RsARDRegression` via the `py_regressor!` macro, whose `predict` calls `FittedARDRegression::predict`). Verification: `cargo run` example yields 1D `coef≈[2.0]`, `intercept≈1.0` (matches sklearn `coef_≈[1.99999997]`, `intercept_≈1.0`). |
| REQ-5 (fit_intercept / HasCoefficients) | SHIPPED | impl `fn fit in ard.rs` centers when `self.fit_intercept` then sets `intercept = *ym - xm.dot(&w)`; `impl HasCoefficients for FittedARDRegression` returns `&self.coefficients` and `self.intercept`. Mirrors sklearn `_set_intercept` (`sklearn/linear_model/_bayes.py:729`) and the `coef_`/`intercept_` attributes (`_bayes.py:725`, `:490`). Non-test consumer: `ferrolearn-python/src/extras.rs` (`RsARDRegression` exposes `coef_`/`intercept_` through `py_regressor!`). Verification: `test_no_intercept` asserts `intercept_ == 0.0` with `fit_intercept=false`. |
| REQ-6 (compute_score / scores_) | NOT-STARTED | open prereq blocker #477. No `compute_score` constructor param and no `scores_` fitted attribute; sklearn computes the objective per iteration (`sklearn/linear_model/_bayes.py:695-704`). |
| REQ-7 (n_iter_) | NOT-STARTED | open prereq blocker #478. `FittedARDRegression` stores no iteration count; sklearn sets `self.n_iter_ = iter_ + 1` (`sklearn/linear_model/_bayes.py:716`). |
| REQ-8 (return_std / full sigma_) | NOT-STARTED | open prereq blocker #479. `fn predict` returns only the mean; `FittedARDRegression` stores `sigma` as the covariance *diagonal* (`Array1`), not the full `(n_keep, n_keep)` matrix sklearn stores (`sklearn/linear_model/_bayes.py:727`) and uses for `predict(return_std=True)` (`_bayes.py:761`). |
| REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker #480. `ard.rs` is built on `ndarray` (`use ndarray::{Array1, Array2, ...}`) with a hand-rolled Cholesky in `fn ard_solve`; the destination substrate is `ferray-core` arrays and `ferray::linalg` (R-SUBSTRATE-1). |

## Architecture

`ARDRegression<F>` (`struct ARDRegression` in `ard.rs`) is the unfitted
estimator carrying the sklearn constructor surface: `max_iter` (300), `tol`
(`1e-3`), the noise-precision Gamma-prior shape/rate `alpha_1`/`alpha_2`
(`1e-6`), the weight-precision Gamma-prior shape/rate `lambda_1`/`lambda_2`
(`1e-6`), `threshold_lambda` (`1e4`), and `fit_intercept` (`true`). These
defaults match sklearn's `__init__` (`sklearn/linear_model/_bayes.py:578-603`)
and `_parameter_constraints` (`_bayes.py:564-576`). The estimator is missing
sklearn's `compute_score`, `copy_X`, and `verbose` params; `copy_X`/`verbose`
are Python/IO ergonomics with no numerical contract, while `compute_score`
gates the `scores_` attribute (REQ-6).

`FittedARDRegression<F>` (`struct FittedARDRegression`) stores `coefficients`
(`coef_`), `intercept` (`intercept_`), `alpha` (`alpha_`), `lambda` (`lambda_`),
and `sigma` — but `sigma` is the *diagonal* of the posterior covariance
(`Array1<F>`), whereas sklearn's `sigma_` is the full
`(n_kept, n_kept)` matrix (`_bayes.py:499-500`, `:727`). The fitted accessors
`alpha`, `lambda`, `sigma`, plus the `HasCoefficients` impl, expose the
introspection surface.

`fn fit` (`impl Fit for ARDRegression`) centers `X`/`y` when `fit_intercept`,
seeds `alpha = F::one()` and `lambda = ones(n_features)`, then iterates: solve
the regularized posterior via `fn ard_solve` (Cholesky of
`alpha * Xᵀ X + diag(lambda)` plus a diagonal-of-inverse pass for the covariance
diagonal), compute `gamma_i = 1 - lambda_i * Sigma_ii`, and apply the
hyperprior-bearing `alpha`/`lambda` updates. It diverges from sklearn's fit
(`_bayes.py:644-730`) in three ways tracked as blockers: (1) sklearn seeds
`alpha_ = 1/(Var(y)+eps)` (`_bayes.py:658`); (2) sklearn maintains a boolean
`keep_lambda` mask and solves only the kept sub-block each iteration
(`update_coeff`/`update_sigma` over `X[:, keep_lambda]`, `_bayes.py:664-692`),
pruning columns mid-loop, whereas ferrolearn solves the full system and prunes
only once after the loop (`if lambda[i] > self.threshold_lambda`); (3) sklearn
converges on `sum(|coef_old - coef_|) < tol` (`_bayes.py:707`) while ferrolearn
uses a relative `alpha`/`lambda` delta. ferrolearn also clamps `alpha`/`lambda`
to `[1e-10, 1e10]`, which has no sklearn analog. On single-feature designs the
mask is a no-op so results match; on multi-feature designs the masking and init
differences flip pruning decisions, producing the 2D divergence in AC-1.

`fn predict` (`impl Predict for FittedARDRegression`) computes
`X @ coefficients + intercept`, validating the feature count. `ARDRegression`
and `FittedARDRegression` also implement `PipelineEstimator`/
`FittedPipelineEstimator` for pipeline composition.

The non-test production consumer is `RsARDRegression` in
`ferrolearn-python/src/extras.rs` (registered via `m.add_class` in
`ferrolearn-python/src/lib.rs`), which wraps `ferrolearn_linear::ARDRegression`
and `FittedARDRegression` for the Python `ARDRegression` in
`ferrolearn-python/python/ferrolearn/_extras.py`. The crate re-export lives in
`ferrolearn-linear/src/lib.rs` (`pub use ard::{ARDRegression, FittedARDRegression}`).

## Verification

Commands establishing the SHIPPED claims (REQ-4, REQ-5):

- `cargo test -p ferrolearn-linear ard` — `fn predict` length/shape, feature
  mismatch, `test_no_intercept` (intercept `0.0` under `fit_intercept=false`),
  `test_has_coefficients_length`, pipeline integration (all green).
- Live sklearn oracle vs ferrolearn (single feature, REQ-4/5 SHIPPED path):
  `python3 -c "from sklearn.linear_model import ARDRegression; import numpy as np; X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([3.,5.,7.,9.,11.]); m=ARDRegression().fit(X,y); print(m.coef_.tolist(), m.intercept_, m.alpha_, m.lambda_.tolist(), m.n_iter_)"`
  yields `coef_=[1.99999997]`, `intercept_=1.00000008`, `alpha_=2.000001e6`,
  `lambda_=[0.2500004]`, `n_iter_=5`; ferrolearn matches to `~1e-7` on this case.

Commands establishing the NOT-STARTED divergence (REQ-1/2/3):

- Live oracle (2-feature design): `python3 -c "from sklearn.linear_model import ARDRegression; import numpy as np; X=np.array([[1.,100.],[2.,200.],[3.,300.],[4.,400.],[5.,500.],[6.,600.]]); y=np.array([2.,4.,6.,8.,10.,12.]); m=ARDRegression(max_iter=1000).fit(X,y); print(m.coef_.tolist(), m.alpha_, m.lambda_.tolist())"`
  yields `coef_=[0.01019,0.01990]`, `alpha_=2.500001e6`,
  `lambda_=[48.80,2500.005]`; ferrolearn returns `coef=[0.0,0.01990]`,
  `alpha=2.009e6`, `lambda=[493087,2475]` (feature 0 wrongly pruned). This
  divergence keeps REQ-1/2/3 NOT-STARTED behind blockers #474/#475/#476; a
  failing `#[test]` pinning the 2D `coef_`/`alpha_`/`lambda_` against this
  oracle is the next critic step.

REQ-6/7/8/9 are NOT-STARTED structurally (absent `scores_`/`n_iter_`/full
`sigma_`/`return_std`; `ndarray` substrate) behind blockers #477/#478/#479/#480.
