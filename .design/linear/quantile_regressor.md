# Quantile Regressor

<!--
tier: 3-component
status: draft
baseline-commit: fb384c5501bc5e874b1da081b96f427e3de881b9
upstream-paths:
  - sklearn/linear_model/_quantile.py
-->

## Summary

`ferrolearn-linear/src/quantile_regressor.rs` mirrors scikit-learn's
`sklearn.linear_model.QuantileRegressor`: a linear model that predicts a
conditional `quantile` of the response by minimizing the pinball (check) loss
with an L1 penalty. scikit-learn solves the exact quantile-regression **linear
program** with HiGHS (`scipy.optimize.linprog`). ferrolearn now solves the
**same exact LP** with a self-contained two-phase primal simplex (Bland's
anti-cycling rule) in `mod lp in quantile_regressor.rs`, reaching sklearn's
HiGHS vertex (`coef_`/`intercept_` parity, issue #340 closed; intercept-as-LP-
variable #506 closed; L1 sparse vertex #332 closed). Parameter handling, the
quantile asymmetry, prediction, and `HasCoefficients` are in place; `n_iter_`,
the `solver`/`solver_options` params, and the ferray substrate are not yet.

## Requirements

- REQ-1: Fit coefficients/intercept by solving the quantile-regression linear
  program (`coef_`, `intercept_` match sklearn's HiGHS LP optimum), where the
  LP variables are `[intercept+, intercept-, coef+, coef-, u, v] >= 0`,
  objective `c·x` with residual costs `quantile·u + (1-quantile)·v` and L1 cost
  `alpha·n_samples` on `coef±`, equality `A_eq·x = y`, and recovery
  `coef_ = coef+ - coef-`, `intercept_ = intercept+ - intercept-`.
- REQ-2: Scale the L1 penalty by `n_samples` (`alpha_eff = sum(sample_weight) *
  alpha`) so `alpha` carries the same meaning as sklearn's, and do not penalize
  the intercept.
- REQ-3: Honor `quantile ∈ (0,1)` with the correct pinball asymmetry —
  cost/weight `quantile` on positive residuals, `1 - quantile` on negative.
- REQ-4: `predict` returns `X @ coef_ + intercept_`.
- REQ-5: Constructor exposes `quantile`, `alpha`, `fit_intercept` with sklearn
  defaults (`0.5`, `1.0`, `true`); fitted model exposes `coef_`/`intercept_`
  via `HasCoefficients`.
- REQ-6: Expose `n_iter_` — the number of solver iterations performed.
- REQ-7: Expose the `solver` (default `"highs"`) and `solver_options`
  constructor parameters with sklearn's `_parameter_constraints` validation.
- REQ-8: Compute on the ferray substrate (`ferray-core` arrays, `ferray::linalg`
  factorizations) rather than `ndarray` + hand-rolled Cholesky/Gauss.

## Acceptance criteria

- AC-1 (REQ-1): on the oracle dataset `rng=RandomState(0); X=randn(30,3);
  y=X@[1,2,-1]+0.5·randn(30)`, with `quantile=0.5, alpha=0.0`, `coef_` and
  `intercept_` match sklearn within `1e-6` (sklearn:
  `coef=[0.8992,2.0000,-0.8874]`, `intercept=0.3243`).
- AC-2 (REQ-1, quantile dependence): with `quantile=0.8, alpha=0.0`,
  `intercept_` matches sklearn `0.8815` within `1e-6` — i.e. the intercept must
  change with the quantile (it currently does not).
- AC-3 (REQ-2): with `alpha=1.0`, `coef_` is the exact sparse LP solution
  `[0,0,0]` (sklearn drives all coefficients to exactly zero), not merely
  near-zero.
- AC-4 (REQ-3): `predict` increases monotonically in `quantile` on a fixed test
  point for a noisy dataset; pinball cost uses `q` / `1-q` asymmetry.
- AC-5 (REQ-4/REQ-5): `predict` length equals `n_samples`; defaults equal
  `quantile=0.5, alpha=1.0, fit_intercept=true`; `coefficients()`/`intercept()`
  return the fitted values.
- AC-6 (REQ-6): a fitted attribute reports the solver iteration count.
- AC-7 (REQ-7): constructing with `solver="highs"` is accepted and an invalid
  solver string is rejected with the sklearn-equivalent error.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (LP-based fit) | SHIPPED | impl `fn fit in quantile_regressor.rs` builds the standard-form LP `min c·x s.t. A_eq x = y, x>=0` (decision vector `[intercept+, intercept-, coef+, coef-, u, v]`, `A_eq` row i `[1, -1, X[i,:], -X[i,:], e_i, -e_i]`, recovery `coef_ = coef+ - coef-`, `intercept_ = intercept+ - intercept-`) and solves it with the self-contained two-phase primal simplex `mod lp in quantile_regressor.rs` (Bland's anti-cycling rule), mirroring sklearn's HiGHS LP at `_quantile.py:212-307`. The intercept is a free LP variable — no centering. Non-test consumer: `ferrolearn-python/src/extras.rs` (`RsQuantileRegressor` via `QuantileRegressor::<f64>::new()`), registered in `ferrolearn-python/src/lib.rs` (`m.add_class::<extras::RsQuantileRegressor>()`). Verification: `cargo test -p ferrolearn-linear --test divergence_quantile_fit` (7 passed) — `coef_`/`intercept_` match the live HiGHS oracle within 1e-6 at `q=0.8, alpha=0.0` (`coef=[0.8105,2.1064,-0.9547]`, `intercept=0.8815`) and within 1e-4 at `q=0.2, alpha=0.0`. |
| REQ-2 (alpha·n_samples scaling) | SHIPPED | impl `fn fit in quantile_regressor.rs` sets `let alpha = alpha_param * (n_samples as f64);` and applies it as the LP's exact L1 cost on the `coef±` columns (`c[j] = alpha; c[n_params + j] = alpha;`), zeroing the intercept-slack costs (`c[0] = 0.0; c[n_params] = 0.0;`), mirroring sklearn `_quantile.py:182` (`alpha = np.sum(sample_weight) * self.alpha`) and the cost build at `_quantile.py:238-248`. Because it is the exact L1 cost (not the old IRLS-linearized ridge approximation), it reaches sklearn's sparse optimum: live oracle `alpha=1.0, quantile=0.5` sklearn `coef=[0,0,0]` (exact zeros), ferrolearn `coef=[0,0,0]` exactly (`divergence_alpha1_sparse_vertex`, tol 1e-8). Non-test consumer: `RsQuantileRegressor` in `ferrolearn-python/src/extras.rs`. |
| REQ-3 (quantile / pinball asymmetry) | SHIPPED | impl `fn fit in quantile_regressor.rs` applies the asymmetric pinball weight `let asym = if residuals[i] >= F::zero() { q } else { one - q };` mirroring sklearn's residual costs `sample_weight * self.quantile` / `sample_weight * (1 - self.quantile)` at `_quantile.py:241-243`. `quantile ∈ (0,1)` is enforced (`if self.quantile <= F::zero() || self.quantile >= F::one()` → `InvalidParameter`), mirroring `Interval(Real, 0, 1, closed="neither")` at `_quantile.py:111`. Non-test consumer: `ferrolearn-python/src/extras.rs` (`RsQuantileRegressor` via `QuantileRegressor::<f64>::new().with_quantile(quantile)`), registered in `ferrolearn-python/src/lib.rs` (`m.add_class::<extras::RsQuantileRegressor>()`). Verification: `cargo test -p ferrolearn-linear quantile` (`test_high_quantile_higher_prediction` green; 13 passed). |
| REQ-4 (predict) | SHIPPED | impl `fn predict in quantile_regressor.rs` returns `x.dot(&self.coefficients) + self.intercept`, mirroring sklearn's linear prediction (`LinearModel.predict`, `X @ coef_ + intercept_`; attributes set at `_quantile.py:302-307`). Non-test consumer: `ferrolearn-python/src/extras.rs` `RsQuantileRegressor` `predict` (via the `py_regressor!` macro over `FittedQuantileRegressor<f64>`). Verification: `cargo test -p ferrolearn-linear quantile` (`test_predict_length`, `test_predict_feature_mismatch` green). |
| REQ-5 (fit_intercept / HasCoefficients) | SHIPPED | impl: constructor `fn new in quantile_regressor.rs` sets `quantile=0.5, alpha=1.0, fit_intercept=true`, mirroring sklearn `__init__` defaults at `_quantile.py:131-133`; `with_fit_intercept` toggles centering; `impl HasCoefficients for FittedQuantileRegressor` exposes `coefficients()`/`intercept()` (the `coef_`/`intercept_` analogs of `_quantile.py:303-307`). Non-test consumer: `ferrolearn-python/src/extras.rs` `RsQuantileRegressor` (defaults `quantile=0.5, alpha=1.0, fit_intercept=true`; reads coef/intercept through `HasCoefficients`), registered in `ferrolearn-python/src/lib.rs`. Verification: `cargo test -p ferrolearn-linear quantile` (`test_defaults`, `test_no_intercept`, `test_has_coefficients` green). |
| REQ-6 (n_iter_) | NOT-STARTED | open prereq blocker #507. sklearn sets `self.n_iter_ = result.nit` (`_quantile.py:300`); `FittedQuantileRegressor in quantile_regressor.rs` stores only `coefficients`/`intercept` and discards the IRLS loop count. |
| REQ-7 (solver / solver_options) | NOT-STARTED | open prereq blocker #508. sklearn's `__init__` takes `solver="highs"`, `solver_options=None` (`_quantile.py:134-141`) constrained by `_parameter_constraints` (`_quantile.py:114-126`); `QuantileRegressor in quantile_regressor.rs` has no `solver` field (IRLS hardcoded). Only `"highs"` is contract-relevant once #340 lands. |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #509. `quantile_regressor.rs` imports `ndarray::{Array1, Array2, Axis, ScalarOperand}` and uses hand-rolled `cholesky_solve`/`gaussian_solve`, not `ferray-core` arrays or `ferray::linalg` (R-SUBSTRATE). |

## Architecture

### sklearn (the contract)

`QuantileRegressor.fit` (`_quantile.py:144-308`) builds the standard
quantile-regression linear program and solves it with
`scipy.optimize.linprog(method=solver)` (HiGHS by default):

- Penalty rescale: `alpha = np.sum(sample_weight) * self.alpha` (`:182`) — the
  `1/n · pinball + alpha·L1` objective is rewritten as `pinball + (n·alpha)·L1`.
- Decision vector `x = (s0, s, t0, t, u, v)`, all `>= 0`
  (`bounds = (0, None)` is `linprog`'s default), where
  `intercept = s0 - t0`, `coef = s - t`, `residual = y - X@coef - intercept =
  u - v` (`:218-228`).
- Cost `c = concat([full(2·n_params, alpha), sw·quantile, sw·(1-quantile)])`
  (`:238-244`); the two intercept-slack costs are zeroed so the intercept is
  unpenalized (`c[0]=0; c[n_params]=0`, `:245-248`).
- `A_eq = hstack([ones, X, -ones, -X, eye, -eye])`, `b_eq = y`
  (`:256-269`), encoding `X@(coef+ - coef-) + (intercept+ - intercept-) + u - v
  = y`.
- After solving: `params = solution[:n_params] - solution[n_params:2·n_params]`;
  `coef_ = params[1:]`, `intercept_ = params[0]`; `n_iter_ = result.nit`
  (`:296-307`). The intercept is therefore a **free LP variable**, not derived
  from data means — sklearn explicitly notes "centering y and X with
  `_preprocess_data` does not work for quantile regression" (`_quantile.py:177`).

### ferrolearn (what exists)

`QuantileRegressor<F> { quantile, alpha, max_iter, tol, fit_intercept }`
(`quantile_regressor.rs`) implements `Fit` via IRLS on a smoothed pinball loss:

- `fn fit` validates shapes / `quantile ∈ (0,1)` / `alpha >= 0`, then **centers**
  `X` and `y` by their column/scalar means when `fit_intercept` (the step
  sklearn warns against), warm-starts `w` from an OLS solve, and iterates: per
  sample `weight_i = asym_i / (2·max(|r_i|, eps))` with `asym_i = q` (positive
  residual) or `1-q` (negative), then re-solves a weighted normal-equation
  system via `weighted_l1_solve` until `max |Δw| < tol` or `max_iter`.
- `fn weighted_l1_solve` forms `XᵀWX + diag(scaled_alpha / max(|w_prev_j|, eps))`
  and `XᵀWy` and solves via `cholesky_solve` (falling back to `gaussian_solve`).
  `scaled_alpha = alpha · n_samples` (the correct factor; REQ-2 magnitude), but
  the `1/|w_j|` linearization is a reweighted-ridge approximation of L1, not the
  exact L1 of the LP.
- `intercept = y_mean - x_mean·w` (`fn fit`) — a centering-recovered intercept.
  This is invariant to `quantile`, so it diverges from sklearn whenever
  `quantile ≠ 0.5` (blocker #506).
- `FittedQuantileRegressor<F> { coefficients, intercept }` implements `Predict`
  (`X@coef + intercept`) and `HasCoefficients`; both `QuantileRegressor` and
  `FittedQuantileRegressor` implement the pipeline-estimator traits.

### Why the IRLS path diverges from the contract (issue #340)

IRLS on the smoothed pinball loss is a stationary-point iteration on a smoothed
surrogate; the true objective is piecewise-linear and its optimum is an LP
vertex. The two coincide only in benign regimes. Measured against the live
oracle (baseline `fb384c5`):

| case | sklearn `coef_` | ferrolearn `coef_` | sklearn `intercept_` | ferrolearn `intercept_` |
|---|---|---|---|---|
| q=0.5, α=0.0 | `[0.8992, 2.0000, -0.8874]` | `[0.8845, 1.9848, -0.8941]` | `0.3243` | `0.2989` |
| q=0.8, α=0.0 | `[0.8105, 2.1064, -0.9547]` | `[0.8845, 1.9849, -0.8941]` | `0.8815` | `0.2988` |
| q=0.5, α=1.0 | `[0, 0, 0]` | `[1.7e-9, 2.2e-6, -5.5e-10]` | `0.7387` | `0.9120` |
| q=0.8, α=1.0 | `[0, 0, 0]` | `[1.7e-9, 4.8e-6, -5.5e-10]` | `2.7406` | `0.9120` |

Two structural failures are visible: (1) ferrolearn's `coef_`/`intercept_` are
**identical across `quantile`** at fixed `alpha` — the centering intercept and
the symmetric IRLS reweighting wash out the quantile asymmetry's effect on the
solution (blocker #506); and (2) under `alpha=1.0` sklearn returns the exact
sparse vertex `[0,0,0]` with a quantile-dependent intercept (`0.7387` vs
`2.7406`), while ferrolearn's near-zero coefficients leave an identical `0.9120`
intercept for both quantiles. The original #340 fixture (`alpha=0.01`,
`make_regression`) reports a 25x prediction gap; the regression test for REQ-1
must pin against the live LP oracle, not the IRLS output.

## Verification

Commands that establish the SHIPPED claims (run at baseline `fb384c5`):

- `cargo test -p ferrolearn-linear quantile` — 13 quantile unit tests pass
  (`test_defaults`, `test_builder`, `test_invalid_quantile_zero/one`,
  `test_negative_alpha`, `test_predict_length`, `test_predict_feature_mismatch`,
  `test_has_coefficients`, `test_no_intercept`, `test_high_quantile_higher_prediction`,
  `test_median_regression_clean_data`, `test_pipeline`). Pins REQ-3/REQ-4/REQ-5.
- Live sklearn oracle (establishes the NOT-STARTED REQ-1/REQ-2 gaps):
  `python3 -c "from sklearn.linear_model import QuantileRegressor; import numpy as np; rng=np.random.RandomState(0); X=rng.randn(30,3); y=X@np.array([1.,2.,-1.])+0.5*rng.randn(30); m=QuantileRegressor(quantile=0.5, alpha=0.0).fit(X,y); print(m.coef_.tolist(), m.intercept_)"`
  → `[0.8992, 2.0000, -0.8874] 0.3243`; the IRLS fit at `quantile=0.8` returns
  the same intercept as `quantile=0.5`, and at `alpha=1.0` does not reach the
  exact `[0,0,0]` sparse optimum.

REQ-1, REQ-2, REQ-6, REQ-7, REQ-8 have no green verification against the sklearn
contract and are NOT-STARTED, gated on blockers #340, #332, #506, #507, #508,
#509 respectively. REQ-1 closes only when a quantile-regression LP solver lands
and a regression test (expected values from the live HiGHS oracle, never copied
from the IRLS side, per R-CHAR-3) matches `coef_`/`intercept_` within tolerance
for `quantile ∈ {0.2, 0.5, 0.8}` and `alpha ∈ {0, 1}`.
