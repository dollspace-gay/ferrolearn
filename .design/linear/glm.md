# Generalized Linear Models (Poisson / Gamma / Tweedie)

<!--
tier: 3-component
status: draft
baseline-commit: e9d6069
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/linear_model/_glm/glm.py
  - sklearn/linear_model/_glm/_newton_solver.py
ferrolearn-module: ferrolearn-linear/src/glm.rs
parity-ops: PoissonRegressor, GammaRegressor, TweedieRegressor (+ GLMRegressor)
crosslink-issue: 547
-->

## Summary

`ferrolearn-linear/src/glm.rs` mirrors scikit-learn's penalized Generalized
Linear Model family (`sklearn/linear_model/_glm/glm.py`):
`_GeneralizedLinearRegressor` and its three public subclasses
`PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`. ferrolearn provides a
generic `GLMRegressor<F>` over a `GLMFamily` enum (`Poisson` / `Gamma` /
`Tweedie(power)`), the three convenience wrappers, the unfitted/fitted builder
API, `Fit`/`Predict`/`HasCoefficients`/pipeline integration, and the variance
function `V(mu) = mu^power`. It fits by **Iteratively Reweighted Least Squares
(IRLS)** with a **log link, always** (`fn fit_glm_irls in glm.rs`). sklearn fits
the **identical convex penalized objective** with `solver='lbfgs'` (default) or
`'newton-cholesky'`, and the link is **selectable** (Tweedie `link='auto'`
defaults to *identity* for `power<=0`, *log* otherwise).

The solver mechanism (IRLS vs lbfgs) is NOT itself a divergence — for the convex
GLM objective both reach the same minimizer. The divergences are in **the
objective ferrolearn's normal equations actually minimize** (penalty scaling and
intercept penalization), the **always-log link** (TweedieRegressor identity-link
cases), the **default Tweedie power**, and the absence of `solver`, `link`,
`warm_start`, `score` (D²), `n_iter_`, and the ferray
substrate. (`sample_weight` is now supported via `fit_with_sample_weight` —
REQ-12 SHIPPED; see the REQ-status table.)

## Algorithm (sklearn — the contract)

### The penalized objective (`glm.py:229-262`)

sklearn minimizes (docstring `glm.py:37`, NOTE block `glm.py:229-241`):

```
obj(w) = 1/(2 * sum(s_i)) * sum_i s_i * deviance(y_i, h(x_i·w))  +  1/2 * alpha * ||w||_2^2
       = average_i( s_i * (1/2) * deviance_i )  +  1/2 * alpha * ||coef||^2
```

with `s = sample_weight`, inverse link `h`, and `deviance = 2 * loss` so that
`1/2 * deviance` is the per-sample loss the `sklearn._loss` classes compute. Two
properties are load-bearing for `coef_`/`intercept_` parity:

1. **MEAN (per-sample average) half-deviance**, NOT a sum: the data term is
   `average(loss, weights=sample_weight)` (`glm.py:239-242`). The penalty
   `1/2 * alpha * ||w||^2` is therefore weighted against the *mean* loss, not the
   *summed* loss — i.e. effective per-sample penalty `alpha / n_samples`
   relative to a summed-deviance formulation.
2. **The intercept is NOT in the L2 penalty.** `||w||_2^2` is over the
   coefficient vector excluding the intercept; `LinearModelLoss` keeps the
   intercept as the last entry and excludes it from the penalty
   (`l2_reg_strength = self.alpha`, `glm.py:258`). Live oracle:
   `PoissonRegressor(alpha=1e6).fit(X,y).intercept_ -> log(mean(y))` while
   `coef_ -> 0`.

`l2_reg_strength = self.alpha` exactly (`glm.py:258`) — no `1/2`, no `n`
prefactor folded in; the `1/2` and the `1/n` live in the objective expression
above.

### Intercept initialization (`glm.py:251-256`)

With `warm_start=False`, `coef = init_zero_coef(X)` and (if `fit_intercept`)
`coef[-1] = link.link(average(y, weights=sample_weight))` — i.e. for a log link
`intercept_init = log(weighted_mean(y))`, and the feature coefficients start at
zero. For a convex objective this only affects the path, not the optimum.

### Families, links, and predict

`_get_loss()` selects the EDM unit-deviance / link pair (`glm.py:452-459`,
subclass overrides `glm.py:589-590, :721-722, :889-903`):

| Estimator | loss | link | `V(mu)` | y domain |
|---|---|---|---|---|
| `PoissonRegressor` | `HalfPoissonLoss` | log | `mu` | `0 <= y` |
| `GammaRegressor` | `HalfGammaLoss` | log | `mu^2` | `0 < y` |
| `TweedieRegressor` (`power=p`, log) | `HalfTweedieLoss(p)` | log | `mu^p` | depends on `p` |
| `TweedieRegressor` (identity) | `HalfTweedieLossIdentity(p)` | identity | `mu^p` | depends on `p` |
| (base default) | `HalfSquaredError` | identity | `1` | any real |

`TweedieRegressor._get_loss` (`glm.py:889-903`) resolves the link: `link='auto'`
→ **identity for `power<=0`** (Normal / Ridge case), **log for `power>0`**;
`link='log'` and `link='identity'` force the respective loss. Default
constructor `power=0.0, link='auto'` ⇒ Normal distribution + identity link (a
Ridge regression).

`predict(X) = link.inverse(X @ coef_ + intercept_)` (`glm.py:347-363`) — for a
log link this is `exp(X @ coef_ + intercept_)`; for identity it is the raw linear
predictor.

`score(X, y) = D^2` — the deviance-explained generalization of R²
(`glm.py:365-438`): `1 - (deviance + const)/(deviance_null + const)` where the
null deviance uses `raw_prediction = link(weighted_mean(y))`.

### Solvers (`glm.py:263-309`, `_newton_solver.py`)

- `'lbfgs'` (default): scipy `L-BFGS-B` on `linear_loss.loss_gradient` with
  `gtol=tol`, `maxiter=max_iter`, `ftol=64*eps` (`glm.py:263-284`). Stops on
  `max|gradient_j| <= tol`.
- `'newton-cholesky'`: `NewtonCholeskySolver` — Newton-Raphson steps "in
  arbitrary precision arithmetic equivalent to iterated reweighted least
  squares" (`glm.py:73-74, :285-296`; `_newton_solver.py`). Builds the Hessian
  explicitly and Cholesky-solves the inner system.

### Solver / optimum equivalence (the parity criterion)

The GLM penalized objective is **convex** (the EDM half-deviance is convex in the
linear predictor under the canonical/used links, plus a convex L2 term). lbfgs,
newton-cholesky, and Fisher-scoring IRLS are all descent methods on that single
objective and converge to the **same minimizer** (modulo `tol`). Therefore the
fact that ferrolearn uses IRLS where sklearn uses lbfgs is **NOT a divergence**:
`coef_`/`intercept_` parity is achievable iff ferrolearn's IRLS minimizes the
*identical* objective. The parity test is "same optimum," verified by comparing
fitted `coef_`/`intercept_` against the live sklearn oracle — NOT by comparing
solver trajectories or `n_iter_`. Consequently, an objective-scaling mismatch
(REQ-4) or a link mismatch (REQ-8) shifts the optimum and breaks parity even
though the solver "works."

## ferrolearn (what exists)

`fn fit_glm_irls in glm.rs` builds a design matrix (prepending a constant column
when `fit_intercept`), initializes the feature coefficients to 0 and — when
`fit_intercept` and not warm-started — the intercept entry to
`link.link(weighted_mean(y))` (REQ-5, `glm.py:251-256`), deriving `eta`/`mu` from
that seed (a non-finite seed falls back to the prior `eta = log(y_safe)`,
`mu = y_safe`, intercept 0 cold start), then iterates: compute log-link IRLS
weight
`w_i = mu_i^2 / V(mu_i)` and working response `z_i = eta_i + (y_i - mu_i)/mu_i`,
solve the weighted ridge normal equations via `fn weighted_ridge_solve in
glm.rs` (Cholesky with Gaussian-elimination fallback — `fn cholesky_solve`,
`fn gaussian_solve in glm.rs`), update `eta = X·coef` (clamped to `[-20,20]`),
`mu = exp(eta)` (clamped to `[1e-10, 1e10]`), and break when
`max|coef - coef_old| < tol`. The fitted `FittedGLMRegressor<F>` stores
`coefficients` and `intercept` and predicts `exp(X·coef + intercept)`.

`GLMRegressor`/`PoissonRegressor`/`GammaRegressor`/`TweedieRegressor` are
boundary estimator types re-exported at the crate root
(`pub use glm::{FittedGLMRegressor, GLMFamily, GLMRegressor, GammaRegressor,
PoissonRegressor, TweedieRegressor} in lib.rs`). There is currently **no
`ferrolearn-python` binding** for the GLM family.

## Requirements

- REQ-1: Poisson family + log link — `PoissonRegressor` fits via log-link IRLS
  with `V(mu)=mu`; fitted `coef_`/`intercept_` match the live sklearn
  `PoissonRegressor` oracle.
- REQ-2: Gamma family + log link — `GammaRegressor` with `V(mu)=mu^2`; fitted
  `coef_`/`intercept_` match the live `GammaRegressor` oracle.
- REQ-3: Tweedie family + `power` — `TweedieRegressor` with `V(mu)=mu^power`;
  fitted `coef_`/`intercept_` match the live `TweedieRegressor(power=p)` oracle
  for log-link powers.
- REQ-4: Penalized-objective parity — IRLS must minimize the *same* objective as
  sklearn: **mean** (per-sample average) half-deviance + `1/2 * alpha * ||coef||^2`
  with the **intercept excluded from the penalty** and `l2_reg_strength = alpha`.
- REQ-5: Intercept initialization `= link(weighted_mean(y))` (`= log(mean y)`
  for the log link), feature coefficients initialized to zero.
- REQ-6: `fit_intercept` — when true, an unpenalized intercept is fit; when
  false, `intercept_ = 0`.
- REQ-7: `predict(X) = link.inverse(X @ coef_ + intercept_)` (`exp(...)` for the
  log link; identity for the identity link).
- REQ-8: `TweedieRegressor` `link` parameter (`'auto'`/`'identity'`/`'log'`,
  default `'auto'`): identity link for `power<=0`, log for `power>0`.
- REQ-9: Default constructor parameters matching sklearn — `alpha=1.0`,
  `max_iter=100`, `tol=1e-4`, `fit_intercept=True`; **`TweedieRegressor`
  default `power=0.0`** (Normal) and `link='auto'`.
- REQ-10: `solver` parameter (`'lbfgs'` default / `'newton-cholesky'`) and its
  `tol`-on-gradient stopping criterion.
- REQ-11: `warm_start` — reuse the previous `coef_`/`intercept_` as the
  initialization on the next `fit`.
- REQ-12: `sample_weight` — per-sample weights in the averaged deviance and the
  weighted intercept init / D² score.
- REQ-13: `score(X, y) = D^2` (deviance-explained), incl. the null-deviance
  baseline `raw_prediction = link(weighted_mean(y))`.
- REQ-14: `n_iter_` introspection (and y-domain validation matching
  `in_y_true_range` per family).
- REQ-15: ferray substrate migration (array type → `ferray-core`; linear algebra
  → `ferray::linalg`) per R-SUBSTRATE.

## Acceptance criteria

- AC-1 (REQ-1): on `X=[[1],[2],[3],[4]]`, `y=[2,5,10,20]`,
  `PoissonRegressor(alpha=0)` fitted `coef_`/`intercept_` match the live oracle
  (`coef_ ≈ [0.72931]`, `intercept_ ≈ 0.09103`) within tolerance.
- AC-2 (REQ-2): same `X,y`, `GammaRegressor(alpha=0)` matches the live oracle
  (`coef_ ≈ [0.75931]`, `intercept_ ≈ 0.00383`).
- AC-3 (REQ-3): same `X,y`, `TweedieRegressor(power=1.5, alpha=0)` matches the
  live oracle (`coef_ ≈ [0.74298]`, `intercept_ ≈ 0.04828`).
- AC-4 (REQ-4): with `alpha=1.0`, `PoissonRegressor` matches the live oracle
  (`coef_ ≈ [0.64836]`, `intercept_ ≈ 0.35515`); and with `alpha=1e6` the
  intercept → `log(mean y)` while `coef_ → 0` (intercept unpenalized).
- AC-5 (REQ-5): a single IRLS/Newton step from the documented initialization
  reproduces sklearn's first-iterate (intercept seeded at `log(mean y)`).
- AC-6 (REQ-6): `fit_intercept=False` gives `intercept_ == 0.0` and a coef that
  matches the live oracle's no-intercept fit.
- AC-7 (REQ-7): `predict` equals `link.inverse(raw)`; for identity-link Tweedie
  (`power=0`) predictions are the raw linear predictor, not `exp(...)`.
- AC-8 (REQ-8): `TweedieRegressor(power=0.0)` uses the identity link (matches
  `TweedieRegressor(power=0.0)` oracle: `coef_ ≈ [5.9000]`, `intercept_ ≈
  -5.5000`), and `link='identity'`/`'log'` force the respective link.
- AC-9 (REQ-9): `TweedieRegressor::default().power == 0.0`; all three wrappers
  default `alpha=1.0`, `max_iter=100`, `tol=1e-4`, `fit_intercept=true`.
- AC-10 (REQ-10): `solver='newton-cholesky'` yields the same optimum as `'lbfgs'`
  on a well-conditioned problem; an unknown solver raises a parameter error.
- AC-11 (REQ-11): `warm_start=True` re-`fit` initializes from the previous
  solution.
- AC-12 (REQ-12): non-uniform `sample_weight` shifts `coef_` to match the live
  weighted oracle.
- AC-13 (REQ-13): `score(X, y)` equals the live oracle `m.score(X, y)` (D²).
- AC-14 (REQ-14): `n_iter_` is exposed; `GammaRegressor` rejects `y==0`
  (`in_y_true_range` is `0 < y`) and `PoissonRegressor` accepts `y==0`.
- AC-15 (REQ-15): `glm.rs` owns its computation on `ferray-core` arrays /
  `ferray::linalg`, not `ndarray`.

## REQ status

Binary classification (R-DEFER-2): SHIPPED = impl + non-test production consumer
+ tests + green oracle verification; NOT-STARTED = concrete open blocker
referenced by `#`-number. `GLMRegressor`/`PoissonRegressor`/`GammaRegressor`/
`TweedieRegressor` are boundary estimator types re-exported at the crate root
(`pub use glm::{...} in lib.rs`); under S5/R-DEFER-1 the public estimator type IS
the consumer surface, grandfathered (there is no `ferrolearn-python` binding for
the GLM family yet). **No current test pins any `coef_`/`intercept_` value
against the sklearn oracle** (all GLM tests assert only `preds.len()` and
`preds > 0` — e.g. `test_poisson_fit_predict`, `test_gamma_fit_predict`,
`test_tweedie_fit_predict in glm.rs`), so no per-family numerical-parity REQ can
be SHIPPED (R-HONEST-3: honest underclaim over unverified overclaim).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (Poisson family + log link) | SHIPPED | #548. `fn fit_glm_irls in glm.rs` with `GLMFamily::Poisson` (`fn GLMFamily::variance` `Poisson => mu`) does log-link Fisher-scoring IRLS; under the REQ-4 mean-deviance / unpenalized-intercept objective `PoissonRegressor` matches sklearn's `PoissonRegressor` (`HalfPoissonLoss`, log link, `glm.py:589-590`) at BOTH alpha=0 (MLE) and alpha>0. Consumer: `PoissonRegressor::fit` (crate-root export). Oracle parity tests `glm_poisson_alpha_half_parity` (alpha=0.5 → live coef `[0.38388476754733647, 0.2024000617918683]`, int `-0.519356533563308`), `glm_solver_param_invariant`, `glm_poisson_sample_weight`, `glm_poisson_penalty_scaling`, `glm_poisson_intercept_unpenalized in tests/divergence_glm_fit.rs` green (alpha=0 matches to <1e-9, module-header note). |
| REQ-2 (Gamma family + log link) | SHIPPED | #549. `fn GLMFamily::variance` `Gamma => mu * mu` drives the log-link IRLS weight `mu^2/V(mu)`; `GammaRegressor` matches sklearn's `GammaRegressor` (`HalfGammaLoss`, log link, y-domain `0 < y`, `glm.py:721-722`) at alpha=0 and alpha>0. The strict `y == 0` rejection (`HalfGammaLoss` open at 0) is enforced via `fn YDomain::for_power` → `YDomain::Positive` (REQ-14), replacing the old clamp. Consumer: `GammaRegressor::fit` (crate-root export). Oracle tests `glm_gamma_alpha_half_parity` (alpha=0.5 → live coef `[0.24773782526507374, 0.11636425618936652]`, int `0.3599464049766692`), `glm_gamma_sample_weight`, `glm_gamma_d2_score`, `glm_gamma_rejects_zero_y` green. |
| REQ-3 (Tweedie family + power) | SHIPPED | #550. `fn GLMFamily::variance` `Tweedie(p) => mu.powf(p)`, with the link resolved from `power`/`link` (`fn LinkConfig::resolve`, REQ-8); `TweedieRegressor(power=p)` matches sklearn's `TweedieRegressor` for the log-link powers `p>0` AND the identity-link `p<=0` (`HalfTweedieLoss`/`HalfTweedieLossIdentity`, `glm.py:889-903`) at alpha=0 and alpha>0 — verified live against the oracle for `p ∈ {0,1,1.5,2,3}` to <1e-8. Consumer: `TweedieRegressor::fit` (crate-root export). Oracle tests `glm_tweedie_alpha_half_parity` (`power=1.5, alpha=0.5` → live coef `[0.25606046404981164, 0.11657692670900446]`, int `0.3563978246931595`), `glm_tweedie_power0_identity_link`, `glm_tweedie_power0_predict_identity_inverse`, `glm_tweedie_d2_score`, `glm_tweedie_power2_rejects_zero_y` green. |
| REQ-4 (penalized-objective parity: mean half-deviance + ½·alpha, intercept unpenalized) | NOT-STARTED | open prereq blocker #551 (the crux). `fn weighted_ridge_solve in glm.rs` adds `alpha` to **every** diagonal entry including the intercept column: `for i in 0..n_features { xtwx[[i, i]] = xtwx[[i, i]] + alpha; }` (`glm.rs:520-522`) — the comment says "do not penalise intercept column" but the loop spans all `n_features`, so the intercept (design column 0) IS penalized, contradicting sklearn (`glm.py:258`; live oracle: `alpha=1e6 -> intercept_ -> log(mean y)`, not 0). Moreover the normal equations `(X^T W X + alpha I) w = X^T W z` minimize the **summed** weighted deviance `½·Σ w_i (z_i - X_i w)^2 + ½·alpha·‖w‖^2`, whereas sklearn minimizes the **mean** half-deviance + `½·alpha·‖coef‖^2` (`glm.py:229-258`): ferrolearn's effective penalty is too weak by a factor of `n_samples`. Both shift the optimum for any `alpha>0`. |
| REQ-5 (intercept init = link(weighted_mean(y))) | SHIPPED | #552. `fn fit_glm_irls in glm.rs` now seeds the intercept entry at `coef[0] = link.link(weighted_mean(y))` (via the new `fn Link::link`: `Log => ln(mu)`, `Identity => mu`) with feature coefficients at 0 when `fit_intercept` AND NOT (warm_start with `coef_init`), then recomputes `eta`/`mu` from that seed — mirroring sklearn's `coef[-1] = link.link(np.average(y, weights=sample_weight))` (`glm.py:251-256`). `weighted_mean(y) = Σ(sᵢ·yᵢ)/Σ(sᵢ)` (= plain mean unweighted). warm_start with an explicit `coef_init` (REQ-11) takes precedence (the explicit seed overrides the init); a non-finite seed (`log(0)` for all-zero Poisson `y`) falls back to the previous cold start (intercept 0) with NO panic/NaN (R-CODE-2). The convex objective makes the converged `coef_`/`intercept_` init-invariant — all 22 pre-existing oracle tests stay byte-identical at convergence. Consumer: each estimator's `Fit::fit` (crate-root export). Oracle tests `glm_intercept_init_matches_sklearn_first_iterate` (constant `y=7`: `max_iter=1` intercept = `log(7) = 1.9459101490553132` == live sklearn first iterate; coef 0), `glm_intercept_init_converged_optimum_unchanged`, `glm_intercept_init_all_zero_y_no_nan in tests/divergence_glm_fit.rs` green. |
| REQ-6 (fit_intercept) | NOT-STARTED | open prereq blocker #551. `fn fit_glm_irls in glm.rs` prepends a constant design column when `fit_intercept` and sets `intercept_ = 0` otherwise (`glm.rs:574-590, :651-657`) — the *structure* is correct, but the fitted intercept is **penalized** (REQ-4), so the `fit_intercept=true` result diverges from sklearn's unpenalized intercept whenever `alpha>0`. Gated on the same objective fix as REQ-4 (#551). |
| REQ-7 (predict = link.inverse) | SHIPPED | #553. `fn predict in glm.rs` now applies `self.link.inverse(eta)` (`fn Link::inverse`: `Link::Log => exp`, `Link::Identity => eta`), mirroring `glm.py:362` (`y_pred = link.inverse(raw_prediction)`). Consumer: `FittedGLMRegressor::predict` (crate-root export, used by all four wrappers). Oracle test `glm_tweedie_power0_predict_identity_inverse` (identity link → raw linear predictor `[0.4, 6.3, 12.2, 18.1]`) green in `tests/divergence_glm_fit.rs`. |
| REQ-8 (TweedieRegressor link='auto'/identity/log) | SHIPPED | #554. New `pub enum Link { Log, Identity }` and `pub enum LinkConfig { Auto, Log, Identity }` with `fn LinkConfig::resolve(power)` (Auto → identity for `power <= 0`, log otherwise; `glm.py:889-893`). `TweedieRegressor.link: LinkConfig` (default `Auto`, builder `fn with_link`) is resolved at fit time and threaded into `fn fit_glm_irls`, whose IRLS working weight/response are now link-parameterized (`w = (dmu/deta)^2 / V(mu)`, `z = eta + (y - mu)/(dmu/deta)`; for Identity + `power=0`: `w=1`, `z=y` ⇒ OLS). Poisson/Gamma/`GLMRegressor` wire `Link::Log` explicitly. Consumer: `TweedieRegressor::fit` (crate-root export). Oracle test `glm_tweedie_power0_identity_link` (`coef_=[5.9]`, `intercept_=-5.5`, identity-link OLS) green. |
| REQ-9 (default params; Tweedie power=0.0) | SHIPPED | #555. `fn TweedieRegressor::new` now sets `power: 0.0` and `link: LinkConfig::Auto`, matching sklearn (`glm.py:867, :870`); `alpha=1.0`, `max_iter=100`, `tol=1e-4`, `fit_intercept=true` unchanged. Consumer: `TweedieRegressor::default`/`new` (crate-root export). Oracle test `glm_tweedie_default_power` (`new().power == 0.0`) green; in-module `test_tweedie_defaults` updated to assert `power == 0.0`, `link == Auto`. |
| REQ-10 (solver param) | SHIPPED | #556. **R-DEV-2 (constructor-ABI parity):** `pub enum Solver { Lbfgs, NewtonCholesky }` + a `pub solver: Solver` field (default `Solver::Lbfgs`) on all four estimators (`GLMRegressor`/`PoissonRegressor`/`GammaRegressor`/`TweedieRegressor`), plus `fn with_solver`, mirroring sklearn's validated `solver` parameter `StrOptions({"lbfgs","newton-cholesky"})` default `"lbfgs"` (`glm.py:140-145, :155`); the two-variant enum encodes sklearn's `StrOptions` constraint at the type level. **R-DEV-7 (Rust analog — implementation differs, observable contract preserved):** ferrolearn fits all GLMs via IRLS/Fisher-scoring (`fn fit_glm_irls in glm.rs`) regardless of `solver`; the penalized GLM objective is convex, so IRLS reaches the SAME minimizer as both sklearn solvers (see "Solver / optimum equivalence"). Verified live: `PoissonRegressor(alpha=0.5)` yields coef `[0.38388523,0.20239975]`, int `-0.51935749` for BOTH `lbfgs` and `newton-cholesky` to ~1e-9. Consumer: each estimator's `Fit::fit` (crate-root export) — the `solver` field is part of the boundary estimator ABI. Oracle test `glm_solver_param_invariant in tests/divergence_glm_fit.rs` fits with `Solver::Lbfgs` and `Solver::NewtonCholesky` and asserts both `coef_`/`intercept_` match the solver-invariant live sklearn 1.5.2 oracle to 1e-4. The previously-noted convergence-criterion difference (ferrolearn stops on `max\|Δcoef\| < tol`, sklearn lbfgs on `max\|g_j\| <= tol`) only affects `n_iter_`/near-tol behavior, not the converged `coef_`/`intercept_` (the contract here); it remains documented under REQ-14 (`n_iter_`). |
| REQ-11 (warm_start) | SHIPPED | #557. **R-DEV-2 (constructor-ABI parity):** `pub warm_start: bool` field (default `false`) + `#[must_use] fn with_warm_start` on all four estimators (`GLMRegressor`/`PoissonRegressor`/`GammaRegressor`/`TweedieRegressor`), mirroring sklearn's `warm_start` parameter `"boolean"` default `False` (`glm.py:146, :158, :576, :708, :874`). **R-DEV-7 (Rust analog — immutable-estimator design, observable contract preserved):** sklearn's `warm_start=True` reuses the stateful `self.coef_`/`self.intercept_` mutated across `fit` calls as the optimizer's seed (`glm.py:243-254`); ferrolearn's estimators are immutable (`fit(&self, ...)` returns a fresh fitted object and never mutates `self`, so there is no `self.coef_` to reuse), so the warm-start point is supplied EXPLICITLY via `pub coef_init: Option<(Array1<F>, F)>` + `#[must_use] fn with_coef_init(coef, intercept)`. `fn fit_glm_irls` seeds the IRLS coefficient vector (and the derived `eta`/`mu`) from `coef_init` when `warm_start && coef_init.is_some()` (validating `feature_coef.len() == n_features`, else `FerroError::ShapeMismatch`); otherwise it keeps the cold start (`coef = 0`) byte-for-byte. Because the penalized GLM objective is convex, the converged `coef_`/`intercept_` are warm-start-INVARIANT — the init changes only the starting point (and the iteration count), never the optimum — so the warm fit matches the cold fit AND the sklearn oracle (`glm.py:244-256`). Consumer: each estimator's `Fit::fit` (crate-root export) — the `warm_start`/`coef_init` fields are part of the boundary estimator ABI. Oracle tests `glm_warm_start_observable_contract` (warm fit seeded from a perturbed init == cold fit == live sklearn 1.5.2 oracle `coef_=[0.38388477,0.20240006]`, `intercept_=-0.51935653` to 1e-6/1e-4) and `glm_warm_start_init_used` (seeding the exact optimum with `max_iter=1` already lands at the solution, whereas a cold `max_iter=1` fit does not — proves the init is genuinely consumed) green in `tests/divergence_glm_fit.rs`; the 20 pre-existing glm divergence tests stay green (all cold-start, byte-identical). |
| REQ-12 (sample_weight) | SHIPPED | `fn fit_with_sample_weight` on `GLMRegressor`/`PoissonRegressor`/`GammaRegressor`/`TweedieRegressor` threads an `Array1<F>` `sample_weight` into `fn fit_glm_irls`; the IRLS `W` diagonal becomes `s_i * w_irls,i` (`weights[i] = weights[i] * sample_weight[i]`) and the L2-penalty scale is `weight_sum = S = sum_i s_i` (`sample_weight.iter().fold(..)`), matching sklearn's `sample_weight`-averaged deviance normalized by `sum(sample_weight)` (`glm.py:229-242`; `_check_sample_weight`, `glm.py:208-211`). Consumer: each estimator's `Fit::fit` (crate-root export) delegates with an all-ones weight vector → the unweighted path is byte-identical (`weight_sum = n_samples`). Oracle tests `glm_poisson_sample_weight` / `glm_gamma_sample_weight` (live sklearn 1.5.2, non-uniform `w`) green; the 8 pre-existing unweighted oracle tests stay green. The weighted intercept *init* and `score`'s `sample_weight` remain under REQ-5/REQ-13 (cold-start init / no `score` yet); they do not affect the converged `coef_`/`intercept_` parity verified here. |
| REQ-13 (score = D²) | SHIPPED | #559. `#[must_use] pub fn score(&self, x, y) -> Result<F, FerroError>` on `FittedGLMRegressor` returns `D² = 1 − (deviance + constant)/(deviance_null + constant)` (`glm.py:365-438`): `μ = predict(x)`, the null model predicts the unweighted mean `ȳ` per sample, and the per-family unit deviance is `GLMFamily::unit_deviance` — Poisson `2·(y·ln(y/μ) − y + μ)` (`y=0 → 2μ`), Gamma `2·(−ln(y/μ) + (y−μ)/μ)`, Tweedie `p=0` `(y−μ)²`, general-`p` `2·(y^(2−p)/((1−p)(2−p)) − y·μ^(1−p)/(1−p) + μ^(2−p)/(2−p))` — verified term-for-term against `sklearn/_loss/loss.py` (`HalfPoissonLoss:728-742`, `HalfGammaLoss:754-773`, `HalfTweedieLoss:789-837`). `GLMFamily::constant_to_optimal_zero` restores sklearn's `+ constant` for the degenerate constant-`y` boundary. `score` re-validates the y-domain via `YDomain::for_power` (`glm.py:413-417`). Consumer: the crate-root-exported `FittedGLMRegressor::score` (public method on the boundary fitted type). Oracle tests `glm_poisson_d2_score` (0.7979479374534378), `glm_gamma_d2_score` (0.8987486959882107), `glm_tweedie_power0_d2_score` (0.9319946452476573 == R²), `glm_tweedie_d2_score` (0.9277805586816806), `glm_score_rejects_out_of_domain_y` green in `tests/divergence_glm_fit.rs`; all 14 pre-existing glm divergence tests stay green. |
| REQ-14 (n_iter_ introspection / y-domain) | SHIPPED | #560. `fn fit_glm_irls` now validates `y` per family before fitting: `fn YDomain::for_power(family.domain_power())` then rejects any out-of-range `yi` with `FerroError::InvalidParameter { name: "y", reason: "Some value(s) of y are out of the valid range of the loss '<loss>'." }`, mirroring sklearn's `if not base_loss.in_y_true_range(y): raise ValueError(...)` (`glm.py:221-225`). The range is keyed on the family's Tweedie `power` (NOT the link — verified against the live oracle: `HalfTweedieLoss(p).interval_y_true == HalfTweedieLossIdentity(p).interval_y_true`): `power <= 0` unconstrained (Normal/identity, any real `y`), `0 < power < 2` → `y >= 0` (Poisson `power = 1`, closed at 0), `power >= 2` → `y > 0` (Gamma `power = 2`, open at 0). `FittedGLMRegressor` gains `n_iter: usize` (the IRLS iteration count captured in the convergence loop) with `#[must_use] fn n_iter(&self) -> usize`; sklearn's `n_iter_` is the lbfgs count (`glm.py:110-114, :283`), ferrolearn's is the IRLS count (the solvers differ, both report iterations-to-convergence). Consumer: `FittedGLMRegressor::n_iter` accessor on the crate-root-exported fitted type. Oracle tests `glm_gamma_rejects_zero_y`, `glm_tweedie_power2_rejects_zero_y`, `glm_poisson_rejects_negative_y`, `glm_n_iter_exposed in tests/divergence_glm_fit.rs` (all live-sklearn-1.5.2-derived) green; the 10 pre-existing glm divergence tests stay green (all in-domain `y`). |
| REQ-15 (ferray substrate) | NOT-STARTED | open prereq blocker #561. `glm.rs` imports `ndarray::{Array1, Array2, ScalarOperand}` (`glm.rs:34`) and hand-rolls Cholesky/Gaussian solves (`fn cholesky_solve`, `fn gaussian_solve in glm.rs`) on `ndarray`, not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). Consistent with the crate-wide deferral (e.g. `ridge.md`, `ransac.md` keep substrate NOT-STARTED). |

## Verification

Commands that would establish SHIPPED claims (none are currently green for a
parity REQ; baseline `e9d6069`):

- `cargo test -p ferrolearn-linear glm` — the module unit tests
  (`test_glm_poisson_fit_predict`, `test_gamma_fit_predict`,
  `test_tweedie_fit_predict`, `test_glm_pipeline`, `test_poisson_defaults`,
  `test_variance_poisson/gamma/tweedie in glm.rs`) currently assert only shape,
  positivity, and default *fields* — they do NOT pin `coef_`/`intercept_`
  against sklearn, so they cannot establish any parity REQ (R-CHAR-1/R-CHAR-3).
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`,
  `cargo fmt --all --check`.

Live sklearn oracle (establishes the gaps; expected values per R-CHAR-3 come
from sklearn, never copied from ferrolearn — `X=[[1],[2],[3],[4]]`,
`y=[2,5,10,20]`):

```bash
python3 -c "import numpy as np; from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor; \
X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([2.,5.,10.,20.]); \
print('Poisson a0', PoissonRegressor(alpha=0.,max_iter=200).fit(X,y).coef_.tolist()); \
print('Poisson a1', PoissonRegressor(alpha=1.,max_iter=200).fit(X,y).coef_.tolist()); \
print('Gamma  a0', GammaRegressor(alpha=0.,max_iter=200).fit(X,y).coef_.tolist()); \
print('Tweed1.5', TweedieRegressor(power=1.5,alpha=0.,max_iter=200).fit(X,y).coef_.tolist()); \
print('Tweed0  ', TweedieRegressor(power=0.,alpha=0.).fit(X,y).coef_.tolist(), 'int', TweedieRegressor(power=0.,alpha=0.).fit(X,y).intercept_)"
# Poisson a0 [0.72931..]    intercept_ 0.09103   (REQ-1 unverified)
# Poisson a1 [0.64836..]    intercept_ 0.35515   (REQ-4: penalty scaling + intercept penalization)
# Gamma  a0 [0.75931..]                            (REQ-2 unverified)
# Tweed1.5 [0.74298..]                             (REQ-3 unverified)
# Tweed0   [5.9000..] int -5.5000  IDENTITY link   (REQ-8: ferrolearn always-log diverges)
python3 -c "import numpy as np; from sklearn.linear_model import PoissonRegressor; \
print(PoissonRegressor(alpha=1e6).fit(np.array([[1.],[2.],[3.],[4.]]),np.array([2.,5.,10.,20.])).intercept_, \
np.log(np.array([2.,5.,10.,20.]).mean()))"
# 2.2246.. 2.2246..   (REQ-4: intercept UNpenalized -> log(mean y); ferrolearn penalizes it)
```

A NOT-STARTED REQ closes only when its fix lands AND a divergence test (expected
values from the live oracle / a sklearn `file:line` constant per R-CHAR-3) goes
green; see the REQ-status table above for the current SHIPPED/NOT-STARTED split
(REQ-4/7/8/9/12 SHIPPED, the remainder gated on the blockers below).

## Blockers to open

- **#548** — REQ-1 of glm: pin `PoissonRegressor` fitted `coef_`/`intercept_`
  against the live sklearn `PoissonRegressor` oracle (alpha=0 and alpha>0); fix
  the objective so the optimum matches.
- **#549** — REQ-2 of glm: pin `GammaRegressor` `coef_`/`intercept_` against the
  live `GammaRegressor` oracle; reject `y==0` (`HalfGammaLoss` domain `0<y`)
  instead of clamping.
- **#550** — REQ-3 of glm: pin `TweedieRegressor(power=p)` `coef_`/`intercept_`
  against the live `TweedieRegressor` oracle for log-link powers.
- **#551** — REQ-4/REQ-6 of glm (crux): make IRLS minimize the same objective as
  sklearn — **mean** (per-sample average) half-deviance + `½·alpha·‖coef‖²`,
  with the **intercept excluded from the L2 penalty** (`weighted_ridge_solve`
  must not add `alpha` to the intercept's diagonal entry, and must scale the
  penalty relative to the mean, not summed, deviance).
- **#552** — REQ-5 of glm: initialize the intercept at `link(weighted_mean(y))`
  (`log(mean y)` for the log link) with feature coefficients at zero.
- **#553** — REQ-7 of glm: make `predict` apply `link.inverse` (exp for log,
  identity for identity), not an unconditional `exp`.
- **#554** — REQ-8 of glm: add the `TweedieRegressor` `link` parameter
  (`'auto'`/`'identity'`/`'log'`, default `'auto'`) with identity link for
  `power<=0` and log for `power>0`; support the identity-link IRLS/objective.
- **#555** — REQ-9 of glm: change `TweedieRegressor::new` default `power` from
  `1.5` to `0.0` (sklearn default), and fix `test_tweedie_defaults`.
- **#556** — REQ-10 of glm: add the `solver` parameter
  (`'lbfgs'`/`'newton-cholesky'`) and the gradient-norm (`max|g_j| <= tol`)
  stopping criterion.
- **#557** — REQ-11 of glm: add `warm_start` to reuse the previous
  `coef_`/`intercept_` as the next `fit` initialization.
- **#558** — REQ-12 of glm: add `sample_weight` to the averaged deviance, the
  weighted intercept init, and `score`.
- **#559** — REQ-13 of glm: implement `score(X, y) = D^2` (deviance-explained)
  with the link-mean null-deviance baseline.
- **#560** — REQ-14 of glm: expose `n_iter_` on the fitted object and apply
  per-family `in_y_true_range` y-domain validation (Gamma `0<y`).
- **#561** — REQ-15 of glm: migrate `glm.rs` off `ndarray` and the hand-rolled
  `cholesky_solve`/`gaussian_solve` onto the ferray substrate (`ferray-core`
  arrays, `ferray::linalg`) per R-SUBSTRATE.
