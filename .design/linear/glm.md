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
`warm_start`, `sample_weight`, `score` (D²), `n_iter_`, and the ferray
substrate. No `coef_`/`intercept_` value is pinned against the sklearn oracle in
any current test, so per-family numerical parity cannot be claimed SHIPPED.

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
when `fit_intercept`), initializes `eta = log(y_safe)`, `mu = y_safe`,
`coef = 0`, then iterates: compute log-link IRLS weight
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
| REQ-1 (Poisson family + log link) | NOT-STARTED | open prereq blocker #548. Impl `fn fit_glm_irls in glm.rs` with `GLMFamily::Poisson` (`glm.rs:66` `GLMFamily::Poisson => mu`) does log-link IRLS, but NO test pins `coef_`/`intercept_` against the sklearn oracle — `test_poisson_fit_predict in glm.rs` asserts only `preds.len()==4` and `p > 0.0`. Per R-CHAR-3 a live oracle (`PoissonRegressor(alpha=0).fit -> coef_ [0.72931], intercept_ 0.09103`) parity test is required and absent; parity is unverified and entangled with the REQ-4 objective-scaling gap whenever `alpha>0`. |
| REQ-2 (Gamma family + log link) | NOT-STARTED | open prereq blocker #549. Impl `GLMFamily::Gamma => mu * mu` (`glm.rs:67`) drives the IRLS weight `mu^2/V(mu)`; `test_gamma_fit_predict in glm.rs` again checks only positivity/length. No oracle parity test (`GammaRegressor(alpha=0).fit -> coef_ [0.75931], intercept_ 0.00383`). sklearn's `HalfGammaLoss` y-domain is `0 < y` (strict); ferrolearn clamps `y<1e-10` up to `1e-10` (`glm.rs:593`) rather than rejecting `y==0`, a behavioral divergence folded here. |
| REQ-3 (Tweedie family + power) | NOT-STARTED | open prereq blocker #550. Impl `GLMFamily::Tweedie(p) => mu.powf(p)` (`glm.rs:68-71`); `test_tweedie_fit_predict in glm.rs` checks only positivity. No oracle parity test (`TweedieRegressor(power=1.5, alpha=0).fit -> coef_ [0.74298], intercept_ 0.04828`). Power values `p<=0` are mishandled by the always-log link (see REQ-8). |
| REQ-4 (penalized-objective parity: mean half-deviance + ½·alpha, intercept unpenalized) | NOT-STARTED | open prereq blocker #551 (the crux). `fn weighted_ridge_solve in glm.rs` adds `alpha` to **every** diagonal entry including the intercept column: `for i in 0..n_features { xtwx[[i, i]] = xtwx[[i, i]] + alpha; }` (`glm.rs:520-522`) — the comment says "do not penalise intercept column" but the loop spans all `n_features`, so the intercept (design column 0) IS penalized, contradicting sklearn (`glm.py:258`; live oracle: `alpha=1e6 -> intercept_ -> log(mean y)`, not 0). Moreover the normal equations `(X^T W X + alpha I) w = X^T W z` minimize the **summed** weighted deviance `½·Σ w_i (z_i - X_i w)^2 + ½·alpha·‖w‖^2`, whereas sklearn minimizes the **mean** half-deviance + `½·alpha·‖coef‖^2` (`glm.py:229-258`): ferrolearn's effective penalty is too weak by a factor of `n_samples`. Both shift the optimum for any `alpha>0`. |
| REQ-5 (intercept init = link(mean y)) | NOT-STARTED | open prereq blocker #552. `fn fit_glm_irls in glm.rs` initializes `coef = Array1::zeros(n_cols)` (`glm.rs:598`) — the intercept entry starts at **0**, with `eta = log(y_safe)`, `mu = y_safe` (`glm.rs:596-597`). sklearn initializes `coef[-1] = link.link(average(y))` (`= log(mean y)` for the log link) with feature coefs at zero (`glm.py:252-256`). The starting point differs; for the convex objective this is path-only (does not by itself break the optimum) but is required to mirror sklearn's first iterate / `n_iter_`. |
| REQ-6 (fit_intercept) | NOT-STARTED | open prereq blocker #551. `fn fit_glm_irls in glm.rs` prepends a constant design column when `fit_intercept` and sets `intercept_ = 0` otherwise (`glm.rs:574-590, :651-657`) — the *structure* is correct, but the fitted intercept is **penalized** (REQ-4), so the `fit_intercept=true` result diverges from sklearn's unpenalized intercept whenever `alpha>0`. Gated on the same objective fix as REQ-4 (#551). |
| REQ-7 (predict = link.inverse) | NOT-STARTED | open prereq blocker #553. `fn predict in glm.rs` computes `exp(X·coef + intercept)` unconditionally (`glm.rs:805-806`: `let eta = x.dot(&self.coefficients) + self.intercept; Ok(eta.mapv(|v| v.exp()))`). This matches sklearn ONLY for the log link; for the identity link (Tweedie `power<=0`, REQ-8) sklearn returns the raw linear predictor (`glm.py:362` `link.inverse`). No oracle test pins `predict` against `m.predict`. |
| REQ-8 (TweedieRegressor link='auto'/identity/log) | NOT-STARTED | open prereq blocker #554. ferrolearn uses a log link **always** (`glm.rs:11-12` doc; `fn fit_glm_irls` initializes `eta = log(y)` and `predict` applies `exp`), with no `link` field on `TweedieRegressor` (`glm.rs:310-321`). sklearn's `TweedieRegressor._get_loss` (`glm.py:889-903`) selects identity for `power<=0` and log for `power>0` under `link='auto'`, and exposes `link={'auto','identity','log'}` (`glm.py:861, :868-870`). ferrolearn diverges for `power<=0` (Normal/Ridge): live oracle `TweedieRegressor(power=0.0, alpha=0).fit -> coef_ [5.9000], intercept_ -5.5000` (identity), which the always-log path cannot reproduce. |
| REQ-9 (default params; Tweedie power=0.0) | NOT-STARTED | open prereq blocker #555. `alpha=1.0`, `max_iter=100`, `tol=1e-4`, `fit_intercept=true` match (`glm.rs:111-114, :191-194, :259-262, :332-335`; `test_tweedie_defaults in glm.rs`). BUT `TweedieRegressor::new` sets `power: 1.5` (`glm.rs:331`), whereas sklearn's default is `power=0.0` (Normal) (`glm.py:737, :867`) — a default-value divergence (R-DEV-2). `test_tweedie_defaults` asserts the wrong default `m.power == 1.5`. |
| REQ-10 (solver param) | NOT-STARTED | open prereq blocker #556. No `solver` field on any GLM struct (`glm.rs:88-100, :171-180, :240-249, :310-321`); ferrolearn always runs IRLS. sklearn exposes `solver={'lbfgs','newton-cholesky'}` default `'lbfgs'` (`glm.py:140-143, :155, :263-309`) with a gradient-norm stopping rule (`max|g_j| <= tol`, `glm.py:86-91`), whereas ferrolearn stops on `max|Δcoef| < tol` (`glm.rs:639-647`) — a different convergence criterion that affects `n_iter_` and near-tol fits. (Solver mechanism per se is not a divergence — see "Solver / optimum equivalence" — but the absent `solver` param and the different stop rule are.) |
| REQ-11 (warm_start) | NOT-STARTED | open prereq blocker #557. No `warm_start` field; `fn fit_glm_irls` always cold-starts from `coef = 0` (`glm.rs:598`). sklearn reuses `coef_`/`intercept_` when `warm_start and hasattr(self,'coef_')` (`glm.py:146, :244-250`). |
| REQ-12 (sample_weight) | NOT-STARTED | open prereq blocker #558. `Fit` for the GLM types takes only `(x, y)` (`glm.rs:669-693, :699-777`); no `sample_weight`. sklearn threads `sample_weight` through the averaged deviance, the weighted intercept init, and `score` (`glm.py:170, :208-211, :255, :419-431`). |
| REQ-13 (score = D²) | NOT-STARTED | open prereq blocker #559. `FittedGLMRegressor` implements `Predict`/`HasCoefficients` only (`glm.rs:783-820`); no `score`. sklearn's `score` returns D² (deviance-explained) with the link-mean null model (`glm.py:365-438`). |
| REQ-14 (n_iter_ introspection / y-domain) | NOT-STARTED | open prereq blocker #560. `FittedGLMRegressor` stores only `coefficients`, `intercept` (`glm.rs:151-157`); the IRLS iteration count is discarded (`for _iter in 0..max_iter`, `glm.rs:603`). sklearn exposes `n_iter_` (`glm.py:283, :296`). Also y-domain validation is family-flat (`y<0 -> error`, `y` clamped to `1e-10`, `glm.rs:563-571, :593`) rather than per-loss `in_y_true_range` (Gamma: `0<y`; `glm.py:221-225`). |
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

Every REQ-1..REQ-15 lacks green oracle verification and is NOT-STARTED, each
gated on the blocker below. A NOT-STARTED REQ closes only when its fix lands AND
a divergence test (expected values from the live oracle / a sklearn `file:line`
constant per R-CHAR-3) goes green.

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
