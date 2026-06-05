# Regression Metrics (sklearn.metrics regression functions)

<!--
tier: 3-component
status: draft
baseline-commit: a726bc5d
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/metrics/_regression.py   # _check_reg_targets (:75); mean_absolute_error (:161); mean_pinball_loss (:241); mean_absolute_percentage_error (:330); mean_squared_error (:426); root_mean_squared_error (:531); mean_squared_log_error (:606); root_mean_squared_log_error (:717); median_absolute_error (:790); _assemble_r2_explained_variance (:866); explained_variance_score (:929); r2_score (:1071); max_error (:1244); _mean_tweedie_deviance (:1277); mean_tweedie_deviance (:1318); mean_poisson_deviance (:1410); mean_gamma_deviance (:1453); d2_tweedie_score (:1501); d2_pinball_score (:1615); d2_absolute_error_score (:1766)
ferrolearn-module: ferrolearn-metrics/src/regression.rs
parity-ops: mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score, median_absolute_error, max_error, mean_squared_log_error, root_mean_squared_error, root_mean_squared_log_error, mean_pinball_loss, mean_tweedie_deviance, mean_poisson_deviance, mean_gamma_deviance, d2_absolute_error_score, d2_pinball_score, d2_tweedie_score
crosslink-issue: 753  # NOTE placeholder — replace with the regression-metrics unit's own tracking issue when the director assigns it (the 753-area "Layer 1 — ferrolearn-metrics" block).
-->

## Summary

`ferrolearn-metrics/src/regression.rs` mirrors the regression-metric functions of
scikit-learn's `sklearn/metrics/_regression.py`. It implements **seventeen** of
that module's public functions: `mean_absolute_error`, `mean_squared_error`,
`root_mean_squared_error`, `r2_score`, `mean_absolute_percentage_error`,
`explained_variance_score`, `median_absolute_error`, `max_error`,
`mean_squared_log_error`, `root_mean_squared_log_error`, `mean_pinball_loss`,
`mean_poisson_deviance`, `mean_gamma_deviance`, `mean_tweedie_deviance`,
`d2_absolute_error_score`, `d2_pinball_score`, and `d2_tweedie_score`. Every
public sklearn function mirrored by `_regression.py` is present — there are **no
MISSING functions** in this unit.

However, under honest underclaim (R-HONEST-3), **every present function DIVERGES
from the live sklearn 1.5.2 oracle on the API contract**, because all seventeen:

1. take **1D `&Array1<F>`** for `y_true`/`y_pred` only — sklearn accepts
   `(n_samples,)` **or `(n_samples, n_outputs)` 2D** via `_check_reg_targets`
   (`:75`), and
2. omit **`sample_weight`** (every sklearn function except `max_error` takes it),
   and
3. omit **`multioutput`** (`'raw_values'`/`'uniform_average'`/`'variance_weighted'`/
   array — present on every function except `max_error`).

On top of those three signature-wide gaps, several functions also have
**deterministic numerical/edge-handling divergences** even on 1D un-weighted
input. The three most important for a critic to pin first (all fully
deterministic, no `sample_weight`/`multioutput` involved):

1. **`r2_score` and `explained_variance_score` raise an error on constant
   `y_true`**; sklearn's default `force_finite=True` returns **0.0** (imperfect)
   or **1.0** (perfect) instead of erroring, and ferrolearn exposes no
   `force_finite` argument at all. Oracle: `r2_score([3,3,3],[3,3,2])` → sklearn
   **0.0**, ferrolearn returns `Err(NumericalInstability)`.
2. **`mean_absolute_percentage_error` silently SKIPS samples where `y_true == 0`**
   (and returns `+inf` if all are zero); sklearn instead divides by
   `max(|y_true|, eps)` with `eps = np.finfo(float64).eps` (`:403-404`), so a
   zero-`y_true` sample contributes a huge finite term, never skipped. Oracle:
   `mape([100,0,200],[110,999,200])` → sklearn **≈1.4997e18**, ferrolearn
   **0.05**.
3. **`r2_score`/`explained_variance_score` lack `multioutput="variance_weighted"`**
   and `force_finite`, and **`max_error` does not reject multi-output input**
   (sklearn raises `ValueError` for `y_type == "continuous-multioutput"`,
   `:1271-1275`).

All seventeen functions are existing pub APIs re-exported at the crate root
(`lib.rs`) and consumed by the in-crate `scorer.rs` (`get_scorer`); they are
grandfathered under S5/R-DEFER-1, so the re-export + scorer wiring is the
non-test production-consumer surface. Their existing `#[test]`s pin only
ferrolearn's narrower (1D, no-weight, no-`multioutput`) behavior and do not
establish sklearn parity.

## Algorithm (sklearn — the contract)

`_check_reg_targets(y_true, y_pred, multioutput)` (`:75`) reshapes 1D input to
`(n, 1)`, validates consistent length, determines `y_type`
(`continuous`/`continuous-multioutput`), and validates `multioutput` length
against `n_outputs`. Every metric below runs through it, so **every metric is
intrinsically multi-output** and column-wise; the scalar 1D case is just
`n_outputs == 1`.

### Simple averages

- **`mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average")`** (`:161`):
  `output_errors = np.average(|y_true - y_pred|, weights=sample_weight, axis=0)`
  (`:226`), then reduced per `multioutput` (`:228-230`).
- **`mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average")`** (`:426`):
  `np.average((y_true - y_pred)**2, weights=sample_weight, axis=0)` (`:516`),
  reduced per `multioutput`.
- **`root_mean_squared_error(..., sample_weight=None, multioutput="uniform_average")`** (`:531`):
  `sqrt(mean_squared_error(..., multioutput="raw_values"))` then averaged
  (`:577-585`) — note the sqrt is taken **per output before** the multioutput
  average, not after.
- **`mean_squared_log_error`** (`:606`) / **`root_mean_squared_log_error`** (`:717`):
  MSE / RMSE of `log1p(y_true)` vs `log1p(y_pred)`; raises `ValueError` if any
  input `< 0` (`:684-688`). RMSLE = `sqrt` of the per-output MSLE before
  averaging (`:773-781`).
- **`mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average")`** (`:241`):
  `np.average(alpha*max(diff,0) + (1-alpha)*max(-diff,0), weights=sample_weight, axis=0)`
  (`:312-316`), reduced per `multioutput`. `alpha` is keyword-only with default
  **0.5**.
- **`mean_absolute_percentage_error(..., sample_weight=None, multioutput="uniform_average")`** (`:330`):
  `mape = |y_pred - y_true| / np.maximum(|y_true|, eps)` with
  `eps = np.finfo(np.float64).eps` (`:403-404`) — **no sample is skipped**;
  zero-`y_true` divides by `eps`. Returns a **fraction**, not a percentage.

### Order statistics

- **`median_absolute_error(y_true, y_pred, *, multioutput="uniform_average", sample_weight=None)`** (`:790`):
  per-output median of `|y_true - y_pred|`; with `sample_weight`, a
  **weighted percentile-50** (`_weighted_percentile`, `:858-862`). Note the
  `multioutput`-before-`sample_weight` argument order.
- **`max_error(y_true, y_pred)`** (`:1244`): `np.max(|y_true - y_pred|)`. **No**
  `sample_weight`, **no** `multioutput`; **raises `ValueError` on
  multi-output** `y` (`:1271-1275`).

### R²-family (`_assemble_r2_explained_variance`, `:866`)

- **`r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", force_finite=True)`** (`:1071`):
  per-output `numerator = sum(w*(y_true-y_pred)^2)`,
  `denominator = sum(w*(y_true - wmean(y_true))^2)`, then `_assemble_*`. With
  **`force_finite=True` (default)**: zero numerator → score **1.0** (even if
  denominator is 0); nonzero numerator & zero denominator → **0.0** (never
  errors). With `force_finite=False`: raw `1 - num/den` (may be NaN/-Inf).
  `multioutput="variance_weighted"` weights outputs by `denominator` (`:900-906`).
- **`explained_variance_score(..., sample_weight=None, multioutput="uniform_average", force_finite=True)`** (`:929`):
  same assembly but `numerator = variance of (y_true - y_pred)` and
  `denominator = variance of y_true` (population, weighted, mean-subtracted),
  so it ignores a constant offset in the residuals. Same `force_finite`/
  `variance_weighted` semantics.

### Tweedie deviances

- **`mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0)`** (`:1318`)
  over `_mean_tweedie_deviance` (`:1277`): unit deviance per the Tweedie power,
  with `power` constraints (`0` normal; `1` Poisson `y_true>=0, y_pred>0`;
  `[1,2)` compound Poisson-Gamma; `2` Gamma `>0`; `>2` `y_true>0, y_pred>0`),
  `power` in `(0,1)` raises `ValueError` (`:1340-1364`). Weighted sample mean.
- **`mean_poisson_deviance(..., sample_weight=None)`** (`:1410`) = Tweedie `power=1`;
  **`mean_gamma_deviance(..., sample_weight=None)`** (`:1453`) = Tweedie `power=2`.

### D² scores (deviance-explained, R²-generalized)

- **`d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0)`** (`:1501`):
  `1 - dev(y_true, y_pred) / dev(y_true, wmean(y_true))`. With **< 2 samples,
  warns `UndefinedMetricWarning` and returns `nan`** (`:1565-1568`).
- **`d2_pinball_score(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average")`** (`:1615`):
  numerator = `mean_pinball_loss(raw_values)`; denominator vs the
  `np.percentile(y_true, alpha*100)` constant (weighted percentile if weighted,
  `:1712-1723`); `_assemble`-style: zero num → 1.0, nonzero num & zero den → 0.0
  (`:1736-1739`). **< 2 samples → warn + `nan`** (`:1699-1702`).
- **`d2_absolute_error_score(..., sample_weight=None, multioutput="uniform_average")`** (`:1766):
  `d2_pinball_score` with `alpha=0.5` (`:1837-1844`); the constant is the
  (weighted) **median** of `y_true`.

## ferrolearn (what exists)

All public functions live in `ferrolearn-metrics/src/regression.rs`, generic over
`F: Float + Send + Sync + 'static`, returning `Result<F, FerroError>`:

- **`pub fn mean_absolute_error(y_true: &Array1<F>, y_pred: &Array1<F>)`** — 1D
  only; no `sample_weight`, no `multioutput`.
- **`pub fn mean_squared_error(...)`** — same signature; always squared mean (no
  RMSE toggle — RMSE is its own function, matching sklearn 1.5.2 which removed the
  `squared=` kwarg).
- **`pub fn root_mean_squared_error(...)`** — `mse.sqrt()`.
- **`pub fn r2_score(...)`** — **returns `Err(FerroError::NumericalInstability)`
  when `SS_tot == 0`** (constant `y_true`); no `force_finite`, no `sample_weight`,
  no `multioutput`.
- **`pub fn mean_absolute_percentage_error(...)`** — **skips `y_true == 0`
  samples; returns `F::infinity()` if all are zero**; no `eps` denominator.
- **`pub fn explained_variance_score(...)`** — `1 - Var(residuals)/Var(y_true)`
  (population variance, mean-subtracted residuals — so it correctly ignores a
  constant offset). **Returns `Err(NumericalInstability)` when `Var(y_true)==0`**;
  no `force_finite`/`sample_weight`/`multioutput`.
- **`pub fn median_absolute_error(...)`** — even-`n` midpoint median (`n % 2 == 0`
  branch); no `sample_weight`, no `multioutput`.
- **`pub fn max_error(...)`** — `max |y_true - y_pred|`; 1D only — there is **no
  multi-output `ValueError` path** because the type system only accepts 1D.
- **`pub fn mean_squared_log_error(...)`** / **`pub fn root_mean_squared_log_error(...)`**
  — `InvalidParameter` on negatives; no `sample_weight`/`multioutput`.
- **`pub fn mean_pinball_loss(y_true, y_pred, alpha: F)`** — `alpha` is a
  **required positional** arg (sklearn: keyword-only, default 0.5); validates
  `alpha ∈ [0,1]`; no `sample_weight`/`multioutput`.
- **`pub fn mean_poisson_deviance(...)`**, **`pub fn mean_gamma_deviance(...)`**,
  **`pub fn mean_tweedie_deviance(y_true, y_pred, power: F)`** — `power` is a
  required positional arg (sklearn: keyword-only, default 0); constraint checks
  per power; reject `power ∈ (0,1)`; no `sample_weight`.
- **`pub fn d2_absolute_error_score(...)`**, **`pub fn d2_pinball_score(..., alpha: F)`**,
  **`pub fn d2_tweedie_score(..., power: F)`** — built on the private helpers
  `fn d2_score_with` (accumulates `num`/`den`), `fn quantile` (linear-interp
  empirical quantile), `fn mean`. **Return `Ok(F::zero())` when `den == 0`**
  (sklearn returns `nan` per output / on `<2` samples) and have **no `< 2`
  samples warn-and-`nan` path**, no `sample_weight`, no `multioutput`.

**Internal helpers (1D):** `fn check_same_length`, `fn check_non_empty` (the
validation path; note sklearn's `_check_reg_targets` additionally validates
multioutput length and y_type).

**Consumers (non-test):** crate re-export (`lib.rs`:
`pub use regression::{ ... all 17 ... }`) and the in-crate scorer
(`scorer.rs`: `pub fn get_scorer` maps `"neg_mean_absolute_error"`,
`"neg_mean_squared_error"`, `"r2"`, … to these functions via `make_scorer`).
These are existing pub APIs (grandfathered, S5/R-DEFER-1). **No `ferrolearn-python`
binding** exposes the regression metrics (a binding gap, REQ-19).

## Requirements

- REQ-1: **`mean_absolute_error` parity (R-DEV-1/2).** Match
  `mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average")`
  (`:161`): 2D `(n_samples, n_outputs)`, `sample_weight`, `multioutput` modes.
- REQ-2: **`mean_squared_error` parity (R-DEV-1/2).** Match `:426` incl.
  `sample_weight` + `multioutput`. (1D un-weighted value already matches.)
- REQ-3: **`root_mean_squared_error` parity (R-DEV-1/2).** Match `:531` —
  per-output sqrt **before** the multioutput average; `sample_weight`,
  `multioutput`.
- REQ-4: **`r2_score` parity (R-DEV-1/2).** Match
  `r2_score(..., sample_weight=None, multioutput="uniform_average", force_finite=True)`
  (`:1071`): **`force_finite` semantics — constant `y_true` returns 0.0/1.0, not
  an error**; `multioutput="variance_weighted"`; `sample_weight`.
- REQ-5: **`mean_absolute_percentage_error` parity (R-DEV-1).** Match `:330`:
  **divide by `max(|y_true|, eps)` — never skip zero-`y_true` samples**;
  `sample_weight`, `multioutput`.
- REQ-6: **`explained_variance_score` parity (R-DEV-1/2).** Match `:929`:
  **`force_finite` — constant `y_true` returns 0.0/1.0, not an error**;
  `multioutput="variance_weighted"`; `sample_weight`.
- REQ-7: **`median_absolute_error` parity (R-DEV-1/2).** Match `:790`:
  `sample_weight` (weighted percentile-50), `multioutput`; argument order
  `multioutput` before `sample_weight`. (1D un-weighted value already matches.)
- REQ-8: **`max_error` parity (R-DEV-1/2).** Match `:1244`: **`ValueError` on
  multi-output input** (`:1271-1275`). (1D scalar value already matches.)
- REQ-9: **`mean_squared_log_error` parity (R-DEV-1/2).** Match `:606`:
  `sample_weight`, `multioutput`. (1D un-weighted value + negative-input guard
  already match; sklearn raises `ValueError`, ferro `InvalidParameter`.)
- REQ-10: **`root_mean_squared_log_error` parity (R-DEV-1/2).** Match `:717`:
  per-output sqrt before average; `sample_weight`, `multioutput`.
- REQ-11: **`mean_pinball_loss` parity (R-DEV-2).** Match `:241`: `alpha`
  **keyword-only, default 0.5**; `sample_weight`, `multioutput`.
- REQ-12: **`mean_poisson_deviance` parity (R-DEV-1/2).** Match `:1410`:
  `sample_weight`. (1D un-weighted value + constraints already match.)
- REQ-13: **`mean_gamma_deviance` parity (R-DEV-1/2).** Match `:1453`:
  `sample_weight`. (1D un-weighted value + constraints already match.)
- REQ-14: **`mean_tweedie_deviance` parity (R-DEV-2).** Match `:1318`: `power`
  **keyword-only, default 0**; `sample_weight`. (1D un-weighted value +
  `(0,1)` rejection already match.)
- REQ-15: **`d2_tweedie_score` parity (R-DEV-1/2).** Match `:1501`: **`< 2`
  samples → `nan` (warn)**, `den == 0` per-output → `nan` not `0.0`;
  `power` keyword-only default 0; `sample_weight`.
- REQ-16: **`d2_pinball_score` parity (R-DEV-1/2).** Match `:1615`: **`< 2`
  samples → `nan`**, zero-num → 1.0 / nonzero-num & zero-den → 0.0 assembly;
  `alpha` keyword-only default 0.5; `sample_weight`, `multioutput`.
- REQ-17: **`d2_absolute_error_score` parity (R-DEV-1/2).** Match `:1766`:
  `d2_pinball_score(alpha=0.5)` semantics incl. **`< 2` samples → `nan`**;
  `sample_weight`, `multioutput`.
- REQ-18: **Cross-cutting 2D / `sample_weight` / `multioutput` shape contract
  (R-DEV-1/2/3).** Every metric (except `max_error`'s no-weight/no-multioutput
  signature) must accept `(n_samples, n_outputs)` via a `_check_reg_targets`
  analog and honor `'raw_values'`/`'uniform_average'`/`'variance_weighted'`/array
  `multioutput`.
- REQ-19: **PyO3 binding (R-DEFER-1).** `import sklearn.metrics` exposes these
  regression metrics; `ferrolearn-python` exposes no shim.
- REQ-20: **ferray substrate (R-SUBSTRATE).** `regression.rs` imports
  `ndarray::Array1` + `num_traits::Float`, not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (run from `/tmp`),
never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-4 pin): `r2_score([3,3,3],[3,3,2])` must equal sklearn **0.0**
  (constant `y_true`, imperfect → `force_finite`). ferrolearn returns
  `Err(NumericalInstability)` and FAILS. `r2_score([3,3,3],[3,3,3])` → sklearn
  **1.0**, ferro errors.
- AC-2 (REQ-5 pin): `mean_absolute_percentage_error([100,0,200],[110,999,200])`
  must equal sklearn **≈1.4997e18** (zero-`y_true` divided by `eps`, not skipped).
  ferrolearn returns **0.05** and FAILS. `mape([0,0],[1,2])` → sklearn
  **≈6.755e15**, ferro returns `+inf`.
- AC-3 (REQ-6 pin): `explained_variance_score([5,5,5],[1,2,3])` must equal sklearn
  **0.0** (constant `y_true`, `force_finite`). ferrolearn returns
  `Err(NumericalInstability)` and FAILS.
- AC-4 (REQ-8 pin): `max_error` on 2D `(n_samples, n_outputs)` `y` must raise the
  sklearn `ValueError` (`continuous-multioutput`). ferrolearn cannot express 2D
  input — signature-blocked.
- AC-5 (REQ-15/16/17 pin): `d2_absolute_error_score([1.0],[1.0])` must equal
  sklearn **`nan`** (`< 2` samples, `UndefinedMetricWarning`). ferrolearn returns
  **0.0** (its `den == 0` → `Ok(0)` path) and FAILS.
- AC-6 (REQ-1/2/7/18 `sample_weight`): `median_absolute_error([1,2,3,4],[1,2,5,6],
  sample_weight=[1,1,1,5])` must equal sklearn **2.0** (weighted percentile-50).
  ferrolearn has no `sample_weight` and FAILS to express it.
- AC-7 (REQ-18 `multioutput`): `mean_absolute_error([[1,1],[2,2]],[[1,0],[2,1]],
  multioutput="raw_values")` must equal sklearn **[0.0, 1.0]**; `"uniform_average"`
  → **0.5**; `r2_score(..., multioutput="variance_weighted")` weights by
  per-output denominator. ferrolearn is 1D-only and FAILS.
- AC-8 (value-correct baselines, must stay green): un-weighted 1D `mse([1,2,3],
  [1,2,4]) == 1/3`, `mae == 1/3`, `mean_poisson_deviance([1,2],[2,1]) == ln 2`,
  `mean_gamma_deviance([1,2],[2,1]) == 0.5`, `mean_tweedie_deviance(...,power=0)
  == mse`, `d2_tweedie_score([1,2,3],[1.5,2.5,2.5],power=0) == 0.625` — these
  already match sklearn and bound the correctness that IS present.

## REQ status table

Binary (R-DEFER-2). All seventeen functions are existing pub APIs re-exported at
the crate root and consumed by `scorer.rs::get_scorer` (the non-test
production-consumer surface; grandfathered per S5/R-DEFER-1). Cites use symbol
anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed
sklearn 1.5.2, run from `/tmp`. **No function is SHIPPED**: each is missing at
least `sample_weight` + `multioutput` + the 2D contract; several have additional
deterministic value/edge divergences. Honest underclaim (R-HONEST-3).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`mean_absolute_error`) | NOT-STARTED | open prereq blocker #761. `pub fn mean_absolute_error` value-correct on 1D un-weighted (`mae([1,2,3],[1.5,2,2.5])==1/3`) but takes only `&Array1`, no `sample_weight`/`multioutput` (sklearn `_regression.py:161,226-230`). |
| REQ-2 (`mean_squared_error`) | NOT-STARTED | open prereq blocker #762. `pub fn mean_squared_error` 1D un-weighted value matches (`==1/3`); no `sample_weight`/`multioutput`/2D (`:426,516`). |
| REQ-3 (`root_mean_squared_error`) | NOT-STARTED | open prereq blocker #763. `pub fn root_mean_squared_error` = `mse.sqrt()`; sklearn takes per-output sqrt **before** multioutput averaging (`:577-585`) — divergence once `multioutput`/2D exist; no `sample_weight`. |
| REQ-4 (`r2_score`) | NOT-STARTED | open prereq blocker #764. VALUE + `force_finite` VERIFIED correct vs live sklearn 1.5.2 (acto-critic audit, `tests/divergence_regression_audit.rs`): `r2_score([3,3,3],[3,3,2])` → **0.0**, perfect-constant → **1.0**, happy path `0.985` — all MATCH (`regression.rs:266-271` mirrors `_assemble_r2_explained_variance:877-891`; the prior "ferro errors" pin is STALE). Remaining gap (keeps this REQ NOT-STARTED): `sample_weight` / `multioutput="variance_weighted"` (the 2D/sample_weight/multioutput infra, REQ-18 #778). |
| REQ-5 (`mean_absolute_percentage_error`) | NOT-STARTED | open prereq blocker #765. VALUE VERIFIED correct vs live sklearn 1.5.2 (acto-critic audit, `tests/divergence_regression_audit.rs`): `mape([100,0,200],[110,999,200])` → **≈1.4996986759e18** (eps-clamp `denom = max(|y|, f64::EPSILON)`, `regression.rs:334-342` mirrors `_regression.py:403-404`) — MATCH (the prior "ferro 0.05 / skips zeros" pin is STALE, fixed in #769). Remaining gap (keeps this REQ NOT-STARTED): `sample_weight` / `multioutput` (REQ-18 #778). |
| REQ-6 (`explained_variance_score`) | NOT-STARTED | open prereq blocker #766. VALUE + `force_finite` VERIFIED correct vs live sklearn 1.5.2 (acto-critic audit, `tests/divergence_regression_audit.rs`): `evs([5,5,5],[1,2,3])` → **0.0**, perfect-constant → **1.0** (`regression.rs:415-420` mirrors `_regression.py:874-891`; offset-invariance correct; the prior "ferro errors" pin is STALE). Remaining gap (keeps this REQ NOT-STARTED): `sample_weight` / `multioutput` (REQ-18 #778). |
| REQ-7 (`median_absolute_error`) | NOT-STARTED | open prereq blocker #767. `pub fn median_absolute_error` 1D un-weighted value matches (even-`n` midpoint); **no `sample_weight` (weighted percentile-50, `:858-862`)**, no `multioutput`. Pin: weighted `[1,2,3,4]/[1,2,5,6]` w=`[1,1,1,5]` → sklearn **2.0**, ferro inexpressible. |
| REQ-8 (`max_error`) | NOT-STARTED | open prereq blocker #768. `pub fn max_error` 1D scalar value matches; **no multi-output `ValueError` path** (`:1271-1275`) — the 1D-only signature cannot represent the `continuous-multioutput` input sklearn rejects. |
| REQ-9 (`mean_squared_log_error`) | NOT-STARTED | open prereq blocker #769. `pub fn mean_squared_log_error` 1D un-weighted value + negative-guard match (ferro `InvalidParameter`, sklearn `ValueError`); no `sample_weight`/`multioutput` (`:606`). |
| REQ-10 (`root_mean_squared_log_error`) | NOT-STARTED | open prereq blocker #770. `pub fn root_mean_squared_log_error` = `msle.sqrt()`; per-output sqrt-before-average divergence under multioutput; no `sample_weight` (`:717,773-781`). |
| REQ-11 (`mean_pinball_loss`) | NOT-STARTED | open prereq blocker #771. `pub fn mean_pinball_loss(y_true,y_pred,alpha)` — `alpha` **required positional**; sklearn keyword-only default 0.5 (`:241-242`). 1D un-weighted value matches; no `sample_weight`/`multioutput`. |
| REQ-12 (`mean_poisson_deviance`) | NOT-STARTED | open prereq blocker #772. `pub fn mean_poisson_deviance` 1D value matches (`[1,2]/[2,1]==ln2`) + constraints; no `sample_weight` (`:1410`). |
| REQ-13 (`mean_gamma_deviance`) | NOT-STARTED | open prereq blocker #773. `pub fn mean_gamma_deviance` 1D value matches (`[1,2]/[2,1]==0.5`) + constraints; no `sample_weight` (`:1453`). |
| REQ-14 (`mean_tweedie_deviance`) | NOT-STARTED | open prereq blocker #774. `pub fn mean_tweedie_deviance(...,power)` — `power` **required positional**; sklearn keyword-only default 0 (`:1318`). 1D value + `(0,1)` rejection match; no `sample_weight`. |
| REQ-15 (`d2_tweedie_score`) | NOT-STARTED | open prereq blocker #775. `pub fn d2_tweedie_score` ratio value matches on `≥2` valid samples (`[1,2,3]/[1.5,2.5,2.5],p=0 == 0.625`), but **`den==0` → `Ok(0.0)`** and **no `<2`-samples→`nan` warn** (sklearn `:1565-1568`); `power` required-positional; no `sample_weight`. |
| REQ-16 (`d2_pinball_score`) | NOT-STARTED | open prereq blocker #776. `pub fn d2_pinball_score` uses `fn quantile` (linear-interp) constant — matches sklearn `np.percentile`; but **`den==0`→`Ok(0)`** vs sklearn zero-num→1.0/nonzero-num&zero-den→0.0 assembly (`:1736-1739`) and **no `<2`→`nan`** (`:1699-1702`); `alpha` required-positional; no `sample_weight`/`multioutput`. |
| REQ-17 (`d2_absolute_error_score`) | NOT-STARTED | open prereq blocker #777. VALUE + `<2`-samples behavior VERIFIED correct vs live sklearn 1.5.2 (acto-critic audit, `tests/divergence_regression_audit.rs`): `d2_absolute_error_score([1.0],[1.0])` → **NaN** (`regression.rs:1069-1071` `d2_score_with` mirrors `_regression.py:1699-1702`; the prior "ferro 0.0" pin is STALE). Remaining gap (keeps this REQ NOT-STARTED): `sample_weight` / `multioutput` (REQ-18 #778). |
| REQ-18 (2D/`sample_weight`/`multioutput`) | NOT-STARTED | open prereq blocker #778. **No `_check_reg_targets` analog**: `fn check_same_length`/`fn check_non_empty` validate only 1D length/non-empty. Every metric is 1D-`Array1`-only with no `sample_weight` and no `'raw_values'`/`'uniform_average'`/`'variance_weighted'`/array `multioutput` (`_regression.py:75,103-150`). Pin: `mae([[1,1],[2,2]],[[1,0],[2,1]],multioutput="raw_values")` → sklearn **[0,1]**, ferro inexpressible. |
| REQ-19 (PyO3 binding) | NOT-STARTED | open prereq blocker #779. `ferrolearn-python` exposes no regression-metric shim; `import ferrolearn` cannot call what `import sklearn.metrics` provides. |
| REQ-20 (ferray substrate) | NOT-STARTED | open prereq blocker #780. `regression.rs` imports `ndarray::Array1` + `num_traits::Float`, not `ferray-core` (R-SUBSTRATE). |

## Architecture

`regression.rs` is a flat module of free functions, each generic over
`F: Float + Send + Sync + 'static` and returning `Result<F, FerroError>`. There
are no fitted/unfitted types — these are stateless metrics. Three families:

1. **Simple averages** (`mean_absolute_error`, `mean_squared_error`,
   `root_mean_squared_error`, `mean_squared_log_error`,
   `root_mean_squared_log_error`, `mean_pinball_loss`,
   `mean_absolute_percentage_error`): single-pass folds over zipped
   `y_true`/`y_pred`. All are 1D and un-weighted. The MAPE fold's **skip-on-zero**
   branch (REQ-5) and the missing per-output-sqrt-before-average for RMSE/RMSLE
   (REQ-3/10) are the structural divergences here.
2. **Order statistics** (`median_absolute_error`, `max_error`): `median_absolute_error`
   collects + sorts the absolute residuals and takes the even-`n` midpoint /
   odd-`n` middle (the `n % 2 == 0` branch — see Verification re clippy #781);
   `max_error` folds a running max. Neither accepts `sample_weight`; `max_error`'s
   1D-only signature precludes the multi-output `ValueError` (REQ-8).
3. **R²-family and deviances** (`r2_score`, `explained_variance_score`,
   `mean_*_deviance`, `mean_tweedie_deviance`, `d2_*_score`): `r2_score`/
   `explained_variance_score` compute `SS_res`/`SS_tot` (resp. residual/target
   population variance) and **return `Err(NumericalInstability)` when the
   denominator is zero** — the single most consequential divergence, since
   sklearn's default `force_finite=True` returns 0.0/1.0 instead (REQ-4/6). The
   `d2_*` family routes through `fn d2_score_with` (num/den accumulation), `fn quantile`
   (linear-interp empirical quantile — matches `np.percentile`), and `fn mean`;
   their `den == 0 → Ok(0.0)` and missing `< 2`-samples→`nan` behavior diverge
   from sklearn's assembly + `UndefinedMetricWarning` (REQ-15/16/17). The Tweedie
   special-cases `power ∈ {0,1,2}` and delegates `power=1→mean_poisson_deviance`,
   `power=2→mean_gamma_deviance`.

**Invariants held vs sklearn (1D, un-weighted):** the core numeric formulas for
MAE, MSE, RMSE, MSLE/RMSLE, pinball, Poisson/Gamma/Tweedie deviance, and the D²
ratios are value-correct (AC-8); `explained_variance_score` correctly ignores a
constant residual offset; the deviance constraint checks and `(0,1)` Tweedie
rejection match. **Invariants NOT held vs sklearn:** 2D `(n_samples, n_outputs)`;
`sample_weight` (all); `multioutput` (all except `max_error`); `force_finite`
(r2/evs — they error instead); MAPE `eps` denominator (skips instead);
`d2_*` `< 2`-samples `nan` and `den==0` `nan`; `max_error` multi-output
`ValueError`; keyword-only `alpha`/`power` defaults.

**No MISSING functions:** every public function in `_regression.py` has a
ferrolearn counterpart. (sklearn also defines private helpers
`_check_reg_targets`, `_assemble_r2_explained_variance`, `_mean_tweedie_deviance`,
`_weighted_percentile` — the analogs of these are what REQ-18 / the weighted REQs
require.)

## Verification

Library crate (green at baseline `a726bc5d` for the existing — narrower —
contract):
```
cargo test -p ferrolearn-metrics --lib regression
cargo clippy -p ferrolearn-metrics --all-targets -- -D warnings
cargo fmt --all --check
```
The existing `#[test]`s (and `#[cfg(kani)]` proofs for MAE/MSE/RMSE/r2/evs
non-negativity/NaN-freedom) pin only ferrolearn's narrower behavior; they do NOT
establish sklearn parity, so they make no REQ SHIPPED.

**Known crate-gauntlet blocker — clippy `manual_is_multiple_of` (#781).**
`regression.rs:431` uses `n % 2 == 0` in the `median_absolute_error` median
computation. On the workspace MSRV (1.88, raised by the ferray substrate)
`n.is_multiple_of(2)` is available, so clippy's `manual_is_multiple_of` lint
fires under `-D warnings`. This blocks the `cargo clippy --all-targets -- -D
warnings` gate for this crate and must be cleared by an acto-fixer
(`n % 2 == 0` → `n.is_multiple_of(2)`). It is a lint-only fix, not a behavior
change. (NO `.rs` edit is made by this doc-author dispatch.)

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the deterministic
divergences a critic should pin first (R-CHAR-3 expected values):
```
# REQ-4 (r2 force_finite on constant y_true): sklearn 0.0 vs ferro Err
python3 -c "from sklearn.metrics import r2_score; print(r2_score([3,3,3],[3,3,2]))"          # 0.0
python3 -c "from sklearn.metrics import r2_score; print(r2_score([3,3,3],[3,3,3]))"          # 1.0
# REQ-6 (evs force_finite): sklearn 0.0 vs ferro Err
python3 -c "from sklearn.metrics import explained_variance_score as e; print(e([5,5,5],[1,2,3]))"  # 0.0
# REQ-5 (MAPE eps not skip): sklearn ~1.4997e18 vs ferro 0.05
python3 -c "from sklearn.metrics import mean_absolute_percentage_error as m; print(m([100,0,200],[110,999,200]))"  # 1.4997e18
python3 -c "from sklearn.metrics import mean_absolute_percentage_error as m; print(m([0,0],[1,2]))"  # 6.755e15 (ferro +inf)
# REQ-17/15/16 (d2 < 2 samples -> nan): sklearn nan vs ferro 0.0
python3 -c "from sklearn.metrics import d2_absolute_error_score as d; print(d([1.0],[1.0]))"  # nan
# REQ-7 (median weighted): sklearn 2.0, ferro inexpressible
python3 -c "from sklearn.metrics import median_absolute_error as md; print(md([1,2,3,4],[1,2,5,6],sample_weight=[1,1,1,5]))"  # 2.0
# REQ-18 (multioutput): sklearn [0,1] / 0.5
python3 -c "import numpy as np; from sklearn.metrics import mean_absolute_error as ma; print(ma(np.array([[1,1],[2,2]]),np.array([[1,0],[2,1]]),multioutput='raw_values'))"  # [0. 1.]
# AC-8 baselines that must stay green (value-correct today):
python3 -c "from sklearn.metrics import mean_poisson_deviance as p,mean_gamma_deviance as g,d2_tweedie_score as d; print(p([1,2],[2,1]),g([1,2],[2,1]),d([1,2,3],[1.5,2.5,2.5],power=0))"  # 0.6931... 0.5 0.625
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-metrics/tests/divergence_regression.rs`, asserting the live-sklearn
expected values above and FAILING against current `regression.rs`. Every REQ is
NOT-STARTED; each carries an open prereq blocker.

## Blockers to open

- #761 — REQ-1 (`mean_absolute_error`): no 2D/`sample_weight`/`multioutput`.
- #762 — REQ-2 (`mean_squared_error`): no 2D/`sample_weight`/`multioutput`.
- #763 — REQ-3 (`root_mean_squared_error`): no `sample_weight`/`multioutput`;
  per-output sqrt-before-average (`_regression.py:577-585`).
- #764 — REQ-4 (`r2_score`): errors on constant `y_true` instead of `force_finite`
  0.0/1.0 (`:866-891`); no `variance_weighted`/`sample_weight`. Pin: `r2_score([3,3,3],[3,3,2])` → sklearn 0.0, ferro Err.
- #765 — REQ-5 (`mean_absolute_percentage_error`): skips zero-`y_true` + returns
  `+inf`; sklearn uses `max(|y_true|, eps)` (`:403-404`). Pin: `[100,0,200]/[110,999,200]` → sklearn ≈1.4997e18, ferro 0.05.
- #766 — REQ-6 (`explained_variance_score`): errors on constant `y_true` instead
  of `force_finite` (`:874-891`); no `variance_weighted`/`sample_weight`. Pin: `[5,5,5]/[1,2,3]` → sklearn 0.0, ferro Err.
- #767 — REQ-7 (`median_absolute_error`): no `sample_weight` (weighted
  percentile-50, `:858-862`)/`multioutput`. Pin: weighted → sklearn 2.0.
- #768 — REQ-8 (`max_error`): no multi-output `ValueError` (`:1271-1275`);
  1D-only signature.
- #769 — REQ-9 (`mean_squared_log_error`): no `sample_weight`/`multioutput`.
- #770 — REQ-10 (`root_mean_squared_log_error`): no `sample_weight`/`multioutput`;
  per-output sqrt-before-average (`:773-781`).
- #771 — REQ-11 (`mean_pinball_loss`): `alpha` positional not keyword-only
  default-0.5; no `sample_weight`/`multioutput` (`:241-242`).
- #772 — REQ-12 (`mean_poisson_deviance`): no `sample_weight` (`:1410`).
- #773 — REQ-13 (`mean_gamma_deviance`): no `sample_weight` (`:1453`).
- #774 — REQ-14 (`mean_tweedie_deviance`): `power` positional not keyword-only
  default-0; no `sample_weight` (`:1318`).
- #775 — REQ-15 (`d2_tweedie_score`): `den==0`→0.0 and no `<2`-samples→`nan`
  warn (`:1565-1568`); `power` positional; no `sample_weight`.
- #776 — REQ-16 (`d2_pinball_score`): `den==0`→0.0 vs assembly (`:1736-1739`); no
  `<2`-samples→`nan` (`:1699-1702`); `alpha` positional; no `sample_weight`/`multioutput`.
- #777 — REQ-17 (`d2_absolute_error_score`): `<2` samples → `0.0` vs sklearn
  `nan` (`:1766,1699-1702`); no `sample_weight`/`multioutput`. Pin: `([1.0],[1.0])` → sklearn nan, ferro 0.0.
- #778 — REQ-18 (cross-cutting): no `_check_reg_targets` analog — every metric is
  1D-only, no `sample_weight`, no `multioutput` modes (`:75,103-150`).
- #779 — REQ-19: no `ferrolearn-python` regression-metric binding.
- #780 — REQ-20: migrate `regression.rs` off `ndarray`/`num-traits` to the ferray
  substrate (R-SUBSTRATE).
- #781 — Crate-gauntlet (not a sklearn divergence): clippy `manual_is_multiple_of`
  at `regression.rs:431` (`n % 2 == 0` → `n.is_multiple_of(2)`; MSRV 1.88). Blocks
  `cargo clippy --all-targets -- -D warnings`; acto-fixer clears it.
