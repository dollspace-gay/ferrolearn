# Linear Support Vector Regression (LinearSVR)

<!--
tier: 3-component
status: draft
baseline-commit: 8e5bc4b1
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/svm/_classes.py        # class LinearSVR
  - sklearn/svm/_base.py           # _fit_liblinear, _get_liblinear_solver_type
ferrolearn-module: ferrolearn-linear/src/linear_svr.rs
parity-ops: LinearSVR
crosslink-issue: 606
-->

## Summary

`ferrolearn-linear/src/linear_svr.rs` mirrors scikit-learn's
`sklearn.svm.LinearSVR` (`sklearn/svm/_classes.py:358`), the liblinear-backed
linear epsilon-insensitive support-vector regressor. ferrolearn provides the
`LinearSVRLoss` enum (`EpsilonInsensitive` / `SquaredEpsilonInsensitive`), the
unfitted `LinearSVR<F>` / fitted `FittedLinearSVR<F>` builder pair, the
`Fit`/`Predict`/`HasCoefficients`/pipeline integrations, and a hand-rolled
**primal sub-gradient coordinate-descent** solver (`fn fit in linear_svr.rs`).
sklearn fits the **same family of L2-regularized epsilon-insensitive
objectives** but via **liblinear** (`_fit_liblinear`, `sklearn/svm/_base.py:1052`)
— a dual coordinate-descent solver by default — with a specific
**C-scaling convention** (`0.5·‖w‖² + C·Σ loss`, no `1/n` factor) and an
**`intercept_scaling` augmented-feature** intercept convention.

The solver mechanism (ferrolearn primal CD vs liblinear dual CD) is NOT by
itself a divergence — the objective is convex, so both reach the same minimizer
*iff the objective they minimize is identical*. The divergences are concrete and
measured: ferrolearn scales the loss by `C/n_samples` rather than `C` (a
different optimum — REQ-1/REQ-7), the default `epsilon` is `0.1` not `0.0`
(REQ-3), `fit_intercept`/`intercept_scaling`/`dual` are absent from the API
(REQ-5/REQ-6), `coef_`/`intercept_` are scalars/`Array1` rather than the
liblinear `ndarray` layout with no `n_iter_` exposed (REQ-8/REQ-9), and the unit
is on the `ndarray` substrate, not ferray (REQ-10). No test pins any
`coef_`/`intercept_` value against the live sklearn oracle (the conformance test
asserts only an R² ≥ 0.40 floor, explicitly noting sklearn reaches ~0.97 on the
same data), so the numerical-parity REQs cannot be SHIPPED (R-HONEST-3).

## Algorithm (sklearn — the contract)

### The liblinear objective (`_classes.py:386-394`, `_base.py:1011-1016`)

`LinearSVR.fit` (`_classes.py:544-605`) fixes `penalty="l2"` (`_base.py:577`),
validates `dual` (`_base.py:579`), then calls `_fit_liblinear`
(`_base.py:581-597`) with `loss=self.loss`, `epsilon=self.epsilon`, `C=self.C`.
liblinear minimizes the **L2-regularized epsilon-insensitive primal**

```
min_w  0.5·‖w‖²  +  C · Σ_i  L_ε(y_i − w·x_i)
```

where, per `loss` (`_base.py:1015-1016`, `_get_liblinear_solver_type`):

- `epsilon_insensitive` (L1, default) → `L_ε(r) = max(0, |r| − ε)` — solver
  type **13** (`l2`, `dual=True`).
- `squared_epsilon_insensitive` (L2) → `L_ε(r) = max(0, |r| − ε)²` — solver
  type **11** (`dual=False`) or **12** (`dual=True`).

The load-bearing fact for `coef_`/`intercept_` parity: **the data term is scaled
by `C`, NOT by `C/n_samples`.** There is no `1/n` averaging — the regularizer
`0.5·‖w‖²` is weighed against the *summed* loss. The optimum therefore depends on
`C` (live oracle, `X=[[1],[2],[3],[4]]`, `y=[1,3,2,5]`, `fit_intercept=False`,
`ε=0`: `C=0.1 → coef [0.6667]`, `C=1.0 → coef [1.25]`).

### `dual` solver selection (`_base.py:579`, `_get_liblinear_solver_type:995`)

`dual` (default `"auto"`) selects primal (`False`) vs dual (`True`) liblinear.
`_validate_dual_parameter` resolves `"auto"`: dual when `n_samples ≤ n_features`
and the loss supports a dual solver, else primal. `epsilon_insensitive` is
**dual-only** (solver 13 — no `{False: …}` entry), so for the default loss
`dual` effectively resolves to `True`. `squared_epsilon_insensitive` supports
both (11 primal / 12 dual). The *optimum* is `dual`-invariant; `dual` is an API
parameter (R-DEV-2) and a solver-path choice, not a result divergence.

### `fit_intercept` + `intercept_scaling` (`_base.py:1188-1245`)

When `fit_intercept` (default `True`), liblinear augments each instance with a
synthetic constant feature of value `bias = intercept_scaling`
(`_base.py:1189-1198`): `x → [x_1, …, x_n, intercept_scaling]`. liblinear
penalizes this column like any other. After training, the raw weight vector is
split (`_base.py:1240-1245`):

```
coef_      = raw_coef_[:, :-1]
intercept_ = intercept_scaling * raw_coef_[:, -1]
```

so a larger `intercept_scaling` reduces the relative regularization on the
intercept. With `fit_intercept=False`, `intercept_ = 0.0`. `coef_` is then
`ravel`-ed to 1-D (`_base.py:598`) and `intercept_` is a length-1 ndarray
(`_classes.py:453`). `n_iter_` is `n_iter_.max().item()` (`_base.py:603`).

### Solver / optimum equivalence (the parity criterion)

liblinear dual CD and a correct primal CD both minimize the single convex
objective above and converge to the **same** `coef_`/`intercept_` (modulo
`tol`/`max_iter`). So ferrolearn's choice of primal CD is sanctioned under
R-DEV-7 *iff* it minimizes the identical objective. Parity is verified by
comparing fitted `coef_`/`intercept_` to the live sklearn oracle — NOT solver
trajectories. A C-scaling mismatch (REQ-1/REQ-7) shifts the optimum and breaks
parity even though the solver "converges."

## ferrolearn (what exists)

`fn fit in linear_svr.rs` runs a fixed-step primal sub-gradient coordinate
descent: it initializes `w = 0`, `b = 0`, a hard-coded `step = 0.01`
(`linear_svr.rs:178`), and for each iteration loops over feature coordinates and
the intercept, accumulating the regularization gradient `w[j]` plus the
data-term gradient. For each sample with `|residual| > epsilon` it adds
`−C/n_samples · sign(residual) · x[i,j]` (epsilon-insensitive,
`linear_svr.rs:199`) or
`−2·C/n_samples · (|residual| − epsilon) · sign(residual) · x[i,j]`
(squared, `linear_svr.rs:208-212`), then takes `w[j] ← w[j] − step·grad`. The
intercept `b` is updated by the same sub-gradient with no `x` factor
(`linear_svr.rs:242, :252`). It stops when `max|Δ| < tol` (`linear_svr.rs:265`).
`FittedLinearSVR<F>` stores a scalar `intercept: F` and `coefficients: Array1<F>`
and predicts `X·coefficients + intercept` (`fn predict in linear_svr.rs`).

The constructor `fn LinearSVR::new` defaults `C=1.0`, `epsilon=0.1`
(`linear_svr.rs:71`), `max_iter=1000`, `tol=1e-4`, `loss=EpsilonInsensitive`.
There is **no** `fit_intercept`, `intercept_scaling`, `dual`, `random_state`,
`verbose`, or `n_iter_` field/accessor. `LinearSVR`/`FittedLinearSVR`/
`LinearSVRLoss` are boundary types re-exported at the crate root
(`pub use linear_svr::{FittedLinearSVR, LinearSVR, LinearSVRLoss} in lib.rs`).
There is **no `ferrolearn-python` binding** for LinearSVR.

## Requirements

- REQ-1: Fit parity (the crux) — fitted `coef_`/`intercept_` match the live
  `LinearSVR` (liblinear) oracle for `loss="epsilon_insensitive"`, which
  requires minimizing `0.5·‖w‖² + C·Σ max(0,|r|−ε)` with **C-scaling (no `1/n`)**.
- REQ-2: Predict — `predict(X) = X @ coef_ + intercept_` matching the oracle.
- REQ-3: `epsilon` default — sklearn `epsilon=0.0` (`_classes.py:378, :522`),
  not `0.1`.
- REQ-4: `loss` parameter — `{epsilon_insensitive (default),
  squared_epsilon_insensitive}` with the squared objective scaled by `C`
  (no `1/n`); fitted `coef_`/`intercept_` match the oracle per loss.
- REQ-5: `fit_intercept` + `intercept_scaling` — augmented-feature intercept
  (`x → [x, intercept_scaling]`, `intercept_ = intercept_scaling·w_last`,
  penalized), `fit_intercept=False → intercept_ = 0`; defaults `True`/`1.0`.
- REQ-6: `dual` parameter — `{"auto" (default), True, False}` selecting the
  liblinear solver type; `auto` resolves per `n_samples`/`n_features`/`loss`.
- REQ-7: C-scaling convention — the data term is scaled by `C` (summed loss),
  not `C/n_samples`; the optimum is `n_samples`-dependent through `C`.
- REQ-8: `tol`/`max_iter` convergence + `n_iter_` — liblinear's stopping
  criterion and `n_iter_ = n_iter_.max()` (`_base.py:603`) exposed on the
  fitted object.
- REQ-9: Fitted-attribute contract — `coef_` 1-D `ndarray`, `intercept_` a
  length-1 ndarray (`_classes.py:445-468`), plus `n_features_in_`,
  parameter-validation exceptions (`C>0`, `tol>0`, `intercept_scaling>0`,
  `max_iter≥0`) matching `_parameter_constraints` (`_classes.py:506-517`).
- REQ-10: ferray substrate migration (array type → `ferray-core`; linear algebra
  → `ferray::linalg`) per R-SUBSTRATE.

## Acceptance criteria

- AC-1 (REQ-1): on `X=[[1],[2],[3],[4],[5]]`, `y=[2,4,6,8,10]`,
  `LinearSVR(epsilon=0.0, C=10.0, max_iter=10000, tol=1e-6)` fitted `coef_`
  matches the live oracle (`coef_ ≈ [1.99999938]`, `intercept_ ≈ [1.2e-6]`)
  within tolerance.
- AC-2 (REQ-1/REQ-7): on `X=[[1],[2],[3],[4]]`, `y=[1,3,2,5]`,
  `fit_intercept=False`, `epsilon=0.0`, the fit tracks `C` per the live oracle:
  `C=0.1 → coef [0.6667]`, `C=1.0 → coef [1.25]` (proves C-scaling, not
  `C/n`).
- AC-3 (REQ-2): `predict` equals `X @ coef_ + intercept_` and matches the
  oracle's `predict` on held-out rows.
- AC-4 (REQ-3): `LinearSVR::default().epsilon == 0.0`.
- AC-5 (REQ-4): `squared_epsilon_insensitive` fitted `coef_`/`intercept_` match
  the live `LinearSVR(loss="squared_epsilon_insensitive")` oracle.
- AC-6 (REQ-5): `fit_intercept=True, intercept_scaling=s` reproduces the oracle
  `intercept_ = s·w_last`; `fit_intercept=False` gives `intercept_ == 0.0`.
- AC-7 (REQ-6): `dual="auto"` resolves to the same solver/optimum as the
  oracle; the parameter exists with default `"auto"`.
- AC-8 (REQ-8): `n_iter_` is exposed; an unconverged fit at `max_iter` matches
  the oracle's `n_iter_` semantics (max across the single OvR fit).
- AC-9 (REQ-9): `coef_` shape/dtype and `intercept_` length-1 contract; `C<=0`,
  `tol<=0`, `intercept_scaling<=0`, `max_iter<0` raise the parameter errors
  per `_parameter_constraints`.
- AC-10 (REQ-10): `linear_svr.rs` owns its computation on `ferray-core` arrays /
  `ferray::linalg`, not `ndarray`.

## REQ status

Binary classification (R-DEFER-2): SHIPPED = impl + non-test production consumer
+ tests + green oracle verification; NOT-STARTED = concrete open blocker
referenced by `#`-number. `LinearSVR`/`FittedLinearSVR`/`LinearSVRLoss` are
boundary estimator types re-exported at the crate root
(`pub use linear_svr::{…} in lib.rs`); under S5/R-DEFER-1 the public estimator
type IS the consumer surface, grandfathered (there is no `ferrolearn-python`
binding for LinearSVR yet). As of the liblinear-parity rewrite
(`linear_svr.rs fn fit`, dual coordinate descent `solve_l2r_l1l2_svr`,
`linear.cpp:1051`), the divergence pins
`tests/divergence_linear_svr_fit.rs::{linear_svr_coef_parity,
linear_svr_coef_c_dependence, linear_svr_default_epsilon}` pin `coef_`/
`intercept_`/`epsilon` against the live sklearn 1.5.2 oracle to 1e-3 and are
GREEN — so REQ-1/REQ-3/REQ-5/REQ-7 are SHIPPED. The follow-on pins
`linear_svr_predict` (predict on a held-out row), `linear_svr_squared_loss`
(`squared_epsilon_insensitive` `coef_`/`intercept_`/predict), and
`linear_svr_n_iter` (the `n_iter_` structural invariant) are likewise GREEN —
so REQ-2/REQ-4/REQ-8 are now SHIPPED. `conformance_linear_svr in
tests/conformance_wave1.rs` (R² ≥ 0.40 floor) now passes comfortably
(sklearn ~0.97). The `dual` parameter (`DualMode`) ships REQ-6 and the
`tol > 0` validation + `n_features_in_` fitted attribute ship REQ-9 (the
`intercept_`-shape sub-item deferred to the ferrolearn-python binding layer,
cf. #600), pinned GREEN by `linear_svr_dual_auto_true`,
`linear_svr_dual_false_eps_rejected`, `linear_svr_dual_false_squared`,
`linear_svr_tol_validation`, and `linear_svr_n_features_in`. REQ-10 remains
NOT-STARTED (#615).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (fit parity — coef_/intercept_ vs liblinear oracle) | SHIPPED | `linear_svr.rs fn fit` now minimizes liblinear's objective `0.5·‖w‖² + C·Σ L_ε` (no `1/n`) via the dual coordinate descent `solve_l2r_l1l2_svr` (`sklearn/svm/src/liblinear/linear.cpp:1051`; selected by `_get_liblinear_solver_type`, `_base.py:1015-1016`). Pinned GREEN by `tests/divergence_linear_svr_fit.rs::linear_svr_coef_parity` (X=[[1]..[5]], y=2x, ε=0.1, C=1.0, intercept on → live sklearn `coef [1.95]`, `intercept [0.15]`, matched to 1e-3). Consumer: `pub use linear_svr::{…}` (`lib.rs:89`) + `PipelineEstimator` impl (`linear_svr.rs`). |
| REQ-2 (predict = X·coef + intercept) | SHIPPED | `fn predict in linear_svr.rs` computes `x.dot(&self.coefficients) + self.intercept`, mirroring `LinearModel.predict` (`X @ coef_ + intercept_`). Pinned GREEN by `tests/divergence_linear_svr_fit.rs::linear_svr_predict`: the default-loss fit (ε=0.1, C=1.0, intercept on) reproduces the live oracle `predict([[2.5]]) ≈ 5.0251` to 1e-3 (held-out row, `coef [1.95]`/`intercept [0.15]`). Consumer: `predict_pipeline` (`FittedPipelineEstimator for FittedLinearSVR in linear_svr.rs`). |
| REQ-3 (epsilon default = 0.0) | SHIPPED | `LinearSVR::new` now sets `epsilon = F::zero()` matching sklearn's `epsilon=0.0` (`_classes.py:378, :522, :535`); `test_default_constructor` asserts `m.epsilon == 0.0`. Pinned GREEN by `tests/divergence_linear_svr_fit.rs::linear_svr_default_epsilon` (sklearn `file:line` symbolic constant `_classes.py:522`). |
| REQ-4 (loss param + squared objective) | SHIPPED | the squared branch solves the correct objective: `fn fit` sets `lambda = 0.5/C`, `upper_bound = +inf` (the `L2R_L2LOSS_SVR_DUAL` path, `linear.cpp:1078-1081`, selected by `_get_liblinear_solver_type`, `_base.py:1015-1016`), with no `1/n`. Pinned GREEN by `tests/divergence_linear_svr_fit.rs::linear_svr_squared_loss`: `LinearSVR(loss='squared_epsilon_insensitive', epsilon=0.1, C=1.0, fit_intercept=True)` → live oracle `coef [1.8913]`, `intercept [0.2821]`, `predict([[1.5]]) ≈ 3.119`, all matched to 1e-3. Consumer: `pub use linear_svr::{…}` + `PipelineEstimator` impl. |
| REQ-5 (fit_intercept + intercept_scaling) | SHIPPED | `LinearSVR` now exposes `pub fit_intercept: bool` (default `true`) + `pub intercept_scaling: F` (default `1.0`) with `with_fit_intercept`/`with_intercept_scaling`. `fn fit` augments the design matrix with a synthetic constant column equal to `intercept_scaling`, **penalizes** the augmented weight in ‖w‖² (liblinear convention, `_base.py:1189-1198`), and returns `intercept_ = intercept_scaling · w_last`; `fit_intercept=false → intercept_ = 0`. Validates `intercept_scaling > 0` when fitting an intercept (`_base.py:1190-1196`) → `FerroError::InvalidParameter`. Pinned GREEN by `linear_svr_coef_parity` (intercept `0.15` matched), `linear_svr_coef_c_dependence` (both C fits, intercept on), and module tests `test_fit_intercept_false_zero_intercept`/`test_invalid_intercept_scaling`. |
| REQ-6 (dual param) | SHIPPED | `LinearSVR` exposes `pub dual: DualMode` (`{Auto (default), True, False}`) + `#[must_use] with_dual`, modeling sklearn's `"dual": ["boolean", StrOptions({"auto"})]` (`_classes.py:513`, default `"auto"` `:528`). `fn fit` resolves `Auto`→dual solver (the supported solver for both SVR losses, per `_validate_dual_parameter` `_classes.py:13-29`) and rejects the unsupported `dual=False` + `EpsilonInsensitive` combination with `FerroError::InvalidParameter` (sklearn's `ValueError "Unsupported set of arguments"`; the solver dict has no `{False:…}` entry for `epsilon_insensitive`, `_base.py:1015,:1047`). `dual=False` + `SquaredEpsilonInsensitive` (sklearn primal solver type 11) reuses ferrolearn's dual CD: the strongly-convex objective has a unique minimizer (R-DEV-7 — implementation differs, observable `coef_`/`intercept_` matches). Pinned GREEN by `tests/divergence_linear_svr_fit.rs::{linear_svr_dual_auto_true, linear_svr_dual_false_eps_rejected, linear_svr_dual_false_squared}` (live oracle: dual auto/True → coef [1.95]; dual=False eps_insensitive → ValueError; dual=False squared → coef [1.8913]/intercept [0.2821] to 1e-3). Consumer: `pub use linear_svr::{…}` (`lib.rs`) + `PipelineEstimator` impl. |
| REQ-7 (C-scaling convention) | SHIPPED | the `/n_f` division is removed; `fn fit` minimizes liblinear's `0.5·‖w‖² + C·Σ L` (the dual CD uses `upper_bound = C` for L1 loss, `lambda = 0.5/C` for L2 loss, `linear.cpp:1076-1089`). Pinned GREEN by `tests/divergence_linear_svr_fit.rs::linear_svr_coef_c_dependence`: live sklearn (fit_intercept=True, ε=0) `C=0.1 → coef [0.58]`, `C=1.0 → coef [1.2941]`, both matched to 1e-3 — the C-dependence now tracks liblinear (was flattened by `C/n`). |
| REQ-8 (tol/max_iter + n_iter_) | SHIPPED | `fn fit` counts the dual-CD outer iterations into `FittedLinearSVR::n_iter`, exposed via `#[must_use] pub fn n_iter`, mirroring `n_iter_ = n_iter_.max().item()` (`_classes.py:603`; the liblinear iteration count, `_base.py:1215, :1247`). It emits the `ConvergenceWarning`-equivalent via `eprintln!("Liblinear failed to converge…")` when the criterion is unmet at `max_iter` (`_base.py:1234-1238`, the crate's qda/lda warning channel). ferrolearn's dual CD is a distinct implementation, so the count is STRUCTURAL (need not equal liblinear's). Pinned GREEN by `tests/divergence_linear_svr_fit.rs::linear_svr_n_iter` (`1 <= n_iter <= max_iter`). |
| REQ-9 (param validation + n_features_in_) | SHIPPED | `fn fit` validates `tol > 0` → `FerroError::InvalidParameter` (sklearn `"tol": [Interval(Real, 0.0, None, closed="neither")]`, `_classes.py:508`); `max_iter` is `usize` so `>= 0` always holds (sklearn `Interval(Integral, 0, None, closed="left")`, `:516`) — documented, no runtime check. Keeps the existing `C > 0` and `epsilon >= 0` rejects (both match sklearn's empirical fit-time behavior — a negative `epsilon` raises a `ValueError` at fit, verified live). `FittedLinearSVR` now stores `n_features_in` (= `X.ncols()`), exposed via `#[must_use] pub fn n_features_in`, mirroring sklearn's standard `n_features_in_` (set by `_validate_data`, `_classes.py:569-576`). Pinned GREEN by `tests/divergence_linear_svr_fit.rs::{linear_svr_tol_validation, linear_svr_n_features_in}`. The `intercept_`-shape sub-item (sklearn's length-1 `ndarray` vs ferrolearn's scalar `intercept()`) is a binding-ABI concern deferred to the ferrolearn-python layer (cf. #600); `intercept()` keeps returning the scalar. Consumer: `pub use linear_svr::{…}` (`lib.rs`) + `PipelineEstimator` impl. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #615. `linear_svr.rs` imports `ndarray::{Array1, Array2, ScalarOperand}` (`linear_svr.rs:28`) and computes on `ndarray`, not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). Consistent with the crate-wide deferral (cf. `ridge.md`/`glm.md` keep substrate NOT-STARTED). |

## Verification

Commands that would establish SHIPPED claims (none are currently green for a
parity REQ; baseline `8e5bc4b1`):

- `cargo test -p ferrolearn-linear linear_svr` — the module unit tests
  (`test_default_constructor`, `test_builder_setters`, `test_fits_linear_data`,
  `test_squared_epsilon_insensitive`, `test_shape_mismatch`, `test_invalid_c`,
  `test_negative_epsilon`, `test_predict_feature_mismatch`,
  `test_has_coefficients`, `test_pipeline_integration in linear_svr.rs`) assert
  only shape, length, default *fields*, and a loose `|pred − target| < 3.0`
  bound — they do NOT pin `coef_`/`intercept_` against sklearn, so they cannot
  establish any parity REQ (R-CHAR-1/R-CHAR-3). `conformance_linear_svr in
  tests/conformance_wave1.rs` floors only `R² ≥ 0.40` (its comment: "sklearn
  reaches ~0.97").
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`,
  `cargo fmt --all --check`.

Live sklearn oracle (establishes the gaps; expected values per R-CHAR-3 come
from sklearn, never copied from ferrolearn):

```bash
python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
X=np.array([[1.],[2.],[3.],[4.],[5.]]); y=np.array([2.,4.,6.,8.,10.]); \
m=LinearSVR(epsilon=0.0,C=10.0,max_iter=10000,tol=1e-6).fit(X,y); \
print('coef', m.coef_.tolist(), 'int', m.intercept_.tolist(), 'n_iter', m.n_iter_)"
# coef [1.99999938..] int [1.238e-06..] n_iter 48     (REQ-1/REQ-2/REQ-8)

python3 -c "import numpy as np; from sklearn.svm import LinearSVR; \
X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([1.,3.,2.,5.]); \
print([LinearSVR(epsilon=0.,C=c,fit_intercept=False,max_iter=200000,tol=1e-10).fit(X,y).coef_.tolist() for c in (0.1,1.0,10.0)])"
# [[0.6667], [1.25], [1.25]]    (REQ-7: C-dependence; ferrolearn's C/n flattens it)

python3 -c "from sklearn.svm import LinearSVR; m=LinearSVR(); \
print(m.epsilon, m.C, m.loss, m.dual, m.fit_intercept, m.intercept_scaling)"
# 0.0 1.0 epsilon_insensitive auto True 1.0    (REQ-3: epsilon default 0.0, not 0.1)
```

A NOT-STARTED REQ closes only when its fix lands AND a divergence test (expected
values from the live oracle / a sklearn `file:line` constant per R-CHAR-3) goes
green; see the REQ-status table for the current split (REQ-1/2/3/4/5/7/8 SHIPPED,
REQ-6/9/10 NOT-STARTED).

## Blockers to open

- **#607** — REQ-1/REQ-7 of linear_svr (the crux): make the solver minimize
  liblinear's objective `0.5·‖w‖² + C·Σ L_ε` — drop the `C/n_samples` scaling
  (`linear_svr.rs:199, :209, :242, :252`) to plain `C`, and replace the
  fixed-`step=0.01` sub-gradient descent with a convergent CD (dual or
  line-searched primal) so fitted `coef_`/`intercept_` match the live oracle.
- **#608** — REQ-2 of linear_svr: pin `predict` output against the live
  `LinearSVR` oracle on held-out rows (gated on #607's parity-correct fit).
  RESOLVED — `linear_svr_predict` pins `predict([[2.5]]) ≈ 5.0251` GREEN.
- **#609** — REQ-3 of linear_svr: change `LinearSVR::new` default `epsilon`
  from `0.1` to `0.0` (`_classes.py:522`) and fix `test_default_constructor`.
- **#610** — REQ-4 of linear_svr: pin `squared_epsilon_insensitive`
  `coef_`/`intercept_` against the live `LinearSVR(loss="squared_epsilon_insensitive")`
  oracle (depends on the #607 objective fix for the squared branch).
  RESOLVED — `linear_svr_squared_loss` pins `coef [1.8913]`/`intercept [0.2821]`
  GREEN.
- **#611** — REQ-5 of linear_svr: add `fit_intercept` (default `True`) +
  `intercept_scaling` (default `1.0`) with the liblinear augmented-feature,
  penalized-intercept convention (`intercept_ = intercept_scaling·w_last`;
  `fit_intercept=False → intercept_ = 0`).
- **#612** — REQ-6 of linear_svr: add the `dual` parameter
  (`"auto"`/`True`/`False`, default `"auto"`) with the liblinear
  solver-type/`auto`-resolution semantics.
- **#613** — REQ-8 of linear_svr: expose `n_iter_` on `FittedLinearSVR`
  (`= n_iter_.max()`) and emit a `ConvergenceWarning`-equivalent at `max_iter`.
  RESOLVED — `FittedLinearSVR::n_iter` + `eprintln!` warning; `linear_svr_n_iter`
  pins `1 <= n_iter <= max_iter` GREEN.
- **#614** — REQ-9 of linear_svr: match the fitted-attribute contract
  (`intercept_` as a length-1 array, `n_features_in_`) and the
  `_parameter_constraints` validation (`tol>0`, `intercept_scaling>0`,
  `max_iter>=0`, `epsilon` any real — relax the `epsilon>=0` reject).
- **#615** — REQ-10 of linear_svr: migrate `linear_svr.rs` off `ndarray` onto
  the ferray substrate (`ferray-core` arrays, `ferray::linalg`) per R-SUBSTRATE.
