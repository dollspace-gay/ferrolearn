# Linear Support Vector Classification (LinearSVC)

<!--
tier: 3-component
status: draft
baseline-commit: 909727d7
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/svm/_classes.py        # class LinearSVC
  - sklearn/svm/_base.py           # _fit_liblinear, _get_liblinear_solver_type
ferrolearn-module: ferrolearn-linear/src/linear_svc.rs
parity-ops: LinearSVC
crosslink-issue: 617
-->

## Summary

`ferrolearn-linear/src/linear_svc.rs` mirrors scikit-learn's
`sklearn.svm.LinearSVC` (`sklearn/svm/_classes.py:32`), the liblinear-backed
linear support-vector classifier. ferrolearn provides the `LinearSVCLoss` enum
(`Hinge` / `SquaredHinge`), the unfitted `LinearSVC<F>` / fitted
`FittedLinearSVC<F>` builder pair, `Fit`/`Predict`/`HasCoefficients`/`HasClasses`
integrations, a `decision_function`, and a hand-rolled **primal coordinate-Newton**
solver (`fn solve_binary_primal in linear_svc.rs`) called per one-vs-rest
sub-problem (`fn fit in linear_svc.rs`). sklearn fits the **same family of
L2-regularized hinge / squared-hinge objectives** but via **liblinear**
(`_fit_liblinear`, `sklearn/svm/_base.py:1052`) with a specific **C-scaling
convention** (`0.5·‖w‖² + C·Σ loss`, no `1/n` factor) and an **augmented-feature
`intercept_scaling`** intercept convention that **penalizes** the intercept
column.

The solver mechanism (ferrolearn dual CD vs liblinear dual / primal CD) is NOT by
itself a divergence — the objectives are convex, so both reach the same minimizer
*iff the objective they minimize is identical*. ferrolearn now ships the
liblinear-parity core: `fn solve_binary_dual` replaced the old `C/n_samples`
primal coordinate-Newton with liblinear's dual coordinate descent
(`solve_l2r_l1l2_svc`, `linear.cpp:819`) on the summed-loss objective
`0.5·‖w‖² + C·Σ L` (no `1/n`), so `coef_`/`intercept_` track `C` like liblinear
(REQ-1/REQ-10 — the crux, mirroring the LinearSVR rewrite). The intercept is now
the penalized augmented `intercept_scaling` column with
`intercept_ = intercept_scaling·w_last` (REQ-7), and the binary
`decision_function` returns 1-D `(n,)` (`DecisionScores::Binary`, sklearn's
`.ravel()`, REQ-2). These are pinned by `tests/divergence_linear_svc_fit.rs`
against the live oracle (R-CHAR-1/R-CHAR-3). `penalty` (`l1`/`l2`, with the L1
feature-major CD `fn solve_binary_l1r_l2`, `linear.cpp:1467`) and `dual`
(`Auto`/`True`/`False`, the `_get_liblinear_solver_type` solver matrix +
`_validate_dual_parameter` auto-resolution + unsupported-combination rejects) are
now shipped (REQ-5/REQ-8). `class_weight` (`None`/`'balanced'`/dict, scaling `C`
per class via the per-`(cp,cn)` solver generalization) is now shipped (REQ-9).
Still absent from the API: `multi_class` (REQ-6 NOT-STARTED), and the unit is on
the `ndarray` substrate, not ferray (REQ-12).

## Algorithm (sklearn — the contract)

### The liblinear objective (`_classes.py:317-333`, `_base.py:1214-1228`)

`LinearSVC.fit` (`_classes.py:278-346`) calls `check_classification_targets`,
sets `classes_ = np.unique(y)` (`_classes.py:311`), resolves
`dual` via `_validate_dual_parameter` (`_classes.py:13-29, :313`), then calls
`_fit_liblinear` (`_base.py:1052`) with the user's `C`, `penalty`, `loss`,
`fit_intercept`, `intercept_scaling`, `class_weight`, `multi_class`, `dual`,
`tol`, `max_iter`. liblinear minimizes the **L2- (or L1-) regularized
hinge / squared-hinge primal**

```
min_w   R(w)  +  C · Σ_i  L(y_i, w·x_i)
```

with `y_i ∈ {-1, +1}` per one-vs-rest sub-problem, where:

- `R(w) = 0.5·‖w‖²` for `penalty='l2'` (default); `R(w) = ‖w‖_1` for
  `penalty='l1'`.
- `L = max(0, 1 − y·f)` (`loss='hinge'`) or `max(0, 1 − y·f)²`
  (`loss='squared_hinge'`, default).

The load-bearing fact for `coef_`/`intercept_` parity: **the data term is scaled
by `C`, NOT by `C/n_samples`.** There is no `1/n` averaging — the regularizer is
weighed against the *summed* loss, so the optimum depends on `C` (live oracle,
binary 8×2 well-separated set, `fit_intercept=False`,
`loss='squared_hinge'`: `C=0.01 → coef [0.04321, 0.04321]`,
`C=0.1 → [0.04643, …]`, `C=1.0 → [0.04678, …]`, `C=10.0 → [0.04682, …]` — a
genuine, saturating `C`-dependence).

### `_get_liblinear_solver_type` (penalty × loss × dual, `_base.py:995-1049`)

The solver "magic number" is selected from
(`_base.py:1011-1018`):

| loss | penalty | dual=False | dual=True |
|---|---|---|---|
| `hinge` | `l2` | — | 3 |
| `squared_hinge` | `l1` | 5 | — |
| `squared_hinge` | `l2` | 2 | 1 |
| `crammer_singer` | (any) | 4 | 4 |

So `hinge` requires `dual=True` (`l2` only); `penalty='l1'` requires
`squared_hinge` + `dual=False` (the `('l1','hinge')` combination is rejected,
`_classes.py:59-60`). `dual='auto'` (default) resolves to `True` when
`n_samples < n_features` and a dual solver exists, else `False`
(`_classes.py:13-29`). The *optimum* is `dual`-invariant; `dual` is an API
parameter (R-DEV-2) and a solver-path choice, not a result divergence.

### `fit_intercept` + `intercept_scaling` (`_base.py:1188-1245`)

When `fit_intercept` (default `True`), liblinear augments each instance with a
synthetic constant feature `bias = intercept_scaling` (`_base.py:1189-1198`):
`x → [x_1, …, x_n, intercept_scaling]`. liblinear **penalizes** this column like
any other. After training the raw weight matrix is split
(`_base.py:1240-1245`):

```
coef_      = raw_coef_[:, :-1]
intercept_ = intercept_scaling * raw_coef_[:, -1]
```

so a larger `intercept_scaling` reduces the relative regularization on the
intercept. With `fit_intercept=False`, `intercept_ = 0.0`. `intercept_scaling`
must be `> 0` when `fit_intercept` (`_base.py:1191-1196`, `ValueError`).

### `class_weight` (`_base.py:1179`, `_classes.py:118-124`)

`class_weight` (`None` / `'balanced'` / dict) multiplies `C` per class
(`compute_class_weight`, `_base.py:1179`): the effective penalty for class `i`
is `class_weight[i]·C`. `'balanced'` uses
`n_samples / (n_classes · bincount(y))`. (On a class-balanced set `'balanced'`
equals `None`, verified live.)

### Classification head — `decision_function` / `predict` / `classes_`
(`LinearClassifierMixin`, `sklearn/linear_model/_base.py`)

`decision_function(X) = X @ coef_.T + intercept_`. For binary
(`n_classes == 2`) liblinear fits **one** weight row and `decision_function`
returns shape `(n_samples,)` (1-D, the `.ravel()` of the single column);
`predict` is `classes_[(scores > 0).astype(int)]` (sign convention, the `>0`
class is `classes_[1]`). For multiclass OvR, `coef_` is `(n_classes, n_features)`,
`intercept_` is `(n_classes,)`, `decision_function` is `(n_samples, n_classes)`,
and `predict` is the argmax across classes. `classes_ = np.unique(y)` (sorted).
Live oracle (binary 8×2 set, `squared_hinge`, `C=1.0`):
`coef_ [[0.12835, 0.12835]]` shape `(1, 2)`, `intercept_ [-1.19438]` shape
`(1,)`, `decision_function(X).shape == (8,)`. Multiclass 9×2 3-class set,
`C=10.0`: `coef_` `(3, 2)`, `intercept_` `(3,)`, `decision_function` `(9, 3)`.

### `n_iter_` / `n_features_in_` / param validation (`_classes.py:233-246, :338`)

`n_iter_ = n_iter_.max().item()` (max iterations across the OvR fits,
`_classes.py:338`). `n_features_in_` is set by `_validate_data`
(`_classes.py:302`). `_parameter_constraints` (`_classes.py:233-246`) requires
`C ∈ (0, ∞)`, `tol ∈ (0, ∞)`, `max_iter ∈ [0, ∞)`, `penalty ∈ {l1, l2}`,
`loss ∈ {hinge, squared_hinge}`, `multi_class ∈ {ovr, crammer_singer}`,
`intercept_scaling ∈ (0, ∞)`, `class_weight ∈ {None, dict, 'balanced'}`.

### Solver / optimum equivalence (the parity criterion)

liblinear's CD and a correct primal coordinate-Newton both minimize the single
convex objective above and converge to the **same** `coef_`/`intercept_` (modulo
`tol`/`max_iter`). ferrolearn's choice of primal coordinate-Newton is sanctioned
under R-DEV-7 *iff* it minimizes the identical objective. Parity is verified by
comparing fitted `coef_`/`intercept_`/`decision_function` to the live sklearn
oracle — NOT solver trajectories. A C-scaling mismatch (REQ-1/REQ-10) shifts the
optimum and breaks parity even though the solver "converges."

## ferrolearn (what exists)

`fn solve_binary_dual in linear_svc.rs` runs liblinear's dual coordinate descent
(`solve_l2r_l1l2_svc`, `linear.cpp:819`) on the summed-loss objective
`0.5·‖w‖² + C·Σ L` (no `1/n`): it initializes `alpha = 0`, `w = 0`, precomputes
`QD[i] = diag + ‖x_i‖²`, and per coordinate computes `G = y_i·(w·x_i) − 1 +
diag·alpha_i`, the projected gradient + active-set shrinking, and the box update
`alpha_i ← clamp(alpha_i − G/QD[i], 0, U)` with `w += (Δalpha_i)·y_i·x_i`.
**hinge** (`L2R_L1LOSS_SVC_DUAL`) sets `diag = 0`, `U = C`; **squared_hinge**
(`L2R_L2LOSS_SVC_DUAL`) sets `diag = 0.5/C`, `U = +∞` (`linear.cpp:849-858`). It
stops when `PGmax_new − PGmin_new ≤ tol` on the full set (`linear.cpp:972-990`).
The solver is parameterized by a small `SolverConfig<F>` (groups `c`/`max_iter`/
`tol`/`loss`/`fit_intercept`/`intercept_scaling`).

When `fit_intercept`, the design matrix is augmented with a synthetic constant
column = `intercept_scaling`, **penalized** in `‖w‖²` like any feature
(`QD[i] += intercept_scaling²`, `w_last += d·intercept_scaling`); `fn fit`
extracts `coef_ = w[:n_features]`, `intercept_ = intercept_scaling·w_last`
(`_base.py:1240-1245`), or `intercept_ = 0` when `fit_intercept=false`.

`fn fit in linear_svc.rs` validates `n_samples == y.len()`, `C > 0`, and
`intercept_scaling > 0` (when `fit_intercept`), computes `classes` by sort+dedup,
errors with `InsufficientSamples` for `< 2` classes, then: binary
(`classes.len() == 2`) maps `y` to `±1` (positive = `classes[1]`,
`linear.cpp:861-871`) and solves once; multiclass loops **one-vs-rest** over each
class. It emits the `ConvergenceWarning`-equivalent (`eprintln!`) when any
sub-problem hits `max_iter` without converging. `FittedLinearSVC<F>` stores
`weight_vectors: Vec<Array1<F>>`, `intercepts: Vec<F>`, `classes: Vec<usize>`,
`is_binary: bool`, `n_features`. `fn decision_function in linear_svc.rs` returns
`DecisionScores::Binary` (1-D `Array1`, sklearn's `.ravel()`, `_base.py:365`) for
binary, `DecisionScores::Multiclass` `(n, n_classes)` for multiclass.
`fn predict in linear_svc.rs` is `sign(Xw+b)` (binary, `>= 0 → classes[1]`) /
argmax (multiclass). `HasCoefficients` returns the *first* weight vector /
intercept; `HasClasses` returns `classes`.

The constructor `fn LinearSVC::new` defaults `C=1.0`, `max_iter=1000`, `tol=1e-4`,
`loss=SquaredHinge`, `penalty=L2`, `dual=Auto`, `fit_intercept=true`,
`intercept_scaling=1.0` (matching sklearn's defaults). `fn fit` resolves
`dual` (`fn resolve_dual`, `_classes.py:13-29`) and validates the
penalty×loss×dual combination against the liblinear solver matrix
(`fn liblinear_solver_type`, `_base.py:995-1018`) before dispatching per
sub-problem to `fn solve_binary_l1r_l2` (`penalty=l1`) or `fn solve_binary_dual`
(`penalty=l2`). `class_weight` (`None`/`Balanced`/`Explicit`, default `None`) is now a field +
`fn with_class_weight in linear_svc.rs`; `fn compute_class_weight in
linear_svc.rs` expands per-class weights (`class_weight.py:63-81`), and `fn fit`
threads the per-sub-problem `(cp, cn)` (binary `C·w[+]`/`C·w[−]`; OvR
`C·w[k]`/base `C`) into the per-sample-`C` solvers. There is **no** `multi_class`,
`random_state`, or `verbose` field/accessor.
`LinearSVC`/`FittedLinearSVC`/`LinearSVCLoss` are boundary types re-exported at
the crate root (`pub use linear_svc::{FittedLinearSVC, LinearSVC, LinearSVCLoss}
in lib.rs`) and the unfitted `LinearSVC` drives the `RsLinearSVC` PyO3 binding
(`py_classifier!(RsLinearSVC, …) in ferrolearn-python/src/extras.rs`, registered
`m.add_class::<extras::RsLinearSVC>() in ferrolearn-python/src/lib.rs`) — both
non-test production consumers.

## Requirements

- REQ-1: Fit parity (the crux) — fitted `coef_`/`intercept_` match the live
  `LinearSVC` (liblinear) oracle for `loss='squared_hinge'`, which requires
  minimizing `0.5·‖w‖² + C·Σ max(0,1−yf)²` with **C-scaling (no `1/n`)** and the
  penalized augmented intercept.
- REQ-2: `decision_function` — `X @ coef_.T + intercept_`; binary shape
  `(n_samples,)` (not `(n_samples, 1)`), multiclass `(n_samples, n_classes)`,
  values matching the oracle.
- REQ-3: `predict` + `classes_` — binary `predict` = sign convention mapped to
  `classes_` (`>0 → classes_[1]`), multiclass = argmax; `classes_ = np.unique(y)`
  sorted, matching the oracle's labels and accuracy.
- REQ-4: `loss` parameter — `{hinge, squared_hinge (default)}`; fitted
  `coef_`/`intercept_` match the oracle per loss (the `hinge` branch must solve
  the actual hinge optimum, not a squared-hinge majorant proxy).
- REQ-5: `penalty` parameter — `{l1, l2 (default)}`; `l1` produces sparse
  `coef_` matching the oracle; `('l1','hinge')` rejected per `_classes.py:59-60`.
- REQ-6: `multi_class` parameter — `{ovr (default), crammer_singer}`; ferrolearn
  has OvR but no `crammer_singer`, and OvR per-class `coef_` must match the
  oracle (not just argmax accuracy).
- REQ-7: `fit_intercept` + `intercept_scaling` — augmented-feature, **penalized**
  intercept (`x → [x, intercept_scaling]`, `intercept_ = intercept_scaling·w_last`),
  `fit_intercept=False → intercept_ = 0`; defaults `True`/`1.0`;
  `intercept_scaling > 0` validated.
- REQ-8: `dual` parameter — `{'auto' (default), True, False}` with the liblinear
  solver-type selection / `auto`-resolution and the unsupported-combination
  rejects (`hinge`+`dual=False`, `l1`+`dual=True`).
- REQ-9: `class_weight` parameter — `{None (default), 'balanced', dict}` scaling
  `C` per class (`class_weight[i]·C`) matching the oracle on imbalanced data.
- REQ-10: C-scaling convention — the data term is scaled by `C` (summed loss),
  not `C/n_samples`; the optimum is `n_samples`-dependent through `C`.
- REQ-11: `n_iter_`/`n_features_in_` + param validation — expose `n_iter_`
  (`= n_iter_.max()`, emit `ConvergenceWarning`-equivalent at `max_iter`) and
  `n_features_in_`; validate `tol > 0`, `max_iter ≥ 0` per
  `_parameter_constraints` (`_classes.py:233-246`).
- REQ-12: ferray substrate migration (array type → `ferray-core`; linear algebra
  → `ferray::linalg`) per R-SUBSTRATE.

## Acceptance criteria

- AC-1 (REQ-1/REQ-10): on the binary 8×2 well-separated set
  (`X=[[1,1],[1,2],[2,1],[2,2],[8,8],[8,9],[9,8],[9,9]]`, `y=[0,0,0,0,1,1,1,1]`),
  `LinearSVC(C=1.0, loss='squared_hinge', max_iter=100000, tol=1e-10)` fitted
  `coef_` matches the live oracle `[[0.12835, 0.12835]]`, `intercept_`
  `[-1.19438]` within tolerance; and with `fit_intercept=False` the `coef_`
  tracks `C` (`0.01→0.04321`, `0.1→0.04643`, `1.0→0.04678`, `10.0→0.04682`)
  proving C-scaling (not `C/n`, which flattens the dependence).
- AC-2 (REQ-2): binary `decision_function(X)` has shape `(8,)` and equals
  `X @ coef_ + intercept_` matching the oracle; multiclass `(9, 3)`.
- AC-3 (REQ-3): `predict` equals the oracle's `predict` on the fixed sets;
  `classes()` equals `np.unique(y)` sorted.
- AC-4 (REQ-4): `loss='hinge'` fitted `coef_`/`intercept_` match the live
  `LinearSVC(loss='hinge')` oracle (`coef [[0.15385, 0.15385]]`,
  `intercept [-1.46154]` on the 8×2 set).
- AC-5 (REQ-5): `penalty='l1', loss='squared_hinge', dual=False` reproduces the
  oracle `coef_` (`[[0.12832, 0.12832]]` on the 8×2 set); `('l1','hinge')` is
  rejected.
- AC-6 (REQ-6): multiclass OvR per-class `coef_`/`intercept_` (shape
  `(n_classes, n_features)`/`(n_classes,)`) match the oracle row-by-row;
  `multi_class='crammer_singer'` is exposed (or pinned NOT-STARTED).
- AC-7 (REQ-7): `fit_intercept=True, intercept_scaling=s` reproduces the oracle
  `intercept_ = s·w_last` (penalized column); `fit_intercept=False` gives
  `intercept_ == 0.0`; `intercept_scaling <= 0` raises.
- AC-8 (REQ-8): `dual='auto'` resolves to the oracle's solver/optimum; the
  parameter exists with default `'auto'`; `hinge`+`dual=False` and
  `l1`+`dual=True` are rejected.
- AC-9 (REQ-9): `class_weight='balanced'`/dict reproduces the oracle `coef_` on
  an imbalanced set.
- AC-10 (REQ-11): `n_iter_` exposed (`= max` across OvR fits); `n_features_in_`
  exposed; `tol <= 0` raises; `max_iter < 0` documented (usize).
- AC-11 (REQ-12): `linear_svc.rs` owns its computation on `ferray-core` arrays /
  `ferray::linalg`, not `ndarray`.

## REQ status

Binary classification (R-DEFER-2): SHIPPED = impl + non-test production consumer
+ tests + green oracle verification; NOT-STARTED = concrete open blocker
referenced by `#`-number. `LinearSVC`/`FittedLinearSVC`/`LinearSVCLoss` are
boundary estimator types re-exported at the crate root (`pub use linear_svc::{…}
in lib.rs`) and drive the `RsLinearSVC` PyO3 binding (`py_classifier!(RsLinearSVC,
…) in extras.rs`, registered in `ferrolearn-python/src/lib.rs`) — under
S5/R-DEFER-1 the public estimator type IS the consumer surface (grandfathered),
and the Python binding is a live non-test consumer.

The crux (REQ-1/REQ-10) is now SHIPPED: `fn solve_binary_dual in linear_svc.rs`
replaced the old `c / n_f` primal coordinate-Newton with liblinear's dual
coordinate descent (`solve_l2r_l1l2_svc`, `linear.cpp:819`), minimizing
liblinear's `0.5·‖w‖² + C·Σ L` (no `1/n`), so `coef_`/`intercept_` track `C`
like liblinear. `fn fit` adds the penalized augmented `intercept_scaling`
column and extracts `coef_ = w[:n_features]`, `intercept_ = intercept_scaling·
w_last` (`_base.py:1240-1245`), shipping REQ-7. `fn decision_function` returns a
1-D `DecisionScores::Binary` for the binary case (sklearn's `.ravel()`,
`linear_model/_base.py:365`), shipping REQ-2's shape. These are pinned against
the live oracle by `tests/divergence_linear_svc_fit.rs::{linear_svc_coef_parity,
linear_svc_coef_c_dependence, linear_svc_decision_function}` (R-CHAR-1/R-CHAR-3).
REQ-3 (`predict`/`classes_`) is incidentally covered (the sign/argmax labels are
now downstream of the parity fit; a dedicated `predict` oracle pin pends #620).
REQ-5 (`penalty`) and REQ-8 (`dual`) are now SHIPPED: `penalty=l1` ships the
genuinely-different sparse L1 optimum via `fn solve_binary_l1r_l2`
(`solve_l1r_l2_svc`, `linear.cpp:1467`, solver type 5), `dual` ships the
`_get_liblinear_solver_type` solver matrix + `_validate_dual_parameter`
auto-resolution + the unsupported-combination rejects, and R-DEV-7
dual-invariance keeps the l2 dual CD for `penalty=l2` regardless of the resolved
`dual`. REQ-9 (`class_weight`) is now SHIPPED: `fn compute_class_weight`
(mirroring `sklearn.utils.compute_class_weight`, `_base.py:1179`) feeds the
per-class `C` scaling into a generalized `SolverConfig<F>` carrying `(cp, cn)` —
`fn solve_binary_dual`/`fn solve_binary_l1r_l2` now apply the per-sample
`C_[i] = (y_i>0?cp:cn)` (`linear.cpp:843-858`,`:1504-1509`), with `fn fit` wiring
the binary `(C·w[+], C·w[−])` and OvR `(C·w[k], base C)` penalties
(`linear.cpp:2543-2571`). The module's API still lacks `multi_class` (REQ-6
NOT-STARTED, its blocker), and the unit remains on `ndarray` (REQ-12).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (fit parity — coef_/intercept_ vs liblinear oracle) | SHIPPED | `fn solve_binary_dual in linear_svc.rs` implements liblinear's dual coordinate descent (`solve_l2r_l1l2_svc`, `linear.cpp:819`) minimizing `0.5·‖w‖² + C·Σ L` (no `1/n`); `fn fit` maps `classes_[1]→+1` (`linear.cpp:861-871`) and extracts `coef_ = w[:n_features]`, `intercept_ = intercept_scaling·w_last` (`_base.py:1240-1245`). Pinned by `linear_svc_coef_parity in tests/divergence_linear_svc_fit.rs` (live oracle 8×2 set, `squared_hinge`, `C=1.0`, `fit_intercept=True`: `coef_ [[0.12835213611984458, 0.12835213611984475]]`, `intercept_ [-1.1943776585907158]`, tol 1e-2). Consumer: `pub use linear_svc::{…} in lib.rs` + `RsLinearSVC` PyO3 binding. |
| REQ-2 (decision_function shape `(n,)` + values) | SHIPPED | `fn decision_function in linear_svc.rs` returns `DecisionScores::Binary` (a 1-D `Array1` = `X·w + b`) for the binary case — sklearn ravels the single column to `(n_samples,)` (`return xp.reshape(scores, (-1,)) if scores.shape[1] == 1 else scores`, `linear_model/_base.py:365`) — and `DecisionScores::Multiclass` `(n, n_classes)` otherwise. Pinned by `linear_svc_decision_function in tests/divergence_linear_svc_fit.rs` (live oracle 1-D `(8,)` values, tol 1e-2; the test asserts `as_binary().len() == 8`). Consumer: `fn predict` reads the binary sign; `api_proof_linear_svc` asserts `as_binary().is_some()`. |
| REQ-3 (predict + classes_) | SHIPPED | `fn predict in linear_svc.rs` uses the sign convention (`>= 0 → classes[1]`) / argmax, and `HasClasses::classes` returns sorted unique labels (`classes_ = np.unique(y)`, `_classes.py:311`). The predicted labels are downstream of the liblinear-parity fit (REQ-1) and are pinned against the live oracle by `linear_svc_predict_parity in tests/divergence_linear_svc_fit.rs` (#620; 8×2 set: `predict [0,0,0,0,1,1,1,1]`, `classes_ [0,1]`). |
| REQ-4 (loss {hinge, squared_hinge}) | SHIPPED | `fn solve_binary_dual` solves BOTH the true `hinge` (`U = C`, `diag = 0`) and `squared_hinge` (`U = ∞`, `diag = 0.5/C`) optima (`linear.cpp:849-858`), so the `Hinge` branch is the actual hinge optimum, not a smooth-majorant approximation. Pinned against the live oracle by `linear_svc_hinge_coef_parity in tests/divergence_linear_svc_fit.rs` (#621; live `hinge` 8×2: `coef [[0.15384615383852776, 0.15384615383915584]]`, `intercept [-1.4615384615168394]`). |
| REQ-5 (penalty {l1, l2}) | SHIPPED | `LinearSVC<F>` exposes `pub penalty: LinearSVCPenalty` (default `L2`) + `fn with_penalty in linear_svc.rs`. `penalty=l1` routes (in `fn fit`'s `solve_one`) to `fn solve_binary_l1r_l2 in linear_svc.rs` — liblinear's feature-major coordinate descent (`solve_l1r_l2_svc`, `linear.cpp:1467`, solver type 5, `_base.py:1014`) minimizing `‖w‖₁ + C·Σ max(0,1−yf)²` over the augmented (penalized-intercept) weights, giving sparse `coef_`; `penalty=l2` keeps `fn solve_binary_dual`. `fn liblinear_solver_type in linear_svc.rs` rejects `('l1','hinge')` (no `l1` under `hinge`, `_base.py:1013`). Pinned by `test_l1_penalty_smoke in linear_svc.rs` (live oracle 8×2 `l1,squared_hinge,dual=False,C=1.0,fit_intercept=True`: `coef_ [[0.1283185834966579, 0.12831858464059265]]`, `intercept_ [-1.2079646017762715]`; ferrolearn's natural-order CD lands within ~1.2e-9) + `test_unsupported_combinations_rejected in linear_svc.rs`. The rigorous oracle pin in `tests/divergence_linear_svc_fit.rs` is the critic's next step. ferrolearn sweeps natural order (no `bounded_rand_int` shuffle, `linear.cpp:1535`); the l1 optimum is unique so the limit is identical (documented RNG-path boundary). Consumer: `pub use linear_svc::{…} in lib.rs` + `RsLinearSVC` PyO3 binding. |
| REQ-6 (multi_class {ovr, crammer_singer}) | NOT-STARTED | open prereq blocker #623. `fn fit in linear_svc.rs` implements one-vs-rest (a binary solve per class) — structurally sklearn's default `'ovr'` — but there is no `multi_class` field and no `crammer_singer` joint solver (`_base.py:1017` solver type 4); and the OvR per-class `coef_` is downstream of the REQ-1 C/n bug, so the per-class rows do not match the oracle (`coef_` shape `(3, 2)` on the 9×2 3-class set). No per-class `coef_` test vs the oracle. |
| REQ-7 (fit_intercept + intercept_scaling) | SHIPPED | `LinearSVC<F>` exposes `pub fit_intercept: bool` (default `true`) + `pub intercept_scaling: F` (default `1.0`) + `#[must_use]` `with_fit_intercept`/`with_intercept_scaling`. `fn solve_binary_dual` augments the design matrix with a synthetic constant column = `intercept_scaling`, penalized in `‖w‖²` like any feature (`QD[i] += intercept_scaling²`, `w_last += d·intercept_scaling`); `fn fit` sets `intercept_ = intercept_scaling·w_last` (`_base.py:1188-1198, :1240-1245`) and `intercept_ = 0` when `fit_intercept=false`, and rejects `intercept_scaling <= 0` with `fit_intercept` (`FerroError::InvalidParameter`, `_base.py:1190-1196`). Pinned by `linear_svc_coef_parity` (`fit_intercept=True`) + module `test_fit_intercept_false_zero_intercept`/`test_invalid_intercept_scaling`. |
| REQ-8 (dual param) | SHIPPED | `LinearSVC<F>` exposes `pub dual: DualMode` (`Auto`/`True`/`False`, default `Auto`) + `fn with_dual in linear_svc.rs`. `fn resolve_dual in linear_svc.rs` mirrors `_validate_dual_parameter` (`_classes.py:13-29`): for `Auto`, `n_samples < n_features` prefers dual (fall back to primal), else prefers primal (fall back to dual) — resolution is checked against `fn liblinear_solver_type in linear_svc.rs` (the `_get_liblinear_solver_type` matrix, `_base.py:995-1018`) so it is automatically consistent. `fn fit` calls `liblinear_solver_type(penalty, loss, dual)?` to validate the resolved combination, rejecting `hinge+dual=false`, `l1+dual=true`, and `l1+hinge` with `FerroError::InvalidParameter` (mirroring sklearn's `ValueError` strings, `_base.py:1033-1043`). **R-DEV-7**: the resolved `dual` is observably immaterial for `penalty=l2` — the l2 dual CD (`solve_l2r_l1l2_svc`) and the l2 primal minimize the same strongly convex `0.5·‖w‖² + C·Σ L` and reach the same `coef_`/`intercept_`, so `penalty=l2` keeps `fn solve_binary_dual` regardless of the resolved `dual` (observable contract preserved; implementation may differ). `dual` is load-bearing only for the rejects and for selecting the genuinely different `l1` primal solver (REQ-5). Pinned by `test_unsupported_combinations_rejected in linear_svc.rs` (the three rejects) + `test_dual_auto_resolution in linear_svc.rs` (auto falls back to dual=true for hinge+l2). Consumer: `pub use linear_svc::{…} in lib.rs` + `RsLinearSVC` PyO3 binding. |
| REQ-9 (class_weight) | SHIPPED | `LinearSVC<F>` exposes `pub class_weight: ClassWeight<F>` (`None`/`Balanced`/`Explicit`, default `None`) + `fn with_class_weight in linear_svc.rs`. `fn compute_class_weight in linear_svc.rs` mirrors `sklearn.utils.compute_class_weight` (`class_weight.py:63-81`) exactly (as `_fit_liblinear` calls it, `compute_class_weight(class_weight, classes=classes_, y=y)`, `_base.py:1179`): `None → 1.0`, `Balanced → n_samples/(n_classes·count_c)`, `Explicit → 1.0 default overridden by map`. The solver config `SolverConfig<F>` now carries `(cp, cn)` instead of a scalar `c`; `fn solve_binary_dual`/`fn solve_binary_l1r_l2` apply the per-sample penalty `C_[i] = (y_i>0 ? cp : cn)` (per-sample `diag[i]`/`upper_bound[i]`/`C[i]`, `linear.cpp:843-858`, `:1504-1509`). `fn fit` scales `C` per class: BINARY `cp = C·weights[idx(classes[1])]`, `cn = C·weights[idx(classes[0])]` (`train_one(Cp=weighted_C[1], Cn=weighted_C[0])`, `linear.cpp:2543-2551`); OvR class `k` `cp = C·weights[k]`, `cn = C` (the base `C` — the negative rest is UNWEIGHTED, `linear.cpp:2559-2571`). When `cp == cn` (the default `None`) the per-sample math collapses to the prior global `diag`/`U`, so the 9 existing divergence pins stay green (REQ-1/4/5/8/10/11 unaffected). Pinned by `test_class_weight_smoke in linear_svc.rs` (live oracle 8×2 imbalanced set, `squared_hinge, dual=True, C=1.0, fit_intercept=True`: `None → coef [[0.10056,0.15957]], int [-1.26346]`; `balanced → coef [[0.09937,0.16666]], int [-1.21320]`, weights `[0.6667,2.0]`; `{0:1,1:5} → coef [[0.11059,0.17164]], int [-1.29547]`; ferrolearn within 1e-2). The rigorous oracle pin in `tests/divergence_linear_svc_fit.rs` is the critic's next step. Consumer: `pub use linear_svc::{…} in lib.rs` + `RsLinearSVC` PyO3 binding. |
| REQ-10 (C-scaling convention) | SHIPPED | the `c / n_f` division is removed; `fn solve_binary_dual` uses `upper_bound = C` (hinge) / `diag = 0.5/C` (squared_hinge) with the summed-loss objective, so `coef_` tracks `C` like liblinear (no `1/n`). Pinned by `linear_svc_coef_c_dependence in tests/divergence_linear_svc_fit.rs` (live oracle, 8×2 set, `squared_hinge`, `fit_intercept=True`: `C=0.1 → coef[0] 0.0784651864625997`, `C=1.0 → 0.12835213611984458`; the bug flattened this spread). |
| REQ-11 (n_iter_/n_features_in_ + param validation) | SHIPPED | `FittedLinearSVC<F>` exposes `fn n_features_in in linear_svc.rs` (returns the stored `n_features`, set by `_validate_data`, `_classes.py:302`) and `fn n_iter in linear_svc.rs` (the max dual-CD outer-iteration count threaded from `solve_binary_dual` across the binary/OvR fits, `n_iter_ = n_iter_.max().item()`, `_classes.py:338`); `fn fit in linear_svc.rs` now validates `tol > 0` (mirroring the `C`/`intercept_scaling` checks; `Interval(Real, 0.0, None, closed="neither")`, `_classes.py:237`) alongside the `ConvergenceWarning`-equivalent (`eprintln!`, `_base.py:1234-1238`). Pinned by `linear_svc_attrs_and_tol_validation in tests/divergence_linear_svc_fit.rs` (#627). `n_features_in_` (oracle `2`) and the `tol <= 0` reject are EXACT; `n_iter_` is the documented shuffle-path RNG boundary (ferrolearn sweeps natural order, sklearn's liblinear shuffles `index` each sweep — cf. the SGD boundary), so the pin bounds `n_iter` in `[1, max_iter]` rather than exact-matching. |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #628. `linear_svc.rs` imports `ndarray::{Array1, Array2, ScalarOperand}` (`use ndarray::… in linear_svc.rs`) and computes on `ndarray`, not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). Consistent with the crate-wide deferral (cf. `linear_svr.md`/`ridge.md`/`glm.md` keep substrate NOT-STARTED). |

## Verification

Module unit tests + the conformance/api-proof harness exist but pin no parity
value, so no numerical REQ can be SHIPPED (R-CHAR-1/R-CHAR-3):

- `cargo test -p ferrolearn-linear linear_svc` — the module unit tests
  (`test_default_constructor`, `test_builder_setters`, `test_binary_classification`,
  `test_binary_hinge_loss`, `test_multiclass_classification`,
  `test_shape_mismatch_fit`, `test_invalid_c`, `test_single_class_error`,
  `test_has_coefficients`, `test_predict_feature_mismatch` in `linear_svc.rs`)
  assert only default *fields*, error paths, `coef_` length, and loose accuracy
  floors (`correct >= 6/8`, `>= 4/6`, `>= 7/9`) — they do NOT pin
  `coef_`/`intercept_`/`decision_function` against sklearn.
  `conformance_linear_svc in tests/conformance_wave1.rs` floors only prediction
  accuracy ≥ 0.90; `api_proof_linear_svc in tests/api_proof.rs` checks only
  `decision_function` nrows + that `fit`/`predict`/`score` run.
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`,
  `cargo fmt --all --check`.

Live sklearn oracle (establishes the gaps; expected values per R-CHAR-3 come
from sklearn, never copied from ferrolearn):

```bash
# REQ-1/REQ-2/REQ-3: binary fit + decision_function shape (1-D)
python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
y=np.array([0,0,0,0,1,1,1,1]); \
m=LinearSVC(C=1.0,loss='squared_hinge',max_iter=100000,tol=1e-10).fit(X,y); \
print('coef',m.coef_.tolist(),'int',m.intercept_.tolist(),'df.shape',m.decision_function(X).shape)"
# coef [[0.12835, 0.12835]] int [-1.19438] df.shape (8,)   (ferrolearn df is (8,1))

# REQ-10: C-scaling (fit_intercept=False isolates the regularizer)
python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
y=np.array([0,0,0,0,1,1,1,1]); \
print([round(LinearSVC(C=c,loss='squared_hinge',fit_intercept=False,max_iter=200000,tol=1e-12).fit(X,y).coef_[0,0],5) for c in (0.01,0.1,1.0,10.0)])"
# [0.04321, 0.04643, 0.04678, 0.04682]   (C/n in ferrolearn flattens this)

# REQ-4: hinge loss optimum
python3 -c "import numpy as np; from sklearn.svm import LinearSVC; \
X=np.array([[1.,1.],[1.,2.],[2.,1.],[2.,2.],[8.,8.],[8.,9.],[9.,8.],[9.,9.]]); \
y=np.array([0,0,0,0,1,1,1,1]); \
m=LinearSVC(loss='hinge',max_iter=100000,tol=1e-10).fit(X,y); \
print('coef',m.coef_.tolist(),'int',m.intercept_.tolist())"
# coef [[0.15385, 0.15385]] int [-1.46154]

# REQ-5: l1 penalty (sparse), REQ-8 defaults
python3 -c "from sklearn.svm import LinearSVC; m=LinearSVC(); \
print(m.penalty,m.loss,m.dual,m.multi_class,m.fit_intercept,m.intercept_scaling,m.class_weight)"
# l2 squared_hinge auto ovr True 1 None
```

A NOT-STARTED REQ closes only when its fix lands AND a divergence test (expected
values from the live oracle / a sklearn `file:line` constant per R-CHAR-3) goes
green; all twelve REQs are currently NOT-STARTED.

## Blockers to open

- **#618** — REQ-1/REQ-10 of linear_svc (the crux): make `solve_binary_primal`
  minimize liblinear's objective `0.5·‖w‖² + C·Σ loss` — drop the `c / n_f`
  scaling (the squared-hinge gradient/Hessian and hinge subgradient terms) to
  plain `C`, and fit the intercept via the penalized augmented column (not the
  separate un-regularized step), so fitted `coef_`/`intercept_` match the live
  `LinearSVC` oracle.
- **#619** — REQ-2 of linear_svc: return binary `decision_function` as 1-D
  `(n_samples,)` (matching sklearn's `.ravel()`) and pin its values against the
  live oracle (gated on #618's parity-correct fit).
- **#620** — REQ-3 of linear_svc: pin `predict` + `classes_` against the live
  `LinearSVC` oracle (labels, not just an accuracy floor; gated on #618).
- **#621** — REQ-4 of linear_svc: solve the true `hinge` optimum (not a
  squared-hinge majorant) and pin per-loss `coef_`/`intercept_` against the live
  oracle (`hinge` → `coef [[0.15385,…]]`, `intercept [-1.46154]`).
- **#622** — REQ-5 of linear_svc: add the `penalty` parameter (`l1`/`l2`,
  default `l2`) with the L1 solver (sparse `coef_`, solver type 5) and the
  `('l1','hinge')` reject.
- **#623** — REQ-6 of linear_svc: add the `multi_class` parameter
  (`ovr` default / `crammer_singer`), the `crammer_singer` joint solver, and pin
  OvR per-class `coef_` against the oracle.
- **#624** — REQ-7 of linear_svc: add `fit_intercept` (default `True`) +
  `intercept_scaling` (default `1.0`) with the liblinear augmented-feature,
  **penalized**-intercept convention (`intercept_ = intercept_scaling·w_last`;
  `fit_intercept=False → intercept_ = 0`; `intercept_scaling > 0` validated).
- **#625** — REQ-8 of linear_svc: add the `dual` parameter
  (`'auto'`/`True`/`False`, default `'auto'`) with the liblinear
  solver-type/`auto`-resolution semantics and the unsupported-combination
  rejects.
- **#626** — REQ-9 of linear_svc: add the `class_weight` parameter
  (`None`/`'balanced'`/dict) scaling `C` per class (`class_weight[i]·C`).
- **#627** — REQ-11 of linear_svc: expose `n_iter_` (`= n_iter_.max()`, emit a
  `ConvergenceWarning`-equivalent at `max_iter`) and `n_features_in_`; validate
  `tol > 0` per `_parameter_constraints` (`_classes.py:237`).
- **#628** — REQ-12 of linear_svc: migrate `linear_svc.rs` off `ndarray` onto
  the ferray substrate (`ferray-core` arrays, `ferray::linalg`) per R-SUBSTRATE.
