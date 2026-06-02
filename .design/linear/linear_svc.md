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

The solver mechanism (ferrolearn primal coordinate-Newton vs liblinear dual /
primal CD) is NOT by itself a divergence — the objectives are convex, so both
reach the same minimizer *iff the objective they minimize is identical*. The
divergences are concrete and measured: ferrolearn scales the loss by
`C/n_samples` rather than `C` (`fn solve_binary_primal` — the `c / n_f` factor),
which shifts the optimum (REQ-1/REQ-10 — the crux, the same C/n bug LinearSVR
carried before its rewrite). ferrolearn's binary `decision_function` returns
shape `(n, 1)` where sklearn returns `(n,)` (REQ-2). The intercept is fit by a
separate **un-regularized** update rather than liblinear's penalized augmented
column, and `fit_intercept`/`intercept_scaling`/`penalty`/`dual`/`multi_class`/
`class_weight`/`n_iter_`/`n_features_in_` are all absent from the API
(REQ-5/6/7/8/9/11). No test pins any `coef_`/`intercept_`/`decision_function`
value against the live sklearn oracle — `conformance_linear_svc` floors only
prediction accuracy ≥ 0.90 and `api_proof_linear_svc` checks only shapes — so the
numerical-parity REQs cannot be SHIPPED (R-HONEST-3). The unit is on the
`ndarray` substrate, not ferray (REQ-12).

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

`fn solve_binary_primal in linear_svc.rs` runs a primal coordinate-Newton
descent: it initializes `w = 0`, `b = 0`, maintains `decision = Xw + b`
incrementally, and for each coordinate `j` accumulates the regularizer gradient
`w[j]` (+ Hessian `1`) plus, for each sample with `margin = y·decision < 1`,
the data-term gradient/Hessian — but **scaled by `c / n_f`** where
`n_f = n_samples`: squared-hinge adds `−2·(C/n)·(1−margin)·y·x` to the gradient
and `2·(C/n)·x²` to the Hessian; hinge uses the subgradient `−(C/n)·y·x` with the
squared-hinge Hessian as a smooth majorant. It applies the Newton step
`w[j] −= grad/hess`. The intercept is updated by a **separate, un-regularized**
coordinate-Newton step (no augmented penalized column). It stops when
`max|Δ| < tol`.

`fn fit in linear_svc.rs` validates `n_samples == y.len()` and `C > 0`, computes
`classes` by sort+dedup, errors with `InsufficientSamples` for `< 2` classes,
then: binary (`classes.len() == 2`) maps `y` to `±1` (positive = `classes[1]`)
and solves once; multiclass loops **one-vs-rest** over each class, solving a
binary sub-problem per class. `FittedLinearSVC<F>` stores
`weight_vectors: Vec<Array1<F>>`, `intercepts: Vec<F>`, `classes: Vec<usize>`,
`is_binary: bool`, `n_features`. `fn decision_function in linear_svc.rs` returns
shape `(n, 1)` for binary (a column), `(n, n_classes)` for multiclass.
`fn predict in linear_svc.rs` is `sign(Xw+b)` (binary, `>= 0 → classes[1]`) /
argmax (multiclass). `HasCoefficients` returns the *first* weight vector /
intercept; `HasClasses` returns `classes`.

The constructor `fn LinearSVC::new` defaults `C=1.0`, `max_iter=1000`, `tol=1e-4`,
`loss=SquaredHinge` (matching sklearn's `C`/`max_iter`/`tol`/`loss` defaults).
There is **no** `penalty`, `dual`, `multi_class`, `fit_intercept`,
`intercept_scaling`, `class_weight`, `random_state`, `verbose`, `n_iter_`, or
`n_features_in_` field/accessor, and no `tol > 0` validation.
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

The crux (REQ-1/REQ-10) diverges, mirroring LinearSVR's pre-rewrite C/n bug:
`fn solve_binary_primal in linear_svc.rs` scales the loss by `c / n_f`
(`n_f = n_samples`), minimizing `0.5·‖w‖² + (C/n)·Σ loss` instead of liblinear's
`0.5·‖w‖² + C·Σ loss` — a different optimum. No test pins `coef_`/`intercept_`/
`decision_function` against the live oracle: `conformance_linear_svc in
tests/conformance_wave1.rs` floors only prediction accuracy ≥ 0.90 and
`api_proof_linear_svc in tests/api_proof.rs` checks only `decision_function`
nrows + that `fit`/`predict`/`score` run (R-CHAR-1/R-CHAR-3) — so NO
numerical-parity REQ is SHIPPED. The module's API also lacks `penalty`, `dual`,
`multi_class`, `fit_intercept`/`intercept_scaling`, `class_weight`, `n_iter_`,
and `n_features_in_`, and the binary `decision_function` returns `(n, 1)` not
`(n,)`. Every REQ is NOT-STARTED.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (fit parity — coef_/intercept_ vs liblinear oracle) | NOT-STARTED | open prereq blocker #618. `fn solve_binary_primal in linear_svc.rs` scales the data term by `c / n_f` (the `c / n_f` factor in the squared-hinge gradient `−2·c/n_f·(1−margin)·y·x` and hinge subgradient `−c/n_f·y·x`), minimizing `0.5·‖w‖² + (C/n)·Σ loss` — not liblinear's `0.5·‖w‖² + C·Σ loss` (`_base.py:1214-1228`, no `1/n`). The optimum is shifted: live oracle (8×2 set, `squared_hinge`, `C=1.0`) `coef_ [[0.12835, 0.12835]]`, `intercept_ [-1.19438]`; no test pins this (only accuracy ≥ 0.90). The intercept is also fit by a separate **un-regularized** step, not the penalized augmented column (`_base.py:1188-1198`). |
| REQ-2 (decision_function shape + values) | NOT-STARTED | open prereq blocker #619. `fn decision_function in linear_svc.rs` returns shape `(n_samples, 1)` for the binary case (a column loop into `Array2`), but sklearn's binary `decision_function` is 1-D `(n_samples,)` (live: `m.decision_function(X).shape == (8,)`). Values also depend on the REQ-1 fit. No test pins `decision_function` values against the oracle. |
| REQ-3 (predict + classes_) | NOT-STARTED | open prereq blocker #620. `fn predict in linear_svc.rs` uses the sign convention (`>= 0 → classes[1]`) and argmax, and `HasClasses::classes` returns sorted unique labels (mirrors `classes_ = np.unique(y)`, `_classes.py:311`) — structurally aligned — but the predicted labels are downstream of the REQ-1 fit and no test pins `predict` against the live oracle (only `conformance_linear_svc` accuracy ≥ 0.90, which a wrong-optimum fit can still pass on separable data). |
| REQ-4 (loss {hinge, squared_hinge}) | NOT-STARTED | open prereq blocker #621. `LinearSVCLoss::{Hinge, SquaredHinge}` exist and `solve_binary_primal` branches on them, but (a) both are under the REQ-1 C/n objective, and (b) the `Hinge` branch uses the squared-hinge Hessian as a "smooth majorant" with a subgradient — it does not solve the true hinge optimum (live `hinge` oracle on the 8×2 set: `coef [[0.15385, 0.15385]]`, `intercept [-1.46154]`). No oracle-pinned `coef_` test per loss. |
| REQ-5 (penalty {l1, l2}) | NOT-STARTED | open prereq blocker #622. `LinearSVC<F>` has no `penalty` field — the solver hardcodes the L2 regularizer (`grad = w[j]`, `hess = 1` in `solve_binary_primal`). sklearn defaults `penalty='l2'` and supports `'l1'` (sparse `coef_`, solver type 5, `_base.py:1014`); `('l1','hinge')` is rejected (`_classes.py:59-60`). l1 is entirely absent. |
| REQ-6 (multi_class {ovr, crammer_singer}) | NOT-STARTED | open prereq blocker #623. `fn fit in linear_svc.rs` implements one-vs-rest (a binary solve per class) — structurally sklearn's default `'ovr'` — but there is no `multi_class` field and no `crammer_singer` joint solver (`_base.py:1017` solver type 4); and the OvR per-class `coef_` is downstream of the REQ-1 C/n bug, so the per-class rows do not match the oracle (`coef_` shape `(3, 2)` on the 9×2 3-class set). No per-class `coef_` test vs the oracle. |
| REQ-7 (fit_intercept + intercept_scaling) | NOT-STARTED | open prereq blocker #624. `LinearSVC<F>` has no `fit_intercept`/`intercept_scaling` fields. The intercept is always fit by a separate **un-regularized** coordinate-Newton step in `solve_binary_primal` (a `1e-12` ridge, no `x` factor), whereas liblinear augments `x` with a penalized `intercept_scaling` column and sets `intercept_ = intercept_scaling·w_last` (`_base.py:1188-1198, :1240-1245`). No `fit_intercept=False → intercept_=0` path and no `intercept_scaling > 0` validation. |
| REQ-8 (dual param) | NOT-STARTED | open prereq blocker #625. `LinearSVC<F>` has no `dual` field. sklearn defaults `dual='auto'` (`_classes.py:253`), resolving to the liblinear solver type per penalty×loss×dual (`_get_liblinear_solver_type`, `_base.py:995`), with `hinge` dual-only and `l1` primal-only — the unsupported combinations raise `ValueError`. None of this exists. |
| REQ-9 (class_weight) | NOT-STARTED | open prereq blocker #626. `LinearSVC<F>` has no `class_weight` field. sklearn scales `C` per class via `compute_class_weight` (`_base.py:1179`); `'balanced'` uses `n_samples/(n_classes·bincount(y))` (`_classes.py:118-124`). ferrolearn's `solve_binary_primal` applies a uniform `C` to every sample. Absent. |
| REQ-10 (C-scaling convention) | NOT-STARTED | open prereq blocker #618 (shared with REQ-1 — the same root cause). The `c / n_f` division in `solve_binary_primal` must be removed so the data term is plain `C` (summed loss). Live oracle proves the C-dependence the bug flattens: `fit_intercept=False`, `squared_hinge`, `C ∈ {0.01,0.1,1.0,10.0} → coef ∈ {0.04321, 0.04643, 0.04678, 0.04682}` (saturating, not constant). |
| REQ-11 (n_iter_/n_features_in_ + param validation) | NOT-STARTED | open prereq blocker #627. `FittedLinearSVC<F>` stores `n_features` but exposes no `n_features_in_` accessor and no `n_iter_` (sklearn `n_iter_ = n_iter_.max().item()`, `_classes.py:338`, with a `ConvergenceWarning` at `max_iter`, `_base.py:1234-1238`). `fn fit` validates `C > 0` and `≥ 2` classes but NOT `tol > 0` (sklearn `Interval(Real, 0.0, None, closed="neither")`, `_classes.py:237`). |
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
