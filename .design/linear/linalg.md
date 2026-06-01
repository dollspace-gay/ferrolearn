# Linear Algebra Helpers (ferrolearn-linear)

<!--
tier: 3-component
status: draft
baseline-commit: 52807aa87fb68b9118962db99e93ef932aa74f59
upstream-paths:
  - sklearn/utils/extmath.py
  - sklearn/linear_model/_base.py
-->

## Summary

`ferrolearn-linear/src/linalg.rs` is the crate-private linear-algebra toolbox
that backs the linear estimators (`LinearRegression`, `Ridge`,
`RidgeClassifier`). It does not mirror a single public scikit-learn estimator;
it serves the *solver contract* those estimators rely on. Two distinct upstream
contracts apply:

- The **least-squares / OLS solve** consumed by `LinearRegression` mirrors
  `self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)`
  (`sklearn/linear_model/_base.py:687`), i.e. `scipy.linalg.lstsq` →
  LAPACK `gelsd` (SVD-based), which returns the unique **minimum-norm**
  least-squares solution for rank-deficient `X` and accepts underdetermined
  systems (`n_samples < n_features`).
- The **matrix-product helper** the route assigns (`safe_sparse_dot`,
  `sklearn/utils/extmath.py:161`) is the numpy/scipy `@`-dispatch wrapper
  scikit-learn uses internally; ferrolearn currently has no analog symbol
  (estimators call `ndarray`'s `.dot()` inline).

This module was the root cause of the OLS minimum-norm divergences (#376,
#377) and is the migration site to the ferray substrate (#375). The OLS solve
now routes through `ferray::linalg::lstsq` (`ferray-linalg/src/solve.rs:208`),
a single-SVD gelsd-equivalent solver, which closed #376/#377 and migrated the
OLS least-squares path onto ferray.

## Requirements

- REQ-1: Full-rank OLS least-squares solve — for `X` with full column rank,
  `solve_lstsq` / `solve_normal_equations` produce coefficients matching
  `linalg.lstsq(X, y)` (`_base.py:687`) to floating tolerance.
- REQ-2: Minimum-norm solution for rank-deficient `X` — for `X` with a
  rank-deficient column space, the solver returns the unique minimum-norm
  least-squares solution (gelsd parity), not an arbitrary basic solution.
- REQ-3: Underdetermined systems accepted — for `n_samples < n_features` the
  solver succeeds and returns the minimum-norm solution, as `linalg.lstsq`
  does; it must not reject the input.
- REQ-4: Effective rank and singular values exposed — the solve yields
  `rank_` and `singular_` matching `linalg.lstsq`'s returns
  (`_base.py:687`), so consumers can surface the sklearn fitted attributes.
- REQ-5: A `safe_sparse_dot`-equivalent matrix-product helper
  (`extmath.py:161`) — a single dispatched dot/matmul entry point usable by
  the linear estimators.
- REQ-6 (substrate): The OLS/least-squares computation runs on the ferray
  substrate (`ferray::linalg`), not on `faer` + `ndarray` directly, per
  R-SUBSTRATE-1/2.

## Acceptance criteria

- AC-1: For a full-rank `X` (e.g. `[[1,1],[2,1],[3,2],[4,2]]`, `y=[6,7,10,11]`),
  ferrolearn `LinearRegression::fit` `coef_`/`intercept_` match the live
  sklearn oracle `[1.0, 2.0]`, intercept `3.0` within 1e-8.
- AC-2: For rank-deficient `X=[[1,1],[2,2],[3,3]]`, `y=[1,2,3]`,
  `fit_intercept=false`, `coef_` equals the gelsd min-norm split
  `[0.5, 0.5]` within 1e-8 (the assertion in
  `divergence_rank_deficient_no_intercept_min_norm`).
- AC-3: For underdetermined `X=[[1,2,3],[4,5,6]]`, `y=[1,2]`,
  `fit_intercept=false`, `fit` succeeds (no `InsufficientSamples`) and `coef_`
  equals `[-0.0555…, 0.1111…, 0.2777…]` within 1e-8
  (`divergence_underdetermined_accepted_min_norm`).
- AC-4: The solver returns an effective rank and singular-value vector matching
  sklearn `rank_`/`singular_` (e.g. for AC-1's `X`, `rank=2`,
  `singular=[2.41421356…, 0.41421356…]`).
- AC-5: A dot/matmul helper symbol exists in `linalg.rs` and is the call site
  used by at least one estimator for its `X @ w` / `Xᵀ X` products.
- AC-6: `linalg.rs`'s OLS solve performs its decomposition via `ferray::linalg`
  rather than `faer::linalg::solvers`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (full-rank OLS solve) | SHIPPED | impl `pub(crate) fn solve_lstsq in linalg.rs` now routes through `ferray::linalg::lstsq` (single SVD), mirroring `linalg.lstsq` (`sklearn/linear_model/_base.py:687`: `self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)`) for the full-rank case too. Non-test consumer: `Fit for LinearRegression in linear_regression.rs` calls `linalg::solve_lstsq(...)` directly; `solve_ridge`/`solve_ridge_multi` (still Cholesky) consumed by `Fit for Ridge in ridge.rs` and `Fit for RidgeClassifier in ridge_classifier.rs`. Verification: live oracle for `X=[[1,1],[2,1],[3,2],[4,2]]`, `y=[6,7,10,11]` → `coef=[1.0,2.0]`, `intercept=3.0`; matched by `test_multiple_linear_regression` in `linear_regression.rs` (still passes post-SVD-switch). |
| REQ-2 (rank-deficient min-norm) | SHIPPED | `pub(crate) fn solve_lstsq in linalg.rs` routes through `ferray::linalg::lstsq` (`ferray-linalg/src/solve.rs:208`), a single-SVD gelsd-equivalent that zeroes sub-`rcond` singular values, yielding the unique minimum-norm least-squares solution. Non-test consumer: `Fit for LinearRegression in linear_regression.rs`. Verification (live sklearn 1.5.2 oracle): rank-1 `X=[[1,1],[2,2],[3,3]]` → `coef=[0.5,0.5]` (`divergence_rank_deficient_no_intercept_min_norm`); with intercept `coef=[1,1]`, `intercept=-1` (`divergence_rank_deficient_with_intercept_min_norm`) — both now passing (`#[ignore]` removed, R-DEFER-3). Closes #376. Module unit test `test_solve_lstsq_rank_deficient_min_norm`. |
| REQ-3 (underdetermined accepted) | SHIPPED | the `n_samples < n_features` rejection (`FerroError::InsufficientSamples`) has been removed from `solve_lstsq in linalg.rs`; `ferray::linalg::lstsq` accepts any m×n and returns the min-norm solution. Non-test consumer: `Fit for LinearRegression in linear_regression.rs`. Verification (live sklearn 1.5.2 oracle): underdetermined `X=[[1,2,3],[4,5,6]]` (2×3), `y=[1,2]` → `coef=[-0.0556,0.1111,0.2778]` (`divergence_underdetermined_accepted_min_norm`, now passing, `#[ignore]` removed). Closes #377. Module unit test `test_solve_lstsq_underdetermined_accepted`. |
| REQ-4 (rank_ / singular_ exposed) | NOT-STARTED | blocker #374. `ferray::linalg::lstsq` (`ferray-linalg/src/solve.rs:208`) now *returns* `(solution, residuals, rank, singular_values)`, and `solve_lstsq in linalg.rs` receives `rank`/`singular` (currently bound as `_rank`/`_singular`). The values are now available at the solve site, but `LinearRegression`/`FittedLinearRegression` does not yet store or expose `rank_`/`singular_` fitted attributes (`_base.py:687`). Surfacing them is REQ-4's remaining work. |
| REQ-5 (safe_sparse_dot helper) | NOT-STARTED | open prereq blocker #380. `linalg.rs` exposes no dot/matmul helper mirroring `safe_sparse_dot` (`sklearn/utils/extmath.py:161`); estimators call `ndarray`'s `.dot()` inline (e.g. `x.t().dot(x)` in `solve_ridge`). No dense/sparse dispatch wrapper exists. |
| REQ-6 (ferray substrate) | SHIPPED | the OLS/least-squares computation now runs on the ferray substrate: `solve_lstsq in linalg.rs` builds `ferray::Array` from a flat Vec + shape (`FerrayArray::<F, Ix2>::from_vec`, `from_vec(IxDyn, ..)`), calls `ferray::linalg::lstsq` (`ferray-linalg/src/solve.rs:208`), and bridges back via `into_ndarray()` (R-SUBSTRATE-4). The faer OLS path (`solve_lstsq_faer`, `ndarray_to_faer_f64`, `faer_col_to_ndarray_f64`) is removed (AC-6 satisfied). Consumer: `Fit for LinearRegression in linear_regression.rs`. Remaining `ndarray`/Cholesky on the Ridge path (`solve_ridge`/`cholesky_solve`) is tracked under #375 for the Ridge unit's iteration. |

## Architecture

`linalg.rs` is `pub(crate)` only; its symbols never cross the crate boundary —
the public surface is the estimator types (`LinearRegression`, `Ridge`,
`RidgeClassifier`), which are the grandfathered boundary API (R-DEFER-1 /
S5). Three solver families exist:

1. **Unregularized least squares.** `solve_lstsq` (entry point) bridges the
   incoming `ndarray` design/target to `ferray::Array` (flat Vec + shape via
   `from_vec`), calls `ferray::linalg::lstsq(a, b, None)`
   (`ferray-linalg/src/solve.rs:208`) — a single SVD that zeroes sub-`rcond`
   singular values (default `rcond = max(m, n) * eps`) to form the
   minimum-norm solution — and bridges the result back via `into_ndarray()`.
   This is generic over `F: ferray::linalg::LinalgFloat` (f32/f64). It matches
   sklearn's single dense path `linalg.lstsq(X, y)` (`_base.py:687`), which is
   *always* SVD (gelsd): full-rank (REQ-1), rank-deficient min-norm (REQ-2),
   and underdetermined `n_samples < n_features` (REQ-3) are all handled by the
   one path. The consumer `Fit for LinearRegression in linear_regression.rs`
   calls `solve_lstsq` directly (on the centered design when
   `fit_intercept`). The former faer-QR / normal-equations OLS path has been
   removed.

2. **Ridge.** `solve_ridge` and `solve_ridge_multi` add `alpha * I` to `XᵀX`
   and Cholesky-solve (`cholesky_solve` / `cholesky_solve_multi`, the latter
   sharing one factorization across target columns). These are positive-definite
   for `alpha > 0`, so the SVD-min-norm concern does not apply; consumed by
   `Fit for Ridge in ridge.rs` and `Fit for RidgeClassifier in
   ridge_classifier.rs`.

3. **Low-level kernels.** `cholesky_solve`, `cholesky_solve_multi`,
   `gaussian_solve` are hand-rolled `ndarray` routines (Ridge path only).

**Resolved invariant gap.** sklearn's contract for the OLS path is
*minimum-norm least squares* on `X` of any shape, with `rank_`/`singular_`
side outputs (`_base.py:687`). The former `linalg.rs` enforced
`n_samples >= n_features` and used non-SVD solvers, so it (a) rejected valid
underdetermined input (#377) and (b) returned non-min-norm solutions on
rank-deficient input (#376). Routing `solve_lstsq` through
`ferray::linalg::lstsq` (a single SVD via `crate::decomp::svd`, effective
`rank` = count of singular values above `rcond * max_sv`, default
`rcond = max(m,n) * eps`, forming `x = V · diag(1/sᵢ) · Uᵀ · b` with
sub-cutoff singular values zeroed) resolves both: it accepts any `m × n` (no
`n < p` guard) and returns the minimum-norm solution. This closed REQ-2/REQ-3
(#376/#377) and migrated the OLS substrate (REQ-6). The function also returns
`(solution, residuals, rank, singular_values)`; the `rank`/`singular` values
are now available at the solve site but not yet stored as `LinearRegression`
fitted attributes (REQ-4 remains NOT-STARTED, #374). The `ndarray ↔ ferray`
conversion happens at the `linalg.rs` boundary (R-SUBSTRATE-4); the estimator
callers keep their `ndarray` signatures. `safe_sparse_dot` (REQ-5) is
independent and remains NOT-STARTED under its own blocker (#380).

## Verification

- Library gauntlet (post-fix, all green):
  `cargo test -p ferrolearn-linear` — unit tests `test_simple_linear_regression`,
  `test_multiple_linear_regression`, `test_no_intercept`,
  `test_solve_lstsq_simple`, `test_solve_lstsq_multi`,
  `test_solve_lstsq_rank_deficient_min_norm`,
  `test_solve_lstsq_underdetermined_accepted`, `test_solve_ridge` pass.
- Full-rank oracle (REQ-1 / AC-1):
  `python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; m=LinearRegression().fit(np.array([[1.,1.],[2.,1.],[3.,2.],[4.,2.]]), np.array([6.,7.,10.,11.])); print(m.coef_.tolist(), m.intercept_, m.rank_, m.singular_.tolist())"`
  → `coef=[1.0, 2.0]`, `intercept=3.0`, `rank=2`,
  `singular=[2.41421356…, 0.41421356…]`; matched by ferrolearn within 1e-8
  (still holds after the SVD switch).
- Divergence pins (REQ-2/REQ-3 — `#[ignore]` removed, now passing per
  R-DEFER-3):
  `cargo test -p ferrolearn-linear --test divergence_linreg_minnorm`
  — `divergence_rank_deficient_no_intercept_min_norm` (coef `[0.5,0.5]`),
  `divergence_rank_deficient_with_intercept_min_norm` (coef `[1,1]`,
  intercept `-1`), `divergence_underdetermined_accepted_min_norm`
  (coef `[-0.0556,0.1111,0.2778]`) — all 3 green against the sklearn 1.5.2
  oracle.

REQ-1, REQ-2, REQ-3, and REQ-6 are SHIPPED; REQ-4 (rank_/singular_ exposure,
#374) and REQ-5 (`safe_sparse_dot`, #380) remain NOT-STARTED.
