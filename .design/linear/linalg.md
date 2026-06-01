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

This module is the root cause of the OLS minimum-norm divergences (#376, #377)
and is the migration site to the ferray substrate (#375). The documented fix
routes the OLS solve through `ferray::linalg::lstsq`
(`ferray-linalg/src/solve.rs:208`), a single-SVD gelsd-equivalent solver.

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
| REQ-1 (full-rank OLS solve) | SHIPPED | impl `pub(crate) fn solve_lstsq in linalg.rs` (f64 → `solve_lstsq_faer`, faer QR `qr.solve_lstsq`) and `pub(crate) fn solve_normal_equations in linalg.rs` (Cholesky on `XᵀX`) mirror the full-rank case of `linalg.lstsq` (`sklearn/linear_model/_base.py:687`: `self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)`). Non-test consumer: `Fit for LinearRegression in linear_regression.rs` calls `linalg::solve_normal_equations(...).or_else(\|_\| linalg::solve_lstsq(...))`; also `solve_ridge`/`solve_ridge_multi` consumed by `Fit for Ridge in ridge.rs` and `Fit for RidgeClassifier in ridge_classifier.rs`. Verification: live oracle for `X=[[1,1],[2,1],[3,2],[4,2]]`, `y=[6,7,10,11]` → `coef=[1.0,2.0]`, `intercept=3.0`; matched by `test_multiple_linear_regression` in `linear_regression.rs`. |
| REQ-2 (rank-deficient min-norm) | NOT-STARTED | open prereq blocker #376. `solve_lstsq` delegates to faer QR (`qr.solve_lstsq` in `solve_lstsq_faer`), which on a rank-deficient design returns a basic (non-minimum-norm) least-squares solution, whereas `linalg.lstsq` (gelsd SVD) zeroes sub-rcond singular values to produce the min-norm solution. Pinned `#[ignore]`d: `divergence_rank_deficient_no_intercept_min_norm` / `divergence_rank_deficient_with_intercept_min_norm` in `tests/divergence_linreg_minnorm.rs`. Fix: route through `ferray::linalg::lstsq` (`ferray-linalg/src/solve.rs:208`). |
| REQ-3 (underdetermined accepted) | NOT-STARTED | open prereq blocker #377. `solve_lstsq in linalg.rs` returns `FerroError::InsufficientSamples` when `n_samples < n_features` (the `if n_samples < n_features` guard), so `LinearRegression::fit` rejects input that `linalg.lstsq` accepts. Pinned `#[ignore]`d: `divergence_underdetermined_accepted_min_norm` in `tests/divergence_linreg_minnorm.rs`. Fix: `ferray::linalg::lstsq` accepts any m×n and returns the min-norm solution. |
| REQ-4 (rank_ / singular_ exposed) | NOT-STARTED | open prereq blocker #374. No symbol in `linalg.rs` returns rank or singular values; `solve_lstsq`/`solve_normal_equations`/`solve_lstsq_faer` return only the coefficient vector, so `LinearRegression` cannot expose sklearn's `rank_`/`singular_` (`_base.py:687`). `ferray::linalg::lstsq` returns `(solution, residuals, rank, singular_values)` — adopting it enables this REQ. |
| REQ-5 (safe_sparse_dot helper) | NOT-STARTED | open prereq blocker #380. `linalg.rs` exposes no dot/matmul helper mirroring `safe_sparse_dot` (`sklearn/utils/extmath.py:161`); estimators call `ndarray`'s `.dot()` inline (e.g. `x.t().dot(x)` in `solve_normal_equations`/`solve_ridge`). No dense/sparse dispatch wrapper exists. |
| REQ-6 (ferray substrate) | NOT-STARTED | open prereq blocker #375. `linalg.rs` uses `faer::Mat` + `faer::linalg::solvers::SolveLstsq` (`solve_lstsq_faer`, `ndarray_to_faer_f64`, `faer_col_to_ndarray_f64`) and `ndarray` throughout — the wrong substrate per R-SUBSTRATE-1. Destination is `ferray::linalg` (`ferray-linalg/src/solve.rs:208`). The planned `lstsq` migration is this REQ's fix. |

## Architecture

`linalg.rs` is `pub(crate)` only; its symbols never cross the crate boundary —
the public surface is the estimator types (`LinearRegression`, `Ridge`,
`RidgeClassifier`), which are the grandfathered boundary API (R-DEFER-1 /
S5). Three solver families exist:

1. **Unregularized least squares.** `solve_lstsq` (entry point) dispatches on
   `F`: for `f64` it converts to `faer::Mat` (`ndarray_to_faer_f64`), runs a
   QR factorization, and calls `qr.solve_lstsq` (`solve_lstsq_faer`); for other
   floats it falls back to `solve_normal_equations` (Cholesky on `XᵀX` via
   `cholesky_solve`, with a partial-pivot `gaussian_solve` fallback). The
   consumer `Fit for LinearRegression in linear_regression.rs` actually tries
   `solve_normal_equations` first and only `.or_else`-falls back to
   `solve_lstsq`. This contrasts with sklearn's single dense path
   `linalg.lstsq(X, y)` (`_base.py:687`), which is *always* SVD (gelsd). The
   normal-equations + QR approach is correct for full-rank, well-conditioned
   `X` (REQ-1) but does not satisfy the min-norm/underdetermined contract
   (REQ-2/REQ-3) because neither Cholesky-on-`XᵀX` nor faer QR truncate
   sub-rcond singular values.

2. **Ridge.** `solve_ridge` and `solve_ridge_multi` add `alpha * I` to `XᵀX`
   and Cholesky-solve (`cholesky_solve` / `cholesky_solve_multi`, the latter
   sharing one factorization across target columns). These are positive-definite
   for `alpha > 0`, so the SVD-min-norm concern does not apply; consumed by
   `Fit for Ridge in ridge.rs` and `Fit for RidgeClassifier in
   ridge_classifier.rs`.

3. **Low-level kernels.** `cholesky_solve`, `cholesky_solve_multi`,
   `gaussian_solve` are hand-rolled `ndarray` routines.

**Invariant gap (the divergence).** sklearn's contract for the OLS path is
*minimum-norm least squares* on `X` of any shape, with `rank_`/`singular_`
side outputs (`_base.py:687`). `linalg.rs` instead enforces
`n_samples >= n_features` and uses non-SVD solvers, so it (a) rejects valid
underdetermined input (REQ-3, #377) and (b) returns non-min-norm solutions on
rank-deficient input (REQ-2, #376), and (c) cannot surface `rank_`/`singular_`
(REQ-4, #374).

**Planned fix (documented path; not yet implemented).** Replace the f64 OLS
path with `ferray::linalg::lstsq` (`ferray-linalg/src/solve.rs:208`). That
function performs a single SVD (`crate::decomp::svd`), computes the effective
`rank` as the count of singular values above `rcond * max_sv` (matching gelsd's
default `rcond = max(m,n) * eps`), and forms `x = V · diag(1/sᵢ) · Uᵀ · b`
with sub-cutoff singular values zeroed — i.e. exactly the minimum-norm
solution. It accepts any `m × n` (no `n < p` guard) and returns
`(solution, residuals, rank, singular_values)`, satisfying REQ-2, REQ-3, and
REQ-4 in one move and migrating the substrate (REQ-6). During the transition
the `ndarray ↔ ferray` conversion happens at the `linalg.rs` boundary
(R-SUBSTRATE-4); the estimator callers keep their `ndarray` signatures.
`safe_sparse_dot` (REQ-5) is independent of the lstsq fix and remains
NOT-STARTED under its own blocker.

## Verification

Commands establishing the SHIPPED claim (REQ-1) and the NOT-STARTED diagnoses:

- Library gauntlet:
  `cargo test -p ferrolearn-linear` (unit tests `test_simple_linear_regression`,
  `test_multiple_linear_regression`, `test_no_intercept`, `test_solve_lstsq_*`,
  `test_solve_ridge` cover the full-rank/regularized paths).
- Full-rank oracle (REQ-1 / AC-1):
  `python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; m=LinearRegression().fit(np.array([[1.,1.],[2.,1.],[3.,2.],[4.,2.]]), np.array([6.,7.,10.,11.])); print(m.coef_.tolist(), m.intercept_, m.rank_, m.singular_.tolist())"`
  → `coef=[1.0, 2.0]`, `intercept=3.0`, `rank=2`,
  `singular=[2.41421356…, 0.41421356…]`; matched by ferrolearn within 1e-8.
- Divergence pins (REQ-2/REQ-3, currently `#[ignore]`d, green only after the
  fix lands per R-DEFER-3):
  `cargo test -p ferrolearn-linear --test divergence_linreg_minnorm -- --ignored`
  — `divergence_rank_deficient_no_intercept_min_norm`,
  `divergence_rank_deficient_with_intercept_min_norm`,
  `divergence_underdetermined_accepted_min_norm`. These assert the gelsd
  min-norm oracle values and must remain failing/ignored until REQ-2/REQ-3
  ship.

Because the divergence tests do not pass today (they are `#[ignore]`d pins),
REQ-2, REQ-3, REQ-4, REQ-5, and REQ-6 are NOT-STARTED; only REQ-1 is SHIPPED.
