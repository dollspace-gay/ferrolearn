# Orthogonal Matching Pursuit (OMP)

<!--
tier: 3-component
status: draft
baseline-commit: 1476a9cecaf42a661ee1802c842cbcbb7b8e29b5
upstream-paths:
  - sklearn/linear_model/_omp.py
-->

## Summary
This module mirrors scikit-learn's `OrthogonalMatchingPursuit` regressor
(`sklearn/linear_model/_omp.py:645`): a greedy sparse-regression estimator that
iteratively adds the feature most correlated with the current residual to an
active set, re-solves OLS on the active columns, and recomputes the residual
until either `n_nonzero_coefs` features are selected or the squared residual
norm falls below `tol`. The ferrolearn target owns the single-target,
non-precomputed (Cholesky/dense) path of `OrthogonalMatchingPursuit`. The
Gram/precompute path, the `orthogonal_mp`/`orthogonal_mp_gram` free functions,
multi-output, `n_iter_`, and `OrthogonalMatchingPursuitCV` are not yet
implemented.

## Requirements
- REQ-1: Greedy OMP fit produces `coef_`/`intercept_` matching sklearn's
  `OrthogonalMatchingPursuit(n_nonzero_coefs=k)` on a real dataset (column
  centering when `fit_intercept=True`, greedy max-|correlation| selection,
  active-set OLS via Cholesky, residual update).
- REQ-2: When both `n_nonzero_coefs` and `tol` are `None`, the default number of
  non-zero coefficients is `max(int(0.1 * n_features), 1)`
  (`_omp.py:785`), and `fit` succeeds (does not error).
- REQ-3: `tol` stopping — the loop terminates when the squared residual norm
  `||residual||^2 <= tol` (`_omp.py:141`).
- REQ-4: `predict` computes `X @ coef_ + intercept_` and validates the feature
  count.
- REQ-5: `fit_intercept` toggles intercept estimation; fitted coefficients and
  intercept are exposed through the `HasCoefficients` introspection trait
  (mirroring sklearn's `coef_` / `intercept_` attributes).
- REQ-6: `precompute='auto'` Gram path — when `n_samples > n_features`, fit via
  the precomputed Gram matrix (`orthogonal_mp_gram` / `_gram_omp`,
  `_omp.py:481`, `_omp.py:152`); sklearn's default `precompute='auto'`.
- REQ-7: `OrthogonalMatchingPursuitCV` (`_omp.py:894`) — cross-validated
  selection of `n_nonzero_coefs` (routed `parity_op`).
- REQ-8: `n_iter_` fitted attribute — number of active features at convergence
  (`_omp.py:799`).
- REQ-9: Multi-output `y` of shape `(n_samples, n_targets)` (`_omp.py:444`
  per-target loop; `coef_` shape `(n_targets, n_features)`).
- REQ-10: ferray-substrate migration — the owned linear algebra (Cholesky,
  Gaussian fallback, centering) lives on `ferray::linalg` / `ferray-core`
  rather than `ndarray` plus hand-rolled solvers (R-SUBSTRATE-2).

## Acceptance criteria
- AC-1 (REQ-1): with `n_nonzero_coefs=5` on `load_diabetes`, ferrolearn `coef_`
  matches sklearn within 1e-9 element-wise and `intercept_` within 1e-9.
- AC-2 (REQ-2): `OrthogonalMatchingPursuit::new().fit(X, y)` on a 10-feature
  dataset selects exactly 1 non-zero coefficient and matches sklearn's
  `n_nonzero_coefs_ == 1` — without raising an error.
- AC-3 (REQ-3): on a perfectly linear single-feature problem,
  `with_tol(1e-10)` reaches a zero-residual fit using one feature.
- AC-4 (REQ-4): `predict` on the training matrix returns a vector of length
  `n_samples`; mismatched feature count returns `ShapeMismatch`.
- AC-5 (REQ-5): `coefficients()` length equals `n_features`; with
  `fit_intercept=false` the intercept is exactly `0`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (greedy fit matches sklearn) | SHIPPED | impl `fn fit in omp.rs` (greedy selection loop: `corr = x_work.column(j).dot(&residual).abs()`, active-set solve `ols_active`, residual update `residual = &y_work - x_work.dot(&w)`) mirrors `_cholesky_omp` (`_omp.py:102` `lam = np.argmax(np.abs(np.dot(X.T, residual)))`; `_omp.py:140` `residual = y - np.dot(X[:, :n_active], gamma)`). Centering mirrors `_pre_fit` + `_set_intercept` (`_omp.py:775`,`:815`). Non-test consumer: `pub use omp::{FittedOMP, OrthogonalMatchingPursuit}` in `lib.rs` (boundary estimator = public API, grandfathered per R-DEFER-1/R-DEFER-5). Live-oracle verification on `load_diabetes`, `n_nonzero_coefs=5`: sklearn `coef_[1]=-235.77241317516382`, ferrolearn `-235.7724131751638`; **max abs coef diff 1.08e-12, intercept diff 0.0**. |
| REQ-2 (default n_nonzero_coefs) | NOT-STARTED | open prereq blocker #488. sklearn defaults to `max(int(0.1 * n_features), 1)` when both `n_nonzero_coefs` and `tol` are `None` (`_omp.py:785`; class docstring `_omp.py:652`); ferrolearn's `fn fit in omp.rs` instead returns `FerroError::InvalidParameter` ("at least one stopping criterion must be set") for that exact input — a divergence: sklearn fits, ferrolearn errors. |
| REQ-3 (tol stopping) | SHIPPED | impl `fn fit in omp.rs` (`if res_norm_sq < tol_val { break; }` where `res_norm_sq = residual.dot(&residual)`) mirrors `_omp.py:141` (`if tol is not None and nrm2(residual) ** 2 <= tol: break`). Non-test consumer: `pub use` re-export in `lib.rs`. Note: ferrolearn checks `<` at the top of the loop before adding a feature; sklearn checks `<=` after adding. Equivalent for typical inputs but a boundary divergence (strict vs non-strict; pre- vs post-selection). Verification: `cargo test -p ferrolearn-linear` (`test_tol_stopping`). |
| REQ-4 (predict) | SHIPPED | impl `fn predict in omp.rs` (`Ok(x.dot(&self.coefficients) + self.intercept)`; `ShapeMismatch` on feature-count mismatch) mirrors `LinearModel._decision_function`. Non-test consumer: `pub use` re-export + `impl FittedPipelineEstimator for FittedOMP` (`predict_pipeline` calls `self.predict`). Verification: `cargo test -p ferrolearn-linear` (`test_predict`, `test_predict_feature_mismatch`). |
| REQ-5 (fit_intercept / HasCoefficients) | SHIPPED | impl `fn fit in omp.rs` centering branch (`x - &x_mean`, `y - y_mean`, `intercept = *ym - xm.dot(&w)`) mirrors `_set_intercept` (`_omp.py:815`); `impl HasCoefficients for FittedOMP` exposes `coefficients()`/`intercept()` (sklearn `coef_`/`intercept_`). Non-test consumer: `pub use` re-export in `lib.rs`. Verification: `cargo test -p ferrolearn-linear` (`test_has_coefficients`, `test_no_intercept`). |
| REQ-6 (precompute Gram path) | NOT-STARTED | open prereq blocker #489. sklearn's default `precompute='auto'` builds the Gram matrix when `n_samples > n_features` and dispatches to `orthogonal_mp_gram`/`_gram_omp` (`_omp.py:481`,`:152`,`:417`); ferrolearn always uses the dense per-feature correlation path. No Gram code path exists. |
| REQ-7 (OrthogonalMatchingPursuitCV) | NOT-STARTED | open prereq blocker #490. sklearn's `OrthogonalMatchingPursuitCV` (`_omp.py:894`) cross-validates `n_nonzero_coefs` via `_omp_path_residues`; it is a routed `parity_op` with no ferrolearn type. |
| REQ-8 (n_iter_) | NOT-STARTED | open prereq blocker #491. sklearn exposes `n_iter_` (active-feature count, `_omp.py:799`); `FittedOMP` stores only `coefficients` and `intercept` — no iteration/active-count attribute. |
| REQ-9 (multi-output) | NOT-STARTED | open prereq blocker #492. sklearn loops over target columns (`_omp.py:444`) and yields `coef_` of shape `(n_targets, n_features)`; ferrolearn's `Fit<Array2<F>, Array1<F>>` accepts only a 1-D target. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #493. `omp.rs` imports `ndarray::{Array1, Array2, ...}` and hand-rolls `cholesky_solve`/`gaussian_solve`; per R-SUBSTRATE-2 the unit is not fully shipped until centering and the active-set solve run on `ferray-core` / `ferray::linalg`. |

## Architecture
The unfitted estimator is `OrthogonalMatchingPursuit<F>` (fields `n_nonzero_coefs:
Option<usize>`, `tol: Option<F>`, `fit_intercept: bool`), constructed via `new()`
plus the `with_n_nonzero_coefs` / `with_tol` / `with_fit_intercept` builders
(defaults `None`/`None`/`true`, mirroring sklearn's `__init__` signature at
`_omp.py:742` — except sklearn's `precompute='auto'` has no ferrolearn field, see
REQ-6). Fitting yields `FittedOMP<F>` holding `coefficients: Array1<F>` (a full
`n_features`-length vector, zero outside the support) and `intercept: F`.

`fn fit in omp.rs` performs: shape/sample validation
(`ShapeMismatch`/`InsufficientSamples`); a stopping-criterion guard that
errors when both `n_nonzero_coefs` and `tol` are `None` (the REQ-2 divergence —
sklearn instead applies the `0.1 * n_features` default at `_omp.py:785`); an
`n_nonzero_coefs > n_features` check mirroring sklearn's "number of atoms cannot
be more than the number of features" (`_omp.py:413`); optional column-mean and
target-mean centering; then the greedy loop. Each iteration optionally checks the
`tol` break, scans non-active columns for the maximum `|X_work[:,j] · residual|`
(`_omp.py:102`), pushes the winner into `support`, re-solves OLS over the active
columns via `ols_active` (which forms `XaᵀXa` / `Xaᵀy` and calls
`cholesky_solve` with a `gaussian_solve` partial-pivot fallback — ferrolearn's
analog of sklearn's LAPACK `potrs` triangular solve at `_omp.py:134`), and
recomputes the residual. The intercept is recovered as `y_mean − x_mean · w`
(`_set_intercept`, `_omp.py:815`).

Key invariants matching sklearn: features are selected at most once (ferrolearn
skips `in_support[j]` columns; sklearn breaks on `lam < n_active`); the active
set grows monotonically; `coef_` is sparse with support equal to the selected
indices. Divergences are catalogued as REQ-2 (default), REQ-3 boundary
(`<` pre-selection vs `<=` post-selection), REQ-6 (Gram path), REQ-8 (`n_iter_`),
REQ-9 (multi-output), REQ-10 (substrate).

The estimator participates in pipelines through `impl PipelineEstimator for
OrthogonalMatchingPursuit` and `impl FittedPipelineEstimator for FittedOMP`, and
is re-exported at the crate boundary by `pub use omp::{FittedOMP,
OrthogonalMatchingPursuit}` in `lib.rs`.

## Verification
- `cargo test -p ferrolearn-linear` — exercises `test_simple_linear`,
  `test_sparsity`, `test_tol_stopping`, `test_predict`,
  `test_predict_feature_mismatch`, `test_has_coefficients`, `test_no_intercept`,
  `test_multivariate_recovery`, `test_pipeline` (pins REQ-1,3,4,5).
- Live sklearn oracle (REQ-1), `load_diabetes`, `n_nonzero_coefs=5`:
  ```
  python3 -c "from sklearn.linear_model import OrthogonalMatchingPursuit as OMP; \
    from sklearn.datasets import load_diabetes; X,y=load_diabetes(return_X_y=True); \
    m=OMP(n_nonzero_coefs=5).fit(X,y); print(m.coef_.tolist(), m.intercept_)"
  ```
  sklearn → `coef_[1] = -235.77241317516382`, `intercept_ = 152.13348416289602`;
  ferrolearn `OrthogonalMatchingPursuit::<f64>::new().with_n_nonzero_coefs(5)` →
  `coef_[1] = -235.7724131751638`, `intercept_ = 152.13348416289602`.
  Max abs `coef_` diff = 1.08e-12, intercept diff = 0.0 (REQ-1 SHIPPED).
- Live sklearn oracle (REQ-2 divergence), `load_diabetes` default:
  sklearn `OMP().fit(X, y)` → `n_nonzero_coefs_ = 1`, 1 non-zero coefficient;
  ferrolearn `OrthogonalMatchingPursuit::new().fit(X, y)` → `Err(InvalidParameter)`.
  This pins REQ-2 NOT-STARTED (blocker #488).
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`;
  `cargo fmt --all --check`.
