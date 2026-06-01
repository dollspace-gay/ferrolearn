# Ridge Regression with Cross-Validated Alpha

<!--
tier: 3-component
status: draft
baseline-commit: daf8c0aa3635d9dd84b7d416172eefcf39d42d51
upstream-paths:
  - sklearn/linear_model/_ridge.py
-->

## Summary
`ridge_cv.rs` mirrors scikit-learn's `class RidgeCV`
(`sklearn/linear_model/_ridge.py:2487`, base `_BaseRidgeCV` at `:2257`): Ridge regression
that selects the regularization strength `alpha` from a candidate grid by cross-validation,
then refits on the full data with the chosen `alpha`. The ferrolearn implementation exposes
`RidgeCV<F>` / `FittedRidgeCV<F>` and mirrors sklearn's **default** routing: `cv = None`
(the default) uses efficient leave-one-out Generalized Cross-Validation (`_RidgeGCV`,
`_ridge.py:1688`), and `cv = Some(k)` falls back to the **brute-force k-fold** grid-search
path (loop over `alphas`, fit `Ridge::<F>` per fold, score by mean squared error, pick the
lowest, refit). It does NOT yet cover the `scoring`/`gcv_mode` parameters,
`store_cv_results`/`cv_results_`/`best_score_`, `alpha_per_target`/multi-output, the Python
binding, or full migration to the ferray substrate (the GCV decompositions are on ferray;
the k-fold refit reuses `Ridge`/`crate::linalg`).

**Default-path parity (SHIPPED, blocker #397 closed).** sklearn `RidgeCV` defaults to
`cv=None`, which routes through `_RidgeGCV` — efficient leave-one-out Generalized
Cross-Validation (`_ridge.py:2382-2412`). ferrolearn's default (`cv = None`) now matches this:
it centers `X`/`y` (uniform weights), decomposes once via the shape-appropriate mode
(`_check_gcv_mode`, `_ridge.py:1569`: SVD of the design when `n_samples > n_features`, else
eigendecomposition of the Gram matrix `X·Xᵀ`), computes the closed-form LOO squared errors per
alpha (`looe = c / diag(G^-1)`, `_ridge.py:1717`/`:2149`), and selects the alpha minimising the
mean squared LOO error (`scoring=None` → `-squared_errors.mean()`, `_ridge.py:2211`). Only when
`cv = Some(k)` does ferrolearn use brute-force k-fold, mirroring sklearn's `GridSearchCV`
fallback (`_ridge.py:2413-2439`).

## Requirements
- REQ-1 (explicit-cv k-fold path): when `cv = Some(k)` is set, select `alpha` from a candidate
  grid by brute-force k-fold cross-validation (loop over `alphas`, score each, pick the
  minimizer) and refit on the full data with the chosen `alpha`; expose the chosen value as
  `alpha_` (sklearn `GridSearchCV` branch, `_ridge.py:2419`/`:2438`).
- REQ-2: `fit_intercept` (default `true`, sklearn `_ridge.py:2273`) and `alphas`
  (sklearn default `(0.1, 1.0, 10.0)`, `_ridge.py:2271`); reject empty/negative `alphas`.
- REQ-3 (default CV method): match sklearn's **default** `cv=None` path — efficient
  leave-one-out Generalized Cross-Validation via `_RidgeGCV` (`_ridge.py:2382-2412`,
  `_check_gcv_mode` at `:1569`), with both the SVD (`n_samples > n_features`) and eigen
  (`n_samples <= n_features`) modes.
- REQ-4: `cv`, `scoring`, and `gcv_mode` constructor parameters (sklearn `_ridge.py:2275-2276`,
  `_parameter_constraints` at `:2258-2266`); ferrolearn exposes only an integer `cv` fold
  count (sklearn's `cv` is a CV-object; `cv=None` selects the GCV path).
- REQ-5: `store_cv_results` plus the `cv_results_` and `best_score_` fitted attributes
  (sklearn `_ridge.py:2410-2412`, `:2439`).
- REQ-6: `alpha_per_target` (independent `alpha` per output) and multi-output `Y`
  (sklearn `_ridge.py:2401`, `:2416-2417`).
- REQ-7: A `ferrolearn-python` binding (`RsRidgeCV`) exposing `RidgeCV` to CPython, mirroring
  `from sklearn.linear_model import RidgeCV`.
- REQ-8 (substrate): run the owned array/linalg computation on ferray (`ferray-core` array
  type, `ferray::linalg`) rather than `ndarray`.

## Acceptance criteria
- AC-1: For `X = [[1],...,[20]]`, `y = 2*x + 1`, `alphas = [0.001,0.01,0.1,1.0,10.0,100.0]`,
  the chosen `alpha_` is small (a near-perfect linear fit prefers weak regularization); the
  refit `predict(X)` is close to `y`.
- AC-2: Empty `alphas`, negative `alphas`, sample/target mismatch, `cv < 2`, and
  `n_samples < cv` each return `Err`.
- AC-3 (default LOO-GCV parity): ferrolearn's default `RidgeCV` (`cv = None`) matches sklearn's
  default `RidgeCV` `alpha_` and `coef_` within 1e-6 against the live sklearn 1.5.2 oracle,
  across both the SVD mode (`n_samples > n_features`) and the eigen mode
  (`n_samples <= n_features`). Pinned in `tests/divergence_ridge_cv_gcv.rs`
  (`divergence_default_alpha_is_loo_gcv_not_kfold`, `divergence_default_coef_follows_gcv_alpha`,
  `oracle_gcv_eigen_mode_n_le_p_parity`, `oracle_gcv_svd_mode_fine_grid_parity`). Closed #397.
- AC-4: `coefficients().len() == n_features` and `best_alpha()` returns the chosen value.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (explicit-cv k-fold alpha selection + refit + alpha_) | SHIPPED | impl `fn select_alpha_kfold in ridge_cv.rs` (`RidgeCV`, routed from `pub fn fit` when `cv = Some(k)`) loops `self.alphas`, runs k-fold (`kfold_indices`), scores each fold's `Ridge::<F>` fit by `mse`, picks `best_alpha`, then `pub fn fit` refits `Ridge::<F>::new().with_alpha(best_alpha)` on the full data; `pub fn best_alpha in ridge_cv.rs` (`FittedRidgeCV`) exposes it — mirrors sklearn's `GridSearchCV` branch `self.alpha_` (`_ridge.py:2438`). Non-test consumer: `pub use ridge_cv::{FittedRidgeCV, RidgeCV}` in `ferrolearn-linear/src/lib.rs` (crate public API boundary; grandfathered per R-DEFER-1/S5). Verification: `cargo test -p ferrolearn-linear` (`test_ridge_cv_with_cv_sets_kfold`, `test_ridge_cv_no_intercept` with `.with_cv(3)`) PASS. |
| REQ-2 (fit_intercept + alphas grid + guards) | SHIPPED | impl `pub fn fit in ridge_cv.rs` rejects empty `alphas` and negative entries (`FerroError::InvalidParameter`), passes `fit_intercept` into each `Ridge::<F>`; `RidgeCV::new` defaults `alphas = [0.1, 1.0, 10.0]` (sklearn `(0.1, 1.0, 10.0)`, `_ridge.py:2271`) and `fit_intercept = true`; `with_alphas`/`with_fit_intercept` setters present. Non-test consumer: `pub use ... RidgeCV` in `lib.rs`. Verification: `cargo test -p ferrolearn-linear` (`test_ridge_cv_empty_alphas_error`, `test_ridge_cv_negative_alpha_error`, `test_ridge_cv_default_builder`, `test_ridge_cv_no_intercept`) PASS. |
| REQ-3 (default LOO-GCV method) | SHIPPED | Closed #397. impl `fn select_alpha_gcv in ridge_cv.rs` (`RidgeCV`, routed from `pub fn fit` when `cv = None`, the default) centers `X`/`y`, then per `_check_gcv_mode` (`_ridge.py:1569`) calls `fn gcv_scores_svd` when `n_samples > n_features` (mirroring `_svd_decompose_design_matrix`/`_solve_svd_design_matrix`, `_ridge.py:2025`/`:2039`) or `fn gcv_scores_eigen` otherwise (mirroring `_eigen_decompose_gram`/`_solve_eigen_gram`, `_ridge.py:1900`/`:1914`); each computes the closed-form LOO squared errors `(c / G_inverse_diag)^2` (`_ridge.py:2149`, derived from `looe = c / diag(G^-1)`, `_ridge.py:1717`) and selects the alpha minimising their mean (`_score_without_scorer`, `_ridge.py:2211`). Decompositions run on the ferray substrate (`fn svd_u_s` → `ferray::linalg::svd`, `fn eigh_sym` → `ferray::linalg::eigh`, bridged at the boundary per R-SUBSTRATE-4). Non-test consumer: `pub use ridge_cv::{FittedRidgeCV, RidgeCV}` in `ferrolearn-linear/src/lib.rs` (default `RidgeCV::new()` invokes this path). Verification: `cargo test -p ferrolearn-linear --test divergence_ridge_cv_gcv` — `divergence_default_alpha_is_loo_gcv_not_kfold` (alpha_=10.0), `divergence_default_coef_follows_gcv_alpha` (coef_[0]=-1.1826814581815008), `oracle_gcv_eigen_mode_n_le_p_parity` (n<=p, alpha_=10.0), `oracle_gcv_svd_mode_fine_grid_parity` (n>p fine grid, alpha_=12.689610031679221) — all PASS to <1e-6 vs the live sklearn 1.5.2 oracle. |
| REQ-4 (cv / scoring / gcv_mode params) | NOT-STARTED | open prereq blocker #398. The `cv = None` (GCV) vs `cv = Some(k)` (k-fold) routing IS now representable (`cv: Option<usize>` in `ridge_cv.rs`, mirroring sklearn `_ridge.py:2382`), but `scoring` (MSE is hard-coded — GCV uses the negative-MSE objective, k-fold uses `mse`) and `gcv_mode` (the SVD/eigen mode is auto-selected by shape, not user-overridable) remain NOT-STARTED (sklearn `_ridge.py:2261-2263`). |
| REQ-5 (store_cv_results / cv_results_ / best_score_) | NOT-STARTED | open prereq blocker #398. `FittedRidgeCV<F>` stores only `best_alpha`/`coefficients`/`intercept`; no `cv_results_`, no `best_score_`, no `store_cv_results` (sklearn `_ridge.py:2410-2412`, `:2439`). |
| REQ-6 (alpha_per_target / multi-output) | NOT-STARTED | open prereq blocker #399. `Fit` is `Fit<Array2<F>, Array1<F>>` (single-output target only); no `alpha_per_target` (sklearn `_ridge.py:2401`, `:2416-2417`). |
| REQ-7 (ferrolearn-python binding) | NOT-STARTED | open prereq blocker #400. No `RsRidgeCV` in `ferrolearn-python/src/extras.rs` and no `RidgeCV` wrapper in `_extras.py`; only the in-crate `pub use` re-export exists. |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #401. The module computes on `ndarray` (`Array1`/`Array2`) and the in-crate `Ridge`/`crate::linalg`, not `ferray-core`/`ferray::linalg` (R-SUBSTRATE-2). |

## Architecture
`RidgeCV<F>` (fields `alphas: Vec<F>`, `cv: Option<usize>`, `fit_intercept: bool`; builders
`new`/`with_alphas`/`with_cv`/`with_fit_intercept`) implements `Fit<Array2<F>, Array1<F>>`.
`fit` validates shapes and parameters, then routes on `self.cv`, mirroring sklearn
`_BaseRidgeCV.fit` (`_ridge.py:2382`: `if cv is None`):

- **`cv = None` (default → GCV).** `select_alpha_gcv` centers `X`/`y` (uniform weights), then
  picks the decomposition by shape (`_check_gcv_mode`, `_ridge.py:1569`): `gcv_scores_svd`
  when `n_samples > n_features` (thin SVD of the intercept-augmented design via `svd_u_s` →
  `ferray::linalg::svd`; closed-form `w`, `c`, `G_inverse_diag` per `_solve_svd_design_matrix`,
  `_ridge.py:2039`), else `gcv_scores_eigen` (eigendecomposition of the Gram `X·Xᵀ` plus the
  rank-1 intercept term via `eigh_sym` → `ferray::linalg::eigh`; per `_solve_eigen_gram`,
  `_ridge.py:1914`). Both compute the squared LOO errors `(c / G_inverse_diag)²` and the alpha
  minimising their mean is selected. `find_intercept_dim` locates the intercept eigenvector
  (the factor column most aligned with `ones/√n`, `_find_smallest_angle`, `_ridge.py:1579`) so
  its regularization is cancelled.
- **`cv = Some(k)` → k-fold.** `select_alpha_kfold` partitions indices into `k` folds with
  `kfold_indices` (round-robin `i % k`), accumulates per-fold MSE of a fresh `Ridge::<F>` via
  `select_rows`/`select_elements`/`mse`, and keeps the minimizer.

In both branches the minimizing `alpha` is kept as `best_alpha`; the final `Ridge::<F>` is
refit on the full `X`/`y` (mirroring sklearn `self.coef_ = estimator.coef_`, `_ridge.py:2441`),
and its `coefficients`/`intercept` (via `HasCoefficients`) are stored in `FittedRidgeCV<F>`.

`FittedRidgeCV<F>` stores `best_alpha`, `coefficients`, and `intercept`; `pub fn best_alpha`
exposes the selection. `Predict<Array2<F>>` returns `X @ coefficients + intercept`;
`HasCoefficients` exposes `coefficients`/`intercept`.

The GCV decompositions run on the ferray substrate (`ferray::linalg::svd`/`eigh`), bridged to
`ndarray` at the helper boundary (R-SUBSTRATE-4); the k-fold refit still reuses `Ridge` /
`crate::linalg` (tracked by REQ-8/#401).

## Verification
- `cargo test -p ferrolearn-linear --test divergence_ridge_cv_gcv` — `4 passed`:
  `divergence_default_alpha_is_loo_gcv_not_kfold` (default `alpha_ = 10.0`),
  `divergence_default_coef_follows_gcv_alpha` (default `coef_[0] = -1.1826814581815008`),
  `oracle_gcv_eigen_mode_n_le_p_parity` (`n_samples=10 <= n_features=14`, eigen mode,
  `alpha_ = 10.0`, full `coef_`+`intercept_`), `oracle_gcv_svd_mode_fine_grid_parity`
  (`n_samples=60 > n_features=5`, svd mode, 30-point log-spaced grid, interior
  `alpha_ = 12.689610031679221`) — all within 1e-6 of the live sklearn 1.5.2 oracle (R-CHAR-3).
- `cargo test -p ferrolearn-linear` — `ridge_cv::tests` (incl. `test_ridge_cv_default_builder`
  asserting `cv == None`, `test_ridge_cv_with_cv_sets_kfold`, `test_ridge_cv_fit_selects_alpha`,
  the error-path guards, and `test_kfold_indices_coverage`) — all PASS.
- REQ-1, REQ-2, REQ-3 are SHIPPED with the verification above; REQ-4..8 have no green
  verification (no implementation) and each is NOT-STARTED with the blocker cited above.
