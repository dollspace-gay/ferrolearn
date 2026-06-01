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
`RidgeCV<F>` / `FittedRidgeCV<F>` and covers the **brute-force k-fold** grid-search path:
loop over `alphas`, fit `Ridge::<F>` per fold, score by mean squared error, pick the lowest,
refit. It does NOT cover sklearn's **default** efficient leave-one-out Generalized
Cross-Validation (`_RidgeGCV`), the `scoring`/`cv`/`gcv_mode` parameters,
`store_cv_results`/`cv_results_`/`best_score_`, `alpha_per_target`/multi-output, the Python
binding, or the ferray substrate.

**Algorithmic-divergence warning (read before claiming parity).** sklearn `RidgeCV` defaults
to `cv=None`, which routes through `_RidgeGCV` — efficient leave-one-out Generalized
Cross-Validation (`_ridge.py:2382-2412`). Only when `cv` is explicitly set does sklearn fall
back to brute-force `GridSearchCV` over `alphas` (`_ridge.py:2413-2439`). ferrolearn always
uses brute-force k-fold (default `cv=5`). LOO-GCV and k-fold compute different validation
scores, so the **selected `alpha_` can differ from sklearn's default**. This is a real
divergence (blocker #397), not GCV parity.

## Requirements
- REQ-1: Select `alpha` from a candidate grid by cross-validation (loop over `alphas`, score
  each, pick the minimizer) and refit on the full data with the chosen `alpha`; expose the
  chosen value as `alpha_` (sklearn `self.alpha_ = ...`, `_ridge.py:2409`/`:2438`).
- REQ-2: `fit_intercept` (default `true`, sklearn `_ridge.py:2273`) and `alphas`
  (sklearn default `(0.1, 1.0, 10.0)`, `_ridge.py:2271`); reject empty/negative `alphas`.
- REQ-3 (default CV method): match sklearn's **default** `cv=None` path — efficient
  leave-one-out Generalized Cross-Validation via `_RidgeGCV` (`_ridge.py:2382-2412`,
  `_check_gcv_mode` at `:1569`). ferrolearn currently uses brute-force k-fold instead, so the
  selected `alpha_` diverges from sklearn's default.
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
- AC-3 (divergence pin, NOT a parity claim): on `X=[[1],...,[20]]`, `y=2*x+1`, sklearn's
  default `RidgeCV(alphas=[...])` uses LOO-GCV (`alpha_ = 0.001` in the oracle), while
  ferrolearn's default uses k-fold (`cv=5`); the two `alpha_` values are NOT guaranteed equal
  in general — the critic pins the gap.
- AC-4: `coefficients().len() == n_features` and `best_alpha()` returns the chosen value.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (CV alpha selection + refit + alpha_) | SHIPPED | impl `pub fn fit in ridge_cv.rs` (`RidgeCV`) loops `self.alphas`, runs k-fold (`kfold_indices`), scores each fold's `Ridge::<F>` fit by `mse`, picks `best_alpha`, then refits `Ridge::<F>::new().with_alpha(best_alpha)` on the full data; `pub fn best_alpha in ridge_cv.rs` (`FittedRidgeCV`) exposes it — mirrors sklearn `self.alpha_` (`_ridge.py:2438`). Non-test consumer: `pub use ridge_cv::{FittedRidgeCV, RidgeCV}` in `ferrolearn-linear/src/lib.rs` (crate public API boundary; grandfathered per R-DEFER-1/S5). Verification: `cargo test -p ferrolearn-linear` (`test_ridge_cv_fit_selects_alpha` asserts `best_alpha() <= 1.0`) PASS. |
| REQ-2 (fit_intercept + alphas grid + guards) | SHIPPED | impl `pub fn fit in ridge_cv.rs` rejects empty `alphas` and negative entries (`FerroError::InvalidParameter`), passes `fit_intercept` into each `Ridge::<F>`; `RidgeCV::new` defaults `alphas = [0.1, 1.0, 10.0]` (sklearn `(0.1, 1.0, 10.0)`, `_ridge.py:2271`) and `fit_intercept = true`; `with_alphas`/`with_fit_intercept` setters present. Non-test consumer: `pub use ... RidgeCV` in `lib.rs`. Verification: `cargo test -p ferrolearn-linear` (`test_ridge_cv_empty_alphas_error`, `test_ridge_cv_negative_alpha_error`, `test_ridge_cv_default_builder`, `test_ridge_cv_no_intercept`) PASS. |
| REQ-3 (default LOO-GCV method) | NOT-STARTED | open prereq blocker #397. ferrolearn's `pub fn fit in ridge_cv.rs` uses brute-force k-fold (`kfold_indices`, default `cv=5`); sklearn's default `cv=None` routes through `_RidgeGCV` efficient leave-one-out GCV (`_ridge.py:2382-2412`). The selected `alpha_` diverges from sklearn's default. |
| REQ-4 (cv / scoring / gcv_mode params) | NOT-STARTED | open prereq blocker #398. `RidgeCV<F>` exposes only an integer fold count `cv` (>= 2); no `scoring` (MSE is hard-coded in `mse`), no `gcv_mode`, and `cv=None` (the GCV-selecting sentinel) is not representable (sklearn `_ridge.py:2261-2263`). |
| REQ-5 (store_cv_results / cv_results_ / best_score_) | NOT-STARTED | open prereq blocker #398. `FittedRidgeCV<F>` stores only `best_alpha`/`coefficients`/`intercept`; no `cv_results_`, no `best_score_`, no `store_cv_results` (sklearn `_ridge.py:2410-2412`, `:2439`). |
| REQ-6 (alpha_per_target / multi-output) | NOT-STARTED | open prereq blocker #399. `Fit` is `Fit<Array2<F>, Array1<F>>` (single-output target only); no `alpha_per_target` (sklearn `_ridge.py:2401`, `:2416-2417`). |
| REQ-7 (ferrolearn-python binding) | NOT-STARTED | open prereq blocker #400. No `RsRidgeCV` in `ferrolearn-python/src/extras.rs` and no `RidgeCV` wrapper in `_extras.py`; only the in-crate `pub use` re-export exists. |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #401. The module computes on `ndarray` (`Array1`/`Array2`) and the in-crate `Ridge`/`crate::linalg`, not `ferray-core`/`ferray::linalg` (R-SUBSTRATE-2). |

## Architecture
`RidgeCV<F>` (fields `alphas: Vec<F>`, `cv: usize`, `fit_intercept: bool`; builders `new`/
`with_alphas`/`with_cv`/`with_fit_intercept`) implements `Fit<Array2<F>, Array1<F>>`. `fit`
validates shapes and parameters, partitions indices into `self.cv` folds with `kfold_indices`
(round-robin `i % k` assignment — note this is a deterministic non-shuffled split, unlike
sklearn's `KFold` defaults), then for each candidate `alpha` accumulates mean-squared error
over the folds using helper free functions `select_rows`/`select_elements`/`mse`, training a
fresh `Ridge::<F>` per fold. The minimizing `alpha` is kept as `best_alpha`; the final
`Ridge::<F>` is refit on the full `X`/`y`, and its `coefficients`/`intercept` (via
`HasCoefficients`) are stored in `FittedRidgeCV<F>`.

`FittedRidgeCV<F>` stores `best_alpha`, `coefficients`, and `intercept`; `pub fn best_alpha`
exposes the selection. `Predict<Array2<F>>` returns `X @ coefficients + intercept`;
`HasCoefficients` exposes `coefficients`/`intercept`.

The defining structural gap vs sklearn (`_BaseRidgeCV.fit`, `_ridge.py:2290-2447`): sklearn
branches on `cv is None`. The `None` branch (the **default**) builds a `_RidgeGCV` estimator
and fits it once — efficient leave-one-out GCV that reuses a single matrix decomposition
across all `alphas` and produces `alpha_`, `best_score_`, and (optionally) `cv_results_`. The
non-`None` branch wraps `Ridge`/`RidgeClassifier` in `GridSearchCV` (brute-force). ferrolearn
implements only an analog of the brute-force branch, with a fixed integer `cv` and a
hard-coded MSE score — so for the common default invocation, ferrolearn's `alpha_` is computed
by a different estimator than sklearn's and may differ (blocker #397).

## Verification
- `cargo test -p ferrolearn-linear` — `ridge_cv::tests` (`test_ridge_cv_default_builder`,
  `test_ridge_cv_custom_alphas`, `test_ridge_cv_fit_selects_alpha`, `test_ridge_cv_predict`,
  `test_ridge_cv_has_coefficients`, `test_ridge_cv_empty_alphas_error`,
  `test_ridge_cv_negative_alpha_error`, `test_ridge_cv_shape_mismatch`,
  `test_ridge_cv_insufficient_samples`, `test_ridge_cv_cv_too_small`,
  `test_ridge_cv_no_intercept`, `test_ridge_cv_predict_feature_mismatch`,
  `test_kfold_indices_coverage`) — all PASS at baseline. These pin the brute-force k-fold
  contract (REQ-1, REQ-2), NOT sklearn-default GCV parity.
- sklearn oracle (live, 1.5.2), divergence-illustrating, NOT a parity assertion:
  `python3 -c "import numpy as np; from sklearn.linear_model import RidgeCV; X=np.arange(1,21,dtype=float).reshape(-1,1); y=2*np.arange(1,21)+1.0; print('default(LOO-GCV)', RidgeCV(alphas=[0.001,0.01,0.1,1.0,10.0,100.0]).fit(X,y).alpha_); print('cv=5(kfold)', RidgeCV(alphas=[0.001,0.01,0.1,1.0,10.0,100.0],cv=5).fit(X,y).alpha_)"`
  → `default(LOO-GCV) 0.001`, `cv=5(kfold) 0.001` (agree on this clean linear data; they are
  NOT guaranteed to agree in general — REQ-3 is NOT-STARTED).
- REQ-3..8 have no green verification (no implementation); each is NOT-STARTED with the
  blocker cited above.
