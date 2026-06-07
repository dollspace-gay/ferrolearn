//! Automatic Relevance Determination (ARD) Regression.
//!
//! This module provides [`ARDRegression`], a Bayesian linear regression model
//! with per-feature weight precision priors. Features whose precision
//! (`lambda_i`) exceeds a threshold are pruned — their weights are driven to
//! zero, achieving automatic feature selection.
//!
//! # Algorithm
//!
//! Initialisation seeds `alpha = 1/(Var(y)+eps)` and `lambda_i = 1` for every
//! feature, with all features kept (`keep_lambda = lambda_ < threshold_lambda`,
//! initially all-true). Each iteration solves only the KEPT columns
//! `Xk = X[:, keep_lambda]` and updates, including the Gamma hyperprior terms:
//!
//! 1. Posterior covariance of the kept block:
//!    `Sigma = (diag(lambda[keep]) + alpha * Xk^T Xk)^{-1}`,
//!    then `w[keep] = alpha * Sigma @ Xk^T y`, `w[~keep] = 0`.
//! 2. Effective degrees of freedom: `gamma_i = 1 - lambda_i * Sigma_{ii}`.
//! 3. Update lambda: `lambda_i = (gamma_i + 2*lambda_1) / (w_i^2 + 2*lambda_2)`.
//! 4. Update alpha:
//!    `alpha = (n - sum(gamma) + 2*alpha_1) / (||y - Xw||^2 + 2*alpha_2)`.
//! 5. Recompute the mask `keep_lambda = lambda_ < threshold_lambda` and zero the
//!    coefficients of pruned features.
//!
//! Convergence is `sum(|coef_old - coef_|) < tol` (checked after the first
//! iteration). This mirrors scikit-learn's `ARDRegression.fit`
//! (`sklearn/linear_model/_bayes.py:644-730`).
//!
//! ## REQ status (per `.design/linear/ard.md`, mirrors `sklearn/linear_model/_bayes.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.ARDRegression` (`_bayes.py:433`): per-iteration `keep_lambda`
//! column-masking + Gamma hyperpriors + `1/var(y)` init + `threshold_lambda` pruning. coef_/
//! alpha_/lambda_ and the pruned-feature set match the live oracle.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (ARD iterative fit) | SHIPPED | `Fit for ARDRegression` (per-iter masking, hyperpriors, coef-delta convergence); coef_/alpha_/lambda_/pruned-set match oracle (2- and 4-feature cases). Consumer: `RsARDRegression` in `ferrolearn-python`. Mirrors `_bayes.py:644-730`. |
//! | REQ-2 (alpha_=1/(var(y)+eps) init) | SHIPPED | `fit` seeds `1/(var_y+eps)` (`_bayes.py:658`). |
//! | REQ-3 (per-iteration keep_lambda masking + threshold pruning) | SHIPPED | `keep_lambda = lambda_ < threshold_lambda(1e4)` recomputed each iter; pruned coef zeroed (`_bayes.py:691`). |
//! | REQ-4 (predict) | SHIPPED | `Predict for FittedARDRegression`. |
//! | REQ-5 (fit_intercept / HasCoefficients) | SHIPPED | centering + `HasCoefficients`. |
//! | REQ-6 (compute_score / scores_) | SHIPPED | `with_compute_score` on `ARDRegression` (default `false`, `_bayes.py:587`); when set, `fn fit` appends the exact ARD objective (`_bayes.py:695-704`: `sum(λ1·log λ − λ2·λ) + α1·log α − α2·α + 0.5·(fast_logdet(σ) + n·log α + sum log λ) − 0.5·(α·rmse + sum(λ·coef²))`, `fast_logdet` via `fn logdet_spd`) per iteration — appended BEFORE the convergence break, so `scores.len() == n_iter` (NO post-loop append, unlike `BayesianRidge`: no `scores_[-1]` aliasing quirk). Getter `fn scores`. Consumer: `RsARDRegression::scores_` (`extras.rs`) → `_extras.py::ARDRegression.scores_`. Verified by `divergence_ard_scores.rs` (Rust) + `test_ard_scores_matches_sklearn` (pytest) vs live sklearn. |
//! | REQ-7 (n_iter_) | SHIPPED | `FittedARDRegression.n_iter` set to `last_iter + 1` in `fn fit` (`_bayes.py:716` `self.n_iter_ = iter_ + 1`); getter `fn n_iter`. Consumer: `RsARDRegression::n_iter_` (`extras.rs`) → `_extras.py::ARDRegression.n_iter_`. Verified by `divergence_ard_scores.rs` (each case asserts `n_iter()` == the live oracle: 3/6/4) + `test_ard_n_iter_matches_sklearn` (pytest). |
//! | REQ-8 (predict return_std / full sigma_) | SHIPPED | `FittedARDRegression.sigma_full` is the kept-feature `(n_kept, n_kept)` posterior covariance (sklearn `sigma_`, `_bayes.py:727`; empty `(0,0)` if all pruned) with `keep_lambda` mask back to full feature space; getters `fn sigma_full`/`fn keep_lambda`. `fn predict_with_std` returns `(mean, sqrt((Xk·σ·Xk).sum(axis=1) + 1/α))` over KEPT cols only (`_bayes.py:787-790`). Consumer: `RsARDRegression::predict(return_std=True)` + `sigma_` getter (`extras.rs`) → `_extras.py::ARDRegression.predict`/`sigma_`. Verified by `divergence_ard_return_std.rs` (Rust) + `test_ard_return_std_matches_sklearn`/`_sigma_matches_sklearn` (pytest). |
//! | REQ-8b (n<p Woodbury branch: structural + observable contract) | SHIPPED | `n_samples < n_features` selects `fn update_sigma_woodbury` (sklearn `_update_sigma_woodbury`, `_bayes.py:670-674`, `:732-748`): inverts the well-conditioned `(n,n)` `eye/alpha + (Xk·invλ)·Xkᵀ` via `fn invert_dense` (`ferray::linalg::inv`) instead of the rank-deficient `(p,p)` Gram block. Constant-y all-pruned (`intercept_=mean(y)`, coef 0, no panic) and recoverable-sparse cases match the live oracle (same kept set, coef within eigensolver-backend tolerance). Verified by `divergence_ard_woodbury.rs` (Rust) + `divergence_ard_woodbury.py` (pytest) vs live sklearn 1.5.2. |
//! | REQ-8c (n<p EXACT bit-parity on chaotic ill-conditioned trajectories) | NOT-STARTED | Blocked on ferray `scipy.linalg.pinvh` primitive (#2165, R-SUBSTRATE-5): sklearn's `_update_sigma_woodbury` inverts `A` with `pinvh` (LAPACK `syev` + eigenvalue cutoff `max|λ|·N·eps`); ferray exposes only an LU `inv`. On cond~2e8 EM trajectories even numpy's `eigh` differs from scipy's `pinvh` (~1.67), so exact `n_iter_`/coef parity in n<p is genuinely substrate-blocked. `fn update_sigma_woodbury` carries a minimal `pinvh`-cutoff floor on the singular-inverse retry (constant-y path) but does NOT reproduce the full eigenvalue-cutoff trajectory. The OBSERVABLE contract is SHIPPED (REQ-8b); only the exact-bit-chaotic tail is deferred. |
//! | REQ-9 (array-type ferray substrate) | NOT-STARTED | #480; the kept-Gram inverse already on `ferray::linalg::inv`, the array type is still `ndarray`. |
//!
//! acto-critic + builder: the fit was rewritten to sklearn's per-iteration ARD (was wrongly
//! pruning relevant features); coef_/alpha_/lambda_/pruned-set now match the live oracle. The
//! kept-block inverse runs on `ferray::linalg::inv` (R-SUBSTRATE). Two states only per R-DEFER-2.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::ard::ARDRegression;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 2), vec![
//!     1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0,
//! ]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let model = ARDRegression::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 5);
//! ```

use ferray::linalg::{LinalgFloat, inv};
use ferray::{Array as FerrayArray, Ix2 as FerrayIx2};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// Automatic Relevance Determination Regression.
///
/// Bayesian linear regression with per-feature precision priors. Features
/// with high precision (small variance) are pruned, achieving sparsity.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct ARDRegression<F> {
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative change in alpha/lambda.
    pub tol: F,
    /// Shape hyperparameter for the alpha (noise precision) Gamma prior.
    pub alpha_1: F,
    /// Rate hyperparameter for the alpha (noise precision) Gamma prior.
    pub alpha_2: F,
    /// Shape hyperparameter for the lambda (weight precision) Gamma prior.
    pub lambda_1: F,
    /// Rate hyperparameter for the lambda (weight precision) Gamma prior.
    pub lambda_2: F,
    /// Features with `lambda_i > threshold_lambda` are pruned.
    pub threshold_lambda: F,
    /// If `true`, compute the ARD objective (log marginal likelihood) at each
    /// EM iteration into `scores_` (sklearn `compute_score`, default `false`,
    /// `_bayes.py:587`).
    pub compute_score: bool,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> ARDRegression<F> {
    /// Create a new `ARDRegression` with default settings.
    ///
    /// Defaults: `max_iter = 300`, `tol = 1e-3`, `alpha_1 = alpha_2 = 1e-6`,
    /// `lambda_1 = lambda_2 = 1e-6`, `threshold_lambda = 1e4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 300,
            tol: F::from(1e-3).unwrap(),
            alpha_1: F::from(1e-6).unwrap(),
            alpha_2: F::from(1e-6).unwrap(),
            lambda_1: F::from(1e-6).unwrap(),
            lambda_2: F::from(1e-6).unwrap(),
            threshold_lambda: F::from(1e4).unwrap(),
            compute_score: false,
            fit_intercept: true,
        }
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the alpha shape hyperparameter.
    #[must_use]
    pub fn with_alpha_1(mut self, alpha_1: F) -> Self {
        self.alpha_1 = alpha_1;
        self
    }

    /// Set the alpha rate hyperparameter.
    #[must_use]
    pub fn with_alpha_2(mut self, alpha_2: F) -> Self {
        self.alpha_2 = alpha_2;
        self
    }

    /// Set the lambda shape hyperparameter.
    #[must_use]
    pub fn with_lambda_1(mut self, lambda_1: F) -> Self {
        self.lambda_1 = lambda_1;
        self
    }

    /// Set the lambda rate hyperparameter.
    #[must_use]
    pub fn with_lambda_2(mut self, lambda_2: F) -> Self {
        self.lambda_2 = lambda_2;
        self
    }

    /// Set the pruning threshold for feature lambda values.
    #[must_use]
    pub fn with_threshold_lambda(mut self, threshold_lambda: F) -> Self {
        self.threshold_lambda = threshold_lambda;
        self
    }

    /// Set whether to compute the ARD objective (log marginal likelihood) at
    /// each EM iteration (sklearn `compute_score`, `_bayes.py:587`). When
    /// `true`, the converged model's [`FittedARDRegression::scores`] holds the
    /// per-iteration objective sequence (length `n_iter_`); when `false` it is
    /// empty.
    #[must_use]
    pub fn with_compute_score(mut self, compute_score: bool) -> Self {
        self.compute_score = compute_score;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for ARDRegression<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted ARD Regression model.
///
/// Stores the posterior mean coefficients, intercept, estimated noise
/// precision (`alpha`), per-feature weight precisions (`lambda`), and
/// the diagonal of the posterior covariance.
#[derive(Debug, Clone)]
pub struct FittedARDRegression<F> {
    /// Posterior mean coefficient vector.
    coefficients: Array1<F>,
    /// Intercept (bias) term.
    intercept: F,
    /// Estimated noise precision (1 / noise_variance).
    alpha: F,
    /// Per-feature weight precisions.
    lambda: Array1<F>,
    /// Diagonal of the posterior covariance matrix (over the full feature
    /// index; pruned features carry 0).
    sigma: Array1<F>,
    /// Full kept-feature posterior covariance matrix `(n_kept, n_kept)`,
    /// mirroring sklearn's `sigma_` (`_bayes.py:727`). Empty `(0, 0)` if all
    /// features are pruned.
    sigma_full: Array2<F>,
    /// Mask of surviving (kept) features, `keep_lambda = lambda_ <
    /// threshold_lambda` at convergence (`_bayes.py:691`). Maps the rows/cols of
    /// [`Self::sigma_full`] back to the full feature space.
    keep_lambda: Vec<bool>,
    /// Actual number of EM iterations run, mirroring sklearn's `n_iter_`
    /// (`_bayes.py:716`, `iter_ + 1`).
    n_iter: usize,
    /// Per-iteration ARD objective (log marginal likelihood), mirroring
    /// sklearn's `scores_` (`_bayes.py:695-704`). Empty unless `compute_score`
    /// was set; otherwise length `n_iter`.
    scores: Vec<F>,
}

impl<F: Float> FittedARDRegression<F> {
    /// Returns the estimated noise precision (alpha = 1/sigma^2_noise).
    #[must_use]
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// Returns the per-feature weight precisions.
    #[must_use]
    pub fn lambda(&self) -> &Array1<F> {
        &self.lambda
    }

    /// Returns the diagonal of the posterior covariance matrix.
    #[must_use]
    pub fn sigma(&self) -> &Array1<F> {
        &self.sigma
    }

    /// Returns the full kept-feature posterior covariance matrix
    /// `(n_kept, n_kept)` (sklearn `sigma_`, `_bayes.py:727`). Empty `(0, 0)`
    /// if every feature was pruned.
    #[must_use]
    pub fn sigma_full(&self) -> &Array2<F> {
        &self.sigma_full
    }

    /// Returns the kept-feature mask (`keep_lambda = lambda_ <
    /// threshold_lambda` at convergence, `_bayes.py:691`): index `i` is `true`
    /// iff feature `i` survived pruning. The `true` positions index the
    /// rows/cols of [`Self::sigma_full`].
    #[must_use]
    pub fn keep_lambda(&self) -> &[bool] {
        &self.keep_lambda
    }

    /// Returns the actual number of EM iterations run to reach the stopping
    /// criterion (sklearn `n_iter_`, `_bayes.py:716`).
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Returns the per-iteration ARD objective (log marginal likelihood)
    /// sequence (sklearn `scores_`, `_bayes.py:695-704`). Empty unless the
    /// model was built with `with_compute_score(true)`; otherwise of length
    /// [`Self::n_iter`].
    #[must_use]
    pub fn scores(&self) -> &[F] {
        &self.scores
    }
}

/// Dense inverse of a square matrix on the ferray substrate
/// (`ferray::linalg::inv`), bridging `ndarray -> ferray -> ndarray` at this
/// boundary (R-SUBSTRATE-4). Returns the ferray error (including the singular-
/// matrix signal) so callers can decide whether to retry.
fn invert_dense<F: LinalgFloat>(m: &Array2<F>) -> Result<Array2<F>, FerroError> {
    let n = m.nrows();
    let flat: Vec<F> = m.iter().copied().collect();
    let a = FerrayArray::<F, FerrayIx2>::from_vec(FerrayIx2::new([n, n]), flat).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray inv: failed to build matrix: {e}"),
        }
    })?;
    let inv_f = inv(&a).map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray inv failed: {e}"),
    })?;
    Array2::from_shape_vec((n, n), inv_f.iter().copied().collect()).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray inv: shape conversion failed: {e}"),
        }
    })
}

/// Posterior covariance of the kept feature block, mirroring scikit-learn's
/// `ARDRegression._update_sigma` (`sklearn/linear_model/_bayes.py:750-759`):
///
/// ```text
/// gram      = Xk^T @ Xk
/// sigma_inv = diag(lambda[keep]) + alpha * gram
/// Sigma     = pinvh(sigma_inv)
/// ```
///
/// where `Xk = X[:, keep]`. The `(k_keep, k_keep)` inverse runs on the ferray
/// linear-algebra substrate (`ferray::linalg::inv`, `ferray-linalg/src/solve.rs:367`)
/// — for the symmetric positive-definite `sigma_inv` (`n_samples >= n_features`,
/// the `_update_sigma` regime) the LU inverse matches scipy's `pinvh`. The
/// `ndarray ↔ ferray` conversion happens at this boundary (R-SUBSTRATE-4); the
/// caller keeps its `ndarray` signature during the workspace-wide migration.
///
/// Returns the full `(k_keep, k_keep)` posterior covariance `Sigma`.
fn update_sigma<F: LinalgFloat>(
    xk: &Array2<F>,
    alpha: F,
    lambda_keep: &[F],
) -> Result<Array2<F>, FerroError> {
    let k = xk.ncols();
    // sigma_inv = diag(lambda[keep]) + alpha * Xk^T Xk.
    let mut sigma_inv = xk.t().dot(xk);
    for i in 0..k {
        for j in 0..k {
            sigma_inv[[i, j]] *= alpha;
        }
        sigma_inv[[i, i]] += lambda_keep[i];
    }

    // Bridge ndarray -> ferray (R-SUBSTRATE-4).
    let flat: Vec<F> = sigma_inv.iter().copied().collect();
    let a = FerrayArray::<F, FerrayIx2>::from_vec(FerrayIx2::new([k, k]), flat).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray inv: failed to build sigma_inv: {e}"),
        }
    })?;
    let sigma_f = inv(&a).map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray inv failed (ARD sigma): {e}"),
    })?;

    // Bridge ferray -> ndarray.
    let sigma = Array2::from_shape_vec((k, k), sigma_f.iter().copied().collect()).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray inv: Sigma shape conversion failed: {e}"),
        }
    })?;
    Ok(sigma)
}

/// Posterior covariance of the kept feature block in the `n_samples <
/// n_features` regime, mirroring scikit-learn's
/// `ARDRegression._update_sigma_woodbury` (`sklearn/linear_model/_bayes.py:732-748`):
///
/// ```text
/// X_keep     = X[:, keep]                                     # (n, k_keep)
/// inv_lambda = 1 / lambda_[keep]                              # len k_keep
/// A          = eye(n)/alpha + (X_keep .* inv_lambda) @ X_keep^T   # (n, n)
/// inv_A      = pinvh(A)
/// S          = inv_A @ (X_keep .* inv_lambda)                 # (n, k_keep)
/// sigma      = -((X_keep^T .* inv_lambda_rows) @ S)           # (k_keep, k_keep)
/// sigma[j,j] += inv_lambda[j]
/// ```
///
/// When `n_samples < n_features` the direct Gram block `diag(lambda) + alpha *
/// Xk^T Xk` is `(k_keep, k_keep)` with `rank(Xk^T Xk) <= n_samples < k_keep`, so
/// it is rank-deficient and the direct inverse diverges. The Woodbury identity
/// inverts the WELL-CONDITIONED `(n, n)` matrix `A = eye(n)/alpha + ...` (SPD,
/// never singular) instead. The `(n, n)` inverse runs on the SAME ferray
/// substrate as the direct path (`ferray::linalg::inv`). The empty-kept edge
/// (`k_keep == 0`) returns the empty `(0, 0)` covariance.
///
/// NOTE (R-SUBSTRATE-5, #2165): sklearn's `_update_sigma_woodbury` inverts `A`
/// with scipy's `pinvh` (LAPACK `syev` symmetric eigendecomposition + an
/// eigenvalue cutoff `max|λ|·N·eps`); ferray exposes only an LU `inv`. The two
/// agree to machine precision on the WELL-CONDITIONED `A` (the structural
/// contract this function delivers), but on chaotic ill-conditioned EM
/// trajectories the eigensolver-backend difference can amplify — exact bit
/// parity there is blocked on the ferray `pinvh` primitive (#2165).
///
/// Returns the full `(k_keep, k_keep)` posterior covariance `Sigma`.
fn update_sigma_woodbury<F: LinalgFloat>(
    xk: &Array2<F>,
    alpha: F,
    lambda_keep: &[F],
) -> Result<Array2<F>, FerroError> {
    let n_samples = xk.nrows();
    let k = xk.ncols();
    let one = <F as num_traits::One>::one();

    // Empty-kept edge: all features pruned -> empty (0, 0) covariance.
    if k == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    let inv_lambda: Vec<F> = lambda_keep.iter().map(|&l| one / l).collect();

    // X_keep .* inv_lambda (broadcast over columns): (n, k).
    let mut xk_scaled = xk.clone();
    for col in 0..k {
        let s = inv_lambda[col];
        for row in 0..n_samples {
            xk_scaled[[row, col]] *= s;
        }
    }

    // A = eye(n)/alpha + (X_keep .* inv_lambda) @ X_keep^T  (n, n), SPD.
    let mut a_mat = xk_scaled.dot(&xk.t());
    let inv_alpha = one / alpha;
    for i in 0..n_samples {
        a_mat[[i, i]] += inv_alpha;
    }

    // inv_A = inverse(A) on the ferray substrate (R-SUBSTRATE-4). A is SPD and
    // well-conditioned for the generic n<p design, where the LU `inv` matches
    // scipy's `pinvh` to machine precision and the fallback below is NEVER
    // taken (keeping those trajectories byte-identical).
    //
    // pinvh shim (#2165, R-SUBSTRATE-5): when alpha is astronomically large
    // (the constant-y / zero-variance signature) the `eye(n)/alpha` term
    // vanishes below rounding and A collapses to the rank-`<= n_samples`
    // outer-product `(X_keep .* inv_lambda) @ X_keep^T`, so the LU `inv`
    // reports a singular matrix. scipy's `pinvh` tolerates this by zeroing
    // eigenvalues below `max|λ|·N·eps`; ferray has no eigensolver yet (#2165).
    // We mirror that cutoff with the minimal equivalent — a diagonal floor of
    // `max|A_ij|·N·eps` added ONLY on the singular-inverse retry — which
    // reproduces sklearn's all-pruned trajectory (kept 10->2->0) on the
    // constant-y case without perturbing any well-conditioned design.
    let inv_a = match invert_dense(&a_mat) {
        Ok(m) => m,
        Err(_) => {
            let eps = <F as Float>::epsilon();
            let n_f = <F as num_traits::NumCast>::from(n_samples)
                .unwrap_or_else(<F as num_traits::One>::one);
            let max_abs = a_mat.iter().fold(<F as num_traits::Zero>::zero(), |m, &v| {
                let a = v.abs();
                if a > m { a } else { m }
            });
            let floor = max_abs * n_f * eps;
            let mut a_reg = a_mat.clone();
            for i in 0..n_samples {
                a_reg[[i, i]] += floor;
            }
            invert_dense(&a_reg).map_err(|e| FerroError::NumericalInstability {
                message: format!("ferray inv failed (ARD Woodbury A, post pinvh-floor): {e}"),
            })?
        }
    };

    // S = inv_A @ (X_keep .* inv_lambda)  (n, k).
    let s_mat = inv_a.dot(&xk_scaled);

    // sigma = -((X_keep^T .* inv_lambda_rows) @ S)  (k, k);
    // (X_keep^T .* inv_lambda_rows) scales row j of X_keep^T by inv_lambda[j],
    // i.e. column j of X_keep by inv_lambda[j] -> that is exactly xk_scaled^T.
    let mut sigma = xk_scaled.t().dot(&s_mat).mapv(|v| -v);

    // sigma[j, j] += inv_lambda[j]  (_bayes.py:747).
    for j in 0..k {
        sigma[[j, j]] += inv_lambda[j];
    }

    Ok(sigma)
}

/// Natural log of the determinant of the symmetric positive-definite kept-block
/// covariance `sigma`, mirroring scikit-learn's `fast_logdet(sigma_)`
/// (`sklearn/utils/extmath.py:93-130`, `sign, ld = slogdet(A); return ld if
/// sign > 0 else -inf`) as used in the ARD objective (`_bayes.py:699`).
///
/// `sigma` is the posterior covariance over the kept features — it is symmetric
/// positive-definite in the `_update_sigma` (`n_samples >= n_features`) regime,
/// so `det(sigma) > 0` and `slogdet` returns `+1` sign. We compute the
/// log-determinant via an unpivoted LU factorization (Doolittle): for an SPD
/// matrix the diagonal pivots are strictly positive and `log det = sum(log
/// u_ii)`. Returns `-inf` if a non-positive pivot is encountered (the `sign <=
/// 0` branch of `fast_logdet`).
fn logdet_spd<F: Float>(sigma: &Array2<F>) -> F {
    let n = sigma.nrows();
    if n == 0 {
        // slogdet of a 0x0 matrix is (1.0, 0.0): det == 1, log det == 0.
        return F::zero();
    }
    // Copy into a working LU buffer (unpivoted Doolittle elimination).
    let mut a = sigma.clone();
    let mut logdet = F::zero();
    for k in 0..n {
        let pivot = a[[k, k]];
        // A non-positive OR NaN pivot is the `sign <= 0` branch of
        // `fast_logdet` (`extmath.py:128`): `partial_cmp(..) != Some(Greater)`
        // is `true` for `pivot <= 0` AND for `NaN` (incomparable), matching
        // `not (sign > 0)`.
        if pivot.partial_cmp(&F::zero()) != Some(core::cmp::Ordering::Greater) {
            return F::neg_infinity();
        }
        logdet = logdet + pivot.ln();
        for i in (k + 1)..n {
            let factor = a[[i, k]] / pivot;
            for j in (k + 1)..n {
                let v = a[[i, j]] - factor * a[[k, j]];
                a[[i, j]] = v;
            }
        }
    }
    logdet
}

impl<F: LinalgFloat + Send + Sync + ScalarOperand + FromPrimitive> Fit<Array2<F>, Array1<F>>
    for ARDRegression<F>
{
    type Fitted = FittedARDRegression<F>;
    type Error = FerroError;

    /// Fit the ARD model via iterative evidence maximization with per-iteration
    /// `keep_lambda` column masking, mirroring scikit-learn's `ARDRegression.fit`
    /// (`sklearn/linear_model/_bayes.py:644-730`).
    ///
    /// After centering (when `fit_intercept`), `alpha` is seeded to
    /// `1/(Var(y)+eps)` (`_bayes.py:658`) and `lambda` to ones (`_bayes.py:659`)
    /// with all features kept. Each iteration solves only the kept sub-block via
    /// [`update_sigma`] (`_bayes.py:677-678`, `:750-759`), updates the Gamma-prior
    /// `lambda`/`alpha` (`_bayes.py:681-688`), recomputes
    /// `keep_lambda = lambda_ < threshold_lambda` and zeros pruned coefficients
    /// (`_bayes.py:691-692`), and converges on
    /// `sum(|coef_old - coef_|) < tol` (`_bayes.py:707`).
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 samples.
    /// - [`FerroError::NumericalInstability`] — numerical failure in solver.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedARDRegression<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "ARDRegression requires at least 2 samples".into(),
            });
        }

        let zero = <F as num_traits::Zero>::zero();
        let one = <F as num_traits::One>::one();
        let n_f = <F as num_traits::NumCast>::from(n_samples).unwrap_or(one);
        let two = one + one;

        // Center data for intercept (_bayes.py:637 `_preprocess_data`).
        let (x_work, y_work, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute column means".into(),
                })?;
            let y_mean = y.mean().ok_or_else(|| FerroError::NumericalInstability {
                message: "failed to compute target mean".into(),
            })?;
            let x_c = x - &x_mean;
            let y_c = y - y_mean;
            (x_c, y_c, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        // Init: alpha = 1/(Var(y)+eps), lambda = ones, coef = zeros, all kept
        // (_bayes.py:645, :658-659). `eps = finfo(f64).eps` matches sklearn,
        // which fixes float64 eps regardless of dtype.
        let eps =
            <F as num_traits::NumCast>::from(f64::EPSILON).unwrap_or_else(<F as Float>::epsilon);
        let var_y = {
            let ym = y_work
                .mean()
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute target mean for variance".into(),
                })?;
            let centered = &y_work - ym;
            centered.dot(&centered) / n_f
        };
        let mut alpha = one / (var_y + eps);
        let mut lambda = Array1::<F>::from_elem(n_features, one);
        let mut keep_lambda: Vec<bool> = vec![true; n_features];

        // Branch selection (`_bayes.py:670-674`): when `n_samples < n_features`
        // (ORIGINAL full feature count, before any pruning) the direct Gram
        // block is rank-deficient, so sklearn switches to the Woodbury update
        // (`_update_sigma_woodbury`), inverting the well-conditioned `(n, n)`
        // matrix. Otherwise it uses the direct `_update_sigma`. Selected once,
        // matching sklearn (which selects it before the loop and never revisits
        // it, even as `keep_lambda` shrinks `n_kept` below `n_samples`).
        let use_woodbury = n_samples < n_features;
        let solve_sigma = |xk: &Array2<F>, alpha: F, lambda_keep: &[F]| {
            if use_woodbury {
                update_sigma_woodbury(xk, alpha, lambda_keep)
            } else {
                update_sigma(xk, alpha, lambda_keep)
            }
        };

        let mut coef = Array1::<F>::zeros(n_features);
        let mut coef_old: Option<Array1<F>> = None;
        // Diagonal of the posterior covariance over the full feature index
        // (pruned features carry 0); kept entries filled from `Sigma` each iter.
        let mut sigma_diag = Array1::<F>::zeros(n_features);

        let half = <F as num_traits::NumCast>::from(0.5).unwrap_or(one / two);
        // Per-iteration ARD objective (`_bayes.py:695-704`); empty unless
        // `compute_score`. sklearn appends ONE score per iteration (inside the
        // loop, BEFORE the convergence break) and does NOT recompute it after
        // the loop, so `len(scores_) == n_iter_` — unlike `BayesianRidge`,
        // there is no post-loop `scores_[-1]` coef-aliasing quirk.
        let mut scores: Vec<F> = Vec::new();
        // `n_iter_ = iter_ + 1` (`_bayes.py:716`). The loop runs at least once.
        let mut last_iter: usize = 0;

        for iter_ in 0..self.max_iter {
            last_iter = iter_;
            // Indices of kept columns.
            let kept: Vec<usize> = (0..n_features).filter(|&i| keep_lambda[i]).collect();
            let k = kept.len();

            // Xk = X[:, keep_lambda].
            let mut xk = Array2::<F>::zeros((n_samples, k));
            for (col, &i) in kept.iter().enumerate() {
                for row in 0..n_samples {
                    xk[[row, col]] = x_work[[row, i]];
                }
            }
            let lambda_keep: Vec<F> = kept.iter().map(|&i| lambda[i]).collect();

            // sigma_ = update_sigma(...) — Woodbury or direct per the n<p branch
            // selected before the loop (_bayes.py:677, :670-674).
            let sigma = solve_sigma(&xk, alpha, &lambda_keep)?;

            // coef_[keep] = alpha * sigma_ @ Xk^T @ y; coef_[~keep] = 0
            // (_bayes.py:665-667, the running zeros from the prior mask).
            let xkt_y = xk.t().dot(&y_work);
            let coef_keep = sigma.dot(&xkt_y).mapv(|v| v * alpha);
            sigma_diag.fill(zero);
            for (col, &i) in kept.iter().enumerate() {
                coef[i] = coef_keep[col];
                sigma_diag[i] = sigma[[col, col]];
            }

            // rmse_ = sum((y - X @ coef_)^2)  (_bayes.py:681).
            let residual = &y_work - x_work.dot(&coef);
            let rmse = residual.dot(&residual);

            // gamma_ = 1 - lambda[keep] * diag(sigma_)  (_bayes.py:682).
            let mut gamma_sum = zero;
            let mut gamma_keep = vec![zero; k];
            for (col, &i) in kept.iter().enumerate() {
                let g = one - lambda[i] * sigma[[col, col]];
                gamma_keep[col] = g;
                gamma_sum += g;
            }

            // lambda[keep] = (gamma_ + 2*lambda_1) / (coef_[keep]^2 + 2*lambda_2)
            // (_bayes.py:683-685).
            for (col, &i) in kept.iter().enumerate() {
                let ci = coef[i];
                lambda[i] =
                    (gamma_keep[col] + two * self.lambda_1) / (ci * ci + two * self.lambda_2);
            }

            // alpha_ = (n - gamma.sum() + 2*alpha_1) / (rmse_ + 2*alpha_2)
            // (_bayes.py:686-688).
            alpha = (n_f - gamma_sum + two * self.alpha_1) / (rmse + two * self.alpha_2);

            // Prune: keep_lambda = lambda_ < threshold; coef_[~keep] = 0
            // (_bayes.py:691-692).
            for i in 0..n_features {
                keep_lambda[i] = lambda[i] < self.threshold_lambda;
                if !keep_lambda[i] {
                    coef[i] = zero;
                }
            }

            // compute_score: the ARD objective (`_bayes.py:695-704`), evaluated
            // with the UPDATED `alpha`/`lambda` and the just-PRUNED `coef`, the
            // PRE-prune `rmse`, and the kept-block `sigma` from the TOP of THIS
            // iteration (`fast_logdet(sigma_)` = log det of the SPD kept
            // covariance, `extmath.py:93-130`):
            //
            //   s  = sum(lambda_1*log(lambda_) - lambda_2*lambda_)        (all features)
            //      + alpha_1*log(alpha_) - alpha_2*alpha_
            //      + 0.5*(logdet(sigma_) + n*log(alpha_) + sum(log(lambda_)))
            //      - 0.5*(alpha_*rmse_ + sum(lambda_*coef_^2))
            if self.compute_score {
                let mut s = zero;
                for i in 0..n_features {
                    s = s + self.lambda_1 * lambda[i].ln() - self.lambda_2 * lambda[i];
                }
                s = s + self.alpha_1 * alpha.ln() - self.alpha_2 * alpha;
                let sum_log_lambda: F = (0..n_features)
                    .map(|i| lambda[i].ln())
                    .fold(zero, |a, b| a + b);
                s += half * (logdet_spd(&sigma) + n_f * alpha.ln() + sum_log_lambda);
                let lambda_coef_sq: F = (0..n_features)
                    .map(|i| lambda[i] * coef[i] * coef[i])
                    .fold(zero, |a, b| a + b);
                s -= half * (alpha * rmse + lambda_coef_sq);
                scores.push(s);
            }

            // Convergence: iter>0 and sum(|coef_old - coef_|) < tol (_bayes.py:707).
            if let Some(prev) = &coef_old {
                let delta: F = (0..n_features)
                    .map(|i| (prev[i] - coef[i]).abs())
                    .fold(zero, |a, b| a + b);
                if delta < self.tol {
                    break;
                }
            }
            coef_old = Some(coef.clone());

            // All features pruned -> stop (_bayes.py:713-714).
            if !keep_lambda.iter().any(|&b| b) {
                break;
            }
        }

        let n_iter = last_iter + 1;

        // Final coef_/sigma_ refresh with the converged params, over the
        // surviving kept set (_bayes.py:718-721). `sigma_` is the full kept
        // `(k, k)` covariance (sklearn `self.sigma_`, `_bayes.py:727`), or the
        // empty `(0, 0)` matrix when every feature is pruned (`_bayes.py:723`:
        // `sigma_ = np.array([]).reshape(0, 0)`).
        let kept: Vec<usize> = (0..n_features).filter(|&i| keep_lambda[i]).collect();
        let k = kept.len();
        let sigma_full: Array2<F> = if k > 0 {
            let mut xk = Array2::<F>::zeros((n_samples, k));
            for (col, &i) in kept.iter().enumerate() {
                for row in 0..n_samples {
                    xk[[row, col]] = x_work[[row, i]];
                }
            }
            let lambda_keep: Vec<F> = kept.iter().map(|&i| lambda[i]).collect();
            let sigma = solve_sigma(&xk, alpha, &lambda_keep)?;
            let xkt_y = xk.t().dot(&y_work);
            let coef_keep = sigma.dot(&xkt_y).mapv(|v| v * alpha);
            coef.fill(zero);
            sigma_diag.fill(zero);
            for (col, &i) in kept.iter().enumerate() {
                coef[i] = coef_keep[col];
                sigma_diag[i] = sigma[[col, col]];
            }
            sigma
        } else {
            coef.fill(zero);
            sigma_diag.fill(zero);
            Array2::<F>::zeros((0, 0))
        };

        // intercept_ = y_offset - X_offset @ coef_ (_bayes.py:729 `_set_intercept`).
        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            *ym - xm.dot(&coef)
        } else {
            zero
        };

        Ok(FittedARDRegression {
            coefficients: coef,
            intercept,
            alpha,
            lambda,
            sigma: sigma_diag,
            sigma_full,
            keep_lambda,
            n_iter,
            scores,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedARDRegression<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values using the posterior mean coefficients.
    ///
    /// Computes `X @ coefficients + intercept`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let preds = x.dot(&self.coefficients) + self.intercept;
        Ok(preds)
    }
}

impl<F: Float + ScalarOperand + 'static> FittedARDRegression<F> {
    /// Predict the posterior mean AND the predictive standard deviation,
    /// mirroring `sklearn.linear_model.ARDRegression.predict(X,
    /// return_std=True)` (`sklearn/linear_model/_bayes.py:761-791`):
    ///
    /// ```text
    /// y_mean = X @ coef_ + intercept_
    /// col_index = lambda_ < threshold_lambda            # the kept-feature mask
    /// Xk = X[:, col_index]                              # kept columns only
    /// y_std = sqrt( (Xk @ sigma_ * Xk).sum(axis=1) + 1/alpha_ )
    /// ```
    ///
    /// The predictive variance uses ONLY the kept columns and the kept-feature
    /// covariance `sigma_` (`_bayes.py:787-790`); pruned features (whose `coef_`
    /// is `0`) contribute nothing. Returns `(y_mean, y_std)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the fitted model.
    pub fn predict_with_std(&self, x: &Array2<F>) -> Result<(Array1<F>, Array1<F>), FerroError> {
        let n_features = x.ncols();
        if n_features != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let y_mean = x.dot(&self.coefficients) + self.intercept;

        // Kept columns (`col_index = lambda_ < threshold_lambda`, `_bayes.py:787`).
        let kept: Vec<usize> = (0..n_features).filter(|&i| self.keep_lambda[i]).collect();
        let k = kept.len();
        let n_samples = x.nrows();
        let inv_alpha = F::one() / self.alpha;

        // Xk = X[:, col_index] (`_bayes.py:788`).
        let mut xk = Array2::<F>::zeros((n_samples, k));
        for (col, &i) in kept.iter().enumerate() {
            for row in 0..n_samples {
                xk[[row, col]] = x[[row, i]];
            }
        }

        // sigmas_squared_data = (Xk @ sigma_ * Xk).sum(axis=1) (`_bayes.py:789`);
        // y_std = sqrt(sigmas_squared_data + 1/alpha_) (`_bayes.py:790`).
        let xs = xk.dot(&self.sigma_full); // (n_samples, k)
        let y_std: Array1<F> = xs
            .outer_iter()
            .zip(xk.outer_iter())
            .map(|(xs_row, xk_row)| {
                let q = xs_row
                    .iter()
                    .zip(xk_row.iter())
                    .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
                (q + inv_alpha).sqrt()
            })
            .collect();

        Ok((y_mean, y_std))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedARDRegression<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for ARDRegression<F>
where
    F: LinalgFloat + FromPrimitive + ScalarOperand + Send + Sync,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedARDRegression<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_default_constructor() {
        let m = ARDRegression::<f64>::new();
        assert_eq!(m.max_iter, 300);
        assert!(m.fit_intercept);
        assert_relative_eq!(m.alpha_1, 1e-6);
    }

    #[test]
    fn test_builder_setters() {
        let m = ARDRegression::<f64>::new()
            .with_max_iter(50)
            .with_tol(1e-6)
            .with_alpha_1(1e-3)
            .with_alpha_2(1e-3)
            .with_lambda_1(1e-3)
            .with_lambda_2(1e-3)
            .with_threshold_lambda(1e5)
            .with_fit_intercept(false);
        assert_eq!(m.max_iter, 50);
        assert!(!m.fit_intercept);
        assert_relative_eq!(m.threshold_lambda, 1e5);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        let result = ARDRegression::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let y = array![1.0];
        let result = ARDRegression::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_fits_linear_data() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();

        // Should recover roughly y = 2x + 1.
        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.5);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1.5);
    }

    #[test]
    fn test_alpha_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        assert!(fitted.alpha() > 0.0);
    }

    #[test]
    fn test_lambda_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        for &v in fitted.lambda().iter() {
            assert!(v > 0.0, "lambda must be positive, got {v}");
        }
    }

    #[test]
    fn test_sigma_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        for &v in fitted.sigma().iter() {
            assert!(v > 0.0, "sigma diagonal must be positive, got {v}");
        }
    }

    #[test]
    fn test_predict_length() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = ARDRegression::<f64>::new()
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sparsity_on_irrelevant_features() {
        // y depends only on x1, x2 is noise-free irrelevant.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0, 5.0, 500.0, 6.0, 600.0,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]; // y = 2 * x1

        let fitted = ARDRegression::<f64>::new()
            .with_max_iter(1000)
            .fit(&x, &y)
            .unwrap();

        // The model should learn that x1 is relevant.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_has_coefficients_length() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.5, 2.0, 1.0, 1.0, 3.0, 0.0, 1.5, 4.0, 1.0, 2.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 3);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = ARDRegression::<f64>::new();
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
