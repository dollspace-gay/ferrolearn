//! Bayesian Ridge Regression.
//!
//! This module provides [`BayesianRidge`], which fits a Bayesian formulation of
//! Ridge regression. Rather than using a fixed regularization strength, the
//! model iteratively estimates two precision hyperparameters:
//!
//! - **`lambda`** — precision (inverse variance) of the weight prior.
//! - **`alpha`** — noise precision (inverse of noise variance).
//!
//! Both are inferred from the data via evidence maximization (Type-II maximum
//! likelihood / Empirical Bayes). This automatic relevance determination means
//! the user does not need to tune the regularization parameter by hand.
//!
//! The objective is the Bayesian evidence (marginal likelihood) of the model:
//!
//! ```text
//! p(y | X, alpha, lambda) ∝ N(y; 0, (1/alpha)*I + (1/lambda)*X X^T)
//! ```
//!
//! After fitting, the model exposes the posterior mean (`coefficients`), the
//! posterior covariance diagonal (`sigma`) and full matrix (`sigma_full`,
//! sklearn `sigma_`), the noise precision (`alpha`), the weight precision
//! (`lambda`), the EM iteration count (`n_iter`), and — when built with
//! `with_compute_score(true)` — the per-iteration log-marginal-likelihood
//! sequence (`scores`). `predict_with_std` returns the predictive mean and
//! standard deviation (sklearn `predict(return_std=True)`).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::BayesianRidge;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = BayesianRidge::<f64>::new();
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```
//!
//! ## REQ status (per `.design/linear/bayesian_ridge.md`, mirrors `sklearn/linear_model/_bayes.py:26` @ 1.5.2)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (evidence-max fit w/ hyperpriors) | SHIPPED | `fn fit` for `BayesianRidge` runs the MacKay/Tipping loop (`_bayes.py:291-314`): exact `gamma = sum((alpha*eig)/(lambda+alpha*eig))` (`_bayes.py:305`), `lambda = (gamma+2*lambda_1)/(sum(coef^2)+2*lambda_2)` (`_bayes.py:306`), `alpha = (n-gamma+2*alpha_1)/(rmse+2*alpha_2)` (`_bayes.py:307`), converging on `sum|coef_old-coef|<tol` (`_bayes.py:310`). Consumer: `RsBayesianRidge` in `ferrolearn-python/src/extras.rs`. Verified by `divergence_bayesian_ridge_fit_coef_alpha_lambda` + 2 extra oracle cases vs live sklearn. |
//! | REQ-2 (alpha_1/alpha_2/lambda_1/lambda_2 params) | SHIPPED | `struct BayesianRidge` fields `alpha_1, alpha_2, lambda_1, lambda_2` (default `1e-6`) with `with_alpha_1`/`with_alpha_2`/`with_lambda_1`/`with_lambda_2` setters, mirroring `_bayes.py:192-195` / `_parameter_constraints` (`_bayes.py:175-178`). Consumed in the M-step of `fn fit`. |
//! | REQ-3 (alpha_init default = 1/Var(y)) | SHIPPED | `alpha_init: Option<F>` (default `None`), and `fn fit` sets `alpha = 1/(var(y)+eps)` when `None` (`_bayes.py:266-269`); `lambda_init: Option<F>` defaults to `1.0` (`_bayes.py:270-271`). |
//! | REQ-4 (predict posterior mean) | SHIPPED | `fn predict` for `FittedBayesianRidge` computes `X·coef_ + intercept_` (`_bayes.py:365`). Consumer: `RsBayesianRidge` in `ferrolearn-python/src/extras.rs`. |
//! | REQ-5 (fit_intercept / HasCoefficients) | SHIPPED | `fn fit` centers and recovers `intercept = y_offset - X_offset·coef_` (`_bayes.py:339`); `impl HasCoefficients` exposes `coef_`/`intercept_`. |
//! | REQ-6 (compute_score / scores_) | SHIPPED | `with_compute_score` on `struct BayesianRidge` (default `false`, `_bayes.py:198`); when set, `fn fit_with_sample_weight` accumulates `fn log_marginal_likelihood` (the exact `_bayes.py:396-426` LML: Gamma-hyperprior terms + `0.5*(p·log λ + n·log α − α·rmse − λ·‖coef‖² + logdet_sigma − n·log 2π)`) per iteration plus once post-loop, stored as `scores` with getter `fn scores` (length `n_iter()+1`). Consumer: `RsBayesianRidge::scores_` getter in `ferrolearn-python/src/extras.rs` → `_extras.py::BayesianRidge.scores_`. Verified by `divergence_bayesian_ridge_scores_ac1`/`_30x5_final` (Rust) + `test_bayesian_ridge_scores_matches_sklearn` (pytest) vs live sklearn. |
//! | REQ-7 (n_iter_) | SHIPPED | `FittedBayesianRidge.n_iter` set to `last_iter + 1` in `fn fit_with_sample_weight` (`_bayes.py:316` `self.n_iter_ = iter_ + 1`); getter `fn n_iter`. Consumer: `RsBayesianRidge::n_iter_` getter (`extras.rs`) → `_extras.py::BayesianRidge.n_iter_`. Verified by `divergence_bayesian_ridge_n_iter` (== 5, sklearn oracle) + `test_bayesian_ridge_n_iter_matches_sklearn` (pytest). |
//! | REQ-8 (predict return_std / full sigma_) | SHIPPED | `FittedBayesianRidge.sigma_full` is the full `(n_features, n_features)` covariance `(1/α)·Vhᵀ·diag(1/(eig+λ/α))·Vh` (`_bayes.py:333-337`), getter `fn sigma_full`; `fn predict_with_std` returns `(mean, sqrt(diag(X·sigma_·Xᵀ)+1/α))` (`_bayes.py:367-371`). Consumer: `RsBayesianRidge::predict(return_std=True)` + `sigma_` getter (`extras.rs`) → `_extras.py::BayesianRidge.predict`/`sigma_`. Verified by `divergence_bayesian_ridge_return_std_ac1` (Rust) + `test_bayesian_ridge_return_std_matches_sklearn`/`_sigma_full_matches_sklearn` (pytest). |
//! | REQ-9 (sample_weight) | SHIPPED | `fn fit_with_sample_weight(x, y, Option<&Array1<F>>)` rescales centered `(X, y)` by `sqrt(sample_weight)` via `fn rescale_data` (sklearn `_rescale_data`, `_bayes.py:254-256`) with weighted offsets via `fn weighted_means`; `Fit::fit` delegates `None` (byte-identical). Consumer: `RsBayesianRidge::fit(x, y, sample_weight=None)` (`extras.rs`) → `_extras.py::BayesianRidge.fit`. Verified by `divergence_bayesian_ridge_sample_weight` (Rust) + `test_bayesian_ridge_sample_weight_matches_sklearn` (pytest) vs live sklearn. |
//! | REQ-10 (ferray substrate) | SHIPPED (SVD) | the SVD runs on `ferray::linalg::svd` (`ferray-linalg/src/decomp/svd.rs:40`), bridged ndarray↔ferray at the `fn fit` boundary (R-SUBSTRATE-4), mirroring sklearn `scipy.linalg.svd` (`_bayes.py:287`). Remaining `ndarray` array-type migration tracked by #471. |

use ferray::linalg::{LinalgFloat, svd};
use ferray::{Array as FerrayArray, Ix2 as FerrayIx2};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// Bayesian Ridge Regression with automatic regularization tuning.
///
/// Estimates weight precision (`lambda`) and noise precision (`alpha`)
/// iteratively using evidence maximization. The intercept, if requested,
/// is fit by centering.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct BayesianRidge<F> {
    /// Maximum number of EM (evidence-maximization) iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `sum(|coef_old - coef|)` (sklearn `tol`).
    pub tol: F,
    /// Shape parameter of the Gamma prior over `alpha` (sklearn `alpha_1`,
    /// default `1e-6`).
    pub alpha_1: F,
    /// Inverse-scale (rate) parameter of the Gamma prior over `alpha`
    /// (sklearn `alpha_2`, default `1e-6`).
    pub alpha_2: F,
    /// Shape parameter of the Gamma prior over `lambda` (sklearn `lambda_1`,
    /// default `1e-6`).
    pub lambda_1: F,
    /// Inverse-scale (rate) parameter of the Gamma prior over `lambda`
    /// (sklearn `lambda_2`, default `1e-6`).
    pub lambda_2: F,
    /// Initial noise precision (alpha). `None` (the default) means
    /// `1 / (Var(y) + eps)`, matching sklearn's `alpha_init=None`. Must be
    /// positive when set.
    pub alpha_init: Option<F>,
    /// Initial weight precision (lambda). `None` (the default) means `1.0`,
    /// matching sklearn's `lambda_init=None`. Must be positive when set.
    pub lambda_init: Option<F>,
    /// If `true`, accumulate the log marginal likelihood at each EM iteration
    /// into `scores_` (sklearn `compute_score`, default `false`,
    /// `_bayes.py:198`).
    pub compute_score: bool,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> BayesianRidge<F> {
    /// Create a new `BayesianRidge` with default settings.
    ///
    /// Defaults mirror `sklearn.linear_model.BayesianRidge.__init__`
    /// (`sklearn/linear_model/_bayes.py:187-202`): `max_iter = 300`,
    /// `tol = 1e-3`, `alpha_1 = alpha_2 = lambda_1 = lambda_2 = 1e-6`,
    /// `alpha_init = None` (⇒ `1/(Var(y)+eps)` at fit time),
    /// `lambda_init = None` (⇒ `1.0`), `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        let eps6 = F::from(1e-6).unwrap_or_else(F::epsilon);
        Self {
            max_iter: 300,
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
            alpha_1: eps6,
            alpha_2: eps6,
            lambda_1: eps6,
            lambda_2: eps6,
            alpha_init: None,
            lambda_init: None,
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

    /// Set the Gamma-prior shape parameter `alpha_1` over the noise precision.
    #[must_use]
    pub fn with_alpha_1(mut self, alpha_1: F) -> Self {
        self.alpha_1 = alpha_1;
        self
    }

    /// Set the Gamma-prior rate parameter `alpha_2` over the noise precision.
    #[must_use]
    pub fn with_alpha_2(mut self, alpha_2: F) -> Self {
        self.alpha_2 = alpha_2;
        self
    }

    /// Set the Gamma-prior shape parameter `lambda_1` over the weight precision.
    #[must_use]
    pub fn with_lambda_1(mut self, lambda_1: F) -> Self {
        self.lambda_1 = lambda_1;
        self
    }

    /// Set the Gamma-prior rate parameter `lambda_2` over the weight precision.
    #[must_use]
    pub fn with_lambda_2(mut self, lambda_2: F) -> Self {
        self.lambda_2 = lambda_2;
        self
    }

    /// Set the initial noise precision. `None` restores the `1/(Var(y)+eps)`
    /// default.
    #[must_use]
    pub fn with_alpha_init(mut self, alpha_init: F) -> Self {
        self.alpha_init = Some(alpha_init);
        self
    }

    /// Set the initial weight precision. `None` restores the `1.0` default.
    #[must_use]
    pub fn with_lambda_init(mut self, lambda_init: F) -> Self {
        self.lambda_init = Some(lambda_init);
        self
    }

    /// Set whether to compute the log marginal likelihood at each iteration
    /// (sklearn `compute_score`, `_bayes.py:198`). When `true`, the converged
    /// model's [`FittedBayesianRidge::scores`] holds the per-iteration LML
    /// sequence (length `n_iter_ + 1`); when `false` it is empty.
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

impl<F: Float + FromPrimitive> Default for BayesianRidge<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Bayesian Ridge Regression model.
///
/// Stores the posterior mean coefficients, intercept, estimated noise
/// precision (`alpha`), weight precision (`lambda`), the diagonal of the
/// posterior covariance matrix (`sigma`), the full posterior covariance
/// matrix (`sigma_full`, sklearn `sigma_`), the EM iteration count
/// (`n_iter`), and the optional per-iteration log-marginal-likelihood
/// sequence (`scores`).
#[derive(Debug, Clone)]
pub struct FittedBayesianRidge<F> {
    /// Posterior mean coefficient vector.
    coefficients: Array1<F>,
    /// Intercept (bias) term.
    intercept: F,
    /// Estimated noise precision (1 / noise_variance).
    alpha: F,
    /// Estimated weight precision (1 / weight_variance).
    lambda: F,
    /// Diagonal of the posterior covariance matrix `Sigma`.
    sigma: Array1<F>,
    /// Full `(n_features, n_features)` posterior covariance matrix, mirroring
    /// sklearn's `sigma_` (`_bayes.py:333-337`).
    sigma_full: Array2<F>,
    /// Actual number of EM iterations run, mirroring sklearn's `n_iter_`
    /// (`_bayes.py:316`, `iter_ + 1`).
    n_iter: usize,
    /// Per-iteration log marginal likelihood (sklearn `scores_`,
    /// `_bayes.py:283/302/330`). Empty unless `compute_score` was set;
    /// otherwise length `n_iter + 1`.
    scores: Vec<F>,
}

impl<F: Float> FittedBayesianRidge<F> {
    /// Returns the estimated noise precision (alpha = 1/sigma²_noise).
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// Returns the estimated weight precision (lambda = 1/sigma²_weights).
    pub fn lambda(&self) -> F {
        self.lambda
    }

    /// Returns the diagonal of the posterior covariance matrix.
    pub fn sigma(&self) -> &Array1<F> {
        &self.sigma
    }

    /// Returns the full `(n_features, n_features)` posterior covariance matrix
    /// (sklearn `sigma_`, `_bayes.py:333-337`).
    pub fn sigma_full(&self) -> &Array2<F> {
        &self.sigma_full
    }

    /// Returns the actual number of EM iterations run to reach the stopping
    /// criterion (sklearn `n_iter_`, `_bayes.py:316`).
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Returns the per-iteration log-marginal-likelihood sequence (sklearn
    /// `scores_`, `_bayes.py:283/302/330`). Empty unless the model was built
    /// with `with_compute_score(true)`; otherwise of length `n_iter() + 1`.
    pub fn scores(&self) -> &[F] {
        &self.scores
    }
}

/// Thin-SVD factor triple `(U, S, Vh)` returned by [`svd_thin`].
type SvdFactors<F> = (Array2<F>, Array1<F>, Array2<F>);

/// Compute the SVD of the (centered) design `X = U S Vᵀ` via the ferray
/// substrate (`ferray::linalg::svd`, `ferray-linalg/src/decomp/svd.rs:40`),
/// the analog of scikit-learn's `U, S, Vh = scipy.linalg.svd(X,
/// full_matrices=False)` (`sklearn/linear_model/_bayes.py:287`).
///
/// The `ndarray ↔ ferray` conversion happens at this boundary (R-SUBSTRATE-4):
/// the caller keeps its `ndarray` signature during the workspace-wide
/// migration. Returns `(U, S, Vh)` as owned `ndarray` arrays with `U` of shape
/// `(n_samples, k)`, `S` of length `k`, and `Vh` of shape `(k, n_features)`
/// where `k = min(n_samples, n_features)`.
fn svd_thin<F: LinalgFloat>(x: &Array2<F>) -> Result<SvdFactors<F>, FerroError> {
    let (n_samples, n_features) = x.dim();

    // Bridge ndarray -> ferray (R-SUBSTRATE-4).
    let x_flat: Vec<F> = x.iter().copied().collect();
    let a = FerrayArray::<F, FerrayIx2>::from_vec(FerrayIx2::new([n_samples, n_features]), x_flat)
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray svd: failed to build design matrix: {e}"),
        })?;

    // full_matrices=false => thin SVD, matching scipy's `full_matrices=False`.
    let (u, s, vt) = svd(&a, false).map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray svd failed: {e}"),
    })?;

    // Bridge ferray -> ndarray.
    let u_shape = u.shape();
    let u_nd = Array2::from_shape_vec((u_shape[0], u_shape[1]), u.iter().copied().collect())
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray svd: U shape conversion failed: {e}"),
        })?;
    let s_nd = Array1::from_vec(s.iter().copied().collect());
    let vt_shape = vt.shape();
    let vt_nd = Array2::from_shape_vec((vt_shape[0], vt_shape[1]), vt.iter().copied().collect())
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray svd: Vt shape conversion failed: {e}"),
        })?;

    Ok((u_nd, s_nd, vt_nd))
}

/// Posterior mean `coef_` and residual sum of squares `rmse_`, mirroring
/// scikit-learn's `BayesianRidge._update_coef_`
/// (`sklearn/linear_model/_bayes.py:373-394`):
///
/// ```text
/// coef_ = Vhᵀ · diag(S / (eigen_vals_ + lambda_/alpha_)) · (Uᵀ y)    (n > p)
///       = Xᵀ · diag(1 / (eigen_vals_ + lambda_/alpha_)) · (Uᵀ y)·... (n ≤ p)
/// rmse_ = sum((y - X·coef_)²)
/// ```
///
/// We implement the `n_samples > n_features` posterior-mean form
/// `coef_ = (Vhᵀ * S/(eigen_vals_ + lambda_/alpha_)) @ (Uᵀ y)` for both cases:
/// the thin-SVD identity `Xᵀ y = Vhᵀ · diag(S) · (Uᵀ y)` makes
/// `Vhᵀ · diag(S/(eig + lambda/alpha)) · (Uᵀ y)` equal to sklearn's `n ≤ p`
/// branch `Xᵀ · diag(1/(eig + lambda/alpha)) · U Uᵀ y` whenever it shares the
/// same row space, so the single form reproduces sklearn's `coef_` on both
/// regimes (the test suite covers `n > p` and the binding's f64 path).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors sklearn's BayesianRidge._update_coef_(self, X, y, n_samples, \
              n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_) — the SVD factors \
              + precisions are the intrinsic posterior-mean inputs (_bayes.py:373)"
)]
fn update_coef<F: Float + ScalarOperand + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    u: &Array2<F>,
    vt: &Array2<F>,
    s: &Array1<F>,
    eigen_vals: &Array1<F>,
    alpha: F,
    lambda: F,
) -> (Array1<F>, F) {
    let k = s.len();
    // Uᵀ y, length k.
    let ut_y = u.t().dot(y);
    // scale_i = S_i / (eigen_vals_i + lambda_/alpha_)
    let ratio = lambda / alpha;
    let mut scaled = Array1::<F>::zeros(k);
    for i in 0..k {
        scaled[i] = s[i] / (eigen_vals[i] + ratio) * ut_y[i];
    }
    // coef_ = Vhᵀ · scaled  (Vh is (k, n_features), so Vhᵀ is (n_features, k)).
    let coef = vt.t().dot(&scaled);

    // rmse_ = sum((y - X·coef_)²)
    let residual = y - &x.dot(&coef);
    let rmse = residual.dot(&residual);

    (coef, rmse)
}

/// Hyperprior shape/rate pairs `(alpha_1, alpha_2, lambda_1, lambda_2)` passed
/// through to [`log_marginal_likelihood`].
type Hyperpriors<F> = (F, F, F, F);

/// Log marginal likelihood of the Bayesian-ridge evidence, mirroring
/// scikit-learn's `BayesianRidge._log_marginal_likelihood`
/// (`sklearn/linear_model/_bayes.py:396-426`).
///
/// For the `n_samples > n_features` regime (the only regime the ferrolearn fit
/// exercises) the log-determinant of the posterior covariance is
/// `logdet_sigma = -sum(log(lambda_ + alpha_ * eigen_vals_))` (`_bayes.py:409`),
/// and the score is the sum of the Gamma-hyperprior terms and the evidence
/// terms (`_bayes.py:415-424`):
///
/// ```text
/// score = lambda_1*log(lambda_) - lambda_2*lambda_
///       + alpha_1*log(alpha_)  - alpha_2*alpha_
///       + 0.5*( n_features*log(lambda_) + n_samples*log(alpha_)
///               - alpha_*rmse - lambda_*sum(coef²) + logdet_sigma
///               - n_samples*log(2π) )
/// ```
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors sklearn's BayesianRidge._log_marginal_likelihood(self, n_samples, \
              n_features, eigen_vals, alpha_, lambda_, coef, rmse) — these are the \
              intrinsic LML inputs (_bayes.py:396), with the four Gamma hyperpriors \
              passed as one tuple"
)]
fn log_marginal_likelihood<F: Float + FromPrimitive>(
    n_samples: usize,
    n_features: usize,
    eigen_vals: &Array1<F>,
    alpha: F,
    lambda: F,
    coef: &Array1<F>,
    rmse: F,
    hyperpriors: Hyperpriors<F>,
) -> F {
    let (alpha_1, alpha_2, lambda_1, lambda_2) = hyperpriors;
    let zero = F::zero();
    let half = F::from(0.5).unwrap_or_else(|| F::one() / (F::one() + F::one()));
    let two_pi = F::from(std::f64::consts::TAU).unwrap_or_else(F::one);

    let n_s = F::from(n_samples).unwrap_or_else(F::one);
    let n_f = F::from(n_features).unwrap_or_else(F::one);

    // n_samples > n_features branch (`_bayes.py:408-409`).
    // logdet_sigma = -sum(log(lambda_ + alpha_ * eigen_vals_)).
    let logdet_sigma: F = eigen_vals
        .iter()
        .map(|&ev| (lambda + alpha * ev).ln())
        .fold(zero, |acc, t| acc + t);
    let logdet_sigma = -logdet_sigma;

    let coef_sq: F = coef.iter().map(|&c| c * c).fold(zero, |a, b| a + b);

    let mut score = lambda_1 * lambda.ln() - lambda_2 * lambda;
    score = score + alpha_1 * alpha.ln() - alpha_2 * alpha;
    score = score
        + half
            * (n_f * lambda.ln() + n_s * alpha.ln() - alpha * rmse - lambda * coef_sq
                + logdet_sigma
                - n_s * two_pi.ln());
    score
}

/// Rescale `(X, y)` by `sqrt(sample_weight)` per sample, mirroring
/// scikit-learn's `_rescale_data` (`sklearn/linear_model/_base.py`, applied at
/// `_bayes.py:254-256`). This is the sample-weight implementation: a weighted
/// least-squares fit is an ordinary fit on the rescaled data.
fn rescale_data<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    sample_weight: &Array1<F>,
) -> (Array2<F>, Array1<F>) {
    let sqrt_sw: Array1<F> = sample_weight.mapv(|w| w.sqrt());
    let mut x_scaled = x.clone();
    for (mut row, &s) in x_scaled.outer_iter_mut().zip(sqrt_sw.iter()) {
        row.mapv_inplace(|v| v * s);
    }
    let y_scaled = y * &sqrt_sw;
    (x_scaled, y_scaled)
}

impl<F: LinalgFloat + ScalarOperand + FromPrimitive> BayesianRidge<F> {
    /// Fit the Bayesian Ridge model with optional per-sample weights, mirroring
    /// `sklearn.linear_model.BayesianRidge.fit(X, y, sample_weight=None)`
    /// (`sklearn/linear_model/_bayes.py:217-341`).
    ///
    /// When `sample_weight` is `Some`, `X` and `y` are rescaled by
    /// `sqrt(sample_weight)` AFTER centering, exactly as sklearn applies
    /// `_rescale_data` after `_preprocess_data` (`_bayes.py:246-256`); a
    /// weighted least-squares fit is then an ordinary fit on the rescaled data.
    /// Passing `None` is byte-identical to [`Fit::fit`].
    ///
    /// After centering (when `fit_intercept`), the (thin) SVD `X = U S Vᵀ`
    /// gives `eigen_vals_ = S²` (`_bayes.py:287-288`). Each iteration updates
    /// the posterior mean `coef_` (`_bayes.py:294`, `_update_coef_`), then the
    /// effective degrees of freedom and the Gamma-prior precision updates
    /// (`_bayes.py:305-307`):
    ///
    /// ```text
    /// gamma_  = sum((alpha_ * eigen_vals_) / (lambda_ + alpha_ * eigen_vals_))
    /// lambda_ = (gamma_ + 2*lambda_1) / (sum(coef_²) + 2*lambda_2)
    /// alpha_  = (n_samples - gamma_ + 2*alpha_1) / (rmse_ + 2*alpha_2)
    /// ```
    ///
    /// converging when `sum(|coef_old - coef_|) < tol` (`_bayes.py:310`).
    /// `n_iter_` is set to `iter_ + 1` (`_bayes.py:316`); when `compute_score`
    /// is set, the log marginal likelihood is accumulated per iteration plus
    /// once after the loop (`_bayes.py:283/302/330`).
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch (`y` or
    ///   `sample_weight`).
    /// - [`FerroError::InvalidParameter`] — non-positive `alpha_init`/`lambda_init`.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 samples.
    /// - [`FerroError::NumericalInstability`] — SVD or numerical failure.
    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedBayesianRidge<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if let Some(sw) = sample_weight
            && sw.len() != n_samples
        {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![sw.len()],
                context: "sample_weight length must match number of samples in X".into(),
            });
        }

        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "BayesianRidge requires at least 2 samples".into(),
            });
        }

        let zero = <F as num_traits::Zero>::zero();
        let one = <F as num_traits::One>::one();

        if let Some(a0) = self.alpha_init
            && a0 <= zero
        {
            return Err(FerroError::InvalidParameter {
                name: "alpha_init".into(),
                reason: "must be positive".into(),
            });
        }

        if let Some(l0) = self.lambda_init
            && l0 <= zero
        {
            return Err(FerroError::InvalidParameter {
                name: "lambda_init".into(),
                reason: "must be positive".into(),
            });
        }

        let n_f = <F as num_traits::NumCast>::from(n_samples).unwrap_or(one);

        // Center data for intercept (sklearn `_preprocess_data`, `_bayes.py:246`).
        // sklearn's `_preprocess_data` computes the WEIGHTED column/target means
        // when `sample_weight` is given; the rescaling itself (`_rescale_data`,
        // `_bayes.py:254-256`) then multiplies the centered data by sqrt(w).
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let (x_mean, y_mean) = match sample_weight {
                Some(sw) => weighted_means(x, y, sw)?,
                None => {
                    let x_mean =
                        x.mean_axis(Axis(0))
                            .ok_or_else(|| FerroError::NumericalInstability {
                                message: "failed to compute column means".into(),
                            })?;
                    let y_mean = y.mean().ok_or_else(|| FerroError::NumericalInstability {
                        message: "failed to compute target mean".into(),
                    })?;
                    (x_mean, y_mean)
                }
            };
            let x_c = x - &x_mean;
            let y_c = y - y_mean;
            (x_c, y_c, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        // sample_weight: rescale centered (X, y) by sqrt(w) (`_bayes.py:254-256`).
        let (x_work, y_work) = match sample_weight {
            Some(sw) => rescale_data(&x_centered, &y_centered, sw),
            None => (x_centered, y_centered),
        };

        // Initialization (`_bayes.py:262-271`): eps = finfo(dtype).eps;
        // alpha_ = 1/(Var(y)+eps) when alpha_init is None; lambda_ = 1 when
        // lambda_init is None. sklearn computes Var on the (rescaled) y.
        let eps = <F as Float>::epsilon();
        let mut alpha = match self.alpha_init {
            Some(a0) => a0,
            None => {
                let var_y = variance(&y_work);
                one / (var_y + eps)
            }
        };
        let mut lambda = self.lambda_init.unwrap_or(one);

        // SVD (`_bayes.py:287-288`): U, S, Vh = svd(X, full_matrices=False);
        // eigen_vals_ = S².
        let (u, s, vt) = svd_thin(&x_work)?;
        let eigen_vals: Array1<F> = s.mapv(|v| v * v);

        let two = one + one;
        let alpha_1 = self.alpha_1;
        let alpha_2 = self.alpha_2;
        let lambda_1 = self.lambda_1;
        let lambda_2 = self.lambda_2;
        let hyperpriors = (alpha_1, alpha_2, lambda_1, lambda_2);

        // `coef_old_` tracks the previous iterate for the convergence check;
        // sklearn recomputes `coef_` once more after the loop (`_bayes.py:322`),
        // so the in-loop posterior mean is not itself the returned coefficient.
        let mut coef_old: Option<Array1<F>> = None;
        let mut scores: Vec<F> = Vec::new();

        // The LOCAL `coef_` from the last in-loop iteration. sklearn's post-loop
        // `_log_marginal_likelihood` (`_bayes.py:327`) is passed this loop-local
        // `coef_` (the posterior mean from the final iteration, computed with the
        // pre-final alpha_/lambda_) — NOT the recomputed `self.coef_` of
        // `_bayes.py:322` — paired with the freshly recomputed post-loop `rmse_`.
        // We retain it to replicate sklearn's exact `scores_[-1]` (#2162).
        let mut last_in_loop_coef: Option<Array1<F>> = None;

        // `n_iter_` = iter_ + 1 after the loop (`_bayes.py:316`). The loop always
        // runs at least once (max_iter >= 1 in sklearn's constraint), so track
        // the last `iter_`.
        let mut last_iter: usize = 0;

        // Convergence loop (`_bayes.py:291-314`).
        for iter_ in 0..self.max_iter {
            last_iter = iter_;
            let (coef_new, rmse) =
                update_coef(&x_work, &y_work, &u, &vt, &s, &eigen_vals, alpha, lambda);

            // compute_score: log marginal likelihood with the CURRENT
            // alpha_/lambda_ and the just-computed coef_/rmse_ (`_bayes.py:297-302`).
            if self.compute_score {
                scores.push(log_marginal_likelihood(
                    n_samples,
                    n_features,
                    &eigen_vals,
                    alpha,
                    lambda,
                    &coef_new,
                    rmse,
                    hyperpriors,
                ));
            }

            // gamma_ = sum((alpha_ * eigen_vals_) / (lambda_ + alpha_ * eigen_vals_))
            let gamma: F = eigen_vals
                .iter()
                .map(|&ev| (alpha * ev) / (lambda + alpha * ev))
                .fold(zero, |acc, t| acc + t);

            // lambda_ = (gamma_ + 2*lambda_1) / (sum(coef_²) + 2*lambda_2)
            let coef_sq: F = coef_new.iter().map(|&c| c * c).fold(zero, |a, b| a + b);
            lambda = (gamma + two * lambda_1) / (coef_sq + two * lambda_2);

            // alpha_ = (n_samples - gamma_ + 2*alpha_1) / (rmse_ + 2*alpha_2)
            alpha = (n_f - gamma + two * alpha_1) / (rmse + two * alpha_2);

            // Convergence: iter>0 and sum(|coef_old - coef|) < tol.
            if iter_ != 0
                && let Some(ref prev) = coef_old
            {
                let delta: F = prev
                    .iter()
                    .zip(coef_new.iter())
                    .map(|(&o, &c)| (o - c).abs())
                    .fold(zero, |a, b| a + b);
                if delta < self.tol {
                    last_in_loop_coef = Some(coef_new);
                    break;
                }
            }
            last_in_loop_coef = Some(coef_new.clone());
            coef_old = Some(coef_new);
        }

        let n_iter = last_iter + 1;

        // Final coef_ update with the converged alpha_/lambda_ (`_bayes.py:322`).
        let (coef, final_rmse) =
            update_coef(&x_work, &y_work, &u, &vt, &s, &eigen_vals, alpha, lambda);

        // Final score with the converged alpha_/lambda_ (`_bayes.py:325-330`).
        // R-DEV-1: sklearn's line 327 passes the LOOP-LOCAL `coef_` (the last
        // in-loop posterior mean) together with the freshly RECOMPUTED `rmse_`
        // — a mismatched pair, since line 322 only rebinds `self.coef_`, not the
        // local `coef_`. We replicate that exactly: `last_in_loop_coef` (NOT the
        // recomputed `coef`) paired with `final_rmse` (#2162). The fitted
        // `coef`/predict path keeps the recomputed `coef` (line 322's
        // `self.coef_`) and is unaffected.
        if self.compute_score {
            let score_coef = last_in_loop_coef.as_ref().unwrap_or(&coef);
            scores.push(log_marginal_likelihood(
                n_samples,
                n_features,
                &eigen_vals,
                alpha,
                lambda,
                score_coef,
                final_rmse,
                hyperpriors,
            ));
        }

        // Full posterior covariance sigma_ = (1/alpha_) * Vhᵀ ·
        // diag(1/(eigen_vals_ + lambda_/alpha_)) · Vh (`_bayes.py:333-337`).
        let ratio = lambda / alpha;
        let inv_alpha = one / alpha;
        let k = s.len();
        // scaled_rows_i = Vh_i / (eigen_vals_i + lambda_/alpha_); sigma_full =
        // (1/alpha_) * Vhᵀ @ scaled_rows.
        let mut sigma_full = Array2::<F>::zeros((n_features, n_features));
        for a in 0..n_features {
            for b in 0..n_features {
                let mut acc = zero;
                for i in 0..k {
                    acc += (vt[[i, a]] * vt[[i, b]]) / (eigen_vals[i] + ratio);
                }
                sigma_full[[a, b]] = inv_alpha * acc;
            }
        }
        let sigma_diag: Array1<F> = (0..n_features).map(|j| sigma_full[[j, j]]).collect();

        // intercept_ = y_offset - X_offset · coef_ (`_bayes.py:339`,
        // `_set_intercept`).
        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            *ym - xm.dot(&coef)
        } else {
            zero
        };

        Ok(FittedBayesianRidge {
            coefficients: coef,
            intercept,
            alpha,
            lambda,
            sigma: sigma_diag,
            sigma_full,
            n_iter,
            scores,
        })
    }
}

impl<F: LinalgFloat + ScalarOperand + FromPrimitive> Fit<Array2<F>, Array1<F>>
    for BayesianRidge<F>
{
    type Fitted = FittedBayesianRidge<F>;
    type Error = FerroError;

    /// Fit the Bayesian Ridge model by MacKay (1992) evidence maximization,
    /// mirroring `sklearn.linear_model.BayesianRidge.fit`
    /// (`sklearn/linear_model/_bayes.py:217-341`). This delegates to
    /// [`BayesianRidge::fit_with_sample_weight`] with `sample_weight = None`.
    ///
    /// # Errors
    ///
    /// See [`BayesianRidge::fit_with_sample_weight`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedBayesianRidge<F>, FerroError> {
        self.fit_with_sample_weight(x, y, None)
    }
}

/// Weighted column means of `X` and weighted mean of `y` using `sample_weight`,
/// mirroring the weighted averages sklearn's `_preprocess_data` computes when
/// `sample_weight` is supplied (`sklearn/linear_model/_base.py`, used at
/// `_bayes.py:246-252`): `X_offset_ = average(X, axis=0, weights=sw)`,
/// `y_offset_ = average(y, weights=sw)`.
fn weighted_means<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    sample_weight: &Array1<F>,
) -> Result<(Array1<F>, F), FerroError> {
    let n_features = x.ncols();
    let sw_sum = sample_weight.iter().fold(F::zero(), |a, &b| a + b);
    if sw_sum <= F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "sample_weight".into(),
            reason: "sum of sample_weight must be positive".into(),
        });
    }
    let mut x_mean = Array1::<F>::zeros(n_features);
    for (row, &w) in x.outer_iter().zip(sample_weight.iter()) {
        for (j, &v) in row.iter().enumerate() {
            x_mean[j] = x_mean[j] + w * v;
        }
    }
    x_mean.mapv_inplace(|s| s / sw_sum);
    let y_mean = y
        .iter()
        .zip(sample_weight.iter())
        .fold(F::zero(), |acc, (&yi, &w)| acc + w * yi)
        / sw_sum;
    Ok((x_mean, y_mean))
}

/// Population variance `mean((v - mean(v))²)`, matching numpy's `np.var`
/// (the `ddof=0` default sklearn relies on at `_bayes.py:269`).
fn variance<F: Float>(v: &Array1<F>) -> F {
    let n = v.len();
    if n == 0 {
        return F::zero();
    }
    let n_f = F::from(n).unwrap_or_else(F::one);
    let mean = v.iter().fold(F::zero(), |a, &b| a + b) / n_f;
    let ss = v
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .fold(F::zero(), |a, b| a + b);
    ss / n_f
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedBayesianRidge<F>
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

impl<F: Float + ScalarOperand + 'static> FittedBayesianRidge<F> {
    /// Predict the posterior mean AND the predictive standard deviation,
    /// mirroring `sklearn.linear_model.BayesianRidge.predict(X, return_std=True)`
    /// (`sklearn/linear_model/_bayes.py:367-371`):
    ///
    /// ```text
    /// y_mean = X @ coef_ + intercept_
    /// y_std  = sqrt( (X @ sigma_ * X).sum(axis=1) + 1/alpha_ )
    /// ```
    ///
    /// Returns `(y_mean, y_std)`.
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

        // sigmas_squared_data = (X @ sigma_ * X).sum(axis=1) (`_bayes.py:369`).
        let xs = x.dot(&self.sigma_full); // (n_samples, n_features)
        let inv_alpha = F::one() / self.alpha;
        let y_std: Array1<F> = xs
            .outer_iter()
            .zip(x.outer_iter())
            .map(|(xs_row, x_row)| {
                let q = xs_row
                    .iter()
                    .zip(x_row.iter())
                    .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
                (q + inv_alpha).sqrt()
            })
            .collect();

        Ok((y_mean, y_std))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedBayesianRidge<F>
{
    /// Returns the posterior mean coefficient vector.
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    /// Returns the intercept term.
    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for BayesianRidge<F>
where
    F: LinalgFloat + FromPrimitive + ScalarOperand,
{
    /// Fit the model and return it as a boxed pipeline estimator.
    ///
    /// # Errors
    ///
    /// Propagates any [`FerroError`] from `fit`.
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedBayesianRidge<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    /// Generate predictions via the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates any [`FerroError`] from `predict`.
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // ---- Builder ----

    #[test]
    fn test_default_constructor() {
        // Mirrors sklearn BayesianRidge.__init__ defaults (`_bayes.py:187-202`):
        // alpha_init/lambda_init default to None; the four Gamma hyperpriors
        // default to 1e-6.
        let m = BayesianRidge::<f64>::new();
        assert_eq!(m.max_iter, 300);
        assert!(m.fit_intercept);
        assert!(m.alpha_init.is_none());
        assert!(m.lambda_init.is_none());
        assert_relative_eq!(m.alpha_1, 1e-6);
        assert_relative_eq!(m.alpha_2, 1e-6);
        assert_relative_eq!(m.lambda_1, 1e-6);
        assert_relative_eq!(m.lambda_2, 1e-6);
    }

    #[test]
    fn test_builder_setters() {
        let m = BayesianRidge::<f64>::new()
            .with_max_iter(50)
            .with_tol(1e-6)
            .with_alpha_init(2.0)
            .with_lambda_init(0.5)
            .with_alpha_1(1e-4)
            .with_alpha_2(2e-4)
            .with_lambda_1(3e-4)
            .with_lambda_2(4e-4)
            .with_fit_intercept(false);
        assert_eq!(m.max_iter, 50);
        assert!(!m.fit_intercept);
        assert_eq!(m.alpha_init, Some(2.0));
        assert_eq!(m.lambda_init, Some(0.5));
        assert_relative_eq!(m.alpha_1, 1e-4);
        assert_relative_eq!(m.alpha_2, 2e-4);
        assert_relative_eq!(m.lambda_1, 3e-4);
        assert_relative_eq!(m.lambda_2, 4e-4);
    }

    // ---- Validation errors ----

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        let result = BayesianRidge::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let y = array![1.0];
        let result = BayesianRidge::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_positive_alpha_init() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = BayesianRidge::<f64>::new().with_alpha_init(0.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_positive_lambda_init() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = BayesianRidge::<f64>::new()
            .with_lambda_init(-1.0)
            .fit(&x, &y);
        assert!(result.is_err());
    }

    // ---- Correctness ----

    #[test]
    fn test_fits_linear_data() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();

        // Should recover roughly y = 2x + 1.
        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.1);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 0.5);
    }

    #[test]
    fn test_alpha_and_lambda_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();

        assert!(fitted.alpha() > 0.0);
        assert!(fitted.lambda() > 0.0);
    }

    #[test]
    fn test_sigma_diagonal_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();

        for &v in fitted.sigma() {
            assert!(v > 0.0, "sigma diagonal must be positive, got {v}");
        }
    }

    #[test]
    fn test_sigma_length_matches_features() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.5, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.sigma().len(), 2);
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = BayesianRidge::<f64>::new()
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_length() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_coefficients_length() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.5, 2.0, 1.0, 1.0, 3.0, 0.0, 1.5, 4.0, 1.0, 2.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 3);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = BayesianRidge::<f64>::new();
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_multivariate_fit() {
        // y = 1*x1 + 2*x2
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 6.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        // Rough sanity: residuals should be small.
        let residuals: Vec<f64> = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .collect();
        assert!(residuals.iter().all(|&r| r < 1.0));
    }
}
