//! Gaussian Process regression.
//!
//! This module implements [`GaussianProcessRegressor`], a Bayesian nonparametric
//! regression model that provides both predictions and uncertainty estimates.
//!
//! # Algorithm
//!
//! Given training data `(X, y)` and a kernel function `k`:
//!
//! 1. Compute `K = k(X, X) + alpha * I` (kernel matrix + noise regularization).
//! 2. Cholesky decompose: `L = cholesky(K)`.
//! 3. Solve `L L^T alpha_vec = y` for `alpha_vec`.
//!
//! When `normalize_y` is enabled, targets are standardized in `fit` by
//! subtracting the population mean and dividing by the population standard
//! deviation (`np.std`, ddof=0; std=0 is replaced with 1 via
//! `_handle_zeros_in_scale`), mirroring sklearn `_gpr.py:269-273`. Predictions
//! are mapped back to the original target scale by multiplying the
//! normalized-space mean by `y_std` and adding `y_mean` (`_gpr.py:443`), and the
//! predictive variance is rescaled by `y_std²` (`_gpr.py:484`).
//!
//! Prediction at new points `X*`:
//!
//! - Mean: `y* = (K(X*, X) @ alpha_vec) * y_std + y_mean`
//! - Variance: `var = (diag(K(X*, X*)) - sum(v^2)) * y_std^2` where
//!   `v = L^{-1} K(X, X*)`.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.gaussian_process.GaussianProcessRegressor` (`_gpr.py:26`,
//! v1.5.2 commit 156ef14). Design doc: `.design/kernel/gaussian_process.md`
//! (15 REQs). Every REQ is BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a
//! concrete blocker). The fixed-kernel GP math is DETERMINISTIC and oracle-
//! verified element-wise against the live sklearn 1.5.2 (`optimizer=None`); the
//! headline gap is that ferrolearn does NOT tune kernel hyperparameters (sklearn
//! default `optimizer="fmin_l_bfgs_b"`), which depends on the gp_kernels
//! `eval_gradient`/`bounds` blockers #1912/#1913.
//!
//! **7 SHIPPED / 8 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-1 (fixed-kernel posterior mean) | SHIPPED | `fit` (`K+alpha·I`, Cholesky, `cho_solve`) + `predict` (`K_trans·alpha_·y_std + y_mean`) match sklearn (`_gpr.py:342-356`,`:439-443`); oracle `predict=[0.0996, 5.8147]` (RBF(1.0), optimizer=None). Guard `green_predict_mean`. |
//! | REQ-2 (predictive variance / std) | SHIPPED | `predict_with_std` `var = (diag − Σv²)·y_std²`, clip<0, `sqrt` matches `_gpr.py:450-490`; oracle std `[0.1184, 0.0900]`. Guard `green_predict_std`. |
//! | REQ-3 (LML value at fitted theta) | SHIPPED | `log_marginal_likelihood` `−0.5·yᵀα − Σln(diag L) − n/2·ln2π` = sklearn `log_marginal_likelihood_value_` (`_gpr.py:605-609`); oracle `-138.9976`. Guards `green_log_marginal`, `green_normalize_y_log_marginal`. |
//! | REQ-4 (score / R²) | SHIPPED | `score` = R² on `predict` (sklearn `RegressorMixin.score`); oracle `1.0`. Guards `green_score_r2`, `green_normalize_y_score_r2`. |
//! | REQ-5 (alpha default 1e-10) | SHIPPED | `new` sets `alpha = 1e-10`, matching `__init__(alpha=1e-10)` (`_gpr.py:204`). Guard `green_alpha_default`. |
//! | REQ-6 (production consumer) | SHIPPED | `lib.rs` re-exports `GaussianProcessRegressor`/`FittedGaussianProcessRegressor` — the boundary estimator API (no Python GP binding; the Rust estimator IS the public surface, R-DEFER-1/S5). |
//! | REQ-7 (normalize_y std scaling) | SHIPPED | FIXED #1921: `fit` divides `y` by the population std (ddof=0, `std=0→1` guard) and `predict`/`predict_with_std`/`sample_y`/`log_marginal_likelihood` rescale by `y_std`/`y_std²`, matching `_gpr.py:268-273`,`:443`,`:480-485`; oracle-verified at normalize_y=True (mean/std/LML/score + constant-y guard). Pinned by `divergence_normalize_y_std_scaling` + `green_normalize_y_*`. |
//! | REQ-8 (hyperparameter optimization) | NOT-STARTED | `fit` never optimizes (`n_restarts_optimizer` inert); sklearn default `optimizer="fmin_l_bfgs_b"` tunes theta via L-BFGS-B on the neg-LML (`_gpr.py:292-333`). Needs gp_kernels `eval_gradient` #1912 + `bounds` #1913 + an L-BFGS-B. Blocker #1922. |
//! | REQ-9 (LML theta-arg + gradient API) | NOT-STARTED | `log_marginal_likelihood(y)` evaluates only at the fitted theta; sklearn's takes `theta` + returns the gradient (`_gpr.py:533-648`). Needs #1912. Blocker #1923. |
//! | REQ-10 (multi-output y / n_targets) | NOT-STARTED | single-output `Array1<F>`; sklearn supports `(n, n_targets)` (`_gpr.py:251`,`:269-278`). Blocker #1924. |
//! | REQ-11 (GP-prior unfitted predict) | NOT-STARTED | only the Fitted type predicts; sklearn returns the prior when unfitted (`_gpr.py:410-436`). Blocker #1925. |
//! | REQ-12 (sample_y numpy RNG) | NOT-STARTED | `Xoshiro256Plus` vs numpy `multivariate_normal` (default random_state=0); exact sample parity blocked on `ferray::random` (R-SUBSTRATE-5, no value test). The posterior mean/cov it draws from are deterministic (REQ-1/2). Blocker #1926. |
//! | REQ-13 (return_cov / covariance output) | NOT-STARTED | no covariance accessor / `return_cov`+`return_std` mutual-exclusion / negative-variance warning (`_gpr.py:398-401`,`:454-477`). Blocker #1927. |
//! | REQ-14 (constructor surface) | NOT-STARTED | no `kernel=None`→`C(1.0)*RBF(1.0)` default, `optimizer`, `copy_X_train`, `random_state`, `n_targets` (`_gpr.py:200-241`). Blocker #1928. |
//! | REQ-15 (ferray substrate) | NOT-STARTED | `ndarray` + hand-rolled `cholesky`/`cho_solve` vs `ferray-core`/`ferray::linalg`/`ferray::random`. Blocker #1929. |

use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256Plus;

use ferrolearn_core::{FerroError, Fit, Predict};

use crate::gp_kernels::{GPKernel, RBFKernel};

/// Gaussian Process regressor.
///
/// A Bayesian nonparametric model that infers a distribution over functions.
/// Provides both point predictions and uncertainty estimates (predictive variance).
///
/// # Type Parameters
///
/// - `F`: Float type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_kernel::gaussian_process::GaussianProcessRegressor;
/// use ferrolearn_kernel::gp_kernels::RBFKernel;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]);
///
/// let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
/// let fitted = gp.fit(&x, &y).unwrap();
/// let predictions = fitted.predict(&x).unwrap();
/// ```
pub struct GaussianProcessRegressor<F: Float + Send + Sync + 'static> {
    /// Covariance kernel.
    kernel: Box<dyn GPKernel<F>>,
    /// Noise regularization added to the diagonal of the kernel matrix.
    /// Default: `1e-10`.
    alpha: F,
    /// Whether to standardize targets (subtract the population mean and divide
    /// by the population standard deviation) before fitting. Predictions are
    /// mapped back to the original target scale (mean rescaled by `y_std` then
    /// `y_mean` added; variance rescaled by `y_std²`). Mirrors sklearn
    /// `_gpr.py:269-273`/`:443`/`:484`. Default: `false`.
    normalize_y: bool,
    /// Number of random restarts for kernel hyperparameter optimization.
    /// Default: `0` (no optimization, use kernel parameters as-is).
    n_restarts_optimizer: usize,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for GaussianProcessRegressor<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaussianProcessRegressor")
            .field("normalize_y", &self.normalize_y)
            .field("n_restarts_optimizer", &self.n_restarts_optimizer)
            .finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> GaussianProcessRegressor<F> {
    /// Create a new GP regressor with the given kernel and default settings.
    ///
    /// Defaults: `alpha = 1e-10`, `normalize_y = false`, `n_restarts_optimizer = 0`.
    pub fn new(kernel: Box<dyn GPKernel<F>>) -> Self {
        Self {
            kernel,
            alpha: F::from(1e-10).unwrap(),
            normalize_y: false,
            n_restarts_optimizer: 0,
        }
    }

    /// Create a GP regressor with an RBF kernel and default length scale.
    pub fn default_rbf() -> Self {
        Self::new(Box::new(RBFKernel::new(F::one())))
    }

    /// Set the noise regularization parameter.
    #[must_use]
    pub fn alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Enable or disable target standardization (subtract mean, divide by
    /// population standard deviation; rescaled back at prediction time).
    #[must_use]
    pub fn normalize_y(mut self, normalize: bool) -> Self {
        self.normalize_y = normalize;
        self
    }

    /// Set the number of random restarts for optimizer.
    #[must_use]
    pub fn n_restarts_optimizer(mut self, n: usize) -> Self {
        self.n_restarts_optimizer = n;
        self
    }
}

/// Fitted Gaussian Process regressor.
///
/// Holds the Cholesky factor and weight vector needed for prediction.
/// Use [`predict`](Predict::predict) for point predictions, or
/// [`predict_with_std`](FittedGaussianProcessRegressor::predict_with_std)
/// for predictions with uncertainty.
pub struct FittedGaussianProcessRegressor<F: Float + Send + Sync + 'static> {
    /// Training features.
    x_train: Array2<F>,
    /// Cholesky factor L of the kernel matrix (lower triangular).
    l_factor: Array2<F>,
    /// Weight vector: alpha_vec = K^{-1} y (via Cholesky solve).
    alpha_vec: Array1<F>,
    /// Mean of y (subtracted during training if normalize_y is true;
    /// otherwise 0). Added back to the predictive mean (sklearn `_y_train_mean`,
    /// `_gpr.py:269`/`:443`).
    y_mean: F,
    /// Population standard deviation of y used to standardize targets when
    /// normalize_y is true (otherwise 1). The predictive mean is multiplied by
    /// `y_std` and the variance by `y_std²` to map back to the original scale
    /// (sklearn `_y_train_std`, `_gpr.py:270-273`/`:443`/`:484`).
    y_std: F,
    /// Kernel used during fitting (stored for prediction).
    kernel: Box<dyn GPKernel<F>>,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for FittedGaussianProcessRegressor<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FittedGaussianProcessRegressor")
            .field("n_train", &self.x_train.nrows())
            .field("n_features", &self.x_train.ncols())
            .finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> FittedGaussianProcessRegressor<F> {
    /// R² coefficient of determination on the given test data.
    /// Equivalent to sklearn's `RegressorMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(crate::r2_score(&preds, y))
    }

    /// Predict mean and standard deviation at new points.
    ///
    /// Returns `(y_mean, y_std)` where `y_std` is the square root of the
    /// posterior predictive variance.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature dimension does
    /// not match the training data.
    pub fn predict_with_std(&self, x: &Array2<F>) -> Result<(Array1<F>, Array1<F>), FerroError> {
        if x.ncols() != self.x_train.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.x_train.ncols()],
                actual: vec![x.nrows(), x.ncols()],
                context: "predict feature count must match training data".into(),
            });
        }

        let k_star = self.kernel.compute(x, &self.x_train);

        // Mean: y_std * (K* @ alpha_vec) + y_mean (sklearn `_gpr.py:443`).
        let y_pred = k_star
            .dot(&self.alpha_vec)
            .mapv(|v| v * self.y_std + self.y_mean);

        // Variance: diag(K**) - sum(v^2, axis=0)
        // where v = L^{-1} K*^T
        let k_star_diag = self.kernel.diagonal(x);
        let n_train = self.x_train.nrows();
        let n_pred = x.nrows();

        // Solve L v = K*^T for v via forward substitution
        // K*^T has shape (n_train, n_pred)
        let k_star_t = k_star.t().to_owned();
        let mut v = Array2::<F>::zeros((n_train, n_pred));
        for col in 0..n_pred {
            // Forward substitution for this column
            for i in 0..n_train {
                let mut sum = k_star_t[[i, col]];
                for j in 0..i {
                    sum = sum - self.l_factor[[i, j]] * v[[j, col]];
                }
                v[[i, col]] = sum / self.l_factor[[i, i]];
            }
        }

        // var = k_star_diag - sum(v^2, axis=0)
        let mut var = k_star_diag;
        for col in 0..n_pred {
            let mut sum_sq = F::zero();
            for row in 0..n_train {
                sum_sq = sum_sq + v[[row, col]] * v[[row, col]];
            }
            var[col] = var[col] - sum_sq;
            // Clamp to avoid negative variance from numerical errors
            // (sklearn `_gpr.py:475-481`).
            if var[col] < F::zero() {
                var[col] = F::zero();
            }
            // Undo target standardization: variance scales by y_std²
            // (sklearn `_gpr.py:484`). y_std=1 when normalize_y is false.
            var[col] = var[col] * self.y_std * self.y_std;
        }

        let std = var.mapv(num_traits::Float::sqrt);
        Ok((y_pred, std))
    }

    /// Draw `n_samples` realizations from the GP posterior at the query
    /// points `x`. Mirrors sklearn `GaussianProcessRegressor.sample_y`.
    ///
    /// Returns shape `(n_query, n_samples)`. Each column is one posterior
    /// draw `mean + L_post @ z` where `L_post` is the Cholesky factor of
    /// the posterior covariance `K** - K*ᵀ K⁻¹ K*` and `z ~ N(0, I)`.
    ///
    /// `random_state = Some(seed)` makes draws reproducible (uses
    /// `Xoshiro256Plus`); `None` reseeds from the OS RNG.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if the feature dimension does not
    ///   match the training data.
    /// - [`FerroError::NumericalInstability`] if the posterior covariance
    ///   fails Cholesky (very rare; a small jitter `1e-10` is added to
    ///   the diagonal first).
    pub fn sample_y(
        &self,
        x: &Array2<F>,
        n_samples: usize,
        random_state: Option<u64>,
    ) -> Result<Array2<F>, FerroError>
    where
        StandardNormal: Distribution<F>,
    {
        if x.ncols() != self.x_train.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.x_train.ncols()],
                actual: vec![x.nrows(), x.ncols()],
                context: "sample_y feature count must match training data".into(),
            });
        }

        let n_query = x.nrows();
        let n_train = self.x_train.nrows();

        let k_star = self.kernel.compute(x, &self.x_train);
        let k_star_star = self.kernel.compute(x, x);

        // Posterior mean: y_std * (K* @ alpha_vec) + y_mean (sklearn
        // `_gpr.py:443`). sklearn `sample_y` draws from `predict(return_cov=True)`,
        // which is on the original (un-standardized) target scale.
        let mean = k_star
            .dot(&self.alpha_vec)
            .mapv(|v| v * self.y_std + self.y_mean);

        // Solve L V = K*^T column-by-column for V (shape (n_train, n_query)).
        let k_star_t = k_star.t().to_owned();
        let mut v = Array2::<F>::zeros((n_train, n_query));
        for col in 0..n_query {
            for i in 0..n_train {
                let mut sum = k_star_t[[i, col]];
                for j in 0..i {
                    sum = sum - self.l_factor[[i, j]] * v[[j, col]];
                }
                v[[i, col]] = sum / self.l_factor[[i, i]];
            }
        }

        // Posterior covariance: (K** - V^T V) * y_std² (undo target
        // standardization, sklearn `_gpr.py:459`), with a small jitter on the
        // diagonal for Cholesky stability. y_std=1 when normalize_y is false.
        let mut k_post = Array2::<F>::zeros((n_query, n_query));
        let jitter = F::from(1e-10).unwrap_or_else(F::epsilon);
        let y_std_sq = self.y_std * self.y_std;
        for i in 0..n_query {
            for j in 0..n_query {
                let mut s = k_star_star[[i, j]];
                for k in 0..n_train {
                    s = s - v[[k, i]] * v[[k, j]];
                }
                k_post[[i, j]] = s * y_std_sq;
                if i == j {
                    k_post[[i, j]] = k_post[[i, j]] + jitter;
                    if k_post[[i, j]] < jitter {
                        k_post[[i, j]] = jitter;
                    }
                }
            }
        }

        let l_post = cholesky(&k_post)?;

        let mut rng = match random_state {
            Some(seed) => Xoshiro256Plus::seed_from_u64(seed),
            None => Xoshiro256Plus::from_seed(rand::random()),
        };

        let mut out = Array2::<F>::zeros((n_query, n_samples));
        for s in 0..n_samples {
            // Draw z ~ N(0, I).
            let mut z = Array1::<F>::zeros(n_query);
            for i in 0..n_query {
                z[i] = StandardNormal.sample(&mut rng);
            }
            // Sample = mean + L_post @ z.
            for i in 0..n_query {
                let mut sum = F::zero();
                for j in 0..=i {
                    sum = sum + l_post[[i, j]] * z[j];
                }
                out[[i, s]] = mean[i] + sum;
            }
        }
        Ok(out)
    }

    /// Get the log marginal likelihood of the fitted model.
    ///
    /// This is useful for model selection and hyperparameter optimization.
    ///
    /// `log p(y|X) = -0.5 * y^T K^{-1} y - sum(log(diag(L))) - n/2 * log(2*pi)`
    #[must_use]
    pub fn log_marginal_likelihood(&self, y: &Array1<F>) -> F {
        let n = F::from(self.x_train.nrows()).unwrap_or_else(F::one);
        // The LML is computed on the standardized targets, consistent with the
        // stored `alpha_vec` = K⁻¹·(y - y_mean)/y_std (sklearn `_gpr.py:273` +
        // `:605-609`). y_std=1, y_mean=0 when normalize_y is false (unchanged).
        let y_centered = if self.y_mean == F::zero() && self.y_std == F::one() {
            y.clone()
        } else {
            y.mapv(|v| (v - self.y_mean) / self.y_std)
        };

        // -0.5 * y^T alpha
        let data_fit = F::from(-0.5).unwrap() * y_centered.dot(&self.alpha_vec);

        // -sum(log(diag(L)))
        let mut log_det = F::zero();
        for i in 0..self.l_factor.nrows() {
            log_det = log_det + self.l_factor[[i, i]].ln();
        }
        let complexity = -log_det;

        // -n/2 * log(2*pi)
        let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap();
        let norm_const = F::from(-0.5).unwrap() * n * two_pi.ln();

        data_fit + complexity + norm_const
    }
}

// ---------------------------------------------------------------------------
// Cholesky decomposition (pure Rust, generic over F)
// ---------------------------------------------------------------------------

/// Compute the lower Cholesky factor L such that A = L L^T.
///
/// Returns `Err` if A is not positive definite.
fn cholesky<F: Float>(a: &Array2<F>) -> Result<Array2<F>, FerroError> {
    let n = a.nrows();
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum = sum - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: format!("kernel matrix is not positive definite at pivot {i}"),
                    });
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Solve L x = b via forward substitution, where L is lower triangular.
fn forward_solve<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let n = l.nrows();
    let mut x = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum = sum - l[[i, j]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// Solve L^T x = b via backward substitution, where L is lower triangular.
fn backward_solve<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let n = l.nrows();
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum = sum - l[[j, i]] * x[j]; // L^T[i,j] = L[j,i]
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// Solve (L L^T) x = b via Cholesky factorization.
fn cholesky_solve<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let z = forward_solve(l, b);
    backward_solve(l, &z)
}

// ---------------------------------------------------------------------------
// Fit / Predict implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for GaussianProcessRegressor<F> {
    type Fitted = FittedGaussianProcessRegressor<F>;
    type Error = FerroError;

    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedGaussianProcessRegressor<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples < 1 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: n_samples,
                context: "GaussianProcessRegressor::fit".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match X rows".into(),
            });
        }

        // Optionally standardize y: subtract the population mean and divide by
        // the population standard deviation (np.std, ddof=0), mirroring sklearn
        // `_gpr.py:269-273`. `_handle_zeros_in_scale` (`_gpr.py:270`) replaces a
        // zero std (constant target) with 1. When normalize_y is false sklearn
        // sets mean=0, std=1 (`_gpr.py:277-278`) so the centering and the
        // rescales below are no-ops (byte-identical to the un-normalized path).
        let (y_mean, y_std) = if self.normalize_y {
            let n = F::from(n_samples).ok_or_else(|| FerroError::NumericalInstability {
                message: "sample count not representable in float type".into(),
            })?;
            let mean = y.sum() / n;
            let var = y.iter().fold(F::zero(), |acc, &v| {
                let d = v - mean;
                acc + d * d
            }) / n;
            let mut std = var.sqrt();
            // _handle_zeros_in_scale: constant-target guard (std == 0 -> 1).
            if std <= F::zero() {
                std = F::one();
            }
            (mean, std)
        } else {
            (F::zero(), F::one())
        };
        let y_centered = if self.normalize_y {
            y.mapv(|v| (v - y_mean) / y_std)
        } else {
            y.clone()
        };

        // Compute kernel matrix K + alpha * I
        let mut k_mat = self.kernel.compute(x, x);
        for i in 0..n_samples {
            k_mat[[i, i]] = k_mat[[i, i]] + self.alpha;
        }

        // Cholesky decomposition
        let l = cholesky(&k_mat)?;

        // Solve K alpha_vec = y_centered
        let alpha_vec = cholesky_solve(&l, &y_centered);

        Ok(FittedGaussianProcessRegressor {
            x_train: x.clone(),
            l_factor: l,
            alpha_vec,
            y_mean,
            y_std,
            kernel: self.kernel.clone_box(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedGaussianProcessRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.x_train.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.x_train.ncols()],
                actual: vec![x.nrows(), x.ncols()],
                context: "predict feature count must match training data".into(),
            });
        }

        let k_star = self.kernel.compute(x, &self.x_train);
        // Undo target standardization: y_mean = y_std * (K* @ alpha) + y_mean
        // (sklearn `_gpr.py:443`). y_std=1, y_mean=0 when normalize_y is false.
        let y_pred = k_star
            .dot(&self.alpha_vec)
            .mapv(|v| v * self.y_std + self.y_mean);
        Ok(y_pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp_kernels::{
        ConstantKernel, DotProductKernel, MaternKernel, ProductKernel, SumKernel, WhiteKernel,
    };
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    // Helper to create simple training data
    fn make_linear_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0];
        (x, y)
    }

    fn make_quadratic_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];
        (x, y)
    }

    // --- Basic fit/predict ---

    #[test]
    fn fit_predict_basic() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 5);
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn fit_predict_interpolation() {
        // GP should near-interpolate training data (with small alpha)
        let (x, y) = make_quadratic_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(2.0))).alpha(1e-10);
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(pred[i], y[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn fit_predict_normalize_y() {
        let (x, y) = make_quadratic_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(2.0)))
            .normalize_y(true)
            .alpha(1e-10);
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(pred[i], y[i], epsilon = 1e-3);
        }
    }

    // --- Predict with std ---

    #[test]
    fn predict_with_std_basic() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();
        let (mean, std) = fitted.predict_with_std(&x).unwrap();
        assert_eq!(mean.len(), 5);
        assert_eq!(std.len(), 5);
        // Std at training points should be near zero
        for &s in &std {
            assert!(s < 1.0, "Training point std should be small, got {s}");
        }
    }

    #[test]
    fn predict_with_std_far_away() {
        // Points far from training data should have high uncertainty
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();

        let x_far = Array2::from_shape_vec((1, 1), vec![100.0]).unwrap();
        let (_, std_far) = fitted.predict_with_std(&x_far).unwrap();
        let (_, std_near) = fitted.predict_with_std(&x).unwrap();

        let max_near_std = std_near.iter().copied().fold(0.0f64, f64::max);
        assert!(
            std_far[0] > max_near_std,
            "Far point std ({}) should exceed near std ({})",
            std_far[0],
            max_near_std
        );
    }

    #[test]
    fn predict_with_std_variance_nonnegative() {
        let (x, y) = make_quadratic_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();

        let x_test = Array2::from_shape_vec((10, 1), (-5..5).map(f64::from).collect()).unwrap();
        let (_, std) = fitted.predict_with_std(&x_test).unwrap();
        for &s in &std {
            assert!(s >= 0.0, "std should be non-negative, got {s}");
        }
    }

    // --- Error handling ---

    #[test]
    fn fit_rejects_mismatched_y() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0]; // Wrong length
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        assert!(gp.fit(&x, &y).is_err());
    }

    #[test]
    fn predict_rejects_wrong_features() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    #[test]
    fn predict_with_std_rejects_wrong_features() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 3), vec![0.0; 6]).unwrap();
        assert!(fitted.predict_with_std(&x_wrong).is_err());
    }

    // --- Different kernels ---

    #[test]
    fn fit_with_matern_15() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(MaternKernel::new(1.0, 1.5)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(pred[i], y[i], epsilon = 0.5);
        }
    }

    #[test]
    fn fit_with_matern_25() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(MaternKernel::new(1.0, 2.5)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn fit_with_dot_product() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(DotProductKernel::new(1.0))).alpha(1e-6);
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn fit_with_sum_kernel() {
        let (x, y) = make_linear_data();
        let kernel = SumKernel::new(
            Box::new(RBFKernel::new(1.0)),
            Box::new(WhiteKernel::new(0.01)),
        );
        let gp = GaussianProcessRegressor::new(Box::new(kernel));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn fit_with_product_kernel() {
        let (x, y) = make_linear_data();
        let kernel = ProductKernel::new(
            Box::new(ConstantKernel::new(2.0)),
            Box::new(RBFKernel::new(1.0)),
        );
        let gp = GaussianProcessRegressor::new(Box::new(kernel));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    // --- Log marginal likelihood ---

    #[test]
    fn log_marginal_likelihood_is_finite() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();
        let lml = fitted.log_marginal_likelihood(&y);
        assert!(lml.is_finite(), "LML should be finite, got {lml}");
    }

    #[test]
    fn log_marginal_likelihood_prefers_right_scale() {
        // LML should be higher when the kernel length scale matches the data
        let x =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| f64::from(i) * 0.5).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);

        let gp_good = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0))).alpha(1e-6);
        let gp_bad = GaussianProcessRegressor::new(Box::new(RBFKernel::new(0.01))).alpha(1e-6);

        let fitted_good = gp_good.fit(&x, &y).unwrap();
        let fitted_bad = gp_bad.fit(&x, &y).unwrap();

        let lml_good = fitted_good.log_marginal_likelihood(&y);
        let lml_bad = fitted_bad.log_marginal_likelihood(&y);

        assert!(
            lml_good > lml_bad,
            "Good length scale LML ({lml_good}) should exceed bad ({lml_bad})"
        );
    }

    // --- Multivariate ---

    #[test]
    fn multivariate_2d() {
        let n = 20;
        let x_data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let t = i as f64 / n as f64;
                vec![t, t * t]
            })
            .collect();
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(0.5)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), n);
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    // --- f32 support ---

    #[test]
    fn f32_fit_predict() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0f32, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![0.0f32, 1.0, 4.0, 9.0, 16.0]);
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(2.0f32))).alpha(1e-6f32);
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert!(
                (pred[i] - y[i]).abs() < 1.0f32,
                "f32 pred {i} too far: {}",
                pred[i]
            );
        }
    }

    // --- Builder pattern ---

    #[test]
    fn builder_pattern() {
        let gp = GaussianProcessRegressor::default_rbf()
            .alpha(1e-6)
            .normalize_y(true)
            .n_restarts_optimizer(5);

        let (x, y) = make_linear_data();
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 5);
    }

    // --- Single sample ---

    #[test]
    fn single_sample() {
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let y = array![5.0f64];
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_abs_diff_eq!(pred[0], 5.0, epsilon = 1e-6);
    }

    // --- Constant target ---

    #[test]
    fn constant_target() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_elem(5, 7.0);
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert_abs_diff_eq!(p, 7.0, epsilon = 1e-4);
        }
    }
}
