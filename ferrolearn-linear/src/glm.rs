//! Generalized Linear Models (GLM).
//!
//! This module provides IRLS-based GLM regressors for count and positive
//! continuous data:
//!
//! - **[`GLMRegressor`]** — Generic GLM with selectable [`GLMFamily`]
//! - **[`PoissonRegressor`]** — Convenience wrapper with Poisson family
//! - **[`GammaRegressor`]** — Convenience wrapper with Gamma family
//! - **[`TweedieRegressor`]** — Convenience wrapper with Tweedie family
//!
//! All models use Iteratively Reweighted Least Squares (IRLS) and L2
//! regularization. The link function is fixed to **log** for Poisson and Gamma
//! (their sklearn losses are log-link only); [`TweedieRegressor`] selects its
//! [`Link`] via a `link` configuration (`auto`/`identity`/`log`), matching
//! `sklearn/linear_model/_glm/glm.py:889-903`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::PoissonRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 5.0, 10.0, 20.0];
//!
//! let model = PoissonRegressor::<f64>::new().with_alpha(0.0);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 4);
//! ```
//!
//! ## REQ status (per `.design/linear/glm.md`, mirrors `sklearn/linear_model/_glm/glm.py` @ 1.5.2, commit 156ef14)
//!
//! Binary classification (R-DEFER-2): SHIPPED = impl + tests + green oracle
//! verification; NOT-STARTED = concrete open blocker referenced by `#`-number.
//! The public estimator types re-exported at the crate root are the consumer
//! surface (R-DEFER-1; no `ferrolearn-python` GLM binding yet).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-4 (penalized objective: mean half-deviance + ½·alpha, intercept-free) | SHIPPED | `fn weighted_ridge_solve` adds the L2 penalty `weight_sum * alpha` to feature columns only, skipping the intercept column (`intercept_col`), matching sklearn's mean-deviance objective + unpenalized intercept (`glm.py:229-258`: `obj = average(½·deviance) + ½·alpha·‖coef‖²`, `l2_reg_strength = self.alpha`). Oracle parity tests `glm_poisson_intercept_unpenalized` (alpha=1e6 → `intercept_ = log(mean y)`, coef → 0) and `glm_poisson_penalty_scaling` (alpha=1.0 → `coef_=[0.34151720,0.18859745]`, `intercept_=-0.37680132`) green in `tests/divergence_glm_fit.rs`. |
//! | REQ-1/REQ-2/REQ-3 (Poisson/Gamma/Tweedie families) | NOT-STARTED | #548/#549/#550 — alpha=0 paths match the oracle; per-family alpha>0 parity now uses the correct objective but the director reconciles the SHIPPED claim. |
//! | REQ-5 (intercept init = link(mean y)) | NOT-STARTED | #552 — `coef` still cold-starts at zero; the alpha=1e6 test reaches `log(mean y)` via convergence of the unpenalized intercept, not via init. |
//! | REQ-7 (predict = link.inverse) | SHIPPED | `fn predict` applies `self.link.inverse(eta)` (`Link::Log => exp`, `Link::Identity => eta`), mirroring `glm.py:362` (`y_pred = link.inverse(raw_prediction)`). Consumer: the crate-root-exported `FittedGLMRegressor::predict` used by every wrapper; oracle test `glm_tweedie_power0_predict_identity_inverse` (identity link → raw linear predictor `[0.4,6.3,12.2,18.1]`) green in `tests/divergence_glm_fit.rs`. |
//! | REQ-8 (Tweedie link='auto'/identity/log) | SHIPPED | `pub enum Link { Log, Identity }` + `pub enum LinkConfig { Auto, Log, Identity }` with `LinkConfig::resolve(power)`: Auto → identity for `power <= 0`, log otherwise (`glm.py:889-893`). `TweedieRegressor.link: LinkConfig` (default `Auto`) is resolved at fit time and threaded into `fit_glm_irls`'s link-parameterized IRLS (`w = dmu_deta^2/V(mu)`, `z = eta + (y-mu)/dmu_deta`) and the fitted struct. Consumer: `TweedieRegressor::fit` (crate-root export); oracle test `glm_tweedie_power0_identity_link` (`coef_=[5.9]`, `intercept_=-5.5`, OLS) green. Poisson/Gamma wire `Link::Log` explicitly. |
//! | REQ-9 (Tweedie default power=0.0) | SHIPPED | `TweedieRegressor::new` sets `power: 0.0` (sklearn default, `glm.py:867`). Consumer: `TweedieRegressor::default`/`new` (crate-root export); oracle test `glm_tweedie_default_power` (`new().power == 0.0`) green. |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};

// ---------------------------------------------------------------------------
// Link
// ---------------------------------------------------------------------------

/// The link function `g` of a Generalized Linear Model, mapping the mean `mu`
/// to the linear predictor `eta = g(mu)` (and back via the inverse link `h`,
/// `mu = h(eta)`).
///
/// Mirrors the link carried by sklearn's loss classes
/// (`sklearn/linear_model/_glm/glm.py:119-131`): `HalfPoissonLoss`,
/// `HalfGammaLoss` and `HalfTweedieLoss` use the **log** link
/// (`y_pred = exp(X @ coef + intercept)`); `HalfSquaredError` and
/// `HalfTweedieLossIdentity` use the **identity** link
/// (`y_pred = X @ coef + intercept`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    /// Log link: `g(mu) = ln(mu)`, inverse `h(eta) = exp(eta)`.
    ///
    /// Used by Poisson, Gamma and Tweedie-with-`power > 0` losses.
    Log,
    /// Identity link: `g(mu) = mu`, inverse `h(eta) = eta`.
    ///
    /// Used by the Normal/least-squares loss and Tweedie-with-`power <= 0`.
    Identity,
}

impl Link {
    /// Inverse link `h(eta) = mu`: maps the linear predictor to the mean.
    ///
    /// - [`Link::Log`] → `exp(eta)`
    /// - [`Link::Identity`] → `eta`
    ///
    /// Mirrors `link.inverse(raw_prediction)` in `glm.py:362`.
    #[must_use]
    fn inverse<F: Float>(self, eta: F) -> F {
        match self {
            Link::Log => eta.exp(),
            Link::Identity => eta,
        }
    }

    /// Link derivative of the mean w.r.t. the linear predictor, `dmu/deta`,
    /// used to form the IRLS working weight and response.
    ///
    /// - [`Link::Log`] (`mu = exp(eta)`) → `dmu/deta = mu`
    /// - [`Link::Identity`] (`mu = eta`) → `dmu/deta = 1`
    #[must_use]
    fn dmu_deta<F: Float>(self, mu: F) -> F {
        match self {
            Link::Log => mu,
            Link::Identity => F::one(),
        }
    }
}

/// Configuration of the GLM link function, resolved to a concrete [`Link`] at
/// fit time.
///
/// Mirrors sklearn's `TweedieRegressor(link={'auto','identity','log'})`
/// (`glm.py:861, :889-903`). `Auto` selects the link from the Tweedie `power`:
/// identity for `power <= 0` (Normal), log otherwise (Poisson/Gamma/etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkConfig {
    /// Resolve the link from the Tweedie `power` at fit time
    /// (`power <= 0` → identity, `power > 0` → log). The default.
    Auto,
    /// Force the log link regardless of `power`.
    Log,
    /// Force the identity link regardless of `power`.
    Identity,
}

impl LinkConfig {
    /// Resolve to a concrete [`Link`] given the Tweedie `power`.
    ///
    /// Mirrors `TweedieRegressor._get_loss` (`glm.py:889-903`):
    /// - `Auto` → identity for `power <= 0`, log for `power > 0`
    /// - `Log` → log; `Identity` → identity.
    #[must_use]
    fn resolve(self, power: f64) -> Link {
        match self {
            LinkConfig::Auto => {
                if power <= 0.0 {
                    Link::Identity
                } else {
                    Link::Log
                }
            }
            LinkConfig::Log => Link::Log,
            LinkConfig::Identity => Link::Identity,
        }
    }
}

// ---------------------------------------------------------------------------
// GLMFamily
// ---------------------------------------------------------------------------

/// The distributional family for a Generalized Linear Model.
///
/// Determines the variance function V(mu):
/// - **Poisson**: V(mu) = mu
/// - **Gamma**: V(mu) = mu^2
/// - **Tweedie(p)**: V(mu) = mu^p
#[derive(Debug, Clone, Copy)]
pub enum GLMFamily {
    /// Poisson family — variance proportional to the mean.
    Poisson,
    /// Gamma family — variance proportional to the squared mean.
    Gamma,
    /// Tweedie family with power parameter `p`.
    ///
    /// - `p = 0` gives Normal (constant variance)
    /// - `p = 1` gives Poisson
    /// - `p = 2` gives Gamma
    /// - `1 < p < 2` gives compound Poisson-Gamma
    Tweedie(f64),
}

impl GLMFamily {
    /// Compute the variance function V(mu) for a given mean `mu`.
    fn variance<F: Float + FromPrimitive>(&self, mu: F) -> F {
        match self {
            GLMFamily::Poisson => mu,
            GLMFamily::Gamma => mu * mu,
            GLMFamily::Tweedie(p) => {
                let power = F::from(*p).unwrap();
                mu.powf(power)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GLMRegressor
// ---------------------------------------------------------------------------

/// Generalized Linear Model regressor.
///
/// Fitted via IRLS with a log link function. The [`GLMFamily`] controls
/// the assumed variance-mean relationship.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GLMRegressor<F> {
    /// Distributional family (Poisson, Gamma, or Tweedie).
    pub family: GLMFamily,
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum coefficient change.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> GLMRegressor<F> {
    /// Create a new `GLMRegressor` with the given family.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 100`, `tol = 1e-4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new(family: GLMFamily) -> Self {
        Self {
            family,
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
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

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

/// Fitted GLM regressor.
///
/// Stores the learned coefficients and intercept on the link scale, together
/// with the [`Link`] used at fit time. Predictions are
/// `link.inverse(X @ coef + intercept)` — `exp(...)` for [`Link::Log`], the raw
/// linear predictor for [`Link::Identity`] (`glm.py:362`).
#[derive(Debug, Clone)]
pub struct FittedGLMRegressor<F> {
    /// Learned coefficient vector on the link scale.
    coefficients: Array1<F>,
    /// Learned intercept on the link scale.
    intercept: F,
    /// Link function applied by `predict` (inverse link maps `eta` to `mu`).
    link: Link,
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Poisson regressor — GLM with Poisson family and log link.
///
/// Suitable for modelling count data (y >= 0, integer-valued).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct PoissonRegressor<F> {
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> PoissonRegressor<F> {
    /// Create a new `PoissonRegressor` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 100`, `tol = 1e-4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
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

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for PoissonRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Gamma regressor — GLM with Gamma family and log link.
///
/// Suitable for modelling positive continuous data (y > 0).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GammaRegressor<F> {
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> GammaRegressor<F> {
    /// Create a new `GammaRegressor` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 100`, `tol = 1e-4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
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

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for GammaRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Tweedie regressor — GLM with Tweedie family and log link.
///
/// The `power` parameter controls the variance-mean relationship:
/// V(mu) = mu^power.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct TweedieRegressor<F> {
    /// Tweedie power parameter.
    pub power: f64,
    /// Link-function configuration (`Auto`/`Log`/`Identity`).
    ///
    /// `Auto` (the default) resolves to the identity link for `power <= 0`
    /// (Normal) and the log link for `power > 0`, matching sklearn's
    /// `link='auto'` (`glm.py:889-893`).
    pub link: LinkConfig,
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> TweedieRegressor<F> {
    /// Create a new `TweedieRegressor` with default settings.
    ///
    /// Defaults match sklearn's `TweedieRegressor.__init__` (`glm.py:864-887`):
    /// `power = 0.0` (Normal), `link = LinkConfig::Auto`, `alpha = 1.0`,
    /// `max_iter = 100`, `tol = 1e-4`, `fit_intercept = true`. With the default
    /// `power = 0.0` and `Auto` link, the model is Normal/identity-link (OLS).
    #[must_use]
    pub fn new() -> Self {
        Self {
            power: 0.0,
            link: LinkConfig::Auto,
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            fit_intercept: true,
        }
    }

    /// Set the Tweedie power parameter.
    #[must_use]
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Set the link-function configuration (`Auto`/`Log`/`Identity`).
    ///
    /// Mirrors sklearn's `link={'auto','identity','log'}` (`glm.py:861`).
    #[must_use]
    pub fn with_link(mut self, link: LinkConfig) -> Self {
        self.link = link;
        self
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
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

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for TweedieRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Cholesky solve for `A x = b`.
fn cholesky_solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s = s - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: "Cholesky: matrix not positive definite".into(),
                    });
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }

    let mut z = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s = s - l[[i, k]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }

    let mut x_sol = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s = s - l[[k, i]] * x_sol[k];
        }
        x_sol[i] = s / l[[i, i]];
    }

    Ok(x_sol)
}

/// Gaussian elimination with partial pivoting.
fn gaussian_solve<F: Float>(
    n: usize,
    a: &Array2<F>,
    b: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "singular matrix in Gaussian elimination".into(),
            });
        }

        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }

    let mut x_sol = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = aug[[i, n]];
        for j in (i + 1)..n {
            s = s - aug[[i, j]] * x_sol[j];
        }
        if aug[[i, i]].abs() < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "near-zero pivot in back substitution".into(),
            });
        }
        x_sol[i] = s / aug[[i, i]];
    }

    Ok(x_sol)
}

/// Solve the weighted ridge system `(X^T W X + P) w = X^T W z`, where the
/// penalty matrix `P` adds the L2 regularization to the diagonal of the
/// feature columns only.
///
/// # Penalty scaling and the intercept (sklearn parity, `glm.py:229-258`)
///
/// scikit-learn minimizes the per-sample-MEAN half-deviance plus an L2 prior on
/// the feature coefficients (NOT the intercept):
///
/// ```text
/// J(w) = 1/(2*S) * sum_i s_i * deviance_i + 1/2 * alpha * ||coef||^2,
/// ```
///
/// with `S = sum_i s_i` (= `n_samples` for unweighted fits). Its stationarity
/// condition is `(1/S) * grad[sum 1/2 dev] + alpha * w_features = 0`.
///
/// The IRLS normal equations `X^T W X w = X^T W z` are the linearization of the
/// SUMMED half-deviance `sum_i 1/2 dev_i` (no `1/S` factor): `X^T W X` is the
/// summed-scale Hessian. To make those summed equations correspond to sklearn's
/// mean-scale objective we multiply the penalty by `S` (the sum of weights, =
/// `n_samples` unweighted) before adding it to the diagonal: the added penalty
/// is `weight_sum * alpha`, applied to feature columns only, leaving the
/// intercept (column 0 of the augmented design, when present) unpenalized.
///
/// `weight_sum` is the sum of the GLM `sample_weight` (= `n_samples` for the
/// unweighted case implemented here); `intercept_col` is `Some(0)` when an
/// intercept column was prepended to `x`, `None` otherwise.
fn weighted_ridge_solve<F: Float + FromPrimitive>(
    x: &Array2<F>,
    z: &Array1<F>,
    weights: &Array1<F>,
    alpha: F,
    weight_sum: F,
    intercept_col: Option<usize>,
) -> Result<Array1<F>, FerroError> {
    let (n_samples, n_features) = x.dim();

    let mut xtwx = Array2::<F>::zeros((n_features, n_features));
    let mut xtwz = Array1::<F>::zeros(n_features);

    for i in 0..n_samples {
        let wi = weights[i];
        let xi = x.row(i);
        for r in 0..n_features {
            xtwz[r] = xtwz[r] + wi * xi[r] * z[i];
            for c in 0..n_features {
                xtwx[[r, c]] = xtwx[[r, c]] + wi * xi[r] * xi[c];
            }
        }
    }

    // Add L2 regularization. The IRLS normal equations are at the SUMMED-deviance
    // scale, so to match sklearn's MEAN-deviance objective the diagonal penalty is
    // `weight_sum * alpha` (glm.py:229-242). The intercept column is excluded:
    // sklearn's `l2_reg_strength = self.alpha` weights only `||coef||^2`
    // (glm.py:258), never the intercept.
    let penalty = weight_sum * alpha;
    for i in 0..n_features {
        if Some(i) == intercept_col {
            continue;
        }
        xtwx[[i, i]] = xtwx[[i, i]] + penalty;
    }

    cholesky_solve(&xtwx, &xtwz).or_else(|_| gaussian_solve(n_features, &xtwx, &xtwz))
}

/// Core IRLS fitting logic shared by all GLM variants.
///
/// The IRLS update is parameterized by the [`Link`]: with linear predictor
/// `eta = X @ coef`, mean `mu = link.inverse(eta)` and link derivative
/// `dmu/deta`, the standard Fisher-scoring working weight and response are
/// `w = (dmu/deta)^2 / V(mu)` and `z = eta + (y - mu) / (dmu/deta)`
/// (`glm.py:362` for the inverse-link mapping). For [`Link::Log`]
/// (`dmu/deta = mu`) this is `w = mu^2 / V(mu)`, `z = eta + (y - mu)/mu`,
/// byte-identical to the previous log-only code. For [`Link::Identity`] with
/// `V(mu) = mu^0 = 1` (Normal/`power = 0`), `w = 1`, `z = y`, so IRLS reduces to
/// ordinary least squares.
#[allow(
    clippy::too_many_arguments,
    reason = "shared IRLS core threads the link \
    alongside the family/penalty/convergence parameters; splitting into a config \
    struct would obscure the 1:1 mapping to sklearn's fit signature"
)]
fn fit_glm_irls<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    family: &GLMFamily,
    link: Link,
    alpha: F,
    max_iter: usize,
    tol: F,
    fit_intercept: bool,
) -> Result<FittedGLMRegressor<F>, FerroError> {
    let (n_samples, n_features_orig) = x.dim();

    if n_samples != y.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "y length must match number of samples in X".into(),
        });
    }

    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "GLM requires at least one sample".into(),
        });
    }

    if alpha < F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "alpha".into(),
            reason: "must be non-negative".into(),
        });
    }

    // The log link requires y >= 0 (mu = exp(eta) > 0 and the working response
    // uses ln(y)). The identity link (Tweedie power <= 0, Normal) has no such
    // restriction — y may be any real number (`glm.py:121-127`,
    // `HalfSquaredError` / `HalfTweedieLossIdentity` target domains).
    let min_y = F::from(1e-10).unwrap_or_else(F::epsilon);
    if link == Link::Log {
        for &yi in y.iter() {
            if yi < F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "y".into(),
                    reason: "target values must be non-negative for GLM with log link".into(),
                });
            }
        }
    }

    // Build design matrix (optionally prepend intercept column).
    let n_cols = if fit_intercept {
        n_features_orig + 1
    } else {
        n_features_orig
    };

    let mut x_design = Array2::<F>::zeros((n_samples, n_cols));
    if fit_intercept {
        for i in 0..n_samples {
            x_design[[i, 0]] = F::one();
            for j in 0..n_features_orig {
                x_design[[i, j + 1]] = x[[i, j]];
            }
        }
    } else {
        x_design.assign(x);
    }

    // The intercept (column 0 of the augmented design when fitting one) is
    // excluded from the L2 penalty, matching sklearn (glm.py:258).
    let intercept_col = if fit_intercept { Some(0) } else { None };

    // Sum of sample weights. Unweighted GLM => every weight is 1, so this is
    // `n_samples`. It scales the L2 penalty so the summed-deviance IRLS normal
    // equations minimize sklearn's mean-deviance objective (glm.py:229-242).
    let weight_sum = F::from(n_samples).unwrap_or_else(|| {
        // n_samples >= 1 was checked above; fall back by accumulating ones.
        (0..n_samples).fold(F::zero(), |acc, _| acc + F::one())
    });

    // For the log link, clamp y away from 0 so `ln(y)` and `mu` stay finite.
    // For the identity link y is used as-is (mu = eta = y, no positivity
    // constraint).
    let y_safe: Array1<F> = match link {
        Link::Log => y.mapv(|v| if v < min_y { min_y } else { v }),
        Link::Identity => y.clone(),
    };

    // Initialise mu = y_safe, eta = g(mu) = link(mu): log link → ln(y),
    // identity link → y.
    let mut mu: Array1<F> = y_safe.clone();
    let mut eta: Array1<F> = match link {
        Link::Log => y_safe.mapv(|v| v.ln()),
        Link::Identity => y_safe.clone(),
    };
    let mut coef = Array1::<F>::zeros(n_cols);

    let min_mu = F::from(1e-10).unwrap_or_else(F::epsilon);
    let max_mu = F::from(1e10).unwrap_or_else(F::max_value);

    for _iter in 0..max_iter {
        let coef_old = coef.clone();

        // Compute IRLS weights and working response.
        let mut weights = Array1::<F>::zeros(n_samples);
        let mut z = Array1::<F>::zeros(n_samples);

        for i in 0..n_samples {
            // IRLS (Fisher scoring) with the configured link:
            //   dmu/deta  : Log => mu, Identity => 1
            //   weight w  = (dmu/deta)^2 / V(mu)
            //   response z = eta + (y - mu) / (dmu/deta)
            // For Log this is `w = mu^2/V(mu)`, `z = eta + (y - mu)/mu`,
            // byte-identical to the previous log-only code (clamped `mu_i`
            // throughout). For Identity + power=0 (V=1): w=1, z=y => OLS.
            match link {
                Link::Log => {
                    let mu_i = mu[i].max(min_mu).min(max_mu);
                    let var_i = family.variance(mu_i).max(min_mu);
                    let g_prime = F::one() / mu_i; // derivative of log link
                    z[i] = eta[i] + (y_safe[i] - mu_i) * g_prime;
                    weights[i] = F::one() / (g_prime * g_prime * var_i);
                }
                Link::Identity => {
                    let mu_i = mu[i];
                    // V(mu) for the identity link can see mu <= 0 (eta is
                    // unbounded); for power=0, V(mu)=mu^0=1 always. Clamp the
                    // magnitude for non-zero powers so V stays finite/positive.
                    let var_i = family.variance(mu_i.abs().max(min_mu)).max(min_mu);
                    let dmu_deta = link.dmu_deta(mu_i); // = 1
                    z[i] = eta[i] + (y_safe[i] - mu_i) / dmu_deta;
                    weights[i] = dmu_deta * dmu_deta / var_i;
                }
            }
            // Clamp weight.
            if weights[i] < min_mu {
                weights[i] = min_mu;
            }
        }

        // Solve weighted ridge. `weight_sum` = sum of sample weights (= n_samples
        // for the unweighted case); the penalty is scaled by it so the
        // summed-deviance normal equations minimize sklearn's mean-deviance
        // objective (glm.py:229-242). The intercept column (column 0 of the
        // augmented design, present iff `fit_intercept`) is left unpenalized
        // (glm.py:258, `l2_reg_strength = self.alpha` weighs only `||coef||^2`).
        coef = weighted_ridge_solve(&x_design, &z, &weights, alpha, weight_sum, intercept_col)?;

        // Update eta = X @ coef and mu = link.inverse(eta).
        eta = x_design.dot(&coef);
        match link {
            Link::Log => {
                let hi = F::from(20.0).unwrap_or_else(F::max_value);
                let lo = F::zero() - hi;
                for i in 0..n_samples {
                    // Clamp eta to prevent overflow in exp.
                    let eta_i = eta[i].max(lo).min(hi);
                    eta[i] = eta_i;
                    mu[i] = link.inverse(eta_i).max(min_mu).min(max_mu);
                }
            }
            Link::Identity => {
                // Identity link: eta is unbounded; mu = eta (no exp clamp).
                for i in 0..n_samples {
                    mu[i] = link.inverse(eta[i]);
                }
            }
        }

        // Check convergence.
        let max_change = coef
            .iter()
            .zip(coef_old.iter())
            .map(|(&c, &co)| (c - co).abs())
            .fold(F::zero(), |a, b| if b > a { b } else { a });

        if max_change < tol {
            break;
        }
    }

    // Extract intercept and feature coefficients.
    let (intercept, coefficients) = if fit_intercept {
        let intercept = coef[0];
        let coefficients = Array1::from_iter(coef.iter().skip(1).copied());
        (intercept, coefficients)
    } else {
        (F::zero(), coef)
    };

    Ok(FittedGLMRegressor {
        coefficients,
        intercept,
        link,
    })
}

// ---------------------------------------------------------------------------
// Fit — GLMRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for GLMRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the GLM via IRLS.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::InvalidParameter`] — negative alpha or negative y.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        // GLMRegressor's families are all log-link (Poisson/Gamma/Tweedie>0).
        // The Tweedie identity link is exposed only through TweedieRegressor's
        // `link` configuration (sklearn similarly only exposes `link` on
        // `TweedieRegressor`, `glm.py:861`).
        fit_glm_irls(
            x,
            y,
            &self.family,
            Link::Log,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
        )
    }
}

// ---------------------------------------------------------------------------
// Fit — PoissonRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for PoissonRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the Poisson GLM via IRLS.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        // Poisson uses the log link only (`HalfPoissonLoss`, `glm.py:589-590`).
        fit_glm_irls(
            x,
            y,
            &GLMFamily::Poisson,
            Link::Log,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
        )
    }
}

// ---------------------------------------------------------------------------
// Fit — GammaRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for GammaRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the Gamma GLM via IRLS.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        // Gamma uses the log link only (`HalfGammaLoss`, `glm.py:721-722`).
        fit_glm_irls(
            x,
            y,
            &GLMFamily::Gamma,
            Link::Log,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
        )
    }
}

// ---------------------------------------------------------------------------
// Fit — TweedieRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for TweedieRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the Tweedie GLM via IRLS.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        // Resolve the link from the configuration and Tweedie power, mirroring
        // `TweedieRegressor._get_loss` (`glm.py:889-903`): `auto` selects
        // identity for `power <= 0` (Normal/OLS) and log for `power > 0`.
        let link = self.link.resolve(self.power);
        fit_glm_irls(
            x,
            y,
            &GLMFamily::Tweedie(self.power),
            link,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
        )
    }
}

// ---------------------------------------------------------------------------
// Predict / HasCoefficients / Pipeline — FittedGLMRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedGLMRegressor<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict using the fitted GLM.
    ///
    /// Computes `link.inverse(X @ coefficients + intercept)` (`glm.py:362`):
    /// `exp(...)` for a [`Link::Log`] model (Poisson/Gamma/Tweedie with
    /// `power > 0`), and the raw linear predictor `X @ coef + intercept` for a
    /// [`Link::Identity`] model (Tweedie with `power <= 0`, Normal).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        let eta = x.dot(&self.coefficients) + self.intercept;
        let link = self.link;
        Ok(eta.mapv(|v| link.inverse(v)))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedGLMRegressor<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration for GLMRegressor.
impl<F> PipelineEstimator<F> for GLMRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
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

impl<F> FittedPipelineEstimator<F> for FittedGLMRegressor<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// Pipeline integration for PoissonRegressor.
impl<F> PipelineEstimator<F> for PoissonRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
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

// Pipeline integration for GammaRegressor.
impl<F> PipelineEstimator<F> for GammaRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
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

// Pipeline integration for TweedieRegressor.
impl<F> PipelineEstimator<F> for TweedieRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // ---- GLMRegressor ----

    #[test]
    fn test_glm_poisson_defaults() {
        let m = GLMRegressor::<f64>::new(GLMFamily::Poisson);
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 100);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_glm_builder() {
        let m = GLMRegressor::<f64>::new(GLMFamily::Gamma)
            .with_alpha(0.5)
            .with_max_iter(200)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert_relative_eq!(m.alpha, 0.5);
        assert_eq!(m.max_iter, 200);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_glm_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        assert!(
            GLMRegressor::<f64>::new(GLMFamily::Poisson)
                .fit(&x, &y)
                .is_err()
        );
    }

    #[test]
    fn test_glm_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(
            GLMRegressor::<f64>::new(GLMFamily::Poisson)
                .with_alpha(-1.0)
                .fit(&x, &y)
                .is_err()
        );
    }

    #[test]
    fn test_glm_poisson_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        // Predictions should be positive.
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_glm_gamma_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GLMRegressor::<f64>::new(GLMFamily::Gamma)
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_glm_tweedie_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GLMRegressor::<f64>::new(GLMFamily::Tweedie(1.5))
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_glm_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .fit(&x, &y)
            .unwrap();
        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_glm_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_glm_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let model = GLMRegressor::<f64>::new(GLMFamily::Poisson).with_alpha(0.0);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- PoissonRegressor ----

    #[test]
    fn test_poisson_defaults() {
        let m = PoissonRegressor::<f64>::new();
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 100);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_poisson_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = PoissonRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_poisson_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let fitted = PoissonRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit_pipeline(&x, &y)
            .unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- GammaRegressor ----

    #[test]
    fn test_gamma_defaults() {
        let m = GammaRegressor::<f64>::new();
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 100);
    }

    #[test]
    fn test_gamma_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GammaRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_gamma_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let fitted = GammaRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit_pipeline(&x, &y)
            .unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- TweedieRegressor ----

    #[test]
    fn test_tweedie_defaults() {
        let m = TweedieRegressor::<f64>::new();
        // sklearn TweedieRegressor default power=0.0 (Normal), link='auto'
        // (glm.py:867, :870).
        assert_relative_eq!(m.power, 0.0);
        assert_eq!(m.link, LinkConfig::Auto);
        assert_relative_eq!(m.alpha, 1.0);
    }

    #[test]
    fn test_tweedie_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = TweedieRegressor::<f64>::new()
            .with_power(1.5)
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_tweedie_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let fitted = TweedieRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit_pipeline(&x, &y)
            .unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- Link ----

    #[test]
    fn test_link_inverse() {
        assert_relative_eq!(Link::Log.inverse(0.0_f64), 1.0);
        assert_relative_eq!(Link::Identity.inverse(3.5_f64), 3.5);
    }

    #[test]
    fn test_link_config_resolve_auto() {
        // glm.py:889-893: auto -> identity for power<=0, log for power>0.
        assert_eq!(LinkConfig::Auto.resolve(0.0), Link::Identity);
        assert_eq!(LinkConfig::Auto.resolve(-1.0), Link::Identity);
        assert_eq!(LinkConfig::Auto.resolve(1.5), Link::Log);
        assert_eq!(LinkConfig::Log.resolve(0.0), Link::Log);
        assert_eq!(LinkConfig::Identity.resolve(2.0), Link::Identity);
    }

    #[test]
    fn test_tweedie_with_link_builder() {
        let m = TweedieRegressor::<f64>::new().with_link(LinkConfig::Log);
        assert_eq!(m.link, LinkConfig::Log);
    }

    // ---- Variance function ----

    #[test]
    fn test_variance_poisson() {
        let v = GLMFamily::Poisson.variance(3.0_f64);
        assert_relative_eq!(v, 3.0);
    }

    #[test]
    fn test_variance_gamma() {
        let v = GLMFamily::Gamma.variance(3.0_f64);
        assert_relative_eq!(v, 9.0);
    }

    #[test]
    fn test_variance_tweedie() {
        let v = GLMFamily::Tweedie(1.5).variance(4.0_f64);
        assert_relative_eq!(v, 4.0_f64.powf(1.5), epsilon = 1e-10);
    }

    #[test]
    fn test_glm_negative_y() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, -2.0, 3.0];
        assert!(
            GLMRegressor::<f64>::new(GLMFamily::Poisson)
                .fit(&x, &y)
                .is_err()
        );
    }
}
