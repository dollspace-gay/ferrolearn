//! Huber Regressor — robust regression via joint `[coef, intercept, scale]`
//! L-BFGS optimization of the scale-aware Huber loss.
//!
//! This module provides [`HuberRegressor`], a robust regression estimator that
//! mirrors scikit-learn's `sklearn.linear_model.HuberRegressor`. Unlike OLS
//! (which uses squared loss), the Huber loss is quadratic for residuals smaller
//! than `epsilon * scale` and linear (i.e., MAE-like) for larger residuals,
//! making it substantially less sensitive to outliers.
//!
//! Following sklearn, the coefficients `w`, the intercept `c` and the scale
//! `sigma` are optimized JOINTLY. The objective (per
//! `sklearn/linear_model/_huber.py:18` `_huber_loss_and_gradient`) is
//!
//! ```text
//! L = n·sigma
//!   + Σ_{inlier}  r_i² / sigma                       (r = y - X·w - c)
//!   + Σ_{outlier} (2·epsilon·|r_i| − sigma·epsilon²)
//!   + alpha·‖w‖²
//! ```
//!
//! with the inlier/outlier split at `|r_i| > epsilon · sigma`
//! (`_huber.py:67`). The `n·sigma` term plus the per-inlier `/sigma` scaling
//! make `sigma` jointly estimable. The `alpha` penalty applies to `w` only,
//! never to the intercept or the scale (`_huber.py:111`, `:124`).
//!
//! The minimizer is ferrolearn's own L-BFGS (`crate::optim::lbfgs`). Because
//! that optimizer is unconstrained while sklearn bounds `sigma >= eps·10`
//! (`_huber.py:322-323`), we reparameterize the scale as
//! `sigma = exp(log_sigma)` and optimize over the unconstrained `log_sigma`,
//! transforming the scale gradient by the chain rule
//! (`∂L/∂log_sigma = sigma · ∂L/∂sigma`). The Huber objective is convex with a
//! unique minimum, so the reparameterized solve reaches sklearn's fixed point.
//!
//! Unlike OLS/Ridge, the intercept is a fit parameter inside the optimization,
//! NOT recovered by mean-centering — matching sklearn, which optimizes
//! `w[-2]` jointly (`_huber.py:344-345`).
//!
//! An L2 penalty (`alpha`) on the coefficients is also supported.
//!
//! ## REQ status
//!
//! See `.design/linear/huber_regressor.md` for the full requirement table.
//!
//! - REQ-1 (joint L-BFGS fit matches sklearn): SHIPPED — `fit` minimizes the
//!   scale-aware Huber objective of `_huber_loss_and_gradient` over
//!   `[coef, intercept, log_sigma]` with `crate::optim::lbfgs`. Consumer:
//!   `ferrolearn-python` `RsHuberRegressor`.
//! - REQ-2 (epsilon default 1.35): SHIPPED — `new` sets `epsilon = 1.35`.
//! - REQ-3 (alpha L2 on coef only): SHIPPED — the penalty term in
//!   `huber_loss_and_gradient` touches only `grad[..n_features]` / `‖w‖²`.
//! - REQ-4 (scale_ jointly estimated, > 0): SHIPPED — `sigma` is the last
//!   optimized parameter, reparameterized `exp(log_sigma)` so it stays > 0;
//!   surfaced as `scale_` / `scale()`.
//! - REQ-5 (outliers_ mask): SHIPPED — `outliers_` set where
//!   `|y − X·coef − intercept| > scale · epsilon`; surfaced as `outliers()`.
//! - REQ-6 (predict): SHIPPED — `Predict` returns `X·coef + intercept`.
//! - REQ-7 (fit_intercept / HasCoefficients): SHIPPED.
//! - REQ-8 (scale_ attribute): SHIPPED — `scale()` accessor.
//! - REQ-9 (n_iter_): SHIPPED (closes #499) — `FittedHuberRegressor` carries an
//!   `n_iter` field (the REAL L-BFGS iteration count), surfaced via
//!   `pub fn n_iter`. `fit_with_sample_weight` now calls the additive
//!   `crate::optim::lbfgs::LbfgsOptimizer::minimize_reporting` overload (which
//!   returns `(params, n_iters)`; `minimize` is unchanged and still used by
//!   `logistic_regression.rs`). Mirrors sklearn `self.n_iter_ = opt_res.nit`
//!   (`sklearn/linear_model/_huber.py:342`). R-DEV-7: ferrolearn's in-repo
//!   L-BFGS is not scipy's L-BFGS-B, so the raw count need not equal sklearn's
//!   exactly; the oracle-comparable contract shipped is positivity, `<= max_iter`,
//!   determinism, and warm < cold (sklearn cold `n_iter_=15`, warm `=1`).
//!   Consumer: `ferrolearn-python` `RsHuberRegressor::n_iter_` →
//!   `_extras.py::HuberRegressor.fit` (`self.n_iter_ = int(self._rs.n_iter_)`).
//! - REQ-10 (warm_start): SHIPPED — `pub warm_start: bool` +
//!   `pub warm_start_state: Option<(coef, intercept, scale)>` with
//!   `with_warm_start`/`with_warm_start_state`; when set, `fit_with_sample_weight`
//!   seeds the L-BFGS from the prior `(coef, intercept, scale)` instead of the
//!   IRLS cold start (sklearn `_huber.py:308-309`), the R-DEV-4 stand-in for
//!   sklearn's reused mutable `self.coef_`. The convex objective's unique minimum
//!   is unchanged; a warm refit converges in FEWER iterations. Consumer:
//!   `ferrolearn-python` `RsHuberRegressor` (`warm_start` ctor param +
//!   `coef_`/`intercept_`/`scale_` getters reused by `_extras.py::HuberRegressor`).
//! - REQ-11 (sample_weight): SHIPPED — `fit_with_sample_weight(x, y,
//!   sample_weight: Option<&Array1<F>>)` threads per-sample weights into
//!   `huber_loss_and_gradient` (each contribution × `w[i]`, `n_samples = Σ w`,
//!   sklearn `_huber.py:18`/`:59`/`:77-80`); `Fit::fit` delegates with `None`
//!   (byte-identical to the old path). Consumer: `RsHuberRegressor::fit(x, y,
//!   sample_weight=None)` → `_extras.py::HuberRegressor.fit(X, y,
//!   sample_weight=None)`.
//! - REQ-12 (ferray substrate): NOT-STARTED (blocker #502).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::HuberRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let model = HuberRegressor::<f64>::new();
//! // Noisy `y ≈ 2x + 1` inliers with one off-trend outlier at the end.
//! let x = Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
//! let y = array![3.35, 4.94, 6.90, 8.47, 11.00, 12.94, 14.89, 25.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use crate::optim::lbfgs::LbfgsOptimizer;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// Convert a finite `f64` constant to `F`. The conversion is total for `f32`
/// and `f64`; the `F::epsilon()` fallback is a never-taken safety branch that
/// keeps production code free of `.unwrap()` (goal.md R-CODE-2).
#[inline]
fn cast<F: Float + FromPrimitive>(v: f64) -> F {
    F::from(v).unwrap_or_else(F::epsilon)
}

/// Robust scale of a target vector: `1.4826 · median(|y − median(y)|)` (the
/// MAD, scaled to be a consistent estimator of the standard deviation under
/// Gaussian noise). Used only as the starting `sigma` for the Huber
/// optimization — being robust, it is not inflated by outliers, so the solver
/// can descend to the true scale from both clean and outlier-heavy data.
/// Returns `0` for an empty input.
fn robust_scale<F: Float + FromPrimitive>(y: &Array1<F>) -> F {
    let n = y.len();
    if n == 0 {
        return F::zero();
    }
    let median = |v: &mut [F]| -> F {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = v.len();
        if m % 2 == 1 {
            v[m / 2]
        } else {
            (v[m / 2 - 1] + v[m / 2]) / cast::<F>(2.0)
        }
    };
    let mut vals: Vec<F> = y.iter().copied().collect();
    let med = median(&mut vals);
    let mut dev: Vec<F> = y.iter().map(|&yi| (yi - med).abs()).collect();
    let mad = median(&mut dev);
    cast::<F>(1.4826) * mad
}

/// Solve the symmetric positive-definite system `a · β = b` (size `p`) by
/// Gaussian elimination with partial pivoting. Returns `None` on a (near-)
/// singular pivot. Used for the warm-start weighted-least-squares steps.
fn gauss_solve<F: Float + FromPrimitive>(a: &Array2<F>, b: &Array1<F>) -> Option<Array1<F>> {
    let p = b.len();
    let mut aug = Array2::<F>::zeros((p, p + 1));
    for r in 0..p {
        for c in 0..p {
            aug[[r, c]] = a[[r, c]];
        }
        aug[[r, p]] = b[r];
    }
    for col in 0..p {
        let mut piv = col;
        let mut best = aug[[col, col]].abs();
        for row in (col + 1)..p {
            let v = aug[[row, col]].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if best <= cast::<F>(1e-30) {
            return None;
        }
        if piv != col {
            for c in 0..=p {
                let t = aug[[col, c]];
                aug[[col, c]] = aug[[piv, c]];
                aug[[piv, c]] = t;
            }
        }
        let pivot = aug[[col, col]];
        for row in (col + 1)..p {
            let factor = aug[[row, col]] / pivot;
            for c in col..=p {
                let above = aug[[col, c]];
                aug[[row, c]] = aug[[row, c]] - factor * above;
            }
        }
    }
    let mut beta = Array1::<F>::zeros(p);
    for i in (0..p).rev() {
        let mut s = aug[[i, p]];
        for j in (i + 1)..p {
            s = s - aug[[i, j]] * beta[j];
        }
        let diag = aug[[i, i]];
        if diag.abs() <= cast::<F>(1e-30) {
            return None;
        }
        beta[i] = s / diag;
    }
    Some(beta)
}

/// Huber-IRLS warm start for the joint optimization. Runs a handful of
/// iteratively-reweighted-least-squares steps on `Z = [X | 1?]` with Huber
/// weights (`w_i = min(1, epsilon·sigma / |r_i|)`, `sigma` = MAD of the current
/// residuals), returning `(coef, intercept, sigma)`.
///
/// This is NOT the Huber fit — it only lands the L-BFGS start near the (unique,
/// convex) joint minimum so the unconstrained in-repo optimizer converges
/// quickly even on large-scale or heavy-outlier targets (where steepest descent
/// from the `coef = 0, sigma = 1` origin stalls). Unlike a plain OLS warm start,
/// the Huber weights down-weight outliers, so the start is not poisoned by them.
/// Returns `None` on any singular solve (caller falls back to the origin).
fn irls_warm_start<F: Float + FromPrimitive>(
    x: &Array2<F>,
    y: &Array1<F>,
    epsilon: F,
    fit_intercept: bool,
) -> Option<(Array1<F>, F, F)> {
    let (n_samples, n_features) = x.dim();
    let p = n_features + usize::from(fit_intercept);
    if p == 0 || n_samples == 0 {
        return None;
    }

    let mut beta = Array1::<F>::zeros(p);
    let mut sigma = robust_scale(y).max(cast::<F>(1e-3));

    // A few reweighting passes suffice to get within the convex basin.
    for _ in 0..8 {
        // Residuals r = y - Z·beta.
        let mut resid = Array1::<F>::zeros(n_samples);
        for i in 0..n_samples {
            let xi = x.row(i);
            let mut pred = if fit_intercept {
                beta[n_features]
            } else {
                F::zero()
            };
            for k in 0..n_features {
                pred = pred + beta[k] * xi[k];
            }
            resid[i] = y[i] - pred;
        }
        // Robust scale of the residuals (MAD), floored.
        sigma = robust_scale(&resid).max(cast::<F>(1e-3));
        let band = epsilon * sigma;

        // Huber weights and the weighted normal equations Zᵀ W Z β = Zᵀ W y.
        let mut ata = Array2::<F>::zeros((p, p));
        let mut aty = Array1::<F>::zeros(p);
        for i in 0..n_samples {
            let xi = x.row(i);
            let ar = resid[i].abs();
            let w = if ar <= band {
                F::one()
            } else {
                (band / ar).max(cast::<F>(1e-10))
            };
            let yi = y[i];
            for r in 0..p {
                let zr = if r < n_features { xi[r] } else { F::one() };
                aty[r] = aty[r] + w * zr * yi;
                for c in 0..p {
                    let zc = if c < n_features { xi[c] } else { F::one() };
                    ata[[r, c]] = ata[[r, c]] + w * zr * zc;
                }
            }
        }
        for d in 0..p {
            ata[[d, d]] = ata[[d, d]] + cast::<F>(1e-8);
        }
        beta = gauss_solve(&ata, &aty)?;
    }

    let coef = beta.slice(ndarray::s![..n_features]).to_owned();
    let intercept = if fit_intercept {
        beta[n_features]
    } else {
        F::zero()
    };
    Some((coef, intercept, sigma.max(cast::<F>(1e-3))))
}

/// Huber Regressor — robust regression less sensitive to outliers.
///
/// Fits by jointly minimizing the scale-aware Huber loss over
/// `[coef, intercept, scale]` with L-BFGS, mirroring
/// `sklearn.linear_model.HuberRegressor`. Samples whose scaled residual
/// `|r| / sigma` exceeds `epsilon` contribute a linear (robust) loss instead
/// of the quadratic loss, limiting outlier influence.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct HuberRegressor<F> {
    /// Threshold between quadratic and linear Huber loss regions.
    /// Typically around 1.35 (the default), which gives ~95% efficiency
    /// for Gaussian-distributed errors.
    pub epsilon: F,
    /// L2 regularization strength applied to the coefficients.
    pub alpha: F,
    /// Maximum number of L-BFGS iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the projected-gradient norm.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
    /// When `true`, initialize the joint optimization from a prior solution
    /// (`coef_init`/`intercept_init`/`scale_init`) instead of the cold IRLS
    /// warm start.
    ///
    /// Mirrors sklearn `HuberRegressor(warm_start=False)`
    /// (`sklearn/linear_model/_huber.py:265`), which "reuse[s] the stored
    /// attributes of a previously used model" — `self.coef_`, `self.intercept_`
    /// and `self.scale_` become the optimizer's start vector
    /// (`_huber.py:308-309`: `parameters = np.concatenate((self.coef_,
    /// [self.intercept_, self.scale_]))`).
    ///
    /// R-DEV-4 adaptation: ferrolearn estimators are immutable value types —
    /// there is no mutable `self.coef_`/`scale_` carried across repeated
    /// `.fit()` calls — so the prior solution is supplied EXPLICITLY through
    /// [`HuberRegressor::warm_start_state`] (mirroring `lasso.rs`/`elastic_net.rs`
    /// `coef_init`). The Huber objective is convex with a unique minimum, so the
    /// converged fit is unchanged; only the path (and iteration count) changes.
    pub warm_start: bool,
    /// Explicit warm-start seed `(coef, intercept, scale)` used when
    /// [`HuberRegressor::warm_start`] is `true` (the R-DEV-4 stand-in for
    /// sklearn's reused `self.coef_`/`self.intercept_`/`self.scale_`).
    ///
    /// `None` (the default) — or `warm_start == false` — uses the IRLS cold
    /// start, the byte-identical pre-`warm_start` path. When `Some`, the coef
    /// length must equal `n_features` and the scale must be strictly positive.
    pub warm_start_state: Option<(Array1<F>, F, F)>,
}

impl<F: Float + FromPrimitive> HuberRegressor<F> {
    /// Create a new `HuberRegressor` with default settings.
    ///
    /// Defaults: `epsilon = 1.35`, `alpha = 0.0001`, `max_iter = 100`,
    /// `tol = 1e-5`, `fit_intercept = true` — matching sklearn's
    /// `HuberRegressor.__init__` (`sklearn/linear_model/_huber.py:259-274`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            epsilon: cast(1.35),
            alpha: cast(1e-4),
            max_iter: 100,
            tol: cast(1e-5),
            fit_intercept: true,
            warm_start: false,
            warm_start_state: None,
        }
    }

    /// Set the Huber threshold `epsilon`.
    ///
    /// Must be strictly greater than 1.0.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: F) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of L-BFGS iterations.
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

    /// Enable or disable `warm_start`.
    ///
    /// Mirrors `sklearn.linear_model.HuberRegressor(warm_start=...)`
    /// (`sklearn/linear_model/_huber.py:265`). When `true` AND a
    /// [`HuberRegressor::warm_start_state`] is supplied, the joint optimization
    /// is seeded from that prior `(coef, intercept, scale)` instead of the IRLS
    /// cold start; `warm_start` alone (no state) is a no-op falling back to the
    /// cold path.
    #[must_use]
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Supply the explicit warm-start seed `(coef, intercept, scale)` reused
    /// when [`HuberRegressor::warm_start`] is `true`.
    ///
    /// This is the R-DEV-4 stand-in for sklearn's reused mutable
    /// `self.coef_`/`self.intercept_`/`self.scale_` (`_huber.py:308-309`):
    /// because ferrolearn estimators are immutable value types, the prior fit's
    /// attributes are threaded in explicitly (mirroring `lasso.rs`/
    /// `elastic_net.rs` `with_coef_init`). The coef length must equal
    /// `n_features` and the scale must be `> 0`.
    #[must_use]
    pub fn with_warm_start_state(mut self, coef: Array1<F>, intercept: F, scale: F) -> Self {
        self.warm_start_state = Some((coef, intercept, scale));
        self
    }
}

impl<F: Float + FromPrimitive> Default for HuberRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Huber Regressor model.
///
/// Stores the learned coefficients, intercept, jointly-estimated scale and the
/// outlier mask. Implements [`Predict`] and [`HasCoefficients`].
#[derive(Debug, Clone)]
pub struct FittedHuberRegressor<F> {
    /// Learned coefficient vector (sklearn `coef_`).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term (sklearn `intercept_`).
    intercept: F,
    /// Jointly-estimated scale `sigma` (sklearn `scale_`); strictly positive.
    scale: F,
    /// Boolean outlier mask: `|y − X·coef − intercept| > scale · epsilon`
    /// (sklearn `outliers_`).
    outliers: Array1<bool>,
    /// Number of L-BFGS optimizer iterations performed during the fit (sklearn
    /// `n_iter_`). A positive integer `<= max_iter`. NOTE (R-DEV-7): ferrolearn's
    /// in-repo L-BFGS is NOT scipy's bounded L-BFGS-B, so this is the REAL count
    /// of ferrolearn iterations and need not equal sklearn's `opt_res.nit`
    /// exactly; the meaningful, oracle-comparable property — a warm-start refit
    /// taking FEWER iterations than the cold fit — is preserved.
    n_iter: usize,
}

/// Huber loss and gradient over the joint parameter vector
/// `params = [coef.., (intercept,) log_sigma]`.
///
/// Translates `_huber_loss_and_gradient` (`sklearn/linear_model/_huber.py:18`)
/// with the scale reparameterized as `sigma = exp(log_sigma)`: the scale
/// gradient is multiplied by `sigma` (chain rule) so the caller optimizes the
/// unconstrained `log_sigma`, keeping `sigma > 0` without bounds.
///
/// `sample_weight` is the per-sample weight vector (sklearn's `sample_weight`,
/// `sklearn/linear_model/_huber.py:18`). `n_samples = Σ sample_weight` (sklearn
/// `_huber.py:59` `n_samples = np.sum(sample_weight)`), and every per-sample
/// loss/gradient contribution is multiplied by that sample's weight
/// (`_huber.py:77-80`, `:87-88`, `:107`, `:120-121`). With unit weights this is
/// byte-identical to the unweighted objective.
fn huber_loss_and_gradient<F: Float + FromPrimitive + ScalarOperand + 'static>(
    params: &Array1<F>,
    x: &Array2<F>,
    y: &Array1<F>,
    sample_weight: &Array1<F>,
    epsilon: F,
    alpha: F,
    fit_intercept: bool,
) -> (F, Array1<F>) {
    let (n_samples, n_features) = x.dim();
    let two = cast::<F>(2.0);

    // Unpack: coef = params[..n_features]; (intercept = params[n_features];)
    // log_sigma = params[last]; sigma = exp(log_sigma), floored to mirror
    // sklearn's lower bound `bounds[-1][0] = eps * 10`
    // (`sklearn/linear_model/_huber.py:323`). On a near-perfect fit the
    // residuals vanish and sigma drives toward 0; flooring sigma — and zeroing
    // its gradient inside the clamped region (`d sigma / d log_sigma = 0`
    // there) — reproduces the projected-gradient behaviour of L-BFGS-B at its
    // bound, which our unconstrained optimizer would otherwise chase to -inf.
    let sigma_floor = cast::<F>(f64::EPSILON * 10.0);
    let log_sigma = params[params.len() - 1];
    let sigma_raw = log_sigma.exp();
    let clamped = sigma_raw < sigma_floor;
    let sigma = if clamped { sigma_floor } else { sigma_raw };
    let intercept = if fit_intercept {
        params[n_features]
    } else {
        F::zero()
    };

    // linear_loss = y - X·coef - intercept   (sklearn `_huber.py:63-65`)
    let coef = params.slice(ndarray::s![..n_features]).to_owned();
    let mut linear_loss = y - &x.dot(&coef);
    if fit_intercept {
        linear_loss.mapv_inplace(|v| v - intercept);
    }

    let threshold = epsilon * sigma;
    let eps2 = epsilon * epsilon;

    // Accumulators.
    let mut grad = Array1::<F>::zeros(params.len());
    let mut squared_loss = F::zero(); // Σ_inlier r²
    let mut outlier_abs_sum = F::zero(); // Σ_outlier |r|
    let mut num_outliers = F::zero(); // count of outliers (as F)
    let mut sum_inlier_r = F::zero(); // Σ_inlier r  (for intercept grad)
    let mut sum_signed_outliers = F::zero(); // Σ_outlier sign(r)

    let mut n_sw = F::zero(); // Σ sample_weight  (sklearn `n_samples`, :59)
    for i in 0..n_samples {
        let sw = sample_weight[i];
        n_sw = n_sw + sw;
        let r = linear_loss[i];
        let abs_r = r.abs();
        let xi = x.row(i);
        if abs_r > threshold {
            // Outlier: linear loss.   (sklearn `_huber.py:69-82`, :102-108)
            // Every contribution is weighted by `sw` (`:77-80`, `:107`).
            let sign = if r < F::zero() { -F::one() } else { F::one() };
            // grad[:n_features] -= 2·epsilon·(sw·sign) · X[i]
            for k in 0..n_features {
                grad[k] = grad[k] - two * epsilon * sw * sign * xi[k];
            }
            outlier_abs_sum = outlier_abs_sum + sw * abs_r;
            num_outliers = num_outliers + sw;
            sum_signed_outliers = sum_signed_outliers + sw * sign;
        } else {
            // Inlier: quadratic loss.  (sklearn `_huber.py:84-100`)
            // weighted_non_outliers = sw·r; grad += (2/sigma)·(-sw·r)·X[i].
            let wr = sw * r;
            let g = -(two / sigma) * wr;
            for k in 0..n_features {
                grad[k] = grad[k] + g * xi[k];
            }
            squared_loss = squared_loss + wr * r;
            sum_inlier_r = sum_inlier_r + wr;
        }
    }

    // Gradient due to the penalty: grad[:n_features] += 2·alpha·coef
    // (sklearn `_huber.py:111`).
    for k in 0..n_features {
        grad[k] = grad[k] + two * alpha * coef[k];
    }

    // sklearn's `n_samples` is `np.sum(sample_weight)` (`_huber.py:59`), NOT the
    // raw sample count — under unit weights the two coincide.
    let n = n_sw;

    // Gradient due to sigma (sklearn `_huber.py:114-116`):
    //   grad_sigma = n - num_outliers·epsilon² - (squared_loss / sigma) / sigma
    // Chain rule for log_sigma: grad_logsigma = sigma · grad_sigma.
    let squared_loss_over_sigma = squared_loss / sigma;
    let grad_sigma = n - num_outliers * eps2 - squared_loss_over_sigma / sigma;
    let last = params.len() - 1;
    // In the clamped region `d sigma / d log_sigma = 0`, so the projected
    // gradient on `log_sigma` is zero (mirrors L-BFGS-B at its bound).
    grad[last] = if clamped {
        F::zero()
    } else {
        sigma * grad_sigma
    };

    // Gradient due to the intercept (sklearn `_huber.py:119-121`):
    //   grad[-2] = -2·Σ_inlier r / sigma - 2·epsilon·Σ_outlier sign(r)
    if fit_intercept {
        grad[n_features] = -(two * sum_inlier_r) / sigma - two * epsilon * sum_signed_outliers;
    }

    // loss = n·sigma + squared_loss/sigma
    //      + (2·epsilon·Σ_outlier|r| - sigma·num_outliers·epsilon²)
    //      + alpha·‖coef‖²            (sklearn `_huber.py:79-82`, :123-124)
    let outlier_loss = two * epsilon * outlier_abs_sum - sigma * num_outliers * eps2;
    let penalty = alpha * coef.dot(&coef);
    let loss = n * sigma + squared_loss_over_sigma + outlier_loss + penalty;

    (loss, grad)
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for HuberRegressor<F>
{
    type Fitted = FittedHuberRegressor<F>;
    type Error = FerroError;

    /// Fit the Huber Regressor by jointly minimizing the scale-aware Huber loss.
    ///
    /// Mirrors `sklearn.linear_model.HuberRegressor.fit`
    /// (`sklearn/linear_model/_huber.py:325`): the parameter vector
    /// `[coef, intercept?, sigma]` is optimized with L-BFGS over
    /// `_huber_loss_and_gradient`. The scale `sigma` is kept strictly positive
    /// by optimizing `log_sigma` (sklearn instead bounds `sigma >= eps·10`,
    /// `_huber.py:322-323`); the convex objective has a unique minimum so both
    /// reach the same fit. The intercept is a fit parameter, NOT recovered by
    /// mean-centering.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InvalidParameter`] — `epsilon <= 1.0` or negative `alpha`.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::NumericalInstability`] / [`FerroError::ConvergenceFailure`]
    ///   — optimizer failure.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedHuberRegressor<F>, FerroError> {
        // Unit weights: byte-identical to the pre-`sample_weight` cold path.
        self.fit_with_sample_weight(x, y, None)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> HuberRegressor<F> {
    /// Fit the Huber Regressor with optional per-sample weights, mirroring
    /// `sklearn.linear_model.HuberRegressor.fit(X, y, sample_weight=None)`
    /// (`sklearn/linear_model/_huber.py:277`, `:306`).
    ///
    /// `sample_weight = None` is byte-identical to [`Fit::fit`] (unit weights,
    /// sklearn `_check_sample_weight` returns `np.ones(n_samples)`,
    /// `_huber.py:306`). When `Some(w)`, each sample's loss/gradient contribution
    /// is multiplied by `w[i]` and `n_samples = Σ w` inside
    /// [`huber_loss_and_gradient`] (sklearn `_huber.py:18`, `:59`, `:77-80`).
    ///
    /// When [`HuberRegressor::warm_start`] is `true` and a
    /// [`HuberRegressor::warm_start_state`] is present, the joint optimization is
    /// seeded from that prior `(coef, intercept, scale)` rather than the IRLS
    /// cold start (sklearn `_huber.py:308-309`); the convex objective's unique
    /// minimum is unchanged, only the path (and iteration count) shortens.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — `y` / `sample_weight` / warm-start coef
    ///   length mismatch.
    /// - [`FerroError::InvalidParameter`] — `epsilon <= 1.0`, negative `alpha`,
    ///   a negative weight, or a non-positive warm-start scale.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::NumericalInstability`] / [`FerroError::ConvergenceFailure`]
    ///   — optimizer failure.
    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedHuberRegressor<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.epsilon <= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "epsilon".into(),
                reason: "must be strictly greater than 1.0".into(),
            });
        }

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "HuberRegressor requires at least one sample".into(),
            });
        }

        // Materialize sample weights: `None` -> unit weights, mirroring sklearn
        // `_check_sample_weight(None, X) == np.ones(n_samples)`
        // (`sklearn/linear_model/_huber.py:306`).
        let weights: Array1<F> = match sample_weight {
            None => Array1::<F>::ones(n_samples),
            Some(w) => {
                if w.len() != n_samples {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![n_samples],
                        actual: vec![w.len()],
                        context: "sample_weight length must match number of samples in X".into(),
                    });
                }
                // No non-negativity constraint: sklearn 1.5.2's `HuberRegressor.fit`
                // runs `sample_weight` through `_check_sample_weight`
                // (`sklearn/linear_model/_huber.py:306`), which validates length/dtype
                // but NOT sign — negative weights flow straight into
                // `_huber_loss_and_gradient` and the fit converges (#2159).
                w.clone()
            }
        };

        // Parameter layout: [coef (n_features), intercept (if fit_intercept),
        // log_sigma]. sklearn inits coef/intercept to 0 and `sigma` to 1
        // (`sklearn/linear_model/_huber.py:311-317`) and relies on the bounded
        // L-BFGS-B solver. Our unconstrained in-repo L-BFGS stalls at that
        // origin on poorly-scaled or heavy-outlier data (every sample is an
        // outlier, the ill-scaled steepest-descent step makes no progress). We
        // therefore warm-start from a few Huber-IRLS reweighting steps — which
        // land coef/intercept/sigma near the joint minimum and, being Huber-
        // weighted, are robust to the outliers (a plain OLS start would be
        // poisoned by them). The objective is convex with a unique minimum, so
        // this start does not change the converged fit, only the path to it
        // (R-DEV-7); on a singular solve we fall back to sklearn's origin.
        let n_params = n_features + usize::from(self.fit_intercept) + 1;
        let mut x0 = Array1::<F>::zeros(n_params);

        // warm_start (REQ-10): if enabled with an explicit prior solution, seed
        // the optimizer from it directly (sklearn `_huber.py:308-309`
        // `parameters = np.concatenate((self.coef_, [self.intercept_,
        // self.scale_]))`). The R-DEV-4 stand-in for sklearn's reused mutable
        // `self.coef_`. Otherwise fall back to the IRLS cold start.
        let warm = if self.warm_start {
            self.warm_start_state.as_ref()
        } else {
            None
        };
        if let Some((coef0, intercept0, scale0)) = warm {
            if coef0.len() != n_features {
                return Err(FerroError::ShapeMismatch {
                    expected: vec![n_features],
                    actual: vec![coef0.len()],
                    context: "warm_start coef length must equal number of features".into(),
                });
            }
            if *scale0 <= F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "warm_start_state.scale".into(),
                    reason: "scale must be strictly positive".into(),
                });
            }
            for k in 0..n_features {
                x0[k] = coef0[k];
            }
            if self.fit_intercept {
                x0[n_features] = *intercept0;
            }
            x0[n_params - 1] = scale0.ln();
        } else if let Some((coef0, intercept0, sigma0)) =
            irls_warm_start(x, y, self.epsilon, self.fit_intercept)
        {
            for k in 0..n_features {
                x0[k] = coef0[k];
            }
            if self.fit_intercept {
                x0[n_features] = intercept0;
            }
            // Start `sigma` from ABOVE the optimum (the numerically stable
            // descent direction): a moderate floor of 0.1 keeps the `2/sigma`
            // inlier-gradient factor bounded at the start even when the IRLS fit
            // is near-perfect (`sigma0 → 0`), where a tiny start would make that
            // factor explode and stall the line search. The converged minimum
            // is unchanged.
            x0[n_params - 1] = sigma0.max(cast::<F>(0.1)).ln();
        } else {
            // Fall back to a robust `sigma` start at the origin.
            x0[n_params - 1] = robust_scale(y).max(cast::<F>(1.0)).ln();
        }

        let epsilon = self.epsilon;
        let alpha = self.alpha;
        let fit_intercept = self.fit_intercept;

        let optimizer = LbfgsOptimizer::<F>::new(self.max_iter, self.tol);
        // `minimize_reporting` also returns the optimizer iteration count
        // (sklearn `self.n_iter_ = opt_res.nit`, `sklearn/linear_model/_huber.py:342`).
        // `minimize` itself is unchanged (and still used by `logistic_regression.rs`).
        let (params, n_iter) = optimizer.minimize_reporting(
            |p| huber_loss_and_gradient(p, x, y, &weights, epsilon, alpha, fit_intercept),
            x0,
        )?;

        // Extract fitted attributes (sklearn `_huber.py:343-351`).
        let coefficients = params.slice(ndarray::s![..n_features]).to_owned();
        let intercept = if self.fit_intercept {
            params[n_features]
        } else {
            F::zero()
        };
        // Mirror sklearn's bounded scale (`bounds[-1][0] = eps * 10`,
        // `sklearn/linear_model/_huber.py:323`): never report below the floor.
        let sigma_floor = cast::<F>(f64::EPSILON * 10.0);
        let scale = params[params.len() - 1].exp().max(sigma_floor);

        // outliers_ = |y - X·coef - intercept| > scale · epsilon
        // (sklearn `_huber.py:350-351`).
        let mut residual = y - &x.dot(&coefficients);
        residual.mapv_inplace(|v| (v - intercept).abs());
        let band = scale * self.epsilon;
        let outliers = residual.mapv(|r| r > band);

        Ok(FittedHuberRegressor {
            coefficients,
            intercept,
            scale,
            outliers,
            n_iter,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> FittedHuberRegressor<F> {
    /// The jointly-estimated scale `sigma` (sklearn `scale_`).
    ///
    /// Always strictly positive. This is the value by which `|y − X·coef − c|`
    /// is scaled when classifying a sample as an outlier.
    #[must_use]
    pub fn scale(&self) -> F {
        self.scale
    }

    /// The boolean outlier mask (sklearn `outliers_`).
    ///
    /// Element `i` is `true` where `|y_i − X_i·coef − intercept| > scale · epsilon`.
    #[must_use]
    pub fn outliers(&self) -> &Array1<bool> {
        &self.outliers
    }

    /// Number of L-BFGS optimizer iterations performed during the fit (sklearn
    /// `n_iter_`, `sklearn/linear_model/_huber.py:342` `self.n_iter_ =
    /// opt_res.nit`). A positive integer `<= max_iter`.
    ///
    /// R-DEV-7: ferrolearn's in-repo L-BFGS is not scipy's bounded L-BFGS-B, so
    /// this is the genuine count of ferrolearn iterations and is not guaranteed
    /// equal to sklearn's `n_iter_`. The oracle-comparable contract that DOES
    /// hold — and that sklearn also exhibits — is that a warm-start refit
    /// converges in FEWER iterations than the cold fit.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedHuberRegressor<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedHuberRegressor<F>
{
    /// Returns the learned coefficient vector.
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    /// Returns the learned intercept term.
    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for HuberRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
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

impl<F> FittedPipelineEstimator<F> for FittedHuberRegressor<F>
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
        let m = HuberRegressor::<f64>::new();
        assert_relative_eq!(m.epsilon, 1.35);
        assert_relative_eq!(m.alpha, 1e-4);
        assert_eq!(m.max_iter, 100);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_builder_setters() {
        let m = HuberRegressor::<f64>::new()
            .with_epsilon(2.0)
            .with_alpha(0.1)
            .with_max_iter(50)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert_relative_eq!(m.epsilon, 2.0);
        assert_relative_eq!(m.alpha, 0.1);
        assert_eq!(m.max_iter, 50);
        assert!(!m.fit_intercept);
    }

    // ---- Validation errors ----

    #[test]
    fn test_epsilon_too_small_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = HuberRegressor::<f64>::new().with_epsilon(0.5).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_epsilon_exactly_one_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = HuberRegressor::<f64>::new().with_epsilon(1.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_alpha_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = HuberRegressor::<f64>::new().with_alpha(-1.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        let result = HuberRegressor::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    // ---- Correctness ----
    //
    // The datasets below carry mild Gaussian noise (seeded NumPy `RandomState`
    // draws) so the Huber scale `sigma` settles at a realistic value (~0.05–0.2)
    // rather than collapsing to 0 on a perfect fit — the regime in which the
    // joint `[coef, intercept, sigma]` optimization is meaningful and the
    // in-repo L-BFGS converges (the perfect-fit degeneracy where `2/sigma` blows
    // up is the substrate concern tracked as REQ-12 / blocker #502).

    #[test]
    fn test_fits_clean_linear_data() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = HuberRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.1);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 0.5);
    }

    #[test]
    fn test_robust_to_outliers() {
        // 11 inliers following y ≈ 2x + 1 (mild noise: RandomState(7) draws of
        // 2x+1+0.2·N(0,1), so scale ≈ 0.1) plus 1 large outlier at the end.
        // With majority inliers, Huber should be much more robust than OLS.
        let x = Array2::from_shape_vec(
            (12, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y_clean = array![
            3.3381, 4.9068, 7.0066, 9.0815, 10.8422, 13.0004, 14.9998, 16.6491, 19.2035, 21.1201,
            22.8749, 24.9657
        ];
        let y_outlier = array![
            3.3381, 4.9068, 7.0066, 9.0815, 10.8422, 13.0004, 14.9998, 16.6491, 19.2035, 21.1201,
            22.8749, 200.0
        ];

        let fitted_clean = HuberRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y_clean)
            .unwrap();

        let fitted_huber = HuberRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y_outlier)
            .unwrap();

        // OLS on the outlier data, pulled high by the outlier.
        let ols_coef = {
            let n = 12.0_f64;
            let xv: Vec<f64> = (1..=12).map(f64::from).collect();
            let yv = vec![
                3.3381, 4.9068, 7.0066, 9.0815, 10.8422, 13.0004, 14.9998, 16.6491, 19.2035,
                21.1201, 22.8749, 200.0,
            ];
            let xmean = xv.iter().sum::<f64>() / n;
            let ymean = yv.iter().sum::<f64>() / n;
            let num: f64 = xv
                .iter()
                .zip(yv.iter())
                .map(|(xi, yi)| xi * yi)
                .sum::<f64>()
                - n * xmean * ymean;
            let den: f64 = xv.iter().map(|xi| xi * xi).sum::<f64>() - n * xmean * xmean;
            num / den
        };

        let huber_coef = fitted_huber.coefficients()[0];
        let clean_coef = fitted_clean.coefficients()[0];

        // The Huber coefficient should be closer to the clean slope than OLS is.
        let huber_err = (huber_coef - clean_coef).abs();
        let ols_err = (ols_coef - clean_coef).abs();
        assert!(
            huber_err < ols_err,
            "Huber error {huber_err:.4} should be less than OLS error {ols_err:.4} \
             (huber coef={huber_coef:.4}, ols coef={ols_coef:.4}, clean coef={clean_coef:.4})"
        );
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = HuberRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_length() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = HuberRegressor::<f64>::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        // 6 samples, 2 features (y = X·[1,2] + 0.2·noise, RandomState(5)),
        // non-degenerate so the fit converges; the assertion is the predict-side
        // shape guard, not a numeric one.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.4412, -0.3309, 2.4308, -0.2521, 0.1096, 1.5825, -0.9092, -0.5916, 0.1876,
                -0.3299, -1.1928, -0.2049,
            ],
        )
        .unwrap();
        let y = array![-0.2923, 2.0473, 2.9416, -2.2325, -0.2419, -1.2311];
        let fitted = HuberRegressor::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_coefficients_length() {
        // 8 samples, 3 features, y = X·[1,2,-1] + 0.3·noise (RandomState(3)),
        // so the system is overdetermined with a non-degenerate scale.
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.7886, 0.4365, 0.0965, -1.8635, -0.2774, -0.3548, -0.0827, -0.627, -0.0438,
                -0.4772, -1.3139, 0.8846, 0.8813, 1.7096, 0.05, -0.4047, -0.5454, -1.5465, 0.9824,
                -1.1011, -1.185, -0.2056, 1.4861, 0.2367,
            ],
        )
        .unwrap();
        let y = array![
            2.258, -2.2774, -1.1054, -4.0377, 4.0198, -0.0179, 0.1888, 3.1228
        ];
        let fitted = HuberRegressor::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 3);
    }

    #[test]
    fn test_scale_positive() {
        // scale_ is jointly estimated and strictly positive (sklearn scale_).
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
        let fitted = HuberRegressor::<f64>::new().fit(&x, &y).unwrap();
        assert!(fitted.scale() > 0.0, "scale must be strictly positive");
    }

    #[test]
    fn test_outliers_mask_length_and_band() {
        // outliers_[i] == |y_i - X_i·coef - intercept| > scale·epsilon.
        // 7 noisy inliers (y ≈ 2x + 1, RandomState(11)) + 1 clear outlier (the
        // last point is ~8 above the y ≈ 2x+1 trend).
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![
            3.3499, 4.9428, 6.9031, 8.4693, 10.9983, 12.9361, 14.8927, 25.0
        ];
        let m = HuberRegressor::<f64>::new();
        let fitted = m.fit(&x, &y).unwrap();
        let outliers = fitted.outliers();
        assert_eq!(outliers.len(), 8);

        // Recompute the band relationship the field is defined by.
        let preds = fitted.predict(&x).unwrap();
        let band = fitted.scale() * m.epsilon;
        for i in 0..8 {
            let resid = (y[i] - preds[i]).abs();
            assert_eq!(outliers[i], resid > band, "outliers mask mismatch at {i}");
        }
        // The off-trend last point must be flagged an outlier.
        assert!(outliers[7], "large outlier must be flagged");
    }

    #[test]
    fn test_large_epsilon_approaches_ols() {
        // With very large epsilon, all residuals fall in the quadratic zone
        // so Huber ≈ WLS with uniform weights ≈ OLS.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = HuberRegressor::<f64>::new()
            .with_epsilon(1000.0)
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.1);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 0.5);
    }

    #[test]
    fn test_l2_regularization_shrinks_coefficients() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let low = HuberRegressor::<f64>::new()
            .with_alpha(0.0001)
            .fit(&x, &y)
            .unwrap();
        let high = HuberRegressor::<f64>::new()
            .with_alpha(100.0)
            .fit(&x, &y)
            .unwrap();

        assert!(
            high.coefficients()[0].abs() <= low.coefficients()[0].abs(),
            "higher alpha should shrink more"
        );
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = HuberRegressor::<f64>::new();
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_multivariate() {
        // 8 samples, 2 features, y = X·[1,2] + 0.3·noise (RandomState(3)),
        // non-degenerate so the joint optimization converges.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                -1.2441, -0.6264, -0.8038, -2.4191, -0.9238, -1.0239, 1.124, -0.1319, -1.6233,
                0.6467, -0.3563, -1.7431, -0.5966, -0.5886, -0.8739, 0.0297,
            ],
        )
        .unwrap();
        let y = array![
            -3.1714, -5.7223, -2.6676, 1.116, 0.0025, -3.5067, -1.3276, -1.1499
        ];

        let fitted = HuberRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }
}
